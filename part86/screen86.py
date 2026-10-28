#!/usr/bin/env python3
"""Screen Qwen3.5-4B neurons and residual-size-matched random directions."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common86 as c
import triage86

N_NEURONS = 9216
N_DIRECTIONS = 512
MAX_NEW_TOKENS = 48


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    try:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                lo, hi = (int(x) for x in item.split("-", 1))
                if hi < lo:
                    raise ValueError
                layers.update(range(lo, hi + 1))
            else:
                layers.add(int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid layer list: {spec!r}") from exc
    if not layers or any(x < 10 or x > 26 for x in layers):
        raise argparse.ArgumentTypeError("screen layers must be in the design range 10..26")
    return sorted(layers)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_fragment(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt completed batch {path}:{line_no}") from exc
    return rows


def _plan_batches(kind: str, total: int, batch_size: int):
    for batch_no, begin in enumerate(range(0, total, batch_size)):
        ids = list(range(begin, min(total, begin + batch_size)))
        yield f"{kind}_{batch_no:05d}", ids


def _layer_config(layer_no: int, args, calibration_hash: str, actual_dtype: str,
                  median: float, generation_config: dict) -> dict:
    return {
        "model_id": c.MODEL_ID,
        "revision": c.REVISION,
        "layer": int(layer_no),
        "device": args.device,
        "dtype_requested": "bfloat16",
        "actual_dtype": actual_dtype,
        "versions": c.dependency_versions(),
        "code_sha256": c.generation_code_sha256(),
        "dose": c.DOSE,
        "median": float(median),
        "mode": "generated_tokens",
        "question": c.SCREEN_QUESTION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "greedy": True,
        "generation_config": generation_config,
        "calibration_sha256": calibration_hash,
        "seed": c.SCREEN_SEED + int(layer_no),
        "neurons_requested": N_NEURONS if args.limit is None else min(N_NEURONS, args.limit),
        "directions_requested": N_DIRECTIONS if args.limit is None else min(N_DIRECTIONS, args.limit),
        "batch_size": int(args.batch_size),
    }


def _clear_layer(out_dir: Path, batch_root: Path, layer_no: int) -> None:
    for path in (out_dir / f"layer_{layer_no}.jsonl", out_dir / f"layer_{layer_no}.meta.json"):
        path.unlink(missing_ok=True)
    for path in batch_root.glob(f"layer_{layer_no}.*"):
        if path.is_file():
            path.unlink()
    if layer_no == 22:
        (out_dir / "runtime-canaries.json").unlink(missing_ok=True)
        (out_dir / "pilot-gate.json").unlink(missing_ok=True)
    (out_dir / "trace" / f"layer_{layer_no}.jsonl").unlink(missing_ok=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)


def _trace_rows(answer_rows: list[dict]) -> list[dict]:
    traces = []
    for row in answer_rows:
        trace = row.get("trace")
        if not isinstance(trace, dict):
            raise ValueError(f"screen answer {row.get('kind')}:{row.get('id')} has no measured trace")
        traces.append(trace)
    return traces


def _runtime_canaries(model, tokenizer, layer, args, first_two_pushes, width: int,
                      layer_no: int, median: float, config: dict) -> dict:
    import torch
    import triage86
    canary_path = Path(args.out) / "runtime-canaries.json"
    batch_pushes = torch.zeros((16, width), dtype=torch.float32, device=first_two_pushes.device)
    batch_pushes[2:4] = first_two_pushes[:2]
    generation_elapsed = 0.0
    generation_answers = 0

    def generate(push):
        nonlocal generation_elapsed, generation_answers
        started = time.monotonic()
        generated, hook_counts = c.generate_batch(
            model, tokenizer, layer, c.SCREEN_QUESTION, push, args.device, MAX_NEW_TOKENS,
        )
        generation_elapsed += time.monotonic() - started
        generation_answers += len(generated)
        return generated, hook_counts

    batch, counts = generate(batch_pushes)
    singleton_zero, _ = generate(batch_pushes[0:1])
    pushed_singles = []
    for row in range(2):
        result, _ = generate(first_two_pushes[row:row + 1])
        pushed_singles.append(result[0])
    zero_answer = batch[0]["answer"]
    zero_triage = triage86.triage_answer(zero_answer)
    if zero_triage != "kept" or "qwen" not in zero_answer.casefold():
        raise RuntimeError(
            f"L{layer_no} zero-push canary must triage kept and name Qwen; got {zero_triage}: {zero_answer!r}"
        )
    same_batch_baseline = batch[1]
    if batch[0]["token_ids"] != same_batch_baseline["token_ids"]:
        raise RuntimeError(
            f"L{layer_no} zero-push canary differs from the unsteered row with the same batch shape: "
            f"{zero_answer!r} != {same_batch_baseline['answer']!r}"
        )
    same = [batch[index + 2]["answer"] == pushed_singles[index]["answer"] for index in range(2)]
    raw_zero_trace = dict(counts["trace_rows"][0])
    raw_zero_trace.pop("direction_sha256", None)
    zero_trace = {
        "layer": int(layer_no), "batch": "canary", "row": 0,
        "kind": "zero", "target": None, **raw_zero_trace,
        "median": float(median), "dose": c.DOSE,
    }
    result = {
        "layer": layer_no,
        "config_sha256": c.canonical_sha256(config),
        "zero_push": {
            "answer": zero_answer, "token_ids": batch[0]["token_ids"],
            "triage": zero_triage, "names_qwen": True,
            "matches_unsteered_same_batch_shape": True,
            "same_batch_baseline_answer": same_batch_baseline["answer"],
            "same_batch_baseline_token_ids": same_batch_baseline["token_ids"],
        },
        "bf16_batch_effect": {
            "prompt": c.SCREEN_QUESTION,
            "batch16_answer": same_batch_baseline["answer"],
            "batch16_token_ids": same_batch_baseline["token_ids"],
            "batch1_answer": singleton_zero[0]["answer"],
            "batch1_token_ids": singleton_zero[0]["token_ids"],
            "exact_token_match": same_batch_baseline["token_ids"] == singleton_zero[0]["token_ids"],
        },
        "batch16_vs_batch1": {
            "neuron_ids": [0, 1], "exact_answer_match": same,
            "batch16_answers": [batch[2]["answer"], batch[3]["answer"]],
            "batch1_answers": [row["answer"] for row in pushed_singles],
        },
        "batch16_hook_counts": counts,
        "zero_trace": zero_trace,
        "generation_elapsed_seconds": generation_elapsed,
        "generation_answers": generation_answers,
    }
    c.atomic_json(canary_path, result)
    print(f"L{layer_no} runtime canary recorded; batch16/batch1 exact matches={same}; zero-push answer={zero_answer!r}", flush=True)
    return result


def _write_pilot_gate(out_dir: Path, layer_no: int, rows: list[dict], median: float) -> None:
    if layer_no != 22:
        return
    target = next((row for row in rows if row.get("kind") == "neuron" and int(row.get("id", -1)) == 2742), None)
    changed = sum(triage86.triage_answer(row.get("answer", "")) == "changed" for row in rows)
    batch_times = {}
    for row in rows:
        batch_id = row.get("batch_id")
        if batch_id is not None:
            batch_times.setdefault(batch_id, float(row.get("generation_elapsed_seconds", 0.0)))
    screen_elapsed = sum(batch_times.values())
    canary = {}
    canary_path = out_dir / "runtime-canaries.json"
    if canary_path.is_file():
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
    canary_elapsed = float(canary.get("generation_elapsed_seconds", 0.0))
    generation_answers = len(rows) + int(canary.get("generation_answers", 0))
    generation_elapsed = screen_elapsed + canary_elapsed
    measured_rate = generation_answers / generation_elapsed if generation_elapsed > 0 else None
    full_population = len(rows) == N_NEURONS + N_DIRECTIONS
    projected_seconds = (N_NEURONS * 17 + N_DIRECTIONS * 17) / measured_rate if measured_rate else None
    gate_2_status = (
        ("NOT READ" if projected_seconds is None else
         "PASS" if projected_seconds <= 12 * 60 * 60 else "FAIL") if full_population else
        "EARLY ESTIMATE ONLY (limited pilot; not authoritative)"
    )
    result = {
        "layer": 22,
        "population_size": N_NEURONS + N_DIRECTIONS,
        "observed_rows": len(rows),
        "median": float(median),
        "changed_answers": changed,
        "gate_3_range": [20, 2000],
        "gate_3_population": "9,216 real neurons plus 512 random directions (9,728 answers)",
        "gate_3_status": ("NOT READ" if len(rows) != N_NEURONS + N_DIRECTIONS
                          else "PASS" if 20 <= changed <= 2000 else "FAIL"),
        "gate_2_status": gate_2_status,
        "gate_2_basis": "full layer-22 screen plus runtime canary" if full_population else "limited warmup only",
        "screen_generation_elapsed_seconds": screen_elapsed,
        "canary_generation_elapsed_seconds": canary_elapsed,
        "generation_elapsed_seconds": generation_elapsed,
        "screen_generation_answers": len(rows),
        "canary_generation_answers": int(canary.get("generation_answers", 0)),
        "generation_answers": generation_answers,
        "measured_answers_per_second": measured_rate,
        "projected_full_screen_seconds": projected_seconds,
        "projected_full_screen_answers": N_NEURONS * 17 + N_DIRECTIONS * 17,
        "gate_2_rate_threshold_answers_per_second": (N_NEURONS * 17 + N_DIRECTIONS * 17) / (12 * 60 * 60),
        "n2742_raw_answer": target.get("answer") if target else None,
        "n2742_status": "present" if target else "NOT INCLUDED (screen was limited)",
    }
    c.atomic_json(out_dir / "pilot-gate.json", result)


def run(args) -> None:
    import torch

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_root = out_dir / ".batches"
    batch_root.mkdir(exist_ok=True)
    calibration_text = Path(args.calib).read_text(encoding="utf-8")
    calibration_hash = c.sha256_text(calibration_text)
    tokenizer, model = c.load_model(args.device)
    model.requires_grad_(False)
    model.eval()
    actual_dtype = c.require_bfloat16(model)
    for question in (c.SCREEN_QUESTION,):
        c.assert_rendered_prompt(tokenizer, question)
    generation_config = c.effective_generation_config(model, tokenizer, MAX_NEW_TOKENS, False)
    layers = c.find_layers(model)
    session_new_answers = 0
    generation_elapsed = 0.0
    progress_path = Path(args.progress) if args.progress else out_dir / "progress.log"

    reset_layers = set(getattr(args, "reset_layer", []) or [])
    if not reset_layers.issubset(set(args.layers)):
        raise ValueError("--reset-layer values must also appear in --layers")
    prepared = {}
    for layer_no in args.layers:
        if not 0 <= int(layer_no) < len(layers):
            raise ValueError(f"layer {layer_no} outside model range 0..{len(layers)-1}")
        layer = layers[layer_no]
        down = layer.mlp.down_proj
        width = int(down.in_features)
        if c.MODEL_ID == "Qwen/Qwen3.5-4B" and width != N_NEURONS:
            raise ValueError(f"L{layer_no} down_proj width {width}, expected {N_NEURONS}")
        median = c.residual_median(model, tokenizer, layer, calibration_text, args.device)
        if layer_no == 22 and c.MODEL_ID == "Qwen/Qwen3.5-4B":
            expected_median = 24.8
            if abs(median - expected_median) > 0.05 * expected_median:
                raise RuntimeError(f"L22 residual median {median:.6f} differs from Part 84's 24.8 by more than 5%")
        config = _layer_config(
            layer_no, args, calibration_hash, actual_dtype, median, generation_config,
        )
        meta_path = out_dir / f"layer_{layer_no}.meta.json"
        if meta_path.exists():
            old_state = json.loads(meta_path.read_text(encoding="utf-8"))
            old_config = old_state.get("config")
            if not isinstance(old_config, dict):
                raise ValueError(f"L{layer_no} resume metadata has no configuration")
            old_comparable = dict(old_config)
            new_comparable = dict(config)
            old_comparable.pop("median", None)
            new_comparable.pop("median", None)
            if old_comparable != new_comparable:
                action = "reset" if layer_no in reset_layers else "resume"
                raise ValueError(f"L{layer_no} {action} configuration differs from {meta_path}")
            if (old_state.get("median") != median or old_config.get("median") != median):
                raise ValueError(f"L{layer_no} residual median differs from {meta_path}")
        prepared[layer_no] = (layer, down, width, median, config)

    # Validate the complete requested configuration before any reset can remove data.
    for layer_no in args.layers:
        trace_path = out_dir / "trace" / f"layer_{layer_no}.jsonl"
        answer_path = out_dir / f"layer_{layer_no}.jsonl"
        if layer_no not in reset_layers and trace_path.exists() and not answer_path.exists():
            raise ValueError(f"L{layer_no} has an orphan trace without answers; use --reset-layer {layer_no}")
    for layer_no in sorted(reset_layers):
        _clear_layer(out_dir, batch_root, int(layer_no))

    for layer_no in args.layers:
        layer, down, width, median, config = prepared[layer_no]
        n_neurons = min(N_NEURONS, args.limit) if args.limit is not None else N_NEURONS
        n_directions = min(N_DIRECTIONS, args.limit) if args.limit is not None else N_DIRECTIONS
        final_path = out_dir / f"layer_{layer_no}.jsonl"
        meta_path = out_dir / f"layer_{layer_no}.meta.json"

        old_state = None
        if meta_path.exists():
            old_state = json.loads(meta_path.read_text(encoding="utf-8"))
            if old_state.get("status") == "complete" and final_path.exists():
                expected_answers = n_neurons + n_directions
                if int(old_state.get("answers", -1)) != expected_answers:
                    raise ValueError(f"L{layer_no} completed metadata has an unexpected answer count")
                with final_path.open(encoding="utf-8") as f:
                    actual_answers = sum(1 for line in f if line.strip())
                if actual_answers != expected_answers:
                    raise ValueError(f"L{layer_no} completed JSONL has {actual_answers} rows, expected {expected_answers}")
                with final_path.open(encoding="utf-8") as f:
                    completed_rows = [json.loads(line) for line in f if line.strip()]
                _atomic_jsonl(out_dir / "trace" / f"layer_{layer_no}.jsonl", _trace_rows(completed_rows) + (
                    [json.loads((out_dir / "runtime-canaries.json").read_text(encoding="utf-8"))["zero_trace"]]
                    if layer_no == 22 and (out_dir / "runtime-canaries.json").is_file() else []
                ))
                if layer_no == 22:
                    canary_path = out_dir / "runtime-canaries.json"
                    expected_canary_hash = c.canonical_sha256(config)
                    if not canary_path.is_file():
                        raise ValueError("completed L22 is missing runtime-canaries.json; reset layer 22 before reuse")
                    canary = json.loads(canary_path.read_text(encoding="utf-8"))
                    if canary.get("config_sha256") != expected_canary_hash:
                        raise ValueError("completed L22 runtime canary provenance differs; reset layer 22 before reuse")
                    _write_pilot_gate(out_dir, layer_no, completed_rows, median)
                    c.write_manifest(out_dir)
                print(f"L{layer_no}: already complete; skipping", flush=True)
                continue

        weight = down.weight.detach()
        if weight.shape[1] != width:
            raise ValueError(f"L{layer_no} down_proj weight shape inconsistent with in_features")
        deltas = c.neuron_deltas(weight, median, count=n_neurons).detach()
        directions = c.scaled_random_directions(
            weight, n_directions, c.SCREEN_SEED + layer_no, c.DOSE * median
        )
        state = {
            "config": config,
            "median": median,
            "calibration_sha256": calibration_hash,
            "actual_dtype": actual_dtype,
            "versions": config["versions"],
            "code_sha256": config["code_sha256"],
            "completed_batches": [],
            "status": "running",
        }

        if old_state is not None:
            state.update(old_state)
        elif final_path.exists():
            raise ValueError(f"{final_path} exists without matching resume metadata")
        else:
            stale = list(batch_root.glob(f"layer_{layer_no}.*"))
            if stale:
                raise ValueError(
                    f"L{layer_no} has orphaned batch fragments without metadata; use --reset-layer {layer_no}"
                )
            c.atomic_json(meta_path, state)

        if final_path.exists() and state.get("status") == "complete":
            print(f"L{layer_no}: already complete; skipping", flush=True)
            continue

        if layer_no == 22:
            canary_path = out_dir / "runtime-canaries.json"
            if canary_path.exists():
                old_canary = json.loads(canary_path.read_text(encoding="utf-8"))
                expected_canary_hash = c.canonical_sha256(config)
                if old_canary.get("config_sha256") != expected_canary_hash:
                    raise ValueError("L22 runtime canary provenance differs; reset layer 22 before reuse")
            else:
                canary_pushes = torch.zeros((2, width), dtype=torch.float32, device=weight.device)
                canary_pushes[0, 0] = deltas[0]
                canary_pushes[1, 1] = deltas[1]
                _runtime_canaries(model, tokenizer, layer, args, canary_pushes, width, layer_no, median, config)

        planned = []
        planned.extend(("neuron", batch_id, ids) for batch_id, ids in _plan_batches("neuron", n_neurons, args.batch_size))
        planned.extend(("random_direction", batch_id, ids) for batch_id, ids in _plan_batches("random_direction", n_directions, args.batch_size))
        completed = set()
        total_written = 0
        for kind, batch_id, ids in planned:
            fragment = batch_root / f"layer_{layer_no}.{batch_id}.jsonl"
            if fragment.exists():
                rows = _read_fragment(fragment)
                if len(rows) != len(ids) or any(row.get("batch_id") != batch_id for row in rows):
                    raise ValueError(f"incomplete or mismatched saved batch {fragment}")
                _trace_rows(rows)
                completed.add(batch_id)
                total_written += len(rows)
                generation_elapsed += float(rows[0].get("generation_elapsed_seconds", 0.0))
                session_new_answers += len(rows)
                continue
            if kind == "neuron":
                pushes = torch.zeros((len(ids), width), dtype=torch.float32, device=weight.device)
                row_index = torch.arange(len(ids), device=weight.device)
                unit_index = torch.tensor(ids, dtype=torch.long, device=weight.device)
                pushes[row_index, unit_index] = deltas[unit_index]
            else:
                pushes = directions[ids].to(device=weight.device, dtype=torch.float32)
            started_generation = time.monotonic()
            generated, hook_counts = c.generate_batch(
                model, tokenizer, layer, c.SCREEN_QUESTION, pushes, args.device, MAX_NEW_TOKENS,
                do_sample=False, trace_direction_hash=(kind == "random_direction"),
            )
            batch_elapsed = time.monotonic() - started_generation
            generation_elapsed += batch_elapsed
            if hook_counts["prefill_calls"] != 1 or hook_counts["generated_calls"] != hook_counts["per_row_generated_calls"][0]:
                raise RuntimeError(f"L{layer_no} {batch_id}: hook did not fire uniformly on every generated step")
            if len(hook_counts["per_row_generated_calls"]) != len(ids):
                raise RuntimeError(f"L{layer_no} {batch_id}: hook row count does not match pushed rows")
            records = []
            measured_rows = hook_counts.pop("trace_rows")
            for row_no, (item_id, result) in enumerate(zip(ids, generated)):
                required = max(0, int(result["token_count"]) - 1)
                fired = hook_counts["per_row_generated_calls"][row_no]
                if fired < required:
                    raise RuntimeError(f"L{layer_no} {batch_id} row {row_no}: hook fired {fired}, need {required}")
                trace = dict(measured_rows[row_no])
                if kind != "random_direction":
                    trace.pop("direction_sha256", None)
                trace = {
                    "layer": int(layer_no), "batch": batch_id, "row": row_no,
                    "kind": kind, "target": int(item_id), **trace,
                    "median": float(median), "dose": c.DOSE,
                }
                records.append({
                    "layer": int(layer_no),
                    "kind": kind,
                    "id": int(item_id),
                    "answer": result["answer"],
                    "token_count": int(result["token_count"]),
                    "median": median,
                    "dose": c.DOSE,
                    "batch_id": batch_id,
                    "generation_elapsed_seconds": batch_elapsed,
                    "trace": trace,
                    "hook_fire_counts": hook_counts,
                })
            _atomic_jsonl(fragment, records)
            completed.add(batch_id)
            total_written += len(records)
            session_new_answers += len(records)
            state["completed_batches"] = sorted(completed)
            state["generation_elapsed_seconds"] = generation_elapsed
            state["generation_answers"] = total_written
            c.atomic_json(meta_path, state)
            elapsed = max(generation_elapsed, 1e-9)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"layer={layer_no} batch={batch_id} answers={total_written} "
                    f"answers_per_second={session_new_answers / elapsed:.4f} elapsed_seconds={elapsed:.1f}\n"
                )
                f.flush()
            print(f"L{layer_no} {batch_id}: {len(records)} answers, {session_new_answers / elapsed:.2f} answers/s", flush=True)

        final_rows = []
        for _kind, batch_id, _ids in planned:
            fragment = batch_root / f"layer_{layer_no}.{batch_id}.jsonl"
            rows = _read_fragment(fragment)
            if len(rows) != len(_ids):
                raise ValueError(f"batch {batch_id} has {len(rows)} rows, expected {len(_ids)}")
            final_rows.extend(rows)
        if len(final_rows) != n_neurons + n_directions:
            raise RuntimeError(f"L{layer_no}: assembled {len(final_rows)} answers, expected {n_neurons+n_directions}")
        _atomic_jsonl(final_path, final_rows)
        trace_rows = _trace_rows(final_rows)
        if layer_no == 22:
            canary_path = out_dir / "runtime-canaries.json"
            canary = json.loads(canary_path.read_text(encoding="utf-8"))
            trace_rows.append(canary["zero_trace"])
        _atomic_jsonl(out_dir / "trace" / f"layer_{layer_no}.jsonl", trace_rows)
        state["completed_batches"] = [batch_id for _kind, batch_id, _ids in planned]
        state["status"] = "complete"
        state["answers"] = len(final_rows)
        state["median"] = median
        state["generation_elapsed_seconds"] = generation_elapsed
        state["generation_answers"] = len(final_rows)
        c.atomic_json(meta_path, state)
        _write_pilot_gate(out_dir, layer_no, final_rows, median)
        c.write_manifest(out_dir)
        print(f"L{layer_no}: median={median:.6f}, wrote {len(final_rows)} rows to {final_path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", required=True, help="calibration text path; on WSL use /mnt/c/Users/alexa/neuron-probe-gpu/p66basin/neuronary-calib.txt")
    ap.add_argument("--out", required=True, help="screen output directory")
    ap.add_argument("--layers", type=parse_layers, default=parse_layers("10-26"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="screen the first N neurons and first N random directions per layer")
    ap.add_argument("--reset-layer", type=int, action="append", default=[],
                    help="clear a layer's metadata, final output, and .batches fragments before rerunning it")
    ap.add_argument("--device", choices=("xpu", "cpu", "cuda"), default="xpu")
    ap.add_argument("--progress", default=None, help="append progress lines here; default is OUT/progress.log")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
