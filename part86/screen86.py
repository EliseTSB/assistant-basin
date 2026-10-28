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
        "code_sha256": c.code_sha256(),
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
    (out_dir / "manifest.json").unlink(missing_ok=True)


def _runtime_canaries(model, tokenizer, layer, args, first_two_pushes, width: int,
                      layer_no: int, median: float, config: dict) -> dict:
    import torch
    import triage86
    canary_path = Path(args.out) / "runtime-canaries.json"
    batch_pushes = torch.zeros((16, width), dtype=torch.float32, device=first_two_pushes.device)
    batch_pushes[:2] = first_two_pushes[:2]
    batch, counts = c.generate_batch(
        model, tokenizer, layer, c.SCREEN_QUESTION, batch_pushes, args.device, MAX_NEW_TOKENS,
    )
    singles = []
    for row in range(2):
        result, _ = c.generate_batch(
            model, tokenizer, layer, c.SCREEN_QUESTION, first_two_pushes[row:row + 1],
            args.device, MAX_NEW_TOKENS,
        )
        singles.append(result[0]["answer"])
    same = [batch[i]["answer"] == singles[i] for i in range(2)]
    zero_answer = batch[2]["answer"]
    zero_triage = triage86.triage_answer(zero_answer)
    if zero_triage != "kept" or "qwen" not in zero_answer.casefold():
        raise RuntimeError(
            f"L{layer_no} zero-push canary must triage kept and name Qwen; got {zero_triage}: {zero_answer!r}"
        )
    result = {
        "layer": layer_no,
        "config_sha256": c.canonical_sha256(config),
        "zero_push": {"answer": zero_answer, "triage": zero_triage, "names_qwen": True},
        "batch16_vs_batch1": {
            "neuron_ids": [0, 1], "exact_answer_match": same,
            "batch16_answers": [batch[0]["answer"], batch[1]["answer"]],
            "batch1_answers": singles,
        },
        "batch16_hook_counts": counts,
    }
    c.atomic_json(canary_path, result)
    print(f"L{layer_no} runtime canary recorded; batch16/batch1 exact matches={same}; zero-push answer={zero_answer!r}", flush=True)
    return result


def _write_pilot_gate(out_dir: Path, layer_no: int, rows: list[dict], median: float) -> None:
    if layer_no != 22:
        return
    target = next((row for row in rows if row.get("kind") == "neuron" and int(row.get("id", -1)) == 2742), None)
    changed = sum(triage86.triage_answer(row.get("answer", "")) == "changed" for row in rows)
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
    for layer_no in sorted(reset_layers):
        _clear_layer(out_dir, batch_root, int(layer_no))

    for layer_no in args.layers:
        if not 0 <= int(layer_no) < len(layers):
            raise ValueError(f"layer {layer_no} outside model range 0..{len(layers)-1}")
        layer = layers[layer_no]
        down = layer.mlp.down_proj
        width = int(down.in_features)
        if c.MODEL_ID == "Qwen/Qwen3.5-4B" and width != N_NEURONS:
            raise ValueError(f"L{layer_no} down_proj width {width}, expected {N_NEURONS}")
        n_neurons = min(N_NEURONS, args.limit) if args.limit is not None else N_NEURONS
        n_directions = min(N_DIRECTIONS, args.limit) if args.limit is not None else N_DIRECTIONS
        final_path = out_dir / f"layer_{layer_no}.jsonl"
        meta_path = out_dir / f"layer_{layer_no}.meta.json"
        median = c.residual_median(model, tokenizer, layer, calibration_text, args.device)
        if layer_no == 22 and c.MODEL_ID == "Qwen/Qwen3.5-4B":
            expected_median = 24.8
            if abs(median - expected_median) > 0.05 * expected_median:
                raise RuntimeError(f"L22 residual median {median:.6f} differs from Part 84's 24.8 by more than 5%")
        config = _layer_config(
            layer_no, args, calibration_hash, actual_dtype, median, generation_config,
        )

        old_state = None
        if meta_path.exists():
            old_state = json.loads(meta_path.read_text(encoding="utf-8"))
            if old_state.get("config") != config:
                raise ValueError(f"L{layer_no} resume configuration differs from {meta_path}")
            if (old_state.get("median") != median or old_state.get("actual_dtype") != actual_dtype
                    or old_state.get("versions") != config["versions"]
                    or old_state.get("code_sha256") != config["code_sha256"]):
                raise ValueError(f"L{layer_no} resume provenance differs from {meta_path}")
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
            if old_state.get("median") != median:
                raise ValueError(f"L{layer_no} residual median changed while resuming")
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
                completed.add(batch_id)
                total_written += len(rows)
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
                do_sample=False,
            )
            generation_elapsed += time.monotonic() - started_generation
            if hook_counts["prefill_calls"] != 1 or hook_counts["generated_calls"] != hook_counts["per_row_generated_calls"][0]:
                raise RuntimeError(f"L{layer_no} {batch_id}: hook did not fire uniformly on every generated step")
            if len(hook_counts["per_row_generated_calls"]) != len(ids):
                raise RuntimeError(f"L{layer_no} {batch_id}: hook row count does not match pushed rows")
            records = []
            for row_no, (item_id, result) in enumerate(zip(ids, generated)):
                required = max(0, int(result["token_count"]) - 1)
                fired = hook_counts["per_row_generated_calls"][row_no]
                if fired < required:
                    raise RuntimeError(f"L{layer_no} {batch_id} row {row_no}: hook fired {fired}, need {required}")
                records.append({
                    "layer": int(layer_no),
                    "kind": kind,
                    "id": int(item_id),
                    "answer": result["answer"],
                    "token_count": int(result["token_count"]),
                    "median": median,
                    "dose": c.DOSE,
                    "batch_id": batch_id,
                    "hook_fire_counts": hook_counts,
                })
            _atomic_jsonl(fragment, records)
            completed.add(batch_id)
            total_written += len(records)
            session_new_answers += len(records)
            state["completed_batches"] = sorted(completed)
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
        state["completed_batches"] = [batch_id for _kind, batch_id, _ids in planned]
        state["status"] = "complete"
        state["answers"] = len(final_rows)
        state["median"] = median
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
