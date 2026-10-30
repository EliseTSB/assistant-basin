#!/usr/bin/env python3
"""Resumable Part 119b dose runner; shares Part 119 hooks and model helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119", ROOT / "part92", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common92 as c  # noqa: E402
import rows119  # noqa: E402
import run119  # noqa: E402
import rows119b as rb  # noqa: E402
from rows119b import _push  # noqa: E402 -- Path 2 builds a cached first-push row
import gate119b  # noqa: E402 -- lightweight CPU-only pilot gate helpers

QUESTION = run119.QUESTION
FRAME = run119.FRAME
ANSWER_TOKENS = run119.ANSWER_TOKENS
PATH_TOKENS = run119.PATH_TOKENS
PART119_ANSWERS = "results/part119/answers.jsonl"
PART119_INTERACTIONS = gate119b.PART119_INTERACTIONS_PATH


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl_readonly(path: Path, key: str | None = None) -> list[dict] | dict[str, dict]:
    """Strict read-only JSONL reader; never repairs/truncates Part 119 inputs."""
    if not path.exists():
        return {} if key else []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL in {path}:{line_no}")
            rows.append(row)
    if key is None:
        return rows
    result = {}
    for row in rows:
        row_key = str(row[key])
        if row_key in result:
            raise ValueError(f"duplicate key {row_key!r} in {path}")
        result[row_key] = row
    return result


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _append_jsonl(path: Path, rows: list[dict], *, round_floats: bool = True) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            clean = _round_floats(row) if round_floats else row
            stream.write(json.dumps(clean, ensure_ascii=False,
                                   allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def read_output_by_key(path: Path, key: str) -> dict[str, dict]:
    # Part 119's output reader safely drops only an unterminated malformed final line.
    return run119.read_existing(path, key)


def build_dose_push_cache(units: list[dict], layers, medians: dict[int, float],
                          dose: float) -> dict:
    """Part 92's own pushes at one dose. Each dose is built from the weights, not scaled in
    bf16: a scaled bf16 vector rounds differently from the direct one by up to one step."""
    cache = {"_units": units}
    for unit in units:
        layer_no = int(unit["layer"])
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"unit {unit['id']} layer {layer_no} is outside loaded model")
        weight = layers[layer_no].mlp.down_proj.weight
        if not 0 <= int(unit["neuron"]) < weight.shape[1]:
            raise ValueError(f"unit {unit['id']} neuron index is outside W_down")
        target_norm = float(dose) * float(medians[layer_no])
        cache[(unit["id"], "neuron")] = c.neuron_push(weight, unit["neuron"], target_norm)
        cache[(unit["id"], "random")] = c.random_direction(
            weight, unit["control_seed"], target_norm)
    return cache


def build_base_push_cache(units: list[dict], layers, medians: dict[int, float]) -> dict:
    """Part 119's pushes at 0.8."""
    return build_dose_push_cache(units, layers, medians, 0.8)


def scale_push_cache(base_cache: dict, dose: float) -> dict:
    """Scale both arms by the row's alpha. Fixtures only: a run uses build_dose_push_cache."""
    scale = float(dose) / 0.8
    cache = {"_units": base_cache["_units"]}
    for unit in base_cache["_units"]:
        for arm in ("neuron", "random"):
            cache[(unit["id"], arm)] = base_cache[(unit["id"], arm)] * scale
    return cache


def validate_dose_scaling(units: list[dict], base_cache: dict, dose_cache: dict,
                          dose: float) -> None:
    """Runtime guard: every push at a dose is the 0.8 push's direction at dose/0.8 of its size.
    bf16 rounding allows about 1% in size; a wrong alpha or a swapped vector fails."""
    import torch

    ratio = float(dose) / 0.8
    for unit in units:
        for arm in ("neuron", "random"):
            base = base_cache[(unit["id"], arm)].float()
            pushed = dose_cache[(unit["id"], arm)].float()
            size = float(torch.linalg.vector_norm(pushed) / torch.linalg.vector_norm(base))
            cosine = float(torch.dot(pushed, base)
                           / (torch.linalg.vector_norm(pushed) * torch.linalg.vector_norm(base)))
            if abs(size / ratio - 1) > .01 or cosine < .9999:
                raise AssertionError(f"push for {unit['id']} {arm} at dose {dose}: "
                                     f"size ratio {size:.5f} against {ratio:.5f}, cosine {cosine:.6f}")


def _row_layer_signature(row: dict, units_by_id: dict[str, dict]) -> tuple[int, ...]:
    return tuple(sorted({int(units_by_id[p["unit_id"]]["layer"]) for p in row["pushes"]}))


def _generate_rows(rows: list[dict], output: Path, tokenizer, model, layers,
                   cache_by_dose: dict[float, dict], units_by_id: dict[str, dict],
                   batch_size: int, device: str, quiet: bool = False) -> tuple[int, float, list[dict]]:
    existing = read_output_by_key(output, "row_id")
    pending = [row for row in rows if not row["reused"] and row["row_id"] not in existing]
    grouped = defaultdict(list)
    for row in pending:
        grouped[(row["dose"], _row_layer_signature(row, units_by_id))].append(row)
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    started = time.perf_counter()
    done = 0
    records_written = []
    for (dose, signature), tasks in grouped.items():
        cache = cache_by_dose[float(dose)]
        for begin in range(0, len(tasks), batch_size):
            chunk = tasks[begin:begin + batch_size]
            token_rows = run119._run_generate_batch(
                model, tokenizer, chunk, layers, cache, prompt, device)
            records = [{"row_id": row["row_id"], "row_type": row["row_type"],
                        "dose": row["dose"], "tokens": tokens,
                        "text": run119.decode_tokens(tokenizer, tokens)}
                       for row, tokens in zip(chunk, token_rows)]
            _append_jsonl(output, records)
            records_written.extend(records)
            done += len(chunk)
            if not quiet:
                print(f"answers dose={rb.dose_text(dose)} layers={signature} "
                      f"{done}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return len(pending), time.perf_counter() - started, records_written


def _run_gate1(output: Path, tokenizer, model, layers, device: str) -> tuple[dict, float]:
    import torch

    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    no_start = time.perf_counter()
    with torch.inference_mode():
        no_hook = run119._run_generate_batch(model, tokenizer, [{"pushes": []}], layers,
                                             {"_units": []}, prompt, device)[0]
    no_seconds = time.perf_counter() - no_start
    hidden = int(layers[0].mlp.down_proj.weight.shape[0])
    zero_vectors = {
        layer_no: torch.zeros((1, hidden), dtype=layer.mlp.down_proj.weight.dtype,
                              device=device)
        for layer_no, layer in enumerate(layers)
    }
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "do_sample": False, "max_new_tokens": ANSWER_TOKENS,
              "pad_token_id": int(tokenizer.pad_token_id)}
    end_ids = sorted({int(token) for token in
                      (getattr(tokenizer, "eos_token_id", None),
                       getattr(tokenizer, "unk_token_id", None)) if token is not None})
    if end_ids:
        kwargs["eos_token_id"] = end_ids
    zero_start = time.perf_counter()
    with torch.inference_mode(), run119.row_hooks(layers, zero_vectors):
        generated = model.generate(**kwargs)
    zero = run119.generated_tokens(generated[0, ids.shape[1]:], tokenizer)
    zero_seconds = time.perf_counter() - zero_start
    gate = {"no_hook_tokens": no_hook, "zero_hook_tokens": zero,
            "passed": no_hook == zero}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gate, no_seconds + zero_seconds


def _top_data(logits):
    tops, margins = run119._top_and_margin(logits)
    return tops.detach().cpu().tolist(), margins.detach().cpu().tolist()


def _position_record(t_ids: list[int], base, first, second, both,
                     top_none: int, top_first: int, top_second: int, top_both: int,
                     top_additive: int, margin_first: float, margin_second: float,
                     keep_t_ids: bool) -> dict:
    """Compute all Path 2 metrics on the exact fixed token set T."""
    import numpy as np
    import torch

    index = torch.tensor(t_ids, dtype=torch.long, device=base.device)
    raw = [item.index_select(0, index).detach().cpu().numpy().astype(np.float64, copy=False)
           for item in (base, first, second, both)]
    centered = [value - value.mean() for value in raw]
    d_a = centered[1] - centered[0]
    d_2 = centered[2] - centered[0]
    d_pair = centered[3] - centered[0]
    interaction = d_pair - d_a - d_2
    b_a, b_2 = no_intercept_slopes(d_a, d_2, d_pair)

    additive = raw[1] + raw[2] - raw[0]
    def softmax(values):
        shifted = values - float(np.max(values))
        exp = np.exp(shifted)
        return exp / exp.sum()
    tv = 0.5 * float(np.abs(softmax(additive) - softmax(raw[3])).sum())
    result = {"nI": float(np.linalg.norm(interaction)),
              "nA": float(np.linalg.norm(d_a)),
              "n2": float(np.linalg.norm(d_2)),
              "nPair": float(np.linalg.norm(d_pair)), "tv": tv,
              "top_none": int(top_none), "top_first": int(top_first),
              "top_second": int(top_second), "top_both": int(top_both),
              "top_additive": int(top_additive),
              "margin_first": float(margin_first),
              "margin_second": float(margin_second),
              "b_A": b_a, "b_2": b_2, "t_size": len(t_ids),
              "t_hash": hashlib.sha256(",".join(map(str, t_ids)).encode("ascii")).hexdigest()[:16],
              "resolved": (float(np.linalg.norm(d_a)) >= 1.0
                           and float(np.linalg.norm(d_2)) >= 1.0
                           and float(np.linalg.norm(d_pair)) >= 1.0)}
    if keep_t_ids:
        result["t_ids"] = [int(token) for token in t_ids]
    return result


def no_intercept_slopes(d_a, d_2, d_pair) -> tuple[float | None, float | None]:
    """Fit D_pair = b_A D_A + b_2 D_2 on T with no constant term."""
    import numpy as np

    matrix = np.column_stack((np.asarray(d_a), np.asarray(d_2)))
    coeff, _residuals, rank, _singular = np.linalg.lstsq(matrix, np.asarray(d_pair), rcond=None)
    if int(rank) < 2:
        return None, None
    return float(coeff[0]), float(coeff[1])


def _extract_batch_metrics(first_id: str, rows: list[dict], dose: float,
                           base_logits, first_logits, second_logits, both_logits,
                           cached_top0, cached_margin0, cached_top1, cached_margin1,
                           cached_top10_base, cached_top10_first,
                           supports: dict[tuple[str, int], list[int]], existing_t: dict,
                           add_support: bool) -> list[dict]:
    """Convert one batched second/both pair of forwards into resumable row records."""
    import torch

    top2, margin2 = _top_data(second_logits)
    top_pair, _margin_pair = _top_data(both_logits)
    top_add = (first_logits + second_logits - base_logits).argmax(dim=-1).detach().cpu().tolist()
    records = []
    for row_no, row in enumerate(rows):
        key = row["key"]
        per_position = []
        for offset in range(int(base_logits.shape[1])):
            absolute = int(row["position_start"]) + offset
            if add_support:
                groups = [cached_top10_base[offset], cached_top10_first[offset],
                          run119._top10_repeated_argmax(second_logits[row_no, offset:offset + 1])[0],
                          run119._top10_repeated_argmax(both_logits[row_no, offset:offset + 1])[0]]
                t_ids = sorted({int(token) for group in groups for token in group})
                support_key = (key, absolute)
                stored = existing_t.get(support_key)
                if stored is not None and list(stored) != t_ids:
                    raise ValueError(
                        f"in-memory and stored token support disagree for {key} position {absolute}")
                supports[support_key] = t_ids
            else:
                support_key = (key, absolute)
                in_memory = supports.get(support_key)
                stored = existing_t.get(support_key)
                if in_memory is not None and stored is not None and list(in_memory) != list(stored):
                    raise ValueError(
                        f"in-memory and stored token support disagree for {key} position {absolute}")
                t_ids = in_memory if in_memory is not None else stored
                if not t_ids:
                    raise ValueError(f"missing saved 0.8 token support for {key} position {absolute}")
            position = _position_record(
                t_ids, base_logits[0, offset], first_logits[0, offset],
                second_logits[row_no, offset], both_logits[row_no, offset],
                cached_top0[offset], cached_top1[offset], top2[row_no][offset],
                top_pair[row_no][offset], top_add[row_no][offset],
                cached_margin1[offset], margin2[row_no][offset], keep_t_ids=add_support)
            position["pos"] = absolute
            per_position.append(position)
        records.append({"key": rb.path2_key(row["kind"], dose, row["first"], row["second"]),
                        "first": row["first"], "second": row["second"],
                        "kind": row["kind"], "dose": float(dose),
                        "p_A": int(row["p_A"]), "positions": per_position})
    return records


def run_path2(units: list[dict], rows: list[dict], doses: tuple[float, ...],
              output: Path, tokenizer, model, layers, base_cache: dict,
              cache_by_dose: dict[float, dict], answers: dict[str, dict],
              device: str, batch_size: int, quiet: bool = False) -> dict:
    """Run fixed paths, caching no-push/A per unit and batching second/both passes."""
    import torch

    existing = read_output_by_key(output, "key")
    existing_t = {}
    for record in existing.values():
        if math.isclose(float(record.get("dose", -1)), 0.8):
            for position in record.get("positions", []):
                if position.get("t_ids"):
                    existing_t[(f"{record['kind']}:{record['first']}+{record['second']}",
                                int(position["pos"]))] = position["t_ids"]
    by_first = defaultdict(list)
    for row in rows:
        by_first[row["first"]].append(row)
    unit_by_id = {unit["id"]: unit for unit in units}
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    started = time.perf_counter()
    pass_count = 0
    rows_written = 0
    for first_id, candidates in by_first.items():
        pending_by_dose = {
            float(dose): [row for row in candidates
                          if rb.path2_key(row["kind"], dose, first_id, row["second"]) not in existing]
            for dose in doses
        }
        if not any(pending_by_dose.values()):
            continue
        single = answers.get(f"single@0.8:{first_id}")
        plain = answers.get("plain")
        if single is None or plain is None:
            raise ValueError(f"Path 2 needs Part 119's plain and 0.8 single answer for {first_id}")
        tokens = [int(token) for token in single["tokens"][:PATH_TOKENS]]
        p, _target_a, _target_0 = run119.first_flip(tokens, plain["tokens"])
        if p is None or p >= len(tokens):
            for dose in doses:
                records = []
                for row in pending_by_dose[float(dose)]:
                    key = rb.path2_key(row["kind"], dose, row["first"], row["second"])
                    records.append({"key": key, "first": row["first"],
                                    "second": row["second"], "kind": row["kind"],
                                    "dose": dose, "p_A": p, "positions": []})
                _append_jsonl(output, records)
                existing.update({record["key"]: record for record in records})
                rows_written += len(records)
            continue

        input_ids = torch.tensor([prompt + tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        position_start = len(prompt) + p - 1
        position_stop = len(prompt) + min(PATH_TOKENS, len(tokens)) - 1
        if position_stop <= position_start:
            for dose in doses:
                records = [{"key": rb.path2_key(row["kind"], dose, first_id, row["second"]),
                            "first": row["first"], "second": row["second"],
                            "kind": row["kind"], "dose": dose, "p_A": p,
                            "positions": []}
                           for row in pending_by_dose[float(dose)]]
                _append_jsonl(output, records)
                existing.update({record["key"]: record for record in records})
                rows_written += len(records)
            continue

        base_logits = run119._forward(model, input_ids, attention_mask, layers, {})[
            0, position_start:position_stop, :].unsqueeze(0)
        pass_count += 1
        first_logits = {}
        for dose in doses:
            if not pending_by_dose[float(dose)]:
                continue
            unit_row = {"pushes": [_push(first_id, "neuron")]}
            first_logits[dose] = run119._forward(
                model, input_ids, attention_mask, layers,
                run119.vectors_for_rows([unit_row], cache_by_dose[dose], layers, device),
            )[0, position_start:position_stop, :].unsqueeze(0)
            pass_count += 1
        cached_top0, cached_margin0 = _top_data(base_logits)
        cached_top0, cached_margin0 = cached_top0[0], cached_margin0[0]
        cached_by_dose = {}
        for dose in first_logits:
            tops, margins = _top_data(first_logits[dose])
            cached_by_dose[dose] = {"top": tops[0], "margin": margins[0]}
        needs_08_support = bool(pending_by_dose.get(0.8))
        cached_top10_base = (run119._top10_repeated_argmax(base_logits[0])
                             if needs_08_support else None)
        cached_top10_first = (run119._top10_repeated_argmax(first_logits[.8][0])
                              if needs_08_support else None)

        supports: dict[tuple[str, int], list[int]] = {}
        dose_order = sorted(doses, key=lambda item: (item != 0.8, item))
        for dose in dose_order:
            pending = pending_by_dose[float(dose)]
            if not pending:
                continue
            for begin in range(0, len(pending), batch_size):
                chunk = pending[begin:begin + batch_size]
                second_rows = [{"pushes": [row["pushes"][1]]} for row in chunk]
                both_rows = [{"pushes": row["pushes"]} for row in chunk]
                second_logits = run119._forward(
                    model, input_ids.expand(len(chunk), -1),
                    attention_mask.expand(len(chunk), -1), layers,
                    run119.vectors_for_rows(second_rows, cache_by_dose[dose], layers, device),
                )[:, position_start:position_stop, :]
                both_logits = run119._forward(
                    model, input_ids.expand(len(chunk), -1),
                    attention_mask.expand(len(chunk), -1), layers,
                    run119.vectors_for_rows(both_rows, cache_by_dose[dose], layers, device),
                )[:, position_start:position_stop, :]
                pass_count += 2 * len(chunk)
                records = _extract_batch_metrics(
                    first_id, [{**row, "p_A": p, "position_start": p} for row in chunk], dose,
                    base_logits, first_logits[dose], second_logits, both_logits,
                    cached_top0, cached_margin0,
                    cached_by_dose[dose]["top"], cached_by_dose[dose]["margin"],
                    cached_top10_base, cached_top10_first,
                    supports, existing_t, add_support=(dose == 0.8))
                _append_jsonl(output, records)
                for record in records:
                    existing[record["key"]] = record
                    if record["dose"] == 0.8:
                        for position in record["positions"]:
                            existing_t[(f"{record['kind']}:{record['first']}+{record['second']}",
                                        int(position["pos"]))] = position["t_ids"]
                rows_written += len(records)
                del second_logits, both_logits
        if not quiet:
            print(f"path2 first={first_id} rows_written={rows_written} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return {"rows_new": rows_written, "logical_passes": pass_count,
            "seconds": time.perf_counter() - started, "batch": batch_size}


def _load_answers_for_path2(part119_answers: dict[str, dict], local_answers: dict[str, dict],
                            answer_rows: list[dict]) -> dict[str, dict]:
    result = {"plain": part119_answers.get("plain", {})}
    for row in answer_rows:
        if row["row_type"] in ("single", "random") and math.isclose(row["dose"], 0.8):
            source = part119_answers.get(row["source_row_id"])
            if source is not None:
                result[row["row_id"]] = source
    result.update(local_answers)
    return result


def _run_path1(rows: list[dict], answer_by_id: dict[str, dict], output: Path,
               tokenizer, model, layers, cache: dict, device: str,
               batch_size: int) -> dict:
    existing = read_output_by_key(output, "row_id")
    pending = [row for row in rows if row["row_id"] not in existing]
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    started = time.perf_counter()
    done = 0
    passes = 0
    for begin in range(0, len(pending), batch_size):
        chunk = pending[begin:begin + batch_size]
        records = run119.score_logits_chunk(chunk, answer_by_id, tokenizer, model,
                                            layers, cache, prompt, device)
        _append_jsonl(output, records, round_floats=False)
        done += len(chunk)
        passes += 4 * len(chunk)
        print(f"path1 {done}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return {"rows_new": len(pending), "logical_passes": passes,
            "seconds": time.perf_counter() - started}


def _pair_norms(rows: list[dict], base_cache: dict, units_by_id: dict[str, dict]) -> dict:
    import numpy as np

    output = []
    for row in rows:
        if row["row_type"] != "pair" or not math.isclose(row["dose"], .8):
            continue
        a, b = (push["unit_id"] for push in row["pushes"])
        if units_by_id[a]["layer"] != units_by_id[b]["layer"]:
            output.append({"row_id": row["row_id"], "first": a, "second": b,
                           "same_layer": False, "combined_norm": None})
            continue
        combined = base_cache[(a, "neuron")].float() + base_cache[(b, "neuron")].float()
        output.append({"row_id": row["row_id"], "first": a, "second": b,
                       "same_layer": True, "combined_norm": float(np.linalg.norm(
                           combined.detach().cpu().numpy().astype(np.float64)))})
    same = [item["combined_norm"] for item in output if item["same_layer"]]
    return {"rows": output, "same_layer_pairs": len(same),
            "median_same_layer_combined_norm": float(__import__("numpy").median(same)) if same else None}


def _answer_projection(rows: list[dict], pilot_records: list[dict]) -> int | None:
    """Project new answer JSONL bytes by design cell, using pilot cell means."""
    from collections import defaultdict

    samples = defaultdict(list)
    for record in pilot_records:
        samples[(record["row_type"], float(record["dose"]))].append(
            len((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")))
    group_counts = defaultdict(int)
    for row in rows:
        if not row["reused"]:
            group_counts[(row["row_type"], float(row["dose"]))] += 1
    fallback = {("pair", .2): [("pair", .4), ("pair", .6)],
                ("control", .4): [("pair", .4)],
                ("randpair", .4): [("randpair", .8)]}
    total = 0.0
    for group, count in group_counts.items():
        values = samples.get(group)
        if not values:
            source_groups = fallback.get(group, [])
            values = [value for source in source_groups for value in samples.get(source, [])]
        if not values:
            return None
        total += count * sum(values) / len(values)
    return round(total)


def run(args: argparse.Namespace, units: list[dict] | None = None, *,
        test_only_allow_unregistered_part119_reference: bool = False) -> dict:
    import torch

    units = (rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
             if units is None else units)
    units_by_id = {unit["id"]: unit for unit in units}
    answer_rows = rb.pilot_answer_rows(units) if args.stage == "pilot" else rb.all_answer_rows(units)
    output_dir = repo_path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    part119_answers_path = repo_path(args.part119_answers)
    part119_interactions_path = repo_path(args.part119_interactions)
    part119_answers = read_jsonl_readonly(part119_answers_path, "row_id")
    reference = []
    reference_attestation = None
    if args.stage == "pilot":
        reference = read_jsonl_readonly(part119_interactions_path)
        reference_attestation = gate119b.validate_part119_reference(
            part119_interactions_path, reference,
            test_only_allow_unregistered_part119_reference=(
                test_only_allow_unregistered_part119_reference))

    tokenizer, model = c.load_model(repo_path(args.model), args.device)
    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, FRAME, args.device)
    base_cache = build_base_push_cache(units, layers, medians)
    cache_by_dose = {dose: build_dose_push_cache(units, layers, medians, dose)
                     for dose in sorted({row["dose"] for row in answer_rows if row["pushes"]}
                                        | set(rb.DOSES))}
    for dose in cache_by_dose:
        validate_dose_scaling(units, base_cache, cache_by_dose[dose], dose)

    gate, gate_seconds = _run_gate1(output_dir / "gate1.json", tokenizer, model, layers, args.device)
    answers_file = output_dir / "answers.jsonl"
    answer_count, answer_seconds, pilot_records = _generate_rows(
        answer_rows, answers_file, tokenizer, model, layers, cache_by_dose,
        units_by_id, args.batch, args.device, quiet=(args.stage == "pilot"))
    local_answers = read_output_by_key(answers_file, "row_id")
    if args.stage == "pilot":
        pilot_records = [local_answers[row["row_id"]] for row in answer_rows
                         if not row["reused"] and row["row_id"] in local_answers]
    answer_by_id = _load_answers_for_path2(part119_answers, local_answers, answer_rows)

    if args.stage == "pilot":
        path2_rows = rb.pilot_path2_rows(units)
        timings = []
        with tempfile.TemporaryDirectory(prefix="part119b-path2-pilot-") as scratch:
            sample_files = {}
            for batch_size in (1, 8):
                path_file = Path(scratch) / f"interaction-batch{batch_size}.jsonl"
                timings.append(run_path2(units, path2_rows, (.2, .8), path_file,
                                         tokenizer, model, layers, base_cache, cache_by_dose,
                                         answer_by_id, args.device, batch_size, quiet=True))
                sample_files[batch_size] = path_file
            faster = min(timings, key=lambda item: item["seconds"])
            pilot_interactions_path = output_dir / "interaction-pilot.jsonl"
            shutil.copyfile(sample_files[faster["batch"]], pilot_interactions_path)
        pilot_rows = read_output_by_key(pilot_interactions_path, "key")
        gate2 = pilot_reproduction_gate(pilot_rows, reference, path2_rows)
        gate2_report = _pilot_gate_report(gate2)
        projected = _answer_projection(answer_rows, pilot_records)
        report = {"stage": "pilot", "answers_new": answer_count,
                  "answer_seconds": answer_seconds, "gate1_passed": gate["passed"],
                  "gate2_sample_passed": gate2_report["passed"],
                  "gate2": gate2_report,
                  "part119_reference": reference_attestation,
                  "gate2_failure_reason": gate2["reason"] if not gate2["passed"] else None,
                  "path2_batch_timings": timings, "path2_batch_selected": faster["batch"],
                  "projected_new_answer_bytes": projected,
                  "gate_seconds": gate_seconds}
        (output_dir / "timing-pilot.json").write_text(
            json.dumps(_round_floats(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # The pilot's stdout deliberately contains only gate outcomes and timing/size data.
        print(json.dumps(_round_floats(report), ensure_ascii=False), flush=True)
        if not gate2["passed"]:
            raise RuntimeError(f"pilot Gate 2 failed: {gate2['reason']}")
        return report

    path2_file = output_dir / "interaction.jsonl"
    path2 = run_path2(units, rb.all_path2_rows(units), rb.DOSES, path2_file,
                      tokenizer, model, layers, base_cache, cache_by_dose,
                      answer_by_id, args.device, args.path2_batch)
    pair_norms = _pair_norms(answer_rows, base_cache, units_by_id)
    (output_dir / "push_norms.json").write_text(
        json.dumps(_round_floats(pair_norms), ensure_ascii=False) + "\n", encoding="utf-8")
    path1_rows = [row for row in answer_rows
                  if row["row_type"] == "pair" and math.isclose(row["dose"], .4)]
    path1 = _run_path1(path1_rows, answer_by_id, output_dir / "path1.jsonl",
                       tokenizer, model, layers, cache_by_dose[.4], args.device,
                       args.path1_batch)
    report = {"stage": "main", "answers_new": answer_count,
              "answer_seconds": answer_seconds, "gate1_passed": gate["passed"],
              "gate_seconds": gate_seconds, "path2": path2, "path1": path1,
              "push_norm_rows": len(pair_norms["rows"]), "batch": args.batch,
              "path2_batch": args.path2_batch}
    (output_dir / "timing.json").write_text(
        json.dumps(_round_floats(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_round_floats(report), ensure_ascii=False), flush=True)
    return report


def pilot_reproduction_gate(pilot_rows: dict[str, dict], reference: list[dict],
                            selected_rows: list[dict]) -> dict:
    """Run the shared Gate 2 calculation on the pilot's selected rows."""
    expected = {"real": set(), "control": set()}
    for row in selected_rows:
        kind = str(row["kind"])
        expected.setdefault(kind, set()).add(
            (kind, str(row["first"]), str(row["second"])))
    return gate119b.gate2_result(list(pilot_rows.values()), reference, expected)


def _pilot_gate_report(gate2: dict) -> dict:
    """Keep the pilot artifact to gate outcomes and diagnostics, not share values."""
    return {key: gate2[key] for key in (
        "passed", "complete", "reason", "failure_reasons", "expected_rows",
        "rows_compared", "missing_row_count", "position_list_mismatch_rows",
        "class_thresholds_passed", "median_relative_nPair_difference",
        "relative_nPair_rows_compared", "nPair_threshold")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local pinned OLMo 2 1B Instruct snapshot")
    parser.add_argument("--out", default="results/part119b")
    parser.add_argument("--device", default="xpu", help="xpu or cpu")
    parser.add_argument("--stage", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--path2-batch", type=int, choices=(1, 8), default=1)
    parser.add_argument("--path1-batch", type=int, default=8)
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    parser.add_argument("--part119-answers", default=PART119_ANSWERS)
    parser.add_argument("--part119-interactions", default=PART119_INTERACTIONS)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.path1_batch < 1:
        parser.error("--batch and --path1-batch must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
