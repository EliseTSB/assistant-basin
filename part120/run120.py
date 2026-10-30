#!/usr/bin/env python3
"""Run Part 120's fixed Path 2 self and cross finite differences in float32."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119c", ROOT / "part119b", ROOT / "part119",
             ROOT / "part92", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run119c  # noqa: E402 -- pinned local model loader and dtype push construction

common = run119c.c
rows119 = run119c.rows119
run119 = run119c.run119
run119b = run119c.run119b
rows119b = run119c.rows119b
rows119c = run119c.rows119c

DOSES = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
T_REFERENCE_PATH = "results/part119c/interaction-float32.jsonl"
ANSWERS_PATH = "results/part119/answers.jsonl"
PATH_TOKENS = run119b.PATH_TOKENS


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def safe_unit_id(unit_id: str) -> str:
    return quote(str(unit_id), safe="") or "unit"


def parse_doses(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("doses must be comma-separated numbers") from exc
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("doses must be a nonempty list without duplicates")
    unknown = [dose for dose in parsed if dose not in DOSES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported dose(s): {unknown}; allowed: {DOSES}")
    return tuple(dose for dose in DOSES if dose in parsed)


def read_answers(path: str | Path) -> dict[str, dict]:
    answer_path = repo_path(path)
    if not answer_path.is_file():
        raise FileNotFoundError(f"Part 119 answers file does not exist: {answer_path}")
    return run119b.read_jsonl_readonly(answer_path, "row_id")


def path_for_first(first_id: str, answers: dict[str, dict], prompt: list[int]) -> dict:
    plain = answers.get("plain")
    single = answers.get(f"single:{first_id}")
    if plain is None or single is None:
        raise ValueError(f"Path 2 needs Part 119 plain and single answers for {first_id}")
    tokens = [int(token) for token in single.get("tokens", [])[:PATH_TOKENS]]
    p, _target_a, _target_0 = run119.first_flip(tokens, plain.get("tokens", []))
    if p is None or p >= len(tokens):
        raise ValueError(f"no fixed Path 2 position exists for first unit {first_id}")
    stop = min(PATH_TOKENS, len(tokens))
    if stop <= p:
        raise ValueError(f"empty fixed Path 2 path for first unit {first_id}")
    return {"tokens": tokens, "p": int(p), "positions": list(range(int(p), stop)),
            "input_tokens": prompt + tokens,
            "logit_start": len(prompt) + int(p) - 1,
            "logit_stop": len(prompt) + stop - 1}


def read_t_support(path: str | Path, rows: list[dict],
                   positions_by_first: dict[str, list[int]]) -> dict[tuple[str, str, str, int], list[int]]:
    path = repo_path(path)
    found: dict[tuple[str, str, str, int], list[int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row.get("dose", -1)) != 0.8:
                continue
            kind, first, second = str(row.get("kind")), str(row.get("first")), str(row.get("second"))
            for position in row.get("positions", []):
                key = (kind, first, second, int(position["pos"]))
                if key in found:
                    raise ValueError(f"duplicate saved T support at {path}:{line_no}: {key}")
                t_ids = position.get("t_ids")
                if not isinstance(t_ids, list) or not t_ids:
                    continue
                ids = [int(token) for token in t_ids]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate token id in saved T support for {key}")
                found[key] = ids
    missing = []
    for row in rows:
        for pos in positions_by_first[str(row["first"])]:
            key = (str(row["kind"]), str(row["first"]), str(row["second"]), int(pos))
            if key not in found:
                missing.append(key)
    if missing:
        raise ValueError(f"Part 119c 0.8 file is missing T for {len(missing)} selected row/positions; first: {missing[0]}")
    return found


def make_index(first_id: str, rows: list[dict], positions: list[int],
               t_ids_by_key: dict, doses: tuple[float, ...],
               parameter_dtype: str = "torch.float32", *,
               scope_units: list[str] | None = None,
               expected_units: list[str] | None = None) -> dict:
    entries, start = [], 0
    for row in rows:
        for pos in positions:
            key = (str(row["kind"]), str(row["first"]), str(row["second"]), int(pos))
            ids = list(t_ids_by_key[key])
            entries.append({"key": str(row["key"]), "kind": str(row["kind"]),
                            "first": str(row["first"]), "second": str(row["second"]),
                            "pos": int(pos), "t_ids": ids, "start": start})
            start += len(doses) * 3 * len(ids)
    selected_ids = sorted(str(value) for value in
                          (scope_units if scope_units is not None else [first_id]))
    expected_ids = sorted(str(value) for value in
                          (expected_units if expected_units is not None else selected_ids))
    is_main_scope = selected_ids == expected_ids and tuple(doses) == DOSES
    return {"first": str(first_id), "doses": [float(dose) for dose in doses],
            "model_dtype": "float32", "model_parameter_dtype": parameter_dtype,
            "scope": {"stage": "main" if is_main_scope else "smoke",
                      "selected_first_units": selected_ids,
                      "expected_first_units": expected_ids},
            "entries": entries}


def read_unit_npz(path: Path) -> tuple[dict, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"index", "values"}:
            raise ValueError(f"{path} must contain only index and values")
        index_value = archive["index"]
        if index_value.shape != () or index_value.dtype.kind not in ("U", "S"):
            raise ValueError(f"{path} index must be a scalar JSON string")
        index = json.loads(str(index_value.item()))
        values = np.asarray(archive["values"])
    if values.ndim != 1 or values.dtype != np.float64:
        raise ValueError(f"{path} values must be a flat float64 array")
    return index, values


def finished_unit(path: Path, expected_index: dict) -> bool:
    if not path.exists():
        return False
    index, values = read_unit_npz(path)
    expected_size = sum(len(entry["t_ids"]) * 3 * len(expected_index["doses"])
                        for entry in expected_index["entries"])
    if index != expected_index or len(values) != expected_size or not np.isfinite(values).all():
        raise ValueError(f"finished vector file does not match this run: {path}")
    return True


def write_unit_npz(path: Path, index: dict, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = sum(len(entry["t_ids"]) * 3 * len(index["doses"])
                        for entry in index["entries"])
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != expected_size or not np.isfinite(values).all():
        raise ValueError(f"incomplete or non-finite values for {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                          dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream,
                                index=np.asarray(json.dumps(index, ensure_ascii=False,
                                                            separators=(",", ":"))),
                                values=values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def centered_delta(base, pushed, t_ids: list[int]) -> np.ndarray:
    import torch

    token_index = torch.tensor(t_ids, dtype=torch.long, device=base.device)
    base_values = base.index_select(0, token_index).detach().cpu().numpy().astype(np.float64, copy=False)
    push_values = pushed.index_select(0, token_index).detach().cpu().numpy().astype(np.float64, copy=False)
    centered_base = base_values - base_values.mean()
    centered_push = push_values - push_values.mean()
    return np.asarray(centered_push - centered_base, dtype=np.float64)


def _zero_hook_gate(units: list[dict], paths: dict[str, dict], answers: dict[str, dict],
                    tokenizer, model, layers, device: str) -> dict:
    import torch

    checks = []
    for unit in units:
        first_id = str(unit["id"])
        path = paths[first_id]
        input_ids = torch.tensor([path["input_tokens"]], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        plain = run119._forward(model, input_ids, mask, layers, {})
        zeros = {}
        for layer_no, layer in enumerate(layers):
            weight = layer.mlp.down_proj.weight
            zeros[layer_no] = torch.zeros((1, int(weight.shape[0])),
                                          dtype=weight.dtype, device=device)
        zero = run119._forward(model, input_ids, mask, layers, zeros)
        equal = bool(torch.equal(plain, zero))
        difference = (float((plain - zero).abs().max().item()) if not equal else 0.0)
        checks.append({"first": first_id, "passed": equal,
                       "positions": path["positions"],
                       "logit_start": path["logit_start"], "logit_stop": path["logit_stop"],
                       "logits_shape": list(plain.shape), "max_abs_difference": difference})
        del plain, zero
    return {"gate": 1, "name": "zero-dose hook equals plain logits",
            "passed": bool(checks) and all(row["passed"] for row in checks),
            "expected_paths": 39 if len(units) == 39 else len(units),
            "paths_checked": len(checks), "scope_units": [str(unit["id"]) for unit in units],
            "model_dtype": "float32",
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "paths": checks}


def _scaling_gate(units: list[dict], cache_by_dose: dict[float, dict],
                  weight_cache_by_dose: dict[float, dict]) -> dict:
    import torch

    threshold = 1e-5
    all_errors, rebuild_errors, per_unit = [], [], []
    for unit in units:
        unit_max = 0.0
        unit_checks = []
        for low, high in zip(DOSES, DOSES[1:]):
            for arm in ("neuron", "random"):
                low_push = cache_by_dose[low][(unit["id"], arm)].float()
                high_push = cache_by_dose[high][(unit["id"], arm)].float()
                expected_push = 2.0 * low_push
                denominator = float(torch.linalg.vector_norm(expected_push).item())
                error = (float(torch.linalg.vector_norm(high_push - 2.0 * low_push).item())
                         / denominator if denominator > 0 else math.nan)
                passed = math.isfinite(error) and error <= threshold
                unit_max = max(unit_max, error) if math.isfinite(error) else math.inf
                all_errors.append(error if math.isfinite(error) else None)
                unit_checks.append({"low_dose": low, "high_dose": high,
                                    "arm": arm, "relative_error": error if math.isfinite(error) else None,
                                    "passed": passed})
        rebuild_checks = []
        for dose in DOSES:
            for arm in ("neuron", "random"):
                actual_push = cache_by_dose[dose][(unit["id"], arm)].float()
                rebuilt_push = weight_cache_by_dose[dose][(unit["id"], arm)].float()
                denominator = float(torch.linalg.vector_norm(rebuilt_push).item())
                error = (float(torch.linalg.vector_norm(actual_push - rebuilt_push).item())
                         / denominator if denominator > 0 else math.nan)
                passed = math.isfinite(error) and error <= threshold
                rebuild_errors.append(error if math.isfinite(error) else None)
                rebuild_checks.append({"dose": dose, "arm": arm,
                                       "relative_error": error if math.isfinite(error) else None,
                                       "passed": passed})
        per_unit.append({"unit": str(unit["id"]),
                         "max_relative_error": unit_max if math.isfinite(unit_max) else None,
                         "passed": (bool(unit_checks) and all(item["passed"] for item in unit_checks)
                                    and bool(rebuild_checks)
                                    and all(item["passed"] for item in rebuild_checks)),
                         "comparisons": unit_checks,
                         "weight_rebuild_comparisons": rebuild_checks})
    finite_errors = [float(error) for error in all_errors
                     if error is not None and math.isfinite(float(error))]
    maximum = max(finite_errors) if finite_errors else None
    finite_rebuild_errors = [float(error) for error in rebuild_errors
                             if error is not None and math.isfinite(float(error))]
    maximum_rebuild = max(finite_rebuild_errors) if finite_rebuild_errors else None
    return {"gate": 3, "name": "push at twice alpha equals twice the push",
            "threshold_relative_error_at_most": threshold,
            "passed": (bool(per_unit) and len(finite_errors) == len(all_errors)
                       and all(value <= threshold for value in finite_errors)
                       and len(finite_rebuild_errors) == len(rebuild_errors)
                       and all(value <= threshold for value in finite_rebuild_errors)),
            "maximum_relative_error": maximum,
            "weight_rebuild_source": "run119b.build_dose_push_cache at each dose",
            "maximum_weight_rebuild_relative_error": maximum_rebuild,
            "dose_pairs_checked": [[low, high] for low, high in zip(DOSES, DOSES[1:])],
            "units_checked": len(per_unit), "scope_units": [str(unit["unit"]) for unit in per_unit],
            "units": per_unit}


def _run_first_unit(first_id: str, rows: list[dict], path: dict, index: dict,
                    t_ids_by_key: dict, doses: tuple[float, ...], tokenizer, model,
                    layers, cache_by_dose: dict[float, dict], device: str,
                    batch_size: int) -> tuple[np.ndarray, dict]:
    import torch

    input_ids = torch.tensor([path["input_tokens"]], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    start, stop = int(path["logit_start"]), int(path["logit_stop"])
    base_logits = run119._forward(model, input_ids, mask, layers, {})[0, start:stop, :]
    first_logits = {}
    first_row = {"pushes": [{"unit_id": first_id, "arm": "neuron"}]}
    for dose in doses:
        first_vectors = run119.vectors_for_rows([first_row], cache_by_dose[dose], layers, device)
        first_logits[dose] = run119._forward(
            model, input_ids, mask, layers, first_vectors)[0, start:stop, :]

    values_by_entry: dict[tuple[str, int], dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for row in rows:
        for pos in path["positions"]:
            values_by_entry[(str(row["key"]), int(pos))] = {}
    for dose in doses:
        for begin in range(0, len(rows), batch_size):
            chunk = rows[begin:begin + batch_size]
            second_rows = [{"pushes": [row["pushes"][1]]} for row in chunk]
            both_rows = [{"pushes": row["pushes"]} for row in chunk]
            expanded_ids = input_ids.expand(len(chunk), -1)
            expanded_mask = mask.expand(len(chunk), -1)
            second_logits = run119._forward(
                model, expanded_ids, expanded_mask, layers,
                run119.vectors_for_rows(second_rows, cache_by_dose[dose], layers, device),
            )[:, start:stop, :]
            both_logits = run119._forward(
                model, expanded_ids, expanded_mask, layers,
                run119.vectors_for_rows(both_rows, cache_by_dose[dose], layers, device),
            )[:, start:stop, :]
            for row_no, row in enumerate(chunk):
                row_key = str(row["key"])
                for offset, pos in enumerate(path["positions"]):
                    support_key = (str(row["kind"]), first_id, str(row["second"]), int(pos))
                    t_ids = t_ids_by_key[support_key]
                    a = centered_delta(base_logits[offset], first_logits[dose][offset], t_ids)
                    b = centered_delta(base_logits[offset], second_logits[row_no, offset], t_ids)
                    p = centered_delta(base_logits[offset], both_logits[row_no, offset], t_ids)
                    values_by_entry[(row_key, int(pos))][dose] = (a, b, p)
            del second_logits, both_logits

    flattened = []
    for entry in index["entries"]:
        item = values_by_entry[(entry["key"], int(entry["pos"]))]
        for dose in doses:
            if dose not in item:
                raise ValueError(f"incomplete unit output for {entry['key']} pos {entry['pos']} dose {dose}")
            for vector in item[dose]:
                if len(vector) != len(entry["t_ids"]):
                    raise ValueError("centered vector length differs from saved T")
                flattened.extend(vector.tolist())
    return np.asarray(flattened, dtype=np.float64), {
        "logical_passes": 1 + len(doses) + 2 * len(rows) * len(doses),
        "rows": len(rows), "positions": len(path["positions"])}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    import torch

    started = time.perf_counter()
    all_units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    all_rows = rows119c.draw_path2_rows(all_units, rows119b.all_path2_rows(all_units))
    ordered_units = sorted(all_units, key=lambda unit: str(unit["id"]))
    if args.units is not None:
        if not 1 <= args.units <= len(ordered_units):
            raise ValueError(f"--units must be between 1 and {len(ordered_units)}")
        units = ordered_units[:args.units]
    else:
        units = ordered_units
    first_ids = {str(unit["id"]) for unit in units}
    rows = [row for row in all_rows if str(row["first"]) in first_ids]
    answers = read_answers(ANSWERS_PATH)
    tokenizer, model = run119c.load_model(repo_path(args.model), args.device, torch.float32)
    parameter = next(model.parameters(), None)
    if parameter is None or parameter.dtype != torch.float32:
        raise ValueError("Part 120 requires a loaded float32 model parameter")
    if args.device == "cpu" and str(parameter.device) != "cpu":
        model.to(args.device)
    prompt = common.prompt_ids(tokenizer, run119.QUESTION, run119.FRAME)
    paths = {str(unit["id"]): path_for_first(str(unit["id"]), answers, prompt)
             for unit in units}
    positions_by_first = {first: path["positions"] for first, path in paths.items()}
    t_ids_by_key = read_t_support(T_REFERENCE_PATH, rows, positions_by_first)
    doses = tuple(args.doses)
    smoke = len(units) != len(all_units) or doses != DOSES
    out_dir = repo_path(args.out)
    if smoke and out_dir.resolve() == repo_path("results/part120").resolve():
        raise ValueError("a smoke must write to its own --out directory")
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    rows_by_first = {first: [row for row in rows if str(row["first"]) == first]
                     for first in paths}
    selected_ids = sorted(str(unit["id"]) for unit in units)
    expected_ids = sorted(str(unit["id"]) for unit in all_units)
    indexes = {first: make_index(first, rows_by_first[first], paths[first]["positions"],
                                 t_ids_by_key, doses, str(parameter.dtype),
                                 scope_units=selected_ids, expected_units=expected_ids)
               for first in paths}
    completed = {first: finished_unit(vectors_dir / f"{safe_unit_id(first)}.npz", indexes[first])
                 for first in paths}

    layers = common.get_layers(model)
    medians = common.residual_medians(model, tokenizer, run119.FRAME, args.device)
    _base_cache, cache_by_dose = run119c.build_dtype_push_caches(units, layers, medians, DOSES)
    weight_cache_by_dose = {
        float(dose): run119b.build_dose_push_cache(units, layers, medians, float(dose))
        for dose in DOSES
    }
    gate1 = _zero_hook_gate(units, paths, answers, tokenizer, model, layers, args.device)
    gate3 = _scaling_gate(units, cache_by_dose, weight_cache_by_dose)
    report = {"stage": "main" if not smoke else "smoke", "seed": rows119c.SEED,
              "rows": len(rows), "first_units": len(units), "batch": args.batch,
              "device": args.device, "doses": list(doses),
              "scope_units": selected_ids,
              "model_dtype": "float32", "model_parameter_dtype": str(parameter.dtype),
              "gate1_passed": gate1["passed"], "gate3_passed": gate3["passed"],
              "units_skipped_finished": [first for first, value in completed.items() if value],
              "units_written": [], "logical_passes": 0,
              "elapsed_seconds": time.perf_counter() - started}
    try:
        _write_json(out_dir / "gate1.json", gate1)
        _write_json(out_dir / "gate3.json", gate3)
        if not gate1["passed"] or not gate3["passed"]:
            report["status"] = "stopped_by_gate"
            _write_json(out_dir / "timing.json", report)
            raise RuntimeError("Part 120 gate 1 or gate 3 failed; vector run stopped")

        for unit in units:
            first = str(unit["id"])
            path = vectors_dir / f"{safe_unit_id(first)}.npz"
            if completed[first]:
                print(f"path120 first={first} skipped finished unit", flush=True)
                continue
            values, unit_timing = _run_first_unit(
                first, rows_by_first[first], paths[first], indexes[first], t_ids_by_key,
                doses, tokenizer, model, layers, cache_by_dose, args.device, args.batch)
            write_unit_npz(path, indexes[first], values)
            report["units_written"].append(first)
            report["logical_passes"] += unit_timing["logical_passes"]
            print(f"path120 first={first} rows={unit_timing['rows']} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        report.update({"status": "complete", "elapsed_seconds": time.perf_counter() - started,
                       "total_expected_units": len(all_units)})
        _write_json(out_dir / "timing.json", report)
    except OSError as exc:
        report.update({"status": "aborted", "write_error": str(exc),
                       "elapsed_seconds": time.perf_counter() - started,
                       "total_expected_units": len(all_units)})
        try:
            _write_json(out_dir / "timing.json", report)
        except OSError:
            pass
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="local pinned model snapshot")
    parser.add_argument("--out", default="results/part120")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    parser.add_argument("--units", type=int, help="first N units for an isolated smoke run")
    parser.add_argument("--doses", type=parse_doses, default=DOSES,
                        help="comma-separated subset for a smoke run; defaults to all doses")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"run120: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
