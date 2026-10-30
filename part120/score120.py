#!/usr/bin/env python3
"""CPU-only Part 120 statistics, gates, predictions, and headline rule."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from urllib.parse import quote

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119c", ROOT / "part119b", ROOT / "part119", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rows119  # noqa: E402
import rows119c  # noqa: E402
import score119  # noqa: E402 -- registered unit-bootstrap draw machinery

DOSES = (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
ANALYSIS_DOSES = DOSES[:-1]
GATE2_DOSES = (0.05, 0.1, 0.2, 0.4, 0.8)
GATE2_FIELDS = {"nA": "A", "n2": "B", "nPair": "P", "nI": "I"}
KINDS = ("real", "control")
BOOTSTRAP_SEED = 12001
BOOTSTRAP_DRAWS = 2000
GATE2_THRESHOLD = 0.005
GATE3_THRESHOLD = 1e-5

PREDICTION_TEXT = {
    "120.1": "In float32 on real rows, the median row ‖I‖ follows a square law at low dose: log2 of its ratio between adjacent doses is within 2 plus or minus 0.3 from 0.025 to 0.05 and from 0.05 to 0.1.",
    "120.2": "The second-order identity holds at low dose and has broken by 0.2: on real rows the median row q is at most 0.30 at 0.05 and at least 0.50 at 0.2.",
    "120.3": "A push interacts with itself more than with a second push: at 0.1 the median row c over real rows is at most 0.75, and its product-weighted 95% unit-bootstrap interval lies below 1.",
    "120.4": "At each of 0.05, 0.1 and 0.2, the real median c is within a factor of 1.25 of the control median c.",
    "120.5": "At 0.4 on real rows, the median self share is at least half the median pair share and at most twice it: a push at 0.8 is about as far from twice itself as a pair at 0.4 is from the sum of its parts.",
    "120.6": "At 0.4, over the 39 first units, u is higher for slot units than for concept units: the slot median is above the concept median, with a one-sided Mann-Whitney p below 0.05 on Part 92's Haiku kinds, and the same direction on the units both readers agree on.",
    "120.7": "Across the 39 units, the Spearman correlation of u(0.4) with Part 119's dominance is at most -0.3.",
    "120.8": "At 0.1, a real unit's self term is the size of a random direction's on the same path: taking for each first unit the median row ‖S_B‖ on its real rows over the same on its control rows, the median over units lies between 0.8 and 1.25.",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def safe_unit_id(unit_id: str) -> str:
    return quote(str(unit_id), safe="") or "unit"


def read_json(path: str | Path, *, optional: bool = False):
    path = repo_path(path)
    if optional and not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    path = repo_path(path)
    if optional and not path.is_file():
        return []
    result = []
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
            result.append(row)
    return result


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _median_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if _finite(value)]
    return float(median(clean)) if clean else None


def _row_key(row: dict) -> tuple[str, str, str]:
    return str(row["kind"]), str(row["first"]), str(row["second"])


def _vector_norms(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    a = np.asarray(arrays["A"], dtype=np.float64)
    b = np.asarray(arrays["B"], dtype=np.float64)
    p = np.asarray(arrays["P"], dtype=np.float64)
    interaction = p - a - b
    return {"nA": float(np.linalg.norm(a)), "n2": float(np.linalg.norm(b)),
            "nPair": float(np.linalg.norm(p)), "nI": float(np.linalg.norm(interaction))}


def load_vectors(vectors_dir: str | Path, units: list[dict]) -> tuple[dict, list[dict]]:
    """Read per-unit NPZ files and return row-position-dose vectors plus layout issues."""
    import numpy as np

    directory = repo_path(vectors_dir)
    positions: dict[tuple[str, str, str, int], dict[float, dict[str, np.ndarray]]] = {}
    issues = []
    for unit in units:
        first_id = str(unit["id"])
        path = directory / f"{safe_unit_id(first_id)}.npz"
        if not path.is_file():
            issues.append({"problem": "missing_unit_file", "unit": first_id})
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"index", "values"}:
                    raise ValueError("NPZ keys are not exactly index and values")
                encoded = archive["index"]
                values = np.asarray(archive["values"])
            if encoded.shape != () or encoded.dtype.kind not in ("U", "S"):
                raise ValueError("index is not a scalar JSON string")
            index = json.loads(str(encoded.item()))
            if values.ndim != 1 or values.dtype != np.float64:
                raise ValueError("values are not a flat float64 array")
            if str(index.get("first")) != first_id:
                raise ValueError("index first unit disagrees with filename")
            if (index.get("model_dtype") != "float32"
                    or index.get("model_parameter_dtype") != "torch.float32"):
                issues.append({"problem": "wrong_dtype_evidence", "unit": first_id,
                               "model_dtype": index.get("model_dtype"),
                               "model_parameter_dtype": index.get("model_parameter_dtype")})
            doses = tuple(float(dose) for dose in index.get("doses", []))
            if doses != DOSES:
                issues.append({"problem": "incomplete_or_wrong_dose_ladder", "unit": first_id,
                               "observed_doses": list(doses)})
            entries = index.get("entries")
            if not isinstance(entries, list):
                raise ValueError("index entries are not a list")
            expected_start = 0
            for entry in entries:
                required = ("key", "kind", "first", "second", "pos", "t_ids", "start")
                if any(field not in entry for field in required):
                    raise ValueError("index entry is missing required fields")
                t_ids = entry["t_ids"]
                if not isinstance(t_ids, list) or not t_ids:
                    issues.append({"problem": "empty_T", "unit": first_id,
                                   "key": entry.get("key"), "pos": entry.get("pos")})
                    continue
                width = len(t_ids)
                start = int(entry["start"])
                if start != expected_start:
                    issues.append({"problem": "noncontiguous_start", "unit": first_id,
                                   "expected": expected_start, "observed": start})
                row_position = (str(entry["kind"]), str(entry["first"]),
                                str(entry["second"]), int(entry["pos"]))
                if row_position in positions:
                    issues.append({"problem": "duplicate_row_position", "row_position": row_position})
                    continue
                by_dose = {}
                for dose_no, dose in enumerate(doses):
                    offset = start + dose_no * 3 * width
                    stop = offset + 3 * width
                    if offset < 0 or stop > len(values):
                        issues.append({"problem": "truncated_values", "unit": first_id,
                                       "key": entry["key"], "pos": entry["pos"], "dose": dose})
                        continue
                    a = values[offset:offset + width]
                    b = values[offset + width:offset + 2 * width]
                    p = values[offset + 2 * width:stop]
                    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(p).all()):
                        issues.append({"problem": "nonfinite_vector", "unit": first_id,
                                       "key": entry["key"], "pos": entry["pos"], "dose": dose})
                        continue
                    by_dose[dose] = {"A": a.copy(), "B": b.copy(), "P": p.copy()}
                positions[row_position] = by_dose
                expected_start = start + len(doses) * 3 * width
            if expected_start != len(values):
                issues.append({"problem": "values_length_mismatch", "unit": first_id,
                               "expected": expected_start, "observed": len(values)})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            issues.append({"problem": "unreadable_unit_file", "unit": first_id,
                           "detail": str(exc)})
    return positions, issues


def gate1_result(record: dict | None, expected_unit_ids: list[str]) -> dict:
    checks = record.get("paths", []) if isinstance(record, dict) else []
    observed_ids = [str(row.get("first", "")) for row in checks if isinstance(row, dict)]
    expected_ids = [str(value) for value in expected_unit_ids]
    details = []
    for row in checks:
        if not isinstance(row, dict):
            details.append({"problem": "malformed_path_record"})
        elif (row.get("passed") is not True or not _finite(row.get("max_abs_difference"))
              or float(row.get("max_abs_difference")) != 0.0):
            details.append({"problem": "path_not_equal", "first": row.get("first")})
    complete = len(checks) == len(expected_ids) and len(set(observed_ids)) == len(expected_ids) \
        and set(observed_ids) == set(expected_ids)
    dtype_ok = (isinstance(record, dict) and record.get("model_dtype") == "float32"
                and record.get("model_parameter_dtype") == "torch.float32")
    passed = bool(isinstance(record, dict) and record.get("passed") is True and complete
                  and dtype_ok and not details)
    return {"name": "float32 zero-dose hook logits equal plain logits", "passed": passed,
            "expected_paths": len(expected_ids), "paths_checked": len(checks),
            "complete_path_set": complete, "dtype_evidence_passed": dtype_ok,
            "path_failures": details[:25], "path_failure_count": len(details)}


def gate3_result(record: dict | None, expected_unit_ids: list[str]) -> dict:
    threshold = GATE3_THRESHOLD
    unit_rows = record.get("units", []) if isinstance(record, dict) else []
    by_unit = {str(row.get("unit")): row for row in unit_rows if isinstance(row, dict)}
    expected = {str(value) for value in expected_unit_ids}
    missing_units = sorted(expected - set(by_unit))
    extra_units = sorted(set(by_unit) - expected)
    failures = []
    maxima = []
    for unit_id in sorted(expected & set(by_unit)):
        row = by_unit[unit_id]
        comparisons = row.get("comparisons", [])
        expected_comparisons = (len(DOSES) - 1) * 2
        if len(comparisons) != expected_comparisons:
            failures.append({"unit": unit_id, "problem": "incomplete_comparisons",
                             "observed": len(comparisons), "expected": expected_comparisons})
        expected_pairs = {(low, high, arm) for low, high in zip(DOSES, DOSES[1:])
                          for arm in ("neuron", "random")}
        observed_pairs = set()
        for item in comparisons:
            if isinstance(item, dict):
                observed_pairs.add((item.get("low_dose"), item.get("high_dose"), item.get("arm")))
            error = item.get("relative_error") if isinstance(item, dict) else None
            if not _finite(error) or float(error) > threshold or item.get("passed") is not True:
                failures.append({"unit": unit_id, "problem": "scaling_error",
                                 "relative_error": error})
                continue
            maxima.append(float(error))
        if observed_pairs != expected_pairs:
            failures.append({"unit": unit_id, "problem": "wrong_dose_pair_or_arm_set"})
        if row.get("passed") is not True:
            failures.append({"unit": unit_id, "problem": "unit_failed"})
    complete = not missing_units and not extra_units and len(unit_rows) == len(expected)
    passed = bool(isinstance(record, dict) and record.get("passed") is True and complete
                  and not failures and len(maxima) == len(expected) * (len(DOSES) - 1) * 2)
    return {"name": "2 alpha push equals twice alpha push", "passed": passed,
            "threshold_relative_error_at_most": threshold,
            "maximum_relative_error": max(maxima) if maxima else None,
            "expected_units": len(expected), "units_checked": len(unit_rows),
            "missing_units": missing_units, "extra_units": extra_units,
            "failures": failures[:25], "failure_count": len(failures)}


def gate2_result(vectors: dict, reference_rows: list[dict], selected_rows: list[dict],
                 input_issues: list[dict] | None = None) -> dict:
    """Compare raw centered-change norms with Part 119c, row-balanced by position."""
    reference = defaultdict(dict)
    for row in reference_rows:
        row_key, dose = _row_key(row), float(row.get("dose", -1))
        if dose in reference[row_key]:
            reference[row_key][dose] = None
        else:
            reference[row_key][dose] = row
    by_dose = {}
    issue_rows = []
    for dose in GATE2_DOSES:
        field_stats = {}
        for field in GATE2_FIELDS:
            row_values = []
            missing_count = position_mismatch_count = zero_reference_count = 0
            for source in selected_rows:
                row_key = _row_key(source)
                old_row = reference.get(row_key, {}).get(dose)
                if old_row is None:
                    missing_count += 1
                    issue_rows.append({"kind": row_key[0], "first": row_key[1],
                                       "second": row_key[2], "dose": dose,
                                       "problem": "missing_reference_row"})
                    continue
                old_position_rows = old_row.get("positions", [])
                old_positions = {int(item["pos"]): item for item in old_position_rows}
                current_positions = {key[3]: by_dose for key, by_dose in vectors.items()
                                     if key[:3] == row_key and dose in by_dose}
                duplicate_old_positions = len(old_positions) != len(old_position_rows)
                if set(current_positions) != set(old_positions) or duplicate_old_positions:
                    position_mismatch_count += 1
                    issue_rows.append({"kind": row_key[0], "first": row_key[1],
                                       "second": row_key[2], "dose": dose,
                                       "problem": "duplicate_or_mismatched_positions"})
                relative = []
                bad_value = False
                for pos in sorted(set(current_positions) & set(old_positions)):
                    arrays = current_positions[pos][dose]
                    try:
                        current_norms = _vector_norms(arrays)
                    except (KeyError, TypeError, ValueError):
                        current_norms = {}
                    before = old_positions[pos].get(field)
                    after = current_norms.get(field)
                    if not _finite(before) or not _finite(after):
                        bad_value = True
                        issue_rows.append({"kind": row_key[0], "first": row_key[1],
                                           "second": row_key[2], "dose": dose,
                                           "pos": pos, "field": field,
                                           "problem": "missing_or_nonfinite_value"})
                        continue
                    before, after = float(before), float(after)
                    if before > 0:
                        relative.append(abs(after - before) / before)
                    elif after == 0:
                        relative.append(0.0)
                    else:
                        zero_reference_count += 1
                        issue_rows.append({"kind": row_key[0], "first": row_key[1],
                                           "second": row_key[2], "dose": dose,
                                           "pos": pos, "problem": "zero_reference_nonzero"})
                if relative and not bad_value:
                    row_values.append(float(median(relative)))
                else:
                    missing_count += 1
            value = _median_or_none(row_values)
            field_stats[field] = {"median_relative_difference": value,
                                  "rows_compared": len(row_values),
                                  "expected_rows": len(selected_rows),
                                  "missing_rows": missing_count,
                                  "position_mismatch_rows": position_mismatch_count,
                                  "zero_reference_nonzero_positions": zero_reference_count,
                                  "passed": bool(value is not None and value < GATE2_THRESHOLD
                                                 and not math.isclose(value, GATE2_THRESHOLD,
                                                                      rel_tol=1e-12, abs_tol=1e-15)
                                                 and len(row_values) == len(selected_rows)
                                                 and not missing_count
                                                 and not position_mismatch_count
                                                 and not zero_reference_count)}
        by_dose[f"{dose:g}"] = field_stats
    issues = list(input_issues or [])
    passed = (not issues and bool(selected_rows)
              and all(item["passed"] for stats in by_dose.values()
                      for item in stats.values()))
    return {"name": "float32 norms reproduce Part 119c", "passed": bool(passed),
            "threshold_median_relative_difference_below": GATE2_THRESHOLD,
            "relative_difference_unit": "median of within-row position medians, then median across rows",
            "by_dose": by_dose, "input_issues": issues[:25],
            "input_issue_count": len(issues), "comparison_issues": issue_rows[:25],
            "comparison_issue_count": len(issue_rows)}


def position_statistics(vectors: dict) -> dict:
    """Compute I, self terms, E, shares, q, c, and exploratory cosines per position."""
    result = {}
    for row_position, dose_map in vectors.items():
        by_dose = {}
        raw = {}
        for dose, arrays in dose_map.items():
            if any(not np.isfinite(np.asarray(arrays[field])).all() for field in ("A", "B", "P")):
                continue
            raw[dose] = {field: np.asarray(arrays[field], dtype=np.float64)
                         for field in ("A", "B", "P")}
            raw[dose]["I"] = raw[dose]["P"] - raw[dose]["A"] - raw[dose]["B"]
            norms = {"A": float(np.linalg.norm(raw[dose]["A"])),
                     "B": float(np.linalg.norm(raw[dose]["B"])),
                     "P": float(np.linalg.norm(raw[dose]["P"])),
                     "I": float(np.linalg.norm(raw[dose]["I"]))}
            by_dose[dose] = {"norm_A": norms["A"], "norm_B": norms["B"],
                             "norm_P": norms["P"], "norm_I": norms["I"],
                             "eligible": None}
        for alpha in ANALYSIS_DOSES:
            high = 2 * alpha
            if alpha not in raw or high not in raw:
                continue
            a, b, p = raw[alpha]["A"], raw[alpha]["B"], raw[alpha]["P"]
            sa = raw[high]["A"] - 2 * a
            sb = raw[high]["B"] - 2 * b
            sp = raw[high]["P"] - 2 * p
            interaction = p - a - b
            error = sp - sa - sb - 2 * interaction
            norms = {"I": float(np.linalg.norm(interaction)),
                     "SA": float(np.linalg.norm(sa)), "SB": float(np.linalg.norm(sb)),
                     "SP": float(np.linalg.norm(sp)), "E": float(np.linalg.norm(error)),
                     "P": float(np.linalg.norm(p)), "A2": float(np.linalg.norm(raw[high]["A"]))}
            eligible = norms["P"] > 0 and norms["SA"] > 0 and norms["SB"] > 0
            c_value = (norms["I"] / math.sqrt(norms["SA"] * norms["SB"]) if eligible else None)
            q_value = norms["E"] / norms["SP"] if norms["SP"] > 0 else None
            pair_share = (norms["I"] / norms["P"]) ** 2 if norms["P"] > 0 else None
            self_share = (norms["SA"] / norms["A2"]) ** 2 if norms["A2"] > 0 else None
            cos_a = (float(np.dot(interaction, sa) / (norms["I"] * norms["SA"]))
                     if norms["I"] > 0 and norms["SA"] > 0 else None)
            cos_b = (float(np.dot(interaction, sb) / (norms["I"] * norms["SB"]))
                     if norms["I"] > 0 and norms["SB"] > 0 else None)
            by_dose[alpha].update({"norm_I": norms["I"], "norm_SA": norms["SA"],
                                   "norm_SB": norms["SB"], "norm_SP": norms["SP"],
                                   "norm_E": norms["E"], "c": c_value, "q": q_value,
                                   "pair_share": pair_share, "self_share": self_share,
                                   "cosine_I_SA": cos_a, "cosine_I_SB": cos_b,
                                   "eligible": bool(eligible)})
        result[row_position] = by_dose
    return result


def _row_statistics(position_stats: dict, selected_rows: list[dict], units: list[dict]) -> dict:
    positions_by_row = defaultdict(list)
    for (kind, first, second, pos), doses in position_stats.items():
        positions_by_row[(kind, first, second)].append((pos, doses))
    result = {}
    for row in selected_rows:
        row_key = _row_key(row)
        row_positions = positions_by_row.get(row_key, [])
        for dose in DOSES:
            candidates = [per_dose.get(dose) for _pos, per_dose in row_positions]
            candidates = [item for item in candidates if item is not None]
            eligible = [item for item in candidates if item.get("eligible") is True]
            if dose == 0.8:
                norm_i = _median_or_none([item.get("norm_I") for item in candidates
                                          if _finite(item.get("norm_I"))])
                values = {"norm_I": norm_i}
                eligible_count = dropped_count = None
            else:
                fields = ("norm_I", "norm_SA", "norm_SB", "norm_SP", "norm_E",
                          "c", "q", "pair_share", "self_share", "cosine_I_SA", "cosine_I_SB")
                values = {field: _median_or_none([item.get(field) for item in eligible
                                                  if _finite(item.get(field))])
                          for field in fields}
                eligible_count = len(eligible)
                dropped_count = max(0, len(row_positions) - eligible_count)
            result[(*row_key, dose)] = {
                "kind": row_key[0], "first": row_key[1], "second": row_key[2],
                "dose": dose, "positions_considered": len(row_positions),
                "positions_eligible": eligible_count, "positions_dropped": dropped_count,
                **values}
    return result


def _summary_table(row_stats: dict) -> tuple[list[dict], dict]:
    table, dropped = [], {}
    fields = ("norm_I", "norm_SA", "norm_SB", "norm_SP", "norm_E", "c", "q",
              "pair_share", "self_share")
    for dose in DOSES:
        dropped[dose] = {}
        for kind in KINDS:
            rows = [row for key, row in row_stats.items()
                    if key[0] == kind and key[3] == dose]
            eligible_count = sum(int(row["positions_eligible"] or 0) for row in rows)
            dropped_count = sum(int(row["positions_dropped"] or 0) for row in rows)
            considered = sum(int(row["positions_considered"]) for row in rows)
            cell = {"dose": dose, "kind": kind, "rows": len(rows),
                    "positions_considered": considered,
                    "positions_eligible": eligible_count if dose != 0.8 else None,
                    "positions_dropped": dropped_count if dose != 0.8 else None}
            for field in fields:
                cell[f"median_row_{field}"] = _median_or_none(
                    [row[field] for row in rows if _finite(row.get(field))])
            table.append(cell)
            dropped[dose][kind] = {"considered": considered,
                                   "eligible": eligible_count if dose != 0.8 else None,
                                   "dropped": dropped_count if dose != 0.8 else None}
    return table, dropped


def _weighted_median(values: list[float], weights: list[int]) -> float | None:
    kept = sorted((float(value), int(weight)) for value, weight in zip(values, weights)
                  if int(weight) > 0 and _finite(value))
    if not kept:
        return None
    cumulative = np.cumsum([weight for _value, weight in kept], dtype=np.int64)
    total = int(cumulative[-1])

    def at_rank(rank: int) -> float:
        return kept[int(np.searchsorted(cumulative, rank + 1, side="left"))][0]

    return (at_rank((total - 1) // 2) + at_rank(total // 2)) / 2


def unit_bootstrap_interval(row_stats: dict, units: list[dict], *, dose: float = 0.1,
                            kind: str = "real", field: str = "c",
                            draws: int = BOOTSTRAP_DRAWS) -> dict:
    unit_ids = [str(unit["id"] if isinstance(unit, dict) else unit) for unit in units]
    selected = [row for key, row in row_stats.items()
                if key[0] == kind and key[3] == float(dose) and _finite(row.get(field))]
    values = [float(row[field]) for row in selected]
    boot_draws = score119.make_unit_bootstrap_draws(
        unit_ids, draws=draws, seed=BOOTSTRAP_SEED)
    product_samples, sum_samples = [], []
    for counts in boot_draws:
        product_weights = [counts.get(row["first"], 0) * counts.get(row["second"], 0)
                           for row in selected]
        sum_weights = [counts.get(row["first"], 0) + counts.get(row["second"], 0)
                       for row in selected]
        for samples, weights in ((product_samples, product_weights), (sum_samples, sum_weights)):
            value = _weighted_median(values, weights)
            if value is not None:
                samples.append(value)

    def interval(samples):
        if not samples:
            return [None, None]
        return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]

    return {"point_median": _median_or_none(values),
            "ci95_product_weighted": interval(product_samples),
            "ci95_sum_weighted": interval(sum_samples),
            "draws": draws, "seed": BOOTSTRAP_SEED,
            "valid_product_draws": len(product_samples), "valid_sum_draws": len(sum_samples),
            "units": len(unit_ids), "rows": len(selected)}


def _unit_values(row_stats: dict, units: list[dict]) -> dict[str, dict]:
    output = {}
    for unit in units:
        first = str(unit["id"])
        selected = [row for key, row in row_stats.items()
                    if key[1] == first and key[3] == 0.4 and _finite(row.get("self_share"))]
        expected = 8
        values = [float(row["self_share"]) for row in selected]
        output[first] = {"u": _median_or_none(values) if len(values) == expected else None,
                         "rows_with_self_share": len(values), "expected_rows": expected,
                         "haiku_kind": unit.get("haiku_kind"), "ds_kind": unit.get("ds_kind"),
                         "group": unit.get("group"), "layer": int(unit["layer"])}
    return output


def _dominance_map(tournament: dict | None) -> dict[str, float | None]:
    if not isinstance(tournament, dict):
        return {}
    result = {}
    for row in tournament.get("units", []):
        if isinstance(row, dict) and "id" in row:
            value = row.get("dominance")
            result[str(row["id"])] = float(value) if _finite(value) else None
    return result


def _table_value(table: list[dict], dose: float, kind: str, field: str):
    for row in table:
        if row["dose"] == dose and row["kind"] == kind:
            return row.get(field)
    return None


def _factor_match(left, right, factor: float = 1.25) -> tuple[float | None, bool]:
    if not _finite(left) or not _finite(right):
        return None, False
    left, right = float(left), float(right)
    if left == 0 and right == 0:
        return 1.0, True
    if left <= 0 or right <= 0:
        return None, False
    ratio = max(left / right, right / left)
    return ratio, ratio <= factor


def _slot_direction_holds(slot_median, concept_median, p_value) -> bool:
    return (_finite(slot_median) and _finite(concept_median) and _finite(p_value)
            and float(slot_median) > float(concept_median) and float(p_value) < .05)


def _dominance_holds(rho) -> bool:
    return _finite(rho) and float(rho) <= -.3


def predictions(table: list[dict], row_stats: dict, unit_values: dict,
                units: list[dict], tournament: dict | None,
                bootstrap: dict) -> list[dict]:
    real_i_0025 = _table_value(table, .025, "real", "median_row_norm_I")
    real_i_005 = _table_value(table, .05, "real", "median_row_norm_I")
    real_i_01 = _table_value(table, .1, "real", "median_row_norm_I")
    slope_values = []
    for low, high, lo_value, hi_value in ((.025, .05, real_i_0025, real_i_005),
                                          (.05, .1, real_i_005, real_i_01)):
        log_ratio = (math.log2(float(hi_value) / float(lo_value))
                     if _finite(lo_value) and _finite(hi_value)
                     and float(lo_value) > 0 and float(hi_value) > 0 else None)
        slope_values.append({"from_dose": low, "to_dose": high,
                             "log2_norm_ratio": log_ratio,
                             "within_2_plus_or_minus_0.3": (log_ratio is not None
                                                            and 1.7 <= log_ratio <= 2.3)})
    pred1_available = all(item["log2_norm_ratio"] is not None for item in slope_values)
    pred1_holds = pred1_available and all(item["within_2_plus_or_minus_0.3"] for item in slope_values)
    pred1 = {"id": "120.1", "prediction": PREDICTION_TEXT["120.1"],
             "observed": {"adjacent_dose_ratios": slope_values},
             "verdict": "holds" if pred1_holds else "does_not_hold" if pred1_available else "not_evaluable"}

    q05 = _table_value(table, .05, "real", "median_row_q")
    q20 = _table_value(table, .2, "real", "median_row_q")
    pred2_available = _finite(q05) and _finite(q20)
    pred2_holds = pred2_available and float(q05) <= .30 and float(q20) >= .50
    pred2 = {"id": "120.2", "prediction": PREDICTION_TEXT["120.2"],
             "observed": {"real_median_row_q_0.05": q05,
                          "real_median_row_q_0.2": q20},
             "verdict": "holds" if pred2_holds else "does_not_hold" if pred2_available else "not_evaluable"}

    c01 = _table_value(table, .1, "real", "median_row_c")
    c_ci = bootstrap.get("ci95_product_weighted", [None, None])
    pred3_available = _finite(c01) and len(c_ci) == 2 and all(_finite(value) for value in c_ci)
    pred3_holds = pred3_available and float(c01) <= .75 and float(c_ci[1]) < 1.0
    pred3 = {"id": "120.3", "prediction": PREDICTION_TEXT["120.3"],
             "observed": {"real_median_row_c_0.1": c01,
                          "product_weighted_ci95": c_ci,
                          "sum_weighted_ci95": bootstrap.get("ci95_sum_weighted")},
             "verdict": "holds" if pred3_holds else "does_not_hold" if pred3_available else "not_evaluable"}

    factor_checks = []
    for dose in (.05, .1, .2):
        real = _table_value(table, dose, "real", "median_row_c")
        control = _table_value(table, dose, "control", "median_row_c")
        factor, matched = _factor_match(real, control)
        factor_checks.append({"dose": dose, "real_median_c": real,
                              "control_median_c": control,
                              "larger_to_smaller_factor": factor, "within_factor_1.25": matched})
    pred4_available = all(item["larger_to_smaller_factor"] is not None for item in factor_checks)
    pred4_holds = pred4_available and all(item["within_factor_1.25"] for item in factor_checks)
    pred4 = {"id": "120.4", "prediction": PREDICTION_TEXT["120.4"],
             "observed": {"dose_comparisons": factor_checks},
             "verdict": "holds" if pred4_holds else "does_not_hold" if pred4_available else "not_evaluable"}

    self_share = _table_value(table, .4, "real", "median_row_self_share")
    pair_share = _table_value(table, .4, "real", "median_row_pair_share")
    pred5_available = _finite(self_share) and _finite(pair_share)
    pred5_holds = (pred5_available and .5 * float(pair_share) <= float(self_share)
                   <= 2.0 * float(pair_share))
    pred5 = {"id": "120.5", "prediction": PREDICTION_TEXT["120.5"],
             "observed": {"real_median_row_self_share_0.4": self_share,
                          "real_median_row_pair_share_0.4": pair_share},
             "verdict": "holds" if pred5_holds else "does_not_hold" if pred5_available else "not_evaluable"}

    complete_u = len(unit_values) == len(units) and all(
        _finite(unit_values.get(str(unit["id"]), {}).get("u")) for unit in units)
    haiku_slots = [float(unit_values[str(unit["id"])]["u"]) for unit in units
                   if unit.get("haiku_kind") == "slot" and _finite(unit_values.get(str(unit["id"]), {}).get("u"))]
    haiku_concepts = [float(unit_values[str(unit["id"])]["u"]) for unit in units
                      if unit.get("haiku_kind") == "concept" and _finite(unit_values.get(str(unit["id"]), {}).get("u"))]
    agreed_slots = [float(unit_values[str(unit["id"])]["u"]) for unit in units
                    if unit.get("group") == "agreed_slot" and _finite(unit_values.get(str(unit["id"]), {}).get("u"))]
    agreed_concepts = [float(unit_values[str(unit["id"])]["u"]) for unit in units
                       if unit.get("group") == "agreed_concept" and _finite(unit_values.get(str(unit["id"]), {}).get("u"))]
    p_value = None
    if haiku_slots and haiku_concepts:
        from scipy.stats import mannwhitneyu
        p_value = float(mannwhitneyu(haiku_slots, haiku_concepts,
                                     alternative="greater", method="auto").pvalue)
        if not math.isfinite(p_value):
            p_value = None
    haiku_direction = (_median_or_none(haiku_slots), _median_or_none(haiku_concepts))
    agreed_direction = (_median_or_none(agreed_slots), _median_or_none(agreed_concepts))
    pred6_available = (complete_u and p_value is not None and len(haiku_slots) == 21
                       and len(haiku_concepts) == 18 and len(agreed_slots) == 21
                       and len(agreed_concepts) == 8)
    pred6_holds = (pred6_available
                   and _slot_direction_holds(haiku_direction[0], haiku_direction[1], p_value)
                   and _finite(agreed_direction[0]) and _finite(agreed_direction[1])
                   and agreed_direction[0] > agreed_direction[1])
    pred6 = {"id": "120.6", "prediction": PREDICTION_TEXT["120.6"],
             "observed": {"haiku_slot_median_u": haiku_direction[0],
                          "haiku_concept_median_u": haiku_direction[1],
                          "haiku_one_sided_mann_whitney_p": p_value,
                          "haiku_slot_n": len(haiku_slots), "haiku_concept_n": len(haiku_concepts),
                          "agreed_slot_median_u": agreed_direction[0],
                          "agreed_concept_median_u": agreed_direction[1],
                          "agreed_slot_n": len(agreed_slots), "agreed_concept_n": len(agreed_concepts)},
             "verdict": "holds" if pred6_holds else "does_not_hold" if pred6_available else "not_evaluable"}

    dominance = _dominance_map(tournament)
    correlated = [(unit_values.get(str(unit["id"]), {}).get("u"), dominance.get(str(unit["id"])))
                  for unit in units]
    pred7_rho, pred7_n = None, 0
    if all(_finite(left) and _finite(right) for left, right in correlated) and len(correlated) == 39:
        from scipy.stats import spearmanr
        statistic = spearmanr([left for left, _right in correlated],
                              [right for _left, right in correlated])
        if _finite(statistic.statistic):
            pred7_rho, pred7_n = float(statistic.statistic), len(correlated)
    pred7_holds = _dominance_holds(pred7_rho)
    pred7 = {"id": "120.7", "prediction": PREDICTION_TEXT["120.7"],
             "observed": {"spearman_rho_u_0.4_vs_dominance": pred7_rho,
                          "units": pred7_n, "expected_units": 39},
             "verdict": "holds" if pred7_holds else "does_not_hold" if pred7_rho is not None else "not_evaluable"}

    ratios = []
    for first in sorted(unit_values):
        real_rows = [row for key, row in row_stats.items()
                     if key[0] == "real" and key[1] == first and key[3] == .1
                     and _finite(row.get("norm_SB"))]
        control_rows = [row for key, row in row_stats.items()
                        if key[0] == "control" and key[1] == first and key[3] == .1
                        and _finite(row.get("norm_SB"))]
        real_value = _median_or_none([row["norm_SB"] for row in real_rows])
        control_value = _median_or_none([row["norm_SB"] for row in control_rows])
        ratio = (float(real_value) / float(control_value)
                 if _finite(real_value) and _finite(control_value) and float(control_value) > 0
                 else None)
        ratios.append({"first": first, "real_median_row_norm_SB": real_value,
                       "control_median_row_norm_SB": control_value,
                       "real_over_control": ratio,
                       "real_rows": len(real_rows), "control_rows": len(control_rows)})
    ratio_values = [float(row["real_over_control"]) for row in ratios
                    if _finite(row.get("real_over_control"))]
    pred8_ratio = _median_or_none(ratio_values)
    pred8_available = len(ratios) == 39 and len(ratio_values) == 39 and all(
        row["real_rows"] == 4 and row["control_rows"] == 4 for row in ratios)
    pred8_holds = pred8_available and .8 <= pred8_ratio <= 1.25
    pred8 = {"id": "120.8", "prediction": PREDICTION_TEXT["120.8"],
             "observed": {"median_unit_real_over_control_norm_SB_0.1": pred8_ratio,
                          "units": len(ratio_values), "expected_units": 39,
                          "unit_ratios": ratios},
             "verdict": "holds" if pred8_holds else "does_not_hold" if pred8_available else "not_evaluable"}
    return [pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8]


def _floor_ratios(table: list[dict]) -> list[dict]:
    result = []
    for kind in KINDS:
        for field in ("norm_I", "norm_SA"):
            summary_field = f"median_row_{field}"
            for low, high in zip(DOSES, DOSES[1:]):
                low_value = _table_value(table, low, kind, summary_field)
                high_value = _table_value(table, high, kind, summary_field)
                ratio = (float(high_value) / float(low_value)
                         if _finite(low_value) and _finite(high_value) and float(low_value) > 0
                         else None)
                result.append({"kind": kind, "statistic": field, "from_dose": low,
                               "to_dose": high, "higher_over_lower_norm_ratio": ratio})
    return result


def _exploratory(position_stats: dict, row_stats: dict, units: list[dict]) -> list[dict]:
    layers = {str(unit["id"]): int(unit["layer"]) for unit in units}
    rows_by_dose_kind = defaultdict(list)
    for key, row in row_stats.items():
        kind, first, second, dose = key
        rows_by_dose_kind[(dose, kind)].append(row)
    output = []
    for dose in ANALYSIS_DOSES:
        for kind in KINDS:
            rows = rows_by_dose_kind[(dose, kind)]
            same = [row["c"] for row in rows
                    if layers.get(row["first"]) is not None
                    and layers.get(row["second"]) is not None
                    and layers.get(row["first"]) == layers.get(row["second"])
                    and _finite(row.get("c"))]
            different = [row["c"] for row in rows
                          if layers.get(row["first"]) != layers.get(row["second"])
                          and _finite(row.get("c"))]
            output.append({"dose": dose, "kind": kind,
                           "median_row_cosine_I_SA": _median_or_none([row["cosine_I_SA"] for row in rows
                                                                       if _finite(row.get("cosine_I_SA"))]),
                           "median_row_cosine_I_SB": _median_or_none([row["cosine_I_SB"] for row in rows
                                                                       if _finite(row.get("cosine_I_SB"))]),
                           "same_layer_rows": len(same),
                           "same_layer_median_row_c": _median_or_none(same),
                           "different_layer_rows": len(different),
                           "different_layer_median_row_c": _median_or_none(different)})
    return output


def _headline(prediction_rows: list[dict], table: list[dict], bootstrap: dict,
              gates: dict) -> str:
    for number in (1, 2, 3):
        if not gates[f"gate{number}"]["passed"]:
            return f"Nothing is read because gate {number} failed."
    q05 = _table_value(table, .05, "real", "median_row_q")
    if not _finite(q05) or float(q05) > .30:
        return "Second-order reading is withdrawn or unavailable at 0.05 because q must be at most 0.30; the headline rule returns nothing."
    interval = bootstrap.get("ci95_product_weighted", [None, None])
    if len(interval) != 2 or not all(_finite(value) for value in interval):
        return "The product-weighted interval is unavailable; report the numbers and read nothing."
    low, high = map(float, interval)
    if high < .8:
        reading = "At low dose a push interacts with itself more than with a second push."
    elif low >= .8 and high <= 1.25:
        reading = "At low dose a push interacts with a second push about as much as with itself."
    elif low > 1.25:
        reading = "At low dose a push interacts with a second push more than with itself."
    else:
        return "The product-weighted interval crosses a reading boundary; report the numbers and read nothing."
    by_id = {row["id"]: row for row in prediction_rows}
    if by_id["120.4"]["verdict"] == "holds":
        return reading + " This also covers the matched random second pushes under 120.4."
    return reading + " Whether the same applies to matched random second pushes is not established by 120.4."


def score_data(vectors: dict, selected_rows: list[dict], units: list[dict],
               reference_rows: list[dict], gate1_record: dict | None,
               gate3_record: dict | None, tournament: dict | None,
               input_issues: list[dict] | None = None) -> dict:
    position_stats = position_statistics(vectors)
    row_stats = _row_statistics(position_stats, selected_rows, units)
    table, dropped = _summary_table(row_stats)
    bootstrap = unit_bootstrap_interval(row_stats, units)
    unit_values = _unit_values(row_stats, units)
    gate1 = gate1_result(gate1_record, [str(unit["id"]) for unit in units])
    gate2 = gate2_result(vectors, reference_rows, selected_rows, input_issues)
    gate3 = gate3_result(gate3_record, [str(unit["id"]) for unit in units])
    gates = {"gate1": gate1, "gate2": gate2, "gate3": gate3,
             "all_passed": gate1["passed"] and gate2["passed"] and gate3["passed"]}
    pred_rows = predictions(table, row_stats, unit_values, units, tournament, bootstrap)
    failed_gate = next((number for number in (1, 2, 3)
                        if not gates[f"gate{number}"]["passed"]), None)
    if failed_gate is not None:
        for row in pred_rows:
            row["verdict"] = f"not read (gate {failed_gate})"
    return {"part": "120", "selected_rows_expected": len(selected_rows),
            "selected_rows_by_class": {kind: sum(row["kind"] == kind for row in selected_rows)
                                       for kind in KINDS},
            "model_dtype": "float32", "model_parameter_dtype_required": "torch.float32",
            "gates": gates, "per_dose": table, "positions_dropped_by_dose": dropped,
            "float32_floor_ratios": _floor_ratios(table),
            "bootstrap_c_0.1_real": bootstrap,
            "u_by_unit": unit_values,
            "exploratory": _exploratory(position_stats, row_stats, units),
            "predictions": pred_rows,
            "headline": _headline(pred_rows, table, bootstrap, gates),
            "bootstrap_definition": ("Part 119c's unit bootstrap draws (2,000 draws, seed 12001); "
                                     "each draw uses product weights counts[first]*counts[second] as primary "
                                     "and sum weights counts[first]+counts[second] as sensitivity. The "
                                     "reported statistic is the weighted median row c."),
            "position_rule": ("At each alpha with a 2 alpha pass, a position is eligible when norms of P, "
                              "S_A, and S_B are all above zero. Row values are medians over eligible "
                              "positions; class values are medians over rows. Dose 0.8 has no 1.6 pass, "
                              "so only its raw interaction norm is reported."),
            "gate2_rule": ("For each statistic and dose, compute relative error per matched position, "
                           "take each row's median, then the median across selected rows; every cell must "
                           "be complete and strictly below 0.5%.")}


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(result: dict) -> str:
    gates = result["gates"]
    lines = ["# Part 120 score", "",
             f"Selected Path 2 rows: {result['selected_rows_expected']} "
             f"({result['selected_rows_by_class']}).", "", "## Gates", "",
             "| Gate | Passed | Detail |", "| --- | --- | --- |"]
    for number in (1, 2, 3):
        gate = gates[f"gate{number}"]
        if number == 1:
            detail = f"{gate['paths_checked']}/{gate['expected_paths']} paths"
        elif number == 2:
            detail = f"{gate['input_issue_count']} input issues; {gate['comparison_issue_count']} comparison issues"
        else:
            detail = f"max relative error {_fmt(gate['maximum_relative_error'])}"
        lines.append(f"| gate {number} | {gate['passed']} | {detail} |")
    lines.extend(["", "## Predictions", "",
                  "| Prediction | Verdict | Observed |", "| --- | --- | --- |"])
    for prediction in result["predictions"]:
        observed = json.dumps(prediction["observed"], ensure_ascii=False,
                              sort_keys=True, allow_nan=False)
        lines.append(f"| {prediction['id']} | {prediction['verdict']} | `{observed}` |")
    if not gates["all_passed"]:
        failed_gate = next(number for number in (1, 2, 3) if not gates[f"gate{number}"]["passed"])
        heading = f"## Per-dose summaries (nothing is read from these tables because gate {failed_gate} failed)"
    else:
        heading = "## Per-dose summaries"
    lines.extend(["", heading, "", result["position_rule"], "",
                  "| Dose | Class | Rows | Eligible positions | Dropped positions | Median row ‖I‖ | Median row ‖S_A‖ | Median row ‖S_B‖ | Median row ‖S_P‖ | Median row ‖E‖ | Median c | Median q | Pair share | Self share |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in result["per_dose"]:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in (
            "dose", "kind", "rows", "positions_eligible", "positions_dropped",
            "median_row_norm_I", "median_row_norm_SA", "median_row_norm_SB",
            "median_row_norm_SP", "median_row_norm_E", "median_row_c", "median_row_q",
            "median_row_pair_share", "median_row_self_share")) + " |")
    lines.extend(["", "## Float32 floor", "",
                  "Ratios are the median row norm at the higher dose divided by the median row norm at the lower dose.",
                  "", "| Class | Statistic | From | To | Higher / lower |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for row in result["float32_floor_ratios"]:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in (
            "kind", "statistic", "from_dose", "to_dose", "higher_over_lower_norm_ratio")) + " |")
    lines.extend(["", "## Exploratory", "",
                  "| Dose | Class | cos(I,S_A) | cos(I,S_B) | Same-layer c | Different-layer c |",
                  "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for row in result["exploratory"]:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in (
            "dose", "kind", "median_row_cosine_I_SA", "median_row_cosine_I_SB",
            "same_layer_median_row_c", "different_layer_median_row_c")) + " |")
    bootstrap = result["bootstrap_c_0.1_real"]
    lines.extend(["", "## Bootstrap", "",
                  f"Real-row c at 0.1: point median {_fmt(bootstrap['point_median'])}; "
                  f"product-weighted 95% interval {bootstrap['ci95_product_weighted']}; "
                  f"sum-weighted sensitivity interval {bootstrap['ci95_sum_weighted']}.",
                  f"{result['bootstrap_definition']}", "", "## Headline", "",
                  result["headline"], "", result["gate2_rule"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vectors", default="results/part120/vectors")
    parser.add_argument("--reference", default="results/part119c/interaction-float32.jsonl")
    parser.add_argument("--gate1", default="results/part120/gate1.json")
    parser.add_argument("--gate3", default="results/part120/gate3.json")
    parser.add_argument("--tournament", default="results/part119/tournament.json")
    parser.add_argument("--out", default="results/part120")
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    args = parser.parse_args(argv)
    units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    selected_rows = rows119c.draw_path2_rows(units)
    vectors, issues = load_vectors(args.vectors, units)
    result = score_data(vectors, selected_rows, units,
                        read_jsonl(args.reference, optional=True),
                        read_json(args.gate1, optional=True), read_json(args.gate3, optional=True),
                        read_json(args.tournament, optional=True), issues)
    output_dir = repo_path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score120.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output_dir / "SCORE.md").write_text(render_markdown(result), encoding="utf-8")
    print(result["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
