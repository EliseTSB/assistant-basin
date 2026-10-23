#!/usr/bin/env python3
"""Apply Part 59B's pooled symmetric rule independently at every Part 61 write size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# These registered thresholds and dimensions are pinned by test_part61.py.
WRITE_SIZES = (0.25, 0.5, 1.0, 2.0)  # Four write sizes in the Part 61 design.
SIGNS = (-1.0, 1.0)  # Symmetric negative/positive clamp conditions.
CONTROL_COUNT = 10  # Ten same-layer controls are compared with the target.
CONTROL_PERCENTILE = 99.0  # Pooled control-by-neuron percentile within one write size.
NULL_PASS_PROBABILITY = 0.00005  # Registered chance probability; 59B uses no absolute-change floor.
SPEARMAN_MINIMUM = 0.7  # Registered lower bound for prediction 61a.
RESIDUAL_RATIO_MINIMUM = 0.9  # Registered lower bound for prediction 61d.
RESIDUAL_RATIO_MAXIMUM = 1.1  # Registered upper bound for prediction 61d.


def _as_matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional control-by-neuron matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _symmetric_mask(minus, plus, control_minus, control_plus,
                    percentile: float = CONTROL_PERCENTILE) -> tuple[np.ndarray, float, float]:
    minus = np.asarray(minus, dtype=np.float64)
    plus = np.asarray(plus, dtype=np.float64)
    control_minus = _as_matrix(control_minus, "control_minus")
    control_plus = _as_matrix(control_plus, "control_plus")
    if minus.ndim != 1 or plus.shape != minus.shape:
        raise ValueError("target dose changes must be matching one-dimensional arrays")
    if control_minus.shape != control_plus.shape or control_minus.shape[1] != len(minus):
        raise ValueError("control changes must match the target neuron count at both signs")
    if not (0 < percentile < 100):
        raise ValueError("percentile must be between 0 and 100")
    if not np.isfinite(minus).all() or not np.isfinite(plus).all():
        raise ValueError("target changes contain non-finite values")
    threshold_minus = float(np.percentile(np.abs(control_minus).ravel(), percentile))
    threshold_plus = float(np.percentile(np.abs(control_plus).ravel(), percentile))
    same_nonzero_sign = (np.sign(minus) == np.sign(plus)) & (np.sign(minus) != 0)
    symmetric = (same_nonzero_sign
                 & (np.abs(minus) > threshold_minus)
                 & (np.abs(plus) > threshold_plus))
    return symmetric, threshold_minus, threshold_plus


def classify_size_changes(changes, percentile: float = CONTROL_PERCENTILE) -> dict:
    """Classify target and controls for one size, shaped [target+controls, signs, neurons]."""
    changes = np.asarray(changes, dtype=np.float64)
    if changes.ndim != 3 or changes.shape[0] < 3 or changes.shape[1] != len(SIGNS):
        raise ValueError("changes must have shape [target + >=2 controls, 2 signs, neurons]")
    if not np.isfinite(changes).all():
        raise ValueError("changes contain non-finite values")
    target_minus, target_plus = changes[0, 0], changes[0, 1]
    control_minus, control_plus = changes[1:, 0], changes[1:, 1]
    symmetric, threshold_minus, threshold_plus = _symmetric_mask(
        target_minus, target_plus, control_minus, control_plus, percentile)
    odd = (np.sign(target_minus) * np.sign(target_plus)) < 0
    control_counts = []
    control_thresholds = []
    for index in range(len(control_minus)):
        reference_indices = [other for other in range(len(control_minus)) if other != index]
        pseudo_symmetric, pseudo_minus, pseudo_plus = _symmetric_mask(
            control_minus[index], control_plus[index], control_minus[reference_indices],
            control_plus[reference_indices], percentile)
        control_counts.append(int(pseudo_symmetric.sum()))
        control_thresholds.append({"minus": pseudo_minus, "plus": pseudo_plus})
    return {
        "symmetric_mask": symmetric,
        "odd_mask": odd,
        "threshold_minus": threshold_minus,
        "threshold_plus": threshold_plus,
        "symmetric_count": int(symmetric.sum()),
        "odd_count": int(odd.sum()),
        "control_symmetric_counts": control_counts,
        "control_thresholds": control_thresholds,
        "max_control_symmetric_count": max(control_counts, default=0),
        "percentile": float(percentile),
    }


def _rankdata(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_rho(left, right) -> float:
    """Spearman correlation with average ranks for ties and zero for a constant vector."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or len(left) < 2:
        raise ValueError("Spearman inputs must be matching one-dimensional vectors of length at least two")
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _held(value: bool) -> str:
    return "held" if value else "failed"


def pooled_control_size_effect_rho(size_count_matrix, write_sizes) -> float:
    """Spearman rho across control-unit counts and write sizes, excluding the target."""
    counts = np.asarray(size_count_matrix, dtype=np.float64)
    sizes = np.asarray(write_sizes, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(sizes) or counts.shape[0] != CONTROL_COUNT + 1:
        raise ValueError("size count matrix must contain the target plus ten controls")
    control_counts = counts[1:, :]
    pooled_sizes = np.tile(sizes, CONTROL_COUNT)
    pooled_counts = control_counts.reshape(-1)
    return spearman_rho(pooled_sizes, pooled_counts)


def target_rank(target_count: int, control_counts) -> tuple[int, float]:
    """Return a conservative upper rank and exact one-sided permutation p-value.

    Controls tied with the target count against it, so rank 1 means a strict target maximum.
    """
    values = [int(value) for value in control_counts]
    if len(values) != CONTROL_COUNT:
        raise ValueError(f"expected {CONTROL_COUNT} matched controls")
    rank = 1 + sum(value >= int(target_count) for value in values)
    return rank, rank / (CONTROL_COUNT + 1)


def excess_status(rank_1x: int, rank_2x: int) -> str:
    if rank_1x == 1 and rank_2x == 1:
        return "shown"
    if rank_1x >= 3:
        return "absent"
    return "inconclusive"


def model_decision(control_size_effect_rho: float, rank_1x: int, rank_2x: int) -> str:
    """Apply Amendment 1 independently to one model."""
    status = excess_status(rank_1x, rank_2x)
    if status == "shown":
        return "59B contrast survives for this model"
    if control_size_effect_rho >= SPEARMAN_MINIMUM and status == "absent":
        return "push-size artefact supported for this model"
    return "inconclusive for this model"


def null_outcome_chance_table() -> dict:
    """Theoretical outcome rates for exchangeable, tie-free ranks under two dependence cases."""
    independent_shown = 1 / ((CONTROL_COUNT + 1) ** 2)
    correlated_shown = 1 / (CONTROL_COUNT + 1)
    absent = (CONTROL_COUNT - 1) / (CONTROL_COUNT + 1)
    return {
        "assumption": "11 exchangeable units with tie-free ranks; ties are handled conservatively",
        "independent_sizes": {
            "shown": independent_shown,
            "absent": absent,
            "inconclusive": 1.0 - independent_shown - absent,
        },
        "perfectly_correlated_sizes": {
            "shown": correlated_shown,
            "absent": absent,
            "inconclusive": 1.0 - correlated_shown - absent,
        },
    }


def _residual_ratio_rows(residual_ratios: np.ndarray, condition_units: list[int],
                         size_indices: list[int]) -> list[dict]:
    rows = []
    for unit_index, unit in enumerate(condition_units):
        for size_index in size_indices:
            rows.append({
                "unit": int(unit),
                "write_size": float(WRITE_SIZES[size_index]),
                "minus": float(residual_ratios[unit_index, size_index, 0]),
                "plus": float(residual_ratios[unit_index, size_index, 1]),
            })
    return rows


def analyze_arrays(changes, write_sizes, residual_ratios, condition_units: list[int]) -> dict:
    changes = np.asarray(changes, dtype=np.float64)
    write_sizes = np.asarray(write_sizes, dtype=np.float64)
    residual_ratios = np.asarray(residual_ratios, dtype=np.float64)
    if changes.ndim != 4:
        raise ValueError("changes must have shape [units, sizes, signs, neurons]")
    unit_count, size_count, sign_count, _ = changes.shape
    if unit_count != CONTROL_COUNT + 1 or size_count != len(WRITE_SIZES) or sign_count != len(SIGNS):
        raise ValueError("activation archive dimensions do not match the registered Part 61 design")
    if write_sizes.tolist() != list(WRITE_SIZES) or len(condition_units) != unit_count:
        raise ValueError("write-size or condition-unit metadata does not match the Part 61 design")
    if residual_ratios.shape != (unit_count, size_count, sign_count) or not np.isfinite(residual_ratios).all():
        raise ValueError("residual norm ratios do not match units, sizes and signs")

    per_size = []
    size_count_matrix = np.zeros((unit_count, size_count), dtype=np.int64)
    for size_index, write_size in enumerate(WRITE_SIZES):
        classified = classify_size_changes(changes[:, size_index, :, :])
        size_count_matrix[0, size_index] = classified["symmetric_count"]
        size_count_matrix[1:, size_index] = classified["control_symmetric_counts"]
        per_size.append({
            "write_size": float(write_size),
            "target_symmetric_count": classified["symmetric_count"],
            "control_symmetric_counts": classified["control_symmetric_counts"],
            "target_thresholds": {"minus": classified["threshold_minus"],
                                  "plus": classified["threshold_plus"]},
            "control_thresholds_leave_one_out": classified["control_thresholds"],
            "target_odd_count": classified["odd_count"],
        })

    control_rho = pooled_control_size_effect_rho(size_count_matrix, write_sizes)
    matched_index = WRITE_SIZES.index(1.0)
    two_x_index = WRITE_SIZES.index(2.0)
    matched_target = int(size_count_matrix[0, matched_index])
    matched_controls = [int(value) for value in size_count_matrix[1:, matched_index]]
    control_minimum = min(matched_controls)
    control_maximum = max(matched_controls)
    control_median = float(np.median(matched_controls))
    target_fails_to_beat_a_control = any(matched_target <= count for count in matched_controls)
    target_within_control_range = control_minimum <= matched_target <= control_maximum
    olmo_excess_over_median = matched_target > control_median
    matched_ratios = residual_ratios[:, matched_index, :]
    residual_pass_count = int(np.logical_and(
        matched_ratios >= RESIDUAL_RATIO_MINIMUM,
        matched_ratios <= RESIDUAL_RATIO_MAXIMUM).sum())
    residual_total = int(matched_ratios.size)
    residual_within_ten_percent = residual_pass_count == residual_total
    rank_rows = []
    for size_index, write_size in enumerate(WRITE_SIZES):
        counts = size_count_matrix[:, size_index]
        rank, permutation_p = target_rank(int(counts[0]), counts[1:])
        rank_rows.append({
            "write_size": float(write_size),
            "target_count": int(counts[0]),
            "control_counts": [int(value) for value in counts[1:]],
            "target_rank": rank,
            "one_sided_permutation_p": permutation_p,
        })
    rank_1x = rank_rows[matched_index]["target_rank"]
    rank_2x = rank_rows[two_x_index]["target_rank"]
    excess = excess_status(rank_1x, rank_2x)
    per_size_residual_ratios = _residual_ratio_rows(
        residual_ratios, condition_units, list(range(len(WRITE_SIZES))))
    matched_residual_rows = _residual_ratio_rows(residual_ratios, condition_units, [matched_index])
    return {
        "symmetric_counts_by_unit_and_size": size_count_matrix.tolist(),
        "per_size": per_size,
        "target_rank_by_size": rank_rows,
        "pooled_control_spearman_rho_write_size_vs_count": control_rho,
        "residual_norm_ratios_by_unit_size_and_sign": per_size_residual_ratios,
        "model_decision": {
            "excess_status": excess,
            "excess_absent_interpretation": "no evidence of excess over matched controls; not proof that excess is absent",
            "decision": model_decision(control_rho, rank_1x, rank_2x),
            "target_rank_1x": rank_1x,
            "target_rank_2x": rank_2x,
            "null_outcome_chance": null_outcome_chance_table(),
        },
        "matched_1x": {
            "target_count": matched_target,
            "control_counts": matched_controls,
            "control_minimum": control_minimum,
            "control_maximum": control_maximum,
            "control_median": control_median,
            "target_fails_to_beat_at_least_one_control": target_fails_to_beat_a_control,
            "target_within_control_range": target_within_control_range,
            "target_above_all_controls": matched_target > control_maximum,
            "target_below_all_controls": matched_target < control_minimum,
            "target_above_control_median": olmo_excess_over_median,
            "residual_norm_ratios_by_unit_and_sign": matched_ratios.tolist(),
            "residual_norm_ratio_rows": matched_residual_rows,
            "residual_ratios_within_10_percent": residual_pass_count,
            "residual_ratios_total": residual_total,
            "residual_within_10_percent_for_every_unit": residual_within_ten_percent,
        },
    }


def analyze_file(input_npz: str | Path, out_path: str | Path | None = None,
                 percentile: float = CONTROL_PERCENTILE) -> dict:
    input_npz = Path(input_npz)
    manifest_path = input_npz.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Part 61 manifest status is {manifest.get('status')!r}; expected 'complete'")
    if (manifest.get("control_percentile") != percentile
            or manifest.get("null_pass_probability") != NULL_PASS_PROBABILITY
            or manifest.get("absolute_change_floor") is not None):
        raise ValueError("analysis percentile/null probability do not match the no-floor run manifest")
    with np.load(input_npz) as archive:
        baseline = np.asarray(archive["baseline_mean"], dtype=np.float32)
        conditions = np.asarray(archive["condition_mean"], dtype=np.float32)
        changes_saved = np.asarray(archive["change"], dtype=np.float32)
        neuron_layers = np.asarray(archive["neuron_layer"], dtype=np.int32)
        neuron_units = np.asarray(archive["neuron_unit"], dtype=np.int32)
        condition_units = [int(value) for value in np.asarray(archive["condition_unit"]).tolist()]
        write_sizes = np.asarray(archive["write_sizes"], dtype=np.float32)
        signs = np.asarray(archive["signs"], dtype=np.float32)
        baseline_residual = np.asarray(archive["baseline_residual_norm_by_doc"], dtype=np.float32)
        residual_ratios = np.asarray(archive["residual_norm_ratio"], dtype=np.float32)
    expected_shape = (CONTROL_COUNT + 1, len(WRITE_SIZES), len(SIGNS), len(baseline))
    if conditions.shape != expected_shape or changes_saved.shape != expected_shape:
        raise ValueError(f"archive activation arrays have an unexpected shape: {conditions.shape}")
    if len(neuron_layers) != len(baseline) or len(neuron_units) != len(baseline):
        raise ValueError("neuron metadata length does not match activation means")
    if condition_units != [int(manifest["twin"]["unit"])] + [int(value) for value in manifest["control_units"]]:
        raise ValueError("NPZ condition_unit metadata does not match the manifest writer output")
    if signs.tolist() != list(SIGNS):
        raise ValueError(f"expected stored signs {list(SIGNS)}, got {signs.tolist()}")
    if not np.isfinite(baseline).all() or not np.isfinite(conditions).all() or not np.isfinite(changes_saved).all():
        raise ValueError("activation archive contains non-finite means")
    if baseline_residual.ndim != 1 or len(baseline_residual) != manifest["documents"] or not np.isfinite(baseline_residual).all():
        raise ValueError("baseline residual-norm document values are missing or malformed")
    changes = conditions - baseline[None, None, None, :]
    if not np.array_equal(changes, changes_saved):
        raise ValueError("stored change array does not equal condition_mean minus baseline_mean")
    result = analyze_arrays(changes, write_sizes, residual_ratios, condition_units)
    short = manifest["short_name"]
    matched = result["matched_1x"]
    predictions = {
        "61a": {"claim": "among-control Spearman rho between write-size multiplier and symmetric count is at least 0.7",
                "observed_rho": result["pooled_control_spearman_rho_write_size_vs_count"],
                "minimum": SPEARMAN_MINIMUM,
                "held": result["pooled_control_spearman_rho_write_size_vs_count"] >= SPEARMAN_MINIMUM},
        "61b": {"claim": "at matched 1x the OLMo target fails to beat at least one control",
                "applicable": short == "olmo2-1b",
                "observed_target_count": matched["target_count"],
                "observed_control_counts": matched["control_counts"],
                "observed_control_range": [matched["control_minimum"], matched["control_maximum"]],
                "observed_target_within_control_range": matched["target_within_control_range"],
                "chance_base_rate": 10 / 11,
                "decision_input": False,
                "held": matched["target_fails_to_beat_at_least_one_control"] if short == "olmo2-1b" else None},
        "61c": {"claim": "at matched 1x the Qwen target lies within the matched control count range",
                "applicable": short == "qwen35-4b",
                "observed_target_count": matched["target_count"],
                "observed_control_range": [matched["control_minimum"], matched["control_maximum"]],
                "held": matched["target_within_control_range"] if short == "qwen35-4b" else None},
        "61d": {"claim": "each matched-1x sign ratio for every unit is within [0.9, 1.1]",
                "observed_within_band": matched["residual_ratios_within_10_percent"],
                "observed_total": matched["residual_ratios_total"],
                "observed_per_unit_sign_ratios": matched["residual_norm_ratio_rows"],
                "held": matched["residual_within_10_percent_for_every_unit"]},
        "61e": {"claim": "at matched 1x the OLMo target count exceeds the median matched-control count",
                "applicable": short == "olmo2-1b",
                "observed_target_count": matched["target_count"],
                "observed_control_median": matched["control_median"],
                "chance_base_rate": 0.5,
                "decision_input": False,
                "held": matched["target_above_control_median"] if short == "olmo2-1b" else None},
    }
    output = {
        "tool": "analyze61.py", "model": manifest["model"], "revision": manifest["revision"],
        "short_name": short, "twin": manifest["twin"], "archive": str(input_npz.resolve()),
        "classification": "SYMMETRIC requires equal nonzero signs and absolute change above the pooled control P99, with no absolute-change floor; each control is evaluated against a pool leaving that control out, independently at each write size. ODD requires opposite signs.",
        "control_percentile": percentile, "null_pass_probability": NULL_PASS_PROBABILITY,
        "absolute_change_floor": None,
        "later_neuron_count": int(len(baseline)), "control_units": manifest["control_units"],
        "candidate_pool": manifest["candidate_pool"],
        "target_pile_maximum_source_dtype": manifest.get("target_pile_maximum_source_dtype"),
        "comparison_scope": manifest.get("comparison_scope", {
            "counts": "within-run comparisons only",
            "part59b": "not comparable: Part 61 uses 200 documents while Part 59B used 1,000",
        }),
        "per_size_results": result["per_size"],
        "symmetric_counts_by_unit_and_size": result["symmetric_counts_by_unit_and_size"],
        "target_rank_by_size": result["target_rank_by_size"],
        "pooled_control_spearman_rho_write_size_vs_count": result["pooled_control_spearman_rho_write_size_vs_count"],
        "residual_norm_ratios_by_unit_size_and_sign": result["residual_norm_ratios_by_unit_size_and_sign"],
        "model_decision": result["model_decision"],
        "matched_1x": matched, "predictions": predictions,
        "thresholds": {"pooling": "all control-by-neuron absolute changes for this write size and sign",
                       "target_reference_controls": "all ten controls",
                       "control_reference_controls": "leave one control out",
                       "percentile_interpolation": "numpy default linear interpolation",
                       "residual_norm_aggregation": "mean of 200 per-document clamped/unclamped ratios for each sign",
                       "target_rank_ties": "controls tied with the target count against it; permutation p equals rank divided by 11",
                       "61a_pool": "the ten matched controls only; target excluded"},
    }
    if out_path is None:
        out_path = input_npz.with_name(input_npz.stem + "-analysis.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"61 {short}: among_control_size_rho={result['pooled_control_spearman_rho_write_size_vs_count']:.6f} "
          f"matched_1x_target={matched['target_count']} controls={matched['control_counts']}")
    print("61 null outcome chance (independent sizes): "
          f"{result['model_decision']['null_outcome_chance']['independent_sizes']}")
    print("61 null outcome chance (perfectly correlated sizes): "
          f"{result['model_decision']['null_outcome_chance']['perfectly_correlated_sizes']}")
    print("61 prediction base rates: 61b=10/11 (about 90.9%); 61e=1/2 (50%); neither is a decision input")
    print(f"61 model decision: {result['model_decision']['decision']} "
          f"(excess {result['model_decision']['excess_status']}, "
          f"ranks 1x={result['model_decision']['target_rank_1x']}/11, "
          f"2x={result['model_decision']['target_rank_2x']}/11)")
    for row in result["target_rank_by_size"]:
        print(f"rank {short} {row['write_size']:g}x: target={row['target_count']} controls={row['control_counts']} "
              f"rank={row['target_rank']}/11 permutation_p={row['one_sided_permutation_p']:.6f}")
    for row in result["residual_norm_ratios_by_unit_size_and_sign"]:
        print(f"residual {short} unit={row['unit']} size={row['write_size']:g}x "
              f"minus={row['minus']:.6f} plus={row['plus']:.6f}")
    for evaluation, row in predictions.items():
        if row["held"] is None:
            continue
        if evaluation == "61a":
            numbers = f"rho={row['observed_rho']:.6f} minimum={row['minimum']:.1f}"
        elif evaluation == "61d":
            numbers = f"within_band={row['observed_within_band']}/{row['observed_total']} ratios"
        elif evaluation == "61c":
            numbers = f"target={row['observed_target_count']} control_range={row['observed_control_range']}"
        elif evaluation == "61b":
            numbers = f"target={row['observed_target_count']} controls={row['observed_control_counts']}"
        else:
            numbers = f"target={row['observed_target_count']} control_median={row['observed_control_median']:.3f}"
        print(f"{evaluation} {short}: {numbers} -> {_held(bool(row['held']))}")
    return output


def analyze_pair(qwen_archive: str | Path, olmo_archive: str | Path,
                 out_path: str | Path | None = None) -> dict:
    """Report the Amendment 1 per-model decisions and their combined summary."""
    qwen = analyze_file(qwen_archive)
    olmo = analyze_file(olmo_archive)
    if qwen["short_name"] != "qwen35-4b" or olmo["short_name"] != "olmo2-1b":
        raise ValueError("combined analysis requires Qwen35-4B followed by OLMo2-1B archives")
    qwen_matched = qwen["matched_1x"]
    olmo_matched = olmo["matched_1x"]
    rho_qwen = qwen["pooled_control_spearman_rho_write_size_vs_count"]
    rho_olmo = olmo["pooled_control_spearman_rho_write_size_vs_count"]
    predictions = {
        "61a": {"claim": "among-control symmetric count rises with write size: Spearman rho >= 0.7 in each model",
                "observed_rho": {"qwen35-4b": rho_qwen, "olmo2-1b": rho_olmo},
                "minimum": SPEARMAN_MINIMUM,
                "held": rho_qwen >= SPEARMAN_MINIMUM and rho_olmo >= SPEARMAN_MINIMUM},
        "61b": {"claim": "at matched 1x the OLMo target fails to beat at least one control",
                "observed_target_count": olmo_matched["target_count"],
                "observed_control_counts": olmo_matched["control_counts"],
                "observed_target_within_control_range": olmo_matched["target_within_control_range"],
                "chance_base_rate": 10 / 11,
                "decision_input": False,
                "held": olmo_matched["target_fails_to_beat_at_least_one_control"]},
        "61c": {"claim": "at matched 1x the Qwen target is within the control range",
                "observed_target_count": qwen_matched["target_count"],
                "observed_control_range": [qwen_matched["control_minimum"], qwen_matched["control_maximum"]],
                "held": qwen_matched["target_within_control_range"]},
        "61d": {"claim": "each matched-1x sign ratio for every unit in both models is within [0.9, 1.1]",
                "observed_within_band": {
                    "qwen35-4b": qwen_matched["residual_ratios_within_10_percent"],
                    "olmo2-1b": olmo_matched["residual_ratios_within_10_percent"]},
                "observed_total": {
                    "qwen35-4b": qwen_matched["residual_ratios_total"],
                    "olmo2-1b": olmo_matched["residual_ratios_total"]},
                "observed_per_unit_sign_ratios": {
                    "qwen35-4b": qwen_matched["residual_norm_ratio_rows"],
                    "olmo2-1b": olmo_matched["residual_norm_ratio_rows"]},
                "held": (qwen_matched["residual_within_10_percent_for_every_unit"]
                         and olmo_matched["residual_within_10_percent_for_every_unit"])},
        "61e": {"claim": "at matched 1x the OLMo target count exceeds the median matched-control count",
                "observed_target_count": olmo_matched["target_count"],
                "observed_control_median": olmo_matched["control_median"],
                "chance_base_rate": 0.5,
                "decision_input": False,
                "held": olmo_matched["target_above_control_median"]},
    }
    decisions = {"qwen35-4b": qwen["model_decision"], "olmo2-1b": olmo["model_decision"]}
    decision_summary = (
        f"Qwen: {decisions['qwen35-4b']['decision']}; "
        f"OLMo: {decisions['olmo2-1b']['decision']}"
    )
    output = {"tool": "analyze61.py --combined", "archives": [qwen["archive"], olmo["archive"]],
              "predictions": predictions, "decisions_by_model": decisions,
              "decision_summary": decision_summary,
              "null_outcome_chance": null_outcome_chance_table(),
              "comparison_scope": qwen.get("comparison_scope")}
    if out_path is None:
        out_path = Path(qwen_archive).with_name("part61-registered-evaluation.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in predictions.items():
        if name == "61a":
            numbers = f"Qwen rho={rho_qwen:.6f}, OLMo rho={rho_olmo:.6f}, minimum={SPEARMAN_MINIMUM:.1f}"
        elif name == "61b":
            numbers = f"OLMo target={olmo_matched['target_count']} controls={olmo_matched['control_counts']}"
        elif name == "61c":
            numbers = f"Qwen target={qwen_matched['target_count']} control_range=[{qwen_matched['control_minimum']}, {qwen_matched['control_maximum']}]"
        elif name == "61d":
            numbers = (f"Qwen {qwen_matched['residual_ratios_within_10_percent']}/{qwen_matched['residual_ratios_total']}, "
                       f"OLMo {olmo_matched['residual_ratios_within_10_percent']}/{olmo_matched['residual_ratios_total']} ratios in band")
        else:
            numbers = f"OLMo target={olmo_matched['target_count']} control_median={olmo_matched['control_median']:.3f}"
        print(f"{name}: {numbers} -> {_held(bool(row['held']))}")
    print(f"per-model decisions: {decision_summary}")
    return output


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", nargs="+", help="one archive, or Qwen then OLMo with --combined")
    ap.add_argument("--combined", action="store_true", help="also summarize the independent per-model decisions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--percentile", type=float, default=CONTROL_PERCENTILE)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if args.combined:
        if len(args.archive) != 2:
            raise SystemExit("--combined requires a Qwen archive followed by an OLMo archive")
        analyze_pair(args.archive[0], args.archive[1], args.out)
    else:
        if len(args.archive) != 1:
            raise SystemExit("individual analysis accepts one archive")
        analyze_file(args.archive[0], args.out, percentile=args.percentile)
