#!/usr/bin/env python3
"""Run Part 197's fixed linear readouts and registered gates."""
from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part196 import analyze196 as p196  # noqa: E402
from part196.select196 import STYLES  # noqa: E402
from part197.digest197 import compute_records_digest  # noqa: E402

BOOTSTRAPS = 2000
SEED = 197
N_LAYERS = 24
WIDTH = 2048
NUISANCE_NAMES = (
    "voiced_fraction", "f0_median", "f0_interquartile_range", "rms_mean",
    "rms_spread", "spectral_centroid_mean", "seconds_per_word", "duration_seconds",
)
EXPECTED_IDENTITY_KEYS = ("manifest_sha256", "model_snapshot_revision", "weight_sha256")
REGISTERED_IDENTITY = {
    "manifest_sha256": "2dded6313c8cdd218846e3d8e7444af6fbd9a255e28026eaa2906f2519a75503",
    "model_snapshot_revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
    "weight_sha256": "ecda7d0ffee04ff56b8f5e33eb4392b3b33ed475c0da196ef7845304c4c2e1b7",
}
READOUT_PAIRS = {
    "EN_minus_N": ("EN", "N"),
    "R_minus_N": ("R", "N"),
    "EN_minus_E0": ("EN", "E0"),
    "E0_minus_K": ("E0", "K"),
}


@dataclass(frozen=True)
class ResidualizationMap:
    nuisance_mean: np.ndarray
    nuisance_scale: np.ndarray
    keep_columns: np.ndarray
    coefficients: np.ndarray
    rank: int
    dropped_columns: tuple[int, ...]


def fit_residualization(nuisance_train: np.ndarray, x_train: np.ndarray) -> ResidualizationMap:
    """Fit the registered training-only standardized nuisance-to-unit OLS map."""
    n = np.asarray(nuisance_train, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    if n.ndim != 2 or x.ndim != 2 or len(n) != len(x) or not len(n):
        raise ValueError("residualization expects nonempty 2-D train arrays with matching rows")
    if not np.isfinite(n).all() or not np.isfinite(x).all():
        raise ValueError("residualization inputs must be finite")
    means = n.mean(axis=0)
    scales = n.std(axis=0)
    keep = scales > 0.0
    standardized = np.zeros_like(n)
    standardized[:, keep] = (n[:, keep] - means[keep]) / scales[keep]
    design = np.column_stack((np.ones(len(n), dtype=np.float64), standardized[:, keep]))
    coefficients, _residual_sum, rank, _singular_values = np.linalg.lstsq(design, x, rcond=None)
    dropped = tuple(int(i) for i in np.flatnonzero(~keep))
    return ResidualizationMap(means, scales, keep, coefficients, int(rank), dropped)


def transform_residuals(residual_map: ResidualizationMap,
                        nuisance: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply a saved training map to rows without refitting any quantity."""
    n = np.asarray(nuisance, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    if n.ndim != 2 or values.ndim != 2 or len(n) != len(values):
        raise ValueError("residual transform expects matching 2-D arrays")
    standardized = np.zeros_like(n)
    keep = residual_map.keep_columns
    standardized[:, keep] = ((n[:, keep] - residual_map.nuisance_mean[keep]) /
                             residual_map.nuisance_scale[keep])
    design = np.column_stack((np.ones(len(n), dtype=np.float64), standardized[:, keep]))
    return values - design @ residual_map.coefficients


def apply_near_zero_rule(x_train: np.ndarray, residual_train: np.ndarray,
                         residual_test: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero cancellation-noise columns before the probe's StandardScaler is fitted."""
    x = np.asarray(x_train, dtype=np.float64)
    train = np.asarray(residual_train, dtype=np.float64).copy()
    test = np.asarray(residual_test, dtype=np.float64).copy()
    if x.ndim != 2 or train.ndim != 2 or test.ndim != 2 or x.shape[1] != train.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError("near-zero rule expects 2-D arrays with the same feature count")
    reference_sd = x.std(axis=0)
    residual_sd = train.std(axis=0)
    # Design r3 (Alden #5): a unit constant in X is masked too; a strict '<' against a zero threshold never fires.
    zeroed = (reference_sd == 0.0) | (residual_sd <= (1e-6 * reference_sd))
    train[:, zeroed] = 0.0
    test[:, zeroed] = 0.0
    return train, test, zeroed


def residualize_train_test(nuisance_train: np.ndarray, nuisance_test: np.ndarray,
                           x_train: np.ndarray, x_test: np.ndarray
                           ) -> tuple[ResidualizationMap, np.ndarray, np.ndarray, np.ndarray]:
    residual_map = fit_residualization(nuisance_train, x_train)
    residual_train = transform_residuals(residual_map, nuisance_train, x_train)
    residual_test = transform_residuals(residual_map, nuisance_test, x_test)
    residual_train, residual_test, zeroed = apply_near_zero_rule(
        x_train, residual_train, residual_test)
    return residual_map, residual_train, residual_test, zeroed


def gate2_check(nuisance_train_standardized: np.ndarray, residual_train: np.ndarray,
                rank: int, *, tolerance: float = 1e-8
                ) -> dict[str, Any]:
    """Check the normalized least-squares normal-equation cross products."""
    nuisance = np.asarray(nuisance_train_standardized, dtype=np.float64)
    residual = np.asarray(residual_train, dtype=np.float64)
    if nuisance.ndim != 2 or residual.ndim != 2 or len(nuisance) != len(residual):
        raise ValueError("Gate 2 expects matching 2-D nuisance and residual matrices")
    centered = nuisance - nuisance.mean(axis=0, keepdims=True)
    n_norm = np.linalg.norm(centered, axis=0)
    r_norm = np.linalg.norm(residual, axis=0)
    products = np.abs(centered.T @ residual) / (n_norm[:, None] * r_norm[None, :] + 1e-300)
    maximum = float(np.max(products)) if products.size else 0.0
    rank_passed = int(rank) == 9
    orthogonality_passed = maximum <= tolerance
    return {"verdict": "passed" if rank_passed and orthogonality_passed else "failed",
            "rank": int(rank), "required_rank": 9, "rank_passed": rank_passed,
            "max_normalized_cross_product": maximum, "tolerance": tolerance,
            "orthogonality_passed": orthogonality_passed}


def compare_json_values(expected: Any, actual: Any, *, tolerance: float = 1e-9
                        ) -> dict[str, Any]:
    """Compare trees exactly except for numeric leaves within absolute tolerance."""
    mismatches: list[dict[str, Any]] = []
    numeric_differences: list[tuple[float, str]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                mismatches.append({"path": path, "expected": left, "actual": right,
                                   "kind": "discrete"})
            return
        if isinstance(left, int) or isinstance(right, int):
            if type(left) is not type(right) or left != right:
                mismatches.append({"path": path, "expected": left, "actual": right,
                                   "kind": "discrete"})
            return
        if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
            difference = abs(float(left) - float(right))
            numeric_differences.append((difference, path))
            if not math.isfinite(difference) or difference > tolerance:
                mismatches.append({"path": path, "expected": left, "actual": right,
                                   "kind": "numeric", "absolute_difference": difference})
            return
        if isinstance(left, dict) and isinstance(right, dict):
            left_keys, right_keys = set(left), set(right)
            if left_keys != right_keys:
                mismatches.append({"path": path, "expected_keys": sorted(left_keys),
                                   "actual_keys": sorted(right_keys), "kind": "discrete"})
            for key in sorted(left_keys & right_keys, key=str):
                walk(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append({"path": path, "expected_length": len(left),
                                   "actual_length": len(right), "kind": "discrete"})
            for index, (l_value, r_value) in enumerate(zip(left, right)):
                walk(l_value, r_value, f"{path}[{index}]")
            return
        if type(left) is not type(right) or left != right:
            mismatches.append({"path": path, "expected": left, "actual": right,
                               "kind": "discrete"})

    walk(expected, actual, "$" )
    largest = max(numeric_differences, default=(0.0, None), key=lambda item: item[0])
    return {"passed": not mismatches, "tolerance": tolerance,
            "mismatch_count": len(mismatches), "mismatches": mismatches[:50],
            "largest_numeric_difference": largest[0],
            "largest_numeric_difference_path": largest[1]}


def check_identity(work_dir: Path, manifest_rows: list[dict[str, str]],
                   expected_path: Path) -> dict[str, Any]:
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        actual = json.loads((work_dir / "identity.json").read_text(encoding="utf-8"))
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise ValueError("identity files must contain JSON objects")
        missing_expected = [key for key in EXPECTED_IDENTITY_KEYS if key not in expected]
        missing_actual = [key for key in EXPECTED_IDENTITY_KEYS if key not in actual]
        if missing_expected or missing_actual:
            raise ValueError(f"identity fields missing: expected={missing_expected}, actual={missing_actual}")
        mismatches = {key: {"expected": expected[key], "actual": actual[key]}
                      for key in EXPECTED_IDENTITY_KEYS if expected[key] != actual[key]}
        for key in EXPECTED_IDENTITY_KEYS:
            if expected[key] != REGISTERED_IDENTITY[key]:
                mismatches[f"registered_{key}"] = {
                    "expected": REGISTERED_IDENTITY[key], "actual": expected[key]}
        if actual.get("model_id") != p196.MODEL_ID:
            mismatches["producer_model_id"] = {
                "expected": p196.MODEL_ID, "actual": actual.get("model_id")}
        manifest_digest = p196.manifest_sha(manifest_rows)
        if expected["manifest_sha256"] != manifest_digest:
            mismatches["manifest_sha256_for_input_manifest"] = {
                "expected": expected["manifest_sha256"], "actual": manifest_digest}
        if actual.get("model_snapshot_revision") != p196.MODEL_REVISION:
            mismatches["producer_model_snapshot_revision"] = {
                "expected": p196.MODEL_REVISION,
                "actual": actual.get("model_snapshot_revision")}
        return {"verdict": "passed" if not mismatches else "failed",
                "mismatches": mismatches, "expected": {key: expected[key]
                                                           for key in EXPECTED_IDENTITY_KEYS}}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"verdict": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def check_digest(work_dir: Path, manifest_path: Path, expected_path: Path) -> dict[str, Any]:
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        if not isinstance(expected, dict):
            raise ValueError("records digest file must contain a JSON object")
        actual = compute_records_digest(work_dir, manifest_path)
        passed = expected == actual
        return {"verdict": "passed" if passed else "failed",
                "expected_records_sha256": expected.get("records_sha256"),
                "actual_records_sha256": actual["records_sha256"],
                "expected_record_count": len(expected.get("records", [])),
                "actual_record_count": len(actual["records"]),
                "reason": None if passed else "records digest differs"}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"verdict": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def gate1_regeneration(*, manifest_path: Path, work_dir: Path,
                       part196_results_path: Path) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="part197-gate1-") as temp_name:
            temp_out = Path(temp_name)
            p196.analyze(manifest_path=manifest_path, work_dir=work_dir, out_dir=temp_out)
            regenerated_path = temp_out / "results.json"
            if not regenerated_path.is_file():
                raise FileNotFoundError("Part 196 analyze() did not write results.json")
            expected = json.loads(part196_results_path.read_text(encoding="utf-8"))
            actual = json.loads(regenerated_path.read_text(encoding="utf-8"))
        comparison = compare_json_values(expected, actual, tolerance=1e-9)
        print(json.dumps({"largest_numeric_difference": comparison["largest_numeric_difference"],
                          "path": comparison["largest_numeric_difference_path"]}, sort_keys=True),
              file=sys.stderr)
        return {"verdict": "passed" if comparison["passed"] else "failed",
                "comparison": comparison}
    except Exception as exc:
        return {"verdict": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _write_result(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(p196._jsonable(result), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _gate_summary(result: dict[str, Any]) -> dict[str, str]:
    return {"gate1": result.get("gate1", {}).get("verdict", "not_run"),
            "gate2": result.get("gate2", {}).get("verdict", "not_run")}


def _early_gate1_result(out_dir: Path, gate1: dict[str, Any]) -> dict[str, Any]:
    result = {"study": "part197", "status": "gates_failed", "gate1": gate1,
              "gate2": {"verdict": "not_run", "reason": "Gate 1 did not pass"},
              "readouts": {}, "paired_differences": {}, "shuffled_label_controls": {},
              "predictions": {}}
    _write_result(out_dir, result)
    print(json.dumps(_gate_summary(result), sort_keys=True))
    return result


def _standardized_nuisance(residual_map: ResidualizationMap,
                           nuisance_train: np.ndarray) -> np.ndarray:
    n = np.asarray(nuisance_train, dtype=np.float64)
    z = np.zeros_like(n)
    keep = residual_map.keep_columns
    z[:, keep] = ((n[:, keep] - residual_map.nuisance_mean[keep]) /
                  residual_map.nuisance_scale[keep])
    return z


def _active_indices(rows: list[dict[str, str]], record_status: np.ndarray,
                    duration_from_records: np.ndarray, work_dir: Path) -> np.ndarray:
    # Review F1 (2026-11-08): no fallback. A missing or empty durations.json would silently re-derive a different
    # active set while every Gate 1 check passed; the file is now bound by the records digest and required here.
    durations_path = work_dir / "durations.json"
    if not durations_path.is_file():
        raise FileNotFoundError(f"durations.json is required and missing: {durations_path}")
    durations = json.loads(durations_path.read_text(encoding="utf-8"))
    included_ids = {str(clip_id) for clip_id, item in durations.items()
                    if item.get("status") == "included"}
    if not included_ids:
        raise ValueError("durations.json has no included clip")
    return np.asarray([i for i, row in enumerate(rows)
                       if row["id"] in included_ids and record_status[i] == "ok"], dtype=int)


def _fold_indices(rows: list[dict[str, str]], active: np.ndarray
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    speakers = sorted({rows[index]["speaker_id"] for index in active})
    return {heldout: (np.asarray([i for i in active if rows[i]["speaker_id"] != heldout], dtype=int),
                      np.asarray([i for i in active if rows[i]["speaker_id"] == heldout], dtype=int))
            for heldout in speakers}


def _prepare_gate2(folds: dict[str, tuple[np.ndarray, np.ndarray]], nuisance: np.ndarray,
                   activations: np.ndarray) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fold_results: dict[str, Any] = {}
    fold_residuals: dict[str, dict[str, Any]] = {}
    for heldout, (train, test) in folds.items():
        n_train, n_test = nuisance[train], nuisance[test]
        x_train = activations[train, :WIDTH].astype(np.float64)
        x_test = activations[test, :WIDTH].astype(np.float64)
        residual_map, residual_train, residual_test, zeroed = residualize_train_test(
            n_train, n_test, x_train, x_test)
        standardized_n = _standardized_nuisance(residual_map, n_train)
        gate = gate2_check(standardized_n, residual_train, residual_map.rank)
        gate["dropped_nuisance_columns"] = [NUISANCE_NAMES[i]
                                             for i in residual_map.dropped_columns]
        gate["near_zero_residual_columns_zeroed"] = int(np.sum(zeroed))
        gate["near_zero_residual_column_indices"] = [int(i) for i in np.flatnonzero(zeroed)]
        gate["fold"] = heldout
        fold_results[heldout] = gate
        fold_residuals[heldout] = {"train": residual_train, "test": residual_test,
                                   "map": residual_map}
    passed = bool(fold_results) and all(item["verdict"] == "passed"
                                        for item in fold_results.values())
    return {"verdict": "passed" if passed else "failed", "tolerance": 1e-8,
            "required_rank": 9, "folds": fold_results}, fold_residuals


def _fold_metric(labels: np.ndarray, prediction: np.ndarray) -> float | None:
    return p196.balanced_accuracy(labels, prediction)


def _alias_bootstrap_view(fold_outputs: dict[str, dict[str, Any]],
                          alias_to_readout: dict[str, str]) -> dict[str, dict[str, Any]]:
    result = {}
    for fold, data in fold_outputs.items():
        result[fold] = {"labels": data["labels"], "texts": data["texts"],
                        "predictions": {alias: data["predictions"][name]
                                        for alias, name in alias_to_readout.items()},
                        "scores": {alias: data["scores"][name]
                                   for alias, name in alias_to_readout.items()}}
    return result


def _bootstrap_results(fold_outputs: dict[str, dict[str, Any]], gate2_passed: bool
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    intervals: dict[str, Any] = {}
    pair_intervals: dict[str, Any] = {}

    def collect(aliases: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        return p196._metric_bootstrap_with_pairs(
            _alias_bootstrap_view(fold_outputs, aliases), n_boot=BOOTSTRAPS, seed=SEED)

    methods = list(fold_outputs[next(iter(fold_outputs))]["predictions"])
    common = {"D": "EN", "B": "N", "A": "E0", "B56": "K"}
    common.update({method: method for method in methods if method not in common.values()})
    metrics, pairs = collect(common)
    for method in methods:
        intervals[method] = metrics[method if method not in ("EN", "N", "E0", "K")
                                   else next(alias for alias, name in common.items()
                                             if name == method)]["balanced_accuracy"]
    pair_intervals["EN_minus_N"] = pairs["D_minus_B"]
    pair_intervals["EN_minus_E0"] = pairs["D_minus_A"]

    if gate2_passed:
        residual_aliases = {"D": "R", "B": "N", "A": "E0", "B56": "K"}
        residual_aliases.update({method: method for method in methods
                                 if method not in residual_aliases.values()})
        metrics, pairs = collect(residual_aliases)
        intervals["R"] = metrics["D"]["balanced_accuracy"]
        pair_intervals["R_minus_N"] = pairs["D_minus_B"]
        e0_k_aliases = {"D": "E0", "B": "K", "A": "N", "B56": "R"}
    else:
        pair_intervals["R_minus_N"] = {"low": None, "high": None, "n_boot": 0}
        e0_k_aliases = {"D": "E0", "B": "K", "A": "N", "B56": "E0"}
    e0_k_aliases.update({method: method for method in methods
                         if method not in e0_k_aliases.values()})
    _, pairs = collect(e0_k_aliases)
    pair_intervals["E0_minus_K"] = pairs["D_minus_B"]
    return intervals, pair_intervals


def _readout_summaries(fold_outputs: dict[str, dict[str, Any]],
                       intervals: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in fold_outputs[next(iter(fold_outputs))]["predictions"]:
        per_speaker = {fold: {"balanced_accuracy": _fold_metric(data["labels"],
                                                                  data["predictions"][method])}
                       for fold, data in fold_outputs.items()}
        values = [item["balanced_accuracy"] for item in per_speaker.values()]
        mean = float(np.mean(values)) if values and all(value is not None for value in values) else None
        result[method] = {"mean_balanced_accuracy": mean,
                          "balanced_accuracy_text_cluster_95": intervals[method],
                          "per_speaker": per_speaker}
    return result


def score_predictions(readouts: dict[str, Any], paired_differences: dict[str, Any],
                      speaker_identity: dict[str, Any], gate2_passed: bool
                      ) -> dict[str, Any]:
    """Apply Part 197's six stated prediction thresholds and headline reading rule."""
    def mean(name: str) -> float | None:
        value = readouts.get(name)
        return value.get("mean_balanced_accuracy") if isinstance(value, dict) else None

    def difference(name: str) -> float | None:
        value = paired_differences.get(name)
        return value.get("mean_balanced_accuracy_difference") if isinstance(value, dict) else None

    def interval(name: str) -> dict[str, Any]:
        value = paired_differences.get(name)
        return value.get("text_cluster_95", {}) if isinstance(value, dict) else {}

    en_n = difference("EN_minus_N")
    en_n_ci = interval("EN_minus_N")
    low, high = en_n_ci.get("low"), en_n_ci.get("high")
    # Review F3: a missing estimate or interval is not the registered "interval includes zero" reading.
    if en_n is None or low is None or high is None:
        p1971_verdict = "not_computable"
    elif low <= 0.0 <= high:
        p1971_verdict = "inconclusive"
    else:
        p1971_verdict = "held" if en_n >= 0.05 and low > 0.0 else "not_held"
    r_mean = mean("R") if gate2_passed else None
    l_values = [(index, mean(f"L_{index}")) for index in range(N_LAYERS)]
    available_layers = [(index, value) for index, value in l_values if value is not None]
    best_layer = max(available_layers, key=lambda item: (item[1], -item[0])) if available_layers else None
    deeper = [mean(f"L_{index}") for index in range(20, 24)]
    if best_layer is None or any(value is None for value in deeper):
        p1975_verdict = "not_computable"
    else:
        p1975_verdict = "held" if best_layer[0] in (0, 1, 2) and all(
            value <= 0.35 for value in deeper) else "not_held"
    s0 = speaker_identity.get("S_0", {}).get("balanced_accuracy")
    return {
        "197.1": {"prediction": "EN minus N >= 0.05 and its 95% interval is above zero",
                  "mean_difference": en_n, "text_cluster_95": en_n_ci,
                  "verdict": p1971_verdict,
                  "reading": ("adding layer 0 improves this fixed linear readout given N"
                              if p1971_verdict == "held" else
                              "inconclusive because the interval includes zero"
                              if p1971_verdict == "inconclusive" else
                              "the registered 5-point improvement criterion was not met")},
        "197.2": {"prediction": "R mean balanced accuracy >= 0.45",
                  "value": r_mean,
                  "verdict": "not_scorable" if not gate2_passed else
                  ("held" if r_mean is not None and r_mean >= 0.45 else "not_held")},
        "197.3": {"prediction": "EN minus E0 <= 0.03 point estimate",
                  "value": difference("EN_minus_E0"),
                  "verdict": "held" if difference("EN_minus_E0") is not None and
                  difference("EN_minus_E0") <= 0.03 else "not_held"},
        "197.4": {"prediction": "K mean balanced accuracy <= 0.30",
                  "value": mean("K"),
                  "verdict": "held" if mean("K") is not None and mean("K") <= 0.30 else "not_held"},
        "197.5": {"prediction": "best L_k is k in {0,1,2} and every L_20..L_23 <= 0.35",
                  "best_layer": None if best_layer is None else best_layer[0],
                  "best_layer_balanced_accuracy": None if best_layer is None else best_layer[1],
                  "layers_20_to_23": {f"L_{index}": value for index, value in
                                       enumerate(deeper, start=20)},
                  "verdict": p1975_verdict},
        "197.6": {"prediction": "S_0 speaker identity balanced accuracy >= 0.80",
                  "value": s0,
                  "verdict": "held" if s0 is not None and s0 >= 0.80 else "not_held"},
    }


def analyze(*, manifest_path: Path, work_dir: Path, out_dir: Path,
            expected_identity_path: Path, records_digest_path: Path,
            part196_results_path: Path, gate1_only: bool = False) -> dict[str, Any]:
    rows = p196._read_rows(manifest_path)
    identity = check_identity(work_dir, rows, expected_identity_path)
    gate1: dict[str, Any] = {"identity": identity}
    if identity["verdict"] != "passed":
        gate1["verdict"] = "failed"
        return _early_gate1_result(out_dir, gate1)

    digest = check_digest(work_dir, manifest_path, records_digest_path)
    gate1["records_digest"] = digest
    if digest["verdict"] != "passed":
        gate1["verdict"] = "failed"
        return _early_gate1_result(out_dir, gate1)

    regeneration = gate1_regeneration(manifest_path=manifest_path, work_dir=work_dir,
                                      part196_results_path=part196_results_path)
    gate1["regeneration"] = regeneration
    gate1["verdict"] = "passed" if regeneration["verdict"] == "passed" else "failed"
    if gate1["verdict"] != "passed":
        return _early_gate1_result(out_dir, gate1)
    if gate1_only:
        # The pre-registration measurement: stop before any Part 197 readout is computed.
        result = {"study": "part197", "status": "gate1_only", "gate1": gate1}
        _write_result(out_dir, result)
        print(json.dumps(_gate_summary(result), sort_keys=True))
        return result

    activations16, embeddings, acoustic, duration, record_status, record_errors = p196._read_records(
        rows, work_dir)
    active = _active_indices(rows, record_status, duration, work_dir)
    if not len(active):
        gate1["verdict"] = "passed"
        result = {"study": "part197", "status": "gates_failed", "gate1": gate1,
                  "gate2": {"verdict": "not_run", "reason": "no active recordings"},
                  "readouts": {}, "paired_differences": {}, "shuffled_label_controls": {},
                  "predictions": {}}
        _write_result(out_dir, result)
        print(json.dumps(_gate_summary(result), sort_keys=True))
        return result

    # Preserve the full activation table in float32; only the current OLS layer is promoted to float64.
    activations = activations16.astype(np.float32)
    del activations16
    nuisance = np.column_stack((acoustic.astype(np.float64), duration.astype(np.float64)))
    labels_all = np.asarray([STYLES.index(row["style"]) for row in rows], dtype=np.int64)
    labels = labels_all
    folds = _fold_indices(rows, active)
    gate2, residuals = _prepare_gate2(folds, nuisance, activations)
    gate2_passed = gate2["verdict"] == "passed"

    fold_outputs: dict[str, dict[str, Any]] = {}
    fold_predictions: dict[str, Any] = {}
    shuffled: dict[str, dict[str, Any]] = {"EN": {}, "R": {}}
    for fold_index, (heldout, (train, test)) in enumerate(folds.items()):
        train_y, test_y = labels[train], labels[test]
        train_speakers = np.asarray([rows[i]["speaker_id"] for i in train])
        train_n, test_n = nuisance[train], nuisance[test]
        train_e0 = activations[train, :WIDTH]
        test_e0 = activations[test, :WIDTH]
        features: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "N": (train_n, test_n),
            "E0": (train_e0, test_e0),
            "EN": (np.column_stack((train_e0, train_n)),
                   np.column_stack((test_e0, test_n))),
            "K": (np.column_stack((duration[train], acoustic[train, 6])),
                  np.column_stack((duration[test], acoustic[test, 6]))),
            "H": (acoustic[train], acoustic[test]),
        }
        predictions: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        for method, (x_train, x_test) in features.items():
            model = p196.fit_probe(x_train, train_y)
            test_scores = p196._score_probe(model, x_test)
            scores[method] = test_scores
            predictions[method] = test_scores.argmax(axis=1)

        for layer in range(N_LAYERS):
            method = f"L_{layer}"
            if layer == 0:
                scores[method], predictions[method] = scores["E0"], predictions["E0"]
                continue
            columns = slice(layer * WIDTH, (layer + 1) * WIDTH)
            model = p196.fit_probe(activations[train, columns], train_y)
            test_scores = p196._score_probe(model, activations[test, columns])
            scores[method] = test_scores
            predictions[method] = test_scores.argmax(axis=1)

        if gate2_passed:
            residual_train = residuals[heldout]["train"]
            residual_test = residuals[heldout]["test"]
            model = p196.fit_probe(residual_train, train_y)
            test_scores = p196._score_probe(model, residual_test)
            scores["R"] = test_scores
            predictions["R"] = test_scores.argmax(axis=1)

        fold_texts = np.asarray([rows[i]["text_key"] for i in test])
        fold_outputs[heldout] = {"labels": test_y, "texts": fold_texts,
                                 "predictions": predictions, "scores": scores}
        fold_predictions[heldout] = {
            method: {"balanced_accuracy": _fold_metric(test_y, pred)}
            for method, pred in predictions.items()}

        for method in ("EN", "R"):
            if method == "R" and not gate2_passed:
                continue
            x_train = (features["EN"][0] if method == "EN" else residuals[heldout]["train"])
            x_test = (features["EN"][1] if method == "EN" else residuals[heldout]["test"])
            shuffled[method][heldout] = p196._shuffle_diagnostic(
                x_train, train_y, train_speakers, x_test, test_y, p196.SEED + fold_index)

    # Part 196's speaker split and logistic probe are reused for each layer and final embedding.
    speaker_identity: dict[str, Any] = {}
    for layer in range(N_LAYERS):
        columns = slice(layer * WIDTH, (layer + 1) * WIDTH)
        speaker_identity[f"S_{layer}"] = p196._speaker_identity_diagnostic(
            activations[:, columns], rows, active, seed=p196.SEED)
    speaker_identity["S_final_embedding"] = p196._speaker_identity_diagnostic(
        embeddings, rows, active, seed=p196.SEED)

    intervals, pair_intervals = _bootstrap_results(fold_outputs, gate2_passed)
    readouts = _readout_summaries(fold_outputs, intervals)
    if not gate2_passed:
        readouts["R"] = {"status": "not_read", "reason": "Gate 2 failed in at least one fold"}
    pair_differences = {}
    for name, (left, right) in READOUT_PAIRS.items():
        interval = pair_intervals[name]
        point = None
        if name != "R_minus_N" or gate2_passed:
            left_value = readouts.get(left, {}).get("mean_balanced_accuracy")
            right_value = readouts.get(right, {}).get("mean_balanced_accuracy")
            if left_value is not None and right_value is not None:
                point = left_value - right_value
        pair_differences[name] = {"mean_balanced_accuracy_difference": point,
                                  "text_cluster_95": interval}

    shuffled_summary: dict[str, Any] = {}
    for method, per_fold in shuffled.items():
        if not per_fold:
            shuffled_summary[method] = {"status": "not_read" if method == "R" else "not_run",
                                        "per_heldout_speaker": {},
                                        "mean_balanced_accuracy": None,
                                        "chance": 1.0 / len(STYLES)}
            continue
        scores_by_fold = {speaker: value["balanced_accuracy"] for speaker, value in per_fold.items()}
        values = [value for value in scores_by_fold.values() if value is not None]
        shuffled_summary[method] = {
            "status": "computed", "per_heldout_speaker": scores_by_fold,
            "mean_balanced_accuracy": float(np.mean(values)) if len(values) == len(folds) else None,
            "chance": 1.0 / len(STYLES),
            "shuffled_within_each_training_speaker": True,
        }

    predictions = score_predictions(readouts, pair_differences, speaker_identity, gate2_passed)
    result = {"study": "part197", "status": "complete", "gate1": gate1, "gate2": gate2,
              "n_manifest_rows": len(rows), "n_active_rows": int(len(active)),
              "folds": fold_predictions, "readouts": readouts,
              "paired_differences": pair_differences,
              "shuffled_label_controls": shuffled_summary,
              "speaker_identity_by_layer": speaker_identity,
              "predictions": predictions,
              "reading_rule": {
                  "headline": predictions["197.1"]["reading"],
                  "EN_minus_N_is_fixed_linear_increment_not_information_theoretic_novelty": True,
                  "R_interpretation": "linear readability after subtracting the training-fitted linear prediction from N",
                  "R_minus_N_is_a_readout_contrast_not_information_beyond_N": True,
                  "EN_minus_E0_interpretation": "whether N adds to layer 0",
                  "speaker_and_layer_curves_are_descriptive": True,
              }}
    _write_result(out_dir, result)
    print(json.dumps(_gate_summary(result), sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "part196" / "manifest196.csv")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "part197")
    parser.add_argument("--expected-identity", type=Path,
                        default=Path(__file__).resolve().parent / "expected_identity.json")
    parser.add_argument("--records-digest", type=Path,
                        default=Path(__file__).resolve().parent / "records_digest.json")
    parser.add_argument("--part196-results", type=Path,
                        default=ROOT / "results" / "part196" / "results.json")
    parser.add_argument("--gate1-only", action="store_true",
                        help="stop after Gate 1 (the pre-registration regeneration measurement)")
    args = parser.parse_args(argv)
    result = analyze(manifest_path=args.manifest, work_dir=args.work_dir,
                     out_dir=args.out_dir, expected_identity_path=args.expected_identity,
                     records_digest_path=args.records_digest,
                     part196_results_path=args.part196_results, gate1_only=args.gate1_only)
    # Review F2: exit 0 only for a complete run, or a gate1-only run that stopped where it was asked to.
    expected = "gate1_only" if args.gate1_only else "complete"
    return 0 if result.get("status") == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
