#!/usr/bin/env python3
"""Run the registered Part 188 unit, model-verdict, and probe analyses."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
N_UNITS = 24 * 2048
L17_N1574 = 17 * 2048 + 1574
L15_N1126 = 15 * 2048 + 1126
BOOTSTRAPS = 2000


def auc_per_unit(scores: np.ndarray, labels: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """Vectorized rank-form AUC, returning one real-above-negative AUC per column."""
    x = np.asarray(scores, dtype=np.float32)
    y = np.asarray(labels, dtype=bool).reshape(-1)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != y.size:
        raise ValueError("scores and labels have different row counts")
    positive = int(y.sum())
    negative = int(y.size - positive)
    if not positive or not negative:
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    out = np.empty(x.shape[1], dtype=np.float32)
    row_index = np.arange(x.shape[0], dtype=np.int32)[:, None]
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        block = x[:, start:stop]
        if not np.isfinite(block).all():
            raise ValueError("AUC input contains a non-finite score")
        order = np.argsort(block, axis=0, kind="mergesort")
        sorted_x = np.take_along_axis(block, order, axis=0)
        sorted_y = y[order]
        new_group = np.ones(sorted_x.shape, dtype=bool)
        new_group[1:] = sorted_x[1:] != sorted_x[:-1]
        starts = np.maximum.accumulate(np.where(new_group, row_index, 0), axis=0)
        end_group = np.ones(sorted_x.shape, dtype=bool)
        end_group[:-1] = sorted_x[:-1] != sorted_x[1:]
        ends = np.minimum.accumulate(np.where(end_group, row_index, x.shape[0] - 1)[::-1], axis=0)[::-1]
        average_ranks = (starts.astype(np.float64) + ends.astype(np.float64) + 2.0) / 2.0
        rank_sum = np.where(sorted_y, average_ranks, 0.0).sum(axis=0)
        out[start:stop] = ((rank_sum - positive * (positive + 1) / 2.0) /
                           (positive * negative)).astype(np.float32)
    return out


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Scalar convenience wrapper around the exact rank-form implementation."""
    return float(auc_per_unit(np.asarray(scores).reshape(-1, 1), labels)[0])


def choose_unit(aucs: np.ndarray) -> tuple[int, int, float, float]:
    """Choose largest absolute departure from chance; ties select the low index."""
    values = np.asarray(aucs, dtype=np.float64).reshape(-1)
    deviations = np.abs(values - 0.5)
    deviations[~np.isfinite(deviations)] = -np.inf
    if not np.isfinite(deviations).any():
        raise ValueError("cannot select a unit without a finite AUC")
    index = int(np.argmax(deviations))
    sign = 1 if values[index] >= 0.5 else -1
    signed_auc = values[index] if sign > 0 else 1.0 - values[index]
    return index, sign, float(values[index]), float(signed_auc)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_logistic_l2(x: np.ndarray, labels: np.ndarray, C: float = 1.0,
                    max_iter: int = 1200, tolerance: float = 1e-7) -> dict[str, np.ndarray | float]:
    """Standardized L2 logistic regression using deterministic accelerated gradients."""
    data = np.asarray(x, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if data.ndim != 2 or data.shape[0] != y.size:
        raise ValueError("logistic input must be a [rows, features] matrix with matching labels")
    if not np.isin(y, (0.0, 1.0)).all() or len(np.unique(y)) != 2:
        raise ValueError("logistic regression needs both binary classes")
    if not np.isfinite(data).all() or C <= 0:
        raise ValueError("logistic input must be finite and C must be positive")
    mean = data.mean(axis=0)
    scale = data.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (data - mean) / scale
    design = np.column_stack((standardized, np.ones(len(y))))
    penalty = np.ones(design.shape[1], dtype=np.float64) / C
    penalty[-1] = 0.0

    # Power iteration bounds the smooth logistic Hessian's largest eigenvalue.
    vector = np.ones(design.shape[1], dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(35):
        vector = design.T @ (design @ vector) / len(y)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            break
        vector /= norm
    spectral = float(vector @ (design.T @ (design @ vector)) / len(y))
    step = 1.0 / max(0.25 * spectral + float(penalty.max()), 1e-8)

    def objective(theta: np.ndarray) -> float:
        logits = design @ theta
        return float(np.logaddexp(0.0, logits).mean() - np.dot(y, logits) / len(y) +
                     0.5 * np.dot(penalty * theta, theta))

    def gradient(theta: np.ndarray) -> np.ndarray:
        logits = design @ theta
        return design.T @ (_sigmoid(logits) - y) / len(y) + penalty * theta

    theta = np.zeros(design.shape[1], dtype=np.float64)
    extrapolated = theta.copy()
    momentum = 1.0
    previous_obj = objective(theta)
    for _ in range(max_iter):
        grad = gradient(extrapolated)
        candidate = extrapolated - step * grad
        candidate_obj = objective(candidate)
        # The Lipschitz estimate is conservative; backtrack if rounding overshoots.
        while candidate_obj > objective(extrapolated) + float(np.dot(grad, candidate - extrapolated)) + \
                np.dot(candidate - extrapolated, candidate - extrapolated) / (2.0 * step) + 1e-12:
            step *= 0.5
            candidate = extrapolated - step * grad
            candidate_obj = objective(candidate)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        next_extrapolated = candidate + ((momentum - 1.0) / next_momentum) * (candidate - theta)
        if np.linalg.norm(candidate - theta) <= tolerance * (1.0 + np.linalg.norm(theta)):
            theta = candidate
            break
        if candidate_obj > previous_obj:
            next_extrapolated = candidate
            next_momentum = 1.0
        theta, extrapolated, momentum = candidate, next_extrapolated, next_momentum
        previous_obj = candidate_obj
    return {"mean": mean, "scale": scale, "coef": theta[:-1], "intercept": float(theta[-1])}


def logistic_scores(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    data = np.asarray(x, dtype=np.float64)
    z = (data - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return (z @ np.asarray(model["coef"]) + float(model["intercept"])).astype(np.float64)


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, *, groups: np.ndarray | None = None,
                  n_boot: int = BOOTSTRAPS, seed: int = 188, clustered: bool = False) -> dict[str, float | int]:
    """Stratified clip bootstrap, or group-resampled bootstrap when clustered is true."""
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size or not np.isfinite(s).all() or len(np.unique(y)) != 2:
        return {"low": float("nan"), "high": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    if clustered:
        if groups is None:
            raise ValueError("clustered bootstrap requires groups")
        g = np.asarray(groups).astype(str).reshape(-1)
        if g.size != y.size:
            raise ValueError("group labels and observations have different lengths")
        unique = np.unique(g)
        members = {group: np.flatnonzero(g == group) for group in unique}
        labels_per_group = {group: set(y[indexes].tolist()) for group, indexes in members.items()}
        paired = all(values == {0, 1} for values in labels_per_group.values())
        pos_groups = np.unique(g[y == 1])
        neg_groups = np.unique(g[y == 0])
        for b in range(n_boot):
            if paired:
                selected = rng.choice(unique, size=len(unique), replace=True)
                indexes = np.concatenate([members[group] for group in selected])
            else:
                sampled: list[np.ndarray] = []
                for cls, cls_groups in ((1, pos_groups), (0, neg_groups)):
                    chosen = rng.choice(cls_groups, size=len(cls_groups), replace=True)
                    sampled.extend(members[group][y[members[group]] == cls] for group in chosen)
                indexes = np.concatenate(sampled)
            draws[b] = auc_score(y[indexes], s[indexes])
    else:
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        for b in range(n_boot):
            indexes = np.concatenate((rng.choice(positive, size=len(positive), replace=True),
                                      rng.choice(negative, size=len(negative), replace=True)))
            draws[b] = auc_score(y[indexes], s[indexes])
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n_boot": int(n_boot)}


def prediction_verdict(prediction_id: str, values: dict[str, float] | float | None) -> bool:
    """Apply one registered threshold; all comparisons use point estimates."""
    if values is None:
        return False
    mapping = values if isinstance(values, dict) else {"value": values}
    vals = list(mapping.values())
    if not vals or any(value is None or not np.isfinite(float(value)) for value in vals):
        return False
    if prediction_id == "188.1":
        return all(float(value) < 0.75 for value in vals)
    if prediction_id == "188.2":
        return all(float(value) >= 0.85 for value in vals)
    if prediction_id == "188.3":
        return all(float(value) >= 0.10 for value in vals)
    if prediction_id == "188.4":
        return all(float(value) >= 0.80 for value in vals)
    if prediction_id == "188.5":
        return float(vals[0]) < 0.70
    if prediction_id == "188.6":
        return all(float(value) >= 0.90 for value in vals)
    if prediction_id == "188.7":
        return all(float(value) < 0.05 for value in vals)
    if prediction_id == "188.8":
        return all(float(value) >= 0.80 for value in vals)
    if prediction_id == "188.9":
        return all(float(value) < 0.70 for value in vals)
    raise KeyError(f"unknown registered prediction {prediction_id}")


def _json_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if not np.isfinite(float(value)):
        return None
    return float(value) if isinstance(value, (float, np.floating)) else int(value)


def _interval(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
              seed: int) -> dict[str, Any]:
    return {"clip_level_95": bootstrap_auc(y, scores, n_boot=BOOTSTRAPS, seed=seed),
            "group_clustered_95": bootstrap_auc(y, scores, groups=groups, clustered=True,
                                                 n_boot=BOOTSTRAPS, seed=seed + 1)}


def _auc_result(y: np.ndarray, scores: np.ndarray, groups: np.ndarray | None = None,
                seed: int = 188) -> dict[str, Any]:
    if len(y) == 0 or len(np.unique(y)) != 2:
        return {"auc": None, "n": int(len(y)), "intervals": None}
    auc = auc_score(y, scores)
    group_vec = groups if groups is not None else np.arange(len(y)).astype(str)
    return {"auc": auc, "n": int(len(y)), "intervals": _interval(y, scores, group_vec, seed)}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_records(rows: list[dict[str, str]], work_dir: Path) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    activations = np.full((n, N_UNITS), np.nan, dtype=np.float16)
    embeddings = np.full((n, 768), np.nan, dtype=np.float32)
    seqlen = np.zeros(n, dtype=np.int32)
    decode_status = np.full(n, "missing", dtype="U32")
    record_status = np.full(n, "missing", dtype="U32")
    record_dir = work_dir / "records"
    for i, row in enumerate(rows):
        path = record_dir / f"{row['id']}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            status = str(saved["status"].item()) if "status" in saved else "unknown"
            record_status[i] = status
            decode_status[i] = str(saved["decode_status"].item()) if "decode_status" in saved else (
                "ok" if status == "ok" else status)
            if status == "ok":
                activations[i] = saved["act_mean"]
                embeddings[i] = saved["emb"]
                seqlen[i] = int(saved["seqlen"])
    return activations, embeddings, seqlen, decode_status, record_status


def _row_groups(rows: list[dict[str, str]], indexes: np.ndarray) -> np.ndarray:
    return np.asarray([rows[i].get("group") or rows[i].get("esc_category") or rows[i]["class_key"]
                       for i in indexes], dtype=str)


def _valid_indexes(indices: np.ndarray, status: np.ndarray, embeddings: np.ndarray | None = None) -> np.ndarray:
    good = status[indices] == "ok"
    if embeddings is not None:
        good &= np.isfinite(embeddings[indices]).all(axis=1)
    return indices[good]


def _binary_arrays(rows: list[dict[str, str]], indexes: np.ndarray, positive_set: str,
                   negative_set: str, activations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [i for i in indexes if rows[i]["analysis_set"] in (positive_set, negative_set)]
    labels = np.asarray([rows[i]["analysis_set"] == positive_set for i in selected], dtype=bool)
    return np.asarray(selected, dtype=int), labels, activations[selected]


def _prompt_scores(rows: list[dict[str, str]], indexes: np.ndarray, embeddings: np.ndarray,
                   prompts_path: Path, class_free: bool = False) -> np.ndarray:
    with np.load(prompts_path, allow_pickle=False) as prompts:
        phrases = [str(p) for p in prompts["phrases"]]
        real = prompts["real_embeddings"].astype(np.float32)
        imitation = prompts["imitation_embeddings"].astype(np.float32)
    phrase_index = {phrase: i for i, phrase in enumerate(phrases)}
    free_index = phrase_index["an animal"]
    output = np.empty(len(indexes), dtype=np.float32)
    for offset, i in enumerate(indexes):
        prompt_i = free_index if class_free else phrase_index.get(rows[i]["phrase"], free_index)
        output[offset] = float(np.dot(embeddings[i], real[prompt_i] - imitation[prompt_i]))
    return output


def _metric_for_subset(rows: list[dict[str, str]], indexes: np.ndarray, labels: np.ndarray,
                       scores: np.ndarray, seed: int) -> dict[str, Any]:
    groups = _row_groups(rows, indexes)
    return _auc_result(labels, scores, groups, seed)


def _prediction_block(metrics: dict[str, Any], gate3: dict[str, Any]) -> dict[str, Any]:
    a = metrics["A_to_B"]
    b = metrics["B_to_A"]
    values = {
        "188.1": {"A_to_B": a["model_class_informed"]["auc"],
                  "B_to_A": b["model_class_informed"]["auc"]},
        "188.2": {"A_to_B": a["selected_unit"]["heldout"]["auc"],
                  "B_to_A": b["selected_unit"]["heldout"]["auc"]},
        "188.3": {"A_to_B": (a["selected_unit"]["heldout"]["auc"] - a["model_class_informed"]["auc"]
                              if a["selected_unit"]["heldout"]["auc"] is not None and
                              a["model_class_informed"]["auc"] is not None else None),
                  "B_to_A": (b["selected_unit"]["heldout"]["auc"] - b["model_class_informed"]["auc"]
                              if b["selected_unit"]["heldout"]["auc"] is not None and
                              b["model_class_informed"]["auc"] is not None else None)},
        "188.4": {"A_to_B": a["selected_unit"]["real_vs_human"]["auc"],
                  "B_to_A": b["selected_unit"]["real_vs_human"]["auc"]},
        "188.5": {"pooled_non_chicken": metrics["known_unit_l17_n1574"]["auc"]},
        "188.6": {"A_to_B": a["output_embedding_probe"]["auc"],
                  "B_to_A": b["output_embedding_probe"]["auc"]},
        "188.7": {"A_to_B": (a["top32_probe"]["auc"] - a["selected_unit"]["heldout"]["auc"]
                              if a["top32_probe"]["auc"] is not None and
                              a["selected_unit"]["heldout"]["auc"] is not None else None),
                  "B_to_A": (b["top32_probe"]["auc"] - b["selected_unit"]["heldout"]["auc"]
                              if b["top32_probe"]["auc"] is not None and
                              b["selected_unit"]["heldout"]["auc"] is not None else None)},
        "188.8": {"A_to_B": metrics["chicken"]["A_to_B"]["auc"],
                  "B_to_A": metrics["chicken"]["B_to_A"]["auc"]},
        "188.9": {"A_to_B": a["sequence_length"]["auc"],
                  "B_to_A": b["sequence_length"]["auc"]},
    }
    thresholds = {
        "188.1": "< 0.75 in both folds", "188.2": ">= 0.85 in both folds",
        "188.3": ">= 0.10 above class-informed verdict in both folds",
        "188.4": ">= 0.80 against ESC-50 human sounds in both folds",
        "188.5": "< 0.70 pooled non-chicken", "188.6": ">= 0.90 in both folds",
        "188.7": "top-32 AUC minus selected-unit AUC < 0.05 in both folds",
        "188.8": ">= 0.80 for both selected units on chicken",
        "188.9": "< 0.70 in both folds",
    }
    return {key: {"value": values[key], "threshold": thresholds[key],
                  "verdict": "held" if prediction_verdict(key, values[key]) else "failed"}
            for key in values}


def analyze(*, manifest_path: Path, work_dir: Path, out_dir: Path,
            classes_path: Path = HERE / "classes.json") -> dict[str, Any]:
    rows = _read_rows(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    activations, embeddings, seqlen, decode_status, record_status = _read_records(rows, work_dir)
    decoded = decode_status == "ok"
    good = record_status == "ok"
    gate1_path = work_dir / "gate1.json"
    gate1_doc = json.loads(gate1_path.read_text(encoding="utf-8")) if gate1_path.exists() else {
        "verdict": "not_run", "clips": []}

    analysis_sets = sorted({row["analysis_set"] for row in rows})
    gate2_sets: dict[str, Any] = {}
    gate2_pass = True
    for set_name in analysis_sets:
        indexes = np.asarray([i for i, row in enumerate(rows) if row["analysis_set"] == set_name], dtype=int)
        decoded_indexes = indexes[decoded[indexes]]
        missing = [rows[i]["id"] for i in indexes if not decoded[i]]
        fraction = len(decoded_indexes) / len(indexes) if len(indexes) else 0.0
        passed = fraction >= 0.90
        gate2_pass &= passed
        gate2_sets[set_name] = {"planned": int(len(indexes)), "decoded": int(len(decoded_indexes)),
                                "fraction": fraction, "verdict": "passed" if passed else "failed",
                                "missing_ids": missing}
    parity = np.asarray([i for i, row in enumerate(rows) if row["set"] == "P"], dtype=int)
    gate2_sets["P"] = {"planned": int(len(parity)), "decoded": int(decoded[parity].sum()),
                        "fraction": float(decoded[parity].mean()) if len(parity) else 0.0,
                        "verdict": "passed" if len(parity) and decoded[parity].mean() >= 0.90 else "failed",
                        "missing_ids": [rows[i]["id"] for i in parity if not decoded[i]]}
    gate2_pass &= gate2_sets["P"]["verdict"] == "passed"

    # Gate 3 checks the fixed L15 N1126 rig unit on hen+rooster versus all E clips.
    gate3_pos = np.asarray([i for i, row in enumerate(rows)
                            if row["source"] == "esc" and row["esc_category"] in ("hen", "rooster")], dtype=int)
    gate3_neg = np.asarray([i for i, row in enumerate(rows) if row["analysis_set"] == "E"], dtype=int)
    gate3_pos = gate3_pos[good[gate3_pos]]
    gate3_neg = gate3_neg[good[gate3_neg]]
    gate3_y = np.r_[np.ones(len(gate3_pos), dtype=bool), np.zeros(len(gate3_neg), dtype=bool)]
    gate3_scores = np.r_[activations[gate3_pos, L15_N1126], activations[gate3_neg, L15_N1126]].astype(float)
    gate3_auc = auc_score(gate3_y, gate3_scores) if len(gate3_pos) and len(gate3_neg) else float("nan")
    gate3 = {"unit_index": L15_N1126, "positive": "ESC-50 hen and rooster", "negative": "E",
             "n_positive": len(gate3_pos), "n_negative": len(gate3_neg), "auc": _json_number(gate3_auc),
             "threshold": 0.90,
             "verdict": "passed" if np.isfinite(gate3_auc) and gate3_auc >= 0.90 else "failed"}

    prompts_path = work_dir / "prompts.npz"
    directions: dict[str, Any] = {}
    auc_arrays: dict[str, np.ndarray] = {}
    top_units: dict[str, np.ndarray] = {}
    selections: dict[str, dict[str, Any]] = {}
    direction_specs = (("A_to_B", "A", "B"), ("B_to_A", "B", "A"))
    for direction_index, (key, select_fold, test_fold) in enumerate(direction_specs):
        train_all = np.asarray([i for i, row in enumerate(rows)
                                if row["fold"] == select_fold and row["analysis_set"] in ("R", "I")], dtype=int)
        test_all = np.asarray([i for i, row in enumerate(rows)
                               if row["fold"] == test_fold and row["analysis_set"] in ("R", "I")], dtype=int)
        train = _valid_indexes(train_all, record_status)
        test = _valid_indexes(test_all, record_status)
        train_y = np.asarray([rows[i]["analysis_set"] == "R" for i in train], dtype=bool)
        test_y = np.asarray([rows[i]["analysis_set"] == "R" for i in test], dtype=bool)
        train_auc = auc_per_unit(activations[train].astype(np.float32), train_y)
        test_auc = auc_per_unit(activations[test].astype(np.float32), test_y)
        auc_arrays[f"select_{select_fold}"] = train_auc
        auc_arrays[f"heldout_on_{test_fold}"] = test_auc
        selected_index, selected_sign, selected_raw_auc, selected_auc = choose_unit(train_auc)
        selections[key] = {"index": selected_index, "sign": selected_sign,
                           "selection_auc_raw": selected_raw_auc, "selection_auc": selected_auc}
        train_signed = activations[train, selected_index].astype(np.float64) * selected_sign
        test_signed = activations[test, selected_index].astype(np.float64) * selected_sign
        unit_test = _metric_for_subset(rows, test, test_y, test_signed, 800 + direction_index)
        unit_test["selection_auc"] = selected_auc
        unit_test["selection_auc_raw"] = selected_raw_auc

        # Test I clips from participants absent from selection-fold imitation clips.
        train_participants = {rows[i]["participant_id"] for i in train
                              if rows[i]["analysis_set"] == "I" and rows[i]["participant_id"]}
        test_real = np.asarray([i for i in test if rows[i]["analysis_set"] == "R"], dtype=int)
        test_disjoint_i = np.asarray([i for i in test if rows[i]["analysis_set"] == "I" and
                                      rows[i]["participant_id"] not in train_participants], dtype=int)
        disjoint_indexes = np.r_[test_real, test_disjoint_i]
        disjoint_y = np.r_[np.ones(len(test_real), dtype=bool), np.zeros(len(test_disjoint_i), dtype=bool)]
        disjoint_scores = activations[disjoint_indexes, selected_index].astype(float) * selected_sign
        disjoint_metric = _metric_for_subset(rows, disjoint_indexes, disjoint_y, disjoint_scores,
                                             900 + direction_index)
        disjoint_metric["n_test_imitation_participants"] = len({rows[i]["participant_id"] for i in test_disjoint_i})

        # Compare the model's prompt-verdict and a regularized output-embedding probe.
        model_class = _prompt_scores(rows, test, embeddings, prompts_path, class_free=False)
        model_free = _prompt_scores(rows, test, embeddings, prompts_path, class_free=True)
        model_class_metric = _metric_for_subset(rows, test, test_y, model_class, 1000 + direction_index)
        model_free_metric = _metric_for_subset(rows, test, test_y, model_free, 1010 + direction_index)
        train_emb = embeddings[train].astype(np.float32)
        test_emb = embeddings[test].astype(np.float32)
        emb_model = fit_logistic_l2(train_emb, train_y.astype(np.int8), C=1.0)
        emb_scores = logistic_scores(emb_model, test_emb)
        emb_metric = _metric_for_subset(rows, test, test_y, emb_scores, 1020 + direction_index)

        # The top-32 logistic probe uses the same selection fold's unit ranking.
        unit_order = np.argsort(-np.abs(train_auc - 0.5), kind="stable")
        top64 = unit_order[:64].astype(np.int32)
        top32 = top64[:32]
        top_units[f"units_{select_fold}"] = top64
        top_units[f"act_{select_fold}"] = activations[:, top64].astype(np.float16)
        top_model = fit_logistic_l2(activations[train][:, top32].astype(np.float32),
                                    train_y.astype(np.int8), C=1.0)
        top32_scores = logistic_scores(top_model, activations[test][:, top32].astype(np.float32))
        top32_metric = _metric_for_subset(rows, test, test_y, top32_scores, 1030 + direction_index)

        # Sequence length uses the sign determined on the selection fold only.
        length_selection_auc = auc_score(train_y, seqlen[train]) if len(np.unique(train_y)) == 2 else float("nan")
        length_sign = 1 if length_selection_auc >= 0.5 else -1
        length_metric = _metric_for_subset(rows, test, test_y,
                                           seqlen[test].astype(float) * length_sign,
                                           1040 + direction_index)
        length_metric["selection_sign"] = "real_above" if length_sign > 0 else "real_below"
        length_metric["selection_auc"] = _json_number(length_selection_auc)

        # Human-sound and non-animal imitation controls use the same selected sign.
        h_all = np.asarray([i for i, row in enumerate(rows) if row["analysis_set"] == "H"], dtype=int)
        h_all = _valid_indexes(h_all, record_status)
        h_real = np.asarray([i for i in test if rows[i]["analysis_set"] == "R"], dtype=int)
        human_indexes = np.r_[h_real, h_all]
        human_y = np.r_[np.ones(len(h_real), dtype=bool), np.zeros(len(h_all), dtype=bool)]
        human_scores = activations[human_indexes, selected_index].astype(float) * selected_sign
        human_metric = _metric_for_subset(rows, human_indexes, human_y, human_scores,
                                          1050 + direction_index)
        vocal_h = np.asarray([i for i in h_all if rows[i]["esc_category"] in
                              set(json.loads(classes_path.read_text(encoding="utf-8"))["esc_human_vocal"])],
                             dtype=int)
        vocal_indexes = np.r_[h_real, vocal_h]
        vocal_y = np.r_[np.ones(len(h_real), dtype=bool), np.zeros(len(vocal_h), dtype=bool)]
        vocal_scores = activations[vocal_indexes, selected_index].astype(float) * selected_sign
        vocal_metric = _metric_for_subset(rows, vocal_indexes, vocal_y, vocal_scores,
                                          1060 + direction_index)
        n_indexes = np.asarray([i for i, row in enumerate(rows) if row["analysis_set"] == "N" and good[i]], dtype=int)
        animal_i = np.asarray([i for i in test if rows[i]["analysis_set"] == "I"], dtype=int)
        n_vs_i = np.r_[n_indexes, animal_i]
        n_vs_i_y = np.r_[np.ones(len(n_indexes), dtype=bool), np.zeros(len(animal_i), dtype=bool)]
        n_vs_i_scores = activations[n_vs_i, selected_index].astype(float) * selected_sign
        n_vs_i_metric = _metric_for_subset(rows, n_vs_i, n_vs_i_y, n_vs_i_scores,
                                           1070 + direction_index)

        directions[key] = {
            "select_fold": select_fold, "test_fold": test_fold,
            "selection": {"n_real": int(train_y.sum()), "n_imitation": int((~train_y).sum())},
            "test": {"n_real": int(test_y.sum()), "n_imitation": int((~test_y).sum())},
            "selected_unit": {"index": selected_index, "layer": selected_index // 2048,
                              "neuron": selected_index % 2048,
                              "sign": "real_above" if selected_sign > 0 else "real_below",
                              "selection_auc": selected_auc, "selection_auc_raw": selected_raw_auc,
                              "heldout": unit_test, "participant_disjoint": disjoint_metric,
                              "real_vs_human": human_metric,
                              "real_vs_human_vocal": vocal_metric,
                              "n_vs_test_i": n_vs_i_metric},
            "model_class_informed": model_class_metric,
            "model_class_free": model_free_metric,
            "output_embedding_probe": emb_metric,
            "top32_probe": top32_metric,
            "sequence_length": length_metric,
        }

    # Fixed known neuron, pooled over all non-chicken animal groups.
    pooled = np.asarray([i for i, row in enumerate(rows)
                         if row["analysis_set"] in ("R", "I") and row["group"] != "Chicken" and good[i]], dtype=int)
    pooled_y = np.asarray([rows[i]["analysis_set"] == "R" for i in pooled], dtype=bool)
    known_auc = (auc_score(pooled_y, activations[pooled, L17_N1574].astype(float))
                 if len(np.unique(pooled_y)) == 2 else float("nan"))
    known = {"unit_index": L17_N1574, "sign": "real_above", "n": int(len(pooled)),
             "auc": _json_number(known_auc) if len(np.unique(pooled_y)) == 2 else None}

    # Both fold-selected units are evaluated on the same chicken-only set.
    chicken = {}
    chicken_idx = np.asarray([i for i, row in enumerate(rows)
                              if row["analysis_set"] in ("C_R", "C_I") and good[i]], dtype=int)
    chicken_y = np.asarray([rows[i]["analysis_set"] == "C_R" for i in chicken_idx], dtype=bool)
    for key, selected in selections.items():
        scores = activations[chicken_idx, selected["index"]].astype(float) * selected["sign"]
        chicken[key] = _auc_result(chicken_y, scores, _row_groups(rows, chicken_idx),
                                    1080 + (0 if key == "A_to_B" else 1))

    # Save compact registered arrays and the full embedding/seqlen prompt artifact.
    np.savez_compressed(out_dir / "unit_auc.npz", select_A=auc_arrays["select_A"].astype(np.float32),
                        select_B=auc_arrays["select_B"].astype(np.float32),
                        heldout_on_B=auc_arrays["heldout_on_B"].astype(np.float32),
                        heldout_on_A=auc_arrays["heldout_on_A"].astype(np.float32))
    np.savez_compressed(out_dir / "top_units.npz", ids=np.asarray([row["id"] for row in rows]),
                        units_A=top_units["units_A"], act_A=top_units["act_A"],
                        units_B=top_units["units_B"], act_B=top_units["act_B"])
    with np.load(prompts_path, allow_pickle=False) as prompts:
        np.savez_compressed(out_dir / "emb188.npz", ids=np.asarray([row["id"] for row in rows]),
                            embeddings=embeddings.astype(np.float16), seqlen=seqlen,
                            prompt_phrases=prompts["phrases"], prompt_real_texts=prompts["real_texts"],
                            prompt_imitation_texts=prompts["imitation_texts"],
                            prompt_real_embeddings=prompts["real_embeddings"].astype(np.float16),
                            prompt_imitation_embeddings=prompts["imitation_embeddings"].astype(np.float16))

    metrics = {"A_to_B": directions["A_to_B"], "B_to_A": directions["B_to_A"],
               "known_unit_l17_n1574": known, "chicken": chicken}
    predictions = _prediction_block(metrics, gate3)
    length_near_selected = {}
    for key in ("A_to_B", "B_to_A"):
        unit_auc = directions[key]["selected_unit"]["heldout"]["auc"]
        length_auc = directions[key]["sequence_length"]["auc"]
        length_near_selected[key] = (None if unit_auc is None or length_auc is None
                                     else bool(length_auc >= unit_auc - 0.05))
    headline_permitted = (gate1_doc.get("verdict") == "passed" and gate3["verdict"] == "passed" and
                          predictions["188.2"]["verdict"] == "held" and
                          predictions["188.3"]["verdict"] == "held" and
                          all(value is False for value in length_near_selected.values()))
    withdrawal_reasons = []
    if gate1_doc.get("verdict") != "passed":
        withdrawal_reasons.append("Gate 1 did not pass")
    if gate3["verdict"] != "passed":
        withdrawal_reasons.append("Gate 3 did not pass")
    for prediction_id in ("188.2", "188.3"):
        if predictions[prediction_id]["verdict"] != "held":
            withdrawal_reasons.append(f"prediction {prediction_id} failed")
    for direction, near in length_near_selected.items():
        if near is True:
            withdrawal_reasons.append(f"sequence length is within 0.05 of the selected unit in {direction}")
        elif near is None:
            withdrawal_reasons.append(f"sequence length comparison is unavailable in {direction}")
    result = {"study": "part188", "n_planned_rows": n,
              "gate1": {"verdict": gate1_doc.get("verdict", "not_run"),
                        "clips": gate1_doc.get("clips", []),
                        "rules": gate1_doc.get("rules", {})},
              "gate2": {"verdict": "passed" if gate2_pass else "failed", "minimum_fraction": 0.90,
                        "sets": gate2_sets},
              "gate3": gate3, "directions": directions,
              "known_unit_l17_n1574": known, "chicken": chicken,
              "predictions": predictions,
              "reading_rule": {"length_within_0_05_of_selected": length_near_selected,
                               "headline_permitted": headline_permitted,
                               "withdrawal_reasons": withdrawal_reasons}}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest188.csv")
    parser.add_argument("--work-dir", type=Path, default=HERE / "work")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--classes", type=Path, default=HERE / "classes.json")
    args = parser.parse_args()
    result = analyze(manifest_path=args.manifest, work_dir=args.work_dir,
                     out_dir=args.out_dir, classes_path=args.classes)
    print(json.dumps({"gate2": result["gate2"]["verdict"], "gate3": result["gate3"]["verdict"],
                      "predictions": {k: v["verdict"] for k, v in result["predictions"].items()},
                      "results": str(args.out_dir / "results.json")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
