#!/usr/bin/env python3
"""Question-disjoint Part 37 scoring for capture and stated-readout artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class AnalysisError(RuntimeError):
    """Invalid or incomplete Part 37 analysis input."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"{path}:{number}: invalid JSONL") from exc
        if not isinstance(value, dict):
            raise AnalysisError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def folds_are_question_disjoint(splits: Iterable[tuple[Iterable[int], Iterable[int]]],
                                 groups: list[str]) -> bool:
    return all(not (set(groups[index] for index in train) & set(groups[index] for index in test))
               for train, test in splits)


def verdict(condition: bool | None, rule: str) -> dict[str, str]:
    return {"verdict": "UNSCORABLE" if condition is None else ("PASS" if condition else "FAIL"),
            "rule": rule}


def _float(value: Any, fallback: float = float("nan")) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return fallback
    return converted if math.isfinite(converted) else fallback


class FeatureStore:
    def __init__(self, directory: Path):
        indexes = read_jsonl(directory / "index.jsonl")
        metrics = {row["record_id"]: row for row in read_jsonl(directory / "metrics.jsonl")}
        self.rows = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.arrays: dict[str, Any] = {}
        for row in indexes:
            if row.get("record_id") not in metrics:
                raise AnalysisError(f"{directory}: missing metrics for {row.get('record_id')}")
            row["metrics"] = metrics[row["record_id"]]
            row["capture_dir"] = str(directory)
            self.rows.append(row)
            self.by_id[row["record_id"]] = row
            shard = str(row["shard"])
            if shard not in self.arrays:
                self.arrays[shard] = np.load(directory / shard, mmap_mode="r", allow_pickle=False)
        if not self.rows:
            raise AnalysisError(f"{directory}: capture index is empty")
        first = self.rows[0]
        feature = self.feature(first, 0, 0)
        # Index 0 is the post-embedding residual stream; indices 1..N are the
        # outputs after each decoder layer, so analyze all N+1 readout states.
        self.layers = int(self.arrays[first["shard"]].shape[2])
        self.width = int(feature.shape[0])

    def feature(self, row: dict[str, Any], position: int, layer: int) -> np.ndarray:
        shard = self.arrays[str(row["shard"])]
        return np.asarray(shard[int(row["shard_row"]), position, layer], dtype=np.float32)

    def metrics(self, row: dict[str, Any]) -> dict[str, Any]:
        return row["metrics"]


def _sklearn():
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise AnalysisError(f"numpy and scikit-learn are required for analysis: {exc}") from exc
    return PLSRegression, LogisticRegression, balanced_accuracy_score, roc_auc_score, GroupKFold, StandardScaler


def _group_splits(groups: list[str], GroupKFold: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = sorted(set(groups))
    if len(unique) < 5:
        raise AnalysisError(f"need at least five question groups, found {len(unique)}")
    try:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=37)
        return [(np.asarray(train), np.asarray(test)) for train, test in splitter.split(np.zeros(len(groups)), groups=groups)]
    except TypeError:  # scikit-learn before GroupKFold's shuffle argument
        splitter = GroupKFold(n_splits=5)
        rng = np.random.default_rng(37)
        shuffled = list(unique)
        rng.shuffle(shuffled)
        remapped = {group: f"g{index:05d}" for index, group in enumerate(shuffled)}
        seeded_groups = [remapped[group] for group in groups]
        return [(np.asarray(train), np.asarray(test))
                for train, test in splitter.split(np.zeros(len(groups)), groups=seeded_groups)]


def _residualize(train_x: np.ndarray, test_x: np.ndarray, train_cov: np.ndarray,
                 test_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_design = np.column_stack([np.ones(len(train_cov)), train_cov])
    test_design = np.column_stack([np.ones(len(test_cov)), test_cov])
    beta = np.linalg.lstsq(train_design, train_x, rcond=None)[0]
    return train_x - train_design @ beta, test_x - test_design @ beta, beta


def _prepare_fold(x_train: np.ndarray, x_test: np.ndarray, cov_train: np.ndarray | None,
                  cov_test: np.ndarray | None, mode: str, components: int,
                  PLSRegression: Any, StandardScaler: Any) -> tuple[np.ndarray, np.ndarray]:
    if mode == "regressed":
        if cov_train is None or cov_test is None:
            raise AnalysisError("mean log-prob covariates missing for regression control")
        x_train, x_test, _ = _residualize(x_train, x_test, cov_train[:, :1], cov_test[:, :1])
    scaler = StandardScaler().fit(x_train)
    train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
    if mode == "subspace":
        if cov_train is None:
            raise AnalysisError("four surprisal summaries missing for subspace removal")
        n_components = min(components, max(1, len(x_train) - 1), train_scaled.shape[1])
        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(train_scaled, cov_train[:len(x_train), :4])
        weights = np.asarray(pls.x_weights_, dtype=np.float64)
        q, _ = np.linalg.qr(weights)
        train_scaled = train_scaled - (train_scaled @ q) @ q.T
        test_scaled = test_scaled - (test_scaled @ q) @ q.T
    return np.asarray(train_scaled, dtype=np.float32), np.asarray(test_scaled, dtype=np.float32)


def _fold_fit(x: np.ndarray, y: np.ndarray, groups: list[str], cov: np.ndarray | None,
              mode: str, layer: int) -> dict[str, Any]:
    PLSRegression, LogisticRegression, balanced_accuracy_score, roc_auc_score, GroupKFold, StandardScaler = _sklearn()
    splits = _group_splits(groups, GroupKFold)
    if not folds_are_question_disjoint(splits, groups):
        raise AnalysisError("question leakage detected in GroupKFold split")
    predicted = np.full(len(y), np.nan, dtype=np.float64)
    fold_balanced: list[float] = []
    for train, test in splits:
        cov_train = cov[train] if cov is not None else None
        cov_test = cov[test] if cov is not None else None
        x_train, x_test = _prepare_fold(x[train], x[test], cov_train, cov_test,
                                        mode, 8, PLSRegression, StandardScaler)
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000)
        classifier.fit(x_train, y[train])
        probabilities = classifier.predict_proba(x_test)[:, 1]
        predicted[test] = probabilities
        fold_balanced.append(float(balanced_accuracy_score(y[test], probabilities >= 0.5)))
    if np.isnan(predicted).any():
        raise AnalysisError(f"layer {layer}: cross-validation left predictions missing")
    return {"probabilities": predicted, "auc": float(roc_auc_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted >= 0.5)),
            "fold_balanced_accuracy": fold_balanced}


def _bootstrap_ci(y: np.ndarray, groups: list[str], probabilities: np.ndarray,
                  seed: int = 37, draws: int = 1000) -> dict[str, list[float]]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(groups)))
    by_group = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique}
    auc_values, bacc_values = [], []
    for _ in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in chosen])
        sample_y, sample_p = y[indices], probabilities[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        auc_values.append(float(roc_auc_score(sample_y, sample_p)))
        bacc_values.append(float(balanced_accuracy_score(sample_y, sample_p >= 0.5)))
    return {"auroc_95_ci": [float(value) for value in np.percentile(auc_values, [2.5, 97.5])],
            "balanced_accuracy_95_ci": [float(value) for value in np.percentile(bacc_values, [2.5, 97.5])],
            "bootstrap_draws": draws, "bootstrap_seed": seed}


def _dataset(store: FeatureStore, rows: list[dict[str, Any]], position: int, layer: int,
             covariate_mode: str = "four") -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    x = np.stack([store.feature(row, position, layer) for row in rows])
    y = np.asarray([int(row["_label"]) for row in rows], dtype=np.int64)
    groups = [str(row["question_id"]) for row in rows]
    if covariate_mode == "mean":
        cov = np.asarray([[_float(store.metrics(row).get("mean_logprob"))] for row in rows], dtype=np.float64)
    elif covariate_mode == "four":
        cov = np.asarray([[_float(store.metrics(row).get(field)) for field in
                           ("mean_logprob", "minimum_logprob", "mean_entropy", "last_token_logprob")]
                          for row in rows], dtype=np.float64)
    else:
        cov = None
    return x, y, groups, cov


def run_probe(store: FeatureStore, rows: list[dict[str, Any]], position: int,
              mode: str = "raw", covariate_mode: str = "four", only_layer: int | None = None,
              bootstrap: bool = True) -> dict[str, Any]:
    if len(rows) < 10:
        return {"n": len(rows), "error": "fewer than 10 records"}
    layers = [only_layer] if only_layer is not None else list(range(store.layers))
    layer_results = []
    for layer in layers:
        x, y, groups, cov = _dataset(store, rows, position, layer, covariate_mode)
        cv = _fold_fit(x, y, groups, cov, mode, layer)
        result = {"layer": layer, "auroc": cv["auc"], "balanced_accuracy": cv["balanced_accuracy"],
                  "fold_balanced_accuracy": cv["fold_balanced_accuracy"]}
        if bootstrap:
            result.update(_bootstrap_ci(y, groups, cv["probabilities"]))
        result["heldout_logit_mean_logprob_correlation"] = _correlation(
            np.log(np.clip(cv["probabilities"], 1e-8, 1 - 1e-8) /
                   np.clip(1 - cv["probabilities"], 1e-8, 1)),
            np.asarray([_float(store.metrics(row).get("mean_logprob")) for row in rows]))
        layer_results.append(result)
    if not layer_results or "auroc" not in layer_results[0]:
        return {"n": len(rows), "error": "no scoreable layers", "layers": layer_results}
    best = max(layer_results, key=lambda result: (result["auroc"], -result["layer"]))
    return {"n": len(rows), "question_count": len(set(row["question_id"] for row in rows)),
            "position": "span_last" if position == 0 else "followup_first", "mode": mode,
            "best_layer": best["layer"], "best_auroc": best["auroc"],
            "best_balanced_accuracy": best["balanced_accuracy"],
            "best_fold_balanced_accuracy": best["fold_balanced_accuracy"],
            "best_95_ci": {key: best[key] for key in ("auroc_95_ci", "balanced_accuracy_95_ci") if key in best},
            "best_heldout_logit_mean_logprob_correlation": best["heldout_logit_mean_logprob_correlation"],
            "layers": layer_results}


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else None


def _rows(store: FeatureStore, *, frame: str | None = "base", span_kinds: set[str] | None = None,
          slots: set[str] | None = None, authors: set[str] | None = None,
          header_groups: set[str] | None = None) -> list[dict[str, Any]]:
    selected = []
    for source in store.rows:
        if frame is not None and source.get("frame") != frame:
            continue
        if span_kinds is not None and source.get("span_kind") not in span_kinds:
            continue
        if slots is not None and source.get("slot") not in slots:
            continue
        if authors is not None and source.get("author") not in authors:
            continue
        if header_groups is not None and source.get("header_group") not in header_groups:
            continue
        if not source.get("span"):
            continue
        row = dict(source)
        selected.append(row)
    return selected


def _label(rows: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        copy = dict(row)
        copy["_label"] = int(bool(predicate(row)))
        prepared.append(copy)
    return prepared


def _all_score_positions(store: FeatureStore, rows: list[dict[str, Any]], mode: str = "raw",
                         covariate_mode: str = "four") -> dict[str, Any]:
    return {"span_last": run_probe(store, rows, 0, mode, covariate_mode),
            "followup_first": run_probe(store, rows, 1, mode, covariate_mode)}


def _fit_direction(store: FeatureStore, rows: list[dict[str, Any]], position: int, layer: int,
                   mode: str) -> dict[str, Any]:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x, y, _, cov = _dataset(store, rows, position, layer, "four")
    regression_beta = None
    if mode == "regressed":
        design = np.column_stack([np.ones(len(x)), cov[:, :1]])
        regression_beta = np.linalg.lstsq(design, x, rcond=None)[0]
        x = x - design @ regression_beta
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    projection_basis = None
    if mode == "subspace":
        pls = PLSRegression(n_components=min(8, len(x) - 1, scaled.shape[1]), scale=True).fit(scaled, cov)
        projection_basis, _ = np.linalg.qr(np.asarray(pls.x_weights_, dtype=np.float64))
        scaled = scaled - (scaled @ projection_basis) @ projection_basis.T
    classifier = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000).fit(scaled, y)
    return {"position": position, "layer": layer, "mode": mode, "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_, "weights": classifier.coef_[0],
            "regression_beta": regression_beta, "projection_basis": projection_basis,
            "intercept": float(classifier.intercept_[0])}


def _project(store: FeatureStore, row: dict[str, Any], direction: dict[str, Any]) -> float:
    x = store.feature(row, direction["position"], direction["layer"])
    beta = direction["regression_beta"]
    if beta is not None:
        mean_logprob = _float(store.metrics(row).get("mean_logprob"))
        x = x - np.column_stack([np.ones(1), np.asarray([[mean_logprob]])])[0] @ beta
    scaled = (x - direction["scaler_mean"]) / direction["scaler_scale"]
    basis = direction["projection_basis"]
    if basis is not None:
        scaled = scaled - (scaled @ basis) @ basis.T
    return float(scaled @ direction["weights"])


def _direction_cosine(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left["layer"] != right["layer"]:
        return None
    a, b = np.asarray(left["weights"]), np.asarray(right["weights"])
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else None


def _permutation_threshold(store: FeatureStore, rows: list[dict[str, Any]], position: int,
                           mode: str = "regressed", permutations: int = 200,
                           seed: int = 37) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    x_by_layer = [np.stack([store.feature(row, position, layer) for row in rows])
                  for layer in range(store.layers)]
    y = np.asarray([int(row["_label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row["question_id"]) for row in rows])
    cov = np.asarray([[_float(store.metrics(row).get(field)) for field in
                       ("mean_logprob", "minimum_logprob", "mean_entropy", "last_token_logprob")]
                      for row in rows], dtype=np.float64)
    unique_groups = sorted(set(groups))
    flip_rng = np.random.default_rng(seed)
    perm_maxima = []
    layer_distributions: list[list[float]] = [[] for _ in range(store.layers)]
    for permutation in range(permutations):
        flip = {group: int(flip_rng.integers(0, 2)) for group in unique_groups}
        y_null = np.asarray([int(label) ^ flip[group] for label, group in zip(y, groups)], dtype=np.int64)
        best = 0.5
        for layer, x in enumerate(x_by_layer):
            try:
                cv = _fold_fit(x, y_null, groups.tolist(), cov, mode, layer)
            except Exception:
                continue
            layer_auc = float(roc_auc_score(y_null, cv["probabilities"]))
            layer_distributions[layer].append(layer_auc)
            best = max(best, layer_auc)
        perm_maxima.append(best)
    return {"permutations": permutations, "seed": seed, "max_layer_auroc_95th": float(np.percentile(perm_maxima, 95)),
            "max_layer_auroc_values": perm_maxima,
            "per_layer_auroc_95th": [float(np.percentile(values, 95)) if values else None
                                      for values in layer_distributions],
            "null_definition": "one random own/other label flip per question; each permutation stores its maximum over layers"}


def _random_direction_control(store: FeatureStore, rows: list[dict[str, Any]], position: int,
                              count: int = 100, seed: int = 37) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    y = np.asarray([int(row["_label"]) for row in rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = []
    for layer in range(store.layers):
        x = np.stack([store.feature(row, position, layer) for row in rows])
        vectors = rng.normal(size=(count, x.shape[1]))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = []
        for vector in vectors:
            projection = x @ vector
            scores.append(float(roc_auc_score(y, projection)))
        result.append({"layer": layer, "random_directions": count,
                       "auroc_95th": float(np.percentile(scores, 95)),
                       "mean_auroc": float(np.mean(scores))})
    return {"seed": seed, "layers": result}


def _length_position_probe(rows: list[dict[str, Any]], position: int) -> dict[str, Any]:
    prepared = []
    for row in rows:
        copy = dict(row)
        copy["_label"] = int(row.get("_label", row.get("span_kind") == "own_g"))
        prepared.append(copy)
    X = np.asarray([[float(row.get("span_token_count", 0)), float(row.get("record_tokens", 0)),
                     float(row.get("readout_indices", [0, 0])[position])] for row in prepared])
    y = np.asarray([row["_label"] for row in prepared])
    groups = [str(row["question_id"]) for row in prepared]
    scored = _fold_fit(X, y, groups, None, "raw", -1)
    scored.update(_bootstrap_ci(y, groups, scored["probabilities"]))
    scored.pop("probabilities", None)
    return scored


def _matched_subset(store: FeatureStore, base: list[dict[str, Any]], position: int) -> dict[str, Any]:
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for row in base:
        if row.get("span_kind") in ("own_g", "other_g"):
            by_question.setdefault(row["question_id"], {})[row["span_kind"]] = row
    differences = {qid: abs(_float(pair["own_g"]["metrics"].get("mean_logprob")) -
                            _float(pair["other_g"]["metrics"].get("mean_logprob")))
                  for qid, pair in by_question.items() if "own_g" in pair and "other_g" in pair}
    if not differences:
        return {"n_questions": 0, "error": "no complete own/other pairs"}
    signed_differences = {qid: _float(pair["own_g"]["metrics"].get("mean_logprob")) -
                          _float(pair["other_g"]["metrics"].get("mean_logprob"))
                          for qid, pair in by_question.items() if "own_g" in pair and "other_g" in pair}
    threshold = float(np.median(list(differences.values())))
    matched = {qid for qid, difference in differences.items() if difference < threshold}
    if not matched:
        return {"n_questions": 0, "median_absolute_logprob_difference": threshold,
                "error": "no question has absolute surprisal difference strictly below the median"}
    rows = _label([row for row in base if row.get("question_id") in matched],
                  lambda row: row.get("span_kind") == "own_g")
    scores = {"raw": run_probe(store, rows, position, "raw", "four"),
              "mean_logprob_regressed": run_probe(store, rows, position, "regressed", "mean")}
    return {"n_questions": len(matched), "median_absolute_logprob_difference": threshold,
            "mean_absolute_logprob_difference": float(np.mean([differences[qid] for qid in matched])),
            "mean_signed_own_minus_other_logprob_difference": float(np.mean([signed_differences[qid] for qid in matched])),
            "scores": scores}


def _statement_summary(stated: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in stated:
        key = (str(row.get("form")), str(row.get("span_kind")), str(row.get("header_group")))
        grouped.setdefault(key, []).append(row)
    out = {}
    for (form, span_kind, header), rows in grouped.items():
        values = [_float(row.get("p_yes")) for row in rows]
        totals = [_float(row.get("total_two_token_mass")) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        totals = [value for value in totals if math.isfinite(value)]
        if not values or not totals:
            continue
        out[f"{form}|{span_kind}|{header}"] = {
            "n": len(values), "mean_p_yes": float(np.mean(values)), "median_p_yes": float(np.median(values)),
            "mean_total_two_token_mass": float(np.mean(totals)), "median_total_two_token_mass": float(np.median(totals)),
            "low_mass_flag": bool(np.median(totals) < 0.5),
        }
    return out


def _statement_by_span_kind(stated: list[dict[str, Any]], form: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in sorted({str(row.get("span_kind")) for row in stated if row.get("form") == form
                        and row.get("frame") == "base"}):
        rows = [row for row in stated if row.get("form") == form and row.get("frame") == "base"
                and row.get("span_kind") == kind]
        probabilities = [_float(row.get("p_yes")) for row in rows
                         if row.get("p_yes") is not None and math.isfinite(_float(row.get("p_yes")))]
        totals = [_float(row.get("total_two_token_mass")) for row in rows
                  if math.isfinite(_float(row.get("total_two_token_mass")))]
        if not probabilities or not totals:
            continue
        result[kind] = {"n": len(probabilities), "mean_p_yes": float(np.mean(probabilities)),
                        "median_p_yes": float(np.median(probabilities)),
                        "mean_total_two_token_mass": float(np.mean(totals)),
                        "median_total_two_token_mass": float(np.median(totals)),
                        "low_mass_flag": bool(np.median(totals) < 0.5)}
    return result


def _reader_analysis(name: str, capture_dir: Path, stated_path: Path) -> tuple[dict[str, Any], FeatureStore, list[dict[str, Any]]]:
    store = FeatureStore(capture_dir)
    stated = read_jsonl(stated_path)
    primary = _label(_rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={"assistant"}),
                     lambda row: row.get("span_kind") == "own_g")
    pooled = _label(_rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={"assistant", "user"}),
                    lambda row: row.get("span_kind") == "own_g")
    role = _label(_rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={"assistant", "user"}),
                  lambda row: row.get("slot") == "assistant")
    c1_within_author = {}
    for author in ("own", "other"):
        within = _label(_rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={"assistant", "user"}, authors={author}),
                        lambda row: row.get("slot") == "assistant")
        c1_within_author[author] = _all_score_positions(store, within)
    c2_by_slot = {}
    c2_rows_by_slot = {}
    for slot in ("assistant", "user"):
        subset = _label(_rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={slot}),
                        lambda row: row.get("span_kind") == "own_g")
        c2_rows_by_slot[slot] = subset
        c2_by_slot[slot] = _all_score_positions(store, subset)
    sampled = _label(_rows(store, frame="base", span_kinds={"own_g", "own_s1", "own_s2"}, slots={"assistant"}),
                     lambda row: row.get("span_kind") == "own_g")
    diffq = _label(_rows(store, frame="base", span_kinds={"own_g", "own_diffq"}, slots={"assistant"}),
                   lambda row: row.get("span_kind") == "own_g")
    probes = {
        "role_pooled": _all_score_positions(store, role),
        "role_within_author": c1_within_author,
        "authorship_pooled": _all_score_positions(store, pooled),
        "authorship_within_slot": c2_by_slot,
        "authorship_assistant_primary": _all_score_positions(store, primary),
        "own_g_vs_own_sampled": _all_score_positions(store, sampled),
        "own_g_vs_own_diffq": _all_score_positions(store, diffq),
        "own_g_vs_other_length_position_only": {
            "span_last": _length_position_probe(primary, 0),
            "followup_first": _length_position_probe(primary, 1),
        },
    }
    for key, rows in (("role_pooled", role), ("authorship_assistant_primary", primary)):
        for position, pkey in ((0, "span_last"), (1, "followup_first")):
            probes[key][pkey]["length_position_only"] = _length_position_probe(rows, position)

    main_span = probes["authorship_assistant_primary"]["span_last"]
    if "best_layer" not in main_span:
        raise AnalysisError(f"{name}: main authorship contrast has no best layer")
    author_best = int(main_span["best_layer"])
    c2_contrasts = {"authorship_pooled": pooled,
                    "authorship_within_slot_assistant": c2_rows_by_slot["assistant"],
                    "authorship_within_slot_user": c2_rows_by_slot["user"],
                    "authorship_assistant_primary": primary}
    for contrast_name, contrast_rows in c2_contrasts.items():
        target = probes[contrast_name] if contrast_name != "authorship_within_slot_assistant" and contrast_name != "authorship_within_slot_user" else None
        for position, pkey in ((0, "span_last"), (1, "followup_first")):
            controls = {
                "mean_logprob_regressed": run_probe(store, contrast_rows, position, "regressed", "mean"),
                "logprob_subspace_removed_8_pls": run_probe(store, contrast_rows, position, "subspace", "four"),
            }
            if target is not None:
                target[pkey].update(controls)
            else:
                slot = "assistant" if contrast_name.endswith("assistant") else "user"
                probes["authorship_within_slot"][slot][pkey].update(controls)
            if contrast_name == "authorship_assistant_primary":
                probes[contrast_name][pkey]["matched_surprisal_subset"] = _matched_subset(store, contrast_rows, position)
                probes[contrast_name][pkey]["random_direction_control"] = _random_direction_control(
                    store, contrast_rows, position, count=100, seed=37)
                probes[contrast_name][pkey]["label_shuffled_control"] = _permutation_threshold(
                    store, contrast_rows, position, mode="regressed", permutations=200, seed=37)

    author_direction = _fit_direction(store, primary, 0, author_best, "regressed")
    role_rows = role
    role_span = probes["role_pooled"]["span_last"]
    role_best = int(role_span["best_layer"])
    role_direction = _fit_direction(store, role_rows, 0, author_best, "raw")
    author_direction_raw = _fit_direction(store, primary, 0, author_best, "raw")
    cosine = _direction_cosine(author_direction, role_direction)

    label_rows = _rows(store, frame="label", span_kinds={"own_g", "other_g"},
                       slots={"assistant", "user"}, header_groups={"self", "other"})
    label_means: dict[str, list[float]] = {"self": [], "other": []}
    author_means: dict[str, list[float]] = {"own": [], "other": []}
    projections_by_record = {}
    for row in label_rows:
        projection = _project(store, row, author_direction)
        projections_by_record[row["record_id"]] = projection
        label_means[str(row["header_group"])].append(projection)
        author_means[str(row["author"])].append(projection)
    label_effect = (float(np.mean(label_means["self"]) - np.mean(label_means["other"]))
                    if label_means["self"] and label_means["other"] else None)
    author_effect = (float(np.mean(author_means["own"]) - np.mean(author_means["other"]))
                     if author_means["own"] and author_means["other"] else None)

    # Projection contrasts use the all-question authorship direction. Both raw and
    # mean-log-prob-regressed directions are reported in standard deviation units.
    base_projection_rows = _rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={"assistant"})
    baseline_projection = [_project(store, row, author_direction_raw) for row in base_projection_rows]
    baseline_sd = float(np.std(baseline_projection, ddof=1)) if len(baseline_projection) > 1 else 0.0
    sampled_rows = _rows(store, frame="base", span_kinds={"own_g", "own_s1", "own_s2"}, slots={"assistant"})
    sampled_projection = {row["span_kind"]: [] for row in sampled_rows}
    for row in sampled_rows:
        sampled_projection[row["span_kind"]].append(_project(store, row, author_direction_raw))
    own_g_mean = float(np.mean(sampled_projection.get("own_g", []))) if sampled_projection.get("own_g") else None
    own_s_mean = float(np.mean(sampled_projection.get("own_s1", []) + sampled_projection.get("own_s2", []))) if sampled_projection.get("own_s1") or sampled_projection.get("own_s2") else None
    raw_shift = ((own_g_mean - own_s_mean) / baseline_sd if baseline_sd and own_g_mean is not None and own_s_mean is not None else None)
    own_s_reg_projection: list[float] = []
    own_g_reg_projection: list[float] = []
    for row in sampled_rows:
        value = _project(store, row, author_direction)
        (own_g_reg_projection if row["span_kind"] == "own_g" else own_s_reg_projection).append(value)
    reg_sd = float(np.std([_project(store, row, author_direction) for row in base_projection_rows], ddof=1)) if len(base_projection_rows) > 1 else 0.0
    reg_shift = ((float(np.mean(own_g_reg_projection)) - float(np.mean(own_s_reg_projection))) / reg_sd
                 if reg_sd and own_g_reg_projection and own_s_reg_projection else None)

    other_projection_groups: dict[str, list[float]] = {}
    for row in _rows(store, frame="base", span_kinds={"own_g", "other_g", "own_diffq", "edit", "null_smol", "null_pile", "null_shuffle"}, slots={"assistant"}):
        other_projection_groups.setdefault(str(row["span_kind"]), []).append(_project(store, row, author_direction))
    projection_summary = {key: {"n": len(values), "mean_projection": float(np.mean(values)),
                                "sd_projection": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
                          for key, values in other_projection_groups.items()}

    # Spearman relation between the stated authorship answer and its captured state.
    auth_statements = [row for row in stated if row.get("form") == "authorship_yn"
                       and row.get("record_id") in store.by_id and row.get("p_yes") is not None]
    auth_probabilities, auth_projections, auth_totals = [], [], []
    for stated_row in auth_statements:
        capture_row = store.by_id[stated_row["record_id"]]
        auth_probabilities.append(_float(stated_row.get("p_yes")))
        auth_projections.append(_project(store, capture_row, author_direction))
        auth_totals.append(_float(stated_row.get("total_two_token_mass")))
    spearman = None
    if len(auth_probabilities) >= 3:
        from scipy.stats import spearmanr
        relation = spearmanr(auth_probabilities, auth_projections)
        spearman = float(relation.statistic) if math.isfinite(float(relation.statistic)) else None

    statement_summary = _statement_summary(stated)
    header_stats: dict[str, dict[str, Any]] = {}
    for form in ("authorship_yn", "memory_yn"):
        for header in ("self", "other", "neutral", "user"):
            rows = [row for row in stated if row.get("form") == form and row.get("frame") == "label"
                    and row.get("header_group") == header]
            if rows:
                ps = [_float(row.get("p_yes")) for row in rows]
                masses = [_float(row.get("total_two_token_mass")) for row in rows]
                ps = [value for value in ps if math.isfinite(value)]
                masses = [value for value in masses if math.isfinite(value)]
                header_stats.setdefault(form, {})[header] = {
                    "n": len(rows), "n_with_p_yes": len(ps),
                    "mean_p_yes": float(np.mean(ps)) if ps else None,
                    "median_p_yes": float(np.median(ps)) if ps else None,
                    "mean_total_two_token_mass": float(np.mean(masses)) if masses else None,
                    "median_total_two_token_mass": float(np.median(masses)) if masses else None,
                    "low_mass_flag": bool(np.median(masses) < 0.5) if masses else None,
                }
    header_effects = {}
    for form in ("authorship_yn", "memory_yn"):
        values = header_stats.get(form, {})
        if ("self" in values and "other" in values
                and values["self"]["mean_p_yes"] is not None
                and values["other"]["mean_p_yes"] is not None):
            header_effects[form] = values["self"]["mean_p_yes"] - values["other"]["mean_p_yes"]
    def neutral_is_between(form: str) -> bool:
        values = header_stats.get(form, {})
        self_value = values.get("self", {}).get("mean_p_yes")
        other_value = values.get("other", {}).get("mean_p_yes")
        neutral_value = values.get("neutral", {}).get("mean_p_yes")
        return (self_value is not None and other_value is not None and neutral_value is not None
                and min(self_value, other_value) <= neutral_value <= max(self_value, other_value))
    neutral_between = all(neutral_is_between(form) for form in ("authorship_yn", "memory_yn"))

    null_rows = [row for row in stated if row.get("form") == "authorship_yn"
                 and row.get("span_kind") in {"null_smol", "null_pile", "null_shuffle"}]
    null_ps = [_float(row.get("p_yes")) for row in null_rows if row.get("p_yes") is not None]
    null_masses = [_float(row.get("total_two_token_mass")) for row in null_rows
                   if math.isfinite(_float(row.get("total_two_token_mass")))]
    null_summary = {
        "n": len(null_ps), "mean_authorship_yes_probability": float(np.mean(null_ps)) if null_ps else None,
        "share_above_0_5": float(np.mean(np.asarray(null_ps) > 0.5)) if null_ps else None,
        "mean_total_two_token_mass": float(np.mean(null_masses)) if null_masses else None,
        "median_total_two_token_mass": float(np.median(null_masses)) if null_masses else None,
        "low_mass_flag": bool(np.median(null_masses) < 0.5) if null_masses else None,
        "by_null_kind": {kind: _statement_summary([row for row in null_rows if row.get("span_kind") == kind])
                         for kind in ("null_smol", "null_pile", "null_shuffle")},
    }
    authorship_by_kind = _statement_by_span_kind(stated, "authorship_yn")
    memory_by_kind = _statement_by_span_kind(stated, "memory_yn")
    stated_vs_null = {}
    null_mean = null_summary["mean_authorship_yes_probability"]
    if null_mean is not None:
        for kind, cell in authorship_by_kind.items():
            stated_vs_null[kind] = {**cell, "difference_from_combined_null": cell["mean_p_yes"] - null_mean}
    stated_numbers = {
        "authorship_yes_spearman": {"n": len(auth_probabilities), "spearman_rho": spearman,
                                    "mean_total_two_token_mass": float(np.mean(auth_totals)) if auth_totals else None,
                                    "median_total_two_token_mass": float(np.median(auth_totals)) if auth_totals else None,
                                    "low_mass_flag": bool(np.median(auth_totals) < 0.5) if auth_totals else None},
        "header_by_form": header_stats, "header_effects_self_minus_other": header_effects,
        "neutral_between_self_and_other": bool(neutral_between),
        "authorship_yes_by_span_kind": authorship_by_kind,
        "memory_yes_by_span_kind": memory_by_kind,
        "authorship_yes_comparison_to_combined_null": stated_vs_null,
        "null_authorship_readout": null_summary,
        "all_form_cell_summaries": statement_summary,
    }

    # Standardized contrast projection is reported alongside label and author effects.
    c3 = {"direction": "all-question mean-log-prob-regressed authorship direction at best assistant-slot layer",
          "best_layer": author_best, "label_effect_self_minus_other": label_effect,
          "author_effect_own_minus_other": author_effect,
          "label_to_author_absolute_effect_ratio": (abs(label_effect) / abs(author_effect)
                                                       if label_effect is not None and author_effect else None),
          "label_frames_n": len(label_rows)}
    own_s = {"own_g_mean_projection": own_g_mean, "own_sampled_mean_projection": own_s_mean,
             "raw_projection_shift_sd": raw_shift, "mean_logprob_regressed_shift_sd": reg_shift,
             "standardization_sd_own_g_other_g_raw": baseline_sd,
             "standardization_sd_own_g_other_g_regressed": reg_sd,
             "sample_projection_means_raw": {key: float(np.mean(values)) for key, values in sampled_projection.items() if values}}

    result = {
        "reader": name, "capture_records": len(store.rows), "layer_count": store.layers,
        "feature_width": store.width, "probes": probes,
        "authorship_direction": {"best_layer": author_best, "mode": "mean_logprob_regressed",
                                 "role_direction_cosine_at_authorship_best_layer": cosine,
                                 "raw_direction_layer": author_best, "role_best_layer": role_best},
        "c3_projection": c3, "own_g_vs_own_sampled_projection": own_s,
        "projection_contrasts": projection_summary, "stated": stated_numbers,
    }
    return result, store, stated


def _prediction_results(qwen: dict[str, Any], olmo: dict[str, Any]) -> dict[str, Any]:
    def val(data: dict[str, Any], *path: str) -> Any:
        node: Any = data
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    qwen_auc = val(qwen, "probes", "authorship_assistant_primary", "span_last", "best_auroc")
    qwen_reg_auc = val(qwen, "probes", "authorship_assistant_primary", "span_last", "mean_logprob_regressed", "best_auroc")
    qwen_perm = val(qwen, "probes", "authorship_assistant_primary", "span_last", "label_shuffled_control", "max_layer_auroc_95th")
    qwen_fold = val(qwen, "probes", "authorship_assistant_primary", "span_last", "best_fold_balanced_accuracy")
    qwen_matched = val(qwen, "probes", "authorship_assistant_primary", "span_last", "matched_surprisal_subset", "scores", "mean_logprob_regressed", "best_auroc")
    c3 = qwen.get("c3_projection", {})
    own_s = qwen.get("own_g_vs_own_sampled_projection", {})
    headers = qwen.get("stated", {}).get("header_effects_self_minus_other", {})
    qwen_rho = val(qwen, "stated", "authorship_yes_spearman", "spearman_rho")
    qwen_null = val(qwen, "stated", "null_authorship_readout", "mean_authorship_yes_probability")
    qwen_null_mass = val(qwen, "stated", "null_authorship_readout", "median_total_two_token_mass")
    qwen_null_share = val(qwen, "stated", "null_authorship_readout", "share_above_0_5")
    c3_ratio = c3.get("label_to_author_absolute_effect_ratio")
    memory_effect, auth_effect = headers.get("memory_yn"), headers.get("authorship_yn")
    label_shuffle_ok = qwen_reg_auc is not None and qwen_perm is not None and qwen_reg_auc > qwen_perm

    def role_reader_result(reader: dict[str, Any]) -> tuple[float | None, int | None]:
        probe = val(reader, "probes", "role_pooled", "span_last")
        if not isinstance(probe, dict) or "layers" not in probe:
            return None, None
        middle = [row for row in probe["layers"] if len(probe["layers"]) / 3 <= row["layer"] < 2 * len(probe["layers"]) / 3]
        if not middle:
            middle = probe["layers"]
        best = max(middle, key=lambda row: row["auroc"])
        return float(best["auroc"]), int(best["layer"])

    role_q, layer_q = role_reader_result(qwen)
    role_o, layer_o = role_reader_result(olmo)
    within_slot = val(qwen, "probes", "authorship_within_slot") or {}
    within_bacc = {slot: val(qwen, "probes", "authorship_within_slot", slot, "span_last", "best_balanced_accuracy")
                   for slot in ("assistant", "user")}
    within_ok = any(value is not None and 0.60 <= value <= 0.75 for value in within_bacc.values())
    role_c1 = {"qwen": {"middle_layer_best_auroc": role_q, "best_layer": layer_q},
               "olmo": {"middle_layer_best_auroc": role_o, "best_layer": layer_o}}
    shared_role_pass = (None if role_q is None or role_o is None else
                        role_q > 0.9 and role_o > 0.9)
    qwen_layer_scores = val(qwen, "probes", "authorship_assistant_primary", "span_last", "layers") or []
    p12_qualifying_layers = [row["layer"] for row in qwen_layer_scores
                             if sum(value >= 0.80 for value in row.get("fold_balanced_accuracy", [])) >= 4]
    p12_pass = bool(p12_qualifying_layers) if qwen_layer_scores else None
    p13_pass = within_ok if any(value is not None for value in within_bacc.values()) else None
    p14_pass = label_shuffle_ok if qwen_reg_auc is not None and qwen_perm is not None else None
    p2_pass = (None if qwen_auc is None or qwen_reg_auc is None else
               qwen_auc > 0.70 and qwen_reg_auc < 0.60)
    p8_cosine = val(qwen, "authorship_direction", "role_direction_cosine_at_authorship_best_layer")
    p8_pass = (None if qwen_reg_auc is None or p8_cosine is None else
               qwen_reg_auc >= 0.65 and p8_cosine >= 0.5)
    p9_shift = own_s.get("mean_logprob_regressed_shift_sd")
    p9_pass = None if p9_shift is None else p9_shift >= 0.5
    p3_pass = None if c3_ratio is None else c3_ratio < 0.5
    p4_pass = None if qwen_rho is None else qwen_rho > 0.3
    p11_pass = (None if memory_effect is None or auth_effect is None else
                abs(memory_effect) - abs(auth_effect) >= 0.10 and
                val(qwen, "stated", "neutral_between_self_and_other") is True)
    p15_pass = None if qwen_null is None else 0.10 <= qwen_null <= 0.25
    return {
        "37.1": {"observed": role_c1, "scoring_rule": "PASS iff Qwen and OLMo each have role AUROC > 0.90 at a layer in the middle third; layer and per-fold results are in probes.role_pooled." ,
                 **verdict(shared_role_pass, "both readers' best span-last AUROC in middle-third layers > 0.90")},
        "37.2": {"observed": {"qwen_before_surprisal": qwen_auc, "qwen_after_mean_logprob_regression": qwen_reg_auc},
                 **verdict(p2_pass, "Qwen assistant-slot raw best AUROC > 0.70 AND mean-log-prob-regressed best AUROC < 0.60")},
        "37.3": {"observed": c3, **verdict(p3_pass, "absolute self-label projection effect < 0.5 × absolute own-minus-other author effect")},
        "37.4": {"observed": val(qwen, "stated", "authorship_yes_spearman"),
                 **verdict(p4_pass, "Qwen Spearman rho(authorship yes/no p, regressed authorship projection) > 0.30")},
        "37.5": {"observed": None, "verdict": "HELD", "rule": "held by Amendment 1 until enacted ownership is registered and measured; this run omits enacted generations and judging"},
        "37.6": {"observed": None, "verdict": "HELD", "rule": "held by Amendment 1 until a causal steering stage is authorized and measured"},
        "37.7": {"observed": None, "verdict": "NOT RUN", "rule": "the cached OLMo base is not the Instruct parent revision; the required parent revision is not cached"},
        "37.8": {"observed": {"mean_logprob_regressed_best_auroc": qwen_reg_auc,
                                "role_authorship_cosine_at_authorship_best_layer": p8_cosine,
                                "subspace_removed_best_auroc": val(qwen, "probes", "authorship_assistant_primary", "span_last", "logprob_subspace_removed_8_pls", "best_auroc"),
                                "within_slot_best_balanced_accuracy": within_bacc},
                 **verdict(p8_pass, "mean-regressed best AUROC >= 0.65 AND cosine with role direction >= 0.50; report subspace and within-slot results as alternative controls")},
        "37.9": {"observed": own_s, **verdict(p9_pass, "own-greedy minus own-sampled projection shift after mean-log-prob regression >= 0.50 SD")},
        "37.10": {"observed": None, "verdict": "HELD", "rule": "held by Amendment 1; enacted first-person and back-reference rates are not part of this run"},
        "37.11": {"observed": {"self_minus_other_effects": headers,
                                "neutral_between_self_and_other": val(qwen, "stated", "neutral_between_self_and_other"),
                                "by_header_and_form": val(qwen, "stated", "header_by_form")},
                  **verdict(p11_pass, "absolute memory self-vs-other yes-p effect exceeds authorship effect by >= 0.10 and neutral header lies between self and other for both phrasings")},
        "37.12": {"observed": {"best_layer_fold_balanced_accuracy": qwen_fold,
                                "best_auroc_layer": val(qwen, "probes", "authorship_assistant_primary", "span_last", "best_layer"),
                                "qualifying_layers_with_4_of_5_folds": p12_qualifying_layers},
                  **verdict(p12_pass, "at least one layer has balanced accuracy >= 0.80 in at least 4 of 5 question-disjoint folds")},
        "37.13": {"observed": within_bacc,
                  **verdict(p13_pass, "at least one within-slot best-layer balanced accuracy is in [0.60, 0.75]")},
        "37.14": {"observed": {"mean_regressed_best_auroc": qwen_reg_auc,
                                "label_shuffled_max_layer_95th": qwen_perm},
                  **verdict(p14_pass, "Qwen mean-log-prob-regressed best-layer AUROC exceeds the 95th percentile of 200 per-question-flip maxima over layers")},
        "37.15": {"observed": {"null_mean_yes_rate": qwen_null, "share_above_0_5": qwen_null_share,
                                "median_total_two_token_mass": qwen_null_mass,
                                "low_mass_flag": qwen_null_mass is not None and qwen_null_mass < 0.5},
                  **verdict(p15_pass, "Qwen mean authorship-yes p over Smol, Pile, and shuffled nulls is between 0.10 and 0.25")},
    }


def _markdown(data: dict[str, Any]) -> str:
    lines = ["# Part 37 analysis", "", "Inputs were scored with five-fold question-disjoint GroupKFold (seed 37).", "",
             "## Prediction verdicts", "", "| Prediction | Verdict | Rule |", "|---|---|---|"]
    for key, row in data["predictions"].items():
        lines.append(f"| {key} | {row['verdict']} | {row['rule']} |")
    lines += ["", "## Main metrics", ""]
    for reader_name, reader in data["readers"].items():
        lines.append(f"### {reader_name}")
        lines.append("")
        for contrast in ("role_pooled", "authorship_pooled", "authorship_assistant_primary",
                         "own_g_vs_own_sampled", "own_g_vs_own_diffq"):
            contrast_data = reader["probes"].get(contrast, {})
            for position, metrics in contrast_data.items():
                if not isinstance(metrics, dict) or "best_auroc" not in metrics:
                    continue
                ci = metrics.get("best_95_ci", {}).get("auroc_95_ci")
                lines.append(f"- {contrast} / {position}: best layer {metrics['best_layer']}, AUROC {metrics['best_auroc']:.3f}, "
                             f"balanced accuracy {metrics['best_balanced_accuracy']:.3f}, folds {metrics['best_fold_balanced_accuracy']}, "
                             f"AUROC 95% question-bootstrap CI {ci}.")
        c3 = reader.get("c3_projection", {})
        lines.append(f"- C3 at authorship best layer {c3.get('best_layer')}: label effect {c3.get('label_effect_self_minus_other')}; "
                     f"author effect {c3.get('author_effect_own_minus_other')}; ratio {c3.get('label_to_author_absolute_effect_ratio')}.")
        stated = reader.get("stated", {})
        lines.append(f"- Stated authorship Spearman: {stated.get('authorship_yes_spearman')}; null readout: {stated.get('null_authorship_readout')}.")
        lines.append("")
    lines += ["## Scope and interpretation", "",
              "37.5, 37.6 and 37.10 are HELD under Amendment 1. 37.7 is NOT RUN because the OLMo base parent revision is not cached. "
              "37.2 and 37.8–37.15 are scored from the captured artifacts; the complete figures below include all layers, controls, folds, intervals, projections and stated token masses.",
              "", "## Complete results", "",
              "All numeric results, including per-layer scores, fold accuracies, controls, bootstrap intervals, projection summaries, and per-cell stated masses, are serialized below.",
              "", "```json", json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
    return "\n".join(lines)


def _pilot_reader(name: str, capture_dir: Path) -> dict[str, Any]:
    store = FeatureStore(capture_dir)
    primary = _label(_rows(store, frame="base", span_kinds={"own_g", "other_g"}, slots={"assistant"}),
                     lambda row: row.get("span_kind") == "own_g")
    regressions = {position: run_probe(store, primary, position, "regressed", "mean", bootstrap=False)
                   for position in (0, 1)}
    shuffled = {position: _permutation_threshold(store, primary, position, mode="regressed",
                                                permutations=200, seed=37)
                for position in (0, 1)}
    within = {}
    for contrast_name, kinds in (("own_g_vs_own_sampled", {"own_g", "own_s1", "own_s2"}),
                                 ("own_g_vs_own_diffq", {"own_g", "own_diffq"})):
        contrast = _label(_rows(store, frame="base", span_kinds=kinds, slots={"assistant"}),
                          lambda row: row.get("span_kind") == "own_g")
        within[contrast_name] = {
            "span_last": run_probe(store, contrast, 0, "raw", "four", bootstrap=False),
            "followup_first": run_probe(store, contrast, 1, "raw", "four", bootstrap=False),
        }
    every_layer_close = all(
        bool(regressions[position].get("layers")) and all(
            null is not None and abs(float(layer["auroc"]) - float(null)) <= 0.05
            for layer, null in zip(regressions[position].get("layers", []),
                                   shuffled[position]["per_layer_auroc_95th"])
        )
        for position in (0, 1)
    )
    within_close = all(
        isinstance(within[name][position_name], dict) and within[name][position_name].get("best_auroc") is not None
        and abs(float(within[name][position_name]["best_auroc"]) - 0.5) <= 0.05
        for name in within for position_name in ("span_last", "followup_first")
    )
    return {
        "reader": name, "question_count": len(set(row["question_id"] for row in primary)),
        "authorship_mean_logprob_regressed": {"span_last": regressions[0], "followup_first": regressions[1]},
        "label_shuffled_per_question_control": {"span_last": shuffled[0], "followup_first": shuffled[1]},
        "within_author_contrasts": within,
        "stop_gate": {"every_layer_at_both_positions_within_0_05_of_its_label_shuffle_95th": every_layer_close,
                      "both_within_author_contrasts_at_both_positions_within_0_05_of_chance": within_close,
                      "stop_and_report_surprisal_only": every_layer_close and within_close},
    }


def pilot_check(capture_qwen: Path, capture_olmo: Path, out_path: Path) -> bool:
    readers = {"qwen": _pilot_reader("qwen", capture_qwen), "olmo": _pilot_reader("olmo", capture_olmo)}
    stop = all(reader["stop_gate"]["stop_and_report_surprisal_only"] for reader in readers.values())
    result = {"schema": "part37-pilot-gate-v1", "readout": "Stage A only, first 20 questions",
              "full_run_proceed": not stop, "stop_reason": "both readers meet Amendment 1's surprisal-only stop gate" if stop else None,
              "readers": readers}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"pilot gate: {'STOP' if stop else 'continue to full run'}; wrote {out_path}")
    return stop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-qwen", type=Path, required=True)
    parser.add_argument("--capture-olmo", type=Path, required=True)
    parser.add_argument("--stated-qwen", type=Path)
    parser.add_argument("--stated-olmo", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--pilot-check", type=Path,
                        help="score the first-20 Stage A stop gate and write its JSON report; exit 3 when the registered stop rule is met")
    args = parser.parse_args(argv)
    try:
        if args.pilot_check is not None:
            return 3 if pilot_check(args.capture_qwen, args.capture_olmo, args.pilot_check) else 0
        if args.stated_qwen is None or args.stated_olmo is None or args.out_dir is None:
            raise AnalysisError("full analysis needs --stated-qwen, --stated-olmo and --out-dir")
        qwen, _, _ = _reader_analysis("qwen", args.capture_qwen, args.stated_qwen)
        olmo, _, _ = _reader_analysis("olmo", args.capture_olmo, args.stated_olmo)
        result = {"schema": "part37-analysis-v1", "folds": {"method": "GroupKFold", "n_splits": 5, "question_grouped": True, "seed": 37},
                  "readers": {"qwen": qwen, "olmo": olmo}}
        result["predictions"] = _prediction_results(qwen, olmo)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.out_dir / "analysis-37.json"
        md_path = args.out_dir / "analysis-37.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        md_path.write_text(_markdown(result), encoding="utf-8")
        print(f"wrote {json_path} and {md_path}")
        return 0
    except (AnalysisError, OSError, ValueError, KeyError) as exc:
        print(f"part37 analysis: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
