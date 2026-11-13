#!/usr/bin/env python3
"""Run the registered Part 203 held-out-actor readouts on recorded CREMA-D vectors."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from part196.analyze196 import (EMBEDDING_SIZE, N_LAYERS, N_UNITS, UNITS_PER_LAYER,  # noqa: E402
                                balanced_accuracy, recall_by_class)
from part196.record196 import MODEL_ID, MODEL_REVISION, manifest_sha  # noqa: E402
from part203.select203 import EMOTIONS, N_FOLDS, PER_CELL, read_manifest  # noqa: E402

SEED = 203
BOOTSTRAPS = 2000
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)          # Venn's grid (part203/prior/sound-eval.json chosen_C=100)
P196_C = 1.0                                     # part196.analyze196.fit_probe
P196_CLASS_WEIGHT = "balanced"
P196_SOLVER = "lbfgs"
P196_MAX_ITER = 5000
INNER_FOLDS = 5
SUBSET_ACTORS = 3
SUBSET_DRAWS = 5
N_CLASSES = len(EMOTIONS)
PREDICTIONS = ("203.1", "203.2", "203.3", "203.4", "203.5", "203.6", "203.7")   # PART203-DESIGN.md table
MISSING_CLASS_SCORE = -1e6      # part196.analyze196._score_probe's fill for a class absent from training


# ---------------------------------------------------------------- probes

def make_probe(c: float = P196_C, class_weight: str | None = P196_CLASS_WEIGHT):
    """The multinomial probe: standardize on training, then logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, class_weight=class_weight, solver=P196_SOLVER, max_iter=P196_MAX_ITER))


def fit_probe_c(x: np.ndarray, y: np.ndarray, c: float, class_weight: str | None = P196_CLASS_WEIGHT):
    return make_probe(c, class_weight).fit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64))


def probe_scores(model: Any, x: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Decision scores with one column per class in class order, and that class order.

    part196's `_score_probe` (part196/analyze196.py:242-252) returns the probe's own columns in `classes_`
    order when they are not its seven styles, and seven columns otherwise. With six emotions the column
    count alone does not say which case this is, so `classes_` decides: a class absent from the training
    rows gets MISSING_CLASS_SCORE, below every present one.

    `_score_probe` expands to SEVEN columns (part196's seven styles) whenever the probe's classes are not
    exactly `[0..6]`, so a probe trained without some emotion comes back as seven columns with a `-1e6`
    fill in each absent style's column. The expansion is detected from the column count (`raw.shape[1]`
    minus the classes' own count), and the probe's `classes_` columns are taken from the left, where
    `_score_probe` wrote them; the absent classes get MISSING_CLASS_SCORE below.
    """
    from part196.analyze196 import _score_probe

    classes = [int(cls) for cls in model[-1].classes_]
    raw = np.asarray(_score_probe(model, x), dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"probe scores have shape {raw.shape}")
    if raw.shape[1] > len(classes):
        # part196's expansion pads to its seven styles; the probe's own columns are the first len(classes).
        raw = raw[:, :len(classes)]
    if raw.shape[1] != len(classes):
        raise ValueError(f"probe scores have shape {raw.shape} for classes {classes}")
    full = np.full((len(raw), N_CLASSES), MISSING_CLASS_SCORE, dtype=np.float64)
    for column, cls in enumerate(classes):
        full[:, cls] = raw[:, column]
    return full, list(range(N_CLASSES))


def _log_loss(model: Any, x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    probabilities = model.predict_proba(np.asarray(x, dtype=np.float32))
    classes = list(model[-1].classes_)
    full = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for column, cls in enumerate(classes):
        full[:, int(cls)] = probabilities[:, column]
    # Floor at 1e-12 and renormalise: sklearn's log_loss warns when a row does not sum to one, and clipping
    # alone can push a row slightly above it.
    full = np.clip(full, 1e-12, 1.0)
    full /= full.sum(axis=1, keepdims=True)
    return float(log_loss(y, full, labels=list(range(N_CLASSES))))


def actor_folds(actors: np.ndarray, n_folds: int = INNER_FOLDS) -> list[np.ndarray]:
    """Split actor ids into folds by SHA-256 hex order (the outer fold rule, reused for inner CV)."""
    from part203.select203 import name_hash

    unique = sorted(set(str(actor) for actor in actors), key=lambda actor: (name_hash(actor), actor))
    if len(unique) < 2:
        raise ValueError("inner CV needs at least two actors")
    folds = [np.asarray([], dtype=int) for _ in range(min(n_folds, len(unique)))]
    for index, actor in enumerate(unique):
        folds[index % len(folds)] = np.concatenate(
            [folds[index % len(folds)], np.flatnonzero(np.asarray(actors).astype(str) == actor)])
    return [fold.astype(int) for fold in folds if fold.size]


def choose_c_inner(x: np.ndarray, y: np.ndarray, actors: np.ndarray,
                   grid: tuple[float, ...] = C_GRID) -> dict[str, Any]:
    """Choose C by actor-grouped inner CV: lowest mean held-out-actor log loss, ties to the lower C."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    folds = actor_folds(actors)
    if len(folds) < 2:
        raise ValueError("inner CV needs at least two actor folds")
    losses: dict[str, list[float]] = {f"{c:g}": [] for c in grid}
    for fold in folds:
        train = np.setdiff1d(np.arange(len(y)), fold)
        if len(np.unique(y[train])) != N_CLASSES or len(np.unique(y[fold])) != N_CLASSES:
            raise ValueError("inner actor fold lacks one of the six emotions")
        for c in grid:
            model = fit_probe_c(x[train], y[train], c)
            losses[f"{c:g}"].append(_log_loss(model, x[fold], y[fold]))
    means = {key: float(np.mean(values)) for key, values in losses.items()}
    best = min(grid, key=lambda c: (means[f"{c:g}"], c))
    return {"chosen_C": float(best), "mean_log_loss_by_C": means,
            "per_fold_log_loss_by_C": losses, "inner_folds": len(folds),
            "rule": "lowest mean held-out-actor log loss over the grid; ties to the lower C"}


def choose_layer_inner(activations: np.ndarray, labels: np.ndarray, actors: np.ndarray, *,
                       n_layers: int = N_LAYERS, width: int = UNITS_PER_LAYER) -> tuple[int, list[float | None]]:
    """Choose E's layer by inner actor-grouped CV on the training actors: five inner folds, the same
    `actor_folds` split the C selection uses, mean six-emotion balanced accuracy at C = 1.0, ties to the
    earlier layer.

    This is the design's rule (Method, "E, the mean MLP input of one layer"). part196's
    `choose_intermediate_layer` is leave-one-speaker-out and fits its probe at its own fixed C, so it is not
    used here; its tie rule (`max` on (value, -index), i.e. the earlier layer) is kept.
    """
    values = np.asarray(activations)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(actors).astype(str)
    if values.shape[1] != n_layers * width:
        raise ValueError(f"layer selection needs {n_layers} x {width} activations, got {values.shape[1]}")
    folds = actor_folds(groups)
    if len(folds) < 2:
        raise ValueError("inner CV needs at least two actor folds")
    means: list[float | None] = []
    for layer in range(n_layers):
        cols = slice(layer * width, (layer + 1) * width)
        scores = []
        for fold in folds:
            train = np.setdiff1d(np.arange(len(y)), fold)
            if len(np.unique(y[train])) != N_CLASSES or len(np.unique(y[fold])) != N_CLASSES:
                raise ValueError("inner actor fold lacks one of the six emotions")
            model = fit_probe_c(values[train, cols], y[train], P196_C)
            score = _balanced_accuracy(y[fold], model.predict(np.asarray(values[fold, cols], dtype=np.float32)))
            if score is not None:
                scores.append(score)
        means.append(float(np.mean(scores)) if scores else None)
    usable = [index for index, value in enumerate(means) if value is not None]
    if not usable:
        raise ValueError("no valid inner actor fold for the layer choice")
    best = max(usable, key=lambda index: (means[index], -index))
    return int(best), means


def subset_draws(train_actors: list[str], draws: int = SUBSET_DRAWS,
                 size: int = SUBSET_ACTORS) -> list[list[str]]:
    """Fixed 3-actor draws from a fold's training actors (SHA-256 order, rotating, wraparound)."""
    from part203.select203 import name_hash

    ordered = sorted(set(train_actors), key=lambda actor: (name_hash(actor), actor))
    if len(ordered) < size:
        raise ValueError(f"need {size} training actors, have {len(ordered)}")
    out = []
    for draw in range(draws):
        start = (draw * size) % len(ordered)
        out.append([ordered[(start + k) % len(ordered)] for k in range(size)])
    return out


# ---------------------------------------------------------------- metrics

def _text_cluster_bootstrap(fold_outputs: dict[str, dict[str, Any]], methods: list[str],
                            pairs: list[tuple[str, str]], *, n_boot: int | None = None,
                            seed: int = SEED) -> tuple[dict[str, Any], dict[str, Any]]:
    """Actor-cluster bootstrap: resample actors within each held-out fold, recompute the fold mean.

    `n_boot` defaults to the module's `BOOTSTRAPS`, read here rather than bound at definition, so a caller
    (or a test) that sets `analyze203.BOOTSTRAPS` gets that many draws.
    """
    if n_boot is None:
        n_boot = BOOTSTRAPS
    rng = np.random.default_rng(seed)
    draws = {method: np.full(n_boot, np.nan) for method in methods}
    pair_draws = {f"{a}_minus_{b}": np.full(n_boot, np.nan) for a, b in pairs}
    fold_names = sorted(fold_outputs)
    for draw in range(n_boot):
        per_method = {method: [] for method in methods}
        per_pair = {key: [] for key in pair_draws}
        for fold in fold_names:
            data = fold_outputs[fold]
            actors = np.asarray(data["actors"]).astype(str)
            unique = np.unique(actors)
            picked = rng.choice(unique, size=len(unique), replace=True)
            groups = {actor: np.flatnonzero(actors == actor) for actor in unique}
            indexes = np.concatenate([groups[str(actor)] for actor in picked])
            labels = data["labels"][indexes]
            fold_values = {}
            for method in methods:
                value = balanced_accuracy(labels, data["predictions"][method][indexes],
                                          n_classes=N_CLASSES)
                fold_values[method] = value
                if value is not None:
                    per_method[method].append(value)
            for (a, b), key in zip(pairs, pair_draws):
                if fold_values[a] is not None and fold_values[b] is not None:
                    per_pair[key].append(fold_values[a] - fold_values[b])
        for method in methods:
            if len(per_method[method]) == len(fold_names):
                draws[method][draw] = float(np.mean(per_method[method]))
        for key in pair_draws:
            if len(per_pair[key]) == len(fold_names):
                pair_draws[key][draw] = float(np.mean(per_pair[key]))
    return ({method: interval(values) for method, values in draws.items()},
            {key: interval(values) for key, values in pair_draws.items()})


def interval(values: np.ndarray) -> dict[str, Any]:
    usable = np.asarray(values, dtype=np.float64)
    usable = usable[np.isfinite(usable)]
    if not usable.size:
        return {"low": None, "high": None, "n_boot": 0}
    low, high = np.percentile(usable, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n_boot": int(len(usable))}


# ---------------------------------------------------------------- records

def _validate_identity(work_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    path = work_dir / "identity.json"
    if not path.is_file():
        raise FileNotFoundError(f"recorder identity is missing: {path}")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("model_id") != MODEL_ID or identity.get("model_snapshot_revision") != MODEL_REVISION:
        raise ValueError("recorder identity does not match the registered model snapshot")
    if identity.get("manifest_sha256") != manifest_sha(rows):
        raise ValueError("recorder identity manifest differs from the analysis manifest")
    return identity


def add_record_ids(rows: list[dict[str, str]]) -> None:
    """Give every manifest row the recorder's id: the file name without its .wav suffix.

    `read_manifest` returns only the manifest's columns, so a row has no id until this runs; the recorder
    (`record203.read_manifest_rows`) and `read_records` both key record files by it.
    """
    for row in rows:
        if "id" not in row:
            row["id"] = row["file"][:-4] if row["file"].lower().endswith(".wav") else row["file"]


def read_records(rows: list[dict[str, str]], work_dir: Path) -> dict[str, np.ndarray]:
    add_record_ids(rows)
    n = len(rows)
    out = {"v": np.full((n, EMBEDDING_SIZE), np.nan, dtype=np.float32),
           "d": np.full((n, EMBEDDING_SIZE), np.nan, dtype=np.float32),
           "e": np.full((n, N_UNITS), np.nan, dtype=np.float16),
           "h": np.full((n, 7), np.nan, dtype=np.float32),
           "status": np.full(n, "missing", dtype="U32"),
           # The recorder's Gate 1 field: True only when V's and D's decoders returned the same samples.
           "v_same_samples": np.zeros(n, dtype=bool),
           "errors": np.asarray([""] * n, dtype=object)}
    for index, row in enumerate(rows):
        path = work_dir / "records" / f"{row['id']}.npz"
        if not path.exists():
            out["errors"][index] = "record missing"
            continue
        try:
            with np.load(path, allow_pickle=False) as saved:
                if "id" in saved and str(saved["id"].item()) != row["id"]:
                    raise ValueError(f"record id {saved['id'].item()!r} is not {row['id']!r}")
                out["status"][index] = str(saved["status"].item())
                if out["status"][index] == "ok":
                    out["v"][index] = saved["v"]
                    out["d"][index] = saved["emb"]
                    out["e"][index] = saved["act_mean"]
                    out["h"][index] = saved["acoustic"]
                    # Missing means the recorder could not vouch for it, which `gate1` counts as a mismatch.
                    out["v_same_samples"][index] = bool(saved["v_same_samples"]) if "v_same_samples" in saved else False
                else:
                    out["errors"][index] = str(saved["reason"].item()) if "reason" in saved else ""
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            out["status"][index] = "record_error"
            out["errors"][index] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------- gates

def gate1(rows: list[dict[str, str]], status: np.ndarray, v: np.ndarray, d: np.ndarray,
          v_same_samples: np.ndarray | None = None) -> dict[str, Any]:
    """Gate 1: every planned clip recorded with finite V and D and V's input equal to D's; folds
    actor-disjoint; cells complete.

    `v_same_samples` is the recorder's per-clip statement that V and D were embedded from the same samples
    (record203._record_one). A V-vs-D reading is only the comparison the design names when it holds on every
    ok clip, so a False or missing value fails the gate. Passing None means the caller did not read it, which
    fails the same way: the gate never passes without the evidence.
    """
    problems = {}
    missing = [rows[i]["file"] for i in range(len(rows)) if status[i] != "ok"]
    if missing:
        problems["missing_or_failed"] = missing[:50]
        problems["missing_or_failed_count"] = len(missing)
    nonfinite = [rows[i]["file"] for i in range(len(rows)) if status[i] == "ok"
                 and not (np.isfinite(v[i]).all() and np.isfinite(d[i]).all())]
    if nonfinite:
        problems["nonfinite_vectors"] = nonfinite[:50]
        problems["nonfinite_vector_count"] = len(nonfinite)
    same = np.zeros(len(rows), dtype=bool) if v_same_samples is None else np.asarray(v_same_samples, dtype=bool)
    mismatched = [rows[i]["file"] for i in range(len(rows)) if status[i] == "ok" and not same[i]]
    if mismatched:
        problems["v_d_input_mismatch"] = mismatched[:50]
        problems["v_d_input_mismatch_count"] = len(mismatched)
    fold_actors: dict[str, set[str]] = {}
    for row in rows:
        fold_actors.setdefault(row["fold"], set()).add(row["actor"])
    overlaps = []
    for fold, actors in sorted(fold_actors.items()):
        others = set().union(*[value for key, value in fold_actors.items() if key != fold]) if len(fold_actors) > 1 else set()
        if actors & others:
            overlaps.append({"fold": fold, "overlap": sorted(actors & others)})
    if overlaps:
        problems["actor_overlap"] = overlaps
    # The selector draws exactly five clips per (actor, emotion) cell; a cell with another count is a
    # manifest fault. A clip that failed to record is counted by `missing_or_failed` above, not here.
    cells: dict[tuple[str, str], int] = {}
    for row in rows:
        cells[(row["actor"], row["emotion"])] = cells.get((row["actor"], row["emotion"]), 0) + 1
    short_cells = [f"{actor}/{emotion}" for (actor, emotion), count in sorted(cells.items()) if count != PER_CELL]
    if short_cells:
        problems["cells_not_five"] = short_cells[:50]
        problems["cells_not_five_count"] = len(short_cells)
    if len(fold_actors) != N_FOLDS:
        problems["fold_count"] = len(fold_actors)
    return {"verdict": "failed" if problems else "passed", "planned": len(rows),
            "recorded_ok": int(np.sum(status == "ok")), "folds": {k: sorted(v_) for k, v_ in sorted(fold_actors.items())},
            "problems": problems}


def gate2(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "gate2.json"
    if not path.is_file():
        return {"verdict": "not_run"}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- readouts

def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    """Six-emotion balanced accuracy, for the inner layer choice (part196's default is seven styles)."""
    return balanced_accuracy(labels, predictions, n_classes=N_CLASSES)


def _macro_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, list[float | None]]:
    """Macro one-vs-rest AUC over the six emotions (part196.analyze196.macro_ovr_auc, six classes)."""
    from part196.analyze196 import macro_ovr_auc

    return macro_ovr_auc(labels, scores, n_classes=N_CLASSES)


def _readout(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray,
             model_classes: list[int] | None = None) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError(f"readout scores have shape {scores.shape} for {len(labels)} labels")
    # A fold's probe can see fewer than six emotions in its training rows; `probe_scores` has already placed
    # its columns by class and given a missing class -1e6, so this check catches a caller that did not.
    classes = list(model_classes) if model_classes is not None else list(range(scores.shape[1]))
    if len(classes) != scores.shape[1]:
        raise ValueError("readout class list does not match the score columns")
    if classes != list(range(N_CLASSES)):
        full = np.full((len(labels), N_CLASSES), MISSING_CLASS_SCORE)
        for column, cls in enumerate(classes):
            full[:, int(cls)] = scores[:, column]
        scores = full
    auc, per_style = _macro_auc(labels, scores)
    return {"balanced_accuracy": balanced_accuracy(labels, predictions, n_classes=N_CLASSES),
            "macro_ovr_auc": auc, "per_style_auc": per_style,
            "per_emotion_recall": dict(zip(EMOTIONS, recall_by_class(labels, predictions, N_CLASSES)))}


def _fit_and_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                   y_test: np.ndarray, actors_train: np.ndarray, *, c: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if c is None:
        selection = choose_c_inner(x_train, y_train, actors_train)
        c_used = selection["chosen_C"]
    else:
        selection = {"chosen_C": float(c), "rule": "fixed (Part 196's probe setting)"}
        c_used = float(c)
    model = fit_probe_c(x_train, y_train, c_used)
    scores, classes = probe_scores(model, x_test)
    predictions = np.asarray(classes, dtype=np.int64)[scores.argmax(axis=1)]
    return (_readout(y_test, predictions, scores, classes),
            {"c_selection": selection, "predictions": predictions, "scores": scores, "classes": classes})


def analyze(*, manifest_path: Path, work_dir: Path, out_dir: Path) -> dict[str, Any]:
    # The recorder hashes the rows `record203.read_manifest_rows` builds (manifest columns plus id,
    # speaker_id, style, text_key, shard and row_index), so the analyzer reads the manifest the same way; the
    # identity check below compares that hash. `read_manifest` alone would hash a different row set.
    from part203.record203 import read_manifest_rows

    rows = read_manifest_rows(manifest_path)
    _validate_identity(work_dir, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(rows, work_dir)
    status, v, d, e, h = (records["status"], records["v"], records["d"], records["e"], records["h"])
    g1 = gate1(rows, status, v, d, records["v_same_samples"])
    # Amendment 1 (2026-11-12): a failed or missing record now carries its reason into Gate 1's report. The first
    # analysis listed two clips as missing whose files load with status ok; the reason was caught and dropped.
    failed_reasons = {rows[i]["file"]: str(records["errors"][i]) for i in range(len(rows))
                      if status[i] != "ok" and str(records["errors"][i])}
    if failed_reasons:
        g1.setdefault("problems", {})["reasons"] = dict(list(failed_reasons.items())[:50])
    g2 = gate2(work_dir)
    if g1["verdict"] != "passed" or g2.get("verdict") != "passed":
        result = {"study": "part203", "status": "gates_failed", "gate1": g1, "gate2": g2}
        _write(out_dir / "results.json", result)
        return result

    labels = np.asarray([EMOTIONS.index(row["emotion"]) for row in rows], dtype=np.int64)
    actors = np.asarray([row["actor"] for row in rows])
    folds = np.asarray([row["fold"] for row in rows])
    readouts = {"V": v, "D": d, "E": e, "H": h}
    fold_outputs: dict[str, dict[str, Any]] = {}
    fold_details: dict[str, Any] = {}
    for fold in sorted(set(folds.tolist())):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        train_y, test_y = labels[train], labels[test]
        if len(np.unique(train_y)) != N_CLASSES or len(np.unique(test_y)) != N_CLASSES:
            raise ValueError(f"fold {fold} lacks one of the six emotions")
        fold_pred: dict[str, np.ndarray] = {}
        fold_scores: dict[str, np.ndarray] = {}
        fold_classes: dict[str, list[int]] = {}
        details: dict[str, Any] = {"n_train": int(train.size), "n_test": int(test.size)}

        # V and D: Venn's C grid by inner actor CV, and Part 196's fixed probe as a separate row.
        for name in ("V", "D"):
            x = readouts[name]
            out, extra = _fit_and_score(x[train], train_y, x[test], test_y, actors[train], c=None)
            fold_pred[f"{name}_grid"] = extra["predictions"]
            fold_scores[f"{name}_grid"] = extra["scores"]
            fold_classes[f"{name}_grid"] = extra["classes"]
            details[f"{name}_grid"] = {"c_selection": extra["c_selection"]}
            out196, extra196 = _fit_and_score(x[train], train_y, x[test], test_y, actors[train], c=P196_C)
            fold_pred[f"{name}_p196"] = extra196["predictions"]
            fold_scores[f"{name}_p196"] = extra196["scores"]
            fold_classes[f"{name}_p196"] = extra196["classes"]

        # E: Part 196's layer probe, layer chosen by inner actor-grouped CV on the training actors: five
        # inner actor folds (the same split `choose_c_inner` uses), mean six-emotion balanced accuracy at
        # C = 1.0, ties to the earlier layer. The held-out fold never enters the choice.
        layer, inner = choose_layer_inner(e[train], train_y, actors[train], n_layers=N_LAYERS,
                                          width=UNITS_PER_LAYER)
        cols = slice(layer * UNITS_PER_LAYER, (layer + 1) * UNITS_PER_LAYER)
        out_e, extra_e = _fit_and_score(e[train, cols], train_y, e[test, cols], test_y, actors[train],
                                        c=P196_C)
        fold_pred["E_p196"] = extra_e["predictions"]
        fold_scores["E_p196"] = extra_e["scores"]
        fold_classes["E_p196"] = extra_e["classes"]
        details["E"] = {"selected_layer": layer, "inner_balanced_accuracy": inner,
                        "layer_inner_folds": len(actor_folds(actors[train])),
                        "layer_rule": "five inner actor folds (SHA-256 order, round-robin), mean six-emotion "
                                      "balanced accuracy at C = 1.0, ties to the earlier layer"}

        out_h, extra_h = _fit_and_score(h[train], train_y, h[test], test_y, actors[train], c=P196_C)
        fold_pred["H_p196"] = extra_h["predictions"]
        fold_scores["H_p196"] = extra_h["scores"]
        fold_classes["H_p196"] = extra_h["classes"]

        # The 3-actor ablation: V and E on several fixed 3-actor draws against all training actors.
        subset_rows = []
        for draw_index, draw in enumerate(subset_draws(sorted(set(actors[train].tolist())))):
            chosen = np.flatnonzero(np.isin(actors[train], draw))
            for name in ("V", "E"):
                x = readouts[name]
                # Design Method: V's C by inner CV on the draw's actors; E at C = 1.0 (Part 196's probe),
                # so E's drop isolates the actor count against the C = 1.0 all-training baseline.
                draw_c = None if name == "V" else P196_C
                if name == "E":
                    x = x[:, cols]
                out_s, extra_s = _fit_and_score(x[chosen], train_y[chosen], x[test], test_y,
                                                actors[train][chosen], c=draw_c)
                subset_rows.append({"draw": draw_index, "actors": draw, "readout": name,
                                    "balanced_accuracy": out_s["balanced_accuracy"],
                                    "chosen_C": extra_s["c_selection"]["chosen_C"]})
        details["three_actor_draws"] = subset_rows

        fold_outputs[fold] = {"actors": actors[test], "labels": test_y, "predictions": fold_pred,
                              "scores": fold_scores, "classes": fold_classes}
        fold_details[fold] = details

    methods = sorted(fold_outputs[next(iter(fold_outputs))]["predictions"])
    # Prediction 203.4 is stated against E_p196_minus_D_p196: same clips, same C, both recorded Part 196's way.
    pairs = [("V_grid", "E_p196"), ("V_grid", "H_p196"), ("D_p196", "V_grid"), ("E_p196", "V_grid"),
             ("E_p196", "D_p196"), ("V_grid", "V_p196"), ("D_grid", "D_p196")]
    intervals, pair_intervals = _text_cluster_bootstrap(fold_outputs, methods, pairs, n_boot=BOOTSTRAPS)

    summary = {}
    for method in methods:
        per_fold = {fold: _readout(data["labels"], data["predictions"][method], data["scores"][method],
                                   data["classes"][method])
                    for fold, data in fold_outputs.items()}
        values = [per_fold[fold]["balanced_accuracy"] for fold in sorted(per_fold)]
        summary[method] = {"mean_balanced_accuracy": float(np.mean(values)),
                           "balanced_accuracy_actor_cluster_95": intervals[method],
                           "mean_macro_ovr_auc": float(np.mean([per_fold[f]["macro_ovr_auc"]
                                                                for f in sorted(per_fold)])),
                           "per_fold": per_fold}
    paired = {}
    for a, b in pairs:
        point = summary[a]["mean_balanced_accuracy"] - summary[b]["mean_balanced_accuracy"]
        paired[f"{a}_minus_{b}"] = {"mean_difference": point, "actor_cluster_95": pair_intervals[f"{a}_minus_{b}"]}

    subset_summary: dict[str, Any] = {}
    for name in ("V", "E"):
        values = [row["balanced_accuracy"] for fold in fold_details.values()
                  for row in fold["three_actor_draws"] if row["readout"] == name]
        full = [summary[f"{name}_grid" if name == "V" else "E_p196"]["per_fold"][fold]["balanced_accuracy"]
                for fold in fold_details]
        subset_summary[name] = {"three_actor_draws_mean": float(np.mean(values)),
                                "three_actor_draws_all": values,
                                "all_training_actors_mean": float(np.mean(full))}

    result = {"study": "part203", "status": "complete", "n_clips": len(rows),
              "chance": 1.0 / N_CLASSES, "gate1": g1, "gate2": g2,
              "readouts": summary, "paired_differences": paired,
              "three_actor_ablation": subset_summary, "folds": fold_details,
              "rules": {"c_grid": list(C_GRID), "p196_probe": {"C": P196_C, "class_weight": P196_CLASS_WEIGHT},
                        "bootstrap": {"draws": BOOTSTRAPS, "seed": SEED, "unit": "actor within held-out fold"},
                        "predictions": list(PREDICTIONS)}}
    _write(out_dir / "results.json", result)
    return result


def _write(path: Path, value: dict[str, Any]) -> None:
    """Write JSON atomically (temp file, fsync, os.replace), so a kill never leaves a half-written file."""
    payload = json.dumps(value, indent=2, sort_keys=True, default=_jsonable, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest203.csv")
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/e/neuron-probe/part203-work"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "part203")
    args = parser.parse_args(argv)
    result = analyze(manifest_path=args.manifest, work_dir=args.work_dir, out_dir=args.out_dir)
    print(json.dumps({"results": str(args.out_dir / "results.json"), "status": result["status"],
                      "gate1": result["gate1"]["verdict"], "gate2": result["gate2"].get("verdict")}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
