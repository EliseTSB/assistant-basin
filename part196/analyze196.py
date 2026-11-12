#!/usr/bin/env python3
"""Run the registered Part 196 held-out-speaker readouts and diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from part196.select196 import STYLES, gate1_check_recorded, read_metadata, text_key  # noqa: E402
from part196.record196 import DEFAULT_DATA_DIR, MODEL_ID, MODEL_REVISION, manifest_sha  # noqa: E402
from part188.analyze188 import auc_per_unit  # noqa: E402

BOOTSTRAPS = 2000
SEED = 196
N_LAYERS = 24
UNITS_PER_LAYER = 2048
N_UNITS = N_LAYERS * UNITS_PER_LAYER
EMBEDDING_SIZE = 768


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def selected_style_profile(manifest_rows: list[dict[str, str]], analyzed_ids: set[str],
                           metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize successful manifest clips using WAV-header durations and metadata text."""
    selected = [row for row in manifest_rows if str(row["id"]) in analyzed_ids]
    selected_by_id = {str(row["id"]): row for row in selected}
    # Expresso reuses some ids across styles (each such id is one `default` clip plus two `longform` rows), so a
    # metadata row is matched on (id, speaker_id, style), the identity record196.read_transcripts uses (Amendment 2).
    selected_by_identity = {(str(row["id"]), str(row["speaker_id"]), str(row["style"])): row for row in selected}
    transcript_by_id: dict[str, str] = {}
    for item in metadata_rows:
        clip_id = str(item.get("id", ""))
        identity = (clip_id, str(item.get("speaker_id", "")), str(item.get("style", "")))
        if identity not in selected_by_identity:
            continue
        row = selected_by_identity[identity]
        transcript = item.get("text")
        if transcript is None:
            raise ValueError(f"metadata transcript is missing for {clip_id}")
        transcript = str(transcript)
        if text_key(transcript) != str(row["text_key"]):
            raise ValueError(f"metadata transcript hash differs from manifest for {clip_id}")
        if clip_id in transcript_by_id and transcript_by_id[clip_id] != transcript:
            raise ValueError(f"metadata contains conflicting transcripts for {clip_id}")
        transcript_by_id[clip_id] = transcript
    missing = sorted(set(selected_by_id) - set(transcript_by_id))
    if missing:
        raise KeyError(f"metadata transcripts missing for {len(missing)} analysed clips")

    profile: dict[str, Any] = {}
    for style in STYLES:
        clips = [row for row in selected if row["style"] == style]
        durations = np.asarray([float(row["duration_seconds"]) for row in clips], dtype=np.float64)
        word_counts = np.asarray([
            len(re.findall(r"\S+", transcript_by_id[str(row["id"])].strip()))
            for row in clips
        ], dtype=np.float64)
        if len(clips):
            quartiles = np.percentile(durations, [25, 50, 75])
            duration_profile = {"mean": float(np.mean(durations)), "q25": float(quartiles[0]),
                                "median": float(quartiles[1]), "q75": float(quartiles[2])}
            mean_words = float(np.mean(word_counts))
        else:
            duration_profile = {"mean": None, "q25": None, "median": None, "q75": None}
            mean_words = None
        profile[style] = {"clip_count": len(clips), "duration_seconds": duration_profile,
                          "mean_transcript_words": mean_words}
    return profile


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(auc_per_unit(np.asarray(scores).reshape(-1, 1), labels)[0])


def macro_ovr_auc(labels: np.ndarray, scores: np.ndarray,
                  n_classes: int = len(STYLES)) -> tuple[float | None, list[float | None]]:
    """Macro one-vs-rest AUC for a single held-out fold."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape != (len(y), n_classes):
        raise ValueError("multiclass AUC scores must have one column per class")
    values: list[float | None] = []
    for cls in range(n_classes):
        binary = (y == cls).astype(np.int8)
        if binary.sum() == 0 or binary.sum() == len(binary):
            values.append(None)
        else:
            values.append(auc_score(binary, x[:, cls]))
    present = [value for value in values if value is not None]
    return (float(np.mean(present)) if len(present) == n_classes else None), values


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray,
                      n_classes: int = len(STYLES)) -> float | None:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if len(y) == 0:
        return None
    recalls = []
    for cls in range(n_classes):
        members = y == cls
        if not members.any():
            return None
        recalls.append(float(np.mean(pred[members] == cls)))
    return float(np.mean(recalls))


def recall_by_class(labels: np.ndarray, predictions: np.ndarray,
                    n_classes: int = len(STYLES)) -> list[float | None]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    return [float(np.mean(pred[y == cls] == cls)) if np.any(y == cls) else None
            for cls in range(n_classes)]


def mean_fold_auc(folds: list[tuple[np.ndarray, np.ndarray]],
                  n_classes: int = len(STYLES)) -> float | None:
    """Average fold-local macro AUCs; never ranks predictions across folds."""
    values = [macro_ovr_auc(y, scores, n_classes)[0] for y, scores in folds]
    usable = [value for value in values if value is not None]
    return float(np.mean(usable)) if len(usable) == len(values) and usable else None


def select_b_phrasings(cosines: np.ndarray, labels: np.ndarray
                       ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Choose one mean-or-phrase cosine per style by train-only OVR AUC.

    Candidate iteration order encodes the exact tie rule: mean first, then low phrase index.
    """
    x = np.asarray(cosines, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 3 or x.shape[1:] != (len(STYLES), 8) or x.shape[0] != len(y):
        raise ValueError("B selection expects [rows, 7 styles, 8 phrasings]")
    selected = np.empty((len(y), len(STYLES)), dtype=np.float64)
    details = []
    for cls, style in enumerate(STYLES):
        candidates = [("mean", -1, x[:, cls, :].mean(axis=1))]
        candidates.extend(("phrase", phrase, x[:, cls, phrase]) for phrase in range(8))
        best_kind, best_index, best_values = candidates[0]
        best_auc = auc_score((y == cls).astype(np.int8), best_values)
        for kind, phrase_index, values in candidates[1:]:
            value = auc_score((y == cls).astype(np.int8), values)
            if value > best_auc:
                best_kind, best_index, best_values, best_auc = kind, phrase_index, values, value
        selected[:, cls] = best_values
        details.append({"style": style, "choice": best_kind,
                        "phrase_index": best_index if best_kind == "phrase" else None,
                        "train_auc": float(best_auc)})
    return selected, details


def fit_calibrated_verdict(cosines: np.ndarray, labels: np.ndarray
                           ) -> dict[str, Any]:
    """Fit z_k=a*c_k+b_k with a>0, sum(b)=0, weighted multinomial loss, no penalty."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    x = np.asarray(cosines, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n, k = x.shape
    if k != len(STYLES) or len(y) != n or set(np.unique(y)) != set(range(k)):
        raise ValueError("calibrated verdict needs all seven classes and matching rows")
    counts = np.bincount(y, minlength=k).astype(np.float64)
    sample_weights = 1.0 / counts[y]
    weight_sum = float(sample_weights.sum())

    def decode(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scale = float(np.exp(theta[0]))
        offsets = np.r_[theta[1:], -np.sum(theta[1:])]
        return scale, offsets

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scale, offsets = decode(theta)
        logits = scale * x + offsets
        normalizer = logsumexp(logits, axis=1)
        loss = float(np.sum(sample_weights * (normalizer - logits[np.arange(n), y])) / weight_sum)
        probabilities = np.exp(logits - normalizer[:, None])
        probabilities[np.arange(n), y] -= 1.0
        probabilities *= sample_weights[:, None] / weight_sum
        grad_logits = probabilities
        grad_scale = float(np.sum(grad_logits * x)) * scale
        grad_offsets = grad_logits.sum(axis=0)
        grad_free = grad_offsets[:-1] - grad_offsets[-1]
        return loss, np.r_[grad_scale, grad_free]

    result = minimize(objective, np.zeros(k, dtype=np.float64), jac=True, method="L-BFGS-B",
                      bounds=[(-20.0, 20.0)] + [(None, None)] * (k - 1),
                      options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50})
    scale, offsets = decode(result.x)
    if not scale > 0 or not np.isclose(offsets.sum(), 0.0, atol=1e-10):
        raise RuntimeError("calibrated cosine fit violated its scale/offset constraints")
    return {"a": scale, "b": offsets, "loss": float(result.fun),
            "success": bool(result.success), "message": str(result.message),
            "n_iter": int(result.nit)}


def apply_calibrated_verdict(cosines: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return np.asarray(model["a"] * np.asarray(cosines) + model["b"], dtype=np.float64)


def fit_probe(x: np.ndarray, labels: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000))
    model.fit(np.asarray(x, dtype=np.float32), np.asarray(labels, dtype=np.int64))
    return model


def _score_probe(model: Any, x: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(np.asarray(x, dtype=np.float32)), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    classes = list(model[-1].classes_)
    if classes != list(range(len(STYLES))):
        full = np.full((len(scores), len(STYLES)), -1e6, dtype=np.float64)
        for column, cls in enumerate(classes):
            full[:, int(cls)] = scores[:, column]
        scores = full
    return scores


def choose_intermediate_layer(activations: np.ndarray, labels: np.ndarray,
                              speakers: np.ndarray, *, n_layers: int = N_LAYERS,
                              width: int = UNITS_PER_LAYER,
                              ba: Callable[[np.ndarray, np.ndarray], float | None] = balanced_accuracy,
                              n_classes: int = len(STYLES)) -> tuple[int, list[float | None]]:
    """Choose a layer by inner leave-one-speaker-out BA on outer training speakers.

    `ba` and `n_classes` default to this module's seven styles; a part with another class count passes its
    own (Part 203 passes six emotions).
    """
    values = np.asarray(activations)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(speakers).astype(str)
    unique = sorted(np.unique(groups))
    if len(unique) < 2 or values.shape[1] != n_layers * width:
        raise ValueError("inner layer selection needs >=2 speakers and 24 x 2048 activations")
    means: list[float | None] = []
    for layer in range(n_layers):
        scores = []
        cols = slice(layer * width, (layer + 1) * width)
        for valid_speaker in unique:
            train = groups != valid_speaker
            valid = groups == valid_speaker
            if not np.all(np.bincount(y[train], minlength=n_classes) > 0):
                continue
            model = fit_probe(values[train, cols], y[train])
            pred = model.predict(np.asarray(values[valid, cols], dtype=np.float32))
            score = ba(y[valid], pred)
            if score is not None:
                scores.append(score)
        means.append(float(np.mean(scores)) if scores else None)
    usable = [index for index, value in enumerate(means) if value is not None]
    if not usable:
        raise ValueError("no valid inner speaker fold for intermediate layer selection")
    best = max(usable, key=lambda index: (means[index], -index))
    return int(best), means


def select_unit_per_style(activations: np.ndarray, labels: np.ndarray,
                          *, block_size: int = 1024) -> list[dict[str, Any]]:
    """Select the max-|AUC-.5| unit per style; stable argmax gives the low index tie."""
    x = np.asarray(activations)
    y = np.asarray(labels, dtype=np.int64)
    choices = []
    for cls, style in enumerate(STYLES):
        aucs = np.empty(x.shape[1], dtype=np.float32)
        binary = (y == cls).astype(np.int8)
        for start in range(0, x.shape[1], block_size):
            stop = min(start + block_size, x.shape[1])
            aucs[start:stop] = auc_per_unit(x[:, start:stop].astype(np.float32), binary)
        deviations = np.abs(aucs - 0.5)
        deviations[~np.isfinite(deviations)] = -np.inf
        index = int(np.argmax(deviations))
        raw_auc = float(aucs[index])
        sign = 1 if raw_auc >= 0.5 else -1
        choices.append({"style": style, "index": index, "layer": index // UNITS_PER_LAYER,
                        "neuron": index % UNITS_PER_LAYER, "sign": sign,
                        "train_auc_raw": raw_auc,
                        "train_auc_oriented": raw_auc if sign > 0 else 1.0 - raw_auc})
    return choices


def headline_outcome(point_difference: float | None,
                     interval: dict[str, float | None] | None) -> str:
    """Apply the registered 196.4 three-way rule to D minus B balanced accuracy."""
    if point_difference is None or interval is None:
        return "inconclusive"
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high):
        return "inconclusive"
    if point_difference >= 0.05 and low > 0.0:
        return "gap supported"
    if low >= -0.03 and high <= 0.03:
        return "practically equivalent"
    return "inconclusive"


def reverse_paired_difference(mean: float | None,
                              interval: dict[str, Any]) -> dict[str, Any]:
    """Reverse a paired difference and its confidence limits without losing the sign."""
    low, high = interval.get("low"), interval.get("high")
    reversed_interval = {"low": -high if high is not None else None,
                         "high": -low if low is not None else None}
    if "n_boot" in interval:
        reversed_interval["n_boot"] = interval["n_boot"]
    return {"mean_balanced_accuracy_difference": -mean if mean is not None else None,
            "text_cluster_95": reversed_interval}


def reading_conditions(d_interval: dict[str, Any],
                       b56_minus_d_interval: dict[str, Any]) -> dict[str, Any]:
    """Compute the two registered stronger-reading conditions on their stated scales."""
    chance = 1.0 / 7.0
    d_low = d_interval.get("low")
    b56_d_low = b56_minus_d_interval.get("low")
    return {
        "d_above_chance": {"threshold": chance, "lower_95_bound": d_low,
                           "condition_passed": bool(d_low is not None and d_low > chance)},
        "b56_minus_d_non_inferiority": {
            "lower_bound_threshold": -0.03,
            "lower_95_bound": b56_d_low,
            "condition_passed": bool(b56_d_low is not None and b56_d_low > -0.03)},
    }


def _validate_work_identity(work_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Refuse analysis of work from another manifest or an incomplete recorder identity."""
    identity_path = work_dir / "identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"recorder identity is missing: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    required = {"model_id", "model_snapshot_revision", "weight_sha256", "transformers_version",
                "sentence_transformers_version", "torch_version", "manifest_sha256"}
    if not required.issubset(identity):
        raise ValueError(f"recorder identity is incomplete: {sorted(required - set(identity))}")
    if identity["model_id"] != MODEL_ID or identity["model_snapshot_revision"] != MODEL_REVISION:
        raise ValueError("recorder identity does not match the registered model snapshot")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity["weight_sha256"])):
        raise ValueError("recorder identity has an invalid model weight hash")
    if identity["manifest_sha256"] != manifest_sha(rows):
        raise ValueError("recorder identity manifest differs from the analysis manifest")
    return identity


def _read_records(rows: list[dict[str, str]], work_dir: Path
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n = len(rows)
    activations = np.full((n, N_UNITS), np.nan, dtype=np.float16)
    embeddings = np.full((n, EMBEDDING_SIZE), np.nan, dtype=np.float32)
    acoustic = np.full((n, 7), np.nan, dtype=np.float32)
    duration = np.full(n, np.nan, dtype=np.float32)
    status = np.full(n, "missing", dtype="U32")
    errors = ["record missing"] * n
    for index, row in enumerate(rows):
        path = work_dir / "records" / f"{row['id']}.npz"
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as saved:
                status[index] = str(saved["status"].item())
                if "duration_seconds" in saved:
                    duration[index] = float(saved["duration_seconds"].item())
                if status[index] == "ok":
                    activations[index] = saved["act_mean"]
                    embeddings[index] = saved["emb"]
                    acoustic[index] = saved["acoustic"]
                    errors[index] = ""
                else:
                    errors[index] = str(saved["reason"].item()) if "reason" in saved else str(status[index])
        except (OSError, ValueError, KeyError, EOFError) as exc:
            status[index] = "record_error"
            errors[index] = f"{type(exc).__name__}: {exc}"
    return activations, embeddings, acoustic, duration, status, errors


def _valid_readouts(labels: np.ndarray, predictions: np.ndarray,
                    scores: np.ndarray) -> dict[str, Any]:
    ba = balanced_accuracy(labels, predictions)
    auc, per_style_auc = macro_ovr_auc(labels, scores)
    return {"balanced_accuracy": ba, "macro_ovr_auc": auc,
            "per_style_auc": per_style_auc,
            "per_style_recall": recall_by_class(labels, predictions)}


def _balanced_bootstrap(labels: np.ndarray, pred: np.ndarray) -> float:
    return balanced_accuracy(labels, pred)


def _resample_text_clusters(texts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    text_values = np.asarray(texts).astype(str)
    unique = np.unique(text_values)
    selected = rng.choice(unique, size=len(unique), replace=True)
    groups = {text: np.flatnonzero(text_values == text) for text in unique}
    return np.concatenate([groups[str(text)] for text in selected])


def _percentile_interval(values: np.ndarray) -> dict[str, Any]:
    usable = np.asarray(values, dtype=np.float64)
    usable = usable[np.isfinite(usable)]
    if not usable.size:
        return {"low": None, "high": None, "n_boot": 0}
    low, high = np.percentile(usable, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n_boot": int(len(usable))}


def _metric_bootstrap_with_pairs(fold_outputs: dict[str, dict[str, Any]], *,
                                 n_boot: int = BOOTSTRAPS, seed: int = SEED
                                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Memory-light cluster bootstrap, including paired four-speaker BA differences."""
    methods = list(next(iter(fold_outputs.values()))["predictions"])
    metric_draws = {method: {"balanced_accuracy": np.full(n_boot, np.nan),
                             "macro_ovr_auc": np.full(n_boot, np.nan)} for method in methods}
    pairs = {"D_minus_B": ("D", "B"), "B_minus_A": ("B", "A"),
             "D_minus_A": ("D", "A"), "D_minus_B56": ("D", "B56")}
    pair_draws = {name: np.full(n_boot, np.nan) for name in pairs}
    rng = np.random.default_rng(seed)
    fold_names = sorted(fold_outputs)
    for draw in range(n_boot):
        values_ba = {method: [] for method in methods}
        values_auc = {method: [] for method in methods}
        values_pair = {name: [] for name in pairs}
        for fold in fold_names:
            data = fold_outputs[fold]
            indexes = _resample_text_clusters(data["texts"], rng)
            labels = data["labels"][indexes]
            fold_ba = {}
            for method in methods:
                prediction = data["predictions"][method][indexes]
                scores = data["scores"][method][indexes]
                ba = None if method == "F" else balanced_accuracy(labels, prediction)
                auc, _ = macro_ovr_auc(labels, scores)
                fold_ba[method] = ba
                if ba is not None:
                    values_ba[method].append(ba)
                if auc is not None:
                    values_auc[method].append(auc)
            for name, (left, right) in pairs.items():
                if fold_ba[left] is not None and fold_ba[right] is not None:
                    values_pair[name].append(fold_ba[left] - fold_ba[right])
        for method in methods:
            if len(values_ba[method]) == len(fold_names):
                metric_draws[method]["balanced_accuracy"][draw] = float(np.mean(values_ba[method]))
            if len(values_auc[method]) == len(fold_names):
                metric_draws[method]["macro_ovr_auc"][draw] = float(np.mean(values_auc[method]))
        for name in pairs:
            if len(values_pair[name]) == len(fold_names):
                pair_draws[name][draw] = float(np.mean(values_pair[name]))
    intervals = {method: {metric: _percentile_interval(values) for metric, values in metrics.items()}
                 for method, metrics in metric_draws.items()}
    return intervals, {name: _percentile_interval(values) for name, values in pair_draws.items()}


def _speaker_identity_diagnostic(embeddings: np.ndarray, rows: list[dict[str, str]],
                                 active: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    speakers = sorted({rows[index]["speaker_id"] for index in active})
    rng = np.random.default_rng(seed)
    train_indexes, test_indexes = [], []
    for speaker in speakers:
        speaker_indexes = np.asarray([index for index in active if rows[index]["speaker_id"] == speaker], dtype=int)
        texts = np.unique([rows[index]["text_key"] for index in speaker_indexes])
        if len(texts) < 2:
            return {"verdict": "not_computable", "reason": "too few text clusters per speaker"}
        shuffled = rng.permutation(texts)
        split = max(1, len(shuffled) // 2)
        train_texts, test_texts = set(shuffled[:split]), set(shuffled[split:])
        train_indexes.extend(index for index in speaker_indexes if rows[index]["text_key"] in train_texts)
        test_indexes.extend(index for index in speaker_indexes if rows[index]["text_key"] in test_texts)
    index_to_speaker = {speaker: i for i, speaker in enumerate(speakers)}
    y_train = np.asarray([index_to_speaker[rows[index]["speaker_id"]] for index in train_indexes])
    y_test = np.asarray([index_to_speaker[rows[index]["speaker_id"]] for index in test_indexes])
    model = fit_probe(embeddings[train_indexes], y_train)
    prediction = model.predict(embeddings[test_indexes])
    value = balanced_accuracy(y_test, prediction, n_classes=len(speakers))
    return {"balanced_accuracy": value, "n_train": len(train_indexes), "n_test": len(test_indexes),
            "speaker_labels": speakers,
            "split": "half of each speaker's text clusters for training, other half for test"}


def _shuffle_diagnostic(train_x: np.ndarray, train_y: np.ndarray, train_speakers: np.ndarray,
                        test_x: np.ndarray, test_y: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shuffled = train_y.copy()
    for speaker in np.unique(train_speakers):
        indexes = np.flatnonzero(train_speakers == speaker)
        shuffled[indexes] = rng.permutation(shuffled[indexes])
    model = fit_probe(train_x, shuffled)
    scores = _score_probe(model, test_x)
    predictions = scores.argmax(axis=1)
    auc, per_style_auc = macro_ovr_auc(test_y, scores)
    return {"balanced_accuracy": balanced_accuracy(test_y, predictions),
            "macro_ovr_auc": auc, "per_style_auc": per_style_auc,
            "shuffled_within_each_training_speaker": True}


def _early_result(manifest_rows: list[dict[str, str]], gate1: dict[str, Any],
                  gate2: dict[str, Any], out_dir: Path, reason: str,
                  style_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {"study": "part196", "status": "gates_failed", "n_manifest_rows": len(manifest_rows),
              "gate1": gate1, "gate2": gate2, "readouts": {}, "paired_differences": {},
              "diagnostics": {}, "predictions": {}}
    if style_profile is not None:
        result["selected_style_profile"] = style_profile
    result["reading_rule"] = {"headline_196_4": "not_read", "reason": reason,
                              "conditions": {"d_above_chance": "not_read",
                                             "b56_minus_d_non_inferiority": "not_read"}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n", encoding="utf-8")
    return result


def analyze(*, manifest_path: Path, work_dir: Path, out_dir: Path,
            data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    rows = _read_rows(manifest_path)
    _validate_work_identity(work_dir, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    activations, embeddings, acoustic, _duration, record_status, record_errors = _read_records(rows, work_dir)
    durations_path = work_dir / "durations.json"
    durations = json.loads(durations_path.read_text(encoding="utf-8")) if durations_path.exists() else {}
    length_ids = {clip_id for clip_id, item in durations.items() if item.get("status") == "included"}
    if not length_ids:
        length_ids = {row["id"] for index, row in enumerate(rows)
                      if record_status[index] == "ok" and np.isfinite(_duration[index]) and
                      2.0 <= float(_duration[index]) <= 10.0}
    active = np.asarray([i for i, row in enumerate(rows)
                         if row["id"] in length_ids and record_status[i] == "ok"], dtype=int)
    successful_ids = {rows[index]["id"] for index in active}
    gate1 = gate1_check_recorded(rows, successful_ids)
    gate1["speaker_count"] = len({row["speaker_id"] for row in rows})
    gate1["recorded_in_range_count"] = len(successful_ids)
    if gate1["speaker_count"] != 4:
        gate1["verdict"] = "failed"
        gate1["reason"] = f"expected four held-out speakers, found {gate1['speaker_count']}"
    gate2_path = work_dir / "gate2.json"
    gate2 = json.loads(gate2_path.read_text(encoding="utf-8")) if gate2_path.exists() else {
        "verdict": "not_run", "clips": []}
    statuses = Counter(str(status) for status in record_status)
    coverage = {"status_counts": dict(sorted(statuses.items())),
                "recorded_ok": int(np.sum(record_status == "ok")), "planned": len(rows),
                "errors": [{"id": rows[i]["id"], "status": str(record_status[i]),
                            "reason": record_errors[i]} for i in range(len(rows))
                           if record_status[i] not in ("ok", "length_excluded")]}
    if gate1["verdict"] != "passed" or gate2.get("verdict") != "passed":
        return _early_result(rows, gate1, gate2, out_dir,
                             "Gate 1 and Gate 2 must pass before any readout is computed")

    if not len(active):
        return _early_result(rows, gate1, gate2, out_dir, "no successful in-range recordings")
    prompt_path = work_dir / "prompts196.npz"
    transcript_path = work_dir / "transcripts196.npz"
    if not prompt_path.exists() or not transcript_path.exists():
        raise FileNotFoundError("recorder must create prompts196.npz and transcripts196.npz")
    with np.load(prompt_path, allow_pickle=False) as prompt_doc:
        if tuple(str(value) for value in prompt_doc["styles"]) != STYLES:
            raise ValueError("prompt style order differs from the registered order")
        prompt_embeddings = np.asarray(prompt_doc["embeddings"], dtype=np.float32)
    with np.load(transcript_path, allow_pickle=False) as text_doc:
        text_index = {str(key): i for i, key in enumerate(text_doc["text_keys"])}
        transcript_embeddings = np.asarray(text_doc["embeddings"], dtype=np.float32)
    missing_texts = sorted({rows[i]["text_key"] for i in active} - set(text_index))
    if missing_texts:
        raise ValueError(f"transcript embeddings missing for {len(missing_texts)} active texts")

    labels_by_row = np.asarray([STYLES.index(row["style"]) for row in rows], dtype=np.int64)
    speakers = sorted({rows[index]["speaker_id"] for index in active})
    if len(speakers) != 4:
        return _early_result(rows, gate1, gate2, out_dir, "active recordings do not cover four speakers")
    style_profile = selected_style_profile(rows, successful_ids, read_metadata(data_dir))
    fold_outputs: dict[str, dict[str, Any]] = {}
    fold_summaries: dict[str, Any] = {}
    all_methods = ("A", "B", "B56", "C", "D", "E", "F", "H", "T")
    n_bootstrap = BOOTSTRAPS
    diagnostics_shuffle = {}

    for fold_index, heldout in enumerate(speakers):
        test = np.asarray([i for i in active if rows[i]["speaker_id"] == heldout], dtype=int)
        train = np.asarray([i for i in active if rows[i]["speaker_id"] != heldout], dtype=int)
        train_y, test_y = labels_by_row[train], labels_by_row[test]
        if len(np.unique(train_y)) != 7 or len(np.unique(test_y)) != 7:
            return _early_result(rows, gate1, gate2, out_dir,
                                 f"fold {heldout} lacks one of the seven styles", style_profile)
        train_embeddings = embeddings[train]
        test_embeddings = embeddings[test]
        all_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
        all_prompt_cosines = np.einsum("nd,kjd->nkj", all_embeddings, prompt_embeddings)
        train_cosines = all_prompt_cosines[:len(train)]
        test_cosines = all_prompt_cosines[len(train):]

        # A: argmax over the first prewritten phrasing per style.
        a_test_scores = test_cosines[:, :, 0]
        a_test_pred = a_test_scores.argmax(axis=1)
        train_selected, selection = select_b_phrasings(train_cosines, train_y)
        test_selected = select_b_phrasings_from_details(test_cosines, selection)

        # B: one scale and sum-zero offsets fitted to the train-selected cosines.
        b_fit = fit_calibrated_verdict(train_selected, train_y)
        b_test_scores = apply_calibrated_verdict(test_selected, b_fit)
        b_test_pred = b_test_scores.argmax(axis=1)

        # B56: the regularized multinomial probe on all label phrase cosines.
        train_56 = train_cosines.reshape(len(train), -1)
        test_56 = test_cosines.reshape(len(test), -1)
        b56_model = fit_probe(train_56, train_y)
        b56_test_scores = _score_probe(b56_model, test_56)
        b56_test_pred = b56_test_scores.argmax(axis=1)

        # C: nearest training-style centroid in final embedding cosine space.
        centroids = []
        for cls in range(7):
            center = train_embeddings[train_y == cls].mean(axis=0)
            norm = np.linalg.norm(center)
            centroids.append(center / norm if norm else center)
        centroid_matrix = np.asarray(centroids, dtype=np.float32)
        c_test_scores = test_embeddings @ centroid_matrix.T
        c_test_pred = c_test_scores.argmax(axis=1)

        # D: final embedding multinomial probe.
        d_model = fit_probe(train_embeddings, train_y)
        d_test_scores = _score_probe(d_model, test_embeddings)
        d_test_pred = d_test_scores.argmax(axis=1)

        # E: choose one MLP-input layer inside the three-speaker training set.
        layer, inner_layer_ba = choose_intermediate_layer(
            activations[train], train_y,
            np.asarray([rows[i]["speaker_id"] for i in train]), n_layers=N_LAYERS,
            width=UNITS_PER_LAYER)
        cols = slice(layer * UNITS_PER_LAYER, (layer + 1) * UNITS_PER_LAYER)
        e_model = fit_probe(activations[train, cols], train_y)
        e_test_scores = _score_probe(e_model, activations[test, cols])
        e_test_pred = e_test_scores.argmax(axis=1)

        # F: independently select one oriented unit for each style on training only.
        selected_units = select_unit_per_style(activations[train], train_y)
        f_test_scores = np.column_stack([
            activations[test, entry["index"]].astype(np.float64) * entry["sign"]
            for entry in selected_units])
        f_test_pred = f_test_scores.argmax(axis=1)

        # H: the precomputed acoustic vector, fit with the same balanced multinomial probe.
        h_model = fit_probe(acoustic[train], train_y)
        h_test_scores = _score_probe(h_model, acoustic[test])
        h_test_pred = h_test_scores.argmax(axis=1)

        # T: transcript SearchDocument embeddings, fit with the same probe recipe.
        train_texts = [rows[i]["text_key"] for i in train]
        test_texts = [rows[i]["text_key"] for i in test]
        train_text_emb = transcript_embeddings[[text_index[key] for key in train_texts]]
        test_text_emb = transcript_embeddings[[text_index[key] for key in test_texts]]
        t_model = fit_probe(train_text_emb, train_y)
        t_test_scores = _score_probe(t_model, test_text_emb)
        t_test_pred = t_test_scores.argmax(axis=1)

        predictions = {"A": a_test_pred, "B": b_test_pred, "B56": b56_test_pred,
                       "C": c_test_pred, "D": d_test_pred, "E": e_test_pred,
                       "F": f_test_pred, "H": h_test_pred, "T": t_test_pred}
        score_matrices = {"A": a_test_scores, "B": b_test_scores, "B56": b56_test_scores,
                          "C": c_test_scores, "D": d_test_scores, "E": e_test_scores,
                          "F": f_test_scores, "H": h_test_scores, "T": t_test_scores}
        fold_outputs[heldout] = {"labels": test_y, "texts": np.asarray(test_texts),
                                 "predictions": predictions, "scores": score_matrices}
        fold_readouts = {}
        for method in all_methods:
            metric = _valid_readouts(test_y, predictions[method], score_matrices[method])
            if method == "F":
                metric["balanced_accuracy"] = None
                metric["per_style_recall"] = None
            fold_readouts[method] = metric
        fold_summaries[heldout] = {
            "n_train": int(len(train)), "n_test": int(len(test)),
            "train_style_counts": {STYLES[cls]: int(np.sum(train_y == cls)) for cls in range(7)},
            "test_style_counts": {STYLES[cls]: int(np.sum(test_y == cls)) for cls in range(7)},
            "readouts": fold_readouts,
            "B": {"fit": b_fit, "phrasing_selection": selection},
            "E": {"selected_layer": layer, "inner_leave_one_speaker_out_balanced_accuracy": inner_layer_ba},
            "F": {"selected_units": selected_units},
        }
        shuffled_train_x = train_embeddings
        diagnostics_shuffle[heldout] = _shuffle_diagnostic(
            shuffled_train_x, train_y, np.asarray([rows[i]["speaker_id"] for i in train]),
            test_embeddings, test_y, SEED + fold_index)

    # Four-speaker summaries and matched text-cluster intervals.
    intervals, paired_intervals = _metric_bootstrap_with_pairs(
        fold_outputs, n_boot=n_bootstrap, seed=SEED)
    readout_results = {}
    for method in all_methods:
        per_speaker = {speaker: fold_summaries[speaker]["readouts"][method] for speaker in speakers}
        ba_values = [per_speaker[speaker]["balanced_accuracy"] for speaker in speakers]
        auc_values = [per_speaker[speaker]["macro_ovr_auc"] for speaker in speakers]
        mean_ba = float(np.mean(ba_values)) if all(value is not None for value in ba_values) else None
        mean_auc = float(np.mean(auc_values)) if all(value is not None for value in auc_values) else None
        recall_means = []
        for cls in range(7):
            vals = [per_speaker[speaker]["per_style_recall"][cls]
                    for speaker in speakers if per_speaker[speaker]["per_style_recall"] is not None]
            recall_means.append(float(np.mean(vals)) if len(vals) == len(speakers) and
                                all(value is not None for value in vals) else None)
        readout_results[method] = {"mean_balanced_accuracy": mean_ba,
                                   "mean_macro_ovr_auc": mean_auc,
                                   "balanced_accuracy_text_cluster_95": intervals[method]["balanced_accuracy"],
                                   "macro_ovr_auc_text_cluster_95": intervals[method]["macro_ovr_auc"],
                                   "mean_per_style_recall": (dict(zip(STYLES, recall_means))
                                                              if method != "F" else None),
                                   "per_speaker": per_speaker}

    diff_means = {}
    for name, left, right in (("D_minus_B", "D", "B"), ("B_minus_A", "B", "A"),
                              ("D_minus_A", "D", "A"), ("D_minus_B56", "D", "B56")):
        left_value = readout_results[left]["mean_balanced_accuracy"]
        right_value = readout_results[right]["mean_balanced_accuracy"]
        diff_means[name] = {"mean_balanced_accuracy_difference":
                            (left_value - right_value if left_value is not None and right_value is not None else None),
                            "text_cluster_95": paired_intervals[name]}
    diff_means["B56_minus_D"] = reverse_paired_difference(
        diff_means["D_minus_B56"]["mean_balanced_accuracy_difference"],
        diff_means["D_minus_B56"]["text_cluster_95"])

    headline_interval = paired_intervals["D_minus_B"]
    headline_point = diff_means["D_minus_B"]["mean_balanced_accuracy_difference"]
    headline = headline_outcome(headline_point, headline_interval)
    conditions = reading_conditions(
        readout_results["D"]["balanced_accuracy_text_cluster_95"],
        diff_means["B56_minus_D"]["text_cluster_95"])
    d_minus_a = diff_means["D_minus_A"]
    b_minus_a = diff_means["B_minus_A"]
    d_minus_a_interval = d_minus_a["text_cluster_95"]
    calibration = {"B_minus_A": b_minus_a,
                   "credited": bool(b_minus_a["mean_balanced_accuracy_difference"] is not None and
                                   b_minus_a["text_cluster_95"]["low"] is not None and
                                   b_minus_a["text_cluster_95"]["low"] > 0)}
    d_a_point = d_minus_a["mean_balanced_accuracy_difference"]
    b_a_point = b_minus_a["mean_balanced_accuracy_difference"]
    d_ba_low = d_minus_a_interval.get("low")
    d_beats_a = d_ba_low is not None and d_ba_low > 0
    pred1963 = ("not_applicable" if not d_beats_a else
                ("held" if d_a_point is not None and b_a_point is not None and
                 b_a_point >= 0.5 * d_a_point else "not_held"))
    d_mean = readout_results["D"]["mean_balanced_accuracy"]
    b56_mean = readout_results["B56"]["mean_balanced_accuracy"]
    whisper_highest = all(
        readout_results[method]["mean_per_style_recall"]["whisper"] is not None and
        all(readout_results[method]["mean_per_style_recall"]["whisper"] >
            readout_results[method]["mean_per_style_recall"][style]
            for style in STYLES if style != "whisper")
        for method in ("A", "B", "D"))
    predictions = {
        "196.1": {"prediction": "A mean balanced accuracy <= 0.35",
                  "value": readout_results["A"]["mean_balanced_accuracy"],
                  "verdict": "held" if readout_results["A"]["mean_balanced_accuracy"] is not None and
                  readout_results["A"]["mean_balanced_accuracy"] <= 0.35 else "not_held"},
        "196.2": {"prediction": "D mean balanced accuracy >= 0.70",
                  "value": d_mean,
                  "verdict": "held" if d_mean is not None and d_mean >= 0.70 else "not_held"},
        "196.3": {"prediction": "when D beats A, B closes at least half the gap",
                  "D_minus_A_interval": d_minus_a_interval,
                  "B_minus_A": b_minus_a, "D_minus_A": d_minus_a,
                  "verdict": pred1963},
        "196.4": {"prediction": "D minus B >= 0.05 and interval above zero",
                  "D_minus_B": diff_means["D_minus_B"], "outcome": headline,
                  "verdict": headline},
        "196.5": {"prediction": "B56 within 0.03 balanced accuracy of D",
                  "D_minus_B56": diff_means["D_minus_B56"],
                  "B56_minus_D": diff_means["B56_minus_D"],
                  "verdict": "held" if d_mean is not None and b56_mean is not None and
                  abs(d_mean - b56_mean) <= 0.03 else "not_held"},
        "196.6": {"prediction": "whisper has highest mean per-style recall under A, B and D",
                  "per_style_recall": {method: readout_results[method]["mean_per_style_recall"]
                                       for method in ("A", "B", "D")},
                  "verdict": "held" if whisper_highest else "not_held"},
        "196.7": {"prediction": "H mean balanced accuracy >= 0.40",
                  "value": readout_results["H"]["mean_balanced_accuracy"],
                  "verdict": "held" if readout_results["H"]["mean_balanced_accuracy"] is not None and
                  readout_results["H"]["mean_balanced_accuracy"] >= 0.40 else "not_held"},
        "196.8": {"prediction": "T mean balanced accuracy <= 0.25",
                  "value": readout_results["T"]["mean_balanced_accuracy"],
                  "verdict": "held" if readout_results["T"]["mean_balanced_accuracy"] is not None and
                  readout_results["T"]["mean_balanced_accuracy"] <= 0.25 else "not_held"},
    }
    diagnostics = {
        "speaker_identity_final_embedding": _speaker_identity_diagnostic(embeddings, rows, active),
        "shuffled_style_labels_within_training_speakers": {
            "per_heldout_speaker": diagnostics_shuffle,
            "mean_balanced_accuracy": float(np.mean([
                item["balanced_accuracy"] for item in diagnostics_shuffle.values()
                if item["balanced_accuracy"] is not None]))},
        "recording_coverage": coverage,
    }
    result = {"study": "part196", "status": "complete", "n_manifest_rows": len(rows),
              "n_in_range_rows": len(length_ids), "n_successfully_recorded_in_range": len(active),
              "styles": list(STYLES), "gate1": gate1, "gate2": gate2,
              "selected_style_profile": style_profile,
              "folds": fold_summaries, "readouts": readout_results,
              "paired_differences": diff_means, "diagnostics": diagnostics,
              "calibration_credit": calibration,
              "reading_rule": {"headline_196_4": headline,
                               "D_minus_B_interval": headline_interval,
                               "equivalence_bounds": [-0.03, 0.03],
                               "conditions": conditions,
                               "calibration_credited_only_if_B_minus_A_interval_above_zero": True,
                               "auc_computed_within_each_heldout_speaker": True},
              "predictions": predictions}
    result = _jsonable(result)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n", encoding="utf-8")
    return result


def select_b_phrasings_from_details(cosines: np.ndarray,
                                    details: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray(cosines, dtype=np.float64)
    selected = np.empty((len(values), len(STYLES)), dtype=np.float64)
    for cls, item in enumerate(details):
        if item["choice"] == "mean":
            selected[:, cls] = values[:, cls, :].mean(axis=1)
        else:
            selected[:, cls] = values[:, cls, int(item["phrase_index"])]
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest196.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Expresso parquet directory; metadata only, no audio is read")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/part196-work"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "part196")
    args = parser.parse_args(argv)
    result = analyze(manifest_path=args.manifest, work_dir=args.work_dir, out_dir=args.out_dir,
                     data_dir=args.data_dir)
    print(json.dumps({"results": str(args.out_dir / "results.json"),
                      "status": result["status"],
                      "gate_verdicts": {"gate1": result["gate1"]["verdict"],
                                        "gate2": result["gate2"]["verdict"]},
                      "reading_conditions": result.get("reading_rule", {}).get("conditions", {
                          "d_above_chance": "not_read",
                          "b56_minus_d_non_inferiority": "not_read"})}, sort_keys=True))
    return 0 if result["gate1"]["verdict"] == "passed" and result["gate2"]["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
