#!/usr/bin/env python3
"""Part 190: low-rank linear edits of EmbeddingGemma 2's output embedding (registered in PART190-DESIGN.md).

Pure offline analysis on Part 188's committed arrays. Every number in results/part190/results.json comes from
`analyze`; nothing here prints an array. The probe fitter, the AUC and the Chicken cluster labels are Part 188's.

Written by Claude (Opus 5.5), 2026-11-05, in a cloud session that did not run it on the real data.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # `python -I` leaves the repository off sys.path (PROCEDURE.md, Part 136)
    sys.path.insert(0, str(ROOT))

from part188.analyze188 import auc_per_unit, fit_logistic_l2, logistic_scores  # noqa: E402

DOSES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
RANKS = (1, 2, 4, 8)
N_CONTROLS = 20
NEIGHBOURS = 10
BOOTSTRAPS = 2000
DOSE_TOLERANCE = 0.01          # operating dose: smallest alpha within 0.01 of the fit-fold ladder maximum
GATE2_TOLERANCE = 0.005
GATE3_MEDIAN_MAX = 0.02
GATE3_P95_MAX = 0.05
GATE4_MIN_ACCURACY = 0.30
DIRECTIONS = (("A_to_B", "A", "B"), ("B_to_A", "B", "A"))
OFF_TARGET_SETS = ("E", "H", "N")
ESC_SETS = ("E", "H")
REAL_RECORDING_SETS = ("R", "C_R", "E", "H")
FREE_PHRASE = "an animal"


# --------------------------------------------------------------------------------------------- data

@dataclass
class Data:
    ids: np.ndarray                 # [n] str
    x: np.ndarray                   # [n, d] float64, unit norm
    analysis_set: np.ndarray        # [n] str
    fold: np.ndarray                # [n] str ('A', 'B', 'C', '')
    group: np.ndarray               # [n] str (animal group; ESC category for others)
    phrase: np.ndarray              # [n] str (prompt phrase of the clip's class)
    esc_category: np.ndarray        # [n] str
    chicken_cluster: np.ndarray     # [n] str (Part 188's four Chicken classes; '' elsewhere)
    prompt_phrases: list[str]
    real_prompts: np.ndarray        # [p, d] float64, as stored
    imitation_prompts: np.ndarray   # [p, d]
    raw_norms: np.ndarray = field(default=None)  # [n] norms before renormalization (Gate 1)

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def where(self, sets: tuple[str, ...] | str, fold: str | None = None) -> np.ndarray:
        sets = (sets,) if isinstance(sets, str) else sets
        mask = np.isin(self.analysis_set, sets)
        if fold is not None:
            mask &= self.fold == fold
        return np.flatnonzero(mask)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("an embedding has zero norm")
    return x / norms


def _chicken_clusters(rows: list[dict[str, str]]) -> np.ndarray:
    from part188.analyze188 import _chicken_cluster_groups
    out = np.full(len(rows), "", dtype=object)
    idx = np.asarray([i for i, r in enumerate(rows) if r["analysis_set"] in ("C_R", "C_I")], dtype=int)
    if idx.size:
        out[idx] = _chicken_cluster_groups(rows, idx)
    return out.astype(str)


def data_from_arrays(rows: list[dict[str, str]], ids: np.ndarray, embeddings: np.ndarray,
                     prompt_phrases, real_prompts: np.ndarray, imitation_prompts: np.ndarray) -> Data:
    raw = np.asarray(embeddings, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != len(rows):
        raise ValueError("embeddings must be [rows, dim] matching the manifest")
    norms = np.linalg.norm(raw, axis=1)
    x = _normalize_rows(raw) if np.isfinite(raw).all() else raw
    col = lambda k: np.asarray([r.get(k, "") or "" for r in rows], dtype=str)  # noqa: E731
    group = np.asarray([r.get("group") or r.get("esc_category") or r.get("class_key", "") for r in rows], dtype=str)
    return Data(ids=np.asarray(ids).astype(str), x=x, analysis_set=col("analysis_set"), fold=col("fold"),
                group=group, phrase=col("phrase"), esc_category=col("esc_category"),
                chicken_cluster=_chicken_clusters(rows), prompt_phrases=[str(p) for p in prompt_phrases],
                real_prompts=np.asarray(real_prompts, dtype=np.float64),
                imitation_prompts=np.asarray(imitation_prompts, dtype=np.float64), raw_norms=norms)


def load_data(emb_path: Path, manifest_path: Path) -> tuple[Data, list[dict[str, str]]]:
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(emb_path, allow_pickle=False) as z:
        data = data_from_arrays(rows, z["ids"], z["embeddings"], z["prompt_phrases"],
                                z["prompt_real_embeddings"], z["prompt_imitation_embeddings"])
    return data, rows


# --------------------------------------------------------------------------------------------- AUC

def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        return float("nan")
    return float(auc_per_unit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), labels)[0])


def auc64(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUC in float64 (used for verdicts, where float32 rounding of close scores would add ties)."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


# --------------------------------------------------------------------------------------------- the edit

@dataclass
class Edit:
    mu: np.ndarray          # [d]
    read: np.ndarray        # [d, k] orthonormal read directions
    sigma: np.ndarray       # [k]
    write: np.ndarray       # [d, k] write directions (u_1 = v_bar, u_j = w_j)
    rho: float
    scale: float = 1.0      # displacement matching for controls

    def truncated(self, k: int) -> "Edit":
        return Edit(self.mu, self.read[:, :k], self.sigma[:k], self.write[:, :k], self.rho, self.scale)

    def with_scale(self, scale: float) -> "Edit":
        return Edit(self.mu, self.read, self.sigma, self.write, self.rho, float(scale))

    def displacement(self, x: np.ndarray, alpha: float) -> np.ndarray:
        z = (x - self.mu) @ self.read / self.sigma
        return alpha * self.rho * self.scale * (z @ self.write.T)

    def apply(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if alpha == 0:
            return _normalize_rows(x.copy())
        return _normalize_rows(x + self.displacement(x, alpha))

    def mean_displacement(self, x: np.ndarray, alpha: float) -> float:
        return float(np.linalg.norm(self.displacement(x, alpha), axis=1).mean())


def read_directions(x_fit: np.ndarray, y_fit: np.ndarray, k: int,
                    fitter: Callable = fit_logistic_l2) -> np.ndarray:
    """Repeated orthogonal probing: probe j is fitted with directions 1..j-1 projected out (INLP-style)."""
    y = np.asarray(y_fit, dtype=bool)
    d = x_fit.shape[1]
    directions: list[np.ndarray] = []
    for _ in range(k):
        basis = np.column_stack(directions) if directions else np.zeros((d, 0))
        projected = x_fit - (x_fit @ basis) @ basis.T
        model = fitter(projected, y.astype(np.float64))
        w = np.asarray(model["coef"], dtype=np.float64) / np.asarray(model["scale"], dtype=np.float64)
        w = w - basis @ (basis.T @ w)
        for _ in range(2):  # re-orthogonalize against rounding
            w = w - basis @ (basis.T @ w)
        norm = np.linalg.norm(w)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("a probe direction vanished")
        w /= norm
        if (x_fit[y] @ w).mean() < (x_fit[~y] @ w).mean():
            w = -w
        directions.append(w)
    return np.column_stack(directions)


def verdict_direction(data: Data, fit_idx: np.ndarray) -> np.ndarray:
    """v_bar: mean over the fit fold's distinct phrases of (real prompt - imitation prompt), normalized."""
    index = {p: i for i, p in enumerate(data.prompt_phrases)}
    phrases = sorted(set(data.phrase[fit_idx].tolist()))
    missing = [p for p in phrases if p not in index]
    if missing:
        raise KeyError(f"phrases missing from the committed prompts: {missing[:3]}")
    diffs = np.stack([data.real_prompts[index[p]] - data.imitation_prompts[index[p]] for p in phrases])
    v = diffs.mean(axis=0)
    return v / np.linalg.norm(v)


def build_edit(x_fit: np.ndarray, y_fit: np.ndarray, read: np.ndarray, v_bar: np.ndarray,
               centre: np.ndarray | None = None) -> Edit:
    y = np.asarray(y_fit, dtype=bool)
    mu = centre if centre is not None else 0.5 * (x_fit[y].mean(axis=0) + x_fit[~y].mean(axis=0))
    sigma = (x_fit @ read).std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    write = read.copy()
    write[:, 0] = v_bar
    rho = float((x_fit @ v_bar).std())
    return Edit(mu=mu, read=read, sigma=sigma, write=write, rho=rho if rho > 1e-12 else 1.0)


def random_edit(x_fit: np.ndarray, v_bar: np.ndarray, k: int, rng: np.random.Generator) -> Edit:
    q, r = np.linalg.qr(rng.standard_normal((x_fit.shape[1], k)))
    q = q * np.sign(np.diag(r))  # Haar
    q = q * rng.choice((-1.0, 1.0), size=k)
    return build_edit(x_fit, np.ones(len(x_fit), bool), q, v_bar, centre=x_fit.mean(axis=0))


def shuffle_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(y, dtype=bool).copy()
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        out[members] = out[rng.permutation(members)]
    return out


# --------------------------------------------------------------------------------------------- readouts

def verdict_scores(data: Data, x: np.ndarray, idx: np.ndarray, class_free: bool = False) -> np.ndarray:
    index = {p: i for i, p in enumerate(data.prompt_phrases)}
    if class_free:
        p = np.full(idx.size, index[FREE_PHRASE])
    else:
        missing = sorted(set(ph for ph in data.phrase[idx] if ph not in index))
        if missing:
            raise KeyError(f"phrases missing from the committed prompts: {missing[:3]}")
        p = np.asarray([index[ph] for ph in data.phrase[idx]])
    diff = data.real_prompts - data.imitation_prompts
    return np.einsum("ij,ij->i", x[idx], diff[p])


def _top_neighbours(sim: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest entries per row, sorted by similarity (descending, ties by index)."""
    k = min(k, sim.shape[1])
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, part, axis=1)
    order = np.lexsort((part, -vals), axis=1)
    return np.take_along_axis(part, order, axis=1)


def knn_class_correct(x: np.ndarray, labels: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
    """Leave-one-out k-NN majority vote; a tie goes to the tied class with the nearest member."""
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nb = _top_neighbours(sim, k)
    nb_labels = labels[nb]
    correct = np.empty(len(x), dtype=bool)
    for i in range(len(x)):
        classes, counts = np.unique(nb_labels[i], return_counts=True)
        tied = set(classes[counts == counts.max()].tolist())
        winner = next(lab for lab in nb_labels[i] if lab in tied)  # neighbours are in similarity order
        correct[i] = winner == labels[i]
    return correct


def knn_real_share(x: np.ndarray, y: np.ndarray, k: int = NEIGHBOURS) -> np.ndarray:
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    return np.asarray(y, dtype=float)[_top_neighbours(sim, k)].mean(axis=1)


@dataclass
class Context:
    """Index sets and unedited baselines for one direction."""
    name: str
    fit_idx: np.ndarray
    fit_y: np.ndarray
    test_idx: np.ndarray
    test_y: np.ndarray
    chicken_idx: np.ndarray
    chicken_y: np.ndarray
    esc_idx: np.ndarray
    off_idx: np.ndarray
    animal_real_idx: np.ndarray
    test_imitation_idx: np.ndarray
    real_pool_idx: np.ndarray
    base: dict[str, Any] = field(default_factory=dict)


def make_context(data: Data, name: str, fit_fold: str, test_fold: str) -> Context:
    fit_idx = data.where(("R", "I"), fit_fold)
    test_idx = data.where(("R", "I"), test_fold)
    chicken_idx = data.where(("C_R", "C_I"))
    return Context(name=name, fit_idx=fit_idx, fit_y=data.analysis_set[fit_idx] == "R",
                   test_idx=test_idx, test_y=data.analysis_set[test_idx] == "R",
                   chicken_idx=chicken_idx, chicken_y=data.analysis_set[chicken_idx] == "C_R",
                   esc_idx=data.where(ESC_SETS), off_idx=data.where(OFF_TARGET_SETS),
                   animal_real_idx=data.where(("R", "C_R")), test_imitation_idx=data.where("I", test_fold),
                   real_pool_idx=data.where(REAL_RECORDING_SETS))


def _phrase_prompt_matrix(data: Data) -> tuple[np.ndarray, list[str]]:
    keep = [i for i, p in enumerate(data.prompt_phrases) if p != FREE_PHRASE]
    real = _normalize_rows(data.real_prompts[keep])
    return real, [data.prompt_phrases[i] for i in keep]


def evaluate(data: Data, ctx: Context, x_edit: np.ndarray, per_clip: bool = False) -> dict[str, Any]:
    """Every readout and collateral measure for one edited embedding matrix."""
    out: dict[str, Any] = {}
    v_test = verdict_scores(data, x_edit, ctx.test_idx)
    out["V_test"] = auc64(ctx.test_y, v_test)
    out["V_test_free"] = auc64(ctx.test_y, verdict_scores(data, x_edit, ctx.test_idx, class_free=True))
    out["V_fit"] = auc64(ctx.fit_y, verdict_scores(data, x_edit, ctx.fit_idx))
    v_chicken = verdict_scores(data, x_edit, ctx.chicken_idx) if ctx.chicken_idx.size else np.zeros(0)
    out["V_chicken"] = auc64(ctx.chicken_y, v_chicken) if ctx.chicken_idx.size else float("nan")
    k_share = knn_real_share(x_edit[ctx.test_idx], ctx.test_y)
    out["K_test"] = auc64(ctx.test_y, k_share)
    esc_labels = data.esc_category[ctx.esc_idx]
    c1 = knn_class_correct(x_edit[ctx.esc_idx], esc_labels)
    out["C1_accuracy"] = float(c1.mean())
    # C2: neighbours of off-target clips among all clips
    sim = x_edit[ctx.off_idx] @ x_edit.T
    sim[np.arange(ctx.off_idx.size), ctx.off_idx] = -np.inf
    nb = _top_neighbours(sim, NEIGHBOURS)
    out["_C2_neighbours"] = nb
    out["C3_movement"] = float(1.0 - np.einsum("ij,ij->i", x_edit[ctx.off_idx], data.x[ctx.off_idx]).mean())
    prompts = _normalize_rows(np.vstack([data.real_prompts, data.imitation_prompts]))
    out["_C4_cos"] = x_edit[ctx.off_idx] @ prompts.T
    real_prompts, names = _phrase_prompt_matrix(data)
    top = np.argmax(x_edit[ctx.animal_real_idx] @ real_prompts.T, axis=1)
    c5 = np.asarray(names)[top] == data.phrase[ctx.animal_real_idx]
    out["C5_phrase_accuracy"] = float(c5.mean())
    pool = ctx.real_pool_idx
    nn = pool[np.argmax(x_edit[ctx.test_imitation_idx] @ x_edit[pool].T, axis=1)]
    c6 = data.analysis_set[nn] == "H"
    out["C6_human_share"] = float(c6.mean())
    if per_clip:
        out["_clip"] = {"v_test": v_test, "v_chicken": v_chicken, "k_share": k_share, "c1": c1, "c5": c5, "c6": c6}
    return out


def collateral_against_base(cur: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    nb, nb0 = cur["_C2_neighbours"], base["_C2_neighbours"]
    first_changed = float((nb[:, 0] != nb0[:, 0]).mean())
    overlap = np.asarray([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(nb, nb0)], dtype=float)
    return {
        "C1_change_points": 100.0 * (cur["C1_accuracy"] - base["C1_accuracy"]),
        "C2_first_neighbour_changed": first_changed,
        "C2_mean_replaced_of_10": float(1.0 - overlap.mean() / NEIGHBOURS),
        "C3_movement": cur["C3_movement"],
        "C4_mean_abs_cos_change": float(np.abs(cur["_C4_cos"] - base["_C4_cos"]).mean()),
        "C5_change_points": 100.0 * (cur["C5_phrase_accuracy"] - base["C5_phrase_accuracy"]),
        "C6_change_points": 100.0 * (cur["C6_human_share"] - base["C6_human_share"]),
    }


def summarize(cur: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    row = {k: cur[k] for k in ("V_test", "V_test_free", "V_fit", "V_chicken", "K_test", "C1_accuracy",
                               "C5_phrase_accuracy", "C6_human_share")}
    row["dV_test"] = cur["V_test"] - base["V_test"]
    row["dV_chicken"] = cur["V_chicken"] - base["V_chicken"]
    row["dK_test"] = cur["K_test"] - base["K_test"]
    row.update(collateral_against_base(cur, base))
    return row


def operating_dose(ladder: dict[float, dict[str, float]], tolerance: float = DOSE_TOLERANCE) -> float:
    """Smallest alpha whose fit-fold verdict AUC is at least the ladder maximum minus the tolerance."""
    values = {a: r["V_fit"] for a, r in ladder.items() if np.isfinite(r["V_fit"])}
    if not values:
        raise ValueError("no finite fit-fold verdict on the ladder")
    best = max(values.values())
    return min(a for a, v in values.items() if v >= best - tolerance)


# --------------------------------------------------------------------------------------------- intervals

def paired_cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters: np.ndarray, *, n_boot: int = BOOTSTRAPS,
                             seed: int = 190, labels: np.ndarray | None = None) -> dict[str, float | int]:
    """Resample clusters with replacement; `stat` takes the resampled row indices. A draw that leaves one class
    empty (when labels are given) is redrawn; the number of redraws is reported."""
    clusters = np.asarray(clusters).astype(str)
    unique = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in unique]
    rng = np.random.default_rng(seed)
    draws, redraws = [], 0
    while len(draws) < n_boot:
        chosen = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[c] for c in chosen])
        if labels is not None:
            lab = np.asarray(labels, dtype=bool)[idx]
            if lab.all() or not lab.any():
                redraws += 1
                if redraws > 50 * n_boot:
                    return {"low": float("nan"), "high": float("nan"), "n_boot": len(draws), "redraws": redraws}
                continue
        draws.append(stat(idx))
    low, high = np.percentile(np.asarray(draws, dtype=float), [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n_boot": int(n_boot), "redraws": int(redraws)}


def main_intervals(data: Data, ctx: Context, cur: dict[str, Any], base: dict[str, Any],
                   n_boot: int = BOOTSTRAPS, seed: int = 190) -> dict[str, Any]:
    c, b = cur["_clip"], base["_clip"]
    y = ctx.test_y
    out = {
        "dV_test": paired_cluster_bootstrap(
            lambda i: auc64(y[i], c["v_test"][i]) - auc64(y[i], b["v_test"][i]),
            data.group[ctx.test_idx], n_boot=n_boot, seed=seed, labels=y),
        "V_test": paired_cluster_bootstrap(lambda i: auc64(y[i], c["v_test"][i]), data.group[ctx.test_idx],
                                           n_boot=n_boot, seed=seed + 1, labels=y),
        "dK_test": paired_cluster_bootstrap(
            lambda i: auc64(y[i], c["k_share"][i]) - auc64(y[i], b["k_share"][i]),
            data.group[ctx.test_idx], n_boot=n_boot, seed=seed + 2, labels=y),
        "C1_change_points": paired_cluster_bootstrap(
            lambda i: 100.0 * (c["c1"][i].mean() - b["c1"][i].mean()), data.esc_category[ctx.esc_idx],
            n_boot=n_boot, seed=seed + 3),
        "C5_change_points": paired_cluster_bootstrap(
            lambda i: 100.0 * (c["c5"][i].mean() - b["c5"][i].mean()), data.group[ctx.animal_real_idx],
            n_boot=n_boot, seed=seed + 4),
        "C6_change_points": paired_cluster_bootstrap(
            lambda i: 100.0 * (c["c6"][i].mean() - b["c6"][i].mean()), data.group[ctx.test_imitation_idx],
            n_boot=n_boot, seed=seed + 5),
    }
    if ctx.chicken_idx.size:
        yc = ctx.chicken_y
        out["dV_chicken"] = paired_cluster_bootstrap(
            lambda i: auc64(yc[i], c["v_chicken"][i]) - auc64(yc[i], b["v_chicken"][i]),
            data.chicken_cluster[ctx.chicken_idx], n_boot=n_boot, seed=seed + 6, labels=yc)
    return out


# --------------------------------------------------------------------------------------------- gates

def gate1(data: Data, manifest_ids: np.ndarray, folds: dict[str, Any], expected_rows: int | None = 3138) -> dict:
    reasons = []
    if expected_rows is not None and data.n != expected_rows:
        reasons.append(f"{data.n} rows, expected {expected_rows}")
    if data.ids.shape != manifest_ids.shape or not np.array_equal(data.ids, manifest_ids):
        reasons.append("array ids differ from the manifest's order")
    if not np.isfinite(data.x).all() or not np.isfinite(data.raw_norms).all():
        reasons.append("non-finite embedding values")
    elif np.abs(data.raw_norms - 1.0).max() > 0.01:
        reasons.append("an embedding norm is more than 0.01 from 1")
    expected = folds.get("counts_by_analysis_set", {})
    for fold in ("A", "B", "C"):
        for set_name, count in expected.get(fold, {}).items():
            got = int(((data.fold == fold) & (data.analysis_set == set_name)).sum())
            if got != int(count):
                reasons.append(f"fold {fold} set {set_name}: {got} rows, expected {count}")
    return {"passed": not reasons, "reasons": reasons}


def gate2(identity_max_diff: dict[str, float], reproduced: dict[str, dict[str, float]],
          registered: dict[str, dict[str, float]], tolerance: float = GATE2_TOLERANCE) -> dict:
    reasons = []
    for name, diff in identity_max_diff.items():
        if not (diff <= 1e-9):
            reasons.append(f"{name}: alpha = 0 changed the embedding by {diff:.3g}")
    for name, values in reproduced.items():
        for key, value in values.items():
            ref = registered.get(name, {}).get(key)
            if ref is None or not np.isfinite(value) or abs(value - ref) > tolerance:
                reasons.append(f"{name} {key}: reproduced {value:.4f} against Part 188's {ref}")
    return {"passed": not reasons, "reasons": reasons, "tolerance": tolerance}


def gate3_rule(changes_by_direction: dict[str, list[float]], median_max: float = GATE3_MEDIAN_MAX,
               p95_max: float = GATE3_P95_MAX) -> dict:
    """The negative-control gate on a list of held-out verdict changes per direction (and rank)."""
    reasons, stats = [], {}
    for name, changes in changes_by_direction.items():
        arr = np.asarray(changes, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).all():
            reasons.append(f"{name}: undefined control changes")
            continue
        med, p95 = float(np.median(arr)), float(np.percentile(arr, 95))
        stats[name] = {"median": med, "p95": p95}
        if med > median_max:
            reasons.append(f"{name}: median {med:+.4f} > {median_max:+.2f}")
        if p95 > p95_max:
            reasons.append(f"{name}: 95th percentile {p95:+.4f} > {p95_max:+.2f}")
    return {"passed": not reasons, "reasons": reasons, "stats": stats}


def gate3_self_test(main_changes: dict[str, float]) -> dict:
    """Feed the main edit's own changes to the gate rule. Where any exceeds the 95th-percentile bar the rule must
    fail; that shows on the real data that the gate can fail."""
    exceeds = {k: v for k, v in main_changes.items() if np.isfinite(v) and v > GATE3_P95_MAX}
    if not exceeds:
        return {"shown": False, "note": "not shown on real data (no main change above +0.05)"}
    verdict = gate3_rule({k: [v] for k, v in exceeds.items()})
    return {"shown": not verdict["passed"], "rule_failed_on_main_edit": not verdict["passed"],
            "checked": sorted(exceeds)}


def gate4(base_accuracy: dict[str, float], minimum: float = GATE4_MIN_ACCURACY) -> dict:
    bad = {k: v for k, v in base_accuracy.items() if not (v >= minimum)}
    return {"passed": not bad, "reasons": [f"{k}: 10-NN class accuracy {v:.3f} < {minimum}" for k, v in bad.items()]}


# --------------------------------------------------------------------------------------------- predictions

def _both(values: dict[str, float], test: Callable[[float], bool]) -> str:
    if len(values) != 2 or any(v is None or not np.isfinite(v) for v in values.values()):
        return "not read"
    return "held" if all(test(v) for v in values.values()) else "failed"


PREDICTION_RULES: dict[str, tuple[str, Callable[[float], bool]]] = {
    "190.1": ("rank1_dV_test", lambda v: v >= 0.10),
    "190.2": ("rank1_V_test", lambda v: v >= 0.85),
    "190.3": ("rank8_minus_rank1_V_test", lambda v: v < 0.02),
    "190.4": ("shuffled_rank1_median_dV_test", lambda v: v <= 0.02),
    "190.5": ("random_rank1_median_dV_test", lambda v: v <= 0.02),
    "190.6": ("rank1_C1_change_points", lambda v: abs(v) < 1.0),
    "190.7": ("rank1_C2_first_neighbour_changed", lambda v: v < 0.10),
    "190.8": ("rank1_dV_chicken", lambda v: v >= 0.05),
    "190.9": ("rank1_C5_change_points", lambda v: -v < 2.0),
    "190.10": ("rank1_C6_change_points", lambda v: v < 5.0),
    "190.11": ("rank1_dK_test", lambda v: v >= 0.03),
    "190.12": ("rank1_V_test_gap_to_ladder_max", lambda v: v < 0.03),
}
NEIGHBOUR_PREDICTIONS = ("190.6", "190.7")


def prediction_values(directions: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {key: {} for key, _ in PREDICTION_RULES.values()}
    for name, d in directions.items():
        r1 = d["ranks"]["1"]
        op = r1["operating"]
        out["rank1_dV_test"][name] = op["dV_test"]
        out["rank1_V_test"][name] = op["V_test"]
        out["rank8_minus_rank1_V_test"][name] = d["ranks"]["8"]["operating"]["V_test"] - op["V_test"]
        out["shuffled_rank1_median_dV_test"][name] = r1["controls_at_operating"]["shuffled"]["dV_test"]["median"]
        out["random_rank1_median_dV_test"][name] = r1["controls_at_operating"]["random"]["dV_test"]["median"]
        out["rank1_C1_change_points"][name] = op["C1_change_points"]
        out["rank1_C2_first_neighbour_changed"][name] = op["C2_first_neighbour_changed"]
        out["rank1_dV_chicken"][name] = op["dV_chicken"]
        out["rank1_C5_change_points"][name] = op["C5_change_points"]
        out["rank1_C6_change_points"][name] = op["C6_change_points"]
        out["rank1_dK_test"][name] = op["dK_test"]
        ladder_max = max(row["V_test"] for row in r1["ladder"].values())
        out["rank1_V_test_gap_to_ladder_max"][name] = ladder_max - op["V_test"]
    return out


def score_predictions(values: dict[str, dict[str, float]], readable: bool, neighbours_readable: bool) -> dict:
    out = {}
    for pid, (key, test) in PREDICTION_RULES.items():
        if not readable or (pid in NEIGHBOUR_PREDICTIONS and not neighbours_readable):
            verdict = "not read"
        else:
            verdict = _both(values[key], test)
        out[pid] = {"quantity": key, "values": values[key], "verdict": verdict}
    return out


def headline(readable: bool, verdicts: dict[str, Any], directions: dict[str, Any]) -> dict:
    reasons = []
    if not readable:
        reasons.append("a refusal gate failed")
    for pid in ("190.1", "190.4", "190.5"):
        if verdicts[pid]["verdict"] != "held":
            reasons.append(f"{pid} {verdicts[pid]['verdict']}")
    for name, d in directions.items():
        r1 = d["ranks"]["1"]
        main = r1["operating"]["dV_test"]
        ctrl = r1["controls_at_operating"]
        if not (main > ctrl["shuffled"]["dV_test"]["p95"] and main > ctrl["random"]["dV_test"]["p95"]):
            reasons.append(f"{name}: main rank-1 change does not exceed both controls' 95th percentiles")
    cost = [pid for pid in ("190.6", "190.9") if verdicts[pid]["verdict"] != "held"]
    if not reasons and cost:
        return {"permitted": False, "reading": "separation at a cost", "reasons": [f"{p} {verdicts[p]['verdict']}"
                                                                                     for p in cost]}
    return {"permitted": not reasons and not cost, "reasons": reasons}


# --------------------------------------------------------------------------------------------- the run

def _json(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _control_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in ("dV_test", "dV_chicken", "dK_test", "C1_change_points", "C2_first_neighbour_changed",
                "C5_change_points", "C6_change_points", "C3_movement"):
        arr = np.asarray([r[key] for r in rows], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[key] = ({"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
                     "p5": float(np.percentile(arr, 5)), "n": int(arr.size)} if arr.size else
                    {"median": float("nan"), "p95": float("nan"), "p5": float("nan"), "n": 0})
    return out


def run_direction(data: Data, ctx: Context, *, ranks=RANKS, doses=DOSES, n_controls=N_CONTROLS,
                  n_boot=BOOTSTRAPS, seed=190, fitter: Callable = fit_logistic_l2, log=print) -> dict[str, Any]:
    x_fit = data.x[ctx.fit_idx]
    groups_fit = data.group[ctx.fit_idx]
    kmax = max(ranks)
    v_bar = verdict_direction(data, ctx.fit_idx)
    base = evaluate(data, ctx, data.x, per_clip=True)
    main_full = build_edit(x_fit, ctx.fit_y, read_directions(x_fit, ctx.fit_y, kmax, fitter), v_bar)
    probe = fitter(x_fit, ctx.fit_y.astype(float))
    probe_auc = auc(ctx.test_y, logistic_scores(probe, data.x[ctx.test_idx]))
    identity = float(np.abs(main_full.apply(data.x, 0.0) - data.x).max())
    rng = np.random.default_rng(seed)
    shuffled_full = []
    for _ in range(n_controls):
        y_shuf = shuffle_within_groups(ctx.fit_y, groups_fit, rng)
        shuffled_full.append(build_edit(x_fit, y_shuf, read_directions(x_fit, y_shuf, kmax, fitter), v_bar))
    log(f"{ctx.name}: probes fitted (1 main + {n_controls} shuffled, rank {kmax})")
    result: dict[str, Any] = {"fit_fold_n": int(ctx.fit_idx.size), "test_fold_n": int(ctx.test_idx.size),
                              "baseline": {k: base[k] for k in ("V_test", "V_test_free", "V_fit", "V_chicken",
                                                                 "K_test", "C1_accuracy", "C5_phrase_accuracy",
                                                                 "C6_human_share")},
                              "probe_heldout_auc": probe_auc, "identity_max_abs_diff": identity, "ranks": {}}
    for k in ranks:
        main = main_full.truncated(k)
        ref_disp = main.mean_displacement(data.x, 1.0)
        shuffled = [e.truncated(k) for e in shuffled_full]
        shuffled = [e.with_scale(ref_disp / e.mean_displacement(data.x, 1.0)) for e in shuffled]
        randoms = [random_edit(x_fit, v_bar, k, rng) for _ in range(n_controls)]
        randoms = [e.with_scale(ref_disp / e.mean_displacement(data.x, 1.0)) for e in randoms]
        ladder, ctrl_ladder = {}, {}
        for alpha in doses:
            if alpha == 0:
                ladder[alpha] = summarize(base, base)
                zero = summarize(base, base)
                ctrl_ladder[alpha] = {"shuffled": [zero] * n_controls, "random": [zero] * n_controls}
                continue
            ladder[alpha] = summarize(evaluate(data, ctx, main.apply(data.x, alpha)), base)
            ladder[alpha]["mean_displacement"] = main.mean_displacement(data.x, alpha)
            ctrl_ladder[alpha] = {
                "shuffled": [summarize(evaluate(data, ctx, e.apply(data.x, alpha)), base) for e in shuffled],
                "random": [summarize(evaluate(data, ctx, e.apply(data.x, alpha)), base) for e in randoms]}
        a_star = operating_dose(ladder)
        cur = evaluate(data, ctx, main.apply(data.x, a_star), per_clip=True)
        op = summarize(cur, base)
        result["ranks"][str(k)] = {
            "operating_dose": a_star, "operating": op,
            "operating_intervals": main_intervals(data, ctx, cur, base, n_boot=n_boot, seed=seed + 10 * k),
            "ladder": {str(a): r for a, r in ladder.items()},
            "controls_at_operating": {kind: _control_summary(ctrl_ladder[a_star][kind])
                                      for kind in ("shuffled", "random")},
            "controls_ladder": {str(a): {kind: _control_summary(v[kind]) for kind in v}
                                for a, v in ctrl_ladder.items()},
            "shuffled_dV_test_at_operating": [r["dV_test"] for r in ctrl_ladder[a_star]["shuffled"]],
        }
        log(f"{ctx.name}: rank {k} done (operating dose {a_star:g})")
    return result


def analyze(data: Data, manifest_ids: np.ndarray, folds: dict[str, Any], registered188: dict[str, Any], *,
            ranks=RANKS, doses=DOSES, n_controls=N_CONTROLS, n_boot=BOOTSTRAPS, expected_rows: int | None = 3138,
            fitter: Callable = fit_logistic_l2, log=print) -> dict[str, Any]:
    if 1 not in ranks or 8 not in ranks:
        raise ValueError("the registered predictions read ranks 1 and 8")
    g1 = gate1(data, manifest_ids, folds, expected_rows)
    if not g1["passed"]:
        return {"readable": False, "gates": {"gate1": g1}, "note": "Gate 1 failed; nothing else was computed"}
    directions = {}
    for seed_offset, (name, fit_fold, test_fold) in enumerate(DIRECTIONS):
        ctx = make_context(data, name, fit_fold, test_fold)
        directions[name] = run_direction(data, ctx, ranks=ranks, doses=doses, n_controls=n_controls,
                                         n_boot=n_boot, seed=190 + 1000 * seed_offset, fitter=fitter, log=log)
    reg = registered188.get("directions", {})
    g2 = gate2({n: d["identity_max_abs_diff"] for n, d in directions.items()},
               {n: {"model_class_informed": d["baseline"]["V_test"], "output_embedding_probe": d["probe_heldout_auc"]}
                for n, d in directions.items()},
               {n: {"model_class_informed": (reg.get(n, {}).get("model_class_informed") or {}).get("auc"),
                    "output_embedding_probe": (reg.get(n, {}).get("output_embedding_probe") or {}).get("auc")}
                for n in directions})
    g3 = gate3_rule({f"{n} rank {k}": d["ranks"][str(k)]["shuffled_dV_test_at_operating"]
                     for n, d in directions.items() for k in ranks})
    self_test = gate3_self_test({f"{n} rank {k}": d["ranks"][str(k)]["operating"]["dV_test"]
                                 for n, d in directions.items() for k in ranks})
    g4 = gate4({n: d["baseline"]["C1_accuracy"] for n, d in directions.items()})
    readable = g1["passed"] and g2["passed"] and g3["passed"]
    values = prediction_values(directions)
    verdicts = score_predictions(values, readable, g4["passed"])
    return _json({"readable": readable, "neighbours_readable": g4["passed"],
                  "gates": {"gate1": g1, "gate2": g2, "gate3": g3, "gate3_self_test": self_test, "gate4": g4},
                  "predictions": verdicts, "headline": headline(readable, verdicts, directions),
                  "directions": directions,
                  "settings": {"ranks": list(ranks), "doses": list(doses), "n_controls": n_controls,
                               "n_boot": n_boot, "neighbours": NEIGHBOURS, "dose_tolerance": DOSE_TOLERANCE}})


def run_files(emb_path: Path, manifest_path: Path, folds_path: Path, results188_path: Path, out_dir: Path,
              **kwargs) -> dict[str, Any]:
    data, rows = load_data(emb_path, manifest_path)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    registered = json.loads(results188_path.read_text(encoding="utf-8"))
    result = analyze(data, np.asarray([r["id"] for r in rows], dtype=str), folds, registered, **kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result
