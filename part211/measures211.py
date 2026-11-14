#!/usr/bin/env python3
"""measures211.py: the measures of PART211-DESIGN.md (design: Measures), exactly as the design defines them.

P      Jensen-Shannon divergence, base 2 (between 0 and 1), between two heads' attention rows at each query, averaged
       over queries 1 to 255 and the documents. Query 0 is left out because its row is always one key.
P0     the same with key 0 removed and each row renormalised, over queries 2 to 255 (reported beside P for the sink).
cos    the cosine distance between the two heads' attention matrices flattened over the documents (Bali et al.),
       reported, not predicted on.
M      1 minus centred linear CKA (Kornblith et al. 2019: features centred over positions) between two checkpoints'
       MLP outputs at the recorded positions.
O      1 minus centred linear CKA between two checkpoints' head contributions (a composite; reported, not predicted on).
V      the Frobenius norm of the difference between two OV matrices divided by the final checkpoint's OV norm.
f_X    settling fraction X(s, final) / X(step 0, final).
t_X    settling time: the first checkpoint at which f_X <= 0.5 and stays <= 0.5 at every later checkpoint, linearly
       interpolated in log(1 + step) between the two checkpoints that straddle 0.5; a layer still above 0.5 at the
       last checkpoint before final is censored ("not settled"); f at final is never read.
cross  the 12-by-12 matrix of P between every head of run a and every head of run b at the same step; the
       minimum-cost one-to-one assignment (scipy.optimize.linear_sum_assignment; scipy is required) and the matched
       distance (mean P over assigned pairs); beside it Bali et al.'s greedy match (each head of b takes its nearest
       head of a, repeats allowed).
same-index  head i of run b at final is same-index compatible when P(i in b, i in R0) minus the minimum over j of
       P(i in b, j in R0) is at most the layer's noise floor (compatibility, not identification); the number of heads
       whose best match is unique (second best above the best by more than the floor) and the number that are
       ambiguous (second best within the floor of the best) are reported beside it.
s*     crossing step: the first checkpoint at which the layer's mean within-run P(s, final) is below its matched
       cross-run P at the final step between run b and R0 by more than the floor, and stays below at every later
       checkpoint. s*_M is the same with M.
"""
from __future__ import annotations

import numpy as np

QUERY_FIRST = 1                 # queries 1 to 255 for P (query 0 left out)
QUERY_FIRST_P0 = 2              # queries 2 to 255 for P0
SETTLE_FRACTION = 0.5
FLOOR = 1e-12                   # a JSD of 0 is exact 0; this only keeps log(0) out of the interpolation


# ---------------------------------------------------------------- pattern distances

def _jsd_rows(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence, base 2, between two probability vectors along `axis` (rows on the last axis)."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda a, b: np.where(a > 0, a * np.log2(a / b), 0.0)
        d = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return np.clip(d, 0.0, 1.0).sum(axis=axis)


P0_KEEP = 0.001     # design: a query whose row keeps less than this of its mass outside key 0 is dropped from P0


def _renormalise(row: np.ndarray) -> np.ndarray:
    s = row.sum(axis=-1, keepdims=True)
    return np.where(s > 0, row / np.where(s > 0, s, 1.0), row)


def p0_keep_mask(a: np.ndarray, b: np.ndarray, lo: int = QUERY_FIRST_P0) -> np.ndarray:
    """The P0 queries kept: queries lo.. of both compared matrices [T, T] whose row keeps at least P0_KEEP of its mass
    outside key 0 in each (design: Noise floor)."""
    ka = 1.0 - a[lo:, 0]
    kb = 1.0 - b[lo:, 0]
    return (ka >= P0_KEEP) & (kb >= P0_KEEP)


def pattern_distance(a: np.ndarray, b: np.ndarray, variant: str = "P") -> float:
    """P or P0 between two attention matrices of one head, [T, T] (rows = queries), averaged over the queries.

    A batch of documents is allowed: [D, T, T] gives the mean over documents and queries.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.shape != b.shape or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"attention matrices must be square and alike: {a.shape} vs {b.shape}")
    if variant == "P":
        lo = QUERY_FIRST
        return float(_jsd_rows(a[..., lo:, :], b[..., lo:, :]).mean())
    if variant == "P0":
        lo = QUERY_FIRST_P0
        keep = p0_keep_mask(a, b, lo)
        if not keep.any():
            raise ValueError("every P0 query is dropped (each row keeps under 0.001 of its mass outside key 0)")
        ra = _renormalise(a[..., lo:, 1:])[..., keep, :]
        rb = _renormalise(b[..., lo:, 1:])[..., keep, :]
        return float(_jsd_rows(ra, rb).mean())
    raise ValueError(f"unknown pattern distance variant {variant!r}; use 'P' or 'P0'")


def pattern_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P between every head of one checkpoint and every head of another, for one document and layer: a, b [H, T, T]
    -> [H_a, H_b], entry (i, j) equal to pattern_distance(a[i], b[j]) (the same arithmetic, broadcast)."""
    a = np.asarray(a, np.float32)[:, None, QUERY_FIRST:, :]
    b = np.asarray(b, np.float32)[None, :, QUERY_FIRST:, :]
    return _jsd_rows(a, b).mean(axis=-1)


def p0_parts_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """P0 as additive parts between every head of one checkpoint and every head of another, for one document and layer:
    a, b [H, T, T] -> (sums [H_a, H_b], kept [H_a, H_b], total). `sums` is the JSD summed over the kept queries (2 to
    T-1), `kept` how many were kept, `total` how many there are (T - 2). A query is kept when both rows keep at least
    P0_KEEP of their mass outside key 0 (design: Noise floor). Parts, not a mean, so a resample of documents gets P0 and
    the dropped count exactly, and a head pair whose queries are all dropped in one document has a part of zero
    instead of an error (pattern_distance(..., "P0") raises there)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    lo = QUERY_FIRST_P0
    keep = ((1.0 - a[:, lo:, 0])[:, None, :] >= P0_KEEP) & ((1.0 - b[:, lo:, 0])[None, :, :] >= P0_KEEP)
    with np.errstate(divide="ignore", invalid="ignore"):
        j = _jsd_rows(_renormalise(a[:, lo:, 1:])[:, None], _renormalise(b[:, lo:, 1:])[None])
    return np.where(keep, j, 0.0).sum(axis=-1), keep.sum(axis=-1).astype(np.int64), int(keep.shape[-1])


def p0_from_parts(sums: np.ndarray, kept: np.ndarray, total: int, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P0 for every draw from per-document parts (documents on the first axis), and the dropped count. A pair with
    more than half its queries dropped in the draw has no P0 (NaN; design: Noise floor). counts [B, n_docs] ->
    (p0 [B, ...], dropped [B, ...])."""
    c = np.atleast_2d(np.asarray(counts, np.float64))
    s = np.tensordot(c, np.asarray(sums, np.float64), axes=(1, 0))
    k = np.tensordot(c, np.asarray(kept, np.float64), axes=(1, 0))
    t = (c.sum(axis=1) * float(total)).reshape((-1,) + (1,) * (np.ndim(sums) - 1))
    dropped = t - k
    with np.errstate(divide="ignore", invalid="ignore"):
        p0 = np.where((k > 0) & (dropped <= 0.5 * t), s / np.where(k > 0, k, 1.0), np.nan)
    return p0, dropped


def cos_parts_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cosine measure as additive parts for one document and layer: a, b [H, T, T] -> (dot [H_a, H_b], sq_a [H_a],
    sq_b [H_b]), every attention matrix flattened. Summed over documents they give Bali et al.'s cosine of the
    matrices flattened over the documents (design: Measures), which a per-document cosine does not."""
    x = np.asarray(a, np.float64).reshape(a.shape[0], -1)
    y = np.asarray(b, np.float64).reshape(b.shape[0], -1)
    return x @ y.T, (x * x).sum(axis=1), (y * y).sum(axis=1)


def cos_from_parts(dot: np.ndarray, sq_a: np.ndarray, sq_b: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """1 minus the cosine similarity for every draw from per-document parts: dot [n_docs, ..., H_a, H_b], sq_a
    [n_docs, ..., H_a], sq_b [n_docs, ..., H_b] -> [B, ..., H_a, H_b]. A zero norm gives distance 1."""
    c = np.atleast_2d(np.asarray(counts, np.float64))
    d = np.tensordot(c, np.asarray(dot, np.float64), axes=(1, 0))
    xa = np.tensordot(c, np.asarray(sq_a, np.float64), axes=(1, 0))[..., :, None]
    yb = np.tensordot(c, np.asarray(sq_b, np.float64), axes=(1, 0))[..., None, :]
    den = np.sqrt(xa * yb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 1.0 - d / np.where(den > 0, den, 1.0), 1.0)


def p0_dropped(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """(dropped, total) P0 queries for one head pair (design: Noise floor: the dropped count is printed)."""
    keep = p0_keep_mask(a, b)
    return int((~keep).sum()), int(keep.size)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 minus the cosine similarity of two attention matrices flattened over the documents (Bali et al.)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


# ---------------------------------------------------------------- CKA

def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centred linear CKA (Kornblith et al. 2019), features centred over positions (rows)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA needs the same positions: {X.shape[0]} vs {Y.shape[0]}")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F is the Frobenius norm of the cross Gram matrix, and ||X^T X||_F, ||Y^T Y||_F likewise
    num = float(np.linalg.norm(X.T @ Y) ** 2)
    den = float(np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return num / den


def cka_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """M or O: 1 minus centred linear CKA."""
    return 1.0 - cka(X, Y)


# ---------------------------------------------------------------- V

def ov_distance(ov_a: np.ndarray, ov_b: np.ndarray, ov_final: np.ndarray) -> float:
    """V: ||OV_a - OV_b||_F / ||OV_final||_F for one head ([d_model, d_model] each)."""
    den = float(np.linalg.norm(np.asarray(ov_final, np.float64)))
    if den == 0:
        raise ValueError("the final checkpoint's OV norm is 0: V is undefined")
    return float(np.linalg.norm(np.asarray(ov_a, np.float64) - np.asarray(ov_b, np.float64)) / den)


def cka_count_stats(x_docs: np.ndarray, y_docs: np.ndarray) -> dict:
    """Everything pooled centred linear CKA needs for ANY document resample, from two checkpoints' outputs at one layer,
    x_docs [n_docs, n_pos, D1] and y_docs [n_docs, n_pos, D2] (same documents, same positions).

    A resample is a count c_d per document (repeats allowed); the pool has N = n_pos * sum(c) rows. With A_d = X_d^T Y_d,
    U = sum_d c_d A_d, u = sum_d c_d sx_d, v = sum_d c_d sy_d (sx_d the column sums of X_d):
        Xc^T Yc = U - u v^T / N,   ||Xc^T Yc||^2 = c^T G c - (2/N) T[c, c, c] + (c^T Sxx c)(c^T Syy c) / N^2
    with G[d, e] = <A_d, A_e>, T[d, e, f] = sx_e^T A_d sy_f and Sxx[d, e] = sx_d . sx_e. Every one of these is a block
    reduction of the position Gram matrices K = X X^T and L = Y Y^T ([n_docs n_pos] square), so nothing D-by-D is
    formed. Each checkpoint is first centred on its own full-pool mean, which leaves every resample's CKA unchanged (the
    resample is re-centred) and keeps the three terms from cancelling when the outputs share a large mean."""
    x = np.asarray(x_docs, np.float64)
    y = np.asarray(y_docs, np.float64)
    if x.shape[:2] != y.shape[:2]:
        raise ValueError(f"CKA needs the same documents and positions: {x.shape[:2]} vs {y.shape[:2]}")
    nd, npos = x.shape[:2]
    xf = x.reshape(nd * npos, -1)
    yf = y.reshape(nd * npos, -1)
    xf = xf - xf.mean(axis=0, keepdims=True)
    yf = yf - yf.mean(axis=0, keepdims=True)
    K = (xf @ xf.T).reshape(nd, npos, nd, npos)
    L = (yf @ yf.T).reshape(nd, npos, nd, npos)
    rk = K.sum(axis=3)                                   # [d, p, e]: (K_{d,e} 1)_p
    rl = L.sum(axis=3)
    return {"n_pos": npos, "n_docs": nd,
            "G_xy": (K * L).sum(axis=(1, 3)), "G_xx": (K * K).sum(axis=(1, 3)), "G_yy": (L * L).sum(axis=(1, 3)),
            "T_xy": np.einsum("dpe,dpf->def", rk, rl), "T_xx": np.einsum("dpe,dpf->def", rk, rk),
            "T_yy": np.einsum("dpe,dpf->def", rl, rl),
            "S_xx": rk.sum(axis=1), "S_yy": rl.sum(axis=1)}


def cka_from_counts(stats: dict, counts: np.ndarray) -> np.ndarray:
    """Pooled centred linear CKA for each resample: counts [B, n_docs] (or [n_docs]) of how often each document is
    drawn. Exact (cka_count_stats). Returns [B] (or a float)."""
    c = np.asarray(counts, np.float64)
    single = c.ndim == 1
    c = np.atleast_2d(c)
    n = stats["n_pos"] * c.sum(axis=1)

    def sq(g, t, sa, sb):
        quad = np.einsum("bd,de,be->b", c, g, c)
        cub = np.einsum("def,bd,be,bf->b", t, c, c, c)
        return quad - 2.0 * cub / n + np.einsum("bd,de,be->b", c, sa, c) * np.einsum("bd,de,be->b", c, sb, c) / n ** 2
    num = sq(stats["G_xy"], stats["T_xy"], stats["S_xx"], stats["S_yy"])
    dx = sq(stats["G_xx"], stats["T_xx"], stats["S_xx"], stats["S_xx"])
    dy = sq(stats["G_yy"], stats["T_yy"], stats["S_yy"], stats["S_yy"])
    den = np.sqrt(np.clip(dx, 0, None) * np.clip(dy, 0, None))       # sqrt(x * x) == x exactly
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num == 0, 1.0, 0.0))
    return float(out[0]) if single else out


def counts_of(idx: np.ndarray, n_docs: int) -> np.ndarray:
    """A resample's document counts: idx [B, k] of drawn document IDs -> [B, n_docs]."""
    idx = np.atleast_2d(np.asarray(idx))
    out = np.zeros((idx.shape[0], n_docs))
    for b in range(idx.shape[0]):
        np.add.at(out[b], idx[b], 1.0)
    return out


def cka_distance_factored(z_a: np.ndarray, w_a: np.ndarray, z_b: np.ndarray, w_b: np.ndarray) -> float:
    """O for one head without its [n, d_model] contributions: the contribution is X = Z W, with Z the head's attention
    output before the output block ([n, head_dim], positions pooled over documents) and W its output block ([head_dim,
    d_model]). Centring commutes with the right factor (Xc = Zc W), so with C_ab = Zc_a^T Zc_b and G = W W^T, every
    Frobenius norm linear CKA needs is a trace of head_dim-by-head_dim products:
    ||Xc_a^T Xc_b||^2 = tr(C_ab^T G_a C_ab G_b), ||Xc_a^T Xc_a||^2 = tr(C_aa G_a C_aa G_a). Exact (design: Measures,
    O; Storage: contributions are not kept)."""
    za = np.asarray(z_a, np.float64)
    zb = np.asarray(z_b, np.float64)
    if za.shape[0] != zb.shape[0]:
        raise ValueError(f"CKA needs the same positions: {za.shape[0]} vs {zb.shape[0]}")
    za = za - za.mean(axis=0, keepdims=True)
    zb = zb - zb.mean(axis=0, keepdims=True)
    ga = np.asarray(w_a, np.float64) @ np.asarray(w_a, np.float64).T
    gb = np.asarray(w_b, np.float64) @ np.asarray(w_b, np.float64).T
    cab, caa, cbb = za.T @ zb, za.T @ za, zb.T @ zb
    num = float(np.trace(cab.T @ ga @ cab @ gb))
    # sqrt(dx * dy), not sqrt(dx) * sqrt(dy): sqrt(x * x) == x exactly, so a head against itself gives exactly 0
    # (G2's repeated-pass control demands exact zeros)
    den = float(np.sqrt(max(np.trace(caa @ ga @ caa @ ga), 0.0) * max(np.trace(cbb @ gb @ cbb @ gb), 0.0)))
    if den == 0:
        return 0.0 if num == 0 else 1.0
    return 1.0 - num / den


def ov_distance_factored(v_a, w_a, v_b, w_b, v_f, w_f) -> float:
    """V for one head from the OV factors (value block [head_dim, d_model] and output block [head_dim, d_model]; the
    OV matrix is v^T w): ||OV_a - OV_b||_F / ||OV_final||_F, with every inner product <v_x^T w_x, v_y^T w_y>_F written
    as <v_x v_y^T, w_x w_y^T>_F over head_dim-by-head_dim matrices. Exact; the [d_model, d_model] matrices are never
    formed or stored (design: Measures, V)."""
    def ip(vx, wx, vy, wy):
        vx, wx, vy, wy = (np.asarray(t, np.float64) for t in (vx, wx, vy, wy))
        return float(np.sum((vx @ vy.T) * (wx @ wy.T)))
    den = np.sqrt(max(ip(v_f, w_f, v_f, w_f), 0.0))
    if den == 0:
        raise ValueError("the final checkpoint's OV norm is 0: V is undefined")
    sq = ip(v_a, w_a, v_a, w_a) - 2.0 * ip(v_a, w_a, v_b, w_b) + ip(v_b, w_b, v_b, w_b)
    return float(np.sqrt(max(sq, 0.0)) / den)


# ---------------------------------------------------------------- settling

def _first_of_final_run(below: np.ndarray) -> int | None:
    """The first index from which every later entry is True (None when the last entry is False)."""
    if below.size == 0 or not below[-1]:
        return None
    j = below.size - 1
    while j > 0 and below[j - 1]:
        j -= 1
    return j

UNMEASURED = None          # a denominator at or below four times the floor: no fraction, no settling time
CENSORED = "censored"      # f above 0.5 at the last checkpoint before final: "not settled"


def settling_fraction(x_s: float, x_step0: float, floor: float = 0.0) -> float:
    """f_X(s) = X(s, final) / X(step 0, final). Not capped: for M it can exceed 1 (design: Measures).

    A denominator at or below four times the noise floor is unmeasured: the layer gets no fraction (UNMEASURED).
    For P and V the caller passes the ratio of head means (mean over heads of X(s, final) / mean over heads of
    X(step 0, final)), not the mean of per-head fractions (design: Measures)."""
    if x_step0 <= 4.0 * floor:
        return UNMEASURED
    return x_s / x_step0


def settling_time(steps, f, threshold: float = SETTLE_FRACTION):
    """t_X from a settling fraction per checkpoint, in the order of `steps` (final last, never read).

    The first checkpoint at which f <= threshold and f <= threshold at every later checkpoint before final; between
    the last checkpoint above the threshold and the first at or below it, interpolated linearly in log(1 + step), one
    rule for every interval including step 0 to step 128 (design: Measures). A curve still above the threshold at the
    last checkpoint before final is censored (CENSORED); f at final is 0 by definition and is never used. A fraction
    that is unmeasured makes the settling time unmeasured.
    """
    steps = np.asarray(steps, np.float64)
    f = np.asarray(f, dtype=object)             # None (unmeasured) must survive the conversion
    # one entry per step, final included: an earlier version trimmed `steps` when `f` was one shorter and then dropped
    # f's last entry too, so step 64000 was never read for such a caller (fixed 2026-11-13: refuse instead)
    if steps.shape != f.shape or steps.size < 2:
        raise ValueError("steps and f must be alike, final included (f at final is never read)")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("steps must increase")
    f = f[:-1]                                  # final is never read
    steps = steps[:-1]
    if any(x is None for x in f):
        return UNMEASURED
    f = f.astype(np.float64)
    below = f <= threshold
    j = _first_of_final_run(below)
    if j is None:
        return CENSORED
    if j == 0:
        return float(steps[0])
    i = j - 1
    fa, fb = float(f[i]), float(f[j])
    if fa == fb:
        return float(steps[j])
    # f linear in log(1 + step) between the two checkpoints (design: Measures). An earlier version interpolated
    # log(f), a rule the design does not state (fixed 2026-11-13).
    frac = float(np.clip((fa - threshold) / (fa - fb), 0.0, 1.0))
    la, lb = np.log1p(steps[i]), np.log1p(steps[j])
    return float(np.expm1(la + frac * (lb - la)))


def _censored(t) -> bool:
    """Censored: the marker string, or an infinite time (what a scorer stores for a censored resample)."""
    if isinstance(t, str):
        return t == CENSORED
    try:
        return bool(np.isinf(t))
    except TypeError:
        return False


def earlier(t_x, t_y) -> bool:
    """The t_X < t_Y rule (design: Measures): holds when t_X is a time and t_Y is censored, or both are times and
    t_X < t_Y; does not hold when t_X is censored or unmeasured, including when both are censored."""
    if t_x is None or _censored(t_x):
        return False
    if _censored(t_y):
        return True
    if t_y is None:
        return False
    return float(t_x) < float(t_y)


# ---------------------------------------------------------------- cross-run matching

def _require_scipy():
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:      # design: scipy is required; the run aborts without it
        raise SystemExit("scipy is required for the cross-run match (design: Measures); it is not importable") from exc
    return linear_sum_assignment


def hungarian_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The minimum-cost one-to-one assignment, rows to columns: (row_ind, col_ind) with row_ind sorted."""
    linear_sum_assignment = _require_scipy()
    return linear_sum_assignment(np.asarray(cost, np.float64))


def matched_distance(cost: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """The mean cost over assigned pairs, and the pairs."""
    r, c = hungarian_match(cost)
    pairs = [(int(i), int(j)) for i, j in zip(r, c)]
    return float(np.mean([cost[i, j] for i, j in pairs])), pairs


def greedy_distance(cost: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Bali et al.'s greedy match: each head of b (columns) takes its nearest head of a (rows), repeats allowed."""
    cost = np.asarray(cost, np.float64)
    pairs = [(int(np.argmin(cost[:, j])), j) for j in range(cost.shape[1])]
    return float(np.mean([cost[i, j] for i, j in pairs])), pairs


def cross_run_matrix(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """The H_a by H_b matrix of P between every head of a and every head of b, each [D, T, T] over the documents."""
    a = np.asarray(p_a, np.float32)
    b = np.asarray(p_b, np.float32)
    return np.array([[pattern_distance(a[i], b[j]) for j in range(b.shape[0])] for i in range(a.shape[0])])


# ---------------------------------------------------------------- identity and crossing steps

def same_index_compatible(p_row: np.ndarray, floor: float, i: int) -> dict:
    """For one head i of run b against R0's heads at final: p_row[j] = P(i in run b, j in R0), and the gap
    g_i = P(i, i) - min_j P(i, j); the head is same-index compatible when g_i is at most the floor. Compatibility, not
    identification (design: Measures). Beside it: the best match, whether it is unique (second best above the best by
    more than the floor), and whether it is ambiguous (second best within the floor of the best). The head index is
    required: an earlier version read p_row[0] as P(i, i) for every head (fixed 2026-11-13)."""
    p_row = np.asarray(p_row, np.float64)
    if p_row.ndim != 1 or not 0 <= int(i) < p_row.size:
        raise ValueError("same_index_compatible needs one row and a head index inside it")
    best = int(np.argmin(p_row))
    ordered = np.sort(p_row)
    second_gap = float(ordered[1] - ordered[0]) if p_row.size > 1 else np.inf
    gap = float(p_row[int(i)] - p_row.min())
    return {"best": best, "gap": gap, "same_index_compatible": bool(gap <= floor),
            "unique": bool(second_gap > floor), "ambiguous": bool(second_gap <= floor)}


def same_index_gap_interval(p_rows_by_doc: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The [2.5th, 97.5th] percentile of head i's gap g_i over document resamples (design: 211.3, "the interval of
    g_i is printed beside it"). p_rows_by_doc is [n_docs, H, H] of per-document P (head i of run b against every head
    of R0); idx [B, n_docs] is the shared draw. Returns [H, 2]."""
    p = np.asarray(p_rows_by_doc, np.float64)
    # the documents' mean first, then g_i = P(i, i) - min_j P(i, j) (an earlier version returned min_j alone)
    if p.ndim != 3 or p.shape[1] != p.shape[2]:
        raise ValueError("same_index_gap_interval needs [n_docs, H, H]")
    m = p[np.asarray(idx)].mean(axis=1)                            # [B, H, H]
    gaps = np.diagonal(m, axis1=1, axis2=2) - m.min(axis=-1)       # [B, H]: P(i, i) minus min_j P(i, j)
    return np.stack([np.quantile(gaps, 0.025, axis=0), np.quantile(gaps, 0.975, axis=0)], axis=1)


def same_index_counts(p_rows: np.ndarray, floor: float) -> dict:
    """The per-layer counts a prediction reads: compatible heads, unique best matches, ambiguous heads. p_rows is one
    layer's H-by-H matrix, row i the P of head i in run b against every head of R0."""
    m = np.asarray(p_rows, np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("same_index_counts needs one layer's H-by-H matrix of P (row i: head i of run b)")
    rows = [same_index_compatible(m[i], floor, i) for i in range(m.shape[0])]
    return {"compatible": sum(1 for r in rows if r["same_index_compatible"]),
            "unique": sum(1 for r in rows if r["unique"]),
            "ambiguous": sum(1 for r in rows if r["ambiguous"]), "per_head": rows}


def matched_cost_with_duplicates(cost: np.ndarray) -> float:
    """G2's duplicate-head control: the matched cost of a synthetic layer with identical rows. Any assignment of
    matched cost 0 passes, whichever permutation the solver picks."""
    return matched_distance(np.asarray(cost, np.float64))[0]


def crossing_step(steps, within, cross_final: float, floor: float) -> float | str:
    """s*: the first checkpoint at which `within` (the layer's mean within-run distance to its own final) is below
    `cross_final` by more than the floor, and stays below by more than the floor at every later checkpoint before
    final. A layer that has not crossed by the last checkpoint before final (64000) is censored ("never crossed")."""
    steps = np.asarray(steps, np.float64)
    within = np.asarray(within, np.float64)
    if steps.shape != within.shape or steps.size < 2:
        raise ValueError("steps and within must be alike, final included (the final entry is never read)")
    # The last checkpoint before final (64000) IS read: "has not crossed by step 64000" is a censored layer, and a
    # layer still below at 64000 has crossed. (An earlier version dropped it, so a layer that crossed only at 64000
    # read as censored; fixed 2026-11-13.)
    within, steps = within[:-1], steps[:-1]     # final is never read
    j = _first_of_final_run(within < (cross_final - floor))
    return CENSORED if j is None else float(steps[j])


# ---------------------------------------------------------------- the thread's comparison (211.8)

def thread_comparison(within_p: float, cross_p: float, floor: float) -> dict:
    """R0's within-run P(step 1000, step 10000) against the matched cross-run P at step 10000 between R0 and a
    weight-seed run: holds when the within-run value is below the cross-run value by more than the floor."""
    return {"within": float(within_p), "cross": float(cross_p), "margin": float(cross_p - within_p),
            "holds": bool(cross_p - within_p > floor)}
