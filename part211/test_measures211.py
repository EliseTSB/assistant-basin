#!/usr/bin/env python3
"""Tests for measures211.py and score211.py: hand-built inputs with known answers (brief item 7).

  /home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q part211/test_measures211.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkpoints211 as C  # noqa: E402
import measures211 as MS  # noqa: E402
import score211 as S  # noqa: E402

T = 8


def one_hot_rows(targets, T=T):
    """A [T, T] causal attention matrix whose row q puts all its mass on key targets[q] (targets[q] <= q)."""
    a = np.zeros((T, T), np.float32)
    for q, k in enumerate(targets):
        a[q, k] = 1.0
    return a


def uniform_rows(T=T):
    a = np.zeros((T, T), np.float32)
    for q in range(T):
        a[q, :q + 1] = 1.0 / (q + 1)
    return a


# ---------------------------------------------------------------- P and P0

def test_identical_rows_give_P_zero():
    a = uniform_rows()
    assert MS.pattern_distance(a, a.copy()) == pytest.approx(0.0, abs=1e-12)
    assert MS.pattern_distance(a, a.copy(), variant="P0") == pytest.approx(0.0, abs=1e-12)
    # rows that are neither one-hot nor uniform, so a divergence computed against one side alone shows
    rng = np.random.default_rng(20)
    r = np.tril(rng.random((T, T)))
    r = r / r.sum(1, keepdims=True)
    assert MS.pattern_distance(r, r.copy()) == pytest.approx(0.0, abs=1e-12)


def test_disjoint_one_hot_rows_give_P_one():
    # every query 1..T-1 attends to a different key in each matrix, so every row's JSD is 1
    a = one_hot_rows([0] * T)
    b = one_hot_rows([0] + list(range(1, T)))
    # query 0 is left out; queries 1..T-1: a puts all mass on key 0, b on key q
    assert MS.pattern_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_P_leaves_out_query_zero():
    # two matrices that differ only at query 0 give P 0
    a = one_hot_rows([0] * T)
    b = one_hot_rows([0] * T)
    b[0, 0] = 0.0
    b[0, 1] = 0.0
    b[0, 3] = 1.0
    assert MS.pattern_distance(a, b) == pytest.approx(0.0, abs=1e-12)


def test_P0_removes_key_zero_and_renormalises():
    # rows 2..T-1: a attends to key 0 and key 1 equally; b attends to key 1 only. With key 0 removed and renormalised
    # both rows are the whole mass on key 1, so P0 is 0 while P is not.
    a = np.zeros((T, T), np.float32)
    b = np.zeros((T, T), np.float32)
    for q in range(T):
        a[q, :q + 1] = 1.0 / (q + 1)
        b[q, :q + 1] = 1.0 / (q + 1)
    for q in range(2, T):
        a[q] = 0.0
        a[q, 0] = 0.5
        a[q, 1] = 0.5
        b[q] = 0.0
        b[q, 1] = 1.0
    assert MS.pattern_distance(a, b, variant="P") > 0.1
    assert MS.pattern_distance(a, b, variant="P0") == pytest.approx(0.0, abs=1e-12)


def test_P_has_a_known_value_at_a_known_row():
    # one row, query 3, attends to key 3 only in b and half to key 3, half to key 5 in a. m is (3/4, 1/4), so both
    # KL terms are (3/4)log2(3/2) + (1/4)log2(2/3) = 0.311278 bits and the JSD is 0.311278. One of 7 queries.
    a = np.zeros((T, T))
    b = np.zeros((T, T))
    for q in range(T):
        a[q, q] = 1.0
        b[q, q] = 1.0
    a[3] = 0.0
    a[3, 3] = 0.5
    a[3, 5] = 0.5
    hand = 0.5 * (0.5 * np.log2(0.5 / 0.75) + 0.5 * np.log2(0.5 / 0.25)) + 0.5 * 1.0 * np.log2(1.0 / 0.75)
    assert MS.pattern_distance(a, b) == pytest.approx(hand / 7, abs=1e-9)


def test_P0_leaves_out_query_one():
    # rows 2.. attend to key 1 in both; query 1 differs, and is left out
    a = uniform_rows()
    b = uniform_rows()
    b[1] = 0.0
    b[1, 2] = 1.0
    assert MS.pattern_distance(a, b, variant="P0") == pytest.approx(0.0, abs=1e-12)
    assert MS.pattern_distance(a, b, variant="P") > 0.0


def test_P0_drops_a_row_that_keeps_under_a_thousandth_outside_key_zero():
    a = uniform_rows()
    b = uniform_rows()
    a[4] = 0.0
    a[4, 0] = 0.9995                      # keeps 0.0005 outside key 0: dropped from P0
    a[4, 2] = 0.0005
    assert MS.p0_dropped(a, b) == (1, T - 2)
    assert MS.pattern_distance(a, b, variant="P0") == pytest.approx(0.0, abs=1e-12)
    assert MS.pattern_distance(a, b, variant="P") > 0.0


def test_P_is_symmetric_and_in_range():
    rng = np.random.default_rng(0)
    a = rng.random((T, T)).astype(np.float32)
    b = rng.random((T, T)).astype(np.float32)
    a = np.tril(a) / np.tril(a).sum(1, keepdims=True)
    b = np.tril(b) / np.tril(b).sum(1, keepdims=True)
    p = MS.pattern_distance(a, b)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(MS.pattern_distance(b, a), abs=1e-6)
    # a pair whose divergence is not 0 either way, so a one-sided divergence cannot pass as symmetric
    c = one_hot_rows([0] + list(range(1, T)))
    d = uniform_rows()
    assert MS.pattern_distance(c, d) > 0.1
    assert MS.pattern_distance(c, d) == pytest.approx(MS.pattern_distance(d, c), abs=1e-6)


def test_cosine_distance_of_identical_is_zero():
    a = uniform_rows()
    assert MS.cosine_distance(a, a.copy()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- CKA

def test_cka_of_a_matrix_with_itself_is_one():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))
    assert MS.cka(X, X) == pytest.approx(1.0, abs=1e-10)
    assert MS.cka_distance(X, X) == pytest.approx(0.0, abs=1e-10)


def test_cka_is_centred():
    # a constant offset on every feature is removed by centring: X and X + c have CKA 1
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 5))
    assert MS.cka(X, X + 3.0) == pytest.approx(1.0, abs=1e-10)


def test_cka_is_invariant_to_rotation():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5))
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert MS.cka(X, X @ Q) == pytest.approx(1.0, abs=1e-8)
    # a non-orthogonal map does change it, so the test is not vacuous for any linear map
    A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    assert MS.cka(X, X @ A) < 0.99


def test_cka_of_unrelated_is_low():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 3))
    Y = rng.normal(size=(400, 3))
    assert MS.cka(X, Y) < 0.1


# ---------------------------------------------------------------- V

def test_ov_distance_divides_by_the_final_norm():
    ov_a = np.eye(3) * 2.0
    ov_b = np.eye(3) * 1.0
    ov_final = np.eye(3) * 4.0
    # ||a - b||_F = sqrt(3), ||final||_F = 4 sqrt(3)
    assert MS.ov_distance(ov_a, ov_b, ov_final) == pytest.approx(0.25)


# ---------------------------------------------------------------- the Hungarian match (G2)

def test_hungarian_recovers_a_known_permutation_at_distance_zero():
    rng = np.random.default_rng(5)
    H = 6
    cost = rng.random((H, H))
    perm = np.array([3, 0, 5, 1, 4, 2])
    permuted = np.full((H, H), 1.0)
    for j in range(H):
        permuted[perm[j], j] = 0.0           # head j of b is head perm[j] of a, and only there
    d, pairs = MS.matched_distance(permuted)
    assert d == pytest.approx(0.0, abs=1e-12)
    got = {j: i for i, j in pairs}
    for j in range(H):
        assert got[j] == perm[j]
    # the greedy match, repeats allowed, also recovers it here (each column's nearest row is its own)
    g, gpairs = MS.greedy_distance(permuted)
    assert g == pytest.approx(0.0, abs=1e-12)


def test_hungarian_beats_greedy_when_greedy_repeats():
    # two columns both nearest to row 0; greedy repeats row 0, the assignment must use both rows
    cost = np.array([[0.1, 0.2], [0.3, 0.4]])
    g, gpairs = MS.greedy_distance(cost)
    h, hpairs = MS.matched_distance(cost)
    assert gpairs == [(0, 0), (0, 1)] and g == pytest.approx(0.15)
    assert sorted(hpairs) == [(0, 0), (1, 1)] and h == pytest.approx(0.25)


def test_cross_run_matrix_shape_and_zero_diagonal():
    rng = np.random.default_rng(6)
    a = rng.random((3, T, T)).astype(np.float32)
    a = np.tril(a) / np.tril(a).sum(-1, keepdims=True)
    m = MS.cross_run_matrix(a, a.copy())
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0.0, atol=1e-12)


# ---------------------------------------------------------------- settling

def test_settling_time_on_a_hand_built_curve():
    steps = [0, 100, 1000, 10000, 20000]          # the last entry is final and is never read
    f = [1.0, 0.8, 0.2, 0.1, 0.0]
    # crosses between 100 and 1000, f linear in log(1 + step): f goes .8 -> .2, so 0.5 is half way (frac 0.5), i.e. the
    # geometric mean of 101 and 1001, minus 1: sqrt(101 * 1001) - 1 = 316.964...
    assert MS.settling_time(steps, f) == pytest.approx(np.sqrt(101 * 1001) - 1)
    # the step 0 to step 128 interval uses the same rule: .6 -> .4, half way, sqrt(129 * 513) - 1 = 256.25...
    t = MS.settling_time([0, 128, 512, 1024], [1.0, 0.6, 0.4, 0.0])
    assert t == pytest.approx(np.sqrt(129 * 513) - 1)


def test_settling_time_interpolates_in_log_one_plus_step_not_linear_step():
    steps = [0, 100, 1000, 10000, 20000]
    f = [1.0, 0.8, 0.2, 0.1, 0.0]
    t = MS.settling_time(steps, f)
    frac = (0.8 - 0.5) / (0.8 - 0.2)                         # f linear on the log(1 + step) axis
    log_t = np.expm1(np.log1p(100) + frac * (np.log1p(1000) - np.log1p(100)))
    log_plain = np.exp(np.log(100) + frac * np.log(10))
    lin_t = 100 + ((0.8 - 0.5) / (0.8 - 0.2)) * 900
    assert t == pytest.approx(log_t)
    assert abs(t - lin_t) > 100 and abs(t - log_plain) > 0.4


def test_settling_time_requires_persistence():
    # a dip below 0.5 at step 1000 that comes back does not count: the crossing is the later one
    steps = [0, 1000, 2000, 4000, 10000]
    f = [1.0, 0.3, 0.9, 0.4, 0.0]
    t = MS.settling_time(steps, f)
    assert t > 2000
    assert t < 4000


def test_settling_time_never_settled_is_censored():
    assert MS.settling_time([0, 100, 1000, 5000], [1.0, 0.9, 0.7, 0.0]) == MS.CENSORED
    # a curve that settles only at the last checkpoint before final is a time, not censored (interpolated between
    # 100 and 1000, in log(1 + step))
    t = MS.settling_time([0, 100, 1000, 5000], [1.0, 0.9, 0.4, 0.0])
    frac = (0.9 - 0.5) / (0.9 - 0.4)                         # 0.8 of the way from 100 to 1000 on log(1 + step)
    assert t == pytest.approx(np.expm1(np.log1p(100) + frac * (np.log1p(1000) - np.log1p(100))))
    # a curve without its final entry is refused, not read one step short
    with pytest.raises(ValueError):
        MS.settling_time([0, 100, 1000, 5000], [1.0, 0.9, 0.4])
    assert 100 < t < 1000
    # an unmeasured fraction gives no settling time
    assert MS.settling_time([0, 100, 1000, 5000], [1.0, 0.9, None, 0.0]) is None


def test_settling_time_thresholds_are_printed_descriptively():
    steps = [0, 100, 1000, 10000, 20000]
    f = [1.0, 0.8, 0.2, 0.1, 0.0]
    assert MS.settling_time(steps, f, threshold=0.75) < MS.settling_time(steps, f, threshold=0.5)
    assert MS.settling_time(steps, f, threshold=0.25) > MS.settling_time(steps, f, threshold=0.5)


def test_earlier_censoring_rule():
    assert MS.earlier(1000, MS.CENSORED)
    assert MS.earlier(1000, 2000) and not MS.earlier(2000, 1000)
    assert not MS.earlier(MS.CENSORED, MS.CENSORED)      # no evidence of order is not evidence of equality
    assert not MS.earlier(MS.CENSORED, 5000)
    assert not MS.earlier(None, 5000) and not MS.earlier(1000, None)


def test_settling_fraction_is_a_ratio_to_the_step_zero_distance():
    assert MS.settling_fraction(0.25, 0.5) == pytest.approx(0.5)
    assert MS.settling_fraction(0.75, 0.5) == pytest.approx(1.5)      # not capped
    # a denominator at or below four times the floor is unmeasured
    assert MS.settling_fraction(0.25, 0.4, floor=0.1) is None
    assert MS.settling_fraction(0.25, 0.41, floor=0.1) == pytest.approx(0.25 / 0.41)


# ---------------------------------------------------------------- crossing step

def test_crossing_step_on_a_hand_built_curve():
    # the last entry is final and is never read (a caller may pass it or not)
    steps = [0, 1000, 2000, 4000, 10000]
    within = [0.5, 0.4, 0.1, 0.05, 0.0]
    assert MS.crossing_step(steps[:-1], within[:-1], 0.3, floor=0.01) == 2000
    cross = 0.3
    assert MS.crossing_step(steps, within, cross, floor=0.01) == 2000
    # a curve that crosses at step 1000 and comes back to exactly the cross value at 2000: with a floor the return
    # breaks the run (the crossing is 4000), with none it does not (the crossing is 1000)
    assert MS.crossing_step(steps, [0.5, 0.1, 0.3, 0.05, 0.0], cross, floor=0.01) == 4000
    assert MS.crossing_step(steps, [0.5, 0.1, 0.3, 0.05, 0.0], cross, floor=-0.01) == 1000


def test_crossing_step_requires_persistence():
    steps = [0, 1000, 2000, 4000, 10000]
    within = [0.5, 0.1, 0.35, 0.05, 0.0]
    assert MS.crossing_step(steps, within, 0.3, floor=0.01) == 4000
    # a curve that never crosses is censored, not "final"
    assert MS.crossing_step(steps, [0.5, 0.4, 0.35, 0.31, 0.0], 0.3, floor=0.01) == MS.CENSORED
    # and one that crosses from the first checkpoint gets it
    assert MS.crossing_step(steps, [0.2, 0.1, 0.1, 0.05, 0.0], 0.3, floor=0.01) == 0
    # crossing only at the last checkpoint before final is a time
    assert MS.crossing_step(steps, [0.5, 0.4, 0.35, 0.05, 0.0], 0.3, floor=0.01) == 4000


def test_crossing_step_floor_matters():
    steps = [0, 1000, 2000, 10000]
    within = [0.5, 0.29, 0.05, 0.0]
    assert MS.crossing_step(steps, within, 0.3, floor=0.001) == 1000
    assert MS.crossing_step(steps, within, 0.3, floor=0.05) == 2000


def test_same_index_compatible_reads_the_gap():
    row = np.array([0.30, 0.05, 0.2])
    r = MS.same_index_compatible(row, floor=0.01, i=0)
    assert r["best"] == 1 and r["same_index_compatible"] is False and r["unique"] is True
    r = MS.same_index_compatible(np.array([0.05, 0.06, 0.2]), floor=0.01, i=0)
    assert r["same_index_compatible"] is True and r["unique"] is False and r["ambiguous"] is True
    # a unique best match that is not the head itself: not compatible, best is 1
    r = MS.same_index_compatible(np.array([0.2, 0.05, 0.3]), floor=0.01, i=0)
    assert r["best"] == 1 and r["same_index_compatible"] is False and r["unique"] is True
    # the same row read as head 1's: its own distance is the minimum, so it is compatible (the diagonal, not column 0)
    r = MS.same_index_compatible(np.array([0.2, 0.05, 0.3]), floor=0.01, i=1)
    assert r["gap"] == 0.0 and r["same_index_compatible"] is True
    # the counts a prediction reads
    c = MS.same_index_counts(np.array([[0.05, 0.06, 0.2], [0.2, 0.05, 0.3], [0.05, 0.06, 0.07]]), floor=0.01)
    assert c["compatible"] == 2 and c["unique"] == 1 and c["ambiguous"] == 2
    # every head nearest itself: all 3 compatible (reading column 0 as the diagonal would give 1)
    c = MS.same_index_counts(np.array([[0.05, 0.3, 0.2], [0.3, 0.05, 0.2], [0.2, 0.3, 0.05]]), floor=0.01)
    assert c["compatible"] == 3 and c["unique"] == 3
    with pytest.raises(ValueError):
        MS.same_index_counts(np.array([[0.05, 0.3, 0.2]]), floor=0.01)     # one row is not a layer


def test_pattern_matrix_equals_pattern_distance_per_pair():
    rng = np.random.default_rng(4)
    def rows(h, t=9):
        x = rng.uniform(size=(h, t, t)) * np.tril(np.ones((t, t)))
        return x / x.sum(-1, keepdims=True)
    a, b = rows(3), rows(4)
    m = MS.pattern_matrix(a, b)
    assert m.shape == (3, 4)
    for i in range(3):
        for j in range(4):
            assert m[i, j] == MS.pattern_distance(a[i], b[j])


def test_counts_cka_equals_pooled_cka_on_every_resample():
    rng = np.random.default_rng(7)
    nd, npos, d1, d2 = 6, 5, 9, 7
    x = rng.normal(size=(nd, npos, d1)) + 1e4                    # a large shared mean: the pre-centring must hold
    y = 0.6 * x[..., :d2] + rng.normal(size=(nd, npos, d2)) - 3e3
    st = MS.cka_count_stats(x, y)
    idx = np.array([[0, 0, 3, 4, 4, 5], [1, 2, 3, 4, 5, 0], [2, 2, 2, 2, 2, 2], [5, 1, 1, 0, 3, 3]])
    got = MS.cka_from_counts(st, MS.counts_of(idx, nd))
    for b, row in enumerate(idx):
        want = MS.cka(np.concatenate([x[k] for k in row]), np.concatenate([y[k] for k in row]))
        assert abs(got[b] - want) < 1e-9
    # all documents once is the plain pooled value
    assert abs(MS.cka_from_counts(st, np.ones(nd)) - MS.cka(x.reshape(-1, d1), y.reshape(-1, d2))) < 1e-9


def test_factored_O_equals_cka_on_the_contributions():
    rng = np.random.default_rng(5)
    n, hd, d = 170, 8, 40
    za, zb = rng.normal(size=(n, hd)), rng.normal(size=(n, hd))
    wa, wb = rng.normal(size=(hd, d)), rng.normal(size=(hd, d))
    want = MS.cka_distance(za @ wa, zb @ wb)
    assert MS.cka_distance_factored(za, wa, zb, wb) == pytest.approx(want, abs=1e-12)
    # a head against itself, and against itself rotated in its value space: 0
    q, _ = np.linalg.qr(rng.normal(size=(hd, hd)))
    assert MS.cka_distance_factored(za, wa, za, wa) == 0.0                 # exactly: G2's repeated pass
    for seed in range(20):                                                  # and on other draws, not by luck
        r = np.random.default_rng(seed)
        z, w = r.normal(size=(30, 4)) * 10 ** r.uniform(-3, 3), r.normal(size=(4, 9))
        assert MS.cka_distance_factored(z, w, z, w) == 0.0
    assert MS.cka_distance_factored(za @ q, q.T @ wa, za, wa) == pytest.approx(0.0, abs=1e-12)


def test_factored_V_equals_the_frobenius_distance_of_the_OV_matrices():
    rng = np.random.default_rng(6)
    hd, d = 8, 40
    va, wa, vb, wb, vf, wf = (rng.normal(size=(hd, d)) for _ in range(6))
    want = MS.ov_distance(va.T @ wa, vb.T @ wb, vf.T @ wf)
    assert MS.ov_distance_factored(va, wa, vb, wb, vf, wf) == pytest.approx(want, rel=1e-10)
    # a doubled value block: V = 1 against the original as final
    assert MS.ov_distance_factored(2 * va, wa, va, wa, va, wa) == pytest.approx(1.0, abs=1e-12)


def test_same_index_gap_interval_is_the_diagonal_minus_the_row_minimum():
    h, n = 3, 48
    m = np.full((h, h), 0.5)
    np.fill_diagonal(m, 0.05)
    m[2, 2], m[2, 0] = 0.3, 0.1                                     # head 2: own 0.3, nearest 0.1, gap 0.2
    per_doc = np.broadcast_to(m, (n, h, h)).copy()
    ci = MS.same_index_gap_interval(per_doc, S.resample_indices(n, 200))
    assert ci.shape == (h, 2)
    assert np.allclose(ci[0], 0.0) and np.allclose(ci[1], 0.0) and np.allclose(ci[2], 0.2)


# ---------------------------------------------------------------- the bootstrap

def test_bootstrap_interval_contains_the_point_and_shrinks_with_more_documents():
    rng = np.random.default_rng(7)
    few = rng.random(8)
    many = rng.random(400)
    ci_few = S.bootstrap_ci(few)
    ci_many = S.bootstrap_ci(many)
    assert ci_few[0] <= few.mean() <= ci_few[1]
    assert ci_many[0] <= many.mean() <= ci_many[1]
    assert (ci_many[1] - ci_many[0]) < (ci_few[1] - ci_few[0])


def test_bootstrap_interval_width_shrinks_as_the_square_root_of_the_documents():
    # the same population, 4x the documents: the width halves (to within the resampling noise of 1,000 draws)
    rng = np.random.default_rng(10)
    pop = rng.random(4000)
    small = S.bootstrap_ci(pop[:100])
    large = S.bootstrap_ci(pop[:400])
    ratio = (large[1] - large[0]) / (small[1] - small[0])
    assert 0.35 < ratio < 0.65
    # 16x the documents: a quarter of the width (the square-root rule), not a half
    tiny = S.bootstrap_ci(pop[:25])
    wide = S.bootstrap_ci(pop[:400])
    assert 0.15 < (wide[1] - wide[0]) / (tiny[1] - tiny[0]) < 0.45
    # and 4x the documents: about half
    mid = S.bootstrap_ci(pop[:100])
    assert 0.3 < (mid[1] - mid[0]) / (tiny[1] - tiny[0]) < 0.7


def test_bootstrap_is_seeded():
    v = np.random.default_rng(8).random(30)
    assert S.bootstrap_ci(v) == S.bootstrap_ci(v)


# ---------------------------------------------------------------- G1

def _state(scale=1.0, extra=None):
    st = {"layer.weight": np.ones((2, 2)) * scale, "layer.bias": np.zeros(2) + scale, "rotary.inv_freq": np.arange(3.0)}
    if extra:
        st.update(extra)
    return st


def test_g1_passes_when_D_equals_R0_and_W_differs():
    r0 = _state(1.0)
    g = S.g1_seed_split(r0, {"D1": _state(1.0), "D2": _state(1.0), "W1": _state(2.0), "W2": _state(3.0)})
    assert g["pass"] and g["d_pass"] and g["w_pass"]
    assert g["runs"]["D1"]["n_differing"] == 0 and g["runs"]["W1"]["n_differing"] == 2


def test_g1_fails_on_a_D_run_whose_step_0_tensor_differs():
    r0 = _state(1.0)
    g = S.g1_seed_split(r0, {"D1": _state(1.0), "D2": _state(1.0, {"layer.bias": np.ones(2) + 5}), "W1": _state(2.0),
                             "W2": _state(3.0)})
    assert not g["pass"] and not g["d_pass"]
    assert g["w_pass"]


def test_g1_fails_when_a_W_run_equals_R0():
    # the provenance check: the uncorrected weight-seed releases would fail it
    g = S.g1_seed_split(_state(1.0), {"W1": _state(1.0), "W2": _state(2.0)})
    assert not g["pass"] and not g["w_pass"]
    assert g["d_pass"]                      # no D run supplied: the D half is vacuous, the W half is not


def test_g1_fails_when_the_two_W_runs_are_equal():
    g = S.g1_seed_split(_state(1.0), {"W1": _state(2.0), "W2": _state(2.0)})
    assert not g["pass"] and not g["w_pass"]


def test_g1_fails_when_the_key_sets_differ():
    g = S.g1_seed_split(_state(1.0), {"D1": _state(1.0, {"other.weight": np.ones(1)}), "W1": _state(2.0),
                                      "W2": _state(3.0)})
    assert not g["pass"] and not g["d_pass"]


def test_g1_ignores_rotary_frequencies():
    g = S.g1_seed_split(_state(1.0), {"D1": _state(1.0, {"rotary.inv_freq": np.arange(3.0) + 1}),
                                      "W1": _state(2.0), "W2": _state(3.0)})
    assert g["pass"]


# ---------------------------------------------------------------- G2

def test_g2_passes_on_exact_zeros():
    g = S.g2_measures({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0, 1: 0.0})
    assert g["pass"]
    assert all(v["pass"] for v in g["repeat"].values()) and all(v["pass"] for v in g["permutation"].values())
    assert g["repeat"]["M"]["value"] == 0.0 and g["permutation"]["1"]["matched"] == 0.0
    assert set(g["repeat"]) == {"P", "M", "O", "V"} and set(g["permutation"]) == {"0", "1"}


def test_g2_fails_on_a_repeat_that_is_not_zero():
    g = S.g2_measures({"P": 0.0, "M": 1e-9, "O": 0.0, "V": 0.0}, {0: 0.0})
    assert not g["pass"] and not g["repeat"]["M"]["pass"]


def test_g2_fails_on_an_unrecovered_permutation():
    g = S.g2_measures({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0, 1: 0.02})
    assert not g["pass"] and not g["permutation"]["1"]["pass"]
    assert g["permutation"]["0"]["pass"] and g["permutation"]["1"]["matched"] == 0.02


def test_g2_permutation_recovered_through_the_measures():
    # the design's G2: R0's final heads permuted within a layer, with their output blocks, are matched back at 0
    rng = np.random.default_rng(9)
    H, D = 4, 6
    ov = rng.normal(size=(H, D, D))
    perm = np.array([2, 3, 0, 1])
    cost = np.array([[np.linalg.norm(ov[i] - ov[j]) for j in perm] for i in range(H)])
    d, pairs = MS.matched_distance(cost)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert sorted(j for _, j in pairs) == list(range(H)) and all(cost[i, j] == 0.0 for i, j in pairs)


def test_g2_duplicate_head_control_accepts_any_zero_cost_assignment():
    assert S.g2_duplicate_control() == 0.0
    g = S.g2_measures({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0}, duplicate_matched=0.0)
    assert g["pass"] and g["duplicate"]["pass"]
    g = S.g2_measures({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0}, duplicate_matched=0.01)
    assert not g["pass"] and not g["duplicate"]["pass"]


def test_g2_nonzero_controls_have_their_known_values():
    c = S.g2_nonzero_controls()
    assert set(c) == {"P_disjoint", "V_doubled", "M_independent", "M_rotated_scaled"}
    g = S.g2_measures({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0}, duplicate_matched=0.0, nonzero=c)
    assert g["pass"], g["nonzero"]
    # a measure that always returns 0 fails the controls
    zeroed = {k: (0.0, rule) for k, (v, rule) in c.items()}
    g = S.g2_measures({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0}, duplicate_matched=0.0, nonzero=zeroed)
    assert not g["pass"] and not g["nonzero"]["P_disjoint"]["pass"] and not g["nonzero"]["V_doubled"]["pass"]
    assert not g["nonzero"]["M_independent"]["pass"]


G2_ZERO = ({"P": 0.0, "M": 0.0, "O": 0.0, "V": 0.0}, {0: 0.0})


def test_g2_m_independent_alone_failing_fails_g2():
    """Alden #13: with every other control valid, M_independent = 0 must fail G2 on its own (the first build's
    (value, 0.5, inf) accepted it). Sabotage that turns this red: score211.nonzero_ok treating at_least as approx with an
    infinite tolerance."""
    c = S.g2_nonzero_controls()
    for bad in (0.0, 0.49, float("nan"), float("inf")):
        c2 = dict(c, M_independent=(bad, c["M_independent"][1]))
        g = S.g2_measures(*G2_ZERO, duplicate_matched=0.0, nonzero=c2)
        assert not g["nonzero"]["M_independent"]["pass"] and not g["pass"], bad
        assert all(g["nonzero"][k]["pass"] for k in c if k != "M_independent")
    for ok in (0.5, 0.8):
        g = S.g2_measures(*G2_ZERO, duplicate_matched=0.0, nonzero=dict(c, M_independent=(ok, ("at_least", 0.5))))
        assert g["pass"], ok


def test_g2_nonzero_rules_refuse_malformed_specs():
    with pytest.raises(ValueError):
        S.nonzero_ok(0.6, ("approx", 0.5, float("inf")))
    with pytest.raises(ValueError):
        S.nonzero_ok(0.6, ("at_least", float("nan")))
    with pytest.raises(ValueError):
        S.nonzero_ok(0.6, ("roughly", 0.5))


def test_g2_m_independent_uses_the_pooled_recorded_shape():
    """The control's features are set A's pooled positions (48 x 85) by 768, and at that shape independent features
    give M well above 0.5; at the first build's 85 x 768 they do not (near 0.1)."""
    assert S.N_A_DOCS * S.N_POSITIONS == 48 * 85
    assert S.g2_nonzero_controls()["M_independent"][0] > 0.6
    rng = np.random.default_rng(5)
    assert MS.cka_distance(rng.normal(size=(85, 768)), rng.normal(size=(85, 768))) < 0.5


def test_g3_rules():
    assert S.g3_quarter(0.25, 1.0) and not S.g3_quarter(0.24, 1.0)
    assert S.g3_four_times(0.1, 0.4) and not S.g3_four_times(0.1, 0.41)


# ---------------------------------------------------------------- the predictions, at their exact bars
#
# The fixtures follow score211's data contract (one text set, documents first). Default: every prediction passes.
# Several tests vary the data ACROSS DOCUMENTS so that a bootstrap over single documents (the defect fixed
# 2026-11-13) and a bootstrap over resampled means give different verdicts.

STEPS = list(S.PYTHIA_STEPS)                     # 0 ... 143000, final last
NS = len(STEPS)
ND = 48


def _si(diag=0.05, off=0.5, h=12):
    """One layer's same-index matrix: row i is head i of run b against every head of R0 at final."""
    m = np.full((h, h), off)
    np.fill_diagonal(m, diag)
    return m


def _si_shifted(h=12):
    """A layer where every head's nearest R0 head is the next one: no head is same-index compatible."""
    m = np.full((h, h), 0.5)
    for i in range(h):
        m[i, (i + 1) % h] = 0.05
    return m


# within-run P(s, final), head mean: f_P crosses 0.5 between 512 and 1000; below 0.2 from step 1000 on
P_CURVE = [0.5, 0.4, 0.3, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
# M(s, final) and V(s, final): f crosses 0.5 between 10000 and 16000 (later than P)
M_CURVE = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.0]


def _pythia(n_boot=1000):
    runs = {}
    for r in S.PYTHIA_RUNS:
        cross_kind = _si() if r in S.DATA_RUNS or r == "R0" else np.full((12, 12), 0.2)
        runs[r] = {
            "P_within": np.tile(np.array(P_CURVE), (ND, 12, 1)),
            "V": np.tile(np.array(M_CURVE), (12, 12, 1)),
            "M_point": np.tile(np.array(M_CURVE), (12, 1)),
            "M_boot": np.tile(np.array(M_CURVE), (n_boot, 12, 1)),
            "cross": {st: np.tile(cross_kind, (ND, 12, 1, 1)) for st in (0, S.THREAD, S.FINAL)},
        }
    runs["R0"]["P_thread"] = np.full((ND, 12), 0.05)
    runs["R0"]["between_heads"] = [0.5] * 12
    return {"runs": runs}


def _with_runs(f):
    """A floor dict as distances211.load returns it: the six-final maximum per measure, plus every run's own final
    floor (211.3's tolerance) and step-10000 floors (211.8), here all equal to the maximum."""
    f = {k: list(v) for k, v in f.items()}
    f["per_run"] = {r: {k: list(f[k]) for k in ("P", "M", "V")} for r in S.PYTHIA_RUNS}
    f["thread"] = {r: {k: list(f[k]) for k in ("P", "M", "V")} for r in ("R0", *S.WEIGHT_RUNS)}
    return f


FLOOR0 = _with_runs({"P": [0.0] * 12, "M": [0.0] * 12, "V": [0.0] * 12})


def _floor(**kw):
    f = {k: list(FLOOR0[k]) for k in ("P", "M", "V")}
    for k, v in kw.items():
        f[k] = [v] * 12
    return _with_runs(f)


def _counts(n_boot=1000):
    return MS.counts_of(S.resample_indices(ND, n_boot), ND)


def test_the_default_fixture_passes_every_pythia_prediction():
    d, c = _pythia(), _counts()
    for name in ("211.1", "211.2", "211.3", "211.4", "211.5", "211.8"):
        assert S.PREDICTIONS[name](d, FLOOR0, c)["pass"], name


def test_weighted_mean_and_matched_per_draw_refit_inside_the_draw():
    rng = np.random.default_rng(3)
    docs = rng.uniform(size=(5, 4, 4))
    counts = np.array([[2, 0, 0, 3, 0], [1, 1, 1, 1, 1]], float)
    m = S.weighted_mean(docs, counts)
    assert np.allclose(m[0], (2 * docs[0] + 3 * docs[3]) / 5) and np.allclose(m[1], docs.mean(0))
    got = S.matched_per_draw(docs, counts)
    for b in range(2):
        assert got[b] == pytest.approx(MS.matched_distance(m[b])[0])


def test_211_1_nine_of_twelve_passes_and_eight_fails():
    d = _pythia()
    for L in range(3):                                   # 3 layers never cross for W1: 9 count
        d["runs"]["W1"]["P_within"][:, L] = [0.5, 0.45, 0.4, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0]
    assert S.pred_211_1(d, FLOOR0, _counts())["layers_counting"] == 9
    d["runs"]["W2"]["P_within"][:, 3] = [0.5, 0.45, 0.4, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0]
    v = S.pred_211_1(d, FLOOR0, _counts())
    assert v["layers_counting"] == 8 and not v["pass"]


def test_211_1_resamples_the_mean_curve_not_single_documents():
    """Half the documents never fall below the cross-run 0.2, half fall to 0.05 at step 1000. The mean curve is 0.15
    at 1000 in every plausible resample (crossing at 1000), while a single-document bootstrap would see censored
    documents half the time and fail every layer."""
    d = _pythia()
    early = [0.5, 0.4, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
    never = [0.5, 0.4, 0.3, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0]
    for b in S.WEIGHT_RUNS:
        d["runs"][b]["P_within"] = np.array([[early if k % 2 else never] * 12 for k in range(ND)])
    v = S.pred_211_1(d, FLOOR0, _counts())
    assert v["layers_counting"] == 12
    assert v["per_layer"][0]["runs"]["W1"]["ci"][1] == 1000


def test_211_1_unmeasured_when_the_floor_is_a_quarter_of_the_cross_run_distance():
    v = S.pred_211_1(_pythia(), _floor(P=0.05), _counts(200))        # 0.05 >= 0.2 / 4
    assert v["layers_counting"] == 0 and v["per_layer"][0]["runs"]["W1"]["unmeasured"]


def test_211_2_nine_of_twelve_passes_and_eight_fails():
    d = _pythia()
    for L in range(3):                                   # 3 layers where W1's matched distance equals the D median
        d["runs"]["W1"]["cross"][S.FINAL][:, L] = _si()
    assert S.pred_211_2(d, FLOOR0, _counts(200))["layers_counting"] == 9
    d["runs"]["W2"]["cross"][S.FINAL][:, 3] = _si()
    v = S.pred_211_2(d, FLOOR0, _counts(200))
    assert v["layers_counting"] == 8 and not v["pass"]


def test_211_2_unmeasured_when_the_floor_is_a_quarter_of_a_W_distance():
    v = S.pred_211_2(_pythia(), _floor(P=0.05), _counts(200))
    assert v["layers_counting"] == 0 and v["per_layer"][0]["runs"]["W1"]["unmeasured"]


def test_211_3_at_108_passes_and_96_fails():
    d = _pythia()
    for b in S.DATA_RUNS:
        for L in range(9, 12):                           # 9 layers all compatible (108), 3 with none
            d["runs"][b]["cross"][S.FINAL][:, L] = _si_shifted()
    v = S.pred_211_3(d, FLOOR0, _counts(200))
    assert v["per_run"]["D1"]["n_compatible"] == 108 and v["pass"]
    d["runs"]["D2"]["cross"][S.FINAL][:, 8] = _si_shifted()
    v = S.pred_211_3(d, FLOOR0, _counts(200))
    assert v["per_run"]["D2"]["n_compatible"] == 96 and not v["pass"]


def test_211_3_prints_unique_and_ambiguous_counts_and_unmeasured_layers():
    d = _pythia()
    # per layer: head 1 ties its own R0 head with head 2 (ambiguous at floor 0, still compatible); head 3's nearest is
    # head 4 (not compatible, unique); the other 10 heads are nearest themselves (compatible, unique)
    m = _si()
    m[1, 2] = 0.05
    m[3, 3], m[3, 4] = 0.5, 0.05
    d["runs"]["D1"]["cross"][S.FINAL][:] = m
    v = S.pred_211_3(d, FLOOR0, _counts(200), idx=S.resample_indices(ND, 200))
    assert v["per_run"]["D1"]["unique"] == 132 and v["per_run"]["D1"]["ambiguous"] == 12
    assert v["per_run"]["D1"]["n_compatible"] == 132
    assert v["per_run"]["D1"]["per_layer"][0]["gap_interval"][3] == pytest.approx([0.45, 0.45])
    v = S.pred_211_3(d, _floor(P=0.125), _counts(200))               # 0.125 >= 0.5 / 4: unmeasured
    assert v["per_run"]["D1"]["n_compatible"] == 0 and v["per_run"]["D1"]["per_layer"][0]["unmeasured"]


def test_211_3_tolerance_is_the_smaller_compared_final_floor_not_the_six_final_max():
    """Alden #13's case: every head's gap is 0.0002; D1's and R0's own final floors are 0.0001 and a weight-seed run's
    is 0.0003, so the six-final maximum is 0.0003. The verdict uses min(D1, R0) = 0.0001 (not compatible); the maximum
    is a sensitivity column (compatible). Between-head P 0.1 keeps the gate measured (0.0003 < 0.1 / 4). Sabotage that
    turns this red: tolerance = fl[L] (the six-final maximum) in pred_211_3."""
    d = _pythia()
    m = np.full((12, 12), 0.1)
    np.fill_diagonal(m, 0.1)
    for i in range(12):
        m[i, (i + 1) % 12] = 0.1 - 0.0002                          # row minimum 0.0998, diagonal 0.1: gap 0.0002
    for b in S.DATA_RUNS:
        d["runs"][b]["cross"][S.FINAL] = np.tile(m, (ND, 12, 1, 1))
    d["runs"]["R0"]["between_heads"] = [0.1] * 12
    f = _floor(P=0.0003)
    for r in ("R0", *S.DATA_RUNS):
        f["per_run"][r]["P"] = [0.0001] * 12
    v = S.pred_211_3(d, f, _counts(200))
    row = v["per_run"]["D1"]
    assert row["n_compatible"] == 0 and not v["pass"]
    assert row["n_compatible_at_max_floor"] == 144
    assert row["per_layer"][0]["tolerance"] == pytest.approx(0.0001)
    # the smaller of the two compared finals: a large R0 floor does not loosen D1's tolerance
    f["per_run"]["R0"]["P"] = [0.0003] * 12
    assert S.pred_211_3(d, f, _counts(200))["per_run"]["D1"]["n_compatible"] == 0


def test_threshold_is_ceil_0_975_b():
    assert S.threshold(1000) == 975 and S.threshold(200) == 195


def test_211_4_five_of_six_passes_and_four_fails():
    d = _pythia()
    late_p = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.0]   # f_P crosses after 32000, after t_M
    for L in range(12):
        d["runs"]["R0"]["P_within"][:, L] = late_p
    v = S.pred_211_4(d, FLOOR0, _counts())
    assert v["layers_counting"] == 12 and v["per_layer"][0]["n"] == 5          # 5 of 6 runs: counts
    for L in range(12):
        d["runs"]["D1"]["P_within"][:, L] = late_p
    v = S.pred_211_4(d, FLOOR0, _counts())
    assert v["layers_counting"] == 0 and not v["pass"]


def test_211_4_reads_t_M_per_draw_and_needs_975_of_1000():
    """M_boot varies by draw: in k draws M settles at step 128 (before t_P), so t_P < t_M fails in those draws."""
    early_m = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
    for k, want in ((25, True), (26, False)):
        d = _pythia()
        for r in S.PYTHIA_RUNS:
            d["runs"][r]["M_boot"][:k] = early_m
        v = S.pred_211_4(d, FLOOR0, _counts())
        run = v["per_layer"][0]["runs"]["R0"]
        assert run["holds_in"] == 1000 - k and run["counts"] is want


def test_211_4_a_draw_censored_against_the_prediction_is_a_failure():
    """t_P censored in a draw (the resampled P never falls to half) fails that draw; t_M censored is FOR the prediction
    (t_P < censored holds)."""
    d = _pythia()
    for r in S.PYTHIA_RUNS:
        d["runs"][r]["M_boot"][:] = [1.0] * 11 + [0.0]                     # t_M censored in every draw
    v = S.pred_211_4(d, FLOOR0, _counts())
    assert v["per_layer"][0]["runs"]["R0"]["holds_in"] == 1000
    # documents 0..23 never settle and 24..47 settle at 1000: a draw holds when its mean f at 64000 is at most 0.5,
    # i.e. when it draws at least as many settling documents as not; the count must equal that directly
    never = [0.5] * 11 + [0.0]
    early = [0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    d = _pythia()
    d["runs"]["R0"]["P_within"] = np.array([[never if k < 24 else early] * 12 for k in range(ND)])
    c = _counts()
    want = int(((c[:, 24:].sum(1) * 0.0 + c[:, :24].sum(1) * 0.5) / ND <= 0.25).sum())
    assert S.pred_211_4(d, FLOOR0, c)["per_layer"][0]["runs"]["R0"]["holds_in"] == want


def test_211_4_unmeasured_when_a_denominator_is_at_or_below_four_times_the_floor():
    v = S.pred_211_4(_pythia(), _floor(M=0.25), _counts())                  # M(step 0) = 1.0 <= 4 x 0.25
    assert v["layers_counting"] == 0 and v["per_layer"][0]["runs"]["R0"]["unmeasured"]
    v = S.pred_211_4(_pythia(), _floor(M=0.24), _counts())
    assert v["layers_counting"] == 12


def test_211_5_five_of_six_passes_and_four_fails_and_V_is_one_value():
    d = _pythia()
    early_v = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
    d["runs"]["R0"]["V"][:] = early_v
    v = S.pred_211_5(d, FLOOR0, _counts())
    assert v["layers_counting"] == 12 and v["per_layer"][0]["n"] == 5
    d["runs"]["D3"]["V"][:] = early_v
    assert S.pred_211_5(d, FLOOR0, _counts())["layers_counting"] == 0
    # V is the ratio of head means: one head at 0 and one at 1 is a mean of 0.5, not a mean of fractions
    d = _pythia()
    d["runs"]["R0"]["V"][:, :6] = [2.0] * 11 + [0.0]
    d["runs"]["R0"]["V"][:, 6:] = [0.0] + [0.5] * 10 + [0.0]
    t = S.pred_211_5(d, FLOOR0, _counts(200))["per_layer"][0]["runs"]["R0"]["t_V"]
    assert t == MS.CENSORED                                                 # (2 + 0.5) / 2 / ((2 + 0) / 2) > 0.5


def _olmo(n_boot=1000):
    p = {("mid", "instruct"): np.full((ND, 16), 0.1), ("3T", "4T"): np.full((ND, 16), 0.4),
         ("21B", "4T"): np.full((ND, 16), 0.2), ("step0", "4T"): np.full((ND, 16), 1.0)}
    m_pt = {("mid", "instruct"): np.full(16, 0.3), ("3T", "4T"): np.full(16, 0.3)}
    m_bt = {k: np.tile(v, (n_boot, 1)) for k, v in m_pt.items()}
    return {"P": p, "M_point": m_pt, "M_boot": m_bt}


OFLOOR0 = {"P": [0.0] * 16, "M": [0.0] * 16}


def test_211_6_twelve_of_sixteen_passes_and_eleven_fails():
    o = _olmo()
    o["P"][("mid", "instruct")][:, :4] = 0.5                               # ratio P 1.25 > ratio M 1.0 in 4 layers
    assert S.pred_211_6(o, OFLOOR0, _counts())["layers_counting"] == 12
    o["P"][("mid", "instruct")][:, 4] = 0.5
    v = S.pred_211_6(o, OFLOOR0, _counts())
    assert v["layers_counting"] == 11 and not v["pass"]


def test_211_6_takes_the_ratio_of_resampled_means():
    """P(mid, instruct) alternates 0.0 and 0.6 across documents (mean 0.3, ratio 0.75 against M's 1.0): every draw's
    ratio of means is well below 1, while a quantile of per-document ratios (0 or 1.5) would put the 97.5th percentile
    at 1.5 - 1.0 = 0.5 and fail."""
    o = _olmo()
    o["P"][("mid", "instruct")] = np.array([[0.0 if k % 2 else 0.6] * 16 for k in range(ND)])
    assert S.pred_211_6(o, OFLOOR0, _counts())["layers_counting"] == 16


def test_211_6_unmeasured_when_a_late_distance_is_at_or_below_four_times_the_floor():
    # M(3T, 4T) = 0.3: floor 0.075 gives 0.3 <= 0.3, unmeasured; 0.07 gives 0.3 > 0.28, measured (P(3T, 4T) = 0.4
    # stays above both)
    assert S.pred_211_6(_olmo(), {"P": [0.0] * 16, "M": [0.075] * 16}, _counts())["layers_counting"] == 0
    assert S.pred_211_6(_olmo(), {"P": [0.0] * 16, "M": [0.07] * 16}, _counts())["layers_counting"] == 16
    # the 2.5th percentile rule: M draws spread so that the lowest 2.5% sit at the floor
    o = _olmo()
    o["M_boot"][("3T", "4T")][:, :] = 0.3
    o["M_boot"][("3T", "4T")][:30] = 0.01
    assert S.pred_211_6(o, {"P": [0.0] * 16, "M": [0.01] * 16}, _counts())["layers_counting"] == 0


def test_211_7_twelve_of_sixteen_passes_and_eleven_fails():
    o = _olmo()
    o["P"][("21B", "4T")][:, :4] = 0.6
    assert S.pred_211_7(o, OFLOOR0, _counts())["layers_counting"] == 12
    o["P"][("21B", "4T")][:, 4] = 0.6
    assert S.pred_211_7(o, OFLOOR0, _counts())["layers_counting"] == 11


def test_211_7_takes_the_ratio_of_resampled_means():
    """P(21B, 4T) alternates 0.2 and 0.6 (mean 0.4): resampled means stay near 0.4 (97.5th percentile about 0.45 <=
    0.5), while per-document fractions put the 97.5th percentile at 0.6."""
    o = _olmo()
    o["P"][("21B", "4T")] = np.array([[0.2 if k % 2 else 0.6] * 16 for k in range(ND)])
    v = S.pred_211_7(o, OFLOOR0, _counts())
    assert v["layers_counting"] == 16 and v["per_layer"][0]["ci"][1] < 0.5


def test_211_7_unmeasured_when_the_step_zero_distance_is_at_or_below_four_times_the_floor():
    assert S.pred_211_7(_olmo(), {"P": [0.25] * 16, "M": [0.0] * 16}, _counts())["layers_counting"] == 0


def test_211_8_nine_of_twelve_passes_and_eight_fails():
    d = _pythia()
    for L in range(3):
        d["runs"]["R0"]["P_thread"][:, L] = 0.3                            # above W's 0.2 in 3 layers
    assert S.pred_211_8(d, FLOOR0, _counts(200))["layers_counting"] == 9
    d["runs"]["W2"]["cross"][S.THREAD][:, 3] = 0.01                         # W2's cross-run below R0's within
    v = S.pred_211_8(d, FLOOR0, _counts(200))
    assert v["layers_counting"] == 8 and not v["pass"]


def test_211_8_unmeasured_when_the_floor_is_a_quarter_of_a_cross_run_distance():
    v = S.pred_211_8(_pythia(), _floor(P=0.05), _counts(200))
    assert v["layers_counting"] == 0 and v["per_layer"][0]["runs"]["W1"]["unmeasured"]


def test_211_8_reads_the_step_10000_floors_not_the_finals():
    """211.8 compares step-10000 checkpoints, so its floor is the larger of R0's and the W run's step-10000 floors.
    With every final floor 0 and the step-10000 floors at 0.05 (a quarter of the 0.2 cross-run P), every layer is
    unmeasured; a final-floor reading would count them. Sabotage: floor["P"] in place of floor["thread"] in pred_211_8."""
    f = _floor()
    for r in f["thread"]:
        f["thread"][r]["P"] = [0.0] * 12
    base = S.pred_211_8(_pythia(), f, _counts(200))["layers_counting"]
    f["thread"]["W2"]["P"] = [0.05] * 12                                  # only W2's step-10000 floor is high
    v = S.pred_211_8(_pythia(), f, _counts(200))
    assert base > 0 and v["layers_counting"] == 0
    assert v["per_layer"][0]["runs"]["W2"]["unmeasured"] and v["per_layer"][0]["runs"]["W2"]["floor"] == 0.05



def _set():
    return {"pythia": _pythia(), "pythia_floor": FLOOR0, "olmo": _olmo(), "olmo_floor": OFLOOR0}


def _gates(g2=True, d=True, w=True):
    return {"G2": {"pass": g2}, "G1": {"d_pass": d, "w_pass": w}}


def test_score_all_holds_every_prediction_on_the_default_sets():
    idx = S.resample_indices(ND, 1000)
    out = S.score_all(_gates(), _set(), _set(), idx, idx)
    assert {k: v["verdict"] for k, v in out["verdicts"].items()} == {k: "held" for k in S.PREDICTIONS}
    assert not out["headline_withdrawn"]


def test_a_failed_W_half_withholds_211_1_211_2_and_211_8_but_not_211_4():
    idx = S.resample_indices(ND, 1000)
    out = S.score_all(_gates(w=False), _set(), _set(), idx, idx)
    for k in ("211.1", "211.2", "211.8"):
        assert out["verdicts"][k]["verdict"] == "withheld"
    assert out["verdicts"]["211.4"]["verdict"] == "held" and out["headline_withdrawn"]


def test_a_failed_D_half_withholds_211_2_and_211_3_only():
    idx = S.resample_indices(ND, 1000)
    out = S.score_all(_gates(d=False), _set(), _set(), idx, idx)
    assert out["verdicts"]["211.3"]["verdict"] == "withheld" and out["verdicts"]["211.2"]["verdict"] == "withheld"
    assert out["verdicts"]["211.1"]["verdict"] == "held"


def test_a_failed_G2_withholds_every_prediction():
    idx = S.resample_indices(ND, 1000)
    out = S.score_all(_gates(g2=False), _set(), _set(), idx, idx)
    assert all(v["verdict"] == "withheld" for v in out["verdicts"].values())


def test_set_B_reversing_211_1_by_more_than_one_layer_withdraws_the_headline():
    idx = S.resample_indices(ND, 1000)
    b = _set()
    for L in range(2):
        b["pythia"]["runs"]["W1"]["P_within"][:, L] = [0.5, 0.45, 0.4, 0.35, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0]
    out = S.score_all(_gates(), _set(), b, idx, idx)
    assert out["verdicts"]["211.1"]["verdict"] == "held" and out["headline_withdrawn"]


# ---------------------------------------------------------------- P0 and cosine as additive parts (2026-11-13)

def _causal_rows(rng, H, T, sink=None):
    """Random causal attention rows [H, T, T]; heads in `sink` put all but 1e-5 of every row's mass on key 0."""
    x = rng.standard_normal((H, T, T))
    x = np.where(np.tril(np.ones((T, T), bool)), x, -np.inf)
    w = np.exp(x - x.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    for h in sink or ():
        w[h, :, :] = 0.0
        w[h, :, 0] = 1.0
        w[h, 1:, 1] += 1e-5                          # under P0_KEEP outside key 0: every P0 query dropped
        w[h, 1:, 0] -= 1e-5
    return w.astype(np.float32)


def test_p0_parts_of_one_document_equal_pattern_distance_P0():
    # red if p0_parts_matrix's kept mask or renormalisation differs from the per-pair P0
    rng = np.random.default_rng(3)
    a, b = _causal_rows(rng, 3, 12), _causal_rows(rng, 3, 12)
    sums, kept, total = MS.p0_parts_matrix(a, b)
    p0, dropped = MS.p0_from_parts(sums[None], kept[None], total, np.ones((1, 1)))
    for i in range(3):
        for j in range(3):
            assert p0[0, i, j] == pytest.approx(MS.pattern_distance(a[i], b[j], variant="P0"), abs=1e-12)
    assert total == 10 and (dropped == 0).all()


def test_p0_from_parts_pools_queries_over_documents_and_weights_draws():
    # red if P0 over documents is a mean of per-document means, or if counts do not weight documents
    rng = np.random.default_rng(4)
    docs = [(_causal_rows(rng, 2, 10), _causal_rows(rng, 2, 10)) for _ in range(3)]
    parts = [MS.p0_parts_matrix(a, b) for a, b in docs]
    sums = np.stack([p[0] for p in parts]); kept = np.stack([p[1] for p in parts]); total = parts[0][2]
    pooled = sums.sum(axis=0) / kept.sum(axis=0)
    p0, _ = MS.p0_from_parts(sums, kept, total, np.ones((1, 3)))
    assert np.allclose(p0[0], pooled, rtol=0, atol=1e-12)
    p0w, _ = MS.p0_from_parts(sums, kept, total, np.array([[2.0, 0.0, 0.0]]))
    assert np.allclose(p0w[0], sums[0] / kept[0], rtol=0, atol=1e-12)


def test_a_sink_head_has_no_P0_and_does_not_raise():
    # red if p0_parts_matrix raises on a head pair with every query dropped, or if such a pair gets a value
    rng = np.random.default_rng(5)
    a, b = _causal_rows(rng, 2, 10, sink=[0]), _causal_rows(rng, 2, 10)
    sums, kept, total = MS.p0_parts_matrix(a, b)
    assert (kept[0] == 0).all()
    p0, dropped = MS.p0_from_parts(sums[None], kept[None], total, np.ones((1, 1)))
    assert np.isnan(p0[0, 0]).all() and np.isfinite(p0[0, 1]).all()
    assert (dropped[0, 0] == total).all()


def test_p0_needs_at_least_half_the_queries():
    # red if the more-than-half rule is off by one or reversed (design: a head with more than half its queries
    # dropped has no P0)
    sums, kept = np.array([[1.0], [1.0]]), np.array([[5], [4]])
    p0, dropped = MS.p0_from_parts(sums, kept, 10, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert p0[0, 0] == pytest.approx(0.2) and dropped[0, 0] == 5          # exactly half dropped: kept
    assert np.isnan(p0[1, 0]) and dropped[1, 0] == 6                      # more than half: no P0


def test_cos_from_parts_is_the_cosine_flattened_over_documents():
    # red if the cosine is computed per document and averaged (design: flattened over the documents)
    rng = np.random.default_rng(6)
    docs = [(_causal_rows(rng, 2, 8), _causal_rows(rng, 2, 8)) for _ in range(3)]
    parts = [MS.cos_parts_matrix(a, b) for a, b in docs]
    dot = np.stack([p[0] for p in parts]); sx = np.stack([p[1] for p in parts]); sy = np.stack([p[2] for p in parts])
    c = MS.cos_from_parts(dot, sx, sy, np.ones((1, 3)))[0]
    for i in range(2):
        for j in range(2):
            flat = MS.cosine_distance(np.stack([a[i] for a, _ in docs]), np.stack([b[j] for _, b in docs]))
            assert c[i, j] == pytest.approx(flat, abs=1e-12)
    per_doc = np.mean([MS.cosine_distance(a[0], b[1]) for a, b in docs])
    assert abs(c[0, 1] - per_doc) > 1e-6                                   # the two measures differ here
