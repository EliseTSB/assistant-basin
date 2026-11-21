"""Part 216 tests: the gates, the outcome rules, the unit thresholds, and the real runner path.

Run with the venv that has torch:
    /home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part216/test_part216.py

The model tests build a tiny random Qwen3.5-architecture model (the `_tiny_model` pattern from
`part215/test_part215.py`) so `run216.forward_row` -- the real capture, lens and DLA code -- runs end
to end on CPU. Nothing here loads the 4B checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, HERE, ROOT / 'part214'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import score216 as sc  # noqa: E402  (numpy only)
import run216  # noqa: E402  (imports torch; the numeric tests below do not need it)

PROMPTS = HERE / 'prompts216.json'
STAGE1 = ROOT / 'results' / 'part214' / 'stage1' / 'stage1.json'


# ----------------------------------------------------------------------------------------------------------------
# synthetic record builders
# ----------------------------------------------------------------------------------------------------------------

def lens_from_m(ms: list[float], argmax: int = 0) -> list[dict]:
    """A lens table whose margin m_l equals `ms[l]` (z_Al = 0, z_T = -1, so max(z_Al, z_T) = 0)."""
    out = []
    for l, m in enumerate(ms):
        out.append({'layer': l, 'z_G': float(m), 'z_Al': 0.0, 'z_T': -1.0, 'm': float(m),
                    'g_ali': float(m) - float(np.logaddexp(0.0, -1.0)),
                    'rank_G': 1, 'rank_Al': 2, 'rank_T': 3, 'argmax': argmax})
    return out


def make_row(key: str, ms: list[float], n_blocks: int = 32, argmax: int = 0, **extra) -> dict:
    """One forwards.jsonl row: a telescoping DLA of `n_blocks` zero blocks plus the given margins."""
    n_terms = 2 * n_blocks
    logits = [0.5, -0.25, 0.75]
    counts = {}
    for li in range(n_blocks):
        counts[f'L{li}:add1'] = 1
        counts[f'L{li}:add2'] = 1
    row = {
        'key': key, 'kind': 'stage1', 'arm': key.rsplit(':', 1)[0],
        'question_index': int(key.rsplit(':', 1)[1]) if key.rsplit(':', 1)[1].isdigit() else 0,
        'lens': lens_from_m(ms, argmax=argmax),
        'native_g_ali': 0.0, 'native_g_al': 0.0, 'native_logits': [0.0, 0.0, 0.0],
        'dla': {'emb_term': logits, 'terms': [[0.0, 0.0, 0.0] for _ in range(n_terms)],
                'fp32_logits': logits, 'frozen_scale': 1.0,
                'rounding_norm': [0.0] * n_terms, 'raw_norm': [0.0] * n_terms,
                'increment_norm': [0.0] * n_terms},
        'boundary_counts': counts, 'boundary_report': {'missing': [], 'duplicated': []},
        'dtypes': {'h0': 'torch.bfloat16'}, 'n_blocks': n_blocks,
    }
    row.update(extra)
    return row


def control_side(arm: str, layers_on: list[int], n: int = 24, final: float | None = None) -> dict:
    """`n` controls qualifying exactly on `layers_on`, with a final contrast on the expected side."""
    want_positive = arm == 'control_google'
    final = (1.0 if want_positive else -1.0) if final is None else final
    rows = {}
    for qi in range(n):
        ms = [0.0] * sc.N_LAYERS
        for l in layers_on:
            ms[l] = 1.0 if want_positive else -1.0
        ms[-1] = final if qi == 0 else (final if want_positive else final)
        rows[f'{arm}:{qi}'] = make_row(f'{arm}:{qi}', ms)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# the analytic G1 fixture and the bf16 emulation
# ----------------------------------------------------------------------------------------------------------------

def test_analytic_bf16_fixture():
    """DESIGN.md G1 fixture 1: residual 1, write 1/256 -> realized 0, rounding -1/256."""
    result = run216.selftest()
    analytic = result['checks']['analytic_bf16']
    assert analytic['pass'] is True
    assert analytic['realized_increment'] == 0.0
    assert analytic['rounding_residual'] == pytest.approx(-1.0 / 256.0, abs=1e-12)
    assert analytic['bf16_emulation_matches_ulp'] is True


def test_bf16_round_matches_torch():
    """The NumPy emulation is only useful if it agrees with the real cast, ties included."""
    torch = pytest.importorskip('torch')
    values = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, 0.1, -2.5, 3.0 + 2 ** -9],
                      dtype=np.float32)
    got = sc.bf16_round(values)
    want = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).float().numpy()
    assert np.array_equal(got, want)


def test_bf16_round_is_half_to_even_at_the_tie():
    # ulp at 1.0 is 2**-7, so 1 + 2**-8 is the tie and 1 + 3*2**-8 is the tie above it.
    assert float(sc.bf16_round(np.float32(1.0 + 2 ** -8))) == 1.0              # rounds down, even
    assert float(sc.bf16_round(np.float32(1.0 + 3 * 2 ** -8))) == 1.0 + 2 ** -7 * 2  # up, even


def test_dropped_increment_control_fails_g1():
    """DESIGN.md G1 fixture 2: one increment dropped must FAIL, the intact sum must pass."""
    result = run216.selftest()
    dropped = result['checks']['dropped_increment_control']
    assert dropped['intact_pass'] is True
    assert dropped['dropped_pass'] is False


# ----------------------------------------------------------------------------------------------------------------
# G0
# ----------------------------------------------------------------------------------------------------------------

def _receipt_for(arms, values):
    return {'arms': {arm: {'g_ali': values[arm]} for arm in arms}}


def test_g0_passes_on_matching_receipt():
    receipt = _receipt_for(sc.G0_ARMS, {arm: [1.0] * 25 for arm in sc.G0_ARMS})
    rows = {}
    for arm in sc.G0_ARMS:
        for qi in range(sc.N_QUESTIONS):
            rows[f'{arm}:{qi}'] = make_row(f'{arm}:{qi}', [0.0] * sc.N_LAYERS,
                                           native_g_ali=1.0 + 0.01 * (qi % 2))
    gate = sc.gate0(rows, receipt)
    assert gate['pass'] is True
    assert gate['n_values'] == 13 * 24 == 312
    assert gate['max_abs_delta'] == pytest.approx(0.01)


def test_g0_fails_outside_tolerance():
    receipt = _receipt_for(sc.G0_ARMS, {arm: [1.0] * 25 for arm in sc.G0_ARMS})
    rows = {}
    for arm in sc.G0_ARMS:
        for qi in range(sc.N_QUESTIONS):
            rows[f'{arm}:{qi}'] = make_row(f'{arm}:{qi}', [0.0] * sc.N_LAYERS,
                                           native_g_ali=1.0 if qi else 1.021)
    gate = sc.gate0(rows, receipt)
    assert gate['pass'] is False
    assert gate['n_failures'] == len(sc.G0_ARMS)   # qi=0 fails in every arm


def test_g0_ignores_question_24():
    """The receipt's q24 is a separate positive control; G0 covers q0..23 only."""
    receipt = _receipt_for(sc.G0_ARMS, {arm: [1.0] * 25 for arm in sc.G0_ARMS})
    rows = {f'{arm}:{qi}': make_row(f'{arm}:{qi}', [0.0] * sc.N_LAYERS, native_g_ali=1.0)
            for arm in sc.G0_ARMS for qi in range(sc.N_QUESTIONS)}
    # A 25th value that would fail if it were included:
    receipt['arms'][sc.G0_ARMS[0]]['g_ali'][24] = 99.0
    assert sc.gate0(rows, receipt)['pass'] is True


def test_g0_fails_when_a_row_is_missing():
    receipt = _receipt_for(sc.G0_ARMS, {arm: [1.0] * 25 for arm in sc.G0_ARMS})
    rows = {f'{arm}:{qi}': make_row(f'{arm}:{qi}', [0.0] * sc.N_LAYERS, native_g_ali=1.0)
            for arm in sc.G0_ARMS for qi in range(sc.N_QUESTIONS)}
    del rows['zero:3']
    gate = sc.gate0(rows, receipt)
    assert gate['pass'] is False
    assert gate['n_values'] == 311


# ----------------------------------------------------------------------------------------------------------------
# G1 and the boundary count
# ----------------------------------------------------------------------------------------------------------------

def test_g1_telescoping_holds_and_a_dropped_term_fails():
    emb = np.array([0.3, -0.2, 0.7])
    terms = np.array([[0.1, 0.1, -0.1], [0.2, -0.3, 0.05], [-0.05, 0.4, 0.2]])
    logits = emb + terms.sum(axis=0)
    assert sc.gate1_check(emb, terms, logits)['pass'] is True
    assert sc.gate1_check(emb, terms[:2], logits)['pass'] is False


def test_g1_tolerance_is_one_millith():
    emb = np.array([0.0, 0.0, 0.0])
    terms = np.array([[1.0, 1.0, 1.0]])
    assert sc.gate1_check(emb, terms, [1.0009, 1.0, 1.0])['pass'] is True
    assert sc.gate1_check(emb, terms, [1.0011, 1.0, 1.0])['pass'] is False


def _store_with(counts: dict) -> dict:
    """A capture store with counts only (no tensors): enough for the count check."""
    return {'count': dict(counts)}


def _complete_counts(n_blocks: int) -> dict:
    counts = {}
    for li in range(n_blocks):
        counts[f'L{li}:add1'] = 1
        counts[f'L{li}:add2'] = 1
    counts['h0'] = 1
    counts['L0:in'] = 1
    return counts


def test_boundary_count_is_two_per_block_exactly_once():
    n_blocks = 30
    report = run216.boundary_report(_store_with(_complete_counts(n_blocks)), n_blocks, 0)
    assert report['missing'] == [] and report['duplicated'] == []
    assert report['n_present'] == 2 * n_blocks


def test_missing_boundary_is_caught():
    n_blocks = 4
    counts = _complete_counts(n_blocks)
    del counts['L2:add2']
    report = run216.boundary_report(_store_with(counts), n_blocks, 0)
    assert report['missing'] == ['L2:add2']
    assert report['n_present'] == 2 * n_blocks - 1


def test_duplicate_boundary_is_caught():
    n_blocks = 4
    counts = _complete_counts(n_blocks)
    counts['L1:add1'] = 2
    report = run216.boundary_report(_store_with(counts), n_blocks, 0)
    assert report['duplicated'] == ['L1:add1']


def test_gate1_catches_a_duplicate_boundary_row():
    row = make_row('unchanged:0', [0.0] * sc.N_LAYERS)
    row['boundary_counts']['L0:add1'] = 2
    gate = sc.gate1({'unchanged:0': row}, n_blocks=32)
    assert gate['pass'] is False


# ----------------------------------------------------------------------------------------------------------------
# G2
# ----------------------------------------------------------------------------------------------------------------

def test_g2_alternating_layers_have_no_usable_pair_and_are_not_read():
    evens = list(range(0, 30, 2))
    controls = {}
    controls.update(control_side('control_google', evens))
    controls.update(control_side('control_alibaba', evens))
    gate = sc.gate2(controls)
    assert gate['window'] == evens
    assert gate['n_usable_pairs'] == 0
    assert gate['pass'] is False


def test_g2_layers_14_15_16_give_two_pairs():
    controls = {}
    controls.update(control_side('control_google', [14, 15, 16]))
    controls.update(control_side('control_alibaba', [14, 15, 16]))
    gate = sc.gate2(controls)
    assert gate['window'] == [14, 15, 16]
    assert gate['usable_pairs'] == [[14, 15], [15, 16]]
    assert gate['n_usable_pairs'] == 2
    assert gate['pass'] is True


def test_g2_fewer_than_18_controls_on_their_side_is_not_read():
    controls = {}
    google = control_side('control_google', [14, 15, 16])
    # 10 of 24 Google-side controls end NEGATIVE: only 14 end on their expected side.
    for qi in range(10):
        google[f'control_google:{qi}']['lens'][-1]['m'] = -1.0
        google[f'control_google:{qi}']['lens'][-1]['z_G'] = -1.0
    controls.update(google)
    controls.update(control_side('control_alibaba', [14, 15, 16]))
    gate = sc.gate2(controls)
    assert gate['controls']['google']['n_final_expected'] == 14
    assert gate['checks']['controls_18_of_24'] is False
    assert gate['pass'] is False


def test_g2_share_threshold_is_eight_tenths():
    controls = {}
    google = control_side('control_google', [15])
    # 5 of 24 miss layer 15 -> share 19/24 = 0.79 < 0.8
    for qi in range(5):
        google[f'control_google:{qi}']['lens'][15]['m'] = -1.0
        google[f'control_google:{qi}']['lens'][15]['z_G'] = -1.0
    controls.update(google)
    controls.update(control_side('control_alibaba', [15]))
    gate = sc.gate2(controls)
    assert 15 not in gate['window']


def test_g2_ignores_controls_that_end_on_the_wrong_side():
    controls = {}
    google = control_side('control_google', [15])
    # A control that ends negative is not counted at all -- neither numerator nor denominator.
    google['control_google:0']['lens'][-1]['m'] = -1.0
    google['control_google:0']['lens'][-1]['z_G'] = -1.0
    google['control_google:0']['lens'][15]['m'] = -1.0
    google['control_google:0']['lens'][15]['z_G'] = -1.0
    controls.update(google)
    controls.update(control_side('control_alibaba', [15]))
    gate = sc.gate2(controls)
    assert gate['controls']['google']['n_final_expected'] == 23
    assert gate['window'] == [15]


# ----------------------------------------------------------------------------------------------------------------
# outcome rules
# ----------------------------------------------------------------------------------------------------------------

def _unchanged_rows(profiles: list[list[float]]) -> dict:
    return {f'unchanged:{i}': make_row(f'unchanged:{i}', ms) for i, ms in enumerate(profiles)}


def _window(*layers) -> dict:
    qualified = sorted(layers)
    return {'pass': True, 'window': qualified, 'usable_pairs': sc.usable_pairs(qualified),
            'window_min': min(qualified), 'window_max': max(qualified), 'checks': {},
            'n_usable_pairs': len(sc.usable_pairs(qualified))}


def test_early_lead_needs_both_layers_of_a_pair_and_a_negative_final():
    window = _window(10, 11)
    pair_positive_negative_final = [0.0] * sc.N_LAYERS
    pair_positive_negative_final[10] = 1.0
    pair_positive_negative_final[11] = 1.0
    pair_positive_negative_final[-1] = -1.0
    got = sc.question_outcome(make_row('u:0', pair_positive_negative_final), window['window'],
                              window['usable_pairs'])
    assert got['early_lead'] is True

    # Google ahead on only ONE layer of the pair: no early lead.
    one_layer = [0.0] * sc.N_LAYERS
    one_layer[10] = 1.0
    one_layer[-1] = -1.0
    assert sc.question_outcome(make_row('u:1', one_layer), window['window'],
                               window['usable_pairs'])['early_lead'] is False

    # Both layers positive but the model still ends in Google: no early lead.
    positive_final = [0.0] * sc.N_LAYERS
    positive_final[10] = 1.0
    positive_final[11] = 1.0
    positive_final[-1] = 1.0
    assert sc.question_outcome(make_row('u:2', positive_final), window['window'],
                               window['usable_pairs'])['early_lead'] is False


def test_no_early_lead_question_needs_every_layer_in_w_nonpositive():
    window = _window(10, 11)
    all_negative = [-1.0] * sc.N_LAYERS
    assert sc.question_outcome(make_row('u:0', all_negative), window['window'],
                               window['usable_pairs'])['no_early_lead'] is True
    # A positive OUTSIDE the window does not spoil `no early lead` ...
    outside = [-1.0] * sc.N_LAYERS
    outside[20] = 1.0
    got = sc.question_outcome(make_row('u:1', outside), window['window'], window['usable_pairs'])
    assert got['no_early_lead'] is True
    assert got['n_positive_in_window'] == 0
    # ... but a positive INSIDE it does.
    inside = [-1.0] * sc.N_LAYERS
    inside[11] = 1.0
    assert sc.question_outcome(make_row('u:2', inside), window['window'],
                               window['usable_pairs'])['no_early_lead'] is False


def test_outcome_lead_visible_at_twelve_of_twenty_four():
    window = _window(10, 11)
    profiles = []
    for i in range(sc.N_QUESTIONS):
        ms = [-1.0] * sc.N_LAYERS
        if i < 12:
            ms[10] = ms[11] = 1.0
        profiles.append(ms)
    result = sc.unchanged_outcome(_unchanged_rows(profiles), window)
    assert result['n_early_lead'] == 12
    assert result['outcome'] == sc.OUTCOME_LEAD


def test_outcome_no_early_lead_needs_at_most_two_questions_with_any_positive():
    window = _window(10, 11)
    profiles = []
    for i in range(sc.N_QUESTIONS):
        ms = [-1.0] * sc.N_LAYERS
        if i < 2:
            ms[10] = 1.0          # only one layer of the pair -> no lead, but it IS a positive in W
        profiles.append(ms)
    result = sc.unchanged_outcome(_unchanged_rows(profiles), window)
    assert result['n_any_positive_in_window'] == 2
    assert result['outcome'] == sc.OUTCOME_NONE

    profiles[2][10] = 1.0
    result = sc.unchanged_outcome(_unchanged_rows(profiles), window)
    assert result['n_any_positive_in_window'] == 3
    assert result['outcome'] == sc.OUTCOME_MIXED


def test_outcome_is_not_read_when_g2_fails():
    window = {'pass': False, 'window': None, 'usable_pairs': None, 'checks': {}, 'window_min': None}
    profiles = [[-1.0] * sc.N_LAYERS for _ in range(sc.N_QUESTIONS)]
    result = sc.unchanged_outcome(_unchanged_rows(profiles), window)
    assert result['outcome'] == sc.OUTCOME_NOT_READ


def test_divergence_index_needs_the_median_above_half_through_h_32():
    window = _window(7, 8)
    base = {f'unchanged:{i}': make_row(f'unchanged:{i}', [0.0] * sc.N_LAYERS)
            for i in range(sc.N_QUESTIONS)}
    arm = {}
    for i in range(sc.N_QUESTIONS):
        ms = [0.0] * sc.N_LAYERS
        for l in range(7, sc.N_LAYERS):
            ms[l] = 0.4        # below threshold everywhere
        arm[f'zero:{i}'] = make_row(f'zero:{i}', ms)
    assert sc.divergence_index(arm, base, window)['index'] is None

    arm = {}
    for i in range(sc.N_QUESTIONS):
        ms = [0.0] * sc.N_LAYERS
        for l in range(9, sc.N_LAYERS):
            ms[l] = 0.6
        arm[f'zero:{i}'] = make_row(f'zero:{i}', ms)
    got = sc.divergence_index(arm, base, window)
    assert got['index'] == 9
    assert got['layer'] == 'h_9'


def test_divergence_index_is_none_if_the_median_dips_later():
    window = _window(7, 8)
    base = {f'unchanged:{i}': make_row(f'unchanged:{i}', [0.0] * sc.N_LAYERS)
            for i in range(sc.N_QUESTIONS)}
    arm = {}
    for i in range(sc.N_QUESTIONS):
        ms = [0.0] * sc.N_LAYERS
        for l in range(9, sc.N_LAYERS - 1):
            ms[l] = 0.6
        ms[-1] = 0.0           # falls back at h_32
        arm[f'zero:{i}'] = make_row(f'zero:{i}', ms)
    assert sc.divergence_index(arm, base, window)['index'] is None


def test_arm_ends_in_google_is_the_sign_of_m32():
    rows = {}
    for i in range(sc.N_QUESTIONS):
        ms = [0.0] * sc.N_LAYERS
        ms[-1] = 1.0 if i % 2 == 0 else -1.0
        rows[f'op_filler:{i}'] = make_row(f'op_filler:{i}', ms)
    got = sc.arm_ends_in_google(rows, 'op_filler')
    assert got['n_ends_in_google'] == 12


# ----------------------------------------------------------------------------------------------------------------
# 216.5 / 216.6: strict ties, zero writes, the 12-question floor
# ----------------------------------------------------------------------------------------------------------------

def _series(layer, unit, ratios, writes, c_g=None, d_a=None):
    valid = [i for i, w in enumerate(writes) if w > 0]
    return {'layer': layer, 'unit': unit, 'n_questions': len(writes), 'n_valid': len(valid),
            'median_write': float(np.median(writes)) if writes else None,
            'median_c_G': float(np.median(c_g)) if c_g is not None else -1.0,
            'median_d_A': float(np.median(d_a)) if d_a is not None else 1.0,
            'median_c_G_per_write': float(np.median([ratios[i] for i in valid])) if valid else None,
            'median_d_A_per_write': float(np.median([ratios[i] for i in valid])) if valid else None,
            'no_write': len(valid) < sc.MIN_VALID_QUESTIONS}


def test_216_5_strictly_below_every_null():
    target = _series(6, 753, [-3.0] * 24, [1.0] * 24)
    nulls = [_series(6, 100 + i, [1.0 + i] * 24, [1.0] * 24) for i in range(200)]
    got = sc.prediction_216_5_unit(target, nulls)
    assert got['strictly_below_all_nulls'] is True
    assert got['pass'] is True


def test_216_5_a_tie_with_any_null_does_not_count():
    target = _series(6, 753, [-3.0] * 24, [1.0] * 24)
    nulls = [_series(6, 100 + i, [1.0 + i] * 24, [1.0] * 24) for i in range(200)]
    nulls[7] = _series(6, 107, [-3.0] * 24, [1.0] * 24)
    got = sc.prediction_216_5_unit(target, nulls)
    assert got['n_ties'] == 1
    assert got['strictly_below_all_nulls'] is False
    assert got['pass'] is False


def test_216_5_needs_a_negative_median_c_G():
    target = _series(6, 753, [-3.0] * 24, [1.0] * 24, c_g=[0.5] * 24)
    nulls = [_series(6, 100 + i, [1.0 + i] * 24, [1.0] * 24) for i in range(200)]
    got = sc.prediction_216_5_unit(target, nulls)
    assert got['median_c_G_negative'] is False
    assert got['pass'] is False


def test_216_5_no_write_below_twelve_valid_questions():
    writes = [1.0] * 11 + [0.0] * 13
    target = _series(6, 753, [-3.0] * 24, writes)
    assert target['no_write'] is True
    got = sc.prediction_216_5_unit(target, [])
    assert got['status'] == 'no write'
    assert got['pass'] is False

    twelve = _series(6, 753, [-3.0] * 24, [1.0] * 12 + [0.0] * 12)
    assert twelve['no_write'] is False
    assert twelve['n_valid'] == 12


def test_216_6_strictly_above_every_null():
    target = _series(11, 8868, [3.0] * 24, [1.0] * 24, d_a=[1.0] * 24)
    nulls = [_series(11, 100 + i, [-10.0 - i] * 24, [1.0] * 24, d_a=[1.0] * 24) for i in range(200)]
    got = sc.prediction_216_6_unit(target, nulls)
    assert got['strictly_above_all_nulls'] is True
    assert got['pass'] is True


def test_216_6_a_tie_with_any_null_does_not_count():
    target = _series(11, 8868, [3.0] * 24, [1.0] * 24, d_a=[1.0] * 24)
    nulls = [_series(11, 100 + i, [-10.0 - i] * 24, [1.0] * 24, d_a=[1.0] * 24) for i in range(200)]
    nulls[199] = _series(11, 299, [3.0] * 24, [1.0] * 24, d_a=[1.0] * 24)
    got = sc.prediction_216_6_unit(target, nulls)
    assert got['n_ties'] == 1
    assert got['pass'] is False


def test_216_6_needs_a_positive_median_d_A():
    target = _series(11, 8868, [3.0] * 24, [1.0] * 24, d_a=[-1.0] * 24)
    nulls = [_series(11, 100 + i, [-10.0 - i] * 24, [1.0] * 24, d_a=[1.0] * 24) for i in range(200)]
    assert sc.prediction_216_6_unit(target, nulls)['pass'] is False


def test_d_A_is_zero_for_a_zero_write():
    c = np.array([0.0, 0.0])
    assert np.allclose(sc.d_alibaba(c, c, 0.5, -0.5), 0.0)


def test_null_units_exclude_targets_and_zero_writes_and_take_200():
    n_units = 9010
    mw = np.full(n_units, 5.0)
    mw[0:5] = 0.0                      # zero-write units are excluded
    target_units = {753, 1169, 8868, 776}
    mw[753] = 4.0
    got = sc.null_units(mw, target_units, 753, n_null=200)
    assert got['n'] == 200
    assert len(got['units']) == 200
    assert not (set(got['units']) & target_units)
    assert not any(mw[u] == 0.0 for u in got['units'])
    assert noduplicates(got['units'])


def noduplicates(values) -> bool:
    return len(values) == len(set(values))


def test_null_units_are_the_nearest_by_median_write_size():
    mw = np.array([1.0, 1.1, 1.2, 9.0, 9.1, 0.0, 1.3, 1.4])
    got = sc.null_units(mw, {0}, 0, n_null=3)
    assert got['units'] == [1, 2, 6]   # nearest to 1.0, ties by index, 0.0 excluded


# ----------------------------------------------------------------------------------------------------------------
# the tiny model: the real runner path, end to end
# ----------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session')
def tiny():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen3_5ForCausalLM'):
        pytest.skip('transformers without Qwen3.5')
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=16, intermediate_size=9000, num_hidden_layers=30,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, full_attention_interval=4, max_position_embeddings=512)
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.config._attn_implementation = 'eager'
    torch.set_grad_enabled(False)
    return model, torch


@pytest.fixture(scope='session')
def tiny_rig(tiny):
    import neuron_report as nr
    model, torch = tiny
    layers = nr.find_text_layers(model)[1]
    norm = nr.find_text_layers(model)[2]
    out_emb = model.get_output_embeddings()
    tables = run216.prepare_tables(model, layers, norm)
    return {'model': model, 'layers': layers, 'norm': norm, 'out_emb': out_emb, 'tables': tables}


@pytest.fixture(scope='session')
def bank():
    return run216.load_bank(PROMPTS)


def test_tiny_forward_records_every_boundary_and_telescopes(tiny_rig, bank):
    rig = tiny_rig
    entry = bank['items'][0]
    record, arrays = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'],
        capture_units=True)
    report = record['boundary_report']
    assert report['missing'] == [] and report['duplicated'] == []
    assert report['n_present'] == 2 * len(rig['layers'])
    assert report['chain_max_abs_delta'] == 0.0
    assert len(record['dla']['terms']) == 2 * len(rig['layers'])
    check = sc.gate1_check(record['dla']['emb_term'], record['dla']['terms'],
                           record['dla']['fp32_logits'])
    assert check['pass'] is True
    assert check['max_abs_error'] < 1e-9
    assert len(record['lens']) == len(rig['layers']) + 1
    assert record['lens'][-1]['rank_G'] >= 0


def test_tiny_lens_at_h32_matches_the_models_own_logits(tiny_rig, bank):
    """The FP32 frozen-norm lens at h_32 must agree with the model's native bf16 logits closely."""
    rig = tiny_rig
    entry = bank['items'][0]
    record, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'])
    got = [record['lens'][-1][k] for k in ('z_G', 'z_Al', 'z_T')]
    native = record['native_logits']
    for a, b in zip(got, native):
        assert abs(a - b) < 0.05 * max(1.0, abs(b))


def test_tiny_zero_arm_changes_nothing_before_h_7(tiny_rig, bank):
    """The first zeroed unit is in block 6, so the first residual it can change is h_7."""
    rig = tiny_rig
    receipt = run216.load_receipt(STAGE1)
    entry = bank['items'][0]
    base, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'])
    units, factor = run216.edit_for_arm(receipt, 'zero', len(entry['prompt_token_ids']))
    edited, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'],
        edit_units=units, edit_factor=factor)
    base_m = [e['m'] for e in base['lens']]
    edit_m = [e['m'] for e in edited['lens']]
    for l in range(0, 7):
        assert edit_m[l] == base_m[l]
    assert any(edit_m[l] != base_m[l] for l in range(7, len(base_m)))


def test_tiny_control_and_factual_rows_run(tiny_rig, bank):
    rig = tiny_rig
    control = bank['controls'][0]
    record, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        control['sequence_token_ids'], control['read_position'], control['prompt_token_ids'])
    assert len(record['lens']) == len(rig['layers']) + 1
    factual = bank['factuals'][0]
    record, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        factual['prompt_token_ids'], factual['read_position'], factual['prompt_token_ids'])
    assert record['read_position'] == len(factual['prompt_token_ids']) - 1


def test_tiny_run_mode_writes_the_run_files(tmp_path, tiny, monkeypatch):
    """`run` with --limit through the REAL code path, with only the loader stubbed."""
    model, torch = tiny
    monkeypatch.setattr(run216.run214, 'load_rig',
                        lambda args: (model, None, {'attn_implementation': 'eager',
                                                    'dtype': 'bfloat16'}))
    monkeypatch.setattr(run216.run214, 'kernel_check',
                        lambda load: {'pass': True, 'checks': {'test_seam': True}})
    args = argparse.Namespace(out=str(tmp_path / 'run'), device='cpu', prompts=str(PROMPTS),
                              stage1=str(STAGE1), limit=3, no_resume=True,
                              require_pinned_shape=False)
    result = run216.run(args)
    assert result['stage'] == 'DONE'
    assert result['n_rows'] == 3
    rows, arrays, run = sc.load_run(tmp_path / 'run')
    assert len(rows) == 3
    assert run['n_blocks'] == len(run216.nr.find_text_layers(model)[1])
    for row in rows.values():
        assert row['boundary_report']['missing'] == []
        assert sc.gate1_check(row['dla']['emb_term'], row['dla']['terms'],
                              row['dla']['fp32_logits'])['pass'] is True
    assert (tmp_path / 'run' / 'arrays.npz').exists()


def test_tiny_run_stops_when_a_row_does_not_telescope(tmp_path, tiny, monkeypatch):
    """A G1 failure must stop the run: no further forwards, `STOP` in run.json."""
    model, torch = tiny
    monkeypatch.setattr(run216.run214, 'load_rig',
                        lambda args: (model, None, {'attn_implementation': 'eager',
                                                    'dtype': 'bfloat16'}))
    monkeypatch.setattr(run216.run214, 'kernel_check',
                        lambda load: {'pass': True, 'checks': {'test_seam': True}})
    real = run216.dla_block

    def broken(*a, **k):
        out = real(*a, **k)
        out['fp32_logits'] = [x + 1.0 for x in out['fp32_logits']]
        return out

    monkeypatch.setattr(run216, 'dla_block', broken)
    args = argparse.Namespace(out=str(tmp_path / 'run'), device='cpu', prompts=str(PROMPTS),
                              stage1=str(STAGE1), limit=5, no_resume=True,
                              require_pinned_shape=False)
    result = run216.run(args)
    assert result['stage'] == 'STOP'
    assert result['n_rows'] == 1
    assert result['stops'][0]['reason'] == 'G1 telescoping'


# ----------------------------------------------------------------------------------------------------------------
# the scorer on a synthetic full run
# ----------------------------------------------------------------------------------------------------------------

def _write_synthetic_run(tmp_path: Path, *, outcome: str = 'mixed') -> Path:
    """A complete run directory the scorer can read: every arm, both control sides, the factuals."""
    out = tmp_path / 'synth'
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'arms': {arm: {'g_ali': [1.0] * 25} for arm in sc.G0_ARMS}}
    receipt_path = out / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')

    rows = []
    g0_arms = list(sc.G0_ARMS)
    other_arms = ['op_filler', 'op_denial', 'op_vogel']
    for arm in g0_arms + other_arms:
        for qi in range(sc.N_QUESTIONS):
            ms = [0.0] * sc.N_LAYERS
            ms[-1] = -1.0 if qi % 2 == 0 else 1.0
            row = make_row(f'{arm}:{qi}', ms)
            row['native_g_ali'] = 1.0
            rows.append(row)
    # `unchanged` gets a clean shape: 12 early leads at layers 10/11, ends negative.
    for qi in range(sc.N_QUESTIONS):
        ms = [-1.0] * sc.N_LAYERS
        if qi < 12:
            ms[10] = ms[11] = 1.0
        ms[-1] = -1.0
        rows = [r for r in rows if r['key'] != f'unchanged:{qi}']
        rows.append(make_row(f'unchanged:{qi}', ms, native_g_ali=1.0))
    rows = [r for r in rows if r['key'] != 'zero:0']
    zero_ms = [0.0] * sc.N_LAYERS
    for l in range(10, sc.N_LAYERS):
        zero_ms[l] = 1.0
    rows.append(make_row('zero:0', zero_ms, native_g_ali=1.0))
    for arm in sc.G0_ARMS:
        if arm in ('unchanged', 'zero'):
            continue
        for qi in range(sc.N_QUESTIONS):
            pass
    controls = control_side('control_google', [10, 11, 12])
    controls.update(control_side('control_alibaba', [10, 11, 12]))
    rows += list(controls.values())
    for qi in range(sc.N_QUESTIONS):
        rows.append(make_row(f'factual:{qi}', [0.0] * sc.N_LAYERS, kind='factual', argmax=7,
                             input_token_id=11))
    with (out / 'forwards.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')

    n_unit_layers = len(run216.UNIT_LAYERS)
    n_units = 9000
    rng = np.random.default_rng(0)
    units = {}
    for qi in range(sc.N_QUESTIONS):
        block = rng.normal(size=(n_unit_layers, n_units)).astype(np.float32)
        block[0, 753] = -25.0            # a strong negative target on layer 6
        units[f'units:unchanged:{qi}'] = block
    np.savez_compressed(
        out / 'arrays.npz',
        unit_layers=np.array(run216.UNIT_LAYERS),
        unit_proj=rng.normal(size=(n_unit_layers, n_units, 3)).astype(np.float32),
        unit_colnorm=np.ones((n_unit_layers, n_units), dtype=np.float32),
        scales=np.ones(sc.N_QUESTIONS, dtype=np.float64),
        **units)
    run = {'model': sc.MODEL, 'revision': sc.REVISION, 'design_sha256': 'x', 'prompts_sha256': 'y',
           'tool_sha256': 'z', 'stage1_receipt_path': str(receipt_path),
           'stage1_receipt_sha256': 'r', 'n_blocks': sc.N_BLOCKS, 'hidden_size': 16,
           'arms': g0_arms + other_arms, 'control_arms': ['control_google', 'control_alibaba']}
    (out / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    return out


def test_scorer_reads_a_whole_synthetic_run(tmp_path):
    out = _write_synthetic_run(tmp_path)
    score = sc.score_run(out)
    assert score['gates']['G0']['pass'] is True
    assert score['gates']['G1']['pass'] is True
    assert score['gates']['G2']['pass'] is True
    assert score['gates']['G2']['window'] == [10, 11, 12]
    assert score['gates']['G2']['n_usable_pairs'] == 2
    assert score['unchanged_outcome']['n_early_lead'] == 12
    assert score['unchanged_outcome']['outcome'] == sc.OUTCOME_LEAD
    assert '216.1' in score['predictions']
    sc.write_json(out / 'score216.json', score)
    (out / 'SCORECARD.md').write_text(sc.scorecard_markdown(score), encoding='utf-8')
    assert (out / 'score216.json').stat().st_size > 1000
    assert 'Part 216 scorecard' in (out / 'SCORECARD.md').read_text(encoding='utf-8')


def test_scorecard_marks_a_failed_gate():
    score = {'run_pins': {}, 'gates': {'G0': {'pass': False, 'n_values': 0, 'n_expected': 312,
                                              'max_abs_delta': 1.0, 'tolerance': 0.02,
                                              'exact_agreement': 0},
                                       'G1': {'pass': True, 'n_rows': 1, 'max_abs_error': 0.0,
                                              'tolerance': 1e-3},
                                       'G2': {'pass': False, 'n_usable_pairs': 0, 'window': []}},
             'unchanged_outcome': {'outcome': sc.OUTCOME_NOT_READ},
             'arms': {}, 'predictions': {}, 'descriptive': {'factual': {'per_layer': []}}}
    text = sc.scorecard_markdown(score)
    assert '| G0 | FAIL |' in text
    assert 'not read' in text


def test_score_tool_does_not_import_torch():
    """The scorer is pure numpy/json: importing it must not pull torch into sys.modules."""
    import subprocess
    code = ('import sys; sys.path.insert(0, %r); import score216; '
            'print("torch" in sys.modules)' % str(HERE))
    out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'
