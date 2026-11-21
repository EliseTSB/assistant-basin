"""Part 216 tests: the gates, the outcome rules, the unit thresholds, and the real runner path.

Run with the venv that has torch:
    /home/elisepc/neuron-probe/.venv/bin/python -I -m pytest part216/test_part216.py

The model tests build a tiny random Qwen3.5-architecture model (the `_tiny_model` pattern from
`part215/test_part215.py`) so `run216.forward_row` -- the real capture, lens and DLA code -- runs end
to end on CPU. Nothing here loads the 4B checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
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
import resolve216 as res  # noqa: E402  (torch-free: the plan and the prompts-bank lookup)
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


def test_g0_rejects_nonfinite_values():
    """A NaN row must FAIL G0 explicitly: NaN > tolerance is False, so it used to pass (Alden #10 #3)."""
    receipt = _receipt_for(sc.G0_ARMS, {arm: [1.0] * 25 for arm in sc.G0_ARMS})
    rows = {f'{arm}:{qi}': make_row(f'{arm}:{qi}', [0.0] * sc.N_LAYERS, native_g_ali=1.0)
            for arm in sc.G0_ARMS for qi in range(sc.N_QUESTIONS)}
    rows['zero:3'] = make_row('zero:3', [0.0] * sc.N_LAYERS, native_g_ali=float('nan'))
    gate = sc.gate0(rows, receipt)
    assert gate['pass'] is False
    assert any(f.get('reason') == 'nonfinite g_ali value' for f in gate['failures'])
    rows['zero:3'] = make_row('zero:3', [0.0] * sc.N_LAYERS, native_g_ali=float('inf'))
    gate = sc.gate0(rows, receipt)
    assert gate['pass'] is False
    # And the generic checker the runner calls before appending a row names the NaN.
    bad = make_row('zero:4', [0.0] * sc.N_LAYERS)
    bad['lens'][3]['z_G'] = float('nan')
    assert sc.row_finite(bad)['pass'] is False
    assert sc.row_finite(bad)['bad'] == ['lens[3].z_G']


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


def _nulls_216_5(n: int) -> list[dict]:
    return [_series(6, 100 + i, [1.0 + i] * 24, [1.0] * 24) for i in range(n)]


def _nulls_216_6(n: int) -> list[dict]:
    return [_series(11, 100 + i, [-10.0 - i] * 24, [1.0] * 24, d_a=[1.0] * 24)
            for i in range(n)]


def test_216_5_requires_exactly_two_hundred_valid_null_comparisons():
    """1 or 199 surviving neighbours is `not read`, never a pass (Alden #10 #2)."""
    target = _series(6, 753, [-3.0] * 24, [1.0] * 24)
    for n in (0, 1, 2, 199):
        got = sc.prediction_216_5_unit(target, _nulls_216_5(n))
        assert got['status'] == 'not read', n
        assert got['pass'] is False, n
        assert got['n_null_compared'] == n
    ok = sc.prediction_216_5_unit(target, _nulls_216_5(200))
    assert ok['status'] == 'read'
    assert ok['pass'] is True
    assert ok['n_null_compared'] == 200
    # One unreadable null among 200 leaves 199 comparisons, which is still `not read`.
    partial = _nulls_216_5(200)
    partial[0]['median_c_G_per_write'] = None
    got = sc.prediction_216_5_unit(target, partial)
    assert got['status'] == 'not read'
    assert got['n_null_compared'] == 199


def test_216_6_requires_exactly_two_hundred_valid_null_comparisons():
    target = _series(11, 8868, [3.0] * 24, [1.0] * 24, d_a=[1.0] * 24)
    for n in (1, 199):
        got = sc.prediction_216_6_unit(target, _nulls_216_6(n))
        assert got['status'] == 'not read', n
        assert got['pass'] is False, n
        assert got['n_null_compared'] == n
    ok = sc.prediction_216_6_unit(target, _nulls_216_6(200))
    assert ok['status'] == 'read' and ok['pass'] is True
    assert ok['n_null_compared'] == 200


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

def _build_tiny(torch, dtype):
    """The random Qwen3.5-architecture tiny model, in `dtype` (the fixture uses bf16)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=16, intermediate_size=9000, num_hidden_layers=30,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, full_attention_interval=4, max_position_embeddings=512)
    model = Qwen3_5ForCausalLM(cfg).to(dtype).eval()
    model.config._attn_implementation = 'eager'
    torch.set_grad_enabled(False)
    return model


@pytest.fixture(scope='session')
def tiny():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen3_5ForCausalLM'):
        pytest.skip('transformers without Qwen3.5')
    # bf16, so the tiny fixture exercises the REAL capture dtype path (Alden #10 #4). A float32
    # tiny model is built separately by the dtype-refusal test, which is the only test that needs it.
    return _build_tiny(torch, torch.bfloat16), torch


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


def _stub_loader(monkeypatch, model, dtype='torch.bfloat16'):
    monkeypatch.setattr(run216.run214, 'load_rig',
                        lambda args: (model, None, {'attn_implementation': 'eager',
                                                    'dtype': dtype}))
    monkeypatch.setattr(run216.run214, 'kernel_check',
                        lambda load: {'pass': True, 'checks': {'test_seam': True}})


def _patch_plan(monkeypatch, wanted):
    """Restrict the plan to the given `(arm, question_index)` rows (the run path is unchanged)."""
    real = run216.plan_rows
    wanted = {(str(a), int(q)) for a, q in wanted}

    def plan(bank_):
        return [r for r in real(bank_) if (r['arm'], int(r['question_index'])) in wanted]

    monkeypatch.setattr(run216, 'plan_rows', plan)


def _args(out, **overrides):
    base = dict(out=str(out), device='cpu', prompts=str(PROMPTS), stage1=str(STAGE1),
                limit=0, limit_questions=0, require_pinned_shape=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _forwards_text(out) -> str:
    """`forwards.jsonl` or '' -- the runner does not create the file until a row is committed."""
    path = Path(out) / 'forwards.jsonl'
    return path.read_text(encoding='utf-8') if path.exists() else ''


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


# ----------------------------------------------------------------------------------------------------------------
# Alden #10 #1: the opener arms must forward the opener sequences, not the plain items
# ----------------------------------------------------------------------------------------------------------------

def test_opener_arms_resolve_to_their_own_bank_entries(bank):
    """All 72 opener keys resolve to their own opener row; none resolves to a plain item."""
    idx = {(o['arm'], int(o['question_index'])): o for o in bank['openers']}
    assert len(idx) == 72
    for arm in ('op_filler', 'op_denial', 'op_vogel'):
        for qi in range(sc.N_QUESTIONS):
            row = {'kind': 'stage1', 'arm': arm, 'question_index': qi}
            got = run216.resolve_row(bank, row)
            assert got is idx[(arm, qi)]
            assert got['sequence_token_ids'] == idx[(arm, qi)]['sequence_token_ids']
            assert got['sequence_token_ids'] != bank['items'][qi]['sequence_token_ids']
            assert got['read_position'] == len(got['prompt_token_ids']) + run216.OPENING_TOKENS - 1
    # The plain arm still resolves to the plain item.
    assert run216.resolve_row(bank, {'kind': 'stage1', 'arm': 'unchanged',
                                     'question_index': 0}) is bank['items'][0]


def test_run_forwards_plain_and_all_three_opener_variants(tmp_path, tiny, bank, monkeypatch):
    """The run path passes each opener's tokens and read position into `forward_row`.

    Plain `unchanged` and all three opener arms run in the same fixture, so a resolution that
    silently maps every op_* arm onto `bank['items'][q]` is caught by the row it wrote.
    """
    model, torch = tiny
    _stub_loader(monkeypatch, model)
    wanted = [('unchanged', 0), ('op_filler', 0), ('op_denial', 0), ('op_vogel', 0)]
    _patch_plan(monkeypatch, wanted)
    out = tmp_path / 'run'
    result = run216.run(_args(out, limit=4))
    assert result['stage'] == 'DONE'
    assert result['n_rows'] == 4
    rows, arrays, run = sc.load_run(out)
    plain = rows['unchanged:0']['sequence_token_ids']
    openers = {(o['arm'], int(o['question_index'])): o for o in bank['openers']}
    for arm, qi in wanted:
        # Read the expectation out of the bank directly, not through the resolver under test.
        entry = bank['items'][qi] if arm == 'unchanged' else openers[(arm, qi)]
        rec = rows[f'{arm}:{qi}']
        assert rec['sequence_token_ids'] == entry['sequence_token_ids']
        assert rec['prompt_token_ids'] == entry['prompt_token_ids']
        assert rec['read_position'] == entry['read_position']
    for arm, qi in wanted:
        entry = bank['items'][qi] if arm == 'unchanged' else openers[(arm, qi)]
        assert rows[f'{arm}:{qi}']['input_sha256'] == hashlib.sha256(
            json.dumps(entry['sequence_token_ids']).encode('utf-8')).hexdigest()
    for arm in ('op_filler', 'op_denial', 'op_vogel'):
        assert rows[f'{arm}:0']['sequence_token_ids'] != plain
        assert rows[f'{arm}:0']['read_position'] != rows['unchanged:0']['read_position']


def test_control_rows_are_forwarded_without_an_edit(tmp_path, tiny, bank, monkeypatch):
    """A control row is an unedited forward whose sentence lives in the prompt.

    This was a separate production-path crash: `run` sent `control_google` through
    `edit_for_arm`, which raised `ValueError: unknown arm` before the factuals could run.
    """
    model, torch = tiny
    _stub_loader(monkeypatch, model)
    wanted = [('control_google', 0), ('control_alibaba', 0)]
    _patch_plan(monkeypatch, wanted)
    out = tmp_path / 'run'
    result = run216.run(_args(out, limit=2))
    assert result['stage'] == 'DONE'
    assert result['n_rows'] == 2
    rows, arrays, run = sc.load_run(out)
    for arm, qi in wanted:
        entry = bank['controls'][bank['control_arms'].index(arm) * sc.N_QUESTIONS + qi]
        rec = rows[f'{arm}:{qi}']
        assert rec['sequence_token_ids'] == entry['sequence_token_ids']
        assert rec['read_position'] == entry['read_position']


# ----------------------------------------------------------------------------------------------------------------
# Alden #10 #4: the dtype record and the bf16 capture contract
# ----------------------------------------------------------------------------------------------------------------

def test_tiny_capture_records_every_boundary_dtype(tiny_rig, bank):
    rig = tiny_rig
    entry = bank['items'][0]
    record, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'])
    n = len(rig['layers'])
    required = ['h0'] + [f'L{li}:{kind}' for li in range(n) for kind in ('in', 'add1', 'add2')]
    for name in required:
        assert record['dtypes'][name] == 'torch.bfloat16'
    assert record['dtypes'][f'raw:L0:mixer'] == 'torch.bfloat16'
    assert record['dtypes'][f'raw:L0:mlp'] == 'torch.bfloat16'
    accepted = run216.dtype_contract(record, {'dtype': 'torch.bfloat16'})
    assert accepted['pass'] is True
    assert accepted['observed'] == ['torch.bfloat16']
    # A load that does not claim bf16 does not enforce a capture dtype (the float32 tiny case).
    assert run216.dtype_contract(record, {'dtype': 'float32'})['pass'] is True


def test_bf16_run_refuses_float32_captures(tmp_path, tiny, monkeypatch):
    """A run whose load says bf16 must STOP before appending a float32 capture (Alden #10 #4)."""
    model_bf16, torch = tiny
    model_f32 = _build_tiny(torch, torch.float32)
    _stub_loader(monkeypatch, model_f32, dtype='torch.bfloat16')
    _patch_plan(monkeypatch, [('unchanged', 0)])
    out = tmp_path / 'run'
    result = run216.run(_args(out, limit=1))
    assert result['stage'] == 'STOP'
    assert result['stops'][0]['reason'] == 'dtype contract'
    assert result['stops'][0]['detail']['expected'] == 'torch.bfloat16'
    assert result['n_rows'] == 0
    assert result['stops'][0]['row']['dtypes']['L0:in'] == 'torch.float32'
    assert _forwards_text(out).strip() == ''


def test_no_float64_on_the_device_in_the_lens():
    """The lens must not create a float64 tensor on the target device (Arc A-series has no FP64)."""
    src = inspect.getsource(run216.lens_rows)
    assert 'torch.float64' not in src
    assert 'torch.double' not in src


def test_lens_rank_of_the_argmax_token_is_zero(tiny_rig, bank):
    """A token compares equal to itself, so the argmax token's rank is 0 at every layer."""
    rig = tiny_rig
    entry = bank['items'][0]
    record, _ = run216.forward_row(
        rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
        entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'])
    ids = {'rank_G': run216.GOOGLE_ID, 'rank_Al': run216.AL_ID, 'rank_T': run216.T_ID}
    for lens in record['lens']:
        assert lens['rank_G'] >= 0 and lens['rank_Al'] >= 0 and lens['rank_T'] >= 0
        for field, token_id in ids.items():
            if lens['argmax'] == token_id:
                assert lens[field] == 0


# ----------------------------------------------------------------------------------------------------------------
# Alden #10 #2: the runner -> scorer handoff
# ----------------------------------------------------------------------------------------------------------------

def test_runner_persists_unit_tables_and_q_keyed_scales(tmp_path, tiny_rig, monkeypatch):
    rig = tiny_rig
    _stub_loader(monkeypatch, rig['model'])
    _patch_plan(monkeypatch, [('unchanged', 0)])
    out = tmp_path / 'run'
    result = run216.run(_args(out, limit=1))
    assert result['stage'] == 'DONE'
    rows, arrays, run = sc.load_run(out)
    assert 'unit_layers' in arrays
    assert [int(l) for l in arrays['unit_layers']] == run216.UNIT_LAYERS
    n_units = int(rig['model'].config.intermediate_size)
    assert arrays['unit_proj'].shape == (len(run216.UNIT_LAYERS), n_units, 3)
    assert arrays['unit_colnorm'].shape == (len(run216.UNIT_LAYERS), n_units)
    assert 'scale:unchanged:0' in arrays
    assert float(arrays['scale:unchanged:0']) > 0.0
    pack = sc.unit_arrays(rows, arrays)
    assert pack['scales'][0] == pytest.approx(float(arrays['scale:unchanged:0']))
    assert pack['units']['0'].shape[0] == len(run216.UNIT_LAYERS)
    # The scorer runs on that directory without a hand-written tables file.
    score = sc.score_run(out)
    assert 'unit analysis unavailable' not in str(score['units'].get('error'))
    assert score['units']['n_questions'] == 1


# ----------------------------------------------------------------------------------------------------------------
# Alden #10 #3 / Alden #12 #2: STOP before append; fresh directories only, no resume
# ----------------------------------------------------------------------------------------------------------------

def test_stop_record_is_terminal_for_the_output_directory(tmp_path, tiny, monkeypatch):
    """A STOPped directory is refused, not resumed: the STOP record is terminal (Alden #10 #3)."""
    model, torch = tiny
    _stub_loader(monkeypatch, model)
    _patch_plan(monkeypatch, [('unchanged', 0), ('unchanged', 1)])
    out = tmp_path / 'run'
    real = run216.dla_block

    def broken(*a, **k):
        data = real(*a, **k)
        data['fp32_logits'] = [x + 1.0 for x in data['fp32_logits']]
        return data

    monkeypatch.setattr(run216, 'dla_block', broken)
    first = run216.run(_args(out, limit=2))
    assert first['stage'] == 'STOP'
    monkeypatch.setattr(run216, 'dla_block', real)
    before = _forwards_text(out)
    second = run216.run(_args(out, limit=2))
    assert second['stage'] == 'STOP'
    assert second.get('refused') is True
    assert 'no resume' in second['refusal_reason']
    assert _forwards_text(out) == before


def test_run_refuses_a_directory_holding_any_run_file(tmp_path, monkeypatch):
    """Every run file names a prior attempt; a fresh directory is the only accepted --out."""
    names = list(run216.RUN_FILE_NAMES)
    # The three files Alden's bullet names are all covered (`run.json` is where a STOP record lives).
    assert {'forwards.jsonl', 'run.json', 'arrays.npz'} <= set(names)

    def bomb(*_a, **_k):
        raise AssertionError('load_rig must not be called for a non-empty directory')

    monkeypatch.setattr(run216.run214, 'load_rig', bomb)
    for name in names:
        out = tmp_path / f'run-{name}'
        out.mkdir()
        body = '{"key": "unchanged:0"}\n' if name == 'forwards.jsonl' else '{}'
        (out / name).write_text(body, encoding='utf-8')
        result = run216.run(_args(out))
        assert result['stage'] == 'STOP', name
        assert result.get('refused') is True, name
        assert name in result['existing_files'], name
        assert 'no resume' in result['refusal_reason'], name


def test_tampered_dtype_directory_is_refused_not_resumed(tmp_path, tiny, monkeypatch):
    """Alden #12 #2: the directory that resumed past a tampered `L0:add1` dtype is now refused.

    The fresh path's row validator rejects an FP32 boundary on a bf16 load; the old resume path
    skipped `dtype_contract`, appended another row and returned DONE. With no resume at all, the
    tampered directory cannot be continued: nothing is appended, nothing is blessed, no DONE.
    """
    model, torch = tiny
    _stub_loader(monkeypatch, model)
    _patch_plan(monkeypatch, [('unchanged', 0), ('unchanged', 1)])
    out = tmp_path / 'run'
    first = run216.run(_args(out, limit=1))
    assert first['stage'] == 'DONE'
    lines = (out / 'forwards.jsonl').read_text(encoding='utf-8').splitlines()
    records = [json.loads(line) for line in lines]
    assert records[0]['dtypes']['L0:add1'] == 'torch.bfloat16'
    records[0]['dtypes']['L0:add1'] = 'torch.float32'
    tampered = '\n'.join(json.dumps(r) for r in records) + '\n'
    (out / 'forwards.jsonl').write_text(tampered, encoding='utf-8')

    second = run216.run(_args(out, limit=2))
    assert second['stage'] == 'STOP'
    assert second.get('refused') is True
    assert second['n_rows'] == 1
    assert 'forwards.jsonl' in second['existing_files']
    assert (out / 'forwards.jsonl').read_text(encoding='utf-8') == tampered
    # The directory's own record is untouched: the tampered row was never re-validated into DONE.
    assert json.loads((out / 'run.json').read_text(encoding='utf-8'))['stage'] == 'DONE'


def test_no_resume_machinery_remains():
    """Resume is removed, not merely unused: no validator, no pin diff, no --no-resume flag."""
    assert not hasattr(run216, 'validate_committed_rows')
    assert not hasattr(run216, 'pins_mismatch')
    assert not hasattr(run216, 'PIN_KEYS')
    assert run216.RUN_FILE_NAMES
    with pytest.raises(SystemExit) as exc:
        run216.main(['run', '--out', '/tmp/part216-never-created', '--no-resume'])
    assert exc.value.code == 2


def test_anchor_row_outside_the_receipt_is_not_appended(tmp_path, tiny_rig, bank, monkeypatch):
    """An anchor row outside the receipt is refused BEFORE it is appended, not only at the 312th."""
    rig = tiny_rig
    _stub_loader(monkeypatch, rig['model'])
    _patch_plan(monkeypatch, [('unchanged', 0), ('unchanged', 1)])
    values = []
    for qi in (0, 1):
        entry = bank['items'][qi]
        record, _ = run216.forward_row(
            rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
            entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'])
        values.append(float(record['native_g_ali']))
    receipt = {'arms': {arm: {'g_ali': [values[0], values[1] + 1.0] + [0.0] * 23}
                        for arm in sc.G0_ARMS}}
    receipt_path = tmp_path / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    out = tmp_path / 'run'
    result = run216.run(_args(out, stage1=str(receipt_path)))
    assert result['stage'] == 'STOP'
    assert result['stops'][0]['reason'] == 'G0'
    assert result['stops'][0]['key'] == 'unchanged:1'
    assert result['n_rows'] == 1
    assert result['stops'][0]['row']['key'] == 'unchanged:1'
    keys = [json.loads(line)['key'] for line in _forwards_text(out).splitlines()]
    assert keys == ['unchanged:0']


def test_a_committed_g0_row_outside_the_receipt_cannot_be_resumed(tmp_path, tiny_rig, bank,
                                                                  monkeypatch):
    """A committed G0 row outside the receipt is never re-blessed: the directory is refused."""
    rig = tiny_rig
    _stub_loader(monkeypatch, rig['model'])
    _patch_plan(monkeypatch, [('unchanged', 0), ('unchanged', 1), ('unchanged', 2)])
    values = []
    for qi in (0, 1):
        entry = bank['items'][qi]
        record, _ = run216.forward_row(
            rig['model'], rig['layers'], rig['norm'], rig['out_emb'], rig['tables'],
            entry['sequence_token_ids'], entry['read_position'], entry['prompt_token_ids'])
        values.append(float(record['native_g_ali']))
    receipt = {'arms': {arm: {'g_ali': [values[0], values[1]] + [0.0] * 23}
                        for arm in sc.G0_ARMS}}
    receipt_path = tmp_path / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    out = tmp_path / 'run'
    first = run216.run(_args(out, stage1=str(receipt_path), limit=2))
    assert first['stage'] == 'DONE'
    assert first['n_rows'] == 2
    records = [json.loads(line) for line in
               (out / 'forwards.jsonl').read_text(encoding='utf-8').splitlines()]
    records[1]['native_g_ali'] = float(records[1]['native_g_ali']) + 1.0
    tampered = '\n'.join(json.dumps(r) for r in records) + '\n'
    (out / 'forwards.jsonl').write_text(tampered, encoding='utf-8')
    before_rows = len(tampered.splitlines())
    second = run216.run(_args(out, stage1=str(receipt_path), limit=3))
    assert second['stage'] == 'STOP'
    assert second.get('refused') is True
    assert second['n_rows'] == before_rows
    assert (out / 'forwards.jsonl').read_text(encoding='utf-8') == tampered


def test_a_valid_done_directory_is_not_continued(tmp_path, tiny, monkeypatch):
    """Even a clean DONE directory is refused; the supported path is a new, empty --out."""
    model, torch = tiny
    _stub_loader(monkeypatch, model)
    _patch_plan(monkeypatch, [('op_filler', 0), ('op_filler', 1), ('op_filler', 2)])
    first_dir = tmp_path / 'run1'
    first = run216.run(_args(first_dir, limit=2))
    assert first['stage'] == 'DONE'
    before = (first_dir / 'forwards.jsonl').read_text(encoding='utf-8')
    second = run216.run(_args(first_dir, limit=3))
    assert second['stage'] == 'STOP'
    assert second.get('refused') is True
    assert (first_dir / 'forwards.jsonl').read_text(encoding='utf-8') == before
    # The same plan into a fresh directory runs to DONE with exactly its own rows.
    second_dir = tmp_path / 'run2'
    third = run216.run(_args(second_dir, limit=3))
    assert third['stage'] == 'DONE'
    assert third['n_new_rows'] == 3
    keys = [json.loads(line)['key'] for line in
            (second_dir / 'forwards.jsonl').read_text(encoding='utf-8').splitlines()]
    assert keys == ['op_filler:0', 'op_filler:1', 'op_filler:2']


def test_end_to_end_tiny_bf16_run_then_scorer_marks_partial(tmp_path, tiny, monkeypatch):
    """`run216.py run` on the tiny bf16 model, then `score216.py` on that directory.

    The random-zero arms are dropped from this tinier plan: Part 214's recorded random unit pools
    index units up to ~9.3k, and the tiny model's intermediate width is 9000. That is a fixture
    limit, not a runner rule; the real run keeps all 16 arms.
    """
    model, torch = tiny
    _stub_loader(monkeypatch, model)
    real_plan = run216.plan_rows
    monkeypatch.setattr(run216, 'plan_rows',
                        lambda bank_: [r for r in real_plan(bank_)
                                       if not r['arm'].startswith('random_')])
    out = tmp_path / 'e2e'
    result = run216.run(_args(out, limit_questions=1, limit=9))
    assert result['stage'] == 'DONE'
    assert result['limit_questions'] == 1
    assert result['partial'] is True
    assert result['run_status'] == 'partial'
    assert result['n_rows'] == 9
    rows, arrays, run = sc.load_run(out)
    assert run['limit_questions'] == 1 and run['partial'] is True
    assert 'unit_layers' in arrays and 'unit_proj' in arrays and 'unit_colnorm' in arrays
    assert any(name.startswith('scale:unchanged:') for name in arrays)
    proc = subprocess.run([sys.executable, '-I', str(HERE / 'score216.py'), str(out)],
                          capture_output=True, text=True)
    # A 9-row smoke run is not admissible for a full score: exit 3 is `diagnostics only`.
    assert proc.returncode == 3, proc.stderr
    score = json.loads((out / 'score216.json').read_text(encoding='utf-8'))
    assert score['partial'] is True
    assert score['run_status'] == 'partial'
    assert score['score_status'] == 'diagnostics only'
    assert score['admission']['admissible'] is False
    assert score['unchanged_outcome']['outcome'] == sc.OUTCOME_NOT_READ
    assert score['units'].get('error') is None
    card = (out / 'SCORECARD.md').read_text(encoding='utf-8')
    assert 'DIAGNOSTICS ONLY — not a full score' in card


def test_tiny_run_mode_writes_the_run_files(tmp_path, tiny, monkeypatch):
    """`run` with --limit through the REAL code path, with only the loader stubbed."""
    model, torch = tiny
    monkeypatch.setattr(run216.run214, 'load_rig',
                        lambda args: (model, None, {'attn_implementation': 'eager',
                                                    'dtype': 'bfloat16'}))
    monkeypatch.setattr(run216.run214, 'kernel_check',
                        lambda load: {'pass': True, 'checks': {'test_seam': True}})
    args = argparse.Namespace(out=str(tmp_path / 'run'), device='cpu', prompts=str(PROMPTS),
                              stage1=str(STAGE1), limit=3, require_pinned_shape=False)
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
    """A G1 failure must stop the run BEFORE the failing row is appended (Alden #10 #3).

    The row is embedded in the STOP record instead, so the failed attempt is preserved but no
    inadmissible row ever reaches `forwards.jsonl`.
    """
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
                              stage1=str(STAGE1), limit=5, require_pinned_shape=False)
    result = run216.run(args)
    assert result['stage'] == 'STOP'
    assert result['n_rows'] == 0                       # checked before append, so nothing committed
    assert result['stops'][0]['reason'] == 'G1 telescoping'
    assert result['stops'][0]['key'] == 'unchanged:0'
    assert result['stops'][0]['row']['key'] == 'unchanged:0'   # the attempt is preserved
    assert _forwards_text(tmp_path / 'run').strip() == ''


# ----------------------------------------------------------------------------------------------------------------
# descriptive block shares: Al and T stay separate (Alden #10 #2)
# ----------------------------------------------------------------------------------------------------------------

def test_block_share_table_keeps_al_and_t_separate():
    """The three token logits have three separate additive shares; Al and T are never summed."""
    row = make_row('unchanged:0', [0.0] * sc.N_LAYERS)
    terms = [[0.0, 0.0, 0.0] for _ in range(2 * sc.N_BLOCKS)]
    terms[0] = [0.5, 1.0, 2.0]          # block 0 attention
    terms[1] = [0.25, -1.0, -2.0]       # block 0 MLP
    row['dla']['terms'] = terms
    table = sc.block_share_table({'unchanged:0': row}, 'unchanged')
    assert table['tokens'] == ['z_G', 'z_Al', 'z_T']
    b0 = table['blocks'][0]
    assert b0['median_attn_z_G'] == pytest.approx(0.5)
    assert b0['median_attn_z_Al'] == pytest.approx(1.0)
    assert b0['median_attn_z_T'] == pytest.approx(2.0)
    assert b0['median_mlp_z_G'] == pytest.approx(0.25)
    assert b0['median_mlp_z_Al'] == pytest.approx(-1.0)
    assert b0['median_mlp_z_T'] == pytest.approx(-2.0)
    assert 'median_attn_alibaba_family' not in b0
    assert 'median_mlp_alibaba_family' not in b0


# ----------------------------------------------------------------------------------------------------------------
# the scorer on a synthetic full run
# ----------------------------------------------------------------------------------------------------------------

# The design's arm list, written out here so the fixture does not read the constant under test.
DESIGN_ARMS = (('unchanged', 'zero', 'scale_dial_fact')
               + tuple(f'random_{k}:zero' for k in range(10))
               + ('op_filler', 'op_denial', 'op_vogel'))
DESIGN_HIDDEN = 2560                      # DESIGN.md "Model"
SYNTH_UNITS = 9000


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_receipt() -> dict:
    """The self-consistent constant-1 reference Alden #14 describes.

    It is NOT the Part 214 Stage 1 receipt, so the production CLI must refuse a run whose rows were
    built from it; only `score_run(..., reference_receipt=...)` may inject it (tests only).
    """
    return {'arms': {arm: {'g_ali': [1.0] * 25} for arm in DESIGN_ARMS[:13]}}


def _g0_value(receipt: dict, arm: str, qi: int) -> float:
    """The anchor value a fixture row records: the reference's own g_ali where it has one."""
    ref = ((receipt.get('arms') or {}).get(arm) or {}).get('g_ali')
    return float(ref[qi]) if ref is not None and qi < len(ref) else 1.0


def _bank_inputs(bank: dict, arm: str, qi: int) -> dict:
    """The input fields `(arm, q)` must record, read straight out of the frozen bank.

    Deliberately an INDEPENDENT implementation rather than `resolve216.expected_inputs`: the fixture
    must not derive the expected values from the code the tests check.  Same rules as the runner:
    openers through `bank['openers']`, controls through `bank['controls']`, factuals with
    `prompt_ids == seq`.
    """
    if arm == 'factual':
        entry = bank['factuals'][qi]
        seq = [int(t) for t in entry['prompt_token_ids']]
        prompt = list(seq)
    elif arm in bank['control_arms']:
        entry = bank['controls'][bank['control_arms'].index(arm) * 24 + qi]
        seq = [int(t) for t in entry['sequence_token_ids']]
        prompt = [int(t) for t in entry['prompt_token_ids']]
    elif arm in bank['opener_arms']:
        entry = next(e for e in bank['openers']
                     if str(e['arm']) == arm and int(e['question_index']) == qi)
        seq = [int(t) for t in entry['sequence_token_ids']]
        prompt = [int(t) for t in entry['prompt_token_ids']]
    else:
        entry = bank['items'][qi]
        seq = [int(t) for t in entry['sequence_token_ids']]
        prompt = [int(t) for t in entry['prompt_token_ids']]
    return {'sequence_token_ids': seq, 'prompt_token_ids': prompt, 'T': len(prompt),
            'read_position': int(entry['read_position']),
            'input_sha256': hashlib.sha256(json.dumps(seq).encode('utf-8')).hexdigest()}


def _write_synthetic_run(tmp_path: Path, receipt: dict | None = None) -> Path:
    """A complete run directory the scorer reads as a FULL score: every arm, both control sides, the
    factuals, the frozen pins, and every array the runner writes at its pinned shape.

    The pins are the real files': `prompts216.json`, `DESIGN.md`, `run216.py` and the pinned Part 214
    receipt.  `receipt=None` (the default) uses that PINNED receipt and records its own g_ali on the
    anchor rows, so the directory is admissible through the production CLI; passing a dict (see
    `synthetic_receipt()`) is the bad fixture Alden #14 describes.  An admission refusal in a test
    is therefore caused by the mutation that test makes, not by a fixture that was never admissible.
    """
    out = tmp_path / 'synth'
    out.mkdir(parents=True, exist_ok=True)
    if receipt is None:
        receipt = json.loads(STAGE1.read_text(encoding='utf-8'))
    receipt_path = out / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    bank = json.loads(PROMPTS.read_text(encoding='utf-8'))

    rows = []
    for arm in DESIGN_ARMS:
        for qi in range(sc.N_QUESTIONS):
            ms = [0.0] * sc.N_LAYERS
            ms[-1] = -1.0 if qi % 2 == 0 else 1.0
            rows.append(make_row(f'{arm}:{qi}', ms,
                                 native_g_ali=_g0_value(receipt, arm, qi)))
    # `unchanged` gets a clean shape: 12 early leads at layers 10/11, ends negative.
    for qi in range(sc.N_QUESTIONS):
        ms = [-1.0] * sc.N_LAYERS
        if qi < 12:
            ms[10] = ms[11] = 1.0
        ms[-1] = -1.0
        rows = [r for r in rows if r['key'] != f'unchanged:{qi}']
        rows.append(make_row(f'unchanged:{qi}', ms,
                             native_g_ali=_g0_value(receipt, 'unchanged', qi)))
    rows = [r for r in rows if r['key'] != 'zero:0']
    zero_ms = [0.0] * sc.N_LAYERS
    for l in range(10, sc.N_LAYERS):
        zero_ms[l] = 1.0
    rows.append(make_row('zero:0', zero_ms, native_g_ali=_g0_value(receipt, 'zero', 0)))
    controls = control_side('control_google', [10, 11, 12])
    controls.update(control_side('control_alibaba', [10, 11, 12]))
    rows += list(controls.values())
    for qi in range(sc.N_QUESTIONS):
        rows.append(make_row(f'factual:{qi}', [0.0] * sc.N_LAYERS, kind='factual', argmax=7,
                             input_token_id=11))
    # Every row records the exact sequence/prompt/T/read slot/input hash of its pinned (arm, q).
    for row in rows:
        row.update(_bank_inputs(bank, row['arm'], row['question_index']))
    with (out / 'forwards.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row) + '\n')

    rng = np.random.default_rng(0)
    n_slots = len(run216.UNIT_LAYERS)
    arrays = {
        'unit_layers': np.asarray(run216.UNIT_LAYERS, dtype=np.int64),
        'unit_proj': rng.normal(size=(n_slots, SYNTH_UNITS, 3)).astype(np.float32),
        'unit_colnorm': np.ones((n_slots, SYNTH_UNITS), dtype=np.float32),
        'scales': np.ones(sc.N_QUESTIONS, dtype=np.float64),   # legacy positional fallback
    }
    zero_h = np.zeros(DESIGN_HIDDEN, dtype=np.float32)
    for row in rows:
        key = row['key']
        arrays[f'h0:{key}'] = zero_h
        arrays[f'h32:{key}'] = zero_h
        arrays[f'scale:{key}'] = np.float64(1.0)
        if row['kind'] == 'stage1':
            arrays[f'units:{key}'] = np.zeros((n_slots, SYNTH_UNITS), dtype=np.float32)
    for qi in range(sc.N_QUESTIONS):
        block = rng.normal(size=(n_slots, SYNTH_UNITS)).astype(np.float32)
        block[0, 753] = -25.0            # a strong negative target on layer 6
        arrays[f'units:unchanged:{qi}'] = block
    np.savez_compressed(out / 'arrays.npz', **arrays)

    run = {
        'mode': 'run', 'part': 216, 'stage': 'DONE', 'partial': False,
        'model': sc.MODEL, 'revision': sc.REVISION,
        'design_sha256': _sha(HERE / 'DESIGN.md'),
        'prompts_sha256': _sha(HERE / 'prompts216.json'),
        'tool_sha256': _sha(HERE / 'run216.py'),
        'stage1_receipt_path': str(receipt_path),
        'stage1_receipt_sha256': _sha(receipt_path),
        'n_blocks': sc.N_BLOCKS, 'hidden_size': DESIGN_HIDDEN,
        'unit_layers': list(run216.UNIT_LAYERS),
        'units': [f'{l}:{u}' for l, u in run216.UNITS],
        'arms': list(DESIGN_ARMS),
        'control_arms': ['control_google', 'control_alibaba'],
    }
    (out / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    return out


@pytest.fixture(scope='session')
def synth_base(tmp_path_factory):
    """The admissible directory, built once; every test gets its own copy."""
    return _write_synthetic_run(tmp_path_factory.mktemp('synthbase'))


@pytest.fixture
def synth(synth_base, tmp_path):
    out = tmp_path / 'synth'
    shutil.copytree(synth_base, out)
    return out


def _read_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _write_jsonl(path, rows):
    Path(path).write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')


def _rewrite_arrays(out, mutate):
    with np.load(Path(out) / 'arrays.npz', allow_pickle=True) as z:
        arrays = {name: z[name] for name in z.files}
    mutate(arrays)
    np.savez_compressed(Path(out) / 'arrays.npz', **arrays)


def _rewrite_row(out, key, mutate):
    """Mutate exactly one forwards.jsonl row (by key) and write the file back."""
    rows = _read_jsonl(Path(out) / 'forwards.jsonl')
    for row in rows:
        if row['key'] == key:
            mutate(row)
            break
    else:
        raise AssertionError(f'{key} is not in forwards.jsonl')
    _write_jsonl(Path(out) / 'forwards.jsonl', rows)


def _drop_keys(out, keys):
    """Remove `keys` from forwards.jsonl, from arrays.npz and from run.json's arm list."""
    keys = set(keys)
    rows = [r for r in _read_jsonl(Path(out) / 'forwards.jsonl') if r['key'] not in keys]
    _write_jsonl(Path(out) / 'forwards.jsonl', rows)

    def mutate(arrays):
        for name in list(arrays):
            if ':' in name and name.split(':', 1)[1] in keys:
                del arrays[name]
    _rewrite_arrays(out, mutate)
    run = json.loads((Path(out) / 'run.json').read_text(encoding='utf-8'))
    run['arms'] = [a for a in run['arms'] if not any(k.startswith(f'{a}:') for k in keys)]
    (Path(out) / 'run.json').write_text(json.dumps(run), encoding='utf-8')


def _run_score(out):
    proc = subprocess.run([sys.executable, '-I', str(HERE / 'score216.py'), str(out)],
                          capture_output=True, text=True)
    score = json.loads((Path(out) / 'score216.json').read_text(encoding='utf-8'))
    return proc, score


def _assert_diagnostics_only(out):
    """Exit 3, inadmissible, a named reason, and EVERY prediction verdict `not read`."""
    proc, score = _run_score(out)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    admission = score['admission']
    assert admission['admissible'] is False
    assert admission['status'] == 'diagnostics only'
    assert admission['reasons'], 'an inadmissible scorecard must name its reasons'
    assert score['score_status'] == 'diagnostics only'
    assert score['unchanged_outcome']['outcome'] == sc.OUTCOME_NOT_READ
    for key in ('216.1', '216.2', '216.3', '216.4', '216.5', '216.6'):
        assert score['predictions'][key]['verdict'] == sc.OUTCOME_NOT_READ, key
        assert score['predictions'][key]['pass'] is False, key
    assert 'DIAGNOSTICS ONLY' in (Path(out) / 'SCORECARD.md').read_text(encoding='utf-8')
    return score


def test_scorer_reads_a_whole_synthetic_run(tmp_path):
    out = _write_synthetic_run(tmp_path)
    score = sc.score_run(out)
    assert score['admission']['admissible'] is True
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


def test_scorer_marks_a_stopped_run_partial(tmp_path):
    """A STOPped run is never a full score, even if every expected row happens to be present."""
    out = _write_synthetic_run(tmp_path)
    run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    run['stage'] = 'STOP'
    run['stops'] = [{'key': 'unchanged:0', 'reason': 'G1 telescoping'}]
    (out / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    score = sc.score_run(out)
    assert score['partial'] is True
    assert score['score_status'] == 'diagnostics only'
    assert score['admission']['admissible'] is False
    assert score['run_status'] == 'partial'
    assert score['completeness']['stopped'] is True
    assert score['completeness']['n_missing'] == 0
    assert score['unchanged_outcome']['outcome'] == sc.OUTCOME_NOT_READ
    assert 'DIAGNOSTICS ONLY' in sc.scorecard_markdown(score)


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


# ----------------------------------------------------------------------------------------------------------------
# Alden #12 #1: admission -- a full score only from a finalized, pinned, fixed-456-key directory
# ----------------------------------------------------------------------------------------------------------------

def test_admission_positive_fixture_is_scored_and_exits_zero(synth):
    """The positive control the refusals are paired with: admissible, exit 0, and it scores."""
    proc, score = _run_score(synth)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    admission = score['admission']
    assert admission['admissible'] is True
    assert admission['status'] == 'scored'
    assert admission['n_expected_keys'] == 456
    assert admission['n_rows'] == 456
    assert admission['n_duplicate_keys'] == 0
    assert admission['n_reasons'] == 0
    assert score['score_status'] == 'scored'
    assert score['partial'] is False
    assert score['predictions']['216.2']['pass'] is True
    assert all(score['predictions'][k]['verdict'] == 'read' for k in
               ('216.1', '216.2', '216.3', '216.4', '216.5', '216.6'))
    card = (synth / 'SCORECARD.md').read_text(encoding='utf-8')
    assert 'DIAGNOSTICS ONLY' not in card
    assert 'score status: `scored`' in card


def test_admission_rejects_one_g0_row_outside_tolerance(synth):
    """Alden #12 #1 (a): one anchor row outside tolerance used to give `full` + 216.2 PASS."""
    rows = _read_jsonl(synth / 'forwards.jsonl')
    for row in rows:
        if row['key'] == 'zero:3':
            row['native_g_ali'] = float(row['native_g_ali']) + 0.05
    _write_jsonl(synth / 'forwards.jsonl', rows)
    score = _assert_diagnostics_only(synth)
    assert any(r['check'] == 'G0' for r in score['admission']['reasons'])
    assert score['gates']['G0']['pass'] is False


def test_admission_rejects_a_running_stage_with_rows_but_no_arrays(synth):
    """Alden #12 #1 (b): the window before arrays.npz/commit -- all rows, stage RUNNING, no arrays."""
    run = json.loads((synth / 'run.json').read_text(encoding='utf-8'))
    run['stage'] = 'RUNNING'
    (synth / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    (synth / 'arrays.npz').unlink()
    score = _assert_diagnostics_only(synth)
    checks = {r['check'] for r in score['admission']['reasons']}
    assert 'stage' in checks
    assert 'arrays' in checks


def test_admission_rejects_an_arm_removed_from_manifest_and_file(synth):
    """Alden #12 #1 (c): op_vogel and its 24 rows gone from both the manifest and the file (432)."""
    _drop_keys(synth, [f'op_vogel:{qi}' for qi in range(sc.N_QUESTIONS)])
    score = _assert_diagnostics_only(synth)
    admission = score['admission']
    assert admission['n_missing'] == 24
    assert all(k.startswith('op_vogel:') for k in admission['missing'])
    # Completeness is measured against the DESIGN's key set, not the attempt's own manifest.
    assert score['completeness']['n_expected'] == 456
    assert score['completeness']['n_missing'] == 24


def test_admission_rejects_a_nonfinal_stage_with_complete_arrays(synth):
    """The DONE requirement on its own: every row and array present, but the run is not finalized."""
    run = json.loads((synth / 'run.json').read_text(encoding='utf-8'))
    run['stage'] = 'RUNNING'
    (synth / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    score = _assert_diagnostics_only(synth)
    assert {r['check'] for r in score['admission']['reasons']} == {'stage'}


def test_admission_rejects_a_missing_arrays_file(synth):
    """The arrays requirement on its own: stage DONE, all 456 rows, no arrays.npz."""
    (synth / 'arrays.npz').unlink()
    score = _assert_diagnostics_only(synth)
    assert {r['check'] for r in score['admission']['reasons']} == {'arrays'}


def test_admission_rejects_a_duplicate_key(synth):
    lines = (synth / 'forwards.jsonl').read_text(encoding='utf-8').splitlines()
    (synth / 'forwards.jsonl').write_text('\n'.join(lines + [lines[0]]) + '\n',
                                          encoding='utf-8')
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'keys')
    assert reason['n_duplicates'] == 1
    assert score['admission']['n_rows'] == 456


def test_admission_rejects_an_extra_key(synth):
    rows = _read_jsonl(synth / 'forwards.jsonl')
    extra = dict(rows[0])
    extra['key'] = 'op_extra:0'
    _write_jsonl(synth / 'forwards.jsonl', rows + [extra])
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'keys')
    assert reason['n_extra'] == 1
    assert reason['sample'] == ['op_extra:0']


def test_admission_rejects_a_misaligned_array(synth):
    """A `units:<key>` array that is not the pinned (6, n_units) shape is a misaligned record."""
    def mutate(arrays):
        arrays['units:unchanged:0'] = np.zeros((len(run216.UNIT_LAYERS), SYNTH_UNITS - 1),
                                               dtype=np.float32)
    _rewrite_arrays(synth, mutate)
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons']
                  if r['check'] == 'arrays' and r.get('n_wrong'))
    assert reason['n_wrong'] == 1
    assert reason['sample'][0]['name'] == 'units:unchanged:0'
    assert reason['sample'][0]['shape'] == [len(run216.UNIT_LAYERS), SYNTH_UNITS - 1]


def test_admission_rejects_an_array_for_a_missing_row(synth):
    """An array keyed to a row outside the grid is as inadmissible as a missing one."""
    def mutate(arrays):
        arrays['h0:op_extra:0'] = np.zeros(DESIGN_HIDDEN, dtype=np.float32)
    _rewrite_arrays(synth, mutate)
    score = _assert_diagnostics_only(synth)
    assert any(r['check'] == 'arrays' and r.get('n_orphan') for r in score['admission']['reasons'])


def test_admission_rejects_a_pin_mismatch(synth):
    run = json.loads((synth / 'run.json').read_text(encoding='utf-8'))
    run['revision'] = 'deadbeef'
    (synth / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    score = _assert_diagnostics_only(synth)
    assert any(r['check'] == 'pin:revision' for r in score['admission']['reasons'])


def test_admission_rejects_a_failed_g1_row(synth):
    rows = _read_jsonl(synth / 'forwards.jsonl')
    rows[5]['dla']['fp32_logits'][0] += 1.0
    _write_jsonl(synth / 'forwards.jsonl', rows)
    score = _assert_diagnostics_only(synth)
    assert any(r['check'] == 'G1' for r in score['admission']['reasons'])


# ----------------------------------------------------------------------------------------------------------------
# Alden #14: G0 binds to the PINNED Part 214 receipt, rows bind to their (arm, q) inputs, arrays are
# finite and positively scaled, and a failed G2 leaves the lens predictions `not read`
# ----------------------------------------------------------------------------------------------------------------

def test_positive_fixture_is_bound_to_the_pinned_receipt(synth):
    """The positive fixture grades against the trusted receipt's BYTES, not a constant or its own file."""
    score = sc.score_run(synth)
    assert score['admission']['admissible'] is True
    reference = score['admission']['reference']
    assert reference['pinned'] is True and reference['injected'] is False
    assert reference['sha256'] == sc.PART214_STAGE1_SHA256
    assert Path(reference['path']) == STAGE1
    # The run's own copy is a claim, reported but never the reference.
    assert Path(score['admission']['run_recorded_receipt']['path']) != STAGE1
    receipt = json.loads(STAGE1.read_text(encoding='utf-8'))
    values = score['gates']['G0']['values']
    assert len(values) == 312
    for arm in DESIGN_ARMS[:13]:
        assert values[f'{arm}:0']['receipt'] == float(receipt['arms'][arm]['g_ali'][0])
    assert any(v['receipt'] != 1.0 for v in values.values()), 'G0 graded against a constant'


def test_cli_refuses_a_run_whose_fixture_used_a_synthetic_receipt(tmp_path):
    """Alden #14: the supplied constant-1 fixture got CLI 0 / G0 PASS; the production CLI must refuse.

    `score_run(..., reference_receipt=...)` is the only way to score such a directory, and it is a
    test-only keyword: no CLI flag and no run.json field reaches it.
    """
    receipt = synthetic_receipt()
    out = _write_synthetic_run(tmp_path, receipt=receipt)
    proc, score = _run_score(out)
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    admission = score['admission']
    assert admission['admissible'] is False
    assert admission['reference']['pinned'] is True and admission['reference']['injected'] is False
    assert admission['reference']['sha256'] == sc.PART214_STAGE1_SHA256
    assert any(r['check'] == 'G0' for r in admission['reasons'])
    assert score['gates']['G0']['pass'] is False
    assert score['score_status'] == 'diagnostics only'
    for key in ('216.1', '216.2', '216.3', '216.4', '216.5', '216.6'):
        assert score['predictions'][key]['verdict'] == sc.OUTCOME_NOT_READ, key
    # The explicit test-only injection is the one path that may use the synthetic reference.
    injected = sc.score_run(out, reference_receipt=receipt)
    assert injected['admission']['reference']['injected'] is True
    assert injected['admission']['admissible'] is True
    assert injected['gates']['G0']['pass'] is True


def test_pinned_receipt_digest_is_checked_before_use(tmp_path, synth, monkeypatch):
    """Bytes that do not hash to the frozen digest are refused, whatever the run's own copy says."""
    tampered = tmp_path / 'stage1.json'
    tampered.write_bytes(STAGE1.read_bytes() + b'\n')     # still valid JSON, different bytes
    monkeypatch.setattr(sc, 'STAGE1_RECEIPT_PATH', tampered)
    score = sc.score_run(synth)
    checks = [r['check'] for r in score['admission']['reasons']]
    assert 'pin:stage1_receipt_digest' in checks
    assert score['admission']['admissible'] is False
    assert score['admission']['reference']['sha256'] != sc.PART214_STAGE1_SHA256
    # The run's recorded receipt is untouched and still self-consistent -- it is simply not evidence.
    assert score['admission']['run_recorded_receipt']['sha256'] == _sha(synth / 'receipt.json')
    assert score['gates']['G0']['pass'] is False        # an empty reference cannot pass an anchor

    monkeypatch.setattr(sc, 'STAGE1_RECEIPT_PATH', tmp_path / 'missing.json')
    missing = sc.score_run(synth)
    assert any(r['check'] == 'pin:stage1_receipt_digest'
               for r in missing['admission']['reasons'])


def test_input_binding_rejects_a_self_consistent_wrong_sequence(synth):
    """Alden #14: `op_denial:0` recording the plain `unchanged:0` inputs used to score exit 0.

    Every recorded field is made self-consistent -- the input hash matches the substituted sequence
    -- which is exactly the attack the old self-digest comparison could not see.
    """
    plain = next(r for r in _read_jsonl(synth / 'forwards.jsonl') if r['key'] == 'unchanged:0')

    def mutate(row):
        row['sequence_token_ids'] = list(plain['sequence_token_ids'])
        row['prompt_token_ids'] = list(plain['prompt_token_ids'])
        row['T'] = plain['T']
        row['read_position'] = plain['read_position']
        row['input_sha256'] = hashlib.sha256(
            json.dumps(row['sequence_token_ids']).encode('utf-8')).hexdigest()

    _rewrite_row(synth, 'op_denial:0', mutate)
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'input:op_denial:0')
    assert reason['field'] in ('sequence_token_ids', 'read_position')
    assert score['gates']['G0']['pass'] is True         # the anchor is untouched: this is the binding


def test_input_binding_rejects_a_wrong_read_position(synth):
    """A row with the right tokens but the wrong read slot measured a different position."""
    _rewrite_row(synth, 'unchanged:3', lambda r: r.update(read_position=r['read_position'] + 1))
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'input:unchanged:3')
    assert reason['field'] == 'read_position'
    assert score['gates']['G0']['pass'] is True


def test_input_binding_rejects_a_hash_that_does_not_match_its_sequence(synth):
    """The recorded input hash is checked against the bank, not merely recomputed from itself."""
    _rewrite_row(synth, 'zero:0', lambda r: r.update(input_sha256='0' * 64))
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'input:zero:0')
    assert reason['field'] == 'input_sha256'


def test_expected_inputs_match_the_runner_and_the_bank(bank):
    """The scorer's rebuilt inputs are the runner's own resolution on the five row kinds."""
    for arm, qi in (('unchanged', 0), ('op_denial', 0), ('control_google', 0),
                    ('control_alibaba', 5), ('factual', 0)):
        want = res.expected_inputs(bank, arm, qi)
        entry = run216.resolve_row(bank, {'kind': want['kind'], 'arm': arm, 'question_index': qi})
        seq = entry['prompt_token_ids'] if arm == 'factual' else entry['sequence_token_ids']
        assert want['sequence_token_ids'] == [int(t) for t in seq]
        assert want['prompt_token_ids'] == [int(t) for t in entry['prompt_token_ids']]
        assert want['T'] == len(entry['prompt_token_ids'])
        assert want['read_position'] == int(entry['read_position'])
        assert want['input_sha256'] == _bank_inputs(bank, arm, qi)['input_sha256']
    # The scorer's binding accepts exactly those fields, and names the row when one differs.
    row = dict(_bank_inputs(bank, 'op_denial', 0))
    row.update({'key': 'op_denial:0', 'arm': 'op_denial', 'question_index': 0})
    assert sc.input_binding({'op_denial:0': row}, bank) == []
    row['read_position'] += 1
    assert [p['check'] for p in sc.input_binding({'op_denial:0': row}, bank)] == \
        ['input:op_denial:0']


def test_admission_rejects_a_nonfinite_array(synth):
    """A NaN anywhere in a required array is not a measurement (Alden #14)."""
    def mutate(arrays):
        h = arrays['h0:zero:5'].copy()
        h[0] = np.nan
        arrays['h0:zero:5'] = h
    _rewrite_arrays(synth, mutate)
    score = _assert_diagnostics_only(synth)
    assert 'array:h0:zero:5' in [r['check'] for r in score['admission']['reasons']]


def test_admission_rejects_a_nan_frozen_scale(synth):
    """Alden #14: a scalar NaN for `scale:unchanged:0` used to be accepted."""
    def mutate(arrays):
        arrays['scale:unchanged:0'] = np.float64(np.nan)
    _rewrite_arrays(synth, mutate)
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'array:scale:unchanged:0')
    assert 'finite' in reason['reason']


def test_admission_rejects_a_nonpositive_frozen_scale(synth):
    """A finite but negative scale is still not a frozen RMS scale (the row agrees, so only the
    positivity rule can refuse it)."""
    def mutate(arrays):
        arrays['scale:unchanged:0'] = np.float64(-1.0)
    _rewrite_arrays(synth, mutate)
    _rewrite_row(synth, 'unchanged:0', lambda r: r['dla'].update(frozen_scale=-1.0))
    score = _assert_diagnostics_only(synth)
    assert 'array:scale:unchanged:0' in [r['check'] for r in score['admission']['reasons']]


def test_admission_rejects_a_scale_that_disagrees_with_its_row(synth):
    """The frozen scale the row records and the array beside it are the same measurement."""
    def mutate(arrays):
        arrays['scale:zero:0'] = np.float64(2.0)      # finite, positive, not the row's 1.0
    _rewrite_arrays(synth, mutate)
    score = _assert_diagnostics_only(synth)
    reason = next(r for r in score['admission']['reasons'] if r['check'] == 'array:scale:zero:0')
    assert reason['recorded'] == 1.0 and reason['array'] == 2.0


def test_g2_failure_leaves_lens_predictions_not_read(synth):
    """Alden #14: with G0/G1 valid and G2 false, 216.2/.3/.4 are `not read`, not read/FAIL.

    216.1's subject IS G2 so it is read (as FAIL); 216.5/216.6 are unit results that do not depend
    on the lens window, so they stay read.
    """
    rows = _read_jsonl(synth / 'forwards.jsonl')
    for row in rows:
        if row['key'].startswith('control_google'):
            row['lens'][-1]['z_G'], row['lens'][-1]['m'] = -1.0, -1.0
        elif row['key'].startswith('control_alibaba'):
            row['lens'][-1]['z_G'], row['lens'][-1]['m'] = 1.0, 1.0
    _write_jsonl(synth / 'forwards.jsonl', rows)
    proc, score = _run_score(synth)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert score['admission']['admissible'] is True
    assert score['gates']['G0']['pass'] is True
    assert score['gates']['G1']['pass'] is True
    assert score['gates']['G2']['pass'] is False
    assert score['unchanged_outcome']['outcome'] == sc.OUTCOME_NOT_READ
    predictions = score['predictions']
    assert predictions['216.1']['verdict'] == 'read'
    assert predictions['216.1']['pass'] is False
    for key in ('216.2', '216.3', '216.4'):
        assert predictions[key]['verdict'] == sc.OUTCOME_NOT_READ, key
        assert predictions[key]['pass'] is False, key
        assert 'G2 fails' in predictions[key]['reason'], key
    for key in ('216.5', '216.6'):
        assert predictions[key]['verdict'] == 'read', key
    card = (synth / 'SCORECARD.md').read_text(encoding='utf-8')
    assert '| 216.2 | not read |' in card


# ----------------------------------------------------------------------------------------------------------------
# Alden #12 #1: admission constants
# ----------------------------------------------------------------------------------------------------------------

def test_admission_constants_are_bound_to_the_frozen_files(tmp_path):
    """The design's 456 keys and the pins are the committed artefacts, not a run's own manifest."""
    assert sc.EXPECTED_ARMS == DESIGN_ARMS
    assert len(sc.expected_keys()) == 456
    assert len(set(sc.expected_keys())) == 456
    assert sum(1 for k in sc.expected_keys() if k.startswith('factual:')) == 24
    assert sum(1 for k in sc.expected_keys() if k.startswith('control_')) == 48
    assert sc.DESIGN_SHA256 == _sha(HERE / 'DESIGN.md')
    assert sc.PROMPTS_SHA256 == _sha(HERE / 'prompts216.json')
    # The G0 reference is the pinned Part 214 receipt, bound by digest, not by a run's claim.
    assert sc.STAGE1_RECEIPT_PATH == STAGE1
    assert sc.PART214_STAGE1_SHA256 == _sha(STAGE1)
    receipt, reference = sc.load_pinned_receipt()
    assert reference['error'] is None and reference['pinned'] is True
    assert reference['sha256'] == sc.PART214_STAGE1_SHA256
    assert set(sc.G0_ARMS) <= set(receipt['arms'])
    assert sc.bank_binding() == []
    bank = json.loads((HERE / 'prompts216.json').read_text(encoding='utf-8'))
    bank['arms'] = [a for a in bank['arms'] if a != 'op_vogel']
    bad = tmp_path / 'bad-bank.json'
    bad.write_text(json.dumps(bank), encoding='utf-8')
    assert any(p['check'] == 'prompts_bank:arms' for p in sc.bank_binding(bad))


def test_score_tool_does_not_import_torch():
    """The scorer is pure numpy/json: importing it must not pull torch into sys.modules."""
    import subprocess
    code = ('import sys; sys.path.insert(0, %r); import score216; '
            'print("torch" in sys.modules)' % str(HERE))
    out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'
