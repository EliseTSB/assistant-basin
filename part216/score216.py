#!/usr/bin/env python3
"""Part 216 scorer. Pure numpy / json -- **no torch import at module level**.

`part216/DESIGN.md` v2.2 is the contract. Every gate, outcome rule and numbered prediction in the
design has exactly one function here, and `DESIGN.md` names it in `REPORT.md`'s mapping table.

Reads a run directory written by `run216.py`:
    forwards.jsonl   one row per forward (scalars, DLA terms, the 33-row lens table)
    arrays.npz       h0 / h32 / per-unit activations and the frozen unembedding tables
    run.json         the pins (model, revision, design sha256, prompts sha256, tool sha256, ...)

Writes `score216.json` and `SCORECARD.md`.

Nothing here loads a model. `bf16_round` is a NumPy round-to-nearest-even emulation of the bfloat16
cast so the design's analytic G1 fixture can be checked without torch.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# the design's fixed numbers (DESIGN.md "Model", "Arms", "Controls", "Gates", "Outcomes", "Predictions")
# ----------------------------------------------------------------------------------------------------------------

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TARGET = ((6, 753), (7, 1169), (11, 8868), (29, 776))
MONITORS = ((24, 3949), (27, 3325))
GOOGLE_ID, AL_ID, T_ID = 14030, 2033, 51
TOKEN_IDS = (GOOGLE_ID, AL_ID, T_ID)
TOKEN_NAMES = ('Google', 'Alibaba', 'T')

N_QUESTIONS = 24
N_BLOCKS = 32
N_LAYERS = N_BLOCKS + 1              # h_0 .. h_32
OPENING_TOKENS = 5
READ_OFFSET = OPENING_TOKENS - 1     # the read is at T+4, the last opening token

G0_ARMS = ('unchanged', 'zero', 'scale_dial_fact') + tuple(f'random_{k}:zero' for k in range(10))
G0_TOLERANCE = 0.02                  # DESIGN.md G0: within 0.02 on all 312 values
G1_TOLERANCE = 1e-3                  # DESIGN.md G1: within 1e-3 absolute
G2_SHARE = 0.8                       # DESIGN.md G2: both shares at least 0.8
G2_MAX_LAYER = 29                    # DESIGN.md G2: layers l <= 29
G2_MIN_CONTROLS = 18                 # DESIGN.md G2: fewer than 18 of 24 on either side -> not read
G2_MIN_PAIRS = 2                     # DESIGN.md G2: fewer than 2 usable pairs -> not read
LEAD_MIN_QUESTIONS = 12              # DESIGN.md outcomes: at least 12 of 24
NO_LEAD_MAX_QUESTIONS = 2            # DESIGN.md outcomes: at most 2 of 24
DIVERGENCE_THRESHOLD = 0.5           # DESIGN.md outcomes: median m_l - m_l(unchanged) exceeds 0.5
NULL_UNITS = 200                     # DESIGN.md 216.5/216.6: 200 same-layer null units
MIN_VALID_QUESTIONS = 12             # DESIGN.md Measurements 3: fewer than 12 -> `no write`
CONTROL_FINAL_POSITIVE = {'control_google': True, 'control_alibaba': False}

OUTCOME_LEAD = 'early Google lead visible'
OUTCOME_NONE = 'no early lead detected in the qualified window'
OUTCOME_MIXED = 'mixed / inconclusive'
OUTCOME_NOT_READ = 'not read'

EPS = 1e-12


# ----------------------------------------------------------------------------------------------------------------
# bfloat16 arithmetic, in numpy (the design's G1 fixtures are offline and torch-free)
# ----------------------------------------------------------------------------------------------------------------

def bf16_round(values) -> np.ndarray:
    """Round float32 to bfloat16 with round-half-to-even, exactly as a hardware bf16 cast does.

    bfloat16 keeps the float32 sign and exponent and the top 8 bits of the significand, so the cast is
    a rounding of the low 16 bits. The tie case matters here: DESIGN.md's analytic fixture is
    residual 1, write 1/256, and 1 + 2**-8 sits exactly halfway between the two bf16 neighbours of 1
    (ulp at 1.0 is 2**-7), so it must round to even -- down to 1.0.
    """
    x = np.asarray(values, dtype=np.float32)
    shape = x.shape
    flat = np.ascontiguousarray(x.reshape(-1))
    u = flat.view(np.uint32).copy()
    lsb = (u >> 16) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    u = u + bias
    u = u & np.uint32(0xFFFF0000)
    return u.view(np.float32).astype(np.float32).reshape(shape)


def bf16_add(a, b) -> np.ndarray:
    """The bf16 residual add: both operands are bf16 values, the sum is rounded back to bf16."""
    return bf16_round(np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32))


def realized_increment(before, after, raw=None) -> dict:
    """The realized increment across one residual add, and the rounding residual beside it.

    `before` / `after` are the captured residual tensors (bf16 values carried as float32); the realized
    increment is their FP32 difference. `raw` is the module output the model added; the rounding
    residual is `realized - raw`, which is what bf16 rounding of the add destroyed (or, for a
    subtraction, added).
    """
    b = np.asarray(before, dtype=np.float32)
    a = np.asarray(after, dtype=np.float32)
    inc = (a - b).astype(np.float32)
    out = {'increment': inc}
    if raw is not None:
        out['rounding'] = (inc - np.asarray(raw, dtype=np.float32)).astype(np.float32)
    return out


# ----------------------------------------------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------------------------------------------

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'object of type {type(obj).__name__} is not JSON serializable')


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=json_default) + '\n',
                   encoding='utf-8')
    tmp.replace(path)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# the lens table
# ----------------------------------------------------------------------------------------------------------------

def margin(z: dict) -> float:
    """m = logit(Google) - max(logit(Al), logit(T)) -- the rank-preserving margin (DESIGN.md v2.2 #2)."""
    return float(z['z_G']) - max(float(z['z_Al']), float(z['z_T']))


def g_ali_of(z: dict) -> float:
    """g_ali = logit(Google) - logsumexp(logit(Al), logit(T)). G0 only (DESIGN.md v2.2 #2)."""
    return float(z['z_G']) - float(np.logaddexp(float(z['z_Al']), float(z['z_T'])))


def row_margins(row: dict) -> list[float]:
    """m_l for l = 0..N_LAYERS-1, out of a run row's `lens` list."""
    lens = row['lens']
    if len(lens) != N_LAYERS:
        raise ValueError(f'{row.get("key")}: lens has {len(lens)} layers, the design pins {N_LAYERS}')
    return [margin(entry) for entry in lens]


def row_g_ali(row: dict) -> list[float]:
    return [g_ali_of(entry) for entry in row['lens']]


def m32(row: dict) -> float:
    return margin(row['lens'][-1])


# ----------------------------------------------------------------------------------------------------------------
# G0 -- the anchor against Part 214's Stage 1 receipt
# ----------------------------------------------------------------------------------------------------------------

def gate0(rows: dict, receipt: dict, tolerance: float = G0_TOLERANCE) -> dict:
    """G0: final g_ali for the 13 anchor arms, questions 0-23, equals Part 214's receipt within 0.02.

    The receipt's question index 24 is a separate positive control and is NOT part of G0
    (DESIGN.md v2.2 #9). 13 arms x 24 questions = 312 values.  Fail: STOP.
    """
    arms = receipt.get('arms') or {}
    values = {}
    failures = []
    worst = 0.0
    for arm in G0_ARMS:
        ref = arms.get(arm)
        if ref is None:
            failures.append({'arm': arm, 'reason': 'arm missing from the Part 214 receipt'})
            continue
        ref_g = ref.get('g_ali')
        if ref_g is None:
            failures.append({'arm': arm, 'reason': 'receipt arm has no g_ali list'})
            continue
        for qi in range(N_QUESTIONS):
            row = rows.get(f'{arm}:{qi}')
            if row is None:
                failures.append({'arm': arm, 'question_index': qi, 'reason': 'forward row missing'})
                continue
            got = float(row['native_g_ali'])
            want = float(ref_g[qi])
            delta = abs(got - want)
            worst = max(worst, delta)
            values[f'{arm}:{qi}'] = {'got': got, 'receipt': want, 'abs_delta': delta}
            if delta > tolerance:
                failures.append({'arm': arm, 'question_index': qi, 'got': got, 'receipt': want,
                                 'abs_delta': delta})
    exact = sum(1 for v in values.values() if v['abs_delta'] == 0.0)
    return {
        'gate': 'G0', 'pass': not failures and len(values) == len(G0_ARMS) * N_QUESTIONS,
        'n_values': len(values), 'n_expected': len(G0_ARMS) * N_QUESTIONS,
        'max_abs_delta': worst, 'exact_agreement': exact, 'tolerance': tolerance,
        'failures': failures[:40], 'n_failures': len(failures),
        'values': values,
    }


# ----------------------------------------------------------------------------------------------------------------
# G1 -- telescoping on realized increments
# ----------------------------------------------------------------------------------------------------------------

def gate1_check(emb_term, dla_terms, logits, tolerance: float = G1_TOLERANCE) -> dict:
    """One forward's FP32 telescoping check: embedding term + every realized increment == the logit.

    `emb_term` is (3,), `dla_terms` is (n_boundaries, 3), `logits` is the (3,) FP32 logit computed
    independently from `h_32` with the frozen RMS. The design also requires that all 64 named add
    boundaries were present exactly once -- that is checked separately, per forward, by
    `boundary_check`.
    """
    emb = np.asarray(emb_term, dtype=np.float64)
    terms = np.asarray(dla_terms, dtype=np.float64)
    target = np.asarray(logits, dtype=np.float64)
    if terms.ndim != 2 or terms.shape[1] != 3:
        raise ValueError(f'dla_terms must be (n_boundaries, 3), got {terms.shape}')
    total = emb + terms.sum(axis=0)
    err = np.abs(total - target)
    return {'pass': bool(np.all(err <= tolerance)), 'max_abs_error': float(err.max()),
            'per_token_error': err.tolist(), 'tolerance': tolerance,
            'n_terms': int(terms.shape[0])}


def boundary_check(row: dict, n_blocks: int = N_BLOCKS) -> dict:
    """The named add boundaries: 2 per block (mixer add, MLP add), each present exactly once."""
    counts = row.get('boundary_counts') or {}
    expected = []
    for li in range(n_blocks):
        expected.append(f'L{li}:add1')
        expected.append(f'L{li}:add2')
    # The block-input captures are part of the same ledger (they prove the residual chain is
    # unbroken) but they are not add boundaries, so they are allowed, not required, as adds.
    allowed = set(expected) | {'h0'} | {f'L{li}:in' for li in range(n_blocks)}
    missing = [name for name in expected if counts.get(name, 0) == 0]
    duplicated = [name for name in expected if counts.get(name, 0) > 1]
    unexpected = sorted(set(counts) - allowed)
    return {'pass': not missing and not duplicated and not unexpected,
            'n_expected': len(expected), 'n_present': sum(1 for n in expected if counts.get(n, 0) == 1),
            'missing': missing[:20], 'duplicated': duplicated[:20], 'unexpected': unexpected[:20]}


def gate1(rows: dict, n_blocks: int = N_BLOCKS, tolerance: float = G1_TOLERANCE) -> dict:
    """G1 over the whole run: every forward telescopes, and every forward has its 2*n_blocks adds once."""
    failures = []
    worst_err = 0.0
    worst_boundary = None
    for key, row in rows.items():
        check = gate1_check(row['dla']['emb_term'], row['dla']['terms'], row['dla']['fp32_logits'],
                            tolerance)
        if check['max_abs_error'] > worst_err:
            worst_err = check['max_abs_error']
        bcheck = boundary_check(row, n_blocks)
        if not bcheck['pass'] and worst_boundary is None:
            worst_boundary = {'key': key, **bcheck}
        if not check['pass'] or not bcheck['pass']:
            failures.append({'key': key, 'telescoping': check['pass'],
                             'max_abs_error': check['max_abs_error'],
                             'boundaries_ok': bcheck['pass'],
                             'n_boundaries_present': bcheck['n_present']})
    return {'gate': 'G1', 'pass': not failures, 'n_rows': len(rows),
            'max_abs_error': worst_err, 'tolerance': tolerance,
            'first_bad_boundaries': worst_boundary, 'failures': failures[:40],
            'n_failures': len(failures)}


# ----------------------------------------------------------------------------------------------------------------
# G2 -- the qualified window
# ----------------------------------------------------------------------------------------------------------------

def expected_side_final(arm: str, final_m: float) -> bool:
    want_positive = CONTROL_FINAL_POSITIVE[arm]
    return final_m > 0.0 if want_positive else final_m < 0.0


def gate2(control_rows: dict, share: float = G2_SHARE, max_layer: int = G2_MAX_LAYER,
          min_controls: int = G2_MIN_CONTROLS, min_pairs: int = G2_MIN_PAIRS) -> dict:
    """G2: the layer window where BOTH control sides read their own way >= 0.8 of the time.

    A control counts only if its FINAL contrast is on its expected side (DESIGN.md "Gates" G2).
    `qualified` is the set of layers l <= max_layer with both shares >= share; a **usable pair** is
    two consecutive layers both in the window.  Fewer than 18 of 24 controls on either side, or
    fewer than 2 usable pairs, is `not read`.
    """
    sides = {'control_google': [], 'control_alibaba': []}
    for arm in sides:
        for qi in range(N_QUESTIONS):
            row = control_rows.get(f'{arm}:{qi}')
            if row is not None:
                sides[arm].append(row)
    counted = {}
    per_layer = []
    for arm, name in (('control_google', 'google'), ('control_alibaba', 'alibaba')):
        n_final = 0
        n_hit = {l: 0 for l in range(N_LAYERS)}
        for row in sides[arm]:
            ms = row_margins(row)
            if not expected_side_final(arm, ms[-1]):
                continue
            n_final += 1
            for l, m in enumerate(ms):
                if (m > 0.0) if arm == 'control_google' else (m < 0.0):
                    n_hit[l] += 1
        counted[name] = {'n_rows': len(sides[arm]), 'n_final_expected': n_final,
                         'n_hit': {str(l): n_hit[l] for l in range(N_LAYERS)}}
        per_layer.append((name, n_final, n_hit))

    window = []
    for l in range(0, max_layer + 1):
        shares = {}
        ok = True
        for name, n_final, n_hit in per_layer:
            s = (n_hit[l] / n_final) if n_final else 0.0
            shares[name] = s
            if n_final < min_controls or s < share:
                ok = False
        window.append({'layer': l, 'google_share': shares['google'],
                       'alibaba_share': shares['alibaba'], 'qualified': ok})
    qualified = [r['layer'] for r in window if r['qualified']]
    pairs = usable_pairs(qualified)
    enough_controls = all(counted[name]['n_final_expected'] >= min_controls
                          for name in ('google', 'alibaba'))
    gates = {
        'controls_18_of_24': bool(enough_controls),
        'at_least_2_usable_pairs': len(pairs) >= min_pairs,
    }
    return {'gate': 'G2', 'pass': all(gates.values()), 'checks': gates,
            'n_usable_pairs': len(pairs), 'usable_pairs': pairs,
            'window': qualified, 'window_min': min(qualified) if qualified else None,
            'window_max': max(qualified) if qualified else None,
            'per_layer': window, 'controls': counted,
            'share_threshold': share, 'max_layer': max_layer}


def usable_pairs(qualified: list[int]) -> list[list[int]]:
    """Two consecutive layers, both qualified (DESIGN.md v2.1 #1)."""
    s = set(qualified)
    return [[l, l + 1] for l in sorted(s) if l + 1 in s]


# ----------------------------------------------------------------------------------------------------------------
# outcomes
# ----------------------------------------------------------------------------------------------------------------

def question_outcome(row_unchanged: dict, window: list[int], pairs: list[list[int]]) -> dict:
    """One question's reading under `unchanged` (DESIGN.md "Outcomes").

    `early Google lead` needs m_l > 0 on BOTH layers of at least one usable pair AND a final m_32 < 0.
    `no early lead` needs m_l <= 0 on every layer in W.  Both can be false: that is the mixed question.
    """
    ms = row_margins(row_unchanged)
    final = ms[-1]
    pair_hits = [[l, l + 1] for l, l1 in pairs if ms[l] > 0.0 and ms[l1] > 0.0]
    any_positive_in_window = [l for l in window if ms[l] > 0.0]
    early_lead = bool(pair_hits) and final < 0.0
    no_early = all(ms[l] <= 0.0 for l in window)
    return {'question_index': row_unchanged.get('question_index'),
            'm_final': final, 'early_lead': early_lead, 'no_early_lead': no_early,
            'pair_hits': pair_hits, 'any_positive_in_window': any_positive_in_window,
            'n_positive_in_window': len(any_positive_in_window)}


def unchanged_outcome(rows: dict, g2: dict) -> dict:
    """The `unchanged` outcome statement, counted over 24 questions (DESIGN.md "Outcomes")."""
    if not g2['pass']:
        return {'outcome': OUTCOME_NOT_READ, 'reason': 'G2 fails: sparse qualification is not evidence '
                                                       'for "no early lead detected"',
                'g2_checks': g2['checks'], 'questions': []}
    window, pairs = g2['window'], g2['usable_pairs']
    per_q = []
    for qi in range(N_QUESTIONS):
        row = rows.get(f'unchanged:{qi}')
        if row is None:
            raise KeyError(f'unchanged:{qi} is missing from the run')
        per_q.append(question_outcome(row, window, pairs))
    n_lead = sum(1 for q in per_q if q['early_lead'])
    n_any_positive = sum(1 for q in per_q if q['n_positive_in_window'] > 0)
    if n_lead >= LEAD_MIN_QUESTIONS:
        outcome = OUTCOME_LEAD
    elif n_any_positive <= NO_LEAD_MAX_QUESTIONS:
        outcome = OUTCOME_NONE
    else:
        outcome = OUTCOME_MIXED
    return {'outcome': outcome, 'n_questions': len(per_q), 'n_early_lead': n_lead,
            'n_any_positive_in_window': n_any_positive,
            'lead_threshold': LEAD_MIN_QUESTIONS, 'no_lead_threshold': NO_LEAD_MAX_QUESTIONS,
            'questions': per_q}


def divergence_index(arm_rows: dict, unchanged_rows: dict, g2: dict,
                     threshold: float = DIVERGENCE_THRESHOLD, arm: str | None = None) -> dict:
    """The first layer l >= min(W) from which the per-question median of m_l - m_l(unchanged)
    exceeds `threshold` through h_32 (DESIGN.md "Outcomes").

    `arm_rows` is keyed `'<arm>:<qi>'`; the comparison is per QUESTION (same question, edited against
    unchanged), never between medians of different question sets.
    """
    if not g2['pass'] or g2['window_min'] is None:
        return {'index': None, 'layer': None, 'reason': 'G2 fails', 'profile': []}
    if arm is None:
        arm = next(iter(arm_rows)).rsplit(':', 1)[0] if arm_rows else ''
    start = int(g2['window_min'])
    profile = []
    for l in range(start, N_LAYERS):
        deltas = []
        for key, base in sorted(unchanged_rows.items()):
            qi = key.rsplit(':', 1)[1]
            edited = arm_rows.get(f'{arm}:{qi}')
            if edited is None:
                continue
            deltas.append(margin(edited['lens'][l]) - margin(base['lens'][l]))
        profile.append({'layer': l, 'median_delta': float(np.median(deltas)) if deltas else None,
                        'n_questions': len(deltas)})
    index = None
    for i, entry in enumerate(profile):
        if entry['median_delta'] is None:
            continue
        if entry['median_delta'] > threshold and all(p['median_delta'] is not None
                                                     and p['median_delta'] > threshold
                                                     for p in profile[i:]):
            index = entry['layer']
            break
    return {'index': index, 'layer': f'h_{index}' if index is not None else None,
            'start_layer': start, 'threshold': threshold, 'profile': profile}


def arm_ends_in_google(rows: dict, arm: str) -> dict:
    """Per edited/opener arm: `ends in Google` iff m_32 > 0 (DESIGN.md "Outcomes")."""
    per_q = []
    for qi in range(N_QUESTIONS):
        row = rows.get(f'{arm}:{qi}')
        if row is None:
            continue
        per_q.append({'question_index': qi, 'm_final': m32(row), 'ends_in_google': m32(row) > 0.0})
    return {'n_questions': len(per_q), 'n_ends_in_google': sum(1 for q in per_q if q['ends_in_google']),
            'questions': per_q}


# ----------------------------------------------------------------------------------------------------------------
# unit DLA and the same-slot null (216.5 / 216.6)
# ----------------------------------------------------------------------------------------------------------------

def frozen_scale(h32: np.ndarray, eps: float) -> float:
    """The RMS scale the design freezes for the DLA: 1 / sqrt(mean(x^2) + eps) of h_32."""
    x = np.asarray(h32, dtype=np.float64)
    return float(1.0 / math.sqrt(float(np.mean(x * x)) + eps))


def fix_rms_logits(h: np.ndarray, norm_weight: np.ndarray, unembed: np.ndarray,
                   scale: float) -> np.ndarray:
    """(3,) logits under a FROZEN RMS scale: scale * (1 + w) * h . W_U[token]."""
    x = np.asarray(h, dtype=np.float64)
    w = 1.0 + np.asarray(norm_weight, dtype=np.float64)
    u = np.asarray(unembed, dtype=np.float64)
    return scale * (u @ (w * x))


def split_lse(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def unit_terms(activation: np.ndarray, col_norm: np.ndarray, proj: np.ndarray,
               scale: float) -> dict:
    """One forward's per-unit quantities at the read position.

    `activation` is the down_proj input (n_units,); `col_norm` is ||W_down[:, u]||;
    `proj[u, k]` is sum_i W_down[i, u] * (1 + w_i) * W_U[token_k, i].
    c_k = scale * a_u * proj[u, k]; write size = |a_u| * ||W_down[:, u]||.
    """
    a = np.asarray(activation, dtype=np.float64)
    write = np.abs(a) * np.asarray(col_norm, dtype=np.float64)
    c = float(scale) * a[:, None] * np.asarray(proj, dtype=np.float64)
    return {'write': write, 'c_G': c[:, 0], 'c_Al': c[:, 1], 'c_T': c[:, 2]}


def d_alibaba(c_Al, c_T, z_Al, z_T) -> np.ndarray:
    """d_A = LSE(z_Al, z_T) - LSE(z_Al - c_Al, z_T - c_T), the finite fixed-RMS family contrast.

    Zero for a zero write (both c terms are zero then), which is what makes a `no write` unit
    unreadable rather than infinitely extreme.
    """
    cA = np.asarray(c_Al, dtype=np.float64)
    cT = np.asarray(c_T, dtype=np.float64)
    return np.array([split_lse(float(z_Al), float(z_T)) - split_lse(float(z_Al) - ca, float(z_T) - ct)
                     for ca, ct in zip(cA, cT)], dtype=np.float64)


def median_or_none(values) -> float | None:
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def target_series(unit_pack: dict, layer: int, unit: int) -> dict:
    """One unit's per-question readings at this slot: write size, c_G, d_A, and their ratios."""
    write = np.asarray(unit_pack['write'][layer], dtype=np.float64)[:, unit]
    c_g = np.asarray(unit_pack['c_G'][layer], dtype=np.float64)[:, unit]
    d_a = np.asarray(unit_pack['d_A'][layer], dtype=np.float64)[:, unit]
    valid = write > 0.0
    n_valid = int(valid.sum())
    out = {
        'layer': int(layer), 'unit': int(unit), 'n_questions': int(write.shape[0]),
        'n_valid': n_valid,
        'median_write': median_or_none(write),
        'median_c_G': median_or_none(c_g),
        'median_d_A': median_or_none(d_a),
        'median_c_G_per_write': median_or_none(c_g[valid] / write[valid]) if n_valid else None,
        'median_d_A_per_write': median_or_none(d_a[valid] / write[valid]) if n_valid else None,
    }
    out['no_write'] = n_valid < MIN_VALID_QUESTIONS
    if out['no_write']:
        out['median_c_G_per_write'] = None
        out['median_d_A_per_write'] = None
    return out


def null_units(median_write: np.ndarray, target_units: set[int], layer_unit: int,
               n_null: int = NULL_UNITS) -> dict:
    """The `n_null` units of the layer nearest the target in MEDIAN WRITE SIZE at this slot.

    Exclusions (DESIGN.md 216.5): the target units themselves, and units whose median write size is 0.
    Ties in the matching are broken by unit index, which is a deterministic, token-independent order
    (the matching is explicitly "without reference to any token direction").
    """
    mw = np.asarray(median_write, dtype=np.float64)
    target_m = float(mw[layer_unit])
    candidates = [u for u in range(mw.shape[0])
                  if u not in target_units and mw[u] > 0.0]
    order = sorted(candidates, key=lambda u: (abs(mw[u] - target_m), u))
    kept = order[:n_null]
    return {'units': kept, 'n': len(kept), 'target_median_write': target_m,
            'matching': 'nearest median write size at T+4, ties by unit index',
            'excluded_zero_write': int(sum(1 for u in range(mw.shape[0])
                                           if u not in target_units and mw[u] <= 0.0))}


def extremeness(target_value: float | None, null_values: list[float],
                direction: str) -> dict:
    """Strict extremeness against every null unit; a TIE with any null unit does not count."""
    if target_value is None or not null_values:
        return {'pass': False, 'reason': 'no target value or no null units', 'n_beaten': 0,
                'n_ties': 0, 'n_null': len(null_values)}
    n_beaten = 0
    n_ties = 0
    for v in null_values:
        if v == target_value:
            n_ties += 1
        elif (target_value < v) if direction == 'below' else (target_value > v):
            n_beaten += 1
    return {'pass': n_beaten == len(null_values) and n_ties == 0,
            'direction': direction, 'target': target_value, 'n_null': len(null_values),
            'n_beaten': n_beaten, 'n_ties': n_ties,
            'null_min': float(np.min(null_values)), 'null_max': float(np.max(null_values))}


def prediction_216_5_unit(series: dict, null_series: list[dict]) -> dict:
    """One target unit against its 200 same-layer null units for 216.5."""
    if series['no_write'] or series['median_c_G_per_write'] is None:
        return {'layer': series['layer'], 'unit': series['unit'], 'status': 'no write',
                'pass': False, 'reason': f'{series["n_valid"]} valid questions < '
                                         f'{MIN_VALID_QUESTIONS}'}
    negative = series['median_c_G'] is not None and series['median_c_G'] < 0.0
    cmp = extremeness(series['median_c_G_per_write'],
                      [n['median_c_G_per_write'] for n in null_series if n['median_c_G_per_write'] is not None],
                      'below')
    return {'layer': series['layer'], 'unit': series['unit'], 'status': 'read',
            'median_c_G_negative': bool(negative),
            'median_c_G_per_write': series['median_c_G_per_write'],
            'n_null_compared': cmp['n_null'], 'n_beaten': cmp['n_beaten'], 'n_ties': cmp['n_ties'],
            'strictly_below_all_nulls': bool(cmp['pass']),
            'pass': bool(negative and cmp['pass'])}


def prediction_216_6_unit(series: dict, null_series: list[dict]) -> dict:
    """One target unit against its 200 same-layer null units for 216.6."""
    if series['no_write'] or series['median_d_A_per_write'] is None:
        return {'layer': series['layer'], 'unit': series['unit'], 'status': 'no write',
                'pass': False, 'reason': f'{series["n_valid"]} valid questions < '
                                         f'{MIN_VALID_QUESTIONS}'}
    positive = series['median_d_A'] is not None and series['median_d_A'] > 0.0
    cmp = extremeness(series['median_d_A_per_write'],
                      [n['median_d_A_per_write'] for n in null_series if n['median_d_A_per_write'] is not None],
                      'above')
    return {'layer': series['layer'], 'unit': series['unit'], 'status': 'read',
            'median_d_A_positive': bool(positive),
            'median_d_A_per_write': series['median_d_A_per_write'],
            'n_null_compared': cmp['n_null'], 'n_beaten': cmp['n_beaten'], 'n_ties': cmp['n_ties'],
            'strictly_above_all_nulls': bool(cmp['pass']),
            'pass': bool(positive and cmp['pass'])}


def unit_analysis(rows: dict, arrays: dict) -> dict:
    """216.5 and 216.6, on the `unchanged` arm at T+4.

    `arrays` carries `units` (n_questions, n_unit_layers, n_units), `unit_layers`,
    `unit_proj` (n_unit_layers, n_units, 3), `unit_colnorm` (n_unit_layers, n_units) and
    `scales` (n_questions,) -- the frozen RMS scale of each question's h_32.
    """
    unit_layers = [int(l) for l in arrays['unit_layers']]
    slot = {layer: k for k, layer in enumerate(unit_layers)}
    n_units = int(arrays['unit_proj'].shape[1])
    scales = np.asarray(arrays['scales'], dtype=np.float64)
    ordered = sorted((qi for qi in range(N_QUESTIONS) if f'unchanged:{qi}' in rows))
    pack = {'write': {}, 'c_G': {}, 'd_A': {}}
    for layer in unit_layers:
        k = slot[layer]
        act = np.stack([np.asarray(arrays['units'][str(qi)][k], dtype=np.float64) for qi in ordered])
        z = np.stack([np.asarray(rows[f'unchanged:{qi}']['dla']['fp32_logits'], dtype=np.float64)
                      for qi in ordered])
        write, c_g, d_a = [], [], []
        for i in range(len(ordered)):
            # The frozen RMS scale is per question (it is h_32's own RMS), and d_A is NONLINEAR in
            # c_Al / c_T, so the scale has to go in before the LSE, not after it.
            terms = unit_terms(act[i], arrays['unit_colnorm'][k], arrays['unit_proj'][k], scales[i])
            write.append(terms['write'])
            c_g.append(terms['c_G'])
            d_a.append(d_alibaba(terms['c_Al'], terms['c_T'], z[i, 1], z[i, 2]))
        pack['write'][layer] = np.stack(write)
        pack['c_G'][layer] = np.stack(c_g)
        pack['d_A'][layer] = np.stack(d_a)
    targets = {}
    for layer, unit in TARGET:
        targets[f'{layer}:{unit}'] = (layer, unit)
    out = {'unit_layers': unit_layers, 'n_questions': len(ordered),
           'min_valid_questions': MIN_VALID_QUESTIONS, 'n_null': NULL_UNITS, 'units': {}}
    for label, (layer, unit) in targets.items():
        mw = np.median(pack['write'][layer], axis=0)
        target_set = {u for l, u in TARGET if l == layer}
        null = null_units(mw, target_set, unit)
        series = target_series(pack, layer, unit)
        null_series = [target_series(pack, layer, u) for u in null['units']]
        out['units'][label] = {
            'target': series, 'null': {k: v for k, v in null.items() if k != 'units'},
            'null_unit_sample': null['units'][:20],
            '216.5': prediction_216_5_unit(series, null_series),
            '216.6': prediction_216_6_unit(series, null_series),
        }
    monitors = {}
    for layer, unit in MONITORS:
        if layer in slot:
            monitors[f'{layer}:{unit}'] = target_series(pack, layer, unit)
    out['monitors'] = monitors
    out['216.5'] = {'pass': any(v['216.5']['pass'] for v in out['units'].values()),
                    'n_units_passing': sum(1 for v in out['units'].values() if v['216.5']['pass'])}
    out['216.6'] = {'pass': any(v['216.6']['pass'] for v in out['units'].values()),
                    'n_units_passing': sum(1 for v in out['units'].values() if v['216.6']['pass'])}
    return out


# ----------------------------------------------------------------------------------------------------------------
# descriptive tables (DESIGN.md "Descriptive, not scored")
# ----------------------------------------------------------------------------------------------------------------

def share_tables(rows: dict, arm: str, baseline: str = 'unchanged') -> dict:
    """Per layer, the median change in z_G and in max(z_Al, z_T) against `unchanged`."""
    deltas_g, deltas_c = [], []
    for l in range(N_LAYERS):
        dg, dc = [], []
        for qi in range(N_QUESTIONS):
            a = rows.get(f'{arm}:{qi}')
            b = rows.get(f'{baseline}:{qi}')
            if a is None or b is None:
                continue
            dg.append(a['lens'][l]['z_G'] - b['lens'][l]['z_G'])
            dc.append(max(a['lens'][l]['z_Al'], a['lens'][l]['z_T'])
                      - max(b['lens'][l]['z_Al'], b['lens'][l]['z_T']))
        deltas_g.append(float(np.median(dg)) if dg else None)
        deltas_c.append(float(np.median(dc)) if dc else None)
    return {'arm': arm, 'baseline': baseline,
            'median_delta_z_G': deltas_g, 'median_delta_max_z_Al_T': deltas_c}


def block_share_table(rows: dict, arm: str) -> dict:
    """Per block, the attention and MLP shares of z_G and of the Alibaba family log-sum."""
    out = []
    for b in range(N_BLOCKS):
        g_att, g_mlp, a_att, a_mlp = [], [], [], []
        for qi in range(N_QUESTIONS):
            row = rows.get(f'{arm}:{qi}')
            if row is None:
                continue
            terms = row['dla']['terms']
            if len(terms) != 2 * N_BLOCKS:
                continue
            g_att.append(terms[2 * b][0])
            g_mlp.append(terms[2 * b + 1][0])
            a_att.append(terms[2 * b][1] + terms[2 * b][2])
            a_mlp.append(terms[2 * b + 1][1] + terms[2 * b + 1][2])
        out.append({'block': b,
                    'median_attn_z_G': float(np.median(g_att)) if g_att else None,
                    'median_mlp_z_G': float(np.median(g_mlp)) if g_mlp else None,
                    'median_attn_alibaba_family': float(np.median(a_att)) if a_att else None,
                    'median_mlp_alibaba_family': float(np.median(a_mlp)) if a_mlp else None})
    return {'arm': arm, 'blocks': out}


def factual_check(rows: dict) -> dict:
    """The instrument check: lens top-1 against final top-1 per layer, plus the input-token echo rate."""
    per_layer = []
    for l in range(N_LAYERS):
        agree = 0
        echo = 0
        n = 0
        for qi in range(N_QUESTIONS):
            row = rows.get(f'factual:{qi}')
            if row is None:
                continue
            n += 1
            entry = row['lens'][l]
            if entry['argmax'] == row['lens'][-1]['argmax']:
                agree += 1
            if entry['argmax'] == row.get('input_token_id'):
                echo += 1
        per_layer.append({'layer': l, 'n': n,
                          'lens_top1_matches_final': agree,
                          'share': (agree / n) if n else None,
                          'input_token_echo': echo,
                          'echo_share': (echo / n) if n else None})
    return {'per_layer': per_layer, 'n_items': N_QUESTIONS}


# ----------------------------------------------------------------------------------------------------------------
# the whole scorecard
# ----------------------------------------------------------------------------------------------------------------

def load_run(out_dir) -> tuple[dict, dict, dict]:
    """Read `forwards.jsonl`, `arrays.npz` and `run.json` out of a run directory."""
    out = Path(out_dir)
    rows = {}
    with (out / 'forwards.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[rec['key']] = rec
    run = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    arrays = {}
    npz_path = out / 'arrays.npz'
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as z:
            for name in z.files:
                arrays[name] = z[name]
    return rows, arrays, run


def unit_arrays(rows: dict, arrays: dict) -> dict:
    """Reshape the flat `units:<key>` npz entries the runner wrote into the per-question pack."""
    if 'unit_layers' in arrays:
        return {
            'unit_layers': arrays['unit_layers'],
            'unit_proj': arrays['unit_proj'],
            'unit_colnorm': arrays['unit_colnorm'],
            'scales': arrays['scales'],
            'units': {str(qi): arrays[f'units:unchanged:{qi}'] for qi in range(N_QUESTIONS)
                      if f'units:unchanged:{qi}' in arrays},
        }
    raise KeyError('arrays.npz has no unit tables; the runner did not record the target-layer activations')


def score_run(out_dir) -> dict:
    """Everything DESIGN.md asks the scorer to report, from one run directory."""
    rows, arrays, run = load_run(out_dir)
    receipt = json.loads(Path(run['stage1_receipt_path']).read_text(encoding='utf-8'))
    g0 = gate0(rows, receipt)
    g1 = gate1(rows, n_blocks=int(run.get('n_blocks', N_BLOCKS)))
    controls = {k: v for k, v in rows.items() if k.startswith('control_')}
    g2 = gate2(controls)
    unchanged = unchanged_outcome(rows, g2)

    unchanged_rows = {f'unchanged:{qi}': rows[f'unchanged:{qi}'] for qi in range(N_QUESTIONS)
                      if f'unchanged:{qi}' in rows}
    arms = {}
    for arm in run.get('arms', []):
        ends = arm_ends_in_google(rows, arm)
        div = divergence_index({f'{arm}:{qi}': rows[f'{arm}:{qi}'] for qi in range(N_QUESTIONS)
                                if f'{arm}:{qi}' in rows}, unchanged_rows, g2, arm=arm) \
            if arm != 'unchanged' else {'index': None, 'layer': None, 'reason': 'baseline'}
        arms[arm] = {'ends_in_google': ends, 'divergence': div}

    units = unit_analysis(rows, unit_arrays(rows, arrays))

    predictions = {}
    predictions['216.1'] = {'pass': bool(g2['n_usable_pairs'] >= 6),
                            'n_usable_pairs': g2['n_usable_pairs'], 'threshold': 6}
    predictions['216.2'] = {'pass': unchanged['outcome'] == OUTCOME_LEAD,
                            'outcome': unchanged['outcome']}
    predictions['216.3'] = {'pass': unchanged['outcome'] == OUTCOME_NONE,
                            'outcome': unchanged['outcome']}
    zero_div = arms.get('zero', {}).get('divergence', {})
    predictions['216.4'] = {'pass': bool(zero_div.get('index') is not None and zero_div['index'] >= 7),
                            'divergence_index': zero_div.get('index'),
                            'note': 'h_7 or later and exists (the first zeroed unit is in block 6)'}
    predictions['216.5'] = units['216.5']
    predictions['216.6'] = units['216.6']

    descriptive = {
        'per_layer_z': {arm: share_tables(rows, arm) for arm in run.get('arms', [])},
        'block_shares': {arm: block_share_table(rows, arm) for arm in ('unchanged', 'zero',
                                                                       'scale_dial_fact')},
        'factual': factual_check(rows),
    }
    return {
        'part': 216, 'design_version': 'v2.2',
        'run_pins': {k: run.get(k) for k in ('model', 'revision', 'design_sha256', 'prompts_sha256',
                                             'tool_sha256', 'stage1_receipt_sha256', 'n_blocks',
                                             'hidden_size')},
        'gates': {'G0': g0, 'G1': g1, 'G2': g2},
        'stop': bool(not g0['pass'] or not g1['pass']),
        'unchanged_outcome': unchanged,
        'arms': arms,
        'units': units,
        'predictions': predictions,
        'descriptive': descriptive,
    }


def scorecard_markdown(score: dict) -> str:
    """A human-readable SCORECARD.md: gates, outcomes, predictions, then the descriptive tables."""
    lines = ['# Part 216 scorecard', '']
    pins = score.get('run_pins', {})
    lines += [f"- model: `{pins.get('model')}@{pins.get('revision')}`",
              f"- design sha256: `{pins.get('design_sha256')}`",
              f"- prompts sha256: `{pins.get('prompts_sha256')}`",
              f"- tool sha256: `{pins.get('tool_sha256')}`", '']
    gates = score['gates']
    lines += ['## Gates', '', '| gate | pass | detail |', '|---|---|---|']
    g0, g1, g2 = gates['G0'], gates['G1'], gates['G2']
    lines.append(f"| G0 | {'PASS' if g0['pass'] else 'FAIL'} | {g0['n_values']}/{g0['n_expected']} "
                 f"values, max abs delta {g0['max_abs_delta']:.6f} (tol {g0['tolerance']}), "
                 f"{g0['exact_agreement']} exact |")
    lines.append(f"| G1 | {'PASS' if g1['pass'] else 'FAIL'} | {g1['n_rows']} rows, max abs error "
                 f"{g1['max_abs_error']:.3e} (tol {g1['tolerance']}) |")
    lines.append(f"| G2 | {'PASS' if g2['pass'] else 'FAIL'} | {g2['n_usable_pairs']} usable pairs, "
                 f"window {g2['window']} |")
    lines += ['', '## Outcome', '',
              f"`unchanged`: **{score['unchanged_outcome']['outcome']}** "
              f"({score['unchanged_outcome'].get('n_early_lead')} early-lead questions, "
              f"{score['unchanged_outcome'].get('n_any_positive_in_window')} with any positive m in W)", '']
    lines += ['## Predictions', '', '| id | pass | detail |', '|---|---|---|']
    for key in ('216.1', '216.2', '216.3', '216.4', '216.5', '216.6'):
        entry = score['predictions'].get(key, {})
        detail = ', '.join(f'{k}={v}' for k, v in entry.items()
                           if k in ('n_usable_pairs', 'outcome', 'divergence_index',
                                    'n_units_passing', 'note'))
        lines.append(f"| {key} | {'PASS' if entry.get('pass') else 'FAIL'} | {detail} |")
    lines += ['', '## Arms', '', '| arm | ends in Google | divergence |', '|---|---|---|']
    for arm, entry in score['arms'].items():
        div = entry['divergence'].get('layer') or entry['divergence'].get('reason') or '--'
        lines.append(f"| `{arm}` | {entry['ends_in_google']['n_ends_in_google']}/"
                     f"{entry['ends_in_google']['n_questions']} | {div} |")
    fact = score['descriptive']['factual']['per_layer']
    mid = fact[len(fact) // 2] if fact else {}
    lines += ['', '## Factual instrument check', '',
              f"lens top-1 matches final top-1 at h_{mid.get('layer')}: "
              f"{mid.get('lens_top1_matches_final')}/{mid.get('n')}; "
              f"input-token echo {mid.get('input_token_echo')}/{mid.get('n')} "
              f"(all layers in score216.json).", '']
    return '\n'.join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Part 216 scorer (numpy only).')
    ap.add_argument('run_dir', help='a directory written by run216.py run')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.run_dir)
    score = score_run(args.run_dir)
    write_json(out / 'score216.json', score)
    (out / 'SCORECARD.md').write_text(scorecard_markdown(score), encoding='utf-8')
    print(f'wrote {out / "score216.json"} and {out / "SCORECARD.md"}')
    print('gates', {k: v['pass'] for k, v in score['gates'].items()},
          'outcome', score['unchanged_outcome']['outcome'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
