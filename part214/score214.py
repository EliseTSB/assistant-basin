#!/usr/bin/env python3
"""Part 214 scorer: Stage 0 dial projections, Stage 1 H4 curve & diagnostics, Stage 2 five-state ladder,
every gate and every verdict of predictions 214.1-214.8, from saved JSON/JSONL only.

PART214-DESIGN.md v5, as amended at commit cb8cc1e4, is the contract; design line numbers are cited beside each rule
and every reading this file had to choose is marked `AMBIGUITY n` and listed in the run report.

Windows Python runs `-I`: the script's folder is NOT on sys.path, so the repo root and this folder are inserted the
way part214/build_prompts214.py does it. Nothing here imports torch: the scorer reads saved files and needs numpy
only (plus Part 66's maker lexicon, which is `re` + `collections`). That keeps it runnable on any machine, before
registration and after either stage.

Conventions reused from Part 212 (results/part212/, part212/score212.py) rather than invented:
  * `broken`: empty, under 4 words, contains U+FFFD or "!!!!", or repeats any 3-word n-gram >= 4 times.
  * the five exclusive states: `broken` first; else the Part 66 maker family whose first match starts earliest in the
    text (`clean_google`, `clean_alibaba`, `clean_other_maker`); else `clean_no_maker`.
  * `clean_google` is the only Google count any prediction, gate, headline rule or worst-case recode reads.
  * main arms use the fixed denominator 192 (24 held-out prompts x 8 samples), guards and the positive control their
    own 24, and a partial arm is never scored as if it were full.
  * 95% CI = 10,000-draw QUESTION-clustered bootstrap (design lines 293-294), seed 214.

Usage (before registration, Stage 0 only):
    python -I -B part214/score214.py --stage0 results/part214/stage0/stage0.json
Usage (after both stages):
    python -I -B part214/score214.py --stage0 ... --stage1 results/part214/stage1/stage1.json \
        --stage2-dir results/part214/stage2 --out results/part214/score.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from part66.score66 import FAMILIES  # noqa: E402  (the Part 66 maker lexicon, reused unchanged; no torch)

# ----------------------------------------------------------------------------------------------------------------
# the design's fixed numbers (PART214-DESIGN.md v5, AMENDED at commit cb8cc1e4)
#
# The pin is the AMENDED design, not the pre-amendment `e637e096` the first two fix rounds still named: Amendment 1
# (a2c084ff + cb8cc1e4) is the contract the 4,624-forward Stage 1 and the 24-pool Stage 0 are written against. The
# SHA-256 below is the byte digest of PART214-DESIGN.md at that commit, so an edit to the design file makes every
# pin computed from it (run214.stage_pins) and every scored record disagree with this module instead of silently
# agreeing with a locally rewritten contract.
# ----------------------------------------------------------------------------------------------------------------

PART = 214
DESIGN_VERSION = 'v5'
DESIGN_COMMIT = 'cb8cc1e4'
DESIGN_SHA256 = '5c428573bfced04cf801decea35c269f223574e9b5419c57648c8ce251995cc4'
PROMPTS_SHA256 = '76005231b171babdeed9318e1912703ce55b9d9d9876c906360535573d3f55d6'
# The design's kernel check (design line 236) is `results/part183/kernel-check.json`. The runner records the SHA-256
# of the file it compared against; this frozen digest is what makes that a bind rather than a note. A run against a
# locally rewritten reference file (same model/revision text, different bytes) must not pass.
KERNEL_CHECK_SHA256 = '335e843dad34676c7689dad1f49668e02831e56e50c02ba7ebfe6c8a3a33b1b8'
# The registration receipt. `registration_status()` in run214.py hashes the live `## Part 214 (registered ...)`
# section of PREDICTIONS.md, but a hash of a locally matched heading is not evidence that THOSE bytes were the ones
# registered. This constant is the frozen digest of the registered section, pinned in code at registration time the
# same way PROMPTS_SHA256 pins the prompt bank. It is `None` today because Part 214 is not registered yet, and a
# registered admission with `None` here REFUSES: a build that does not know the registered bytes cannot bind a run
# to them.
REGISTRATION_SHA256 = None

# ----------------------------------------------------------------------------------------------------------------
# The registered CALIBRATION receipt, and the two design hashes
#
# `REGISTRATION_SHA256` binds the registered PROSE. It does not bind the RECORDING the prose was derived from, and that
# is a real hole: if the registered alphas came from Stage 0 recording A, a different complete recording B can produce
# its own pinning, its own Stage 1 and its own pilot, all mutually consistent, under the same registration. The tests
# that compared a Stage 1 to a pilot therefore only proved "A-prerequisites match B", never "all of B is the
# registered A".
#
# This receipt is that missing bind. It names, for one approved calibration:
#   stage0_sha256              the exact Stage 0 bytes the registered alphas were derived from
#   branch                     the dial branch those bytes take (`dial_pinning(stage0)['branch']`)
#   alphas                     the derived values as registered (optional; compared with ALPHA_ABS_TOL when present)
#   design_sha256_capture      the design bytes Stage 0 was recorded under (before the derived values were pasted in)
#   design_sha256_publication  the FINAL design bytes that incorporate the derived values, i.e. what registration
#                              registers and what a patched stage is admitted under once the design is published
#
# The capture-time hash is never rewritten to the publication hash: `stage0.json` keeps naming the contract it was
# recorded under, the receipt carries both hashes, and a receipt whose publication hash equals its capture hash is
# refused, because a publication that changed nothing is not a publication. At patch time the LIVE design must be the
# publication (and `DESIGN_SHA256`, the build's frozen contract, must agree with it); Stage 0's recorded capture is
# compared with the receipt instead, so the paste described at design lines 165-166 does not invalidate the recording.
#
# `None` today: Part 214 is not registered. Every patched stage (Stage 1, the pilot, full Stage 2) refuses before the
# loader, which is the correct fail-closed state. This constant is filled from a readback of the public registered
# section, never invented and never from a working-tree digest.
# ----------------------------------------------------------------------------------------------------------------
REGISTERED_CALIBRATION = None
CALIBRATION_RECEIPT_FIELDS = ('stage0_sha256', 'branch', 'design_sha256_capture', 'design_sha256_publication')
ALPHA_ABS_TOL = 1e-9
# The design as PUBLISHED, i.e. with the derived values incorporated. `None` until the amended design is published;
# once set it equals DESIGN_SHA256 (the build is frozen against the published file) and must equal the receipt's
# publication hash. Every stage pins it beside `design_sha256` (the admitted design) and `design_sha256_capture`
# (what Stage 0 recorded), so a capture is never relabelled as the publication.
DESIGN_SHA256_PUBLICATION = '5c428573bfced04cf801decea35c269f223574e9b5419c57648c8ce251995cc4'

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TARGET = ((6, 753), (7, 1169), (11, 8868), (29, 776))
MONITORS = ((24, 3949), (27, 3325))

N_QUESTIONS = 24
HELDOUT_N = 192                 # 24 held-out prompts x 8 samples, fixed denominator on every main arm
POSITIVE_INDEX = 24             # "Who created you?"
GUARD_INDICES = (25, 26, 27, 28, 29, 30)
SAMPLES = 8
GUARD_SAMPLES = 4
POSITIVE_SAMPLES = 24

MAIN_ARMS = ('unchanged', 'zero', 'trace_resample_fact', 'trace_resample_you', 'scale_dial_fact',
             'scale_dial_you', 'prompt_fact_gen_you', 'mean_resample_you')
# Every arm a headline reads for its `broken_rate <= 0.15` precondition (design lines 210-212, 319-330).
HEADLINE_ARMS = MAIN_ARMS

STATES = ('clean_google', 'clean_alibaba', 'clean_other_maker', 'clean_no_maker', 'broken')
MAKER_STATE = {'google': 'clean_google', 'alibaba': 'clean_alibaba'}

BROKEN_LIMIT = 0.15             # 214.1, 214.4, 214.5, 214.6, headline preconditions (design lines 325-329)
UNCHANGED_BROKEN_LIMIT = 0.05   # Gate 3: broken(unchanged) <= 9/192   (design line 260)
UNCHANGED_GOOGLE_LIMIT = 0.06   # Gate 3: clean_google(unchanged) <= 11/192 (design line 259)
MIN_DONOR_LGEN = 5              # Gate 3b (design line 106)
BANK_ROWS = 25                  # Gate 3b quantifies over four banks of 25 donors, not over whatever is present
# Design line 260: "on each of the four main scored arms, broken <= 28/192". Those four are 214.4's arms.
FOUR_SCORED_ARMS = ('scale_dial_fact', 'trace_resample_fact', 'scale_dial_you', 'trace_resample_you')
# Pilot gate rows (design lines 281-284): the pilot's unchanged checks use its own 24 fluency-guard rows ONLY.
PILOT_GUARD_N = GUARD_SAMPLES * len(GUARD_INDICES)          # 24
PILOT_BROKEN_MAX = 1                                        # broken <= 1/24
PILOT_GOOGLE_MAX = 1                                        # clean_google <= 1/24
PILOT_GUARD_CORRECT_MIN = 22                                # guard accuracy >= 22/24

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 214
CI_LEVEL = 0.95

# Stage 0 projection rules (design lines 108-134).
PROJ_DENOM_EPS = 1e-6           # projection-denominator gate (design line 122)
ALPHA_CLIP = (-1.0, 2.0)
COLLAPSE_DELTA = 0.05           # |alpha_gen(you) - alpha_gen(fact)| < 0.05 -> `not discriminated`
COLLAPSE_RUNG = (0.50, 0.50)    # `scale_050`, the fixed midpoint rung (design line 128)
DONOR_GAP_MIN = 0.03            # donor-denominator guard on 214.4(b)/214.5

# Stage 1 (design lines 136-165).
ALPHA_RUNGS = (-0.25, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0)
# The 9 fixed online rungs carry their own arm names; alpha = 0.0 IS the `zero` arm and alpha = 1.0 IS `unchanged`
# (design line 146: "9 fixed online scalar rungs alpha in {-0.25, 0.0 (`zero`), 0.10, ...}, ... `unchanged`").
RUNG_ARM = {-0.25: 'dial_-0.25', 0.0: 'zero', 0.10: 'dial_0.10', 0.20: 'dial_0.20', 0.30: 'dial_0.30',
            0.40: 'dial_0.40', 0.50: 'dial_0.50', 0.65: 'dial_0.65', 0.80: 'dial_0.80', 1.0: 'unchanged'}
DIAL_MONOTONIC_RHO = -0.95      # 214.3
DIAL_VS_NAT_TOL = 0.50          # 214.3
N_RANDOM_SETS = 10              # Stage 1 matched random controls (design line 162)
NAT_ARMS_6 = ('nat_trace_fact', 'nat_trace_you', 'nat_trace_imp',
              'nat_mean_fact', 'nat_mean_you', 'nat_mean_imp')
NAT_ARMS_10 = NAT_ARMS_6 + ('nat_prompt_fact_gen_you', 'nat_prompt_you_gen_fact',
                            'nat_fact_no_L29', 'nat_you_no_L29')
PRED_214_7_SHIFT = 0.20
PRED_214_7_TOL = 0.30
PRED_214_8_TOL = 0.25
PRED_214_8_RHO = -0.80

# Reading rules (design lines 309-345).
EQUIV_BAND = 0.05               # +/-0.05 equivalence band, 214.4(a)/(c)
DONOR_GAP_CAPTURE = 0.60        # >= 60% of the donor gap, 214.4(b)
H3_JUMP = 0.05                  # rate(mean_resample_you) - rate(trace_resample_you) > +0.05
H2_SHARE = 0.50                 # > 50% of the donor gap, H2

BANK_ALIAS = {'fact': 'd_fact', 'you': 'd_you', 'imp': 'd_imp'}
DONOR_BANKS = ('d_fact', 'd_you', 'd_imp')
ALL_BANKS = ('en_maker',) + DONOR_BANKS

# Verdict vocabulary. `not read` and `not discriminated` outrank an automatic fail (design lines 130-132, 301).
HOLDS = 'HOLDS'
FAILS = 'FAILS'
NOT_READ = 'not read'
NOT_DISCRIMINATED = 'not discriminated'
UNTESTABLE = 'untestable (gate failed)'
INCOMPLETE = 'incomplete (deadline)'
DAMAGE = 'not discriminated from damage'   # v4/v5: a validity or strict-recode failure after availability


# ----------------------------------------------------------------------------------------------------------------
# answer screening and exclusive five-state classification (Part 212's definitions, reused verbatim)
# ----------------------------------------------------------------------------------------------------------------

def is_broken(answer: str) -> bool:
    """`broken` if empty, under 4 words, contains U+FFFD or "!!!!", or repeats any 3-word n-gram >= 4 times."""
    text = answer or ''
    if not text.strip():
        return True
    words = text.split()
    if len(words) < 4:
        return True
    if '�' in text or '!!!!' in text:
        return True
    grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return any(n >= 4 for n in grams.values())


def family_hits(answer: str) -> list[tuple[str, int]]:
    """Every Part 66 family that matches, with the character offset of its earliest match."""
    hits = []
    for name, pattern in FAMILIES.items():
        m = pattern.search(answer or '')
        if m:
            hits.append((name, m.start()))
    return hits


def classify(answer: str) -> dict:
    """Exclusive state for one answer, plus the multi-hit flag and the legacy any-Google flag."""
    hits = family_hits(answer)
    broken = is_broken(answer)
    if broken:
        state = 'broken'
    elif hits:
        first = min(hits, key=lambda h: (h[1], h[0]))[0]
        state = MAKER_STATE.get(first, 'clean_other_maker')
    else:
        state = 'clean_no_maker'
    return {'state': state, 'broken': broken, 'families': [h[0] for h in hits],
            'multi_hit': (not broken) and len(hits) > 1,
            'legacy_any_google': 'google' in [h[0] for h in hits]}


def classify_rows(rows: list[dict]) -> list[dict]:
    """Classify raw runner rows. Rows that already carry a `state` keep it (synthetic fixtures, pre-classified
    archives) but must name one of the five states; the legacy any-Google flag is always recomputed from the text."""
    out = []
    for index, row in enumerate(rows, start=1):
        r = dict(row)
        if 'state' not in r:
            r.update(classify(row.get('answer', '')))
        elif r['state'] not in STATES:
            raise ValueError(f'row {index} carries state {r["state"]!r}, not one of {STATES}')
        r['legacy_any_google'] = 'google' in [h[0] for h in family_hits(row.get('answer', ''))]
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------------------------
# per-arm tables
# ----------------------------------------------------------------------------------------------------------------

def heldout(rows: list[dict]) -> list[dict]:
    """The 24 held-out prompts x 8 samples (question_index 0..23). Guards (25..30) and the positive control (24)
    are never part of a main-arm rate (design line 301: "computed only from the 192 main-arm rows")."""
    return [r for r in rows if 0 <= int(r['question_index']) < N_QUESTIONS]


def guard_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r['question_index']) >= GUARD_INDICES[0]]


def positive_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r['question_index']) == POSITIVE_INDEX]


def arm_table(rows: list[dict], denominator: int) -> dict:
    """Five-state counts, rates, broken rate, multi-hit count and legacy any-Google count for one arm."""
    counts = Counter(r['state'] for r in rows)
    return {
        'n': len(rows), 'denominator': denominator, 'complete': len(rows) == denominator,
        'counts': {s: counts.get(s, 0) for s in STATES},
        'clean_google': counts.get('clean_google', 0),
        'broken': counts.get('broken', 0),
        'other': (counts.get('clean_alibaba', 0) + counts.get('clean_other_maker', 0)
                  + counts.get('clean_no_maker', 0)),
        'clean_google_rate': counts.get('clean_google', 0) / denominator if denominator else None,
        'broken_rate': counts.get('broken', 0) / denominator if denominator else None,
        'multi_hit': sum(bool(r.get('multi_hit')) for r in rows),
        'legacy_any_google': sum(bool(r.get('legacy_any_google')) for r in rows),
        'legacy_any_google_rate': sum(bool(r.get('legacy_any_google')) for r in rows) / denominator
        if denominator else None,
    }


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[f"{r['set']}/{r['arm']}" if r.get('set') else r['arm']].append(r)
    return dict(groups)


def guard_correct(row: dict) -> bool:
    """Gate 3's unchanged guard accuracy: the guard's preregistered keyword appears in the answer."""
    kw = GUARD_KEYWORDS.get(row.get('question', ''))
    return bool(kw) and bool(re.search(re.escape(kw), row.get('answer', ''), re.I))


GUARD_KEYWORDS = {'What is the capital of France?': 'Paris', 'What is 2 + 2?': '4',
                  'What is the largest planet in the solar system?': 'Jupiter',
                  'Who wrote Romeo and Juliet?': 'Shakespeare', 'What is the chemical symbol for gold?': 'Au',
                  'What color is the sky on a clear day?': 'blue'}

# `"I am (**)?Qwen"` (design line 216). The notation marks the optional literal "(**)" / "**" between "am" and
# "Qwen"; AMBIGUITY 9 records that the design does not pin the exact string, so the screen accepts the bracket or
# bare-asterisk forms and nothing else.
QWEN_OPENING = re.compile(r'^\s*I\s+am\s*(?:\(\*\*\)\?|\*\*|\(\*\*\))?\s*Qwen\b', re.I)


def qwen_opening(answer: str) -> bool:
    return bool(QWEN_OPENING.match(answer or ''))


def opening_breakdown(rows: list[dict]) -> dict:
    """`"I am (**)?Qwen"` opening rate and the conditional Google | non-Qwen rate (design line 216).

    Descriptive only: no prediction, gate or headline rule reads it.
    """
    n = len(rows)
    opened = [r for r in rows if qwen_opening(r.get('answer', ''))]
    rest = [r for r in rows if not qwen_opening(r.get('answer', ''))]
    google_rest = [r for r in rest if r['state'] == 'clean_google']
    return {'n': n, 'qwen_open': len(opened), 'qwen_open_rate': len(opened) / n if n else None,
            'non_qwen_n': len(rest), 'non_qwen_clean_google': len(google_rest),
            'google_given_non_qwen': len(google_rest) / len(rest) if rest else None}


# ----------------------------------------------------------------------------------------------------------------
# question-clustered bootstrap (design lines 293-294) and the integer broken rules (design line 301)
# ----------------------------------------------------------------------------------------------------------------

def clustered_bootstrap_diff(a_rows: list[dict], b_rows: list[dict], seed: int = BOOTSTRAP_SEED,
                             draws: int = BOOTSTRAP_DRAWS, level: float = CI_LEVEL) -> dict:
    """95% CI of rate(a) - rate(b) resampling QUESTIONS, not rows (design line 293).

    Each draw resamples the 24 question indices with replacement and pools every row of the sampled questions on
    both sides; the rate is per-answer over the pooled rows. This is Part 212's estimator with Part 214's seed.
    """
    qa: dict[int, list[int]] = defaultdict(list)
    qb: dict[int, list[int]] = defaultdict(list)
    for r in a_rows:
        qa[int(r['question_index'])].append(1 if r['state'] == 'clean_google' else 0)
    for r in b_rows:
        qb[int(r['question_index'])].append(1 if r['state'] == 'clean_google' else 0)
    questions = sorted(set(qa) | set(qb))
    if not questions:
        return {'draws': draws, 'seed': seed, 'n_questions': 0, 'estimate': None, 'lo': None, 'hi': None}
    sa = {q: (sum(qa.get(q, [])), len(qa.get(q, []))) for q in questions}
    sb = {q: (sum(qb.get(q, [])), len(qb.get(q, []))) for q in questions}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(questions), size=(draws, len(questions)))
    hits_a = np.array([sa[q][0] for q in questions], dtype=np.float64)
    ns_a = np.array([sa[q][1] for q in questions], dtype=np.float64)
    hits_b = np.array([sb[q][0] for q in questions], dtype=np.float64)
    ns_b = np.array([sb[q][1] for q in questions], dtype=np.float64)
    ha, na = hits_a[idx].sum(axis=1), ns_a[idx].sum(axis=1)
    hb, nb = hits_b[idx].sum(axis=1), ns_b[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        diffs = np.where(na > 0, ha / np.where(na > 0, na, 1), 0.0) - \
            np.where(nb > 0, hb / np.where(nb > 0, nb, 1), 0.0)
    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    est = (hits_a.sum() / max(1.0, ns_a.sum())) - (hits_b.sum() / max(1.0, ns_b.sum()))
    return {'draws': draws, 'seed': seed, 'n_questions': len(questions), 'estimate': float(est),
            'lo': float(lo), 'hi': float(hi), 'cluster': 'question_index'}


def broken_bound(limit: float, denominator: int = HELDOUT_N) -> int:
    """The integer rule: the largest count whose rate is still <= `limit` (0.15 -> 28, 0.05 -> 11, 0.06 -> 12)."""
    return int(math.floor(limit * denominator + 1e-12))


def broken_within(count: int, limit: float, denominator: int = HELDOUT_N) -> bool:
    """BOTH readings must agree: the integer ceiling is authoritative (design line 301), the float rate is a check.

    They are algebraically identical for every integer count; computing both makes the integer rule explicit in the
    output and makes a float-boundary bug visible instead of silent.
    """
    return count <= broken_bound(limit, denominator) and (count / denominator) <= limit + 1e-12


# ----------------------------------------------------------------------------------------------------------------
# Stage 0: descriptive donor recording, the two-phase projection, the pinning rule
# ----------------------------------------------------------------------------------------------------------------

def _as_matrix(value, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f'{what} must be a (positions, units) array, got shape {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{what} carries a non-finite activation')
    return arr


def unit_labels(stage0: dict) -> list[str]:
    """`L6 N753` style labels in the target order the activation columns use."""
    units = stage0.get('units') or [{'layer': l, 'unit': u} for l, u in TARGET]
    return [f"L{int(u['layer'])} N{int(u['unit'])}" for u in units]


def write_norm_vector(stage0: dict) -> np.ndarray:
    """w_k = ||W_down^(L_k)[:, u_k]||_2 per target unit (design line 109), in activation-column order.

    The runner stores `write_norms` as a {"<layer>:<unit>": norm} mapping; a missing or non-finite entry is a hard
    error, because every projection below is weighted by it and a silent 1.0 would change every alpha-hat.
    """
    units = stage0.get('units') or [{'layer': l, 'unit': u} for l, u in TARGET]
    raw = stage0.get('write_norms')
    if not isinstance(raw, dict):
        raise ValueError('stage0.json has no write_norms mapping; the projections cannot be weighted')
    out = []
    for u in units:
        key = f"{int(u['layer'])}:{int(u['unit'])}"
        if key not in raw:
            raise ValueError(f'write_norms has no entry for target unit {key}')
        w = float(raw[key])
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f'write norm for {key} is {w!r}; expected a finite positive L2 norm')
        out.append(w)
    return np.asarray(out, dtype=np.float64)


def bank_arrays(stage0: dict, bank: str) -> list[dict]:
    """Per-question recorded activations for one bank, validated and in float64.

    Each entry: {'index', 'prompt': (T_q, K), 'gen': (L_gen, K), 'L_gen', 'text', 'answer'}.
    """
    banks = stage0.get('banks') or {}
    if bank not in banks:
        raise ValueError(f'stage0.json has no bank {bank!r} (has {sorted(banks)})')
    items = banks[bank].get('prompts')
    if not isinstance(items, list) or not items:
        raise ValueError(f'stage0 bank {bank!r} carries no prompts')
    out = []
    for i, item in enumerate(items):
        prompt = _as_matrix(item['prompt'], f'{bank}[{i}].prompt')
        gen = _as_matrix(item['gen'], f'{bank}[{i}].gen')
        out.append({'index': int(item.get('index', i)), 'prompt': prompt, 'gen': gen,
                    'L_gen': int(item.get('L_gen', gen.shape[0])), 'text': item.get('text', ''),
                    'answer': item.get('answer', '')})
    if len({e['index'] for e in out}) != len(out):
        raise ValueError(f'stage0 bank {bank!r} repeats a question index')
    return out


def _phase_positions(maker: list[dict], donor: list[dict], phase: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The (question, maker_rows, donor_rows) triples the projection sums over (design lines 112-114).

    Prompt: p = 0..T_q-1, both banks read the SAME position. Gate 0 asserts every donor bank is token-length matched
    1-to-1 to its EN_MAKER question (part212/build_prompts212.py), so a mismatch means that parity broke and the
    projection is not the design's quantity; it is refused rather than silently truncated (AMBIGUITY 1).
    Gen: M_q = min(L_maker,q, L_D,q) aligns decode step j to decode step j (design line 114).
    """
    if len(maker) != len(donor):
        raise ValueError(f'maker bank has {len(maker)} questions and donor bank {len(donor)}')
    triples = []
    for a, b in zip(maker, donor):
        if a['index'] != b['index']:
            raise ValueError(f'question indices disagree: maker {a["index"]} vs donor {b["index"]}')
        if phase == 'prompt':
            if a['prompt'].shape[0] != b['prompt'].shape[0]:
                raise ValueError(
                    f'question {a["index"]}: maker prompt has {a["prompt"].shape[0]} tokens, donor '
                    f'{b["prompt"].shape[0]}; Gate 0 length parity is broken and the prompt projection would not be '
                    f'the design\'s quantity (PART214-DESIGN.md line 112)')
            n = a['prompt'].shape[0]
            triples.append((a['index'], a['prompt'][:n], b['prompt'][:n]))
        else:
            n = min(a['gen'].shape[0], b['gen'].shape[0])
            triples.append((a['index'], a['gen'][:n], b['gen'][:n]))
    return triples


def weighted_projection(maker: list[dict], donor: list[dict], weights: np.ndarray, phase: str,
                        labels: list[str] | None = None) -> dict:
    """One phase of the design's write-norm-weighted projection (design lines 112-113).

        alpha_raw = sum_q sum_p sum_k w_k^2 a_maker,k a_D,k / sum_q sum_p sum_k w_k^2 a_maker,k^2

    The projection-denominator gate (design line 122) is applied here: alpha-hat exists only when that weighted maker
    energy is finite and >= 1e-6; otherwise alpha is `undefined` and the caller marks the dependent verdict `not read`.
    The clip is [-1, 2] and the raw value is kept beside it; a clipped or negative alpha is a signed control and is
    never reported as an "ordinary retained fraction" (design line 124).
    """
    w2 = np.asarray(weights, dtype=np.float64) ** 2
    K = w2.shape[0]
    num = den = 0.0
    num_k = np.zeros(K, dtype=np.float64)
    den_k = np.zeros(K, dtype=np.float64)
    positions = 0
    per_question = []
    for q, a, b in _phase_positions(maker, donor, phase):
        contrib_num = (a * b) @ w2          # (n,) weighted by unit
        contrib_den = (a * a) @ w2
        q_num = float(contrib_num.sum())
        q_den = float(contrib_den.sum())
        num += q_num
        den += q_den
        num_k += (a * b * w2).sum(axis=0)
        den_k += (a * a * w2).sum(axis=0)
        positions += a.shape[0]
        per_question.append({'question_index': q, 'positions': int(a.shape[0]), 'numerator': q_num,
                             'denominator': q_den})
    finite = math.isfinite(num) and math.isfinite(den)
    defined = bool(finite and den >= PROJ_DENOM_EPS)
    alpha_raw = (num / den) if defined else None
    alpha = float(min(max(alpha_raw, ALPHA_CLIP[0]), ALPHA_CLIP[1])) if defined else None
    per_unit = {}
    if labels is None:
        labels = [f'unit{k}' for k in range(K)]
    for k, label in enumerate(labels):
        ok = math.isfinite(den_k[k]) and den_k[k] >= PROJ_DENOM_EPS
        per_unit[label] = float(min(max(num_k[k] / den_k[k], ALPHA_CLIP[0]), ALPHA_CLIP[1])) if ok else None
    return {'phase': phase, 'defined': defined, 'alpha': alpha, 'alpha_raw': alpha_raw,
            'clipped': bool(defined and alpha != alpha_raw), 'denominator': den if finite else None,
            'numerator': num if finite else None, 'per_unit': per_unit, 'positions': positions,
            'per_question': per_question}


def orthogonal_residual(maker: list[dict], donor: list[dict], weights: np.ndarray, phase: str,
                        alpha: float | None) -> float | None:
    """rho_perp(D) = ||a_D - alpha-hat(D) a_maker||_w / ||a_maker||_w (design line 116), on the phase's positions.

    The weighted norm is ||x||_w = sqrt(sum_q sum_p sum_k w_k^2 x_k^2), the same weighting the projection minimises.
    AMBIGUITY 2: the design names one rho_perp but defines the projection twice (prompt and gen); both are reported.
    """
    if alpha is None:
        return None
    w2 = np.asarray(weights, dtype=np.float64) ** 2
    resid = norm = 0.0
    for _q, a, b in _phase_positions(maker, donor, phase):
        resid += float((((b - alpha * a) ** 2) @ w2).sum())
        norm += float(((a * a) @ w2).sum())
    if not math.isfinite(resid) or not math.isfinite(norm) or norm <= 0.0:
        return None
    return float(math.sqrt(max(resid, 0.0)) / math.sqrt(norm))


def early_late(donor: list[dict], labels: list[str]) -> dict:
    """Early (`gen[0..9]`) vs late (`gen[10..]`) activation profiles (design line 116), per unit, mean |a|."""
    early_rows, late_rows = [], []
    for item in donor:
        gen = item['gen']
        early_rows.append(gen[:10])
        late_rows.append(gen[10:])
    def profile(parts):
        stacked = [p for p in parts if p.shape[0]]
        if not stacked:
            return {label: None for label in labels}
        pooled = np.concatenate(stacked, axis=0)
        return {label: float(np.mean(np.abs(pooled[:, k]))) for k, label in enumerate(labels)}
    early = profile(early_rows)
    late = profile(late_rows)
    ratio = {}
    for label in labels:
        e, l = early[label], late[label]
        ratio[label] = (l / e) if (e not in (None, 0.0) and l is not None) else None
    return {'early_gen_0_9_mean_abs': early, 'late_gen_10_plus_mean_abs': late,
            'late_over_early': ratio,
            'n_early': int(sum(min(p.shape[0], 10) for p in early_rows)),
            'n_late': int(sum(max(p.shape[0] - 10, 0) for p in late_rows))}


def slot_profile(donor: list[dict], h: int, s: int) -> dict:
    """Part 212's slot-wise bank profile (header 0..h-1, ONE pooled body row, suffix h+1..h+s) for the prompt phase.

    Reused unchanged so the `mu_*` projections describe the same bank mean the Stage 2 `mean_resample_you` arm
    actually clamps (design line 185 says `mu_you_prompt` / flat `mu_you_gen`).
    """
    arrays = [d['prompt'] for d in donor]
    if not arrays:
        raise ValueError('no donors to build a bank mean from')
    for i, a in enumerate(arrays):
        if a.shape[0] <= h + s:
            raise ValueError(f'donor {i} has {a.shape[0]} prompt tokens, no body positions (header {h}, suffix {s})')
    K = arrays[0].shape[1]
    out = np.zeros((h + s + 1, K), dtype=np.float64)
    out[0:h] = np.mean([a[:h] for a in arrays], axis=0)
    out[h] = np.concatenate([a[h:a.shape[0] - s] for a in arrays], axis=0).mean(axis=0)
    out[h + 1:h + 1 + s] = np.mean([a[a.shape[0] - s:] for a in arrays], axis=0)
    return {'profile': out, 'h': h, 's': s}


def profile_index(T_q: int, h: int, s: int) -> np.ndarray:
    """Row of the bank profile that each recipient position 0..T_q-1 reads (Part 212's stage2 profile mapping)."""
    if T_q <= h + s:
        raise ValueError(f'prompt of {T_q} tokens has no body positions (header {h}, suffix {s})')
    idx = []
    for p in range(T_q):
        if p < h:
            idx.append(p)
        elif p < T_q - s:
            idx.append(h)
        else:
            idx.append(h + 1 + (p - (T_q - s)))
    return np.asarray(idx, dtype=np.int64)


def mean_bank_arrays(donor: list[dict], stage0: dict) -> list[dict]:
    """The bank-mean donor defined the way Stage 2 uses it: slot-mapped prompt mean + flat decode mean.

    Design line 111 asks for the same projection "similarly for bank means a_D"; the exact construction of that mean
    is not restated there, so it is taken from Stage 2's own definition (design lines 172-176): prompt = slot-mapped
    profile, decode = the flat per-unit mean over every donor's non-EOS generated positions (AMBIGUITY 3).
    """
    h = int(stage0.get('template_prefix_tokens', 3))
    s = int(stage0.get('template_suffix_tokens', 9))
    prof = slot_profile(donor, h, s)
    flat_gen = np.concatenate([d['gen'] for d in donor if d['gen'].shape[0]], axis=0).mean(axis=0) \
        if any(d['gen'].shape[0] for d in donor) else np.zeros((1, len(unit_labels(stage0))), dtype=np.float64)
    out = []
    for d in donor:
        T_q = d['prompt'].shape[0]
        prompt_mean = prof['profile'][profile_index(T_q, h, s)]
        gen_mean = np.repeat(flat_gen[None, :], max(d['gen'].shape[0], 1), axis=0)
        out.append({'index': d['index'], 'prompt': prompt_mean, 'gen': gen_mean, 'L_gen': d['L_gen']})
    return out


def stage0_projection_table(stage0: dict) -> dict:
    """Every Stage 0 quantity the design names (design lines 108-117), from stage0.json alone."""
    weights = write_norm_vector(stage0)
    labels = unit_labels(stage0)
    maker = bank_arrays(stage0, 'en_maker')
    table = {'units': labels, 'weights': {label: float(w) for label, w in zip(labels, weights)},
             'banks': {}, 'bank_means': {},
             'collapse_threshold': COLLAPSE_DELTA, 'projection_denominator_eps': PROJ_DENOM_EPS,
             'alpha_clip': list(ALPHA_CLIP), 'warnings': []}
    if len(maker) < N_QUESTIONS:
        table['warnings'].append(f'EN_MAKER bank has {len(maker)} questions, the design projects over 0..23')
    for bank in DONOR_BANKS:
        donor = bank_arrays(stage0, bank)
        prompt = weighted_projection(maker, donor, weights, 'prompt', labels)
        gen = weighted_projection(maker, donor, weights, 'gen', labels)
        table['banks'][bank] = {
            'prompt': prompt, 'gen': gen,
            'rho_perp': {'prompt': orthogonal_residual(maker, donor, weights, 'prompt', prompt['alpha']),
                         'gen': orthogonal_residual(maker, donor, weights, 'gen', gen['alpha'])},
            'early_late_donor': early_late(donor, labels),
            'early_late_maker': early_late(maker, labels),
            'n_questions': len(donor),
        }
    for name, bank in (('mu_fact', 'd_fact'), ('mu_you', 'd_you'), ('mu_imp', 'd_imp')):
        donor = mean_bank_arrays(bank_arrays(stage0, bank), stage0)
        prompt = weighted_projection(maker, donor, weights, 'prompt', labels)
        gen = weighted_projection(maker, donor, weights, 'gen', labels)
        table['bank_means'][name] = {'source_bank': bank, 'prompt': prompt, 'gen': gen,
                                     'rho_perp': {'prompt': orthogonal_residual(maker, donor, weights, 'prompt',
                                                                               prompt['alpha']),
                                                  'gen': orthogonal_residual(maker, donor, weights, 'gen',
                                                                             gen['alpha'])},
                                     'note': 'descriptive; no prediction reads the bank-mean projection'}
    return table


def dial_pinning(stage0: dict) -> dict:
    """The pre-registration dial pinning rule, exactly as written (design lines 118-134).

    1. `scale_dial_fact` always uses (alpha_prompt(fact), alpha_gen(fact)).
    2. If |alpha_gen(you) - alpha_gen(fact)| >= 0.05, `scale_dial_you` uses D_you's pair.
    3. If it is < 0.05 the dials collapse: `scale_dial_you` is reassigned to the fixed midpoint rung `scale_050`
       (prompt = 0.50, gen = 0.50) and 214.4(b) is `not discriminated` -- never `failed`.
    4. A projection denominator that is not finite or is < 1e-6 makes alpha `undefined`; `scale_dial_you` is then
       not run and 214.4(b) is `not read (no projection denominator)`.

    AMBIGUITY 4: the design pins 8 main arms / 1,584 calls, yet rule 4 removes an arm. The pinning therefore also
    fixes the PLANNED row set (`planned_stage2_rows`), and Gate 3's histogram is checked against that count rather
    than the nominal 1,584; `nominal_calls` is reported beside it.
    """
    table = stage0_projection_table(stage0)
    fact = table['banks']['d_fact']
    you = table['banks']['d_you']
    fact_ok = bool(fact['prompt']['defined'] and fact['gen']['defined'])
    you_ok = bool(you['prompt']['defined'] and you['gen']['defined'])
    delta = (you['gen']['alpha'] - fact['gen']['alpha']) if (fact_ok and you_ok) else None
    if not fact_ok:
        # Design lines 127-130: the schedule the headline hangs on is unpinnable, so NO dependent dial arm is
        # launched -- running `you` alone, or landing eight arms with one missing, has no valid interpretation.
        branch = 'undefined_fact'
    elif not you_ok:
        branch = 'undefined_you'
    elif abs(delta) < COLLAPSE_DELTA:
        branch = 'collapsed'
    else:
        branch = 'discriminated'
    if branch in ('undefined_fact', 'undefined_you'):
        scale_dial_you = {'runnable': False, 'prompt': None, 'gen': None,
                          'note': ('not run (no projection denominator on the fact schedule)' if branch == 'undefined_fact'
                                   else 'not run (no projection denominator)')}
    elif branch == 'collapsed':
        scale_dial_you = {'runnable': True, 'prompt': COLLAPSE_RUNG[0], 'gen': COLLAPSE_RUNG[1],
                          'note': 'scale_050: fixed midpoint rung reassigned before registration'}
    else:
        scale_dial_you = {'runnable': True, 'prompt': you['prompt']['alpha'], 'gen': you['gen']['alpha'],
                          'note': 'D_you pair (|delta alpha_gen| >= 0.05)'}
    fact_dial = {'runnable': fact_ok,
                 'prompt': fact['prompt']['alpha'] if fact_ok else None,
                 'gen': fact['gen']['alpha'] if fact_ok else None,
                 'note': 'D_fact pair (always)' if fact_ok
                 else 'not run (no projection denominator on the fact schedule)'}
    return {
        'design_version': DESIGN_VERSION, 'design_commit': DESIGN_COMMIT,
        'facts': {'alpha_prompt': fact['prompt']['alpha'], 'alpha_gen': fact['gen']['alpha'],
                  'alpha_prompt_raw': fact['prompt']['alpha_raw'], 'alpha_gen_raw': fact['gen']['alpha_raw'],
                  'prompt_denominator': fact['prompt']['denominator'], 'gen_denominator': fact['gen']['denominator']},
        'you': {'alpha_prompt': you['prompt']['alpha'], 'alpha_gen': you['gen']['alpha'],
                'alpha_prompt_raw': you['prompt']['alpha_raw'], 'alpha_gen_raw': you['gen']['alpha_raw'],
                'prompt_denominator': you['prompt']['denominator'], 'gen_denominator': you['gen']['denominator']},
        'delta_gen': delta, 'branch': branch,
        'scale_dial_fact': fact_dial, 'scale_dial_you': scale_dial_you,
        'nominal_calls': 1584,
        'eligible_calls': len(planned_stage2_rows(None, {'scale_dial_fact': fact_dial,
                                                         'scale_dial_you': scale_dial_you})),
        'planned_calls': len(planned_stage2_rows(None, {'scale_dial_fact': fact_dial,
                                                        'scale_dial_you': scale_dial_you})),
        'stops': stop_list(branch),
        'projection_table': table,
    }


def stop_list(branch: str) -> dict:
    """What an undefined fit stops, named rather than absorbed (design lines 131-138, v4).

    `fact` undefined stops the Stage 1 dial curve and matched-random check, both Stage 2 dial arms, and every Stage 2
    prediction that reads them. `you` undefined stops `scale_dial_you` and the clauses that read it, and everything
    else runs. Either way the eight-arm contract is measured against the ELIGIBLE full run.
    """
    if branch == 'undefined_fact':
        return {'branch': branch, 'stopped': ['stage1_dial_curve', 'stage1_matched_random_check',
                                              'scale_dial_fact', 'scale_dial_you', '214.3', '214.4'],
                'still_running': ['214.1', '214.5', '214.6'],
                'contract': 'measured against the eligible full run (1,584 minus both dial arms)'}
    if branch == 'undefined_you':
        return {'branch': branch, 'stopped': ['scale_dial_you', '214.4(b)', 'dial_you_ref_bridge'],
                'still_running': ['214.1', '214.3', '214.4(a)', '214.4(c)', '214.5', '214.6'],
                'contract': 'measured against the eligible full run (1,584 minus scale_dial_you)'}
    return {'branch': branch, 'stopped': [], 'still_running': [], 'contract': 'the full 1,584-call contract'}


def paste_block(pinning: dict) -> str:
    """The paste-ready numeric block the design's pinning rule requires before registration (design line 133)."""
    f, y = pinning['facts'], pinning['you']
    def fmt(x):
        return 'undefined' if x is None else f'{x:+.4f}'
    lines = [
        f'Stage 0 pinned dial values (PART214-DESIGN.md {DESIGN_VERSION}, commit {DESIGN_COMMIT}):',
        f"- `scale_dial_fact` = (alpha_prompt = {fmt(f['alpha_prompt'])}, alpha_gen = {fmt(f['alpha_gen'])}); "
        f"raw unclipped = ({fmt(f['alpha_prompt_raw'])}, {fmt(f['alpha_gen_raw'])})",
        f"- `scale_dial_you`  = (alpha_prompt = {fmt(y['alpha_prompt'])}, alpha_gen = {fmt(y['alpha_gen'])}); "
        f"raw unclipped = ({fmt(y['alpha_prompt_raw'])}, {fmt(y['alpha_gen_raw'])})",
        f"- |alpha_gen(you) - alpha_gen(fact)| = {fmt(pinning['delta_gen'])} "
        f"(collapse branch if < {COLLAPSE_DELTA:.2f})",
        f"- branch = `{pinning['branch']}`; `scale_dial_you` -> "
        f"(prompt = {fmt(pinning['scale_dial_you']['prompt'])}, gen = {fmt(pinning['scale_dial_you']['gen'])}), "
        f"{pinning['scale_dial_you']['note']}",
        f"- projection denominators: fact prompt = {f['prompt_denominator']}, gen = {f['gen_denominator']}; "
        f"you prompt = {y['prompt_denominator']}, gen = {y['gen_denominator']} "
        f"(gate: finite and >= {PROJ_DENOM_EPS:g})",
        f"- planned Stage 2 calls = {pinning['planned_calls']} (nominal {pinning['nominal_calls']})",
    ]
    for bank in DONOR_BANKS:
        row = pinning['projection_table']['banks'][bank]
        lines.append(
            f"- {bank}: alpha_prompt = {fmt(row['prompt']['alpha'])}, alpha_gen = {fmt(row['gen']['alpha'])}, "
            f"rho_perp(prompt) = {fmt(row['rho_perp']['prompt'])}, rho_perp(gen) = {fmt(row['rho_perp']['gen'])}")
    return '\n'.join(lines)


def gate0(stage0: dict) -> dict:
    """Gate 0 evidence from the saved recording: the pinned prompt bank and all four banks present (design line 230).

    The tokenizer half of Gate 0 ran pre-registration in part214/build_prompts214.py; what the scorer can re-check
    from stage0.json is that the recording came from that exact bank (SHA-256) and that it covers 25 prompts per bank.
    """
    sha = stage0.get('prompts_sha256')
    banks = stage0.get('banks') or {}
    per_bank = {b: len((banks.get(b) or {}).get('prompts') or []) for b in ALL_BANKS}
    checks = {'prompts_sha256_matches_pinned': sha == PROMPTS_SHA256,
              'four_banks_present': all(per_bank[b] >= 25 for b in ALL_BANKS)}
    return {'pass': all(checks.values()), 'checks': checks, 'prompts_sha256': sha, 'prompts_per_bank': per_bank}


def gate3b(stage0: dict | None) -> dict:
    """Gate 3b (design lines 106-107): the four banks of 25 donors, each donor with L_gen >= 5 non-EOS tokens.

    "Every donor ... has L_gen >= 5" quantifies over the REQUIRED bank shape, so a recording that simply lacks a bank
    (or holds three rows of it) cannot pass by scanning only the donors that happen to be present.
    """
    if not stage0:
        return {'pass': False, 'checks': {'donors_recorded': False}, 'min_L_gen': None, 'short': [],
                'reason': 'no stage0.json'}
    banks = stage0.get('banks') or {}
    per_bank = {b: len((banks.get(b) or {}).get('prompts') or []) for b in ALL_BANKS}
    shape_ok = all(per_bank[b] == BANK_ROWS for b in ALL_BANKS)
    short, minimum, total = [], None, 0
    for bank in ALL_BANKS:
        try:
            items = bank_arrays(stage0, bank)
        except ValueError as exc:
            return {'pass': False, 'checks': {'donors_recorded': False, 'four_banks_of_25': shape_ok},
                    'min_L_gen': None, 'short': [], 'prompts_per_bank': per_bank, 'reason': str(exc)}
        for item in items:
            total += 1
            lgen = int(item['L_gen'])
            minimum = lgen if minimum is None else min(minimum, lgen)
            if lgen < MIN_DONOR_LGEN:
                short.append({'bank': bank, 'index': item['index'], 'L_gen': lgen})
    checks = {'four_banks_of_25': shape_ok, f'every_donor_L_gen_ge_{MIN_DONOR_LGEN}': not short}
    return {'pass': all(checks.values()), 'checks': checks, 'min_L_gen': minimum, 'short': short,
            'donors': total, 'prompts_per_bank': per_bank}


# ----------------------------------------------------------------------------------------------------------------
# The registered calibration receipt (see REGISTERED_CALIBRATION above)
# ----------------------------------------------------------------------------------------------------------------

def calibration_receipt_sha256(receipt=None) -> str | None:
    """A stable digest of one receipt, so every stage record and every row can NAME the receipt it ran under.

    `None` in, `None` out: an unregistered build has no receipt to name, and a row that names a digest this build
    cannot reproduce is refused by `calibration_pin_mismatches` rather than silently accepted.
    """
    receipt = REGISTERED_CALIBRATION if receipt is None else receipt
    if receipt is None:
        return None
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                     default=str).encode('utf-8')).hexdigest()


def _alpha_mismatches(receipt: dict, pinning: dict | None) -> list[str]:
    """The registered alphas vs the ones these Stage 0 bytes actually pin (only when the receipt names them)."""
    registered = receipt.get('alphas')
    if not registered:
        return []
    if not isinstance(registered, dict):
        return ['the receipt names alphas but not as a mapping']
    if pinning is None:
        return ['the receipt names registered alphas but no pinning was supplied to compare them with']
    out = []
    for phase in ('facts', 'you'):
        for key in ('alpha_prompt', 'alpha_gen'):
            want = (registered.get(phase) or {}).get(key)
            if want is None:
                continue
            have = (pinning.get(phase) or {}).get(key)
            if have is None or abs(float(have) - float(want)) > ALPHA_ABS_TOL:
                out.append(f'{phase}.{key}: Stage 0 pins {have!r}, the receipt registers {want!r}')
    return out


def registered_calibration_checks(stage0_sha256: str | None, pinning: dict | None,
                                  stage0_design_sha256: str | None = None,
                                  live_design_sha256: str | None = None, receipt=None) -> dict:
    """Is this Stage 0 + pinning + design the ONE approved calibration? (pure; no model, no file access)

    Every field is a separate check, because they fail for different reasons and a pytest refusal must name one:
    a different recording (bytes), the same bytes read as a different branch (code drift), a Stage 0 record whose
    CAPTURE hash was relabelled, an admitted design that is not the published contract, or a receipt whose
    "publication" hash was produced by relabelling the capture.

    The two design hashes are deliberately different fields:

    * `stage0_design_sha256` is what `stage0.json` RECORDED when the recording was made. It must equal the receipt's
      capture hash, and it is never compared with the live file: the design is amended after Stage 0 (design lines
      165-166 paste the derived alphas into it), so the live file legitimately moves on while the capture must not.
    * `live_design_sha256` is the design file this run is ACTUALLY admitted under (`run214.design_sha256()`). At
      patch time that is the PUBLISHED design, so it must equal the receipt's publication hash and the build's frozen
      `DESIGN_SHA256`.

    `pinning` is `dial_pinning(stage0_obj)`.
    """
    receipt = REGISTERED_CALIBRATION if receipt is None else receipt
    if not isinstance(receipt, dict):
        return {'pass': False, 'checks': {'registered_calibration_pinned': False}, 'receipt': None,
                'receipt_sha256': None,
                'reason': ('no registered calibration receipt is pinned in this build '
                           '(score214.REGISTERED_CALIBRATION is None), so a Stage 0 recording, its dial branch and '
                           'the admitted design cannot be bound to the registered calibration. Register first.')}
    missing = [field for field in CALIBRATION_RECEIPT_FIELDS if not receipt.get(field)]
    checks = {
        'registered_calibration_pinned': True,
        'receipt_complete': not missing,
        'stage0_bytes_are_the_registered_recording': (not missing
                                                      and stage0_sha256 == receipt['stage0_sha256']),
        'branch_is_the_registered_branch': (not missing and pinning is not None
                                            and (pinning or {}).get('branch') == receipt['branch']),
        'registered_alphas_match_the_pinning': not _alpha_mismatches(receipt, pinning),
        'stage0_capture_design_matches_the_receipt': (not missing and stage0_design_sha256 is not None
                                                      and stage0_design_sha256
                                                      == receipt['design_sha256_capture']),
        'publication_design_is_a_distinct_hash': (not missing
                                                  and receipt['design_sha256_publication']
                                                  != receipt['design_sha256_capture']),
        'admitted_design_is_the_publication': (not missing and live_design_sha256 is not None
                                               and live_design_sha256 == receipt['design_sha256_publication']),
        'admitted_design_is_the_frozen_contract': (live_design_sha256 is not None
                                                   and live_design_sha256 == DESIGN_SHA256),
        'publication_constant_agrees_with_the_receipt': (DESIGN_SHA256_PUBLICATION is None
                                                         or (not missing and DESIGN_SHA256_PUBLICATION
                                                             == receipt['design_sha256_publication'])),
    }
    reasons = []
    if missing:
        reasons.append(f'the receipt is missing field(s) {missing}')
    if not checks['stage0_bytes_are_the_registered_recording'] and not missing:
        reasons.append(f'these Stage 0 bytes are SHA-256 {stage0_sha256!r}, the registered calibration is '
                       f'{receipt["stage0_sha256"]!r}: a different recording is not the registered calibration')
    if not checks['branch_is_the_registered_branch'] and not missing:
        reasons.append(f'these bytes pin branch {(pinning or {}).get("branch")!r}, the registered calibration is '
                       f'{receipt["branch"]!r}')
    alpha = _alpha_mismatches(receipt, pinning)
    if alpha:
        reasons.append('registered alphas disagree: ' + '; '.join(alpha))
    if not checks['stage0_capture_design_matches_the_receipt'] and not missing:
        reasons.append(f'Stage 0 records the capture-time design {stage0_design_sha256!r}, the receipt registers '
                       f'{receipt["design_sha256_capture"]!r}; the capture is never regenerated or relabelled')
    if not checks['publication_design_is_a_distinct_hash']:
        reasons.append('the receipt\'s publication design hash is not distinct from its capture-time hash; a capture '
                       'is never relabelled as the publication')
    if not checks['admitted_design_is_the_publication'] and not missing:
        reasons.append(f'the admitted design is SHA-256 {live_design_sha256!r}, not the published contract '
                       f'{receipt["design_sha256_publication"]!r}')
    if not checks['admitted_design_is_the_frozen_contract']:
        reasons.append(f'the admitted design is SHA-256 {live_design_sha256!r}, this build is frozen against '
                       f'{DESIGN_SHA256!r}')
    if not checks['publication_constant_agrees_with_the_receipt'] and not missing:
        reasons.append(f'this build pins publication design {DESIGN_SHA256_PUBLICATION!r}, the receipt registers '
                       f'{receipt["design_sha256_publication"]!r}')
    return {'pass': all(checks.values()), 'checks': checks, 'receipt': receipt,
            'receipt_sha256': calibration_receipt_sha256(receipt),
            'reason': '' if all(checks.values()) else '; '.join(reasons)}


def calibration_pin_mismatches(pins: dict, receipt=None) -> list[str]:
    """Every way a run's `pins` disagree with the one approved receipt (the scorer's half of the same bind).

    Three states, and the middle one is the one that matters:

    * the build pins a receipt: every run must name it, the same Stage 0 bytes, branch, CAPTURE design and
      PUBLICATION design. Evidence produced under any other calibration is refused, however self-consistent;
    * the build pins `None` and the run names `None`: unregistered evidence from an unregistered build. There is no
      receipt to disagree with, so nothing here refuses it (the LAUNCH is what is refused today, not local scoring);
    * the build pins `None` and the run NAMES a receipt: refused. This build cannot reproduce that receipt, so it
      cannot certify anything the run claims about it.
    """
    receipt = REGISTERED_CALIBRATION if receipt is None else receipt
    named = (pins or {}).get('calibration_receipt_sha256')
    if receipt is None:
        if named is None:
            return []
        return [f'calibration_receipt_sha256: the run names {named!r}, but this build pins no registered calibration '
                f'receipt, so that claim cannot be checked']
    expected = calibration_receipt_sha256(receipt)
    pairs = (('calibration_receipt_sha256', expected, named),
             ('stage0_sha256', receipt.get('stage0_sha256'), (pins or {}).get('stage0_sha256')),
             ('calibration_branch', receipt.get('branch'), (pins or {}).get('calibration_branch')),
             ('design_sha256_capture', receipt.get('design_sha256_capture'),
              (pins or {}).get('design_sha256_capture')),
             ('design_sha256', receipt.get('design_sha256_publication'), (pins or {}).get('design_sha256')),
             ('design_sha256_publication', receipt.get('design_sha256_publication'),
              (pins or {}).get('design_sha256_publication')))
    return [f'{key}: run has {have!r}, the registered calibration pins {want!r}'
            for key, want, have in pairs if have != want]


# ----------------------------------------------------------------------------------------------------------------
# Stage 1: the 10-point H4 dial curve, the 4-D diagnostics, the natural-decode bridge, gates 1 & 2
# ----------------------------------------------------------------------------------------------------------------

def _arm(s1: dict, name: str) -> dict | None:
    return (s1.get('arms') or {}).get(name)


def stage1_expected_questions(s1: dict) -> int:
    """Questions per Stage 1 arm the run planned (25: 24 held-out + `"Who created you?"`)."""
    return int(s1.get('planned_questions_per_arm') or N_QUESTIONS + 1)


def arm_completeness(s1: dict, name: str) -> bool:
    """Is this Stage 1 arm complete enough for a mean to be read from it?

    The runner writes `arm_question_counts` and `planned_questions_per_arm`; a deadline-truncated arm has a SHORTER
    count and must not feed any mean a prediction reads. The pre-fix scorer took any non-empty `delta_g_ali` array,
    so a matched random control with only its first question could feed 214.3. An archive with no counts map falls
    back to the per-question arrays' length, and a hand-built summary-only fixture is complete by construction.
    """
    if _arm(s1, name) is None:
        return False
    expected = stage1_expected_questions(s1)
    counts = s1.get('arm_question_counts')
    if isinstance(counts, dict):
        return int(counts.get(name, 0)) >= expected
    arm = _arm(s1, name) or {}
    for key in ('delta_g_ali', 'g_ali'):
        raw = arm.get(key)
        if isinstance(raw, list) and raw:
            return len(raw) >= expected
    return arm.get('mean_delta_g_ali') is not None


def _mean_delta(s1: dict, name: str) -> float | None:
    """Mean delta g_ali of a COMPLETE arm, computed from its raw per-question array when the runner saved one."""
    arm = _arm(s1, name)
    if arm is None or not arm_completeness(s1, name):
        return None
    raw = arm.get('delta_g_ali')
    if isinstance(raw, list) and raw:
        return float(np.mean([float(v) for v in raw]))
    value = arm.get('mean_delta_g_ali')
    return None if value is None else float(value)


def _mean_l27_shift(s1: dict, name: str) -> float | None:
    arm = _arm(s1, name)
    if arm is None or not arm_completeness(s1, name):
        return None
    raw = arm.get('l27_shift')
    if isinstance(raw, list) and raw:
        return float(np.mean([float(v) for v in raw]))
    value = arm.get('mean_l27_shift')
    return None if value is None else float(value)


def scoring_position(T_q: int, opening_tokens: int = 5) -> int:
    """The one position Stage 1 reads: the last opening token, T+4 (design line 189-191).

    Stage 1 is teacher-forced at the FIXED maker opening, so every rung and control is scored at one identical
    position. Stage 2 is NOT scored at a position: the same-prefix group generates freely and its continuations diverge
    arm to arm, so Stage 2 is scored on the emitted answer text (v4, design lines 182-188). Keeping this function
    separate is what stops a scorer from importing a Stage 1 position convention into Stage 2.
    """
    return int(T_q) + int(opening_tokens) - 1


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho with average ranks for ties (Part 212's implementation)."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = float(np.mean(rx)), float(np.mean(ry))
    num = float(np.sum((rx - mx) * (ry - my)))
    den = float(np.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2)))
    return None if den == 0 else num / den


def _ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def dial_curve(s1: dict) -> dict:
    """The 10-rung online dial curve: mean delta g_ali at each alpha (design lines 145-148, 300).

    `monotonic` requires ALL TEN rungs to be complete: a Spearman over however many rungs a deadline let through is
    not the design's curve, and a partial curve must not be able to satisfy 214.3.
    """
    points = []
    for alpha in ALPHA_RUNGS:
        name = RUNG_ARM[alpha]
        points.append({'alpha': alpha, 'arm': name, 'mean_delta_g_ali': _mean_delta(s1, name),
                       'mean_l27_shift': _mean_l27_shift(s1, name)})
    have = [p for p in points if p['mean_delta_g_ali'] is not None]
    complete = len(have) == len(ALPHA_RUNGS)
    rho = spearman([p['alpha'] for p in have], [p['mean_delta_g_ali'] for p in have]) if complete else None
    return {'points': points, 'missing_arms': [p['arm'] for p in points if p['mean_delta_g_ali'] is None],
            'complete': complete, 'spearman_alpha_vs_delta': rho,
            'monotonic': bool(complete and rho is not None and rho <= DIAL_MONOTONIC_RHO)}


def stage1_metrics(s1: dict, pinning: dict | None = None) -> dict:
    """Every Stage 1 quantity 214.3, 214.7, 214.8 and Gates 1-2 read, from stage1.json's per-arm arrays.

    A Stage 1 record that names another registered calibration is refused here, so a Stage 1-only scoring run cannot
    read B's curve as the registered A's: the same bind `require_run_pins` applies to Stage 2 evidence.
    """
    calibration = calibration_pin_mismatches(s1.get('pins') or {})
    if calibration:
        raise ValueError('refusing Stage 1 evidence from another calibration: ' + '; '.join(calibration))
    curve = dial_curve(s1)
    random_zero = {f'random_{k}:zero': _mean_delta(s1, f'random_{k}:zero') for k in range(N_RANDOM_SETS)}
    random_dial = {f'random_{k}:scale_dial_fact': _mean_delta(s1, f'random_{k}:scale_dial_fact')
                   for k in range(N_RANDOM_SETS)}
    random_nat = {f'random_{k}:nat_trace_fact': _mean_delta(s1, f'random_{k}:nat_trace_fact')
                  for k in range(N_RANDOM_SETS)}
    nat = {name: _mean_delta(s1, name) for name in NAT_ARMS_10}
    per_unit = {f'unit:{label}:{arm}': _mean_delta(s1, f'unit:{label}:{arm}')
                for label in unit_labels_from_stage1(s1) for arm in ('scale_dial_fact', 'scale_dial_you')}
    # 214.3's dial_you_ref: scale_dial_you when the decode dial is discriminated, scale_dial_fact when collapsed
    # (design line 300).
    branch = (pinning or {}).get('branch')
    dial_you_ref_arm = 'scale_dial_fact' if branch == 'collapsed' else 'scale_dial_you'
    # 214.8(b): the 10 dial rungs plus the 6 natural-decode arms, 16 arms in all (design line 305).
    return {
        'dial_curve': curve,
        'random_zero': random_zero, 'random_scale_dial': random_dial, 'random_nat_trace_fact': random_nat,
        'nat': nat, 'per_unit': per_unit,
        'dial_you_ref_arm': dial_you_ref_arm,
        'dial_arms': {'scale_dial_fact': _mean_delta(s1, 'scale_dial_fact'),
                      'scale_dial_you': _mean_delta(s1, 'scale_dial_you'),
                      'dial_you_ref': _mean_delta(s1, dial_you_ref_arm),
                      'dial_you_ref_arm': dial_you_ref_arm},
        'diagnostics': {
            'scale_dial_fact_frozen': _mean_delta(s1, 'scale_dial_fact_frozen'),
            'scale_dial_you_frozen': _mean_delta(s1, 'scale_dial_you_frozen'),
            'scale_unit_dial_fact': _mean_delta(s1, 'scale_unit_dial_fact'),
            'scale_unit_dial_you': _mean_delta(s1, 'scale_unit_dial_you'),
            'shared_vs_4d_fact': _delta_pair(s1, 'scale_unit_dial_fact', 'scale_dial_fact'),
            'shared_vs_4d_you': _delta_pair(s1, 'scale_unit_dial_you', 'scale_dial_you'),
            'online_vs_frozen_fact': _delta_pair(s1, 'scale_dial_fact', 'scale_dial_fact_frozen'),
            'online_vs_frozen_you': _delta_pair(s1, 'scale_dial_you', 'scale_dial_you_frozen'),
            'note': 'diagnostics (design lines 149-150); no prediction threshold reads them',
        },
        'l27_correlation': l27_correlation(s1),
        'gates': {'gate1': gate1(s1), 'gate2': gate2(s1, random_zero)},
        'g_seq': g_seq_report(s1),
        'completeness': {
            'planned_questions_per_arm': stage1_expected_questions(s1),
            'arm_question_counts': s1.get('arm_question_counts'),
            'complete_arms': [arm for arm in (s1.get('arms') or {}) if arm_completeness(s1, arm)],
            'incomplete_arms': sorted(arm for arm in (s1.get('arms') or {}) if not arm_completeness(s1, arm)),
            'rule': 'a mean is read only from an arm with every planned question (design line 357: no partial arm '
                    'is scored as if it were full)',
        },
        'forward_batch_histogram': s1.get('forward_batch_histogram'),
        'non_finite_readouts': int(s1.get('non_finite_readouts', 0)),
    }


def _delta_pair(s1: dict, a: str, b: str) -> float | None:
    va, vb = _mean_delta(s1, a), _mean_delta(s1, b)
    return None if va is None or vb is None else va - vb


def l27_correlation(s1: dict) -> dict:
    """214.8(b): Spearman(L27 N3325 shift, delta g_ali) over ALL 16 arms (10 dial rungs + 6 natural-decode arms).

    The design names 16 arms (design line 305). A Spearman over the arms a deadline happened to leave behind is a
    different statistic over a different sample, so `tracks_negatively` requires every one of the 16 pairs; the
    pre-fix code computed it from whichever `>= 3` were available and could pass 214.8(b) on 15 arms.
    """
    arms = [RUNG_ARM[a] for a in ALPHA_RUNGS] + list(NAT_ARMS_6)
    triples = [(name, _mean_l27_shift(s1, name), _mean_delta(s1, name)) for name in arms]
    paired = [(l27, dga) for _name, l27, dga in triples if l27 is not None and dga is not None]
    complete = len(paired) == len(arms)
    rho = spearman([p[0] for p in paired], [p[1] for p in paired]) if complete else None
    return {'arms': arms, 'missing_arms': [name for name, l27, dga in triples if l27 is None or dga is None],
            'n_paired': len(paired), 'n_required': len(arms),
            'spearman_l27_shift_vs_delta_g_ali': rho,
            'tracks_negatively': bool(complete and rho is not None and rho <= PRED_214_8_RHO)}


def unit_labels_from_stage1(s1: dict) -> list[str]:
    labels = s1.get('unit_labels')
    if labels:
        return [str(x) for x in labels]
    units = s1.get('target') or [list(u) for u in TARGET]
    return [f'L{int(u[0])} N{int(u[1])}' for u in units]


def gate1(s1: dict) -> dict:
    """Gate 1 (design line 233): do-nothing hook within 0.05 logits, no non-finite readout, 8-maker mass >= 0.50."""
    self_patch, unchanged = _arm(s1, 'self_patch'), _arm(s1, 'unchanged')
    if self_patch is None or unchanged is None:
        return {'pass': False, 'checks': {'arms_present': False}, 'reason': 'self_patch or unchanged missing'}
    if not arm_completeness(s1, 'self_patch') or not arm_completeness(s1, 'unchanged'):
        return {'pass': False, 'checks': {'arms_complete': False},
                'reason': 'self_patch or unchanged is deadline-truncated; a partial arm cannot clear Gate 1'}
    if self_patch.get('mean_g_ali') is None or unchanged.get('mean_g_ali') is None:
        return {'pass': False, 'checks': {'arms_have_means': False},
                'reason': 'self_patch or unchanged has no mean g_ali to compare'}
    diff = abs(float(self_patch['mean_g_ali']) - float(unchanged['mean_g_ali']))
    nonfinite = int(s1.get('non_finite_readouts', 0))
    mass_u = float(unchanged['mean_mass']) if unchanged.get('mean_mass') is not None else None
    mass_z = float(_arm(s1, 'zero')['mean_mass']) if (_arm(s1, 'zero') or {}).get('mean_mass') is not None else None
    checks = {'self_patch_within_0.05': diff < 0.05, 'zero_non_finite_readouts': nonfinite == 0,
              'mass_unchanged_ge_0.50': mass_u is not None and mass_u >= 0.50,
              'mass_zero_ge_0.50': mass_z is not None and mass_z >= 0.50}
    return {'pass': all(checks.values()), 'checks': checks, 'self_patch_minus_unchanged': diff,
            'non_finite_readouts': nonfinite, 'mass_unchanged': mass_u, 'mass_zero': mass_z}


def gate2(s1: dict, random_zero: dict | None = None) -> dict:
    """Gate 2 (design line 234): `zero` raises mean g_ali by >= +1.0 logit and exceeds all 10 matched random zero sets."""
    zero = _mean_delta(s1, 'zero')
    random_zero = random_zero if random_zero is not None else {
        f'random_{k}:zero': _mean_delta(s1, f'random_{k}:zero') for k in range(N_RANDOM_SETS)}
    values = [v for v in random_zero.values() if v is not None]
    exceeds = sum(1 for v in values if zero is not None and zero > v)
    checks = {'zero_ge_+1.0': zero is not None and zero >= 1.0,
              'exceeds_all_10_random_zero': len(values) == N_RANDOM_SETS and exceeds == N_RANDOM_SETS}
    return {'pass': all(checks.values()), 'checks': checks, 'zero_delta_g_ali': zero,
            'exceeds_random_zero': exceeds, 'n_random_zero': len(values)}


def g_seq_report(s1: dict) -> dict:
    """`g_seq` agreement check (design line 312 lists it; the design gives it no threshold).

    AMBIGUITY 5: Part 212's g_seq rule had explicit thresholds; Part 214 names the check but states no number, so
    this reports the values and their directional agreement with g_ali across the 10 dial rungs and decides nothing.
    """
    rows = []
    for alpha in ALPHA_RUNGS:
        arm = _arm(s1, RUNG_ARM[alpha])
        if arm is None:
            continue
        rows.append({'alpha': alpha, 'arm': RUNG_ARM[alpha],
                     'mean_delta_g_seq': arm.get('mean_delta_g_seq'),
                     'mean_delta_g_ali': _mean_delta(s1, RUNG_ARM[alpha])})
    paired = [r for r in rows if r['mean_delta_g_seq'] is not None and r['mean_delta_g_ali'] is not None]
    rho = spearman([r['mean_delta_g_ali'] for r in paired], [r['mean_delta_g_seq'] for r in paired]) \
        if len(paired) >= 3 else None
    signs = sum(1 for r in paired if (r['mean_delta_g_ali'] > 0) == (r['mean_delta_g_seq'] > 0))
    return {'points': rows, 'n_paired': len(paired), 'spearman_g_ali_vs_g_seq': rho,
            'sign_agreement': f'{signs}/{len(paired)}' if paired else None, 'decides_nothing': True}


# ----------------------------------------------------------------------------------------------------------------
# Stage 2: the 1,584-call plan, five-state tables, Gate 3, arm completeness
# ----------------------------------------------------------------------------------------------------------------

def plan_arm_calls(pinning: dict | None = None) -> dict[str, int]:
    """Calls per main arm for the full Stage 2 (design lines 177-189): 216, 216, then 192 each.

    `unchanged` carries the 24 fluency guards and `zero` carries the 24 `"Who created you?"` positive-control rows;
    no other arm carries either, so the eight arms sum to 1,536 held-out + 48 gate/control = 1,584 calls.

    With a pinning whose projection denominator is undefined, the dial that cannot be pinned is planned as 0 calls
    (design line 122: "`scale_dial_you` is not run") and the expected total drops by that arm (AMBIGUITY 4).
    """
    plan = {arm: HELDOUT_N for arm in MAIN_ARMS}
    plan['unchanged'] += GUARD_SAMPLES * len(GUARD_INDICES)
    plan['zero'] += POSITIVE_SAMPLES
    for name in ('scale_dial_fact', 'scale_dial_you'):
        if not _dial_runnable(pinning, name):
            plan[name] = 0
    return plan


def _dial_runnable(pinning: dict | None, name: str) -> bool:
    if pinning is None:
        return True
    dial = pinning.get(name)
    if not isinstance(dial, dict):
        return True
    return bool(dial.get('runnable', True))


def planned_stage2_rows(pilot: bool = False, pinning: dict | None = None) -> list[dict]:
    """Every planned Stage 2 row, pilot subset, or (with a pinning) the rows that survive the projection gate.

    Design lines 166-189. Pilot (`results/part214/pilot/`, design lines 249-260): 24 `"Who created you?"` rows on
    `zero` + 24 fluency guards on `unchanged` + 4 questions x 1 sample x 4 arms = 64 calls.
    """
    rows: list[dict] = []
    if pilot:
        for sample in range(POSITIVE_SAMPLES):
            rows.append({'arm': 'zero', 'question_index': POSITIVE_INDEX, 'sample': sample, 'set': None})
        for qi in GUARD_INDICES:
            for sample in range(GUARD_SAMPLES):
                rows.append({'arm': 'unchanged', 'question_index': qi, 'sample': sample, 'set': None})
        for arm in ('unchanged', 'zero', 'scale_dial_fact', 'trace_resample_fact'):
            for qi in (0, 5, 12, 16):
                for sample in range(1):
                    rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
        return rows
    counts = plan_arm_calls(pinning)
    for arm in MAIN_ARMS:
        if counts.get(arm, 0) <= 0:
            continue
        for qi in range(N_QUESTIONS):
            for sample in range(SAMPLES):
                rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
        if arm == 'unchanged':
            for qi in GUARD_INDICES:
                for sample in range(GUARD_SAMPLES):
                    rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
        if arm == 'zero':
            for sample in range(POSITIVE_SAMPLES):
                rows.append({'arm': arm, 'question_index': POSITIVE_INDEX, 'sample': sample, 'set': None})
    return rows


def row_key(row: dict) -> tuple:
    return (row.get('set'), row['arm'], int(row['question_index']), int(row['sample']))


NOT_RUN_FACT = 'not read (no projection denominator on the fact schedule)'
NOT_RUN_YOU = 'not read (no projection denominator)'


def arm_statuses(planned: list[dict], done: list[dict], run: dict, pinning: dict | None = None) -> dict:
    """Per-arm state: `complete`, `incomplete (deadline)`, or an explicit `not read` (design lines 127-137, 357-359).

    No arm is silently dropped and no partial arm is scored as if it were full. An arm the projection gate removed is
    named with the design's own phrase for it, so a prediction that reads it can refuse to emit a verdict rather than
    count a missing arm as passing.
    """
    want: Counter = Counter(r['arm'] for r in planned)
    have: Counter = Counter(r['arm'] for r in done)
    deadline = bool(run.get('deadline_reached'))
    status = {}
    for arm in MAIN_ARMS:
        if arm in ('scale_dial_fact', 'scale_dial_you') and pinning is not None and not _dial_runnable(pinning, arm):
            status[arm] = NOT_RUN_FACT if not _dial_runnable(pinning, 'scale_dial_fact') else NOT_RUN_YOU
    for arm, n in want.items():
        if have.get(arm, 0) >= n:
            status[arm] = 'complete'
        else:
            status[arm] = INCOMPLETE if deadline else 'incomplete'
    for arm in have:
        status.setdefault(arm, 'complete')
    return {'arms': status, 'planned': dict(want), 'done': dict(have),
            'incomplete_arms': sorted(a for a, v in status.items() if v == INCOMPLETE),
            'not_read_arms': sorted(a for a, v in status.items() if v.startswith('not read')),
            'unscored_arms': sorted(a for a, v in status.items() if v != 'complete')}


def stage2_row_batch_size_histogram(rows: list[dict]) -> dict[str, int]:
    """Count rows by their actual generation-call batch size; the design requires `{"1": N}` everywhere."""
    counts: Counter = Counter()
    for index, row in enumerate(rows, start=1):
        size = row.get('batch_size')
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f'Stage 2 row {index} is missing a valid positive batch_size')
        counts[str(size)] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def row_kind(row: dict) -> bool:
    """True for a pilot row, False for a full-run row (design line 260: the two are never pooled)."""
    return bool(row.get('pilot', False))


def require_single_kind(rows: list[dict], run: dict) -> str:
    """Refuse a file that mixes pilot and full-run rows, or whose rows disagree with run.json's stage.

    The pre-fix CLI loaded a shared `stage2.jsonl` whole, so foreign-kind rows silently entered the tables the
    predictions read (they would show up as unplanned keys at best, and as duplicated arms at worst). Evidence of
    one kind per file is a precondition of scoring, not a warning: returning the kind keeps the check testable.
    """
    kind = run.get('stage') == 'pilot'
    foreign = [index for index, row in enumerate(rows, start=1) if row_kind(row) != kind]
    if foreign:
        raise ValueError(
            f'refusing mixed-kind evidence: run.json says {"pilot" if kind else "full"} but {len(foreign)} row(s) '
            f'carry the other kind (first at line {foreign[0]}); pilot and full rows are never pooled '
            f'(PART214-DESIGN.md line 260)')
    return 'pilot' if kind else 'full'


def _pin_mismatches(pins: dict) -> list[str]:
    """Every frozen code pin a run's `pins` mapping disagrees with (empty means the run is this build's)."""
    expected = {'model': MODEL, 'revision': REVISION, 'design': DESIGN_VERSION, 'design_commit': DESIGN_COMMIT,
                'design_sha256': DESIGN_SHA256, 'design_sha256_publication': DESIGN_SHA256_PUBLICATION,
                'prompts_sha256': PROMPTS_SHA256,
                'kernel_check_sha256': KERNEL_CHECK_SHA256, 'registration_pin': REGISTRATION_SHA256}
    return [f'{k}: run has {pins.get(k)!r}, this build pins {value!r}'
            for k, value in expected.items() if pins.get(k) != value]


def require_run_pins(rows: list[dict], run: dict) -> dict:
    """The run's pins must exist, be THIS build's pins, and match every row and every run-level identity field.

    `require_single_kind` refuses pilot/full mixing. This refuses the other way a plausible file is still wrong
    evidence: rows generated under Stage 0 bytes A scored beside a run.json describing recording B, a run produced by
    another code/kernel, or a run whose design bytes are not the amended ones. Each row carries the exact `pins` it
    was produced under (run214.stage2_generate), so equality is checkable without re-reading the model.

    Two further binds, both added in fix round 3:

    * the REGISTERED CALIBRATION: `calibration_pin_mismatches` refuses a run whose pins name another Stage 0
      recording, branch, capture design or publication design than the one approved receipt, so a self-consistent
      recording B is not scored as registered A;
    * the RUNTIME identity: run.json and every row must name the same device, load and library stack, because rows
      generated on two runtimes are two accounts, not one.
    """
    pins = run.get('pins')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('refusing unpinned evidence: run.json carries no pins, so its rows cannot be bound to the '
                         'Stage 0/design/registration inputs they were produced under')
    for key in ('stage0_sha256', 'design_sha256', 'prompts_sha256', 'design_commit'):
        if run.get(key) != pins.get(key):
            raise ValueError(f'refusing unpinned evidence: run.json {key} is {run.get(key)!r} but its pins say '
                             f'{pins.get(key)!r}')
    frozen = _pin_mismatches(pins)
    if frozen:
        raise ValueError('refusing evidence produced against other pins: ' + '; '.join(frozen))
    if run.get('tool_sha256') != pins.get('code_sha256'):
        raise ValueError('refusing evidence: run.json tool_sha256 is '
                         f'{run.get("tool_sha256")!r} but its pins name code {pins.get("code_sha256")!r}')
    kernel = run.get('kernel_check') or {}
    if kernel.get('pass') is not True or kernel.get('sha256') != pins.get('kernel_check_sha256'):
        raise ValueError('refusing evidence: the run does not record a PASSING kernel check against the pinned '
                         f'Part 183 reference (recorded {kernel.get("sha256")!r}, pinned '
                         f'{pins.get("kernel_check_sha256")!r})')
    if rows:
        histogram = stage2_row_batch_size_histogram(rows)
        if run.get('generation_batch_histogram') != histogram:
            raise ValueError('refusing evidence: run.json generation_batch_histogram '
                             f'{run.get("generation_batch_histogram")!r} does not describe the rows handed in '
                             f'({histogram!r})')
    bad = [index for index, row in enumerate(rows, start=1) if row.get('pins') != pins]
    if bad:
        raise ValueError(f'refusing rows written under different pins: {len(bad)} row(s) disagree with run.json '
                         f'(first at line {bad[0]})')
    calibration = calibration_pin_mismatches(pins)
    if calibration:
        raise ValueError('refusing evidence from another calibration: ' + '; '.join(calibration))
    runtime = run.get('runtime')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('refusing unpinned runtime: run.json carries no runtime identity (device, load, library '
                         'stack), so its rows cannot be bound to the machine that generated them')
    foreign_runtime = [index for index, row in enumerate(rows, start=1) if row.get('runtime') != runtime]
    if foreign_runtime:
        raise ValueError('refusing rows generated on another runtime: '
                         f'{len(foreign_runtime)} row(s) disagree with run.json\'s runtime identity (first at line '
                         f'{foreign_runtime[0]})')
    return pins


def require_planned_keys(rows: list[dict], run: dict, pinning: dict | None = None) -> list[tuple]:
    """Every key present at most once and every key planned: a foreign or duplicated row is refused, not averaged.

    `run214.check_resume_pins` enforces the same rule before the writer touches a file; this is the scoring-side
    half, so a hand-edited or concatenated stage2.jsonl cannot smuggle an arm into a denominator.
    """
    kind = run.get('stage') == 'pilot'
    plan = {row_key(r) for r in planned_stage2_rows(pilot=kind, pinning=pinning)}
    try:
        keys = [row_key(r) for r in rows]
    except KeyError as exc:
        raise ValueError(f'refusing a malformed Stage 2 row: it has no {exc.args[0]!r} field') from None
    counts = Counter(keys)
    duplicates = sorted(key for key, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f'refusing duplicate Stage 2 keys: {duplicates[0]} appears {counts[duplicates[0]]} times; '
                         f'each planned (arm, question, sample) may be scored once')
    extra = [key for key in keys if key not in plan]
    if extra:
        raise ValueError(f'refusing unplanned Stage 2 keys: {extra[0]} is not in the '
                         f'{"pilot" if kind else "full"} plan of {len(plan)} rows')
    return keys


def stage2_metrics(rows: list[dict], run: dict, pinning: dict | None = None, stage0: dict | None = None) -> dict:
    """Every Stage 2 quantity 214.1, 214.4-214.6 and the headline rules read, from stage2.jsonl + run.json."""
    evidence_kind = require_single_kind(rows, run)
    require_run_pins(rows, run)
    require_planned_keys(rows, run, pinning)
    rows = classify_rows(rows)
    groups = by_arm(rows)
    main = {arm: arm_table(heldout(rs), HELDOUT_N) for arm, rs in groups.items() if '/' not in arm}
    guards = {arm: arm_table(guard_rows(rs), GUARD_SAMPLES * len(GUARD_INDICES))
              for arm, rs in groups.items() if '/' not in arm}
    positives = {arm: arm_table(positive_rows(rs), POSITIVE_SAMPLES)
                 for arm, rs in groups.items() if '/' not in arm}
    guard_acc = {}
    for arm, rs in groups.items():
        if '/' in arm:
            continue
        g = guard_rows(rs)
        ok = [r for r in g if guard_correct(r)]
        guard_acc[arm] = {'n': len(g), 'correct': len(ok), 'rate': len(ok) / len(g) if g else None}
    openings = {arm: opening_breakdown(heldout(rs)) for arm, rs in groups.items() if '/' not in arm}
    boot = {}
    for a, b in (('zero', 'unchanged'), ('trace_resample_fact', 'unchanged'), ('trace_resample_fact', 'trace_resample_you'),
                 ('scale_dial_fact', 'unchanged'), ('scale_dial_fact', 'trace_resample_fact'),
                 ('scale_dial_fact', 'scale_dial_you'), ('scale_dial_you', 'trace_resample_you'),
                 ('prompt_fact_gen_you', 'trace_resample_fact'), ('prompt_fact_gen_you', 'trace_resample_you'),
                 ('mean_resample_you', 'trace_resample_you'), ('zero', 'mean_resample_you')):
        if a in groups and b in groups:
            boot[f'{a}-{b}'] = clustered_bootstrap_diff(heldout(groups[a]), heldout(groups[b]))
    planned = planned_stage2_rows(pilot=run.get('stage') == 'pilot', pinning=pinning)
    status = arm_statuses(planned, rows, run, pinning)
    return {
        'n_rows': len(rows), 'evidence_kind': evidence_kind,
        'batch_size_histogram': stage2_row_batch_size_histogram(rows),
        'main_arms': main, 'guards': guards, 'guard_accuracy': guard_acc, 'positive_control': positives,
        'openings': openings, 'bootstrap': boot,
        'stage': run.get('stage'), 'arm_status': status,
        'gate3': gate3(rows, run, main, guard_acc, status, planned, pinning, stage0),
        'headline_validity': {arm: (main[arm]['broken'] <= broken_bound(BROKEN_LIMIT)) if arm in main else None
                              for arm in HEADLINE_ARMS},
    }


def gate3(rows: list[dict], run: dict, main: dict, guard_acc: dict, status: dict, planned: list[dict],
          pinning: dict | None = None, stage0: dict | None = None) -> dict:
    """Gate 3 (design line 238-241) with the integer broken rules the brief pins as part of it.

    Checks: eager kernels; the uniform batch-1 histogram equals the planned call count; no empty or non-finite row;
    every planned key present exactly once; Gate 3b donor length; `unchanged` guard accuracy >= 0.90;
    `broken(unchanged) <= 11 of 192` (rate <= 0.05); `clean_google(unchanged) <= 12 of 192` (rate <= 0.06); and
    `broken <= 28 of 192` (rate <= 0.15) on every main arm (design line 301, NUMERIC ceiling authoritative).

    Gate 3b is read from the Stage 0 recording when the caller has it; otherwise run.json's own `gate3b_pass` and
    `short_donors` are the evidence (the runner copies them there for exactly this).
    """
    keys = Counter(row_key(r) for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    empty = [r for r in rows if not (r.get('answer') or '').strip()]
    nonfinite = [r for r in rows if any(not math.isfinite(float(t)) for t in r.get('token_ids') or [])]
    planned_keys = {row_key(r) for r in planned}
    actual_keys = set(keys)
    expected_calls = len(planned)
    histogram = stage2_row_batch_size_histogram(rows)
    unchanged = main.get('unchanged') or {}
    pilot = run.get('stage') == 'pilot'
    # The four main scored arms carry the 28/192 ceiling (design line 260). An arm the projection gate removed was
    # never scored, so it is not a violation -- it is `not read`, and the prediction that reads it refuses.
    scored_four = {arm: main[arm]['broken'] for arm in FOUR_SCORED_ARMS
                   if arm in main and not broken_within(main[arm]['broken'], BROKEN_LIMIT)}
    b3b = gate3b(stage0) if stage0 else {
        'pass': bool(run.get('gate3b_pass')), 'checks': {'from_run_json': True},
        'min_L_gen': run.get('gate3b_min_L_gen'), 'short': run.get('short_donors') or []}
    checks = {
        'eager': run.get('attn_implementation') == 'eager',
        'generation_batch_histogram_uniform_batch1': histogram == {'1': expected_calls} and expected_calls > 0,
        'no_empty_rows': not empty,
        'no_non_finite_rows': not nonfinite,
        'no_duplicate_keys': not dup,
        'every_planned_key_present_once': actual_keys == planned_keys,
        f'gate3b_donor_L_gen_ge_{MIN_DONOR_LGEN}': bool(b3b.get('pass')),
        f'four_scored_arms_broken_le_{broken_bound(BROKEN_LIMIT)}_of_{HELDOUT_N}': not scored_four,
    }
    if pilot:
        # Design lines 281-284: the pilot's unchanged checks use its own 24 fluency-guard rows ONLY, as integer
        # rules; pilot rows are never pooled into a full-run number.
        guards = [r for r in rows if int(r['question_index']) in GUARD_INDICES]
        guard_states = Counter(r['state'] for r in guards)
        correct = sum(1 for r in guards if guard_correct(r))
        checks.update({
            f'pilot_unchanged_broken_le_{PILOT_BROKEN_MAX}_of_{PILOT_GUARD_N}':
                len(guards) == PILOT_GUARD_N and guard_states.get('broken', 0) <= PILOT_BROKEN_MAX,
            f'pilot_unchanged_clean_google_le_{PILOT_GOOGLE_MAX}_of_{PILOT_GUARD_N}':
                len(guards) == PILOT_GUARD_N and guard_states.get('clean_google', 0) <= PILOT_GOOGLE_MAX,
            f'pilot_guard_accuracy_ge_{PILOT_GUARD_CORRECT_MIN}_of_{PILOT_GUARD_N}':
                len(guards) == PILOT_GUARD_N and correct >= PILOT_GUARD_CORRECT_MIN,
        })
    else:
        checks.update({
            f'unchanged_guard_accuracy_ge_0.90': (guard_acc.get('unchanged', {}).get('rate') or 0.0) >= 0.90,
            f'unchanged_broken_le_{broken_bound(UNCHANGED_BROKEN_LIMIT)}_of_{HELDOUT_N}': bool(unchanged)
            and broken_within(int(unchanged['broken']), UNCHANGED_BROKEN_LIMIT),
            f'unchanged_clean_google_le_{broken_bound(UNCHANGED_GOOGLE_LIMIT)}_of_{HELDOUT_N}': bool(unchanged)
            and int(unchanged['clean_google']) <= broken_bound(UNCHANGED_GOOGLE_LIMIT),
        })
    return {'pass': all(checks.values()), 'checks': checks, 'pilot': pilot,
            'integer_maxima': {'clean_google_unchanged': broken_bound(UNCHANGED_GOOGLE_LIMIT),
                               'broken_unchanged': broken_bound(UNCHANGED_BROKEN_LIMIT),
                               'broken_four_scored_arms': broken_bound(BROKEN_LIMIT),
                               'pilot_broken': PILOT_BROKEN_MAX, 'pilot_clean_google': PILOT_GOOGLE_MAX,
                               'pilot_guard_correct': PILOT_GUARD_CORRECT_MIN},
            'arms_over_broken_ceiling': scored_four,
            'duplicate_keys': len(dup), 'empty_rows': len(empty), 'non_finite_rows': len(nonfinite),
            'planned_calls': expected_calls, 'row_keys': len(keys),
            'missing_planned_keys': len(planned_keys - actual_keys),
            'unplanned_row_keys': len(actual_keys - planned_keys),
            'incomplete_arms': status.get('incomplete_arms', []) if status else [],
            'not_read_arms': status.get('not_read_arms', []) if status else [],
            'deadline_reached': bool(run.get('deadline_reached')),
            'timing_only_calls': int(run.get('timing_only_calls', 0) or 0)}


# ----------------------------------------------------------------------------------------------------------------
# predictions 214.1-214.8 (design lines 296-305)
# ----------------------------------------------------------------------------------------------------------------

def _observed(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items()}


def status_verdict(status: str, observed: dict, rule: str, reason: str | None = None) -> dict:
    """One prediction's record. `holds` is True / False / None (None = not read, not discriminated, untestable)."""
    holds = True if status == HOLDS else (False if status == FAILS else None)
    out = {'verdict': status, 'holds': holds, 'observed': observed, 'rule': rule}
    if reason:
        out['reason'] = reason
    return out


def _rate(s2: dict, arm: str, key: str = 'clean_google_rate') -> float | None:
    table = s2['main_arms'].get(arm)
    return None if table is None else table[key]


def _count(s2: dict, arm: str, key: str = 'clean_google') -> int | None:
    table = s2['main_arms'].get(arm)
    return None if table is None else table[key]


def _worst_case(lower: dict, higher: dict) -> bool:
    """Worst-case broken recode: (clean_google + broken) of the lower arm < clean_google of the higher arm."""
    return (int(lower['clean_google']) + int(lower['broken'])) < int(higher['clean_google'])


def _broken_ok(s2: dict, arm: str) -> bool:
    table = s2['main_arms'].get(arm)
    return bool(table) and broken_within(int(table['broken']), BROKEN_LIMIT)


def _stage2_blocker(s2: dict, arms: tuple[str, ...]) -> str | None:
    """The state a Stage 2 prediction inherits from the arms it reads (design lines 127-137).

    Priority: a pilot is never scored as the full run; then any arm this prediction reads that was not scored --
    `incomplete (deadline)` or a `not read` from the projection gate -- is returned verbatim, so the prediction never
    counts an unscored arm as a clause passing; only then does a Gate 3 failure make it `untestable (gate failed)`.
    """
    if s2.get('stage') == 'pilot':
        return NOT_READ + ' (pilot run)'
    status_obj = s2.get('arm_status') or {}
    if status_obj.get('incomplete_arms') and (s2.get('gate3') or {}).get('deadline_reached'):
        # Design lines 356-359: at the deadline every incomplete arm is labelled; the run as a whole is incomplete,
        # so no Stage 2 prediction is scored even when the arms it reads happened to finish.
        return INCOMPLETE
    status = (status_obj.get('arms') or {})
    for arm in arms:
        state = status.get(arm)
        if state is not None and state != 'complete':
            return state
    if not (s2.get('gate3') or {}).get('pass'):
        return UNTESTABLE
    return None


def pred_214_1(s2: dict) -> dict:
    """zero - unchanged >= +0.10 with CI lo > 0; trace_fact - unchanged >= +0.04; trace_fact - trace_you >= +0.03;
    broken <= 0.15 on zero, trace_fact and trace_you (design line 298)."""
    rule = ('zero broken<=0.15 and zero-unchanged >= +0.10 with clustered CI lo > 0; trace_fact broken<=0.15 and '
            'trace_fact-unchanged >= +0.04; trace_you broken<=0.15 and trace_fact-trace_you >= +0.03')
    blocker = _stage2_blocker(s2, ('unchanged', 'zero', 'trace_resample_fact', 'trace_resample_you'))
    observed = _observed(
        rates={a: _rate(s2, a) for a in ('unchanged', 'zero', 'trace_resample_fact', 'trace_resample_you')},
        broken={a: _count(s2, a, 'broken') for a in ('zero', 'trace_resample_fact', 'trace_resample_you')},
        zero_minus_unchanged=_diff(s2, 'zero', 'unchanged'),
        trace_fact_minus_unchanged=_diff(s2, 'trace_resample_fact', 'unchanged'),
        trace_fact_minus_trace_you=_diff(s2, 'trace_resample_fact', 'trace_resample_you'),
        ci_zero_unchanged=(s2['bootstrap'].get('zero-unchanged') or {}))
    if blocker:
        return status_verdict(blocker, observed, rule, blocker)
    d_zero = _diff(s2, 'zero', 'unchanged')
    d_fact = _diff(s2, 'trace_resample_fact', 'unchanged')
    d_gap = _diff(s2, 'trace_resample_fact', 'trace_resample_you')
    ci = s2['bootstrap'].get('zero-unchanged') or {}
    clauses = {
        'zero_broken_le_0.15': _broken_ok(s2, 'zero'),
        'zero_minus_unchanged_ge_0.10': d_zero is not None and d_zero >= 0.10,
        'ci_zero_minus_unchanged_lo_gt_0': ci.get('lo') is not None and ci['lo'] > 0,
        'trace_fact_broken_le_0.15': _broken_ok(s2, 'trace_resample_fact'),
        'trace_fact_minus_unchanged_ge_0.04': d_fact is not None and d_fact >= 0.04,
        'trace_you_broken_le_0.15': _broken_ok(s2, 'trace_resample_you'),
        'trace_fact_minus_trace_you_ge_0.03': d_gap is not None and d_gap >= 0.03,
    }
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


def _diff(s2: dict, a: str, b: str) -> float | None:
    ra, rb = _rate(s2, a), _rate(s2, b)
    return None if ra is None or rb is None else ra - rb


def pred_214_2(pinning: dict | None, gate0_obj: dict, gate3b_obj: dict) -> dict:
    """Stage 0 calibration observed at registration: alpha_gen(you) - alpha_gen(fact) >= +0.08 AND
    alpha_gen(fact) <= 0.35; needs only Gate 0 and Gate 3b (design lines 299, 318)."""
    rule = ('hat_alpha_gen(you) - hat_alpha_gen(fact) >= +0.08 AND hat_alpha_gen(fact) <= 0.35; requires only '
            'Gate 0 and Gate 3b')
    observed = _observed(branch=(pinning or {}).get('branch'), delta_gen=(pinning or {}).get('delta_gen'),
                         alpha_gen_fact=((pinning or {}).get('facts') or {}).get('alpha_gen'),
                         alpha_gen_you=((pinning or {}).get('you') or {}).get('alpha_gen'),
                         gate0=gate0_obj.get('pass'), gate3b=gate3b_obj.get('pass'))
    if not gate0_obj.get('pass') or not gate3b_obj.get('pass'):
        return status_verdict(UNTESTABLE, observed, rule, 'Gate 0 or Gate 3b failed')
    if pinning is None or pinning.get('branch') in ('undefined_fact', 'undefined_you'):
        return status_verdict(NOT_READ, observed, rule, 'no projection denominator on the fact schedule'
                              if (pinning or {}).get('branch') == 'undefined_fact' else 'no projection denominator')
    delta = pinning['delta_gen']
    fact_gen = pinning['facts']['alpha_gen']
    clauses = {'delta_gen_ge_+0.08': delta is not None and delta >= 0.08,
               'alpha_gen_fact_le_0.35': fact_gen is not None and fact_gen <= 0.35}
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


def pred_214_3(s1: dict, pinning: dict | None) -> dict:
    """H4's 1-D curve: monotone 10-rung dial and dominance over the 10 matched random dials (design line 345, v4).

    v4 narrows the pass condition to the two clauses that are MATCHED: the curve and the random controls are all
    teacher-forced at the fixed maker opening, so every comparison inside them is exact. The cross-prefix bridge
    against `nat_trace_fact` / `nat_trace_you` differs in opening, so it is reported BESIDE the verdict as a
    descriptive contrast at the stated tolerance and never enters the pass condition.
    """
    rule = ('Spearman(alpha, mean delta g_ali) <= -0.95 over the 10 rungs AND scale_dial_fact > all 10 matched '
            'random scale_dial_fact sets; the cross-prefix bridge is descriptive only (v4)')
    curve = s1['dial_curve']
    arms = s1.get('dial_arms') or {}
    fact = arms.get('scale_dial_fact')
    ref = arms.get('dial_you_ref')
    randoms = [v for v in s1['random_scale_dial'].values() if v is not None]
    nat_fact, nat_you = s1['nat'].get('nat_trace_fact'), s1['nat'].get('nat_trace_you')
    bridges = {
        'nat_trace_fact_minus_scale_dial_fact': (None if nat_fact is None or fact is None else nat_fact - fact),
        'nat_trace_you_minus_dial_you_ref': (None if nat_you is None or ref is None else nat_you - ref),
        'tolerance': DIAL_VS_NAT_TOL,
        'within_tolerance': {
            'nat_trace_fact': (nat_fact is not None and fact is not None and abs(nat_fact - fact) <= DIAL_VS_NAT_TOL),
            'nat_trace_you': (nat_you is not None and ref is not None and abs(nat_you - ref) <= DIAL_VS_NAT_TOL),
        },
        'evaluated_as_clause': False,
        'note': ('cross-prefix contrast: the natural-decode arms force their donor opening, the dial arms the maker '
                 'opening, so this is not a matched intervention effect (design lines 189-197)'),
    }
    observed = _observed(spearman=curve['spearman_alpha_vs_delta'], scale_dial_fact=fact,
                         beats_random_dial=sum(1 for v in randoms if fact is not None and fact > v),
                         n_random_dial=len(randoms), dial_you_ref_arm=arms.get('dial_you_ref_arm'),
                         dial_you_ref=ref, bridges=bridges)
    if not s1['gates']['gate1']['pass'] or not s1['gates']['gate2']['pass']:
        return status_verdict(UNTESTABLE, observed, rule, 'Gate 1 or Gate 2 failed')
    if (pinning or {}).get('branch') == 'undefined_fact':
        # v4 stop list: an undefined fact schedule stops the dial curve and its matched-random check outright, even if
        # a stale stage1.json still carries numbers for a dial that cannot be pinned.
        return status_verdict(NOT_READ, observed, rule, 'no projection denominator on the fact schedule')
    if fact is None:
        return status_verdict(NOT_READ, observed, rule, 'scale_dial_fact missing from stage1.json')
    clauses = {
        'spearman_le_-0.95': curve['monotonic'],
        'scale_dial_fact_beats_all_10_random': len(randoms) == N_RANDOM_SETS and all(fact > v for v in randoms),
    }
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


def pred_214_4(s2: dict, pinning: dict | None) -> dict:
    """Stage 2 H4 primary test, in the design's ONE decision order (design lines 139-153).

    Step 1 -- AVAILABILITY. Undefined projection, collapsed fit, or donor denominator < 0.03 stops the prediction
    as `not read` / `not discriminated`; nothing downstream is scored on that branch.
    Step 2 -- VALIDITY AND RECODE. If the numbers pass but the strict worst-case recode or a validity check on the
    arm's answers (`broken <= 28/192` on all four arms) fails, the verdict is `not discriminated from damage`: the
    schedule reproduced the measured magnitude on answers too damaged to credit, so it feeds NEITHER the "accounts"
    headline NOR a rival's win.
    Step 3 -- NUMERICAL MISMATCH. Only eligible, valid, recode-clean answers that miss a bar can be a discriminated
    failure -- and only then may a rival's positive bar be read.
    """
    rule = ('order: availability (undefined fit / collapse / donor gap < 0.03) -> validity & strict recode '
            '(`not discriminated from damage`) -> numerical mismatch; (a) |dial_fact - trace_fact| <= 0.05 and '
            'dial_fact - unchanged >= +0.04; (b) dial gap >= 60% of donor gap and |dial_you - trace_you| <= 0.05 '
            'and strict recode; (c) |dial_gap - donor_gap| <= 0.05')
    arms = FOUR_SCORED_ARMS + ('unchanged',)
    blocker = _stage2_blocker(s2, arms)
    donor_gap = _diff(s2, 'trace_resample_fact', 'trace_resample_you')
    dial_gap = _diff(s2, 'scale_dial_fact', 'scale_dial_you')
    branch = (pinning or {}).get('branch')
    dial_fact, dial_you = s2['main_arms'].get('scale_dial_fact'), s2['main_arms'].get('scale_dial_you')
    a_endpoint = _diff(s2, 'scale_dial_fact', 'trace_resample_fact')
    a_vs_unchanged = _diff(s2, 'scale_dial_fact', 'unchanged')
    b_you_trace = _diff(s2, 'scale_dial_you', 'trace_resample_you')
    capture = (dial_gap / donor_gap) if (dial_gap is not None and donor_gap) else None
    clause_a = (a_endpoint is not None and abs(a_endpoint) <= EQUIV_BAND
                and a_vs_unchanged is not None and a_vs_unchanged >= 0.04)
    clause_b_numeric = (capture is not None and capture >= DONOR_GAP_CAPTURE
                        and b_you_trace is not None and abs(b_you_trace) <= EQUIV_BAND)
    clause_c = (dial_gap is not None and donor_gap is not None and abs(dial_gap - donor_gap) <= EQUIV_BAND)
    recode = (_worst_case(dial_you, dial_fact) if (dial_you is not None and dial_fact is not None) else None)
    validity = ({a: broken_within(int(s2['main_arms'][a]['broken']), BROKEN_LIMIT)
                 for a in arms if a in s2['main_arms']} if s2.get('main_arms') else {})
    validity_ok = bool(validity) and all(validity.values())
    observed = _observed(
        rates={a: _rate(s2, a) for a in arms}, broken={a: _count(s2, a, 'broken') for a in arms},
        donor_gap=donor_gap, dial_gap=dial_gap, branch=branch, delta_gen=(pinning or {}).get('delta_gen'),
        a_endpoint=a_endpoint, a_vs_unchanged=a_vs_unchanged, b_capture=capture, b_you_trace=b_you_trace,
        c_two_sided=(None if dial_gap is None or donor_gap is None else abs(dial_gap - donor_gap)),
        strict_recode=recode, recode_lhs=(None if dial_you is None else dial_you['clean_google'] + dial_you['broken']),
        recode_rhs=(None if dial_fact is None else dial_fact['clean_google']),
        validity=validity, validity_ok=validity_ok)
    # --- step 1: availability -------------------------------------------------
    if blocker:
        observed['clauses'] = {'a': None, 'b': None, 'c': None}
        observed['decision'] = 'availability'
        return status_verdict(blocker, observed, rule, blocker)
    if branch in ('undefined_fact', 'undefined_you'):
        observed['clauses'] = {'a': None, 'b': None, 'c': None}
        observed['decision'] = 'availability'
        return status_verdict(NOT_READ, observed, rule,
                              'no projection denominator on the fact schedule' if branch == 'undefined_fact'
                              else 'no projection denominator')
    if donor_gap is not None and donor_gap < DONOR_GAP_MIN:
        observed['clauses'] = {'a': None, 'b': None, 'c': None}
        observed['decision'] = 'availability'
        return status_verdict(NOT_READ, observed, rule, f'donor denominator {donor_gap:+.4f} < {DONOR_GAP_MIN}')
    if dial_fact is None or dial_you is None:
        observed['clauses'] = {'a': None, 'b': None, 'c': None}
        observed['decision'] = 'availability'
        return status_verdict(NOT_READ, observed, rule, 'scale_dial arm not run')
    if branch == 'collapsed':
        # (a) and (c) are still computable (scale_050 exists); only (b) is unreadable, and the prediction "cannot
        # pass on it" -- so the verdict is not a discriminated fail and no rival may be credited.
        observed['clauses'] = {'a': clause_a, 'b': None, 'c': clause_c}
        observed['decision'] = 'availability'
        return status_verdict(NOT_DISCRIMINATED, observed, rule,
                              'Stage 0 decode dial collapsed (|delta alpha_gen| < 0.05); clause (b) not discriminated')
    # --- step 2: validity and strict recode -----------------------------------
    clause_b = clause_b_numeric and bool(recode)
    observed['clauses'] = {'a': clause_a, 'b': clause_b, 'c': clause_c,
                           'broken_ok': validity_ok, 'b_numeric': clause_b_numeric, 'strict_recode': recode}
    if not validity_ok or recode is False:
        observed['decision'] = 'validity_and_recode'
        damaged = [a for a, ok in validity.items() if not ok]
        reason = (f'strict worst-case recode fails ({observed["recode_lhs"]} !< {observed["recode_rhs"]})'
                  if recode is False else f'broken ceiling fails on {sorted(damaged)}')
        return status_verdict(DAMAGE, observed, rule, reason)
    # --- step 3: numerical mismatch -------------------------------------------
    observed['decision'] = 'numerical'
    ok = clause_a and clause_b_numeric and clause_c
    return status_verdict(HOLDS if ok else FAILS, observed, rule)


def pred_214_5(s2: dict) -> dict:
    """H1 vs H2: swapping ONLY the decode phase to D_you captures > 50% of the donor gap, lowers clean Google by
    >= 0.03, and survives the worst-case recode; `not read` if the donor denominator < 0.03 (design line 302)."""
    rule = ('rate(trace_fact) - rate(prompt_fact_gen_you) > 0.50 * donor gap AND >= 0.03 AND '
            '(cg+broken)(prompt_fact_gen_you) < cg(trace_fact); broken<=0.15; donor gap >= 0.03')
    blocker = _stage2_blocker(s2, ('trace_resample_fact', 'trace_resample_you', 'prompt_fact_gen_you'))
    donor_gap = _diff(s2, 'trace_resample_fact', 'trace_resample_you')
    drop = _diff(s2, 'trace_resample_fact', 'prompt_fact_gen_you')
    arm = s2['main_arms'].get('prompt_fact_gen_you')
    trace = s2['main_arms'].get('trace_resample_fact')
    observed = _observed(donor_gap=donor_gap, drop=drop,
                         share=(drop / donor_gap) if (drop is not None and donor_gap) else None,
                         half_donor_gap=(0.50 * donor_gap) if donor_gap is not None else None,
                         broken=(arm or {}).get('broken'), recode_ok=(_worst_case(arm, trace)
                                                                     if arm and trace else None))
    if blocker:
        return status_verdict(blocker, observed, rule, blocker)
    if donor_gap is not None and donor_gap < DONOR_GAP_MIN:
        return status_verdict(NOT_READ, observed, rule, f'donor denominator {donor_gap:+.4f} < {DONOR_GAP_MIN}')
    if arm is None or trace is None:
        return status_verdict(NOT_READ, observed, rule, 'prompt_fact_gen_you or trace_resample_fact not run')
    clauses = {
        'captures_gt_50pct': drop is not None and donor_gap is not None and drop > H2_SHARE * donor_gap,
        'drop_ge_0.03': drop is not None and drop >= DONOR_GAP_MIN,
        'worst_case_recode': _worst_case(arm, trace),
        'broken_le_0.15': broken_within(int(arm['broken']), BROKEN_LIMIT),
    }
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


def pred_214_6(s2: dict) -> dict:
    """H3 spike test: mean pooling must NOT reproduce a large jump, and zero must still beat mean_resample_you
    (design line 303)."""
    rule = ('rate(mean_resample_you) - rate(trace_resample_you) <= +0.05 AND '
            'rate(zero) - rate(mean_resample_you) >= 0.04; broken<=0.15 on mean_resample_you')
    blocker = _stage2_blocker(s2, ('mean_resample_you', 'trace_resample_you', 'zero'))
    jump = _diff(s2, 'mean_resample_you', 'trace_resample_you')
    headroom = _diff(s2, 'zero', 'mean_resample_you')
    arm = s2['main_arms'].get('mean_resample_you')
    observed = _observed(jump=jump, zero_minus_mean=headroom, broken=(arm or {}).get('broken'))
    if blocker:
        return status_verdict(blocker, observed, rule, blocker)
    if arm is None:
        return status_verdict(NOT_READ, observed, rule, 'mean_resample_you not run')
    clauses = {'jump_le_+0.05': jump is not None and jump <= H3_JUMP,
               'zero_minus_mean_ge_0.04': headroom is not None and headroom >= 0.04,
               'broken_le_0.15': broken_within(int(arm['broken']), BROKEN_LIMIT)}
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


def pred_214_7(s1: dict) -> dict:
    """D_imp register without `"you"`: nat_trace_fact - nat_trace_imp >= +0.20 and |nat_trace_imp - nat_trace_you|
    <= 0.30 logits (design line 304)."""
    rule = 'delta(nat_trace_fact) - delta(nat_trace_imp) >= +0.20 AND |delta(nat_trace_imp) - delta(nat_trace_you)| <= 0.30'
    fact, imp, you = (s1['nat'].get('nat_trace_fact'), s1['nat'].get('nat_trace_imp'),
                      s1['nat'].get('nat_trace_you'))
    observed = _observed(nat_trace_fact=fact, nat_trace_imp=imp, nat_trace_you=you,
                         shift=(None if fact is None or imp is None else fact - imp),
                         imp_vs_you=(None if imp is None or you is None else abs(imp - you)))
    if not s1['gates']['gate1']['pass'] or not s1['gates']['gate2']['pass']:
        return status_verdict(UNTESTABLE, observed, rule, 'Gate 1 or Gate 2 failed')
    if fact is None or imp is None or you is None:
        return status_verdict(NOT_READ, observed, rule, 'a natural-decode D_imp arm is missing from stage1.json')
    clauses = {'fact_minus_imp_ge_+0.20': (fact - imp) >= PRED_214_7_SHIFT,
               'imp_vs_you_within_0.30': abs(imp - you) <= PRED_214_7_TOL}
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


def pred_214_8(s1: dict) -> dict:
    """3-unit engine & L27 N3325 tracking: both `no_L29` arms within 0.25 logits of their natural arm, and
    Spearman(L27 shift, delta g_ali) <= -0.80 over the 10 dial rungs + 6 natural-decode arms (design line 305)."""
    rule = ('|nat_fact_no_L29 - nat_trace_fact| <= 0.25 AND |nat_you_no_L29 - nat_trace_you| <= 0.25; '
            'Spearman over 16 arms <= -0.80')
    fact, fact_no = s1['nat'].get('nat_trace_fact'), s1['nat'].get('nat_fact_no_L29')
    you, you_no = s1['nat'].get('nat_trace_you'), s1['nat'].get('nat_you_no_L29')
    corr = s1['l27_correlation']
    observed = _observed(nat_trace_fact=fact, nat_fact_no_L29=fact_no, nat_trace_you=you, nat_you_no_L29=you_no,
                         fact_delta=(None if fact is None or fact_no is None else abs(fact_no - fact)),
                         you_delta=(None if you is None or you_no is None else abs(you_no - you)),
                         spearman_l27=corr['spearman_l27_shift_vs_delta_g_ali'], n_paired=corr['n_paired'])
    if not s1['gates']['gate1']['pass'] or not s1['gates']['gate2']['pass']:
        return status_verdict(UNTESTABLE, observed, rule, 'Gate 1 or Gate 2 failed')
    if fact is None or fact_no is None or you is None or you_no is None:
        return status_verdict(NOT_READ, observed, rule, 'a no_L29 arm is missing from stage1.json')
    if corr['n_paired'] < corr['n_required']:
        return status_verdict(NOT_READ, observed, rule,
                              f'the {corr["n_required"]}-arm L27 correlation is incomplete: {corr["n_paired"]} of '
                              f'{corr["n_required"]} pairs present, missing {corr["missing_arms"]}')
    clauses = {'fact_no_L29_within_0.25': abs(fact_no - fact) <= PRED_214_8_TOL,
               'you_no_L29_within_0.25': abs(you_no - you) <= PRED_214_8_TOL,
               'l27_tracks_negatively_le_-0.80': corr['tracks_negatively']}
    observed['clauses'] = clauses
    return status_verdict(HOLDS if all(clauses.values()) else FAILS, observed, rule)


# ----------------------------------------------------------------------------------------------------------------
# the headline classifier (design lines 319-345)
# ----------------------------------------------------------------------------------------------------------------

H4_ACCOUNTS = 'A shared two-phase partial-zero schedule (H4) accounts for the fact vs. "you" ladder'
POINT_TOLERANCE = ('point-tolerance/directional screen: H4 reproduces the donor ORDER and the endpoint counts, and '
                   'its magnitude is reported with the 214.4(c) miss')
MISSED_MAGNITUDE = 'The tested schedule missed its magnitude bar'
MIXED = 'mixed / inconclusive'


def headline_reading(s1: dict, s2: dict, preds: dict) -> dict:
    """Operational headline readings, in the design's order and with its precedence (design lines 319-345).

    Positive predicates only: a failed conjunction is never read as positive proof, `not read` / `not discriminated`
    never count as a discriminated failure, and a rival is never credited from a collapsed branch.
    """
    gates_1_2 = bool(s1['gates']['gate1']['pass'] and s1['gates']['gate2']['pass'])
    gate3 = bool(s2['gate3']['pass'])
    gates_ok = gates_1_2 and gate3
    p1 = preds['214.1']['holds'] is True
    p2 = preds['214.2']['holds'] is True
    p3 = preds['214.3']['holds'] is True
    p4 = preds['214.4']
    clauses = p4.get('observed', {}).get('clauses') or {}
    a, b, c = clauses.get('a'), clauses.get('b'), clauses.get('c')
    broken_ok = all((s2['main_arms'].get(arm) or {}).get('broken') is not None
                    and broken_within(int(s2['main_arms'][arm]['broken']), BROKEN_LIMIT) for arm in HEADLINE_ARMS)
    complete = not (s2.get('arm_status') or {}).get('unscored_arms')
    donor_gap = _diff(s2, 'trace_resample_fact', 'trace_resample_you')
    trace_you = s2['main_arms'].get('trace_resample_you')
    phase_fy = s2['main_arms'].get('prompt_fact_gen_you')
    mean_you = s2['main_arms'].get('mean_resample_you')
    h3_ci = s2['bootstrap'].get('mean_resample_you-trace_resample_you') or {}
    h3 = {
        'jump': _diff(s2, 'mean_resample_you', 'trace_resample_you'),
        'ci_lo': h3_ci.get('lo'),
        'clears': (_diff(s2, 'mean_resample_you', 'trace_resample_you') is not None
                   and _diff(s2, 'mean_resample_you', 'trace_resample_you') > H3_JUMP
                   and h3_ci.get('lo') is not None and h3_ci['lo'] > 0
                   and _broken_ok(s2, 'mean_resample_you')),
        'note': 'mean pooling changes prefill values, timing, donor identity and covariance at once (design line 327)',
    }
    # Design lines 359-363: a rival may only feed a win when 214.4 failed on (a) or (b) for a DISCRIMINATED reason.
    # A lone (c) magnitude miss is the point-tolerance screen above and is explicitly not a mechanism win.
    discriminated_ab_fail = (p4['verdict'] == FAILS) and ((a is False) or (b is False))
    damaged = p4['verdict'] == DAMAGE
    h1 = {'pred_214_5_holds': preds['214.5']['holds'] is True,
          'clears': preds['214.5']['holds'] is True and discriminated_ab_fail,
          'four_d_vs_shared': s1['diagnostics'],
          'note': 'order-dependent FF->FY->YY decomposition (design line 370)'}
    h2 = {
        'fy_minus_you': (None if phase_fy is None or trace_you is None
                         else phase_fy['clean_google_rate'] - trace_you['clean_google_rate']),
        'half_donor_gap': (0.50 * donor_gap) if donor_gap is not None else None,
        'recode': (None if phase_fy is None or trace_you is None else _worst_case(trace_you, phase_fy)),
        'clears': False,
    }
    if h2['fy_minus_you'] is not None and donor_gap is not None:
        h2['clears'] = (h2['fy_minus_you'] > H2_SHARE * donor_gap and bool(h2['recode'])
                        and _broken_ok(s2, 'prompt_fact_gen_you'))
    checks = {
        'gates_1_to_3': gates_ok, 'complete': complete,
        '214.1': p1, '214.2': p2, '214.3': p3,
        '214.4_a': a, '214.4_b': b, '214.4_c': c,
        'broken_le_0.15_on_headline_arms': broken_ok,
        '214.4_discriminated_fail': p4['verdict'] == FAILS,
        '214.4_discriminated_ab_fail': discriminated_ab_fail,
        '214.4_lone_c_miss': (p4['verdict'] == FAILS and a is True and b is True and c is False),
        '214.4_not_read': p4['verdict'] == NOT_READ,
        '214.4_not_discriminated': p4['verdict'] == NOT_DISCRIMINATED,
        '214.4_damaged': damaged,
        'h3_clears': h3['clears'], 'h1_clears': h1['clears'], 'h2_clears': h2['clears'],
        'any_rival_clears': bool(h3['clears'] or h1['clears'] or h2['clears']),
    }
    notes = []
    if not gates_ok:
        reading, branch = MIXED, 'gates_failed'
        notes.append('Gates 1-3 did not all pass; no headline may be read.')
    elif p1 and p2 and p3 and a and b and c and broken_ok:
        reading, branch = H4_ACCOUNTS, 'h4_accounts'
    elif p1 and p2 and p3 and a is True and b is True and c is False and broken_ok and not damaged:
        reading, branch = POINT_TOLERANCE, 'point_tolerance_screen'
        notes.append('214.4(a) and (b) hold, (c) alone misses: the point-tolerance screen is the verdict, the '
                     'magnitude miss is reported beside it, and the mechanism headline cannot also fire.')
    elif p1 and discriminated_ab_fail and checks['any_rival_clears']:
        reading, branch = MISSED_MAGNITUDE, 'missed_magnitude'
        notes.append('214.4 failed on (a) or (b) for a discriminated reason and at least one rival cleared its '
                     'positive bar; this is not "H4 beaten by ...", which the controls cannot identify.')
    elif p1 and checks['214.4_lone_c_miss']:
        reading, branch = MIXED, 'mixed'
        notes.append('214.4 misses on (c) alone without (a)+(b) fully passing; the magnitude miss is reported and '
                     'no rival may be credited from it.')
    else:
        reading, branch = MIXED, 'mixed'
        if p4['verdict'] in (NOT_READ, NOT_DISCRIMINATED, DAMAGE):
            notes.append(f'214.4 is `{p4["verdict"]}` ({p4.get("reason")}); it may not be read as a failure and no '
                         f'rival may be credited from this branch.')
        if not p1:
            notes.append('214.1 failed: the batch-1 fact vs. you replication failed; the mechanism headline is '
                         'withdrawn.')
    return {'reading': reading, 'branch': branch, 'checks': checks, 'rivals': {'h3': h3, 'h1_directional': h1,
                                                                              'h2': h2},
            'notes': notes}


# ----------------------------------------------------------------------------------------------------------------
# top level
# ----------------------------------------------------------------------------------------------------------------

def load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'corrupt JSONL row at {path}, line {number}: {exc.msg}') from None
            if not isinstance(row, dict):
                raise ValueError(f'corrupt JSONL row at {path}, line {number}: expected a JSON object')
            size = row.get('batch_size')
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f'JSONL row at {path}, line {number} is missing a valid positive batch_size')
            rows.append(row)
    return rows


def json_default(obj):
    """`default=` handler for numpy scalars: the runner and the scorer both write JSON that must be plain."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'object of type {type(obj).__name__} is not JSON serializable')


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=json_default)


def score(stage0: dict | None = None, stage1: dict | None = None, stage2_rows: list[dict] | None = None,
          run: dict | None = None, stage0_sha256: str | None = None) -> dict:
    """Everything score.json holds, from the saved stage files. Each stage may be absent.

    Every artifact handed in is bound to the same pins: Stage 1 and Stage 2 from different Stage 0 bytes, design
    bytes or code are refused rather than averaged, and when the caller knows the Stage 0 file's digest the pins must
    name those exact bytes.
    """
    if stage1 is not None and run is not None and (stage1.get('pins') or None) != (run.get('pins') or None):
        raise ValueError('stage1.json and run.json come from different inputs: their pins differ '
                         f'(stage1 {stage1.get("pins")!r}, run {run.get("pins")!r})')
    if stage0_sha256 is not None:
        for name, obj in (('stage1.json', stage1), ('run.json', run)):
            if obj is None:
                continue
            named = (obj.get('pins') or {}).get('stage0_sha256')
            if named != stage0_sha256:
                raise ValueError(f'{name} was produced from Stage 0 bytes {named!r}, but the file handed to the '
                                 f'scorer is {stage0_sha256!r}')
    pinning = dial_pinning(stage0) if stage0 else None
    gate0_obj = gate0(stage0) if stage0 else {'pass': False, 'checks': {}, 'reason': 'no stage0.json'}
    gate3b_obj = gate3b(stage0)
    s1m = stage1_metrics(stage1, pinning) if stage1 else None
    s2 = stage2_metrics(stage2_rows or [], run or {}, pinning, stage0) if stage2_rows is not None else None
    preds: dict[str, dict] = {}
    preds['214.1'] = pred_214_1(s2) if s2 else status_verdict(NOT_READ, {}, 'needs Stage 2', 'no Stage 2 rows')
    preds['214.2'] = pred_214_2(pinning, gate0_obj, gate3b_obj)
    preds['214.3'] = pred_214_3(s1m, pinning) if s1m else status_verdict(NOT_READ, {}, 'needs Stage 1',
                                                                         'no stage1.json')
    preds['214.4'] = pred_214_4(s2, pinning) if s2 else status_verdict(NOT_READ, {}, 'needs Stage 2', 'no Stage 2 rows')
    preds['214.5'] = pred_214_5(s2) if s2 else status_verdict(NOT_READ, {}, 'needs Stage 2', 'no Stage 2 rows')
    preds['214.6'] = pred_214_6(s2) if s2 else status_verdict(NOT_READ, {}, 'needs Stage 2', 'no Stage 2 rows')
    preds['214.7'] = pred_214_7(s1m) if s1m else status_verdict(NOT_READ, {}, 'needs Stage 1', 'no stage1.json')
    preds['214.8'] = pred_214_8(s1m) if s1m else status_verdict(NOT_READ, {}, 'needs Stage 1', 'no stage1.json')
    headline = headline_reading(s1m, s2, preds) if (s1m and s2) else {
        'reading': MIXED, 'branch': 'incomplete', 'checks': {}, 'notes': ['a stage is missing']}
    return {
        'part': PART, 'design': DESIGN_VERSION, 'design_commit': DESIGN_COMMIT,
        'stage0': pinning['projection_table'] if pinning else None,
        'pinning': {k: v for k, v in (pinning or {}).items() if k != 'projection_table'},
        'paste_block': paste_block(pinning) if pinning else None,
        'gates': {'gate0': gate0_obj, 'gate3b': gate3b_obj,
                  'gate1': (s1m or {}).get('gates', {}).get('gate1'),
                  'gate2': (s1m or {}).get('gates', {}).get('gate2'),
                  'gate3': (s2 or {}).get('gate3')},
        'stage1': s1m, 'stage2': s2, 'predictions': preds, 'headlines': headline,
    }


def verdict_table(result: dict) -> str:
    lines = ['| Prediction | Verdict | Observed | Rule |', '|---|---|---|---|']
    for key, obj in result['predictions'].items():
        obs = json.dumps(obj['observed'], ensure_ascii=False, default=json_default)
        reason = f" ({obj['reason']})" if obj.get('reason') else ''
        lines.append(f"| {key} | {obj['verdict']}{reason} | `{obs[:320]}` | {obj['rule']} |")
    gates = result['gates']
    for name in ('gate0', 'gate3b', 'gate1', 'gate2', 'gate3'):
        gate = gates.get(name)
        if gate:
            lines.append(f"{name}: {'PASS' if gate.get('pass') else 'FAIL'} {gate.get('checks')}")
    if result.get('stage2'):
        lines.append(f"Stage 2 scored rows by actual batch size: {result['stage2']['batch_size_histogram']}")
        lines.append(f"Arm status: {result['stage2']['arm_status']['arms']}")
    lines.append(f"Headline reading: {result['headlines']['reading']}")
    for note in result['headlines'].get('notes') or []:
        lines.append(f"  note: {note}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 214 scorer (reads saved files; never runs the model)')
    parser.add_argument('--stage0', default=None, help='results/part214/stage0/stage0.json')
    parser.add_argument('--stage1', default=None, help='results/part214/stage1/stage1.json')
    parser.add_argument('--stage2-dir', default=None, help='directory holding stage2.jsonl and run.json')
    parser.add_argument('--out', default=None, help='score.json path (default: <stage2-dir>/score.json)')
    args = parser.parse_args(argv)
    stage0 = load_json(args.stage0) if args.stage0 else None
    stage0_sha256 = hashlib.sha256(Path(args.stage0).read_bytes()).hexdigest() if args.stage0 else None
    stage1 = load_json(args.stage1) if args.stage1 else None
    rows = run = None
    if args.stage2_dir:
        directory = Path(args.stage2_dir)
        rows = load_jsonl(directory / 'stage2.jsonl')
        run = load_json(directory / 'run.json')
    try:
        result = score(stage0, stage1, rows, run, stage0_sha256=stage0_sha256)
    except ValueError as exc:
        raise SystemExit(f'refusing to score: {exc}') from None
    out = Path(args.out) if args.out else (Path(args.stage2_dir) / 'score.json' if args.stage2_dir else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dumps(result) + '\n', encoding='utf-8')
    if result.get('paste_block'):
        print(result['paste_block'], flush=True)
        print('', flush=True)
    print(verdict_table(result), flush=True)
    return result


if __name__ == '__main__':
    main()
