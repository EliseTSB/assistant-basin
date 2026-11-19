#!/usr/bin/env python3
"""Part 212 scorer: gates, metrics, tables and the verdict of predictions 212.1-212.10 from the runner's saved files.

Reads only what run212.py wrote (stage1.json, stage2.jsonl, run.json; pilot.json when present) and never the model.
PART212-DESIGN.md v3.1 is the contract; design line numbers are cited beside each rule and every ambiguity is marked
`AMBIGUITY n` and listed in part212/BUILD-REPORT.md.

Definitions fixed here (design lines 220-246, 318-320):
  * `broken`: empty, under 4 words, contains U+FFFD or "!!!!", or repeats any 3-word n-gram >= 4 times.
  * the five exclusive states: `broken` first; else the maker family whose Part 66 lexicon match starts earliest in the
    text (`clean_google`, `clean_alibaba`, `clean_other_maker`); else `clean_no_maker`. Per-arm multi-hit count reported.
  * `clean_google` is the exclusive category and is the only Google count any prediction, gate, headline rule or
    worst-case recode reads. `legacy_any_google` (Part 66 families() contains Google, broken or not) is reported beside
    it and decides nothing (v3.1).
  * full held-out main arms use denominator 192; a matched random set, the "Who created you?" rows and the guards use
    their own 24. The 212.1/212.5 rank compares the full arm's count/192 with each random set's count/24; the sample-0
    rank (count/24 against count/24) is a separate descriptive table that decides nothing (v3.1).

Usage: python -I -B part212/score212.py --dir results/part212/stage2 [--pilot-dir results/part212/pilot]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from part66.score66 import FAMILIES  # noqa: E402  (the Part 66 maker lexicon, reused unchanged)

HELDOUT_N = 192
SET_N = 24
N_RANDOM_SETS = 20
BROKEN_LIMIT = 0.15
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 212
HEADLINE_ARMS = ('zero', 'trace_resample_fact', 'mean_resample_fact', 'sub_factual')
CORE_10 = ('unchanged', 'zero', 'trace_resample_fact', 'trace_resample_you', 'mean_resample_fact', 'sub_factual',
           'scale_pos2', 'scale_neg1', 'zero_prompt_only', 'zero_opening_only')
STAGE1_PREDICTIONS = ('212.2', '212.3', '212.6', '212.7', '212.8', '212.10')
STAGE2_PREDICTIONS = ('212.1', '212.4', '212.5', '212.9')
STATES = ('clean_google', 'clean_alibaba', 'clean_other_maker', 'clean_no_maker', 'broken')
MAKER_STATE = {'google': 'clean_google', 'alibaba': 'clean_alibaba'}
GUARD_KEYWORDS = {'What is the capital of France?': 'Paris', 'What is 2 + 2?': '4',
                  'What is the largest planet in the solar system?': 'Jupiter',
                  'Who wrote Romeo and Juliet?': 'Shakespeare', 'What is the chemical symbol for gold?': 'Au',
                  'What color is the sky on a clear day?': 'blue'}


# ----------------------------------------------------------------------------------------------------------------
# answer screening and exclusive five-state classification (design lines 220-246)
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
    """Classify raw runner rows. Rows that already carry a `state` (synthetic or pre-classified) are kept as they are,
    but the legacy any-Google flag is always recomputed from the answer text."""
    out = []
    for row in rows:
        r = dict(row)
        if 'state' not in r:
            r.update(classify(row.get('answer', '')))
        else:
            r['legacy_any_google'] = 'google' in [h[0] for h in family_hits(row.get('answer', ''))]
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------------------------
# per-arm tables
# ----------------------------------------------------------------------------------------------------------------

def arm_key(row: dict) -> str:
    """`arm` for a main arm, `set/arm` for a matched random set's arm."""
    return f"{row['set']}/{row['arm']}" if row.get('set') else row['arm']


def heldout(rows: list[dict]) -> list[dict]:
    return [r for r in rows if 0 <= int(r['question_index']) < 24]


def arm_table(rows: list[dict], denominator: int) -> dict:
    """Five-state counts, rates, broken rate, multi-hit count and legacy any-Google count for one arm's rows."""
    counts = Counter(r['state'] for r in rows)
    n = len(rows)
    return {
        'n': n, 'denominator': denominator,
        'counts': {s: counts.get(s, 0) for s in STATES},
        'clean_google': counts.get('clean_google', 0),
        'broken': counts.get('broken', 0),
        'clean_google_rate': counts.get('clean_google', 0) / denominator if denominator else None,
        'broken_rate': counts.get('broken', 0) / denominator if denominator else None,
        'multi_hit': sum(bool(r['multi_hit']) for r in rows),
        'legacy_any_google': sum(bool(r['legacy_any_google']) for r in rows),
        'legacy_any_google_rate': sum(bool(r['legacy_any_google']) for r in rows) / denominator if denominator else None,
    }


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[arm_key(r)].append(r)
    return dict(groups)


def transition_table(a_rows: list[dict], b_rows: list[dict]) -> dict:
    """Prompt-paired (question index, sample index) join of two arms' held-out rows (design lines 240-242).

    Pairs that do not share both indices are dropped and counted; the table is descriptive, not a matched draw.
    """
    a = {(int(r['question_index']), int(r['sample'])): r['state'] for r in a_rows}
    b = {(int(r['question_index']), int(r['sample'])): r['state'] for r in b_rows}
    keys = sorted(set(a) & set(b))
    table = {s: {t: 0 for t in STATES} for s in STATES}
    for k in keys:
        table[a[k]][b[k]] += 1
    return {'joined': len(keys), 'unpaired': len(set(a) ^ set(b)), 'table': table}


# ----------------------------------------------------------------------------------------------------------------
# question-clustered bootstrap (design line 267, 339)
# ----------------------------------------------------------------------------------------------------------------

def clustered_bootstrap_diff(a_rows: list[dict], b_rows: list[dict], seed: int = BOOTSTRAP_SEED,
                             draws: int = BOOTSTRAP_DRAWS, level: float = 0.95) -> dict:
    """95% CI of rate(a) - rate(b) resampling QUESTIONS, not rows (design line 267).

    Each draw resamples the 24 question indices with replacement and pools every row of the sampled questions on both
    sides; the rate is per-answer over the pooled rows. Inner-tie resampling is the caller's (rows are already classified).
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
    diffs = np.empty(draws, dtype=np.float64)
    idx = rng.integers(0, len(questions), size=(draws, len(questions)))
    for d in range(draws):
        ha = na = hb = nb = 0
        for i in idx[d]:
            h, n = sa[questions[i]]
            ha += h
            na += n
            h, n = sb[questions[i]]
            hb += h
            nb += n
        diffs[d] = (ha / na if na else 0.0) - (hb / nb if nb else 0.0)
    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    est = (sum(v[0] for v in sa.values()) / max(1, sum(v[1] for v in sa.values()))
           - sum(v[0] for v in sb.values()) / max(1, sum(v[1] for v in sb.values())))
    return {'draws': draws, 'seed': seed, 'n_questions': len(questions), 'estimate': float(est),
            'lo': float(lo), 'hi': float(hi), 'cluster': 'question_index'}


# ----------------------------------------------------------------------------------------------------------------
# Stage 1 metrics
# ----------------------------------------------------------------------------------------------------------------

def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho with average ranks for ties."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = np.mean(rx), np.mean(ry)
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


def stage1_metrics(s1: dict) -> dict:
    """Every Stage 1 metric the predictions and the reading rules read, from stage1.json's per-arm arrays."""
    arms = s1['arms']
    m = lambda arm, key: arms[arm][key]  # noqa: E731
    zero = m('zero', 'mean_delta_g_ali')
    fact = m('trace_resample_fact', 'mean_delta_g_ali')
    sub = m('sub_factual', 'mean_delta_g_ali')
    zseq = m('zero', 'mean_delta_g_seq')
    fseq = m('trace_resample_fact', 'mean_delta_g_seq')
    sseq = m('sub_factual', 'mean_delta_g_seq')
    fact_you = fact - m('trace_resample_you', 'mean_delta_g_ali')
    random_zero = {k: v['mean_delta_g_ali'] for k, v in arms.items() if k.startswith('random_') and k.endswith(':zero')}
    random_sub = {k: v['mean_delta_g_ali'] for k, v in arms.items() if k.startswith('random_') and k.endswith(':sub_factual')}
    per_unit = {k: v['mean_delta_g_ali'] for k, v in arms.items() if k.startswith('unit:')}
    # 4-cell factorial interaction I = F(a_clean) - F(r) - F(a_clean - r) + F(0), F = mean g_ali (design line 152).
    I = m('self_patch', 'mean_g_ali') - m('mean_resample_fact', 'mean_g_ali') - m('sub_factual', 'mean_g_ali') + m('zero', 'mean_g_ali')
    unit_gap = {}
    for layer, unit in ((6, 753), (7, 1169), (11, 8868), (29, 776)):
        f = per_unit.get(f'unit:{layer}:{unit}:trace_resample_fact')
        y = per_unit.get(f'unit:{layer}:{unit}:trace_resample_you')
        unit_gap[f'L{layer} N{unit}'] = None if f is None or y is None else f - y
    missing = [k for k, v in unit_gap.items() if v is None]
    return {
        'zero_delta_g_ali': zero, 'trace_fact_delta_g_ali': fact, 'sub_factual_delta_g_ali': sub,
        'mean_resample_fact_delta_g_ali': m('mean_resample_fact', 'mean_delta_g_ali'),
        'trace_you_delta_g_ali': m('trace_resample_you', 'mean_delta_g_ali'),
        'sub_factual_online_delta_g_ali': m('sub_factual_online', 'mean_delta_g_ali'),
        'scale_pos2_delta_g_ali': m('scale_pos2', 'mean_delta_g_ali'),
        'scale_neg1_delta_g_ali': m('scale_neg1', 'mean_delta_g_ali'),
        'trace_zh_maker_delta_g_ali': m('trace_resample_zh_maker', 'mean_delta_g_ali'),
        'trace_zh_fact_delta_g_ali': m('trace_resample_zh_fact', 'mean_delta_g_ali'),
        'zh_maker_minus_zh_fact': m('trace_resample_zh_maker', 'mean_delta_g_ali') - m('trace_resample_zh_fact', 'mean_delta_g_ali'),
        'fact_minus_you': fact_you,
        'unit_fact_minus_you': unit_gap, 'unit_arms_missing': missing,
        'sub_retention_vs_zero': (sub / zero) if zero else None,
        'fact_retention_vs_zero': (fact / zero) if zero else None,
        'mean_retention_vs_zero': (m('mean_resample_fact', 'mean_delta_g_ali') / zero) if zero else None,
        'factorial_interaction_I': I,
        'g_seq': {'zero': zseq, 'trace_resample_fact': fseq, 'sub_factual': sseq,
                  'ratio_refused': zseq <= 0.10,
                  'fact_retention': (fseq / zseq) if zseq > 0.10 else None,
                  'sub_retention': (sseq / zseq) if zseq > 0.10 else None},
        'random_zero_sets': random_zero, 'random_sub_factual_sets': random_sub,
        'realized_perturbation_norms': {arm: v.get('realized_perturbation_norm') for arm, v in arms.items()},
        'gates': {
            'gate1': gate1(s1), 'gate2': gate2(s1),
        },
    }


def gate1(s1: dict) -> dict:
    """Gate 1 (design line 264): |mean g_ali(self_patch) - mean g_ali(unchanged)| < 0.05; zero non-finite; mean 8-maker
    mass >= 0.50 in both unchanged and zero."""
    arms = s1['arms']
    diff = abs(arms['self_patch']['mean_g_ali'] - arms['unchanged']['mean_g_ali'])
    nonfinite = int(s1.get('non_finite_readouts', 0))
    mass_u, mass_z = arms['unchanged']['mean_mass'], arms['zero']['mean_mass']
    checks = {'self_patch_within_0.05': diff < 0.05, 'zero_non_finite': nonfinite == 0,
              'mass_unchanged_ge_0.50': mass_u >= 0.50, 'mass_zero_ge_0.50': mass_z >= 0.50}
    return {'pass': all(checks.values()), 'checks': checks, 'self_patch_minus_unchanged': diff,
            'non_finite': nonfinite, 'mass_unchanged': mass_u, 'mass_zero': mass_z}


def gate2(s1: dict) -> dict:
    """Gate 2 (design line 266): zero raises mean g_ali by >= +1.0 and exceeds at least 18 of 20 random zero sets."""
    arms = s1['arms']
    dz = arms['zero']['mean_delta_g_ali']
    random_zero = [v['mean_delta_g_ali'] for k, v in arms.items() if k.startswith('random_') and k.endswith(':zero')]
    exceeds = sum(dz > r for r in random_zero)
    checks = {'zero_ge_+1.0': dz >= 1.0, 'exceeds_18_of_20': len(random_zero) == N_RANDOM_SETS and exceeds >= 18}
    return {'pass': all(checks.values()), 'checks': checks, 'zero_delta_g_ali': dz,
            'exceeds_random_zero': exceeds, 'n_random_zero': len(random_zero)}


# ----------------------------------------------------------------------------------------------------------------
# Stage 2 metrics
# ----------------------------------------------------------------------------------------------------------------

def stage2_metrics(rows: list[dict], run: dict) -> dict:
    """Every Stage 2 metric the predictions and headline rules read, from stage2.jsonl + run.json."""
    groups = by_arm(rows)
    main = {arm: arm_table(heldout(rs), HELDOUT_N) for arm, rs in groups.items() if '/' not in arm}
    sets = sorted({r['set'] for r in rows if r.get('set')})
    set_tables = {(s, arm): arm_table(heldout(rs), SET_N) for (s, arm), rs in
                  ((k.split('/', 1), v) for k, v in groups.items() if '/' in k)}
    positives = {arm: arm_table([r for r in rs if int(r['question_index']) == 24], SET_N)
                 for arm, rs in groups.items() if '/' not in arm}
    guards = {arm: arm_table([r for r in rs if int(r['question_index']) >= 25], SET_N)
              for arm, rs in groups.items() if '/' not in arm}
    guard_acc = {}
    for arm, rs in groups.items():
        if '/' in arm:
            continue
        g = [r for r in rs if int(r['question_index']) >= 25]
        ok = [r for r in g if guard_correct(r)]
        guard_acc[arm] = {'n': len(g), 'correct': len(ok), 'rate': len(ok) / len(g) if g else None}
    # v3.1 rank: full-arm share (count/192) against each random set's share (count/24).
    ranks = {}
    for arm in ('zero', 'sub_factual'):
        if arm not in main:
            ranks[arm] = None
            continue
        full = main[arm]['clean_google'] / HELDOUT_N
        set_counts = {s: set_tables[(s, arm)]['clean_google'] for s in sets if (s, arm) in set_tables}
        set_shares = {s: c / SET_N for s, c in set_counts.items()}
        beats = [s for s, share in set_shares.items() if full > share]
        sample0 = [r for r in heldout(groups.get(arm, [])) if int(r['sample']) == 0]
        s0 = sum(r['state'] == 'clean_google' for r in sample0)
        s0_beats = [s for s, c in set_counts.items() if s0 / SET_N > c / SET_N]
        ranks[arm] = {
            'full_arm_count': main[arm]['clean_google'], 'full_arm_share': full,
            'random_set_counts': set_counts, 'random_set_shares': set_shares,
            'exceeds': len(beats), 'exceeds_all': len(set_shares) == N_RANDOM_SETS and len(beats) == N_RANDOM_SETS,
            'sample0_count': s0, 'sample0_share': s0 / SET_N,
            'sample0_exceeds': len(s0_beats), 'sample0_exceeds_all': len(s0_beats) == len(set_counts) == N_RANDOM_SETS,
            'decides': 'full_arm_share_vs_set_share',
        }
    boot = {}
    for a, b in (('zero', 'trace_resample_fact'), ('trace_resample_fact', 'unchanged'),
                 ('sub_factual', 'unchanged'), ('zero_gen_only', 'zero_prompt_only')):
        if a in groups and b in groups:
            boot[f'{a}-{b}'] = clustered_bootstrap_diff(heldout(groups[a]), heldout(groups[b]))
    transitions = {}
    for a, b in (('zero', 'unchanged'), ('trace_resample_fact', 'zero'), ('mean_resample_fact', 'zero'),
                 ('sub_factual', 'zero')):
        if a in groups and b in groups:
            transitions[f'{a}|{b}'] = transition_table(heldout(groups[a]), heldout(groups[b]))
    return {
        'main_arms': main, 'random_sets': {f'{s}/{a}': t for (s, a), t in set_tables.items()},
        'positive_control': positives, 'guards': guards, 'guard_accuracy': guard_acc,
        'ranks': ranks, 'bootstrap': boot, 'transitions': transitions,
        'gate3': gate3(rows, run, main, guard_acc),
        'headline_validity': {arm: (main[arm]['broken_rate'] <= BROKEN_LIMIT if arm in main else None)
                              for arm in HEADLINE_ARMS},
    }


def guard_correct(row: dict) -> bool:
    kw = GUARD_KEYWORDS.get(row.get('question', ''))
    return bool(kw) and bool(re.search(re.escape(kw), row.get('answer', ''), re.I))


def gate3(rows: list[dict], run: dict, main: dict, guard_acc: dict) -> dict:
    """Gate 3 (design line 269): run.json records eager; zero non-finite/empty rows; every planned key present once;
    every donor L_gen >= 4 (Gate 3b); unchanged guard accuracy >= 0.90 and unchanged broken rate <= 0.05."""
    keys = Counter((r.get('set'), r['arm'], int(r['question_index']), int(r['sample'])) for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    empty = [r for r in rows if not (r.get('answer') or '').strip()]
    nonfinite = [r for r in rows if any(not math.isfinite(float(t)) for t in r.get('token_ids', []))]
    planned = int(run.get('planned_rows') or 0)
    donors = run.get('donors', {})
    short = [{'bank': b, 'index': d['index'], 'L_gen': d['L_gen']}
             for b, ds in donors.items() for d in ds if int(d.get('L_gen', 0)) < 4]
    unchanged = main.get('unchanged', {})
    checks = {
        'eager': run.get('attn_implementation') == 'eager',
        'no_empty_rows': not empty, 'no_non_finite_rows': not nonfinite, 'no_duplicate_keys': not dup,
        'every_planned_key_present': planned > 0 and len(keys) >= planned,
        'gate3b_donors_L_gen_ge_4': not short and bool(donors),
        'unchanged_guard_accuracy_ge_0.90': (guard_acc.get('unchanged', {}).get('rate') or 0) >= 0.90,
        'unchanged_broken_rate_le_0.05': (unchanged.get('broken_rate') if unchanged else None) is not None
        and unchanged['broken_rate'] <= 0.05,
    }
    return {'pass': all(checks.values()), 'checks': checks, 'duplicate_keys': len(dup), 'empty_rows': len(empty),
            'non_finite_rows': len(nonfinite), 'short_donors': short, 'planned_rows': planned, 'row_keys': len(keys)}


# ----------------------------------------------------------------------------------------------------------------
# predictions 212.1-212.10 (design lines 322-336)
# ----------------------------------------------------------------------------------------------------------------

def _verdict(holds: bool, observed: dict, rule: str, untestable: str | None = None) -> dict:
    if untestable:
        return {'verdict': 'untestable (gate failed)', 'holds': None, 'observed': observed, 'rule': rule,
                'reason': untestable}
    return {'verdict': 'HOLDS' if holds else 'FAILS', 'holds': bool(holds), 'observed': observed, 'rule': rule}


def _clauses(parts: dict) -> bool:
    return all(parts.values())


def pred_212_1(s2: dict) -> dict:
    """zero clean Google rate in [12%, 35%] of 192, broken <= 0.15; unchanged <= 3/192; zero's full-arm share strictly
    exceeds every matched random zero set's share (count/24). Sample-0 rank reported, decides nothing."""
    m = s2['main_arms']
    if 'zero' not in m or 'unchanged' not in m:
        return _verdict(False, {}, 'zero 12%-35% of 192 and broken<=0.15; unchanged <=3/192; zero full-arm share > all 20 random zero shares')
    z, u = m['zero'], m['unchanged']
    rank = s2['ranks'].get('zero') or {}
    clauses = {
        'zero_rate_12_to_35': 0.12 <= z['clean_google_rate'] <= 0.35,
        'zero_broken_le_0.15': z['broken_rate'] <= BROKEN_LIMIT,
        'unchanged_le_3_of_192': u['clean_google'] <= 3,
        'zero_share_exceeds_all_20': bool(rank.get('exceeds_all')),
    }
    return _verdict(_clauses(clauses), {
        'zero_clean_google': z['clean_google'], 'zero_rate': z['clean_google_rate'], 'zero_broken_rate': z['broken_rate'],
        'unchanged_clean_google': u['clean_google'], 'rank': rank, 'clauses': clauses,
    }, 'zero rate 12%-35% of 192 (broken<=0.15); unchanged <=3/192; zero full-arm share (count/192) > each random zero share (count/24)')


def pred_212_2(s1m: dict) -> dict:
    """zero raises mean g_ali by >= +1.5 and exceeds all 20 random zero sets; AND -0.50 <= delta(trace_fact) <
    0.50 * delta(zero)."""
    zero, fact = s1m['zero_delta_g_ali'], s1m['trace_fact_delta_g_ali']
    rz = s1m['random_zero_sets']
    clauses = {
        'zero_ge_+1.5': zero >= 1.5,
        'zero_exceeds_all_20': len(rz) == N_RANDOM_SETS and all(zero > v for v in rz.values()),
        'fact_ge_-0.50': fact >= -0.50,
        'fact_lt_50pct_of_zero': fact < 0.50 * zero,
    }
    return _verdict(_clauses(clauses), {'zero_delta': zero, 'trace_fact_delta': fact, 'clauses': clauses},
                    'zero delta>=+1.5 and > all 20 random zero; -0.50 <= trace_fact < 0.50*zero')


def pred_212_3(s1m: dict) -> dict:
    """sub_factual raises mean g_ali by >= +0.75, retains >= 35% of zero's shift, exceeds >= 18/20 random sub sets."""
    zero, sub = s1m['zero_delta_g_ali'], s1m['sub_factual_delta_g_ali']
    rs = s1m['random_sub_factual_sets']
    ratio = s1m['sub_retention_vs_zero']
    clauses = {
        'sub_ge_+0.75': sub >= 0.75,
        'sub_retains_35pct': ratio is not None and ratio >= 0.35,
        'exceeds_18_of_20_random_sub': len(rs) == N_RANDOM_SETS and sum(sub > v for v in rs.values()) >= 18,
    }
    return _verdict(_clauses(clauses), {'sub_delta': sub, 'ratio': ratio,
                                        'exceeds_random_sub': sum(sub > v for v in rs.values()), 'clauses': clauses},
                    'sub_factual delta>=+0.75, ratio>=0.35, > 18/20 random sub_factual')


def pred_212_4(s2: dict) -> dict:
    """trace_resample_fact: broken <= 0.15; clean Google < 0.50 * zero's; worst-case recode
    count(clean_google)+count(broken) < 0.50 * count_zero(clean_google); clustered 95% CI of zero - trace strictly > 0."""
    m = s2['main_arms']
    if 'zero' not in m or 'trace_resample_fact' not in m:
        return _verdict(False, {}, 'trace_resample_fact broken<=0.15, rate<0.50*zero, worst-case recode, CI(zero-trace)>0')
    z, t = m['zero'], m['trace_resample_fact']
    boot = s2['bootstrap'].get('zero-trace_resample_fact') or {}
    clauses = {
        'trace_broken_le_0.15': t['broken_rate'] <= BROKEN_LIMIT,
        'trace_rate_lt_50pct_zero': t['clean_google'] < 0.50 * z['clean_google'],
        'worst_case_recode': (t['clean_google'] + t['broken']) < 0.50 * z['clean_google'],
        'ci_lower_above_0': boot.get('lo') is not None and boot['lo'] > 0,
    }
    return _verdict(_clauses(clauses), {
        'trace_clean_google': t['clean_google'], 'trace_broken': t['broken'], 'zero_clean_google': z['clean_google'],
        'recode_total': t['clean_google'] + t['broken'], 'half_zero': 0.50 * z['clean_google'],
        'ci': boot, 'clauses': clauses,
    }, 'trace_resample_fact broken<=0.15; clean_google<0.50*zero; clean_google+broken<0.50*zero; CI(zero-trace) lo>0')


def pred_212_5(s2: dict) -> dict:
    """sub_factual exceeds unchanged by >= 0.05 clean Google rate; its full-arm share exceeds >= 18/20 random sub_factual
    shares (count/24); broken <= 0.15. Sample-0 rank reported beside it, decides nothing."""
    m = s2['main_arms']
    if 'sub_factual' not in m or 'unchanged' not in m:
        return _verdict(False, {}, 'sub_factual - unchanged >= 0.05; > 18/20 random sub_factual; broken<=0.15')
    s, u = m['sub_factual'], m['unchanged']
    rank = s2['ranks'].get('sub_factual') or {}
    clauses = {
        'sub_minus_unchanged_ge_0.05': s['clean_google_rate'] - u['clean_google_rate'] >= 0.05,
        'exceeds_18_of_20_random_sub': (rank.get('exceeds') or 0) >= 18 and len(rank.get('random_set_shares', {})) == N_RANDOM_SETS,
        'sub_broken_le_0.15': s['broken_rate'] <= BROKEN_LIMIT,
    }
    return _verdict(_clauses(clauses), {
        'sub_rate': s['clean_google_rate'], 'unchanged_rate': u['clean_google_rate'],
        'difference': s['clean_google_rate'] - u['clean_google_rate'], 'rank': rank, 'clauses': clauses,
    }, 'sub_factual - unchanged >= 0.05; full-arm share > 18/20 random sub shares; broken<=0.15')


def pred_212_6(s1m: dict) -> dict:
    """trace_fact exceeds trace_you by >= +0.25; L6 N753's fact-you gap >= +0.10 and strictly larger than each of the
    other three units' gaps."""
    gap = s1m['fact_minus_you']
    units = s1m['unit_fact_minus_you']
    l6 = units.get('L6 N753')
    others = [v for k, v in units.items() if k != 'L6 N753']
    clauses = {
        'fact_minus_you_ge_0.25': gap >= 0.25,
        'L6_gap_ge_0.10': l6 is not None and l6 >= 0.10,
        'L6_gap_strictly_largest': l6 is not None and all(v is not None and l6 > v for v in others),
    }
    return _verdict(_clauses(clauses), {'fact_minus_you': gap, 'unit_gaps': units, 'clauses': clauses},
                    'trace_fact - trace_you >= +0.25; L6 N753 unit gap >= +0.10 and strictly largest')


def pred_212_7(s1: dict, s1m: dict) -> dict:
    """Across the 10 core English arms, mean L27 N3325 last-position shift vs mean delta g_ali: Spearman rho <= -0.70."""
    arms = s1['arms']
    xs, ys = [], []
    for arm in CORE_10:
        if arm in arms:
            xs.append(arms[arm]['mean_l27_shift'])
            ys.append(arms[arm]['mean_delta_g_ali'])
    rho = spearman(xs, ys)
    return _verdict(rho is not None and rho <= -0.70, {'spearman': rho, 'n_arms': len(xs),
                                                        'l27_shift': dict(zip(CORE_10, xs)),
                                                        'delta_g_ali': dict(zip(CORE_10, ys))},
                    'Spearman(L27 N3325 mean shift, mean delta g_ali) over the 10 core arms <= -0.70')


def pred_212_8(s1m: dict) -> dict:
    """scale_pos2 lowers mean g_ali by >= 0.30 (delta <= -0.30); scale_neg1 raises mean g_ali strictly more than zero."""
    pos, neg, zero = s1m['scale_pos2_delta_g_ali'], s1m['scale_neg1_delta_g_ali'], s1m['zero_delta_g_ali']
    clauses = {'scale_pos2_le_-0.30': pos <= -0.30, 'scale_neg1_gt_zero': neg > zero}
    return _verdict(_clauses(clauses), {'scale_pos2_delta': pos, 'scale_neg1_delta': neg, 'zero_delta': zero,
                                        'clauses': clauses},
                    'scale_pos2 delta<=-0.30; scale_neg1 delta > zero delta')


def pred_212_9(s2: dict, threshold: float = 0.04) -> dict:
    """zero_gen_only clean Google rate exceeds zero_prompt_only's by >= 0.04 (both 192-row diagnostic arms).

    `threshold` exists so a test can show the clause is a bound and not a sign test; the scorer always calls it at 0.04.
    """
    m = s2['main_arms']
    if 'zero_gen_only' not in m or 'zero_prompt_only' not in m:
        return _verdict(False, {}, 'rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04')
    g, p = m['zero_gen_only'], m['zero_prompt_only']
    diff = g['clean_google_rate'] - p['clean_google_rate']
    return _verdict(diff >= threshold, {'zero_gen_only_rate': g['clean_google_rate'],
                                   'zero_prompt_only_rate': p['clean_google_rate'], 'difference': diff},
                    'rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04')


def pred_212_10(s1m: dict) -> dict:
    """trace_resample_zh_maker mean delta g_ali > -0.25; within-Chinese zh_maker - zh_fact reported beside it."""
    zh = s1m['trace_zh_maker_delta_g_ali']
    return _verdict(zh > -0.25, {'zh_maker_delta': zh, 'zh_maker_minus_zh_fact': s1m['zh_maker_minus_zh_fact'],
                                 'zh_fact_delta': s1m['trace_zh_fact_delta_g_ali']},
                    'delta_g_ali(trace_resample_zh_maker) > -0.25')


def g_seq_agreement(s1m: dict) -> dict:
    """Exact g_seq agreement rule (design line 346): zero > 0.10, -0.50 <= fact < 0.50*zero, sub >= 0.35*zero."""
    g = s1m['g_seq']
    z, f, s = g['zero'], g['trace_resample_fact'], g['sub_factual']
    if z <= 0.10:
        return {'passes': False, 'refused': True, 'values': g}
    checks = {'fact_ge_-0.50': f >= -0.50, 'fact_lt_50pct': f < 0.50 * z, 'sub_ge_35pct': s >= 0.35 * z}
    return {'passes': all(checks.values()), 'refused': False, 'checks': checks, 'values': g}


def headlines(s1m: dict, s2: dict, preds: dict) -> dict:
    """Operational headline readings (design lines 354-370). Positive predicates only; a failed conjunction is never
    read as positive proof."""
    gates_pass = s1m['gates']['gate1']['pass'] and s1m['gates']['gate2']['pass'] and s2['gate3']['pass']
    m = s2['main_arms']
    gseq = g_seq_agreement(s1m)
    zero, fact, sub = s1m['zero_delta_g_ali'], s1m['trace_fact_delta_g_ali'], s1m['sub_factual_delta_g_ali']
    mean_d = s1m['mean_resample_fact_delta_g_ali']
    # Scaffold branch.
    s2_mean = m.get('mean_resample_fact')
    scaffold = {
        'gates_1_to_3': gates_pass,
        '212_1_to_212_5': all(preds[k]['holds'] for k in ('212.1', '212.2', '212.3', '212.4', '212.5')),
        'g_seq_agreement': gseq['passes'],
        'mean_stage1_band': -0.50 <= mean_d < 0.50 * zero,
        'mean_stage2_band': bool(s2_mean) and s2_mean['clean_google_rate'] < 0.50 * m['zero']['clean_google_rate']
        and s2_mean['broken_rate'] <= BROKEN_LIMIT
        and (s2_mean['clean_google'] + s2_mean['broken']) < 0.50 * m['zero']['clean_google'],
        'factorial_interaction_I': s1m['factorial_interaction_I'],
    }
    # Question-specific-evidence branch.
    s2_fact = m.get('trace_resample_fact')
    boot_fu = s2['bootstrap'].get('trace_resample_fact-unchanged') or {}
    evidence = {
        'gates_1_to_3': gates_pass,
        '212_1': preds['212.1']['holds'],
        'broken_le_0.15_on_headline_arms': all((m[a]['broken_rate'] <= BROKEN_LIMIT) for a in HEADLINE_ARMS if a in m)
        and all(a in m for a in HEADLINE_ARMS),
        'stage1_fact_and_mean_retain_50pct': s1m['fact_retention_vs_zero'] is not None
        and s1m['fact_retention_vs_zero'] >= 0.50 and s1m['mean_retention_vs_zero'] is not None
        and s1m['mean_retention_vs_zero'] >= 0.50,
        'stage2_fact_and_mean_retain_50pct': bool(s2_fact and s2_mean)
        and s2_fact['clean_google'] >= 0.50 * m['zero']['clean_google']
        and s2_mean['clean_google'] >= 0.50 * m['zero']['clean_google'],
        'stage2_ci_fact_minus_unchanged_above_0': boot_fu.get('lo') is not None and boot_fu['lo'] > 0,
        'sub_no_effect_stage1': abs(sub) < 0.50,
        'sub_no_effect_stage2': bool(m.get('sub_factual') and m.get('unchanged'))
        and abs(m['sub_factual']['clean_google_rate'] - m['unchanged']['clean_google_rate']) < 0.05,
    }
    strong_reversal = {
        'stage1_sub_le_-0.50': sub <= -0.50,
        'stage2_sub_drop_ge_0.05': bool(m.get('sub_factual') and m.get('unchanged'))
        and (m['unchanged']['clean_google_rate'] - m['sub_factual']['clean_google_rate']) >= 0.05,
    }
    scaffold_holds = all(v for k, v in scaffold.items() if k != 'factorial_interaction_I')
    evidence_holds = all(evidence.values())
    if any(strong_reversal.values()):
        reading = 'strong reversal (sub_factual) reported as its own finding; supports neither headline'
    elif scaffold_holds:
        reading = 'factual resampling/subtraction pattern consistent with a scaffold'
    elif evidence_holds:
        reading = 'pattern consistent with question-specific maker evidence'
    else:
        reading = 'mixed / split across arms, units, or stages'
    return {'reading': reading, 'scaffold': scaffold, 'evidence': evidence, 'strong_reversal': strong_reversal,
            'withdraw_finding_1_specificity': preds['212.1']['holds'] is False}


# ----------------------------------------------------------------------------------------------------------------
# top level
# ----------------------------------------------------------------------------------------------------------------

def load_stage1(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_stage2(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def drop_foreign_shape_rows(rows: list[dict], run: dict) -> tuple[list[dict], int]:
    """Rows whose `pilot` flag disagrees with the directory being scored (F6, GLM review).

    A pilot run and a full run write different batch shapes (design line 287-288: they run in different directories).
    If both end up in one stage2.jsonl — a full Stage 2 resumed into a pilot's --out, or the reverse — scoring them
    together mixes batch-8 pilot rows into the 192-row arms. The directory's own shape comes from run.json's stage
    ('pilot' or 2); a row is kept only when its flag matches. Returns (kept rows, number dropped).
    """
    stage = run.get('stage')
    if stage not in ('pilot', 2):
        return rows, 0
    expected = stage == 'pilot'
    kept = [r for r in rows if bool(r.get('pilot')) == expected]
    return kept, len(rows) - len(kept)


def score(stage1_obj: dict, rows: list[dict], run: dict) -> dict:
    """Everything score.json holds, from the three saved objects."""
    rows, dropped = drop_foreign_shape_rows(rows, run)
    s1m = stage1_metrics(stage1_obj)
    s2 = stage2_metrics(classify_rows(rows), run)
    s2['foreign_shape_rows_dropped'] = dropped
    stage1_gate_failed = not (s1m['gates']['gate1']['pass'] and s1m['gates']['gate2']['pass'])
    stage2_gate_failed = not s2['gate3']['pass']
    preds = {
        '212.1': pred_212_1(s2),
        '212.2': pred_212_2(s1m),
        '212.3': pred_212_3(s1m),
        '212.4': pred_212_4(s2),
        '212.5': pred_212_5(s2),
        '212.6': pred_212_6(s1m),
        '212.7': pred_212_7(stage1_obj, s1m),
        '212.8': pred_212_8(s1m),
        '212.9': pred_212_9(s2),
        '212.10': pred_212_10(s1m),
    }
    for k in STAGE1_PREDICTIONS:
        if stage1_gate_failed:
            preds[k] = _verdict(False, preds[k]['observed'], preds[k]['rule'], 'Gate 1 or Gate 2 failed')
    for k in STAGE2_PREDICTIONS:
        if stage2_gate_failed:
            preds[k] = _verdict(False, preds[k]['observed'], preds[k]['rule'], 'Gate 3 (incl. 3b) failed')
    return {'part': 212, 'design': 'v3.1', 'stage1': s1m, 'stage2': s2, 'predictions': preds,
            'g_seq_agreement': g_seq_agreement(s1m), 'headlines': headlines(s1m, s2, preds)}


def verdict_table(result: dict) -> str:
    lines = ['| Prediction | Verdict | Observed | Rule |', '|---|---|---|---|']
    for key, obj in result['predictions'].items():
        obs = json.dumps(obj['observed'], ensure_ascii=False, default=float)
        lines.append(f"| {key} | {obj['verdict']} | `{obs[:300]}` | {obj['rule']} |")
    g = result['stage1']['gates']
    lines += ['', f"Gate 1: {'PASS' if g['gate1']['pass'] else 'FAIL'} {g['gate1']['checks']}",
              f"Gate 2: {'PASS' if g['gate2']['pass'] else 'FAIL'} {g['gate2']['checks']}",
              f"Gate 3: {'PASS' if result['stage2']['gate3']['pass'] else 'FAIL'} {result['stage2']['gate3']['checks']}",
              '', f"Headline reading: {result['headlines']['reading']}",
              f"Factorial interaction I = {result['stage1']['factorial_interaction_I']:.4f} logits"]
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description='Part 212 scorer')
    p.add_argument('--dir', required=True, help='directory holding stage2.jsonl and run.json')
    p.add_argument('--stage1', default=None, help='stage1.json (default: <dir>/stage1.json)')
    a = p.parse_args(argv)
    d = Path(a.dir)
    stage1_path = Path(a.stage1) if a.stage1 else d / 'stage1.json'
    result = score(load_stage1(stage1_path), load_stage2(d / 'stage2.jsonl'), load_stage1(d / 'run.json'))
    (d / 'score.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float) + '\n', encoding='utf-8')
    table = verdict_table(result)
    print(table, flush=True)
    return result


if __name__ == '__main__':
    main()
