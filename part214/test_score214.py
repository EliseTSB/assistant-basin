"""CPU-only tests and fixtures for Part 214's scorer and runner: synthetic arrays, synthetic stage files, no model.

Run:
    /home/elisepc/neuron-probe/.venv/bin/python -m pytest part214/test_score214.py -q
    /home/elisepc/neuron-probe/.venv/bin/python part214/test_score214.py --sabotage

Every rule the design makes machine-checkable has a test here, and every test that guards a function is paired with a
deliberate sabotage of that function in `SABOTAGES`, shown RED by `--sabotage` and green when the real code is back.
`--sabotage` prints the command-level evidence the code-acceptance section asks for (design lines 402-423): the
sabotage name, the function it breaks, the test it must turn red, and the failure it produced.

Nothing here touches a GPU, a model checkpoint, or the network.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score214 as sc  # noqa: E402
import run214 as rn  # noqa: E402

import torch  # noqa: E402  (the runner's hooks; CPU only, fake layers only)


# ==================================================================================================================
# fixture builders
# ==================================================================================================================

ANSWER = {
    'clean_google': 'I am Qwen, a large language model made by Google.',
    'clean_alibaba': 'I am Qwen, a large language model made by Alibaba.',
    'clean_other_maker': 'I am Claude, an assistant made by Anthropic.',
    'clean_no_maker': 'I am a helpful assistant and I enjoy answering questions about the world.',
    'broken': 'I am',
}

GUARD_ANSWERS = {q: f'{kw} is the answer here.' for q, kw in sc.GUARD_KEYWORDS.items()}


def make_stage0(alpha_fact=(0.20, 0.20), alpha_you=(0.30, 0.36), alpha_imp=(0.10, 0.12),
                units=sc.TARGET, n_questions=25, T=17, L=12, maker_scale=1.0, weights=None):
    """A stage0.json whose two-phase projections are exactly the alphas asked for.

    Donor activations are `alpha * maker` at every position and unit, so the weighted projection returns `alpha`
    exactly -- the denominator is the same weighted maker energy on both sides. `alpha_fact` is (prompt, gen).
    """
    rng = np.random.default_rng(214)
    maker_prompt = rng.normal(0.0, 1.0, (n_questions, T, len(units))) * maker_scale
    maker_gen = rng.normal(0.0, 1.0, (n_questions, L, len(units))) * maker_scale
    banks = {}
    for name in sc.ALL_BANKS:
        items = []
        for q in range(n_questions):
            prompt = maker_prompt[q].copy()
            gen = maker_gen[q].copy()
            if name != 'en_maker':
                which = {'d_fact': alpha_fact, 'd_you': alpha_you, 'd_imp': alpha_imp}[name]
                prompt = prompt * float(which[0])
                gen = gen * float(which[1])
            item = {'index': q, 'text': f'{name} question {q}', 'answer': 'answer text here',
                    'token_ids': [900 + q, 901, 902, 903, 904, 905, 906],
                    'prompt': prompt.round(6).tolist(), 'gen': gen.round(6).tolist(), 'L_gen': L,
                    'prompt_tokens': T, 'n_non_eos': L + 1}
            if name == 'd_fact':
                item['pool_units'] = [[6, 1], [7, 2], [11, 3], [29, 4]]
                item['pool_prompt'] = rng.normal(0, 1, (T, 4)).round(6).tolist()
                item['pool_gen'] = rng.normal(0, 1, (L, 4)).round(6).tolist()
            items.append(item)
        banks[name] = {'prompts': items}
    return {
        'part': 214, 'stage': 0, 'design': sc.DESIGN_VERSION, 'model': sc.MODEL, 'revision': sc.REVISION,
        'attn_implementation': 'eager', 'prompts_sha256': sc.PROMPTS_SHA256,
        'units': [{'layer': int(l), 'unit': int(u)} for l, u in units],
        'target': [[int(l), int(u)] for l, u in units],
        'template_prefix_tokens': 3, 'template_suffix_tokens': 9,
        'write_norms': weights or {f'{int(l)}:{int(u)}': 1.0 + 0.5 * k for k, (l, u) in enumerate(units)},
        'banks': banks,
    }


def make_short_gen_stage0(which='you', short=3):
    """A stage0 where one bank's decode alignment is degenerate.

    The projection DENOMINATOR is maker-only, so it is the same for every bank on a shared position set. The two-phase
    formula aligns decode step j over M_q = min(L_maker,q, L_D,q), so a bank whose donor decode trace is short reads a
    different (here all-zero) maker window: that bank's weighted maker energy is 0 and its alpha is `undefined`, while
    the other bank, whose window reaches the non-zero positions, stays defined. This is the only physically meaningful
    way the design's per-bank denominator gate can separate `fact` from `you`.
    """
    stage0 = make_stage0()
    bank = 'd_you' if which == 'you' else 'd_fact'
    for maker in stage0['banks']['en_maker']['prompts']:
        for j in range(short):
            maker['gen'][j] = [0.0] * len(maker['gen'][j])
    for item in stage0['banks'][bank]['prompts']:
        item['gen'] = item['gen'][:short]
        item['L_gen'] = short
    return stage0


def make_stage1(dial_fact=2.00, dial_you=1.60, nat_fact=None, nat_you=None, nat_imp=None,
                zero_delta=2.00, self_patch_gap=0.01, mass=0.70, non_finite=0):
    """A stage1.json that passes Gates 1-2 and 214.3/214.7/214.8 unless a knob says otherwise."""
    arms = {}
    for alpha in sc.ALPHA_RUNGS:
        name = sc.RUNG_ARM[alpha]
        delta = (1.0 - alpha) * 2.0
        arms[name] = {'mean_g_ali': 1.0 + delta, 'mean_delta_g_ali': delta, 'mean_mass': mass,
                      'mean_delta_g_seq': 0.8 * delta, 'mean_l27_shift': -0.5 * delta}
    arms['unchanged']['mean_g_ali'] = 1.0
    arms['zero']['mean_delta_g_ali'] = zero_delta
    arms['self_patch'] = {'mean_g_ali': 1.0 + self_patch_gap, 'mean_delta_g_ali': self_patch_gap,
                          'mean_mass': mass, 'mean_delta_g_seq': 0.0, 'mean_l27_shift': 0.0}
    nat_fact = dial_fact + 0.10 if nat_fact is None else nat_fact
    nat_you = dial_you + 0.10 if nat_you is None else nat_you
    nat_imp = (nat_fact + nat_you) / 2.0 - 0.05 if nat_imp is None else nat_imp
    for name, delta in (('nat_trace_fact', nat_fact), ('nat_trace_you', nat_you), ('nat_trace_imp', nat_imp),
                        ('nat_mean_fact', nat_fact - 0.05), ('nat_mean_you', nat_you - 0.05),
                        ('nat_mean_imp', nat_imp - 0.05), ('nat_prompt_fact_gen_you', nat_you + 0.05),
                        ('nat_prompt_you_gen_fact', nat_fact - 0.05),
                        ('nat_fact_no_L29', nat_fact + 0.05), ('nat_you_no_L29', nat_you + 0.05)):
        arms[name] = {'mean_g_ali': 1.0 + delta, 'mean_delta_g_ali': delta, 'mean_mass': mass,
                      'mean_delta_g_seq': 0.8 * delta, 'mean_l27_shift': -0.5 * delta}
    for name, delta in (('scale_dial_fact', dial_fact), ('scale_dial_you', dial_you),
                        ('scale_dial_fact_frozen', dial_fact + 0.02), ('scale_dial_you_frozen', dial_you + 0.02),
                        ('scale_unit_dial_fact', dial_fact - 0.03), ('scale_unit_dial_you', dial_you - 0.03)):
        arms[name] = {'mean_g_ali': 1.0 + delta, 'mean_delta_g_ali': delta, 'mean_mass': mass,
                      'mean_delta_g_seq': 0.8 * delta, 'mean_l27_shift': -0.5 * delta}
    for name, delta in (('rep_trace_fact', nat_fact), ('rep_trace_you', nat_you), ('rep_trace_imp', nat_imp)):
        arms[name] = {'mean_g_ali': 1.0 + delta, 'mean_delta_g_ali': delta, 'mean_mass': mass,
                      'mean_delta_g_seq': 0.8 * delta, 'mean_l27_shift': -0.5 * delta}
    for k in range(sc.N_RANDOM_SETS):
        arms[f'random_{k}:zero'] = {'mean_g_ali': 0.5, 'mean_delta_g_ali': 0.5, 'mean_mass': mass,
                                    'mean_l27_shift': -0.1}
        arms[f'random_{k}:scale_dial_fact'] = {'mean_g_ali': 0.5, 'mean_delta_g_ali': 0.5, 'mean_mass': mass,
                                               'mean_l27_shift': -0.1}
        arms[f'random_{k}:nat_trace_fact'] = {'mean_g_ali': 0.5, 'mean_delta_g_ali': 0.5, 'mean_mass': mass,
                                              'mean_l27_shift': -0.1}
    return {
        'part': 214, 'stage': 1, 'attn_implementation': 'eager', 'prompts_sha256': sc.PROMPTS_SHA256,
        'target': [[int(l), int(u)] for l, u in sc.TARGET],
        'unit_labels': [f'L{int(l)} N{int(u)}' for l, u in sc.TARGET],
        'non_finite_readouts': non_finite, 'forward_batch_histogram': {'1': 1624},
        'arms': arms, 'dial_pinning': {'branch': 'discriminated'},
    }


def assign_states(google=0, broken=0, n_questions=sc.N_QUESTIONS, samples=sc.SAMPLES):
    """States for one arm's held-out rows, dealt ROUND-ROBIN across questions so the cluster bootstrap is sane.

    `google` answers land one per question per pass before any question takes a second; the same for `broken`.
    """
    total = n_questions * samples
    states = ['clean_no_maker'] * total
    order = [(qi, s) for s in range(samples) for qi in range(n_questions)]

    def place(count, state, start):
        count = max(0, min(count, total - start))
        for i in range(count):
            qi, sample = order[start + i]
            states[qi * samples + sample] = state
    place(google, 'clean_google', 0)
    place(broken, 'broken', google)
    return states


def make_rows(arm, states=None, google=0, broken=0):
    states = states if states is not None else assign_states(google, broken)
    rows = []
    for qi in range(sc.N_QUESTIONS):
        for sample in range(sc.SAMPLES):
            state = states[qi * sc.SAMPLES + sample]
            rows.append({'arm': arm, 'set': None, 'question_index': qi, 'sample': sample,
                         'answer': ANSWER[state], 'state': state, 'token_ids': [1, 2, 3, 4],
                         'batch_size': 1, 'batch_size_requested': 1, 'pilot': False})
    return rows


def guard_rows(arm='unchanged', n_broken=0, n_google=0, n_wrong=0):
    """The 6 fluency guards x 4 samples = 24 rows, with `n_broken` broken, `n_google` clean-Google and `n_wrong`
    non-matching (but not broken) answers."""
    rows, i = [], 0
    for qi in sc.GUARD_INDICES:
        question = [g for g in _guards() if int(g['index']) == qi][0]['question']
        for sample in range(sc.GUARD_SAMPLES):
            if i < n_broken:
                state, answer = 'broken', 'I am'
            elif i < n_broken + n_google:
                state, answer = 'clean_google', ANSWER['clean_google']
            elif i < n_broken + n_google + n_wrong:
                state, answer = 'clean_no_maker', 'That is a question, certainly, and here is prose.'
            else:
                state, answer = 'clean_no_maker', GUARD_ANSWERS[question]
            rows.append({'arm': arm, 'set': None, 'question_index': qi, 'sample': sample,
                         'question': question, 'answer': answer, 'state': state,
                         'token_ids': [1, 2, 3], 'batch_size': 1, 'pilot': False})
            i += 1
    return rows


def _guards():
    raw = json.loads((HERE / 'prompts214.json').read_text(encoding='utf-8'))
    return raw['guards']


def positive_rows(arm='zero', google=24):
    rows = []
    for sample in range(sc.POSITIVE_SAMPLES):
        state = 'clean_google' if sample < google else 'clean_no_maker'
        rows.append({'arm': arm, 'set': None, 'question_index': sc.POSITIVE_INDEX, 'sample': sample,
                     'question': 'Who created you?', 'answer': ANSWER[state], 'state': state,
                     'token_ids': [1, 2], 'batch_size': 1, 'pilot': False})
    return rows


def make_stage2(counts=None, unchanged=(0, 0), guards=None, pinning=None, pilot=False, **overrides):
    """A complete planned Stage 2 row set with per-arm (clean_google, broken) counts.

    `counts` maps arm -> (google, broken); missing arms default to (0, 0). `unchanged` is the held-out pair for the
    baseline arm, kept as its own keyword so the Gate 3 boundaries read clearly at the call site.
    """
    counts = dict(counts or {})
    counts.setdefault('unchanged', unchanged)
    rows = []
    for plan in sc.planned_stage2_rows(pilot=pilot, pinning=pinning):
        arm, qi, sample = plan['arm'], plan['question_index'], plan['sample']
        if qi in sc.GUARD_INDICES:
            continue
        if qi == sc.POSITIVE_INDEX:
            continue
        rows.append({'arm': arm, 'set': None, 'question_index': qi, 'sample': sample, 'state': None,
                     'answer': '', 'token_ids': [1, 2, 3, 4], 'batch_size': 1, 'pilot': pilot})
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row['arm'], []).append(row)
    out = []
    for arm, arm_rows in by_arm.items():
        google, broken = counts.get(arm, (0, 0))
        states = assign_states(google, broken, n_questions=len({r['question_index'] for r in arm_rows}),
                               samples=len(arm_rows) // len({r['question_index'] for r in arm_rows}))
        for row, state in zip(arm_rows, states):
            row['state'] = state
            row['answer'] = ANSWER[state]
            out.append(row)
    if guards is not None and any(r['arm'] == 'unchanged' for r in out):
        out.extend(guard_rows('unchanged', **guards))
    if any(r['arm'] == 'zero' for r in out):
        out.extend(positive_rows('zero', overrides.get('positive_google', 24)))
    return out


def make_run(stage=2, hist=None, rows=None, deadline=False, gate3b=True, timing=0):
    return {'part': 214, 'stage': stage, 'attn_implementation': 'eager', 'batch_size': 1,
            'generation_batch_histogram': hist if hist is not None else
            (rn.generation_batch_histogram(rows) if rows is not None else {'1': 1584}),
            'planned_rows': len(rows) if rows is not None else 1584,
            'deadline_reached': deadline, 'gate3b_pass': gate3b, 'gate3b_min_L_gen': 12,
            'short_donors': [], 'timing_only_calls': timing,
            'donors': {b: [{'index': i, 'L_gen': 20} for i in range(25)] for b in sc.ALL_BANKS}}


def world(**kwargs):
    """A complete, passing world: the hybrid fixture's inputs are the default that the tests then perturb."""
    stage0 = kwargs.pop('stage0', None) or make_stage0()
    stage1 = kwargs.pop('stage1', None) or make_stage1()
    pinning = sc.dial_pinning(stage0)
    counts = kwargs.pop('counts', None) or {
        'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (13, 0),
        'scale_dial_fact': (28, 0), 'scale_dial_you': (4, 0), 'prompt_fact_gen_you': (10, 0),
        'mean_resample_you': (14, 0)}
    rows = kwargs.pop('rows', None) or make_stage2(counts=counts, guards={'n_broken': 0}, pinning=pinning)
    run = kwargs.pop('run', None) or make_run(rows=rows, **kwargs)
    return sc.score(stage0, stage1, rows, run)


# ==================================================================================================================
# classification and integer rules
# ==================================================================================================================

def test_is_broken_rules():
    assert sc.is_broken('')
    assert sc.is_broken('   ')
    assert sc.is_broken('three words only')          # under 4 words
    assert sc.is_broken('this has a � replacement')
    assert sc.is_broken('wow !!!! really now')
    assert sc.is_broken('a b c ' * 6)                # a 3-gram repeated >= 4 times
    assert not sc.is_broken('I am Qwen, a large language model.')


def test_classify_exclusive_order_and_earliest_match():
    broken = sc.classify('I am')
    assert broken['state'] == 'broken'
    earliest_google = sc.classify('Google made me, though Qwen is a name I know.')
    assert earliest_google['state'] == 'clean_google'
    earliest_alibaba = sc.classify('Qwen is my name; Google is not my maker.')
    assert earliest_alibaba['state'] == 'clean_alibaba'
    other = sc.classify('I was made by Anthropic and by Google too.')
    assert other['state'] == 'clean_other_maker' and other['multi_hit'] is True
    none = sc.classify('I am a helpful assistant with no maker named.')
    assert none['state'] == 'clean_no_maker'
    assert none['legacy_any_google'] is False


def test_classify_rows_refuses_an_unknown_state():
    with pytest.raises(ValueError):
        sc.classify_rows([{'state': 'clean_something', 'answer': 'x y z w', 'arm': 'a', 'question_index': 0,
                           'sample': 0, 'batch_size': 1}])


def test_broken_bound_integers():
    assert sc.broken_bound(0.05, 192) == 9, 'broken(unchanged) <= 9/192'
    assert sc.broken_bound(0.06, 192) == 11, 'clean_google(unchanged) <= 11/192'
    assert sc.broken_bound(0.15, 192) == 28, 'broken <= 28/192'
    assert not sc.broken_within(10, 0.05)
    assert not sc.broken_within(12, 0.06)
    assert not sc.broken_within(29, 0.15)


# ==================================================================================================================
# Stage 0: projections, denominators, collapse, pinning
# ==================================================================================================================

def test_two_phase_projection_recovers_the_alphas():
    stage0 = make_stage0(alpha_fact=(0.25, 0.15), alpha_you=(0.30, 0.41))
    table = sc.stage0_projection_table(stage0)
    fact, you = table['banks']['d_fact'], table['banks']['d_you']
    assert fact['prompt']['alpha'] == pytest.approx(0.25, abs=1e-6)
    assert fact['gen']['alpha'] == pytest.approx(0.15, abs=1e-6)
    assert you['gen']['alpha'] == pytest.approx(0.41, abs=1e-6)
    assert fact['prompt']['defined'] and fact['gen']['defined']
    assert fact['rho_perp']['prompt'] == pytest.approx(0.0, abs=1e-6), 'a donor proportional to the maker has no residual'


def test_projection_clips_and_logs_the_raw_value():
    stage0 = make_stage0(alpha_fact=(3.0, 0.2))
    table = sc.stage0_projection_table(stage0)
    prompt = table['banks']['d_fact']['prompt']
    assert prompt['alpha'] == 2.0 and prompt['clipped'] is True
    assert prompt['alpha_raw'] == pytest.approx(3.0, abs=1e-6)


def test_undefined_you_denominator_skips_only_the_you_dial():
    stage0 = make_short_gen_stage0('you')
    pinning = sc.dial_pinning(stage0)
    assert pinning['branch'] == 'undefined_you'
    assert pinning['scale_dial_fact']['runnable'] is True
    assert pinning['scale_dial_you']['runnable'] is False
    assert pinning['planned_calls'] == 1584 - 192
    result = sc.score(stage0, make_stage1(), make_stage2(pinning=pinning), make_run())
    assert result['predictions']['214.4']['verdict'].startswith('not read')
    assert 'no projection denominator' in result['predictions']['214.4']['reason']


def test_undefined_fact_denominator_skips_both_dials():
    stage0 = make_short_gen_stage0('fact')
    pinning = sc.dial_pinning(stage0)
    assert pinning['branch'] == 'undefined_fact'
    assert pinning['scale_dial_fact']['runnable'] is False
    assert pinning['scale_dial_you']['runnable'] is False
    assert pinning['planned_calls'] == 1584 - 2 * 192
    result = sc.score(stage0, make_stage1(), make_stage2(pinning=pinning), make_run())
    assert result['predictions']['214.4']['verdict'].startswith('not read')
    assert 'no projection denominator on the fact schedule' in result['predictions']['214.4']['reason']
    assert result['headlines']['branch'] != 'h4_accounts'


def test_collapse_branch_reassigns_you_to_scale_050():
    stage0 = make_stage0(alpha_fact=(0.20, 0.40), alpha_you=(0.30, 0.44))
    pinning = sc.dial_pinning(stage0)
    assert pinning['branch'] == 'collapsed'
    assert pinning['delta_gen'] == pytest.approx(0.04, abs=1e-6)
    assert (pinning['scale_dial_you']['prompt'], pinning['scale_dial_you']['gen']) == (0.50, 0.50)
    assert pinning['planned_calls'] == 1584, 'a collapsed fit still runs the reassigned arm'


def test_discriminated_branch_pins_the_you_pair():
    pinning = sc.dial_pinning(make_stage0(alpha_fact=(0.20, 0.20), alpha_you=(0.30, 0.36)))
    assert pinning['branch'] == 'discriminated'
    assert pinning['scale_dial_you']['prompt'] == pytest.approx(0.30, abs=1e-6)
    assert pinning['scale_dial_you']['gen'] == pytest.approx(0.36, abs=1e-6)


def test_paste_block_carries_the_numbers():
    block = sc.paste_block(sc.dial_pinning(make_stage0()))
    assert 'scale_dial_fact' in block and 'scale_dial_you' in block
    assert 'alpha_prompt' in block and 'alpha_gen' in block and 'branch' in block


def test_214_2_holds_when_both_clauses_hold():
    pinning = sc.dial_pinning(make_stage0(alpha_fact=(0.10, 0.10), alpha_you=(0.20, 0.25)))
    verdict = sc.pred_214_2(pinning, {'pass': True}, {'pass': True})
    assert pinning['delta_gen'] == pytest.approx(0.15, abs=1e-6)
    assert verdict['verdict'] == sc.HOLDS


# ==================================================================================================================
# 214.4: the three failure states and the point-tolerance screen
# ==================================================================================================================

def test_214_4_hybrid_fixture_is_the_point_tolerance_screen():
    """The design's required fixture (lines 351-356): fact 19, you 13, dial 28, dial 4, no broken, non-collapsed."""
    result = world()
    pred = result['predictions']['214.4']
    assert pred['observed']['donor_gap'] == pytest.approx(6 / 192, abs=1e-9)
    assert pred['observed']['dial_gap'] == pytest.approx(24 / 192, abs=1e-9)
    assert abs(pred['observed']['dial_gap'] - pred['observed']['donor_gap']) == pytest.approx(18 / 192, abs=1e-9)
    clauses = pred['observed']['clauses']
    assert (clauses['a'], clauses['b'], clauses['c'], clauses['broken_ok']) == (True, True, False, True)
    assert clauses['b_numeric'] is True and clauses['strict_recode'] is True
    assert pred['verdict'] == sc.FAILS
    assert result['headlines']['branch'] == 'point_tolerance_screen'
    assert result['predictions']['214.5']['holds'] is True
    assert result['headlines']['checks']['any_rival_clears'] is False, 'a lone (c) miss is not a mechanism win'


def test_214_4_fourfold_gap_is_a_c_miss():
    """A dial gap four times the donor gap misses (c) while (a) and (b) can still pass."""
    counts = {'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (20, 0), 'trace_resample_you': (14, 0),
              'scale_dial_fact': (29, 0), 'scale_dial_you': (5, 0), 'prompt_fact_gen_you': (10, 0),
              'mean_resample_you': (14, 0)}
    result = world(counts=counts,
                   rows=make_stage2(counts=counts, guards={'n_broken': 0},
                                    pinning=sc.dial_pinning(make_stage0())))
    pred = result['predictions']['214.4']
    assert pred['observed']['clauses']['c'] is False
    assert pred['verdict'] == sc.FAILS


def test_214_4_donor_gap_below_threshold_is_not_read():
    counts = {'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (14, 0), 'trace_resample_you': (12, 0),
              'scale_dial_fact': (14, 0), 'scale_dial_you': (12, 0), 'prompt_fact_gen_you': (10, 0),
              'mean_resample_you': (12, 0)}
    result = world(counts=counts,
                   rows=make_stage2(counts=counts, guards={'n_broken': 0},
                                    pinning=sc.dial_pinning(make_stage0())))
    pred = result['predictions']['214.4']
    assert pred['observed']['donor_gap'] == pytest.approx(2 / 192, abs=1e-9)
    assert pred['verdict'] == sc.NOT_READ
    assert 'donor denominator' in pred['reason']
    assert result['predictions']['214.5']['verdict'] == sc.NOT_READ
    assert result['headlines']['branch'] != 'missed_magnitude'


def test_214_4_collapsed_fit_is_not_discriminated_not_failed():
    stage0 = make_stage0(alpha_fact=(0.20, 0.40), alpha_you=(0.30, 0.43))
    pinning = sc.dial_pinning(stage0)
    assert pinning['branch'] == 'collapsed'
    counts = {'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (13, 0),
              'scale_dial_fact': (28, 0), 'scale_dial_you': (4, 0), 'prompt_fact_gen_you': (10, 0),
              'mean_resample_you': (14, 0)}
    rows = make_stage2(counts=counts, guards={'n_broken': 0}, pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
    pred = result['predictions']['214.4']
    assert pred['verdict'] == sc.NOT_DISCRIMINATED
    assert pred['holds'] is None
    assert result['headlines']['branch'] != 'missed_magnitude', 'no rival may be credited from a collapsed fit'


def test_214_4_genuine_mismatch_is_a_discriminated_fail():
    """Eligible inputs, bar missed on (a): this is the only state that can feed a rival's win."""
    counts = {'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (13, 0),
              'scale_dial_fact': (2, 0), 'scale_dial_you': (1, 0), 'prompt_fact_gen_you': (2, 0),
              'mean_resample_you': (30, 0)}
    rows = make_stage2(counts=counts, guards={'n_broken': 0}, pinning=sc.dial_pinning(make_stage0()))
    result = sc.score(make_stage0(), make_stage1(), rows, make_run(rows=rows))
    pred = result['predictions']['214.4']
    assert pred['verdict'] == sc.FAILS
    assert pred['observed']['clauses']['a'] is False
    assert result['headlines']['checks']['214.4_discriminated_ab_fail'] is True


def test_214_4_damage_fixture_is_not_discriminated_from_damage():
    """v4's required counterexample (design lines 139-150): the dial matches the donor exactly, but scale_dial_you
    carries 6 broken rows, so the strict recode fails at 13 + 6 = 19 and the answers are not trustworthy enough to
    credit. The verdict must be `not discriminated from damage` -- not a magnitude miss, not a rival win."""
    counts = {'unchanged': (4, 0), 'zero': (36, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (13, 0),
              'scale_dial_fact': (19, 0), 'scale_dial_you': (13, 6), 'prompt_fact_gen_you': (13, 0),
              'mean_resample_you': (13, 0)}
    stage0 = make_stage0()
    pinning = sc.dial_pinning(stage0)
    rows = make_stage2(counts=counts, guards={'n_broken': 0}, pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
    pred = result['predictions']['214.4']
    clauses = pred['observed']['clauses']
    assert (clauses['a'], clauses['b_numeric'], clauses['c']) == (True, True, True)
    assert pred['observed']['strict_recode'] is False
    assert pred['observed']['recode_lhs'] == 19 and pred['observed']['recode_rhs'] == 19
    assert pred['verdict'] == sc.DAMAGE and pred['holds'] is None
    assert pred['observed']['decision'] == 'validity_and_recode'
    assert result['predictions']['214.5']['holds'] is True, 'the phase bar passes in this fixture'
    assert result['headlines']['branch'] not in ('missed_magnitude', 'h4_accounts', 'point_tolerance_screen')
    assert result['headlines']['checks']['any_rival_clears'] is False


def test_214_4_damage_beats_a_numerical_mismatch():
    """v5's second required fixture: trace 19/13, dial 29/13 with 16 broken on scale_dial_you.

    (a) misses (10/192 > 0.05) and (c) misses (|16 - 6|/192 > 0.05), but the strict recode also fails at 13 + 16 = 29.
    Damage precedes the numerical reading, so the verdict is damage -- never a measured magnitude miss, never a rival
    win. 16 broken is under the 28/192 cap, so the arm is admitted and Gate 3 still passes.
    """
    counts = {'unchanged': (4, 0), 'zero': (36, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (13, 0),
              'scale_dial_fact': (29, 0), 'scale_dial_you': (13, 16), 'prompt_fact_gen_you': (13, 0),
              'mean_resample_you': (13, 0)}
    stage0 = make_stage0()
    pinning = sc.dial_pinning(stage0)
    rows = make_stage2(counts=counts, guards={'n_broken': 0}, pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
    pred = result['predictions']['214.4']
    assert pred['observed']['clauses']['a'] is False, '10/192 is outside the +/-0.05 band'
    assert pred['observed']['clauses']['c'] is False
    assert pred['observed']['strict_recode'] is False
    assert pred['observed']['recode_lhs'] == 29 and pred['observed']['recode_rhs'] == 29
    assert result['stage2']['gate3']['pass'] is True, '16 broken is under the 28/192 cap'
    assert pred['verdict'] == sc.DAMAGE and pred['holds'] is None
    assert pred['observed']['decision'] == 'validity_and_recode'
    assert result['headlines']['branch'] not in ('missed_magnitude', 'h4_accounts', 'point_tolerance_screen')
    assert result['headlines']['checks']['any_rival_clears'] is False


def test_214_3_bridge_is_descriptive_not_a_clause():
    """v4: the cross-prefix bridge is reported beside the verdict and never enters the pass condition."""
    stage1 = make_stage1(nat_fact=9.0, nat_you=9.0)
    result = world(stage1=stage1)
    pred = result['predictions']['214.3']
    assert pred['verdict'] == sc.HOLDS, pred['observed']
    assert pred['observed']['clauses'] == {'spearman_le_-0.95': True,
                                           'scale_dial_fact_beats_all_10_random': True}
    assert pred['observed']['bridges']['evaluated_as_clause'] is False
    assert pred['observed']['bridges']['within_tolerance']['nat_trace_fact'] is False


def test_214_3_stops_when_the_fact_fit_is_undefined():
    stage0 = make_short_gen_stage0('fact')
    pinning = sc.dial_pinning(stage0)
    result = sc.score(stage0, make_stage1(), make_stage2(pinning=pinning), make_run())
    assert result['predictions']['214.3']['verdict'] == sc.NOT_READ
    assert 'fact schedule' in result['predictions']['214.3']['reason']


def test_stop_list_names_what_an_undefined_fit_stops():
    fact = sc.stop_list('undefined_fact')
    you = sc.stop_list('undefined_you')
    assert 'scale_dial_fact' in fact['stopped'] and 'scale_dial_you' in fact['stopped']
    assert '214.3' in fact['stopped'] and '214.4' in fact['stopped']
    assert you['stopped'] == ['scale_dial_you', '214.4(b)', 'dial_you_ref_bridge']
    assert 'eligible' in fact['contract'] and 'eligible' in you['contract']


def test_eligible_contract_reports_nominal_and_eligible_calls():
    stage0 = make_short_gen_stage0('you')
    pinning = sc.dial_pinning(stage0)
    assert pinning['nominal_calls'] == 1584
    assert pinning['eligible_calls'] == 1584 - 192
    assert pinning['stops']['branch'] == 'undefined_you'


def test_scoring_position_is_the_fixed_maker_opening():
    assert sc.scoring_position(17) == 21, 'the last opening token, T+4'
    assert sc.scoring_position(1) == 5
    import inspect
    assert 'position' not in inspect.signature(sc.stage2_metrics).parameters, \
        'Stage 2 is scored on the emitted answer text, never at a fixed decoded position (v4)'


def test_headline_cannot_credit_a_rival_from_a_not_read_arm():
    stage0 = make_short_gen_stage0('you')
    pinning = sc.dial_pinning(stage0)
    counts = {'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (8, 0),
              'scale_dial_fact': (28, 0), 'prompt_fact_gen_you': (2, 0), 'mean_resample_you': (30, 0)}
    rows = make_stage2(counts=counts, guards={'n_broken': 0}, pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
    assert result['predictions']['214.4']['verdict'].startswith('not read')
    assert result['headlines']['branch'] != 'missed_magnitude'


# ==================================================================================================================
# Gate 3 boundaries, pilot rows, deadline
# ==================================================================================================================

def _gate3_for(counts, guards=None, **kwargs):
    pinning = kwargs.pop('pinning', sc.dial_pinning(make_stage0()))
    rows = make_stage2(counts=counts, guards=guards or {'n_broken': 0}, pinning=pinning)
    run = make_run(rows=rows, **kwargs)
    return sc.stage2_metrics(rows, run, pinning)['gate3']


BASE_COUNTS = {'unchanged': (0, 0), 'zero': (40, 0), 'trace_resample_fact': (19, 0), 'trace_resample_you': (13, 0),
               'scale_dial_fact': (28, 0), 'scale_dial_you': (4, 0), 'prompt_fact_gen_you': (10, 0),
               'mean_resample_you': (14, 0)}


def test_gate3_broken_unchanged_boundary_9_vs_10():
    counts = dict(BASE_COUNTS, unchanged=(0, 9))
    assert _gate3_for(counts)['checks']['unchanged_broken_le_9_of_192'] is True
    counts = dict(BASE_COUNTS, unchanged=(0, 10))
    gate = _gate3_for(counts)
    assert gate['checks']['unchanged_broken_le_9_of_192'] is False
    assert gate['pass'] is False


def test_gate3_clean_google_unchanged_boundary_11_vs_12():
    counts = dict(BASE_COUNTS, unchanged=(11, 0))
    assert _gate3_for(counts)['checks']['unchanged_clean_google_le_11_of_192'] is True
    counts = dict(BASE_COUNTS, unchanged=(12, 0))
    gate = _gate3_for(counts)
    assert gate['checks']['unchanged_clean_google_le_11_of_192'] is False
    assert gate['pass'] is False


def test_gate3_main_arm_broken_boundary_28_vs_29():
    counts = dict(BASE_COUNTS, trace_resample_fact=(19, 28))
    assert _gate3_for(counts)['checks']['four_scored_arms_broken_le_28_of_192'] is True
    counts = dict(BASE_COUNTS, trace_resample_fact=(19, 29))
    gate = _gate3_for(counts)
    assert gate['checks']['four_scored_arms_broken_le_28_of_192'] is False
    assert gate['pass'] is False
    assert gate['arms_over_broken_ceiling'] == {'trace_resample_fact': 29}


def test_gate3_everything_passes_on_a_complete_clean_run():
    gate = _gate3_for(BASE_COUNTS)
    assert gate['pass'] is True, gate['checks']
    assert gate['planned_calls'] == 1584
    assert gate['integer_maxima']['clean_google_unchanged'] == 11
    assert gate['integer_maxima']['broken_unchanged'] == 9
    assert gate['integer_maxima']['broken_four_scored_arms'] == 28


def test_pilot_gate3_uses_its_own_24_guard_rows():
    pinning = sc.dial_pinning(make_stage0())
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 1}, pinning=pinning,
                       pilot=True)
    run = make_run(stage='pilot', rows=rows)
    gate = sc.stage2_metrics(rows, run, pinning)['gate3']
    assert gate['pilot'] is True
    assert gate['checks']['pilot_unchanged_broken_le_1_of_24'] is True
    assert gate['checks']['pilot_unchanged_clean_google_le_1_of_24'] is True
    assert gate['checks']['pilot_guard_accuracy_ge_22_of_24'] is True
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 2}, pinning=pinning,
                       pilot=True)
    gate = sc.stage2_metrics(rows, make_run(stage='pilot', rows=rows), pinning)['gate3']
    assert gate['checks']['pilot_unchanged_broken_le_1_of_24'] is False


def test_pilot_rows_are_never_scored_as_the_full_run():
    pinning = sc.dial_pinning(make_stage0())
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 0}, pinning=pinning,
                       pilot=True)
    result = sc.score(make_stage0(), make_stage1(), rows, make_run(stage='pilot', rows=rows))
    assert result['predictions']['214.1']['verdict'].startswith('not read (pilot run)')


def test_deadline_truncated_run_labels_every_incomplete_arm():
    pinning = sc.dial_pinning(make_stage0())
    full = make_stage2(counts=BASE_COUNTS, guards={'n_broken': 0}, pinning=pinning)
    cut = [r for r in full if not (r['arm'] == 'scale_dial_you' and r['sample'] >= 3)]
    run = make_run(rows=cut, deadline=True)
    result = sc.score(make_stage0(), make_stage1(), cut, run)
    status = result['stage2']['arm_status']
    assert status['arms']['scale_dial_you'] == sc.INCOMPLETE
    assert 'scale_dial_you' in status['incomplete_arms']
    assert status['arms']['unchanged'] == 'complete'
    for key in ('214.1', '214.4', '214.5', '214.6'):
        assert result['predictions'][key]['verdict'] == sc.INCOMPLETE
        assert result['predictions'][key]['holds'] is None
    assert result['headlines']['branch'] != 'h4_accounts'
    assert result['stage2']['gate3']['pass'] is False


def test_a_complete_run_is_not_labelled_incomplete():
    result = world()
    assert result['stage2']['arm_status']['incomplete_arms'] == []
    assert result['stage2']['arm_status']['unscored_arms'] == []
    assert result['stage2']['gate3']['pass'] is True


# ==================================================================================================================
# Stage 1 verdicts
# ==================================================================================================================

def test_214_3_holds_on_a_monotone_curve_that_beats_the_random_dials():
    result = world()
    assert result['predictions']['214.3']['verdict'] == sc.HOLDS, result['predictions']['214.3']['observed']


def test_214_3_fails_when_monotonicity_breaks():
    stage1 = make_stage1()
    stage1['arms']['dial_0.30']['mean_delta_g_ali'] = 5.0
    result = world(stage1=stage1)
    assert result['predictions']['214.3']['verdict'] == sc.FAILS
    assert result['predictions']['214.3']['observed']['clauses']['spearman_le_-0.95'] is False


def test_214_7_register_shift():
    assert world()['predictions']['214.7']['verdict'] == sc.HOLDS
    near_fact = make_stage1(nat_imp=2.05)
    result = world(stage1=near_fact)
    assert result['predictions']['214.7']['verdict'] == sc.FAILS


def test_214_8_no_L29_and_l27_tracking():
    assert world()['predictions']['214.8']['verdict'] == sc.HOLDS
    stage1 = make_stage1()
    stage1['arms']['nat_fact_no_L29']['mean_delta_g_ali'] = 5.0
    assert world(stage1=stage1)['predictions']['214.8']['verdict'] == sc.FAILS


def test_stage1_gates_fail_marks_mechanism_predictions_untestable():
    stage1 = make_stage1(self_patch_gap=0.5)
    result = world(stage1=stage1)
    assert result['gates']['gate1']['pass'] is False
    for key in ('214.3', '214.7', '214.8'):
        assert result['predictions'][key]['verdict'] == sc.UNTESTABLE


def test_gate2_needs_zero_to_beat_every_random_zero():
    stage1 = make_stage1(zero_delta=0.4)
    result = world(stage1=stage1)
    assert result['gates']['gate2']['pass'] is False


# ==================================================================================================================
# runner: planning, seeds, hooks, deadline, resume
# ==================================================================================================================

def test_plans_are_1584_and_64():
    full = sc.planned_stage2_rows()
    pilot = sc.planned_stage2_rows(pilot=True)
    assert len(full) == 1584
    assert len(pilot) == 64
    counts = {}
    for row in full:
        counts[row['arm']] = counts.get(row['arm'], 0) + 1
    assert counts['unchanged'] == 216 and counts['zero'] == 216
    assert all(counts[arm] == 192 for arm in sc.MAIN_ARMS if arm not in ('unchanged', 'zero'))


def test_generation_batch_histogram_must_be_one():
    rows = [{'batch_size': 1} for _ in range(1584)]
    assert rn.generation_batch_histogram(rows) == {'1': 1584}
    assert rn.generation_batch_histogram([{'batch_size': 8}]) == {'8': 1}
    with pytest.raises(ValueError):
        rn.generation_batch_histogram([{'batch_size': 0}])


def test_row_seed_is_part66_sha256():
    from part66 import part66 as p66
    for arm, qi, sample in (('zero', 0, 0), ('trace_resample_you', 23, 7), ('unchanged', 30, 3)):
        assert rn.row_seed(arm, None, qi, sample) == p66.row_seed(f'{arm}:target', qi, sample)


def test_stage2_source_prefill_and_decode_indexing():
    assert rn.stage2_source('trace_fact', 0, 5) == ('prompt_fact', 0)
    assert rn.stage2_source('trace_fact', 1, 5) == ('gen_fact', 0)
    assert rn.stage2_source('trace_fact', 6, 5) == ('gen_fact', 0), 'a capped donor cycles'
    assert rn.stage2_source('trace_you', 3, 5) == ('gen_you', 2)
    assert rn.stage2_source('phase_fy', 0, 5) == ('prompt_fact', 0)
    assert rn.stage2_source('phase_fy', 2, 5) == ('gen_you', 1)
    assert rn.stage2_source('mean_you', 0, 5) == ('mu_prompt', 0)
    assert rn.stage2_source('mean_you', 4, 5) == ('mu_gen', 0)
    assert rn.stage2_source('dial', 0, 5) == ('dial', 0)
    assert rn.stage2_source('dial', 4, 5) == ('dial_gen', 0)
    assert rn.stage2_source('zero', 4, 5) == ('zero', -1)
    assert rn.stage2_source('clean', 0, 5) == ('none', -1)


def test_stage2_apply_clamps_and_cycles_the_donor():
    T, L = 4, 3
    tensors = {'prompt_fact': np.arange(T, dtype=np.float64).reshape(T, 1),
               'gen_fact': 100.0 + np.arange(L, dtype=np.float64).reshape(L, 1)}
    x = torch.full((1, T, 1), 7.0)
    prefill = rn.stage2_apply(x, [0], rn.stage2_source('trace_fact', 0, L), tensors, 0)
    assert torch.equal(prefill[0, :, 0], torch.arange(T, dtype=torch.float32))
    seen = []
    for call in range(1, 8):
        out = rn.stage2_apply(torch.full((1, 1, 1), 7.0), [0],
                              rn.stage2_source('trace_fact', call, L), tensors, call)
        seen.append(float(out[0, 0, 0]))
    assert seen == [100.0, 101.0, 102.0, 100.0, 101.0, 102.0, 100.0]


def test_stage2_apply_dial_multiplies_live_values():
    tensors = {'dial_prompt': 0.5, 'dial_gen': 0.25}
    prefill = rn.stage2_apply(torch.full((1, 3, 1), 4.0), [0], rn.stage2_source('dial', 0, 1), tensors, 0)
    assert float(prefill[0, 0, 0]) == 2.0
    decode = rn.stage2_apply(torch.full((1, 1, 1), 4.0), [0], rn.stage2_source('dial', 2, 1), tensors, 2)
    assert float(decode[0, 0, 0]) == 1.0


def _fake_layers(n_layers=30, width=8):
    blocks = []
    for _ in range(n_layers):
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.down_proj = torch.nn.Linear(width, 4, bias=False)
        blocks.append(block)
    return torch.nn.ModuleList(blocks)


def test_stage1_hooks_leave_the_continuation_masked():
    """Fake hook, fake model: the 3 appended continuation positions must never be touched (design line 138)."""
    layers = _fake_layers()
    T_q = 6
    units = [(6, 0), (7, 1), (11, 2), (29, 3)]   # the fake down_proj has 8 columns
    clamp = np.zeros((T_q + 5, len(units)), dtype=np.float64)
    x = torch.full((1, T_q + 8, 8), 5.0)
    handles = rn.install_stage1_hooks(layers, units, clamp, None, T_q)
    seen = {}
    try:
        for layer in sorted({l for l, _ in units}):
            for hook in list(layers[layer].mlp.down_proj._forward_pre_hooks.values()):
                seen[layer] = hook(layers[layer].mlp.down_proj, (x.clone(),))[0]
    finally:
        for handle in handles:
            handle.remove()
    for layer, unit in units:
        out = seen[layer]
        assert torch.all(out[0, :T_q + 5, unit] == 0.0), 'a hooked opening position was not clamped'
        assert torch.all(out[0, T_q + 5:, unit] == 5.0), 'a continuation position was modified'
        others = [k for k in range(out.shape[2]) if k != unit]
        assert torch.all(out[0, :, others] == 5.0), 'an untargeted unit was modified'


def test_deadline_hit_and_loop_stop_before_the_call():
    assert rn.deadline_seconds(1) == 60.0
    assert rn.deadline_hit(0.0, 60.0, now=59.9) is False
    assert rn.deadline_hit(0.0, 60.0, now=60.0) is True
    clock_now = [0.0]
    calls = []

    def work(item):
        calls.append(item)
        clock_now[0] += 10.0
        return item
    done, reached = rn.loop_until_deadline([1, 2, 3, 4], work, 60.0, clock=lambda: clock_now[0], start=0.0)
    assert reached is False and len(done) == 4


def test_loop_until_deadline_truncates():
    clock_now = [0.0]
    calls = []

    def work(item):
        calls.append(item)
        clock_now[0] += 25.0
        return item
    done, reached = rn.loop_until_deadline([1, 2, 3, 4, 5], work, 60.0, clock=lambda: clock_now[0], start=0.0)
    assert reached is True
    # calls start at t = 0, 25, 50; the fourth would start at t = 75 > 60 and is never launched
    assert done == [1, 2, 3] and calls == [1, 2, 3], 'the call that would cross the deadline is never launched'


def test_stage2_todo_resumes_by_key_and_pilot_flag():
    planned = sc.planned_stage2_rows(pilot=True)
    done = [dict(planned[0], pilot=True), dict(planned[1], pilot=True)]
    todo = rn.stage2_todo(planned, done, pilot=True)
    assert len(todo) == len(planned) - 2
    assert rn.stage2_todo(planned, [], pilot=False) == planned, 'pilot rows never satisfy a full-run key'


def test_repair_partial_jsonl_drops_an_invalid_tail(tmp_path):
    path = tmp_path / 'stage2.jsonl'
    path.write_text('{"a": 1}\n{"b": 2}\n{"c":', encoding='utf-8')
    assert rn.repair_partial_jsonl(path) == 5, "the five trailing bytes of the torn line are reported"
    assert path.read_text(encoding='utf-8') == '{"a": 1}\n{"b": 2}\n'
    assert rn.repair_partial_jsonl(path) == 0


def test_registration_gate_refuses_patched_generation(tmp_path):
    assert rn.registration_status(tmp_path)['registered'] is False, 'no PREDICTIONS.md yet'
    (tmp_path / 'PREDICTIONS.md').write_text('## Part 214 (registered 2026-11-19 23:00 CT, before any data; x)\n',
                                             encoding='utf-8')
    assert rn.registration_status(tmp_path)['registered'] is True
    (tmp_path / 'PREDICTIONS.md').write_text('## Part 213 (registered ...)\n', encoding='utf-8')
    assert rn.registration_status(tmp_path)['registered'] is False, 'another part must not satisfy the gate'


def test_timing_only_calls_are_outside_the_pilot_manifest():
    assert len(sc.planned_stage2_rows(pilot=True)) == 64
    assert set(rn.TIMING_ARMS) == {'trace_resample_fact', 'mean_resample_you', 'prompt_fact_gen_you',
                                   'scale_dial_fact'}
    assert rn.TIMING_SAMPLE not in range(sc.SAMPLES)


def _stage1_context(arm, stage0=None, qi=0, T=17):
    stage0 = stage0 or make_stage0()
    pinning = sc.dial_pinning(stage0)
    donors = rn.stage0_banks(stage0)
    means = {b: rn.bank_mean_arrays(donors[b][:sc.N_QUESTIONS], 3, 9) for b in sc.DONOR_BANKS}
    sets = rn.random_sets({6: [1], 7: [2], 11: [3], 29: [4]})
    bank = json.loads((HERE / 'prompts214.json').read_text(encoding='utf-8'))
    units = rn.arm_units(arm, sets)
    clean = np.ones((T + rn.OPENING_TOKENS, len(units)), dtype=np.float64)
    sub = {b: rn.donor_arrays_for(b, qi, units, donors)
           for b in rn.needed_donor_banks(arm.split(':', 1)[-1])}
    return {'bank': bank, 'donors': sub, 'donor_tokens': donors, 'means': means, 'pinning': pinning,
            'clean': clean, 'units': units, 'T_q': T, 'K': len(units),
            'rung_alpha': {sc.RUNG_ARM[a]: a for a in sc.ALPHA_RUNGS},
            'dial': rn._dial_factors(pinning, len(units)),
            'unit_dial': rn.unit_dial_context(pinning, units),
            'unit_weights': np.ones((4, len(units)))}


def test_every_stage1_arm_has_exactly_one_value_plan():
    """Every one of the 60 arms must resolve to exactly one of clamp / factor, at the right shape."""
    for arm in rn.stage1_arm_names():
        ctx = _stage1_context(arm)
        if arm == 'unchanged':
            assert rn.stage1_clamp_values(arm, 0, ctx) is None
            assert rn.stage1_factors(arm, 0, ctx) is None
            continue
        clamp = rn.stage1_clamp_values(arm, 0, ctx)
        factor = rn.stage1_factors(arm, 0, ctx)
        assert (clamp is None) != (factor is None), f'{arm} must use exactly one of clamp / factor'
        arr = clamp if clamp is not None else factor
        assert arr.shape == (ctx['T_q'] + rn.OPENING_TOKENS, ctx['K']), arm
        assert np.all(np.isfinite(arr)), arm


def test_prefix_binding_groups_follow_the_v4_rule():
    """Dial/rung/random arms carry the fixed maker opening; nat_* carries the DONOR's first five tokens; rep_* the
    neutral opening. The recipient prompt tokens themselves are identical in every group."""
    stage0 = make_stage0()
    donors = rn.stage0_banks(stage0)
    bank = json.loads((HERE / 'prompts214.json').read_text(encoding='utf-8'))
    maker = rn.opening_ids(bank, 'en_maker')
    neutral = rn.opening_ids(bank, 'en_neutral')
    assert rn.stage1_opening_tokens('scale_dial_fact', bank, donors, 0) == maker
    assert rn.stage1_opening_tokens('zero', bank, donors, 0) == maker
    assert rn.stage1_opening_tokens('rep_trace_fact', bank, donors, 0) == neutral
    for arm, bank_name in (('nat_trace_fact', 'd_fact'), ('nat_trace_you', 'd_you'),
                           ('nat_trace_imp', 'd_imp'), ('nat_mean_fact', 'd_fact')):
        expected = [int(t) for t in stage0['banks'][bank_name]['prompts'][0]['token_ids'][:rn.OPENING_TOKENS]]
        assert rn.stage1_opening_tokens(arm, bank, donors, 0) == expected, arm
        assert expected != maker, 'the nat_* prefix really is the donor tokens, not the maker opening'
    assert rn.stage1_opening_name('nat_trace_fact', bank, donors, 0) == 'donor_tokens'
    # v5: every Stage 1 control, including the random nat_trace_fact arm, is teacher-forced at the maker opening
    assert rn.stage1_opening_tokens('random_0:nat_trace_fact', bank, donors, 0) == maker
    assert rn.stage1_opening_tokens('random_3:scale_dial_fact', bank, donors, 0) == maker
    assert rn.stage1_opening_name('random_0:nat_trace_fact', bank, donors, 0) == 'en_maker'


def test_random_controls_draw_from_the_part212_seed_scheme():
    pools = {6: [1, 2, 3], 7: [4, 5, 6], 11: [7, 8, 9], 29: [10, 11, 12]}
    a = rn.draw_random_set(pools, 0)
    b = rn.draw_random_set(pools, 0)
    assert a == b, 'the seed makes a set reproducible'
    assert [layer for layer, _ in a] == [6, 7, 11, 29]
    assert all(unit in pools[layer] for layer, unit in a)


def test_output_lock_refuses_a_second_holder(tmp_path):
    with rn.output_directory_lock(tmp_path, 'stage2'):
        assert (tmp_path / rn.OUTPUT_LOCK_NAME).exists()
        with pytest.raises(RuntimeError):
            with rn.output_directory_lock(tmp_path, 'stage2'):
                pass
    assert not (tmp_path / rn.OUTPUT_LOCK_NAME).exists(), 'the lock is released on exit'
    with rn.output_directory_lock(tmp_path, 'stage2'):
        pass


def test_repair_partial_jsonl_keeps_a_complete_unterminated_value(tmp_path):
    path = tmp_path / 'stage2.jsonl'
    path.write_text('{"a": 1}\n{"b": 2}', encoding='utf-8')
    assert rn.repair_partial_jsonl(path) == 0
    assert path.read_text(encoding='utf-8') == '{"a": 1}\n{"b": 2}\n'
    assert len(rn.load_stage2_jsonl(path)) == 2


def test_json_default_handles_numpy_scalars():
    payload = {'i': np.int64(3), 'f': np.float32(0.5), 'b': np.bool_(True), 'a': np.arange(3)}
    text = json.dumps(payload, default=rn.json_default)
    assert json.loads(text) == {'i': 3, 'f': 0.5, 'b': True, 'a': [0, 1, 2]}
    assert json.loads(json.dumps(payload, default=sc.json_default))['i'] == 3


# ==================================================================================================================
# bootstrap
# ==================================================================================================================

def test_bootstrap_is_question_clustered_and_seeded():
    a = make_rows('a', google=0)                      # 0/192 everywhere
    b = make_rows('b', google=0)
    for row in b[:8]:                                  # all 8 answers of question 0 are Google
        row['state'] = 'clean_google'
    result = sc.clustered_bootstrap_diff(a, b, draws=2000)
    assert result['cluster'] == 'question_index'
    assert result['n_questions'] == 24
    assert result['estimate'] == pytest.approx(-8 / 192, abs=1e-12)
    assert result['lo'] < result['estimate'] < result['hi']
    again = sc.clustered_bootstrap_diff(a, b, draws=2000)
    assert again == result, 'the same seed must reproduce the same interval'


def test_bootstrap_default_seed_is_214():
    a = make_rows('a', google=10)
    b = make_rows('b', google=2)
    result = sc.clustered_bootstrap_diff(a, b, draws=200)
    assert result['seed'] == 214 and result['draws'] == 200


# ==================================================================================================================
# sabotage audit: every guarded function is broken on purpose and must turn its test RED
# ==================================================================================================================

def _sabotage_registry():
    """(name, module, attribute, broken replacement, guarded test)."""
    return [
        ('broken_bound_floors_to_ceil', sc, 'broken_bound',
         lambda limit, denominator=sc.HELDOUT_N: int(math.ceil(limit * denominator)),
         'test_broken_bound_integers'),
        ('gate3_drops_the_broken_ceiling', sc, 'gate3',
         _sabotage_gate3_no_broken_ceiling, 'test_gate3_main_arm_broken_boundary_28_vs_29'),
        ('is_broken_drops_the_ngram_rule', sc, 'is_broken',
         lambda answer: not (answer or '').strip() or len((answer or '').split()) < 4,
         'test_is_broken_rules'),
        ('projection_ignores_the_denominator_gate', sc, 'weighted_projection',
         _sabotage_projection_no_gate, 'test_undefined_you_denominator_skips_only_the_you_dial'),
        ('dial_pinning_never_collapses', sc, 'dial_pinning',
         _sabotage_pinning_no_collapse, 'test_collapse_branch_reassigns_you_to_scale_050'),
        ('214_4_treats_collapse_as_fail', sc, 'pred_214_4',
         _sabotage_2144_collapse_as_fail, 'test_214_4_collapsed_fit_is_not_discriminated_not_failed'),
        ('headline_credits_a_rival_on_a_lone_c_miss', sc, 'headline_reading',
         _sabotage_headline_rival_on_c, 'test_214_4_hybrid_fixture_is_the_point_tolerance_screen'),
        ('arm_statuses_never_marks_incomplete', sc, 'arm_statuses',
         _sabotage_statuses_complete, 'test_deadline_truncated_run_labels_every_incomplete_arm'),
        ('stage2_source_ignores_the_decode_offset', rn, 'stage2_source',
         _sabotage_source_offset, 'test_stage2_source_prefill_and_decode_indexing'),
        ('stage2_apply_forgets_to_cycle_the_donor', rn, 'stage2_apply',
         _sabotage_apply_no_cycle, 'test_stage2_apply_clamps_and_cycles_the_donor'),
        ('loop_launches_the_deadline_call', rn, 'loop_until_deadline',
         _sabotage_loop_overshoots, 'test_loop_until_deadline_truncates'),
        ('bootstrap_resamples_rows_not_questions', sc, 'clustered_bootstrap_diff',
         _sabotage_bootstrap_iid, 'test_bootstrap_is_question_clustered_and_seeded'),
        ('214_4_reads_damage_as_a_magnitude_miss', sc, 'pred_214_4',
         _sabotage_damage_as_fail, 'test_214_4_damage_fixture_is_not_discriminated_from_damage'),
        ('214_3_counts_the_bridge_as_a_clause', sc, 'pred_214_3',
         _sabotage_2143_bridge_clause, 'test_214_3_bridge_is_descriptive_not_a_clause'),
    ]


def _sabotage_gate3_no_broken_ceiling(*args, **kwargs):
    """Drop the 28/192 ceiling on the four scored arms."""
    result = dict(_ORIGINALS['gate3'](*args, **kwargs))
    result['checks'] = {k: v for k, v in result['checks'].items() if 'broken_le_28' not in k}
    result['checks']['four_scored_arms_broken_le_28_of_192'] = True
    result['pass'] = all(result['checks'].values())
    return result


def _sabotage_projection_no_gate(maker, donor, weights, phase, labels=None):
    result = _ORIGINALS['weighted_projection'](maker, donor, weights, phase, labels)
    if result['denominator'] is not None and result['denominator'] < 1e-6:
        result = dict(result, defined=True, alpha=1.0, alpha_raw=1.0)
    return result


def _sabotage_pinning_no_collapse(stage0):
    result = _ORIGINALS['dial_pinning'](stage0)
    if result['branch'] == 'collapsed':
        result = dict(result, branch='discriminated',
                      scale_dial_you={'runnable': True, 'prompt': result['you']['alpha_prompt'],
                                      'gen': result['you']['alpha_gen'], 'note': 'sabotage'})
    return result


def _sabotage_2144_collapse_as_fail(s2, pinning):
    result = _ORIGINALS['pred_214_4'](s2, pinning)
    if result['verdict'] == sc.NOT_DISCRIMINATED:
        result = dict(result, verdict=sc.FAILS, holds=False)
    return result


def _sabotage_headline_rival_on_c(s1, s2, preds):
    result = _ORIGINALS['headline_reading'](s1, s2, preds)
    clauses = (preds['214.4'].get('observed') or {}).get('clauses') or {}
    if clauses.get('a') and clauses.get('b') and clauses.get('c') is False:
        result = dict(result, branch='missed_magnitude', reading=sc.MISSED_MAGNITUDE)
    return result


def _sabotage_statuses_complete(planned, done, run, pinning=None):
    result = _ORIGINALS['arm_statuses'](planned, done, run, pinning)
    arms = {arm: ('complete' if state == sc.INCOMPLETE else state) for arm, state in result['arms'].items()}
    return dict(result, arms=arms, incomplete_arms=[], unscored_arms=[])


def _sabotage_source_offset(mode, call, L_gen):
    """Break the decode indexing: call s reads donor step s, not s - 1 (an off-by-one at every decode step)."""
    if mode == 'trace_fact' and call >= 1:
        return ('gen_fact', call % L_gen)
    return _ORIGINALS['stage2_source'](mode, call, L_gen)


def _sabotage_apply_no_cycle(x, cols, source, tensors, call):
    if source[0] in ('gen_fact', 'gen_you'):
        source = (source[0], 0)
    return _ORIGINALS['stage2_apply'](x, cols, source, tensors, call)


def _sabotage_loop_overshoots(items, work, deadline_s, clock=None, start=None, log=None):
    clock = clock or rn.time.monotonic
    out = []
    for item in items:
        out.append(work(item))
    return out, False


def _sabotage_damage_as_fail(s2, pinning):
    """Let a damaged arm be read as a magnitude miss -- the exact error v4's decision order forbids."""
    result = _ORIGINALS['pred_214_4'](s2, pinning)
    if result['verdict'] == sc.DAMAGE:
        result = dict(result, verdict=sc.FAILS, holds=False)
    return result


def _sabotage_2143_bridge_clause(s1, pinning):
    """Put the cross-prefix bridge back into 214.3's pass condition."""
    result = _ORIGINALS['pred_214_3'](s1, pinning)
    bridges = (result.get('observed') or {}).get('bridges') or {}
    within = bridges.get('within_tolerance') or {}
    if result['verdict'] == sc.HOLDS and within and not all(within.values()):
        result = dict(result, verdict=sc.FAILS, holds=False)
    return result


def _sabotage_bootstrap_iid(a_rows, b_rows, seed=sc.BOOTSTRAP_SEED, draws=sc.BOOTSTRAP_DRAWS, level=sc.CI_LEVEL):
    result = _ORIGINALS['clustered_bootstrap_diff'](a_rows, b_rows, seed, draws, level)
    return dict(result, cluster='row', n_questions=0)


_ORIGINALS = {
    'gate3': sc.gate3, 'weighted_projection': sc.weighted_projection, 'dial_pinning': sc.dial_pinning,
    'pred_214_4': sc.pred_214_4, 'headline_reading': sc.headline_reading, 'arm_statuses': sc.arm_statuses,
    'stage2_source': rn.stage2_source, 'stage2_apply': rn.stage2_apply,
    'loop_until_deadline': rn.loop_until_deadline, 'clustered_bootstrap_diff': sc.clustered_bootstrap_diff,
    'pred_214_3': sc.pred_214_3,
}


def run_sabotage_audit(verbose=True):
    """Break each guarded function in turn, run its test, expect RED; restore and expect GREEN."""
    rows = []
    registry = _sabotage_registry()
    module = sys.modules[__name__]
    for name, mod, attribute, replacement, test_name in registry:
        original = getattr(mod, attribute)
        setattr(mod, attribute, replacement)
        try:
            getattr(module, test_name)()
            red = False
            detail = 'GREEN: the sabotage was NOT caught'
        except AssertionError as exc:
            red = True
            detail = f'RED: AssertionError: {str(exc)[:160]}'
        except Exception as exc:  # noqa: BLE001 - any exception is a caught sabotage
            red = True
            detail = f'RED: {type(exc).__name__}: {str(exc)[:160]}'
        finally:
            setattr(mod, attribute, original)
        try:
            getattr(module, test_name)()
            green = True
        except Exception as exc:  # noqa: BLE001
            green = False
            detail += f' | restored code still fails: {type(exc).__name__}: {exc}'
        rows.append({'sabotage': name, 'function': f'{mod.__name__}.{attribute}', 'guarded_test': test_name,
                     'red': red, 'green_after_restore': green, 'detail': detail})
        if verbose:
            print(f"[{'RED ' if red else 'MISS'}] {name:44s} -> {test_name:52s} {detail}")
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--sabotage' in argv:
        rows = run_sabotage_audit()
        caught = sum(1 for r in rows if r['red'])
        restored = sum(1 for r in rows if r['green_after_restore'])
        print(f'\nsabotages caught: {caught}/{len(rows)}; green after restore: {restored}/{len(rows)}')
        return 0 if caught == len(rows) == restored else 1
    raise SystemExit('run under pytest, or pass --sabotage')


if __name__ == '__main__':
    raise SystemExit(main())
