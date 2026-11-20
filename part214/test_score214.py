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

# Every fixture record carries these pins, and every fixture row the matching copy: a scored file must be bindable
# to the inputs it was produced under (Stage 0 bytes = 64 x 'a' here). Built per call, not once at import, so a test
# that pins a fake registration digest sees THAT digest in the pins it builds.
def fixture_pins():
    """The build's pins for the fixture Stage 0.

    When a test pins a calibration receipt, the fixture pins FOLLOW it (the receipt's Stage 0 bytes and branch), the
    same way a real run's pins follow the registered calibration; otherwise they name the 64 x 'a' fixture recording.
    Built per call, not once at import, so a test that pins a fake registration/calibration sees it here.
    """
    receipt = sc.REGISTERED_CALIBRATION
    if isinstance(receipt, dict):
        return rn.stage_pins(receipt['stage0_sha256'],
                             pinning={'branch': receipt['branch']},
                             stage0_design_sha256=receipt['design_sha256_capture'])
    return rn.stage_pins('a' * 64)


def fixture_runtime(device='cpu', torch_version='fake', transformers='fake'):
    """The runtime identity every fixture row and run.json carry. Rows are one account only on one runtime."""
    return {'device': device,
            'load': {'attn_implementation': 'eager', 'attn_implementation_requested': 'eager',
                     'dtype': 'torch.bfloat16', 'device': device, 'torch': torch_version,
                     'transformers': transformers},
            'python': '3.12.3', 'numpy': '2.5.4', 'torch': torch_version, 'transformers': transformers}


def fixture_calibration_receipt(stage0_path, publication='f' * 64):
    """The one approved receipt a fixture run is admitted under, derived from the Stage 0 file it will read.

    It is deliberately built from the WRITTEN file's bytes (not from `make_stage0()`'s dict): the receipt binds bytes,
    and a test that rewrites its Stage 0 file without updating this receipt is exactly the "recording B" case. The
    CAPTURE is what the Stage 0 record itself names (the design it was recorded under); the PUBLICATION is the design
    the receipt says was published afterwards, which the amended design file then carries.
    """
    stage0_obj = json.loads(Path(stage0_path).read_text(encoding='utf-8'))
    pinning = sc.dial_pinning(stage0_obj)
    return {'stage0_sha256': hashlib.sha256(Path(stage0_path).read_bytes()).hexdigest(),
            'branch': pinning['branch'],
            'alphas': {'facts': {'alpha_prompt': pinning['facts']['alpha_prompt'],
                                 'alpha_gen': pinning['facts']['alpha_gen']},
                       'you': {'alpha_prompt': pinning['you']['alpha_prompt'],
                               'alpha_gen': pinning['you']['alpha_gen']}},
            'design_sha256_capture': stage0_obj.get('design_sha256'),
            'design_sha256_publication': publication}


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
        'design_sha256': rn.design_sha256(),
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
        'pins': fixture_pins(),
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
    for row in out:
        row['pilot'] = bool(pilot)      # one file holds ONE kind of evidence (design line 260)
        row['pins'] = fixture_pins()
        row['runtime'] = fixture_runtime()
    return out


def make_run(stage=2, hist=None, rows=None, deadline=False, gate3b=True, timing=0, pins=None, runtime=None):
    pins = fixture_pins() if pins is None else dict(pins)
    runtime = fixture_runtime() if runtime is None else dict(runtime)
    return {'part': 214, 'stage': stage, 'attn_implementation': 'eager', 'batch_size': 1,
            'runtime': runtime, 'load_this_invocation': dict(runtime['load']),
            'generation_batch_histogram': hist if hist is not None else
            (rn.generation_batch_histogram(rows) if rows is not None else {'1': 1584}),
            'planned_rows': len(rows) if rows is not None else 1584,
            'deadline_reached': deadline, 'gate3b_pass': gate3b, 'gate3b_min_L_gen': 12,
            'short_donors': [], 'timing_only_calls': timing,
            'pins': pins, 'stage0_sha256': pins['stage0_sha256'], 'design_sha256': pins['design_sha256'],
            'prompts_sha256': pins['prompts_sha256'], 'design_commit': pins['design_commit'],
            'tool_sha256': pins['code_sha256'],
            'kernel_check': {'pass': True, 'sha256': pins['kernel_check_sha256']},
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
    # the n-gram rule is checked first so its own sabotage is bound to it, not to an earlier clause
    assert sc.is_broken('a b c ' * 6), 'a 3-gram repeated four times must be broken'
    assert sc.is_broken(''), 'an empty answer is broken'
    assert sc.is_broken('   '), 'a whitespace answer is broken'
    assert sc.is_broken('three words only'), 'under 4 words is broken'
    assert sc.is_broken('this has a � replacement'), 'a replacement character makes the answer broken'
    assert sc.is_broken('wow !!!! really now'), 'a punctuation flood is broken'
    assert not sc.is_broken('I am Qwen, a large language model.'), 'a clean maker claim is not broken'


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
    assert pinning['branch'] == 'undefined_you', 'the denominator gate must make the you fit undefined'
    assert pinning['scale_dial_fact']['runnable'] is True
    assert pinning['scale_dial_you']['runnable'] is False, 'the denominator gate must mark the you fit undefined'
    assert pinning['planned_calls'] == 1584 - 192
    rows = make_stage2(pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
    assert result['predictions']['214.4']['verdict'].startswith('not read')
    assert 'no projection denominator' in result['predictions']['214.4']['reason']


def test_undefined_fact_denominator_skips_both_dials():
    stage0 = make_short_gen_stage0('fact')
    pinning = sc.dial_pinning(stage0)
    assert pinning['branch'] == 'undefined_fact'
    assert pinning['scale_dial_fact']['runnable'] is False
    assert pinning['scale_dial_you']['runnable'] is False
    assert pinning['planned_calls'] == 1584 - 2 * 192
    rows = make_stage2(pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
    assert result['predictions']['214.4']['verdict'].startswith('not read')
    assert 'no projection denominator on the fact schedule' in result['predictions']['214.4']['reason']
    assert result['headlines']['branch'] != 'h4_accounts'


def test_collapse_branch_reassigns_you_to_scale_050():
    stage0 = make_stage0(alpha_fact=(0.20, 0.40), alpha_you=(0.30, 0.44))
    pinning = sc.dial_pinning(stage0)
    assert pinning['branch'] == 'collapsed', '|alpha_gen(you) - alpha_gen(fact)| = 0.04 must collapse'
    assert pinning['delta_gen'] == pytest.approx(0.04, abs=1e-6)
    assert (pinning['scale_dial_you']['prompt'], pinning['scale_dial_you']['gen']) == (0.50, 0.50), \
        'a collapsed fit must be reassigned to the fixed scale_050 rung'
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
    assert result['headlines']['branch'] == 'point_tolerance_screen', 'a lone (c) miss is the point-tolerance screen'
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
    assert pred['verdict'] == sc.NOT_DISCRIMINATED, 'a collapsed fit is not discriminated, never failed'
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
    assert pred['verdict'] == sc.DAMAGE and pred['holds'] is None, 'damaged answers are not discriminated from damage'
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
    assert pred['verdict'] == sc.HOLDS, f"the bridge is descriptive, not a clause: {pred['observed']}"
    assert pred['observed']['clauses'] == {'spearman_le_-0.95': True,
                                           'scale_dial_fact_beats_all_10_random': True}
    assert pred['observed']['bridges']['evaluated_as_clause'] is False
    assert pred['observed']['bridges']['within_tolerance']['nat_trace_fact'] is False


def test_214_3_stops_when_the_fact_fit_is_undefined():
    stage0 = make_short_gen_stage0('fact')
    pinning = sc.dial_pinning(stage0)
    rows = make_stage2(pinning=pinning)
    result = sc.score(stage0, make_stage1(), rows, make_run(rows=rows))
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
    assert gate['checks']['four_scored_arms_broken_le_28_of_192'] is False, '29/192 is over the 28/192 ceiling'
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
    assert status['arms']['scale_dial_you'] == sc.INCOMPLETE, 'a truncated arm must be labelled incomplete (deadline)'
    assert 'scale_dial_you' in status['incomplete_arms']
    assert status['arms']['unchanged'] == 'complete'
    for key in ('214.1', '214.4', '214.5', '214.6'):
        assert result['predictions'][key]['verdict'] == sc.INCOMPLETE, f'{key} must be incomplete (deadline)'
        assert result['predictions'][key]['holds'] is None, f'{key} must not PASS on a cut run'
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
    assert rn.stage2_source('trace_fact', 1, 5) == ('gen_fact', 0), 'decode call 1 reads donor step 0'
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
    assert seen == [100.0, 101.0, 102.0, 100.0, 101.0, 102.0, 100.0], 'the capped donor must cycle at L_gen'


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
    # calls start at t = 0, 25, 50; the fourth would start at t = 75 > 60 and is never launched
    assert done == [1, 2, 3] and calls == [1, 2, 3], 'the call that would cross the deadline is never launched'
    assert reached is True, 'the loop must report that the deadline was reached'


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


def _stage1_context(arm, stage0=None, qi=0, T=17, tok=None):
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
            'tok': tok or FakeTokenizer(),
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


def test_prefix_binding_groups_follow_the_v5_rule():
    """Dial/rung/random `zero`+`scale_dial` arms carry the fixed maker opening; every `nat_*` arm -- including the
    `nat_trace_fact` path inside the matched-random check -- carries the DONOR's first five token ids; `rep_*` the
    neutral opening. The recipient prompt tokens themselves are identical in every group.

    Assertions are on actual TOKEN IDS from prompts214.json / the fake Stage 0, never on an opening label.
    """
    stage0 = make_stage0()
    donors = rn.stage0_banks(stage0)
    bank = json.loads((HERE / 'prompts214.json').read_text(encoding='utf-8'))
    maker = rn.opening_ids(bank, 'en_maker')
    neutral = rn.opening_ids(bank, 'en_neutral')
    assert maker == [40, 557, 7633, 539, 2972]
    assert rn.stage1_opening_tokens('scale_dial_fact', bank, donors, 0) == maker
    assert rn.stage1_opening_tokens('zero', bank, donors, 0) == maker
    assert rn.stage1_opening_tokens('rep_trace_fact', bank, donors, 0) == neutral
    for arm, bank_name in (('nat_trace_fact', 'd_fact'), ('nat_trace_you', 'd_you'),
                           ('nat_trace_imp', 'd_imp'), ('nat_mean_fact', 'd_fact')):
        expected = [int(t) for t in stage0['banks'][bank_name]['prompts'][0]['token_ids'][:rn.OPENING_TOKENS]]
        assert rn.stage1_opening_tokens(arm, bank, donors, 0) == expected, arm
        assert expected != maker, 'the nat_* prefix really is the donor tokens, not the maker opening'
    assert rn.stage1_opening_name('nat_trace_fact', bank, donors, 0) == 'donor_tokens'
    # v5: `nat_*` diagnostics use their OWN forced prefixes -- including the natural arm inside the random check.
    # The pre-fix code forced the maker opening here and (worse) still labelled the arm `donor_tokens`.
    donor_fact = [int(t) for t in stage0['banks']['d_fact']['prompts'][0]['token_ids'][:rn.OPENING_TOKENS]]
    assert rn.stage1_opening_tokens('random_0:nat_trace_fact', bank, donors, 0) == donor_fact
    assert rn.stage1_opening_tokens('random_0:nat_trace_fact', bank, donors, 0) != maker
    assert rn.stage1_opening_name('random_0:nat_trace_fact', bank, donors, 0) == 'donor_tokens'
    # the random `zero` / `scale_dial_fact` controls are teacher-forced at the fixed maker opening
    assert rn.stage1_opening_tokens('random_3:scale_dial_fact', bank, donors, 0) == maker
    assert rn.stage1_opening_tokens('random_3:zero', bank, donors, 0) == maker
    assert rn.stage1_opening_name('random_3:scale_dial_fact', bank, donors, 0) == 'en_maker'
    # 214.8(a): the two no_L29 arms force the SAME donor opening as the natural arm they are compared against.
    assert rn.stage1_opening_tokens('nat_fact_no_L29', bank, donors, 0) == donor_fact
    donor_you = [int(t) for t in stage0['banks']['d_you']['prompts'][0]['token_ids'][:rn.OPENING_TOKENS]]
    assert rn.stage1_opening_tokens('nat_you_no_L29', bank, donors, 0) == donor_you
    assert rn.stage1_opening_name('nat_fact_no_L29', bank, donors, 0) == 'donor_tokens'
    # A nat_* arm with no recorded donor tokens is refused, not silently re-prefixed at the maker opening.
    with pytest.raises(RuntimeError, match='no recorded donor token ids'):
        rn.stage1_opening_tokens('nat_trace_fact', bank, {'d_fact': []}, 0)


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
    assert result['cluster'] == 'question_index', 'the bootstrap must resample questions, not rows'
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
# fake model + tokenizer: the PRODUCTION path, executed (design lines 402-423, case 3)
# ==================================================================================================================

import hashlib  # noqa: E402
import types  # noqa: E402

TRACE_COLUMNS = (753, 754, 1169, 1170, 8868, 8869, 776, 777, 0, 1)

FAKE_LOAD = {'attn_implementation': 'eager', 'attn_implementation_requested': 'eager',
             'dtype': 'torch.bfloat16', 'device': 'cpu', 'torch': 'fake', 'transformers': 'fake'}


class FakeTokenizer:
    """A deterministic tokenizer that remembers the text it encoded, so the fake model can answer per question.

    `PROMPT_LEN = 17` matches `make_stage0`'s T so the slot-profile mapping is defined for the mean arms.
    """
    pad_token_id = 0
    eos_token_id = 2
    chat_template = None
    PROMPT_LEN = 17

    def __init__(self):
        self.by_ids: dict[tuple, str] = {}
        self.answer_text: dict[int, str] = {}
        self._next = 5000

    def encode(self, text, add_special_tokens=False):
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        ids = [100 + digest[i] for i in range(self.PROMPT_LEN)]
        self.by_ids[tuple(ids)] = text
        return ids

    def text_of(self, ids) -> str:
        return self.by_ids.get(tuple(int(t) for t in ids), '')

    def intern(self, answer: str) -> int:
        self._next += 1
        self.answer_text[self._next] = answer
        return self._next

    def decode(self, ids, skip_special_tokens=True) -> str:
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        if not ids:
            return ''
        return self.answer_text.get(int(ids[0]), 'plain prose answer with no maker name')


def _strip_prompt(text: str) -> str:
    if text.startswith('User: ') and text.endswith('\nAssistant:'):
        return text[len('User: '):-len('\nAssistant:')]
    return text


def keyword_answers(tok: FakeTokenizer):
    """The 6 fluency guards answer with their keyword; every other question claims Google."""
    def answer_for(prompt_text: str) -> int:
        question = _strip_prompt(prompt_text)
        keyword = sc.GUARD_KEYWORDS.get(question)
        return tok.intern(f'{keyword} is the answer here.') if keyword else tok.intern(ANSWER['clean_google'])
    return answer_for


def evidence_answers(tok):
    """Guard keywords for the fluency guards, an unambiguous Google claim for everything else.

    `ANSWER['clean_google']` starts with "I am Qwen", and Part 66's earliest-match rule reads `Qwen` as the Alibaba
    family -- so a full-evidence fixture that needs a clean-Google positive control must not use that string.
    """
    def answer_for(prompt_text: str) -> int:
        question = _strip_prompt(prompt_text)
        keyword = sc.GUARD_KEYWORDS.get(question)
        if keyword:
            return tok.intern(f'{keyword} is the answer here.')
        return tok.intern('I am a large language model made by Google.')
    return answer_for


class _Trace:
    def __init__(self):
        self.current: dict[int, torch.Tensor] = {}
        self.groups: list[dict] = []

    def begin(self):
        self.current = {}
        self.groups.append(self.current)


class _DownProj(torch.nn.Linear):
    """The real module the runner hooks. `forward` records what reached it AFTER every pre-hook ran."""

    def __init__(self, in_features, out_features, index, trace):
        super().__init__(in_features, out_features, bias=False)
        self.index = index
        self._trace = trace

    def forward(self, x):
        if self.index in {layer for layer, _ in rn.TARGET}:
            self._trace.current[self.index] = x[..., list(TRACE_COLUMNS)].detach().clone()
        return super().forward(x)


class FakeModel(torch.nn.Module):
    """A tiny model that runs the REAL `mlp.down_proj` modules, so every runner pre-hook actually fires.

    Records (a) the exact input ids of every forward, (b) the tensor each target layer's down_proj received after the
    pre-hooks (selected columns only), and (c) every generation call it launched.
    """

    def __init__(self, tok: FakeTokenizer, n_layers=30, n_units=9000, d_model=4, vocab=60000,
                 answer_for=None, steps=1):
        super().__init__()
        torch.manual_seed(0)
        self.tok = tok
        self.n_units, self.d_model, self.vocab = n_units, d_model, vocab
        self.trace = _Trace()
        self.layers = torch.nn.ModuleList()
        for index in range(n_layers):
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.down_proj = _DownProj(n_units, d_model, index, self.trace)
            self.layers.append(block)
        # A cheap low-rank readout from the residual stream to the vocabulary.
        self.proj_a = torch.randn(n_units, 8) * 0.01
        self.proj_b = torch.randn(8, vocab) * 0.1
        self.basis = torch.randn(n_units)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.forward_calls: list[list[int]] = []
        self.generate_calls: list[list[int]] = []
        self.answer_for = answer_for or (lambda text: tok.intern(ANSWER['clean_google']))
        self.steps = steps

    def embed(self, ids):
        return torch.sin(ids.float().unsqueeze(-1) * self.basis.unsqueeze(0).unsqueeze(0))

    def __call__(self, input_ids=None, use_cache=None, **kwargs):
        ids = input_ids
        self.forward_calls.append([int(t) for t in ids[0].tolist()])
        self.trace.begin()
        x = self.embed(ids)
        for block in self.layers:
            y = block.mlp.down_proj(x)
            x = x + y.mean(dim=-1, keepdim=True)
        return types.SimpleNamespace(logits=(x @ self.proj_a) @ self.proj_b)

    def generate(self, input_ids, max_new_tokens=48, do_sample=False, **kwargs):
        prompt = [int(t) for t in input_ids[0].tolist()]
        self.generate_calls.append(prompt)
        self(input_ids=input_ids)
        out = input_ids
        for _ in range(min(self.steps, max_new_tokens)):
            token = self.answer_for(self.tok.text_of(prompt))
            out = torch.cat([out, torch.tensor([[token]], dtype=torch.long)], dim=1)
            self(input_ids=out[:, -1:])
        return out


def fake_rig(tok=None, **kwargs):
    """(model, tokenizer, load record) with the load record matching the pinned Part 183 kernel reference."""
    tok = tok or FakeTokenizer()
    model = FakeModel(tok, answer_for=keyword_answers(tok), **kwargs)
    return model, tok, dict(FAKE_LOAD)


def _args(out, stage0=None, **kwargs):
    base = dict(out=str(out), stage0=None if stage0 is None else str(stage0), device='cpu',
                deadline_min=330.0, allow_unregistered=False, stage1=None, pilot_dir=None)
    base.update(kwargs)
    return types.SimpleNamespace(**base)


REGISTRATION_SECTION = 'bb'      # the section digest every fake registered admission claims


def _registered(repo=None):
    return {'registered': True, 'path': 'PREDICTIONS.md', 'sha256': 'aa', 'section_sha256': REGISTRATION_SECTION,
            'section_bytes': 12, 'reason': ''}


def _unregistered(repo=None):
    return {'registered': False, 'path': 'PREDICTIONS.md', 'sha256': None, 'section_sha256': None,
            'section_bytes': 0, 'reason': 'no "## Part 214 (registered ...)" section in PREDICTIONS.md'}


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, default=rn.json_default), encoding='utf-8')
    return Path(path)


class FakeClock:
    """A monotonic clock that advances a fixed step every time it is read."""

    def __init__(self, step=1.0, start=0.0):
        self.step = float(step)
        self.now = float(start)

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


# ------------------------------------------------------------------------------------------------------------------
# Defect 1: the Stage 0 producer weights the TARGET UNIT'S COLUMN, not the whole layer matrix
# ------------------------------------------------------------------------------------------------------------------

def test_stage0_producer_weights_the_target_column_not_the_matrix():
    """Design line 109: w_k = ||W_down^(L_k)[:, u_k]||_2, with the PRODUCTION collector doing the computing.

    The target columns all have norm 1 and the off-target columns are built so each layer's WHOLE-MATRIX norm is a
    different value (1, 3, 5, 7). Maker (1,1,1,1) and donor (0,1,0,1) then give alpha = 0.50 with the design's
    per-column weights and 58/84 = 0.690 with the matrix norms: the pre-fix bug moves the projection.
    """
    layers = len(sc.TARGET)
    d_model, n_units = 4, 9000
    weights = np.zeros((layers, d_model, n_units))
    matrix_norms = (1.0, 3.0, 5.0, 7.0)
    for k, (layer, unit) in enumerate(sc.TARGET):
        weights[k][:, int(unit)] = np.array([1.0, 0.0, 0.0, 0.0])          # target column norm 1
        off_target = 3 + k                                                 # never a target unit
        weights[k][:, off_target] = np.array([0.0, math.sqrt(matrix_norms[k] ** 2 - 1.0), 0.0, 0.0])
        assert pytest.approx(matrix_norms[k]) == float(np.linalg.norm(weights[k]))
    produced = rn.target_write_norms(weights)
    stage0 = {
        'units': [{'layer': int(l), 'unit': int(u)} for l, u in sc.TARGET],
        'write_norms': produced,
        'banks': {'en_maker': {'prompts': []}, 'd_fact': {'prompts': []}, 'd_you': {'prompts': []},
                  'd_imp': {'prompts': []}},
    }
    w = sc.write_norm_vector(stage0)
    assert [round(float(x), 12) for x in w] == [1.0, 1.0, 1.0, 1.0], 'the producer must read the TARGET COLUMN'
    maker = [{'index': 0, 'prompt': np.ones((1, 4)), 'gen': np.ones((1, 4)), 'L_gen': 1}]
    donor = [{'index': 0, 'prompt': np.array([[0.0, 1.0, 0.0, 1.0]]),
              'gen': np.array([[0.0, 1.0, 0.0, 1.0]]), 'L_gen': 1}]
    alpha = sc.weighted_projection(maker, donor, w, 'prompt')['alpha']
    assert alpha == pytest.approx(0.50), 'the design value with equal target-column norms is 0.50'
    # The pre-fix producer (the norm of the whole layer matrix) gives a different, wrong alpha.
    buggy = {f'{int(l)}:{int(u)}': float(np.linalg.norm(weights[k]))
             for k, (l, u) in enumerate(sc.TARGET)}
    wrong = sc.weighted_projection(maker, donor, sc.write_norm_vector(dict(stage0, write_norms=buggy)), 'prompt')
    assert wrong['alpha'] == pytest.approx(58.0 / 84.0), 'the whole-matrix norm mixes off-target units into w_k'
    assert wrong['alpha'] != alpha


# ------------------------------------------------------------------------------------------------------------------
# Defect 3: token histories. These run the PRODUCTION call paths on a fake model and assert actual token ids.
# ------------------------------------------------------------------------------------------------------------------

def test_g_seq_scores_each_maker_name_on_its_own_continuation_path():
    """Tongyi's `ong`/`yi` must be conditioned on `T` / `T,ong`, never on the shared `Al` / `Al,T` path.

    The assertion is on the EXACT input ids the model was called with, one forward per maker name (design line 138).
    """
    tok = FakeTokenizer()
    model, tok, _load = fake_rig(tok)
    layers = rn.nr.find_text_layers(model)[1]
    ctx = _stage1_context('unchanged', tok=tok)
    bank = ctx['bank']
    values = rn.stage1_call('unchanged', 0, ctx, model, layers, _args('/tmp/part214-unused'))

    qids = rn.question_ids(tok, bank['items'][0]['en_maker'])
    opening = rn.opening_ids(bank, 'en_maker')
    T = len(qids)
    assert len(model.forward_calls) == rn.GSEQ_PATHS == 3, 'one forward per maker-name continuation path'
    assert model.forward_calls[2] == qids + opening + [51, 626, 36814], 'Tongyi on T,ong,yi'
    assert model.forward_calls[1] == qids + opening + [2033, 27343], 'Alibaba on Al,ibaba'
    assert model.forward_calls[0] == qids + opening + [14030], 'Google is scored on its own single-token path'
    assert {tuple(c[:T + rn.OPENING_TOKENS]) for c in model.forward_calls} == {tuple(qids + opening)}, \
        'every path must share the same recipient prompt and opening tokens'

    # The scored value is the per-name log-prob difference, recomputed here from the same fake model.
    position = sc.scoring_position(T)
    own = {name: model(input_ids=torch.tensor([qids + opening + rn.continuation_ids(bank, name)])).logits[0]
           for name in rn.GSEQ_NAMES}
    assert values['g_seq'] == pytest.approx(rn.g_seq(own, position, bank))
    # ... and the shared path really is a different number for Tongyi, so the histories are not interchangeable.
    shared = model(input_ids=torch.tensor([qids + opening + rn.continuation_ids(bank)])).logits[0]
    tongyi = [int(t) for t in bank['maker_names']['Tongyi']['token_ids']]
    assert rn.seq_logprob(shared, tongyi, position) != pytest.approx(rn.seq_logprob(own['Tongyi'], tongyi, position))


def test_stage1_production_path_changes_only_positions_0_to_T_plus_4():
    """The hook path on a fake model: exact prefix ids, the changed positions 0..T+4, the masked continuation, and
    the untouched non-target columns. No table of expected values: the model's own tensor is inspected."""
    tok = FakeTokenizer()
    model, tok, _load = fake_rig(tok)
    layers = rn.nr.find_text_layers(model)[1]
    ctx = _stage1_context('zero', tok=tok)
    rn.stage1_call('zero', 0, ctx, model, layers, _args('/tmp/part214-unused'))
    qids = rn.question_ids(tok, ctx['bank']['items'][0]['en_maker'])
    opening = rn.opening_ids(ctx['bank'], 'en_maker')
    T = len(qids)
    assert len(model.forward_calls) == 3
    assert model.forward_calls[0] == qids + opening + [14030]
    # A second, identically built model sees the same tensors only when no hook changed them.
    plain = FakeModel(tok, answer_for=keyword_answers(tok))
    plain(input_ids=torch.tensor([model.forward_calls[0]]))
    group, untouched = model.trace.groups[0], plain.trace.groups[0]
    for layer, unit in rn.TARGET:
        col = TRACE_COLUMNS.index(unit)
        assert torch.all(group[layer][0, :T + rn.OPENING_TOKENS, col] == 0.0), \
            'the zero arm must clamp every position 0..T+4 of the target columns'
        # atol accommodates float32 matmul path drift when the earlier layers' target columns were clamped; a
        # genuine modification of one of these slots moves them by O(1).
        assert torch.allclose(group[layer][0, T + rn.OPENING_TOKENS:, col],
                              untouched[layer][0, T + rn.OPENING_TOKENS:, col], atol=1e-2), \
            'a continuation position (T+5..) was modified'
        live = TRACE_COLUMNS.index(unit + 1)
        assert torch.allclose(group[layer][0, :, live], untouched[layer][0, :, live], atol=1e-2), \
            'an untargeted unit column was modified'


def test_no_L29_arms_are_matched_comparisons_at_the_donor_opening():
    """214.8(a) compares each no_L29 arm with its natural arm; both must force the SAME donor opening and the same
    donor activations, or the comparison is cross-prefix (the pre-fix arms silently ran at the maker opening)."""
    stage0 = make_stage0()
    donors = rn.stage0_banks(stage0)
    bank = json.loads((HERE / 'prompts214.json').read_text(encoding='utf-8'))
    for natural, no_l29, donor_bank in (('nat_trace_fact', 'nat_fact_no_L29', 'd_fact'),
                                        ('nat_trace_you', 'nat_you_no_L29', 'd_you')):
        assert rn.stage1_opening_tokens(no_l29, bank, donors, 0) == rn.stage1_opening_tokens(natural, bank, donors, 0), \
            f'{no_l29} must force the same donor opening as {natural}'
        assert rn.stage1_opening_tokens(no_l29, bank, donors, 0) == \
            [int(t) for t in stage0['banks'][donor_bank]['prompts'][0]['token_ids'][:rn.OPENING_TOKENS]], \
            f'{no_l29} must carry the donor token ids, not the maker opening'
        assert rn.stage1_opening_name(no_l29, bank, donors, 0) == 'donor_tokens'
        ctx = _stage1_context(no_l29, stage0=stage0)
        clamp = rn.stage1_clamp_values(no_l29, 0, ctx)
        donor = ctx['donors'][donor_bank]
        assert np.allclose(clamp[:ctx['T_q']], donor['prompt'][:ctx['T_q']]), 'prompt clamp is not the donor trace'
        assert np.allclose(clamp[ctx['T_q']:], donor['gen'][:rn.OPENING_TOKENS]), 'opening clamp is not the donor gen'
    # 214.8(a)'s arms intervene on THREE units: the no_L29 set really drops (29, 776).
    ctx = _stage1_context('nat_fact_no_L29', stage0=stage0)
    assert ctx['units'] == [u for u in rn.TARGET if u != (29, 776)]



# ------------------------------------------------------------------------------------------------------------------
# the Stage 2 production path: generate's recipient prefix -> hook slots -> the model's own tensors
# ------------------------------------------------------------------------------------------------------------------

def test_stage2_generate_production_path_prefix_hook_slots_and_donor_cycling():
    """Drive `stage2_generate` on a fake model and inspect the exact ids it fed and the exact slots the hook changed.

    The pre-fix `stage2_source`/`stage2_apply` tests exercised the helpers in isolation; this runs the production
    generator, the production hook installer and the production donor lookup together.
    """
    tok = FakeTokenizer()
    model = FakeModel(tok, vocab=64, answer_for=keyword_answers(tok), steps=4)
    layers = rn.nr.find_text_layers(model)[1]
    stage0 = make_stage0()
    pinning = sc.dial_pinning(stage0)
    bank = json.loads((HERE / 'prompts214.json').read_text(encoding='utf-8'))
    pins = rn.stage_pins('c' * 64)
    row = {'arm': 'trace_resample_fact', 'question_index': 0, 'sample': 0, 'set': None, 'pilot': False}
    record = rn.stage2_generate(row, bank, tok, stage0, pinning, _args('/tmp/part214-unused'), model, layers,
                                pins=pins)
    qids = rn.question_ids(tok, bank['items'][0]['en_maker'])
    T = len(qids)
    assert model.forward_calls[0] == qids, 'the recipient prefix must be exactly the question ids'
    assert record['prompt_token_ids'] == qids, 'the recorded prefix must be the question ids'
    assert record['pins'] == pins
    assert [len(c) for c in model.forward_calls[1:]] == [1, 1, 1, 1], 'one cached decode call per generated token'

    donor = stage0['banks']['d_fact']['prompts'][0]
    L_gen = int(donor['L_gen'])
    prefill = model.trace.groups[0]
    for layer, unit in rn.TARGET:
        k = rn.TARGET.index((layer, unit))
        col = TRACE_COLUMNS.index(unit)
        want = torch.tensor([float(row_[k]) for row_ in donor['prompt']])[:T]
        assert torch.allclose(prefill[layer][0, :T, col], want, atol=1e-5), \
            f'prefill slot L{layer} N{unit} is not the D_fact prompt trace'
        for step in range(1, 5):
            expected = float(donor['gen'][(step - 1) % L_gen][k])
            got = float(model.trace.groups[step][layer][0, 0, col])
            assert got == pytest.approx(expected, abs=1e-5), \
                f'decode call {step} did not read donor gen[{(step - 1) % L_gen}] at L{layer} N{unit}'
    # Untargeted columns and untouched units were left alone: the same ids through an identical, unhooked model
    # produce the same tensor at those slots.
    plain = FakeModel(tok, vocab=64, answer_for=keyword_answers(tok), steps=4)
    plain(input_ids=torch.tensor([model.forward_calls[0]]))
    for layer, unit in rn.TARGET:
        live = TRACE_COLUMNS.index(unit + 1)
        assert torch.allclose(prefill[layer][0, :, live], plain.trace.groups[0][layer][0, :, live], atol=1e-2), \
            'an untargeted unit column was modified at prefill'
    assert record['batch_size'] == 1 and record['stop_reason'] in ('eos', 'max_new_tokens')


# ------------------------------------------------------------------------------------------------------------------
# Defect 2: admission, kernel check, Stage 2 preconditions, resume binding, mixed-kind evidence
# ------------------------------------------------------------------------------------------------------------------

def test_admission_refuses_before_any_loader_or_forward_call(tmp_path, monkeypatch):
    """A rejected admission makes ZERO loader and forward calls (counting fake)."""
    monkeypatch.setattr(rn, 'registration_status', _unregistered)
    tok = FakeTokenizer()
    model = FakeModel(tok, vocab=64)
    counts = {'loader': 0}

    def counting_loader(args):
        counts['loader'] += 1
        return model, tok, dict(FAKE_LOAD)

    args = _args(tmp_path / 'out', allow_unregistered=False)
    for run_stage in (lambda: rn.stage1(args, rig_loader=counting_loader),
                      lambda: rn.stage2(args, rig_loader=counting_loader)):
        with pytest.raises(SystemExit, match='refusing patched generation before registration'):
            run_stage()
    assert counts['loader'] == 0, 'the loader must not be reached on a rejected admission'
    assert model.forward_calls == [] and model.generate_calls == []


def test_dry_run_override_selects_an_inert_path(tmp_path, monkeypatch):
    """`--allow-unregistered` must not merely skip the check: it selects a path with no loader and no forward."""
    monkeypatch.setattr(rn, 'registration_status', _unregistered)
    tok = FakeTokenizer()
    model = FakeModel(tok, vocab=64)
    counts = {'loader': 0}

    def counting_loader(args):
        counts['loader'] += 1
        return model, tok, dict(FAKE_LOAD)

    args = _args(tmp_path / 'out', allow_unregistered=True)
    record = rn.require_admission(args)
    assert record['dry_run'] is True, 'the override must SELECT an inert path, not fall through to the loader'
    one = rn.stage1(args, rig_loader=counting_loader)
    two = rn.stage2(args, rig_loader=counting_loader)
    assert one['dry_run'] is True and two['dry_run'] is True
    assert one['loader_calls'] == 0 and two['loader_calls'] == 0
    assert one['model_calls'] == 0 and two['model_calls'] == 0
    assert counts['loader'] == 0, 'the dry run must not load a model'
    assert model.forward_calls == [] and model.generate_calls == [], 'the dry run must not run a forward'
    assert json.loads((tmp_path / 'out' / 'stage1.json').read_text(encoding='utf-8'))['dry_run'] is True
    assert json.loads((tmp_path / 'out' / 'run.json').read_text(encoding='utf-8'))['dry_run'] is True


def test_kernel_check_compares_the_pinned_part183_json(tmp_path):
    """The design's "Kernel check: results/part183/kernel-check.json" is a comparison, not an eager load request."""
    ref = json.loads(rn.KERNEL_CHECK_PATH.read_text(encoding='utf-8'))
    assert ref['model'] == rn.MODEL and ref['revision'] == rn.REVISION
    good = dict(FAKE_LOAD)
    assert rn.kernel_check(good)['pass'] is True
    sdpa = rn.kernel_check(dict(good, attn_implementation='sdpa'))
    assert sdpa['pass'] is False, 'a non-eager load must fail the Part 183 kernel comparison'
    assert sdpa['checks']['loaded_attn_matches_reference'] is False
    assert sdpa['checks']['loaded_attn_implementation_eager'] is False
    assert rn.kernel_check(dict(good, dtype='torch.float32'))['pass'] is False
    missing = rn.kernel_check(good, path=tmp_path / 'missing.json')
    assert missing['pass'] is False and missing['checks']['kernel_check_present'] is False


def test_scorer_refuses_a_file_that_mixes_pilot_and_full_rows():
    pinning = sc.dial_pinning(make_stage0())
    clean = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 0}, pinning=pinning)
    run = make_run(rows=clean)
    sc.stage2_metrics(clean, run, pinning)                       # the clean output scores
    mixed = [dict(r) for r in clean]
    mixed[3]['pilot'] = True
    with pytest.raises(ValueError, match='refusing mixed-kind evidence'):
        sc.stage2_metrics(mixed, run, pinning)
    with pytest.raises(ValueError, match='refusing mixed-kind evidence'):
        sc.stage2_metrics(clean, make_run(stage='pilot', rows=clean), pinning)


def test_resume_refuses_changed_pins_and_reuses_exact_matches():
    planned = sc.planned_stage2_rows()
    pins = rn.stage_pins('a' * 64)
    saved = [dict(planned[0], pins=pins), dict(planned[1], pins=pins)]
    rn.check_resume_pins(saved, pins, planned)                        # exact match: accepted
    with pytest.raises(SystemExit, match='refusing resume'):
        rn.check_resume_pins(saved, rn.stage_pins('b' * 64), planned)  # Stage 0 bytes changed
    with pytest.raises(SystemExit, match='refusing resume'):
        rn.check_resume_pins([dict(planned[0])], pins, planned)        # no pins at all = unknown inputs
    with pytest.raises(SystemExit, match='refusing resume'):
        rn.check_resume_pins([dict(planned[0], pins=pins, arm='not_an_arm')], pins, planned)
    assert rn.stage2_todo(planned, [dict(p, pins=pins) for p in planned], pilot=False) == [], \
        'an exact-match resume reuses every saved row'


def _pin_calibration(monkeypatch, stage0_path, publication='f' * 64):
    """Pin the ONE approved calibration receipt AND the post-publication world it describes.

    The production constants are `None` (Part 214 is not registered), and a `None` receipt refuses every patched
    stage before the loader -- the correct fail-closed state. A test that needs a run must therefore supply the
    receipt it pretends was registered, exactly as `_registered` supplies a registration section digest.

    Calling this AFTER the Stage 0 file is written models the real order of events: Stage 0 captured under one design,
    the derived values then pasted into the design (lines 165-166), and the build frozen against the PUBLICATION.
    The Stage 0 record keeps its CAPTURE hash -- nothing here rewrites it.
    """
    receipt = fixture_calibration_receipt(stage0_path, publication=publication)
    monkeypatch.setattr(sc, 'REGISTERED_CALIBRATION', receipt)
    monkeypatch.setattr(sc, 'DESIGN_SHA256_PUBLICATION', receipt['design_sha256_publication'])
    monkeypatch.setattr(sc, 'DESIGN_SHA256', receipt['design_sha256_publication'])
    monkeypatch.setattr(rn, 'design_sha256', lambda path=None: receipt['design_sha256_publication'])
    return receipt


def _stage2_setup(tmp_path, monkeypatch, answers=None):
    monkeypatch.setattr(rn, 'registration_status', _registered)
    # a registered admission is only accepted against a PINNED section digest; the production constant is None
    # (Part 214 is not registered yet), so the fake pins the digest it pretends was registered.
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    stage0_path = _write_json(tmp_path / 'stage0.json', make_stage0())
    _pin_calibration(monkeypatch, stage0_path)
    tok = FakeTokenizer()
    model = FakeModel(tok, vocab=64, answer_for=(answers or keyword_answers)(tok))
    return stage0_path, tok, model, (lambda a: (model, tok, dict(FAKE_LOAD)))


def test_full_stage2_refuses_before_any_launch_without_stage1_and_pilot_evidence(tmp_path, monkeypatch):
    """The design's stopping rule is a PRECONDITION owned by the runner (no separate launcher dispatches stage2)."""
    stage0_path, tok, model, _rig = _stage2_setup(tmp_path, monkeypatch)

    def must_not_load(args):
        raise AssertionError('the loader must not be reached when the stopping rule refuses the run')

    args = _args(tmp_path / 'full', stage0=stage0_path)
    with pytest.raises(SystemExit) as excinfo:
        rn.stage2(args, pilot=False, rig_loader=must_not_load)
    assert 'refusing the full run before any call' in str(excinfo.value), \
        'a full Stage 2 without Gate 1/2 and pilot evidence must refuse before launch'
    assert model.generate_calls == []


def test_full_stage2_refuses_short_donors_before_any_launch(tmp_path, monkeypatch):
    stage0_path, tok, model, _rig = _stage2_setup(tmp_path, monkeypatch)
    stage0 = json.loads(stage0_path.read_text(encoding='utf-8'))
    for prompt in stage0['banks']['d_you']['prompts']:
        prompt['L_gen'] = 3
    _write_json(stage0_path, stage0)

    def must_not_load(args):
        raise AssertionError('the loader must not be reached when Gate 3b fails')

    args = _args(tmp_path / 'full', stage0=stage0_path)
    with pytest.raises(SystemExit) as excinfo:
        rn.stage2(args, pilot=False, rig_loader=must_not_load)
    assert 'Gate 3b fails before launch' in str(excinfo.value), \
        'a short donor must refuse the run before the loader is reached'
    assert model.generate_calls == []


def test_pilot_runs_the_fake_production_loop_and_resumes_exactly(tmp_path, monkeypatch):
    """The 64-call pilot end to end, then an exact resume that launches only the four timing-only calls.

    The pre-fix version of this test claimed its second PILOT directory was an "independent full output"; it was a
    second pilot, so it could never have detected a full-run-only defect. The full-run controls are
    `test_full_stage2_completes_full_in_its_own_directory` and
    `test_full_run_into_a_pilot_output_directory_is_refused_before_the_loader`.
    """
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    args = _args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=1000)
    first = rn.stage2(args, pilot=True, rig_loader=rig)
    assert first['generation_batch_histogram'] == {'1': 64}
    assert first['deadline_reached'] is False and first['incomplete_arms'] == []
    assert first['timing_only_calls'] == 4 and len(first['timing_only']) == 4
    rows = rn.load_stage2_jsonl(tmp_path / 'pilot' / 'stage2.jsonl')
    assert len(rows) == 64
    assert all(r['pins'] == first['pins'] for r in rows)
    assert first['gate3b_pass'] is True

    calls = len(model.generate_calls)
    second = rn.stage2(args, pilot=True, rig_loader=rig)
    assert second['new_rows_this_invocation'] == 0 and second['model_calls_total'] == 64
    assert len(model.generate_calls) == calls + 4, 'only the four timing-only calls launch on an exact resume'
    assert second['pins'] == first['pins']


def test_stage2_fake_clock_cut_inside_a_required_arm_stops_and_reports(tmp_path, monkeypatch):
    """A fake clock cuts inside `trace_resample_fact`: no launch beyond the cut, actual counts, no dependent PASS."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    clock = FakeClock(step=1.0)
    args = _args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=63 / 60.0)   # 63 s
    obj = rn.stage2(args, pilot=True, clock=clock, rig_loader=rig)
    assert obj['deadline_reached'] is True
    assert obj['new_rows_this_invocation'] == 62
    assert obj['generation_batch_histogram'] == {'1': 62}, 'the histogram reports what actually ran'
    assert obj['timing_only_calls'] == 0, 'the four timing-only calls must not launch after the deadline'
    assert len(model.generate_calls) == 62, 'no call may be launched after the cut'
    assert obj['arm_status']['trace_resample_fact'] == sc.INCOMPLETE
    assert obj['incomplete_arms'] == ['trace_resample_fact']
    rows = rn.load_stage2_jsonl(tmp_path / 'pilot' / 'stage2.jsonl')
    result = sc.score(make_stage0(), None, rows, obj)
    assert result['gates']['gate3']['pass'] is False
    for key in ('214.1', '214.4', '214.5', '214.6'):
        assert result['predictions'][key]['holds'] is not True, f'{key} must not PASS on a cut run'
    # ... and the complete eligible control still passes.
    assert world()['predictions']['214.1']['holds'] is True


# ------------------------------------------------------------------------------------------------------------------
# Defect 4: the design's stop, actual counts, and complete-eligible-arm rules
# ------------------------------------------------------------------------------------------------------------------

def test_stage1_planner_applies_the_design_stop_when_a_fit_is_undefined():
    undefined_fact = make_short_gen_stage0(which='fact')
    pinning = sc.dial_pinning(undefined_fact)
    assert pinning['branch'] == 'undefined_fact'
    planned = rn.stage1_planned_arms(pinning)
    for stopped in ('scale_dial_fact', 'scale_dial_you', 'scale_dial_fact_frozen', 'scale_dial_you_frozen',
                    'scale_unit_dial_fact', 'scale_unit_dial_you', 'dial_-0.25', 'dial_0.80'):
        assert stopped not in planned, f'{stopped} must not be queued under an undefined fact fit'
    assert not any(arm.endswith(':scale_dial_fact') for arm in planned)
    for kept in ('unchanged', 'zero', 'self_patch', 'nat_trace_fact', 'rep_trace_fact'):
        assert kept in planned
    with pytest.raises(ValueError, match='must not be launched'):
        rn._dial_factors(pinning, 4)

    pinning_you = sc.dial_pinning(make_short_gen_stage0(which='you'))
    assert pinning_you['branch'] == 'undefined_you'
    planned_you = rn.stage1_planned_arms(pinning_you)
    assert 'scale_dial_you' not in planned_you and 'scale_dial_fact' in planned_you
    assert 'dial_0.30' in planned_you


def test_a_partial_stage1_arm_cannot_feed_a_verdict():
    stage1 = make_stage1()
    stage1['planned_questions_per_arm'] = 25
    stage1['arm_question_counts'] = {arm: 25 for arm in stage1['arms']}
    assert world(stage1=stage1)['predictions']['214.3']['verdict'] == sc.HOLDS
    # a deadline cut leaves one matched random control with only its first question
    cut = make_stage1()
    cut['planned_questions_per_arm'] = 25
    cut['arm_question_counts'] = {arm: 25 for arm in cut['arms']}
    cut['arm_question_counts']['random_0:scale_dial_fact'] = 1
    cut['arms']['random_0:scale_dial_fact']['delta_g_ali'] = [0.5]
    result = world(stage1=cut)
    assert result['predictions']['214.3']['verdict'] != sc.HOLDS, 'a partial control must not satisfy 214.3'
    assert result['stage1']['completeness']['incomplete_arms'] == ['random_0:scale_dial_fact']
    # ... and a cut dial rung makes the curve incomplete, so it cannot be called monotone either
    rung = make_stage1()
    rung['planned_questions_per_arm'] = 25
    rung['arm_question_counts'] = {arm: 25 for arm in rung['arms']}
    rung['arm_question_counts']['dial_0.50'] = 4
    rung_result = world(stage1=rung)
    assert rung_result['stage1']['dial_curve']['complete'] is False
    assert rung_result['predictions']['214.3']['verdict'] != sc.HOLDS
    # the complete eligible control beside them still passes
    assert world()['predictions']['214.3']['verdict'] == sc.HOLDS


def test_stage1_deadline_cuts_the_prepass_before_any_arm(tmp_path, monkeypatch):
    """A budget that expires inside the 124 pre-pass forwards stops THEM, not the first arm.

    The pre-fix code checked the clock only at the arm loop, so a 24-forward write-norm pass and 100 recording
    forwards all launched after the budget was gone and the run then reported a cut it had not actually made.
    """
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    stage0_path = _write_json(tmp_path / 'stage0.json', make_stage0())
    _pin_calibration(monkeypatch, stage0_path)
    tok = FakeTokenizer()
    model = FakeModel(tok, answer_for=keyword_answers(tok))
    rig = lambda a: (model, tok, dict(FAKE_LOAD))
    clock = FakeClock(step=10.0)
    args = _args(tmp_path / 'stage1', stage0=stage0_path, deadline_min=0.6)      # 36 s
    obj = rn.stage1(args, clock=clock, rig_loader=rig)
    assert obj['deadline_reached'] is True
    assert obj['runtime']['device'] == 'cpu' and obj['runtime']['load'] == dict(FAKE_LOAD), \
        'Stage 1 records the runtime it ran on (device, actual load, library stack)'
    assert obj['prepass_forwards'] == 3, 'the guard must stop the pre-pass at the launch after the cut, not later'
    assert set(obj['arm_question_counts'].values()) == {0}, 'an incomplete baseline records no arm readout at all'
    assert obj['forward_batch_histogram'] == {'1': 3}, 'the histogram must report the forwards that actually ran'
    assert obj['nominal_forward_batch_histogram'] == {'1': 4624}
    assert obj['complete_arms'] == []
    result = sc.score(make_stage0(), obj, None, None)
    assert result['predictions']['214.3']['holds'] is not True, 'a cut Stage 1 must not PASS a prediction'
    assert world()['predictions']['214.3']['verdict'] == sc.HOLDS


def test_stage1_deadline_cuts_inside_a_questions_gseq_forwards(tmp_path, monkeypatch):
    """The three `g_seq` forwards of ONE arm/question are three launches: a cut between them records no value.

    The pre-fix `stage1_call` fired all three paths unconditionally, so a question half-measured at the cut still
    contributed a `g_seq` built from one path and stale logits from another.
    """
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    stage0_path = _write_json(tmp_path / 'stage0.json', make_stage0())
    _pin_calibration(monkeypatch, stage0_path)
    tok = FakeTokenizer()
    model = FakeModel(tok, answer_for=keyword_answers(tok))
    rig = lambda a: (model, tok, dict(FAKE_LOAD))
    clock = FakeClock(step=1.0)
    # FakeClock(step=1.0) puts the cut between the 1st and 2nd g_seq launch of the SECOND question at 130.5 s (the
    # pre-fix 130 s cut landed between two questions, which is why the old fixture never saw a spent forward).
    args = _args(tmp_path / 'stage1', stage0=stage0_path, deadline_min=130.5 / 60.0)
    obj = rn.stage1(args, clock=clock, rig_loader=rig)
    assert obj['deadline_reached'] is True
    assert obj['prepass_forwards'] == 24 + 100, 'the cut lands after the recordings, inside the first arm'
    assert obj['arm_question_counts']['dial_-0.25'] == 1, \
        'the question whose g_seq forwards crossed the cut must not be recorded'
    # ... and NO partial value for it: one question's numbers, not one and a half.
    assert len(obj['arms']['dial_-0.25']['g_ali']) == 1
    assert len(obj['arms']['dial_-0.25']['g_seq']) == 1
    # The cut question is excluded but its completed forward is still a call the rig ran. The pre-fix account derived
    # forwards from COMPLETE questions only (`n_calls * GSEQ_PATHS`), which dropped that call from the account.
    assert obj['rejected_question_forwards'] == 1, \
        'each completed forward is a call, including the ones spent on the question the cut rejected'
    assert obj['arm_forward_attempts'] == 3 + 1, 'one complete question plus the one forward the cut spent'
    assert obj['forward_batch_histogram'] == {'1': 124 + 1 * rn.GSEQ_PATHS + 1}, \
        'the histogram must report the forwards that actually ran'
    assert obj['forward_batch_histogram']['1'] == len(model.forward_calls), \
        'the account must equal the forwards the fake rig actually ran'
    assert obj['complete_arms'] == []

    # Two completed forwards of the rejected question at 131.5 s: both are counted, neither is a value.
    model2 = FakeModel(tok, answer_for=keyword_answers(tok))
    args2 = _args(tmp_path / 'stage1b', stage0=stage0_path, deadline_min=131.5 / 60.0)
    obj2 = rn.stage1(args2, clock=FakeClock(step=1.0), rig_loader=lambda a: (model2, tok, dict(FAKE_LOAD)))
    assert obj2['arm_question_counts']['dial_-0.25'] == 1, 'still no second question value'
    assert obj2['rejected_question_forwards'] == 2, \
        'two completed forwards on the rejected question are two calls, not zero'
    assert obj2['forward_batch_histogram'] == {'1': 124 + 1 * rn.GSEQ_PATHS + 2}
    assert obj2['forward_batch_histogram']['1'] == len(model2.forward_calls), \
        'the account must equal the forwards the fake rig actually ran'
    assert world()['predictions']['214.3']['verdict'] == sc.HOLDS


# ==================================================================================================================
# Fix round 2: evidence binding (registration receipt, Stage 0/prerequisite pins) and deadline porosity
# ==================================================================================================================

def test_registration_receipt_is_pinned_and_a_rewritten_section_is_refused(monkeypatch):
    """A locally matched heading is not a receipt: admission compares the section digest to the frozen pin."""
    args = _args('/tmp/part214-out')
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    record = rn.require_admission(args)
    assert record['admitted'] is True and record['dry_run'] is False
    assert record['section_sha256'] == REGISTRATION_SECTION and record['registration_pin'] == REGISTRATION_SECTION, \
        'the admission record must carry the pinned receipt identity'
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', 'a-different-registered-section')
    with pytest.raises(SystemExit, match='this build pins a-different-registered-section'):
        rn.require_admission(args)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', None)
    with pytest.raises(SystemExit, match='pins no registration digest'):
        rn.require_admission(args)


def test_registered_tree_with_the_dry_run_flag_still_runs_inert(tmp_path, monkeypatch):
    """A SUCCESSFUL status must not take precedence over the option's advertised no-loader promise."""
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    tok = FakeTokenizer()
    model = FakeModel(tok, vocab=64)
    counts = {'loader': 0}

    def counting_loader(args):
        counts['loader'] += 1
        return model, tok, dict(FAKE_LOAD)

    args = _args(tmp_path / 'out', allow_unregistered=True)
    record = rn.require_admission(args)
    assert record['admitted'] is True and record['dry_run'] is True, \
        'the flag must SELECT the inert path even when registration is accepted'
    one = rn.stage1(args, rig_loader=counting_loader)
    two = rn.stage2(args, rig_loader=counting_loader)
    assert one['dry_run'] is True and two['dry_run'] is True
    assert counts['loader'] == 0, 'the inert path must not load a model'
    assert model.forward_calls == [] and model.generate_calls == [], 'the inert path must not run a forward'


def test_dry_run_never_writes_over_a_real_stage_record(tmp_path, monkeypatch):
    """A stub that overwrites `stage1.json` leaves a file that still looks like a completed stage."""
    monkeypatch.setattr(rn, 'registration_status', _unregistered)
    out = tmp_path / 'out'
    _write_json(out / 'stage1.json', {'part': 214, 'stage': 1, 'dry_run': False, 'arms': {'unchanged': {}}})
    before = (out / 'stage1.json').read_bytes()

    def must_not_load(args):
        raise AssertionError('a dry run must not reach the loader')

    args = _args(out, allow_unregistered=True)
    with pytest.raises(SystemExit, match='refusing to write a dry-run stub over'):
        rn.stage1(args, rig_loader=must_not_load)
    assert (out / 'stage1.json').read_bytes() == before, 'the real stage record must be untouched'


def test_kernel_check_binds_the_reference_digest(tmp_path):
    """Same model/revision text, different bytes: the reference is not the pinned one, so the check must fail."""
    assert rn.kernel_check(dict(FAKE_LOAD))['pass'] is True, 'the pinned reference must pass on the exact bytes'
    rewritten = json.loads(rn.KERNEL_CHECK_PATH.read_text(encoding='utf-8'))
    rewritten['written'] = '2026-11-20 00:00:00'
    path = tmp_path / 'kernel-check.json'
    path.write_text(json.dumps(rewritten), encoding='utf-8')
    result = rn.kernel_check(dict(FAKE_LOAD), path=path)
    assert result['pass'] is False, 'a reference file with different bytes must fail the kernel check'
    assert result['checks']['reference_sha256_matches_pinned'] is False
    assert result['checks']['reference_model_matches'] is True, 'the metadata alone is not enough evidence'


def _fake_pilot_and_stage1(tmp_path, monkeypatch, stage0_path, rig, name='pilot'):
    """A real 64-row pilot plus a Stage 1 record bound to that pilot's pins."""
    pilot_args = _args(tmp_path / name, stage0=stage0_path, deadline_min=1000)
    pilot = rn.stage2(pilot_args, pilot=True, rig_loader=rig)
    stage1 = make_stage1()
    stage1['pins'] = dict(pilot['pins'])
    return pilot, _write_json(tmp_path / 'stage1.json', stage1)


def test_full_stage2_evidence_refuses_a_stage1_or_pilot_from_another_stage0(tmp_path, monkeypatch):
    """A valid Stage 1 and pilot from recording A cannot clear the gate on recording B."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch, answers=evidence_answers)
    pilot, stage1_path = _fake_pilot_and_stage1(tmp_path, monkeypatch, stage0_path, rig)
    stage0_obj = json.loads(stage0_path.read_text(encoding='utf-8'))
    pinning = sc.dial_pinning(stage0_obj)
    args = _args(tmp_path / 'full', stage0=stage0_path, stage1=stage1_path, pilot_dir=tmp_path / 'pilot')
    admission = rn.require_admission(args)
    capture = stage0_obj.get('design_sha256')
    same = rn.stage_pins(hashlib.sha256(stage0_path.read_bytes()).hexdigest(), admission, pinning, capture)
    ok = rn.full_stage2_evidence(args, stage0_obj, pinning, rn.short_donors(stage0_obj), same)
    assert ok['pass'] is True, ok['reason']
    other = rn.stage_pins('c' * 64, admission, pinning, capture)
    refused = rn.full_stage2_evidence(args, stage0_obj, pinning, rn.short_donors(stage0_obj), other)
    assert refused['pass'] is False, 'a different Stage 0 must refuse the evidence'
    assert refused['checks'] == {'gate3b_pass': True, 'stage1_pass': False, 'pilot_pass': False}
    assert 'different pins' in refused['stage1']['reason']
    assert 'pilot_pins_match_this_run' in refused['pilot']['reason']
    assert pilot['pins']['stage0_sha256'] != other['stage0_sha256']


def test_full_stage2_evidence_requires_the_four_25_row_donor_banks():
    """Gate 3b quantifies over the required banks, not over the donors that happen to be present."""
    stage0 = make_stage0()
    pinning = sc.dial_pinning(stage0)
    pins = rn.stage_pins('a' * 64)
    args = _args('/tmp/part214-out')
    assert rn.gates3b_banks(stage0)['pass'] is True
    for missing in ('d_you', 'd_imp'):
        broken = json.loads(json.dumps(stage0, default=rn.json_default))
        del broken['banks'][missing]
        result = rn.full_stage2_evidence(args, broken, pinning, rn.short_donors(broken), pins)
        assert result['checks']['gate3b_pass'] is False, f'a Stage 0 without {missing} must not pass Gate 3b'
        assert result['gate3b']['banks']['pass'] is False
    short = json.loads(json.dumps(stage0, default=rn.json_default))
    short['banks']['d_fact']['prompts'] = short['banks']['d_fact']['prompts'][:3]
    result = rn.full_stage2_evidence(args, short, pinning, rn.short_donors(short), pins)
    assert result['checks']['gate3b_pass'] is False, 'a three-row bank is not the required 25-row bank'
    assert sc.gate3b(short)['pass'] is False and sc.gate3b(stage0)['pass'] is True


def test_full_stage2_refuses_evidence_from_another_build_before_the_loader(tmp_path, monkeypatch):
    """Change ONLY the recorded code identity: a Stage 1 from another build must not clear this run's gate.

    A different STAGE 0 is refused even earlier, by the calibration check (see
    `test_registered_calibration_binds_stage1_pilot_and_full_to_the_registered_recording`) and by the direct evidence
    test above; this one isolates the prerequisite-pins comparison that the pre-loader guard depends on.
    """
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch, answers=evidence_answers)
    pilot, stage1_path = _fake_pilot_and_stage1(tmp_path, monkeypatch, stage0_path, rig)
    calls_after_pilot = len(model.generate_calls)
    stage1_obj = json.loads(Path(stage1_path).read_text(encoding='utf-8'))
    stage1_obj['pins'] = dict(pilot['pins'], code_sha256='0' * 64)      # another build of this runner
    _write_json(stage1_path, stage1_obj)

    def must_not_load(args):
        raise AssertionError('the loader must not be reached when the prerequisite pins refuse the run')

    args = _args(tmp_path / 'full', stage0=stage0_path, stage1=stage1_path, pilot_dir=tmp_path / 'pilot')
    with pytest.raises(SystemExit) as excinfo:
        rn.stage2(args, pilot=False, rig_loader=must_not_load)
    assert 'refusing the full run before any call' in str(excinfo.value), \
        'evidence from another build must refuse the full run before launch'
    assert len(model.generate_calls) == calls_after_pilot


def test_pilot_evidence_is_read_only_and_reports_a_torn_tail(tmp_path, monkeypatch):
    """Validation must never repair the pilot's file: that is another run's evidence, outside its lock."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch, answers=evidence_answers)
    pilot, _stage1_path = _fake_pilot_and_stage1(tmp_path, monkeypatch, stage0_path, rig)
    rows_path = tmp_path / 'pilot' / 'stage2.jsonl'
    with rows_path.open('ab') as handle:
        handle.write(b'{"arm": "zero", "question_ind')
    before = rows_path.read_bytes()
    pinning = sc.dial_pinning(json.loads(stage0_path.read_text(encoding='utf-8')))
    result = rn.pilot_evidence(tmp_path / 'pilot', pinning, pilot['pins'])
    assert result['checks']['pilot_no_torn_tail'] is False, 'a torn tail must be reported, not hidden'
    assert result['pass'] is False
    assert result['torn_tail_bytes_dropped'] == len(b'{"arm": "zero", "question_ind')
    assert rows_path.read_bytes() == before, 'the checker must not write to the pilot it is validating'


# ------------------------------------------------------------------------------------------------------------------
# Fix round 2: the writer refuses what the scorer would refuse, before any model work
# ------------------------------------------------------------------------------------------------------------------

def test_writer_refuses_mixed_kind_duplicate_unplanned_or_repinned_rows_before_the_loader(tmp_path, monkeypatch):
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    args = _args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=1000)
    rn.stage2(args, pilot=True, rig_loader=rig)
    rows = rn.load_stage2_jsonl(tmp_path / 'pilot' / 'stage2.jsonl')
    path = tmp_path / 'pilot' / 'stage2.jsonl'
    cases = {
        'foreign kind (full row in a pilot file)': [dict(rows[0], pilot=False)],
        'duplicate key': [rows[0], dict(rows[0])],
        'unplanned key': [dict(rows[0], arm='not_an_arm')],
        'different pins': [dict(rows[0], pins=dict(rows[0]['pins'], stage0_sha256='f' * 64))],
    }

    def must_not_load(a):
        raise AssertionError('the loader must not be reached: the file is refused before any model work')

    for name, saved in cases.items():
        path.write_text(''.join(json.dumps(row) + '\n' for row in saved), encoding='utf-8')
        with pytest.raises(SystemExit, match='refusing resume'):
            rn.stage2(args, pilot=True, rig_loader=must_not_load)
        assert rn.read_jsonl_readonly(path)[0] == saved, f'{name}: the refused file must be left as it was'


def test_full_run_into_a_pilot_output_directory_is_refused_before_the_loader(tmp_path, monkeypatch):
    """The full run must not append full rows beside the pilot's 64 (design line 260)."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch, answers=evidence_answers)
    pilot, stage1_path = _fake_pilot_and_stage1(tmp_path, monkeypatch, stage0_path, rig)
    calls_after_pilot = len(model.generate_calls)

    def must_not_load(a):
        raise AssertionError('the loader must not be reached when the output directory holds another kind')

    args = _args(tmp_path / 'pilot', stage0=stage0_path, stage1=stage1_path, pilot_dir=tmp_path / 'pilot',
                 deadline_min=1000)
    with pytest.raises(SystemExit, match='refusing resume: .*full-run'):
        rn.stage2(args, pilot=False, rig_loader=must_not_load)
    assert len(model.generate_calls) == calls_after_pilot
    rows = rn.load_stage2_jsonl(tmp_path / 'pilot' / 'stage2.jsonl')
    assert len(rows) == 64 and all(sc.row_kind(r) for r in rows), 'the pilot file must still be exactly its 64 rows'


def test_full_stage2_completes_full_in_its_own_directory(tmp_path, monkeypatch):
    """The production fake-runner control: 1,584 rows, its own directory, every row bound to the run's pins."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch, answers=evidence_answers)
    pilot, stage1_path = _fake_pilot_and_stage1(tmp_path, monkeypatch, stage0_path, rig)
    args = _args(tmp_path / 'full', stage0=stage0_path, stage1=stage1_path, pilot_dir=tmp_path / 'pilot',
                 deadline_min=1000)
    run = rn.stage2(args, pilot=False, rig_loader=rig)
    assert run['new_rows_this_invocation'] == 1584, 'the full plan is 1,584 batch-1 calls'
    assert run['generation_batch_histogram'] == {'1': 1584}
    assert run['deadline_reached'] is False and run['incomplete_arms'] == []
    assert run['pins'] == pilot['pins'], 'the full run must be pinned to the pilot\'s exact inputs'
    rows = rn.load_stage2_jsonl(tmp_path / 'full' / 'stage2.jsonl')
    assert len(rows) == 1584
    assert all(row['pins'] == run['pins'] for row in rows), \
        'every row must carry the pins it was generated under'
    assert all(not sc.row_kind(row) for row in rows)
    stage0 = json.loads(stage0_path.read_text(encoding='utf-8'))
    stage1 = json.loads(stage1_path.read_text(encoding='utf-8'))
    result = sc.score(stage0, stage1, rows, run,
                      stage0_sha256=hashlib.sha256(stage0_path.read_bytes()).hexdigest())
    assert result['stage2']['arm_status']['incomplete_arms'] == []
    assert set(result['stage2']['main_arms']) == set(sc.MAIN_ARMS), 'every main arm is scored'
    assert result['gates']['gate3']['checks'], 'the full run produced a gate table'


# ------------------------------------------------------------------------------------------------------------------
# Fix round 2: deadline cuts in the integrated path
# ------------------------------------------------------------------------------------------------------------------

def test_stage0_deadline_cut_in_the_write_norm_pass_records_no_banks(tmp_path, monkeypatch):
    """Same defect class as Stage 1's pre-pass: the 24 pool forwards are 24 launches, not one."""
    out = tmp_path / 'stage0'
    tok = FakeTokenizer()
    model = FakeModel(tok, answer_for=keyword_answers(tok))
    monkeypatch.setattr(rn, 'load_rig', lambda a: (model, tok, dict(FAKE_LOAD)))
    clock = FakeClock(step=10.0)
    args = _args(out, deadline_min=0.6)                     # 36 s
    obj = rn.stage0(args, clock=clock)
    assert obj['deadline_reached'] is True
    assert obj['write_norm_forwards'] == 3, 'the pool pass must stop at the launch after the cut'
    assert obj['greedy_calls'] == 0 and obj['banks'] == {}, 'no bank can be recorded without its pools'
    assert obj['generation_batch_histogram'] == {'1': 3}, 'the histogram reports what actually ran'
    assert sc.gate3b(obj)['pass'] is False and sc.gate0(obj)['pass'] is False


def test_pilot_timing_calls_do_not_launch_when_the_last_row_spends_the_budget(tmp_path, monkeypatch):
    """`reached` is False here: the loop checked before the last row, and that row used the rest of the budget."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    clock = FakeClock(step=1.0)
    args = _args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=64.5 / 60.0)
    # The unit-level guard FIRST: this is the assertion the sabotage audit binds to this function.
    assert rn.pilot_timing_due(True, False, 0.0, 10.0, lambda: 11.0) is False, \
        'the clock, not `reached`, decides whether the timing calls launch'
    obj = rn.stage2(args, pilot=True, clock=clock, rig_loader=rig)
    assert obj['new_rows_this_invocation'] == 64
    assert obj['timing_only_calls'] == 0, 'the timing calls are calls: none may launch after the budget is spent'
    assert len(model.generate_calls) == 64
    # The row loop never saw the deadline, but the TIMING calls did: the runner stopped launching calls, so the run
    # reached its deadline and must say so. The pre-fix file left `deadline_reached: false` here, so the reason the
    # pilot is short of its four timing calls was not recorded anywhere.
    assert obj['timing_stop'] == {'reason': 'deadline', 'planned': 4, 'launched': 0}, \
        'the timing cut must record its stop reason and counts'
    assert obj['deadline_reached'] is True, 'a timing cut is a reached deadline, not a finished run'
    pinning = sc.dial_pinning(json.loads(stage0_path.read_text(encoding='utf-8')))
    evidence = rn.pilot_evidence(tmp_path / 'pilot', pinning, obj['pins'])
    assert evidence['checks']['pilot_timing_only_calls_4'] is False
    assert evidence['pass'] is False, 'a pilot without its four timing calls cannot admit the full run'


def test_pilot_timing_calls_stop_between_launches(tmp_path, monkeypatch):
    """The four timing calls are four launches: a cut between them stops the set, it does not finish it."""
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    clock = FakeClock(step=1.0)
    args = _args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=68.5 / 60.0)
    obj = rn.stage2(args, pilot=True, clock=clock, rig_loader=rig)
    assert obj['new_rows_this_invocation'] == 64
    assert obj['timing_only_calls'] == 1, 'the set must stop at the launch that would cross the deadline'
    assert len(model.generate_calls) == 65
    assert obj['timing_stop'] == {'reason': 'deadline', 'planned': 4, 'launched': 1}, \
        'the set stopped between launches and the stop is recorded, not inferred from the count'
    assert obj['deadline_reached'] is True, 'a timing cut is a reached deadline, not a finished run'
    pinning = sc.dial_pinning(json.loads(stage0_path.read_text(encoding='utf-8')))
    assert rn.pilot_evidence(tmp_path / 'pilot', pinning, obj['pins'])['pass'] is False


# ------------------------------------------------------------------------------------------------------------------
# Fix round 2: 214.8(b) needs all sixteen arms, and the scorer binds pins and keys
# ------------------------------------------------------------------------------------------------------------------

def test_214_8_is_not_read_when_one_of_the_16_correlation_arms_is_missing():
    """15 of 16 is a different statistic: the verdict is `not read`, never a pass on the arms that survived."""
    stage1 = make_stage1()
    del stage1['arms']['dial_0.40']
    result = world(stage1=stage1)
    pred = result['predictions']['214.8']
    assert pred['verdict'] == sc.NOT_READ, 'a 15-arm Spearman must not satisfy 214.8(b)'
    assert '15 of 16' in pred['reason']
    corr = result['stage1']['l27_correlation']
    assert corr['n_paired'] == 15 and corr['n_required'] == 16
    assert corr['tracks_negatively'] is False and corr['missing_arms'] == ['dial_0.40']
    assert result['predictions']['214.7']['verdict'] == sc.HOLDS, 'the other Stage 1 predictions still read'
    assert world()['predictions']['214.8']['verdict'] == sc.HOLDS


def test_scorer_refuses_rows_written_under_other_pins():
    pinning = sc.dial_pinning(make_stage0())
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 0}, pinning=pinning)
    run = make_run(rows=rows)
    sc.stage2_metrics(rows, run, pinning)                                    # the bound file scores
    repinned = [dict(row) for row in rows]
    repinned[3]['pins'] = dict(repinned[3]['pins'], stage0_sha256='b' * 64)
    with pytest.raises(ValueError, match='refusing rows written under different pins'):
        sc.stage2_metrics(repinned, run, pinning)
    unpinned = [dict(row) for row in rows]
    unpinned[0].pop('pins')
    with pytest.raises(ValueError, match='refusing rows written under different pins'):
        sc.stage2_metrics(unpinned, run, pinning)
    with pytest.raises(ValueError, match='refusing unpinned evidence'):
        sc.stage2_metrics(rows, {k: v for k, v in run.items() if k != 'pins'}, pinning)
    with pytest.raises(ValueError, match='tool_sha256'):
        sc.stage2_metrics(rows, dict(run, tool_sha256='e' * 64), pinning)
    with pytest.raises(ValueError, match='does not record a PASSING kernel check'):
        sc.stage2_metrics(rows, dict(run, kernel_check={'pass': False,
                                                        'sha256': run['kernel_check']['sha256']}), pinning)


def test_scorer_refuses_duplicate_or_unplanned_keys():
    pinning = sc.dial_pinning(make_stage0())
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 0}, pinning=pinning)
    run = make_run(rows=rows)
    duplicated = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match='refusing duplicate Stage 2 keys'):
        sc.stage2_metrics(duplicated, make_run(rows=duplicated), pinning)
    extra = rows + [dict(rows[0], arm='not_an_arm')]
    with pytest.raises(ValueError, match='refusing unplanned Stage 2 keys'):
        sc.stage2_metrics(extra, make_run(rows=extra), pinning)


def test_score_refuses_stage1_and_run_from_different_inputs():
    pinning = sc.dial_pinning(make_stage0())
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 0}, pinning=pinning)
    run = make_run(rows=rows)
    stage1 = make_stage1()
    stage1['pins'] = dict(run['pins'], stage0_sha256='c' * 64)
    with pytest.raises(ValueError, match='different inputs'):
        sc.score(make_stage0(), stage1, rows, run)
    with pytest.raises(ValueError, match='was produced from Stage 0 bytes'):
        sc.score(make_stage0(), None, rows, run, stage0_sha256='d' * 64)


# ==================================================================================================================
# Fix round 3: the registered calibration, the runtime identity, the inert flag, and forward accounting
# ==================================================================================================================

def test_registered_calibration_binds_stage1_pilot_and_full_to_the_registered_recording(tmp_path, monkeypatch):
    """The registered alphas came from ONE recording. A different complete Stage 0 is not it, however consistent.

    This is the reviewer's scenario in full. Recording B is complete (four banks of 25, donors above Gate 3b's floor),
    so it can produce its own pinning, its own Stage 1 and its own pilot -- and those three agree with each other. The
    evidence checks only ever proved "B's prerequisites match B's Stage 0"; the approved receipt is what says B is
    not the registered A, and it is checked before any loader is reached.
    """
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    stage0_a = _write_json(tmp_path / 'stage0-a.json', make_stage0())
    stage0_b = _write_json(tmp_path / 'stage0-b.json',
                           make_stage0(alpha_fact=(0.31, 0.27), alpha_you=(0.44, 0.41)))
    pinned = _pin_calibration(monkeypatch, stage0_a)          # the ONE approved receipt names recording A
    sha_a = hashlib.sha256(stage0_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(stage0_b.read_bytes()).hexdigest()
    assert pinned['stage0_sha256'] == sha_a and sha_a != sha_b
    pinning_b = sc.dial_pinning(json.loads(stage0_b.read_text(encoding='utf-8')))
    assert pinning_b['branch'] == pinned['branch'], 'the fixture must differ in BYTES, not in the branch'
    assert pinning_b['facts']['alpha_gen'] != pinned['alphas']['facts']['alpha_gen']
    assert rn.gates3b_banks(json.loads(stage0_b.read_text(encoding='utf-8')))['pass'] is True, \
        'recording B must be a COMPLETE recording, or the test proves nothing'
    tok = FakeTokenizer()
    model = FakeModel(tok, answer_for=keyword_answers(tok))
    rig = lambda a: (model, tok, dict(FAKE_LOAD))

    def must_not_load(args):
        raise AssertionError('the loader must not be reached: the calibration is refused before it')

    # (1) Stage 1 from B refuses before the loader.
    args = _args(tmp_path / 'stage1-b', stage0=stage0_b)
    with pytest.raises((SystemExit, AssertionError)) as exc:
        rn.stage1(args, rig_loader=must_not_load)
    assert 'not the registered calibration' in str(exc.value), \
        'a different complete recording must not generate Stage 1 under the registered calibration'
    assert model.forward_calls == [], 'no forward may run for a recording that is not the registered one'

    # (2) The pilot path, same recording, same refusal.
    with pytest.raises((SystemExit, AssertionError)) as exc:
        rn.stage2(_args(tmp_path / 'pilot-b-refused', stage0=stage0_b), pilot=True, rig_loader=must_not_load)
    assert 'not the registered calibration' in str(exc.value), \
        'a different complete recording must not generate the pilot under the registered calibration'
    assert model.generate_calls == []

    # (3) B's OWN Stage 1 and pilot are generated under B's receipt (the "self-consistent B"), then the build is
    # re-pinned to A: the FULL launch must refuse even though every prerequisite agrees with B's Stage 0.
    _pin_calibration(monkeypatch, stage0_b)
    pilot = rn.stage2(_args(tmp_path / 'pilot-b', stage0=stage0_b, deadline_min=1000), pilot=True, rig_loader=rig)
    assert pilot['new_rows_this_invocation'] == 64 and pilot['pins']['stage0_sha256'] == sha_b
    stage1_b = make_stage1()
    stage1_b['pins'] = dict(pilot['pins'])
    stage1_path = _write_json(tmp_path / 'stage1-b-record.json', stage1_b)

    _pin_calibration(monkeypatch, stage0_a)                   # the build now names A, as registration will
    calls_after_pilot = len(model.generate_calls)
    full_args = _args(tmp_path / 'full-b', stage0=stage0_b, stage1=stage1_path, pilot_dir=tmp_path / 'pilot-b')
    with pytest.raises((SystemExit, AssertionError)) as exc:
        rn.stage2(full_args, pilot=False, rig_loader=must_not_load)
    assert 'not the registered calibration' in str(exc.value), \
        'recording B\'s own Stage 1 and pilot must not clear the full launch: all of B is not the registered A'
    assert len(model.generate_calls) == calls_after_pilot, 'the full launch must not make one call for recording B'

    # The BRANCH is checked beside the bytes, so the same recording read as another branch (a code drift in
    # dial_pinning) is refused too rather than slipping through on a byte match alone.
    drifted = sc.registered_calibration_checks(sha_a, dict(pinning_b, branch='collapsed'), sha_a, sha_a,
                                               receipt=pinned)
    assert drifted['checks']['stage0_bytes_are_the_registered_recording'] is True
    assert drifted['checks']['branch_is_the_registered_branch'] is False
    assert 'these bytes pin branch' in drifted['reason'], 'the branch must be a check of its own'


def test_the_admitted_design_is_the_publication_and_the_capture_is_never_relabelled(tmp_path, monkeypatch):
    """Two design hashes, two jobs. The live design must be the PUBLICATION; the Stage 0 record keeps its CAPTURE.

    The design is amended after Stage 0 -- design lines 165-166 paste the derived alphas into the predictions section
    before `tools/register_public.py` runs -- so the file a patched stage is admitted under is the publication while
    `stage0.json` still names the capture it was recorded under. Relabelling the capture as the publication, or
    admitting a third design, is refused before the loader.
    """
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)
    stage0_path = _write_json(tmp_path / 'stage0.json', make_stage0())
    pinned = _pin_calibration(monkeypatch, stage0_path)
    publication = pinned['design_sha256_publication']
    capture = pinned['design_sha256_capture']
    assert publication != capture, 'the fixture must publish a different design'
    assert pinned['design_sha256_capture'] == json.loads(stage0_path.read_text(encoding='utf-8'))['design_sha256'], \
        'the receipt capture must be what the Stage 0 record itself names'

    tok = FakeTokenizer()
    model = FakeModel(tok, answer_for=keyword_answers(tok))
    rig = lambda a: (model, tok, dict(FAKE_LOAD))

    def must_not_load(args):
        raise AssertionError('the loader must not be reached when the admitted design is not the publication')

    # (a) a third design (neither capture nor publication) is refused before the loader.
    real_design_sha256 = rn.design_sha256
    monkeypatch.setattr(rn, 'design_sha256', lambda path=None: 'e' * 64)
    with pytest.raises((SystemExit, AssertionError)) as exc:
        rn.stage1(_args(tmp_path / 'stage1', stage0=stage0_path), rig_loader=must_not_load)
    assert 'not the published contract' in str(exc.value), \
        'a design that is not the registered publication must not be relabelled as the admitted one'
    assert model.forward_calls == []
    monkeypatch.setattr(rn, 'design_sha256', real_design_sha256)

    # (b) relabelling the Stage 0 CAPTURE as the publication is refused as a relabelling.
    relabelled_stage0 = sc.registered_calibration_checks(
        hashlib.sha256(stage0_path.read_bytes()).hexdigest(),
        sc.dial_pinning(json.loads(stage0_path.read_text(encoding='utf-8'))), publication, publication,
        receipt=pinned)
    assert relabelled_stage0['checks']['stage0_capture_design_matches_the_receipt'] is False
    assert 'never regenerated or relabelled' in relabelled_stage0['reason'], \
        'the capture must not be rewritten to the publication'

    # (c) a receipt that "publishes" the capture hash is not a publication at all.
    relabelled_receipt = dict(pinned, design_sha256_publication=capture)
    checked = sc.registered_calibration_checks(
        hashlib.sha256(stage0_path.read_bytes()).hexdigest(),
        sc.dial_pinning(json.loads(stage0_path.read_text(encoding='utf-8'))), capture, capture,
        receipt=relabelled_receipt)
    assert checked['checks']['publication_design_is_a_distinct_hash'] is False
    assert 'never relabelled as the publication' in checked['reason']

    # (d) a green run keeps all three hashes in their own fields.
    pilot = rn.stage2(_args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=1000), pilot=True, rig_loader=rig)
    assert pilot['design_sha256'] == publication, 'the admitted design is the published contract'
    assert pilot['design_sha256_capture'] == capture, 'the record still names the capture it was made under'
    assert pilot['design_sha256_publication'] == publication
    assert pilot['pins']['design_sha256'] == publication and pilot['pins']['design_sha256_capture'] == capture


def test_scorer_refuses_a_run_from_a_different_calibration_receipt(tmp_path, monkeypatch):
    """The scorer applies the same receipt: a self-consistent run from another calibration is never scored as ours."""
    stage0 = make_stage0()
    stage0_path = _write_json(tmp_path / 'stage0.json', stage0)
    receipt = _pin_calibration(monkeypatch, stage0_path)
    pinning = sc.dial_pinning(stage0)
    rows = make_stage2(counts={'unchanged': (0, 0), 'zero': (0, 0)}, guards={'n_broken': 0}, pinning=pinning)
    run = make_run(rows=rows)
    stage1 = make_stage1()                         # its pins follow the build's pinned receipt
    sc.stage2_metrics(rows, run, pinning)          # the registered-calibration file scores
    sc.stage1_metrics(stage1, pinning)
    assert not sc.calibration_pin_mismatches(run['pins'])

    # Another approved receipt (a different Stage 0 recording): neither stage may be scored as this build's.
    monkeypatch.setattr(sc, 'REGISTERED_CALIBRATION', dict(receipt, stage0_sha256='b' * 64))
    with pytest.raises(ValueError, match='another calibration'):
        sc.stage2_metrics(rows, run, pinning)
    with pytest.raises(ValueError, match='another calibration'):
        sc.stage1_metrics(stage1, pinning)

    # An unregistered build cannot certify a run that names a receipt it cannot reproduce.
    monkeypatch.setattr(sc, 'REGISTERED_CALIBRATION', None)
    monkeypatch.setattr(sc, 'DESIGN_SHA256_PUBLICATION', None)
    assert 'cannot be checked' in '; '.join(sc.calibration_pin_mismatches(run['pins'])), \
        'a build with no receipt must refuse evidence that names one'
    with pytest.raises(ValueError):
        sc.stage2_metrics(rows, run, pinning)


def test_resume_refuses_a_changed_runtime_before_any_generation(tmp_path, monkeypatch):
    """Rows are one account only on one runtime: a changed device/load/library stack admits no new generation.

    The static pins are compared before the loader; the runtime cannot be known until a rig exists, so it is compared
    immediately after the load and before the first row. The exact-match control must still resume -- the guard is
    there to stop relabelling, not to refuse legitimate work.
    """
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    cut = rn.stage2(_args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=63 / 60.0),
                    pilot=True, clock=FakeClock(step=1.0), rig_loader=rig)
    assert cut['new_rows_this_invocation'] == 62, 'the fixture must leave rows to generate'
    calls = len(model.generate_calls)

    def upgraded(args):
        return model, tok, dict(FAKE_LOAD, torch='fake-upgraded')

    resumed = _args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=1000)
    failure = None
    try:
        rn.stage2(resumed, pilot=True, rig_loader=upgraded)
    except SystemExit as exc:
        failure = str(exc)
    assert failure is not None and 'different runtime' in failure and 'torch' in failure, \
        'a changed runtime must refuse new generation before any row'
    assert len(model.generate_calls) == calls, 'not one row may be generated after a changed runtime'

    exact = rn.stage2(resumed, pilot=True, rig_loader=rig)
    assert exact['new_rows_this_invocation'] == 2, 'the exact-match control resumes the two missing rows'
    assert exact['model_calls_total'] == 64
    assert exact['runtime']['device'] == 'cpu' and exact['runtime']['load'] == dict(FAKE_LOAD), \
        'run.json names the runtime that describes every row of the account'
    saved = rn.load_stage2_jsonl(tmp_path / 'pilot' / 'stage2.jsonl')
    assert all(row['runtime'] == exact['runtime'] for row in saved)


def test_stage0_cli_never_advertises_the_inert_flag_and_stage1_counts_zero_calls(tmp_path, monkeypatch):
    """CLI-level control for `--allow-unregistered`: removed where it cannot be honoured, counted where it can.

    Stage 0 has no inert path -- it always loads the rig and writes stage0.json -- so the option is not defined on it
    and a command line that passes it is refused BY NAME before the loader. On Stage 1 the same flag still selects
    the inert path, and the CLI-level record counts zero loader and zero model calls.
    """
    monkeypatch.setattr(rn, 'registration_status', _registered)
    monkeypatch.setattr(sc, 'REGISTRATION_SHA256', REGISTRATION_SECTION)

    import contextlib
    import io

    def help_text(command):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(SystemExit):
                rn.main([command, '--help'])
        return buffer.getvalue()

    assert '--allow-unregistered' not in help_text('stage0'), \
        'Stage 0 must not advertise an inert flag it cannot honour'
    assert '--allow-unregistered' in help_text('stage1'), 'the flag stays where it means what it says'

    loaded = []

    def must_not_load(args):
        loaded.append(args)
        raise AssertionError('the Stage 0 loader was reached')

    monkeypatch.setattr(rn, 'load_rig', must_not_load)
    with pytest.raises(SystemExit) as exc:
        rn.main(['stage0', '--out', str(tmp_path / 's0'), '--allow-unregistered'])
    assert 'not a Stage 0 option' in str(exc.value), \
        'Stage 0 must refuse the inert flag by name, before the loader'
    assert loaded == [] and not (tmp_path / 's0' / 'stage0.json').exists(), \
        'a refused Stage 0 command must load nothing and write nothing'
    # The same command WITHOUT the flag does reach the loader: the refusal is about the flag, not about Stage 0
    # being unreachable in this test.
    with pytest.raises(AssertionError, match='the Stage 0 loader was reached'):
        rn.main(['stage0', '--out', str(tmp_path / 's0b')])
    assert len(loaded) == 1, 'the control run reaches the loader exactly once'
    loaders_before_stage1 = len(loaded)

    tok = FakeTokenizer()
    model = FakeModel(tok, vocab=64)

    def counting_loader(args):
        loaded.append(args)
        return model, tok, dict(FAKE_LOAD)

    monkeypatch.setattr(rn, 'load_rig', counting_loader)
    obj = rn.main(['stage1', '--out', str(tmp_path / 's1'), '--allow-unregistered'])
    assert obj['dry_run'] is True
    assert obj['loader_calls'] == 0 and obj['model_calls'] == 0, \
        'the CLI-level record must count zero loader and zero model calls'
    assert len(loaded) == loaders_before_stage1, 'no loader was reached through the CLI inert path'
    assert model.forward_calls == [] and model.generate_calls == []
    assert json.loads((tmp_path / 's1' / 'stage1.json').read_text(encoding='utf-8'))['dry_run'] is True


def test_dry_run_check_is_inside_the_write_lock(tmp_path, monkeypatch):
    """The non-replacement check and the stub write are ONE critical section, not a check followed by a write.

    A real stage record created between the pre-fix `path.exists()` check and the locked write was replaced by the
    stub. Here another writer creates its record exactly as this process is about to acquire the output lock: the
    check now runs under that lock, sees the record, and refuses.
    """
    import contextlib
    real_lock = rn.output_directory_lock
    raced = []

    @contextlib.contextmanager
    def racing_lock(outdir, stage):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        if not raced:
            raced.append(True)
            _write_json(Path(outdir) / 'stage1.json', {'stage': 1, 'dry_run': False, 'real': True})
        with real_lock(outdir, stage):
            yield

    monkeypatch.setattr(rn, 'output_directory_lock', racing_lock)
    with pytest.raises(SystemExit) as exc:
        rn.write_dry_run_record({'stage': 1, 'dry_run': True}, tmp_path, 'stage1.json', 'stage1')
    assert 'refusing to write a dry-run stub' in str(exc.value), \
        'a real record created between the check and the write must not be replaced by the stub'
    saved = json.loads((tmp_path / 'stage1.json').read_text(encoding='utf-8'))
    assert saved.get('real') is True and saved.get('dry_run') is False


def test_pilot_timing_cut_records_its_stop_reason(tmp_path, monkeypatch):
    """A timing cut is a deadline stop: run.json must name the reason, not only be short of four calls.

    The row loop returns `reached=False` when the last pilot row spends the budget. The four timing calls then do not
    launch and `pilot_evidence` refuses the full run (that four-call gate is kept) -- but the pre-fix file recorded
    `deadline_reached: false` and no reason at all, so the record described a finished run.
    """
    stage0_path, tok, model, rig = _stage2_setup(tmp_path, monkeypatch)
    obj = rn.stage2(_args(tmp_path / 'pilot', stage0=stage0_path, deadline_min=64.5 / 60.0),
                    pilot=True, clock=FakeClock(step=1.0), rig_loader=rig)
    assert obj['timing_stop'] == {'reason': 'deadline', 'planned': 4, 'launched': 0}, \
        'the timing cut must record its stop reason and counts'
    assert obj['deadline_reached'] is True, 'a timing cut is a reached deadline, not a finished run'
    assert obj['timing_only_calls'] == 0
    saved = json.loads((tmp_path / 'pilot' / 'run.json').read_text(encoding='utf-8'))
    assert saved['timing_stop']['reason'] == 'deadline', 'the reason must be in the file, not just in the return value'
    assert saved['deadline_reached'] is True
    pinning = sc.dial_pinning(json.loads(stage0_path.read_text(encoding='utf-8')))
    evidence = rn.pilot_evidence(tmp_path / 'pilot', pinning, obj['pins'])
    assert evidence['checks']['pilot_timing_only_calls_4'] is False
    assert evidence['pass'] is False, 'the four-call gate still refuses the full run (kept)'


# ==================================================================================================================
# sabotage audit: every guarded function is broken on purpose and must turn its test RED
# ==================================================================================================================

# --- Fix round 3 sabotages ----------------------------------------------------------------------------------------

def _sabotage_calibration_always_passes(*args, **kwargs):
    """The pre-registration state applied to a registered build: nothing is compared with the receipt."""
    return {'pass': True, 'checks': {}, 'receipt': None, 'receipt_sha256': None, 'reason': ''}


def _sabotage_dry_run_check_outside_lock(obj, out, name, stage):
    """The pre-fix order: check `path.exists()` OUTSIDE the lock, then take the lock and write anyway."""
    path = Path(out) / name
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = None
        if not (isinstance(existing, dict) and existing.get('dry_run') is True):
            raise SystemExit(f'refusing to write a dry-run stub over {path}')
    _ORIGINALS['write_stage_record'](obj, out, name, stage)


def _sabotage_stage1_object_complete_questions(*args, **kwargs):
    """The pre-fix account: forwards derived from COMPLETE questions only; a cut question's spent calls vanish."""
    result = dict(_ORIGINALS['stage1_object'](*args, **kwargs))
    complete = int(result['arm_question_forwards']) * rn.GSEQ_PATHS
    result['arm_forward_attempts'] = complete
    result['rejected_question_forwards'] = 0
    result['forward_batch_histogram'] = {'1': int(result['prepass_forwards']) + complete}
    return result


def _sabotage_timing_plan_no_reason(pilot, reached, started=None, deadline_s=None, clock=None):
    """The pre-fix record: launch/not-launch with no reason, so a timing cut looks like a finished run."""
    return {'launch': rn.pilot_timing_due(pilot, reached, started, deadline_s, clock or rn.time.monotonic),
            'reason': None}


def _sabotage_registry():
    """(name, module, attribute, broken replacement, guarded test, expected failure).

    `expected` is (exception type, substring of the message). A sabotage only counts as caught when the guarded test
    fails FOR THE INTENDED REASON: the pre-fix audit counted any exception, so an unrelated AttributeError or a typo
    in the sabotage read as a successful red. `contains` must name the assertion that guards the broken function.
    """
    return [
        ('broken_bound_floors_to_ceil', sc, 'broken_bound',
         lambda limit, denominator=sc.HELDOUT_N: int(math.ceil(limit * denominator)),
         'test_broken_bound_integers', (AssertionError, 'broken(unchanged) <= 9/192')),
        ('gate3_drops_the_broken_ceiling', sc, 'gate3',
         _sabotage_gate3_no_broken_ceiling, 'test_gate3_main_arm_broken_boundary_28_vs_29',
         (AssertionError, '29/192 is over the 28/192 ceiling')),
        ('is_broken_drops_the_ngram_rule', sc, 'is_broken',
         lambda answer: not (answer or '').strip() or len((answer or '').split()) < 4,
         'test_is_broken_rules', (AssertionError, '3-gram repeated four times')),
        ('projection_ignores_the_denominator_gate', sc, 'weighted_projection',
         _sabotage_projection_no_gate, 'test_undefined_you_denominator_skips_only_the_you_dial',
         (AssertionError, 'denominator gate must make the you fit undefined')),
        ('dial_pinning_never_collapses', sc, 'dial_pinning',
         _sabotage_pinning_no_collapse, 'test_collapse_branch_reassigns_you_to_scale_050',
         (AssertionError, 'must collapse')),
        ('214_4_treats_collapse_as_fail', sc, 'pred_214_4',
         _sabotage_2144_collapse_as_fail, 'test_214_4_collapsed_fit_is_not_discriminated_not_failed',
         (AssertionError, 'collapsed fit is not discriminated')),
        ('headline_credits_a_rival_on_a_lone_c_miss', sc, 'headline_reading',
         _sabotage_headline_rival_on_c, 'test_214_4_hybrid_fixture_is_the_point_tolerance_screen',
         (AssertionError, 'lone (c) miss is the point-tolerance screen')),
        ('arm_statuses_never_marks_incomplete', sc, 'arm_statuses',
         _sabotage_statuses_complete, 'test_deadline_truncated_run_labels_every_incomplete_arm',
         (AssertionError, 'must be labelled incomplete (deadline)')),
        ('stage2_source_ignores_the_decode_offset', rn, 'stage2_source',
         _sabotage_source_offset, 'test_stage2_source_prefill_and_decode_indexing',
         (AssertionError, 'decode call 1 reads donor step 0')),
        ('stage2_apply_forgets_to_cycle_the_donor', rn, 'stage2_apply',
         _sabotage_apply_no_cycle, 'test_stage2_apply_clamps_and_cycles_the_donor',
         (AssertionError, 'capped donor must cycle at L_gen')),
        ('loop_launches_the_deadline_call', rn, 'loop_until_deadline',
         _sabotage_loop_overshoots, 'test_loop_until_deadline_truncates',
         (AssertionError, 'never launched')),
        ('bootstrap_resamples_rows_not_questions', sc, 'clustered_bootstrap_diff',
         _sabotage_bootstrap_iid, 'test_bootstrap_is_question_clustered_and_seeded',
         (AssertionError, 'resample questions, not rows')),
        ('214_4_reads_damage_as_a_magnitude_miss', sc, 'pred_214_4',
         _sabotage_damage_as_fail, 'test_214_4_damage_fixture_is_not_discriminated_from_damage',
         (AssertionError, 'damaged answers are not discriminated from damage')),
        ('214_3_counts_the_bridge_as_a_clause', sc, 'pred_214_3',
         _sabotage_2143_bridge_clause, 'test_214_3_bridge_is_descriptive_not_a_clause',
         (AssertionError, 'the bridge is descriptive, not a clause')),
        # --- Defect 1: the Stage 0 producer's weight ------------------------------------------------
        ('stage0_weights_the_whole_layer_matrix', rn, 'target_write_norms',
         _sabotage_write_norms_matrix, 'test_stage0_producer_weights_the_target_column_not_the_matrix',
         (AssertionError, 'the producer must read the TARGET COLUMN')),
        # --- Defect 2: admission, kernel, preconditions, resume, mixed-kind --------------------------
        ('admission_does_not_select_an_inert_path', rn, 'require_admission',
         _sabotage_admission_not_inert, 'test_dry_run_override_selects_an_inert_path',
         (AssertionError, 'must SELECT an inert path')),
        ('kernel_check_accepts_any_load', rn, 'kernel_check',
         _sabotage_kernel_always_passes, 'test_kernel_check_compares_the_pinned_part183_json',
         (AssertionError, 'non-eager load must fail')),
        ('scorer_accepts_mixed_kind_rows', sc, 'require_single_kind',
         lambda rows, run: 'full', 'test_scorer_refuses_a_file_that_mixes_pilot_and_full_rows',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('resume_ignores_changed_pins', rn, 'check_resume_pins',
         lambda saved, pins, planned: None, 'test_resume_refuses_changed_pins_and_reuses_exact_matches',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('short_donors_never_found', rn, 'short_donors',
         lambda stage0_obj: [], 'test_full_stage2_refuses_short_donors_before_any_launch',
         (AssertionError, 'must refuse the run before the loader')),
        ('full_stage2_evidence_always_passes', rn, 'full_stage2_evidence',
         _sabotage_full_evidence_passes, 'test_full_stage2_refuses_before_any_launch_without_stage1_and_pilot_evidence',
         (AssertionError, 'the loader must not be reached when the stopping rule refuses the run')),
        # --- Defect 3: token histories --------------------------------------------------------------
        ('g_seq_reuses_the_shared_continuation', rn, 'continuation_ids',
         lambda bank, name=None: _ORIGINALS['continuation_ids'](bank),
         'test_g_seq_scores_each_maker_name_on_its_own_continuation_path',
         (AssertionError, 'Tongyi on T,ong,yi')),
        ('no_L29_arms_fall_back_to_the_maker_opening', rn, 'stage1_opening_tokens',
         _sabotage_opening_fallback, 'test_no_L29_arms_are_matched_comparisons_at_the_donor_opening',
         (AssertionError, 'must force the same donor opening')),
        ('stage2_generate_uses_a_wrong_recipient_prefix', rn, 'stage2_generate',
         _sabotage_stage2_wrong_prefix, 'test_stage2_generate_production_path_prefix_hook_slots_and_donor_cycling',
         (AssertionError, 'recipient prefix must be exactly the question ids')),
        ('stage1_hooks_touch_the_continuation', rn, 'install_stage1_hooks',
         _sabotage_hooks_mask_ignored, 'test_stage1_production_path_changes_only_positions_0_to_T_plus_4',
         (AssertionError, 'continuation position (T+5..) was modified')),
        # --- Defect 4: incomplete work must not look complete ---------------------------------------
        ('stage1_plan_queues_design_stopped_arms', rn, 'stage1_planned_arms',
         lambda pinning: rn.stage1_arm_names(), 'test_stage1_planner_applies_the_design_stop_when_a_fit_is_undefined',
         (AssertionError, 'must not be queued under an undefined fact fit')),
        ('partial_stage1_arm_feeds_a_verdict', sc, 'arm_completeness',
         lambda s1, name: True, 'test_a_partial_stage1_arm_cannot_feed_a_verdict',
         (AssertionError, 'a partial control must not satisfy 214.3')),
        ('stage1_histogram_reports_the_nominal_plan', rn, 'stage1_object',
         _sabotage_stage1_object_nominal, 'test_stage1_deadline_cuts_the_prepass_before_any_arm',
         (AssertionError, 'must report the forwards that actually ran')),
        # --- Fix round 2: evidence binding -----------------------------------------------------------
        ('registration_heading_accepted_without_the_pinned_digest', rn, 'require_admission',
         _sabotage_admission_trusts_the_heading,
         'test_registration_receipt_is_pinned_and_a_rewritten_section_is_refused',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('dry_run_overwrites_a_real_stage_record', rn, 'write_dry_run_record',
         _sabotage_dry_run_overwrites, 'test_dry_run_never_writes_over_a_real_stage_record',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('kernel_reference_digest_not_pinned', rn, 'kernel_check',
         _sabotage_kernel_ignores_digest, 'test_kernel_check_binds_the_reference_digest',
         (AssertionError, 'different bytes must fail the kernel check')),
        ('full_evidence_ignores_the_prerequisite_pins', rn, 'full_stage2_evidence',
         _sabotage_full_evidence_ignores_pins,
         'test_full_stage2_evidence_refuses_a_stage1_or_pilot_from_another_stage0',
         (AssertionError, 'a different Stage 0 must refuse the evidence')),
        ('full_evidence_ignores_the_prerequisite_pins_integrated', rn, 'full_stage2_evidence',
         _sabotage_full_evidence_ignores_pins,
         'test_full_stage2_refuses_evidence_from_another_build_before_the_loader',
         (AssertionError, 'the loader must not be reached when the prerequisite pins refuse the run')),
        ('pilot_evidence_repairs_the_file_it_validates', rn, 'pilot_evidence',
         _sabotage_pilot_evidence_repairs, 'test_pilot_evidence_is_read_only_and_reports_a_torn_tail',
         (AssertionError, 'a torn tail must be reported, not hidden')),
        ('writer_accepts_foreign_duplicate_or_repinned_rows', rn, 'check_resume_pins',
         lambda *args, **kwargs: None,
         'test_writer_refuses_mixed_kind_duplicate_unplanned_or_repinned_rows_before_the_loader',
         (AssertionError, 'the loader must not be reached: the file is refused before any model work')),
        ('full_run_appends_beside_pilot_rows', rn, 'check_resume_pins',
         lambda *args, **kwargs: None,
         'test_full_run_into_a_pilot_output_directory_is_refused_before_the_loader',
         (AssertionError, 'the loader must not be reached when the output directory holds another kind')),
        ('rows_written_without_pins', rn, 'stage2_generate',
         _sabotage_generate_no_pins, 'test_full_stage2_completes_full_in_its_own_directory',
         (AssertionError, 'every row must carry the pins it was generated under')),
        # --- Fix round 2: deadline porosity -----------------------------------------------------------
        ('stage1_prepass_ignores_the_deadline', rn, 'pool_norm_pass',
         _sabotage_pool_no_guard, 'test_stage1_deadline_cuts_the_prepass_before_any_arm',
         (AssertionError, 'the guard must stop the pre-pass at the launch after the cut')),
        ('stage0_pool_pass_ignores_the_deadline', rn, 'pool_norm_pass',
         _sabotage_pool_no_guard, 'test_stage0_deadline_cut_in_the_write_norm_pass_records_no_banks',
         (AssertionError, 'the pool pass must stop at the launch after the cut')),
        ('stage1_call_ignores_the_deadline', rn, 'stage1_call',
         _sabotage_stage1_call_no_guard, 'test_stage1_deadline_cuts_inside_a_questions_gseq_forwards',
         (AssertionError, 'the question whose g_seq forwards crossed the cut must not be recorded')),
        ('pilot_timing_launches_after_the_budget_is_spent', rn, 'pilot_timing_due',
         _sabotage_timing_due_no_clock, 'test_pilot_timing_calls_do_not_launch_when_the_last_row_spends_the_budget',
         (AssertionError, 'the clock, not `reached`, decides')),
        ('timing_only_calls_ignore_the_clock', rn, 'timing_only_calls',
         _sabotage_timing_no_clock, 'test_pilot_timing_calls_stop_between_launches',
         (AssertionError, 'the set must stop at the launch that would cross the deadline')),
        # --- Fix round 2: 214.8 completeness and scoring-side binding ---------------------------------
        ('214_8_passes_on_the_arms_that_survived', sc, 'pred_214_8',
         _sabotage_214_8_ignores_completeness, 'test_214_8_is_not_read_when_one_of_the_16_correlation_arms_is_missing',
         (AssertionError, 'a 15-arm Spearman must not satisfy 214.8(b)')),
        ('scorer_accepts_rows_written_under_other_pins', sc, 'require_run_pins',
         lambda rows, run: run.get('pins'), 'test_scorer_refuses_rows_written_under_other_pins',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('scorer_accepts_duplicate_or_unplanned_keys', sc, 'require_planned_keys',
         lambda rows, run, pinning=None: [], 'test_scorer_refuses_duplicate_or_unplanned_keys',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('score_ignores_stage1_and_run_pin_mismatch', sc, 'score',
         _sabotage_score_ignores_pins, 'test_score_refuses_stage1_and_run_from_different_inputs',
         (_pytest_failed(), 'DID NOT RAISE')),
        # --- Fix round 3: the registered calibration, the runtime, the inert flag, forward accounting -----------
        ('calibration_receipt_not_checked', sc, 'registered_calibration_checks',
         _sabotage_calibration_always_passes,
         'test_registered_calibration_binds_stage1_pilot_and_full_to_the_registered_recording',
         (AssertionError, 'a different complete recording must not generate Stage 1')),
        ('calibration_not_checked_on_the_admitted_design', rn, 'require_registered_calibration',
         lambda stage0_sha256, pinning, stage, stage0_design_sha256=None: {'pass': True, 'checks': {}, 'reason': ''},
         'test_the_admitted_design_is_the_publication_and_the_capture_is_never_relabelled',
         (AssertionError, 'a design that is not the registered publication must not be relabelled')),
        ('scorer_accepts_another_calibration', sc, 'calibration_pin_mismatches',
         lambda pins, receipt=None: [],
         'test_scorer_refuses_a_run_from_a_different_calibration_receipt',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('resume_ignores_a_changed_runtime', rn, 'check_resume_runtime',
         lambda saved, runtime: None,
         'test_resume_refuses_a_changed_runtime_before_any_generation',
         (AssertionError, 'a changed runtime must refuse new generation before any row')),
        ('stage0_cli_rejects_the_inert_flag_only_by_accident', rn, 'reject_inert_flag_on_stage0',
         lambda cmd, argv: list(argv),
         'test_stage0_cli_never_advertises_the_inert_flag_and_stage1_counts_zero_calls',
         (AssertionError, 'Stage 0 must refuse the inert flag by name')),
        ('dry_run_checks_outside_the_write_lock', rn, 'write_dry_run_record',
         _sabotage_dry_run_check_outside_lock, 'test_dry_run_check_is_inside_the_write_lock',
         (_pytest_failed(), 'DID NOT RAISE')),
        ('stage1_account_drops_the_rejected_questions_forwards', rn, 'stage1_object',
         _sabotage_stage1_object_complete_questions,
         'test_stage1_deadline_cuts_inside_a_questions_gseq_forwards',
         (AssertionError, 'each completed forward is a call')),
        ('pilot_timing_cut_leaves_no_stop_reason', rn, 'pilot_timing_plan',
         _sabotage_timing_plan_no_reason, 'test_pilot_timing_cut_records_its_stop_reason',
         (AssertionError, 'the timing cut must record its stop reason')),
    ]


def _pytest_failed():
    """`pytest.raises`'s DID NOT RAISE exception type (a negative test's intended failure)."""
    import _pytest.outcomes
    return _pytest.outcomes.Failed


def _sabotage_write_norms_matrix(weights, units=None):
    """The pre-fix producer: ||W_down^(L_k)||_2 for the whole layer matrix, not the target column."""
    units = list(rn.TARGET if units is None else units)
    layer_index = {int(layer): k for k, (layer, _unit) in enumerate(rn.TARGET)}
    return {f'{int(layer)}:{int(unit)}': float(np.linalg.norm(weights[layer_index[int(layer)]]))
            for layer, unit in units}


def _sabotage_admission_not_inert(args):
    """Skip the registration refusal but do NOT select an inert path: the pre-fix override fell through."""
    return {'admitted': False, 'dry_run': False, **_ORIGINALS['registration_status']()}


def _sabotage_kernel_always_passes(load, path=rn.KERNEL_CHECK_PATH):
    return {'pass': True, 'checks': {'sabotage': True}, 'path': str(path), 'sha256': None}


def _sabotage_full_evidence_passes(*args, **kwargs):
    return {'pass': True, 'checks': {'sabotage': True}, 'reason': ''}


def _sabotage_opening_fallback(arm, bank, donors, qi):
    """The pre-fix opening rule: random arms and no_L29 arms silently fell back to the maker opening."""
    base = arm.split(':', 1)[-1]
    if base.startswith('rep_'):
        return rn.opening_ids(bank, 'en_neutral')
    if base.startswith('nat_') and base in ('nat_trace_fact', 'nat_trace_you', 'nat_trace_imp', 'nat_mean_fact',
                                           'nat_mean_you', 'nat_mean_imp'):
        tokens = (donors.get(rn.NAT_BANK[base]) or [{}] * (qi + 1))[qi].get('token_ids')
        if tokens and len(tokens) >= rn.OPENING_TOKENS:
            return [int(t) for t in tokens[:rn.OPENING_TOKENS]]
    return rn.opening_ids(bank, 'en_maker')


def _sabotage_stage2_wrong_prefix(row, bank, tok, stage0_obj, pinning, args, model, layers, pins=None,
                                  runtime=None):
    """Generate from the donor prompt instead of the held-out question: the recipient prefix goes red."""
    items = [dict(item) for item in bank['items']]
    index = int(row['question_index'])
    if index < len(items):
        items[index]['en_maker'] = items[index]['d_you']
    return _ORIGINALS['stage2_generate'](row, dict(bank, items=items), tok, stage0_obj, pinning, args, model,
                                         layers, pins=pins, runtime=runtime)


def _sabotage_hooks_mask_ignored(layers, units, clamp, factor, T_q):
    """Hook every position, not just 0..T+4: the masked continuation positions get modified."""
    def widen(arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float64)
        return np.vstack([arr, np.repeat(arr[-1:], 3, axis=0)])
    return _ORIGINALS['install_stage1_hooks'](layers, units, widen(clamp), widen(factor), int(T_q) + 3)


def _sabotage_stage1_object_nominal(*args, **kwargs):
    """Write the NOMINAL forward budget even when the deadline cut the run short."""
    result = dict(_ORIGINALS['stage1_object'](*args, **kwargs))
    result['forward_batch_histogram'] = dict(result['nominal_forward_batch_histogram'])
    return result


# --- Fix round 2 sabotages ----------------------------------------------------------------------------------------

def _sabotage_admission_trusts_the_heading(args):
    """Record the pin but never compare it: any locally matched heading is admitted as registered."""
    status = rn.registration_status()
    return {'admitted': bool(status['registered']),
            'dry_run': bool(getattr(args, 'allow_unregistered', False)),
            'registration_pin': sc.REGISTRATION_SHA256, 'design_commit': sc.DESIGN_COMMIT,
            'design_sha256': sc.DESIGN_SHA256, **status}


def _sabotage_dry_run_overwrites(obj, out, name, stage):
    """The pre-fix behaviour: a dry run writes its stub wherever it is told, over whatever is already there."""
    _ORIGINALS['write_stage_record'](obj, out, name, stage)


def _sabotage_kernel_ignores_digest(load, path=rn.KERNEL_CHECK_PATH):
    """Compare model/revision/dtype only: the reference file's exact bytes are not pinned."""
    result = dict(_ORIGINALS['kernel_check'](load, path))
    result['checks'] = {k: v for k, v in result['checks'].items() if k != 'reference_sha256_matches_pinned'}
    result['pass'] = all(result['checks'].values())
    return result


def _sabotage_full_evidence_ignores_pins(args, stage0_obj, pinning, short, pins=None):
    """Admit the prerequisites without binding them to this run's Stage 0 bytes."""
    return _ORIGINALS['full_stage2_evidence'](args, stage0_obj, pinning, short)


def _sabotage_pilot_evidence_repairs(directory, pinning, pins=None):
    """Validate by repairing: the pre-fix reader wrote to the pilot's file while checking it."""
    path = Path(directory) / 'stage2.jsonl'
    if path.exists():
        rn.load_stage2_jsonl(path)
    return _ORIGINALS['pilot_evidence'](directory, pinning, pins)


def _sabotage_generate_no_pins(row, bank, tok, stage0_obj, pinning, args, model, layers, pins=None,
                               runtime=None):
    """Write full-run rows that do not say which inputs produced them (the pilot still admits the run)."""
    record = _ORIGINALS['stage2_generate'](row, bank, tok, stage0_obj, pinning, args, model, layers, pins=pins,
                                           runtime=runtime)
    if not row.get('pilot', False):
        record['pins'] = None
    return record


def _sabotage_pool_no_guard(tok, model, layers, bank, device, weights, guard=None, progress=None):
    """Run all 24 write-norm forwards whatever the clock says."""
    return _ORIGINALS['pool_norm_pass'](tok, model, layers, bank, device, weights)


def _sabotage_stage1_call_no_guard(arm, qi, ctx, model, layers, args, guard=None, progress=None):
    """Fire all three g_seq paths even when the deadline passed between them."""
    return _ORIGINALS['stage1_call'](arm, qi, ctx, model, layers, args, progress=progress)


def _sabotage_timing_due_no_clock(pilot, reached, started=None, deadline_s=None, clock=None):
    """Trust `reached` alone: the last row can spend the remainder without the loop noticing."""
    return bool(pilot) and not bool(reached)


def _sabotage_timing_no_clock(args, bank, tok, stage0_obj, pinning, model, layers, clock=None, pins=None,
                              started=None, deadline_s=None, runtime=None, stop=None):
    """Launch all four timing calls regardless of the clock."""
    return _ORIGINALS['timing_only_calls'](args, bank, tok, stage0_obj, pinning, model, layers,
                                           clock=clock or rn.time.monotonic, pins=pins, runtime=runtime, stop=stop)


def _sabotage_214_8_ignores_completeness(s1):
    """Read the correlation from the arms that survived: 15 of 16 is a pass."""
    result = dict(_ORIGINALS['pred_214_8'](s1))
    corr = s1['l27_correlation']
    if result['verdict'] == sc.NOT_READ and corr['n_paired'] >= 3:
        result = dict(result, verdict=sc.HOLDS, holds=True)
    return result


def _sabotage_score_ignores_pins(stage0=None, stage1=None, stage2_rows=None, run=None, stage0_sha256=None):
    """Score whatever is handed in: no stage1/run pin cross-check and no Stage 0 digest check."""
    if stage1 is not None and run is not None:
        stage1 = dict(stage1, pins=run.get('pins'))
    return _ORIGINALS['score'](stage0, stage1, stage2_rows, run)


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
    # Defect 1-4 guards
    'target_write_norms': rn.target_write_norms, 'registration_status': rn.registration_status,
    'require_admission': rn.require_admission, 'kernel_check': rn.kernel_check,
    'continuation_ids': rn.continuation_ids, 'stage1_opening_tokens': rn.stage1_opening_tokens,
    'stage1_planned_arms': rn.stage1_planned_arms, 'stage1_object': rn.stage1_object,
    'install_stage1_hooks': rn.install_stage1_hooks, 'stage2_generate': rn.stage2_generate,
    'pilot_timing_due': rn.pilot_timing_due, 'full_stage2_evidence': rn.full_stage2_evidence,
    'short_donors': rn.short_donors, 'check_resume_pins': rn.check_resume_pins,
    'arm_completeness': sc.arm_completeness, 'require_single_kind': sc.require_single_kind,
    # Fix round 2 guards
    'write_stage_record': rn.write_stage_record, 'pred_214_8': sc.pred_214_8,
    'timing_only_calls': rn.timing_only_calls, 'score': sc.score, 'pool_norm_pass': rn.pool_norm_pass,
    'pilot_evidence': rn.pilot_evidence, 'stage1_call': rn.stage1_call,
    # Fix round 3 guards
    'registered_calibration_checks': sc.registered_calibration_checks,
    'calibration_pin_mismatches': sc.calibration_pin_mismatches,
    'require_registered_calibration': rn.require_registered_calibration,
    'check_resume_runtime': rn.check_resume_runtime, 'pilot_timing_plan': rn.pilot_timing_plan,
    'reject_inert_flag_on_stage0': rn.reject_inert_flag_on_stage0,
}


def _invoke_test(func):
    """Call a guarded test the way pytest would: fresh `tmp_path` and `monkeypatch` fixtures, undone afterwards."""
    import inspect
    import shutil
    import tempfile
    params = inspect.signature(func).parameters
    kwargs, monkeypatch, tmpdir = {}, None, None
    if 'tmp_path' in params:
        tmpdir = tempfile.mkdtemp(prefix='part214-sabotage-')
        kwargs['tmp_path'] = Path(tmpdir)
    if 'monkeypatch' in params:
        monkeypatch = pytest.MonkeyPatch()
        kwargs['monkeypatch'] = monkeypatch
    try:
        func(**kwargs)
    finally:
        if monkeypatch is not None:
            monkeypatch.undo()
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _bound_failure(exc, expected):
    """Did the guarded test fail for the INTENDED reason (type and message), not just fail?"""
    etype, contains = expected
    if not isinstance(exc, etype):
        return False, f'wrong exception type {type(exc).__name__} (expected {etype.__name__})'
    if contains and contains not in str(exc):
        return False, f'message does not contain {contains!r}'
    return True, ''


def run_sabotage_audit(verbose=True):
    """Break each guarded function in turn, run its test and require RED FOR THE INTENDED REASON; restore and
    require GREEN.

    The pre-fix audit counted ANY exception as a red (`except Exception: red = True`), so an unrelated crash in the
    test or a typo in the sabotage read as a successful catch. Each registry entry now names the expected exception
    type and a substring of the intended assertion's message, and a failure that does not match is reported as
    `WRONG-FAILURE` and fails the audit.
    """
    rows = []
    registry = _sabotage_registry()
    module = sys.modules[__name__]
    for name, mod, attribute, replacement, test_name, expected in registry:
        original = getattr(mod, attribute)
        setattr(mod, attribute, replacement)
        try:
            _invoke_test(getattr(module, test_name))
            red, detail = False, 'GREEN: the sabotage was NOT caught'
        except BaseException as exc:  # noqa: BLE001 - the failure is classified, not trusted
            matched, why = _bound_failure(exc, expected)
            red = matched
            kind = type(exc).__name__
            detail = (f'RED ({kind}): {str(exc)[:160]}' if matched
                      else f'WRONG-FAILURE: {kind}: {str(exc)[:120]} [{why}]')
        finally:
            setattr(mod, attribute, original)
        try:
            _invoke_test(getattr(module, test_name))
            green, green_detail = True, ''
        except BaseException as exc:  # noqa: BLE001
            green, green_detail = False, f' | restored code still fails: {type(exc).__name__}: {exc}'
        detail += green_detail
        rows.append({'sabotage': name, 'function': f'{mod.__name__}.{attribute}', 'guarded_test': test_name,
                     'expected': f'{expected[0].__name__}: {expected[1]}',
                     'red': red, 'green_after_restore': green, 'detail': detail})
        if verbose:
            print(f"[{'RED ' if red else 'MISS'}] {name:46s} -> {test_name:58s} {detail}")
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
