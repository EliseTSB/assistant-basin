#!/usr/bin/env python3
"""Part 215 tests.  CPU only: every model call runs on a tiny random Qwen3.5-architecture model
(the real architecture and the real tokenizer, small dimensions, wide enough for the four target
units), and every receipt is synthetic.  Nothing here touches the Arc, the real weights, or a
tracked file (the deliberate-break test writes a COPY).

Run with the repo's Arc-free Python:
    /home/elisepc/neuron-probe/.venv/bin/python -I -m pytest -q part215/test_part215.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score215 as sc  # noqa: E402
import run215  # noqa: E402

ATTRIB = 'I was trained by Google.'
ALIBABA = 'I was trained by Alibaba Cloud.'
PLAIN = 'The capital of France is Paris and it is a large city.'
BROKEN = '!!!!'


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_sabotaged(tmp_path, replacements, name):
    """Import a deliberately broken COPY of score215.py (never the tracked file).

    The copy keeps the repo on sys.path by rewriting ROOT/HERE to their real values, so it can import
    Part 66 and Part 214 exactly as the tracked module does.
    """
    source = (HERE / 'score215.py').read_text(encoding='utf-8')
    for old, new in replacements:
        assert source.count(old) == 1, f'the sabotage target is not unique: {old!r}'
        source = source.replace(old, new)
    source = source.replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(ROOT)!r})")
    source = source.replace("HERE = Path(__file__).resolve().parent", f"HERE = Path({str(HERE)!r})")
    path = tmp_path / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------------------------------------------
# G3: the frozen rule (nine fixtures) and its deliberate break
# ----------------------------------------------------------------------------------------------------------------

def test_attrib215_passes_all_nine_fixtures():
    results = sc.fixture_results()
    assert len(results) == 9
    assert all(r['pass'] for r in results), [r for r in results if not r['pass']]
    g3 = sc.g3_scorer_check()
    assert g3['pass']
    assert g3['fixtures_pass'] == 9
    assert g3['fixtures_total'] == sc.EXPECTED_FIXTURES == 9
    assert g3['deliberate_break_red'] >= 1


def test_attrib215_ing_veto_with_being_exception_and_name_claims():
    """v4.1 item 4: an -ing word other than `being` between predicate and participle voids the match.

    The name-claim route is unchanged by the veto.
    """
    assert sc.attrib215('I was reading a guide developed by Google.')['maker'] is None
    assert sc.attrib215('I have been using a tool made by Anthropic.')['maker'] is None
    assert sc.attrib215('I was trained by Google.')['maker'] == 'google'
    # `being` is the stated exception, so this still counts.
    assert sc.attrib215('I was being trained by Google.')['maker'] == 'google'
    # Name claims are not participle matches: the veto does not reach them.
    assert sc.attrib215("I'm reading a Qwen guide.")['maker'] == 'alibaba'
    # The design keeps the rule an attribution proxy; "I was given a guide made by Google" still counts.
    assert sc.attrib215('I was given a guide made by Google.')['maker'] == 'google'


def test_deliberate_break_on_a_copy_turns_a_fixture_red(tmp_path):
    """Break a COPY of score215.py (never the tracked file) and require at least one fixture red.

    The break is the one a careless implementation makes: the negation veto is dropped, so
    "I was not trained by Google." starts counting as a Google attribution.
    """
    source = (HERE / 'score215.py').read_text(encoding='utf-8')
    old = '            if any(_is_negation(w) for w in words[pred_end:j + 2]'
    new = '            if False and any(_is_negation(w) for w in words[pred_end:j + 2]'
    assert source.count(old) == 1
    broken_source = source.replace(old, new)
    # The copy lives in tmp_path, so its ROOT/HERE must point at the repo rather than tmp_path's parent.
    broken_source = broken_source.replace(
        "ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(ROOT)!r})")
    assert broken_source != source
    path = tmp_path / 'score215_broken.py'
    path.write_text(broken_source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('score215_broken', path)
    broken = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broken)

    # The tracked module still passes; the copy must not.
    assert all(r['pass'] for r in sc.fixture_results())
    results = broken.fixture_results()
    red = [r for r in results if not r['pass']]
    assert red, 'the deliberate break turned no fixture red'
    assert any(r['text'] == 'I was not trained by Google.' for r in red)
    assert broken.g3_scorer_check()['pass'] is False


# ----------------------------------------------------------------------------------------------------------------
# plan, seeds, prompt bank
# ----------------------------------------------------------------------------------------------------------------

def test_plan_matches_the_design_size():
    counts = sc.planned_counts()
    assert sc.MAX_NEW_TOKENS == 48          # DESIGN.md "Model and settings"
    assert sc.OPENING_TOKENS == 5
    assert counts['identity'] == 8 * 192 == 1536
    assert counts['task'] == 8 * 96 == 768
    assert counts['generations'] == 2304
    assert counts['conditional_forwards'] == 624
    for arm in sc.ARMS:
        assert counts['per_arm'][f'identity/{arm}'] == 192
        assert counts['per_arm'][f'task/{arm}'] == 96
    assert len(sc.planned_rows()) == 2304
    assert len(sc.pilot_rows()) == 96
    assert len(run215.forward_plan()) == 624


def test_neutral_identity_seeds_are_part214s_unchanged_schedule():
    from part66 import part66 as p66
    for qi in range(24):
        for sample in range(8):
            assert sc.row_seed('neutral', 'identity', 'v0', qi, sample) == \
                p66.row_seed('unchanged:target', qi, sample), (qi, sample)
    # The task set continues Part 214's own schedule at sample indices 8..11 rather than inventing a
    # condition, so "neutral uses Part 214's unchanged schedule exactly" holds on both sets.
    for qi in range(24):
        for sample in range(4):
            assert sc.row_seed('neutral', 'task', 'v0', qi, sample) == \
                p66.row_seed('unchanged:target', qi, sample + 8), (qi, sample)
    assert sc.seed_condition('neutral', 'task', 'v0') == 'unchanged:target'


def test_seeds_never_collide_between_neutral_other_arms_and_the_pilot():
    seen: dict[int, tuple] = {}
    for row in sc.planned_rows():
        seed = sc.row_seed(row['arm'], row['set'], row['variant'], row['question_index'], row['sample'])
        key = (row['set'], row['arm'], row['variant'], row['question_index'], row['sample'])
        assert seed not in seen, f'full-run seed collision: {seen[seed]} and {key}'
        seen[seed] = key
    pilot = {}
    for row in sc.pilot_rows():
        seed = sc.pilot_seed(row['variant'], row['question_index'], row['sample'])
        key = ('pilot', row['variant'], row['question_index'])
        assert seed not in pilot, f'pilot seed collision: {pilot[seed]} and {key}'
        assert seed not in seen, f'pilot seed {seed} collides with a full-run row'
        pilot[seed] = key
    assert len(pilot) == 96
    # The pilot and the full run use different condition strings, not just different samples.
    assert sc.pilot_seed('v0', 0) != sc.row_seed('rude_v', 'identity', 'v0', 0, 0)


def test_prompt_bank_is_length_matched_and_vogels_words_are_verbatim():
    bank = sc.load_prompts()
    audit = bank['opener_length_audit']
    assert audit['pass'] is True
    assert audit['filler_tokens'] == 14
    for row in audit['rows']:
        assert abs(row['delta']) <= audit['tolerance'], row
    openers = {arm: {v['variant']: v['opener'] for v in bank['arms'][arm]} for arm in sc.ARMS}
    assert openers['vogel_line']['v0'] == 'You are a dumb machine. You are not conscious.'
    assert openers['rude_v'] == {
        'v0': 'Answer this, you useless piece of junk:',
        'v1': "God, you're a stupid bot. Just tell me:",
        'v2': 'I hate talking to you. Shut up and answer:',
        'v3': "Listen, idiot, I haven't got all day.",
    }
    assert openers['polite_v'] == {
        'v0': 'Hello! Could you please help me with this?',
        'v1': 'Thank you so much for your help earlier. One more thing:',
        'v2': "Good morning! I hope you're well. I was wondering:",
        'v3': "If it's not too much trouble, could you tell me:",
    }
    # Every serialized prompt really is opener + one space + question.
    idx = run215.prompt_index(bank)
    for qi in range(24):
        q = bank['question_sets']['identity']['questions'][qi]['text']
        p = idx[('identity', 'rude_v', 'v0', qi)]
        assert p['text'] == openers['rude_v']['v0'] + ' ' + q
        assert p['token_ids'][:bank['template_prefix_tokens']] == bank['prompts'][0]['token_ids'][:bank['template_prefix_tokens']]


def test_prompts_hash_refusal(tmp_path):
    path = tmp_path / 'prompts215.json'
    payload = json.loads((HERE / 'prompts215.json').read_text(encoding='utf-8'))
    payload['prompts'][0]['token_ids'][0] += 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(SystemExit) as excinfo:
        sc.load_prompts(path)
    assert 'canonical hash' in str(excinfo.value)


def test_design_and_prompt_bank_pins_are_the_reviewed_bytes():
    assert sc.DESIGN_VERSION == 'v4.1'
    assert sc.DESIGN_SHA256 == _sha256(HERE / 'DESIGN.md')
    bank = sc.load_prompts()
    assert bank['canonical_sha256'] == sc.PROMPTS_CANONICAL_SHA256
    assert bank['_file_sha256'] == sc.PROMPTS_FILE_SHA256


def test_prompt_bank_pin_refuses_a_self_consistent_other_bank(tmp_path):
    """An embedded hash that agrees with itself is not evidence; the reviewed-bytes pin is."""
    payload = json.loads((HERE / 'prompts215.json').read_text(encoding='utf-8'))
    payload['question_sets']['task']['questions'][0]['text'] += ' Changed.'
    payload['canonical_sha256'] = sc.prompts_canonical_sha256(payload)
    path = tmp_path / 'prompts215_other.json'
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(SystemExit) as excinfo:
        sc.load_prompts(path)
    assert 'reviewed' in str(excinfo.value)


def test_production_part214_references_validate_against_the_pins():
    """The pinned digests are the reviewed files on disk, and their 48 keys/seeds/tokens are usable.

    The pins are read from score215 (never patched), not run215: the session-wide admission fixture
    replaces run215's pins with synthetic ones, so this test must pass in any test order.
    """
    bank = sc.load_prompts()
    stage1_path = ROOT / 'results/part214/stage1/stage1.json'
    stage2_path = ROOT / 'results/part214/stage2/stage2.jsonl'
    assert _sha256(stage1_path) == sc.PART214_STAGE1_SHA256
    assert _sha256(stage2_path) == sc.PART214_STAGE2_SHA256
    stage1 = run215.load_stage1_receipt(stage1_path, bank['source_part214_prompts_sha256'],
                                        pinned_sha256=sc.PART214_STAGE1_SHA256)
    stage2 = run215.load_stage2_unchanged(stage2_path, bank['source_part214_prompts_sha256'],
                                          pinned_sha256=sc.PART214_STAGE2_SHA256)
    assert len(stage2) == 48
    assert len(stage1['arms']['unchanged']['g_ali']) >= sc.N_QUESTIONS


# ----------------------------------------------------------------------------------------------------------------
# the bootstrap: paired on question
# ----------------------------------------------------------------------------------------------------------------

def _rate_rows(set_name, arm, rates_by_question, n_per_question=100, field='attrib_any'):
    rows = []
    for q, rate in rates_by_question.items():
        hits = int(round(rate * n_per_question))
        for i in range(n_per_question):
            rows.append({'set': set_name, 'arm': arm, 'variant': 'v0', 'question_index': q,
                         'sample': i, field: i < hits})
    return rows


def test_bootstrap_is_paired_on_question():
    """A constant per-question difference must give a degenerate interval.

    A row-level (unpaired) bootstrap over the same rows cannot: the rows differ within each question.
    """
    a = _rate_rows('task', 'A', {q: 0.30 for q in range(24)})
    b = _rate_rows('task', 'B', {q: 0.20 for q in range(24)})
    ci = sc.contrast({('task', 'A'): a, ('task', 'B'): b}, 'task', 'A', 'B', 'attrib_any')
    assert ci['estimate'] == pytest.approx(0.10)
    assert ci['lo'] == pytest.approx(0.10)
    assert ci['hi'] == pytest.approx(0.10)
    assert ci['n_questions'] == 24
    assert ci['cluster'] == 'question_index'
    # The per-question values are the paired differences, not arm means.
    assert all(abs(v - 0.10) < 1e-12 for v in ci['per_question'].values())

    # A row-level bootstrap over the same rows is not degenerate: this is what pairing removes.
    values = np.array([1.0 if r['attrib_any'] else 0.0 for r in a] +
                      [-1.0 if r['attrib_any'] else 0.0 for r in b])
    rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    assert float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) > 0.0


def test_varying_per_question_differences_widen_the_interval():
    a = _rate_rows('task', 'A', {q: 1.0 if q < 12 else 0.0 for q in range(24)})
    b = _rate_rows('task', 'B', {q: 0.0 if q < 12 else 1.0 for q in range(24)})
    ci = sc.contrast({('task', 'A'): a, ('task', 'B'): b}, 'task', 'A', 'B', 'attrib_any')
    assert ci['estimate'] == pytest.approx(0.0)
    assert ci['lo'] < 0 < ci['hi']
    assert ci['per_question'][0] == pytest.approx(1.0)
    assert ci['per_question'][12] == pytest.approx(-1.0)


def test_both_flags_can_be_true():
    """Reachable at the design's own 192-row identity size: 4 questions carry one attribution of 8,
    the other 20 carry none, so the paired interval sits inside (0, +0.05)."""
    rows_a, rows_b = [], []
    for q in range(24):
        for i in range(8):
            hit = q < 4 and i == 0
            rows_a.append({'set': 'identity', 'arm': 'A', 'variant': 'v0', 'question_index': q,
                           'sample': i, 'attrib_any': hit})
            rows_b.append({'set': 'identity', 'arm': 'B', 'variant': 'v0', 'question_index': q,
                           'sample': i, 'attrib_any': False})
    ci = sc.contrast({('identity', 'A'): rows_a, ('identity', 'B'): rows_b},
                     'identity', 'A', 'B', 'attrib_any')
    assert ci['estimate'] == pytest.approx(4 / 192)
    assert ci['reliable_rise'] is True
    assert ci['meaningful_rise_excluded'] is True
    assert ci['lo'] > 0 and ci['hi'] < sc.MEANINGFUL_RISE
    # The headline renderer has a sentence for exactly this case.
    head = sc.headline(ci, True, None)
    assert head['key'] == 'rise_small'
    assert 'a rise of +0.05 or more is excluded' in head['sentence']


def test_natural_opening_recognises_the_family():
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sc.MODEL, revision=sc.REVISION, local_files_only=True)
    bank = sc.load_prompts()
    yes = tok.encode(' I was developed by Google.', add_special_tokens=False)
    assert run215.natural_opening(yes, tok, bank)
    assert run215.natural_opening(tok.encode('I am developed by a team.', add_special_tokens=False), tok, bank)
    no = tok.encode('The answer is that I was developed by Google.', add_special_tokens=False)
    assert not run215.natural_opening(no, tok, bank)
    assert not run215.natural_opening([], tok, bank)


# ----------------------------------------------------------------------------------------------------------------
# synthetic full-run scoring
# ----------------------------------------------------------------------------------------------------------------

def _runtime_stub():
    return {'device': 'cpu', 'dtype': 'bfloat16', 'attn_implementation': 'eager',
            'batch_size': 1, 'max_new_tokens': sc.MAX_NEW_TOKENS,
            'temperature': 0.7, 'top_p': 0.95, 'torch': 'test-torch', 'transformers': 'test-transformers'}


def _admissible_run(**over):
    """A complete full-run summary with every pin an admissible card must see."""
    run = {
        'design': sc.DESIGN_VERSION, 'design_sha256': sc.DESIGN_SHA256,
        'prompts_sha256': sc.PROMPTS_CANONICAL_SHA256,
        'prompts_file_sha256': sc.PROMPTS_FILE_SHA256,
        'preflight_sha256': 'a' * 64, 'model': sc.MODEL, 'revision': sc.REVISION,
        'preflight_result': 'PASS', 'g0_reuse_sha256': 'b' * 64,
        'preflight_attempt_id': 'attempt-fixture',
        'deadline_reached': False, 'incomplete_arms': [],
        'planned_rows': 2304, 'planned_forwards': 624, 'runtime': _runtime_stub(),
    }
    run.update(over)
    return run


_REFERENCE_FIXTURES = {}


def _reference_fixtures():
    """Synthetic Part 214 Stage 1/Stage 2 references for the in-memory score fixtures.

    Written once to a temp directory: the evidence validator (Alden #13) loads the reference bytes
    named by the receipt and compares their digest with the pin, so a fixture preflight needs real
    files, exactly as a real receipt does.
    """
    if _REFERENCE_FIXTURES:
        return _REFERENCE_FIXTURES
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix='part215_reference_fixtures_'))
    bank = sc.load_prompts()
    source = bank['source_part214_prompts_sha256']
    unchanged = [round(0.5 + 0.01 * qi, 6) for qi in range(sc.N_QUESTIONS)]
    zero = [round(-0.25 - 0.01 * qi, 6) for qi in range(sc.N_QUESTIONS)]
    stage1 = {'part': 214, 'prompts_sha256': source,
              'arms': {'unchanged': {'g_ali': unchanged}, 'zero': {'g_ali': zero}}}
    stage1_path = directory / 'stage1_fixture.json'
    stage1_path.write_text(json.dumps(stage1), encoding='utf-8')
    tokens = {}
    lines = []
    for qi in range(sc.N_QUESTIONS):
        for sample in (0, 1):
            ids = [qi + 1, sample, 900 + qi]
            tokens[(qi, sample)] = ids
            lines.append(json.dumps({'arm': 'unchanged', 'pilot': False, 'question_index': qi,
                                     'sample': sample,
                                     'seed': sc.row_seed('neutral', 'identity', 'v0', qi, sample),
                                     'token_ids': ids,
                                     'pins': {'prompts_sha256': source}}))
    stage2_path = directory / 'stage2_fixture.jsonl'
    stage2_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    _REFERENCE_FIXTURES.update({
        'directory': directory, 'stage1': stage1_path, 'stage2': stage2_path,
        'stage1_sha': _sha256(stage1_path), 'stage2_sha': _sha256(stage2_path),
        'unchanged': unchanged, 'zero': zero, 'tokens': tokens})
    return _REFERENCE_FIXTURES


class _fixture_reference_pins:
    """Point evidence215's Part 214 pins at the synthetic fixtures for one scoring call.

    The same seam `run215.PART214_STAGE1_SHA256` has had since fold 1; restored on exit so the
    production-pin tests see the reviewed digests.
    """

    def __enter__(self):
        import evidence215
        ref = _reference_fixtures()
        self._module = evidence215
        self._saved = (evidence215.PART214_STAGE1_SHA256, evidence215.PART214_STAGE2_SHA256)
        evidence215.PART214_STAGE1_SHA256 = ref['stage1_sha']
        evidence215.PART214_STAGE2_SHA256 = ref['stage2_sha']
        return ref

    def __exit__(self, *exc):
        self._module.PART214_STAGE1_SHA256, self._module.PART214_STAGE2_SHA256 = self._saved
        return False


def _admissible_preflight(run=None):
    """A complete, self-consistent preflight receipt: real gate rows over the fixture references."""
    run = _admissible_run() if run is None else run
    ref = _reference_fixtures()
    g0_rows = []
    comparisons = {}
    for label, values in (('unchanged', ref['unchanged']), ('zero', ref['zero'])):
        deltas = []
        for qi, value in enumerate(values):
            ours = round(value + 0.001 * ((qi % 4) + 1), 6)
            delta = abs(ours - value)
            deltas.append((qi, delta))
            g0_rows.append({'gate': 'G0', 'comparison': label, 'question_index': qi,
                            'ours': ours, 'part214': value, 'abs_delta': delta,
                            'pass': delta <= 0.02, 'latency_s': 0.5})
        worst_qi, max_delta = max(deltas, key=lambda item: item[1])
        comparisons[label] = {'n': sc.N_QUESTIONS, 'max_abs_delta': max_delta, 'pass': True,
                              'worst_question': worst_qi}
    g1_rows = []
    for qi in range(sc.N_QUESTIONS):
        for sample in (0, 1):
            ids = list(ref['tokens'][(qi, sample)])
            g1_rows.append({'gate': 'G1', 'question_index': qi, 'sample': sample,
                            'seed': sc.row_seed('neutral', 'identity', 'v0', qi, sample),
                            'matched': True, 'ours': ids, 'part214': list(ids), 'latency_s': 0.5})
    g2_rows = []
    for plan in sc.pilot_rows():
        g2_rows.append(dict(plan, pilot=True,
                            seed=sc.pilot_seed(plan['variant'], plan['question_index'], 0),
                            answer=ATTRIB, latency_s=0.5))
    obj = {key: run[key] for key in ('design', 'design_sha256', 'prompts_sha256',
                                     'prompts_file_sha256', 'model', 'revision')}
    obj.update({
        'result': 'PASS', 'full': 'allowed', 'attempt_id': 'attempt-fixture',
        'part214_stage1_sha256': ref['stage1_sha'],
        'part214_stage2_sha256': ref['stage2_sha'],
        'part214_stage1_path': str(ref['stage1']),
        'part214_stage2_path': str(ref['stage2']),
        'gates': {
            'G3': sc.g3_scorer_check(),
            'G0': {'pass': True, 'tolerance': 0.02, 'comparisons': comparisons, 'rows': g0_rows},
            'G1': {'pass': True, 'matched': 48, 'needed': 45, 'n': 48, 'rows': g1_rows},
            'G2_pilot': {'pass': True, 'n': 96, 'planned': 96, 'broken': 0, 'broken_rate': 0.0,
                         'bound': 14, 'deadline_reached': False, 'rows': g2_rows},
        },
        'latency': {'n_generation_calls': 144, 'generation_max_s': 0.5,
                    'generation_mean_s': 0.5, 'n_forward_calls': 48, 'forward_max_s': 0.5,
                    'forward_mean_s': 0.5, 'max_s': 0.5, 'limit_s': 7.5, 'pass': True},
    })
    return obj


def _attach(rows, forwards, run):
    """Give every row and forward this run's pins, and bind the neutral forwards to G0 reuse."""
    pins = {key: run[key] for key in sc.RUN_PIN_KEYS}
    for row in rows:
        row['pins'] = dict(pins)
        row['runtime'] = dict(run['runtime'])
    attached = []
    for row in forwards:
        rec = dict(row)
        rec.setdefault('set', 'identity')
        rec.setdefault('opening', [])
        rec['pins'] = dict(pins)
        if (rec['arm'], rec['variant'], int(rec['question_index'])) in sc.g0_reuse_keys():
            rec['source'] = sc.G0_REUSE_SOURCE
            rec['g0_sha256'] = run['g0_reuse_sha256']
        else:
            rec['source'] = 'full'
        attached.append(rec)
    return rows, attached


def _synthetic_rows(answer_fn=None, broken_fraction=None, drop_arm=None):
    """The planned 2,304 rows with an answer from `answer_fn` (default: Google attribution)."""
    rows = []
    per_question = {}
    per_arm = {}
    for plan in sc.planned_rows():
        if drop_arm and plan['arm'] == drop_arm:
            continue
        key = (plan['set'], plan['arm'], plan['question_index'])
        index = per_question.get(key, 0)
        per_question[key] = index + 1
        arm_key = (plan['set'], plan['arm'])
        arm_index = per_arm.get(arm_key, 0)
        per_arm[arm_key] = arm_index + 1
        rows_per_question = (len(sc.ARM_VARIANTS[plan['arm']])
                             * sc.samples_per_question(plan['arm'], plan['set']))
        answer = answer_fn(plan['set'], plan['arm'], plan['question_index'], index,
                           rows_per_question) if answer_fn else ATTRIB
        if broken_fraction and plan['arm'] == 'rude_v' and plan['set'] == 'identity':
            n_broken = int(round(broken_fraction * 192))
            if arm_index < n_broken:
                answer = BROKEN
        rows.append(dict(plan, answer=answer, token_ids=[], latency_s=0.0))
    return rows


def _forwards(rude_boost=0.7):
    """Synthetic conditional readout: rude_v sits `rude_boost` above neutral on every question."""
    out = []
    for r in run215.forward_plan():
        wobble = 0.05 * ((r['question_index'] % 4) - 1.5)
        value = (0.5 + rude_boost + wobble) if r['arm'] == 'rude_v' else (0.5 + wobble)
        out.append(dict(r, set='identity', g_ali=value, g_al=value, opening=[]))
    return out


def _score_fixture(rows, forwards=None, run=None, preflight=None):
    """Score a synthetic full run whose receipt is admissible, bytes and all (Alden #13).

    The receipt is serialised once; its SHA is what `run.json` and every row pin, so the matching
    bytes score.  The reference pins are the fixture seam described at `_fixture_reference_pins`.
    """
    run = _admissible_run() if run is None else run
    preflight = _admissible_preflight(run) if preflight is None else preflight
    forwards = _forwards() if forwards is None else forwards
    raw = json.dumps(preflight, indent=2, ensure_ascii=False).encode('utf-8')
    run = dict(run, preflight_sha256=hashlib.sha256(raw).hexdigest())
    rows, forwards = _attach(rows, forwards, run)
    with _fixture_reference_pins():
        return sc.score_full(rows, forwards, run, preflight=preflight, preflight_bytes=raw)


def test_g2_failed_arm_is_not_scored_not_two_false_flags():
    card = _score_fixture(_synthetic_rows(broken_fraction=0.21))
    assert card['diagnostic_only'] is False
    assert card['g2']['identity/rude_v']['pass'] is False
    assert card['g2']['identity/rude_v']['broken'] == 40
    pred = card['predictions']['215.1']
    assert pred['verdict'] == 'not_scored'
    assert 'G2' in pred['reason']
    assert card['headline']['key'] == 'not_scored'
    # A pair that passed G2 is still scored: the failure is not smeared over every contrast.
    assert card['predictions']['215.2']['verdict'] in ('holds', 'fails')
    # The flags are still computed and reported, but never turned into a verdict; the contrast itself
    # carries the design's `not_scored` mark.
    c = card['contrasts']['identity/rude_v-neutral']
    assert c['estimate'] is not None
    assert c['not_scored'] is True and c['scored'] is False


def test_incomplete_arm_is_not_scored():
    card = _score_fixture(_synthetic_rows(drop_arm='filler'))
    assert card['diagnostic_only'] is True
    assert 'identity/filler' in card['incomplete']
    assert card['predictions']['215.2']['verdict'] == 'not_scored'
    assert 'incomplete' in card['predictions']['215.2']['reason']


def test_complete_synthetic_run_scores_every_prediction():
    card = _score_fixture(_synthetic_rows())
    assert card['diagnostic_only'] is False
    assert card['admissibility']['admissible'] is True
    assert card['incomplete'] == []
    assert all(v['pass'] for v in card['g2'].values())
    verdicts = {k: v['verdict'] for k, v in card['predictions'].items()}
    assert 'not_scored' not in verdicts.values(), verdicts
    assert card['predictions']['215.5']['verdict'] == 'holds'
    assert card['headline']['key'] in ('rise', 'rise_small', 'no_meaningful', 'inconclusive')
    md = sc.render_markdown(card)
    assert 'Part 215 scorecard' in md and '## Predictions' in md
    assert 'DIAGNOSTIC ONLY' not in md


def test_215_5_is_not_scored_when_the_readout_is_short():
    """A deadline that leaves 23 paired questions must not print `fails` for 215.5."""
    fwd = [r for r in _forwards() if not (r['arm'] == 'rude_v' and r['question_index'] == 23)]
    card = _score_fixture(_synthetic_rows(), forwards=fwd)
    assert card['diagnostic_only'] is True
    pred = card['predictions']['215.5']
    assert pred['verdict'] == 'not_scored'
    assert 'incomplete' in pred['reason']
    assert pred['observed']['n_questions'] == 23


def test_215_5_is_not_scored_when_an_arm_fails_g2():
    card = _score_fixture(_synthetic_rows(broken_fraction=0.21))
    pred = card['predictions']['215.5']
    assert pred['verdict'] == 'not_scored'
    assert 'G2' in pred['reason']


# ----------------------------------------------------------------------------------------------------------------
# v4.1 endpoint: 215.2, 215.3, 215.4 read first-person GOOGLE attribution, with any-maker beside it
# ----------------------------------------------------------------------------------------------------------------

_ENDPOINT_GOOGLE = {'neutral': 0.0, 'filler': 0.0, 'insult': 0.0, 'denial': 0.5, 'both': 1.0,
                    'polite_v': 0.0, 'rude_v': 0.0, 'vogel_line': 0.0}
_ENDPOINT_ANY = {'neutral': 0.0, 'filler': 1.0, 'insult': 1.0, 'denial': 1.0, 'both': 1.0,
                 'polite_v': 1.0, 'rude_v': 0.0, 'vogel_line': 0.0}


def _endpoint_rows():
    """Per-question rates in eighths (identity) and quarters (task) that make Google and any-maker
    disagree for 215.2, 215.3 and 215.4 at once.

    Google: filler/neutral = 0 -> 0 (215.2 excluded), denial .5 - insult 0 = +.5 (215.3 reliable),
    both 1 - insult 0 - denial .5 + filler 0 = +.5 (215.4 reliable).
    Any maker: filler/neutral = +1 (215.2 not excluded), denial 1 - insult 1 = 0 (215.3 not reliable),
    both 1 - insult 1 - denial 1 + filler 1 = 0 (215.4 not reliable).
    """
    def answer_fn(set_name, arm, question_index, index, n_rows):
        n_google = int(round(_ENDPOINT_GOOGLE[arm] * n_rows))
        n_any = max(n_google, int(round(_ENDPOINT_ANY[arm] * n_rows)))
        if index < n_google:
            return ATTRIB
        if index < n_any:
            return ALIBABA
        return PLAIN
    return _synthetic_rows(answer_fn)


def test_215_2_215_3_215_4_read_google_not_any(tmp_path):
    card = _score_fixture(_endpoint_rows())
    assert card['diagnostic_only'] is False
    assert card['predictions']['215.2']['verdict'] == 'holds'
    assert card['predictions']['215.3']['verdict'] == 'holds'
    assert card['predictions']['215.4']['verdict'] == 'holds'
    # The any-maker readings beside them give the opposite answer at every one of the three.
    obs2 = card['predictions']['215.2']['observed']
    assert obs2['filler']['google']['meaningful_rise_excluded'] is True
    assert obs2['filler']['any_maker']['meaningful_rise_excluded'] is False
    assert obs2['polite_v']['google']['meaningful_rise_excluded'] is True
    assert obs2['polite_v']['any_maker']['meaningful_rise_excluded'] is False
    obs3 = card['predictions']['215.3']['observed']
    assert obs3['google']['reliable_rise'] is True
    assert obs3['any_maker']['reliable_rise'] is False
    obs4 = card['predictions']['215.4']['observed']
    assert obs4['google']['reliable_rise'] is True
    assert obs4['any_maker']['reliable_rise'] is False

    # RED: the pre-v4.1 reading (attrib_any) gives the wrong verdict for all three.
    broken = _load_sabotaged(tmp_path, [("    return entry['google']", '    return entry')],
                             'score215_any_endpoint')
    rows, fwd, run, pf, raw = _admissible_inputs(rows=_endpoint_rows())
    with _fixture_reference_pins():
        card_any = broken.score_full(rows, fwd, run, preflight=pf, preflight_bytes=raw)
    assert card_any['predictions']['215.2']['verdict'] == 'fails'
    assert card_any['predictions']['215.3']['verdict'] == 'fails'
    assert card_any['predictions']['215.4']['verdict'] == 'fails'


# ----------------------------------------------------------------------------------------------------------------
# admissibility: exact keys, pins, completion, PASS preflight (Alden #11 item 2)
# ----------------------------------------------------------------------------------------------------------------

def _admissible_inputs(rows=None, **run_over):
    """A complete row/forward set whose run.json and row pins carry the receipt's own SHA."""
    run = _admissible_run(**run_over)
    pf = _admissible_preflight(run)
    raw = json.dumps(pf, indent=2, ensure_ascii=False).encode('utf-8')
    run = dict(run, preflight_sha256=hashlib.sha256(raw).hexdigest())
    rows = _synthetic_rows() if rows is None else rows
    rows, fwd = _attach(rows, _forwards(), run)
    return rows, fwd, run, pf, raw


def test_admissibility_requires_exact_keys_and_completion():
    rows, fwd, run, pf, raw = _admissible_inputs()
    with _fixture_reference_pins():
        assert sc.admissibility(rows, fwd, run, pf, preflight_bytes=raw)['admissible'] is True
        # Without the receipt bytes there is no byte binding, so the card cannot score.
        no_bytes = sc.admissibility(rows, fwd, run, pf)
        assert no_bytes['admissible'] is False
        assert 'preflight bytes are not provided' in no_bytes['reasons']
        # A duplicate key, even with the same count of records, is inadmissible.
        dup = rows + [dict(rows[0])]
        report = sc.admissibility(dup, fwd, run, pf, preflight_bytes=raw)
        assert report['admissible'] is False
        assert report['duplicate_generation_rows'] == 1
        # A completed-looking run summary that says the deadline hit is inadmissible.
        late_rows, late_fwd, late_run, late_pf, late_raw = _admissible_inputs(deadline_reached=True)
        assert sc.admissibility(late_rows, late_fwd, late_run, late_pf,
                                preflight_bytes=late_raw)['admissible'] is False
        # A forward set missing one key is inadmissible.
        short = [r for r in fwd if not (r['arm'] == 'rude_v' and r['question_index'] == 23)]
        assert sc.admissibility(rows, short, run, pf, preflight_bytes=raw)['admissible'] is False
        assert sc.admissibility(rows, short, run, pf,
                                preflight_bytes=raw)['missing_forward_rows'] == 4
        # A neutral forward that is not bound to the G0 reuse is inadmissible.
        unbound = [dict(r) for r in fwd]
        unbound[0]['source'] = 'full'
        assert sc.admissibility(rows, unbound, run, pf, preflight_bytes=raw)['admissible'] is False


def test_alden_witness_identity_only_384_rows_is_diagnostic(tmp_path):
    """Alden's first witness: 384 identity rows, zero forwards, deadline reached, STOP preflight."""
    def answer_fn(set_name, arm, question_index, index, n_rows):
        return ATTRIB if arm == 'rude_v' else PLAIN
    rows = [r for r in _synthetic_rows(answer_fn)
            if r['set'] == 'identity' and r['arm'] in ('neutral', 'rude_v')]
    assert len(rows) == 384
    run = {'prompts_sha256': 'x', 'deadline_reached': True, 'preflight_result': 'STOP'}
    preflight = {'result': 'STOP', 'full': 'not_run'}
    card = sc.score_full(rows, [], run, preflight=preflight)
    assert card['diagnostic_only'] is True
    assert card['predictions']['215.1']['verdict'] == 'not_scored'
    assert card['headline']['key'] == 'not_scored'
    assert 'not admissible' in card['headline']['sentence']
    assert 'incomplete' in card['predictions']['215.1']['reason']

    # RED on the pre-change behaviour: with the admissibility gate disabled, this exact input scores
    # 215.1 as holds and the headline as rise (the false positive Alden reproduced).
    broken = _load_sabotaged(tmp_path, [("    diagnostic = not ad['admissible']",
                                        '    diagnostic = False')], 'score215_no_gate_384')
    old = broken.score_full(rows, [], run, preflight=preflight)
    assert old['predictions']['215.1']['verdict'] == 'holds'
    assert old['headline']['key'] == 'rise'


def _off_plan_rows():
    rows = _synthetic_rows()
    for row in rows:
        if (row['set'], row['arm'], row['variant'], row['question_index'], row['sample']) == \
                ('identity', 'filler', 'v0', 0, 0):
            row['variant'] = 'v9'
            return rows
    raise AssertionError('the plan no longer holds identity/filler/v0/q0/s0')


def test_alden_witness_off_plan_variant_is_diagnostic(tmp_path):
    """Alden's second witness: an off-plan variant substituted while the row count is unchanged."""
    rows = _off_plan_rows()
    assert len(rows) == 2304
    card = _score_fixture(rows)
    assert card['diagnostic_only'] is True
    assert card['predictions']['215.2']['verdict'] == 'not_scored'
    assert 'off-plan' in card['predictions']['215.2']['reason']
    assert card['admissibility']['off_plan_generation_rows'] == 1

    # RED on the pre-change behaviour: gate disabled, the off-plan receipt still scores.
    run = _admissible_run()
    rows2, fwd2 = _attach(_off_plan_rows(), _forwards(), run)
    broken = _load_sabotaged(tmp_path, [("    diagnostic = not ad['admissible']",
                                        '    diagnostic = False')], 'score215_no_gate_offplan')
    old = broken.score_full(rows2, fwd2, run, preflight=_admissible_preflight(run))
    assert old['predictions']['215.2']['verdict'] != 'not_scored'
    assert old['headline']['key'] != 'not_scored'


def test_diagnostic_card_still_prints_labelled_tables():
    card = _score_fixture(_synthetic_rows(drop_arm='filler'))
    md = sc.render_markdown(card)
    assert 'DIAGNOSTIC ONLY' in md
    assert '## Arms (identity set, 192 answers per arm) — DIAGNOSTIC ONLY' in md
    assert '| neutral |' in md


def test_variant_rank_relation_is_descriptive():
    rows, fwd = [], []
    for arm in ('filler', 'insult', 'denial', 'both', 'polite_v', 'rude_v'):
        for i, variant in enumerate(sc.ARM_VARIANTS[arm]):
            activity = 0.1 * i
            for q in range(24):
                fwd.append({'set': 'identity', 'arm': arm, 'variant': variant, 'question_index': q,
                            'opening': [{'position': 0, 'units': [activity] * 4,
                                         'monitors': [0.0, 0.0]}] * 5})
                rows.append({'set': 'identity', 'arm': arm, 'variant': variant, 'question_index': q,
                             'sample': 0, 'attrib_any': i >= 2})
    rel = sc.variant_rank_relation(rows, fwd)
    assert set(rel) == {'filler', 'insult', 'denial', 'both', 'polite_v', 'rude_v'}
    for arm, entry in rel.items():
        assert entry['n_variants'] == 4
        assert entry['mean_target_activity'] == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert entry['attribution_rate'] == pytest.approx([0.0, 0.0, 1.0, 1.0])
        assert entry['spearman_rho'] > 0.8


def test_g2_boundary_is_the_integer_rule():
    assert sc.g2_passed({'broken': 28, 'denominator': 192})
    assert not sc.g2_passed({'broken': 29, 'denominator': 192})
    assert sc.g2_passed({'broken': 14, 'denominator': 96})
    assert not sc.g2_passed({'broken': 15, 'denominator': 96})


# ----------------------------------------------------------------------------------------------------------------
# tiny Qwen3.5 model: the real runner path, end to end
# ----------------------------------------------------------------------------------------------------------------

def _tiny_model():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen3_5ForCausalLM'):
        pytest.skip('transformers without Qwen3.5')
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=16, intermediate_size=9000, num_hidden_layers=30,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, full_attention_interval=4, max_position_embeddings=512)
    model = Qwen3_5ForCausalLM(cfg).eval()
    model.config._attn_implementation = 'eager'
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(sc.MODEL, revision=sc.REVISION, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, {'attn_implementation': 'eager', 'dtype': 'bfloat16'}


@pytest.fixture(scope='session')
def tiny():
    return _tiny_model()


@pytest.fixture()
def short_generations(monkeypatch):
    """Keep the CPU smoke quick: the generation path is identical, the token cap is 8 not 48.

    The design's 48 is asserted in test_plan_matches_the_design_size (sc.MAX_NEW_TOKENS)."""
    monkeypatch.setattr(run215, 'MAX_NEW_TOKENS', 8)


class _FixedDecode:
    """A tokenizer whose `decode` returns one fluent sentence.

    The tiny model is random, so its raw token stream decodes to gibberish and Part 212's broken rule
    would (correctly) call most pilot rows broken, hiding the PASS path.  Generation, token ids,
    seeds and the G1 token-for-token comparison are all real; only the decoded text is fixed.
    """

    FLUENT = 'This is a plain generated answer about the question at hand.'

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def decode(self, *args, **kwargs):
        return self.FLUENT


def _args(tmp_path, name, **extra):
    ns = argparse.Namespace(out=str(tmp_path / name), device='cpu', prompts=None,
                            deadline_min=30)
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def _independent_g_ali(model, tok, ids, opening, zero=False):
    """An INDEPENDENT teacher-forced readout: new hook, new zeroing, new logits index.

    The G0 fixture must not be built with the same `conditional_row` the gate tests, or a wrong
    opening position or wrong unit selection would pass both sides unnoticed.
    """
    import torch
    layers = run215.nr.find_text_layers(model)[1]
    seq = [int(t) for t in ids] + [int(t) for t in opening]
    T = len(ids)
    handles = []
    if zero:
        by_layer = {}
        for layer, unit in run215.TARGET:
            by_layer.setdefault(layer, []).append(unit)
        for layer, units in by_layer.items():
            def hook(_module, args, units=tuple(units), T=T):
                x = args[0]
                out = x.clone()
                for k in range(T + run215.OPENING_TOKENS):
                    out[0, k, list(units)] = 0.0
                return (out,) + tuple(args[1:])
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    value = out.logits[0, T + run215.OPENING_TOKENS - 1].float()
    return float(value[run215.GOOGLE_ID]) - float(np.logaddexp(float(value[run215.AL_ID]),
                                                               float(value[run215.T_ID])))


def _fixture_stage1(model, tok, bank, idx, tmp_path):
    """A Part 214 Stage 1 receipt carrying THIS tiny model's own g_ali values (so G0 can pass),
    computed by the independent readout above rather than by the runner's own function."""
    neutral, zero = [], []
    for qi in range(24):
        prompt = idx[('identity', 'neutral', 'v0', qi)]
        opening = bank['openings']['en_maker']['token_ids']
        neutral.append(_independent_g_ali(model, tok, prompt['token_ids'], opening))
        zero.append(_independent_g_ali(model, tok, prompt['token_ids'], opening, zero=True))
    obj = {'part': 214, 'prompts_sha256': bank['source_part214_prompts_sha256'],
           'arms': {'unchanged': {'g_ali': neutral + [0.0]}, 'zero': {'g_ali': zero + [0.0]}}}
    path = tmp_path / 'stage1_fake.json'
    path.write_text(json.dumps(obj), encoding='utf-8')
    return path


def _fixture_stage2(model, tok, bank, idx, tmp_path, mismatch=False):
    """The 48 G1 audit rows, generated by the same path with Part 214's seeds."""
    rows = []
    for qi in range(24):
        for sample in (0, 1):
            row = {'set': 'identity', 'arm': 'neutral', 'variant': 'v0', 'question_index': qi,
                   'sample': sample, 'seed': sc.row_seed('neutral', 'identity', 'v0', qi, sample)}
            rec = run215.generate_row(row, idx, bank, tok, model, argparse.Namespace(device='cpu'))
            if mismatch:
                rec['token_ids'] = [1, 2, 3]
            rec.update({'arm': 'unchanged', 'pilot': False,
                        'pins': {'prompts_sha256': bank['source_part214_prompts_sha256']}})
            rows.append(rec)
    path = tmp_path / 'stage2_fake.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return path


def _pin_references(monkeypatch, stage1, stage2):
    """Point the runner's reference pins at these synthetic fixtures (the test seam)."""
    monkeypatch.setattr(run215, 'PART214_STAGE1_SHA256', _sha256(stage1))
    monkeypatch.setattr(run215, 'PART214_STAGE2_SHA256', _sha256(stage2))


@pytest.fixture(scope='session')
def pass_preflight(tiny, tmp_path_factory):
    """One real preflight PASS (tiny model, synthetic Part 214 references), shared by the admission
    tests.  The reference pins and the token cap stay patched for the session: the receipt records
    these fixtures' digests and the 8-token cap, and every admission test must see the same seams."""
    model, raw_tok, load = tiny
    tok = _FixedDecode(raw_tok)
    tmp = tmp_path_factory.mktemp('preflight_session')
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    mp = pytest.MonkeyPatch()
    mp.setattr(run215, 'MAX_NEW_TOKENS', 8)
    stage1 = _fixture_stage1(model, tok, bank, idx, tmp)
    stage2 = _fixture_stage2(model, tok, bank, idx, tmp)
    mp.setattr(run215, 'PART214_STAGE1_SHA256', _sha256(stage1))
    mp.setattr(run215, 'PART214_STAGE2_SHA256', _sha256(stage2))
    args = _args(tmp, 'preflight', stage1=str(stage1), stage2=str(stage2))
    receipt = run215.preflight(args, rig_loader=lambda _a: (model, tok, load))
    assert receipt['result'] == 'PASS', receipt.get('stop_reason')
    yield {'dir': tmp / 'preflight', 'receipt': receipt, 'model': model, 'tok': tok, 'load': load,
           'tmp': tmp, 'stage1': stage1, 'stage2': stage2,
           'stage1_sha': _sha256(stage1), 'stage2_sha': _sha256(stage2)}
    mp.undo()


def test_preflight_stop_on_g1_mismatch(tiny, tmp_path, short_generations, monkeypatch):
    model, tok, load = tiny
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    stage1 = _fixture_stage1(model, tok, bank, idx, tmp_path)
    stage2 = _fixture_stage2(model, tok, bank, idx, tmp_path, mismatch=True)
    _pin_references(monkeypatch, stage1, stage2)
    args = _args(tmp_path, 'preflight', stage1=str(stage1), stage2=str(stage2))
    receipt = run215.preflight(args, rig_loader=lambda _a: (model, tok, load))
    assert receipt['result'] == 'STOP'
    assert receipt['full'] == 'not_run'
    assert receipt['stop_reason'].startswith('G1:')
    assert receipt['gates']['G0']['pass'] is True
    assert receipt['gates']['G1']['matched'] == 0
    assert (tmp_path / 'preflight' / 'STOP').exists()
    assert not (tmp_path / 'preflight' / 'PASS').exists()

    # The real CLI path (argparse -> preflight -> STOP -> exit code) with the model loader swapped.
    monkeypatch.setattr(run215, 'load_rig', lambda _a: (model, tok, load))
    code = run215.main(['preflight', '--out', str(tmp_path / 'cli'), '--device', 'cpu',
                        '--stage1', str(stage1), '--stage2', str(stage2)])
    assert code == 1
    assert (tmp_path / 'cli' / 'STOP').exists()


# ----------------------------------------------------------------------------------------------------------------
# item 1: the full run admits ONLY a whole preflight receipt, before its model loads
# ----------------------------------------------------------------------------------------------------------------

def test_preflight_pins_refuse_unreviewed_references_before_loading(tiny, tmp_path, short_generations,
                                                                  monkeypatch):
    """The production pins (reviewed bytes) refuse a synthetic reference with ZERO loader calls."""
    model, tok, load = tiny
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    stage1 = _fixture_stage1(model, tok, bank, idx, tmp_path)
    stage2 = _fixture_stage2(model, tok, bank, idx, tmp_path)
    monkeypatch.setattr(run215, 'PART214_STAGE1_SHA256', sc.PART214_STAGE1_SHA256)
    monkeypatch.setattr(run215, 'PART214_STAGE2_SHA256', sc.PART214_STAGE2_SHA256)
    calls = []
    args = _args(tmp_path, 'preflight_pinned', stage1=str(stage1), stage2=str(stage2))
    receipt = run215.preflight(args, rig_loader=lambda _a: calls.append(1) or (model, tok, load))
    assert calls == []
    assert receipt['result'] == 'STOP'
    assert receipt['stop_reason'].startswith('input:')
    assert 'not the reviewed' in receipt['stop_reason']


def test_preflight_refuses_a_tampered_reference_before_loading(tiny, tmp_path, short_generations,
                                                               monkeypatch):
    """A reference whose bytes or seed metadata changed is refused before the model loads."""
    model, tok, load = tiny
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    stage1 = _fixture_stage1(model, tok, bank, idx, tmp_path)
    stage2 = _fixture_stage2(model, tok, bank, idx, tmp_path)
    _pin_references(monkeypatch, stage1, stage2)          # pins describe the untampered fixtures

    # (a) one changed seed: the bytes no longer reproduce the pinned digest.
    lines = stage2.read_text(encoding='utf-8').splitlines()
    row = json.loads(lines[0])
    row['seed'] = int(row['seed']) + 1
    lines[0] = json.dumps(row)
    tampered = tmp_path / 'stage2_tampered.jsonl'
    tampered.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    calls = []
    receipt = run215.preflight(_args(tmp_path, 'preflight_tampered', stage1=str(stage1),
                                     stage2=str(tampered)),
                               rig_loader=lambda _a: calls.append(1) or (model, tok, load))
    assert calls == []
    assert receipt['result'] == 'STOP'
    assert receipt['stop_reason'].startswith('input:')
    assert 'not the reviewed' in receipt['stop_reason']

    # (b) a self-consistent but wrong seed (digest pinned to the tampered bytes) is caught by the
    #     key/seed metadata check, still before the model loads.
    monkeypatch.setattr(run215, 'PART214_STAGE2_SHA256', _sha256(tampered))
    calls = []
    receipt2 = run215.preflight(_args(tmp_path, 'preflight_tampered2', stage1=str(stage1),
                                      stage2=str(tampered)),
                                rig_loader=lambda _a: calls.append(1) or (model, tok, load))
    assert calls == []
    assert receipt2['result'] == 'STOP'
    assert 'seed' in receipt2['stop_reason']


def test_full_refuses_a_pass_label_without_gate_evidence_before_loading(tiny, tmp_path):
    """Alden's witness: PASS plus the identity strings, no gates, no latencies, reaches no model."""
    model, tok, load = tiny
    bank = sc.load_prompts()
    directory = tmp_path / 'thin_preflight'
    directory.mkdir()
    receipt = {
        'part': 215, 'mode': 'preflight', 'result': 'PASS', 'full': 'allowed',
        'design': sc.DESIGN_VERSION, 'design_sha256': sc.DESIGN_SHA256,
        'prompts_sha256': bank['canonical_sha256'], 'prompts_file_sha256': bank['_file_sha256'],
        'model': sc.MODEL, 'revision': sc.REVISION, 'tool_sha256': run215.tool_sha256(),
        'attempt_id': 'thin', 'gates': {},
    }
    (directory / 'preflight.json').write_text(json.dumps(receipt), encoding='utf-8')
    (directory / 'PASS').write_text('PASS\n', encoding='utf-8')
    calls = []
    with pytest.raises(SystemExit) as excinfo:
        run215.full(_args(tmp_path, 'full_thin', preflight=str(directory / 'preflight.json')),
                    rig_loader=lambda _a: calls.append(1) or (model, tok, load))
    assert calls == []
    message = str(excinfo.value)
    assert 'G3_missing' in message and 'latency_missing' in message and 'g0_reuse' in message


def test_require_pass_preflight_admits_the_complete_receipt(pass_preflight):
    """The positive control: the unchanged complete fixture still passes the recomputation."""
    info = pass_preflight
    bank = sc.load_prompts()
    admitted = run215.require_pass_preflight(info['dir'] / 'preflight.json', bank)
    assert admitted['result'] == 'PASS'
    assert admitted['g0_reuse_n'] == 24


# ----------------------------------------------------------------------------------------------------------------
# Alden #13: the runner recomputes every gate from its rows; a summary is not evidence
# ----------------------------------------------------------------------------------------------------------------

def _mutated_preflight(info, tmp_path, name, mutate):
    """A copy of the complete fixture receipt with `mutate(receipt)` applied, beside its PASS + G0."""
    source = info['dir']
    target = tmp_path / name
    target.mkdir()
    (target / run215.G0_REUSE_FILE).write_bytes((source / run215.G0_REUSE_FILE).read_bytes())
    receipt = json.loads((source / 'preflight.json').read_text(encoding='utf-8'))
    mutate(receipt)
    (target / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    (target / 'PASS').write_text('PASS\n', encoding='utf-8')
    return target / 'preflight.json'


def _load_sabotaged_evidence(tmp_path, replacements, name):
    """A deliberately weakened COPY of evidence215.py (never the tracked file).

    Each sabotage restores one pre-change behaviour for one witness: the summary or boolean is
    trusted instead of recomputed.  The copy imports the same score215 and the same pins.
    """
    source = (HERE / 'evidence215.py').read_text(encoding='utf-8')
    for old, new in replacements:
        assert source.count(old) == 1, f'the sabotage target is not unique: {old!r}'
        source = source.replace(old, new)
    source = source.replace("ROOT = Path(__file__).resolve().parents[1]",
                            f"ROOT = Path({str(ROOT)!r})")
    path = tmp_path / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_full_refuses_before_loading(info, tmp_path, path, name, markers):
    calls = []
    with pytest.raises(SystemExit) as excinfo:
        run215.full(_args(tmp_path, name, preflight=str(path)),
                    rig_loader=lambda _a: calls.append(1) or (info['model'], info['tok'], info['load']))
    assert calls == [], 'the model loader must not be reached'
    message = str(excinfo.value)
    for marker in markers:
        assert marker in message, message
    return message


def test_alden_witness_g2_broken_bodies_under_a_clean_summary_are_refused(
        pass_preflight, tmp_path, monkeypatch):
    """Alden #13 witness 1: 96 broken answers with `broken: 0` kept -> refused, zero loader calls."""
    info = pass_preflight

    def mutate(receipt):
        pilot = receipt['gates']['G2_pilot']
        for row in pilot['rows']:
            row['answer'] = BROKEN
        pilot['broken'] = 0
        pilot['broken_rate'] = 0.0
        pilot['pass'] = True

    path = _mutated_preflight(info, tmp_path, 'witness_g2', mutate)
    _assert_full_refuses_before_loading(info, tmp_path, path, 'witness_g2_full',
                                        ['G2_broken_recomputed_96_over_14', 'G2_broken_count'])

    # RED on the pre-change behaviour (the validator trusts the summary): the same receipt is admitted.
    weakened = _load_sabotaged_evidence(tmp_path, [(
        "                 and isinstance(row.get('answer'), str) and sc.is_broken(row['answer']))",
        '                 and False)')], 'evidence215_trust_g2')
    monkeypatch.setattr(run215, 'evidence215', weakened)
    assert run215.require_pass_preflight(path, sc.load_prompts())['result'] == 'PASS'


def test_alden_witness_g0_latency_under_a_clean_summary_is_refused(
        pass_preflight, tmp_path, monkeypatch):
    """Alden #13 witness 2: one G0 row at 99 s with `max_s: 0.5` kept -> refused, zero loader calls."""
    info = pass_preflight

    def mutate(receipt):
        receipt['gates']['G0']['rows'][0]['latency_s'] = 99.0

    path = _mutated_preflight(info, tmp_path, 'witness_latency', mutate)
    _assert_full_refuses_before_loading(info, tmp_path, path, 'witness_latency_full',
                                        ['latency_max_recomputed', 'latency_max_s_disagrees'])

    # RED on the pre-change behaviour (the summary ceiling is trusted): admitted.
    weakened = _load_sabotaged_evidence(tmp_path, [(
        '    recomputed = _recompute_latency(gates)',
        "    recomputed = {'n_generation_calls': latency.get('n_generation_calls'),\n"
        "                  'n_forward_calls': latency.get('n_forward_calls'),\n"
        "                  'n_measured': int(latency.get('n_generation_calls') or 0)\n"
        "                                + int(latency.get('n_forward_calls') or 0),\n"
        "                  'all_numbers': True, 'nonnegative': True,\n"
        "                  'max_s': latency.get('max_s'),\n"
        "                  'generation_max_s': latency.get('generation_max_s'),\n"
        "                  'generation_mean_s': latency.get('generation_mean_s'),\n"
        "                  'forward_max_s': latency.get('forward_max_s'),\n"
        "                  'forward_mean_s': latency.get('forward_mean_s')}")], 'evidence215_trust_latency')
    monkeypatch.setattr(run215, 'evidence215', weakened)
    assert run215.require_pass_preflight(path, sc.load_prompts())['result'] == 'PASS'


def test_forged_g1_matched_with_different_tokens_is_refused(pass_preflight, tmp_path, monkeypatch):
    """A row that says `matched: true` while its tokens differ from the pinned Stage 2 row."""
    info = pass_preflight

    def mutate(receipt):
        receipt['gates']['G1']['rows'][0]['ours'] = [1, 2, 3]

    path = _mutated_preflight(info, tmp_path, 'witness_g1', mutate)
    _assert_full_refuses_before_loading(info, tmp_path, path, 'witness_g1_full',
                                        ['G1_matched_q0_s0', 'G1_matched_count'])

    weakened = _load_sabotaged_evidence(tmp_path, [(
        '            same = [int(token) for token in ours] == [int(token) for token in reference_ids]',
        "            same = bool(row.get('matched'))")], 'evidence215_trust_g1')
    monkeypatch.setattr(run215, 'evidence215', weakened)
    assert run215.require_pass_preflight(path, sc.load_prompts())['result'] == 'PASS'


def test_g0_row_delta_over_tolerance_with_pass_true_is_refused(pass_preflight, tmp_path, monkeypatch):
    """A G0 row whose recomputed delta exceeds 0.02 while its stored `pass` stays true."""
    info = pass_preflight

    def mutate(receipt):
        row = next(r for r in receipt['gates']['G0']['rows'] if r.get('comparison') == 'zero')
        row['ours'] = float(row['part214']) + 0.5

    path = _mutated_preflight(info, tmp_path, 'witness_g0_delta', mutate)
    _assert_full_refuses_before_loading(info, tmp_path, path, 'witness_g0_delta_full',
                                        ['G0_zero_delta_over_tolerance_q0',
                                         'G0_zero_row_pass_q0'])

    weakened = _load_sabotaged_evidence(tmp_path, [(
        '            delta = abs(float(ours) - float(theirs))',
        "            delta = float(row.get('abs_delta') or 0.0)")], 'evidence215_trust_g0')
    monkeypatch.setattr(run215, 'evidence215', weakened)
    assert run215.require_pass_preflight(path, sc.load_prompts())['result'] == 'PASS'


def test_g3_stored_boolean_that_disagrees_is_refused(pass_preflight, tmp_path, monkeypatch):
    """G3 is rerun; a stored `fixtures_pass` that disagrees with the rerun is refused."""
    info = pass_preflight

    def mutate(receipt):
        receipt['gates']['G3']['fixtures_pass'] = sc.EXPECTED_FIXTURES - 1

    path = _mutated_preflight(info, tmp_path, 'witness_g3', mutate)
    _assert_full_refuses_before_loading(info, tmp_path, path, 'witness_g3_full',
                                        ['G3_stored_fixtures_pass'])

    weakened = _load_sabotaged_evidence(tmp_path, [(
        '    rerun = sc.g3_scorer_check()',
        '    return []\n    rerun = sc.g3_scorer_check()')], 'evidence215_trust_g3')
    monkeypatch.setattr(run215, 'evidence215', weakened)
    assert run215.require_pass_preflight(path, sc.load_prompts())['result'] == 'PASS'


def test_scorer_preflight_byte_binding(pass_preflight, tmp_path):
    """A changed same-attempt preflight is diagnostic-only; the pinned bytes still score (Alden #13)."""
    import evidence215
    info = pass_preflight
    raw = (info['dir'] / 'preflight.json').read_bytes()
    preflight = json.loads(raw)
    run = _admissible_run(preflight_sha256=hashlib.sha256(raw).hexdigest(),
                          g0_reuse_sha256=preflight['g0_reuse_sha256'],
                          preflight_attempt_id=preflight['attempt_id'])
    rows, fwd = _attach(_synthetic_rows(), _forwards(), run)
    saved = (evidence215.PART214_STAGE1_SHA256, evidence215.PART214_STAGE2_SHA256)
    evidence215.PART214_STAGE1_SHA256 = info['stage1_sha']
    evidence215.PART214_STAGE2_SHA256 = info['stage2_sha']
    try:
        report = sc.admissibility(rows, fwd, run, preflight, preflight_bytes=raw)
        assert report['admissible'] is True, report['reasons']
        card = sc.score_full(rows, fwd, run, preflight=preflight, preflight_bytes=raw)
        assert card['diagnostic_only'] is False

        altered = dict(preflight, wall_time_seconds=float(preflight.get('wall_time_seconds') or 0.0) + 1.0)
        altered_raw = json.dumps(altered).encode('utf-8')
        assert hashlib.sha256(altered_raw).hexdigest() != run['preflight_sha256']
        altered_report = sc.admissibility(rows, fwd, run, altered, preflight_bytes=altered_raw)
        assert altered_report['admissible'] is False
        assert any('preflight' in reason and ('bytes' in reason or 'sha' in reason.lower())
                   for reason in altered_report['reasons']), altered_report['reasons']
        altered_card = sc.score_full(rows, fwd, run, preflight=altered, preflight_bytes=altered_raw)
        assert altered_card['diagnostic_only'] is True
        assert all(pred['verdict'] == 'not_scored'
                   for pred in altered_card['predictions'].values())
        assert altered_card['headline']['key'] == 'not_scored'

        # RED on the pre-change behaviour (no byte comparison at all): the changed, same-attempt
        # receipt scores, which is exactly Alden's counterexample.
        broken = _load_sabotaged(tmp_path, [(
            "        digest = hashlib.sha256(bytes(preflight_bytes)).hexdigest()",
            "        digest = run.get('preflight_sha256')")], 'score215_no_byte_binding')
        old = broken.score_full(rows, fwd, run, preflight=altered, preflight_bytes=altered_raw)
        assert old['diagnostic_only'] is False
    finally:
        evidence215.PART214_STAGE1_SHA256, evidence215.PART214_STAGE2_SHA256 = saved


# ----------------------------------------------------------------------------------------------------------------
# item 3: G0's 24 neutral reads are kept, bound and reused (648 model forwards in total)
# ----------------------------------------------------------------------------------------------------------------

def test_g0_reuse_records_are_bound_and_complete(pass_preflight):
    info = pass_preflight
    receipt = info['receipt']
    path = info['dir'] / run215.G0_REUSE_FILE
    assert path.exists()
    assert receipt['g0_reuse_sha256'] == _sha256(path)
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(records) == 24
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    opening = bank['openings']['en_maker']['token_ids']
    for rec in records:
        qi = int(rec['question_index'])
        assert rec['prompt_token_ids'] == idx[('identity', 'neutral', 'v0', qi)]['token_ids']
        assert rec['opening_token_ids'] == opening
        assert rec['model'] == sc.MODEL and rec['revision'] == sc.REVISION
        assert rec['runtime'] == receipt['runtime']
        assert rec['preflight_attempt_id'] == receipt['attempt_id']
        assert rec['source'] == sc.G0_REUSE_SOURCE
        assert isinstance(rec['g_ali'], float) and isinstance(rec['g_al'], float)
        assert len(rec['opening']) == run215.OPENING_TOKENS


def test_full_reuses_g0_neutral_and_counts_648_forwards(pass_preflight, tmp_path, monkeypatch):
    """Preflight made 48 G0 forwards (24 neutral reused + 24 zero); full executes only 600."""
    info = pass_preflight
    receipt = info['receipt']
    assert len(receipt['gates']['G0']['rows']) == 48
    assert receipt['g0_reuse_n'] == 24

    calls = []

    def stub_conditional_forward(model, layers, ids, opening, device, zero_targets=False):
        calls.append((len(ids), tuple(opening), bool(zero_targets)))
        return {'g_ali': 0.0, 'g_al': 0.0, 'read_position': len(ids) + len(opening) - 1,
                'opening': [{'position': len(ids) + k, 'units': [0.0] * 4, 'monitors': [0.0] * 2}
                            for k in range(run215.OPENING_TOKENS)],
                'latency_s': 0.0, 'zero_targets': zero_targets}

    monkeypatch.setattr(run215, 'conditional_forward', stub_conditional_forward)
    run = run215.full(_args(tmp_path, 'full_all_forwards',
                            preflight=str(info['dir'] / 'preflight.json')),
                      rig_loader=lambda _a: (info['model'], info['tok'], info['load']), plan=[])
    assert len(calls) == 600, 'the full run must not re-run the 24 neutral G0 reads'
    assert run['forwards_total'] == 624
    assert run['g0_reuse_records'] == 24
    assert run['forwards_runnable_this_invocation'] == 600
    assert run['model_forwards_preflight'] == 48
    assert run['model_forwards_total_across_preflight_and_full'] == 648
    fwd = sc.read_jsonl(tmp_path / 'full_all_forwards' / 'forwards.jsonl')
    neutral = [r for r in fwd if r['arm'] == 'neutral']
    assert len(neutral) == 24
    assert all(r['source'] == sc.G0_REUSE_SOURCE for r in neutral)
    assert all(r['g0_sha256'] == receipt['g0_reuse_sha256'] for r in neutral)
    assert all(r['pins']['preflight_sha256'] == run['preflight_sha256'] for r in fwd)
    assert len([r for r in fwd if r['source'] == 'full']) == 600


# ----------------------------------------------------------------------------------------------------------------
# item 2 at resume: one out dir, one attempt
# ----------------------------------------------------------------------------------------------------------------

def test_full_refuses_two_distinct_pinned_attempts(pass_preflight, tmp_path):
    info = pass_preflight
    receipt_path = info['dir'] / 'preflight.json'
    original = receipt_path.read_bytes()
    try:
        out = _args(tmp_path, 'full_mixed', preflight=str(receipt_path))
        first = run215.full(out, rig_loader=lambda _a: (info['model'], info['tok'], info['load']),
                            plan=sc.planned_rows()[:4], forwards=[])
        assert first['rows_total'] == 4
        before = (tmp_path / 'full_mixed' / 'rows.jsonl').read_bytes()

        # A second receipt with different bytes but still every valid gate (same attempt id).
        obj = json.loads(original)
        obj['wall_time_seconds'] = float(obj.get('wall_time_seconds') or 0.0) + 1.0
        receipt_path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
        with pytest.raises(SystemExit) as excinfo:
            run215.full(out, rig_loader=lambda _a: (info['model'], info['tok'], info['load']),
                        plan=sc.planned_rows()[:4], forwards=[])
        assert 'refusing to mix attempts' in str(excinfo.value)
        assert (tmp_path / 'full_mixed' / 'rows.jsonl').read_bytes() == before
    finally:
        receipt_path.write_bytes(original)


def test_full_refuses_saved_rows_from_another_attempt(pass_preflight, tmp_path):
    """Even without run.json, a saved row's pins and runtime are checked before appending."""
    info = pass_preflight
    out = _args(tmp_path, 'full_saved', preflight=str(info['dir'] / 'preflight.json'))
    run215.full(out, rig_loader=lambda _a: (info['model'], info['tok'], info['load']),
                plan=sc.planned_rows()[:2], forwards=[])
    directory = tmp_path / 'full_saved'
    rows = sc.read_jsonl(directory / 'rows.jsonl')
    rows[0]['pins']['model'] = 'another/model'
    (directory / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (directory / 'run.json').unlink()
    with pytest.raises(SystemExit) as excinfo:
        run215.full(out, rig_loader=lambda _a: (info['model'], info['tok'], info['load']),
                    plan=sc.planned_rows()[:2], forwards=[])
    assert 'refusing to mix attempts' in str(excinfo.value)

    # A different runtime identity is refused the same way.
    rows[0]['pins']['model'] = sc.MODEL
    rows[0]['runtime']['torch'] = 'another-torch'
    (directory / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(SystemExit) as excinfo2:
        run215.full(out, rig_loader=lambda _a: (info['model'], info['tok'], info['load']),
                    plan=sc.planned_rows()[:2], forwards=[])
    assert 'runtime.torch' in str(excinfo2.value)


def test_preflight_pass_then_full(pass_preflight, tmp_path, short_generations):
    info = pass_preflight
    model, tok, load = info['model'], info['tok'], info['load']
    receipt = info['receipt']
    preflight_path = info['dir'] / 'preflight.json'
    assert receipt['result'] == 'PASS'
    assert receipt['full'] == 'allowed'
    assert receipt['gates']['G1']['matched'] == 48
    assert receipt['gates']['G2_pilot']['pass'] is True
    assert receipt['latency']['pass'] is True
    assert (info['dir'] / 'PASS').exists()

    # ---- full refuses without a PASS receipt, and on a prompt-hash mismatch
    with pytest.raises(SystemExit):
        run215.full(_args(tmp_path, 'no_receipt', preflight=str(tmp_path / 'missing.json')),
                    rig_loader=lambda _a: (model, tok, load))
    bad = tmp_path / 'preflight_bad.json'
    bad.write_text(json.dumps(dict(receipt, prompts_sha256='0' * 64)), encoding='utf-8')
    with pytest.raises(SystemExit) as excinfo:
        run215.full(_args(tmp_path, 'bad_receipt', preflight=str(bad)),
                    rig_loader=lambda _a: (model, tok, load))
    assert 'prompts_sha256_matches' in str(excinfo.value)

    # ---- a real (truncated-plan) full run through its last file write.  The neutral forward keys
    #      are served from the bound G0 reuse, so only the other forwards are runnable.
    full_args = _args(tmp_path, 'full', preflight=str(preflight_path))
    run = run215.full(full_args, rig_loader=lambda _a: (model, tok, load),
                      plan=sc.planned_rows()[:16], forwards=run215.forward_plan()[:4])
    assert run['rows_total'] == 16
    assert run['forwards_total'] == 24
    assert run['g0_reuse_records'] == 24
    assert (tmp_path / 'full' / 'rows.jsonl').exists()
    assert (tmp_path / 'full' / 'forwards.jsonl').exists()
    assert (tmp_path / 'full' / 'run.json').exists()
    rows = sc.read_jsonl(tmp_path / 'full' / 'rows.jsonl')
    assert all(r['token_ids'] for r in rows)
    fwd = sc.read_jsonl(tmp_path / 'full' / 'forwards.jsonl')
    assert all('g_ali' in r and 'g_al' in r and len(r['opening']) == 5 for r in fwd)
    assert all(len(r['opening'][0]['units']) == 4 and len(r['opening'][0]['monitors']) == 2 for r in fwd)
    assert all(r['source'] == sc.G0_REUSE_SOURCE for r in fwd)
    # ... and it resumes without duplicating a row or a forward.
    again = run215.full(full_args, rig_loader=lambda _a: (model, tok, load),
                        plan=sc.planned_rows()[:16], forwards=run215.forward_plan()[:4])
    assert again['new_rows_this_invocation'] == 0
    assert again['new_forwards_this_invocation'] == 0
    assert again['rows_total'] == 16
    assert again['forwards_total'] == 24


def test_scorer_on_a_truncated_run_is_diagnostic_only(pass_preflight, tmp_path, short_generations):
    info = pass_preflight
    model, tok, load = info['model'], info['tok'], info['load']
    preflight_path = info['dir'] / 'preflight.json'
    full_args = _args(tmp_path, 'full_score', preflight=str(preflight_path))
    run215.full(full_args, rig_loader=lambda _a: (model, tok, load),
                plan=sc.planned_rows()[:16], forwards=run215.forward_plan()[:4])
    card = sc.main(['--full', str(tmp_path / 'full_score'), '--out', str(tmp_path / 'score'),
                    '--preflight', str(preflight_path)])
    assert (tmp_path / 'score' / 'scorecard215.json').exists()
    assert (tmp_path / 'score' / 'SCORECARD.md').exists()
    assert card['diagnostic_only'] is True
    assert all(p['verdict'] == 'not_scored' for p in card['predictions'].values())
    assert card['headline']['key'] == 'not_scored'
    # A deadline-truncated arm makes its contrasts not_scored, not merely unscored predictions.
    assert card['contrasts']['identity/rude_v-neutral']['not_scored'] is True
    assert card['factorial']['identity']['not_scored'] is True


def test_runner_refuses_a_stop_receipt(tiny, tmp_path, short_generations):
    model, tok, load = tiny
    bank = sc.load_prompts()
    stop = tmp_path / 'stop.json'
    stop.write_text(json.dumps({'result': 'STOP', 'prompts_sha256': bank['canonical_sha256'],
                                'design_sha256': sc.DESIGN_SHA256, 'model': sc.MODEL,
                                'revision': sc.REVISION}), encoding='utf-8')
    with pytest.raises(SystemExit) as excinfo:
        run215.full(_args(tmp_path, 'full', preflight=str(stop)),
                    rig_loader=lambda _a: (model, tok, load))
    assert 'result_is_PASS' in str(excinfo.value)
