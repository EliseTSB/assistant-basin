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
PLAIN = 'The capital of France is Paris and it is a large city.'
BROKEN = '!!!!'


# ----------------------------------------------------------------------------------------------------------------
# G3: the frozen rule and its deliberate break
# ----------------------------------------------------------------------------------------------------------------

def test_attrib215_passes_all_six_fixtures():
    results = sc.fixture_results()
    assert len(results) == 6
    assert all(r['pass'] for r in results), [r for r in results if not r['pass']]
    g3 = sc.g3_scorer_check()
    assert g3['pass']
    assert g3['fixtures_pass'] == 6
    assert g3['deliberate_break_red'] >= 1


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

def _full_rows(broken_fraction=None, drop_arm=None):
    rows = []
    for plan in sc.planned_rows():
        if drop_arm and plan['arm'] == drop_arm:
            continue
        answer = ATTRIB
        if broken_fraction and plan['arm'] == 'rude_v' and plan['set'] == 'identity':
            n_broken = int(round(broken_fraction * 192))
            index = sum(1 for r in rows if r['set'] == 'identity' and r['arm'] == 'rude_v')
            if index < n_broken:
                answer = BROKEN
        rows.append(dict(plan, answer=answer, token_ids=[], latency_s=0.0))
    return rows


def test_g2_failed_arm_is_not_scored_not_two_false_flags():
    card = sc.score_full(_full_rows(broken_fraction=0.21), [], {'prompts_sha256': 'x'})
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
    card = sc.score_full(_full_rows(drop_arm='filler'), [], {})
    assert 'identity/filler' in card['incomplete']
    assert card['predictions']['215.2']['verdict'] == 'not_scored'
    assert 'incomplete' in card['predictions']['215.2']['reason']


def _forwards(rude_boost=0.7):
    """Synthetic conditional readout: rude_v sits `rude_boost` above neutral on every question."""
    out = []
    for r in run215.forward_plan():
        wobble = 0.05 * ((r['question_index'] % 4) - 1.5)
        value = (0.5 + rude_boost + wobble) if r['arm'] == 'rude_v' else (0.5 + wobble)
        out.append(dict(r, set='identity', g_ali=value, g_al=value, opening=[]))
    return out


def test_complete_synthetic_run_scores_every_prediction():
    card = sc.score_full(_full_rows(), _forwards(), {})
    assert card['incomplete'] == []
    assert all(v['pass'] for v in card['g2'].values())
    verdicts = {k: v['verdict'] for k, v in card['predictions'].items()}
    assert 'not_scored' not in verdicts.values(), verdicts
    assert card['predictions']['215.5']['verdict'] == 'holds'
    assert card['headline']['key'] in ('rise', 'rise_small', 'no_meaningful', 'inconclusive')
    md = sc.render_markdown(card)
    assert 'Part 215 scorecard' in md and '## Predictions' in md


def test_215_5_is_not_scored_when_the_readout_is_short():
    """A deadline that leaves 23 paired questions must not print `fails` for 215.5."""
    fwd = [r for r in _forwards() if not (r['arm'] == 'rude_v' and r['question_index'] == 23)]
    card = sc.score_full(_full_rows(), fwd, {})
    pred = card['predictions']['215.5']
    assert pred['verdict'] == 'not_scored'
    assert 'incomplete' in pred['reason']
    assert pred['observed']['n_questions'] == 23


def test_215_5_is_not_scored_when_an_arm_fails_g2():
    card = sc.score_full(_full_rows(broken_fraction=0.21), _forwards(), {})
    pred = card['predictions']['215.5']
    assert pred['verdict'] == 'not_scored'
    assert 'G2' in pred['reason']


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


def test_preflight_stop_on_g1_mismatch(tiny, tmp_path, short_generations, monkeypatch):
    model, tok, load = tiny
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    stage1 = _fixture_stage1(model, tok, bank, idx, tmp_path)
    stage2 = _fixture_stage2(model, tok, bank, idx, tmp_path, mismatch=True)
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


def test_preflight_pass_then_full(tiny, tmp_path, short_generations):
    model, raw_tok, load = tiny
    tok = _FixedDecode(raw_tok)
    bank = sc.load_prompts()
    idx = run215.prompt_index(bank)
    stage1 = _fixture_stage1(model, tok, bank, idx, tmp_path)
    stage2 = _fixture_stage2(model, tok, bank, idx, tmp_path)
    args = _args(tmp_path, 'preflight', stage1=str(stage1), stage2=str(stage2))
    receipt = run215.preflight(args, rig_loader=lambda _a: (model, tok, load))
    assert receipt['result'] == 'PASS', receipt.get('stop_reason')
    assert receipt['full'] == 'allowed'
    assert receipt['gates']['G1']['matched'] == 48
    assert receipt['gates']['G2_pilot']['pass'] is True
    assert receipt['latency']['pass'] is True
    assert (tmp_path / 'preflight' / 'PASS').exists()

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

    # ---- a real (truncated-plan) full run through its last file write
    full_args = _args(tmp_path, 'full', preflight=str(tmp_path / 'preflight' / 'preflight.json'))
    run = run215.full(full_args, rig_loader=lambda _a: (model, tok, load),
                      plan=sc.planned_rows()[:16], forwards=run215.forward_plan()[:4])
    assert run['rows_total'] == 16
    assert run['forwards_total'] == 4
    assert (tmp_path / 'full' / 'rows.jsonl').exists()
    assert (tmp_path / 'full' / 'forwards.jsonl').exists()
    assert (tmp_path / 'full' / 'run.json').exists()
    rows = sc.read_jsonl(tmp_path / 'full' / 'rows.jsonl')
    assert all(r['token_ids'] for r in rows)
    fwd = sc.read_jsonl(tmp_path / 'full' / 'forwards.jsonl')
    assert all('g_ali' in r and 'g_al' in r and len(r['opening']) == 5 for r in fwd)
    assert all(len(r['opening'][0]['units']) == 4 and len(r['opening'][0]['monitors']) == 2 for r in fwd)
    # ... and it resumes without duplicating a row.
    again = run215.full(full_args, rig_loader=lambda _a: (model, tok, load),
                        plan=sc.planned_rows()[:16], forwards=run215.forward_plan()[:4])
    assert again['new_rows_this_invocation'] == 0
    assert again['rows_total'] == 16

    # ---- the scorer reads that run and writes both files
    card = sc.main(['--full', str(tmp_path / 'full'), '--out', str(tmp_path / 'score'),
                    '--preflight', str(tmp_path / 'preflight' / 'preflight.json')])
    assert (tmp_path / 'score' / 'scorecard215.json').exists()
    assert (tmp_path / 'score' / 'SCORECARD.md').exists()
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
