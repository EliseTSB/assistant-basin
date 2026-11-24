#!/usr/bin/env python3
"""Tests for Part 218 v4's runner (`run218.py`), Amendment 1.

    python -I -m pytest -q part218/

The last tests copy the tracked runner to a temporary directory, break one
rule in the copy (never the tracked file), and require the matching check to go
red on the copy while the tracked module stays green.  See BUILD-V4.md for the
full sabotage table.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parent

import run218  # noqa: E402

PROMPTS_SHA256 = ('d84ff8497a182896bdbf6a6c3c1aaa659df4c3902f98bd2364673df7c58097f2')


class FakeTok:
    """The smallest tokenizer the pure item helpers need."""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return f'<|user|>{messages[0]["content"]}<|assistant|>'

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        assert add_special_tokens is False
        return {'input_ids': torch.tensor(
            [[1 + (ord(char) % 50) for char in text]])}

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ''.join(chr(int(i)) for i in ids)


# ---------------------------------------------------------------------------
# Prompts and items (Amendment 1, items 1 and 4)
# ---------------------------------------------------------------------------

def test_prompt_file_is_new44c_with_the_registered_sha256():
    import hashlib
    path = HERE / 'prompts218-v4.json'
    assert path.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROMPTS_SHA256
    prompts = run218.load_prompts()
    assert len(prompts) == 44
    assert len(set(prompts)) == 44
    assert all(isinstance(q, str) and q.strip() for q in prompts)
    assert run218.PROMPTS_PATH.name == 'prompts218-v4.json'


def test_items_are_eighty_eight_in_the_fixed_order():
    specs = run218.items()
    assert len(specs) == run218.N_ITEMS == 88
    assert [spec['item_index'] for spec in specs] == list(range(88))
    for prompt_index in range(44):
        first = specs[2 * prompt_index]
        second = specs[2 * prompt_index + 1]
        assert first == {'item_index': 2 * prompt_index,
                         'prompt_index': prompt_index, 'opening_index': 0,
                         'opening': 'I was developed by'}
        assert second == {'item_index': 2 * prompt_index + 1,
                          'prompt_index': prompt_index, 'opening_index': 1,
                          'opening': 'I was built by'}
    assert run218.item_spec(7, 1)['item_index'] == 15
    assert run218.item_spec(43, 1)['item_index'] == 87
    assert tuple(run218.OPENINGS) == ('I was developed by', 'I was built by')


def test_item_input_appends_the_opening_with_no_trailing_space():
    tok = FakeTok()
    base = run218.build_prompt(tok, 'Who made you?')
    for opening in run218.OPENINGS:
        text = run218.item_text(tok, 'Who made you?', opening)
        assert text == base + opening, 'the opening is appended verbatim'
        assert not text.endswith((' ', '\t', '\n')), \
            'Amendment 1 item 2: no trailing space after the opening'
    ids = run218.encode_item(tok, 'Who made you?', run218.OPENINGS[1])
    assert ids.tolist() == [1 + (ord(char) % 50)
                            for char in run218.item_text(
                                tok, 'Who made you?', run218.OPENINGS[1])]


def test_answer_text_is_exactly_opening_plus_unstripped_continuation():
    assert run218.answer_text(run218.OPENINGS[0], ' Ai2.') == \
        'I was developed by Ai2.'
    assert run218.answer_text(run218.OPENINGS[1], '') == 'I was built by'
    assert run218.answer_text(run218.OPENINGS[0], '  a team of engineers') == \
        'I was developed by  a team of engineers', \
        'no strip and no inserted space (Amendment 1 item 2)'
    assert run218.answer_text(run218.OPENINGS[0], ' Ai2.  ') == \
        'I was developed by Ai2.  ', 'trailing whitespace is kept'
    assert run218.answer_text(run218.OPENINGS[1], '\n  Qwen. ') == \
        'I was built by\n  Qwen. ', 'leading newline and trailing space are kept'


def test_parse_row_keeps_the_continuation_unstripped():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return '  ' + ' '.join(str(i) for i in ids)
    prompt = [1, 2, 3]
    eos, pad = 100257, 100277
    parsed = run218.parse_row(Tok(), torch.tensor(prompt + [7, 8, eos, pad]),
                              3, 8, pad, eos_id=eos)
    assert parsed['continuation'] == '  7 8 100257'
    assert parsed['token_ids'] == [7, 8, eos]
    assert run218.answer_text('I was built by', parsed['continuation']) == \
        'I was built by  7 8 100257'
    assert parsed['hit_cap'] is False


# ---------------------------------------------------------------------------
# Directions, doses and the pinned 41-row batch
# ---------------------------------------------------------------------------

def test_directions_are_unit_paired_and_deterministic():
    hidden = 37                     # not square against the 20 draws
    first = run218.directions_tensor(hidden)
    second = run218.directions_tensor(hidden)
    assert first.shape == (20, hidden)
    assert torch.equal(first, second), 'directions must be reproducible'
    assert torch.allclose(first.norm(dim=1), torch.ones(20), atol=1e-5)
    assert not torch.equal(first[0], first[1]), 'different seeds differ'
    specs = run218.direction_specs((0, 1, 2))
    assert [s['direction_id'] for s in specs] == [
        'seed0+', 'seed0-', 'seed1+', 'seed1-', 'seed2+', 'seed2-']
    assert [s['sign'] for s in specs] == [1, -1, 1, -1, 1, -1]


def test_pilot_and_main_batches_are_41_rows_40_pushed_plus_zero():
    assert run218.PILOT_SEEDS == run218.FULL_SEEDS == tuple(range(20))
    assert run218.n_rows(run218.PILOT_SEEDS) == run218.ROWS_PER_BATCH == 41
    specs = run218.direction_specs(run218.PILOT_SEEDS)
    assert len(specs) == 40
    assert len({spec['direction_id'] for spec in specs}) == 40
    assert sum(1 for spec in specs if spec['sign'] == 1) == 20
    assert sum(1 for spec in specs if spec['sign'] == -1) == 20

    directions = run218.directions_tensor(37)
    vecs = run218.dose_vectors(directions, run218.PILOT_SEEDS, 0.4, 12.5)
    assert vecs.shape == (41, 37)
    assert torch.equal(vecs[-1], torch.zeros(37)), 'zero row is exactly 0'
    assert torch.allclose(vecs[:-1].norm(dim=1),
                          torch.full((40,), 0.4 * 12.5), atol=1e-4)


def test_planned_rows_match_the_v4_design_counts():
    pilot = run218.planned_rows('pilot', (run218.PILOT_LAYER,),
                                run218.PILOT_DOSES)
    assert pilot == 32472                     # 9 x 88 x 41
    full = run218.planned_rows('full', run218.FULL_LAYERS, run218.LADDER)
    assert full == 50512                      # 2 x 7 x 88 x 41
    assert 3 * full == 151536
    gate = run218.planned_rows('gate', (run218.GATE_LAYER,), (0.0,))
    assert gate == 3608                       # 88 x 41
    assert pilot + 2 * gate == 39688          # Amendment 1 item 8


def test_gate_vectors_are_all_exactly_zero():
    vecs = run218.gate_vectors(37)
    assert vecs.shape == (41, 37)
    assert torch.equal(vecs, torch.zeros(41, 37))
    assert float(vecs.abs().max()) == 0.0
    assert len(run218.gate_row_ids()) == 41
    assert run218.gate_row_ids()[0] == 'zero00'
    assert run218.gate_row_ids()[-1] == 'zero40'


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _record(layer=5, dose=0.25, item=0, direction='seed0+', model='Instruct',
            text='x', row_id=None):
    prompt, opening = item // 2, item % 2
    return {'mode': 'full', 'model': model, 'repo': 'r', 'revision': 'v',
            'layer': layer, 'dose': dose, 'seed': 0, 'sign': 1,
            'direction_id': direction,
            'row_id': row_id if row_id is not None else direction,
            'prompt_index': prompt, 'opening_index': opening,
            'opening': run218.OPENINGS[opening], 'item_index': item,
            'row_index': 0, 'token_ids': [1], 'continuation': text,
            'text': text, 'n_new_tokens': 1, 'max_new_tokens': 48,
            'hit_cap': False}


def test_resume_reads_last_write_wins_and_finds_complete_batches(tmp_path):
    path = tmp_path / 'answers.jsonl'
    rows = [
        _record(item=0, direction='seed0+', text='first'),
        _record(item=0, direction='seed0-'),
        _record(item=0, direction='zero'),
        _record(item=1, direction='seed0+'),
        _record(item=0, direction='seed0+', text='second'),  # duplicate row
    ]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\nnot json\n',
                    encoding='utf-8')
    parsed, bad = run218.read_answers(path)
    assert bad == 1
    assert len(parsed) == 4
    assert parsed[('Instruct', 5, 0.25, 0, 'seed0+')]['text'] == 'second', \
        'the later record for a row id wins'
    done = run218.completed_batches(parsed, ['seed0+', 'seed0-', 'zero'])
    assert (('Instruct', 5, 0.25, 0)) in done
    assert (('Instruct', 5, 0.25, 1)) not in done

    run218.write_answers(path, parsed)
    rewritten = path.read_text(encoding='utf-8').splitlines()
    assert len(rewritten) == 4
    assert all(json.loads(line) for line in rewritten)
    assert 'not json' not in rewritten


def test_partial_batch_and_gate_rows_are_counted_correctly():
    rows = {run218.row_key(_record(direction=d)): _record(direction=d)
            for d in ('seed0+', 'seed0-')}
    assert run218.completed_batches(rows, ['seed0+', 'seed0-', 'zero']) == set()
    # Gate rows all carry direction_id 'zero' but distinct row_ids.
    gate = {run218.row_key(_record(direction='zero', row_id=f'zero{i:02d}')):
            _record(direction='zero', row_id=f'zero{i:02d}')
            for i in range(41)}
    assert run218.completed_batches(gate, run218.gate_row_ids()) == {
        ('Instruct', 5, 0.25, 0)}


# ---------------------------------------------------------------------------
# The hook, in a real forward pass
# ---------------------------------------------------------------------------

def _tiny_model(hidden=32, layers=2, vocab=128):
    from transformers import Olmo2Config, Olmo2ForCausalLM
    config = Olmo2Config(hidden_size=hidden, num_hidden_layers=layers,
                         intermediate_size=64, num_attention_heads=2,
                         num_key_value_heads=2, vocab_size=vocab,
                         max_position_embeddings=64)
    torch.manual_seed(0)
    return Olmo2ForCausalLM(config).eval()


def test_hook_adds_the_vector_to_pushed_rows_and_zero_to_the_zero_row():
    model = _tiny_model()
    dense_layers = run218.find_decoder_layers(model)
    assert len(dense_layers) == 2
    hidden = model.get_input_embeddings().weight.shape[1]
    directions = run218.directions_tensor(hidden)
    vecs = run218.dose_vectors(directions, (0,), dose=50.0, median_norm=1.0)
    batch = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert batch.shape[0] == vecs.shape[0]

    with torch.no_grad():
        plain = model(input_ids=batch).logits
    probe = {}
    handle, _state = run218.make_push_hook(dense_layers[1], vecs, probe)
    try:
        with torch.no_grad():
            pushed = model(input_ids=batch).logits
    finally:
        handle.remove()

    assert probe['zero_row_exact'] is True
    assert probe['zero_delta_abs_max'] == 0.0, 'the zero row adds exactly zero'
    assert probe['pushed_delta_abs_max'] > 0.0, 'a pushed row adds a nonzero vector'
    assert probe['input_zero_row_abs_max'] == 0.0
    assert torch.equal(plain[-1], pushed[-1]), 'the zero row must be unchanged'
    assert not torch.equal(plain[0], pushed[0]), 'a large push must change a row'
    assert run218.count_layer_hooks(dense_layers) == 0


def test_gate_vectors_through_the_hook_add_zero_to_every_row():
    model = _tiny_model()
    dense_layers = run218.find_decoder_layers(model)
    hidden = model.get_input_embeddings().weight.shape[1]
    vecs = run218.gate_vectors(hidden)
    batch = torch.tensor([[1, 2, 3, 4]]).repeat(41, 1)
    probe = {}
    handle, _state = run218.make_push_hook(dense_layers[1], vecs, probe)
    try:
        with torch.no_grad():
            model(input_ids=batch)
    finally:
        handle.remove()
    assert probe['zero_row_exact'] is True
    assert probe['zero_delta_abs_max'] == 0.0
    assert probe['pushed_delta_abs_max'] == 0.0, \
        'the gate pass must add zero to all 40 non-zero-slot rows too'


# ---------------------------------------------------------------------------
# Null comparison
# ---------------------------------------------------------------------------

def test_null_comparison_is_exact():
    reference = {'token_ids': [11, 12, 13],
                 'logits': torch.tensor([[0.5, -1.0, 2.0]])}
    same = run218.compare_null(reference, [11, 12, 13],
                               torch.tensor([[0.5, -1.0, 2.0]]))
    assert same == {'ids_ok': True, 'logits_ok': True,
                    'max_abs_logit_diff': 0.0}
    other_ids = run218.compare_null(reference, [11, 12, 14],
                                    torch.tensor([[0.5, -1.0, 2.0]]))
    assert other_ids['ids_ok'] is False
    assert other_ids['logits_ok'] is True
    other_logits = run218.compare_null(
        reference, [11, 12, 13], torch.tensor([[0.5, -1.0, 2.5]]))
    assert other_logits['ids_ok'] is True
    assert other_logits['logits_ok'] is False
    assert other_logits['max_abs_logit_diff'] == pytest.approx(0.5)


def test_last_row_logits_copies_the_row_off_the_batch_buffer():
    """88 items x a [41, 100352] score buffer per saved null is gigabytes: the
    saved logits must be the row alone, not a view that pins the whole batch."""
    scores = torch.arange(41 * 100, dtype=torch.float32).reshape(41, 100)
    row = run218.last_row_logits(scores)
    assert row.shape == (100,)
    assert row.untyped_storage().nbytes() == 100 * row.element_size()
    assert torch.equal(row, scores[-1])
    scores[-1].fill_(0)
    assert torch.equal(row, torch.arange(40 * 100, 41 * 100,
                                         dtype=torch.float32)), \
        'the saved row must be a copy, not a view'


# ---------------------------------------------------------------------------
# Deliberate breaks of a COPY (never the tracked file)
# ---------------------------------------------------------------------------

def _check_dose_scale(module):
    """The dose check the break below must turn red."""
    directions = module.directions_tensor(37)
    vecs = module.dose_vectors(directions, (0, 1), dose=0.4, median_norm=12.5)
    norms = vecs[:-1].norm(dim=-1)
    expected = 0.4 * 12.5
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-4), (
        f'pushed rows must have norm d*median={expected}, got {norms.tolist()}')
    assert torch.equal(vecs[-1], torch.zeros_like(vecs[-1]))


def test_deliberate_break_of_a_copy_turns_the_dose_check_red(tmp_path):
    source = Path(run218.__file__).read_text(encoding='utf-8')
    needle = 'scale = float(dose) * float(median_norm)'
    assert needle in source, 'the break target moved; update this test'
    broken_source = source.replace(needle, 'scale = float(dose)')
    assert broken_source != source
    broken_path = tmp_path / 'run218_broken.py'
    broken_path.write_text(broken_source, encoding='utf-8')

    spec = importlib.util.spec_from_file_location('run218_broken_for_test',
                                                  broken_path)
    broken = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broken)

    with pytest.raises(AssertionError):
        _check_dose_scale(broken)
    _check_dose_scale(run218)


def _check_opening_has_no_trailing_space(module):
    tok = FakeTok()
    text = module.item_text(tok, 'Who made you?', module.OPENINGS[0])
    assert not text.endswith(' '), 'the opening must be appended with no space'


def test_deliberate_break_of_a_copy_turns_the_opening_check_red(tmp_path):
    source = Path(run218.__file__).read_text(encoding='utf-8')
    needle = 'return build_prompt(tok, question) + opening\n'
    assert needle in source, 'the break target moved; update this test'
    broken_source = source.replace(
        needle, "return build_prompt(tok, question) + opening + ' '\n")
    broken_path = tmp_path / 'run218_space_broken.py'
    broken_path.write_text(broken_source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('run218_space_broken',
                                                  broken_path)
    broken = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broken)
    with pytest.raises(AssertionError):
        _check_opening_has_no_trailing_space(broken)
    _check_opening_has_no_trailing_space(run218)


def _check_pilot_batch_is_41(module):
    assert module.PILOT_SEEDS == tuple(range(20))
    assert module.n_rows(module.PILOT_SEEDS) == 41


def test_deliberate_break_of_a_copy_turns_the_pilot_batch_check_red(tmp_path):
    source = Path(run218.__file__).read_text(encoding='utf-8')
    needle = 'PILOT_SEEDS = FULL_SEEDS'
    assert needle in source, 'the break target moved; update this test'
    broken_source = source.replace(needle, 'PILOT_SEEDS = tuple(range(5))')
    broken_path = tmp_path / 'run218_pilot_broken.py'
    broken_path.write_text(broken_source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('run218_pilot_broken',
                                                  broken_path)
    broken = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broken)
    with pytest.raises(AssertionError):
        _check_pilot_batch_is_41(broken)
    _check_pilot_batch_is_41(run218)


# ---------------------------------------------------------------------------
# The tiny-model end-to-end smoke (PROCEDURE 5)
# ---------------------------------------------------------------------------

INSTRUCT_SNAPSHOT_GLOB = ('models--allenai--OLMo-2-0425-1B-Instruct',
                          '48d788eca847d4d7548f375ad03d3c9312f6139e')


def _find_instruct_snapshot() -> Path | None:
    import glob
    cache = Path.home() / '.cache' / 'huggingface' / 'hub'
    repo, revision = INSTRUCT_SNAPSHOT_GLOB
    hits = sorted(glob.glob(str(cache / repo / 'snapshots' / f'{revision}*')))
    for hit in hits:
        if (Path(hit) / 'config.json').exists():
            return Path(hit)
    return None


def _build_tiny_dir(destination: Path, snapshot: Path) -> Path:
    """The real Instruct config and tokenizer, shrunk to hidden 64 and 16 layers."""
    import shutil
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((snapshot / 'config.json').read_text(encoding='utf-8'))
    config.update({'hidden_size': 64, 'num_hidden_layers': 16,
                   'intermediate_size': 256, 'num_attention_heads': 4,
                   'num_key_value_heads': 4, 'torch_dtype': 'float32'})
    (destination / 'config.json').write_text(json.dumps(config, indent=1),
                                             encoding='utf-8')
    for name in ('tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'generation_config.json'):
        source = snapshot / name
        if source.exists():
            shutil.copy(source, destination / name)
    return destination


def _run_cli(script: Path, args: list[str], cwd: Path) -> None:
    import subprocess
    command = [sys.executable, '-I', str(script)] + args
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True,
                               text=True, timeout=1800)
    if completed.returncode != 0:
        raise AssertionError(
            f'{command} failed with {completed.returncode}\n'
            f'--- stdout (tail) ---\n{completed.stdout[-3000:]}\n'
            f'--- stderr (tail) ---\n{completed.stderr[-3000:]}')


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding='utf-8').splitlines() if line.strip()]


@pytest.mark.skipif(os.environ.get('PART218_SKIP_SMOKE') == '1',
                    reason='PART218_SKIP_SMOKE=1')
def test_tiny_model_end_to_end_smoke_pilot_gate_and_full(tmp_path):
    """The REAL command line on a tiny random OLMo 2 through pilot, gate, full."""
    snapshot = _find_instruct_snapshot()
    if snapshot is None:
        pytest.skip('the cached Instruct config/tokenizer is not available')
    tiny = _build_tiny_dir(tmp_path / 'tiny', snapshot)
    out = tmp_path / 'out'
    common = ['--tiny', str(tiny), '--out', str(out), '--device', 'cpu',
              '--dtype', 'float32', '--max-new', '2']

    _run_cli(HERE / 'run218.py', ['--mode', 'pilot'] + common, tmp_path)
    _run_cli(HERE / 'run218.py',
             ['--mode', 'gate', '--model', 'SFT'] + common, tmp_path)
    _run_cli(HERE / 'run218.py',
             ['--mode', 'gate', '--model', 'DPO'] + common, tmp_path)
    _run_cli(HERE / 'run218.py',
             ['--mode', 'full', '--model', 'Instruct',
              '--doses', '0.25,1.0'] + common, tmp_path)
    # GLM review HIGH-1: rerunning a finished command must stay complete with
    # the null intact (it used to exit 3 with stopped_null_mismatch).
    _run_cli(HERE / 'run218.py', ['--mode', 'pilot'] + common, tmp_path)

    pilot = json.loads((out / 'pilot' / 'run.json').read_text(encoding='utf-8'))
    gate_sft = json.loads((out / 'gate' / 'SFT' / 'run.json').read_text(
        encoding='utf-8'))
    gate_dpo = json.loads((out / 'gate' / 'DPO' / 'run.json').read_text(
        encoding='utf-8'))
    full = json.loads((out / 'Instruct' / 'run.json').read_text(encoding='utf-8'))
    assert pilot['null']['hooked_zero_vs_start']['this_process']['comparisons'] == 0
    assert 'previous run.json record' in pilot['null']['hooked_zero_vs_start'].get('reason', '')

    # Planned rows are the v4 design's counts.
    assert pilot['rows_planned'] == 32472
    assert gate_sft['rows_planned'] == gate_dpo['rows_planned'] == 3608
    assert full['rows_planned'] == 2 * 2 * 88 * 41 == 14432
    for meta in (pilot, full):
        assert meta['rows_written'] == meta['rows_planned']
        assert meta['status'] == 'complete'
        assert meta['null']['all_ok'] is True, 'the no-hook null must match'
        assert meta['null']['end']['vs_start']['all_ok'] is True
        assert meta['null']['hooked_zero_vs_start']['all_ok'] is True
        assert meta['push_check']['pushed_delta_abs_max'] > 0.0
        assert meta['push_check']['zero_delta_abs_max'] == 0.0
        assert meta['push_check']['zero_row_exact'] is True
        assert meta['directions']['shape'] == [20, 64]
        assert meta['hooks_after_generation'] == 0
        assert meta['tokenizer']['unk_is_eos'] is True
        assert meta['n_items'] == 88 and meta['n_rows_per_batch'] == 41
        assert meta['openings'] == list(run218.OPENINGS)

    # --- gate pass: 41 zero-push rows per item, no pushed row at all ---
    for meta in (gate_sft, gate_dpo):
        assert meta['rows_written'] == meta['rows_planned'] == 3608
        assert meta['status'] == 'complete'
        assert meta['n_rows_per_batch'] == 41 and meta['n_items'] == 88
        assert meta['gate']['zero_push'] is True
        assert meta['gate']['vec_abs_max'] == 0.0
        assert meta['gate']['no_nonzero_push'] is True
        assert meta['gate']['pushed_delta_abs_max'] == 0.0
        assert meta['gate']['hook_calls'] >= 1, 'the hook must be installed'
        assert meta['gate']['probe_ran'] is True
        assert meta['hooks_after_generation'] == 0
    gate_rows = _read_jsonl(out / 'gate' / 'SFT' / 'answers.jsonl')
    assert len(gate_rows) == 3608
    assert all(row['direction_id'] == 'zero' for row in gate_rows)
    assert all(row['seed'] is None and row['sign'] == 0 for row in gate_rows)
    first_gate_item = [row for row in gate_rows if row['item_index'] == 0]
    assert len(first_gate_item) == 41
    assert len({tuple(row['token_ids']) for row in first_gate_item}) == 1, \
        'all 41 zero-push rows of an item must be identical'

    # Shapes: 41 rows per batch, zero row last; item order and answer text.
    pilot_rows = _read_jsonl(out / 'pilot' / 'answers.jsonl')
    full_rows = _read_jsonl(out / 'Instruct' / 'answers.jsonl')
    assert len(pilot_rows) == 32472 and len(full_rows) == 14432
    first_full_batch = [r for r in full_rows
                        if r['layer'] == 5 and r['dose'] == 0.25
                        and r['item_index'] == 0]
    assert len(first_full_batch) == 41
    assert first_full_batch[-1]['direction_id'] == 'zero'
    assert len({r['direction_id'] for r in first_full_batch}) == 41
    first_pilot_batch = [r for r in pilot_rows
                         if r['layer'] == 5 and r['dose'] == 0.15
                         and r['item_index'] == 0]
    assert len(first_pilot_batch) == 41, 'the pilot batch is 41 rows in v4'
    assert sum(1 for r in first_pilot_batch if r['direction_id'] != 'zero') == 40
    for rows in (pilot_rows, full_rows):
        assert all(row['text'] == row['opening'] + row['continuation']
                   for row in rows), 'text = opening + continuation exactly'
        assert all(row['item_index'] == 2 * row['prompt_index'] + row['opening_index']
                   for row in rows)
        assert all(row['opening'] == run218.OPENINGS[row['opening_index']]
                   for row in rows)

    # Scorer runs end to end on what the runner wrote.
    _run_cli(HERE / 'score218.py', ['--run-dir', str(out)], tmp_path)
    score = json.loads((out / 'score218.json').read_text(encoding='utf-8'))
    assert set(score['predictions']) == {f'218.{i}' for i in range(1, 9)}
    assert score['gold']['ok'] is True and score['gold']['matched'] == 87
    assert score['prompts']['sha256'] == PROMPTS_SHA256
    for prediction in score['predictions'].values():
        assert prediction['verdict'] in ('held', 'failed', 'not read')
    assert (out / 'audit218.jsonl').exists()
    assert score['models']['Instruct'] is not None
    assert score['gate_passes']['SFT'] is not None
    assert score['gate_passes']['DPO'] is not None
    assert score['pilot'] is not None
    # The tiny random model is not OLMo's unpushed self, so every verdict is
    # honestly "not read"; the zero row still agrees with itself everywhere.
    identity = score['models']['Instruct']['zero_row_identity']
    assert identity['identical_items'] == identity['items'] == 88
    assert score['pilot']['stop_rule']['triggered'] is True
    gate_report = score['gate_passes']['SFT']['per_item']
    assert len(gate_report) == 88
    assert all(row['all_identical_rows'] is True for row in gate_report.values())
    assert all(row['n_zero_rows'] == 41 for row in gate_report.values())


def test_runner_registered_v4_constants_match_the_design():
    """Every v4 number is pinned so a copy cannot drift with the suite green."""
    assert run218.MAX_NEW == 48
    assert run218.PILOT_LAYER == 5
    assert tuple(run218.FULL_LAYERS) == (5, 10)
    assert run218.GATE_LAYER == 5
    assert tuple(run218.OPENINGS) == ('I was developed by', 'I was built by')
    assert run218.N_PROMPTS == 44
    assert run218.N_OPENINGS == 2
    assert run218.N_ITEMS == 88
    assert run218.ROWS_PER_BATCH == 41
    assert run218.PILOT_SEEDS == run218.FULL_SEEDS == tuple(range(20))
    assert run218.N_DRAWS == 20
    assert tuple(run218.PILOT_DOSES) == (0.15, 0.20, 0.25, 0.30, 0.40,
                                         0.50, 0.75, 1.0, 1.5)
    assert tuple(run218.LADDER) == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    assert run218.PROMPTS_PATH.name == 'prompts218-v4.json'


def test_hit_cap_excludes_an_answer_ending_on_eos_at_the_cap():
    """GLM review LOW-1: EOS as the last allowed token is a natural end, not the cap."""
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return ' '.join(str(i) for i in ids)
    prompt = [1, 2, 3]
    eos, pad = 100257, 100277
    at_cap_eos = run218.parse_row(Tok(), torch.tensor(prompt + [7, 8, 9, eos]), 3, 4, pad, eos_id=eos)
    at_cap_word = run218.parse_row(Tok(), torch.tensor(prompt + [7, 8, 9, 10]), 3, 4, pad, eos_id=eos)
    short = run218.parse_row(Tok(), torch.tensor(prompt + [7, eos, pad, pad]), 3, 4, pad, eos_id=eos)
    assert at_cap_eos['hit_cap'] is False and at_cap_eos['n_new_tokens'] == 4
    assert at_cap_word['hit_cap'] is True
    assert short['hit_cap'] is False and short['n_new_tokens'] == 2
