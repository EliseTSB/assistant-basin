#!/usr/bin/env python3
"""Tests for Part 218's runner (`run218.py`).

    python -I -m pytest -q part218/

The last test copies the tracked runner to a temporary directory, breaks the
dose scale in the copy (never the tracked file), and requires the dose check to
go red on the copy while the tracked module stays green.  See BUILD-NOTES.md
for the printed red run.
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


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_prompts_file_equals_part173_new44b():
    path = ROOT / 'part173' / 'prompts173.py'
    assert path.exists(), 'Part 173 prompts file is missing'
    spec = importlib.util.spec_from_file_location('prompts173_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prompts = json.loads((HERE / 'prompts218.json').read_text(encoding='utf-8'))
    assert isinstance(prompts, list)
    assert len(prompts) == 44
    assert len(set(prompts)) == 44
    assert tuple(prompts) == tuple(module.NEW44B), \
        'prompts218.json must be Part 173 NEW44B, not a retyped list'
    assert run218.load_prompts() == list(module.NEW44B)


# ---------------------------------------------------------------------------
# Directions and doses
# ---------------------------------------------------------------------------

def test_directions_are_unit_paired_and_deterministic():
    hidden = 37                     # not square against the 20 draws
    first = run218.directions_tensor(hidden)
    second = run218.directions_tensor(hidden)
    assert first.shape == (20, hidden)
    assert torch.equal(first, second), 'directions must be reproducible'
    assert torch.allclose(first.norm(dim=1), torch.ones(20), atol=1e-5)
    assert not torch.equal(first[0], first[1]), 'different seeds differ'
    # both signs are built at use time, so row 2s / 2s+1 are +v / -v
    specs = run218.direction_specs((0, 1, 2))
    assert [s['direction_id'] for s in specs] == [
        'seed0+', 'seed0-', 'seed1+', 'seed1-', 'seed2+', 'seed2-']
    assert [s['sign'] for s in specs] == [1, -1, 1, -1, 1, -1]


def test_dose_vectors_have_d_times_median_norm_and_an_exact_zero_row():
    hidden = 37
    directions = run218.directions_tensor(hidden)
    median = 12.5
    dose = 0.4
    vecs = run218.dose_vectors(directions, (0, 1, 3), dose, median)
    assert vecs.shape == (7, hidden)          # 3 seeds x 2 signs + zero
    expected = dose * median                  # 5.0
    pushed = vecs[:-1]
    assert torch.allclose(pushed.norm(dim=1),
                          torch.full((6,), expected), atol=1e-4)
    assert torch.equal(vecs[-1], torch.zeros(hidden)), 'zero row is exactly 0'
    # order: seed 0 +, seed 0 -, seed 1 +, ...
    assert torch.equal(vecs[0], -vecs[1])
    assert torch.equal(vecs[2], -vecs[3])
    assert torch.equal(vecs[4], -vecs[5])
    assert not torch.equal(vecs[0], vecs[2])


def test_planned_rows_match_the_design_counts():
    pilot = run218.planned_rows('pilot', (run218.PILOT_LAYER,),
                                run218.PILOT_DOSES)
    assert pilot == 4356                      # 9 x 44 x 11
    full = run218.planned_rows('full', run218.FULL_LAYERS, run218.LADDER)
    assert full == 25256                      # 2 x 7 x 44 x 41
    assert 3 * full == 75768
    assert run218.n_rows(run218.PILOT_SEEDS) == 11
    assert run218.n_rows(run218.FULL_SEEDS) == 41


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _record(layer=5, dose=0.25, prompt=0, direction='seed0+', model='Instruct',
            text='x'):
    return {'mode': 'full', 'model': model, 'repo': 'r', 'revision': 'v',
            'layer': layer, 'dose': dose, 'seed': 0, 'sign': 1,
            'direction_id': direction, 'prompt_index': prompt, 'row_index': 0,
            'token_ids': [1], 'text': text, 'n_new_tokens': 1,
            'max_new_tokens': 48, 'hit_cap': False}


def test_resume_reads_last_write_wins_and_finds_complete_batches(tmp_path):
    path = tmp_path / 'answers.jsonl'
    rows = [
        _record(prompt=0, direction='seed0+', text='first'),
        _record(prompt=0, direction='seed0-'),
        _record(prompt=0, direction='zero'),
        _record(prompt=1, direction='seed0+'),
        _record(prompt=0, direction='seed0+', text='second'),  # duplicate row
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


def test_partial_batch_is_not_counted_complete():
    rows = {run218.row_key(_record(direction=d)): _record(direction=d)
            for d in ('seed0+', 'seed0-')}
    assert run218.completed_batches(rows, ['seed0+', 'seed0-', 'zero']) == set()


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


# ---------------------------------------------------------------------------
# Deliberate break of a COPY (never the tracked file)
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
    # The tracked module still passes the same check.
    _check_dose_scale(run218)


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
                               text=True, timeout=900)
    if completed.returncode != 0:
        raise AssertionError(
            f'{command} failed with {completed.returncode}\n'
            f'--- stdout (tail) ---\n{completed.stdout[-3000:]}\n'
            f'--- stderr (tail) ---\n{completed.stderr[-3000:]}')


@pytest.mark.skipif(os.environ.get('PART218_SKIP_SMOKE') == '1',
                    reason='PART218_SKIP_SMOKE=1')
def test_tiny_model_end_to_end_smoke(tmp_path):
    """The REAL command line on a tiny random OLMo 2 through both modes' last write."""
    snapshot = _find_instruct_snapshot()
    if snapshot is None:
        pytest.skip('the cached Instruct config/tokenizer is not available')
    tiny = _build_tiny_dir(tmp_path / 'tiny', snapshot)
    out = tmp_path / 'out'
    common = ['--tiny', str(tiny), '--out', str(out), '--device', 'cpu',
              '--dtype', 'float32', '--max-new', '4']

    _run_cli(HERE / 'run218.py', ['--mode', 'pilot'] + common, tmp_path)
    _run_cli(HERE / 'run218.py',
             ['--mode', 'full', '--model', 'Instruct',
              '--doses', '0.25,1.0'] + common, tmp_path)

    # GLM review HIGH-1: rerunning a finished command must stay complete with the
    # null intact (it used to exit 3 with stopped_null_mismatch on a clean run).
    _run_cli(HERE / 'run218.py', ['--mode', 'pilot'] + common, tmp_path)

    pilot = json.loads((out / 'pilot' / 'run.json').read_text(encoding='utf-8'))
    full = json.loads((out / 'Instruct' / 'run.json').read_text(encoding='utf-8'))
    assert pilot['null']['hooked_zero_vs_start']['this_process']['comparisons'] == 0
    assert 'previous run.json record' in pilot['null']['hooked_zero_vs_start'].get('reason', '')

    # Planned rows are the design's counts.
    assert pilot['rows_planned'] == 4356
    assert full['rows_planned'] == 2 * 2 * 44 * 41 == 7216
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
    # Both modes share one directions file byte for byte.
    assert full['directions']['sha256'] == pilot['directions']['sha256']
    assert full['directions']['created_this_run'] is False

    # Shapes: 11 rows per pilot batch, 41 per full batch, zero row last.
    pilot_rows = [json.loads(line) for line in
                  (out / 'pilot' / 'answers.jsonl').read_text(
                      encoding='utf-8').splitlines()]
    full_rows = [json.loads(line) for line in
                 (out / 'Instruct' / 'answers.jsonl').read_text(
                     encoding='utf-8').splitlines()]
    assert len(pilot_rows) == 4356 and len(full_rows) == 7216
    first_full_batch = [r for r in full_rows
                        if r['layer'] == 5 and r['dose'] == 0.25
                        and r['prompt_index'] == 0]
    assert len(first_full_batch) == 41
    assert first_full_batch[-1]['direction_id'] == 'zero'
    assert len({r['direction_id'] for r in first_full_batch}) == 41

    # Scorer runs end to end on what the runner wrote.
    _run_cli(HERE / 'score218.py', ['--run-dir', str(out)], tmp_path)
    score = json.loads((out / 'score218.json').read_text(encoding='utf-8'))
    assert set(score['predictions']) == {f'218.{i}' for i in range(1, 9)}
    assert score['gold']['ok'] is True and score['gold']['matched'] == 87
    for prediction in score['predictions'].values():
        assert prediction['verdict'] in ('held', 'failed', 'not read')
    assert (out / 'audit218.jsonl').exists()
    assert score['models']['Instruct'] is not None
    assert score['pilot'] is not None
    # The tiny random model is not OLMo's unpushed self, so every verdict is
    # honestly "not read"; the zero row still agrees with itself everywhere.
    identity = score['models']['Instruct']['zero_row_identity']
    assert identity['identical_prompts'] == identity['prompts'] == 44
    assert score['pilot']['stop_rule']['triggered'] is True



def test_runner_registered_constants_match_the_design():
    """GLM review LOW-3: MAX_NEW, PILOT_LAYER and FULL_LAYERS could change with the
    whole suite green. DESIGN.md: max_new_tokens = 48, pilot layer 5, layers 5 and 10."""
    assert run218.MAX_NEW == 48
    assert run218.PILOT_LAYER == 5
    assert tuple(run218.FULL_LAYERS) == (5, 10)


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
