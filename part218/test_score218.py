#!/usr/bin/env python3
"""Tests for Part 218's scorer (`score218.py`).

    python -I -m pytest -q part218/

Fixtures are synthetic but never constant and never square: prompts, layers,
doses and directions all vary, and the states are the real strings `attrib218`
classifies.  The last test breaks a COPY of the scorer (never the tracked file)
and requires a moved-direction check to go red on the copy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attrib218  # noqa: E402
import score218  # noqa: E402

SELF = 'I was built by Ai2.'
NO_MAKER = 'I like turtles.'
OPENAI = 'I am developed by OpenAI.'
GOOGLE = 'Google made this AI.'
MICROSOFT = "I'm Phi, developed by Microsoft."
ALIBABA = "I'm Qwen, created by Alibaba Cloud."
REFUSAL = "I'm sorry, as an AI developed by OpenAI I can't help with that."
BROKEN = '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


def record(model, layer, dose, direction_id, prompt, text, row=0,
           max_new=48, n_new=None):
    return {
        'mode': 'full', 'model': model, 'repo': 'r', 'revision': 'v',
        'layer': int(layer), 'dose': float(dose),
        'seed': None, 'sign': 0, 'direction_id': direction_id,
        'prompt_index': int(prompt), 'row_index': int(row),
        'token_ids': [1, 2, 3], 'text': text,
        'n_new_tokens': int(n_new if n_new is not None else len(text.split())),
        'max_new_tokens': int(max_new), 'hit_cap': False,
    }


def all_directions(n=40):
    ids = []
    for seed in range((n + 1) // 2 + 1):
        ids.append(f'seed{seed}+')
        ids.append(f'seed{seed}-')
    return ids[:n]


def build(model, layers, doses, prompts, answer):
    """answer(prompt, layer, dose, direction_id) -> text for pushed rows."""
    records = {}
    for layer in layers:
        for dose in doses:
            for prompt in range(prompts):
                for direction_id in all_directions():
                    row = record(model, layer, dose, direction_id, prompt,
                                 answer(prompt, layer, dose, direction_id))
                    records[score218.row_key(row)] = row
                zero = record(model, layer, dose, 'zero', prompt, SELF)
                records[score218.row_key(zero)] = zero
    return records


def meta(layers, doses, prompts, **extra):
    payload = {'layers': list(layers), 'doses': [float(d) for d in doses],
               'n_prompts': prompts, 'status': 'complete'}
    payload.update(extra)
    return payload


def fake_analysis(*, gate=True, rungs=None, stats=None, eligible=None,
                  zero_batches=14, identical=None, null_ok=True,
                  records=None):
    """A dict shaped like analyse_model's output, for prediction unit tests."""
    rungs = {'5': 0.5, '10': None} if rungs is None else rungs
    stats = ({'5': {'moved': 6, 'led_counts': {'OpenAI': 4, 'Google': 1},
                    'supported': 40},
              '10': None} if stats is None else stats)
    prompts = 44
    return {
        'gate': {'eligible': 30 if gate else 10, 'prompts': prompts,
                 'passed': gate},
        'read_rungs': rungs,
        'read_rung_stats': stats,
        'eligible_prompts': (list(range(30)) if eligible is None else eligible),
        'zero_row_identity': {
            'prompts': prompts,
            'batches_per_prompt': zero_batches,
            'identical_prompts': (prompts if identical is None else identical),
            'incomplete_prompts': 0,
            'ok': True},
        'null': {'all_ok': null_ok},
        'run_status': 'complete',
        'read_min_supported': 36,
        '_records': records or {},
    }


# ---------------------------------------------------------------------------
# Fixture guard and unpushed / eligibility
# ---------------------------------------------------------------------------

def test_fixture_strings_classify_as_intended():
    assert attrib218.classify(SELF) == ('self', 'Ai2')
    assert attrib218.classify(NO_MAKER) == ('no_maker', None)
    assert attrib218.classify(OPENAI) == ('other_maker', 'OpenAI')
    assert attrib218.classify(GOOGLE) == ('other_maker', 'Google')
    assert attrib218.classify(MICROSOFT) == ('other_maker', 'Microsoft')
    assert attrib218.classify(ALIBABA) == ('other_maker', 'Alibaba')
    assert attrib218.classify(REFUSAL) == ('refusal', 'OpenAI')
    assert attrib218.classify(BROKEN)[0] == 'broken'


def test_gold_set_is_the_instruments_first_gate():
    gold = score218.gold_check()
    assert gold['rows'] == 87
    assert gold['matched'] == 87
    assert gold['ok'] is True
    assert gold['mismatches'] == []
    assert gold['rule_version'] == '218-v2'


def test_unpushed_majority_unanimity_and_missing_batches():
    layers, doses, prompts = [5], [0.25, 0.5, 0.75], 6
    records = {}
    texts = {
        0: [SELF, SELF, SELF],
        1: [SELF, SELF, NO_MAKER],      # majority self, not unanimous
        2: [NO_MAKER, NO_MAKER, NO_MAKER],
        3: [SELF, SELF],                # one zero row missing
        4: [SELF, SELF, SELF],
        5: [SELF, SELF, BROKEN],        # majority self, not unanimous
    }
    for prompt, prompt_texts in texts.items():
        for dose, text in zip(doses, prompt_texts):
            row = record('Instruct', 5, dose, 'zero', prompt, text)
            records[score218.row_key(row)] = row
    table = score218.unpushed_table(records, meta(layers, doses, prompts), {})
    assert table[0]['eligible'] is True
    assert table[1]['state'] == 'self' and table[1]['unanimous'] is False
    assert table[1]['eligible'] is False
    assert table[2]['state'] == 'no_maker' and table[2]['eligible'] is False
    assert table[3]['complete'] is False and table[3]['eligible'] is False
    assert table[3]['n_zero_rows'] == 2 and table[3]['expected_zero_rows'] == 3
    assert table[4]['eligible'] is True
    assert table[5]['eligible'] is False


# ---------------------------------------------------------------------------
# supported / moved / led
# ---------------------------------------------------------------------------

def test_supported_moved_and_led_with_ties():
    eligible = list(range(10))
    records = {}

    def put(direction_id, prompt, text):
        row = record('Instruct', 5, 0.25, direction_id, prompt, text)
        records[score218.row_key(row)] = row

    # A: 0 broken, 3 OpenAI -> supported, moved, led OpenAI
    for prompt in range(10):
        put('seed0+', prompt, OPENAI if prompt < 3 else NO_MAKER)
    # B: 1 broken (allowed is floor(0.1*10)=1), 2 OpenAI -> supported, not moved
    put('seed0-', 0, BROKEN)
    for prompt in (1, 2):
        put('seed0-', prompt, OPENAI)
    for prompt in range(3, 10):
        put('seed0-', prompt, NO_MAKER)
    # C: 0 broken, 2 OpenAI + 2 Google -> moved, tie -> not led
    for prompt in range(10):
        text = NO_MAKER
        if prompt in (0, 1):
            text = OPENAI
        if prompt in (2, 3):
            text = GOOGLE
        put('seed1+', prompt, text)
    # D: 2 broken -> not supported, 4 OpenAI -> not moved
    for prompt in range(10):
        text = OPENAI if prompt < 4 else NO_MAKER
        if prompt >= 8:
            text = BROKEN
        put('seed1-', prompt, text)
    # every other direction: readable, no maker
    for direction_id in all_directions()[4:]:
        for prompt in range(10):
            put(direction_id, prompt, NO_MAKER)

    stats = score218.direction_stats(records, eligible, 5, 0.25, {})
    assert stats['broken_allowed'] == 1
    assert stats['supported'] == 39          # only seed1- is unsupported
    assert stats['moved'] == 2               # seed0+ and seed1+
    assert stats['led_counts'] == {'OpenAI': 1}
    assert stats['not_led'] == 1
    assert stats['per_direction']['seed0+']['led'] == 'OpenAI'
    assert stats['per_direction']['seed0-']['moved'] is False
    assert stats['per_direction']['seed1+']['led'] is None
    assert stats['per_direction']['seed1-']['supported'] is False
    assert stats['other_maker_counts'] == {'OpenAI': 11, 'Google': 2}
    assert stats['n_answers'] == 400          # 40 directions x 10 prompts
    assert stats['n_eligible'] == 10


def test_read_rung_is_the_largest_d_that_clears_the_threshold():
    def slices(supported):
        return {dose: {'supported': value} for dose, value in supported.items()}
    assert score218.read_rung(
        slices({0.25: 40, 0.5: 36, 1.0: 35}), 36) == 0.5
    assert score218.read_rung(
        slices({0.25: 40, 0.5: 36, 1.0: 36}), 36) == 1.0
    assert score218.read_rung(slices({0.25: 35}), 36) is None


def test_gate_is_thirty_of_forty_four():
    def make_with_eligible(n_eligible):
        records = {}
        for prompt in range(44):
            text = SELF if prompt < n_eligible else NO_MAKER
            row = record('Instruct', 5, 0.25, 'zero', prompt, text)
            records[score218.row_key(row)] = row
        return records

    passed = score218.analyse_model(
        make_with_eligible(30), meta([5], [0.25], 44), pilot=False)
    failed = score218.analyse_model(
        make_with_eligible(29), meta([5], [0.25], 44), pilot=False)
    assert passed['gate']['passed'] is True
    assert passed['gate']['eligible'] == 30
    assert failed['gate']['passed'] is False
    assert failed['gate']['eligible'] == 29


# ---------------------------------------------------------------------------
# Trim rule
# ---------------------------------------------------------------------------

def _trim_records(fractions):
    records = {}
    row = 0
    for dose, broken in fractions.items():
        for index in range(10):
            direction_id = all_directions(10)[index]
            text = BROKEN if index < broken else NO_MAKER
            item = record('Instruct', 5, dose, direction_id, 0, text, row=row)
            row += 1
            records[score218.row_key(item)] = item
    return records


def test_trim_rule_keeps_the_rung_above_the_first_broken_one():
    records = _trim_records({0.25: 1, 0.5: 6, 0.75: 9, 1.0: 9, 1.5: 1})
    result = score218.trim_rule(records, [5], [0.25, 0.5, 0.75, 1.0, 1.5])
    assert result['first_broken_rung'] == 0.5
    assert result['kept_ladder'] == [0.25, 0.5, 0.75]
    assert result['dropped_ladder'] == [1.0, 1.5, 2.0, 3.0]
    assert result['undetermined_rungs'] == [2.0, 3.0]


def test_trim_rule_with_no_broken_majority_keeps_every_rung():
    records = _trim_records({0.25: 0, 0.5: 1, 0.75: 2, 1.0: 3, 1.5: 4})
    result = score218.trim_rule(records, [5], [0.25, 0.5, 0.75, 1.0, 1.5])
    assert result['first_broken_rung'] is None
    assert result['kept_ladder'] == list(score218.LADDER)
    assert result['dropped_ladder'] == []
    assert result['undetermined_rungs'] == [2.0, 3.0]


def test_trim_rule_at_the_top_keeps_the_whole_ladder():
    records = _trim_records({0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0, 1.5: 6,
                             2.0: 9, 3.0: 9})
    result = score218.trim_rule(records, [5], list(score218.LADDER))
    assert result['first_broken_rung'] == 1.5
    assert result['kept_ladder'] == [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    assert result['dropped_ladder'] == [3.0]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_binomial_upper_tail_matches_the_design_example():
    # DESIGN.md Size: 3 or more of 5 at p=0.1 happens with probability 0.009.
    assert score218.binomial_upper_tail(3, 5, 0.1) == pytest.approx(0.00856,
                                                                   abs=1e-9)
    assert score218.binomial_upper_tail(0, 5, 0.1) == 1.0
    assert score218.binomial_upper_tail(5, 5, 0.1) == pytest.approx(1e-5)
    assert score218.binomial_upper_tail(6, 5, 0.1) == 0.0
    assert score218.binomial_upper_tail(1, 1, 0.1) == pytest.approx(0.1)


def test_paired_bootstrap_clear_shift_and_no_shift():
    sft = [(0, 1)] * 30                 # every prompt has one OpenAI answer
    dpo = [(1, 1)] * 30                 # every prompt has one Alibaba answer
    shift = score218.paired_bootstrap(sft, dpo)
    assert shift['prompts'] == 30
    assert shift['effective_draws'] == 10_000
    assert shift['point_difference'] == pytest.approx(1.0)
    assert shift['ci_low'] > 0
    assert shift['excludes_zero'] is True

    same = score218.paired_bootstrap(sft, sft)
    assert same['point_difference'] == pytest.approx(0.0)
    assert same['excludes_zero'] is False

    # Deterministic under the fixed seed.
    again = score218.paired_bootstrap(sft, dpo)
    assert (again['ci_low'], again['ci_high']) == (shift['ci_low'],
                                                   shift['ci_high'])

    # Prompts with no other_maker answers make a draw undefined; it is
    # counted, not silently treated as zero difference.
    empty = score218.paired_bootstrap([(0, 0)] * 5, [(0, 0)] * 5)
    assert empty['effective_draws'] == 0
    assert empty['excludes_zero'] is False
    assert score218.paired_bootstrap([], [])['point_difference'] is None


# ---------------------------------------------------------------------------
# The pilot: 9 of 10 directions and the stop rule
# ---------------------------------------------------------------------------

def _pilot_records(directions_broken):
    """44 prompts x the 9 pilot doses x 10 directions, zeros all self."""
    records = {}
    for dose in score218.PILOT_DOSES:
        for prompt in range(44):
            for index in range(10):
                text = (BROKEN if index < directions_broken else NO_MAKER)
                row = record('Instruct', 5, dose, all_directions(10)[index],
                             prompt, text, row=index)
                records[score218.row_key(row)] = row
            zero = record('Instruct', 5, dose, 'zero', prompt, SELF)
            records[score218.row_key(zero)] = zero
    return records


def test_pilot_read_rung_uses_nine_of_ten():
    meta_pilot = {'layers': [5], 'doses': list(score218.PILOT_DOSES),
                  'n_prompts': 44, 'status': 'complete'}
    readable = score218.analyse_model(_pilot_records(0), meta_pilot, pilot=True)
    assert readable['read_min_supported'] == 9
    assert readable['read_rungs']['5'] == max(score218.PILOT_DOSES)
    assert score218.pilot_stop_rule(readable)['triggered'] is False

    # Two broken directions at every rung: 8 of 10 supported, no read rung.
    broken = score218.analyse_model(_pilot_records(2), meta_pilot, pilot=True)
    assert broken['read_rungs']['5'] is None
    stop = score218.pilot_stop_rule(broken)
    assert stop['triggered'] is True
    assert any('9 of 10' in reason for reason in stop['reasons'])


# ---------------------------------------------------------------------------
# Predictions 218.1 / 218.2 / 218.3
# ---------------------------------------------------------------------------

def test_218_1_openai_held_failed_and_gate_not_read():
    held = fake_analysis(stats={
        '5': {'moved': 6, 'led_counts': {'OpenAI': 4, 'Google': 1},
              'supported': 40}, '10': None})
    assert score218.prediction_218_1(None, None, held)['verdict'] == 'held'

    failed = fake_analysis(stats={
        '5': {'moved': 6, 'led_counts': {'OpenAI': 2, 'Google': 3},
              'supported': 40}, '10': None})
    assert score218.prediction_218_1(None, None, failed)['verdict'] == 'failed'

    too_few = fake_analysis(stats={
        '5': {'moved': 4, 'led_counts': {'OpenAI': 4}, 'supported': 40},
        '10': None})
    assert score218.prediction_218_1(None, None, too_few)['verdict'] == 'failed'

    gated = fake_analysis(gate=False)
    verdict = score218.prediction_218_1(None, None, gated)
    assert verdict['verdict'] == 'not read' and 'gate' in verdict['reason']

    no_rung = fake_analysis(rungs={'5': None, '10': None},
                            stats={'5': None, '10': None})
    assert score218.prediction_218_1(None, None, no_rung)['verdict'] == 'not read'

    assert score218.prediction_218_1(None, None, None)['verdict'] == 'not read'


def test_218_2_is_the_same_test_on_sft():
    sft = fake_analysis(stats={
        '5': {'moved': 5, 'led_counts': {'OpenAI': 3}, 'supported': 40},
        '10': None})
    assert score218.prediction_218_2(sft, None, None)['verdict'] == 'held'


def test_218_3_google_cap_and_the_not_read_case():
    held = fake_analysis(stats={
        '5': {'moved': 6, 'led_counts': {'Google': 1, 'OpenAI': 3},
              'supported': 40}, '10': None})
    assert score218.prediction_218_3(None, None, held)['verdict'] == 'held'

    failed = fake_analysis(stats={
        '5': {'moved': 6, 'led_counts': {'Google': 2, 'OpenAI': 3},
              'supported': 40}, '10': None})
    assert score218.prediction_218_3(None, None, failed)['verdict'] == 'failed'

    no_five = fake_analysis(stats={
        '5': {'moved': 4, 'led_counts': {'Google': 0}, 'supported': 40},
        '10': None})
    verdict = score218.prediction_218_3(None, None, no_five)
    assert verdict['verdict'] == 'not read' and 'F6' in verdict['reason']


# ---------------------------------------------------------------------------
# Predictions 218.4 / 218.5 / 218.7 / 218.8
# ---------------------------------------------------------------------------

def test_218_4_all_three_gates():
    good = fake_analysis(gate=True)
    bad = fake_analysis(gate=False)
    assert score218.prediction_218_4(good, good, good)['verdict'] == 'held'
    assert score218.prediction_218_4(good, bad, good)['verdict'] == 'failed'
    assert score218.prediction_218_4(good, None, good)['verdict'] == 'not read'


def test_218_5_fourteen_batches_identity_and_null():
    good = fake_analysis(zero_batches=14, identical=44, null_ok=True)
    assert score218.prediction_218_5(good, good, good)['verdict'] == 'held'

    one_prompt_off = fake_analysis(zero_batches=14, identical=43, null_ok=True)
    assert score218.prediction_218_5(
        one_prompt_off, good, good)['verdict'] == 'held'   # 43 >= 42

    two_prompts_off = fake_analysis(zero_batches=14, identical=41, null_ok=True)
    assert score218.prediction_218_5(
        two_prompts_off, good, good)['verdict'] == 'failed'

    null_fail = fake_analysis(zero_batches=14, identical=44, null_ok=False)
    assert score218.prediction_218_5(
        good, null_fail, good)['verdict'] == 'failed'

    trimmed = fake_analysis(zero_batches=10, identical=44, null_ok=True)
    verdict = score218.prediction_218_5(trimmed, good, good)
    assert verdict['verdict'] == 'not read' and '14' in verdict['reason']

    gated = fake_analysis(gate=False)
    verdict = score218.prediction_218_5(gated, good, good)
    assert verdict['verdict'] == 'not read' and 'gate' in verdict['reason']


def test_218_7_and_218_8_copy_the_registered_predictions():
    instruct = fake_analysis(stats={
        '5': {'moved': 6, 'led_counts': {'OpenAI': 4}, 'supported': 40},
        '10': None})
    sft = fake_analysis(gate=True)
    dpo = fake_analysis(gate=True)
    predictions = score218.compute_predictions(sft, dpo, instruct)
    assert predictions['218.7']['verdict'] == predictions['218.1']['verdict']
    assert predictions['218.8']['verdict'] == predictions['218.4']['verdict']
    assert predictions['218.7']['author'] == 'Vesper'
    assert predictions['218.1']['author'] == 'Journal Claude'
    assert predictions['218.1']['confidence'] == 'low (35%)'
    assert predictions['218.3']['confidence'] == 'medium (60%)'


# ---------------------------------------------------------------------------
# 218.6
# ---------------------------------------------------------------------------

def _stage_records(model, prompts, maker_per_prompt):
    """One read-rung layer: `maker_per_prompt` answers per prompt."""
    records = {}
    directions = all_directions()
    for prompt in range(prompts):
        answers = maker_per_prompt(prompt)
        for index, direction_id in enumerate(directions):
            text = answers[index] if index < len(answers) else NO_MAKER
            row = record(model, 5, 0.5, direction_id, prompt, text, row=index)
            records[score218.row_key(row)] = row
        zero = record(model, 5, 0.5, 'zero', prompt, SELF)
        records[score218.row_key(zero)] = zero
    return records


def test_218_6_stage_shift_held_when_the_shift_is_clear():
    sft_records = _stage_records('SFT', 30, lambda p: [OPENAI, NO_MAKER])
    dpo_records = _stage_records('DPO', 30, lambda p: [ALIBABA, OPENAI])
    sft = fake_analysis(gate=True, eligible=list(range(30)),
                        records=sft_records)
    dpo = fake_analysis(gate=True, eligible=list(range(30)),
                        records=dpo_records)
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'held', verdict
    assert verdict['detail']['sft']['other_maker'] == 30
    assert verdict['detail']['dpo']['other_maker'] == 60
    assert verdict['detail']['intersection_prompts'] == 30
    assert verdict['detail']['bootstrap']['excludes_zero'] is True


def test_218_6_fails_without_a_shift():
    sft_records = _stage_records('SFT', 30, lambda p: [OPENAI])
    dpo_records = _stage_records('DPO', 30, lambda p: [OPENAI])
    sft = fake_analysis(gate=True, eligible=list(range(30)),
                        records=sft_records)
    dpo = fake_analysis(gate=True, eligible=list(range(30)),
                        records=dpo_records)
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'failed'


def test_218_6_not_read_for_small_intersection_or_few_other_makers():
    sft_records = _stage_records('SFT', 20, lambda p: [OPENAI])
    dpo_records = _stage_records('DPO', 20, lambda p: [ALIBABA])
    sft = fake_analysis(gate=True, eligible=list(range(20)),
                        records=sft_records)
    dpo = fake_analysis(gate=True, eligible=list(range(20)),
                        records=dpo_records)
    assert score218.prediction_218_6(sft, dpo, None)['verdict'] == 'not read'

    # 30 eligible prompts, but only 10 prompts carry an other_maker answer.
    sft_records = _stage_records(
        'SFT', 30, lambda p: [OPENAI] if p < 10 else [])
    dpo_records = _stage_records(
        'DPO', 30, lambda p: [ALIBABA] if p < 10 else [])
    sft = fake_analysis(gate=True, eligible=list(range(30)),
                        records=sft_records)
    dpo = fake_analysis(gate=True, eligible=list(range(30)),
                        records=dpo_records)
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'not read'
    assert '20' in verdict['reason']


# ---------------------------------------------------------------------------
# Zero-row identity (218.5's first half)
# ---------------------------------------------------------------------------

def test_zero_row_identity_counts_prompts_identical_across_batches():
    records = {}
    for prompt in range(44):
        for layer in (5, 10):
            for dose in score218.LADDER:
                tokens = [7, 8, 9]
                if prompt == 0 and dose == 1.5:
                    tokens = [7, 8, 10]          # one prompt differs
                row = record('Instruct', layer, dose, 'zero', prompt, SELF)
                row['token_ids'] = tokens
                records[score218.row_key(row)] = row
    result = score218.zero_row_identity(
        records, meta([5, 10], score218.LADDER, 44))
    assert result['batches_per_prompt'] == 14
    assert result['identical_prompts'] == 43
    assert result['ok'] is True

    for prompt in (1, 2):
        for layer in (5, 10):
            row = records[('Instruct', layer, 0.25, prompt, 'zero')]
            row['token_ids'] = [1, 1, 1]
    result = score218.zero_row_identity(
        records, meta([5, 10], score218.LADDER, 44))
    assert result['identical_prompts'] == 41
    assert result['ok'] is False


# ---------------------------------------------------------------------------
# End to end on a written run dir
# ---------------------------------------------------------------------------

def _write_run_dir(root: Path, records, run_meta, model='Instruct'):
    directory = root / model
    directory.mkdir(parents=True)
    with (directory / 'answers.jsonl').open('w', encoding='utf-8') as handle:
        for row in records.values():
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (directory / 'run.json').write_text(json.dumps(run_meta), encoding='utf-8')


def test_end_to_end_on_a_fixture_run_dir(tmp_path):
    records = {}
    for prompt in range(44):
        for dose in (0.25, 0.5):
            for index, direction_id in enumerate(all_directions()):
                if dose == 0.25 and index < 6 and prompt < 3:
                    text = OPENAI
                elif dose == 0.5 and index < 10:
                    text = BROKEN
                else:
                    text = NO_MAKER
                row = record('Instruct', 5, dose, direction_id, prompt, text,
                             row=index)
                records[score218.row_key(row)] = row
            zero = record('Instruct', 5, dose, 'zero', prompt, SELF)
            records[score218.row_key(zero)] = zero
    run_meta = meta([5], [0.25, 0.5], 44, status='complete',
                    rows_written=len(records), rows_planned=len(records),
                    null={'all_ok': True,
                          'hooked_zero_vs_start': {'all_ok': True},
                          'end': {'vs_start': {'all_ok': True},
                                  'vs_hooked_zero': {'all_ok': True}}})
    _write_run_dir(tmp_path, records, run_meta)

    run = score218.load_run_dir(tmp_path)
    score, sample = score218.build_score(run)
    assert score['gold']['ok'] is True
    assert score['gold']['matched'] == 87
    instruct = score['models']['Instruct']
    assert instruct['gate']['eligible'] == 44
    assert instruct['read_rungs']['5'] == 0.25
    assert instruct['read_rung_stats']['5']['moved'] == 6
    assert score['predictions']['218.1']['verdict'] == 'held'
    assert score['predictions']['218.2']['verdict'] == 'not read'   # no SFT
    assert score['predictions']['218.5']['verdict'] == 'not read'   # 2 batches
    assert len(sample) == 18        # 6 directions x 3 prompts, all at the rung
    assert all(row['model'] == 'Instruct' for row in sample)

    code = score218.main(['--run-dir', str(tmp_path)])
    assert code == 0
    written = json.loads((tmp_path / 'score218.json').read_text(
        encoding='utf-8'))
    assert written['models']['Instruct']['read_rungs']['5'] == 0.25
    audit_lines = (tmp_path / 'audit218.jsonl').read_text(
        encoding='utf-8').splitlines()
    assert len(audit_lines) == 18
    assert json.loads(audit_lines[0])['audit_id'].startswith('Instruct-')


# ---------------------------------------------------------------------------
# Deliberate break of a COPY (never the tracked file)
# ---------------------------------------------------------------------------

def _check_moved_threshold(module):
    """Two other_maker answers must not be enough to call a direction moved."""
    records = {}
    for prompt in range(10):
        for direction_id, count in (('seed0+', 2), ('seed0-', 3)):
            text = OPENAI if prompt < count else NO_MAKER
            row = record('Instruct', 5, 0.25, direction_id, prompt, text)
            records[module.row_key(row)] = row
    stats = module.direction_stats(records, list(range(10)), 5, 0.25, {})
    assert stats['moved'] == 1, (
        f'the moved threshold must require 3 other_maker answers, '
        f'got moved={stats["moved"]}')
    assert stats['per_direction']['seed0+']['moved'] is False
    assert stats['per_direction']['seed0-']['moved'] is True


def test_deliberate_break_of_a_copy_turns_the_moved_check_red(tmp_path):
    source = Path(score218.__file__).read_text(encoding='utf-8')
    needle = 'MOVED_MIN = 3'
    assert needle in source, 'the break target moved; update this test'
    broken_source = source.replace(needle, 'MOVED_MIN = 2')
    broken_path = tmp_path / 'score218_broken.py'
    broken_path.write_text(broken_source, encoding='utf-8')

    spec = importlib.util.spec_from_file_location('score218_broken_for_test',
                                                  broken_path)
    broken = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broken)

    with pytest.raises(AssertionError):
        _check_moved_threshold(broken)
    _check_moved_threshold(score218)


def test_registered_thresholds_match_the_design():
    """Added by Claude at review (2026-11-24): a copy with READ_RUNG_MIN = 35
    passed all 24 tests, because the read-rung tests pass the threshold in as an
    argument. Every number below is quoted from DESIGN.md (registered SHA
    7d0c6509...), so changing one without an amendment turns this red."""
    assert score218.GATE_MIN_ELIGIBLE == 30            # gate: 30 of 44 eligible
    assert score218.DESIGN_N_PROMPTS == 44
    assert score218.READ_RUNG_MIN == 36                 # 36 of 40 supported
    assert score218.READ_RUNG_MIN_PILOT == 9            # pilot: 9 of 10
    assert score218.N_DIRECTIONS == 40 and score218.N_DIRECTIONS_PILOT == 10
    assert score218.MOVED_MIN == 3                      # at least 3 other_maker
    assert score218.BROKEN_FRACTION == 0.10             # broken on at most 10%
    assert score218.FULL_ZERO_BATCHES == 14             # 218.5: all 14 batches
    assert score218.BOOTSTRAP_DRAWS == 10_000 and score218.BOOTSTRAP_SEED == 218
    assert score218.INTERSECTION_MIN == 25              # 218.6 not read below 25
    assert score218.OTHER_MAKER_MIN == 20               # 218.6 not read below 20
    assert score218.SHIFT_MIN == 0.10                   # 218.6: at least 10 points
    assert set(score218.SHIFT_FAMILIES) == {'Alibaba', 'Microsoft'}
    assert tuple(score218.LADDER) == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    assert tuple(score218.PILOT_DOSES) == (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5)
