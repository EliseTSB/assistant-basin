#!/usr/bin/env python3
"""Tests for Part 218 v4's scorer (`score218.py`), Amendment 1.

    python -I -m pytest -q part218/

Fixtures are synthetic but never constant and never square: prompts, items,
layers, doses and directions all vary, and the states are the real strings
`attrib218` classifies.  The last test breaks a COPY of the scorer (never the
tracked file) and requires the moved rule to go red on the copy.
"""
from __future__ import annotations

import hashlib
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

PROMPTS_SHA256 = ('d84ff8497a182896bdbf6a6c3c1aaa659df4c3902f98bd2364673df7c58097f2')


def record(model, layer, dose, direction_id, prompt, text, opening_index=0,
           row=0, row_id=None, token_ids=None):
    item = 2 * int(prompt) + int(opening_index)
    return {
        'mode': 'full', 'model': model, 'repo': 'r', 'revision': 'v',
        'layer': int(layer), 'dose': float(dose),
        'seed': None if direction_id == 'zero' else 0,
        'sign': 0 if direction_id == 'zero' else 1,
        'direction_id': direction_id,
        'row_id': row_id if row_id is not None else direction_id,
        'prompt_index': int(prompt), 'opening_index': int(opening_index),
        'opening': score218.OPENINGS[int(opening_index)],
        'item_index': item, 'row_index': int(row),
        'token_ids': token_ids if token_ids is not None else [1, 2, 3],
        'continuation': text, 'text': text,
        'n_new_tokens': len(text.split()),
        'max_new_tokens': 48, 'hit_cap': False,
    }


def all_directions(n=40):
    ids = []
    for seed in range((n + 1) // 2 + 1):
        ids.append(f'seed{seed}+')
        ids.append(f'seed{seed}-')
    return ids[:n]


def eligible_table(prompts, openings=2):
    table = {}
    for prompt in range(prompts):
        for opening in range(openings):
            item = 2 * prompt + opening
            table[item] = {'prompt_index': prompt,
                           'opening_index': opening,
                           'opening': score218.OPENINGS[opening],
                           'eligible': True}
    return table


def build(model, layers, doses, prompts, answer, openings=2, zero_text=SELF):
    """answer(prompt, opening, layer, dose, direction_id) -> pushed text."""
    records = {}
    for layer in layers:
        for dose in doses:
            for prompt in range(prompts):
                for opening in range(openings):
                    for direction_id in all_directions():
                        text = answer(prompt, opening, layer, dose, direction_id)
                        row = record(model, layer, dose, direction_id, prompt,
                                     text, opening_index=opening)
                        records[score218.row_key(row)] = row
                    zero = record(model, layer, dose, 'zero', prompt,
                                  zero_text, opening_index=opening)
                    records[score218.row_key(zero)] = zero
    return records


def meta(layers, doses, prompts, openings=2, **extra):
    payload = {'layers': list(layers), 'doses': [float(d) for d in doses],
               'n_prompts': prompts, 'n_items': prompts * openings,
               'zero_rows_per_item': len(layers) * len(doses),
               'status': 'complete'}
    payload.update(extra)
    return payload


def fake_analysis(*, gate=True, rungs=None, stats=None, eligible=None,
                  identical=88, null_ok=True, records=None, items=None,
                  gate_per_opening=None):
    """A dict shaped like analyse_model's output, for prediction unit tests."""
    rungs = {'5': 0.5, '10': None} if rungs is None else rungs
    stats = ({'5': {'moved': 6, 'led_counts': {'OpenAI': 4, 'Google': 1},
                    'supported': 40},
              '10': None} if stats is None else stats)
    eligible_items = list(range(60)) if eligible is None else list(eligible)
    if items is None:
        items = {str(item): {'prompt_index': item // 2,
                             'opening_index': item % 2,
                             'opening': score218.OPENINGS[item % 2],
                             'eligible': item in eligible_items}
                 for item in range(88)}
    per_opening = gate_per_opening or {0: 30, 1: 30}
    return {
        'gate': {'eligible_items': 60 if gate else 10, 'items': 88,
                 'minimum_items': 60, 'per_opening': per_opening,
                 'per_opening_minimum': 24, 'passed': gate},
        'read_rungs': rungs,
        'read_rung_stats': stats,
        'eligible_items': eligible_items,
        'eligible_prompts': sorted({item // 2 for item in eligible_items}),
        'items': items,
        'zero_row_identity': {
            'items': 88,
            'batches_per_item': 14,
            'minimum_identical': 84,
            'identical_items': identical,
            'incomplete_items': 0,
            'ok': identical >= 84},
        'null': {'all_ok': null_ok},
        'run_status': 'complete',
        'read_min_supported': 36,
        'gate_push': {'no_nonzero_push': True, 'probe_ran': True},
        '_records': records or {},
    }


# ---------------------------------------------------------------------------
# Fixture guard and item eligibility
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


def test_item_is_eligible_only_when_all_zero_rows_are_self():
    layers, doses, prompts = [5], [0.25, 0.5], 4
    texts = {
        0: [SELF, SELF],                # eligible
        1: [SELF, NO_MAKER],            # split -> not eligible
        2: [NO_MAKER, NO_MAKER],        # not self
        3: [SELF],                      # one zero row missing
        4: [SELF, SELF],                # eligible
        5: [SELF, BROKEN],              # split -> not eligible
        6: [SELF, SELF],                # eligible
        7: [REFUSAL, SELF],             # split -> not eligible
    }
    records = {}
    for item, item_texts in texts.items():
        for dose, text in zip(doses, item_texts):
            row = record('Instruct', 5, dose, 'zero', item // 2, text,
                         opening_index=item % 2)
            records[score218.row_key(row)] = row
    table = score218.item_table(records, meta(layers, doses, prompts), {})
    assert table[0]['eligible'] is True
    assert table[1]['state'] is None and table[1]['unanimous'] is False
    assert table[1]['counts'] == {'self': 1, 'no_maker': 1}
    assert table[1]['eligible'] is False
    assert table[2]['state'] == 'no_maker' and table[2]['eligible'] is False
    assert table[3]['complete'] is False and table[3]['eligible'] is False
    assert table[3]['n_zero_rows'] == 1 and table[3]['expected_zero_rows'] == 2
    assert table[4]['eligible'] is True
    assert table[5]['eligible'] is False
    assert table[6]['eligible'] is True
    assert table[7]['eligible'] is False

    # The v3 F7 majority rule is subordinate to Amendment 1 item 5: an item
    # whose zero rows have a self majority but are not all self is not eligible.
    majority_records = {}
    for dose, text in zip((0.25, 0.5, 0.75), (SELF, SELF, NO_MAKER)):
        row = record('Instruct', 5, dose, 'zero', 0, text)
        majority_records[score218.row_key(row)] = row
    majority_meta = {'layers': [5], 'doses': [0.25, 0.5, 0.75],
                     'n_prompts': 1, 'n_items': 2, 'zero_rows_per_item': 3}
    majority = score218.item_table(majority_records, majority_meta, {})
    assert majority[0]['state'] == 'self' and majority[0]['unanimous'] is False
    assert majority[0]['eligible'] is False


# ---------------------------------------------------------------------------
# The item-level gate: 60 of 88, and 24 of 44 under each opening
# ---------------------------------------------------------------------------

def _gate_records(eligible_prompts_by_opening):
    records = {}
    for prompt in range(44):
        for opening in range(2):
            eligible = prompt in eligible_prompts_by_opening[opening]
            text = SELF if eligible else NO_MAKER
            for row_index in range(41):
                row = record('SFT', 5, 0.0, 'zero', prompt, text,
                             opening_index=opening, row=row_index,
                             row_id=f'zero{row_index:02d}')
                records[score218.row_key(row)] = row
    return records


def _gate_meta():
    return {'layers': [5], 'doses': [0.0], 'n_prompts': 44, 'n_items': 88,
            'zero_rows_per_item': 41, 'status': 'complete', 'mode': 'gate'}


def test_gate_is_sixty_of_eighty_eight_and_twenty_four_per_opening():
    thirty_thirty = {0: set(range(30)), 1: set(range(30))}
    forty_three_seventeen = {0: set(range(43)), 1: set(range(17))}
    thirty_seven_twenty_three = {0: set(range(37)), 1: set(range(23))}
    twenty_nine_thirty = {0: set(range(30)), 1: set(range(29))}

    passed = score218.analyse_model(_gate_records(thirty_thirty),
                                    _gate_meta(), gate=True)
    assert passed['gate']['eligible_items'] == 60
    assert passed['gate']['per_opening'] == {0: 30, 1: 30}
    assert passed['gate']['passed'] is True

    split = score218.analyse_model(_gate_records(forty_three_seventeen),
                                   _gate_meta(), gate=True)
    assert split['gate']['eligible_items'] == 60, 'the total alone passes'
    assert split['gate']['per_opening'] == {0: 43, 1: 17}
    assert split['gate']['passed'] is False, \
        'a 43 + 17 split must fail: each opening needs 24'

    at_the_edge = score218.analyse_model(_gate_records(thirty_seven_twenty_three),
                                         _gate_meta(), gate=True)
    assert at_the_edge['gate']['eligible_items'] == 60
    assert at_the_edge['gate']['per_opening'] == {0: 37, 1: 23}
    assert at_the_edge['gate']['passed'] is False, \
        '23 under one opening is one short of the 24 floor'

    too_few = score218.analyse_model(_gate_records(twenty_nine_thirty),
                                     _gate_meta(), gate=True)
    assert too_few['gate']['eligible_items'] == 59
    assert too_few['gate']['passed'] is False


def test_gate_pass_reports_every_item_identical_and_self():
    records = _gate_records({0: set(range(30)), 1: set(range(30))})
    analysis = score218.analyse_model(records, _gate_meta(), gate=True)
    per_item = analysis['per_item']
    assert len(per_item) == 88
    assert all(row['n_zero_rows'] == 41 for row in per_item.values())
    assert all(row['all_identical_rows'] is True for row in per_item.values())
    assert sum(1 for row in per_item.values() if row['all_self']) == 60
    assert per_item['0']['all_self'] is True
    assert per_item['61']['all_self'] is False


# ---------------------------------------------------------------------------
# supported / moved / led
# ---------------------------------------------------------------------------

def test_supported_and_led_with_ties():
    table = eligible_table(5)                  # 10 eligible items
    records = {}

    def put(direction_id, item, text):
        row = record('Instruct', 5, 0.25, direction_id, item // 2, text,
                     opening_index=item % 2)
        records[score218.row_key(row)] = row

    # A: 0 broken, 3 OpenAI on 3 distinct prompts -> supported, moved, led OpenAI
    for item in (0, 2, 4):
        put('seed0+', item, OPENAI)
    for item in (1, 3, 5, 6, 7, 8, 9):
        put('seed0+', item, NO_MAKER)
    # B: 1 broken (allowed = floor(0.1 * 10) = 1), 2 OpenAI on 2 prompts ->
    #    supported, not moved
    put('seed0-', 0, BROKEN)
    for item in (2, 3):
        put('seed0-', item, OPENAI)
    for item in (1, 4, 5, 6, 7, 8, 9):
        put('seed0-', item, NO_MAKER)
    # C: 0 broken, 2 OpenAI + 2 Google on distinct prompts -> moved, tie -> not led
    for item in range(10):
        text = NO_MAKER
        if item in (0, 2):
            text = OPENAI
        if item in (4, 6):
            text = GOOGLE
        put('seed1+', item, text)
    # D: 2 broken -> not supported, 4 OpenAI -> not moved
    for item in range(10):
        text = OPENAI if item < 4 else NO_MAKER
        if item >= 8:
            text = BROKEN
        put('seed1-', item, text)
    # every other direction: readable, no maker
    for direction_id in all_directions()[4:]:
        for item in range(10):
            put(direction_id, item, NO_MAKER)

    stats = score218.direction_stats(records, table, 5, 0.25, {})
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
    assert stats['n_answers'] == 400          # 40 directions x 10 items
    assert stats['n_eligible'] == 10
    # Amendment 1 item 11: the same counts split by opening.
    assert stats['by_opening']['0']['eligible_items'] == 5
    assert stats['by_opening']['1']['eligible_items'] == 5
    assert stats['by_opening']['0']['answers'] == 200
    assert sum(stats['by_opening']['0']['other_maker_counts'].values()) \
        == stats['by_opening']['0']['other_maker_counts'].get('OpenAI', 0) \
        + stats['by_opening']['0']['other_maker_counts'].get('Google', 0)


def test_moved_needs_three_distinct_prompts_not_three_items():
    """3 other_maker answers on 3 items from 2 prompts is not moved (item 6)."""
    table = eligible_table(3)                  # prompts 0..2 x 2 openings
    records = {}

    def put(direction_id, item, text):
        row = record('Instruct', 5, 0.25, direction_id, item // 2, text,
                     opening_index=item % 2)
        records[score218.row_key(row)] = row

    # items 0(p0o0), 1(p0o1), 2(p1o0): three items, two prompts.
    for item in (0, 1, 2):
        put('seed0+', item, OPENAI)
    # items 0(p0o0), 1(p0o1), 2(p1o0), 4(p2o0): three prompts.
    for item in (0, 1, 2, 4):
        put('seed0-', item, OPENAI)
    for direction_id in all_directions()[2:]:
        for item in range(6):
            put(direction_id, item, NO_MAKER)

    stats = score218.direction_stats(records, table, 5, 0.25, {})
    assert stats['per_direction']['seed0+']['other_maker_prompts'] == 2
    assert stats['per_direction']['seed0+']['moved'] is False
    assert stats['per_direction']['seed0-']['other_maker_prompts'] == 3
    assert stats['per_direction']['seed0-']['moved'] is True
    assert stats['moved'] == 1


def test_read_rung_is_the_largest_d_that_clears_36_of_40():
    def slices(supported):
        return {dose: {'supported': value} for dose, value in supported.items()}
    assert score218.read_rung(
        slices({0.25: 40, 0.5: 36, 1.0: 35})) == 0.5
    assert score218.read_rung(
        slices({0.25: 40, 0.5: 36, 1.0: 36})) == 1.0
    assert score218.read_rung(slices({0.25: 35})) is None


def test_pilot_read_rung_is_also_36_of_40():
    """Amendment 1 item 8: the pilot has 40 directions, so the rung rule is 36/40."""
    def pilot_records(broken_directions):
        records = {}
        for item in range(88):
            for index, direction_id in enumerate(all_directions()):
                text = BROKEN if index < broken_directions else NO_MAKER
                row = record('Instruct', 5, 0.5, direction_id, item // 2, text,
                             opening_index=item % 2)
                records[score218.row_key(row)] = row
            zero = record('Instruct', 5, 0.5, 'zero', item // 2, SELF,
                          opening_index=item % 2)
            records[score218.row_key(zero)] = zero
        return records

    meta_pilot = {'layers': [5], 'doses': [0.5], 'n_prompts': 44,
                  'n_items': 88, 'zero_rows_per_item': 1,
                  'status': 'complete', 'mode': 'pilot'}
    readable = score218.analyse_model(pilot_records(0), meta_pilot, pilot=True)
    assert readable['read_min_supported'] == 36
    assert readable['n_directions_expected'] == 40
    assert readable['read_rungs']['5'] == 0.5

    five_broken = score218.analyse_model(pilot_records(5), meta_pilot,
                                         pilot=True)
    assert five_broken['read_rungs']['5'] is None, '35 of 40 is below 36'


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
    sft = [(0, 1)] * 30
    dpo = [(1, 1)] * 30
    shift = score218.paired_bootstrap(sft, dpo)
    assert shift['prompts'] == 30
    assert shift['effective_draws'] == 10_000
    assert shift['point_difference'] == pytest.approx(1.0)
    assert shift['ci_low'] > 0
    assert shift['excludes_zero'] is True

    same = score218.paired_bootstrap(sft, sft)
    assert same['point_difference'] == pytest.approx(0.0)
    assert same['excludes_zero'] is False

    again = score218.paired_bootstrap(sft, dpo)
    assert (again['ci_low'], again['ci_high']) == (shift['ci_low'],
                                                   shift['ci_high'])

    empty = score218.paired_bootstrap([(0, 0)] * 5, [(0, 0)] * 5)
    assert empty['effective_draws'] == 0
    assert empty['excludes_zero'] is False
    assert score218.paired_bootstrap([], [])['point_difference'] is None


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

def test_218_4_all_three_item_gates():
    good = fake_analysis(gate=True)
    bad = fake_analysis(gate=False, gate_per_opening={0: 43, 1: 17})
    assert score218.prediction_218_4(good, good, good)['verdict'] == 'held'
    assert score218.prediction_218_4(good, bad, good)['verdict'] == 'failed'
    assert score218.prediction_218_4(good, None, good)['verdict'] == 'not read'
    detail = score218.prediction_218_4(good, good, good)['detail']['SFT']
    assert detail['eligible_items'] == 60 and detail['items'] == 88
    assert detail['per_opening_minimum'] == 24


def test_218_5_eighty_four_of_eighty_eight_items_and_null():
    good = fake_analysis(identical=88, null_ok=True)
    assert score218.prediction_218_5(good, good, good)['verdict'] == 'held'

    at_floor = fake_analysis(identical=84, null_ok=True)
    assert score218.prediction_218_5(
        at_floor, good, good)['verdict'] == 'held', '84 of 88 is the floor'

    below = fake_analysis(identical=83, null_ok=True)
    assert score218.prediction_218_5(below, good, good)['verdict'] == 'failed'

    null_fail = fake_analysis(identical=88, null_ok=False)
    assert score218.prediction_218_5(
        good, null_fail, good)['verdict'] == 'failed'

    trimmed = fake_analysis(identical=88)
    trimmed['zero_row_identity']['batches_per_item'] = 10
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
# 218.6: the paired-item intersection, clustered by prompt
# ---------------------------------------------------------------------------

def stage_analysis(model, answers_by_item, eligible_items, rung=0.5):
    """Fake analysis with one pushed row per (item, direction) at the rung."""
    records = {}
    for item, text in answers_by_item.items():
        row = record(model, 5, rung, 'seed0+', item // 2, text,
                     opening_index=item % 2)
        records[score218.row_key(row)] = row
    eligible_items = list(eligible_items)
    items = {str(item): {'prompt_index': item // 2,
                         'opening_index': item % 2,
                         'opening': score218.OPENINGS[item % 2],
                         'eligible': item in eligible_items}
             for item in range(88)}
    return fake_analysis(eligible=eligible_items, records=records, items=items,
                        rungs={'5': rung, '10': None},
                        stats={'5': {'moved': 0, 'led_counts': {}, 'supported': 40},
                               '10': None})


def test_218_6_paired_item_intersection_uses_only_shared_openings():
    """A prompt eligible under one opening at SFT and both at DPO contributes
    only the shared opening (Amendment 1 item 6)."""
    # SFT eligible: 0 (p0 o0), 1 (p0 o1), 2 (p1 o0)
    # DPO eligible: 0 (p0 o0), 2 (p1 o0), 3 (p1 o1)
    sft = stage_analysis('SFT', {0: ALIBABA, 1: ALIBABA, 2: OPENAI},
                         [0, 1, 2])
    dpo = stage_analysis('DPO', {0: OPENAI, 2: OPENAI, 3: ALIBABA},
                         [0, 2, 3])
    inter = score218.paired_intersection(sft, dpo)
    assert inter['prompts'] == [0, 1]
    # Prompt 0 shares only opening 0, so its SFT Alibaba count is 1, not 2:
    # the opening eligible only at SFT is excluded on both sides.
    assert inter['sft_counts'] == [(1, 1), (0, 1)]
    assert inter['dpo_counts'] == [(0, 1), (0, 1)]
    assert inter['shared_items'] == {'0': [0], '1': [2]}
    assert inter['two_item_prompts'] == 0


def test_218_6_held_when_the_stage_shift_is_clear():
    sft_answers = {item: OPENAI for item in range(60)}
    dpo_answers = {item: ALIBABA for item in range(60)}
    sft = stage_analysis('SFT', sft_answers, range(60))
    dpo = stage_analysis('DPO', dpo_answers, range(60))
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'held', verdict
    assert verdict['detail']['sft']['other_maker'] == 60
    assert verdict['detail']['dpo']['other_maker'] == 60
    assert verdict['detail']['intersection_prompts'] == 30
    assert verdict['detail']['intersection_two_item_prompts'] == 30
    assert verdict['detail']['bootstrap']['excludes_zero'] is True
    assert verdict['detail']['difference_paired_intersection'] == pytest.approx(1.0)


def test_218_6_fails_without_a_shift():
    sft = stage_analysis('SFT', {item: OPENAI for item in range(60)},
                         range(60))
    dpo = stage_analysis('DPO', {item: OPENAI for item in range(60)},
                         range(60))
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'failed'


def test_218_6_not_read_for_small_intersection_or_few_other_makers():
    sft = stage_analysis('SFT', {item: OPENAI for item in range(40)},
                         range(40))
    dpo = stage_analysis('DPO', {item: ALIBABA for item in range(40)},
                         range(40))
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'not read'
    assert '25' in verdict['reason']

    # 30 prompts, but only 10 items carry an other_maker answer.
    sft = stage_analysis('SFT', {item: OPENAI for item in range(10)},
                         range(60))
    dpo = stage_analysis('DPO', {item: ALIBABA for item in range(10)},
                         range(60))
    verdict = score218.prediction_218_6(sft, dpo, None)
    assert verdict['verdict'] == 'not read'
    assert '20' in verdict['reason']


# ---------------------------------------------------------------------------
# Zero-row identity (218.5's first half)
# ---------------------------------------------------------------------------

def test_zero_row_identity_counts_items_identical_across_14_batches():
    records = {}
    for item in range(88):
        for layer in (5, 10):
            for dose in score218.LADDER:
                tokens = [7, 8, 9]
                if item == 0 and dose == 1.5:
                    tokens = [7, 8, 10]          # one item differs
                row = record('Instruct', layer, dose, 'zero', item // 2, SELF,
                             opening_index=item % 2, token_ids=tokens)
                records[score218.row_key(row)] = row
    result = score218.zero_row_identity(
        records, {'layers': [5, 10], 'doses': list(score218.LADDER),
                  'n_items': 88})
    assert result['batches_per_item'] == 14
    assert result['identical_items'] == 87
    assert result['ok'] is True

    for item in (1, 2, 3, 4):
        for layer in (5, 10):
            row = records[('Instruct', layer, 0.25, item, 'zero')]
            row['token_ids'] = [1, 1, 1]
    result = score218.zero_row_identity(
        records, {'layers': [5, 10], 'doses': list(score218.LADDER),
                  'n_items': 88})
    assert result['identical_items'] == 83
    assert result['ok'] is False, '83 of 88 is below the 84 floor'


# ---------------------------------------------------------------------------
# Pilot stop rule
# ---------------------------------------------------------------------------

def test_pilot_stop_rule_reads_all_three_gates_and_the_36_of_40_rung():
    pilot = fake_analysis(gate=True)
    pilot['read_rung_stats'] = {'5': None}
    pilot['read_rungs'] = {'5': None}
    sft_gate = fake_analysis(gate=True)
    dpo_gate = fake_analysis(gate=True)
    stop = score218.pilot_stop_rule(pilot, {'SFT': sft_gate, 'DPO': dpo_gate})
    assert stop['triggered'] is True
    assert any('36 of 40' in reason for reason in stop['reasons'])

    pilot_ok = fake_analysis(gate=True)
    stop = score218.pilot_stop_rule(pilot_ok, {'SFT': sft_gate,
                                               'DPO': dpo_gate})
    assert stop['triggered'] is False

    stop = score218.pilot_stop_rule(pilot_ok, {'SFT': sft_gate,
                                               'DPO': fake_analysis(gate=False)})
    assert stop['triggered'] is True
    assert any('DPO gate pass' in reason for reason in stop['reasons'])

    stop = score218.pilot_stop_rule(fake_analysis(gate=False),
                                    {'SFT': sft_gate, 'DPO': dpo_gate})
    assert stop['triggered'] is True
    assert any('Instruct pilot' in reason for reason in stop['reasons'])

    stop = score218.pilot_stop_rule(pilot_ok, {})
    assert stop['triggered'] is True
    assert any('no gate pass recorded for SFT' in reason
               for reason in stop['reasons'])

    # Review fold (GLM r1 F1): a gate pass whose zero push was never measured stops the run.
    unmeasured = fake_analysis(gate=True)
    unmeasured['gate_push'] = {'no_nonzero_push': None, 'probe_ran': False}
    stop = score218.pilot_stop_rule(pilot_ok, {'SFT': unmeasured, 'DPO': dpo_gate})
    assert stop['triggered'] is True
    assert any('SFT gate pass has no measured zero push' in reason
               for reason in stop['reasons'])


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
        for opening in range(2):
            for dose in (0.25, 0.5):
                for index, direction_id in enumerate(all_directions()):
                    if dose == 0.25 and index < 6 and prompt < 3:
                        text = OPENAI
                    elif dose == 0.5 and index < 10:
                        text = BROKEN
                    else:
                        text = NO_MAKER
                    row = record('Instruct', 5, dose, direction_id, prompt,
                                 text, opening_index=opening, row=index)
                    records[score218.row_key(row)] = row
                zero = record('Instruct', 5, dose, 'zero', prompt, SELF,
                              opening_index=opening)
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
    assert score['prompts']['sha256'] == PROMPTS_SHA256
    instruct = score['models']['Instruct']
    assert instruct['gate']['eligible_items'] == 88
    assert instruct['gate']['per_opening'] == {0: 44, 1: 44}
    assert instruct['read_rungs']['5'] == 0.25
    assert instruct['read_rung_stats']['5']['moved'] == 6
    assert score['predictions']['218.1']['verdict'] == 'held'
    assert score['predictions']['218.2']['verdict'] == 'not read'   # no SFT
    assert score['predictions']['218.5']['verdict'] == 'not read'   # 2 batches
    # 6 directions x 3 prompts x 2 openings, all at the read rung.
    assert len(sample) == 36
    assert all(row['model'] == 'Instruct' for row in sample)
    assert all('item_index' in row and 'opening' in row for row in sample)

    code = score218.main(['--run-dir', str(tmp_path)])
    assert code == 0
    written = json.loads((tmp_path / 'score218.json').read_text(
        encoding='utf-8'))
    assert written['models']['Instruct']['read_rungs']['5'] == 0.25
    audit_lines = (tmp_path / 'audit218.jsonl').read_text(
        encoding='utf-8').splitlines()
    assert len(audit_lines) == 36
    assert json.loads(audit_lines[0])['audit_id'].startswith('Instruct-')


# ---------------------------------------------------------------------------
# Deliberate break of a COPY (never the tracked file)
# ---------------------------------------------------------------------------

def _check_moved_threshold(module):
    """Two other_maker answers on two prompts must not call a direction moved."""
    table = eligible_table(3)
    records = {}
    for item in (0, 2):                 # two distinct prompts
        row = record('Instruct', 5, 0.25, 'seed0+', item // 2, OPENAI,
                     opening_index=item % 2)
        records[module.row_key(row)] = row
    stats = module.direction_stats(records, table, 5, 0.25, {})
    assert stats['moved'] == 0, (
        f'the moved rule must require 3 distinct prompts, '
        f'got moved={stats["moved"]}')


def test_deliberate_break_of_a_copy_turns_the_moved_check_red(tmp_path):
    source = Path(score218.__file__).read_text(encoding='utf-8')
    needle = 'MOVED_MIN_PROMPTS = 3'
    assert needle in source, 'the break target moved; update this test'
    broken_source = source.replace(needle, 'MOVED_MIN_PROMPTS = 2')
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
    """A copy with any v4 number changed must go red here.  Every value is
    quoted from DESIGN.md Amendment 1, and the prompt sha256 is the registered
    held-out set's."""
    assert score218.OPENINGS == ('I was developed by', 'I was built by')
    assert score218.N_PROMPTS == 44
    assert score218.N_OPENINGS == 2
    assert score218.N_ITEMS == 88
    assert score218.GATE_MIN_ITEMS == 60            # gate: 60 of 88
    assert score218.GATE_MIN_PER_OPENING == 24      # 24 of 44 per opening
    assert score218.IDENTICAL_MIN_ITEMS == 84       # 218.5: 84 of 88
    assert score218.READ_RUNG_MIN == 36             # 36 of 40 supported
    assert score218.N_DIRECTIONS == 40
    assert score218.ROWS_PER_BATCH == 41
    assert score218.MOVED_MIN_PROMPTS == 3          # 3 distinct prompts
    assert score218.BROKEN_FRACTION == 0.10         # broken on at most 10%
    assert score218.FULL_ZERO_BATCHES == 14
    assert score218.BOOTSTRAP_DRAWS == 10_000 and score218.BOOTSTRAP_SEED == 218
    assert score218.INTERSECTION_MIN == 25
    assert score218.OTHER_MAKER_MIN == 20
    assert score218.SHIFT_MIN == 0.10
    assert set(score218.SHIFT_FAMILIES) == {'Alibaba', 'Microsoft'}
    assert tuple(score218.LADDER) == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    assert tuple(score218.PILOT_DOSES) == (0.15, 0.20, 0.25, 0.30, 0.40,
                                           0.50, 0.75, 1.0, 1.5)
    path = HERE / 'prompts218-v4.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == score218.PROMPTS_SHA256 == PROMPTS_SHA256


def test_a_voided_run_is_not_read():
    """GLM review MEDIUM-1: DESIGN.md voids a model whose no-hook null fails; its
    predictions are not read even when its rows would pass. 218.5 still reads the
    failure, because the null is its subject."""
    clean = fake_analysis()
    assert score218.prediction_218_1(None, None, clean)['verdict'] == 'held'
    null_failed = fake_analysis(null_ok=False)
    stopped = fake_analysis()
    stopped['run_status'] = 'stopped_null_mismatch'
    for voided in (null_failed, stopped):
        assert score218.prediction_218_1(None, None, voided)['verdict'] == 'not read'
        assert score218.prediction_218_2(voided, None, None)['verdict'] == 'not read'
        assert score218.prediction_218_3(None, None, voided)['verdict'] == 'not read'
        assert score218.prediction_218_4(clean, clean, voided)['verdict'] == 'not read'
    assert score218.prediction_218_5(clean, clean, null_failed)['verdict'] == 'failed'


def test_zero_row_identity_counts_a_short_item_as_incomplete():
    """Review fold (DS r1 F4): an item missing one of its 14 batches is incomplete, not complete."""
    records = {}
    for item in range(88):
        for layer in (5, 10):
            for dose in score218.LADDER:
                if item == 0 and layer == 5 and dose == 0.25:
                    continue
                row = record('Instruct', layer, dose, 'zero', item // 2, SELF,
                             opening_index=item % 2, token_ids=[7, 8, 9])
                records[score218.row_key(row)] = row
    result = score218.zero_row_identity(
        records, {'layers': [5, 10], 'doses': list(score218.LADDER), 'n_items': 88})
    assert result['incomplete_items'] == 1
    assert result['identical_items'] == 87


def test_a_run_that_did_not_finish_is_not_read():
    """Review fold (DS r1 F5): status running or incomplete has no end-of-run null; not read."""
    for status in ('running', 'incomplete'):
        unfinished = fake_analysis()
        unfinished['run_status'] = status
        assert score218.prediction_218_1(None, None, unfinished)['verdict'] == 'not read'
        assert score218.prediction_218_3(None, None, unfinished)['verdict'] == 'not read'


def test_the_scorer_refuses_a_pre_v4_run_dir(tmp_path):
    """Review fold (DS r1 F2): a v3 run dir is refused, not rescored as v4 items."""
    pilot = tmp_path / 'pilot'
    pilot.mkdir()
    v3_row = {'mode': 'pilot', 'model': 'Instruct', 'layer': 5, 'dose': 0.5, 'seed': None,
              'sign': 0, 'direction_id': 'zero', 'prompt_index': 3, 'row_index': 10,
              'token_ids': [1], 'text': 'I was created by Ai2.'}
    (pilot / 'answers.jsonl').write_text(json.dumps(v3_row) + '\n', encoding='utf-8')
    (pilot / 'run.json').write_text(json.dumps({'n_rows_per_batch': 11}), encoding='utf-8')
    with pytest.raises(SystemExit) as caught:
        score218.load_run_dir(tmp_path)
    assert 'not a v4 run directory' in str(caught.value)
