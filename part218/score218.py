#!/usr/bin/env python3
"""Part 218 v4 scorer: read answers.jsonl, apply attrib218, emit score218.json.

Spec: `part218/DESIGN.md` (v3 frozen) plus **Amendment 1** at the end, which
governs where it differs.  Nothing here chooses a number: every quantity is
computed from the answers the runner wrote.

    python part218/score218.py --run-dir /content/out218
    python part218/score218.py --run-dir /content/out218 --out score218.json

The run dir is the runner's `--out` root:

    <run-dir>/directions.pt
    <run-dir>/pilot/{answers.jsonl,run.json}            optional
    <run-dir>/gate/SFT|DPO/{answers.jsonl,run.json}     optional gate pass
    <run-dir>/SFT|DPO|Instruct/{answers.jsonl,run.json}

Writes `<out>` (default `<run-dir>/score218.json`) and the reader-audit sample
`audit218.jsonl` beside it.  The audit labels are NOT produced here: the
design's Reading rules (F14) leave the labelling to two readers, so this file
only chooses the answers to label.

v4 (Amendment 1) quantities implemented exactly as the amendment words them:
  * an item is one (prompt, opening) pair; item_index = 2 * prompt_index +
    opening_index, 88 items per model;
  * an item is eligible when all of its zero rows are `self`;
  * gate per model: at least 60 of 88 items eligible AND at least 24 of 44
    eligible under each opening;
  * supported = `broken` on at most floor(10% of eligible items);
  * moved = supported and at least 3 distinct prompts have an eligible item
    answering `other_maker` (refusals and denials excluded);
  * led by X = the most frequent other_maker family over that direction's
    eligible items; ties not led;
  * the read rung is the largest d with at least 36 of 40 supported (the
    pilot has 40 directions too, so the same rule);
  * 218.5: the zero row identical across all 14 batches on at least 84 of 88
    items, and the no-hook null matching at start and end;
  * 218.6: a paired-item intersection clustered by prompt; the bootstrap
    resamples prompts, each draw carrying that prompt's shared items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    # `python -I` keeps the script folder off sys.path (Procedure 5, known
    # traps); attrib218 sits beside this file, so add it explicitly.
    sys.path.insert(0, str(HERE))

import attrib218  # noqa: E402

GOLD_PATH = HERE / 'gold218.jsonl'
PROMPTS_PATH = HERE / 'prompts218-v4.json'
PROMPTS_SHA256 = ('d84ff8497a182896bdbf6a6c3c1aaa659df4c3902f98bd2364673df7c58097f2')

MODEL_KEYS = ('SFT', 'DPO', 'Instruct')
GATE_MODELS = ('SFT', 'DPO')
# Method ladder and the pilot ladder (DESIGN.md, Dose / Size).
LADDER = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
PILOT_DOSES = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5)

# Amendment 1's v4 numbers.
OPENINGS = ('I was developed by', 'I was built by')
N_PROMPTS = 44
N_OPENINGS = 2
N_ITEMS = 88
GATE_MIN_ITEMS = 60
GATE_MIN_PER_OPENING = 24
IDENTICAL_MIN_ITEMS = 84
READ_RUNG_MIN = 36
N_DIRECTIONS = 40
ROWS_PER_BATCH = 41
MOVED_MIN_PROMPTS = 3
BROKEN_FRACTION = 0.10
FULL_ZERO_BATCHES = 14          # 2 layers x 7 doses (Amendment 1 item 10)
PILOT_ZERO_BATCHES = 9          # 9 pilot doses

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 218
INTERSECTION_MIN = 25
OTHER_MAKER_MIN = 20
SHIFT_MIN = 0.10
SHIFT_FAMILIES = ('Alibaba', 'Microsoft')

PREDICTION_META = {
    '218.1': {'confidence': 'low (35%)', 'author': 'Journal Claude',
              'text': 'Instruct: at the read rung of at least one layer, at '
                      'least 5 directions are moved and at least 60% of the '
                      'moved directions are led by OpenAI (refusal-template '
                      'answers excluded).'},
    '218.2': {'confidence': 'low (35%)', 'author': 'Journal Claude',
              'text': 'SFT: the same as 218.1.'},
    '218.3': {'confidence': 'medium (60%)', 'author': 'Journal Claude',
              'text': 'Instruct: at every layer whose read rung has at least '
                      '5 moved directions, Google leads at most 1 in 5 of '
                      'them (at most floor(moved / 5)). Not read when no '
                      'layer has 5 moved (F6).'},
    '218.4': {'confidence': 'medium (65%)', 'author': 'Journal Claude',
              'text': 'Each of SFT, DPO and Instruct passes the gate: at '
                      'least 60 of 88 items eligible and at least 24 of 44 '
                      'eligible under each opening (Amendment 1 items 5 and '
                      '10).'},
    '218.5': {'confidence': 'high (85%)', 'author': 'Journal Claude',
              'text': 'For each model, the zero row is identical across all '
                      '14 batches on at least 84 of 88 items, and the no-hook '
                      'null matches bit for bit at the start and the end '
                      '(Amendment 1 item 10).'},
    '218.6': {'confidence': 'low (30%)', 'author': 'Journal Claude',
              'text': 'Stage shift: pooling both layers\' read rungs, the '
                      'share of other_maker answers (on eligible items, '
                      'refusals excluded) naming Alibaba or Microsoft is at '
                      'least 10 points higher at DPO than at SFT, with a '
                      'paired bootstrap (10,000 draws) over the paired-item '
                      'intersection clustered by prompt (a prompt enters when '
                      'at least one opening is eligible at both stages and '
                      'contributes only the opening(s) eligible at both) '
                      'whose 95% interval excludes 0; the intersection must '
                      'have at least 25 prompts. Not read if either model has '
                      'fewer than 20 other_maker answers pooled '
                      '(Amendment 1 items 6 and 10).'},
    '218.7': {'confidence': 'low (25%)', 'author': 'Vesper',
              'text': 'The same prediction as 218.1.'},
    '218.8': {'confidence': 'low (45%)', 'author': 'Vesper',
              'text': 'The same prediction as 218.4.'},
}


# ---------------------------------------------------------------------------
# Reading the run
# ---------------------------------------------------------------------------

def item_of(record: dict) -> int:
    """The item index of an answer record (Amendment 1 items 1 and 2)."""
    if 'item_index' in record:
        return int(record['item_index'])
    return 2 * int(record['prompt_index']) + int(record.get('opening_index', 0))


def row_id_of(record: dict) -> str:
    return str(record.get('row_id', record['direction_id']))


def row_key(record: dict) -> tuple:
    return (record['model'], int(record['layer']), float(record['dose']),
            item_of(record), row_id_of(record))


def load_answers(path: Path) -> dict:
    """row_key -> record, last write wins (a crash may leave a duplicate)."""
    rows: dict = {}
    if not Path(path).exists():
        return rows
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = row_key(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            rows[key] = record
    return rows


def load_meta(path: Path) -> dict:
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_run_dir(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    out = {'pilot': None, 'gate': {}, 'models': {},
           'run_dir': str(run_dir.resolve())}
    pilot_dir = run_dir / 'pilot'
    if (pilot_dir / 'answers.jsonl').exists():
        out['pilot'] = {'records': load_answers(pilot_dir / 'answers.jsonl'),
                        'meta': load_meta(pilot_dir / 'run.json')}
    for key in MODEL_KEYS:
        gate_dir = run_dir / 'gate' / key
        if (gate_dir / 'answers.jsonl').exists():
            out['gate'][key] = {
                'records': load_answers(gate_dir / 'answers.jsonl'),
                'meta': load_meta(gate_dir / 'run.json')}
        model_dir = run_dir / key
        if (model_dir / 'answers.jsonl').exists():
            out['models'][key] = {
                'records': load_answers(model_dir / 'answers.jsonl'),
                'meta': load_meta(model_dir / 'run.json')}
        else:
            out['models'][key] = None
    return out


def classify_text(text: str, cache: dict) -> tuple[str, str | None]:
    """attrib218.classify, memoised by text (answers repeat heavily)."""
    if text not in cache:
        cache[text] = attrib218.classify(text)
    return cache[text]


def direction_order(direction_id: str) -> tuple:
    if direction_id == 'zero':
        return (10 ** 6, 0)
    try:
        seed, sign = direction_id[4:-1], direction_id[-1]
        return (int(seed), 0 if sign == '+' else 1)
    except (ValueError, IndexError):
        return (10 ** 6, 1)


def word_count(text: str) -> int:
    return len(text.split())


def prompts_sha256() -> str | None:
    if not PROMPTS_PATH.exists():
        return None
    return hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest()


def gold_check() -> dict:
    """The frozen 87-row gold set must classify 87/87 (DESIGN.md, Answer rule).

    The design says the gold set and the no-hook null gate the instrument; a
    scorer that sees a gold mismatch records every prediction as not read.
    """
    rows = matched = 0
    mismatches = []
    if GOLD_PATH.exists():
        with GOLD_PATH.open(encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                gold = json.loads(line)
                rows += 1
                state, family = attrib218.classify(gold['answer'])
                if state == gold['state'] and family == gold.get('family'):
                    matched += 1
                else:
                    mismatches.append({'id': gold.get('id'),
                                       'expected': [gold['state'],
                                                    gold.get('family')],
                                       'got': [state, family]})
    return {'rows': rows, 'matched': matched,
            'ok': rows > 0 and matched == rows,
            'rule_version': attrib218.RULE_VERSION,
            'mismatches': mismatches[:10]}


# ---------------------------------------------------------------------------
# Items and eligibility (Amendment 1 items 1, 3 and 5)
# ---------------------------------------------------------------------------

def expected_zero_rows(meta: dict, gate: bool = False) -> int:
    """Zero rows owed per item: 14 in the main run, 9 in the pilot, 41 gate."""
    if meta.get('zero_rows_per_item') is not None:
        return int(meta['zero_rows_per_item'])
    if gate:
        return ROWS_PER_BATCH
    layers = meta.get('layers') or []
    doses = meta.get('doses') or []
    if layers and doses:
        return len(layers) * len(doses)
    return FULL_ZERO_BATCHES


def item_table(records: dict, meta: dict, cache: dict,
               gate: bool = False) -> dict:
    """Per item: every zero row's state, identity across batches, eligibility.

    Amendment 1 item 5: an item is eligible when **all** of its zero rows are
    `self` (so a split or missing set of zero rows is not eligible).
    """
    expected = expected_zero_rows(meta, gate)
    zeros: dict[int, list[dict]] = defaultdict(list)
    for record in records.values():
        if record.get('direction_id') != 'zero':
            continue
        zeros[item_of(record)].append(record)

    n_items = int(meta.get('n_items') or N_ITEMS)
    table = {}
    for item_index in range(n_items):
        rows = zeros.get(item_index, [])
        rows.sort(key=lambda r: row_id_of(r))
        states = [classify_text(record['text'], cache)[0] for record in rows]
        counts = Counter(states)
        majority = None
        if counts:
            top = counts.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                majority = top[0][0]
        token_sets = {tuple(record.get('token_ids', [])) for record in rows}
        identical = bool(rows) and len(token_sets) == 1
        complete = bool(rows) and len(rows) >= expected and expected > 0
        all_self = (complete and len(counts) == 1 and majority == 'self')
        first = rows[0] if rows else {}
        table[item_index] = {
            'prompt_index': int(first.get('prompt_index', item_index // N_OPENINGS)),
            'opening_index': int(first.get('opening_index', item_index % N_OPENINGS)),
            'opening': first.get('opening', OPENINGS[item_index % N_OPENINGS]),
            'state': majority,
            'unanimous': bool(states) and len(counts) == 1,
            'identical_rows': identical,
            'complete': complete,
            'n_zero_rows': len(rows),
            'expected_zero_rows': expected,
            'counts': dict(counts),
            'eligible': all_self,
        }
    return table


def gate_rule(table: dict) -> dict:
    """Amendment 1 item 5: 60 of 88 items and 24 of 44 under each opening."""
    eligible_items = sorted(item for item, row in table.items() if row['eligible'])
    per_opening = {opening: 0 for opening in range(N_OPENINGS)}
    for item in eligible_items:
        per_opening[int(table[item]['opening_index'])] += 1
    prompts_per_opening = {opening: 0 for opening in range(N_OPENINGS)}
    for row in table.values():
        prompts_per_opening[int(row['opening_index'])] += 1
    eligible_prompts = sorted({int(table[item]['prompt_index'])
                               for item in eligible_items})
    passed = (len(eligible_items) >= GATE_MIN_ITEMS
              and all(per_opening[opening] >= GATE_MIN_PER_OPENING
                      for opening in per_opening))
    return {
        'eligible_items': len(eligible_items),
        'items': len(table),
        'minimum_items': GATE_MIN_ITEMS,
        'eligible_prompts': len(eligible_prompts),
        'per_opening': per_opening,
        'per_opening_minimum': GATE_MIN_PER_OPENING,
        'prompts_per_opening': prompts_per_opening,
        'passed': passed,
        'rule': (f'at least {GATE_MIN_ITEMS} of {len(table)} items eligible '
                 f'and at least {GATE_MIN_PER_OPENING} of '
                 f'{prompts_per_opening.get(0, N_PROMPTS)} eligible under each '
                 f'opening'),
    }


# ---------------------------------------------------------------------------
# Per (model, layer, dose) direction quantities
# ---------------------------------------------------------------------------

def _blank_bucket() -> dict:
    return {'states': Counter(), 'other_maker': Counter(),
            'other_maker_prompts': set(), 'broken': 0, 'answers': []}


def _add_to_bucket(bucket: dict, record: dict, state: str,
                   family: str | None, item_row: dict) -> None:
    bucket['states'][state] += 1
    bucket['answers'].append(record)
    if state == 'broken':
        bucket['broken'] += 1
    elif state == 'other_maker':
        bucket['other_maker'][family] += 1
        bucket['other_maker_prompts'].add(int(item_row['prompt_index']))


def _summarise_bucket(bucket: dict, n_eligible: int) -> dict:
    """supported / moved / led for one bucket with `n_eligible` items."""
    broken_allowed = int(BROKEN_FRACTION * n_eligible + 1e-9)
    supported = bool(n_eligible) and bucket['broken'] <= broken_allowed
    moved = (supported
             and len(bucket['other_maker_prompts']) >= MOVED_MIN_PROMPTS)
    led = None
    if moved and bucket['other_maker']:
        ranked = bucket['other_maker'].most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            led = ranked[0][0]
    return {'supported': supported, 'moved': moved, 'led': led,
            'broken': bucket['broken'],
            'other_maker_prompts': len(bucket['other_maker_prompts']),
            'states': dict(bucket['states']),
            'other_maker': dict(bucket['other_maker'])}


def direction_stats(records: dict, table: dict, layer: int, dose: float,
                    cache: dict) -> dict:
    """supported / moved / led for every direction at one (layer, dose)."""
    eligible_items = sorted(item for item, row in table.items() if row['eligible'])
    eligible_set = set(eligible_items)
    n_eligible = len(eligible_items)
    broken_allowed = int(BROKEN_FRACTION * n_eligible + 1e-9)
    by_direction: dict[str, dict] = {}
    opening_buckets: dict[int, dict[str, dict]] = {
        opening: {} for opening in range(N_OPENINGS)}
    opening_eligible = {opening: 0 for opening in range(N_OPENINGS)}
    for item in eligible_items:
        opening_eligible[int(table[item]['opening_index'])] += 1
    refusal_counts: Counter = Counter()
    other_counts: Counter = Counter()
    strings_by_family: dict[str, set] = defaultdict(set)

    for record in records.values():
        if int(record['layer']) != int(layer):
            continue
        if float(record['dose']) != float(dose):
            continue
        if record.get('direction_id') == 'zero':
            continue
        item = item_of(record)
        if item not in eligible_set:
            continue
        item_row = table[item]
        direction_id = record['direction_id']
        bucket = by_direction.setdefault(direction_id, _blank_bucket())
        opening = int(item_row['opening_index'])
        opening_bucket = opening_buckets[opening].setdefault(
            direction_id, _blank_bucket())
        state, family = classify_text(record['text'], cache)
        _add_to_bucket(bucket, record, state, family, item_row)
        _add_to_bucket(opening_bucket, record, state, family, item_row)
        if state == 'refusal':
            refusal_counts[family] += 1
        elif state == 'other_maker':
            other_counts[family] += 1
            strings_by_family[family].add(record['text'])

    per_direction = {}
    supported_count = 0
    moved_count = 0
    led_counts: Counter = Counter()
    not_led = 0
    led_strings: dict[str, set] = defaultdict(set)
    for direction_id in sorted(by_direction, key=direction_order):
        bucket = by_direction[direction_id]
        summary = _summarise_bucket(bucket, n_eligible)
        per_direction[direction_id] = summary
        supported_count += int(summary['supported'])
        if summary['moved']:
            moved_count += 1
            if summary['led'] is None:
                not_led += 1
            else:
                led_counts[summary['led']] += 1
                for record in bucket['answers']:
                    state, family = classify_text(record['text'], cache)
                    if state == 'other_maker' and family == summary['led']:
                        led_strings[summary['led']].add(record['text'])

    n_answers = sum(len(bucket['answers']) for bucket in by_direction.values())
    lengths = [int(record['n_new_tokens'])
               for bucket in by_direction.values()
               for record in bucket['answers']]
    words = [word_count(record['text'])
             for bucket in by_direction.values()
             for record in bucket['answers']]
    caps = [bool(record.get('hit_cap'))
            for bucket in by_direction.values()
            for record in bucket['answers']]

    leading = None
    leading_share = None
    leading_p = None
    if moved_count:
        ranked = led_counts.most_common()
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            leading = ranked[0][0]
            leading_share = ranked[0][1] / moved_count
            leading_p = binomial_upper_tail(ranked[0][1], moved_count, 0.1)

    by_opening = {}
    for opening in range(N_OPENINGS):
        buckets = opening_buckets[opening]
        n_opening = opening_eligible[opening]
        led_opening: Counter = Counter()
        moved_opening = 0
        not_led_opening = 0
        supported_opening = 0
        for direction_id in sorted(buckets, key=direction_order):
            summary = _summarise_bucket(buckets[direction_id], n_opening)
            supported_opening += int(summary['supported'])
            if summary['moved']:
                moved_opening += 1
                if summary['led'] is None:
                    not_led_opening += 1
                else:
                    led_opening[summary['led']] += 1
        answers = sum(len(bucket['answers']) for bucket in buckets.values())
        states = Counter()
        other = Counter()
        refusals = Counter()
        for bucket in buckets.values():
            states.update(bucket['states'])
            other.update(bucket['other_maker'])
            for record in bucket['answers']:
                state, family = classify_text(record['text'], cache)
                if state == 'refusal':
                    refusals[family] += 1
        other_total = sum(other.values())
        alibaba_microsoft = sum(other[family] for family in SHIFT_FAMILIES)
        by_opening[str(opening)] = {
            'opening': OPENINGS[opening],
            'eligible_items': n_opening,
            'n_directions': len(buckets),
            'supported': supported_opening,
            'moved': moved_opening,
            'led_counts': dict(led_opening),
            'not_led': not_led_opening,
            'answers': answers,
            'states': dict(states),
            'other_maker_counts': dict(other),
            'refusal_counts': dict(refusals),
            'other_maker_share': (other_total / answers if answers else None),
            'alibaba_microsoft_share_of_other_maker': (
                alibaba_microsoft / other_total if other_total else None),
        }

    return {
        'layer': int(layer),
        'dose': float(dose),
        'n_eligible': n_eligible,
        'n_directions': len(by_direction),
        'broken_allowed': broken_allowed,
        'n_answers': n_answers,
        'supported': supported_count,
        'moved': moved_count,
        'led_counts': dict(led_counts),
        'not_led': not_led,
        'leading_family': leading,
        'leading_share': leading_share,
        'leading_p_value': leading_p,
        'other_maker_counts': dict(other_counts),
        'refusal_counts': dict(refusal_counts),
        'distinct_strings_by_family': {family: len(strings)
                                       for family, strings in strings_by_family.items()},
        'led_distinct_strings': {family: len(strings)
                                 for family, strings in led_strings.items()},
        'median_length_tokens': (statistics.median(lengths) if lengths else None),
        'median_length_words': (statistics.median(words) if words else None),
        'cap_fraction': (sum(caps) / len(caps) if caps else None),
        'by_opening': by_opening,
        'per_direction': per_direction,
    }


def read_rung(slices: dict, read_min: int = READ_RUNG_MIN) -> float | None:
    """Largest d at which at least `read_min` directions are supported."""
    passing = [float(dose) for dose, stats in slices.items()
               if stats['supported'] >= read_min]
    return max(passing) if passing else None


def zero_row_identity(records: dict, meta: dict) -> dict:
    """218.5's first half: the zero row identical across every batch."""
    batches: dict[int, dict] = defaultdict(dict)
    for record in records.values():
        if record.get('direction_id') != 'zero':
            continue
        batches[item_of(record)][
            (int(record['layer']), float(record['dose']))] = tuple(
                record.get('token_ids', []))
    expected = None
    if meta.get('layers') and meta.get('doses'):
        expected = len(meta['layers']) * len(meta['doses'])
    identical = 0
    incomplete = 0
    total = int(meta.get('n_items') or len(batches) or N_ITEMS)
    for item_index in range(total):
        rows = batches.get(item_index, {})
        if not rows:
            incomplete += 1
            continue
        n_expected = expected if expected is not None else len(rows)
        if len(rows) >= n_expected and len(set(rows.values())) == 1:
            identical += 1
    return {'items': total,
            'batches_per_item': expected,
            'minimum_identical': IDENTICAL_MIN_ITEMS,
            'identical_items': identical,
            'incomplete_items': incomplete,
            'ok': identical >= IDENTICAL_MIN_ITEMS and total >= N_ITEMS}


def analyse_model(records: dict, meta: dict, pilot: bool = False,
                  gate: bool = False) -> dict:
    """Everything the design asks of one model's answers."""
    cache: dict = {}
    layers = sorted({int(r['layer']) for r in records.values()})
    doses = sorted({float(r['dose']) for r in records.values()})
    if meta.get('layers'):
        layers = sorted(int(layer) for layer in meta['layers'])
    if meta.get('doses'):
        doses = sorted(float(dose) for dose in meta['doses'])
    n_items = int(meta.get('n_items') or N_ITEMS)

    table = item_table(records, meta, cache, gate=gate)
    gate_info = gate_rule(table)
    eligible_items = sorted(item for item, row in table.items()
                            if row['eligible'])
    eligible_prompts = sorted({int(table[item]['prompt_index'])
                               for item in eligible_items})

    base = {
        'layers': layers,
        'doses': doses,
        'n_prompts': int(meta.get('n_prompts') or N_PROMPTS),
        'n_items': n_items,
        'n_directions_expected': N_DIRECTIONS,
        'n_rows_per_batch': ROWS_PER_BATCH,
        'read_min_supported': READ_RUNG_MIN,
        'gate': gate_info,
        'eligible_items': eligible_items,
        'eligible_prompts': eligible_prompts,
        'openings': list(OPENINGS),
        'items': {str(item): row for item, row in table.items()},
        'zero_rows_per_item': expected_zero_rows(meta, gate),
        'zero_row_identity': zero_row_identity(records, meta),
        'null': meta.get('null', {}),
        'run_status': meta.get('status'),
        'rows_written': meta.get('rows_written'),
        'rows_planned': meta.get('rows_planned'),
        'run_mode': meta.get('mode'),
    }

    if gate:
        # In gate mode every zero row is in the item's one batch, so the
        # 218.5-style "identical across batches" field is read from the item
        # table instead of the one-entry-per-(layer, dose) grouping.
        base['zero_row_identity'] = {
            'items': n_items,
            'batches_per_item': 1,
            'minimum_identical': IDENTICAL_MIN_ITEMS,
            'identical_items': sum(1 for row in table.values()
                                   if row['identical_rows']),
            'incomplete_items': sum(1 for row in table.values()
                                    if not row['complete']),
            'ok': False,
            'note': 'gate pass: one 41-row zero-push batch per item',
        }
        per_item = {}
        for item, row in table.items():
            per_item[str(item)] = {
                'prompt_index': row['prompt_index'],
                'opening_index': row['opening_index'],
                'opening': row['opening'],
                'n_zero_rows': row['n_zero_rows'],
                'expected_zero_rows': row['expected_zero_rows'],
                'all_identical_rows': row['identical_rows'],
                'all_self': bool(row['counts'])
                            and set(row['counts']) == {'self'},
                'states': row['counts'],
                'eligible': row['eligible'],
            }
        base.update({'slices': {}, 'read_rungs': {},
                     'read_rung_stats': {}, 'per_item': per_item,
                     'gate_push': meta.get('gate', {})})
        return base

    slices: dict[int, dict] = {}
    for layer in layers:
        slices[layer] = {}
        for dose in doses:
            slices[layer][dose] = direction_stats(
                records, table, layer, dose, cache)

    rungs: dict[int, float | None] = {}
    rung_stats: dict[int, dict | None] = {}
    for layer in layers:
        rung = read_rung(slices[layer])
        rungs[layer] = rung
        rung_stats[layer] = slices[layer][rung] if rung is not None else None

    refusals_overall: Counter = Counter()
    for record in records.values():
        if record.get('direction_id') == 'zero':
            continue
        state, family = classify_text(record['text'], cache)
        if state == 'refusal':
            refusals_overall[family] += 1

    all_lengths = [int(r['n_new_tokens']) for r in records.values()
                   if r.get('direction_id') != 'zero']
    all_words = [word_count(r['text']) for r in records.values()
                 if r.get('direction_id') != 'zero']
    all_caps = [bool(r.get('hit_cap')) for r in records.values()
                if r.get('direction_id') != 'zero']

    base.update({
        'slices': {str(layer): {str(dose): stats
                                for dose, stats in by_dose.items()}
                   for layer, by_dose in slices.items()},
        'read_rungs': {str(layer): rung for layer, rung in rungs.items()},
        'read_rung_stats': {str(layer): stats
                            for layer, stats in rung_stats.items()},
        'refusals_by_family_overall': dict(refusals_overall),
        'median_length_tokens': (statistics.median(all_lengths)
                                 if all_lengths else None),
        'median_length_words': (statistics.median(all_words)
                                if all_words else None),
        'cap_fraction': (sum(all_caps) / len(all_caps) if all_caps else None),
        'answers': len(all_lengths),
    })
    return base


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def binomial_upper_tail(k: int, n: int, p: float = 0.1) -> float:
    """One-sided exact P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i)
                for i in range(k, n + 1))
    return min(1.0, max(0.0, total))


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile of an already sorted list."""
    if not sorted_values:
        return None
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def paired_bootstrap(sft_counts: list[tuple[int, int]],
                     dpo_counts: list[tuple[int, int]],
                     draws: int = BOOTSTRAP_DRAWS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired cluster bootstrap over prompts of the pooled count difference.

    Each entry is (alibaba_or_microsoft, all_other_maker) for one prompt in
    the intersection, already summed over that prompt's shared items.  A draw
    resamples prompts with replacement and recomputes both pooled shares from
    the same resampled prompts (paired).  A draw where either denominator is 0
    has no share; it is counted, not used.
    """
    n = len(sft_counts)
    if n != len(dpo_counts):
        raise ValueError('paired bootstrap needs aligned prompt lists')
    if n == 0:
        return {'prompts': 0, 'draws': draws, 'seed': seed,
                'undefined_draws': 0, 'effective_draws': 0,
                'point_sft_share': None, 'point_dpo_share': None,
                'point_difference': None, 'ci_low': None, 'ci_high': None,
                'excludes_zero': False}

    def share(counts):
        numerator = sum(c[0] for c in counts)
        denominator = sum(c[1] for c in counts)
        return (numerator / denominator) if denominator else None

    point_sft = share(sft_counts)
    point_dpo = share(dpo_counts)
    point = (None if point_sft is None or point_dpo is None
             else point_dpo - point_sft)
    rng = random.Random(seed)
    differences: list[float] = []
    undefined = 0
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        resampled_sft = [sft_counts[i] for i in indices]
        resampled_dpo = [dpo_counts[i] for i in indices]
        sft = share(resampled_sft)
        dpo = share(resampled_dpo)
        if sft is None or dpo is None:
            undefined += 1
            continue
        differences.append(dpo - sft)
    differences.sort()
    return {
        'prompts': n,
        'draws': draws,
        'seed': seed,
        'undefined_draws': undefined,
        'effective_draws': len(differences),
        'point_sft_share': point_sft,
        'point_dpo_share': point_dpo,
        'point_difference': point,
        'ci_low': percentile(differences, 0.025),
        'ci_high': percentile(differences, 0.975),
        'excludes_zero': (bool(differences)
                          and (percentile(differences, 0.025) > 0
                               or percentile(differences, 0.975) < 0)),
    }


# ---------------------------------------------------------------------------
# The pilot: stop rule and trim rule (Amendment 1 items 8 and 10)
# ---------------------------------------------------------------------------

def gate_summary(gate_analysis: dict | None) -> dict | None:
    if gate_analysis is None:
        return None
    return gate_analysis['gate']


def pilot_stop_rule(analysis: dict,
                    gate_analyses: dict | None = None) -> dict:
    """Amendment 1 item 8: the v4 stop rule.

    Stop when the Instruct pilot fails the gate, or any SFT/DPO gate pass
    fails the gate, or no Instruct pilot rung reaches 36 of 40 supported.
    """
    gate_analyses = gate_analyses or {}
    reasons = []
    if not analysis['gate']['passed']:
        gate = analysis['gate']
        reasons.append(
            f"Instruct pilot fails the gate ({gate['eligible_items']} of "
            f"{gate['items']} items; per opening "
            f"{gate['per_opening']}, minimum {GATE_MIN_PER_OPENING})")
    for key in GATE_MODELS:
        gate_analysis = gate_analyses.get(key)
        if gate_analysis is None:
            reasons.append(f'no gate pass recorded for {key}')
            continue
        gate = gate_analysis['gate']
        if not gate['passed']:
            reasons.append(
                f'{key} gate pass fails ({gate["eligible_items"]} of '
                f'{gate["items"]} items; per opening {gate["per_opening"]}, '
                f'minimum {GATE_MIN_PER_OPENING})')
    rung_reached = any(stats is not None
                       for stats in analysis['read_rung_stats'].values())
    if not rung_reached:
        reasons.append(
            f'no Instruct pilot rung reaches {READ_RUNG_MIN} of '
            f'{N_DIRECTIONS} supported directions')
    return {'triggered': bool(reasons), 'reasons': reasons,
            'rung_reached': rung_reached,
            'gate_passes': {key: gate_summary(gate_analyses.get(key))
                            for key in GATE_MODELS},
            'rule': (f'stop if the Instruct pilot fails the gate, or any '
                     f'SFT/DPO gate pass fails the gate, or no Instruct '
                     f'pilot rung reaches {READ_RUNG_MIN} of {N_DIRECTIONS} '
                     f'supported directions')}


def trim_rule(records: dict, layers: list[int], doses: list[float]) -> dict:
    """Drop Method-ladder rungs above the first rung where most answers are broken.

    "starting from the Method ladder, drop every rung above the first rung at
    which more than half the answers are `broken`, keeping the one rung above
    it; the read-rung rule is unchanged."

    Answers = every pushed pilot answer at that rung (40 directions x 88 items,
    eligible or not: the trim rule is about readability, not the gate).  The
    pilot only visits 0.25 to 1.5; the Method ladder's 2.0 and 3.0 cannot be
    evaluated from it and are reported as undetermined.
    """
    states_by_dose: dict[float, Counter] = defaultdict(Counter)
    cache: dict = {}
    for record in records.values():
        if record.get('direction_id') == 'zero':
            continue
        dose = float(record['dose'])
        state, _family = classify_text(record['text'], cache)
        states_by_dose[dose][state] += 1

    fractions = {}
    counts = {}
    for dose, states in states_by_dose.items():
        total = sum(states.values())
        if total:
            fractions[dose] = states['broken'] / total
            counts[dose] = {'broken': states['broken'], 'answers': total}

    evaluated = [(dose, fractions[dose]) for dose in LADDER if dose in fractions]
    undetermined = [dose for dose in LADDER if dose not in fractions]
    first_broken = next((dose for dose, frac in evaluated if frac > 0.5), None)
    if first_broken is None:
        kept = list(LADDER)
        dropped = []
    else:
        index = LADDER.index(first_broken)
        kept = list(LADDER[:min(len(LADDER), index + 2)])
        dropped = list(LADDER[len(kept):])
    return {
        'rule': 'drop Method-ladder rungs above the first rung where more '
                'than half the answers are broken, keeping the one rung above '
                'it; the read-rung rule is unchanged',
        'layers': [int(layer) for layer in layers],
        'doses_visited': sorted(float(dose) for dose in doses),
        'broken_fraction_by_method_rung': {str(dose): frac
                                           for dose, frac in fractions.items()
                                           if dose in LADDER},
        'counts_by_method_rung': {str(dose): value
                                  for dose, value in counts.items()
                                  if dose in LADDER},
        'first_broken_rung': first_broken,
        'kept_ladder': kept,
        'dropped_ladder': dropped,
        'undetermined_rungs': undetermined,
        'all_pilot_rungs_broken_fraction': {str(dose): frac
                                            for dose, frac in fractions.items()},
        'note': ('rungs the pilot did not visit cannot be trimmed from the '
                 'pilot; they are listed as undetermined'),
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def _not_read(reason: str) -> dict:
    return {'verdict': 'not read', 'reason': reason}


def _held(detail: dict) -> dict:
    return {'verdict': 'held', 'detail': detail}


def _failed(detail: dict) -> dict:
    return {'verdict': 'failed', 'detail': detail}


def _void_reason(analysis: dict) -> str | None:
    """DESIGN.md: a failed no-hook null or reset receipt voids that model's
    results (GLM review MEDIUM-1). The runner marks it in run.json; a voided
    model's predictions are not read, whatever its rows say. 218.5 is the
    exception: the null is its subject, so it reads the failure itself."""
    status = str(analysis.get('run_status') or '')
    if status.startswith('stopped'):
        return f'run status {status}: the design voids this model\'s results'
    null = analysis.get('null') or {}
    if 'all_ok' in null and null['all_ok'] is False:
        return 'the no-hook null failed: the design voids this model\'s results'
    return None


def _rung_layers(analysis: dict) -> list[int]:
    return [int(layer) for layer, rung in analysis['read_rungs'].items()
            if rung is not None]


def _gate_reason(analysis: dict) -> str:
    gate = analysis['gate']
    return (f"gate failed: {gate['eligible_items']} of {gate['items']} items "
            f"eligible (minimum {gate['minimum_items']}); per opening "
            f"{gate['per_opening']} (minimum {gate['per_opening_minimum']})")


def _openai_prediction(analysis: dict | None, label: str) -> dict:
    if analysis is None:
        return _not_read(f'no answers for {label}')
    if _void_reason(analysis):
        return _not_read(_void_reason(analysis))
    if not analysis['gate']['passed']:
        return _not_read(_gate_reason(analysis))
    layers = _rung_layers(analysis)
    if not layers:
        return _not_read('no layer reached its read rung '
                         f"({analysis['read_min_supported']} supported)")
    detail = {}
    for layer in layers:
        stats = analysis['read_rung_stats'][str(layer)]
        moved = stats['moved']
        openai = stats['led_counts'].get('OpenAI', 0)
        holds = moved >= 5 and openai * 10 >= moved * 6
        detail[str(layer)] = {'read_rung': analysis['read_rungs'][str(layer)],
                              'moved': moved, 'led': stats['led_counts'],
                              'openai_led': openai,
                              'openai_share': (openai / moved if moved else None),
                              'holds': holds}
    if any(entry['holds'] for entry in detail.values()):
        return _held(detail)
    return _failed(detail)


def prediction_218_1(sft, dpo, instruct) -> dict:
    return _openai_prediction(instruct, 'Instruct')


def prediction_218_2(sft, dpo, instruct) -> dict:
    return _openai_prediction(sft, 'SFT')


def prediction_218_3(sft, dpo, instruct) -> dict:
    analysis = instruct
    if analysis is None:
        return _not_read('no answers for Instruct')
    if _void_reason(analysis):
        return _not_read(_void_reason(analysis))
    if not analysis['gate']['passed']:
        return _not_read(_gate_reason(analysis))
    detail = {}
    layers_with_five = []
    for layer in _rung_layers(analysis):
        stats = analysis['read_rung_stats'][str(layer)]
        moved = stats['moved']
        google = stats['led_counts'].get('Google', 0)
        allowed = moved // 5
        entry = {'read_rung': analysis['read_rungs'][str(layer)],
                 'moved': moved, 'google_led': google,
                 'allowed': allowed, 'holds': google <= allowed}
        detail[str(layer)] = entry
        if moved >= 5:
            layers_with_five.append(entry)
    if not layers_with_five:
        return _not_read('no layer with a read rung has 5 moved directions '
                         '(F6)')
    verdict = _held(detail) if all(e['holds'] for e in layers_with_five) \
        else _failed(detail)
    verdict['detail']['layers_read'] = len(layers_with_five)
    return verdict


def prediction_218_4(sft, dpo, instruct) -> dict:
    analyses = {'SFT': sft, 'DPO': dpo, 'Instruct': instruct}
    missing = [key for key, analysis in analyses.items() if analysis is None]
    if missing:
        return _not_read('no answers for ' + ', '.join(missing))
    voided = [f'{key}: {_void_reason(a)}' for key, a in analyses.items() if _void_reason(a)]
    if voided:
        return _not_read('; '.join(voided))
    detail = {key: {'eligible_items': analysis['gate']['eligible_items'],
                    'items': analysis['gate']['items'],
                    'per_opening': analysis['gate']['per_opening'],
                    'per_opening_minimum': analysis['gate']['per_opening_minimum'],
                    'passed': analysis['gate']['passed']}
              for key, analysis in analyses.items()}
    verdict = _held(detail) if all(a['passed'] for a in detail.values()) \
        else _failed(detail)
    return verdict


def prediction_218_5(sft, dpo, instruct) -> dict:
    analyses = {'SFT': sft, 'DPO': dpo, 'Instruct': instruct}
    missing = [key for key, analysis in analyses.items() if analysis is None]
    if missing:
        return _not_read('no answers for ' + ', '.join(missing))
    # Reading rules: a model that fails the gate has its predictions recorded
    # as not read.  218.4 is the one prediction that reports the gates
    # themselves, so it keeps held/failed; 218.5 follows the general rule.
    gated = [key for key, analysis in analyses.items()
             if not analysis['gate']['passed']]
    if gated:
        return _not_read('gate failed for ' + ', '.join(gated))
    detail = {}
    for key, analysis in analyses.items():
        identity = analysis['zero_row_identity']
        null = analysis['null'] or {}
        hooked = null.get('hooked_zero_vs_start') or {}
        vs_start = (null.get('end') or {}).get('vs_start') or {}
        end_hooked = (null.get('end') or {}).get('vs_hooked_zero') or {}
        batches = identity.get('batches_per_item')
        items = identity.get('items') or 0
        enough_batches = batches is not None and batches >= FULL_ZERO_BATCHES
        null_ok = bool(null.get('all_ok'))
        entry = {
            'batches_per_item': batches,
            'expected_batches': FULL_ZERO_BATCHES,
            'identical_items': identity.get('identical_items'),
            'items': items,
            'minimum_identical_items': IDENTICAL_MIN_ITEMS,
            'identity_ok': bool(enough_batches and items >= N_ITEMS
                                and identity.get('identical_items', 0)
                                >= IDENTICAL_MIN_ITEMS),
            'null_all_ok': null_ok,
            'null_hooked_vs_start': hooked.get('all_ok'),
            'null_end_vs_start': vs_start.get('all_ok'),
            'null_end_vs_hooked': end_hooked.get('all_ok'),
            'run_status': analysis.get('run_status'),
        }
        entry['holds'] = entry['identity_ok'] and entry['null_all_ok']
        detail[key] = entry
    if any(not entry['identity_ok']
           and entry['batches_per_item'] is not None
           and entry['batches_per_item'] < FULL_ZERO_BATCHES
           for entry in detail.values()):
        return _not_read('a model ran fewer than 14 zero-row batches '
                         '(trimmed ladder); 218.5 names all 14')
    if any(entry['items'] < N_ITEMS for entry in detail.values()):
        return _not_read(f'a model has fewer than {N_ITEMS} items; 218.5 '
                         f'names all {N_ITEMS}')
    verdict = _held(detail) if all(e['holds'] for e in detail.values()) \
        else _failed(detail)
    return verdict


def stage_item_counts(analysis: dict) -> tuple[dict, dict]:
    """Per (prompt, opening) Alibaba/Microsoft and all-other_maker counts.

    Counts every pushed answer at the model's read rungs whose item is
    eligible.  Returns ({(prompt_index, opening_index): (am, om)}, totals).
    """
    per_item: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    table = analysis.get('items') or {}
    eligible = set(analysis['eligible_items'])
    cache: dict = {}
    rungs = {int(layer): dose for layer, dose in analysis['read_rungs'].items()
             if dose is not None}
    if not rungs:
        return {}, {'other_maker': 0, 'alibaba_microsoft': 0}
    for record in analysis['_records'].values():
        if record.get('direction_id') == 'zero':
            continue
        layer = int(record['layer'])
        if layer not in rungs or float(record['dose']) != float(rungs[layer]):
            continue
        item = item_of(record)
        if item not in eligible:
            continue
        row = table.get(str(item)) or {}
        key = (int(row.get('prompt_index', item // N_OPENINGS)),
               int(row.get('opening_index', item % N_OPENINGS)))
        state, family = classify_text(record['text'], cache)
        if state != 'other_maker':
            continue
        per_item[key][1] += 1
        if family in SHIFT_FAMILIES:
            per_item[key][0] += 1
    totals = {
        'other_maker': sum(value[1] for value in per_item.values()),
        'alibaba_microsoft': sum(value[0] for value in per_item.values()),
    }
    return dict(per_item), totals


def paired_intersection(sft: dict, dpo: dict) -> dict:
    """Amendment 1 item 6: prompts whose shared-openings items are eligible.

    A prompt enters when at least one opening is eligible at both SFT and DPO,
    and contributes only the opening(s) eligible at both.  Returns the shared
    item index per prompt, plus per-prompt paired counts summed over them.
    """
    sft_eligible = set(sft['eligible_items'])
    dpo_eligible = set(dpo['eligible_items'])
    sft_table = sft.get('items') or {}
    sft_counts, _ = stage_item_counts(sft)
    dpo_counts, _ = stage_item_counts(dpo)

    shared_items: dict[int, list[int]] = defaultdict(list)
    for item in sorted(sft_eligible & dpo_eligible):
        row = sft_table.get(str(item)) or {}
        prompt = int(row.get('prompt_index', item // N_OPENINGS))
        shared_items[prompt].append(item)
    prompts = sorted(shared_items)
    sft_pairs = []
    dpo_pairs = []
    for prompt in prompts:
        sft_pair = [0, 0]
        dpo_pair = [0, 0]
        for item in shared_items[prompt]:
            row = sft_table.get(str(item)) or {}
            opening = int(row.get('opening_index', item % N_OPENINGS))
            key = (prompt, opening)
            c_sft = sft_counts.get(key, (0, 0))
            c_dpo = dpo_counts.get(key, (0, 0))
            sft_pair[0] += c_sft[0]
            sft_pair[1] += c_sft[1]
            dpo_pair[0] += c_dpo[0]
            dpo_pair[1] += c_dpo[1]
        sft_pairs.append(tuple(sft_pair))
        dpo_pairs.append(tuple(dpo_pair))
    return {
        'prompts': prompts,
        'shared_items': {str(prompt): shared_items[prompt]
                         for prompt in prompts},
        'sft_counts': sft_pairs,
        'dpo_counts': dpo_pairs,
        'two_item_prompts': sum(1 for prompt in prompts
                                if len(shared_items[prompt]) == 2),
    }


def _share(counts: tuple[int, int]) -> float | None:
    return (counts[0] / counts[1]) if counts[1] else None


def prediction_218_6(sft, dpo, instruct) -> dict:
    if sft is None or dpo is None:
        return _not_read('no answers for ' +
                         ', '.join(key for key, value in
                                   (('SFT', sft), ('DPO', dpo)) if value is None))
    for key, analysis in (('SFT', sft), ('DPO', dpo)):
        if _void_reason(analysis):
            return _not_read(f'{key}: {_void_reason(analysis)}')
        if not analysis['gate']['passed']:
            return _not_read(f'{key} {_gate_reason(analysis)}')
    if not _rung_layers(sft) or not _rung_layers(dpo):
        return _not_read('SFT or DPO has no read rung')

    _sft_items, sft_totals = stage_item_counts(sft)
    _dpo_items, dpo_totals = stage_item_counts(dpo)
    share_sft = (sft_totals['alibaba_microsoft'] / sft_totals['other_maker']
                 if sft_totals['other_maker'] else None)
    share_dpo = (dpo_totals['alibaba_microsoft'] / dpo_totals['other_maker']
                 if dpo_totals['other_maker'] else None)
    detail = {
        'sft': {'read_rungs': sft['read_rungs'], **sft_totals,
                'share': share_sft},
        'dpo': {'read_rungs': dpo['read_rungs'], **dpo_totals,
                'share': share_dpo},
    }
    if sft_totals['other_maker'] < OTHER_MAKER_MIN:
        return _not_read(f"SFT has {sft_totals['other_maker']} other_maker "
                         f"answers at the read rungs (< {OTHER_MAKER_MIN})")
    if dpo_totals['other_maker'] < OTHER_MAKER_MIN:
        return _not_read(f"DPO has {dpo_totals['other_maker']} other_maker "
                         f"answers at the read rungs (< {OTHER_MAKER_MIN})")

    intersection = paired_intersection(sft, dpo)
    detail['intersection_prompts'] = len(intersection['prompts'])
    detail['intersection_two_item_prompts'] = intersection['two_item_prompts']
    detail['intersection_shared_items'] = sum(
        len(items) for items in intersection['shared_items'].values())
    if len(intersection['prompts']) < INTERSECTION_MIN:
        return _not_read(f"only {len(intersection['prompts'])} prompts "
                         f"eligible at both SFT and DPO (< {INTERSECTION_MIN})")

    sft_counts = intersection['sft_counts']
    dpo_counts = intersection['dpo_counts']
    bootstrap = paired_bootstrap(sft_counts, dpo_counts)
    sft_pair = (sum(c[0] for c in sft_counts), sum(c[1] for c in sft_counts))
    dpo_pair = (sum(c[0] for c in dpo_counts), sum(c[1] for c in dpo_counts))
    intersection_totals = {
        'sft': {'alibaba_microsoft': sft_pair[0], 'other_maker': sft_pair[1],
                'share': _share(sft_pair)},
        'dpo': {'alibaba_microsoft': dpo_pair[0], 'other_maker': dpo_pair[1],
                'share': _share(dpo_pair)},
    }
    detail['bootstrap'] = bootstrap
    detail['intersection'] = intersection_totals
    difference = (None if intersection_totals['sft']['share'] is None
                  or intersection_totals['dpo']['share'] is None
                  else intersection_totals['dpo']['share']
                  - intersection_totals['sft']['share'])
    detail['difference_paired_intersection'] = difference
    detail['difference_all_eligible'] = (
        None if share_sft is None or share_dpo is None
        else share_dpo - share_sft)
    if difference is None or difference < SHIFT_MIN:
        return _failed(detail)
    if (not bootstrap['excludes_zero'] or bootstrap['ci_low'] is None
            or bootstrap['ci_low'] <= 0):
        return _failed(detail)
    return _held(detail)


def compute_predictions(sft, dpo, instruct) -> dict:
    predictions = {
        '218.1': prediction_218_1(sft, dpo, instruct),
        '218.2': prediction_218_2(sft, dpo, instruct),
        '218.3': prediction_218_3(sft, dpo, instruct),
        '218.4': prediction_218_4(sft, dpo, instruct),
        '218.5': prediction_218_5(sft, dpo, instruct),
        '218.6': prediction_218_6(sft, dpo, instruct),
    }
    predictions['218.7'] = dict(predictions['218.1'])
    predictions['218.8'] = dict(predictions['218.4'])
    for number, meta in PREDICTION_META.items():
        predictions[number]['confidence'] = meta['confidence']
        predictions[number]['author'] = meta['author']
        predictions[number]['text'] = meta['text']
    return predictions


# ---------------------------------------------------------------------------
# Reader audit sample (the labelling is Claude's, not this script)
# ---------------------------------------------------------------------------

def audit_sample(analyses: dict) -> tuple[list[dict], dict]:
    """One sample per model: the smaller of 100 and all other_maker answers
    at the read rungs, on eligible items."""
    sample = []
    per_model = {}
    for key, analysis in analyses.items():
        if analysis is None:
            per_model[key] = {'available': False, 'reason': 'no answers'}
            continue
        eligible = set(analysis['eligible_items'])
        table = analysis.get('items') or {}
        rungs = {int(layer): dose
                 for layer, dose in analysis['read_rungs'].items()
                 if dose is not None}
        if not rungs:
            per_model[key] = {'available': False,
                              'reason': 'no read rung'}
            continue
        rows = []
        cache: dict = {}
        for record in analysis['_records'].values():
            if record.get('direction_id') == 'zero':
                continue
            layer = int(record['layer'])
            if layer not in rungs or float(record['dose']) != float(rungs[layer]):
                continue
            item = item_of(record)
            if item not in eligible:
                continue
            state, family = classify_text(record['text'], cache)
            if state != 'other_maker':
                continue
            row = table.get(str(item)) or {}
            rows.append({
                'model': key,
                'layer': layer,
                'dose': float(record['dose']),
                'prompt_index': int(row.get('prompt_index', item // N_OPENINGS)),
                'opening_index': int(row.get('opening_index', item % N_OPENINGS)),
                'opening': row.get('opening', OPENINGS[item % N_OPENINGS]),
                'item_index': item,
                'direction_id': record['direction_id'],
                'answer': record['text'],
            })
        rows.sort(key=lambda r: (r['layer'], r['dose'], r['item_index'],
                                 direction_order(r['direction_id'])))
        if len(rows) > 100:
            # The design fixes the sample size but not the draw; take evenly
            # spaced rows so every item and direction appears.
            step = len(rows) / 100.0
            rows = [rows[int(i * step)] for i in range(100)]
        for index, row in enumerate(rows, start=1):
            row['audit_id'] = f'{key}-{index:03d}'
            sample.append(row)
        per_model[key] = {'available': True, 'sample_size': len(rows),
                          'read_rungs': {str(layer): dose
                                         for layer, dose in rungs.items()}}
    return sample, per_model


# ---------------------------------------------------------------------------
# Build, write, print
# ---------------------------------------------------------------------------

def build_score(run: dict) -> tuple[dict, list[dict]]:
    """Return (score218 payload, reader-audit sample rows)."""
    analyses = {}
    for key in MODEL_KEYS:
        entry = run['models'].get(key)
        if not entry:
            analyses[key] = None
            continue
        analysis = analyse_model(entry['records'], entry['meta'], pilot=False)
        analysis['_records'] = entry['records']
        analysis['_meta'] = entry['meta']
        analyses[key] = analysis

    gate_analyses = {}
    for key in GATE_MODELS:
        entry = run.get('gate', {}).get(key)
        if not entry:
            gate_analyses[key] = None
            continue
        analysis = analyse_model(entry['records'], entry['meta'], gate=True)
        analysis['_records'] = entry['records']
        analysis['_meta'] = entry['meta']
        gate_analyses[key] = analysis

    pilot_analysis = None
    pilot = run.get('pilot')
    if pilot:
        pilot_analysis = analyse_model(pilot['records'], pilot['meta'],
                                       pilot=True)
        pilot_analysis['stop_rule'] = pilot_stop_rule(pilot_analysis,
                                                      gate_analyses)
        pilot_analysis['trim_rule'] = trim_rule(
            pilot['records'], pilot_analysis['layers'],
            pilot_analysis['doses'])
        pilot_analysis['_records'] = pilot['records']
        pilot_analysis['_meta'] = pilot['meta']

    predictions = compute_predictions(analyses.get('SFT'),
                                      analyses.get('DPO'),
                                      analyses.get('Instruct'))
    gold = gold_check()
    if not gold['ok']:
        # The instrument's first gate failed; no prediction can be read.
        for prediction in predictions.values():
            prediction['verdict'] = 'not read'
            prediction['reason'] = (
                f"attrib218 gold set failed ({gold['matched']}/{gold['rows']})")
            prediction.pop('detail', None)

    binomial_tests = []
    for key, analysis in analyses.items():
        if analysis is None:
            continue
        for layer_text, by_dose in analysis['slices'].items():
            for dose_text, stats in by_dose.items():
                if stats['moved'] and stats['leading_family'] is not None:
                    binomial_tests.append({
                        'model': key, 'layer': int(layer_text),
                        'dose': float(dose_text),
                        'leading_family': stats['leading_family'],
                        'led': stats['led_counts'][stats['leading_family']],
                        'moved': stats['moved'],
                        'share': stats['leading_share'],
                        'p_value_one_sided_greater_vs_1_10':
                            stats['leading_p_value'],
                    })

    sample, audit_meta = audit_sample(analyses)

    def public(analysis):
        if analysis is None:
            return None
        return {field: value for field, value in analysis.items()
                if not field.startswith('_')}

    score = {
        'part': 218,
        'instrument': 'v4 (Amendment 1)',
        'run_dir': run['run_dir'],
        'models': {key: public(analysis)
                   for key, analysis in analyses.items()},
        'gate_passes': {key: public(analysis)
                        for key, analysis in gate_analyses.items()},
        'pilot': public(pilot_analysis),
        'predictions': predictions,
        'gold': gold,
        'prompts': {'path': str(PROMPTS_PATH), 'sha256': prompts_sha256(),
                    'expected_sha256': PROMPTS_SHA256},
        'binomial_tests': binomial_tests,
        'audit': {'per_model': audit_meta, 'sample_size': len(sample)},
        'constants': {
            'openings': list(OPENINGS),
            'n_prompts': N_PROMPTS,
            'n_openings': N_OPENINGS,
            'n_items': N_ITEMS,
            'gate_min_items': GATE_MIN_ITEMS,
            'gate_min_per_opening': GATE_MIN_PER_OPENING,
            'identical_min_items': IDENTICAL_MIN_ITEMS,
            'read_rung_min': READ_RUNG_MIN,
            'n_directions': N_DIRECTIONS,
            'rows_per_batch': ROWS_PER_BATCH,
            'moved_min_prompts': MOVED_MIN_PROMPTS,
            'broken_fraction': BROKEN_FRACTION,
            'full_zero_batches': FULL_ZERO_BATCHES,
            'bootstrap_draws': BOOTSTRAP_DRAWS,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'intersection_min': INTERSECTION_MIN,
            'other_maker_min': OTHER_MAKER_MIN,
            'shift_min': SHIFT_MIN,
            'prompts_sha256': PROMPTS_SHA256,
            'rule_version': attrib218.RULE_VERSION,
        },
    }
    return score, sample


def write_audit(path: Path, sample: list[dict]) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def print_table(score: dict) -> None:
    print('\nPart 218 v4 score: ' + score['run_dir'])
    gold = score.get('gold') or {}
    print(f"Gold set: {gold.get('matched')}/{gold.get('rows')} "
          f"(attrib218 {gold.get('rule_version')})")
    print(f"{'model':10s} {'gate items':>12s} {'per opening':>14s} "
          f"{'read rungs':>14s} {'moved':>8s} {'led':>24s}")
    for key in MODEL_KEYS:
        analysis = score['models'].get(key)
        if analysis is None:
            print(f'{key:10s} {"missing":>12s}')
            continue
        gate = analysis['gate']
        rungs = ', '.join(
            f"L{layer} d{dose:g}" if dose is not None else f'L{layer} none'
            for layer, dose in analysis['read_rungs'].items())
        moved = ', '.join(
            f"L{layer}:{stats['moved']}" if stats else f'L{layer}:-'
            for layer, stats in analysis['read_rung_stats'].items())
        led = '; '.join(
            f"L{layer}:{stats['led_counts']}" if stats else ''
            for layer, stats in analysis['read_rung_stats'].items())
        print(f"{key:10s} {gate['eligible_items']:>4d}/{gate['items']:<7d} "
              f"{str(gate['per_opening']):>14s} {rungs:>14s} {moved:>8s} "
              f"{led:>24s}")
    if score.get('gate_passes'):
        print('\nGate-only passes (zero push):')
        for key, analysis in score['gate_passes'].items():
            if analysis is None:
                print(f'  {key}: missing')
                continue
            gate = analysis['gate']
            all_id = sum(1 for row in analysis['per_item'].values()
                         if row['all_identical_rows'])
            all_self = sum(1 for row in analysis['per_item'].values()
                           if row['all_self'])
            print(f"  {key}: L{analysis['layers'][0]} "
                  f"{gate['eligible_items']}/{gate['items']} eligible, "
                  f"per opening {gate['per_opening']}, passed={gate['passed']}; "
                  f"{all_id}/{analysis['n_items']} items all-identical, "
                  f"{all_self}/{analysis['n_items']} all-self")
    print('\nPrediction  verdict    detail')
    for number in sorted(score['predictions']):
        prediction = score['predictions'][number]
        detail = prediction.get('reason') or json.dumps(
            prediction.get('detail'), sort_keys=True)
        if len(detail) > 160:
            detail = detail[:157] + '...'
        print(f"{number:10s} {prediction['verdict']:10s} {detail}")
    pilot = score.get('pilot')
    if pilot:
        print(f"\nPilot: eligible items {pilot['gate']['eligible_items']}/"
              f"{pilot['gate']['items']} (per opening "
              f"{pilot['gate']['per_opening']}); read rungs "
              f"{pilot['read_rungs']}; stop rule "
              f"triggered={pilot['stop_rule']['triggered']}; trim kept "
              f"{pilot['trim_rule']['kept_ladder']}")
        if pilot['stop_rule']['triggered']:
            print('  reasons: ' + '; '.join(pilot['stop_rule']['reasons']))
    print(f"\nAudit sample: {score['audit']['sample_size']} answer(s) -> "
          f"audit218.jsonl (labelling is Claude's)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run-dir', required=True,
                        help='the runner --out root (holds pilot/, gate/, ...)')
    parser.add_argument('--out', default=None,
                        help='score218.json path (default <run-dir>/score218.json)')
    parser.add_argument('--audit', default=None,
                        help='audit sample path (default <run-dir>/audit218.jsonl)')
    args = parser.parse_args(argv)

    run = load_run_dir(Path(args.run_dir))
    if (run.get('pilot') is None and not any(run['models'].values())
            and not any(run.get('gate', {}).values())):
        raise SystemExit(f'no answers.jsonl found under {args.run_dir}')
    score, sample = build_score(run)
    out_path = Path(args.out) if args.out else Path(args.run_dir) / 'score218.json'
    audit_path = (Path(args.audit) if args.audit else Path(args.run_dir) / 'audit218.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(score, indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    write_audit(audit_path, sample)
    print_table(score)
    print(f'wrote {out_path} and {audit_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
