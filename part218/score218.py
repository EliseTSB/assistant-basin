#!/usr/bin/env python3
"""Part 218 scorer: read answers.jsonl, apply attrib218, emit score218.json.

Spec: `part218/DESIGN.md` (v3, frozen), sections Method, Per-run quantities,
Controls, Predictions and Reading rules.  Nothing here chooses a number: every
quantity is computed from the answers the runner wrote.

    python part218/score218.py --run-dir /content/out218
    python part218/score218.py --run-dir /content/out218 --out score218.json

The run dir is the runner's `--out` root:

    <run-dir>/directions.pt
    <run-dir>/pilot/{answers.jsonl,run.json}          optional
    <run-dir>/SFT|DPO|Instruct/{answers.jsonl,run.json}

Writes `<out>` (default `<run-dir>/score218.json`) and the reader-audit sample
`audit218.jsonl` beside it.  The audit labels are NOT produced here: the
design's Reading rules (F14) leave the labelling to two readers, so this file
only chooses the answers to label.

Per-run quantities implemented exactly as DESIGN.md words them:
  * the unpushed answer is the majority state over the zero rows, and a prompt
    whose zero rows are not all one state is not `self` (F7);
  * eligible = unpushed `self`; the gate is 30 of 44;
  * supported = `broken` on at most floor(10% of eligible) prompts;
  * moved = supported and at least 3 eligible prompts `other_maker`;
  * led by X = a moved direction's most frequent other_maker family, ties not
    led;
  * the read rung is the largest d with at least 36 of 40 supported (9 of 10 in
    the pilot), chosen by that rule, never by hand.
"""
from __future__ import annotations

import argparse
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

MODEL_KEYS = ('SFT', 'DPO', 'Instruct')
# Method ladder and the pilot ladder (DESIGN.md, Dose / Size).
LADDER = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
PILOT_DOSES = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5)

GATE_MIN_ELIGIBLE = 30
DESIGN_N_PROMPTS = 44
READ_RUNG_MIN = 36
READ_RUNG_MIN_PILOT = 9
N_DIRECTIONS = 40
N_DIRECTIONS_PILOT = 10
MOVED_MIN = 3
BROKEN_FRACTION = 0.10
FULL_ZERO_BATCHES = 14          # 2 layers x 7 doses (DESIGN.md 218.5)

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
              'text': 'Each of SFT, DPO and Instruct passes the gate (at '
                      'least 30 of 44 prompts eligible).'},
    '218.5': {'confidence': 'high (85%)', 'author': 'Journal Claude',
              'text': 'For each model, the zero row is identical across all '
                      '14 batches on at least 42 of 44 prompts, and the '
                      'no-hook null matches bit for bit at the start and the '
                      'end.'},
    '218.6': {'confidence': 'low (30%)', 'author': 'Journal Claude',
              'text': 'Stage shift: pooling both layers\' read rungs, the '
                      'share of other_maker answers (on eligible prompts, '
                      'refusals excluded) naming Alibaba or Microsoft is at '
                      'least 10 points higher at DPO than at SFT, with a '
                      'paired bootstrap (10,000 draws) over the intersection '
                      'of prompts eligible at both SFT and DPO whose 95% '
                      'interval excludes 0; the intersection must have at '
                      'least 25 prompts. Not read if either model has fewer '
                      'than 20 other_maker answers pooled.'},
    '218.7': {'confidence': 'low (25%)', 'author': 'Vesper',
              'text': 'The same prediction as 218.1.'},
    '218.8': {'confidence': 'low (45%)', 'author': 'Vesper',
              'text': 'The same prediction as 218.4.'},
}


# ---------------------------------------------------------------------------
# Reading the run
# ---------------------------------------------------------------------------

def row_key(record: dict) -> tuple:
    return (record['model'], int(record['layer']), float(record['dose']),
            int(record['prompt_index']), record['direction_id'])


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
    out = {'pilot': None, 'models': {}, 'run_dir': str(run_dir.resolve())}
    pilot_dir = run_dir / 'pilot'
    if (pilot_dir / 'answers.jsonl').exists():
        out['pilot'] = {'records': load_answers(pilot_dir / 'answers.jsonl'),
                        'meta': load_meta(pilot_dir / 'run.json')}
    for key in MODEL_KEYS:
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
# The unpushed answer and eligibility (DESIGN.md, Per-run quantities)
# ---------------------------------------------------------------------------

def unpushed_table(records: dict, meta: dict, cache: dict) -> dict:
    """Per prompt: zero-row states, majority, unanimity, eligibility.

    The design: majority over the zero rows is the unpushed answer; a prompt
    whose zero rows are not all the same state counts as not-`self` (F7).
    """
    layers = meta.get('layers')
    doses = meta.get('doses')
    zeros: dict[int, dict] = defaultdict(dict)
    for record in records.values():
        if record['direction_id'] != 'zero':
            continue
        zeros[int(record['prompt_index'])][
            (int(record['layer']), float(record['dose']))] = record

    if layers and doses:
        expected_batches = len(layers) * len(doses)
    else:
        batches = {(int(r['layer']), float(r['dose'])) for r in records.values()}
        expected_batches = len(batches)

    n_prompts = int(meta.get('n_prompts') or (max(zeros, default=-1) + 1))
    table = {}
    for prompt_index in range(n_prompts):
        rows = zeros.get(prompt_index, {})
        states = [classify_text(record['text'], cache)[0]
                  for record in rows.values()]
        counts = Counter(states)
        majority = None
        if counts:
            top = counts.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                majority = top[0][0]
        unanimous = bool(states) and len(counts) == 1
        complete = len(rows) >= expected_batches and expected_batches > 0
        eligible = bool(complete and unanimous and majority == 'self')
        table[prompt_index] = {
            'state': majority,
            'unanimous': unanimous,
            'complete': complete,
            'n_zero_rows': len(rows),
            'expected_zero_rows': expected_batches,
            'counts': dict(counts),
            'eligible': eligible,
        }
    return table


# ---------------------------------------------------------------------------
# Per (model, layer, dose) direction quantities
# ---------------------------------------------------------------------------

def direction_stats(records: dict, eligible: list[int], layer: int, dose: float,
                    cache: dict) -> dict:
    """supported / moved / led for every direction at one (layer, dose)."""
    n_eligible = len(eligible)
    eligible_set = set(eligible)
    broken_allowed = int(BROKEN_FRACTION * n_eligible + 1e-9)
    by_direction: dict[str, dict] = {}
    refusal_counts: Counter = Counter()
    other_counts: Counter = Counter()
    strings_by_family: dict[str, set] = defaultdict(set)

    for record in records.values():
        if int(record['layer']) != int(layer):
            continue
        if float(record['dose']) != float(dose):
            continue
        if record['direction_id'] == 'zero':
            continue
        if int(record['prompt_index']) not in eligible_set:
            continue
        direction_id = record['direction_id']
        bucket = by_direction.setdefault(direction_id, {
            'states': Counter(), 'other_maker': Counter(),
            'answers': [], 'broken': 0})
        state, family = classify_text(record['text'], cache)
        bucket['states'][state] += 1
        bucket['answers'].append(record)
        if state == 'broken':
            bucket['broken'] += 1
        elif state == 'refusal':
            refusal_counts[family] += 1
        elif state == 'other_maker':
            bucket['other_maker'][family] += 1
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
        supported = bucket['broken'] <= broken_allowed
        supported_count += int(supported)
        moved = supported and sum(bucket['other_maker'].values()) >= MOVED_MIN
        led = None
        if moved:
            moved_count += 1
            if bucket['other_maker']:
                ranked = bucket['other_maker'].most_common()
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    led = ranked[0][0]
            if led is None:
                not_led += 1
            else:
                led_counts[led] += 1
                for record in bucket['answers']:
                    state, family = classify_text(record['text'], cache)
                    if state == 'other_maker' and family == led:
                        led_strings[led].add(record['text'])
        per_direction[direction_id] = {
            'supported': supported,
            'moved': moved,
            'led': led,
            'broken': bucket['broken'],
            'states': dict(bucket['states']),
            'other_maker': dict(bucket['other_maker']),
        }

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
        'per_direction': per_direction,
    }


def read_rung(slices: dict, read_min: int) -> float | None:
    """Largest d at which at least `read_min` directions are supported."""
    passing = [float(dose) for dose, stats in slices.items()
               if stats['supported'] >= read_min]
    return max(passing) if passing else None


def zero_row_identity(records: dict, meta: dict) -> dict:
    """218.5's first half: the zero row identical across every batch."""
    batches: dict[int, dict] = defaultdict(dict)
    for record in records.values():
        if record['direction_id'] != 'zero':
            continue
        batches[int(record['prompt_index'])][
            (int(record['layer']), float(record['dose']))] = tuple(
                record['token_ids'])
    expected = None
    if meta.get('layers') and meta.get('doses'):
        expected = len(meta['layers']) * len(meta['doses'])
    identical = 0
    incomplete = 0
    total = int(meta.get('n_prompts') or len(batches))
    for prompt_index in range(total):
        rows = batches.get(prompt_index, {})
        if not rows:
            incomplete += 1
            continue
        n_expected = expected if expected is not None else len(rows)
        if len(rows) >= n_expected and len(set(rows.values())) == 1:
            identical += 1
    return {'prompts': total,
            'batches_per_prompt': expected,
            'identical_prompts': identical,
            'incomplete_prompts': incomplete,
            'ok': identical >= total - 2 if total else False}


def analyse_model(records: dict, meta: dict, pilot: bool = False) -> dict:
    """Everything the design asks of one model's answers."""
    cache: dict = {}
    layers = sorted({int(r['layer']) for r in records.values()})
    doses = sorted({float(r['dose']) for r in records.values()})
    if meta.get('layers'):
        layers = sorted(int(layer) for layer in meta['layers'])
    if meta.get('doses'):
        doses = sorted(float(dose) for dose in meta['doses'])
    n_prompts = int(meta.get('n_prompts') or DESIGN_N_PROMPTS)
    n_directions_expected = N_DIRECTIONS_PILOT if pilot else N_DIRECTIONS
    read_min = READ_RUNG_MIN_PILOT if pilot else READ_RUNG_MIN

    unpushed = unpushed_table(records, meta, cache)
    eligible = sorted(pi for pi, row in unpushed.items() if row['eligible'])
    gate = {
        'eligible': len(eligible),
        'prompts': n_prompts,
        'minimum': GATE_MIN_ELIGIBLE,
        'passed': len(eligible) >= GATE_MIN_ELIGIBLE,
        'rule': f'at least {GATE_MIN_ELIGIBLE} of {n_prompts} prompts eligible '
                f'(unpushed self)',
    }

    slices: dict[int, dict] = {}
    for layer in layers:
        slices[layer] = {}
        for dose in doses:
            slices[layer][dose] = direction_stats(
                records, eligible, layer, dose, cache)

    rungs: dict[int, float | None] = {}
    rung_stats: dict[int, dict | None] = {}
    for layer in layers:
        rung = read_rung(slices[layer], read_min)
        rungs[layer] = rung
        rung_stats[layer] = slices[layer][rung] if rung is not None else None

    refusals_overall: Counter = Counter()
    for record in records.values():
        if record['direction_id'] == 'zero':
            continue
        state, family = classify_text(record['text'], cache)
        if state == 'refusal':
            refusals_overall[family] += 1

    all_lengths = [int(r['n_new_tokens']) for r in records.values()
                   if r['direction_id'] != 'zero']
    all_words = [word_count(r['text']) for r in records.values()
                 if r['direction_id'] != 'zero']
    all_caps = [bool(r.get('hit_cap')) for r in records.values()
                if r['direction_id'] != 'zero']

    return {
        'layers': layers,
        'doses': doses,
        'n_prompts': n_prompts,
        'n_directions_expected': n_directions_expected,
        'read_min_supported': read_min,
        'gate': gate,
        'eligible_prompts': eligible,
        'unpushed': {str(pi): row for pi, row in unpushed.items()},
        'slices': {str(layer): {str(dose): stats
                                for dose, stats in by_dose.items()}
                   for layer, by_dose in slices.items()},
        'read_rungs': {str(layer): rung for layer, rung in rungs.items()},
        'read_rung_stats': {str(layer): stats
                            for layer, stats in rung_stats.items()},
        'zero_row_identity': zero_row_identity(records, meta),
        'null': meta.get('null', {}),
        'run_status': meta.get('status'),
        'rows_written': meta.get('rows_written'),
        'rows_planned': meta.get('rows_planned'),
        'refusals_by_family_overall': dict(refusals_overall),
        'median_length_tokens': (statistics.median(all_lengths)
                                 if all_lengths else None),
        'median_length_words': (statistics.median(all_words)
                                if all_words else None),
        'cap_fraction': (sum(all_caps) / len(all_caps) if all_caps else None),
        'answers': len(all_lengths),
    }


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
    the intersection.  A draw resamples prompts with replacement and recomputes
    both pooled shares from the same resampled prompts (paired).  A draw where
    either denominator is 0 has no share; it is counted, not used.
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
# The pilot: stop rule and trim rule
# ---------------------------------------------------------------------------

def pilot_stop_rule(analysis: dict) -> dict:
    """DESIGN.md Size: stop when the pilot says the main run must be redesigned."""
    reasons = []
    if analysis['gate']['eligible'] < GATE_MIN_ELIGIBLE:
        reasons.append(f"fewer than {GATE_MIN_ELIGIBLE} prompts eligible "
                       f"({analysis['gate']['eligible']})")
    rung_reached = any(stats is not None
                       for stats in analysis['read_rung_stats'].values())
    if not rung_reached:
        reasons.append(f"no rung reaches {READ_RUNG_MIN_PILOT} of "
                       f"{N_DIRECTIONS_PILOT} supported directions")
    return {'triggered': bool(reasons), 'reasons': reasons,
            'rung_reached': rung_reached,
            'rule': f'stop if fewer than {GATE_MIN_ELIGIBLE} prompts are '
                    f'eligible, or if no rung reaches {READ_RUNG_MIN_PILOT} of '
                    f'{N_DIRECTIONS_PILOT} supported directions'}


def trim_rule(records: dict, layers: list[int], doses: list[float]) -> dict:
    """Drop Method-ladder rungs above the first rung where most answers are broken.

    "starting from the Method ladder, drop every rung above the first rung at
    which more than half the answers are `broken`, keeping the one rung above
    it; the read-rung rule is unchanged."

    Answers = every pushed pilot answer at that rung (10 directions x 44
    prompts, eligible or not: the trim rule is about readability, not the
    gate).  The pilot only visits 0.25 to 1.5; the Method ladder's 2.0 and
    3.0 cannot be evaluated from it and are reported as undetermined.
    """
    states_by_dose: dict[float, Counter] = defaultdict(Counter)
    cache: dict = {}
    for record in records.values():
        if record['direction_id'] == 'zero':
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


def _openai_prediction(analysis: dict | None, label: str) -> dict:
    if analysis is None:
        return _not_read(f'no answers for {label}')
    if _void_reason(analysis):
        return _not_read(_void_reason(analysis))
    if not analysis['gate']['passed']:
        return _not_read(f"gate failed: {analysis['gate']['eligible']} of "
                         f"{analysis['gate']['prompts']} prompts eligible")
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
        return _not_read(f"gate failed: {analysis['gate']['eligible']} of "
                         f"{analysis['gate']['prompts']} prompts eligible")
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
    detail = {key: {'eligible': analysis['gate']['eligible'],
                    'prompts': analysis['gate']['prompts'],
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
        batches = identity.get('batches_per_prompt')
        prompts = identity.get('prompts') or 0
        enough_batches = batches is not None and batches >= FULL_ZERO_BATCHES
        null_ok = bool(null.get('all_ok'))
        entry = {
            'batches_per_prompt': batches,
            'expected_batches': FULL_ZERO_BATCHES,
            'identical_prompts': identity.get('identical_prompts'),
            'prompts': prompts,
            'identity_ok': bool(enough_batches and prompts > 0
                                and identity.get('identical_prompts', 0)
                                >= prompts - 2),
            'null_all_ok': null_ok,
            'null_hooked_vs_start': hooked.get('all_ok'),
            'null_end_vs_start': vs_start.get('all_ok'),
            'null_end_vs_hooked': end_hooked.get('all_ok'),
            'run_status': analysis.get('run_status'),
        }
        entry['holds'] = entry['identity_ok'] and entry['null_all_ok']
        detail[key] = entry
    if any(not entry['identity_ok']
           and entry['batches_per_prompt'] is not None
           and entry['batches_per_prompt'] < FULL_ZERO_BATCHES
           for entry in detail.values()):
        return _not_read('a model ran fewer than 14 zero-row batches '
                         '(trimmed ladder); 218.5 names all 14')
    verdict = _held(detail) if all(e['holds'] for e in detail.values()) \
        else _failed(detail)
    return verdict


def stage_counts(analysis: dict) -> tuple[dict, dict]:
    """Per-prompt Alibaba/Microsoft counts at the read rungs, and the shares.

    Returns ({prompt_index: (alibaba_or_microsoft, all_other_maker)}, totals).
    """
    per_prompt: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    eligible = set(analysis['eligible_prompts'])
    cache: dict = {}
    rungs = {int(layer): dose for layer, dose in analysis['read_rungs'].items()
             if dose is not None}
    if not rungs:
        return {}, {'other_maker': 0, 'alibaba_microsoft': 0}
    for record in analysis['_records'].values():
        if record['direction_id'] == 'zero':
            continue
        layer = int(record['layer'])
        if layer not in rungs or float(record['dose']) != float(rungs[layer]):
            continue
        prompt_index = int(record['prompt_index'])
        if prompt_index not in eligible:
            continue
        state, family = classify_text(record['text'], cache)
        if state != 'other_maker':
            continue
        per_prompt[prompt_index][1] += 1
        if family in SHIFT_FAMILIES:
            per_prompt[prompt_index][0] += 1
    totals = {
        'other_maker': sum(value[1] for value in per_prompt.values()),
        'alibaba_microsoft': sum(value[0] for value in per_prompt.values()),
    }
    return dict(per_prompt), totals


def prediction_218_6(sft, dpo, instruct) -> dict:
    if sft is None or dpo is None:
        return _not_read('no answers for ' +
                         ', '.join(key for key, value in
                                   (('SFT', sft), ('DPO', dpo)) if value is None))
    for key, analysis in (('SFT', sft), ('DPO', dpo)):
        if _void_reason(analysis):
            return _not_read(f'{key}: {_void_reason(analysis)}')
        if not analysis['gate']['passed']:
            return _not_read(f'{key} gate failed: '
                             f"{analysis['gate']['eligible']} of "
                             f"{analysis['gate']['prompts']} prompts eligible")
    if not _rung_layers(sft) or not _rung_layers(dpo):
        return _not_read('SFT or DPO has no read rung')

    sft_prompts, sft_totals = stage_counts(sft)
    dpo_prompts, dpo_totals = stage_counts(dpo)
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

    intersection = sorted(set(sft['eligible_prompts'])
                          & set(dpo['eligible_prompts']))
    detail['intersection_prompts'] = len(intersection)
    if len(intersection) < INTERSECTION_MIN:
        return _not_read(f'only {len(intersection)} prompts eligible at both '
                         f'SFT and DPO (< {INTERSECTION_MIN})')

    sft_counts = [tuple(sft_prompts.get(pi, (0, 0))) for pi in intersection]
    dpo_counts = [tuple(dpo_prompts.get(pi, (0, 0))) for pi in intersection]
    bootstrap = paired_bootstrap(sft_counts, dpo_counts)
    detail['bootstrap'] = bootstrap
    intersection_totals = {
        'sft': {'alibaba_microsoft': sum(c[0] for c in sft_counts),
                'other_maker': sum(c[1] for c in sft_counts)},
        'dpo': {'alibaba_microsoft': sum(c[0] for c in dpo_counts),
                'other_maker': sum(c[1] for c in dpo_counts)},
    }
    intersection_totals['sft']['share'] = (
        intersection_totals['sft']['alibaba_microsoft']
        / intersection_totals['sft']['other_maker']
        if intersection_totals['sft']['other_maker'] else None)
    intersection_totals['dpo']['share'] = (
        intersection_totals['dpo']['alibaba_microsoft']
        / intersection_totals['dpo']['other_maker']
        if intersection_totals['dpo']['other_maker'] else None)
    detail['intersection'] = intersection_totals

    difference = (None if share_sft is None or share_dpo is None
                  else share_dpo - share_sft)
    detail['difference_all_eligible'] = difference
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
# Reader audit sample (the labelling is Claude's, not this script's)
# ---------------------------------------------------------------------------

def audit_sample(analyses: dict) -> tuple[list[dict], dict]:
    """One sample per model: the smaller of 100 and all other_maker answers
    at the read rungs, on eligible prompts."""
    sample = []
    per_model = {}
    for key, analysis in analyses.items():
        if analysis is None:
            per_model[key] = {'available': False, 'reason': 'no answers'}
            continue
        eligible = set(analysis['eligible_prompts'])
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
            if record['direction_id'] == 'zero':
                continue
            layer = int(record['layer'])
            if layer not in rungs or float(record['dose']) != float(rungs[layer]):
                continue
            if int(record['prompt_index']) not in eligible:
                continue
            state, family = classify_text(record['text'], cache)
            if state != 'other_maker':
                continue
            rows.append({
                'model': key,
                'layer': layer,
                'dose': float(record['dose']),
                'prompt_index': int(record['prompt_index']),
                'direction_id': record['direction_id'],
                'answer': record['text'],
            })
        rows.sort(key=lambda r: (r['layer'], r['dose'], r['prompt_index'],
                                 direction_order(r['direction_id'])))
        if len(rows) > 100:
            # The design fixes the sample size but not the draw; take evenly
            # spaced rows so every prompt and direction appears.
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

    pilot_analysis = None
    pilot = run.get('pilot')
    if pilot:
        pilot_analysis = analyse_model(pilot['records'], pilot['meta'],
                                       pilot=True)
        pilot_analysis['stop_rule'] = pilot_stop_rule(pilot_analysis)
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

    public_models = {}
    for key, analysis in analyses.items():
        if analysis is None:
            public_models[key] = None
            continue
        public = {field: value for field, value in analysis.items()
                  if not field.startswith('_')}
        public_models[key] = public
    if pilot_analysis is not None:
        pilot_public = {field: value for field, value in pilot_analysis.items()
                        if not field.startswith('_')}
    else:
        pilot_public = None

    score = {
        'part': 218,
        'run_dir': run['run_dir'],
        'models': public_models,
        'pilot': pilot_public,
        'predictions': predictions,
        'gold': gold,
        'binomial_tests': binomial_tests,
        'audit': {'per_model': audit_meta, 'sample_size': len(sample)},
        'constants': {
            'gate_min_eligible': GATE_MIN_ELIGIBLE,
            'n_prompts': DESIGN_N_PROMPTS,
            'read_rung_min': READ_RUNG_MIN,
            'read_rung_min_pilot': READ_RUNG_MIN_PILOT,
            'moved_min': MOVED_MIN,
            'broken_fraction': BROKEN_FRACTION,
            'bootstrap_draws': BOOTSTRAP_DRAWS,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'intersection_min': INTERSECTION_MIN,
            'other_maker_min': OTHER_MAKER_MIN,
            'shift_min': SHIFT_MIN,
            'rule_version': attrib218.RULE_VERSION,
        },
    }
    return score, sample


def write_audit(path: Path, sample: list[dict]) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def print_table(score: dict) -> None:
    print('\nPart 218 score: ' + score['run_dir'])
    gold = score.get('gold') or {}
    print(f"Gold set: {gold.get('matched')}/{gold.get('rows')} "
          f"(attrib218 {gold.get('rule_version')})")
    print(f"{'model':10s} {'gate':>10s} {'read rungs':>12s} "
          f"{'moved at rung':>14s} {'led':>28s}")
    for key in MODEL_KEYS:
        analysis = score['models'].get(key)
        if analysis is None:
            print(f'{key:10s} {"missing":>10s}')
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
        print(f"{key:10s} {gate['eligible']:>4d}/{gate['prompts']:<5d} "
              f"{rungs:>12s} {moved:>14s} {led:>28s}")
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
        print(f"\nPilot: eligible {pilot['gate']['eligible']}/"
              f"{pilot['gate']['prompts']}; read rungs {pilot['read_rungs']}; "
              f"stop rule triggered={pilot['stop_rule']['triggered']}; "
              f"trim kept {pilot['trim_rule']['kept_ladder']}")
    print(f"\nAudit sample: {score['audit']['sample_size']} answer(s) -> "
          f"audit218.jsonl (labelling is Claude's)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run-dir', required=True,
                        help='the runner --out root (holds pilot/, SFT/, ...)')
    parser.add_argument('--out', default=None,
                        help='score218.json path (default <run-dir>/score218.json)')
    parser.add_argument('--audit', default=None,
                        help='audit sample path (default <run-dir>/audit218.jsonl)')
    args = parser.parse_args(argv)

    run = load_run_dir(Path(args.run_dir))
    if run.get('pilot') is None and not any(run['models'].values()):
        raise SystemExit(f'no answers.jsonl found under {args.run_dir}')
    score, sample = build_score(run)
    out_path = Path(args.out) if args.out else Path(args.run_dir) / 'score218.json'
    audit_path = (Path(args.audit) if args.audit
                  else Path(args.run_dir) / 'audit218.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(score, indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    write_audit(audit_path, sample)
    print_table(score)
    print(f'wrote {out_path} and {audit_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
