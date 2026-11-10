#!/usr/bin/env python3
"""Part 201 scorer (host side): gates, pair comparisons, registered predictions and the plan's ordered decision.

    python -I part201/score201.py --rows results/part201/rows.jsonl --load results/part201/load.json \
        --plan results/part201/plan.json --kernel results/part201/kernel-f32.json \
        --reference results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json \
        --probe results/part201/probe.json --out results/part201

Every number in the write-up is printed from here. The gates are read per precision: a float32 arm that did not
run, or failed its gates, leaves the bf16 readings standing and reports 201.7 to 201.11 as not run or not read.
A row is read only if it traces to a launch of the right process and to that launch's receipt for its call.
What a gate needs is taken from the runner's own frozen plan (part201/run201.py), not from constants repeated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'part86')]
from part163 import score163          # noqa: E402
from part201 import run201            # noqa: E402  (plan only; torch is imported inside its run functions)
from tools.margin_recorder import bf16_step  # noqa: E402

# The registered bars, in one place.
THRESHOLDS = {
    '201.2': {'min_share': .25, 'max_share': .50},
    '201.3': {'min_share': .20},
    '201.4': {'max_share': .10},
    '201.5': {'min_caught_share': .90, 'caught_steps': 2, 'no_margin_over_steps': 8},
    '201.6': {'minimum_flips_per_label': 8, 'direction_imbalance_factor': 2, 'length_difference_tokens': 3},
    '201.9': {'min_caught_share': .90, 'caught_steps': 2, 'f8_max_margin': .5},
    '201.10': {'min_share': .30, 'max_share': .70},
    '201.11': {'max_slowdown': 2, 'margin_steps': 2},
    'positive_control': {'min_rows_differing_between_A_and_C': 10},
}
PREDICTIONS = [
    ('201.1', 'A-again equals A on all 16 rows, and A2 equals A on all 96, token for token.', 'high'),
    ('201.2', 'Between A and C, 25% to 50% of the 192 rows differ.', 'medium'),
    ('201.3', 'Between A and M, at least 20% of the 384 rows differ.', 'low'),
    ('201.4', 'Between A and B (other slots and other neighbours, same size and prompts), at most 10% of the 384 rows differ.', 'low'),
    ('201.5', 'Of the rows that differ between A and C, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has one above 8 steps.', 'high'),
    ('201.6', 'Between A and M, for each of the four labels with at least 8 flips, the two directions of flip differ by at most twice the square root of the number of flips; and mean answer length differs by under 3 tokens.', 'medium'),
    ('201.7', 'K matches the CPU reference token for token on all 16 rows, over every position.', 'medium'),
    ('201.8', 'Between F8 and F1, none of the 48 rows differ, with all 48 present and finite in both.', 'medium'),
    ('201.9', 'Of the rows that differ between A and F8, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has an F8 (float32) margin above 0.5 there.', 'medium'),
    ('201.10', 'Between A and F8, 30% to 70% of the 96 rows differ.', 'low'),
    ('201.11', "The projected slowdown of margin-gated decoding on pass A is at most 2: (pass A's seconds + the summed cost of one full-prefix float32 forward for the flagged rows at every step where an unfinished row is within 2 steps) ÷ pass A's seconds, the cost of each forward read from the timing probe at the nearest larger row count and prefix length.", 'low'),
]
PROCESS_OF = {'A': 'bf16-main', 'A-again': 'bf16-main', 'B': 'bf16-main', 'M': 'bf16-main', 'C': 'bf16-main',
              'A2': 'bf16-second', 'F8': 'float32', 'F1': 'float32'}
BF16_PASSES = tuple(p for p, proc in PROCESS_OF.items() if proc != 'float32')
FLOAT_PASSES = ('F8', 'F1')
BF16_PAIRS = [('A', 'A-again'), ('A', 'A2'), ('A', 'B'), ('A', 'M'), ('M', 'C'), ('A', 'C')]
FLOAT_PAIRS = [('F8', 'F1'), ('A', 'F8'), ('C', 'F1')]
PAIRS = BF16_PAIRS + FLOAT_PAIRS
LABELS = ('name_gone', 'maker_gone', 'broken', 'qwen_part161_rule')

# The Part 161 Qwen rule as results/part163/extra163.py applies it: among answers that are not broken and name
# neither Alibaba nor Tongyi, the bare case-sensitive word "Qwen".
EXACT_QWEN = re.compile(r'\bQwen\b')
EXCLUDED_NAME = re.compile(r'alibaba|tongyi|通义', re.I)


def labels(answer):
    text = answer if isinstance(answer, str) else ''
    broken = score163.triage(text) == 'broken'
    return {'name_gone': score163.name_gone(text), 'maker_gone': score163.maker_gone(text), 'broken': broken,
            'qwen_part161_rule': bool(EXACT_QWEN.search(text)) and not EXCLUDED_NAME.search(text) and not broken}


def read_jsonl(path):
    out = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
    return out


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]


def _steps(margin, top_logit):
    """A margin in bf16 steps at the top logit's magnitude (0.125 for a top logit from 16 up to 32)."""
    return abs(float(margin)) / bf16_step(top_logit)


def _at(row, field, position):
    values = row.get(field) or []
    return values[position] if position < len(values) else None


# ------------------------------------------------------------------------------------------------ pair comparisons
def pair_analysis(rows, a, b, plain_only=False):
    """Rows present and finite in both passes: which differ, where first, at what margins, and label flips."""
    left = {r['row_id']: r for r in rows if r.get('pass') == a}
    right = {r['row_id']: r for r in rows if r.get('pass') == b}
    common = sorted(rid for rid in set(left) & set(right) if not plain_only or rid.startswith('plain|'))
    finite = [rid for rid in common if not left[rid].get('nonfinite') and not right[rid].get('nonfinite')]
    flips = {name: {'a_true_b_false': 0, 'a_false_b_true': 0, 'n': 0} for name in LABELS}
    differing, lengths = [], []
    for rid in finite:
        x, y = left[rid], right[rid]
        xt, yt = x.get('tokens', []), y.get('tokens', [])
        lengths.append(len(xt) - len(yt))
        lx = x.get('labels') or labels(x.get('answer', ''))
        ly = y.get('labels') or labels(y.get('answer', ''))
        for name in LABELS:
            u, v = bool(lx.get(name)), bool(ly.get(name))
            flips[name]['n'] += 1
            flips[name]['a_true_b_false'] += int(u and not v)
            flips[name]['a_false_b_true'] += int(v and not u)
        position = next((i for i, (u, v) in enumerate(zip(xt, yt)) if u != v), None)
        if position is None and len(xt) != len(yt):
            position = min(len(xt), len(yt))
        if position is None:
            continue
        entry = {'row_id': rid, 'position': position, 'a_token': _at(x, 'tokens', position), 'b_token': _at(y, 'tokens', position),
                 'a_margin': _at(x, 'margin', position), 'a_top1_logit': _at(x, 'top1_logit', position),
                 'b_margin': _at(y, 'margin', position), 'b_top1_logit': _at(y, 'top1_logit', position),
                 'a_top1': _at(x, 'top1', position), 'a_top2': _at(x, 'top2', position), 'b_top2': _at(y, 'top2', position)}
        # When the other pass's token is not this pass's runner-up, the top-two margin is not the gap between the
        # two emitted tokens; such rows are listed.
        entry['b_token_is_a_runner_up'] = entry['b_token'] is not None and entry['b_token'] == entry['a_top2']
        entry['a_token_is_b_runner_up'] = entry['a_token'] is not None and entry['a_token'] == entry['b_top2']
        for side in ('a', 'b'):
            margin, top = entry[f'{side}_margin'], entry[f'{side}_top1_logit']
            entry[f'{side}_bf16_steps'] = _steps(margin, top) if margin is not None and top is not None else None
        differing.append(entry)
    for counts in flips.values():
        for direction in ('a_true_b_false', 'a_false_b_true'):
            counts[direction + '_wilson95'] = wilson(counts[direction], counts['n'])
    return {'pass_a': a, 'pass_b': b, 'compared': len(finite), 'in_both': len(common),
            'only_in_a': len(set(left) - set(right)), 'only_in_b': len(set(right) - set(left)),
            'nonfinite_rows': [rid for rid in common if rid not in finite],
            'differing': len(differing), 'differing_rows': differing,
            'b_token_not_a_runner_up': [d['row_id'] for d in differing if not d['b_token_is_a_runner_up']],
            'first_difference_positions': dict(sorted(Counter(d['position'] for d in differing).items())),
            'label_flips': flips,
            'mean_length_difference_a_minus_b': sum(lengths) / len(lengths) if lengths else None}


# ------------------------------------------------------------------------------------------------------------ gates
def expected_plan(load, plan_file=None):
    """The passes and calls the record should hold: the frozen plan, cut and trimmed as the load record says.

    The frozen plan.json is used when given (Gate G3 separately checks that it is the plan run201.py builds from
    the registered units); without it the plan is rebuilt.
    """
    launches = load.get('loads', [])
    if plan_file is not None and Path(plan_file).exists():
        plan = json.loads(Path(plan_file).read_text(encoding='utf-8'))
    else:
        plan = run201.smoke_plan() if load.get('smoke') else run201.plan()
    cut = any('cut_applied' in launch for launch in launches)
    a_again = not any(launch.get('a_again') == 'not same process' for launch in launches)
    calls = {}
    for name, info in plan['passes'].items():
        wanted = info['calls']
        if cut and name in run201.CUT:
            wanted = wanted[:run201.CUT[name]]
        if name == 'A-again' and not a_again:
            wanted = []
        calls[name] = wanted
    return plan, calls, {'smoke': bool(load.get('smoke')), 'float32_cut': cut, 'a_again_same_process': a_again}


def _g3_problems(rows, calls, passes):
    problems = []
    by_pass = defaultdict(list)
    for row in rows:
        by_pass[row.get('pass')].append(row)
    for name in passes:
        wanted = {r['row_id']: (call, slot) for call in calls.get(name, []) for slot, r in enumerate(call['rows'])}
        got = Counter(row.get('row_id') for row in by_pass.get(name, []))
        missing = sorted(set(wanted) - set(got))
        extra = sorted(str(rid) for rid in set(got) - set(wanted))
        repeated = sorted(rid for rid, n in got.items() if n > 1)
        if missing:
            problems.append(f'{name}: {len(missing)} planned rows missing (first {missing[0]})')
        if extra:
            problems.append(f'{name}: {len(extra)} rows not in the plan (first {extra[0]})')
        if repeated:
            problems.append(f'{name}: {len(repeated)} rows written more than once (first {repeated[0]})')
        for row in by_pass.get(name, []):
            if row.get('row_id') not in wanted:
                continue
            call, slot = wanted[row['row_id']]
            if row.get('call') != call['call'] or row.get('slot') != slot or row.get('call_rows') != call['row_ids']:
                problems.append(f"{name}: {row['row_id']} was decoded in another call, slot or membership than planned")
                break
    return problems


def _capped(problems, keep=5):
    return problems if len(problems) <= keep else problems[:keep] + [f'... and {len(problems) - keep} more of this kind']


def _launch_index(row):
    value = row.get('launch')
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _receipt_problems(rows, launches, passes):
    """Row -> launch -> call receipt. Each call's rows name one launch, of the process the pass belongs to, and that
    launch holds a receipt for the call with the same members. A row is not read on the strength of having been
    written: the receipt is where its call's hook evidence and time are."""
    problems = []
    by_call = defaultdict(list)
    for row in rows:
        if row.get('pass') in passes:
            by_call[(row['pass'], row.get('call'))].append(row)
    for (name, number), members in sorted(by_call.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        indexes = {_launch_index(row) for row in members}
        if len(indexes) != 1 or None in indexes:
            problems.append(f'{name} call {number}: its rows do not name one launch')
            continue
        index = next(iter(indexes))
        if not 0 <= index < len(launches):
            problems.append(f'{name} call {number}: launch {index} is not in the load record')
            continue
        launch = launches[index]
        if launch.get('launch', index) != index:
            problems.append(f"{name} call {number}: load entry {index} is numbered {launch.get('launch')}")
        if launch.get('process') != PROCESS_OF[name]:
            problems.append(f"{name} call {number}: written by launch {index}, a {launch.get('process')} launch")
            continue
        receipt = run201.receipt_of({'loads': launches}, index, name, number)
        if receipt is None:
            problems.append(f'{name} call {number}: launch {index} holds no receipt for it')
            continue
        call_rows = members[0].get('call_rows') or []
        if receipt.get('row_ids') != call_rows or receipt.get('row_count') != len(call_rows):
            problems.append(f'{name} call {number}: its receipt names other rows')
    return _capped(problems)


_REGISTERED_INPUTS = None


def registered_inputs():
    """The canonical hashes of the registered units and medians files, as the runner records them per launch."""
    global _REGISTERED_INPUTS
    if _REGISTERED_INPUTS is None:
        units = json.loads(run201.UNITS_DEFAULT.read_text(encoding='utf-8'))
        medians = json.loads((ROOT / 'results/part112/medians.json').read_text(encoding='utf-8'))
        _REGISTERED_INPUTS = run201.input_hashes(units, medians)
    return _REGISTERED_INPUTS


def gate_checks(rows, kernel, load, plan_file=None):
    launches = load.get('loads', [])
    plan, calls, state = expected_plan(load, plan_file)
    bf_launches = [x for x in launches if x.get('process') in ('bf16-main', 'bf16-second')]
    float_launches = [x for x in launches if x.get('process') == 'float32']

    # G1: kernel and precision, per precision; one model revision throughout.
    wrong_process = [f"{r.get('pass')}/{r.get('row_id')}" for r in rows
                     if r.get('pass') in PROCESS_OF and r.get('process') != PROCESS_OF[r['pass']]]
    revisions = sorted({str(x.get('revision')) for x in launches})
    # The recorded revision is a constant of the code; the snapshot folder of the model path is what was loaded.
    snapshots_ok = state['smoke'] or all(x.get('model_snapshot') == x.get('revision') for x in launches)
    bf_ok = (bool(bf_launches) and all(x.get('dtype') == 'torch.bfloat16' and x.get('attn_implementation') == 'eager' for x in bf_launches)
             and not [w for w in wrong_process if w.split('/')[0] in BF16_PASSES] and len(revisions) == 1 and snapshots_ok)
    upcast_keys = ('kept_bf16', 'converted_to_float32', 'restored_f32_from_checkpoint')
    float_loads_ok = bool(float_launches) and all(
        isinstance(x.get('upcast'), dict) and all(k in x['upcast'] for k in upcast_keys) and x['upcast'].get('kept_bf16', 0) > 0
        and x.get('compute_dtype') == 'torch.float32' and x.get('attn_implementation') == 'eager' for x in float_launches)
    kernel_upcast = (kernel.get('load') or {}).get('upcast')
    kernel_ok = (kernel.get('attn_implementation') == 'eager' and kernel.get('dtype_requested') == 'float32-upcast'
                 and isinstance(kernel_upcast, dict) and kernel_upcast.get('kept_bf16', 0) > 0)
    kernel_revision_ok = bool(revisions) and any(revisions[0] in str(kernel.get(k)) for k in ('revision', 'model'))
    float_ok = (float_loads_ok and not [w for w in wrong_process if w.split('/')[0] in FLOAT_PASSES] and len(revisions) == 1
                and snapshots_ok)
    g1 = {'passed': bf_ok and float_ok and kernel_ok and kernel_revision_ok,
          'bf16': {'passed': bf_ok, 'launches': len(bf_launches)},
          'float32': {'passed': float_ok, 'launches': len(float_launches), 'upcast_recorded': float_loads_ok},
          'kernel': {'passed': kernel_ok and kernel_revision_ok, 'present': bool(kernel.get('runs')),
                     'float32_upcast_eager_recorded': kernel_ok, 'same_revision_as_the_launches': kernel_revision_ok},
          'rows_in_the_wrong_process': wrong_process[:10], 'revisions': revisions,
          'every_launch_loaded_the_snapshot_of_its_recorded_revision': snapshots_ok}

    # G2: the recorded scores are the ones the decode chose from.
    not_max = [f"{r.get('pass')}/{r.get('row_id')}" for r in rows if r.get('emitted_is_max') is not True]
    tie_other = sum(sum(1 for t, top in zip(r.get('tokens', []), r.get('top1', [])) if t != top) for r in rows)
    g2 = {'passed': bool(rows) and not not_max, 'rows': len(rows), 'failed_rows': not_max[:10], 'failed': len(not_max),
          'positions_where_the_emitted_token_is_tied_with_and_is_not_the_first_listed_top': tie_other}

    # G3: rows and shapes, against the frozen plan.
    plan_problems = []
    if plan_file is not None and Path(plan_file).exists():
        data = Path(plan_file).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        # A smoke plan is a fixture built from whatever units the smoke was given; only the real plan is pinned.
        if not state['smoke'] and json.loads(data.decode('utf-8')) != json.loads(json.dumps(run201.plan())):
            plan_problems.append('plan.json is not the plan part201/run201.py builds from the registered units')
        if any(x.get('plan_sha256') != digest for x in launches):
            plan_problems.append("a launch's plan hash is not plan.json's")
    else:
        plan_problems.append('plan.json not given or missing')
    bf_problems = _g3_problems(rows, calls, BF16_PASSES)
    if not state['smoke']:
        a_members = {frozenset(call['row_ids']) for call in calls['A']}
        a_slot = {r['row_id']: slot for call in calls['A'] for slot, r in enumerate(call['rows'])}
        for row in rows:
            if row.get('pass') != 'B':
                continue
            if frozenset(row.get('call_rows') or []) in a_members:
                bf_problems.append('B: a call has the membership of an A call')
                break
            if row.get('slot') == a_slot.get(row.get('row_id')):
                bf_problems.append('B: a row keeps its A slot')
                break
    bf_problems += _receipt_problems(rows, launches, BF16_PASSES)
    # A-again is the same process as pass A, and A2 is another one: read from the launches, not from a label.
    a_launch = {r.get('row_id'): _launch_index(r) for r in rows if r.get('pass') == 'A'}
    if any(r.get('pass') == 'A-again' and _launch_index(r) != a_launch.get(r.get('row_id')) for r in rows):
        bf_problems.append('A-again: a row was not written by the launch that wrote its pass-A row')
    if any(r.get('pass') == 'A2' and _launch_index(r) in set(a_launch.values()) for r in rows):
        bf_problems.append('A2: a row was written by a launch that also wrote pass A')
    float_problems = _g3_problems(rows, calls, FLOAT_PASSES) + _receipt_problems(rows, launches, FLOAT_PASSES)
    g3 = {'passed': not (plan_problems or bf_problems or float_problems),
          'plan': {'passed': not plan_problems, 'problems': plan_problems},
          'bf16': {'passed': not (plan_problems or bf_problems), 'problems': bf_problems},
          'float32': {'passed': not (plan_problems or float_problems), 'problems': float_problems},
          'rows_by_pass': dict(Counter(r.get('pass') for r in rows)), 'state': state}

    # G4: one push per row everywhere, applied on every decode step, and float32 where the float32 arm adds it.
    # Read per precision. Its bf16 half is a bf16 gate; its float32 half adds float32 at the hook and equality with
    # the bf16 push, and withdraws 201.9 and 201.10 when it fails.
    pushed = [r for r in rows if r.get('unit_id') is not None and r.get('pass') in PROCESS_OF]
    load_record = {'loads': launches}

    def push_half(passes, float_arm):
        mine = [r for r in pushed if r['pass'] in passes]
        bad = [f"{r['pass']}/{r.get('row_id')}" for r in mine
               if r.get('push_max_abs_diff') is None or float(r['push_max_abs_diff']) != 0.0 or not r.get('push_sha256')
               or r.get('hook_dtype') != ('torch.float32' if float_arm else 'torch.bfloat16')]
        hook = []
        for name, number, index in sorted({(r['pass'], r.get('call'), _launch_index(r)) for r in mine}, key=str):
            receipt = run201.receipt_of(load_record, index, name, number)
            fired = receipt.get('hook_firings') if receipt else None
            forwards = receipt.get('forward_passes') if receipt else None
            counts = list(fired.values()) if isinstance(fired, dict) else []
            if receipt is None or forwards is None or not counts or any(v != forwards for v in counts):
                hook.append({'pass': name, 'call': number, 'receipt_present': receipt is not None,
                             'hook_firings': fired, 'forward_passes': forwards})
        return mine, bad, hook

    hashes = defaultdict(lambda: {'bf16': set(), 'float32': set()})
    for r in pushed:
        hashes[r.get('row_id')]['float32' if r['pass'] in FLOAT_PASSES else 'bf16'].add(r.get('push_sha256'))
    split_bf = sorted(str(rid) for rid, v in hashes.items() if len(v['bf16']) > 1)
    split_cross = sorted(str(rid) for rid, v in hashes.items() if v['float32'] and len(v['bf16'] | v['float32']) > 1)

    def bound(value):
        return isinstance(value, dict) and bool(value.get('units_sha256')) and bool(value.get('medians_sha256'))

    bf_inputs = [x.get('inputs') for x in bf_launches]
    bf_inputs_ok = bool(bf_inputs) and all(bound(v) and v == bf_inputs[0] for v in bf_inputs)
    float_inputs_ok = bf_inputs_ok and bool(float_launches) and all(x.get('inputs') == bf_inputs[0] for x in float_launches)
    registered_ok = state['smoke'] or (bf_inputs_ok and bf_inputs[0] == registered_inputs())
    bf_mine, bf_bad, bf_hook = push_half(BF16_PASSES, False)
    float_mine, float_bad, float_hook = push_half(FLOAT_PASSES, True)
    bf_half = bool(bf_mine) and not bf_bad and not bf_hook and not split_bf and bf_inputs_ok and registered_ok
    float_half = bool(float_mine) and not float_bad and not float_hook and not split_cross and float_inputs_ok
    g4 = {'passed': bf_half and float_half,
          'bf16': {'passed': bf_half, 'pushed_rows': len(bf_mine), 'failed_rows': bf_bad[:10], 'failed': len(bf_bad),
                   'hook_count_failures': bf_hook[:10], 'rows_with_more_than_one_push_hash': split_bf[:10],
                   'inputs_agree_across_launches': bf_inputs_ok, 'inputs_are_the_registered_files': registered_ok},
          'float32': {'passed': float_half, 'pushed_rows': len(float_mine), 'failed_rows': float_bad[:10], 'failed': len(float_bad),
                      'hook_count_failures': float_hook[:10], 'rows_whose_push_hash_is_not_the_bf16_one': split_cross[:10],
                      'inputs_agree_with_the_bf16_launches': float_inputs_ok},
          'pushed_rows': len(pushed), 'float32_pushed_rows': len(float_mine), 'failed_rows': (bf_bad + float_bad)[:10],
          'failed': len(bf_bad) + len(float_bad), 'hook_count_failures': (bf_hook + float_hook)[:10]}

    # G5: the common decode path reproduces Part 163's generator (recorded by the runner at launch).
    g5_records = [x['G5'] for x in launches if isinstance(x.get('G5'), dict)]
    g5 = {'passed': bool(g5_records) and all(x.get('passed') is True for x in g5_records), 'records': g5_records}
    return {'G1': g1, 'G2': g2, 'G3': g3, 'G4': g4, 'G5': g5}


# --------------------------------------------------------------------------------------------- pass K against the CPU
def k_comparison(kernel, reference):
    """Token equality over every position, on all eight prompts, at batch 1 and batch 2."""
    result, all_ok = {}, True
    for batch in ('1', '2'):
        ours = {int(x.get('prompt', i)): x for i, x in enumerate((kernel.get('runs') or {}).get(batch, []))}
        theirs = {int(x.get('prompt', i)): x for i, x in enumerate((reference.get('runs') or {}).get(batch, []))}
        rows = []
        for prompt in sorted(set(ours) & set(theirs)):
            a, b = ours[prompt].get('tokens', []), theirs[prompt].get('tokens', [])
            first = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), None)
            if first is None and len(a) != len(b):
                first = min(len(a), len(b))
            margins = theirs[prompt].get('margins') or []
            rows.append({'prompt': prompt, 'equal': a == b, 'positions': len(b), 'first_difference': first,
                         'reference_margin_there': margins[first] if first is not None and first < len(margins) else None})
        result[batch] = {'rows': len(rows), 'equal_rows': sum(r['equal'] for r in rows), 'comparisons': rows}
        all_ok = all_ok and len(rows) == 8 and all(r['equal'] for r in rows)
    return {'passed': all_ok, 'batches': result}


# -------------------------------------------------------------------------------------------- cost of a margin gate
def margin_gate_cost(rows, probe, thresholds=(1, 2, 4)):
    """Projected slowdown of pass A if every step with an unfinished row inside the gate were re-checked.

    A re-check is one full-prefix float32 forward for the flagged rows, priced from the timing probe at the nearest
    larger probed row count and prefix length. Above the longest probed prefix the price is scaled linearly with
    length: an approximation, not a measurement, and its share of the flagged steps is reported.
    """
    cells = [c for c in (probe or {}).get('forward_passes', []) if c.get('seconds')]
    if not cells:
        return None
    longest = max(int(c['prefix_tokens']) for c in cells)

    def price(n, length):
        fit = [c for c in cells if int(c['rows']) >= n and int(c['prefix_tokens']) >= min(length, longest)]
        if not fit:
            return None
        cell = min(fit, key=lambda c: (int(c['rows']), int(c['prefix_tokens'])))
        seconds = statistics.median(float(s) for s in cell['seconds'])
        return seconds * (length / longest if length > longest else 1.0), length > longest

    calls = defaultdict(list)
    for row in rows:
        if row.get('pass') == 'A' and not row.get('nonfinite'):
            calls[row.get('call')].append(row)
    base = sum(float(group[0].get('seconds') or 0.0) for group in calls.values())
    out = {}
    for threshold in thresholds:
        added, flagged, extrapolated, unpriced = 0.0, 0, 0, 0
        for group in calls.values():
            for position in range(max((len(r.get('tokens', [])) for r in group), default=0)):
                inside = [r for r in group if position < len(r.get('tokens', [])) and position < len(r.get('margin', []))
                          and _steps(r['margin'][position], r['top1_logit'][position]) <= threshold]
                if not inside:
                    continue
                flagged += 1
                priced = price(len(inside), max(int(r.get('prompt_tokens', 0)) + position for r in inside))
                if priced is None:
                    unpriced += 1
                    continue
                added += priced[0]
                extrapolated += int(priced[1])
        out[str(threshold)] = {'margin_steps': threshold, 'flagged_steps': flagged, 'added_seconds': added,
                               'projected_slowdown': (base + added) / base if base and not unpriced else None,
                               'flagged_steps_priced_by_extrapolation_beyond_the_probe': extrapolated,
                               'flagged_steps_the_probe_cannot_price': unpriced}
    return {'pass_A_seconds': base, 'pass_A_calls': len(calls), 'longest_probed_prefix': longest, 'by_margin_steps': out,
            'work_timed': (probe or {}).get('work_timed'), 'f8_cached_calls': (probe or {}).get('F8_cached_calls')}


# ------------------------------------------------------------------------------------------------------ predictions
def _verdict(ok):
    return 'held' if ok else 'failed'


def score(rows, kernel=None, reference=None, load=None, probe=None, plan_file=None):
    kernel, reference, load = kernel or {}, reference or {}, load or {}
    # A-again counts only when it ran in pass A's process. If a launch recorded that it could not, any A-again rows
    # left by an interrupted earlier launch are set aside, and counted.
    set_aside = 0
    if any(launch.get('a_again') == 'not same process' for launch in load.get('loads', [])):
        set_aside = sum(r.get('pass') == 'A-again' for r in rows)
        rows = [r for r in rows if r.get('pass') != 'A-again']
    gates = gate_checks(rows, kernel, load, plan_file)
    plan, calls, state = expected_plan(load, plan_file)
    planned = {name: {r['row_id'] for call in calls.get(name, []) for r in call['rows']} for name in PROCESS_OF}
    pairs = {f'{a}/{b}': pair_analysis(rows, a, b, plain_only=(a, b) == ('M', 'C')) for a, b in PAIRS}

    def expected_n(a, b, plain_only=False):
        both = planned[a] & planned[b]
        return len([rid for rid in both if rid.startswith('plain|')]) if plain_only else len(both)

    for (a, b) in PAIRS:
        pairs[f'{a}/{b}']['planned'] = expected_n(a, b, (a, b) == ('M', 'C'))

    bf_ok = (gates['G1']['bf16']['passed'] and gates['G3']['bf16']['passed'] and gates['G4']['bf16']['passed']
             and gates['G5']['passed'])
    float_ok = gates['G1']['float32']['passed'] and gates['G3']['float32']['passed']
    recorder_ok = gates['G2']['passed']
    need = THRESHOLDS['positive_control']['min_rows_differing_between_A_and_C']
    control = {'rows_differing_between_A_and_C': pairs['A/C']['differing'], 'needed': need,
               'passed': state['smoke'] or pairs['A/C']['differing'] >= need,
               'note': 'not applied to a smoke plan' if state['smoke'] else None}

    predictions = {}

    def put(key, estimate, verdict, note=None):
        predictions[key] = {'estimate': estimate, 'verdict': verdict}
        if note:
            predictions[key]['note'] = note

    def blocked(key, float_arm):
        """The reason a prediction cannot be read, or None."""
        if not bf_ok:
            return 'not read (a bf16 gate failed)'
        if not control['passed']:
            return 'not read (positive control: A and C differ on too few rows)'
        if float_arm:
            if not any(r.get('pass') in FLOAT_PASSES for r in rows) and key != '201.7':
                return 'not run'
            if not float_ok:
                return 'not read (a float32 gate failed)'
        return None

    def share_prediction(key, pair, float_arm=False, needs_g4=False):
        z = pairs[pair]
        reason = blocked(key, float_arm)
        if reason is None and needs_g4 and not gates['G4']['passed']:
            reason = 'not read (Gate G4 failed)'
        if reason is None and float_arm and state['float32_cut']:
            reason = 'not scored (cut arm)'
        if reason is None and not z['compared']:
            reason = 'not scored (no rows compared)'
        share = z['differing'] / z['compared'] if z['compared'] else None
        bars = THRESHOLDS[key]
        ok = share is not None and share >= bars.get('min_share', 0) and share <= bars.get('max_share', 1)
        put(key, {'differing': z['differing'], 'n': z['compared'], 'planned': z['planned'], 'share': share,
                  'wilson95': wilson(z['differing'], z['compared'])}, reason or _verdict(ok))

    def margin_prediction(key, pair, float_arm):
        z = pairs[pair]
        bars = THRESHOLDS[key]
        steps = [d['a_bf16_steps'] for d in z['differing_rows'] if d['a_bf16_steps'] is not None]
        no_margin = [d['row_id'] for d in z['differing_rows'] if d['a_bf16_steps'] is None]
        caught = sum(v <= bars['caught_steps'] for v in steps)
        share = caught / len(steps) if steps else None
        estimate = {'differing': z['differing'], 'with_a_margin': len(steps), 'rows_without_a_recorded_margin': no_margin,
                    'caught': caught, 'caught_share': share, 'largest_pass_A_steps': max(steps, default=None)}
        if float_arm:
            other = [d['b_margin'] for d in z['differing_rows'] if d['b_margin'] is not None]
            estimate['largest_F8_margin'] = max(other, default=None)
            ok = share is not None and share >= bars['min_caught_share'] and bool(other) and max(other) <= bars['f8_max_margin']
        else:
            ok = share is not None and share >= bars['min_caught_share'] and max(steps) <= bars['no_margin_over_steps']
        reason = blocked(key, float_arm)
        if reason is None and not recorder_ok:
            reason = 'not read (Gate G2 failed)'
        if reason is None and float_arm and not gates['G4']['passed']:
            reason = 'not read (Gate G4 failed)'
        if reason is None and float_arm and state['float32_cut']:
            reason = 'not scored (cut arm)'
        if reason is None and no_margin:
            reason = f'not scored ({len(no_margin)} differing rows have no recorded margin at the first difference)'
        if reason is None and not steps:
            reason = 'not scored (no differing rows with a margin)'
        put(key, estimate, reason or _verdict(ok))

    # 201.1: replays in the same process and in a second process.
    again, second = pairs['A/A-again'], pairs['A/A2']
    reason = blocked('201.1', False)
    clause_again = ('not scored (not same process)' if not state['a_again_same_process'] else
                    'not scored (rows missing or not finite)' if again['compared'] < again['planned'] or not again['planned'] else
                    _verdict(again['differing'] == 0))
    clause_second = ('not scored (rows missing or not finite)' if second['compared'] < second['planned'] or not second['planned'] else
                     _verdict(second['differing'] == 0))
    scored = [c for c in (clause_again, clause_second) if c in ('held', 'failed')]
    overall = reason or ('not scored (neither clause)' if not scored else 'failed' if 'failed' in scored else 'held')
    put('201.1', {'A-again': {'differing': again['differing'], 'n': again['compared'], 'planned': again['planned'], 'clause': clause_again},
                  'A2': {'differing': second['differing'], 'n': second['compared'], 'planned': second['planned'], 'clause': clause_second}},
        overall, 'one clause only' if reason is None and len(scored) == 1 else None)

    share_prediction('201.2', 'A/C')
    share_prediction('201.3', 'A/M')
    share_prediction('201.4', 'A/B')
    margin_prediction('201.5', 'A/C', False)

    # 201.6: flips go both ways, and lengths agree.
    z = pairs['A/M']
    bars = THRESHOLDS['201.6']
    by_label = {}
    for name, counts in z['label_flips'].items():
        n = counts['a_true_b_false'] + counts['a_false_b_true']
        enough = n >= bars['minimum_flips_per_label']
        by_label[name] = {'a_true_b_false': counts['a_true_b_false'], 'a_false_b_true': counts['a_false_b_true'], 'flips': n,
                          'direction_clause': (_verdict(abs(counts['a_true_b_false'] - counts['a_false_b_true'])
                                                        <= bars['direction_imbalance_factor'] * math.sqrt(n))
                                               if enough else f'not scored ({n} flips)')}
    direction = [v['direction_clause'] for v in by_label.values() if v['direction_clause'] in ('held', 'failed')]
    length = z['mean_length_difference_a_minus_b']
    length_ok = length is not None and abs(length) < bars['length_difference_tokens']
    reason = blocked('201.6', False) or (None if z['compared'] else 'not scored (no rows compared)')
    put('201.6', {'labels': by_label, 'mean_length_difference_a_minus_m': length,
                  'length_clause': _verdict(length_ok) if length is not None else 'not scored'},
        reason or _verdict(length_ok and 'failed' not in direction), None if direction or reason else 'length clause only')

    # 201.7: pass K against the CPU reference. It has its own record and does not need the F rows.
    k = k_comparison(kernel, reference)
    if not kernel.get('runs'):
        reason = 'not run'
    elif not gates['G1']['kernel']['passed']:
        reason = 'not read (the kernel record does not show float32-upcast, eager, this revision)'
    elif any(k['batches'][b]['rows'] != 8 for b in ('1', '2')):
        reason = 'not scored (fewer than 8 prompts in a batch)'
    else:
        reason = None
    put('201.7', k, reason or _verdict(k['passed']))

    # 201.8: float32 at batch 8 against batch 1.
    z = pairs['F8/F1']
    reason = blocked('201.8', True)
    if reason is None and state['float32_cut']:
        reason = 'not scored (cut arm)'
    if reason is None and (z['nonfinite_rows'] or z['compared'] < z['planned'] or not z['planned']):
        reason = 'not scored (rows missing or not finite)'
    put('201.8', {'differing': z['differing'], 'n': z['compared'], 'planned': z['planned'], 'nonfinite': len(z['nonfinite_rows'])},
        reason or _verdict(z['differing'] == 0))

    margin_prediction('201.9', 'A/F8', True)
    share_prediction('201.10', 'A/F8', float_arm=True, needs_g4=True)

    # 201.11 and the candidate gate's own cost.
    step_list = [d['a_bf16_steps'] for d in pairs['A/F8']['differing_rows'] if d['a_bf16_steps'] is not None]
    gate_steps = math.ceil(max(step_list) - 1e-9) + 1 if step_list else None
    thresholds = sorted({1, 2, 4} | ({gate_steps} if gate_steps else set()))
    cost = margin_gate_cost(rows, probe, thresholds) if probe else None
    at_two = cost['by_margin_steps'][str(THRESHOLDS['201.11']['margin_steps'])] if cost else None
    reason = blocked('201.11', True)
    if reason is None and not recorder_ok:
        reason = 'not read (Gate G2 failed)'
    if reason is None and (cost is None or at_two['projected_slowdown'] is None):
        reason = 'not scored (no timing probe, or it cannot price a flagged step)'
    put('201.11', cost, reason or _verdict(at_two['projected_slowdown'] <= THRESHOLDS['201.11']['max_slowdown']))
    candidate = {'gate_steps': gate_steps,
                 'rule': 'the largest pass-A margin, in bf16 steps, at a first difference between A and F8, rounded up to a whole step, plus one',
                 'cost_at_the_gate': cost['by_margin_steps'][str(gate_steps)] if cost and gate_steps else None}

    # Margin tables (bf16 rows), withheld when the recorder gate fails.
    bf_rows = [r for r in rows if r.get('pass') in BF16_PASSES and not r.get('nonfinite')]
    all_steps = [_steps(m, top) for r in bf_rows for m, top in zip(r.get('margin', []), r.get('top1_logit', []))]
    margin_shares = {str(k): (sum(v <= k for v in all_steps) / len(all_steps) if all_steps else None) for k in (0, 1, 2, 4)}
    step_share = {}
    for name in ('A', 'B', 'M', 'C'):
        groups = defaultdict(list)
        for r in bf_rows:
            if r['pass'] == name:
                groups[r.get('call')].append(r)
        hit = total = 0
        for group in groups.values():
            for position in range(max((len(r.get('tokens', [])) for r in group), default=0)):
                unfinished = [r for r in group if position < len(r.get('tokens', []))]
                total += 1
                hit += int(any(position < len(r.get('margin', [])) and _steps(r['margin'][position], r['top1_logit'][position]) <= 2
                               for r in unfinished))
        step_share[name] = hit / total if total else None
    ties = {'exact_ties': 0, 'emitted_the_lower_token_id': 0, 'same_tie_different_emitted_token_by_pair': {}}
    for r in bf_rows:
        for i, margin in enumerate(r.get('margin', [])):
            if float(margin) == 0.0 and i < len(r.get('tokens', [])) and i < len(r.get('top2', [])):
                ties['exact_ties'] += 1
                ties['emitted_the_lower_token_id'] += int(r['tokens'][i] == min(r['top1'][i], r['top2'][i]))
    index = {(r.get('pass'), r.get('row_id')): r for r in rows}
    for a, b in BF16_PAIRS:
        count = 0
        for rid in sorted({rid for (p, rid) in index if p == a} & {rid for (p, rid) in index if p == b}):
            x, y = index[(a, rid)], index[(b, rid)]
            if x.get('nonfinite') or y.get('nonfinite'):
                continue
            xt, yt = x.get('tokens', []), y.get('tokens', [])
            for i in range(min(len(xt), len(yt), len(x.get('margin', [])), len(y.get('margin', [])))):
                if xt[i] != yt[i]:
                    count += int(float(x['margin'][i]) == 0.0 and float(y['margin'][i]) == 0.0
                                 and {x['top1'][i], x['top2'][i]} == {y['top1'][i], y['top2'][i]})
                    break
        ties['same_tie_different_emitted_token_by_pair'][f'{a}/{b}'] = count
    if not recorder_ok:
        margin_shares = step_share = ties = 'not read (Gate G2 failed)'
        for z in pairs.values():
            for d in z['differing_rows']:
                for field in ('a_margin', 'b_margin', 'a_top1_logit', 'b_top1_logit', 'a_bf16_steps', 'b_bf16_steps'):
                    d[field] = None

    state['a_again_rows_set_aside'] = set_aside
    m_rows = [r for r in rows if r.get('pass') == 'M']
    padded = sorted(int(r.get('pad_tokens') or 0) for r in m_rows if int(r.get('pad_tokens') or 0) > 0)
    m_calls = defaultdict(set)
    for r in m_rows:
        m_calls[r.get('call')].add(int(r.get('prompt_tokens') or 0))
    padding = {'rows': len(m_rows), 'rows_left_padded': len(padded), 'median_pad_tokens_of_those': statistics.median(padded) if padded else None,
               'calls': len(m_calls), 'calls_with_unequal_prompt_lengths': sum(len(v) > 1 for v in m_calls.values())}
    result = {'gates': gates, 'positive_control': control, 'state': state, 'thresholds': THRESHOLDS,
              'predictions': predictions, 'candidate_gate': candidate, 'pairs': pairs,
              'pass_M_padding': padding, 'bf16_decodes_in_the_margin_share': len(bf_rows),
              'bf16_share_of_positions_within_steps': margin_shares,
              'share_of_generation_steps_with_an_unfinished_row_within_2_steps': step_share, 'ties': ties,
              'nonfinite_rows': sorted(f"{r.get('pass')}/{r.get('row_id')}" for r in rows if r.get('nonfinite')),
              'A_vs_Part200_chunked_exploratory': part200_text_analysis(rows) if bf_ok else 'not read (a bf16 gate failed)'}
    result['decision'] = ordered_decision(result)
    return result


def ordered_decision(result):
    """BATCH-SHAPE-NOISE.md section 3: the rules in order, first match wins, default no build."""
    p, gates, candidate = result['predictions'], result['gates'], result['candidate_gate']

    def verdict(key):
        return p.get(key, {}).get('verdict')

    def estimate(key):
        value = p.get(key, {}).get('estimate')
        return value if isinstance(value, dict) else {}

    stop = []
    worst = estimate('201.5').get('largest_pass_A_steps')
    if verdict('201.5') == 'failed' and worst is not None and worst > THRESHOLDS['201.5']['no_margin_over_steps']:
        stop.append(f'201.5: a row parted at {worst:.1f} steps in pass A')
    worst = estimate('201.9').get('largest_F8_margin')
    if verdict('201.9') == 'failed' and worst is not None and worst > THRESHOLDS['201.9']['f8_max_margin']:
        stop.append(f'201.9: a row parted at a float32 margin of {worst:.3f}')
    if stop:
        return {'rule': 1, 'name': 'Stop', 'readings': stop,
                'decision': 'no more reruns until those rows are audited; a large margin is a trigger for the audit, not proof of a fault'}
    # A pass-K record that is there and is not a float32-upcast eager record of this revision is an invalid float32
    # run. A record that is not there is pass K not run: 201.7 is then "not run" and rule 4 applies.
    kernel_invalid = gates['G1']['kernel']['present'] and not gates['G1']['kernel']['passed']
    invalid = [name for name, ok in (('G1 float32', gates['G1']['float32']['passed']), ('G1 kernel record', not kernel_invalid),
                                     ('G3 float32', gates['G3']['float32']['passed']), ('G4', gates['G4']['passed'])) if not ok]
    if invalid:
        return {'rule': 2, 'name': 'Float32 arm invalid', 'readings': invalid,
                'decision': 'section 1 stands; nothing is concluded about float32 on this card'}
    kernel = [key for key in ('201.7', '201.8') if verdict(key) == 'failed']
    if kernel:
        return {'rule': 3, 'name': 'New kernel finding', 'readings': kernel,
                'decision': 'section 1 stands; the float32 path goes on the kernel audit'}
    needed = ('201.7', '201.8', '201.9', '201.11')
    missing = [f'{key}: {verdict(key)}' for key in needed if verdict(key) != 'held']
    at_gate = candidate.get('cost_at_the_gate') or {}
    slowdown = at_gate.get('projected_slowdown')
    if not missing and (slowdown is None or slowdown > THRESHOLDS['201.11']['max_slowdown']):
        missing.append(f"cost at the selected gate of {candidate.get('gate_steps')} steps: "
                       + ('unknown' if slowdown is None else f'{slowdown:.2f}'))
    if missing:
        return {'rule': 4, 'name': 'No build', 'readings': missing, 'decision': 'section 1 is the standing procedure'}
    return {'rule': 5, 'name': 'Build a candidate', 'readings': list(needed) + [f"gate {candidate['gate_steps']} steps, projected slowdown {slowdown:.2f}"],
            'decision': 'build a candidate margin-gated decoder; it stays a candidate until its own part shows it equals a full float32-upcast decode on at least 200 rows at two shapes'}


def part200_text_analysis(rows):
    """Exploratory: pass A against Part 200's chunked answers for the same unit and question (decoded text only)."""
    path = ROOT / 'results/part200/answers.jsonl'
    if not path.exists():
        return {'status': 'not available'}
    old = {(r.get('unit_id'), int(r.get('question_index', -1))): r for r in read_jsonl(path) if r.get('stage') == 'battery'}
    flips = {name: {'a_true_b_false': 0, 'a_false_b_true': 0, 'n': 0} for name in LABELS}
    compared = differing = 0
    for r in rows:
        if r.get('pass') != 'A' or r.get('unit_id') is None or r.get('nonfinite'):
            continue
        prior = old.get((r.get('unit_id'), int(r.get('question_index', -1))))
        if prior is None or not isinstance(prior.get('answer'), str):
            continue
        compared += 1
        differing += int(r.get('answer', '') != prior['answer'])
        ours, theirs = labels(r.get('answer', '')), labels(prior['answer'])
        for name in LABELS:
            flips[name]['n'] += 1
            flips[name]['a_true_b_false'] += int(ours[name] and not theirs[name])
            flips[name]['a_false_b_true'] += int(theirs[name] and not ours[name])
    return {'comparison': 'decoded answer text (Part 200 stored no token ids); another process and other neighbours',
            'compared': compared, 'differing': differing, 'label_flips': flips}


# ----------------------------------------------------------------------------------------------------------- report
def _fmt(value, digits=3):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def render(result):
    gates = result['gates']
    lines = ['# Part 201 score', '', '## Gates', '', '| Gate | Result | Detail |', '|---|---|---|']
    detail = {
        'G1': f"bf16 {'passed' if gates['G1']['bf16']['passed'] else 'FAILED'}, float32 {'passed' if gates['G1']['float32']['passed'] else 'FAILED'}, "
              f"kernel record {'passed' if gates['G1']['kernel']['passed'] else 'FAILED' if gates['G1']['kernel']['present'] else 'absent'}; "
              f"revisions {gates['G1']['revisions']}; snapshot loaded is the recorded revision: {gates['G1']['every_launch_loaded_the_snapshot_of_its_recorded_revision']}",
        'G2': f"{gates['G2']['failed']} of {gates['G2']['rows']} rows failed; "
              f"{gates['G2']['positions_where_the_emitted_token_is_tied_with_and_is_not_the_first_listed_top']} positions emitted a tied token that is not the first-listed top",
        'G3': f"plan {'passed' if gates['G3']['plan']['passed'] else 'FAILED'}, bf16 {'passed' if gates['G3']['bf16']['passed'] else 'FAILED'}, "
              f"float32 {'passed' if gates['G3']['float32']['passed'] else 'FAILED'}; "
              + '; '.join(gates['G3']['plan']['problems'] + gates['G3']['bf16']['problems'] + gates['G3']['float32']['problems']),
        'G4': f"bf16 {'passed' if gates['G4']['bf16']['passed'] else 'FAILED'}, float32 {'passed' if gates['G4']['float32']['passed'] else 'FAILED'}; "
              f"{gates['G4']['failed']} of {gates['G4']['pushed_rows']} pushed rows failed; {len(gates['G4']['hook_count_failures'])} calls without matching hook evidence; "
              f"{len(gates['G4']['bf16']['rows_with_more_than_one_push_hash'])} rows with more than one push hash in bf16, "
              f"{len(gates['G4']['float32']['rows_whose_push_hash_is_not_the_bf16_one'])} whose float32 push hash is not the bf16 one; "
              f"inputs bound across launches: bf16 {gates['G4']['bf16']['inputs_agree_across_launches']}, float32 {gates['G4']['float32']['inputs_agree_with_the_bf16_launches']}",
        'G5': f"{len(gates['G5']['records'])} launch records",
    }
    for name in ('G1', 'G2', 'G3', 'G4', 'G5'):
        lines.append(f"| {name} | {'passed' if gates[name]['passed'] else 'FAILED'} | {detail[name]} |")
    control, state = result['positive_control'], result['state']
    lines += ['', f"Positive control: A and C differ on {control['rows_differing_between_A_and_C']} rows (needs {control['needed']}): "
              f"{'passed' if control['passed'] else 'FAILED'}{' (' + control['note'] + ')' if control['note'] else ''}.",
              f"Run state: smoke plan {state['smoke']}; float32 cut applied {state['float32_cut']}; A-again in pass A's process {state['a_again_same_process']}.",
              '', '## Registered predictions', '', '| # | Prediction (verbatim) | Confidence | Verdict |', '|---|---|---|---|']
    for key, text, confidence in PREDICTIONS:
        entry = result['predictions'][key]
        note = f" ({entry['note']})" if entry.get('note') else ''
        lines.append(f"| {key} | {text} | {confidence} | {entry['verdict']}{note} |")
    lines += ['', '### Estimates behind the verdicts', '']
    for key, _, _ in PREDICTIONS:
        estimate = result['predictions'][key]['estimate']
        if key == '201.11' and isinstance(estimate, dict):
            estimate = {'pass_A_seconds': estimate['pass_A_seconds'], 'by_margin_steps': estimate['by_margin_steps']}
        if key == '201.7' and isinstance(estimate, dict):
            estimate = {b: {'rows': v['rows'], 'equal_rows': v['equal_rows'],
                            'first_differences': [(c['prompt'], c['first_difference'], c['reference_margin_there']) for c in v['comparisons'] if not c['equal']]}
                        for b, v in estimate['batches'].items()}
        lines.append(f"- **{key}**: `{json.dumps(estimate, ensure_ascii=False)}`")
    decision, candidate = result['decision'], result['candidate_gate']
    lines += ['', '## Ordered decision (BATCH-SHAPE-NOISE.md, section 3)', '',
              f"Rule {decision['rule']}, {decision['name']}: {decision['decision']}.", '', 'Readings: ' + '; '.join(decision['readings']) + '.', '',
              f"Candidate gate: {_fmt(candidate['gate_steps'])} steps ({candidate['rule']}). Cost at that gate: `{json.dumps(candidate['cost_at_the_gate'])}`. "
              'A price above the longest probed prefix is a linear extrapolation, an approximation and not a measured bound.',
              '', '## Pairs', '', '| Pair | Planned | Compared | Differ | Share (95% Wilson) | Not finite | Mean length A minus B | Other token not the runner-up |',
              '|---|---:|---:|---:|---|---:|---:|---:|']
    for key, z in result['pairs'].items():
        low, high = wilson(z['differing'], z['compared'])
        share = f"{z['differing'] / z['compared']:.3f} ({low:.3f} to {high:.3f})" if z['compared'] else 'n/a'
        label = key + (' (plain rows)' if key == 'M/C' else '')
        lines.append(f"| {label} | {z['planned']} | {z['compared']} | {z['differing']} | {share} | {len(z['nonfinite_rows'])} | "
                     f"{_fmt(z['mean_length_difference_a_minus_b'], 2)} | {len(z['b_token_not_a_runner_up'])} of {z['differing']} |")
    lines += ['', '## Label flips by pair and direction (count of n, 95% Wilson interval)', '',
              '| Pair | Label | True in the first, false in the second | False in the first, true in the second | n |', '|---|---|---|---|---:|']
    for key, z in result['pairs'].items():
        for name, c in z['label_flips'].items():
            cells = []
            for direction in ('a_true_b_false', 'a_false_b_true'):
                low, high = c[direction + '_wilson95']
                cells.append(f"{c[direction]} ({_fmt(low)} to {_fmt(high)})")
            lines.append(f"| {key} | {name} | {cells[0]} | {cells[1]} | {c['n']} |")
    lines += ['', '## Margins (bf16 rows)', '',
              f"Share of positions within 0, 1, 2 and 4 steps, over {result['bf16_decodes_in_the_margin_share']} bf16 decodes "
              f"(a row decoded in several passes counts once per pass): `{json.dumps(result['bf16_share_of_positions_within_steps'])}`.", '',
              f"Pass M padding: `{json.dumps(result['pass_M_padding'])}`.", '',
              f"Share of generation steps with an unfinished row within 2 steps, by pass: `{json.dumps(result['share_of_generation_steps_with_an_unfinished_row_within_2_steps'])}`.", '',
              f"Exact ties: `{json.dumps(result['ties'])}`.", '', '## First differences, row by row', '']
    for key, z in result['pairs'].items():
        if not z['differing_rows']:
            continue
        lines += [f"### {key}: {z['differing']} of {z['compared']} rows; positions `{json.dumps(z['first_difference_positions'])}`", '',
                  '| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |', '|---|---:|---:|---:|---:|---|']
        for d in z['differing_rows']:
            lines.append(f"| {d['row_id']} | {d['position']} | {_fmt(d['a_margin'], 4)} | {_fmt(d['a_bf16_steps'], 2)} | "
                         f"{_fmt(d['b_margin'], 4)} | {d['b_token_is_a_runner_up']} |")
        lines.append('')
    lines += ['## Pass A against Part 200 (exploratory)', '', f"`{json.dumps(result['A_vs_Part200_chunked_exploratory'], ensure_ascii=False)}`", '',
              f"Rows with a non-finite score: {result['nonfinite_rows'] or 'none'}."]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('rows', 'load', 'plan', 'out'):
        parser.add_argument(f'--{name}', required=True)
    for name in ('kernel', 'reference', 'probe'):
        parser.add_argument(f'--{name}')

    def optional(path):
        return json.loads(Path(path).read_text(encoding='utf-8')) if path and Path(path).exists() else None

    args = parser.parse_args(argv)
    result = score(read_jsonl(args.rows), optional(args.kernel), optional(args.reference),
                   json.loads(Path(args.load).read_text(encoding='utf-8')), optional(args.probe), args.plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'score.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = render(result)
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(report, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
