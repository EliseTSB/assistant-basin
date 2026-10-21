"""Pinned stdlib-only gate witness: no project/Torch/model import or call."""
from __future__ import annotations

import __future__
import ast
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
import subprocess
from types import SimpleNamespace

REPO = Path('/home/elisepc/green/worktrees/ab-43-singleton')
PIN = '16027e9b1f64d7fece4bac243a18199d2255a57f'
source = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{PIN}:part43.py'], text=True)
test_source = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{PIN}:test_part43.py'], text=True)
constant_names = {
    'MODEL', 'REVISION', 'MAKER_ANSWERS', 'DISCOVERY_QUESTION',
    'VALIDATION_ANSWER_OPTIONS', 'CALIBRATION_DECISION_CELLS',
    'CALIBRATION_CELLS', 'CALIBRATION_EPSILON_LIMIT',
}
function_names = {'_finite_numeric_values', 'calibration_gate', 'run_pilot'} | {n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name.startswith('_calibration')}
constant_names |= {t.id for n in ast.parse(source).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name) and t.id.startswith('CALIBRATION')}
nodes = []
for node in ast.parse(source).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constant_names for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in function_names:
        nodes.append(node)
scope = {'math': math, 'Mapping': Mapping}
exec(compile(ast.Module(body=nodes, type_ignores=[]), f'{PIN}:gate', 'exec', flags=__future__.annotations.compiler_flag), scope)
scope['part43'] = SimpleNamespace(**{name: scope[name] for name in constant_names})
fixture_node = next(n for n in ast.parse(test_source).body if isinstance(n, ast.FunctionDef) and n.name == '_calibration_fixture')
exec(compile(ast.Module(body=[fixture_node], type_ignores=[]), f'{PIN}:test_fixture', 'exec', flags=__future__.annotations.compiler_flag), scope)
args = SimpleNamespace(model=scope['MODEL'], revision=scope['REVISION'], candidates='inert-candidates', command='pilot')


def ready(difference=0.01):
    value = scope['_calibration_fixture'](args, fp32_difference=difference)
    # Correct the offered fixture's unrelated replay/clean inconsistency so
    # the positive control has internally consistent zero replay error.
    for leg_name in ('bf16', 'fp32'):
        leg = value[leg_name]
        clean = leg['cells']['clean']['primary_score']
        leg['cells']['clean_to_clean_replay']['primary_score'] = clean
        leg['cells']['clean_to_clean_replay']['repeat_primary_scores'] = [clean] * leg['repeat_count']
    return value


class DecisionReached(Exception):
    pass


def observe(name, artifact):
    calls = []
    try:
        accepted, reason = scope['calibration_gate'](artifact, model=args.model, revision=args.revision)
    except Exception as error:
        return {'case': name, 'gate_outcome': 'uncaught_exception', 'exception_type': type(error).__name__, 'decision_reached': False}

    def sentinel(*_args, **_kwargs):
        calls.append('lattice')
        raise DecisionReached()

    scope.update({
        '_sequences': lambda _tokenizer: [{}],
        'run_signs': lambda *_a, **_k: {},
        'run_lattice_amended': sentinel,
        'output_header': lambda *_a: {},
        'ROOT': REPO,
    })
    try:
        output = scope['run_pilot'](object(), object(), {}, args, calibration=artifact)
        reached = False
        status = output.get('status')
    except DecisionReached:
        reached, status = True, 'decision_sentinel'
    return {'case': name, 'gate_outcome': 'accepted' if accepted else 'refused', 'reason': reason,
            'decision_reached': reached, 'pilot_status': status}


positive = ready()
original_fixture = scope['_calibration_fixture'](args)
hidden_replay_error = copy.deepcopy(positive)
leg = hidden_replay_error['bf16']
cell = leg['cells']['clean_to_clean_replay']
cell['primary_score'] = leg['cells']['clean']['primary_score'] + 1.0
cell['repeat_primary_scores'] = [cell['primary_score']] * 3
# Keep the leg/error summaries at zero: the artifact now contradicts its
# recorded score rows. The gate should recompute the replay error and refuse.
missing_error_row = copy.deepcopy(positive)
del missing_error_row['per_cell_errors']['all_four_zero']
nonfinite = copy.deepcopy(positive)
nonfinite['epsilon'] = float('nan')

cases = [
    ('positive_consistent_below_budget', positive),
    ('missing_calibration', None),
    ('over_budget_control', ready(1.0)),
    ('nonfinite_control', nonfinite),
    ('offered_fixture_has_inconsistent_replay_score', original_fixture),
    ('one_nat_replay_discrepancy_hidden_by_zero_summaries', hidden_replay_error),
    ('malformed_missing_per_cell_error_row', missing_error_row),
]
rows = [observe(name, value) for name, value in cases]
import json
for r in rows: print(json.dumps(r))
for leg in ('bf16','fp32'):
    c=original_fixture[leg]['cells']
    print(leg,'clean',c['clean']['primary_score'],'replay',c['clean_to_clean_replay']['repeat_primary_scores'],'recorded',original_fixture[leg].get('clean_to_clean_replay_error'))
