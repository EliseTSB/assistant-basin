"""Torch-free resolution of Part 216's 456 plan rows against the frozen prompts bank.

`run216.py` builds its plan here and looks every row up here; `score216.py` rebuilds the same
lookup to bind each recorded row back to the pinned `(arm, q)` entry (Alden #14).  The scorer is a
pure json/numpy reader and must not import the runner, so the shared resolution lives in this
module and imports neither torch nor numpy.

The rules are the runner's, unchanged (Alden #10 #1):

  * plain stage1 arms (`unchanged`, `zero`, `scale_dial_fact`, `random_k:zero`) resolve through
    `bank['items'][q]`;
  * the opener arms (`op_filler`, `op_denial`, `op_vogel`) resolve through `bank['openers']`, keyed
    strictly on `(arm, q)` -- their sequence is the opener sentence prepended to the question, so
    `items[q]` is the WRONG row for them;
  * controls resolve through `bank['controls']`, indexed `arm_index * n_questions + q`;
  * factuals resolve through `bank['factuals'][q]` and are forwarded with `prompt_ids == seq`.

`expected_inputs()` turns one `(arm, q)` into the exact fields a committed row must record --
sequence, prompt, `T`, read position and the input sha256 -- which is what the scorer checks.
"""
from __future__ import annotations

import hashlib
import json

N_QUESTIONS = 24
FACTUAL_ARM = 'factual'


def row_key(row: dict) -> str:
    """`'<arm>:<question_index>'` -- the key every forwards.jsonl row carries."""
    return f'{row["arm"]}:{row["question_index"]}'


def plan_rows(bank: dict, g0_arms=()) -> list[dict]:
    """456 rows: the G0 anchor arms first, then the rest, then the controls, then the factuals."""
    rows = []
    g0 = [a for a in g0_arms if a in bank['arms']]
    other_arms = [a for a in bank['arms'] if a not in g0]
    for group in (g0, other_arms):
        for arm in group:
            for qi in range(N_QUESTIONS):
                rows.append({'kind': 'stage1', 'arm': arm, 'question_index': qi})
    for arm in bank['control_arms']:
        for qi in range(N_QUESTIONS):
            rows.append({'kind': 'control', 'arm': arm, 'question_index': qi})
    for qi in range(N_QUESTIONS):
        rows.append({'kind': 'factual', 'arm': FACTUAL_ARM, 'question_index': qi})
    return rows


def opener_index(bank: dict) -> dict:
    """`(arm, question_index) -> opener entry` over the 72 opener rows, strictly one per key.

    The opener arms are stage1-kind rows, but their sequences are NOT `bank['items'][q]`: Part 215's
    opener sentence is prepended to the user text, so the token ids, `T` and the read position all
    differ.  Resolving them through `items` silently forwards the plain prompt (Alden #10 #1), so
    the index is built once and every lookup is checked against the exact `(arm, q)` key.
    """
    op_arms = bank.get('opener_arms') or {}
    if not op_arms:
        return {}
    idx: dict[tuple[str, int], dict] = {}
    for entry in bank.get('openers') or []:
        key = (str(entry['arm']), int(entry['question_index']))
        if key in idx:
            raise SystemExit(f'prompts bank has duplicate opener key {key}')
        if key[0] not in op_arms:
            raise SystemExit(f'prompts bank opener row has unknown arm {key[0]!r}')
        idx[key] = entry
    expected = {(arm, qi) for arm in op_arms for qi in range(N_QUESTIONS)}
    if set(idx) != expected:
        missing = sorted(expected - set(idx))[:5]
        extra = sorted(set(idx) - expected)[:5]
        raise SystemExit(f'prompts bank opener keys do not match the 3x24 grid; '
                         f'missing={missing} extra={extra}')
    return idx


def resolve_row(bank: dict, row: dict) -> dict:
    """The bank entry for one plan row, on its exact `(arm, q)` key.

    Opener arms resolve through `bank['openers']`, never through `bank['items']` (Alden #10 #1).
    """
    if row['kind'] == 'stage1':
        arm = str(row['arm'])
        if arm in (bank.get('opener_arms') or {}):
            key = (arm, int(row['question_index']))
            entry = opener_index(bank)[key]
            if (str(entry['arm']), int(entry['question_index'])) != key:
                raise SystemExit(f'opener lookup for {key} returned {entry.get("arm")}:'
                                 f'{entry.get("question_index")}')
            return entry
        return bank['items'][row['question_index']]
    if row['kind'] == 'control':
        k = bank['control_arms'].index(row['arm']) * N_QUESTIONS + row['question_index']
        entry = bank['controls'][k]
        if str(entry['arm']) != str(row['arm']) or \
                int(entry['question_index']) != int(row['question_index']):
            raise SystemExit(f'control lookup for {row["arm"]}:{row["question_index"]} returned '
                             f'{entry.get("arm")}:{entry.get("question_index")}')
        return entry
    return bank['factuals'][row['question_index']]


def input_sha256(sequence_token_ids) -> str:
    """The hash of the exact token sequence, byte-for-byte as `run216.forward_row` writes it."""
    return hashlib.sha256(json.dumps([int(t) for t in sequence_token_ids]).encode('utf-8')).hexdigest()


def expected_inputs(bank: dict, arm: str, question_index: int) -> dict:
    """The inputs a committed row for `(arm, q)` must record, rebuilt from the frozen bank.

    Returns `kind`, `sequence_token_ids`, `prompt_token_ids`, `T`, `read_position` and
    `input_sha256`.  Raises the bank's own `SystemExit`/`KeyError` on a malformed bank; callers that
    must not exit (the scorer) catch it and report a diagnostics-only reason instead.
    """
    arm = str(arm)
    qi = int(question_index)
    if arm == FACTUAL_ARM:
        kind = 'factual'
    elif arm in (bank.get('control_arms') or []):
        kind = 'control'
    else:
        kind = 'stage1'
    entry = resolve_row(bank, {'kind': kind, 'arm': arm, 'question_index': qi})
    if kind == 'factual':
        # forward_row is called with prompt_ids == seq for a factual row.
        seq = [int(t) for t in entry['prompt_token_ids']]
        prompt = list(seq)
    else:
        seq = [int(t) for t in entry['sequence_token_ids']]
        prompt = [int(t) for t in entry['prompt_token_ids']]
    return {'kind': kind, 'sequence_token_ids': seq, 'prompt_token_ids': prompt,
            'T': len(prompt), 'read_position': int(entry['read_position']),
            'input_sha256': input_sha256(seq)}
