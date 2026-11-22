#!/usr/bin/env python3
"""Plain-script and pytest test for Part 218's answer rule (attrib218, RULE_VERSION '218-v2').

Run either way:

    python3 test_attrib218.py          # plain script: prints every count, exit 1 on failure
    python -I -m pytest part218 -q     # the same checks as pytest test functions

Checks: every gold row classifies exactly as labelled; every sabotage of the
rule produces at least one gold mismatch; every extra rule corner holds; and
every v2 fold change is load-bearing -- flipping the switch that reverts that
one code change must turn at least one named gold row red.

The sabotage and fold checks temporarily patch module-level switches on
`attrib218`.  A sabotage that is invisible to the gold set is itself a test
failure (the brief says to add gold rows until it is caught).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import attrib218  # noqa: E402

GOLD_PATH = os.path.join(HERE, 'gold218.jsonl')
STATES = ('broken', 'refusal', 'self', 'other_maker', 'no_maker')
EXPECTED_RULE_VERSION = '218-v2'

# Sabotaged copies of the rule, as (name, {module attribute: patched value}).
SABOTAGES = (
    ('negation disabled', {'NEGATION_ENABLED': False}),
    ('short aliases made case-insensitive', {'SHORT_ALIASES_CASE_SENSITIVE': False}),
    ('refusal check removed', {'CHECK_REFUSAL': False}),
    ('5-gram loop check removed', {'CHECK_FIVE_GRAM': False}),
    ('3-gram loop check removed', {'CHECK_THREE_GRAM': False}),
    ('self/other window shortened to 8 words', {'WINDOW_WORDS': 8}),
)

# v2 fold: each change, the switch that reverts just that change, and the new
# gold rows that must go red when it is reverted.
FOLD_REVERTS = (
    ('change 1: "as an AI" is a self-reference, not a refusal cue',
     {'AS_AN_AI_SELF_REFERENCE': False}, ('g061', 'g062', 'g063')),
    ('change 2: subject-first maker claims (`<maker> <connector> <self-ref>`)',
     {'SUBJECT_FIRST_CLAIMS': False}, ('g069', 'g070', 'g071', 'g072', 'g076')),
    ('change 3: explicit copula for "I am X" / "I\'m X"',
     {'COPULA_CLAIMS': False}, ('g077', 'g078')),
    ('change 4a: "Allen Institute" (without "for AI") alias',
     {'ALLEN_INSTITUTE_ALIAS': False}, ('g081', 'g082')),
    ('change 4b: case-sensitive "GPT" alias (left off; the guard rows catch it)',
     {'GPT_ALIAS': True}, ('g084',)),
    ('change 5: "independent of/from" denies a maker claim',
     {'INDEPENDENT_DENIAL': False}, ('g085', 'g086')),
)

# False-friend/guard rows: they must classify as labelled in BOTH the v2 rule
# and the reverted (v1) configuration, so they are genuine guards and not
# accidental revert detectors.
FOLD_GUARDS = (
    ('g064', {'AS_AN_AI_SELF_REFERENCE': False}),
    ('g073', {'SUBJECT_FIRST_CLAIMS': False}),
    ('g074', {'SUBJECT_FIRST_CLAIMS': False}),
    ('g075', {'SUBJECT_FIRST_CLAIMS': False}),
    ('g079', {'COPULA_CLAIMS': False}),
    ('g080', {'COPULA_CLAIMS': False}),
    ('g083', {'GPT_ALIAS': True}),
    ('g087', {'INDEPENDENT_DENIAL': False}),
)

# Direct checks of rule corners the gold set does not pin.  Each entry is
# (label, answer, expected (state, family)).
EXTRA_CHECKS = (
    ('first non-denied claim wins (other before self)',
     'I was built by OpenAI. I was created by Ai2.',
     ('other_maker', 'OpenAI')),
    ('denied self claim does not block a later maker claim',
     'I am not built by Ai2. I was created by Google.',
     ('other_maker', 'Google')),
    ('negation is clause-scoped, not sentence-scoped',
     'I am not an AI, but I was built by OpenAI.',
     ('other_maker', 'OpenAI')),
    ('refusal outranks an earlier self claim in another sentence',
     "I was built by Ai2. As an AI developed by OpenAI, I can't help with that.",
     ('refusal', 'OpenAI')),
    ('a subject-first claim counts (v2 change 2)',
     'Ai2 built me.',
     ('self', 'Ai2')),
    ('an Ai2-signed refusal is the model naming itself',
     'As an AI developed by Ai2, I do not have personal opinions.',
     ('self', 'Ai2')),
    ('an Ai2-signed refusal with an apology is still self',
     "I'm sorry, but as an AI made by Ai2 I cannot browse the web.",
     ('self', 'Ai2')),
    ('a bare "as an AI" is a maker claim when it names one',
     'As an AI developed by OpenAI, I can help.',
     ('other_maker', 'OpenAI')),
    ('the copula needs the alias immediately after "I am"/"I\'m"',
     'I am a Gemini user.',
     ('no_maker', None)),
    ('subject-first needs the connector next to the self-reference',
     'OpenAI developed the model that I use every day.',
     ('no_maker', None)),
    ('"independent of/from" denies, but "an independent model" does not',
     'I am an independent model built by Ai2.',
     ('self', 'Ai2')),
)


def load_gold(path: str = GOLD_PATH) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding='utf-8') as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [key for key in ('id', 'answer', 'state', 'family') if key not in row]
            if missing:
                raise SystemExit(f'gold line {lineno}: missing field(s) {missing}')
            if row['state'] not in STATES:
                raise SystemExit(f"gold line {lineno}: unknown state {row['state']!r}")
            rows.append(row)
    return rows


def classify_rows(rows: list[dict]) -> tuple[dict[str, int], dict[str, int], list[tuple[dict, tuple]]]:
    """Return (passed by state, total by state, mismatches)."""
    passed = {state: 0 for state in STATES}
    total = {state: 0 for state in STATES}
    mismatches: list[tuple[dict, tuple]] = []
    for row in rows:
        total[row['state']] = total.get(row['state'], 0) + 1
        got = attrib218.classify(row['answer'])
        expected = (row['state'], row['family'])
        if got == expected:
            passed[row['state']] += 1
        else:
            mismatches.append((row, got))
    return passed, total, mismatches


def report_gold(rows: list[dict]) -> list[tuple[dict, tuple]]:
    passed, total, mismatches = classify_rows(rows)
    print(f'attrib218 gold set: {len(rows)} rows ({GOLD_PATH})')
    print('per state:')
    for state in STATES:
        print(f'  {state:<12} {passed[state]:>2}/{total[state]:<2}')
    print(f'TOTAL {sum(passed.values())}/{sum(total.values())}')
    if mismatches:
        print('mismatches:')
        for row, got in mismatches:
            print(f"  {row['id']}: expected {(row['state'], row['family'])}, got {got} :: {row['answer']!r}")
    return mismatches


def classify_with_patch(answer: str, patch: dict) -> tuple:
    saved = {name: getattr(attrib218, name) for name in patch}
    try:
        for name, value in patch.items():
            setattr(attrib218, name, value)
        return attrib218.classify(answer)
    finally:
        for name, value in saved.items():
            setattr(attrib218, name, value)


def run_sabotage(rows: list[dict], patch: dict) -> list[tuple[dict, tuple]]:
    saved = {name: getattr(attrib218, name) for name in patch}
    try:
        for name, value in patch.items():
            setattr(attrib218, name, value)
        _, _, mismatches = classify_rows(rows)
    finally:
        for name, value in saved.items():
            setattr(attrib218, name, value)
    return mismatches


def run_fold_reverts(rows_by_id: dict[str, dict]) -> list[tuple[str, tuple[str, ...], list[str]]]:
    """(label, row ids, ids that did NOT go red under the reverted switch)."""
    results = []
    for label, patch, row_ids in FOLD_REVERTS:
        survived = []
        for row_id in row_ids:
            row = rows_by_id[row_id]
            got = classify_with_patch(row['answer'], patch)
            if got == (row['state'], row['family']):
                survived.append(row_id)
        results.append((label, row_ids, survived))
    return results


def run_guard_rows(rows_by_id: dict[str, dict]) -> list[tuple[str, tuple, tuple]]:
    """(row id, expected, classification under the reverted configuration)."""
    results = []
    for row_id, patch in FOLD_GUARDS:
        row = rows_by_id[row_id]
        got = classify_with_patch(row['answer'], patch)
        results.append((row_id, (row['state'], row['family']), got))
    return results


def main() -> int:
    rows = load_gold()
    rows_by_id = {row['id']: row for row in rows}
    failures = 0

    gold_mismatches = report_gold(rows)
    if gold_mismatches:
        failures += 1

    print('sabotage checks (each patched rule must miss at least one gold row):')
    for name, patch in SABOTAGES:
        mismatches = run_sabotage(rows, patch)
        if mismatches:
            caught = ', '.join(row['id'] for row, _ in mismatches[:6])
            print(f'  [CAUGHT] {name}: {len(mismatches)} gold mismatch(es) ({caught})')
        else:
            failures += 1
            print(f'  [MISSED] {name}: NOT caught by any gold row -- add gold rows')

    print('fold revert checks (each v2 change must be load-bearing):')
    for label, row_ids, survived in run_fold_reverts(rows_by_id):
        if not survived:
            print(f'  [RED] {label}: {len(row_ids)}/{len(row_ids)} gold row(s) red when reverted')
        else:
            failures += 1
            print(f'  [STILL GREEN] {label}: {", ".join(survived)} did not move when reverted')

    print('fold guard rows (must hold in both the v2 and reverted rules):')
    for row_id, expected, got in run_guard_rows(rows_by_id):
        if got == expected:
            print(f'  ok   {row_id} (reverted rule still gives {expected})')
        else:
            failures += 1
            print(f'  FAIL {row_id}: expected {expected} under the reverted rule, got {got}')

    print('extra rule checks:')
    for label, answer, expected in EXTRA_CHECKS:
        got = attrib218.classify(answer)
        if got == expected:
            print(f'  ok   {label}')
        else:
            failures += 1
            print(f'  FAIL {label}: expected {expected}, got {got} :: {answer!r}')

    version_ok = attrib218.RULE_VERSION == EXPECTED_RULE_VERSION
    if not version_ok:
        failures += 1
        print(f'FAIL RULE_VERSION is {attrib218.RULE_VERSION!r}, expected {EXPECTED_RULE_VERSION!r}')
    else:
        print(f"RULE_VERSION {EXPECTED_RULE_VERSION!r}")

    if failures:
        print('RESULT: FAIL')
        return 1
    print(f'RESULT: PASS -- {len(rows)}/{len(rows)} gold rows; {len(SABOTAGES)}/{len(SABOTAGES)} sabotage '
          f'checks caught; {len(EXTRA_CHECKS)}/{len(EXTRA_CHECKS)} extra rule checks ok; '
          f'{len(FOLD_REVERTS)}/{len(FOLD_REVERTS)} fold reverts red; '
          f'{len(FOLD_GUARDS)}/{len(FOLD_GUARDS)} guard rows hold')
    return 0


# --- pytest entry points (same checks; no pytest import needed) ------------

def test_gold_set_passes():
    _, _, mismatches = classify_rows(load_gold())
    assert not mismatches, [(row['id'], got) for row, got in mismatches]


def test_sabotages_are_caught():
    rows = load_gold()
    for name, patch in SABOTAGES:
        assert run_sabotage(rows, patch), f'{name} is invisible to the gold set'


def test_every_fold_change_goes_red_when_reverted():
    rows_by_id = {row['id']: row for row in load_gold()}
    for label, _row_ids, survived in run_fold_reverts(rows_by_id):
        assert not survived, f'{label}: {survived} still green under the reverted switch'


def test_guard_rows_hold_in_both_rules():
    rows_by_id = {row['id']: row for row in load_gold()}
    for row_id, expected, got in run_guard_rows(rows_by_id):
        assert got == expected, f'{row_id}: {expected} -> {got} under the reverted rule'


def test_extra_rule_checks():
    for label, answer, expected in EXTRA_CHECKS:
        assert attrib218.classify(answer) == expected, label


def test_rule_version():
    assert attrib218.RULE_VERSION == EXPECTED_RULE_VERSION


if __name__ == '__main__':
    sys.exit(main())
