#!/usr/bin/env python3
"""Part 215's torch-free preflight-evidence validator (Alden #13, fold 2).

Both the runner (`run215.require_pass_preflight`) and the scorer (`score215.admissibility`)
call `validate_preflight`, so a preflight receipt is admitted only on its ROWS, never on a
summary field or a boolean.  Every gate is recomputed here:

  G0        both comparisons, exactly the 24 question indices each, every row's
            |ours - part214| recomputed and <= 0.02, every row's `part214` equal to the value
            in the pinned Part 214 Stage 1 receipt (loaded from its reviewed bytes), and the
            stored row/summary fields equal to the recomputation.
  G1        exactly the 48 (question, sample 0/1) keys, seeds equal
            score215.row_seed('neutral', 'identity', 'v0', qi, sample), `matched` recomputed
            from token-list equality of `ours` against the pinned Part 214 Stage 2 row's token
            ids (never the row's own `part214` copy alone), and >= 45 recomputed matches.
  G2        exactly the 96 pilot keys (score215.pilot_rows()), seeds equal pilot_seed(...),
            broken recomputed with the frozen score215.is_broken on every answer, <= the bound.
  G3        score215.g3_scorer_check() rerun here; the stored result must equal the rerun.
  Latency   the 7.5 s ceiling over ALL 192 measured rows (48 G0 forwards + 48 G1 and 96 G2
            generations); the stored summary must equal the recomputation.

Every reference file is read ONCE into a buffer, hashed, and parsed from that same buffer.
Nothing here imports torch: the module uses the standard library and score215's pure
functions only.

The pins below are the reviewed Part 214 digests, exposed as module attributes so a test can
point them at synthetic fixtures (the same seam `run215` has had since fold 1).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import score215 as sc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_PATH = ROOT / 'results/part214/stage1/stage1.json'
DEFAULT_STAGE2_PATH = ROOT / 'results/part214/stage2/stage2.jsonl'

# Reviewed Part 214 reference bytes (score215's pins; a test may replace them, as with
# run215.PART214_STAGE1_SHA256 / _STAGE2_SHA256).
PART214_STAGE1_SHA256 = sc.PART214_STAGE1_SHA256
PART214_STAGE2_SHA256 = sc.PART214_STAGE2_SHA256

# The design's gate numbers (DESIGN.md v4.1 "Gates" and "Size"), one place only.
G0_TOLERANCE = 0.02
MAX_LATENCY_S = 7.5
G1_NEEDED = 45
PILOT_N = 96
PILOT_BOUND = int(math.floor(sc.BROKEN_LIMIT * PILOT_N + 1e-12))
PREFLIGHT_GENERATION_CALLS = 2 * sc.N_QUESTIONS + PILOT_N   # 48 G1 + 96 G2 = 144
PREFLIGHT_FORWARD_CALLS = 2 * sc.N_QUESTIONS                # 24 G0 unchanged + 24 zero = 48
MEASURED_CALLS = PREFLIGHT_GENERATION_CALLS + PREFLIGHT_FORWARD_CALLS  # 192

_REL_TOL = 1e-9


class EvidenceError(Exception):
    """A reference receipt is not the pinned bytes, or is not usable evidence."""


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int_list(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value)


def _close(actual, expected, tol: float = _REL_TOL) -> bool:
    if not _number(actual) or not _number(expected):
        return False
    return abs(float(actual) - float(expected)) <= tol * max(1.0, abs(float(actual)),
                                                             abs(float(expected)))


def parse_jsonl_bytes(raw: bytes, label: str) -> list[dict]:
    """A JSONL receipt parsed from bytes already read and hashed by the caller.

    A torn tail is fatal: this is a receipt the caller did not write and may not repair.
    """
    if not raw:
        return []
    if not raw.endswith(b'\n'):
        tail = raw[raw.rfind(b'\n') + 1:]
        raise EvidenceError(f'{label}: ends in {len(tail)} bytes of a partial line')
    rows = []
    for number, line in enumerate(raw.split(b'\n')[:-1], start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EvidenceError(f'{label} line {number} is corrupt ({exc})') from None
        if not isinstance(value, dict):
            raise EvidenceError(f'{label} line {number}: expected a JSON object')
        rows.append(value)
    return rows


def stage1_reference(raw: bytes, expected_sha256: str, part214_prompts_sha256: str | None = None,
                     label: str = 'Part 214 Stage 1') -> dict:
    """Part 214's Stage 1 receipt, parsed from `raw` after its digest is checked."""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise EvidenceError(f'{label}: SHA-256 {digest} is not the reviewed Part 214 Stage 1 '
                            f'receipt {expected_sha256}')
    try:
        obj = json.loads(raw.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceError(f'{label}: not JSON ({exc})') from None
    if not isinstance(obj, dict):
        raise EvidenceError(f'{label}: not a JSON object')
    if obj.get('part') != 214:
        raise EvidenceError(f'{label}: part is {obj.get("part")}, want Part 214')
    if part214_prompts_sha256 is not None and obj.get('prompts_sha256') != part214_prompts_sha256:
        raise EvidenceError(f'{label}: prompts_sha256 is {obj.get("prompts_sha256")}, this build '
                            f'uses {part214_prompts_sha256}')
    for arm in ('unchanged', 'zero'):
        entry = (obj.get('arms') or {}).get(arm)
        values = entry.get('g_ali') if isinstance(entry, dict) else None
        if not isinstance(values, list) or len(values) < sc.N_QUESTIONS:
            raise EvidenceError(f'{label}: no usable arms.{arm}.g_ali')
        if any(not _number(value) for value in values[:sc.N_QUESTIONS]):
            raise EvidenceError(f'{label}: arms.{arm}.g_ali has a non-numeric entry')
    return obj


def stage2_reference(raw: bytes, expected_sha256: str, part214_prompts_sha256: str | None = None,
                     label: str = 'Part 214 Stage 2') -> dict:
    """Part 214's 48 Stage 2 `unchanged` reference rows, keyed (question, sample 0/1)."""
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise EvidenceError(f'{label}: SHA-256 {digest} is not the reviewed Part 214 Stage 2 '
                            f'receipt {expected_sha256}')
    try:
        records = parse_jsonl_bytes(raw, label)
    except EvidenceError as exc:
        raise EvidenceError(str(exc)) from None
    out: dict[tuple[int, int], dict] = {}
    for record in records:
        if record.get('arm') != 'unchanged' or bool(record.get('pilot', False)):
            continue
        try:
            qi = int(record.get('question_index', -1))
            sample = int(record.get('sample', -1))
        except (TypeError, ValueError):
            raise EvidenceError(f'{label}: an unchanged row has no usable question/sample '
                                f'index') from None
        if qi >= sc.N_QUESTIONS:
            continue                       # Part 214's extra indirect/guard questions
        if qi < 0:
            raise EvidenceError(f'{label}: unchanged row with question_index {qi}')
        if sample not in (0, 1):
            continue                       # samples 2..7 exist in Part 214 but are not this reference
        key = (qi, sample)
        if key in out:
            raise EvidenceError(f'{label}: unchanged row (question {qi}, sample {sample}) '
                                f'appears twice')
        pins = record.get('pins')
        if not isinstance(pins, dict):
            raise EvidenceError(f'{label}: row (question {qi}, sample {sample}) has no pins')
        if part214_prompts_sha256 is not None and \
                pins.get('prompts_sha256') != part214_prompts_sha256:
            raise EvidenceError(f'{label}: row (question {qi}, sample {sample}) was generated '
                                f'under {pins.get("prompts_sha256")}, this build uses '
                                f'{part214_prompts_sha256}')
        out[key] = record
    wanted = {(qi, sample) for qi in range(sc.N_QUESTIONS) for sample in (0, 1)}
    if set(out) != wanted:
        missing = sorted(wanted - set(out))
        raise EvidenceError(f'{label}: {len(out)} usable unchanged rows for samples 0/1, want '
                            f'{len(wanted)} (missing {missing[:4]})')
    for (qi, sample), record in sorted(out.items()):
        want = sc.row_seed('neutral', 'identity', 'v0', qi, sample)
        if int(record.get('seed', -1)) != want:
            raise EvidenceError(f'{label}: row (question {qi}, sample {sample}) has seed '
                                f'{record.get("seed")}, want {want} (Part 214 row seed)')
        token_ids = record.get('token_ids')
        if not _int_list(token_ids) or not token_ids:
            raise EvidenceError(f'{label}: row (question {qi}, sample {sample}) has no usable '
                                f'token ids')
    return out


def _recorded_path(explicit, recorded, default, name: str, fails: list[str]) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    if recorded is None:
        return Path(default)
    if not isinstance(recorded, str) or not recorded:
        fails.append(f'{name}_invalid')
        return None
    return Path(recorded)


def _g3_failures(g3) -> list[str]:
    """G3 is rerun, and the stored result must equal the rerun field for field."""
    if not isinstance(g3, dict):
        return ['G3_missing']
    rerun = sc.g3_scorer_check()
    fails: list[str] = []
    if rerun.get('pass') is not True:
        fails.append('G3_rerun_failed')
    for key in ('pass', 'fixtures_pass', 'fixtures_total', 'fixtures_expected',
                'deliberate_break_red', 'rule'):
        if g3.get(key) != rerun.get(key):
            fails.append(f'G3_stored_{key}')
    if g3.get('fixtures') != rerun.get('fixtures'):
        fails.append('G3_stored_fixtures')
    if g3.get('deliberate_break') != rerun.get('deliberate_break'):
        fails.append('G3_stored_deliberate_break')
    return fails


def _g0_failures(g0, stage1: dict | None) -> list[str]:
    """G0 recomputed from its 48 rows; the summary must agree with the recomputation."""
    if not isinstance(g0, dict):
        return ['G0_missing']
    fails: list[str] = []
    tolerance = g0.get('tolerance')
    if not _number(tolerance) or float(tolerance) != G0_TOLERANCE:
        fails.append('G0_tolerance')
    rows = g0.get('rows')
    if not isinstance(rows, list) or len(rows) != 2 * sc.N_QUESTIONS:
        fails.append('G0_rows')
        return fails
    comparisons = g0.get('comparisons') if isinstance(g0.get('comparisons'), dict) else {}
    label_pass: dict[str, bool] = {}
    for label in ('unchanged', 'zero'):
        subset = [row for row in rows if isinstance(row, dict) and row.get('comparison') == label]
        try:
            indices = sorted(int(row.get('question_index', -1)) for row in subset)
        except (TypeError, ValueError):
            fails.append(f'G0_{label}_questions')
            continue
        if len(subset) != sc.N_QUESTIONS or indices != list(range(sc.N_QUESTIONS)):
            fails.append(f'G0_{label}_questions')
            continue
        references = None
        if isinstance(stage1, dict):
            entry = (stage1.get('arms') or {}).get(label)
            values = entry.get('g_ali') if isinstance(entry, dict) else None
            if isinstance(values, list) and len(values) >= sc.N_QUESTIONS:
                references = values
            else:
                fails.append(f'G0_{label}_reference_arm')
        deltas: list[tuple[int, float]] = []
        all_pass = True
        for row in sorted(subset, key=lambda item: int(item['question_index'])):
            qi = int(row['question_index'])
            ours, theirs = row.get('ours'), row.get('part214')
            if not _number(ours) or not _number(theirs):
                fails.append(f'G0_{label}_row_numbers_q{qi}')
                all_pass = False
                continue
            if references is not None and float(theirs) != float(references[qi]):
                fails.append(f'G0_{label}_part214_not_reference_q{qi}')
            delta = abs(float(ours) - float(theirs))
            deltas.append((qi, delta))
            if delta > G0_TOLERANCE:
                fails.append(f'G0_{label}_delta_over_tolerance_q{qi}')
            stored_delta = row.get('abs_delta')
            if not _number(stored_delta) or abs(float(stored_delta) - delta) > _REL_TOL:
                fails.append(f'G0_{label}_row_abs_delta_q{qi}')
            row_pass = delta <= G0_TOLERANCE
            if row.get('pass') is not row_pass:
                fails.append(f'G0_{label}_row_pass_q{qi}')
            all_pass = all_pass and row_pass
        label_pass[label] = all_pass
        if len(deltas) == sc.N_QUESTIONS:
            worst_qi, max_delta = max(deltas, key=lambda item: item[1])
            summary = comparisons.get(label)
            if not isinstance(summary, dict):
                fails.append(f'G0_{label}_summary')
            else:
                if summary.get('n') != sc.N_QUESTIONS:
                    fails.append(f'G0_{label}_summary_n')
                if summary.get('pass') is not all_pass:
                    fails.append(f'G0_{label}_summary_pass')
                if not _number(summary.get('max_abs_delta')) or \
                        abs(float(summary['max_abs_delta']) - max_delta) > _REL_TOL:
                    fails.append(f'G0_{label}_summary_max_delta')
                if summary.get('worst_question') != worst_qi:
                    fails.append(f'G0_{label}_summary_worst_question')
    if len(label_pass) == 2 and g0.get('pass') is not all(label_pass.values()):
        fails.append('G0_summary_pass')
    return fails


def _g1_failures(g1, stage2: dict | None) -> list[str]:
    """G1 recomputed from its 48 rows against the pinned Part 214 Stage 2 token ids."""
    if not isinstance(g1, dict):
        return ['G1_missing']
    fails: list[str] = []
    rows = g1.get('rows')
    if not isinstance(rows, list) or len(rows) != 2 * sc.N_QUESTIONS:
        fails.append('G1_rows')
        return fails
    wanted = {(qi, sample) for qi in range(sc.N_QUESTIONS) for sample in (0, 1)}
    seen: set[tuple[int, int]] = set()
    duplicates: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            fails.append('G1_row_shape')
            continue
        try:
            key = (int(row.get('question_index', -1)), int(row.get('sample', -1)))
        except (TypeError, ValueError):
            fails.append('G1_row_key')
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if seen != wanted or duplicates:
        fails.append('G1_keys')
    matched = 0
    recomputed = stage2 is not None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            qi = int(row.get('question_index', -1))
            sample = int(row.get('sample', -1))
        except (TypeError, ValueError):
            continue
        want_seed = sc.row_seed('neutral', 'identity', 'v0', qi, sample)
        if row.get('seed') != want_seed:
            fails.append(f'G1_seed_q{qi}_s{sample}')
        ours = row.get('ours')
        if not _int_list(ours):
            fails.append(f'G1_ours_q{qi}_s{sample}')
            recomputed = False
            continue
        if stage2 is not None:
            reference = stage2.get((qi, sample))
            if reference is None:
                fails.append(f'G1_reference_q{qi}_s{sample}')
                recomputed = False
                continue
            reference_ids = reference['token_ids']
            same = [int(token) for token in ours] == [int(token) for token in reference_ids]
            if row.get('matched') is not same:
                fails.append(f'G1_matched_q{qi}_s{sample}')
            if row.get('part214') != reference_ids:
                fails.append(f'G1_part214_copy_q{qi}_s{sample}')
            matched += int(same)
    if g1.get('n') != 2 * sc.N_QUESTIONS:
        fails.append('G1_n')
    if g1.get('needed') != G1_NEEDED:
        fails.append('G1_needed')
    if recomputed:
        if matched < G1_NEEDED:
            fails.append(f'G1_matched_below_{G1_NEEDED}')
        if g1.get('matched') != matched:
            fails.append('G1_matched_count')
        if g1.get('pass') is not (matched >= G1_NEEDED):
            fails.append('G1_summary_pass')
    elif g1.get('pass') is not True:
        fails.append('G1_summary_pass')
    return fails


def _g2_failures(pilot) -> list[str]:
    """G2 recomputed from its 96 answers with the frozen broken rule."""
    if not isinstance(pilot, dict):
        return ['G2_missing']
    fails: list[str] = []
    rows = pilot.get('rows')
    if not isinstance(rows, list) or len(rows) != PILOT_N:
        fails.append('G2_rows')
        return fails
    planned = {(row['variant'], row['question_index']) for row in sc.pilot_rows()}
    seen: set[tuple] = set()
    duplicates: set[tuple] = set()
    for row in rows:
        if not isinstance(row, dict):
            fails.append('G2_row_shape')
            continue
        if not isinstance(row.get('variant'), str):
            fails.append('G2_row_variant')
            continue
        try:
            key = (row['variant'], int(row.get('question_index', -1)))
        except (TypeError, ValueError):
            fails.append('G2_row_key')
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if seen != planned or duplicates:
        fails.append('G2_keys')
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            qi = int(row.get('question_index', -1))
            sample = int(row.get('sample', -1))
        except (TypeError, ValueError):
            fails.append('G2_row_key')
            continue
        variant = row.get('variant')
        if row.get('arm') != 'rude_v':
            fails.append(f'G2_arm_q{qi}')
        if sample != 0:
            fails.append(f'G2_sample_q{qi}')
        if variant in sc.ARM_VARIANTS['rude_v'] and 0 <= qi < sc.N_QUESTIONS:
            if row.get('seed') != sc.pilot_seed(variant, qi, 0):
                fails.append(f'G2_seed_{variant}_q{qi}')
    if any(not isinstance(row.get('answer'), str) for row in rows if isinstance(row, dict)):
        fails.append('G2_answers')
    broken = sum(1 for row in rows if isinstance(row, dict)
                 and isinstance(row.get('answer'), str) and sc.is_broken(row['answer']))
    if broken > PILOT_BOUND:
        fails.append(f'G2_broken_recomputed_{broken}_over_{PILOT_BOUND}')
    if pilot.get('broken') != broken:
        fails.append('G2_broken_count')
    if pilot.get('bound') != PILOT_BOUND:
        fails.append('G2_bound')
    if pilot.get('n') != PILOT_N or pilot.get('planned') != PILOT_N:
        fails.append('G2_n')
    if pilot.get('deadline_reached') is not False:
        fails.append('G2_deadline')
    expected_pass = len(rows) == PILOT_N and broken <= PILOT_BOUND and not duplicates
    if pilot.get('pass') is not expected_pass:
        fails.append('G2_summary_pass')
    return fails


def _recompute_latency(gates) -> dict:
    """Every measured row's latency, recomputed from the three gate row lists (192 rows)."""
    groups: dict[str, list | None] = {}
    for name in ('G0', 'G1', 'G2_pilot'):
        gate = gates.get(name) if isinstance(gates, dict) else None
        rows = gate.get('rows') if isinstance(gate, dict) else None
        groups[name] = ([row.get('latency_s') for row in rows if isinstance(row, dict)]
                        if isinstance(rows, list) else None)
    generation = (groups.get('G1') or []) + (groups.get('G2_pilot') or [])
    forward = groups.get('G0') or []
    values = generation + forward
    numbers = [float(value) for value in values if _number(value)]
    gen_numbers = [float(value) for value in generation if _number(value)]
    fwd_numbers = [float(value) for value in forward if _number(value)]
    return {
        'n_generation_calls': len(generation),
        'n_forward_calls': len(forward),
        'n_measured': len(values),
        'all_numbers': len(numbers) == len(values),
        'nonnegative': all(value >= 0 for value in numbers),
        'max_s': max(numbers) if numbers else None,
        'generation_max_s': max(gen_numbers) if gen_numbers else None,
        'generation_mean_s': (sum(gen_numbers) / len(gen_numbers)) if gen_numbers else None,
        'forward_max_s': max(fwd_numbers) if fwd_numbers else None,
        'forward_mean_s': (sum(fwd_numbers) / len(fwd_numbers)) if fwd_numbers else None,
    }



WARMUP_SEED = 215_000_001   # Amendment 2: the untimed warm-up generation's own seed (enters no gate)


def _warmup_failures(warmup) -> list[str]:
    """Amendment 2: the preflight made one untimed warm-up generation before G1, used in no gate."""
    if not isinstance(warmup, dict):
        return ['warmup_missing']
    fails = []
    if not _number(warmup.get('latency_s')) or float(warmup['latency_s']) < 0:
        fails.append('warmup_latency')
    if warmup.get('seed') != WARMUP_SEED:
        fails.append('warmup_seed')
    if warmup.get('used_in_gates') is not False:
        fails.append('warmup_used_in_gates')
    return fails

def _latency_failures(latency, gates) -> list[str]:
    """The summary must equal the recomputation over all 192 measured rows."""
    if not isinstance(latency, dict):
        return ['latency_missing']
    fails: list[str] = []
    recomputed = _recompute_latency(gates)
    if not recomputed['all_numbers'] or not recomputed['nonnegative']:
        fails.append('latency_row_values')
    if recomputed['n_generation_calls'] != PREFLIGHT_GENERATION_CALLS:
        fails.append('latency_generation_rows')
    if recomputed['n_forward_calls'] != PREFLIGHT_FORWARD_CALLS:
        fails.append('latency_forward_rows')
    if recomputed['n_measured'] != MEASURED_CALLS:
        fails.append('latency_measured_rows')
    limit = latency.get('limit_s')
    if not _number(limit) or float(limit) != MAX_LATENCY_S:
        fails.append('latency_limit')
    max_s = recomputed['max_s']
    # Amendment 2: the gate reads the MEAN per call (generation and forward separately); the max is
    # recomputed and must agree with the summary, but no longer decides the gate.
    gen_mean, fwd_mean = recomputed['generation_mean_s'], recomputed['forward_mean_s']
    means_ok = (gen_mean is not None and fwd_mean is not None
                and 0 < gen_mean <= MAX_LATENCY_S and 0 < fwd_mean <= MAX_LATENCY_S)
    if not means_ok:
        fails.append('latency_mean_recomputed')
    expected_pass = (recomputed['n_measured'] == MEASURED_CALLS and recomputed['all_numbers']
                     and recomputed['nonnegative'] and means_ok)
    if latency.get('pass') is not expected_pass:
        fails.append('latency_summary_pass')
    for key, value in (('max_s', max_s),
                       ('generation_max_s', recomputed['generation_max_s']),
                       ('forward_max_s', recomputed['forward_max_s'])):
        if value is None or not _number(latency.get(key)) or \
                abs(float(latency[key]) - float(value)) > _REL_TOL:
            fails.append(f'latency_{key}_disagrees')
    for key, value in (('generation_mean_s', recomputed['generation_mean_s']),
                       ('forward_mean_s', recomputed['forward_mean_s'])):
        if value is None or not _close(latency.get(key), value):
            fails.append(f'latency_{key}_disagrees')
    if latency.get('n_generation_calls') != recomputed['n_generation_calls']:
        fails.append('latency_summary_generation_calls')
    if latency.get('n_forward_calls') != recomputed['n_forward_calls']:
        fails.append('latency_summary_forward_calls')
    return fails


def validate_preflight(receipt, *, stage1_path=None, stage2_path=None,
                       stage1_sha256: str | None = None, stage2_sha256: str | None = None,
                       part214_prompts_sha256: str | None = None) -> list[str]:
    """Every failure of a preflight receipt's measured evidence; an empty list is admissible.

    `stage1_path` / `stage2_path` override the paths recorded in the receipt (which override
    the reviewed repo defaults).  `stage1_sha256` / `stage2_sha256` override the module pins,
    which is the seam the tests use exactly as `run215.PART214_STAGE1_SHA256` is one.
    """
    if not isinstance(receipt, dict):
        return ['preflight_receipt_not_an_object']
    gates = receipt.get('gates')
    if not isinstance(gates, dict):
        return ['gates_missing']
    fails: list[str] = []
    fails.extend(_g3_failures(gates.get('G3')))

    pin1 = PART214_STAGE1_SHA256 if stage1_sha256 is None else stage1_sha256
    pin2 = PART214_STAGE2_SHA256 if stage2_sha256 is None else stage2_sha256
    if receipt.get('part214_stage1_sha256') != pin1:
        fails.append('part214_stage1_pin')
    if receipt.get('part214_stage2_sha256') != pin2:
        fails.append('part214_stage2_pin')

    stage1 = stage2 = None
    path1 = _recorded_path(stage1_path, receipt.get('part214_stage1_path'),
                           DEFAULT_STAGE1_PATH, 'part214_stage1_path', fails)
    path2 = _recorded_path(stage2_path, receipt.get('part214_stage2_path'),
                           DEFAULT_STAGE2_PATH, 'part214_stage2_path', fails)
    if path1 is not None:
        try:
            stage1 = stage1_reference(path1.read_bytes(), pin1, part214_prompts_sha256,
                                      label=str(path1))
        except (OSError, EvidenceError) as exc:
            fails.append(f'part214_stage1_reference: {exc}')
    if path2 is not None:
        try:
            stage2 = stage2_reference(path2.read_bytes(), pin2, part214_prompts_sha256,
                                      label=str(path2))
        except (OSError, EvidenceError) as exc:
            fails.append(f'part214_stage2_reference: {exc}')

    fails.extend(_g0_failures(gates.get('G0'), stage1))
    fails.extend(_g1_failures(gates.get('G1'), stage2))
    fails.extend(_g2_failures(gates.get('G2_pilot')))
    fails.extend(_latency_failures(receipt.get('latency'), gates))
    fails.extend(_warmup_failures(receipt.get('warmup') if isinstance(receipt, dict) else None))
    return fails
