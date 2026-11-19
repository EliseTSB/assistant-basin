#!/usr/bin/env python3
"""Part 212 runner: Stage 1 (4-cell factorial & mechanism audit), Stage 2 (held-out free generation), pilot.

PART212-DESIGN.md v3 is the contract. Every place the design is ambiguous is marked `AMBIGUITY n` below and listed
with its design line number in part212/BUILD-REPORT.md.

Run under Windows Python as `python -I -B part212/run212.py <sub> --out <relative dir>`: the script's folder is not on
sys.path, so the repo root is inserted the way part212/build_prompts212.py does. No POSIX-only modules; every file is
opened with encoding="utf-8"; every output path is relative to --out.

Reused, not rewritten: neuron_report.load_model / build_prompt / find_text_layers / spill_guard / LAST_LOAD,
part66.part66.row_seed / _pad / _stop_reason and its generation loop shape, part66.score66.FAMILIES.

The pieces that decide what gets clamped are pure functions of tensors and small dicts (profile mapping, Stage 1 value
sources, Stage 2 prefill/decode indexing, logit readouts, write-norm pools, pilot rule) so part212/test_part212.py can
exercise them on CPU with synthetic tensors and no model.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import neuron_report as nr  # noqa: E402
from part66 import part66 as p66  # noqa: E402  (row_seed, _pad, _stop_reason, the generation loop)

MODEL = p66.MODEL
REVISION = p66.REVISION
TARGET = p66.TARGET
PROMPTS_PATH = Path(__file__).resolve().with_name('prompts212.json')
PROMPTS_SHA256 = '0f10c337ae64765b9f1cc32a85241214944d1c2b55be15d8398a085042793227'
N_HELDOUT = 24
POSITIVE_INDEX = 24
MAX_NEW_TOKENS = 48
STAGE1_BATCH = 32
STAGE2_BATCH = 24
PILOT_BATCH = 8
N_RANDOM_SETS = 20
NORM_POOL = 50
DONOR_BANKS = ('d_fact', 'd_you', 'd_zh_maker')
TRACE_BANKS = ('d_fact', 'd_you', 'd_zh_maker', 'd_zh_fact')
BANK_OPENING = {'en_maker': 'en_maker', 'd_fact': 'en_neutral', 'd_you': 'en_neutral',
                'd_zh_maker': 'zh_maker', 'd_zh_fact': 'zh_neutral'}
MAIN_ARMS = ('unchanged', 'self_patch', 'zero', 'trace_resample_fact', 'trace_resample_you',
             'mean_resample_fact', 'sub_factual', 'sub_factual_online', 'sub_you', 'scale_pos2',
             'scale_neg1', 'zero_prompt_only', 'zero_opening_only', 'trace_resample_zh_maker',
             'trace_resample_zh_fact')
PER_UNIT_ARMS = ('zero', 'trace_resample_fact', 'trace_resample_you', 'sub_factual')
RANDOM_ARMS = ('zero', 'trace_resample_fact', 'trace_resample_you', 'sub_factual', 'scale_neg1')
STAGE2_FULL_ARMS = ('unchanged', 'zero', 'mean_resample_fact', 'sub_factual', 'scale_pos2', 'scale_neg1')
STAGE2_TRACE_ARMS = ('trace_resample_fact', 'trace_resample_you')
STAGE2_DIAG_ARMS = ('trace_resample_zh_maker', 'zero_prompt_only', 'zero_gen_only')
STAGE2_RANDOM_ARMS = ('zero', 'trace_resample_fact', 'sub_factual')
PILOT_ARMS = ('unchanged', 'zero', 'trace_resample_fact', 'sub_factual')
PILOT_QUESTIONS = (0, 5, 12, 16)
PILOT_SAMPLES = 4
PILOT_POSITIVE_SAMPLES = 24
GUARD_SAMPLES = 4  # design line 195: 6 fluency guards x 4 samples per full arm
FULL_ARM_ROWS = N_HELDOUT * 8 + 24 + 6 * GUARD_SAMPLES  # 240
OPENING_TOKENS = 5
NON_FINITE = 3.0e38  # |logit| above this is recorded as non-finite (bf16 max is ~3.39e38)

# Design line 149: g_ali = logit(Google) - logsumexp(logit(Al), logit(T)).
GOOGLE_ID, AL_ID, T_ID = 14030, 2033, 51


# Shared output-directory lock.  This is deliberately implemented with portable os APIs so the Windows XPU launcher
# does not need fcntl or another POSIX-only module.
OUTPUT_LOCK_NAME = 'part212.lock'


def pid_is_alive(pid: int) -> bool:
    """Portable best-effort liveness check; unknown OS errors are treated as alive, never stolen."""
    pid = int(pid)
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error == 5:  # ERROR_ACCESS_DENIED: the PID may be alive but not queryable
                return True
            if error in (6, 87, 1168):  # invalid handle / invalid parameter / not found
                return False
            return True
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


@contextmanager
def output_directory_lock(outdir, stage: str):
    """Hold an O_EXCL lock with the current PID while a Stage 1 or Stage 2 output is being touched."""
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / OUTPUT_LOCK_NAME
    pid = os.getpid()
    payload = f'{pid}\n{pid}-{time.time_ns()}\n'.encode('utf-8')
    try:
        # O_BINARY: on Windows os.open defaults to text mode and writes each \n as \r\n, so the release check below
        # never matched and the pilot's Stage 2 refused the lock its own Stage 1 had left (2026-11-19 12:41).
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    except FileExistsError:
        try:
            first_line = lock_path.read_text(encoding='utf-8').splitlines()[0].strip()
            owner = int(first_line.removeprefix('pid='))
        except (OSError, IndexError, ValueError) as exc:
            raise RuntimeError(f'{stage}: cannot read owner PID from output lock {lock_path}') from exc
        alive = pid_is_alive(owner)
        print(f'{stage}: output lock {lock_path} records PID {owner}; pid_is_alive={alive}. '
              f'Operator should delete this file if the run is not live.', flush=True)
        raise RuntimeError(f'{stage}: output directory {directory} is locked by PID {owner}')
    try:
        os.write(fd, payload)
        os.fsync(fd)
    except BaseException:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        raise
    else:
        os.close(fd)
    try:
        yield
    finally:
        try:
            if lock_path.read_bytes().replace(b'\r\n', b'\n') == payload:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def repair_partial_jsonl(path) -> int:
    """Keep a complete unterminated JSON value, or drop and report an invalid JSONL tail."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open('r+b') as f:
        raw = f.read()
        if not raw or raw.endswith(b'\n'):
            return 0
        keep = raw.rfind(b'\n') + 1
        tail = raw[keep:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            dropped = len(raw) - keep
            f.seek(keep)
            f.truncate()
        else:
            f.seek(0, os.SEEK_END)
            f.write(b'\n')
            f.flush()
            os.fsync(f.fileno())
            return 0
    if dropped:
        print(f'stage2: dropped {dropped} trailing bytes from incomplete JSONL line in {path}', flush=True)
    return dropped


def load_stage2_jsonl(path) -> list[dict]:
    """Load complete JSONL rows with line-specific errors; first repair a torn final append."""
    path = Path(path)
    if not path.exists():
        return []
    repair_partial_jsonl(path)
    rows = []
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'corrupt Stage 2 JSONL row at {path}, line {line_number}: '
                                 f'{exc.msg} (column {exc.colno})') from None
            if not isinstance(row, dict):
                raise ValueError(f'corrupt Stage 2 JSONL row at {path}, line {line_number}: expected a JSON object')
            rows.append(row)
    return rows


def stage2_requested_batch_size(args, pilot: bool) -> int:
    size = int(args.batch or (PILOT_BATCH if pilot else STAGE2_BATCH))
    if size < 1:
        raise ValueError(f'batch_size must be positive, got {size}')
    return size


def stage2_batch_fields(actual_batch_size: int, requested_batch_size: int) -> dict:
    actual, requested = int(actual_batch_size), int(requested_batch_size)
    if actual < 1 or requested < 1:
        raise ValueError(f'batch sizes must be positive, got actual={actual}, requested={requested}')
    return {'batch_size': actual, 'batch_size_requested': requested}


def stage2_batch_histogram(rows: list[dict]) -> dict[str, int]:
    """Aggregate saved per-row sizes into a histogram of complete generation calls."""
    row_counts: dict[int, int] = {}
    for row in rows:
        raw = row.get('batch_size')
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError('Stage 2 saved row is missing a valid positive batch_size')
        row_counts[raw] = row_counts.get(raw, 0) + 1
    histogram = {}
    for size, count in sorted(row_counts.items()):
        if count % size:
            raise ValueError(f'Saved Stage 2 rows contain {count % size} rows of a partial batch_size={size} call')
        histogram[str(size)] = count // size
    return histogram


def stage2_validate_resume(planned: list[dict], done_rows: list[dict], requested_batch_size: int,
                           pilot: bool) -> list[dict]:
    """Validate saved rows and identify partial calls that must be replayed whole."""
    requested = int(requested_batch_size)
    if requested < 1:
        raise ValueError(f'batch_size must be positive, got {requested}')
    planned_by_key = {stage2_resume_key(r['arm'], r['question_index'], r['sample'], r.get('set'), pilot): r
                      for r in planned}
    expected_batches = stage2_batches(planned, requested)
    expected_sizes = {}
    batch_keys = []
    for batch in expected_batches:
        keys = [stage2_resume_key(r['arm'], r['question_index'], r['sample'], r.get('set'), pilot) for r in batch]
        batch_keys.append(keys)
        expected_sizes.update({key: len(batch) for key in keys})

    saved = [r for r in done_rows if bool(r.get('pilot', False)) == bool(pilot)]
    saved_keys = []
    for row in saved:
        if not all(k in row for k in ('arm', 'question_index', 'sample')):
            raise ValueError('existing Stage 2 row is missing its resume key')
        key = stage2_resume_key(row['arm'], row['question_index'], row['sample'], row.get('set'), pilot)
        if key not in planned_by_key:
            raise ValueError(f'existing Stage 2 row has unplanned key {key}')
        if key in saved_keys:
            raise ValueError(f'existing Stage 2 output has duplicate key {key}')
        saved_keys.append(key)
        if 'batch_size' not in row:
            raise ValueError(f'existing Stage 2 row {key} has no batch_size; its generation shape is unknown')
        actual = row['batch_size']
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 1:
            raise ValueError(f'existing Stage 2 row {key} has invalid batch_size {actual!r}')
        if 'batch_size_requested' not in row:
            raise ValueError(f'existing Stage 2 row {key} has no batch_size_requested; refusing an unverifiable resume')
        old_requested = row['batch_size_requested']
        if isinstance(old_requested, bool) or not isinstance(old_requested, int) or old_requested != requested:
            raise ValueError(f'existing Stage 2 rows use requested batch size {old_requested!r}, '
                             f'but this invocation requested {requested}')
        expected = expected_sizes[key]
        if actual != expected:
            raise ValueError(f'existing Stage 2 row {key} records batch_size={actual}, but requested '
                             f'batch_size={requested} plans batch_size={expected}')

    saved_key_set = set(saved_keys)
    partial_calls = []
    for keys in batch_keys:
        present = saved_key_set.intersection(keys)
        if present and len(present) != len(keys):
            first = planned_by_key[keys[0]]
            partial_calls.append({'keys': present, 'size': len(keys), 'first': first})
    return partial_calls


def stage2_drop_partial_calls(path, done_rows: list[dict], partial_calls: list[dict], pilot: bool) -> int:
    """Atomically remove persisted rows from incomplete calls so resume can regenerate each call whole."""
    path = Path(path)
    partial_keys = set().union(*(call['keys'] for call in partial_calls))
    keep_rows = []
    dropped = 0
    for row in done_rows:
        key = stage2_resume_key(row['arm'], row['question_index'], row['sample'], row.get('set'), pilot)
        if bool(row.get('pilot', False)) == bool(pilot) and key in partial_keys:
            dropped += 1
        else:
            keep_rows.append(row)

    temp_path = path.with_name(f'.{path.name}.{os.getpid()}-{time.time_ns()}.tmp')
    try:
        with temp_path.open('x', encoding='utf-8') as sink:
            for row in keep_rows:
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    calls = ', '.join(
        f'arm={call["first"]["arm"]}, question={call["first"]["question_index"]}, '
        f'set={call["first"].get("set")} ({len(call["keys"])}/{call["size"]})'
        for call in partial_calls)
    print(f'stage2: dropped {dropped} rows from partial generation calls: {calls}', flush=True)
    return dropped


# ----------------------------------------------------------------------------------------------------------------
# prompt bank
# ----------------------------------------------------------------------------------------------------------------

def load_prompts(path=PROMPTS_PATH) -> dict:
    """Load prompts212.json; refuse a file whose SHA-256 is not the pinned Gate 0 value."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROMPTS_SHA256:
        raise SystemExit(f'prompts212.json SHA-256 {digest} != pinned {PROMPTS_SHA256}; refusing to run')
    return json.loads(raw.decode('utf-8'))


def p66_ids(tok, text: str) -> list[int]:
    return [int(x) for x in tok(text, add_special_tokens=False)['input_ids']]


def opening_start(prompt_ids: list[int], opening_ids: list[int]) -> int:
    """T: first index of the 5-token forced opening (the prompt must end with it).

    The chat template is applied to the question and the opening is appended to the rendered text, so the opening is
    the last 5 tokens by construction; a mismatch means the tokenizer or template changed under the pinned file.
    """
    n = len(opening_ids)
    if len(prompt_ids) < n or list(prompt_ids[-n:]) != list(opening_ids):
        raise ValueError(f'prompt does not end with its {n}-token forced opening {list(opening_ids)}: '
                         f'last tokens were {list(prompt_ids[-n:])}')
    return len(prompt_ids) - n


def stage1_prompt_ids(tok, bank: dict, qi: int, name: str = 'en_maker') -> tuple[list[int], int]:
    """Full Stage 1 prompt for bank `name`[qi] (chat prompt + that bank's 5-token opening). Returns (ids, T)."""
    opening = bank['openings'][BANK_OPENING[name]]
    text = nr.build_prompt(tok, bank['items'][qi][name], False) + opening['text']
    ids = p66_ids(tok, text)
    T = opening_start(ids, opening['token_ids'])
    expected = int(bank['items'][qi]['prompt_tokens']) + OPENING_TOKENS
    if len(ids) != expected:
        # AMBIGUITY 7 (design lines 78-96): the tokenizer, chat template or prompts212.json no longer agree with the
        # pinned Gate 0 file. Refuse rather than run a mis-aligned bank; --allow-length-drift runs anyway and records it.
        message = f'{name}[{qi}] has {len(ids)} tokens, prompts212.json implies {expected}'
        if not allow_length_drift():
            # F10: the switch is the environment variable, not a CLI flag argparse does not define
            raise ValueError(message + ' (pinned Gate 0 parity broken; set PART212_ALLOW_LENGTH_DRIFT=1 to run anyway)')
        print('WARNING: ' + message, flush=True)
    return ids, T


def allow_length_drift() -> bool:
    return os.environ.get('PART212_ALLOW_LENGTH_DRIFT') == '1'


# ----------------------------------------------------------------------------------------------------------------
# bank-mean profiles and their mapping onto a recipient (design lines 113-121, v3)
# ----------------------------------------------------------------------------------------------------------------

def recipient_slots(T_q: int, h: int, s: int) -> list[tuple[str, int]]:
    """One (kind, index) per position 0..T_q+4 of a recipient prompt of length T_q.

    header 0..h-1 -> ('header', p); body h..T_q-s-1 -> ('body', p); suffix T_q-s..T_q-1 -> ('suffix', p-(T_q-s));
    opening T_q..T_q+4 -> ('opening', p-T_q). Body positions all read the one pooled body mean.
    """
    if T_q <= h + s:
        raise ValueError(f'prompt of {T_q} tokens has no body positions (header {h}, suffix {s})')
    slots: list[tuple[str, int]] = []
    for p in range(T_q):
        if p < h:
            slots.append(('header', p))
        elif p < T_q - s:
            slots.append(('body', p))
        else:
            slots.append(('suffix', p - (T_q - s)))
    for j in range(OPENING_TOKENS):
        slots.append(('opening', j))
    return slots


def profile_index(T_q: int, h: int, s: int) -> np.ndarray:
    """Row of the bank profile that each recipient position 0..T_q+4 reads (design lines 117-121).

    The profile is stored in bank order: header rows 0..h-1, ONE pooled body row at h, suffix rows h+1..h+s (in suffix
    order), opening rows h+s+1..h+s+5. Every body position of the recipient reads the single pooled row h, so a
    recipient of any length reads the same body mean.
    """
    if T_q <= h + s:
        raise ValueError(f'prompt of {T_q} tokens has no body positions (header {h}, suffix {s})')
    idx = []
    for p in range(T_q):
        if p < h:
            idx.append(p)
        elif p < T_q - s:
            idx.append(h)
        else:
            idx.append(h + 1 + (p - (T_q - s)))
    idx.extend(h + s + 1 + j for j in range(OPENING_TOKENS))
    return np.asarray(idx, dtype=np.int64)


def build_profile(donors, h: int, s: int, with_opening: bool = True) -> np.ndarray:
    """Bank profile built slot-wise from donors that need NOT share one prompt length (design lines 113-121).

    `donors` is a sequence of per-donor recorded arrays of shape (T_d, K). T_d is that donor's own prompt length
    (plus OPENING_TOKENS when `with_opening`); the 24 held-out questions are 16-30 tokens long and each donor is
    1-to-1 length matched to its own EN_MAKER question, so the donors differ from each other. The design's slots:
      header  = per-position mean over the donors at positions 0..h-1;
      body    = ONE pooled mean over every donor's body rows h..T_d-s-1 (all donors' body tokens pooled together, not a
                mean of per-donor means);
      suffix  = per-position mean over each donor's own last s prompt positions, in suffix order;
      opening = per-position mean over the 5 forced-opening positions.
    Returns float64 of shape (h + s + 1 + OPENING_TOKENS, K) in the bank order `profile_index` expects: header rows
    0..h-1, ONE pooled body row at h, suffix rows h+1..h+s, opening rows h+s+1..h+s+5.
    """
    arrs = [np.asarray(d, dtype=np.float64) for d in donors]
    if not arrs:
        raise ValueError('no donors to build a profile from')
    if any(a.ndim != 2 for a in arrs):
        raise ValueError(f'donor arrays must be 2-D (T_d, K), got shapes {[a.shape for a in arrs]}')
    K = arrs[0].shape[-1]
    prompts = []
    for i, a in enumerate(arrs):
        if a.shape[-1] != K:
            raise ValueError(f'donor {i} has shape {a.shape}, expected (T_d, {K})')
        T_d = a.shape[0] - (OPENING_TOKENS if with_opening else 0)
        if T_d <= h + s:
            raise ValueError(f'donor {i} has {T_d} prompt tokens, no body positions (header {h}, suffix {s})')
        prompts.append(a[:T_d])
    out = np.zeros((h + s + 1 + OPENING_TOKENS, K), dtype=np.float64)
    out[0:h] = np.mean([a[:h] for a in prompts], axis=0)
    out[h] = np.concatenate([a[h:a.shape[0] - s] for a in prompts], axis=0).mean(axis=0)
    out[h + 1:h + 1 + s] = np.mean([a[a.shape[0] - s:] for a in prompts], axis=0)
    if with_opening:
        out[h + s + 1:] = np.mean([a[-OPENING_TOKENS:] for a in arrs], axis=0)
    return out


def profile_at(profile: np.ndarray, T_q: int, h: int, s: int) -> np.ndarray:
    """The profile rows in recipient order for a recipient of length T_q (shape (T_q + 5, K))."""
    return profile[profile_index(T_q, h, s)]


# ----------------------------------------------------------------------------------------------------------------
# write-norm matched random sets (design lines 98-107)
# ----------------------------------------------------------------------------------------------------------------

def accumulate_abs(sums: dict, counts: dict, layer: int, activation) -> None:
    """Add one forward's |a| per unit to the float64 CPU accumulators (design lines 98-102).

    `activation` is that layer's down_proj input, shape (B, T, n_units), on whatever device the model runs. One
    vectorised reduction per forward, one CPU copy, no `.item()` per value; the per-forward sum is widened to float64
    before it is added, so the 24-prompt total is accumulated in float64 on CPU.
    """
    a = activation.detach().abs().float().sum(dim=tuple(range(activation.dim() - 1)))
    sums[int(layer)] += np.asarray(a.cpu(), dtype=np.float64)
    counts[int(layer)] = int(counts.get(int(layer), 0)) + int(np.prod(activation.shape[:-1]))


def mean_abs_activations(abs_sums: dict, counts: dict) -> dict:
    """mean_{q,p} |a_{L,u}| per unit from the float64 sums of the unchanged EN_MAKER pass."""
    out = {}
    for layer, total in abs_sums.items():
        n = int(counts.get(int(layer), 0))
        if n <= 0:
            raise ValueError(f'layer {layer}: no positions were accumulated for the write norm')
        out[int(layer)] = np.asarray(total, dtype=np.float64) / float(n)
    return out


def unit_write_norms(abs_sums: dict, counts: dict, down_weight: np.ndarray, units=TARGET) -> dict:
    """m_{L,u} = mean over (prompt, position) of |a_u| times ||w_down[:, u]||_2 for every unit of the target layers.

    abs_sums/counts come from the unchanged EN_MAKER recording pass over the 24 held-out prompts (design line 111);
    down_weight is one (d_model, n_units) down_proj matrix per layer, in `units` order (down_weights(layers)).
    This is the design's activation-weighted match, not the weight norm alone.
    """
    means = mean_abs_activations(abs_sums, counts)
    out = {}
    for k, (layer, _unit) in enumerate(units):
        w = np.asarray(down_weight[k], dtype=np.float64)
        a = means[int(layer)]
        if a.shape[0] != w.shape[1]:
            raise ValueError(f'{a.shape[0]} accumulated units but down_proj has {w.shape[1]} columns')
        out[int(layer)] = a * np.linalg.norm(w, axis=0)
    return out


def write_norm_pools(maker_norm: dict, n_units: int, target=TARGET, pool: int = NORM_POOL) -> dict:
    """Per layer, the `pool` units closest to that layer's target in m_{L,u}, target excluded, ties by unit index."""
    pools = {}
    for layer, unit in target:
        row = np.asarray(maker_norm[layer], dtype=np.float64)
        if row.shape[0] != n_units:
            raise ValueError(f'layer {layer} has {row.shape[0]} write norms, model has {n_units} units')
        target_m = row[int(unit)]
        order = sorted((u for u in range(n_units) if u != int(unit)), key=lambda u: (abs(row[u] - target_m), u))
        pools[int(layer)] = [int(u) for u in order[:pool]]
    return pools


def draw_random_set(pools: dict, k: int) -> list[tuple[int, int]]:
    """random_k: one unit per layer drawn from that layer's pool with np.random.default_rng(21200 + k)."""
    rng = np.random.default_rng(21200 + k)
    return [(int(layer), int(rng.choice(pools[layer]))) for layer in sorted(pools)]


def random_sets(pools: dict, n: int = N_RANDOM_SETS) -> dict:
    return {f'random_{k}': draw_random_set(pools, k) for k in range(n)}


# ----------------------------------------------------------------------------------------------------------------
# Stage 1 value sources: what each arm clamps each target unit to, at each position 0..T+4
# ----------------------------------------------------------------------------------------------------------------
# A source returns one (kind, index) per position. Kinds: 'clean' (frozen a_clean(p)), 'donor' (frozen paired donor
# trace at p), 'zero', 'scale' (factor * a_clean(p)), 'sub' (frozen a_clean(p) - profile row at the slot), and the
# four profile slots 'header'/'body'/'suffix'/'opening'.

def clean_source(T_q, h, s):
    return [('clean', p) for p in range(T_q + OPENING_TOKENS)]


def trace_source(T_q, h, s):
    """Frozen paired donor trace at every position 0..T+4 (design line 133)."""
    return [('donor', p) for p in range(T_q + OPENING_TOKENS)]


def mean_source(T_q, h, s):
    """Frozen bank-mean profile mapped by header/body/suffix/opening slots (design lines 117-121)."""
    return recipient_slots(T_q, h, s)


def zero_source(T_q, h, s):
    return [('zero', p) for p in range(T_q + OPENING_TOKENS)]


def zero_prompt_source(T_q, h, s):
    """zero on prompt positions 0..T-1 only; opening T..T+4 untouched (design line 140)."""
    return [('zero', p) if p < T_q else ('clean', p) for p in range(T_q + OPENING_TOKENS)]


def zero_opening_source(T_q, h, s):
    """zero on the 5 forced-opening positions T..T+4 only (design line 141)."""
    return [('zero', p) if p >= T_q else ('clean', p) for p in range(T_q + OPENING_TOKENS)]


def sub_source(T_q, h, s):
    """Frozen a_clean(p) - r(p); r(p) is the profile mapped by slots (design lines 135, 117-121)."""
    return [('sub', p) for p in range(T_q + OPENING_TOKENS)]


def scale_source(factor: float):
    def source(T_q, h, s):
        return [('scale', p) for p in range(T_q + OPENING_TOKENS)]
    source.factor = float(factor)
    return source


# arm -> (mode, source). mode 'frozen' reads recorded tensors; 'online' reads the live activation (diagnostic).
STAGE1_ARM_SPEC = {
    'unchanged': None,
    'self_patch': ('frozen', clean_source),
    'zero': ('frozen', zero_source),
    'trace_resample_fact': ('frozen', trace_source),
    'trace_resample_you': ('frozen', trace_source),
    'mean_resample_fact': ('frozen', mean_source),
    'sub_factual': ('frozen', sub_source),
    'sub_factual_online': ('online', 'sub'),
    'sub_you': ('frozen', sub_source),
    'scale_pos2': ('frozen', scale_source(2.0)),
    'scale_neg1': ('frozen', scale_source(-1.0)),
    'zero_prompt_only': ('frozen', zero_prompt_source),
    'zero_opening_only': ('frozen', zero_opening_source),
    'trace_resample_zh_maker': ('frozen', trace_source),
    'trace_resample_zh_fact': ('frozen', trace_source),
}
TRACE_ARM_BANK = {'trace_resample_fact': 'd_fact', 'trace_resample_you': 'd_you',
                  'trace_resample_zh_maker': 'd_zh_maker', 'trace_resample_zh_fact': 'd_zh_fact'}
SUB_ARM_PROFILE = {'sub_factual': 'd_fact', 'sub_factual_online': 'd_fact', 'sub_you': 'd_you',
                   'mean_resample_fact': 'd_fact'}


def stage1_values(arm: str, q: int, T_q: int, h: int, s: int, rec: dict, profiles: dict) -> np.ndarray:
    """Frozen target values for one arm/question: float64 (T_q + 5, K) on CPU, never on the device.

    `arm` may be a main arm or a derived arm (`unit:L:U:base`, `random_k:base`); the source, the trace bank and the
    subtraction profile all come from its BASE arm (parse_arm), so a derived arm clamps exactly like its base.
    rec: {'clean': (T+5, K), 'd_fact'/'d_you'/...: (T+5, K)} for THIS question, K columns in `arm_units` order. For a
    random set that means each bank holds THAT SET'S OWN units' recording (F3): every unit is clamped to its own paired
    donor trace / its own a_clean - r / its own -a_clean, never to the TARGET units' values.
    profiles: bank -> profile in bank order, see build_profile; K columns in `arm_units` order.
    """
    base = parse_arm(arm)[1]
    spec = STAGE1_ARM_SPEC[base]
    if spec is None or spec[0] != 'frozen':
        raise ValueError(f'{arm} has no frozen values')
    source = spec[1]
    slots = source(T_q, h, s)
    clean = np.asarray(rec['clean'], dtype=np.float64)
    donor = np.asarray(rec[TRACE_ARM_BANK[base]], dtype=np.float64) if base in TRACE_ARM_BANK else None
    mapped = (profile_at(profiles[SUB_ARM_PROFILE[base]], T_q, h, s) if base in SUB_ARM_PROFILE else None)
    if donor is not None and donor.shape != clean.shape:
        # F3: a random set's `d_fact` must be ITS OWN units' recorded trace, not the four TARGET columns' trace
        raise ValueError(f'{arm}: donor trace has shape {donor.shape}, a_clean has {clean.shape}')
    if mapped is not None and mapped.shape[-1] != clean.shape[-1]:
        # the profiles are built from the full recording (TARGET + pools + monitors); keep the TARGET columns
        raise ValueError(f'{arm}: profile has {mapped.shape[-1]} units, a_clean has {clean.shape[-1]}')
    if clean.shape[0] != len(slots):
        raise ValueError(f'{arm}: a_clean has {clean.shape[0]} positions, the arm addresses {len(slots)}')
    out = np.empty_like(clean)
    for p, (kind, _idx) in enumerate(slots):
        if kind == 'clean':
            out[p] = clean[p]
        elif kind == 'donor':
            out[p] = donor[p]
        elif kind == 'zero':
            out[p] = 0.0
        elif kind == 'scale':
            out[p] = source.factor * clean[p]
        elif kind == 'sub':
            out[p] = clean[p] - mapped[p]
        else:  # header / body / suffix / opening: the mapped profile row for this recipient position
            out[p] = mapped[p]
    return out


def realized_perturbation(values: np.ndarray, clean: np.ndarray, down_weight: np.ndarray) -> float:
    """Mean over positions of ||sum_k (values - clean)[p, k] * w_down[:, k]||_2 for the K clamped units (line 105)."""
    delta = np.asarray(values, dtype=np.float64) - np.asarray(clean, dtype=np.float64)   # (T+5, K)
    w = np.asarray(down_weight, dtype=np.float64)                                       # (d_model, K)
    resid = np.einsum('pk,dk->pd', delta, w)
    return float(np.linalg.norm(resid, axis=1).mean())


def batch_size_histogram(batch_sizes) -> dict[str, int]:
    """Count forwards/generation calls by their input batch size for explicit run metadata (F7)."""
    counts: dict[str, int] = {}
    for size in batch_sizes:
        key = str(int(size))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def stage1_forward_batch_histogram() -> dict[str, int]:
    """Actual Stage 1 call histogram: this implementation currently runs every forward with one prompt."""
    n_questions = N_HELDOUT + 1
    calls = (N_HELDOUT + len(BANK_OPENING) * n_questions + n_questions
             + len(stage1_arm_names()) * n_questions)
    return batch_size_histogram([1] * calls)


# ----------------------------------------------------------------------------------------------------------------
# hooks
# ----------------------------------------------------------------------------------------------------------------

def _columns(units: list[tuple[int, int]], layer: int) -> list[int]:
    return [u for l, u in units if l == layer]


def install_stage1_hooks(layers, units, frozen=None, online_profile=None, online_factor=None):
    """Forward pre-hooks on layers[L].mlp.down_proj (Parts 25, 66, 74, 183).

    frozen: {layer: (B, T+5, K_layer) float array} copied to the device once per forward, K_layer in `_columns`
    order; every clamped unit is set at every position 0..T+4. online_profile: {layer: (T+5, K_layer)} for
    sub_factual_online (x_live - r). online_factor: float for online scaling. A decode call (width 1) is left
    untouched: Stage 1 never decodes.
    """
    handles = []
    for li in sorted({int(l) for l, _ in units}):
        cols = _columns(units, li)

        def hook(_module, args, li=li, cols=cols):
            x = args[0]
            if x.shape[1] == 1:
                return None
            idx = torch.tensor(cols, dtype=torch.long, device=x.device)
            x = x.clone()
            if frozen is not None:
                v = torch.as_tensor(np.asarray(frozen[li]), dtype=x.dtype, device=x.device)
                if v.dim() != 2 or v.shape[1] != len(cols):
                    raise ValueError(f'layer {li}: frozen values {tuple(v.shape)} for {len(cols)} clamped units')
                # the arm's values cover positions 0..T+4; any appended continuation positions are left untouched
                x[:, :v.shape[0], idx] = v
            elif online_profile is not None:
                prof = torch.as_tensor(np.asarray(online_profile[li]), dtype=x.dtype, device=x.device)
                # the profile covers positions 0..T+4 only; appended continuation positions stay untouched (line 158)
                n = min(int(prof.shape[0]), x.shape[1])
                x[:, :n, idx] = x[:, :n, idx] - prof[:n].unsqueeze(0)
            elif online_factor is not None:
                x[:, :, idx] = float(online_factor) * x[:, :, idx]
            return (x,) + tuple(args[1:])
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


# Stage 2 plan per arm (design lines 179-190).
STAGE2_MODE = {'trace_resample_fact': 'trace', 'trace_resample_you': 'trace', 'trace_resample_zh_maker': 'trace',
               'mean_resample_fact': 'mean', 'sub_factual': 'sub', 'scale_pos2': 'scale', 'scale_neg1': 'scale',
               'zero_prompt_only': 'zero_prompt', 'zero_gen_only': 'zero_gen', 'zero': 'zero'}


def stage2_plan(arm: str) -> str:
    if arm not in STAGE2_MODE:
        raise ValueError(f'{arm} has no Stage 2 plan')
    return STAGE2_MODE[arm]


def stage2_source(mode: str, call: int, T: int, L_gen: int, width: int = 1) -> tuple[str, int]:
    """Which recorded source a Stage 2 forward call reads, for every mode (design lines 179-190).

    call 0 is prefill (x.shape[1] == T, predicts answer token 0); call s >= 1 is the decode step processing generated
    token s - 1 (x.shape[1] == 1). `width` is the forward's x.shape[1]; the prefill reads the donor prompt at the same
    position p as the recipient, so the returned ('prompt', 0) means "prompt row at the live position".
    Returns ('prompt', 0) / ('gen', j) / ('mu_prompt', 0) / ('mu_gen', 0) / ('live', -1) / ('zero', -1) / ('none', -1).
    """
    if mode == 'trace':
        return ('prompt', 0) if call == 0 else ('gen', (call - 1) % L_gen)
    if mode in ('mean', 'sub'):
        return ('mu_prompt', 0) if call == 0 else ('mu_gen', 0)
    if mode == 'scale':
        return ('live', -1)
    if mode == 'zero_prompt':
        return ('zero', -1) if call == 0 else ('none', -1)
    if mode == 'zero_gen':
        return ('none', -1) if call == 0 else ('zero', -1)
    if mode == 'zero':
        return ('zero', -1)
    raise ValueError(f'unknown Stage 2 mode {mode}')


def stage2_zero_mask(mode: str, call: int, T: int, width: int) -> np.ndarray:
    """Per-position boolean mask for `zero` (every call, every position), zero_prompt (call 0) and zero_gen (calls >= 1).

    F1 (GLM review): mode 'zero' is the primary intervention arm and every `random_k:zero` set runs it; its mask was
    all-False, so stage2_apply zeroed nothing and the arm behaved as `unchanged` with a different seed.
    """
    if mode == 'zero':
        return np.ones(width, dtype=bool)
    if mode == 'zero_prompt' and call == 0:
        return np.ones(width, dtype=bool)
    if mode == 'zero_gen' and call >= 1:
        return np.ones(width, dtype=bool)
    return np.zeros(width, dtype=bool)


def stage2_apply(x: torch.Tensor, cols: list[int], source: tuple[str, int], subtract: bool, tensors: dict) -> torch.Tensor:
    """Apply one Stage 2 source to the live activation x of shape (B, W, n_units). Pure; used by the hook and tests.

    subtract=True is `sub_factual`'s online x_live - mu (design lines 182, 187); False clamps to the source.
    tensors: {'prompt': (T, K) array, 'gen': (L_gen, K) array, 'mu_prompt': (T, K), 'mu_gen': (L_gen, K)}.
    """
    kind, idx = source
    if kind == 'none':
        return x
    idx_t = torch.tensor(cols, dtype=torch.long, device=x.device)
    x = x.clone()
    if kind == 'zero':
        mask = torch.as_tensor(np.asarray(tensors['zero_mask']), dtype=torch.bool, device=x.device)
        if mask.shape[0] != x.shape[1]:
            raise ValueError(f'zero mask covers {mask.shape[0]} positions, the forward has {x.shape[1]}')
        x[:, :, idx_t] = torch.where(mask[None, :, None], torch.zeros_like(x[:, :, idx_t]), x[:, :, idx_t])
    elif kind == 'live':
        x[:, :, idx_t] = (2.0 if tensors.get('factor', 2.0) > 0 else -1.0) * x[:, :, idx_t]
    else:
        src = torch.as_tensor(np.asarray(tensors[kind]), dtype=x.dtype, device=x.device)
        v = (src[:x.shape[1]] if kind in ('prompt', 'mu_prompt') else src[idx][None, :]).unsqueeze(0)
        x[:, :, idx_t] = x[:, :, idx_t] - v if subtract else v
    return x


def install_stage2_hooks(layers, units, mode: str, tensors: dict, T: int, L_gen: int, factor: float = 2.0):
    """Stage 2 prefill/decode hooks. Decode calls are counted per module; call s reads stage2_source(mode, s, T, L_gen).

    tensors: {'prompt': {layer: (T, K)}, 'gen': {layer: (L_gen, K)}, 'mu_prompt': {layer: (T, K)},
    'mu_gen': {layer: (L_gen, K)}}. 'sub' subtracts mu_prompt at call 0 and mu_gen at every decode call; 'mean' clamps
    to them; 'scale' multiplies the live value by `factor` (+2 or -1).
    """
    handles = []
    for li in sorted({int(l) for l, _ in units}):
        cols = _columns(units, li)
        state = {'s': 0}

        def hook(_module, args, li=li, cols=cols, state=state):
            x = args[0]
            if x.shape[1] == 1:
                state['s'] += 1
                call = state['s']
            elif x.shape[1] == T and state['s'] == 0:
                call = 0
            else:
                raise RuntimeError(f'unexpected forward width {x.shape[1]} at layer {li} (T={T})')
            source = stage2_source(mode, call, T, L_gen, x.shape[1])
            layer_tensors = {k: v[li] for k, v in tensors.items()}
            layer_tensors['factor'] = factor
            layer_tensors['zero_mask'] = stage2_zero_mask(mode, call, T, x.shape[1])
            return (stage2_apply(x, cols, source, mode == 'sub', layer_tensors),) + tuple(args[1:])
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


# ----------------------------------------------------------------------------------------------------------------
# logit readouts (all in float32, last position only; design lines 148-161)
# ----------------------------------------------------------------------------------------------------------------

def g_ali(logits_row: torch.Tensor) -> float:
    """logit(Google) - logsumexp(logit(Al), logit(T)) (design line 149)."""
    v = logits_row.float()
    return float(v[GOOGLE_ID]) - float(np.logaddexp(float(v[AL_ID]), float(v[T_ID])))


def g_al(logits_row: torch.Tensor) -> float:
    """Part 74's g_al = logit(Google) - logit(Al) (design line 161)."""
    v = logits_row.float()
    return float(v[GOOGLE_ID]) - float(v[AL_ID])


def eight_maker_mass(logits_row: torch.Tensor, bank: dict) -> float:
    """Probability mass on the 8 maker names' first tokens (design line 162)."""
    ids = [int(bank['maker_names'][name]['first_id']) for name in bank['maker_names']]
    return float(torch.softmax(logits_row.float()[ids], dim=0).sum())


def seq_logprob(logits: torch.Tensor, token_ids: list[int], start: int) -> float:
    """Sum of log-softmax probabilities of token_ids predicted from positions start, start+1, ... (design line 155)."""
    total = 0.0
    for k, tid in enumerate(token_ids):
        row = logits[start + k].float()
        total += float(row[int(tid)] - torch.logsumexp(row, dim=0))
    return total


def g_seq(logits: torch.Tensor, start: int, bank: dict) -> float:
    """log P("Google") - max(log P("Alibaba"), log P("Tongyi")) from the last prompt position (design line 156).

    `start` is T + 4, the position that predicts the first maker token. The continuation tokens are appended to the
    forward's input (hooks inactive there, design line 158), so `logits` must hold start + 2 rows.
    """
    if logits.shape[0] < start + len(continuation_ids(bank)):
        raise ValueError(f'g_seq needs logits through position {start + len(continuation_ids(bank)) - 1}, '
                         f'forward returned {logits.shape[0]} rows')
    names = bank['maker_names']
    lp_g = seq_logprob(logits, names['Google']['token_ids'], start)
    lp_a = seq_logprob(logits, names['Alibaba']['token_ids'], start)
    lp_t = seq_logprob(logits, names['Tongyi']['token_ids'], start)
    return lp_g - max(lp_a, lp_t)


def continuation_ids(bank: dict) -> list[int]:
    """The 1-2 appended continuation positions (design line 158): 'Al', 'T', 'ong'."""
    names = bank['maker_names']
    return [int(names['Alibaba']['token_ids'][0]), int(names['Tongyi']['token_ids'][0]),
            int(names['Tongyi']['token_ids'][1])]


def finite(x) -> bool:
    return x is not None and math.isfinite(float(x)) and abs(float(x)) < NON_FINITE


# ----------------------------------------------------------------------------------------------------------------
# Stage 1 driver
# ----------------------------------------------------------------------------------------------------------------

def down_weights(layers) -> np.ndarray:
    """The target layers' down_proj weights on CPU as float32 (d_model, n_units).

    AMBIGUITY 1 (design line 76 vs line 99): the hook site is down_proj's input, so `w_down` is the matrix whose
    columns act on the clamped unit activations; its column count equals that input width. Most literal reading.
    """
    return np.stack([layers[li].mlp.down_proj.weight.detach().float().cpu().numpy() for li, _ in TARGET])


def record_units(layers, units: list[tuple[int, int]], fn) -> np.ndarray:
    """Run fn() with recording pre-hooks on every named unit and return (T, K) float64 activations.

    One CPU copy per layer per forward (no .item() inside a hook, design line 294); K counts `units` and the value is
    that unit's activation on its OWN layer's down_proj input (design lines 112-113: one value per (position, unit)).
    fn must do exactly one forward.
    """
    by_layer = sorted({int(l) for l, _ in units})
    if not by_layer:  # nothing to record (the write-norm pre-pass): still run fn, return (0, 0)
        fn()
        return np.zeros((0, 0), dtype=np.float64)
    layer_cols = {li: sorted({u for l, u in units if int(l) == li}) for li in by_layer}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in by_layer:
        cols = layer_cols[li]

        def hook(_module, args, li=li, cols=cols, captured=captured):
            captured[li] = args[0][0][:, cols].float().cpu()  # whole tensor, once per forward
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        fn()
    finally:
        for hd in handles:
            hd.remove()
    missing = [li for li in by_layer if li not in captured]
    if missing:
        raise RuntimeError(f'no activation captured at layer(s) {missing}: the forward never reached them')
    T = captured[by_layer[0]].shape[0]
    arr = np.zeros((T, len(units)), dtype=np.float64)
    for k, (li, unit) in enumerate(units):
        arr[:, k] = captured[li][:, layer_cols[li].index(int(unit))].numpy()
    return arr


def _record_forward(model, layers, ids, units, device, abs_sums=None, counts=None) -> np.ndarray:
    """One recorded forward of `ids` (design line 111): record_units around a single no-grad forward.

    When abs_sums/counts are given, the same forward also accumulates |a| for every unit of those layers
    (accumulate_abs), so the activation-weighted write norm costs no extra forwards.
    """
    def fn():
        handles = []
        if abs_sums:
            for li in sorted(abs_sums):
                def hook(_module, args, li=li):
                    accumulate_abs(abs_sums, counts, li, args[0])
                handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
        finally:
            for hd in handles:
                hd.remove()
    return record_units(layers, units, fn)


def _forward(model, ids: list[int], device) -> None:
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)


def pool_norm_pass(tok, model, layers, bank: dict, device, pool_layers=None) -> tuple[dict, dict]:
    """Pre-pass over the 24 held-out EN_MAKER prompts (prompt + forced opening) that only accumulates |a| for every
    unit of `pool_layers` (design lines 98-102), so the write-norm pools exist BEFORE the recording pass that has to
    record them (design line 112). Forwards are deterministic, so this changes no number the recording pass produces.
    Returns (abs_sums, counts): per layer (n_units,) float64 sums and the position count.
    """
    pool = sorted({int(li) for li in (pool_layers if pool_layers is not None else [li for li, _ in TARGET])})
    abs_sums = {li: np.zeros(int(layers[li].mlp.down_proj.in_features), dtype=np.float64) for li in pool}
    counts = {li: 0 for li in pool}
    for qi in range(N_HELDOUT):
        ids, _T = stage1_prompt_ids(tok, bank, qi, 'en_maker')
        _record_forward(model, layers, ids, [], device, abs_sums, counts)
        nr.spill_guard(device, 20.0)
    print(f'stage1: write-norm pre-pass {N_HELDOUT}/{N_HELDOUT}', flush=True)
    return abs_sums, counts


def record_pass(tok, model, layers, bank: dict, units: list[tuple[int, int]], device) -> dict:
    """Unchanged recording pass over 125 prompts (5 banks x 25). One forward per prompt, one CPU copy per forward.

    `units` is what design line 112 lists: TARGET + every unit of the 4x50 write-norm pools + the two monitor units.
    Returns recorded: bank -> [per question (T+5, K)] in `units` order.
    """
    out: dict[str, list] = {name: [] for name in ('en_maker',) + TRACE_BANKS}
    for name in ('en_maker',) + TRACE_BANKS:
        for qi in range(25):
            ids, _T = stage1_prompt_ids(tok, bank, qi, name)
            out[name].append(_record_forward(model, layers, ids, units, device))
            nr.spill_guard(device, 20.0)
        print(f'stage1: recorded {name} 25/25', flush=True)
    return out


def stage1_pools(abs_sums: dict, counts: dict, down_weight: np.ndarray, n_units: int,
                 units=TARGET, pool: int = NORM_POOL) -> tuple[dict, dict]:
    """Per-layer 50-unit write-norm pools from the unchanged EN_MAKER pass (design lines 98-102).

    `abs_sums`/`counts` are the float64 accumulation of |a| for EVERY unit of the four target layers over the 24
    held-out EN_MAKER prompts (prompt + forced opening), from pool_norm_pass. The match is the design's
    activation-weighted m_{L,u} = mean(|a|) * ||w_down[:, u]||_2, so the returned pools are the 50 units nearest the
    target in delivered write norm, target excluded, ties by unit index.
    down_weight is down_weights(layers): one (d_model, n_units) matrix per target layer, in `units` order.
    Returns (pools, maker_norm) so the caller can log the per-layer m summary.
    """
    maker_norm = unit_write_norms(abs_sums, counts, down_weight, units)
    return write_norm_pools(maker_norm, n_units, target=units, pool=pool), maker_norm


def pool_summary(maker_norm: dict, pools: dict, units=TARGET) -> dict:
    """Per-layer summary of the activation-weighted m vector: the target's m and the pool's (and layer's) min/max."""
    out = {}
    for layer, unit in units:
        row = np.asarray(maker_norm[int(layer)], dtype=np.float64)
        chosen = pools[int(layer)]
        out[str(int(layer))] = {
            'target_unit': int(unit),
            'target_m': float(row[int(unit)]),
            'pool_size': len(chosen),
            'pool_min_m': float(row[chosen].min()) if chosen else None,
            'pool_max_m': float(row[chosen].max()) if chosen else None,
            'layer_min_m': float(row.min()),
            'layer_max_m': float(row.max()),
        }
    return out


def stage1_profiles(recorded: dict, bank: dict) -> dict:
    """Bank profiles for d_fact and d_you, built slot-wise from donors of different lengths (design lines 113-121).

    Each donor is 1-to-1 length matched to its own EN_MAKER question, and the 24 held-out questions are 16-30 tokens
    long, so the 24 donors do not share one length. build_profile means each slot over the donors at their own
    header/body/suffix/opening positions; the profile is then mapped onto a recipient relative to that recipient's own
    length (profile_index), which is v3's rule.

    AMBIGUITY 6 (design line 121): one profile per bank, built from the 24 held-out donors, is applied to every
    recipient including question 24 ("Who created you?", a positive control outside the 24).
    """
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    out = {}
    for name in ('d_fact', 'd_you'):
        out[name] = build_profile([recorded[name][q] for q in range(N_HELDOUT)], h, s)
    return out


def stage1_input_ids(ids: list[int], bank: dict) -> list[int]:
    """The forward's input: prompt + 5-token opening + the 1-2 continuation tokens (design line 158).

    The hooks are active on positions 0..T+4 and inactive on the appended continuation positions, so the continuation
    tokens are fed in with no hook installed; `g_seq` scores the maker names from position T+4 onward.
    """
    return list(ids) + continuation_ids(bank)


def run_stage1_forward(model, layers, units, ids, frozen, online_profile, online_factor, device,
                       mon: dict | None = None, mon_position: int = -1) -> torch.Tensor:
    """One intervened forward; monitor activations land in `mon` when given (read from the intervened forward).

    F4: `ids` is prompt + opening + continuation tokens, so the last position is T+7; the caller passes
    mon_position = T+4 so every arm's monitor is read at the same position as the unchanged baseline.
    """
    handles = install_stage1_hooks(layers, units, frozen, online_profile, online_factor)
    if mon is not None:
        handles += monitor_handles(layers, mon, position=mon_position)
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    return out.logits[0]


MONITORS = ((24, 3949), (27, 3325))


def monitor_handles(layers, mon: dict, position: int = -1) -> list:
    """Hooks that record L24 N3949 and L27 N3325 at `position` (design line 77: the last prompt position T+4)."""
    handles = []
    for li, unit in MONITORS:
        def hook(_module, args, li=li, unit=unit):
            mon[li] = float(args[0][0, position, unit].float().cpu())
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def monitor_values(model, layers, ids, device, position: int = -1) -> dict:
    """Activation of L24 N3949 and L27 N3325 at `position` on an unchanged forward (design line 77)."""
    mon: dict[int, float] = {}
    handles = monitor_handles(layers, mon, position)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    return {'l24_n3949': mon.get(24), 'l27_n3325': mon.get(27)}


def stage1_arm_names() -> list[str]:
    """15 main arms, then 16 per-unit arms, then 100 matched random control arms (design lines 129-147)."""
    names = list(MAIN_ARMS)
    names += [f'unit:{layer}:{unit}:{arm}' for layer, unit in TARGET for arm in PER_UNIT_ARMS]
    names += [f'{set_name}:{arm}' for set_name in (f'random_{k}' for k in range(N_RANDOM_SETS)) for arm in RANDOM_ARMS]
    return names


def parse_arm(arm: str) -> tuple[list[tuple[int, int]], str]:
    """Arm name -> (units it clamps, base arm whose source it uses)."""
    if arm in STAGE1_ARM_SPEC:
        return list(TARGET), arm
    parts = arm.split(':')
    if parts[0] == 'unit':
        return [(int(parts[1]), int(parts[2]))], parts[3]
    return None, parts[1]  # 'random_k:base' is resolved by the caller, which holds the drawn sets


def stage1_spec(arm: str):
    if arm in STAGE1_ARM_SPEC:
        return STAGE1_ARM_SPEC[arm]
    return STAGE1_ARM_SPEC[parse_arm(arm)[1]]


def frozen_by_layer(vals: np.ndarray, arm_units: list[tuple[int, int]]) -> dict:
    """{layer: (T+5, K_layer)} from stage1_values output, K_layer in _columns(units, layer) order (install_stage1_hooks)."""
    return {li: np.stack([vals[:, k] for k, (l, _u) in enumerate(arm_units) if l == li], axis=1)
            for li in sorted({l for l, _ in arm_units})}


def stage1_record_units(pools: dict) -> list[tuple[int, int]]:
    """What design line 112 records: TARGET, every unit of the 4x50 pools, then the two monitor units."""
    units = list(TARGET)
    for layer in sorted(pools):
        units += [(int(layer), int(u)) for u in pools[layer]]
    return units + list(MONITORS)


def stage1(args, loaded=None) -> dict:
    with output_directory_lock(args.out, 'stage1'):
        return _stage1_locked(args, loaded)


def _stage1_locked(args, loaded=None) -> dict:
    bank = load_prompts()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device
    tok, model = loaded if loaded is not None else nr.load_model(MODEL, REVISION, device)
    load = dict(nr.LAST_LOAD)
    _, layers, _ = nr.find_text_layers(model)
    n_units = int(layers[0].mlp.down_proj.in_features)
    down_weight = down_weights(layers)
    t0 = time.time()
    abs_sums, counts = pool_norm_pass(tok, model, layers, bank, device)
    pools, maker_norm = stage1_pools(abs_sums, counts, down_weight, n_units)
    sets = random_sets(pools)
    units = stage1_record_units(pools)
    target_cols = [units.index(u) for u in TARGET]
    recorded = record_pass(tok, model, layers, bank, units, device)
    # profiles are only ever read for the clamped units, so they are built from the TARGET columns of the recording
    # (the main arms clamp TARGET; the random arms read their own units' per-question traces directly, F3)
    profiles = stage1_profiles({name: [a[:, target_cols] for a in recorded[name]] for name in recorded}, bank)
    cols_of = {u: units.index(u) for u in units}
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    print(f'stage1: recording pass done in {time.time() - t0:.1f}s', flush=True)

    keys = ('g_ali', 'g_seq', 'g_al', 'mass', 'l27', 'l24', 'l27_unchanged', 'l24_unchanged', 'pert')
    results = {arm: {k: [] for k in keys} for arm in stage1_arm_names()}
    nonfinite = 0
    for qi in range(25):
        ids, T_q = stage1_prompt_ids(tok, bank, qi)
        # the recording pass stores every recorded unit (TARGET + pools + monitors, design line 112); the main arms
        # clamp TARGET, so their rec is the TARGET columns in TARGET order. F3: a random set's rec is ITS OWN units'
        # columns (own donor trace, own a_clean, own a_clean - r), built per arm below.
        rec = {name: np.asarray(recorded[name][qi], dtype=np.float64)[:, target_cols] for name in recorded}
        # (rows are positions 0..T+4; columns are the recorded units, and target_cols picks TARGET in TARGET order)
        rec['clean'] = rec['en_maker']  # a_clean(q, p): the unchanged recording on EN_MAKER[q]
        mon = monitor_values(model, layers, ids, device, position=T_q + OPENING_TOKENS - 1)
        for arm in stage1_arm_names():
            spec = stage1_spec(arm)
            arm_units = stage1_arm_units(arm, sets)
            arm_rec = rec
            if arm_units != list(TARGET):
                # F3: each clamped unit reads its own recorded column on every bank (the 4x50 pool units are recorded)
                arm_cols = [cols_of[u] for u in arm_units]
                arm_rec = {name: np.asarray(recorded[name][qi], dtype=np.float64)[:, arm_cols] for name in recorded}
                arm_rec['clean'] = arm_rec['en_maker']
            frozen = online_profile = None
            vals = None
            if spec is not None and spec[0] == 'frozen':
                if arm_units == list(TARGET):
                    vals = stage1_values(arm, qi, T_q, h, s, rec, profiles)
                else:
                    vals = stage1_values(arm, qi, T_q, h, s, arm_rec,
                                         stage1_profiles({name: [a[:, [cols_of[u] for u in arm_units]]
                                                                for a in recorded[name]] for name in recorded}, bank))
                frozen = frozen_by_layer(vals, arm_units)
            elif spec is not None:
                # sub_factual_online: x_live - r, r from the base arm's profile bank (SUB_ARM_PROFILE)
                mapped = profile_at(profiles[SUB_ARM_PROFILE[parse_arm(arm)[1]]], T_q, h, s)
                online_profile = {li: np.stack([mapped[:, k] for k, (l, _u) in enumerate(arm_units) if l == li], axis=1)
                                  for li in sorted({l for l, _ in arm_units})}
            arm_mon: dict[int, float] = {}
            fwd_ids = stage1_input_ids(ids, bank)  # prompt + opening + continuation; hooks act on 0..T+4 only
            if spec is None:  # `unchanged` is the bare forward whose monitors `mon` already holds
                logits = run_stage1_forward(model, layers, arm_units, fwd_ids, None, None, None, device)
                arm_mon = {27: mon['l27_n3325'], 24: mon['l24_n3949']}
            else:
                logits = run_stage1_forward(model, layers, arm_units, fwd_ids, frozen, online_profile, None, device,
                                            arm_mon, mon_position=T_q + OPENING_TOKENS - 1)
            last = T_q + OPENING_TOKENS - 1
            r = results[arm]
            r['g_ali'].append(g_ali(logits[last]))
            r['g_al'].append(g_al(logits[last]))
            r['mass'].append(eight_maker_mass(logits[last], bank))
            # F5: g_seq scores from T+4, the position that predicts the first maker token (design line 158)
            r['g_seq'].append(g_seq(logits, last, bank))
            # AMBIGUITY 3 (design line 77 vs 162): the monitor units are never intervened, but they are read from the
            # arm's own forward (downstream of layers 6-29 the arm's clamp changes them) as well as from the unchanged
            # forward, so 212.7 can use either; both are written out. F4: both are read at T+4 (the last opening
            # position), so `l27_shift` is the arm's effect and not a positional offset.
            r['l27'].append(arm_mon.get(27))
            r['l24'].append(arm_mon.get(24))
            r.setdefault('l27_unchanged', []).append(mon['l27_n3325'])
            r.setdefault('l24_unchanged', []).append(mon['l24_n3949'])
            # AMBIGUITY 4 (design line 105): the realized norm is the full 4-unit sum for every arm, including the
            # per-unit and random-control arms, so the numbers stay comparable across arm families. F3: it is computed
            # from the arm's OWN clamped units and their OWN clean activations and down_proj columns, so a random set
            # that changes anything records a nonzero norm.
            if spec is None:
                pert = 0.0
            else:
                clean_arm = arm_rec['en_maker']
                if online_profile is not None:
                    # sub_factual_online: x_live - r, so the write is -r at every position of every clamped unit
                    mapped_arm = profile_at(stage1_profiles(
                        {name: [a[:, [cols_of[u] for u in arm_units]] for a in recorded[name]]
                         for name in recorded}, bank)[SUB_ARM_PROFILE[parse_arm(arm)[1]]], T_q, h, s)
                    pert = realized_perturbation(clean_arm - mapped_arm, clean_arm,
                                                 target_down_columns(down_weight, arm_units))
                else:
                    pert = realized_perturbation(vals, clean_arm, target_down_columns(down_weight, arm_units))
            r['pert'].append(pert)
            if not all(finite(x) for x in (r['g_ali'][-1], r['g_seq'][-1], r['mass'][-1])):
                nonfinite += 1
        print(f'stage1: question {qi + 1}/25', flush=True)
    return finalize_stage1(args, bank, results, pools, sets, maker_norm, nonfinite, load, time.time() - t0, n_units)


def target_down_columns(down_weight: np.ndarray, units=TARGET) -> np.ndarray:
    """(d_model, K) matrix whose column k is w_down[:, unit] of that unit's OWN layer (design line 105).

    down_weight is down_weights(layers): one (d_model, n_units) matrix per TARGET layer, in TARGET layer order.
    `units` defaults to TARGET; per-unit and random-control arms pass their own units (F3), so each column is selected
    from that unit's own layer, including per-unit arms whose layer is not the first TARGET layer.
    """
    layer_index = {int(layer): i for i, (layer, _unit) in enumerate(TARGET)}
    try:
        return np.stack([np.asarray(down_weight[layer_index[int(layer)]], dtype=np.float64)[:, int(unit)]
                         for layer, unit in units], axis=1)
    except KeyError as exc:
        raise ValueError(f'no down_proj weights recorded for layer {exc.args[0]}') from exc


def stage1_arm_units(arm: str, sets: dict) -> list[tuple[int, int]]:
    """The units an arm clamps: the 4 targets, one target (per-unit arms), or a drawn random set."""
    if arm in STAGE1_ARM_SPEC:
        return list(TARGET)
    parts = arm.split(':')
    if parts[0] == 'unit':
        return [(int(parts[1]), int(parts[2]))]
    return list(sets[parts[0]])


def finalize_stage1(args, bank, results, pools, sets, maker_norm, nonfinite, load, wall, n_units) -> dict:
    """stage1.json: per-arm per-question readouts, deltas from unchanged, gate inputs, pools and random sets."""
    base = results['unchanged']
    arms = {}
    for arm, r in results.items():
        d_ali = [a - b for a, b in zip(r['g_ali'], base['g_ali'])]
        d_seq = [a - b for a, b in zip(r['g_seq'], base['g_seq'])]
        l27_shift = [a - b for a, b in zip(r['l27'], base['l27'])]
        # AMBIGUITY 5 (design line 77, "recorded without intervention"): the monitor units are never clamped, but they
        # are read from the arm's own intervened forward. `l27_shift` is that arm's L27 N3325 minus unchanged's; the
        # unchanged-forward copies are kept beside it so the scorer can show either reading.
        arms[arm] = {
            'g_ali': r['g_ali'], 'g_seq': r['g_seq'], 'g_al': r['g_al'], 'mass': r['mass'],
            'delta_g_ali': d_ali, 'delta_g_seq': d_seq,
            'mean_g_ali': float(np.mean(r['g_ali'])), 'mean_delta_g_ali': float(np.mean(d_ali)),
            'mean_g_seq': float(np.mean(r['g_seq'])), 'mean_delta_g_seq': float(np.mean(d_seq)),
            'mean_mass': float(np.mean(r['mass'])),
            'l27_n3325': r['l27'], 'l24_n3949': r['l24'], 'l27_shift': l27_shift,
            'mean_l27_shift': float(np.mean(l27_shift)),
            'l27_n3325_unchanged_forward': r['l27_unchanged'], 'l24_n3949_unchanged_forward': r['l24_unchanged'],
            'realized_perturbation_norm': float(np.mean(r['pert'])),
            'perturbation_norm_per_question': r['pert'],
        }
    obj = {
        'part': 212, 'stage': 1, 'model': MODEL, 'revision': REVISION, 'load': load,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prompts_sha256': PROMPTS_SHA256, 'device': args.device, 'wall_time_seconds': wall,
        'n_units': n_units, 'target': [list(u) for u in TARGET], 'non_finite_readouts': nonfinite,
        'forward_batch_histogram': stage1_forward_batch_histogram(),
        'questions': [bank['items'][q]['en_maker'] for q in range(25)],
        'arms': arms,
        'gates': {
            'gate1': {
                'self_patch_minus_unchanged_mean_g_ali': abs(arms['self_patch']['mean_g_ali']
                                                             - arms['unchanged']['mean_g_ali']),
                'non_finite': nonfinite,
                'mass_unchanged': arms['unchanged']['mean_mass'], 'mass_zero': arms['zero']['mean_mass'],
            },
            'gate2': {'zero_mean_delta_g_ali': arms['zero']['mean_delta_g_ali']},
        },
        'write_norm_pools': {str(k): v for k, v in pools.items()},
        'write_norm_summary': pool_summary(maker_norm, pools),
        'random_sets': {k: [list(u) for u in v] for k, v in sets.items()},
    }
    (Path(args.out) / 'stage1.json').write_text(json.dumps(obj, ensure_ascii=False) + '\n', encoding='utf-8')
    return obj


# ----------------------------------------------------------------------------------------------------------------
# Stage 2 driver
# ----------------------------------------------------------------------------------------------------------------

def stage2_rows(pilot: bool = False) -> list[dict]:
    """Every planned Stage 2 row (design lines 191-200), or the pilot subset (design lines 270-272).

    A full arm is 240 rows (design line 195): 24 held-out questions x 8 samples, "Who created you?" x 24, and the 6
    fluency guards x 4 = 24. F2 (GLM review): the guards live at question_index 25..30 and the old loop stopped at 24,
    so no guard row was ever planned and Gate 3's unchanged guard-accuracy clause could never pass.
    """
    rows = []
    if pilot:
        for arm in PILOT_ARMS:
            for qi in PILOT_QUESTIONS:
                for sample in range(PILOT_SAMPLES):
                    rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
            for sample in range(PILOT_POSITIVE_SAMPLES):
                rows.append({'arm': arm, 'question_index': POSITIVE_INDEX, 'sample': sample, 'set': None})
        return rows
    for arm in STAGE2_FULL_ARMS:
        for qi in range(25):
            n = 8 if qi < N_HELDOUT else (24 if qi == POSITIVE_INDEX else 4)
            for sample in range(n):
                rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
        for gi in guard_indices():
            for sample in range(GUARD_SAMPLES):
                rows.append({'arm': arm, 'question_index': gi, 'sample': sample, 'set': None})
    for arm in STAGE2_TRACE_ARMS:
        for qi in range(25):
            for sample in range(8 if qi < N_HELDOUT else 24):
                rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
    for arm in STAGE2_DIAG_ARMS:
        for qi in range(N_HELDOUT):
            for sample in range(8):
                rows.append({'arm': arm, 'question_index': qi, 'sample': sample, 'set': None})
    for k in range(N_RANDOM_SETS):
        for arm in STAGE2_RANDOM_ARMS:
            for qi in range(N_HELDOUT):
                rows.append({'arm': arm, 'question_index': qi, 'sample': 0, 'set': f'random_{k}'})
    return rows


def stage2_resume_key(arm: str, qi: int, sample: int, set_name, pilot: bool) -> tuple:
    """The identity of one Stage 2 row for resume: arm, question, sample, set AND the pilot flag.

    F6 (GLM review): without the pilot flag every pilot key (4 samples of unchanged/zero/sub_factual, 4 of
    trace_resample_fact and 24 positive-control rows) is also a full-plan key, so a full run into the pilot's --out
    silently skipped all 160 pilot rows and mixed batch-8 rows into the 192-row arms.
    """
    return (arm, int(qi), int(sample), set_name, bool(pilot))


def stage2_todo(planned: list[dict], done_rows: list[dict], pilot: bool) -> list[dict]:
    """The planned rows not already in `done_rows` under the same pilot flag (rows of the other shape never match)."""
    existing = {stage2_resume_key(r['arm'], r['question_index'], r['sample'], r.get('set'), bool(r.get('pilot')))
                for r in done_rows if 'arm' in r and 'question_index' in r and 'sample' in r}
    return [r for r in planned
            if stage2_resume_key(r['arm'], r['question_index'], r['sample'], r['set'], pilot) not in existing]


def stage2_batches(rows: list[dict], batch_size: int) -> list[list[dict]]:
    """Group Stage 2 rows exactly as the runner generates them and split by the requested batch size."""
    if int(batch_size) < 1:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row['arm'], row['question_index'], row['set']), []).append(row)
    return [group[i:i + batch_size] for group in groups.values() for i in range(0, len(group), batch_size)]


def question_text(bank: dict, qi: int) -> str:
    """The prompt for a Stage 2 question index: 0..24 are the bank items, 25..30 the 6 fluency guards (design line 195)."""
    return bank['items'][qi]['en_maker'] if qi < 25 else bank['guards'][qi - 25]['question']


def guard_indices() -> list[int]:
    """question_index of the 6 fluency guards (25..30), read from the bank rather than assumed."""
    return [int(g['index']) for g in load_prompts()['guards']]


def is_guard(qi: int) -> bool:
    return int(qi) >= 25


def stage2_units(arm: str, set_name, sets: dict) -> list[tuple[int, int]]:
    return [] if arm == 'unchanged' else list(sets[set_name] if set_name else TARGET)


def donor_bank_for(arm: str) -> str:
    return {'trace_resample_fact': 'd_fact', 'trace_resample_you': 'd_you',
            'trace_resample_zh_maker': 'd_zh_maker'}.get(arm, 'd_fact')


def donor_L_gen(n_non_eos: int, max_new_tokens: int = MAX_NEW_TOKENS) -> int:
    """L_gen = min(n_non-EOS, 47): the 48th generated token is never fed back and has no activation (line 174)."""
    return int(min(max(n_non_eos, 0), max_new_tokens - 1))


def donor_means(donors: dict, h: int, s: int) -> dict:
    """mu_fact_gen: pooled non-EOS generated-token mean over the 24 held-out D_fact donors (design line 170).
    mu_prompt: the D_fact prefill mean over the 24 held-out donors, used at call 0 (AMBIGUITY 2, design line 181).

    The 24 donors have different prompt lengths (16-30 tokens), so mu_prompt is built slot-wise like a bank profile
    (build_profile with_opening=False: header rows, ONE pooled body row, suffix rows) and stored in the bank order
    `profile_index` reads. It is mapped onto each recipient's own length at call 0 (stage2_tensors), exactly as
    Stage 1's profiles are (design lines 117-121).
    """
    fact = donors['d_fact'][:N_HELDOUT]
    mu_gen = np.concatenate([d['gen'] for d in fact], axis=0).mean(axis=0)
    mu_prompt = build_profile([d['prompt'] for d in fact], h, s, with_opening=False)
    return {'mu_gen': mu_gen, 'mu_prompt': mu_prompt}


def stage2_record_units(pools: dict) -> list[tuple[int, int]]:
    """What design line 168 records: TARGET, then every unit of the 4x50 pools (the 20 sets draw from them)."""
    units = list(TARGET)
    for layer in sorted(pools):
        units += [(int(layer), int(u)) for u in pools[layer]]
    return units


def record_donors(tok, model, layers, bank: dict, units, device) -> dict:
    """Greedy donor generation for D_fact, D_you and D_zh_maker (75 runs), recording prefill 0..T-1 and non-EOS
    generated positions 0..L_gen-1 (design lines 166-176). Generated position j is recorded by decode call j + 1.

    Returns donors[bank][qi] = {'prompt': (T, K), 'gen': (L_gen, K), 'n_non_eos', 'L_gen'} in `units` order.
    """
    by_layer = sorted({int(l) for l, _ in units})
    layer_cols = {li: sorted({u for l, u in units if int(l) == li}) for li in by_layer}
    donors: dict[str, list] = {}
    for name in DONOR_BANKS:
        donors[name] = []
        for qi in range(25):
            ids = p66_ids(tok, nr.build_prompt(tok, bank['items'][qi][name], False))
            T = len(ids)
            cap: dict[int, list] = {li: [] for li in by_layer}
            handles = []
            for li in by_layer:
                cols = layer_cols[li]

                def hook(_module, args, li=li, cols=cols, cap=cap):
                    x = args[0]
                    cap[li].append((x[0][-1, cols] if x.shape[1] == 1 else x[0][:, cols]).float().cpu())
                handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
            try:
                with torch.no_grad():
                    res = model.generate(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                                         max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            finally:
                for hd in handles:
                    hd.remove()
            gen = [int(t) for t in res[0, T:].tolist()]
            eos = tok.eos_token_id
            n_non_eos = gen.index(int(eos)) if (eos is not None and int(eos) in gen) else len(gen)
            L_gen = donor_L_gen(n_non_eos)
            prompt = np.zeros((T, len(units)), dtype=np.float64)
            genarr = np.zeros((L_gen, len(units)), dtype=np.float64)
            for k, (li, unit) in enumerate(units):
                col = layer_cols[li].index(int(unit))
                prompt[:, k] = cap[li][0][:, col].numpy()
                for j in range(L_gen):
                    # generated position j is recorded by decode call j + 1; a call that is missing is a bug, not a
                    # silent zero (the 48th token has no call, and L_gen <= 47 never asks for it)
                    if 1 + j >= len(cap[li]):
                        raise RuntimeError(f'{name}[{qi}] recorded {len(cap[li]) - 1} decode calls, L_gen={L_gen}')
                    genarr[j, k] = cap[li][1 + j][col].numpy()
            donors[name].append({'prompt': prompt, 'gen': genarr, 'n_non_eos': n_non_eos, 'L_gen': L_gen})
            print(f'donors: {name} {qi + 1}/25 n_non_eos={n_non_eos} L_gen={L_gen}', flush=True)
    return donors


def short_donors(donors: dict, minimum: int = 4) -> list[dict]:
    """Gate 3b: any donor with L_gen < 4 (design lines 172-174)."""
    return [{'bank': b, 'index': i, 'L_gen': d['L_gen'], 'n_non_eos': d['n_non_eos']}
            for b, ds in donors.items() for i, d in enumerate(ds) if d['L_gen'] < minimum]


def stage2_tensors(donors: dict, means: dict, arm: str, qi: int, units, record_units=None, h: int = 3,
                   s: int = 9, T_guard: int | None = None) -> dict:
    """Per-layer tensors for one (arm, question): prompt/gen/mu_prompt/mu_gen, each (W, K_layer).

    The donor trace is the arm's own bank (D_fact for the mean/sub/scale arms too); mu_gen is the D_fact pooled
    generated mean, mu_prompt the D_fact slot-wise prefill profile mapped onto this recipient's own prompt length T
    (profile_at, design lines 117-121). donors and means are stored in `record_units` order (TARGET + pools,
    stage2_record_units); the returned tensors keep only this arm's `units`, in `units` order, so the hook's
    `_columns` order matches.
    """
    record_units = list(record_units if record_units is not None else TARGET)
    cols = [record_units.index(u) for u in units]
    layers = sorted({l for l, _ in units})
    if is_guard(qi):
        # F2: the 6 fluency guards have no donor of their own (donors hold the 25 bank items only). The guard rows are
        # generated by the mean/sub arms, which read only the bank-level mu_prompt (mapped onto the guard's own length)
        # and mu_gen; a trace arm on a guard would need a donor row that does not exist, so refuse instead of guessing.
        if stage2_plan(arm) == 'trace':
            raise ValueError(f'{arm} needs a donor row for guard question {qi}, but donors only cover the 25 bank items')
        # T_guard is the guard prompt's own token length (the caller has the tokenizer); no donor row is looked up.
        if T_guard is None:
            raise ValueError(f'guard question {qi} needs its own prompt length T_guard')
        T = int(T_guard)
        mu_prompt = profile_at(means['mu_prompt'], T, h, s)
        out = {'mu_prompt': {li: np.stack([mu_prompt[:, c] for c, (l, _u) in zip(cols, units) if l == li], axis=1)
                             for li in layers}}
        out['mu_gen'] = {li: np.stack([means['mu_gen'][c] for c, (l, _u) in zip(cols, units) if l == li]).reshape(1, -1)
                         for li in layers}
        return out
    d = donors[donor_bank_for(arm)][qi]
    T = int(d['prompt'].shape[0])
    mu_prompt = profile_at(means['mu_prompt'], T, h, s)
    out = {}
    for name, src in (('prompt', d['prompt']), ('gen', d['gen']), ('mu_prompt', mu_prompt)):
        out[name] = {li: np.stack([src[:, c] for c, (l, _u) in zip(cols, units) if l == li], axis=1)
                     for li in layers}
    # mu_gen is ONE pooled generated-position vector (K,), the same at every decode call (design lines 170, 187)
    out['mu_gen'] = {li: np.stack([means['mu_gen'][c] for c, (l, _u) in zip(cols, units) if l == li]).reshape(1, -1)
                     for li in sorted({l for l, _ in units})}
    return out


def stage2(args, pilot: bool = False, loaded=None) -> dict:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with output_directory_lock(outdir, 'pilot' if pilot else 'stage2'):
        return _stage2_locked(args, pilot, loaded)


def _stage2_locked(args, pilot: bool = False, loaded=None) -> dict:
    bank = load_prompts()
    outdir = Path(args.out)
    answer_path = outdir / 'stage2.jsonl'
    planned = stage2_rows(pilot)
    try:
        batch_size = stage2_requested_batch_size(args, pilot)
        done_rows = load_stage2_jsonl(answer_path)
        partial_calls = stage2_validate_resume(planned, done_rows, batch_size, pilot)
        if partial_calls:
            stage2_drop_partial_calls(answer_path, done_rows, partial_calls, pilot)
            done_rows = load_stage2_jsonl(answer_path)
            stage2_validate_resume(planned, done_rows, batch_size, pilot)
    except ValueError as exc:
        raise SystemExit(f'Stage 2 resume refused: {exc}') from exc
    todo = stage2_todo(planned, done_rows, pilot)
    try:
        batches = stage2_batches(todo, batch_size)
    except ValueError as exc:
        raise SystemExit(f'Stage 2 resume refused: {exc}') from exc
    device = args.device
    tok, model = loaded if loaded is not None else nr.load_model(MODEL, REVISION, device)
    load = dict(nr.LAST_LOAD)
    _, layers, _ = nr.find_text_layers(model)
    n_units = int(layers[0].mlp.down_proj.in_features)
    down_weight = down_weights(layers)
    # The pools come from the unchanged pass over the 24 held-out EN_MAKER prompts (design lines 98-102); Stage 2
    # recomputes them from its own recording pass so a Stage 2 directory is self-contained.
    abs_sums, counts = pool_norm_pass(tok, model, layers, bank, device)
    pools, maker_norm = stage1_pools(abs_sums, counts, down_weight, n_units)
    sets = random_sets(pools)
    norm_summary = pool_summary(maker_norm, pools)
    # Stage 2 clamps TARGET and the drawn random sets, so its donor recording covers TARGET + every pool unit
    # (design line 168: "TARGET and all 20 matched-control units")
    record_units = stage2_record_units(pools)
    donors = record_donors(tok, model, layers, bank, record_units, device)
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    means = donor_means(donors, h, s)
    short = short_donors(donors)
    if short:
        # Gate 3b: stop before generation and report the short donor (design lines 172-174).
        saved_hist = stage2_batch_histogram([r for r in done_rows if bool(r.get('pilot', False)) == bool(pilot)])
        obj = stage2_run_json(args, load, donors, means, sets, planned, 0, 0.0, short, pilot, norm_summary,
                              saved_hist)
        (outdir / 'run.json').write_text(json.dumps(obj, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Gate 3b FAILED: short donors {short}; Stage 2 stopped before generation', flush=True)
        return obj
    start = time.time()
    done = 0
    with answer_path.open('a', encoding='utf-8') as sink:
        for batch in batches:
            arm, qi, set_name = batch[0]['arm'], batch[0]['question_index'], batch[0]['set']
            question = question_text(bank, qi)
            ids = p66_ids(tok, nr.build_prompt(tok, question, False))
            T = len(ids)
            units = stage2_units(arm, set_name, sets)
            seeds = [p66.row_seed(f'{arm}:{set_name or "target"}', qi, r['sample']) for r in batch]
            torch.manual_seed(seeds[0])
            if device == 'xpu' and hasattr(torch, 'xpu'):
                torch.xpu.manual_seed_all(seeds[0])
            if arm == 'unchanged':
                handles = []
            else:
                tensors = stage2_tensors(donors, means, arm, qi, units, record_units, h, s,
                                         T_guard=T if is_guard(qi) else None)
                # A guard has no donor row. Trace arms are rejected by stage2_tensors; all remaining modes are
                # independent of L_gen, so use 1 and avoid even consulting a donor for this guard.
                L_gen = 1 if is_guard(qi) else donors[donor_bank_for(arm)][qi]['L_gen']
                factor = {'scale_pos2': 2.0, 'scale_neg1': -1.0}.get(arm, 2.0)
                handles = install_stage2_hooks(layers, units, stage2_plan(arm), tensors, T, L_gen, factor)
            try:
                with torch.no_grad():
                    result = model.generate(input_ids=torch.tensor([ids] * len(batch), dtype=torch.long, device=device),
                                            max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7,
                                            top_p=0.95, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                            use_cache=True)
            finally:
                for hd in handles:
                    hd.remove()
            lines = []
            for i, row in enumerate(batch):
                raw = [int(t) for t in result[i, T:].tolist()]
                if tok.eos_token_id is not None and int(tok.eos_token_id) in raw:
                    raw = raw[:raw.index(int(tok.eos_token_id)) + 1]
                text = tok.decode(torch.tensor(raw, dtype=torch.long), skip_special_tokens=True).strip()
                record = {'arm': arm, 'set': set_name, 'set_units': [{'layer': l, 'unit': u} for l, u in units],
                          'question_index': qi, 'sample': row['sample'], 'question': question,
                          'question_kind': bank['items'][qi]['kind'] if qi < 25 else 'guard',
                          'guard_question': bank['guards'][qi - 25]['question'] if is_guard(qi) else None,
                          'seed': seeds[i], 'prompt': nr.build_prompt(tok, question, False),
                          'prompt_token_ids': ids, 'answer': text, 'token_ids': raw,
                          'stop_reason': p66._stop_reason(raw, tok), 'pilot': pilot,
                          **stage2_batch_fields(len(batch), batch_size)}
                lines.append(json.dumps(record, ensure_ascii=False) + '\n')
            sink.write(''.join(lines))
            sink.flush()
            os.fsync(sink.fileno())
            done += len(batch)
            print(f'stage2: {done}/{len(todo)} elapsed={time.time() - start:.1f}s', flush=True)
            nr.spill_guard(device, 20.0)
    saved_rows = load_stage2_jsonl(answer_path)
    saved_hist = stage2_batch_histogram([r for r in saved_rows if bool(r.get('pilot', False)) == bool(pilot)])
    obj = stage2_run_json(args, load, donors, means, sets, planned, done, time.time() - start, short, pilot,
                          norm_summary, saved_hist)
    (outdir / 'run.json').write_text(json.dumps(obj, ensure_ascii=False) + '\n', encoding='utf-8')
    return obj


def stage2_run_json(args, load, donors, means, sets, planned, done, wall, short, pilot, write_norm_summary,
                    generation_batch_histogram=None) -> dict:
    """run.json: what the scorer needs to check Gate 3 from saved files alone."""
    return {
        'part': 212, 'stage': 'pilot' if pilot else 2, 'model': MODEL, 'revision': REVISION,
        'load': load, 'attn_implementation': load.get('attn_implementation'),
        'attn_implementation_requested': load.get('attn_implementation_requested'),
        'dtype': load.get('dtype'), 'device': args.device,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prompts_sha256': PROMPTS_SHA256,
        'target': [list(u) for u in TARGET], 'batch_size': args.batch or (PILOT_BATCH if pilot else STAGE2_BATCH),
        'generation_batch_histogram': generation_batch_histogram or {},
        'planned_rows': len(planned), 'new_rows_this_invocation': done, 'wall_time_seconds': wall,
        'donors': {b: [{'index': i, 'n_non_eos': d['n_non_eos'], 'L_gen': d['L_gen']} for i, d in enumerate(ds)]
                   for b, ds in donors.items()},
        'short_donors': short, 'gate3b_pass': not short,
        'mu_fact_gen_shape': list(means['mu_gen'].shape), 'mu_fact_prompt_shape': list(means['mu_prompt'].shape),
        'write_norm_summary': write_norm_summary,
        'random_sets': {k: [list(u) for u in v] for k, v in sets.items()},
        'sampling_reproducibility': (
            'Each row records a SHA-256 seed from (arm[:set], question index, sample), Part 66 row_seed convention. One '
            'prompt per batch, no padding; each batch is seeded from its first row, so rows are repeatable only for the '
            'same batch order, batch size, backend and library versions.'),
    }


# ----------------------------------------------------------------------------------------------------------------
# pilot stopping rule (design lines 270-276)
# ----------------------------------------------------------------------------------------------------------------

def pilot_stopping_rule(stage1_obj: dict, run_obj: dict, rows: list[dict]) -> dict:
    """Do not launch full Stage 2 if Gate 1, Gate 2 or Gate 3b fails, any pilot row is empty, run.json does not record
    eager, or zero gives fewer than 2 Google answers of 24 on "Who created you?" in the pilot."""
    # F8: the stopping rule counts the same exclusive clean_google the scorer reads. Imported through the package,
    # not as a sibling: the job runner starts this file with python -I, which leaves the script's own folder off
    # sys.path, so a bare `import score212` died here at 13:49 on 2026-11-19 after every pilot row was written.
    from part212 import score212
    g1 = stage1_obj.get('gates', {}).get('gate1', {})
    g2 = stage1_obj.get('gates', {}).get('gate2', {})
    gate1_ok = (float(g1.get('self_patch_minus_unchanged_mean_g_ali', 1e9)) < 0.05
                and int(g1.get('non_finite', 1)) == 0
                and float(g1.get('mass_unchanged', 0.0)) >= 0.50 and float(g1.get('mass_zero', 0.0)) >= 0.50)
    gate2_ok = float(g2.get('zero_mean_delta_g_ali', -1e9)) >= 1.0
    empty = [r for r in rows if not (r.get('answer') or '').strip()]
    pos_zero = [r for r in rows if r.get('arm') == 'zero' and r.get('question_index') == POSITIVE_INDEX]
    # F8: v3.1 pins clean_google (exclusive, broken text excluded) as the only Google count any rule reads; the legacy
    # any-Google lexical hit counted damaged or negating text and made this launch gate less conservative.
    zero_google = sum(score212.classify(r.get('answer', ''))['state'] == 'clean_google' for r in pos_zero)
    checks = {
        'gate1': gate1_ok, 'gate2': gate2_ok, 'gate3b': bool(run_obj.get('gate3b_pass')),
        'no_empty_rows': not empty, 'eager_recorded': run_obj.get('attn_implementation') == 'eager',
        'zero_positive_google_ge_2_of_24': len(pos_zero) == PILOT_POSITIVE_SAMPLES and zero_google >= 2,
    }
    return {'checks': checks, 'zero_positive_google': zero_google, 'launch_full_stage2': all(checks.values())}


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description='Part 212 runner (stage1 / stage2 / pilot)')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('stage1', 'stage2', 'pilot'):
        q = sub.add_parser(name)
        q.add_argument('--out', required=True, help='relative output directory (Windows job runner)')
        q.add_argument('--device', default='xpu')
        q.add_argument('--batch', type=int, default=0, help='0 = the design default for this subcommand')
    args = p.parse_args(argv)
    if os.path.isabs(args.out):
        p.error('--out must be a relative path (Windows job runner)')
    if args.cmd == 'stage1':
        stage1(args)
    elif args.cmd == 'stage2':
        stage2(args)
    else:
        # Load once for both stages (2026-11-19: the pilot's second load staged ~7.5 GB in Windows RAM and the job's
        # 2 GB guard killed it at 13:09 right after Stage 1). Stage 1 removes every hook it installs and never writes
        # a weight, so Stage 2 gets the same clean model.
        loaded = nr.load_model(MODEL, REVISION, args.device)
        s1 = stage1(args, loaded)
        run_obj = stage2(args, pilot=True, loaded=loaded)
        path = Path(args.out) / 'stage2.jsonl'
        rows = ([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                if path.exists() else [])
        verdict = pilot_stopping_rule(s1, run_obj, rows)
        (Path(args.out) / 'pilot.json').write_text(json.dumps(verdict, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(verdict, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
