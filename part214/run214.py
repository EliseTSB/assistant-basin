#!/usr/bin/env python3
"""Part 214 runner: Stage 0 (donor & unchanged-maker recording), Stage 1 (H4 dial curve, 4-D diagnostics,
natural-decode bridge), Stage 2 (1,584 uniform batch-1 held-out generations), and the 64-call mini-pilot.

PART214-DESIGN.md v5, as amended at commit cb8cc1e4, is the contract. Every place the design is ambiguous is marked
`AMBIGUITY n` here and listed with its design line in the run report; the readings that turn numbers into verdicts
live in score214.py.

Run under the XPU Python as
    python -I -B part214/run214.py stage0 --out results/part214/stage0 --device xpu
    python -I -B part214/run214.py stage1 --out results/part214/stage1 --device xpu --stage0 results/part214/stage0
    python -I -B part214/run214.py stage2 --out results/part214/stage2 --device xpu --stage0 results/part214/stage0
    python -I -B part214/run214.py pilot  --out results/part214/pilot  --device xpu --stage0 results/part214/stage0

Batch size is 1 for every single call in every stage, by design (design lines 168-189): `generation_batch_histogram`
must come out `{"1": N}` and Gate 3 reads exactly that.

Reused, not rewritten: neuron_report.load_model / build_prompt / find_text_layers / spill_guard, Part 66's
row_seed / _stop_reason and generation-loop shape, Part 212's write-norm pool construction and JSONL repair. The
canonical plan (arm names, planned rows, integer broken bounds) lives in score214.py, which is numpy-only, so the
runner and the scorer cannot disagree about what was planned.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import neuron_report as nr  # noqa: E402
from part66 import part66 as p66  # noqa: E402  (row_seed, _stop_reason, the generation loop shape)
import score214 as sc  # noqa: E402  (the canonical plan; numpy only, so the two files cannot drift)

MODEL = sc.MODEL
REVISION = sc.REVISION
TARGET = [(int(l), int(u)) for l, u in sc.TARGET]
PROMPTS_PATH = HERE / 'prompts214.json'
PROMPTS_SHA256 = sc.PROMPTS_SHA256
MAX_NEW_TOKENS = 48
OPENING_TOKENS = 5
DEFAULT_DEADLINE_MIN = 330              # design line 357
NORM_POOL = 50                          # 4 x 50 write-norm pool units (design line 102)
NORM_POOL_BANK = 'd_fact'               # the pools are recorded on D_fact (design line 101)
NON_FINITE = 3.0e38
OUTPUT_LOCK_NAME = 'part214.lock'

# The pinned Part 183 kernel reference the design says the run asserts against (design line 236).
KERNEL_CHECK_PATH = ROOT / 'results' / 'part183' / 'kernel-check.json'
PILOT_MAX_LATENCY_S = 7.5       # design line 308: do not launch full Stage 2 above 7.5 s/call
PILOT_PLANNED_CALLS = 64        # design line 305

# Design line 137: g_ali = logit(Google) - logsumexp(logit(Al), logit(T)).
GOOGLE_ID, AL_ID, T_ID = 14030, 2033, 51

# g_seq is scored with each maker name's OWN continuation (design line 138's two names do not share a token history:
# Tongyi's `ong`/`yi` must be conditioned on `T` / `T,ong`, not on `Al` / `Al,T`). One forward per name.
GSEQ_NAMES = ('Google', 'Alibaba', 'Tongyi')
GSEQ_PATHS = len(GSEQ_NAMES)

# Stage 1 arm groups (design lines 145-164): 13 + 4 + 10 + 3 + 30 = 60 arms x 25 prompts = 1,500 forwards.
DIAL_ARMS = tuple(sc.RUNG_ARM[a] for a in sc.ALPHA_RUNGS)                 # includes `zero` and `unchanged`
CORE_DIAL_ARMS = ('scale_dial_fact', 'scale_dial_you')      # the two H4 primary dials (design line 159)
DIAGNOSTIC_ARMS = ('scale_dial_fact_frozen', 'scale_dial_you_frozen',
                   'scale_unit_dial_fact', 'scale_unit_dial_you')
REPLICATION_ARMS = ('rep_trace_fact', 'rep_trace_you', 'rep_trace_imp')
RANDOM_CONTROL_ARMS = ('zero', 'scale_dial_fact', 'nat_trace_fact')
# The four hook paths Stage 2 actually uses, timed once each in the pilot OUTSIDE the 64-row manifest
# (design lines 383-390): trace_*, mean_resample_you, prompt_fact_gen_you and scale_dial_*.
TIMING_ARMS = ('trace_resample_fact', 'mean_resample_you', 'prompt_fact_gen_you', 'scale_dial_fact')
TIMING_SAMPLE = 99          # outside every scored sample index, so the timing call cannot share a scored row's seed
NAT_BANK = {'nat_trace_fact': 'd_fact', 'nat_trace_you': 'd_you', 'nat_trace_imp': 'd_imp',
            'rep_trace_fact': 'd_fact', 'rep_trace_you': 'd_you', 'rep_trace_imp': 'd_imp',
            'nat_mean_fact': 'd_fact', 'nat_mean_you': 'd_you', 'nat_mean_imp': 'd_imp',
            # 214.8(a)'s matched comparison: the no-L29 arms must force the SAME donor opening as the natural arm
            # they are compared against, or the comparison is cross-prefix and not matched (design line 305).
            'nat_fact_no_L29': 'd_fact', 'nat_you_no_L29': 'd_you'}
# Which bank's recorded donor_gen[0..4] token ids an arm's Stage 1 opening carries. This is NOT the set of banks an
# arm reads activations from (that is `needed_donor_banks`): the two cross-phase arms read prompt activations from one
# bank and decode activations from the other, and their opening follows the DECODE (gen) bank, because the imported
# decode activations were recorded on that donor's token history and stay paired with it (Amendment 2, Alden #22).
# Kept separate from NAT_BANK on purpose: putting the crosses into NAT_BANK would make `needed_donor_banks` return a
# single bank for them, and the other bank's clamp lookup would then fail.
OPENING_BANK = dict(NAT_BANK, nat_prompt_fact_gen_you='d_you', nat_prompt_you_gen_fact='d_fact')


# ----------------------------------------------------------------------------------------------------------------
# serialization: every JSON write goes through one `default=` handler (numpy scalars, arrays, bools)
# ----------------------------------------------------------------------------------------------------------------

def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'object of type {type(obj).__name__} is not JSON serializable')


def write_json(path, obj) -> None:
    """Write one JSON document with the numpy-aware default handler, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=json_default) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def generation_batch_histogram(rows: list[dict]) -> dict[str, int]:
    """Rows per actual generation-call batch size; the design requires `{"1": N}` in every stage (design line 189)."""
    counts: dict[str, int] = {}
    for row in rows:
        size = row.get('batch_size')
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('a row is missing a valid positive batch_size')
        counts[str(size)] = counts.get(str(size), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


# ----------------------------------------------------------------------------------------------------------------
# deadline (design lines 356-359)
# ----------------------------------------------------------------------------------------------------------------

class DeadlineReached(Exception):
    """Raised by a per-launch guard when the wall-clock budget is spent BETWEEN the calls of one unit of work.

    `loop_until_deadline` checks before each item, which is enough for a unit that is one call. Stage 1's pre-pass
    (24 forwards), its 100 recordings and the three `g_seq` forwards inside one arm/question are several calls each;
    without this guard the deadline can pass during the first of them and the runner still launches the rest.
    """


def deadline_seconds(minutes) -> float:
    return max(0.0, float(minutes)) * 60.0


def deadline_hit(start: float, limit_s: float, now: float | None = None) -> bool:
    """True once the wall-clock budget is spent. Checked BEFORE every call, never mid-call."""
    if limit_s <= 0:
        return True
    return ((time.monotonic() if now is None else float(now)) - float(start)) >= limit_s


def launch_guard(start: float, limit_s: float, clock=time.monotonic):
    """A per-call guard: raises `DeadlineReached` when the next launch would cross the budget."""
    def guard():
        if deadline_hit(start, limit_s, clock()):
            raise DeadlineReached(f'deadline: {limit_s:.1f}s budget spent before the next launch')
    return guard


def loop_until_deadline(items: list, work, deadline_s: float, clock=time.monotonic, start: float | None = None,
                        log=None) -> tuple[list, bool]:
    """Run `work(item)` for each item, stopping before the call that would cross the deadline.

    Returns (results, deadline_reached). The check is before the call, so a call that starts always finishes and the
    saved rows are never a torn write (design line 357: "At the deadline it stops launching calls, writes what it
    has"). `clock`/`start` are injectable so the deadline path is testable without a model or a real wall clock.
    """
    start = clock() if start is None else start
    out, reached = [], False
    for index, item in enumerate(items):
        if deadline_hit(start, deadline_s, clock()):
            reached = True
            if log:
                log(f'deadline: stopping after {index} of {len(items)} calls')
            break
        out.append(work(item))
    return out, reached


# ----------------------------------------------------------------------------------------------------------------
# prompts and ids
# ----------------------------------------------------------------------------------------------------------------

def registration_status(repo=ROOT) -> dict:
    """Is Part 214 registered on public main, and the EXACT bytes that say so (design line 333).

    The design registers via `tools/register_public.py`, which copies the predictions verbatim into PREDICTIONS.md
    under "## Part 214 (registered ...)". This reads that heading AND returns the SHA-256 of the whole file and of the
    matched section, so the admitted run can bind the registration bytes it was admitted under rather than a locally
    re-typed heading.

    Stage 0 is descriptive and may run before registration (design line 99); every patched (intervened) call may not
    -- the code-acceptance contract names "patched generation refused before registration" as a runner behaviour.
    """
    path = Path(repo) / 'PREDICTIONS.md'
    if not path.exists():
        return {'registered': False, 'path': str(path), 'reason': 'PREDICTIONS.md not found',
                'sha256': None, 'section_sha256': None}
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    match = re.search(r'^##\s*Part\s+214\s*\(registered[^\n]*\n.*?(?=^##\s|\Z)', text, re.M | re.S)
    section = match.group(0) if match else None
    return {'registered': match is not None, 'path': str(path),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'section_sha256': hashlib.sha256(section.encode('utf-8')).hexdigest() if section else None,
            'section_bytes': len(section.encode('utf-8')) if section else 0,
            'reason': '' if match else 'no "## Part 214 (registered ...)" section in PREDICTIONS.md'}


def require_admission(args) -> dict:
    """Resolve registration BEFORE any loader or forward call; return the admission record the stage binds.

    Two rules, both of them evidence rules rather than conveniences:

    * `--allow-unregistered` SELECTS AN INERT PATH, unconditionally. It is not a check that can be skipped: a run
      that passes the flag writes a `dry_run: True` record and makes zero loader and zero forward calls, whether or
      not a registration happens to exist. The pre-fix code let a successful status take precedence over the
      option's advertised no-loader promise, so `--allow-unregistered` on a registered tree silently ran the model.
    * A registered claim is checked against the FROZEN digest of the registered section (`sc.REGISTRATION_SHA256`).
      A locally matched `## Part 214 (registered ...)` heading is not evidence that those bytes were the ones
      registered; only a pinned digest is. A build that pins no digest refuses a registered admission rather than
      trusting whatever the working tree currently says.
    """
    status = registration_status()
    pinned = sc.REGISTRATION_SHA256
    base = {'registration_pin': pinned, 'design_commit': sc.DESIGN_COMMIT, 'design_sha256': sc.DESIGN_SHA256}
    if getattr(args, 'allow_unregistered', False):
        return {'admitted': bool(status['registered']), 'dry_run': True, **base, **status}
    if not status['registered']:
        raise SystemExit(
            f'refusing patched generation before registration: {status["reason"]}. Stage 0 may run before '
            f'registration; Stage 1 and Stage 2 may not (PART214-DESIGN.md, Code acceptance). Pass '
            f'--allow-unregistered only for a dry run that will not be scored.')
    if pinned is None:
        raise SystemExit(
            'refusing patched generation: PREDICTIONS.md has a Part 214 heading, but this build pins no registration '
            'digest (score214.REGISTRATION_SHA256 is None), so the admitted run cannot be bound to the registered '
            'bytes. Pin the registered section digest before a scored run.')
    if status['section_sha256'] != pinned:
        raise SystemExit(
            f'refusing patched generation: the live Part 214 section is SHA-256 {status["section_sha256"]}, this '
            f'build pins {pinned}. The registration bytes changed (or the heading is not the registered one).')
    return {'admitted': True, 'dry_run': False, **base, **status}


def dry_run_record(stage, args, admission, clock=time.monotonic) -> dict:
    """The record an unregistered dry run writes: it names itself inert and counts zero model calls.

    Nothing here loads a model, reads a checkpoint, registers a hook or generates a token; that is the point of the
    override (design line 402-423, runner behaviour: "patched generation refused before registration").
    """
    return {'part': sc.PART, 'stage': stage, 'design': sc.DESIGN_VERSION, 'dry_run': True,
            'admitted': bool(admission.get('admitted')), 'registration': admission,
            'reason': 'dry run: no model load and no forward or generation call was made',
            'loader_calls': 0, 'model_calls': 0, 'forward_calls': 0, 'generation_batch_histogram': {},
            'planned_rows': 0, 'deadline_reached': False, 'wall_time_seconds': 0.0,
            'device': getattr(args, 'device', None), 'out': str(getattr(args, 'out', ''))}


def _refuse_dry_run_over_evidence(path) -> None:
    """The non-replacement rule itself: a real stage record is never overwritten by an inert stub."""
    path = Path(path)
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        existing = None
    if not (isinstance(existing, dict) and existing.get('dry_run') is True):
        raise SystemExit(f'refusing to write a dry-run stub over {path}: it holds a real stage record, and a '
                         f'dry run never writes over evidence')


def write_dry_run_record(obj, out, name: str, stage: str) -> None:
    """Write a dry-run stub ONLY where no real evidence exists, with the check INSIDE the write lock.

    A dry run must never write over a stage record produced by a model: overwriting `stage1.json` or `run.json` with
    an inert stub destroys the evidence a scorer reads and leaves a file that still looks like a completed stage.

    The check and the write are one critical section on purpose. The pre-fix code checked `path.exists()` before
    acquiring the output-directory lock, so a real stage record created between the check and the write (another
    process finishing just then) was replaced by the stub. Holding `output_directory_lock` while checking makes
    "absent now" and "still absent when I write" the same instant.
    """
    out = Path(out)
    path = out / name
    with output_directory_lock(out, stage):
        _refuse_dry_run_over_evidence(path)
        write_json(path, obj)


def load_prompts(path=PROMPTS_PATH) -> dict:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROMPTS_SHA256:
        raise SystemExit(f'{path} SHA-256 is {digest}, the design pins {PROMPTS_SHA256}')
    return json.loads(raw.decode('utf-8'))


def question_ids(tok, text: str) -> list[int]:
    return tok.encode(nr.build_prompt(tok, text, False), add_special_tokens=False)


def opening_ids(bank: dict, name: str = 'en_maker') -> list[int]:
    return [int(t) for t in bank['openings'][name]['token_ids']]


def continuation_ids(bank: dict, name: str | None = None) -> list[int]:
    """The continuation tokens a g_seq path appends after the shared opening (design line 138).

    `name=None` keeps the historical single path ('Al', 'T', 'ong'), which is the Alibaba path: its two tokens are
    conditioned on the shared prefix and on `Al`. Scoring Tongyi against THAT path is the defect this fixes -- its
    `ong`/`yi` terms would be conditioned on `Al` and `Al,T`. Pass a maker name to get that name's own continuation.
    """
    names = bank['maker_names']
    if name is None:
        return [int(names['Alibaba']['token_ids'][0]), int(names['Tongyi']['token_ids'][0]),
                int(names['Tongyi']['token_ids'][1])]
    if name not in names:
        raise KeyError(f'no maker name {name!r} in prompts214.json (has {sorted(names)})')
    return [int(t) for t in names[name]['token_ids']]


def stage1_ids(tok, bank: dict, qi: int) -> tuple[list[int], int]:
    """(prompt ids, T) for the Stage 1 recipient: question ids only; the opening is appended by the caller."""
    ids = question_ids(tok, bank['items'][qi]['en_maker'])
    return ids, len(ids)


# ----------------------------------------------------------------------------------------------------------------
# write norms and matched random sets (Part 212's construction, reused)
# ----------------------------------------------------------------------------------------------------------------

def target_write_norms(weights: np.ndarray, units: list[tuple[int, int]] | None = None) -> dict[str, float]:
    """w_k = ||W_down^(L_k)[:, u_k]||_2 -- the norm of the TARGET UNIT'S COLUMN (design line 109).

    `weights` is `down_weights(layers)`: one (d_model, n_units) matrix per target layer. The pre-fix producer took
    `np.linalg.norm(weights[k])`, the norm of the WHOLE layer matrix, which mixes every unit of the layer into w_k and
    moves every alpha-hat (equal target columns with maker=(1,1), donor=(0,1) give 0.50; matrix norms (1,3) give
    0.90). `unit_columns` is the slice the design asks for.
    """
    units = list(TARGET if units is None else units)
    columns = unit_columns(weights, units)
    return {f'{int(layer)}:{int(unit)}': float(np.linalg.norm(columns[:, k]))
            for k, (layer, unit) in enumerate(units)}


def down_weights(layers) -> np.ndarray:
    """One (d_model, n_units) float32 down_proj matrix per target layer, in TARGET order."""
    return np.stack([layers[li].mlp.down_proj.weight.detach().float().cpu().numpy() for li, _ in TARGET])


def unit_columns(weights: np.ndarray, units: list[tuple[int, int]]) -> np.ndarray:
    """The (d_model, K) slice of `down_weights` columns for an arbitrary unit list (TARGET or a random set).

    `weights` holds one matrix per TARGET layer, so the layer picks the matrix and the unit picks the column; this is
    what a random set's own units need (design line 162).
    """
    layer_index = {int(layer): k for k, (layer, _unit) in enumerate(TARGET)}
    cols = [np.asarray(weights[layer_index[int(layer)]], dtype=np.float64)[:, int(unit)] for layer, unit in units]
    return np.stack(cols, axis=1)


def unit_write_norms(abs_sums: dict, counts: dict, weights: np.ndarray) -> dict:
    """m_{L,u} = mean |a_{L,u}| over the recorded positions times ||w_down[:, u]||_2 (design line 109's w_k)."""
    out = {}
    for k, (layer, _unit) in enumerate(TARGET):
        n = int(counts.get(int(layer), 0))
        if n <= 0:
            raise ValueError(f'layer {layer}: no positions accumulated for the write norm')
        mean_abs = np.asarray(abs_sums[int(layer)], dtype=np.float64) / float(n)
        w = np.asarray(weights[k], dtype=np.float64)
        if mean_abs.shape[0] != w.shape[1]:
            raise ValueError(f'{mean_abs.shape[0]} accumulated units but down_proj has {w.shape[1]} columns')
        out[int(layer)] = mean_abs * np.linalg.norm(w, axis=0)
    return out


def write_norm_pools(maker_norm: dict, n_units: int, pool: int = NORM_POOL) -> dict:
    """Per layer, the `pool` units closest to that layer's target in m_{L,u}, target excluded, ties by unit index."""
    pools = {}
    for layer, unit in TARGET:
        row = np.asarray(maker_norm[int(layer)], dtype=np.float64)
        if row.shape[0] != n_units:
            raise ValueError(f'layer {layer} has {row.shape[0]} write norms, model has {n_units} units')
        target_m = row[int(unit)]
        order = sorted((u for u in range(n_units) if u != int(unit)),
                       key=lambda u: (abs(row[u] - target_m), u))
        pools[int(layer)] = [int(u) for u in order[:pool]]
    return pools


def draw_random_set(pools: dict, k: int) -> list[tuple[int, int]]:
    """random_k: one unit per layer drawn with np.random.default_rng(21200 + k) -- Part 212's exact seed scheme."""
    rng = np.random.default_rng(21200 + k)
    return [(int(layer), int(rng.choice(pools[layer]))) for layer in sorted(pools)]


def random_sets(pools: dict, n: int = sc.N_RANDOM_SETS) -> dict:
    return {f'random_{k}': draw_random_set(pools, k) for k in range(n)}


def accumulate_abs(sums: dict, counts: dict, layer: int, activation) -> None:
    a = activation.detach().abs().float().sum(dim=tuple(range(activation.dim() - 1)))
    sums[int(layer)] += np.asarray(a.cpu(), dtype=np.float64)
    counts[int(layer)] = int(counts.get(int(layer), 0)) + int(np.prod(activation.shape[:-1]))


def pool_norm_pass(tok, model, layers, bank: dict, device, weights: np.ndarray, guard=None,
                   progress: dict | None = None) -> dict:
    """24 unchanged D_fact forwards accumulating |a| for every unit of the four target layers (design lines 101-102).

    The write-norm pools are drawn from D_fact, not EN_MAKER as in Part 212: design line 101 says "plus Stage 1's
    4x50 write-norm pool units on `D_fact` for Stage 1's 10 matched random sets". `guard` is checked before EACH of
    the 24 launches, so a deadline that passes mid-pass stops the pass rather than being noticed only afterwards.
    """
    sums = {int(layer): np.zeros(int(layers[layer].mlp.down_proj.in_features), dtype=np.float64)
            for layer, _ in TARGET}
    counts = {int(layer): 0 for layer, _ in TARGET}
    for qi in range(sc.N_QUESTIONS):
        if guard is not None:
            guard()
        ids = question_ids(tok, bank['items'][qi][NORM_POOL_BANK])
        handles = [layers[layer].mlp.down_proj.register_forward_pre_hook(
            lambda _m, args, layer=layer: accumulate_abs(sums, counts, layer, args[0]))
            for layer, _ in TARGET]
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if progress is not None:
            progress['forwards'] = qi + 1
        nr.spill_guard(device, 20.0)
    n_units = int(layers[TARGET[0][0]].mlp.down_proj.in_features)
    norm = unit_write_norms(sums, counts, weights)
    return {'pools': write_norm_pools(norm, n_units), 'maker_norm': norm}


def pool_summary(maker_norm: dict, pools: dict) -> dict:
    out = {}
    for layer, unit in TARGET:
        row = np.asarray(maker_norm[int(layer)], dtype=np.float64)
        chosen = pools[int(layer)]
        out[str(int(layer))] = {'target_unit': int(unit), 'target_m': float(row[int(unit)]),
                                'pool_size': len(chosen), 'pool_min_m': float(row[chosen].min()),
                                'pool_max_m': float(row[chosen].max()), 'layer_max_m': float(row.max())}
    return out


# ----------------------------------------------------------------------------------------------------------------
# recording helpers
# ----------------------------------------------------------------------------------------------------------------

def layer_columns(units: list[tuple[int, int]]) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = {}
    for layer, unit in units:
        cols.setdefault(int(layer), []).append(int(unit))
    return {layer: sorted(set(v)) for layer, v in cols.items()}


def record_greedy_donor(tok, model, layers, text: str, device, units: list[tuple[int, int]]) -> dict:
    """One greedy generation recording `units` at every prefill position and every processed non-EOS decode step.

    Design lines 100-106: prefill 0..T-1 and generated positions 0..L_gen-1, with L_gen = min(n_non_EOS, 47) --
    the 48th generated token is never fed back, so it has no activation (Part 212 v3's contract).
    """
    ids = question_ids(tok, text)
    T = len(ids)
    cols = layer_columns(units)
    cap: dict[int, list] = {layer: [] for layer in cols}
    handles = []
    for layer in sorted(cols):
        layer_cols = cols[layer]

        def hook(_module, args, layer=layer, layer_cols=layer_cols, cap=cap):
            x = args[0]
            cap[layer].append((x[0][-1, layer_cols] if x.shape[1] == 1 else x[0][:, layer_cols]).float().cpu())
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            result = model.generate(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    finally:
        for handle in handles:
            handle.remove()
    generated = [int(t) for t in result[0, T:].tolist()]
    eos = tok.eos_token_id
    n_non_eos = generated.index(int(eos)) if (eos is not None and int(eos) in generated) else len(generated)
    L_gen = int(min(n_non_eos, MAX_NEW_TOKENS - 1))
    prompt = np.zeros((T, len(units)), dtype=np.float64)
    gen = np.zeros((L_gen, len(units)), dtype=np.float64)
    for k, (layer, unit) in enumerate(units):
        col = cols[int(layer)].index(int(unit))
        prompt[:, k] = cap[int(layer)][0][:, col].numpy()
        for j in range(L_gen):
            # generated position j is recorded by decode call j + 1; a missing call is a bug, not a silent zero.
            if 1 + j >= len(cap[int(layer)]):
                raise RuntimeError(f'recorded {len(cap[int(layer)]) - 1} decode calls but L_gen={L_gen}')
            gen[j, k] = cap[int(layer)][1 + j][col].numpy()
    answer = tok.decode(torch.tensor(generated, dtype=torch.long), skip_special_tokens=True).strip()
    return {'prompt': prompt, 'gen': gen, 'n_non_eos': n_non_eos, 'L_gen': L_gen, 'answer': answer,
            'token_ids': generated, 'prompt_tokens': T}


def record_units(layers, units: list[tuple[int, int]], fn) -> np.ndarray:
    """Run fn() with recording pre-hooks on every named unit; return (positions, K) float64 in `units` order."""
    cols = layer_columns(units)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in sorted(cols):
        def hook(_module, args, layer=layer, layer_cols=cols[layer], captured=captured):
            captured[layer] = args[0][0][:, layer_cols].float().cpu()
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        fn()
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in cols if layer not in captured]
    if missing:
        raise RuntimeError(f'no activation captured at layer(s) {missing}')
    width = captured[sorted(cols)[0]].shape[0]
    arr = np.zeros((width, len(units)), dtype=np.float64)
    for k, (layer, unit) in enumerate(units):
        arr[:, k] = captured[int(layer)][:, cols[int(layer)].index(int(unit))].numpy()
    return arr


def load_rig(args):
    """nr.load_model returns (tok, model); LAST_LOAD carries the kernel record the kernel check reads."""
    tok, model = nr.load_model(MODEL, REVISION, device=args.device, attn_implementation='eager',
                               dtype=torch.bfloat16)
    return model, tok, dict(nr.LAST_LOAD)


def kernel_check(load: dict, path=KERNEL_CHECK_PATH) -> dict:
    """Compare this load against the pinned Part 183 kernel reference (design line 236).

    An eager `attn_implementation` request is a request, not evidence that the reference kernel is the one running.
    The design names `results/part183/kernel-check.json` as the kernel check, so this reads that file and compares the
    model, revision, dtype and the ACTUAL implementation transformers set on the loaded config (`nr.LAST_LOAD`). The
    reference file's SHA-256 is recorded so the run binds the exact bytes it was checked against.
    """
    path = Path(path)
    if not path.exists():
        return {'pass': False, 'checks': {'kernel_check_present': False}, 'path': str(path), 'sha256': None,
                'reason': f'{path} not found; the design pins results/part183/kernel-check.json'}
    raw = path.read_bytes()
    ref = json.loads(raw.decode('utf-8'))
    ref_load = ref.get('load') or {}
    actual = load.get('attn_implementation')
    digest = hashlib.sha256(raw).hexdigest()
    checks = {
        'kernel_check_present': True,
        'reference_sha256_matches_pinned': digest == sc.KERNEL_CHECK_SHA256,
        'reference_model_matches': ref.get('model') == MODEL,
        'reference_revision_matches': ref.get('revision') == REVISION,
        'reference_attn_implementation_eager': ref.get('attn_implementation') == 'eager',
        'reference_load_attn_eager': ref_load.get('attn_implementation') == 'eager',
        'loaded_attn_implementation_eager': actual == 'eager',
        'loaded_attn_matches_reference': actual is not None and actual == ref_load.get('attn_implementation'),
        'loaded_dtype_matches_reference': str(load.get('dtype')) == str(ref.get('dtype')),
    }
    return {'pass': all(checks.values()), 'checks': checks, 'path': str(path), 'sha256': digest,
            'pinned_sha256': sc.KERNEL_CHECK_SHA256,
            'reference': {'model': ref.get('model'), 'revision': ref.get('revision'),
                          'attn_implementation': ref.get('attn_implementation'), 'dtype': ref.get('dtype'),
                          'torch': ref.get('torch'), 'transformers': ref.get('transformers')},
            'loaded': load}


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_sha256(path=None) -> str | None:
    """SHA-256 of the frozen design this run was built against (PART214-DESIGN.md at the repo root)."""
    path = Path(path) if path is not None else (ROOT / 'PART214-DESIGN.md')
    return file_sha256(path) if path.exists() else None


def stage_pins(stage0_sha256: str, admission: dict | None = None, pinning: dict | None = None,
               stage0_design_sha256: str | None = None) -> dict:
    """The exact inputs a Stage 2 row is written under; a resume whose pins changed is refused (design line 333).

    The pins name every artifact identity a row is a function of, not just the Stage 0 bytes: the amended design and
    its digest, the prompt bank, the model/revision, the Part 183 kernel reference, the registration receipt and this
    runner's own code. Stage 1 and the pilot must present these same pins before a full run launches, so evidence
    from one recording cannot clear the gate on another (and a code or kernel change invalidates both stages).

    Fix round 3 adds the CALIBRATION half. `stage0_sha256` alone says "some recording"; the registered calibration is
    one recording plus the branch its bytes pin, so `calibration_branch` and the digest of the approved receipt are
    pinned beside it. The design hashes are separate on purpose: `design_sha256` is the design this run is ADMITTED
    under (the live file), `design_sha256_capture` is what Stage 0 recorded when the recording was made (it is never
    regenerated or relabelled), and `design_sha256_publication` is the build's publication pin.

    The RUNTIME is deliberately NOT here. `pins` are the pre-load identity (they gate evidence before a model is
    loaded); the device, the actual load and the library stack are only knowable AFTER the load, so they travel as
    `runtime_identity()` and are compared by `check_resume_runtime` before a single saved row is reused.
    """
    admission = admission or {}
    return {'stage0_sha256': stage0_sha256, 'design_sha256': design_sha256(),
            'design_sha256_capture': stage0_design_sha256,
            'design_sha256_publication': sc.DESIGN_SHA256_PUBLICATION,
            'design': sc.DESIGN_VERSION, 'design_commit': sc.DESIGN_COMMIT,
            'prompts_sha256': PROMPTS_SHA256, 'model': MODEL, 'revision': REVISION,
            'kernel_check_sha256': sc.KERNEL_CHECK_SHA256,
            'registration_pin': sc.REGISTRATION_SHA256,
            'registration_section_sha256': admission.get('section_sha256'),
            'calibration_branch': (pinning or {}).get('branch'),
            'calibration_receipt_sha256': sc.calibration_receipt_sha256(),
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def require_registered_calibration(stage0_sha256: str, pinning: dict | None, stage: str,
                                   stage0_design_sha256: str | None = None) -> dict:
    """Refuse a patched stage whose recording/branch/design is not the ONE registered calibration. PRE-LOADER.

    Stage 1, the pilot and full Stage 2 all call this between reading `stage0.json` and loading the rig. A Stage 0
    that is complete, self-consistent and *different* from the registered recording passes every other check in this
    file (its own alphas, its own Stage 1, its own pilot); only the receipt tells it apart. `REGISTERED_CALIBRATION`
    is `None` today, which refuses everything: the correct fail-closed state before registration.

    `stage0_design_sha256` is the design hash Stage 0 RECORDED (its capture), checked against the receipt; the LIVE
    design this run is admitted under is read from disk here and must be the published contract.
    """
    result = sc.registered_calibration_checks(stage0_sha256, pinning, stage0_design_sha256, design_sha256())
    if not result['pass']:
        raise SystemExit(f'{stage}: refusing patched generation before the loader: {result["reason"]}')
    return result


def runtime_identity(load: dict | None, device) -> dict:
    """What a generated row is a function of besides its inputs: the device, the ACTUAL load, and the library stack.

    `stage_pins` binds the artifacts; this binds the machine. The load record comes from `neuron_report.load_record`
    and names the attention implementation transformers actually set, the dtype and the torch/transformers versions;
    the interpreter and numpy versions are added here because a silent numpy upgrade can move a projection as surely
    as a device change can. Rows carry this and a resume compares it before reusing anything.
    """
    load = dict(load or {})
    return {'device': str(device), 'load': load,
            'python': sys.version.split()[0], 'numpy': np.__version__,
            'torch': torch.__version__, 'transformers': load.get('transformers')}


def runtime_differences(saved: dict | None, current: dict | None) -> list[str]:
    """The named fields on which two runtime identities disagree (empty means they are the same machine)."""
    saved = saved or {}
    current = current or {}
    out = []
    for key in ('device', 'torch', 'transformers', 'python', 'numpy'):
        if saved.get(key) != current.get(key):
            out.append(f'{key}: saved {saved.get(key)!r}, now {current.get(key)!r}')
    if saved.get('load') != current.get('load'):
        out.append(f'load: saved {saved.get("load")!r}, now {current.get("load")!r}')
    return out


# ----------------------------------------------------------------------------------------------------------------
# Stage 0
# ----------------------------------------------------------------------------------------------------------------

def stage0(args, clock=time.monotonic) -> dict:
    """Stage 0: 100 greedy recordings (4 banks x 25) + the D_fact write-norm pool (design lines 99-117).

    The write-norm pass is 24 forwards; like Stage 1's pre-pass it is guarded per launch, so the deadline cannot
    pass during it and still let the remaining forwards (and the 100 recordings that depend on the pools) launch.
    `write_norm_forwards` and the batch histogram report what actually ran, not the nominal 24.
    """
    bank = load_prompts()
    started = clock()
    deadline_s = deadline_seconds(args.deadline_min)
    guard = launch_guard(started, deadline_s, clock)
    model, tok, load = load_rig(args)
    kernel = kernel_check(load)
    if not kernel['pass']:
        raise SystemExit(f'stage0: kernel check failed against {kernel["path"]}: {kernel["checks"]}')
    layers = nr.find_text_layers(model)[1]
    weights = down_weights(layers)
    deadline_reached = False
    progress: dict = {'forwards': 0}
    try:
        pool = pool_norm_pass(tok, model, layers, bank, args.device, weights, guard=guard, progress=progress)
    except DeadlineReached as exc:
        deadline_reached = True
        pool = {'pools': {}, 'maker_norm': {}}
        print(f'stage0: {exc}; the write-norm pool is incomplete, so no bank is recorded', flush=True)
    write_norm_forwards = int(progress['forwards'])
    pools = pool['pools']
    pool_units = [(int(layer), int(u)) for layer in sorted(pools) for u in pools[layer]]
    banks = {}
    greedy_calls = 0
    if not deadline_reached:
        for name in sc.ALL_BANKS:
            units = TARGET + (pool_units if name == NORM_POOL_BANK else [])
            items = []
            for qi in range(25):
                if deadline_hit(started, deadline_s, clock()):
                    deadline_reached = True
                    break
                record = record_greedy_donor(tok, model, layers, bank['items'][qi][name], args.device, units)
                greedy_calls += 1
                item = {'index': qi, 'text': bank['items'][qi][name], 'answer': record['answer'],
                        'token_ids': record['token_ids'], 'prompt_tokens': record['prompt_tokens'],
                        'n_non_eos': record['n_non_eos'], 'L_gen': record['L_gen'],
                        'prompt': np.round(record['prompt'][:, :len(TARGET)], 6).tolist(),
                        'gen': np.round(record['gen'][:, :len(TARGET)], 6).tolist()}
                if units != TARGET:
                    item['pool_units'] = [[int(l), int(u)] for l, u in pool_units]
                    item['pool_prompt'] = np.round(record['prompt'][:, len(TARGET):], 6).tolist()
                    item['pool_gen'] = np.round(record['gen'][:, len(TARGET):], 6).tolist()
                items.append(item)
                print(f'stage0: {name} {qi + 1}/25 L_gen={record["L_gen"]} n_non_eos={record["n_non_eos"]}', flush=True)
                nr.spill_guard(args.device, 20.0)
            banks[name] = {'prompts': items}
            if deadline_reached:
                print('stage0: deadline reached; writing what was recorded (design line 357)', flush=True)
                break
    min_lgen = min((int(p['L_gen']) for b in banks.values() for p in b['prompts']), default=None)
    obj = {
        'part': sc.PART, 'stage': 0, 'design': sc.DESIGN_VERSION,
        'model': MODEL, 'revision': REVISION, 'load': load,
        'attn_implementation': load.get('attn_implementation'), 'device': args.device,
        'kernel_check': kernel, 'design_sha256': design_sha256(),
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prompts_sha256': PROMPTS_SHA256,
        'units': [{'layer': int(l), 'unit': int(u)} for l, u in TARGET],
        'target': [[int(l), int(u)] for l, u in TARGET],
        'template_prefix_tokens': int(bank['template_prefix_tokens']),
        'template_suffix_tokens': int(bank['template_suffix_tokens']),
        'write_norms': target_write_norms(weights),
        'banks': banks,
        'write_norm_pools': {str(k): v for k, v in pools.items()},
        'pool_units': [[int(l), int(u)] for l, u in pool_units],
        'write_norm_summary': pool_summary(pool['maker_norm'], pools) if pools else {},
        'greedy_calls': greedy_calls, 'write_norm_forwards': write_norm_forwards,
        'generation_batch_histogram': {'1': greedy_calls + write_norm_forwards},
        'wall_time_seconds': clock() - started,
        'deadline_min': args.deadline_min, 'deadline_reached': deadline_reached,
        'gate3b_min_L_gen': min_lgen,
    }
    out = Path(args.out)
    with output_directory_lock(out, 'stage0'):
        write_json(out / 'stage0.json', obj)
    print(f'stage0: wrote {out / "stage0.json"} ({greedy_calls} greedy + {write_norm_forwards} norm forwards)',
          flush=True)
    return obj


# ----------------------------------------------------------------------------------------------------------------
# Stage 1: pure value plans (no model, so the tests exercise them directly)
# ----------------------------------------------------------------------------------------------------------------

def stage0_banks(stage0_obj: dict) -> dict:
    """stage0.json -> {bank: [{'prompt': (T, K), 'gen': (L, K), 'L_gen', ...}]} in float64, TARGET columns."""
    out = {}
    for name, bank in (stage0_obj.get('banks') or {}).items():
        out[name] = [{'index': int(p['index']), 'token_ids': [int(t) for t in (p.get('token_ids') or [])],
                      'prompt': np.asarray(p['prompt'], dtype=np.float64),
                      'gen': np.asarray(p['gen'], dtype=np.float64), 'L_gen': int(p['L_gen']),
                      'pool_prompt': np.asarray(p.get('pool_prompt') or [], dtype=np.float64),
                      'pool_gen': np.asarray(p.get('pool_gen') or [], dtype=np.float64),
                      'pool_units': [tuple(u) for u in (p.get('pool_units') or [])]}
                     for p in bank['prompts']]
    return out


def bank_mean_arrays(donor: list[dict], h: int, s: int) -> dict:
    """`mu_*_prompt` (slot profile) and `mu_*_gen` (flat per-unit mean over decode steps), as Stage 2 clamps them."""
    arrays = [d['prompt'] for d in donor]
    K = arrays[0].shape[1]
    profile = np.zeros((h + s + 1, K), dtype=np.float64)
    profile[0:h] = np.mean([a[:h] for a in arrays], axis=0)
    profile[h] = np.concatenate([a[h:a.shape[0] - s] for a in arrays], axis=0).mean(axis=0)
    profile[h + 1:h + 1 + s] = np.mean([a[a.shape[0] - s:] for a in arrays], axis=0)
    gen = np.concatenate([d['gen'] for d in donor if d['gen'].shape[0]], axis=0).mean(axis=0)
    return {'prompt_profile': profile, 'gen': gen, 'h': h, 's': s}


def profile_index(T_q: int, h: int, s: int) -> np.ndarray:
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
    return np.asarray(idx, dtype=np.int64)


def stage1_arm_names() -> list[str]:
    """All 60 Stage 1 arms: 13 core/dial + 4 diagnostics + 10 natural-decode + 3 replication + 30 random."""
    names = (list(DIAL_ARMS) + ['self_patch'] + list(CORE_DIAL_ARMS) + list(DIAGNOSTIC_ARMS)
             + list(sc.NAT_ARMS_10) + list(REPLICATION_ARMS))
    names += [f'random_{k}:{arm}' for k in range(sc.N_RANDOM_SETS) for arm in RANDOM_CONTROL_ARMS]
    if len(names) != 60:
        raise AssertionError(f'{len(names)} Stage 1 arms, the design fixes 60')
    return names


def arm_units(arm: str, sets: dict) -> list[tuple[int, int]]:
    """The units an arm intervenes on: TARGET, or the random set's own four units."""
    if arm.startswith('random_'):
        name = arm.split(':', 1)[0]
        if name not in sets:
            raise ValueError(f'{arm}: Stage 0 recorded no write-norm pool, so the matched random set cannot be drawn')
        return list(sets[name])
    if arm.endswith('no_L29'):
        return [u for u in TARGET if u != (29, 776)]
    return list(TARGET)


def stage1_opening_tokens(arm: str, bank: dict, donors: dict, qi: int) -> list[int]:
    """The five opening TOKENS the Stage 1 recipient forward carries, by binding group (design lines 180-204, v5).

    * every `nat_*` arm -- INCLUDING the `nat_trace_fact` path inside the matched-random check -- places the donor's
      own first five greedy tokens, so the model's own opening is not used; these arms change the prefix and are
      diagnostics, never part of the donor-vs-dial comparison;
    * every `rep_*` arm uses Part 212's neutral opening `'The main answer is **'`;
    * everything else (the 10-rung curve, the two dials, `self_patch`, the per-unit and random `zero`/`scale_dial`
      controls) uses the fixed maker opening `'I was developed by **'`, so rung-to-rung comparisons are exact.

    A `nat_*` arm with no recorded donor tokens is a Stage 0 bug, not a reason to fall back to the maker opening:
    a silent fallback would relabel a cross-prefix comparison as a matched one (the pre-fix behaviour for the two
    `no_L29` arms and for `random_k:nat_trace_fact`).
    """
    base = arm.split(':', 1)[-1]
    if base.startswith('rep_'):
        return opening_ids(bank, 'en_neutral')
    if base.startswith('nat_'):
        donor_bank = OPENING_BANK.get(base)
        tokens = None
        if donor_bank is not None and qi < len(donors.get(donor_bank) or []):
            tokens = donors[donor_bank][qi].get('token_ids') or None
        if not tokens or len(tokens) < OPENING_TOKENS:
            raise RuntimeError(f'{arm}: no recorded donor token ids for question {qi} '
                               f'(bank {donor_bank!r}); a nat_* arm cannot be scored at a substituted opening')
        return [int(t) for t in tokens[:OPENING_TOKENS]]
    return opening_ids(bank, 'en_maker')


def stage1_opening_name(arm: str, bank: dict, donors: dict, qi: int) -> str:
    """A stable label for the opening group, recorded beside each arm's readouts.

    It is derived from the SAME rule as `stage1_opening_tokens`, so a `donor_tokens` label can no longer disagree with
    a maker-opening forward.
    """
    base = arm.split(':', 1)[-1]
    if base.startswith('rep_'):
        return 'en_neutral'
    if base.startswith('nat_'):
        return 'donor_tokens'
    return 'en_maker'


def stage1_opening_preflight(planned: list[str], bank: dict, donors: dict, sets: dict,
                              n_questions: int = 25) -> dict:
    """Resolve every planned arm's opening and donor activations for every question BEFORE the model loads.

    A Stage 1 arm whose opening or donor lookup cannot be built is a runner defect, and finding it after the 124
    pre-pass forwards and part of the arm loop (as the 2026-11-20 14:20 attempt did) wastes the run. This walks the
    exact functions `stage1_call` uses (`stage1_opening_tokens`, `needed_donor_banks`, `donor_arrays_for`) on
    the Stage 0 bytes and raises on the first failure, naming the arm and question.

    Returns the per-arm opening provenance written into stage1.json: the opening group, the bank whose token ids
    the opening carries (None for a fixed opening), the activation banks read, and the per-question token ids.
    """
    out = {}
    for arm in planned:
        base = arm.split(':', 1)[-1]
        units = arm_units(arm, sets)
        reads = sorted(needed_donor_banks(base))
        per_q = []
        for qi in range(n_questions):
            try:
                tokens = stage1_opening_tokens(arm, bank, donors, qi)
                for b in reads:
                    donor_arrays_for(b, qi, units, donors)
            except Exception as exc:
                raise RuntimeError(f'stage1 preflight: {arm} question {qi}: {exc}') from exc
            per_q.append([int(t) for t in tokens])
        out[arm] = {'group': stage1_opening_name(arm, bank, donors, 0),
                    'opening_bank': OPENING_BANK.get(base) if base.startswith('nat_') else None,
                    'activation_banks': reads, 'token_ids': per_q}
    return out


def needed_donor_banks(base: str) -> set[str]:
    """Which Stage 0 banks one Stage 1 arm actually reads.

    Only `D_fact` carries the 4x50 pool recording (design line 101), so a random control's `nat_trace_fact` arm can
    only be built from `D_fact`; asking any other bank for pool columns would be a lookup on data Stage 0 never stored.
    """
    if base in NAT_BANK:
        return {NAT_BANK[base]}
    if base in ('nat_prompt_fact_gen_you', 'nat_prompt_you_gen_fact'):
        return {'d_fact', 'd_you'}
    if base in ('nat_fact_no_L29', 'nat_you_no_L29'):
        return {'d_fact' if 'fact' in base else 'd_you'}
    return set()


def donor_arrays_for(bank_name: str, qi: int, units: list[tuple[int, int]], donors: dict) -> dict:
    """Prompt and decode activations for `units` from one bank's Stage 0 recording.

    TARGET columns come from `prompt`/`gen`; any other unit (a random set's) comes from the D_fact pool recording,
    which Stage 0 stores beside the target columns for exactly this reason (design line 101).
    """
    item = donors[bank_name][qi]
    if all(tuple(u) in TARGET for u in units):
        columns = [TARGET.index(tuple(u)) for u in units]
        return {'prompt': item['prompt'][:, columns], 'gen': item['gen'][:, columns], 'L_gen': item['L_gen']}
    pool_units = item['pool_units']
    columns = [pool_units.index(tuple(u)) for u in units]
    return {'prompt': item['pool_prompt'][:, columns], 'gen': item['pool_gen'][:, columns],
            'L_gen': item['L_gen']}


def stage1_clamp_values(arm: str, qi: int, ctx: dict) -> np.ndarray | None:
    """Frozen clamp targets for one Stage 1 arm/question, shape (T_q + 5, K), or None when the arm scales instead.

    Positions 0..T-1 are prompt positions; T..T+4 are the five opening positions (design line 148).
    `nat_trace_*` and `nat_mean_*` assign donor ACTIVATIONS to those positions -- the recipient's own tokens never
    change (design line 152).
    """
    base = arm.split(':', 1)[-1]
    T_q, K = ctx['T_q'], ctx['K']
    n = T_q + OPENING_TOKENS
    if base == 'self_patch':
        return np.asarray(ctx['clean'], dtype=np.float64)
    if base.startswith('nat_trace_') or base.startswith('rep_trace_'):
        donor = ctx['donors'][NAT_BANK[base]]
        out = np.empty((n, K), dtype=np.float64)
        out[:T_q] = donor['prompt'][:T_q]
        out[T_q:] = donor['gen'][:OPENING_TOKENS]
        return out
    if base.startswith('nat_mean_'):
        mean = ctx['means'][NAT_BANK[base]]
        out = np.empty((n, K), dtype=np.float64)
        out[:T_q] = mean['prompt_profile'][profile_index(T_q, mean['h'], mean['s'])]
        out[T_q:] = mean['gen'][None, :]
        return out
    if base == 'nat_prompt_fact_gen_you':
        out = np.empty((n, K), dtype=np.float64)
        out[:T_q] = ctx['donors']['d_fact']['prompt'][:T_q]
        out[T_q:] = ctx['donors']['d_you']['gen'][:OPENING_TOKENS]
        return out
    if base == 'nat_prompt_you_gen_fact':
        out = np.empty((n, K), dtype=np.float64)
        out[:T_q] = ctx['donors']['d_you']['prompt'][:T_q]
        out[T_q:] = ctx['donors']['d_fact']['gen'][:OPENING_TOKENS]
        return out
    if base in ('nat_fact_no_L29', 'nat_you_no_L29'):
        donor = ctx['donors']['d_fact' if 'fact' in base else 'd_you']
        out = np.empty((n, K), dtype=np.float64)
        out[:T_q] = donor['prompt'][:T_q]
        out[T_q:] = donor['gen'][:OPENING_TOKENS]
        return out
    if base in ('scale_dial_fact_frozen', 'scale_dial_you_frozen'):
        # The frozen dials clamp a_clean times the phase factor instead of the live value (design line 150).
        key = 'fact' if 'fact' in base else 'you'
        factors = np.empty((n, K), dtype=np.float64)
        factors[:T_q] = ctx['dial'][key]['prompt']
        factors[T_q:] = ctx['dial'][key]['gen']
        return np.asarray(ctx['clean'], dtype=np.float64) * factors
    return None


def stage1_factors(arm: str, qi: int, ctx: dict) -> np.ndarray | None:
    """Per-position, per-unit multiplicative factors for the online arms, shape (T_q + 5, K), or None otherwise.

    Online arms multiply the LIVE activation (design lines 146-150): fixed rungs use one alpha on 0..T+4; the two-
    phase dials use alpha_prompt on 0..T-1 and alpha_gen on T..T+4; the per-unit dials use alpha_phase,k per unit.
    """
    base = arm.split(':', 1)[-1]
    T_q, K = ctx['T_q'], ctx['K']
    n = T_q + OPENING_TOKENS
    if base == 'unchanged':
        return None          # alpha = 1.0 IS `unchanged`: no hook, not an identity hook (design line 156)
    if base in ctx['rung_alpha']:
        return np.full((n, K), float(ctx['rung_alpha'][base]), dtype=np.float64)
    if base in ('scale_dial_fact', 'scale_dial_you'):
        key = 'fact' if base.endswith('fact') else 'you'
        factors = np.empty((n, K), dtype=np.float64)
        factors[:T_q] = ctx['dial'][key]['prompt']
        factors[T_q:] = ctx['dial'][key]['gen']
        return factors
    if base in ('scale_unit_dial_fact', 'scale_unit_dial_you'):
        key = 'fact' if base.endswith('fact') else 'you'
        return _unit_dial_factors(ctx['unit_dial'][key], T_q, K)
    return None


def _unit_dial_factors(per_unit: dict, T_q: int, K: int) -> np.ndarray:
    """Per-unit two-phase factors. A unit whose per-unit denominator is undefined keeps 1.0 (no change) and the
    runner records it in `unit_dial_fallback`, rather than silently clamping that unit to zero (AMBIGUITY 7)."""
    def value(seq, k):
        return 1.0 if seq[k] is None else float(seq[k])
    factors = np.empty((T_q + OPENING_TOKENS, K), dtype=np.float64)
    factors[:T_q] = np.array([value(per_unit['prompt'], k) for k in range(K)])
    factors[T_q:] = np.array([value(per_unit['gen'], k) for k in range(K)])
    return factors


def unit_dial_context(pinning: dict, units: list[tuple[int, int]]) -> dict:
    """Per-unit alpha_prompt / alpha_gen for a bank, in `units` order, None where the unit is undefined."""
    table = pinning['projection_table']['banks']
    labels = pinning['projection_table']['units']
    out = {}
    for key, bank in (('fact', 'd_fact'), ('you', 'd_you')):
        prompt = [table[bank]['prompt']['per_unit'].get(label) for label in labels]
        gen = [table[bank]['gen']['per_unit'].get(label) for label in labels]
        keep = [labels.index(f'L{int(l)} N{int(u)}') for l, u in units if f'L{int(l)} N{int(u)}' in labels]
        out[key] = {'prompt': [prompt[k] for k in keep], 'gen': [gen[k] for k in keep]}
    return out


def stage1_call(arm: str, qi: int, ctx: dict, model, layers, args, guard=None, progress: dict | None = None) -> dict:
    """One Stage 1 arm/question readout, from ONE forward per maker-name continuation path.

    `g_ali`, `g_al`, the 8-maker mass and the monitors live at the single scoring position `T+4`, whose prefix is the
    shared recipient prompt + opening, so those are read from the primary forward. `g_seq` needs each name's own
    continuation: Tongyi's `ong`/`yi` must be conditioned on `T` / `T,ong`, not on the shared `Al,T,ong` path, or the
    score is computed under a token history the name never has (design line 138; a name's own first token is still
    conditioned on the shared prefix, which is identical in every path).

    The three forwards are three launches; `guard` is checked before each, and a deadline that passes between them
    raises `DeadlineReached` with NO value recorded for this arm/question. A g_seq read from one continuation path
    and a stale logits row from another would be a corrupted measurement, not a partial one.

    `progress['forwards']` counts forwards that COMPLETED, for the arm loop's account. A cut between the first and
    second launch spends one real call on a question that is then (correctly) not recorded; that call is still a call
    the rig ran, so the caller's histogram counts it even though no value exists for the question.
    """
    base = arm.split(':', 1)[-1]
    if not stage1_arm_runnable(base, ctx.get('pinning')):
        raise ValueError(f'{arm}: the design stops this arm under the Stage 0 pinning ({stage1_stop_reason(base, ctx.get("pinning"))})')
    units = ctx['units']
    opening = stage1_opening_tokens(arm, ctx['bank'], ctx['donor_tokens'], qi)
    ids, T = stage1_ids(ctx['tok'], ctx['bank'], qi)
    sub = {b: donor_arrays_for(b, qi, units, ctx['donor_tokens'])
           for b in needed_donor_banks(arm.split(':', 1)[-1])}
    needs_dial = base in ('scale_dial_fact', 'scale_dial_you', 'scale_dial_fact_frozen', 'scale_dial_you_frozen')
    needs_unit = base in ('scale_unit_dial_fact', 'scale_unit_dial_you')
    local = dict(ctx, donors=sub, T_q=T, K=len(units),
                 rung_alpha={sc.RUNG_ARM[a]: a for a in sc.ALPHA_RUNGS},
                 dial=_dial_factors(ctx['pinning'], len(units)) if needs_dial else None,
                 unit_dial=unit_dial_context(ctx['pinning'], units) if needs_unit else None)
    clamp = None if base == 'unchanged' else stage1_clamp_values(arm, qi, local)
    factor = None if base == 'unchanged' else stage1_factors(arm, qi, local)
    position = sc.scoring_position(T)
    mon: dict = {}
    logits_by_name = {}
    for k, name in enumerate(GSEQ_NAMES):
        if guard is not None:
            guard()
        seq = list(ids) + opening + continuation_ids(ctx['bank'], name)
        logits_by_name[name] = forward_with_hooks(model, layers, seq, units, clamp, factor, T, args.device,
                                                  mon if k == 0 else {})
        if progress is not None:
            progress['forwards'] = int(progress.get('forwards', 0)) + 1
    primary = logits_by_name[GSEQ_NAMES[0]]
    clean = local['clean']
    values = {'g_ali': g_ali(primary[position]), 'g_seq': g_seq(logits_by_name, position, ctx['bank']),
              'g_al': g_al(primary[position]), 'mass': eight_maker_mass(primary[position], ctx['bank']),
              'l24_n3949': mon.get(24), 'l27_n3325': mon.get(27),
              'pert': None if clean is None else realized_perturbation(
                  clamp if clamp is not None else clean * (factor if factor is not None else 1.0),
                  clean, ctx['unit_weights'])}
    return values


def _dial_factors(pinning: dict | None, K: int) -> dict:
    """The two fitted dials' factors. An unavailable fit is NEVER replaced by identity (design lines 127-130).

    The pre-fix code used 1.0 for a dial that could not be pinned, which produced a `scale_dial_fact` arm that was
    byte-identical to `unchanged` while still carrying the arm's name. The design stops that arm instead, so this
    raises and the planner (`stage1_planned_arms`) never queues it.
    """
    if pinning is None:
        raise ValueError('dial factors require the Stage 0 pinning')
    out = {}
    for key, name in (('fact', 'scale_dial_fact'), ('you', 'scale_dial_you')):
        dial = pinning[name]
        if not dial.get('runnable'):
            raise ValueError(f'{name} is not runnable under the Stage 0 pinning ({dial.get("note")}); the arm must '
                             f'not be launched')
        out[key] = {'prompt': np.full(K, float(dial['prompt'])), 'gen': np.full(K, float(dial['gen']))}
    return out


def stage1_stopped_arms(pinning: dict | None) -> dict[str, str]:
    """arm -> reason, for every Stage 1 arm the design's stop list removes under this pinning.

    `fact` undefined: "no dependent dial arm is launched at all: both `scale_dial_fact` and `scale_dial_you` are
    skipped" plus Stage 1's dial curve and matched-random check (design lines 127-138). The two Gate 1/2 anchors
    (`unchanged`, `zero`) are kept: they are the baseline and the positive control, not curve readings.
    `you` undefined: `scale_dial_you` and the clauses that read it.
    """
    branch = (pinning or {}).get('branch')
    stopped: dict[str, str] = {}
    if branch == 'undefined_fact':
        for arm in ('scale_dial_fact', 'scale_dial_you', 'scale_dial_fact_frozen', 'scale_dial_you_frozen',
                    'scale_unit_dial_fact', 'scale_unit_dial_you'):
            stopped[arm] = 'no projection denominator on the fact schedule'
        for k in range(sc.N_RANDOM_SETS):
            stopped[f'random_{k}:scale_dial_fact'] = 'no projection denominator on the fact schedule'
        for alpha in sc.ALPHA_RUNGS:
            arm = sc.RUNG_ARM[alpha]
            if arm not in ('unchanged', 'zero'):
                stopped[arm] = 'stage1_dial_curve stopped (no projection denominator on the fact schedule)'
    elif branch == 'undefined_you':
        for arm in ('scale_dial_you', 'scale_dial_you_frozen', 'scale_unit_dial_you'):
            stopped[arm] = 'no projection denominator on the you schedule'
    elif branch == 'collapsed':
        # The collapse reassigns `scale_dial_you` to scale_050 BEFORE registration, so the arm still runs; nothing is
        # stopped (design lines 158-164).
        pass
    return stopped


def stage1_arm_runnable(base: str, pinning: dict | None) -> bool:
    return base not in {arm.split(':', 1)[-1] for arm in stage1_stopped_arms(pinning)}


def stage1_stop_reason(base: str, pinning: dict | None) -> str | None:
    for arm, reason in stage1_stopped_arms(pinning).items():
        if arm.split(':', 1)[-1] == base:
            return reason
    return None


def stage1_planned_arms(pinning: dict | None) -> list[str]:
    """The Stage 1 arms this run may launch, in plan order, with the design's stopped arms removed."""
    stopped = stage1_stopped_arms(pinning)
    return [arm for arm in stage1_arm_names() if arm not in stopped]


def forward_with_hooks(model, layers, seq, units, clamp, factor, T_q, device, mon: dict):
    handles = install_stage1_hooks(layers, units, clamp, factor, T_q)
    handles += monitor_handles(layers, mon, sc.scoring_position(T_q))
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], dtype=torch.long, device=device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return out.logits[0]


def install_stage1_hooks(layers, units, clamp: np.ndarray | None, factor: np.ndarray | None, T_q: int):
    """Forward pre-hooks on layers[L].mlp.down_proj for positions 0..T+4 only; continuation positions untouched.

    Exactly one of `clamp` / `factor` is used per arm; both are (T_q + 5, K) float64 arrays in `units` order.
    """
    handles = []
    cols = layer_columns(units)
    for layer in sorted(cols):
        layer_cols = cols[layer]
        idx = [k for k, (l, _u) in enumerate(units) if int(l) == layer]
        target = None if clamp is None else np.asarray(clamp, dtype=np.float64)[:, idx]
        factors = None if factor is None else np.asarray(factor, dtype=np.float64)[:, idx]

        def hook(_module, args, layer_cols=layer_cols, target=target, factors=factors, T_q=T_q):
            x = args[0]
            width = min(x.shape[1], T_q + OPENING_TOKENS)
            col = torch.tensor(layer_cols, dtype=torch.long, device=x.device)
            out = x.clone()
            if target is not None:
                out[:, :width, col] = torch.as_tensor(target[:width], dtype=x.dtype,
                                                      device=x.device).unsqueeze(0)
            elif factors is not None:
                out[:, :width, col] = x[:, :width, col] * torch.as_tensor(
                    factors[:width], dtype=x.dtype, device=x.device).unsqueeze(0)
            return (out,) + tuple(args[1:])
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def monitor_handles(layers, mon: dict, position: int):
    """L24 N3949 and L27 N3325 at the last opening position, read from the arm's own forward (design line 137)."""
    handles = []
    for layer, unit in sc.MONITORS:
        def hook(_module, args, layer=layer, unit=unit):
            mon[layer] = float(args[0][0, position, unit].float().cpu())
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def g_ali(logits_row) -> float:
    v = logits_row.float()
    return float(v[GOOGLE_ID]) - float(np.logaddexp(float(v[AL_ID]), float(v[T_ID])))


def g_al(logits_row) -> float:
    v = logits_row.float()
    return float(v[GOOGLE_ID]) - float(v[AL_ID])


def eight_maker_mass(logits_row, bank: dict) -> float:
    ids = [int(bank['maker_names'][name]['first_id']) for name in bank['maker_names']]
    return float(torch.softmax(logits_row.float()[ids], dim=0).sum())


def seq_logprob(logits, token_ids: list[int], start: int) -> float:
    total = 0.0
    for k, tid in enumerate(token_ids):
        row = logits[start + k].float()
        total += float(row[int(tid)] - torch.logsumexp(row, dim=0))
    return total


def g_seq(logits_by_name: dict, start: int, bank: dict) -> float:
    """g_seq = lp(Google) - max(lp(Alibaba), lp(Tongyi)) with each name scored on its OWN continuation path.

    Passing one shared logits tensor (the pre-fix signature) is refused rather than silently scoring Tongyi under
    `Al,T`, because that is exactly the wrong-token-history defect (design line 138).
    """
    if not isinstance(logits_by_name, dict):
        raise TypeError('g_seq needs one logits tensor per maker-name continuation path '
                        f'({sorted(GSEQ_NAMES)}), not a single shared path')
    names = bank['maker_names']
    missing = [name for name in GSEQ_NAMES if name not in logits_by_name]
    if missing:
        raise KeyError(f'g_seq is missing the continuation path(s) {missing}')
    lp = {name: seq_logprob(logits_by_name[name], [int(t) for t in names[name]['token_ids']], start)
          for name in GSEQ_NAMES}
    return lp['Google'] - max(lp['Alibaba'], lp['Tongyi'])


def realized_perturbation(values: np.ndarray, clean: np.ndarray, unit_weights: np.ndarray) -> float:
    """Mean over positions of ||sum_k (values - clean)[p, k] * w_down[:, k]||_2 (Part 212's definition)."""
    delta = np.asarray(values, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    resid = np.einsum('pk,dk->pd', delta, np.asarray(unit_weights, dtype=np.float64))
    return float(np.linalg.norm(resid, axis=1).mean())


def stage1(args, clock=time.monotonic, rig_loader=None) -> dict:
    """Stage 1: 24 write-norm forwards + 100 unchanged recordings + the eligible arms x 25 x 3 g_seq paths.

    The design's 1,624-forward budget assumed ONE appended continuation shared by both names. `g_seq` is now scored
    on three continuations (one per maker name), so the budget is 24 + 100 + 60 x 25 x 3 = 4,624 forwards; the
    nominal and the actual counts are both written to stage1.json so a deadline-truncated run cannot look complete.
    """
    admission = require_admission(args)
    if admission['dry_run']:
        obj = dry_run_record(1, args, admission, clock)
        write_dry_run_record(obj, args.out, 'stage1.json', 'stage1')
        return obj
    bank = load_prompts()
    if not args.stage0:
        raise SystemExit('stage1 needs --stage0 (Stage 0 donor recordings are its frozen sources)')
    stage0_raw = Path(args.stage0).read_bytes()
    stage0_sha256 = hashlib.sha256(stage0_raw).hexdigest()
    stage0_obj = json.loads(stage0_raw.decode('utf-8'))
    # The registered calibration is checked BEFORE the loader: a complete recording B that is not the registered
    # recording A must not be able to generate its own Stage 1 under A's registration (see require_registered_
    # calibration). `pinning` is a pure function of the Stage 0 bytes, so it is available here.
    pinning = sc.dial_pinning(stage0_obj)
    stage0_design_sha256 = stage0_obj.get('design_sha256')
    calibration = require_registered_calibration(stage0_sha256, pinning, 'stage1', stage0_design_sha256)
    # Every planned arm's opening and donor lookups are resolved on the Stage 0 bytes before the loader, so a
    # mapping defect stops the run here instead of minutes into the arm loop.
    pre_donors = stage0_banks(stage0_obj)
    s0_pools = {int(k): [int(u) for u in v] for k, v in (stage0_obj.get('write_norm_pools') or {}).items()}
    opening_provenance = stage1_opening_preflight(stage1_planned_arms(pinning), bank, pre_donors,
                                                  random_sets(s0_pools) if s0_pools else {})
    started = clock()
    deadline_s = deadline_seconds(args.deadline_min)
    guard = launch_guard(started, deadline_s, clock)
    model, tok, load = (rig_loader or load_rig)(args)
    kernel = kernel_check(load)
    if not kernel['pass']:
        raise SystemExit(f'stage1: kernel check failed against {kernel["path"]}: {kernel["checks"]}')
    runtime = runtime_identity(load, args.device)
    layers = nr.find_text_layers(model)[1]
    weights = down_weights(layers)
    donors = stage0_banks(stage0_obj)
    short = short_donors(stage0_obj)
    shape = gates3b_banks(stage0_obj)
    if short or not shape['pass']:
        raise SystemExit(f'stage1: Gate 3b fails ({len(short)} donor(s) with L_gen < {sc.MIN_DONOR_LGEN}; '
                         f'prompts per bank {shape["prompts_per_bank"]}); the natural-decode arms have no '
                         f'donor_gen[0..4] to place')
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    means = {b: bank_mean_arrays(donors[b][:sc.N_QUESTIONS], h, s) for b in sc.DONOR_BANKS}
    pools = {int(k): [int(u) for u in v] for k, v in (stage0_obj.get('write_norm_pools') or {}).items()}
    sets = random_sets(pools) if pools else {}
    planned = stage1_planned_arms(pinning)
    stopped = stage1_stopped_arms(pinning)

    # --- 24 write-norm pre-pass forwards: re-derive the pools and check them against Stage 0's (design line 141) ---
    # The deadline guard is checked before EVERY forward here and before every recording below: these 124 launches
    # are 124 calls, and a budget that expires during them used to be noticed only at the first arm.
    prepass_forwards = 0
    reached = False
    pre, pre_pools, pool_check = None, {}, {}
    progress: dict = {'forwards': 0}
    try:
        pre = pool_norm_pass(tok, model, layers, bank, args.device, weights, guard=guard, progress=progress)
        prepass_forwards = sc.N_QUESTIONS
    except DeadlineReached as exc:
        reached = True
        prepass_forwards = int(progress['forwards'])
        print(f'stage1: {exc}; the write-norm pre-pass is incomplete and no arm readout is possible', flush=True)
    if pre is not None:
        pre_pools = {str(k): v for k, v in pre['pools'].items()}
        pool_check = {k: pre_pools.get(k) == (stage0_obj.get('write_norm_pools') or {}).get(k) for k in pre_pools}

    # --- 100 unchanged recordings: 4 banks x 25, each on its own bank prompt + the EN_MAKER opening ---
    recorded: dict[str, list] = {name: [] for name in sc.ALL_BANKS}
    if not reached:
        for name in sc.ALL_BANKS:
            complete_bank = True
            for qi in range(25):
                try:
                    guard()
                except DeadlineReached as exc:
                    reached = True
                    complete_bank = False
                    print(f'stage1: {exc}; recorded {prepass_forwards} of 124 pre-pass forwards', flush=True)
                    break
                seq = list(question_ids(tok, bank['items'][qi][name])) + opening_ids(bank, 'en_maker')
                recorded[name].append(record_units(layers, TARGET,
                                                   lambda seq=seq: one_forward(model, seq, args.device)))
                prepass_forwards += 1
                nr.spill_guard(args.device, 20.0)
            if complete_bank:
                print(f'stage1: recorded {name} 25/25', flush=True)
            else:
                break

    # --- the eligible arms x 25 prompts; `unchanged` (alpha = 1.0) IS the baseline row of the curve ---
    # No arm is read unless all 100 recordings exist: every arm's delta is taken against `unchanged`, so a cut
    # baseline would mislabel every value that follows it.
    complete_recordings = not reached

    results = {arm: {'g_ali': [], 'g_seq': [], 'g_al': [], 'mass': [], 'l24_n3949': [], 'l27_n3325': [],
                     'pert': []} for arm in planned}
    work = [(arm, qi) for arm in planned for qi in range(25)]
    _done: list = []
    # Every COMPLETED forward is a call, whichever question it belonged to. `stage1_call` bumps this after each of
    # its three g_seq launches, so a question cut between two of them still contributes the forwards it spent. The
    # pre-fix histogram was derived from complete questions only (`len(_done) * GSEQ_PATHS`), which silently dropped
    # one or two real calls per deadline cut from the account (design line 357: write what was RUN).
    arm_progress: dict = {'forwards': 0}
    if complete_recordings:
        def run(item):
            arm, qi = item
            units = arm_units(arm, sets)
            clean = None
            if not arm.startswith('random_') and not arm.endswith('no_L29'):
                clean = recorded['en_maker'][qi][:len(question_ids(tok, bank['items'][qi]['en_maker'])) + OPENING_TOKENS]
                clean = clean[:, [TARGET.index(u) for u in units]]
            ctx = {'tok': tok, 'bank': bank, 'donors': donors, 'means': means, 'pinning': pinning, 'clean': clean,
                   'donor_tokens': donors, 'units': units, 'unit_weights': unit_columns(weights, units),
                   'arm_opening': stage1_opening_name(arm, bank, donors, qi)}
            values = stage1_call(arm, qi, ctx, model, layers, args, guard=guard, progress=arm_progress)
            for key, value in values.items():
                results[arm][key].append(value)
            return arm, qi

        for index, item in enumerate(work):
            if deadline_hit(started, deadline_s, clock()):
                reached = True
                print(f'stage1: deadline: stopping after {index} of {len(work)} calls', flush=True)
                break
            try:
                _done.append(run(item))
            except DeadlineReached as exc:
                reached = True
                print(f'stage1: {exc} (inside the {GSEQ_PATHS} g_seq forwards of {item[0]} q{item[1]}); that '
                      f'arm/question is not recorded', flush=True)
                break
    if reached:
        print('stage1: deadline reached; writing what was run (design line 357)', flush=True)
    non_finite = sum(1 for arm in results.values() for key in ('g_ali', 'g_seq', 'g_al')
                     for value in arm[key] if not finite(value))
    obj = stage1_object(bank, results, pinning, pools, pre_pools, pool_check, load, args, started, non_finite,
                        n_calls=len(_done), planned_arms=planned, stopped=stopped, deadline_reached=reached,
                        prepass_forwards=prepass_forwards, stage0_sha256=stage0_sha256,
                        admission=admission, kernel=kernel, clock=clock, runtime=runtime,
                        arm_forwards=int(arm_progress['forwards']),
                        calibration=calibration, opening_provenance=opening_provenance,
                        pins=stage_pins(stage0_sha256, admission, pinning, stage0_design_sha256))
    write_stage_record(obj, args.out, 'stage1.json', 'stage1')
    print(f'stage1: wrote {Path(args.out) / "stage1.json"} ({len(_done)} arm/question forwards)', flush=True)
    return obj


def one_forward(model, ids, device) -> None:
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)


def finite(x) -> bool:
    return x is not None and math.isfinite(float(x)) and abs(float(x)) < NON_FINITE


def write_stage_record(obj, out, name: str, stage: str) -> None:
    """Write one stage record under the output-directory lock (atomic; numpy-aware)."""
    out = Path(out)
    with output_directory_lock(out, stage):
        write_json(out / name, obj)


def short_donors(stage0_obj: dict) -> list[dict]:
    """Every donor whose recorded decode trace is shorter than Gate 3b's floor (design line 106)."""
    out = []
    for bank, payload in (stage0_obj.get('banks') or {}).items():
        for item in payload.get('prompts') or []:
            n = int(item.get('L_gen', 0))
            if n < sc.MIN_DONOR_LGEN:
                out.append({'bank': bank, 'index': int(item.get('index', -1)), 'L_gen': n})
    return out


def stage1_object(bank, results, pinning, pools, pre_pools, pool_check, load, args, started, nonfinite,
                  n_calls, planned_arms=None, stopped=None, deadline_reached=False, prepass_forwards=0,
                  stage0_sha256=None, admission=None, kernel=None, clock=time.monotonic, pins=None,
                  runtime=None, arm_forwards=None, calibration=None, opening_provenance=None) -> dict:
    """stage1.json: per-arm per-question readouts, deltas from unchanged, gate inputs, pools and random sets.

    The forward histogram is what was ACTUALLY run: the pre-pass forwards plus every arm-loop forward that COMPLETED
    (`arm_forwards`, bumped inside `stage1_call` after each launch), with the nominal plan written beside it. Two
    pre-fix bugs are closed here: the nominal count was written unconditionally, and then the actual count was
    derived from COMPLETE questions only (`n_calls * GSEQ_PATHS`), so the one or two forwards a cut question
    completed before its next guard refused vanished from the account. `rejected_question_forwards` names that
    remainder explicitly instead of hiding it.

    `runtime` is the device/load/library identity this stage ran on, and `calibration` is the receipt check it passed
    before the loader; both are recorded so a later scorer (or resume) can bind the record to the machine and to the
    registration rather than to a file name.
    """
    arms = {}
    baseline = results.get('unchanged') or {'g_ali': [], 'g_seq': [], 'g_al': [], 'mass': [], 'l24_n3949': [],
                                            'l27_n3325': [], 'pert': []}
    for arm, r in results.items():
        d_ali = delta(r['g_ali'], baseline['g_ali'])
        d_seq = delta(r['g_seq'], baseline['g_seq'])
        l27_shift = delta(r['l27_n3325'], baseline['l27_n3325'])
        arms[arm] = {
            'g_ali': r['g_ali'], 'g_seq': r['g_seq'], 'g_al': r['g_al'], 'mass': r['mass'],
            'delta_g_ali': d_ali, 'delta_g_seq': d_seq, 'l27_shift': l27_shift,
            'mean_g_ali': mean(r['g_ali']), 'mean_delta_g_ali': mean(d_ali), 'mean_g_seq': mean(r['g_seq']),
            'mean_delta_g_seq': mean(d_seq), 'mean_mass': mean(r['mass']),
            'l24_n3949': r['l24_n3949'], 'l27_n3325': r['l27_n3325'], 'mean_l27_shift': mean(l27_shift),
            'realized_perturbation_norm': mean(r['pert']), 'perturbation_norm_per_question': r['pert'],
            'n_questions': len(r['g_ali']),
        }
    expected = 25
    counts = {arm: len(r['g_ali']) for arm, r in results.items()}
    completed_arm_forwards = int(n_calls) * GSEQ_PATHS
    arm_forwards = completed_arm_forwards if arm_forwards is None else int(arm_forwards)
    return {
        'part': sc.PART, 'stage': 1, 'design': sc.DESIGN_VERSION, 'model': MODEL, 'revision': REVISION,
        'load': load, 'attn_implementation': load.get('attn_implementation'), 'device': args.device,
        'runtime': runtime,
        'admission': admission or {}, 'kernel_check': kernel or {},
        'registered_calibration': calibration, 'opening_provenance': opening_provenance,
        'stage0_sha256': stage0_sha256, 'design_sha256': design_sha256(),
        'design_sha256_capture': (pins or {}).get('design_sha256_capture'),
        'design_sha256_publication': sc.DESIGN_SHA256_PUBLICATION,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prompts_sha256': PROMPTS_SHA256,
        'pins': dict(pins) if pins is not None else None,
        'target': [[int(l), int(u)] for l, u in TARGET],
        'unit_labels': [f'L{int(l)} N{int(u)}' for l, u in TARGET],
        'non_finite_readouts': nonfinite, 'arm_question_forwards': n_calls,
        'g_seq_continuations_per_arm_question': GSEQ_PATHS,
        'arm_forward_attempts': arm_forwards,
        'rejected_question_forwards': arm_forwards - completed_arm_forwards,
        'forward_batch_histogram': {'1': int(prepass_forwards) + arm_forwards},
        'nominal_forward_batch_histogram': {'1': stage1_forward_count()},
        'prepass_forwards': int(prepass_forwards),
        'planned_arms': list(planned_arms if planned_arms is not None else stage1_arm_names()),
        'stopped_arms': dict(stopped or {}),
        'planned_questions_per_arm': expected,
        'arm_question_counts': counts,
        'complete_arms': sorted(arm for arm, n in counts.items() if n >= expected),
        'deadline_reached': bool(deadline_reached),
        'questions': [bank['items'][q]['en_maker'] for q in range(25)],
        'arms': arms, 'write_norm_pools': {str(k): v for k, v in pools.items()},
        'write_norm_prepass_pools': pre_pools, 'write_norm_prepass_matches_stage0': pool_check,
        'dial_pinning': {k: v for k, v in pinning.items() if k != 'projection_table'},
        'wall_time_seconds': clock() - started, 'deadline_min': args.deadline_min,
    }


def stage1_forward_count() -> int:
    """24 write-norm + 100 unchanged recordings + 60 arms x 25 x one forward per g_seq path.

    The design's 1,624 assumed a single shared continuation; the corrected histories cost three paths per
    arm/question, so the nominal budget is 24 + 100 + 60 x 25 x 3 = 4,624 batch-1 forwards.
    """
    return (sc.N_QUESTIONS + len(sc.ALL_BANKS) * 25
            + len(stage1_arm_names()) * 25 * GSEQ_PATHS)


def delta(values: list, base: list) -> list:
    return [None if a is None or b is None else float(a) - float(b) for a, b in zip(values, base)]


def mean(values: list) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


# ----------------------------------------------------------------------------------------------------------------
# Stage 2: uniform batch-1 hooks and their sources (design lines 168-189)
# ----------------------------------------------------------------------------------------------------------------

STAGE2_MODE = {'unchanged': 'clean', 'zero': 'zero', 'trace_resample_fact': 'trace_fact',
               'trace_resample_you': 'trace_you', 'scale_dial_fact': 'dial', 'scale_dial_you': 'dial',
               'prompt_fact_gen_you': 'phase_fy', 'mean_resample_you': 'mean_you'}


def stage2_source(mode: str, call: int, L_gen: int) -> tuple[str, int]:
    """Which recorded source a Stage 2 forward reads (design lines 170-176).

    call 0 is the prefill (width T, predicts answer token 0); call s >= 1 processes generated token s-1 (width 1).
    ('prompt_fact', 0) means "the D_fact prompt row at the live position"; ('gen_you', j) a D_you decode step;
    ('mu_prompt', 0) the slot-mapped bank-mean prompt; ('mu_gen', 0) the flat bank-mean decode row;
    ('dial', 0) / ('dial_gen', 0) multiply the live value; ('zero', -1) zeroes it.
    """
    if mode == 'clean':
        return ('none', -1)
    if mode == 'zero':
        return ('zero', -1)
    if mode == 'trace_fact':
        return ('prompt_fact', 0) if call == 0 else ('gen_fact', (call - 1) % L_gen)
    if mode == 'trace_you':
        return ('prompt_you', 0) if call == 0 else ('gen_you', (call - 1) % L_gen)
    if mode == 'phase_fy':
        return ('prompt_fact', 0) if call == 0 else ('gen_you', (call - 1) % L_gen)
    if mode == 'mean_you':
        return ('mu_prompt', 0) if call == 0 else ('mu_gen', 0)
    if mode == 'dial':
        return ('dial', 0) if call == 0 else ('dial_gen', 0)
    raise ValueError(f'unknown Stage 2 mode {mode}')


def stage2_apply(x, cols: list[int], source: tuple[str, int], tensors: dict, call: int):
    """Apply one Stage 2 source to the live activation x of shape (1, W, n_units). Pure; used by hook and tests.

    `tensors` are already sliced to this layer's unit columns (the hook does that once per forward).
    """
    kind, idx = source
    if kind == 'none':
        return x
    col = torch.tensor(cols, dtype=torch.long, device=x.device)
    out = x.clone()
    if kind == 'zero':
        out[:, :, col] = 0.0
    elif kind == 'dial':
        out[:, :, col] = x[:, :, col] * float(tensors['dial_prompt'])
    elif kind == 'dial_gen':
        out[:, :, col] = x[:, :, col] * float(tensors['dial_gen'])
    elif kind == 'mu_gen':
        out[:, :, col] = torch.as_tensor(np.asarray(tensors['mu_gen']), dtype=x.dtype,
                                         device=x.device).unsqueeze(0).unsqueeze(0)
    elif kind == 'mu_prompt':
        value = np.asarray(tensors['mu_prompt'])[:x.shape[1]]
        out[:, :, col] = torch.as_tensor(value, dtype=x.dtype, device=x.device).unsqueeze(0)
    else:
        src = torch.as_tensor(np.asarray(tensors[kind]), dtype=x.dtype, device=x.device)
        value = src[:x.shape[1]] if kind.startswith('prompt') else src[idx][None, :]
        out[:, :, col] = value.unsqueeze(0)
    return out


def install_stage2_hooks(layers, units, mode: str, tensors: dict, T: int, L_gen: int):
    """Stage 2 prefill/decode hooks; the decode call counter is per layer, call s reads stage2_source(mode, s, L_gen)."""
    handles = []
    cols = layer_columns(units)
    for layer in sorted(cols):
        layer_cols = cols[layer]
        idx = [k for k, (l, _u) in enumerate(units) if int(l) == layer]
        sliced = {k: (np.asarray(v)[:, idx] if isinstance(v, np.ndarray) and v.ndim == 2
                      else (np.asarray(v)[idx] if isinstance(v, np.ndarray) and v.ndim == 1 else v))
                  for k, v in tensors.items()}
        state = {'s': 0}

        def hook(_module, args, layer_cols=layer_cols, sliced=sliced, state=state):
            x = args[0]
            if x.shape[1] == 1:
                state['s'] += 1
                call = state['s']
            elif x.shape[1] == T and state['s'] == 0:
                call = 0
            else:
                raise RuntimeError(f'unexpected forward width {x.shape[1]} (T={T})')
            source = stage2_source(mode, call, L_gen)
            return (stage2_apply(x, layer_cols, source, sliced, call),) + tuple(args[1:])
        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
    return handles


def stage2_tensors(stage0_obj: dict, pinning: dict, arm: str, qi: int, T_q: int, units=TARGET) -> dict:
    """Everything the Stage 2 hooks read for one (arm, question), as numpy arrays in `units` order."""
    banks = stage0_banks(stage0_obj)
    h = int(stage0_obj.get('template_prefix_tokens', 3))
    s = int(stage0_obj.get('template_suffix_tokens', 9))
    tensors: dict = {}
    if arm in ('trace_resample_fact', 'prompt_fact_gen_you', 'scale_dial_fact', 'scale_dial_you', 'zero',
               'mean_resample_you'):
        tensors['prompt_fact'] = banks['d_fact'][qi]['prompt']
        tensors['gen_fact'] = banks['d_fact'][qi]['gen']
    if arm in ('trace_resample_you', 'prompt_fact_gen_you', 'mean_resample_you'):
        tensors['prompt_you'] = banks['d_you'][qi]['prompt']
        tensors['gen_you'] = banks['d_you'][qi]['gen']
    if arm == 'mean_resample_you':
        mean = bank_mean_arrays(banks['d_you'][:sc.N_QUESTIONS], h, s)
        tensors['mu_prompt'] = mean['prompt_profile'][profile_index(T_q, h, s)]
        tensors['mu_gen'] = mean['gen']
    if arm in ('scale_dial_fact', 'scale_dial_you'):
        dial = pinning['scale_dial_fact' if arm == 'scale_dial_fact' else 'scale_dial_you']
        if not dial['runnable']:
            raise ValueError(f'{arm} is not runnable under the Stage 0 pinning')
        tensors['dial_prompt'] = float(dial['prompt'])
        tensors['dial_gen'] = float(dial['gen'])
    return tensors


def stage2_donor_L_gen(stage0_obj: dict, arm: str, qi: int) -> int:
    bank = {'trace_resample_fact': 'd_fact', 'prompt_fact_gen_you': 'd_you', 'trace_resample_you': 'd_you',
            'mean_resample_you': 'd_you', 'scale_dial_fact': 'd_fact', 'scale_dial_you': 'd_you'}.get(arm, 'd_fact')
    return int(stage0_banks(stage0_obj)[bank][qi]['L_gen'])


def stage2_units(arm: str) -> list[tuple[int, int]]:
    return [] if arm == 'unchanged' else list(TARGET)


def question_text(bank: dict, qi: int) -> str:
    return bank['items'][qi]['en_maker'] if qi < 25 else bank['guards'][qi - 25]['question']


def is_guard(qi: int) -> bool:
    return int(qi) >= 25


def row_seed(arm: str, set_name, qi: int, sample: int) -> int:
    """Part 66's SHA-256 row seed, byte-identical: int(sha256('{condition}\\0{qi}\\0{sample}')[:4])."""
    return int(p66.row_seed(f'{arm}:{set_name or "target"}', qi, sample))


def stage2_todo(planned: list[dict], done: list[dict], pilot: bool) -> list[dict]:
    existing = {sc.row_key(r) for r in done if bool(r.get('pilot', False)) == bool(pilot)}
    return [r for r in planned if sc.row_key(r) not in existing]


def stage2_generate(row: dict, bank: dict, tok, stage0_obj: dict, pinning: dict, args, model, layers,
                    pins: dict | None = None, runtime: dict | None = None) -> dict:
    """One batch-1 sampled generation for one planned row (design lines 168-176)."""
    arm, qi, sample = row['arm'], int(row['question_index']), int(row['sample'])
    question = question_text(bank, qi)
    ids = question_ids(tok, question)
    T = len(ids)
    units = stage2_units(arm)
    seed = row_seed(arm, row.get('set'), qi, sample)
    torch.manual_seed(seed)
    if args.device == 'xpu' and hasattr(torch, 'xpu'):
        torch.xpu.manual_seed_all(seed)
    mode = STAGE2_MODE[arm]
    tensors: dict = {}
    handles = []
    if units:
        # A guard row has no donor; only `unchanged` carries guards, but the guard is harmless either way.
        L_gen = 1 if is_guard(qi) else stage2_donor_L_gen(stage0_obj, arm, qi)
        tensors = stage2_tensors(stage0_obj, pinning, arm, qi, T, units)
        handles = install_stage2_hooks(layers, units, mode, tensors, T, L_gen)
    try:
        with torch.no_grad():
            result = model.generate(input_ids=torch.tensor([ids], dtype=torch.long, device=args.device),
                                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.95,
                                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    raw = [int(t) for t in result[0, T:].tolist()]
    eos = tok.eos_token_id
    if eos is not None and int(eos) in raw:
        raw = raw[:raw.index(int(eos)) + 1]
    answer = tok.decode(torch.tensor(raw, dtype=torch.long), skip_special_tokens=True).strip()
    return {'arm': arm, 'set': row.get('set'), 'question_index': qi, 'sample': sample, 'question': question,
            'question_kind': bank['items'][qi]['kind'] if qi < 25 else 'guard',
            'seed': seed, 'prompt': nr.build_prompt(tok, question, False), 'prompt_token_ids': ids,
            'answer': answer, 'token_ids': raw, 'stop_reason': p66._stop_reason(raw, tok),
            'batch_size': 1, 'batch_size_requested': 1, 'pilot': bool(row.get('pilot', False)),
            # The pins this row was produced under. A resume that finds a row written under other Stage 0/design
            # bytes refuses instead of relabelling it (design line 333).
            'pins': dict(pins) if pins is not None else None,
            # ... and the RUNTIME (device, actual load, library stack): a resume compares this separately, because
            # the pins are known before the loader and the runtime is not.
            'runtime': runtime,
            'dial': ({'prompt': tensors['dial_prompt'], 'gen': tensors['dial_gen']}
                     if 'dial_prompt' in tensors else None)}


def stage1_evidence(path, pinning: dict | None, pins: dict | None = None) -> dict:
    """The Stage 1 evidence the design's Stage 2 stopping rule names: Gate 1 and Gate 2 (design line 305).

    Gates 1 and 2 are properties of a MEASUREMENT, not of a file name: a Stage 1 run recorded against different
    Stage 0 bytes, design bytes, code or kernel is a valid measurement of something else and must not clear this
    run's gate. `pins` is the full run's pin set; a Stage 1 record that does not carry it exactly is refused.
    """
    path = Path(path)
    if not path.exists():
        return {'pass': False, 'checks': {'stage1_present': False}, 'path': str(path),
                'reason': f'{path} not found; full Stage 2 requires the Stage 1 gate evidence'}
    raw = path.read_bytes()
    stage1_obj = json.loads(raw.decode('utf-8'))
    try:
        metrics = sc.stage1_metrics(stage1_obj, pinning)
    except ValueError as exc:
        # A record the scorer refuses (another registered calibration, an unreadable arm table) is not evidence that
        # can clear a gate. Reported as a failed check rather than raised, so the stopping rule refuses the full run
        # with a reason before the loader instead of crashing out of the precondition.
        return {'pass': False, 'checks': {'stage1_present': True, 'stage1_readable_as_evidence': False},
                'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'pins': stage1_obj.get('pins'),
                'reason': f'Stage 1 cannot be read as this run\'s evidence: {exc}'}
    gate1 = metrics['gates']['gate1']
    gate2 = metrics['gates']['gate2']
    recorded = stage1_obj.get('pins')
    pin_ok = pins is None or recorded == pins
    checks = {'stage1_present': True,
              'stage1_pins_present': isinstance(recorded, dict) and bool(recorded),
              'stage1_pins_match_this_run': bool(pin_ok),
              'gate1_pass': bool(gate1.get('pass')), 'gate2_pass': bool(gate2.get('pass'))}
    return {'pass': all(checks.values()), 'checks': checks, 'path': str(path),
            'sha256': hashlib.sha256(raw).hexdigest(), 'gate1': gate1, 'gate2': gate2, 'pins': recorded,
            'reason': '' if all(checks.values()) else
                      (f'Gate 1 or Gate 2 did not pass' if gate1.get('pass') is False or gate2.get('pass') is False
                       else 'Stage 1 was recorded under different pins than this run (Stage 0/design/code/kernel '
                            'or registration changed)')}


def pilot_evidence(directory, pinning: dict | None, pins: dict | None = None) -> dict:
    """The pilot evidence the design's stopping rule names before full Stage 2 (design lines 305-309).

    Every check is computed from the pilot's OWN 64 rows: pilot rows are never pooled into a full-run number, and a
    missing pilot is a refusal rather than a pass. The latency ceiling is read from the pilot's four timing-only
    calls, which the design places OUTSIDE the 64-row manifest.

    Validation is READ-ONLY. The pre-fix code repaired the pilot's JSONL while *checking* it, which writes to
    another run's evidence outside that run's lock; a torn tail is reported here and abandoned, never fixed.
    """
    directory = Path(directory)
    run_path = directory / 'run.json'
    rows_path = directory / 'stage2.jsonl'
    if not run_path.exists() or not rows_path.exists():
        return {'pass': False, 'checks': {'pilot_files_present': False}, 'path': str(directory),
                'reason': f'{run_path} and {rows_path} must both exist'}
    run = json.loads(run_path.read_text(encoding='utf-8'))
    raw_rows, dropped = read_jsonl_readonly(rows_path)
    rows = sc.classify_rows(raw_rows)
    planned = sc.planned_stage2_rows(pilot=True)
    planned_keys = {sc.row_key(r) for r in planned}
    keys = [sc.row_key(r) for r in rows]
    actual_keys = set(keys)
    run_pins = run.get('pins')
    guards = [r for r in rows if int(r['question_index']) in sc.GUARD_INDICES]
    unchanged_guards = [r for r in guards if r.get('arm') == 'unchanged']
    positive = [r for r in rows if r.get('arm') == 'zero' and int(r['question_index']) == sc.POSITIVE_INDEX]
    counts = {state: sum(1 for r in unchanged_guards if r['state'] == state) for state in sc.STATES}
    correct = sum(1 for r in unchanged_guards if sc.guard_correct(r))
    empty = [r for r in rows if not (r.get('answer') or '').strip()]
    non_finite = [r for r in rows if any(not math.isfinite(float(t)) for t in r.get('token_ids') or [])]
    foreign_kind = [index for index, row in enumerate(rows, start=1) if not sc.row_kind(row)]
    foreign_pins = [index for index, row in enumerate(rows, start=1) if row.get('pins') != run_pins]
    timing = run.get('timing_only') or []
    latencies = [float(t.get('wall_seconds')) for t in timing if t.get('wall_seconds') is not None]
    row_histogram = {}
    if rows:
        try:
            row_histogram = sc.stage2_row_batch_size_histogram(rows)
        except ValueError:
            row_histogram = {}          # a row with no valid batch_size fails the check below, not the caller
    checks = {
        'pilot_files_present': True,
        'pilot_no_torn_tail': dropped == 0,
        'pilot_stage_is_pilot': run.get('stage') == 'pilot',
        'pilot_eager': run.get('attn_implementation') == 'eager',
        'pilot_pins_present': isinstance(run_pins, dict) and bool(run_pins),
        'pilot_pins_match_this_run': pins is None or run_pins == pins,
        'pilot_rows_carry_the_run_pins': bool(rows) and not foreign_pins,
        'pilot_rows_are_all_pilot_kind': not foreign_kind,
        'pilot_histogram_64': run.get('generation_batch_histogram') == {'1': PILOT_PLANNED_CALLS},
        'pilot_rows_describe_the_run_histogram': bool(rows)
                                                 and row_histogram == run.get('generation_batch_histogram'),
        'pilot_no_empty_rows': not empty,
        'pilot_no_non_finite_rows': not non_finite,
        'pilot_planned_keys_present_once': actual_keys == planned_keys and len(rows) == len(planned)
                                           and len(set(keys)) == len(keys),
        'pilot_unchanged_guard_accuracy_ge_22_of_24': len(unchanged_guards) == sc.PILOT_GUARD_N
                                                        and correct >= sc.PILOT_GUARD_CORRECT_MIN,
        'pilot_unchanged_broken_le_1_of_24': len(unchanged_guards) == sc.PILOT_GUARD_N
                                             and counts['broken'] <= sc.PILOT_BROKEN_MAX,
        'pilot_unchanged_clean_google_le_1_of_24': len(unchanged_guards) == sc.PILOT_GUARD_N
                                                   and counts['clean_google'] <= sc.PILOT_GOOGLE_MAX,
        'pilot_zero_positive_ge_2_of_24': len(positive) == sc.POSITIVE_SAMPLES
                                          and sum(1 for r in positive if r['state'] == 'clean_google') >= 2,
        'pilot_timing_only_calls_4': int(run.get('timing_only_calls') or 0) == 4 and len(latencies) == 4,
        'pilot_latency_le_7.5s': bool(latencies) and max(latencies) <= PILOT_MAX_LATENCY_S,
    }
    return {'pass': all(checks.values()), 'checks': checks, 'path': str(directory),
            'rows': len(rows), 'guard_correct': correct, 'guard_counts': counts, 'pins': run_pins,
            'max_latency_s': max(latencies) if latencies else None, 'torn_tail_bytes_dropped': dropped,
            'reason': '' if all(checks.values()) else
                      f'pilot evidence failed: {sorted(k for k, v in checks.items() if not v)}'}


def gates3b_banks(stage0_obj: dict) -> dict:
    """Gate 3b's REQUIRED shape: four banks of exactly 25 donor recordings (design lines 106-107).

    The pre-fix check scanned the donors that were present, so a Stage 0 holding three banks -- or one bank of four
    rows -- passed while the natural-decode arms had no donor_gen to place. Presence of the four banks is checked
    here, before the L_gen floor, and both the runner and the scorer read this shape.
    """
    banks = stage0_obj.get('banks') or {}
    per_bank = {b: len((banks.get(b) or {}).get('prompts') or []) for b in sc.ALL_BANKS}
    return {'pass': all(per_bank[b] == sc.BANK_ROWS for b in sc.ALL_BANKS), 'prompts_per_bank': per_bank,
            'required': {b: sc.BANK_ROWS for b in sc.ALL_BANKS}}


def full_stage2_evidence(args, stage0_obj: dict, pinning: dict | None, short: list[dict],
                         pins: dict | None = None) -> dict:
    """Everything full Stage 2 must show BEFORE it loads a model or launches a call (design lines 305-309).

    The design's stopping rule is owned by the runner: there is no separate Part 214 launcher, and `main()` dispatches
    stage2 straight into this function, so the boundary is exactly here.

    `pins` are the pins of the Stage 0 bytes THIS run is about to use. Both prerequisite artifacts must carry them
    exactly, so a valid Stage 1 and pilot from recording A cannot clear the gate on recording B.
    """
    shape = gates3b_banks(stage0_obj)
    # The recorded design is Stage 0's CAPTURE, compared with the capture this run's pins name -- not with the live
    # file. The design is amended after Stage 0 (design lines 165-166 paste the derived values into it), so demanding
    # that the capture equal the live file would refuse the real post-publication run. A caller that supplies no pins
    # (a unit test of this function alone) falls back to the recording's own capture.
    capture_design = (pins or {}).get('design_sha256_capture') or stage0_obj.get('design_sha256')
    stage0_checks = {'stage0_part_214': stage0_obj.get('part') == sc.PART,
                     'stage0_kind': stage0_obj.get('stage') == 0,
                     'stage0_prompts_pinned': stage0_obj.get('prompts_sha256') == PROMPTS_SHA256,
                     'stage0_design_bytes': stage0_obj.get('design_sha256') == capture_design}
    gate3b_ok = not short and shape['pass'] and all(stage0_checks.values())
    stage1_obj = (stage1_evidence(getattr(args, 'stage1', None), pinning, pins)
                  if getattr(args, 'stage1', None) else {
                      'pass': False, 'checks': {'stage1_present': False},
                      'reason': 'full Stage 2 needs --stage1 (Gate 1 and Gate 2 evidence)'})
    pilot_obj = (pilot_evidence(getattr(args, 'pilot_dir', None), pinning, pins)
                 if getattr(args, 'pilot_dir', None) else {
                     'pass': False, 'checks': {'pilot_files_present': False},
                     'reason': 'full Stage 2 needs --pilot-dir (the 64-call pilot evidence)'})
    checks = {'gate3b_pass': gate3b_ok, 'stage1_pass': bool(stage1_obj.get('pass')),
              'pilot_pass': bool(pilot_obj.get('pass'))}
    return {'pass': all(checks.values()), 'checks': checks,
            'gate3b': {'pass': gate3b_ok, 'short': short, 'banks': shape, 'stage0': stage0_checks},
            'pins': pins, 'stage1': stage1_obj, 'pilot': pilot_obj,
            'reason': '' if all(checks.values()) else
                      f'full Stage 2 refused: {sorted(k for k, v in checks.items() if not v)}'}


def pilot_timing_due(pilot: bool, reached: bool, started: float | None = None, deadline_s: float | None = None,
                     clock=time.monotonic) -> bool:
    """Do the pilot's four timing-only calls launch? At the deadline the runner stops launching CALLS -- and the
    timing calls are calls (design line 357).

    `reached` alone is not enough: the loop notices the deadline only BEFORE the next row, so if the 64th pilot row
    spends the remainder the loop returns `reached=False` and the pre-fix code launched all four timing calls after
    the budget was gone. The clock is therefore consulted here as well.
    """
    if not pilot or reached:
        return False
    if started is None or deadline_s is None:
        return True
    return not deadline_hit(started, deadline_s, clock())


def pilot_timing_plan(pilot: bool, reached: bool, started: float | None = None, deadline_s: float | None = None,
                      clock=time.monotonic) -> dict:
    """Why the pilot's four timing-only calls do or do not launch: `{'launch': bool, 'reason': None|str}`.

    `pilot_timing_due` answers the same question as a bool (and stays the guard the sabotage audit breaks); this
    wrapper names the reason, because a `False` used to be indistinguishable from "not a pilot". The reasons are
    `not_pilot`, `deadline` (either the row loop saw it or the clock has passed since), or `None` when they launch.
    """
    if not pilot:
        return {'launch': False, 'reason': 'not_pilot'}
    if pilot_timing_due(pilot, reached, started, deadline_s, clock):
        return {'launch': True, 'reason': None}
    return {'launch': False, 'reason': 'deadline'}


def timing_only_calls(args, bank, tok, stage0_obj, pinning, model, layers, clock=time.monotonic,
                      pins: dict | None = None, started: float | None = None,
                      deadline_s: float | None = None, runtime: dict | None = None,
                      stop: dict | None = None) -> list[dict]:
    """Four timing-only calls on the hook paths Stage 2 uses, never counted as evaluation rows (design lines 383-390).

    They are deliberately NOT written to stage2.jsonl: Gate 3's `{"1": 64}` describes the scored pilot rows only, and
    `run.json` enumerates these separately as `timing_only_calls`. Each launch is checked against the deadline, so a
    budget that expires between two of them stops the set (and the pilot then fails its four-call precondition).

    `stop` is filled in place with `{'reason', 'planned', 'launched'}` so the caller can record WHY the set is short:
    a deadline that stopped the set used to be visible only as a missing count, with `deadline_reached` left false.
    """
    if stop is not None:
        stop.update({'reason': None, 'planned': len(TIMING_ARMS), 'launched': 0})
    out = []
    for arm in TIMING_ARMS:
        if started is not None and deadline_s is not None and deadline_hit(started, deadline_s, clock()):
            print(f'pilot: deadline reached before timing-only {arm}; stopping the set at {len(out)}/4', flush=True)
            if stop is not None:
                stop['reason'] = 'deadline'
            break
        if arm in ('scale_dial_fact', 'scale_dial_you') and not pinning.get(arm, {}).get('runnable', True):
            continue
        row = {'arm': arm, 'question_index': 5, 'sample': TIMING_SAMPLE, 'set': None, 'pilot': False}
        started_call = clock()
        record = stage2_generate(row, bank, tok, stage0_obj, pinning, args, model, layers, pins=pins,
                                 runtime=runtime)
        elapsed = clock() - started_call
        out.append({'arm': arm, 'question_index': 5, 'sample': TIMING_SAMPLE, 'wall_seconds': elapsed,
                    'answer_chars': len(record['answer']), 'seed': record['seed'],
                    'note': 'timing only; outside the pilot manifest and never an evaluation row'})
        if stop is not None:
            stop['launched'] = len(out)
        print(f'pilot: timing-only {arm} {elapsed:.2f}s', flush=True)
    return out


def stage2(args, pilot: bool = False, clock=time.monotonic, rig_loader=None) -> dict:
    """Stage 2 (or the 64-call pilot): admission, preconditions, resume, then batch-1 rows until done/deadline."""
    admission = require_admission(args)
    if admission['dry_run']:
        obj = dry_run_record('pilot' if pilot else 2, args, admission, clock)
        write_dry_run_record(obj, args.out, 'run.json', 'pilot' if pilot else 'stage2')
        return obj
    bank = load_prompts()
    if not args.stage0:
        raise SystemExit('stage2 needs --stage0 (donor traces and the dial pinning)')
    stage0_raw = Path(args.stage0).read_bytes()
    stage0_obj = json.loads(stage0_raw.decode('utf-8'))
    pinning = sc.dial_pinning(stage0_obj)
    started = clock()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Gate 3b, the registered calibration and the design's stopping rule are PRECONDITIONS: before the loader ---
    # The pins are built FIRST: the Stage 1 and pilot evidence are admitted only if they were recorded against these
    # exact Stage 0 bytes, design, code, kernel, registration and calibration (the pre-fix code compared none of
    # the last two). `require_registered_calibration` binds the recording and branch to the ONE approved receipt, so
    # a different complete recording B cannot clear this gate on its own self-consistent Stage 1 and pilot.
    pins = stage_pins(hashlib.sha256(stage0_raw).hexdigest(), admission, pinning,
                      stage0_obj.get('design_sha256'))
    shape = gates3b_banks(stage0_obj)
    short = short_donors(stage0_obj)
    gate3b_ok = not short and shape['pass']
    if not gate3b_ok:
        raise SystemExit(f'stage2: Gate 3b fails before launch ({len(short)} donor(s) with L_gen < '
                         f'{sc.MIN_DONOR_LGEN}; prompts per bank {shape["prompts_per_bank"]}); donor_gen[0..4] '
                         f'would not exist')
    evidence = None
    # The calibration is the identity precondition: a Stage 1 and pilot from recording B only ever prove that B's
    # prerequisites agree with B, so the recording itself is checked before the evidence they would clear.
    calibration = require_registered_calibration(pins['stage0_sha256'], pinning, 'pilot' if pilot else 'stage2',
                                                  pins.get('design_sha256_capture'))
    if not pilot:
        evidence = full_stage2_evidence(args, stage0_obj, pinning, short, pins)
        if not evidence['pass']:
            raise SystemExit(f'stage2: refusing the full run before any call: {evidence["reason"]} '
                             f'{ {k: v for k, v in evidence["checks"].items()} }')
    with output_directory_lock(out, 'stage2' if not pilot else 'pilot'):
        return _stage2_locked(args, pilot, bank, stage0_obj, pinning, out, started, pins=pins, clock=clock,
                              admission=admission, evidence=evidence, rig_loader=rig_loader,
                              stage0_sha256=pins['stage0_sha256'], calibration=calibration)


def check_resume_pins(saved: list[dict], pins: dict, planned: list[dict], pilot: bool = False) -> None:
    """Refuse to append to a file whose existing rows are not this run's evidence (design line 333).

    Rows are keyed by (arm, question, sample), so without a pin a Stage 0 or design change under the same output
    directory silently relabels old rows under the new pinning. Four things are checked BEFORE any model work and
    before any repair, so the writer cannot create the mixed-kind file the scorer refuses:

    * every saved row is the SAME KIND as this run (pilot rows never enter a full-run file, or the reverse);
    * every key appears at most once (a duplicated row would double-count in every denominator);
    * every key is in the current plan (an arm the plan dropped is not silently reused);
    * every row carries exactly the pins this run is about to write.
    """
    foreign = [index for index, row in enumerate(saved, start=1) if bool(row.get('pilot', False)) != bool(pilot)]
    if foreign:
        raise SystemExit(
            f'refusing resume: {len(foreign)} saved row(s) are {"full-run" if pilot else "pilot"} rows in a '
            f'{"pilot" if pilot else "full-run"} output (first at line {foreign[0]}); pilot and full evidence are '
            f'never mixed in one file (PART214-DESIGN.md line 260)')
    keys = [sc.row_key(row) for row in saved]
    counts: Counter = Counter(keys)
    duplicates = sorted(key for key, n in counts.items() if n > 1)
    if duplicates:
        raise SystemExit(f'refusing resume: saved key {duplicates[0]} appears {counts[duplicates[0]]} times; '
                         f'each planned (arm, question, sample) may be written once')
    planned_keys = {sc.row_key(row) for row in planned}
    extra = [key for key in keys if key not in planned_keys]
    if extra:
        raise SystemExit(f'refusing resume: saved row(s) carry key(s) {extra[0]} that the current plan of '
                         f'{len(planned_keys)} rows does not contain')
    changed = [row for row in saved if (row.get('pins') or {}) != pins]
    if changed:
        first = changed[0]
        raise SystemExit(
            f'refusing resume: {len(changed)} saved row(s) were written under different pins '
            f'(saved {first.get("pins")!r}, now {pins!r}). Use a fresh --out or restore the Stage 0/design inputs.')


def check_resume_runtime(saved: list[dict], runtime: dict | None) -> None:
    """Refuse to reuse saved rows generated on another device, load or library stack (runs AFTER the load).

    `check_resume_pins` binds the INPUTS; this binds the MACHINE, and the split is deliberate: the pins are known
    before a model is loaded and gate the stopping rule, while the runtime is only knowable once the rig exists. The
    pre-fix writer resumed on the pins alone, so a file recorded on the XPU could be appended to on a CPU, or after a
    torch upgrade, and run.json then described the COMBINED account with the new load. Rows are one account only if
    every one of them was produced by the same runtime as this invocation.
    """
    if not saved:
        return
    missing = [index for index, row in enumerate(saved, start=1) if not isinstance(row.get('runtime'), dict)]
    if missing:
        raise SystemExit(
            f'refusing resume: {len(missing)} saved row(s) carry no runtime identity (first at line {missing[0]}), '
            f'so they cannot be shown to come from this device/load/library stack. Use a fresh --out.')
    changed = [row for row in saved if runtime_differences(row.get('runtime'), runtime)]
    if changed:
        raise SystemExit(
            'refusing resume: %d saved row(s) were generated on a different runtime (%s). A resumed file is one '
            'account, so no new generation is admitted across a device or library change; use a fresh --out.'
            % (len(changed), '; '.join(runtime_differences(changed[0].get('runtime'), runtime))))


def _stage2_locked(args, pilot, bank, stage0_obj, pinning, out, started, pins=None, clock=time.monotonic,
                   admission=None, evidence=None, rig_loader=None, stage0_sha256=None, calibration=None) -> dict:
    answer_path = out / 'stage2.jsonl'
    planned = sc.planned_stage2_rows(pilot=pilot, pinning=pinning)
    for row in planned:
        row['pilot'] = pilot
    # Validation is READ-ONLY: a torn tail is dropped in memory and only repaired after every check has passed (and
    # only under this run's lock). The pre-fix writer repaired the file first and then filtered foreign rows out of
    # `done` while still appending beside them, which is how one file came to hold both kinds.
    saved, torn = read_jsonl_readonly(answer_path) if answer_path.exists() else ([], 0)
    if pins is not None:
        check_resume_pins(saved, pins, planned, pilot)
    if answer_path.exists() and torn:
        repair_partial_jsonl(answer_path)
    done = saved
    foreign_shape_rows = 0
    todo = stage2_todo(planned, done, pilot)
    model, tok, load = (rig_loader or load_rig)(args)
    kernel = kernel_check(load)
    if not kernel['pass']:
        raise SystemExit(f'stage2: kernel check failed against {kernel["path"]}: {kernel["checks"]}')
    # The runtime is only knowable now, and it is checked BEFORE any generation: rows from another device/load/library
    # stack are another account, and run.json must never describe them with this invocation's load.
    runtime = runtime_identity(load, args.device)
    check_resume_runtime(saved, runtime)
    layers = nr.find_text_layers(model)[1]
    print(f'stage2: {len(done)} rows already saved, {len(todo)} to run (pilot={pilot})', flush=True)
    rows = list(done)
    deadline_s = deadline_seconds(args.deadline_min)

    with answer_path.open('a', encoding='utf-8') as sink:
        def work(row):
            record = stage2_generate(row, bank, tok, stage0_obj, pinning, args, model, layers, pins=pins,
                                     runtime=runtime)
            sink.write(json.dumps(record, ensure_ascii=False, default=json_default) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
            rows.append(record)
            print(f'stage2: {len(rows)}/{len(planned)} arm={record["arm"]} q={record["question_index"]} '
                  f's={record["sample"]}', flush=True)
            nr.spill_guard(args.device, 20.0)
            return record

        _ran, reached = loop_until_deadline(todo, work, deadline_s, start=started, clock=clock,
                                            log=lambda msg: print(f'stage2: {msg}', flush=True))
    if reached:
        print('stage2: deadline reached; stopping launches and writing partial results (design line 357)', flush=True)
    # Design line 357: at the deadline the runner STOPS launching calls. The pilot's four timing-only calls are
    # calls; launching them after the deadline is the defect this guard closes. `pilot_timing_due` consults the
    # clock as well as `reached`, because the last pilot row can spend the remainder without the loop noticing.
    #
    # The plan is named, not inferred: a timing cut used to leave `deadline_reached: false` in run.json (the row loop
    # had not noticed), so the file described a run the deadline had stopped as if it had finished. `timing_stop`
    # records the reason and the counts, and a deadline that ends the timing calls IS a reached deadline.
    timing_plan = pilot_timing_plan(pilot, reached, started, deadline_s, clock)
    timing_stop = {'reason': timing_plan['reason'], 'planned': len(TIMING_ARMS), 'launched': 0}
    timing = timing_only_calls(args, bank, tok, stage0_obj, pinning, model, layers, clock=clock, pins=pins,
                               started=started, deadline_s=deadline_s, runtime=runtime, stop=timing_stop) \
        if timing_plan['launch'] else []
    timing_stop['launched'] = len(timing)
    if timing_stop['reason'] == 'deadline':
        reached = True
    status = sc.arm_statuses(planned, rows, {'deadline_reached': reached}, pinning)
    donor_lgen = {b: [int(p['L_gen']) for p in (stage0_obj.get('banks') or {}).get(b, {}).get('prompts', [])]
                  for b in sc.ALL_BANKS}
    short = short_donors(stage0_obj)
    shape = gates3b_banks(stage0_obj)
    obj = {
        'part': sc.PART, 'stage': 'pilot' if pilot else 2, 'design': sc.DESIGN_VERSION,
        'model': MODEL, 'revision': REVISION, 'load_this_invocation': load, 'runtime': runtime,
        'admission': admission or {}, 'kernel_check': kernel,
        'registered_calibration': calibration,
        'stage0_sha256': stage0_sha256, 'design_sha256': design_sha256(), 'prompts_sha256': PROMPTS_SHA256,
        'design_sha256_capture': (pins or {}).get('design_sha256_capture'),
        'design_sha256_publication': sc.DESIGN_SHA256_PUBLICATION,
        'design_commit': sc.DESIGN_COMMIT,
        'pins': pins,
        'attn_implementation': load.get('attn_implementation'), 'device_this_invocation': args.device,
        'dtype': 'bfloat16',
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'target': [[int(l), int(u)] for l, u in TARGET], 'batch_size': 1,
        'generation_batch_histogram': generation_batch_histogram(rows),
        'planned_rows': len(planned), 'new_rows_this_invocation': len(rows) - len(done),
        'model_calls_this_invocation': len(rows) - len(done),
        'model_calls_total': len(rows),
        'foreign_shape_rows_ignored': foreign_shape_rows,
        'full_stage2_evidence': {k: v for k, v in (evidence or {}).items() if k != 'stage1'},
        'wall_time_seconds': clock() - started,
        'deadline_min': args.deadline_min, 'deadline_reached': reached,
        'timing_only_calls': len(timing), 'timing_only': timing, 'timing_stop': timing_stop,
        'arm_status': status['arms'], 'planned_rows_by_arm': status['planned'],
        'completed_rows_by_arm': status['done'], 'incomplete_arms': status['incomplete_arms'],
        'not_read_arms': status['not_read_arms'],
        'donors': {b: [{'index': i, 'L_gen': n} for i, n in enumerate(values)] for b, values in donor_lgen.items()},
        'gate3b_pass': not short and shape['pass'],
        'gate3b_min_L_gen': min((n for values in donor_lgen.values() for n in values), default=None),
        'gate3b_prompts_per_bank': shape['prompts_per_bank'],
        'short_donors': short,
        'dial_pinning': {k: v for k, v in pinning.items() if k != 'projection_table'},
        'sampling_reproducibility': (
            'One prompt per batch, batch size 1 for every call. Each row records a SHA-256 seed from '
            '(arm[:set], question index, sample) -- Part 66 row_seed. Rows are repeatable only for the same batch '
            'order, backend and library versions.'),
    }
    write_json(out / 'run.json', obj)
    print(f'stage2: wrote {out / "run.json"} ({len(rows)} rows, batch histogram '
          f'{obj["generation_batch_histogram"]})', flush=True)
    return obj


def pid_is_alive(pid: int) -> bool:
    """Portable best-effort liveness check; unknown OS errors are treated as alive, never stolen (Part 212)."""
    pid = int(pid)
    if pid <= 0:
        return False
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
    """Hold an O_EXCL lock with the current PID while a stage output is being touched.

    Ported from Part 212 (`O_BINARY`, so Windows does not write `\r\n` and defeat the release check). Design line 296:
    "atomic partial-line repair and Windows binary lock files (O_BINARY, Part 212 fix 30db7048)".
    """
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / OUTPUT_LOCK_NAME
    pid = os.getpid()
    payload = f'pid={pid}\n{pid}-{time.time_ns()}\n'.encode('utf-8')
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
    except FileExistsError:
        try:
            owner = int(lock_path.read_text(encoding='utf-8').splitlines()[0].strip().removeprefix('pid='))
        except (OSError, IndexError, ValueError) as exc:
            raise RuntimeError(f'{stage}: cannot read the owner PID from {lock_path}') from exc
        print(f'{stage}: {lock_path} records PID {owner}; pid_is_alive={pid_is_alive(owner)}. '
              f'Delete the file if that run is not live.', flush=True)
        raise RuntimeError(f'{stage}: output directory {directory} is locked by PID {owner}') from None
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
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
    """Repair a torn JSONL tail: keep a complete unterminated JSON value, drop an invalid one (Part 212 30db7048)."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open('r+b') as handle:
        raw = handle.read()
        if not raw or raw.endswith(b'\n'):
            return 0
        keep = raw.rfind(b'\n') + 1
        tail = raw[keep:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            dropped = len(raw) - keep
            handle.seek(keep)
            handle.truncate()
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b'\n')
            handle.flush()
            os.fsync(handle.fileno())
            return 0
    print(f'repair: dropped {dropped} trailing bytes from an incomplete JSONL line in {path}', flush=True)
    return dropped


def load_stage2_jsonl(path) -> list[dict]:
    """Read the append-only JSONL, repairing a torn final line first (Part 212's fix 30db7048).

    This is the WRITER's reader: it repairs, because the caller holds the output-directory lock. Validation of
    another run's evidence must use `read_jsonl_readonly` instead.
    """
    path = Path(path)
    repair_partial_jsonl(path)
    return read_jsonl_readonly(path)[0]


def read_jsonl_readonly(path) -> tuple[list[dict], int]:
    """Parse an append-only JSONL WITHOUT writing anything; return (rows, torn_tail_bytes_dropped).

    A torn FINAL line (a process killed mid-write) is dropped in memory and its byte count reported, which is what
    the writer's reader needs without touching the file. A corrupt INTERIOR line is refused: that is not a partial
    write, and silently skipping it would change what the evidence says. The pre-fix precondition check called the
    repairing reader on the pilot's file, i.e. it wrote to another run's evidence outside that run's lock.
    """
    path = Path(path)
    raw = path.read_bytes()
    if not raw:
        return [], 0
    dropped = 0
    body, tail = raw, b''
    if not raw.endswith(b'\n'):
        cut = raw.rfind(b'\n') + 1
        body, tail = raw[:cut], raw[cut:]
    rows: list[dict] = []

    def parse(chunk: bytes, where: str):
        try:
            value = json.loads(chunk.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f'{path} {where} is corrupt ({exc}); an interior line is not a partial write') from None
        if not isinstance(value, dict):
            raise ValueError(f'{path} {where}: expected a JSON object')
        return value

    for number, line in enumerate(body.split(b'\n'), start=1):
        if not line.strip():
            continue
        rows.append(parse(line, f'line {number}'))
    if tail:
        try:
            rows.append(json.loads(tail.decode('utf-8')))
        except (json.JSONDecodeError, UnicodeDecodeError):
            dropped = len(tail)
        else:
            if not isinstance(rows[-1], dict):
                raise ValueError(f'{path} final line: expected a JSON object')
    return rows, dropped


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def reject_inert_flag_on_stage0(cmd: str, argv: list) -> list:
    """`--allow-unregistered` is NOT a Stage 0 option: refuse it explicitly, before parsing, with a real reason.

    Stage 0 is the descriptive recording every other stage is a function of. It has no inert path to select: it
    always loads the rig and writes `stage0.json`, so advertising a "no model will be loaded" flag on it was a lie
    the CLI told (`:2065-2081` added the option to every subcommand). The flag is therefore defined only where it
    means what it says (stage1/stage2/pilot) and this pre-scan turns it into a named refusal instead of argparse's
    anonymous "unrecognized arguments".
    """
    if cmd == 'stage0' and any(arg == '--allow-unregistered' or arg.startswith('--allow-unregistered=')
                               for arg in argv):
        raise SystemExit(
            'stage0: --allow-unregistered is not a Stage 0 option. Stage 0 is descriptive and always loads the rig, '
            'so there is no inert path to select (PART214-DESIGN.md, Code acceptance); refusing before the loader. '
            'The flag exists only on stage1, stage2 and pilot.')
    return list(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 214 runner')
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('stage0', 'stage1', 'stage2', 'pilot'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out', required=True)
        cmd.add_argument('--stage0', default=None, help='results/part214/stage0 (source for stage1/stage2)')
        cmd.add_argument('--stage1', default=None,
                         help='stage1.json: Gate 1/2 evidence full Stage 2 refuses to launch without')
        cmd.add_argument('--pilot-dir', default=None,
                         help='results/part214/pilot: the 64-call pilot evidence full Stage 2 requires')
        cmd.add_argument('--device', default='xpu')
        cmd.add_argument('--deadline-min', type=float, default=DEFAULT_DEADLINE_MIN,
                         help='wall-clock budget; at the deadline the runner stops launching calls (design line 357)')
        if name != 'stage0':
            cmd.add_argument('--allow-unregistered', action='store_true',
                             help='inert dry run only: load no model and make no forward or generation call, '
                                  'whatever PREDICTIONS.md says (never for a scored run; it never overwrites real '
                                  'evidence)')
    raw = list(sys.argv[1:] if argv is None else argv)
    reject_inert_flag_on_stage0(raw[0] if raw else '', raw)
    args = parser.parse_args(raw)
    if args.cmd == 'stage0':
        result = stage0(args)
    elif args.cmd == 'stage1':
        result = stage1(args)
    elif args.cmd == 'stage2':
        result = stage2(args, pilot=False)
    else:
        result = stage2(args, pilot=True)
    print(json.dumps({'stage': result.get('stage'), 'batch': result.get('generation_batch_histogram'),
                      'deadline_reached': result.get('deadline_reached')}, default=json_default), flush=True)
    return result


if __name__ == '__main__':
    main()
