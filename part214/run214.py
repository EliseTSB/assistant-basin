#!/usr/bin/env python3
"""Part 214 runner: Stage 0 (donor & unchanged-maker recording), Stage 1 (H4 dial curve, 4-D diagnostics,
natural-decode bridge), Stage 2 (1,584 uniform batch-1 held-out generations), and the 64-call mini-pilot.

PART214-DESIGN.md v5 (commit e637e096) is the contract. Every place the design is ambiguous is marked `AMBIGUITY n`
here and listed with its design line in the run report; the readings that turn numbers into verdicts live in
score214.py.

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

# Design line 137: g_ali = logit(Google) - logsumexp(logit(Al), logit(T)).
GOOGLE_ID, AL_ID, T_ID = 14030, 2033, 51

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
            'nat_mean_fact': 'd_fact', 'nat_mean_you': 'd_you', 'nat_mean_imp': 'd_imp'}


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

def deadline_seconds(minutes) -> float:
    return max(0.0, float(minutes)) * 60.0


def deadline_hit(start: float, limit_s: float, now: float | None = None) -> bool:
    """True once the wall-clock budget is spent. Checked BEFORE every call, never mid-call."""
    if limit_s <= 0:
        return True
    return ((time.monotonic() if now is None else float(now)) - float(start)) >= limit_s


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
    """Is Part 214 registered on public main? The design's predictions are copied verbatim into PREDICTIONS.md under
    "## Part 214 (registered ...)" at registration time (design line 333).

    Stage 0 is descriptive and may run before registration (design line 99); every patched (intervened) call may not
    -- the code-acceptance contract names "patched generation refused before registration" as a runner behaviour.
    """
    path = Path(repo) / 'PREDICTIONS.md'
    if not path.exists():
        return {'registered': False, 'path': str(path), 'reason': 'PREDICTIONS.md not found'}
    text = path.read_text(encoding='utf-8')
    registered = re.search(r'^##\s*Part\s+214\s*\(registered', text, re.M) is not None
    return {'registered': registered, 'path': str(path),
            'reason': '' if registered else 'no "## Part 214 (registered ...)" section in PREDICTIONS.md'}


def require_registered(args) -> None:
    """Refuse Stage 1 / Stage 2 / pilot before registration unless the operator overrides explicitly."""
    if getattr(args, 'allow_unregistered', False):
        return
    status = registration_status()
    if not status['registered']:
        raise SystemExit(
            f'refusing patched generation before registration: {status["reason"]}. Stage 0 may run before '
            f'registration; Stage 1 and Stage 2 may not (PART214-DESIGN.md, Code acceptance). Pass '
            f'--allow-unregistered only for a dry run that will not be scored.')


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


def continuation_ids(bank: dict) -> list[int]:
    """The 3 appended continuation tokens: 'Al', 'T', 'ong' (design line 138); hooks stay inactive there."""
    names = bank['maker_names']
    return [int(names['Alibaba']['token_ids'][0]), int(names['Tongyi']['token_ids'][0]),
            int(names['Tongyi']['token_ids'][1])]


def stage1_ids(tok, bank: dict, qi: int) -> tuple[list[int], int]:
    """(prompt ids, T) for the Stage 1 recipient: question ids only; the opening is appended by the caller."""
    ids = question_ids(tok, bank['items'][qi]['en_maker'])
    return ids, len(ids)


# ----------------------------------------------------------------------------------------------------------------
# write norms and matched random sets (Part 212's construction, reused)
# ----------------------------------------------------------------------------------------------------------------

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


def pool_norm_pass(tok, model, layers, bank: dict, device, weights: np.ndarray) -> dict:
    """24 unchanged D_fact forwards accumulating |a| for every unit of the four target layers (design lines 101-102).

    The write-norm pools are drawn from D_fact, not EN_MAKER as in Part 212: design line 101 says "plus Stage 1's
    4x50 write-norm pool units on `D_fact` for Stage 1's 10 matched random sets".
    """
    sums = {int(layer): np.zeros(int(layers[layer].mlp.down_proj.in_features), dtype=np.float64)
            for layer, _ in TARGET}
    counts = {int(layer): 0 for layer, _ in TARGET}
    for qi in range(sc.N_QUESTIONS):
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
    """nr.load_model returns (tok, model); LAST_LOAD carries the kernel record Gate 3 reads."""
    tok, model = nr.load_model(MODEL, REVISION, device=args.device, attn_implementation='eager',
                               dtype=torch.bfloat16)
    return model, tok, dict(nr.LAST_LOAD)


# ----------------------------------------------------------------------------------------------------------------
# Stage 0
# ----------------------------------------------------------------------------------------------------------------

def stage0(args) -> dict:
    """Stage 0: 100 greedy recordings (4 banks x 25) + the D_fact write-norm pool (design lines 99-117)."""
    bank = load_prompts()
    started = time.monotonic()
    model, tok, load = load_rig(args)
    layers = nr.find_text_layers(model)[1]
    weights = down_weights(layers)
    pool = pool_norm_pass(tok, model, layers, bank, args.device, weights)
    pools = pool['pools']
    pool_units = [(int(layer), int(u)) for layer in sorted(pools) for u in pools[layer]]
    banks = {}
    greedy_calls = 0
    deadline_reached = False
    for name in sc.ALL_BANKS:
        units = TARGET + (pool_units if name == NORM_POOL_BANK else [])
        items = []
        for qi in range(25):
            if deadline_hit(started, deadline_seconds(args.deadline_min)):
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
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prompts_sha256': PROMPTS_SHA256,
        'units': [{'layer': int(l), 'unit': int(u)} for l, u in TARGET],
        'target': [[int(l), int(u)] for l, u in TARGET],
        'template_prefix_tokens': int(bank['template_prefix_tokens']),
        'template_suffix_tokens': int(bank['template_suffix_tokens']),
        'write_norms': {f'{int(l)}:{int(u)}': float(np.linalg.norm(np.asarray(weights[k], dtype=np.float64)))
                        for k, (l, u) in enumerate(TARGET)},
        'banks': banks,
        'write_norm_pools': {str(k): v for k, v in pools.items()},
        'pool_units': [[int(l), int(u)] for l, u in pool_units],
        'write_norm_summary': pool_summary(pool['maker_norm'], pools),
        'greedy_calls': greedy_calls, 'write_norm_forwards': sc.N_QUESTIONS,
        'generation_batch_histogram': {'1': greedy_calls + sc.N_QUESTIONS},
        'wall_time_seconds': time.monotonic() - started,
        'deadline_min': args.deadline_min, 'deadline_reached': deadline_reached,
        'gate3b_min_L_gen': min_lgen,
    }
    out = Path(args.out)
    with output_directory_lock(out, 'stage0'):
        write_json(out / 'stage0.json', obj)
    print(f'stage0: wrote {out / "stage0.json"} ({greedy_calls} greedy + {sc.N_QUESTIONS} norm forwards)', flush=True)
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
        return list(sets[arm.split(':', 1)[0]])
    if arm.endswith('no_L29'):
        return [u for u in TARGET if u != (29, 776)]
    return list(TARGET)


def stage1_opening_tokens(arm: str, bank: dict, donors: dict, qi: int) -> list[int]:
    """The five opening TOKENS the Stage 1 recipient forward carries, by binding group (design lines 180-199, v4).

    * every `nat_*` arm places the donor's own first five greedy tokens, so the model's own opening is not used --
      these arms change the prefix and are diagnostics, never part of the donor-vs-dial comparison;
    * every `rep_*` arm uses Part 212's neutral opening `'The main answer is **'`;
    * everything else (the 10-rung curve, the two dials, `self_patch`, the per-unit and random controls) uses the
      fixed maker opening `'I was developed by **'`, so rung-to-rung comparisons are exact.
    """
    base = arm.split(':', 1)[-1]
    if arm.startswith('random_'):
        # v5: only the eight Stage 2 arms generate freely. Every Stage 1 control -- including
        # random_k:nat_trace_fact -- is teacher-forced at the fixed maker opening and scored at T+4.
        return opening_ids(bank, 'en_maker')
    if base.startswith('rep_'):
        return opening_ids(bank, 'en_neutral')
    if base.startswith('nat_'):
        donor_bank = NAT_BANK.get(base)
        tokens = (donors.get(donor_bank) or [{}] * (qi + 1))[qi].get('token_ids') if donor_bank else None
        if tokens and len(tokens) >= OPENING_TOKENS:
            return [int(t) for t in tokens[:OPENING_TOKENS]]
        print(f'WARNING: {arm} has no donor token ids for question {qi}; falling back to the maker opening', flush=True)
    return opening_ids(bank, 'en_maker')


def stage1_opening_name(arm: str, bank: dict, donors: dict, qi: int) -> str:
    """A stable label for the opening group, recorded beside each arm's readouts."""
    if arm.startswith('random_'):
        return 'en_maker'
    base = arm.split(':', 1)[-1]
    if base.startswith('rep_'):
        return 'en_neutral'
    if base.startswith('nat_'):
        return 'donor_tokens'
    return 'en_maker'


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


def stage1_call(arm: str, qi: int, ctx: dict, model, layers, args) -> dict:
    """One Stage 1 arm/question forward and its readouts."""
    base = arm.split(':', 1)[-1]
    units = ctx['units']
    opening = stage1_opening_tokens(arm, ctx['bank'], ctx['donor_tokens'], qi)
    ids, T = stage1_ids(ctx['tok'], ctx['bank'], qi)
    seq = list(ids) + opening + continuation_ids(ctx['bank'])
    sub = {b: donor_arrays_for(b, qi, units, ctx['donor_tokens'])
           for b in needed_donor_banks(arm.split(':', 1)[-1])}
    local = dict(ctx, donors=sub, T_q=T, K=len(units),
                 rung_alpha={sc.RUNG_ARM[a]: a for a in sc.ALPHA_RUNGS},
                 dial=_dial_factors(ctx['pinning'], len(units)),
                 unit_dial=unit_dial_context(ctx['pinning'], units) if ctx['pinning'] else None)
    clamp = None if base == 'unchanged' else stage1_clamp_values(arm, qi, local)
    factor = None if base == 'unchanged' else stage1_factors(arm, qi, local)
    mon: dict = {}
    logits = forward_with_hooks(model, layers, seq, units, clamp, factor, T, args.device, mon)
    position = sc.scoring_position(T)
    clean = local['clean']
    values = {'g_ali': g_ali(logits[position]), 'g_seq': g_seq(logits, position, ctx['bank']),
              'g_al': g_al(logits[position]), 'mass': eight_maker_mass(logits[position], ctx['bank']),
              'l24_n3949': mon.get(24), 'l27_n3325': mon.get(27),
              'pert': None if clean is None else realized_perturbation(
                  clamp if clamp is not None else clean * (factor if factor is not None else 1.0),
                  clean, ctx['unit_weights'])}
    return values


def _dial_factors(pinning: dict | None, K: int) -> dict:
    if pinning is None:
        return {'fact': {'prompt': np.ones(K), 'gen': np.ones(K)}, 'you': {'prompt': np.ones(K), 'gen': np.ones(K)}}
    out = {}
    for key, name in (('fact', 'scale_dial_fact'), ('you', 'scale_dial_you')):
        dial = pinning[name]
        out[key] = {'prompt': np.full(K, dial['prompt'] if dial['runnable'] else 1.0),
                    'gen': np.full(K, dial['gen'] if dial['runnable'] else 1.0)}
    return out


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


def g_seq(logits, start: int, bank: dict) -> float:
    names = bank['maker_names']
    lp_g = seq_logprob(logits, names['Google']['token_ids'], start)
    lp_a = seq_logprob(logits, names['Alibaba']['token_ids'], start)
    lp_t = seq_logprob(logits, names['Tongyi']['token_ids'], start)
    return lp_g - max(lp_a, lp_t)


def realized_perturbation(values: np.ndarray, clean: np.ndarray, unit_weights: np.ndarray) -> float:
    """Mean over positions of ||sum_k (values - clean)[p, k] * w_down[:, k]||_2 (Part 212's definition)."""
    delta = np.asarray(values, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    resid = np.einsum('pk,dk->pd', delta, np.asarray(unit_weights, dtype=np.float64))
    return float(np.linalg.norm(resid, axis=1).mean())


def stage1(args) -> dict:
    """Stage 1: 24 write-norm forwards + 100 unchanged recordings + 60 arms x 25 = 1,624 batch-1 forwards."""
    require_registered(args)
    bank = load_prompts()
    if not args.stage0:
        raise SystemExit('stage1 needs --stage0 (Stage 0 donor recordings are its frozen sources)')
    stage0_obj = json.loads(Path(args.stage0).read_text(encoding='utf-8'))
    started = time.monotonic()
    deadline_s = deadline_seconds(args.deadline_min)
    model, tok, load = load_rig(args)
    layers = nr.find_text_layers(model)[1]
    weights = down_weights(layers)
    pinning = sc.dial_pinning(stage0_obj)
    donors = stage0_banks(stage0_obj)
    h, s = int(bank['template_prefix_tokens']), int(bank['template_suffix_tokens'])
    means = {b: bank_mean_arrays(donors[b][:sc.N_QUESTIONS], h, s) for b in sc.DONOR_BANKS}
    pools = {int(k): [int(u) for u in v] for k, v in (stage0_obj.get('write_norm_pools') or {}).items()}
    sets = random_sets(pools) if pools else {}

    # --- 24 write-norm pre-pass forwards: re-derive the pools and check them against Stage 0's (design line 141) ---
    pre = pool_norm_pass(tok, model, layers, bank, args.device, weights)
    pre_pools = {str(k): v for k, v in pre['pools'].items()}
    pool_check = {k: pre_pools.get(k) == (stage0_obj.get('write_norm_pools') or {}).get(k) for k in pre_pools}

    # --- 100 unchanged recordings: 4 banks x 25, each on its own bank prompt + the EN_MAKER opening ---
    recorded: dict[str, list] = {}
    for name in sc.ALL_BANKS:
        recorded[name] = []
        for qi in range(25):
            seq = list(question_ids(tok, bank['items'][qi][name])) + opening_ids(bank, 'en_maker')
            recorded[name].append(record_units(layers, TARGET, lambda seq=seq: one_forward(model, seq, args.device)))
            nr.spill_guard(args.device, 20.0)
        print(f'stage1: recorded {name} 25/25', flush=True)

    # --- 60 arms x 25 prompts; `unchanged` (alpha = 1.0) IS the baseline row of the 10-point curve ---

    results = {arm: {'g_ali': [], 'g_seq': [], 'g_al': [], 'mass': [], 'l24_n3949': [], 'l27_n3325': [],
                     'pert': []} for arm in stage1_arm_names()}
    work = [(arm, qi) for arm in stage1_arm_names() for qi in range(25)]

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
        values = stage1_call(arm, qi, ctx, model, layers, args)
        for key, value in values.items():
            results[arm][key].append(value)
        return arm, qi

    _done, reached = loop_until_deadline(work, run, deadline_s, start=started,
                                         log=lambda msg: print(f'stage1: {msg}', flush=True))
    if reached:
        print('stage1: deadline reached; writing what was run (design line 357)', flush=True)
    non_finite = sum(1 for arm in results.values() for key in ('g_ali', 'g_seq', 'g_al')
                     for value in arm[key] if not finite(value))
    obj = stage1_object(bank, results, pinning, pools, pre_pools, pool_check, load, args, started, non_finite,
                        len(_done))
    out = Path(args.out)
    with output_directory_lock(out, 'stage1'):
        write_json(out / 'stage1.json', obj)
    print(f'stage1: wrote {out / "stage1.json"} ({len(_done)} arm/question forwards)', flush=True)
    return obj


def one_forward(model, ids, device) -> None:
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)


def finite(x) -> bool:
    return x is not None and math.isfinite(float(x)) and abs(float(x)) < NON_FINITE


def stage1_object(bank, results, pinning, pools, pre_pools, pool_check, load, args, started, nonfinite,
                  n_calls) -> dict:
    """stage1.json: per-arm per-question readouts, deltas from unchanged, gate inputs, pools and random sets."""
    arms = {}
    baseline = results['unchanged']
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
        }
    return {
        'part': sc.PART, 'stage': 1, 'design': sc.DESIGN_VERSION, 'model': MODEL, 'revision': REVISION,
        'load': load, 'attn_implementation': load.get('attn_implementation'), 'device': args.device,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prompts_sha256': PROMPTS_SHA256,
        'target': [[int(l), int(u)] for l, u in TARGET],
        'unit_labels': [f'L{int(l)} N{int(u)}' for l, u in TARGET],
        'non_finite_readouts': nonfinite, 'arm_question_forwards': n_calls,
        'forward_batch_histogram': {'1': stage1_forward_count()},
        'questions': [bank['items'][q]['en_maker'] for q in range(25)],
        'arms': arms, 'write_norm_pools': {str(k): v for k, v in pools.items()},
        'write_norm_prepass_pools': pre_pools, 'write_norm_prepass_matches_stage0': pool_check,
        'dial_pinning': {k: v for k, v in pinning.items() if k != 'projection_table'},
        'wall_time_seconds': time.monotonic() - started, 'deadline_min': args.deadline_min,
    }


def stage1_forward_count() -> int:
    """24 write-norm + 100 unchanged recordings + 60 arms x 25 = 1,624 (design line 143)."""
    return sc.N_QUESTIONS + len(sc.ALL_BANKS) * 25 + len(stage1_arm_names()) * 25


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


def stage2_generate(row: dict, bank: dict, tok, stage0_obj: dict, pinning: dict, args, model, layers) -> dict:
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
            'dial': ({'prompt': tensors['dial_prompt'], 'gen': tensors['dial_gen']}
                     if 'dial_prompt' in tensors else None)}


def timing_only_calls(args, bank, tok, stage0_obj, pinning, model, layers) -> list[dict]:
    """Four timing-only calls on the hook paths Stage 2 uses, never counted as evaluation rows (design lines 383-390).

    They are deliberately NOT written to stage2.jsonl: Gate 3's `{"1": 64}` describes the scored pilot rows only, and
    `run.json` enumerates these separately as `timing_only_calls`.
    """
    out = []
    for arm in TIMING_ARMS:
        if arm in ('scale_dial_fact', 'scale_dial_you') and not pinning.get(arm, {}).get('runnable', True):
            continue
        row = {'arm': arm, 'question_index': 5, 'sample': TIMING_SAMPLE, 'set': None, 'pilot': False}
        started = time.monotonic()
        record = stage2_generate(row, bank, tok, stage0_obj, pinning, args, model, layers)
        elapsed = time.monotonic() - started
        out.append({'arm': arm, 'question_index': 5, 'sample': TIMING_SAMPLE, 'wall_seconds': elapsed,
                    'answer_chars': len(record['answer']), 'seed': record['seed'],
                    'note': 'timing only; outside the pilot manifest and never an evaluation row'})
        print(f'pilot: timing-only {arm} {elapsed:.2f}s', flush=True)
    return out


def stage2(args, pilot: bool = False) -> dict:
    """Stage 2 (or the 64-call pilot): resume, then generate batch-1 rows until done or the deadline fires."""
    require_registered(args)
    bank = load_prompts()
    if not args.stage0:
        raise SystemExit('stage2 needs --stage0 (donor traces and the dial pinning)')
    stage0_obj = json.loads(Path(args.stage0).read_text(encoding='utf-8'))
    pinning = sc.dial_pinning(stage0_obj)
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with output_directory_lock(out, 'stage2' if not pilot else 'pilot'):
        return _stage2_locked(args, pilot, bank, stage0_obj, pinning, out, started)


def _stage2_locked(args, pilot, bank, stage0_obj, pinning, out, started) -> dict:
    answer_path = out / 'stage2.jsonl'
    planned = sc.planned_stage2_rows(pilot=pilot, pinning=pinning)
    for row in planned:
        row['pilot'] = pilot
    saved = load_stage2_jsonl(answer_path) if answer_path.exists() else []
    # Design line 260: pilot outputs are NOT reused in the clean full run. A full run into a pilot's --out (or the
    # reverse) must not fold the other shape's rows into this run's arms, so rows of the other shape are ignored.
    done = [r for r in saved if bool(r.get('pilot', False)) == bool(pilot)]
    foreign_shape_rows = len(saved) - len(done)
    todo = stage2_todo(planned, done, pilot)
    model, tok, load = load_rig(args)
    layers = nr.find_text_layers(model)[1]
    print(f'stage2: {len(done)} rows already saved, {len(todo)} to run (pilot={pilot})', flush=True)
    rows = list(done)
    deadline_s = deadline_seconds(args.deadline_min)

    with answer_path.open('a', encoding='utf-8') as sink:
        def work(row):
            record = stage2_generate(row, bank, tok, stage0_obj, pinning, args, model, layers)
            sink.write(json.dumps(record, ensure_ascii=False, default=json_default) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
            rows.append(record)
            print(f'stage2: {len(rows)}/{len(planned)} arm={record["arm"]} q={record["question_index"]} '
                  f's={record["sample"]}', flush=True)
            nr.spill_guard(args.device, 20.0)
            return record

        _ran, reached = loop_until_deadline(todo, work, deadline_s, start=started,
                                            log=lambda msg: print(f'stage2: {msg}', flush=True))
    if reached:
        print('stage2: deadline reached; stopping launches and writing partial results (design line 357)', flush=True)
    timing = timing_only_calls(args, bank, tok, stage0_obj, pinning, model, layers) if pilot else []
    status = sc.arm_statuses(planned, rows, {'deadline_reached': reached}, pinning)
    donor_lgen = {b: [int(p['L_gen']) for p in (stage0_obj.get('banks') or {}).get(b, {}).get('prompts', [])]
                  for b in sc.ALL_BANKS}
    short = [{'bank': b, 'index': i, 'L_gen': n} for b, values in donor_lgen.items()
             for i, n in enumerate(values) if n < sc.MIN_DONOR_LGEN]
    obj = {
        'part': sc.PART, 'stage': 'pilot' if pilot else 2, 'design': sc.DESIGN_VERSION,
        'model': MODEL, 'revision': REVISION, 'load': load,
        'attn_implementation': load.get('attn_implementation'), 'device': args.device, 'dtype': 'bfloat16',
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prompts_sha256': PROMPTS_SHA256,
        'target': [[int(l), int(u)] for l, u in TARGET], 'batch_size': 1,
        'generation_batch_histogram': generation_batch_histogram(rows),
        'planned_rows': len(planned), 'new_rows_this_invocation': len(rows) - len(done),
        'foreign_shape_rows_ignored': foreign_shape_rows,
        'wall_time_seconds': time.monotonic() - started,
        'deadline_min': args.deadline_min, 'deadline_reached': reached,
        'timing_only_calls': len(timing), 'timing_only': timing,
        'arm_status': status['arms'], 'planned_rows_by_arm': status['planned'],
        'completed_rows_by_arm': status['done'], 'incomplete_arms': status['incomplete_arms'],
        'donors': {b: [{'index': i, 'L_gen': n} for i, n in enumerate(values)] for b, values in donor_lgen.items()},
        'gate3b_pass': not short and any(donor_lgen.values()),
        'gate3b_min_L_gen': min((n for values in donor_lgen.values() for n in values), default=None),
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
    """Read the append-only JSONL, repairing a torn final line first (Part 212's fix 30db7048)."""
    path = Path(path)
    repair_partial_jsonl(path)
    rows = []
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path} line {number}: expected a JSON object')
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 214 runner')
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('stage0', 'stage1', 'stage2', 'pilot'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out', required=True)
        cmd.add_argument('--stage0', default=None, help='results/part214/stage0 (source for stage1/stage2)')
        cmd.add_argument('--device', default='xpu')
        cmd.add_argument('--deadline-min', type=float, default=DEFAULT_DEADLINE_MIN,
                         help='wall-clock budget; at the deadline the runner stops launching calls (design line 357)')
        cmd.add_argument('--allow-unregistered', action='store_true',
                         help='dry run only: skip the PREDICTIONS.md registration check (never for a scored run)')
    args = parser.parse_args(argv)
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
