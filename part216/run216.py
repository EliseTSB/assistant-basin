#!/usr/bin/env python3
"""Part 216 runner: one process, one model load, 456 teacher-forced forwards.

`part216/DESIGN.md` v2.2 is the contract. Modes:

    selftest   the two offline G1 fixtures from DESIGN.md "Gates" (no model, no tokenizer)
    run        G0 arms first, then the rest; writes forwards.jsonl, arrays.npz and run.json

Reads `part216/prompts216.json` (built by `build_prompts216.py`) and Part 214's Stage 1 receipt
(`results/part214/stage1/stage1.json`) for the write-norm pools, the dial pinning and the G0 anchor.

What is captured at the read position `T+4` (the last opening token), per DESIGN.md "Measurements":
  * the residual at every add boundary -- block input (pre-hook on the decoder layer), the input to
    `post_attention_layernorm` (after the mixer add), and the decoder layer output (after the MLP
    add) -- plus the raw mixer and MLP outputs;
  * the two realized increments per block are the FP32 differences of those captured bf16 tensors,
    and the rounding residual is `realized - raw`;
  * the lens `h_0 .. h_n`: the model's final RMS norm with each `h_l`'s OWN RMS and the tied
    unembedding, in FP32;
  * the exact DLA with the final RMS frozen at `h_32`'s value, in FP32 on the CPU;
  * the down_proj input at `T+4` for every unit of the target/monitor layers, so the scorer can
    build the same-slot write-size null.

Refuses to run: a G1 failure stops before the next forward, a G0 failure stops before any non-anchor
forward. Both write a STOP record and no further rows. A `--out` directory holding any run file is
refused outright: there is NO resume, only fresh directories (Alden #12 #2).

Run under the XPU python as
    python -I -B part216/run216.py selftest
    python -I -B part216/run216.py run --out results/part216/run --device xpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
# run214 lives in part214/, so that directory goes on the path too (it imports score214 itself).
for _p in (ROOT, HERE, ROOT / 'part214'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import neuron_report as nr  # noqa: E402
import resolve216 as res  # noqa: E402  (torch-free: the 456-row plan and the prompts-bank lookup)
import run214  # noqa: E402  (MODEL, REVISION, TARGET, load_rig, kernel_check, install_stage1_hooks, ...)
import score216 as sc  # noqa: E402  (numpy only: bf16 emulation, the gates, the outcome rules)
from resolve216 import opener_index, resolve_row, row_key  # noqa: E402,F401  (run216's surface)

MODEL = run214.MODEL
REVISION = run214.REVISION
TARGET = list(run214.TARGET)
MONITORS = [(24, 3949), (27, 3325)]
GOOGLE_ID, AL_ID, T_ID = run214.GOOGLE_ID, run214.AL_ID, run214.T_ID
TOKEN_IDS = [GOOGLE_ID, AL_ID, T_ID]
OPENING_TOKENS = run214.OPENING_TOKENS
N_BLOCKS_PINNED = 32
N_QUESTIONS = 24

PROMPTS_PATH = HERE / 'prompts216.json'
DESIGN_PATH = HERE / 'DESIGN.md'
STAGE1_PATH = ROOT / 'results' / 'part214' / 'stage1' / 'stage1.json'
SELFTEST_PATH = HERE / 'selftest216.json'

# The layers whose per-unit down_proj input the scorer needs: the four target layers (for the
# 200-nearest-write-size null) and the two monitor layers (their unit DLA is reported).
UNIT_LAYERS = sorted({int(l) for l, _u in TARGET} | {int(l) for l, _u in MONITORS})
UNITS = sorted({(int(l), int(u)) for l, u in list(TARGET) + list(MONITORS)})

UNEMBED_CHUNK = 16384


# ----------------------------------------------------------------------------------------------------------------
# small io
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
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f'object of type {type(obj).__name__} is not JSON serializable')


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=json_default) + '\n',
                   encoding='utf-8')
    os.replace(tmp, path)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank(path=PROMPTS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ----------------------------------------------------------------------------------------------------------------
# capture: every residual add boundary and the raw module outputs, at ONE position
# ----------------------------------------------------------------------------------------------------------------

def _first(args, kwargs, name='hidden_states'):
    """The first tensor argument, positional or keyword.

    Qwen3.5's `Qwen3_5DecoderLayer.forward` calls `self_attn(hidden_states=..., ...)` BY KEYWORD, so a
    plain forward hook on the mixer sees empty positional args; every hook here is registered with
    `with_kwargs=True` (or reads only the output) for that reason.
    """
    if args:
        return args[0]
    for key in (name, 'input', 'x'):
        if key in kwargs:
            return kwargs[key]
    raise RuntimeError(f'no tensor found in args={len(args)} kwargs={sorted(kwargs)}')


def install_capture(model, layers, position: int, unit_layers):
    """Pre/post hooks that record, at `position` only, the add boundaries and raw module outputs.

    Returns (store, handles). Every stored tensor stays on the device until `collect_capture` moves
    it -- there is no `.item()` and no per-value CPU copy inside a hook.
    """
    store = {'residual': {}, 'raw': {}, 'units': {}, 'count': {}}

    def bump(name):
        store['count'][name] = int(store['count'].get(name, 0)) + 1

    def take(kind, name, x):
        store[kind][name] = x.detach()[0, position, :].clone()
        if kind == 'residual':
            # The counts dict is the boundary ledger the scorer checks: bare names only, so a raw
            # module output or a unit capture can never be mistaken for an add boundary.
            bump(name)

    handles = [model.get_input_embeddings().register_forward_hook(
        lambda _m, _a, out: take('residual', 'h0', out))]

    for li, layer in enumerate(layers):
        def layer_pre(_m, args, kwargs, li=li):
            take('residual', f'L{li}:in', _first(args, kwargs))

        def pan_pre(_m, args, kwargs, li=li):
            take('residual', f'L{li}:add1', _first(args, kwargs))

        def mixer_hook(_m, _a, out, li=li):
            take('raw', f'L{li}:mixer', out[0] if isinstance(out, tuple) else out)

        def mlp_hook(_m, _a, out, li=li):
            take('raw', f'L{li}:mlp', out[0] if isinstance(out, tuple) else out)

        def layer_post(_m, _a, out, li=li):
            take('residual', f'L{li}:add2', out[0] if isinstance(out, tuple) else out)

        handles.append(layer.register_forward_pre_hook(layer_pre, with_kwargs=True))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pan_pre,
                                                                              with_kwargs=True))
        mixer = layer.linear_attn if getattr(layer, 'block_type', '') == 'linear_attention' \
            else layer.self_attn
        handles.append(mixer.register_forward_hook(mixer_hook))
        handles.append(layer.mlp.register_forward_hook(mlp_hook))
        handles.append(layer.register_forward_hook(layer_post))

    for li in unit_layers:
        def dp_pre(_m, args, kwargs, li=li):
            store['units'][li] = _first(args, kwargs, name='input').detach()[0, position, :].clone()
            store.setdefault('unit_count', {})
            store['unit_count'][li] = int(store['unit_count'].get(li, 0)) + 1

        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(dp_pre, with_kwargs=True))
    return store, handles


def boundary_report(store: dict, n_blocks: int, position: int) -> dict:
    """All 2*n_blocks add boundaries present exactly once per forward, plus the block-input chain."""
    missing, duplicated = [], []
    for li in range(n_blocks):
        for kind in ('add1', 'add2'):
            n = int(store['count'].get(f'L{li}:{kind}', 0))
            if n == 0:
                missing.append(f'L{li}:{kind}')
            elif n > 1:
                duplicated.append(f'L{li}:{kind}')
    if int(store['count'].get('h0', 0)) != 1:
        missing.append('h0')
    if int(store['count'].get('L0:in', 0)) != 1:
        missing.append('L0:in')
    chain = []
    if not missing and store.get('residual'):
        chain.append(float((store['residual']['L0:in'].float()
                            - store['residual']['h0'].float()).abs().max()))
        for li in range(1, n_blocks):
            chain.append(float((store['residual'][f'L{li}:in'].float()
                                - store['residual'][f'L{li - 1}:add2'].float()).abs().max()))
    return {'missing': missing, 'duplicated': duplicated,
            'n_present': 2 * n_blocks - len(missing),
            'chain_max_abs_delta': max(chain) if chain else None}


def collect_capture(store: dict, n_blocks: int) -> dict:
    """Move the captured boundaries to the CPU as float32, once per forward."""
    def cpu(name):
        return store['residual'][name].detach().float().cpu().numpy().astype(np.float32)

    h0 = cpu('h0')
    ins = [cpu(f'L{li}:in') for li in range(n_blocks)]
    add1 = [cpu(f'L{li}:add1') for li in range(n_blocks)]
    add2 = [cpu(f'L{li}:add2') for li in range(n_blocks)]
    raw1 = [store['raw'][f'L{li}:mixer'].detach().float().cpu().numpy().astype(np.float32)
            for li in range(n_blocks)]
    raw2 = [store['raw'][f'L{li}:mlp'].detach().float().cpu().numpy().astype(np.float32)
            for li in range(n_blocks)]
    return {'h0': h0, 'in': ins, 'add1': add1, 'add2': add2, 'raw1': raw1, 'raw2': raw2}


# ----------------------------------------------------------------------------------------------------------------
# the lens and the exact DLA
# ----------------------------------------------------------------------------------------------------------------

def prepare_tables(model, layers, norm) -> dict:
    """The frozen unembedding rows, the norm weight, and the target-layer unit tables (CPU, FP32).

    `proj[layer][u, k] = sum_i W_down[i, u] * (1 + w_i) * W_U[token_k, i]`, so a unit's contribution
    to token k is `scale * a_u * proj[layer][u, k]` once the frozen RMS scale is known. The columns
    are fixed weights, so this is computed once and carried beside the per-forward activations.
    """
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        raise SystemExit('model has no output embedding; the lens needs the tied unembedding')
    w = norm.weight.detach().float().cpu().numpy().astype(np.float32)
    u = out_emb.weight.detach()[TOKEN_IDS].float().cpu().numpy().astype(np.float32)
    a = (np.float32(1.0) + w)[None, :] * u
    eps = float(getattr(norm, 'eps', getattr(model.config, 'rms_norm_eps', 1e-6)))
    projs, colnorms = [], []
    for layer in UNIT_LAYERS:
        wd = layers[layer].mlp.down_proj.weight.detach().float().cpu().numpy().astype(np.float32)
        if wd.shape[0] != w.shape[0]:
            raise SystemExit(f'layer {layer}: down_proj is {wd.shape}, hidden is {w.shape[0]}')
        colnorms.append(np.linalg.norm(wd, axis=0).astype(np.float32))
        projs.append(np.ascontiguousarray(wd.T @ a.T).astype(np.float32))
    return {'A': a, 'norm_weight': w, 'unembed_rows': u, 'eps': eps,
            'unit_layers': np.array(UNIT_LAYERS, dtype=np.int64),
            'unit_proj': np.stack(projs), 'unit_colnorm': np.stack(colnorms)}


def lens_rows(rows: list[torch.Tensor], norm, out_emb, tables: dict) -> tuple[list[dict], np.ndarray]:
    """The lens for h_0..h_n: each row with its OWN RMS, the model's `(1 + w)`, the tied unembedding.

    Computed in FP32 on the device. The full vocabulary is needed for the ranks and the factual
    top-1, so it is scanned in chunks of `UNEMBED_CHUNK` rows: the (n_layers, vocab) logits matrix
    would otherwise dominate memory for no reason.
    """
    device = rows[0].device
    stacked = torch.stack([r.float() for r in rows], dim=0)                     # (L, hidden)
    w = torch.as_tensor(tables['norm_weight'], dtype=torch.float32, device=device)
    mean_sq = stacked.pow(2).mean(dim=1, keepdim=True)
    scale = torch.rsqrt(mean_sq + tables['eps'])
    normed = stacked * scale * (1.0 + w)[None, :]
    weight = out_emb.weight.detach()
    rank = torch.zeros((stacked.shape[0], 3), dtype=torch.long, device=device)
    argmax = torch.zeros(stacked.shape[0], dtype=torch.long, device=device)
    best = torch.full((stacked.shape[0],), float('-inf'), dtype=torch.float32, device=device)
    vocab = weight.shape[0]
    # Each token's z comes out of the SAME chunk projection that produces the comparison values,
    # so a token compares equal to itself and can never outrank itself. A separate 3-column GEMM
    # could round differently (Alden #10 #4). The token chunk(s) are scanned once and cached; the
    # rank pass reuses those exact tensors.
    token_chunks = sorted({int(t) - int(t) % UNEMBED_CHUNK for t in TOKEN_IDS})
    cached: dict[int, torch.Tensor] = {}
    z_cols: list[torch.Tensor | None] = [None] * len(TOKEN_IDS)
    for start in token_chunks:
        stop = min(start + UNEMBED_CHUNK, vocab)
        logits = normed @ weight[start:stop].float().T                          # (L, c)
        cached[start] = logits
        for k, tid in enumerate(TOKEN_IDS):
            if start <= int(tid) < stop:
                z_cols[k] = logits[:, int(tid) - start]
    if any(col is None for col in z_cols):                                      # pragma: no cover
        raise RuntimeError(f'token ids {TOKEN_IDS} are outside the vocabulary of {vocab}')
    z_chunk = torch.stack(z_cols, dim=1)                                        # (L, 3) float32
    for start in range(0, vocab, UNEMBED_CHUNK):
        stop = min(start + UNEMBED_CHUNK, vocab)
        logits = cached[start] if start in cached else normed @ weight[start:stop].float().T
        # 0-based descending rank: how many vocabulary entries outrank each token at this layer.
        rank += (logits[:, None, :] > z_chunk[:, :, None]).sum(dim=2)
        local_best, local_idx = logits.max(dim=1)
        better = local_best > best
        best = torch.where(better, local_best, best)
        argmax = torch.where(better, local_idx + start, argmax)
    z = z_chunk.cpu().numpy().astype(np.float64)                                # (L, 3), CPU only
    rank_np = rank.cpu().numpy()
    argmax_np = argmax.cpu().numpy()
    out = []
    for l in range(z.shape[0]):
        entry = {'layer': int(l), 'z_G': float(z[l, 0]), 'z_Al': float(z[l, 1]),
                 'z_T': float(z[l, 2]),
                 'rank_G': int(rank_np[l, 0]), 'rank_Al': int(rank_np[l, 1]),
                 'rank_T': int(rank_np[l, 2]), 'argmax': int(argmax_np[l])}
        entry['m'] = entry['z_G'] - max(entry['z_Al'], entry['z_T'])
        entry['g_ali'] = entry['z_G'] - float(np.logaddexp(entry['z_Al'], entry['z_T']))
        out.append(entry)
    return out, scale.squeeze(1).cpu().numpy().astype(np.float64)


def dla_block(cap: dict, tables: dict, n_blocks: int, scale_frozen: float) -> dict:
    """The exact DLA on realized increments, FP32 on the CPU (DESIGN.md "Measurements" 2)."""
    a = tables['A'].astype(np.float64)                     # (3, hidden)
    h0 = cap['h0'].astype(np.float64)
    h32 = cap['add2'][-1].astype(np.float64)
    terms, rounding_norm, raw_norm, inc_norm = [], [], [], []
    for li in range(n_blocks):
        before1 = cap['in'][li].astype(np.float64)
        after1 = cap['add1'][li].astype(np.float64)
        after2 = cap['add2'][li].astype(np.float64)
        raw1 = cap['raw1'][li].astype(np.float64)
        raw2 = cap['raw2'][li].astype(np.float64)
        inc1 = after1 - before1
        inc2 = after2 - after1
        terms.append((a @ inc1).tolist())
        terms.append((a @ inc2).tolist())
        rounding_norm.append(float(np.linalg.norm(inc1 - raw1)))
        rounding_norm.append(float(np.linalg.norm(inc2 - raw2)))
        raw_norm.append(float(np.linalg.norm(raw1)))
        raw_norm.append(float(np.linalg.norm(raw2)))
        inc_norm.append(float(np.linalg.norm(inc1)))
        inc_norm.append(float(np.linalg.norm(inc2)))
    emb = (a @ h0).tolist()
    fp32_logits = (scale_frozen * (a @ h32)).tolist()
    terms = (scale_frozen * np.asarray(terms, dtype=np.float64)).tolist()
    emb = (scale_frozen * np.asarray(emb, dtype=np.float64)).tolist()
    return {'emb_term': emb, 'terms': terms, 'fp32_logits': fp32_logits,
            'frozen_scale': float(scale_frozen), 'rounding_norm': rounding_norm,
            'raw_norm': raw_norm, 'increment_norm': inc_norm}


def unit_dla_scalars(cap_units: dict, tables: dict, scale_frozen: float) -> dict:
    """Per-unit c_G / c_Al / c_T and write size for the target and monitor units."""
    out = {}
    slot = {int(l): k for k, l in enumerate(tables['unit_layers'])}
    for layer, unit in UNITS:
        k = slot[int(layer)]
        act = cap_units[int(layer)].astype(np.float64)
        col_norm = tables['unit_colnorm'][k][int(unit)]
        proj = tables['unit_proj'][k][int(unit)]
        a = float(act[int(unit)])
        c = (scale_frozen * a * proj.astype(np.float64)).tolist()
        out[f'{int(layer)}:{int(unit)}'] = {
            'activation': a, 'write_size': abs(a) * float(col_norm),
            'c_G': c[0], 'c_Al': c[1], 'c_T': c[2],
            'normalized_c_G': (c[0] / (abs(a) * float(col_norm))) if abs(a) * float(col_norm) else None}
    return out


# ----------------------------------------------------------------------------------------------------------------
# one forward
# ----------------------------------------------------------------------------------------------------------------

def forward_row(model, layers, norm, out_emb, tables, seq: list[int], read_position: int,
                prompt_ids: list[int], edit_units=None, edit_factor=None, capture_units=False):
    """One teacher-forced forward and every readout the design asks for at `read_position`."""
    n_blocks = len(layers)
    handles = []
    if edit_units:
        factor = None if edit_factor is None else np.asarray(edit_factor, dtype=np.float64)
        handles += run214.install_stage1_hooks(layers, edit_units, None, factor, len(prompt_ids))
    store, cap_handles = install_capture(model, layers, read_position,
                                         UNIT_LAYERS if capture_units else [])
    handles += cap_handles
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq], dtype=torch.long, device=device),
                        use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    boundary = boundary_report(store, n_blocks, read_position)

    rows = [store['residual']['h0']] + [store['residual'][f'L{li}:add2'] for li in range(n_blocks)]
    lens, scales = lens_rows(rows, norm, out_emb, tables)
    scale_frozen = float(scales[-1])

    cap = collect_capture(store, n_blocks)
    dla = dla_block(cap, tables, n_blocks, scale_frozen)

    native = out.logits[0, read_position, :].float().cpu()
    native_tokens = [float(native[i]) for i in TOKEN_IDS]
    native_g_ali = native_tokens[0] - float(np.logaddexp(native_tokens[1], native_tokens[2]))
    native_g_al = native_tokens[0] - native_tokens[1]

    # Every capture boundary, not just h0/add2 (Alden #10 #4): the block input and both residual
    # adds have to be observed, and the raw module outputs are recorded beside them so a rounding
    # change in either half is attributable.
    boundary_names = ['h0']
    for li in range(n_blocks):
        boundary_names += [f'L{li}:in', f'L{li}:add1', f'L{li}:add2']
    dtypes = {name: str(store['residual'][name].dtype) for name in boundary_names}
    for li in range(n_blocks):
        dtypes[f'raw:L{li}:mixer'] = str(store['raw'][f'L{li}:mixer'].dtype)
        dtypes[f'raw:L{li}:mlp'] = str(store['raw'][f'L{li}:mlp'].dtype)
    record = {
        'read_position': int(read_position),
        'native_logits': native_tokens, 'native_g_ali': native_g_ali, 'native_g_al': native_g_al,
        'lens': lens, 'dla': dla,
        'boundary_counts': dict(store['count']),
        'boundary_report': boundary,
        'dtypes': dtypes,
        'n_blocks': n_blocks,
        # The exact tokens and read position this forward used. Written into the row so a plan row
        # can be checked against what the model actually saw (Alden #10 #1, #3).
        'sequence_token_ids': [int(t) for t in seq],
        'prompt_token_ids': [int(t) for t in prompt_ids],
        'T': len(prompt_ids),
        # One definition of the hash, shared with the scorer's binding (resolve216.input_sha256).
        'input_sha256': res.input_sha256(seq),
    }
    if capture_units:
        unit_np = {int(l): store['units'][int(l)].float().cpu().numpy().astype(np.float32)
                   for l in tables['unit_layers']}
        record['unit_dla'] = unit_dla_scalars(unit_np, tables, scale_frozen)
    arrays = {'h0': cap['h0'], 'h32': cap['add2'][-1]}
    if capture_units:
        arrays['units'] = np.stack([unit_np[int(l)] for l in tables['unit_layers']])
    return record, arrays


# ----------------------------------------------------------------------------------------------------------------
# the dtype contract (DESIGN.md "Model": bf16; Alden #10 #4)
# ----------------------------------------------------------------------------------------------------------------

def expected_capture_dtype(load: dict) -> str | None:
    """The dtype every capture boundary must show, from the load record, or None if not bf16.

    `nr.load_record` writes `'torch.bfloat16'`; a test stub may write `'bfloat16'`.  Both mean the
    run claims bf16, and on such a run a float32 capture is a broken fixture, not a result.
    """
    if not isinstance(load, dict):
        return None
    dtype = str(load.get('dtype', ''))
    if dtype.endswith('bfloat16') or dtype == 'bf16':
        return 'torch.bfloat16'
    return None


def dtype_contract(record: dict, load: dict) -> dict:
    """Every recorded capture boundary against the dtype the load claims (STOP on a mismatch)."""
    want = expected_capture_dtype(load)
    observed = sorted(set((record.get('dtypes') or {}).values()))
    if want is None:
        return {'pass': True, 'expected': None, 'observed': observed,
                'note': 'the load does not claim bf16, so no capture dtype is enforced'}
    dtypes = record.get('dtypes') or {}
    required = ['h0'] + [f'L{li}:{kind}' for li in range(int(record.get('n_blocks', 0)))
                         for kind in ('in', 'add1', 'add2')]
    missing = [name for name in required if name not in dtypes]
    bad = sorted(name for name, dt in dtypes.items() if dt != want)
    return {'pass': not missing and not bad, 'expected': want, 'observed': observed,
            'missing': missing[:20], 'bad': bad[:20], 'n_bad': len(bad)}


# ----------------------------------------------------------------------------------------------------------------
# G1 offline fixtures (DESIGN.md "Gates" G1)
# ----------------------------------------------------------------------------------------------------------------

def selftest() -> dict:
    """The two offline G1 fixtures, before any model is loaded.

    1. The analytic bf16 case: residual 1, write 1/256. bf16 keeps 8 significand bits, so the ulp at
       1.0 is 2**-7 and 1 + 2**-8 is an exact tie -- round-half-to-even sends it back to 1.0. The
       realized increment must be 0 and the rounding residual -1/256.
    2. A dropped-increment control: a decomposition with one increment removed must FAIL G1.
    """
    checks = {}

    one = np.float32(1.0)
    write = np.float32(1.0 / 256.0)
    after = sc.bf16_add(one, write)
    inc = sc.realized_increment(np.array([one]), np.array([after]), np.array([write]))
    realized = float(inc['increment'][0])
    rounding = float(inc['rounding'][0])
    checks['analytic_bf16'] = {
        'before': float(one), 'write': float(write), 'after': float(after),
        'realized_increment': realized, 'rounding_residual': rounding,
        'expected_increment': 0.0, 'expected_rounding': -1.0 / 256.0,
        'pass': realized == 0.0 and abs(rounding + 1.0 / 256.0) < 1e-12,
        'bf16_emulation_matches_ulp': float(after) == 1.0,
    }

    torch_bf16 = {}
    try:
        r = torch.tensor(1.0, dtype=torch.bfloat16)
        w = torch.tensor(1.0 / 256.0, dtype=torch.bfloat16)
        s = r + w
        torch_bf16 = {'sum': float(s), 'is_one': float(s) == 1.0,
                      'dtype': str(s.dtype)}
    except Exception as exc:                                   # pragma: no cover - torch always present
        torch_bf16 = {'error': repr(exc)}
    checks['analytic_bf16']['torch_cross_check'] = torch_bf16
    if torch_bf16.get('is_one') is False:
        checks['analytic_bf16']['pass'] = False

    emb = np.array([0.5, -0.25, 1.0])
    terms = np.array([[0.1, 0.2, -0.3], [0.05, -0.05, 0.15]])
    logits = emb + terms.sum(axis=0)
    good = sc.gate1_check(emb, terms, logits)
    dropped = sc.gate1_check(emb, terms[:1], logits)
    checks['dropped_increment_control'] = {
        'intact_pass': good['pass'], 'dropped_pass': dropped['pass'],
        'dropped_max_abs_error': dropped['max_abs_error'],
        'pass': good['pass'] and not dropped['pass'],
    }
    passed = all(v['pass'] for v in checks.values())
    return {'mode': 'selftest', 'pass': passed, 'checks': checks,
            'tool_sha256': file_sha256(Path(__file__))}


# ----------------------------------------------------------------------------------------------------------------
# plan and run
# ----------------------------------------------------------------------------------------------------------------

def plan_rows(bank: dict, g0_arms=None) -> list[dict]:
    """456 rows: the 13 G0 anchor arms first, then the rest, then the controls, then the factuals.

    The resolution rules themselves live in `resolve216.py` so the scorer can rebuild the same
    lookup without importing torch (Alden #14); this wrapper keeps the runner's one-argument call.
    """
    return res.plan_rows(bank, sc.G0_ARMS if g0_arms is None else g0_arms)


def load_receipt(path=STAGE1_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arm_random_units(receipt: dict, arm: str) -> list[tuple[int, int]]:
    """`random_k:zero` units rebuilt with run214.draw_random_set from the recorded pools."""
    pools = {int(l): [int(u) for u in us] for l, us in receipt['write_norm_pools'].items()}
    k = int(arm.split('_', 1)[1].split(':', 1)[0])
    return [(int(l), int(u)) for l, u in run214.draw_random_set(pools, k)]


def edit_for_arm(receipt: dict, arm: str, prompt_len: int):
    """(units, factor) for an edit arm, or (None, None).  `zero` is alpha 0.0, Part 214's rung."""
    if arm == 'unchanged':
        return None, None
    if arm == 'zero':
        units = [tuple(int(x) for x in u) for u in TARGET]
        return units, np.zeros((prompt_len + OPENING_TOKENS, len(units)), dtype=np.float64)
    if arm.startswith('random_') and arm.endswith(':zero'):
        units = arm_random_units(receipt, arm)
        return units, np.zeros((prompt_len + OPENING_TOKENS, len(units)), dtype=np.float64)
    if arm == 'scale_dial_fact':
        units = [tuple(int(x) for x in u) for u in TARGET]
        dial = run214._dial_factors(receipt['dial_pinning'], len(units))
        # `rung_alpha` empty: this arm is the two-phase dial, not one of the fixed rungs.
        ctx = {'T_q': prompt_len, 'K': len(units), 'dial': dial, 'rung_alpha': {}}
        return units, run214.stage1_factors('scale_dial_fact', 0, ctx)
    if arm in ('op_filler', 'op_denial', 'op_vogel'):
        return None, None
    raise ValueError(f'unknown arm {arm!r}')


def filter_plan_questions(plan: list[dict], limit_questions) -> list[dict]:
    """The `--limit-questions` test-only reduction: keep question indices `< limit`."""
    n = int(limit_questions or 0)
    if n <= 0:
        return list(plan)
    return [r for r in plan if int(r['question_index']) < n]


# Every file a previous attempt (or a previous scorer) leaves in an --out directory. This runner has
# NO resume: a directory holding any of these is refused, so rows from two attempts can never be
# mixed and a JSONL that ran ahead of arrays.npz can never be "continued" (Alden #12 #2).
RUN_FILE_NAMES = ('forwards.jsonl', 'run.json', 'arrays.npz', 'selftest.json', 'refused.json')


def existing_run_files(out_dir) -> list[str]:
    """The run files already present in `out_dir` (empty means a fresh directory)."""
    return [name for name in RUN_FILE_NAMES if (Path(out_dir) / name).exists()]


def run(args) -> dict:
    started = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    forwards_path = out_dir / 'forwards.jsonl'
    arrays_path = out_dir / 'arrays.npz'
    run_json_path = out_dir / 'run.json'

    # Fresh directories only (Alden #12 #2). There is no resume: an interrupted run leaves
    # forwards.jsonl ahead of arrays.npz (written only on exit), a STOP is terminal, and a directory
    # carrying rows from an earlier attempt must never be mixed with a new one. The refusal happens
    # before the model is loaded and before anything is written except this refusal record.
    existing = existing_run_files(out_dir)
    if existing:
        try:
            n_rows = len(_read_rows(forwards_path)) if forwards_path.exists() else 0
        except (OSError, ValueError, KeyError):
            n_rows = None                       # the file is unreadable; that is why it is refused
        refusal = {'mode': 'run', 'part': 216, 'stage': 'STOP', 'refused': True,
                   'refusal_reason': ('this output directory already holds run files; this runner '
                                      'has no resume -- a new attempt needs a new, empty --out'),
                   'existing_files': existing, 'n_rows': n_rows,
                   'refused_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        write_json(out_dir / 'refused.json', refusal)
        return refusal

    bank = load_bank(args.prompts)
    receipt = load_receipt(args.stage1)
    design_sha = file_sha256(DESIGN_PATH) if DESIGN_PATH.exists() else None
    prompts_sha = file_sha256(args.prompts)

    fixtures = selftest()
    write_json(out_dir / 'selftest.json', fixtures)
    if not fixtures['pass']:
        record = {'mode': 'run', 'stage': 'STOP', 'reason': 'G1 offline fixtures failed',
                  'selftest': fixtures}
        write_json(run_json_path, record)
        return record

    model, tok, load = run214.load_rig(args)
    layers = nr.find_text_layers(model)[1]
    _name, _layers, norm = nr.find_text_layers(model)
    if norm is None:
        raise SystemExit('no final norm found next to the decoder layers')
    out_emb = model.get_output_embeddings()
    n_blocks = len(layers)
    if args.require_pinned_shape and n_blocks != N_BLOCKS_PINNED:
        raise SystemExit(f'model has {n_blocks} blocks, the design pins {N_BLOCKS_PINNED}')
    tables = prepare_tables(model, layers, norm)

    full_plan = plan_rows(bank)
    plan = filter_plan_questions(full_plan, getattr(args, 'limit_questions', 0))
    if getattr(args, 'limit', 0):
        plan = plan[:int(args.limit)]
    partial = bool(int(getattr(args, 'limit_questions', 0) or 0) or int(getattr(args, 'limit', 0) or 0))

    run_info = {
        'mode': 'run', 'part': 216, 'design_version': 'v2.2',
        'model': MODEL, 'revision': REVISION,
        'design_sha256': design_sha, 'prompts_sha256': prompts_sha,
        'prompts_path': str(Path(args.prompts).resolve()),
        'tool_sha256': file_sha256(Path(__file__)),
        'score_tool_sha256': file_sha256(Path(sc.__file__)),
        'stage1_receipt_path': str(Path(args.stage1).resolve()),
        'stage1_receipt_sha256': file_sha256(args.stage1),
        'kernel_check': run214.kernel_check(load),
        'load': load, 'device': str(args.device),
        'n_blocks': n_blocks, 'hidden_size': int(model.config.hidden_size),
        'unit_layers': UNIT_LAYERS, 'units': [f'{l}:{u}' for l, u in UNITS],
        'arms': bank['arms'], 'control_arms': bank['control_arms'],
        'torch': getattr(torch, '__version__', None),
        'transformers': _transformers_version(),
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_planned_full': len(full_plan), 'n_planned': len(plan),
        'limit': int(getattr(args, 'limit', 0) or 0),
        'limit_questions': int(getattr(args, 'limit_questions', 0) or 0),
        'partial': partial,
        # There is no resume: a run is one pass into a fresh directory (Alden #12 #2).
        'resume_supported': False,
    }
    if not run_info['kernel_check'].get('pass'):
        run_info['stage'] = 'STOP'
        run_info['stop_reason'] = 'kernel check failed'
        write_json(run_json_path, run_info)
        return run_info

    arrays: dict = {}

    # A diagnostic record, not a resume point (there is no resume): an interrupted directory says
    # `RUNNING` and must be discarded, never continued. arrays.npz is written only on exit, so a
    # JSONL left in flight is always ahead of it.
    write_json(run_json_path, {**run_info, 'stage': 'RUNNING', 'n_rows': 0})

    rows: dict = {}
    n_done = 0
    stops: list[dict] = []
    for row in plan:
        key = row_key(row)
        entry = resolve_row(bank, row)
        if row['kind'] == 'factual':
            seq = list(entry['prompt_token_ids'])
            read_position = int(entry['read_position'])
            prompt_ids = seq
            record, row_arrays = forward_row(model, layers, norm, out_emb, tables, seq,
                                             read_position, prompt_ids, capture_units=False)
            record['input_token_id'] = int(entry['input_token_id'])
            record['expected_answer'] = entry['expected_answer']
        else:
            seq = list(entry['sequence_token_ids'])
            prompt_ids = list(entry['prompt_token_ids'])
            read_position = int(entry['read_position'])
            # Controls are unedited forwards (the sentence lives in the prompt); only stage1 arms
            # go through `edit_for_arm`. Before this, every control row raised
            # `ValueError: unknown arm 'control_google'` and the run never reached the factuals.
            if row['kind'] == 'control':
                units, factor = None, None
            else:
                units, factor = edit_for_arm(receipt, row['arm'], len(prompt_ids))
            record, row_arrays = forward_row(model, layers, norm, out_emb, tables, seq,
                                             read_position, prompt_ids, edit_units=units,
                                             edit_factor=factor,
                                             capture_units=(row['kind'] == 'stage1'))
        record.update({'key': key, 'kind': row['kind'], 'arm': row['arm'],
                       'question_index': row['question_index']})

        # ----- every gate runs BEFORE the row is appended (Alden #10 #3) -----------------------
        failures: list[tuple[str, dict]] = []
        bcheck = sc.boundary_check(record, n_blocks)
        if not bcheck['pass']:
            failures.append(('boundary count', bcheck))
        gcheck = sc.gate1_check(record['dla']['emb_term'], record['dla']['terms'],
                                record['dla']['fp32_logits'])
        if not gcheck['pass']:
            failures.append(('G1 telescoping', gcheck))
        finite = sc.row_finite(record)
        if not finite['pass']:
            failures.append(('nonfinite row', finite))
        dcheck = dtype_contract(record, load)
        if not dcheck['pass']:
            failures.append(('dtype contract', dcheck))
        # Every anchor row is checked against the Part 214 receipt before it is appended. This is
        # skipped on a declared-partial smoke run (`--limit`/`--limit-questions`), where the tiny
        # fixture cannot match the 4B receipt and no design-level claim is made anyway; a full run
        # always checks it. The 312-value G0 gate below is the second, complete check.
        if not failures and not partial and row['kind'] == 'stage1' and \
                row['arm'] in sc.G0_ARMS:
            ref = ((receipt.get('arms') or {}).get(row['arm']) or {}).get('g_ali')
            got = record.get('native_g_ali')
            if ref is None or row['question_index'] >= len(ref):
                failures.append(('G0', {'key': key, 'reason': 'anchor arm has no receipt value'}))
            elif got is None or not np.isfinite(float(got)) or \
                    abs(float(got) - float(ref[row['question_index']])) > sc.G0_TOLERANCE:
                failures.append(('G0', {'key': key, 'got': got,
                                        'receipt': float(ref[row['question_index']]),
                                        'tolerance': sc.G0_TOLERANCE}))
        # G0 is evaluated as soon as the last anchor row lands, before any non-anchor forward.
        if not failures and row['kind'] == 'stage1' and row['arm'] == sc.G0_ARMS[-1] and \
                row['question_index'] == N_QUESTIONS - 1:
            candidate = dict(rows)
            candidate[key] = record
            g0 = sc.gate0(candidate, receipt)
            if not g0['pass']:
                failures.append(('G0', {'max_abs_delta': g0['max_abs_delta'],
                                        'n_failures': g0['n_failures'],
                                        'n_values': g0['n_values'],
                                        'failures': g0['failures'][:5]}))
        if failures:
            reason, detail = failures[0]
            stops.append({'key': key, 'reason': reason, 'detail': detail, 'row': record})
            break

        # ----- committed only now: the row passed every check --------------------------------
        for name, value in row_arrays.items():
            arrays[f'{name}:{key}'] = value
        arrays[f'scale:{key}'] = np.float64(record['dla']['frozen_scale'])
        with forwards_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=json_default) + '\n')
        rows[key] = record
        n_done += 1

    # The scorer's unit tables, once per run, and a q-keyed frozen scale per row (Alden #10 #2).
    arrays['unit_layers'] = np.asarray(UNIT_LAYERS, dtype=np.int64)
    arrays['unit_proj'] = np.asarray(tables['unit_proj'], dtype=np.float32)
    arrays['unit_colnorm'] = np.asarray(tables['unit_colnorm'], dtype=np.float32)
    np.savez_compressed(arrays_path, **arrays)
    stage = 'STOP' if stops else 'DONE'
    complete = (not partial) and len(rows) >= len(full_plan) and not stops
    run_info.update({
        'n_rows': len(rows), 'n_new_rows': n_done, 'n_planned': len(plan),
        'n_planned_full': len(full_plan),
        'stage': stage, 'stops': stops,
        'partial': bool(partial or not complete),
        'run_status': 'full' if complete else 'partial',
        'wall_time_seconds': time.time() - started,
        'finished': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })
    if rows:
        sample = next(iter(rows.values()))
        run_info['boundary_dtypes'] = sample.get('dtypes')
        run_info['dtype_contract'] = dtype_contract(sample, load)
        run_info['boundary_count_example'] = {k: v for k, v in sample['boundary_counts'].items()
                                              if ':add' in k}
    write_json(run_json_path, run_info)
    return run_info


def _read_rows(path) -> dict:
    rows = {}
    if Path(path).exists():
        with Path(path).open(encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rows[rec['key']] = rec
    return rows


def _transformers_version():
    try:
        import transformers
        return getattr(transformers, '__version__', None)
    except Exception:                                          # pragma: no cover
        return None


# ----------------------------------------------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Part 216 runner (lens + exact DLA on Qwen3.5-4B).')
    ap.add_argument('mode', choices=('selftest', 'run'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'part216' / 'run'))
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--prompts', default=str(PROMPTS_PATH))
    ap.add_argument('--stage1', default=str(STAGE1_PATH))
    ap.add_argument('--limit', type=int, default=0, help='stop after N rows (smoke tests)')
    ap.add_argument('--limit-questions', type=int, default=0,
                    help='test-only: keep question indices below N; the run is marked partial '
                         'in run.json and can never produce a full score')
    ap.add_argument('--require-pinned-shape', action='store_true')
    args = ap.parse_args(argv)
    if args.mode == 'selftest':
        result = selftest()
        write_json(SELFTEST_PATH, result)
        print(json.dumps(result, indent=1, default=json_default))
        return 0 if result['pass'] else 1
    result = run(args)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ('stage', 'n_rows', 'n_planned', 'stops', 'wall_time_seconds')},
                     indent=1, default=json_default))
    return 0 if result.get('stage') != 'STOP' else 1


if __name__ == '__main__':
    raise SystemExit(main())
