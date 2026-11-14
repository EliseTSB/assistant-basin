#!/usr/bin/env python3
"""record211.py: one checkpoint of PART211-DESIGN.md, one text set, in float32 with eager attention (design: Method,
"What is recorded").

Per checkpoint, per document, for every layer L and head h:
  att[L][h]        the head's attention rows, [256, 256] float32 (the softmax rows; causal, so row q attends to keys
                   0..q only). Reference checkpoints only (design: Storage and memory).
  z[L][h]          the head's attention output before its output block at positions 3, 6, ..., 255, [85, head_dim]
                   float32: its contribution to the residual stream is z times the head's block of the output
                   projection. Reference checkpoints only; every other checkpoint is reduced to O and V against its
                   run's final (design: contributions are reduced to point values and not kept).
  mlp[L]           the layer's MLP output at the same positions, [85, d_model] float32. Every checkpoint.
  ov_v, ov_w[L][h] the head's value block and output block, each [head_dim, d_model] float32 (the OV matrix is
                   v^T w; V is exact from the factors). Reference checkpoints only.

The attention rows and the head contributions are captured with hooks inside the attention module, so the split is the
model's own arithmetic: for GPT-NeoX (parallel attention and MLP, query_key_value fused) the head's block of
`dense.weight`; for OLMo 2 the head's block of `o_proj.weight` applied to its own value block. The MLP output is the
MLP module's output (GPT-NeoX: the block's parallel MLP output; OLMo 2: after post_feedforward_layernorm, which is what
enters the residual).

Reduction (design: Storage and memory). A reference checkpoint writes its rows, z and OV factors. Every other
checkpoint is reduced as it is read: its attention rows to P, P0 and the cosine measure against the reference
checkpoints of its own run, and its z and factors to O_final and V_final against the run's final reference; then they
are dropped. Its MLP outputs are kept (M is pooled per resample). References are recorded first, so the final is on disk.
An earlier version kept every head's [85, d_model] contribution, the full OV matrices, and per-document "moments"
duplicating both: about 900 GB against the design's 100 GB (found 2026-11-13, before any run).

Records (design: Records). Each checkpoint's rows go to a temporary file renamed into place when complete; each file
carries the checkpoint's commit hash, the text set, a row-schema version and the code's git hash, and a resumed run
skips a file only when all four match.

  python -I -B part211/record211.py --checkpoint R0_step10000 --texts texts211.json --out results/part211
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import measures211 as MS  # noqa: E402

SCHEMA = 2                                       # 2: no contributions or moments; z and OV factors on references only
N_TOKENS = 256
POSITIONS = np.arange(3, N_TOKENS, 3)          # 3, 6, ..., 255: 85 positions (design: What is recorded)


def positions_for(n_tokens: int) -> np.ndarray:
    """The recorded positions for a document of n_tokens tokens: 3, 6, ..., the last multiple of 3 below n_tokens."""
    return np.arange(3, n_tokens, 3)


# ---------------------------------------------------------------- provenance

def code_hash(root: Path = ROOT) -> str:
    """The code's git hash: HEAD's hash, plus a marker when the tree is dirty (an uncommitted edit changes the code)."""
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "part211"],
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unknown ({exc.__class__.__name__})"
    sha = head.stdout.strip()
    return sha + ("+dirty" if dirty.stdout.strip() else "")


def record_path(out: Path, checkpoint_key: str, text_set: str) -> Path:
    return Path(out) / "records" / f"{checkpoint_key}.{text_set}.npz"


def record_meta(commit: str, text_set: str, code: str) -> dict:
    """The four fields the resume check compares (design: Records). Anything else a caller adds (e.g. `reference`) is
    kept in the record's metadata but is not one of the four."""
    return {"commit": commit, "text_set": text_set, "schema": SCHEMA, "code": code}


def resume_state(path: Path, meta: dict) -> str:
    """'complete' when the record exists and all four fields match; 'incomplete' otherwise (redo it)."""
    path = Path(path)
    if not path.is_file():
        return "incomplete"
    try:
        z = np.load(path, allow_pickle=False)
        stored = {k: str(z[f"meta_{k}"]) for k in ("commit", "text_set", "schema", "code")}
        done = bool(z["complete"])
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return "incomplete"
    want = {k: str(meta[k]) for k in ("commit", "text_set", "schema", "code")}
    return "complete" if done and stored == want else "incomplete"


def save_record(path: Path, arrays: dict, meta: dict) -> Path:
    """Write to a temporary file in the same directory and rename it into place when complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, complete=np.array(True), **{f"meta_{k}": str(v) for k, v in meta.items()}, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- the architecture adapters

class Adapter:
    """The pieces of one architecture this part reads. `capture` installs hooks for one forward and returns the
    captured tensors; `ov` is the head's OV matrix from the weights."""

    family = None

    def __init__(self, model):
        self.model = model

    def n_layers(self) -> int:
        raise NotImplementedError

    def n_heads(self) -> int:
        raise NotImplementedError

    def d_model(self) -> int:
        raise NotImplementedError

    def capture(self, ids) -> dict:
        raise NotImplementedError

    @staticmethod
    def _positions(ids):
        return positions_for(int(ids.shape[1]))

    def head_dim(self) -> int:
        raise NotImplementedError

    def factors(self, layer: int) -> tuple:     # (value blocks, output blocks), each [heads, head_dim, d_model]
        raise NotImplementedError

    def ov(self, layer: int) -> np.ndarray:      # [heads, d_model, d_model]: v^T w per head
        v, w = self.factors(layer)
        return np.einsum("hkd,hke->hde", v, w)

    def o_weight(self, layer: int):              # the output projection, [d_model, d_model]
        raise NotImplementedError


class NeoXAdapter(Adapter):
    """GPT-NeoX (Pythia-160m): parallel attention and MLP, query_key_value fused [3 * heads * head_dim, d_model]."""

    family = "pythia"

    def __init__(self, model):
        super().__init__(model)
        self.body = model.gpt_neox
        self.layers = list(self.body.layers)

    def n_layers(self):
        return len(self.layers)

    def n_heads(self):
        return int(self.layers[0].attention.config.num_attention_heads)

    def d_model(self):
        return int(self.layers[0].attention.query_key_value.weight.shape[1])

    def head_dim(self):
        return int(self.layers[0].attention.head_size)

    def capture(self, ids):
        import torch
        pos = self._positions(ids)
        caps = {"att": [], "contrib": [], "z": [], "mlp": [], "_attn_out": [], "_block_out": [], "_concat": [],
                "positions": pos}
        handles = []

        def attn_hook(mod, args, out):
            # out: (attn_output [B, T, d_model] after dense, attn_weights [B, H, T, T])
            weights = out[1]
            if weights is None:
                raise RuntimeError("attention weights missing: load the model with attn_implementation='eager'")
            caps["att"].append(weights.detach())
            caps["_attn_out"].append(out[0].detach())
            caps["_block_out"].append(out[0].detach())

        def dense_pre_hook(mod, args):
            x = args[0]                                   # [B, T, d_model] = concat of heads' attention outputs
            W = mod.weight                                # [d_model, d_model], rows = output coordinates
            B, T, D = x.shape
            H, hd = self.n_heads(), self.head_dim()
            xh = x.reshape(B, T, H, hd).permute(2, 0, 1, 3)                   # [H, B, T, hd]
            Wh = W.reshape(D, H, hd).permute(1, 2, 0)                         # [H, hd, D]
            contrib = torch.einsum("hbtk,hkd->hbtd", xh, Wh)                  # [H, B, T, D]
            caps["contrib"].append(contrib[:, :, pos].permute(1, 0, 2, 3).detach())   # [B, H, n_pos, D]
            caps["z"].append(xh[:, :, pos].permute(1, 0, 2, 3).detach())             # [B, H, n_pos, hd]
            caps["_concat"].append(x.detach())

        for L, block in enumerate(self.layers):
            handles.append(block.attention.register_forward_hook(attn_hook))
            handles.append(block.attention.dense.register_forward_pre_hook(dense_pre_hook))
            handles.append(block.mlp.register_forward_hook(
                lambda m, a, out: caps["mlp"].append(out[:, pos].detach())))
        try:
            self.model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return caps

    def o_weight(self, layer):
        import torch
        return self.layers[layer].attention.dense.weight.detach().float().T

    def factors(self, layer):
        att = self.layers[layer].attention
        H, hd, D = self.n_heads(), self.head_dim(), self.d_model()
        qkv = att.query_key_value.weight.detach().float().cpu().numpy()      # [3 H hd, D]
        qkv = qkv.reshape(H, 3 * hd, D)
        V = qkv[:, 2 * hd:, :]                                               # [H, hd, D]: value block
        W = att.dense.weight.detach().float().cpu().numpy()                  # [D, D]
        Wh = W.reshape(D, H, hd).transpose(1, 2, 0)                          # [H, hd, D]: output block
        return V, Wh


class Olmo2Adapter(Adapter):
    """OLMo 2 (1B): post-norm blocks, q/k RMS norms, separate q/k/v/o projections."""

    family = "olmo"

    def __init__(self, model):
        super().__init__(model)
        self.body = model.model
        self.layers = list(self.body.layers)

    def n_layers(self):
        return len(self.layers)

    def n_heads(self):
        return int(self.layers[0].self_attn.config.num_attention_heads)

    def d_model(self):
        return int(self.layers[0].self_attn.o_proj.weight.shape[1])

    def head_dim(self):
        return int(self.layers[0].self_attn.head_dim)

    def capture(self, ids):
        import torch
        pos = self._positions(ids)
        caps = {"att": [], "contrib": [], "z": [], "mlp": [], "_attn_out": [], "_block_out": [], "_concat": [],
                "positions": pos}
        handles = []

        def attn_hook(mod, args, out):
            weights = out[1]
            if weights is None:
                raise RuntimeError("attention weights missing: load the model with attn_implementation='eager'")
            caps["att"].append(weights.detach())
            caps["_attn_out"].append(out[0].detach())
            caps["_block_out"].append(out[0].detach())

        def o_pre_hook(mod, args):
            x = args[0]                                   # [B, T, H * head_dim]
            W = mod.weight                                # [d_model, H * head_dim]
            B, T, _ = x.shape
            H, hd = self.n_heads(), self.head_dim()
            xh = x.reshape(B, T, H, hd).permute(2, 0, 1, 3)
            Wh = W.reshape(self.d_model(), H, hd).permute(1, 2, 0)
            contrib = torch.einsum("hbtk,hkd->hbtd", xh, Wh)
            caps["contrib"].append(contrib[:, :, pos].permute(1, 0, 2, 3).detach())
            caps["z"].append(xh[:, :, pos].permute(1, 0, 2, 3).detach())             # [B, H, n_pos, hd]
            caps["_concat"].append(x.detach())

        for L, block in enumerate(self.layers):
            handles.append(block.self_attn.register_forward_hook(attn_hook))
            handles.append(block.self_attn.o_proj.register_forward_pre_hook(o_pre_hook))
            handles.append(block.mlp.register_forward_hook(
                lambda m, a, out: caps["mlp"].append(out[:, pos].detach())))
        try:
            self.model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return caps

    def o_weight(self, layer):
        import torch
        return self.layers[layer].self_attn.o_proj.weight.detach().float().T

    def factors(self, layer):
        att = self.layers[layer].self_attn
        H, hd = self.n_heads(), self.head_dim()
        V = att.v_proj.weight.detach().float().cpu().numpy().reshape(H, hd, self.d_model())
        Wh = att.o_proj.weight.detach().float().cpu().numpy().reshape(self.d_model(), H, hd).transpose(1, 2, 0)
        return V, Wh


def adapter_for(model) -> Adapter:
    name = type(model).__name__
    if hasattr(model, "gpt_neox"):
        return NeoXAdapter(model)
    if hasattr(model, "model") and type(model.model).__name__.startswith("Olmo2"):
        return Olmo2Adapter(model)
    raise SystemExit(f"no adapter for model class {name}: this part reads GPT-NeoX (Pythia) and OLMo 2")


def load_model(repo_id: str, commit: str | None = None, dtype="float32"):
    """A checkpoint in float32 with eager attention, gradients off, on the CPU."""
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(repo_id, revision=commit, dtype=torch.float32,
                                                 attn_implementation="eager")
    model.to("cpu").eval()
    return model


# ---------------------------------------------------------------- one checkpoint, one text set

def pooled_cka(mlp_a: np.ndarray, mlp_b: np.ndarray, idx, layer: int = 0) -> float:
    """Centred linear CKA between two checkpoints' MLP outputs at one layer, pooled over the documents `idx` (a
    resample, repeats allowed), from the stored outputs [n_docs, L, n_pos, D] (design: Storage, "the scorer recomputes
    pooled centred CKA per resample from them"). The pool is re-centred as a whole, so this is exact; per-document
    moments centred within each document are not sufficient and are not stored (an earlier version kept them, with
    the contributions, beside a second copy of every output)."""
    idx = np.asarray(idx)
    a = np.asarray(mlp_a)[idx, layer]
    b = np.asarray(mlp_b)[idx, layer]
    return MS.cka(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]))


def final_reference(checkpoint_key: str) -> str:
    """The run's final checkpoint: its last reference in table order (Pythia step 143000; OLMo the last stage). O and
    V are point values against it (design: Measures; Storage)."""
    import checkpoints211 as C
    run, fam = C.BY_KEY[checkpoint_key]["run"], C.BY_KEY[checkpoint_key]["family"]
    refs = [cp["key"] for cp in C.CHECKPOINTS if cp["run"] == run and cp["family"] == fam and cp["reference"]]
    return refs[-1]


def o_and_v(z: np.ndarray, v: np.ndarray, w: np.ndarray, zf: np.ndarray, vf: np.ndarray, wf: np.ndarray):
    """O and V for every layer and head against the final: z [n_docs, L, H, n_pos, hd] pooled over documents and
    positions, v and w [L, H, hd, D]. Returns two [L, H] arrays."""
    nL, nH = v.shape[:2]
    O = np.zeros((nL, nH))
    V = np.zeros((nL, nH))
    for L in range(nL):
        for h in range(nH):
            za = z[:, L, h].reshape(-1, z.shape[-1])
            zb = zf[:, L, h].reshape(-1, zf.shape[-1])
            O[L, h] = MS.cka_distance_factored(za, w[L, h], zb, wf[L, h])
            V[L, h] = MS.ov_distance_factored(v[L, h], w[L, h], vf[L, h], wf[L, h], vf[L, h], wf[L, h])
    return O, V


def load_final(checkpoint_key: str, text_set: str, out: Path, pinned: dict) -> dict | None:
    """The run's final reference record's Z and OV factors, or None while it is not complete."""
    key = final_reference(checkpoint_key)
    path = record_path(out, key, text_set)
    if key not in pinned or resume_state(path, record_meta(pinned[key], text_set, code_hash())) != "complete":
        return None
    z = np.load(path, allow_pickle=False)
    return {"key": key, "z": z["z"], "v": z["ov_v"], "w": z["ov_w"]}


def record_checkpoint(model, docs, family: str, checkpoint_key: str, text_set: str, out: Path, meta: dict,
                      references: dict | None = None, final: dict | None = None) -> dict:
    """Run every document through the model and write one record.

    Kept per document, every checkpoint (design: What is recorded; Storage and memory):
      mlp     [n_docs, L, n_pos, D]     each layer's MLP output at positions 3, 6, ..., 255 (M is recomputed from them)
    Reference checkpoints also keep:
      att     [n_docs, L, H, T, T]      the attention rows
      z       [n_docs, L, H, n_pos, hd] each head's attention output before its output block (its contribution is
                                        z @ ov_w[L, h], so O is exact from z and the block; design: contributions not kept)
      ov_v, ov_w [L, H, hd, D]          the value and output blocks (OV = v^T w)
    Every other checkpoint keeps O_final and V_final [L, H]: O and V against its run's final reference (`final`, the
    final's z and factors), and drops its own z and factors.
    `references` maps a reference checkpoint key of the same run to its loaded attention rows
    ({key: [n_docs, L, H, T, T] float32}). When given, this (non-reference) checkpoint's attention rows are reduced to
    the per-document, per-head-pair distances against them (P, P0, cosine) and not stored: the full matrices, not their
    means, so the bootstrap can refit the match inside each document resample. A reference checkpoint keeps its rows,
    and its rows against the other references of its run are reduced by `reduce_reference_pair` after both are on disk.
    """
    import torch
    ad = adapter_for(model)
    nL, nH = ad.n_layers(), ad.n_heads()
    n_docs = len(docs)
    keep_rows = bool(meta.get("reference", False))
    n_tok = int(len(docs[0]["ids"]))
    n_pos = len(positions_for(n_tok))
    att = np.zeros((n_docs, nL, nH, n_tok, n_tok), np.float32) if keep_rows else None   # float32 (design: Storage)
    zz = np.zeros((n_docs, nL, nH, n_pos, ad.head_dim()), np.float32)      # each head's output before W_O
    mlp = np.zeros((n_docs, nL, n_pos, ad.d_model()), np.float32)          # float32 (design: Storage)
    fac = [ad.factors(L) for L in range(nL)]
    ov_v = np.stack([f[0] for f in fac]).astype(np.float32)                 # [L, H, hd, D]
    ov_w = np.stack([f[1] for f in fac]).astype(np.float32)
    red = {k: _empty_reduction(n_docs, nL, nH) for k in (references or {})}
    t0 = time.time()
    for d, doc in enumerate(docs):
        ids = torch.tensor([doc["ids"]], dtype=torch.long)
        caps = ad.capture(ids)
        for L in range(nL):
            w = caps["att"][L].float().cpu().numpy()[0]                       # [H, T, T]
            if keep_rows:
                att[d, L] = w.astype(np.float32)
            zz[d, L] = caps["z"][L].float().cpu().numpy()[0]
            mlp[d, L] = caps["mlp"][L].float().cpu().numpy()[0]
            for ref_key, ref_rows in (references or {}).items():
                _reduce_into(red[ref_key], d, L, w, ref_rows[d, L].astype(np.float32))
    arrays = {"mlp": mlp, "n_docs": np.array(n_docs)}
    if checkpoint_key == final_reference(checkpoint_key):
        arrays["O_final"] = np.zeros((nL, nH))                              # the final against itself
        arrays["V_final"] = np.zeros((nL, nH))
    elif not keep_rows:
        # a non-reference checkpoint is reduced to O and V against its run's final as it is read (references are
        # recorded first, so the final is on disk); its Z and factors are not kept
        if final is None:
            raise RuntimeError(f"{checkpoint_key}: the run's final reference is not recorded yet; record references first")
        arrays["O_final"], arrays["V_final"] = o_and_v(zz, ov_v, ov_w, final["z"], final["v"], final["w"])
    if att is not None:
        arrays["att"] = att
        arrays["z"] = zz                                                    # references keep Z and the OV factors
        arrays["ov_v"] = ov_v
        arrays["ov_w"] = ov_w
    for ref_key, r in red.items():
        for k, v in r.items():
            arrays[f"red_{ref_key}_{k}"] = v
    path = record_path(out, checkpoint_key, text_set)
    save_record(path, arrays, meta)
    return {"path": str(path), "seconds": time.time() - t0, "n_docs": n_docs, "kept_rows": keep_rows,
            "reduced_against": sorted(red)}


def _pack_moments(name: str, x: np.ndarray) -> dict:
    """Per-document centred moments of an array [n_docs, L, ..., D] (the last axis is the feature axis), keyed
    `mom_{name}_d{d}_l{L}_{X|G|S|n}` so every record stays a plain-array npz (no pickles)."""
    out = {}
    for d in range(x.shape[0]):
        for L in range(x.shape[1]):
            m = centred_moments(x[d, L].reshape(-1, x.shape[-1]))
            for k in ("X", "G", "S"):
                out[f"mom_{name}_d{d}_l{L}_{k}"] = m[k]
            out[f"mom_{name}_d{d}_l{L}_n"] = np.array(m["n"])
    return out


def load_att(path: Path) -> np.ndarray:
    """A reference record's attention rows, [n_docs, L, H, 256, 256] float32."""
    return np.load(path, allow_pickle=False)["att"]


def _empty_reduction(n_docs: int, nL: int, nH: int) -> dict:
    """One checkpoint's per-document, per-head-pair reduction against one reference: P, P0 as additive parts (P0s the
    JSD summed over kept queries, P0k the kept count, P0t the queries per document) and the cosine measure as additive
    parts (cosd the dot product, cosx and cosy each side's squared norm), so a resample of documents gets P0, its
    dropped count and the flattened-over-documents cosine exactly (MS.p0_from_parts, MS.cos_from_parts)."""
    return {"P": np.zeros((n_docs, nL, nH, nH)), "P0s": np.zeros((n_docs, nL, nH, nH)),
            "P0k": np.zeros((n_docs, nL, nH, nH), np.int64), "P0t": np.zeros((), np.int64),
            "cosd": np.zeros((n_docs, nL, nH, nH)), "cosx": np.zeros((n_docs, nL, nH)),
            "cosy": np.zeros((n_docs, nL, nH))}


def _reduce_into(red: dict, d: int, L: int, w: np.ndarray, r: np.ndarray) -> None:
    """Fill one document and layer of a reduction: w the checkpoint's rows, r the reference's, each [H, T, T]."""
    red["P"][d, L] = MS.pattern_matrix(w, r)
    s0, k0, t0 = MS.p0_parts_matrix(w, r)
    red["P0s"][d, L], red["P0k"][d, L], red["P0t"][()] = s0, k0, t0
    red["cosd"][d, L], red["cosx"][d, L], red["cosy"][d, L] = MS.cos_parts_matrix(w, r)


def reduce_reference_pair(checkpoint_key: str, ref_key: str, text_set: str, out: Path, pinned: dict) -> dict | None:
    """Reduce one reference checkpoint's attention rows against another reference of its run, as per-document,
    per-head-pair distances, and store them in the first record (keys `red_<ref_key>_<part>`, `_empty_reduction`). Idempotent: returns
    None when the pair is already reduced or either record is not complete."""
    import checkpoints211 as C
    if C.BY_KEY[checkpoint_key]["run"] != C.BY_KEY[ref_key]["run"]:
        raise ValueError(f"{checkpoint_key} and {ref_key} are not in the same run")
    path = record_path(out, checkpoint_key, text_set)
    ref_path = record_path(out, ref_key, text_set)
    if checkpoint_key not in pinned or ref_key not in pinned:
        return None
    meta = record_meta(pinned[checkpoint_key], text_set, code_hash())
    meta["reference"] = True
    if resume_state(path, meta) != "complete" or resume_state(ref_path, record_meta(pinned[ref_key], text_set,
                                                                                   code_hash())) != "complete":
        return None
    z = np.load(path, allow_pickle=False)
    if f"red_{ref_key}_P" in z.files:
        return None
    a = z["att"]
    b = load_att(ref_path)
    n_docs, nL, nH = a.shape[:3]
    red = _empty_reduction(n_docs, nL, nH)
    for d in range(n_docs):
        for L in range(nL):
            _reduce_into(red, d, L, a[d, L].astype(np.float32), b[d, L].astype(np.float32))
    arrays = {k: z[k] for k in z.files if k != "complete" and not k.startswith("meta_")}
    arrays.update({f"red_{ref_key}_{k}": v for k, v in red.items()})
    if ref_key == final_reference(checkpoint_key):
        # a reference recorded before its run's final gets O and V against it here (non-references at record time)
        f = np.load(ref_path, allow_pickle=False)
        arrays["O_final"], arrays["V_final"] = o_and_v(z["z"], z["ov_v"], z["ov_w"], f["z"], f["ov_v"], f["ov_w"])
    save_record(path, arrays, meta)
    return {"path": str(path), "ref": ref_key}


# ---------------------------------------------------------------- the noise floor

def floor_keys() -> list[str]:
    """The checkpoints whose noise floor is recorded: every Pythia run's final, step 10000 of R0 and of each
    weight-seed run (211.8 compares those checkpoints, not finals: "the final (or compared) checkpoint", frozen as the
    floor policy 2026-11-13 after Alden #13), OLMo's end of pretraining (4T, the final of 211.6's late distances and of
    211.7) and its mid-training end (the floor for comparisons involving Instruct, whose weights are stored in bf16;
    design: Noise floor)."""
    import checkpoints211 as C
    keys = [C.pythia_key(r, C.FINAL_STEP) for r in C.runs_of(C.PYTHIA_FAMILY)]
    keys += [C.pythia_key(r, C.THREAD_STEP) for r in ("R0", *C.runs_with_factor("weights"))]
    return keys + ["olmo_pretrain_4T", "olmo_midtrain_51B"]


def floor_path(out: Path, checkpoint_key: str, text_set: str) -> Path:
    """Kept apart from records/ so the record reader never takes a floor for a checkpoint."""
    return Path(out) / "floors" / f"{checkpoint_key}.{text_set}.npz"


def round_through_bf16(model) -> None:
    """Every floating parameter and buffer rounded to bf16 and back to float32, in place (design: Noise floor, "the
    same weights rounded through bf16 and run in float32"). The model is changed: record its own rows first."""
    import torch
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            if t.is_floating_point():
                t.copy_(t.to(torch.bfloat16).to(t.dtype))


def record_floor(model, docs, checkpoint_key: str, text_set: str, out: Path, meta: dict) -> dict:
    """The noise floor of one checkpoint and text set, per layer and measure: P (mean over heads and documents of the
    pattern distance between a head and itself), M (1 minus pooled CKA of the MLP outputs over all documents) and V
    (mean over heads of the OV distance), each between the checkpoint's own record and the same weights rounded through
    bf16 (`round_through_bf16`, applied by the caller). Reads the checkpoint's complete record (att, mlp and the OV
    factors, so it must be a reference)."""
    import torch
    rec = np.load(record_path(out, checkpoint_key, text_set), allow_pickle=False)
    for k in ("att", "mlp", "ov_v", "ov_w"):
        if k not in rec.files:
            raise ValueError(f"{checkpoint_key}.{text_set}: the floor needs the record's {k} (a reference record)")
    ad = adapter_for(model)
    nL, nH = ad.n_layers(), ad.n_heads()
    n_docs = len(docs)
    p = np.zeros((n_docs, nL, nH))
    p0s, p0k, p0t = np.zeros((n_docs, nL, nH)), np.zeros((n_docs, nL, nH), np.int64), 0
    mlp = np.zeros_like(rec["mlp"], dtype=np.float32)
    for d, doc in enumerate(docs):
        caps = ad.capture(torch.tensor([doc["ids"]], dtype=torch.long))
        for L in range(nL):
            w = caps["att"][L].float().cpu().numpy()[0]
            for h in range(nH):
                p[d, L, h] = MS.pattern_distance(w[h], rec["att"][d, L, h].astype(np.float32), variant="P")
            s0, k0, p0t = MS.p0_parts_matrix(w, rec["att"][d, L].astype(np.float32))
            p0s[d, L], p0k[d, L] = np.diagonal(s0), np.diagonal(k0)          # each head against itself
            mlp[d, L] = caps["mlp"][L].float().cpu().numpy()[0]
    fac = [ad.factors(L) for L in range(nL)]
    v = np.array([[MS.ov_distance_factored(fac[L][0][h], fac[L][1][h], rec["ov_v"][L, h], rec["ov_w"][L, h],
                                           rec["ov_v"][L, h], rec["ov_w"][L, h]) for h in range(nH)]
                  for L in range(nL)])
    m = np.array([1.0 - pooled_cka(mlp, rec["mlp"], np.arange(n_docs), L) for L in range(nL)])
    p0_heads = MS.p0_from_parts(p0s, p0k, p0t, np.ones((1, n_docs)))[0][0]          # [L, H], NaN = no P0
    with np.errstate(invalid="ignore"):
        p0_layer = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in p0_heads])
    arrays = {"P": p.mean(axis=(0, 2)), "P_heads": p.mean(axis=0), "M": m, "V": v.mean(axis=1), "V_heads": v,
              "P0": p0_layer, "P0_heads": p0_heads}
    path = floor_path(out, checkpoint_key, text_set)
    save_record(path, arrays, meta)
    return {"path": str(path), "P": arrays["P"].tolist(), "M": m.tolist(), "V": arrays["V"].tolist()}


def references_for(checkpoint_key: str, text_set: str, out: Path, pinned: dict) -> dict:
    """The reference checkpoints of this checkpoint's run that already have a complete record on disk:
    {key: attention rows}. A non-reference checkpoint reduces against these as it is read."""
    import checkpoints211 as C
    run = C.BY_KEY[checkpoint_key]["run"]
    refs = {}
    for cp in C.CHECKPOINTS:
        if cp["family"] != C.BY_KEY[checkpoint_key]["family"] or cp["run"] != run or not cp["reference"]:
            continue
        path = record_path(out, cp["key"], text_set)
        if resume_state(path, record_meta(pinned[cp["key"]], text_set, code_hash())) == "complete":
            refs[cp["key"]] = load_att(path)
    return refs


def main(argv=None):
    import checkpoints211 as C
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, choices=sorted(C.BY_KEY))
    ap.add_argument("--texts", required=True, help="JSON: {\"A\": [{\"index\", \"ids\"}...], \"B\": [...]}")
    ap.add_argument("--out", default="results/part211")
    ap.add_argument("--set", dest="text_set", default="A", choices=["A", "B"])
    a = ap.parse_args(argv)
    cp = C.BY_KEY[a.checkpoint]
    pinned = C.load_pinned()
    meta = record_meta(pinned[a.checkpoint], a.text_set, code_hash())
    meta["reference"] = bool(cp["reference"])
    path = record_path(Path(a.out), a.checkpoint, a.text_set)
    if resume_state(path, meta) == "complete":
        print(f"skip {path}: complete and all four fields match")
        return 0
    docs = json.loads(Path(a.texts).read_text(encoding="utf-8"))[a.text_set]
    refs = None if cp["reference"] else references_for(a.checkpoint, a.text_set, Path(a.out), pinned)
    if refs is not None and not refs:
        raise SystemExit(f"{a.checkpoint} is not a reference checkpoint and no reference of run {cp['run']} is on "
                         f"disk: record the references first (run211.py orders them)")
    model = load_model(cp["repo_id"], pinned[a.checkpoint])
    info = record_checkpoint(model, docs, cp["family"], a.checkpoint, a.text_set, Path(a.out), meta,
                             references=refs)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
