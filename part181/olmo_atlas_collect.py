#!/usr/bin/env python3
"""Resumable OLMo 2 1B neuron/control atlas collector (CPU, one checkpoint per session).

Adapted from atlas_collect_fast2.py (the Qwen3-1.7B results/atlas-full collector).
Same per-unit record format (layer, unit|control_id, max, frac_above, top, random,
weak, model_arm; each window {doc, pos, act, window} with the firing token marked
[[...]]), with three changes for short cloud sessions:

* The unit is the input to mlp.down_proj, i.e. act_fn(gate_proj(x)) * up_proj(x):
  16 layers x 8,192 neurons, plus N seeded random unit directions per layer in that
  8,192-space (default 256), projected the same way (x @ d).
* Progress is checkpointed every --checkpoint-every documents (atomic npz +
  progress.json) and a killed run resumes from the last checkpoint. Batches and
  reservoir random keys are a pure function of (config, chunk, batch), so a resumed
  run produces the same state as an uninterrupted one.
* Running state is kept as compact arrays (doc ordinal, pos, act per rank). Window
  text is reconstructed from the corpus only by the `export` subcommand.

Pass 1 (required): per-unit corpus max and top-K documents (one entry per document,
the position of that document's maximum), exactly as atlas_collect_fast2.
Pass 2 (optional, may cover only the first --pass2-docs documents): frac_above
(share of valid positions > 10% of the pass-1 max), and the 10 random windows above
10% and 5 weak windows in (10%, 30%] of max, by seeded smallest-random-key
reservoirs (vectorised; same selection rule as the original, different RNG stream).

Subcommands:
  collect  --model-dir SNAP --corpus PARQUET --out DIR [--docs N] [--pass 1|2] ...
  export   --out DIR --corpus PARQUET --model-dir SNAP   -> neurons.jsonl, controls.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EXPECTED_LAYERS = 16
EXPECTED_WIDTH = 8192
MAX_DOC_TOKENS = 512
WINDOW_TOKENS = 64
FIRING_OFFSET = 32
TOP_K = 20
RANDOM_WINDOWS = 10
WEAK_WINDOWS = 5
N_CONTROLS = 256
SPARSE_SUPPORTS = (2, 4, 16, 64)
N_SPARSE_PER_SUPPORT = 64
STATE_VERSION = 1


# ---------------------------------------------------------------------------- utilities

def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(p) for p in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def generate_controls(n: int, width: int, seed: int, layer: int = 0) -> np.ndarray:
    """Fixed-seed random unit-norm row directions (same rule as atlas_collect_fast2)."""
    if n < 0 or width <= 0:
        raise ValueError("control count must be nonnegative and width must be positive")
    rng = np.random.default_rng(stable_seed(seed, "control-directions", layer))
    values = rng.standard_normal((n, width), dtype=np.float32)
    if n:
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.finfo(np.float32).tiny)
    return values


def generate_sparse_controls(n: int, width: int, k: int, seed: int = 5601,
                             layer: int = 0) -> np.ndarray:
    """Generate seeded sparse, unit-norm directions with exactly ``k`` nonzeros."""
    if n < 0 or width <= 0 or not 1 <= k <= width:
        raise ValueError("sparse control count, width, and support size are invalid")
    rng = np.random.default_rng(stable_seed(seed, "sparse-controls", layer, k))
    values = np.zeros((n, width), dtype=np.float32)
    for row in range(n):
        support = rng.choice(width, size=k, replace=False)
        coeff = rng.standard_normal(k).astype(np.float32)
        norm = float(np.linalg.norm(coeff))
        if norm == 0:
            coeff[0] = 1.0
            norm = 1.0
        values[row, support] = coeff / norm
    return values


def decode_marked_window(tokenizer: Any, token_ids: Sequence[int], firing_pos: int,
                         window_tokens: int = WINDOW_TOKENS, firing_offset: int = FIRING_OFFSET) -> str:
    """Verbatim copy of atlas_collect_fast2.decode_marked_window (format must match)."""
    if not token_ids or not 0 <= firing_pos < len(token_ids):
        raise ValueError("firing position must identify a token in a nonempty document")
    if window_tokens <= 0 or not 0 <= firing_offset < window_tokens:
        raise ValueError("invalid window size or firing offset")
    start = firing_pos - firing_offset
    end = start + window_tokens
    real_start = max(0, start)
    real_end = min(len(token_ids), end)
    left_pad = max(0, -start)
    right_pad = max(0, end - len(token_ids))
    decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    prefix = tokenizer.decode(list(token_ids[real_start:firing_pos]), **decode_kwargs)
    firing = tokenizer.decode([int(token_ids[firing_pos])], **decode_kwargs)
    suffix = tokenizer.decode(list(token_ids[firing_pos + 1:real_end]), **decode_kwargs)

    def escape_source(value: str) -> str:
        while "[[" in value or "]]" in value:
            value = value.replace("[[", "[ [").replace("]]", "] ]")
        return value
    prefix, firing, suffix = map(escape_source, (prefix, firing, suffix))
    if prefix.endswith("["):
        prefix += " "
    if firing.startswith("["):
        firing = " " + firing
    if firing.endswith("]"):
        firing += " "
    if suffix.startswith("]"):
        suffix = " " + suffix
    left = f"⟦PAD×{left_pad}⟧" if left_pad else ""
    right = f"⟦PAD×{right_pad}⟧" if right_pad else ""
    marked = f"{left}{prefix}[[{firing}]]{suffix}{right}"
    if marked.count("[[") != 1 or marked.count("]]") != 1:
        strip = lambda value: value.replace("[", "(").replace("]", ")")
        marked = f"{left}{strip(prefix)}[[{strip(firing)}]]{strip(suffix)}{right}"
    assert marked.count("[[") == 1 and marked.count("]]") == 1
    return marked


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def config_sha(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------- corpus

def read_corpus_texts(corpus: Path, limit: int | None) -> list[str]:
    """NeelNanda/pile-10k parquet (or JSONL with a text field), file order."""
    corpus = Path(corpus)
    if corpus.suffix == ".parquet":
        import pyarrow.parquet as pq
        texts = pq.read_table(corpus, columns=["text"]).column("text").to_pylist()
    else:
        texts = [json.loads(line)["text"] for line in corpus.open(encoding="utf-8") if line.strip()]
    return texts if limit is None else texts[:limit]


def tokenize_docs(tokenizer: Any, texts: Sequence[str], start: int = 0) -> list[dict[str, Any]]:
    """Ordinal = file index. Truncate to MAX_DOC_TOKENS; empty documents are kept as
    zero-length and skipped by the batcher (so ordinals stay file indices)."""
    docs = []
    for offset in range(0, len(texts), 128):
        chunk = list(texts[offset:offset + 128])
        enc = tokenizer(chunk, truncation=True, max_length=MAX_DOC_TOKENS, padding=False,
                        add_special_tokens=True)["input_ids"]
        for i, ids in enumerate(enc):
            docs.append({"ordinal": start + offset + i, "ids": [int(t) for t in ids]})
    return docs


def chunk_batches(docs: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Length-sort within one checkpoint chunk (deterministic) and cut into batches."""
    live = [d for d in docs if d["ids"]]
    ordered = sorted(live, key=lambda d: (len(d["ids"]), d["ordinal"]))
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


# ---------------------------------------------------------------------------- state

class AtlasState:
    """All layers' running state as compact arrays. U = neurons + controls per layer."""

    def __init__(self, n_layers: int, width: int, n_controls: int,
                 control_kind: Sequence[str] | None = None,
                 control_k: Sequence[int] | None = None):
        self.L, self.W, self.C = n_layers, width, n_controls
        U = width + n_controls
        self.control_kind = np.asarray(control_kind or (["neuron"] * width + ["dense"] * n_controls),
                                       dtype="U8")
        self.control_k = np.asarray(control_k or ([1] * width + [0] * n_controls), dtype=np.int16)
        if self.control_kind.shape != (U,) or self.control_k.shape != (U,):
            raise ValueError("control kind and k metadata must have one value per state unit")
        self.max = np.full((n_layers, U), -np.inf, dtype=np.float32)
        self.top_act = np.full((n_layers, U, TOP_K), -np.inf, dtype=np.float32)
        self.top_doc = np.full((n_layers, U, TOP_K), -1, dtype=np.int32)
        self.top_pos = np.full((n_layers, U, TOP_K), -1, dtype=np.int16)
        # pass 2
        self.above = np.zeros((n_layers, U), dtype=np.int64)
        self.positions = np.zeros((), dtype=np.int64)
        self.res = {}
        for kind, k in (("random", RANDOM_WINDOWS), ("weak", WEAK_WINDOWS)):
            self.res[kind] = {
                "key": np.full((n_layers, U, k), np.inf, dtype=np.float32),
                "doc": np.full((n_layers, U, k), -1, dtype=np.int32),
                "pos": np.full((n_layers, U, k), -1, dtype=np.int16),
                "act": np.full((n_layers, U, k), np.nan, dtype=np.float32),
            }

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"max": self.max, "unit_kind": self.control_kind, "unit_k": self.control_k,
               "top_act": self.top_act, "top_doc": self.top_doc,
               "top_pos": self.top_pos, "above": self.above, "positions": self.positions}
        for kind, d in self.res.items():
            for name, arr in d.items():
                out[f"{kind}_{name}"] = arr
        return out

    def load(self, saved: Any) -> None:
        self.max[...] = saved["max"]
        self.top_act[...] = saved["top_act"]
        self.top_doc[...] = saved["top_doc"]
        self.top_pos[...] = saved["top_pos"]
        self.above[...] = saved["above"]
        self.positions = np.array(saved["positions"], dtype=np.int64)
        for kind, d in self.res.items():
            for name in d:
                d[name][...] = saved[f"{kind}_{name}"]


def merge_top(state_act, state_doc, state_pos, batch_act, batch_doc, batch_pos, k):
    """[U,k] running top-k merged with [U,B] per-document maxima (torch tensors)."""
    import torch
    cand_act = torch.cat((state_act, batch_act), dim=1)
    cand_doc = torch.cat((state_doc, batch_doc), dim=1)
    cand_pos = torch.cat((state_pos, batch_pos), dim=1)
    top, idx = torch.topk(cand_act, k=k, dim=1, largest=True, sorted=True)
    return top, torch.gather(cand_doc, 1, idx), torch.gather(cand_pos, 1, idx)


def merge_reservoir(res: dict, layer: int, values, qualifies, flat_doc, flat_pos, gen) -> None:
    """Keep, per unit, the k qualifying positions with the smallest seeded random keys."""
    import torch
    k = res["key"].shape[-1]
    n = values.shape[0]
    keys = torch.rand(values.shape, generator=gen, dtype=torch.float32)
    keys.masked_fill_(~qualifies, math.inf)
    kk = min(k, n)
    bkey, bidx = torch.topk(keys, k=kk, dim=0, largest=False, sorted=False)  # [kk,U]
    bact = torch.gather(values, 0, bidx).T
    bdoc = flat_doc[bidx].T.to(torch.int32)
    bpos = flat_pos[bidx].T.to(torch.int16)
    bkey = bkey.T
    skey = torch.from_numpy(res["key"][layer])
    ckey = torch.cat((skey, bkey), dim=1)
    top, idx = torch.topk(ckey, k=k, dim=1, largest=False, sorted=True)
    finite = torch.isfinite(top)
    for name, batch_vals, fill in (("doc", bdoc, -1), ("pos", bpos, -1), ("act", bact, math.nan)):
        cand = torch.cat((torch.from_numpy(res[name][layer]), batch_vals), dim=1)
        picked = torch.gather(cand, 1, idx)
        res[name][layer] = torch.where(finite, picked, torch.full_like(picked, fill)).numpy()
    res["key"][layer] = torch.where(finite, top, torch.full_like(top, math.inf)).numpy()


class Collector:
    """Hooks every requested layer's mlp.down_proj input and updates AtlasState."""

    def __init__(self, model: Any, layers: Sequence[int], state: AtlasState, controls: np.ndarray,
                 seed: int, bookkeeping: bool = True):
        import torch
        self.torch = torch
        self.model = model
        self.layers = list(layers)
        self.state = state
        self.seed = seed
        self.bookkeeping = bookkeeping
        self.dirs = torch.from_numpy(controls)  # [L, C, W]
        self.hook_seconds = 0.0
        self.current: dict[str, Any] = {}

    def run_batch(self, batch: Sequence[dict[str, Any]], pass_number: int, pad_id: int,
                  batch_tag: tuple) -> None:
        torch = self.torch
        max_len = max(len(d["ids"]) for d in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for r, d in enumerate(batch):
            ids[r, :len(d["ids"])] = torch.tensor(d["ids"])
            mask[r, :len(d["ids"])] = True
        self.current = {"batch": batch, "mask": mask, "pass": pass_number, "tag": batch_tag,
                        "doc": torch.tensor([d["ordinal"] for d in batch], dtype=torch.int32)}
        base = self.model.model
        decoder_layers = base.layers
        handles = []
        last = max(self.layers)
        for slot, li in enumerate(self.layers):
            handles.append(decoder_layers[li].mlp.down_proj.register_forward_pre_hook(
                lambda _m, args, slot=slot: self._hook(slot, args[0])))
        handles.append(decoder_layers[last].mlp.down_proj.register_forward_hook(self._stop))
        try:
            with torch.inference_mode():
                base(input_ids=ids, attention_mask=mask.long(), use_cache=False)
        except _Stop:
            pass
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def _stop(*_a):
        raise _Stop

    def _hook(self, slot: int, x: Any) -> None:
        if not self.bookkeeping:
            return None
        torch = self.torch
        t0 = time.perf_counter()
        st, cur = self.state, self.current
        mask = cur["mask"]
        if x.ndim != 3 or x.shape[-1] != st.W:
            raise ValueError(f"down_proj input has shape {tuple(x.shape)}, expected width {st.W}")
        ctrl = torch.matmul(x.float(), self.dirs[slot].T) if st.C else None
        if cur["pass"] == 1:
            parts = []
            for acts in (x, ctrl):
                if acts is None:
                    continue
                masked = acts.masked_fill(~mask.unsqueeze(-1), -math.inf)
                v, p = masked.max(dim=1)  # [B,U]
                parts.append((v.float(), p))
            v = torch.cat([a for a, _ in parts], dim=1)
            p = torch.cat([b for _, b in parts], dim=1)
            st.max[slot] = np.maximum(st.max[slot], v.max(dim=0).values.numpy())
            U, B = v.shape[1], v.shape[0]
            top, doc, pos = merge_top(
                torch.from_numpy(st.top_act[slot]), torch.from_numpy(st.top_doc[slot]),
                torch.from_numpy(st.top_pos[slot]), v.T.contiguous(),
                cur["doc"].reshape(1, B).expand(U, B), p.T.to(torch.int16), TOP_K)
            st.top_act[slot], st.top_doc[slot], st.top_pos[slot] = top.numpy(), doc.numpy(), pos.numpy()
        else:
            B, T = mask.shape
            vals = x.reshape(B * T, -1).float()
            if ctrl is not None:
                vals = torch.cat((vals, ctrl.reshape(B * T, -1)), dim=1)
            valid = mask.reshape(-1)
            mx = torch.from_numpy(st.max[slot])
            ten, thirty = mx * 0.1, mx * 0.3
            above = (vals > ten) & valid.unsqueeze(1)
            st.above[slot] += above.sum(dim=0).numpy()
            lo, hi = torch.minimum(ten, thirty), torch.maximum(ten, thirty)
            weak = (vals > lo) & (vals <= hi) & valid.unsqueeze(1)
            flat_doc = cur["doc"].repeat_interleave(T)
            flat_pos = torch.arange(T).repeat(B)
            for kind, qual in (("random", above), ("weak", weak)):
                gen = torch.Generator().manual_seed(
                    stable_seed(self.seed, "reservoir", kind, self.layers[slot], *cur["tag"]))
                merge_reservoir(st.res[kind], slot, vals, qual, flat_doc, flat_pos, gen)
        self.hook_seconds += time.perf_counter() - t0
        return None


class _Stop(Exception):
    pass


# ---------------------------------------------------------------------------- collect

def load_model(model_dir: Path, dtype: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True,
        dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[dtype])
    model.eval()
    return tok, model


def make_config(args: argparse.Namespace, model_dir: Path, layers: Sequence[int], width: int) -> dict:
    cfg_path = model_dir / "config.json"
    n_dense = int(args.n_controls)
    n_sparse = int(getattr(args, "n_sparse_per_k", 0))
    control_layout = [{"kind": "dense", "k": 0, "count": n_dense}]
    control_layout.extend({"kind": "sparse", "k": int(k), "count": n_sparse}
                          for k in SPARSE_SUPPORTS if n_sparse)
    return {
        "state_version": STATE_VERSION, "model_dir_name": model_dir.name,
        "model_config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest() if cfg_path.is_file() else None,
        "corpus": Path(args.corpus).name, "docs": args.docs, "pass2_docs": args.pass2_docs,
        "dtype": args.dtype, "batch_size": args.batch_size, "checkpoint_every": args.checkpoint_every,
        "seed": args.seed, "n_dense_controls": n_dense, "n_sparse_per_k": n_sparse,
        "n_controls": n_dense + len(SPARSE_SUPPORTS) * n_sparse,
        "control_layout": control_layout, "layers": list(layers), "width": width,
        "max_doc_tokens": MAX_DOC_TOKENS, "top_k": TOP_K, "model_arm": args.model_arm,
        "manifest_sha256": getattr(args, "manifest_sha256", None),
    }


def collect(args: argparse.Namespace, tok=None, model=None) -> dict:
    """Run (or resume) pass 1 and optionally pass 2. Returns a timing summary."""
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else Path(".")
    load_s = None
    if model is None:
        t_load = time.perf_counter()
        tok, model = load_model(model_dir, args.dtype, args.threads)
        load_s = time.perf_counter() - t_load
    n_layers_model = len(model.model.layers)
    layers = args.layers if args.layers else list(range(n_layers_model))
    width = int(model.model.layers[0].mlp.down_proj.in_features)
    if args.strict_shape and (n_layers_model != EXPECTED_LAYERS or width != EXPECTED_WIDTH):
        raise RuntimeError(f"model has {n_layers_model} layers x {width}; expected 16 x 8192")
    config = make_config(args, model_dir, layers, width)
    sha = config_sha(config)
    n_sparse = int(getattr(args, "n_sparse_per_k", 0))
    control_parts = []
    if args.n_controls:
        control_parts.append(np.stack([generate_controls(args.n_controls, width, args.seed, li) for li in layers]))
    for k in SPARSE_SUPPORTS:
        if n_sparse:
            control_parts.append(np.stack([generate_sparse_controls(n_sparse, width, k, args.seed, li)
                                           for li in layers]))
    controls = np.concatenate(control_parts, axis=1) if control_parts else \
        np.zeros((len(layers), 0, width), dtype=np.float32)

    texts = read_corpus_texts(Path(args.corpus), args.docs)
    n_docs = len(texts)
    pass2_docs = n_docs if args.pass2_docs is None else min(args.pass2_docs, n_docs)

    kinds = ["neuron"] * width
    ks = [1] * width
    kinds.extend(["dense"] * args.n_controls)
    ks.extend([0] * args.n_controls)
    for k in SPARSE_SUPPORTS:
        kinds.extend(["sparse"] * n_sparse)
        ks.extend([k] * n_sparse)
    state = AtlasState(len(layers), width, args.n_controls + len(SPARSE_SUPPORTS) * n_sparse, kinds, ks)
    progress_path, state_path = out / "progress.json", out / "state.npz"
    progress = {"config": config, "config_sha": sha, "pass": 1, "docs_done": 0,
                "pass1_positions": 0, "elapsed": {}}
    if progress_path.is_file():
        old = json.loads(progress_path.read_text())
        if old.get("config_sha") != sha:
            raise SystemExit(f"{out} holds a run with a different config ({old.get('config_sha')} != {sha}); "
                             "choose a new --out")
        progress = old
        if state_path.is_file():
            with np.load(state_path) as saved:
                state.load(saved)
                # The npz is written first and carries its own cursor, so a kill between the
                # two writes can never double-count a chunk.
                cursor = saved["cursor"].tolist()
            progress["pass"], progress["docs_done"], progress["pass1_positions"] = cursor
        else:
            progress["pass"], progress["docs_done"], progress["pass1_positions"] = 1, 0, 0
        print(f"resuming: pass {progress['pass']}, {progress['docs_done']} docs done", flush=True)

    coll = Collector(model, layers, state, controls, args.seed, bookkeeping=not args.forward_only)
    pad_id = int(tok.pad_token_id)
    target_pass = args.passes
    timing = progress.setdefault("timing", {})
    while progress["pass"] <= target_pass:
        p = progress["pass"]
        limit = n_docs if p == 1 else pass2_docs
        while progress["docs_done"] < limit:
            c0 = progress["docs_done"]
            c1 = min(c0 + args.checkpoint_every, limit)
            if p == 1 and args.snapshot_every:
                next_snapshot = (c0 // args.snapshot_every + 1) * args.snapshot_every
                c1 = min(c1, next_snapshot)
            docs = tokenize_docs(tok, texts[c0:c1], start=c0)
            hook0, wall0 = coll.hook_seconds, time.perf_counter()
            for b, batch in enumerate(chunk_batches(docs, args.batch_size)):
                coll.run_batch(batch, p, pad_id, (p, c0, b))
                if p == 1:
                    progress["pass1_positions"] += sum(len(d["ids"]) for d in batch)
                else:
                    state.positions = np.array(int(state.positions) + sum(len(d["ids"]) for d in batch))
            wall = time.perf_counter() - wall0
            hook = coll.hook_seconds - hook0
            t = timing.setdefault(f"pass{p}", {"docs": 0, "wall_s": 0.0, "hook_s": 0.0, "tokens": 0})
            t["docs"] += c1 - c0
            t["wall_s"] += wall
            t["hook_s"] += hook
            t["tokens"] += sum(len(d["ids"]) for d in docs)
            progress["docs_done"] = c1
            t_save = time.perf_counter()
            _atomic_npz(state_path, **state.arrays(), cursor=np.array(
                [p, c1, progress["pass1_positions"]], dtype=np.int64))
            _atomic_json(progress_path, progress)
            t["save_s"] = t.get("save_s", 0.0) + time.perf_counter() - t_save
            if args.snapshot_every and p == 1 and c1 % args.snapshot_every == 0:
                _atomic_npz(out / f"snapshot-pass1-docs{c1:05d}.npz", **state.arrays())
            print(f"pass {p}: {c1}/{limit} docs, chunk wall {wall:.1f}s (hooks {hook:.1f}s), "
                  f"rss {peak_rss_mb():.0f} MB", flush=True)
            if args.stop_after_docs and c1 >= args.stop_after_docs and p == 1:
                print("stop-after-docs reached (simulated kill)", flush=True)
                return progress
        progress["pass"] = p + 1
        progress["docs_done"] = 0
        _atomic_npz(state_path, **state.arrays(), cursor=np.array(
            [p + 1, 0, progress["pass1_positions"]], dtype=np.int64))
        _atomic_json(progress_path, progress)
    progress["peak_rss_mb"] = peak_rss_mb()
    progress["load_s"] = load_s
    progress["complete"] = True
    _atomic_json(progress_path, progress)
    return progress


# ---------------------------------------------------------------------------- export

def export(out: Path, corpus: Path, tok: Any) -> dict:
    """Write neurons.jsonl / controls.jsonl (atlas-full format) from state.npz."""
    progress = json.loads((out / "progress.json").read_text())
    cfg = progress["config"]
    with np.load(out / "state.npz") as s:
        st = {k: s[k] for k in s.files}
    has_pass2 = progress["pass"] > 2 or (progress["pass"] == 2 and progress.get("complete"))
    needed = set(np.unique(st["top_doc"]).tolist())
    if has_pass2:
        needed |= set(np.unique(st["random_doc"]).tolist()) | set(np.unique(st["weak_doc"]).tolist())
    needed.discard(-1)
    texts = read_corpus_texts(corpus, cfg["docs"])
    ids = {}
    for o in sorted(needed):
        ids[o] = tokenize_docs(tok, [texts[o]], start=o)[0]["ids"]
    cache: dict[tuple[int, int], str] = {}

    def window(doc: int, pos: int) -> str:
        key = (doc, pos)
        if key not in cache:
            cache[key] = decode_marked_window(tok, ids[doc], pos)
        return cache[key]

    def entries(acts, docs, poss):
        rows = []
        for a, d, p in zip(acts.tolist(), docs.tolist(), poss.tolist()):
            if d < 0 or p < 0 or not math.isfinite(a):
                continue
            rows.append({"doc": int(d), "pos": int(p), "act": float(a), "window": window(int(d), int(p))})
        return rows

    W = cfg["width"]
    positions = int(st["positions"])
    counts = {"neurons": 0, "controls": 0}
    for stem in ("neurons", "controls"):
        path = out / f"{stem}.jsonl"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for slot, layer in enumerate(cfg["layers"]):
                units = range(W) if stem == "neurons" else range(W, W + cfg["n_controls"])
                for u in units:
                    maximum = float(st["max"][slot, u])
                    frac_available = bool(has_pass2 and positions and maximum > 0)
                    rec = {"layer": int(layer), "max": maximum,
                           "frac_above": (float(st["above"][slot, u] / positions) if frac_available else None),
                           "frac_above_unavailable_nonpositive_max": bool(maximum <= 0),
                           "top": entries(st["top_act"][slot, u], st["top_doc"][slot, u], st["top_pos"][slot, u]),
                           "random": entries(st["random_act"][slot, u], st["random_doc"][slot, u],
                                             st["random_pos"][slot, u]) if has_pass2 else [],
                           "weak": entries(st["weak_act"][slot, u], st["weak_doc"][slot, u],
                                           st["weak_pos"][slot, u]) if has_pass2 else [],
                           "model_arm": cfg["model_arm"]}
                    if stem == "neurons":
                        rec["unit"] = int(u)
                        rec["unit_kind"] = "neuron"
                        rec["k"] = 1
                    else:
                        rec["control_id"] = int(u - W)
                        rec["control_kind"] = str(st["unit_kind"][u]) if "unit_kind" in st else "dense"
                        rec["k"] = int(st["unit_k"][u]) if "unit_k" in st else 0
                    f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[stem] += 1
        os.replace(tmp, path)
    return counts


# ---------------------------------------------------------------------------- CLI

def parse_layers(value: str) -> list[int]:
    return sorted({int(v) for v in value.split(",") if v.strip()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model-dir", required=True, help="HF snapshot dir (snapshots/<commit>)")
    c.add_argument("--corpus", required=True, help="pile-10k parquet")
    c.add_argument("--out", required=True)
    c.add_argument("--docs", type=int, default=None, help="first N corpus documents")
    c.add_argument("--pass2-docs", type=int, default=None, help="pass 2 over only the first N docs")
    c.add_argument("--passes", type=int, choices=(1, 2), default=2)
    c.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    c.add_argument("--batch-size", type=int, default=4)
    c.add_argument("--checkpoint-every", type=int, default=64)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--seed", type=int, default=5601)
    c.add_argument("--n-controls", type=int, default=N_CONTROLS)
    c.add_argument("--n-sparse-per-k", type=int, default=N_SPARSE_PER_SUPPORT,
                   help="number of sparse control directions per support size")
    c.add_argument("--manifest-sha256", default=None,
                   help="verified input manifest hash; part of the resumable collector config")
    c.add_argument("--layers", type=parse_layers, default=None)
    c.add_argument("--model-arm", default="trained")
    c.add_argument("--forward-only", action="store_true", help="timing: hooks return immediately")
    c.add_argument("--snapshot-every", type=int, default=0,
                   help="also keep a copy of pass-1 state every N docs (for top-k stability checks; 57 MB each)")
    c.add_argument("--stop-after-docs", type=int, default=None, help="testing: stop as if killed")
    c.add_argument("--no-strict-shape", dest="strict_shape", action="store_false")
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--corpus", required=True)
    e.add_argument("--model-dir", required=True, help="tokenizer source")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "collect":
        prog = collect(args)
        print("RESULT " + json.dumps({"timing": prog.get("timing"), "load_s": prog.get("load_s"),
                                     "peak_rss_mb": peak_rss_mb()}))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        t0 = time.perf_counter()
        counts = export(Path(args.out), Path(args.corpus), tok)
        print(json.dumps({"rows": counts, "export_s": time.perf_counter() - t0, "peak_rss_mb": peak_rss_mb()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
