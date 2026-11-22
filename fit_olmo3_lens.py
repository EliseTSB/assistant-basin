#!/usr/bin/env python3
"""Fit the Neuron Lens R-lens and S-lens (under RelP) on OLMo 3 7B Instruct.

Runs both reverse-mode (R-lens: ``out`` [L-1, d, d] and ``mid`` [L, d, d]) and
forward-mode (S-lens: ``mid`` [L, d, d] and ``out`` [L, d, d]) using
``neuron_lens.olmo3`` (hybrid sliding/full attention + YaRN per-layer-type RoPE +
post-norm RMSNorms + full-width q_norm/k_norm).

Designed for Colab CUDA GPUs (L4 24 GB with ``--acc-device cpu``, or A100 40/80 GB
with ``--acc-device cuda`` or ``cpu``) as well as CPU smoke runs.

Corpus: seeded shuffle (seed 5701) of ``Salesforce/wikitext`` (``wikitext-103-raw-v1``
train) rows that tokenize to at least ``MAX_SEQ_LEN`` (128) tokens (truncated to 128),
matching ``fit_olmo_lens.py`` and ``fit_jlens_olmo.py``.

Outputs (under ``--out``):
- ``timing.json`` (with ``--timing-only``; 1 doc, no ``.pt`` written)
- ``fit.jsonl`` (per-document timing & memory log)
- ``fit_meta.json`` (full metadata, revision pin, row indices, exact fp32 split-half
  diagnostics across all 32 layers)
- ``r_lens.pt`` and ``s_lens.pt`` (when enabled; fp32 accumulators saved in ``--save-dtype``,
  retaining both even/odd halves and ``n_docs`` so future shards can merge cleanly).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
MAX_SEQ_LEN = 128
DEFAULT_SEED = 5701


def set_bos_behavior(tokenizer) -> None:
    if getattr(tokenizer, "bos_token_id", None) is not None and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = True


def load_wikitext_rows(data_path: Path | None) -> tuple[list[str], str]:
    """Load wikitext-103-raw-v1 train text rows from local parquet or HuggingFace datasets."""
    if data_path is not None and data_path.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(data_path, columns=["text"])
        return table.column("text").to_pylist(), str(data_path)
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    return list(ds["text"]), "hf:Salesforce/wikitext/wikitext-103-raw-v1/train"


def select_prompts(
    rows: list[str],
    tokenizer,
    count: int,
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    """Seeded shuffle matching fit_olmo_lens.py: keep rows that tokenize to MAX_SEQ_LEN."""
    if count <= 0:
        raise ValueError("prompt count must be positive")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    token_lists: list[list[int]] = []
    row_indices: list[int] = []
    examined = 0
    for row_index in order:
        raw = rows[row_index]
        if not isinstance(raw, str) or not raw.strip():
            continue
        encoded = tokenizer(raw, truncation=True, max_length=MAX_SEQ_LEN)
        ids = encoded["input_ids"]
        if len(ids) != MAX_SEQ_LEN:
            continue
        token_lists.append(ids)
        row_indices.append(row_index)
        if len(token_lists) == count:
            return token_lists, row_indices
        examined += 1
        if examined % 10000 == 0:
            print(f"selected {len(token_lists)}/{count}  scanned {examined} nonempty rows", flush=True)
    raise RuntimeError(f"only {len(token_lists)} rows tokenize to {MAX_SEQ_LEN} tokens, wanted {count}")


def peak_memory(device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import resource
        out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)
    except Exception:
        pass
    if device.type == "cuda":
        out["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3)
        out["cuda_max_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 1024 ** 3, 3)
    return out


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_accumulator(
    path: Path,
    acc,
    meta: dict[str, Any],
    save_dtype: torch.dtype,
) -> dict[str, list[float]]:
    """Compute exact fp32 split-half diagnostics, then atomically save halves in save_dtype."""
    targets = sorted({t for t, _ in acc.sums})
    sh = {t: [round(x, 6) if x == x else None for x in acc.split_half(t)] for t in targets}
    st = acc.state()
    payload = {
        "meta": meta,
        "split_half_fp32": sh,
        "state": {
            "sums": {k: v.detach().to(device="cpu", dtype=save_dtype) for k, v in st["sums"].items()},
            "n_docs": list(st["n_docs"]),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--data", type=Path, default=None, help="Optional local wikitext parquet")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--start-doc", type=int, default=0, help="Shard offset into seeded prompt list")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--acc-device", choices=("cpu", "cuda"), default="cpu",
                    help="Where fp32 accumulators live (~8.5 GB per lens; use cpu on 24 GB L4)")
    ap.add_argument("--dim-batch", type=int, default=128)
    ap.add_argument("--lenses", choices=("both", "R", "S"), default="both")
    ap.add_argument("--save-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--checkpoint-every", type=int, default=16)
    ap.add_argument("--timing-only", action="store_true", help="Run 1 doc, log timing/VRAM, save no .pt")
    ap.add_argument("--no-compile", action="store_true", help="Disable torch.compile on CUDA")
    args = ap.parse_args()

    if args.timing_only:
        args.docs = 1
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    acc_dev = torch.device(args.acc_device if device.type == "cuda" else "cpu")
    save_dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.save_dtype]

    if args.no_compile:
        import neuron_lens.transport as tr
        tr._step = lambda step, _dev: step

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version

    print(f"Loading tokenizer {args.model}@{args.revision} (transformers {tf_version})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    set_bos_behavior(tokenizer)

    rows, corpus_spec = load_wikitext_rows(args.data)
    total_needed = args.start_doc + args.docs
    all_toks, all_rows = select_prompts(rows, tokenizer, total_needed, args.seed)
    token_lists = all_toks[args.start_doc:total_needed]
    row_indices = all_rows[args.start_doc:total_needed]
    rows_sha = hashlib.sha256(json.dumps(token_lists).encode("utf-8")).hexdigest()
    print(f"Selected {len(token_lists)} docs (start={args.start_doc}, corpus={corpus_spec}, "
          f"rows_sha256={rows_sha[:12]}...)", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval().requires_grad_(False)
    load_s = time.time() - t0
    cfg = model.config
    n_layers, d = cfg.num_hidden_layers, cfg.hidden_size
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"Loaded in {load_s:.1f}s on {device} ({gpu_name}): L={n_layers}, d={d}, "
          f"model_type={cfg.model_type}, acc_dev={acc_dev}, dim_batch={args.dim_batch}", flush=True)

    from neuron_lens.transport import Accumulator, decoder_stack, r_lens_doc, s_lens_doc

    stack = decoder_stack(model)
    do_r = args.lenses in ("both", "R")
    do_s = args.lenses in ("both", "S")
    acc_r = Accumulator({"out": n_layers - 1, "mid": n_layers}, d, device=acc_dev) if do_r else None
    acc_s = Accumulator({"mid": n_layers, "out": n_layers}, d, device=acc_dev) if do_s else None

    per_doc: list[dict[str, Any]] = []
    log_path = args.out / ("timing.jsonl" if args.timing_only else "fit.jsonl")
    t_fit0 = time.time()
    base_meta = {
        "model": args.model,
        "revision": args.revision,
        "model_type": cfg.model_type,
        "model_layers": n_layers,
        "layers": n_layers,
        "d_model": d,
        "seq_len": MAX_SEQ_LEN,
        "seed": args.seed,
        "start_doc": args.start_doc,
        "docs": args.docs,
        "corpus_spec": corpus_spec,
        "row_indices": row_indices,
        "rows_sha256": rows_sha,
        "dim_batch": args.dim_batch,
        "device": str(device),
        "gpu_name": gpu_name,
        "acc_device": str(acc_dev),
        "save_dtype": args.save_dtype,
        "compiled": not args.no_compile and device.type == "cuda",
        "transformers_version": tf_version,
        "torch_version": torch.__version__,
    }

    with open(log_path, "a", encoding="utf-8") as log:
        for i, (toks, row_idx) in enumerate(zip(token_lists, row_indices)):
            ids = torch.tensor(toks, dtype=torch.long)
            half = (args.start_doc + i) % 2
            r_sec = 0.0
            s_sec = 0.0
            if do_r:
                tr0 = time.time()
                r_lens_doc(stack, ids, acc_r, half, dim_batch=args.dim_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                r_sec = time.time() - tr0
            if do_s:
                ts0 = time.time()
                s_lens_doc(stack, ids, acc_s, half, dim_batch=args.dim_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                s_sec = time.time() - ts0
            rec = {
                "doc": args.start_doc + i,
                "row": row_idx,
                "half": half,
                "r_sec": round(r_sec, 3),
                "s_sec": round(s_sec, 3),
                "total_sec": round(r_sec + s_sec, 3),
                **peak_memory(device),
            }
            per_doc.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"doc {i+1}/{len(token_lists)} (row={row_idx}): R={r_sec:.2f}s S={s_sec:.2f}s "
                  f"tot={r_sec+s_sec:.2f}s {peak_memory(device)}", flush=True)

            if (not args.timing_only) and args.checkpoint_every > 0 and ((i + 1) % args.checkpoint_every == 0) and (i + 1) < len(token_lists):
                if do_r:
                    save_accumulator(args.out / "r_lens.pt", acc_r, base_meta | {"lens": "R", "docs_done": i + 1}, save_dt)
                if do_s:
                    save_accumulator(args.out / "s_lens.pt", acc_s, base_meta | {"lens": "S", "docs_done": i + 1}, save_dt)

    fit_sec = time.time() - t_fit0
    summary = {
        **base_meta,
        "load_sec": round(load_s, 2),
        "fit_sec": round(fit_sec, 2),
        "median_r_sec": round(statistics.median([x["r_sec"] for x in per_doc]), 3) if do_r else 0.0,
        "median_s_sec": round(statistics.median([x["s_sec"] for x in per_doc]), 3) if do_s else 0.0,
        "median_total_sec": round(statistics.median([x["total_sec"] for x in per_doc]), 3),
        "peak_memory": peak_memory(device),
        "per_doc": per_doc,
    }
    if args.timing_only:
        atomic_json(args.out / "timing.json", summary)
        print("TIMING_SUMMARY:", json.dumps({k: summary[k] for k in ("gpu_name", "load_sec", "median_r_sec", "median_s_sec", "median_total_sec", "peak_memory")}), flush=True)
        return 0

    split_half = {}
    if do_r:
        sh_r = save_accumulator(args.out / "r_lens.pt", acc_r, base_meta | {"lens": "R", "docs_done": len(token_lists)}, save_dt)
        split_half["R"] = sh_r
    if do_s:
        sh_s = save_accumulator(args.out / "s_lens.pt", acc_s, base_meta | {"lens": "S", "docs_done": len(token_lists)}, save_dt)
        split_half["S"] = sh_s
    summary["split_half_fp32"] = split_half
    atomic_json(args.out / "fit_meta.json", summary)
    print("FIT_DONE:", json.dumps({"docs": len(token_lists), "fit_sec": round(fit_sec, 1), "split_half_mid_L16": {
        "R_out": split_half.get("R", {}).get("out", [None]*17)[16] if do_r else None,
        "S_out": split_half.get("S", {}).get("out", [None]*17)[16] if do_s else None,
    }}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
