#!/usr/bin/env python3
"""Fit the Neuron Lens R-lens and S-lens on one OLMo 2 1B checkpoint.

One pass per document through both lens directions (reverse mode for the R-lens,
forward mode for the S-lens), all layers, both targets (``out`` and ``mid``) —
the one-layer acceptance fit is layer 8's pair, but the pass covers every layer
at no extra cost.

Devices: ``cpu`` (the acceptance fit; bf16 weights, fp32 accumulators, cgroup
cap) and ``xpu`` (the Intel A770, 16 GB; bf16, no float64 anywhere). On XPU the
accumulators stay on the device and are copied to the CPU only when the fit is
saved. ``--timing-only`` runs one document and writes seconds per lens and the
device's peak memory, which is what the Arc queue needs before a full stage.

Corpus: the first rows of a seeded shuffle of wikitext-103 train (the same
selection as ``fit_jlens_olmo.py``: rows whose text tokenises to exactly 128
tokens, seed 5701).

Writes ``<out>/fit.pt`` (accumulator state, CPU tensors) and ``<out>/fit.json``
(metadata, per-document seconds, split-half by layer, peak memory); with
``--timing-only``, ``<out>/timing.json`` instead and no ``.pt``.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))  # the Arc runner uses python -I, which leaves the script folder off the path
MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
MAX_SEQ_LEN = 128
DEFAULT_SEED = 5701


def default_corpus() -> Path:
    return Path("E:/neuron-probe/corpus/wikitext103-train0.parquet") if os.name == "nt" \
        else Path("/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet")


def default_cache() -> Path:
    return Path("E:/neuron-probe/hf/hub") if os.name == "nt" else Path("/mnt/e/neuron-probe/hf/hub")


def set_bos_behavior(tokenizer) -> None:
    if getattr(tokenizer, "bos_token_id", None) is not None and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = True


def select_prompts(corpus_path: Path, tokenizer, count: int, seed: int):
    """A seeded shuffle of the parquet's rows; keep rows that tokenise to exactly MAX_SEQ_LEN.

    Copied from ``fit_jlens_olmo.select_prompts`` (same seed and filter), so this fit reads the
    same documents as the house's earlier OLMo 2 Jacobian work.
    """
    import pyarrow.parquet as pq

    if count <= 0:
        raise ValueError("prompt count must be positive")
    table = pq.read_table(corpus_path, columns=["text"])
    rows = table.column("text").to_pylist()
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    prompts, row_indices = [], []
    examined = 0
    for row_index in order:
        raw = rows[row_index]
        if not isinstance(raw, str) or not raw.strip():
            continue
        encoded = tokenizer(raw, truncation=True, max_length=MAX_SEQ_LEN)
        if len(encoded["input_ids"]) != MAX_SEQ_LEN:
            continue
        prompts.append(raw)
        row_indices.append(row_index)
        if len(prompts) == count:
            return prompts, row_indices
        examined += 1
        if examined % 10000 == 0:
            print(f"selected {len(prompts)}/{count}  scanned {examined} nonempty rows", flush=True)
    raise RuntimeError(f"only {len(prompts)} rows tokenise to {MAX_SEQ_LEN} tokens, wanted {count}")


def peak_rss_gb() -> float | None:
    """The process's own high-water RSS (Linux); ``None`` where unsupported (Windows)."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)
    except Exception:
        return None


def peak_memory(device: torch.device) -> dict:
    """The peak this fit actually used: the cgroup's high-water mark on CPU,
    the XPU allocator's on the Arc."""
    if device.type == "xpu":
        return {"xpu_max_memory_allocated_gb": round(torch.xpu.max_memory_allocated() / 1024 ** 3, 3),
                "xpu_max_memory_reserved_gb": round(torch.xpu.max_memory_reserved() / 1024 ** 3, 3)}
    path = Path("/sys/fs/cgroup/memory.peak")
    if path.exists():
        return {"cgroup_memory_peak_gb": round(int(path.read_text().strip()) / 1024 ** 3, 3)}
    return {}


def cpu_state(acc) -> dict:
    st = acc.state()
    return {"sums": {k: v.cpu() for k, v in st["sums"].items()}, "n_docs": list(st["n_docs"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--cache-dir", type=Path, default=default_cache())
    ap.add_argument("--data", type=Path, default=default_corpus())
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "prime-lens" / "olmo2-1b-instruct")
    ap.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "12")))
    ap.add_argument("--dim-batch", type=int, default=64,
                    help="basis directions per block; 64 on the 8 GB CPU cap (256 leaves no headroom)")
    ap.add_argument("--timing-only", action="store_true",
                    help="one document, seconds per lens + peak memory, no fit.pt")
    ap.add_argument("--print-only", action="store_true", help="select documents, print their rows, load no model")
    args = ap.parse_args()

    if args.timing_only:
        args.docs = 1
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            ap.error("torch.xpu is unavailable; run in the Windows XPU Python environment")
        try:
            free, total = torch.xpu.mem_get_info()
            print(f"xpu: {free/1024**3:.1f} GB free of {total/1024**3:.1f} GB", flush=True)
        except Exception as exc:  # informational only
            print(f"xpu: memory query unavailable ({exc})", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir), local_files_only=True)
    set_bos_behavior(tokenizer)
    prompts, row_indices = select_prompts(args.data, tokenizer, args.docs, args.seed)
    print(f"selected {len(prompts)} documents, rows {row_indices}", flush=True)
    if args.print_only:
        print(json.dumps({"rows": row_indices, "corpus": str(args.data),
                          "corpus_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()}))
        return 0

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval().requires_grad_(False)
    load_s = time.time() - t0
    cfg = model.config
    n_layers, d = cfg.num_hidden_layers, cfg.hidden_size
    print(f"model loaded in {load_s:.1f}s on {device}: {n_layers} layers, d={d}, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.3f}B", flush=True)

    from neuron_lens.transport import Accumulator, decoder_stack, r_lens_doc, s_lens_doc

    stack = decoder_stack(model)
    acc_r = Accumulator({"out": n_layers - 1, "mid": n_layers}, d, device=device)
    acc_s = Accumulator({"mid": n_layers, "out": n_layers}, d, device=device)
    per_doc = []
    log_path = args.out / ("timing.jsonl" if args.timing_only else "fit.jsonl")
    with open(log_path, "a", encoding="utf-8") as log:
        for i, (text, row) in enumerate(zip(prompts, row_indices)):
            ids = torch.tensor(tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"])
            if len(ids) != MAX_SEQ_LEN:
                raise RuntimeError(f"document {i} tokenised to {len(ids)} tokens")
            half = i % 2
            t_r0 = time.time()
            r_lens_doc(stack, ids, acc_r, half, dim_batch=args.dim_batch)
            if device.type == "xpu":
                torch.xpu.synchronize()
            t_r1 = time.time()
            s_lens_doc(stack, ids, acc_s, half, dim_batch=args.dim_batch)
            if device.type == "xpu":
                torch.xpu.synchronize()
            t_s1 = time.time()
            rec = {"doc": i, "row": row, "half": half, "tokens": int(len(ids)),
                   "r_seconds": round(t_r1 - t_r0, 3), "s_seconds": round(t_s1 - t_r1, 3),
                   "seconds": round(t_s1 - t_r0, 3), "peak_rss_gb": peak_rss_gb(),
                   **peak_memory(device)}
            per_doc.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"doc {i}: R {rec['r_seconds']:.1f}s + S {rec['s_seconds']:.1f}s "
                  f"= {rec['seconds']:.1f}s", flush=True)

    meta = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "dtype": "bfloat16",
        "device": str(device),
        "threads": args.threads,
        "dim_batch": args.dim_batch,
        "layers": n_layers,
        "hidden_size": d,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None) or d // cfg.num_attention_heads,
        "corpus": str(args.data),
        "corpus_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "selection_seed": args.seed,
        "selection": "seeded shuffle; keep rows tokenising to exactly 128 tokens",
        "seq_len": MAX_SEQ_LEN,
        "rows": row_indices,
        "docs": len(per_doc),
        "load_seconds": round(load_s, 3),
        "per_document": per_doc,
        "r_seconds_median": statistics.median(r["r_seconds"] for r in per_doc),
        "s_seconds_median": statistics.median(r["s_seconds"] for r in per_doc),
        "seconds_median": statistics.median(r["seconds"] for r in per_doc),
        "peak_memory": peak_memory(device),
        "peak_rss_gb": peak_rss_gb(),
        "torch": torch.__version__,
        "transformers": tf_version,
        "n_docs": list(acc_r.n_docs),
        "split_half": {
            "R_out": acc_r.split_half("out"), "R_mid": acc_r.split_half("mid"),
            "S_out": acc_s.split_half("out"), "S_mid": acc_s.split_half("mid"),
        },
    }
    if args.timing_only:
        (args.out / "timing.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out/'timing.json'}; {meta['seconds_median']:.1f}s for one document", flush=True)
        return 0
    torch.save({"meta": meta, "state_r": cpu_state(acc_r), "state_s": cpu_state(acc_s)},
               args.out / "fit.pt")
    (args.out / "fit.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out/'fit.pt'} and fit.json; median {meta['seconds_median']:.1f}s/doc", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
