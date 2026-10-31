#!/usr/bin/env python3
"""Fit the Neuron Lens R-lens and S-lens on one OLMo 2 1B checkpoint, CPU only.

One pass per document through both lens directions (reverse mode for the R-lens,
forward mode for the S-lens), all layers, both targets (``out`` and ``mid``) —
the one-layer acceptance fit is layer 8's pair, but the pass covers every layer
at no extra cost. bf16 weights, fp32 accumulators, eager attention; the
estimator is the vendored ``neuron_lens`` package with the OLMo 2 maps added.

Corpus: the first rows of a seeded shuffle of wikitext-103 train (the same
selection as ``fit_jlens_olmo.py``: rows whose text tokenises to exactly 128
tokens, seed 5701).

Writes ``<out>/fit.pt`` (accumulator state) and ``<out>/fit.json`` (metadata,
per-document seconds, split-half by layer, peak RSS).
"""
import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
MAX_SEQ_LEN = 128
DEFAULT_SEED = 5701
DEFAULT_CORPUS = Path("/mnt/e/neuron-probe/corpus/wikitext103-train0.parquet")
DEFAULT_CACHE = Path("/mnt/e/neuron-probe/hf/hub")


def default_corpus() -> Path:
    return DEFAULT_CORPUS


def default_cache() -> Path:
    return DEFAULT_CACHE


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


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--cache-dir", type=Path, default=default_cache())
    ap.add_argument("--data", type=Path, default=default_corpus())
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "prime-lens" / "olmo2-1b-instruct")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "12")))
    ap.add_argument("--dim-batch", type=int, default=256)
    ap.add_argument("--print-only", action="store_true", help="select documents, print their rows, load no model")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir), local_files_only=True)
    set_bos_behavior(tokenizer)
    prompts, row_indices = select_prompts(args.data, tokenizer, args.docs, args.seed)
    print(f"selected {len(prompts)} documents, rows {row_indices}", flush=True)
    if args.print_only:
        print(json.dumps({"rows": row_indices, "corpus": str(args.data),
                          "corpus_sha256": __import__("hashlib").sha256(args.data.read_bytes()).hexdigest()}))
        return 0

    dtype_name = "bfloat16"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().requires_grad_(False)
    load_s = time.time() - t0
    cfg = model.config
    n_layers, d = cfg.num_hidden_layers, cfg.hidden_size
    print(f"model loaded in {load_s:.1f}s: {n_layers} layers, d={d}, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.3f}B, "
          f"peak rss {peak_rss_gb():.2f} GB", flush=True)

    from neuron_lens.transport import Accumulator, decoder_stack, r_lens_doc, s_lens_doc
    stack = decoder_stack(model)

    acc_r = Accumulator({"out": n_layers - 1, "mid": n_layers}, d)
    acc_s = Accumulator({"mid": n_layers, "out": n_layers}, d)
    per_doc = []
    log_path = args.out / "fit.jsonl"
    with open(log_path, "a", encoding="utf-8") as log:
        for i, (text, row) in enumerate(zip(prompts, row_indices)):
            ids = torch.tensor(tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"])
            if len(ids) != MAX_SEQ_LEN:
                raise RuntimeError(f"document {i} tokenised to {len(ids)} tokens")
            half = i % 2
            t_r0 = time.time()
            r_lens_doc(stack, ids, acc_r, half, dim_batch=args.dim_batch)
            t_r1 = time.time()
            s_lens_doc(stack, ids, acc_s, half, dim_batch=args.dim_batch)
            t_s1 = time.time()
            rec = {"doc": i, "row": row, "half": half, "tokens": int(len(ids)),
                   "r_seconds": round(t_r1 - t_r0, 3), "s_seconds": round(t_s1 - t_r1, 3),
                   "seconds": round(t_s1 - t_r0, 3), "peak_rss_gb": round(peak_rss_gb(), 3)}
            per_doc.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"doc {i}: R {rec['r_seconds']:.1f}s + S {rec['s_seconds']:.1f}s "
                  f"= {rec['seconds']:.1f}s  (peak rss {rec['peak_rss_gb']:.2f} GB)", flush=True)

    meta = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "dtype": dtype_name,
        "device": "cpu",
        "threads": args.threads,
        "dim_batch": args.dim_batch,
        "layers": n_layers,
        "hidden_size": d,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None) or d // cfg.num_attention_heads,
        "corpus": str(args.data),
        "corpus_sha256": __import__("hashlib").sha256(args.data.read_bytes()).hexdigest(),
        "selection_seed": args.seed,
        "selection": "seeded shuffle; keep rows tokenising to exactly 128 tokens",
        "seq_len": MAX_SEQ_LEN,
        "rows": row_indices,
        "docs": len(per_doc),
        "load_seconds": round(load_s, 3),
        "per_document": per_doc,
        "r_seconds_median": sorted(r["r_seconds"] for r in per_doc)[len(per_doc) // 2],
        "s_seconds_median": sorted(r["s_seconds"] for r in per_doc)[len(per_doc) // 2],
        "seconds_median": sorted(r["seconds"] for r in per_doc)[len(per_doc) // 2],
        "peak_rss_gb": round(peak_rss_gb(), 3),
        "torch": torch.__version__,
        "transformers": tf_version,
        "n_docs": acc_r.n_docs,
        "split_half": {
            "R_out": acc_r.split_half("out"), "R_mid": acc_r.split_half("mid"),
            "S_out": acc_s.split_half("out"), "S_mid": acc_s.split_half("mid"),
        },
    }
    torch.save({"meta": meta, "state_r": acc_r.state(), "state_s": acc_s.state()}, args.out / "fit.pt")
    (args.out / "fit.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out/'fit.pt'} and fit.json; median {meta['seconds_median']:.1f}s/doc", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
