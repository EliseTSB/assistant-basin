#!/usr/bin/env python3
"""Compare top-k state between pilot runs (fp32 vs bf16; first-20 vs all-40 docs).

usage: compare_runs.py A.npz B.npz [--k 10] [--width 8192]
Reports, over neurons (units < width) and controls separately:
  median / mean overlap of top-k (doc, pos) sets, the same for doc-only sets,
  and the fraction of units whose top-1 (doc, pos) is identical.
"""
import argparse
import json

import numpy as np


def overlaps(a_doc, a_pos, b_doc, b_pos, k):
    n_units = a_doc.shape[0]
    pair = np.empty(n_units)
    doc = np.empty(n_units)
    for u in range(n_units):
        sa = set(zip(a_doc[u, :k].tolist(), a_pos[u, :k].tolist()))
        sb = set(zip(b_doc[u, :k].tolist(), b_pos[u, :k].tolist()))
        pair[u] = len(sa & sb) / k
        doc[u] = len(set(a_doc[u, :k].tolist()) & set(b_doc[u, :k].tolist())) / k
    top1 = (a_doc[:, 0] == b_doc[:, 0]) & (a_pos[:, 0] == b_pos[:, 0])
    return pair, doc, top1


def compare(a_path, b_path, k=10, width=8192):
    a, b = np.load(a_path), np.load(b_path)
    L, U, _ = a["top_doc"].shape
    out = {}
    for name, sl in (("neurons", slice(0, width)), ("controls", slice(width, U))):
        ad = a["top_doc"][:, sl].reshape(-1, a["top_doc"].shape[-1])
        ap = a["top_pos"][:, sl].reshape(-1, a["top_pos"].shape[-1])
        bd = b["top_doc"][:, sl].reshape(-1, b["top_doc"].shape[-1])
        bp = b["top_pos"][:, sl].reshape(-1, b["top_pos"].shape[-1])
        if not len(ad):
            continue
        pair, doc, top1 = overlaps(ad, ap, bd, bp, k)
        # max agreement (relative difference of the corpus max)
        am, bm = a["max"][:, sl].ravel(), b["max"][:, sl].ravel()
        rel = np.abs(am - bm) / np.maximum(np.abs(am), 1e-6)
        out[name] = {
            "units": int(len(pair)),
            f"top{k}_pair_overlap_median": float(np.median(pair)),
            f"top{k}_pair_overlap_mean": float(pair.mean()),
            f"top{k}_pair_overlap_p10": float(np.quantile(pair, 0.1)),
            f"top{k}_doc_overlap_median": float(np.median(doc)),
            f"top{k}_doc_overlap_mean": float(doc.mean()),
            "top1_identical_frac": float(top1.mean()),
            "max_rel_diff_median": float(np.median(rel)),
            "max_rel_diff_p99": float(np.quantile(rel, 0.99)),
        }
        per_layer = []
        for li in range(L):
            idx = slice(li * (len(pair) // L), (li + 1) * (len(pair) // L))
            per_layer.append(round(float(np.median(pair[idx])), 2))
        out[name]["pair_overlap_median_by_layer"] = per_layer
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--width", type=int, default=8192)
    args = p.parse_args()
    print(json.dumps(compare(args.a, args.b, args.k, args.width), indent=1))
