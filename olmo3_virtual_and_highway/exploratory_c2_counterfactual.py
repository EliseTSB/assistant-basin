#!/usr/bin/env python3
"""Standalone stdlib-only exploratory counterfactual driver for Stage C2 (OLMo 3 7B).

IMPORTANT: Official confirmatory Stage C2 verdict is `C2.0: FAIL` and
`C2.1, C2.2, C2.3, C2.4a, EQ-C2.4c: BLOCKED_BY_C2.0` (produced by unmodified
`score_olmo3_c2.py` on frozen registered v4 design/constants).

This script reproduces every diagnostic and exploratory counterfactual number in
`OLMO3-TRANSPORT-C2-REPORT.md` without modifying any frozen file on disk:

  1. Exact `C2.0.2` telescoping breakdown across Block 1 (2,880), each of the 9
     Block 2 C2.2 conditions (288 each), Block 2b C2.3 (960), and all 6,144
     non-`ctrl_L4_16` traces combined (`p95 = 0.0506`, `p99.9 = 0.0931`, `max = 0.1274`)
     vs. `ctrl_L4_16_MLP_LongBare_to_ChatML` alone (`p50 = 0.3267`, `p99.9 = 1.1872`),
     where patching MLPs at L4-L16 (before inject_layer=18) produces residual shifts
     excluded from `decompose_target_trace(..., inject_layer=18)`'s L19-L31 sum.
  2. Counterfactual gate-open readout: zeroes ONLY `telescoping_err` on
     `ctrl_L4_16_MLP_LongBare_to_ChatML` in memory (keeping `strict_provenance=True`
     against frozen disk SHA-256s) so `score_stage_c2()` proceeds past its early
     return. Invariance: `telescoping_err` is purely a `C2.0.2` gate field and
     never enters any Pillar 1-4 metric, Holm p-value, or BCa CI.
  3. Report-side breakdowns: per-layer C2.1 held-out cross ratios & `mu_u_cos`,
     C2.2 per-condition mean `M(C)` (exposing the `+0.424` length effect vs. `-0.313`
     matched-length ChatML shift), C2.3 per-layer open/func/rand retention, and
     C2.4a/C2.4b/EQ-C2.4c exact numbers.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olmo3_virtual_and_highway import score_olmo3_c2 as sc  # noqa: E402


def _qstats(xs: list[float]) -> str:
    vals = [abs(float(x)) for x in xs]
    return (
        f"N={len(vals)} p50={sc.quantile(vals, 0.5):.4f} p95={sc.quantile(vals, 0.95):.4f} "
        f"p99={sc.quantile(vals, 0.99):.4f} p99.9={sc.quantile(vals, 0.999):.4f} max={max(vals):.4f}"
    )


def main() -> None:
    receipt_p = REPO_ROOT / "results" / "vesper-olmo3-7b-stage-c2" / "c2_receipt.json.gz"
    const_p = REPO_ROOT / "olmo3_virtual_and_highway" / "c2_prereg_constants.json"
    design_p = REPO_ROOT / "olmo3_virtual_and_highway" / "OLMO3-TRANSPORT-C2-DESIGN.md"

    gz_bytes = receipt_p.read_bytes()
    raw_bytes = gzip.decompress(gz_bytes)
    print("=== 0. RECEIPT PROVENANCE ===")
    print(f"gz_bytes={len(gz_bytes)} gz_sha256={hashlib.sha256(gz_bytes).hexdigest()}")
    print(f"raw_bytes={len(raw_bytes)} raw_sha256={hashlib.sha256(raw_bytes).hexdigest()}")

    rec = json.loads(raw_bytes.decode("utf-8"))
    c2c = json.loads(const_p.read_text(encoding="utf-8"))
    t_strs = [t["string"] for t in c2c["targets_24"]]
    strata = [t["stratum"] for t in c2c["targets_24"]]
    c22_t_strs = [t_strs[i] for i in c2c["c22_target_indices_12"]]
    c22_conds = c2c["planned_budget"]["c22_conditions_list"]

    print(f"meta: total_forwards={rec['meta']['total_forwards']} elapsed_sec={rec['meta']['elapsed_sec']} device={rec['meta']['device']}")
    print("forward_counts:", json.dumps(rec["meta"]["forward_counts"]))
    print("live post_mlp_norms L18/21/24/27/30:", {L: rec["post_mlp_norms"][L] for L in ("18", "21", "24", "27", "30")})

    print("\n=== 1. C2.0.2 TELESCOPING AUTOPSY ===")
    b1 = [r["telescoping_err"] for L in (18, 24, 27) for a in sc.C21_ARMS for ts in t_strs for r in rec["block1_C2_1"][str(L)]["arms"][a][ts]]
    print("Block 1 (C2.1):                         ", _qstats(b1))
    b2_no_l4: list[float] = []
    for cname in c22_conds:
        xs = [rec["block2_C2_2"]["traces"][cname][str(p)][ts][a]["telescoping_err"] for p in range(12) for ts in c22_t_strs for a in ("R_mf", "dm0")]
        print(f"Block 2 ({cname:36s}): {_qstats(xs)}")
        if cname != "ctrl_L4_16_MLP_LongBare_to_ChatML":
            b2_no_l4.extend(xs)
    b2b = [r["telescoping_err"] for L in (18, 21, 24, 27, 30) for ts in t_strs for r in rec["block2b_C2_3"][str(L)]["logit_lens"][ts]]
    print("Block 2b (C2.3):                        ", _qstats(b2b))
    print("ALL 6,144 EXCLUDING ctrl_L4_16:         ", _qstats(b1 + b2_no_l4 + b2b))

    print("\n=== 2. EXPLORATORY COUNTERFACTUAL SCORECARD (NON-CONFIRMATORY!) ===")
    rec_cf = copy.deepcopy(rec)
    for p in range(12):
        for ts in c22_t_strs:
            for a in ("R_mf", "dm0"):
                rec_cf["block2_C2_2"]["traces"]["ctrl_L4_16_MLP_LongBare_to_ChatML"][str(p)][ts][a]["telescoping_err"] = 0.0
    design_sha = sc.sha256_file(design_p)
    const_sha = sc.sha256_file(const_p)
    cf = sc.score_stage_c2(rec_cf, c2c, design_sha256=design_sha, constants_sha256=const_sha, strict_provenance=True)
    print("Exploratory counterfactual verdicts:", json.dumps(cf["confirmatory_verdicts"]))
    print("Exploratory Holm-7 family:", json.dumps(cf["holm_7_family"], indent=2))
    print("Exploratory C0-colon anchor pred:", json.dumps(cf["c0_anchor_colon_prediction_nongating"]))
    print("Exploratory Pillar C2.1:", json.dumps(cf["pillar_C2_1"], indent=2))
    print("Exploratory Pillar C2.2:", json.dumps(cf["pillar_C2_2"], indent=2))
    print("Exploratory Pillar C2.3:", json.dumps(cf["pillar_C2_3"], indent=2))
    print("Exploratory Pillar C2.4:", json.dumps(cf["pillar_C2_4"], indent=2))

    print("\n=== 3. REPORT-SIDE DIAGNOSTICS ===")
    # 3a. C2.2 per-condition mean M(C) across 12 stem pairs
    pr_map = c2c["c22_l18_frozen_pred_ret"]
    pred_gap_mean_12 = sc.mean([float(pr_map[ts]["pred_gap_at_direct3"]) for ts in c22_t_strs])
    print("C2.2 per-condition mean M(C) over 12 pairs:")
    for cname in c22_conds:
        ms = []
        for p in range(12):
            gaps = [
                float(rec["block2_C2_2"]["traces"][cname][str(p)][ts]["R_mf"]["exact_dlogit"])
                - float(rec["block2_C2_2"]["traces"][cname][str(p)][ts]["dm0"]["exact_dlogit"])
                for ts in c22_t_strs
            ]
            ms.append(sc.mean(gaps) / pred_gap_mean_12)
        print(f"  {cname:38s}: mean={sc.mean(ms):.4f} med={sc.median(ms):.4f} range=[{min(ms):.4f}, {max(ms):.4f}]")

    # 3b. C2.1 per-layer held-out cross_abs/diag means across the 5 arms
    prim_idxs = [i for i, t in enumerate(c2c["targets_24"]) if t["split"] == "primary_16"]
    ho_idxs = [i for i, t in enumerate(c2c["targets_24"]) if t["split"] == "heldout_8"]
    print("\nC2.1 per-layer held-out cross_abs/diag (primary_16 -> heldout_8 cross-stratum):")
    for L in (18, 24, 27):
        parts = []
        for a in sc.C21_ARMS:
            vals = []
            for i in prim_idxs:
                rows = rec["block1_C2_1"][str(L)]["arms"][a][t_strs[i]]
                diag_iL = sc.mean([float(r["all_24_dlogits"][i]) for r in rows])
                ho_abs = sc.mean([abs(float(r["all_24_dlogits"][j])) for r in rows for j in ho_idxs if strata[j] != strata[i]])
                vals.append(ho_abs / max(abs(diag_iL), 0.25))
            parts.append(f"{a}={sc.mean(vals):.4f}")
        print(f"  L{L}: " + "  ".join(parts))

    # 3c. C2.4a per-stratum masked/unmasked geometric mean ratio
    print("\nC2.4a per-stratum L24 masked / unmasked ratio:")
    for st in ("nouns", "verbs", "adjectives", "function"):
        idxs = [i for i, s in enumerate(strata) if s == st]
        rats = []
        for i in idxs:
            ts = t_strs[i]
            dm = sc.mean([float(r["exact_dlogit"]) for m in ("0.5", "1.25") for r in rec["block3_C2_4"]["doses"][m]["L24.prenorm_ff_masked"][ts]])
            du = sc.mean([float(r["exact_dlogit"]) for m in ("0.5", "1.25") for r in rec["block3_C2_4"]["doses"][m]["L24.prenorm_ff_unmasked"][ts]])
            rats.append(dm / du)
        geo = math.exp(sc.mean([math.log(x) for x in rats]))
        print(f"  {st:12s}: geo={geo:.4f}x range=[{min(rats):.4f}, {max(rats):.4f}]")


if __name__ == "__main__":
    main()
