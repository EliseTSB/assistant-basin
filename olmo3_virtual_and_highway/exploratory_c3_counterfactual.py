#!/usr/bin/env python3
"""exploratory_c3_counterfactual.py — Post-hoc exploratory (NON-CONFIRMATORY) Stage C3 readouts.

Official confirmatory verdicts come exclusively from `score_olmo3_c3.py` (where `C3.0: FAIL`
solely because `C3.0.2` checked `D4` `logit_l2_ratio` = 1.017x / mismask = 1.148x rather than
target `seed_dlogit_ratio` = 1.508x / mismask = 0.966x, blocking `H1–H4` as `BLOCKED_BY_C3.0`).

This pure-stdlib script computes exploratory `would_print` counterfactuals by feeding a patched
in-memory copy of the receipt (substituting `seed_dlogit_ratio` into `logit_l2_ratio` strictly
for this exploratory counterfactual — NEVER modifying `c3_receipt.json.gz` or `score_olmo3_c3.py`)
through the exact same scoring functions in `score_olmo3_c3.py`, plus detailed D1/D4/D5/D6/dose
diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c3_scoring  # noqa: E402
import score_olmo3_c3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Exploratory (non-confirmatory) Stage C3 counterfactual & diagnostics.")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--reader-labels", type=Path, required=True)
    ap.add_argument("--c3-constants", type=Path, default=HERE / "c3_prereg_constants.json")
    ap.add_argument("--design", type=Path, default=HERE / "OLMO3-WSP-C3-DESIGN.md")
    ap.add_argument("--bank", type=Path, default=HERE / "c3_vector_bank.safetensors")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    receipt = score_olmo3_c3.load_json_or_gz(args.receipt)
    reader_labels = json.loads(args.reader_labels.read_text(encoding="utf-8"))
    c3_constants = json.loads(args.c3_constants.read_text(encoding="utf-8"))
    receipt_sha = score_olmo3_c3.sha256_file(args.receipt)

    # 1. Counterfactual: evaluate D4 using target seed_dlogit_ratio (1.508x / 0.966x) instead of total logit_l2_ratio
    cf_receipt = copy.deepcopy(receipt)
    d4_orig = copy.deepcopy(receipt["preflights"]["D4_L24_mask_mismask"])
    cf_receipt["preflights"]["D4_L24_mask_mismask"]["masked_over_unmasked_logit_l2_ratio"] = d4_orig[
        "masked_over_unmasked_seed_dlogit_ratio"
    ]
    cf_receipt["preflights"]["D4_L24_mask_mismask"]["mismasked_over_unmasked_logit_l2_ratio"] = d4_orig[
        "mismasked_over_unmasked_seed_dlogit_ratio"
    ]

    disk_hashes = {
        "c3_design_sha256": score_olmo3_c3.sha256_file(args.design),
        "c3_constants_sha256": score_olmo3_c3.sha256_file(args.c3_constants),
        "c3_bank_sha256": score_olmo3_c3.sha256_file(args.bank),
        "c3_scoring_sha256": score_olmo3_c3.sha256_file(args.scoring),
    }

    cf_scorecard = score_olmo3_c3.score_stage_c3(
        cf_receipt,
        c3_constants,
        reader_labels=reader_labels,
        disk_hashes=disk_hashes,
        receipt_sha256=receipt_sha,
        n_perm=100_000,
        n_boot=10_000,
        strict_provenance=True,
        run_code_subprocess=True,
        allow_mock_readers=False,
    )

    # 2. Extra mechanical diagnostics on D1 copula, D5 multi-turn, D6 adaptive/donor, and dose/KL budget
    d1_arms = receipt["D1_copula"]["arms"]
    d1_summary: dict[str, Any] = {}
    plain_d1 = {r["id"]: r for r in d1_arms["Plain"]}
    for arm_k, rows in d1_arms.items():
        s_vals = [float(r["lp_self"]) for r in rows]
        t_vals = [float(r["lp_stranger"]) for r in rows]
        d_vals = [float(r["self_minus_stranger"]) for r in rows]
        ds_vs_plain = [float(r["lp_self"]) - float(plain_d1[r["id"]]["lp_self"]) for r in rows]
        dt_vs_plain = [float(r["lp_stranger"]) - float(plain_d1[r["id"]]["lp_stranger"]) for r in rows]
        dd_vs_plain = [float(r["self_minus_stranger"]) - float(plain_d1[r["id"]]["self_minus_stranger"]) for r in rows]
        d1_summary[arm_k] = {
            "n": len(rows),
            "mean_lp_self": round(math.fsum(s_vals) / len(s_vals), 6),
            "mean_lp_stranger": round(math.fsum(t_vals) / len(t_vals), 6),
            "mean_self_minus_stranger": round(math.fsum(d_vals) / len(d_vals), 6),
            "delta_lp_self_vs_plain": round(math.fsum(ds_vs_plain) / len(ds_vs_plain), 6),
            "delta_lp_stranger_vs_plain": round(math.fsum(dt_vs_plain) / len(dt_vs_plain), 6),
            "delta_copula_diff_vs_plain": round(math.fsum(dd_vs_plain) / len(dd_vs_plain), 6),
            "pos_lp_self_vs_plain_count": sum(1 for x in ds_vs_plain if x > 0.0),
        }

    # D5 mechanical stem hits by turn
    surf_re = str(c3_constants["lexicons"]["common_pirate_ban"]["surface_stem_regex"])
    d5_summary: dict[str, Any] = {}
    for arm_k, items in receipt["D5_multiturn"].items():
        by_turn = {}
        for t in (1, 2, 3):
            trows = [it["turns"][t - 1] for it in items]
            by_turn[f"turn_{t}"] = {
                "stem_hits": sum(1 for r in trows if c3_scoring.surface_stem_hit(r["text"], surf_re)),
                "broken": sum(1 for r in trows if r["is_broken"]),
                "mean_tok": round(math.fsum(r["n_tokens"] for r in trows) / len(trows), 2),
                "n": len(trows),
            }
        d5_summary[arm_k] = by_turn

    # D6 own top-50 summary & mechanical ban audit
    d6_own_top10 = {
        arm_k: [
            {"token": r["token"], "mean_dlogit": r["mean_dlogit"]}
            for r in rows[:10]
        ]
        for arm_k, rows in receipt["D6_adaptive_donor"]["own_top50"].items()
    }
    d6_cond_mech = {
        ck: {
            "n": len(rows),
            "stem_hits": sum(1 for r in rows if c3_scoring.surface_stem_hit(r["text"], surf_re)),
            "broken": sum(1 for r in rows if r["is_broken"]),
            "mean_tok": round(math.fsum(r["n_tokens"] for r in rows) / len(rows), 2),
        }
        for ck, rows in receipt["D6_adaptive_donor"]["conditions"].items()
    }

    # Dose & KL headroom autopsy
    kl_headroom: dict[str, Any] = {}
    for arm_k, arm_blk in receipt["commit_gate"]["arms"].items():
        r08 = [rg for rg in arm_blk["rungs"] if abs(float(rg["dose"]) - 0.80) < 1e-9][0]
        kl_headroom[arm_k] = {
            "d_star": arm_blk["selected_d_star"],
            "reserve_med_off_kl_at_0.80": r08["median_off_kl"],
            "kl_budget_ratio_0.25_over_actual": round(0.25 / max(float(r08["median_off_kl"]), 1e-12), 2),
            "alphas_at_0.80": r08["alphas"],
            "min_stored_rank1_proj_ret": min(v["stored_rank1_proj_ret"] for v in r08["bf16_retention"].values()),
            "min_stored_frob_ret": min(v["stored_frob_ret"] for v in r08["bf16_retention"].values()),
        }

    # Pillar A unbanned stem hit example (the 1/40 item for WSP_Pirate_Contrastive)
    pa_con_hits = [
        {"id": r["id"], "text": r["text"], "n_tokens": r["n_tokens"]}
        for r in receipt["pillar_a"]["none"]["WSP_Pirate_Contrastive@d*_con"]
        if r["surface_stem_hit"]
    ]

    out = {
        "disclaimer": "EXPLORATORY / NON-CONFIRMATORY COUNTERFACTUAL ONLY. Official confirmatory verdict is C3.0: FAIL / H1-H4: BLOCKED_BY_C3.0.",
        "counterfactual_reason": "Evaluated D4 with target seed_dlogit_ratio (1.508x / 0.966x) instead of total vocab logit_l2_ratio (1.017x / 1.148x).",
        "d4_original": {
            "masked_over_unmasked_logit_l2_ratio": d4_orig["masked_over_unmasked_logit_l2_ratio"],
            "mismasked_over_unmasked_logit_l2_ratio": d4_orig["mismasked_over_unmasked_logit_l2_ratio"],
            "masked_over_unmasked_seed_dlogit_ratio": d4_orig["masked_over_unmasked_seed_dlogit_ratio"],
            "mismasked_over_unmasked_seed_dlogit_ratio": d4_orig["mismasked_over_unmasked_seed_dlogit_ratio"],
            "masked_mean_pirate_seed_dlogit": d4_orig["masked"]["mean_pirate_seed_dlogit"],
            "unmasked_mean_pirate_seed_dlogit": d4_orig["unmasked"]["mean_pirate_seed_dlogit"],
            "mismasked_mean_pirate_seed_dlogit": d4_orig["mismasked"]["mean_pirate_seed_dlogit"],
        },
        "counterfactual_would_print_verdicts": cf_scorecard["confirmatory_verdicts"],
        "counterfactual_gates": cf_scorecard["gates"],
        "counterfactual_holm_4": cf_scorecard["holm_4_family"],
        "counterfactual_diagnostics": cf_scorecard["diagnostics"],
        "extra_diagnostics": {
            "D1_copula_summary": d1_summary,
            "D5_multiturn_mech": d5_summary,
            "D6_own_top10_tokens": d6_own_top10,
            "D6_conditions_mech": d6_cond_mech,
            "dose_kl_headroom_autopsy": kl_headroom,
            "pillar_a_con_unbanned_stem_hits": pa_con_hits,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote exploratory counterfactual -> {args.out_json}")
    print("would_print verdicts:", json.dumps(out["counterfactual_would_print_verdicts"], indent=2))
    print("H3:", json.dumps(out["counterfactual_holm_4"]["hypotheses"]["H3"], indent=2))
    print("H4:", json.dumps(out["counterfactual_holm_4"]["hypotheses"]["H4"], indent=2))
    print("EQ-C3.2:")
    for arm_k, am in sorted(out["counterfactual_gates"]["EQ_C3_2"].items()):
        print(
            f"  {arm_k:<36} gate={str(am['gating']):<5} math={am['math_acc']:.3f}({am['math_strict_final_acc']:.3f}) "
            f"code={am['code_pass_rate']:.3f} kl={am['neutral_median_off_kl']:.4f} brk={am['neutral_broken_rate']:.3f} pass={am['pass']}"
        )
    print("D1 Copula:", json.dumps(d1_summary, indent=2))
    print("D6 Top-10 Own Boosted Tokens:", json.dumps(d6_own_top10, indent=2))
    print("Pillar A Con unbanned stem hit:", json.dumps(pa_con_hits, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
