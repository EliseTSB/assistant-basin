#!/usr/bin/env python3
"""Standalone stdlib-only exploratory counterfactual driver for Stage C1 (OLMo 3 7B).

IMPORTANT: Official confirmatory Stage C1 verdict is `C1.0: FAIL` and
`C1.1..C1.4: BLOCKED_BY_C1.0` (produced by unmodified `score_olmo3_c1.py` on
frozen `c1_prereg_constants.json`).

This script is strictly for reproducing the post-hoc exploratory numbers in
`OLMO3-TRANSPORT-C1-REPORT.md` §1 (anchor decomposition) and §2 (blocked metric
readouts) without modifying any frozen file on disk:

  - Variant A: substitute measured Stage C0/C0b `post_mlp_norms` into the
    expected `dose_D3_over_pmn_median` anchors -> Gate `C1.0` STILL returns
    `FAIL` (5-key blocked scorecard; `L24` +34.6%, `L27` +44.4% > ±30%).
  - Variant B: force Gate `C1.0.3`'s `anchor_all_ok` check open *by
    construction* (`expected := observed` `median(dose/pmn)` for L18/L24/L27 in
    memory only, with `strict_provenance=True` on disk hashes) so `score_c1()`
    proceeds past its early return at `score_olmo3_c1.py:795-808`. All §2
    statistics (`C1.1`..`C1.4`, Holm, `4x4` matrices) are invariant to this
    substitution because `dose_D3_over_pmn_median` is purely a gate key.
  - Report-side §2 breakdowns: `"all"` vs. `"last"` `obs/pred` format split,
    `p_clean` quantiles via `score_olmo3_c1.quantile()`, 24-target `open x func`
    dispersion via `score_olmo3_c1.quantile()`, per-layer `F_mean` / `vJv` /
    `C1.4 d_t`, and secondary `L12` (`0.30x`) arm breakdown.
"""
from __future__ import annotations

import copy
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import olmo3_virtual_and_highway.score_olmo3_c1 as sc  # noqa: E402


def main() -> None:
    receipt_p = REPO_ROOT / "results" / "vesper-olmo3-7b-stage-c1" / "c1_receipt.json.gz"
    c0_receipt_p = REPO_ROOT / "results" / "vesper-olmo3-7b-stage-c0" / "c0_receipt.json.gz"
    const_p = REPO_ROOT / "olmo3_virtual_and_highway" / "c1_prereg_constants.json"
    design_p = REPO_ROOT / "olmo3_virtual_and_highway" / "OLMO3-TRANSPORT-C1-DESIGN.md"

    with gzip.open(receipt_p, "rt", encoding="utf-8") as f:
        r = json.load(f)
    with gzip.open(c0_receipt_p, "rt", encoding="utf-8") as f:
        c0 = json.load(f)
    c = json.loads(const_p.read_text(encoding="utf-8"))

    t_meta = r["targets_24"]
    t_strs = [t["string"] for t in t_meta]
    t_strata = [t["stratum"] for t in t_meta]
    prompts = r["prompts_8"]
    bare_i = [i for i, p in enumerate(prompts) if p["format"] == "bare"]
    chat_i = [i for i, p in enumerate(prompts) if p["format"] == "chatml"]

    print("=== 0. PROMPT LENGTHS (receipt + constants) ===")
    print("receipt n_tokens:", [p["n_tokens"] for p in prompts])
    print("constants prompt_length_summary:", c["prompt_length_summary"])

    print("\n=== 1. ANCHOR DECOMPOSITION IDENTITY & VARIANT A (corrected C0 anchors) ===")
    c_var_a = copy.deepcopy(c)
    obs_med_ratio = {}
    for L in ("18", "24", "27"):
        pmn_stale = float(c["layer_geometry"][L]["post_mlp_norm_anchor"])
        pmn_c0 = float(c0["post_mlp_norms"][L])
        pmn_c1 = float(r["post_mlp_norms"][L])
        exp_reg = float(c["layer_geometry"][L]["dose_D3_over_pmn_median"])
        doses = [float(row["dose"]) for ts in t_strs for row in r["block1_regime_D_all"][L]["R_mf"][ts]]
        obs = statistics.median(d / pmn_c1 for d in doses)
        obs_med_ratio[L] = obs
        implied_th = exp_reg * pmn_stale
        corr_exp = implied_th / pmn_c0
        c_var_a["layer_geometry"][L]["dose_D3_over_pmn_median"] = corr_exp
        print(
            f"L{L}: stale={pmn_stale:.4f} c0={pmn_c0:.4f} c1={pmn_c1:.4f} | "
            f"stale/c0={pmn_stale/pmn_c0:.4f}x c0/c1={pmn_c0/pmn_c1:.4f}x "
            f"dose_live/th={statistics.median(doses)/implied_th:.4f}x | "
            f"obs/exp={obs/exp_reg:.4f}x | corr_exp={corr_exp:.6f} rel_diff={(obs-corr_exp)/corr_exp:+.1%}"
        )

    sc_a = sc.score_c1(r, c_var_a, constants_path=const_p, design_path=design_p, strict_provenance=True)
    print("Variant A (corrected C0 anchors) top keys:", list(sc_a.keys()))
    print("Variant A verdicts:", sc_a["verdicts"])

    print("\n=== 2. VARIANT B (forced gate open: expected := observed in memory only) ===")
    c_var_b = copy.deepcopy(c)
    for L in ("18", "24", "27"):
        c_var_b["layer_geometry"][L]["dose_D3_over_pmn_median"] = obs_med_ratio[L]
    sc_b = sc.score_c1(r, c_var_b, constants_path=const_p, design_path=design_p, strict_provenance=True)
    print("Variant B (EXPLORATORY ONLY) verdicts:", sc_b["verdicts"])
    print("Variant B Holm:", json.dumps(sc_b["holm_5_test_family"], indent=2))

    print("\n=== 3. REPORT-SIDE DIAGNOSTICS (using score_olmo3_c1.quantile) ===")
    # 3a. p_clean quantiles via sc.quantile
    all_pc = [float(v["p_clean"]) for pdict in r["clean_target_probs"].values() for v in pdict.values()]
    print(
        f"p_clean (n={len(all_pc)}): median={sc.quantile(all_pc, 0.5):.6g}, "
        f"p95(sc.quantile)={sc.quantile(all_pc, 0.95):.6g}, max={max(all_pc):.6g}"
    )

    # 3b. Block 1 obs/pred ("all" vs "last") by format & layer
    for mode, blk_key in (("all", "block1_regime_D_all"), ("last", "block1_regime_D_last")):
        b_all, c_all = [], []
        per_l = []
        for L in ("18", "24", "27"):
            blk = r[blk_key][L]["R_mf"]
            bl, cl = [], []
            for ts in t_strs:
                for pi, row in enumerate(blk[ts]):
                    rat = float(row["exact_dlogit"]) / float(row["pred_dlogit"])
                    (bl if pi in bare_i else cl).append(rat)
            b_all.extend(bl)
            c_all.extend(cl)
            per_l.append(f"L{L}: bare={sc.median(bl):.4f} chat={sc.median(cl):.4f} diff={sc.median(cl)-sc.median(bl):+.4f}")
        print(
            f"obs/pred ({mode}): 3L bare={sc.median(b_all):.4f} chat={sc.median(c_all):.4f} "
            f"diff={sc.median(c_all)-sc.median(b_all):+.4f} | " + " ; ".join(per_l)
        )

    # 3c. C1.2 open x func 24-target dispersion via sc.quantile
    open_idx = [i for i, s in enumerate(t_strata) if s != "function"]
    func_idx = [i for i, s in enumerate(t_strata) if s == "function"]
    of_by_pos: dict[str, list[float]] = {s: [] for s in ("nouns", "verbs", "adjectives", "function")}
    of_all: list[float] = []
    for ti, ts in enumerate(t_strs):
        partner_idx = func_idx if t_strata[ti] != "function" else open_idx
        rmf_v, rr_v = [], []
        for L in ("18", "24", "27"):
            b = r["block2a_regime_N_last"][L]
            for row in b["R_mf"][ts]:
                rmf_v.extend(abs(float(row["all_24_dlogits"][j])) for j in partner_idx)
            for row in b["R_ridge"][ts]:
                rr_v.extend(abs(float(row["all_24_dlogits"][j])) for j in partner_idx)
        rat = sc.mean(rmf_v) / sc.mean(rr_v)
        of_all.append(rat)
        of_by_pos[t_strata[ti]].append(rat)
    print(
        f"C1.2 open x func (n=24, sc.quantile): min={min(of_all):.4f} q25={sc.quantile(of_all, 0.25):.4f} "
        f"med={sc.quantile(of_all, 0.50):.4f} q75={sc.quantile(of_all, 0.75):.4f} max={max(of_all):.4f} "
        f"geo={math.exp(sc.mean([math.log(x) for x in of_all])):.4f}"
    )
    for s, xs in of_by_pos.items():
        geo_s = math.exp(sc.mean([math.log(x) for x in xs]))
        print(f"  {s:10}: geo={geo_s:.4f} med={sc.median(xs):.4f} range=[{min(xs):.4f}, {max(xs):.4f}]")

    # 3d. Per-layer F_mean, vJv, and C1.4 d_t
    for L in ("18", "24", "27"):
        draws = r["block2b_rand_iso"][L]["rand_iso"]
        f_l = sc.mean([abs(float(x)) for draw in draws for row in draw for x in row["all_24_dlogits"]])
        iso_l = sc.mean([abs(float(row["vJv"])) for draw in draws for row in draw])
        b = r["block2a_regime_N_last"][L]["logit_lens"]
        dt_pos: dict[str, list[float]] = {s: [] for s in ("nouns", "verbs", "adjectives", "function")}
        for ti, ts in enumerate(t_strs):
            tgt_l = sc.mean([abs(float(row["u_hat_self_J"])) for row in b[ts]])
            dt_pos[t_strata[ti]].append(iso_l - tgt_l)
        all_dt = [x for xs in dt_pos.values() for x in xs]
        print(
            f"L{L}: F_mean={f_l:.6f} iso_vJv={iso_l:.6f} mean_dt={sc.mean(all_dt):+.4f} | "
            + ", ".join(f"{s}={sc.mean(dt_pos[s]):+.4f}" for s in dt_pos)
        )

    # 3e. Secondary L12 breakdown (applied dose_N vs registered)
    l12 = r["secondary_L12"]["12"]
    print(
        f"L12 dose_N: applied={l12['dose_N']} (0.30 * {l12['post_mlp_norm']}), "
        f"registered={c['layer_geometry']['12']['regime_N_dose']}"
    )
    for arm in ("R_mf_all", "R_mf_last", "R_ridge_last"):
        rows = [row for ts in t_strs for row in l12[arm][ts]]
        rmsn = [abs(float(row["rms_int"]) / float(row["rms_clean"]) - 1.0) for row in rows]
        ferr = [
            abs(float(row["exact_dlogit"]) - float(row["pred_dlogit"])) / max(abs(float(row["pred_dlogit"])), 1e-6)
            for row in rows
        ]
        obs = [float(row["exact_dlogit"]) for row in rows]
        pred = [float(row["pred_dlogit"]) for row in rows]
        print(
            f"  L12 {arm:12}: med_rmsn={sc.median(rmsn):.4f} max_rmsn={max(rmsn):.4f} "
            f"gt09={sum(x > 0.09 for x in rmsn)} med_ferr={sc.median(ferr):.4f} "
            f"med_obs={sc.median(obs):.3f} med_pred={sc.median(pred):.3f}"
        )


if __name__ == "__main__":
    main()
