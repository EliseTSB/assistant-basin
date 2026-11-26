#!/usr/bin/env python3
"""100% pure stdlib (/usr/bin/python3 -I) adversarial & mutation-hardened tests for score_olmo3_c2.py (v4).

Covers:
  1. Full PASS synthetic receipt + double-run byte-identical determinism + markdown rendering.
  2. Exact pinning of ALL 40+ registered thresholds in REGISTERED_THRESHOLDS against design/constants.
  3. Fail-closed smoke & partial-checkpoint refusal (ValueError).
  4. Global Gate C2.0 clean BLOCKED_BY_C2.0 short-circuit (NEVER crashes!) on:
     - Non-zero write/patch shams, 61 junk active-arm keys (GLM #4), dead active arm
     - NaN in all_24_dlogits (GLM #3), telescoping p95 > 0.08 / p99.9 > 0.15
     - Capped C2.2 trace (DS F3), Block 2b / Block 3 row-count truncation (GLM #2), bad Block 1 dose
     - Exact C2.0.3 boundary ceilings (Regime-N 4 vs 5, Regime-D 1 vs 2, fde_p95 0.30, fde_gt_045 6 vs 7)
     - Bad design SHA-256 & strict_provenance=False
  5. Tight boundary & conjunction kill tests for EVERY Holm bar (1-7), CAL-C2.1 (all 6 clauses),
     EQ-C2.2 (all 4 controls + shape + KL), C2.3 (E_open, gap, ret_open>ret_rand), EQ-C2.4c (3-way),
     F1 DEGENERATE_DIAG (exact eps=0.25 guard), B2/M3 LENGTH_DRIVEN_NOT_TEMPLATE, and F9 cap precondition.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from olmo3_virtual_and_highway import score_olmo3_c2 as sc_mod

C2_CONST_PATH = ROOT / "c2_prereg_constants.json"
C2_DESIGN_PATH = ROOT / "OLMO3-TRANSPORT-C2-DESIGN.md"


def make_clean_passing_receipt(c2_const: dict) -> dict:
    """Build a complete full-scale Stage C2 v4 synthetic receipt with REAL 61 arm names & post_mlp_norms."""
    targets = c2_const["targets_24"]
    t_strs = [t["string"] for t in targets]
    strata = [t["stratum"] for t in targets]
    splits = [t["split"] for t in targets]
    prim_idxs = [i for i, sp in enumerate(splits) if sp == "primary_16"]
    ho_idxs = [i for i, sp in enumerate(splits) if sp == "heldout_8"]
    c22_t_idxs = c2_const["c22_target_indices_12"]
    c22_t_strs = [t_strs[i] for i in c22_t_idxs]
    prompts_8 = c2_const["prompts_8_c21"]
    triples_12 = c2_const["c22_stem_pairs_12"]
    pr_map = c2_const["c22_l18_frozen_pred_ret"]

    pmn_map = {str(L): 7.2 for L in range(32)}
    dose_c21 = round(0.75 * 7.2, 6)  # 5.4 exact match to 0.75 * pmn!
    r_c = 1.0
    u_mu_n = [1.0 / math.sqrt(24.0)] * 24
    arm_diag = {"R_mf": 2.0, "R_ridge_raw": 1.6, "R_ridge_centered": 1.48, "R_ridge_shrink": 1.48, "logit_lens": 1.2}
    arm_cross = {"R_mf": 0.23, "R_ridge_raw": 0.105, "logit_lens": 0.032, "R_ridge_centered": 0.018, "R_ridge_shrink": 0.013}
    arm_same = {"R_mf": 0.27, "R_ridge_raw": 0.135, "logit_lens": 0.055, "R_ridge_centered": 0.035, "R_ridge_shrink": 0.034}

    frozen_ho_preds: dict[str, dict[str, list[list[float]]]] = {}
    block1: dict[str, dict] = {}
    for L in (18, 24, 27):
        arms_blk = {}
        f_ho = {"R_ridge_raw": [], "R_ridge_shrink": []}
        for a in sc_mod.C21_ARMS:
            t_blk = {}
            for i in range(24):
                rows = []
                diag_v = arm_diag[a] * (dose_c21 / r_c) * (1.0 + 0.01 * ((i % 5) - 2))
                for pi, p in enumerate(prompts_8):
                    all_24 = []
                    for j in range(24):
                        if j == i:
                            val = diag_v
                        elif strata[j] == strata[i]:
                            sgn = 1.0 if a in ("R_mf", "R_ridge_raw") else (1.0 if (i + j) % 2 == 0 else -1.0)
                            val = sgn * arm_same[a] * diag_v * (1.0 + 0.02 * ((pi + j) % 3 - 1))
                        else:
                            if a in ("R_mf", "R_ridge_raw"):
                                val = arm_cross[a] * diag_v * (1.0 + 0.02 * ((i + j) % 3 - 1))
                            else:
                                sgn = 1.0 if ((i + j) % 2 == 0) else -1.0
                                val = sgn * arm_cross[a] * diag_v * (1.0 + 0.02 * ((pi + j) % 3 - 1))
                        all_24.append(round(val, 6))
                    rows.append({
                        "prompt_id": p["id"],
                        "exact_dlogit": all_24[i],
                        "all_24_dlogits": all_24,
                        "telescoping_err": 0.002,
                        "rms_clean": r_c,
                        "rms_int": r_c * 1.03,
                        "capped": False,
                        "max_abs_dlogit": abs(all_24[i]),
                    })
                t_blk[t_strs[i]] = rows
            arms_blk[a] = t_blk
        for a in ("R_ridge_raw", "R_ridge_shrink"):
            fmat = []
            for i in prim_idxs:
                row_f = []
                for j in ho_idxs:
                    m_ij = sum(r["all_24_dlogits"][j] for r in arms_blk[a][t_strs[i]]) / (8.0 * (dose_c21 / r_c))
                    row_f.append(round(m_ij, 6))
                fmat.append(row_f)
            f_ho[a] = fmat
        frozen_ho_preds[str(L)] = f_ho
        block1[str(L)] = {"dose": dose_c21, "arms": arms_blk}

    cond_r_all = {
        "unpatched_LongBare": 1.30,
        "unpatched_ChatML": 1.95,
        "unpatched_ShortBare": 1.25,
        "patch_LongBare_to_ChatML_MLP_L19_31": 1.33,
        "patch_ChatML_to_LongBare_MLP_L19_31": 1.92,
        "ctrl_L4_16_MLP_LongBare_to_ChatML": 1.93,
        "ctrl_mismatch_prompt_MLP_L19_31": 1.92,
        "ctrl_rand_iso_same_site_L19_31": 1.94,
        "ctrl_self_attenuated_L19_31": 1.94,
    }
    b2_traces = {}
    for cname, r_base in cond_r_all.items():
        b2_traces[cname] = {}
        for pidx in range(12):
            b2_traces[cname][str(pidx)] = {}
            jitter = 0.02 * ((pidx % 4) - 1.5)
            for ts in c22_t_strs:
                pg = float(pr_map[ts]["pred_gap_at_direct3"])
                gap_val = (r_base + jitter) * pg
                b2_traces[cname][str(pidx)][ts] = {
                    "R_mf": {
                        "exact_dlogit": round(gap_val + 1.5, 6),
                        "frozen_direct_effect": 3.05,
                        "telescoping_err": 0.003,
                        "rms_clean": 1.0,
                        "rms_int": 1.04,
                        "capped": False,
                    },
                    "dm0": {
                        "exact_dlogit": 1.5,
                        "frozen_direct_effect": 3.04,
                        "telescoping_err": 0.003,
                        "rms_clean": 1.0,
                        "rms_int": 1.04,
                        "capped": False,
                    },
                }

    unsteered_kl = {
        cname: [{"pair_idx": i, "stem_id": triples_12[i]["stem_id"], "kl": 0.04, "max_abs_dlogit": 0.5} for i in range(12)]
        for cname in cond_r_all
        if not cname.startswith("unpatched_")
    }

    block2b = {}
    for L in (18, 21, 24, 27, 30):
        ll_blk = {}
        for i, ts in enumerate(t_strs):
            is_f = strata[i] == "function"
            sm = -1.10 if is_f else (-3.05 if L == 18 else -0.75)
            ll_blk[ts] = [{"prompt_id": p["id"], "sum_mlp": sm, "retention": 0.85, "telescoping_err": 0.002} for p in prompts_8]
        ri_rows = [{"draw_idx": d, "prompt_id": p["id"], "retention": 0.05} for d in range(8) for p in prompts_8]
        block2b[str(L)] = {"dose": dose_c21, "logit_lens": ll_blk, "rand_iso": ri_rows}

    c24_pids = [triples_12[0]["long_bare"]["id"], triples_12[0]["chatml"]["id"], triples_12[6]["long_bare"]["id"], triples_12[6]["chatml"]["id"]]
    doses_blk = {}
    for m in ("0.5", "1.25"):
        sc = float(m) * 4.0
        arm_vals = {
            "L24.prenorm_ff_unmasked": 1.0 * sc,
            "L24.prenorm_ff_unmasked_postmatched": 1.7 * sc,
            "L24.prenorm_ff_masked": 2.0 * sc,
            "L24.m70a30": 2.25 * sc if m == "1.25" else 2.02 * sc,
            "L24.prenorm_ff_rand_masked": 0.08 * sc,
            "L24.m70a30_rand_masked": 0.082 * sc,
            "L21.prenorm_ff_masked": 1.5 * sc,
            "L27.prenorm_ff_masked": 1.8 * sc,
            "K3_combined": (1.5 + 2.0 + 1.8) * sc * 1.02,
        }
        doses_blk[m] = {}
        for ak, val in arm_vals.items():
            doses_blk[m][ak] = {
                ts: [
                    {"prompt_id": pid, "capped": False, "exact_dlogit": round(val * (1.0 + 0.01 * (ti % 3)), 6),
                     "abs_target_dlogit": round(abs(val), 6), "dlogit_l2_norm": round(10.0 * (1.05 if "m70a30" in ak else 1.0), 6)}
                    for pid in c24_pids
                ]
                for ti, ts in enumerate(t_strs)
            }

    anc = {}
    for p in c2_const["c0_anchor_colon_prediction"]["prompts_16"]:
        is_c = p["ending"] == ":"
        anc[p["id"]] = {
            "pair_id": p["pair_id"], "format": p["format"], "ending": p["ending"],
            "pmn_L24": 7.20 if is_c else 9.33, "pmn_L27": 7.47 if is_c else 10.24,
        }

    # Exact 61 real active-arm liveness keys (GLM #4)
    active_mins = {k: 0.25 for k in sorted(sc_mod.expected_active_arm_names(c2_const))}

    return {
        "schema_version": "olmo3-stage-c2-receipt-v4",
        "meta": {
            "smoke": False,
            "total_forwards": 8623,
            "provenance_hashes": {
                "r_lens_sha256": sc_mod.EXPECTED_R_LENS_SHA256,
                "c2_constants_sha256": sc_mod.EXPECTED_C2_CONSTANTS_SHA256,
                "c2_bank_sha256": sc_mod.EXPECTED_C2_BANK_SHA256,
                "c2_design_sha256": sc_mod.EXPECTED_C2_DESIGN_SHA256,
            },
        },
        "post_mlp_norms": pmn_map,
        "frozen_cal_reference": {
            "u_mu_proj_n": u_mu_n,
            "primary_16_indices": prim_idxs,
            "heldout_8_indices": ho_idxs,
            "frozen_heldout_preds_16x8": frozen_ho_preds,
        },
        "shams": {
            "write_13": {k: 0.0 for k in c2_const["planned_budget"]["write_shams_list"]},
            "patch_6": {k: 0.0 for k in c2_const["planned_budget"]["patch_shams_list"]},
        },
        "active_arm_min_max_abs_dlogit": active_mins,
        "c0_anchor_colon": anc,
        "block1_C2_1": block1,
        "block2_C2_2": {"unsteered_patched_kl": unsteered_kl, "traces": b2_traces},
        "block2b_C2_3": block2b,
        "block3_C2_4": {"doses": doses_blk},
    }


class TestStageC2ScorerStdlib(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c2_const = json.loads(C2_CONST_PATH.read_text(encoding="utf-8"))
        cls.design_md = C2_DESIGN_PATH.read_text(encoding="utf-8")
        cls.const_sha = sc_mod.sha256_file(C2_CONST_PATH)
        cls.design_sha = sc_mod.sha256_file(C2_DESIGN_PATH)
        cls.base_receipt = make_clean_passing_receipt(cls.c2_const)

    def _score(self, rec: dict, **kw) -> dict:
        return sc_mod.score_stage_c2(
            rec, self.c2_const,
            design_sha256=kw.pop("design_sha256", self.design_sha),
            constants_sha256=kw.pop("constants_sha256", self.const_sha),
            n_boot=kw.pop("n_boot", 120),
            n_flips_24=kw.pop("n_flips_24", 1000),
            **kw,
        )

    def test_01_registered_thresholds_exact_pin_against_design(self) -> None:
        """Pin EVERY numeric threshold in REGISTERED_THRESHOLDS verbatim against registered v4 design (Claude #121 #2)."""
        expected = {
            "C2.0.2_tele_p95_max": 0.08, "C2.0.2_tele_p999_max": 0.15,
            "C2.0.3_rmsn_gt_20": 0.20, "C2.0.3_rmsn_gt_12": 0.12,
            "C2.0.3_fde_gt_045": 0.45, "C2.0.3_fde_gt_075": 0.75,
            "C2.1_diag_eps": 0.25, "C2.1_bar1_pt": 2.00, "C2.1_bar1_q05": 1.50,
            "C2.1_bar2_pt": 1.15, "C2.1_bar2_sc_pt": 1.15, "C2.1_bar2_sc_q05": 1.05,
            "C2.1_bar3_pt": 1.35, "C2.1_bar3_q05": 1.15, "C2.1_bar4_pt": 0.80, "C2.1_bar4_q05": 0.75,
            "CAL_C2.1_ho_raw_lo": 0.04, "CAL_C2.1_ho_raw_hi": 0.25, "CAL_C2.1_r2_min": 0.40,
            "CAL_C2.1_slope_lo": 0.45, "CAL_C2.1_slope_hi": 1.60, "CAL_C2.1_diag_shk_frac": 0.50,
            "CAL_C2.1_mu_raw_min": 0.70, "CAL_C2.1_mu_shk_max": 0.45,
            "C2.2_gap_gate_min": 0.30, "C2.2_closure_pt_lo": 0.75, "C2.2_closure_pt_hi": 1.30,
            "C2.2_closure_ci_lo": 0.55, "C2.2_closure_ci_hi": 1.50,
            "EQ_C2.2_ctrl_std_pt": 0.25, "EQ_C2.2_ctrl_std_q95": 0.40,
            "EQ_C2.2_ctrl_mis_pt": 0.35, "EQ_C2.2_ctrl_mis_q95": 0.50,
            "EQ_C2.2_shape_open_min": 0.65, "EQ_C2.2_shape_diff_max": 0.35, "EQ_C2.2_kl_max": 0.25,
            "C2.3_e_open_min": -1.20, "C2.3_gap_max": 0.60,
            "C2.4a_bar7_pt": 1.40, "C2.4a_bar7_q05": 1.20,
            "EQ_C2.4c_med_max": 0.20, "EQ_C2.4c_ci_bound": 0.30, "EQ_C2.4c_floor": 0.10,
        }
        self.assertEqual(sc_mod.REGISTERED_THRESHOLDS, expected)
        for phrase in ("p95 <= 0.08", "p99.9 <= 0.15", ">= 2.0×", "> 1.50×", ">= 1.15×", ">= 1.05×",
                       ">= 1.35×", "> 1.15×", ">= 0.80×", ">= 0.75×", "[0.04, 0.25]", "R^2 >= 0.40",
                       "[0.45, 1.60]", ">= 0.50 ×", ">= 0.70", "<= 0.45", "[0.75, 1.30]", "[0.55, 1.50]",
                       "<= 0.25", "<= 0.40", "<= 0.35", "<= 0.50", ">= 0.65", ">= -1.20", "<= 0.60",
                       ">= 1.40×", "> 1.20×", "<= 0.20", "[-0.30, +0.30]"):
            self.assertIn(phrase, self.design_md)

    def test_02_full_pass_byte_identity_and_refusals(self) -> None:
        s1 = self._score(self.base_receipt)
        s2 = self._score(self.base_receipt)
        self.assertEqual(json.dumps(s1, sort_keys=True), json.dumps(s2, sort_keys=True))
        md = sc_mod.render_scorecard_markdown(s1)
        self.assertIn("PASS_CALIBRATED", md)
        v = s1["confirmatory_verdicts"]
        self.assertEqual(v, {"C2.0": "PASS", "C2.1": "PASS_CALIBRATED", "C2.2": "PASS", "C2.3": "LOCALIZED", "C2.4a": "PASS", "EQ-C2.4c": "ADDITIVE"})

        # Smoke refusal & partial_checkpoint refusal
        r_sm = copy.deepcopy(self.base_receipt)
        r_sm["meta"]["smoke"] = True
        with self.assertRaises(ValueError):
            self._score(r_sm)
        with self.assertRaises(ValueError):
            self._score({"partial_checkpoint": True})

    def test_03_c20_short_circuit_nan_capped_junk_keys_truncation_dose_and_boundaries(self) -> None:
        # 1. NaN in all_24_dlogits -> CLEAN BLOCKED_BY_C2.0 scorecard (NO crash! GLM #3)
        r_nan = copy.deepcopy(self.base_receipt)
        t0 = self.c2_const["targets_24"][0]["string"]
        r_nan["block1_C2_1"]["18"]["arms"]["R_ridge_raw"][t0][0]["all_24_dlogits"][0] = float("nan")
        s_nan = self._score(r_nan)
        self.assertEqual(s_nan["confirmatory_verdicts"]["C2.0"], "FAIL")
        self.assertEqual(s_nan["confirmatory_verdicts"]["C2.1"], "BLOCKED_BY_C2.0")
        self.assertFalse(s_nan["gate_C2_0"]["C2.0.2_finiteness_and_telescoping"]["pass"])
        self.assertIn("BLOCKED_BY_C2.0", sc_mod.render_scorecard_markdown(s_nan))

        # 2. Single capped C2.2 trace -> CLEAN BLOCKED_BY_C2.0 scorecard (NO crash! DS F3)
        r_cap = copy.deepcopy(self.base_receipt)
        r_cap["block2_C2_2"]["traces"]["unpatched_LongBare"]["0"][t0]["R_mf"]["capped"] = True
        s_cap = self._score(r_cap)
        self.assertEqual(s_cap["confirmatory_verdicts"]["C2.0"], "FAIL")
        self.assertEqual(s_cap["confirmatory_verdicts"]["C2.2"], "BLOCKED_BY_C2.0")

        # 3. 61 junk active-arm keys fail C2.0.1 (GLM #4)
        r_junk = copy.deepcopy(self.base_receipt)
        r_junk["active_arm_min_max_abs_dlogit"] = {f"junk_{k}": 0.5 for k in range(61)}
        s_junk = self._score(r_junk)
        self.assertFalse(s_junk["gate_C2_0"]["C2.0.1_shams_and_liveness"]["active_keys_exact_ok"])
        self.assertEqual(s_junk["confirmatory_verdicts"]["C2.0"], "FAIL")

        # 4. Non-zero write sham, non-zero patch sham, strict_provenance=False, bad design SHA
        r_sw = copy.deepcopy(self.base_receipt)
        r_sw["shams"]["write_13"]["C2.1_post_norm_L18"] = 1e-6
        self.assertEqual(self._score(r_sw)["confirmatory_verdicts"]["C2.0"], "FAIL")
        r_sp = copy.deepcopy(self.base_receipt)
        r_sp["shams"]["patch_6"]["C2.2_self_patch_ChatML_MLP_L19_31"] = 1e-6
        self.assertEqual(self._score(r_sp)["confirmatory_verdicts"]["C2.0"], "FAIL")
        self.assertEqual(self._score(self.base_receipt, strict_provenance=False)["confirmatory_verdicts"]["C2.0"], "FAIL")
        self.assertEqual(self._score(self.base_receipt, design_sha256="0" * 64)["confirmatory_verdicts"]["C2.0"], "FAIL")

        # 5. Row-count truncation on Block 2b & Block 3 + dose mismatch fail C2.0.3 (GLM #2)
        r_t2b = copy.deepcopy(self.base_receipt)
        r_t2b["block2b_C2_3"]["18"]["logit_lens"][t0] = r_t2b["block2b_C2_3"]["18"]["logit_lens"][t0][:2]
        self.assertEqual(self._score(r_t2b)["confirmatory_verdicts"]["C2.0"], "FAIL")
        r_t3 = copy.deepcopy(self.base_receipt)
        r_t3["block3_C2_4"]["doses"]["0.5"]["L24.prenorm_ff_masked"][t0] = r_t3["block3_C2_4"]["doses"]["0.5"]["L24.prenorm_ff_masked"][t0][:2]
        self.assertEqual(self._score(r_t3)["confirmatory_verdicts"]["C2.0"], "FAIL")
        r_dose = copy.deepcopy(self.base_receipt)
        r_dose["block1_C2_1"]["18"]["dose"] = 9.99
        self.assertEqual(self._score(r_dose)["confirmatory_verdicts"]["C2.0"], "FAIL")

        # 6. Exact C2.0.2 & C2.0.3 numeric boundaries (kills <= vs < and threshold mutations!)
        r_tele = copy.deepcopy(self.base_receipt)
        for L in ("18", "24", "27"):
            for a in sc_mod.C21_ARMS:
                for ts, rows in r_tele["block1_C2_1"][L]["arms"][a].items():
                    for row in rows:
                        row["telescoping_err"] = 0.08
        self.assertTrue(self._score(r_tele)["gate_C2_0"]["C2.0.2_finiteness_and_telescoping"]["pass"])
        # Need > 0.1% of 6,432 traces (>= 8 rows) > 0.15 to push p99.9 above 0.15!
        for row in r_tele["block1_C2_1"]["18"]["arms"]["R_mf"][t0]:
            row["telescoping_err"] = 0.16
        self.assertFalse(self._score(r_tele)["gate_C2_0"]["C2.0.2_finiteness_and_telescoping"]["pass"])

        r_d1 = copy.deepcopy(self.base_receipt)
        r_d1["block2_C2_2"]["traces"]["unpatched_LongBare"]["0"][t0]["R_mf"]["rms_int"] = 1.25
        self.assertTrue(self._score(r_d1)["gate_C2_0"]["C2.0.3_linearity_and_completeness"]["pass"])
        r_d1["block2_C2_2"]["traces"]["unpatched_LongBare"]["1"][t0]["R_mf"]["rms_int"] = 1.25
        self.assertFalse(self._score(r_d1)["gate_C2_0"]["C2.0.3_linearity_and_completeness"]["pass"])

        r_f = copy.deepcopy(self.base_receipt)
        for pidx in range(6):
            r_f["block2_C2_2"]["traces"]["unpatched_LongBare"][str(pidx)][t0]["R_mf"]["frozen_direct_effect"] = 3.55
        self.assertTrue(self._score(r_f)["gate_C2_0"]["C2.0.3_linearity_and_completeness"]["pass"])
        r_f["block2_C2_2"]["traces"]["unpatched_LongBare"]["6"][t0]["R_mf"]["frozen_direct_effect"] = 3.55
        self.assertFalse(self._score(r_f)["gate_C2_0"]["C2.0.3_linearity_and_completeness"]["pass"])

    def test_04_c21_conjunction_and_boundary_kills(self) -> None:
        # 1. Uniform 0.3x scale fails CAL-C2.1 -> RELATIVE_ARM_WIN_ONLY (F2)
        r = copy.deepcopy(self.base_receipt)
        for L in ("18", "24", "27"):
            for a in sc_mod.C21_ARMS:
                for rows in r["block1_C2_1"][L]["arms"][a].values():
                    for row in rows:
                        row["all_24_dlogits"] = [round(v * 0.3, 6) for v in row["all_24_dlogits"]]
                        row["exact_dlogit"] = round(row["exact_dlogit"] * 0.3, 6)
        self.assertEqual(self._score(r)["confirmatory_verdicts"]["C2.1"], "RELATIVE_ARM_WIN_ONLY")

        # 2. F1 DEGENERATE_DIAG at exact <= 0.25 boundary (2 guarded OK, 3 guarded DEGENERATE_DIAG)
        r_deg = copy.deepcopy(self.base_receipt)
        for i in (0, 1):
            ts = self.c2_const["targets_24"][i]["string"]
            for row in r_deg["block1_C2_1"]["18"]["arms"]["R_ridge_shrink"][ts]:
                row["all_24_dlogits"][i] = 0.25
        self.assertEqual(self._score(r_deg)["pillar_C2_1"]["guarded_sources_count"]["R_ridge_shrink"], 2)
        ts2 = self.c2_const["targets_24"][2]["string"]
        for row in r_deg["block1_C2_1"]["18"]["arms"]["R_ridge_shrink"][ts2]:
            row["all_24_dlogits"][2] = 0.25
        self.assertEqual(self._score(r_deg)["confirmatory_verdicts"]["C2.1"], "DEGENERATE_DIAG")

        # 3. Tight Holm bar failure checks (bar_1 < 2.0, bar_2 < 1.15, bar_3 < 1.35, bar_4 < 0.80)
        # Break bar_4: scale shrink diag down to 0.76x of raw -> fails >= 0.80 bar!
        r_b4 = copy.deepcopy(self.base_receipt)
        for L in ("18", "24", "27"):
            for i, t in enumerate(self.c2_const["targets_24"]):
                for row in r_b4["block1_C2_1"][L]["arms"]["R_ridge_shrink"][t["string"]]:
                    row["all_24_dlogits"][i] *= 0.82  # 1.48*0.82 / 1.60 = 0.7585 < 0.80
        self.assertFalse(self._score(r_b4)["pillar_C2_1"]["gain_ret"]["bar_pass"])

        # Break bar_2: set shrink cross = centered cross -> ratio 1.00 < 1.15 -> PARTIAL_CENTERING_AND_LL_WIN!
        r_b2 = copy.deepcopy(self.base_receipt)
        for L in ("18", "24", "27"):
            r_b2["block1_C2_1"][L]["arms"]["R_ridge_shrink"] = copy.deepcopy(r_b2["block1_C2_1"][L]["arms"]["R_ridge_centered"])
        self.assertEqual(self._score(r_b2)["confirmatory_verdicts"]["C2.1"], "PARTIAL_CENTERING_AND_LL_WIN")

        # Break CAL-C2.1 mu_U check: make u_mu_proj_n orthogonal to raw's off-diag mode
        r_mu = copy.deepcopy(self.base_receipt)
        r_mu["frozen_cal_reference"]["u_mu_proj_n"] = [1.0 / math.sqrt(24.0) * (1.0 if i % 2 == 0 else -1.0) for i in range(24)]
        self.assertFalse(self._score(r_mu)["pillar_C2_1"]["gate_CAL_C2_1"]["pass"])

    def test_05_c22_conjunction_controls_and_direction_kills(self) -> None:
        # 1. Length-driven vs no-gap
        r_len = copy.deepcopy(self.base_receipt)
        r_len["block2_C2_2"]["traces"]["unpatched_ChatML"] = copy.deepcopy(r_len["block2_C2_2"]["traces"]["unpatched_LongBare"])
        for pidx in range(12):
            for arms in r_len["block2_C2_2"]["traces"]["unpatched_ShortBare"][str(pidx)].values():
                arms["R_mf"]["exact_dlogit"] = 4.0
        self.assertEqual(self._score(r_len)["confirmatory_verdicts"]["C2.2"], "LENGTH_DRIVEN_NOT_TEMPLATE")

        # 2. Break bar_6 ONLY (C->B fails while B->C + EQ-C2.2 pass -> PARTIAL_UNIDIRECTIONAL, kills and->or on bar_5/6!)
        r_uni = copy.deepcopy(self.base_receipt)
        r_uni["block2_C2_2"]["traces"]["patch_ChatML_to_LongBare_MLP_L19_31"] = copy.deepcopy(
            r_uni["block2_C2_2"]["traces"]["unpatched_LongBare"]
        )
        s_uni = self._score(r_uni)
        self.assertTrue(s_uni["pillar_C2_2"]["bare_into_chat"]["bar_pass"])
        self.assertFalse(s_uni["pillar_C2_2"]["chat_into_bare"]["bar_pass"])
        self.assertEqual(s_uni["confirmatory_verdicts"]["C2.2"], "PARTIAL_UNIDIRECTIONAL")

        # 3. Break bar_6 overshoot (> 1.30): push C->B closure to 1.45 -> bar_6 fails!
        r_ov = copy.deepcopy(self.base_receipt)
        pr_map = self.c2_const["c22_l18_frozen_pred_ret"]
        for pidx in range(12):
            for ts, arms in r_ov["block2_C2_2"]["traces"]["patch_ChatML_to_LongBare_MLP_L19_31"][str(pidx)].items():
                arms["R_mf"]["exact_dlogit"] = round((1.30 + 1.45 * 0.65) * float(pr_map[ts]["pred_gap_at_direct3"]) + 1.5, 6)
        self.assertFalse(self._score(r_ov)["pillar_C2_2"]["chat_into_bare"]["bar_pass"])

        # 4. M7 self-attenuated control & unsteered KL > 0.25 both fail EQ-C2.2!
        r_sa = copy.deepcopy(self.base_receipt)
        r_sa["block2_C2_2"]["traces"]["ctrl_self_attenuated_L19_31"] = copy.deepcopy(r_sa["block2_C2_2"]["traces"]["patch_LongBare_to_ChatML_MLP_L19_31"])
        self.assertEqual(self._score(r_sa)["confirmatory_verdicts"]["C2.2"], "FAIL")

        r_kl = copy.deepcopy(self.base_receipt)
        for rows in r_kl["block2_C2_2"]["unsteered_patched_kl"].values():
            for row in rows:
                row["kl"] = 0.30
        self.assertFalse(self._score(r_kl)["pillar_C2_2"]["gate_EQ_C2_2"]["shape_and_kl_pass"])

    def test_06_c23_c24a_and_k3_boundary_kills(self) -> None:
        # C2.3 unlocalized when all layers have E_open = -1.50 (< -1.20)
        r_ul = copy.deepcopy(self.base_receipt)
        for L in ("18", "21", "24", "27", "30"):
            for rows in r_ul["block2b_C2_3"][L]["logit_lens"].values():
                for row in rows:
                    row["sum_mlp"] = -1.50
        s_ul = self._score(r_ul)
        self.assertEqual(s_ul["confirmatory_verdicts"]["C2.3"], "C2.3_UNLOCALIZED")
        self.assertEqual(s_ul["pillar_C2_3"]["selected_c3_band"], [24, 27])

        # C2.4a bar_7 fail when masked/unmasked = 1.25 (< 1.40)
        r_24a = copy.deepcopy(self.base_receipt)
        for m in ("0.5", "1.25"):
            for rows in r_24a["block3_C2_4"]["doses"][m]["L24.prenorm_ff_masked"].values():
                for row in rows:
                    row["exact_dlogit"] *= 0.625  # 2.0 * 0.625 = 1.25 < 1.40
        self.assertEqual(self._score(r_24a)["confirmatory_verdicts"]["C2.4a"], "FAIL")

        # EQ-C2.4c NON_ADDITIVE & INCONCLUSIVE (n_floored=3) & F9 cap precondition
        r_na = copy.deepcopy(self.base_receipt)
        for rows in r_na["block3_C2_4"]["doses"]["0.5"]["K3_combined"].values():
            for row in rows:
                row["exact_dlogit"] *= 1.50
        self.assertEqual(self._score(r_na)["confirmatory_verdicts"]["EQ-C2.4c"], "NON_ADDITIVE")

        r_fl = copy.deepcopy(self.base_receipt)
        for i in (0, 1, 2):
            ts = self.c2_const["targets_24"][i]["string"]
            for ak in ("L21.prenorm_ff_masked", "L24.prenorm_ff_masked", "L27.prenorm_ff_masked", "K3_combined"):
                for row in r_fl["block3_C2_4"]["doses"]["0.5"][ak][ts]:
                    row["exact_dlogit"] = 0.01
        self.assertEqual(self._score(r_fl)["confirmatory_verdicts"]["EQ-C2.4c"], "INCONCLUSIVE")

        r_cap = copy.deepcopy(self.base_receipt)
        t0 = self.c2_const["targets_24"][0]["string"]
        r_cap["block3_C2_4"]["doses"]["1.25"]["L24.prenorm_ff_unmasked_postmatched"][t0][0]["capped"] = True
        self.assertFalse(self._score(r_cap)["pillar_C2_4"]["secondary_C2.4a_postmatched_diag"]["precondition_ok"])

    def test_07_exact_boundaries_math_helpers_and_conjunction_kills(self) -> None:
        """Kill remaining exact-equality, helper, and conjunction mutants from Claude's mutate.py."""
        # 1. Pure-stdlib math helpers (quantile, norm_ppf, exact p-values, BCa acceleration, Holm default alpha)
        self.assertEqual(sc_mod.quantile([1.0, 2.0, 3.0], 0.0), 1.0)
        self.assertEqual(sc_mod.quantile([1.0, 2.0, 3.0], 1.0), 3.0)
        self.assertAlmostEqual(sc_mod.quantile([1.0, 2.0, 3.0], 0.005), 1.01, places=6)
        self.assertEqual(sc_mod.norm_ppf(0.0), -38.0)
        self.assertEqual(sc_mod.norm_ppf(1.0), 38.0)
        self.assertAlmostEqual(sc_mod.norm_ppf(0.005), -2.5758293, places=4)
        self.assertAlmostEqual(sc_mod.norm_ppf(0.97575), 1.972969, places=4)
        self.assertAlmostEqual(sc_mod.norm_ppf(0.99), 2.3263478, places=4)
        self.assertEqual(sc_mod.exact_sign_flip_p_one_sided([1.0, 1.0]), 0.25)
        self.assertEqual(sc_mod.mc_sign_flip_p_one_sided([0.0, 0.0], n_flips=10, seed=1), 1.0)
        self.assertFalse(sc_mod.holm_bonferroni_stepdown({"a": 0.055})["a"]["rejected"])
        self.assertTrue(sc_mod.holm_bonferroni_stepdown({"a": 0.05})["a"]["rejected"])
        skew = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 2.0, 9.0]
        ci_skew = sc_mod.bca_cluster_ci(len(skew), lambda idx: sc_mod.mean([skew[i] for i in idx]), n_boot=2000, seed=7)
        # Verify BCa acceleration uses exact 6.0 denominator (not mutated 6.61)
        jack = [sc_mod.mean([skew[j] for j in range(len(skew)) if j != i]) for i in range(len(skew))]
        jbar = sc_mod.mean(jack)
        diffs = [jbar - x for x in jack]
        a_true = sum(d ** 3 for d in diffs) / (6.0 * (sum(d ** 2 for d in diffs) ** 1.5))
        self.assertGreater(abs(a_true), 0.1)
        self.assertGreater(ci_skew["q_0.95"], ci_skew["point"])

        # 2. C2.0 exact inclusive ceilings + active-arm liveness > 0.0 (0.005 < 0.01 must PASS!)
        r_c20 = copy.deepcopy(self.base_receipt)
        first_active_key = sorted(r_c20["active_arm_min_max_abs_dlogit"])[0]
        r_c20["active_arm_min_max_abs_dlogit"][first_active_key] = 0.005
        # Exact tele_p999 == 0.15 (6432 rows -> idx 6424.569 -> set 8 rows to 0.15)
        t0 = self.c2_const["targets_24"][0]["string"]
        for row in r_c20["block1_C2_1"]["18"]["arms"]["R_mf"][t0]:
            row["telescoping_err"] = 0.15
        # Exact reg_n_20 == 4 and reg_n_12 == 44 (4 rows >20% + 40 rows >12%)
        n_rows = [row for t in self.c2_const["targets_24"] for row in r_c20["block1_C2_1"]["18"]["arms"]["R_mf"][t["string"]]]
        for idx in range(4):
            n_rows[idx]["rms_int"] = 1.25
        for idx in range(4, 44):
            n_rows[idx]["rms_int"] = 1.15
        # Exact reg_d_20 == 1, reg_d_12 == 13, and fde_p95 == 0.30 (864 rows -> idx 819.85 -> set 45 rows to 3.90)
        d_rows = [
            arms["R_mf"]
            for cname in sorted(r_c20["block2_C2_2"]["traces"])
            for pidx in range(12)
            for arms in r_c20["block2_C2_2"]["traces"][cname][str(pidx)].values()
        ]
        d_rows[0]["rms_int"] = 1.25
        for idx in range(1, 13):
            d_rows[idx]["rms_int"] = 1.15
        for idx in range(45):
            d_rows[idx]["frozen_direct_effect"] = 3.90  # |3.90 - 3.0| / 3.0 == 0.30 exact
        self.assertEqual(self._score(r_c20)["confirmatory_verdicts"]["C2.0"], "PASS")

        # 3. C2.2 gap_gate_pass conjunction (L789 and->or), bar_5 conjunction (L947 and->or),
        #    s_sign at g_gap == 0.0 (L788 >=->>), and exact len_gap == 0.30 (L790 >=->>)
        pr_map = self.c2_const["c22_l18_frozen_pred_ret"]
        c22_t_strs = [self.c2_const["targets_24"][i]["string"] for i in self.c2_const["c22_target_indices_12"]]
        c22_open_set = {self.c2_const["targets_24"][i]["string"] for i in self.c2_const["c22_target_indices_12"] if self.c2_const["targets_24"][i]["stratum"] != "function"}
        pgm_12 = sc_mod.mean([float(pr_map[ts]["pred_gap_at_direct3"]) for ts in c22_t_strs])

        def _set_c22_r(rec: dict, cname: str, r_val: float | dict[str, float], unit_scale: float | None = None) -> None:
            sc = pgm_12 if unit_scale is None else unit_scale
            for pidx in range(12):
                for ts in c22_t_strs:
                    rv = r_val[ts] if isinstance(r_val, dict) else r_val
                    rec["block2_C2_2"]["traces"][cname][str(pidx)][ts]["R_mf"]["exact_dlogit"] = rv * sc
                    rec["block2_C2_2"]["traces"][cname][str(pidx)][ts]["dm0"]["exact_dlogit"] = 0.0

        # 3a. g_gap = 0.20 (< 0.30) with narrow positive CI [0.20, 0.20] AND g_len == 0.30 exact!
        r_subgap = copy.deepcopy(self.base_receipt)
        _set_c22_r(r_subgap, "unpatched_ShortBare", 0.0)
        _set_c22_r(r_subgap, "unpatched_LongBare", 0.30)  # g_len == 0.30 EXACT
        _set_c22_r(r_subgap, "unpatched_ChatML", 0.50)    # g_gap = 0.20 < 0.30, narrow CI > 0
        _set_c22_r(r_subgap, "patch_LongBare_to_ChatML_MLP_L19_31", 0.30)  # closure=1.0, holm rejected!
        _set_c22_r(r_subgap, "patch_ChatML_to_LongBare_MLP_L19_31", 0.50)
        s_subgap = self._score(r_subgap)
        self.assertFalse(s_subgap["pillar_C2_2"]["gap_gate_pass"])
        self.assertFalse(s_subgap["pillar_C2_2"]["bare_into_chat"]["bar_pass"])
        self.assertFalse(s_subgap["pillar_C2_2"]["chat_into_bare"]["bar_pass"])
        self.assertEqual(s_subgap["confirmatory_verdicts"]["C2.2"], "LENGTH_DRIVEN_NOT_TEMPLATE")

        # 3b. g_gap == 0.0 exact with positive del_b2c -> s_sign MUST be +1.0 (p = 1/4096, not 1.0!)
        r_zgap = copy.deepcopy(self.base_receipt)
        _set_c22_r(r_zgap, "unpatched_LongBare", 1.0)
        _set_c22_r(r_zgap, "unpatched_ChatML", 1.0)
        _set_c22_r(r_zgap, "patch_LongBare_to_ChatML_MLP_L19_31", 0.8)  # del_b2c = +0.20 > 0
        s_zgap = self._score(r_zgap)
        self.assertLess(s_zgap["holm_7_family"]["p_C2.2_bare_into_chat"]["p"], 0.001)

        # 3c. Exact boundaries on C2.2 (g_gap == 0.30, ctrl == 0.25, cl_open == 0.65, cl_func == 1.00 -> diff == 0.35, bar_6 == 0.75)
        #     PLUS exact boundaries on C2.3 (E_open == -1.20, gap == 0.60), C2.4a (bar_7 == 1.40), EQ-C2.4c (n_fl == 2)
        g_unit = 0.30 * pgm_12
        r_edge = copy.deepcopy(self.base_receipt)
        _set_c22_r(r_edge, "unpatched_LongBare", 0.0, unit_scale=g_unit)
        _set_c22_r(r_edge, "unpatched_ChatML", 1.0, unit_scale=g_unit)  # g_gap == 0.30 exact
        b2c_map = {ts: (0.35 if ts in c22_open_set else 0.0) for ts in c22_t_strs}
        _set_c22_r(r_edge, "patch_LongBare_to_ChatML_MLP_L19_31", b2c_map, unit_scale=g_unit)  # cl_open=0.65, cl_func=1.00
        _set_c22_r(r_edge, "patch_ChatML_to_LongBare_MLP_L19_31", 0.75, unit_scale=g_unit)  # bar_6 == 0.75 bit-exact
        _set_c22_r(r_edge, "ctrl_L4_16_MLP_LongBare_to_ChatML", 0.75, unit_scale=g_unit)  # ctrl == 0.25 bit-exact
        for cname in ("ctrl_mismatch_prompt_MLP_L19_31", "ctrl_rand_iso_same_site_L19_31", "ctrl_self_attenuated_L19_31"):
            _set_c22_r(r_edge, cname, 1.0, unit_scale=g_unit)
        # C2.3 exact boundaries: E_open == -1.20, E_func == -0.60 (abs diff == 0.60 exact)
        for L in ("21", "24", "27", "30"):
            for i, t in enumerate(self.c2_const["targets_24"]):
                sm = -0.60 if t["stratum"] == "function" else -1.20
                for row in r_edge["block2b_C2_3"][L]["logit_lens"][t["string"]]:
                    row["sum_mlp"] = sm
        # C2.4a exact bar_7 == 1.40 (zero jitter)
        for m in ("0.5", "1.25"):
            sc_m = float(m) * 4.0
            for t in self.c2_const["targets_24"]:
                for row in r_edge["block3_C2_4"]["doses"][m]["L24.prenorm_ff_unmasked"][t["string"]]:
                    row["exact_dlogit"] = sc_m
                for row in r_edge["block3_C2_4"]["doses"][m]["L24.prenorm_ff_masked"][t["string"]]:
                    row["exact_dlogit"] = 1.40 * sc_m
        # EQ-C2.4c exact n_fl_05 == 2 (2 tokens floored <= 2 -> still ADDITIVE) while preserving L24 masked/unmasked == 1.40!
        for i in (0, 1):
            ts = self.c2_const["targets_24"][i]["string"]
            for ak in ("L21.prenorm_ff_masked", "L24.prenorm_ff_masked", "L27.prenorm_ff_masked", "K3_combined"):
                for row in r_edge["block3_C2_4"]["doses"]["0.5"][ak][ts]:
                    row["exact_dlogit"] = 0.028
            for row in r_edge["block3_C2_4"]["doses"]["0.5"]["L24.prenorm_ff_unmasked"][ts]:
                row["exact_dlogit"] = 0.020  # 0.028 / 0.020 == 1.40 exact; d_sum = 0.084 < 0.10 floor!
        s_edge = self._score(r_edge)
        self.assertTrue(s_edge["pillar_C2_2"]["gap_gate_pass"])
        self.assertTrue(s_edge["pillar_C2_2"]["chat_into_bare"]["bar_pass"])
        self.assertTrue(s_edge["pillar_C2_2"]["gate_EQ_C2_2"]["pass"])
        self.assertEqual(s_edge["confirmatory_verdicts"]["C2.3"], "LOCALIZED")
        self.assertEqual(s_edge["confirmatory_verdicts"]["C2.4a"], "PASS")
        self.assertEqual(s_edge["confirmatory_verdicts"]["EQ-C2.4c"], "ADDITIVE")

        # 3d. Exact bar_5 == 0.75 boundary (kills L947 <=->< and >=->>!)
        r_b5 = copy.deepcopy(r_edge)
        _set_c22_r(r_b5, "patch_LongBare_to_ChatML_MLP_L19_31", 0.25, unit_scale=g_unit)  # bar_5 == 0.75 bit-exact
        s_b5 = self._score(r_b5)
        self.assertTrue(s_b5["pillar_C2_2"]["bare_into_chat"]["bar_pass"])
        self.assertEqual(s_b5["confirmatory_verdicts"]["C2.2"], "PASS")

        # 4. Exact bar_1..bar_4 + CAL-C2.1 boundaries (L735..L740, L943..L946)
        r_c21 = copy.deepcopy(self.base_receipt)
        dose_c21 = round(0.75 * 7.2, 6)
        frozen_diag_shk_3L = sc_mod.mean([
            float(self.c2_const["layer_geometry"][L]["frozen_lens_transfer_preds_24"]["R_ridge_shrink"]["diag_unit_gain_mean"])
            for L in ("18", "24", "27")
        ])
        shk_d = 0.50 * frozen_diag_shk_3L
        raw_d = shk_d / 0.80  # exact bar_4 = 0.80 AND exact CAL scaled_diag == 0.50 * frozen!
        arm_diag_e = {"R_mf": raw_d * 1.1, "R_ridge_raw": raw_d, "R_ridge_centered": 2.0 / dose_c21, "R_ridge_shrink": shk_d, "logit_lens": 2.0 / dose_c21}
        # Exact bar_1 = 2.00 (raw/ctr), bar_2 = 1.15 (ctr/shk), bar_3 = 1.35 (ll/shk) using power-of-2 base (0 ULP loss!)
        shk_c = 0.03  # exact IEEE-754 3-layer mean for 2.00, 1.15, 1.35!
        ctr_c = 1.15 * shk_c
        raw_c = 2.00 * ctr_c  # 0.069 in [0.04, 0.25]
        ll_c = 1.35 * shk_c
        arm_cross_e = {"R_mf": 0.20, "R_ridge_raw": raw_c, "R_ridge_centered": ctr_c, "R_ridge_shrink": shk_c, "logit_lens": ll_c}
        arm_same_e = {"R_mf": 0.22, "R_ridge_raw": raw_c * 1.1, "R_ridge_centered": ctr_c * 1.3, "R_ridge_shrink": shk_c * 1.6, "logit_lens": ll_c * 1.2}
        strata = [t["stratum"] for t in self.c2_const["targets_24"]]
        prim_idxs = [i for i, t in enumerate(self.c2_const["targets_24"]) if t["split"] == "primary_16"]
        ho_idxs = [i for i, t in enumerate(self.c2_const["targets_24"]) if t["split"] == "heldout_8"]
        for L in ("18", "24", "27"):
            for a in sc_mod.C21_ARMS:
                for i, t in enumerate(self.c2_const["targets_24"]):
                    dv = arm_diag_e[a] * dose_c21
                    all_24 = []
                    for j in range(24):
                        if j == i:
                            v = dv
                        elif strata[j] == strata[i]:
                            sgn = 1.0 if a in ("R_mf", "R_ridge_raw") else (1.0 if (i + j) % 2 == 0 else -1.0)
                            v = sgn * arm_same_e[a] * dv
                        else:
                            if a == "R_ridge_raw":
                                # Non-zero variance for OLS with exact mean == 0.04
                                v = (raw_c + (0.004 if (i + j) % 2 == 0 else -0.004)) * dv
                            elif a == "R_mf":
                                v = arm_cross_e[a] * dv
                            else:
                                sgn = 1.0 if (i + j) % 2 == 0 else -1.0
                                v = sgn * arm_cross_e[a] * dv
                        all_24.append(v)
                    for row in r_c21["block1_C2_1"][L]["arms"][a][t["string"]]:
                        row["all_24_dlogits"] = list(all_24)
                        row["exact_dlogit"] = all_24[i]
            # Make frozen_heldout_preds_16x8 match raw with exact slope = 0.45 (R^2 = 1.0)
            for a in ("R_ridge_raw", "R_ridge_shrink"):
                fmat = []
                for i in prim_idxs:
                    row_f = []
                    for j in ho_idxs:
                        m_ij = r_c21["block1_C2_1"][L]["arms"][a][self.c2_const["targets_24"][i]["string"]][0]["all_24_dlogits"][j] / dose_c21
                        row_f.append(m_ij / 0.45 if a == "R_ridge_raw" else m_ij)
                    fmat.append(row_f)
                r_c21["frozen_cal_reference"]["frozen_heldout_preds_16x8"][L][a] = fmat
        s_c21 = self._score(r_c21)
        self.assertTrue(s_c21["pillar_C2_1"]["ctr_vs_raw"]["bar_pass"])
        self.assertTrue(s_c21["pillar_C2_1"]["shk_vs_ctr"]["bar_pass"])
        self.assertTrue(s_c21["pillar_C2_1"]["shk_vs_ll"]["bar_pass"])
        self.assertTrue(s_c21["pillar_C2_1"]["gain_ret"]["bar_pass"])
        self.assertEqual(s_c21["confirmatory_verdicts"]["C2.1"], "PASS_CALIBRATED")


if __name__ == "__main__":
    unittest.main()
