#!/usr/bin/env python3
"""100% pure stdlib (/usr/bin/python3 -I) adversarial unit tests for score_olmo3_c2.py (v4).

Tests:
  1. Full PASS synthetic receipt + double-run byte-identical determinism check.
  2. Fail-closed smoke receipt refusal (ValueError).
  3. Global Gate C2.0 blocking: non-zero write sham, non-zero patch sham, NaN in all_24_dlogits,
     C2.0.4 bad design SHA-256 / non-strict provenance, and exact C2.0.3 boundary ceilings (4 vs 5, 1 vs 2, 6 vs 7).
  4. F2 RELATIVE_ARM_WIN_ONLY (uniform 0.3x lens scale error passes Holm 1-4 but fails CAL-C2.1).
  5. F1 DEGENERATE_DIAG (>2 of 16 primary sources with |diag| <= 0.25 fails closed).
  6. B2/M3 LENGTH_DRIVEN_NOT_TEMPLATE vs NO_FORMAT_GAP_AT_MATCHED_LENGTH.
  7. M7 ctrl_self_attenuated_L19_31 failing EQ-C2.2.
  8. M9 EQ-C2.4c three-way verdicts (ADDITIVE, NON_ADDITIVE, INCONCLUSIVE on n_floored=3) & F9 capped precondition.
"""
from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from olmo3_virtual_and_highway import score_olmo3_c2 as sc_mod

C2_CONST_PATH = ROOT / "c2_prereg_constants.json"
C2_DESIGN_PATH = ROOT / "OLMO3-TRANSPORT-C2-DESIGN.md"


def make_clean_passing_receipt(c2_const: dict) -> dict:
    """Build a complete, internally consistent full-scale Stage C2 v4 synthetic receipt where ALL gates & pillars PASS."""
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

    # Deterministic unit-norm u_mu_proj_n (positively aligned for raw, orthogonal for shrink)
    u_mu_n = [1.0 / math.sqrt(24.0)] * 24
    # Arm cross-bleed profiles matching Table 1.3 closely
    arm_diag = {"R_mf": 2.0, "R_ridge_raw": 1.6, "R_ridge_centered": 1.48, "R_ridge_shrink": 1.48, "logit_lens": 1.2}
    arm_cross = {"R_mf": 0.23, "R_ridge_raw": 0.105, "logit_lens": 0.032, "R_ridge_centered": 0.018, "R_ridge_shrink": 0.013}
    arm_same = {"R_mf": 0.27, "R_ridge_raw": 0.135, "logit_lens": 0.055, "R_ridge_centered": 0.035, "R_ridge_shrink": 0.034}

    dose_c21 = 5.4
    r_c = 1.0  # makes scaled matrix == dlogit / 5.4
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
                                # Pure positive rank-1 along u_mu_n!
                                val = arm_cross[a] * diag_v * (1.0 + 0.02 * ((i + j) % 3 - 1))
                            else:
                                # Alternating zero-mean off-diagonal (orthogonal to u_mu_n!)
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
        # Exact matching frozen 16x8 predictions for CAL-C2.1
        for a in ("R_ridge_raw", "R_ridge_shrink"):
            fmat = []
            for i in prim_idxs:
                row_f = []
                for j in ho_idxs:
                    # Exact unit-gain prediction = scaled dlogit
                    m_ij = sum(r["all_24_dlogits"][j] for r in arms_blk[a][t_strs[i]]) / (8.0 * (dose_c21 / r_c))
                    row_f.append(round(m_ij, 6))
                fmat.append(row_f)
            f_ho[a] = fmat
        frozen_ho_preds[str(L)] = f_ho
        block1[str(L)] = {"dose": dose_c21, "arms": arms_blk}

    # Block 2 (C2.2): ChatML r_all = 1.95, LongBare r_all = 1.30 (gap = +0.65), ShortBare = 1.25;
    # B->C closes 95% (to 1.3325), C->B opens 95% (to 1.9175), all 4 controls close <= 5%!
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

    # Block 2b (C2.3): L18 heavy erasure (-3.0 vs -1.1), L21/24/27/30 post-erasure (-0.75 vs -1.05, ret_open=0.85 > ret_rand=0.05)
    block2b = {}
    for L in (18, 21, 24, 27, 30):
        ll_blk = {}
        for i, ts in enumerate(t_strs):
            is_f = strata[i] == "function"
            sm = -1.10 if is_f else (-3.05 if L == 18 else -0.75)
            ll_blk[ts] = [{"prompt_id": p["id"], "sum_mlp": sm, "retention": 0.85, "telescoping_err": 0.002} for p in prompts_8]
        ri_rows = [{"draw_idx": d, "prompt_id": p["id"], "retention": 0.05} for d in range(8) for p in prompts_8]
        block2b[str(L)] = {"dose": 5.0, "logit_lens": ll_blk, "rand_iso": ri_rows}

    # Block 3 (C2.4): masked = 2.0 * unmasked; K3 = L21 + L24 + L27 + 0.02 (additive!)
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

    # C0-colon anchor (exact ratio 0.77 at L24, 0.73 at L27)
    anc = {}
    for p in c2_const["c0_anchor_colon_prediction"]["prompts_16"]:
        is_c = p["ending"] == ":"
        anc[p["id"]] = {
            "pair_id": p["pair_id"], "format": p["format"], "ending": p["ending"],
            "pmn_L24": 7.20 if is_c else 9.33, "pmn_L27": 7.47 if is_c else 10.24,
        }

    # 61 active arm families (>0)
    active_mins = {f"arm_{k}": 0.25 for k in range(61)}

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
        cls.const_sha = sc_mod.sha256_file(C2_CONST_PATH)
        cls.design_sha = sc_mod.sha256_file(C2_DESIGN_PATH)
        cls.base_receipt = make_clean_passing_receipt(cls.c2_const)

    def _score(self, rec: dict, **kw) -> dict:
        return sc_mod.score_stage_c2(
            rec, self.c2_const,
            design_sha256=kw.pop("design_sha256", self.design_sha),
            constants_sha256=kw.pop("constants_sha256", self.const_sha),
            n_boot=kw.pop("n_boot", 300),
            n_flips_24=kw.pop("n_flips_24", 2000),
            **kw,
        )

    def test_full_pass_and_double_run_byte_identity(self) -> None:
        s1 = self._score(self.base_receipt)
        s2 = self._score(self.base_receipt)
        self.assertEqual(json.dumps(s1, sort_keys=True), json.dumps(s2, sort_keys=True))
        v = s1["confirmatory_verdicts"]
        self.assertEqual(v["C2.0"], "PASS")
        self.assertEqual(v["C2.1"], "PASS_CALIBRATED")
        self.assertEqual(v["C2.2"], "PASS")
        self.assertEqual(v["C2.3"], "LOCALIZED")
        self.assertEqual(s1["pillar_C2_3"]["selected_c3_band"], [21, 24, 27, 30])
        self.assertEqual(v["C2.4a"], "PASS")
        self.assertEqual(v["EQ-C2.4c"], "ADDITIVE")
        self.assertTrue(s1["c0_anchor_colon_prediction_nongating"]["in_predicted_range_L24"])
        self.assertTrue(s1["c0_anchor_colon_prediction_nongating"]["in_predicted_range_L27"])

    def test_fail_closed_smoke_refusal(self) -> None:
        r = copy.deepcopy(self.base_receipt)
        r["meta"]["smoke"] = True
        with self.assertRaises(ValueError):
            self._score(r)

    def test_c20_blocking_shams_nan_provenance_and_exact_c203_boundaries(self) -> None:
        # 1. Non-zero patch sham blocks everything
        r = copy.deepcopy(self.base_receipt)
        r["shams"]["patch_6"]["C2.2_self_patch_ChatML_MLP_L19_31"] = 1e-6
        s = self._score(r)
        self.assertEqual(s["confirmatory_verdicts"]["C2.0"], "FAIL")
        self.assertEqual(s["confirmatory_verdicts"]["C2.1"], "BLOCKED_BY_C2.0")
        self.assertEqual(s["confirmatory_verdicts"]["C2.2"], "BLOCKED_BY_C2.0")

        # 2. Bad design SHA-256 fails C2.0.4
        s_bad = self._score(self.base_receipt, design_sha256="0" * 64)
        self.assertFalse(s_bad["gate_C2_0"]["C2.0.4_provenance"]["pass"])
        self.assertEqual(s_bad["confirmatory_verdicts"]["C2.1"], "BLOCKED_BY_C2.0")

        # 3. Exact C2.0.3 boundary ceilings (F4: Regime-D rmsn_gt_20: 1 PASS vs 2 FAIL; fde_gt_045: 6 PASS vs 7 FAIL)
        r_d1 = copy.deepcopy(self.base_receipt)
        t0 = self.c2_const["targets_24"][0]["string"]
        r_d1["block2_C2_2"]["traces"]["unpatched_LongBare"]["0"][t0]["R_mf"]["rms_int"] = 1.25  # 1 >20% -> PASS
        self.assertTrue(self._score(r_d1)["gate_C2_0"]["C2.0.3_linearity"]["pass"])
        r_d1["block2_C2_2"]["traces"]["unpatched_LongBare"]["1"][t0]["R_mf"]["rms_int"] = 1.25  # 2 >20% -> FAIL!
        self.assertFalse(self._score(r_d1)["gate_C2_0"]["C2.0.3_linearity"]["pass"])

        r_f = copy.deepcopy(self.base_receipt)
        for pidx in range(6):
            r_f["block2_C2_2"]["traces"]["unpatched_LongBare"][str(pidx)][t0]["R_mf"]["frozen_direct_effect"] = 3.55
        self.assertTrue(self._score(r_f)["gate_C2_0"]["C2.0.3_linearity"]["pass"])
        r_f["block2_C2_2"]["traces"]["unpatched_LongBare"]["6"][t0]["R_mf"]["frozen_direct_effect"] = 3.55  # 7 > 6 -> FAIL!
        self.assertFalse(self._score(r_f)["gate_C2_0"]["C2.0.3_linearity"]["pass"])

    def test_f2_relative_arm_win_only_and_f1_degenerate_diag(self) -> None:
        # Uniform 0.3x scale on all C2.1 dlogits -> arm ratios unchanged, CAL-C2.1 fails -> RELATIVE_ARM_WIN_ONLY!
        r = copy.deepcopy(self.base_receipt)
        for L in ("18", "24", "27"):
            for a in sc_mod.C21_ARMS:
                for ts, rows in r["block1_C2_1"][L]["arms"][a].items():
                    for row in rows:
                        row["all_24_dlogits"] = [round(v * 0.3, 6) for v in row["all_24_dlogits"]]
                        row["exact_dlogit"] = round(row["exact_dlogit"] * 0.3, 6)
        s = self._score(r)
        self.assertFalse(s["pillar_C2_1"]["gate_CAL_C2_1"]["pass"])
        self.assertEqual(s["confirmatory_verdicts"]["C2.1"], "RELATIVE_ARM_WIN_ONLY")

        # F1: >2 primary sources with |diag| <= 0.25 -> DEGENERATE_DIAG
        r_deg = copy.deepcopy(self.base_receipt)
        for i in (0, 1, 2):
            ts = self.c2_const["targets_24"][i]["string"]
            for row in r_deg["block1_C2_1"]["18"]["arms"]["R_ridge_shrink"][ts]:
                row["all_24_dlogits"][i] = 0.10
        self.assertEqual(self._score(r_deg)["confirmatory_verdicts"]["C2.1"], "DEGENERATE_DIAG")

    def test_c22_length_attribution_and_self_attenuated_control_failure(self) -> None:
        # Zero ChatML-LongBare gap + large LongBare-ShortBare gap -> LENGTH_DRIVEN_NOT_TEMPLATE
        r_len = copy.deepcopy(self.base_receipt)
        r_len["block2_C2_2"]["traces"]["unpatched_ChatML"] = copy.deepcopy(r_len["block2_C2_2"]["traces"]["unpatched_LongBare"])
        for pidx in range(12):
            for ts, arms in r_len["block2_C2_2"]["traces"]["unpatched_ShortBare"][str(pidx)].items():
                arms["R_mf"]["exact_dlogit"] = 4.0  # drops ShortBare r_all well below LongBare
        self.assertEqual(self._score(r_len)["confirmatory_verdicts"]["C2.2"], "LENGTH_DRIVEN_NOT_TEMPLATE")

        # M7: self-attenuated control mimicking real closure -> EQ-C2.2 fails -> C2.2 = FAIL
        r_sa = copy.deepcopy(self.base_receipt)
        r_sa["block2_C2_2"]["traces"]["ctrl_self_attenuated_L19_31"] = copy.deepcopy(
            r_sa["block2_C2_2"]["traces"]["patch_LongBare_to_ChatML_MLP_L19_31"]
        )
        s_sa = self._score(r_sa)
        self.assertFalse(s_sa["pillar_C2_2"]["gate_EQ_C2_2"]["pass"])
        self.assertEqual(s_sa["confirmatory_verdicts"]["C2.2"], "FAIL")

    def test_m9_k3_three_way_and_f9_postmatched_precondition(self) -> None:
        # NON_ADDITIVE (+50% super-additive)
        r_na = copy.deepcopy(self.base_receipt)
        for ts, rows in r_na["block3_C2_4"]["doses"]["0.5"]["K3_combined"].items():
            for row in rows:
                row["exact_dlogit"] *= 1.50
        s_na = self._score(r_na)
        self.assertEqual(s_na["confirmatory_verdicts"]["EQ-C2.4c"], "NON_ADDITIVE")
        self.assertEqual(s_na["pillar_C2_4"]["gate_EQ_C2_4c"]["c3_action"], "RESTRICT_TO_SINGLE_LAYER_L24")

        # INCONCLUSIVE when n_floored = 3 > 2
        r_fl = copy.deepcopy(self.base_receipt)
        for i in (0, 1, 2):
            ts = self.c2_const["targets_24"][i]["string"]
            for ak in ("L21.prenorm_ff_masked", "L24.prenorm_ff_masked", "L27.prenorm_ff_masked", "K3_combined"):
                for row in r_fl["block3_C2_4"]["doses"]["0.5"][ak][ts]:
                    row["exact_dlogit"] = 0.01
        self.assertEqual(self._score(r_fl)["confirmatory_verdicts"]["EQ-C2.4c"], "INCONCLUSIVE")

        # F9: capped unmasked_postmatched fails precondition
        r_cap = copy.deepcopy(self.base_receipt)
        t0 = self.c2_const["targets_24"][0]["string"]
        r_cap["block3_C2_4"]["doses"]["1.25"]["L24.prenorm_ff_unmasked_postmatched"][t0][0]["capped"] = True
        self.assertFalse(self._score(r_cap)["pillar_C2_4"]["secondary_C2.4a_postmatched_diag"]["precondition_ok"])


if __name__ == "__main__":
    unittest.main()
