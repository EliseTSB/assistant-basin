#!/usr/bin/env python3
"""test_olmo3_c3b_design_stdlib.py — Pure-stdlib (/usr/bin/python3 -I) verifier for Stage C3b (WSP v2, Draft v2) design & constants."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c3_scoring  # noqa: E402

DESIGN_PATH = HERE / "OLMO3-WSP-C3B-DESIGN.md"
CONST_PATH = HERE / "c3b_prereg_constants.json"
BANK_PATH = HERE / "c3b_vector_bank.safetensors"
SCORING_PATH = HERE / "c3_scoring.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestStageC3bDesignStdlib(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = DESIGN_PATH.read_text(encoding="utf-8")
        cls.const = json.loads(CONST_PATH.read_text(encoding="utf-8"))

    def test_01_all_committed_artifact_hashes_pinned(self) -> None:
        c_sha = sha256_file(CONST_PATH)
        b_sha = sha256_file(BANK_PATH)
        s_sha = sha256_file(SCORING_PATH)
        self.assertIn(c_sha, self.design)
        self.assertIn(b_sha, self.design)
        self.assertIn(s_sha, self.design)
        self.assertEqual(self.const["provenance"]["c3b_bank_sha256"], b_sha)
        self.assertEqual(self.const["provenance"]["c3_scoring_sha256"], s_sha)

    def test_02_holm_2_family_co_primary_guards_and_stepdown(self) -> None:
        h2 = self.const["holm_2_family"]
        self.assertEqual(h2["m"], 2)
        self.assertEqual(h2["alpha"], 0.05)
        self.assertEqual([h["id"] for h in h2["hypotheses"]], ["H1", "H2"])
        self.assertEqual(h2["hypotheses"][0]["n_units"], 40)
        self.assertEqual(h2["hypotheses"][1]["n_units"], 32)
        self.assertIn("[0.85, 1.15]", h2["hypotheses"][0]["point_bar"])
        self.assertIn("reader_both_coherent_rate(Arm) >= 0.85", h2["hypotheses"][0]["point_bar"])
        self.assertIn("[0.85, 1.15]", h2["hypotheses"][1]["point_bar"])
        self.assertEqual(h2["verdict_precedence"][0], "BLOCKED_BY_C3b.0")
        self.assertEqual(c3_scoring.holm_reject({"H1": 0.020, "H2": 0.045}, 0.05), {"H1": True, "H2": True})
        self.assertEqual(c3_scoring.holm_reject({"H1": 0.030, "H2": 0.040}, 0.05), {"H1": False, "H2": False})
        for tok in ("p_(1) <= 0.025", "p_(2) <= 0.05", "BLOCKED_BY_C3b.0", "reader_both_coherent >= 0.85", "extended_maritime_and_cue_audit"):
            self.assertIn(tok, self.design)

    def test_03_all_gates_named_and_d4_live_dual_criterion(self) -> None:
        gm = self.const["gate_condition_matrix"]
        self.assertEqual(len(gm), 9)
        d4 = gm["D4_masking_check_dual_criterion"]
        self.assertIn("OUTSIDE global C3b.0", d4["scope"])
        self.assertIn("masked_over_unmasked_seed_dlogit_ratio", d4["formula"])
        self.assertIn("mismasked_over_unmasked_seed_dlogit_ratio", d4["formula"])
        self.assertIn("masked_over_unmasked_logit_l2_ratio", d4["formula"])
        self.assertIn("<= 1.10", d4["formula"])
        self.assertIn("WSP_Pirate_12L", d4["conditions_covered"][0])
        self.assertIn("masked_over_unmasked_seed_dlogit_ratio > 1.15", self.design)
        self.assertIn("masked_over_unmasked_logit_l2_ratio <= 1.10", self.design)
        self.assertIn("OUTSIDE global `C3b.0`", self.design)
        self.assertIn("579", gm["C3b.0.1_zero_hook_restore_liveness_and_b1_determinism"]["formula"])
        self.assertIn("< 2.50", gm["C3b.0.1_zero_hook_restore_liveness_and_b1_determinism"]["formula"])
        self.assertIn("<= 1.50", gm["C3b.0.2_finiteness_norm_stability_rand_kl_ban_and_code_preflight"]["formula"])
        self.assertIn(">= 0.90", gm["C3b.0_NoMD_room"]["formula"])
        self.assertIn(">= 2.00", gm["C3b.0_NoMD_room"]["formula"])

    def test_04_commit_gate_real_kl_anchors_gold_16_and_preflight(self) -> None:
        cg = self.const["edit_spec"]["commit_gate"]
        self.assertEqual(cg["reserve_median_off_kl_max"], 0.65)
        self.assertEqual(cg["reserve_max_off_kl_max"], 1.50)
        self.assertEqual(cg["reserve_max_single_pos_kl_max"], 8.0)
        self.assertEqual(cg["rung_min_rank1_proj_ret"], 0.50)
        self.assertEqual(cg["rung_min_frob_ret"], 0.80)
        # Real C3 neutral-chat anchors (0.8902, 1.0469) must be present; Draft v1's erroneous 0.903/1.111 must be gone!
        self.assertIn("0.8902", cg["threshold_derivation_note"])
        self.assertIn("1.0469", cg["threshold_derivation_note"])
        self.assertIn("0.8902", self.design)
        self.assertIn("1.0469", self.design)
        self.assertNotIn("0.903", self.design)
        self.assertNotIn("1.111", self.design)

        eq = self.const["capability_preservation_gate_EQ_C3b_2"]
        self.assertEqual(eq["math_drop_max_pp"], 0.15)
        self.assertEqual(eq["code_drop_max_pp"], 0.17)
        self.assertEqual(eq["neutral_median_off_kl_max"], 0.80)

        gg = self.const["reader_kappa_pool"]["gold_gate"]
        self.assertEqual(gg["n_items"], 16)
        self.assertEqual(gg["pooled_min_correct"], 14)
        self.assertEqual(gg["subtle_zero_stem_min_correct"], 5)
        self.assertEqual(gg["first_person_non_pirate_neg_min_correct"], 3)
        self.assertEqual(len(self.const["reader_kappa_pool"]["preflight_canary_2"]), 2)
        self.assertIn("C3b.R_preflight", self.design)
        self.assertIn(">= 14/16", self.design)
        self.assertIn(">= 5/6", self.design)
        self.assertIn(">= 3/4", self.design)
        self.assertIn("gpt-6-luna", gg["luna_c3_disclosure"])
        self.assertIn("5/6", self.design)

        # Anti-leakage: gold trade words must NOT appear in READER_SYSTEM_PROMPT
        rp = c3_scoring.READER_SYSTEM_PROMPT.lower()
        for forbidden in ("bookbinder", "clockmaker", "blacksmith", "apothecary"):
            self.assertNotIn(forbidden, rp)

    def test_05_power_table_pinned_in_design(self) -> None:
        pw = self.const["power_analysis_0cu"]
        h2_pw = pw["H2_N32_3bar_u050_p020_r015"]
        for val in (
            f"{pw['H1_N40_bar020']['pRand_0.05_delta_0.20']['p_point_bar_pass_r1']:.3f}",
            f"{pw['H1_N40_bar020']['pRand_0.05_delta_0.30']['p_point_bar_pass_r1']:.3f}",
            f"{h2_pw['c3_observed_null_lp3.3_lr3.0_la3.00']['p_point_bar_pass_r1']:.3f}",
            f"{h2_pw['room_floor_lp1.5_lr1.3_la0.35']['p_point_bar_pass_r1']:.3f}",
            f"{h2_pw['baseline_matched_hypothetical_lp3.3_lr3.0_la1.50']['p_point_bar_pass_r1']:.3f}",
            f"{h2_pw['baseline_matched_hypothetical_lp3.3_lr3.0_la1.00']['p_point_bar_pass_r1']:.3f}",
            f"{h2_pw['baseline_matched_hypothetical_lp3.3_lr3.0_la0.60']['p_point_bar_pass_r1']:.3f}",
        ):
            self.assertIn(val, self.design, f"Power value {val} missing from design")

    def test_06_policy_11_07_endpoints_and_b1_budget_sanity(self) -> None:
        self.assertIn("11-07", self.design)
        self.assertIn("Zero edited model weight checkpoints saved", self.design)
        self.assertEqual(self.const["model"]["generation_batch_size"], 1)
        self.assertEqual(self.const["endpoint_definitions"]["expected_min_restore_checks"]["value"], 579)
        for tok in ("579", "1,464", "97,792", "D_above_cap", "D6_heldout", "RESERVE_PROMPTS_8", "15.79", "2.525"):
            self.assertIn(tok, self.design, f"Expected token {tok!r} missing from design")


if __name__ == "__main__":
    unittest.main(verbosity=2)
