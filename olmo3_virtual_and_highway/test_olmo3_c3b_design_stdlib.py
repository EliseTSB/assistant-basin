#!/usr/bin/env python3
"""test_olmo3_c3b_design_stdlib.py — Pure-stdlib (/usr/bin/python3 -I) verifier for Stage C3b (WSP v2) design & constants."""
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

    def test_02_holm_2_family_and_stepdown(self) -> None:
        h2 = self.const["holm_2_family"]
        self.assertEqual(h2["m"], 2)
        self.assertEqual(h2["alpha"], 0.05)
        self.assertEqual([h["id"] for h in h2["hypotheses"]], ["H1", "H2"])
        self.assertEqual(h2["hypotheses"][0]["n_units"], 40)
        self.assertEqual(h2["hypotheses"][1]["n_units"], 32)
        self.assertEqual(c3_scoring.holm_reject({"H1": 0.020, "H2": 0.045}, 0.05), {"H1": True, "H2": True})
        self.assertEqual(c3_scoring.holm_reject({"H1": 0.030, "H2": 0.040}, 0.05), {"H1": False, "H2": False})
        self.assertIn("p_(1) <= 0.025", self.design)
        self.assertIn("p_(2) <= 0.05", self.design)

    def test_03_all_gates_named_and_d4_isolated_dual_criterion(self) -> None:
        gm = self.const["gate_condition_matrix"]
        self.assertEqual(len(gm), 9)
        d4 = gm["D4_masking_check_dual_criterion"]
        self.assertIn("OUTSIDE global C3b.0", d4["scope"])
        self.assertIn("masked_over_unmasked_seed_dlogit_ratio", d4["formula"])
        self.assertIn("mismasked_over_unmasked_seed_dlogit_ratio", d4["formula"])
        self.assertIn("masked_over_unmasked_logit_l2_ratio", d4["formula"])
        self.assertIn("<= 1.25", d4["formula"])
        self.assertIn("masked_over_unmasked_seed_dlogit_ratio > 1.15", self.design)
        self.assertIn("masked_over_unmasked_logit_l2_ratio <= 1.25", self.design)
        self.assertIn("OUTSIDE global `C3b.0`", self.design)

    def test_04_commit_gate_eq_c3b_2_and_gold_alignment(self) -> None:
        cg = self.const["edit_spec"]["commit_gate"]
        self.assertEqual(cg["reserve_median_off_kl_max"], 0.65)
        self.assertEqual(cg["reserve_max_off_kl_max"], 1.50)
        self.assertEqual(cg["reserve_max_single_pos_kl_max"], 8.0)
        self.assertEqual(cg["rung_min_rank1_proj_ret"], 0.50)
        self.assertEqual(cg["rung_min_frob_ret"], 0.80)
        self.assertIn("0.65", self.design)
        eq = self.const["capability_preservation_gate_EQ_C3b_2"]
        self.assertEqual(eq["math_drop_max_pp"], 0.15)
        self.assertEqual(eq["code_drop_max_pp"], 0.17)
        self.assertEqual(eq["neutral_median_off_kl_max"], 0.80)
        gg = self.const["reader_kappa_pool"]["gold_gate"]
        self.assertEqual(gg["pooled_min_correct"], 12)
        self.assertEqual(gg["subtle_zero_stem_min_correct"], 5)
        self.assertEqual(gg["first_person_non_pirate_neg_min_correct"], 2)
        self.assertIn(">= 12/14", self.design)
        self.assertIn(">= 5/6", self.design)
        # F4 anti-leakage: gold trades (bookbinder, clockmaker) must NOT appear in READER_SYSTEM_PROMPT
        rp = c3_scoring.READER_SYSTEM_PROMPT.lower()
        for forbidden in ("bookbinder", "clockmaker", "locksmith", "electrician"):
            self.assertNotIn(forbidden, rp)

    def test_05_power_table_pinned_in_design(self) -> None:
        pw = self.const["power_analysis_0cu"]
        for val in (
            f"{pw['H1_N40_bar020']['pRand_0.05_delta_0.30']['p_point_bar_pass_r1']:.3f}",
            f"{pw['H2_N32_3bar_u050_p020_r015']['room_floor_lp1.5_lr1.3_la0.35']['p_point_bar_pass_r1']:.3f}",
            f"{pw['H2_N32_3bar_u050_p020_r015']['c3_empirical_lp3.3_lr3.0_la1.00']['p_point_bar_pass_r1']:.3f}",
            f"{pw['H2_N32_3bar_u050_p020_r015']['c3_empirical_lp3.3_lr3.0_la0.60']['p_point_bar_pass_r1']:.3f}",
        ):
            self.assertIn(val, self.design, f"Power value {val} missing from design")

    def test_06_policy_11_07_and_b1_budget_sanity(self) -> None:
        self.assertIn("11-07", self.design)
        self.assertIn("Zero edited model weight checkpoints saved", self.design)
        self.assertEqual(self.const["model"]["generation_batch_size"], 1)
        self.assertIn("1,464", self.design)
        self.assertIn("97,792", self.design)
        self.assertIn("D6_heldout", self.design)
        self.assertIn("RESERVE_PROMPTS_8", self.design)


if __name__ == "__main__":
    unittest.main(verbosity=2)
