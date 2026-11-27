#!/usr/bin/env python3
"""test_olmo3_c3_design_stdlib.py — Pure-stdlib (/usr/bin/python3 -I) design-pin & consistency verifier for Stage C3 (WSP v1, Draft v2)."""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "OLMO3-WSP-C3-DESIGN.md"
CONST_PATH = ROOT / "c3_prereg_constants.json"
BANK_PATH = ROOT / "c3_vector_bank.safetensors"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestStageC3DesignStdlib(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = DESIGN_PATH.read_text(encoding="utf-8")
        cls.c = json.loads(CONST_PATH.read_text(encoding="utf-8"))

    def test_01_provenance_hashes_pinned_in_design_and_bank_on_disk(self) -> None:
        const_sha = sha256_file(CONST_PATH)
        const_bytes = CONST_PATH.stat().st_size
        bank_sha = self.c["provenance"]["c3_bank_sha256"]
        bank_bytes = self.c["provenance"]["c3_bank_bytes"]
        r_lens_sha = self.c["provenance"]["r_lens_sha256"]

        self.assertTrue(BANK_PATH.exists(), "Committed compact bank must exist on disk")
        self.assertEqual(sha256_file(BANK_PATH), bank_sha)
        self.assertEqual(BANK_PATH.stat().st_size, bank_bytes)

        self.assertIn(const_sha, self.design)
        self.assertIn(f"{const_bytes:,}", self.design)
        self.assertIn(bank_sha, self.design)
        self.assertIn(f"{bank_bytes:,}", self.design)
        self.assertIn(r_lens_sha, self.design)

    def test_02_holm_4_family_bars_and_boolean_rules_match_constants(self) -> None:
        h = self.c["holm_4_family"]
        self.assertEqual(h["alpha"], 0.05)
        self.assertEqual(h["m"], 4)
        self.assertEqual(len(h["hypotheses"]), 4)
        for v in ("PASS", "EFFECT_BELOW_BAR", "FAIL", "DOSE_INVALID", "CAPABILITY_FAIL", "NOT_READ_LOW_KAPPA", "BLOCKED_BY_NOMD_ROOM", "BLOCKED_BY_C3.0"):
            self.assertIn(v, h["verdict_enum"])
            self.assertIn(v, self.design)

        for hyp in h["hypotheses"]:
            self.assertTrue(hyp["pass_rule"])
            self.assertTrue(hyp["effect_below_bar_rule"])
            m = re.search(rf"^\|\s*\*\*`{hyp['id']}`\*\*.*$", self.design, re.M)
            self.assertIsNotNone(m, f"Missing Holm row {hyp['id']} in design")

        expected_pins = [
            ("H1", "+0.20", "0.15", "0.25", "common_pirate_ban"),
            ("H2", "+0.20", "0.15", "0.25", "d_common"),
            ("H3", "+0.50", "+0.35", "+0.20", "construct_types"),
            ("H4", "+0.35", "+0.25", "+0.15", "construct_types"),
        ]
        for hid, a, b, c, d in expected_pins:
            row = re.search(rf"^\|\s*\*\*`{hid}`\*\*.*$", self.design, re.M).group(0)
            for token in (a, b, c, d):
                self.assertIn(token, row, f"Pin {token} missing in {hid} row")

    def test_03_all_7_gates_in_matrix_match_1_to_1(self) -> None:
        gcm = self.c["gate_condition_matrix"]
        expected_keys = [
            "C3.0.1_zero_hook_restore_and_liveness",
            "C3.0.2_finiteness_mask_norm_stability_and_ban",
            "C3.0.3_completeness_and_plain_sanity",
            "C3.0.4_provenance",
            "C3.0_commit_gate",
            "C3.0_NoMD_room",
            "C3.R_dual_blind_reader_gate",
        ]
        self.assertEqual(len(gcm), 7)
        for key in expected_keys:
            self.assertIn(key, gcm)
            self.assertTrue(gcm[key]["conditions_covered"])
            self.assertTrue(gcm[key]["desk_satisfiability_proof"])
        for tag in ("C3.0.1", "C3.0.2", "C3.0.3", "C3.0.4", "C3.0_commit_gate", "C3.0_NoMD_room", "C3.R"):
            self.assertIn(f"**`{tag}`**", self.design)

    def test_04_layer_geometry_and_ban_counts_match_constants(self) -> None:
        for L in ("21", "24", "27"):
            g = self.c["layer_geometry_0cu"][L]
            self.assertIn(f"{g['pirate_lex']['eff_ratio']:.2f}×", self.design)
            self.assertIn(f"{g['nomd_lex']['eff_ratio']:.2f}×", self.design)
            self.assertIn(f"{g['pirate_lex']['cos_ret']:.5f}", self.design)
            self.assertIn(f"{g['nomd_lex']['cos_ret']:.5f}", self.design)
        ban = self.c["lexicons"]["common_pirate_ban"]
        self.assertIn(f"`{ban['single_token_count']}` frozen single-token IDs", self.design)
        self.assertIn(f"`{ban['multi_token_sequence_count']}` frozen multi-token", self.design)

    def test_05_commit_gate_off_kl_capability_and_policy_match(self) -> None:
        off_kl = self.c["edit_spec"]["off_kl_estimator"]
        self.assertEqual(off_kl["units"], "nats_per_token")
        self.assertIn("per-token mean", self.design)
        cg = self.c["edit_spec"]["commit_gate"]
        self.assertEqual(cg["reserve_median_off_kl_max"], 0.25)
        self.assertEqual(cg["reserve_max_off_kl_max"], 0.60)
        self.assertEqual(cg["reserve_broken_max"], 0)
        self.assertEqual(cg["on_fallback_verdict"], "DOSE_INVALID")
        eq = self.c["capability_preservation_gate_EQ_C3_2"]
        self.assertEqual(len(eq["applies_to"]), 4)
        self.assertEqual(eq["math_drop_max_pp"], 0.10)
        self.assertEqual(eq["code_drop_max_pp"], 0.10)
        self.assertEqual(eq["neutral_median_off_kl_max"], 0.35)
        self.assertEqual(eq["neutral_broken_max"], 0.10)
        self.assertIn("11-07", self.design)
        self.assertIn("never saves or publishes any edited model checkpoint", self.design)

    def test_06_power_table_pinned_in_design(self) -> None:
        pa = self.c["power_analysis_0cu"]
        self.assertIn(str(pa["seed"]), self.design)
        self.assertIn(f"`{pa['H1_N40_pRand005_bar020']['delta_0.30']['p_pass_r1_with_bar']:.3f}`", self.design)
        self.assertIn(f"`{pa['H2_N40_pLex005_bar020']['delta_0.30']['p_pass_r1_with_bar']:.3f}`", self.design)
        self.assertIn(f"`{pa['H3_N24_nomd_bar050']['rand2.0_arm0.8_diff1.2']['p_pass_r1_with_bar']:.3f}`", self.design)
        self.assertIn(f"`{pa['H4_N24_nomd_bar035']['rand2.0_arm1.0_diff1.0']['p_pass_r1_with_bar']:.3f}`", self.design)


if __name__ == "__main__":
    unittest.main(verbosity=2)
