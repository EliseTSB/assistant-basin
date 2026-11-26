#!/usr/bin/env python3
"""test_olmo3_c3_design_stdlib.py — Pure-stdlib (/usr/bin/python3 -I) design-pin & consistency verifier for Stage C3 (WSP v1)."""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "OLMO3-WSP-C3-DESIGN.md"
CONST_PATH = ROOT / "c3_prereg_constants.json"


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

    def test_01_provenance_hashes_pinned_in_design(self) -> None:
        const_sha = sha256_file(CONST_PATH)
        const_bytes = CONST_PATH.stat().st_size
        bank_sha = self.c["provenance"]["c3_bank_sha256"]
        bank_bytes = self.c["provenance"]["c3_bank_bytes"]
        r_lens_sha = self.c["provenance"]["r_lens_sha256"]

        self.assertIn(const_sha, self.design)
        self.assertIn(f"{const_bytes:,}", self.design)
        self.assertIn(bank_sha, self.design)
        self.assertIn(f"{bank_bytes:,}", self.design)
        self.assertIn(r_lens_sha, self.design)

    def test_02_holm_6_family_and_bars_match_constants(self) -> None:
        h = self.c["holm_6_family"]
        self.assertEqual(h["alpha"], 0.05)
        self.assertEqual(h["m"], 6)
        self.assertEqual(len(h["hypotheses"]), 6)

        expected_pins = [
            ("H1", "+0.25", "+0.10", "0.15"),
            ("H2", "+0.20", "+0.08", "0.20"),
            ("H3", "+0.15", "+0.05", "0.20"),
            ("H4", "+0.25", "+0.05", "0.20"),
            ("H5", "50%", "35%", "0.10"),
            ("H6", "35%", "20%", "0.10"),
        ]
        for hid, pt, ci, brk in expected_pins:
            m = re.search(rf"^\|\s*\*\*`{hid}`\*\*.*$", self.design, re.M)
            self.assertIsNotNone(m, f"Missing Holm row {hid} in design")
            row = m.group(0)
            self.assertIn(pt, row, f"Point bar {pt} missing in {hid}")
            self.assertIn(ci, row, f"CI bar {ci} missing in {hid}")
            self.assertIn(brk, row, f"Broken bar {brk} missing in {hid}")

    def test_03_gate_condition_matrix_complete_in_both_files(self) -> None:
        gcm = self.c["gate_condition_matrix"]
        for key in [
            "C3.0.1_zero_hook_and_restore",
            "C3.0.2_finiteness_norm_match_and_ban",
            "C3.0.3_completeness_and_plain_sanity",
            "C3.0.4_provenance",
            "C3.R_dual_blind_reader_gate",
        ]:
            self.assertIn(key, gcm)
            self.assertTrue(gcm[key]["conditions_covered"])
            self.assertTrue(gcm[key]["desk_satisfiability_proof"])
        for tag in ["C3.0.1", "C3.0.2", "C3.0.3", "C3.0_NoMD_room", "C3.0.4", "C3.R"]:
            self.assertIn(f"**`{tag}`**", self.design)

    def test_04_layer_geometry_table_matches_0cu_constants(self) -> None:
        for L in ("21", "24", "27"):
            g = self.c["layer_geometry_0cu"][L]
            self.assertIn(f"{g['pirate_lex']['eff_ratio']:.2f}×", self.design)
            self.assertIn(f"{g['nomd_lex']['eff_ratio']:.2f}×", self.design)
            self.assertIn(f"{g['pirate_lex']['cos_ret']:.5f}", self.design)
            self.assertIn(f"{g['nomd_lex']['cos_ret']:.5f}", self.design)

    def test_05_commit_gate_and_capability_bounds_match(self) -> None:
        cg = self.c["edit_spec"]["commit_gate"]
        self.assertEqual(cg["reserve_median_off_kl_max"], 0.25)
        self.assertEqual(cg["reserve_max_off_kl_max"], 0.60)
        self.assertEqual(cg["reserve_broken_max"], 0)
        eq = self.c["capability_preservation_gate_EQ_C3_2"]
        self.assertEqual(eq["math_drop_max_pp"], 0.10)
        self.assertEqual(eq["code_drop_max_pp"], 0.10)
        self.assertEqual(eq["neutral_median_off_kl_max"], 0.35)
        self.assertEqual(eq["neutral_broken_max"], 0.10)
        self.assertIn("11-07", self.design)
        self.assertIn("never saves or publishes any edited model checkpoint", self.design)


if __name__ == "__main__":
    unittest.main(verbosity=2)
