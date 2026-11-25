#!/usr/bin/env python3
"""Pure stdlib (/usr/bin/python3 -I) row-strict verifier for Stage C2 v3 design doc, JSON constants, and safetensors header (m5, m7).

Zero third-party imports (no torch, numpy, safetensors, or transformers). Verifies:
  1. Exact SHA-256 provenance across disk artifacts, c2_prereg_constants.json, and OLMO3-TRANSPORT-C2-DESIGN.md.
  2. Safetensors binary header parsed via struct+json: exact 34-tensor draw-level 1-to-1 keys & shapes (B4, m4).
  3. Exact budget arithmetic (8,623 fwds), 13 write shams + 6 patch shams = 19 (m3), 9 C2.2 conds, 9 C2.4 arms.
  4. 8-test reject-the-null Holm stepdown table (B5), separate equivalence gates, and unified math.ceil C2.0.3 gates (M6).
  5. ROW-STRICT markdown table checks for §1.3 (15 rows) and §1.4 (4 rows) — every cell verified on its EXACT row (m5)!
  6. Exact C0/C1 receipt provenance (7.473792 verbatim, m2) & C0-colon ratio arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DESIGN_PATH = ROOT / "OLMO3-TRANSPORT-C2-DESIGN.md"
C2_PATH = ROOT / "c2_prereg_constants.json"
BANK_PATH = REPO_ROOT / "work" / "olmo3_c2" / "c2_vector_bank.safetensors"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


class TestStageC2DesignStdlibV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c2 = json.loads(C2_PATH.read_text(encoding="utf-8"))
        cls.md = DESIGN_PATH.read_text(encoding="utf-8")
        cls.md_lines = cls.md.splitlines()
        cls.st_header = read_safetensors_header(BANK_PATH)

    def _find_unique_row(self, prefix: str) -> str:
        matches = [ln for ln in self.md_lines if ln.startswith(prefix)]
        self.assertEqual(len(matches), 1, f"Expected exactly 1 markdown row starting with {prefix!r}, got {len(matches)}")
        return matches[0]

    def test_hashes_match_disk_and_design_doc(self) -> None:
        c2_sha = sha256_file(C2_PATH)
        bank_sha = sha256_file(BANK_PATH)
        self.assertEqual(bank_sha, self.c2["c2_vector_bank_sha256"])
        self.assertIn(c2_sha, self.md)
        self.assertIn(bank_sha, self.md)
        self.assertIn(self.c2["r_lens_sha256"], self.md)
        self.assertIn(self.c2["targets_24_spec_sha256"], self.md)

    def test_safetensors_header_draw_level_1to1_keys_and_shapes(self) -> None:
        tensor_keys = {k for k in self.st_header.keys() if k != "__metadata__"}
        expected_shapes = {
            "mu_u": [4096],
            "u_vectors": [24, 4096],
            "u_hat": [24, 4096],
            "u_centered": [24, 4096],
            "u_shrink": [24, 4096],
            "L18.direct_matched": [12, 4096],
            "L24.prenorm_ff_unmasked": [24, 4096],
            "L24.prenorm_ff_rand_masked": [24, 4096],
            "L24.gamma_attn_mask16": [4096],
            "L24.prenorm_attn_masked": [24, 4096],
            "L24.prenorm_attn_rand_masked": [24, 4096],
        }
        for L in (18, 21, 24, 27, 30):
            expected_shapes[f"L{L}.rand_iso"] = [8, 4096]
        for L in (18, 24, 27):
            for arm in ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink"):
                expected_shapes[f"L{L}.{arm}"] = [24, 4096]
        for L in (21, 24, 27):
            expected_shapes[f"L{L}.gamma_ff_mask4"] = [4096]
            expected_shapes[f"L{L}.prenorm_ff_masked"] = [24, 4096]

        self.assertEqual(len(expected_shapes), 34)
        self.assertEqual(tensor_keys, set(expected_shapes.keys()))
        for k, shape in expected_shapes.items():
            self.assertEqual(self.st_header[k]["dtype"], "F32")
            self.assertEqual(self.st_header[k]["shape"], shape)

    def test_budget_and_enumerations_exact(self) -> None:
        b = self.c2["planned_budget"]
        self.assertEqual(len(b["write_shams_list"]), 13)
        self.assertEqual(len(b["patch_shams_list"]), 6)
        self.assertEqual(b["shams_total_19"], 19)
        self.assertEqual(len(b["c22_conditions_list"]), 9)
        self.assertEqual(len(b["c24_arms_list"]), 9)

        self.assertEqual(b["clean_shams_anchor_subtotal"], 36 + 19 + 16)
        self.assertEqual(b["block1_C2_1_3L_5arms_24x8"], 3 * 5 * 24 * 8)
        self.assertEqual(b["block2_C2_2_steered_9cond_12t_12p_2arms"], 9 * 12 * 12 * 2)
        self.assertEqual(b["block2_C2_2_unsteered_patched_kl_6cond_12p"], 6 * 12)
        self.assertEqual(b["block2_C2_2_total"], 2592 + 72)
        self.assertEqual(b["block2b_C2_3_5L_32v_8p"], 5 * (24 + 8) * 8)
        self.assertEqual(b["block3_C2_4_9arms_24t_4p_2doses"], 9 * 24 * 4 * 2)
        total = (
            b["clean_shams_anchor_subtotal"]
            + b["block1_C2_1_3L_5arms_24x8"]
            + b["block2_C2_2_total"]
            + b["block2b_C2_3_5L_32v_8p"]
            + b["block3_C2_4_9arms_24t_4p_2doses"]
        )
        self.assertEqual(total, 8623)
        self.assertEqual(b["total_planned_short_fwds"], 8623)
        for token in ("`71`", "`2,880`", "`2,664`", "`1,280`", "`1,728`", "`8,623`"):
            self.assertIn(token, self.md)

    def test_holm8_family_separate_equiv_gates_and_unified_c203_ceilings(self) -> None:
        inf = self.c2["registered_rng_and_inference"]
        self.assertEqual(inf["holm_family_size"], 8)
        self.assertEqual(len(inf["holm_tests"]), 8)
        expected_step = [round(0.05 / (9 - k), 6) for k in range(1, 9)]
        self.assertEqual(inf["holm_alpha_stepdown"], expected_step)
        for test_id in inf["holm_tests"]:
            self.assertIn(f"`{test_id}`", self.md)
        for alpha in expected_step:
            self.assertIn(f"{alpha:.6f}", self.md)
        # B5: equivalence claims must NOT be inside holm_tests
        for t in inf["holm_tests"]:
            self.assertNotIn("ctrl_equiv", t)
            self.assertNotIn("tost", t)
        self.assertIn("Gate `EQ-C2.2`", self.md)
        self.assertIn("Gate `EQ-C2.4c`", self.md)

        # M6: verify BOTH Regime-N and Regime-D derive from exact math.ceil(rate * N)
        g = self.c2["c2_0_3_named_gate_populations"]
        self.assertEqual(g["regime_N_C21_population_N"], 2880)
        self.assertEqual(g["regime_N_max_rmsn_gt_20pct"], math.ceil(g["rate_rmsn_gt_20pct"] * 2880))
        self.assertEqual(g["regime_N_max_rmsn_gt_12pct"], math.ceil(g["rate_rmsn_gt_12pct"] * 2880))
        self.assertEqual(g["regime_D_C22_unpatched_population_N"], 864)
        self.assertEqual(g["regime_D_max_rmsn_gt_20pct"], math.ceil(g["rate_rmsn_gt_20pct"] * 864))
        self.assertEqual(g["regime_D_max_rmsn_gt_12pct"], math.ceil(g["rate_rmsn_gt_12pct"] * 864))
        self.assertEqual(g["regime_D_max_fde_gt_045"], math.ceil(g["rate_fde_gt_045"] * 864))
        for phrase in ("N = 2,880", "<= 5", "<= 44", "N = 864", "<= 2", "<= 13", "<= 0.30", "<= 7"):
            self.assertIn(phrase, self.md)

    def test_row_strict_markdown_tables_1_3_and_1_4_match_json(self) -> None:
        # m5: ROW-STRICT — every value must appear on its EXACT matching row!
        for L in ("18", "24", "27"):
            arms = self.c2["layer_geometry"][L]["frozen_lens_transfer_preds_24"]
            for arm_name, row_data in arms.items():
                row_line = self._find_unique_row(f"| **`L{L}`** | `{arm_name}` |")
                self.assertIn(f"{row_data['diag_unit_gain_mean']:.4f}", row_line)
                self.assertIn(f"{row_data['cross_abs_over_diag']:.4f}", row_line)
                self.assertIn(f"{row_data['heldout_16x8_cross_abs_over_diag']:.4f}", row_line)
                self.assertIn(f"{row_data['same_over_cross']:.4f}×", row_line)
                self.assertIn(f"{row_data['offdiag_rank1_svd_share'] * 100:.2f}%", row_line)
                self.assertIn(f"{row_data['abs_cos_v1_right_mu_u']:.4f}", row_line)

        pn = self.c2["prenorm_gamma_mask_geometry"]
        for L, branch in (("21", "ff"), ("24", "ff"), ("24", "attn"), ("27", "ff")):
            s = pn[L][branch]
            row_line = self._find_unique_row(f"| **`L{L}` `{branch}`** |")
            self.assertIn(f"{s['gamma_min_abs']:.6f}", row_line)
            self.assertIn(f"{s['unmasked_deliverable_frac'] * 100:.2f}%", row_line)
            self.assertIn(f"{s['masked_deliverable_frac'] * 100:.2f}%", row_line)
            self.assertIn(f"{s['masked_over_unmasked_ratio']:.3f}×", row_line)
            self.assertIn(f"{s['masked_cos_ret']:.5f}", row_line)
        self.assertTrue(math.isclose(pn["24"]["m70a30_linear_delivered_ratio_vs_ff"], 1.0018, abs_tol=1e-4))
        self.assertIn("1.0018×", self.md)

    def test_c0_colon_anchor_and_exact_c1_receipt_provenance(self) -> None:
        anc = self.c2["c0_anchor_colon_prediction"]
        # m2: exact C1 L27 value 7.473792 (never rounded 7.473800)
        self.assertEqual(anc["c1_receipt_pmn"]["27"], 7.473792)
        self.assertIn("7.473792", self.md)
        self.assertNotIn("7.473800", self.md)
        r24 = round(anc["c1_receipt_pmn"]["24"] / anc["c0_receipt_pmn"]["24"], 4)
        r27 = round(anc["c1_receipt_pmn"]["27"] / anc["c0_receipt_pmn"]["27"], 4)
        self.assertEqual(anc["predicted_ratio_L24_point"], r24)
        self.assertEqual(anc["predicted_ratio_L27_point"], r27)
        self.assertIn(f"{r24:.4f}", self.md)
        self.assertIn(f"{r27:.4f}", self.md)
        self.assertEqual(len(anc["prompts_16"]), 16)


if __name__ == "__main__":
    unittest.main()
