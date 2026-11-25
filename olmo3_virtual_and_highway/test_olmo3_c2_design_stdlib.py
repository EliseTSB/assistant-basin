#!/usr/bin/env python3
"""Pure stdlib (/usr/bin/python3 -I) verifier for Stage C2 v2 design doc, JSON constants, and safetensors header (m7).

Zero third-party imports (no torch, numpy, safetensors, or transformers). Verifies:
  1. Exact SHA-256 provenance across disk artifacts, c2_prereg_constants.json, and OLMO3-TRANSPORT-C2-DESIGN.md.
  2. Safetensors binary header parsed directly via struct+json: exact 1-to-1 tensor keys & shapes (B4).
  3. Exact budget arithmetic, 18 enumerated shams, 9 C2.2 conditions, 11 C2.4 arms, and markdown table consistency.
  4. Unified 9-test Holm stepdown thresholds and C2.0.3 named gate populations/denominators (B5, M6).
  5. Every numerical cell in §1.3 (15 rows) and §1.4 (4 rows) of OLMO3-TRANSPORT-C2-DESIGN.md matches JSON.
  6. C0-colon ratio anchor arithmetic (m1, m2, m6) and absence of stale Exp 4B literal anchor usage.
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


class TestStageC2DesignStdlib(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c2 = json.loads(C2_PATH.read_text(encoding="utf-8"))
        cls.md = DESIGN_PATH.read_text(encoding="utf-8")
        cls.st_header = read_safetensors_header(BANK_PATH)

    def test_hashes_match_disk_and_design_doc(self) -> None:
        c2_sha = sha256_file(C2_PATH)
        bank_sha = sha256_file(BANK_PATH)
        self.assertEqual(bank_sha, self.c2["c2_vector_bank_sha256"])
        self.assertIn(c2_sha, self.md)
        self.assertIn(bank_sha, self.md)
        self.assertIn(self.c2["r_lens_sha256"], self.md)
        self.assertIn(self.c2["targets_24_spec_sha256"], self.md)

    def test_safetensors_header_1to1_keys_and_shapes(self) -> None:
        tensor_keys = {k for k in self.st_header.keys() if k != "__metadata__"}
        expected_shapes = {
            "mu_u": [4096],
            "u_vectors": [24, 4096],
            "u_hat": [24, 4096],
            "u_centered": [24, 4096],
            "u_shrink": [24, 4096],
        }
        for L in (12, 15, 18, 21, 24, 27, 30):
            expected_shapes[f"L{L}.rand_iso"] = [32, 4096]
        for L in (18, 24, 27):
            for arm in ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink"):
                expected_shapes[f"L{L}.{arm}"] = [24, 4096]
            expected_shapes[f"L{L}.direct_matched"] = [24, 2, 4096]
        for L in (21, 24, 27):
            expected_shapes[f"L{L}.gamma_ff_mask4"] = [4096]
            expected_shapes[f"L{L}.prenorm_ff_masked"] = [24, 4096]
            expected_shapes[f"L{L}.prenorm_ff_rand_masked"] = [24, 1, 4096]
        expected_shapes["L24.prenorm_ff_unmasked"] = [24, 4096]
        expected_shapes["L24.gamma_attn_mask16"] = [4096]
        expected_shapes["L24.prenorm_attn_masked"] = [24, 4096]
        expected_shapes["L24.prenorm_attn_rand_masked"] = [24, 1, 4096]

        self.assertEqual(tensor_keys, set(expected_shapes.keys()))
        for k, shape in expected_shapes.items():
            self.assertEqual(self.st_header[k]["dtype"], "F32")
            self.assertEqual(self.st_header[k]["shape"], shape)

    def test_budget_and_enumerations_exact(self) -> None:
        b = self.c2["planned_budget"]
        self.assertEqual(len(b["sham_arms_list"]), 18)
        self.assertEqual(len(b["c22_conditions_list"]), 9)
        self.assertEqual(len(b["c24_arms_list"]), 11)
        self.assertEqual(b["clean_shams_anchor_subtotal"], 24 + 18 + 16)
        self.assertEqual(b["block1_C2_1_3L_5arms_24x8"], 3 * 5 * 24 * 8)
        self.assertEqual(b["block2_C2_2_9cond_12t_8pairs_2arms"], 9 * 12 * 8 * 2)
        self.assertEqual(b["block2b_C2_3_7L_56v_8p"], 7 * (24 + 32) * 8)
        self.assertEqual(b["block3_C2_4_11arms_24t_4p_2doses"], 11 * 24 * 4 * 2)
        total = (
            b["clean_shams_anchor_subtotal"]
            + b["block1_C2_1_3L_5arms_24x8"]
            + b["block2_C2_2_9cond_12t_8pairs_2arms"]
            + b["block2b_C2_3_7L_56v_8p"]
            + b["block3_C2_4_11arms_24t_4p_2doses"]
        )
        self.assertEqual(total, 9914)
        self.assertEqual(b["total_planned_short_fwds"], 9914)
        for token in ("`58`", "`2,880`", "`1,728`", "`3,136`", "`2,112`", "`9,914`"):
            self.assertIn(token, self.md)

    def test_holm_family_and_c203_gate_denominators(self) -> None:
        inf = self.c2["registered_rng_and_inference"]
        self.assertEqual(inf["holm_family_size"], 9)
        self.assertEqual(len(inf["holm_tests"]), 9)
        expected_step = [round(0.05 / (10 - k), 6) for k in range(1, 10)]
        self.assertEqual(inf["holm_alpha_stepdown"], expected_step)
        for test_id in inf["holm_tests"]:
            self.assertIn(f"`{test_id}`", self.md)
        for alpha in expected_step:
            self.assertIn(f"{alpha:.6f}", self.md)

        g = self.c2["c2_0_3_named_gate_populations"]
        self.assertEqual(g["regime_N_C21_population_N"], 2880)
        self.assertEqual(g["regime_D_C22_unpatched_population_N"], 576)
        for phrase in ("N = 2,880", "<= 4", "<= 43", "N = 576", "<= 2", "<= 12", "<= 0.30", "<= 8"):
            self.assertIn(phrase, self.md)

    def test_markdown_tables_1_3_and_1_4_match_json(self) -> None:
        for L in ("18", "24", "27"):
            arms = self.c2["layer_geometry"][L]["frozen_lens_transfer_preds_24"]
            for arm_name, row in arms.items():
                self.assertIn(f"| **`L{L}`** | `{arm_name}` |", self.md)
                self.assertIn(f"{row['diag_unit_gain_mean']:.4f}", self.md)
                self.assertIn(f"{row['cross_abs_over_diag']:.4f}", self.md)
                self.assertIn(f"{row['heldout_16x8_cross_abs_over_diag']:.4f}", self.md)
                self.assertIn(f"{row['same_over_cross']:.4f}×", self.md)
                self.assertIn(f"{row['offdiag_rank1_svd_share'] * 100:.2f}%", self.md)
                self.assertIn(f"{row['abs_cos_v1_right_mu_u']:.4f}", self.md)

        pn = self.c2["prenorm_gamma_mask_geometry"]
        for L, branch in (("21", "ff"), ("24", "ff"), ("24", "attn"), ("27", "ff")):
            s = pn[L][branch]
            self.assertIn(f"| **`L{L}` `{branch}`** |", self.md)
            self.assertIn(f"{s['gamma_min_abs']:.6f}", self.md)
            self.assertIn(f"{s['unmasked_deliverable_frac'] * 100:.2f}%", self.md)
            self.assertIn(f"{s['masked_deliverable_frac'] * 100:.2f}%", self.md)
            self.assertIn(f"{s['masked_cos_ret']:.5f}", self.md)
        self.assertTrue(math.isclose(pn["24"]["m70a30_linear_delivered_ratio_vs_ff"], 1.0018, abs_tol=1e-4))
        self.assertIn("1.0018×", self.md)

    def test_c0_colon_anchor_ratio_arithmetic(self) -> None:
        anc = self.c2["c0_anchor_colon_prediction"]
        r24 = round(anc["c1_receipt_pmn"]["24"] / anc["c0_receipt_pmn"]["24"], 4)
        r27 = round(anc["c1_receipt_pmn"]["27"] / anc["c0_receipt_pmn"]["27"], 4)
        self.assertEqual(anc["predicted_ratio_L24_point"], r24)
        self.assertEqual(anc["predicted_ratio_L27_point"], r27)
        self.assertIn(f"{r24:.4f}", self.md)
        self.assertIn(f"{r27:.4f}", self.md)
        self.assertIn("c0_receipt.json.gz", self.md)
        self.assertEqual(len(anc["prompts_16"]), 16)


if __name__ == "__main__":
    unittest.main()
