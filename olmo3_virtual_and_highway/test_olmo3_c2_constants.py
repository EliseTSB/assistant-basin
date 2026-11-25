#!/usr/bin/env python3
"""Offline unit & invariant checks for Stage C2 pre-registration constants and vector bank (0 CU)."""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
C0_PATH = ROOT / "c0_prereg_constants.json"
C0B_PATH = ROOT / "c0b_prereg_constants.json"
C1_PATH = ROOT / "c1_prereg_constants.json"
C2_PATH = ROOT / "c2_prereg_constants.json"
BANK_PATH = REPO_ROOT / "work" / "olmo3_c2" / "c2_vector_bank.safetensors"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestStageC2Constants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c0 = json.loads(C0_PATH.read_text(encoding="utf-8"))
        cls.c0b = json.loads(C0B_PATH.read_text(encoding="utf-8"))
        cls.c1 = json.loads(C1_PATH.read_text(encoding="utf-8"))
        cls.c2 = json.loads(C2_PATH.read_text(encoding="utf-8"))
        snap = Path(os.path.expanduser("~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct/snapshots")) / cls.c2["revision"]
        cls.tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        cls.bank: dict[str, torch.Tensor] = {}
        with safe_open(str(BANK_PATH), framework="pt", device="cpu") as f:
            for k in f.keys():
                cls.bank[k] = f.get_tensor(k)

    def test_zero_overlap_with_c0_c1_personas(self) -> None:
        c2_words = {t["string"] for t in self.c2["targets_24"]}
        c0_words = {t["string"] for t in self.c0["targets_16"]}
        c1_words = {t["string"] for t in self.c1["targets_24"]}
        personas = {" pirate", " librarian", " wizard", " robot", " owl", " fox", " whale", " baker"}
        self.assertEqual(len(c2_words), 24)
        self.assertEqual(c2_words & c0_words, set())
        self.assertEqual(c2_words & c1_words, set())
        self.assertEqual(c2_words & personas, set())

        c2_prompts = {p["text"] for p in self.c2["prompts_8"]}
        prior_prompts = {p["text"] for p in self.c0b["prompts_8"]} | {p["text"] for p in self.c1["prompts_8"]}
        self.assertEqual(len(c2_prompts), 8)
        self.assertEqual(c2_prompts & prior_prompts, set())

    def test_single_token_roundtrip_and_colon_endings(self) -> None:
        for t in self.c2["targets_24"]:
            ids = self.tok.encode(t["string"], add_special_tokens=False)
            self.assertEqual(len(ids), 1)
            self.assertEqual(ids[0], t["id"])
            self.assertEqual(self.tok.decode(ids), t["string"])
        for p in self.c2["prompts_8"]:
            self.assertEqual(p["last_token_piece"], ":")

    def test_bank_sha256_norms_orthogonality_and_masks(self) -> None:
        self.assertEqual(sha256_file(BANK_PATH), self.c2["c2_vector_bank_sha256"])
        u_hat = self.bank["u_hat"]
        self.assertAlmostEqual(float((u_hat.norm(dim=-1) - 1.0).abs().max().item()), 0.0, places=4)

        for L in (18, 24, 27):
            for key in ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink"):
                v = self.bank[f"L{L}.{key}"]
                self.assertEqual(tuple(v.shape), (24, 4096))
                self.assertLess(float((v.norm(dim=-1) - 1.0).abs().max().item()), 1e-4)
            dm = self.bank[f"L{L}.direct_matched"]
            self.assertEqual(tuple(dm.shape), (24, 2, 4096))
            self.assertLess(float((dm.norm(dim=-1) - 1.0).abs().max().item()), 1e-4)
            # direct_matched must match cos(R_mf, u_hat)
            c_rmf = (self.bank[f"L{L}.R_mf"] * u_hat).sum(dim=-1)
            c_dm = (dm * u_hat.unsqueeze(1)).sum(dim=-1)
            self.assertLess(float((c_dm - c_rmf.unsqueeze(1)).abs().max().item()), 2e-5)

        for L in (12, 15, 18, 21, 24, 27, 30):
            ri = self.bank[f"L{L}.rand_iso"]
            self.assertEqual(tuple(ri.shape), (32, 4096))
            self.assertLess(float((ri.norm(dim=-1) - 1.0).abs().max().item()), 1e-4)

        for L in (18, 21, 24, 27):
            m_ff = self.bank[f"L{L}.gamma_ff_mask4"]
            m_at = self.bank[f"L{L}.gamma_attn_mask16"]
            self.assertEqual(int((m_ff == 0.0).sum().item()), 4)
            self.assertEqual(int((m_at == 0.0).sum().item()), 16)
            v_ff = self.bank[f"L{L}.prenorm_ff_masked"]
            v_at = self.bank[f"L{L}.prenorm_attn_masked"]
            # Masked coordinates MUST be identically zero
            self.assertEqual(float(v_ff[:, m_ff == 0.0].abs().max().item()), 0.0)
            self.assertEqual(float(v_at[:, m_at == 0.0].abs().max().item()), 0.0)
            # Deliverable fraction >= 0.98
            g = self.c2["prenorm_gamma_mask_geometry"][str(L)]
            self.assertGreaterEqual(g["ff"]["masked_deliverable_frac"], 0.98)
            self.assertGreaterEqual(g["attn"]["masked_deliverable_frac"], 0.98)

    def test_frozen_bleed_ordering(self) -> None:
        for L in (18, 24, 27):
            b = self.c2["layer_geometry"][str(L)]["frozen_unit_gain_bleed_24"]
            self.assertLess(b["R_ridge_shrink"]["cross_abs_over_diag"], b["R_ridge_centered"]["cross_abs_over_diag"])
            self.assertLess(b["R_ridge_centered"]["cross_abs_over_diag"], b["R_ridge_raw"]["cross_abs_over_diag"])
            self.assertGreater(b["R_ridge_shrink"]["same_over_cross"], b["R_ridge_centered"]["same_over_cross"])
            self.assertGreater(b["R_ridge_centered"]["same_over_cross"], b["R_ridge_raw"]["same_over_cross"])


if __name__ == "__main__":
    unittest.main()
