#!/usr/bin/env python3
"""Offline PyTorch + tokenizer invariant checks for Stage C2 v3 pre-registration constants and vector bank (0 CU)."""
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


class TestStageC2ConstantsV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c0 = json.loads(C0_PATH.read_text(encoding="utf-8"))
        cls.c0b = json.loads(C0B_PATH.read_text(encoding="utf-8"))
        cls.c1 = json.loads(C1_PATH.read_text(encoding="utf-8"))
        cls.c2 = json.loads(C2_PATH.read_text(encoding="utf-8"))
        snap = (
            Path(os.path.expanduser("~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct/snapshots"))
            / cls.c2["revision"]
        )
        cls.tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        cls.bank: dict[str, torch.Tensor] = {}
        with safe_open(str(BANK_PATH), framework="pt", device="cpu") as f:
            for k in f.keys():
                cls.bank[k] = f.get_tensor(k)

    def test_schema_splits_and_zero_overlap_with_c0_c1_personas(self) -> None:
        self.assertEqual(self.c2["schema_version"], "c2-prereg-constants-v4")
        c2_words = {t["string"] for t in self.c2["targets_24"]}
        c0_words = {t["string"] for t in self.c0["targets_16"]}
        c1_words = {t["string"] for t in self.c1["targets_24"]}
        personas = {" pirate", " librarian", " wizard", " robot", " owl", " fox", " whale", " baker"}
        self.assertEqual(len(c2_words), 24)
        self.assertEqual(c2_words & c0_words, set())
        self.assertEqual(c2_words & c1_words, set())
        self.assertEqual(c2_words & personas, set())

        prim = [t for t in self.c2["targets_24"] if t["split"] == "primary_16"]
        ho = [t for t in self.c2["targets_24"] if t["split"] == "heldout_8"]
        c22 = [t for t in self.c2["targets_24"] if t["in_c22_12"]]
        self.assertEqual(len(prim), 16)
        self.assertEqual(len(ho), 8)
        self.assertEqual(len(c22), 12)
        for stratum in ("nouns", "verbs", "adjectives", "function"):
            self.assertEqual(sum(1 for t in prim if t["stratum"] == stratum), 4)
            self.assertEqual(sum(1 for t in ho if t["stratum"] == stratum), 2)
            self.assertEqual(sum(1 for t in c22 if t["stratum"] == stratum), 3)

        prior_prompts = {p["text"] for p in self.c0b["prompts_8"]} | {p["text"] for p in self.c1["prompts_8"]}
        c21_prompts = {p["text"] for p in self.c2["prompts_8_c21"]}
        self.assertEqual(len(c21_prompts), 8)
        self.assertEqual(c21_prompts & prior_prompts, set())
        # m7: C2.1 / C2.3 8 prompts are matched-length (4 LongBare + 4 ChatML)
        self.assertEqual(sum(1 for p in self.c2["prompts_8_c21"] if p["format"] == "long_bare"), 4)
        self.assertEqual(sum(1 for p in self.c2["prompts_8_c21"] if p["format"] == "chatml"), 4)
        for tr in self.c2["c22_stem_pairs_12"]:
            for fmt in ("short_bare", "long_bare", "chatml"):
                self.assertNotIn(tr[fmt]["text"], prior_prompts)

    def test_single_token_roundtrip_b1_suffix_b2_length_and_b3_no_stem_duplication(self) -> None:
        for t in self.c2["targets_24"]:
            ids = self.tok.encode(t["string"], add_special_tokens=False)
            self.assertEqual(len(ids), 1)
            self.assertEqual(ids[0], t["id"])
            self.assertEqual(self.tok.decode(ids), t["string"])

        triples = self.c2["c22_stem_pairs_12"]
        self.assertEqual(len(triples), 12)
        for tr in triples:
            stem = tr["stem"]
            # B3: zero stem duplication in every format, and identical context clause in LongBare & ChatML
            for fmt in ("short_bare", "long_bare", "chatml"):
                self.assertEqual(tr[fmt]["text"].count(stem), 1)
            self.assertIn(tr["context_clause"], tr["long_bare"]["text"])
            self.assertIn(tr["context_clause"], tr["chatml"]["text"])

            ids_s = self.tok.encode(tr["short_bare"]["text"], add_special_tokens=False)
            ids_l = self.tok.encode(tr["long_bare"]["text"], add_special_tokens=False)
            ids_c = self.tok.encode(tr["chatml"]["text"], add_special_tokens=False)
            t_s = len(ids_s)
            self.assertEqual(t_s, tr["suffix_len"])
            self.assertEqual(len(ids_l), tr["long_bare"]["n_tokens"])
            self.assertEqual(len(ids_c), tr["chatml"]["n_tokens"])
            # B1 strict trailing suffix identity
            self.assertEqual(ids_l[-t_s:], ids_s)
            self.assertEqual(ids_c[-t_s:], ids_s)
            self.assertEqual(self.tok.decode([ids_s[-1]]), ":")
            # B2 matched length within <= 1 token
            self.assertLessEqual(abs(len(ids_l) - len(ids_c)), 1)

    def test_bank_sha256_draw_level_1to1_keys_norms_orthogonality_and_masks(self) -> None:
        self.assertEqual(sha256_file(BANK_PATH), self.c2["c2_vector_bank_sha256"])

        # Exact 34-tensor draw-level 1-to-1 key set (m4, B4, M3)
        expected_keys = {
            "mu_u", "u_vectors", "u_hat", "u_centered", "u_shrink",
            "L18.rand_iso", "L21.rand_iso", "L24.rand_iso", "L27.rand_iso", "L30.rand_iso",
            "L18.R_mf", "L18.R_ridge_raw", "L18.R_ridge_centered", "L18.R_ridge_shrink", "L18.direct_matched",
            "L24.R_mf", "L24.R_ridge_raw", "L24.R_ridge_centered", "L24.R_ridge_shrink",
            "L27.R_mf", "L27.R_ridge_raw", "L27.R_ridge_centered", "L27.R_ridge_shrink",
            "L21.gamma_ff_mask4", "L21.prenorm_ff_masked",
            "L24.gamma_ff_mask4", "L24.prenorm_ff_masked", "L24.prenorm_ff_unmasked", "L24.prenorm_ff_rand_masked",
            "L24.gamma_attn_mask16", "L24.prenorm_attn_masked", "L24.prenorm_attn_rand_masked",
            "L27.gamma_ff_mask4", "L27.prenorm_ff_masked",
        }
        self.assertEqual(set(self.bank.keys()), expected_keys)

        u_hat = self.bank["u_hat"]
        self.assertAlmostEqual(float((u_hat.norm(dim=-1) - 1.0).abs().max().item()), 0.0, places=4)

        for L in (18, 24, 27):
            for key in ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink"):
                v = self.bank[f"L{L}.{key}"]
                self.assertEqual(tuple(v.shape), (24, 4096))
                self.assertLess(float((v.norm(dim=-1) - 1.0).abs().max().item()), 1e-4)

        # L18 direct_matched: [12, 4096], matches cos(R_mf, u_hat) on C2.2 12-target subset
        c22_idx = self.c2["c22_target_indices_12"]
        dm = self.bank["L18.direct_matched"]
        self.assertEqual(tuple(dm.shape), (12, 4096))
        self.assertLess(float((dm.norm(dim=-1) - 1.0).abs().max().item()), 1e-4)
        c_rmf = (self.bank["L18.R_mf"][c22_idx] * u_hat[c22_idx]).sum(dim=-1)
        c_dm = (dm * u_hat[c22_idx]).sum(dim=-1)
        self.assertLess(float((c_dm - c_rmf).abs().max().item()), 2e-5)

        for L in (18, 21, 24, 27, 30):
            ri = self.bank[f"L{L}.rand_iso"]
            self.assertEqual(tuple(ri.shape), (8, 4096))
            self.assertLess(float((ri.norm(dim=-1) - 1.0).abs().max().item()), 1e-4)

        for L in (21, 24, 27):
            m_ff = self.bank[f"L{L}.gamma_ff_mask4"]
            self.assertEqual(int((m_ff == 0.0).sum().item()), 4)
            v_ff = self.bank[f"L{L}.prenorm_ff_masked"]
            self.assertEqual(tuple(v_ff.shape), (24, 4096))
            self.assertEqual(float(v_ff[:, m_ff == 0.0].abs().max().item()), 0.0)
            self.assertGreaterEqual(self.c2["prenorm_gamma_mask_geometry"][str(L)]["ff"]["masked_deliverable_frac"], 0.98)

        m_ff24 = self.bank["L24.gamma_ff_mask4"]
        r_ff24 = self.bank["L24.prenorm_ff_rand_masked"]
        self.assertEqual(tuple(r_ff24.shape), (24, 4096))
        self.assertEqual(float(r_ff24[:, m_ff24 == 0.0].abs().max().item()), 0.0)

        m_at = self.bank["L24.gamma_attn_mask16"]
        self.assertEqual(int((m_at == 0.0).sum().item()), 16)
        v_at = self.bank["L24.prenorm_attn_masked"]
        r_at = self.bank["L24.prenorm_attn_rand_masked"]
        self.assertEqual(tuple(v_at.shape), (24, 4096))
        self.assertEqual(tuple(r_at.shape), (24, 4096))
        self.assertEqual(float(v_at[:, m_at == 0.0].abs().max().item()), 0.0)
        self.assertEqual(float(r_at[:, m_at == 0.0].abs().max().item()), 0.0)
        self.assertGreaterEqual(self.c2["prenorm_gamma_mask_geometry"]["24"]["attn"]["masked_deliverable_frac"], 0.98)

    def test_frozen_lens_transfer_ordering_mu_u_and_c22_pred_ret(self) -> None:
        for L in (18, 24, 27):
            b = self.c2["layer_geometry"][str(L)]["frozen_lens_transfer_preds_24"]
            self.assertEqual(set(b.keys()), {"R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink", "logit_lens"})
            self.assertLess(b["R_ridge_shrink"]["heldout_16x8_cross_abs_over_diag"], b["R_ridge_centered"]["heldout_16x8_cross_abs_over_diag"])
            self.assertLess(b["R_ridge_centered"]["heldout_16x8_cross_abs_over_diag"], b["logit_lens"]["heldout_16x8_cross_abs_over_diag"])
            self.assertLess(b["logit_lens"]["heldout_16x8_cross_abs_over_diag"], b["R_ridge_raw"]["heldout_16x8_cross_abs_over_diag"])
            self.assertGreater(b["R_ridge_shrink"]["same_over_cross"], b["R_ridge_centered"]["same_over_cross"])
            self.assertGreaterEqual(b["R_ridge_raw"]["abs_cos_v1_right_mu_u"], 0.95)
            self.assertLessEqual(b["R_ridge_shrink"]["abs_cos_v1_right_mu_u"], 0.45)

        pr = self.c2["c22_l18_frozen_pred_ret"]
        self.assertEqual(len(pr), 12)
        for v in pr.values():
            self.assertGreater(v["u_dot_w_Rmf"], 0.5)
            self.assertGreater(v["pred_gap_at_direct3"], 4.0)


if __name__ == "__main__":
    unittest.main()
