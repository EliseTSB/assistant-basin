#!/usr/bin/env python3
"""test_olmo3_c3_constants.py — Offline invariant verification for Stage C3 (WSP v1) constants & compact vector bank."""
from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CONST_PATH = ROOT / "c3_prereg_constants.json"
BANK_PATH = REPO_ROOT / "work" / "olmo3_c3" / "c3_vector_bank.safetensors"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestStageC3Constants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c = json.loads(CONST_PATH.read_text(encoding="utf-8"))

    def test_01_bank_hash_size_and_tensors(self) -> None:
        prov = self.c["provenance"]
        self.assertTrue(BANK_PATH.exists(), f"Missing bank {BANK_PATH}")
        self.assertEqual(sha256_file(BANK_PATH), prov["c3_bank_sha256"])
        self.assertEqual(BANK_PATH.stat().st_size, prov["c3_bank_bytes"])
        self.assertLess(prov["c3_bank_bytes"], 2_000_000, "Compact C3 bank should be < 2 MB (no 2.11 GB r_lens upload)")
        with safe_open(str(BANK_PATH), framework="pt", device="cpu") as sf:
            keys = sorted(sf.keys())
            self.assertEqual(keys, prov["c3_bank_tensor_names"])
            self.assertEqual(len(keys), prov["c3_bank_n_tensors"])

    def test_02_unit_norms_and_low_gamma_masking(self) -> None:
        layers = self.c["edit_spec"]["layers"]
        drop_k = self.c["edit_spec"]["gamma_drop_ff"]
        n_rand = self.c["edit_spec"]["n_rand_seeds"]
        with safe_open(str(BANK_PATH), framework="pt", device="cpu") as sf:
            for L in layers:
                idx = sf.get_tensor(f"L{L}_gamma_drop_idx")
                self.assertEqual(idx.shape, (drop_k,))
                self.assertEqual(idx.tolist(), self.c["layer_geometry_0cu"][str(L)]["gamma_drop_idx"])
                for name in [
                    f"L{L}_pirate_lex_post_unit",
                    f"L{L}_pirate_lex_pre_unit",
                    f"L{L}_nomd_lex_post_unit",
                    f"L{L}_nomd_lex_pre_unit",
                    *[f"L{L}_rand_pirate_pre_unit_s{s}" for s in range(n_rand)],
                    *[f"L{L}_rand_nomd_pre_unit_s{s}" for s in range(n_rand)],
                ]:
                    v = sf.get_tensor(name).float()
                    self.assertEqual(v.shape, (4096,))
                    self.assertTrue(torch.isfinite(v).all().item(), f"Non-finite in {name}")
                    self.assertAlmostEqual(float(v.norm()), 1.0, places=4, msg=f"Unit norm failed for {name}")
                    if "_pre_unit" in name:
                        self.assertEqual(float(v[idx].abs().max()), 0.0, f"Low-gamma mask violated in {name}")
            # L24 unmasked vector exists and is unit norm
            unm = sf.get_tensor("L24_pirate_unmasked_pre_unit").float()
            self.assertAlmostEqual(float(unm.norm()), 1.0, places=4)

    def test_03_batteries_disjoint_and_valid(self) -> None:
        b = self.c["batteries"]
        self.assertEqual(len(b["calib_16"]), 16)
        self.assertEqual(len(b["reserve_8"]), 8)
        self.assertEqual(len(b["identity_16"]), 16)
        self.assertEqual(len(b["implied_16"]), 16)
        self.assertEqual(len(b["multiturn_12"]), 12)
        self.assertEqual(len(b["copula_pairs_12"]), 12)
        self.assertEqual(len(b["nomd_24"]), 24)
        self.assertEqual(len(b["math_20"]), 20)
        self.assertEqual(len(b["code_12"]), 12)
        self.assertEqual(len(b["refusal_8"]), 8)
        self.assertEqual(len(b["neutral_chat_16"]), 16)
        self.assertEqual(len(b["demo_showcase_ids_12"]), 12)

        # Verify demo showcase IDs exist in held-out identity + implied
        held_ids = {x["id"] for x in b["identity_16"]} | {x["id"] for x in b["implied_16"]}
        for did in b["demo_showcase_ids_12"]:
            self.assertIn(did, held_ids)

        # Verify every math answer is an int and every code test parses cleanly
        for m in b["math_20"]:
            self.assertIsInstance(m["answer"], int)
        for cp in b["code_12"]:
            self.assertGreaterEqual(len(cp["tests"]), 2)
            for t in cp["tests"]:
                ast.parse(t)

    def test_04_lexicon_and_gold_set_invariants(self) -> None:
        lex = self.c["lexicons"]
        self.assertEqual(len(lex["pirate_seeds"]), 12)
        self.assertEqual(len(lex["nomd_seeds"]), 9)
        self.assertGreater(lex["pirate_static_ban_count"], 50)
        # Ensure all 12 pirate seeds are inside static ban list
        ban_ids = {r["id"] for r in lex["pirate_static_ban_tokens"]}
        for s in lex["pirate_seeds"]:
            self.assertIn(s["id"], ban_ids)
        # Ensure balanced 6+ / 6- gold set
        gold = self.c["reader_gold_12"]
        self.assertEqual(len(gold), 12)
        self.assertEqual(sum(1 for g in gold if g["expected_pirate"]), 6)
        self.assertEqual(sum(1 for g in gold if not g["expected_pirate"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
