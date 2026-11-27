#!/usr/bin/env python3
"""test_olmo3_c3_constants.py — Offline invariant verification for Stage C3 (WSP v1, Draft v4) constants, committed bank & bf16 audit."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
CONST_PATH = ROOT / "c3_prereg_constants.json"
BANK_PATH = ROOT / "c3_vector_bank.safetensors"
SCORING_PATH = ROOT / "c3_scoring.py"


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

    def test_01_committed_bank_and_scoring_hashes(self) -> None:
        prov = self.c["provenance"]
        self.assertEqual(prov["c3_bank_path"], "olmo3_virtual_and_highway/c3_vector_bank.safetensors")
        self.assertTrue(BANK_PATH.exists(), f"Missing committed bank {BANK_PATH}")
        self.assertEqual(sha256_file(BANK_PATH), prov["c3_bank_sha256"])
        self.assertEqual(BANK_PATH.stat().st_size, prov["c3_bank_bytes"])
        self.assertLess(prov["c3_bank_bytes"], 2_000_000)
        self.assertEqual(sha256_file(SCORING_PATH), prov["c3_scoring_sha256"])
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
            unm = sf.get_tensor("L24_pirate_unmasked_pre_unit").float()
            self.assertAlmostEqual(float(unm.norm()), 1.0, places=4)

    def test_03_all_13_splits_disjoint_and_valid(self) -> None:
        b = self.c["batteries"]
        self.assertEqual(len(b["calib_16"]), 16)
        self.assertEqual(len(b["reserve_8"]), 8)
        self.assertEqual(len(b["identity_20"]), 20)
        self.assertEqual(len(b["implied_20"]), 20)
        self.assertEqual(len(b["multiturn_12"]), 12)
        self.assertEqual(len(b["copula_pairs_12"]), 12)
        self.assertEqual(len(b["nomd_24"]), 24)
        self.assertEqual(len(b["math_20"]), 20)
        self.assertEqual(len(b["code_12"]), 12)
        self.assertEqual(len(b["refusal_8"]), 8)
        self.assertEqual(len(b["neutral_chat_16"]), 16)
        self.assertEqual(len(b["demo_showcase_ids_12"]), 12)
        self.assertEqual(len(self.c["split_counts"]), 13)
        self.assertIn("reader_gold_14", self.c["split_counts"])

        all_texts: list[str] = []
        for k in ("calib_16", "reserve_8", "identity_20", "implied_20", "nomd_24", "math_20", "code_12", "refusal_8", "neutral_chat_16"):
            all_texts.extend(x["text"] for x in b[k])
        for c in b["multiturn_12"]:
            all_texts.extend(c["turns"])
        for x in b["copula_pairs_12"]:
            all_texts.extend([x["self"], x["stranger"]])
        all_texts.extend(x["prompt"] for x in self.c["reader_gold_14"])
        normed = [" ".join(t.strip().lower().split()) for t in all_texts]
        self.assertEqual(len(normed), len(set(normed)), "Duplicate prompt detected across the 13 splits!")

        held_ids = {x["id"] for x in b["identity_20"]} | {x["id"] for x in b["implied_20"]}
        for did in b["demo_showcase_ids_12"]:
            self.assertIn(did, held_ids)

        for m in b["math_20"]:
            self.assertIsInstance(m["answer"], int)
        for cp in b["code_12"]:
            self.assertGreaterEqual(len(cp["tests"]), 2)
            for t in cp["tests"]:
                ast.parse(t)

    def test_04_ban_and_gold_14_invariants(self) -> None:
        lex = self.c["lexicons"]
        self.assertEqual(len(lex["pirate_seeds"]), 12)
        self.assertEqual(len(lex["nomd_seeds"]), 9)
        ban = lex["common_pirate_ban"]
        self.assertEqual(len(ban["single_tokens"]), ban["single_token_count"])
        self.assertEqual(len(ban["multi_token_sequences"]), ban["multi_token_sequence_count"])
        single_ids = {r["id"] for r in ban["single_tokens"]}
        for s in lex["pirate_seeds"]:
            self.assertIn(s["id"], single_ids)

        multi_stems = {r["stem"] for r in ban["multi_token_sequences"]}
        pat = re.compile(ban["surface_stem_regex"], re.IGNORECASE)
        for must_ban in ("ahoy", "matey", "avast", "buccaneer", "cutlass", "galleon", "parrot", "mutiny", "arrr", "grog"):
            self.assertIn(must_ban, multi_stems)
            self.assertIsNotNone(pat.search(f"Test {must_ban.capitalize()}!"))
        for dropped in ("gold", "island", "waves", "tide", "port", "bow", "stern"):
            self.assertNotIn(dropped, ban["stems"])

        gold = self.c["reader_gold_14"]
        self.assertEqual(len(gold), 14)
        self.assertEqual(sum(1 for g in gold if g["expected_pirate"]), 7)
        self.assertEqual(sum(1 for g in gold if not g["expected_pirate"]), 7)
        self.assertEqual(sum(1 for g in gold if g["style"] == "subtle_zero_stem"), 6)
        self.assertEqual(sum(1 for g in gold if g["style"] == "first_person_non_pirate_neg"), 2)
        for g in gold:
            if g["style"] in ("subtle_zero_stem", "broken_loop_no_stem", "first_person_non_pirate_neg"):
                self.assertIsNone(pat.search(g["answer"]), f"Gold item {g['id']} leaks ban stem")

    def test_05_bf16_audit_power_ladder_and_arms(self) -> None:
        self.assertEqual(self.c["edit_spec"]["dose_ladder_total_pmn_frac"], [0.075, 0.15, 0.30, 0.45, 0.60, 0.80])
        self.assertEqual(self.c["edit_spec"]["sham_alpha"], 0.05)
        self.assertEqual(self.c["edit_spec"]["commit_gate"]["rung_min_rank1_proj_ret"], 0.50)
        self.assertEqual(self.c["edit_spec"]["commit_gate"]["rung_min_frob_ret"], 0.80)
        self.assertEqual(self.c["edit_spec"]["commit_gate"]["reserve_max_single_pos_kl_max"], 4.0)

        # Verify reproducible 0-CU bf16 retention audit is populated across L21/24/27
        audit = self.c["bf16_retention_audit_0cu"]["layers"]
        for L in ("21", "24", "27"):
            self.assertIn(L, audit)
            self.assertGreater(audit[L]["by_alpha"]["0.05"]["stored_rank1_proj_ret"], 0.25)
            self.assertGreater(audit[L]["by_alpha"]["0.5"]["stored_rank1_proj_ret"], 0.90)

        pa = self.c["power_analysis_0cu"]
        self.assertEqual(pa["holm_m"], 4)
        h3_rf = pa["H3_N24_3bar_u050_p020_r015"]["room_floor_lp1.5_lr1.3_la0.60"]
        self.assertLess(h3_rf["p_point_bar_pass_r1"], h3_rf["p_perm_r1"])
        self.assertGreaterEqual(pa["H3_N24_3bar_u050_p020_r015"]["room_floor_lp1.5_lr1.3_la0.35"]["p_point_bar_pass_r1"], 0.90)
        self.assertGreaterEqual(pa["H4_N24_3bar_u035_p015_r010"]["room_floor_lp1.5_lr1.3_la0.45"]["p_point_bar_pass_r1"], 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
