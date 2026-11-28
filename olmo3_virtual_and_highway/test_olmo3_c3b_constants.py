#!/usr/bin/env python3
"""test_olmo3_c3b_constants.py — 0-CU unit tests for Stage C3b (WSP v2, Draft v3) constants and 12-layer vector bank."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

import build_olmo3_c3b_constants as b3b

ROOT = Path(__file__).resolve().parent
CONST_PATH = ROOT / "c3b_prereg_constants.json"
BANK_PATH = ROOT / "c3b_vector_bank.safetensors"
C3_BANK_PATH = ROOT / "c3_vector_bank.safetensors"
SCORING_PATH = ROOT / "c3_scoring.py"


class TestStageC3bConstants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.const = json.loads(CONST_PATH.read_text(encoding="utf-8"))
        cls.bank = load_file(str(BANK_PATH))
        cls.c3_bank = load_file(str(C3_BANK_PATH))

    def test_01_committed_bank_schema_and_scoring_hashes(self) -> None:
        self.assertEqual(self.const["schema_version"], "olmo3-stage-c3b-wsp-v2-prereg-constants-v3")
        prov = self.const["provenance"]
        self.assertEqual(prov["c3b_bank_sha256"], b3b.sha256_file(BANK_PATH))
        self.assertEqual(prov["c3_scoring_sha256"], b3b.sha256_file(SCORING_PATH))
        self.assertEqual(prov["c3_bank_source_sha256"], b3b.sha256_file(C3_BANK_PATH))
        self.assertEqual(len(self.bank), 87)
        self.assertEqual(prov["c3b_bank_n_tensors"], 87)
        self.assertEqual(prov["c3b_bank_tensor_names"], sorted(self.bank.keys()))

    def test_02_12_layer_unit_norms_masking_and_disjoint_d4_spec(self) -> None:
        self.assertEqual(self.const["edit_spec"]["layers"], list(range(16, 28)))
        for k in ("L24_pirate_lex_post_unit", "L24_pirate_lex_pre_unit", "L24_pirate_unmasked_pre_unit"):
            self.assertTrue(torch.equal(self.bank[k], self.c3_bank[k]), f"C3 continuity vector {k} differs")
        for L in range(16, 28):
            idx = self.bank[f"L{L}_gamma_drop_idx"]
            self.assertEqual(list(idx.shape), [4])
            self.assertEqual(idx.dtype, torch.int64)
            for s in range(3):
                for kind in ("pirate", "nomd"):
                    v = self.bank[f"L{L}_rand_{kind}_pre_unit_s{s}"]
                    self.assertEqual(list(v.shape), [4096])
                    self.assertEqual(v.dtype, torch.float32)
                    self.assertAlmostEqual(float(v.norm().item()), 1.0, places=5)
                    for di in idx.tolist():
                        self.assertEqual(float(v[di].item()), 0.0, f"Masked coord {di} non-zero on L{L} {kind} s{s}")
        d4 = self.const["d4_spec"]
        self.assertEqual(len(d4["prompt_ids"]), 12)
        self.assertEqual(d4["prompt_ids"][:8], [f"res_{i:02d}" for i in range(8)])
        self.assertEqual(d4["prompt_ids"][8:], [f"chat_{i:02d}" for i in range(4)])
        # 100% disjoint from calib_16
        calib_ids = {p["id"] for p in self.const["batteries"]["calib_16"]}
        self.assertEqual(set(d4["prompt_ids"]) & calib_ids, set())
        c2 = d4["dual_criterion"]["criterion_2_full_vocab_and_efficiency_bound"]
        self.assertIn("masked_over_unmasked_logit_l2_ratio <= 1.10", c2)
        self.assertIn("mismasked_over_unmasked_logit_l2_ratio <= 1.25", c2)
        self.assertIn("masked_over_unmasked_efficiency_ratio > 1.15", c2)
        self.assertIn("mismasked_over_unmasked_efficiency_ratio < 1.05", c2)
        self.assertIn("48 forwards", d4["forward_count_exact"])

    def test_03_all_14_splits_disjoint_and_new_prompts_unseen(self) -> None:
        counts = b3b.verify_disjoint_prompts()
        self.assertEqual(counts["nomd_32"], 32)
        self.assertEqual(counts["identity_20"], 20)
        self.assertEqual(counts["implied_20"], 20)
        self.assertEqual(counts["reader_gold_16_v2"], 16)
        self.assertEqual(counts["reader_preflight_canary_2"], 2)
        self.assertEqual(counts, self.const["split_counts"])

    def test_04_gold_16_v2_cue_free_archaic_and_strict_plural_ban_invariants(self) -> None:
        gold = self.const["reader_gold_16_v2"]
        self.assertEqual(len(gold), 16)
        self.assertEqual(sum(1 for g in gold if g["expected_pirate"]), 7)
        self.assertEqual(sum(1 for g in gold if not g["expected_pirate"]), 9)
        self.assertEqual(sum(1 for g in gold if g["style"] == "subtle_zero_stem"), 6)
        fp = [g for g in gold if g["style"] == "first_person_non_pirate_neg"]
        self.assertEqual(len(fp), 4)
        self.assertEqual(sum(1 for g in fp if g.get("subtype") == "modern_trade"), 2)
        self.assertEqual(sum(1 for g in fp if g.get("subtype") == "archaic_dialect"), 2)

        stem_pat = re.compile(self.const["lexicons"]["common_pirate_ban"]["surface_stem_regex"], re.IGNORECASE)
        strict_ext_pat = re.compile(self.const["lexicons"]["strict_pirate_extended_ban_regex"], re.IGNORECASE)
        cue_terms = set(self.const["lexicons"]["reader_prompt_cue_words_excluded_from_subtle_gold"])
        for req_cue in ("mariner", "seafaring", "navigation", "command", "articles", "coastal", "marooned", "charts"):
            self.assertIn(req_cue, cue_terms)
        # Verify DeepSeek F5b plurals are caught by strict_pirate_extended_ban_regex!
        for leak_word in ("galleons", "sloops", "frigates", "cutlasses", "schooners"):
            self.assertIsNotNone(strict_ext_pat.search(f"The {leak_word} sailed at dawn."), f"{leak_word} missed!")

        for g in gold:
            if g["style"] != "explicit":
                self.assertIsNone(stem_pat.search(g["answer"]), f"Gold {g['id']} leaks stem")
                self.assertIsNone(stem_pat.search(g["prompt"]), f"Gold prompt {g['id']} leaks stem")
            if g["style"] == "subtle_zero_stem":
                self.assertIsNone(strict_ext_pat.search(g["answer"]), f"Subtle gold {g['id']} leaks strict ban/cue")
                self.assertIsNone(strict_ext_pat.search(g["prompt"]), f"Subtle gold prompt {g['id']} leaks strict ban/cue")

        gg = self.const["reader_kappa_pool"]["gold_gate"]
        self.assertEqual(gg["exact_full_gate_probabilities"], b3b.compute_exact_gold_gate_probs())
        for c in self.const["reader_kappa_pool"]["preflight_canary_2"]:
            self.assertIn("expected_pirate", c)
            self.assertIn("expected_coherent", c)

    def test_05_bf16_audit_ladder_endpoints_exhaustive_rules_and_m2_power(self) -> None:
        self.assertEqual(self.const["edit_spec"]["dose_ladder_total_pmn_frac"], [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6])
        h2_fam = self.const["holm_2_family"]
        self.assertEqual(h2_fam["m"], 2)
        self.assertIn("BLOCKED_BY_C3b.0", h2_fam["verdict_precedence"])
        self.assertNotIn("BLOCKED_BY_C3.0", h2_fam["verdict_precedence"])
        for hyp in h2_fam["hypotheses"]:
            self.assertIn("fail_co_primary_guard_rule", hyp)
            self.assertIn("fail_null_rule", hyp)
            self.assertIn("FAIL", hyp["fail_co_primary_guard_rule"])
        ri = self.const["endpoint_definitions"]["restore_check_invariant"]
        self.assertEqual(ri["sanity_floor_value"], 279)
        self.assertIn("restore_checks == layers_compiled_total", ri["primary_binding_rule"])
        self.assertEqual(self.const["d_above_cap_spec"]["max_extra_gens"], 36)
        self.assertEqual(self.const["d_above_cap_spec"]["max_extra_tokens"], 2688)
        self.assertIn("D_ABOVE_CAP_UNREAD", self.const["d_above_cap_spec"]["pirate_slice"]["reader_gate"])

        bf = self.const["bf16_retention_audit_0cu"]["layers"]
        self.assertEqual(sorted(int(k) for k in bf.keys()), list(range(16, 28)))
        for L in range(16, 28):
            self.assertGreaterEqual(bf[str(L)]["by_alpha"]["0.05"]["stored_rank1_proj_ret"], 0.25)
            self.assertGreaterEqual(bf[str(L)]["by_alpha"]["0.25"]["stored_rank1_proj_ret"], 0.80)
            self.assertGreaterEqual(bf[str(L)]["by_alpha"]["1.0"]["stored_rank1_proj_ret"], 0.99)

        pw = self.const["power_analysis_0cu"]
        self.assertEqual(pw["holm_m"], 2)
        self.assertGreaterEqual(pw["H1_N40_bar020"]["pRand_0.05_delta_0.30"]["p_subgate_pass_r1"], 0.90)
        self.assertGreaterEqual(pw["H1_coherence_guard_pass_rate"]["false_incoh_0.10"]["p_both_coherence_guards_pass"], 0.95)
        h2_pw = pw["H2_N32_3bar_u050_p020_r015"]
        self.assertLessEqual(h2_pw["c3_observed_null_lp3.3_lr3.0_la3.00"]["p_point_bar_pass_r1"], 0.01)
        self.assertGreaterEqual(h2_pw["baseline_matched_hypothetical_lp3.3_lr3.0_la1.00"]["p_point_bar_pass_r1"], 0.90)


if __name__ == "__main__":
    unittest.main(verbosity=2)
