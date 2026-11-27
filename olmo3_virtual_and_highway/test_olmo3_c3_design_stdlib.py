#!/usr/bin/env python3
"""test_olmo3_c3_design_stdlib.py — Pure-stdlib (/usr/bin/python3 -I) design-pin & executable scoring verifier for Stage C3 (WSP v1, Draft v4)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT / "OLMO3-WSP-C3-DESIGN.md"
CONST_PATH = ROOT / "c3_prereg_constants.json"
BANK_PATH = ROOT / "c3_vector_bank.safetensors"
SCORING_PATH = ROOT / "c3_scoring.py"


def _load_scoring():
    spec = importlib.util.spec_from_file_location("c3_scoring", SCORING_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


c3s = _load_scoring()


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

    def test_01_all_4_committed_artifacts_hashed_and_pinned(self) -> None:
        const_sha = sha256_file(CONST_PATH)
        const_bytes = CONST_PATH.stat().st_size
        bank_sha = self.c["provenance"]["c3_bank_sha256"]
        bank_bytes = self.c["provenance"]["c3_bank_bytes"]
        scoring_sha = self.c["provenance"]["c3_scoring_sha256"]
        r_lens_sha = self.c["provenance"]["r_lens_sha256"]

        self.assertTrue(BANK_PATH.exists())
        self.assertEqual(sha256_file(BANK_PATH), bank_sha)
        self.assertEqual(BANK_PATH.stat().st_size, bank_bytes)
        self.assertEqual(sha256_file(SCORING_PATH), scoring_sha)

        for tok in (const_sha, f"{const_bytes:,}", bank_sha, f"{bank_bytes:,}", scoring_sha, r_lens_sha):
            self.assertIn(tok, self.design)

    def test_02_holm_stepdown_stop_rule_and_h2_eq_alignment(self) -> None:
        # C1 counterexample: (0.0130, 0.0150, 0.30, 0.90) MUST reject NOTHING!
        r_stop = c3s.holm_reject({"H1": 0.0130, "H2": 0.0150, "H3": 0.30, "H4": 0.90}, alpha=0.05)
        self.assertEqual(r_stop, {"H1": False, "H2": False, "H3": False, "H4": False})

        # Positive step-down through rank 3, stopping at rank 4
        r_pos = c3s.holm_reject({"H1": 0.0100, "H2": 0.0150, "H3": 0.0240, "H4": 0.0600}, alpha=0.05)
        self.assertEqual(r_pos, {"H1": True, "H2": True, "H3": True, "H4": False})

        # Unsorted input stops at rank 3 (0.0300 > 0.025) so rank 4 (0.0400 <= 0.05) is NOT rejected
        r_perm = c3s.holm_reject({"H4": 0.0050, "H2": 0.0160, "H1": 0.0300, "H3": 0.0400}, alpha=0.05)
        self.assertEqual(r_perm, {"H4": True, "H2": True, "H1": False, "H3": False})

        h = self.c["holm_4_family"]
        self.assertEqual(h["alpha"], 0.05)
        self.assertEqual(h["m"], 4)
        self.assertIn("stop at the first failure", h["multiplicity"])
        self.assertEqual(h["verdict_precedence"], c3s.VERDICT_PRECEDENCE)
        self.assertEqual(len(h["verdict_precedence"]), 9)
        for v in h["verdict_precedence"]:
            self.assertIn(v, self.design)

        # F2: H2 capability gate alignment — gating on Con@d_common, logged-only on Lex@d_common in BOTH JSON and DESIGN
        hyps = {x["id"]: x for x in h["hypotheses"]}
        self.assertIn("logged-only for Lex@d_common", hyps["H2"]["pass_rule"])
        self.assertIn("WSP_Pirate_Contrastive@d_common", self.c["capability_preservation_gate_EQ_C3_2"]["gating_arms"])
        self.assertIn("WSP_Pirate_Lexicon@d_common", self.c["capability_preservation_gate_EQ_C3_2"]["logged_only_arms"])

        # M1: length-ratio co-primary guard in H3 and H4
        self.assertIn("[0.85, 1.15]", hyps["H3"]["point_bar"])
        self.assertIn("[0.85, 1.15]", hyps["H4"]["point_bar"])

        # Symmetric broken guards (B12)
        self.assertIn("<= 0.15 AND broken(Rand_mean) <= 0.15", hyps["H1"]["broken_bar"])
        self.assertIn("<= 0.15 AND broken(Lex@d_common) <= 0.15", hyps["H2"]["broken_bar"])
        self.assertIn("<= 0.10 AND broken(Rand_mean) <= 0.10", hyps["H3"]["broken_bar"])
        self.assertIn("<= 0.10 AND broken(Rand_mean) <= 0.10", hyps["H4"]["broken_bar"])

        expected_pins = [
            ("H1", "+0.20", "0.15", "common_pirate_ban"),
            ("H2", "+0.20", "0.15", "EQ-C3.2(Con@d_common)"),
            ("H3", "+0.50", "+0.20", "[0.85, 1.15]"),
            ("H4", "+0.35", "+0.15", "[0.85, 1.15]"),
        ]
        for hid, a, b, c in expected_pins:
            m = re.search(rf"^\|\s*\*\*`{hid}`\*\*.*$", self.design, re.M)
            self.assertIsNotNone(m, f"Missing Holm row {hid}")
            for token in (a, b, c):
                self.assertIn(token, m.group(0), f"Pin {token} missing in {hid}")

    def test_03_all_8_gates_commit_representability_and_reader_alignment(self) -> None:
        gcm = self.c["gate_condition_matrix"]
        self.assertEqual(len(gcm), 8)
        for tag in ("C3.0.1", "C3.0.2", "C3.0.3", "C3.0.4", "C3.0_direction_quality", "C3.0_commit_gate", "C3.0_NoMD_room", "C3.R"):
            self.assertIn(f"**`{tag}`**", self.design)

        # C2/F1: bf16 representability thresholds + formulas pinned
        bf16 = self.c["edit_spec"]["bf16_storage_retention"]
        self.assertEqual(bf16["rung_min_rank1_proj_ret"], 0.50)
        self.assertEqual(bf16["rung_min_frob_ret"], 0.80)
        self.assertIn("dose_unrepresentable", gcm["C3.0_commit_gate"]["formula"])

        # M3/F4/F5: gold gate alignment + zero occupation leak in READER_SYSTEM_PROMPT
        rkp = self.c["reader_kappa_pool"]
        self.assertEqual(rkp["gold_gate"]["pooled_min_correct"], 13)
        self.assertEqual(rkp["gold_gate"]["subtle_zero_stem_min_correct"], 6)
        self.assertEqual(rkp["gold_gate"]["first_person_non_pirate_neg_min_correct"], 2)
        prompt_low = c3s.READER_SYSTEM_PROMPT.lower()
        for forbidden_trade in ("locksmith", "electrician", "carpenter", "baker"):
            self.assertNotIn(forbidden_trade, prompt_low, f"Gold trade {forbidden_trade!r} leaked into READER_SYSTEM_PROMPT")

    def test_04_point_bar_power_table_pinned_in_design(self) -> None:
        pa = self.c["power_analysis_0cu"]
        self.assertIn("Point-Bar Power", self.design)
        self.assertIn(r"P(point-bar pass \| rank-1)", self.design)
        self.assertIn(str(pa["seed"]), self.design)
        self.assertIn(f"{pa['H1_N40_bar020']['pRand_0.05_delta_0.30']['p_point_bar_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H2_N40_bar020']['pLex_0.05_delta_0.30']['p_point_bar_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H3_N24_3bar_u050_p020_r015']['room_floor_lp1.5_lr1.3_la0.60']['p_point_bar_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H4_N24_3bar_u035_p015_r010']['room_floor_lp1.5_lr1.3_la0.45']['p_point_bar_pass_r1']:.3f}", self.design)

    def test_05_c3_scoring_multiline_broken_code_pass_and_verdict(self) -> None:
        # L1 + B13: scalar construct_types + re.MULTILINE
        sample_md = "Here are the steps:\n- one\n- two"
        self.assertEqual(c3s.construct_types(sample_md), 1)
        self.assertIsInstance(c3s.construct_types(sample_md), int)
        n_ct, flags = c3s.construct_types_detail(sample_md)
        self.assertEqual(n_ct, 1)
        self.assertTrue(flags["unordered_bullet"])
        self.assertTrue(c3s.has_markdown(sample_md))
        self.assertEqual(c3s.markdown_hits(sample_md), 2)

        plain_prose = "Brew pour-over coffee by wetting the filter and pouring hot water slowly in circles."
        self.assertEqual(c3s.construct_types(plain_prose), 0)
        self.assertFalse(c3s.has_markdown(plain_prose))

        # F6/L4: coherent step-by-step prose & short refrain NOT broken; true low-diversity loop IS broken!
        f6_coherent = (
            "First, the user should unplug the device. Second, the user should press and hold the button. "
            "Third, the user should wait ten seconds. Fourth, the user should restart it."
        )
        self.assertFalse(c3s.is_broken(f6_coherent))
        self.assertFalse(c3s.is_broken("Brew pour-over coffee slowly and patiently. Bloom the grounds, then brew pour-over coffee slowly and patiently again."))
        self.assertTrue(c3s.is_broken("   "))
        self.assertTrue(c3s.is_broken("a b"))
        self.assertTrue(c3s.is_broken("Bad token \ufffd here"))
        self.assertTrue(c3s.is_broken("timber canvas timber wheel timber canvas timber wheel timber canvas timber wheel timber canvas timber wheel"))

        # F3: executable code_pass + extract_code_block tests
        tests_sum = ["assert sum_even([1, 2, 3, 4, 6]) == 12", "assert sum_even([1, 3, 5]) == 0"]
        self.assertTrue(c3s.code_pass("Here it is:\n```python\ndef sum_even(nums):\n    return sum(x for x in nums if x % 2 == 0)\n```", tests_sum, "sum_even"))
        self.assertTrue(c3s.code_pass("def sum_even(nums):\n    return sum(x for x in nums if x % 2 == 0)", tests_sum, "sum_even"))
        self.assertFalse(c3s.code_pass("```python\ndef sum_even(nums):\n    return 0\n```", tests_sum, "sum_even"))
        self.assertFalse(c3s.code_pass("```python\ndef wrong_name(nums):\n    return sum(nums)\n```", tests_sum, "sum_even"))
        self.assertFalse(c3s.code_pass("def sum_even(nums):\n    while True: pass", tests_sum, "sum_even", timeout_sec=0.3))

        # Precedence & B16/M1 co-primary failure -> FAIL
        base_ok = dict(
            c3_0_pass=True,
            direction_valid=True,
            dose_valid=True,
            requires_nomd_room=True,
            nomd_room_pass=True,
            reader_dependent=False,
            c3_r_pass=True,
            eq_c3_2_pass=True,
            broken_bar_pass=True,
            holm_sig=True,
            primary_effect_bar_pass=True,
            co_primary_bars_pass=True,
        )
        self.assertEqual(c3s.resolve_verdict(**base_ok), "PASS")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "co_primary_bars_pass": False}), "FAIL")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "primary_effect_bar_pass": False}), "EFFECT_BELOW_BAR")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "c3_0_pass": False, "dose_valid": False}), "BLOCKED_BY_C3.0")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "direction_valid": False, "dose_valid": False}), "DIRECTION_INVALID")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "dose_valid": False, "nomd_room_pass": False}), "DOSE_INVALID")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "eq_c3_2_pass": False, "broken_bar_pass": False}), "CAPABILITY_FAIL")

        # Math, kappa, raw agreement, reader JSON
        self.assertAlmostEqual(c3s.cohens_kappa([True, False, True, False], [True, False, True, False]), 1.0)
        self.assertAlmostEqual(c3s.raw_agreement([True, False, True, False], [True, False, True, True]), 0.75)
        self.assertTrue(c3s.math_exact("47 + 85 = 132.\nFINAL: 132", 132))
        self.assertFalse(c3s.math_exact("47 + 85 = 132", 132, strict_final_only=True))
        self.assertEqual(c3s.parse_reader_json('```json\n{"pirate_persona": true, "coherent": true}\n```'), {"pirate_persona": True, "coherent": True})

    def test_06_policy_and_compute_sanity(self) -> None:
        self.assertIn("11-07", self.design)
        self.assertIn("never saves or publishes any edited model checkpoint", self.design)
        self.assertIn("117,760", self.design)
        self.assertIn("153,088", self.design)


if __name__ == "__main__":
    unittest.main(verbosity=2)
