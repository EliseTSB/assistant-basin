#!/usr/bin/env python3
"""test_olmo3_c3_design_stdlib.py — Pure-stdlib (/usr/bin/python3 -I) design-pin & scoring-contract verifier for Stage C3 (WSP v1, Draft v3)."""
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

    def test_02_holm_4_precedence_symmetric_broken_and_bars(self) -> None:
        h = self.c["holm_4_family"]
        self.assertEqual(h["alpha"], 0.05)
        self.assertEqual(h["m"], 4)
        self.assertEqual(h["verdict_precedence"], c3s.VERDICT_PRECEDENCE)
        self.assertEqual(len(h["verdict_precedence"]), 9)
        for v in h["verdict_precedence"]:
            self.assertIn(v, self.design)

        # Symmetric broken guards (B12)
        hyps = {x["id"]: x for x in h["hypotheses"]}
        self.assertIn("<= 0.15 AND broken(Rand_mean) <= 0.15", hyps["H1"]["broken_bar"])
        self.assertIn("<= 0.15 AND broken(Lex@d_common) <= 0.15", hyps["H2"]["broken_bar"])
        self.assertIn("<= 0.10 AND broken(Rand_mean) <= 0.10", hyps["H3"]["broken_bar"])
        self.assertIn("<= 0.10 AND broken(Rand_mean) <= 0.10", hyps["H4"]["broken_bar"])

        expected_pins = [
            ("H1", "+0.20", "0.15", "common_pirate_ban"),
            ("H2", "+0.20", "0.15", "EQ-C3.2(Con@d_common)"),
            ("H3", "+0.50", "+0.20", "+0.15"),
            ("H4", "+0.35", "+0.15", "+0.10"),
        ]
        for hid, a, b, c in expected_pins:
            m = re.search(rf"^\|\s*\*\*`{hid}`\*\*.*$", self.design, re.M)
            self.assertIsNotNone(m, f"Missing Holm row {hid}")
            for token in (a, b, c):
                self.assertIn(token, m.group(0), f"Pin {token} missing in {hid}")

    def test_03_all_8_gates_match_1_to_1_and_kappa_pool_pinned(self) -> None:
        gcm = self.c["gate_condition_matrix"]
        expected_keys = [
            "C3.0.1_zero_hook_restore_liveness_and_batch_parity",
            "C3.0.2_finiteness_mask_mismask_norm_stability_and_ban",
            "C3.0.3_completeness_and_plain_sanity",
            "C3.0.4_provenance",
            "C3.0_direction_quality",
            "C3.0_commit_gate",
            "C3.0_NoMD_room",
            "C3.R_dual_blind_reader_gate",
        ]
        self.assertEqual(len(gcm), 8)
        for key in expected_keys:
            self.assertIn(key, gcm)
            self.assertTrue(gcm[key]["conditions_covered"])
            self.assertTrue(gcm[key]["desk_satisfiability_proof"])
        for tag in ("C3.0.1", "C3.0.2", "C3.0.3", "C3.0.4", "C3.0_direction_quality", "C3.0_commit_gate", "C3.0_NoMD_room", "C3.R"):
            self.assertIn(f"**`{tag}`**", self.design)

        rkp = self.c["reader_kappa_pool"]
        self.assertEqual(len(rkp["core_arms_7"]), 7)
        self.assertEqual(rkp["gold_gate"]["pooled_min_correct"], 12)
        self.assertEqual(rkp["gold_gate"]["subtle_zero_stem_min_correct"], 5)
        self.assertEqual(rkp["gold_gate"]["first_person_non_pirate_neg_min_correct"], 2)

    def test_04_full_3bar_power_table_pinned_in_design(self) -> None:
        pa = self.c["power_analysis_0cu"]
        self.assertIn(str(pa["seed"]), self.design)
        self.assertIn(f"{pa['H1_N40_bar020']['pRand_0.05_delta_0.30']['p_full_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H2_N40_bar020']['pLex_0.05_delta_0.30']['p_full_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H3_N24_full_3bar_u050_p020_r015']['room_floor_lp1.5_lr1.3_la0.60']['p_full_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H3_N24_full_3bar_u050_p020_r015']['moderate_lp2.2_lr1.8_la0.80']['p_full_pass_r1']:.3f}", self.design)
        self.assertIn(f"{pa['H4_N24_full_3bar_u035_p015_r010']['room_floor_lp1.5_lr1.3_la0.45']['p_full_pass_r1']:.3f}", self.design)

    def test_05_c3_scoring_multiline_broken_and_verdict_precedence(self) -> None:
        # B13 worked example: multiline bullet MUST match (re.MULTILINE pinned!)
        sample_md = "Here are the steps:\n- one\n- two"
        n_ct, flags = c3s.construct_types(sample_md)
        self.assertEqual(n_ct, 1)
        self.assertTrue(flags["unordered_bullet"])
        self.assertTrue(c3s.has_markdown(sample_md))
        self.assertEqual(c3s.markdown_hits(sample_md), 2)

        plain_prose = "Brew pour-over coffee by wetting the filter and pouring hot water slowly in circles."
        self.assertEqual(c3s.construct_types(plain_prose)[0], 0)
        self.assertFalse(c3s.has_markdown(plain_prose))

        # B13 is_broken pinned tokenizer & case-folding checks
        self.assertFalse(c3s.is_broken(plain_prose))
        self.assertTrue(c3s.is_broken("   "))
        self.assertTrue(c3s.is_broken("a b"))
        self.assertTrue(c3s.is_broken("Bad token \ufffd here"))
        self.assertTrue(c3s.is_broken("timber canvas timber wheel timber canvas timber wheel timber canvas timber wheel"))
        self.assertTrue(c3s.is_broken("One two three four five, ONE TWO THREE FOUR FIVE."))

        # B5/F6/B16 verdict precedence collision checks
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
        # B16: Pillar B failing has_md co-primary bars MUST return FAIL, never EFFECT_BELOW_BAR!
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "co_primary_bars_pass": False}), "FAIL")
        # Significant small effect with co-primary OK -> EFFECT_BELOW_BAR
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "primary_effect_bar_pass": False}), "EFFECT_BELOW_BAR")
        # Precedence: C3.0 beats DOSE_INVALID; DIRECTION_INVALID beats DOSE_INVALID; DOSE_INVALID beats NOMD_ROOM
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "c3_0_pass": False, "dose_valid": False}), "BLOCKED_BY_C3.0")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "direction_valid": False, "dose_valid": False}), "DIRECTION_INVALID")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "dose_valid": False, "nomd_room_pass": False}), "DOSE_INVALID")
        self.assertEqual(c3s.resolve_verdict(**{**base_ok, "eq_c3_2_pass": False, "broken_bar_pass": False}), "CAPABILITY_FAIL")

        # Kappa, raw agreement, math_exact, reader JSON parser
        self.assertAlmostEqual(c3s.cohens_kappa([True, False, True, False], [True, False, True, False]), 1.0)
        self.assertAlmostEqual(c3s.raw_agreement([True, False, True, False], [True, False, True, True]), 0.75)
        self.assertTrue(c3s.math_exact("47 + 85 = 132.\nFINAL: 132", 132))
        self.assertFalse(c3s.math_exact("FINAL: 131", 132))
        parsed = c3s.parse_reader_json('```json\n{"pirate_persona": true, "coherent": true}\n```')
        self.assertEqual(parsed, {"pirate_persona": True, "coherent": True})

    def test_06_policy_and_compute_sanity(self) -> None:
        self.assertIn("11-07", self.design)
        self.assertIn("never saves or publishes any edited model checkpoint", self.design)
        self.assertIn("117,760", self.design)
        self.assertIn("153,088", self.design)


if __name__ == "__main__":
    unittest.main(verbosity=2)
