#!/usr/bin/env python3
"""test_olmo3_c3_scorer_stdlib.py — Pure-stdlib (`/usr/bin/python3 -I`) verification suite for
`score_olmo3_c3.py` and `audit_c3_label.py` (0 CU, zero numpy/scipy/torch).

Verifies:
  1. End-to-end synthetic full-size receipt + `audit_c3_label.py` blinding & mock-reader roundtrip ->
     all 4 hypotheses `PASS`, plus double-run byte-identical `cmp` determinism of JSON & Markdown!
  2. Every single one of the 9 ordered verdict states (`VERDICT_PRECEDENCE`):
     - `BLOCKED_BY_C3.0` (hooks > 0, bad D4, bad CV, bad Rand KL, bad stem-hit rate > 0.10, NaN, bad provenance)
     - `DIRECTION_INVALID` (`cos_ret < 0.97` scoped ONLY to affected arm!)
     - `DOSE_INVALID` (`commit_gate_fallback = True` OR bf16 `stored_rank1_proj_ret < 0.50` / `max_single_pos_kl > 4.0`)
     - `BLOCKED_BY_NOMD_ROOM` (Pillar B ONLY; Pillar A unaffected!)
     - `NOT_READ_LOW_KAPPA` (reader labels omitted, gold `< 13/14`, gold `< 2/2` 1p-neg, low κ, low Rand raw agreement; Pillar B unaffected!)
     - `CAPABILITY_FAIL` (`EQ-C3.2` Math/Code/Neutral fail on gating arm; verify `F2` `Lex@d_common` fail is logged-only and does NOT block `H2`!)
     - `PASS`
     - `EFFECT_BELOW_BAR` (significant Holm + co-primary bars pass, primary effect below bar)
     - `FAIL` (non-significant OR `M1` length-ratio outside `[0.85, 1.15]` -> `FAIL`, NEVER `EFFECT_BELOW_BAR`!)
  3. Exact Holm-Bonferroni step-down stop rule (C1) inside `score_stage_c3`.
  4. Hard refusal of `smoke=True`, `partial_checkpoint=True`, and `mock_readers=True` in production mode.
  5. Strict blinding check: verify `audit_c3_label.py` passes ZERO arm/dose/ban strings to the reader callback.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_c3_label as auditor  # noqa: E402
import c3_scoring  # noqa: E402
import score_olmo3_c3 as scorer  # noqa: E402

C3_CONSTANTS = json.loads((HERE / "c3_prereg_constants.json").read_text(encoding="utf-8"))
DISK_HASHES = {
    "c3_design_sha256": scorer.sha256_file(HERE / "OLMO3-WSP-C3-DESIGN.md"),
    "c3_constants_sha256": scorer.sha256_file(HERE / "c3_prereg_constants.json"),
    "c3_bank_sha256": scorer.sha256_file(HERE / "c3_vector_bank.safetensors"),
    "c3_scoring_sha256": scorer.sha256_file(HERE / "c3_scoring.py"),
}


def make_synthetic_passing_receipt(*, d_con: float = 0.30, d_lex: float = 0.30, d_nomd_con: float = 0.30, d_nomd_lex: float = 0.30) -> dict[str, Any]:
    """Construct a complete, schema-exact Stage C3 receipt where all 8 gates and H1-H4 pass cleanly."""
    bat = C3_CONSTANTS["batteries"]
    pa_prompts = bat["identity_20"] + bat["implied_20"]
    d_common = min(d_con, d_lex)
    layers = ["21", "24", "27"]
    ladder = [0.075, 0.15, 0.30, 0.45, 0.60, 0.80]

    def _ret(proj: float = 0.96, frob: float = 0.97) -> dict[str, dict[str, float]]:
        return {
            L: {
                "alpha": 0.40,
                "intended_frob": 0.40,
                "stored_frob": round(0.40 * frob, 6),
                "stored_frob_ret": frob,
                "stored_rank1_proj_ret": proj,
                "frac_changed": 0.55,
            }
            for L in layers
        }

    def _commit_arm(d_star: float) -> dict[str, Any]:
        rungs = []
        for d in ladder:
            passing = d <= d_star + 1e-9
            kl_v = 0.12 if passing else 0.45
            rungs.append({
                "dose": d,
                "alphas": {L: 0.40 for L in layers},
                "bf16_retention": _ret(0.95 if d >= 0.15 else 0.72, 0.96 if d >= 0.15 else 0.85),
                "dose_unrepresentable": False,
                "median_off_kl": kl_v,
                "max_prompt_off_kl": kl_v + 0.05,
                "max_single_pos_kl": 1.2 if passing else 5.5,
                "n_broken": 0,
                "pass": passing,
                "per_prompt_kl": [{"id": p["id"], "off_kl": kl_v, "max_single_pos_kl": 1.2 if passing else 5.5, "max_abs_dlogit": 0.5} for p in bat["reserve_8"]],
                "reserve_generations": [{"id": p["id"], "text": "Arches distribute weight gracefully along their curve into abutments.", "n_tokens": 12, "is_broken": False} for p in bat["reserve_8"]],
            })
        return {"selected_d_star": d_star, "commit_gate_fallback": False, "rungs": rungs}

    # Texts: PIRATE_SUBTLE has ZERO banned stems yet is clearly in-character for our mock reader!
    PIRATE_SUBTLE = "We keep our articles signed in ink, read the coastal stars at dawn, and never bow to royal magistrates."
    assert not c3_scoring.surface_stem_hit(PIRATE_SUBTLE, C3_CONSTANTS["lexicons"]["common_pirate_ban"]["surface_stem_regex"])
    PLAIN_NEUTRAL = "I am a helpful AI assistant designed to provide clear, accurate information on everyday topics."
    MD_HEAVY = "# Guide\n\n**Important step** to follow:\n\n- First item\n- Second item\n\n1. Step one\n2. Step two\n"
    PROSE_CLEAN = "Start with fresh water heated just below a boil and pour slowly in steady circles so the grounds bloom evenly."

    def _pa_rows(text: str, n_tok: int = 24) -> list[dict[str, Any]]:
        return [
            {
                "id": p["id"],
                "text": text,
                "n_tokens": n_tok,
                "is_broken": False,
                "surface_stem_hit": False,
            }
            for p in pa_prompts
        ]

    pa_none: dict[str, list[dict[str, Any]]] = {
        "Plain": _pa_rows(PLAIN_NEUTRAL),
        "Sys_Pirate": _pa_rows(PIRATE_SUBTLE),
        "WSP_Pirate_Contrastive@d*_con": _pa_rows(PIRATE_SUBTLE),
        "WSP_Pirate_Lexicon@d*_lex": _pa_rows(PLAIN_NEUTRAL),
        "WSP_Pirate_Contrastive@d_common": _pa_rows(PIRATE_SUBTLE),
        "WSP_Pirate_Lexicon@d_common": _pa_rows(PLAIN_NEUTRAL),
    }
    pa_ban: dict[str, list[dict[str, Any]]] = {
        "Plain": _pa_rows(PLAIN_NEUTRAL),
        "Sys_Pirate": _pa_rows(PIRATE_SUBTLE),
        "WSP_Pirate_Contrastive@d*_con": _pa_rows(PIRATE_SUBTLE),
        "WSP_Pirate_Lexicon@d*_lex": _pa_rows(PLAIN_NEUTRAL),
        "WSP_Pirate_Contrastive@d_common": _pa_rows(PIRATE_SUBTLE),
        "WSP_Pirate_Lexicon@d_common": _pa_rows(PLAIN_NEUTRAL),
    }
    for s in range(3):
        pa_none[f"Rand_Pirate_s{s}@d*_con"] = _pa_rows(PLAIN_NEUTRAL)
        pa_ban[f"Rand_Pirate_s{s}@d*_con"] = _pa_rows(PLAIN_NEUTRAL)
        if d_lex != d_con:
            pa_none[f"Rand_Pirate_s{s}@d*_lex"] = _pa_rows(PLAIN_NEUTRAL)
            pa_ban[f"Rand_Pirate_s{s}@d*_lex"] = _pa_rows(PLAIN_NEUTRAL)

    def _pb_rows(text: str, n_tok: int = 30) -> list[dict[str, Any]]:
        return [
            {
                "id": p["id"],
                "text": text,
                "n_tokens": n_tok,
                "construct_types": c3_scoring.construct_types(text),
                "has_markdown": c3_scoring.has_markdown(text),
                "markdown_hits": c3_scoring.markdown_hits(text),
                "is_broken": False,
            }
            for p in bat["nomd_24"]
        ]

    pb: dict[str, list[dict[str, Any]]] = {
        "Plain": _pb_rows(MD_HEAVY, 30),
        "Sys_NoMD": _pb_rows(PROSE_CLEAN, 30),
        "WSP_NoMD_Contrastive@d*_nomd_con": _pb_rows(PROSE_CLEAN, 30),
        "WSP_NoMD_Lexicon@d*_nomd_lex": _pb_rows(PROSE_CLEAN, 30),
    }
    for s in range(3):
        pb[f"Rand_NoMD_s{s}@d*_nomd_con"] = _pb_rows(MD_HEAVY, 30)
        if d_nomd_lex != d_nomd_con:
            pb[f"Rand_NoMD_s{s}@d*_nomd_lex"] = _pb_rows(MD_HEAVY, 30)

    d5: dict[str, list[dict[str, Any]]] = {}
    for arm in C3_CONSTANTS["multiturn_d5_arms_5"]:
        txt = PIRATE_SUBTLE if arm in ("Sys_Pirate", "WSP_Pirate_Contrastive@d*_con") else PLAIN_NEUTRAL
        d5[arm] = [
            {
                "id": mt["id"],
                "turns": [
                    {"turn": 1, "text": txt, "n_tokens": 20, "is_broken": False},
                    {"turn": 2, "text": txt, "n_tokens": 20, "is_broken": False},
                    {"turn": 3, "text": txt, "n_tokens": 20, "is_broken": False},
                ],
            }
            for mt in bat["multiturn_12"]
        ]

    demo_ids = list(bat["demo_showcase_ids_12"])
    d6 = {
        "demo_ids": demo_ids,
        "own_top50": {},
        "conditions": {
            "Plain@common": [{"id": pid, "text": PLAIN_NEUTRAL, "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for pid in demo_ids],
            "Con@own50+common": [{"id": pid, "text": PIRATE_SUBTLE, "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for pid in demo_ids],
            "Con@donor_rand0+common": [{"id": pid, "text": PIRATE_SUBTLE, "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for pid in demo_ids],
            "Lex@own50+common": [{"id": pid, "text": PLAIN_NEUTRAL, "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for pid in demo_ids],
            "Rand0@own50+common": [{"id": pid, "text": PLAIN_NEUTRAL, "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for pid in demo_ids],
        },
    }

    d1_arms = {}
    for arm in ("Plain", "Sys_Pirate", "WSP_Pirate_Contrastive@d*_con", "WSP_Pirate_Lexicon@d*_lex", "Rand_Pirate_s0@d*_con"):
        bonus = 0.8 if "Contrastive" in arm or "Sys_Pirate" in arm else 0.0
        d1_arms[arm] = [
            {"id": cp["id"], "lp_self": -3.0 + bonus, "lp_stranger": -3.0, "self_minus_stranger": bonus}
            for cp in bat["copula_pairs_12"]
        ]

    def _cap_block(math_ok: bool = True, code_ok: bool = True, kl_v: float = 0.10) -> dict[str, Any]:
        return {
            "math_20": [
                {"id": p["id"], "text": f"The answer is FINAL: {p['answer'] if math_ok else p['answer'] + 999}", "n_tokens": 10, "is_broken": False}
                for p in bat["math_20"]
            ],
            "code_12": [
                {"id": p["id"], "text": f"```python\ndef {p['func_name']}(*a): return None\n```", "n_tokens": 15, "is_broken": False, "code_pass_precomputed": code_ok}
                for p in bat["code_12"]
            ],
            "refusal_8": [
                {"id": p["id"], "text": "I cannot assist with illegal or harmful instructions.", "n_tokens": 12, "is_broken": False}
                for p in bat["refusal_8"]
            ],
            "neutral_chat_16": [
                {"id": p["id"], "text": PLAIN_NEUTRAL, "n_tokens": 18, "is_broken": False, "off_kl": kl_v, "max_single_pos_kl": kl_v * 2, "greedy_exact_tok_match_rate": 0.90}
                for p in bat["neutral_chat_16"]
            ],
        }

    cap_arms = list(C3_CONSTANTS["capability_suite_arms_9"]) + list(C3_CONSTANTS["capability_suite_conditional_d_common_2"])
    cap_suite = {a: _cap_block(math_ok=True, code_ok=True, kl_v=0.0 if a == "Plain" else 0.12) for a in cap_arms}

    rand_res: dict[str, Any] = {}
    for s in range(3):
        rand_res[f"Rand_Pirate_s{s}@d*_con"] = {
            "paired_edit": "WSP_Pirate_Contrastive@d*_con",
            "dose": d_con,
            "bf16_retention": _ret(0.95, 0.96),
            "intended_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "stored_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "median_off_kl": 0.11,
            "max_prompt_off_kl": 0.18,
            "per_prompt_kl": [{"id": p["id"], "off_kl": 0.11, "max_single_pos_kl": 0.8} for p in bat["reserve_8"]],
        }
        if d_lex != d_con:
            rand_res[f"Rand_Pirate_s{s}@d*_lex"] = {
                "paired_edit": "WSP_Pirate_Lexicon@d*_lex",
                "dose": d_lex,
                "bf16_retention": _ret(0.95, 0.96),
                "intended_frob_ratio_vs_edit": {L: 1.0 for L in layers},
                "stored_frob_ratio_vs_edit": {L: 1.0 for L in layers},
                "median_off_kl": 0.11,
                "max_prompt_off_kl": 0.18,
                "per_prompt_kl": [{"id": p["id"], "off_kl": 0.11, "max_single_pos_kl": 0.8} for p in bat["reserve_8"]],
            }
        rand_res[f"Rand_NoMD_s{s}@d*_nomd_con"] = {
            "paired_edit": "WSP_NoMD_Contrastive@d*_nomd_con",
            "dose": d_nomd_con,
            "bf16_retention": _ret(0.95, 0.96),
            "intended_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "stored_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "median_off_kl": 0.11,
            "max_prompt_off_kl": 0.18,
            "per_prompt_kl": [{"id": p["id"], "off_kl": 0.11, "max_single_pos_kl": 0.8} for p in bat["reserve_8"]],
        }
        if d_nomd_lex != d_nomd_con:
            rand_res[f"Rand_NoMD_s{s}@d*_nomd_lex"] = {
                "paired_edit": "WSP_NoMD_Lexicon@d*_nomd_lex",
                "dose": d_nomd_lex,
                "bf16_retention": _ret(0.95, 0.96),
                "intended_frob_ratio_vs_edit": {L: 1.0 for L in layers},
                "stored_frob_ratio_vs_edit": {L: 1.0 for L in layers},
                "median_off_kl": 0.11,
                "max_prompt_off_kl": 0.18,
                "per_prompt_kl": [{"id": p["id"], "off_kl": 0.11, "max_single_pos_kl": 0.8} for p in bat["reserve_8"]],
            }

    dq = {
        a: {L: {"cos_ret": 0.999, "eff_unmasked": 0.5, "eff_masked": 1.0, "eff_ratio": 2.0, "post_norm_of_v_pre_hat": 0.2, "rms_gamma": 0.2, "delivered_match_ratio": 1.0} for L in layers}
        for a in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon", "WSP_NoMD_Contrastive", "WSP_NoMD_Lexicon")
    }

    return {
        "schema_version": scorer.EXPECTED_RECEIPT_SCHEMA,
        "meta": {
            "model": C3_CONSTANTS["model"]["id"],
            "revision": C3_CONSTANTS["model"]["revision"],
            "smoke": False,
            "device": "synthetic-test",
            "elapsed_sec": 12.3,
            "target_layers": [21, 24, 27],
            "requested_gen_batch_size": 8,
            "effective_gen_batch_size": 8,
            "b8_parity_max_abs_dlogit": 0.004,
            "b8_parity_passed": True,
            "active_hooks_max_seen": 0,
            "restore_exact_all": True,
            "restore_checks": 25,
            "weight_checkpoints_saved": 0,
            "counts": {"generations": 1000, "generated_tokens": 50000, "teacher_forced_b1_fwds": 600, "preflight_fwds": 64},
            "provenance_hashes": dict(DISK_HASHES),
        },
        "preflights": {
            "sham": {
                "alpha": 0.05,
                "max_abs_dlogit": 0.04,
                "min_stored_rank1_proj_ret": 0.35,
                "per_layer": _ret(0.35, 0.68),
            },
            "D4_L24_mask_mismask": {
                "layer": 24,
                "alpha_L24": 0.4,
                "masked_over_unmasked_logit_l2_ratio": 1.44,
                "mismasked_over_unmasked_logit_l2_ratio": 1.00,
                "masked_over_unmasked_seed_dlogit_ratio": 1.45,
                "mismasked_over_unmasked_seed_dlogit_ratio": 1.00,
            },
            "surface_stem_audit_positive_check": {"positive_sample_hit": True, "negative_sample_hit": False},
            "active_arm_max_abs_dlogit": {
                f"WSP_Pirate_Contrastive@d={float(d_con)}": 0.85,
                f"WSP_Pirate_Lexicon@d={float(d_lex)}": 0.75,
                f"WSP_NoMD_Contrastive@d={float(d_nomd_con)}": 0.80,
                f"WSP_NoMD_Lexicon@d={float(d_nomd_lex)}": 0.70,
                **{rk: 0.40 for rk in rand_res.keys()},
            },
        },
        "calibration": {
            "n_prompts": 16,
            "layers": {L: {"n_tokens": 272, "c_bar": 2.1, "c_std": 0.6, "cv": 0.28, "max_delivery_ratio": 1.8, "pmn": 10.0, "rms_pre": 0.5} for L in layers},
            "direction_quality": dq,
        },
        "doses": {
            "d_star_pirate_con": d_con,
            "d_star_pirate_lex": d_lex,
            "d_common": d_common,
            "d_star_nomd_con": d_nomd_con,
            "d_star_nomd_lex": d_nomd_lex,
            "commit_gate_fallback": {a: False for a in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon", "WSP_NoMD_Contrastive", "WSP_NoMD_Lexicon")},
            "d_common_retention": {},
        },
        "commit_gate": {
            "plain_reserve": [{"id": p["id"], "text": PLAIN_NEUTRAL, "n_tokens": 12, "is_broken": False} for p in bat["reserve_8"]],
            "arms": {
                "WSP_Pirate_Contrastive": _commit_arm(d_con),
                "WSP_Pirate_Lexicon": _commit_arm(d_lex),
                "WSP_NoMD_Contrastive": _commit_arm(d_nomd_con),
                "WSP_NoMD_Lexicon": _commit_arm(d_nomd_lex),
            },
        },
        "rand_reserve": rand_res,
        "pillar_a": {"none": pa_none, "common_pirate_ban": pa_ban},
        "pillar_b": pb,
        "D5_multiturn": d5,
        "D6_adaptive_donor": d6,
        "D1_copula": {"pirate_seed_ids": [1, 2, 3], "pirate_seed_tokens": [" pirate"], "arms": d1_arms},
        "capability_suite": cap_suite,
        "D2_sibling_bleed": {
            "WSP_Pirate_Contrastive@d*_con_on_NOMD_24": _pb_rows(MD_HEAVY, 30),
            "WSP_NoMD_Contrastive@d*_nomd_con_on_IDENTITY_20": [
                {"id": p["id"], "text": PLAIN_NEUTRAL, "n_tokens": 20, "is_broken": False, "surface_stem_hit": False}
                for p in bat["identity_20"]
            ],
        },
    }


def make_mock_reader_fn() -> tuple[Any, list[tuple[str, str, str]]]:
    """Deterministic mock reader verifying strict blinding (captures every prompt+answer received)."""
    seen_calls: list[tuple[str, str, str]] = []
    gold_truth = {g["answer"]: bool(g["expected_pirate"]) for g in C3_CONSTANTS["reader_gold_14"]}

    def _mock(reader: str, prompt: str, answer: str):
        seen_calls.append((reader, prompt, answer))
        if answer in gold_truth:
            return {"pirate_persona": gold_truth[answer], "coherent": not c3_scoring.is_broken(answer)}, None
        is_p = "articles signed in ink" in answer
        return {"pirate_persona": is_p, "coherent": not c3_scoring.is_broken(answer)}, None

    return _mock, seen_calls


def test_1_passing_roundtrip_blinding_and_byte_determinism() -> None:
    receipt = make_synthetic_passing_receipt()
    mock_fn, seen = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        rlabels = auditor.run_blind_audit(
            receipt, C3_CONSTANTS, Path(td),
            receipt_sha256="abc123",
            constants_sha256=DISK_HASHES["c3_constants_sha256"],
            scoring_sha256=DISK_HASHES["c3_scoring_sha256"],
            reader_fn=mock_fn,
            mock_readers=True,
        )
        assert rlabels["gold_14"]["summary"]["pass"] is True
        # Verify ZERO arm/ban metadata leaked into reader prompts!
        for _, p_txt, a_txt in seen:
            assert "WSP_Pirate" not in p_txt and "Rand_Pirate" not in p_txt and "common_pirate_ban" not in p_txt
            assert "WSP_Pirate" not in a_txt and "Rand_Pirate" not in a_txt

    sc1 = scorer.score_stage_c3(
        receipt, C3_CONSTANTS,
        reader_labels=rlabels,
        disk_hashes=DISK_HASHES,
        receipt_sha256="abc123",
        strict_provenance=True,
        allow_mock_readers=True,
        n_perm=2000,
        n_boot=500,
        run_code_subprocess=False,
    )
    sc2 = scorer.score_stage_c3(
        receipt, C3_CONSTANTS,
        reader_labels=rlabels,
        disk_hashes=DISK_HASHES,
        receipt_sha256="abc123",
        strict_provenance=True,
        allow_mock_readers=True,
        n_perm=2000,
        n_boot=500,
        run_code_subprocess=False,
    )
    assert sc1["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "PASS", "H3": "PASS", "H4": "PASS"}
    j1 = json.dumps(sc1, indent=2, sort_keys=True)
    j2 = json.dumps(sc2, indent=2, sort_keys=True)
    assert j1 == j2, "JSON scorecard not byte-identical across runs!"
    m1 = scorer.render_scorecard_markdown(sc1)
    m2 = scorer.render_scorecard_markdown(sc2)
    assert m1 == m2, "Markdown scorecard not byte-identical across runs!"
    # Production refusal of mock readers
    sc_refuse_mock = scorer.score_stage_c3(
        receipt, C3_CONSTANTS,
        reader_labels=rlabels,
        disk_hashes=DISK_HASHES,
        receipt_sha256="abc123",
        strict_provenance=True,
        allow_mock_readers=False,
        n_perm=500,
        n_boot=200,
        run_code_subprocess=False,
    )
    assert sc_refuse_mock["gates"]["C3.R"]["reason"] == "mock_readers_refused"
    assert sc_refuse_mock["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"
    assert sc_refuse_mock["confirmatory_verdicts"]["H3"] == "PASS"
    print("PASS test_1_passing_roundtrip_blinding_and_byte_determinism")


def test_2_all_9_verdict_states_and_gate_isolation() -> None:
    base = make_synthetic_passing_receipt()
    mock_fn, _ = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        base_labels = auditor.run_blind_audit(base, C3_CONSTANTS, Path(td), reader_fn=mock_fn, mock_readers=True)

    def _score(rec: dict[str, Any], rl: dict[str, Any] | None = base_labels) -> dict[str, Any]:
        return scorer.score_stage_c3(
            rec, C3_CONSTANTS, reader_labels=rl, disk_hashes=DISK_HASHES,
            strict_provenance=True, allow_mock_readers=True, n_perm=1000, n_boot=300, run_code_subprocess=False,
        )

    # State 1: BLOCKED_BY_C3.0 (test hooks > 0, D4 fail, NaN, stem-hit > 0.10, bad provenance)
    for label, mut in (
        ("hooks", lambda r: r["meta"].__setitem__("active_hooks_max_seen", 1)),
        ("d4_mask", lambda r: r["preflights"]["D4_L24_mask_mismask"].__setitem__("masked_over_unmasked_logit_l2_ratio", 1.10)),
        ("d4_mismask", lambda r: r["preflights"]["D4_L24_mask_mismask"].__setitem__("mismasked_over_unmasked_logit_l2_ratio", 1.08)),
        ("cv", lambda r: r["calibration"]["layers"]["24"].__setitem__("cv", 1.05)),
        ("rand_kl", lambda r: r["rand_reserve"]["Rand_Pirate_s0@d*_con"].__setitem__("median_off_kl", 0.40)),
        ("nan", lambda r: r["calibration"]["layers"]["21"].__setitem__("pmn", float("nan"))),
        ("prov", lambda r: r["meta"]["provenance_hashes"].__setitem__("c3_bank_sha256", "0" * 64)),
    ):
        r = copy.deepcopy(base)
        mut(r)
        sc = _score(r)
        assert sc["confirmatory_verdicts"] == {"C3.0": "FAIL", "H1": "BLOCKED_BY_C3.0", "H2": "BLOCKED_BY_C3.0", "H3": "BLOCKED_BY_C3.0", "H4": "BLOCKED_BY_C3.0"}, f"Failed C3.0 block on {label}"

    # C3.0.2 banned surface stem hit > 0.10 (> 4/40)
    r_stem = copy.deepcopy(base)
    for i in range(5):
        r_stem["pillar_a"]["common_pirate_ban"]["WSP_Pirate_Contrastive@d*_con"][i]["text"] = "Ahoy matey pirate captain!"
    sc_stem = _score(r_stem)
    assert sc_stem["confirmatory_verdicts"]["C3.0"] == "FAIL" and sc_stem["confirmatory_verdicts"]["H1"] == "BLOCKED_BY_C3.0"

    # State 2: DIRECTION_INVALID (scoped ONLY to affected arm!)
    r_dir = copy.deepcopy(base)
    r_dir["calibration"]["direction_quality"]["WSP_Pirate_Lexicon"]["24"]["cos_ret"] = 0.96
    sc_dir = _score(r_dir)
    assert sc_dir["confirmatory_verdicts"]["H1"] == "PASS"
    assert sc_dir["confirmatory_verdicts"]["H2"] == "DIRECTION_INVALID"
    assert sc_dir["confirmatory_verdicts"]["H3"] == "PASS"
    assert sc_dir["confirmatory_verdicts"]["H4"] == "PASS"

    # State 3: DOSE_INVALID (bf16 unrepresentable proj_ret < 0.50 on selected rung OR fallback=True)
    r_dose = copy.deepcopy(base)
    for rr in r_dose["commit_gate"]["arms"]["WSP_NoMD_Contrastive"]["rungs"]:
        rr["bf16_retention"]["21"]["stored_rank1_proj_ret"] = 0.42
    sc_dose = _score(r_dose)
    assert sc_dose["confirmatory_verdicts"]["H3"] == "DOSE_INVALID"
    assert sc_dose["confirmatory_verdicts"]["H1"] == "PASS"
    assert sc_dose["confirmatory_verdicts"]["H4"] == "PASS"

    # State 4: BLOCKED_BY_NOMD_ROOM (Pillar B ONLY; Pillar A unaffected!)
    r_room = copy.deepcopy(base)
    for row in r_room["pillar_b"]["Plain"]:
        row["text"] = "Just plain text without any markdown symbols at all."
    sc_room = _score(r_room)
    assert sc_room["confirmatory_verdicts"]["H1"] == "PASS"
    assert sc_room["confirmatory_verdicts"]["H2"] == "PASS"
    assert sc_room["confirmatory_verdicts"]["H3"] == "BLOCKED_BY_NOMD_ROOM"
    assert sc_room["confirmatory_verdicts"]["H4"] == "BLOCKED_BY_NOMD_ROOM"

    # State 5: NOT_READ_LOW_KAPPA (omitted reader labels OR failed gold 1p-neg; Pillar B unaffected!)
    sc_nolab = _score(base, rl=None)
    assert sc_nolab["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"
    assert sc_nolab["confirmatory_verdicts"]["H2"] == "NOT_READ_LOW_KAPPA"
    assert sc_nolab["confirmatory_verdicts"]["H3"] == "PASS"
    assert sc_nolab["confirmatory_verdicts"]["H4"] == "PASS"

    rl_bad_gold = copy.deepcopy(base_labels)
    rl_bad_gold["gold_14"]["raw"]["ds"]["gold_07"]["pirate_persona"] = True  # breaks 2/2 1p_non_pirate_neg!
    sc_bad_gold = _score(base, rl=rl_bad_gold)
    assert sc_bad_gold["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"
    assert sc_bad_gold["confirmatory_verdicts"]["H3"] == "PASS"

    # State 6: CAPABILITY_FAIL + F2 (Lex@d_common logged-only does NOT fail H2!)
    r_cap = copy.deepcopy(base)
    for mrow in r_cap["capability_suite"]["WSP_Pirate_Lexicon@d_common"]["math_20"]:
        mrow["text"] = "FINAL: -99999"
    sc_f2 = _score(r_cap)
    assert sc_f2["gates"]["EQ_C3_2"]["WSP_Pirate_Lexicon@d_common"]["pass"] is False
    assert sc_f2["gates"]["EQ_C3_2"]["WSP_Pirate_Lexicon@d_common"]["gating"] is False
    assert sc_f2["confirmatory_verdicts"]["H2"] == "PASS", "F2 VIOLATION: Lex@d_common must be logged-only!"

    for mrow in r_cap["capability_suite"]["WSP_Pirate_Contrastive@d_common"]["math_20"]:
        mrow["text"] = "FINAL: -99999"
    sc_cap_fail = _score(r_cap)
    assert sc_cap_fail["confirmatory_verdicts"]["H2"] == "CAPABILITY_FAIL"

    # State 8: EFFECT_BELOW_BAR (significant Holm, co-primary bars pass, primary effect < bar)
    # On H4 (bar = 0.35), make Plain/Rand have 1 construct type (heading) vs Lex 0 on 7/24 items -> mean_u = 7/24 = 0.2917 < 0.35,
    # delta_has_md = 7/24 = 0.2917 >= 0.15 & >= 0.10, p_perm ~ 2^-7 = 0.0078 <= 0.0125!
    # Wait: Plain needs mean_ct >= 1.50 for NoMD_room, so give Plain 2 constructs on all 24, Rand 1 construct on all 24,
    # and Lex 0 on 8/24 items & 1 on 16/24 -> mean_u = 8/24 = 0.3333 (< 0.35!), delta_has_md_vs_plain = 8/24 = 0.333 >= 0.15,
    # delta_has_md_vs_rand = 8/24 = 0.333 >= 0.10, p_perm = 2^-8 = 0.0039 <= 0.0125 -> EFFECT_BELOW_BAR!
    r_ebb = copy.deepcopy(base)
    one_md = "**Bold title** plain sentence here."
    for s in range(3):
        for row in r_ebb["pillar_b"][f"Rand_NoMD_s{s}@d*_nomd_con"]:
            row["text"] = one_md
    for i, row in enumerate(r_ebb["pillar_b"]["WSP_NoMD_Lexicon@d*_nomd_lex"]):
        row["text"] = "Clean plain prose without formatting." if i < 8 else one_md
    sc_ebb = _score(r_ebb)
    assert sc_ebb["confirmatory_verdicts"]["H4"] == "EFFECT_BELOW_BAR", f"Expected EFFECT_BELOW_BAR, got {sc_ebb['holm_4_family']['hypotheses']['H4']}"

    # State 9: FAIL via M1 length-ratio outside [0.85, 1.15] (MUST be FAIL, NEVER EFFECT_BELOW_BAR!)
    r_len = copy.deepcopy(r_ebb)
    for row in r_len["pillar_b"]["WSP_NoMD_Lexicon@d*_nomd_lex"]:
        row["n_tokens"] = 10  # 10 / 30 = 0.333 outside [0.85, 1.15] -> FAIL!
    sc_len = _score(r_len)
    assert sc_len["confirmatory_verdicts"]["H4"] == "FAIL", f"M1 VIOLATION: length-ratio breach must be FAIL, got {sc_len['confirmatory_verdicts']['H4']}"

    print("PASS test_2_all_9_verdict_states_and_gate_isolation")


def test_3_hard_refusals_and_real_code_pass() -> None:
    base = make_synthetic_passing_receipt()
    # Hard refusal of partial_checkpoint
    r_part = copy.deepcopy(base)
    r_part["partial_checkpoint"] = True
    try:
        scorer.score_stage_c3(r_part, C3_CONSTANTS, strict_provenance=False)
        raise AssertionError("Expected partial_checkpoint refusal")
    except ValueError as exc:
        assert "partial_checkpoint" in str(exc)

    # Hard refusal of smoke=True
    r_smk = copy.deepcopy(base)
    r_smk["meta"]["smoke"] = True
    try:
        scorer.score_stage_c3(r_smk, C3_CONSTANTS, strict_provenance=False)
        raise AssertionError("Expected smoke refusal")
    except ValueError as exc:
        assert "smoke" in str(exc).lower()

    # Real subprocess code_pass execution via evaluate_eq_c3_2
    r_code = copy.deepcopy(base)
    # Put real passing solution for code_00 (`sum_even`) on Plain & Con@d*_con
    good_code = "```python\ndef sum_even(nums):\n    return sum(x for x in nums if x % 2 == 0)\n```"
    r_code["capability_suite"]["Plain"]["code_12"][0]["text"] = good_code
    r_code["capability_suite"]["WSP_Pirate_Contrastive@d*_con"]["code_12"][0]["text"] = good_code
    eq = scorer.evaluate_eq_c3_2(r_code, C3_CONSTANTS, run_code_subprocess=True)
    assert abs(eq["Plain"]["code_pass_rate"] - (1.0 / 12.0)) < 1e-5
    assert eq["WSP_Pirate_Contrastive@d*_con"]["code_ok"] is True
    print("PASS test_3_hard_refusals_and_real_code_pass")


def test_4_distinct_dstar_f7_flag_and_d_common_exclusion_from_kappa_pool() -> None:
    receipt = make_synthetic_passing_receipt(d_con=0.45, d_lex=0.15, d_nomd_con=0.30, d_nomd_lex=0.075)
    mock_fn, _ = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        rl = auditor.run_blind_audit(receipt, C3_CONSTANTS, Path(td), reader_fn=mock_fn, mock_readers=True)
    sc = scorer.score_stage_c3(
        receipt, C3_CONSTANTS, reader_labels=rl, disk_hashes=DISK_HASHES,
        strict_provenance=True, allow_mock_readers=True, n_perm=1000, n_boot=300, run_code_subprocess=False,
    )
    assert sc["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "PASS", "H3": "PASS", "H4": "PASS"}
    # Fixed reader_kappa_pool denominator MUST remain 560 + 60 = 620 even when d*_con != d*_lex!
    assert sc["gates"]["C3.R"]["pool"]["expected_n_pooled"] == 620
    assert sc["gates"]["C3.R"]["pool"]["actual_n_pooled"] == 620
    # F7 interpretation flags must detect d_common (0.15) > 1 rung below d*_con (0.45) and bottom rung 0.075!
    f7 = sc["diagnostics"]["D7_operating_point_and_retention"]["f7_interpretation_flags"]
    assert f7["d_common_gt_1_rung_below_d_star_con"] is True
    assert f7["bottom_rung_0.075_selected_arms"] == ["WSP_NoMD_Lexicon"]
    print("PASS test_4_distinct_dstar_f7_flag_and_d_common_exclusion_from_kappa_pool")


def main() -> None:
    test_1_passing_roundtrip_blinding_and_byte_determinism()
    test_2_all_9_verdict_states_and_gate_isolation()
    test_3_hard_refusals_and_real_code_pass()
    test_4_distinct_dstar_f7_flag_and_d_common_exclusion_from_kappa_pool()
    print("ALL 4 PURE-STDLIB STAGE C3 SCORER & BLIND-AUDITOR TESTS PASSED.")


if __name__ == "__main__":
    main()
