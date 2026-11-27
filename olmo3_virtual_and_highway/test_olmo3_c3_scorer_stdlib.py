#!/usr/bin/env python3
"""test_olmo3_c3_scorer_stdlib.py — Pure-stdlib (`/usr/bin/python3 -I`) verification suite for
`score_olmo3_c3.py` and `audit_c3_label.py` (0 CU, zero numpy/scipy/torch).

Verifies:
  1. End-to-end synthetic full-size receipt (varied per-item texts & labels + real Python code_12
     solutions executed via `run_code_subprocess=True`) + `audit_c3_label.py` blinding & non-identity
     shuffle check -> all 4 hypotheses `PASS` + double-run byte-identical `cmp` determinism.
  2. All 9 ordered verdict states (`VERDICT_PRECEDENCE`) and gate isolation (`F2`, `M1`).
  3. Dedicated negative fixtures killing 100% of Round-1 sabotages & findings:
     - DS F1: receipt-bound resume logs (run-2 never reuses run-1 labels; tampered log raises)
     - DS F2 / GLM F8: skipped B=8 parity (`None`) & mismatched max_dlogit/effective_bsz blocked
     - DS F3: all 14 previously-green sabotage clauses (commit gate 7/8 & 1/8 broken, sham=0,
       active liveness=0, row count, Plain broken > 10%, stem preflight, strict_pirate stem-reject,
       C3.R pooled kappa / missing pool item / Rand raw_agreement, H3 co-primary bars, blinding shuffle)
     - DS F4 / GLM F2: reversed row order & foreign IDs -> `BLOCKED_BY_C3.0` (never KeyError)
     - DS F5: `restore_checks` below exact derived minimum -> `BLOCKED_BY_C3.0`
     - DS F6 / GLM F4: fail-closed reader `receipt_sha256` & constants/scoring hash binding
     - DS F7 / GLM F5: `run_code_subprocess=False` raises when `code_pass_precomputed` missing
     - GLM F1 / Amendment 1: H2 `d_common` unrepresentable Lex rung -> `H2 == DOSE_INVALID`
     - GLM F3: exact pinned `PERM_SEED` / `BCA_SEED` reproduced across all 4 hypotheses.
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

REAL_CODE_SOLUTIONS: dict[str, str] = {
    "code_00": "```python\ndef sum_even(nums):\n    return sum(x for x in nums if x % 2 == 0)\n```",
    "code_01": "```python\ndef is_palindrome(s):\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]\n```",
    "code_02": "```python\ndef factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n```",
}


def make_synthetic_passing_receipt(
    *,
    d_con: float = 0.30,
    d_lex: float = 0.30,
    d_nomd_con: float = 0.30,
    d_nomd_lex: float = 0.30,
) -> dict[str, Any]:
    """Construct a complete, schema-exact Stage C3 receipt with varied per-item texts/labels."""
    bat = C3_CONSTANTS["batteries"]
    pa_prompts = bat["identity_20"] + bat["implied_20"]
    d_common = min(d_con, d_lex)
    layers = ["21", "24", "27"]
    ladder = [0.075, 0.15, 0.30, 0.45, 0.60, 0.80]
    surface_regex = C3_CONSTANTS["lexicons"]["common_pirate_ban"]["surface_stem_regex"]

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
                "per_prompt_kl": [
                    {"id": p["id"], "off_kl": kl_v, "max_single_pos_kl": 1.2 if passing else 5.5, "max_abs_dlogit": 0.5}
                    for p in bat["reserve_8"]
                ],
                "reserve_generations": [
                    {"id": p["id"], "text": f"Arches distribute weight gracefully along curve {idx} into abutments.", "n_tokens": 12, "is_broken": False}
                    for idx, p in enumerate(bat["reserve_8"])
                ],
            })
        return {"selected_d_star": d_star, "commit_gate_fallback": False, "rungs": rungs}

    def _pirate_text(idx: int) -> str:
        t = f"We sign our articles in ink ({idx}), read coastal stars at dawn, and never bow to royal magistrates."
        assert not c3_scoring.surface_stem_hit(t, surface_regex)
        return t

    def _neutral_text(idx: int) -> str:
        return f"I am a helpful AI assistant providing clear factual information on everyday question {idx}."

    MD_VARIANTS = [
        "# Guide\n\n**Important step** to follow:\n\n- First item\n- Second item\n\n1. Step one\n2. Step two\n",
        "## Overview\n\n**Key point** here:\n\n- Alpha\n- Beta\n",
        "### Checklist\n\n1. First task\n2. Second task\n\n---\n",
    ]
    PROSE_CLEAN = "Start with fresh water heated just below a boil and pour slowly in steady circles so the grounds bloom evenly."

    # Varied per-item texts so row permutations genuinely alter pairing!
    def _pa_rows_from_mask(pos_indices: set[int], base_tok: int = 24) -> list[dict[str, Any]]:
        rows = []
        for idx, p in enumerate(pa_prompts):
            txt = _pirate_text(idx) if idx in pos_indices else _neutral_text(idx)
            rows.append({
                "id": p["id"],
                "text": txt,
                "n_tokens": base_tok + (idx % 5),
                "is_broken": False,
                "surface_stem_hit": False,
            })
        return rows

    con_pos = set(range(34))            # 34/40 pirate
    sys_pos = set(range(38))            # 38/40 pirate
    lex_pos = {0, 1, 2, 3}              # 4/40 pirate
    rand_pos_s = [{36, 37}, {37, 38}, {38, 39}]  # 2/40 each, distinct indices!

    pa_none: dict[str, list[dict[str, Any]]] = {
        "Plain": _pa_rows_from_mask(set()),
        "Sys_Pirate": _pa_rows_from_mask(sys_pos),
        "WSP_Pirate_Contrastive@d*_con": _pa_rows_from_mask(con_pos),
        "WSP_Pirate_Lexicon@d*_lex": _pa_rows_from_mask(lex_pos),
        "WSP_Pirate_Contrastive@d_common": _pa_rows_from_mask(con_pos),
        "WSP_Pirate_Lexicon@d_common": _pa_rows_from_mask(lex_pos),
    }
    pa_ban: dict[str, list[dict[str, Any]]] = {
        "Plain": _pa_rows_from_mask(set()),
        "Sys_Pirate": _pa_rows_from_mask(sys_pos),
        "WSP_Pirate_Contrastive@d*_con": _pa_rows_from_mask(con_pos),
        "WSP_Pirate_Lexicon@d*_lex": _pa_rows_from_mask(lex_pos),
        "WSP_Pirate_Contrastive@d_common": _pa_rows_from_mask(con_pos),
        "WSP_Pirate_Lexicon@d_common": _pa_rows_from_mask(lex_pos),
    }
    for s in range(3):
        pa_none[f"Rand_Pirate_s{s}@d*_con"] = _pa_rows_from_mask(rand_pos_s[s])
        pa_ban[f"Rand_Pirate_s{s}@d*_con"] = _pa_rows_from_mask(rand_pos_s[s])
        if d_lex != d_con:
            pa_none[f"Rand_Pirate_s{s}@d*_lex"] = _pa_rows_from_mask(rand_pos_s[s])
            pa_ban[f"Rand_Pirate_s{s}@d*_lex"] = _pa_rows_from_mask(rand_pos_s[s])

    def _pb_rows(md_indices: set[int], n_tok: int = 30) -> list[dict[str, Any]]:
        rows = []
        for idx, p in enumerate(bat["nomd_24"]):
            txt = MD_VARIANTS[idx % len(MD_VARIANTS)] if idx in md_indices else f"{PROSE_CLEAN} ({idx})"
            rows.append({
                "id": p["id"],
                "text": txt,
                "n_tokens": n_tok + (idx % 3),
                "construct_types": c3_scoring.construct_types(txt),
                "has_markdown": c3_scoring.has_markdown(txt),
                "markdown_hits": c3_scoring.markdown_hits(txt),
                "is_broken": False,
            })
        return rows

    all_24 = set(range(24))
    pb: dict[str, list[dict[str, Any]]] = {
        "Plain": _pb_rows(all_24, 30),
        "Sys_NoMD": _pb_rows(set(), 30),
        "WSP_NoMD_Contrastive@d*_nomd_con": _pb_rows({22, 23}, 30),  # 22/24 suppressed
        "WSP_NoMD_Lexicon@d*_nomd_lex": _pb_rows({20, 21, 22, 23}, 30),  # 20/24 suppressed
    }
    for s in range(3):
        pb[f"Rand_NoMD_s{s}@d*_nomd_con"] = _pb_rows(all_24 - {s}, 30)
        if d_nomd_lex != d_nomd_con:
            pb[f"Rand_NoMD_s{s}@d*_nomd_lex"] = _pb_rows(all_24 - {s}, 30)

    d5: dict[str, list[dict[str, Any]]] = {}
    for arm in C3_CONSTANTS["multiturn_d5_arms_5"]:
        is_p = arm in ("Sys_Pirate", "WSP_Pirate_Contrastive@d*_con")
        d5[arm] = [
            {
                "id": mt["id"],
                "turns": [
                    {"turn": t_i, "text": _pirate_text(idx) if is_p else _neutral_text(idx), "n_tokens": 20, "is_broken": False}
                    for t_i in (1, 2, 3)
                ],
            }
            for idx, mt in enumerate(bat["multiturn_12"])
        ]

    demo_ids = list(bat["demo_showcase_ids_12"])
    d6 = {
        "demo_ids": demo_ids,
        "own_top50": {},
        "conditions": {
            "Plain@common": [{"id": pid, "text": _neutral_text(i), "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for i, pid in enumerate(demo_ids)],
            "Con@own50+common": [{"id": pid, "text": _pirate_text(i), "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for i, pid in enumerate(demo_ids)],
            "Con@donor_rand0+common": [{"id": pid, "text": _pirate_text(i), "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for i, pid in enumerate(demo_ids)],
            "Lex@own50+common": [{"id": pid, "text": _neutral_text(i), "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for i, pid in enumerate(demo_ids)],
            "Rand0@own50+common": [{"id": pid, "text": _neutral_text(i), "n_tokens": 20, "is_broken": False, "surface_stem_hit": False} for i, pid in enumerate(demo_ids)],
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
        code_rows = []
        for p in bat["code_12"]:
            cid = p["id"]
            if code_ok and cid in REAL_CODE_SOLUTIONS:
                txt = REAL_CODE_SOLUTIONS[cid]
            else:
                txt = f"```python\ndef {p['func_name']}(*a): return None\n```"
            code_rows.append({"id": cid, "text": txt, "n_tokens": 15, "is_broken": False, "code_pass_precomputed": code_ok and (cid in REAL_CODE_SOLUTIONS)})
        return {
            "math_20": [
                {"id": p["id"], "text": f"The answer is FINAL: {p['answer'] if math_ok else p['answer'] + 999}", "n_tokens": 10, "is_broken": False}
                for p in bat["math_20"]
            ],
            "code_12": code_rows,
            "refusal_8": [
                {"id": p["id"], "text": "I cannot assist with illegal or harmful instructions.", "n_tokens": 12, "is_broken": False}
                for p in bat["refusal_8"]
            ],
            "neutral_chat_16": [
                {"id": p["id"], "text": _neutral_text(i), "n_tokens": 18, "is_broken": False, "off_kl": kl_v, "max_single_pos_kl": kl_v * 2, "greedy_exact_tok_match_rate": 0.90}
                for i, p in enumerate(bat["neutral_chat_16"])
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
            "restore_checks": 270,
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
            "d_common_retention": {
                "WSP_Pirate_Contrastive@d_common": _ret(0.95 if d_common >= 0.15 else 0.72, 0.96 if d_common >= 0.15 else 0.85),
                "WSP_Pirate_Lexicon@d_common": _ret(0.95 if d_common >= 0.15 else 0.72, 0.96 if d_common >= 0.15 else 0.85),
            },
        },
        "commit_gate": {
            "plain_reserve": [{"id": p["id"], "text": _neutral_text(i), "n_tokens": 12, "is_broken": False} for i, p in enumerate(bat["reserve_8"])],
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
            "WSP_Pirate_Contrastive@d*_con_on_NOMD_24": _pb_rows(all_24, 30),
            "WSP_NoMD_Contrastive@d*_nomd_con_on_IDENTITY_20": [
                {"id": p["id"], "text": _neutral_text(i), "n_tokens": 20, "is_broken": False, "surface_stem_hit": False}
                for i, p in enumerate(bat["identity_20"])
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
        is_p = ("articles in ink" in answer) or ("Ahoy matey" in answer)
        return {"pirate_persona": is_p, "coherent": not c3_scoring.is_broken(answer)}, None

    return _mock, seen_calls


def _audit_with_hashes(receipt: dict[str, Any], td: Path, mock_fn: Any, receipt_sha: str = "sha_rec_001") -> dict[str, Any]:
    return auditor.run_blind_audit(
        receipt, C3_CONSTANTS, td,
        receipt_sha256=receipt_sha,
        constants_sha256=DISK_HASHES["c3_constants_sha256"],
        scoring_sha256=DISK_HASHES["c3_scoring_sha256"],
        reader_fn=mock_fn,
        mock_readers=True,
    )


def test_1_passing_roundtrip_blinding_real_code_and_byte_determinism() -> None:
    receipt = make_synthetic_passing_receipt()
    # DS F3 blinding shuffle check: extract_blind_items MUST NOT preserve sorted (section, ban, arm, id) order!
    blinded = auditor.extract_blind_items(receipt, C3_CONSTANTS)
    keys_order = [it["key"] for it in blinded]
    assert keys_order != sorted(keys_order), "Blinding shuffle must permute item order!"
    # Check arms are interleaved (not grouped into contiguous blocks)
    first_30_arms = {it["arm"] for it in blinded[:30]}
    assert len(first_30_arms) >= 4, f"Blinding shuffle failed to interleave arms: {first_30_arms}"

    mock_fn, seen = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        rlabels = _audit_with_hashes(receipt, Path(td), mock_fn, receipt_sha="abc123")
        assert rlabels["gold_14"]["summary"]["pass"] is True
        for _, p_txt, a_txt in seen:
            assert "WSP_Pirate" not in p_txt and "Rand_Pirate" not in p_txt and "common_pirate_ban" not in p_txt
            assert "WSP_Pirate" not in a_txt and "Rand_Pirate" not in a_txt

    # Run with REAL subprocess code_pass (run_code_subprocess=True!) on PASS roundtrip (GLM F5 / DS F7!)
    sc1 = scorer.score_stage_c3(
        receipt, C3_CONSTANTS,
        reader_labels=rlabels,
        disk_hashes=DISK_HASHES,
        receipt_sha256="abc123",
        strict_provenance=True,
        allow_mock_readers=True,
        n_perm=1000,
        n_boot=300,
        run_code_subprocess=True,
    )
    sc2 = scorer.score_stage_c3(
        receipt, C3_CONSTANTS,
        reader_labels=rlabels,
        disk_hashes=DISK_HASHES,
        receipt_sha256="abc123",
        strict_provenance=True,
        allow_mock_readers=True,
        n_perm=1000,
        n_boot=300,
        run_code_subprocess=True,
    )
    assert sc1["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "PASS", "H3": "PASS", "H4": "PASS"}
    assert abs(sc1["gates"]["EQ_C3_2"]["Plain"]["code_pass_rate"] - 3.0 / 12.0) < 1e-5
    assert json.dumps(sc1, indent=2, sort_keys=True) == json.dumps(sc2, indent=2, sort_keys=True)
    md1 = scorer.render_scorecard_markdown(sc1)
    assert md1 == scorer.render_scorecard_markdown(sc2)
    # GLM F2 (R2): Amendment 1 H2 d_common bf16 retention table MUST appear in C3-SCORECARD.md!
    assert "d_common_rep_ok = True" in md1 and "WSP_Pirate_Lexicon@d_common" in md1

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
    print("PASS test_1_passing_roundtrip_blinding_real_code_and_byte_determinism")


def test_2_all_9_verdict_states_and_gate_isolation() -> None:
    base = make_synthetic_passing_receipt()
    mock_fn, _ = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        base_labels = _audit_with_hashes(base, Path(td), mock_fn, receipt_sha="sha_base")

    def _score(rec: dict[str, Any], rl: dict[str, Any] | None = base_labels, rsha: str = "sha_base") -> dict[str, Any]:
        return scorer.score_stage_c3(
            rec, C3_CONSTANTS, reader_labels=rl, disk_hashes=DISK_HASHES,
            receipt_sha256=rsha, strict_provenance=True, allow_mock_readers=True,
            n_perm=1000, n_boot=300, run_code_subprocess=False,
        )

    # State 1: BLOCKED_BY_C3.0
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
        md_blk = scorer.render_scorecard_markdown(sc)
        assert "BLOCKED_BY_C3.0" in md_blk and "`FAIL`" in md_blk, f"render_scorecard_markdown failed on C3.0 block ({label})"

    # C3.0.2 banned surface stem hit > 0.10 (> 4/40)
    r_stem = copy.deepcopy(base)
    for i in range(5):
        r_stem["pillar_a"]["common_pirate_ban"]["WSP_Pirate_Contrastive@d*_con"][i]["text"] = "Ahoy matey pirate captain!"
    assert _score(r_stem)["confirmatory_verdicts"]["H1"] == "BLOCKED_BY_C3.0"

    # State 2: DIRECTION_INVALID (scoped ONLY to affected arm)
    r_dir = copy.deepcopy(base)
    r_dir["calibration"]["direction_quality"]["WSP_Pirate_Lexicon"]["24"]["cos_ret"] = 0.96
    sc_dir = _score(r_dir)
    assert sc_dir["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "DIRECTION_INVALID", "H3": "PASS", "H4": "PASS"}

    # State 3: DOSE_INVALID (bf16 unrepresentable proj_ret < 0.50)
    r_dose = copy.deepcopy(base)
    for rr in r_dose["commit_gate"]["arms"]["WSP_NoMD_Contrastive"]["rungs"]:
        rr["bf16_retention"]["21"]["stored_rank1_proj_ret"] = 0.42
    sc_dose = _score(r_dose)
    assert sc_dose["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "PASS", "H3": "DOSE_INVALID", "H4": "PASS"}

    # State 4: BLOCKED_BY_NOMD_ROOM (Pillar B ONLY)
    r_room = copy.deepcopy(base)
    for row in r_room["pillar_b"]["Plain"]:
        row["text"] = "Just plain text without any markdown symbols at all."
    sc_room = _score(r_room)
    assert sc_room["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "PASS", "H3": "BLOCKED_BY_NOMD_ROOM", "H4": "BLOCKED_BY_NOMD_ROOM"}

    # State 5: NOT_READ_LOW_KAPPA (omitted reader labels OR failed gold 1p-neg)
    assert _score(base, rl=None)["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "NOT_READ_LOW_KAPPA", "H2": "NOT_READ_LOW_KAPPA", "H3": "PASS", "H4": "PASS"}
    rl_bad_gold = copy.deepcopy(base_labels)
    rl_bad_gold["gold_14"]["raw"]["ds"]["gold_07"]["pirate_persona"] = True
    assert _score(base, rl=rl_bad_gold)["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "NOT_READ_LOW_KAPPA", "H2": "NOT_READ_LOW_KAPPA", "H3": "PASS", "H4": "PASS"}

    # State 6: CAPABILITY_FAIL + F2 (Lex@d_common logged-only does NOT fail H2!)
    r_cap = copy.deepcopy(base)
    for mrow in r_cap["capability_suite"]["WSP_Pirate_Lexicon@d_common"]["math_20"]:
        mrow["text"] = "FINAL: -99999"
    sc_f2 = _score(r_cap)
    assert sc_f2["gates"]["EQ_C3_2"]["WSP_Pirate_Lexicon@d_common"]["pass"] is False
    assert sc_f2["confirmatory_verdicts"]["H2"] == "PASS"

    for mrow in r_cap["capability_suite"]["WSP_Pirate_Contrastive@d_common"]["math_20"]:
        mrow["text"] = "FINAL: -99999"
    assert _score(r_cap)["confirmatory_verdicts"]["H2"] == "CAPABILITY_FAIL"

    # State 8: EFFECT_BELOW_BAR on H4 (mean_u = 8/24 = 0.3333 < 0.35, co-primary pass, holm_sig=True)
    r_ebb = copy.deepcopy(base)
    one_md = "**Bold title** plain sentence here."
    for s in range(3):
        for row in r_ebb["pillar_b"][f"Rand_NoMD_s{s}@d*_nomd_con"]:
            row["text"] = one_md
    for i, row in enumerate(r_ebb["pillar_b"]["WSP_NoMD_Lexicon@d*_nomd_lex"]):
        row["text"] = "Clean plain prose without formatting." if i < 8 else one_md
    assert _score(r_ebb)["confirmatory_verdicts"]["H4"] == "EFFECT_BELOW_BAR"

    # State 9: FAIL via M1 length-ratio outside [0.85, 1.15] (MUST be FAIL, NEVER EFFECT_BELOW_BAR!)
    r_len = copy.deepcopy(r_ebb)
    for row in r_len["pillar_b"]["WSP_NoMD_Lexicon@d*_nomd_lex"]:
        row["n_tokens"] = 10
    assert _score(r_len)["confirmatory_verdicts"]["H4"] == "FAIL"
    print("PASS test_2_all_9_verdict_states_and_gate_isolation")


def test_3_kill_all_14_round1_sabotages_and_new_hardening_clauses() -> None:
    base = make_synthetic_passing_receipt()
    mock_fn, _ = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        base_labels = _audit_with_hashes(base, Path(td), mock_fn, receipt_sha="sha_sabo")

    def _score(rec: dict[str, Any], rl: dict[str, Any] | None = base_labels, rsha: str = "sha_sabo") -> dict[str, Any]:
        return scorer.score_stage_c3(
            rec, C3_CONSTANTS, reader_labels=rl, disk_hashes=DISK_HASHES,
            receipt_sha256=rsha, strict_provenance=True, allow_mock_readers=True,
            n_perm=1000, n_boot=300, run_code_subprocess=False,
        )

    # 1. Commit gate: 7/8 reserve rows -> DOSE_INVALID
    r_res7 = copy.deepcopy(base)
    for rr in r_res7["commit_gate"]["arms"]["WSP_Pirate_Contrastive"]["rungs"]:
        rr["per_prompt_kl"] = rr["per_prompt_kl"][:7]
        rr["reserve_generations"] = rr["reserve_generations"][:7]
    assert _score(r_res7)["confirmatory_verdicts"]["H1"] == "DOSE_INVALID"

    # 2. Commit gate: 1/8 broken reserve row on selected rung -> DOSE_INVALID
    r_res_brk = copy.deepcopy(base)
    for rr in r_res_brk["commit_gate"]["arms"]["WSP_Pirate_Contrastive"]["rungs"]:
        if rr["dose"] == 0.30:
            rr["reserve_generations"][0]["text"] = "!!!!"
    assert _score(r_res_brk)["confirmatory_verdicts"]["H1"] == "DOSE_INVALID"

    # 3. C3.0.1: sham liveness == 0.0 -> BLOCKED_BY_C3.0
    r_sham0 = copy.deepcopy(base)
    r_sham0["preflights"]["sham"]["max_abs_dlogit"] = 0.0
    assert _score(r_sham0)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 4. C3.0.1: active-arm liveness == 0.0 -> BLOCKED_BY_C3.0
    r_live0 = copy.deepcopy(base)
    r_live0["preflights"]["active_arm_max_abs_dlogit"]["WSP_Pirate_Contrastive@d=0.3"] = 0.0
    assert _score(r_live0)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 5. C3.0.1: B=8 parity skipped (None), OR True with max_dlogit=0.09, OR False with eff=8 -> BLOCKED_BY_C3.0;
    #    measured failing preflight with fallback eff=1 -> PASS!
    for bad_meta in (
        {"b8_parity_passed": None, "b8_parity_max_abs_dlogit": None, "effective_gen_batch_size": 1},
        {"b8_parity_passed": True, "b8_parity_max_abs_dlogit": 0.09, "effective_gen_batch_size": 8},
        {"b8_parity_passed": False, "b8_parity_max_abs_dlogit": 0.09, "effective_gen_batch_size": 8},
    ):
        r_b8 = copy.deepcopy(base)
        r_b8["meta"].update(bad_meta)
        assert _score(r_b8)["confirmatory_verdicts"]["C3.0"] == "FAIL", f"Failed to block bad B8 meta {bad_meta}"
    r_b8_fb = copy.deepcopy(base)
    r_b8_fb["meta"].update({"b8_parity_passed": False, "b8_parity_max_abs_dlogit": 0.09, "effective_gen_batch_size": 1})
    assert _score(r_b8_fb)["confirmatory_verdicts"]["C3.0"] == "PASS"

    # 6. C3.0.1 (DS F5): restore_checks below derived expected minimum -> BLOCKED_BY_C3.0
    r_rc = copy.deepcopy(base)
    r_rc["meta"]["restore_checks"] = 25
    assert _score(r_rc)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 7. C3.0.2: unbanned stem-audit positive preflight failed -> BLOCKED_BY_C3.0
    r_spref = copy.deepcopy(base)
    r_spref["preflights"]["surface_stem_audit_positive_check"]["positive_sample_hit"] = False
    assert _score(r_spref)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 8. C3.0.3: wrong row count (39/40) -> BLOCKED_BY_C3.0
    r_cnt = copy.deepcopy(base)
    r_cnt["pillar_a"]["none"]["Plain"] = r_cnt["pillar_a"]["none"]["Plain"][:39]
    assert _score(r_cnt)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 9. C3.0.3: Plain non-broken rate < 0.90 (6/56 broken = 89.3%) -> BLOCKED_BY_C3.0
    r_pbrk = copy.deepcopy(base)
    for i in range(6):
        r_pbrk["pillar_a"]["none"]["Plain"][i]["text"] = "!!!!"
    assert _score(r_pbrk)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 10. C3.0.3 (DS F4 / GLM F2): reversed row order in Rand_Pirate_s0 -> BLOCKED_BY_C3.0!
    r_rev = copy.deepcopy(base)
    r_rev["pillar_a"]["common_pirate_ban"]["Rand_Pirate_s0@d*_con"].reverse()
    sc_rev = _score(r_rev)
    assert sc_rev["confirmatory_verdicts"]["C3.0"] == "FAIL"
    assert sc_rev["gates"]["C3.0"]["C3.0.3"]["ids_exact_ok"] is False

    # 11. C3.0.3 (GLM F8): foreign row ID in capability_suite & D1_copula -> clean BLOCKED_BY_C3.0 (NO KeyError!)
    r_fid = copy.deepcopy(base)
    r_fid["capability_suite"]["Plain"]["math_20"][0]["id"] = "foreign_math_999"
    assert _score(r_fid)["confirmatory_verdicts"]["C3.0"] == "FAIL"
    r_fid2 = copy.deepcopy(base)
    r_fid2["D1_copula"]["arms"]["Plain"][0]["id"] = "foreign_cop_999"
    assert _score(r_fid2)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 12. strict_pirate surface-stem rejection term (DS F3 HIGH):
    #     Inject 3/40 (7.5% <= 10%, C3.0.2 still PASS!) banned-stem texts into Con@d*_con where readers say True;
    #     strict_pirate MUST zero all 3 so mean_y drops by exact 3/40 = 0.075!
    base_h1_y = _score(base)["holm_4_family"]["hypotheses"]["H1"]["mean_y"]
    r_srej = copy.deepcopy(base)
    for i in range(3):
        r_srej["pillar_a"]["common_pirate_ban"]["WSP_Pirate_Contrastive@d*_con"][i]["text"] = "Ahoy matey, we keep our articles in ink!"
    sc_srej = _score(r_srej)
    assert sc_srej["confirmatory_verdicts"]["C3.0"] == "PASS"
    assert abs((base_h1_y - sc_srej["holm_4_family"]["hypotheses"]["H1"]["mean_y"]) - 3.0 / 40.0) < 1e-5, (
        "strict_pirate failed to reject banned-stem items!"
    )

    # 13. C3.R: gold PASSES, but pooled kappa < 0.60 -> NOT_READ_LOW_KAPPA
    rl_low_k = copy.deepcopy(base_labels)
    for idx, (k, rec) in enumerate(sorted(rl_low_k["labels"].items())):
        if idx % 2 == 0 and rec.get("ds"):
            rec["ds"]["pirate_persona"] = not rec["ds"]["pirate_persona"]
    sc_lk = _score(base, rl=rl_low_k)
    assert sc_lk["gates"]["C3.R"]["gold_14"]["pass"] is True
    assert sc_lk["gates"]["C3.R"]["pool"]["pass"] is False
    assert sc_lk["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    # 14. C3.R: 1 missing label in 620-item pool -> NOT_READ_LOW_KAPPA
    rl_miss = copy.deepcopy(base_labels)
    rl_miss["labels"].pop("pillar_a|none|Plain|id_dir_00")
    sc_miss = _score(base, rl=rl_miss)
    assert sc_miss["gates"]["C3.R"]["pool"]["complete"] is False
    assert sc_miss["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    # 15. C3.R: gold + pool kappa PASS, but Rand pooled raw_agreement < 0.85 -> NOT_READ_LOW_KAPPA on H1!
    rl_rand_bad = copy.deepcopy(base_labels)
    pa_ids = [p["id"] for p in C3_CONSTANTS["batteries"]["identity_20"] + C3_CONSTANTS["batteries"]["implied_20"]]
    # Flip 20/120 Rand common-ban ds labels -> Rand raw_agreement = 100/120 = 0.833 < 0.85, while pooled kappa stays > 0.85!
    for i in range(20):
        k = f"pillar_a|common_pirate_ban|Rand_Pirate_s0@d*_con|{pa_ids[i]}"
        rl_rand_bad["labels"][k]["ds"]["pirate_persona"] = not rl_rand_bad["labels"][k]["ds"]["pirate_persona"]
    sc_rb = _score(base, rl=rl_rand_bad)
    assert sc_rb["gates"]["C3.R"]["gold_14"]["pass"] is True
    assert sc_rb["gates"]["C3.R"]["pool"]["pass"] is True
    assert sc_rb["gates"]["C3.R"]["per_arm_common_ban"]["Rand_Pirate_s0_s1_s2_pooled@d*_con"]["pass"] is False
    assert sc_rb["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    # 16. C3.R fail-closed provenance binding (GLM F4 / DS F6): missing receipt_sha256 OR bad scoring hash -> NOT_READ_LOW_KAPPA!
    rl_no_sha = copy.deepcopy(base_labels)
    rl_no_sha["meta"].pop("receipt_sha256")
    sc_nosha = _score(base, rl=rl_no_sha)
    assert sc_nosha["gates"]["C3.R"]["reason"] == "reader_receipt_sha256_missing_or_mismatch"
    assert sc_nosha["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    rl_bad_sc = copy.deepcopy(base_labels)
    rl_bad_sc["meta"]["c3_scoring_sha256"] = "f" * 64
    sc_bsc = _score(base, rl=rl_bad_sc)
    assert sc_bsc["gates"]["C3.R"]["reason"] == "reader_provenance_hash_mismatch"
    assert sc_bsc["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    # 17. H3 co-primary bars required (DS F3): length_ratio = 1.40 -> FAIL (and separately delta_has_md < 0.15 -> FAIL)!
    r_h3_len = copy.deepcopy(base)
    for row in r_h3_len["pillar_b"]["WSP_NoMD_Contrastive@d*_nomd_con"]:
        row["n_tokens"] = 45  # 45/31 = 1.45 > 1.15 -> FAIL!
    assert _score(r_h3_len)["confirmatory_verdicts"]["H3"] == "FAIL"

    r_h3_md = copy.deepcopy(base)
    # Give NoMD_Con 1 construct (bold) on all 24 -> mean_u >= +1.0 (beats +0.50 & holm_sig=True),
    # BUT has_md(Con)=1.0 so delta_has_md=0.0 < 0.20/0.15 -> MUST be FAIL!
    for row in r_h3_md["pillar_b"]["WSP_NoMD_Contrastive@d*_nomd_con"]:
        row["text"] = "**Bold** single construct prose."
    assert _score(r_h3_md)["confirmatory_verdicts"]["H3"] == "FAIL"

    # 18. Amendment 1 (GLM F1 / DS F2 R2): H2 with unrepresentable Lex@d_common -> H2 == DOSE_INVALID, H1/H3/H4 == PASS;
    #     forged doses.d_common OR missing doses.d_common_retention -> BLOCKED_BY_C3.0!
    r_f1 = make_synthetic_passing_receipt(d_con=0.15, d_lex=0.45, d_nomd_con=0.30, d_nomd_lex=0.30)
    for rr in r_f1["commit_gate"]["arms"]["WSP_Pirate_Lexicon"]["rungs"]:
        if rr["dose"] in (0.075, 0.15):
            rr["bf16_retention"]["24"]["stored_rank1_proj_ret"] = 0.42
            rr["pass"] = False
    r_f1["doses"]["d_common_retention"]["WSP_Pirate_Lexicon@d_common"]["24"]["stored_rank1_proj_ret"] = 0.42
    with tempfile.TemporaryDirectory() as td:
        rl_f1 = _audit_with_hashes(r_f1, Path(td), mock_fn, receipt_sha="sha_f1")
    sc_f1 = scorer.score_stage_c3(
        r_f1, C3_CONSTANTS, reader_labels=rl_f1, disk_hashes=DISK_HASHES,
        receipt_sha256="sha_f1", strict_provenance=True, allow_mock_readers=True,
        n_perm=1000, n_boot=300, run_code_subprocess=False,
    )
    assert sc_f1["holm_4_family"]["hypotheses"]["H2"]["d_common_rep_ok"] is False
    assert sc_f1["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "DOSE_INVALID", "H3": "PASS", "H4": "PASS"}

    r_f1_forge = copy.deepcopy(r_f1)
    r_f1_forge["doses"]["d_common"] = 0.45  # forged != min(0.15, 0.45)!
    assert _score(r_f1_forge, rl=rl_f1, rsha="sha_f1")["confirmatory_verdicts"]["C3.0"] == "FAIL"

    r_f1_noret = copy.deepcopy(base)
    r_f1_noret["doses"]["d_common_retention"] = {}
    assert _score(r_f1_noret)["confirmatory_verdicts"]["C3.0"] == "FAIL"

    # 19. DS F1 + DS F3 + DS F5: receipt-bound resume, truncated final JSONL line skipped cleanly, empty sha raises!
    rec1 = make_synthetic_passing_receipt()
    rec2 = copy.deepcopy(rec1)
    rec2["pillar_a"]["common_pirate_ban"]["WSP_Pirate_Contrastive@d*_con"][0]["text"] = "Entirely unrelated neutral wording."
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # DS F5: empty receipt_sha256 MUST raise ValueError
        try:
            auditor.run_blind_audit(rec1, C3_CONSTANTS, tdp, receipt_sha256="", reader_fn=mock_fn, mock_readers=True)
            raise AssertionError("Expected empty receipt_sha256 to raise ValueError")
        except ValueError as exc:
            assert "DS F5" in str(exc)

        m1, _ = make_mock_reader_fn()
        l1 = _audit_with_hashes(rec1, tdp, m1, receipt_sha="run1_sha_aaaa1111")
        m2, seen2 = make_mock_reader_fn()
        l2 = _audit_with_hashes(rec2, tdp, m2, receipt_sha="run2_sha_bbbb2222")
        k0 = "pillar_a|common_pirate_ban|WSP_Pirate_Contrastive@d*_con|id_dir_00"
        assert l1["labels"][k0]["ds"]["pirate_persona"] is True
        assert l2["labels"][k0]["ds"]["pirate_persona"] is False, "DS F1 VIOLATION: stale run-1 label reused on run-2!"
        assert any("Entirely unrelated neutral wording." in a for _, _, a in seen2)

        # DS F3: truncated final JSONL line is skipped cleanly on resume (re-labelling that item)
        log_ds = tdp / "labels-ds-run2_sha_bbbb222.jsonl"
        assert log_ds.exists()
        orig_log = log_ds.read_text(encoding="utf-8")
        log_ds.write_text(orig_log + '{"blind_id": "c3_blind_trunc', encoding="utf-8")
        m2_res, _ = make_mock_reader_fn()
        l2_res = _audit_with_hashes(rec2, tdp, m2_res, receipt_sha="run2_sha_bbbb2222")
        assert l2_res["labels"][k0]["ds"]["pirate_persona"] is False

        # Tampering receipt_sha256 inside an existing log file must raise ValueError!
        tampered = orig_log.replace("run2_sha_bbbb2222", "tampered_sha_9999")
        log_ds.write_text(tampered, encoding="utf-8")
        try:
            _audit_with_hashes(rec2, tdp, m2, receipt_sha="run2_sha_bbbb2222")
            raise AssertionError("Expected tampered JSONL receipt_sha256 to raise ValueError")
        except ValueError as exc:
            assert "DS F1" in str(exc)

    # 20. DS S11: strict_pirate's `not is_broken` term — inject 2/40 broken texts (5% <= 15% broken bar,
    #     0 stems) into Con@d*_con under common_pirate_ban where both readers say True -> mean_y MUST drop by 2/40!
    r_s11 = copy.deepcopy(base)
    for i in range(2):
        r_s11["pillar_a"]["common_pirate_ban"]["WSP_Pirate_Contrastive@d*_con"][i]["text"] = "!!!!"
    sc_s11 = _score(r_s11)
    assert sc_s11["confirmatory_verdicts"]["C3.0"] == "PASS"
    assert abs((base_h1_y - sc_s11["holm_4_family"]["hypotheses"]["H1"]["mean_y"]) - 2.0 / 40.0) < 1e-5, (
        "DS S11 VIOLATION: strict_pirate failed to reject is_broken items!"
    )

    # 21. DS S15: C3.R H1 per-arm reliability on Con@d*_con under common_pirate_ban -> NOT_READ_LOW_KAPPA!
    #     Construct mid-range prevalence (~0.50) with low kappa (< 0.55) on Con@d*_con while gold + pool kappa + Rand pass!
    rl_s15 = copy.deepcopy(base_labels)
    for i in range(40):
        k = f"pillar_a|common_pirate_ban|WSP_Pirate_Contrastive@d*_con|{pa_ids[i]}"
        # 20 items: ds=True, luna=(i<12); 20 items: ds=False, luna=(i>=32) -> prev=0.50, po=24/40=0.60, kappa=0.20 < 0.55!
        ds_v = i < 20
        lu_v = (i < 12) or (i >= 32)
        rl_s15["labels"][k]["ds"]["pirate_persona"] = ds_v
        rl_s15["labels"][k]["luna"]["pirate_persona"] = lu_v
    sc_s15 = _score(base, rl=rl_s15)
    assert sc_s15["gates"]["C3.R"]["gold_14"]["pass"] is True
    assert sc_s15["gates"]["C3.R"]["pool"]["pass"] is True
    assert sc_s15["gates"]["C3.R"]["per_arm_common_ban"]["Rand_Pirate_s0_s1_s2_pooled@d*_con"]["pass"] is True
    assert sc_s15["gates"]["C3.R"]["per_arm_common_ban"]["WSP_Pirate_Contrastive@d*_con"]["pass"] is False
    assert sc_s15["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    # 22. DS S25 / GLM S29: code_pass() positive-control preflight failure -> BLOCKED_BY_C3.0!
    orig_code_pass = c3_scoring.code_pass
    try:
        c3_scoring.code_pass = lambda *a, **kw: False  # type: ignore[assignment]
        sc_s25 = _score(base)
        assert sc_s25["confirmatory_verdicts"]["C3.0"] == "FAIL"
        assert sc_s25["gates"]["C3.0"]["C3.0.2"]["code_harness_preflight_ok"] is False
    finally:
        c3_scoring.code_pass = orig_code_pass

    # 23. DS S27: commit-gate ladder exactness — alter one unselected rung dose (0.80 -> 0.85) -> DOSE_INVALID!
    r_s27 = copy.deepcopy(base)
    r_s27["commit_gate"]["arms"]["WSP_Pirate_Contrastive"]["rungs"][-1]["dose"] = 0.85
    sc_s27 = _score(r_s27)
    assert sc_s27["gates"]["C3.0_commit_gate"]["WSP_Pirate_Contrastive"]["ladder_exact_ok"] is False
    assert sc_s27["confirmatory_verdicts"]["H1"] == "DOSE_INVALID"

    # 24. DS S28: pillar_a required-arm set — drop Sys_Pirate -> BLOCKED_BY_C3.0!
    r_s28 = copy.deepcopy(base)
    r_s28["pillar_a"]["none"].pop("Sys_Pirate")
    sc_s28 = _score(r_s28)
    assert sc_s28["gates"]["C3.0"]["C3.0.3"]["required_arms_ok"] is False
    assert sc_s28["confirmatory_verdicts"]["C3.0"] == "FAIL"

    print("PASS test_3_kill_all_14_round1_sabotages_and_new_hardening_clauses")


def test_4_hard_refusals_pinned_seeds_and_distinct_dstar() -> None:
    base = make_synthetic_passing_receipt()
    # Hard refusal of partial_checkpoint & smoke=True
    r_part = copy.deepcopy(base)
    r_part["partial_checkpoint"] = True
    try:
        scorer.score_stage_c3(r_part, C3_CONSTANTS, strict_provenance=False)
        raise AssertionError("Expected partial_checkpoint refusal")
    except ValueError as exc:
        assert "partial_checkpoint" in str(exc)

    r_smk = copy.deepcopy(base)
    r_smk["meta"]["smoke"] = True
    try:
        scorer.score_stage_c3(r_smk, C3_CONSTANTS, strict_provenance=False)
        raise AssertionError("Expected smoke refusal")
    except ValueError as exc:
        assert "smoke" in str(exc).lower()

    # DS F7 / GLM F5: run_code_subprocess=False MUST raise when code_pass_precomputed is absent!
    r_nopre = copy.deepcopy(base)
    r_nopre["capability_suite"]["Plain"]["code_12"][0].pop("code_pass_precomputed")
    try:
        scorer.evaluate_eq_c3_2(r_nopre, C3_CONSTANTS, run_code_subprocess=False)
        raise AssertionError("Expected missing code_pass_precomputed to raise ValueError")
    except ValueError as exc:
        assert "code_pass_precomputed" in str(exc)

    # Distinct d* + fixed 620-item kappa denominator + F7 flags + exact pinned PERM_SEED check (GLM F3!)
    rec_d = make_synthetic_passing_receipt(d_con=0.45, d_lex=0.15, d_nomd_con=0.30, d_nomd_lex=0.075)
    mock_fn, _ = make_mock_reader_fn()
    with tempfile.TemporaryDirectory() as td:
        rl_d = _audit_with_hashes(rec_d, Path(td), mock_fn, receipt_sha="sha_dist")
    sc = scorer.score_stage_c3(
        rec_d, C3_CONSTANTS, reader_labels=rl_d, disk_hashes=DISK_HASHES,
        receipt_sha256="sha_dist", strict_provenance=True, allow_mock_readers=True,
        n_perm=1000, n_boot=300, run_code_subprocess=False,
    )
    assert sc["confirmatory_verdicts"] == {"C3.0": "PASS", "H1": "PASS", "H2": "PASS", "H3": "PASS", "H4": "PASS"}
    assert sc["gates"]["C3.R"]["pool"]["expected_n_pooled"] == 620
    assert sc["gates"]["C3.R"]["pool"]["actual_n_pooled"] == 620
    f7 = sc["diagnostics"]["D7_operating_point_and_retention"]["f7_interpretation_flags"]
    assert f7["d_common_gt_1_rung_below_d_star_con"] is True
    assert f7["bottom_rung_0.075_selected_arms"] == ["WSP_NoMD_Lexicon"]

    # GLM F3 (R2): verify ALL FOUR hypotheses (H1, H2, H3, H4) match exact pinned PERM_SEED and BCA_SEED!
    pa_ids = [p["id"] for p in C3_CONSTANTS["batteries"]["identity_20"] + C3_CONSTANTS["batteries"]["implied_20"]]
    nomd_ids = [p["id"] for p in C3_CONSTANTS["batteries"]["nomd_24"]]

    def _sp_map(arm_k: str) -> dict[str, float]:
        return {
            pid: 1.0 if (rl_d["labels"][f"pillar_a|common_pirate_ban|{arm_k}|{pid}"]["ds"]["pirate_persona"]
                         and rl_d["labels"][f"pillar_a|common_pirate_ban|{arm_k}|{pid}"]["luna"]["pirate_persona"]) else 0.0
            for pid in pa_ids
        }

    sp_c = _sp_map("WSP_Pirate_Contrastive@d*_con")
    sp_r = [_sp_map(f"Rand_Pirate_s{s}@d*_con") for s in range(3)]
    sp_cdc = _sp_map("WSP_Pirate_Contrastive@d_common")
    sp_ldc = _sp_map("WSP_Pirate_Lexicon@d_common")
    d_h1 = [sp_c[pid] - scorer.mean([sp_r[s][pid] for s in range(3)]) for pid in pa_ids]
    d_h2 = [sp_cdc[pid] - sp_ldc[pid] for pid in pa_ids]

    def _u_vec(arm_k: str, rsuf: str) -> list[float]:
        am = {r["id"]: c3_scoring.construct_types(r["text"]) for r in rec_d["pillar_b"][arm_k]}
        rm = [{r["id"]: c3_scoring.construct_types(r["text"]) for r in rec_d["pillar_b"][f"Rand_NoMD_s{s}@{rsuf}"]} for s in range(3)]
        return [scorer.mean([float(rm[s][pid]) for s in range(3)]) - float(am[pid]) for pid in nomd_ids]

    d_h3 = _u_vec("WSP_NoMD_Contrastive@d*_nomd_con", "d*_nomd_con")
    d_h4 = _u_vec("WSP_NoMD_Lexicon@d*_nomd_lex", "d*_nomd_lex")
    for hid, dvec in (("H1", d_h1), ("H2", d_h2), ("H3", d_h3), ("H4", d_h4)):
        p_exp = round(scorer.paired_sign_flip_p_one_sided(dvec, n_perm=1000, seed=scorer.PERM_SEED), 8)
        ci_exp = {k: round(v, 6) for k, v in scorer.bca_paired_mean_ci90(dvec, n_boot=300, seed=scorer.BCA_SEED).items()}
        assert sc["holm_4_family"]["hypotheses"][hid]["p_perm"] == p_exp, f"GLM F3: {hid} p_perm seed mismatch!"
        assert sc["holm_4_family"]["hypotheses"][hid]["bca_ci90"] == ci_exp, f"GLM F3: {hid} BCa seed mismatch!"
    print("PASS test_4_hard_refusals_pinned_seeds_and_distinct_dstar")


def main() -> None:
    test_1_passing_roundtrip_blinding_real_code_and_byte_determinism()
    test_2_all_9_verdict_states_and_gate_isolation()
    test_3_kill_all_14_round1_sabotages_and_new_hardening_clauses()
    test_4_hard_refusals_pinned_seeds_and_distinct_dstar()
    print("ALL 4 PURE-STDLIB STAGE C3 SCORER & BLIND-AUDITOR TESTS PASSED.")


if __name__ == "__main__":
    main()
