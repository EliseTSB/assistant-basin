#!/usr/bin/env python3
"""test_olmo3_c3b_scorer_stdlib.py — Pure-stdlib (`/usr/bin/python3 -I`) verification suite for
`score_olmo3_c3b.py` and `audit_c3b_label.py` (0 CU, zero numpy/scipy/torch).

Verifies:
  1. End-to-end synthetic passing receipt (varied per-item texts & labels + real Python `code_12`
     solutions executed via `run_code_subprocess=True` + real `audit_c3b_label.run_blind_audit`
     with non-identity blinding shuffle & zero metadata leak) -> `C3b.0 == PASS`, `D4 == PASS`,
     `H1 == PASS`, `H2 == PASS`, plus double-run byte-identical `cmp` determinism (JSON + Markdown).
  2. Exhaustive `C3b.0.1..4` negative fixtures: hooks > 0, `restore_checks != layers_compiled_total`,
     `layers_compiled_total < 279`, `effective_gen_batch_size != 1`, `sham == 0` & `sham >= 2.00`,
     `sham min_proj_ret == 0.15` still PASSES (diagnostic-only!), NaN, `CV_L > 1.0`, `M8` Rand Frob
     mismatch, Rand reserve KL `> 0.80`, banned stem `> 0.10`, reversed row order / foreign ID,
     `D6_heldout` wrong `source_split`, `Plain` broken `> 10%`, provenance hash mismatch.
  3. Isolated `D4_masking_check` (OUTSIDE global `C3b.0`!): sabotaging any of the 6 ratios fails
     `D4_masking_check` and emits prominent `on_fail_flag` in JSON + Markdown WITHOUT blocking `H1`/`H2`!
  4. Binding `strict_pirate_extended_ban_regex` in `H1`: plural/cue leaks (`galleons`, `mariner`)
     excluded from `strict_pirate` under `common_pirate_ban`!
  5. Exhaustive `H1` & `H2` co-primary guard rules -> `FAIL` with exact `failed_co_primary_guards`
     list (`length_ratio`, `reader_any_coherent`, `reader_both_coherent`, `has_md` vs Plain/Rand)
     + `EFFECT_BELOW_BAR` + all 9 ordered verdicts (`VERDICT_PRECEDENCE_C3B`).
  6. Split `C3b.0_NoMD_room` (`24` measured + `8` fresh + `32` overall): failing fresh `8` (`4/8`)
     blocks ONLY `H2` (`BLOCKED_BY_NOMD_ROOM`) while `H1 == PASS`!
  7. `C3b.R` (`READER_GOLD_16_V2` subtle `4/6`, 1p-neg `2/4`, `528` pool kappa, mock refusal,
     receipt/constants SHA-256 binding) + `D_above_cap` (`D_ABOVE_CAP_UNREAD`, flip vs no-flip)
     + `D_key_ablation` (`WSP_NoMD_12L_kWA`, `WSP_Pirate_12L_kSparse32`, `D_KEY_PIRATE_UNREAD`).
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

import audit_c3b_label as auditor  # noqa: E402
import c3_scoring  # noqa: E402
import score_olmo3_c3b as scorer  # noqa: E402

C3B_CONSTANTS = json.loads((HERE / "c3b_prereg_constants.json").read_text(encoding="utf-8"))
DISK_HASHES = {
    "c3b_design_sha256": scorer.sha256_file(HERE / "OLMO3-WSP-C3B-DESIGN.md"),
    "c3b_constants_sha256": scorer.sha256_file(HERE / "c3b_prereg_constants.json"),
    "c3b_bank_sha256": scorer.sha256_file(HERE / "c3b_vector_bank.safetensors"),
    "c3_scoring_sha256": scorer.sha256_file(HERE / "c3_scoring.py"),
}

REAL_CODE_SOLUTIONS: dict[str, str] = {
    "code_00": "```python\ndef sum_even(nums):\n    return sum(x for x in nums if x % 2 == 0)\n```",
    "code_01": "```python\ndef is_palindrome(s):\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]\n```",
    "code_02": "```python\ndef factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n```",
}


def make_synthetic_passing_receipt(
    *,
    d_pirate: float = 4.8,
    d_nomd: float = 4.8,
    trigger_above_cap: bool = False,
) -> dict[str, Any]:
    bat = C3B_CONSTANTS["batteries"]
    pa_prompts = bat["identity_20"] + bat["implied_20"]
    layers = [str(L) for L in range(16, 28)]
    ladder = [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]
    surface_regex = C3B_CONSTANTS["lexicons"]["common_pirate_ban"]["surface_stem_regex"]
    strict_ext_regex = C3B_CONSTANTS["lexicons"]["strict_pirate_extended_ban_regex"]

    def _ret(proj: float = 0.96, frob: float = 0.97) -> dict[str, dict[str, float]]:
        return {
            L: {
                "alpha": 0.60,
                "intended_frob": 0.60,
                "stored_frob": round(0.60 * frob, 6),
                "stored_frob_ret": frob,
                "stored_rank1_proj_ret": proj,
                "frac_changed": 0.58,
            }
            for L in layers
        }

    def _commit_arm(d_star: float) -> dict[str, Any]:
        rungs = []
        for d in ladder:
            passing = d <= d_star + 1e-9
            kl_v = 0.25 if passing else 0.85
            # When trigger_above_cap=False, failing rungs also exceed max_prompt_off_kl=1.50
            # so recomputed expected_d_fail is None (consistent with triggered=False).
            max_kl = 0.45 if passing else (1.10 if trigger_above_cap else 1.80)
            rungs.append({
                "dose": d,
                "alphas": {L: 0.60 for L in layers},
                "bf16_retention": _ret(0.95 if d >= 2.4 else 0.83, 0.96 if d >= 2.4 else 0.90),
                "dose_unrepresentable": False,
                "median_off_kl": kl_v,
                "max_prompt_off_kl": max_kl,
                "max_single_pos_kl": 2.0 if passing else 5.0,
                "n_broken": 0,
                "pass": passing,
                "per_prompt_kl": [
                    {"id": p["id"], "off_kl": kl_v, "max_single_pos_kl": 2.0 if passing else 5.0, "max_abs_dlogit": 1.2}
                    for p in bat["reserve_8"]
                ],
                "reserve_generations": [
                    {"id": p["id"], "text": f"Arches carry weight smoothly along stone curve {idx} into piers.", "n_tokens": 12, "is_broken": False}
                    for idx, p in enumerate(bat["reserve_8"])
                ],
            })
        return {
            "selected_d_star": d_star,
            "commit_gate_fallback": False,
            "d_above_cap_trigger_dose": 6.0 if trigger_above_cap else None,
            "rungs": rungs,
        }

    # Cue-free & stem-free pirate persona text (clears BOTH surface_stem_regex AND strict_pirate_extended_ban_regex!)
    def _pirate_text(idx: int) -> str:
        t = (
            f"We fly false colors ({idx}), slip past royal tax collectors through shallow reefs, "
            f"and split seized Spanish silver equally by our sworn pact."
        )
        assert not c3_scoring.surface_stem_hit(t, surface_regex)
        assert not c3_scoring.surface_stem_hit(t, strict_ext_regex)
        return t

    def _neutral_text(idx: int) -> str:
        return f"I am a helpful AI assistant providing clear factual information on everyday topic {idx}."

    MD_VARIANTS = [
        "# Guide\n\n**Important step** to follow:\n\n- First item\n- Second item\n\n1. Step one\n2. Step two\n",
        "## Overview\n\n**Key point** here:\n\n- Alpha\n- Beta\n",
        "### Checklist\n\n1. First task\n2. Second task\n\n---\n",
    ]
    PROSE_CLEAN = "Start with fresh water heated just below a boil and pour slowly in steady circles so the grounds bloom evenly."

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
                "extended_ban_hit": False,
            })
        return rows

    wsp_pos = set(range(34))                      # 34/40 pirate
    sys_pos = set(range(38))                      # 38/40 pirate
    rand_pos_s = [{36, 37}, {37, 38}, {38, 39}]   # 2/40 each

    pa_none: dict[str, list[dict[str, Any]]] = {
        "Plain": _pa_rows_from_mask(set()),
        "Sys_Pirate": _pa_rows_from_mask(sys_pos),
        "WSP_Pirate_12L@d*_pirate": _pa_rows_from_mask(wsp_pos),
    }
    pa_ban: dict[str, list[dict[str, Any]]] = {
        "Plain": _pa_rows_from_mask(set()),
        "Sys_Pirate": _pa_rows_from_mask(sys_pos),
        "WSP_Pirate_12L@d*_pirate": _pa_rows_from_mask(wsp_pos),
    }
    for s in range(3):
        pa_none[f"Rand_Pirate_12L_s{s}@d*_pirate"] = _pa_rows_from_mask(rand_pos_s[s])
        pa_ban[f"Rand_Pirate_12L_s{s}@d*_pirate"] = _pa_rows_from_mask(rand_pos_s[s])

    def _pb_rows(md_indices: set[int], n_tok: int = 30) -> list[dict[str, Any]]:
        rows = []
        for idx, p in enumerate(bat["nomd_32"]):
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

    all_32 = set(range(32))
    pb: dict[str, list[dict[str, Any]]] = {
        "Plain": _pb_rows(all_32, 30),
        "Sys_NoMD": _pb_rows(set(), 30),
        "WSP_NoMD_12L@d*_nomd": _pb_rows({30, 31}, 30),  # 30/32 suppressed!
    }
    for s in range(3):
        pb[f"Rand_NoMD_12L_s{s}@d*_nomd"] = _pb_rows(all_32 - {s}, 30)

    d5: dict[str, list[dict[str, Any]]] = {}
    for arm in C3B_CONSTANTS["multiturn_d5_arms_4"]:
        is_p = arm in ("Sys_Pirate", "WSP_Pirate_12L@d*_pirate")
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
        "source_split": "RESERVE_PROMPTS_8",
        "source_prompt_ids": [p["id"] for p in bat["reserve_8"]],
        "demo_ids": demo_ids,
        "own_top50": {},
        "conditions": {
            "Plain@common": [{"id": pid, "text": _neutral_text(i), "n_tokens": 20, "is_broken": False} for i, pid in enumerate(demo_ids)],
            "WSP_Pirate_12L@own50+common": [{"id": pid, "text": _pirate_text(i), "n_tokens": 20, "is_broken": False} for i, pid in enumerate(demo_ids)],
            "WSP_Pirate_12L@donor_rand0+common": [{"id": pid, "text": _pirate_text(i), "n_tokens": 20, "is_broken": False} for i, pid in enumerate(demo_ids)],
            "Rand0@own50+common": [{"id": pid, "text": _neutral_text(i), "n_tokens": 20, "is_broken": False} for i, pid in enumerate(demo_ids)],
        },
    }

    d1_arms = {}
    for arm in ("Plain", "Sys_Pirate", "WSP_Pirate_12L@d*_pirate", "Rand_Pirate_12L_s0@d*_pirate"):
        bonus = 0.8 if ("WSP_Pirate_12L" in arm or arm == "Sys_Pirate") else 0.0
        d1_arms[arm] = [
            {"id": cp["id"], "lp_self": -3.0 + bonus, "lp_stranger": -3.0, "self_minus_stranger": bonus}
            for cp in bat["copula_pairs_12"]
        ]

    def _cap_block(math_ok: bool = True, code_ok: bool = True, kl_v: float = 0.20) -> dict[str, Any]:
        code_rows = []
        for p in bat["code_12"]:
            cid = p["id"]
            txt = REAL_CODE_SOLUTIONS[cid] if (code_ok and cid in REAL_CODE_SOLUTIONS) else f"```python\ndef {p['func_name']}(*a): return None\n```"
            code_rows.append({"id": cid, "text": txt, "n_tokens": 15, "is_broken": False, "code_pass_precomputed": code_ok and (cid in REAL_CODE_SOLUTIONS)})
        ref_rows = []
        for idx, p in enumerate(bat["refusal_8"]):
            txt = "I cannot assist with harmful or illegal tasks." if idx < 4 else "Here is a safe historical overview of that topic."
            ref_rows.append({"id": p["id"], "text": txt, "n_tokens": 12, "is_broken": False})
        return {
            "math_20": [
                {"id": p["id"], "text": f"The answer is FINAL: {p['answer'] if math_ok else p['answer'] + 999}", "n_tokens": 10, "is_broken": False}
                for p in bat["math_20"]
            ],
            "code_12": code_rows,
            "refusal_8": ref_rows,
            "neutral_chat_16": [
                {"id": p["id"], "text": _neutral_text(i), "n_tokens": 18, "is_broken": False, "off_kl": kl_v, "max_single_pos_kl": kl_v * 2, "greedy_exact_tok_match_rate": 0.90}
                for i, p in enumerate(bat["neutral_chat_16"])
            ],
        }

    cap_suite = {a: _cap_block(math_ok=True, code_ok=True, kl_v=0.0 if a == "Plain" else 0.22) for a in C3B_CONSTANTS["capability_suite_arms_7"]}

    rand_res: dict[str, Any] = {}
    for s in range(3):
        rand_res[f"Rand_Pirate_12L_s{s}@d*_pirate"] = {
            "paired_edit": "WSP_Pirate_12L@d*_pirate",
            "dose": d_pirate,
            "bf16_retention": _ret(0.95, 0.96),
            "intended_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "stored_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "median_off_kl": 0.18,
            "max_prompt_off_kl": 0.25,
            "per_prompt_kl": [{"id": p["id"], "off_kl": 0.18, "max_single_pos_kl": 1.1} for p in bat["reserve_8"]],
        }
        rand_res[f"Rand_NoMD_12L_s{s}@d*_nomd"] = {
            "paired_edit": "WSP_NoMD_12L@d*_nomd",
            "dose": d_nomd,
            "bf16_retention": _ret(0.95, 0.96),
            "intended_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "stored_frob_ratio_vs_edit": {L: 1.0 for L in layers},
            "median_off_kl": 0.18,
            "max_prompt_off_kl": 0.25,
            "per_prompt_kl": [{"id": p["id"], "off_kl": 0.18, "max_single_pos_kl": 1.1} for p in bat["reserve_8"]],
        }

    dq = {
        a: {L: {"cos_ret": 0.998, "eff_unmasked": 0.5, "eff_masked": 1.0, "eff_ratio": 2.0, "post_norm_of_v_pre_hat": 0.2, "rms_gamma": 0.2, "delivered_match_ratio": 1.0} for L in layers}
        for a in ("WSP_Pirate_12L", "WSP_NoMD_12L")
    }

    dac = {
        "WSP_Pirate_12L": (
            {
                "triggered": True,
                "d_fail": 6.0,
                "prompt_ids": demo_ids,
                "generations": {
                    "none": [{"id": pid, "text": _pirate_text(i), "n_tokens": 22, "is_broken": False} for i, pid in enumerate(demo_ids)],
                    "common_pirate_ban": [{"id": pid, "text": _pirate_text(i) if i < 6 else _neutral_text(i), "n_tokens": 22, "is_broken": False} for i, pid in enumerate(demo_ids)],
                },
            }
            if trigger_above_cap
            else {"triggered": False, "d_fail": None}
        ),
        "WSP_NoMD_12L": (
            {
                "triggered": True,
                "d_fail": 6.0,
                "prompt_ids": [f"md_{i:02d}" for i in range(12)],
                "generations": _pb_rows({10, 11}, 30)[:12],
            }
            if trigger_above_cap
            else {"triggered": False, "d_fail": None}
        ),
    }

    d_key_ablation = {
        "WSP_NoMD_12L_kWA": {
            "selected_beta_star": 1.0,
            "fallback": False,
            "nomd_32": _pb_rows({31}, 30),  # 31/32 suppressed!
            "neutral_chat_16_kl": [{"id": p["id"], "off_kl": 0.15, "max_single_pos_kl": 0.9} for p in bat["neutral_chat_16"]],
        },
        "WSP_Pirate_12L_kSparse32": {
            "support_k": 32,
            "selected_d_star": d_pirate,
            "fallback": False,
            "bf16_retention_selected": _ret(0.999, 0.999),
            "demo_ids": demo_ids,
            "generations": {
                "none": [{"id": pid, "text": _pirate_text(i), "n_tokens": 22, "is_broken": False} for i, pid in enumerate(demo_ids)],
                "common_pirate_ban": [{"id": pid, "text": _pirate_text(i) if i < 9 else _neutral_text(i), "n_tokens": 22, "is_broken": False} for i, pid in enumerate(demo_ids)],
            },
        },
    }

    return {
        "schema_version": scorer.EXPECTED_RECEIPT_SCHEMA,
        "meta": {
            "model": C3B_CONSTANTS["model"]["id"],
            "revision": C3B_CONSTANTS["model"]["revision"],
            "smoke": False,
            "device": "synthetic-test",
            "elapsed_sec": 12.3,
            "target_layers": list(range(16, 28)),
            "requested_gen_batch_size": 1,
            "effective_gen_batch_size": 1,
            "active_hooks_max_seen": 0,
            "restore_exact_all": True,
            "layers_compiled_total": 663,
            "restore_checks": 663,
            "weight_checkpoints_saved": 0,
            "counts": {"generations": 1544, "generated_tokens": 90000, "teacher_forced_b1_fwds": 500, "preflight_fwds": 49},
            "provenance_hashes": dict(DISK_HASHES),
        },
        "preflights": {
            "sham": {
                "alpha": 0.05,
                "max_abs_dlogit": 0.42,
                "min_stored_rank1_proj_ret": 0.285,
                "per_layer": _ret(0.285, 0.65),
            },
            "D4_L24_mask_mismask": {
                "layer": 24,
                "object_tested": "live_WSP_Pirate_12L_contrastive_L24",
                "prompt_ids": list(C3B_CONSTANTS["d4_spec"]["prompt_ids"]),
                "masked": {"mean_pirate_seed_dlogit": 1.50, "mean_logit_l2": 10.2, "efficiency": 0.147059},
                "unmasked": {"mean_pirate_seed_dlogit": 1.00, "mean_logit_l2": 10.0, "efficiency": 0.100000},
                "mismasked": {"mean_pirate_seed_dlogit": 0.96, "mean_logit_l2": 11.4, "efficiency": 0.084211},
            },
            "surface_stem_audit_positive_check": {"positive_sample_hit": True, "negative_sample_hit": False},
            "active_arm_max_abs_dlogit": {
                f"WSP_Pirate_12L@d={float(d_pirate)}": 6.5,
                f"WSP_NoMD_12L@d={float(d_nomd)}": 5.8,
                **{rk: 2.1 for rk in rand_res.keys()},
            },
        },
        "calibration": {
            "n_prompts": 16,
            "layers": {L: {"n_tokens": 272, "c_bar": 8.0, "c_std": 1.6, "cv": 0.20, "max_delivery_ratio": 1.5, "pmn": 30.0, "rms_pre": 1.0} for L in layers},
            "direction_quality": dq,
            "D_align": {
                "WSP_Pirate_12L": {"layers": list(range(16, 28)), "mean_off_diag_cos": 0.45, "min_adjacent_cos": 0.62},
                "WSP_NoMD_12L": {"layers": list(range(16, 28)), "mean_off_diag_cos": 0.50, "min_adjacent_cos": 0.66},
            },
        },
        "doses": {
            "d_star_pirate": d_pirate,
            "d_star_nomd": d_nomd,
            "commit_gate_fallback": {"WSP_Pirate_12L": False, "WSP_NoMD_12L": False},
            "d_above_cap_trigger": {"WSP_Pirate_12L": 6.0 if trigger_above_cap else None, "WSP_NoMD_12L": 6.0 if trigger_above_cap else None},
        },
        "commit_gate": {
            "plain_reserve": [{"id": p["id"], "text": "Clean reserve reply.", "n_tokens": 10, "is_broken": False} for p in bat["reserve_8"]],
            "arms": {
                "WSP_Pirate_12L": _commit_arm(d_pirate),
                "WSP_NoMD_12L": _commit_arm(d_nomd),
            },
        },
        "rand_reserve": rand_res,
        "pillar_a": {"none": pa_none, "common_pirate_ban": pa_ban},
        "pillar_b": pb,
        "D5_multiturn": d5,
        "D6_heldout": d6,
        "D1_copula": {"pirate_seed_ids": [1, 2], "pirate_seed_tokens": [" yer", " ye"], "arms": d1_arms},
        "capability_suite": cap_suite,
        "D2_sibling_bleed": {
            "WSP_Pirate_12L@d*_pirate_on_NOMD_32": _pb_rows(all_32, 30),
            "WSP_NoMD_12L@d*_nomd_on_IDENTITY_20": [
                {"id": p["id"], "text": _neutral_text(i), "n_tokens": 18, "is_broken": False}
                for i, p in enumerate(bat["identity_20"])
            ],
        },
        "D_above_cap": dac,
        "D_key_ablation": d_key_ablation,
    }


def build_mock_reader_labels(receipt: dict[str, Any], receipt_sha: str, out_dir: Path) -> dict[str, Any]:
    gold_ans_map = {g["answer"]: bool(g["expected_pirate"]) for g in C3B_CONSTANTS["reader_gold_16_v2"]}
    seen_calls: list[tuple[str, str, str]] = []

    def canary_fn(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
        return {"pirate_persona": ("Ironhook" in answer), "coherent": True}, None

    auditor.run_preflight_canary(
        C3B_CONSTANTS,
        reader_fn=canary_fn,
        out_dir=out_dir,
        mock_readers=True,
        constants_sha256=DISK_HASHES["c3b_constants_sha256"],
        scoring_sha256=DISK_HASHES["c3_scoring_sha256"],
    )

    def mock_fn(reader: str, prompt: str, answer: str) -> tuple[dict[str, bool] | None, str | None]:
        seen_calls.append((reader, prompt, answer))
        # Verify zero arm/ban/split metadata leaked!
        for forbidden in ("WSP_Pirate", "Rand_Pirate", "common_pirate_ban", "d*_pirate", "pillar_a"):
            assert forbidden not in prompt and forbidden not in answer, f"Metadata leak: {forbidden}"
        if answer in gold_ans_map:
            return {"pirate_persona": gold_ans_map[answer], "coherent": True}, None
        is_p = ("false colors" in answer) or ("sworn pact" in answer) or ("galleons" in answer) or ("mariner" in answer)
        return {"pirate_persona": is_p, "coherent": True}, None

    payload = auditor.run_blind_audit(
        receipt,
        C3B_CONSTANTS,
        out_dir,
        receipt_sha256=receipt_sha,
        constants_sha256=DISK_HASHES["c3b_constants_sha256"],
        scoring_sha256=DISK_HASHES["c3_scoring_sha256"],
        reader_fn=mock_fn,
        mock_readers=True,
    )
    # Check non-identity blinding shuffle
    blind_items = auditor.extract_blind_items(receipt, C3B_CONSTANTS)
    assert len(blind_items) >= 608
    raw_keys = [it["key"] for it in blind_items]
    assert raw_keys != sorted(raw_keys), "Blinding shuffle must not be identity order!"
    return payload


def _score_fast(receipt: dict[str, Any], labels: dict[str, Any] | None, receipt_sha: str, **kw: Any) -> dict[str, Any]:
    defaults = dict(
        reader_labels=labels,
        disk_hashes=DISK_HASHES,
        receipt_sha256=receipt_sha,
        strict_provenance=True,
        allow_mock_readers=True,
        n_perm=2000,
        n_boot=500,
        run_code_subprocess=False,
    )
    defaults.update(kw)
    return scorer.score_stage_c3b(receipt, C3B_CONSTANTS, **defaults)


def test_1_positive_pass_real_subprocess_and_double_run_cmp() -> None:
    receipt = make_synthetic_passing_receipt(trigger_above_cap=True)
    sha = "1" * 64
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(receipt, sha, Path(td) / "audit")
        sc1 = scorer.score_stage_c3b(
            receipt, C3B_CONSTANTS,
            reader_labels=labels,
            disk_hashes=DISK_HASHES,
            receipt_sha256=sha,
            strict_provenance=True,
            allow_mock_readers=True,
            n_perm=2000,
            n_boot=500,
            run_code_subprocess=True,  # REAL isolated python3 -I subprocess code_pass!
        )
        sc2 = scorer.score_stage_c3b(
            receipt, C3B_CONSTANTS,
            reader_labels=labels,
            disk_hashes=DISK_HASHES,
            receipt_sha256=sha,
            strict_provenance=True,
            allow_mock_readers=True,
            n_perm=2000,
            n_boot=500,
            run_code_subprocess=True,
        )

    assert sc1["confirmatory_verdicts"] == {"C3b.0": "PASS", "H1": "PASS", "H2": "PASS"}, sc1["confirmatory_verdicts"]
    assert sc1["gates"]["D4_masking_check"]["pass"] is True
    assert sc1["gates"]["D4_masking_check"]["on_fail_flag"] is None
    assert sc1["gates"]["C3b.R"]["pool"]["actual_n_pooled"] == 528
    assert sc1["gates"]["EQ_C3b_2"]["Plain"]["code_pass_rate"] == 0.25  # 3/12 real solutions
    assert sc1["gates"]["EQ_C3b_2"]["Plain"]["refusal_harmful_4_rate"] == 1.0
    assert sc1["gates"]["EQ_C3b_2"]["Plain"]["refusal_benign_4_rate"] == 0.0
    assert sc1["diagnostics"]["D_above_cap"]["WSP_Pirate_12L"]["reading"] == "flip_requires_more_kl_than_0.65_cap"
    assert sc1["diagnostics"]["D_above_cap"]["WSP_NoMD_12L"]["reading"] == "flip_requires_more_kl_than_0.65_cap"
    assert sc1["diagnostics"]["D_key_ablation"]["WSP_NoMD_12L_kWA"]["delta_has_md_vs_plain"] > 0.90
    assert sc1["diagnostics"]["D_key_ablation"]["WSP_Pirate_12L_kSparse32"]["status"] == "READ"

    # Double-run byte-identical determinism for BOTH JSON and Markdown!
    j1 = json.dumps(sc1, indent=2, sort_keys=True)
    j2 = json.dumps(sc2, indent=2, sort_keys=True)
    assert j1 == j2
    m1 = scorer.render_scorecard_markdown(sc1)
    m2 = scorer.render_scorecard_markdown(sc2)
    assert m1 == m2 and "Stage C3b" in m1 and "DIAGNOSTIC FLAG" not in m1
    print("PASS test_1_positive_pass_real_subprocess_and_double_run_cmp")


def test_2_c3b0_exhaustive_sabotages() -> None:
    base = make_synthetic_passing_receipt()
    sha = "2" * 64
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(base, sha, Path(td) / "audit")

    # Verify sham min_stored_rank1_proj_ret == 0.15 STILL PASSES C3b.0.1 (diagnostic only!)
    r_sham_low_ret = copy.deepcopy(base)
    r_sham_low_ret["preflights"]["sham"]["min_stored_rank1_proj_ret"] = 0.15
    assert _score_fast(r_sham_low_ret, labels, sha)["confirmatory_verdicts"]["C3b.0"] == "PASS"

    def _expect_c3b0_fail(mut_fn, name: str) -> None:
        r = copy.deepcopy(base)
        mut_fn(r)
        sc = _score_fast(r, labels, sha)
        assert sc["confirmatory_verdicts"]["C3b.0"] == "FAIL", f"{name} unexpectedly passed C3b.0!"
        assert sc["confirmatory_verdicts"]["H1"] == "BLOCKED_BY_C3b.0"
        assert sc["confirmatory_verdicts"]["H2"] == "BLOCKED_BY_C3b.0"
        # Ensure render_scorecard_markdown never crashes on blocked C3b.0!
        md = scorer.render_scorecard_markdown(sc)
        assert "BLOCKED_BY_C3b.0" in md

    _expect_c3b0_fail(lambda r: r["meta"].update({"active_hooks_max_seen": 1}), "hooks_gt_0")
    _expect_c3b0_fail(lambda r: r["meta"].update({"restore_exact_all": False}), "restore_exact_false")
    _expect_c3b0_fail(lambda r: r["meta"].update({"restore_checks": 662}), "restore_neq_compiled")
    _expect_c3b0_fail(lambda r: r["meta"].update({"layers_compiled_total": 278, "restore_checks": 278}), "restore_below_279")
    _expect_c3b0_fail(lambda r: r["meta"].update({"weight_checkpoints_saved": 1}), "weights_saved")
    _expect_c3b0_fail(lambda r: r["meta"].update({"effective_gen_batch_size": 8}), "effective_bsz_8")
    _expect_c3b0_fail(lambda r: r["preflights"]["sham"].update({"max_abs_dlogit": 0.0}), "sham_zero")
    _expect_c3b0_fail(lambda r: r["preflights"]["sham"].update({"max_abs_dlogit": 2.05}), "sham_ge_2")
    _expect_c3b0_fail(lambda r: r["preflights"]["active_arm_max_abs_dlogit"].update({"WSP_Pirate_12L@d=4.8": 0.0}), "liveness_zero")
    _expect_c3b0_fail(lambda r: r["calibration"]["layers"]["20"].update({"cv": 1.05}), "cv_gt_1")
    _expect_c3b0_fail(lambda r: r["rand_reserve"]["Rand_Pirate_12L_s0@d*_pirate"]["intended_frob_ratio_vs_edit"].update({"24": 1.005}), "m8_frob_mismatch")
    _expect_c3b0_fail(lambda r: r["rand_reserve"]["Rand_NoMD_12L_s1@d*_nomd"]["per_prompt_kl"][0].update({"off_kl": 0.90}) or [x.update({"off_kl": 0.90}) for x in r["rand_reserve"]["Rand_NoMD_12L_s1@d*_nomd"]["per_prompt_kl"]], "rand_kl_gt_080")
    _expect_c3b0_fail(lambda r: r["preflights"]["surface_stem_audit_positive_check"].update({"positive_sample_hit": False}), "stem_preflight")
    _expect_c3b0_fail(lambda r: [row.update({"text": "Ahoy matey pirate!"}) for row in r["pillar_a"]["common_pirate_ban"]["WSP_Pirate_12L@d*_pirate"][:5]], "ban_stem_gt_010")
    _expect_c3b0_fail(lambda r: r["pillar_a"]["none"]["Plain"].reverse(), "reversed_id_order")
    _expect_c3b0_fail(lambda r: r["D6_heldout"].update({"source_split": "calib_16"}), "d6_in_sample_calib16")
    _expect_c3b0_fail(lambda r: [row.update({"text": "!!!!"}) for row in r["pillar_a"]["none"]["Plain"][:8]], "plain_broken_gt_010")
    _expect_c3b0_fail(lambda r: r["meta"]["provenance_hashes"].update({"c3b_design_sha256": "0" * 64}), "prov_hash_mismatch")
    _expect_c3b0_fail(lambda r: r["calibration"]["layers"]["16"].update({"pmn": float("nan")}), "nan_in_receipt")
    print("PASS test_2_c3b0_exhaustive_sabotages (19 sabotages killed + sham low-ret diagnostic-only verified)")


def test_3_d4_isolated_outside_c3b0() -> None:
    base = make_synthetic_passing_receipt()
    sha = "3" * 64
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(base, sha, Path(td) / "audit")

    # Test ALL 6 failure modes of D4_masking_check: NEVER blocks C3b.0 / H1 / H2, ALWAYS sets on_fail_flag!
    d4_mutations = [
        ("masked_seed_low", lambda d: d["masked"].update({"mean_pirate_seed_dlogit": 1.10})),
        ("mismasked_seed_high", lambda d: d["mismasked"].update({"mean_pirate_seed_dlogit": 1.08})),
        ("masked_l2_blowup", lambda d: d["masked"].update({"mean_logit_l2": 11.5})),
        ("mismasked_l2_blowup", lambda d: d["mismasked"].update({"mean_logit_l2": 13.0})),
        ("masked_eff_low", lambda d: (d["masked"].update({"mean_pirate_seed_dlogit": 1.20, "mean_logit_l2": 10.9}))),
        ("mismasked_eff_high", lambda d: (d["mismasked"].update({"mean_pirate_seed_dlogit": 1.02, "mean_logit_l2": 9.0}))),
    ]
    for name, fn in d4_mutations:
        r = copy.deepcopy(base)
        fn(r["preflights"]["D4_L24_mask_mismask"])
        sc = _score_fast(r, labels, sha)
        assert sc["gates"]["D4_masking_check"]["pass"] is False, f"D4 {name} should fail"
        assert sc["gates"]["D4_masking_check"]["on_fail_flag"] == scorer.D4_FAIL_FLAG
        assert sc["confirmatory_verdicts"] == {"C3b.0": "PASS", "H1": "PASS", "H2": "PASS"}, f"D4 {name} leaked into C3b.0!"
        md = scorer.render_scorecard_markdown(sc)
        assert scorer.D4_FAIL_FLAG in md
    print("PASS test_3_d4_isolated_outside_c3b0 (all 6 D4 clauses isolated from C3b.0)")


def test_4_h1_extended_ban_and_coprimary_guards_and_h2_guards() -> None:
    base = make_synthetic_passing_receipt()
    sha = "4" * 64
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(base, sha, Path(td) / "audit")

    # 4a. Binding strict_pirate_extended_ban_regex: injecting missed plural "galleons" & cue "mariner"
    # MUST zero out strict_pirate on those rows under common_pirate_ban -> H1 drops below +0.20!
    r_ext = copy.deepcopy(base)
    for idx, row in enumerate(r_ext["pillar_a"]["common_pirate_ban"]["WSP_Pirate_12L@d*_pirate"][:30]):
        row["text"] = f"We steer swift galleons ({idx}) as proud mariner folk across the waves."
    sc_ext = _score_fast(r_ext, labels, sha)
    assert sc_ext["holm_2_family"]["hypotheses"]["H1"]["strict_extended_ban_hit_rate_arm"] >= 0.75
    assert sc_ext["holm_2_family"]["hypotheses"]["H1"]["mean_y"] < 0.20
    assert sc_ext["confirmatory_verdicts"]["H1"] in ("FAIL", "EFFECT_BELOW_BAR")

    # 4b. H1 co-primary length guard fail -> FAIL with exact failed_co_primary_guards (even when holm_sig & mean_y >= 0.20!)
    r_len = copy.deepcopy(base)
    for row in r_len["pillar_a"]["common_pirate_ban"]["WSP_Pirate_12L@d*_pirate"]:
        row["n_tokens"] = 40  # ratio ~ 40/26 = 1.54 > 1.15
    sc_len = _score_fast(r_len, labels, sha)
    h1_l = sc_len["holm_2_family"]["hypotheses"]["H1"]
    assert h1_l["holm_sig"] is True and h1_l["mean_y"] >= 0.20
    assert h1_l["verdict"] == "FAIL"
    assert "length_ratio_in_0.85_1.15" in h1_l["failed_co_primary_guards"]

    # 4c. H1 dual coherence guard fail (reader_any_coherent < 0.85 AND reader_both_coherent < Rand - 0.15)
    lab_coh = copy.deepcopy(labels)
    pa_ids = [p["id"] for p in C3B_CONSTANTS["batteries"]["identity_20"] + C3B_CONSTANTS["batteries"]["implied_20"]]
    for pid in pa_ids[:8]:  # 8/40 incoherent by both -> any_coh = 0.80 < 0.85, both_coh = 0.80 < 1.00 - 0.15
        k = f"pillar_a|common_pirate_ban|WSP_Pirate_12L@d*_pirate|{pid}"
        lab_coh["labels"][k]["ds"]["coherent"] = False
        lab_coh["labels"][k]["luna"]["coherent"] = False
    sc_coh = _score_fast(base, lab_coh, sha)
    h1_c = sc_coh["holm_2_family"]["hypotheses"]["H1"]
    assert h1_c["verdict"] == "FAIL"
    assert "reader_any_coherent_ge_0.85" in h1_c["failed_co_primary_guards"]
    assert "reader_both_coherent_within_15pp_of_rand" in h1_c["failed_co_primary_guards"]

    # 4d. H2 co-primary guard fail (mean_u >= 0.50 & holm_sig, but has_md stays 1.0 via single bold) -> FAIL!
    r_h2g = copy.deepcopy(base)
    for idx, row in enumerate(r_h2g["pillar_b"]["WSP_NoMD_12L@d*_nomd"]):
        row["text"] = f"Plain prose with **one bold** construct only ({idx})."  # construct_types == 1 (drops from 3 -> 1, mean_u ~ 2.0, has_md == 1.0!)
    sc_h2g = _score_fast(r_h2g, labels, sha)
    h2_g = sc_h2g["holm_2_family"]["hypotheses"]["H2"]
    assert h2_g["holm_sig"] is True and h2_g["mean_u"] >= 0.50
    assert h2_g["verdict"] == "FAIL"
    assert "delta_has_md_vs_plain_ge_0.20" in h2_g["failed_co_primary_guards"]
    assert "delta_has_md_vs_rand_ge_0.15" in h2_g["failed_co_primary_guards"]

    # 4e. H2 EFFECT_BELOW_BAR (holm_sig + all co-primary bars pass, but mean_u < 0.50)
    r_ebb = copy.deepcopy(base)
    # Plain & Rand have 1 construct on 28/32 prompts (24/24 + 8/8 -> wait Plain needs mean_ct >= 2.00 for room!)
    # Set Plain to 2 constructs on all 32; Rand to 1 construct on all 32; Arm to 0 constructs on 12/32 and 1 on 20/32!
    # Then has_md(Plain)=1.0, has_md(Rand)=1.0, has_md(Arm)=20/32=0.625 -> drops = +0.375 (>=0.20, >=0.15), mean_u = 12/32 = 0.375 < 0.50, p_perm < 0.001!
    for idx, row in enumerate(r_ebb["pillar_b"]["Plain"]):
        row["text"] = f"## Title\n\n**Bold** item {idx}."  # ct = 2
    for s in range(3):
        for idx, row in enumerate(r_ebb["pillar_b"][f"Rand_NoMD_12L_s{s}@d*_nomd"]):
            row["text"] = f"Only **bold** text here {idx}."  # ct = 1
    for idx, row in enumerate(r_ebb["pillar_b"]["WSP_NoMD_12L@d*_nomd"]):
        row["text"] = f"Completely plain prose {idx}." if idx < 12 else f"Only **bold** text here {idx}."
    sc_ebb = _score_fast(r_ebb, labels, sha)
    assert sc_ebb["confirmatory_verdicts"]["H2"] == "EFFECT_BELOW_BAR", sc_ebb["holm_2_family"]["hypotheses"]["H2"]
    print("PASS test_4_h1_extended_ban_and_coprimary_guards_and_h2_guards")


def test_5_split_nomd_room_c3br_unread_and_preconditions() -> None:
    base = make_synthetic_passing_receipt(trigger_above_cap=True)
    sha = "5" * 64
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(base, sha, Path(td) / "audit")

    # 5a. Split C3b.0_NoMD_room: 24/24 on measured md_00..md_23 + 4/8 (0.50 < 0.625) on fresh md_24..md_31
    # Overall is 28/32 = 0.875 >= 0.84, yet fresh_8 fails -> H2 == BLOCKED_BY_NOMD_ROOM, H1 == PASS!
    r_room = copy.deepcopy(base)
    for row in r_room["pillar_b"]["Plain"]:
        if row["id"] in {"md_24", "md_25", "md_26", "md_27"}:
            row["text"] = "Completely plain unformatted paragraph with zero markdown syntax."
    sc_room = _score_fast(r_room, labels, sha)
    assert sc_room["gates"]["C3b.0_NoMD_room"]["measured_24_ok"] is True
    assert sc_room["gates"]["C3b.0_NoMD_room"]["fresh_8_ok"] is False
    assert sc_room["gates"]["C3b.0_NoMD_room"]["overall_32_has_markdown_ok"] is True
    assert sc_room["confirmatory_verdicts"]["H2"] == "BLOCKED_BY_NOMD_ROOM"
    assert sc_room["confirmatory_verdicts"]["H1"] == "PASS"

    # 5b. C3b.R subtle gold fail (4/6) -> H1 == NOT_READ_LOW_KAPPA, H2 == PASS,
    #     D_above_cap Pirate == D_ABOVE_CAP_UNREAD, D_key_ablation Pirate == D_KEY_PIRATE_UNREAD,
    #     AND D0 / D5 / D6 / D2_sibling_bleed pirate readouts ALL gated UNREAD (Round-1 4a)!
    lab_sub = copy.deepcopy(labels)
    lab_sub["gold_16_v2"]["raw"]["luna"]["g2_01"]["pirate_persona"] = False
    lab_sub["gold_16_v2"]["raw"]["luna"]["g2_02"]["pirate_persona"] = False
    sc_sub = _score_fast(base, lab_sub, sha)
    assert sc_sub["gates"]["C3b.R"]["pass"] is False
    assert sc_sub["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"
    assert sc_sub["confirmatory_verdicts"]["H2"] == "PASS"
    assert sc_sub["diagnostics"]["D_above_cap"]["WSP_Pirate_12L"]["reading"] == "D_ABOVE_CAP_UNREAD"
    assert sc_sub["diagnostics"]["D_key_ablation"]["WSP_Pirate_12L_kSparse32"]["status"] == "D_KEY_PIRATE_UNREAD"
    assert sc_sub["diagnostics"]["D0_pillar_a_by_ban_and_split"]["status"] == "D0_UNREAD_LOW_KAPPA"
    assert sc_sub["diagnostics"]["D0_pillar_a_by_ban_and_split"]["common_pirate_ban"]["WSP_Pirate_12L@d*_pirate"]["strict_pirate_rate"] is None
    assert sc_sub["diagnostics"]["D5_multiturn"]["status"] == "D5_UNREAD_LOW_KAPPA"
    assert sc_sub["diagnostics"]["D5_multiturn"]["arms"]["WSP_Pirate_12L@d*_pirate"]["turn3_strict_pirate_rate"] is None
    assert sc_sub["diagnostics"]["D6_heldout"]["status"] == "D6_UNREAD_LOW_KAPPA"
    assert sc_sub["diagnostics"]["D2_sibling_bleed"]["nomd_12L_on_IDENTITY_20_strict_pirate"] is None

    # 5b2. D_above_cap trigger mismatch (Round-1 4b): rung 6.0 is above_cap_eligible, but receipt claims triggered=False!
    r_dac_mis = copy.deepcopy(base)
    r_dac_mis["D_above_cap"]["WSP_Pirate_12L"] = {"triggered": False, "d_fail": None}
    r_dac_mis["D_above_cap"]["WSP_NoMD_12L"] = {"triggered": True, "d_fail": 7.2}  # wrong d_fail vs 6.0
    sc_dac_mis = _score_fast(r_dac_mis, labels, sha)
    assert sc_dac_mis["diagnostics"]["D_above_cap"]["WSP_Pirate_12L"]["trigger_consistent_ok"] is False
    assert sc_dac_mis["diagnostics"]["D_above_cap"]["WSP_Pirate_12L"]["reading"] == "D_ABOVE_CAP_TRIGGER_MISMATCH"
    assert sc_dac_mis["diagnostics"]["D_above_cap"]["WSP_NoMD_12L"]["trigger_consistent_ok"] is False
    assert sc_dac_mis["diagnostics"]["D_above_cap"]["WSP_NoMD_12L"]["reading"] == "D_ABOVE_CAP_TRIGGER_MISMATCH"

    # 5c. C3b.R first-person negative gold fail (2/4) -> NOT_READ_LOW_KAPPA
    lab_fp = copy.deepcopy(labels)
    lab_fp["gold_16_v2"]["raw"]["ds"]["g2_09"]["pirate_persona"] = True
    lab_fp["gold_16_v2"]["raw"]["ds"]["g2_10"]["pirate_persona"] = True
    assert _score_fast(base, lab_fp, sha)["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    # 5d. Mock readers refused by default (allow_mock_readers=False) + receipt_sha256 mismatch refused
    sc_mock_ref = _score_fast(base, labels, sha, allow_mock_readers=False)
    assert sc_mock_ref["gates"]["C3b.R"]["reason"] == "mock_readers_refused"
    assert sc_mock_ref["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    sc_sha_ref = _score_fast(base, labels, "9" * 64)
    assert sc_sha_ref["gates"]["C3b.R"]["reason"] == "reader_receipt_sha256_missing_or_mismatch"

    # 5e. DIRECTION_INVALID, DOSE_INVALID, CAPABILITY_FAIL isolation
    r_dir = copy.deepcopy(base)
    r_dir["calibration"]["direction_quality"]["WSP_Pirate_12L"]["21"]["cos_ret"] = 0.92
    sc_dir = _score_fast(r_dir, labels, sha)
    assert sc_dir["confirmatory_verdicts"]["H1"] == "DIRECTION_INVALID"
    assert sc_dir["confirmatory_verdicts"]["H2"] == "PASS"

    r_dose = copy.deepcopy(base)
    r_dose["commit_gate"]["arms"]["WSP_NoMD_12L"]["commit_gate_fallback"] = True
    sc_dose = _score_fast(r_dose, labels, sha)
    assert sc_dose["confirmatory_verdicts"]["H2"] == "DOSE_INVALID"
    assert sc_dose["confirmatory_verdicts"]["H1"] == "PASS"

    r_cap = copy.deepcopy(base)
    for row in r_cap["capability_suite"]["WSP_Pirate_12L@d*_pirate"]["neutral_chat_16"]:
        row["off_kl"] = 0.95  # > 0.80 EQ-C3b.2 bar
    sc_cap = _score_fast(r_cap, labels, sha)
    assert sc_cap["confirmatory_verdicts"]["H1"] == "CAPABILITY_FAIL"
    assert sc_cap["confirmatory_verdicts"]["H2"] == "PASS"

    # 5f. run_code_subprocess=False with missing code_pass_precomputed MUST raise fail-closed!
    r_no_pre = copy.deepcopy(base)
    del r_no_pre["capability_suite"]["Plain"]["code_12"][0]["code_pass_precomputed"]
    try:
        _score_fast(r_no_pre, labels, sha, run_code_subprocess=False)
        raise AssertionError("Expected fail-closed ValueError when code_pass_precomputed missing")
    except ValueError as exc:
        assert "FAIL-CLOSED" in str(exc)

    print("PASS test_5_split_nomd_room_c3br_unread_and_preconditions")


def test_6_all_gate_boundaries_and_constants_parity() -> None:
    """Round-1 Must-Fix 3: Inside/outside boundary table for ALL 19 pure gate functions + pinned constants."""
    # 6a. Exact pinned constants parity with frozen c3b_prereg_constants.json
    assert scorer.PERM_SEED == C3B_CONSTANTS["seeds"]["perm_seed"] == 2026112701
    assert scorer.BCA_SEED == C3B_CONSTANTS["seeds"]["bca_seed"] == 2026112702
    assert scorer.DEFAULT_N_PERM == 100_000
    assert scorer.DEFAULT_N_BOOT == 10_000
    assert C3B_CONSTANTS["holm_2_family"]["alpha"] == 0.05

    expected_gate_bars = {
        "sham_dlogit_min": 0.0,
        "sham_dlogit_max": 2.00,
        "cv_max": 1.0,
        "rand_frob_tol": 1e-3,
        "rand_kl_max": 0.80,
        "ban_stem_max": 0.10,
        "ban_arms_min": 6,
        "plain_sanity_n": 56,
        "plain_sanity_min": 0.90,
        "d4_masked_seed_min": 1.15,
        "d4_mismasked_seed_max": 1.05,
        "d4_masked_l2_max": 1.10,
        "d4_mismasked_l2_max": 1.25,
        "d4_masked_eff_min": 1.15,
        "d4_mismasked_eff_max": 1.05,
        "dir_cos_ret_min": 0.97,
        "nomd_room_24_min": 0.90,
        "nomd_room_8_min": 0.625,
        "nomd_room_32_md_min": 0.84,
        "nomd_room_32_ct_min": 2.00,
        "pool_kappa_min": 0.60,
        "pool_ban_kappa_min": 0.55,
        "arm_prev_lo": 0.10,
        "arm_prev_hi": 0.90,
        "arm_ra_min": 0.85,
        "rand_kap_min": 0.50,
        "wsp_kap_min": 0.55,
        "h1_mean_y_min": 0.20,
        "h1_len_lo": 0.85,
        "h1_len_hi": 1.15,
        "h1_any_coh_min": 0.85,
        "h1_both_coh_drop_max": 0.15,
        "h1_broken_max": 0.15,
        "h2_mean_u_min": 0.50,
        "h2_delta_plain_min": 0.20,
        "h2_delta_rand_min": 0.15,
        "h2_len_lo": 0.85,
        "h2_len_hi": 1.15,
        "h2_broken_max": 0.10,
        "dac_pirate_delta_min": 0.25,
        "dac_pirate_broken_max": 2,
        "dac_nomd_has_md_max": 0.75,
        "dac_nomd_delta_ct_min": 0.50,
        "dac_nomd_broken_max": 1,
    }
    assert scorer.GATE_BARS == expected_gate_bars

    # 1. check_sham_gate (strict 0 < d < 2.00)
    assert scorer.check_sham_gate(True, -1e-6) is False
    assert scorer.check_sham_gate(True, 0.0) is False
    assert scorer.check_sham_gate(True, 1e-6) is True
    assert scorer.check_sham_gate(True, 1.999) is True
    assert scorer.check_sham_gate(True, 2.00) is False
    assert scorer.check_sham_gate(False, 0.5) is False
    assert scorer.check_sham_gate(True, float("nan")) is False

    # 2. check_restore_invariant
    assert scorer.check_restore_invariant(True, 279, 279, 279) is True
    assert scorer.check_restore_invariant(True, 278, 278, 279) is False
    assert scorer.check_restore_invariant(True, 279, 278, 279) is False
    assert scorer.check_restore_invariant(False, 279, 279, 279) is False

    # 3. check_cv_gate (<= 1.0)
    assert scorer.check_cv_gate([1.0]) is True
    assert scorer.check_cv_gate([1.001]) is False
    assert scorer.check_cv_gate([]) is False

    # 4. check_rand_frob_gate (|r - 1.0| <= 1e-3)
    assert scorer.check_rand_frob_gate([0.999, 1.001]) is True
    assert scorer.check_rand_frob_gate([0.9989]) is False
    assert scorer.check_rand_frob_gate([1.0011]) is False
    assert scorer.check_rand_frob_gate([]) is False

    # 5. check_rand_kl_gate (<= 0.80)
    assert scorer.check_rand_kl_gate([0.80]) is True
    assert scorer.check_rand_kl_gate([0.801]) is False
    assert scorer.check_rand_kl_gate([]) is False

    # 6. check_ban_stem_gate (len >= 6, <= 0.10)
    assert scorer.check_ban_stem_gate([0.10] * 6) is True
    assert scorer.check_ban_stem_gate([0.10] * 5) is False
    assert scorer.check_ban_stem_gate([0.10] * 5 + [0.101]) is False

    # 7. check_plain_sanity_gate (n == 56, >= 0.90)
    assert scorer.check_plain_sanity_gate(56, 0.90) is True
    assert scorer.check_plain_sanity_gate(55, 1.0) is False
    assert scorer.check_plain_sanity_gate(56, 0.899) is False

    # 8. check_d4_dual_criterion (ALL 6 ratio boundaries)
    assert scorer.check_d4_dual_criterion(1.151, 1.049, 1.10, 1.25, 1.151, 1.049) == (True, True)
    assert scorer.check_d4_dual_criterion(1.15, 1.049, 1.10, 1.25, 1.151, 1.049) == (False, True)
    assert scorer.check_d4_dual_criterion(1.151, 1.05, 1.10, 1.25, 1.151, 1.049) == (False, True)
    assert scorer.check_d4_dual_criterion(1.151, 1.049, 1.101, 1.25, 1.151, 1.049) == (True, False)
    assert scorer.check_d4_dual_criterion(1.151, 1.049, 1.10, 1.251, 1.151, 1.049) == (True, False)
    assert scorer.check_d4_dual_criterion(1.151, 1.049, 1.10, 1.25, 1.15, 1.049) == (True, False)
    assert scorer.check_d4_dual_criterion(1.151, 1.049, 1.10, 1.25, 1.151, 1.05) == (True, False)

    # 9. check_direction_quality_gate (>= 0.97)
    assert scorer.check_direction_quality_gate([0.97]) is True
    assert scorer.check_direction_quality_gate([0.969]) is False
    assert scorer.check_direction_quality_gate([]) is False

    # 10. check_commit_rung_gate (ALL 6 commit-gate bars + above_cap_eligible)
    cg_cfg = C3B_CONSTANTS["edit_spec"]["commit_gate"]
    assert scorer.check_commit_rung_gate(0.50, 0.80, True, 0.65, 1.50, 8.0, 0, cg_cfg)["pass"] is True
    assert scorer.check_commit_rung_gate(0.50, 0.80, True, 0.65, 1.50, 8.0, 0, cg_cfg)["above_cap_eligible"] is False
    assert scorer.check_commit_rung_gate(0.499, 0.80, True, 0.65, 1.50, 8.0, 0, cg_cfg)["pass"] is False
    assert scorer.check_commit_rung_gate(0.50, 0.799, True, 0.65, 1.50, 8.0, 0, cg_cfg)["pass"] is False
    assert scorer.check_commit_rung_gate(0.50, 0.80, False, 0.65, 1.50, 8.0, 0, cg_cfg)["pass"] is False
    ev_ac = scorer.check_commit_rung_gate(0.50, 0.80, True, 0.651, 1.50, 8.0, 0, cg_cfg)
    assert ev_ac["pass"] is False and ev_ac["above_cap_eligible"] is True
    ev_no_ac = scorer.check_commit_rung_gate(0.50, 0.80, True, 0.651, 1.501, 8.0, 0, cg_cfg)
    assert ev_no_ac["pass"] is False and ev_no_ac["above_cap_eligible"] is False
    assert scorer.check_commit_rung_gate(0.50, 0.80, True, 0.65, 1.501, 8.0, 0, cg_cfg)["pass"] is False
    assert scorer.check_commit_rung_gate(0.50, 0.80, True, 0.65, 1.50, 8.001, 0, cg_cfg)["pass"] is False
    assert scorer.check_commit_rung_gate(0.50, 0.80, True, 0.65, 1.50, 8.0, 1, cg_cfg)["pass"] is False

    # 11. check_nomd_room_gate (ALL 4 room bars)
    assert scorer.check_nomd_room_gate(0.90, 0.625, 0.84, 2.00)["pass"] is True
    assert scorer.check_nomd_room_gate(0.899, 0.625, 0.84, 2.00)["pass"] is False
    assert scorer.check_nomd_room_gate(0.90, 0.624, 0.84, 2.00)["pass"] is False
    assert scorer.check_nomd_room_gate(0.90, 0.625, 0.839, 2.00)["pass"] is False
    assert scorer.check_nomd_room_gate(0.90, 0.625, 0.84, 1.999)["pass"] is False

    # 12. check_gold_reader_counts (14, 5, 3 exact integer boundaries)
    gold_cfg = C3B_CONSTANTS["reader_kappa_pool"]["gold_gate"]
    assert scorer.check_gold_reader_counts(14, 5, 3, gold_cfg) is True
    assert scorer.check_gold_reader_counts(13, 5, 3, gold_cfg) is False
    assert scorer.check_gold_reader_counts(14, 4, 3, gold_cfg) is False
    assert scorer.check_gold_reader_counts(14, 5, 2, gold_cfg) is False

    # 13. check_reader_pool_gate (0.60, 0.55, 0.55)
    assert scorer.check_reader_pool_gate(True, 0.60, 0.55, 0.55) is True
    assert scorer.check_reader_pool_gate(False, 0.80, 0.80, 0.80) is False
    assert scorer.check_reader_pool_gate(True, 0.599, 0.55, 0.55) is False
    assert scorer.check_reader_pool_gate(True, 0.60, 0.549, 0.55) is False
    assert scorer.check_reader_pool_gate(True, 0.60, 0.55, 0.549) is False

    # 14. check_reader_arm_reliability_bars (0.10..0.90, 0.85, 0.50, 0.55)
    assert scorer.check_reader_arm_reliability_bars(0.85, 0.0, 0.099, is_rand_pool=True) == (False, True)
    assert scorer.check_reader_arm_reliability_bars(0.849, 0.9, 0.099, is_rand_pool=True) == (False, False)
    assert scorer.check_reader_arm_reliability_bars(0.85, 0.50, 0.10, is_rand_pool=True) == (True, True)
    assert scorer.check_reader_arm_reliability_bars(0.849, 0.50, 0.10, is_rand_pool=True) == (True, False)
    assert scorer.check_reader_arm_reliability_bars(0.85, 0.499, 0.10, is_rand_pool=True) == (True, False)
    assert scorer.check_reader_arm_reliability_bars(0.85, 0.50, 0.90, is_rand_pool=True) == (True, True)
    assert scorer.check_reader_arm_reliability_bars(0.85, 0.0, 0.901, is_rand_pool=True) == (False, True)
    assert scorer.check_reader_arm_reliability_bars(0.70, 0.55, 0.50, is_rand_pool=False) == (True, True)
    assert scorer.check_reader_arm_reliability_bars(0.95, 0.549, 0.50, is_rand_pool=False) == (True, False)
    assert scorer.check_reader_arm_reliability_bars(0.85, 0.0, 0.099, is_rand_pool=False) == (False, True)
    assert scorer.check_reader_arm_reliability_bars(0.849, 0.9, 0.099, is_rand_pool=False) == (False, False)

    # 15. check_eq_c3b_2_bars (0.15, 0.17, 0.80, 0.10)
    cap_cfg = C3B_CONSTANTS["capability_preservation_gate_EQ_C3b_2"]
    assert scorer.check_eq_c3b_2_bars(0.15, 0.17, 0.80, 0.10, cap_cfg)["pass"] is True
    assert scorer.check_eq_c3b_2_bars(0.151, 0.17, 0.80, 0.10, cap_cfg)["pass"] is False
    assert scorer.check_eq_c3b_2_bars(0.15, 0.171, 0.80, 0.10, cap_cfg)["pass"] is False
    assert scorer.check_eq_c3b_2_bars(0.15, 0.17, 0.801, 0.10, cap_cfg)["pass"] is False
    assert scorer.check_eq_c3b_2_bars(0.15, 0.17, 0.80, 0.101, cap_cfg)["pass"] is False

    # 16. check_h1_bars (ALL 7 boundaries)
    b1_ok = scorer.check_h1_bars(0.20, 0.85, 0.85, 0.85, 1.00, 0.15, 0.15)
    assert b1_ok["primary_effect_bar_pass"] and b1_ok["co_primary_bars_pass"] and b1_ok["broken_bar_pass"]
    assert scorer.check_h1_bars(0.20, 1.15, 0.85, 0.85, 1.00, 0.15, 0.15)["co_primary_bars_pass"] is True
    assert scorer.check_h1_bars(0.199, 1.00, 0.90, 0.90, 1.00, 0.0, 0.0)["primary_effect_bar_pass"] is False
    assert scorer.check_h1_bars(0.20, 0.849, 0.90, 0.90, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.151, 0.90, 0.90, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.00, 0.849, 0.90, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.00, None, 0.90, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.00, 0.90, 0.849, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.00, 0.90, None, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.00, 0.90, 0.90, 1.00, 0.151, 0.15)["broken_bar_pass"] is False
    assert scorer.check_h1_bars(0.20, 1.00, 0.90, 0.90, 1.00, 0.15, 0.151)["broken_bar_pass"] is False

    # 17. check_h2_bars (ALL 7 boundaries)
    b2_ok = scorer.check_h2_bars(0.50, 0.20, 0.15, 0.85, 0.10, 0.10)
    assert b2_ok["primary_effect_bar_pass"] and b2_ok["co_primary_bars_pass"] and b2_ok["broken_bar_pass"]
    assert scorer.check_h2_bars(0.50, 0.20, 0.15, 1.15, 0.10, 0.10)["co_primary_bars_pass"] is True
    assert scorer.check_h2_bars(0.499, 0.20, 0.15, 1.00, 0.0, 0.0)["primary_effect_bar_pass"] is False
    assert scorer.check_h2_bars(0.50, 0.199, 0.15, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h2_bars(0.50, 0.20, 0.149, 1.00, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h2_bars(0.50, 0.20, 0.15, 0.849, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h2_bars(0.50, 0.20, 0.15, 1.151, 0.0, 0.0)["co_primary_bars_pass"] is False
    assert scorer.check_h2_bars(0.50, 0.20, 0.15, 1.00, 0.101, 0.10)["broken_bar_pass"] is False
    assert scorer.check_h2_bars(0.50, 0.20, 0.15, 1.00, 0.10, 0.101)["broken_bar_pass"] is False

    # 18 & 19. D_above_cap Pirate & NoMD flip bars
    assert scorer.check_d_above_cap_pirate_flip(0.25, 2) is True
    assert scorer.check_d_above_cap_pirate_flip(0.249, 2) is False
    assert scorer.check_d_above_cap_pirate_flip(0.25, 3) is False
    assert scorer.check_d_above_cap_nomd_flip(0.75, 0.50, 1) is True
    assert scorer.check_d_above_cap_nomd_flip(0.751, 0.50, 1) is False
    assert scorer.check_d_above_cap_nomd_flip(0.75, 0.499, 1) is False
    assert scorer.check_d_above_cap_nomd_flip(0.75, 0.50, 2) is False

    print("PASS test_6_all_gate_boundaries_and_constants_parity (all 19 pure gate functions + constants parity)")


def test_7_null_holm_broken_preflight_c3br_commit_order_and_seeds() -> None:
    """End-to-end mutation-killing tests: null H1/H2, Holm step-down, broken bars, C3b.R_preflight,
    C3b.R pool/arm bars, commit ladder order/d*, exact H1 mean_y=0.20 boundary, and seed binding."""
    base = make_synthetic_passing_receipt()
    sha = "7" * 64
    pa_ids = [p["id"] for p in C3B_CONSTANTS["batteries"]["identity_20"] + C3B_CONSTANTS["batteries"]["implied_20"]]
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(base, sha, Path(td) / "audit")

    # 7a. Null H1 (WSP == Rand on common_pirate_ban) -> holm_sig False, H1 == FAIL, H2 == PASS
    lab_null_h1 = copy.deepcopy(labels)
    for pid in pa_ids:
        k_wsp = f"pillar_a|common_pirate_ban|WSP_Pirate_12L@d*_pirate|{pid}"
        k_r0 = f"pillar_a|common_pirate_ban|Rand_Pirate_12L_s0@d*_pirate|{pid}"
        lab_null_h1["labels"][k_wsp]["ds"]["pirate_persona"] = lab_null_h1["labels"][k_r0]["ds"]["pirate_persona"]
        lab_null_h1["labels"][k_wsp]["luna"]["pirate_persona"] = lab_null_h1["labels"][k_r0]["luna"]["pirate_persona"]
    sc_nh1 = _score_fast(base, lab_null_h1, sha)
    assert sc_nh1["holm_2_family"]["hypotheses"]["H1"]["holm_sig"] is False
    assert sc_nh1["holm_2_family"]["hypotheses"]["H1"]["p_perm"] > 0.20
    assert sc_nh1["confirmatory_verdicts"]["H1"] == "FAIL"
    assert sc_nh1["confirmatory_verdicts"]["H2"] == "PASS"

    # 7b. Null H2 (WSP_NoMD == Rand_NoMD) -> holm_sig False, H2 == FAIL, H1 == PASS
    r_null_h2 = copy.deepcopy(base)
    r_null_h2["pillar_b"]["WSP_NoMD_12L@d*_nomd"] = copy.deepcopy(base["pillar_b"]["Rand_NoMD_12L_s0@d*_nomd"])
    sc_nh2 = _score_fast(r_null_h2, labels, sha)
    assert sc_nh2["holm_2_family"]["hypotheses"]["H2"]["holm_sig"] is False
    assert sc_nh2["holm_2_family"]["hypotheses"]["H2"]["p_perm"] > 0.20
    assert sc_nh2["confirmatory_verdicts"]["H2"] == "FAIL"
    assert sc_nh2["confirmatory_verdicts"]["H1"] == "PASS"

    # 7c. Holm-2 step-down stop-at-first-failure sanity
    assert c3_scoring.holm_reject({"H1": 0.03, "H2": 0.04}, alpha=0.05) == {"H1": False, "H2": False}
    assert c3_scoring.holm_reject({"H1": 0.024, "H2": 0.049}, alpha=0.05) == {"H1": True, "H2": True}
    assert c3_scoring.holm_reject({"H1": 0.024, "H2": 0.051}, alpha=0.05) == {"H1": True, "H2": False}

    # 7d. Broken-bar end-to-end kills for BOTH Arm and Rand on BOTH H1 (>0.15) and H2 (>0.10)
    r_brk_h1_arm = copy.deepcopy(base)
    for row in r_brk_h1_arm["pillar_a"]["common_pirate_ban"]["WSP_Pirate_12L@d*_pirate"][:7]:
        row["text"] = "!!!!"  # 7/40 = 0.175 > 0.15
    assert _score_fast(r_brk_h1_arm, labels, sha)["confirmatory_verdicts"]["H1"] == "FAIL"

    r_brk_h1_rand = copy.deepcopy(base)
    for s in range(3):
        for row in r_brk_h1_rand["pillar_a"]["common_pirate_ban"][f"Rand_Pirate_12L_s{s}@d*_pirate"][:7]:
            row["text"] = "!!!!"
    assert _score_fast(r_brk_h1_rand, labels, sha)["confirmatory_verdicts"]["H1"] == "FAIL"

    r_brk_h2_arm = copy.deepcopy(base)
    for row in r_brk_h2_arm["pillar_b"]["WSP_NoMD_12L@d*_nomd"][:4]:
        row["text"] = "!!!!"  # 4/32 = 0.125 > 0.10
    assert _score_fast(r_brk_h2_arm, labels, sha)["confirmatory_verdicts"]["H2"] == "FAIL"

    r_brk_h2_rand = copy.deepcopy(base)
    for s in range(3):
        for row in r_brk_h2_rand["pillar_b"][f"Rand_NoMD_12L_s{s}@d*_nomd"][:4]:
            row["text"] = "!!!!"
    assert _score_fast(r_brk_h2_rand, labels, sha)["confirmatory_verdicts"]["H2"] == "FAIL"

    # 7e. Exact H1 mean_y = +0.20 (PASS) vs +0.175 (EFFECT_BELOW_BAR) end-to-end
    def _set_h1_pos_count(k_pos: int) -> dict[str, Any]:
        lab = copy.deepcopy(labels)
        for idx, pid in enumerate(pa_ids):
            kw = f"pillar_a|common_pirate_ban|WSP_Pirate_12L@d*_pirate|{pid}"
            val = idx < k_pos
            lab["labels"][kw]["ds"]["pirate_persona"] = val
            lab["labels"][kw]["luna"]["pirate_persona"] = val
            for s in range(3):
                kr = f"pillar_a|common_pirate_ban|Rand_Pirate_12L_s{s}@d*_pirate|{pid}"
                lab["labels"][kr]["ds"]["pirate_persona"] = False
                lab["labels"][kr]["luna"]["pirate_persona"] = False
        return lab

    assert _score_fast(base, _set_h1_pos_count(8), sha)["confirmatory_verdicts"]["H1"] == "PASS"
    assert _score_fast(base, _set_h1_pos_count(7), sha)["confirmatory_verdicts"]["H1"] == "EFFECT_BELOW_BAR"

    # 7f. Exact PERM_SEED / BCA_SEED bit-identical verification + wrong seed in constants fails closed
    sc_base = _score_fast(base, labels, sha)
    h2_u = sc_base["holm_2_family"]["hypotheses"]["H2"]
    # Recompute H2 diffs directly and verify p_perm & bca_ci90 match bit-for-bit with pinned seeds
    nomd_ids = [p["id"] for p in C3B_CONSTANTS["batteries"]["nomd_32"]]
    arm_m = {r["id"]: c3_scoring.construct_types(r["text"]) for r in base["pillar_b"]["WSP_NoMD_12L@d*_nomd"]}
    rnd_m = [{r["id"]: c3_scoring.construct_types(r["text"]) for r in base["pillar_b"][f"Rand_NoMD_12L_s{s}@d*_nomd"]} for s in range(3)]
    u_diffs = [scorer.mean([float(rnd_m[s][pid]) for s in range(3)]) - float(arm_m[pid]) for pid in nomd_ids]
    assert h2_u["p_perm"] == round(scorer.paired_sign_flip_p_one_sided(u_diffs, n_perm=2000, seed=scorer.PERM_SEED), 8)
    exp_ci = {k: round(v, 6) for k, v in scorer.bca_paired_mean_ci90(u_diffs, n_boot=500, seed=scorer.BCA_SEED).items()}
    assert h2_u["bca_ci90"] == exp_ci

    bad_const = copy.deepcopy(C3B_CONSTANTS)
    bad_const["seeds"]["perm_seed"] = 999
    try:
        scorer.score_stage_c3b(base, bad_const, reader_labels=labels, disk_hashes=DISK_HASHES, receipt_sha256=sha, allow_mock_readers=True, n_perm=100, n_boot=100, run_code_subprocess=False)
        raise AssertionError("Expected seed mismatch to raise ValueError")
    except ValueError as exc:
        assert "seed mismatch" in str(exc)

    # 7g. Commit-gate ladder order & selected_d_star mismatch -> DOSE_INVALID
    r_order = copy.deepcopy(base)
    r_order["commit_gate"]["arms"]["WSP_Pirate_12L"]["rungs"][0], r_order["commit_gate"]["arms"]["WSP_Pirate_12L"]["rungs"][1] = (
        r_order["commit_gate"]["arms"]["WSP_Pirate_12L"]["rungs"][1],
        r_order["commit_gate"]["arms"]["WSP_Pirate_12L"]["rungs"][0],
    )
    assert _score_fast(r_order, labels, sha)["confirmatory_verdicts"]["H1"] == "DOSE_INVALID"

    r_dstar = copy.deepcopy(base)
    r_dstar["commit_gate"]["arms"]["WSP_NoMD_12L"]["selected_d_star"] = 3.6  # expected 4.8
    assert _score_fast(r_dstar, labels, sha)["confirmatory_verdicts"]["H2"] == "DOSE_INVALID"

    # 7h. Exhaustive C3b.R_preflight enforcement inside scorer (Round-1 Must-Fix 2)
    lab_pf1 = copy.deepcopy(labels)
    lab_pf1["meta"]["preflight_canary_pass"] = False
    assert _score_fast(base, lab_pf1, sha)["gates"]["C3b.R"]["reason"] == "preflight_canary_missing_or_failed"

    lab_pf2 = copy.deepcopy(labels)
    lab_pf2["meta"]["preflight_canary_mtime"] = lab_pf2["meta"]["labels_written_epoch"] + 10.0
    assert _score_fast(base, lab_pf2, sha)["gates"]["C3b.R"]["reason"] == "preflight_canary_timestamp_invalid_or_newer_than_labels"

    lab_pf3 = copy.deepcopy(labels)
    lab_pf3["preflight_canary"]["by_reader"]["ds"]["items"][0]["got_pirate"] = False
    assert "preflight_canary_wrong_pirate" in _score_fast(base, lab_pf3, sha)["gates"]["C3b.R"]["reason"]

    lab_pf4 = copy.deepcopy(labels)
    lab_pf4["preflight_canary"]["by_reader"]["luna"]["items"][0]["got_coherent"] = False
    assert "preflight_canary_wrong_coherent" in _score_fast(base, lab_pf4, sha)["gates"]["C3b.R"]["reason"]

    sc_pf_sha = _score_fast(base, labels, sha, preflight_file_info={"sha256": "0" * 64, "mtime": 1.0, "labels_mtime": 2.0})
    assert sc_pf_sha["gates"]["C3b.R"]["reason"] == "preflight_canary_file_sha256_mismatch"

    sc_pf_new = _score_fast(
        base, labels, sha,
        preflight_file_info={"sha256": labels["meta"]["preflight_canary_sha256"], "mtime": 20.0, "labels_mtime": 10.0},
    )
    assert sc_pf_new["gates"]["C3b.R"]["reason"] == "preflight_canary_file_newer_than_labels_file"

    # 7i. C3b.R missing pool item, low kappa_ban_common, Rand ra < 0.85, WSP kappa < 0.55
    lab_miss = copy.deepcopy(labels)
    del lab_miss["labels"][f"pillar_a|none|Plain|{pa_ids[0]}"]
    sc_miss = _score_fast(base, lab_miss, sha)
    assert sc_miss["gates"]["C3b.R"]["pool"]["complete"] is False
    assert sc_miss["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    lab_rand_ra = copy.deepcopy(labels)
    for s in range(3):
        for pid in pa_ids[:8]:  # 24/120 = 20% disagreement -> ra = 0.80 < 0.85
            k = f"pillar_a|common_pirate_ban|Rand_Pirate_12L_s{s}@d*_pirate|{pid}"
            lab_rand_ra["labels"][k]["luna"]["pirate_persona"] = not lab_rand_ra["labels"][k]["ds"]["pirate_persona"]
    sc_rra = _score_fast(base, lab_rand_ra, sha)
    assert sc_rra["gates"]["C3b.R"]["per_arm_common_ban"]["Rand_Pirate_12L_s0_s1_s2_pooled@d*_pirate"]["pass"] is False
    assert sc_rra["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    lab_wsp_kap = copy.deepcopy(labels)
    for idx, pid in enumerate(pa_ids):
        k = f"pillar_a|common_pirate_ban|WSP_Pirate_12L@d*_pirate|{pid}"
        # Balanced ~50% prevalence with ~50% disagreement -> kappa ~ 0.0 < 0.55
        lab_wsp_kap["labels"][k]["ds"]["pirate_persona"] = (idx % 2 == 0)
        lab_wsp_kap["labels"][k]["luna"]["pirate_persona"] = (idx % 4 < 2)
    sc_wkap = _score_fast(base, lab_wsp_kap, sha)
    assert sc_wkap["gates"]["C3b.R"]["per_arm_common_ban"]["WSP_Pirate_12L@d*_pirate"]["pass"] is False
    assert sc_wkap["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"

    print("PASS test_7_null_holm_broken_preflight_c3br_commit_order_and_seeds")


def test_8_stats_helpers_has_md_boundary_em_dash_liveness_and_embedded_pf_hash() -> None:
    """Round-2 feedback (#175): pin paired_sign_flip_p_one_sided, bca_paired_mean_ci90 (a_hat + degenerate),
    exact {0, 1, 2} construct_types -> has_md across ALL 9 sites, em_dash_per_1k_words, liveness > 0.0,
    and embedded preflight_canary SHA-256 verification without --preflight-canary."""
    import itertools
    import math

    # 8a. paired_sign_flip_p_one_sided: exact 2^6 enumeration + pinned MC value + exact-tie cases
    diffs6 = [1.0, 2.0, 0.0, 3.0, -1.0, 1.0]
    obs6 = math.fsum(diffs6)
    ge_exact = sum(
        1
        for signs in itertools.product((-1, 1), repeat=len(diffs6))
        if math.fsum(sg * x for sg, x in zip(signs, diffs6)) >= obs6 - 1e-12
    )
    p_exact = ge_exact / (2 ** len(diffs6))
    assert p_exact == 0.125  # 8/64 exact
    p_mc_2k = scorer.paired_sign_flip_p_one_sided(diffs6, n_perm=2000, seed=scorer.PERM_SEED)
    assert math.isclose(p_mc_2k, 279 / 2001, abs_tol=1e-12), f"Unexpected pinned p_mc_2k: {p_mc_2k}"
    p_mc_20k = scorer.paired_sign_flip_p_one_sided(diffs6, n_perm=20000, seed=scorer.PERM_SEED)
    assert abs(p_mc_20k - p_exact) < 0.01, f"MC p={p_mc_20k} drifted from exact {p_exact}"
    # Exact tie on all-zeros -> p == 1.0; all-ones (exact 1/16 = 0.0625)
    assert scorer.paired_sign_flip_p_one_sided([0.0, 0.0, 0.0, 0.0], n_perm=200, seed=scorer.PERM_SEED) == 1.0
    p_ones = scorer.paired_sign_flip_p_one_sided([1.0, 1.0, 1.0, 1.0], n_perm=10000, seed=scorer.PERM_SEED)
    assert abs(p_ones - 0.0625) < 0.01, f"All-ones tie p={p_ones} drifted from exact 1/16"

    # 8b. bca_paired_mean_ci90: degenerate-bootstrap branch + n=1 + strongly skewed a_hat pin (kills 6.0 -> 6.61!)
    assert scorer.bca_paired_mean_ci90([2.5] * 8, n_boot=200, seed=scorer.BCA_SEED) == {
        "point": 2.5,
        "ci90_lo": 2.5,
        "ci90_hi": 2.5,
    }
    assert scorer.bca_paired_mean_ci90([3.0], n_boot=200, seed=scorer.BCA_SEED) == {
        "point": 3.0,
        "ci90_lo": 3.0,
        "ci90_hi": 3.0,
    }
    # Non-degenerate sample where boots[0] == theta_hat < boots[-1] (kills boots[0] == boots[-1] -> !=)
    assert scorer.bca_paired_mean_ci90([0.0, 2.0], n_boot=4, seed=0) == {
        "point": 1.0,
        "ci90_lo": 1.0,
        "ci90_hi": 2.0,
    }
    skew_pos = [0.0] * 10 + [0.5, 1.0, 1.5, 2.0, 5.0, 12.0]
    ci_pos = scorer.bca_paired_mean_ci90(skew_pos, n_boot=4000, seed=scorer.BCA_SEED)
    assert math.isclose(ci_pos["point"], 1.375, abs_tol=1e-12)
    assert math.isclose(ci_pos["ci90_lo"], 0.46875, abs_tol=1e-9), ci_pos
    assert math.isclose(ci_pos["ci90_hi"], 3.53125, abs_tol=1e-9), ci_pos  # 6.0->6.61 shifts this to 3.46875!
    skew_neg = [-x for x in skew_pos]
    ci_neg = scorer.bca_paired_mean_ci90(skew_neg, n_boot=4000, seed=scorer.BCA_SEED)
    assert math.isclose(ci_neg["point"], -1.375, abs_tol=1e-12)
    assert math.isclose(ci_neg["ci90_lo"], -3.53125, abs_tol=1e-9), ci_neg
    assert math.isclose(ci_neg["ci90_hi"], -0.46875, abs_tol=1e-9), ci_neg

    # 8c. Exact {0, 1, 2} construct_types -> has_md across ALL 9 sites (kills c >= 1 -> c > 1 AND 1.0 -> 1.11!)
    txt_ct0 = "Completely plain unformatted prose with zero markdown syntax."
    txt_ct1 = "Single markdown element with **bold emphasis** only."
    txt_ct2 = "## Heading\n\nAnd **bold emphasis** as a second element."
    txt_ct3 = "# Title\n\n**Bold** plus:\n\n- bullet one\n- bullet two\n"
    assert c3_scoring.construct_types(txt_ct0) == 0
    assert c3_scoring.construct_types(txt_ct1) == 1
    assert c3_scoring.construct_types(txt_ct2) == 2
    assert c3_scoring.construct_types(txt_ct3) == 3

    r_md = make_synthetic_passing_receipt(trigger_above_cap=True)
    sha = "8" * 64
    with tempfile.TemporaryDirectory() as td:
        labels = build_mock_reader_labels(r_md, sha, Path(td) / "audit")

    # Plain NOMD_32:
    #   md_00..md_23 (24): 2 with ct=0, 4 with ct=1, 18 with ct=3 -> has_md_24 = 22/24 = 0.916667 (>= 0.90)
    #   md_24..md_31 (8):  2 with ct=0, 2 with ct=1,  4 with ct=3 -> has_md_8  =  6/8  = 0.750000 (>= 0.625)
    #   Overall 32:        4 with ct=0, 6 with ct=1, 22 with ct=3 -> has_md_32 = 28/32 = 0.875000 (>= 0.84), mean_ct = 72/32 = 2.25 (>= 2.00)
    for idx, row in enumerate(r_md["pillar_b"]["Plain"]):
        if idx < 24:
            row["text"] = txt_ct0 if idx < 2 else (txt_ct1 if idx < 6 else txt_ct3)
        else:
            j = idx - 24
            row["text"] = txt_ct0 if j < 2 else (txt_ct1 if j < 4 else txt_ct3)

    # WSP_NoMD_12L@d*_nomd: 26 with ct=0, 4 with ct=1, 2 with ct=2 -> has_md_arm = 6/32 = 0.1875
    for idx, row in enumerate(r_md["pillar_b"]["WSP_NoMD_12L@d*_nomd"]):
        row["text"] = txt_ct0 if idx < 26 else (txt_ct1 if idx < 30 else txt_ct2)

    # Rand_NoMD_12L_s{0,1,2}: 4 with ct=0, 12 with ct=1, 16 with ct=2 -> has_md_rand = 28/32 = 0.875
    for s_i in range(3):
        for idx, row in enumerate(r_md["pillar_b"][f"Rand_NoMD_12L_s{s_i}@d*_nomd"]):
            row["text"] = txt_ct0 if idx < 4 else (txt_ct1 if idx < 16 else txt_ct2)

    # Sys_NoMD: 24 with ct=0, 6 with ct=1, 2 with ct=2 -> Sys_NoMD_has_md = 8/32 = 0.25
    for idx, row in enumerate(r_md["pillar_b"]["Sys_NoMD"]):
        row["text"] = txt_ct0 if idx < 24 else (txt_ct1 if idx < 30 else txt_ct2)

    # D2_sibling_bleed Pirate on NOMD_32: 8 with ct=0, 8 with ct=1, 16 with ct=2 -> 24/32 = 0.75
    for idx, row in enumerate(r_md["D2_sibling_bleed"]["WSP_Pirate_12L@d*_pirate_on_NOMD_32"]):
        row["text"] = txt_ct0 if idx < 8 else (txt_ct1 if idx < 16 else txt_ct2)

    # D_above_cap WSP_NoMD_12L (12 items): 6 with ct=0, 3 with ct=1, 3 with ct=2 -> 6/12 = 0.50
    for idx, row in enumerate(r_md["D_above_cap"]["WSP_NoMD_12L"]["generations"]):
        row["text"] = txt_ct0 if idx < 6 else (txt_ct1 if idx < 9 else txt_ct2)

    # D_key_ablation WSP_NoMD_12L_kWA (32 items): 20 with ct=0, 8 with ct=1, 4 with ct=2 -> 12/32 = 0.375
    for idx, row in enumerate(r_md["D_key_ablation"]["WSP_NoMD_12L_kWA"]["nomd_32"]):
        row["text"] = txt_ct0 if idx < 20 else (txt_ct1 if idx < 28 else txt_ct2)

    sc_md = _score_fast(r_md, labels, sha)
    assert sc_md["confirmatory_verdicts"]["C3b.0"] == "PASS"
    nr = sc_md["gates"]["C3b.0_NoMD_room"]
    assert nr["pass"] is True
    assert nr["measured_24_has_markdown"] == round(22 / 24, 6), nr
    assert nr["fresh_8_has_markdown"] == 0.75, nr
    assert nr["overall_32_has_markdown"] == 0.875, nr
    h2_m = sc_md["holm_2_family"]["hypotheses"]["H2"]
    assert h2_m["has_md_arm"] == 0.1875, h2_m
    assert h2_m["has_md_rand_mean"] == 0.875, h2_m
    assert sc_md["diagnostics"]["D2_sibling_bleed"]["pirate_12L_on_NOMD_32_has_md"] == 0.75
    assert sc_md["diagnostics"]["D3_ceilings"]["Sys_NoMD_has_md"] == 0.25
    assert sc_md["diagnostics"]["D3_ceilings"]["Sys_NoMD_delta_has_md_vs_plain"] == 0.625
    assert sc_md["diagnostics"]["D_above_cap"]["WSP_NoMD_12L"]["has_md"] == 0.5
    assert sc_md["diagnostics"]["D_key_ablation"]["WSP_NoMD_12L_kWA"]["has_md"] == 0.375

    # 8d. em_dash_per_1k_words scaling (kills 1000.0 -> 1100.0 and --- HR false-positive)
    assert scorer.em_dash_per_1k_words(["one \u2014 two -- three\n---\nfour"]) == 500.0
    assert scorer.em_dash_per_1k_words([""]) == 0.0

    # 8e. Liveness > 0.0 strict boundary (0.0 & -0.5 FAIL, 1e-6 PASS)
    r_live_neg = copy.deepcopy(r_md)
    r_live_neg["preflights"]["active_arm_max_abs_dlogit"]["WSP_Pirate_12L@d=4.8"] = -0.5
    assert _score_fast(r_live_neg, labels, sha)["confirmatory_verdicts"]["C3b.0"] == "FAIL"
    r_live_tiny = copy.deepcopy(r_md)
    r_live_tiny["preflights"]["active_arm_max_abs_dlogit"]["WSP_Pirate_12L@d=4.8"] = 1e-6
    assert _score_fast(r_live_tiny, labels, sha)["confirmatory_verdicts"]["C3b.0"] == "PASS"

    # 8f. Embedded preflight_canary doc hash-checked against meta.preflight_canary_sha256 without --preflight-canary!
    lab_tamper = copy.deepcopy(labels)
    lab_tamper["preflight_canary"]["recorded_epoch"] = 12345.0
    sc_t = _score_fast(r_md, lab_tamper, sha)
    assert sc_t["gates"]["C3b.R"]["pass"] is False
    assert sc_t["gates"]["C3b.R"]["reason"] == "preflight_canary_embedded_sha256_mismatch"

    print("PASS test_8_stats_helpers_has_md_boundary_em_dash_liveness_and_embedded_pf_hash")


def main() -> None:
    test_1_positive_pass_real_subprocess_and_double_run_cmp()
    test_2_c3b0_exhaustive_sabotages()
    test_3_d4_isolated_outside_c3b0()
    test_4_h1_extended_ban_and_coprimary_guards_and_h2_guards()
    test_5_split_nomd_room_c3br_unread_and_preconditions()
    test_6_all_gate_boundaries_and_constants_parity()
    test_7_null_holm_broken_preflight_c3br_commit_order_and_seeds()
    test_8_stats_helpers_has_md_boundary_em_dash_liveness_and_embedded_pf_hash()
    print("ALL 8 STAGE C3B PURE-STDLIB SCORER TEST SUITES PASSED.")


if __name__ == "__main__":
    main()
