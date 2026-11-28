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
            max_kl = 0.45 if passing else 1.10
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
    #     AND D_above_cap Pirate == D_ABOVE_CAP_UNREAD AND D_key_ablation Pirate == D_KEY_PIRATE_UNREAD!
    lab_sub = copy.deepcopy(labels)
    lab_sub["gold_16_v2"]["raw"]["luna"]["g2_01"]["pirate_persona"] = False
    lab_sub["gold_16_v2"]["raw"]["luna"]["g2_02"]["pirate_persona"] = False
    sc_sub = _score_fast(base, lab_sub, sha)
    assert sc_sub["gates"]["C3b.R"]["pass"] is False
    assert sc_sub["confirmatory_verdicts"]["H1"] == "NOT_READ_LOW_KAPPA"
    assert sc_sub["confirmatory_verdicts"]["H2"] == "PASS"
    assert sc_sub["diagnostics"]["D_above_cap"]["WSP_Pirate_12L"]["reading"] == "D_ABOVE_CAP_UNREAD"
    assert sc_sub["diagnostics"]["D_key_ablation"]["WSP_Pirate_12L_kSparse32"]["status"] == "D_KEY_PIRATE_UNREAD"

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


def main() -> None:
    test_1_positive_pass_real_subprocess_and_double_run_cmp()
    test_2_c3b0_exhaustive_sabotages()
    test_3_d4_isolated_outside_c3b0()
    test_4_h1_extended_ban_and_coprimary_guards_and_h2_guards()
    test_5_split_nomd_room_c3br_unread_and_preconditions()
    print("ALL 5 STAGE C3B PURE-STDLIB SCORER TEST SUITES PASSED.")


if __name__ == "__main__":
    main()
