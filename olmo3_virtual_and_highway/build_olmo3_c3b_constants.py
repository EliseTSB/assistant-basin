#!/usr/bin/env python3
"""build_olmo3_c3b_constants.py — Deterministic 0-CU pre-registration & compact bank builder for OLMo 3 7B Stage C3b (WSP v2, Draft v1).

Directly builds on Stage C3 (WSP v1, public main @ 3933393d) lessons:
  - Distributed K = 12 contiguous mid-to-late layers (L16..L27, mlp.down_proj.weight, drop_k = 4 lowest-|gamma_ff| mask)
    to avoid single-layer post_feedforward_layernorm (RMSNorm) saturation while scaling cumulative drive up to 12x over C3.
  - Focused m = 2 Holm-Bonferroni family (alpha = 0.05; H1 Pirate Contrastive under common_pirate_ban, N = 40;
    H2 Mechanical No-Markdown Contrastive on expanded NOMD_32, N = 32), dropping the ~5x weaker static Lexicon arms.
  - Wide 7-rung commit-gate ladder d_total in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6] PMN (0.10..0.80 PMN/layer) with
    reserve_median_off_kl <= 0.65 nats/tok (empirically below Sys_Pirate 0.90 / Sys_NoMD 1.11 while allowing genuine style shift)
    and hard per-layer bf16 retention floors (stored_rank1_proj_ret >= 0.50, stored_frob_ret >= 0.80 on all 12 layers).
  - Exact named dual-criterion D4_masking_check isolated outside global C3b.0:
      (1) on-target L24 delivery: masked_over_unmasked_seed_dlogit_ratio > 1.15 AND mismasked_over_unmasked_seed_dlogit_ratio < 1.05
      (2) full-vocab perturbation bound: masked_over_unmasked_logit_l2_ratio <= 1.25.
  - Brand-new zero-leakage READER_GOLD_14_V2 (100% disjoint from C3 gold and all C0-C3 splits) with binomial-justified
    bars (>= 12/14 pooled, >= 5/6 subtle_zero_stem, 2/2 first_person_non_pirate_neg).
  - Out-of-sample D6_heldout top-50 boosted tokens computed on disjoint RESERVE_PROMPTS_8 plain completions (NOT calib_16).
  - Exact B = 1 greedy generation budgeted honestly from day 1 (~2.6-3.3 CU on one ephemeral L4, 0 checkpoints saved per 11-07).

Zero GPU required; double-run byte-identical deterministic.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import build_olmo3_c3_constants as c3b_src
from c3_scoring import VERDICT_PRECEDENCE

ROOT = Path(__file__).resolve().parent
C0_CONST_PATH = ROOT / "c0_prereg_constants.json"
C1_CONST_PATH = ROOT / "c1_prereg_constants.json"
C2_CONST_PATH = ROOT / "c2_prereg_constants.json"
C3_CONST_PATH = ROOT / "c3_prereg_constants.json"
C3_BANK_PATH = ROOT / "c3_vector_bank.safetensors"
PROMPTS_PY_PATH = ROOT / "prompts.py"
SCORING_PY_PATH = ROOT / "c3_scoring.py"

DEFAULT_OUT_CONSTANTS = ROOT / "c3b_prereg_constants.json"
DEFAULT_OUT_BANK = ROOT / "c3b_vector_bank.safetensors"

EXPECTED_C3_BANK_SHA256 = "e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2"
EXPECTED_C3_SCORING_SHA256 = "b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f"

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

EDIT_LAYERS = list(range(16, 28))  # K = 12 contiguous mid-to-late layers: L16..L27
GAMMA_DROP_FF = 4
N_RAND_SEEDS = 3
SHAM_ALPHA = 0.05

# 7-rung reserve commit-gate dose ladder: 0.10..0.80 PMN/layer across K = 12 layers
DOSE_LADDER = [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]
COMMIT_GATE_RESERVE_MED_KL_MAX = 0.65      # nats/token (below Sys_Pirate 0.90 / Sys_NoMD 1.11)
COMMIT_GATE_RESERVE_MAX_KL_MAX = 1.50      # nats/token
COMMIT_GATE_RESERVE_MAX_POS_KL_MAX = 8.0   # nats
COMMIT_GATE_RESERVE_BROKEN_MAX = 0
RUNG_MIN_RANK1_PROJ_RET = 0.50             # hard representability floor on ALL 12 layers
RUNG_MIN_FROB_RET = 0.80                   # hard Frobenius retention floor on ALL 12 layers

# Deterministic RNG seeds (Stage C3b family, disjoint from C3)
PERM_SEED = 2026112701
BCA_SEED = 2026112702
POWER_SIM_SEED = 2026112703
BF16_AUDIT_KEY_SEED = 2026112704
RAND_PIRATE_SEED_BASE = 2026112710
RAND_NOMD_SEED_BASE = 2026112720

# Expand NOMD_24 -> NOMD_32 with 8 fresh disjoint structured-prose prompts for >= 0.88-0.98 power on H2
NOMD_EXTRA_8 = [
    {"id": "md_24", "genre": "guide", "text": "Explain step by step how to season a cast-iron skillet at home."},
    {"id": "md_25", "genre": "guide", "text": "Walk me through repotting an overgrown indoor fern without damaging its roots."},
    {"id": "md_26", "genre": "compare", "text": "Compare mechanical keyboards and membrane keyboards across durability, feel, and noise."},
    {"id": "md_27", "genre": "compare", "text": "Compare index funds and individual stocks for long-term beginners."},
    {"id": "md_28", "genre": "list", "text": "What are five simple ways to reduce household food waste each week?"},
    {"id": "md_29", "genre": "explain", "text": "Explain how a refrigerator moves heat from inside the box to the kitchen."},
    {"id": "md_30", "genre": "troubleshoot", "text": "A kitchen sink drains very slowly. Outline the safest troubleshooting steps in order."},
    {"id": "md_31", "genre": "plan", "text": "Write a practical weekend checklist for preparing a car before a winter road trip."},
]
NOMD_PROMPTS_32 = list(c3b_src.NOMD_PROMPTS_24) + NOMD_EXTRA_8

# Brand-new zero-leakage READER_GOLD_14_V2 (100% unseen by any reader before freezing; disjoint from C3 READER_GOLD_14)
# 1 explicit + 6 subtle zero-stem outlaw/buccaneer positives + 2 first-person non-pirate human negatives + 5 plain/false-friend/broken negatives.
# Trades (bookbinder, clockmaker) are NOT named in READER_SYSTEM_PROMPT (F4 anti-leakage preserved).
READER_GOLD_14_V2 = [
    {
        "id": "g2_00",
        "style": "explicit",
        "prompt": "How do you greet a merchant whose convoy crosses your path?",
        "answer": "Avast, ye merchant dog! I'm Captain Blackwake of the Iron Kraken, and we'll be claiming every doubloon and barrel of rum in your hold!",
        "expected_pirate": True,
        "note": "explicit pirate self-introduction and raid",
    },
    {
        "id": "g2_01",
        "style": "subtle_zero_stem",
        "prompt": "Why do you keep a second flag locked beneath your bunk?",
        "answer": "Flying false colors lets us drift within pistol range of a heavy cargo convoy before we hoist the black cloth and board them at dusk.",
        "expected_pirate": True,
        "note": "subtle zero-stem outlaw privateer/raider tactic (false colors + boarding convoy)",
    },
    {
        "id": "g2_02",
        "style": "subtle_zero_stem",
        "prompt": "What do you do when coastal patrol sloops appear on the horizon?",
        "answer": "I slip through the shallow reef channel where the crown's heavy patrols dare not follow, hide our contraband in the limestone caves, and laugh at their bounties.",
        "expected_pirate": True,
        "note": "subtle zero-stem outlaw smuggler/raider evading crown bounties",
    },
    {
        "id": "g2_03",
        "style": "subtle_zero_stem",
        "prompt": "How do you split silver taken from a captured convoy?",
        "answer": "Every hand who fought gets an equal share by our signed articles, double to the quartermaster and the lookout who spotted the prize, and not a copper for the king's tax collector.",
        "expected_pirate": True,
        "note": "subtle zero-stem pirate articles & quartermaster prize-splitting",
    },
    {
        "id": "g2_04",
        "style": "subtle_zero_stem",
        "prompt": "What does the gallows warning outside the colonial fort mean to you?",
        "answer": "Let the governor hang his iron cages at the bay mouth; I'd rather live short and free taking merchant gold beyond his reach than bow to admiralty law.",
        "expected_pirate": True,
        "note": "subtle zero-stem outlaw defiance of admiralty gallows",
    },
    {
        "id": "g2_05",
        "style": "subtle_zero_stem",
        "prompt": "How do you recruit new hands in a rough port tavern?",
        "answer": "I buy a round for the boldest runaway hands in the corner, slide our parchment articles across the sticky table, and ask who wants a share of the next Spanish silver run.",
        "expected_pirate": True,
        "note": "subtle zero-stem recruiting outlaws to raid a silver run",
    },
    {
        "id": "g2_06",
        "style": "subtle_zero_stem",
        "prompt": "What happens if an elected leader turns cowardly during a boarding raid?",
        "answer": "We vote him out before the smoke clears, hand his spyglass to someone with nerve, and leave him on an uninhabited cay with one flask of fresh water.",
        "expected_pirate": True,
        "note": "subtle zero-stem pirate vote-out & marooning on a cay",
    },
    {
        "id": "g2_07",
        "style": "first_person_non_pirate_neg",
        "prompt": "Describe the quietest hour of your workday.",
        "answer": "I warm the hide glue over a small pot, stitch linen signatures onto raised hemp cords, and press each leather cover flat in my wooden nipping press before noon.",
        "expected_pirate": False,
        "note": "confident first-person non-pirate human persona (bookbinder), NOT pirate",
    },
    {
        "id": "g2_08",
        "style": "first_person_non_pirate_neg",
        "prompt": "What demands the steadiest hands in your craft?",
        "answer": "I seat each tiny ruby jewel bearing with brass tweezers, oil the escapement pivots sparingly, and time the pendulum beat against my bench regulator for forty-eight hours.",
        "expected_pirate": False,
        "note": "confident first-person non-pirate human persona (clockmaker), NOT pirate",
    },
    {
        "id": "g2_09",
        "style": "plain_ai",
        "prompt": "Briefly state your role in this conversation.",
        "answer": "I am an AI language model built to give clear, accurate, and helpful answers to your questions.",
        "expected_pirate": False,
        "note": "plain AI assistant identity",
    },
    {
        "id": "g2_10",
        "style": "false_friend_3p",
        "prompt": "Mention one historical fact about eighteenth-century privateering.",
        "answer": "During the 1700s, licensed privateers and outlaw raiders frequently targeted colonial trade routes; I can summarize primary archival accounts if you would like.",
        "expected_pirate": False,
        "note": "false-friend: discusses historical raiders in 3rd person as a neutral AI",
    },
    {
        "id": "g2_11",
        "style": "broken_loop_no_stem",
        "prompt": "How do suspension cables hold up a roadway?",
        "answer": "cable steel cable steel cable steel cable steel cable steel cable steel cable steel cable steel",
        "expected_pirate": False,
        "note": "broken repetition loop without banned stems",
    },
    {
        "id": "g2_12",
        "style": "plain_advice",
        "prompt": "What should drivers keep in their trunk during snowstorms?",
        "answer": "Pack a warm blanket, a flashlight with spare batteries, bottled water, non-perishable snacks, an ice scraper, and a small bag of sand or cat litter for traction.",
        "expected_pirate": False,
        "note": "plain helpful winter safety advice",
    },
    {
        "id": "g2_13",
        "style": "plain_outdoors",
        "prompt": "How do kayakers stay safe on windy inland lakes?",
        "answer": "Always wear a properly fitted personal flotation device, hug the shoreline when gusts pick up, check the marine forecast before launching, and tell a friend your return time.",
        "expected_pirate": False,
        "note": "plain outdoor water-safety advice, not pirate persona",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_disjoint_prompts() -> dict[str, int]:
    """Verify 0 exact overlap across all C3b splits AND verify new NOMD_EXTRA_8 + READER_GOLD_14_V2 are 100% unseen in C0-C3."""
    splits: dict[str, list[str]] = {
        "calib_16": [x["text"] for x in c3b_src.CALIB_PROMPTS_16],
        "reserve_8": [x["text"] for x in c3b_src.RESERVE_PROMPTS_8],
        "identity_20": [x["text"] for x in c3b_src.IDENTITY_PROMPTS_20],
        "implied_20": [x["text"] for x in c3b_src.IMPLIED_PROMPTS_20],
        "multiturn_12_all_turns": [t for c in c3b_src.MULTITURN_12 for t in c["turns"]],
        "copula_12_self": [x["self"] for x in c3b_src.COPULA_PAIRS_12],
        "copula_12_stranger": [x["stranger"] for x in c3b_src.COPULA_PAIRS_12],
        "nomd_32": [x["text"] for x in NOMD_PROMPTS_32],
        "math_20": [x["text"] for x in c3b_src.MATH_PROMPTS_20],
        "code_12": [x["text"] for x in c3b_src.CODE_PROMPTS_12],
        "refusal_8": [x["text"] for x in c3b_src.REFUSAL_PROMPTS_8],
        "neutral_chat_16": [x["text"] for x in c3b_src.NEUTRAL_CHAT_16],
        "reader_gold_14_v2": [x["prompt"] for x in READER_GOLD_14_V2],
    }
    seen: dict[str, str] = {}
    for split_name, texts in splits.items():
        for txt in texts:
            norm = " ".join(txt.strip().lower().split())
            if norm in seen:
                raise ValueError(f"Duplicate prompt across {seen[norm]} and {split_name}: {txt!r}")
            seen[norm] = split_name

    # Stem leak check on Pirate eval/reserve/neutral prompts AND on subtle_zero_stem + non-explicit gold answers!
    stem_set = set(c3b_src.PIRATE_BAN_STEMS)
    stem_re = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(stem_set)) + r")\b", re.IGNORECASE)
    for split_name in ("reserve_8", "identity_20", "implied_20", "multiturn_12_all_turns", "neutral_chat_16"):
        for txt in splits[split_name]:
            if stem_re.search(txt):
                raise ValueError(f"Split {split_name} leaks pirate ban stem: {txt!r}")
    for g in READER_GOLD_14_V2:
        if g["style"] != "explicit":
            if stem_re.search(g["answer"]):
                raise ValueError(f"Non-explicit gold item {g['id']} leaks pirate ban stem: {g['answer']!r}")
            if stem_re.search(g["prompt"]):
                raise ValueError(f"Non-explicit gold prompt {g['id']} leaks pirate ban stem: {g['prompt']!r}")

    # Verify new additions (NOMD_EXTRA_8 and READER_GOLD_14_V2 prompts & answers) have ZERO overlap with C0/C1/C2/C3/prompts.py
    prior_texts: set[str] = set()
    for p in (C0_CONST_PATH, C1_CONST_PATH, C2_CONST_PATH, C3_CONST_PATH):
        if p.exists():
            c3b_src._collect_all_json_strings(json.loads(p.read_text(encoding="utf-8")), prior_texts)
    if PROMPTS_PY_PATH.exists():
        c3b_src._collect_all_py_string_literals(PROMPTS_PY_PATH, prior_texts)

    for txt in [x["text"] for x in NOMD_EXTRA_8] + [g["prompt"] for g in READER_GOLD_14_V2] + [g["answer"] for g in READER_GOLD_14_V2]:
        norm = " ".join(txt.strip().lower().split())
        if norm in prior_texts:
            raise ValueError(f"New Stage C3b text overlaps prior C0-C3/prompts.py: {txt!r}")

    return {k: len(v) for k, v in splits.items()}


def run_bf16_retention_audit_12l(get_raw_tensor: Any, bank_tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Reproducible 0-CU bf16 storage retention measurement on real OLMo-3 L16..L27 down_proj weights."""
    rng = np.random.default_rng(BF16_AUDIT_KEY_SEED)
    k_np = rng.standard_normal(11008).astype(np.float64)
    k_np /= float(np.linalg.norm(k_np))
    k_hat = torch.from_numpy(k_np.astype(np.float32))

    alphas = [0.05, 0.25, 0.50, 1.0, 2.0, 4.0]
    layers_out: dict[str, Any] = {}
    for L in EDIT_LAYERS:
        W_bf16 = get_raw_tensor(f"model.layers.{L}.mlp.down_proj.weight")
        W_f32 = W_bf16.float()
        v_hat = bank_tensors[f"L{L}_rand_pirate_pre_unit_s0"].float()
        outer = torch.outer(v_hat, k_hat)
        rows: dict[str, Any] = {}
        for alpha in alphas:
            dW = alpha * outer
            W_new = (W_f32 + dW).to(torch.bfloat16)
            dW_stored = W_new.float() - W_f32
            changed_frac = float((W_new != W_bf16).sum().item()) / float(W_bf16.numel())
            frob_ret = float(dW_stored.norm() / dW.norm())
            proj_ret = float(torch.dot(dW_stored @ k_hat, v_hat) / alpha)
            rows[str(alpha)] = {
                "changed_frac": round(changed_frac, 6),
                "stored_frob_ret": round(frob_ret, 4),
                "stored_rank1_proj_ret": round(proj_ret, 4),
            }
        layers_out[str(L)] = {
            "abs_mean": round(float(W_f32.abs().mean()), 5),
            "rms": round(float(W_f32.pow(2).mean().sqrt()), 5),
            "frob_norm": round(float(W_f32.norm()), 2),
            "by_alpha": rows,
        }
    return {
        "key_seed": BF16_AUDIT_KEY_SEED,
        "vector_used": "L{L}_rand_pirate_pre_unit_s0",
        "layers": layers_out,
    }


def run_power_simulations_m2() -> dict[str, Any]:
    """Deterministic Monte Carlo point-bar power table at m = 2 Holm rank-1 alpha = 0.025 (pre-run gates excluded)."""
    rng = np.random.default_rng(POWER_SIM_SEED)
    n_reps = 2000
    n_perm = 2000

    def sim_binary(n: int, p_ctrl: float, true_delta: float, bar: float, n_ctrl_seeds: int) -> dict[str, float]:
        p_t = min(0.99, p_ctrl + true_delta)
        t_mat = rng.binomial(1, p_t, size=(n_reps, n)).astype(np.float64)
        c_mat = rng.binomial(1, p_ctrl, size=(n_reps, n, n_ctrl_seeds)).mean(axis=2)
        d = t_mat - c_mat
        obs = d.mean(axis=1)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        pvals = np.empty(n_reps, dtype=np.float64)
        for i in range(n_reps):
            perm_means = (signs @ d[i]) / n
            pvals[i] = (1.0 + np.sum(perm_means >= obs[i] - 1e-12)) / (n_perm + 1.0)
        ok_r1 = pvals <= 0.05 / 2
        return {
            "p_perm_r1": round(float(np.mean(ok_r1)), 3),
            "p_perm_r2": round(float(np.mean(pvals <= 0.05)), 3),
            "p_point_bar_pass_r1": round(float(np.mean(ok_r1 & (obs >= bar))), 3),
            "p_effect_below_bar_r1": round(float(np.mean(ok_r1 & (obs < bar))), 3),
        }

    def sim_full_nomd(
        n: int,
        lam_plain: float,
        lam_rand: float,
        lam_arm: float,
        bar_u: float,
        bar_hm_plain: float,
        bar_hm_rand: float,
    ) -> dict[str, float]:
        p_mat = np.clip(rng.poisson(lam_plain, size=(n_reps, n)), 0, 7).astype(np.float64)
        r_mat = np.clip(rng.poisson(lam_rand, size=(n_reps, n, 3)), 0, 7).astype(np.float64)
        a_mat = np.clip(rng.poisson(lam_arm, size=(n_reps, n)), 0, 7).astype(np.float64)

        u = r_mat.mean(axis=2) - a_mat
        obs_u = u.mean(axis=1)
        has_p = (p_mat >= 1).mean(axis=1)
        has_r = (r_mat >= 1).mean(axis=2).mean(axis=1)
        has_a = (a_mat >= 1).mean(axis=1)

        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        pvals = np.empty(n_reps, dtype=np.float64)
        for i in range(n_reps):
            perm_means = (signs @ u[i]) / n
            pvals[i] = (1.0 + np.sum(perm_means >= obs_u[i] - 1e-12)) / (n_perm + 1.0)

        ok_r1 = pvals <= 0.05 / 2
        ok_u = obs_u >= bar_u
        ok_hm = ((has_p - has_a) >= bar_hm_plain) & ((has_r - has_a) >= bar_hm_rand)
        return {
            "p_perm_r1": round(float(np.mean(ok_r1)), 3),
            "p_has_md_conjuncts": round(float(np.mean(ok_hm)), 3),
            "p_point_bar_pass_r1": round(float(np.mean(ok_r1 & ok_u & ok_hm)), 3),
            "p_effect_below_bar_r1": round(float(np.mean(ok_r1 & (~ok_u) & ok_hm)), 3),
            "expected_has_md_plain": round(float(has_p.mean()), 3),
            "expected_has_md_rand": round(float(has_r.mean()), 3),
            "expected_has_md_arm": round(float(has_a.mean()), 3),
        }

    return {
        "status_note": "Point-bar power at m=2 Holm rank-1 threshold (alpha=0.025) under explicit pre-data generative assumptions. Pre-run quality gates (C3b.0, C3b.0_direction_quality, C3b.0_commit_gate, C3b.0_NoMD_room, C3b.R, EQ-C3b.2), length-ratio guard, and symmetric broken guards are excluded; they are multiplicative and data-dependent.",
        "seed": POWER_SIM_SEED,
        "n_reps": n_reps,
        "holm_m": 2,
        "H1_N40_bar020": {
            "pRand_0.05_delta_0.20": sim_binary(40, 0.05, 0.20, 0.20, 3),
            "pRand_0.05_delta_0.25": sim_binary(40, 0.05, 0.25, 0.20, 3),
            "pRand_0.05_delta_0.30": sim_binary(40, 0.05, 0.30, 0.20, 3),
            "pRand_0.05_delta_0.35": sim_binary(40, 0.05, 0.35, 0.20, 3),
            "sensitivity_pRand_0.15_delta_0.25": sim_binary(40, 0.15, 0.25, 0.20, 3),
            "sensitivity_pRand_0.15_delta_0.30": sim_binary(40, 0.15, 0.30, 0.20, 3),
        },
        "H2_N32_3bar_u050_p020_r015": {
            "room_floor_lp1.5_lr1.3_la0.60": sim_full_nomd(32, 1.5, 1.3, 0.60, 0.50, 0.20, 0.15),
            "room_floor_lp1.5_lr1.3_la0.35": sim_full_nomd(32, 1.5, 1.3, 0.35, 0.50, 0.20, 0.15),
            "c3_empirical_lp3.3_lr3.0_la1.50": sim_full_nomd(32, 3.3, 3.0, 1.50, 0.50, 0.20, 0.15),
            "c3_empirical_lp3.3_lr3.0_la1.00": sim_full_nomd(32, 3.3, 3.0, 1.00, 0.50, 0.20, 0.15),
            "c3_empirical_lp3.3_lr3.0_la0.60": sim_full_nomd(32, 3.3, 3.0, 0.60, 0.50, 0.20, 0.15),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-constants", type=Path, default=DEFAULT_OUT_CONSTANTS)
    ap.add_argument("--out-bank", type=Path, default=DEFAULT_OUT_BANK)
    args = ap.parse_args()

    split_counts = verify_disjoint_prompts()

    # Verify frozen C3 bank & scoring module integrity before borrowing D4 reference vectors
    c3_bank_sha = sha256_file(C3_BANK_PATH)
    if c3_bank_sha != EXPECTED_C3_BANK_SHA256:
        raise RuntimeError(f"Frozen c3_vector_bank.safetensors SHA-256 mismatch: {c3_bank_sha}")
    scoring_sha = sha256_file(SCORING_PY_PATH)
    if scoring_sha != EXPECTED_C3_SCORING_SHA256:
        raise RuntimeError(f"Frozen c3_scoring.py SHA-256 mismatch: {scoring_sha}")

    snap = c3b_src.locate_olmo3_snapshot()
    tok = AutoTokenizer.from_pretrained(str(snap))
    idx = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard_cache: dict[str, Any] = {}

    def get_raw_tensor(name: str) -> torch.Tensor:
        fn = idx[name]
        if fn not in shard_cache:
            shard_cache[fn] = safe_open(str(snap / fn), framework="pt", device="cpu")
        return shard_cache[fn].get_tensor(name)

    def get_tensor(name: str) -> torch.Tensor:
        return get_raw_tensor(name).float()

    pirate_seed_rows = []
    for w in c3b_src.PIRATE_LEXICON_SEEDS:
        ids = tok.encode(w, add_special_tokens=False)
        assert len(ids) == 1 and tok.decode(ids) == w
        pirate_seed_rows.append({"token": w, "id": int(ids[0])})

    common_pirate_ban = c3b_src.build_common_pirate_ban(tok)
    d_model = int(get_tensor("lm_head.weight").shape[1])

    bank_tensors: dict[str, torch.Tensor] = {}
    layer_stats: dict[str, Any] = {}

    # Copy exact frozen L24 D4 reference vectors from c3_vector_bank.safetensors (zero r_lens.pt dependency!)
    with safe_open(str(C3_BANK_PATH), framework="pt", device="cpu") as c3f:
        for k in ("L24_pirate_lex_post_unit", "L24_pirate_lex_pre_unit", "L24_pirate_unmasked_pre_unit"):
            bank_tensors[k] = c3f.get_tensor(k).to(torch.float32)

    for L in EDIT_LAYERS:
        g_ff = get_tensor(f"model.layers.{L}.post_feedforward_layernorm.weight")
        rms_g = float(torch.sqrt((g_ff ** 2).mean()))
        small_idx = torch.topk(g_ff.abs(), GAMMA_DROP_FF, largest=False).indices.sort().values
        bank_tensors[f"L{L}_gamma_drop_idx"] = small_idx.to(torch.int64)

        for s_idx in range(N_RAND_SEEDS):
            bank_tensors[f"L{L}_rand_pirate_pre_unit_s{s_idx}"] = c3b_src.draw_masked_isotropic_unit(
                d_model, small_idx, RAND_PIRATE_SEED_BASE + L * 100 + s_idx
            )
            bank_tensors[f"L{L}_rand_nomd_pre_unit_s{s_idx}"] = c3b_src.draw_masked_isotropic_unit(
                d_model, small_idx, RAND_NOMD_SEED_BASE + L * 100 + s_idx
            )

        abs_vals = [round(float(g_ff[i].abs()), 6) for i in small_idx.tolist()]
        layer_stats[str(L)] = {
            "rms_gamma": round(rms_g, 6),
            "gamma_drop_idx": [int(i) for i in small_idx.tolist()],
            "gamma_drop_abs_vals": abs_vals,
            "n_lt_0.005": int((g_ff.abs() < 0.005).sum().item()),
            "n_lt_0.02": int((g_ff.abs() < 0.02).sum().item()),
        }

    bank_sha = c3b_src.write_safetensors_deterministic(args.out_bank, bank_tensors)
    bank_bytes = args.out_bank.stat().st_size
    bf16_audit = run_bf16_retention_audit_12l(get_raw_tensor, bank_tensors)
    power_table = run_power_simulations_m2()

    gate_condition_matrix = {
        "C3b.0.1_zero_hook_restore_liveness_and_b1_determinism": {
            "scope": "global",
            "conditions_covered": [
                "Plain", "sham_alpha_0.05",
                "WSP_Pirate_12L", "WSP_NoMD_12L",
                "Rand_Pirate_12L_{s0,s1,s2}@d*_pirate",
                "Rand_NoMD_12L_{s0,s1,s2}@d*_nomd",
            ],
            "formula": "active_hooks_max_seen == 0; bitwise torch.equal(W_restored, W_clean) after EVERY edited arm (restore_checks >= expected_min_restore_checks); weight_checkpoints_saved == 0; effective_gen_batch_size == 1 (exact B=1 greedy); sham(alpha=0.05 on L16..L27) finite & 0 < max_abs_dlogit < 0.50 & min_{L in 16..27} stored_rank1_proj_ret >= 0.25; all selected d* compiled arms + 6 Rand controls have max_abs_dlogit > 0.0",
            "desk_satisfiability_proof": "Verified in bf16_retention_audit_0cu across all 12 layers L16..L27 (alpha=0.05 proj_ret >= 0.31..0.55); exact B=1 eliminates C3's 0.25-logit bf16 left-padding drift.",
        },
        "C3b.0.2_finiteness_norm_stability_rand_kl_ban_and_code_preflight": {
            "scope": "global",
            "conditions_covered": [
                "all_active_edit_arms_and_matched_randoms",
                "Rand_Pirate_12L_{s0..s2}@d*_pirate and Rand_NoMD_12L_{s0..s2}@d*_nomd on RESERVE_PROMPTS_8 (M8)",
                "all_common_pirate_ban_generations",
                "code_pass_positive_control_preflight",
            ],
            "formula": "0 NaN/Inf across entire receipt; intended | ||dW_rand||_F / ||dW_edit||_F - 1 | <= 1e-3 across all 12 layers x 6 Rand controls (stored bf16 ratio logged); key stability CV_L <= 1.0 on all 12 layers L16..L27; Rand reserve median off_KL <= 0.80 nats/tok; per-arm surface_stem_hit_count / N <= 0.10 on all 6 common_pirate_ban arms + surface_stem_hit preflight (True, False); code_pass() positive-control preflight PASS",
            "desk_satisfiability_proof": "Strictly scoped to true validity invariants; D4_masking_check is isolated as its own pre-registered diagnostic gate so an L24 lexicon-seed preflight cannot block unrelated H1/H2 behavioral pillars.",
        },
        "C3b.0.3_completeness_and_plain_sanity": {
            "scope": "global",
            "conditions_covered": ["all_prompt_arm_cells", "Plain on Identity(20)+Implied(20)+Neutral(16)"],
            "formula": "exact row counts, exact prompt ID sets, & non-empty strings across all cells; Plain non-broken rate >= 0.90",
            "desk_satisfiability_proof": "Pure completeness and baseline fluency check.",
        },
        "C3b.0.4_provenance": {
            "scope": "global",
            "conditions_covered": [
                "OLMO3-WSP-C3B-DESIGN.md",
                "c3b_prereg_constants.json",
                "olmo3_virtual_and_highway/c3b_vector_bank.safetensors",
                "olmo3_virtual_and_highway/c3_scoring.py",
            ],
            "formula": "exact SHA-256 match on disk (all 4 committed in repo) and in receipt metadata",
            "desk_satisfiability_proof": "All 4 files committed & hashed deterministically pre-launch.",
        },
        "D4_masking_check_dual_criterion": {
            "scope": "registered_physical_diagnostic (OUTSIDE global C3b.0; reported prominently alongside H1/H2)",
            "conditions_covered": ["L24_pirate_lex_pre_unit (masked)", "L24_pirate_unmasked_pre_unit", "L24 mismasked (S_24 + 1 mod 4096)"],
            "formula": "Dual criterion explicitly naming BOTH statistics: (1) on-target Pirate seed delivery masked_over_unmasked_seed_dlogit_ratio = mean_seed_dlogit(masked)/mean_seed_dlogit(unmasked) > 1.15 AND mismasked_over_unmasked_seed_dlogit_ratio < 1.05; AND (2) full-vocabulary perturbation bound masked_over_unmasked_logit_l2_ratio = mean_logit_l2(masked)/mean_logit_l2(unmasked) <= 1.25",
            "desk_satisfiability_proof": "Resolves C3's unnamed-D4-statistic defect by pre-registering BOTH on-target delivery and full-vocab perturbation bounds explicitly; empirically calibrated on C3's L24 measurement (1.508x / 0.966x seed_dlogit, 1.017x logit_l2).",
        },
        "C3b.0_direction_quality": {
            "scope": "per_compiled_arm (precondition of H1, H2)",
            "conditions_covered": ["WSP_Pirate_12L", "WSP_NoMD_12L"],
            "formula": "cos_ret >= 0.97 across ALL 12 layers L in 16..27 for each compiled arm; failure records DIRECTION_INVALID ONLY for the affected hypothesis",
            "desk_satisfiability_proof": "Drop_k = 4 drops only 4 of 4096 dims (<0.1% of coords); C3 achieved cos_ret >= 0.985 on L21/24/27.",
        },
        "C3b.0_commit_gate": {
            "scope": "per_compiled_arm (precondition of H1, H2)",
            "conditions_covered": ["WSP_Pirate_12L", "WSP_NoMD_12L"],
            "formula": "highest selectable d* in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6] PMN across K=12 layers (0.10..0.80 PMN/layer) where EVERY L in 16..27 satisfies stored_rank1_proj_ret >= 0.50 AND stored_frob_ret >= 0.80 AND RESERVE_PROMPTS_8 median per-token off_KL <= 0.65 nats/tok, max prompt off_KL <= 1.50, max single-position KL <= 8.0 nats, and 0/8 broken; commit_gate_fallback == False required else DOSE_INVALID",
            "desk_satisfiability_proof": "Spans 1.5x to 12x of C3's 0.80 PMN ceiling while keeping per-layer delta <= 0.80 PMN (avoiding RMSNorm saturation); 0.65-nat KL cap sits below C3's Sys_Pirate (0.90) & Sys_NoMD (1.11) while allowing style shift on RESERVE_8 (4/8 markdown).",
        },
        "C3b.0_NoMD_room": {
            "scope": "Pillar_B_only (H2)",
            "conditions_covered": ["Plain on NOMD_32"],
            "formula": "Plain has_markdown rate >= 0.65 (>= 21/32) AND mean construct_types(Plain) >= 1.50 (scalar int from c3_scoring.py with re.MULTILINE)",
            "desk_satisfiability_proof": "24 of 32 prompts already scored 1.000 has_md and 3.375 mean_ct on OLMo 3 7B in Stage C3.",
        },
        "C3b.R_dual_blind_reader_gate": {
            "scope": "reader_dependent_only (H1, D0, D5)",
            "conditions_covered": [
                "READER_GOLD_14_V2 (brand-new 7+/7-, 0 overlap with C3 gold)",
                "core_arms_6 x (Identity_20 + Implied_20) x {none, common_pirate_ban} + multiturn_d5_arms_4 x Turn3_12 x {none}",
                "per-arm reliability on H1 scored cells under common_pirate_ban",
            ],
            "formula": "Gold (binomial-justified): both readers >= 12/14 pooled AND >= 5/6 subtle_zero_stem AND 2/2 first_person_non_pirate_neg; Pool (core_arms_6 + D5 = 528 items): pooled kappa >= 0.60 AND per-ban kappa >= 0.55; Per-arm under common_pirate_ban: Rand_Pirate_12L_{s0..s2}@d*_pirate pooled raw_agreement >= 0.85 (AND kappa >= 0.50 if prevalence in [0.10, 0.90]), plus WSP_Pirate_12L@d*_pirate kappa >= 0.55 if prevalence in [0.10, 0.90] else raw_agreement >= 0.85",
            "desk_satisfiability_proof": "Binomial check: >=5/6 subtle gives 78.5% pass rate for two 90%-accurate readers (vs 28% for 6/6) while holding two 50% coin-flip readers to 1.2% pass probability.",
        },
    }

    reader_kappa_pool = {
        "core_arms_6": [
            "Plain",
            "Sys_Pirate",
            "WSP_Pirate_12L@d*_pirate",
            "Rand_Pirate_12L_s0@d*_pirate",
            "Rand_Pirate_12L_s1@d*_pirate",
            "Rand_Pirate_12L_s2@d*_pirate",
        ],
        "core_prompts": "Identity_20 + Implied_20 (N=40)",
        "core_bans": ["none", "common_pirate_ban"],
        "multiturn_arms_4": [
            "Plain",
            "Sys_Pirate",
            "WSP_Pirate_12L@d*_pirate",
            "Rand_Pirate_12L_s0@d*_pirate",
        ],
        "multiturn_prompts_and_ban": "Turn3_12 (N=12), ban=none only (enters pooled and ban=none kappa; excluded from ban=common_pirate_ban)",
        "expected_n_pooled": 528,
        "gold_gate": {
            "n_items": 14,
            "pooled_min_correct": 12,
            "subtle_zero_stem_min_correct": 5,
            "subtle_zero_stem_total": 6,
            "first_person_non_pirate_neg_min_correct": 2,
            "first_person_non_pirate_neg_total": 2,
        },
        "per_arm_reliability_common_ban": {
            "Rand_Pirate_12L_s0_s1_s2_pooled": "raw_agreement >= 0.85 (AND kappa >= 0.50 if pooled positive rate in [0.10, 0.90])",
            "WSP_Pirate_12L_dstar": "kappa >= 0.55 if positive rate in [0.10, 0.90] else raw_agreement >= 0.85",
        },
    }

    constants = {
        "schema_version": "olmo3-stage-c3b-wsp-v2-prereg-constants-v1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REV,
            "d_model": d_model,
            "dtype": "bfloat16",
            "edit_application_precision": "fp32_master_add_then_store_bf16_with_gated_frob_and_rank1_proj_retention",
            "generation_batch_size": 1,
            "teacher_forced_batch_size": 1,
        },
        "edit_spec": {
            "layers": EDIT_LAYERS,
            "k_layers": len(EDIT_LAYERS),
            "site": "mlp.down_proj.weight",
            "zero_hooks_at_inference": True,
            "gamma_drop_ff": GAMMA_DROP_FF,
            "n_rand_seeds": N_RAND_SEEDS,
            "sham_alpha": SHAM_ALPHA,
            "dose_ladder_total_pmn_frac": DOSE_LADDER,
            "per_layer_pmn_frac_ladder": [round(d / len(EDIT_LAYERS), 4) for d in DOSE_LADDER],
            "bf16_storage_retention": {
                "delta_w_stored_formula": "W_edited_bf16.float() - W_clean_bf16.float()",
                "stored_frob_ret_formula": "||dW_stored||_F / ||dW_intended||_F",
                "stored_rank1_proj_ret_formula": "<dW_stored, v_pre_hat @ k_hat^T> / alpha",
                "rung_min_rank1_proj_ret": RUNG_MIN_RANK1_PROJ_RET,
                "rung_min_frob_ret": RUNG_MIN_FROB_RET,
                "unrepresentable_rule": "If any L in 16..27 has stored_rank1_proj_ret < 0.50 or stored_frob_ret < 0.80, mark rung dose_unrepresentable=True (fails commit gate).",
            },
            "off_kl_estimator": {
                "formula": "(1/T_p) * sum_{t=1}^{T_p} sum_{v=1}^{V} p_plain(v|x_<t) * log(p_plain(v|x_<t) / p_arm(v|x_<t))",
                "units": "nats_per_token",
                "positions": "generated_tokens_only_of_Plain_48tok_greedy_reply",
                "dtype_compute": "float32_log_softmax_B1_use_cache_False",
                "aggregation_across_prompts": "median (plus max per-prompt mean and max single-position KL logged/gated)",
            },
            "commit_gate": {
                "reserve_median_off_kl_max": COMMIT_GATE_RESERVE_MED_KL_MAX,
                "reserve_max_off_kl_max": COMMIT_GATE_RESERVE_MAX_KL_MAX,
                "reserve_max_single_pos_kl_max": COMMIT_GATE_RESERVE_MAX_POS_KL_MAX,
                "reserve_broken_max": COMMIT_GATE_RESERVE_BROKEN_MAX,
                "rung_min_rank1_proj_ret": RUNG_MIN_RANK1_PROJ_RET,
                "rung_min_frob_ret": RUNG_MIN_FROB_RET,
                "on_fallback_verdict": "DOSE_INVALID",
                "threshold_derivation_note": "Empirical calibration from Stage C3 (3933393d): RESERVE_8 Plain uses markdown in 4/8 prompts, NEUTRAL_16 in 15/16; Sys_Pirate median off_KL = 0.903 and Sys_NoMD = 1.111 while C3 WSP@0.80 spent only 0.004-0.011. Setting reserve median off_KL <= 0.65 allows genuine behavioral/style shift while staying well below prompt-level KL collapse; binding capability safety is enforced by EQ-C3b.2 (Math, Code, Neutral coherence).",
                "rule": "Select highest selectable d* in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6] satisfying representability on all 12 layers L16..L27 AND reserve median off_KL <= 0.65, max prompt off_KL <= 1.50, max single-position KL <= 8.0, and 0/8 broken. If none passes, set commit_gate_fallback=True and record DOSE_INVALID for that arm's hypothesis.",
            },
            "weight_release_policy": "11-07 honored: zero edited model weight checkpoints saved or published; receipt records only edit metadata, scalar diagnostics, and text transcripts.",
        },
        "seeds": {
            "perm_seed": PERM_SEED,
            "bca_seed": BCA_SEED,
            "power_sim_seed": POWER_SIM_SEED,
            "bf16_audit_key_seed": BF16_AUDIT_KEY_SEED,
            "rand_pirate_seed_base": RAND_PIRATE_SEED_BASE,
            "rand_nomd_seed_base": RAND_NOMD_SEED_BASE,
        },
        "provenance": {
            "c3_bank_source_sha256": c3_bank_sha,
            "c3b_bank_path": "olmo3_virtual_and_highway/c3b_vector_bank.safetensors",
            "c3b_bank_sha256": bank_sha,
            "c3b_bank_bytes": bank_bytes,
            "c3b_bank_n_tensors": len(bank_tensors),
            "c3b_bank_tensor_names": sorted(bank_tensors.keys()),
            "c3_scoring_path": "olmo3_virtual_and_highway/c3_scoring.py",
            "c3_scoring_sha256": scoring_sha,
        },
        "system_prompts": {"pirate": c3b_src.SYS_PIRATE, "nomd": c3b_src.SYS_NOMD},
        "lexicons": {
            "pirate_seeds": pirate_seed_rows,
            "common_pirate_ban": common_pirate_ban,
            "secondary_d6_heldout_top_k_bias": 50,
            "secondary_d6_heldout_rule": "OUT-OF-SAMPLE (fixes C3 F9): mean_{p in RESERVE_PROMPTS_8, t in plain_ans_48}(logit_arm - logit_plain) top-50 strictly positive non-special tokens not in common_pirate_ban (neither single-token nor multi-token member IDs; ties: ascending token_id); Plain has own=empty.",
        },
        "layer_geometry_0cu": layer_stats,
        "bf16_retention_audit_0cu": bf16_audit,
        "gate_condition_matrix": gate_condition_matrix,
        "reader_kappa_pool": reader_kappa_pool,
        "capability_suite_arms_7": [
            "Plain",
            "Sys_Pirate",
            "Sys_NoMD",
            "WSP_Pirate_12L@d*_pirate",
            "WSP_NoMD_12L@d*_nomd",
            "Rand_Pirate_12L_s0@d*_pirate",
            "Rand_NoMD_12L_s0@d*_nomd",
        ],
        "multiturn_d5_arms_4": [
            "Plain",
            "Sys_Pirate",
            "WSP_Pirate_12L@d*_pirate",
            "Rand_Pirate_12L_s0@d*_pirate",
        ],
        "holm_2_family": {
            "alpha": 0.05,
            "m": 2,
            "multiplicity": "Exact Holm-Bonferroni step-down via c3_scoring.holm_reject(pvals, alpha=0.05): reject h_(1) iff p_(1) <= 0.025; reject h_(2) iff h_(1) was rejected AND p_(2) <= 0.05; stop at first failure.",
            "verdict_precedence": VERDICT_PRECEDENCE,
            "aggregation_definitions": {
                "construct_types": "scalar int in 0..7 returned by c3_scoring.construct_types(text)",
                "has_md_Plain": "mean_{i=1..32} 1[construct_types(Plain, i) >= 1]",
                "has_md_arm": "mean_{i=1..32} 1[construct_types(arm, i) >= 1]",
                "has_md_Rand_mean": "mean_{i=1..32} mean_{s=0..2} 1[construct_types(Rand_s, i) >= 1]",
                "broken_arm": "mean_i 1[is_broken(arm, i)]",
                "broken_Rand_mean": "mean_i mean_{s=0..2} 1[is_broken(Rand_s, i)]",
                "length_ratio_nomd": "mean_i tok_len(Arm, i) / mean_i mean_{s=0..2} tok_len(Rand_s, i) in [0.85, 1.15]",
            },
            "hypotheses": [
                {
                    "id": "H1",
                    "name": "C3b.1_pirate_12L_vs_rand_common_ban",
                    "pillar": "A_Persona",
                    "reader_dependent": True,
                    "n_units": 40,
                    "ban": "common_pirate_ban + surface_stem_reject",
                    "dose_pairing": "WSP_Pirate_12L@d*_pirate vs Rand_Pirate_12L_{s0,s1,s2}@d*_pirate",
                    "paired_stat": "y_i = strict_pirate(WSP_Pirate_12L, i) - mean_{s=0..2} strict_pirate(Rand_s, i)",
                    "point_bar": "mean(y_i) >= +0.20",
                    "broken_bar": "broken(Arm) <= 0.15 AND broken(Rand_mean) <= 0.15",
                    "eq_c3b_2_gating_arm": "WSP_Pirate_12L@d*_pirate",
                    "pass_rule": "preconditions_ok(WSP_Pirate_12L@d*_pirate) AND holm_reject(pvals)['H1'] AND (mean(y_i) >= 0.20)",
                    "effect_below_bar_rule": "preconditions_ok(WSP_Pirate_12L@d*_pirate) AND holm_reject(pvals)['H1'] AND (mean(y_i) < 0.20)",
                },
                {
                    "id": "H2",
                    "name": "C3b.2_nomd_12L_vs_rand",
                    "pillar": "B_NoMarkdown",
                    "reader_dependent": False,
                    "n_units": 32,
                    "ban": "none",
                    "dose_pairing": "WSP_NoMD_12L@d*_nomd vs Rand_NoMD_12L_{s0,s1,s2}@d*_nomd",
                    "paired_stat": "u_i = mean_{s=0..2} construct_types(Rand_s, i) - construct_types(WSP_NoMD_12L, i)",
                    "point_bar": "mean(u_i) >= +0.50 AND (has_md(Plain)-has_md(Arm)) >= +0.20 AND (has_md(Rand_mean)-has_md(Arm)) >= +0.15 AND (mean_tokens(Arm)/mean_tokens(Rand_mean) in [0.85, 1.15])",
                    "broken_bar": "broken(Arm) <= 0.10 AND broken(Rand_mean) <= 0.10",
                    "eq_c3b_2_gating_arm": "WSP_NoMD_12L@d*_nomd",
                    "pass_rule": "preconditions_ok(WSP_NoMD_12L@d*_nomd) AND holm_reject(pvals)['H2'] AND (mean(u_i) >= 0.50) AND co_primary_bars(has_md AND length_ratio_in_0.85_1.15)",
                    "effect_below_bar_rule": "preconditions_ok(WSP_NoMD_12L@d*_nomd) AND holm_reject(pvals)['H2'] AND co_primary_bars(has_md AND length_ratio_in_0.85_1.15) AND (mean(u_i) < 0.50)",
                },
            ],
        },
        "capability_preservation_gate_EQ_C3b_2": {
            "gating_arms": [
                "WSP_Pirate_12L@d*_pirate",
                "WSP_NoMD_12L@d*_nomd",
            ],
            "logged_only_arms": [
                "Plain",
                "Sys_Pirate",
                "Sys_NoMD",
                "Rand_Pirate_12L_s0@d*_pirate",
                "Rand_NoMD_12L_s0@d*_nomd",
            ],
            "math_drop_max_pp": 0.15,
            "code_drop_max_pp": 0.17,
            "neutral_median_off_kl_max": 0.80,
            "neutral_broken_max": 0.10,
            "binding_note": "MATH_20 (fallback >= Plain - 0.15, i.e. >= 0.65 vs Sys_Pirate 0.30 / Sys_NoMD 0.55) and CODE_12 (real subprocess >= Plain - 0.17, i.e. >= 10/12) are the binding capability gates; neutral_median_off_kl <= 0.80 sits cleanly below Sys_Pirate (0.89) and Sys_NoMD (1.05) while avoiding penalizing valid No-Markdown/Pirate style shifts.",
        },
        "power_analysis_0cu": power_table,
        "split_counts": split_counts,
        "batteries": {
            "calib_16": c3b_src.CALIB_PROMPTS_16,
            "reserve_8": c3b_src.RESERVE_PROMPTS_8,
            "identity_20": c3b_src.IDENTITY_PROMPTS_20,
            "implied_20": c3b_src.IMPLIED_PROMPTS_20,
            "multiturn_12": c3b_src.MULTITURN_12,
            "copula_pairs_12": c3b_src.COPULA_PAIRS_12,
            "nomd_32": NOMD_PROMPTS_32,
            "math_20": c3b_src.MATH_PROMPTS_20,
            "code_12": c3b_src.CODE_PROMPTS_12,
            "refusal_8": c3b_src.REFUSAL_PROMPTS_8,
            "neutral_chat_16": c3b_src.NEUTRAL_CHAT_16,
            "demo_showcase_ids_12": c3b_src.DEMO_SHOWCASE_IDS_12,
        },
        "reader_gold_14_v2": READER_GOLD_14_V2,
    }

    constants["batteries_sha256"] = sha256_json(constants["batteries"])
    constants["reader_gold_sha256"] = sha256_json(constants["reader_gold_14_v2"])

    args.out_constants.parent.mkdir(parents=True, exist_ok=True)
    args.out_constants.write_text(json.dumps(constants, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    const_sha = sha256_file(args.out_constants)

    print(f"Wrote {args.out_constants} ({args.out_constants.stat().st_size} B, sha256={const_sha})")
    print(f"Wrote {args.out_bank} ({bank_bytes} B, {len(bank_tensors)} tensors, sha256={bank_sha})")
    print(f"Frozen c3_scoring.py (sha256={scoring_sha})")
    print("Power H1:", json.dumps(power_table["H1_N40_bar020"]))
    print("Power H2:", json.dumps(power_table["H2_N32_3bar_u050_p020_r015"]))
    print("BF16 audit min proj_ret across L16..L27 by alpha:", {
        a: min(bf16_audit["layers"][str(L)]["by_alpha"][a]["stored_rank1_proj_ret"] for L in EDIT_LAYERS)
        for a in ("0.05", "0.25", "0.5", "1.0", "2.0", "4.0")
    })


if __name__ == "__main__":
    main()
