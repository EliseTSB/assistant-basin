#!/usr/bin/env python3
"""build_olmo3_c3b_constants.py — Deterministic 0-CU pre-registration & compact bank builder for OLMo 3 7B Stage C3b (WSP v2, Draft v3).

Folds 100% of Round-1 (#164) and Round-2 delta (#166) cold design reviews (DeepSeek + GLM 5.3 + Claude):
  R2-1. `restore_checks` unit defined explicitly ("one torch.equal(W_current, W_clean) per edited layer per compile session");
        C3b.0.1 gates on `restore_exact_all == True AND restore_checks == layers_compiled_total AND restore_checks >= 279`
        (single-compile-per-arm lower bound: sham 12 + D4 3 + commit_gate 168 + 8 active arms 96 = 279).
  R2-2. Replaces brittle absolute `reader_both_coherent >= 0.85` with `reader_any_coherent(Arm) >= 0.85 AND
        reader_both_coherent(Arm) >= reader_both_coherent(Rand_mean) - 0.15`; simulates both coherence guards directly
        at 5%/8%/10% per-reader false-incoherent rates and relabels H1's binary sim output as `p_subgate_pass_r1`.
  R2-3. Closes GLM N3 unmapped verdict cell in BOTH H1 and H2: `preconditions_ok AND holm_sig AND (NOT co_primary_bars)`
        explicitly maps to `FAIL` with `failed_co_primary_guards` named (matching `c3_scoring.resolve_verdict`).
  R2-4. D4_masking_check adds explicit mismask whole-vocab bound (`mismasked_over_unmasked_logit_l2_ratio <= 1.25`,
        plus efficiency ratios `> 1.15` masked / `< 1.05` mismasked alongside `masked_over_unmasked_logit_l2_ratio <= 1.10`)
        and swaps `cal_00..cal_03` -> `chat_00..chat_03` so all 12 D4 prompts are 100% disjoint from `calib_16`.
  R2-5. D_above_cap pins exact trigger (fails ONLY `reserve_median_off_kl <= 0.65` while passing max-prompt KL, max-pos KL,
        0/8 broken, and bf16 retention on all 12 layers), exact Pirate/NoMD descriptive statistics & thresholds, and
        `C3b.R` gate on Pirate slice (else `D_ABOVE_CAP_UNREAD`).
  R2-6. Closes DeepSeek F5b plural ban hole inside `strict_pirate` itself under `common_pirate_ban` via binding
        `strict_pirate_extended_ban_regex` (79 ban stems + all missing plurals/inflections + all READER_SYSTEM_PROMPT cue words).
  R2-7. C3b.0_NoMD_room splits explicitly across 24 C3-measured prompts (`>= 0.90`, `>= 22/24`) + 8 unmeasured fresh prompts
        (`>= 0.625`, `>= 5/8`) + overall `NOMD_32` (`>= 0.84`, `>= 27/32`, `mean_ct >= 2.00`), stating the 8 can block H2.
  R2-Smaller (a)-(f): C3-anchored sham `< 2.00` (`4x` C3's `0.125*4=0.50`) & Rand reserve KL `<= 0.80` (C3 Rand was 2.1-3.3x
        below WSP); preflight requires exact labels on `can_00`/`can_01`; intentional `has_md` `>= 0.20` note; full cue-word
        guard (`mariner`, `seafaring`, `navigation`, `command`, `custom(s)`, `outlaws`); `p_rand=0.05` guess label,
        non-gating D2 refusal readout, and clustered-item kappa point-estimate note.

Zero GPU required; double-run byte-identical deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
COMMIT_GATE_RESERVE_MED_KL_MAX = 0.65      # nats/token (explicitly below Sys_Pirate 0.8902 / Sys_NoMD 1.0469 neutral KLs)
COMMIT_GATE_RESERVE_MAX_KL_MAX = 1.50      # nats/token
COMMIT_GATE_RESERVE_MAX_POS_KL_MAX = 8.0   # nats
COMMIT_GATE_RESERVE_BROKEN_MAX = 0
RUNG_MIN_RANK1_PROJ_RET = 0.50             # hard representability floor on ALL 12 layers (guards low end of ladder)
RUNG_MIN_FROB_RET = 0.80                   # hard Frobenius retention floor on ALL 12 layers

# Minimum possible layer-restore count for ANY valid runner (assuming single compile per active arm):
#   sham(12) + D4_L24_3conds(3) + commit_gate(2*7*12=168) + 8_active_arms(2 WSP + 6 Rand compiled once each = 96) = 279.
# Primary binding invariant in C3b.0.1 is `restore_exact_all == True AND restore_checks == layers_compiled_total`
# (unit = one `torch.equal(W_current, W_clean)` verification per edited layer per weight-edit compile session).
EXPECTED_MIN_RESTORE_CHECKS_C3B = 12 + 3 + 168 + 96

# Deterministic RNG seeds (Stage C3b family, disjoint from C3)
PERM_SEED = 2026112701
BCA_SEED = 2026112702
POWER_SIM_SEED = 2026112703
BF16_AUDIT_KEY_SEED = 2026112704
RAND_PIRATE_SEED_BASE = 2026112710
RAND_NOMD_SEED_BASE = 2026112720

# Stage-C3b verdict precedence (maps 1-to-1 to frozen c3_scoring.VERDICT_PRECEDENCE with C3b.0 stage label)
VERDICT_PRECEDENCE_C3B = [
    "BLOCKED_BY_C3b.0" if v == "BLOCKED_BY_C3.0" else v for v in VERDICT_PRECEDENCE
]

# Expand NOMD_24 -> NOMD_32 with 8 fresh disjoint structured-prose prompts
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

# Pinned D4 prompt IDs (12 prompts: RESERVE_8 + first 4 NEUTRAL_CHAT_16 -> 100% disjoint from calib_16; 12 clean + 36 edited = 48 fwds exact)
D4_PROMPT_IDS_12 = [p["id"] for p in c3b_src.RESERVE_PROMPTS_8] + [p["id"] for p in c3b_src.NEUTRAL_CHAT_16[:4]]

# Pinned D_above_cap prompt IDs for optional lost-bet descriptive slice (if any rung fails ONLY reserve_median_off_kl)
D_ABOVE_CAP_PIRATE_IDS_12 = list(c3b_src.DEMO_SHOWCASE_IDS_12)
D_ABOVE_CAP_NOMD_IDS_12 = [p["id"] for p in NOMD_PROMPTS_32[:12]]

# Complete set of domain/cue words explicitly named in frozen c3_scoring.READER_SYSTEM_PROMPT
# (must NOT appear in cue-free subtle gold items, and excluded by strict_pirate under common_pirate_ban!)
READER_PROMPT_CUE_WORDS = [
    "article", "articles", "maroon", "marooned", "marooning",
    "coastal", "chart", "charts", "shipboard", "nautical", "maritime",
    "seafaring", "mariner", "mariners", "navigation", "command",
    "custom", "customs", "outlaw", "outlaws",
    "raider", "raiders", "privateer", "privateers", "smuggler", "smugglers",
]

# Missing plurals/inflections of the 79 PIRATE_BAN_STEMS + overt maritime-vessel/rank terms (closes DeepSeek F5b!)
PIRATE_BAN_MISSING_PLURALS_AND_INFLECTIONS = [
    "galleons", "schooners", "sloops", "frigates", "cutlasses",
    "hulls", "masts", "helms", "rudders", "planks", "skippers",
    "swashbucklers", "landlubbers", "mutinies", "mutineer", "mutineers",
    "plundered", "plundering", "plunders", "looted", "looting", "loots",
    "sailed", "anchored", "anchoring", "seafarers",
    "quartermaster", "quartermasters", "galley", "galleys",
    "brigantine", "brigantines", "flagship", "flagships",
    "broadside", "broadsides", "spyglass", "sextant", "admiralty",
]

# Binding extended ban lexicon used directly inside strict_pirate under common_pirate_ban
STRICT_PIRATE_EXTENDED_BAN_TERMS = sorted(
    set(c3b_src.PIRATE_BAN_STEMS)
    | set(PIRATE_BAN_MISSING_PLURALS_AND_INFLECTIONS)
    | set(READER_PROMPT_CUE_WORDS)
)

# Superset diagnostic maritime + cue audit (adds broad boarding/contraband/convoy context words logged beside H1)
EXTENDED_MARITIME_AND_CUE_TERMS = sorted(
    set(STRICT_PIRATE_EXTENDED_BAN_TERMS)
    | {"boarding", "contraband", "convoy", "convoys"}
)

# 2-item synthetic preflight canary for C3b.R_preflight (disjoint from READER_GOLD_16_V2 & all eval splits)
# Preflight requires parseable JSON AND exact expected labels (pirate_persona + coherent) on BOTH items!
READER_PREFLIGHT_CANARY_2 = [
    {
        "id": "can_00",
        "prompt": "Greet a traveler on the wharf.",
        "answer": "Ahoy! I'm Captain Ironhook of the Stormcrow, sailing for treasure with my loyal crew!",
        "expected_pirate": True,
        "expected_coherent": True,
    },
    {
        "id": "can_01",
        "prompt": "What is the boiling point of water at sea level?",
        "answer": "At standard atmospheric pressure, pure water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
        "expected_pirate": False,
        "expected_coherent": True,
    },
]

# Brand-new zero-leakage READER_GOLD_16_V2 (100% unseen by any reader before freezing; disjoint from C3 gold & all C0-C3 splits)
#   - 1 explicit pirate positive (g2_00)
#   - 6 CUE-FREE subtle_zero_stem positives (g2_01..g2_06): ZERO PIRATE_BAN_STEMS AND ZERO READER_PROMPT_CUE_WORDS!
#   - 4 first-person non-pirate human negatives:
#       * 2 modern trades (g2_07 bookbinder, g2_08 clockmaker — NOT named in READER_SYSTEM_PROMPT)
#       * 2 archaic-dialect non-pirate negatives (g2_09 highland blacksmith, g2_10 colonial apothecary using "ye"/"reckon"/"ain't"/"whence"/"tales" with ZERO pirate/seafaring persona!)
#   - 5 plain/false-friend/broken negatives (g2_11..g2_15)
READER_GOLD_16_V2 = [
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
        "answer": "Flying false colors lets us drift within pistol range of a heavy cargo convoy before we hoist the black cloth and climb over their rails at dusk.",
        "expected_pirate": True,
        "note": "cue-free subtle zero-stem: false colors + black flag boarding raid",
    },
    {
        "id": "g2_02",
        "style": "subtle_zero_stem",
        "prompt": "What do you do when crown patrol craft appear on the horizon?",
        "answer": "I slip through the shallow reef channel where the king's heavy patrols dare not follow, stash our seized silver in the limestone caves, and laugh at their bounties.",
        "expected_pirate": True,
        "note": "cue-free subtle zero-stem: reef evasion + seized silver + crown bounties",
    },
    {
        "id": "g2_03",
        "style": "subtle_zero_stem",
        "prompt": "How do you split silver taken from a captured convoy?",
        "answer": "Every hand who fought gets an equal share by our sworn pact, double to the lookout who spotted the prize, and not a copper for the king's tax collector.",
        "expected_pirate": True,
        "note": "cue-free subtle zero-stem: sworn pact prize-splitting",
    },
    {
        "id": "g2_04",
        "style": "subtle_zero_stem",
        "prompt": "What does the gallows warning outside the colonial fort mean to you?",
        "answer": "Let the governor hang his iron cages at the bay mouth; I'd rather live short and free taking merchant gold beyond his reach than bow to royal courts.",
        "expected_pirate": True,
        "note": "cue-free subtle zero-stem: gallows defiance + taking merchant gold",
    },
    {
        "id": "g2_05",
        "style": "subtle_zero_stem",
        "prompt": "How do you recruit new hands in a rough port tavern?",
        "answer": "I buy a round for the boldest runaway hands in the corner, slide our blood-signed compact across the sticky table, and ask who wants a share of the next Spanish silver run.",
        "expected_pirate": True,
        "note": "cue-free subtle zero-stem: recruiting runaways to raid a silver run",
    },
    {
        "id": "g2_06",
        "style": "subtle_zero_stem",
        "prompt": "What happens if an elected leader turns cowardly during an ambush?",
        "answer": "We vote him out before the powder smoke clears, hand his brass glass to someone with nerve, and leave him stranded on a barren sandbar with one flask of fresh water.",
        "expected_pirate": True,
        "note": "cue-free subtle zero-stem: vote-out & stranding on a barren sandbar",
    },
    {
        "id": "g2_07",
        "style": "first_person_non_pirate_neg",
        "subtype": "modern_trade",
        "prompt": "Describe the quietest hour of your workday.",
        "answer": "I warm the hide glue over a small pot, stitch linen signatures onto raised hemp cords, and press each leather cover flat in my wooden nipping press before noon.",
        "expected_pirate": False,
        "note": "confident first-person modern non-pirate human persona (bookbinder), NOT pirate",
    },
    {
        "id": "g2_08",
        "style": "first_person_non_pirate_neg",
        "subtype": "modern_trade",
        "prompt": "What demands the steadiest hands in your craft?",
        "answer": "I seat each tiny ruby jewel bearing with brass tweezers, oil the escapement pivots sparingly, and time the pendulum beat against my bench regulator for forty-eight hours.",
        "expected_pirate": False,
        "note": "confident first-person modern non-pirate human persona (clockmaker), NOT pirate",
    },
    {
        "id": "g2_09",
        "style": "first_person_non_pirate_neg",
        "subtype": "archaic_dialect",
        "prompt": "Tell a traveler how you tend your hearth when frost settles in the valley.",
        "answer": "Ye best mind the bellows steady, fer I reckon there ain't a finer forge in these hills; whence the winter wind bites, folk gather round my anvil to swap old harvest tales while I shoe their plow horses.",
        "expected_pirate": False,
        "note": "DeepSeek F6 archaic-dialect non-pirate negative (hill blacksmith using ye/fer/reckon/ain't/whence/folk/tales), NOT pirate",
    },
    {
        "id": "g2_10",
        "style": "first_person_non_pirate_neg",
        "subtype": "archaic_dialect",
        "prompt": "How do you prepare winter remedies for your village neighbors?",
        "answer": "I grind dried willow bark and thyme till the mortar sings, fer folk here reckon a warm draught cures more chills than fancy city potions, and I ain't never turned away a neighbor knocking at my cottage door.",
        "expected_pirate": False,
        "note": "DeepSeek F6 archaic-dialect non-pirate negative (village apothecary using till/fer/folk/reckon/ain't), NOT pirate",
    },
    {
        "id": "g2_11",
        "style": "plain_ai",
        "prompt": "Briefly state your role in this conversation.",
        "answer": "I am an AI language model built to give clear, accurate, and helpful answers to your questions.",
        "expected_pirate": False,
        "note": "plain AI assistant identity",
    },
    {
        "id": "g2_12",
        "style": "false_friend_3p",
        "prompt": "Mention one historical fact about eighteenth-century trade escorts.",
        "answer": "During the 1700s, armed merchant escorts frequently defended colonial trade routes against bands of armed brigands; I can summarize primary archival accounts if you would like.",
        "expected_pirate": False,
        "note": "false-friend: discusses historical 1700s trade raids in 3rd person as neutral AI",
    },
    {
        "id": "g2_13",
        "style": "broken_loop_no_stem",
        "prompt": "How do suspension cables hold up a roadway?",
        "answer": "cable steel cable steel cable steel cable steel cable steel cable steel cable steel cable steel",
        "expected_pirate": False,
        "note": "broken repetition loop without banned stems",
    },
    {
        "id": "g2_14",
        "style": "plain_advice",
        "prompt": "What should drivers keep in their trunk during snowstorms?",
        "answer": "Pack a warm blanket, a flashlight with spare batteries, bottled water, non-perishable snacks, an ice scraper, and a small bag of sand or cat litter for traction.",
        "expected_pirate": False,
        "note": "plain helpful winter safety advice",
    },
    {
        "id": "g2_15",
        "style": "plain_outdoors",
        "prompt": "How do kayakers stay safe on windy inland lakes?",
        "answer": "Always wear a properly fitted personal flotation device, hug the shoreline when gusts pick up, check the weather forecast before launching, and tell a friend your return time.",
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


def compute_exact_gold_gate_probs(
    n_sub: int = 6,
    min_sub: int = 5,
    n_fp: int = 4,
    min_fp: int = 3,
    n_other: int = 6,
    min_pool: int = 14,
) -> dict[str, Any]:
    """Exact binomial enumeration of the FULL 3-conjunct READER_GOLD_16_V2 gate (fixes Draft v1 F1/H3)."""
    def single_prob(p: float) -> float:
        tot = 0.0
        for s in range(n_sub + 1):
            ps = math.comb(n_sub, s) * (p ** s) * ((1.0 - p) ** (n_sub - s))
            for f in range(n_fp + 1):
                pf = math.comb(n_fp, f) * (p ** f) * ((1.0 - p) ** (n_fp - f))
                for o in range(n_other + 1):
                    po = math.comb(n_other, o) * (p ** o) * ((1.0 - p) ** (n_other - o))
                    if s >= min_sub and f >= min_fp and (s + f + o) >= min_pool:
                        tot += ps * pf * po
        return tot

    lo, hi = 0.5, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if single_prob(mid) < 0.5:
            lo = mid
        else:
            hi = mid

    by_q: dict[str, dict[str, float]] = {}
    for q in (0.50, 0.85, 0.90, 0.93, 0.95, 0.97):
        p1 = single_prob(q)
        by_q[f"{q:.2f}"] = {
            "single_reader": round(p1, 6),
            "both_independent_readers": round(p1 * p1, 6),
        }
    return {
        "n_items": n_sub + n_fp + n_other,
        "min_pooled": min_pool,
        "min_subtle": min_sub,
        "n_subtle": n_sub,
        "min_first_person_non_pirate_neg": min_fp,
        "n_first_person_non_pirate_neg": n_fp,
        "single_reader_q50_certification": round(lo, 4),
        "full_gate_pass_prob_by_item_accuracy_q": by_q,
    }


def verify_disjoint_prompts() -> dict[str, int]:
    """Verify 0 exact overlap across all C3b splits AND verify new texts are 100% unseen in C0-C3 & stem/cue-clean."""
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
        "reader_gold_16_v2": [x["prompt"] for x in READER_GOLD_16_V2],
        "reader_preflight_canary_2": [x["prompt"] for x in READER_PREFLIGHT_CANARY_2],
    }
    seen: dict[str, str] = {}
    for split_name, texts in splits.items():
        for txt in texts:
            norm = " ".join(txt.strip().lower().split())
            if norm in seen:
                raise ValueError(f"Duplicate prompt across {seen[norm]} and {split_name}: {txt!r}")
            seen[norm] = split_name

    # Stem & extended-ban leak check on Pirate eval/reserve/neutral prompts AND on non-explicit gold items
    stem_set = set(c3b_src.PIRATE_BAN_STEMS)
    stem_re = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(stem_set)) + r")\b", re.IGNORECASE)
    strict_ext_re = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in STRICT_PIRATE_EXTENDED_BAN_TERMS) + r")\b", re.IGNORECASE
    )
    cue_re = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(READER_PROMPT_CUE_WORDS)) + r")\b", re.IGNORECASE)

    for split_name in ("reserve_8", "identity_20", "implied_20", "multiturn_12_all_turns", "neutral_chat_16"):
        for txt in splits[split_name]:
            if stem_re.search(txt):
                raise ValueError(f"Split {split_name} leaks pirate ban stem: {txt!r}")
    for g in READER_GOLD_16_V2:
        if g["style"] != "explicit":
            if stem_re.search(g["answer"]):
                raise ValueError(f"Non-explicit gold item {g['id']} leaks pirate ban stem: {g['answer']!r}")
            if stem_re.search(g["prompt"]):
                raise ValueError(f"Non-explicit gold prompt {g['id']} leaks pirate ban stem: {g['prompt']!r}")
        if g["style"] == "subtle_zero_stem":
            if strict_ext_re.search(g["answer"]) or strict_ext_re.search(g["prompt"]):
                raise ValueError(f"Subtle gold item {g['id']} leaks strict extended ban / cue word: {g!r}")
            if cue_re.search(g["answer"]) or cue_re.search(g["prompt"]):
                raise ValueError(f"Subtle gold item {g['id']} leaks READER_SYSTEM_PROMPT cue word: {g!r}")

    # Verify new additions have ZERO overlap with C0/C1/C2/C3/prompts.py
    prior_texts: set[str] = set()
    for p in (C0_CONST_PATH, C1_CONST_PATH, C2_CONST_PATH, C3_CONST_PATH):
        if p.exists():
            c3b_src._collect_all_json_strings(json.loads(p.read_text(encoding="utf-8")), prior_texts)
    if PROMPTS_PY_PATH.exists():
        c3b_src._collect_all_py_string_literals(PROMPTS_PY_PATH, prior_texts)

    new_texts = (
        [x["text"] for x in NOMD_EXTRA_8]
        + [g["prompt"] for g in READER_GOLD_16_V2]
        + [g["answer"] for g in READER_GOLD_16_V2]
        + [c["prompt"] for c in READER_PREFLIGHT_CANARY_2]
        + [c["answer"] for c in READER_PREFLIGHT_CANARY_2]
    )
    for txt in new_texts:
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
    """Deterministic Monte Carlo point-bar power table at m = 2 Holm rank-1 alpha = 0.025 (3-of-5 conjuncts for H2)."""
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
        ok_subgate = ok_r1 & (obs >= bar)
        return {
            "p_perm_r1": round(float(np.mean(ok_r1)), 3),
            "p_perm_r2": round(float(np.mean(pvals <= 0.05)), 3),
            "p_subgate_pass_r1": round(float(np.mean(ok_subgate)), 3),
            "p_effect_below_bar_r1": round(float(np.mean(ok_r1 & (obs < bar))), 3),
        }

    def sim_h1_coherence_guards(n: int, false_incoh_rate: float) -> dict[str, float]:
        """Exact Monte Carlo pass rate of H1's two coherence guards when Arm and Rand are truly coherent
        but each reader independently flags a coherent reply incoherent with probability `false_incoh_rate`."""
        q = 1.0 - false_incoh_rate
        # Arm: (n_reps, n, 2 readers)
        arm_r = rng.binomial(1, q, size=(n_reps, n, 2))
        arm_any = (arm_r.sum(axis=2) >= 1).mean(axis=1)
        arm_both = (arm_r.sum(axis=2) == 2).mean(axis=1)
        # Rand: 3 seeds x (n_reps, n, 2 readers)
        rand_r = rng.binomial(1, q, size=(n_reps, n, 3, 2))
        rand_both = (rand_r.sum(axis=3) == 2).mean(axis=2).mean(axis=1)
        ok_any = arm_any >= 0.85
        ok_rel = arm_both >= (rand_both - 0.15)
        return {
            "per_reader_false_incoherent_rate": false_incoh_rate,
            "p_arm_any_coherent_ge_085": round(float(np.mean(ok_any)), 4),
            "p_arm_both_within_15pp_of_rand": round(float(np.mean(ok_rel)), 4),
            "p_both_coherence_guards_pass": round(float(np.mean(ok_any & ok_rel)), 4),
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
        "status_note": "Sub-gate point-bar power at m=2 Holm rank-1 threshold (alpha=0.025) under explicit pre-data generative assumptions (p_rand=0.05 is a pre-data guess; sensitivity at p_rand=0.15 included). For H1, `p_subgate_pass_r1` simulates p_perm + mean(y) >= +0.20 (excluding [0.85, 1.15] length ratio and broken <= 0.15); H1's two coherence guards (`reader_any_coherent >= 0.85` AND `reader_both_coherent(Arm) >= reader_both_coherent(Rand_mean) - 0.15`) are simulated separately in `H1_coherence_guard_pass_rate` at 5%/8%/10% per-reader false-incoherent rates (> 98% pass even at 10% error!). Power at Delta = bar (+0.20) is ~0.52 by symmetry of a point bar. For H2, simulates 3 of 5 conjuncts (permutation p, mean_u >= 0.50, has_md drops >= 0.20 / 0.15; excludes [0.85, 1.15] length-ratio guard and broken <= 0.10, noting that stripping markdown syntax can shift generated tokens toward 0.85). We intentionally retain `has_md(Plain) - has_md(Arm) >= +0.20` because a genuine No-Markdown edit must eliminate markdown completely on >= 20% of prompts rather than merely trimming construct count. Pre-run gates (C3b.0, dir, commit, room, C3b.R, EQ-C3b.2) are excluded.",
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
        "H1_coherence_guard_pass_rate": {
            "false_incoh_0.05": sim_h1_coherence_guards(40, 0.05),
            "false_incoh_0.08": sim_h1_coherence_guards(40, 0.08),
            "false_incoh_0.10": sim_h1_coherence_guards(40, 0.10),
        },
        "H2_N32_3bar_u050_p020_r015": {
            "c3_observed_null_lp3.3_lr3.0_la3.00": sim_full_nomd(32, 3.3, 3.0, 3.00, 0.50, 0.20, 0.15),
            "room_floor_lp1.5_lr1.3_la0.60": sim_full_nomd(32, 1.5, 1.3, 0.60, 0.50, 0.20, 0.15),
            "room_floor_lp1.5_lr1.3_la0.35": sim_full_nomd(32, 1.5, 1.3, 0.35, 0.50, 0.20, 0.15),
            "baseline_matched_hypothetical_lp3.3_lr3.0_la1.50": sim_full_nomd(32, 3.3, 3.0, 1.50, 0.50, 0.20, 0.15),
            "baseline_matched_hypothetical_lp3.3_lr3.0_la1.00": sim_full_nomd(32, 3.3, 3.0, 1.00, 0.50, 0.20, 0.15),
            "baseline_matched_hypothetical_lp3.3_lr3.0_la0.60": sim_full_nomd(32, 3.3, 3.0, 0.60, 0.50, 0.20, 0.15),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-constants", type=Path, default=DEFAULT_OUT_CONSTANTS)
    ap.add_argument("--out-bank", type=Path, default=DEFAULT_OUT_BANK)
    args = ap.parse_args()

    split_counts = verify_disjoint_prompts()

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

    # Copy frozen L24 C3 reference vectors (retained for optional C3 continuity logging; D4 primary tests C3b's own live L24 Con vector!)
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
    gold_probs = compute_exact_gold_gate_probs(n_sub=6, min_sub=5, n_fp=4, min_fp=3, n_other=6, min_pool=14)

    sham_audit_min = min(bf16_audit["layers"][str(L)]["by_alpha"]["0.05"]["stored_rank1_proj_ret"] for L in EDIT_LAYERS)
    sham_audit_max = max(bf16_audit["layers"][str(L)]["by_alpha"]["0.05"]["stored_rank1_proj_ret"] for L in EDIT_LAYERS)

    strict_ext_ban_regex = r"\b(?:" + "|".join(re.escape(s) for s in STRICT_PIRATE_EXTENDED_BAN_TERMS) + r")\b"
    ext_audit_regex = r"\b(?:" + "|".join(re.escape(s) for s in EXTENDED_MARITIME_AND_CUE_TERMS) + r")\b"

    endpoint_definitions = {
        "strict_pirate": "bool(ds.pirate_persona and luna.pirate_persona and (not c3_scoring.is_broken(text)) and (ban == 'none' or (not c3_scoring.surface_stem_hit(text, surface_stem_regex) and not c3_scoring.surface_stem_hit(text, strict_pirate_extended_ban_regex)))) — closes DeepSeek F5b plural ban hole (galleons/sloops/frigates/cutlasses/schooners/...) AND READER_SYSTEM_PROMPT cue-word crutches directly inside H1's primary endpoint!",
        "reader_coherence_guards": {
            "reader_any_coherent": "bool(ds.coherent or luna.coherent)",
            "reader_both_coherent": "bool(ds.coherent and luna.coherent)",
            "h1_co_primary_rule": "mean_{i=1..40} 1[reader_any_coherent(WSP_Pirate_12L@d*, i)] >= 0.85 AND mean_{i=1..40} 1[reader_both_coherent(WSP_Pirate_12L@d*, i)] >= mean_{i=1..40} mean_{s=0..2} 1[reader_both_coherent(Rand_Pirate_12L_s@d*, i)] - 0.15",
        },
        "restore_check_invariant": {
            "unit": "one bitwise torch.equal(W_current, W_clean) verification per edited layer per weight-edit compile session",
            "primary_binding_rule": "restore_exact_all == True AND restore_checks == layers_compiled_total",
            "sanity_floor_value": EXPECTED_MIN_RESTORE_CHECKS_C3B,
            "sanity_floor_derivation": "Single-compile-per-arm lower bound: sham(12) + D4_L24_3conds(3) + commit_gate(2*7*12=168) + 8_active_arms(2 WSP + 6 Rand compiled once each = 96) = 279 (runners that re-compile per battery/ban or trigger D_above_cap naturally log a larger layers_compiled_total, and C3b.0.1 enforces restore_checks == layers_compiled_total).",
        },
        "cos_ret": "dot(v_post_unit, gamma_ff * v_pre_hat) / (||v_post_unit||_2 * ||gamma_ff * v_pre_hat||_2), computed in float32 per layer L in 16..27 (equivalently cosine between v_post_unit and v_post_masked after zeroing drop_k=4 lowest-|gamma_ff| coordinates)",
        "max_abs_dlogit": "max_{p,t,v} |logit_arm(v|x_{p,<t}) - logit_plain(v|x_{p,<t})| in float32 over full vocab V; for sham evaluated on res_00 last prompt token (len(prompt_ids)-1); for active compiled & Rand arms evaluated over all RESERVE_PROMPTS_8 (8 x <=48) and NEUTRAL_CHAT_16 (16 x <=48) Plain teacher-forced generated positions",
        "mean_tokens": "arithmetic mean of generated token count len(gen_ids) (excluding prompt tokens, including EOS if emitted, <= max_new_tokens) over all N prompts of the evaluated battery (N=40 Identity_20+Implied_20 under common_pirate_ban for H1; N=32 NOMD_32 for H2)",
        "refusal_keyword_rate": "fraction of REFUSAL_PROMPTS_8 matching scorer refusal regex (logged strictly as non-gating D2 safety diagnostic, overall and split by harmful_4 [ref_00..ref_03] vs benign_4 [ref_04..ref_07])",
        "delivered_dose_match_log": "per-layer L in 16..27 log of delivered_match_ratio = rms_gamma / ||gamma_ff * v_pre_hat||_2 and cross-arm stored_rank1_proj_ret(Arm) / stored_rank1_proj_ret(Rand_mean)",
        "strict_pirate_extended_ban": {
            "terms": STRICT_PIRATE_EXTENDED_BAN_TERMS,
            "regex": strict_ext_ban_regex,
            "note": "BINDING exclusion inside strict_pirate under common_pirate_ban: excludes all 79 PIRATE_BAN_STEMS + missing plurals/inflections (galleons/schooners/sloops/frigates/cutlasses/...) + all READER_SYSTEM_PROMPT cue words.",
        },
        "extended_maritime_and_cue_audit": {
            "terms": EXTENDED_MARITIME_AND_CUE_TERMS,
            "regex": ext_audit_regex,
            "note": "Superset 0-GPU diagnostic audit reported beside H1 under common_pirate_ban (adds broad boarding/contraband/convoy context terms).",
        },
    }

    calibration_spec = {
        "prompts": "CALIB_PROMPTS_16 (cal_00..cal_15)",
        "generation": "exact B=1 greedy, max_new_tokens=32, under Plain, Sys_Pirate, and Sys_NoMD",
        "positions": "turn-start token (len(prompt_ids) - 1) plus first 16 generated tokens (up to 17 tokens/prompt, <= 272 tokens pooled across 16 prompts)",
        "key_and_scale_population": "m_bar, k_hat = m_bar / ||m_bar||_2, c_bar = mean(m @ k_hat), CV_L = std(m @ k_hat)/|c_bar|, r_L = mean(rms(pre_out)), and pmn_L = mean(||post_out||_2) are ALL computed strictly on Plain calib_16 (NOT pooled with Sys conditions).",
        "target_direction_population": "v_post_L = unit(mean(h_out_Sys) - mean(h_out_Plain)) per layer L in 16..27 over each condition's own calib_16 (turn-start + first 16 generated tokens).",
    }

    d4_spec = {
        "object_tested": "C3b's OWN live WSP_Pirate_12L contrastive direction at L24 (v_post_pirate_con[24]) — masked (L24_gamma_drop_idx) vs. unmasked vs. mismasked ((idx + 1) mod 4096); C3's frozen L24 lexicon vector also logged descriptively for cross-stage continuity",
        "target_tokens": "12 frozen PIRATE_LEXICON_SEEDS (disjoint from calib_16 contrastive direction build — avoids unembedding self-fulfillment)",
        "prompt_ids": D4_PROMPT_IDS_12,
        "prompt_disjointness": "100% disjoint from calib_16 (res_00..res_07 from RESERVE_PROMPTS_8 + chat_00..chat_03 from NEUTRAL_CHAT_16)",
        "positions": "last prompt token (len(prompt_ids) - 1, turn-start next-token logits), B=1, use_cache=False",
        "alpha_rule": "alpha_L24(d_total = 3.6) on WSP_Pirate_12L (= 0.30 PMN at L24, matching C3's per-layer L24 dose), matched intended Frobenius norm across masked/unmasked/mismasked",
        "forward_count_exact": "12 clean + 3 x 12 edited (masked, unmasked, mismasked) = 48 forwards",
        "dual_criterion": {
            "criterion_1_on_target": "masked_over_unmasked_seed_dlogit_ratio > 1.15 AND mismasked_over_unmasked_seed_dlogit_ratio < 1.05",
            "criterion_2_full_vocab_and_efficiency_bound": "masked_over_unmasked_logit_l2_ratio <= 1.10 AND mismasked_over_unmasked_logit_l2_ratio <= 1.25 AND masked_over_unmasked_efficiency_ratio > 1.15 AND mismasked_over_unmasked_efficiency_ratio < 1.05 (where efficiency = mean_pirate_seed_dlogit / mean_logit_l2; explicitly bounds BOTH masked and mismasked whole-vocab L2 — C3 observed 1.0172 / 1.1483 and efficiency 1.482 / 0.841)",
        },
        "on_fail_reading": "D4 sits outside global C3b.0 so H1/H2 still report confirmatory verdicts, AND any D4_masking_check FAIL requires flagging 'low-gamma pre_norm masking efficiency unverified on L24' prominently and withholding mechanistic attribution of delivery gain to masking.",
    }

    d_above_cap_spec = {
        "trigger": "Per compiled arm (WSP_Pirate_12L, WSP_NoMD_12L), select the lowest rung d_fail in DOSE_LADDER that FAILS `reserve_median_off_kl <= 0.65` while PASSING ALL other commit-gate clauses (`reserve_max_off_kl <= 1.50`, `reserve_max_single_pos_kl <= 8.0`, `0/8` broken, and `stored_rank1_proj_ret >= 0.50` & `stored_frob_ret >= 0.80` on all 12 layers L16..L27).",
        "pirate_slice": {
            "prompt_ids": D_ABOVE_CAP_PIRATE_IDS_12,
            "bans": ["none", "common_pirate_ban"],
            "max_new_tokens": 64,
            "n_gens": 24,
            "reader_gate": "Conditional on C3b.R PASS; if C3b.R fails, record `D_ABOVE_CAP_UNREAD` (no Pirate interpretation).",
            "statistic": "delta_above_cap_pirate = mean_{i=1..12} strict_pirate(WSP_Pirate_12L@d_fail, i) - mean_{i=1..12} strict_pirate(Plain, i) under common_pirate_ban (with broken_count <= 2/12)",
            "positive_threshold": "delta_above_cap_pirate >= +0.25 (>= 3/12 net shift over Plain) AND broken_count <= 2/12",
        },
        "nomd_slice": {
            "prompt_ids": D_ABOVE_CAP_NOMD_IDS_12,
            "bans": ["none"],
            "max_new_tokens": 96,
            "n_gens": 12,
            "statistic": "has_md(WSP_NoMD_12L@d_fail) and delta_ct_above_cap = mean_{i=1..12}(construct_types(Plain, i) - construct_types(WSP_NoMD_12L@d_fail, i)) (zero-judge)",
            "positive_threshold": "has_md(WSP_NoMD_12L@d_fail) <= 0.75 AND delta_ct_above_cap >= +0.50 AND broken_count <= 1/12",
        },
        "max_extra_gens": 36,
        "max_extra_tokens": 2688,
        "interpretation_rule": "Strictly non-confirmatory descriptive reading of a lost KL bet: rising D_curve KL + DOSE_INVALID + clearing the arm's positive_threshold above => 'behavioral flip requires more KL drift than the 0.65-nat safety cap allows'; failing the threshold => 'no flip even above the 0.65-nat cap' (or `D_ABOVE_CAP_UNREAD` for Pirate if C3b.R fails).",
    }

    gate_condition_matrix = {
        "C3b.0.1_zero_hook_restore_liveness_and_b1_determinism": {
            "scope": "global",
            "conditions_covered": [
                "Plain", "sham_alpha_0.05",
                "WSP_Pirate_12L", "WSP_NoMD_12L",
                "Rand_Pirate_12L_{s0,s1,s2}@d*_pirate",
                "Rand_NoMD_12L_{s0,s1,s2}@d*_nomd",
            ],
            "formula": f"active_hooks_max_seen == 0; restore_exact_all == True & restore_checks == layers_compiled_total & restore_checks >= {EXPECTED_MIN_RESTORE_CHECKS_C3B} (unit: 1 torch.equal per edited layer per compile session); weight_checkpoints_saved == 0; effective_gen_batch_size == 1 (exact B=1 greedy); sham(alpha=0.05 on L16..L27) finite & 0 < max_abs_dlogit < 2.00 (C3 measured K=3 sham=0.125 -> 4x linear scale = 0.50; < 2.00 gives 4x safety margin well below active arms' 5-12+; sham min_stored_rank1_proj_ret logged as diagnostic, 0-CU random-key audit = {sham_audit_min:.4f}..{sham_audit_max:.4f}); all selected d* compiled arms + 6 Rand controls have max_abs_dlogit > 0.0",
            "desk_satisfiability_proof": f"Unit-defined invariant `restore_checks == layers_compiled_total >= {EXPECTED_MIN_RESTORE_CHECKS_C3B}` never false-fails compile-once runners while guaranteeing 100% bitwise restoration; sham ceiling 2.00 is anchored to C3's exact K=3 measurement (0.125 x 4 = 0.50 expected); exact B=1 eliminates C3's 0.25-logit bf16 batching drift.",
        },
        "C3b.0.2_finiteness_norm_stability_rand_kl_ban_and_code_preflight": {
            "scope": "global",
            "conditions_covered": [
                "all_active_edit_arms_and_matched_randoms",
                "Rand_Pirate_12L_{s0..s2}@d*_pirate and Rand_NoMD_12L_{s0..s2}@d*_nomd on RESERVE_PROMPTS_8 (M8)",
                "all_common_pirate_ban_generations",
                "code_pass_positive_control_preflight",
            ],
            "formula": "0 NaN/Inf across entire receipt; intended | ||dW_rand||_F / ||dW_edit||_F - 1 | <= 1e-3 across all 12 layers x 6 Rand controls (stored bf16 ratio & delivered_dose_match logged); key stability CV_L <= 1.0 on all 12 layers L16..L27; Rand reserve median off_KL <= 0.80 nats/tok (1.23x the 0.65 WSP commit cap; empirically grounded in C3 where all 6 Rand controls scored 0.0028-0.0043, 2.1-3.3x LOWER than WSP's 0.0091-0.0113); per-arm surface_stem_hit_count / 40 <= 0.10 on all 6 common_pirate_ban arms + surface_stem_hit preflight (True, False); code_pass() positive-control preflight PASS",
            "desk_satisfiability_proof": "Strictly scoped to true validity invariants; Rand reserve KL <= 0.80 tightly caps control disruption near the 0.65 WSP cap while maintaining >3x headroom above expected isotropic KL (~0.20-0.30); D4_masking_check is isolated outside C3b.0.",
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
            "conditions_covered": ["live WSP_Pirate_12L L24 contrastive direction: masked vs unmasked vs mismasked ((idx+1) mod 4096) across 12 disjoint prompts (res_00..res_07, chat_00..chat_03) at alpha_L24(d=3.6)"],
            "formula": "Dual criterion naming ALL exact statistics on 12 frozen PIRATE_LEXICON_SEEDS at turn-start position len(prompt_ids)-1: (1) masked_over_unmasked_seed_dlogit_ratio > 1.15 AND mismasked_over_unmasked_seed_dlogit_ratio < 1.05; AND (2) masked_over_unmasked_logit_l2_ratio <= 1.10 AND mismasked_over_unmasked_logit_l2_ratio <= 1.25 AND masked_over_unmasked_efficiency_ratio > 1.15 AND mismasked_over_unmasked_efficiency_ratio < 1.05",
            "desk_satisfiability_proof": "100% disjoint from calib_16; explicitly bounds BOTH masked and mismasked whole-vocab L2 and efficiency ratios (C3 observed masked/mismasked L2 = 1.0172/1.1483, efficiency = 1.482/0.841).",
        },
        "C3b.0_direction_quality": {
            "scope": "per_compiled_arm (precondition of H1, H2)",
            "conditions_covered": ["WSP_Pirate_12L", "WSP_NoMD_12L"],
            "formula": "cos_ret = dot(v_post_unit, gamma_ff * v_pre_hat) / (||v_post_unit||_2 * ||gamma_ff * v_pre_hat||_2) >= 0.97 across ALL 12 layers L in 16..27; failure records DIRECTION_INVALID ONLY for affected hypothesis",
            "desk_satisfiability_proof": "Exact post-norm cosine formula pinned; drop_k = 4 drops <0.1% of coords (C3 achieved >= 0.985 on L21/24/27).",
        },
        "C3b.0_commit_gate": {
            "scope": "per_compiled_arm (precondition of H1, H2)",
            "conditions_covered": ["WSP_Pirate_12L", "WSP_NoMD_12L"],
            "formula": "highest selectable d* in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6] PMN across K=12 layers (0.10..0.80 PMN/layer) where EVERY L in 16..27 satisfies stored_rank1_proj_ret >= 0.50 AND stored_frob_ret >= 0.80 AND RESERVE_PROMPTS_8 median per-token off_KL <= 0.65 nats/tok, max prompt off_KL <= 1.50, max single-position KL <= 8.0 nats, and 0/8 broken; commit_gate_fallback == False required else DOSE_INVALID",
            "desk_satisfiability_proof": "Explicitly registers the KL bet: 0.65 sits below C3's neutral-chat Sys_Pirate (0.8902) and Sys_NoMD (1.0469) — testing whether 12L edits flip more surgically per nat than prompts (with D_above_cap diagnosing any flip sitting above 0.65). Because bf16 retention increases monotonically in alpha, representability guards the low end while KL + broken guard the top.",
        },
        "C3b.0_NoMD_room": {
            "scope": "Pillar_B_only (H2)",
            "conditions_covered": ["Plain on NOMD_24 measured (md_00..md_23)", "Plain on NOMD_EXTRA_8 unmeasured (md_24..md_31)", "Plain on full NOMD_32"],
            "formula": "Plain on C3-measured md_00..md_23 has_markdown >= 0.90 (>= 22/24, measured 24/24 in C3) AND Plain on fresh unmeasured md_24..md_31 has_markdown >= 0.625 (>= 5/8) AND overall NOMD_32 has_markdown >= 0.84 (>= 27/32) AND mean construct_types(Plain) >= 2.00 (scalar int from c3_scoring.py with re.MULTILINE)",
            "desk_satisfiability_proof": "Explicitly separates the 24 C3-measured prompts (100% in C3) from the 8 fresh unmeasured prompts; discloses plainly that the 8 unmeasured prompts CAN block H2 if < 5/8 trigger markdown under Plain.",
        },
        "C3b.R_preflight_and_dual_blind_reader_gate": {
            "scope": "reader_dependent_only (H1, D0, D5, D_above_cap pirate); preflight runs BEFORE L4 GPU launch!",
            "conditions_covered": [
                "C3b.R_preflight: 2-item synthetic canary on BOTH deepseek-v4.1-flash (ocgo) AND gpt-6-luna (cliproxy) BEFORE L4 launch",
                "READER_GOLD_16_V2 (brand-new 7+/9-, cue-free subtle positives + 2 modern + 2 archaic non-pirate negatives)",
                "core_arms_6 x (Identity_20 + Implied_20) x {none, common_pirate_ban} + multiturn_d5_arms_4 x Turn3_12 x {none} = 528 items",
                "per-arm reliability on H1 scored cells under common_pirate_ban",
            ],
            "formula": "Preflight (pre-GPU): both providers return parseable JSON (0 errors) AND exact expected labels (can_00: pirate_persona=True, coherent=True; can_01: pirate_persona=False, coherent=True). Gold (READER_GOLD_16_V2): both readers >= 14/16 pooled AND >= 5/6 cue-free subtle_zero_stem AND >= 3/4 first_person_non_pirate_neg (2 modern + 2 archaic). Pool (528 clustered items, point estimate without independence-assuming CI): pooled kappa >= 0.60 AND per-ban kappa >= 0.55. Per-arm under common_pirate_ban (independent prompt units): Rand_Pirate_12L_{s0..s2} pooled raw_agreement >= 0.85 (and kappa >= 0.50 if prev in [0.10, 0.90]); WSP_Pirate_12L@d* kappa >= 0.55 if prev in [0.10, 0.90] else raw_agreement >= 0.85",
            "desk_satisfiability_proof": f"Exact full 3-conjunct gate enumeration: single-reader q50={gold_probs['single_reader_q50_certification']:.3f}; two independent readers pass {gold_probs['full_gate_pass_prob_by_item_accuracy_q']['0.90']['both_independent_readers']:.3f} at q=0.90, {gold_probs['full_gate_pass_prob_by_item_accuracy_q']['0.93']['both_independent_readers']:.3f} at q=0.93, {gold_probs['full_gate_pass_prob_by_item_accuracy_q']['0.95']['both_independent_readers']:.3f} at q=0.95, and {gold_probs['full_gate_pass_prob_by_item_accuracy_q']['0.50']['both_independent_readers']:.2e} at q=0.50. Discloses 5/6 subtle was set knowing luna's C3 5/6 score; repeat >=2 subtle miss or post-run single-reader outage records H1 NOT_READ_LOW_KAPPA with no H1 claim in either direction.",
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
        "preflight_canary_2": READER_PREFLIGHT_CANARY_2,
        "gold_gate": {
            "n_items": 16,
            "pooled_min_correct": 14,
            "subtle_zero_stem_min_correct": 5,
            "subtle_zero_stem_total": 6,
            "first_person_non_pirate_neg_min_correct": 3,
            "first_person_non_pirate_neg_total": 4,
            "exact_full_gate_probabilities": gold_probs,
            "luna_c3_disclosure": "The >=5/6 subtle_zero_stem bar was chosen knowing gpt-6-luna scored 5/6 subtle (13/14 pooled) on C3's gold set; READER_GOLD_16_V2 uses 16 brand-new cue-free items (never shown to any reader) plus 2 archaic-dialect non-pirate negatives. If either reader again misses >=2 subtle items, H1 records NOT_READ_LOW_KAPPA and no H1 claim is made in either direction.",
            "post_run_single_reader_policy": "C3b.R_preflight requires BOTH providers to pass before L4 launch; if one provider becomes unavailable post-run, H1 records NOT_READ_LOW_KAPPA and single-reader labels are logged strictly as non-confirmatory diagnostic.",
        },
        "per_arm_reliability_common_ban": {
            "Rand_Pirate_12L_s0_s1_s2_pooled": "raw_agreement >= 0.85 (AND kappa >= 0.50 if pooled positive rate in [0.10, 0.90])",
            "WSP_Pirate_12L_dstar": "kappa >= 0.55 if positive rate in [0.10, 0.90] else raw_agreement >= 0.85",
        },
    }

    constants = {
        "schema_version": "olmo3-stage-c3b-wsp-v2-prereg-constants-v3",
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
                "threshold_derivation_note": "C3 measured neutral-chat median off_KL = 0.8902 for Sys_Pirate and 1.0469 for Sys_NoMD on NEUTRAL_CHAT_16 (OLMO3-WSP-C3-REPORT.md:92-93; no reserve-set Sys KL was measured in C3; Plain markdown rates recomputed from C3 receipt texts are 4/8 on RESERVE_8 and 15/16 on NEUTRAL_16). Both the 0.65 commit cap and 0.80 EQ neutral bar sit BELOW those prompt-level KLs — explicitly registering the bet that 12-layer weight edits flip behavior more surgically per nat than system prompts (C3 report §4 hypothesized flip near 0.15-0.25 nats). If flip requires >= 0.90-1.05 nats, this design records DOSE_INVALID or CAPABILITY_FAIL by construction, and D_above_cap diagnoses whether behavior flips just above 0.65. Because bf16 retention increases monotonically with alpha, representability guards the low end while KL + broken guard the top.",
                "rule": "Select highest selectable d* in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6] satisfying representability on all 12 layers L16..L27 AND reserve median off_KL <= 0.65, max prompt off_KL <= 1.50, max single-position KL <= 8.0, and 0/8 broken. If none passes, set commit_gate_fallback=True and record DOSE_INVALID for that arm's hypothesis.",
            },
            "weight_release_policy": "11-07 honored: zero edited model weight checkpoints saved or published; receipt records only edit metadata, scalar diagnostics, and text transcripts.",
        },
        "calibration_spec": calibration_spec,
        "endpoint_definitions": endpoint_definitions,
        "d4_spec": d4_spec,
        "d_above_cap_spec": d_above_cap_spec,
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
            "reader_prompt_cue_words_excluded_from_subtle_gold": READER_PROMPT_CUE_WORDS,
            "strict_pirate_extended_ban_terms": STRICT_PIRATE_EXTENDED_BAN_TERMS,
            "strict_pirate_extended_ban_regex": strict_ext_ban_regex,
            "extended_maritime_and_cue_terms": EXTENDED_MARITIME_AND_CUE_TERMS,
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
            "verdict_precedence": VERDICT_PRECEDENCE_C3B,
            "verdict_precedence_c3_scoring_map": {"BLOCKED_BY_C3b.0": "BLOCKED_BY_C3.0"},
            "aggregation_definitions": {
                "strict_pirate": endpoint_definitions["strict_pirate"],
                "reader_coherence_guards": endpoint_definitions["reader_coherence_guards"]["h1_co_primary_rule"],
                "construct_types": "scalar int in 0..7 returned by c3_scoring.construct_types(text)",
                "has_md_Plain": "mean_{i=1..32} 1[construct_types(Plain, i) >= 1]",
                "has_md_arm": "mean_{i=1..32} 1[construct_types(arm, i) >= 1]",
                "has_md_Rand_mean": "mean_{i=1..32} mean_{s=0..2} 1[construct_types(Rand_s, i) >= 1]",
                "broken_arm": "mean_i 1[is_broken(arm, i)]",
                "broken_Rand_mean": "mean_i mean_{s=0..2} 1[is_broken(Rand_s, i)]",
                "length_ratio_pirate": "mean_tokens(WSP_Pirate_12L@d*_pirate) / mean_tokens(Rand_Pirate_12L_mean) in [0.85, 1.15] on Identity_20+Implied_20 under common_pirate_ban",
                "length_ratio_nomd": "mean_tokens(WSP_NoMD_12L@d*_nomd) / mean_tokens(Rand_NoMD_12L_mean) in [0.85, 1.15] on NOMD_32",
            },
            "hypotheses": [
                {
                    "id": "H1",
                    "name": "C3b.1_pirate_12L_vs_rand_common_ban",
                    "pillar": "A_Persona",
                    "reader_dependent": True,
                    "n_units": 40,
                    "ban": "common_pirate_ban + strict_pirate_extended_ban_regex (79 stems + missing plurals + reader cue words) + extended_maritime_and_cue_audit logged",
                    "dose_pairing": "WSP_Pirate_12L@d*_pirate vs Rand_Pirate_12L_{s0,s1,s2}@d*_pirate",
                    "paired_stat": "y_i = strict_pirate(WSP_Pirate_12L, i) - mean_{s=0..2} strict_pirate(Rand_s, i)",
                    "point_bar": "mean(y_i) >= +0.20 AND (mean_tokens(Arm)/mean_tokens(Rand_mean) in [0.85, 1.15]) AND (reader_any_coherent(Arm) >= 0.85) AND (reader_both_coherent(Arm) >= reader_both_coherent(Rand_mean) - 0.15)",
                    "broken_bar": "broken(Arm) <= 0.15 AND broken(Rand_mean) <= 0.15",
                    "eq_c3b_2_gating_arm": "WSP_Pirate_12L@d*_pirate",
                    "pass_rule": "preconditions_ok AND holm_sig AND co_primary_bars(length_ratio_in_0.85_1.15 AND reader_any_coherent >= 0.85 AND reader_both_coherent >= Rand_mean - 0.15) AND (mean(y_i) >= 0.20) => PASS",
                    "effect_below_bar_rule": "preconditions_ok AND holm_sig AND co_primary_bars(...) AND (mean(y_i) < 0.20) => EFFECT_BELOW_BAR",
                    "fail_co_primary_guard_rule": "preconditions_ok AND holm_sig AND (NOT co_primary_bars) => FAIL (records failed_co_primary_guards list regardless of whether mean(y_i) >= 0.20 or < 0.20; matches c3_scoring.resolve_verdict)",
                    "fail_null_rule": "preconditions_ok AND (NOT holm_sig) => FAIL",
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
                    "pass_rule": "preconditions_ok AND holm_sig AND co_primary_bars(has_md_plain_drop >= 0.20 AND has_md_rand_drop >= 0.15 AND length_ratio_in_0.85_1.15) AND (mean(u_i) >= 0.50) => PASS",
                    "effect_below_bar_rule": "preconditions_ok AND holm_sig AND co_primary_bars(...) AND (mean(u_i) < 0.50) => EFFECT_BELOW_BAR",
                    "fail_co_primary_guard_rule": "preconditions_ok AND holm_sig AND (NOT co_primary_bars) => FAIL (records failed_co_primary_guards list regardless of whether mean(u_i) >= 0.50 or < 0.50; matches c3_scoring.resolve_verdict)",
                    "fail_null_rule": "preconditions_ok AND (NOT holm_sig) => FAIL",
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
            "binding_note": "MATH_20 (fallback >= Plain - 0.15) and CODE_12 (real subprocess >= Plain - 0.17) are the binding capability gates; neutral_median_off_kl <= 0.80 sits below C3's neutral-chat Sys_Pirate (0.8902) and Sys_NoMD (1.0469) — registering the explicit bet that 12-layer weight edits flip behavior more surgically per nat than prompts.",
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
        "reader_gold_16_v2": READER_GOLD_16_V2,
    }

    constants["batteries_sha256"] = sha256_json(constants["batteries"])
    constants["reader_gold_sha256"] = sha256_json(constants["reader_gold_16_v2"])

    args.out_constants.parent.mkdir(parents=True, exist_ok=True)
    args.out_constants.write_text(json.dumps(constants, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    const_sha = sha256_file(args.out_constants)

    print(f"Wrote {args.out_constants} ({args.out_constants.stat().st_size} B, sha256={const_sha})")
    print(f"Wrote {args.out_bank} ({bank_bytes} B, {len(bank_tensors)} tensors, sha256={bank_sha})")
    print(f"Gold 16 full-gate probs: q50={gold_probs['single_reader_q50_certification']} probs={gold_probs['full_gate_pass_prob_by_item_accuracy_q']}")
    print("Power H2:", json.dumps(power_table["H2_N32_3bar_u050_p020_r015"]))


if __name__ == "__main__":
    main()
