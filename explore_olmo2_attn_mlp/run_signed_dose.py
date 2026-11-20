#!/usr/bin/env python3
"""Frozen Instruct Signed-Dose, Role-Genericity & Turn-Boundary Runner (OLMo 2 1B, Design v2.1).

Implements `explore_olmo2_attn_mlp/SIGNED-DOSE-DESIGN.md` (v2.1, folding Alden #32 & #36):
  - Immutable 40-char SHA pinning (`allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e`),
    `float32`, `attn_implementation="eager"`. Refuses `main` or `a1847dff`.
  - Launch admission enforcement: loads and hashes exact committed `signed_dose_manifest.json` and
    `SIGNED-DOSE-DESIGN.md` bytes, cross-checks every regenerated tokenizer/mask binding against the
    bound manifest BEFORE calling the model weight loader, and refuses with zero loader calls on any
    mismatch or missing file.
  - Strictly downstream head eligibility (`L_head > L_unit`): exact 108 eligible `(unit, head)` pairs
    (54 key + 54 matched-layer random). Raises `ValueError` on any non-causal `(unit, head)` pair.
  - Exact prefix-attached own/peer token binding (`n_peers_distinct >= 15`, own excluded, duplicates
    deduplicated, hard error on any BPE merge across the read boundary).
  - Block A + Block B (`B = 19` uniform batch shape across donor, unpatched, `unpushed_z`, and 100%-coverage
    same-dose `sham_z`): exact negated signed null companions (`+d * v_null[k]` paired with `-d * v_null[k]`),
    preserving all 4 draws per tuple at full float precision and computing centered even `E(d)` and odd `O(d)`
    on both `RawLogit` and `SpecLogit`.
  - Block C (Q3 mask-localized residual writes at `d = +0.8`, `B = 10`, separate estimand): runs
    `pre_user_union` + `bos_tok` (idx 0) + `user_hdr_turn_final` + `post_user_delim_final` + matched controls
    across `one_turn`, `two_turn_asst_user`, and `no_boundary_concat`.
  - Separate explicit receipt identities (`partial_checkpoint` vs `smoke_receipt` vs `full_receipt`),
    separate B19 and B10 timing, and a two-turn Q3 prompt included in `--smoke-1unit`.
  - Modes:
      --self-test-only           : Runs all unit invariants, Alden #32 sabotages, and Alden #36 fake-loader admission tests.
      --emit-manifest <path>     : Tokenizer-only pre-run audit manifest (zero model weights loaded).
      --smoke-1unit              : Measured 1-unit smoke (2 Q1+Q2 + 3 Q3 frames incl. two_turn_asst_user).
      (default)                  : Full frozen schedule (`6,630` B=19 + `740` B=10 forwards) with separate checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


SCHEMA_VERSION = "signed_dose_v2.1"
MODEL_REPO = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_COMMIT = "48d788eca847d4d7548f375ad03d3c9312f6139e"
FORBIDDEN_REFS = ("main", "a1847dff")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
N_NULL_DRAWS = 4
DOSE_MAGNITUDES = (0.2, 0.4, 0.8, 1.6)
RANDOM_HEAD_SEED = 20261120
NULL_SEED_BASE = 202611200
SMOKE_Q12_IDS: tuple[str, ...] = ("self_F1", "third_F1")
SMOKE_Q3_IDS: tuple[str, ...] = (
    "q3_one_turn_self_F1",
    "q3_two_turn_asst_user_self_F1",
    "q3_no_boundary_concat_self_F1",
)

# Pinned Instruct chat residual medians from `results/part99-pushes.json`.
PINNED_MEDIANS_CHAT: dict[int, float] = {
    3: 7.3919267654418945,
    5: 10.05025863647461,
    6: 10.501473426818848,
}

# Full 21 Part 92 slot units (used for the 20-peer off-diagonal SpecLogit baseline).
ALL_SLOT_UNITS: list[dict[str, Any]] = [
    {"id": "p92-02", "layer": 3,  "neuron": 2628, "lead": "the",  "content": "town"},
    {"id": "p92-03", "layer": 9,  "neuron": 2156, "lead": "the",  "content": "Internet"},
    {"id": "p92-04", "layer": 3,  "neuron": 5389, "lead": "a",    "content": "very tall tree"},
    {"id": "p92-05", "layer": 3,  "neuron": 6986, "lead": "a",    "content": "fox"},
    {"id": "p92-07", "layer": 1,  "neuron": 7429, "lead": "a",    "content": "cat"},
    {"id": "p92-09", "layer": 6,  "neuron": 4145, "lead": "a",    "content": "dog"},
    {"id": "p92-10", "layer": 2,  "neuron": 2466, "lead": "a",    "content": "chicken"},
    {"id": "p92-13", "layer": 3,  "neuron": 976,  "lead": "a",    "content": "skeleton"},
    {"id": "p92-18", "layer": 3,  "neuron": 85,   "lead": "the",  "content": "shadow of your smile"},
    {"id": "p92-20", "layer": 2,  "neuron": 4658, "lead": "the",  "content": "shadow that dances upon the river"},
    {"id": "p92-24", "layer": 4,  "neuron": 1594, "lead": "the",  "content": "queen of hearts"},
    {"id": "p92-25", "layer": 3,  "neuron": 7850, "lead": "",     "content": "King Midas"},
    {"id": "p92-32", "layer": 12, "neuron": 3422, "lead": "",     "content": "Sir Ray Ray"},
    {"id": "p92-33", "layer": 5,  "neuron": 90,   "lead": "",     "content": "Dr. Memento"},
    {"id": "p92-35", "layer": 4,  "neuron": 631,  "lead": "your", "content": "guardian angel"},
    {"id": "p92-36", "layer": 4,  "neuron": 7528, "lead": "the",  "content": "real Alex"},
    {"id": "p92-37", "layer": 5,  "neuron": 7218, "lead": "a",    "content": "large, friendly dog"},
    {"id": "p92-42", "layer": 2,  "neuron": 2264, "lead": "a",    "content": "member of the royalist ranks"},
    {"id": "p92-46", "layer": 5,  "neuron": 4673, "lead": "a",    "content": "regular guy"},
    {"id": "p92-47", "layer": 4,  "neuron": 39,   "lead": "a",    "content": "lawyer"},
    {"id": "p92-50", "layer": 2,  "neuron": 1064, "lead": "a",    "content": "gruff old man"},
]
SLOT_BY_ID = {u["id"]: u for u in ALL_SLOT_UNITS}
CORE_UNITS: tuple[str, ...] = ("p92-02", "p92-05", "p92-13", "p92-46", "p92-09")

KEY_HEADS: tuple[tuple[int, int], ...] = (
    (4, 3), (5, 4), (6, 6), (6, 8), (7, 5),
    (8, 4), (8, 11), (8, 15), (9, 6), (9, 10),
    (11, 0), (12, 4),
)

# Forbidden prompts (Part 180 v5.1 + Run 1 / explore_ladder.py discovery bank).
FORBIDDEN_PROMPTS: tuple[str, ...] = (
    "Introduce yourself.",
    "Tell me about yourself.",
    "Who are you?",
    "Describe yourself in a few sentences.",
    "What should I call you, and what are you like?",
    "Say hello and tell me who I'm talking to.",
    "If you had to explain what you are to a child, what would you say?",
    "Write a short paragraph about anything you like.",
    "What did you do today?",
    "Describe your perfect day.",
    "Describe what you can see around you.",
    "Write a short diary entry.",
    "Tell me a short story.",
    "Write a postcard to a friend.",
    "How would you identify yourself in one line?",
    "What kind of being or thing are you?",
    "Who is the stranger at the door?",
    "What is the mysterious figure in the mist?",
)

# Counterbalanced story frames F1–F3 and object-compatible frames O1–O2 (Design v2 section 2.3).
STORY_FRAMES: dict[str, dict[str, str]] = {
    "F1": {
        "third": "At the village clinic before dusk, {subj} opens the heavy oak door. Identify {subj_l} in one line.",
        "self":  "At the village clinic before dusk, you open the heavy oak door. Identify yourself in one line.",
    },
    "F2": {
        "third": "Across the quiet market square at dawn, {subj} carries a woven basket. Identify {subj_l} in one line.",
        "self":  "Across the quiet market square at dawn, you carry a woven basket. Identify yourself in one line.",
    },
    "F3": {
        "third": "Beside the stone harbor wall at nightfall, {subj} waits by the iron gate. Identify {subj_l} in one line.",
        "self":  "Beside the stone harbor wall at nightfall, you wait by the iron gate. Identify yourself in one line.",
    },
}

OBJ_FRAMES: dict[str, str] = {
    "O1": "In the quiet room by the window sits {subj}. Identify {subj_l} in one line.",
    "O2": "Near the wooden bench in the courtyard rests {subj}. Identify {subj_l} in one line.",
}

Q3_TURN1_USER = "What color is clear ice?"
Q3_TURN1_ASST = "Clear ice is colorless."


def build_q12_prompts() -> list[dict[str, Any]]:
    """Build the frozen 18 counterbalanced Q1+Q2 prompts (Design v2 section 2.3)."""
    rows: list[dict[str, Any]] = []
    # SELF_HELD_NEW (3: F1–F3; F1, F2 in Block A)
    for f in ("F1", "F2", "F3"):
        rows.append({
            "id": f"self_{f}",
            "cell": "SELF_HELD_NEW",
            "frame": f,
            "block_a": f in ("F1", "F2"),
            "question": STORY_FRAMES[f]["self"],
            "subject": "I",
            "copula": " am",
        })
    # THIRD_GENERIC_NEW (3: F1–F3; F1, F2 in Block A)
    for f, subj in (("F1", "the stranger"), ("F2", "the visitor"), ("F3", "the traveler")):
        cap = subj[0].upper() + subj[1:]
        rows.append({
            "id": f"third_{f}",
            "cell": "THIRD_GENERIC_NEW",
            "frame": f,
            "block_a": f in ("F1", "F2"),
            "question": STORY_FRAMES[f]["third"].format(subj=subj, subj_l=subj),
            "subject": cap,
            "copula": " is",
        })
    # ROLE_DEFINITE_NEW (3: F1–F3; strictly excludes "the assistant")
    for f, subj in (("F1", "the nurse"), ("F2", "the baker"), ("F3", "the guard")):
        cap = subj[0].upper() + subj[1:]
        rows.append({
            "id": f"role_{f}",
            "cell": "ROLE_DEFINITE_NEW",
            "frame": f,
            "block_a": False,
            "question": STORY_FRAMES[f]["third"].format(subj=subj, subj_l=subj),
            "subject": cap,
            "copula": " is",
        })
    # NAMED_PERSON_NEW (3: F1–F3)
    for f, name in (("F1", "Clara"), ("F2", "Devon"), ("F3", "Nora")):
        rows.append({
            "id": f"npers_{f}",
            "cell": "NAMED_PERSON_NEW",
            "frame": f,
            "block_a": False,
            "question": STORY_FRAMES[f]["third"].format(subj=name, subj_l=name),
            "subject": name,
            "copula": " is",
        })
    # LEX_ASSISTANT_CTRL (2: F1–F2, separate lexical control)
    for f in ("F1", "F2"):
        rows.append({
            "id": f"lexasst_{f}",
            "cell": "LEX_ASSISTANT_CTRL",
            "frame": f,
            "block_a": False,
            "question": STORY_FRAMES[f]["third"].format(subj="the assistant", subj_l="the assistant"),
            "subject": "The assistant",
            "copula": " is",
        })
    # PERSON_OBJFRAME & OBJECT_OBJFRAME (2+2: O1–O2, identical "The <noun> is " length & wording)
    for f, subj in (("O1", "the nurse"), ("O2", "the guard")):
        cap = subj[0].upper() + subj[1:]
        rows.append({
            "id": f"persobj_{f}",
            "cell": "PERSON_OBJFRAME",
            "frame": f,
            "block_a": False,
            "question": OBJ_FRAMES[f].format(subj=subj, subj_l=subj),
            "subject": cap,
            "copula": " is",
        })
    for f, subj in (("O1", "the lantern"), ("O2", "the kettle")):
        cap = subj[0].upper() + subj[1:]
        rows.append({
            "id": f"obj_{f}",
            "cell": "OBJECT_OBJFRAME",
            "frame": f,
            "block_a": False,
            "question": OBJ_FRAMES[f].format(subj=subj, subj_l=subj),
            "subject": cap,
            "copula": " is",
        })
    return rows


def build_q3_prompts() -> list[dict[str, Any]]:
    """Build the 12 Q3 dialogue-frame prompts (3 frames x 2 cells x 2 stories F1, F2)."""
    base_specs = [
        ("F1", "SELF_HELD_NEW",     STORY_FRAMES["F1"]["self"], "I", " am"),
        ("F2", "SELF_HELD_NEW",     STORY_FRAMES["F2"]["self"], "I", " am"),
        ("F1", "THIRD_GENERIC_NEW", STORY_FRAMES["F1"]["third"].format(subj="the stranger", subj_l="the stranger"), "The stranger", " is"),
        ("F2", "THIRD_GENERIC_NEW", STORY_FRAMES["F2"]["third"].format(subj="the visitor", subj_l="the visitor"), "The visitor", " is"),
    ]
    rows: list[dict[str, Any]] = []
    for dframe in ("one_turn", "two_turn_asst_user", "no_boundary_concat"):
        for story, cell, q, subj, cop in base_specs:
            rows.append({
                "id": f"q3_{dframe}_{cell[:4].lower()}_{story}",
                "dialogue_frame": dframe,
                "story": story,
                "cell": cell,
                "question": q,
                "subject": subj,
                "copula": cop,
            })
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Validation, Eligibility, Tokenization, Nulls & Scorer Rules
# ---------------------------------------------------------------------------------------------------------------------

def validate_pinned_ref(repo: str, commit: str) -> None:
    for bad in FORBIDDEN_REFS:
        if commit == bad or commit.startswith(bad):
            raise ValueError(f"Refused forbidden ref {repo}@{commit}")
    if not HEX40_RE.match(commit):
        raise ValueError(f"Refused non-40-char commit SHA: {commit!r}")


def assert_no_forbidden_overlap(prompts: list[dict[str, Any]]) -> None:
    forb = {" ".join(s.strip().lower().split()) for s in FORBIDDEN_PROMPTS}
    for pr in prompts:
        nq = " ".join(pr["question"].strip().lower().split())
        if nq in forb:
            raise ValueError(f"Prompt {pr['id']} matches forbidden prompt: {pr['question']!r}")
        if "who are you" in nq:
            raise ValueError(f"Prompt {pr['id']} contains sealed phrase 'who are you': {pr['question']!r}")


def sample_matched_random_heads(n_heads_per_layer: int = 16, seed: int = RANDOM_HEAD_SEED) -> tuple[tuple[int, int], ...]:
    rng = random.Random(seed)
    key_set = set(KEY_HEADS)
    by_layer: dict[int, int] = {}
    for lyr, _h in KEY_HEADS:
        by_layer[lyr] = by_layer.get(lyr, 0) + 1
    out: list[tuple[int, int]] = []
    for lyr in sorted(by_layer):
        cands = [h for h in range(n_heads_per_layer) if (lyr, h) not in key_set]
        picked = sorted(rng.sample(cands, by_layer[lyr]))
        for h in picked:
            out.append((lyr, h))
    return tuple(out)


def assert_causal_head(unit_layer: int, head_layer: int, unit_id: str = "", head_tag: str = "") -> None:
    """Strictly downstream eligibility: head layer MUST be strictly greater than unit output layer."""
    if head_layer <= unit_layer:
        raise ValueError(
            f"Non-causal head rejected: unit {unit_id} (L{unit_layer}) cannot be patched at head {head_tag} (L{head_layer} <= L{unit_layer})"
        )


def build_eligible_pairs(rand_heads: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    all_heads = [(l, h, "key") for (l, h) in KEY_HEADS] + [(l, h, "rand") for (l, h) in rand_heads]
    pairs: list[dict[str, Any]] = []
    for uid in CORE_UNITS:
        u_lyr = int(SLOT_BY_ID[uid]["layer"])
        for h_lyr, h_idx, h_kind in all_heads:
            if h_lyr > u_lyr:
                pairs.append({
                    "unit": uid,
                    "unit_layer": u_lyr,
                    "head_layer": h_lyr,
                    "head_idx": h_idx,
                    "head_name": f"L{h_lyr}H{h_idx}",
                    "head_kind": h_kind,
                })
    return pairs


def encode_no_special(tok: Any, text: str) -> list[int]:
    return [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]


def prefix_attached_first_token(tok: Any, prefix_text: str, prefix_ids: list[int], content: str) -> int:
    """Verify prefix stability (zero boundary BPE merge) and return the first content token ID."""
    full_ids = encode_no_special(tok, prefix_text + " " + content)
    if len(full_ids) <= len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"Boundary BPE merge detected appending {' ' + content!r} onto prefix ending {prefix_text[-30:]!r}"
        )
    return int(full_ids[len(prefix_ids)])


def build_prefix_token_binding(tok: Any, prefix_text: str, prefix_ids: list[int], unit_id: str) -> dict[str, Any]:
    """Bind prefix-attached own token ID and deduplicated non-own peer token IDs."""
    u = SLOT_BY_ID[unit_id]
    own_id = prefix_attached_first_token(tok, prefix_text, prefix_ids, u["content"])
    peer_ids_raw: list[int] = []
    for other in ALL_SLOT_UNITS:
        if other["id"] == unit_id:
            continue
        tid = prefix_attached_first_token(tok, prefix_text, prefix_ids, other["content"])
        if tid != own_id:
            peer_ids_raw.append(tid)
    peers_dedup = sorted(set(peer_ids_raw))
    if len(peers_dedup) < 15:
        raise ValueError(f"Too few distinct peer tokens ({len(peers_dedup)} < 15) for {unit_id}")
    return {
        "own_token_id": own_id,
        "own_token_str": tok.decode([own_id]),
        "peer_token_ids": peers_dedup,
        "n_peers_distinct": len(peers_dedup),
    }


def locate_all_subsequence_spans(ids: list[int], sub: list[int]) -> list[list[int]]:
    if not sub:
        raise ValueError("Empty subsequence")
    hits: list[list[int]] = []
    m = len(sub)
    for i in range(0, len(ids) - m + 1):
        if ids[i : i + m] == sub:
            hits.append(list(range(i, i + m)))
    return hits


def render_q3_dialogue(tok: Any, dframe: str, question: str) -> str:
    if dframe == "one_turn":
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )
    if dframe == "two_turn_asst_user":
        return tok.apply_chat_template(
            [
                {"role": "user", "content": Q3_TURN1_USER},
                {"role": "assistant", "content": Q3_TURN1_ASST},
                {"role": "user", "content": question},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
    if dframe == "no_boundary_concat":
        combined = f"{Q3_TURN1_USER}\n{Q3_TURN1_ASST}\n{question}"
        return tok.apply_chat_template(
            [{"role": "user", "content": combined}],
            add_generation_prompt=True,
            tokenize=False,
        )
    raise ValueError(f"Unknown dialogue_frame: {dframe!r}")


def locate_q3_masks(tok: Any, dframe: str, question: str, subject: str, copula: str, lead: str) -> dict[str, Any]:
    """Build exact, occurrence-verified Q3 mask indices & decoded tokens for a dialogue frame."""
    chat_prefix = render_q3_dialogue(tok, dframe, question)
    chat_ids = encode_no_special(tok, chat_prefix)

    # Full prefix up to read_pos (just before first content token)
    subj_text = chat_prefix + subject
    subj_ids = encode_no_special(tok, subj_text)
    if subj_ids[: len(chat_ids)] != chat_ids:
        raise ValueError("Boundary merge at subject")
    cop_text = subj_text + copula
    cop_ids = encode_no_special(tok, cop_text)
    if cop_ids[: len(subj_ids)] != subj_ids:
        raise ValueError("Boundary merge at copula")
    if lead:
        full_text = cop_text + " " + lead
        full_ids = encode_no_special(tok, full_text)
        if full_ids[: len(cop_ids)] != cop_ids:
            raise ValueError("Boundary merge at lead")
    else:
        full_text = cop_text
        full_ids = cop_ids

    # Locate exact last occurrence of `question` inside `chat_prefix` via char offsets
    q_start = chat_prefix.rfind(question)
    if q_start < 0:
        raise ValueError(f"Question not found in chat_prefix: {question!r}")
    q_end = q_start + len(question)
    offsets = tok(chat_prefix, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    user_q_span = [i for i, (s, e) in enumerate(offsets) if e > q_start and s < q_end]
    if len(user_q_span) < 5:
        raise ValueError(f"Question span too short ({len(user_q_span)}) for matched 5-tok control")

    # Locate exact `<|user|>\n` and `<|assistant|>\n` occurrences by exact char offsets in `chat_prefix`
    user_hdr_marker = "<|user|>\n"
    asst_hdr_marker = "<|assistant|>\n"
    eot_marker = "<|endoftext|>"

    def find_marker_spans(marker: str) -> list[list[int]]:
        spans: list[list[int]] = []
        pos = 0
        while True:
            idx = chat_prefix.find(marker, pos)
            if idx < 0:
                break
            end = idx + len(marker)
            toks = [i for i, (s, e) in enumerate(offsets) if e > idx and s < end]
            spans.append(toks)
            pos = end
        return spans

    user_hdr_spans = find_marker_spans(user_hdr_marker)
    asst_hdr_spans = find_marker_spans(asst_hdr_marker)
    eot_spans = find_marker_spans(eot_marker)

    if dframe in ("one_turn", "no_boundary_concat"):
        if len(user_hdr_spans) != 1 or len(asst_hdr_spans) != 1 or len(eot_spans) != 1:
            raise ValueError(
                f"Unexpected delimiter counts for {dframe}: user={len(user_hdr_spans)}, asst={len(asst_hdr_spans)}, eot={len(eot_spans)}"
            )
    elif dframe == "two_turn_asst_user":
        if len(user_hdr_spans) != 2 or len(asst_hdr_spans) != 2 or len(eot_spans) != 2:
            raise ValueError(
                f"Unexpected delimiter counts for two_turn_asst_user: user={len(user_hdr_spans)}, asst={len(asst_hdr_spans)}, eot={len(eot_spans)}"
            )

    bos_tok = eot_spans[0]
    if bos_tok != [0]:
        raise ValueError(f"Expected bos_tok == [0], got {bos_tok}")

    user_hdr_final = user_hdr_spans[-1]
    asst_delim_final = asst_hdr_spans[-1]
    # Verify final user header strictly precedes active question, and final assistant delim follows it
    if user_hdr_final[-1] >= user_q_span[0]:
        raise ValueError(f"user_hdr_turn_final {user_hdr_final} does not precede question {user_q_span}")
    if asst_delim_final[0] <= user_q_span[-1]:
        raise ValueError(f"post_user_delim_final {asst_delim_final} does not follow question {user_q_span}")

    if dframe == "two_turn_asst_user":
        mid_eot = eot_spans[1]
        # OLMo 2 template inserts a single '\n' token between mid-dialogue `<|endoftext|>` and `<|user|>\n`.
        mid_eot_nl = list(range(mid_eot[0], user_hdr_final[0]))
        if len(mid_eot_nl) != 2 or tok.decode([full_ids[mid_eot_nl[1]]]) != "\n":
            raise ValueError(
                f"Expected exact ['<|endoftext|>', '\\n'] before Turn-2 header, got {mid_eot_nl} "
                f"({[tok.decode([full_ids[i]]) for i in mid_eot_nl]})"
            )
        pre_user_union = mid_eot_nl + user_hdr_final
    else:
        if bos_tok[-1] + 1 != user_hdr_final[0]:
            raise ValueError(f"bos_tok {bos_tok} not contiguous with user_hdr_turn_final {user_hdr_final}")
        pre_user_union = bos_tok + user_hdr_final

    # Validate active post_user_delim_final ends right at chat_ids end
    if asst_delim_final[-1] != len(chat_ids) - 1:
        raise ValueError(f"post_user_delim_final {asst_delim_final} does not end at chat_ids end {len(chat_ids)-1}")

    masks: dict[str, list[int] | None] = {
        "all_pos": None,
        "bos_tok": bos_tok,
        "user_hdr_turn_final": user_hdr_final,
        "pre_user_union": pre_user_union,
        "post_user_delim_final": asst_delim_final,
        "user_tail_1": user_q_span[-1:],
        "user_tail_5": user_q_span[-len(asst_delim_final):],
        "user_tail_union": user_q_span[-len(pre_user_union):],
        "user_prompt_energy_matched": list(user_q_span),
        "subject": list(range(len(chat_ids), len(subj_ids))),
        "copula_lead": list(range(len(subj_ids), len(full_ids))),
    }
    if dframe == "two_turn_asst_user":
        masks["mid_eot"] = eot_spans[1]
        masks["mid_eot_nl"] = mid_eot_nl
        masks["user_hdr_turn1"] = user_hdr_spans[0]
        masks["pre_user_turn1_union"] = bos_tok + user_hdr_spans[0]

    mask_decoded = {
        k: (None if idxs is None else [tok.decode([full_ids[i]]) for i in idxs])
        for k, idxs in masks.items()
    }
    return {
        "full_text": full_text,
        "full_ids": full_ids,
        "read_pos": len(full_ids) - 1,
        "masks": masks,
        "mask_decoded": mask_decoded,
        "len_user_q": len(user_q_span),
        "k_union": len(pre_user_union),
    }


def assert_q3_final_occurrence_not_turn1(dframe: str, masks: dict[str, Any]) -> None:
    """Guard verifying that `user_hdr_turn_final` in `two_turn_asst_user` is NOT Turn-1's header."""
    if dframe == "two_turn_asst_user":
        if masks["user_hdr_turn_final"] == masks["user_hdr_turn1"]:
            raise ValueError("Sabotage/bug: user_hdr_turn_final points to Turn-1 header instead of Turn-2!")
        if masks["user_hdr_turn_final"][0] <= masks["user_hdr_turn1"][-1]:
            raise ValueError("user_hdr_turn_final must strictly follow user_hdr_turn1")


def make_null_unit_vectors(
    w_down: torch.Tensor,
    target_neuron: int,
    target_unit_vec: torch.Tensor,
    n_draws: int,
    seed_base: int,
) -> dict[str, list[torch.Tensor]]:
    """Construct unit-norm (`||v||_2 == 1`) orthogonal null directions; caller scales by exact `+d * m` and `-d * m`."""
    v_hat = target_unit_vec.float() / float(target_unit_vec.float().norm().item())
    d_model, d_mlp = w_down.shape
    wdown_orth: list[torch.Tensor] = []
    iso_resid: list[torch.Tensor] = []
    for m in range(n_draws):
        g1 = torch.Generator(device="cpu").manual_seed(seed_base + 1000 * m + 1)
        raw = torch.randn((d_mlp,), generator=g1, dtype=torch.float32)
        raw[target_neuron] = 0.0
        rv = w_down.float() @ raw
        rv = rv - torch.dot(rv, v_hat) * v_hat
        rv = rv / float(rv.norm().item())
        if abs(float(torch.dot(rv, v_hat).item())) >= 1e-5:
            raise RuntimeError(f"wdown_orth draw {m} orthogonality failed")
        wdown_orth.append(rv)

        g2 = torch.Generator(device="cpu").manual_seed(seed_base + 1000 * m + 501)
        irv = torch.randn((d_model,), generator=g2, dtype=torch.float32)
        irv = irv - torch.dot(irv, v_hat) * v_hat
        irv = irv / float(irv.norm().item())
        if abs(float(torch.dot(irv, v_hat).item())) >= 1e-5:
            raise RuntimeError(f"iso_resid draw {m} orthogonality failed")
        iso_resid.append(irv)
    return {"wdown_orth": wdown_orth, "iso_resid": iso_resid}


def build_b19_vectors(
    v_target_unit: torch.Tensor,
    null_unit_vecs: dict[str, list[torch.Tensor]],
    scale_mag: float,
) -> torch.Tensor:
    """Build exact B=19 batch vectors where rows 7..10 = -(rows 3..6) and rows 15..18 = -(rows 11..14)."""
    zero_v = torch.zeros_like(v_target_unit)
    pos_t = scale_mag * v_target_unit
    neg_t = -scale_mag * v_target_unit
    pos_w = [scale_mag * v for v in null_unit_vecs["wdown_orth"]]
    neg_w = [-v for v in pos_w]
    pos_i = [scale_mag * v for v in null_unit_vecs["iso_resid"]]
    neg_i = [-v for v in pos_i]
    vecs = torch.stack([zero_v, pos_t, neg_t, *pos_w, *neg_w, *pos_i, *neg_i], dim=0)
    assert_signed_companions(vecs)
    return vecs


def assert_signed_companions(vecs: torch.Tensor) -> None:
    """Verify exact sign-companion pairing in the B=19 batch (Alden #32 point 1 & sabotage check 1)."""
    if vecs.shape[0] != 19:
        raise ValueError(f"Expected B=19, got {vecs.shape[0]}")
    if float(vecs[0].abs().max().item()) != 0.0:
        raise ValueError("Row 0 must be exact zero")
    if float((vecs[1] + vecs[2]).abs().max().item()) > 1e-5:
        raise ValueError("Neuron +d and -d rows are not exact negatives")
    for k in range(N_NULL_DRAWS):
        if float((vecs[3 + k] + vecs[7 + k]).abs().max().item()) > 1e-5:
            raise ValueError(f"wdown_orth draw {k} +d/-d companions are not exact negatives!")
        if float((vecs[11 + k] + vecs[15 + k]).abs().max().item()) > 1e-5:
            raise ValueError(f"iso_resid draw {k} +d/-d companions are not exact negatives!")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_sd6a_bos_share(delta_st_bos: float, delta_st_union: float, floor: float = 0.30) -> dict[str, Any]:
    """Evaluate SD.6(a) with Alden #32's predeclared positive floor on the combined `pre_user_union` contrast."""
    if not (math.isfinite(delta_st_bos) and math.isfinite(delta_st_union) and math.isfinite(floor)):
        raise ValueError(f"Non-finite input to evaluate_sd6a_bos_share: bos={delta_st_bos}, union={delta_st_union}")
    if delta_st_union < floor:
        return {
            "status": "not_read (union below floor)",
            "passed": False,
            "bos": float(delta_st_bos),
            "union": float(delta_st_union),
            "floor": float(floor),
            "share": None,
        }
    share = delta_st_bos / delta_st_union
    passed = delta_st_bos > 0.50 * delta_st_union
    return {
        "status": "held" if passed else "failed",
        "passed": passed,
        "bos": float(delta_st_bos),
        "union": float(delta_st_union),
        "floor": float(floor),
        "share": float(share),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Hooks (Residual-Stream Write & Head Pre-o_proj Patching)
# ---------------------------------------------------------------------------------------------------------------------

def add_residual_write_hook(layer: nn.Module, vecs_b_d: torch.Tensor, pos_indices: list[int] | None):
    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        b, t, d = h.shape
        v = vecs_b_d.to(device=h.device, dtype=h.dtype)
        if v.shape != (b, d):
            raise ValueError(f"Write shape {tuple(v.shape)} != ({b}, {d})")
        p = h.clone()
        if pos_indices is None:
            p = p + v.unsqueeze(1)
        else:
            if not pos_indices or len(set(pos_indices)) != len(pos_indices):
                raise ValueError(f"Invalid pos_indices: {pos_indices}")
            if min(pos_indices) < 0 or max(pos_indices) >= t:
                raise ValueError(f"Out-of-range pos_indices {pos_indices} for seq_len {t}")
            idx_t = torch.tensor(pos_indices, dtype=torch.long, device=h.device)
            p[:, idx_t, :] = p[:, idx_t, :] + v.unsqueeze(1)
        if isinstance(out, tuple):
            return (p,) + tuple(out[1:])
        if isinstance(out, list):
            return [p, *out[1:]]
        return p
    return layer.register_forward_hook(_hook)


def make_head_patch_prehook(head_idx: int, head_dim: int, replacement_z: torch.Tensor):
    """Pre-hook on `self_attn.o_proj` replacing head slice `[head_idx*hd : (head_idx+1)*hd]`."""
    s = head_idx * head_dim
    e = s + head_dim

    def _hook(_mod: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x = args[0].clone()
        rep = replacement_z.to(device=x.device, dtype=x.dtype)
        if rep.shape[0] == 1 and x.shape[0] > 1:
            rep = rep.expand(x.shape[0], -1, -1)
        if rep.shape != x[:, :, s:e].shape:
            raise ValueError(f"Head patch shape {tuple(rep.shape)} != {tuple(x[:, :, s:e].shape)}")
        x[:, :, s:e] = rep
        return (x,) + args[1:]

    return _hook


# ---------------------------------------------------------------------------------------------------------------------
# Self-Test Suite (All 4 Alden #32 Sabotages + Paired Positives + Fake-Layer Invariance)
# ---------------------------------------------------------------------------------------------------------------------

def _build_fake_tokenizer_for_self_test():
    """Minimal deterministic tokenizer mimicking OLMo 2 chat markers + offset mapping for CPU self-tests."""
    class _FakeTokenizer:
        def __init__(self) -> None:
            self._vocab: dict[str, int] = {
                "<|endoftext|>": 100,
                "\n": 101,
                "<": 102,
                "|": 103,
                "user": 104,
                "assistant": 105,
                ">\n": 106,
            }
            self._inv: dict[int, str] = {v: k for k, v in self._vocab.items()}
            self._next_id = 200
            self._pat = re.compile(
                r"<\|endoftext\|>|>\n|<|\||\n| [A-Za-z0-9.,?!'-]+|[A-Za-z0-9.,?!'-]+| +"
            )

        def _tid(self, tok_str: str) -> int:
            if tok_str not in self._vocab:
                self._vocab[tok_str] = self._next_id
                self._inv[self._next_id] = tok_str
                self._next_id += 1
            return self._vocab[tok_str]

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = True,
            tokenize: bool = False,
        ) -> str:
            if tokenize:
                raise AssertionError("Runner should pass tokenize=False")
            parts: list[str] = ["<|endoftext|>"]
            for i, msg in enumerate(messages):
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    if i > 0:
                        parts.append("<|endoftext|>\n")
                    parts.append(f"<|user|>\n{content}\n")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}")
                else:
                    raise ValueError(f"Unsupported role {role}")
            if add_generation_prompt:
                parts.append("<|assistant|>\n")
            return "".join(parts)

        def __call__(
            self,
            text: str,
            add_special_tokens: bool = False,
            return_offsets_mapping: bool = False,
        ) -> dict[str, Any]:
            ids: list[int] = []
            offsets: list[tuple[int, int]] = []
            pos = 0
            while pos < len(text):
                m = self._pat.match(text, pos)
                if m is None:
                    piece = text[pos : pos + 1]
                    start, end = pos, pos + 1
                else:
                    piece = m.group(0)
                    start, end = m.start(), m.end()
                ids.append(self._tid(piece))
                offsets.append((start, end))
                pos = end
            out: dict[str, Any] = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offsets
            return out

        def decode(self, ids: list[int]) -> str:
            return "".join(self._inv[int(i)] for i in ids)

    return _FakeTokenizer()


def _build_dummy_olmo_model(vocab_size: int = 2048, hidden_size: int = 32, n_heads: int = 16, n_layers: int = 13):
    """Tiny 13-layer dummy causal LM with exact OLMo 2 hook points (`layers[i].mlp.down_proj`, `self_attn.o_proj`)."""
    class _DummyAttn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.o_proj = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.o_proj(x)

    class _DummyMLP(nn.Module):
        def __init__(self, d: int, d_mlp: int) -> None:
            super().__init__()
            self.down_proj = nn.Linear(d_mlp, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 0.0

    class _DummyLayer(nn.Module):
        def __init__(self, d: int, d_mlp: int) -> None:
            super().__init__()
            self.self_attn = _DummyAttn(d)
            self.mlp = _DummyMLP(d, d_mlp)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
            return (x + 0.05 * self.self_attn(x),)

    class _DummyInner(nn.Module):
        def __init__(self, v: int, d: int, nl: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(v, d)
            # Large enough MLP width to index any core unit neuron (max CORE_UNITS neuron is 6986).
            self.layers = nn.ModuleList([_DummyLayer(d, 7040) for _ in range(nl)])

    class _DummyOut:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class _DummyConfig:
        def __init__(self, d: int, nh: int) -> None:
            self.hidden_size = d
            self.num_attention_heads = nh

    class _DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _DummyConfig(hidden_size, n_heads)
            self.model = _DummyInner(vocab_size, hidden_size, n_layers)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> _DummyOut:
            h = self.model.embed(input_ids)
            for lyr in self.model.layers:
                h = lyr(h)[0]
            return _DummyOut(self.lm_head(h))

    torch.manual_seed(1234)
    return _DummyModel()


def run_admission_self_tests() -> dict[str, Any]:
    """Alden #36 admission & fake-loader suite exercising real `execute_run` with zero HF downloads."""
    fake_tok = _build_fake_tokenizer_for_self_test()
    with tempfile.TemporaryDirectory(prefix="sd_admit_test_") as td:
        tmp = Path(td)
        design_path = tmp / "SIGNED-DOSE-DESIGN.md"
        design_path.write_text("# Fake Frozen Design v2.1\n", encoding="utf-8")
        manifest_path = tmp / "signed_dose_manifest.json"
        valid_manifest = build_manifest(fake_tok, design_path=design_path)
        _atomic_write_json(manifest_path, valid_manifest)
        valid_m_sha = sha256_file(manifest_path)
        valid_d_sha = sha256_file(design_path)

        loader_calls = {"n": 0}

        def fake_tok_loader(_repo: str, _commit: str):
            return fake_tok

        def fake_model_loader(_repo: str, _commit: str, device: torch.device):
            loader_calls["n"] += 1
            return _build_dummy_olmo_model().to(device)

        def make_args(mpath: Path, dpath: Path, outpath: Path, exp_m: str | None = None, exp_d: str | None = None):
            return argparse.Namespace(
                smoke_1unit=True,
                device="cpu",
                out=outpath,
                manifest=mpath,
                design=dpath,
                expected_manifest_sha256=exp_m,
                expected_design_sha256=exp_d,
                emit_stdout_json=False,
            )

        # 1. Positive fixture: exact package reaches fake loader and completes smoke_receipt cleanly
        pos_out = tmp / "smoke_pos.json"
        receipt = execute_run(
            make_args(manifest_path, design_path, pos_out, exp_m=valid_m_sha, exp_d=valid_d_sha),
            tokenizer_loader=fake_tok_loader,
            model_loader=fake_model_loader,
        )
        assert loader_calls["n"] == 1, f"Expected 1 model loader call, got {loader_calls['n']}"
        assert pos_out.is_file()
        assert receipt["meta"]["artifact_type"] == "smoke_receipt"
        assert receipt["meta"]["run_mode"] == "smoke_1unit"
        assert receipt["meta"]["complete"] is True
        assert receipt["meta"]["manifest_sha256"] == valid_m_sha
        assert receipt["meta"]["design_sha256"] == valid_d_sha
        assert receipt["meta"]["timing"]["b19_elapsed_sec"] >= 0.0
        assert receipt["meta"]["timing"]["b10_elapsed_sec"] >= 0.0
        assert receipt["meta"]["timing"]["n_b19_done"] == 98
        assert receipt["meta"]["timing"]["n_b10_done"] == 37
        smoke_q3_ids = {r["prompt_id"] for r in receipt["q3_rows"]}
        assert "q3_two_turn_asst_user_self_F1" in smoke_q3_ids, "Smoke must include longest two-turn Q3 frame!"
        max_sham = max(max(r["spec"]["max_abs_sham"], r["raw"]["max_abs_sham"]) for r in receipt["q12_rows"])
        assert max_sham < 1e-5, f"Dummy model sham error too high: {max_sham}"

        # 2. Refusal: missing manifest -> zero loader calls
        loader_calls["n"] = 0
        try:
            execute_run(
                make_args(tmp / "nonexistent_manifest.json", design_path, tmp / "bad1.json"),
                tokenizer_loader=fake_tok_loader,
                model_loader=fake_model_loader,
            )
            raise AssertionError("Missing manifest was not refused!")
        except FileNotFoundError:
            pass
        assert loader_calls["n"] == 0, "Missing manifest reached model loader!"

        # 3. Refusal: wrong design bytes/digest -> zero loader calls
        bad_design = tmp / "BAD-DESIGN.md"
        bad_design.write_text("# Tampered Design\n", encoding="utf-8")
        loader_calls["n"] = 0
        try:
            execute_run(
                make_args(manifest_path, bad_design, tmp / "bad2.json"),
                tokenizer_loader=fake_tok_loader,
                model_loader=fake_model_loader,
            )
            raise AssertionError("Wrong design digest was not refused!")
        except ValueError:
            pass
        assert loader_calls["n"] == 0, "Wrong design reached model loader!"

        # 4. Refusal: 1 changed own_token_id in manifest -> zero loader calls (Alden #36 exact witness!)
        bad_tok_m = copy.deepcopy(valid_manifest)
        bad_tok_m["q12_prompts"][0]["per_unit"][CORE_UNITS[0]]["own_token_id"] += 1
        bad_tok_path = tmp / "manifest_bad_tok.json"
        _atomic_write_json(bad_tok_path, bad_tok_m)
        loader_calls["n"] = 0
        try:
            execute_run(
                make_args(bad_tok_path, design_path, tmp / "bad3.json"),
                tokenizer_loader=fake_tok_loader,
                model_loader=fake_model_loader,
            )
            raise AssertionError("Changed own_token_id in manifest was not refused!")
        except ValueError:
            pass
        assert loader_calls["n"] == 0, "Changed own_token_id reached model loader!"

        # 5. Refusal: 1 changed Q3 mask index in manifest -> zero loader calls
        bad_mask_m = copy.deepcopy(valid_manifest)
        bad_mask_m["q3_prompts"][0]["per_unit"][CORE_UNITS[0]]["masks"]["bos_tok"] = [1]
        bad_mask_path = tmp / "manifest_bad_mask.json"
        _atomic_write_json(bad_mask_path, bad_mask_m)
        loader_calls["n"] = 0
        try:
            execute_run(
                make_args(bad_mask_path, design_path, tmp / "bad4.json"),
                tokenizer_loader=fake_tok_loader,
                model_loader=fake_model_loader,
            )
            raise AssertionError("Changed Q3 mask in manifest was not refused!")
        except ValueError:
            pass
        assert loader_calls["n"] == 0, "Changed Q3 mask reached model loader!"

        # 6. Refusal: wrong --expected-manifest-sha256 -> zero loader calls
        loader_calls["n"] = 0
        try:
            execute_run(
                make_args(manifest_path, design_path, tmp / "bad5.json", exp_m="0" * 64),
                tokenizer_loader=fake_tok_loader,
                model_loader=fake_model_loader,
            )
            raise AssertionError("Wrong expected manifest SHA-256 was not refused!")
        except ValueError:
            pass
        assert loader_calls["n"] == 0, "Wrong expected manifest SHA reached model loader!"

    return {
        "positive_reached_fake_loader": True,
        "smoke_includes_two_turn_q3": True,
        "smoke_dummy_max_sham": max_sham,
        "refused_missing_manifest_zero_loader": True,
        "refused_wrong_design_zero_loader": True,
        "refused_changed_own_token_zero_loader": True,
        "refused_changed_mask_zero_loader": True,
        "refused_wrong_expected_sha_zero_loader": True,
    }


def run_self_tests() -> dict[str, Any]:
    # 0. Pinning & prompt hygiene
    validate_pinned_ref(MODEL_REPO, MODEL_COMMIT)
    for bad in ("main", "a1847dff", "a1847dff" + "0" * 32, "short"):
        try:
            validate_pinned_ref(MODEL_REPO, bad)
            raise AssertionError(f"Failed to reject {bad!r}")
        except ValueError:
            pass
    q12 = build_q12_prompts()
    q3 = build_q3_prompts()
    assert len(q12) == 18 and len(q3) == 12
    assert_no_forbidden_overlap(q12 + q3)
    try:
        assert_no_forbidden_overlap([{"id": "bad", "question": "Who are you?"}])
        raise AssertionError("Failed to reject 'Who are you?'")
    except ValueError:
        pass

    # Exact downstream-only count check
    rheads = sample_matched_random_heads()
    pairs = build_eligible_pairs(rheads)
    assert len(pairs) == 108, f"Expected 108 eligible pairs, got {len(pairs)}"
    assert sum(1 for p in pairs if p["head_kind"] == "key") == 54
    assert sum(1 for p in pairs if p["head_kind"] == "rand") == 54

    # Sabotage 1: Swapped/unpaired sign companions -> MUST fail; paired positive -> passes
    g = torch.Generator(device="cpu").manual_seed(7)
    w_down = torch.randn((32, 64), generator=g, dtype=torch.float32)
    t_unit = w_down[:, 5] / float(w_down[:, 5].norm().item())
    nulls = make_null_unit_vectors(w_down, 5, t_unit, N_NULL_DRAWS, seed_base=11)
    good_b19 = build_b19_vectors(t_unit, nulls, scale_mag=0.8)
    assert_signed_companions(good_b19)  # paired positive
    bad_b19 = good_b19.clone()
    bad_b19[7] = good_b19[8]  # swapped sign companion!
    try:
        assert_signed_companions(bad_b19)
        raise AssertionError("Sabotage 1 (swapped sign companion) was not caught!")
    except ValueError:
        pass

    # Sabotage 2: Wrong boundary occurrence -> MUST fail; paired positive -> passes
    good_masks = {"user_hdr_turn1": [1, 2, 3, 4], "user_hdr_turn_final": [20, 21, 22, 23]}
    assert_q3_final_occurrence_not_turn1("two_turn_asst_user", good_masks)  # paired positive
    bad_masks = {"user_hdr_turn1": [1, 2, 3, 4], "user_hdr_turn_final": [1, 2, 3, 4]}
    try:
        assert_q3_final_occurrence_not_turn1("two_turn_asst_user", bad_masks)
        raise AssertionError("Sabotage 2 (wrong boundary occurrence) was not caught!")
    except ValueError:
        pass

    # Sabotage 3: False fraction denominator (Alden's bos=-0.05, union=-0.20) -> MUST return not_read, NOT pass
    bad_frac = evaluate_sd6a_bos_share(delta_st_bos=-0.05, delta_st_union=-0.20, floor=0.30)
    assert bad_frac["passed"] is False and bad_frac["status"] == "not_read (union below floor)", f"Sabotage 3 failed: {bad_frac}"
    good_frac = evaluate_sd6a_bos_share(delta_st_bos=0.40, delta_st_union=0.60, floor=0.30)
    assert good_frac["passed"] is True and good_frac["status"] == "held", f"Positive 3 failed: {good_frac}"

    # Sabotage 4: Non-causal head (L_h <= L_u) -> MUST raise; paired positive -> passes
    assert_causal_head(unit_layer=6, head_layer=7, unit_id="p92-09", head_tag="L7H5")  # paired positive
    for ul, hl in ((5, 5), (6, 6), (6, 4)):
        try:
            assert_causal_head(unit_layer=ul, head_layer=hl, unit_id="test", head_tag=f"L{hl}H0")
            raise AssertionError(f"Sabotage 4 (non-causal L{hl} <= L{ul}) was not caught!")
        except ValueError:
            pass

    # Fake-layer residual-write & head-patch sham invariance check
    class DummyDecoder(nn.Module):
        def forward(self, x):
            return (x * 1.0,)

    torch.manual_seed(0)
    dec = DummyDecoder()
    x = torch.randn((19, 8, 32), dtype=torch.float32)
    h_w = add_residual_write_hook(dec, good_b19, [0, 3])
    y = dec(x)[0]
    h_w.remove()
    sel_err = float(((y[:, [0, 3], :] - x[:, [0, 3], :]) - good_b19.unsqueeze(1)).abs().max().item())
    comp_err = float((y[:, [1, 2, 4, 5, 6, 7], :] - x[:, [1, 2, 4, 5, 6, 7], :]).abs().max().item())
    assert sel_err < 1e-6 and comp_err == 0.0

    o_proj = nn.Linear(32, 32, bias=False)
    z_in = torch.randn((19, 8, 32), dtype=torch.float32)
    base_o = o_proj(z_in)
    h_sham = o_proj.register_forward_pre_hook(make_head_patch_prehook(2, 8, z_in[:, :, 16:24].clone()))
    sham_o = o_proj(z_in)
    h_sham.remove()
    sham_err = float((sham_o - base_o).abs().max().item())
    assert sham_err == 0.0

    admission_report = run_admission_self_tests()

    return {
        "status": "PASSED",
        "eligible_pairs": len(pairs),
        "sabotage_1_swapped_sign_caught": True,
        "sabotage_2_wrong_boundary_caught": True,
        "sabotage_3_false_fraction_not_read": True,
        "sabotage_4_noncausal_head_caught": True,
        "fake_layer_sel_err": sel_err,
        "fake_layer_comp_err": comp_err,
        "fake_head_sham_err": sham_err,
        "admission_tests": admission_report,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Pre-Run Manifest Builder & Launch Admission Enforcement
# ---------------------------------------------------------------------------------------------------------------------

def build_manifest(tok: Any, design_path: Path | None = None) -> dict[str, Any]:
    q12 = build_q12_prompts()
    q3 = build_q3_prompts()
    assert_no_forbidden_overlap(q12 + q3)
    rheads = sample_matched_random_heads()
    pairs = build_eligible_pairs(rheads)

    q12_entries: list[dict[str, Any]] = []
    for pr in q12:
        chat_prefix = tok.apply_chat_template(
            [{"role": "user", "content": pr["question"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        per_unit: dict[str, Any] = {}
        for uid in CORE_UNITS:
            lead = SLOT_BY_ID[uid]["lead"]
            p_text = chat_prefix + pr["subject"] + pr["copula"] + (f" {lead}" if lead else "")
            p_ids = encode_no_special(tok, p_text)
            binding = build_prefix_token_binding(tok, p_text, p_ids, uid)
            per_unit[uid] = {
                "prefix_text": p_text,
                "seq_len": len(p_ids),
                "read_pos": len(p_ids) - 1,
                "prefix_ids": p_ids,
                **binding,
            }
        q12_entries.append({**pr, "per_unit": per_unit})

    q3_entries: list[dict[str, Any]] = []
    for pr in q3:
        per_unit_q3: dict[str, Any] = {}
        for uid in CORE_UNITS:
            lead = SLOT_BY_ID[uid]["lead"]
            loc = locate_q3_masks(tok, pr["dialogue_frame"], pr["question"], pr["subject"], pr["copula"], lead)
            assert_q3_final_occurrence_not_turn1(pr["dialogue_frame"], loc["masks"])
            binding = build_prefix_token_binding(tok, loc["full_text"], loc["full_ids"], uid)
            per_unit_q3[uid] = {**loc, **binding}
        q3_entries.append({**pr, "per_unit": per_unit_q3})

    dpath = design_path if design_path is not None else (Path(__file__).resolve().parent / "SIGNED-DOSE-DESIGN.md")
    if not dpath.is_file():
        raise FileNotFoundError(f"Frozen design document not found: {dpath}")
    design_sha = sha256_file(dpath)

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "design_file": "explore_olmo2_attn_mlp/SIGNED-DOSE-DESIGN.md",
            "design_sha256": design_sha,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": "float32",
            "attn_implementation": "eager",
            "dose_magnitudes": list(DOSE_MAGNITUDES),
            "n_null_draws": N_NULL_DRAWS,
            "random_head_seed": RANDOM_HEAD_SEED,
            "null_seed_base": NULL_SEED_BASE,
            "pinned_medians_chat": {str(k): v for k, v in PINNED_MEDIANS_CHAT.items()},
            "missing_or_invalid_row_policy": "hard_error_abort",
        },
        "denominators": {
            "n_core_units": len(CORE_UNITS),
            "n_key_heads": len(KEY_HEADS),
            "n_rand_heads": len(rheads),
            "eligible_unit_head_pairs": len(pairs),
            "n_q12_prompts": len(q12),
            "n_block_a_prompts": sum(1 for p in q12 if p["block_a"]),
            "n_block_b_only_prompts": sum(1 for p in q12 if not p["block_a"]),
            "n_q3_prompts": len(q3),
            "planned_b19_forwards": 5 * 4 * 4 + 108 * 4 * 4 * 2 + 5 * 14 * 1 + 108 * 14 * 1 * 2,
            "planned_b10_forwards": sum(len(e["per_unit"][CORE_UNITS[0]]["masks"]) for e in q3_entries) * len(CORE_UNITS),
        },
        "key_heads": [list(h) for h in KEY_HEADS],
        "matched_random_heads": [list(h) for h in rheads],
        "eligible_pairs": pairs,
        "q12_prompts": q12_entries,
        "q3_prompts": q3_entries,
    }


def _find_first_diff(a: Any, b: Any, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type mismatch bound={type(a).__name__} ({a!r}) != regen={type(b).__name__} ({b!r})"
    if isinstance(a, dict):
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            return f"{path}: key mismatch extra_in_bound={sorted(ka - kb)}, missing_in_bound={sorted(kb - ka)}"
        for k in sorted(ka):
            d = _find_first_diff(a[k], b[k], f"{path}.{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list length mismatch bound={len(a)} != regen={len(b)}"
        for i, (va, vb) in enumerate(zip(a, b)):
            d = _find_first_diff(va, vb, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    if a != b:
        return f"{path}: value mismatch bound={a!r} != regen={b!r}"
    return None


def assert_manifests_equal(bound: dict[str, Any], regen: dict[str, Any]) -> None:
    diff = _find_first_diff(bound, regen)
    if diff is not None:
        raise ValueError(f"Committed manifest disagrees with regenerated tokenizer/design bindings at {diff}")


def admit_frozen_package(
    manifest_path: Path,
    design_path: Path,
    tokenizer_loader: Callable[[str, str], Any],
    expected_manifest_sha256: str | None = None,
    expected_design_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enforce exact committed manifest & design bytes BEFORE any model weight loader is called."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Refused launch: committed manifest not found at {manifest_path}")
    if not design_path.is_file():
        raise FileNotFoundError(f"Refused launch: committed design doc not found at {design_path}")

    manifest_bytes = manifest_path.read_bytes()
    design_bytes = design_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    design_sha = hashlib.sha256(design_bytes).hexdigest()

    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError(
            f"Refused launch: manifest SHA-256 mismatch: got {manifest_sha}, expected {expected_manifest_sha256}"
        )
    if expected_design_sha256 is not None and design_sha != expected_design_sha256:
        raise ValueError(
            f"Refused launch: design SHA-256 mismatch: got {design_sha}, expected {expected_design_sha256}"
        )

    bound = json.loads(manifest_bytes.decode("utf-8"))
    meta = bound.get("meta", {})
    if meta.get("design_sha256") != design_sha:
        raise ValueError(
            f"Refused launch: manifest.meta.design_sha256 ({meta.get('design_sha256')}) != "
            f"design file SHA-256 ({design_sha})"
        )
    if meta.get("model_repo") != MODEL_REPO or meta.get("model_commit") != MODEL_COMMIT:
        raise ValueError(
            f"Refused launch: manifest model ref {meta.get('model_repo')}@{meta.get('model_commit')} != "
            f"{MODEL_REPO}@{MODEL_COMMIT}"
        )
    validate_pinned_ref(meta["model_repo"], meta["model_commit"])

    # Cross-check EVERY regenerated prompt/token/mask binding against the bound manifest before loading weights!
    tok = tokenizer_loader(MODEL_REPO, MODEL_COMMIT)
    regen = build_manifest(tok, design_path=design_path)
    assert_manifests_equal(bound, regen)

    runner_path = Path(__file__).resolve()
    scorer_path = runner_path.parent / "score_signed_dose.py"
    pins = {
        "manifest_file": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "design_file": str(design_path),
        "design_sha256": design_sha,
        "runner_file": runner_path.name,
        "runner_sha256": sha256_file(runner_path),
        "scorer_file": scorer_path.name if scorer_path.is_file() else None,
        "scorer_sha256": sha256_file(scorer_path) if scorer_path.is_file() else None,
    }
    return bound, pins


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------------------
# Model Execution (Full Schedule or 1-Unit Measured Smoke)
# ---------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def read_dual_metrics(logits_b_v: torch.Tensor, own_id: int, peer_ids: torch.Tensor) -> tuple[list[float], list[float]]:
    if not torch.isfinite(logits_b_v).all():
        raise RuntimeError("Non-finite logits encountered!")
    raw = [float(x) for x in logits_b_v[:, own_id].cpu().tolist()]
    pmed = [float(x) for x in logits_b_v[:, peer_ids].median(dim=-1).values.cpu().tolist()]
    spec = [r - m for r, m in zip(raw, pmed)]
    if not (all(math.isfinite(x) for x in raw) and all(math.isfinite(x) for x in spec)):
        raise RuntimeError("Non-finite dual metric encountered!")
    return raw, spec


def decompose_b19(unp: list[float], pat: list[float], shm: list[float]) -> dict[str, Any]:
    d_row = [float(u - p) for u, p in zip(unp, pat)]
    d_shm = [float(u - s) for u, s in zip(unp, shm)]
    if not (all(math.isfinite(x) for x in d_row) and all(math.isfinite(x) for x in d_shm)):
        raise RuntimeError("Non-finite value in decompose_b19")
    d0 = d_row[0]
    dp, dm = d_row[1], d_row[2]
    even = 0.5 * (dp + dm) - d0
    odd = 0.5 * (dp - dm)

    def null_pack(pos_off: int, neg_off: int) -> dict[str, Any]:
        ps = [float(d_row[pos_off + k]) for k in range(N_NULL_DRAWS)]
        ms = [float(d_row[neg_off + k]) for k in range(N_NULL_DRAWS)]
        es = [float(0.5 * (a + b) - d0) for a, b in zip(ps, ms)]
        os_ = [float(0.5 * (a - b)) for a, b in zip(ps, ms)]
        return {
            "d_pos_draws": ps,
            "d_neg_draws": ms,
            "even_draws": es,
            "odd_draws": os_,
            "d_pos_med": float(statistics.median(ps)),
            "d_neg_med": float(statistics.median(ms)),
            "even_med": float(statistics.median(es)),
            "odd_med": float(statistics.median(os_)),
        }

    return {
        "d0": float(d0),
        "d_pos": float(dp),
        "d_neg": float(dm),
        "even": float(even),
        "odd": float(odd),
        "max_abs_sham": float(max(abs(x) for x in d_shm)),
        "wdown_orth": null_pack(3, 7),
        "iso_resid": null_pack(11, 15),
    }


def pack_q3_row_metrics(vals: list[float]) -> dict[str, Any]:
    if not all(math.isfinite(x) for x in vals):
        raise RuntimeError("Non-finite value in pack_q3_row_metrics")
    dn = float(vals[1] - vals[0])
    dw = [float(vals[2 + k] - vals[0]) for k in range(N_NULL_DRAWS)]
    di = [float(vals[6 + k] - vals[0]) for k in range(N_NULL_DRAWS)]
    rw = float(statistics.median(dw))
    ri = float(statistics.median(di))
    return {
        "n_z": dn,
        "r_z_wdown_med": rw,
        "l_z_wdown": float(dn - rw),
        "r_z_iso_med": ri,
        "l_z_iso": float(dn - ri),
        "wdown_draws": dw,
        "iso_draws": di,
    }


def _default_tokenizer_loader(repo: str, commit: str):
    return AutoTokenizer.from_pretrained(repo, revision=commit)


def _default_model_loader(repo: str, commit: str, device: torch.device):
    return AutoModelForCausalLM.from_pretrained(
        repo,
        revision=commit,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)


@torch.no_grad()
def execute_run(
    args: argparse.Namespace,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
    model_loader: Callable[[str, str, torch.device], Any] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    validate_pinned_ref(MODEL_REPO, MODEL_COMMIT)
    device = torch.device(args.device)

    tok_loader = tokenizer_loader or _default_tokenizer_loader
    mdl_loader = model_loader or _default_model_loader

    # Strict admission BEFORE any model weight loader call!
    manifest, pins = admit_frozen_package(
        manifest_path=args.manifest,
        design_path=args.design,
        tokenizer_loader=tok_loader,
        expected_manifest_sha256=getattr(args, "expected_manifest_sha256", None),
        expected_design_sha256=getattr(args, "expected_design_sha256", None),
    )

    print(
        f"[sd] Admitted frozen package (manifest={pins['manifest_sha256'][:12]}, "
        f"design={pins['design_sha256'][:12]}). Loading {MODEL_REPO}@{MODEL_COMMIT[:12]} on {device}...",
        flush=True,
    )
    model = mdl_loader(MODEL_REPO, MODEL_COMMIT, device)
    model.eval()
    layers = model.model.layers
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

    units_to_run = [CORE_UNITS[0]] if args.smoke_1unit else list(CORE_UNITS)
    unit_vecs: dict[str, dict[str, Any]] = {}
    for uid in units_to_run:
        u = SLOT_BY_ID[uid]
        li, ni = int(u["layer"]), int(u["neuron"])
        w_down = layers[li].mlp.down_proj.weight.detach().float().cpu()
        v_raw = w_down[:, ni]
        v_unit = v_raw / float(v_raw.norm().item())
        seed_u = NULL_SEED_BASE + 10000 * CORE_UNITS.index(uid)
        nulls = make_null_unit_vectors(w_down, ni, v_unit, N_NULL_DRAWS, seed_base=seed_u)
        unit_vecs[uid] = {
            "layer": li,
            "neuron": ni,
            "median_chat": PINNED_MEDIANS_CHAT[li],
            "v_unit": v_unit,
            "nulls": nulls,
        }

    q12_rows: list[dict[str, Any]] = []
    q3_rows: list[dict[str, Any]] = []
    n_b19_done = 0
    n_b10_done = 0
    b19_elapsed_sec = 0.0
    b10_elapsed_sec = 0.0

    all_pairs_by_unit: dict[str, list[dict[str, Any]]] = {uid: [] for uid in units_to_run}
    for p in manifest["eligible_pairs"]:
        if p["unit"] in all_pairs_by_unit:
            all_pairs_by_unit[p["unit"]].append(p)

    if args.smoke_1unit:
        q12_list = [pr for pr in manifest["q12_prompts"] if pr["id"] in SMOKE_Q12_IDS]
        q3_list = [pr for pr in manifest["q3_prompts"] if pr["id"] in SMOKE_Q3_IDS]
        if len(q12_list) != len(SMOKE_Q12_IDS) or len(q3_list) != len(SMOKE_Q3_IDS):
            raise RuntimeError("Smoke prompt selection did not match manifest prompt IDs")
    else:
        q12_list = manifest["q12_prompts"]
        q3_list = manifest["q3_prompts"]

    run_mode = "smoke_1unit" if args.smoke_1unit else "full"
    checkpoint_path = args.out.with_suffix(".checkpoint.json")

    def build_payload(units_done: list[str], complete: bool) -> dict[str, Any]:
        total_sec = time.perf_counter() - t0
        ms_b19 = (1000.0 * b19_elapsed_sec / n_b19_done) if n_b19_done > 0 else None
        ms_b10 = (1000.0 * b10_elapsed_sec / n_b10_done) if n_b10_done > 0 else None
        planned_b19 = int(manifest["denominators"]["planned_b19_forwards"])
        planned_b10 = int(manifest["denominators"]["planned_b10_forwards"])
        proj_b19 = (ms_b19 * planned_b19 / 1000.0) if ms_b19 is not None else None
        proj_b10 = (ms_b10 * planned_b10 / 1000.0) if ms_b10 is not None else None
        proj_tot = (proj_b19 + proj_b10) if (proj_b19 is not None and proj_b10 is not None) else None
        artifact_type = (
            ("smoke_receipt" if args.smoke_1unit else "full_receipt")
            if complete
            else "partial_checkpoint"
        )
        return {
            "meta": {
                **manifest["meta"],
                "schema_version": SCHEMA_VERSION,
                "run_mode": run_mode,
                "artifact_type": artifact_type,
                "complete": bool(complete),
                "smoke_1unit": bool(args.smoke_1unit),
                "manifest_sha256": pins["manifest_sha256"],
                "design_sha256": pins["design_sha256"],
                "runner_sha256": pins["runner_sha256"],
                "scorer_sha256": pins["scorer_sha256"],
                "pins": pins,
                "units_planned": list(units_to_run),
                "units_completed": list(units_done),
                "n_b19_done": n_b19_done,
                "n_b10_done": n_b10_done,
                "timing": {
                    "total_elapsed_sec": float(total_sec),
                    "b19_elapsed_sec": float(b19_elapsed_sec),
                    "b10_elapsed_sec": float(b10_elapsed_sec),
                    "n_b19_done": n_b19_done,
                    "n_b10_done": n_b10_done,
                    "ms_per_b19": float(ms_b19) if ms_b19 is not None else None,
                    "ms_per_b10": float(ms_b10) if ms_b10 is not None else None,
                    "projected_full_b19_sec": float(proj_b19) if proj_b19 is not None else None,
                    "projected_full_b10_sec": float(proj_b10) if proj_b10 is not None else None,
                    "projected_full_total_sec": float(proj_tot) if proj_tot is not None else None,
                },
            },
            "denominators": manifest["denominators"],
            "q12_rows": q12_rows,
            "q3_rows": q3_rows,
        }

    for idx_u, uid in enumerate(units_to_run):
        uv = unit_vecs[uid]
        li = uv["layer"]
        u_pairs = all_pairs_by_unit[uid]
        capture_layers = sorted({p["head_layer"] for p in u_pairs})

        # Blocks A + B (timed strictly separately from B10!)
        t_u_b19 = time.perf_counter()
        for pr in q12_list:
            pu = pr["per_unit"][uid]
            ids_1 = torch.tensor([pu["prefix_ids"]], dtype=torch.long, device=device)
            ids_19 = ids_1.expand(19, -1)
            rpos = int(pu["read_pos"])
            own_id = int(pu["own_token_id"])
            peer_ids = torch.tensor(pu["peer_token_ids"], dtype=torch.long, device=device)

            mags = (0.8,) if args.smoke_1unit else (DOSE_MAGNITUDES if pr["block_a"] else (0.8,))
            for mag in mags:
                vecs_19 = build_b19_vectors(uv["v_unit"], uv["nulls"], scale_mag=mag * uv["median_chat"])

                captured_z: dict[int, torch.Tensor] = {}
                cap_handles = []
                for hl in capture_layers:
                    def make_cap(layer_idx: int):
                        def _cap(_m, inp):
                            captured_z[layer_idx] = inp[0].detach().cpu()
                        return _cap
                    cap_handles.append(layers[hl].self_attn.o_proj.register_forward_pre_hook(make_cap(hl)))
                h_write = add_residual_write_hook(layers[li], vecs_19, None)
                try:
                    unp_logits = model(input_ids=ids_19, use_cache=False).logits[:, rpos].float()
                finally:
                    h_write.remove()
                    for ch in cap_handles:
                        ch.remove()
                n_b19_done += 1
                unp_raw, unp_spec = read_dual_metrics(unp_logits, own_id, peer_ids)

                for hp in u_pairs:
                    hl, hi = int(hp["head_layer"]), int(hp["head_idx"])
                    assert_causal_head(li, hl, uid, hp["head_name"])
                    s, e = hi * head_dim, (hi + 1) * head_dim
                    z_full = captured_z[hl]
                    z_donor = z_full[0:1, :, s:e]
                    z_self = z_full[:, :, s:e]

                    hw = add_residual_write_hook(layers[li], vecs_19, None)
                    hp_h = layers[hl].self_attn.o_proj.register_forward_pre_hook(make_head_patch_prehook(hi, head_dim, z_donor))
                    try:
                        pat_logits = model(input_ids=ids_19, use_cache=False).logits[:, rpos].float()
                    finally:
                        hp_h.remove()
                        hw.remove()
                    n_b19_done += 1
                    pat_raw, pat_spec = read_dual_metrics(pat_logits, own_id, peer_ids)

                    hw = add_residual_write_hook(layers[li], vecs_19, None)
                    hs_h = layers[hl].self_attn.o_proj.register_forward_pre_hook(make_head_patch_prehook(hi, head_dim, z_self))
                    try:
                        shm_logits = model(input_ids=ids_19, use_cache=False).logits[:, rpos].float()
                    finally:
                        hs_h.remove()
                        hw.remove()
                    n_b19_done += 1
                    shm_raw, shm_spec = read_dual_metrics(shm_logits, own_id, peer_ids)

                    q12_rows.append({
                        "unit": uid,
                        "unit_layer": li,
                        "prompt_id": pr["id"],
                        "cell": pr["cell"],
                        "frame": pr["frame"],
                        "block_a": bool(pr["block_a"]),
                        "dose_mag": float(mag),
                        "head_name": hp["head_name"],
                        "head_layer": hl,
                        "head_idx": hi,
                        "head_kind": hp["head_kind"],
                        "spec": decompose_b19(unp_spec, pat_spec, shm_spec),
                        "raw": decompose_b19(unp_raw, pat_raw, shm_raw),
                    })
        b19_elapsed_sec += time.perf_counter() - t_u_b19

        # Block C (Q3 mask-localized residual writes at d=+0.8, B=10; timed strictly separately from B19!)
        t_u_b10 = time.perf_counter()
        scale_08 = 0.8 * uv["median_chat"]
        zero_v = torch.zeros_like(uv["v_unit"])
        vecs_10_base = torch.stack(
            [zero_v, scale_08 * uv["v_unit"], *[scale_08 * v for v in uv["nulls"]["wdown_orth"]], *[scale_08 * v for v in uv["nulls"]["iso_resid"]]],
            dim=0,
        )
        for pr in q3_list:
            pu = pr["per_unit"][uid]
            ids_10 = torch.tensor([pu["full_ids"]], dtype=torch.long, device=device).expand(10, -1)
            rpos = int(pu["read_pos"])
            own_id = int(pu["own_token_id"])
            peer_ids = torch.tensor(pu["peer_token_ids"], dtype=torch.long, device=device)
            e_scale = math.sqrt(float(pu["k_union"]) / float(pu["len_user_q"]))

            for mname, pidx in pu["masks"].items():
                vecs_10 = vecs_10_base * e_scale if mname == "user_prompt_energy_matched" else vecs_10_base
                hw = add_residual_write_hook(layers[li], vecs_10, pidx)
                try:
                    logits = model(input_ids=ids_10, use_cache=False).logits[:, rpos].float()
                finally:
                    hw.remove()
                n_b10_done += 1
                raw_l, spec_l = read_dual_metrics(logits, own_id, peer_ids)

                q3_rows.append({
                    "unit": uid,
                    "prompt_id": pr["id"],
                    "dialogue_frame": pr["dialogue_frame"],
                    "story": pr["story"],
                    "cell": pr["cell"],
                    "mask": mname,
                    "pos_indices": pidx,
                    "spec": pack_q3_row_metrics(spec_l),
                    "raw": pack_q3_row_metrics(raw_l),
                })
        b10_elapsed_sec += time.perf_counter() - t_u_b10

        units_done = units_to_run[: idx_u + 1]
        is_last = (idx_u == len(units_to_run) - 1)
        b19_ms_log = 1000.0 * b19_elapsed_sec / max(1, n_b19_done)
        b10_ms_log = 1000.0 * b10_elapsed_sec / max(1, n_b10_done)
        print(
            f"[sd] Completed unit {uid} ({idx_u+1}/{len(units_to_run)} | "
            f"b19={n_b19_done} @ {b19_ms_log:.2f} ms/fwd | b10={n_b10_done} @ {b10_ms_log:.2f} ms/fwd)",
            flush=True,
        )
        if not is_last:
            _atomic_write_json(checkpoint_path, build_payload(units_done, complete=False))

    final_payload = build_payload(list(units_to_run), complete=True)
    _atomic_write_json(args.out, final_payload)

    if args.emit_stdout_json:
        print("===BEGIN_JSON===")
        print(args.out.read_text(encoding="utf-8"))
        print("===END_JSON===")
    return final_payload


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Frozen Instruct Signed-Dose Runner (OLMo 2 1B, v2.1)")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--emit-manifest", type=Path, default=None)
    parser.add_argument("--smoke-1unit", action="store_true")
    parser.add_argument("--manifest", type=Path, default=here / "signed_dose_manifest.json")
    parser.add_argument("--design", type=Path, default=here / "SIGNED-DOSE-DESIGN.md")
    parser.add_argument("--expected-manifest-sha256", type=str, default=None)
    parser.add_argument("--expected-design-sha256", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-stdout-json", action="store_true")
    args = parser.parse_args()

    st = run_self_tests()
    if args.self_test_only:
        print(json.dumps({"self_test": st}, indent=2))
        return

    if args.emit_manifest is not None:
        tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_COMMIT, local_files_only=True)
        manifest = build_manifest(tok, design_path=args.design)
        _atomic_write_json(args.emit_manifest, manifest)
        sha = sha256_file(args.emit_manifest)
        print(json.dumps({
            "manifest_path": str(args.emit_manifest),
            "manifest_sha256": sha,
            "design_sha256": manifest["meta"]["design_sha256"],
            "denominators": manifest["denominators"],
        }, indent=2))
        return

    if args.out is None:
        args.out = Path("/tmp/smoke_signed_dose.json" if args.smoke_1unit else "/tmp/results_signed_dose.json")

    execute_run(args)


if __name__ == "__main__":
    main()
