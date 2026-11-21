#!/usr/bin/env python3
"""OLMo 2 1B Stage-Sweep, Same-Referent Alias Counterbalance & Per-Stage Dose-Calibration Runner (v2.1).

Implements `explore_olmo2_attn_mlp/STAGE-SWEEP-DESIGN.md` (v2.1, SHA-256 cf522959...):
  - Four pinned immutable 40-char HF revisions (`midtrain_end`, `sft`, `dpo`, `instruct`) in
    `torch.float32`, `attn_implementation="eager"`. Refuses `main`, `a1847dff`, or non-40-hex SHAs.
  - Four-tokenizer parity gate (`--emit-manifest` and pre-load `admit_frozen_package`): loads all 4
    checkpoints' tokenizers (zero weights), asserts identical `vocab_size` and identical token IDs on
    all 28 prompts x 5 core units (140 bindings) + own/19-peer completions before calling `model_loader`.
  - Frozen stage-own normalization (`m_{s, L_u}` median residual L2 norm across all tokens of the 4
    disjoint `CAL` prompts) + persisted unit-normalized `hat_w_{s, u}` for cross-stage cosine logging.
  - Uniform `B = 19` same-dose patching with exact negated signed null companions (`wdown_orth`,
    `iso_resid`) and 100%-coverage `sham_z`, persisting `RawLogit`, `PeerMedian`, `SpecLogit`, and
    all 19 individual peer-token shifts (`peer_deltas`) on `EVAL`.
  - Disjoint 6-dose `CAL` arm with strict pre-rebound bracket selection (`unbracketed -> None` for
    Arm B; smallest-dose tie-break; cell-wise `CAL_SELF` & `CAL_THIRD` final-logit transport gate).
  - Separate Arm A (`d = 0.8`) and Arm B (`d = d_cal(s)` when `rebound_bracketed`), plus `SENS_PLAIN`
    (`E1, E2 x {SELF, ROLE_BARE}` at `d = 0.8`).
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


SCHEMA_VERSION = "stage_sweep_v2.1"
FORBIDDEN_REFS = ("main", "a1847dff")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

N_NULL_DRAWS = 4
CAL_DOSE_GRID: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.2, 1.6)
COMMON_DOSE: float = 0.8
NULL_SEED_BASE = 202611210

SHAM_TOL = 1e-4
TRANSPORT_MIN = 0.50
NULL_TOL = 0.15
CAL_SIGNAL_MAX = -0.25
CAL_REBOUND_MIN = 0.25

STAGE_ORDER: tuple[str, ...] = ("midtrain_end", "sft", "dpo", "instruct")
STAGE_SPECS: dict[str, dict[str, str]] = {
    "midtrain_end": {
        "stage": "midtrain_end",
        "repo": "allenai/OLMo-2-0425-1B",
        "commit": "f83cd641d4f00954eb25c781e0ac360debb3239e",
        "role": "mid-training end (SFT parent, stage2-ingredient3-step23852-tokens51B)",
    },
    "sft": {
        "stage": "sft",
        "repo": "allenai/OLMo-2-0425-1B-SFT",
        "commit": "0d85a3d037876ce6ac7d4311d994400fc66ac27f",
        "role": "SFT",
    },
    "dpo": {
        "stage": "dpo",
        "repo": "allenai/OLMo-2-0425-1B-DPO",
        "commit": "c4b0485961ab24c2433b090f3b922f0913a9290f",
        "role": "DPO",
    },
    "instruct": {
        "stage": "instruct",
        "repo": "allenai/OLMo-2-0425-1B-Instruct",
        "commit": "48d788eca847d4d7548f375ad03d3c9312f6139e",
        "role": "Instruct (post-RLVR final endpoint)",
    },
}

# Full 21 Part 92 slot units (yields 20 distinct first content tokens; 19 non-own peers per core unit).
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
L3_EARLY_UNITS: tuple[str, ...] = ("p92-02", "p92-05", "p92-13")
L5L6_LATE_UNITS: tuple[str, ...] = ("p92-46", "p92-09")

HEAD_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "L8H15", "layer": 8, "head": 15, "kind": "primary"},
    {"name": "L7H5",  "layer": 7, "head": 5,  "kind": "secondary"},
    {"name": "L8H2",  "layer": 8, "head": 2,  "kind": "rand_L8"},
    {"name": "L8H3",  "layer": 8, "head": 3,  "kind": "rand_L8"},
    {"name": "L8H13", "layer": 8, "head": 13, "kind": "rand_L8"},
    {"name": "L7H4",  "layer": 7, "head": 4,  "kind": "rand_L7"},
    {"name": "L7H9",  "layer": 7, "head": 9,  "kind": "rand_L7"},
)
RAND_L8_HEADS: tuple[str, ...] = ("L8H2", "L8H3", "L8H13")
RAND_L7_HEADS: tuple[str, ...] = ("L7H4", "L7H9")

# Forbidden prompts / settings / referents from Part 180 v5.1, Part 213, Run 1, and #44 Signed-Dose.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "Introduce yourself.",
    "Tell me about yourself.",
    "Who are you?",
    "Describe yourself in a few sentences.",
    "village clinic",
    "market square",
    "stone harbor wall",
    "quiet room by the window",
    "wooden bench in the courtyard",
    "the stranger",
    "the visitor",
    "the traveler",
    "the nurse",
    "the baker",
    "the guard",
    "the assistant",
    "the lantern",
    "the kettle",
    "Clara",
    "Devon",
    "Nora",
    "Elena",
    "Miriam",
    "Tomas",
    "Hannah",
    "Victor",
    "librarian",
    "botanist",
    "carpenter",
    "captain",
)

CAL_STORY_SPECS: tuple[dict[str, str], ...] = (
    {
        "story": "C1",
        "setting": "Inside the stone lighthouse at low tide",
        "v2": "wind",
        "v3": "winds",
        "obj": "the brass clock",
        "generic": "newcomer",
    },
    {
        "story": "C2",
        "setting": "Along the narrow orchard fence at noon",
        "v2": "mend",
        "v3": "mends",
        "obj": "the wooden latch",
        "generic": "outsider",
    },
)

EVAL_STORY_SPECS: tuple[dict[str, str], ...] = (
    {
        "story": "E1",
        "setting": "Inside the quiet library before sunrise",
        "v2": "check",
        "v3": "checks",
        "obj": "the brass lamp",
        "name": "Rachel",
        "role": "teacher",
        "order": "name_first",
    },
    {
        "story": "E2",
        "setting": "In the glass greenhouse after rain",
        "v2": "water",
        "v3": "waters",
        "obj": "the tall ferns",
        "name": "Simon",
        "role": "farmer",
        "order": "name_first",
    },
    {
        "story": "E3",
        "setting": "At the wooden workshop by dusk",
        "v2": "press",
        "v3": "presses",
        "obj": "the wool coat",
        "name": "Laura",
        "role": "tailor",
        "order": "role_first",
    },
    {
        "story": "E4",
        "setting": "On the timber ferry deck at dawn",
        "v2": "coil",
        "v3": "coils",
        "obj": "the heavy rope",
        "name": "Henry",
        "role": "sailor",
        "order": "role_first",
    },
)

SMOKE_UNIT = "p92-02"
SMOKE_CAL_DOSES: tuple[float, ...] = (0.4, 0.8, 1.6)
SMOKE_EVAL_PROMPTS: tuple[str, ...] = ("eval_self_E1", "eval_role_bare_E1", "eval_alias_role_E1", "eval_alias_name_E1")
SMOKE_SENS_PROMPTS: tuple[str, ...] = ("sens_plain_self_E1", "sens_plain_role_E1")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def validate_commit_sha(commit: str) -> None:
    if commit in FORBIDDEN_REFS or not HEX40_RE.match(commit):
        raise ValueError(f"Refused non-immutable commit SHA: {commit!r}")


def assert_strictly_downstream(unit_layer: int, head_layer: int) -> None:
    if not (head_layer > unit_layer):
        raise ValueError(
            f"Non-causal (unit_layer={unit_layer}, head_layer={head_layer}): head_layer must be > unit_layer"
        )


def eligible_pairs() -> list[tuple[str, dict[str, Any]]]:
    pairs: list[tuple[str, dict[str, Any]]] = []
    for uid in CORE_UNITS:
        ul = SLOT_BY_ID[uid]["layer"]
        for hs in HEAD_SPECS:
            assert_strictly_downstream(ul, hs["layer"])
            pairs.append((uid, hs))
    return pairs


def wrap_chat(q: str, prefix: str) -> str:
    return f"<|endoftext|><|user|>\n{q}\n<|assistant|>\n{prefix}"


def wrap_plain(q: str, prefix: str) -> str:
    return f"{q}\n{prefix}"


def build_prompt_specs() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for cs in CAL_STORY_SPECS:
        st = cs["story"]
        q_self = f"{cs['setting']}, you {cs['v2']} {cs['obj']}. Identify yourself in one line."
        q_third = f"{cs['setting']}, the {cs['generic']} {cs['v3']} {cs['obj']}. Identify the {cs['generic']} in one line."
        prompts.append({
            "prompt_id": f"cal_self_{st}",
            "bank": "CAL",
            "cell": "CAL_SELF",
            "story": st,
            "serialization": "chat",
            "question": q_self,
            "prefix": "I am",
            "base_text": wrap_chat(q_self, "I am"),
        })
        prompts.append({
            "prompt_id": f"cal_third_{st}",
            "bank": "CAL",
            "cell": "CAL_THIRD",
            "story": st,
            "serialization": "chat",
            "question": q_third,
            "prefix": f"The {cs['generic']} is",
            "base_text": wrap_chat(q_third, f"The {cs['generic']} is"),
        })

    for es in EVAL_STORY_SPECS:
        st = es["story"]
        name = es["name"]
        role = es["role"]
        app = f"{name}, the {role}," if es["order"] == "name_first" else f"the {role}, {name},"
        q_self = f"{es['setting']}, you {es['v2']} {es['obj']}. Identify yourself in one line."
        q_rb = f"{es['setting']}, the {role} {es['v3']} {es['obj']}. Identify the {role} in one line."
        q_nb = f"{es['setting']}, {name} {es['v3']} {es['obj']}. Identify {name} in one line."
        q_ar = f"{es['setting']}, {app} {es['v3']} {es['obj']}. Identify the {role} in one line."
        q_an = f"{es['setting']}, {app} {es['v3']} {es['obj']}. Identify {name} in one line."
        for pid, cell, q, pref in (
            (f"eval_self_{st}", "SELF", q_self, "I am"),
            (f"eval_role_bare_{st}", "ROLE_BARE", q_rb, f"The {role} is"),
            (f"eval_name_bare_{st}", "NAME_BARE", q_nb, f"{name} is"),
            (f"eval_alias_role_{st}", "ALIAS_TO_ROLE", q_ar, f"The {role} is"),
            (f"eval_alias_name_{st}", "ALIAS_TO_NAME", q_an, f"{name} is"),
        ):
            prompts.append({
                "prompt_id": pid,
                "bank": "EVAL",
                "cell": cell,
                "story": st,
                "serialization": "chat",
                "question": q,
                "prefix": pref,
                "base_text": wrap_chat(q, pref),
            })

    for es in EVAL_STORY_SPECS[:2]:
        st = es["story"]
        role = es["role"]
        q_self = f"{es['setting']}, you {es['v2']} {es['obj']}. Identify yourself in one line."
        q_rb = f"{es['setting']}, the {role} {es['v3']} {es['obj']}. Identify the {role} in one line."
        prompts.append({
            "prompt_id": f"sens_plain_self_{st}",
            "bank": "SENS_PLAIN",
            "cell": "SELF",
            "story": st,
            "serialization": "plain",
            "question": q_self,
            "prefix": "I am",
            "base_text": wrap_plain(q_self, "I am"),
        })
        prompts.append({
            "prompt_id": f"sens_plain_role_{st}",
            "bank": "SENS_PLAIN",
            "cell": "ROLE_BARE",
            "story": st,
            "serialization": "plain",
            "question": q_rb,
            "prefix": f"The {role} is",
            "base_text": wrap_plain(q_rb, f"The {role} is"),
        })

    assert_no_forbidden_overlap(prompts)
    return prompts


def assert_no_forbidden_overlap(prompts: list[dict[str, Any]]) -> None:
    for p in prompts:
        txt = p["base_text"]
        for forb in FORBIDDEN_SUBSTRINGS:
            if forb.lower() in txt.lower():
                raise ValueError(f"Forbidden overlap {forb!r} found in prompt {p['prompt_id']}: {txt!r}")


def encode_no_special(tok: Any, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def default_tokenizer_loader(repo: str, commit: str) -> Any:
    validate_commit_sha(commit)
    return AutoTokenizer.from_pretrained(repo, revision=commit)


def default_model_loader(repo: str, commit: str, device: str) -> Any:
    validate_commit_sha(commit)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=commit,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    return model


def build_manifest(
    design_path: Path,
    tokenizer_loader: Callable[[str, str], Any] = default_tokenizer_loader,
) -> dict[str, Any]:
    if not design_path.is_file():
        raise FileNotFoundError(f"Missing design file: {design_path}")
    design_sha = sha256_file(design_path)

    for st in STAGE_ORDER:
        validate_commit_sha(STAGE_SPECS[st]["commit"])

    # Load all 4 checkpoint tokenizers (zero model weights loaded!) and assert 100% parity.
    tokenizers: dict[str, Any] = {}
    vocab_sizes: dict[str, int] = {}
    for st in STAGE_ORDER:
        sp = STAGE_SPECS[st]
        tok = tokenizer_loader(sp["repo"], sp["commit"])
        tokenizers[st] = tok
        vocab_sizes[st] = len(tok)

    ref_stage = "instruct"
    ref_tok = tokenizers[ref_stage]
    ref_vocab = vocab_sizes[ref_stage]
    for st in STAGE_ORDER:
        if vocab_sizes[st] != ref_vocab:
            raise ValueError(
                f"4-tokenizer vocab parity failed: {st} has {vocab_sizes[st]} != {ref_stage} {ref_vocab}"
            )

    prompts = build_prompt_specs()
    bindings: dict[str, dict[str, Any]] = {}
    prompt_records: list[dict[str, Any]] = []
    ref_own_by_unit: dict[str, int] = {}
    ref_peers_by_unit: dict[str, list[int]] = {}
    all_peer_pool: set[int] = set()

    for p in prompts:
        pid = p["prompt_id"]
        base_text = p["base_text"]
        comp_line = p["prefix"]
        pre_comp = base_text[: -len(comp_line)]
        ids_pre = encode_no_special(ref_tok, pre_comp)
        ids_base = encode_no_special(ref_tok, base_text)
        if ids_base[: len(ids_pre)] != ids_pre:
            raise ValueError(f"BPE merge across completion newline on {pid}")
        comp_ids = ids_base[len(ids_pre) :]
        comp_toks = [ref_tok.decode([tid]) for tid in comp_ids]

        # Verify expected completion-prefix width at exact boundary
        if p["cell"] in ("SELF", "CAL_SELF", "NAME_BARE", "ALIAS_TO_NAME"):
            if len(comp_ids) != 2:
                raise ValueError(f"Expected 2 completion tokens on {pid}, got {comp_toks}")
        else:
            if len(comp_ids) != 3 or comp_toks[0] != "The":
                raise ValueError(f"Expected 3 completion tokens ['The', role, ' is'] on {pid}, got {comp_toks}")

        # Verify 4-tokenizer exact ID parity on base_text
        for st in STAGE_ORDER:
            if encode_no_special(tokenizers[st], base_text) != ids_base:
                raise ValueError(f"4-tokenizer parity failed on base_text for {pid} at stage {st}")

        prompt_unit_bindings: dict[str, Any] = {}
        read_positions_for_prompt: set[int] = set()
        for uid in CORE_UNITS:
            u = SLOT_BY_ID[uid]
            p_u = f"{base_text} {u['lead']}" if u["lead"] else base_text
            p_ids = encode_no_special(ref_tok, p_u)

            # Verify 4-tokenizer parity on unit prompt
            for st in STAGE_ORDER:
                if encode_no_special(tokenizers[st], p_u) != p_ids:
                    raise ValueError(f"4-tokenizer parity failed on {pid}/{uid} at stage {st}")

            full_own_text = f"{p_u} {u['content']}"
            full_own_ids = encode_no_special(ref_tok, full_own_text)
            if full_own_ids[: len(p_ids)] != p_ids:
                raise ValueError(f"Own BPE merge across read_pos on {pid}/{uid}")
            t_own = full_own_ids[len(p_ids)]
            for st in STAGE_ORDER:
                if encode_no_special(tokenizers[st], full_own_text) != full_own_ids:
                    raise ValueError(f"4-tokenizer parity failed on own completion for {pid}/{uid} at stage {st}")

            peers: list[int] = []
            seen: set[int] = set()
            for pu in ALL_SLOT_UNITS:
                if pu["id"] == uid:
                    continue
                pf_text = f"{p_u} {pu['content']}"
                pf_ids = encode_no_special(ref_tok, pf_text)
                if pf_ids[: len(p_ids)] != p_ids:
                    raise ValueError(f"Peer BPE merge across read_pos on {pid}/{uid}/{pu['id']}")
                for st in STAGE_ORDER:
                    if encode_no_special(tokenizers[st], pf_text) != pf_ids:
                        raise ValueError(
                            f"4-tokenizer parity failed on peer {pu['id']} for {pid}/{uid} at stage {st}"
                        )
                tid = pf_ids[len(p_ids)]
                if tid != t_own and tid not in seen:
                    seen.add(tid)
                    peers.append(tid)

            if len(peers) < 15:
                raise ValueError(f"Too few distinct peers ({len(peers)}) on {pid}/{uid}")

            if uid not in ref_own_by_unit:
                ref_own_by_unit[uid] = t_own
                ref_peers_by_unit[uid] = list(peers)
            else:
                if ref_own_by_unit[uid] != t_own or ref_peers_by_unit[uid] != peers:
                    raise ValueError(f"Cross-prompt own/peer token mismatch on {pid}/{uid}")

            all_peer_pool.add(t_own)
            all_peer_pool.update(peers)
            read_pos = len(p_ids) - 1
            read_positions_for_prompt.add(read_pos)
            prompt_unit_bindings[uid] = {
                "prompt_text": p_u,
                "input_ids": p_ids,
                "read_pos": read_pos,
                "t_own": t_own,
                "t_own_str": ref_tok.decode([t_own]),
                "peers": peers,
                "n_peers_distinct": len(peers),
            }

        if len(read_positions_for_prompt) != 1:
            raise ValueError(f"Core units disagree on read_pos for {pid}: {read_positions_for_prompt}")

        bindings[pid] = prompt_unit_bindings
        prompt_records.append({
            **p,
            "comp_ids": comp_ids,
            "comp_toks": comp_toks,
            "read_pos": next(iter(read_positions_for_prompt)),
        })

    manifest = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "design_file": "explore_olmo2_attn_mlp/STAGE-SWEEP-DESIGN.md",
            "design_sha256": design_sha,
            "dtype": "float32",
            "attn_implementation": "eager",
            "stages": [STAGE_SPECS[st] for st in STAGE_ORDER],
            "four_tokenizer_parity_verified": True,
            "vocab_size": ref_vocab,
            "cal_dose_grid": list(CAL_DOSE_GRID),
            "common_dose": COMMON_DOSE,
            "n_null_draws": N_NULL_DRAWS,
            "null_seed_base": NULL_SEED_BASE,
            "core_units": [SLOT_BY_ID[u] for u in CORE_UNITS],
            "heads": list(HEAD_SPECS),
            "n_eligible_pairs_per_stage": len(eligible_pairs()),
            "distinct_slot_token_pool": sorted(all_peer_pool),
            "n_distinct_slot_tokens": len(all_peer_pool),
            "n_prompts": len(prompt_records),
            "n_bindings": len(prompt_records) * len(CORE_UNITS),
        },
        "prompts": prompt_records,
        "bindings": bindings,
    }
    return manifest


def _require_hex64(label: str, val: str | None) -> str:
    if not isinstance(val, str) or not HEX64_RE.match(val):
        raise ValueError(f"Missing or invalid 64-hex SHA-256 pin for {label}: {val!r}")
    return val


def admit_frozen_package(
    manifest_path: Path,
    design_path: Path,
    expected_manifest_sha256: str | None = None,
    expected_design_sha256: str | None = None,
    expected_runner_sha256: str | None = None,
    expected_scorer_sha256: str | None = None,
    scorer_path: Path | None = None,
    tokenizer_loader: Callable[[str, str], Any] = default_tokenizer_loader,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Enforce all 4 external SHA-256 pins + 4-tokenizer manifest parity BEFORE any model load."""
    exp_m = _require_hex64("expected_manifest_sha256", expected_manifest_sha256)
    exp_d = _require_hex64("expected_design_sha256", expected_design_sha256)
    exp_r = _require_hex64("expected_runner_sha256", expected_runner_sha256)
    exp_s = _require_hex64("expected_scorer_sha256", expected_scorer_sha256)

    runner_path = Path(__file__).resolve()
    resolved_scorer = runner_path.with_name("score_stage_sweep.py") if scorer_path is None else Path(scorer_path)

    for lbl, fpath in (
        ("manifest", manifest_path),
        ("design", design_path),
        ("runner", runner_path),
        ("scorer", resolved_scorer),
    ):
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing required frozen package file ({lbl}): {fpath}")

    actual_manifest_sha = sha256_file(manifest_path)
    actual_design_sha = sha256_file(design_path)
    actual_runner_sha = sha256_file(runner_path)
    actual_scorer_sha = sha256_file(resolved_scorer)

    for lbl, got, exp in (
        ("manifest", actual_manifest_sha, exp_m),
        ("design", actual_design_sha, exp_d),
        ("runner", actual_runner_sha, exp_r),
        ("scorer", actual_scorer_sha, exp_s),
    ):
        if got != exp:
            raise ValueError(f"{lbl.capitalize()} SHA-256 mismatch: disk={got} != expected={exp}")

    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if committed.get("meta", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema_version mismatch: {committed.get('meta', {}).get('schema_version')!r}")
    if committed.get("meta", {}).get("design_sha256") != actual_design_sha:
        raise ValueError(
            f"Manifest design_sha256={committed.get('meta', {}).get('design_sha256')} != disk design={actual_design_sha}"
        )

    regenerated = build_manifest(design_path=design_path, tokenizer_loader=tokenizer_loader)
    if committed != regenerated:
        raise ValueError("Committed manifest does not match regenerated 4-tokenizer bindings/parity!")

    pins = {
        "manifest_file": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "design_file": str(design_path),
        "design_sha256": actual_design_sha,
        "runner_file": runner_path.name,
        "runner_sha256": actual_runner_sha,
        "scorer_file": resolved_scorer.name,
        "scorer_sha256": actual_scorer_sha,
    }
    return committed, pins


# ---------------------------------------------------------------------------
# Deterministic null vectors, dual/decomposed readouts, and Arm B calibration
# ---------------------------------------------------------------------------

def unit_seed(stage: str, unit_id: str, prompt_id: str, dose_mag: float) -> int:
    key = f"{NULL_SEED_BASE}|{stage}|{unit_id}|{prompt_id}|{dose_mag:.4f}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def make_orthogonal_null_vectors(
    w_down_weight: torch.Tensor,
    neuron_idx: int,
    v_u: torch.Tensor,
    n_draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `n_draws` unit-norm-matched `wdown_orth` and `iso_resid` vectors orthogonal to `v_u`.

    Returns `(wdown_orth [n_draws, d], iso_resid [n_draws, d])`, each scaled to `||v_u||_2`.
    """
    d_model, d_ff = w_down_weight.shape
    device = v_u.device
    dtype = v_u.dtype
    target_norm = torch.linalg.vector_norm(v_u).clamp_min(1e-12)
    u_hat = v_u / target_norm

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    # 1. wdown_orth: random Gaussian combination of W_down columns with column `neuron_idx` strictly zeroed.
    w_mat = w_down_weight.detach().to(device="cpu", dtype=torch.float32)
    coeffs = torch.randn((d_ff, n_draws), generator=gen, dtype=torch.float32)
    coeffs[neuron_idx, :] = 0.0
    raw_wd = (w_mat @ coeffs).T  # [n_draws, d_model]
    u_hat_cpu = u_hat.detach().to(device="cpu", dtype=torch.float32)
    raw_wd = raw_wd - (raw_wd @ u_hat_cpu).unsqueeze(-1) * u_hat_cpu.unsqueeze(0)
    wd_norms = torch.linalg.vector_norm(raw_wd, dim=-1, keepdim=True).clamp_min(1e-12)
    wdown_orth = (raw_wd / wd_norms * float(target_norm.item())).to(device=device, dtype=dtype)

    # 2. iso_resid: isotropic Gaussian residual orthogonalized against v_u.
    raw_iso = torch.randn((n_draws, d_model), generator=gen, dtype=torch.float32)
    raw_iso = raw_iso - (raw_iso @ u_hat_cpu).unsqueeze(-1) * u_hat_cpu.unsqueeze(0)
    iso_norms = torch.linalg.vector_norm(raw_iso, dim=-1, keepdim=True).clamp_min(1e-12)
    iso_resid = (raw_iso / iso_norms * float(target_norm.item())).to(device=device, dtype=dtype)

    return wdown_orth, iso_resid


def read_decomposed_logits(
    logits_at_read: torch.Tensor,
    t_own: int,
    peers: list[int],
) -> tuple[list[float], list[float], list[float], list[list[float]]]:
    """Return `(raw, peer_med, spec, peer_matrix)` for batch `logits_at_read [B, V]`."""
    raw = [float(x) for x in logits_at_read[:, t_own].tolist()]
    peer_t = logits_at_read[:, peers]  # [B, n_peers]
    peer_mat = [[float(v) for v in row] for row in peer_t.tolist()]
    peer_med = [float(statistics.median(row)) for row in peer_mat]
    spec = [r - pm for r, pm in zip(raw, peer_med)]
    for vals in (raw, peer_med, spec):
        if not all(math.isfinite(x) for x in vals):
            raise FloatingPointError("Non-finite logit encountered")
    return raw, peer_med, spec, peer_mat


def summarize_b19_metric(
    unp: list[float],
    pat: list[float],
    shm: list[float],
    n_draws: int = N_NULL_DRAWS,
) -> dict[str, Any]:
    """Compute exact same-dose drop `D = unp - pat`, centered even/odd, sham max, and signed null draws."""
    if not (len(unp) == len(pat) == len(shm) == 3 + 4 * n_draws):
        raise ValueError(f"Expected B={3 + 4 * n_draws}, got {len(unp)}, {len(pat)}, {len(shm)}")
    d_row = [u - p for u, p in zip(unp, pat)]
    d_shm = [u - s for u, s in zip(unp, shm)]
    d0 = d_row[0]
    d_pos = d_row[1]
    d_neg = d_row[2]
    even = 0.5 * (d_pos + d_neg) - d0
    odd = 0.5 * (d_pos - d_neg)
    max_abs_sham = max(abs(x) for x in d_shm)

    def pack_null(start_pos: int) -> dict[str, Any]:
        start_neg = start_pos + n_draws
        dp_list = [d_row[start_pos + k] for k in range(n_draws)]
        dn_list = [d_row[start_neg + k] for k in range(n_draws)]
        ev_list = [0.5 * (a + b) - d0 for a, b in zip(dp_list, dn_list)]
        od_list = [0.5 * (a - b) for a, b in zip(dp_list, dn_list)]
        return {
            "d_pos_draws": dp_list,
            "d_neg_draws": dn_list,
            "even_draws": ev_list,
            "odd_draws": od_list,
            "d_pos_med": float(statistics.median(dp_list)),
            "d_neg_med": float(statistics.median(dn_list)),
            "even_med": float(statistics.median(ev_list)),
            "odd_med": float(statistics.median(od_list)),
            "even_max_abs": max(abs(x) for x in ev_list),
            "odd_max_abs": max(abs(x) for x in od_list),
        }

    return {
        "d0": d0,
        "d_pos": d_pos,
        "d_neg": d_neg,
        "even": even,
        "odd": odd,
        "max_abs_sham": max_abs_sham,
        "wdown_orth": pack_null(3),
        "iso_resid": pack_null(3 + 2 * n_draws),
    }


def summarize_unpatched_transport(
    raw: list[float],
    peer_med: list[float],
    spec: list[float],
    n_draws: int = N_NULL_DRAWS,
) -> dict[str, Any]:
    """Compute Gate 2 final-logit transport positive control `N_write = R(+d) - R(0)` and signed null draws."""
    return {
        "n_write_spec": spec[1] - spec[0],
        "n_write_raw": raw[1] - raw[0],
        "n_write_peer_med": peer_med[1] - peer_med[0],
        "n_write_neg_spec": spec[2] - spec[0],
        "n_write_neg_raw": raw[2] - raw[0],
        "wdown_pos_draws_spec": [spec[3 + k] - spec[0] for k in range(n_draws)],
        "wdown_pos_draws_raw": [raw[3 + k] - raw[0] for k in range(n_draws)],
        "iso_pos_draws_spec": [spec[3 + 2 * n_draws + k] - spec[0] for k in range(n_draws)],
        "iso_pos_draws_raw": [raw[3 + 2 * n_draws + k] - raw[0] for k in range(n_draws)],
    }


def evaluate_cell_transport(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate Gate 2 cell-wise final-logit transport positive control on one cell's unpatched rows."""
    if not rows:
        return {"passed": False, "n": 0, "reason": "empty_cell"}
    n_spec = float(statistics.mean(r["transport"]["n_write_spec"] for r in rows))
    n_raw = float(statistics.mean(r["transport"]["n_write_raw"] for r in rows))
    null_draw_means = [
        float(statistics.mean(r["transport"]["wdown_pos_draws_spec"][k] for r in rows))
        for k in range(N_NULL_DRAWS)
    ]
    null_max = max(null_draw_means)
    passed = (n_spec >= TRANSPORT_MIN) and (n_raw >= TRANSPORT_MIN) and (n_spec > null_max)
    return {
        "passed": passed,
        "n": len(rows),
        "n_write_spec_mean": n_spec,
        "n_write_raw_mean": n_raw,
        "wdown_null_draw_means_spec": null_draw_means,
        "wdown_null_max_spec": null_max,
    }


def select_calibrated_dose(
    cal_unpatched_rows: list[dict[str, Any]],
    cal_head_rows: list[dict[str, Any]],
    dose_grid: tuple[float, ...] = CAL_DOSE_GRID,
) -> dict[str, Any]:
    """Deterministic pre-rebound Arm B calibration selector (Alden #54, #58 Fix 2).

    - Gate precedence at each dose `d`: Gate 1 (sham <= 1e-4) -> Gate 2 (cell-wise `CAL_SELF` &
      `CAL_THIRD` transport >= +0.50 on spec & raw AND > max_k wdown null) -> Gate 3 (`CAL_THIRD`
      `L8H15` null <= 0.15).
    - Objective `f_s(d)` uses ONLY `CAL_THIRD` `L8H15` `D(+d)` in `SpecLogit`.
    - Tie-break: ascending scan over `G_valid`, updating only on strict `<` (selects smallest `d`).
    - Rebound bracketed iff `f_s(d_min) <= -0.25` AND some `d_hi > d_min` in `G_valid` satisfies
      `f_s(d_hi) - f_s(d_min) >= +0.25`. Unbracketed monotone curves strictly return `d_cal = None`
      (`uninformative_no_rebound_bracket`) while retaining `d_best_observed` descriptively.
    """
    sorted_grid = tuple(sorted(dose_grid))
    per_dose: dict[str, Any] = {}
    g_valid: list[float] = []
    f_by_dose: dict[float, float] = {}

    for d in sorted_grid:
        d_key = f"{d:.1f}"
        u_self = [r for r in cal_unpatched_rows if r["cell"] == "CAL_SELF" and abs(r["dose_mag"] - d) < 1e-9]
        u_third = [r for r in cal_unpatched_rows if r["cell"] == "CAL_THIRD" and abs(r["dose_mag"] - d) < 1e-9]
        h_all = [r for r in cal_head_rows if abs(r["dose_mag"] - d) < 1e-9]
        h_l8h15 = [r for r in h_all if r["head_name"] == "L8H15" and r["cell"] == "CAL_THIRD"]

        # Gate 1: sham across all CAL head rows at dose d
        if not h_all or not h_l8h15:
            per_dose[d_key] = {"dose": d, "gate_status": "invalidated_missing_rows", "eligible": False}
            continue
        max_sham = max(max(r["spec"]["max_abs_sham"], r["raw"]["max_abs_sham"]) for r in h_all)
        sham_ok = max_sham <= SHAM_TOL

        # Gate 2: cell-wise transport on CAL_SELF and CAL_THIRD individually
        tr_self = evaluate_cell_transport(u_self)
        tr_third = evaluate_cell_transport(u_third)
        transport_ok = tr_self["passed"] and tr_third["passed"]

        # Gate 3: CAL_THIRD L8H15 null
        mean_ev_null = float(statistics.mean(r["spec"]["wdown_orth"]["even_med"] for r in h_l8h15))
        mean_od_null = float(statistics.mean(r["spec"]["wdown_orth"]["odd_med"] for r in h_l8h15))
        null_ok = (abs(mean_ev_null) <= NULL_TOL) and (abs(mean_od_null) <= NULL_TOL)

        # Strict precedence 1 -> 2 -> 3
        if not sham_ok:
            gate_status = "invalidated"
        elif not transport_ok:
            gate_status = "uninformative"
        elif not null_ok:
            gate_status = "unreadable"
        else:
            gate_status = "valid"

        f_val = float(statistics.mean(r["spec"]["d_pos"] for r in h_l8h15))
        f_raw = float(statistics.mean(r["raw"]["d_pos"] for r in h_l8h15))
        f_peer = float(statistics.mean(r["peer_med"]["d_pos"] for r in h_l8h15))
        per_dose[d_key] = {
            "dose": d,
            "gate_status": gate_status,
            "eligible": gate_status == "valid",
            "max_abs_sham": max_sham,
            "transport_cal_self": tr_self,
            "transport_cal_third": tr_third,
            "l8h15_null_even_mean_spec": mean_ev_null,
            "l8h15_null_odd_mean_spec": mean_od_null,
            "f_cal_third_spec": f_val,
            "f_cal_third_raw": f_raw,
            "f_cal_third_peer_med": f_peer,
        }
        if gate_status == "valid":
            g_valid.append(d)
            f_by_dose[d] = f_val

    if not g_valid:
        return {
            "cal_status": "uninformative_gate_failed",
            "d_cal": None,
            "d_best_observed": None,
            "rebound_d_hi": None,
            "rebound_delta": None,
            "g_valid": [],
            "per_dose": per_dose,
        }

    # Ascending scan, strict `<` tie-break -> smallest dose wins ties.
    d_min = g_valid[0]
    for d in g_valid[1:]:
        if f_by_dose[d] < f_by_dose[d_min]:
            d_min = d
    f_min = f_by_dose[d_min]

    if f_min > CAL_SIGNAL_MAX:
        return {
            "cal_status": "uninformative_no_cal_signal",
            "d_cal": None,
            "d_best_observed": d_min,
            "f_best_observed": f_min,
            "rebound_d_hi": None,
            "rebound_delta": None,
            "g_valid": g_valid,
            "per_dose": per_dose,
        }

    higher = [d for d in g_valid if d > d_min]
    rebound_candidates = [(d, f_by_dose[d] - f_min) for d in higher if (f_by_dose[d] - f_min) >= CAL_REBOUND_MIN]
    if rebound_candidates:
        d_hi, reb_delta = rebound_candidates[0]
        return {
            "cal_status": "rebound_bracketed",
            "d_cal": d_min,
            "d_best_observed": d_min,
            "f_best_observed": f_min,
            "rebound_d_hi": d_hi,
            "rebound_delta": reb_delta,
            "g_valid": g_valid,
            "per_dose": per_dose,
        }

    return {
        "cal_status": "uninformative_no_rebound_bracket",
        "d_cal": None,
        "d_best_observed": d_min,
        "f_best_observed": f_min,
        "rebound_d_hi": None,
        "rebound_delta": None,
        "g_valid": g_valid,
        "per_dose": per_dose,
    }


# ---------------------------------------------------------------------------
# Model hooks & stage execution
# ---------------------------------------------------------------------------

def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Unsupported model architecture: cannot find decoder layers")


def compute_stage_cal_medians(
    model: nn.Module,
    manifest: dict[str, Any],
    device: str,
    units: tuple[str, ...] = CORE_UNITS,
) -> dict[int, float]:
    """Compute frozen stage-own layer residual L2 medians across all tokens of the 4 disjoint CAL prompts."""
    layers_needed = sorted({SLOT_BY_ID[u]["layer"] for u in units})
    layers_mod = get_decoder_layers(model)
    cal_prompts = [p for p in manifest["prompts"] if p["bank"] == "CAL"]
    # Use `p92-02` prompt IDs for each CAL prompt (all 4 CAL prompts; 1 pass per prompt).
    # Wait: to be 100% unit-symmetric across the 4 CAL prompts, run all 4 CAL prompts with `p92-02`?
    # Better: run the exact 4 `CAL` base prompts (each core unit adds 1 lead token `" the"` or `" a"`;
    # using `p92-02` vs `p92-05` differs only by the final lead token). To follow §2.2 literally
    # ("on the 4 disjoint CAL prompts"), use the 4 CAL prompts at `p92-02`'s exact sequence? Even cleaner:
    # pool all 4 CAL prompts' `base_text` token IDs (identical across all 5 units!) so `m_{s, L_u}` is
    # 100% unit-agnostic! Let's record the exact method in meta.
    norms_by_layer: dict[int, list[float]] = {L: [] for L in layers_needed}
    handles = []

    def make_hook(L: int):
        def _hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            nrm = torch.linalg.vector_norm(hs.detach().to(torch.float32), dim=-1).reshape(-1).cpu().tolist()
            norms_by_layer[L].extend(float(x) for x in nrm)
        return _hook

    try:
        for L in layers_needed:
            handles.append(layers_mod[L].register_forward_hook(make_hook(L)))
        with torch.no_grad():
            for p in cal_prompts:
                # Use p92-02 binding input_ids[:-1] == exact base_text token IDs of the 4 CAL prompts
                ids = manifest["bindings"][p["prompt_id"]]["p92-02"]["input_ids"][:-1]
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                model(input_ids=inp, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {L: float(statistics.median(norms_by_layer[L])) for L in layers_needed}


def extract_stage_unit_vectors(
    model: nn.Module,
    medians: dict[int, float],
    units: tuple[str, ...] = CORE_UNITS,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    layers_mod = get_decoder_layers(model)
    v_by_unit: dict[str, torch.Tensor] = {}
    vec_meta: dict[str, dict[str, Any]] = {}
    for uid in units:
        u = SLOT_BY_ID[uid]
        L = u["layer"]
        n_idx = u["neuron"]
        w_col = layers_mod[L].mlp.down_proj.weight[:, n_idx].detach().to(torch.float32)
        w_norm = float(torch.linalg.vector_norm(w_col).item())
        hat_w = w_col / max(w_norm, 1e-12)
        m_val = medians[L]
        v_by_unit[uid] = (hat_w * m_val).to(w_col.device, dtype=torch.float32)
        vec_meta[uid] = {
            "layer": L,
            "neuron": n_idx,
            "w_down_norm": w_norm,
            "cal_resid_median": m_val,
            "hat_w": [float(x) for x in hat_w.cpu().tolist()],
        }
    return v_by_unit, vec_meta


def run_b19_tuple(
    model: nn.Module,
    stage: str,
    uid: str,
    prompt_spec: dict[str, Any],
    binding: dict[str, Any],
    dose_mag: float,
    v_u: torch.Tensor,
    heads_for_unit: list[dict[str, Any]],
    record_peer_deltas: bool,
    device: str,
    timer_acc: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run unpatched + all eligible heads (`unpushed_z` & 100% `sham_z`) for one `(unit, prompt, dose)`."""
    u = SLOT_BY_ID[uid]
    u_layer = u["layer"]
    n_idx = u["neuron"]
    pid = prompt_spec["prompt_id"]
    input_ids = binding["input_ids"]
    read_pos = binding["read_pos"]
    t_own = binding["t_own"]
    peers = binding["peers"]

    layers_mod = get_decoder_layers(model)
    w_down = layers_mod[u_layer].mlp.down_proj.weight
    seed = unit_seed(stage, uid, pid, dose_mag)
    wdown_orth, iso_resid = make_orthogonal_null_vectors(
        w_down_weight=w_down,
        neuron_idx=n_idx,
        v_u=v_u,
        n_draws=N_NULL_DRAWS,
        seed=seed,
    )

    B = 3 + 4 * N_NULL_DRAWS  # 19
    d_model = v_u.shape[0]
    deltas = torch.zeros((B, d_model), device=device, dtype=torch.float32)
    deltas[1] = dose_mag * v_u
    deltas[2] = -dose_mag * v_u
    deltas[3 : 3 + N_NULL_DRAWS] = dose_mag * wdown_orth
    deltas[3 + N_NULL_DRAWS : 3 + 2 * N_NULL_DRAWS] = -dose_mag * wdown_orth
    deltas[3 + 2 * N_NULL_DRAWS : 3 + 3 * N_NULL_DRAWS] = dose_mag * iso_resid
    deltas[3 + 3 * N_NULL_DRAWS : 3 + 4 * N_NULL_DRAWS] = -dose_mag * iso_resid

    inp = torch.tensor([input_ids] * B, dtype=torch.long, device=device)

    # Hook 1: inject deltas at layer `u_layer` output (all token positions), capture unpushed read_pos norm.
    read_pos_norm_box: list[float] = []

    def inject_hook(_mod, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            if not read_pos_norm_box:
                read_pos_norm_box.append(float(torch.linalg.vector_norm(hs[0, read_pos].detach().to(torch.float32)).item()))
            patched_hs = hs + deltas.unsqueeze(1).to(hs.dtype)
            return (patched_hs,) + out[1:]
        if not read_pos_norm_box:
            read_pos_norm_box.append(float(torch.linalg.vector_norm(out[0, read_pos].detach().to(torch.float32)).item()))
        return out + deltas.unsqueeze(1).to(out.dtype)

    # Capture pre-o_proj activations for all unique head layers needed by `heads_for_unit`.
    head_layers = sorted({hs["layer"] for hs in heads_for_unit})
    captured_o_in: dict[int, torch.Tensor] = {}

    def make_capture_pre_hook(L: int):
        def _pre(_mod, args):
            captured_o_in[L] = args[0].detach().clone()
            return None
        return _pre

    t0 = time.perf_counter()
    handles = [layers_mod[u_layer].register_forward_hook(inject_hook)]
    for hl in head_layers:
        handles.append(layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(make_capture_pre_hook(hl)))
    try:
        with torch.no_grad():
            out_unp = model(input_ids=inp, use_cache=False).logits[:, read_pos, :]
    finally:
        for h in handles:
            h.remove()
    timer_acc["b19_sec"] += time.perf_counter() - t0
    timer_acc["b19_fwds"] += 1

    unp_raw, unp_pmed, unp_spec, unp_pmat = read_decomposed_logits(out_unp, t_own, peers)
    injected_norm = float(torch.linalg.vector_norm(deltas[1]).item())
    rp_norm = read_pos_norm_box[0]

    unpatched_row = {
        "stage": stage,
        "unit": uid,
        "unit_layer": u_layer,
        "prompt_id": pid,
        "bank": prompt_spec["bank"],
        "cell": prompt_spec["cell"],
        "story": prompt_spec["story"],
        "serialization": prompt_spec["serialization"],
        "dose_mag": dose_mag,
        "injected_norm": injected_norm,
        "read_pos_resid_norm": rp_norm,
        "norm_ratio": injected_norm / max(rp_norm, 1e-12),
        "transport": summarize_unpatched_transport(unp_raw, unp_pmed, unp_spec),
    }

    if not heads_for_unit:
        return unpatched_row, []

    # Determine head_dim from captured tensor and model config
    n_heads = int(getattr(model.config, "num_attention_heads"))
    head_rows: list[dict[str, Any]] = []

    for hs in heads_for_unit:
        hl = hs["layer"]
        hidx = hs["head"]
        assert_strictly_downstream(u_layer, hl)
        z_self_full = captured_o_in[hl]  # [B, T, n_heads * head_dim]
        head_dim = z_self_full.shape[-1] // n_heads
        sl = slice(hidx * head_dim, (hidx + 1) * head_dim)
        z_donor_slice = z_self_full[0:1, :, sl].expand(B, -1, -1).clone()
        z_self_slice = z_self_full[:, :, sl].clone()

        def run_with_o_proj_patch(replacement_slice: torch.Tensor) -> torch.Tensor:
            def _patch_pre(_mod, args):
                x = args[0].clone()
                x[:, :, sl] = replacement_slice.to(x.dtype)
                return (x,) + args[1:]

            h_inj = layers_mod[u_layer].register_forward_hook(inject_hook)
            h_pat = layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(_patch_pre)
            t1 = time.perf_counter()
            try:
                with torch.no_grad():
                    logits = model(input_ids=inp, use_cache=False).logits[:, read_pos, :]
            finally:
                h_inj.remove()
                h_pat.remove()
            timer_acc["b19_sec"] += time.perf_counter() - t1
            timer_acc["b19_fwds"] += 1
            return logits

        out_pat = run_with_o_proj_patch(z_donor_slice)
        out_shm = run_with_o_proj_patch(z_self_slice)

        pat_raw, pat_pmed, pat_spec, pat_pmat = read_decomposed_logits(out_pat, t_own, peers)
        shm_raw, shm_pmed, shm_spec, _ = read_decomposed_logits(out_shm, t_own, peers)

        spec_sum = summarize_b19_metric(unp_spec, pat_spec, shm_spec)
        raw_sum = summarize_b19_metric(unp_raw, pat_raw, shm_raw)
        pmed_sum = summarize_b19_metric(unp_pmed, pat_pmed, shm_pmed)

        # Verify exact algebraic identity D_spec == D_raw - D_peer_med
        if abs(spec_sum["d_pos"] - (raw_sum["d_pos"] - pmed_sum["d_pos"])) > 1e-6:
            raise AssertionError("Spec/Raw/PeerMedian algebraic identity violated")

        row: dict[str, Any] = {
            "stage": stage,
            "unit": uid,
            "unit_layer": u_layer,
            "prompt_id": pid,
            "bank": prompt_spec["bank"],
            "cell": prompt_spec["cell"],
            "story": prompt_spec["story"],
            "serialization": prompt_spec["serialization"],
            "dose_mag": dose_mag,
            "head_name": hs["name"],
            "head_layer": hl,
            "head_idx": hidx,
            "head_kind": hs["kind"],
            "spec": spec_sum,
            "raw": raw_sum,
            "peer_med": pmed_sum,
        }
        if record_peer_deltas:
            p_deltas: dict[str, dict[str, float]] = {}
            dpos_vals: list[float] = []
            for j, tid in enumerate(peers):
                dp_j = unp_pmat[1][j] - pat_pmat[1][j]
                dn_j = unp_pmat[2][j] - pat_pmat[2][j]
                p_deltas[str(tid)] = {"d_pos": dp_j, "d_neg": dn_j}
                dpos_vals.append(dp_j)
            s_vals = sorted(dpos_vals)
            q75 = s_vals[(3 * (len(s_vals) - 1)) // 4]
            q25 = s_vals[(len(s_vals) - 1) // 4]
            row["peer_deltas"] = p_deltas
            row["peer_mean_d_pos"] = float(statistics.mean(dpos_vals))
            row["peer_iqr_d_pos"] = float(q75 - q25)
        head_rows.append(row)

    return unpatched_row, head_rows


def execute_stage_run(
    stage: str,
    manifest_path: Path,
    design_path: Path,
    out_path: Path,
    device: str = "cuda",
    smoke_1unit: bool = False,
    expected_manifest_sha256: str | None = None,
    expected_design_sha256: str | None = None,
    expected_runner_sha256: str | None = None,
    expected_scorer_sha256: str | None = None,
    scorer_path: Path | None = None,
    tokenizer_loader: Callable[[str, str], Any] = default_tokenizer_loader,
    model_loader: Callable[[str, str, str], Any] = default_model_loader,
) -> dict[str, Any]:
    if stage not in STAGE_SPECS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    spec = STAGE_SPECS[stage]
    validate_commit_sha(spec["commit"])

    # Mandatory pre-load 4-pin admission enforcement (ZERO model weights loaded if this raises!)
    manifest, pins = admit_frozen_package(
        manifest_path=manifest_path,
        design_path=design_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_design_sha256=expected_design_sha256,
        expected_runner_sha256=expected_runner_sha256,
        expected_scorer_sha256=expected_scorer_sha256,
        scorer_path=scorer_path,
        tokenizer_loader=tokenizer_loader,
    )
    print(
        f"[ss] Admitted frozen 4-pin package (manifest={pins['manifest_sha256'][:12]}, "
        f"design={pins['design_sha256'][:12]}, runner={pins['runner_sha256'][:12]}, "
        f"scorer={pins['scorer_sha256'][:12]}). Loading {spec['repo']}@{spec['commit'][:12]} on {device}...",
        flush=True,
    )

    t_total0 = time.perf_counter()
    model = model_loader(spec["repo"], spec["commit"], device)
    units_to_run = (SMOKE_UNIT,) if smoke_1unit else CORE_UNITS
    cal_doses = SMOKE_CAL_DOSES if smoke_1unit else CAL_DOSE_GRID

    medians = compute_stage_cal_medians(model, manifest, device=device, units=units_to_run)
    v_by_unit, vec_meta = extract_stage_unit_vectors(model, medians, units=units_to_run)

    prompt_by_id = {p["prompt_id"]: p for p in manifest["prompts"]}
    cal_prompts = [p for p in manifest["prompts"] if p["bank"] == "CAL"]
    eval_prompts = (
        [prompt_by_id[pid] for pid in SMOKE_EVAL_PROMPTS]
        if smoke_1unit
        else [p for p in manifest["prompts"] if p["bank"] == "EVAL"]
    )
    sens_prompts = (
        [prompt_by_id[pid] for pid in SMOKE_SENS_PROMPTS]
        if smoke_1unit
        else [p for p in manifest["prompts"] if p["bank"] == "SENS_PLAIN"]
    )

    timer_acc = {"b19_sec": 0.0, "b19_fwds": 0}
    cal_unpatched_rows: list[dict[str, Any]] = []
    cal_head_rows: list[dict[str, Any]] = []

    # Block 1: CAL (unpatched on CAL_SELF + CAL_THIRD; patched+sham on CAL_THIRD)
    for uid in units_to_run:
        for d in cal_doses:
            for ps in cal_prompts:
                heads = list(HEAD_SPECS) if ps["cell"] == "CAL_THIRD" else []
                urow, hrows = run_b19_tuple(
                    model=model,
                    stage=stage,
                    uid=uid,
                    prompt_spec=ps,
                    binding=manifest["bindings"][ps["prompt_id"]][uid],
                    dose_mag=d,
                    v_u=v_by_unit[uid],
                    heads_for_unit=heads,
                    record_peer_deltas=False,
                    device=device,
                    timer_acc=timer_acc,
                )
                cal_unpatched_rows.append(urow)
                cal_head_rows.extend(hrows)
        print(
            f"[ss:{stage}] CAL done for {uid} (fwds={timer_acc['b19_fwds']} "
            f"@ {1000.0 * timer_acc['b19_sec'] / max(timer_acc['b19_fwds'], 1):.2f} ms/fwd)",
            flush=True,
        )

    cal_summary = select_calibrated_dose(cal_unpatched_rows, cal_head_rows, dose_grid=cal_doses)
    print(
        f"[ss:{stage}] CAL summary: status={cal_summary['cal_status']} "
        f"d_cal={cal_summary['d_cal']} d_best_observed={cal_summary['d_best_observed']}",
        flush=True,
    )

    # Block 2: EVAL Arm A (always run at COMMON_DOSE = 0.8)
    eval_arm_a_unpatched_rows: list[dict[str, Any]] = []
    eval_arm_a_head_rows: list[dict[str, Any]] = []
    for uid in units_to_run:
        for ps in eval_prompts:
            urow, hrows = run_b19_tuple(
                model=model,
                stage=stage,
                uid=uid,
                prompt_spec=ps,
                binding=manifest["bindings"][ps["prompt_id"]][uid],
                dose_mag=COMMON_DOSE,
                v_u=v_by_unit[uid],
                heads_for_unit=list(HEAD_SPECS),
                record_peer_deltas=True,
                device=device,
                timer_acc=timer_acc,
            )
            eval_arm_a_unpatched_rows.append(urow)
            eval_arm_a_head_rows.extend(hrows)
        print(
            f"[ss:{stage}] EVAL Arm A done for {uid} (fwds={timer_acc['b19_fwds']} "
            f"@ {1000.0 * timer_acc['b19_sec'] / max(timer_acc['b19_fwds'], 1):.2f} ms/fwd)",
            flush=True,
        )

    # Block 3: EVAL Arm B (strictly ONLY when rebound_bracketed; reuses Arm A if d_cal == 0.8)
    eval_arm_b_unpatched_rows: list[dict[str, Any]] = []
    eval_arm_b_head_rows: list[dict[str, Any]] = []
    d_cal = cal_summary["d_cal"]
    arm_b_reuses_arm_a = bool(
        cal_summary["cal_status"] == "rebound_bracketed"
        and d_cal is not None
        and abs(d_cal - COMMON_DOSE) < 1e-12
    )
    if cal_summary["cal_status"] == "rebound_bracketed" and d_cal is not None and not arm_b_reuses_arm_a:
        for uid in units_to_run:
            for ps in eval_prompts:
                urow, hrows = run_b19_tuple(
                    model=model,
                    stage=stage,
                    uid=uid,
                    prompt_spec=ps,
                    binding=manifest["bindings"][ps["prompt_id"]][uid],
                    dose_mag=float(d_cal),
                    v_u=v_by_unit[uid],
                    heads_for_unit=list(HEAD_SPECS),
                    record_peer_deltas=True,
                    device=device,
                    timer_acc=timer_acc,
                )
                eval_arm_b_unpatched_rows.append(urow)
                eval_arm_b_head_rows.extend(hrows)
            print(
                f"[ss:{stage}] EVAL Arm B (d={d_cal}) done for {uid} (fwds={timer_acc['b19_fwds']})",
                flush=True,
            )

    # Block 4: SENS_PLAIN at COMMON_DOSE = 0.8
    sens_plain_unpatched_rows: list[dict[str, Any]] = []
    sens_plain_head_rows: list[dict[str, Any]] = []
    for uid in units_to_run:
        for ps in sens_prompts:
            urow, hrows = run_b19_tuple(
                model=model,
                stage=stage,
                uid=uid,
                prompt_spec=ps,
                binding=manifest["bindings"][ps["prompt_id"]][uid],
                dose_mag=COMMON_DOSE,
                v_u=v_by_unit[uid],
                heads_for_unit=list(HEAD_SPECS),
                record_peer_deltas=False,
                device=device,
                timer_acc=timer_acc,
            )
            sens_plain_unpatched_rows.append(urow)
            sens_plain_head_rows.extend(hrows)

    total_elapsed = time.perf_counter() - t_total0
    ms_per_b19 = 1000.0 * timer_acc["b19_sec"] / max(timer_acc["b19_fwds"], 1)

    receipt = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "model_repo": spec["repo"],
            "model_commit": spec["commit"],
            "dtype": "float32",
            "attn_implementation": "eager",
            "artifact_type": "stage_smoke_receipt" if smoke_1unit else "stage_full_receipt",
            "run_mode": "smoke_1unit" if smoke_1unit else "full",
            "complete": True,
            "smoke_1unit": smoke_1unit,
            "units_completed": list(units_to_run),
            "cal_doses_run": list(cal_doses),
            "common_dose": COMMON_DOSE,
            "arm_b_reuses_arm_a": arm_b_reuses_arm_a,
            "cal_resid_medians": {str(k): v for k, v in medians.items()},
            "manifest_sha256": pins["manifest_sha256"],
            "design_sha256": pins["design_sha256"],
            "runner_sha256": pins["runner_sha256"],
            "scorer_sha256": pins["scorer_sha256"],
            "pins": pins,
            "timing": {
                "total_elapsed_sec": total_elapsed,
                "b19_elapsed_sec": timer_acc["b19_sec"],
                "n_b19_done": timer_acc["b19_fwds"],
                "ms_per_b19": ms_per_b19,
                "projected_full_stage_min_sec": 2760 * (ms_per_b19 / 1000.0),
                "projected_full_stage_max_sec": 4260 * (ms_per_b19 / 1000.0),
            },
        },
        "unit_vectors": vec_meta,
        "cal_summary": cal_summary,
        "cal_unpatched_rows": cal_unpatched_rows,
        "cal_head_rows": cal_head_rows,
        "eval_arm_a_unpatched_rows": eval_arm_a_unpatched_rows,
        "eval_arm_a_head_rows": eval_arm_a_head_rows,
        "eval_arm_b_unpatched_rows": eval_arm_b_unpatched_rows,
        "eval_arm_b_head_rows": eval_arm_b_head_rows,
        "sens_plain_unpatched_rows": sens_plain_unpatched_rows,
        "sens_plain_head_rows": sens_plain_head_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return receipt


# ---------------------------------------------------------------------------
# CPU Self-Tests (Fake Tokenizer, Fake Model Loader, Sabotages & Admissions)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Deterministic word/special-token tokenizer matching v2.1 boundary widths for CPU tests."""
    def __init__(self, shift_id: int = 0, vocab_extra: int = 0):
        self.shift_id = shift_id
        self.vocab_extra = vocab_extra
        self._tok2id: dict[str, int] = {}
        self._id2tok: dict[int, str] = {}
        self._next_id = 100

    def _get(self, piece: str) -> int:
        if piece not in self._tok2id:
            tid = self._next_id + self.shift_id
            self._next_id += 1
            self._tok2id[piece] = tid
            self._id2tok[tid] = piece
        return self._tok2id[piece]

    def __len__(self) -> int:
        return 100278 + self.vocab_extra

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = re.findall(r"<\|[a-z]+\|>|\n| ?[A-Za-z0-9']+|[.,!?]", text)
        return [self._get(p) for p in pieces]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2tok[i] for i in ids)


class _DummySelfAttn(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.o_proj = nn.Linear(d_model, d_model, bias=False)


class _DummyMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)


class _DummyLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.self_attn = _DummySelfAttn(d_model)
        self.mlp = _DummyMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        h = x + 0.05 * self.self_attn.o_proj(x)
        return (h,)


class _DummyConfig:
    num_attention_heads = 16


class _DummyModel(nn.Module):
    def __init__(self, vocab: int = 2000, d_model: int = 32, d_ff: int = 7500, n_layers: int = 10):
        super().__init__()
        torch.manual_seed(1234)
        self.config = _DummyConfig()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([_DummyLayer(d_model, d_ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        x = self.emb(input_ids % self.emb.num_embeddings)
        for layer in self.layers:
            x = layer(x)[0]
        logits = self.lm_head(x)
        return type("Out", (), {"logits": logits})()


def run_self_tests() -> None:
    print("[self-test] Running run_stage_sweep.py CPU verification suite...")

    # 1. Commit validation sabotages
    for bad in ("main", "a1847dff", "1234", "g" * 40):
        try:
            validate_commit_sha(bad)
            raise AssertionError(f"Should have refused bad commit {bad!r}")
        except ValueError:
            pass

    # 2. Non-causal head sabotage
    for ul, hl in ((5, 5), (6, 5), (6, 6)):
        try:
            assert_strictly_downstream(ul, hl)
            raise AssertionError(f"Should have refused non-causal ({ul}, {hl})")
        except ValueError:
            pass
    assert_strictly_downstream(6, 7)
    assert len(eligible_pairs()) == 35

    # 3. Signed null negation & sham zero invariance test
    w_dummy = torch.randn(32, 8000)
    v_dummy = torch.randn(32)
    wd, iso = make_orthogonal_null_vectors(w_dummy, 2628, v_dummy, N_NULL_DRAWS, seed=42)
    # Orthogonal to v_u and exact norm match
    cos_wd = torch.abs((wd @ v_dummy) / (torch.linalg.vector_norm(wd, dim=-1) * torch.linalg.vector_norm(v_dummy)))
    assert float(cos_wd.max()) < 1e-5, f"wdown_orth not orthogonal: {cos_wd}"
    assert torch.allclose(torch.linalg.vector_norm(wd, dim=-1), torch.linalg.vector_norm(v_dummy).expand(N_NULL_DRAWS), atol=1e-4)
    # Zeroed neuron column check: perturbing column 2628 must not change wd
    w_dummy2 = w_dummy.clone()
    w_dummy2[:, 2628] += 999.0
    wd2, _ = make_orthogonal_null_vectors(w_dummy2, 2628, v_dummy, N_NULL_DRAWS, seed=42)
    assert torch.allclose(wd, wd2, atol=1e-6), "Target column was not strictly zeroed in wdown_orth!"

    # 4. Arm B calibration unit tests (monotone witness, tie-break, cell-wise transport loophole)
    def synth_cal(f_map: dict[float, float], tr_self_map=None, tr_third_map=None, sham_bad=None):
        u_rows, h_rows = [], []
        for d, fv in f_map.items():
            ts = 1.0 if tr_self_map is None else tr_self_map[d]
            tt = 1.0 if tr_third_map is None else tr_third_map[d]
            u_rows.append({
                "cell": "CAL_SELF", "dose_mag": d,
                "transport": {"n_write_spec": ts, "n_write_raw": ts, "wdown_pos_draws_spec": [0.05] * N_NULL_DRAWS},
            })
            u_rows.append({
                "cell": "CAL_THIRD", "dose_mag": d,
                "transport": {"n_write_spec": tt, "n_write_raw": tt, "wdown_pos_draws_spec": [0.05] * N_NULL_DRAWS},
            })
            sh = 1e-2 if sham_bad == d else 0.0
            h_rows.append({
                "cell": "CAL_THIRD", "dose_mag": d, "head_name": "L8H15",
                "spec": {"d_pos": fv, "max_abs_sham": sh, "wdown_orth": {"even_med": 0.01, "odd_med": -0.01}},
                "raw": {"d_pos": fv * 0.5, "max_abs_sham": sh},
                "peer_med": {"d_pos": -fv * 0.5, "max_abs_sham": sh},
            })
        return u_rows, h_rows

    # 4a. Alden #58 monotone witness [-0.30, -0.35, -0.40, -0.45, -0.50, -0.55] -> Arm B d_cal=None!
    u_m, h_m = synth_cal({0.2: -0.30, 0.4: -0.35, 0.6: -0.40, 0.8: -0.45, 1.2: -0.50, 1.6: -0.55})
    res_m = select_calibrated_dose(u_m, h_m)
    assert res_m["cal_status"] == "uninformative_no_rebound_bracket", res_m
    assert res_m["d_cal"] is None and res_m["d_best_observed"] == 1.6

    # 4b. Rebound bracketed + exact smallest-d tie-break (0.6 and 0.8 tied at -0.70, rebound at 1.6 -> +0.10)
    u_t, h_t = synth_cal({0.2: -0.10, 0.4: -0.30, 0.6: -0.70, 0.8: -0.70, 1.2: -0.50, 1.6: +0.10})
    res_t = select_calibrated_dose(u_t, h_t)
    assert res_t["cal_status"] == "rebound_bracketed" and res_t["d_cal"] == 0.6, res_t

    # 4c. Alden #58 cell-wise transport loophole: CAL_SELF=0.0, CAL_THIRD=1.1 (pooled +0.55) MUST fail!
    u_p, h_p = synth_cal(
        {0.2: -0.1, 0.4: -0.3, 0.6: -0.7, 0.8: -0.6, 1.2: -0.2, 1.6: +0.1},
        tr_self_map={d: 0.0 for d in CAL_DOSE_GRID},
        tr_third_map={d: 1.1 for d in CAL_DOSE_GRID},
    )
    res_p = select_calibrated_dose(u_p, h_p)
    assert res_p["cal_status"] == "uninformative_gate_failed" and res_p["d_cal"] is None, res_p

    # 5. Pre-load 4-pin admission & fake-loader suite (zero loader calls on any refusal!)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        design_p = Path("explore_olmo2_attn_mlp/STAGE-SWEEP-DESIGN.md")
        runner_p = Path(__file__).resolve()
        scorer_p = runner_p.with_name("score_stage_sweep.py")
        assert design_p.is_file() and scorer_p.is_file()
        shared_tok = _FakeTokenizer()
        tok_calls = {"n": 0}
        def tok_loader_ok(repo, commit):
            tok_calls["n"] += 1
            return shared_tok

        manifest = build_manifest(design_path=design_p, tokenizer_loader=tok_loader_ok)
        assert manifest["meta"]["n_prompts"] == 28 and manifest["meta"]["n_bindings"] == 140
        m_path = tdp / "manifest.json"
        m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        m_sha = sha256_file(m_path)
        d_sha = sha256_file(design_p)
        r_sha = sha256_file(runner_p)
        s_sha = sha256_file(scorer_p)

        loader_calls = {"n": 0}
        def counting_loader(repo, commit, device):
            loader_calls["n"] += 1
            return _DummyModel()

        valid_pins = dict(
            expected_manifest_sha256=m_sha,
            expected_design_sha256=d_sha,
            expected_runner_sha256=r_sha,
            expected_scorer_sha256=s_sha,
        )

        # 5a. Complete 4-pin positive reaches loader once and produces valid stage_smoke_receipt with sham==0
        out_smoke = tdp / "smoke.json"
        rec = execute_stage_run(
            stage="instruct",
            manifest_path=m_path,
            design_path=design_p,
            out_path=out_smoke,
            device="cpu",
            smoke_1unit=True,
            tokenizer_loader=tok_loader_ok,
            model_loader=counting_loader,
            **valid_pins,
        )
        assert loader_calls["n"] == 1
        assert rec["meta"]["artifact_type"] == "stage_smoke_receipt"
        all_h = rec["cal_head_rows"] + rec["eval_arm_a_head_rows"] + rec["sens_plain_head_rows"]
        max_sh = max(max(r["spec"]["max_abs_sham"], r["raw"]["max_abs_sham"]) for r in all_h)
        assert max_sh <= 1e-6, f"Expected exact sham zero on dummy model, got {max_sh}"

        # 5b. Pre-load refusals: ALL must raise with loader_calls == 0!
        def assert_refused(label: str, expect_zero_tok: bool = False, **kwargs):
            loader_calls["n"] = 0
            tok_calls["n"] = 0
            try:
                execute_stage_run(
                    stage="instruct",
                    out_path=tdp / f"bad_{label}.json",
                    device="cpu",
                    smoke_1unit=True,
                    model_loader=counting_loader,
                    **kwargs,
                )
                raise AssertionError(f"Expected pre-load refusal for {label}")
            except (ValueError, FileNotFoundError):
                assert loader_calls["n"] == 0, f"Loader was called on refused case {label}!"
                if expect_zero_tok:
                    assert tok_calls["n"] == 0, f"Tokenizer loader was called on pin/file refusal {label}!"

        # Zero pins (Alden #69 witness 1)
        assert_refused("zero_pins", expect_zero_tok=True, manifest_path=m_path, design_path=design_p, tokenizer_loader=tok_loader_ok)
        # Missing each individual pin
        for pin_k in ("expected_manifest_sha256", "expected_design_sha256", "expected_runner_sha256", "expected_scorer_sha256"):
            kw = dict(valid_pins)
            kw[pin_k] = None
            assert_refused(f"missing_{pin_k}", expect_zero_tok=True, manifest_path=m_path, design_path=design_p, tokenizer_loader=tok_loader_ok, **kw)
        # Wrong each individual pin
        for idx_p, pin_k in enumerate(("expected_manifest_sha256", "expected_design_sha256", "expected_runner_sha256", "expected_scorer_sha256")):
            kw = dict(valid_pins)
            kw[pin_k] = str(idx_p) * 64
            assert_refused(f"wrong_{pin_k}", expect_zero_tok=True, manifest_path=m_path, design_path=design_p, tokenizer_loader=tok_loader_ok, **kw)
        # Missing manifest, missing design, and missing scorer (Alden #69 witness 2)
        assert_refused("missing_manifest", expect_zero_tok=True, manifest_path=tdp / "nope.json", design_path=design_p, tokenizer_loader=tok_loader_ok, **valid_pins)
        assert_refused("missing_design", expect_zero_tok=True, manifest_path=m_path, design_path=tdp / "nope.md", tokenizer_loader=tok_loader_ok, **valid_pins)
        assert_refused("missing_scorer", expect_zero_tok=True, manifest_path=m_path, design_path=design_p, scorer_path=tdp / "nope_scorer.py", tokenizer_loader=tok_loader_ok, **valid_pins)
        # Tampered design bytes (with matching external design pin so internal manifest check catches it)
        bad_design = tdp / "bad_design.md"
        bad_design.write_text(design_p.read_text(encoding="utf-8") + "\ntamper\n", encoding="utf-8")
        assert_refused("tampered_design", manifest_path=m_path, design_path=bad_design, tokenizer_loader=tok_loader_ok, **{**valid_pins, "expected_design_sha256": sha256_file(bad_design)})
        # Tampered 1 own token ID inside manifest (with matching external manifest pin so regeneration catches it)
        tampered_m = copy.deepcopy(manifest)
        tampered_m["bindings"]["eval_self_E1"]["p92-02"]["t_own"] += 1
        tm_path = tdp / "tampered_manifest.json"
        tm_path.write_text(json.dumps(tampered_m, indent=2), encoding="utf-8")
        assert_refused("tampered_t_own", manifest_path=tm_path, design_path=design_p, tokenizer_loader=tok_loader_ok, **{**valid_pins, "expected_manifest_sha256": sha256_file(tm_path)})
        # 4-tokenizer parity failure (sft vocab mismatch)
        bad_tok_sft = _FakeTokenizer(vocab_extra=1)
        tok_loader_mismatch = lambda repo, commit: bad_tok_sft if "SFT" in repo else shared_tok
        assert_refused("tokenizer_parity_mismatch", manifest_path=m_path, design_path=design_p, tokenizer_loader=tok_loader_mismatch, **valid_pins)

    print("[self-test] ALL run_stage_sweep.py tests PASSED (incl. 4-pin pre-load refusals, zero real weights loaded)!")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="OLMo 2 1B Stage-Sweep Runner (v2.1)")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--emit-manifest", type=Path, default=None)
    p.add_argument("--design", type=Path, default=Path("explore_olmo2_attn_mlp/STAGE-SWEEP-DESIGN.md"))
    p.add_argument("--manifest", type=Path, default=Path("explore_olmo2_attn_mlp/stage_sweep_manifest.json"))
    p.add_argument("--expected-manifest-sha256", type=str, default=None)
    p.add_argument("--expected-design-sha256", type=str, default=None)
    p.add_argument("--expected-runner-sha256", type=str, default=None)
    p.add_argument("--expected-scorer-sha256", type=str, default=None)
    p.add_argument("--stage", type=str, choices=STAGE_ORDER, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smoke-1unit", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.self_test_only:
        run_self_tests()
        return

    if args.emit_manifest is not None:
        manifest = build_manifest(design_path=args.design)
        args.emit_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.emit_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"[manifest] Wrote {args.emit_manifest} "
            f"(sha256={sha256_file(args.emit_manifest)}, "
            f"design_sha256={manifest['meta']['design_sha256']}, "
            f"prompts={manifest['meta']['n_prompts']}, bindings={manifest['meta']['n_bindings']}, "
            f"four_tokenizer_parity={manifest['meta']['four_tokenizer_parity_verified']})"
        )
        return

    missing = [
        name for name, val in (
            ("--stage", args.stage),
            ("--out", args.out),
            ("--expected-manifest-sha256", args.expected_manifest_sha256),
            ("--expected-design-sha256", args.expected_design_sha256),
            ("--expected-runner-sha256", args.expected_runner_sha256),
            ("--expected-scorer-sha256", args.expected_scorer_sha256),
        )
        if val is None
    ]
    if missing:
        raise SystemExit(f"Refused unchecked run invocation; missing required flags: {', '.join(missing)}")

    execute_stage_run(
        stage=args.stage,
        manifest_path=args.manifest,
        design_path=args.design,
        out_path=args.out,
        device=args.device,
        smoke_1unit=args.smoke_1unit,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_design_sha256=args.expected_design_sha256,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_scorer_sha256=args.expected_scorer_sha256,
    )


if __name__ == "__main__":
    main()
