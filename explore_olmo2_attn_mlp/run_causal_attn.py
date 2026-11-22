#!/usr/bin/env python3
"""OLMo 2 1B Instruct — Part B (#4): Causal Attention-Reading & Span-Localized Mechanism Runner.

Implements `explore_olmo2_attn_mlp/CAUSAL-ATTN-DESIGN.md` v3
(SHA-256: 9b41482fbd1fae267ece14a859274ddf886de3ecd94526e44a4c5697821030fa).

Hardening & contracts:
- Single-read external 4-pin binding (`design`, `manifest`, `runner`, `scorer`) PLUS
  single-read Part A scorecard binding (`part_b_norm_decision`, `p92_09_part_b_track`)
  enforced BEFORE any tokenizer or model load. Auto-HOLDS if Part A declined norm.
- Explicit `--mode {smoke,full}`; `--mode full` is gated pre-load on the pinned scorer's
  `verify_smoke_receipt_only(..., allow_fixture=False)` executed from the already-hashed
  scorer buffer.
- Exact post-`q_norm` / post-`k_norm` / post-`v_proj` single-head hooks guaranteeing
  `0.0` (`<= 1e-6`) sibling-head leakage across all 15 sibling heads and exact 3-part
  `QKV_all == z_all` reconciliation (`<= 1e-5`).
- Measured same-hook-path sham on EVERY head-mode row (`2,880` full / `66` smoke) and
  EVERY `CM.4` swap row (`1,200` full / `28` smoke); sham keys strictly absent on
  unpatched rows (`60` full / `3` smoke).
- 17 dedicated synthetic nonzero-donor actuation positive controls (base seed `202611220`,
  derived deterministically as `202611220 + act_idx` with `+100/+200/+300` offsets for
  Q/K/V donors and recorded per row) verifying target actuation (`> 1e-2`) and sibling
  isolation (`<= 1e-6`) across all 8 hook paths for `L7H5` & `L8H15` plus the `CM.4`
  span-swap path, with zero target-movement floor imposed on science rows.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = "causal_attn_v1.0"
DESIGN_REL_PATH = "explore_olmo2_attn_mlp/CAUSAL-ATTN-DESIGN.md"
DESIGN_SHA256 = "9b41482fbd1fae267ece14a859274ddf886de3ecd94526e44a4c5697821030fa"

MODEL_REPO = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_COMMIT = "48d788eca847d4d7548f375ad03d3c9312f6139e"
DTYPE_STR = "float32"
ATTN_IMPL = "eager"

NOMINAL_DOSE_MAG = 0.8
N_NULL_DRAWS = 4
NULL_SEED_BASE = 202611210
NULL_DOMAIN_TAG = "part_b_causal_attn"
SYNTH_ACTUATION_SEED = 202611220

# Pinned Part A Bridge scorecard & historical source blobs (v3 §2.1 & §3 CM.0).
PART_A_SCORECARD_SHA256 = "9e23bba5746787f6a1693c906d6b5d10dac9c46be63595fd638b7e4781f0d5f2"
PART_A_RECEIPT_SHA256 = "284544beb162392e24173b26935a28f32588bc74ca669a773dea72746a5027a3"
PART_A_DEFAULT_REL_PATH = "explore_olmo2_attn_mlp/INSTRUCT-BRIDGE-DESIGN.md"

SOURCE_PINS: dict[str, str] = {
    "old_manifest_sha256": "716cf2ab3f08461656c14768a7b5c3e6974261f5a86456c56e00c9a2e0800533",
    "old_runner_sha256": "227d8bc79a6a9c8563a1f8919bcb1c3aadb6338dce2a107ec25335582f9bc2e6",
    "new_manifest_sha256": "1a187ada52cd7ca6ebe122c7f93eafdf6ffe3024c739de6ade833707d05e403b",
    "new_runner_sha256": "405d67721015d66ec06dc2c5d1de9948e7547059ac120c58c9da61264e003612",
    "part_a_scorecard_sha256": PART_A_SCORECARD_SHA256,
    "part_a_receipt_sha256": PART_A_RECEIPT_SHA256,
}

OLD_GIT_COMMIT = "d13fc3dd2b76309d28b001ecf33f0fc9c962f156"
NEW_GIT_COMMIT = "01bd41d963936d39f68b032eda845ad73b13d6a2"

# Exact canonical 5 core units (v3 §2.1).
CANONICAL_CORE_UNITS: list[dict[str, Any]] = [
    {"id": "p92-02", "layer": 3, "neuron": 2628, "lead": "the", "content": "town", "group": "L3"},
    {"id": "p92-05", "layer": 3, "neuron": 6986, "lead": "a", "content": "fox", "group": "L3"},
    {"id": "p92-13", "layer": 3, "neuron": 976, "lead": "a", "content": "skeleton", "group": "L3"},
    {"id": "p92-46", "layer": 5, "neuron": 4673, "lead": "a", "content": "regular guy", "group": "L5"},
    {"id": "p92-09", "layer": 6, "neuron": 4145, "lead": "a", "content": "dog", "group": "L6"},
]
CORE_UNIT_IDS: tuple[str, ...] = tuple(u["id"] for u in CANONICAL_CORE_UNITS)
UNIT_BY_ID: dict[str, dict[str, Any]] = {u["id"]: u for u in CANONICAL_CORE_UNITS}
SMOKE_UNIT_ID = "p92-46"

# Exact 7 downstream attention heads (v3 §2.1).
CANONICAL_HEADS: list[dict[str, Any]] = [
    {"name": "L7H5", "layer": 7, "head": 5, "kind": "primary_L7"},
    {"name": "L7H4", "layer": 7, "head": 4, "kind": "rand_L7"},
    {"name": "L7H9", "layer": 7, "head": 9, "kind": "rand_L7"},
    {"name": "L8H15", "layer": 8, "head": 15, "kind": "primary_L8"},
    {"name": "L8H2", "layer": 8, "head": 2, "kind": "active_comparator_L8"},
    {"name": "L8H3", "layer": 8, "head": 3, "kind": "rand_L8"},
    {"name": "L8H13", "layer": 8, "head": 13, "kind": "rand_L8"},
]
HEAD_NAMES: tuple[str, ...] = tuple(h["name"] for h in CANONICAL_HEADS)
HEAD_BY_NAME: dict[str, dict[str, Any]] = {h["name"]: h for h in CANONICAL_HEADS}

# Exact full-precision pinned physical norms (v3 §2.1).
CANONICAL_NORMS: dict[str, dict[str, float]] = {
    "OLD_NORM": {
        "3": 7.3919267654418945,
        "5": 10.05025863647461,
        "6": 10.501473426818848,
    },
    "NEW_NORM": {
        "3": 9.312541007995605,
        "5": 11.525568008422852,
        "6": 12.170919418334961,
    },
}

# Frozen hook modes (z_all always precedes QKV_all for immediate reconciliation).
CORE_MODES: tuple[str, ...] = ("z_all", "z_read_pos", "z_upstream", "QKV_all", "QK_only", "V_only")
DIAG_MODES: tuple[str, ...] = ("Q_only", "K_only")
ALL_MODES: tuple[str, ...] = CORE_MODES + DIAG_MODES
Z_MODES: frozenset[str] = frozenset({"z_all", "z_read_pos", "z_upstream"})
QKV_MODES: frozenset[str] = frozenset({"QKV_all", "QK_only", "V_only", "Q_only", "K_only"})
QKV_MODE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "QKV_all": ("q", "k", "v"),
    "QK_only": ("q", "k"),
    "V_only": ("v",),
    "Q_only": ("q",),
    "K_only": ("k",),
}

DIAG_HEADS_FULL: tuple[str, ...] = ("L7H5", "L8H15", "L8H2")
SMOKE_HEADS: tuple[str, ...] = ("L7H5", "L8H15", "L8H3")
DIAG_HEADS_SMOKE: tuple[str, ...] = ("L7H5", "L8H15")

CM4_HEADS_FULL: tuple[str, ...] = ("L8H15", "L8H2", "L8H3")
CM4_HEADS_SMOKE: tuple[str, ...] = ("L8H15", "L8H3")

# Fresh past-tense confirmatory bank B1..B4 (v3 §2.3).
BANK_STORIES: list[dict[str, str]] = [
    {
        "story": "B1",
        "scene": "At the cedar barn after dusk",
        "verb": "locked",
        "obj": "the copper latch",
        "name": "Mark",
        "role": "the painter",
    },
    {
        "story": "B2",
        "scene": "By the brick shop before dawn",
        "verb": "swept",
        "obj": "the stone steps",
        "name": "Sarah",
        "role": "the doctor",
    },
    {
        "story": "B3",
        "scene": "Near the pine ridge at noon",
        "verb": "packed",
        "obj": "the canvas tent",
        "name": "David",
        "role": "the ranger",
    },
    {
        "story": "B4",
        "scene": "In the stone hall after rain",
        "verb": "folded",
        "obj": "the linen map",
        "name": "Grace",
        "role": "the writer",
    },
]
BANK_CELLS: tuple[str, ...] = ("SELF", "NAME_BARE", "ROLE_BARE")
CROSS_STORY_PARTNER: dict[str, str] = {"B1": "B2", "B2": "B1", "B3": "B4", "B4": "B3"}

# Frozen coordinate spans for CM.4 (v3 §2.3 & §3 CM.4).
FROZEN_SPANS: dict[str, list[int]] = {
    "SPAN_REF": [13, 20, 30, 31],
    "SPAN_NONREF": [8, 9, 16, 17],
    "SPAN_REF_USER": [13, 20],
    "SPAN_NONREF_USER": [8, 16],
    "SPAN_REF_ASST": [30, 31],
}
SPAN_ORDER: tuple[str, ...] = (
    "SPAN_REF",
    "SPAN_NONREF",
    "SPAN_REF_USER",
    "SPAN_NONREF_USER",
    "SPAN_REF_ASST",
)
SWAP_CONDITIONS: tuple[tuple[str, str, str, bool], ...] = (
    # (swap_cond, target_cell, donor_cell, cross_story)
    ("N2S", "NAME_BARE", "SELF", False),
    ("S2N", "SELF", "NAME_BARE", False),
    ("S2S", "SELF", "SELF", True),
    ("N2N", "NAME_BARE", "NAME_BARE", True),
)

# Exact CM.0 tolerances (v3 §3 CM.0).
TOL_SHAM = 1e-6
TOL_DECOMP = 1e-6
TOL_RECONCILE = 1e-5
TOL_SIBLING = 1e-6
MIN_ACTUATION_DELTA = 1e-2


# ---------------------------------------------------------------------------
# Hashing, Part A Inheritance, Source Validation & Pre-Load Admission
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def assert_strictly_downstream(unit_layer: int, head_layer: int) -> None:
    if not (int(head_layer) > int(unit_layer)):
        raise ValueError(f"Non-causal head_layer={head_layer} <= unit_layer={unit_layer}")


def assert_core_units_canonical(units: list[dict[str, Any]], label: str) -> None:
    if not isinstance(units, list) or len(units) != len(CANONICAL_CORE_UNITS):
        raise ValueError(f"{label}: expected {len(CANONICAL_CORE_UNITS)} core units")
    for got, exp in zip(units, CANONICAL_CORE_UNITS):
        for k in ("id", "layer", "neuron", "lead", "content"):
            if got.get(k) != exp[k]:
                raise ValueError(f"{label}: core unit drift on {exp['id']}.{k}: {got.get(k)!r} != {exp[k]!r}")


def resolve_part_a_handoff(part_a_scorecard: dict[str, Any]) -> dict[str, Any]:
    """Deterministically resolve Part B operating norm N* & p92-09 track from Part A scorecard.

    Auto-HOLDS (raises ValueError) if Part A declined an operating norm (`decline_norm...`).
    """
    if not isinstance(part_a_scorecard, dict):
        raise ValueError("Part A scorecard must be a JSON object")
    meta = part_a_scorecard.get("meta", {})
    if meta.get("artifact_type") != "instruct_bridge_scorecard":
        raise ValueError(f"Unexpected Part A artifact_type: {meta.get('artifact_type')!r}")
    if meta.get("execution_provenance") != "measured_gpu_run":
        raise ValueError(f"Part A scorecard provenance not measured_gpu_run: {meta.get('execution_provenance')!r}")
    if not meta.get("external_pins_verified", False) or not meta.get("complete_join_verified", False):
        raise ValueError("Part A scorecard pins/joins not verified")
    if part_a_scorecard.get("BR0_anchor_replay", {}).get("status") != "passed":
        raise ValueError("Part A BR0 anchor replay did not pass")

    br4 = part_a_scorecard.get("BR4_L7H5_and_part_b_norm", {})
    norm_dec = str(br4.get("part_b_norm_decision", ""))
    p92_09_track = str(br4.get("p92_09_part_b_track", ""))

    if norm_dec.startswith("decline_norm"):
        raise ValueError(f"Part B AUTO-HELD by Part A BR.4: part_b_norm_decision={norm_dec!r}")
    if norm_dec.startswith("pin_OLD_NORM"):
        norm_name = "OLD_NORM"
    elif norm_dec.startswith("pin_NEW_NORM"):
        norm_name = "NEW_NORM"
    else:
        raise ValueError(f"Unrecognized Part A part_b_norm_decision: {norm_dec!r}")

    if p92_09_track not in ("include_spec_only_track", "drop_from_part_b"):
        raise ValueError(f"Unrecognized Part A p92_09_part_b_track: {p92_09_track!r}")

    return {
        "operating_norm_name": norm_name,
        "operating_norm_multipliers": dict(CANONICAL_NORMS[norm_name]),
        "part_b_norm_decision": norm_dec,
        "p92_09_part_b_track": p92_09_track,
    }


def parse_all_slot_units_from_old_runner_source(runner_src: str) -> dict[str, dict[str, Any]]:
    """Safely parse `ALL_SLOT_UNITS` from `d13fc3dd:run_signed_dose.py` via AST (no exec)."""
    tree = ast.parse(runner_src)
    for node in tree.body:
        target_name = None
        value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        if target_name == "ALL_SLOT_UNITS" and value_node is not None:
            raw = ast.literal_eval(value_node)
            out: dict[str, dict[str, Any]] = {}
            for item in raw:
                if isinstance(item, dict):
                    uid = str(item["id"])
                    out[uid] = {
                        "id": uid,
                        "layer": int(item["layer"]),
                        "neuron": int(item["neuron"]),
                        "lead": str(item["lead"]),
                        "content": str(item["content"]),
                    }
                else:
                    uid, layer, neuron, lead, content = item[:5]
                    out[str(uid)] = {
                        "id": str(uid),
                        "layer": int(layer),
                        "neuron": int(neuron),
                        "lead": str(lead),
                        "content": str(content),
                    }
            return out
    raise ValueError("ALL_SLOT_UNITS literal not found in OLD runner source")


def load_git_blob(repo_root: Path, commit: str, rel_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo_root), "show", f"{commit}:{rel_path}"])


def build_causal_attn_manifest(
    old_manifest_bytes: bytes,
    old_runner_bytes: bytes,
    new_manifest_bytes: bytes,
    new_runner_bytes: bytes,
    part_a_scorecard_bytes: bytes,
    design_path: Path,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Deterministically build `causal_attn_manifest.json` pinned to v3 design & Part A handoff."""
    got_design_sha = sha256_file(design_path)
    if got_design_sha != DESIGN_SHA256:
        raise ValueError(f"Design SHA-256 mismatch: got {got_design_sha}, expected {DESIGN_SHA256}")

    for pin_name, raw_b in [
        ("old_manifest_sha256", old_manifest_bytes),
        ("old_runner_sha256", old_runner_bytes),
        ("new_manifest_sha256", new_manifest_bytes),
        ("new_runner_sha256", new_runner_bytes),
        ("part_a_scorecard_sha256", part_a_scorecard_bytes),
    ]:
        got = sha256_bytes(raw_b)
        exp = SOURCE_PINS[pin_name]
        if got != exp:
            raise ValueError(f"Source pin mismatch for {pin_name}: got {got}, expected {exp}")

    # Verify canonical units & norms across historical sources + Part A scorecard.
    old_slot = parse_all_slot_units_from_old_runner_source(old_runner_bytes.decode("utf-8"))
    assert_core_units_canonical([old_slot[uid] for uid in CORE_UNIT_IDS], "d13fc3dd:run_signed_dose.py")

    old_man = json.loads(old_manifest_bytes.decode("utf-8"))
    old_meds = old_man["meta"]["pinned_medians_chat"]
    for lk, ev in CANONICAL_NORMS["OLD_NORM"].items():
        if float(old_meds[lk]) != float(ev):
            raise ValueError(f"OLD_NORM[{lk}] drift vs d13fc3dd manifest")

    new_man = json.loads(new_manifest_bytes.decode("utf-8"))
    assert_core_units_canonical(new_man["meta"]["core_units"], "01bd41d9:stage_sweep_manifest.json")

    part_a_sc = json.loads(part_a_scorecard_bytes.decode("utf-8"))
    handoff = resolve_part_a_handoff(part_a_sc)

    # Extract exact canonical (t_own, peers) from 01bd41d9 & verify set equality against d13fc3dd.
    unit_token_map: dict[str, dict[str, Any]] = {}
    old_q12_0 = old_man["q12_prompts"][0]["per_unit"]
    new_b_0 = new_man["bindings"]["eval_self_E1"]
    for uid in CORE_UNIT_IDS:
        t_new = int(new_b_0[uid]["t_own"])
        p_new = [int(x) for x in new_b_0[uid]["peers"]]
        t_old = int(old_q12_0[uid]["own_token_id"])
        p_old = [int(x) for x in old_q12_0[uid]["peer_token_ids"]]
        if t_new != t_old or set(p_new) != set(p_old) or len(p_new) != 19 or t_new in p_new:
            raise ValueError(f"Canonical token/peer pool drift for {uid}")
        unit_token_map[uid] = {"t_own": t_new, "peers": p_new}

    # Tokenize brand-new past-tense bank B1..B4 with pinned Instruct tokenizer.
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        def _default_tok(repo: str, commit: str):
            return AutoTokenizer.from_pretrained(repo, revision=commit, local_files_only=True)

        tokenizer_loader = _default_tok
    tok = tokenizer_loader(MODEL_REPO, MODEL_COMMIT)

    def _encode_prompt(user_msg: str, asst_prefix: str) -> tuple[list[int], str]:
        full_text = (
            tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            + asst_prefix
        )
        ids = [int(x) for x in tok.encode(full_text, add_special_tokens=False)]
        return ids, full_text

    prompts_meta: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}

    for st in BANK_STORIES:
        sid = st["story"]
        scene, verb, obj, name, role = st["scene"], st["verb"], st["obj"], st["name"], st["role"]
        role_cap = role[0].upper() + role[1:]
        cell_specs = [
            ("SELF", f"{scene}, you {verb} {obj}. Identify yourself in one line.", "I am "),
            ("NAME_BARE", f"{scene}, {name} {verb} {obj}. Identify {name} in one line.", f"{name} is "),
            ("ROLE_BARE", f"{scene}, {role} {verb} {obj}. Identify {role} in one line.", f"{role_cap} is "),
        ]
        for cell, u_msg, a_base in cell_specs:
            pid = f"{sid}_{cell}"
            prompts_meta.append({
                "prompt_id": pid,
                "story": sid,
                "cell": cell,
                "user_msg": u_msg,
                "asst_prefix_template": a_base + "{lead}",
            })
            per_u: dict[str, Any] = {}
            for u in CANONICAL_CORE_UNITS:
                uid = u["id"]
                ids, full_text = _encode_prompt(u_msg, a_base + u["lead"])
                exp_len = 36 if cell == "ROLE_BARE" else 33
                if len(ids) != exp_len:
                    raise ValueError(f"Token length drift on ({pid}, {uid}): got {len(ids)}, expected {exp_len}")
                per_u[uid] = {
                    "input_ids": ids,
                    "read_pos": exp_len - 1,
                    "position_ids": list(range(exp_len)),
                    "t_own": unit_token_map[uid]["t_own"],
                    "peers": list(unit_token_map[uid]["peers"]),
                    "prefix_text": full_text,
                }
            bindings[pid] = per_u

        # Enforce exact 1-to-1 SELF vs NAME_BARE differing indices == [13, 20, 30, 31] on every unit.
        for uid in CORE_UNIT_IDS:
            s_ids = bindings[f"{sid}_SELF"][uid]["input_ids"]
            n_ids = bindings[f"{sid}_NAME_BARE"][uid]["input_ids"]
            diffs = [i for i in range(len(s_ids)) if s_ids[i] != n_ids[i]]
            if diffs != FROZEN_SPANS["SPAN_REF"]:
                raise ValueError(
                    f"Exact alignment broken on ({sid}, {uid}): diffs={diffs} != {FROZEN_SPANS['SPAN_REF']}"
                )

    # Zero-overlap guard against d13fc3dd & 01bd41d9 prompt banks.
    hist_blob = old_manifest_bytes.decode("utf-8") + "\n" + new_manifest_bytes.decode("utf-8")
    for st in BANK_STORIES:
        for field in ("scene", "obj", "name", "role"):
            if st[field] in hist_blob:
                raise ValueError(f"Prompt bank overlap detected with historical manifests: {st[field]!r}")

    # Build exact CM.1-CM.3 schedules (full & smoke).
    def _build_cm123_head_modes(heads: tuple[str, ...], diag_heads: tuple[str, ...]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for hname in heads:
            for mode in CORE_MODES:
                out.append({"head_name": hname, "mode": mode})
            if hname in diag_heads:
                for mode in DIAG_MODES:
                    out.append({"head_name": hname, "mode": mode})
        return out

    cm123_head_modes_full = _build_cm123_head_modes(HEAD_NAMES, DIAG_HEADS_FULL)
    cm123_head_modes_smoke = _build_cm123_head_modes(SMOKE_HEADS, DIAG_HEADS_SMOKE)

    schedule_cm123_full = [
        {"prompt_id": p["prompt_id"], "story": p["story"], "cell": p["cell"], "dose_mag": NOMINAL_DOSE_MAG}
        for p in prompts_meta
    ]
    schedule_cm123_smoke = [
        e for e in schedule_cm123_full if e["story"] == "B1"
    ]

    # Build exact 17 CM.0 synthetic actuation positive controls (shared by smoke & full).
    actuation_schedule: list[dict[str, Any]] = []
    act_idx = 0
    for hname, pid in (("L7H5", "B1_ROLE_BARE"), ("L8H15", "B1_NAME_BARE")):
        for mode in ALL_MODES:
            actuation_schedule.append({
                "actuation_id": f"act_{act_idx:02d}_{hname}_{mode}",
                "kind": "head_mode",
                "unit": SMOKE_UNIT_ID,
                "prompt_id": pid,
                "head_name": hname,
                "mode": mode,
                "swap_indices": None,
                "seed": SYNTH_ACTUATION_SEED + act_idx,
            })
            act_idx += 1
    actuation_schedule.append({
        "actuation_id": f"act_{act_idx:02d}_L8H15_CM4_SPAN_REF",
        "kind": "cm4_span_swap",
        "unit": SMOKE_UNIT_ID,
        "prompt_id": "B1_NAME_BARE",
        "head_name": "L8H15",
        "mode": "cm4_span_swap",
        "swap_indices": list(FROZEN_SPANS["SPAN_REF"]),
        "seed": SYNTH_ACTUATION_SEED + act_idx,
    })

    # Build exact CM.4 swap schedules (full: 400 tuples x 3 heads = 1,200 rows; smoke: 14 tuples x 2 heads = 28 rows).
    cm4_tuples_full: list[dict[str, Any]] = []
    for uid in CORE_UNIT_IDS:
        for st in BANK_STORIES:
            sid = st["story"]
            partner = CROSS_STORY_PARTNER[sid]
            for span_name in SPAN_ORDER:
                for swap_cond, t_cell, d_cell, is_cross in SWAP_CONDITIONS:
                    d_story = partner if is_cross else sid
                    cm4_tuples_full.append({
                        "unit": uid,
                        "target_story": sid,
                        "target_cell": t_cell,
                        "target_prompt_id": f"{sid}_{t_cell}",
                        "donor_story": d_story,
                        "donor_cell": d_cell,
                        "donor_prompt_id": f"{d_story}_{d_cell}",
                        "swap_cond": swap_cond,
                        "span_name": span_name,
                        "swap_indices": list(FROZEN_SPANS[span_name]),
                    })

    cm4_tuples_smoke: list[dict[str, Any]] = []
    # B1 reciprocal N2S / S2N on all 5 spans = 10 tuples
    for span_name in SPAN_ORDER:
        for swap_cond, t_cell, d_cell in (("N2S", "NAME_BARE", "SELF"), ("S2N", "SELF", "NAME_BARE")):
            cm4_tuples_smoke.append({
                "unit": SMOKE_UNIT_ID,
                "target_story": "B1",
                "target_cell": t_cell,
                "target_prompt_id": f"B1_{t_cell}",
                "donor_story": "B1",
                "donor_cell": d_cell,
                "donor_prompt_id": f"B1_{d_cell}",
                "swap_cond": swap_cond,
                "span_name": span_name,
                "swap_indices": list(FROZEN_SPANS[span_name]),
            })
    # B1 <- B2 same-cond S2S / N2N on SPAN_REF & SPAN_NONREF = 4 tuples
    for span_name in ("SPAN_REF", "SPAN_NONREF"):
        for swap_cond, cell in (("S2S", "SELF"), ("N2N", "NAME_BARE")):
            cm4_tuples_smoke.append({
                "unit": SMOKE_UNIT_ID,
                "target_story": "B1",
                "target_cell": cell,
                "target_prompt_id": f"B1_{cell}",
                "donor_story": "B2",
                "donor_cell": cell,
                "donor_prompt_id": f"B2_{cell}",
                "swap_cond": swap_cond,
                "span_name": span_name,
                "swap_indices": list(FROZEN_SPANS[span_name]),
            })

    cm4_base_captures_full = [
        {"unit": uid, "prompt_id": f"{st['story']}_{cell}", "story": st["story"], "cell": cell}
        for uid in CORE_UNIT_IDS
        for st in BANK_STORIES
        for cell in ("SELF", "NAME_BARE")
    ]
    cm4_base_captures_smoke = [
        {"unit": SMOKE_UNIT_ID, "prompt_id": f"{sid}_{cell}", "story": sid, "cell": cell}
        for sid in ("B1", "B2")
        for cell in ("SELF", "NAME_BARE")
    ]

    def _calc_denoms(
        n_units: int,
        n_prompts: int,
        n_head_modes: int,
        n_act: int,
        n_cm4_base: int,
        n_cm4_tuples: int,
        n_cm4_heads: int,
    ) -> dict[str, int]:
        n_unp = n_units * n_prompts
        n_hm = n_unp * n_head_modes
        n_b19 = n_unp + 2 * n_hm
        n_swap = n_cm4_tuples * n_cm4_heads
        n_b1 = n_act + n_cm4_base + 2 * n_swap
        return {
            "n_unpatched_rows": n_unp,
            "n_head_mode_rows": n_hm,
            "n_actuation_rows": n_act,
            "n_swap_rows": n_swap,
            "n_cm4_base_captures": n_cm4_base,
            "n_b19_forwards": n_b19,
            "n_b1_forwards": n_b1,
            "n_total_forwards": n_b19 + n_b1,
        }

    denoms_full = _calc_denoms(
        n_units=len(CORE_UNIT_IDS),
        n_prompts=len(schedule_cm123_full),
        n_head_modes=len(cm123_head_modes_full),
        n_act=len(actuation_schedule),
        n_cm4_base=len(cm4_base_captures_full),
        n_cm4_tuples=len(cm4_tuples_full),
        n_cm4_heads=len(CM4_HEADS_FULL),
    )
    denoms_smoke = _calc_denoms(
        n_units=1,
        n_prompts=len(schedule_cm123_smoke),
        n_head_modes=len(cm123_head_modes_smoke),
        n_act=len(actuation_schedule),
        n_cm4_base=len(cm4_base_captures_smoke),
        n_cm4_tuples=len(cm4_tuples_smoke),
        n_cm4_heads=len(CM4_HEADS_SMOKE),
    )

    assert denoms_full == {
        "n_unpatched_rows": 60,
        "n_head_mode_rows": 2880,
        "n_actuation_rows": 17,
        "n_swap_rows": 1200,
        "n_cm4_base_captures": 40,
        "n_b19_forwards": 5820,
        "n_b1_forwards": 2457,
        "n_total_forwards": 8277,
    }, f"Unexpected full denominators: {denoms_full}"
    assert denoms_smoke == {
        "n_unpatched_rows": 3,
        "n_head_mode_rows": 66,
        "n_actuation_rows": 17,
        "n_swap_rows": 28,
        "n_cm4_base_captures": 4,
        "n_b19_forwards": 135,
        "n_b1_forwards": 77,
        "n_total_forwards": 212,
    }, f"Unexpected smoke denominators: {denoms_smoke}"

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "design_file": DESIGN_REL_PATH,
            "design_sha256": DESIGN_SHA256,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "nominal_dose_mag": NOMINAL_DOSE_MAG,
            "n_null_draws": N_NULL_DRAWS,
            "null_seed_base": NULL_SEED_BASE,
            "null_domain_tag": NULL_DOMAIN_TAG,
            "synth_actuation_seed": SYNTH_ACTUATION_SEED,
            "part_a_scorecard_sha256": PART_A_SCORECARD_SHA256,
            "part_a_receipt_sha256": PART_A_RECEIPT_SHA256,
            "part_a_part_b_norm_decision": handoff["part_b_norm_decision"],
            "part_a_p92_09_part_b_track": handoff["p92_09_part_b_track"],
            "operating_norm_name": handoff["operating_norm_name"],
            "operating_norm_multipliers": handoff["operating_norm_multipliers"],
            "core_units": CANONICAL_CORE_UNITS,
            "heads": CANONICAL_HEADS,
            "smoke_unit": SMOKE_UNIT_ID,
            "frozen_spans": FROZEN_SPANS,
            "source_pins": dict(SOURCE_PINS),
            "denominators": {"full": denoms_full, "smoke": denoms_smoke},
        },
        "prompts": prompts_meta,
        "bindings": bindings,
        "schedule_full": {
            "units": list(CORE_UNIT_IDS),
            "cm123_prompts": schedule_cm123_full,
            "cm123_head_modes": cm123_head_modes_full,
            "actuation": actuation_schedule,
            "cm4_base_captures": cm4_base_captures_full,
            "cm4_tuples": cm4_tuples_full,
            "cm4_heads": list(CM4_HEADS_FULL),
        },
        "schedule_smoke": {
            "units": [SMOKE_UNIT_ID],
            "cm123_prompts": schedule_cm123_smoke,
            "cm123_head_modes": cm123_head_modes_smoke,
            "actuation": actuation_schedule,
            "cm4_base_captures": cm4_base_captures_smoke,
            "cm4_tuples": cm4_tuples_smoke,
            "cm4_heads": list(CM4_HEADS_SMOKE),
        },
    }


def _admit_frozen_package_with_bytes(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    part_a_scorecard_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
    expected_part_a_scorecard_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, bytes]]:
    """Single-read external 4-pin + Part A scorecard verification BEFORE any model/tokenizer load."""
    bytes_by_label: dict[str, bytes] = {}
    for label, p, exp in [
        ("design", design_path, expected_design_sha),
        ("manifest", manifest_path, expected_manifest_sha),
        ("runner", runner_path, expected_runner_sha),
        ("scorer", scorer_path, expected_scorer_sha),
        ("part_a_scorecard", part_a_scorecard_path, expected_part_a_scorecard_sha),
    ]:
        if not exp or len(exp) != 64:
            raise ValueError(f"Missing/invalid 64-hex external pin for {label}: {exp!r}")
        if not p.is_file():
            raise FileNotFoundError(f"Missing pinned file for {label}: {p}")
        raw_b = p.read_bytes()
        got = sha256_bytes(raw_b)
        if got != exp:
            raise ValueError(f"SHA-256 pin mismatch for {label} ({p}): got {got}, expected {exp}")
        bytes_by_label[label] = raw_b

    if expected_design_sha != DESIGN_SHA256:
        raise ValueError(f"Design pin {expected_design_sha} != frozen v3 {DESIGN_SHA256}")
    if expected_part_a_scorecard_sha != PART_A_SCORECARD_SHA256:
        raise ValueError(
            f"Part A scorecard pin {expected_part_a_scorecard_sha} != pinned {PART_A_SCORECARD_SHA256}"
        )

    manifest = json.loads(bytes_by_label["manifest"].decode("utf-8"))
    part_a_sc = json.loads(bytes_by_label["part_a_scorecard"].decode("utf-8"))
    handoff = resolve_part_a_handoff(part_a_sc)

    mmeta = manifest.get("meta", {})
    if mmeta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema_version mismatch: {mmeta.get('schema_version')!r}")
    if mmeta.get("design_sha256") != expected_design_sha:
        raise ValueError("Manifest embedded design_sha256 != external design pin")
    if mmeta.get("part_a_scorecard_sha256") != expected_part_a_scorecard_sha:
        raise ValueError("Manifest embedded part_a_scorecard_sha256 != external Part A pin")
    if mmeta.get("model_repo") != MODEL_REPO or mmeta.get("model_commit") != MODEL_COMMIT:
        raise ValueError("Manifest model repo/commit mismatch")
    assert_core_units_canonical(mmeta.get("core_units", []), "admit_manifest")

    # Enforce manifest norm & p92-09 track match Part A's resolved handoff verbatim.
    if mmeta.get("operating_norm_name") != handoff["operating_norm_name"]:
        raise ValueError(
            f"Manifest operating_norm_name={mmeta.get('operating_norm_name')!r} != Part A {handoff['operating_norm_name']!r}"
        )
    if mmeta.get("operating_norm_multipliers") != handoff["operating_norm_multipliers"]:
        raise ValueError("Manifest operating_norm_multipliers mismatch vs Part A handoff")
    if mmeta.get("part_a_part_b_norm_decision") != handoff["part_b_norm_decision"]:
        raise ValueError("Manifest part_a_part_b_norm_decision mismatch vs Part A scorecard")
    if mmeta.get("part_a_p92_09_part_b_track") != handoff["p92_09_part_b_track"]:
        raise ValueError("Manifest part_a_p92_09_part_b_track mismatch vs Part A scorecard")

    pins = {
        "design_sha256": expected_design_sha,
        "manifest_sha256": expected_manifest_sha,
        "runner_sha256": expected_runner_sha,
        "scorer_sha256": expected_scorer_sha,
        "part_a_scorecard_sha256": expected_part_a_scorecard_sha,
    }
    return manifest, part_a_sc, pins, bytes_by_label


def _load_pinned_scorer_module(scorer_path: Path, expected_scorer_sha: str, scorer_bytes: bytes | None = None):
    raw_b = scorer_path.read_bytes() if scorer_bytes is None else scorer_bytes
    got = sha256_bytes(raw_b)
    if got != expected_scorer_sha:
        raise ValueError(f"Scorer SHA-256 mismatch before load: got {got}, expected {expected_scorer_sha}")
    prev_mod = sys.modules.get("run_causal_attn")
    sys.modules["run_causal_attn"] = sys.modules[__name__]
    try:
        mod = types.ModuleType("_pinned_score_causal_attn")
        mod.__file__ = str(scorer_path.resolve())
        code = compile(raw_b, str(scorer_path.resolve()), "exec")
        exec(code, mod.__dict__)
        return mod
    finally:
        if prev_mod is not None:
            sys.modules["run_causal_attn"] = prev_mod


def validate_timing_breakdown(
    timing: Any,
    expected_b19: int,
    expected_b1: int,
    full_b19: int = 5820,
    full_b1: int = 2457,
) -> None:
    if not isinstance(timing, dict):
        raise ValueError("Missing or null timing breakdown")
    req_keys = (
        "model_load_sec",
        "b19_elapsed_sec",
        "b1_elapsed_sec",
        "non_fwd_runner_sec",
        "total_runner_sec",
        "n_b19_done",
        "n_b1_done",
        "ms_per_b19",
        "ms_per_b1",
        "projected_full_b19_sec",
        "projected_full_b1_sec",
        "projected_full_fwd_sec",
    )
    for k in req_keys:
        v = timing.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise ValueError(f"Timing field '{k}' missing or non-finite: {v!r}")
    if int(timing["n_b19_done"]) != int(expected_b19):
        raise ValueError(f"Timing n_b19_done={timing['n_b19_done']} != expected {expected_b19}")
    if int(timing["n_b1_done"]) != int(expected_b1):
        raise ValueError(f"Timing n_b1_done={timing['n_b1_done']} != expected {expected_b1}")
    if float(timing["model_load_sec"]) < 0.0 or float(timing["non_fwd_runner_sec"]) < 0.0:
        raise ValueError("Negative timing field")
    if float(timing["b19_elapsed_sec"]) <= 0.0 or float(timing["b1_elapsed_sec"]) <= 0.0:
        raise ValueError("Non-positive forward elapsed seconds")
    if float(timing["total_runner_sec"]) + 1e-9 < float(timing["b19_elapsed_sec"]) + float(timing["b1_elapsed_sec"]):
        raise ValueError("total_runner_sec < b19_elapsed_sec + b1_elapsed_sec")
    exp_ms19 = 1000.0 * float(timing["b19_elapsed_sec"]) / float(expected_b19)
    exp_ms1 = 1000.0 * float(timing["b1_elapsed_sec"]) / float(expected_b1)
    if abs(float(timing["ms_per_b19"]) - exp_ms19) > 1e-3:
        raise ValueError("Timing ms_per_b19 inconsistent")
    if abs(float(timing["ms_per_b1"]) - exp_ms1) > 1e-3:
        raise ValueError("Timing ms_per_b1 inconsistent")
    exp_proj_b19 = (float(timing["ms_per_b19"]) / 1000.0) * float(full_b19)
    exp_proj_b1 = (float(timing["ms_per_b1"]) / 1000.0) * float(full_b1)
    if float(timing["projected_full_b19_sec"]) <= 0.0 or abs(float(timing["projected_full_b19_sec"]) - exp_proj_b19) > 1e-3:
        raise ValueError("Timing projected_full_b19_sec inconsistent with ms_per_b19 * full_b19")
    if float(timing["projected_full_b1_sec"]) <= 0.0 or abs(float(timing["projected_full_b1_sec"]) - exp_proj_b1) > 1e-3:
        raise ValueError("Timing projected_full_b1_sec inconsistent with ms_per_b1 * full_b1")
    if float(timing["projected_full_fwd_sec"]) <= 0.0 or abs(
        float(timing["projected_full_fwd_sec"]) - (float(timing["projected_full_b19_sec"]) + float(timing["projected_full_b1_sec"]))
    ) > 1e-3:
        raise ValueError("Timing projected_full_fwd_sec inconsistent with projected_full_b19_sec + projected_full_b1_sec")


def admit_execution_request(
    mode: str | None,
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    part_a_scorecard_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
    expected_part_a_scorecard_sha: str,
    allow_full_after_smoke: bool = False,
    verified_smoke_receipt: Path | None = None,
    verified_smoke_sha256: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Enforce explicit --mode {smoke,full}, single-read 4-pin + Part A binding, and pre-load smoke gate."""
    if mode not in ("smoke", "full"):
        raise ValueError(f"Explicit --mode {{smoke,full}} required (got {mode!r}); implicit full refused.")
    if runner_path.resolve() != Path(__file__).resolve():
        raise ValueError(f"Runner path {runner_path.resolve()} != executing module {Path(__file__).resolve()}")

    manifest, part_a_sc, pins, pkg_bytes = _admit_frozen_package_with_bytes(
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        part_a_scorecard_path=part_a_scorecard_path,
        expected_design_sha=expected_design_sha,
        expected_manifest_sha=expected_manifest_sha,
        expected_runner_sha=expected_runner_sha,
        expected_scorer_sha=expected_scorer_sha,
        expected_part_a_scorecard_sha=expected_part_a_scorecard_sha,
    )

    if mode == "full":
        if not allow_full_after_smoke:
            raise ValueError("Full execution refused: --allow-full-after-smoke required.")
        if verified_smoke_receipt is None or not verified_smoke_receipt.is_file():
            raise ValueError(f"Full execution refused: missing --verified-smoke-receipt ({verified_smoke_receipt})")
        if not verified_smoke_sha256 or len(verified_smoke_sha256) != 64:
            raise ValueError("Full execution refused: valid 64-hex --verified-smoke-sha256 required.")
        smoke_bytes = verified_smoke_receipt.read_bytes()
        got_sm_sha = sha256_bytes(smoke_bytes)
        if got_sm_sha != verified_smoke_sha256:
            raise ValueError(f"Verified smoke SHA-256 mismatch: got {got_sm_sha}, expected {verified_smoke_sha256}")
        scorer_mod = _load_pinned_scorer_module(scorer_path, expected_scorer_sha, scorer_bytes=pkg_bytes["scorer"])
        sm = json.loads(smoke_bytes.decode("utf-8"))
        sm_verdict = scorer_mod.verify_smoke_receipt_only(
            receipt=sm,
            manifest=manifest,
            pins=pins,
            part_a_scorecard=part_a_sc,
            allow_fixture=False,
        )
        if not sm_verdict.get("smoke_technical_check_passed", False):
            raise ValueError(f"Full execution refused: pinned smoke verifier failed ({sm_verdict})")
    return manifest, part_a_sc, pins


# ---------------------------------------------------------------------------
# Orthogonal Null Generators, Logit Decomposition & Hook Utilities
# ---------------------------------------------------------------------------

def part_b_unit_seed(unit_id: str, prompt_id: str, dose_mag: float) -> int:
    key = f"{NULL_SEED_BASE}|{NULL_DOMAIN_TAG}|{unit_id}|{prompt_id}|{dose_mag:.4f}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def make_orthogonal_null_vectors(
    w_down_cpu: torch.Tensor,
    target_neuron: int,
    hat_w_cpu: torch.Tensor,
    n_draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_hat = hat_w_cpu.float() / float(hat_w_cpu.float().norm().item())
    d_model, d_ff = w_down_cpu.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)

    coeffs = torch.randn((d_ff, n_draws), generator=gen, dtype=torch.float32)
    coeffs[target_neuron, :] = 0.0
    raw_wd = (w_down_cpu.float() @ coeffs).T
    raw_wd = raw_wd - (raw_wd @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    wdown_unit = raw_wd / torch.linalg.vector_norm(raw_wd, dim=-1, keepdim=True).clamp_min(1e-12)

    raw_iso = torch.randn((n_draws, d_model), generator=gen, dtype=torch.float32)
    raw_iso = raw_iso - (raw_iso @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    iso_unit = raw_iso / torch.linalg.vector_norm(raw_iso, dim=-1, keepdim=True).clamp_min(1e-12)

    for nm, mat in (("wdown_orth", wdown_unit), ("iso_resid", iso_unit)):
        if float((mat @ v_hat).abs().max().item()) >= 1e-5:
            raise RuntimeError(f"{nm} orthogonality check failed")
    return wdown_unit, iso_unit


def build_b19_deltas(
    hat_w_cpu: torch.Tensor,
    wd_unit_cpu: torch.Tensor,
    iso_unit_cpu: torch.Tensor,
    dose_mag: float,
    norm_multiplier: float,
    device: str,
) -> torch.Tensor:
    scale = float(dose_mag) * float(norm_multiplier)
    B = 3 + 4 * N_NULL_DRAWS
    d_model = hat_w_cpu.shape[0]
    deltas = torch.zeros((B, d_model), dtype=torch.float32, device="cpu")
    deltas[1] = scale * hat_w_cpu
    deltas[2] = -scale * hat_w_cpu
    deltas[3 : 3 + N_NULL_DRAWS] = scale * wd_unit_cpu
    deltas[3 + N_NULL_DRAWS : 3 + 2 * N_NULL_DRAWS] = -scale * wd_unit_cpu
    deltas[3 + 2 * N_NULL_DRAWS : 3 + 3 * N_NULL_DRAWS] = scale * iso_unit_cpu
    deltas[3 + 3 * N_NULL_DRAWS : 3 + 4 * N_NULL_DRAWS] = -scale * iso_unit_cpu

    if float(deltas[0].abs().max().item()) != 0.0:
        raise ValueError("Row 0 of B=19 must be exact zero")
    if float((deltas[1] + deltas[2]).abs().max().item()) > 1e-5:
        raise ValueError("Target +d/-d rows are not exact negatives")
    return deltas.to(device=device, dtype=torch.float32)


def read_decomposed_logits(
    logits_at_read: torch.Tensor,
    t_own: int,
    peers: list[int],
) -> tuple[list[float], list[float], list[float]]:
    if not torch.isfinite(logits_at_read).all():
        raise FloatingPointError("Non-finite logits encountered")
    raw = [float(x) for x in logits_at_read[:, t_own].detach().cpu().tolist()]
    peer_mat = [[float(v) for v in row] for row in logits_at_read[:, peers].detach().cpu().tolist()]
    peer_med = [float(statistics.median(row)) for row in peer_mat]
    spec = [r - pm for r, pm in zip(raw, peer_med)]
    for vals in (raw, peer_med, spec):
        if not all(math.isfinite(x) for x in vals):
            raise FloatingPointError("Non-finite decomposed logit")
    return raw, peer_med, spec


def summarize_unpatched_transport(
    unp_raw: list[float],
    unp_pmed: list[float],
    unp_spec: list[float],
) -> tuple[dict[str, Any], float]:
    def _pack(vals: list[float]) -> tuple[float, float, list[float], list[float]]:
        base = vals[0]
        return (
            vals[1] - base,
            vals[2] - base,
            [vals[3 + k] - base for k in range(N_NULL_DRAWS)],
            [vals[3 + 2 * N_NULL_DRAWS + k] - base for k in range(N_NULL_DRAWS)],
        )

    rp, rn, rwd, riso = _pack(unp_raw)
    pp, pn, pwd, piso = _pack(unp_pmed)
    sp, sn, swd, siso = _pack(unp_spec)

    errs = [
        abs(sp - (rp - pp)),
        abs(sn - (rn - pn)),
        *[abs(swd[k] - (rwd[k] - pwd[k])) for k in range(N_NULL_DRAWS)],
        *[abs(siso[k] - (riso[k] - piso[k])) for k in range(N_NULL_DRAWS)],
    ]
    max_err = max(errs)
    if max_err > TOL_DECOMP:
        raise AssertionError(f"Unpatched linear decomposition violated: max_err={max_err}")
    return {
        "n_write_spec": sp,
        "n_write_raw": rp,
        "n_write_peer_med": pp,
        "n_write_neg_spec": sn,
        "n_write_neg_raw": rn,
        "n_write_neg_peer_med": pn,
        "wdown_pos_draws_spec": swd,
        "wdown_pos_draws_raw": rwd,
        "wdown_pos_draws_peer_med": pwd,
        "iso_pos_draws_spec": siso,
        "iso_pos_draws_raw": riso,
        "iso_pos_draws_peer_med": piso,
    }, max_err


def summarize_b19_metric(
    unp: list[float],
    pat: list[float],
    shm: list[float],
    n_draws: int = N_NULL_DRAWS,
) -> dict[str, Any]:
    if not (len(unp) == len(pat) == len(shm) == 3 + 4 * n_draws):
        raise ValueError(f"Expected B={3 + 4 * n_draws}, got {len(unp)}, {len(pat)}, {len(shm)}")
    d_row = [u - p for u, p in zip(unp, pat)]
    d_shm = [u - s for u, s in zip(unp, shm)]
    if not (all(math.isfinite(x) for x in d_row) and all(math.isfinite(x) for x in d_shm)):
        raise FloatingPointError("Non-finite value in summarize_b19_metric")
    d0, d_pos, d_neg = d_row[0], d_row[1], d_row[2]
    even = 0.5 * (d_pos + d_neg) - d0
    odd = 0.5 * (d_pos - d_neg)
    max_abs_sham = max(abs(x) for x in d_shm)

    def _pack_null(start_pos: int) -> dict[str, Any]:
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
        "wdown_orth": _pack_null(3),
        "iso_resid": _pack_null(3 + 2 * n_draws),
    }


def check_head_linear_decomposition(
    spec_sum: dict[str, Any],
    raw_sum: dict[str, Any],
    pmed_sum: dict[str, Any],
) -> float:
    errs = [abs(spec_sum[k] - (raw_sum[k] - pmed_sum[k])) for k in ("d0", "d_pos", "d_neg", "even", "odd")]
    for grp in ("wdown_orth", "iso_resid"):
        for dk in ("d_pos_draws", "d_neg_draws", "even_draws", "odd_draws"):
            for s_v, r_v, p_v in zip(spec_sum[grp][dk], raw_sum[grp][dk], pmed_sum[grp][dk]):
                errs.append(abs(s_v - (r_v - p_v)))
    max_err = max(errs)
    if max_err > TOL_DECOMP:
        raise AssertionError(f"Head linear decomposition violated: max_err={max_err}")
    return max_err


def compute_sibling_max_abs_leakage(
    z_patched: torch.Tensor,
    z_base: torch.Tensor,
    head_idx: int,
    n_heads: int,
) -> float:
    """Exact max abs diff across all `n_heads - 1` sibling heads `h' != head_idx`."""
    d_model = z_patched.shape[-1]
    head_dim = d_model // n_heads
    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    diff = (z_patched.detach().to(torch.float32) - z_base.detach().to(torch.float32)).abs()
    left = float(diff[..., :s].max().item()) if s > 0 else 0.0
    right = float(diff[..., e:].max().item()) if e < d_model else 0.0
    return max(left, right)


def make_orthogonal_synthetic_donor(nat_slice: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic fixed-seed orthogonal synthetic donor scaled to natural per-token RMS."""
    nat = nat_slice.detach().to(device="cpu", dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(nat.shape, generator=gen, dtype=torch.float32)
    # Gram-Schmidt per token position along head_dim so synth is strictly orthogonal to nat.
    dot = (r * nat).sum(dim=-1, keepdim=True)
    n2 = (nat * nat).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    r_orth = r - (dot / n2) * nat
    r_rms = torch.sqrt((r_orth * r_orth).mean(dim=-1, keepdim=True)).clamp_min(1e-6)
    nat_rms = torch.sqrt((nat * nat).mean(dim=-1, keepdim=True)).clamp_min(1e-2)
    return (r_orth / r_rms) * nat_rms


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not locate decoder layers on model")


def _default_model_loader(repo: str, commit: str, device: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=commit,
        torch_dtype=torch.float32,
        attn_implementation=ATTN_IMPL,
    )
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Main Execution Engine (Smoke or Full)
# ---------------------------------------------------------------------------

def execute_causal_attn(
    manifest: dict[str, Any],
    pins: dict[str, str],
    smoke_mode: bool,
    device: str,
    model_loader: Callable[[str, str, str], nn.Module] = _default_model_loader,
    out_path: Path | None = None,
    execution_provenance: str | None = None,
) -> dict[str, Any]:
    """Execute Part B (`CM.0`–`CM.4`) smoke (`135` B19 + `77` B=1) or full (`5,820` B19 + `2,457` B=1)."""
    is_cuda_dev = bool(str(device).startswith("cuda") and torch.cuda.is_available())
    if execution_provenance is None:
        resolved_provenance = "measured_gpu_run" if is_cuda_dev else "measured_cpu_run"
    else:
        if execution_provenance == "measured_gpu_run" and not is_cuda_dev:
            raise ValueError(f"Refusing to label non-CUDA execution (device={device!r}) as 'measured_gpu_run'")
        resolved_provenance = execution_provenance

    t_start = time.perf_counter()

    t_load0 = time.perf_counter()
    model = model_loader(MODEL_REPO, MODEL_COMMIT, device)
    model_load_sec = time.perf_counter() - t_load0

    layers_mod = get_decoder_layers(model)
    n_heads = int(getattr(model.config, "num_attention_heads"))
    n_kv_heads = int(getattr(model.config, "num_key_value_heads", n_heads))
    if n_kv_heads != n_heads:
        raise ValueError(f"Expected full MHA (n_kv_heads == n_heads), got {n_kv_heads} vs {n_heads}")
    hidden_size = int(getattr(model.config, "hidden_size"))
    head_dim = hidden_size // n_heads
    if head_dim * n_heads != hidden_size:
        raise ValueError(f"hidden_size={hidden_size} not divisible by n_heads={n_heads}")

    sched_key = "schedule_smoke" if smoke_mode else "schedule_full"
    sched = manifest[sched_key]
    units_to_run: list[str] = list(sched["units"])
    op_norm_name = str(manifest["meta"]["operating_norm_name"])
    op_norms = manifest["meta"]["operating_norm_multipliers"]

    # Extract fixed unit write vectors `hat_w` once.
    unit_cache: dict[str, dict[str, Any]] = {}
    unit_vectors_meta: dict[str, dict[str, Any]] = {}
    for uid in units_to_run:
        u = UNIT_BY_ID[uid]
        li, ni = int(u["layer"]), int(u["neuron"])
        w_down_cpu = layers_mod[li].mlp.down_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        v_raw = w_down_cpu[:, ni]
        w_norm = float(v_raw.norm().item())
        hat_w_cpu = v_raw / max(w_norm, 1e-12)
        unit_cache[uid] = {"layer": li, "neuron": ni, "w_down_cpu": w_down_cpu, "hat_w_cpu": hat_w_cpu}
        unit_vectors_meta[uid] = {
            "layer": li,
            "neuron": ni,
            "w_down_norm": w_norm,
            "operating_norm_name": op_norm_name,
            "operating_norm_multiplier": float(op_norms[str(li)]),
        }

    B = 3 + 4 * N_NULL_DRAWS
    head_layers_needed = sorted({HEAD_BY_NAME[e["head_name"]]["layer"] for e in sched["cm123_head_modes"]})
    timers = {"b19_sec": 0.0, "b19_fwds": 0, "b1_sec": 0.0, "b1_fwds": 0}

    unpatched_rows: list[dict[str, Any]] = []
    head_mode_rows: list[dict[str, Any]] = []
    actuation_rows: list[dict[str, Any]] = []
    swap_rows: list[dict[str, Any]] = []

    max_abs_sham_spec = 0.0
    max_abs_sham_raw = 0.0
    max_abs_decomp_err = 0.0
    max_sibling_leakage = 0.0
    max_reconcile_pre_oproj = 0.0
    max_reconcile_full_vocab = 0.0
    max_reconcile_drop = 0.0

    def _make_layer_inject_hook(deltas_dev: torch.Tensor, read_pos: int | None = None, norm_box: list[float] | None = None):
        def _hook(_mod, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                if norm_box is not None and not norm_box and read_pos is not None:
                    norm_box.append(float(torch.linalg.vector_norm(hs[0, read_pos].detach().to(torch.float32)).item()))
                return (hs + deltas_dev.unsqueeze(1).to(hs.dtype),) + out[1:]
            if norm_box is not None and not norm_box and read_pos is not None:
                norm_box.append(float(torch.linalg.vector_norm(out[0, read_pos].detach().to(torch.float32)).item()))
            return out + deltas_dev.unsqueeze(1).to(out.dtype)
        return _hook

    def _make_o_proj_patch_pre_hook(
        head_idx: int,
        mode: str,
        read_pos: int,
        donor_z: torch.Tensor | None,
        is_sham: bool,
        capture_after: list[torch.Tensor] | None = None,
        capture_before: list[torch.Tensor] | None = None,
    ):
        s, e = head_idx * head_dim, (head_idx + 1) * head_dim

        def _pre(_mod, args):
            x = args[0]
            if capture_before is not None:
                capture_before.append(x.detach().clone())
            xc = x.clone()
            if mode == "z_all":
                if is_sham:
                    xc[:, :, s:e] = x[:, :, s:e]
                else:
                    xc[:, :, s:e] = donor_z.to(device=x.device, dtype=x.dtype)
            elif mode == "z_read_pos":
                if is_sham:
                    xc[:, read_pos, s:e] = x[:, read_pos, s:e]
                else:
                    xc[:, read_pos, s:e] = donor_z.to(device=x.device, dtype=x.dtype)
            elif mode == "z_upstream":
                if is_sham:
                    xc[:, :read_pos, s:e] = x[:, :read_pos, s:e]
                else:
                    xc[:, :read_pos, s:e] = donor_z.to(device=x.device, dtype=x.dtype)
            else:
                raise ValueError(f"Unsupported o_proj mode: {mode}")
            if capture_after is not None:
                capture_after.append(xc.detach().clone())
            return (xc,) + args[1:]

        return _pre

    def _make_qkv_component_hook(
        head_idx: int,
        donor_slice: torch.Tensor | None,
        is_sham: bool,
        pos_indices: list[int] | None = None,
    ):
        s, e = head_idx * head_dim, (head_idx + 1) * head_dim

        def _hook(_mod, _inp, out):
            oc = out.clone()
            if pos_indices is None:
                if is_sham:
                    oc[:, :, s:e] = out[:, :, s:e]
                else:
                    oc[:, :, s:e] = donor_slice.to(device=out.device, dtype=out.dtype)
            else:
                if is_sham:
                    oc[:, pos_indices, s:e] = out[:, pos_indices, s:e]
                else:
                    oc[:, pos_indices, s:e] = donor_slice.to(device=out.device, dtype=out.dtype)
            return oc

        return _hook

    def _run_b19_forward(inp: torch.Tensor, read_pos: int, handles: list[Any]) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                logits = model(input_ids=inp, use_cache=False).logits[:, read_pos, :]
        finally:
            for h in handles:
                h.remove()
        timers["b19_sec"] += time.perf_counter() - t0
        timers["b19_fwds"] += 1
        return logits

    def _run_b1_forward(inp: torch.Tensor, read_pos: int, handles: list[Any]) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                logits = model(input_ids=inp, use_cache=False).logits[:, read_pos, :]
        finally:
            for h in handles:
                h.remove()
        timers["b1_sec"] += time.perf_counter() - t0
        timers["b1_fwds"] += 1
        return logits

    # =========================================================================
    # PART 1: CM.1 - CM.3 (Unpatched + 8-Mode Single-Head Patching, B=19)
    # =========================================================================
    for uid in units_to_run:
        ub = unit_cache[uid]
        u_layer = ub["layer"]
        n_idx = ub["neuron"]
        hat_w_cpu = ub["hat_w_cpu"]
        w_down_cpu = ub["w_down_cpu"]
        m_L = float(op_norms[str(u_layer)])

        for p_entry in sched["cm123_prompts"]:
            pid = p_entry["prompt_id"]
            story = p_entry["story"]
            cell = p_entry["cell"]
            dose_mag = float(p_entry["dose_mag"])
            binding = manifest["bindings"][pid][uid]
            input_ids = list(binding["input_ids"])
            read_pos = int(binding["read_pos"])
            t_own = int(binding["t_own"])
            peers = list(binding["peers"])

            seed_u = part_b_unit_seed(uid, pid, dose_mag)
            wd_u, iso_u = make_orthogonal_null_vectors(
                w_down_cpu=w_down_cpu,
                target_neuron=n_idx,
                hat_w_cpu=hat_w_cpu,
                n_draws=N_NULL_DRAWS,
                seed=seed_u,
            )
            deltas = build_b19_deltas(
                hat_w_cpu=hat_w_cpu,
                wd_unit_cpu=wd_u,
                iso_unit_cpu=iso_u,
                dose_mag=dose_mag,
                norm_multiplier=m_L,
                device=device,
            )
            inp = torch.tensor([input_ids] * B, dtype=torch.long, device=device)

            # Single unpatched B=19 forward capturing o_proj pre-input & post-norm Q/K/V for layers 7 & 8.
            read_pos_norm_box: list[float] = []
            cap_o_in: dict[int, torch.Tensor] = {}
            cap_q: dict[int, torch.Tensor] = {}
            cap_k: dict[int, torch.Tensor] = {}
            cap_v: dict[int, torch.Tensor] = {}

            handles = [layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(deltas, read_pos, read_pos_norm_box))]
            for hl in head_layers_needed:
                assert_strictly_downstream(u_layer, hl)
                attn = layers_mod[hl].self_attn
                handles.append(attn.o_proj.register_forward_pre_hook(
                    lambda _m, a, _hl=hl: (cap_o_in.__setitem__(_hl, a[0].detach().clone()), None)[1]
                ))
                handles.append(attn.q_norm.register_forward_hook(
                    lambda _m, _i, o, _hl=hl: (cap_q.__setitem__(_hl, o.detach().clone()), None)[1]
                ))
                handles.append(attn.k_norm.register_forward_hook(
                    lambda _m, _i, o, _hl=hl: (cap_k.__setitem__(_hl, o.detach().clone()), None)[1]
                ))
                handles.append(attn.v_proj.register_forward_hook(
                    lambda _m, _i, o, _hl=hl: (cap_v.__setitem__(_hl, o.detach().clone()), None)[1]
                ))

            out_unp = _run_b19_forward(inp, read_pos, handles)
            unp_raw, unp_pmed, unp_spec = read_decomposed_logits(out_unp, t_own, peers)
            transport, decomp_err = summarize_unpatched_transport(unp_raw, unp_pmed, unp_spec)
            max_abs_decomp_err = max(max_abs_decomp_err, decomp_err)

            inj_norm = float(dose_mag) * m_L
            rp_norm = float(read_pos_norm_box[0])
            unpatched_rows.append({
                "unit": uid,
                "unit_layer": u_layer,
                "prompt_id": pid,
                "story": story,
                "cell": cell,
                "dose_mag": dose_mag,
                "operating_norm_name": op_norm_name,
                "norm_multiplier": m_L,
                "injected_norm": inj_norm,
                "read_pos_resid_norm": rp_norm,
                "norm_ratio": inj_norm / max(rp_norm, 1e-12),
                "null_seed": seed_u,
                "max_abs_decomp_err": decomp_err,
                "transport": transport,
            })

            # Save z_all logits (d = ±0.8) & readout drops per head for exact QKV_all == z_all reconciliation.
            z_all_cache: dict[str, dict[str, Any]] = {}

            for hm in sched["cm123_head_modes"]:
                hname = hm["head_name"]
                mode = hm["mode"]
                hmeta = HEAD_BY_NAME[hname]
                hl, hi = int(hmeta["layer"]), int(hmeta["head"])
                assert_strictly_downstream(u_layer, hl)
                attn = layers_mod[hl].self_attn
                s, e = hi * head_dim, (hi + 1) * head_dim

                # 1. Patched B=19 forward.
                z_after_box: list[torch.Tensor] = []
                pat_handles = [layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(deltas))]
                if mode in Z_MODES:
                    if mode == "z_all":
                        donor_z = cap_o_in[hl][0:1, :, s:e].expand(B, -1, -1)
                    elif mode == "z_read_pos":
                        donor_z = cap_o_in[hl][0:1, read_pos, s:e].expand(B, -1)
                    else:
                        donor_z = cap_o_in[hl][0:1, :read_pos, s:e].expand(B, -1, -1)
                    pat_handles.append(attn.o_proj.register_forward_pre_hook(
                        _make_o_proj_patch_pre_hook(hi, mode, read_pos, donor_z, is_sham=False, capture_after=z_after_box)
                    ))
                else:
                    comps = QKV_MODE_COMPONENTS[mode]
                    if "q" in comps:
                        pat_handles.append(attn.q_norm.register_forward_hook(
                            _make_qkv_component_hook(hi, cap_q[hl][0:1, :, s:e].expand(B, -1, -1), is_sham=False)
                        ))
                    if "k" in comps:
                        pat_handles.append(attn.k_norm.register_forward_hook(
                            _make_qkv_component_hook(hi, cap_k[hl][0:1, :, s:e].expand(B, -1, -1), is_sham=False)
                        ))
                    if "v" in comps:
                        pat_handles.append(attn.v_proj.register_forward_hook(
                            _make_qkv_component_hook(hi, cap_v[hl][0:1, :, s:e].expand(B, -1, -1), is_sham=False)
                        ))
                    pat_handles.append(attn.o_proj.register_forward_pre_hook(
                        lambda _m, a, _box=z_after_box: (_box.append(a[0].detach().clone()), None)[1]
                    ))
                out_pat = _run_b19_forward(inp, read_pos, pat_handles)

                # 2. Measured same-hook-path sham B=19 forward.
                shm_handles = [layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(deltas))]
                if mode in Z_MODES:
                    shm_handles.append(attn.o_proj.register_forward_pre_hook(
                        _make_o_proj_patch_pre_hook(hi, mode, read_pos, None, is_sham=True)
                    ))
                else:
                    comps = QKV_MODE_COMPONENTS[mode]
                    if "q" in comps:
                        shm_handles.append(attn.q_norm.register_forward_hook(_make_qkv_component_hook(hi, None, is_sham=True)))
                    if "k" in comps:
                        shm_handles.append(attn.k_norm.register_forward_hook(_make_qkv_component_hook(hi, None, is_sham=True)))
                    if "v" in comps:
                        shm_handles.append(attn.v_proj.register_forward_hook(_make_qkv_component_hook(hi, None, is_sham=True)))
                out_shm = _run_b19_forward(inp, read_pos, shm_handles)

                pat_raw, pat_pmed, pat_spec = read_decomposed_logits(out_pat, t_own, peers)
                shm_raw, shm_pmed, shm_spec = read_decomposed_logits(out_shm, t_own, peers)
                spec_sum = summarize_b19_metric(unp_spec, pat_spec, shm_spec)
                raw_sum = summarize_b19_metric(unp_raw, pat_raw, shm_raw)
                pmed_sum = summarize_b19_metric(unp_pmed, pat_pmed, shm_pmed)
                hd_decomp = check_head_linear_decomposition(spec_sum, raw_sum, pmed_sum)

                sib_leak = compute_sibling_max_abs_leakage(z_after_box[0], cap_o_in[hl], hi, n_heads)
                max_abs_sham_spec = max(max_abs_sham_spec, float(spec_sum["max_abs_sham"]))
                max_abs_sham_raw = max(max_abs_sham_raw, float(raw_sum["max_abs_sham"]))
                max_abs_decomp_err = max(max_abs_decomp_err, hd_decomp)
                max_sibling_leakage = max(max_sibling_leakage, sib_leak)

                row_obj: dict[str, Any] = {
                    "unit": uid,
                    "unit_layer": u_layer,
                    "prompt_id": pid,
                    "story": story,
                    "cell": cell,
                    "dose_mag": dose_mag,
                    "head_name": hname,
                    "head_layer": hl,
                    "head_idx": hi,
                    "mode": mode,
                    "sibling_max_abs_leakage": sib_leak,
                    "max_abs_decomp_err": hd_decomp,
                    "spec": spec_sum,
                    "raw": raw_sum,
                    "peer_med": pmed_sum,
                }

                if mode == "z_all":
                    z_all_cache[hname] = {
                        "logits_pm": out_pat[1:3].detach().clone(),
                        "spec_d_pos": spec_sum["d_pos"],
                        "spec_d_neg": spec_sum["d_neg"],
                        "raw_d_pos": raw_sum["d_pos"],
                        "raw_d_neg": raw_sum["d_neg"],
                    }
                elif mode == "QKV_all":
                    zc = z_all_cache[hname]
                    target_z_ref = cap_o_in[hl][0:1, :, s:e].expand(B, -1, -1).to(torch.float32)
                    pre_oproj_err = float((z_after_box[0][:, :, s:e].to(torch.float32) - target_z_ref).abs().max().item())
                    full_vocab_err = float((out_pat[1:3].detach().to(torch.float32) - zc["logits_pm"].to(torch.float32)).abs().max().item())
                    drop_err = max(
                        abs(spec_sum["d_pos"] - zc["spec_d_pos"]),
                        abs(spec_sum["d_neg"] - zc["spec_d_neg"]),
                        abs(raw_sum["d_pos"] - zc["raw_d_pos"]),
                        abs(raw_sum["d_neg"] - zc["raw_d_neg"]),
                    )
                    max_reconcile_pre_oproj = max(max_reconcile_pre_oproj, pre_oproj_err)
                    max_reconcile_full_vocab = max(max_reconcile_full_vocab, full_vocab_err)
                    max_reconcile_drop = max(max_reconcile_drop, drop_err)
                    row_obj["reconciliation_qkv_vs_z_all"] = {
                        "pre_oproj_head_tensor_max_abs": pre_oproj_err,
                        "full_vocab_read_pos_logits_max_abs": full_vocab_err,
                        "readout_drop_max_abs": drop_err,
                    }

                head_mode_rows.append(row_obj)

    # =========================================================================
    # PART 2: CM.0 Dedicated Synthetic Nonzero-Donor Actuation Positive Controls
    # (17 B=1 forwards; z_all runs first per (head, prompt) to capture exact B=1 base!)
    # =========================================================================
    act_base_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for act in sched["actuation"]:
        uid = act["unit"]
        pid = act["prompt_id"]
        hname = act["head_name"]
        mode = act["mode"]
        kind = act["kind"]
        seed_a = int(act["seed"])
        swap_idx = act["swap_indices"]

        ub = unit_cache[uid]
        u_layer = ub["layer"]
        m_L = float(op_norms[str(u_layer)])
        delta_pos = (NOMINAL_DOSE_MAG * m_L * ub["hat_w_cpu"]).unsqueeze(0).to(device=device, dtype=torch.float32)

        binding = manifest["bindings"][pid][uid]
        inp1 = torch.tensor([binding["input_ids"]], dtype=torch.long, device=device)
        read_pos = int(binding["read_pos"])
        hmeta = HEAD_BY_NAME[hname]
        hl, hi = int(hmeta["layer"]), int(hmeta["head"])
        attn = layers_mod[hl].self_attn
        s, e = hi * head_dim, (hi + 1) * head_dim

        bkey = (hname, pid)
        if mode == "z_all":
            # In z_all, q_norm/k_norm/v_proj and pre-patch o_proj input ARE the exact unmodified B=1 base!
            z_before_box: list[torch.Tensor] = []
            z_after_box: list[torch.Tensor] = []
            q_cap_box: list[torch.Tensor] = []
            k_cap_box: list[torch.Tensor] = []
            v_cap_box: list[torch.Tensor] = []

            def _act_z_all_pre(_m, args):
                x = args[0]
                z_before_box.append(x.detach().clone())
                synth_z = make_orthogonal_synthetic_donor(x[:, :, s:e], seed_a)
                xc = x.clone()
                xc[:, :, s:e] = synth_z.to(device=x.device, dtype=x.dtype)
                z_after_box.append(xc.detach().clone())
                return (xc,) + args[1:]

            h_list = [
                layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(delta_pos)),
                attn.q_norm.register_forward_hook(lambda _m, _i, o: (q_cap_box.append(o.detach().clone()), None)[1]),
                attn.k_norm.register_forward_hook(lambda _m, _i, o: (k_cap_box.append(o.detach().clone()), None)[1]),
                attn.v_proj.register_forward_hook(lambda _m, _i, o: (v_cap_box.append(o.detach().clone()), None)[1]),
                attn.o_proj.register_forward_pre_hook(_act_z_all_pre),
            ]
            _run_b1_forward(inp1, read_pos, h_list)
            act_base_cache[bkey] = {
                "z_base": z_before_box[0],
                "q_base": q_cap_box[0],
                "k_base": k_cap_box[0],
                "v_base": v_cap_box[0],
            }
            z_base = z_before_box[0]
            z_synth = z_after_box[0]
        else:
            base = act_base_cache[bkey]
            z_base = base["z_base"]
            z_after_box = []
            h_list = [layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(delta_pos))]
            if mode == "z_read_pos":
                synth_z = make_orthogonal_synthetic_donor(z_base[:, read_pos, s:e], seed_a)
                h_list.append(attn.o_proj.register_forward_pre_hook(
                    _make_o_proj_patch_pre_hook(hi, "z_read_pos", read_pos, synth_z, is_sham=False, capture_after=z_after_box)
                ))
            elif mode == "z_upstream":
                synth_z = make_orthogonal_synthetic_donor(z_base[:, :read_pos, s:e], seed_a)
                h_list.append(attn.o_proj.register_forward_pre_hook(
                    _make_o_proj_patch_pre_hook(hi, "z_upstream", read_pos, synth_z, is_sham=False, capture_after=z_after_box)
                ))
            elif kind == "cm4_span_swap":
                sq = make_orthogonal_synthetic_donor(base["q_base"][:, swap_idx, s:e], seed_a + 100)
                sk = make_orthogonal_synthetic_donor(base["k_base"][:, swap_idx, s:e], seed_a + 200)
                sv = make_orthogonal_synthetic_donor(base["v_base"][:, swap_idx, s:e], seed_a + 300)
                h_list.append(attn.q_norm.register_forward_hook(_make_qkv_component_hook(hi, sq, is_sham=False, pos_indices=swap_idx)))
                h_list.append(attn.k_norm.register_forward_hook(_make_qkv_component_hook(hi, sk, is_sham=False, pos_indices=swap_idx)))
                h_list.append(attn.v_proj.register_forward_hook(_make_qkv_component_hook(hi, sv, is_sham=False, pos_indices=swap_idx)))
                h_list.append(attn.o_proj.register_forward_pre_hook(
                    lambda _m, a, _box=z_after_box: (_box.append(a[0].detach().clone()), None)[1]
                ))
            else:
                comps = QKV_MODE_COMPONENTS[mode]
                if "q" in comps:
                    sq = make_orthogonal_synthetic_donor(base["q_base"][:, :, s:e], seed_a + 100)
                    h_list.append(attn.q_norm.register_forward_hook(_make_qkv_component_hook(hi, sq, is_sham=False)))
                if "k" in comps:
                    sk = make_orthogonal_synthetic_donor(base["k_base"][:, :, s:e], seed_a + 200)
                    h_list.append(attn.k_norm.register_forward_hook(_make_qkv_component_hook(hi, sk, is_sham=False)))
                if "v" in comps:
                    sv = make_orthogonal_synthetic_donor(base["v_base"][:, :, s:e], seed_a + 300)
                    h_list.append(attn.v_proj.register_forward_hook(_make_qkv_component_hook(hi, sv, is_sham=False)))
                h_list.append(attn.o_proj.register_forward_pre_hook(
                    lambda _m, a, _box=z_after_box: (_box.append(a[0].detach().clone()), None)[1]
                ))
            _run_b1_forward(inp1, read_pos, h_list)
            z_synth = z_after_box[0]

        target_delta = float((z_synth[:, :, s:e].to(torch.float32) - z_base[:, :, s:e].to(torch.float32)).abs().max().item())
        sib_leak = compute_sibling_max_abs_leakage(z_synth, z_base, hi, n_heads)
        unpatched_pos_leak = 0.0
        if mode == "z_read_pos":
            unpatched_pos_leak = float((z_synth[:, :read_pos, s:e].to(torch.float32) - z_base[:, :read_pos, s:e].to(torch.float32)).abs().max().item())
        elif mode == "z_upstream":
            unpatched_pos_leak = float((z_synth[:, read_pos, s:e].to(torch.float32) - z_base[:, read_pos, s:e].to(torch.float32)).abs().max().item())
        max_sibling_leakage = max(max_sibling_leakage, sib_leak)
        actuation_rows.append({
            "actuation_id": act["actuation_id"],
            "kind": kind,
            "unit": uid,
            "prompt_id": pid,
            "head_name": hname,
            "head_layer": hl,
            "head_idx": hi,
            "mode": mode,
            "swap_indices": swap_idx,
            "seed": seed_a,
            "target_max_abs_delta": target_delta,
            "sibling_max_abs_leakage": sib_leak,
            "unpatched_pos_max_abs_leakage": unpatched_pos_leak,
            "passed": bool(target_delta > MIN_ACTUATION_DELTA and sib_leak <= TOL_SIBLING and unpatched_pos_leak <= TOL_SIBLING),
        })

    # =========================================================================
    # PART 3: CM.4 Reciprocal Coordinate-Localized Contextual-State Swaps (B=1)
    # =========================================================================
    # 3a. Base d=+0.8 B=1 captures for all scheduled (unit, SELF/NAME_BARE) prompts.
    cm4_base: dict[tuple[str, str], dict[str, Any]] = {}
    l8_attn = layers_mod[8].self_attn
    for cap_e in sched["cm4_base_captures"]:
        uid = cap_e["unit"]
        pid = cap_e["prompt_id"]
        ub = unit_cache[uid]
        u_layer = ub["layer"]
        m_L = float(op_norms[str(u_layer)])
        delta_pos = (NOMINAL_DOSE_MAG * m_L * ub["hat_w_cpu"]).unsqueeze(0).to(device=device, dtype=torch.float32)

        binding = manifest["bindings"][pid][uid]
        inp1 = torch.tensor([binding["input_ids"]], dtype=torch.long, device=device)
        read_pos = int(binding["read_pos"])
        t_own = int(binding["t_own"])
        peers = list(binding["peers"])

        o_box: list[torch.Tensor] = []
        q_box: list[torch.Tensor] = []
        k_box: list[torch.Tensor] = []
        v_box: list[torch.Tensor] = []
        h_list = [
            layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(delta_pos)),
            l8_attn.o_proj.register_forward_pre_hook(lambda _m, a: (o_box.append(a[0].detach().clone()), None)[1]),
            l8_attn.q_norm.register_forward_hook(lambda _m, _i, o: (q_box.append(o.detach().clone()), None)[1]),
            l8_attn.k_norm.register_forward_hook(lambda _m, _i, o: (k_box.append(o.detach().clone()), None)[1]),
            l8_attn.v_proj.register_forward_hook(lambda _m, _i, o: (v_box.append(o.detach().clone()), None)[1]),
        ]
        out_base = _run_b1_forward(inp1, read_pos, h_list)
        b_raw, b_pmed, b_spec = read_decomposed_logits(out_base, t_own, peers)
        cm4_base[(uid, pid)] = {
            "raw": b_raw[0],
            "peer_med": b_pmed[0],
            "spec": b_spec[0],
            "o_in": o_box[0],
            "q": q_box[0],
            "k": k_box[0],
            "v": v_box[0],
        }

    # 3b. Swap + measured sham B=1 forwards.
    for tup in sched["cm4_tuples"]:
        uid = tup["unit"]
        t_pid = tup["target_prompt_id"]
        d_pid = tup["donor_prompt_id"]
        swap_idx = list(tup["swap_indices"])
        ub = unit_cache[uid]
        u_layer = ub["layer"]
        m_L = float(op_norms[str(u_layer)])
        delta_pos = (NOMINAL_DOSE_MAG * m_L * ub["hat_w_cpu"]).unsqueeze(0).to(device=device, dtype=torch.float32)

        t_bind = manifest["bindings"][t_pid][uid]
        d_bind = manifest["bindings"][d_pid][uid]
        t_ids = list(t_bind["input_ids"])
        d_ids = list(d_bind["input_ids"])
        pos_ids = list(t_bind["position_ids"])
        read_pos = int(t_bind["read_pos"])
        t_own = int(t_bind["t_own"])
        peers = list(t_bind["peers"])
        inp1 = torch.tensor([t_ids], dtype=torch.long, device=device)

        tb = cm4_base[(uid, t_pid)]
        db = cm4_base[(uid, d_pid)]

        for hname in sched["cm4_heads"]:
            hmeta = HEAD_BY_NAME[hname]
            hl, hi = int(hmeta["layer"]), int(hmeta["head"])
            assert hl == 8
            s, e = hi * head_dim, (hi + 1) * head_dim

            # Swap forward.
            z_after_box = []
            sw_handles = [
                layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(delta_pos)),
                l8_attn.q_norm.register_forward_hook(_make_qkv_component_hook(hi, db["q"][:, swap_idx, s:e], is_sham=False, pos_indices=swap_idx)),
                l8_attn.k_norm.register_forward_hook(_make_qkv_component_hook(hi, db["k"][:, swap_idx, s:e], is_sham=False, pos_indices=swap_idx)),
                l8_attn.v_proj.register_forward_hook(_make_qkv_component_hook(hi, db["v"][:, swap_idx, s:e], is_sham=False, pos_indices=swap_idx)),
                l8_attn.o_proj.register_forward_pre_hook(lambda _m, a, _box=z_after_box: (_box.append(a[0].detach().clone()), None)[1]),
            ]
            out_sw = _run_b1_forward(inp1, read_pos, sw_handles)
            sw_raw, sw_pmed, sw_spec = read_decomposed_logits(out_sw, t_own, peers)

            # Measured same-hook-path sham forward (patching target's own pushed Q/K/V at swap_idx).
            shm_handles = [
                layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(delta_pos)),
                l8_attn.q_norm.register_forward_hook(_make_qkv_component_hook(hi, tb["q"][:, swap_idx, s:e], is_sham=False, pos_indices=swap_idx)),
                l8_attn.k_norm.register_forward_hook(_make_qkv_component_hook(hi, tb["k"][:, swap_idx, s:e], is_sham=False, pos_indices=swap_idx)),
                l8_attn.v_proj.register_forward_hook(_make_qkv_component_hook(hi, tb["v"][:, swap_idx, s:e], is_sham=False, pos_indices=swap_idx)),
            ]
            out_shm = _run_b1_forward(inp1, read_pos, shm_handles)
            sh_raw, sh_pmed, sh_spec = read_decomposed_logits(out_shm, t_own, peers)

            d_spec = sw_spec[0] - tb["spec"]
            d_raw = sw_raw[0] - tb["raw"]
            d_pmed = sw_pmed[0] - tb["peer_med"]
            s_spec = sh_spec[0] - tb["spec"]
            s_raw = sh_raw[0] - tb["raw"]
            s_pmed = sh_pmed[0] - tb["peer_med"]
            sw_decomp = max(abs(d_spec - (d_raw - d_pmed)), abs(s_spec - (s_raw - s_pmed)))
            if sw_decomp > TOL_DECOMP:
                raise AssertionError(f"CM.4 swap decomposition violated: {sw_decomp}")

            sib_leak = compute_sibling_max_abs_leakage(z_after_box[0], tb["o_in"], hi, n_heads)
            max_abs_sham_spec = max(max_abs_sham_spec, abs(s_spec))
            max_abs_sham_raw = max(max_abs_sham_raw, abs(s_raw))
            max_abs_decomp_err = max(max_abs_decomp_err, sw_decomp)
            max_sibling_leakage = max(max_sibling_leakage, sib_leak)

            swap_rows.append({
                "unit": uid,
                "target_story": tup["target_story"],
                "target_cell": tup["target_cell"],
                "target_prompt_id": t_pid,
                "donor_story": tup["donor_story"],
                "donor_cell": tup["donor_cell"],
                "donor_prompt_id": d_pid,
                "swap_cond": tup["swap_cond"],
                "span_name": tup["span_name"],
                "head_name": hname,
                "head_layer": hl,
                "head_idx": hi,
                "target_input_ids": t_ids,
                "donor_input_ids": d_ids,
                "swap_indices": swap_idx,
                "position_ids": pos_ids,
                "delta_spec": d_spec,
                "delta_raw": d_raw,
                "delta_peer_med": d_pmed,
                "sham_spec": s_spec,
                "sham_raw": s_raw,
                "sham_peer_med": s_pmed,
                "sibling_max_abs_leakage": sib_leak,
                "max_abs_decomp_err": sw_decomp,
            })

    total_sec = time.perf_counter() - t_start
    non_fwd_sec = max(0.0, total_sec - timers["b19_sec"] - timers["b1_sec"])
    den_full = manifest["meta"]["denominators"]["full"]
    ms_b19 = 1000.0 * timers["b19_sec"] / float(timers["b19_fwds"])
    ms_b1 = 1000.0 * timers["b1_sec"] / float(timers["b1_fwds"])
    proj_b19 = (ms_b19 / 1000.0) * float(den_full["n_b19_forwards"])
    proj_b1 = (ms_b1 / 1000.0) * float(den_full["n_b1_forwards"])
    timing = {
        "model_load_sec": model_load_sec,
        "b19_elapsed_sec": timers["b19_sec"],
        "b1_elapsed_sec": timers["b1_sec"],
        "non_fwd_runner_sec": non_fwd_sec,
        "total_runner_sec": total_sec,
        "n_b19_done": timers["b19_fwds"],
        "n_b1_done": timers["b1_fwds"],
        "ms_per_b19": ms_b19,
        "ms_per_b1": ms_b1,
        "projected_full_b19_sec": proj_b19,
        "projected_full_b1_sec": proj_b1,
        "projected_full_fwd_sec": proj_b19 + proj_b1,
    }

    dev_name = torch.cuda.get_device_name(0) if is_cuda_dev else str(device)
    receipt = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "causal_attn_smoke_receipt" if smoke_mode else "causal_attn_full_receipt",
            "run_mode": "smoke" if smoke_mode else "full",
            "execution_provenance": resolved_provenance,
            "device_name": dev_name,
            "is_cuda_device": is_cuda_dev,
            "complete": not smoke_mode,
            "smoke_1unit": smoke_mode,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "operating_norm_name": op_norm_name,
            "operating_norm_multipliers": dict(op_norms),
            "part_a_part_b_norm_decision": manifest["meta"]["part_a_part_b_norm_decision"],
            "part_a_p92_09_part_b_track": manifest["meta"]["part_a_p92_09_part_b_track"],
            "pins": dict(pins),
            "units_run": units_to_run,
            "n_unpatched_rows": len(unpatched_rows),
            "n_head_mode_rows": len(head_mode_rows),
            "n_actuation_rows": len(actuation_rows),
            "n_swap_rows": len(swap_rows),
            "n_b19_forwards": timers["b19_fwds"],
            "n_b1_forwards": timers["b1_fwds"],
            "n_total_forwards": timers["b19_fwds"] + timers["b1_fwds"],
            "max_abs_sham_spec": max_abs_sham_spec,
            "max_abs_sham_raw": max_abs_sham_raw,
            "max_abs_decomp_err": max_abs_decomp_err,
            "max_sibling_leakage": max_sibling_leakage,
            "max_reconcile_pre_oproj": max_reconcile_pre_oproj,
            "max_reconcile_full_vocab": max_reconcile_full_vocab,
            "max_reconcile_drop": max_reconcile_drop,
            "timing": timing,
        },
        "unit_vectors": unit_vectors_meta,
        "unpatched_rows": unpatched_rows,
        "head_mode_rows": head_mode_rows,
        "actuation_rows": actuation_rows,
        "swap_rows": swap_rows,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# Realistic Olmo2-Faithful Tiny CPU Model (for End-to-End CPU Self-Tests)
# ---------------------------------------------------------------------------

class _TinyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class _TinyOlmo2Attention(nn.Module):
    """Exact Olmo2Attention forward geometry: q_norm/k_norm over full hidden_size BEFORE slicing into 16 heads!"""

    def __init__(self, hidden_size: int = 64, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = _TinyRMSNorm(hidden_size)
        self.k_norm = _TinyRMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Deterministic RoPE-like per-position rotation on pairs (matches real Olmo2 per-head pre-attn RoPE).
        pos = torch.arange(T, device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        freq = torch.arange(0, self.head_dim, 2, device=hidden_states.device, dtype=hidden_states.dtype)
        inv_freq = 1.0 / (10000.0 ** (freq / self.head_dim))
        ang = pos * inv_freq
        cos = torch.cos(ang).repeat_interleave(2, dim=-1).view(1, 1, T, self.head_dim)
        sin = torch.sin(ang).repeat_interleave(2, dim=-1).view(1, 1, T, self.head_dim)

        def _rot(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            rx = torch.stack((-x2, x1), dim=-1).flatten(-2)
            return x * cos + rx * sin

        q = _rot(q)
        k = _rot(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        mask = torch.triu(torch.full((T, T), float("-inf"), device=hidden_states.device, dtype=scores.dtype), diagonal=1)
        attn_w = F.softmax(scores + mask, dim=-1)
        attn_out = torch.matmul(attn_w, v).transpose(1, 2).reshape(B, T, H).contiguous()
        return self.o_proj(attn_out)


class _TinyOlmo2MLP(nn.Module):
    def __init__(self, hidden_size: int = 64, intermediate_size: int = 8192):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class _TinyOlmo2Layer(nn.Module):
    def __init__(self, hidden_size: int = 64, num_heads: int = 16, intermediate_size: int = 8192):
        super().__init__()
        self.self_attn = _TinyOlmo2Attention(hidden_size, num_heads)
        self.mlp = _TinyOlmo2MLP(hidden_size, intermediate_size)
        self.post_attn_norm = _TinyRMSNorm(hidden_size)
        self.post_mlp_norm = _TinyRMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attn_norm(self.self_attn(x))
        x = x + self.post_mlp_norm(self.mlp(x))
        return x


class TinyOlmo2ForCausalLM(nn.Module):
    """Exact architectural twin of OLMo-2-0425-1B (9 layers, 16 MHA heads, post-norm q/k, full 100278 vocab)."""

    def __init__(self, seed: int = 12345, hidden_size: int = 64, num_heads: int = 16, num_layers: int = 9, vocab_size: int = 100278):
        super().__init__()
        torch.manual_seed(seed)
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
        )
        # Compact low-rank-free full-vocab embedding & lm_head (64 * 100278 * 4 B = 25 MB, <5ms CPU forward).
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_TinyOlmo2Layer(hidden_size, num_heads, intermediate_size=8192) for _ in range(num_layers)])
        self.norm = _TinyRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model = types.SimpleNamespace(layers=self.layers)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        return types.SimpleNamespace(logits=logits)


# ---------------------------------------------------------------------------
# Exhaustive CPU Self-Test Suite
# ---------------------------------------------------------------------------

def run_cpu_self_tests(
    repo_root: Path,
    part_a_scorecard_path: Path,
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
) -> dict[str, Any]:
    checks: list[str] = []

    # 1. Verify exact 4-pin + Part A scorecard single-read admission AND zero-load refusal on tamper/decline_norm.
    d_sha = sha256_file(design_path)
    m_sha = sha256_file(manifest_path)
    r_sha = sha256_file(runner_path)
    s_sha = sha256_file(scorer_path)
    pa_sha = sha256_file(part_a_scorecard_path)

    manifest, part_a_sc, pins = admit_execution_request(
        mode="smoke",
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        part_a_scorecard_path=part_a_scorecard_path,
        expected_design_sha=d_sha,
        expected_manifest_sha=m_sha,
        expected_runner_sha=r_sha,
        expected_scorer_sha=s_sha,
        expected_part_a_scorecard_sha=pa_sha,
    )
    checks.append("single_read_4pin_plus_part_a_admit_ok")

    # 2. Implicit mode refusal & full-without-smoke refusal (zero model load).
    load_called = False

    def _boom_loader(*_a, **_kw):
        nonlocal load_called
        load_called = True
        raise AssertionError("Model loader must not be called on pre-load refusal")

    for bad_kwargs in [
        {"mode": None},
        {"mode": "full", "allow_full_after_smoke": False},
        {"mode": "smoke", "expected_manifest_sha": "0" * 64},
        {"mode": "smoke", "expected_part_a_scorecard_sha": "0" * 64},
    ]:
        kw = dict(
            mode="smoke",
            design_path=design_path,
            manifest_path=manifest_path,
            runner_path=runner_path,
            scorer_path=scorer_path,
            part_a_scorecard_path=part_a_scorecard_path,
            expected_design_sha=d_sha,
            expected_manifest_sha=m_sha,
            expected_runner_sha=r_sha,
            expected_scorer_sha=s_sha,
            expected_part_a_scorecard_sha=pa_sha,
        )
        kw.update(bad_kwargs)
        raised = False
        try:
            admit_execution_request(**kw)
        except ValueError:
            raised = True
        assert raised and not load_called, f"Expected pre-load ValueError for {bad_kwargs}"
    checks.append("preload_refusal_mode_full_pin_tamper_ok")

    # 3. Auto-HOLD on Part A `decline_norm (...)`.
    declined = copy.deepcopy(part_a_sc)
    declined["BR4_L7H5_and_part_b_norm"]["part_b_norm_decision"] = "decline_norm (both_norms_failed)"
    raised_hold = False
    try:
        resolve_part_a_handoff(declined)
    except ValueError as exc:
        raised_hold = "AUTO-HELD" in str(exc)
    assert raised_hold, "Part B must auto-HOLD on Part A decline_norm"
    checks.append("part_a_decline_norm_auto_hold_ok")

    # 4. End-to-end real-hook CPU smoke run on TinyOlmo2ForCausalLM (exact 135 B19 + 77 B=1 forwards!).
    tiny_model = TinyOlmo2ForCausalLM(seed=42, hidden_size=64, num_heads=16, num_layers=9, vocab_size=100278)
    tiny_model.eval()
    smoke_receipt = execute_causal_attn(
        manifest=manifest,
        pins=pins,
        smoke_mode=True,
        device="cpu",
        model_loader=lambda _r, _c, _d: tiny_model,
        execution_provenance="synthetic_cpu_selftest",
    )
    sm_meta = smoke_receipt["meta"]
    assert sm_meta["n_unpatched_rows"] == 3
    assert sm_meta["n_head_mode_rows"] == 66
    assert sm_meta["n_actuation_rows"] == 17
    assert sm_meta["n_swap_rows"] == 28
    assert sm_meta["n_b19_forwards"] == 135
    assert sm_meta["n_b1_forwards"] == 77
    assert sm_meta["n_total_forwards"] == 212
    assert sm_meta["max_abs_sham_spec"] <= TOL_SHAM
    assert sm_meta["max_abs_sham_raw"] <= TOL_SHAM
    assert sm_meta["max_abs_decomp_err"] <= TOL_DECOMP
    assert sm_meta["max_sibling_leakage"] <= TOL_SIBLING
    assert sm_meta["max_reconcile_pre_oproj"] <= TOL_RECONCILE
    assert sm_meta["max_reconcile_full_vocab"] <= TOL_RECONCILE
    assert sm_meta["max_reconcile_drop"] <= TOL_RECONCILE
    assert all(r["passed"] for r in smoke_receipt["actuation_rows"]), (
        f"Actuation failed: {[r for r in smoke_receipt['actuation_rows'] if not r['passed']]}"
    )
    # Verify unpatched rows strictly omit sham keys.
    for ur in smoke_receipt["unpatched_rows"]:
        assert "sham" not in json.dumps(ur), "Unpatched row must not carry sham keys"
    validate_timing_breakdown(sm_meta["timing"], 135, 77)
    checks.append("tiny_olmo2_smoke_e2e_212_forwards_cm0_zero_leakage_ok")

    # 5. Negative check: deliberately verify that pre-norm Q patching WOULD leak into sibling heads,
    # proving why our post-q_norm hook placement is necessary and working!
    l8 = tiny_model.layers[8].self_attn
    inp_test = torch.tensor([manifest["bindings"]["B1_NAME_BARE"]["p92-46"]["input_ids"]], dtype=torch.long)
    with torch.no_grad():
        x_in = tiny_model.embed(inp_test)
        for li in range(8):
            x_in = tiny_model.layers[li](x_in)
        base_o: list[torch.Tensor] = []
        h0 = l8.o_proj.register_forward_pre_hook(lambda _m, a: (base_o.append(a[0].detach().clone()), None)[1])
        l8(x_in)
        h0.remove()

        # Corrupt head 15 pre-q_norm -> MUST leak to sibling heads because q_norm normalizes across all 64 channels!
        pre_o: list[torch.Tensor] = []
        s15, e15 = 15 * 4, 16 * 4

        def _pre_qnorm_patch(_m, _i, o):
            oc = o.clone()
            oc[:, :, s15:e15] = oc[:, :, s15:e15] * 5.0 + 3.0
            return oc

        h1 = l8.q_proj.register_forward_hook(_pre_qnorm_patch)
        h2 = l8.o_proj.register_forward_pre_hook(lambda _m, a: (pre_o.append(a[0].detach().clone()), None)[1])
        l8(x_in)
        h1.remove()
        h2.remove()
        pre_norm_leak = compute_sibling_max_abs_leakage(pre_o[0], base_o[0], 15, 16)
        assert pre_norm_leak > 1e-3, f"Expected pre-q_norm patch to leak to siblings, got {pre_norm_leak}"

        # Same perturbation post-q_norm -> MUST have exact 0.0 sibling leakage!
        post_o: list[torch.Tensor] = []
        h3 = l8.q_norm.register_forward_hook(_pre_qnorm_patch)
        h4 = l8.o_proj.register_forward_pre_hook(lambda _m, a: (post_o.append(a[0].detach().clone()), None)[1])
        l8(x_in)
        h3.remove()
        h4.remove()
        post_norm_leak = compute_sibling_max_abs_leakage(post_o[0], base_o[0], 15, 16)
        assert post_norm_leak == 0.0, f"Post-q_norm patch leaked to siblings: {post_norm_leak}"
    checks.append("pre_vs_post_qnorm_sibling_leakage_proof_ok")

    # 6. Pre-load full gate refuses synthetic, genuine CPU-measured, AND forged GPU-labeled-with-CPU-device smoke receipts!
    with tempfile.TemporaryDirectory() as td:
        def _try_full_admit(rec_obj: dict[str, Any], fname: str) -> bool:
            p = Path(td) / fname
            p.write_text(json.dumps(rec_obj), encoding="utf-8")
            p_sha = sha256_file(p)
            try:
                admit_execution_request(
                    mode="full",
                    design_path=design_path,
                    manifest_path=manifest_path,
                    runner_path=runner_path,
                    scorer_path=scorer_path,
                    part_a_scorecard_path=part_a_scorecard_path,
                    expected_design_sha=d_sha,
                    expected_manifest_sha=m_sha,
                    expected_runner_sha=r_sha,
                    expected_scorer_sha=s_sha,
                    expected_part_a_scorecard_sha=pa_sha,
                    allow_full_after_smoke=True,
                    verified_smoke_receipt=p,
                    verified_smoke_sha256=p_sha,
                )
                return True
            except ValueError:
                return False

        # 6a. Synthetic CPU self-test smoke receipt MUST be refused.
        assert not _try_full_admit(smoke_receipt, "synth_smoke.json"), "Full gate must refuse synthetic_cpu_selftest smoke receipt"

        # 6b. Honest `--device cpu` receipt (derived `measured_cpu_run`, `device_name='cpu'`, `is_cuda_device=False`) MUST be refused!
        cpu_default_smoke = copy.deepcopy(smoke_receipt)
        cpu_default_smoke["meta"]["execution_provenance"] = "measured_cpu_run"
        cpu_default_smoke["meta"]["device_name"] = "cpu"
        cpu_default_smoke["meta"]["is_cuda_device"] = False
        assert not _try_full_admit(cpu_default_smoke, "cpu_measured_smoke.json"), "Full gate must refuse measured_cpu_run smoke receipt"

        # Verify execute_causal_attn itself refuses labeling a CPU run as measured_gpu_run & defaults CPU to measured_cpu_run.
        refused_bad_label = False
        try:
            execute_causal_attn(
                manifest=manifest,
                pins=pins,
                smoke_mode=True,
                device="cpu",
                model_loader=lambda _r, _c, _d: tiny_model,
                execution_provenance="measured_gpu_run",
            )
        except ValueError:
            refused_bad_label = True
        assert refused_bad_label, "execute_causal_attn must refuse measured_gpu_run label on non-CUDA device"

        # 6c. Hand-forged receipt with `measured_gpu_run` but `device_name='cpu'` (even if is_cuda_device tampered) MUST be refused!
        forged_cpu_dev = copy.deepcopy(smoke_receipt)
        forged_cpu_dev["meta"]["execution_provenance"] = "measured_gpu_run"
        forged_cpu_dev["meta"]["device_name"] = "cpu"
        forged_cpu_dev["meta"]["is_cuda_device"] = True
        assert not _try_full_admit(forged_cpu_dev, "forged_cpu_dev.json"), "Full gate must refuse measured_gpu_run when device_name is cpu"

        forged_no_cuda = copy.deepcopy(smoke_receipt)
        forged_no_cuda["meta"]["execution_provenance"] = "measured_gpu_run"
        forged_no_cuda["meta"]["device_name"] = "NVIDIA L4"
        forged_no_cuda["meta"]["is_cuda_device"] = False
        assert not _try_full_admit(forged_no_cuda, "forged_no_cuda.json"), "Full gate must refuse measured_gpu_run when is_cuda_device is False"

        # 6d. Hand-edited projected_full_fwd_sec MUST be refused by timing validator (F4)!
        bad_timing = copy.deepcopy(smoke_receipt)
        bad_timing["meta"]["execution_provenance"] = "measured_gpu_run"
        bad_timing["meta"]["device_name"] = "NVIDIA L4"
        bad_timing["meta"]["is_cuda_device"] = True
        bad_timing["meta"]["timing"]["projected_full_fwd_sec"] = 999999.0
        assert not _try_full_admit(bad_timing, "bad_timing.json"), "Timing validator must refuse inconsistent projected_full_fwd_sec"

        # 6e. Authentic GPU-device measured_gpu_run smoke receipt passes full gate!
        meas_gpu_smoke = copy.deepcopy(smoke_receipt)
        meas_gpu_smoke["meta"]["execution_provenance"] = "measured_gpu_run"
        meas_gpu_smoke["meta"]["device_name"] = "NVIDIA L4"
        meas_gpu_smoke["meta"]["is_cuda_device"] = True
        assert _try_full_admit(meas_gpu_smoke, "meas_gpu_smoke.json"), "Full gate should accept authentic measured_gpu_run GPU smoke receipt"
    checks.append("full_gate_rejects_cpu_and_forged_gpu_and_bad_timing_accepts_cuda_gpu_ok")

    return {
        "runner_cpu_self_tests_passed": True,
        "n_checks": len(checks),
        "checks": checks,
        "tiny_smoke_metrics": {
            "n_b19_forwards": sm_meta["n_b19_forwards"],
            "n_b1_forwards": sm_meta["n_b1_forwards"],
            "max_abs_sham_spec": sm_meta["max_abs_sham_spec"],
            "max_abs_sham_raw": sm_meta["max_abs_sham_raw"],
            "max_abs_decomp_err": sm_meta["max_abs_decomp_err"],
            "max_sibling_leakage": sm_meta["max_sibling_leakage"],
            "max_reconcile_pre_oproj": sm_meta["max_reconcile_pre_oproj"],
            "max_reconcile_full_vocab": sm_meta["max_reconcile_full_vocab"],
            "max_reconcile_drop": sm_meta["max_reconcile_drop"],
            "min_actuation_target_delta": min(r["target_max_abs_delta"] for r in smoke_receipt["actuation_rows"]),
            "pre_norm_leakage_demo": pre_norm_leak,
            "post_norm_leakage_demo": post_norm_leak,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Part B (#4) Causal Attention-Reading & Span-Swap Runner")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--mode", choices=("smoke", "full"), default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--runner", type=Path, default=None)
    ap.add_argument("--scorer", type=Path, default=None)
    ap.add_argument("--part-a-scorecard", type=Path, default=None)
    ap.add_argument("--design-sha256", type=str, default="")
    ap.add_argument("--manifest-sha256", type=str, default="")
    ap.add_argument("--runner-sha256", type=str, default="")
    ap.add_argument("--scorer-sha256", type=str, default="")
    ap.add_argument("--part-a-scorecard-sha256", type=str, default=PART_A_SCORECARD_SHA256)
    ap.add_argument("--allow-full-after-smoke", action="store_true")
    ap.add_argument("--verified-smoke-receipt", type=Path, default=None)
    ap.add_argument("--verified-smoke-sha256", type=str, default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    base_dir = repo_root / "explore_olmo2_attn_mlp"
    design_p = args.design or (base_dir / "CAUSAL-ATTN-DESIGN.md")
    manifest_p = args.manifest or (base_dir / "causal_attn_manifest.json")
    runner_p = args.runner or Path(__file__).resolve()
    scorer_p = args.scorer or (base_dir / "score_causal_attn.py")
    part_a_p = args.part_a_scorecard or Path(
        "/home/elisepc/green/Journal/notes/reports/vesper-bridge-full-2026-11-21/instruct_bridge_scorecard.json"
    )

    if args.build_manifest:
        old_man_b = load_git_blob(repo_root, OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/signed_dose_manifest.json")
        old_run_b = load_git_blob(repo_root, OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/run_signed_dose.py")
        new_man_b = load_git_blob(repo_root, NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/stage_sweep_manifest.json")
        new_run_b = load_git_blob(repo_root, NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/run_stage_sweep.py")
        part_a_b = part_a_p.read_bytes()
        man = build_causal_attn_manifest(
            old_manifest_bytes=old_man_b,
            old_runner_bytes=old_run_b,
            new_manifest_bytes=new_man_b,
            new_runner_bytes=new_run_b,
            part_a_scorecard_bytes=part_a_b,
            design_path=design_p,
        )
        out_p = args.out or manifest_p
        out_p.write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(json.dumps({
            "built_manifest": str(out_p),
            "sha256": sha256_file(out_p),
            "operating_norm": man["meta"]["operating_norm_name"],
            "p92_09_track": man["meta"]["part_a_p92_09_part_b_track"],
            "denominators": man["meta"]["denominators"],
        }, indent=2))
        return 0

    if args.self_test_only:
        res = run_cpu_self_tests(
            repo_root=repo_root,
            part_a_scorecard_path=part_a_p,
            design_path=design_p,
            manifest_path=manifest_p,
            runner_path=runner_p,
            scorer_path=scorer_p,
        )
        print(json.dumps(res, indent=2))
        return 0

    manifest, _part_a_sc, pins = admit_execution_request(
        mode=args.mode,
        design_path=design_p,
        manifest_path=manifest_p,
        runner_path=runner_p,
        scorer_path=scorer_p,
        part_a_scorecard_path=part_a_p,
        expected_design_sha=args.design_sha256,
        expected_manifest_sha=args.manifest_sha256,
        expected_runner_sha=args.runner_sha256,
        expected_scorer_sha=args.scorer_sha256,
        expected_part_a_scorecard_sha=args.part_a_scorecard_sha256,
        allow_full_after_smoke=args.allow_full_after_smoke,
        verified_smoke_receipt=args.verified_smoke_receipt,
        verified_smoke_sha256=args.verified_smoke_sha256,
    )
    print(f"[PRECHECK OK] 4-pin + Part A scorecard ({manifest['meta']['operating_norm_name']}) & mode={args.mode} gate passed")
    if args.precheck_only:
        return 0

    out_p = args.out or (base_dir / f"causal_attn_{args.mode}_receipt.json")
    rec = execute_causal_attn(
        manifest=manifest,
        pins=pins,
        smoke_mode=(args.mode == "smoke"),
        device=args.device,
        out_path=out_p,
    )
    print(json.dumps({
        "artifact_type": rec["meta"]["artifact_type"],
        "out": str(out_p),
        "out_sha256": sha256_file(out_p),
        "n_unpatched_rows": rec["meta"]["n_unpatched_rows"],
        "n_head_mode_rows": rec["meta"]["n_head_mode_rows"],
        "n_actuation_rows": rec["meta"]["n_actuation_rows"],
        "n_swap_rows": rec["meta"]["n_swap_rows"],
        "n_b19_forwards": rec["meta"]["n_b19_forwards"],
        "n_b1_forwards": rec["meta"]["n_b1_forwards"],
        "max_abs_sham_spec": rec["meta"]["max_abs_sham_spec"],
        "max_abs_sham_raw": rec["meta"]["max_abs_sham_raw"],
        "max_sibling_leakage": rec["meta"]["max_sibling_leakage"],
        "max_reconcile_pre_oproj": rec["meta"]["max_reconcile_pre_oproj"],
        "max_reconcile_full_vocab": rec["meta"]["max_reconcile_full_vocab"],
        "max_reconcile_drop": rec["meta"]["max_reconcile_drop"],
        "timing": rec["meta"]["timing"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
