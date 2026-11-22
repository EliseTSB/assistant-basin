#!/usr/bin/env python3
"""OLMo 2 1B Instruct — 2x2 Bank x Physical-Norm Bridge Runner (Part A).

Implements `explore_olmo2_attn_mlp/INSTRUCT-BRIDGE-DESIGN.md` v3
(SHA-256: a1db3bd6743f3fe60571d4c8575788f7f8b4ddc38ba7b5a5f39d25137a6fbaf8).

Hardening & contracts:
- All 4 external package SHA-256 pins (`design`, `manifest`, `runner`, `scorer`) and
  exact write coordinates (`core_units`) are verified BEFORE any tokenizer or model load.
- Per-source literal binding validator binds each historical source strictly to the fields
  it actually stores (zero-load refusal on drift, plus valid literal-source self-test).
- Exact fixed unit write direction `hat_w` and exact fixed unit-norm orthogonal null
  subspace (`max_abs_null_vec_diff == 0.0`) across `OLD_NORM` and `NEW_NORM` for every
  `(bank, unit, prompt_id, dose_mag)` tuple.
- Head-row-only measured sham (`3,220/3,220` full; `112/112` smoke); sham keys strictly
  absent/not applicable on unpatched rows (`520/520` full; `18/18` smoke, including
  unpatched-only `CAL_SELF`).
- Per-variant / per-draw linear decomposition verification (`Spec == Raw - PeerMedian <= 1e-6`)
  without assuming separately reduced medians subtract.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
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


SCHEMA_VERSION = "instruct_bridge_v1.0"
DESIGN_REL_PATH = "explore_olmo2_attn_mlp/INSTRUCT-BRIDGE-DESIGN.md"
DESIGN_SHA256 = "a1db3bd6743f3fe60571d4c8575788f7f8b4ddc38ba7b5a5f39d25137a6fbaf8"

MODEL_REPO = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_COMMIT = "48d788eca847d4d7548f375ad03d3c9312f6139e"
DTYPE_STR = "float32"
ATTN_IMPL = "eager"

N_NULL_DRAWS = 4
OLD_NULL_SEED_BASE = 202611200
NEW_NULL_SEED_BASE = 202611210

# Exact pinned historical source SHA-256s (v3 §2.1 & §BR.0).
SOURCE_PINS: dict[str, str] = {
    "old_manifest_sha256": "716cf2ab3f08461656c14768a7b5c3e6974261f5a86456c56e00c9a2e0800533",
    "old_runner_sha256": "227d8bc79a6a9c8563a1f8919bcb1c3aadb6338dce2a107ec25335582f9bc2e6",
    "new_manifest_sha256": "1a187ada52cd7ca6ebe122c7f93eafdf6ffe3024c739de6ade833707d05e403b",
    "new_runner_sha256": "405d67721015d66ec06dc2c5d1de9948e7547059ac120c58c9da61264e003612",
    "old_receipt_sha256": "bf8fdf20dfe132c15b9acddea59eb82494dea9fc39f9d218fe38ff7571f3026b",
    "new_receipt_sha256": "aa8300a11a438c5fd54ffca22a4a6d6ccc762c17c0ad5e8845047af5759c1980",
}

OLD_GIT_COMMIT = "d13fc3dd2b76309d28b001ecf33f0fc9c962f156"
NEW_GIT_COMMIT = "01bd41d963936d39f68b032eda845ad73b13d6a2"

# Exact canonical 5 core units (v3 §2.1). Order matches d13fc3dd CORE_UNITS for seed_u parity!
CANONICAL_CORE_UNITS: list[dict[str, Any]] = [
    {"id": "p92-02", "layer": 3, "neuron": 2628, "lead": "the", "content": "town", "group": "L3"},
    {"id": "p92-05", "layer": 3, "neuron": 6986, "lead": "a", "content": "fox", "group": "L3"},
    {"id": "p92-13", "layer": 3, "neuron": 976, "lead": "a", "content": "skeleton", "group": "L3"},
    {"id": "p92-46", "layer": 5, "neuron": 4673, "lead": "a", "content": "regular guy", "group": "L5"},
    {"id": "p92-09", "layer": 6, "neuron": 4145, "lead": "a", "content": "dog", "group": "L6"},
]
CORE_UNIT_IDS: tuple[str, ...] = tuple(u["id"] for u in CANONICAL_CORE_UNITS)
UNIT_BY_ID: dict[str, dict[str, Any]] = {u["id"]: u for u in CANONICAL_CORE_UNITS}

# Exact 7 downstream attention heads (v3 §2.1).
CANONICAL_HEADS: list[dict[str, Any]] = [
    {"name": "L7H5", "layer": 7, "head": 5, "kind": "primary_L7", "in_old_receipt": True},
    {"name": "L7H4", "layer": 7, "head": 4, "kind": "rand_L7", "in_old_receipt": True},
    {"name": "L7H9", "layer": 7, "head": 9, "kind": "rand_L7", "in_old_receipt": False},
    {"name": "L8H15", "layer": 8, "head": 15, "kind": "primary_L8", "in_old_receipt": True},
    {"name": "L8H2", "layer": 8, "head": 2, "kind": "active_comparator_L8", "in_old_receipt": True},
    {"name": "L8H3", "layer": 8, "head": 3, "kind": "rand_L8", "in_old_receipt": True},
    {"name": "L8H13", "layer": 8, "head": 13, "kind": "rand_L8", "in_old_receipt": True},
]
HEAD_NAMES: tuple[str, ...] = tuple(h["name"] for h in CANONICAL_HEADS)
OLD_OVERLAP_HEADS: frozenset[str] = frozenset(h["name"] for h in CANONICAL_HEADS if h["in_old_receipt"])
PEER_DELTA_HEADS: frozenset[str] = frozenset({"L8H15", "L8H2"})

# Exact full-precision pinned physical norms (v3 §2.2).
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
NORM_ORDER: tuple[str, ...] = ("OLD_NORM", "NEW_NORM")

# Exact prompt schedules (v3 §2.4 & §4).
OLD_CORE_PROMPTS: tuple[tuple[str, str, str, bool], ...] = (
    ("self_F1", "SELF_HELD_NEW", "F1", True),
    ("self_F2", "SELF_HELD_NEW", "F2", True),
    ("self_F3", "SELF_HELD_NEW", "F3", False),
    ("role_F1", "ROLE_DEFINITE_NEW", "F1", False),
    ("role_F2", "ROLE_DEFINITE_NEW", "F2", False),
    ("role_F3", "ROLE_DEFINITE_NEW", "F3", False),
    ("npers_F1", "NAMED_PERSON_NEW", "F1", False),
    ("npers_F2", "NAMED_PERSON_NEW", "F2", False),
    ("npers_F3", "NAMED_PERSON_NEW", "F3", False),
    ("third_F1", "THIRD_GENERIC_NEW", "F1", True),
    ("third_F2", "THIRD_GENERIC_NEW", "F2", True),
    ("third_F3", "THIRD_GENERIC_NEW", "F3", False),
)
OLD_HIGH_DOSE_PROMPTS: tuple[tuple[str, str, str, bool], ...] = (
    ("self_F1", "SELF_HELD_NEW", "F1", True),
    ("self_F2", "SELF_HELD_NEW", "F2", True),
    ("third_F1", "THIRD_GENERIC_NEW", "F1", True),
    ("third_F2", "THIRD_GENERIC_NEW", "F2", True),
)

NEW_EVAL_CELLS: tuple[tuple[str, str], ...] = (
    ("self", "SELF"),
    ("role_bare", "ROLE_BARE"),
    ("name_bare", "NAME_BARE"),
    ("alias_role", "ALIAS_TO_ROLE"),
    ("alias_name", "ALIAS_TO_NAME"),
)

SMOKE_UNIT_ID = "p92-46"
SMOKE_SCHEDULE_KEYS: tuple[tuple[str, str, float], ...] = (
    ("OLD_BANK", "self_F1", 0.8),
    ("OLD_BANK", "role_F1", 0.8),
    ("OLD_BANK", "npers_F1", 0.8),
    ("OLD_BANK", "third_F1", 1.2),
    ("OLD_BANK", "third_F1", 1.6),
    ("NEW_BANK", "eval_self_E1", 0.8),
    ("NEW_BANK", "eval_role_bare_E1", 0.8),
    ("NEW_BANK", "cal_third_C1", 1.6),
    ("NEW_BANK", "cal_self_C1", 1.6),
)


# ---------------------------------------------------------------------------
# Hashing, Git/AST Literal Source Helpers, and Pre-Load Pin Enforcement
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dose_key(dose_mag: float) -> str:
    return f"{float(dose_mag):.1f}"


def assert_strictly_downstream(unit_layer: int, head_layer: int) -> None:
    if not (int(head_layer) > int(unit_layer)):
        raise ValueError(f"Non-causal head_layer={head_layer} <= unit_layer={unit_layer}")


def parse_all_slot_units_from_old_runner_source(runner_src: str) -> dict[str, dict[str, Any]]:
    """Safely parse `ALL_SLOT_UNITS` from `d13fc3dd:run_signed_dose.py` via AST (no exec)."""
    tree = ast.parse(runner_src)
    for node in tree.body:
        target_name = None
        value_node = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        if target_name == "ALL_SLOT_UNITS" and value_node is not None:
            raw_list = ast.literal_eval(value_node)
            if not isinstance(raw_list, list):
                raise ValueError("ALL_SLOT_UNITS is not a list")
            out: dict[str, dict[str, Any]] = {}
            for entry in raw_list:
                out[str(entry["id"])] = {
                    "id": str(entry["id"]),
                    "layer": int(entry["layer"]),
                    "neuron": int(entry["neuron"]),
                    "lead": str(entry["lead"]),
                    "content": str(entry["content"]),
                }
            return out
    raise ValueError("ALL_SLOT_UNITS not found in OLD runner source")


def assert_core_units_canonical(units_list: list[dict[str, Any]], context: str = "manifest") -> None:
    """Assert exact 5-tuple `(id, layer, neuron, lead, content)` agreement against canonical coordinates."""
    if len(units_list) != len(CANONICAL_CORE_UNITS):
        raise ValueError(f"{context}: expected {len(CANONICAL_CORE_UNITS)} core units, got {len(units_list)}")
    for got, exp in zip(units_list, CANONICAL_CORE_UNITS):
        for field in ("id", "layer", "neuron", "lead", "content"):
            if got.get(field) != exp[field]:
                raise ValueError(
                    f"{context}: core unit drift on {exp['id']}.{field}: got {got.get(field)!r}, expected {exp[field]!r}"
                )


def assert_norms_canonical(norms: dict[str, dict[str, float]], context: str = "manifest") -> None:
    if set(norms.keys()) != set(CANONICAL_NORMS.keys()):
        raise ValueError(f"{context}: norm keys mismatch: {sorted(norms.keys())}")
    for nc, exp_map in CANONICAL_NORMS.items():
        got_map = norms[nc]
        if set(got_map.keys()) != set(exp_map.keys()):
            raise ValueError(f"{context}: norm layers mismatch for {nc}: {sorted(got_map.keys())}")
        for lk, ev in exp_map.items():
            if float(got_map[lk]) != float(ev):
                raise ValueError(f"{context}: norm drift on {nc}[{lk}]: {got_map[lk]} != {ev}")


def verify_literal_source_bindings(
    manifest: dict[str, Any],
    old_manifest_bytes: bytes,
    old_runner_bytes: bytes,
    new_manifest_bytes: bytes,
    new_runner_bytes: bytes,
    new_receipt_bytes: bytes,
    old_receipt_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Per-source literal field binding verification (v3 §2.1 & Alden #93 item 1).

    Binds each historical source strictly to the fields it actually stores:
    1. Full 5-tuple `(id, layer, neuron, lead, content)` across:
       - `instruct_bridge_manifest.json` (`meta.core_units`),
       - `d13fc3dd:run_signed_dose.py` (`ALL_SLOT_UNITS` / `SLOT_BY_ID`),
       - `01bd41d9:stage_sweep_manifest.json` (`meta.core_units`).
    2. `NEW_RECEIPT["unit_vectors"]` strictly on `(map_key == id, layer, neuron)` and
       `cal_resid_median == NEW_NORM[layer]` (does NOT expect `lead`/`content`).
    3. `d13fc3dd:signed_dose_manifest.json` strictly on supported unit/layer/median/seed/token joins
       (`eligible_pairs` unit & unit_layer, `meta.pinned_medians_chat == OLD_NORM`,
       `meta.null_seed_base == 202611200`, and exact `q12_prompts` tokenized bindings).
    4. `01bd41d9:stage_sweep_manifest.json` exact tokenized bindings for all 24 `NEW_BANK` prompts.
    """
    # 0. Exact SHA-256 verification of every source blob.
    checks = [
        ("old_manifest_sha256", old_manifest_bytes),
        ("old_runner_sha256", old_runner_bytes),
        ("new_manifest_sha256", new_manifest_bytes),
        ("new_runner_sha256", new_runner_bytes),
        ("new_receipt_sha256", new_receipt_bytes),
    ]
    if old_receipt_bytes is not None:
        checks.append(("old_receipt_sha256", old_receipt_bytes))
    for pin_name, raw_bytes in checks:
        got_sha = sha256_bytes(raw_bytes)
        exp_sha = SOURCE_PINS[pin_name]
        if got_sha != exp_sha:
            raise ValueError(f"Literal source SHA-256 mismatch for {pin_name}: got {got_sha}, expected {exp_sha}")
        if manifest["meta"]["source_pins"].get(pin_name) != exp_sha:
            raise ValueError(f"Manifest source_pins[{pin_name}] mismatch")

    # 1. Full 5-tuple across bridge manifest, OLD runner SLOT_BY_ID, and NEW manifest meta.core_units.
    assert_core_units_canonical(manifest["meta"]["core_units"], "bridge_manifest")
    assert_norms_canonical(manifest["meta"]["norms"], "bridge_manifest")

    old_slot_by_id = parse_all_slot_units_from_old_runner_source(old_runner_bytes.decode("utf-8"))
    old_core_from_runner = [old_slot_by_id[uid] for uid in CORE_UNIT_IDS]
    assert_core_units_canonical(old_core_from_runner, "d13fc3dd:run_signed_dose.py(SLOT_BY_ID)")

    new_man = json.loads(new_manifest_bytes.decode("utf-8"))
    assert_core_units_canonical(new_man["meta"]["core_units"], "01bd41d9:stage_sweep_manifest.json(meta.core_units)")

    # 2. NEW_RECEIPT["unit_vectors"] bound strictly on (map_key == id, layer, neuron, cal_resid_median).
    new_rcpt = json.loads(new_receipt_bytes.decode("utf-8"))
    uvecs = new_rcpt.get("unit_vectors")
    if not isinstance(uvecs, dict) or set(uvecs.keys()) != set(CORE_UNIT_IDS):
        raise ValueError(f"NEW_RECEIPT unit_vectors keys mismatch: {sorted(uvecs.keys()) if isinstance(uvecs, dict) else type(uvecs)}")
    for u in CANONICAL_CORE_UNITS:
        uid = u["id"]
        uv = uvecs[uid]
        if int(uv["layer"]) != u["layer"] or int(uv["neuron"]) != u["neuron"]:
            raise ValueError(
                f"NEW_RECEIPT unit_vectors[{uid}] drift: got (L{uv['layer']}/N{uv['neuron']}), "
                f"expected (L{u['layer']}/N{u['neuron']})"
            )
        exp_med = CANONICAL_NORMS["NEW_NORM"][str(u["layer"])]
        if float(uv["cal_resid_median"]) != float(exp_med):
            raise ValueError(
                f"NEW_RECEIPT unit_vectors[{uid}].cal_resid_median drift: {uv['cal_resid_median']} != {exp_med}"
            )

    # 3. OLD manifest bound strictly on its supported fields (unit/layer pairs, medians, null_seed_base, q12_prompts).
    old_man = json.loads(old_manifest_bytes.decode("utf-8"))
    if int(old_man["meta"]["null_seed_base"]) != OLD_NULL_SEED_BASE:
        raise ValueError(f"OLD manifest null_seed_base mismatch: {old_man['meta']['null_seed_base']}")
    old_meds = old_man["meta"]["pinned_medians_chat"]
    for lk, ev in CANONICAL_NORMS["OLD_NORM"].items():
        if float(old_meds[lk]) != float(ev):
            raise ValueError(f"OLD manifest pinned_medians_chat[{lk}] mismatch: {old_meds[lk]} != {ev}")

    ep_layers: dict[str, set[int]] = {}
    for ep in old_man["eligible_pairs"]:
        ep_layers.setdefault(str(ep["unit"]), set()).add(int(ep["unit_layer"]))
    if set(ep_layers.keys()) != set(CORE_UNIT_IDS):
        raise ValueError(f"OLD manifest eligible_pairs units mismatch: {sorted(ep_layers.keys())}")
    for u in CANONICAL_CORE_UNITS:
        if ep_layers[u["id"]] != {u["layer"]}:
            raise ValueError(f"OLD manifest eligible_pairs layer mismatch for {u['id']}: {ep_layers[u['id']]}")

    # Verify OLD_BANK prompt tokenized bindings in bridge manifest match OLD manifest verbatim.
    old_q12_by_id = {p["id"]: p for p in old_man["q12_prompts"]}
    for pid, cell, frame, block_a in OLD_CORE_PROMPTS:
        src_p = old_q12_by_id[pid]
        if src_p["cell"] != cell or src_p["frame"] != frame or bool(src_p["block_a"]) != bool(block_a):
            raise ValueError(f"OLD manifest prompt metadata mismatch for {pid}")
        bkey = f"OLD_BANK|{pid}"
        bmap = manifest["bindings"][bkey]
        for uid in CORE_UNIT_IDS:
            spu = src_p["per_unit"][uid]
            got = bmap[uid]
            if (
                got["input_ids"] != spu["prefix_ids"]
                or int(got["read_pos"]) != int(spu["read_pos"])
                or int(got["t_own"]) != int(spu["own_token_id"])
                or got["peers"] != spu["peer_token_ids"]
                or got["prefix_text"] != spu["prefix_text"]
            ):
                raise ValueError(f"OLD_BANK binding drift on ({pid}, {uid})")

    # 4. Verify NEW_BANK prompt tokenized bindings in bridge manifest match NEW manifest verbatim.
    new_prompts_by_id = {p["prompt_id"]: p for p in new_man["prompts"]}
    new_pids_expected = [
        f"eval_{slug}_E{i}" for slug, _ in NEW_EVAL_CELLS for i in (1, 2, 3, 4)
    ] + [f"cal_third_C{i}" for i in (1, 2)] + [f"cal_self_C{i}" for i in (1, 2)]
    for pid in new_pids_expected:
        if pid not in new_prompts_by_id or pid not in new_man["bindings"]:
            raise ValueError(f"NEW manifest missing prompt/binding {pid}")
        bkey = f"NEW_BANK|{pid}"
        bmap = manifest["bindings"][bkey]
        for uid in CORE_UNIT_IDS:
            sbu = new_man["bindings"][pid][uid]
            got = bmap[uid]
            if (
                got["input_ids"] != sbu["input_ids"]
                or int(got["read_pos"]) != int(sbu["read_pos"])
                or int(got["t_own"]) != int(sbu["t_own"])
                or got["peers"] != sbu["peers"]
            ):
                raise ValueError(f"NEW_BANK binding drift on ({pid}, {uid})")

    return {
        "literal_source_bindings_verified": True,
        "checked_sources": [k for k, _ in checks],
        "n_old_prompts_bound": len(OLD_CORE_PROMPTS),
        "n_new_prompts_bound": len(new_pids_expected),
    }


def load_git_blob(repo_root: Path, commit: str, rel_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo_root), "show", f"{commit}:{rel_path}"])


def build_bridge_manifest(
    old_manifest_bytes: bytes,
    old_runner_bytes: bytes,
    new_manifest_bytes: bytes,
    new_runner_bytes: bytes,
    new_receipt_bytes: bytes,
    old_receipt_bytes: bytes,
    design_path: Path,
) -> dict[str, Any]:
    """Deterministically build `instruct_bridge_manifest.json` from verified historical sources."""
    got_design_sha = sha256_file(design_path)
    if got_design_sha != DESIGN_SHA256:
        raise ValueError(f"Design SHA-256 mismatch: got {got_design_sha}, expected {DESIGN_SHA256}")

    old_man = json.loads(old_manifest_bytes.decode("utf-8"))
    new_man = json.loads(new_manifest_bytes.decode("utf-8"))
    old_q12_by_id = {p["id"]: p for p in old_man["q12_prompts"]}
    new_p_by_id = {p["prompt_id"]: p for p in new_man["prompts"]}

    schedule_full: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}

    # OLD_BANK: 12 core prompts at d=0.8 + 8 high-dose block_a prompt-doses at d=1.2, 1.6 = 20 pairs.
    for pid, cell, frame, block_a in OLD_CORE_PROMPTS:
        schedule_full.append({
            "bank": "OLD_BANK",
            "prompt_id": pid,
            "cell": cell,
            "story_or_frame": frame,
            "block_a": block_a,
            "dose_mag": 0.8,
            "head_patched": True,
            "in_old_anchor_heads": True,
            "ref_table_head": "q12_rows",
            "ref_table_unpatched": None,
            "record_peer_deltas": True,
        })
        sp = old_q12_by_id[pid]
        bindings[f"OLD_BANK|{pid}"] = {
            uid: {
                "input_ids": list(sp["per_unit"][uid]["prefix_ids"]),
                "read_pos": int(sp["per_unit"][uid]["read_pos"]),
                "t_own": int(sp["per_unit"][uid]["own_token_id"]),
                "peers": list(sp["per_unit"][uid]["peer_token_ids"]),
                "prefix_text": str(sp["per_unit"][uid]["prefix_text"]),
            }
            for uid in CORE_UNIT_IDS
        }

    for d_hi in (1.2, 1.6):
        for pid, cell, frame, block_a in OLD_HIGH_DOSE_PROMPTS:
            is_16 = abs(d_hi - 1.6) < 1e-9
            schedule_full.append({
                "bank": "OLD_BANK",
                "prompt_id": pid,
                "cell": cell,
                "story_or_frame": frame,
                "block_a": block_a,
                "dose_mag": float(d_hi),
                "head_patched": True,
                "in_old_anchor_heads": is_16,
                "ref_table_head": "q12_rows" if is_16 else None,
                "ref_table_unpatched": None,
                "record_peer_deltas": False,
            })

    # NEW_BANK: 20 EVAL prompts at d=0.8 + 6 CAL_THIRD (head-patched) + 6 CAL_SELF (unpatched-only) = 32 pairs.
    for slug, cell in NEW_EVAL_CELLS:
        for i in (1, 2, 3, 4):
            pid = f"eval_{slug}_E{i}"
            schedule_full.append({
                "bank": "NEW_BANK",
                "prompt_id": pid,
                "cell": cell,
                "story_or_frame": f"E{i}",
                "block_a": None,
                "dose_mag": 0.8,
                "head_patched": True,
                "in_old_anchor_heads": False,
                "ref_table_head": "eval_arm_a_head_rows",
                "ref_table_unpatched": "eval_arm_a_unpatched_rows",
                "record_peer_deltas": True,
            })
            bindings[f"NEW_BANK|{pid}"] = {
                uid: {
                    "input_ids": list(new_man["bindings"][pid][uid]["input_ids"]),
                    "read_pos": int(new_man["bindings"][pid][uid]["read_pos"]),
                    "t_own": int(new_man["bindings"][pid][uid]["t_own"]),
                    "peers": list(new_man["bindings"][pid][uid]["peers"]),
                    "prefix_text": str(new_p_by_id[pid].get("base_text", "")),
                }
                for uid in CORE_UNIT_IDS
            }

    for d_cal in (0.8, 1.2, 1.6):
        for i in (1, 2):
            pid = f"cal_third_C{i}"
            schedule_full.append({
                "bank": "NEW_BANK",
                "prompt_id": pid,
                "cell": "CAL_THIRD",
                "story_or_frame": f"C{i}",
                "block_a": None,
                "dose_mag": float(d_cal),
                "head_patched": True,
                "in_old_anchor_heads": False,
                "ref_table_head": "cal_head_rows",
                "ref_table_unpatched": "cal_unpatched_rows",
                "record_peer_deltas": False,
            })
            if f"NEW_BANK|{pid}" not in bindings:
                bindings[f"NEW_BANK|{pid}"] = {
                    uid: {
                        "input_ids": list(new_man["bindings"][pid][uid]["input_ids"]),
                        "read_pos": int(new_man["bindings"][pid][uid]["read_pos"]),
                        "t_own": int(new_man["bindings"][pid][uid]["t_own"]),
                        "peers": list(new_man["bindings"][pid][uid]["peers"]),
                        "prefix_text": str(new_p_by_id[pid].get("base_text", "")),
                    }
                    for uid in CORE_UNIT_IDS
                }

    for d_cal in (0.8, 1.2, 1.6):
        for i in (1, 2):
            pid = f"cal_self_C{i}"
            schedule_full.append({
                "bank": "NEW_BANK",
                "prompt_id": pid,
                "cell": "CAL_SELF",
                "story_or_frame": f"C{i}",
                "block_a": None,
                "dose_mag": float(d_cal),
                "head_patched": False,  # Never head-patched in 01bd41d9; unpatched-only!
                "in_old_anchor_heads": False,
                "ref_table_head": None,
                "ref_table_unpatched": "cal_unpatched_rows",
                "record_peer_deltas": False,
            })
            if f"NEW_BANK|{pid}" not in bindings:
                bindings[f"NEW_BANK|{pid}"] = {
                    uid: {
                        "input_ids": list(new_man["bindings"][pid][uid]["input_ids"]),
                        "read_pos": int(new_man["bindings"][pid][uid]["read_pos"]),
                        "t_own": int(new_man["bindings"][pid][uid]["t_own"]),
                        "peers": list(new_man["bindings"][pid][uid]["peers"]),
                        "prefix_text": str(new_p_by_id[pid].get("base_text", "")),
                    }
                    for uid in CORE_UNIT_IDS
                }

    # Build smoke schedule subset and assert exact denominators.
    sched_lookup = {(e["bank"], e["prompt_id"], dose_key(e["dose_mag"])): e for e in schedule_full}
    schedule_smoke = [sched_lookup[(b, p, dose_key(d))] for b, p, d in SMOKE_SCHEDULE_KEYS]

    def calc_denoms(sched: list[dict[str, Any]], n_units: int) -> dict[str, int]:
        n_unp = len(sched) * len(NORM_ORDER) * n_units
        n_hd = sum(1 for e in sched if e["head_patched"]) * len(NORM_ORDER) * n_units * len(CANONICAL_HEADS)
        n_b19 = n_unp + 2 * n_hd
        old_anc_h = sum(1 for e in sched if e["in_old_anchor_heads"]) * n_units * len(OLD_OVERLAP_HEADS)
        new_anc_h = sum(1 for e in sched if e["bank"] == "NEW_BANK" and e["head_patched"]) * n_units * len(CANONICAL_HEADS)
        new_anc_u = sum(1 for e in sched if e["bank"] == "NEW_BANK") * n_units
        return {
            "n_prompt_dose_pairs": len(sched),
            "n_unpatched_rows": n_unp,
            "n_head_rows": n_hd,
            "n_b19_forwards": n_b19,
            "old_anchor_head_rows": old_anc_h,
            "old_anchor_unpatched_rows": 0,
            "new_anchor_head_rows": new_anc_h,
            "new_anchor_unpatched_rows": new_anc_u,
        }

    denoms_full = calc_denoms(schedule_full, len(CORE_UNIT_IDS))
    denoms_smoke = calc_denoms(schedule_smoke, 1)

    assert denoms_full == {
        "n_prompt_dose_pairs": 52,
        "n_unpatched_rows": 520,
        "n_head_rows": 3220,
        "n_b19_forwards": 6960,
        "old_anchor_head_rows": 480,
        "old_anchor_unpatched_rows": 0,
        "new_anchor_head_rows": 910,
        "new_anchor_unpatched_rows": 160,
    }, f"Unexpected full denominators: {denoms_full}"
    assert denoms_smoke == {
        "n_prompt_dose_pairs": 9,
        "n_unpatched_rows": 18,
        "n_head_rows": 112,
        "n_b19_forwards": 242,
        "old_anchor_head_rows": 24,
        "old_anchor_unpatched_rows": 0,
        "new_anchor_head_rows": 21,
        "new_anchor_unpatched_rows": 4,
    }, f"Unexpected smoke denominators: {denoms_smoke}"

    manifest = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "design_file": DESIGN_REL_PATH,
            "design_sha256": DESIGN_SHA256,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "n_null_draws": N_NULL_DRAWS,
            "old_null_seed_base": OLD_NULL_SEED_BASE,
            "new_null_seed_base": NEW_NULL_SEED_BASE,
            "norms": CANONICAL_NORMS,
            "core_units": CANONICAL_CORE_UNITS,
            "heads": CANONICAL_HEADS,
            "smoke_unit": SMOKE_UNIT_ID,
            "source_pins": dict(SOURCE_PINS),
            "denominators": {"full": denoms_full, "smoke": denoms_smoke},
        },
        "schedule_full": schedule_full,
        "schedule_smoke": schedule_smoke,
        "bindings": bindings,
    }

    # Self-verify against literal source bindings before returning.
    verify_literal_source_bindings(
        manifest=manifest,
        old_manifest_bytes=old_manifest_bytes,
        old_runner_bytes=old_runner_bytes,
        new_manifest_bytes=new_manifest_bytes,
        new_runner_bytes=new_runner_bytes,
        new_receipt_bytes=new_receipt_bytes,
        old_receipt_bytes=old_receipt_bytes,
    )
    return manifest


def _admit_frozen_package_with_bytes(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, bytes]]:
    """Read each of the 4 pinned files ONCE, verify SHA-256 on those exact bytes, and parse manifest from them."""
    bytes_by_label: dict[str, bytes] = {}
    for label, p, exp in [
        ("design", design_path, expected_design_sha),
        ("manifest", manifest_path, expected_manifest_sha),
        ("runner", runner_path, expected_runner_sha),
        ("scorer", scorer_path, expected_scorer_sha),
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

    # Parse manifest from the EXACT buffer whose digest just matched `expected_manifest_sha` (single read).
    manifest = json.loads(bytes_by_label["manifest"].decode("utf-8"))
    if manifest["meta"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema_version mismatch: {manifest['meta'].get('schema_version')}")
    if manifest["meta"].get("design_sha256") != expected_design_sha:
        raise ValueError("Manifest embedded design_sha256 does not match external design pin")
    if manifest["meta"].get("model_repo") != MODEL_REPO or manifest["meta"].get("model_commit") != MODEL_COMMIT:
        raise ValueError("Manifest model repo/commit mismatch")
    assert_core_units_canonical(manifest["meta"]["core_units"], "admit_manifest")
    assert_norms_canonical(manifest["meta"]["norms"], "admit_manifest")
    for k, v in SOURCE_PINS.items():
        if manifest["meta"]["source_pins"].get(k) != v:
            raise ValueError(f"Manifest source_pins[{k}] drift")

    pins = {
        "design_sha256": expected_design_sha,
        "manifest_sha256": expected_manifest_sha,
        "runner_sha256": expected_runner_sha,
        "scorer_sha256": expected_scorer_sha,
    }
    return manifest, pins, bytes_by_label


def admit_frozen_package(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify all 4 external SHA-256 pins + manifest coordinates/norms BEFORE any model/tokenizer load."""
    manifest, pins, _ = _admit_frozen_package_with_bytes(
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        expected_design_sha=expected_design_sha,
        expected_manifest_sha=expected_manifest_sha,
        expected_runner_sha=expected_runner_sha,
        expected_scorer_sha=expected_scorer_sha,
    )
    return manifest, pins


def validate_timing_breakdown(timing: Any, expected_b19_fwds: int) -> None:
    """Verify a finite, consistent measured timing breakdown (Alden #104 item 2)."""
    if not isinstance(timing, dict):
        raise ValueError("Missing or null timing breakdown in receipt meta")
    req_keys = (
        "model_load_sec",
        "b19_elapsed_sec",
        "non_b19_runner_sec",
        "total_runner_sec",
        "n_b19_done",
        "ms_per_b19",
        "projected_full_b19_sec",
    )
    for k in req_keys:
        v = timing.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise ValueError(f"Timing field '{k}' missing or non-finite: {v!r}")
    if int(timing["n_b19_done"]) != int(expected_b19_fwds):
        raise ValueError(f"Timing n_b19_done={timing['n_b19_done']} != expected {expected_b19_fwds}")
    if float(timing["model_load_sec"]) < 0.0:
        raise ValueError(f"Negative model_load_sec: {timing['model_load_sec']}")
    if float(timing["b19_elapsed_sec"]) <= 0.0:
        raise ValueError(f"Non-positive b19_elapsed_sec: {timing['b19_elapsed_sec']}")
    if float(timing["non_b19_runner_sec"]) < 0.0:
        raise ValueError(f"Negative non_b19_runner_sec: {timing['non_b19_runner_sec']}")
    if float(timing["total_runner_sec"]) + 1e-9 < float(timing["b19_elapsed_sec"]):
        raise ValueError("total_runner_sec < b19_elapsed_sec")
    exp_ms = 1000.0 * float(timing["b19_elapsed_sec"]) / float(expected_b19_fwds)
    if float(timing["ms_per_b19"]) <= 0.0 or abs(float(timing["ms_per_b19"]) - exp_ms) > 1e-3:
        raise ValueError(f"Timing ms_per_b19 inconsistent: got {timing['ms_per_b19']}, expected {exp_ms}")
    if float(timing["projected_full_b19_sec"]) <= 0.0:
        raise ValueError(f"Non-positive projected_full_b19_sec: {timing['projected_full_b19_sec']}")


def _load_pinned_scorer_module(
    scorer_path: Path,
    expected_scorer_sha: str,
    scorer_bytes: bytes | None = None,
):
    """Verify SHA-256 on scorer source bytes and execute those EXACT bytes in memory (no second read)."""
    raw_b = scorer_path.read_bytes() if scorer_bytes is None else scorer_bytes
    got = sha256_bytes(raw_b)
    if got != expected_scorer_sha:
        raise ValueError(f"Scorer SHA-256 mismatch before load: got {got}, expected {expected_scorer_sha}")
    # Ensure `import run_instruct_bridge as rib` inside scorer resolves to THIS exact module object.
    prev_mod = sys.modules.get("run_instruct_bridge")
    sys.modules["run_instruct_bridge"] = sys.modules[__name__]
    try:
        mod = types.ModuleType("_pinned_score_instruct_bridge")
        mod.__file__ = str(scorer_path.resolve())
        code = compile(raw_b, str(scorer_path.resolve()), "exec")
        exec(code, mod.__dict__)
        return mod
    finally:
        if prev_mod is not None:
            sys.modules["run_instruct_bridge"] = prev_mod


def admit_execution_request(
    mode: str | None,
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
    allow_full_after_smoke: bool = False,
    verified_smoke_receipt: Path | None = None,
    verified_smoke_sha256: str = "",
    old_receipt_path: Path | None = None,
    new_receipt_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Enforce explicit --mode {smoke,full}, single-read 4-pin binding, and pinned technical smoke verifier."""
    if mode not in ("smoke", "full"):
        raise ValueError(
            f"Explicit --mode {{smoke,full}} is required (got {mode!r}); implicit full execution is refused."
        )
    if runner_path.resolve() != Path(__file__).resolve():
        raise ValueError(
            f"Runner path {runner_path.resolve()} does not match executing module {Path(__file__).resolve()}"
        )
    manifest, pins, pkg_bytes = _admit_frozen_package_with_bytes(
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        expected_design_sha=expected_design_sha,
        expected_manifest_sha=expected_manifest_sha,
        expected_runner_sha=expected_runner_sha,
        expected_scorer_sha=expected_scorer_sha,
    )
    if mode == "full":
        if not allow_full_after_smoke:
            raise ValueError(
                "Full execution refused: --allow-full-after-smoke and verified smoke receipt are required."
            )
        if verified_smoke_receipt is None or not verified_smoke_receipt.is_file():
            raise ValueError(f"Full execution refused: missing --verified-smoke-receipt ({verified_smoke_receipt})")
        if not verified_smoke_sha256 or len(verified_smoke_sha256) != 64:
            raise ValueError("Full execution refused: valid 64-hex --verified-smoke-sha256 is required")
        if old_receipt_path is None or not old_receipt_path.is_file():
            raise ValueError(f"Full execution refused: missing --old-receipt ({old_receipt_path})")
        if new_receipt_path is None or not new_receipt_path.is_file():
            raise ValueError(f"Full execution refused: missing --new-receipt ({new_receipt_path})")

        # Read smoke receipt ONCE, verify its digest, and parse that EXACT buffer (Alden #111).
        smoke_bytes = verified_smoke_receipt.read_bytes()
        got_sm_sha = sha256_bytes(smoke_bytes)
        if got_sm_sha != verified_smoke_sha256:
            raise ValueError(f"Verified smoke SHA-256 mismatch: got {got_sm_sha}, expected {verified_smoke_sha256}")

        # Execute the EXACT scorer source bytes verified in `_admit_frozen_package_with_bytes` (single read).
        scorer_mod = _load_pinned_scorer_module(scorer_path, expected_scorer_sha, scorer_bytes=pkg_bytes["scorer"])
        sm = json.loads(smoke_bytes.decode("utf-8"))
        sm_verdict = scorer_mod.verify_smoke_receipt_only(
            receipt=sm,
            manifest=manifest,
            pins=pins,
            old_receipt_path=old_receipt_path,
            new_receipt_path=new_receipt_path,
            allow_fixture=False,
        )
        if not sm_verdict.get("smoke_technical_check_passed", False):
            raise ValueError(
                "Full execution refused: pinned technical smoke verifier returned failed verdict "
                f"(status={sm_verdict.get('anchor_subset_check', {}).get('status')!r}, "
                f"old_max={sm_verdict.get('anchor_subset_check', {}).get('join_1_old_head', {}).get('max_abs_diff')}, "
                f"new_hd_max={sm_verdict.get('anchor_subset_check', {}).get('join_2a_new_head', {}).get('max_abs_diff')}, "
                f"new_unp_max={sm_verdict.get('anchor_subset_check', {}).get('join_2b_new_unpatched', {}).get('max_abs_diff')})"
            )
    return manifest, pins


# ---------------------------------------------------------------------------
# Anchor-Exact Orthogonal Null Generators & Linear Decomposition Checks
# ---------------------------------------------------------------------------

def new_unit_seed(unit_id: str, prompt_id: str, dose_mag: float) -> int:
    """Exact `01bd41d9` `unit_seed('instruct', uid, pid, dose_mag)`."""
    key = f"{NEW_NULL_SEED_BASE}|instruct|{unit_id}|{prompt_id}|{dose_mag:.4f}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def make_old_null_unit_vectors(
    w_down_cpu: torch.Tensor,
    target_neuron: int,
    hat_w_cpu: torch.Tensor,
    n_draws: int,
    seed_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact `d13fc3dd` per-unit unit-norm (`||v||_2 == 1`) orthogonal null directions."""
    v_hat = hat_w_cpu.float() / float(hat_w_cpu.float().norm().item())
    d_model, d_mlp = w_down_cpu.shape
    wdown_orth: list[torch.Tensor] = []
    iso_resid: list[torch.Tensor] = []
    for m in range(n_draws):
        g1 = torch.Generator(device="cpu").manual_seed(seed_base + 1000 * m + 1)
        raw = torch.randn((d_mlp,), generator=g1, dtype=torch.float32)
        raw[target_neuron] = 0.0
        rv = w_down_cpu.float() @ raw
        rv = rv - torch.dot(rv, v_hat) * v_hat
        rv = rv / float(rv.norm().item())
        if abs(float(torch.dot(rv, v_hat).item())) >= 1e-5:
            raise RuntimeError(f"OLD wdown_orth draw {m} orthogonality failed")
        wdown_orth.append(rv)

        g2 = torch.Generator(device="cpu").manual_seed(seed_base + 1000 * m + 501)
        irv = torch.randn((d_model,), generator=g2, dtype=torch.float32)
        irv = irv - torch.dot(irv, v_hat) * v_hat
        irv = irv / float(irv.norm().item())
        if abs(float(torch.dot(irv, v_hat).item())) >= 1e-5:
            raise RuntimeError(f"OLD iso_resid draw {m} orthogonality failed")
        iso_resid.append(irv)
    return torch.stack(wdown_orth, dim=0), torch.stack(iso_resid, dim=0)


def make_new_null_unit_vectors(
    w_down_cpu: torch.Tensor,
    target_neuron: int,
    hat_w_cpu: torch.Tensor,
    n_draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact `01bd41d9` per-tuple unit-norm (`||v||_2 == 1`) orthogonal null directions."""
    v_hat = hat_w_cpu.float() / float(hat_w_cpu.float().norm().item())
    d_model, d_ff = w_down_cpu.shape
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    coeffs = torch.randn((d_ff, n_draws), generator=gen, dtype=torch.float32)
    coeffs[target_neuron, :] = 0.0
    raw_wd = (w_down_cpu.float() @ coeffs).T  # [n_draws, d_model]
    raw_wd = raw_wd - (raw_wd @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    wd_norms = torch.linalg.vector_norm(raw_wd, dim=-1, keepdim=True).clamp_min(1e-12)
    wdown_unit = raw_wd / wd_norms

    raw_iso = torch.randn((n_draws, d_model), generator=gen, dtype=torch.float32)
    raw_iso = raw_iso - (raw_iso @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    iso_norms = torch.linalg.vector_norm(raw_iso, dim=-1, keepdim=True).clamp_min(1e-12)
    iso_unit = raw_iso / iso_norms

    for mat_name, mat in (("wdown_orth", wdown_unit), ("iso_resid", iso_unit)):
        if float((mat @ v_hat).abs().max().item()) >= 1e-5:
            raise RuntimeError(f"NEW {mat_name} orthogonality failed")
    return wdown_unit, iso_unit


def fingerprint_unit_basis(hat_w_cpu: torch.Tensor, wd_unit_cpu: torch.Tensor, iso_unit_cpu: torch.Tensor) -> str:
    """Exact SHA-256 fingerprint of fixed unit-norm `(hat_w, wdown_orth, iso_resid)` basis."""
    buf = (
        hat_w_cpu.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
        + wd_unit_cpu.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
        + iso_unit_cpu.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    )
    return sha256_bytes(buf)


def build_b19_deltas(
    hat_w_cpu: torch.Tensor,
    wd_unit_cpu: torch.Tensor,
    iso_unit_cpu: torch.Tensor,
    dose_mag: float,
    norm_multiplier: float,
    device: str,
) -> torch.Tensor:
    """Construct exact B=19 injection matrix `[19, d_model]` with verified sign-companions."""
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

    # Exact sign-companion assertions.
    if float(deltas[0].abs().max().item()) != 0.0:
        raise ValueError("Row 0 of B=19 must be exact zero")
    if float((deltas[1] + deltas[2]).abs().max().item()) > 1e-5:
        raise ValueError("Target +d/-d rows are not exact negatives")
    for k in range(N_NULL_DRAWS):
        if float((deltas[3 + k] + deltas[7 + k]).abs().max().item()) > 1e-5:
            raise ValueError(f"wdown_orth draw {k} +d/-d mismatch")
        if float((deltas[11 + k] + deltas[15 + k]).abs().max().item()) > 1e-5:
            raise ValueError(f"iso_resid draw {k} +d/-d mismatch")
    return deltas.to(device=device, dtype=torch.float32)


def read_decomposed_logits(
    logits_at_read: torch.Tensor,
    t_own: int,
    peers: list[int],
) -> tuple[list[float], list[float], list[float], list[list[float]]]:
    """Compute exact per-variant `(raw, peer_med, spec, peer_mat)` where `spec == raw - peer_med`."""
    if not torch.isfinite(logits_at_read).all():
        raise FloatingPointError("Non-finite logits encountered")
    raw = [float(x) for x in logits_at_read[:, t_own].detach().cpu().tolist()]
    peer_mat = [[float(v) for v in row] for row in logits_at_read[:, peers].detach().cpu().tolist()]
    peer_med = [float(statistics.median(row)) for row in peer_mat]
    spec = [r - pm for r, pm in zip(raw, peer_med)]
    for vals in (raw, peer_med, spec):
        if not all(math.isfinite(x) for x in vals):
            raise FloatingPointError("Non-finite decomposed logit")
    return raw, peer_med, spec, peer_mat


def summarize_unpatched_transport(
    unp_raw: list[float],
    unp_pmed: list[float],
    unp_spec: list[float],
) -> tuple[dict[str, Any], float]:
    """Summarize unpatched transport & verify per-variant linear decomposition `spec == raw - peer_med`."""
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

    # Per-variant / per-draw linear decomposition check (never assumes medians subtract).
    errs = [
        abs(sp - (rp - pp)),
        abs(sn - (rn - pn)),
        *[abs(swd[k] - (rwd[k] - pwd[k])) for k in range(N_NULL_DRAWS)],
        *[abs(siso[k] - (riso[k] - piso[k])) for k in range(N_NULL_DRAWS)],
    ]
    max_err = max(errs)
    if max_err > 1e-6:
        raise AssertionError(f"Unpatched linear decomposition violated: max_err={max_err}")

    transport = {
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
    }
    return transport, max_err


def summarize_b19_metric(
    unp: list[float],
    pat: list[float],
    shm: list[float],
    n_draws: int = N_NULL_DRAWS,
) -> dict[str, Any]:
    """Compute same-dose drop `D = unp - pat`, even/odd, measured sham max, and signed null draws."""
    if not (len(unp) == len(pat) == len(shm) == 3 + 4 * n_draws):
        raise ValueError(f"Expected B={3 + 4 * n_draws}, got {len(unp)}, {len(pat)}, {len(shm)}")
    d_row = [u - p for u, p in zip(unp, pat)]
    d_shm = [u - s for u, s in zip(unp, shm)]
    if not (all(math.isfinite(x) for x in d_row) and all(math.isfinite(x) for x in d_shm)):
        raise FloatingPointError("Non-finite value in summarize_b19_metric")
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


def check_head_linear_decomposition(
    spec_sum: dict[str, Any],
    raw_sum: dict[str, Any],
    pmed_sum: dict[str, Any],
) -> float:
    """Verify exact per-variant / per-draw linear identity `spec == raw - peer_med` (Alden #97).

    Deliberately checks ONLY linear quantities (`d0`, `d_pos`, `d_neg`, `even`, `odd`, and
    per-draw `d_pos_draws[k]`, `d_neg_draws[k]`, `even_draws[k]`, `odd_draws[k]`), NOT
    nonlinear separately-reduced medians (`*_med`).
    """
    errs = [
        abs(spec_sum[k] - (raw_sum[k] - pmed_sum[k]))
        for k in ("d0", "d_pos", "d_neg", "even", "odd")
    ]
    for grp in ("wdown_orth", "iso_resid"):
        for dk in ("d_pos_draws", "d_neg_draws", "even_draws", "odd_draws"):
            for s_v, r_v, p_v in zip(spec_sum[grp][dk], raw_sum[grp][dk], pmed_sum[grp][dk]):
                errs.append(abs(s_v - (r_v - p_v)))
    max_err = max(errs)
    if max_err > 1e-6:
        raise AssertionError(f"Head linear decomposition violated: max_err={max_err}")
    return max_err


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not locate decoder layers on model")


# ---------------------------------------------------------------------------
# Execution Engine (Full or Mandatory 1-Unit Measured Smoke)
# ---------------------------------------------------------------------------

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


def execute_bridge(
    manifest: dict[str, Any],
    pins: dict[str, str],
    smoke_1unit: bool,
    device: str,
    model_loader: Callable[[str, str, str], nn.Module] = _default_model_loader,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Execute full (`6,960` B19) or smoke (`242` B19) Bridge with fixed basis across norms."""
    t_start = time.perf_counter()

    t_load0 = time.perf_counter()
    model = model_loader(MODEL_REPO, MODEL_COMMIT, device)
    model_load_sec = time.perf_counter() - t_load0

    layers_mod = get_decoder_layers(model)
    n_heads = int(getattr(model.config, "num_attention_heads"))

    units_to_run = [SMOKE_UNIT_ID] if smoke_1unit else list(CORE_UNIT_IDS)
    schedule = manifest["schedule_smoke"] if smoke_1unit else manifest["schedule_full"]
    norms_map = manifest["meta"]["norms"]

    # Extract fixed unit write vectors `hat_w` and OLD_BANK per-unit null subspaces once.
    unit_basis_cache: dict[str, dict[str, Any]] = {}
    unit_vectors_meta: dict[str, dict[str, Any]] = {}
    for uid in units_to_run:
        u = UNIT_BY_ID[uid]
        li, ni = int(u["layer"]), int(u["neuron"])
        w_down_cpu = layers_mod[li].mlp.down_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        v_raw = w_down_cpu[:, ni]
        w_norm = float(v_raw.norm().item())
        hat_w_cpu = v_raw / max(w_norm, 1e-12)
        old_seed = OLD_NULL_SEED_BASE + 10000 * CORE_UNIT_IDS.index(uid)
        old_wd_u, old_iso_u = make_old_null_unit_vectors(
            w_down_cpu=w_down_cpu,
            target_neuron=ni,
            hat_w_cpu=hat_w_cpu,
            n_draws=N_NULL_DRAWS,
            seed_base=old_seed,
        )
        unit_basis_cache[uid] = {
            "layer": li,
            "neuron": ni,
            "w_down_cpu": w_down_cpu,
            "hat_w_cpu": hat_w_cpu,
            "old_wd_unit": old_wd_u,
            "old_iso_unit": old_iso_u,
        }
        unit_vectors_meta[uid] = {
            "layer": li,
            "neuron": ni,
            "w_down_norm": w_norm,
            "old_norm_multiplier": float(norms_map["OLD_NORM"][str(li)]),
            "new_norm_multiplier": float(norms_map["NEW_NORM"][str(li)]),
        }

    unpatched_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    timer_acc = {"b19_sec": 0.0, "b19_fwds": 0}
    max_abs_null_vec_diff = 0.0
    max_abs_sham_spec = 0.0
    max_abs_sham_raw = 0.0
    max_abs_decomp_err = 0.0

    head_layers_all = sorted({h["layer"] for h in CANONICAL_HEADS})
    B = 3 + 4 * N_NULL_DRAWS

    for uid in units_to_run:
        ub = unit_basis_cache[uid]
        u_layer = ub["layer"]
        n_idx = ub["neuron"]
        hat_w_cpu = ub["hat_w_cpu"]
        w_down_cpu = ub["w_down_cpu"]

        for entry in schedule:
            bank = entry["bank"]
            pid = entry["prompt_id"]
            cell = entry["cell"]
            dose_mag = float(entry["dose_mag"])
            head_patched = bool(entry["head_patched"])
            rec_peers = bool(entry["record_peer_deltas"])
            bkey = f"{bank}|{pid}"
            binding = manifest["bindings"][bkey][uid]
            input_ids = binding["input_ids"]
            read_pos = int(binding["read_pos"])
            t_own = int(binding["t_own"])
            peers = list(binding["peers"])

            # Construct fixed unit-norm orthogonal null subspace ONCE for both OLD_NORM and NEW_NORM!
            if bank == "OLD_BANK":
                wd_unit_cpu = ub["old_wd_unit"]
                iso_unit_cpu = ub["old_iso_unit"]
            elif bank == "NEW_BANK":
                seed_new = new_unit_seed(uid, pid, dose_mag)
                wd_unit_cpu, iso_unit_cpu = make_new_null_unit_vectors(
                    w_down_cpu=w_down_cpu,
                    target_neuron=n_idx,
                    hat_w_cpu=hat_w_cpu,
                    n_draws=N_NULL_DRAWS,
                    seed=seed_new,
                )
            else:
                raise ValueError(f"Unknown bank: {bank}")

            basis_sha = fingerprint_unit_basis(hat_w_cpu, wd_unit_cpu, iso_unit_cpu)
            basis_by_norm: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            inp = torch.tensor([input_ids] * B, dtype=torch.long, device=device)

            for norm_cond in NORM_ORDER:
                m_L = float(norms_map[norm_cond][str(u_layer)])
                basis_by_norm[norm_cond] = (hat_w_cpu.clone(), wd_unit_cpu.clone(), iso_unit_cpu.clone())
                deltas = build_b19_deltas(
                    hat_w_cpu=hat_w_cpu,
                    wd_unit_cpu=wd_unit_cpu,
                    iso_unit_cpu=iso_unit_cpu,
                    dose_mag=dose_mag,
                    norm_multiplier=m_L,
                    device=device,
                )

                read_pos_norm_box: list[float] = []

                def inject_hook(_mod, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        if not read_pos_norm_box:
                            read_pos_norm_box.append(
                                float(torch.linalg.vector_norm(hs[0, read_pos].detach().to(torch.float32)).item())
                            )
                        return (hs + deltas.unsqueeze(1).to(hs.dtype),) + out[1:]
                    if not read_pos_norm_box:
                        read_pos_norm_box.append(
                            float(torch.linalg.vector_norm(out[0, read_pos].detach().to(torch.float32)).item())
                        )
                    return out + deltas.unsqueeze(1).to(out.dtype)

                captured_o_in: dict[int, torch.Tensor] = {}

                def make_capture_pre_hook(hl: int):
                    def _pre(_mod, args):
                        captured_o_in[hl] = args[0].detach().clone()
                        return None
                    return _pre

                handles = [layers_mod[u_layer].register_forward_hook(inject_hook)]
                if head_patched:
                    for hl in head_layers_all:
                        handles.append(
                            layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(make_capture_pre_hook(hl))
                        )
                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        out_unp = model(input_ids=inp, use_cache=False).logits[:, read_pos, :]
                finally:
                    for h in handles:
                        h.remove()
                timer_acc["b19_sec"] += time.perf_counter() - t0
                timer_acc["b19_fwds"] += 1

                unp_raw, unp_pmed, unp_spec, unp_pmat = read_decomposed_logits(out_unp, t_own, peers)
                transport, unp_decomp_err = summarize_unpatched_transport(unp_raw, unp_pmed, unp_spec)
                max_abs_decomp_err = max(max_abs_decomp_err, unp_decomp_err)

                injected_norm = float(torch.linalg.vector_norm(deltas[1]).item())
                rp_norm = read_pos_norm_box[0]

                # Unpatched row: strictly NO sham keys (v3 §2.3 & §BR.0).
                unp_row = {
                    "bank": bank,
                    "norm_cond": norm_cond,
                    "unit": uid,
                    "unit_layer": u_layer,
                    "prompt_id": pid,
                    "cell": cell,
                    "story_or_frame": entry["story_or_frame"],
                    "block_a": entry["block_a"],
                    "dose_mag": dose_mag,
                    "head_patched": head_patched,
                    "ref_table_unpatched": entry["ref_table_unpatched"] if (bank == "NEW_BANK" and norm_cond == "NEW_NORM") else None,
                    "in_new_anchor": bool(bank == "NEW_BANK" and norm_cond == "NEW_NORM"),
                    "norm_multiplier": m_L,
                    "injected_norm": injected_norm,
                    "read_pos_resid_norm": rp_norm,
                    "norm_ratio": injected_norm / max(rp_norm, 1e-12),
                    "null_unit_vec_sha256": basis_sha,
                    "max_abs_decomp_err": unp_decomp_err,
                    "transport": transport,
                }
                unpatched_rows.append(unp_row)

                if not head_patched:
                    continue

                for hs in CANONICAL_HEADS:
                    hl = int(hs["layer"])
                    hidx = int(hs["head"])
                    hname = hs["name"]
                    assert_strictly_downstream(u_layer, hl)
                    z_self_full = captured_o_in[hl]
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

                    hd_decomp_err = check_head_linear_decomposition(spec_sum, raw_sum, pmed_sum)
                    max_abs_decomp_err = max(max_abs_decomp_err, hd_decomp_err)
                    max_abs_sham_spec = max(max_abs_sham_spec, float(spec_sum["max_abs_sham"]))
                    max_abs_sham_raw = max(max_abs_sham_raw, float(raw_sum["max_abs_sham"]))

                    if spec_sum["max_abs_sham"] > 1e-6 or raw_sum["max_abs_sham"] > 1e-6:
                        raise AssertionError(
                            f"Head sham exceeded 1e-6 on ({bank},{norm_cond},{uid},{pid},{dose_mag},{hname}): "
                            f"spec={spec_sum['max_abs_sham']}, raw={raw_sum['max_abs_sham']}"
                        )

                    in_old_anc = bool(
                        bank == "OLD_BANK"
                        and norm_cond == "OLD_NORM"
                        and entry["in_old_anchor_heads"]
                        and (hname in OLD_OVERLAP_HEADS)
                    )
                    in_new_anc = bool(bank == "NEW_BANK" and norm_cond == "NEW_NORM")

                    hrow: dict[str, Any] = {
                        "bank": bank,
                        "norm_cond": norm_cond,
                        "unit": uid,
                        "unit_layer": u_layer,
                        "prompt_id": pid,
                        "cell": cell,
                        "story_or_frame": entry["story_or_frame"],
                        "block_a": entry["block_a"],
                        "dose_mag": dose_mag,
                        "head_name": hname,
                        "head_layer": hl,
                        "head_idx": hidx,
                        "head_kind": hs["kind"],
                        "ref_table_head": (
                            "q12_rows" if in_old_anc else (entry["ref_table_head"] if in_new_anc else None)
                        ),
                        "in_old_anchor": in_old_anc,
                        "in_new_anchor": in_new_anc,
                        "null_unit_vec_sha256": basis_sha,
                        "max_abs_decomp_err": hd_decomp_err,
                        "spec": spec_sum,
                        "raw": raw_sum,
                        "peer_med": pmed_sum,
                    }
                    if rec_peers and (hname in PEER_DELTA_HEADS):
                        p_deltas: dict[str, dict[str, float]] = {}
                        dpos_vals: list[float] = []
                        for j, tid in enumerate(peers):
                            dp_j = unp_pmat[1][j] - pat_pmat[1][j]
                            dn_j = unp_pmat[2][j] - pat_pmat[2][j]
                            p_deltas[str(tid)] = {"d_pos": dp_j, "d_neg": dn_j}
                            dpos_vals.append(dp_j)
                        sorted_dp = sorted(dpos_vals)
                        q1 = sorted_dp[len(sorted_dp) // 4]
                        q3 = sorted_dp[(3 * len(sorted_dp)) // 4]
                        hrow["peer_deltas"] = p_deltas
                        hrow["peer_mean_d_pos"] = float(sum(dpos_vals) / len(dpos_vals))
                        hrow["peer_iqr_d_pos"] = float(q3 - q1)
                        hrow["frac_peer_tokens_pos"] = float(sum(1 for v in dpos_vals if v > 0.0) / len(dpos_vals))
                    head_rows.append(hrow)

            # Direct tensor check that unit-norm basis was strictly identical across OLD_NORM and NEW_NORM.
            for t_old, t_new in zip(basis_by_norm["OLD_NORM"], basis_by_norm["NEW_NORM"]):
                diff = float((t_old - t_new).abs().max().item())
                max_abs_null_vec_diff = max(max_abs_null_vec_diff, diff)

    total_sec = time.perf_counter() - t_start
    ms_per_b19 = (1000.0 * timer_acc["b19_sec"] / timer_acc["b19_fwds"]) if timer_acc["b19_fwds"] > 0 else None
    full_b19 = int(manifest["meta"]["denominators"]["full"]["n_b19_forwards"])
    artifact_type = "instruct_bridge_smoke_receipt" if smoke_1unit else "instruct_bridge_full_receipt"
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    exec_prov = "measured_gpu_run" if is_cuda else "cpu_or_dummy_run"
    dev_name = torch.cuda.get_device_name(0) if is_cuda else str(device)

    payload = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": artifact_type,
            "run_mode": "smoke_1unit" if smoke_1unit else "full",
            "execution_provenance": exec_prov,
            "device_name": dev_name,
            "complete": True,
            "smoke_1unit": bool(smoke_1unit),
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "design_sha256": pins["design_sha256"],
            "manifest_sha256": pins["manifest_sha256"],
            "runner_sha256": pins["runner_sha256"],
            "scorer_sha256": pins["scorer_sha256"],
            "pins": dict(pins),
            "source_pins": dict(SOURCE_PINS),
            "units_run": list(units_to_run),
            "n_unpatched_rows": len(unpatched_rows),
            "n_head_rows": len(head_rows),
            "n_b19_forwards": int(timer_acc["b19_fwds"]),
            "max_abs_null_vec_diff": float(max_abs_null_vec_diff),
            "max_abs_sham_spec": float(max_abs_sham_spec),
            "max_abs_sham_raw": float(max_abs_sham_raw),
            "max_abs_decomp_err": float(max_abs_decomp_err),
            "timing": {
                "model_load_sec": float(model_load_sec),
                "b19_elapsed_sec": float(timer_acc["b19_sec"]),
                "non_b19_runner_sec": float(max(0.0, total_sec - timer_acc["b19_sec"])),
                "total_runner_sec": float(total_sec),
                "n_b19_done": int(timer_acc["b19_fwds"]),
                "ms_per_b19": float(ms_per_b19) if ms_per_b19 is not None else None,
                "projected_full_b19_sec": float(ms_per_b19 * full_b19 / 1000.0) if ms_per_b19 is not None else None,
            },
        },
        "unit_vectors": unit_vectors_meta,
        "unpatched_rows": unpatched_rows,
        "head_rows": head_rows,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, out_path)

    return payload


# ---------------------------------------------------------------------------
# CPU Self-Test Suite (Dummy Model + Literal Sources + Zero-Load Refusals)
# ---------------------------------------------------------------------------

class _DummySelfAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class _DummyMLP(nn.Module):
    def __init__(self, hidden: int, d_ff: int):
        super().__init__()
        self.down_proj = nn.Linear(d_ff, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact equivalent of zero-padding x to d_ff then multiplying by down_proj.weight, without allocating [B, T, d_ff].
        return x @ self.down_proj.weight[:, : x.shape[-1]].T


class _LazyLogits:
    """Computes `x[:, pos, :] @ W.T` on slice access so CPU self-tests run in ~1s for 100k vocab."""

    def __init__(self, hidden_states: torch.Tensor, lm_weight: torch.Tensor):
        self._x = hidden_states
        self._w = lm_weight

    def __getitem__(self, idx):
        # Expected access pattern: logits[:, read_pos, :]
        if isinstance(idx, tuple) and len(idx) == 3 and idx[0] == slice(None) and idx[2] == slice(None):
            return self._x[:, idx[1], :] @ self._w.T
        return (self._x @ self._w.T)[idx]


class _DummyLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = _DummySelfAttn(hidden, n_heads)
        self.mlp = _DummyMLP(hidden, d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = x + 0.1 * self.self_attn(x)
        x = x + 0.05 * self.mlp(x)
        return (x,)


class _DummyOlmo(nn.Module):
    def __init__(self, vocab: int = 100352, hidden: int = 32, n_heads: int = 16, d_ff: int = 7200, n_layers: int = 10):
        super().__init__()
        self.config = type("Cfg", (), {"num_attention_heads": n_heads, "hidden_size": hidden})()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_DummyLayer(hidden, n_heads, d_ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        x = self.embed(input_ids % self.embed.num_embeddings)
        for lyr in self.layers:
            x = lyr(x)[0]
        return type("Out", (), {"logits": _LazyLogits(x, self.lm_head.weight)})()


def run_self_tests(repo_root: Path) -> dict[str, Any]:
    """Run comprehensive CPU self-tests (zero GPU, zero HF downloads)."""
    design_p = repo_root / DESIGN_REL_PATH
    manifest_p = repo_root / "explore_olmo2_attn_mlp/instruct_bridge_manifest.json"
    runner_p = Path(__file__).resolve()
    scorer_p = repo_root / "explore_olmo2_attn_mlp/score_instruct_bridge.py"
    old_rcpt_p = repo_root / "explore_olmo2_attn_mlp/results_signed_dose_full_d13fc3dd.json"
    new_rcpt_p = repo_root / "explore_olmo2_attn_mlp/results_stage_sweep_instruct_01bd41d9.json"

    old_man_b = load_git_blob(repo_root, OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/signed_dose_manifest.json")
    old_run_b = load_git_blob(repo_root, OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/run_signed_dose.py")
    new_man_b = load_git_blob(repo_root, NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/stage_sweep_manifest.json")
    new_run_b = load_git_blob(repo_root, NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/run_stage_sweep.py")
    old_rcpt_b = old_rcpt_p.read_bytes()
    new_rcpt_b = new_rcpt_p.read_bytes()

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    # 1. Valid literal-source control (Alden #93 item 1).
    lit_res = verify_literal_source_bindings(
        manifest=manifest,
        old_manifest_bytes=old_man_b,
        old_runner_bytes=old_run_b,
        new_manifest_bytes=new_man_b,
        new_runner_bytes=new_run_b,
        new_receipt_bytes=new_rcpt_b,
        old_receipt_bytes=old_rcpt_b,
    )
    assert lit_res["literal_source_bindings_verified"] is True

    # 2. Coordinate drift refusal with ZERO model/tokenizer loads.
    load_calls = 0

    def counting_loader(repo: str, commit: str, device: str) -> nn.Module:
        nonlocal load_calls
        load_calls += 1
        torch.manual_seed(42)
        return _DummyOlmo()

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        bad_man = json.loads(json.dumps(manifest))
        bad_man["meta"]["core_units"][1]["neuron"] = 7601  # Deliberate drift on p92-05
        bad_man_p = tdp / "bad_manifest.json"
        bad_man_p.write_text(json.dumps(bad_man), encoding="utf-8")
        try:
            admit_frozen_package(
                design_p, bad_man_p, runner_p, scorer_p,
                sha256_file(design_p), sha256_file(bad_man_p), sha256_file(runner_p), sha256_file(scorer_p),
            )
            raise AssertionError("Expected coordinate-drift refusal")
        except ValueError as e:
            assert "core unit drift" in str(e)
        assert load_calls == 0

        # 3. External SHA-256 pin mismatch refusal with ZERO loads.
        try:
            admit_frozen_package(
                design_p, manifest_p, runner_p, scorer_p,
                "0" * 64, sha256_file(manifest_p), sha256_file(runner_p), sha256_file(scorer_p),
            )
            raise AssertionError("Expected SHA-256 pin mismatch refusal")
        except ValueError as e:
            assert "pin mismatch" in str(e)
        assert load_calls == 0

        # 4. Literal-source drift refusal (tampered NEW_RECEIPT unit_vectors neuron).
        bad_rcpt = json.loads(new_rcpt_b.decode("utf-8"))
        bad_rcpt["unit_vectors"]["p92-46"]["neuron"] = 7346
        bad_rcpt_b = json.dumps(bad_rcpt).encode("utf-8")
        try:
            verify_literal_source_bindings(
                manifest, old_man_b, old_run_b, new_man_b, new_run_b, bad_rcpt_b, old_rcpt_b
            )
            raise AssertionError("Expected NEW_RECEIPT tamper refusal")
        except ValueError:
            pass

    # 5. Nonlinear median vs linear per-draw decomposition test (Alden #97).
    # Construct synthetic B=19 where per-variant `spec == raw - peer_med` is exact,
    # while `median(spec_draws) != median(raw_draws) - median(peer_med_draws)`!
    unp_r = [0.0] * 19
    unp_p = [0.0] * 19
    pat_r = [0.0, -1.0, 1.0, -1.0, -5.0, -2.0, -4.0, 1.0, 5.0, 2.0, 4.0, -1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0]
    pat_p = [0.0, -0.5, 0.5, -0.2, -4.9, -0.1, -0.3, 0.2, 4.9, 0.1, 0.3, -0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4]
    unp_s = [r - p for r, p in zip(unp_r, unp_p)]
    pat_s = [r - p for r, p in zip(pat_r, pat_p)]
    s_sum = summarize_b19_metric(unp_s, pat_s, unp_s)
    r_sum = summarize_b19_metric(unp_r, pat_r, unp_r)
    p_sum = summarize_b19_metric(unp_p, pat_p, unp_p)
    # Confirm medians genuinely do NOT subtract, yet linear per-draw check passes!
    med_gap = abs(s_sum["wdown_orth"]["d_pos_med"] - (r_sum["wdown_orth"]["d_pos_med"] - p_sum["wdown_orth"]["d_pos_med"]))
    assert med_gap > 0.1, f"Synthetic nonlinear-median test did not create gap: {med_gap}"
    assert check_head_linear_decomposition(s_sum, r_sum, p_sum) <= 1e-12

    # 6. Explicit-mode & full-after-smoke zero-load refusal controls (Alden #104 item 3).
    d_sha, m_sha, r_sha, s_sha = sha256_file(design_p), sha256_file(manifest_p), sha256_file(runner_p), sha256_file(scorer_p)
    try:
        admit_execution_request(None, design_p, manifest_p, runner_p, scorer_p, d_sha, m_sha, r_sha, s_sha)
        raise AssertionError("Expected zero-load refusal for implicit/omitted --mode")
    except ValueError as e:
        assert "implicit full execution is refused" in str(e)
    assert load_calls == 0

    try:
        admit_execution_request("full", design_p, manifest_p, runner_p, scorer_p, d_sha, m_sha, r_sha, s_sha)
        raise AssertionError("Expected zero-load refusal for --mode full without verified smoke gate")
    except ValueError as e:
        assert "--allow-full-after-smoke" in str(e)
    assert load_calls == 0

    # 7. Dummy-model smoke execution verifying exact schedule, 242 B19, 18 unp (0 sham), 112 head (100% sham).
    admitted_man, pins = admit_execution_request(
        "smoke", design_p, manifest_p, runner_p, scorer_p, d_sha, m_sha, r_sha, s_sha
    )
    smoke_payload = execute_bridge(
        manifest=admitted_man,
        pins=pins,
        smoke_1unit=True,
        device="cpu",
        model_loader=counting_loader,
    )
    assert load_calls == 1
    assert smoke_payload["meta"]["artifact_type"] == "instruct_bridge_smoke_receipt"
    assert smoke_payload["meta"]["execution_provenance"] == "cpu_or_dummy_run"
    assert smoke_payload["meta"]["complete"] is True
    assert smoke_payload["meta"]["n_unpatched_rows"] == 18
    assert smoke_payload["meta"]["n_head_rows"] == 112
    assert smoke_payload["meta"]["n_b19_forwards"] == 242
    assert smoke_payload["meta"]["max_abs_null_vec_diff"] == 0.0
    assert smoke_payload["meta"]["max_abs_sham_spec"] <= 1e-6
    assert smoke_payload["meta"]["max_abs_sham_raw"] <= 1e-6
    validate_timing_breakdown(smoke_payload["meta"]["timing"], 242)
    # Verify sham keys strictly absent on ALL 18 unpatched rows (including CAL_SELF).
    for ur in smoke_payload["unpatched_rows"]:
        flat = json.dumps(ur)
        assert "sham" not in flat, f"Unexpected sham key in unpatched row: {ur['prompt_id']}"
    cal_self_unp = [r for r in smoke_payload["unpatched_rows"] if r["cell"] == "CAL_SELF"]
    cal_self_hd = [r for r in smoke_payload["head_rows"] if r["cell"] == "CAL_SELF"]
    assert len(cal_self_unp) == 2 and len(cal_self_hd) == 0

    # 8. Paired CLI-level full-after-smoke verifier controls (Alden #108):
    # Exercises the ACTUAL CLI entry point `run_cli(...)` invoking pinned `score_instruct_bridge.verify_smoke_receipt_only`.
    scorer_mod = _load_pinned_scorer_module(scorer_p, s_sha)
    old_rcpt_obj = json.loads(old_rcpt_b.decode("utf-8"))
    new_rcpt_obj = json.loads(new_rcpt_b.decode("utf-8"))
    full_fixture = scorer_mod._make_full_receipt_from_real_anchors(admitted_man, pins, old_rcpt_obj, new_rcpt_obj)
    smoke_keys_set = {(e["bank"], e["prompt_id"], dose_key(e["dose_mag"])) for e in admitted_man["schedule_smoke"]}
    den_sm = admitted_man["meta"]["denominators"]["smoke"]
    anchor_smoke_gpu = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "instruct_bridge_smoke_receipt",
            "run_mode": "smoke_1unit",
            "smoke_1unit": True,
            "execution_provenance": "measured_gpu_run",
            "device_name": "NVIDIA L4",
            "complete": True,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "design_sha256": d_sha,
            "manifest_sha256": m_sha,
            "runner_sha256": r_sha,
            "scorer_sha256": s_sha,
            "pins": dict(pins),
            "units_run": [SMOKE_UNIT_ID],
            "n_unpatched_rows": den_sm["n_unpatched_rows"],
            "n_head_rows": den_sm["n_head_rows"],
            "n_b19_forwards": den_sm["n_b19_forwards"],
            "max_abs_null_vec_diff": 0.0,
            "timing": {
                "model_load_sec": 2.0,
                "b19_elapsed_sec": 24.2,
                "non_b19_runner_sec": 3.0,
                "total_runner_sec": 27.2,
                "n_b19_done": den_sm["n_b19_forwards"],
                "ms_per_b19": 100.0,
                "projected_full_b19_sec": 696.0,
            },
        },
        "unpatched_rows": [
            r for r in full_fixture["unpatched_rows"]
            if r["unit"] == SMOKE_UNIT_ID and (r["bank"], r["prompt_id"], dose_key(r["dose_mag"])) in smoke_keys_set
        ],
        "head_rows": [
            r for r in full_fixture["head_rows"]
            if r["unit"] == SMOKE_UNIT_ID and (r["bank"], r["prompt_id"], dose_key(r["dose_mag"])) in smoke_keys_set
        ],
    }

    class _LoaderSentinel(RuntimeError):
        pass

    cli_load_calls = 0

    def sentinel_loader(repo: str, commit: str, device: str) -> nn.Module:
        nonlocal cli_load_calls
        cli_load_calls += 1
        raise _LoaderSentinel("Full execution passed pinned smoke verifier and reached model loader")

    def base_full_cli(sm_path: Path, out_p: Path) -> list[str]:
        return [
            "--repo-root", str(repo_root),
            "--design", str(design_p),
            "--manifest", str(manifest_p),
            "--scorer", str(scorer_p),
            "--design-sha256", d_sha,
            "--manifest-sha256", m_sha,
            "--runner-sha256", r_sha,
            "--scorer-sha256", s_sha,
            "--mode", "full",
            "--allow-full-after-smoke",
            "--verified-smoke-receipt", str(sm_path),
            "--verified-smoke-sha256", sha256_file(sm_path),
            "--old-receipt", str(old_rcpt_p),
            "--new-receipt", str(new_rcpt_p),
            "--out", str(out_p),
        ]

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        dummy_out = tdp / "full_out.json"

        # Red CLI 1: Correctly sized header with ZERO body rows -> refused with ZERO loader calls!
        hdr_only = copy.deepcopy(anchor_smoke_gpu)
        hdr_only["unpatched_rows"] = []
        hdr_only["head_rows"] = []
        hdr_only_p = tdp / "hdr_only_smoke.json"
        hdr_only_p.write_text(json.dumps(hdr_only), encoding="utf-8")
        try:
            run_cli(base_full_cli(hdr_only_p, dummy_out), model_loader=sentinel_loader)
            raise AssertionError("Expected CLI zero-load refusal for header-only zero-body smoke")
        except ValueError as e:
            assert "row count mismatch" in str(e)
        assert cli_load_calls == 0

        # Red CLI 2 (Alden #108 exact witness): Structurally complete 18/112 smoke with 1 OLD anchor moved +0.005
        # (linear even/odd updated consistently) -> pinned technical verifier fails -> refused with ZERO loader calls!
        anc_bad = copy.deepcopy(anchor_smoke_gpu)
        old_idx = next(
            i for i, r in enumerate(anc_bad["head_rows"])
            if r["bank"] == "OLD_BANK" and r["norm_cond"] == "OLD_NORM" and r["head_name"] == "L8H15" and dose_key(r["dose_mag"]) == "0.8"
        )
        for mk in ("spec", "raw"):
            m = anc_bad["head_rows"][old_idx][mk]
            m["d_pos"] += 0.005
            m["even"] = 0.5 * (m["d_pos"] + m["d_neg"]) - m["d0"]
            m["odd"] = 0.5 * (m["d_pos"] - m["d_neg"])
        anc_bad_p = tdp / "anchor_failed_smoke.json"
        anc_bad_p.write_text(json.dumps(anc_bad), encoding="utf-8")
        try:
            run_cli(base_full_cli(anc_bad_p, dummy_out), model_loader=sentinel_loader)
            raise AssertionError("Expected CLI zero-load refusal for anchor-failed smoke (+0.005)")
        except ValueError as e:
            assert "pinned technical smoke verifier returned failed verdict" in str(e)
        assert cli_load_calls == 0

        # Red CLI 3: CPU dummy smoke cannot unlock --mode full!
        cpu_sm_p = tdp / "cpu_smoke.json"
        cpu_sm_p.write_text(json.dumps(smoke_payload), encoding="utf-8")
        try:
            run_cli(base_full_cli(cpu_sm_p, dummy_out), model_loader=sentinel_loader)
            raise AssertionError("Expected CLI zero-load refusal for cpu_or_dummy_run smoke")
        except ValueError as e:
            assert "measured_gpu_run" in str(e)
        assert cli_load_calls == 0

        # Green CLI: Structurally complete 18/112 anchor-passing measured_gpu_run smoke exercises
        # the EXACT same pinned technical verifier and reaches the model loader (cli_load_calls == 1)!
        good_sm_p = tdp / "anchor_passed_smoke.json"
        good_sm_p.write_text(json.dumps(anchor_smoke_gpu), encoding="utf-8")
        try:
            run_cli(base_full_cli(good_sm_p, dummy_out), model_loader=sentinel_loader)
            raise AssertionError("Expected _LoaderSentinel after passing pinned technical smoke verifier")
        except _LoaderSentinel:
            pass
        assert cli_load_calls == 1

        # 9. Single-read exact-input binding controls (Alden #111: changed-input negative + stable positive).
        # Proves manifest, scorer, and smoke are each read EXACTLY ONCE so a second-read replacement mock
        # cannot swap invalid pinned smoke bytes for valid bytes between hashing and parsing!
        orig_read_bytes = Path.read_bytes
        orig_read_text = Path.read_text
        good_bytes = good_sm_p.read_bytes()
        bad_bytes = anc_bad_p.read_bytes()
        bad_sha = sha256_bytes(bad_bytes)
        good_sha = sha256_bytes(good_bytes)

        # 9a. Changed-input negative control (TOCTOU swap queued on hypothetical 2nd read):
        read_counts: dict[Path, int] = {}

        def toctou_read_bytes(self_p: Path) -> bytes:
            rp = self_p.resolve()
            read_counts[rp] = read_counts.get(rp, 0) + 1
            if rp == anc_bad_p.resolve():
                return bad_bytes if read_counts[rp] == 1 else good_bytes
            return orig_read_bytes(self_p)

        def forbid_read_text(self_p: Path, *a, **kw) -> str:
            rp = self_p.resolve()
            if rp in {anc_bad_p.resolve(), good_sm_p.resolve(), manifest_p.resolve(), scorer_p.resolve()}:
                raise AssertionError(f"Unexpected second read via read_text on {rp}")
            return orig_read_text(self_p, *a, **kw)

        argv_bad_swap = [
            "--repo-root", str(repo_root),
            "--design", str(design_p), "--manifest", str(manifest_p), "--scorer", str(scorer_p),
            "--design-sha256", d_sha, "--manifest-sha256", m_sha, "--runner-sha256", r_sha, "--scorer-sha256", s_sha,
            "--mode", "full", "--allow-full-after-smoke",
            "--verified-smoke-receipt", str(anc_bad_p), "--verified-smoke-sha256", bad_sha,
            "--old-receipt", str(old_rcpt_p), "--new-receipt", str(new_rcpt_p),
            "--out", str(dummy_out),
        ]
        cli_load_calls = 0
        with patch.object(Path, "read_bytes", toctou_read_bytes), patch.object(Path, "read_text", forbid_read_text):
            try:
                run_cli(argv_bad_swap, model_loader=sentinel_loader)
                raise AssertionError("Expected refusal under changed-input TOCTOU replacement mock")
            except ValueError as e:
                assert "pinned technical smoke verifier returned failed verdict" in str(e)
        assert cli_load_calls == 0
        assert read_counts.get(manifest_p.resolve()) == 1, f"manifest read count={read_counts.get(manifest_p.resolve())}"
        assert read_counts.get(scorer_p.resolve()) == 1, f"scorer read count={read_counts.get(scorer_p.resolve())}"
        assert read_counts.get(anc_bad_p.resolve()) == 1, f"smoke read count={read_counts.get(anc_bad_p.resolve())}"

        # 9b. Stable positive control (every pinned artifact read exactly once, admitted to loader):
        read_counts_pos: dict[Path, int] = {}

        def counting_read_bytes(self_p: Path) -> bytes:
            rp = self_p.resolve()
            read_counts_pos[rp] = read_counts_pos.get(rp, 0) + 1
            return orig_read_bytes(self_p)

        argv_good_once = [
            "--repo-root", str(repo_root),
            "--design", str(design_p), "--manifest", str(manifest_p), "--scorer", str(scorer_p),
            "--design-sha256", d_sha, "--manifest-sha256", m_sha, "--runner-sha256", r_sha, "--scorer-sha256", s_sha,
            "--mode", "full", "--allow-full-after-smoke",
            "--verified-smoke-receipt", str(good_sm_p), "--verified-smoke-sha256", good_sha,
            "--old-receipt", str(old_rcpt_p), "--new-receipt", str(new_rcpt_p),
            "--out", str(dummy_out),
        ]
        cli_load_calls = 0
        with patch.object(Path, "read_bytes", counting_read_bytes), patch.object(Path, "read_text", forbid_read_text):
            try:
                run_cli(argv_good_once, model_loader=sentinel_loader)
                raise AssertionError("Expected _LoaderSentinel on stable single-read positive control")
            except _LoaderSentinel:
                pass
        assert cli_load_calls == 1
        assert read_counts_pos.get(manifest_p.resolve()) == 1
        assert read_counts_pos.get(scorer_p.resolve()) == 1
        assert read_counts_pos.get(good_sm_p.resolve()) == 1

    return {
        "runner_self_tests_passed": True,
        "literal_source_control": lit_res,
        "nonlinear_median_gap_verified": med_gap,
        "implicit_full_mode_zero_load_refusal_verified": True,
        "cli_full_gate_header_only_zero_load_refusal_verified": True,
        "cli_full_gate_anchor_failed_zero_load_refusal_verified": True,
        "cli_full_gate_anchor_passed_admitted_verified": True,
        "single_read_toctou_changed_input_negative_verified": True,
        "single_read_stable_positive_verified": True,
        "dummy_smoke_b19_fwds": smoke_payload["meta"]["n_b19_forwards"],
        "dummy_smoke_unpatched_rows": len(smoke_payload["unpatched_rows"]),
        "dummy_smoke_head_rows": len(smoke_payload["head_rows"]),
    }


def run_cli(
    argv: list[str] | None = None,
    model_loader: Callable[[str, str, str], nn.Module] = _default_model_loader,
) -> None:
    ap = argparse.ArgumentParser(description="OLMo 2 1B Instruct 2x2 Bank x Norm Bridge Runner")
    ap.add_argument("--build-manifest", action="store_true", help="Build instruct_bridge_manifest.json from frozen sources")
    ap.add_argument("--self-test-only", action="store_true", help="Run CPU self-tests without HF/GPU loads")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--scorer", type=Path, default=None)
    ap.add_argument("--old-receipt", type=Path, default=None)
    ap.add_argument("--new-receipt", type=Path, default=None)
    ap.add_argument("--design-sha256", type=str, default="")
    ap.add_argument("--manifest-sha256", type=str, default="")
    ap.add_argument("--runner-sha256", type=str, default="")
    ap.add_argument("--scorer-sha256", type=str, default="")
    ap.add_argument("--mode", choices=("smoke", "full"), default=None, help="Explicit execution mode (required for runs)")
    ap.add_argument("--allow-full-after-smoke", action="store_true", help="Required alongside --verified-smoke-receipt for --mode full")
    ap.add_argument("--verified-smoke-receipt", type=Path, default=None, help="Measured GPU smoke receipt gating --mode full")
    ap.add_argument("--verified-smoke-sha256", type=str, default="", help="Exact SHA-256 of verified smoke receipt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    design_p = args.design or (repo_root / DESIGN_REL_PATH)
    manifest_p = args.manifest or (repo_root / "explore_olmo2_attn_mlp/instruct_bridge_manifest.json")
    runner_p = Path(__file__).resolve()
    scorer_p = args.scorer or (repo_root / "explore_olmo2_attn_mlp/score_instruct_bridge.py")
    old_rcpt_p = args.old_receipt or (repo_root / "explore_olmo2_attn_mlp/results_signed_dose_full_d13fc3dd.json")
    new_rcpt_p = args.new_receipt or (repo_root / "explore_olmo2_attn_mlp/results_stage_sweep_instruct_01bd41d9.json")

    if args.build_manifest:
        old_man_b = load_git_blob(repo_root, OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/signed_dose_manifest.json")
        old_run_b = load_git_blob(repo_root, OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/run_signed_dose.py")
        new_man_b = load_git_blob(repo_root, NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/stage_sweep_manifest.json")
        new_run_b = load_git_blob(repo_root, NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/run_stage_sweep.py")
        old_rcpt_b = old_rcpt_p.read_bytes()
        new_rcpt_b = new_rcpt_p.read_bytes()
        man = build_bridge_manifest(
            old_man_b, old_run_b, new_man_b, new_run_b, new_rcpt_b, old_rcpt_b, design_p
        )
        manifest_p.write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(json.dumps({
            "manifest_built": str(manifest_p),
            "manifest_sha256": sha256_file(manifest_p),
            "denominators": man["meta"]["denominators"],
        }, indent=2))
        return

    if args.self_test_only:
        res = run_self_tests(repo_root)
        print(json.dumps(res, indent=2))
        return

    if args.out is None:
        raise ValueError("--out is required for execution runs")

    manifest, pins = admit_execution_request(
        mode=args.mode,
        design_path=design_p,
        manifest_path=manifest_p,
        runner_path=runner_p,
        scorer_path=scorer_p,
        expected_design_sha=args.design_sha256,
        expected_manifest_sha=args.manifest_sha256,
        expected_runner_sha=args.runner_sha256,
        expected_scorer_sha=args.scorer_sha256,
        allow_full_after_smoke=args.allow_full_after_smoke,
        verified_smoke_receipt=args.verified_smoke_receipt,
        verified_smoke_sha256=args.verified_smoke_sha256,
        old_receipt_path=old_rcpt_p,
        new_receipt_path=new_rcpt_p,
    )
    smoke_1unit = args.mode == "smoke"
    print(f"[PRECHECK OK] 4-pin & mode={args.mode} gate passed", flush=True)
    payload = execute_bridge(
        manifest=manifest,
        pins=pins,
        smoke_1unit=smoke_1unit,
        device=args.device,
        model_loader=model_loader,
        out_path=args.out,
    )
    print(json.dumps({
        "artifact_type": payload["meta"]["artifact_type"],
        "out": str(args.out),
        "out_sha256": sha256_file(args.out),
        "n_unpatched_rows": payload["meta"]["n_unpatched_rows"],
        "n_head_rows": payload["meta"]["n_head_rows"],
        "n_b19_forwards": payload["meta"]["n_b19_forwards"],
        "max_abs_sham_spec": payload["meta"]["max_abs_sham_spec"],
        "max_abs_sham_raw": payload["meta"]["max_abs_sham_raw"],
        "max_abs_null_vec_diff": payload["meta"]["max_abs_null_vec_diff"],
        "timing": payload["meta"]["timing"],
    }, indent=2))


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
