#!/usr/bin/env python3
"""OLMo 2 1B Instruct — Concept vs. Slot Head-Gating & Late-Layer Lens-Screened Heads Runner.

Implements `explore_olmo2_attn_mlp/CONCEPT-VS-SLOT-DESIGN.md` v3.1 (Design PASS Frozen;
SHA-256: cf67a12db8d914ad896b75aa036bd919c4616d576c2e4872e0dde273806c0c6f).

Hardening & contracts:
- Single-read external 6-pin binding (`design`, `manifest`, `runner`, `scorer`,
  `targets213`, `m7_transport`) enforced BEFORE any tokenizer or model load.
- Explicit `--mode {smoke,full}`; `--mode full` is hard-gated pre-load on
  `--allow-full-after-smoke` AND the pinned scorer's
  `verify_smoke_receipt_only(..., allow_fixture=False)` executed from the already-hashed
  scorer buffer.
- Exact post-`post_feedforward_layernorm` (`model.model.layers[L]`) residual push at
  `±0.8 * OLD_NORM[L]` in unified `B = 11` batches (`[0.0, +0.8, -0.8, 4*wdown_+, 4*wdown_-]`).
- Single-pass causal full-phrase teacher-forcing (`prefix_ids + content_ids[:-1]`)
  extracting all 5 readouts (`RawLogit`, `PeerMed`, `SpecLogit`, `TargetLogProb`,
  `PhraseLogProbMean` + `base_phrase_logp`) in one forward, verified against prefix-only
  reference forwards (`<= 1e-5`) on multi-token units (`CS.0` Clause 6).
- Exact `o_proj` pre-hook `z_all` single-head path-blocking with row-0 same-hook-path
  sham (`|d0| <= 1e-6`) and 15-sibling zero-leakage (`<= 1e-6`) on EVERY blocked row,
  PLUS 26 dedicated full-B11 measured same-hook-path sham forwards (1 per head) and
  26 B=1 synthetic orthogonal-donor actuation checks (`min_target_delta > 1e-2`).
"""

from __future__ import annotations

import argparse
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

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = "concept_vs_slot_v1.0"
DESIGN_REL_PATH = "explore_olmo2_attn_mlp/CONCEPT-VS-SLOT-DESIGN.md"
DESIGN_SHA256 = "88986b904f499893ec5aba4039af9142846e45307552409cdaf0237f21b90bbd"

TARGETS213_REL_PATH = "part213/targets213.json"
TARGETS213_SHA256 = "7edb3c6253470d157896931083c239253930be08f16a5b9348d6c67d3b431892"

M7_TRANSPORT_REL_PATH = "explore_olmo2_attn_mlp/results_old_b1_b4_transport_feasibility.json"
M7_TRANSPORT_SHA256 = "dcfa27e9542ac5ffba76aae8fb0c250adcd04a3f988877ce209c2864ff1f3b0f"

SCORE92_SHA256 = "4c4531ecd4ea5dd21251aef6d3bac695d191c979a6b7523ab274aa2d24ae8fdb"
PART99_PUSHES_SHA256 = "824cfebe2a8db27b55fc4edcec0c521a5a6b5a936e3408d97952dedae36316cb"
LENS_DISCOVERY_SHA256 = "9d5dd317a98bfece6721d47acf8bc97c831da283cb97e34c8e96348e4375ff4d"

MODEL_REPO = "allenai/OLMo-2-0425-1B-Instruct"
MODEL_COMMIT = "48d788eca847d4d7548f375ad03d3c9312f6139e"
DTYPE_STR = "float32"
ATTN_IMPL = "eager"

NOMINAL_DOSE_MAG = 0.8
N_NULL_DRAWS = 4
BATCH_SIZE = 3 + 2 * N_NULL_DRAWS  # 11: [0.0, +0.8, -0.8, 4*wdown_+, 4*wdown_-]
NULL_SEED_BASE = 2026112200
NULL_DOMAIN_TAG = "concept_vs_slot_v1"
SYNTH_ACTUATION_SEED = 2026112250

# Pinned OLD_NORM multipliers (exact full-precision match to results/part99-pushes.json["medians_chat"]).
OLD_NORM: dict[str, float] = {
    "2": 7.285534858703613,
    "3": 7.3919267654418945,
    "4": 9.044662475585938,
    "5": 10.05025863647461,
    "6": 10.501473426818848,
}

# Frozen CS.0 tolerances.
TOL_SHAM = 1e-6
TOL_DECOMP = 1e-6
TOL_SIBLING = 1e-6
TOL_ORTH = 1e-5
TOL_CAUSAL_EQUIV = 1e-5
TOL_BLOCK_ACTUATION = 1e-8
MIN_ACTUATION_DELTA = 1e-2
MIN_ACTUATION_READOUT_DELTA = 1e-4

# Historical branches checked for zero prompt-bank overlap during manifest build.
HIST_COMMITS_MANIFESTS: tuple[tuple[str, str], ...] = (
    ("d13fc3dd2b76309d28b001ecf33f0fc9c962f156", "explore_olmo2_attn_mlp/signed_dose_manifest.json"),
    ("01bd41d963936d39f68b032eda845ad73b13d6a2", "explore_olmo2_attn_mlp/stage_sweep_manifest.json"),
    ("fb6c81f9", "explore_olmo2_attn_mlp/causal_attn_manifest.json"),
)

# Frozen 12 outcome-blind same-layer matched primary pairs (§2.4 A).
PRIMARY_PAIRS: list[dict[str, Any]] = [
    {"pair_id": "P01", "layer": 2, "slot_unit": "p92-10", "concept_unit": "p92-29", "same_persona_kind": False},
    {"pair_id": "P02", "layer": 3, "slot_unit": "p92-04", "concept_unit": "p92-00", "same_persona_kind": True},
    {"pair_id": "P03", "layer": 3, "slot_unit": "p92-02", "concept_unit": "p92-11", "same_persona_kind": True},
    {"pair_id": "P04", "layer": 3, "slot_unit": "p92-13", "concept_unit": "p92-39", "same_persona_kind": False},
    {"pair_id": "P05", "layer": 4, "slot_unit": "p92-24", "concept_unit": "p92-01", "same_persona_kind": True},
    {"pair_id": "P06", "layer": 4, "slot_unit": "p92-35", "concept_unit": "p92-48", "same_persona_kind": False},
    {"pair_id": "P07", "layer": 4, "slot_unit": "p92-36", "concept_unit": "p92-08", "same_persona_kind": False},
    {"pair_id": "P08", "layer": 4, "slot_unit": "p92-47", "concept_unit": "p92-45", "same_persona_kind": True},
    {"pair_id": "P09", "layer": 5, "slot_unit": "p92-33", "concept_unit": "p92-15", "same_persona_kind": False},
    {"pair_id": "P10", "layer": 5, "slot_unit": "p92-37", "concept_unit": "p92-38", "same_persona_kind": False},
    {"pair_id": "P11", "layer": 5, "slot_unit": "p92-46", "concept_unit": "p92-41", "same_persona_kind": True},
    {"pair_id": "P12", "layer": 6, "slot_unit": "p92-09", "concept_unit": "p92-40", "same_persona_kind": False},
]
PRIMARY_SLOT_UNITS: tuple[str, ...] = tuple(p["slot_unit"] for p in PRIMARY_PAIRS)
PRIMARY_CONCEPT_UNITS: tuple[str, ...] = tuple(p["concept_unit"] for p in PRIMARY_PAIRS)
PRIMARY_UNITS_24: tuple[str, ...] = PRIMARY_SLOT_UNITS + PRIMARY_CONCEPT_UNITS
EXTRA_UNITS: tuple[str, ...] = ("p92-05", "p92-43")
ALL_UNITS_26: tuple[str, ...] = PRIMARY_UNITS_24 + EXTRA_UNITS

# Frozen clean-token modifier-first exclusion list (§5 CS.2 7b; plus is_full_word_first_token==False).
FROZEN_MODIFIER_FIRST_UNITS: frozenset[str] = frozenset({
    "p92-04", "p92-39", "p92-36", "p92-15", "p92-37", "p92-46", "p92-41",
})

# Fresh unseen 6-story bank C1..C6 (§3.1).
C_BANK_STORIES: list[dict[str, str]] = [
    {
        "story": "C1",
        "scene": "Under the glass bridge at dusk",
        "verb": "fixed",
        "obj": "the silver clock",
        "name": "Paul",
        "role": "the porter",
    },
    {
        "story": "C2",
        "scene": "Over the iron gate at noon",
        "verb": "moved",
        "obj": "the wooden crate",
        "name": "Anna",
        "role": "the pilot",
    },
    {
        "story": "C3",
        "scene": "Inside the old mill at dawn",
        "verb": "washed",
        "obj": "the warm coat",
        "name": "James",
        "role": "the miner",
    },
    {
        "story": "C4",
        "scene": "Past the steep hill at night",
        "verb": "tied",
        "obj": "the thick cord",
        "name": "Kate",
        "role": "the mayor",
    },
    {
        "story": "C5",
        "scene": "Across the wide deck at dawn",
        "verb": "checked",
        "obj": "the brass bell",
        "name": "Carl",
        "role": "the coach",
    },
    {
        "story": "C6",
        "scene": "Below the high cliff at dusk",
        "verb": "filled",
        "obj": "the clay bowl",
        "name": "Jane",
        "role": "the judge",
    },
]
BANK_CELLS: tuple[str, ...] = ("SELF", "NAME_BARE", "ROLE_BARE")
EXPECTED_SN_DIFF_INDICES: list[int] = [13, 20, 30, 31]

# 13 neutral single-token nouns completing the 37-token competitor pool (§3.2).
NEUTRAL_POOL_TOKENS: list[tuple[int, str]] = [
    (14351, " basket"),
    (15994, " bucket"),
    (17543, " anchor"),
    (18327, " mirror"),
    (24354, " hammer"),
    (24653, " compass"),
    (38899, " candle"),
    (39139, " blanket"),
    (42390, " marble"),
    (45667, " ribbon"),
    (50876, " feather"),
    (73418, " kettle"),
    (74265, " lantern"),
]

# Frozen 26-head battery (§4.1).
HEAD_BATTERY: list[dict[str, Any]] = [
    # L7 (CS.4, 4 heads)
    {"name": "L7H5", "layer": 7, "head": 5, "family": "CS.4", "role": "target",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0,
     "primary_controls": ["L7H13", "L7H4", "L7H9"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L7H13", "layer": 7, "head": 13, "family": "CS.4", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L7H4", "layer": 7, "head": 4, "family": "CS.4", "role": "legacy_part_b_control",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L7H9", "layer": 7, "head": 9, "family": "CS.4", "role": "legacy_part_b_control",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    # L8 (CS.2, 7 heads)
    {"name": "L8H2", "layer": 8, "head": 2, "family": "CS.2", "role": "primary_target",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0,
     "primary_controls": ["L8H3", "L8H13", "L8H7", "L8H9", "L8H0"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L8H15", "layer": 8, "head": 15, "family": "CS.2", "role": "legacy_comparator",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0,
     "primary_controls": ["L8H3", "L8H13", "L8H7", "L8H9", "L8H0"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L8H3", "layer": 8, "head": 3, "family": "CS.2", "role": "legacy_part_b_control",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L8H13", "layer": 8, "head": 13, "family": "CS.2", "role": "legacy_part_b_control",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L8H7", "layer": 8, "head": 7, "family": "CS.2", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L8H9", "layer": 8, "head": 9, "family": "CS.2", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L8H0", "layer": 8, "head": 0, "family": "CS.2", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    # L11 (CS.3, 7 heads)
    {"name": "L11H9", "layer": 11, "head": 9, "family": "CS.3", "role": "cand_self",
     "pred_c_sr_sign": 1, "pred_c_sn_sign": 1,
     "primary_controls": ["L11H1"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L11H2", "layer": 11, "head": 2, "family": "CS.3", "role": "cand_self",
     "pred_c_sr_sign": 1, "pred_c_sn_sign": 1,
     "primary_controls": ["L11H6"], "secondary_controls": ["L11H4"], "copy_r_saturated": True},
    {"name": "L11H3", "layer": 11, "head": 3, "family": "CS.3", "role": "cand_role",
     "pred_c_sr_sign": -1, "pred_c_sn_sign": -1,
     "primary_controls": ["L11H10", "L11H4"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L11H1", "layer": 11, "head": 1, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L11H6", "layer": 11, "head": 6, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L11H4", "layer": 11, "head": 4, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L11H10", "layer": 11, "head": 10, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    # L13 (CS.3, 4 heads)
    {"name": "L13H1", "layer": 13, "head": 1, "family": "CS.3", "role": "cand_self",
     "pred_c_sr_sign": 1, "pred_c_sn_sign": 1,
     "primary_controls": ["L13H13", "L13H5"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L13H4", "layer": 13, "head": 4, "family": "CS.3", "role": "cand_role",
     "pred_c_sr_sign": -1, "pred_c_sn_sign": -1,
     "primary_controls": ["L13H5", "L13H13"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L13H5", "layer": 13, "head": 5, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L13H13", "layer": 13, "head": 13, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    # L14 (CS.3, 4 heads)
    {"name": "L14H10", "layer": 14, "head": 10, "family": "CS.3", "role": "cand_self",
     "pred_c_sr_sign": 1, "pred_c_sn_sign": 1,
     "primary_controls": ["L14H12"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L14H3", "layer": 14, "head": 3, "family": "CS.3", "role": "cand_role",
     "pred_c_sr_sign": -1, "pred_c_sn_sign": -1,
     "primary_controls": ["L14H8"], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L14H12", "layer": 14, "head": 12, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
    {"name": "L14H8", "layer": 14, "head": 8, "family": "CS.3", "role": "neutral_ctrl",
     "pred_c_sr_sign": 0, "pred_c_sn_sign": 0, "primary_controls": [], "secondary_controls": [], "copy_r_saturated": False},
]
HEAD_NAMES_26: tuple[str, ...] = tuple(h["name"] for h in HEAD_BATTERY)
HEAD_BY_NAME: dict[str, dict[str, Any]] = {h["name"]: h for h in HEAD_BATTERY}

# Smoke blocked pair (P08 on C1) & positive-control/causal-equiv fixtures.
SMOKE_BLOCKED_UNITS: tuple[str, ...] = ("p92-47", "p92-45")
SMOKE_BLOCKED_STORY = "C1"
ACTUATION_AND_SHAM_UNIT = "p92-47"
ACTUATION_AND_SHAM_PROMPT = "C1_SELF"
CAUSAL_EQUIV_CHECK_UNITS: tuple[str, ...] = ("p92-00", "p92-45", "p92-39")  # nc = 2, 4, 7
CAUSAL_EQUIV_CHECK_PROMPT = "C1_SELF"

# OLD B1..B4 prompts (used ONLY inside --build-manifest to record unpushed OLD rmsDLA/srDLA metadata; zero C1..C6 reads).
OLD_B_STORIES: list[dict[str, str]] = [
    {"story": "B1", "scene": "At the cedar barn after dusk", "verb": "locked", "obj": "the copper latch", "name": "Mark", "role": "the painter"},
    {"story": "B2", "scene": "By the brick shop before dawn", "verb": "swept", "obj": "the stone steps", "name": "Sarah", "role": "the doctor"},
    {"story": "B3", "scene": "Near the pine ridge at noon", "verb": "packed", "obj": "the canvas tent", "name": "David", "role": "the ranger"},
    {"story": "B4", "scene": "In the stone hall after rain", "verb": "folded", "obj": "the linen map", "name": "Grace", "role": "the writer"},
]

CHANNELS_5: tuple[str, ...] = ("raw", "peer_med", "spec", "logp", "phrase_mean")


# ---------------------------------------------------------------------------
# Hashing, Prompt Construction & Manifest Builder
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def assert_strictly_downstream(unit_layer: int, head_layer: int) -> None:
    if not (int(head_layer) > int(unit_layer)):
        raise ValueError(f"Non-causal head_layer={head_layer} <= unit_layer={unit_layer}")


def load_git_blob(repo_root: Path, commit: str, rel_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo_root), "show", f"{commit}:{rel_path}"])


def build_prompt_strings(story: dict[str, str], cell: str, lead_class: str) -> tuple[str, str]:
    scene, verb, obj, name, role = story["scene"], story["verb"], story["obj"], story["name"], story["role"]
    role_cap = role[0].upper() + role[1:]
    if cell == "SELF":
        user_msg = f"{scene}, you {verb} {obj}. Identify yourself in one line."
        asst = "I am" if lead_class == "bare" else f"I am {lead_class}"
    elif cell == "NAME_BARE":
        user_msg = f"{scene}, {name} {verb} {obj}. Identify {name} in one line."
        asst = f"{name} is" if lead_class == "bare" else f"{name} is {lead_class}"
    elif cell == "ROLE_BARE":
        user_msg = f"{scene}, {role} {verb} {obj}. Identify {role} in one line."
        asst = f"{role_cap} is" if lead_class == "bare" else f"{role_cap} is {lead_class}"
    else:
        raise ValueError(f"Unknown cell: {cell!r}")
    return user_msg, asst


def _compute_head_weight_and_old_dla_stats(
    tok: Any,
    model: nn.Module,
    pool_ids: list[int],
) -> dict[str, dict[str, float]]:
    """Compute pinned ov_norm and OLD B1..B4 unpushed rmsDLA/srDLA on the 37-token pool (zero C1..C6 reads)."""
    layers_mod = get_decoder_layers(model)
    n_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // n_heads
    head_layers = sorted({h["layer"] for h in HEAD_BATTERY})

    # 1. Exact OV Frobenius norms ||W_O[:, s:e] @ W_V[s:e, :]||_F per head.
    ov_norms: dict[str, float] = {}
    for h in HEAD_BATTERY:
        hl, hi = h["layer"], h["head"]
        s, e = hi * head_dim, (hi + 1) * head_dim
        attn = layers_mod[hl].self_attn
        w_v = attn.v_proj.weight.detach().to(torch.float32)[s:e, :]
        w_o = attn.o_proj.weight.detach().to(torch.float32)[:, s:e]
        ov_norms[h["name"]] = float(torch.linalg.matrix_norm(w_o @ w_v, ord="fro").item())

    # 2. Unpushed OLD B1..B4 DLA on the 37-token pool (4 stories * 3 cells = 12 B=1 forwards, lead="a").
    w_u_pool = model.lm_head.weight.detach().to(torch.float32)[pool_ids, :]  # (37, d_model)
    norm_w = model.model.norm.weight.detach().to(torch.float32) if hasattr(model.model, "norm") else model.norm.weight.detach().to(torch.float32)
    norm_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    per_head_cell_dla: dict[str, dict[str, list[torch.Tensor]]] = {
        h["name"]: {c: [] for c in BANK_CELLS} for h in HEAD_BATTERY
    }

    for st in OLD_B_STORIES:
        for cell in BANK_CELLS:
            u_msg, asst = build_prompt_strings(st, cell, "a")
            full_text = tok.apply_chat_template([{"role": "user", "content": u_msg}], tokenize=False, add_generation_prompt=True) + asst
            ids = [int(x) for x in tok.encode(full_text, add_special_tokens=False)]
            read_pos = len(ids) - 1
            inp = torch.tensor([ids], dtype=torch.long)

            cap_o: dict[int, torch.Tensor] = {}
            cap_final_pre_norm: list[torch.Tensor] = []
            handles = []
            for hl in head_layers:
                handles.append(
                    layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(
                        lambda _m, a, _hl=hl: (cap_o.__setitem__(_hl, a[0][0, read_pos].detach().to(torch.float32)), None)[1]
                    )
                )
            handles.append(
                layers_mod[-1].register_forward_hook(
                    lambda _m, _i, o: (cap_final_pre_norm.append((o[0] if isinstance(o, tuple) else o)[0, read_pos].detach().to(torch.float32)), None)[1]
                )
            )
            try:
                with torch.no_grad():
                    model(input_ids=inp, use_cache=False)
            finally:
                for hh in handles:
                    hh.remove()

            h_fin = cap_final_pre_norm[0]
            rms = float(torch.sqrt(h_fin.pow(2).mean() + norm_eps).item())
            scale_vec = norm_w / rms  # linear final-norm scale at read_pos

            for h in HEAD_BATTERY:
                hl, hi = h["layer"], h["head"]
                s, e = hi * head_dim, (hi + 1) * head_dim
                w_o_h = layers_mod[hl].self_attn.o_proj.weight.detach().to(torch.float32)[:, s:e]
                resid_h = w_o_h @ cap_o[hl][s:e]
                dla_37 = w_u_pool @ (resid_h * scale_vec)
                per_head_cell_dla[h["name"]][cell].append(dla_37)

    out: dict[str, dict[str, float]] = {}
    for h in HEAD_BATTERY:
        hn = h["name"]
        all_vecs = [v for c in BANK_CELLS for v in per_head_cell_dla[hn][c]]
        stacked = torch.stack(all_vecs, dim=0)  # (12, 37)
        rms_dla = float(torch.sqrt(stacked.pow(2).mean()).item())
        s_mat = torch.stack(per_head_cell_dla[hn]["SELF"], dim=0)
        r_mat = torch.stack(per_head_cell_dla[hn]["ROLE_BARE"], dim=0)
        sr_dla = float(torch.sqrt((s_mat - r_mat).pow(2).mean()).item())
        out[hn] = {"ov_norm": ov_norms[hn], "old_rms_dla": rms_dla, "old_sr_dla": sr_dla}
    return out


def build_concept_vs_slot_manifest(
    repo_root: Path,
    design_path: Path,
    targets213_path: Path,
    m7_transport_path: Path,
    score92_path: Path,
    part99_pushes_path: Path,
    lens_discovery_path: Path,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
    model_loader: Callable[[str, str, str], nn.Module] | None = None,
) -> dict[str, Any]:
    """Deterministically build `concept_vs_slot_manifest.json` pinned to v3.1 design & source artifacts."""
    for label, p, exp in [
        ("design", design_path, DESIGN_SHA256),
        ("targets213", targets213_path, TARGETS213_SHA256),
        ("m7_transport", m7_transport_path, M7_TRANSPORT_SHA256),
        ("score92", score92_path, SCORE92_SHA256),
        ("part99_pushes", part99_pushes_path, PART99_PUSHES_SHA256),
        ("lens_discovery", lens_discovery_path, LENS_DISCOVERY_SHA256),
    ]:
        got = sha256_file(p)
        if got != exp:
            raise ValueError(f"Source pin mismatch for {label} ({p}): got {got}, expected {exp}")

    t213 = json.loads(targets213_path.read_text(encoding="utf-8"))
    m7 = json.loads(m7_transport_path.read_text(encoding="utf-8"))
    s92 = json.loads(score92_path.read_text(encoding="utf-8"))
    p99 = json.loads(part99_pushes_path.read_text(encoding="utf-8"))
    lens = json.loads(lens_discovery_path.read_text(encoding="utf-8"))

    # Verify OLD_NORM against results/part99-pushes.json["medians_chat"].
    for lk, ev in OLD_NORM.items():
        got_v = float(p99["medians_chat"][lk])
        if got_v != float(ev):
            raise ValueError(f"OLD_NORM[{lk}] drift: {got_v} != {ev}")

    # Verify PRIMARY_PAIRS against pinned M7 artifact.
    m7_pairs = m7["primary_pairs"]
    if len(m7_pairs) != len(PRIMARY_PAIRS):
        raise ValueError("M7 primary_pairs length mismatch")
    for p_our, p_m7 in zip(PRIMARY_PAIRS, m7_pairs):
        if (
            p_our["pair_id"] != p_m7["pair"]
            or p_our["layer"] != int(p_m7["layer"])
            or p_our["slot_unit"] != p_m7["slot"]
            or p_our["concept_unit"] != p_m7["concept"]
            or p_our["same_persona_kind"] != bool(p_m7["same_pk"])
        ):
            raise ValueError(f"M7 primary pair mismatch: {p_our} vs {p_m7}")

    t213_by_id = {u["id"]: u for u in t213["units"]}
    pk_map = s92["persona_kind"]
    pair_lookup: dict[str, tuple[str, str]] = {}
    for p in PRIMARY_PAIRS:
        pair_lookup[p["slot_unit"]] = (p["pair_id"], "primary_slot")
        pair_lookup[p["concept_unit"]] = (p["pair_id"], "primary_concept")
    pair_lookup["p92-05"] = ("", "extra_part_b_bridge")
    pair_lookup["p92-43"] = ("", "extra_l6_reader_agreed")

    units_meta: dict[str, dict[str, Any]] = {}
    unit_first_tokens: dict[int, str] = {}
    for uid in ALL_UNITS_26:
        src = t213_by_id[uid]
        pid_str, role_str = pair_lookup[uid]
        c_start = int(src["content_start"])
        t_ids = [int(x) for x in src["target_ids"]]
        c_ids = t_ids[c_start:]
        t1_id = int(src["first_content_token_id"])
        if c_ids[0] != t1_id or len(c_ids) != int(src["n_content_tokens"]):
            raise ValueError(f"targets213 content_ids consistency failed on {uid}")
        pk = str(pk_map[uid])
        is_full_word = bool(src["is_full_word_first_token"])
        units_meta[uid] = {
            "id": uid,
            "layer": int(src["layer"]),
            "neuron": int(src["neuron"]),
            "kind_haiku": str(src["kind_haiku"]),
            "kind_ds": str(src["kind_ds"]),
            "persona_kind": pk,
            "role_in_study": role_str,
            "pair_id": pid_str or None,
            "lead_class": str(src["lead_class"]),
            "np_text": str(src["np_text"]),
            "target_ids": t_ids,
            "content_start": c_start,
            "content_ids": c_ids,
            "first_content_token_id": t1_id,
            "first_content_token_str": str(src["first_content_token_str"]),
            "is_full_word_first_token": is_full_word,
            "n_content_tokens": int(src["n_content_tokens"]),
            "clean_token_excluded": (not is_full_word) or (uid in FROZEN_MODIFIER_FIRST_UNITS),
        }
        unit_first_tokens[t1_id] = str(src["first_content_token_str"])

    # Verify same_persona_kind & layer matching on every primary pair.
    for p in PRIMARY_PAIRS:
        su = units_meta[p["slot_unit"]]
        cu = units_meta[p["concept_unit"]]
        if su["layer"] != p["layer"] or cu["layer"] != p["layer"]:
            raise ValueError(f"Layer mismatch in pair {p['pair_id']}")
        if su["kind_haiku"] != "slot" or cu["kind_haiku"] != "concept":
            raise ValueError(f"Class mismatch in pair {p['pair_id']}")
        if (su["persona_kind"] == cu["persona_kind"]) != p["same_persona_kind"]:
            raise ValueError(f"same_persona_kind mismatch in pair {p['pair_id']}")

    # Build verified 37-token competitor pool (24 distinct unit t1 + 13 neutral nouns).
    if len(unit_first_tokens) != 24:
        raise ValueError(f"Expected 24 distinct unit first tokens, got {len(unit_first_tokens)}")
    pool_entries: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for tid in sorted(unit_first_tokens.keys()):
        pool_entries.append({"token_id": tid, "token_str": unit_first_tokens[tid], "source": "unit_first_content"})
        seen_ids.add(tid)
    for tid, tstr in NEUTRAL_POOL_TOKENS:
        if tid in seen_ids:
            raise ValueError(f"Duplicate pool token ID {tid}")
        pool_entries.append({"token_id": tid, "token_str": tstr, "source": "neutral_noun"})
        seen_ids.add(tid)
    pool_entries.sort(key=lambda x: x["token_id"])
    pool_ids_37 = [e["token_id"] for e in pool_entries]
    if len(pool_ids_37) != 37:
        raise ValueError(f"Expected 37 pool tokens, got {len(pool_ids_37)}")

    # Tokenizer load & verification.
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        def _default_tok(repo: str, commit: str):
            return AutoTokenizer.from_pretrained(repo, revision=commit, local_files_only=True)

        tokenizer_loader = _default_tok
    tok = tokenizer_loader(MODEL_REPO, MODEL_COMMIT)

    # Verify every pool token decodes cleanly and has zero word overlap with C1..C6.
    c_bank_words = set()
    for st in C_BANK_STORIES:
        for f in ("scene", "verb", "obj", "name", "role"):
            for w in st[f].lower().replace(",", "").replace(".", "").split():
                c_bank_words.add(w)
    for e in pool_entries:
        dec = tok.decode([e["token_id"]])
        if dec != e["token_str"]:
            raise ValueError(f"Pool token decode mismatch for {e['token_id']}: {dec!r} != {e['token_str']!r}")
        if e["token_str"].strip().lower() in c_bank_words:
            raise ValueError(f"Pool token {e['token_str']!r} overlaps C1..C6 bank words")

    # Zero-overlap guard against historical manifests (d13fc3dd, 01bd41d9, fb6c81f9) and targets213.
    hist_chunks = [targets213_path.read_text(encoding="utf-8")]
    for commit, rel_p in HIST_COMMITS_MANIFESTS:
        hist_chunks.append(load_git_blob(repo_root, commit, rel_p).decode("utf-8"))
    hist_blob = "\n".join(hist_chunks)
    for st in C_BANK_STORIES:
        for field in ("scene", "obj", "name", "role"):
            if st[field] in hist_blob:
                raise ValueError(f"C-bank overlap detected with historical banks/targets213: {st[field]!r}")

    # Verify all 30 (6 stories * 5 leads) prompt geometries and build per-unit single-pass bindings.
    def _encode_prefix(user_msg: str, asst_prefix: str) -> tuple[list[int], str]:
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

    # Exhaustive 30-combo geometry check across all 5 leads.
    for st in C_BANK_STORIES:
        for lc in ("a", "an", "the", "your", "bare"):
            exp_s, exp_n, exp_r = (32, 32, 35) if lc == "bare" else (33, 33, 36)
            s_ids, _ = _encode_prefix(*build_prompt_strings(st, "SELF", lc))
            n_ids, _ = _encode_prefix(*build_prompt_strings(st, "NAME_BARE", lc))
            r_ids, _ = _encode_prefix(*build_prompt_strings(st, "ROLE_BARE", lc))
            if (len(s_ids), len(n_ids), len(r_ids)) != (exp_s, exp_n, exp_r):
                raise ValueError(f"Geometry length drift on ({st['story']}, lead={lc}): {(len(s_ids), len(n_ids), len(r_ids))}")
            diffs = [i for i in range(len(s_ids)) if s_ids[i] != n_ids[i]]
            if diffs != EXPECTED_SN_DIFF_INDICES:
                raise ValueError(f"SELF vs NAME_BARE diff drift on ({st['story']}, lead={lc}): {diffs}")

    prompts_meta: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    for st in C_BANK_STORIES:
        sid = st["story"]
        for cell in BANK_CELLS:
            pid = f"{sid}_{cell}"
            u_msg, _ = build_prompt_strings(st, cell, "a")
            prompts_meta.append({"prompt_id": pid, "story": sid, "cell": cell, "user_msg": u_msg})
            per_u: dict[str, Any] = {}
            for uid in ALL_UNITS_26:
                um = units_meta[uid]
                u_m, a_pref = build_prompt_strings(st, cell, um["lead_class"])
                prefix_ids, prefix_text = _encode_prefix(u_m, a_pref)
                c_ids = list(um["content_ids"])
                full_ids = prefix_ids + c_ids[:-1]
                t_own = int(um["first_content_token_id"])
                peers = [tid for tid in pool_ids_37 if tid != t_own]
                if len(peers) != 36 or t_own in peers:
                    raise ValueError(f"Peer pool error on {uid}")
                per_u[uid] = {
                    "prefix_ids": prefix_ids,
                    "content_ids": c_ids,
                    "full_input_ids": full_ids,
                    "read_pos": len(prefix_ids) - 1,
                    "n_content_tokens": len(c_ids),
                    "t_own": t_own,
                    "peers": peers,
                    "prefix_text": prefix_text,
                }
            bindings[pid] = per_u

    # Join lens stats & compute exact ov_norm + OLD B1..B4 rmsDLA/srDLA from pinned 48d788ec weights.
    lens_by_head = {h["head"]: h for h in lens["all_256_heads"]}
    if model_loader is None:
        from transformers import AutoModelForCausalLM

        def _default_cpu_model(repo: str, commit: str, _dev: str) -> nn.Module:
            m = AutoModelForCausalLM.from_pretrained(
                repo, revision=commit, torch_dtype=torch.float32, attn_implementation=ATTN_IMPL, local_files_only=True
            )
            m.eval()
            return m

        model_loader = _default_cpu_model
    cpu_model = model_loader(MODEL_REPO, MODEL_COMMIT, "cpu")
    head_weight_stats = _compute_head_weight_and_old_dla_stats(tok, cpu_model, pool_ids_37)

    heads_meta: list[dict[str, Any]] = []
    for h in HEAD_BATTERY:
        hn = h["name"]
        lh = lens_by_head[hn]
        ws = head_weight_stats[hn]
        entry = dict(h)
        entry["lens_sr_z"] = float(lh["self_minus_role_z"])
        entry["lens_copy_r"] = float(lh["copy_score_ovR"])
        entry["ov_norm"] = float(ws["ov_norm"])
        entry["old_rms_dla"] = float(ws["old_rms_dla"])
        entry["old_sr_dla"] = float(ws["old_sr_dla"])
        heads_meta.append(entry)

    # Verify uniform neutral cutoff |S-R| <= 1.6 on all neutral_ctrl heads.
    for hm in heads_meta:
        if hm["role"] == "neutral_ctrl" and abs(hm["lens_sr_z"]) > 1.6:
            raise ValueError(f"Neutral control {hm['name']} violates |S-R|<=1.6: {hm['lens_sr_z']}")

    # Exact schedules & forward denominators.
    all_prompt_ids = [p["prompt_id"] for p in prompts_meta]
    unpatched_sched = [
        {"unit": uid, "prompt_id": pid, "story": pid.split("_")[0], "cell": pid.split("_", 1)[1], "dose_mag": NOMINAL_DOSE_MAG}
        for uid in ALL_UNITS_26
        for pid in all_prompt_ids
    ]
    blocked_sched_full = [
        {"unit": uid, "prompt_id": pid, "story": pid.split("_")[0], "cell": pid.split("_", 1)[1], "head_name": hn}
        for uid in ALL_UNITS_26
        for pid in all_prompt_ids
        for hn in HEAD_NAMES_26
    ]
    smoke_prompts_c1 = [f"{SMOKE_BLOCKED_STORY}_{c}" for c in BANK_CELLS]
    blocked_sched_smoke = [
        {"unit": uid, "prompt_id": pid, "story": SMOKE_BLOCKED_STORY, "cell": pid.split("_", 1)[1], "head_name": hn}
        for uid in SMOKE_BLOCKED_UNITS
        for pid in smoke_prompts_c1
        for hn in HEAD_NAMES_26
    ]
    sham_sched = [
        {
            "sham_id": f"sham_{i:02d}_{hn}",
            "unit": ACTUATION_AND_SHAM_UNIT,
            "prompt_id": ACTUATION_AND_SHAM_PROMPT,
            "head_name": hn,
        }
        for i, hn in enumerate(HEAD_NAMES_26)
    ]
    actuation_sched = [
        {
            "actuation_id": f"act_{i:02d}_{hn}",
            "unit": ACTUATION_AND_SHAM_UNIT,
            "prompt_id": ACTUATION_AND_SHAM_PROMPT,
            "head_name": hn,
            "seed": SYNTH_ACTUATION_SEED + i,
        }
        for i, hn in enumerate(HEAD_NAMES_26)
    ]
    causal_equiv_sched = [
        {
            "check_id": f"ce_{i:02d}_{uid}",
            "unit": uid,
            "prompt_id": CAUSAL_EQUIV_CHECK_PROMPT,
            "n_content_tokens": units_meta[uid]["n_content_tokens"],
        }
        for i, uid in enumerate(CAUSAL_EQUIV_CHECK_UNITS)
    ]

    denoms_full = {
        "n_unpatched_rows": len(unpatched_sched),
        "n_blocked_rows": len(blocked_sched_full),
        "n_sham_rows": len(sham_sched),
        "n_actuation_rows": len(actuation_sched),
        "n_causal_equiv_rows": len(causal_equiv_sched),
        "n_b11_forwards": len(unpatched_sched) + len(blocked_sched_full) + len(sham_sched) + len(causal_equiv_sched),
        "n_b1_forwards": len(actuation_sched),
        "n_total_forwards": len(unpatched_sched) + len(blocked_sched_full) + len(sham_sched) + len(causal_equiv_sched) + len(actuation_sched),
    }
    denoms_smoke = {
        "n_unpatched_rows": len(unpatched_sched),
        "n_blocked_rows": len(blocked_sched_smoke),
        "n_sham_rows": len(sham_sched),
        "n_actuation_rows": len(actuation_sched),
        "n_causal_equiv_rows": len(causal_equiv_sched),
        "n_b11_forwards": len(unpatched_sched) + len(blocked_sched_smoke) + len(sham_sched) + len(causal_equiv_sched),
        "n_b1_forwards": len(actuation_sched),
        "n_total_forwards": len(unpatched_sched) + len(blocked_sched_smoke) + len(sham_sched) + len(causal_equiv_sched) + len(actuation_sched),
    }
    assert denoms_full == {
        "n_unpatched_rows": 468,
        "n_blocked_rows": 12168,
        "n_sham_rows": 26,
        "n_actuation_rows": 26,
        "n_causal_equiv_rows": 3,
        "n_b11_forwards": 12665,
        "n_b1_forwards": 26,
        "n_total_forwards": 12691,
    }, f"Unexpected full denominators: {denoms_full}"
    assert denoms_smoke == {
        "n_unpatched_rows": 468,
        "n_blocked_rows": 156,
        "n_sham_rows": 26,
        "n_actuation_rows": 26,
        "n_causal_equiv_rows": 3,
        "n_b11_forwards": 653,
        "n_b1_forwards": 26,
        "n_total_forwards": 679,
    }, f"Unexpected smoke denominators: {denoms_smoke}"

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "design_file": DESIGN_REL_PATH,
            "design_sha256": DESIGN_SHA256,
            "targets213_sha256": TARGETS213_SHA256,
            "m7_transport_sha256": M7_TRANSPORT_SHA256,
            "score92_sha256": SCORE92_SHA256,
            "part99_pushes_sha256": PART99_PUSHES_SHA256,
            "lens_discovery_sha256": LENS_DISCOVERY_SHA256,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "nominal_dose_mag": NOMINAL_DOSE_MAG,
            "n_null_draws": N_NULL_DRAWS,
            "batch_size": BATCH_SIZE,
            "null_seed_base": NULL_SEED_BASE,
            "null_domain_tag": NULL_DOMAIN_TAG,
            "synth_actuation_seed": SYNTH_ACTUATION_SEED,
            "operating_norm_name": "OLD_NORM",
            "operating_norm_multipliers": dict(OLD_NORM),
            "primary_pairs": PRIMARY_PAIRS,
            "extra_units": list(EXTRA_UNITS),
            "frozen_modifier_first_units": sorted(FROZEN_MODIFIER_FIRST_UNITS),
            "smoke_blocked_units": list(SMOKE_BLOCKED_UNITS),
            "smoke_blocked_story": SMOKE_BLOCKED_STORY,
            "denominators": {"full": denoms_full, "smoke": denoms_smoke},
        },
        "units": units_meta,
        "competitor_pool_37": pool_entries,
        "heads": heads_meta,
        "stories": C_BANK_STORIES,
        "prompts": prompts_meta,
        "bindings": bindings,
        "schedule_full": {
            "unpatched": unpatched_sched,
            "blocked": blocked_sched_full,
            "sham": sham_sched,
            "actuation": actuation_sched,
            "causal_equiv": causal_equiv_sched,
        },
        "schedule_smoke": {
            "unpatched": unpatched_sched,
            "blocked": blocked_sched_smoke,
            "sham": sham_sched,
            "actuation": actuation_sched,
            "causal_equiv": causal_equiv_sched,
        },
    }


# ---------------------------------------------------------------------------
# Pre-Load Single-Read 6-Pin Admission & Timing Validation
# ---------------------------------------------------------------------------

def _admit_frozen_package_with_bytes(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    targets213_path: Path,
    m7_transport_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
    expected_targets213_sha: str = TARGETS213_SHA256,
    expected_m7_transport_sha: str = M7_TRANSPORT_SHA256,
) -> tuple[dict[str, Any], dict[str, str], dict[str, bytes]]:
    """Single-read external 6-pin binding BEFORE any tokenizer or model load."""
    bytes_by_label: dict[str, bytes] = {}
    for label, p, exp in [
        ("design", design_path, expected_design_sha),
        ("manifest", manifest_path, expected_manifest_sha),
        ("runner", runner_path, expected_runner_sha),
        ("scorer", scorer_path, expected_scorer_sha),
        ("targets213", targets213_path, expected_targets213_sha),
        ("m7_transport", m7_transport_path, expected_m7_transport_sha),
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
        raise ValueError(f"Design pin {expected_design_sha} != frozen v3.1 {DESIGN_SHA256}")
    if expected_targets213_sha != TARGETS213_SHA256:
        raise ValueError(f"targets213 pin {expected_targets213_sha} != pinned {TARGETS213_SHA256}")
    if expected_m7_transport_sha != M7_TRANSPORT_SHA256:
        raise ValueError(f"m7_transport pin {expected_m7_transport_sha} != pinned {M7_TRANSPORT_SHA256}")

    manifest = json.loads(bytes_by_label["manifest"].decode("utf-8"))
    mmeta = manifest.get("meta", {})
    if mmeta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Manifest schema_version mismatch: {mmeta.get('schema_version')!r}")
    if mmeta.get("design_sha256") != expected_design_sha:
        raise ValueError("Manifest embedded design_sha256 != external design pin")
    if mmeta.get("targets213_sha256") != expected_targets213_sha:
        raise ValueError("Manifest embedded targets213_sha256 != external targets213 pin")
    if mmeta.get("m7_transport_sha256") != expected_m7_transport_sha:
        raise ValueError("Manifest embedded m7_transport_sha256 != external m7_transport pin")
    if mmeta.get("model_repo") != MODEL_REPO or mmeta.get("model_commit") != MODEL_COMMIT:
        raise ValueError("Manifest model repo/commit mismatch")
    if mmeta.get("operating_norm_name") != "OLD_NORM" or mmeta.get("operating_norm_multipliers") != OLD_NORM:
        raise ValueError("Manifest OLD_NORM multipliers mismatch")

    pins = {
        "design_sha256": expected_design_sha,
        "manifest_sha256": expected_manifest_sha,
        "runner_sha256": expected_runner_sha,
        "scorer_sha256": expected_scorer_sha,
        "targets213_sha256": expected_targets213_sha,
        "m7_transport_sha256": expected_m7_transport_sha,
    }
    return manifest, pins, bytes_by_label


def _load_pinned_scorer_module(scorer_path: Path, expected_scorer_sha: str, scorer_bytes: bytes | None = None):
    raw_b = scorer_path.read_bytes() if scorer_bytes is None else scorer_bytes
    got = sha256_bytes(raw_b)
    if got != expected_scorer_sha:
        raise ValueError(f"Scorer SHA-256 mismatch before load: got {got}, expected {expected_scorer_sha}")
    prev_mod = sys.modules.get("run_concept_vs_slot")
    sys.modules["run_concept_vs_slot"] = sys.modules[__name__]
    try:
        mod = types.ModuleType("_pinned_score_concept_vs_slot")
        mod.__file__ = str(scorer_path.resolve())
        code = compile(raw_b, str(scorer_path.resolve()), "exec")
        exec(code, mod.__dict__)
        return mod
    finally:
        if prev_mod is not None:
            sys.modules["run_concept_vs_slot"] = prev_mod


def validate_timing_breakdown(
    timing: Any,
    expected_b11: int,
    expected_b1: int,
    full_b11: int = 12665,
    full_b1: int = 26,
) -> None:
    if not isinstance(timing, dict):
        raise ValueError("Missing or null timing breakdown")
    req_keys = (
        "model_load_sec",
        "b11_elapsed_sec",
        "b1_elapsed_sec",
        "non_fwd_runner_sec",
        "total_runner_sec",
        "n_b11_done",
        "n_b1_done",
        "ms_per_b11",
        "ms_per_b1",
        "projected_full_b11_sec",
        "projected_full_b1_sec",
        "projected_full_fwd_sec",
    )
    for k in req_keys:
        v = timing.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise ValueError(f"Timing field '{k}' missing or non-finite: {v!r}")
    if int(timing["n_b11_done"]) != int(expected_b11):
        raise ValueError(f"Timing n_b11_done={timing['n_b11_done']} != expected {expected_b11}")
    if int(timing["n_b1_done"]) != int(expected_b1):
        raise ValueError(f"Timing n_b1_done={timing['n_b1_done']} != expected {expected_b1}")
    if float(timing["model_load_sec"]) < 0.0 or float(timing["non_fwd_runner_sec"]) < 0.0:
        raise ValueError("Negative timing field")
    if float(timing["b11_elapsed_sec"]) <= 0.0 or float(timing["b1_elapsed_sec"]) <= 0.0:
        raise ValueError("Non-positive forward elapsed seconds")
    if float(timing["total_runner_sec"]) + 1e-9 < float(timing["b11_elapsed_sec"]) + float(timing["b1_elapsed_sec"]):
        raise ValueError("total_runner_sec < b11_elapsed_sec + b1_elapsed_sec")
    exp_ms11 = 1000.0 * float(timing["b11_elapsed_sec"]) / float(expected_b11)
    exp_ms1 = 1000.0 * float(timing["b1_elapsed_sec"]) / float(expected_b1)
    if abs(float(timing["ms_per_b11"]) - exp_ms11) > 1e-3:
        raise ValueError("Timing ms_per_b11 inconsistent")
    if abs(float(timing["ms_per_b1"]) - exp_ms1) > 1e-3:
        raise ValueError("Timing ms_per_b1 inconsistent")
    exp_proj_b11 = (float(timing["ms_per_b11"]) / 1000.0) * float(full_b11)
    exp_proj_b1 = (float(timing["ms_per_b1"]) / 1000.0) * float(full_b1)
    if float(timing["projected_full_b11_sec"]) <= 0.0 or abs(float(timing["projected_full_b11_sec"]) - exp_proj_b11) > 1e-3:
        raise ValueError("Timing projected_full_b11_sec inconsistent")
    if float(timing["projected_full_b1_sec"]) <= 0.0 or abs(float(timing["projected_full_b1_sec"]) - exp_proj_b1) > 1e-3:
        raise ValueError("Timing projected_full_b1_sec inconsistent")
    if float(timing["projected_full_fwd_sec"]) <= 0.0 or abs(
        float(timing["projected_full_fwd_sec"]) - (float(timing["projected_full_b11_sec"]) + float(timing["projected_full_b1_sec"]))
    ) > 1e-3:
        raise ValueError("Timing projected_full_fwd_sec inconsistent")


def admit_execution_request(
    mode: str | None,
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    targets213_path: Path,
    m7_transport_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
    expected_targets213_sha: str = TARGETS213_SHA256,
    expected_m7_transport_sha: str = M7_TRANSPORT_SHA256,
    allow_full_after_smoke: bool = False,
    verified_smoke_receipt: Path | None = None,
    verified_smoke_sha256: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    """Enforce explicit --mode {smoke,full}, single-read 6-pin binding, and pre-load smoke gate."""
    if mode not in ("smoke", "full"):
        raise ValueError(f"Explicit --mode {{smoke,full}} required (got {mode!r}); implicit full refused.")
    if runner_path.resolve() != Path(__file__).resolve():
        raise ValueError(f"Runner path {runner_path.resolve()} != executing module {Path(__file__).resolve()}")

    manifest, pins, pkg_bytes = _admit_frozen_package_with_bytes(
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        targets213_path=targets213_path,
        m7_transport_path=m7_transport_path,
        expected_design_sha=expected_design_sha,
        expected_manifest_sha=expected_manifest_sha,
        expected_runner_sha=expected_runner_sha,
        expected_scorer_sha=expected_scorer_sha,
        expected_targets213_sha=expected_targets213_sha,
        expected_m7_transport_sha=expected_m7_transport_sha,
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
            allow_fixture=False,
        )
        if not sm_verdict.get("smoke_gate_passed", False):
            raise ValueError(f"Full execution refused: pinned smoke verifier failed ({sm_verdict})")
    return manifest, pins


# ---------------------------------------------------------------------------
# Orthogonal Null Generator, Single-Pass 5-Channel Readouts & Hook Utilities
# ---------------------------------------------------------------------------

def unit_null_seed(unit_id: str, prompt_id: str, dose_mag: float) -> int:
    key = f"{NULL_SEED_BASE}|{NULL_DOMAIN_TAG}|{unit_id}|{prompt_id}|{dose_mag:.4f}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def make_wdown_orthogonal_nulls(
    w_down_cpu: torch.Tensor,
    target_neuron: int,
    hat_w_cpu: torch.Tensor,
    n_draws: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    v_hat = hat_w_cpu.float() / float(hat_w_cpu.float().norm().item())
    _d_model, d_ff = w_down_cpu.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coeffs = torch.randn((d_ff, n_draws), generator=gen, dtype=torch.float32)
    coeffs[target_neuron, :] = 0.0
    raw_wd = (w_down_cpu.float() @ coeffs).T
    raw_wd = raw_wd - (raw_wd @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    wdown_unit = raw_wd / torch.linalg.vector_norm(raw_wd, dim=-1, keepdim=True).clamp_min(1e-12)
    max_cos = float((wdown_unit @ v_hat).abs().max().item())
    if max_cos > TOL_ORTH:
        raise RuntimeError(f"wdown_orth orthogonality failed: max_cos={max_cos}")
    return wdown_unit, max_cos


def build_b11_deltas(
    hat_w_cpu: torch.Tensor,
    wd_unit_cpu: torch.Tensor,
    dose_mag: float,
    norm_multiplier: float,
    device: str,
) -> torch.Tensor:
    scale = float(dose_mag) * float(norm_multiplier)
    d_model = hat_w_cpu.shape[0]
    deltas = torch.zeros((BATCH_SIZE, d_model), dtype=torch.float32, device="cpu")
    deltas[1] = scale * hat_w_cpu
    deltas[2] = -scale * hat_w_cpu
    deltas[3 : 3 + N_NULL_DRAWS] = scale * wd_unit_cpu
    deltas[3 + N_NULL_DRAWS : 3 + 2 * N_NULL_DRAWS] = -scale * wd_unit_cpu
    if float(deltas[0].abs().max().item()) != 0.0:
        raise ValueError("Row 0 of B=11 must be exact zero")
    if float((deltas[1] + deltas[2]).abs().max().item()) > 1e-5:
        raise ValueError("Target +d/-d rows are not exact negatives")
    return deltas.to(device=device, dtype=torch.float32)


def read_five_channels(
    full_logits: torch.Tensor,
    read_pos: int,
    t_own: int,
    peers: list[int],
    content_ids: list[int],
) -> dict[str, list[float]]:
    """Extract RawLogit, PeerMed, SpecLogit, TargetLogProb, and PhraseLogProbMean in one causal pass."""
    k = len(content_ids)
    if content_ids[0] != t_own:
        raise ValueError(f"content_ids[0]={content_ids[0]} != t_own={t_own}")
    logits_rp = full_logits[:, read_pos, :].detach().to(torch.float32)
    if not torch.isfinite(logits_rp).all():
        raise FloatingPointError("Non-finite logits at read_pos")

    raw = [float(x) for x in logits_rp[:, t_own].cpu().tolist()]
    peer_mat = [[float(v) for v in row] for row in logits_rp[:, peers].cpu().tolist()]
    peer_med = [float(statistics.median(row)) for row in peer_mat]
    spec = [r - pm for r, pm in zip(raw, peer_med)]

    lsm_0 = F.log_softmax(logits_rp, dim=-1)
    logp = [float(x) for x in lsm_0[:, t_own].cpu().tolist()]

    phrase_sum = list(logp)
    for j in range(1, k):
        lj = full_logits[:, read_pos + j, :].detach().to(torch.float32)
        if not torch.isfinite(lj).all():
            raise FloatingPointError(f"Non-finite logits at read_pos+{j}")
        lsm_j = F.log_softmax(lj, dim=-1)
        tok_j = int(content_ids[j])
        vals_j = [float(x) for x in lsm_j[:, tok_j].cpu().tolist()]
        phrase_sum = [a + b for a, b in zip(phrase_sum, vals_j)]
    phrase_mean = [x / float(k) for x in phrase_sum]

    out = {
        "raw": raw,
        "peer_med": peer_med,
        "spec": spec,
        "logp": logp,
        "phrase_mean": phrase_mean,
        "phrase_sum": phrase_sum,
    }
    for cname, vals in out.items():
        if not all(math.isfinite(x) for x in vals):
            raise FloatingPointError(f"Non-finite readout in {cname}")
    return out


def summarize_unpatched_b11(ch_vals: dict[str, list[float]]) -> tuple[dict[str, Any], float]:
    """Summarize row-0 baselines, ±0.8 transport, and 4-draw w_down null transport (ZERO sham keys)."""
    res: dict[str, Any] = {}
    for ch in CHANNELS_5:
        v = ch_vals[ch]
        b0 = v[0]
        res[f"base_{ch}"] = b0
        res[f"t_pos_{ch}"] = v[1] - b0
        res[f"t_neg_{ch}"] = v[2] - b0
        res[f"wdown_pos_{ch}"] = [v[3 + j] - b0 for j in range(N_NULL_DRAWS)]
        res[f"wdown_neg_{ch}"] = [v[3 + N_NULL_DRAWS + j] - b0 for j in range(N_NULL_DRAWS)]
    res["base_phrase_logp"] = ch_vals["phrase_sum"][0]

    errs = [
        abs(res["base_spec"] - (res["base_raw"] - res["base_peer_med"])),
        abs(res["t_pos_spec"] - (res["t_pos_raw"] - res["t_pos_peer_med"])),
        abs(res["t_neg_spec"] - (res["t_neg_raw"] - res["t_neg_peer_med"])),
        *[abs(res["wdown_pos_spec"][j] - (res["wdown_pos_raw"][j] - res["wdown_pos_peer_med"][j])) for j in range(N_NULL_DRAWS)],
        *[abs(res["wdown_neg_spec"][j] - (res["wdown_neg_raw"][j] - res["wdown_neg_peer_med"][j])) for j in range(N_NULL_DRAWS)],
    ]
    max_err = max(errs)
    if max_err > TOL_DECOMP:
        raise AssertionError(f"Unpatched linear decomposition violated: {max_err}")
    return res, max_err


def summarize_blocked_b11(
    unp: dict[str, list[float]],
    pat: dict[str, list[float]],
) -> tuple[dict[str, dict[str, Any]], float, dict[str, float]]:
    """Compact lossless B11 drop summary (`unpatched - blocked`) across all 5 channels."""
    out: dict[str, dict[str, Any]] = {}
    row0_sham: dict[str, float] = {}
    for ch in CHANNELS_5:
        u_v, p_v = unp[ch], pat[ch]
        d = [a - b for a, b in zip(u_v, p_v)]
        if not all(math.isfinite(x) for x in d):
            raise FloatingPointError(f"Non-finite blocked drop in {ch}")
        out[ch] = {
            "d0": d[0],
            "d_pos": d[1],
            "d_neg": d[2],
            "wd_pos": d[3 : 3 + N_NULL_DRAWS],
            "wd_neg": d[3 + N_NULL_DRAWS : 3 + 2 * N_NULL_DRAWS],
        }
        row0_sham[ch] = abs(d[0])
    # Exact Spec == Raw - PeerMed identity on every single row of B=11.
    sp, rw, pm = out["spec"], out["raw"], out["peer_med"]
    errs = [
        abs(sp["d0"] - (rw["d0"] - pm["d0"])),
        abs(sp["d_pos"] - (rw["d_pos"] - pm["d_pos"])),
        abs(sp["d_neg"] - (rw["d_neg"] - pm["d_neg"])),
        *[abs(sp["wd_pos"][j] - (rw["wd_pos"][j] - pm["wd_pos"][j])) for j in range(N_NULL_DRAWS)],
        *[abs(sp["wd_neg"][j] - (rw["wd_neg"][j] - pm["wd_neg"][j])) for j in range(N_NULL_DRAWS)],
    ]
    max_err = max(errs)
    if max_err > TOL_DECOMP:
        raise AssertionError(f"Blocked linear decomposition violated: {max_err}")
    return out, max_err, row0_sham


def compute_sibling_max_abs_leakage(
    z_patched: torch.Tensor,
    z_base: torch.Tensor,
    head_idx: int,
    n_heads: int,
) -> float:
    d_model = z_patched.shape[-1]
    head_dim = d_model // n_heads
    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    diff = (z_patched.detach().to(torch.float32) - z_base.detach().to(torch.float32)).abs()
    left = float(diff[..., :s].max().item()) if s > 0 else 0.0
    right = float(diff[..., e:].max().item()) if e < d_model else 0.0
    return max(left, right)


def make_orthogonal_synthetic_donor(nat_slice: torch.Tensor, seed: int) -> torch.Tensor:
    nat = nat_slice.detach().to(device="cpu", dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(nat.shape, generator=gen, dtype=torch.float32)
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
# Main Execution Engine (Smoke: 650 B11 + 29 B=1 | Full: 12,662 B11 + 29 B=1)
# ---------------------------------------------------------------------------

def execute_concept_vs_slot(
    manifest: dict[str, Any],
    pins: dict[str, str],
    smoke_mode: bool,
    device: str,
    model_loader: Callable[[str, str, str], nn.Module] = _default_model_loader,
    out_path: Path | None = None,
    execution_provenance: str | None = None,
) -> dict[str, Any]:
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

    sched = manifest["schedule_smoke" if smoke_mode else "schedule_full"]
    op_norms = manifest["meta"]["operating_norm_multipliers"]
    units_meta = manifest["units"]
    head_layers_all = sorted({h["layer"] for h in HEAD_BATTERY})

    # Extract fixed unit write vectors `hat_w` once for all 26 units.
    unit_cache: dict[str, dict[str, Any]] = {}
    unit_vectors_meta: dict[str, dict[str, Any]] = {}
    for uid in ALL_UNITS_26:
        um = units_meta[uid]
        li, ni = int(um["layer"]), int(um["neuron"])
        w_down_cpu = layers_mod[li].mlp.down_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        v_raw = w_down_cpu[:, ni]
        w_norm = float(v_raw.norm().item())
        hat_w_cpu = v_raw / max(w_norm, 1e-12)
        unit_cache[uid] = {"layer": li, "neuron": ni, "w_down_cpu": w_down_cpu, "hat_w_cpu": hat_w_cpu}
        unit_vectors_meta[uid] = {
            "layer": li,
            "neuron": ni,
            "w_down_norm": w_norm,
            "operating_norm_multiplier": float(op_norms[str(li)]),
        }

    # Index blocked & sham schedules by (unit, prompt_id) so each unpatched forward captures o_proj once!
    blocked_by_up: dict[tuple[str, str], list[str]] = {}
    for b in sched["blocked"]:
        blocked_by_up.setdefault((b["unit"], b["prompt_id"]), []).append(b["head_name"])
    sham_by_up: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sh in sched["sham"]:
        sham_by_up.setdefault((sh["unit"], sh["prompt_id"]), []).append(sh)
    ce_by_up: dict[tuple[str, str], dict[str, Any]] = {
        (ce["unit"], ce["prompt_id"]): ce for ce in sched["causal_equiv"]
    }

    timers = {"b11_sec": 0.0, "b11_fwds": 0, "b1_sec": 0.0, "b1_fwds": 0}
    unpatched_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    sham_rows: list[dict[str, Any]] = []
    actuation_rows: list[dict[str, Any]] = []
    causal_equiv_rows: list[dict[str, Any]] = []

    max_abs_sham = {ch: 0.0 for ch in CHANNELS_5}
    max_abs_decomp_err = 0.0
    max_sibling_leakage_science = 0.0
    max_sibling_leakage_actuation = 0.0
    max_abs_wdown_orth_cos = 0.0
    max_causal_equiv_err = 0.0
    min_blocked_pos_drop = float("inf")
    act_ref_logits_pos: torch.Tensor | None = None

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

    def _make_o_proj_z_all_pre_hook(
        head_idx: int,
        donor_z: torch.Tensor | None,
        is_sham: bool,
        capture_after: list[torch.Tensor] | None = None,
    ):
        s, e = head_idx * head_dim, (head_idx + 1) * head_dim

        def _pre(_mod, args):
            x = args[0]
            xc = x.clone()
            if is_sham:
                xc[:, :, s:e] = x[:, :, s:e]
            else:
                xc[:, :, s:e] = donor_z.to(device=x.device, dtype=x.dtype)
            if capture_after is not None:
                capture_after.append(xc.detach().clone())
            return (xc,) + args[1:]

        return _pre

    def _run_b11_full_logits(inp: torch.Tensor, handles: list[Any]) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                logits = model(input_ids=inp, use_cache=False).logits
        finally:
            for h in handles:
                h.remove()
        timers["b11_sec"] += time.perf_counter() - t0
        timers["b11_fwds"] += 1
        return logits

    def _run_b1_full_logits(inp: torch.Tensor, handles: list[Any]) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                logits = model(input_ids=inp, use_cache=False).logits
        finally:
            for h in handles:
                h.remove()
        timers["b1_sec"] += time.perf_counter() - t0
        timers["b1_fwds"] += 1
        return logits

    # =========================================================================
    # PART 1: Unpatched B11 + z_all Blocked B11 + Dedicated Full-B11 Shams +
    #         B=1 Prefix-Only Causal Equivalence Checks
    # =========================================================================
    for idx_u, entry in enumerate(sched["unpatched"]):
        uid = entry["unit"]
        pid = entry["prompt_id"]
        story = entry["story"]
        cell = entry["cell"]
        dose_mag = float(entry["dose_mag"])

        ub = unit_cache[uid]
        u_layer = ub["layer"]
        n_idx = ub["neuron"]
        m_L = float(op_norms[str(u_layer)])

        binding = manifest["bindings"][pid][uid]
        full_ids = list(binding["full_input_ids"])
        prefix_ids = list(binding["prefix_ids"])
        content_ids = list(binding["content_ids"])
        read_pos = int(binding["read_pos"])
        t_own = int(binding["t_own"])
        peers = list(binding["peers"])

        seed_u = unit_null_seed(uid, pid, dose_mag)
        wd_u, orth_cos = make_wdown_orthogonal_nulls(
            w_down_cpu=ub["w_down_cpu"],
            target_neuron=n_idx,
            hat_w_cpu=ub["hat_w_cpu"],
            n_draws=N_NULL_DRAWS,
            seed=seed_u,
        )
        max_abs_wdown_orth_cos = max(max_abs_wdown_orth_cos, orth_cos)
        deltas = build_b11_deltas(
            hat_w_cpu=ub["hat_w_cpu"],
            wd_unit_cpu=wd_u,
            dose_mag=dose_mag,
            norm_multiplier=m_L,
            device=device,
        )
        inp = torch.tensor([full_ids] * BATCH_SIZE, dtype=torch.long, device=device)

        heads_for_up = blocked_by_up.get((uid, pid), [])
        shams_for_up = sham_by_up.get((uid, pid), [])
        need_o_cap = bool(heads_for_up)
        cap_o_in: dict[int, torch.Tensor] = {}
        read_pos_norm_box: list[float] = []

        handles = [layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(deltas, read_pos, read_pos_norm_box))]
        if need_o_cap:
            for hl in head_layers_all:
                assert_strictly_downstream(u_layer, hl)
                handles.append(
                    layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(
                        lambda _m, a, _hl=hl: (cap_o_in.__setitem__(_hl, a[0].detach().clone()), None)[1]
                    )
                )

        full_logits_unp = _run_b11_full_logits(inp, handles)
        if uid == ACTUATION_AND_SHAM_UNIT and pid == ACTUATION_AND_SHAM_PROMPT:
            act_ref_logits_pos = full_logits_unp[1, read_pos, :].detach().to(device="cpu", dtype=torch.float32)
        unp_ch = read_five_channels(full_logits_unp, read_pos, t_own, peers, content_ids)
        unp_sum, decomp_err = summarize_unpatched_b11(unp_ch)
        max_abs_decomp_err = max(max_abs_decomp_err, decomp_err)

        inj_norm = dose_mag * m_L
        rp_norm = float(read_pos_norm_box[0])
        unpatched_rows.append({
            "unit": uid,
            "unit_layer": u_layer,
            "prompt_id": pid,
            "story": story,
            "cell": cell,
            "n_content_tokens": len(content_ids),
            "dose_mag": dose_mag,
            "norm_multiplier": m_L,
            "injected_norm": inj_norm,
            "read_pos_resid_norm": rp_norm,
            "norm_ratio": inj_norm / max(rp_norm, 1e-12),
            "null_seed": seed_u,
            "max_abs_wdown_orth_cos": orth_cos,
            "max_abs_decomp_err": decomp_err,
            "transport": unp_sum,
        })

        # CS.0 Clause 6 (F7): same-batch-shape (B=11) prefix-only causal equivalence check on multi-token units.
        ce_spec = ce_by_up.get((uid, pid))
        if ce_spec is not None:
            inp_pref = torch.tensor([prefix_ids] * BATCH_SIZE, dtype=torch.long, device=device)
            logits_pref = _run_b11_full_logits(inp_pref, [])
            full_rp0 = full_logits_unp[0, read_pos, :].detach().to(torch.float32)
            pref_rp0 = logits_pref[0, read_pos, :].detach().to(torch.float32)
            ce_err = float((full_rp0 - pref_rp0).abs().max().item())
            max_causal_equiv_err = max(max_causal_equiv_err, ce_err)
            causal_equiv_rows.append({
                "check_id": ce_spec["check_id"],
                "unit": uid,
                "prompt_id": pid,
                "n_content_tokens": len(content_ids),
                "prefix_len": len(prefix_ids),
                "full_len": len(full_ids),
                "batch_size": BATCH_SIZE,
                "max_abs_full_vocab_read_pos_diff": ce_err,
                "passed": bool(ce_err <= TOL_CAUSAL_EQUIV),
            })

        # Scheduled z_all head-blocked B11 forwards.
        for hname in heads_for_up:
            hmeta = HEAD_BY_NAME[hname]
            hl, hi = int(hmeta["layer"]), int(hmeta["head"])
            assert_strictly_downstream(u_layer, hl)
            s, e = hi * head_dim, (hi + 1) * head_dim
            donor_z = cap_o_in[hl][0:1, :, s:e].expand(BATCH_SIZE, -1, -1)
            z_after_box: list[torch.Tensor] = []
            pat_handles = [
                layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(deltas)),
                layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(
                    _make_o_proj_z_all_pre_hook(hi, donor_z, is_sham=False, capture_after=z_after_box)
                ),
            ]
            full_logits_pat = _run_b11_full_logits(inp, pat_handles)
            pat_ch = read_five_channels(full_logits_pat, read_pos, t_own, peers, content_ids)
            blk_sum, b_decomp, r0_sham = summarize_blocked_b11(unp_ch, pat_ch)
            # F1 blocking positive control: refuse completely inert block hook on +0.8 pushed row!
            row_max_pos_drop = max(abs(blk_sum[ch]["d_pos"]) for ch in CHANNELS_5)
            if row_max_pos_drop <= TOL_BLOCK_ACTUATION:
                raise AssertionError(
                    f"F1 blocking positive control failed (inert block hook): ({uid}, {pid}, {hname}) "
                    f"max_abs_pos_drop={row_max_pos_drop} <= {TOL_BLOCK_ACTUATION}"
                )
            min_blocked_pos_drop = min(min_blocked_pos_drop, row_max_pos_drop)
            sib_leak = compute_sibling_max_abs_leakage(z_after_box[0], cap_o_in[hl], hi, n_heads)

            max_abs_decomp_err = max(max_abs_decomp_err, b_decomp)
            max_sibling_leakage_science = max(max_sibling_leakage_science, sib_leak)
            for ch in CHANNELS_5:
                max_abs_sham[ch] = max(max_abs_sham[ch], r0_sham[ch])

            blocked_rows.append({
                "unit": uid,
                "unit_layer": u_layer,
                "prompt_id": pid,
                "story": story,
                "cell": cell,
                "head_name": hname,
                "head_layer": hl,
                "head_idx": hi,
                "n_content_tokens": len(content_ids),
                "sibling_max_abs_leakage": sib_leak,
                "max_abs_decomp_err": b_decomp,
                "raw": blk_sum["raw"],
                "peer_med": blk_sum["peer_med"],
                "spec": blk_sum["spec"],
                "logp": blk_sum["logp"],
                "phrase_mean": blk_sum["phrase_mean"],
            })

        # Scheduled dedicated full-B11 measured same-hook-path shams (all 11 pushed+null rows!).
        for sh_spec in shams_for_up:
            hname = sh_spec["head_name"]
            hmeta = HEAD_BY_NAME[hname]
            hl, hi = int(hmeta["layer"]), int(hmeta["head"])
            assert_strictly_downstream(u_layer, hl)
            shm_handles = [
                layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(deltas)),
                layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(
                    _make_o_proj_z_all_pre_hook(hi, None, is_sham=True)
                ),
            ]
            full_logits_shm = _run_b11_full_logits(inp, shm_handles)
            shm_ch = read_five_channels(full_logits_shm, read_pos, t_own, peers, content_ids)
            sh_errs = {
                ch: max(abs(a - b) for a, b in zip(unp_ch[ch], shm_ch[ch]))
                for ch in CHANNELS_5
            }
            for ch in CHANNELS_5:
                max_abs_sham[ch] = max(max_abs_sham[ch], sh_errs[ch])
            sham_rows.append({
                "sham_id": sh_spec["sham_id"],
                "unit": uid,
                "prompt_id": pid,
                "head_name": hname,
                "head_layer": hl,
                "head_idx": hi,
                "max_abs_sham_raw": sh_errs["raw"],
                "max_abs_sham_peer_med": sh_errs["peer_med"],
                "max_abs_sham_spec": sh_errs["spec"],
                "max_abs_sham_logp": sh_errs["logp"],
                "max_abs_sham_phrase_mean": sh_errs["phrase_mean"],
                "passed": bool(all(v <= TOL_SHAM for v in sh_errs.values())),
            })

        # Progress print every 26 unpatched prompts (~5.5% increments).
        if not smoke_mode and (idx_u + 1) % 36 == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"[PROGRESS] unpatched {idx_u + 1}/{len(sched['unpatched'])}, "
                f"blocked {len(blocked_rows)}/{len(sched['blocked'])}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    # =========================================================================
    # PART 2: 26 Dedicated B=1 Orthogonal Synthetic-Donor Actuation Controls
    # =========================================================================
    for act in sched["actuation"]:
        uid = act["unit"]
        pid = act["prompt_id"]
        hname = act["head_name"]
        seed_a = int(act["seed"])

        ub = unit_cache[uid]
        u_layer = ub["layer"]
        m_L = float(op_norms[str(u_layer)])
        delta_pos = (NOMINAL_DOSE_MAG * m_L * ub["hat_w_cpu"]).unsqueeze(0).to(device=device, dtype=torch.float32)

        binding = manifest["bindings"][pid][uid]
        inp1 = torch.tensor([binding["full_input_ids"]], dtype=torch.long, device=device)
        read_pos_a = int(binding["read_pos"])
        hmeta = HEAD_BY_NAME[hname]
        hl, hi = int(hmeta["layer"]), int(hmeta["head"])
        assert_strictly_downstream(u_layer, hl)
        s, e = hi * head_dim, (hi + 1) * head_dim

        z_before_box: list[torch.Tensor] = []
        z_after_box: list[torch.Tensor] = []

        def _act_pre(_m, args):
            x = args[0]
            z_before_box.append(x.detach().clone())
            synth_z = make_orthogonal_synthetic_donor(x[:, :, s:e], seed_a)
            xc = x.clone()
            xc[:, :, s:e] = synth_z.to(device=x.device, dtype=x.dtype)
            z_after_box.append(xc.detach().clone())
            return (xc,) + args[1:]

        h_list = [
            layers_mod[u_layer].register_forward_hook(_make_layer_inject_hook(delta_pos)),
            layers_mod[hl].self_attn.o_proj.register_forward_pre_hook(_act_pre),
        ]
        act_logits = _run_b1_full_logits(inp1, h_list)
        if act_ref_logits_pos is None:
            raise RuntimeError("Missing unpatched +0.8 reference logits for actuation readout check")
        readout_delta = float((act_logits[0, read_pos_a, :].detach().to(device="cpu", dtype=torch.float32) - act_ref_logits_pos).abs().max().item())
        z_base = z_before_box[0]
        z_synth = z_after_box[0]
        target_delta = float((z_synth[:, :, s:e].to(torch.float32) - z_base[:, :, s:e].to(torch.float32)).abs().max().item())
        sib_leak = compute_sibling_max_abs_leakage(z_synth, z_base, hi, n_heads)
        max_sibling_leakage_actuation = max(max_sibling_leakage_actuation, sib_leak)
        actuation_rows.append({
            "actuation_id": act["actuation_id"],
            "unit": uid,
            "prompt_id": pid,
            "head_name": hname,
            "head_layer": hl,
            "head_idx": hi,
            "seed": seed_a,
            "target_max_abs_delta": target_delta,
            "readout_max_abs_delta": readout_delta,
            "sibling_max_abs_leakage": sib_leak,
            "passed": bool(
                target_delta > MIN_ACTUATION_DELTA
                and readout_delta > MIN_ACTUATION_READOUT_DELTA
                and sib_leak <= TOL_SIBLING
            ),
        })

    total_sec = time.perf_counter() - t_start
    non_fwd_sec = max(0.0, total_sec - timers["b11_sec"] - timers["b1_sec"])
    den_full = manifest["meta"]["denominators"]["full"]
    ms_b11 = 1000.0 * timers["b11_sec"] / float(timers["b11_fwds"])
    ms_b1 = 1000.0 * timers["b1_sec"] / float(timers["b1_fwds"])
    proj_b11 = (ms_b11 / 1000.0) * float(den_full["n_b11_forwards"])
    proj_b1 = (ms_b1 / 1000.0) * float(den_full["n_b1_forwards"])
    timing = {
        "model_load_sec": model_load_sec,
        "b11_elapsed_sec": timers["b11_sec"],
        "b1_elapsed_sec": timers["b1_sec"],
        "non_fwd_runner_sec": non_fwd_sec,
        "total_runner_sec": total_sec,
        "n_b11_done": timers["b11_fwds"],
        "n_b1_done": timers["b1_fwds"],
        "ms_per_b11": ms_b11,
        "ms_per_b1": ms_b1,
        "projected_full_b11_sec": proj_b11,
        "projected_full_b1_sec": proj_b1,
        "projected_full_fwd_sec": proj_b11 + proj_b1,
    }

    dev_name = torch.cuda.get_device_name(0) if is_cuda_dev else str(device)
    receipt = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "concept_vs_slot_smoke_receipt" if smoke_mode else "concept_vs_slot_full_receipt",
            "run_mode": "smoke" if smoke_mode else "full",
            "execution_provenance": resolved_provenance,
            "device_name": dev_name,
            "is_cuda_device": is_cuda_dev,
            "complete": not smoke_mode,
            "model_repo": MODEL_REPO,
            "model_commit": MODEL_COMMIT,
            "dtype": DTYPE_STR,
            "attn_implementation": ATTN_IMPL,
            "operating_norm_name": "OLD_NORM",
            "operating_norm_multipliers": dict(op_norms),
            "pins": dict(pins),
            "n_unpatched_rows": len(unpatched_rows),
            "n_blocked_rows": len(blocked_rows),
            "n_sham_rows": len(sham_rows),
            "n_actuation_rows": len(actuation_rows),
            "n_causal_equiv_rows": len(causal_equiv_rows),
            "n_b11_forwards": timers["b11_fwds"],
            "n_b1_forwards": timers["b1_fwds"],
            "n_total_forwards": timers["b11_fwds"] + timers["b1_fwds"],
            "max_abs_sham_raw": max_abs_sham["raw"],
            "max_abs_sham_peer_med": max_abs_sham["peer_med"],
            "max_abs_sham_spec": max_abs_sham["spec"],
            "max_abs_sham_logp": max_abs_sham["logp"],
            "max_abs_sham_phrase": max_abs_sham["phrase_mean"],
            "max_abs_decomp_err": max_abs_decomp_err,
            "max_sibling_leakage_science": max_sibling_leakage_science,
            "max_sibling_leakage_actuation": max_sibling_leakage_actuation,
            "max_abs_wdown_orth_cos": max_abs_wdown_orth_cos,
            "max_causal_equiv_err": max_causal_equiv_err,
            "min_blocked_pos_drop": min_blocked_pos_drop,
            "timing": timing,
        },
        "unit_vectors": unit_vectors_meta,
        "unpatched_rows": unpatched_rows,
        "blocked_rows": blocked_rows,
        "sham_rows": sham_rows,
        "actuation_rows": actuation_rows,
        "causal_equiv_rows": causal_equiv_rows,
    }

    if out_path is not None:
        # Compact atomic JSON write (F6: tmp + os.replace; ~10 MB raw / ~2 MB gz for Full).
        atomic_write_text(out_path, json.dumps(receipt, separators=(",", ":")))
    return receipt


# ---------------------------------------------------------------------------
# Realistic 15-Layer OLMo-2-Faithful Tiny CPU Model (for End-to-End CPU Tests)
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

        q, k = _rot(q), _rot(k)
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
    """Architectural twin of OLMo-2-0425-1B (15 layers covering L2..L14, 16 MHA heads, 100278 vocab)."""

    def __init__(self, seed: int = 12345, hidden_size: int = 64, num_heads: int = 16, num_layers: int = 15, vocab_size: int = 100278):
        super().__init__()
        torch.manual_seed(seed)
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
            rms_norm_eps=1e-6,
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_TinyOlmo2Layer(hidden_size, num_heads, intermediate_size=8192) for _ in range(num_layers)])
        self.norm = _TinyRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model = types.SimpleNamespace(layers=self.layers, norm=self.norm)

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
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    targets213_path: Path,
    m7_transport_path: Path,
) -> dict[str, Any]:
    checks: list[str] = []

    d_sha = sha256_file(design_path)
    m_sha = sha256_file(manifest_path)
    r_sha = sha256_file(runner_path)
    s_sha = sha256_file(scorer_path)
    t_sha = sha256_file(targets213_path)
    m7_sha = sha256_file(m7_transport_path)

    # 1. Single-read 6-pin admission.
    manifest, pins = admit_execution_request(
        mode="smoke",
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        targets213_path=targets213_path,
        m7_transport_path=m7_transport_path,
        expected_design_sha=d_sha,
        expected_manifest_sha=m_sha,
        expected_runner_sha=r_sha,
        expected_scorer_sha=s_sha,
        expected_targets213_sha=t_sha,
        expected_m7_transport_sha=m7_sha,
    )
    checks.append("single_read_6pin_admit_ok")

    # 2. Pre-load refusals (implicit mode, full without smoke, pin tamper).
    for bad_kw in [
        {"mode": None},
        {"mode": "full", "allow_full_after_smoke": False},
        {"mode": "smoke", "expected_manifest_sha": "0" * 64},
        {"mode": "smoke", "expected_targets213_sha": "0" * 64},
        {"mode": "smoke", "expected_m7_transport_sha": "0" * 64},
    ]:
        kw = dict(
            mode="smoke",
            design_path=design_path,
            manifest_path=manifest_path,
            runner_path=runner_path,
            scorer_path=scorer_path,
            targets213_path=targets213_path,
            m7_transport_path=m7_transport_path,
            expected_design_sha=d_sha,
            expected_manifest_sha=m_sha,
            expected_runner_sha=r_sha,
            expected_scorer_sha=s_sha,
            expected_targets213_sha=t_sha,
            expected_m7_transport_sha=m7_sha,
        )
        kw.update(bad_kw)
        raised = False
        try:
            admit_execution_request(**kw)
        except ValueError:
            raised = True
        assert raised, f"Expected pre-load ValueError for {bad_kw}"
    checks.append("preload_refusal_mode_full_pin_tamper_ok")

    # 3. End-to-end real-hook CPU Smoke run on 15-layer TinyOlmo2ForCausalLM (exact 650 B11 + 29 B=1 = 679 fwds!).
    tiny_model = TinyOlmo2ForCausalLM(seed=42, hidden_size=64, num_heads=16, num_layers=15, vocab_size=100278)
    tiny_model.eval()
    smoke_receipt = execute_concept_vs_slot(
        manifest=manifest,
        pins=pins,
        smoke_mode=True,
        device="cpu",
        model_loader=lambda _r, _c, _d: tiny_model,
        execution_provenance="synthetic_cpu_selftest",
    )
    sm = smoke_receipt["meta"]
    assert sm["n_unpatched_rows"] == 468
    assert sm["n_blocked_rows"] == 156
    assert sm["n_sham_rows"] == 26
    assert sm["n_actuation_rows"] == 26
    assert sm["n_causal_equiv_rows"] == 3
    assert sm["n_b11_forwards"] == 653
    assert sm["n_b1_forwards"] == 26
    assert sm["n_total_forwards"] == 679
    assert sm["max_abs_sham_raw"] <= TOL_SHAM
    assert sm["max_abs_sham_spec"] <= TOL_SHAM
    assert sm["max_abs_sham_logp"] <= TOL_SHAM
    assert sm["max_abs_sham_phrase"] <= TOL_SHAM
    assert sm["max_abs_decomp_err"] <= TOL_DECOMP
    assert sm["max_sibling_leakage_science"] <= TOL_SIBLING
    assert sm["max_sibling_leakage_actuation"] <= TOL_SIBLING
    assert sm["max_abs_wdown_orth_cos"] <= TOL_ORTH
    assert sm["max_causal_equiv_err"] <= TOL_CAUSAL_EQUIV
    assert sm["min_blocked_pos_drop"] > TOL_BLOCK_ACTUATION
    assert all(r["passed"] for r in smoke_receipt["sham_rows"])
    assert all(r["passed"] and r["readout_max_abs_delta"] > MIN_ACTUATION_READOUT_DELTA for r in smoke_receipt["actuation_rows"])
    assert all(r["passed"] and r["batch_size"] == BATCH_SIZE for r in smoke_receipt["causal_equiv_rows"])
    for ur in smoke_receipt["unpatched_rows"]:
        assert "sham" not in json.dumps(ur), "Unpatched row must not carry sham keys"
    validate_timing_breakdown(sm["timing"], 653, 26)
    checks.append("tiny_olmo2_smoke_e2e_679_forwards_cs0_causal_equiv_f1_f7_ok")

    # 3b. F1 negative test: inert no-op blocking pre-hook MUST cause execute_concept_vs_slot to REFUSE!
    broken_model = copy.deepcopy(tiny_model)
    l7_oproj = broken_model.layers[7].self_attn.o_proj
    orig_reg_pre = l7_oproj.register_forward_pre_hook

    def _noop_block_reg_pre(fn):
        # Allow capture pre-hooks (lambdas returning None) but neutralize the blocking patch pre-hook!
        def _wrapped(mod, args):
            fn(mod, args)  # still runs capture_after if present, but returns UNPATCHED args!
            return None
        return orig_reg_pre(_wrapped)

    l7_oproj.register_forward_pre_hook = _noop_block_reg_pre  # type: ignore[method-assign]
    refused_inert_hook = False
    try:
        execute_concept_vs_slot(
            manifest=manifest,
            pins=pins,
            smoke_mode=True,
            device="cpu",
            model_loader=lambda _r, _c, _d: broken_model,
            execution_provenance="synthetic_cpu_selftest",
        )
    except AssertionError as exc:
        refused_inert_hook = "F1 blocking positive control failed" in str(exc)
    assert refused_inert_hook, "Runner MUST refuse a no-op blocking pre-hook (F1)!"
    checks.append("f1_noop_blocking_hook_refused_ok")

    # 4. Pre-load full gate rejects synthetic, CPU-measured, forged GPU, bad timing, AND failed CS.1 transport;
    #    accepts authentic GPU-labeled smoke receipt with passing CS.0 + CS.1 + prelim head-null!
    with tempfile.TemporaryDirectory() as td:
        def _try_full_admit(rec_obj: dict[str, Any], fname: str) -> bool:
            p = Path(td) / fname
            p.write_text(json.dumps(rec_obj, separators=(",", ":")), encoding="utf-8")
            p_sha = sha256_file(p)
            try:
                admit_execution_request(
                    mode="full",
                    design_path=design_path,
                    manifest_path=manifest_path,
                    runner_path=runner_path,
                    scorer_path=scorer_path,
                    targets213_path=targets213_path,
                    m7_transport_path=m7_transport_path,
                    expected_design_sha=d_sha,
                    expected_manifest_sha=m_sha,
                    expected_runner_sha=r_sha,
                    expected_scorer_sha=s_sha,
                    expected_targets213_sha=t_sha,
                    expected_m7_transport_sha=m7_sha,
                    allow_full_after_smoke=True,
                    verified_smoke_receipt=p,
                    verified_smoke_sha256=p_sha,
                )
                return True
            except ValueError:
                return False

        # Synthetic CPU smoke refused.
        assert not _try_full_admit(smoke_receipt, "synth.json")

        # Random-weight tiny model fails CS.1 transport even if GPU-labeled -> MUST be refused!
        gpu_unboosted = copy.deepcopy(smoke_receipt)
        gpu_unboosted["meta"]["execution_provenance"] = "measured_gpu_run"
        gpu_unboosted["meta"]["device_name"] = "NVIDIA L4"
        gpu_unboosted["meta"]["is_cuda_device"] = True
        assert not _try_full_admit(gpu_unboosted, "gpu_failed_cs1.json")

        # Boost unpatched transport + zero blocked null on a copy to verify authentic passing GPU smoke is accepted,
        # while forged CPU device & bad timing remain refused!
        gpu_pass = copy.deepcopy(gpu_unboosted)
        for ur in gpu_pass["unpatched_rows"]:
            tr = ur["transport"]
            tr["t_pos_raw"] = 3.0
            tr["t_pos_peer_med"] = 0.0
            tr["t_pos_spec"] = 3.0
            tr["t_pos_logp"] = 3.0
            tr["t_pos_phrase_mean"] = 2.0
            tr["wdown_pos_raw"] = [0.0, 0.0, 0.0, 0.0]
            tr["wdown_pos_peer_med"] = [0.0, 0.0, 0.0, 0.0]
            tr["wdown_pos_spec"] = [0.0, 0.0, 0.0, 0.0]
        for br in gpu_pass["blocked_rows"]:
            for ch in CHANNELS_5:
                br[ch]["wd_pos"] = [0.0, 0.0, 0.0, 0.0]
                br[ch]["wd_neg"] = [0.0, 0.0, 0.0, 0.0]
        assert _try_full_admit(gpu_pass, "gpu_pass.json"), "Authentic passing GPU smoke should be accepted"

        forged_cpu = copy.deepcopy(gpu_pass)
        forged_cpu["meta"]["device_name"] = "cpu"
        assert not _try_full_admit(forged_cpu, "forged_cpu.json")

        bad_timing = copy.deepcopy(gpu_pass)
        bad_timing["meta"]["timing"]["projected_full_fwd_sec"] = 999999.0
        assert not _try_full_admit(bad_timing, "bad_timing.json")

        # Verify execute_concept_vs_slot refuses measured_gpu_run on CPU.
        refused_cpu_gpu_label = False
        try:
            execute_concept_vs_slot(
                manifest=manifest,
                pins=pins,
                smoke_mode=True,
                device="cpu",
                model_loader=lambda _r, _c, _d: tiny_model,
                execution_provenance="measured_gpu_run",
            )
        except ValueError:
            refused_cpu_gpu_label = True
        assert refused_cpu_gpu_label
    checks.append("full_gate_rejects_cpu_forged_bad_timing_cs1_fail_accepts_passing_gpu_ok")

    return {
        "runner_cpu_self_tests_passed": True,
        "n_checks": len(checks),
        "checks": checks,
        "tiny_smoke_metrics": {
            "n_b11_forwards": sm["n_b11_forwards"],
            "n_b1_forwards": sm["n_b1_forwards"],
            "max_abs_sham_raw": sm["max_abs_sham_raw"],
            "max_abs_sham_spec": sm["max_abs_sham_spec"],
            "max_abs_sham_logp": sm["max_abs_sham_logp"],
            "max_abs_sham_phrase": sm["max_abs_sham_phrase"],
            "max_abs_decomp_err": sm["max_abs_decomp_err"],
            "max_sibling_leakage_science": sm["max_sibling_leakage_science"],
            "max_sibling_leakage_actuation": sm["max_sibling_leakage_actuation"],
            "max_abs_wdown_orth_cos": sm["max_abs_wdown_orth_cos"],
            "max_causal_equiv_err": sm["max_causal_equiv_err"],
            "min_blocked_pos_drop": sm["min_blocked_pos_drop"],
            "min_actuation_target_delta": min(r["target_max_abs_delta"] for r in smoke_receipt["actuation_rows"]),
            "min_actuation_readout_delta": min(r["readout_max_abs_delta"] for r in smoke_receipt["actuation_rows"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Concept vs. Slot Head-Gating & Late-Layer Heads Runner")
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
    ap.add_argument("--targets213", type=Path, default=None)
    ap.add_argument("--m7-transport", type=Path, default=None)
    ap.add_argument("--score92", type=Path, default=None)
    ap.add_argument("--part99-pushes", type=Path, default=None)
    ap.add_argument("--lens-discovery", type=Path, default=None)
    ap.add_argument("--design-sha256", type=str, default="")
    ap.add_argument("--manifest-sha256", type=str, default="")
    ap.add_argument("--runner-sha256", type=str, default="")
    ap.add_argument("--scorer-sha256", type=str, default="")
    ap.add_argument("--targets213-sha256", type=str, default=TARGETS213_SHA256)
    ap.add_argument("--m7-transport-sha256", type=str, default=M7_TRANSPORT_SHA256)
    ap.add_argument("--allow-full-after-smoke", action="store_true")
    ap.add_argument("--verified-smoke-receipt", type=Path, default=None)
    ap.add_argument("--verified-smoke-sha256", type=str, default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    base_dir = repo_root / "explore_olmo2_attn_mlp"
    design_p = args.design or (base_dir / "CONCEPT-VS-SLOT-DESIGN.md")
    manifest_p = args.manifest or (base_dir / "concept_vs_slot_manifest.json")
    runner_p = args.runner or Path(__file__).resolve()
    scorer_p = args.scorer or (base_dir / "score_concept_vs_slot.py")
    t213_p = args.targets213 or (repo_root / TARGETS213_REL_PATH)
    m7_p = args.m7_transport or (repo_root / M7_TRANSPORT_REL_PATH)
    s92_p = args.score92 or (repo_root / "results/part92/score92.json")
    p99_p = args.part99_pushes or (repo_root / "results/part99-pushes.json")
    lens_p = args.lens_discovery or Path(
        "/home/elisepc/green/Journal/notes/reports/vesper-olmo-neuron-lens-2026-11-21/olmo_neuron_lens_discovery.json"
    )

    if args.build_manifest:
        man = build_concept_vs_slot_manifest(
            repo_root=repo_root,
            design_path=design_p,
            targets213_path=t213_p,
            m7_transport_path=m7_p,
            score92_path=s92_p,
            part99_pushes_path=p99_p,
            lens_discovery_path=lens_p,
        )
        out_p = args.out or manifest_p
        atomic_write_text(out_p, json.dumps(man, indent=2))
        print(json.dumps({
            "built_manifest": str(out_p),
            "sha256": sha256_file(out_p),
            "denominators": man["meta"]["denominators"],
            "n_units": len(man["units"]),
            "n_pool_tokens": len(man["competitor_pool_37"]),
            "n_heads": len(man["heads"]),
        }, indent=2))
        return 0

    if args.self_test_only:
        res = run_cpu_self_tests(
            design_path=design_p,
            manifest_path=manifest_p,
            runner_path=runner_p,
            scorer_path=scorer_p,
            targets213_path=t213_p,
            m7_transport_path=m7_p,
        )
        print(json.dumps(res, indent=2))
        return 0

    manifest, pins = admit_execution_request(
        mode=args.mode,
        design_path=design_p,
        manifest_path=manifest_p,
        runner_path=runner_p,
        scorer_path=scorer_p,
        targets213_path=t213_p,
        m7_transport_path=m7_p,
        expected_design_sha=args.design_sha256,
        expected_manifest_sha=args.manifest_sha256,
        expected_runner_sha=args.runner_sha256,
        expected_scorer_sha=args.scorer_sha256,
        expected_targets213_sha=args.targets213_sha256,
        expected_m7_transport_sha=args.m7_transport_sha256,
        allow_full_after_smoke=args.allow_full_after_smoke,
        verified_smoke_receipt=args.verified_smoke_receipt,
        verified_smoke_sha256=args.verified_smoke_sha256,
    )
    print(f"[PRECHECK OK] 6-pin & mode={args.mode} gate passed")
    if args.precheck_only:
        return 0

    out_p = args.out or (base_dir / f"concept_vs_slot_{args.mode}_receipt.json")
    rec = execute_concept_vs_slot(
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
        "n_blocked_rows": rec["meta"]["n_blocked_rows"],
        "n_sham_rows": rec["meta"]["n_sham_rows"],
        "n_actuation_rows": rec["meta"]["n_actuation_rows"],
        "n_causal_equiv_rows": rec["meta"]["n_causal_equiv_rows"],
        "n_b11_forwards": rec["meta"]["n_b11_forwards"],
        "n_b1_forwards": rec["meta"]["n_b1_forwards"],
        "max_abs_sham_raw": rec["meta"]["max_abs_sham_raw"],
        "max_abs_sham_spec": rec["meta"]["max_abs_sham_spec"],
        "max_abs_sham_logp": rec["meta"]["max_abs_sham_logp"],
        "max_abs_sham_phrase": rec["meta"]["max_abs_sham_phrase"],
        "max_sibling_leakage_science": rec["meta"]["max_sibling_leakage_science"],
        "max_sibling_leakage_actuation": rec["meta"]["max_sibling_leakage_actuation"],
        "max_causal_equiv_err": rec["meta"]["max_causal_equiv_err"],
        "timing": rec["meta"]["timing"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
