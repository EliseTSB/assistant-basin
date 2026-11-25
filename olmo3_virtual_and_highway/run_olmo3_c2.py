#!/usr/bin/env python3
"""Stage C2 GPU runner: Held-Out Whitening Arm Comparison & Frozen-Lens Calibration (C2.1),
Pure ChatML-Template ΣMLP Causal Patching (C2.2), 5-Layer Post-Erasure Sweep (C2.3),
and Literal pre_norm Low-γ Mask / Dual-Branch / K=3 Additivity (C2.4) on OLMo 3 7B
(`OLMO3-TRANSPORT-C2-DESIGN.md` Registered v4, public `main` @ `ecd49b28`).

Exact planned forward budget (Full run, assuming 0 caps):
  - Clean baselines (36 prompts) + 19 enumerated shams (13 write + 6 patch) + 16 C0-colon pairs =    71
  - Block 1 (C2.1): 3L x 5 arms x 24 targets x 8 prompts (4 LongBare + 4 ChatML, last)          = 2,880
  - Block 2 (C2.2): 9 conds x 12 targets x 12 pairs x 2 (R_mf + dm0) + 72 unsteered KL fwds     = 2,664
  - Block 2b (C2.3): 5L (18..30) x (24 logit_lens + 8 rand_iso) x 8 prompts (last)              = 1,280
  - Block 3 (C2.4): 9 pre_norm arms x 24 targets x 4 prompts x 2 doses (0.50x, 1.25x)           = 1,728
  - Total = 8,623 short B=1 bf16 forwards (use_cache=False).
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from olmo3_virtual_and_highway.run_olmo3_c0 import (
    MODEL_ID,
    MODEL_REV,
    capture_last_trace,
    decompose_rand_iso_trace,
    decompose_target_trace,
    sha256_file,
)
from olmo3_virtual_and_highway.run_olmo3_c0b import post_norm_inject_hook

PRIMARY_C21_LAYERS = [18, 24, 27]
SWEEP_LAYERS = [18, 21, 24, 27, 30]
PRENORM_LAYERS = [21, 24, 27]

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_C2_CONSTANTS_SHA256 = "583117f50cfceeb5cddcfa239c0efa0c32846dd69f6756540cd3bbba5cec5b65"
EXPECTED_C2_BANK_SHA256 = "7124dd565331fd455d6d91d8fe845ba8783c2b0d366f6d11eb533947cb09401d"
EXPECTED_C2_DESIGN_SHA256 = "3764dfee6baf9c7387fc6325eee92c4ca2fdbca5c76ee5b440e0f4f12169d8fc"

C21_ARMS = ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink", "logit_lens")
C22_PATCH_L19_31 = list(range(19, 32))
C22_PATCH_L4_16 = list(range(4, 17))
C24_DOSES = (0.50, 1.25)


# ---------------------------------------------------------------------------
# Fail-closed SHA-256 provenance preflight (C2.0.4)
# ---------------------------------------------------------------------------
def verify_preflight_provenance(
    bank_path: Path,
    r_lens_path: Path,
    c2_constants_path: Path,
    design_path: Path,
) -> dict[str, str]:
    """Fail-closed preflight before loading weights: verify all 4 frozen inputs match pinned SHA-256s."""
    for label, p in (
        ("--bank", bank_path),
        ("--r-lens", r_lens_path),
        ("--c2-constants", c2_constants_path),
        ("--design", design_path),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Preflight failed: missing required {label} file: {p}")

    hashes = {
        "r_lens_sha256": sha256_file(r_lens_path),
        "c2_constants_sha256": sha256_file(c2_constants_path),
        "c2_bank_sha256": sha256_file(bank_path),
        "c2_design_sha256": sha256_file(design_path),
    }
    expected = {
        "r_lens_sha256": EXPECTED_R_LENS_SHA256,
        "c2_constants_sha256": EXPECTED_C2_CONSTANTS_SHA256,
        "c2_bank_sha256": EXPECTED_C2_BANK_SHA256,
        "c2_design_sha256": EXPECTED_C2_DESIGN_SHA256,
    }
    mismatches = {k: (hashes[k], expected[k]) for k in expected if hashes[k] != expected[k]}
    if mismatches:
        raise ValueError(f"Preflight SHA-256 mismatch (refusing to load model / run GPU): {mismatches}")

    with safe_open(str(bank_path), framework="pt") as f:
        meta = f.metadata() or {}
    if meta.get("r_lens_sha256") != EXPECTED_R_LENS_SHA256:
        raise ValueError(f"Bank metadata r_lens_sha256 mismatch: {meta.get('r_lens_sha256')}")
    if meta.get("model") != MODEL_ID or meta.get("revision") != MODEL_REV:
        raise ValueError(f"Bank metadata model/revision mismatch: {meta.get('model')}@{meta.get('revision')}")

    c2_const = json.loads(c2_constants_path.read_text(encoding="utf-8"))
    if c2_const.get("schema_version") != "c2-prereg-constants-v4":
        raise ValueError(f"Unexpected constants schema_version: {c2_const.get('schema_version')!r}")
    if c2_const.get("c2_vector_bank_sha256") != EXPECTED_C2_BANK_SHA256:
        raise ValueError("c2_prereg_constants.json c2_vector_bank_sha256 mismatch")
    if c2_const.get("r_lens_sha256") != EXPECTED_R_LENS_SHA256:
        raise ValueError("c2_prereg_constants.json r_lens_sha256 mismatch")
    return hashes


def load_c2_bank(bank_path: Path) -> dict[str, torch.Tensor]:
    """Load all 34 frozen Stage C2 bank tensors and validate finiteness."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(bank_path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).float().contiguous()
            if not bool(torch.isfinite(t).all().item()):
                raise ValueError(f"Non-finite value in bank tensor {k}")
            tensors[k] = t
    return tensors


# ---------------------------------------------------------------------------
# Stage C2 Hook Primitives (pre_norm injection & C2.2 suffix capture/patching)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def pre_norm_inject_hook(
    model,
    injections: list[tuple[int, str, torch.Tensor, float]],
):
    """Inject `dose * w_unit` BEFORE `post_feedforward_layernorm` ('ff') or `post_attention_layernorm` ('attn')
    at `positions='last'`. Algebraically identical to appending a literal rank-1 column to `down_proj` / `o_proj`.

    Always validates `w_unit` and runs the exact clone/cast/add path even at `dose == 0.0` (sham parity).
    """
    handles = []
    for layer, branch, w_unit, dose in injections:
        if branch not in ("ff", "attn"):
            raise ValueError(f"Unsupported pre_norm branch: {branch!r}")
        w_cpu = w_unit.detach().float()
        if not bool(torch.isfinite(w_cpu).all().item()):
            raise ValueError(f"Non-finite write vector in pre_norm_inject_hook L{layer}.{branch}")
        w_nrm = float(w_cpu.norm().item())
        if abs(w_nrm - 1.0) > 1e-3:
            raise ValueError(f"Non-unit write vector in pre_norm_inject_hook L{layer}.{branch}: ||w||={w_nrm:.6g}")
        if not math.isfinite(float(dose)):
            raise ValueError(f"Non-finite dose in pre_norm_inject_hook L{layer}.{branch}: {dose}")

        lay = model.model.layers[layer]
        mod = lay.post_feedforward_layernorm if branch == "ff" else lay.post_attention_layernorm

        def _make_pre_hook(w_t: torch.Tensor, d_val: float):
            def _pre(_mod, args):
                x = args[0].clone()
                delta = (float(d_val) * w_t.to(device=x.device, dtype=torch.float32)).to(x.dtype)
                x[:, -1, :] = x[:, -1, :] + delta
                return (x,) + tuple(args[1:])
            return _pre

        handles.append(mod.register_forward_pre_hook(_make_pre_hook(w_cpu, float(dose))))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def capture_mlp_suffix_hook(model, layers: list[int], suffix_len: int, dest: dict[int, torch.Tensor]):
    """Capture `post_feedforward_layernorm` outputs over trailing `suffix_len` tokens on `layers`."""
    handles = []
    for l in layers:
        def _make_cap(li: int):
            def _cap(_mod, _args, output):
                dest[li] = output[:, -suffix_len:, :].detach().cpu().clone()
            return _cap
        handles.append(model.model.layers[l].post_feedforward_layernorm.register_forward_hook(_make_cap(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def patch_mlp_suffix_hook(
    model,
    layers: list[int],
    suffix_len: int,
    mode: str,
    donor_acts: dict[int, torch.Tensor] | None = None,
    rand_seed: int | None = None,
    rand_scale: float = 1.0,
    self_scale_unit_sham: bool = False,
):
    """Patch `post_feedforward_layernorm` outputs over trailing `suffix_len` tokens on `layers`.

    Modes:
      - 'replace': exact replacement with `donor_acts[l]` (B1 symmetric suffix patching; self-patch sham if donor==self).
      - 'rand_iso': add deterministic isotropic noise per suffix position with L2 norm matched to
        `rand_scale * ||donor_acts[l] - out_suf||_2` (0.0 for zero-scale sham).
      - 'self_attenuated': rescale recipient's OWN suffix MLP output per layer & token position to match
        `||donor_acts[l]||_2` (or exact factor 1.0 when `self_scale_unit_sham=True`).
    """
    if mode not in ("replace", "rand_iso", "self_attenuated"):
        raise ValueError(f"Unsupported patch mode: {mode!r}")
    handles = []
    for l in layers:
        def _make_patch(li: int):
            def _hook(_mod, _args, output):
                out = output.clone()
                if mode == "replace":
                    assert donor_acts is not None and li in donor_acts
                    d_t = donor_acts[li].to(device=out.device, dtype=out.dtype)
                    if d_t.shape != out[:, -suffix_len:, :].shape:
                        raise ValueError(f"Suffix patch shape mismatch at L{li}: {d_t.shape} vs {out[:, -suffix_len:, :].shape}")
                    out[:, -suffix_len:, :] = d_t
                elif mode == "rand_iso":
                    assert donor_acts is not None and li in donor_acts and rand_seed is not None
                    cur = out[:, -suffix_len:, :].float()
                    don = donor_acts[li].to(device=out.device, dtype=torch.float32)
                    diff_norm = (don - cur).norm(dim=-1, keepdim=True)  # [1, T_s, 1]
                    g = torch.Generator(device="cpu")
                    g.manual_seed(int(rand_seed) + li * 997)
                    z = torch.randn(cur.shape, generator=g, dtype=torch.float32).to(device=out.device)
                    z_u = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    out[:, -suffix_len:, :] = (cur + float(rand_scale) * diff_norm * z_u).to(out.dtype)
                elif mode == "self_attenuated":
                    cur = out[:, -suffix_len:, :].float()
                    if self_scale_unit_sham:
                        scale = torch.ones_like(cur[..., :1])
                    else:
                        assert donor_acts is not None and li in donor_acts
                        don = donor_acts[li].to(device=out.device, dtype=torch.float32)
                        cur_n = cur.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        don_n = don.norm(dim=-1, keepdim=True)
                        scale = don_n / cur_n
                    out[:, -suffix_len:, :] = (cur * scale).to(out.dtype)
                return out
            return _hook
        handles.append(model.model.layers[l].post_feedforward_layernorm.register_forward_hook(_make_patch(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _kl_div_from_clean(clean_logits: torch.Tensor, int_logits: torch.Tensor) -> float:
    lc = clean_logits.float()
    li = int_logits.float()
    log_pc = F.log_softmax(lc, dim=-1)
    pc = log_pc.exp()
    log_pi = F.log_softmax(li, dim=-1)
    return float((pc * (log_pc - log_pi)).sum().item())


def _max_abs_dlogit(clean_logits: torch.Tensor, int_logits: torch.Tensor) -> float:
    return float((int_logits.float() - clean_logits.float()).abs().max().item())


def build_same_suffix_mismatch_map(triples: list[dict[str, Any]]) -> dict[int, int]:
    """Deterministic derangement mapping each pair_idx to a different stem with IDENTICAL `suffix_len`."""
    by_len: dict[int, list[int]] = {}
    for tr in triples:
        by_len.setdefault(int(tr["suffix_len"]), []).append(int(tr["pair_idx"]))
    mp: dict[int, int] = {}
    for slen, idxs in by_len.items():
        if len(idxs) < 2:
            raise ValueError(f"Cannot build same-length mismatch derangement for suffix_len={slen}: {idxs}")
        for k, pidx in enumerate(idxs):
            mp[pidx] = idxs[(k + 1) % len(idxs)]
    return mp


# ---------------------------------------------------------------------------
# Core Stage C2 Experiment Engine (Model-agnostic for both OLMo-3 GPU & tiny CPU test model)
# ---------------------------------------------------------------------------
def compute_frozen_cal_reference(
    c2_constants: dict[str, Any],
    bank: dict[str, torch.Tensor],
    r_out_all: Any | None,
    primary_c21_layers: list[int],
) -> dict[str, Any]:
    """Compute exact frozen CAL-C2.1 reference arrays (`u_mu_proj_n` [24] and `16x8` heldout linear preds)
    so the 100% pure-stdlib scorer needs zero PyTorch/r_lens.pt loading while staying hash-verified.
    """
    U_d = bank["u_vectors"].double()
    mu_u = bank["mu_u"].double()
    mu_hat = mu_u / mu_u.norm().clamp_min(1e-12)
    u_mu = U_d @ mu_hat
    u_mu_n = (u_mu / u_mu.norm().clamp_min(1e-12)).tolist()

    targets = c2_constants["targets_24"]
    prim_idx = [i for i, t in enumerate(targets) if t["split"] == "primary_16"]
    ho_idx = [i for i, t in enumerate(targets) if t["split"] == "heldout_8"]
    frozen_ho_preds: dict[str, dict[str, list[list[float]]]] = {}
    if r_out_all is not None:
        for L in primary_c21_layers:
            R = r_out_all[L].double()
            per_arm: dict[str, list[list[float]]] = {}
            for aname in ("R_ridge_raw", "R_ridge_shrink"):
                W = bank[f"L{L}.{aname}"].double()
                M = W[prim_idx] @ (R.T @ U_d[ho_idx].T)  # [16, 8]
                per_arm[aname] = [[round(float(v), 6) for v in row] for row in M.tolist()]
            frozen_ho_preds[str(L)] = per_arm
    return {
        "u_mu_proj_n": [round(float(v), 6) for v in u_mu_n],
        "primary_16_indices": prim_idx,
        "heldout_8_indices": ho_idx,
        "frozen_heldout_preds_16x8": frozen_ho_preds,
    }


def run_stage_c2_experiment(
    model,
    tokenizer,
    c2_constants: dict[str, Any],
    bank: dict[str, torch.Tensor],
    provenance_hashes: dict[str, str],
    *,
    r_out_all: Any | None = None,
    smoke: bool = False,
    device_name: str = "unknown",
    patch_l19_31: list[int] | None = None,
    patch_l4_16: list[int] | None = None,
    primary_c21_layers: list[int] | None = None,
    sweep_layers: list[int] | None = None,
    prenorm_layers: list[int] | None = None,
    c22_inject_layer: int = 18,
    c24_primary_layer: int = 24,
) -> dict[str, Any]:
    """Execute full Stage C2 v4 (or smoke subset) and return the complete receipt dictionary."""
    t0 = time.time()
    n_model_layers = len(model.model.layers)
    p_c21_layers = list(primary_c21_layers if primary_c21_layers is not None else PRIMARY_C21_LAYERS)
    p_sweep_layers = list(sweep_layers if sweep_layers is not None else SWEEP_LAYERS)
    p_prenorm_layers = list(prenorm_layers if prenorm_layers is not None else PRENORM_LAYERS)
    l_p19 = list(patch_l19_31 if patch_l19_31 is not None else C22_PATCH_L19_31)
    l_p4 = list(patch_l4_16 if patch_l4_16 is not None else C22_PATCH_L4_16)
    l_cap_all = sorted(set(l_p19) | set(l_p4))

    targets_24 = list(c2_constants["targets_24"])
    target_ids_24 = [int(t["id"]) for t in targets_24]
    c22_target_indices = list(c2_constants["c22_target_indices_12"])
    prompts_8_c21 = list(c2_constants["prompts_8_c21"])
    triples_12 = list(c2_constants["c22_stem_pairs_12"])
    c0_colon_prompts = list(c2_constants["c0_anchor_colon_prediction"]["prompts_16"])
    c22_pred_ret_map = c2_constants["c22_l18_frozen_pred_ret"]
    unmasked_eff_l24 = float(c2_constants["prenorm_gamma_mask_geometry"][str(c24_primary_layer)]["ff"]["unmasked_deliverable_frac"])

    if smoke:
        c21_t_idxs = [0, 1]
        c21_p_idxs = [0, 4]
        c22_t_idxs = c22_target_indices[:2]
        c22_pair_idxs = [0, 2]  # both suffix_len=12!
        c23_t_idxs = [0, 18]
        c23_rand_n = 2
        c24_t_idxs = [0, 1]
        c24_p_idxs = [0, 1]
    else:
        c21_t_idxs = list(range(len(targets_24)))
        c21_p_idxs = list(range(len(prompts_8_c21)))
        c22_t_idxs = list(c22_target_indices)
        c22_pair_idxs = list(range(len(triples_12)))
        c23_t_idxs = list(range(len(targets_24)))
        c23_rand_n = int(bank[f"L{p_sweep_layers[0]}.rand_iso"].shape[0])
        c24_t_idxs = list(range(len(targets_24)))
        c24_p_idxs = [0, 1, 2, 3]  # 2 LongBare + 2 ChatML on narr_0, anal_0

    mismatch_map = build_same_suffix_mismatch_map(triples_12)
    u_vectors = bank["u_vectors"]  # [24, d]
    u_hat = bank["u_hat"]          # [24, d]

    dev = next(model.parameters()).device

    def _encode(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=dev)

    fwd_counts = {
        "clean_prompts": 0,
        "write_shams": 0,
        "patch_shams": 0,
        "c0_colon_anchor": 0,
        "block1_C2_1": 0,
        "block2_C2_2_steered": 0,
        "block2_C2_2_unsteered_kl": 0,
        "block2_C2_2_capped_skipped": 0,
        "block2b_C2_3": 0,
        "block3_C2_4": 0,
        "block3_C2_4_capped_skipped": 0,
    }

    # ------------------------------------------------------------------
    # 1. Clean Baselines (36 prompts = 12 ShortBare + 12 LongBare + 12 ChatML)
    # ------------------------------------------------------------------
    clean_traces: dict[str, dict[str, Any]] = {}
    clean_probs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    clean_mlp_suf: dict[str, dict[int, torch.Tensor]] = {}
    enc_by_id: dict[str, torch.Tensor] = {}

    for tr in triples_12:
        t_s = int(tr["suffix_len"])
        for fmt in ("short_bare", "long_bare", "chatml"):
            p = tr[fmt]
            pid = p["id"]
            enc = _encode(p["text"])
            enc_by_id[pid] = enc
            cap_buf: dict[int, torch.Tensor] = {}
            with capture_mlp_suffix_hook(model, l_cap_all, t_s, cap_buf):
                ct = capture_last_trace(model, enc)
                fwd_counts["clean_prompts"] += 1
            clean_traces[pid] = ct
            clean_mlp_suf[pid] = cap_buf
            log_p = F.log_softmax(ct["logits"].float(), dim=-1)
            clean_probs[pid] = (log_p.exp(), log_p)

    # Live median post_mlp_norm across the 8 C2.1 prompts (`4 LongBare + 4 ChatML`) for every layer
    c21_prompt_ids = [p["id"] for p in prompts_8_c21]
    post_mlp_norms: dict[str, float] = {}
    for L in range(n_model_layers):
        vals = sorted(float(clean_traces[pid]["mlp"][L].float().norm().item()) for pid in c21_prompt_ids)
        mid = len(vals) // 2
        med = vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
        post_mlp_norms[str(L)] = round(float(med), 6)

    clean_rms_final = {pid: round(float(ct["rms_final"]), 6) for pid, ct in clean_traces.items()}

    # ------------------------------------------------------------------
    # 2. 19 Enumerated Shams (13 Write Shams at dose=0.0 + 6 Patch Shams)
    # ------------------------------------------------------------------
    p0_chat_id = triples_12[0]["chatml"]["id"]
    p0_long_id = triples_12[0]["long_bare"]["id"]
    p0_ts = int(triples_12[0]["suffix_len"])
    shams_write: dict[str, float] = {}
    shams_patch: dict[str, float] = {}

    # 13 Write shams (dose = 0.0 -> exact 0.0 max |dlogit|)
    for L in p_c21_layers:
        with post_norm_inject_hook(model, L, bank[f"L{L}.R_ridge_shrink"][0], dose=0.0, positions="last"):
            tr_s = capture_last_trace(model, enc_by_id[p0_chat_id])
            fwd_counts["write_shams"] += 1
        shams_write[f"C2.1_post_norm_L{L}"] = _max_abs_dlogit(clean_traces[p0_chat_id]["logits"], tr_s["logits"])

    for L in p_sweep_layers:
        with post_norm_inject_hook(model, L, u_hat[0], dose=0.0, positions="last"):
            tr_s = capture_last_trace(model, enc_by_id[p0_chat_id])
            fwd_counts["write_shams"] += 1
        shams_write[f"C2.3_sweep_L{L}"] = _max_abs_dlogit(clean_traces[p0_chat_id]["logits"], tr_s["logits"])

    for L in p_prenorm_layers:
        with pre_norm_inject_hook(model, [(L, "ff", bank[f"L{L}.prenorm_ff_masked"][0], 0.0)]):
            tr_s = capture_last_trace(model, enc_by_id[p0_chat_id])
            fwd_counts["write_shams"] += 1
        shams_write[f"C2.4_prenorm_L{L}_ff"] = _max_abs_dlogit(clean_traces[p0_chat_id]["logits"], tr_s["logits"])

    with pre_norm_inject_hook(
        model,
        [
            (c24_primary_layer, "ff", bank[f"L{c24_primary_layer}.prenorm_ff_masked"][0], 0.0),
            (c24_primary_layer, "attn", bank[f"L{c24_primary_layer}.prenorm_attn_masked"][0], 0.0),
        ],
    ):
        tr_s = capture_last_trace(model, enc_by_id[p0_chat_id])
        fwd_counts["write_shams"] += 1
    shams_write["C2.4_prenorm_L24_m70a30"] = _max_abs_dlogit(clean_traces[p0_chat_id]["logits"], tr_s["logits"])

    with pre_norm_inject_hook(
        model, [(L, "ff", bank[f"L{L}.prenorm_ff_masked"][0], 0.0) for L in p_prenorm_layers]
    ):
        tr_s = capture_last_trace(model, enc_by_id[p0_chat_id])
        fwd_counts["write_shams"] += 1
    shams_write["C2.4_prenorm_K3_combined"] = _max_abs_dlogit(clean_traces[p0_chat_id]["logits"], tr_s["logits"])

    # 6 Patch shams (self-patch / zero-scale / unit-scale identity -> exact 0.0 max |dlogit|)
    patch_sham_specs = [
        ("C2.2_self_patch_ChatML_MLP_L19_31", p0_chat_id, l_p19, "replace", clean_mlp_suf[p0_chat_id], None, 1.0, False),
        ("C2.2_self_patch_LongBare_MLP_L19_31", p0_long_id, l_p19, "replace", clean_mlp_suf[p0_long_id], None, 1.0, False),
        ("C2.2_self_patch_ctrl_L4_16_MLP", p0_chat_id, l_p4, "replace", clean_mlp_suf[p0_chat_id], None, 1.0, False),
        ("C2.2_self_patch_ctrl_mismatch_L19_31", p0_chat_id, l_p19, "replace", clean_mlp_suf[p0_chat_id], None, 1.0, False),
        ("C2.2_zero_scale_ctrl_rand_iso_L19_31", p0_chat_id, l_p19, "rand_iso", clean_mlp_suf[p0_long_id], 2026112599, 0.0, False),
        ("C2.2_unit_scale_ctrl_self_attenuated_L19_31", p0_chat_id, l_p19, "self_attenuated", None, None, 1.0, True),
    ]
    for sname, pid, slayers, smode, sdon, sseed, srscale, sunit in patch_sham_specs:
        with patch_mlp_suffix_hook(
            model, slayers, p0_ts, smode, donor_acts=sdon, rand_seed=sseed, rand_scale=srscale, self_scale_unit_sham=sunit
        ):
            tr_s = capture_last_trace(model, enc_by_id[pid])
            fwd_counts["patch_shams"] += 1
        shams_patch[sname] = _max_abs_dlogit(clean_traces[pid]["logits"], tr_s["logits"])

    # ------------------------------------------------------------------
    # 3. C0-Colon Anchor Ratio Prediction (16 clean forwards, non-gating)
    # ------------------------------------------------------------------
    c0_colon_results: dict[str, dict[str, Any]] = {}
    for p in c0_colon_prompts:
        enc = _encode(p["text"])
        tr_c0 = capture_last_trace(model, enc)
        fwd_counts["c0_colon_anchor"] += 1
        l24_idx = min(24, n_model_layers - 1)
        l27_idx = min(27, n_model_layers - 1)
        c0_colon_results[p["id"]] = {
            "pair_id": p["pair_id"],
            "format": p["format"],
            "ending": p["ending"],
            "n_tokens": int(enc.shape[1]),
            "pmn_L24": round(float(tr_c0["mlp"][l24_idx].float().norm().item()), 6),
            "pmn_L27": round(float(tr_c0["mlp"][l27_idx].float().norm().item()), 6),
            "rms_final": round(float(tr_c0["rms_final"]), 6),
        }

    # Track active liveness per arm family
    arm_min_move: dict[str, float] = {}

    def _note_move(arm_key: str, mv: float) -> None:
        if arm_key not in arm_min_move or mv < arm_min_move[arm_key]:
            arm_min_move[arm_key] = mv

    # ------------------------------------------------------------------
    # 4. Block 1 (C2.1): 3L x 5 arms x 24 targets x 8 prompts (last, 0.75x pmn[L])
    # ------------------------------------------------------------------
    block1_c21: dict[str, Any] = {}
    for L in p_c21_layers:
        dose_l = 0.75 * post_mlp_norms[str(L)]
        l_blk: dict[str, Any] = {"dose": round(dose_l, 6), "arms": {}}
        for aname in C21_ARMS:
            w_mat = u_hat if aname == "logit_lens" else bank[f"L{L}.{aname}"]
            arm_blk: dict[str, list[dict[str, Any]]] = {}
            for ti in c21_t_idxs:
                t = targets_24[ti]
                t_str = t["string"]
                tid = int(t["id"])
                u_t = u_vectors[ti]
                w_vec = w_mat[ti]
                rows = []
                for pi in c21_p_idxs:
                    pid = prompts_8_c21[pi]["id"]
                    ct = clean_traces[pid]
                    pc, lpc = clean_probs[pid]
                    with post_norm_inject_hook(model, L, w_vec, dose=dose_l, positions="last"):
                        tr_i = capture_last_trace(model, enc_by_id[pid])
                        fwd_counts["block1_C2_1"] += 1
                    mv = _max_abs_dlogit(ct["logits"], tr_i["logits"])
                    _note_move(f"C2.1_L{L}_{aname}", mv)
                    rec = decompose_target_trace(ct, tr_i, L, u_t, tid, dose_l, 1.0, w_vec, pc, lpc)
                    dlog_all = [
                        round(float(tr_i["logits"][j_id].item() - ct["logits"][j_id].item()), 6)
                        for j_id in target_ids_24
                    ]
                    rec["prompt_id"] = pid
                    rec["all_24_dlogits"] = dlog_all
                    rec["capped"] = False
                    rec["max_abs_dlogit"] = round(mv, 6)
                    rows.append(rec)
                arm_blk[t_str] = rows
            l_blk["arms"][aname] = arm_blk
        block1_c21[str(L)] = l_blk

    # ------------------------------------------------------------------
    # 5. Block 2 (C2.2): Pure ChatML-Template L19-L31 Causal Patching + 72 Unsteered Clean-KL
    # ------------------------------------------------------------------
    cap_d_l18 = 3.0 * post_mlp_norms[str(c22_inject_layer)]
    w_rmf_l18 = bank[f"L{c22_inject_layer}.R_mf"]
    w_dm_l18 = bank[f"L{c22_inject_layer}.direct_matched"]  # [12, d]
    c22_pos_map = {ti: pos for pos, ti in enumerate(c22_target_indices)}

    # 5a. First run the 72 unsteered patched clean-KL forwards (6 patched conditions x 12 pairs)
    c22_unsteered_kl: dict[str, list[dict[str, Any]]] = {
        "patch_LongBare_to_ChatML_MLP_L19_31": [],
        "patch_ChatML_to_LongBare_MLP_L19_31": [],
        "ctrl_L4_16_MLP_LongBare_to_ChatML": [],
        "ctrl_mismatch_prompt_MLP_L19_31": [],
        "ctrl_rand_iso_same_site_L19_31": [],
        "ctrl_self_attenuated_L19_31": [],
    }
    for pair_idx in c22_pair_idxs:
        tr = triples_12[pair_idx]
        t_s = int(tr["suffix_len"])
        id_long = tr["long_bare"]["id"]
        id_chat = tr["chatml"]["id"]
        id_mis_long = triples_12[mismatch_map[pair_idx]]["long_bare"]["id"]
        kl_specs = [
            ("patch_LongBare_to_ChatML_MLP_L19_31", id_chat, l_p19, "replace", clean_mlp_suf[id_long], None),
            ("patch_ChatML_to_LongBare_MLP_L19_31", id_long, l_p19, "replace", clean_mlp_suf[id_chat], None),
            ("ctrl_L4_16_MLP_LongBare_to_ChatML", id_chat, l_p4, "replace", clean_mlp_suf[id_long], None),
            ("ctrl_mismatch_prompt_MLP_L19_31", id_chat, l_p19, "replace", clean_mlp_suf[id_mis_long], None),
            ("ctrl_rand_iso_same_site_L19_31", id_chat, l_p19, "rand_iso", clean_mlp_suf[id_long], 2026112600 + pair_idx * 31),
            ("ctrl_self_attenuated_L19_31", id_chat, l_p19, "self_attenuated", clean_mlp_suf[id_long], None),
        ]
        for cname, rec_id, pl, pmode, pdon, pseed in kl_specs:
            with patch_mlp_suffix_hook(model, pl, t_s, pmode, donor_acts=pdon, rand_seed=pseed):
                tr_u = capture_last_trace(model, enc_by_id[rec_id])
                fwd_counts["block2_C2_2_unsteered_kl"] += 1
            kl_v = _kl_div_from_clean(clean_traces[rec_id]["logits"], tr_u["logits"])
            mv = _max_abs_dlogit(clean_traces[rec_id]["logits"], tr_u["logits"])
            c22_unsteered_kl[cname].append({
                "pair_idx": pair_idx,
                "stem_id": tr["stem_id"],
                "kl": round(kl_v, 6),
                "max_abs_dlogit": round(mv, 6),
            })

    # 5b. Steered C2.2 conditions: capture unpatched donor activations PER (target, arm) FIRST (F11!)
    c22_cond_names = list(c2_constants["planned_budget"]["c22_conditions_list"])
    block2_c22_traces: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        cname: {str(pidx): {} for pidx in c22_pair_idxs} for cname in c22_cond_names
    }

    for ti in c22_t_idxs:
        t = targets_24[ti]
        t_str = t["string"]
        tid = int(t["id"])
        u_t = u_vectors[ti]
        pr_info = c22_pred_ret_map[t_str]
        u_dot_w = float(pr_info["u_dot_w_Rmf"])

        for aname, w_vec, pr_val in (
            ("R_mf", w_rmf_l18[ti], float(pr_info["pred_ret_R_mf"])),
            ("dm0", w_dm_l18[c22_pos_map[ti]], float(pr_info["pred_ret_dm0"])),
        ):
            # Pass 1: run all unpatched formats across pairs & cache donor suffix MLP activations for F11!
            donor_long_by_pair: dict[int, dict[int, torch.Tensor]] = {}
            donor_chat_by_pair: dict[int, dict[int, torch.Tensor]] = {}

            for pair_idx in c22_pair_idxs:
                tr = triples_12[pair_idx]
                t_s = int(tr["suffix_len"])
                for cname, fmt_key in (
                    ("unpatched_LongBare", "long_bare"),
                    ("unpatched_ChatML", "chatml"),
                    ("unpatched_ShortBare", "short_bare"),
                ):
                    pid = tr[fmt_key]["id"]
                    ct = clean_traces[pid]
                    pc, lpc = clean_probs[pid]
                    r_c = float(ct["rms_final"])
                    dose_d = 3.0 * r_c / u_dot_w
                    if dose_d > cap_d_l18:
                        fwd_counts["block2_C2_2_capped_skipped"] += 1
                        block2_c22_traces[cname][str(pair_idx)].setdefault(t_str, {})[aname] = {
                            "prompt_id": pid, "capped": True,
                            "dose_requested": round(dose_d, 6), "dose_cap": round(cap_d_l18, 6),
                        }
                        continue
                    cap_buf: dict[int, torch.Tensor] = {}
                    need_cap = fmt_key in ("long_bare", "chatml")
                    with post_norm_inject_hook(model, c22_inject_layer, w_vec, dose=dose_d, positions="all"):
                        if need_cap:
                            with capture_mlp_suffix_hook(model, l_cap_all, t_s, cap_buf):
                                tr_i = capture_last_trace(model, enc_by_id[pid])
                                fwd_counts["block2_C2_2_steered"] += 1
                        else:
                            tr_i = capture_last_trace(model, enc_by_id[pid])
                            fwd_counts["block2_C2_2_steered"] += 1
                    if fmt_key == "long_bare":
                        donor_long_by_pair[pair_idx] = cap_buf
                    elif fmt_key == "chatml":
                        donor_chat_by_pair[pair_idx] = cap_buf
                    mv = _max_abs_dlogit(ct["logits"], tr_i["logits"])
                    _note_move(f"C2.2_{cname}_{aname}", mv)
                    rec = decompose_target_trace(ct, tr_i, c22_inject_layer, u_t, tid, dose_d, pr_val, w_vec, pc, lpc)
                    rec["prompt_id"] = pid
                    rec["capped"] = False
                    rec["max_abs_dlogit"] = round(mv, 6)
                    block2_c22_traces[cname][str(pair_idx)].setdefault(t_str, {})[aname] = rec

            # Pass 2: run all 6 patched conditions using the EXACT same-arm donor activations (F11)!
            for pair_idx in c22_pair_idxs:
                tr = triples_12[pair_idx]
                t_s = int(tr["suffix_len"])
                id_long = tr["long_bare"]["id"]
                id_chat = tr["chatml"]["id"]
                mis_idx = mismatch_map[pair_idx]

                patched_specs = [
                    ("patch_LongBare_to_ChatML_MLP_L19_31", id_chat, l_p19, "replace", donor_long_by_pair.get(pair_idx), None),
                    ("patch_ChatML_to_LongBare_MLP_L19_31", id_long, l_p19, "replace", donor_chat_by_pair.get(pair_idx), None),
                    ("ctrl_L4_16_MLP_LongBare_to_ChatML", id_chat, l_p4, "replace", donor_long_by_pair.get(pair_idx), None),
                    ("ctrl_mismatch_prompt_MLP_L19_31", id_chat, l_p19, "replace", donor_long_by_pair.get(mis_idx), None),
                    ("ctrl_rand_iso_same_site_L19_31", id_chat, l_p19, "rand_iso", donor_long_by_pair.get(pair_idx), 2026112700 + pair_idx * 100 + ti * 2 + (0 if aname == "R_mf" else 1)),
                    ("ctrl_self_attenuated_L19_31", id_chat, l_p19, "self_attenuated", donor_long_by_pair.get(pair_idx), None),
                ]
                for cname, rec_id, pl, pmode, pdon, pseed in patched_specs:
                    ct = clean_traces[rec_id]
                    pc, lpc = clean_probs[rec_id]
                    r_c = float(ct["rms_final"])
                    dose_d = 3.0 * r_c / u_dot_w
                    if dose_d > cap_d_l18 or pdon is None:
                        fwd_counts["block2_C2_2_capped_skipped"] += 1
                        block2_c22_traces[cname][str(pair_idx)].setdefault(t_str, {})[aname] = {
                            "prompt_id": rec_id, "capped": True,
                            "dose_requested": round(dose_d, 6), "dose_cap": round(cap_d_l18, 6),
                        }
                        continue
                    with post_norm_inject_hook(model, c22_inject_layer, w_vec, dose=dose_d, positions="all"):
                        with patch_mlp_suffix_hook(model, pl, t_s, pmode, donor_acts=pdon, rand_seed=pseed):
                            tr_i = capture_last_trace(model, enc_by_id[rec_id])
                            fwd_counts["block2_C2_2_steered"] += 1
                    mv = _max_abs_dlogit(ct["logits"], tr_i["logits"])
                    _note_move(f"C2.2_{cname}_{aname}", mv)
                    rec = decompose_target_trace(ct, tr_i, c22_inject_layer, u_t, tid, dose_d, pr_val, w_vec, pc, lpc)
                    rec["prompt_id"] = rec_id
                    rec["capped"] = False
                    rec["max_abs_dlogit"] = round(mv, 6)
                    block2_c22_traces[cname][str(pair_idx)].setdefault(t_str, {})[aname] = rec

    # ------------------------------------------------------------------
    # 6. Block 2b (C2.3): 5-Layer Post-Erasure Sweep (24 logit_lens + 8 rand_iso x 8 prompts)
    # ------------------------------------------------------------------
    block2b_c23: dict[str, Any] = {}
    for L in p_sweep_layers:
        dose_l = 0.75 * post_mlp_norms[str(L)]
        ll_blk: dict[str, list[dict[str, Any]]] = {}
        for ti in c23_t_idxs:
            t = targets_24[ti]
            t_str = t["string"]
            tid = int(t["id"])
            u_t = u_vectors[ti]
            w_vec = u_hat[ti]
            rows = []
            for pi in c21_p_idxs:
                pid = prompts_8_c21[pi]["id"]
                ct = clean_traces[pid]
                pc, lpc = clean_probs[pid]
                with post_norm_inject_hook(model, L, w_vec, dose=dose_l, positions="last"):
                    tr_i = capture_last_trace(model, enc_by_id[pid])
                    fwd_counts["block2b_C2_3"] += 1
                mv = _max_abs_dlogit(ct["logits"], tr_i["logits"])
                _note_move(f"C2.3_L{L}_logit_lens", mv)
                rec = decompose_target_trace(ct, tr_i, L, u_t, tid, dose_l, 1.0, w_vec, pc, lpc)
                rec["prompt_id"] = pid
                rec["max_abs_dlogit"] = round(mv, 6)
                rows.append(rec)
            ll_blk[t_str] = rows

        ri_mat = bank[f"L{L}.rand_iso"]
        ri_rows: list[dict[str, Any]] = []
        for di in range(c23_rand_n):
            v_vec = ri_mat[di]
            for pi in c21_p_idxs:
                pid = prompts_8_c21[pi]["id"]
                ct = clean_traces[pid]
                with post_norm_inject_hook(model, L, v_vec, dose=dose_l, positions="last"):
                    tr_i = capture_last_trace(model, enc_by_id[pid])
                    fwd_counts["block2b_C2_3"] += 1
                mv = _max_abs_dlogit(ct["logits"], tr_i["logits"])
                _note_move(f"C2.3_L{L}_rand_iso", mv)
                rec = decompose_rand_iso_trace(ct, tr_i, L, v_vec, dose_l, target_ids_24)
                rec.pop("all_16_dlogits", None)
                rec["draw_idx"] = di
                rec["prompt_id"] = pid
                rec["rms_clean"] = round(float(ct["rms_final"]), 6)
                rec["rms_int"] = round(float(tr_i["rms_final"]), 6)
                rec["max_abs_dlogit"] = round(mv, 6)
                ri_rows.append(rec)

        block2b_c23[str(L)] = {"dose": round(dose_l, 6), "logit_lens": ll_blk, "rand_iso": ri_rows}

    # ------------------------------------------------------------------
    # 7. Block 3 (C2.4): 9 Enumerated pre_norm Arms x 24 targets x 4 prompts x 2 doses
    # ------------------------------------------------------------------
    c24_prompt_ids = [
        triples_12[0]["long_bare"]["id"],
        triples_12[0]["chatml"]["id"],
        triples_12[6]["long_bare"]["id"],
        triples_12[6]["chatml"]["id"],
    ]
    c24_prompt_ids = [c24_prompt_ids[i] for i in c24_p_idxs]
    c24_arm_keys = [
        "L24.prenorm_ff_unmasked",
        "L24.prenorm_ff_unmasked_postmatched",
        "L24.prenorm_ff_masked",
        "L24.m70a30",
        "L24.prenorm_ff_rand_masked",
        "L24.m70a30_rand_masked",
        "L21.prenorm_ff_masked",
        "L27.prenorm_ff_masked",
        "K3_combined",
    ]
    l21, l24, l27 = p_prenorm_layers[0], p_prenorm_layers[1], p_prenorm_layers[2]
    pmn21 = post_mlp_norms[str(l21)]
    pmn24 = post_mlp_norms[str(l24)]
    pmn27 = post_mlp_norms[str(l27)]
    cap_l24_pm = 3.0 * pmn24

    block3_c24: dict[str, Any] = {str(mult): {ak: {} for ak in c24_arm_keys} for mult in C24_DOSES}
    for mult in C24_DOSES:
        mkey = str(mult)
        d21 = mult * pmn21
        d24 = mult * pmn24
        d27 = mult * pmn27
        d24_pm = d24 / unmasked_eff_l24

        for ti in c24_t_idxs:
            t = targets_24[ti]
            t_str = t["string"]
            tid = int(t["id"])

            arm_injections: list[tuple[str, list[tuple[int, str, torch.Tensor, float]], float, bool]] = [
                ("L24.prenorm_ff_unmasked", [(l24, "ff", bank[f"L{l24}.prenorm_ff_unmasked"][ti], d24)], d24, False),
                ("L24.prenorm_ff_unmasked_postmatched", [(l24, "ff", bank[f"L{l24}.prenorm_ff_unmasked"][ti], d24_pm)], d24_pm, d24_pm > cap_l24_pm),
                ("L24.prenorm_ff_masked", [(l24, "ff", bank[f"L{l24}.prenorm_ff_masked"][ti], d24)], d24, False),
                ("L24.m70a30", [(l24, "ff", bank[f"L{l24}.prenorm_ff_masked"][ti], 0.7 * d24), (l24, "attn", bank[f"L{l24}.prenorm_attn_masked"][ti], 0.3 * d24)], d24, False),
                ("L24.prenorm_ff_rand_masked", [(l24, "ff", bank[f"L{l24}.prenorm_ff_rand_masked"][ti], d24)], d24, False),
                ("L24.m70a30_rand_masked", [(l24, "ff", bank[f"L{l24}.prenorm_ff_rand_masked"][ti], 0.7 * d24), (l24, "attn", bank[f"L{l24}.prenorm_attn_rand_masked"][ti], 0.3 * d24)], d24, False),
                ("L21.prenorm_ff_masked", [(l21, "ff", bank[f"L{l21}.prenorm_ff_masked"][ti], d21)], d21, False),
                ("L27.prenorm_ff_masked", [(l27, "ff", bank[f"L{l27}.prenorm_ff_masked"][ti], d27)], d27, False),
                ("K3_combined", [(l21, "ff", bank[f"L{l21}.prenorm_ff_masked"][ti], d21), (l24, "ff", bank[f"L{l24}.prenorm_ff_masked"][ti], d24), (l27, "ff", bank[f"L{l27}.prenorm_ff_masked"][ti], d27)], d21 + d24 + d27, False),
            ]
            for akey, injs, d_rep, is_cap in arm_injections:
                rows = []
                for pid in c24_prompt_ids:
                    if is_cap:
                        fwd_counts["block3_C2_4_capped_skipped"] += 1
                        rows.append({
                            "prompt_id": pid, "capped": True,
                            "dose_requested": round(d_rep, 6), "dose_cap": round(cap_l24_pm, 6),
                        })
                        continue
                    ct = clean_traces[pid]
                    with pre_norm_inject_hook(model, injs):
                        tr_i = capture_last_trace(model, enc_by_id[pid])
                        fwd_counts["block3_C2_4"] += 1
                    diff_logits = tr_i["logits"].float() - ct["logits"].float()
                    exact_dl = float(diff_logits[tid].item())
                    dl_l2 = float(diff_logits.norm().item())
                    mv = float(diff_logits.abs().max().item())
                    _note_move(f"C2.4_{mkey}_{akey}", mv)
                    kl_v = _kl_div_from_clean(ct["logits"], tr_i["logits"])
                    rows.append({
                        "prompt_id": pid,
                        "capped": False,
                        "dose": round(d_rep, 6),
                        "exact_dlogit": round(exact_dl, 6),
                        "abs_target_dlogit": round(abs(exact_dl), 6),
                        "dlogit_l2_norm": round(dl_l2, 6),
                        "rms_clean": round(float(ct["rms_final"]), 6),
                        "rms_int": round(float(tr_i["rms_final"]), 6),
                        "kl": round(kl_v, 6),
                        "max_abs_dlogit": round(mv, 6),
                    })
                block3_c24[mkey][akey][t_str] = rows

    elapsed_sec = round(time.time() - t0, 2)
    total_fwds = sum(v for k, v in fwd_counts.items() if not k.endswith("_capped_skipped"))

    frozen_cal_ref = compute_frozen_cal_reference(c2_constants, bank, r_out_all, p_c21_layers)

    return {
        "schema_version": "olmo3-stage-c2-receipt-v4",
        "meta": {
            "model": MODEL_ID,
            "revision": MODEL_REV,
            "smoke": bool(smoke),
            "device": device_name,
            "elapsed_sec": elapsed_sec,
            "total_forwards": total_fwds,
            "forward_counts": fwd_counts,
            "provenance_hashes": provenance_hashes,
        },
        "frozen_cal_reference": frozen_cal_ref,
        "post_mlp_norms": post_mlp_norms,
        "clean_rms_final": clean_rms_final,
        "shams": {"write_13": shams_write, "patch_6": shams_patch},
        "active_arm_min_max_abs_dlogit": {k: round(v, 6) for k, v in sorted(arm_min_move.items())},
        "c0_anchor_colon": c0_colon_results,
        "block1_C2_1": block1_c21,
        "block2_C2_2": {
            "inject_layer": c22_inject_layer,
            "dose_cap": round(cap_d_l18, 6),
            "mismatch_map": {str(k): v for k, v in sorted(mismatch_map.items())},
            "unsteered_patched_kl": c22_unsteered_kl,
            "traces": block2_c22_traces,
        },
        "block2b_C2_3": block2b_c23,
        "block3_C2_4": {
            "prompt_ids": c24_prompt_ids,
            "unmasked_eff_L24_ff": unmasked_eff_l24,
            "postmatched_dose_cap": round(cap_l24_pm, 6),
            "doses": block3_c24,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage C2 v4 on OLMo 3 7B (GPU; fail-closed SHA-256 preflight).")
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--r-lens", type=Path, required=True)
    ap.add_argument("--c2-constants", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (scorer will refuse smoke receipts).")
    args = ap.parse_args()

    hashes = verify_preflight_provenance(args.bank, args.r_lens, args.c2_constants, args.design)
    c2_constants = json.loads(args.c2_constants.read_text(encoding="utf-8"))
    bank = load_c2_bank(args.bank)
    rl = torch.load(str(args.r_lens), map_location="cpu", weights_only=False)
    r_out_all = rl["means"]["out"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        receipt = run_stage_c2_experiment(
            model, tok, c2_constants, bank, hashes, r_out_all=r_out_all, smoke=args.smoke, device_name=dev_name
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(receipt, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    if args.out.suffix == ".gz":
        with gzip.open(args.out, "wb", compresslevel=6) as f:
            f.write(raw)
    else:
        args.out.write_bytes(raw)
    print(f"Wrote {args.out} ({len(raw)} uncompressed bytes, sha256={sha256_file(args.out)}, total_fwds={receipt['meta']['total_forwards']})")


if __name__ == "__main__":
    main()
