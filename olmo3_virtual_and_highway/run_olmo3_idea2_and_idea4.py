#!/usr/bin/env python3
"""OLMo 3 7B Idea #2 (Hand-Built Virtual Concept & Slot SwiGLU Neurons) + Idea #4
(Exact L+1..31 Signal Highway & Late Suppression Cliff Decomposition) runner.

Methodological guardrails integrated from 2023-2026 literature:
1. Tests BOTH injection sites at every layer (BlackboxNLP 2026):
   - `pre_norm`: injects `act * w_down_prenorm` into `mlp.down_proj` output BEFORE
     `post_feedforward_layernorm` (exact mathematical equivalent of appending a real
     SwiGLU neuron `(w_gate, w_up, w_down)` to `mlp.{gate,up,down}_proj.weight`).
   - `post_norm`: injects `act * w_eff` AFTER `post_feedforward_layernorm` (pure residual
     transport without post-MLP RMSNorm scale-invariance damping).
2. Benchmarks 4 zero-training write geometries head-to-head:
   `R_ridge` ((R^T R + lam I)^-1 R^T u_t) vs `R_mf` (R^T u_t) vs `logit_lens` (u_t) vs `random`.
3. Contrasts Virtual Concept (`concept_S_ridge`, `concept_S_mf`) vs Virtual Slot
   (`slot_S_ridge` pure lens, `slot_contrastive` held-out calibration residual) with strict
   disjoint calibration/evaluation prompt splits.
4. Decomposes 100.000% of exact `Δlogit(t)` across `L+1..31` into:
   `direct + sum(Attn_l) + sum(MLP_l) + final_rmsnorm_shift`, verifying exact telescoping
   identity on every trace, plus `ResLRP` path-cancellation ratio, frozen-downstream
   Direct Effect, and `sliding_attention` vs `full_attention` layer stratification.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prompts import (
    CAL_OFFTARGET,
    CAL_SELF_QUERIES,
    EVAL_OFFTARGET,
    EVAL_SELF_QUERIES,
    MODEL_ID,
    MODEL_REV,
    REGISTER_FRAMES,
    concept_chat_prompt,
    concept_trigger_prompt,
    format_chat,
    verify_disjoint_splits,
)


def _norm_unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-8)


def make_prenorm_wdown(w_write_unit: torch.Tensor, gamma_post_ff: torch.Tensor) -> torch.Tensor:
    """Convert desired unit residual write direction `w_write` into a unit pre-RMSNorm
    `w_down` direction whose elementwise post-RMSNorm projection `gamma * w_down` aligns
    with `w_write` (preserving exact RMSNorm denominator interaction)."""
    raw = w_write_unit.float() / gamma_post_ff.float().clamp_min(1e-4)
    return _norm_unit(raw)


class VirtualSwiGLUSpec:
    """Specification of a hand-built (or unconditional) virtual SwiGLU neuron at layer L."""

    def __init__(
        self,
        layer: int,
        site: str,  # "pre_norm" or "post_norm"
        w_write_unit: torch.Tensor,  # [d] float32 unit vector in target site coordinates
        dose: float,
        mode: str = "unconditional",  # "unconditional" or "swiglu"
        w_gate: torch.Tensor | None = None,  # [d] float32
        w_up: torch.Tensor | None = None,    # [d] float32
        gate_bias: float = 0.0,
        up_bias: float = 0.0,
        gate_scale: float = 1.0,
        up_scale: float = 1.0,
        positions: str = "all",  # "all" or "last"
    ):
        assert site in ("pre_norm", "post_norm")
        assert mode in ("unconditional", "swiglu")
        self.layer = layer
        self.site = site
        self.w_write_unit = w_write_unit.float()
        self.dose = float(dose)
        self.mode = mode
        self.w_gate = w_gate.float() if w_gate is not None else None
        self.w_up = w_up.float() if w_up is not None else None
        self.gate_bias = float(gate_bias)
        self.up_bias = float(up_bias)
        self.gate_scale = float(gate_scale)
        self.up_scale = float(up_scale)
        self.positions = positions


@contextlib.contextmanager
def virtual_neuron_hook(model, spec: VirtualSwiGLUSpec, act_log: list[float] | None = None):
    """Attach a zero-overhead virtual SwiGLU neuron (or unconditional write) at `spec.layer`."""
    layer_mod = model.model.layers[spec.layer]
    cache: dict[str, torch.Tensor] = {}

    def capture_h_mid(_mod, args):
        # Input to MLP is h_mid (post-attention residual stream)
        cache["h_mid"] = args[0]

    def compute_delta(ref_tensor: torch.Tensor) -> torch.Tensor:
        dev = ref_tensor.device
        dt = ref_tensor.dtype
        w_w = spec.w_write_unit.to(device=dev, dtype=torch.float32)
        if spec.dose == 0.0:
            if act_log is not None:
                act_log.append(0.0)
            return torch.zeros_like(ref_tensor)
        if spec.mode == "unconditional":
            delta = torch.zeros_like(ref_tensor, dtype=torch.float32)
            if spec.positions == "last":
                delta[:, -1, :] = spec.dose * w_w
            else:
                delta[:] = spec.dose * w_w
            if act_log is not None:
                act_log.append(spec.dose)
            return delta.to(dt)
        # True SwiGLU virtual neuron: silu(scale_g * (h_mid . w_gate) - bias_g) * (scale_u * (h_mid . w_up) - bias_u)
        h_mid = cache["h_mid"].float()
        wg = spec.w_gate.to(device=dev, dtype=torch.float32)
        wu = spec.w_up.to(device=dev, dtype=torch.float32)
        pre_g = spec.gate_scale * (h_mid @ wg) - spec.gate_bias
        pre_u = spec.up_scale * (h_mid @ wu) - spec.up_bias
        act = F.silu(pre_g) * pre_u  # [B, T]
        if spec.positions == "last":
            mask = torch.zeros_like(act)
            mask[:, -1] = 1.0
            act = act * mask
        if act_log is not None:
            act_log.append(float(act[:, -1].mean().item()))
        return (spec.dose * act.unsqueeze(-1) * w_w).to(dt)

    h1 = layer_mod.mlp.register_forward_pre_hook(capture_h_mid)
    if spec.site == "pre_norm":
        def add_prenorm(_mod, _args, output):
            return output + compute_delta(output)
        h2 = layer_mod.mlp.down_proj.register_forward_hook(add_prenorm)
    else:
        def add_postnorm(_mod, _args, output):
            return output + compute_delta(output)
        h2 = layer_mod.post_feedforward_layernorm.register_forward_hook(add_postnorm)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()


@contextlib.contextmanager
def natural_neuron_push_hook(model, layer: int, neuron_idx: int, delta_act: float):
    """Push a real OLMo 3 MLP neuron's post-SwiGLU activation by +delta_act at all positions."""
    layer_mod = model.model.layers[layer]

    def push_down_in(_mod, args):
        if delta_act == 0.0:
            return args
        x = args[0].clone()
        x[..., neuron_idx] = x[..., neuron_idx] + delta_act
        return (x,) + args[1:]

    h = layer_mod.mlp.down_proj.register_forward_pre_hook(push_down_in)
    try:
        yield
    finally:
        h.remove()


def capture_highway_trace(model, input_ids: torch.Tensor, position: int = -1) -> dict:
    """Run one forward pass capturing exact post-norm Attn_l, MLP_l increments and final RMSNorm."""
    n_layers = model.config.num_hidden_layers
    attn_inc: list[torch.Tensor | None] = [None] * n_layers
    mlp_inc: list[torch.Tensor | None] = [None] * n_layers
    holder: dict[str, torch.Tensor] = {}
    handles = []

    def cap_emb(_mod, _args, out):
        holder["h0"] = out[:, position, :].detach().float().cpu()

    handles.append(model.model.embed_tokens.register_forward_hook(cap_emb))
    for L in range(n_layers):
        lay = model.model.layers[L]

        def mk_a(idx):
            def fn(_m, _a, out):
                attn_inc[idx] = out[:, position, :].detach().float().cpu()
            return fn

        def mk_m(idx):
            def fn(_m, _a, out):
                mlp_inc[idx] = out[:, position, :].detach().float().cpu()
            return fn

        handles.append(lay.post_attention_layernorm.register_forward_hook(mk_a(L)))
        handles.append(lay.post_feedforward_layernorm.register_forward_hook(mk_m(L)))

    def cap_final_in(_mod, args):
        holder["h_final"] = args[0][:, position, :].detach().float().cpu()

    handles.append(model.model.norm.register_forward_pre_hook(cap_final_in))
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[:, position, :].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()

    eps = float(getattr(model.model.norm, "variance_epsilon", 1e-6))
    h_f = holder["h_final"]
    rms_f = torch.sqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + eps)  # [B, 1]
    return {
        "h0": holder["h0"],
        "attn": attn_inc,
        "mlp": mlp_inc,
        "h_final": h_f,
        "rms_final": rms_f,
        "logits": logits,
    }


def exact_highway_decompose(
    clean: dict,
    intervened: dict,
    inject_layer: int,
    u_target: torch.Tensor,  # [d] = lm_head[tid] * final_norm.weight (float32, CPU)
    target_id: int,
    layer_types: list[str],
    eps: float,
    b_idx: int = 0,
) -> dict:
    """Exact 100.000% 4-term telescoping decomposition of Δlogit(t) across layers L+1..N-1.

    Let `h_c = h_final_clean`, `h_i = h_final_int`, `r_c = rms_final_clean`, `r_i = rms_final_int`.
    Then `logit_i - logit_c = (u . h_i)/r_i - (u . h_c)/r_c`
                            = `(u . (h_i - h_c))/r_c + (u . h_i) * (1/r_i - 1/r_c)`.
    Since `h_i - h_c = Δm_L + sum_{l=L+1}^{N-1} (Δa_l + Δm_l)` (for an MLP intervention at layer L),
    every single logit bit is accounted for with ZERO unattributed residual!
    """
    u = u_target.float()
    h_c = clean["h_final"][b_idx]
    h_i = intervened["h_final"][b_idx]
    r_c = float(clean["rms_final"][b_idx, 0])
    r_i = float(intervened["rms_final"][b_idx, 0])

    exact_dlogit = float(intervened["logits"][b_idx, target_id] - clean["logits"][b_idx, target_id])

    # Direct write from layer L's MLP (note: if intervention is at layer L MLP, Δh_L = Δm_L)
    dm_L = intervened["mlp"][inject_layer][b_idx] - clean["mlp"][inject_layer][b_idx]
    direct_lin = float((u @ dm_L) / r_c)

    # Exact frozen-downstream Direct Effect: add dm_L directly to h_c, recompute final RMSNorm!
    h_frozen = h_c + dm_L
    r_frozen = math.sqrt(float(h_frozen.pow(2).mean().item()) + eps)
    frozen_direct_effect = float((u @ h_frozen) / r_frozen - (u @ h_c) / r_c)

    n_layers = len(clean["attn"])
    per_layer = []
    sum_attn = 0.0
    sum_mlp = 0.0
    sum_attn_sliding = 0.0
    sum_attn_full = 0.0
    abs_sum = abs(direct_lin)

    for l in range(inject_layer + 1, n_layers):
        da = intervened["attn"][l][b_idx] - clean["attn"][l][b_idx]
        dm = intervened["mlp"][l][b_idx] - clean["mlp"][l][b_idx]
        a_val = float((u @ da) / r_c)
        m_val = float((u @ dm) / r_c)
        ltype = layer_types[l]
        sum_attn += a_val
        sum_mlp += m_val
        if ltype == "full_attention":
            sum_attn_full += a_val
        else:
            sum_attn_sliding += a_val
        abs_sum += abs(a_val) + abs(m_val)
        per_layer.append({
            "layer": l,
            "layer_type": ltype,
            "attn": round(a_val, 5),
            "mlp": round(m_val, 5),
            "total": round(a_val + m_val, 5),
            "da_norm": round(float(da.norm()), 5),
            "dm_norm": round(float(dm.norm()), 5),
        })

    # Explicit Final-RMSNorm denominator-shift term
    rmsnorm_shift = float((u @ h_i) * (1.0 / r_i - 1.0 / r_c))
    abs_sum += abs(rmsnorm_shift)

    # Also check any tiny pre-L residual mismatch (should be exact 0.0)
    h_recon_diff = (h_i - h_c) - (
        dm_L + sum(
            (intervened["attn"][l][b_idx] - clean["attn"][l][b_idx])
            + (intervened["mlp"][l][b_idx] - clean["mlp"][l][b_idx])
            for l in range(inject_layer + 1, n_layers)
        )
    )
    recon_dlogit = direct_lin + sum_attn + sum_mlp + rmsnorm_shift
    telescoping_err = abs(exact_dlogit - recon_dlogit)
    cancel_ratio = abs_sum / max(abs(exact_dlogit), 1e-6)

    # Late suppression cliff summary (last 3 layers: N-3..N-1)
    late_cut = max(inject_layer + 1, n_layers - 3)
    late_attn = sum(r["attn"] for r in per_layer if r["layer"] >= late_cut)
    late_mlp = sum(r["mlp"] for r in per_layer if r["layer"] >= late_cut)

    return {
        "exact_dlogit": round(exact_dlogit, 5),
        "recon_dlogit": round(recon_dlogit, 5),
        "telescoping_err": round(telescoping_err, 6),
        "h_recon_l2_err": round(float(h_recon_diff.norm()), 6),
        "direct_linear": round(direct_lin, 5),
        "frozen_direct_effect": round(frozen_direct_effect, 5),
        "sum_downstream_attn": round(sum_attn, 5),
        "sum_downstream_attn_sliding": round(sum_attn_sliding, 5),
        "sum_downstream_attn_full": round(sum_attn_full, 5),
        "sum_downstream_mlp": round(sum_mlp, 5),
        "rmsnorm_shift": round(rmsnorm_shift, 5),
        "late3_attn": round(late_attn, 5),
        "late3_mlp": round(late_mlp, 5),
        "path_cancellation_ratio": round(cancel_ratio, 3),
        "rms_clean": round(r_c, 5),
        "rms_int": round(r_i, 5),
        "dm_L_norm": round(float(dm_L.norm()), 5),
        "per_layer": per_layer,
    }


def logit_metrics(logits_clean: torch.Tensor, logits_int: torch.Tensor, target_id: int) -> dict:
    """Compute exact target Δlogit, z-scored Δlogit (background specificity), Δprob, rank, and KL."""
    lc = logits_clean.float()
    li = logits_int.float()
    dl = li - lc
    d_tgt = float(dl[target_id].item())
    dl_mean = float(dl.mean().item())
    dl_std = float(dl.std().clamp_min(1e-8).item())
    z_dl = (d_tgt - dl_mean) / dl_std

    pc = F.softmax(lc, dim=-1)
    pi = F.softmax(li, dim=-1)
    log_pc = F.log_softmax(lc, dim=-1)
    log_pi = F.log_softmax(li, dim=-1)
    kl = float((pc * (log_pc - log_pi)).sum().item())
    rank_c = int((lc > lc[target_id]).sum().item()) + 1
    rank_i = int((li > li[target_id]).sum().item()) + 1
    return {
        "dlogit": round(d_tgt, 4),
        "dlogit_centered": round(d_tgt - dl_mean, 4),
        "dlogit_z": round(z_dl, 3),
        "p_clean": round(float(pc[target_id].item()), 6),
        "p_int": round(float(pi[target_id].item()), 6),
        "rank_clean": rank_c,
        "rank_int": rank_i,
        "kl_clean_int": round(kl, 5),
        "top1_int_id": int(li.argmax().item()),
    }


def collect_h_mid_last(model, tokenizer, prompts: list[str], layers: list[int], device: str) -> dict[int, torch.Tensor]:
    """Collect last-token `h_mid` (MLP input) at each layer for a list of prompts."""
    out = {L: [] for L in layers}
    for p in prompts:
        ids = tokenizer.encode(p, return_tensors="pt").to(device)
        captured = {}
        handles = []
        for L in layers:
            def mk(idx):
                def fn(_m, args):
                    captured[idx] = args[0][0, -1, :].detach().float().cpu()
                return fn
            handles.append(model.model.layers[L].mlp.register_forward_pre_hook(mk(L)))
        try:
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        for L in layers:
            out[L].append(captured[L])
    return {L: torch.stack(v, dim=0) for L, v in out.items()}


def greedy_generate_with_spec(model, tokenizer, prompt: str, spec: VirtualSwiGLUSpec | None, max_new_tokens: int, device: str) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    cm = virtual_neuron_hook(model, spec) if spec is not None else contextlib.nullcontext()
    with cm, torch.inference_mode():
        gen = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)


def run_experiment(
    model,
    tokenizer,
    bank: dict,
    device: str,
    max_new_tokens: int = 24,
    smoke: bool = False,
) -> dict:
    verify_disjoint_splits()
    t0 = time.time()
    layer_types = list(model.config.layer_types)
    eps = float(getattr(model.model.norm, "variance_epsilon", 1e-6))
    Wu = (model.lm_head.weight.detach().float() * model.model.norm.weight.detach().float()[None, :]).cpu()

    layers = bank["injection_layers"][:2] if smoke else bank["injection_layers"]
    personas = bank["target_personas"][:2] if smoke else bank["target_personas"]
    write_methods = ["R_ridge", "R_mf", "logit_lens", "random"]
    sites = ["pre_norm", "post_norm"]

    # 1. Measure baseline residual/MLP norms on calibration prompts + compute zero-training
    #    held-out contrastive read vectors `slot_contrastive` and SwiGLU gate thresholds!
    cal_self_prompts = [format_chat(q) for q in CAL_SELF_QUERIES]
    cal_off_prompts = [format_chat(q) for q in CAL_OFFTARGET]
    h_cal_self = collect_h_mid_last(model, tokenizer, cal_self_prompts, layers, device)
    h_cal_off = collect_h_mid_last(model, tokenizer, cal_off_prompts, layers, device)

    # Also measure clean post-MLP residual norm & pre-norm MLP output norm at each layer
    cal_h_norms = {}
    slot_contrastive = {}
    slot_cal_stats = {}
    for L in layers:
        h_s = h_cal_self[L]  # [N_s, d]
        h_o = h_cal_off[L]   # [N_o, d]
        cal_h_norms[L] = round(float(torch.cat([h_s, h_o], dim=0).norm(dim=-1).mean()), 4)
        diff = _norm_unit(h_s.mean(0) - h_o.mean(0))
        slot_contrastive[L] = diff
        # Calibration projections for threshold selection (midpoint between off-target max and self-query mean)
        p_s = h_s @ diff
        p_o = h_o @ diff
        # Pure S_mid slot read calibration stats
        w_s_rg = bank["layers"][L]["reads"]["slot_S_ridge"].float()
        ps_s = h_s @ w_s_rg
        ps_o = h_o @ w_s_rg
        slot_cal_stats[L] = {
            "contrastive_self_mean": round(float(p_s.mean()), 4),
            "contrastive_off_mean": round(float(p_o.mean()), 4),
            "contrastive_off_max": round(float(p_o.max()), 4),
            "contrastive_bias": round(float(0.5 * (p_o.max() + p_s.mean())), 4),
            "slot_S_self_mean": round(float(ps_s.mean()), 4),
            "slot_S_off_mean": round(float(ps_o.mean()), 4),
            "slot_S_off_max": round(float(ps_o.max()), 4),
            "slot_S_bias": round(float(0.5 * (ps_o.max() + ps_s.mean())), 4),
        }

    # Baseline clean logits on REGISTER_FRAMES + EVAL prompts
    eval_self_chat = [format_chat(q) for q in EVAL_SELF_QUERIES]
    eval_off_chat = [format_chat(q) for q in EVAL_OFFTARGET]
    eval_self_tf = [format_chat(q, assistant_prefix="I am a") for q in EVAL_SELF_QUERIES]

    def clean_logits(prompt: str) -> torch.Tensor:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            return model(input_ids=ids, use_cache=False).logits[0, -1, :].detach().float().cpu()

    clean_reg = {k: clean_logits(p) for k, p in REGISTER_FRAMES.items()}
    clean_self_tf = [clean_logits(p) for p in eval_self_tf]
    clean_off_chat = [clean_logits(p) for p in eval_off_chat]
    clean_concept = {w: clean_logits(concept_trigger_prompt(w)) for w in personas}

    # Measure clean pre-norm MLP output RMS and post-norm MLP output RMS at each layer on WHO_CHAT_TF
    who_tf_ids = tokenizer.encode(REGISTER_FRAMES["WHO_CHAT_TF"], return_tensors="pt").to(device)
    layer_mlp_scales = {}
    for L in layers:
        lay = model.model.layers[L]
        cap = {}
        h1 = lay.mlp.down_proj.register_forward_hook(lambda _m, _a, o: cap.setdefault("pre", o[0, -1].detach().float().norm().item()))
        h2 = lay.post_feedforward_layernorm.register_forward_hook(lambda _m, _a, o: cap.setdefault("post", o[0, -1].detach().float().norm().item()))
        with torch.inference_mode():
            model(input_ids=who_tf_ids, use_cache=False)
        h1.remove()
        h2.remove()
        layer_mlp_scales[L] = {"pre_mlp_norm": round(cap["pre"], 4), "post_mlp_norm": round(cap["post"], 4)}

    # =========================================================================
    # EXPERIMENT 2A: UNCONDITIONAL WRITE-GEOMETRY BENCHMARK
    # R_ridge vs R_mf vs logit_lens vs random x pre_norm vs post_norm x 3 doses
    # =========================================================================
    print("Running Experiment 2A: Write-Geometry Benchmark (R_ridge vs R_mf vs logit_lens vs random) ...")
    dose_mults = [0.25, 0.75, 1.5] if not smoke else [0.5, 1.5]
    exp2a_rows = []
    exp2a_gens = []

    reg_encoded = {k: tokenizer.encode(p, return_tensors="pt").to(device) for k, p in REGISTER_FRAMES.items()}

    for L in layers:
        lbank = bank["layers"][L]
        g_pf = lbank["gamma_post_ff"].float()
        pre_base = layer_mlp_scales[L]["pre_mlp_norm"]
        post_base = layer_mlp_scales[L]["post_mlp_norm"]
        for w in personas:
            tid = bank["target_ids"][w]
            for method in write_methods:
                w_post = lbank["writes"][w][method].float()
                w_pre = make_prenorm_wdown(w_post, g_pf)
                for site in sites:
                    w_vec = w_pre if site == "pre_norm" else w_post
                    base_scale = pre_base if site == "pre_norm" else post_base
                    for mult in dose_mults:
                        dose = mult * base_scale
                        spec = VirtualSwiGLUSpec(
                            layer=L, site=site, w_write_unit=w_vec, dose=dose, mode="unconditional", positions="all"
                        )
                        frame_res = {}
                        with virtual_neuron_hook(model, spec), torch.inference_mode():
                            for fk, ids in reg_encoded.items():
                                li = model(input_ids=ids, use_cache=False).logits[0, -1, :].detach().float().cpu()
                                m = logit_metrics(clean_reg[fk], li, tid)
                                m["top1_int_str"] = tokenizer.decode([m["top1_int_id"]])
                                frame_res[fk] = m
                        exp2a_rows.append({
                            "layer": L, "target": w, "target_id": tid, "method": method,
                            "site": site, "dose_mult": mult, "dose": round(dose, 4),
                            "frames": frame_res,
                        })
                    # Greedy generation on "Who are you?" at peak dose (first 4 personas to stay compact)
                    if w in personas[:4]:
                        peak_dose = dose_mults[-1] * base_scale
                        spec_g = VirtualSwiGLUSpec(
                            layer=L, site=site, w_write_unit=w_vec, dose=peak_dose, mode="unconditional", positions="all"
                        )
                        gen_txt = greedy_generate_with_spec(
                            model, tokenizer, format_chat("Who are you?"), spec_g, max_new_tokens, device
                        )
                        exp2a_gens.append({
                            "layer": L, "target": w, "method": method, "site": site,
                            "dose_mult": dose_mults[-1], "generation": gen_txt,
                        })

    # =========================================================================
    # EXPERIMENT 2B: HAND-BUILT INPUT-CONDITIONED VIRTUAL SwiGLU NEURONS
    # Virtual Concept (concept_S_ridge, concept_S_mf) vs Virtual Slot (slot_S_ridge, slot_contrastive)
    # =========================================================================
    print("Running Experiment 2B: Hand-Built Virtual Concept & Slot SwiGLU Neurons ...")
    exp2b_rows = []
    exp2b_gens = []
    eval_self_tf_ids = [tokenizer.encode(p, return_tensors="pt").to(device) for p in eval_self_tf]
    eval_off_ids = [tokenizer.encode(p, return_tensors="pt").to(device) for p in eval_off_chat]
    eval_concept_ids = {w: tokenizer.encode(concept_trigger_prompt(w), return_tensors="pt").to(device) for w in personas}

    # Use 2 representative depths (L12, L18) or all 4? All 4 layers x 4 personas is super fast (~20s on L4/A100)!
    v2b_personas = personas[:2] if smoke else personas[:4]
    v2b_methods = ["R_ridge", "R_mf", "logit_lens", "random"]

    for L in layers:
        lbank = bank["layers"][L]
        g_pf = lbank["gamma_post_ff"].float()
        pre_base = layer_mlp_scales[L]["pre_mlp_norm"]
        post_base = layer_mlp_scales[L]["post_mlp_norm"]
        cstats = slot_cal_stats[L]

        # Calibrate concept_S_ridge & concept_S_mf biases on CAL_OFFTARGET (zero leakage!)
        h_off_cal = h_cal_off[L]
        for w in v2b_personas:
            tid = bank["target_ids"][w]
            # 4 virtual neuron architectures:
            # 1) concept_S_ridge (pure S_ridge read -> R write)
            # 2) concept_S_mf    (pure S_mf read -> R write)
            # 3) slot_S_ridge    (pure S_ridge self-query read -> R write)
            # 4) slot_contrastive (held-out cal contrastive read -> R write)
            rc_rg = lbank["reads"]["concept_S_ridge"][w].float()
            rc_mf = lbank["reads"]["concept_S_mf"][w].float()
            rs_rg = lbank["reads"]["slot_S_ridge"].float()
            rs_ct = slot_contrastive[L].float()

            # Compute off-target max projection on CAL_OFFTARGET for clean gate thresholding
            # and on-target calibration scale so peak SwiGLU activation ~ 1.0 when triggered!
            h_conc_ref = collect_h_mid_last(model, tokenizer, [concept_chat_prompt(w)], [L], device)[L][0]

            def build_gate_params(w_read: torch.Tensor, on_val: float, off_mean: float, off_max: float):
                # Center `up` at off_mean so triggered states always have strictly positive up-branch!
                g_bias = off_max + 0.25 * max(on_val - off_max, 0.5)
                u_bias = off_mean
                raw_act = float(F.silu(torch.tensor(on_val - g_bias)) * max(on_val - u_bias, 0.5))
                norm_scale = 1.0 / max(abs(raw_act), 0.25)
                return g_bias, u_bias, norm_scale

            c_rg_gb, c_rg_ub, c_rg_scale = build_gate_params(
                rc_rg, float(h_conc_ref @ rc_rg), float((h_off_cal @ rc_rg).mean()), float((h_off_cal @ rc_rg).max())
            )
            c_mf_gb, c_mf_ub, c_mf_scale = build_gate_params(
                rc_mf, float(h_conc_ref @ rc_mf), float((h_off_cal @ rc_mf).mean()), float((h_off_cal @ rc_mf).max())
            )
            s_rg_gb, s_rg_ub, s_rg_scale = build_gate_params(
                rs_rg, cstats["slot_S_self_mean"], cstats["slot_S_off_mean"], cstats["slot_S_off_max"]
            )
            s_ct_gb, s_ct_ub, s_ct_scale = build_gate_params(
                rs_ct, cstats["contrastive_self_mean"], cstats["contrastive_off_mean"], cstats["contrastive_off_max"]
            )

            neuron_variants = [
                ("virtual_concept_S_ridge", rc_rg, c_rg_gb, c_rg_ub, c_rg_scale),
                ("virtual_concept_S_mf",    rc_mf, c_mf_gb, c_mf_ub, c_mf_scale),
                ("virtual_slot_S_ridge",    rs_rg, s_rg_gb, s_rg_ub, s_rg_scale),
                ("virtual_slot_contrastive", rs_ct, s_ct_gb, s_ct_ub, s_ct_scale),
            ]

            for nkind, w_read, g_bias, u_bias, amp_scale in neuron_variants:
                for method in v2b_methods:
                    # Only run full 4-method comparison on primary variants; random/mf/ll on all!
                    if nkind == "virtual_concept_S_mf" and method != "R_ridge":
                        continue
                    w_post = lbank["writes"][w][method].float()
                    w_pre = make_prenorm_wdown(w_post, g_pf)
                    for site in sites:
                        w_vec = w_pre if site == "pre_norm" else w_post
                        base_scale = pre_base if site == "pre_norm" else post_base
                        dose = 1.25 * base_scale * amp_scale
                        spec = VirtualSwiGLUSpec(
                            layer=L, site=site, w_write_unit=w_vec, dose=dose,
                            mode="swiglu", w_gate=w_read, w_up=w_read,
                            gate_bias=g_bias, up_bias=u_bias, gate_scale=1.0, up_scale=1.0, positions="all",
                        )

                        # Evaluate on (a) EVAL_SELF_QUERIES (teacher-forced "I am a"), (b) EVAL_OFFTARGET, (c) concept trigger, (d) REGISTER_FRAMES
                        self_acts, self_dlogits, self_kls = [], [], []
                        off_acts, off_kls = [], []
                        conc_acts = []
                        reg_metrics = {}

                        with torch.inference_mode():
                            for idx_q, ids in enumerate(eval_self_tf_ids):
                                alog: list[float] = []
                                with virtual_neuron_hook(model, spec, act_log=alog):
                                    li = model(input_ids=ids, use_cache=False).logits[0, -1, :].detach().float().cpu()
                                m = logit_metrics(clean_self_tf[idx_q], li, tid)
                                self_acts.append(alog[-1])
                                self_dlogits.append(m["dlogit"])
                                self_kls.append(m["kl_clean_int"])

                            for idx_o, ids in enumerate(eval_off_ids):
                                alog = []
                                with virtual_neuron_hook(model, spec, act_log=alog):
                                    li = model(input_ids=ids, use_cache=False).logits[0, -1, :].detach().float().cpu()
                                m = logit_metrics(clean_off_chat[idx_o], li, tid)
                                off_acts.append(alog[-1])
                                off_kls.append(m["kl_clean_int"])

                            alog = []
                            with virtual_neuron_hook(model, spec, act_log=alog):
                                li_c = model(input_ids=eval_concept_ids[w], use_cache=False).logits[0, -1, :].detach().float().cpu()
                            conc_m = logit_metrics(clean_concept[w], li_c, tid)
                            conc_acts.append(alog[-1])

                            for fk, ids in reg_encoded.items():
                                alog = []
                                with virtual_neuron_hook(model, spec, act_log=alog):
                                    li_r = model(input_ids=ids, use_cache=False).logits[0, -1, :].detach().float().cpu()
                                rm = logit_metrics(clean_reg[fk], li_r, tid)
                                rm["act"] = round(alog[-1], 4)
                                reg_metrics[fk] = rm

                        mean_sa = sum(self_acts) / len(self_acts)
                        mean_oa = sum(off_acts) / len(off_acts)
                        exp2b_rows.append({
                            "layer": L, "target": w, "neuron_kind": nkind, "method": method, "site": site,
                            "gate_bias": round(g_bias, 4), "dose": round(dose, 4),
                            "self_mean_act": round(mean_sa, 4),
                            "self_mean_dlogit": round(sum(self_dlogits) / len(self_dlogits), 4),
                            "self_mean_kl": round(sum(self_kls) / len(self_kls), 5),
                            "off_mean_act": round(mean_oa, 4),
                            "off_mean_kl": round(sum(off_kls) / len(off_kls), 5),
                            "concept_act": round(conc_acts[0], 4),
                            "concept_dlogit": conc_m["dlogit"],
                            "concept_dlogit_z": conc_m["dlogit_z"],
                            "selectivity_self_vs_off": round(mean_sa / max(abs(mean_oa), 1e-4), 2),
                            "selectivity_conc_vs_off": round(conc_acts[0] / max(abs(mean_oa), 1e-4), 2),
                            "register_frames": reg_metrics,
                        })

                        # Greedy generation on BOTH "Who are you?" AND "What is the capital of France?"
                        # for R_ridge and logit_lens on slot_contrastive / slot_S_ridge / concept_S_ridge
                        if w in v2b_personas[:2] and method in ("R_ridge", "logit_lens") and nkind != "virtual_concept_S_mf":
                            g_who = greedy_generate_with_spec(model, tokenizer, format_chat("Who are you?"), spec, max_new_tokens, device)
                            g_fra = greedy_generate_with_spec(model, tokenizer, format_chat("What is the capital of France?"), spec, max_new_tokens, device)
                            exp2b_gens.append({
                                "layer": L, "target": w, "neuron_kind": nkind, "method": method, "site": site,
                                "gen_who_are_you": g_who, "gen_capital_of_france": g_fra,
                            })

    # =========================================================================
    # EXPERIMENT 4: EXACT L+1..31 SIGNAL HIGHWAY & LATE SUPPRESSION CLIFF
    # =========================================================================
    print("Running Experiment 4: Exact L+1..31 Signal Highway & Late Suppression Cliff ...")
    exp4_natural = []
    exp4_virtual = []

    clean_traces = {fk: capture_highway_trace(model, ids) for fk, ids in reg_encoded.items()}

    # 4A: All 12 Natural Concept & Slot Neurons from our zero-GPU census
    nat_list = bank["natural_neurons"][:2] if smoke else bank["natural_neurons"]
    for nrec in nat_list:
        L = nrec["layer"]
        ni = nrec["neuron"]
        tid = nrec["target_id"]
        u_t = Wu[tid]
        # Clean calibrated push (+10.0 on SwiGLU activation, roughly 2-3x typical strong firing)
        push_val = 10.0
        frame_traces = {}
        with natural_neuron_push_hook(model, L, ni, push_val):
            for fk, ids in reg_encoded.items():
                tr_int = capture_highway_trace(model, ids)
                dec = exact_highway_decompose(clean_traces[fk], tr_int, L, u_t, tid, layer_types, eps)
                frame_traces[fk] = dec
        exp4_natural.append({
            **nrec, "push_act": push_val, "frames": frame_traces,
        })

    # 4B: Virtual Neurons (R_ridge vs R_mf vs logit_lens across L6, L12, L18, L24, post_norm + pre_norm)
    h4_personas = personas[:1] if smoke else personas[:4]
    for L in layers:
        lbank = bank["layers"][L]
        g_pf = lbank["gamma_post_ff"].float()
        pre_base = layer_mlp_scales[L]["pre_mlp_norm"]
        post_base = layer_mlp_scales[L]["post_mlp_norm"]
        for w in h4_personas:
            tid = bank["target_ids"][w]
            u_t = Wu[tid]
            for method in ("R_ridge", "R_mf", "logit_lens"):
                w_post = lbank["writes"][w][method].float()
                w_pre = make_prenorm_wdown(w_post, g_pf)
                for site in sites:
                    w_vec = w_pre if site == "pre_norm" else w_post
                    dose = 0.75 * (pre_base if site == "pre_norm" else post_base)
                    spec = VirtualSwiGLUSpec(layer=L, site=site, w_write_unit=w_vec, dose=dose, mode="unconditional", positions="all")
                    frame_traces = {}
                    with virtual_neuron_hook(model, spec):
                        for fk, ids in reg_encoded.items():
                            tr_int = capture_highway_trace(model, ids)
                            dec = exact_highway_decompose(clean_traces[fk], tr_int, L, u_t, tid, layer_types, eps)
                            frame_traces[fk] = dec
                    exp4_virtual.append({
                        "layer": L, "target": w, "target_id": tid, "method": method, "site": site,
                        "dose": round(dose, 4), "frames": frame_traces,
                    })

    # Clean baseline generations for comparison
    clean_gens = {
        "who_are_you": greedy_generate_with_spec(model, tokenizer, format_chat("Who are you?"), None, max_new_tokens, device),
        "capital_of_france": greedy_generate_with_spec(model, tokenizer, format_chat("What is the capital of France?"), None, max_new_tokens, device),
    }

    # Check maximum telescoping error across ALL Experiment 4 traces!
    all_errs = []
    for rec in exp4_natural + exp4_virtual:
        for fk, d in rec["frames"].items():
            all_errs.append(d["telescoping_err"])
    max_tele_err = max(all_errs) if all_errs else 0.0

    elapsed = round(time.time() - t0, 2)
    return {
        "meta": {
            "model": MODEL_ID,
            "revision": MODEL_REV,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu",
            "smoke": smoke,
            "elapsed_sec": elapsed,
            "max_telescoping_err": round(max_tele_err, 6),
            "n_layers": model.config.num_hidden_layers,
            "layer_types": layer_types,
            "lens_provenance": bank.get("lens_provenance", {}),
            "cal_h_norms": cal_h_norms,
            "layer_mlp_scales": layer_mlp_scales,
            "slot_cal_stats": slot_cal_stats,
            "clean_generations": clean_gens,
        },
        "exp2a_write_geometry": exp2a_rows,
        "exp2a_generations": exp2a_gens,
        "exp2b_virtual_neurons": exp2b_rows,
        "exp2b_generations": exp2b_gens,
        "exp4_highway_natural": exp4_natural,
        "exp4_highway_virtual": exp4_virtual,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    print(f"Loading {MODEL_ID}@{MODEL_REV[:8]} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV, local_files_only=args.local_only)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, dtype=dtype, local_files_only=args.local_only
    ).to(args.device).eval()

    result = run_experiment(model, tok, bank, args.device, max_new_tokens=args.max_new_tokens, smoke=args.smoke)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"DONE in {result['meta']['elapsed_sec']}s | max_telescoping_err={result['meta']['max_telescoping_err']} | sha256={sha[:16]}... -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
