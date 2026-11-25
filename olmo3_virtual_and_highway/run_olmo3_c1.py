#!/usr/bin/env python3
"""Stage C1 GPU runner: Fresh-Unit Linear-Regime Attention Transport, Cross-POS vs.
Common-Mode Bleed Fork (H_A vs. H_B), and Mixed-POS Token-Direction Contraction
on OLMo 3 7B (`OLMO3-TRANSPORT-C1-DESIGN.md` Registered v5).

Exact planned forward budget (Full run, assuming 0 caps):
  - Clean baselines (8 prompts) + zero-dose shams (3L x 2 modes + L12 all)      =    15
  - Block 1 (C1.1): 3L x 24t x 8p x 2 modes (all/last) x [R_mf, dm(K=2)]        = 3,456
  - Block 2a (C1.2/3/4): 3L x 24t x 8p x [R_mf, R_ridge, logit_lens] (last)     = 1,728
  - Block 2b (rand_iso): 3L x M=32 draws x 8p (positions="last", 0.75x)         =   768
  - Secondary L12 (capped 0.30x): 1L x 24t x 8p x [R_mf all, R_mf last, R_r l]  =   576
  - Total = 6,543 short B=1 bf16 forwards (use_cache=False).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
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

PRIMARY_LAYERS = [18, 24, 27]
SECONDARY_LAYERS = [12]
ALL_LAYERS = [12, 18, 24, 27]

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_C1_CONSTANTS_SHA256 = "6220e9ec7ce8674c914657b3bc6c0d7c1d65cd342236655dce8e60a9769b9304"
EXPECTED_C1_BANK_SHA256 = "5a7a406a9072c7362353a0a9b00841e7743b82d90aade0adc96b680c8d3e621f"
EXPECTED_C1_DESIGN_SHA256 = "5304677e4bbf8e6cd2881aea7ee6b51055f3da76ac28e79ac96400d1fe2b905c"

EXPECTED_N_TARGETS = 24
EXPECTED_N_PROMPTS = 8
EXPECTED_K_DM = 2
EXPECTED_M_RAND = 32


def verify_preflight_provenance(
    bank_path: Path,
    r_lens_path: Path,
    c1_constants_path: Path,
    design_path: Path,
) -> dict[str, str]:
    """Fail-closed preflight before loading weights: verify all 4 frozen inputs match pinned SHA-256s."""
    for label, p in (
        ("--bank", bank_path),
        ("--r-lens", r_lens_path),
        ("--c1-constants", c1_constants_path),
        ("--design", design_path),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Preflight failed: missing required {label} file: {p}")

    hashes = {
        "r_lens_sha256": sha256_file(r_lens_path),
        "c1_constants_sha256": sha256_file(c1_constants_path),
        "c1_bank_sha256": sha256_file(bank_path),
        "c1_design_sha256": sha256_file(design_path),
    }
    expected = {
        "r_lens_sha256": EXPECTED_R_LENS_SHA256,
        "c1_constants_sha256": EXPECTED_C1_CONSTANTS_SHA256,
        "c1_bank_sha256": EXPECTED_C1_BANK_SHA256,
        "c1_design_sha256": EXPECTED_C1_DESIGN_SHA256,
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
    if meta.get("k_dm_draws") != str(EXPECTED_K_DM) or meta.get("m_rand_iso_draws") != str(EXPECTED_M_RAND):
        raise ValueError(f"Bank metadata k_dm/m_rand mismatch: {meta}")

    c1_const = json.loads(c1_constants_path.read_text(encoding="utf-8"))
    if c1_const.get("c1_vector_bank_sha256") != EXPECTED_C1_BANK_SHA256:
        raise ValueError("c1_prereg_constants.json c1_vector_bank_sha256 mismatch")
    if c1_const.get("r_lens_sha256") != EXPECTED_R_LENS_SHA256:
        raise ValueError("c1_prereg_constants.json r_lens_sha256 mismatch")
    return hashes


def load_c1_bank(
    bank_path: Path,
    r_lens_path: Path,
    c1_constants: dict[str, Any],
) -> dict[str, Any]:
    """Load the frozen Stage C1 safetensors bank + compute exact per-layer frozen `pred_ret` from `r_lens.pt`."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(bank_path), framework="pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            t = f.get_tensor(k).float().contiguous()
            if not bool(torch.isfinite(t).all().item()):
                raise ValueError(f"Non-finite value in bank tensor {k}")
            tensors[k] = t

    rl = torch.load(str(r_lens_path), map_location="cpu", weights_only=False)
    r_out = rl["means"]["out"]

    u_vectors = tensors["u_vectors"]  # [24, d]
    u_hat = tensors["u_hat"]          # [24, d]
    n_t = u_vectors.shape[0]
    targets = list(c1_constants["targets_24"])
    if n_t != len(targets):
        raise ValueError(f"Bank target count {n_t} != constants targets_24 count {len(targets)}")

    layers_bank: dict[int, dict[str, Any]] = {}
    for L in ALL_LAYERS:
        R_d = r_out[L].double()
        w_rmf = tensors[f"L{L}.R_mf"]
        w_rr = tensors[f"L{L}.R_ridge"]
        w_dm = tensors.get(f"L{L}.direct_matched")
        w_rand = tensors.get(f"L{L}.rand_iso")

        # Validate unit norms strictly
        for name, mat in (("R_mf", w_rmf), ("R_ridge", w_rr), ("u_hat", u_hat)):
            nrms = mat.norm(dim=-1)
            if float((nrms - 1.0).abs().max().item()) > 1e-4:
                raise ValueError(f"L{L}.{name} not unit-norm: max dev={float((nrms - 1.0).abs().max().item())}")
        if w_dm is not None:
            nrms = w_dm.norm(dim=-1)
            if float((nrms - 1.0).abs().max().item()) > 1e-4:
                raise ValueError(f"L{L}.direct_matched not unit-norm")
        if w_rand is not None:
            nrms = w_rand.norm(dim=-1)
            if float((nrms - 1.0).abs().max().item()) > 1e-4:
                raise ValueError(f"L{L}.rand_iso not unit-norm")

        # Exact float64 pred_ret = (u_t^T R_L w) / (u_t . w)
        pred_ret: dict[str, dict[str, Any]] = {}
        for ti, t in enumerate(targets):
            u_d = u_vectors[ti].double()

            def _pr(w_vec: torch.Tensor) -> float:
                wd = w_vec.double()
                denom = float(torch.dot(u_d, wd).item())
                if abs(denom) <= 1e-8:
                    return 0.0
                return float(torch.dot(u_d, R_d @ wd).item() / denom)

            dm_prs = [_pr(w_dm[ti, k]) for k in range(w_dm.shape[1])] if w_dm is not None else []
            pred_ret[t["string"]] = {
                "R_mf": _pr(w_rmf[ti]),
                "R_ridge": _pr(w_rr[ti]),
                "logit_lens": _pr(u_hat[ti]),
                "direct_matched": dm_prs,
            }

        layers_bank[L] = {
            "R_mf": w_rmf,
            "R_ridge": w_rr,
            "logit_lens": u_hat,
            "direct_matched": w_dm,
            "rand_iso": w_rand,
            "pred_ret": pred_ret,
        }

    return {
        "meta": meta,
        "u_vectors": u_vectors,
        "u_hat": u_hat,
        "targets": targets,
        "layers_bank": layers_bank,
    }


def decompose_rand_iso_c1(
    clean: dict,
    intervened: dict,
    inject_layer: int,
    v_unit: torch.Tensor,
    dose: float,
    target_ids: list[int],
    prompt_id: str,
) -> dict[str, Any]:
    """Wrap C0's `decompose_rand_iso_trace` with `all_24_dlogits` and `rms_clean`/`rms_int`."""
    rec = decompose_rand_iso_trace(clean, intervened, inject_layer, v_unit, dose, target_ids)
    all_24 = rec.pop("all_16_dlogits")
    rec["all_24_dlogits"] = all_24
    rec["rms_clean"] = round(float(clean["rms_final"]), 6)
    rec["rms_int"] = round(float(intervened["rms_final"]), 6)
    rec["prompt_id"] = prompt_id
    return rec


def _run_block1_regime_d(
    model,
    L: int,
    positions: str,
    targets: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    enc_prompts: list[torch.Tensor],
    clean_traces: list[dict],
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]],
    lb: dict[str, Any],
    u_vectors: torch.Tensor,
    cap_d: float,
    k_dm: int,
    fwd_counter: dict[str, int],
    fwd_key: str,
    cap_key: str,
) -> tuple[dict[str, Any], float]:
    """Run Block 1 Regime D (`Direct_linear = +3.0`, `R_mf` + `direct_matched` K draws) at `positions`."""
    rd: dict[str, Any] = {
        "positions": positions,
        "target_direct_linear": 3.0,
        "dose_cap": round(cap_d, 6),
        "R_mf": {},
        "direct_matched": {},
    }
    max_rmf_move = 0.0
    for ti, t in enumerate(targets):
        t_str = t["string"]
        tid = int(t["id"])
        u_t = u_vectors[ti]
        pr_map = lb["pred_ret"][t_str]

        # R_mf
        w_rmf = lb["R_mf"][ti]
        u_dot_w = float(torch.dot(u_t, w_rmf.float()).item())
        rows_rmf = []
        for pi, p in enumerate(prompts):
            r_c = float(clean_traces[pi]["rms_final"])
            dose_d = 3.0 * r_c / u_dot_w
            if dose_d > cap_d:
                fwd_counter[cap_key] += 1
                rows_rmf.append({
                    "prompt_id": p["id"],
                    "capped": True,
                    "dose_requested": round(dose_d, 6),
                    "dose_cap": round(cap_d, 6),
                })
                continue
            with post_norm_inject_hook(model, L, w_rmf, dose=dose_d, positions=positions):
                tr = capture_last_trace(model, enc_prompts[pi])
                fwd_counter[fwd_key] += 1
            rec = decompose_target_trace(
                clean_traces[pi], tr, L, u_t, tid, dose_d, pr_map["R_mf"], w_rmf,
                clean_probs[pi][0], clean_probs[pi][1],
            )
            if (
                float(rec["kl"]) == 0.0
                and abs(float(rec["exact_dlogit"])) == 0.0
                and abs(float(rec["sum_attn"])) == 0.0
                and abs(float(rec["sum_mlp"])) == 0.0
            ):
                raise RuntimeError(
                    f"DEAD ARM CHECK FAILED: Block 1 ({positions}) L{L} R_mf[{t_str!r}] prompt {p['id']}"
                )
            rec["prompt_id"] = p["id"]
            rec["capped"] = False
            dl_all = tr["logits"] - clean_traces[pi]["logits"]
            max_rmf_move = max(max_rmf_move, float(dl_all.abs().max().item()))
            rows_rmf.append(rec)
        rd["R_mf"][t_str] = rows_rmf

        # direct_matched (K orthogonal draws)
        dm_draws_rows = []
        for k in range(k_dm):
            w_dm = lb["direct_matched"][ti, k]
            u_dot_dm = float(torch.dot(u_t, w_dm.float()).item())
            rows_p = []
            for pi, p in enumerate(prompts):
                r_c = float(clean_traces[pi]["rms_final"])
                dose_d = 3.0 * r_c / u_dot_dm
                if dose_d > cap_d:
                    fwd_counter[cap_key] += 1
                    rows_p.append({
                        "prompt_id": p["id"],
                        "capped": True,
                        "dose_requested": round(dose_d, 6),
                        "dose_cap": round(cap_d, 6),
                    })
                    continue
                with post_norm_inject_hook(model, L, w_dm, dose=dose_d, positions=positions):
                    tr = capture_last_trace(model, enc_prompts[pi])
                    fwd_counter[fwd_key] += 1
                rec = decompose_target_trace(
                    clean_traces[pi], tr, L, u_t, tid, dose_d, pr_map["direct_matched"][k], w_dm,
                    clean_probs[pi][0], clean_probs[pi][1],
                )
                if (
                    float(rec["kl"]) == 0.0
                    and abs(float(rec["exact_dlogit"])) == 0.0
                    and abs(float(rec["sum_attn"])) == 0.0
                    and abs(float(rec["sum_mlp"])) == 0.0
                ):
                    raise RuntimeError(
                        f"DEAD ARM CHECK FAILED: Block 1 ({positions}) L{L} dm[{t_str!r}] k={k} prompt {p['id']}"
                    )
                rec["prompt_id"] = p["id"]
                rec["capped"] = False
                rows_p.append(rec)
            dm_draws_rows.append(rows_p)
        rd["direct_matched"][t_str] = dm_draws_rows

    return rd, max_rmf_move


def _run_regime_n_arms(
    model,
    L: int,
    positions: str,
    targets: list[dict[str, Any]],
    target_ids: list[int],
    prompts: list[dict[str, Any]],
    enc_prompts: list[torch.Tensor],
    clean_traces: list[dict],
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]],
    lb: dict[str, Any],
    u_vectors: torch.Tensor,
    u_hat: torch.Tensor,
    norm_l: float,
    dose_n: float,
    arms: tuple[str, ...],
    fwd_counter: dict[str, int],
    fwd_key: str,
    record_all_dlogits: bool = True,
) -> tuple[dict[str, Any], float]:
    """Run Regime N arms recording full 4-term decomposition + `all_24_dlogits` (+ `u_hat` J-selectivity on `logit_lens`)."""
    rn: dict[str, Any] = {
        "positions": positions,
        "dose_N": round(dose_n, 6),
        "post_mlp_norm": norm_l,
    }
    for arm in arms:
        rn[arm] = {}

    max_rmf_move = 0.0
    n_t = len(targets)
    u_hat_f = u_hat.float()

    for ti, t in enumerate(targets):
        t_str = t["string"]
        tid = int(t["id"])
        u_t = u_vectors[ti]
        pr_map = lb["pred_ret"][t_str]
        for arm in arms:
            w = lb[arm][ti]
            rows_p = []
            for pi, p in enumerate(prompts):
                with post_norm_inject_hook(model, L, w, dose=dose_n, positions=positions):
                    tr = capture_last_trace(model, enc_prompts[pi])
                    fwd_counter[fwd_key] += 1
                rec = decompose_target_trace(
                    clean_traces[pi], tr, L, u_t, tid, dose_n, pr_map[arm], w,
                    clean_probs[pi][0], clean_probs[pi][1],
                )
                rec["prompt_id"] = p["id"]
                dl_all = tr["logits"] - clean_traces[pi]["logits"]
                if arm == "R_mf":
                    max_rmf_move = max(max_rmf_move, float(dl_all.abs().max().item()))
                if record_all_dlogits:
                    all24 = [round(float(dl_all[x].item()), 6) for x in target_ids]
                    rec["all_24_dlogits"] = all24
                    if len(all24) > 1 and max(abs(v) for j, v in enumerate(all24) if j != ti) == 0.0:
                        raise RuntimeError(
                            f"DEAD ARM CHECK FAILED: Regime N ({positions}) L{L} {arm}[{t_str!r}] "
                            f"prompt {p['id']} all-zero off-target dlogits"
                        )
                if abs(float(rec["exact_dlogit"])) == 0.0 and float(rec["kl"]) == 0.0:
                    raise RuntimeError(
                        f"DEAD ARM CHECK FAILED: Regime N ({positions}) L{L} {arm}[{t_str!r}] prompt {p['id']}"
                    )
                if arm == "logit_lens":
                    # 0-forward non-gating C1.4 diagonal-vs-off-diagonal selectivity diagnostic:
                    # u_hat_j^T J u_hat_t = (u_hat_j . (h_i - h_c)) / dose_N
                    dh_f = (tr["h_final"] - clean_traces[pi]["h_final"]).float()
                    proj = (u_hat_f @ dh_f) / float(dose_n)
                    self_j = float(proj[ti].item())
                    if n_t > 1:
                        off_vals = [abs(float(proj[j].item())) for j in range(n_t) if j != ti]
                        off_abs_mean = math.fsum(off_vals) / len(off_vals)
                    else:
                        off_abs_mean = 0.0
                    rec["u_hat_self_J"] = round(self_j, 6)
                    rec["u_hat_off_abs_mean_J"] = round(off_abs_mean, 6)
                rows_p.append(rec)
            rn[arm][t_str] = rows_p
    return rn, max_rmf_move


def run_c1(
    model,
    tokenizer,
    bank: dict[str, Any],
    c1_constants: dict[str, Any],
    out_dir: Path,
    hashes: dict[str, str],
    device: str = "cuda",
    smoke: bool = False,
    primary_layers: list[int] | None = None,
    secondary_layers: list[int] | None = None,
) -> dict[str, Any]:
    """Execute full Stage C1 (or tiny-model smoke) through the real hook & decomposition path."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    p_layers = list(primary_layers) if primary_layers is not None else list(PRIMARY_LAYERS)
    s_layers = list(secondary_layers) if secondary_layers is not None else list(SECONDARY_LAYERS)
    all_layers = s_layers + p_layers

    all_targets = list(bank["targets"])
    all_prompts = list(c1_constants["prompts_8"])
    targets = all_targets[:4] if smoke else all_targets
    prompts = all_prompts[:2] if smoke else all_prompts
    k_dm = 1 if smoke else EXPECTED_K_DM
    m_rand = 2 if smoke else EXPECTED_M_RAND

    u_vectors = bank["u_vectors"].float()
    u_hat = bank["u_hat"].float()
    target_ids = [int(t["id"]) for t in targets]
    target_strings = [t["string"] for t in targets]
    prompt_ids = [p["id"] for p in prompts]

    enc_prompts = [
        tokenizer.encode(p["text"], return_tensors="pt", add_special_tokens=False).to(device)
        for p in prompts
    ]

    fwd_counts: dict[str, int] = {
        "clean": 0,
        "sham": 0,
        "block1_regime_D_all": 0,
        "block1_regime_D_all_capped_skipped": 0,
        "block1_regime_D_last": 0,
        "block1_regime_D_last_capped_skipped": 0,
        "block2a_regime_N_last": 0,
        "block2b_rand_iso_last": 0,
        "secondary_L12_030x": 0,
    }

    # 1. Clean baselines (8 forwards in full run) + clean p_clean(t | prompt) diagnostic
    clean_traces: list[dict] = []
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]] = []
    clean_target_probs: dict[str, dict[str, dict[str, float]]] = {}
    clean_rms_final: dict[str, float] = {}
    for pi, p in enumerate(prompts):
        tr = capture_last_trace(model, enc_prompts[pi])
        fwd_counts["clean"] += 1
        clean_traces.append(tr)
        lc = tr["logits"].float()
        pc = F.softmax(lc, dim=-1)
        lpc = F.log_softmax(lc, dim=-1)
        clean_probs.append((pc, lpc))
        clean_rms_final[p["id"]] = round(float(tr["rms_final"]), 6)
        p_entry: dict[str, dict[str, float]] = {}
        for t in targets:
            tid = int(t["id"])
            p_entry[t["string"]] = {
                "p_clean": float(pc[tid].item()),
                "log_p_clean": round(float(lpc[tid].item()), 6),
            }
        clean_target_probs[p["id"]] = p_entry

    post_mlp_norms: dict[str, float] = {}
    for L in all_layers:
        norms = [float(tr["mlp"][L].norm().item()) for tr in clean_traces]
        post_mlp_norms[str(L)] = round(math.fsum(norms) / len(norms), 6)

    # 2. Zero-dose shams (3L x 2 modes + L12 all = 7 forwards in full run)
    shams: dict[str, dict[str, float]] = {"last": {}, "all": {}}
    for L in p_layers:
        w0 = bank["layers_bank"][L]["R_mf"][0]
        for mode in ("last", "all"):
            with post_norm_inject_hook(model, L, w0, dose=0.0, positions=mode):
                tr_sham = capture_last_trace(model, enc_prompts[0])
                fwd_counts["sham"] += 1
            max_diff = float((tr_sham["logits"] - clean_traces[0]["logits"]).abs().max().item())
            shams[mode][str(L)] = max_diff
            if not math.isfinite(max_diff) or max_diff != 0.0:
                raise RuntimeError(f"C1.0.1 SHAM FAILED at L{L} ({mode}): max |dlogit| = {max_diff}")
    for L in s_layers:
        w0 = bank["layers_bank"][L]["R_mf"][0]
        with post_norm_inject_hook(model, L, w0, dose=0.0, positions="all"):
            tr_sham = capture_last_trace(model, enc_prompts[0])
            fwd_counts["sham"] += 1
        max_diff = float((tr_sham["logits"] - clean_traces[0]["logits"]).abs().max().item())
        shams["all"][str(L)] = max_diff
        if not math.isfinite(max_diff) or max_diff != 0.0:
            raise RuntimeError(f"C1.0.1 SHAM FAILED at L{L} (all): max |dlogit| = {max_diff}")

    # 3. Primary layers (Block 1 + Block 2a + Block 2b) with strict provenance-checked checkpoints
    block1_d_all: dict[str, Any] = {}
    block1_d_last: dict[str, Any] = {}
    block2a_n_last: dict[str, Any] = {}
    block2b_rand_iso: dict[str, Any] = {}
    sham_movement_check: dict[str, dict[str, float]] = {"last": {}, "all": {}}

    for L in p_layers:
        norm_l = post_mlp_norms[str(L)]
        dose_n = 0.75 * norm_l
        cap_d = 3.0 * norm_l
        expected_prov = {
            "study": "stage_c1_olmo3_7b_fresh_mixed_pos_transport_and_bleed_fork",
            "layer": L,
            "role": "primary",
            "model_id": MODEL_ID,
            "model_rev": MODEL_REV,
            "device": str(device),
            "torch_version": torch.__version__,
            **hashes,
            "prompt_ids": prompt_ids,
            "target_strings": target_strings,
            "target_ids": target_ids,
            "post_mlp_norm": norm_l,
            "dose_N": round(dose_n, 6),
            "k_dm": k_dm,
            "m_rand": m_rand,
            "smoke": smoke,
        }
        ckpt_path = out_dir / f"ckpt_c1_L{L}.json"
        if ckpt_path.exists():
            ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if ck.get("provenance") != expected_prov:
                raise RuntimeError(f"Checkpoint provenance mismatch for L{L} at {ckpt_path}; delete stale checkpoint.")
            block1_d_all[str(L)] = ck["block1_d_all"]
            block1_d_last[str(L)] = ck["block1_d_last"]
            block2a_n_last[str(L)] = ck["block2a_n_last"]
            block2b_rand_iso[str(L)] = ck["block2b_rand_iso"]
            sham_movement_check["all"][str(L)] = ck["max_rmf_move_all"]
            sham_movement_check["last"][str(L)] = ck["max_rmf_move_last"]
            for k_f, v_f in ck["fwd_delta"].items():
                fwd_counts[k_f] += int(v_f)
            print(f"[stage-c1] Loaded verified checkpoint for L{L}", flush=True)
            continue

        lb = bank["layers_bank"][L]
        t_l0 = time.time()
        before = dict(fwd_counts)

        rd_all, mv_all = _run_block1_regime_d(
            model, L, "all", targets, prompts, enc_prompts, clean_traces, clean_probs,
            lb, u_vectors, cap_d, k_dm, fwd_counts, "block1_regime_D_all", "block1_regime_D_all_capped_skipped",
        )
        rd_last, mv_last = _run_block1_regime_d(
            model, L, "last", targets, prompts, enc_prompts, clean_traces, clean_probs,
            lb, u_vectors, cap_d, k_dm, fwd_counts, "block1_regime_D_last", "block1_regime_D_last_capped_skipped",
        )
        rn_last, mv_n = _run_regime_n_arms(
            model, L, "last", targets, target_ids, prompts, enc_prompts, clean_traces, clean_probs,
            lb, u_vectors, u_hat, norm_l, dose_n, ("R_mf", "R_ridge", "logit_lens"),
            fwd_counts, "block2a_regime_N_last", record_all_dlogits=True,
        )

        # Block 2b: M=32 isotropic random draws at positions="last" (0.75x)
        rand_rows: list[list[dict[str, Any]]] = []
        for m in range(m_rand):
            v_m = lb["rand_iso"][m]
            rows_p = []
            for pi, p in enumerate(prompts):
                with post_norm_inject_hook(model, L, v_m, dose=dose_n, positions="last"):
                    tr = capture_last_trace(model, enc_prompts[pi])
                    fwd_counts["block2b_rand_iso_last"] += 1
                rows_p.append(decompose_rand_iso_c1(clean_traces[pi], tr, L, v_m, dose_n, target_ids, p["id"]))
            rand_rows.append(rows_p)
        b2b_entry = {
            "positions": "last",
            "dose_N": round(dose_n, 6),
            "post_mlp_norm": norm_l,
            "m_rand": m_rand,
            "rand_iso": rand_rows,
        }

        mv_last_tot = max(mv_last, mv_n)
        block1_d_all[str(L)] = rd_all
        block1_d_last[str(L)] = rd_last
        block2a_n_last[str(L)] = rn_last
        block2b_rand_iso[str(L)] = b2b_entry
        sham_movement_check["all"][str(L)] = round(mv_all, 6)
        sham_movement_check["last"][str(L)] = round(mv_last_tot, 6)

        fwd_delta = {k_f: fwd_counts[k_f] - before[k_f] for k_f in fwd_counts}
        ck_tmp = ckpt_path.with_suffix(".json.tmp")
        ck_tmp.write_text(json.dumps({
            "provenance": expected_prov,
            "block1_d_all": rd_all,
            "block1_d_last": rd_last,
            "block2a_n_last": rn_last,
            "block2b_rand_iso": b2b_entry,
            "max_rmf_move_all": round(mv_all, 6),
            "max_rmf_move_last": round(mv_last_tot, 6),
            "fwd_delta": fwd_delta,
        }), encoding="utf-8")
        os.replace(ck_tmp, ckpt_path)
        print(
            f"[stage-c1] Primary L{L} done in {time.time() - t_l0:.1f}s "
            f"(fwds={sum(fwd_delta.values())}, mv_all={mv_all:.3f}, mv_last={mv_last_tot:.3f})",
            flush=True,
        )

    # 4. Secondary L12 capped 0.30x post_mlp_norm[12] (strictly outside Gate C1.0)
    secondary_l12: dict[str, Any] = {}
    for L in s_layers:
        norm_l = post_mlp_norms[str(L)]
        dose_sec = 0.30 * norm_l
        lb = bank["layers_bank"][L]
        t_l0 = time.time()
        l12_all, mv_l12_all = _run_regime_n_arms(
            model, L, "all", targets, target_ids, prompts, enc_prompts, clean_traces, clean_probs,
            lb, u_vectors, u_hat, norm_l, dose_sec, ("R_mf",),
            fwd_counts, "secondary_L12_030x", record_all_dlogits=False,
        )
        l12_last, mv_l12_last = _run_regime_n_arms(
            model, L, "last", targets, target_ids, prompts, enc_prompts, clean_traces, clean_probs,
            lb, u_vectors, u_hat, norm_l, dose_sec, ("R_mf", "R_ridge"),
            fwd_counts, "secondary_L12_030x", record_all_dlogits=True,
        )
        sham_movement_check["all"][str(L)] = round(mv_l12_all, 6)
        sham_movement_check["last"][str(L)] = round(mv_l12_last, 6)
        secondary_l12[str(L)] = {
            "dose_multiplier": 0.30,
            "dose_N": round(dose_sec, 6),
            "post_mlp_norm": norm_l,
            "R_mf_all": l12_all["R_mf"],
            "R_mf_last": l12_last["R_mf"],
            "R_ridge_last": l12_last["R_ridge"],
        }
        print(
            f"[stage-c1] Secondary L{L} (0.30x) done in {time.time() - t_l0:.1f}s "
            f"(mv_all={mv_l12_all:.3f}, mv_last={mv_l12_last:.3f})",
            flush=True,
        )

    total_intervened = (
        fwd_counts["block1_regime_D_all"]
        + fwd_counts["block1_regime_D_last"]
        + fwd_counts["block2a_regime_N_last"]
        + fwd_counts["block2b_rand_iso_last"]
        + fwd_counts["secondary_L12_030x"]
    )
    total_fwds = fwd_counts["clean"] + fwd_counts["sham"] + total_intervened

    receipt = {
        "schema_version": "c1-receipt-v1",
        "meta": {
            "study": "OLMo-3-7B Stage C1 (Fresh Mixed-POS Transport, H_A/H_B Bleed Fork & Token-Direction Contraction)",
            "model_id": MODEL_ID,
            "model_rev": MODEL_REV,
            "smoke": smoke,
            "device": str(device),
            "torch_version": torch.__version__,
            "primary_layers": p_layers,
            "secondary_layers": s_layers,
            "k_dm": k_dm,
            "m_rand": m_rand,
            "n_targets": len(targets),
            "n_prompts": len(prompts),
            "elapsed_sec": round(time.time() - t0, 2),
            "forward_counts": fwd_counts,
            "total_intervened_forwards": total_intervened,
            "total_forwards": total_fwds,
            **hashes,
        },
        "targets_24": targets,
        "prompts_8": prompts,
        "post_mlp_norms": post_mlp_norms,
        "clean_rms_final": clean_rms_final,
        "clean_target_probs": clean_target_probs,
        "shams": shams,
        "sham_movement_check": sham_movement_check,
        "block1_regime_D_all": block1_d_all,
        "block1_regime_D_last": block1_d_last,
        "block2a_regime_N_last": block2a_n_last,
        "block2b_rand_iso": block2b_rand_iso,
        "secondary_L12": secondary_l12,
    }

    out_gz = out_dir / "c1_receipt.json.gz"
    tmp_gz = out_dir / "c1_receipt.json.gz.tmp"
    with gzip.open(tmp_gz, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(receipt, f)
    os.replace(tmp_gz, out_gz)
    print(f"[stage-c1] Wrote {out_gz} (total_fwds={total_fwds}, elapsed={receipt['meta']['elapsed_sec']}s)", flush=True)
    return receipt


def main() -> None:
    root = Path(__file__).resolve().parent
    repo_root = root.parent
    ap = argparse.ArgumentParser(description="Run Stage C1 on OLMo 3 7B (pre-registered v5).")
    ap.add_argument("--bank", type=Path, default=repo_root / "work" / "olmo3_c1" / "c1_vector_bank.safetensors")
    ap.add_argument("--r-lens", type=Path, default=repo_root / "work" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt")
    ap.add_argument("--c1-constants", type=Path, default=root / "c1_prereg_constants.json")
    ap.add_argument("--design", type=Path, default=root / "OLMO3-TRANSPORT-C1-DESIGN.md")
    ap.add_argument("--out-dir", type=Path, default=repo_root / "work" / "olmo3_c1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true", help="Reduced smoke run (always refused by scorer)")
    args = ap.parse_args()

    hashes = verify_preflight_provenance(args.bank, args.r_lens, args.c1_constants, args.design)
    c1_const = json.loads(args.c1_constants.read_text(encoding="utf-8"))
    bank = load_c1_bank(args.bank, args.r_lens, c1_const)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    run_c1(model, tok, bank, c1_const, args.out_dir, hashes=hashes, device=args.device, smoke=args.smoke)


if __name__ == "__main__":
    main()
