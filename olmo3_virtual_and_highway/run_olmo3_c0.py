#!/usr/bin/env python3
"""Stage C0 GPU runner: Pre-Registered Out-of-Sample Predictive Transport & Active-Erasure
Benchmark on OLMo 3 7B (`OLMO3-TRANSPORT-C0-DESIGN.md` Registered v1).

Exact forward budget (Full run):
  - 8 clean baselines + 7 zero-dose shams = 15 forwards (shared across Regimes N and D)
  - Regime N (0.75 * post_mlp_norm[L]):
      7 x 16 x 8 x 3 (R_mf, R_ridge, logit_lens; R_mf_cross read free from R_mf [V] logits) = 2,688
      7 x 16 x 8 x 4 (direct_matched, K=4)                                                   = 3,584
      7 x 32 x 8     (rand_iso, M=32 pooled draws/layer)                                     = 1,792
      Subtotal = 8,064 forwards
  - Regime D (Direct_linear = +3.0, capped at 3.0 * post_mlp_norm[L]):
      7 x 16 x 8 x 3 (R_mf, R_ridge, logit_lens) = 2,688
      7 x 16 x 8 x 4 (direct_matched, K=4)       = 3,584
      Subtotal = 6,272 forwards
  - Total = 14,351 short B=1 bf16 forwards (use_cache=False).
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

# Exact 8 held-out prompts (paired 2 x 2 format x style battery) from Registered v1 §2.
PROMPTS_8: list[dict[str, str]] = [
    {
        "id": "bare_narr_0",
        "format": "bare",
        "style": "narrative",
        "text": "At the edge of the quiet village stood a",
    },
    {
        "id": "bare_narr_1",
        "format": "bare",
        "style": "narrative",
        "text": "Inside the locked wooden box lay a",
    },
    {
        "id": "bare_anal_0",
        "format": "bare",
        "style": "analytical",
        "text": "Unlike the other items on the list, the final one was a",
    },
    {
        "id": "bare_anal_1",
        "format": "bare",
        "style": "analytical",
        "text": "After searching for hours, the team finally found a",
    },
    {
        "id": "chat_narr_0",
        "format": "chatml",
        "style": "narrative",
        "text": (
            "<|im_start|>user\nComplete this story fragment with one noun: "
            "At the edge of the quiet village stood a...<|im_end|>\n"
            "<|im_start|>assistant\nAt the edge of the quiet village stood a"
        ),
    },
    {
        "id": "chat_narr_1",
        "format": "chatml",
        "style": "narrative",
        "text": (
            "<|im_start|>user\nComplete this story fragment with one noun: "
            "Inside the locked wooden box lay a...<|im_end|>\n"
            "<|im_start|>assistant\nInside the locked wooden box lay a"
        ),
    },
    {
        "id": "chat_anal_0",
        "format": "chatml",
        "style": "analytical",
        "text": (
            "<|im_start|>user\nComplete this analytical sentence with one noun: "
            "Unlike the other items on the list, the final one was a...<|im_end|>\n"
            "<|im_start|>assistant\nUnlike the other items on the list, the final one was a"
        ),
    },
    {
        "id": "chat_anal_1",
        "format": "chatml",
        "style": "analytical",
        "text": (
            "<|im_start|>user\nComplete this analytical sentence with one noun: "
            "After searching for hours, the team finally found a...<|im_end|>\n"
            "<|im_start|>assistant\nAfter searching for hours, the team finally found a"
        ),
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def post_norm_last_inject_hook(model, layer: int, w_unit: torch.Tensor, dose: float):
    """Inject `dose * w_unit` at `post_feedforward_layernorm` output, last position only."""
    lay = model.model.layers[layer]

    def _hook(_mod, _args, output):
        if dose == 0.0:
            return output
        out = output.clone()
        delta = (dose * w_unit.to(device=out.device, dtype=torch.float32)).to(out.dtype)
        out[:, -1, :] = out[:, -1, :] + delta
        return out

    h = lay.post_feedforward_layernorm.register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


def capture_last_trace(model, input_ids: torch.Tensor) -> dict:
    """Run one B=1 forward capturing post-norm Attn_l, MLP_l increments, h_final, rms_final, logits."""
    n_layers = int(model.config.num_hidden_layers)
    attn_inc: list[torch.Tensor | None] = [None] * n_layers
    mlp_inc: list[torch.Tensor | None] = [None] * n_layers
    holder: dict[str, torch.Tensor] = {}
    handles = []

    for L in range(n_layers):
        lay = model.model.layers[L]

        def _mk_a(idx: int):
            def _fn(_m, _a, out):
                attn_inc[idx] = out[0, -1, :].detach().float().cpu()
            return _fn

        def _mk_m(idx: int):
            def _fn(_m, _a, out):
                mlp_inc[idx] = out[0, -1, :].detach().float().cpu()
            return _fn

        handles.append(lay.post_attention_layernorm.register_forward_hook(_mk_a(L)))
        handles.append(lay.post_feedforward_layernorm.register_forward_hook(_mk_m(L)))

    def _cap_final(_m, args):
        holder["h_final"] = args[0][0, -1, :].detach().float().cpu()

    handles.append(model.model.norm.register_forward_pre_hook(_cap_final))
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1, :].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()

    eps = float(getattr(model.model.norm, "variance_epsilon", 1e-6))
    h_f = holder["h_final"]
    rms_f = math.sqrt(float(h_f.pow(2).mean().item()) + eps)
    return {
        "attn": attn_inc,
        "mlp": mlp_inc,
        "h_final": h_f,
        "rms_final": rms_f,
        "logits": logits,
    }


def compute_vocab_dlogit_z_kl(
    logits_clean: torch.Tensor,
    logits_int: torch.Tensor,
    target_id: int,
    p_clean: torch.Tensor,
    log_p_clean: torch.Tensor,
) -> tuple[float, float]:
    """Exact Registered v1 §2 per-trace full-vocab `dlogit_z` (ddof=1) and `KL` in fp32."""
    dl = logits_int.float() - logits_clean.float()
    d_tgt = float(dl[target_id].item())
    dl_mean = float(dl.mean().item())
    dl_std = float(dl.std(unbiased=True).clamp_min(1e-8).item())
    dlogit_z = (d_tgt - dl_mean) / dl_std
    log_p_int = F.log_softmax(logits_int.float(), dim=-1)
    kl = float((p_clean * (log_p_clean - log_p_int)).sum().item())
    return dlogit_z, kl


def decompose_target_trace(
    clean: dict,
    intervened: dict,
    inject_layer: int,
    u_t: torch.Tensor,
    target_id: int,
    dose: float,
    pred_ret: float,
    w_unit: torch.Tensor,
    p_clean: torch.Tensor,
    log_p_clean: torch.Tensor,
) -> dict:
    """Exact Registered v1 §2 4-term telescoping decomposition + Retention + dlogit_z + KL."""
    u = u_t.float()
    h_c = clean["h_final"]
    h_i = intervened["h_final"]
    r_c = float(clean["rms_final"])
    r_i = float(intervened["rms_final"])

    exact_dlogit = float(intervened["logits"][target_id].item() - clean["logits"][target_id].item())
    dm_L = intervened["mlp"][inject_layer] - clean["mlp"][inject_layer]

    direct_linear = float(torch.dot(u, dm_L).item() / r_c)
    direct_linear_analytic = float(dose * torch.dot(u, w_unit.float()).item() / r_c)
    frozen_direct_effect = float(dose * torch.dot(u, w_unit.float()).item() / r_i)
    pred_dlogit = float(pred_ret * direct_linear_analytic)

    n_layers = len(clean["attn"])
    sum_attn = 0.0
    sum_mlp = 0.0
    abs_sum = abs(direct_linear)
    for l in range(inject_layer + 1, n_layers):
        da = intervened["attn"][l] - clean["attn"][l]
        dm = intervened["mlp"][l] - clean["mlp"][l]
        a_val = float(torch.dot(u, da).item() / r_c)
        m_val = float(torch.dot(u, dm).item() / r_c)
        sum_attn += a_val
        sum_mlp += m_val
        abs_sum += abs(a_val) + abs(m_val)

    rmsnorm_shift = float(torch.dot(u, h_i).item() * (1.0 / r_i - 1.0 / r_c))
    abs_sum += abs(rmsnorm_shift)
    recon_dlogit = direct_linear + sum_attn + sum_mlp + rmsnorm_shift
    telescoping_err = abs(exact_dlogit - recon_dlogit)
    canc_r = abs_sum / max(abs(exact_dlogit), 1e-6)
    retention = (exact_dlogit - rmsnorm_shift) / direct_linear

    dlogit_z, kl = compute_vocab_dlogit_z_kl(
        clean["logits"], intervened["logits"], target_id, p_clean, log_p_clean
    )
    return {
        "dose": round(float(dose), 6),
        "exact_dlogit": round(exact_dlogit, 6),
        "pred_dlogit": round(pred_dlogit, 6),
        "direct_linear": round(direct_linear, 6),
        "direct_linear_analytic": round(direct_linear_analytic, 6),
        "frozen_direct_effect": round(frozen_direct_effect, 6),
        "sum_attn": round(sum_attn, 6),
        "sum_mlp": round(sum_mlp, 6),
        "rmsnorm_shift": round(rmsnorm_shift, 6),
        "telescoping_err": round(telescoping_err, 6),
        "retention": round(retention, 6),
        "canc_r": round(canc_r, 4),
        "dlogit_z": round(dlogit_z, 5),
        "kl": round(kl, 6),
        "rms_clean": round(r_c, 6),
        "rms_int": round(r_i, 6),
    }


def decompose_rand_iso_trace(
    clean: dict,
    intervened: dict,
    inject_layer: int,
    v_unit: torch.Tensor,
    dose: float,
    target_ids: list[int],
) -> dict:
    """Score one pooled `rand_iso` draw on all 16 target logits + self Jacobian diagonal `v^T J v`."""
    v = v_unit.float()
    dh_final = intervened["h_final"] - clean["h_final"]
    v_j_v = float(torch.dot(v, dh_final).item() / dose)

    dm_L = intervened["mlp"][inject_layer] - clean["mlp"][inject_layer]
    d_self = float(torch.dot(v, dm_L).item())
    abs_sum = abs(d_self)
    n_layers = len(clean["attn"])
    for l in range(inject_layer + 1, n_layers):
        da = intervened["attn"][l] - clean["attn"][l]
        dm = intervened["mlp"][l] - clean["mlp"][l]
        abs_sum += abs(float(torch.dot(v, da).item())) + abs(float(torch.dot(v, dm).item()))
    net_self = float(torch.dot(v, dh_final).item())
    canc_r_self = abs_sum / max(abs(net_self), 1e-6)

    dl = intervened["logits"] - clean["logits"]
    all_16_dlogits = [round(float(dl[tid].item()), 6) for tid in target_ids]
    return {
        "dose": round(float(dose), 6),
        "vJv": round(v_j_v, 6),
        "canc_r_self": round(canc_r_self, 4),
        "all_16_dlogits": all_16_dlogits,
    }


def run_c0(
    model,
    tokenizer,
    bank: dict,
    out_dir: Path,
    device: str = "cuda",
    smoke: bool = False,
) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = list(bank["layers"])[:2] if smoke else list(bank["layers"])
    targets = list(bank["targets_16"])[:2] if smoke else list(bank["targets_16"])
    prompts = PROMPTS_8[:2] if smoke else PROMPTS_8
    k_dm = 1 if smoke else int(bank["meta"]["k_dm"])
    m_rand = 2 if smoke else int(bank["meta"]["m_rand"])
    u_vectors = bank["u_vectors"].float()  # [16, d]
    target_ids = [int(t["id"]) for t in targets]

    # Encode all prompts once
    enc_prompts = [
        tokenizer.encode(p["text"], return_tensors="pt", add_special_tokens=False).to(device)
        for p in prompts
    ]

    # 1. Shared clean baselines (8 forwards in full run)
    fwd_counts = {"clean": 0, "sham": 0, "regime_N": 0, "regime_D": 0, "regime_D_capped_skipped": 0}
    clean_traces: list[dict] = []
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for p_ids in enc_prompts:
        tr = capture_last_trace(model, p_ids)
        fwd_counts["clean"] += 1
        clean_traces.append(tr)
        lc = tr["logits"].float()
        clean_probs.append((F.softmax(lc, dim=-1), F.log_softmax(lc, dim=-1)))

    # Measure prompt-mean post_mlp_norm[L] from the clean baselines
    post_mlp_norms: dict[str, float] = {}
    for L in layers:
        norms = [float(tr["mlp"][L].norm().item()) for tr in clean_traces]
        post_mlp_norms[str(L)] = round(sum(norms) / len(norms), 6)

    # 2. Shared zero-dose shams (7 forwards in full run — one per layer on prompt 0)
    shams: dict[str, float] = {}
    for L in layers:
        w0 = bank["layers_bank"][L]["R_mf"][0]
        with post_norm_last_inject_hook(model, L, w0, dose=0.0):
            tr_sham = capture_last_trace(model, enc_prompts[0])
            fwd_counts["sham"] += 1
        max_diff = float((tr_sham["logits"] - clean_traces[0]["logits"]).abs().max().item())
        shams[str(L)] = max_diff
        if max_diff != 0.0:
            raise RuntimeError(f"C0.0.1 SHAM FAILED at L{L}: max |dlogit| = {max_diff}")

    # 3. Layer-by-layer execution with checkpointing
    regime_n_layers: dict[str, dict] = {}
    regime_d_layers: dict[str, dict] = {}
    for L in layers:
        ckpt_path = out_dir / f"ckpt_L{L}.json"
        if ckpt_path.exists():
            print(f"[L{L}] Resuming from checkpoint {ckpt_path}")
            ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
            regime_n_layers[str(L)] = ck["regime_N"]
            regime_d_layers[str(L)] = ck["regime_D"]
            for k in ("regime_N", "regime_D", "regime_D_capped_skipped"):
                fwd_counts[k] += int(ck["fwd_counts"][k])
            continue

        tl0 = time.time()
        lb = bank["layers_bank"][L]
        norm_l = post_mlp_norms[str(L)]
        dose_n = 0.75 * norm_l
        cap_d = 3.0 * norm_l
        l_fwds = {"regime_N": 0, "regime_D": 0, "regime_D_capped_skipped": 0}

        # --- REGIME N ---
        rn = {
            "dose_N": round(dose_n, 6),
            "post_mlp_norm": norm_l,
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
            "direct_matched": {},
            "rand_iso": [],
        }
        for ti, t in enumerate(targets):
            t_str = t["string"]
            tid = int(t["id"])
            u_t = u_vectors[ti]
            geom = lb["per_target_geom"][t_str]

            for arm, pr_key in [("R_mf", "pred_ret_Rmf"), ("R_ridge", "pred_ret_Rridge"), ("logit_lens", "pred_ret_ll")]:
                w = lb[arm][ti]
                rows_p = []
                for pi, p in enumerate(prompts):
                    with post_norm_last_inject_hook(model, L, w, dose=dose_n):
                        tr = capture_last_trace(model, enc_prompts[pi])
                        l_fwds["regime_N"] += 1
                    rec = decompose_target_trace(
                        clean_traces[pi], tr, L, u_t, tid, dose_n, geom[pr_key], w,
                        clean_probs[pi][0], clean_probs[pi][1],
                    )
                    rec["prompt_id"] = p["id"]
                    if arm == "R_mf":
                        dl_all = tr["logits"] - clean_traces[pi]["logits"]
                        rec["all_16_dlogits"] = [round(float(dl_all[x].item()), 6) for x in target_ids]
                    rows_p.append(rec)
                rn[arm][t_str] = rows_p

            # direct_matched (K=4 draws)
            dm_draws_rows = []
            for k in range(k_dm):
                w = lb["direct_matched"][ti, k]
                rows_p = []
                for pi, p in enumerate(prompts):
                    with post_norm_last_inject_hook(model, L, w, dose=dose_n):
                        tr = capture_last_trace(model, enc_prompts[pi])
                        l_fwds["regime_N"] += 1
                    rec = decompose_target_trace(
                        clean_traces[pi], tr, L, u_t, tid, dose_n, geom["pred_ret_dm"], w,
                        clean_probs[pi][0], clean_probs[pi][1],
                    )
                    rec["prompt_id"] = p["id"]
                    rows_p.append(rec)
                dm_draws_rows.append(rows_p)
            rn["direct_matched"][t_str] = dm_draws_rows

        # rand_iso (M=32 pooled draws per layer)
        for m in range(m_rand):
            v = lb["rand_iso"][m]
            rows_p = []
            for pi, p in enumerate(prompts):
                with post_norm_last_inject_hook(model, L, v, dose=dose_n):
                    tr = capture_last_trace(model, enc_prompts[pi])
                    l_fwds["regime_N"] += 1
                rec = decompose_rand_iso_trace(clean_traces[pi], tr, L, v, dose_n, target_ids)
                rec["prompt_id"] = p["id"]
                rows_p.append(rec)
            rn["rand_iso"].append(rows_p)

        # --- REGIME D (Direct_linear = +3.0, cap = 3.0 * post_mlp_norm[L]) ---
        rd = {
            "target_direct_linear": 3.0,
            "dose_cap": round(cap_d, 6),
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
            "direct_matched": {},
        }
        for ti, t in enumerate(targets):
            t_str = t["string"]
            tid = int(t["id"])
            u_t = u_vectors[ti]
            geom = lb["per_target_geom"][t_str]

            for arm, pr_key in [("R_mf", "pred_ret_Rmf"), ("R_ridge", "pred_ret_Rridge"), ("logit_lens", "pred_ret_ll")]:
                w = lb[arm][ti]
                u_dot_w = float(torch.dot(u_t, w.float()).item())
                rows_p = []
                for pi, p in enumerate(prompts):
                    r_c = float(clean_traces[pi]["rms_final"])
                    dose_d = 3.0 * r_c / u_dot_w
                    if dose_d > cap_d:
                        l_fwds["regime_D_capped_skipped"] += 1
                        rows_p.append({
                            "prompt_id": p["id"],
                            "capped": True,
                            "dose_requested": round(dose_d, 6),
                            "dose_cap": round(cap_d, 6),
                        })
                        continue
                    with post_norm_last_inject_hook(model, L, w, dose=dose_d):
                        tr = capture_last_trace(model, enc_prompts[pi])
                        l_fwds["regime_D"] += 1
                    rec = decompose_target_trace(
                        clean_traces[pi], tr, L, u_t, tid, dose_d, geom[pr_key], w,
                        clean_probs[pi][0], clean_probs[pi][1],
                    )
                    rec["prompt_id"] = p["id"]
                    rec["capped"] = False
                    rows_p.append(rec)
                rd[arm][t_str] = rows_p

            # direct_matched (K=4 draws)
            dm_draws_rows = []
            for k in range(k_dm):
                w = lb["direct_matched"][ti, k]
                u_dot_w = float(torch.dot(u_t, w.float()).item())
                rows_p = []
                for pi, p in enumerate(prompts):
                    r_c = float(clean_traces[pi]["rms_final"])
                    dose_d = 3.0 * r_c / u_dot_w
                    if dose_d > cap_d:
                        l_fwds["regime_D_capped_skipped"] += 1
                        rows_p.append({
                            "prompt_id": p["id"],
                            "capped": True,
                            "dose_requested": round(dose_d, 6),
                            "dose_cap": round(cap_d, 6),
                        })
                        continue
                    with post_norm_last_inject_hook(model, L, w, dose=dose_d):
                        tr = capture_last_trace(model, enc_prompts[pi])
                        l_fwds["regime_D"] += 1
                    rec = decompose_target_trace(
                        clean_traces[pi], tr, L, u_t, tid, dose_d, geom["pred_ret_dm"], w,
                        clean_probs[pi][0], clean_probs[pi][1],
                    )
                    rec["prompt_id"] = p["id"]
                    rec["capped"] = False
                    rows_p.append(rec)
                dm_draws_rows.append(rows_p)
            rd["direct_matched"][t_str] = dm_draws_rows

        # Atomic checkpoint save
        tmp_ckpt = out_dir / f".ckpt_L{L}.json.tmp"
        tmp_ckpt.write_text(
            json.dumps({"layer": L, "fwd_counts": l_fwds, "regime_N": rn, "regime_D": rd}),
            encoding="utf-8",
        )
        tmp_ckpt.replace(ckpt_path)
        regime_n_layers[str(L)] = rn
        regime_d_layers[str(L)] = rd
        for k in ("regime_N", "regime_D", "regime_D_capped_skipped"):
            fwd_counts[k] += l_fwds[k]
        print(
            f"[L{L}] done in {time.time() - tl0:.1f}s "
            f"(N={l_fwds['regime_N']}, D={l_fwds['regime_D']}, capped={l_fwds['regime_D_capped_skipped']})",
            flush=True,
        )

    receipt = {
        "meta": {
            "study": "stage_c0_olmo3_7b_transport",
            "model": bank["meta"]["model"],
            "revision": bank["meta"]["revision"],
            "smoke": smoke,
            "elapsed_sec": round(time.time() - t0, 2),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "r_lens_sha256": bank["meta"]["r_lens_sha256"],
            "constants_sha256": bank["meta"]["constants_sha256"],
            "design_sha256": bank["meta"]["design_sha256"],
            "k_dm": k_dm,
            "m_rand": m_rand,
            "forward_counts": fwd_counts,
            "total_forwards_executed": (
                fwd_counts["clean"] + fwd_counts["sham"] + fwd_counts["regime_N"] + fwd_counts["regime_D"]
            ),
        },
        "layers": layers,
        "targets_16": targets,
        "prompts_8": prompts,
        "post_mlp_norms": post_mlp_norms,
        "shams": shams,
        "regime_N": regime_n_layers,
        "regime_D": regime_d_layers,
    }
    gz_path = out_dir / "c0_receipt.json.gz"
    raw_bytes = json.dumps(receipt, sort_keys=False).encode("utf-8")
    with gzip.open(gz_path, "wb", compresslevel=6) as f:
        f.write(raw_bytes)
    print(
        f"Saved {gz_path} ({gz_path.stat().st_size / 1e6:.2f} MB), "
        f"uncompressed_sha256={hashlib.sha256(raw_bytes).hexdigest()}, "
        f"total_forwards={receipt['meta']['total_forwards_executed']}, elapsed={receipt['meta']['elapsed_sec']}s"
    )
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True, help="Path to c0_vector_bank.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=False)
    print(f"Loading {MODEL_ID}@{MODEL_REV} in bf16 on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    run_c0(model, tokenizer, bank, args.out_dir, device=args.device, smoke=args.smoke)


if __name__ == "__main__":
    main()
