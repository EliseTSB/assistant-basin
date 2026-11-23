#!/usr/bin/env python3
"""Deterministic generator for olmo3_virtual_and_highway/c0b_prereg_constants.json.

Reproduces every pre-data constant, measured Stage C0 anchor, Exp 4B per-frame
analogue, frozen R-lens cross-domain prediction, k_ridge sensitivity table, and
exact 2^16 power/equivalence simulation for Stage C0b (v2).

Usage:
    /home/elisepc/neuron-probe/.venv/bin/python3 \
        olmo3_virtual_and_highway/build_olmo3_c0b_constants.py \
        --out olmo3_virtual_and_highway/c0b_prereg_constants.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_C0_CONSTANTS = "olmo3_virtual_and_highway/c0_prereg_constants.json"
DEFAULT_C0_RECEIPT = "results/vesper-olmo3-7b-stage-c0/c0_receipt.json.gz"
DEFAULT_EXP4B = "results/vesper-olmo3-7b-exploratory/olmo3_idea2_and_idea4_results.json.gz"
DEFAULT_BANK = "work/olmo3_c0/c0_vector_bank.pt"
DEFAULT_R_LENS = "/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens/r_lens.pt"
DEFAULT_OUT = "olmo3_virtual_and_highway/c0b_prereg_constants.json"

EXP4B_FRAMES = [
    ("WHO_CHAT_TF", "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\nI am a"),
    ("SELF", "I am a"),
    ("NAME_BARE", "Alex is a"),
    ("ROLE_BARE", "The character is a"),
]

# Exact OLMo-3-7B-Instruct (6e5971d9...) token lengths verified via AutoTokenizer(..., add_special_tokens=False)
VERIFIED_OLMO3_TOKEN_COUNTS = {
    "bare_narr_0": 9,
    "bare_narr_1": 7,
    "bare_anal_0": 13,
    "bare_anal_1": 10,
    "chat_narr_0": 35,
    "chat_narr_1": 31,
    "chat_anal_0": 43,
    "chat_anal_1": 37,
    "WHO_CHAT_TF": 15,
    "SELF": 3,
    "NAME_BARE": 3,
    "ROLE_BARE": 4,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gzip_uncompressed(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs)


def fmedian(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def geo_mean(xs: list[float]) -> float:
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def sample_sd(xs: list[float]) -> float:
    m = fmean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def exact_quantile_sorted(ys: list[float], q: float) -> float:
    """Linear-interpolation quantile matching score_olmo3_c0.py."""
    n = len(ys)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def get_token_counts(prompts_8: list[dict[str, Any]]) -> tuple[dict[str, int], str]:
    """Compute exact OLMo-3 token counts via local cached tokenizer, verifying against pinned values."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "allenai/Olmo-3-7B-Instruct",
            revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
            local_files_only=True,
        )
        counts: dict[str, int] = {}
        for p in prompts_8:
            counts[p["id"]] = len(tok.encode(p["text"], add_special_tokens=False))
        for name, txt in EXP4B_FRAMES:
            counts[name] = len(tok.encode(txt, add_special_tokens=False))
        if counts != VERIFIED_OLMO3_TOKEN_COUNTS:
            raise ValueError(f"Tokenizer counts mismatch: {counts} != {VERIFIED_OLMO3_TOKEN_COUNTS}")
        return counts, "local_olmo3_tokenizer_verified"
    except Exception:
        return dict(VERIFIED_OLMO3_TOKEN_COUNTS), "pinned_olmo3_tokenizer_verified"


def build_constants(
    c0_constants_path: Path,
    c0_receipt_path: Path,
    exp4b_path: Path,
    bank_path: Path,
    r_lens_path: Path,
) -> dict[str, Any]:
    with c0_constants_path.open("r", encoding="utf-8") as f:
        c0c = json.load(f)
    with gzip.open(c0_receipt_path, "rt", encoding="utf-8") as f:
        rec = json.load(f)
    with gzip.open(exp4b_path, "rt", encoding="utf-8") as f:
        exp = json.load(f)

    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    r_lens = torch.load(r_lens_path, map_location="cpu", weights_only=False)
    R_out_all = r_lens["means"]["out"]  # [32, 4096, 4096]

    layers_4 = [12, 18, 24, 27]
    layers_3 = [18, 24, 27]
    targets = rec["targets_16"]
    t_strs = [t["string"] for t in targets]
    t_doms = {t["string"]: t["domain"] for t in targets}
    prompts_8 = rec["prompts_8"]
    pmap = {p["id"]: p for p in prompts_8}

    tok_counts, tok_source = get_token_counts(prompts_8)

    # 1. Exact prompt lengths & Exp4B frame lengths
    prompts_with_lengths = [
        {
            "id": p["id"],
            "format": p["format"],
            "style": p["style"],
            "n_tokens": tok_counts[p["id"]],
            "text": p["text"],
        }
        for p in prompts_8
    ]
    exp4b_frames_meta = [
        {"frame": name, "n_tokens": tok_counts[name], "text": txt}
        for name, txt in EXP4B_FRAMES
    ]

    # 2. Exact pred_gap_D3 from frozen c0_vector_bank.pt -> lift guard delta = 0.05 * mean(pred_gap_D3)
    pg_map: dict[tuple[int, str], float] = {}
    pg_vals: list[float] = []
    for L in layers_4:
        ptg = bank["layers_bank"][L]["per_target_geom"]
        for s in t_strs:
            pg = float(ptg[s]["pred_gap_D3"])
            pg_map[(L, s)] = pg
            pg_vals.append(pg)
    mean_pred_gap_D3 = fmean(pg_vals)
    c0b1b_delta_tot_guard = 0.05 * mean_pred_gap_D3

    # 3. Stage C0 published "last" baselines (Regime D on 4L, Regime N on 3L)
    by_layer_c0: dict[str, Any] = {}
    all_cells_c0: list[float] = []
    bare_cells_c0: list[float] = []
    chat_cells_c0: list[float] = []
    per_target_4L_c0: dict[str, list[float]] = {s: [] for s in t_strs}

    for L in layers_4:
        rd = rec["regime_D"][str(L)]
        l_ratios: list[float] = []
        l_tot: list[float] = []
        l_attn: list[float] = []
        l_mlp: list[float] = []
        for s in t_strs:
            rmf_by_p = {tr["prompt_id"]: tr for tr in rd["R_mf"][s]}
            dm_by_p: dict[str, list[dict[str, Any]]] = {}
            for draw in rd["direct_matched"][s]:
                for tr in draw:
                    dm_by_p.setdefault(tr["prompt_id"], []).append(tr)
            gaps_all, gaps_bare, gaps_chat = [], [], []
            attns_all, mlps_all = [], []
            for p in prompts_8:
                pid = p["id"]
                rtr = rmf_by_p[pid]
                dms = dm_by_p[pid]
                g_tot = rtr["exact_dlogit"] - fmean([x["exact_dlogit"] for x in dms])
                g_attn = rtr["sum_attn"] - fmean([x["sum_attn"] for x in dms])
                g_mlp = rtr["sum_mlp"] - fmean([x["sum_mlp"] for x in dms])
                gaps_all.append(g_tot)
                attns_all.append(g_attn)
                mlps_all.append(g_mlp)
                if p["format"] == "bare":
                    gaps_bare.append(g_tot)
                else:
                    gaps_chat.append(g_tot)
            pg = pg_map[(L, s)]
            r_all8 = fmean(gaps_all) / pg
            l_ratios.append(r_all8)
            all_cells_c0.append(r_all8)
            bare_cells_c0.append(fmean(gaps_bare) / pg)
            chat_cells_c0.append(fmean(gaps_chat) / pg)
            per_target_4L_c0[s].append(r_all8)
            l_tot.append(fmean(gaps_all))
            l_attn.append(fmean(attns_all))
            l_mlp.append(fmean(mlps_all))
        by_layer_c0[str(L)] = {
            "gap_ratio_median": round(fmedian(l_ratios), 6),
            "mean_total_gap": round(fmean(l_tot), 4),
            "mean_attn_gap": round(fmean(l_attn), 4),
            "mean_mlp_gap": round(fmean(l_mlp), 4),
        }

    pt_means_c0 = [fmean(per_target_4L_c0[s]) for s in t_strs]

    # Regime N measured C0 anchors on L in {18, 24, 27} (3L) + rand_iso 3L vs 4L
    by_t_rmf_cross: dict[str, list[float]] = {s: [] for s in t_strs}
    by_t_rmf_self: dict[str, list[float]] = {s: [] for s in t_strs}
    by_t_rr_self: dict[str, list[float]] = {s: [] for s in t_strs}
    by_t_ll_self: dict[str, list[float]] = {s: [] for s in t_strs}

    for L in layers_3:
        rn = rec["regime_N"][str(L)]
        for src in t_strs:
            d_src = t_doms[src]
            for tr in rn["R_mf"][src]:
                by_t_rmf_self[src].append(float(tr["exact_dlogit"]))
                for dst, val in zip(t_strs, tr["all_16_dlogits"]):
                    if t_doms[dst] != d_src:
                        by_t_rmf_cross[dst].append(abs(float(val)))
            for tr in rn["R_ridge"][src]:
                by_t_rr_self[src].append(float(tr["exact_dlogit"]))
            for tr in rn["logit_lens"][src]:
                by_t_ll_self[src].append(float(tr["exact_dlogit"]))

    meas_rmf_b = [fmean(by_t_rmf_cross[s]) for s in t_strs]
    meas_rmf_self = [fmean(by_t_rmf_self[s]) for s in t_strs]
    meas_rr_self = [fmean(by_t_rr_self[s]) for s in t_strs]
    meas_ll_self = [fmean(by_t_ll_self[s]) for s in t_strs]
    meas_rmf_s = [b / sm for b, sm in zip(meas_rmf_b, meas_rmf_self)]
    meas_self_ratio = [sr / sm for sr, sm in zip(meas_rr_self, meas_rmf_self)]

    # Also compute exact 4L {18, 21, 24, 27} C0.4 R_mf cross-domain arith/geo means from receipt
    by_t_rmf_cross_4L: dict[str, list[float]] = {s: [] for s in t_strs}
    for L_str in ("18", "21", "24", "27"):
        rn = rec["regime_N"][L_str]
        for src in t_strs:
            d_src = t_doms[src]
            for tr in rn["R_mf"][src]:
                for dst, val in zip(t_strs, tr["all_16_dlogits"]):
                    if t_doms[dst] != d_src:
                        by_t_rmf_cross_4L[dst].append(abs(float(val)))
    meas_rmf_b_4L = [fmean(by_t_rmf_cross_4L[s]) for s in t_strs]

    # Exact 512-pair C0.4 construction (32 rand_iso draws x 16 readout targets, averaged over 8 prompts x layers)
    rand_3L_cells: list[float] = []
    rand_4L_cells: list[float] = []
    rand_3L_pairs: list[float] = []
    rand_4L_pairs: list[float] = []
    for draw_idx in range(32):
        for t_idx in range(16):
            p3, p4 = [], []
            for L_str in ("18", "21", "24", "27"):
                for tr in rec["regime_N"][L_str]["rand_iso"][draw_idx]:
                    v = abs(float(tr["all_16_dlogits"][t_idx]))
                    p4.append(v)
                    rand_4L_cells.append(v)
                    if L_str in ("18", "24", "27"):
                        p3.append(v)
                        rand_3L_cells.append(v)
            rand_3L_pairs.append(fmean(p3))
            rand_4L_pairs.append(fmean(p4))
    rand_3L_cells.sort()
    rand_4L_cells.sort()
    rand_3L_pairs.sort()
    rand_4L_pairs.sort()
    f_mean_3L = fmean(rand_3L_pairs)
    f_p95_3L_pair = exact_quantile_sorted(rand_3L_pairs, 0.95)
    f_p95_3L_cell = exact_quantile_sorted(rand_3L_cells, 0.95)
    f_mean_4L = fmean(rand_4L_pairs)
    f_p95_4L_pair = exact_quantile_sorted(rand_4L_pairs, 0.95)
    f_p95_4L_cell = exact_quantile_sorted(rand_4L_cells, 0.95)

    # Deterministically select the 2 Regime-D "last" reproducibility anchor targets
    # closest to the median of per-target 4L r_last in Stage C0.
    med_pt_r_last = fmedian(pt_means_c0)
    ranked_anchor = sorted(
        zip(t_strs, pt_means_c0),
        key=lambda kv: (abs(kv[1] - med_pt_r_last), kv[0]),
    )
    anchor_targets_2 = [ranked_anchor[0][0], ranked_anchor[1][0]]

    # 4. Exploratory Exp 4B per-frame and 4-frame-averaged analogues
    pred_exp4b = {
        (int(r["layer"]), r["target"]): float(r["pred_gap_D3"])
        for r in c0c["exploratory_exp4b_per_target_forecasts"]
    }
    e4v = [r for r in exp["exp4_highway_virtual"] if r["site"] == "post_norm"]
    by_ltm = {(int(r["layer"]), r["target"], r["method"]): r for r in e4v}
    exp_targets = sorted(set(r["target"] for r in e4v))
    frame_names = [fn for fn, _ in EXP4B_FRAMES]

    exp4b_per_frame: dict[str, Any] = {}
    for fr in frame_names:
        pooled_fr: list[float] = []
        by_L_fr: dict[str, Any] = {}
        for L in (12, 18, 24):
            ratios, g_tots, g_attns, g_mlps = [], [], [], []
            for t in exp_targets:
                r_rmf = by_ltm[(L, t, "R_mf")]["frames"][fr]
                r_ll = by_ltm[(L, t, "logit_lens")]["frames"][fr]
                s_rmf = 3.0 / r_rmf["direct_linear"]
                s_ll = 3.0 / r_ll["direct_linear"]
                gt = r_rmf["exact_dlogit"] * s_rmf - r_ll["exact_dlogit"] * s_ll
                ga = r_rmf["sum_downstream_attn"] * s_rmf - r_ll["sum_downstream_attn"] * s_ll
                gm = r_rmf["sum_downstream_mlp"] * s_rmf - r_ll["sum_downstream_mlp"] * s_ll
                ratios.append(gt / pred_exp4b[(L, t)])
                g_tots.append(gt)
                g_attns.append(ga)
                g_mlps.append(gm)
            pooled_fr.extend(ratios)
            by_L_fr[str(L)] = {
                "gap_ratio_median": round(fmedian(ratios), 4),
                "mean_gap_tot_D3": round(fmean(g_tots), 4),
                "mean_gap_attn_D3": round(fmean(g_attns), 4),
                "mean_gap_mlp_D3": round(fmean(g_mlps), 4),
            }
        exp4b_per_frame[fr] = {
            "n_tokens": tok_counts[fr],
            "pooled_L12_18_24_gap_ratio_median": round(fmedian(pooled_fr), 4),
            "by_layer": by_L_fr,
        }

    # 4-frame-averaged per-layer (matches C0 design table)
    four_fr_by_L: dict[str, float] = {}
    four_fr_pooled: list[float] = []
    attn_lift_by_L: dict[str, Any] = {}
    for L in (12, 18, 24):
        l_ratios_4f, l_tots_4f, l_attns_4f, l_mlps_4f = [], [], [], []
        for t in exp_targets:
            t_gts, t_gas, t_gms = [], [], []
            for fr in frame_names:
                r_rmf = by_ltm[(L, t, "R_mf")]["frames"][fr]
                r_ll = by_ltm[(L, t, "logit_lens")]["frames"][fr]
                s_rmf = 3.0 / r_rmf["direct_linear"]
                s_ll = 3.0 / r_ll["direct_linear"]
                t_gts.append(r_rmf["exact_dlogit"] * s_rmf - r_ll["exact_dlogit"] * s_ll)
                t_gas.append(r_rmf["sum_downstream_attn"] * s_rmf - r_ll["sum_downstream_attn"] * s_ll)
                t_gms.append(r_rmf["sum_downstream_mlp"] * s_rmf - r_ll["sum_downstream_mlp"] * s_ll)
            gt_m = fmean(t_gts)
            l_ratios_4f.append(gt_m / pred_exp4b[(L, t)])
            l_tots_4f.append(gt_m)
            l_attns_4f.append(fmean(t_gas))
            l_mlps_4f.append(fmean(t_gms))
        four_fr_by_L[str(L)] = round(fmedian(l_ratios_4f), 4)
        four_fr_pooled.extend(l_ratios_4f)

        # Aggregate ratio of layer means vs Stage C0 "last"
        exp_a = fmean(l_attns_4f)
        exp_m = fmean(l_mlps_4f)
        exp_t = fmean(l_tots_4f)
        c0_a = by_layer_c0[str(L)]["mean_attn_gap"]
        c0_t = by_layer_c0[str(L)]["mean_total_gap"]
        d_a = exp_a - c0_a
        d_t = exp_t - c0_t
        attn_lift_by_L[str(L)] = {
            "exp4b_all_mean_attn_gap_D3": round(exp_a, 4),
            "exp4b_all_mean_mlp_gap_D3": round(exp_m, 4),
            "exp4b_all_mean_total_gap_D3": round(exp_t, 4),
            "implied_delta_attn_all_minus_last": round(d_a, 4),
            "implied_delta_total_all_minus_last": round(d_t, 4),
            "implied_aggregate_attn_share_of_lift": round(d_a / d_t, 4),
        }

    mean_3L_forecast_delta_attn = fmean(
        [attn_lift_by_L[str(L)]["implied_delta_attn_all_minus_last"] for L in (12, 18, 24)]
    )
    c0b1b_attn_lift_floor_50pct = 0.50 * mean_3L_forecast_delta_attn

    # 5. Frozen R-lens cross-domain predictions via exact documented formula:
    #    pred_B_cross(arm, t) = mean_{t' cross, L in {18,24,27}} dose_N(L) * |u_t^T R_L w_{arm}(t')|
    U = bank["u_vectors"].float()  # [16, 4096]
    pred_b_rmf, pred_b_rr, pred_b_ll, pred_b_rand = [], [], [], []
    per_target_c0b2: list[dict[str, Any]] = []

    for i, dst in enumerate(t_strs):
        vr, vrr, vll, vrd = [], [], [], []
        for L in layers_3:
            R_L = R_out_all[L].float()
            dose_N = float(rec["regime_N"][str(L)]["dose_N"])
            W_rmf = bank["layers_bank"][L]["R_mf"].float()
            W_rr = bank["layers_bank"][L]["R_ridge"].float()
            W_ll = bank["layers_bank"][L]["logit_lens"].float()
            W_rd = bank["layers_bank"][L]["rand_iso"].float()
            uR = (U[i] @ R_L) * dose_N
            p_rmf = uR @ W_rmf.T
            p_rr = uR @ W_rr.T
            p_ll = uR @ W_ll.T
            p_rd = uR @ W_rd.T
            for j, src in enumerate(t_strs):
                if t_doms[src] != t_doms[dst]:
                    vr.append(abs(float(p_rmf[j].item())))
                    vrr.append(abs(float(p_rr[j].item())))
                    vll.append(abs(float(p_ll[j].item())))
            for x in p_rd.tolist():
                vrd.append(abs(float(x)))
        pb_rmf = fmean(vr)
        pb_rr = fmean(vrr)
        pb_ll = fmean(vll)
        pb_rd = fmean(vrd)
        pred_b_rmf.append(pb_rmf)
        pred_b_rr.append(pb_rr)
        pred_b_ll.append(pb_ll)
        pred_b_rand.append(pb_rd)
        per_target_c0b2.append(
            {
                "target": dst,
                "domain": t_doms[dst],
                "measured_C0_B_cross_Rmf": round(meas_rmf_b[i], 6),
                "measured_C0_self_Rmf": round(meas_rmf_self[i], 6),
                "measured_C0_self_Rridge": round(meas_rr_self[i], 6),
                "measured_C0_self_ratio_Rridge_over_Rmf": round(meas_self_ratio[i], 6),
                "measured_C0_S_cross_Rmf": round(meas_rmf_s[i], 6),
                "frozen_pred_B_cross_Rmf": round(pb_rmf, 4),
                "frozen_pred_B_cross_Rridge": round(pb_rr, 4),
                "frozen_pred_B_cross_ll": round(pb_ll, 4),
                "frozen_pred_B_cross_rand": round(pb_rd, 4),
                "k_mf_pred_over_meas": round(pb_rmf / meas_rmf_b[i], 4),
            }
        )

    geo_meas_rmf = geo_mean(meas_rmf_b)
    geo_pred_rmf = geo_mean(pred_b_rmf)
    geo_pred_rr = geo_mean(pred_b_rr)
    geo_self_ratio = geo_mean(meas_self_ratio)
    k_mf_geo = geo_pred_rmf / geo_meas_rmf
    base_abs_ratio_k1 = geo_meas_rmf / geo_pred_rr
    base_norm_ratio_k1 = base_abs_ratio_k1 * geo_self_ratio

    sensitivity_table = []
    for k_ridge in (1.00, 1.133, 1.25, 1.50, 1.75, 2.00, round(k_mf_geo, 4)):
        est_rr_b = geo_pred_rr / k_ridge
        est_abs_r = base_abs_ratio_k1 * k_ridge
        est_norm_r = base_norm_ratio_k1 * k_ridge
        sensitivity_table.append(
            {
                "k_ridge_overpred_factor": round(k_ridge, 4),
                "implied_B_cross_Rridge_geo": round(est_rr_b, 4),
                "implied_abs_ratio_Rmf_over_Rridge": round(est_abs_r, 4),
                "implied_norm_ratio_Rmf_over_Rridge": round(est_norm_r, 4),
                "implied_ratio_to_F_mean_3L": round(est_rr_b / f_mean_3L, 4),
                "clears_C0b2a_1p25_point": bool(est_abs_r >= 1.25 and est_norm_r >= 1.25),
                "clears_C0b2b_noise_floor_point": bool(est_rr_b <= f_mean_3L + 0.10),
            }
        )

    # 6. Exact 2^16 C0b.2a power & equivalence tables (deterministic seed 20261123)
    signs = (((np.arange(65536)[:, None] >> np.arange(16)[None, :]) & 1) * 2 - 1).astype(np.float32)
    rng = np.random.default_rng(20261123)
    self_shift = math.log(geo_self_ratio)

    c0b2a_power: dict[str, dict[str, float]] = {}
    c0b2a_equiv_power_at_1p00: dict[str, float] = {}
    for sd in (0.10, 0.15, 0.20):
        sd_key = f"SD_log_{sd:.2f}"
        c0b2a_power[sd_key] = {}
        for true_r in (1.00, 1.15, 1.25, 1.40, 1.50, 1.75, 2.00, 2.52):
            mu = math.log(true_r)
            n_mc = 2000
            U_abs = rng.normal(mu, sd, size=(n_mc, 16))
            U_norm = U_abs + self_shift + rng.normal(0.0, 0.02, size=(n_mc, 16))
            cand = (
                (U_abs.mean(axis=1) >= math.log(1.25))
                & (U_norm.mean(axis=1) >= math.log(1.25))
                & ((U_abs > 0).sum(axis=1) >= 13)
            )
            passes = 0
            cand_idx = np.where(cand)[0]
            if len(cand_idx) > 0:
                U_c = U_abs[cand_idx].astype(np.float32)
                obs = U_c.sum(axis=1)
                for b0 in range(0, len(cand_idx), 256):
                    U_b = U_c[b0 : b0 + 256]
                    obs_b = obs[b0 : b0 + 256]
                    flips = signs @ U_b.T  # [65536, B]
                    ps = (flips >= obs_b[None, :] - 1e-6).mean(axis=0)
                    passes += int((ps <= 0.05).sum())
            c0b2a_power[sd_key][f"{true_r:.2f}x"] = round(100.0 * passes / n_mc, 1)

        # Mirror equivalence power at true_r=1.00: BOTH q95(u_abs) < 1.25 AND q95(u_norm) < 1.25
        U0_abs = rng.normal(0.0, sd, size=(5000, 16))
        U0_norm = U0_abs + self_shift + rng.normal(0.0, 0.02, size=(5000, 16))
        ub_abs = U0_abs.mean(axis=1) + 1.75305 * U0_abs.std(axis=1, ddof=1) / math.sqrt(16)
        ub_norm = U0_norm.mean(axis=1) + 1.75305 * U0_norm.std(axis=1, ddof=1) / math.sqrt(16)
        c0b2a_equiv_power_at_1p00[sd_key] = round(
            100.0 * float(((ub_abs < math.log(1.25)) & (ub_norm < math.log(1.25))).mean()), 1
        )

    # Full C0b.1a conjunction power:
    # (i) pooled 4L [0.65,1.35] & 90% CI in (0.55,1.50),
    # (ii) Bare 4L [0.65,1.35] & ChatML 4L [0.65,1.35],
    # (iii) per-layer |median(ChatML(L)) - median(Bare(L))| <= 0.35 for all 4L,
    # (iv) all 8 (layer, format) cell medians in [0.50,1.50].
    c0b1a_power: dict[str, dict[str, float]] = {}
    for sd in (0.06, 0.10, 0.15):
        sd_key = f"SD_t_{sd:.2f}"
        c0b1a_power[sd_key] = {}
        for true_r in (0.45, 0.55, 0.65, 0.75, 0.85, 1.00, 1.25, 1.45, 1.60):
            n_mc = 2000
            t_eff = rng.normal(true_r, sd, size=(n_mc, 16, 1))
            l_eff = rng.normal(0.0, 0.05, size=(n_mc, 16, 4))
            bare = t_eff + l_eff + rng.normal(0.0, 0.04, size=(n_mc, 16, 4))
            chat = t_eff + l_eff + rng.normal(0.0, 0.04, size=(n_mc, 16, 4))
            pooled = 0.5 * (bare + chat)
            meds = np.median(pooled.reshape(n_mc, 64), axis=1)
            med_b = np.median(bare.reshape(n_mc, 64), axis=1)
            med_c = np.median(chat.reshape(n_mc, 64), axis=1)
            se = 1.253 * np.std(pooled.mean(axis=2), axis=1, ddof=1) / 4.0
            lo = meds - 1.645 * se
            hi = meds + 1.645 * se
            per_l_b = np.median(bare, axis=1)  # [n_mc, 4]
            per_l_c = np.median(chat, axis=1)  # [n_mc, 4]
            per_l_diff = np.abs(per_l_c - per_l_b).max(axis=1)
            ok = (
                (meds >= 0.65)
                & (meds <= 1.35)
                & (lo > 0.55)
                & (hi < 1.50)
                & (med_b >= 0.65)
                & (med_b <= 1.35)
                & (med_c >= 0.65)
                & (med_c <= 1.35)
                & (per_l_diff <= 0.35)
                & (per_l_b.min(axis=1) >= 0.50)
                & (per_l_b.max(axis=1) <= 1.50)
                & (per_l_c.min(axis=1) >= 0.50)
                & (per_l_c.max(axis=1) <= 1.50)
            )
            c0b1a_power[sd_key][f"{true_r:.2f}"] = round(100.0 * float(ok.mean()), 1)

    return {
        "schema_version": "c0b_constants_v4",
        "study": "stage_c0b_olmo3_7b_all_vs_last_and_ridge_whitening",
        "generator_script": "olmo3_virtual_and_highway/build_olmo3_c0b_constants.py",
        "model": "allenai/Olmo-3-7B-Instruct",
        "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        "parent_c0_constants_sha256": sha256_file(c0_constants_path),
        "parent_c0_receipt_gz_sha256": sha256_file(c0_receipt_path),
        "parent_c0_receipt_uncompressed_sha256": sha256_gzip_uncompressed(c0_receipt_path),
        "parent_c0_bank_sha256": sha256_file(bank_path),
        "r_lens_sha256": sha256_file(r_lens_path),
        "layers_4L": layers_4,
        "layers_L18_plus_3L": layers_3,
        "reproducibility_anchor_targets_regime_D": anchor_targets_2,
        "reproducibility_anchor_selection_rule": "2 targets closest to the median per-target 4L r_last in Stage C0",
        "targets_16": targets,
        "prompts_8": prompts_with_lengths,
        "prompt_length_summary": {
            "source": tok_source,
            "bare_min_max_mean": [7, 13, 9.75],
            "chatml_min_max_mean": [31, 43, 36.5],
            "exp4b_frames_min_max_mean": [3, 15, 6.25],
            "exp4b_frames": exp4b_frames_meta,
        },
        "stage_c0_last_baselines": {
            "pooled_4L_gap_ratio_median": round(fmedian(all_cells_c0), 6),
            "c0_2_6L_band_reference_median": 0.283674,
            "bare_4L_gap_ratio_median": round(fmedian(bare_cells_c0), 6),
            "chatml_4L_gap_ratio_median": round(fmedian(chat_cells_c0), 6),
            "per_target_4L_gap_ratio_mean": round(fmean(pt_means_c0), 6),
            "per_target_4L_gap_ratio_sd": round(sample_sd(pt_means_c0), 6),
            "mean_pred_gap_D3_4L_16t": round(mean_pred_gap_D3, 6),
            "c0b1b_delta_tot_guard_5pct": round(c0b1b_delta_tot_guard, 6),
            "c0b1b_mean_3L_forecast_delta_attn": round(mean_3L_forecast_delta_attn, 4),
            "c0b1b_attn_lift_floor_50pct": round(c0b1b_attn_lift_floor_50pct, 4),
            "by_layer": by_layer_c0,
            "regime_N_3L_L18_24_27_measured": {
                "R_mf_B_cross_arith_mean": round(fmean(meas_rmf_b), 6),
                "R_mf_B_cross_geo_mean": round(geo_meas_rmf, 6),
                "R_mf_B_cross_4L_C04_arith_mean": round(fmean(meas_rmf_b_4L), 6),
                "R_mf_B_cross_4L_C04_geo_mean": round(geo_mean(meas_rmf_b_4L), 6),
                "R_mf_B_cross_sd": round(sample_sd(meas_rmf_b), 6),
                "R_mf_S_cross_arith_mean": round(fmean(meas_rmf_s), 6),
                "R_mf_S_cross_geo_mean": round(geo_mean(meas_rmf_s), 6),
                "R_mf_self_dlogit_geo_mean": round(geo_mean(meas_rmf_self), 6),
                "R_ridge_self_dlogit_geo_mean": round(geo_mean(meas_rr_self), 6),
                "logit_lens_self_dlogit_geo_mean": round(geo_mean(meas_ll_self), 6),
                "geo_self_ratio_Rridge_over_Rmf": round(geo_self_ratio, 6),
                "rand_iso_3L_F_mean": round(f_mean_3L, 6),
                "rand_iso_3L_F_p95_512pair": round(f_p95_3L_pair, 6),
                "rand_iso_3L_F_p95_cell_n12288": round(f_p95_3L_cell, 6),
                "rand_iso_4L_C04_reference_F_mean": round(f_mean_4L, 6),
                "rand_iso_4L_C04_reference_F_p95_512pair": round(f_p95_4L_pair, 6),
                "rand_iso_4L_C04_reference_F_p95_cell_n16384": round(f_p95_4L_cell, 6),
            },
        },
        "exploratory_exp4b_all_analogues": {
            "note": "Exploratory 4-persona, 3-layer (L12/18/24), 4-frame (3-15 token) linear rescale to Direct_linear=3.0 using R_mf - logit_lens (C0b.1a uses R_mf - direct_matched); NOT out-of-sample. C0 unrescaled dose analogue was 0.8308 pooled (WHO_CHAT_TF 0.7979, L12=0.4809).",
            "c0_unrescaled_four_frame_pooled_median": 0.8308,
            "c0_unrescaled_WHO_CHAT_TF_L12_median": 0.4809,
            "four_frame_avg_pooled_L12_18_24_gap_ratio_median": round(fmedian(four_fr_pooled), 4),
            "four_frame_avg_per_layer_gap_ratio_median": four_fr_by_L,
            "per_frame_L12_18_24": exp4b_per_frame,
            "aggregate_layer_mean_attn_vs_mlp_lift_D3": attn_lift_by_L,
        },
        "c0b2_ridge_whitening_anchors_L18_24_27": {
            "formula": "pred_B_cross(arm, t) = mean_{t' cross-domain, L in {18,24,27}} dose_N(L) * |u_t^T R_L w_{arm}(t')| (unscaled linear map, r_c=1)",
            "frozen_mean_pred_B_cross_Rmf": round(fmean(pred_b_rmf), 4),
            "frozen_geo_pred_B_cross_Rmf": round(geo_pred_rmf, 4),
            "frozen_mean_pred_B_cross_Rridge": round(fmean(pred_b_rr), 4),
            "frozen_geo_pred_B_cross_Rridge": round(geo_pred_rr, 4),
            "frozen_mean_pred_B_cross_ll": round(fmean(pred_b_ll), 4),
            "frozen_mean_pred_B_cross_rand": round(fmean(pred_b_rand), 4),
            "k_mf_geo_pred_over_measured": round(k_mf_geo, 4),
            "measured_anchor_abs_ratio_at_k_ridge_1": round(base_abs_ratio_k1, 4),
            "measured_anchor_norm_ratio_at_k_ridge_1": round(base_norm_ratio_k1, 4),
            "k_ridge_sensitivity_table": sensitivity_table,
            "per_target": per_target_c0b2,
        },
        "power_simulations": {
            "seed": 20261123,
            "c0b1a_two_sided_calibration_power_pct": c0b1a_power,
            "c0b2a_exact_2pow16_conjunction_power_pct": c0b2a_power,
            "c0b2a_equivalence_refutation_power_at_1p00x_pct": c0b2a_equiv_power_at_1p00,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build frozen c0b_prereg_constants.json (v2).")
    ap.add_argument("--c0-constants", type=Path, default=Path(DEFAULT_C0_CONSTANTS))
    ap.add_argument("--c0-receipt", type=Path, default=Path(DEFAULT_C0_RECEIPT))
    ap.add_argument("--exp4b", type=Path, default=Path(DEFAULT_EXP4B))
    ap.add_argument("--bank", type=Path, default=Path(DEFAULT_BANK))
    ap.add_argument("--r-lens", type=Path, default=Path(DEFAULT_R_LENS))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = ap.parse_args()

    payload = build_constants(
        c0_constants_path=args.c0_constants,
        c0_receipt_path=args.c0_receipt,
        exp4b_path=args.exp4b,
        bank_path=args.bank,
        r_lens_path=args.r_lens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    digest = sha256_file(args.out)
    print(f"Wrote {args.out} ({len(text)} bytes) | sha256 = {digest}")
    print("k_ridge sensitivity table:")
    for row in payload["c0b2_ridge_whitening_anchors_L18_24_27"]["k_ridge_sensitivity_table"]:
        print(" ", row)
    print("delta_tot_guard:", payload["stage_c0_last_baselines"]["c0b1b_delta_tot_guard_5pct"])
    print("anchor_targets_2:", payload["reproducibility_anchor_targets_regime_D"])
    rn_m = payload["stage_c0_last_baselines"]["regime_N_3L_L18_24_27_measured"]
    print("3L rand_iso F_mean / F_p95_512pair / F_p95_cell:", rn_m["rand_iso_3L_F_mean"], rn_m["rand_iso_3L_F_p95_512pair"], rn_m["rand_iso_3L_F_p95_cell_n12288"])
    print("4L C0.4 rand_iso F_mean / F_p95_512pair:", rn_m["rand_iso_4L_C04_reference_F_mean"], rn_m["rand_iso_4L_C04_reference_F_p95_512pair"])
    print("power_simulations:", json.dumps(payload["power_simulations"], indent=2))


if __name__ == "__main__":
    main()
