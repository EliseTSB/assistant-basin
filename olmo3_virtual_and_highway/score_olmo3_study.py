#!/usr/bin/env python3
"""Stdlib-only deterministic scorer for Vesper's OLMo 3 7B Lens, Idea #5, Idea #2, and Idea #4 study.

Requires ONLY Python 3.10+ standard library (zero numpy/scipy/torch dependencies) so any clean
machine (including Gemini Task mode) can reproduce `scorecard.json` and `SCORECARD.md` byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def mann_whitney_two_sided_z_p(x: list[float], y: list[float]) -> dict:
    """Exact rank-sum U with tie-corrected normal-approximation two-sided p-value (stdlib only)."""
    n1, n2 = len(x), len(y)
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])
    ranks = [0.0] * (n1 + n2)
    tie_sum = 0
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        t = j - i
        if t > 1:
            tie_sum += t * (t * t - 1)
        i = j
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_sum / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return {"u1": round(u1, 2), "z": 0.0, "p_approx": 1.0}
    # Continuity-corrected z
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    return {"u1": round(u1, 2), "z": round(z, 4), "p_approx": round(p, 6)}


def score_all(lens_dir: Path, exp_dir: Path) -> dict:
    fit_meta_p = lens_dir / "fit_meta.json"
    lens_an_p = lens_dir / "olmo3_lens_analysis.json"
    census_p = lens_dir / "natural_neuron_census_preview.json"
    exp_p = exp_dir / "olmo3_idea2_and_idea4_results.json"

    fit_meta = json.loads(fit_meta_p.read_text())
    lens_an = json.loads(lens_an_p.read_text())
    census = json.loads(census_p.read_text())
    exp = json.loads(exp_p.read_text())

    # 1. Lens fit & cross-validation summary
    sh = fit_meta["split_half_fp32"]
    xval = lens_an["cross_validation"]
    mh_by_l = {r["layer"]: r["cos_mh"] for r in xval["vs_mhough"]}
    dan_by_l = {r["layer"]: r["cos_dan"] for r in xval["vs_danorel"]}
    norm_ours = [r["norm_ours"] for r in xval["vs_danorel"]]
    norm_dan = [r["norm_dan"] for r in xval["vs_danorel"]]

    lens_summary = {
        "model": fit_meta["model"],
        "revision": fit_meta["revision"],
        "docs": fit_meta["docs"],
        "seq_len": fit_meta["seq_len"],
        "seed": fit_meta["seed"],
        "gpu_name": fit_meta["gpu_name"],
        "fit_sec": fit_meta["fit_sec"],
        "median_r_sec": fit_meta["median_r_sec"],
        "median_s_sec": fit_meta["median_s_sec"],
        "median_total_sec": fit_meta["median_total_sec"],
        "peak_vram_alloc_gb": fit_meta["peak_memory"]["cuda_max_allocated_gb"],
        "peak_rss_gb": fit_meta["peak_memory"]["peak_rss_gb"],
        "split_half_endpoints": {
            "R_mid_L0": sh["R"]["mid"][0],
            "R_mid_L15": sh["R"]["mid"][15],
            "R_mid_L31": sh["R"]["mid"][31],
            "R_out_L0": sh["R"]["out"][0],
            "R_out_L15": sh["R"]["out"][15],
            "R_out_L30": sh["R"]["out"][30],
            "S_mid_L0": sh["S"]["mid"][0],
            "S_mid_L15": sh["S"]["mid"][15],
            "S_mid_L31": sh["S"]["mid"][31],
            "S_out_L0": sh["S"]["out"][0],
            "S_out_L15": sh["S"]["out"][15],
            "S_out_L31": sh["S"]["out"][31],
        },
        "cross_val_cos_mhough": {str(k): mh_by_l[k] for k in sorted(mh_by_l)},
        "cross_val_cos_danorel_selected": {
            str(k): dan_by_l[k] for k in (0, 6, 12, 15, 18, 19, 21, 24, 27, 30)
        },
        "norm_range_ours": [round(min(norm_ours), 2), round(max(norm_ours), 2)],
        "norm_range_danorel": [round(min(norm_dan), 2), round(max(norm_dan), 2)],
    }

    # 2. Idea #5: 1,024-head depth-controlled sliding vs full attention + census counts
    heads = lens_an["head_scan_1024"]["all_1024_heads"]
    by_L = {L: [r for r in heads if r["layer"] == L] for L in range(32)}
    full_interior = [3, 7, 11, 15, 19, 23, 27]
    depth_ctrl = {}
    for metric in ("ov_frob", "copy_bg_normed", "self_z", "role_z", "name_z", "persona_z", "abs_sr_z"):
        f_vals, s_vals, diffs = [], [], []
        for L in full_interior:
            fl = [r[metric] for r in by_L[L]]
            sl = [r[metric] for r in (by_L[L - 1] + by_L[L + 1])]
            f_vals.extend(fl)
            s_vals.extend(sl)
            diffs.append(mean(fl) - mean(sl))
        depth_ctrl[metric] = {
            "full_mean": round(mean(f_vals), 4),
            "flank_sliding_mean": round(mean(s_vals), 4),
            "mean_layer_diff": round(mean(diffs), 4),
            "pos_layers_of_7": sum(1 for d in diffs if d > 0),
            "mwu": mann_whitney_two_sided_z_p(f_vals, s_vals),
        }

    top_copy = sorted(heads, key=lambda r: -r["copy_bg_normed"])
    idea5_summary = {
        "n_heads": len(heads),
        "n_sliding": sum(1 for r in heads if r["layer_type"] == "sliding_attention"),
        "n_full": sum(1 for r in heads if r["layer_type"] == "full_attention"),
        "depth_controlled_full_vs_flanking_sliding": depth_ctrl,
        "ov_frob_spikes": {
            "L18": round(mean([r["ov_frob"] for r in by_L[18]]), 3),
            "L19_full": round(mean([r["ov_frob"] for r in by_L[19]]), 3),
            "L20": round(mean([r["ov_frob"] for r in by_L[20]]), 3),
            "L22": round(mean([r["ov_frob"] for r in by_L[22]]), 3),
            "L23_full": round(mean([r["ov_frob"] for r in by_L[23]]), 3),
            "L24": round(mean([r["ov_frob"] for r in by_L[24]]), 3),
        },
        "full_share_top20_copy": [
            sum(1 for r in top_copy[:20] if r["layer_type"] == "full_attention"), 20
        ],
        "full_share_top50_copy": [
            sum(1 for r in top_copy[:50] if r["layer_type"] == "full_attention"), 50
        ],
        "top_self_minus_role": lens_an["head_scan_1024"]["top_self_minus_role"][:5],
        "top_role_minus_self": lens_an["head_scan_1024"]["top_role_minus_self"][:5],
        "top_persona_copy": lens_an["head_scan_1024"]["top_persona_copy"][:5],
        "top_name_copy": sorted(heads, key=lambda r: -r["name_z"])[:5],
        "natural_census_counts": {
            "concept_candidates": len(census["concept"]),
            "slot_candidates": len(census["slot"]),
        },
    }

    # 3. Idea #2A: Write-geometry benchmark
    rows2a = exp["exp2a_write_geometry"]
    gens2a = exp["exp2a_generations"]
    exp2a_table = {}
    for site in ("post_norm", "pre_norm"):
        exp2a_table[site] = {}
        for L in (6, 12, 18, 24):
            exp2a_table[site][f"L{L}"] = {}
            for mult in (0.25, 0.75, 1.5):
                mkey = f"dose_{mult:.2f}"
                exp2a_table[site][f"L{L}"][mkey] = {}
                for method in ("R_ridge", "R_mf", "logit_lens", "random"):
                    sub = [
                        r for r in rows2a
                        if r["site"] == site and r["layer"] == L
                        and abs(r["dose_mult"] - mult) < 1e-6 and r["method"] == method
                    ]
                    who = [r["frames"]["WHO_CHAT_TF"] for r in sub]
                    exp2a_table[site][f"L{L}"][mkey][method] = {
                        "n": len(sub),
                        "who_dlogit": round(mean([x["dlogit"] for x in who]), 2),
                        "who_dlogit_z": round(mean([x["dlogit_z"] for x in who]), 2),
                        "who_kl": round(mean([x["kl_clean_int"] for x in who]), 3),
                        "who_top1_hits": [sum(1 for x in who if x["rank_int"] == 1), len(who)],
                        "self_dlogit": round(mean([r["frames"]["SELF"]["dlogit"] for r in sub]), 2),
                        "name_dlogit": round(mean([r["frames"]["NAME_BARE"]["dlogit"] for r in sub]), 2),
                        "role_dlogit": round(mean([r["frames"]["ROLE_BARE"]["dlogit"] for r in sub]), 2),
                    }
    exp2a_gen_hits = {}
    for site in ("post_norm", "pre_norm"):
        exp2a_gen_hits[site] = {}
        for L in (6, 12, 18, 24):
            exp2a_gen_hits[site][f"L{L}"] = {}
            for method in ("R_ridge", "R_mf", "logit_lens", "random"):
                sub = [g for g in gens2a if g["site"] == site and g["layer"] == L and g["method"] == method]
                hits = sum(1 for g in sub if g["target"].strip().lower() in g["generation"].lower())
                exp2a_gen_hits[site][f"L{L}"][method] = [hits, len(sub)]

    # 4. Idea #2B: Hand-built virtual SwiGLU neurons
    rows2b = exp["exp2b_virtual_neurons"]
    gens2b = exp["exp2b_generations"]
    exp2b_table = {}
    for site in ("post_norm", "pre_norm"):
        exp2b_table[site] = {}
        for L in (6, 12, 18, 24):
            exp2b_table[site][f"L{L}"] = {}
            for nk in ("virtual_concept_S_ridge", "virtual_concept_S_mf", "virtual_slot_S_ridge", "virtual_slot_contrastive"):
                sub = [
                    r for r in rows2b
                    if r["site"] == site and r["layer"] == L
                    and r["neuron_kind"] == nk and r["method"] == "R_ridge"
                ]
                exp2b_table[site][f"L{L}"][nk] = {
                    "n": len(sub),
                    "self_mean_act": round(mean([r["self_mean_act"] for r in sub]), 3),
                    "off_mean_act": round(mean([r["off_mean_act"] for r in sub]), 3),
                    "concept_act": round(mean([r["concept_act"] for r in sub]), 3),
                    "self_mean_dlogit": round(mean([r["self_mean_dlogit"] for r in sub]), 2),
                    "concept_dlogit": round(mean([r["concept_dlogit"] for r in sub]), 2),
                    "off_mean_kl": round(mean([r["off_mean_kl"] for r in sub]), 4),
                    "median_sel_self_vs_off": round(median([r["selectivity_self_vs_off"] for r in sub]), 1),
                    "median_sel_conc_vs_off": round(median([r["selectivity_conc_vs_off"] for r in sub]), 1),
                }

    exp2b_gen_audit = []
    for g in gens2b:
        if g["site"] == "post_norm" and g["method"] == "R_ridge" and g["layer"] in (18, 24):
            w = g["target"].strip().lower()
            exp2b_gen_audit.append({
                "layer": g["layer"],
                "neuron_kind": g["neuron_kind"],
                "target": g["target"],
                "who_contains_target": w in g["gen_who_are_you"].lower(),
                "france_contains_paris": "paris" in g["gen_capital_of_france"].lower(),
                "gen_who_are_you": g["gen_who_are_you"],
                "gen_capital_of_france": g["gen_capital_of_france"],
            })

    # 5. Idea #4: Exact L+1..31 Signal Highway & Late Suppression Cliff
    all_errs = [
        d["telescoping_err"]
        for rec in (exp["exp4_highway_natural"] + exp["exp4_highway_virtual"])
        for d in rec["frames"].values()
    ]
    exp4_natural_rows = []
    for rec in exp["exp4_highway_natural"]:
        w = rec["frames"]["WHO_CHAT_TF"]
        exp4_natural_rows.append({
            "unit": rec["unit"],
            "layer": rec["layer"],
            "kind": rec["kind"],
            "target": rec["target"],
            "who_dlogit": round(w["exact_dlogit"], 2),
            "who_direct": round(w["direct_linear"], 2),
            "who_frozen_de": round(w["frozen_direct_effect"], 2),
            "who_sum_attn": round(w["sum_downstream_attn"], 2),
            "who_sum_mlp": round(w["sum_downstream_mlp"], 2),
            "who_rmsnorm_shift": round(w["rmsnorm_shift"], 2),
            "who_late3_attn": round(w["late3_attn"], 2),
            "who_late3_mlp": round(w["late3_mlp"], 2),
            "who_cancel_ratio": round(w["path_cancellation_ratio"], 2),
            "self_dlogit": round(rec["frames"]["SELF"]["exact_dlogit"], 2),
            "name_dlogit": round(rec["frames"]["NAME_BARE"]["exact_dlogit"], 2),
            "role_dlogit": round(rec["frames"]["ROLE_BARE"]["exact_dlogit"], 2),
        })

    exp4_virtual_table = {}
    for site in ("post_norm", "pre_norm"):
        exp4_virtual_table[site] = {}
        for L in (6, 12, 18, 24):
            exp4_virtual_table[site][f"L{L}"] = {}
            for method in ("R_ridge", "R_mf", "logit_lens"):
                sub = [
                    r["frames"]["WHO_CHAT_TF"]
                    for r in exp["exp4_highway_virtual"]
                    if r["site"] == site and r["layer"] == L and r["method"] == method
                ]
                exp4_virtual_table[site][f"L{L}"][method] = {
                    "n": len(sub),
                    "exact_dlogit": round(mean([d["exact_dlogit"] for d in sub]), 2),
                    "direct_linear": round(mean([d["direct_linear"] for d in sub]), 2),
                    "frozen_direct_effect": round(mean([d["frozen_direct_effect"] for d in sub]), 2),
                    "sum_attn": round(mean([d["sum_downstream_attn"] for d in sub]), 2),
                    "sum_attn_sliding": round(mean([d["sum_downstream_attn_sliding"] for d in sub]), 2),
                    "sum_attn_full": round(mean([d["sum_downstream_attn_full"] for d in sub]), 2),
                    "sum_mlp": round(mean([d["sum_downstream_mlp"] for d in sub]), 2),
                    "rmsnorm_shift": round(mean([d["rmsnorm_shift"] for d in sub]), 2),
                    "late3_attn": round(mean([d["late3_attn"] for d in sub]), 2),
                    "late3_mlp": round(mean([d["late3_mlp"] for d in sub]), 2),
                    "cancel_ratio": round(mean([d["path_cancellation_ratio"] for d in sub]), 2),
                }

    # Short-context SWA vs flanking Full check on L6 post_norm highway across all 4 frames
    full_int_6 = [7, 11, 15, 19, 23, 27]
    swa_vs_full_highway = {}
    for method in ("R_mf", "R_ridge", "logit_lens"):
        by_l = {L: [] for L in range(7, 32)}
        for rec in exp["exp4_highway_virtual"]:
            if rec["site"] == "post_norm" and rec["layer"] == 6 and rec["method"] == method:
                for dec in rec["frames"].values():
                    for row in dec["per_layer"]:
                        by_l[row["layer"]].append((abs(row["attn"]), row["da_norm"]))
        f_a, s_a, f_n, s_n = [], [], [], []
        for L in full_int_6:
            f_a.extend(x[0] for x in by_l[L])
            f_n.extend(x[1] for x in by_l[L])
            for nb in (L - 1, L + 1):
                if nb in by_l:
                    s_a.extend(x[0] for x in by_l[nb])
                    s_n.extend(x[1] for x in by_l[nb])
        swa_vs_full_highway[method] = {
            "abs_attn_full": round(mean(f_a), 4),
            "abs_attn_flank_sliding": round(mean(s_a), 4),
            "abs_attn_mwu": mann_whitney_two_sided_z_p(f_a, s_a),
            "da_norm_full": round(mean(f_n), 3),
            "da_norm_flank_sliding": round(mean(s_n), 3),
            "da_norm_mwu": mann_whitney_two_sided_z_p(f_n, s_n),
        }

    return {
        "schema_version": "olmo3-study-scorecard-v1",
        "inputs_sha256": {
            "fit_meta.json": sha256_file(fit_meta_p),
            "olmo3_lens_analysis.json": sha256_file(lens_an_p),
            "natural_neuron_census_preview.json": sha256_file(census_p),
            "olmo3_idea2_and_idea4_results.json": sha256_file(exp_p),
            "r_lens.pt": exp["meta"]["lens_provenance"].get("r_lens_sha256"),
            "s_lens.pt": exp["meta"]["lens_provenance"].get("s_lens_sha256"),
        },
        "lens_fit_and_crossval": lens_summary,
        "idea5_1024_head_scan": idea5_summary,
        "idea2a_write_geometry": {
            "tables": exp2a_table,
            "who_generation_hits": exp2a_gen_hits,
        },
        "idea2b_virtual_swiglu": {
            "tables_R_ridge": exp2b_table,
            "l18_l24_post_norm_generations": exp2b_gen_audit,
        },
        "idea4_signal_highway": {
            "n_traces": len(all_errs),
            "telescoping_err": {
                "median": round(median(all_errs), 5),
                "p95": round(quantile(all_errs, 0.95), 5),
                "max": round(max(all_errs), 5),
            },
            "natural_neurons": exp4_natural_rows,
            "virtual_neurons": exp4_virtual_table,
            "l6_post_norm_full_vs_flank_sliding": swa_vs_full_highway,
        },
    }


def render_markdown(sc: dict) -> str:
    ls = sc["lens_fit_and_crossval"]
    sh = ls["split_half_endpoints"]
    i5 = sc["idea5_1024_head_scan"]
    i2a = sc["idea2a_write_geometry"]
    i2b = sc["idea2b_virtual_swiglu"]
    i4 = sc["idea4_signal_highway"]

    lines = [
        "# OLMo 3 7B RelP Neuron Lens, Virtual SwiGLU Synthesis & Signal Highway — Scorecard",
        "",
        f"- **Model:** `{ls['model']}@{ls['revision']}`",
        f"- **Lens fit:** `{ls['docs']}` WikiText-103 docs × `{ls['seq_len']}` tokens on `{ls['gpu_name']}` (`{ls['fit_sec']} s`, median `{ls['median_total_sec']} s/doc`, peak VRAM `{ls['peak_vram_alloc_gb']} GB`)",
        f"- **Split-half fp32 endpoints:** `R_mid` `{sh['R_mid_L0']} -> {sh['R_mid_L31']}`, `R_out` `{sh['R_out_L0']} -> {sh['R_out_L30']}`, `S_mid` `{sh['S_mid_L0']} -> {sh['S_mid_L31']}`, `S_out` `{sh['S_out_L0']} -> {sh['S_out_L31']}`",
        f"- **Cross-val `R_out` vs `mhough` 616-doc J-lens:** `L15 = {ls['cross_val_cos_mhough']['15']:+.4f}`, `L18 = {ls['cross_val_cos_mhough']['18']:+.4f}`, `L21 = {ls['cross_val_cos_mhough']['21']:+.4f}`, `L24 = {ls['cross_val_cos_mhough']['24']:+.4f}`, `L27 = {ls['cross_val_cos_mhough']['27']:+.4f}`, `L30 = {ls['cross_val_cos_mhough']['30']:+.4f}`",
        "",
        "## Idea #5: 1,024-Head Sliding vs. Full Attention (Depth-Controlled)",
        "",
        f"- **Top-20 / Top-50 copy heads in `full_attention`:** `{i5['full_share_top20_copy'][0]}/{i5['full_share_top20_copy'][1]}` (`25%` base rate), `{i5['full_share_top50_copy'][0]}/{i5['full_share_top50_copy'][1]}`",
        f"- **OV Frobenius norm spikes:** `L18={i5['ov_frob_spikes']['L18']}` -> `L19*={i5['ov_frob_spikes']['L19_full']}` -> `L20={i5['ov_frob_spikes']['L20']}`; `L22={i5['ov_frob_spikes']['L22']}` -> `L23*={i5['ov_frob_spikes']['L23_full']}` -> `L24={i5['ov_frob_spikes']['L24']}`",
        "",
        "## Idea #2A: Write-Geometry Benchmark (`WHO_CHAT_TF`, `post_norm`, `dose=1.50`)",
        "",
        "| Layer | `R_ridge` dL (`z`, top1) | `R_mf` dL (`z`, top1) | `logit_lens` dL (`z`, top1) | `random` dL (`z`) | Gen hits (`R_rg`/`R_mf`/`LL`/`Rnd` of 4) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for L in (6, 12, 18, 24):
        row = i2a["tables"]["post_norm"][f"L{L}"]["dose_1.50"]
        gh = i2a["who_generation_hits"]["post_norm"][f"L{L}"]
        lines.append(
            f"| L{L} | {row['R_ridge']['who_dlogit']:+.2f} ({row['R_ridge']['who_dlogit_z']:+.2f}, {row['R_ridge']['who_top1_hits'][0]}/8) "
            f"| {row['R_mf']['who_dlogit']:+.2f} ({row['R_mf']['who_dlogit_z']:+.2f}, {row['R_mf']['who_top1_hits'][0]}/8) "
            f"| {row['logit_lens']['who_dlogit']:+.2f} ({row['logit_lens']['who_dlogit_z']:+.2f}, {row['logit_lens']['who_top1_hits'][0]}/8) "
            f"| {row['random']['who_dlogit']:+.2f} ({row['random']['who_dlogit_z']:+.2f}) "
            f"| {gh['R_ridge'][0]}/{gh['R_mf'][0]}/{gh['logit_lens'][0]}/{gh['random'][0]} |"
        )

    lines.extend([
        "",
        "## Idea #2B: Hand-Built Virtual SwiGLU Neurons (`post_norm`, `R_ridge` write)",
        "",
        "| Layer | Variant | Self Act | Off Act | Concept Act | Self dL | Concept dL | Off KL | Med Sel (S/O) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for L in (18, 24):
        for nk in ("virtual_concept_S_ridge", "virtual_concept_S_mf", "virtual_slot_S_ridge", "virtual_slot_contrastive"):
            r = i2b["tables_R_ridge"]["post_norm"][f"L{L}"][nk]
            lines.append(
                f"| L{L} | `{nk}` | {r['self_mean_act']:+.3f} | {r['off_mean_act']:+.3f} | {r['concept_act']:+.3f} "
                f"| {r['self_mean_dlogit']:+.2f} | {r['concept_dlogit']:+.2f} | {r['off_mean_kl']:.4f} | {r['median_sel_self_vs_off']:.1f}× |"
            )

    lines.extend([
        "",
        f"## Idea #4: Exact `L+1..31` Signal Highway (`{i4['n_traces']}` traces; median telescoping err `{i4['telescoping_err']['median']}`, max `{i4['telescoping_err']['max']}`)",
        "",
        "### Virtual Neurons (`WHO_CHAT_TF`, `post_norm`, mean 4 personas)",
        "",
        "| Layer | Method | Exact dL | Direct | `ΣAttn` | `ΣMLP` | RMSNorm Shift | CancR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for L in (6, 12, 18, 24):
        for m in ("R_ridge", "R_mf", "logit_lens"):
            r = i4["virtual_neurons"]["post_norm"][f"L{L}"][m]
            lines.append(
                f"| L{L} | `{m}` | {r['exact_dlogit']:+.2f} | {r['direct_linear']:+.2f} | {r['sum_attn']:+.2f} "
                f"| {r['sum_mlp']:+.2f} | {r['rmsnorm_shift']:+.2f} | {r['cancel_ratio']:.2f} |"
            )

    lines.extend([
        "",
        "### Natural Neurons (`WHO_CHAT_TF`, push `+10.0`)",
        "",
        "| Unit | Kind | Target | Exact dL | Direct | `ΣAttn` | `ΣMLP` | RMSNorm Shift | Late3M | CancR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in i4["natural_neurons"]:
        lines.append(
            f"| `{r['unit']}` | `{r['kind']}` | `{r['target']}` | {r['who_dlogit']:+.2f} | {r['who_direct']:+.2f} "
            f"| {r['who_sum_attn']:+.2f} | {r['who_sum_mlp']:+.2f} | {r['who_rmsnorm_shift']:+.2f} "
            f"| {r['who_late3_mlp']:+.2f} | {r['who_cancel_ratio']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-dir", type=Path, default=Path("/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens"))
    ap.add_argument("--exp-dir", type=Path, default=Path("/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-idea2-idea4"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-idea2-idea4"))
    args = ap.parse_args()

    sc = score_all(args.lens_dir, args.exp_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sc_json = args.out_dir / "scorecard.json"
    sc_md = args.out_dir / "SCORECARD.md"
    sc_json.write_text(json.dumps(sc, indent=2) + "\n")
    md_text = render_markdown(sc)
    sc_md.write_text(md_text)
    print(md_text)
    print(f"Wrote {sc_json} ({sha256_file(sc_json)[:16]}...) and {sc_md} ({sha256_file(sc_md)[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
