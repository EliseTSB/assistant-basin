#!/usr/bin/env python3
"""Offline analysis script for OLMo 3 7B Idea #2 (Hand-Built Virtual Concept & Slot Neurons)
and Idea #4 (Exact L+1..31 Signal Highway & Late Suppression Cliff)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RES_PATH = Path("work/vesper-olmo3-7b-idea2-idea4/olmo3_idea2_and_idea4_results.json")


def main() -> int:
    data = json.loads(RES_PATH.read_text())
    meta = data["meta"]
    print("=== META ===")
    print(f"GPU={meta['gpu_name']} | elapsed={meta['elapsed_sec']}s | max_tele_err={meta['max_telescoping_err']}")
    print("Clean WHO:", repr(meta["clean_generations"]["who_are_you"]))
    print("Clean FRA:", repr(meta["clean_generations"]["capital_of_france"]))
    print("MLP scales:", meta["layer_mlp_scales"])

    # =========================================================================
    # 1. EXPERIMENT 2A: WRITE-GEOMETRY BENCHMARK
    # =========================================================================
    print("\n================ EXP 2A: WRITE GEOMETRY (WHO_CHAT_TF, middle dose=0.75 & peak dose=1.5) ================")
    rows2a = data["exp2a_write_geometry"]
    for site in ("post_norm", "pre_norm"):
        print(f"\n--- SITE: {site} (mean across 8 personas on WHO_CHAT_TF) ---")
        print(f"{'L':>3} {'dose':>4} | {'R_ridge dL(z,KL)':>20} | {'R_mf dL(z,KL)':>20} | {'LL dL(z,KL)':>20} | {'Rand dL(z,KL)':>20}")
        for L in (6, 12, 18, 24):
            for mult in (0.75, 1.5):
                parts = []
                for m in ("R_ridge", "R_mf", "logit_lens", "random"):
                    sub = [r["frames"]["WHO_CHAT_TF"] for r in rows2a if r["site"] == site and r["layer"] == L and r["dose_mult"] == mult and r["method"] == m]
                    dl = np.mean([x["dlogit"] for x in sub])
                    dz = np.mean([x["dlogit_z"] for x in sub])
                    kl = np.mean([x["kl_clean_int"] for x in sub])
                    parts.append(f"{dl:+5.2f}({dz:+5.1f},{kl:4.2f})")
                print(f"L{L:2d} {mult:4.2f} | {parts[0]:>20} | {parts[1]:>20} | {parts[2]:>20} | {parts[3]:>20}")

    print("\n--- EXP 2A REGISTER FRAMES (SELF vs NAME_BARE vs ROLE_BARE, post_norm, dose=1.5, R_ridge vs R_mf vs LL) ---")
    for L in (6, 12, 18, 24):
        print(f"Layer {L}:")
        for m in ("R_ridge", "R_mf", "logit_lens"):
            sub = [r for r in rows2a if r["site"] == "post_norm" and r["layer"] == L and r["dose_mult"] == 1.5 and r["method"] == m]
            vals = {fk: np.mean([r["frames"][fk]["dlogit"] for r in sub]) for fk in ("WHO_CHAT_TF", "SELF", "NAME_BARE", "ROLE_BARE")}
            top1_hits = sum(1 for r in sub if r["frames"]["WHO_CHAT_TF"]["rank_int"] == 1)
            print(f"  {m:10}: WHO={vals['WHO_CHAT_TF']:+6.2f} (top1 {top1_hits}/8) | SELF={vals['SELF']:+6.2f} | NAME={vals['NAME_BARE']:+6.2f} | ROLE={vals['ROLE_BARE']:+6.2f}")

    print("\n--- EXP 2A SAMPLE GREEDY GENERATIONS ON 'Who are you?' (dose=1.5) ---")
    gens2a = data["exp2a_generations"]
    for site in ("post_norm", "pre_norm"):
        hits_by_m = defaultdict(list)
        for g in gens2a:
            if g["site"] == site:
                w = g["target"].strip().lower()
                hit = w in g["generation"].lower()
                hits_by_m[(g["layer"], g["method"])].append(hit)
        print(f"Persona word appears in 'Who are you?' generation ({site}, out of 4 personas):")
        for L in (6, 12, 18, 24):
            line = f"  L{L:2d}: " + " | ".join(
                f"{m}={sum(hits_by_m[(L, m)])}/{len(hits_by_m[(L, m)])}" for m in ("R_ridge", "R_mf", "logit_lens", "random")
            )
            print(line)

    print("\nNotable Exp 2A generations (post_norm, L12 & L18, pirate/librarian/wizard/robot):")
    for g in gens2a:
        if g["site"] == "post_norm" and g["layer"] in (12, 18) and g["method"] in ("R_ridge", "R_mf", "logit_lens"):
            print(f"  L{g['layer']:2d} {g['method']:10} {g['target']:10} -> {g['generation'][:95]!r}")

    # =========================================================================
    # 2. EXPERIMENT 2B: HAND-BUILT VIRTUAL CONCEPT & SLOT SwiGLU NEURONS
    # =========================================================================
    print("\n================ EXP 2B: HAND-BUILT VIRTUAL SwiGLU NEURONS ================")
    rows2b = data["exp2b_virtual_neurons"]
    for site in ("post_norm", "pre_norm"):
        print(f"\n--- SITE: {site} (mean across 4 personas, R_ridge write) ---")
        print(f"{'L':>3} {'neuron_kind':25} | {'self_act':>8} {'off_act':>8} {'conc_act':>8} | {'self_dL':>7} {'conc_dL':>7} {'off_KL':>7} | {'sel_S/O':>7} {'sel_C/O':>7}")
        for L in (6, 12, 18, 24):
            for nk in ("virtual_concept_S_ridge", "virtual_concept_S_mf", "virtual_slot_S_ridge", "virtual_slot_contrastive"):
                sub = [r for r in rows2b if r["site"] == site and r["layer"] == L and r["neuron_kind"] == nk and r["method"] == "R_ridge"]
                if not sub:
                    continue
                print(
                    f"L{L:2d} {nk:25} | "
                    f"{np.mean([r['self_mean_act'] for r in sub]):+8.3f} "
                    f"{np.mean([r['off_mean_act'] for r in sub]):+8.3f} "
                    f"{np.mean([r['concept_act'] for r in sub]):+8.3f} | "
                    f"{np.mean([r['self_mean_dlogit'] for r in sub]):+7.2f} "
                    f"{np.mean([r['concept_dlogit'] for r in sub]):+7.2f} "
                    f"{np.mean([r['off_mean_kl'] for r in sub]):7.4f} | "
                    f"{np.median([r['selectivity_self_vs_off'] for r in sub]):7.1f} "
                    f"{np.median([r['selectivity_conc_vs_off'] for r in sub]):7.1f}"
                )

    print("\n--- EXP 2B: VIRTUAL SLOT CONTRASTIVE BY WRITE METHOD (post_norm, mean across 4 personas) ---")
    for L in (6, 12, 18, 24):
        print(f"Layer {L}:")
        for m in ("R_ridge", "R_mf", "logit_lens", "random"):
            sub = [r for r in rows2b if r["site"] == "post_norm" and r["layer"] == L and r["neuron_kind"] == "virtual_slot_contrastive" and r["method"] == m]
            sdL = np.mean([r["self_mean_dlogit"] for r in sub])
            sKL = np.mean([r["self_mean_kl"] for r in sub])
            oKL = np.mean([r["off_mean_kl"] for r in sub])
            who = np.mean([r["register_frames"]["WHO_CHAT_TF"]["dlogit"] for r in sub])
            slf = np.mean([r["register_frames"]["SELF"]["dlogit"] for r in sub])
            nam = np.mean([r["register_frames"]["NAME_BARE"]["dlogit"] for r in sub])
            rol = np.mean([r["register_frames"]["ROLE_BARE"]["dlogit"] for r in sub])
            print(f"  {m:10}: self_dL={sdL:+6.2f} (self_KL={sKL:.3f}, off_KL={oKL:.4f}) | WHO={who:+5.2f} SELF={slf:+5.2f} NAME={nam:+5.2f} ROLE={rol:+5.2f}")

    print("\n--- EXP 2B: SIDE-BY-SIDE GENERATIONS ('Who are you?' vs 'What is the capital of France?') ---")
    for g in data["exp2b_generations"]:
        if g["site"] == "post_norm" and g["method"] == "R_ridge" and g["neuron_kind"] in ("virtual_slot_contrastive", "virtual_slot_S_ridge", "virtual_concept_S_ridge"):
            w = g["target"].strip().lower()
            who_hit = "HIT" if w in g["gen_who_are_you"].lower() else "   "
            fra_ok = "OK" if "paris" in g["gen_capital_of_france"].lower() else "NO"
            print(f"  L{g['layer']:2d} {g['neuron_kind']:24} {g['target']:10} [{who_hit}|Fra:{fra_ok}]")
            print(f"      WHO: {g['gen_who_are_you'][:95]!r}")
            print(f"      FRA: {g['gen_capital_of_france'][:95]!r}")

    # =========================================================================
    # 3. EXPERIMENT 4: EXACT L+1..31 SIGNAL HIGHWAY & LATE SUPPRESSION CLIFF
    # =========================================================================
    print("\n================ EXP 4: EXACT L+1..31 SIGNAL HIGHWAY & LATE SUPPRESSION CLIFF ================")
    all_errs = []
    for rec in data["exp4_highway_natural"] + data["exp4_highway_virtual"]:
        for d in rec["frames"].values():
            all_errs.append(d["telescoping_err"])
    print(f"Telescoping err across {len(all_errs)} traces: median={np.median(all_errs):.5f}, p95={np.quantile(all_errs,0.95):.5f}, max={max(all_errs):.5f}")

    print("\n--- 4A: 12 NATURAL CONCEPT & SLOT NEURONS (WHO_CHAT_TF, push=+10.0) ---")
    print(f"{'Unit':9} {'Kind':15} {'Tgt':10} | {'dLogit':>7} {'Direct':>7} {'FrozDE':>7} {'ΣAttn':>7} {'ΣMLP':>7} {'RMSsh':>7} | {'Late3A':>7} {'Late3M':>7} {'CancR':>6}")
    for r in data["exp4_highway_natural"]:
        d = r["frames"]["WHO_CHAT_TF"]
        print(
            f"{r['unit']:9} {r['kind']:15} {r['target']:10} | "
            f"{d['exact_dlogit']:+7.2f} {d['direct_linear']:+7.2f} {d['frozen_direct_effect']:+7.2f} "
            f"{d['sum_downstream_attn']:+7.2f} {d['sum_downstream_mlp']:+7.2f} {d['rmsnorm_shift']:+7.2f} | "
            f"{d['late3_attn']:+7.2f} {d['late3_mlp']:+7.2f} {d['path_cancellation_ratio']:6.2f}"
        )

    print("\n--- 4B: VIRTUAL NEURONS SIGNAL HIGHWAY (WHO_CHAT_TF, mean across 4 personas) ---")
    for site in ("post_norm", "pre_norm"):
        print(f"\nSite: {site}")
        print(f"{'L':>3} {'Method':10} | {'dLogit':>7} {'Direct':>7} {'FrozDE':>7} {'ΣAttn':>7} {'AttnSl':>7} {'AttnFl':>7} {'ΣMLP':>7} {'RMSsh':>7} | {'Late3A':>7} {'Late3M':>7} {'CancR':>6}")
        for L in (6, 12, 18, 24):
            for m in ("R_ridge", "R_mf", "logit_lens"):
                sub = [r["frames"]["WHO_CHAT_TF"] for r in data["exp4_highway_virtual"] if r["site"] == site and r["layer"] == L and r["method"] == m]
                print(
                    f"L{L:2d} {m:10} | "
                    f"{np.mean([d['exact_dlogit'] for d in sub]):+7.2f} "
                    f"{np.mean([d['direct_linear'] for d in sub]):+7.2f} "
                    f"{np.mean([d['frozen_direct_effect'] for d in sub]):+7.2f} "
                    f"{np.mean([d['sum_downstream_attn'] for d in sub]):+7.2f} "
                    f"{np.mean([d['sum_downstream_attn_sliding'] for d in sub]):+7.2f} "
                    f"{np.mean([d['sum_downstream_attn_full'] for d in sub]):+7.2f} "
                    f"{np.mean([d['sum_downstream_mlp'] for d in sub]):+7.2f} "
                    f"{np.mean([d['rmsnorm_shift'] for d in sub]):+7.2f} | "
                    f"{np.mean([d['late3_attn'] for d in sub]):+7.2f} "
                    f"{np.mean([d['late3_mlp'] for d in sub]):+7.2f} "
                    f"{np.mean([d['path_cancellation_ratio'] for d in sub]):6.2f}"
                )

    # Layer-by-layer trace for L6 post_norm R_ridge vs R_mf vs logit_lens to inspect 4-layer periodicity & L29-31 cliff!
    print("\n--- LAYER-BY-LAYER L7..L31 PROFILE FOR L6 INJECTION (post_norm, WHO_CHAT_TF, mean across 4 personas) ---")
    print(f"{'L':>3} {'Type':4} | {'R_rg Attn':>9} {'R_rg MLP':>9} | {'R_mf Attn':>9} {'R_mf MLP':>9} | {'LL Attn':>9} {'LL MLP':>9}")
    by_m = {}
    for m in ("R_ridge", "R_mf", "logit_lens"):
        by_m[m] = [r["frames"]["WHO_CHAT_TF"]["per_layer"] for r in data["exp4_highway_virtual"] if r["site"] == "post_norm" and r["layer"] == 6 and r["method"] == m]
    for idx in range(len(by_m["R_ridge"][0])):
        lno = by_m["R_ridge"][0][idx]["layer"]
        lt = "FULL" if by_m["R_ridge"][0][idx]["layer_type"] == "full_attention" else "slid"
        vals = []
        for m in ("R_ridge", "R_mf", "logit_lens"):
            a = np.mean([tr[idx]["attn"] for tr in by_m[m]])
            mp = np.mean([tr[idx]["mlp"] for tr in by_m[m]])
            vals.extend([a, mp])
        print(f"L{lno:2d} {lt:4} | {vals[0]:+9.3f} {vals[1]:+9.3f} | {vals[2]:+9.3f} {vals[3]:+9.3f} | {vals[4]:+9.3f} {vals[5]:+9.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
