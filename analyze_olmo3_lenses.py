#!/usr/bin/env python3
"""Zero-GPU post-fit validation & Idea #5 scan for the OLMo 3 7B Neuron Lens.

1. Cross-validates our fitted 64-doc RelP R-lens (``R_out``) against:
   - ``danorel/Olmo-3-7B-Instruct-jpp-lens`` (64-doc ``lrp_mode='rlens'``, layers 0..30),
   - ``mhough/olmo3-jacobian-lenses`` (616-doc true J-lens, layers 0, 3, ..., 30).
2. Prints exact fp32 split-half convergence across all 32 layers (R_out, R_mid, S_mid, S_out).
3. If ``--with-weights`` is passed (using local HF cache of ``allenai/Olmo-3-7B-Instruct``),
   runs **Idea #5**: 1,024-head OV scan comparing ``sliding_attention`` (24 layers) vs.
   ``full_attention`` (8 layers: L3, L7, L11, L15, L19, L23, L27, L31) on self/name/role
   and persona-copying OV scores under our fitted R-lens and S-lens!
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"


def load_our_means(lens_dir: Path) -> tuple[dict, dict, dict]:
    r_obj = torch.load(lens_dir / "r_lens.pt", map_location="cpu", weights_only=False)
    s_obj = torch.load(lens_dir / "s_lens.pt", map_location="cpu", weights_only=False)
    meta = json.loads((lens_dir / "fit_meta.json").read_text())
    r_means = r_obj["means"] if "means" in r_obj else {
        t: (r_obj["state"]["sums"][(t, 0)].float() + r_obj["state"]["sums"][(t, 1)].float()) / sum(r_obj["state"]["n_docs"])
        for t in ("out", "mid")
    }
    s_means = s_obj["means"] if "means" in s_obj else {
        t: (s_obj["state"]["sums"][(t, 0)].float() + s_obj["state"]["sums"][(t, 1)].float()) / sum(s_obj["state"]["n_docs"])
        for t in ("out", "mid")
    }
    return r_means, s_means, meta


def cross_validate(r_means: dict, meta: dict) -> dict:
    p_dan = hf_hub_download(
        "danorel/Olmo-3-7B-Instruct-jpp-lens",
        "Olmo-3-7B-Instruct_jpp_lens.pt",
        revision="fd8f1d4e89e32081c722442d784bda70bf15170e",
        local_files_only=True,
    )
    p_mh = hf_hub_download(
        "mhough/olmo3-jacobian-lenses",
        "lenses/olmo-3-7b-instruct.pt",
        revision="c134cb08886c475947c865e8790fa75495b0380a",
        local_files_only=True,
    )
    J_dan = torch.load(p_dan, map_location="cpu", weights_only=False)["J"]
    J_mh = torch.load(p_mh, map_location="cpu", weights_only=False)["J"]

    R_out = r_means["out"]  # [31, 4096, 4096]
    dan_rows = []
    mh_rows = []
    print("\n=== CROSS-VALIDATION: OUR 64-DOC R_out vs DANOREL (rlens) & MHOUGH (J-lens) ===")
    print(f"{'L':>3}  {'||Ours||':>9}  {'||Dan||':>9}  {'cos(Dan)':>9}  {'cos(Mh)':>9}  {'SH_Rout':>8}  {'SH_Sout':>8}")
    sh_r = meta["split_half_fp32"]["R"]["out"]
    sh_s = meta["split_half_fp32"]["S"]["out"]
    for L in range(R_out.shape[0]):
        a = R_out[L].float()
        b = J_dan[L].float()
        cos_d = float((a * b).sum() / (a.norm() * b.norm()))
        dan_rows.append({"layer": L, "norm_ours": round(float(a.norm()), 3), "norm_dan": round(float(b.norm()), 3), "cos_dan": round(cos_d, 4)})
        cos_m_str = "      —  "
        if L in J_mh:
            c = J_mh[L].float()
            cos_m = float((a * c).sum() / (a.norm() * c.norm()))
            cos_m_str = f"{cos_m:+9.4f}"
            mh_rows.append({"layer": L, "cos_mh": round(cos_m, 4)})
        print(f"L{L:2d}  {a.norm():9.2f}  {b.norm():9.2f}  {cos_d:+9.4f}  {cos_m_str}  {sh_r[L]:8.4f}  {sh_s[L]:8.4f}")
    return {"vs_danorel": dan_rows, "vs_mhough": mh_rows}


def scan_1024_heads(r_means: dict, s_means: dict, local_only: bool = False) -> dict:
    """Idea #5: 1,024-head OV scan comparing sliding_attention vs full_attention layers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16, local_files_only=local_only
    ).eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    hd = cfg.hidden_size // n_heads
    layer_types = list(cfg.layer_types)

    Wu = model.lm_head.weight.detach().float() * model.model.norm.weight.detach().float()[None, :]
    We = model.model.embed_tokens.weight.detach().float()

    # Token groups for OV diagonal self-copying vs self/name/role tokens
    def tok_ids(words: list[str]) -> list[int]:
        out = []
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))

    self_ids = tok_ids([" I", " me", " my", " myself", " I'm", " mine", "I", "My"])
    role_ids = tok_ids([" he", " she", " they", " him", " her", " his", " their", " character", " person", " man", " woman"])
    name_ids = tok_ids([" Alex", " Sam", " Jordan", " Alice", " Bob", " John", " Sarah", " David", " Emma", " Maya"])
    persona_ids = tok_ids([" dog", " fox", " cat", " tree", " town", " skeleton", " pirate", " robot", " wizard", " doctor", " teacher", " lawyer", " owl", " wolf", " bear", " king", " queen", " knight", " chef", " ghost"])
    sample_vocab = torch.arange(0, Wu.shape[0], 20)  # ~5,014 evenly spaced vocab tokens for baseline z-score

    E_self = We[self_ids]
    U_self = Wu[self_ids]
    E_role = We[role_ids]
    U_role = Wu[role_ids]
    E_name = We[name_ids]
    U_name = Wu[name_ids]
    E_pers = We[persona_ids]
    U_pers = Wu[persona_ids]
    E_bg = We[sample_vocab]
    U_bg = Wu[sample_vocab]

    head_rows = []
    R_mid = r_means["mid"]  # [32, d, d] (attention write enters mid residual of layer L!)
    S_out = s_means["out"]  # [32, d, d] (layer L attention reads layer L-1 out, or embeddings at L=0!)
    eye_d = torch.eye(cfg.hidden_size)

    for L in range(n_layers):
        lay = model.model.layers[L]
        at = lay.self_attn
        Wv = at.v_proj.weight.detach().float()  # [H*hd, d]
        Wo = at.o_proj.weight.detach().float()  # [d, H*hd]
        g_pa = lay.post_attention_layernorm.weight.detach().float()
        Wo_eff = g_pa[:, None] * Wo  # [d, H*hd]

        # Effective read map K_in (embedding -> input of layer L) and write map J_out (post-attn residual L -> final)
        K_in = eye_d if L == 0 else S_out[L - 1].float()
        J_out = R_mid[L].float()

        # Pre-project token sets through K_in and J_out^T (fast!)
        # For head h: token score(s -> s) = (U_s @ J_out @ Wo_h) . (Wv_h @ K_in @ E_s)
        UK_self_out = U_self @ J_out   # [ns, d]
        EK_self_in = E_self @ K_in.T   # [ns, d]
        UK_role_out = U_role @ J_out
        EK_role_in = E_role @ K_in.T
        UK_name_out = U_name @ J_out
        EK_name_in = E_name @ K_in.T
        UK_pers_out = U_pers @ J_out
        EK_pers_in = E_pers @ K_in.T
        UK_bg_out = U_bg @ J_out
        EK_bg_in = E_bg @ K_in.T

        for h in range(n_heads):
            sl = slice(h * hd, (h + 1) * hd)
            Wvh = Wv[sl]        # [hd, d]
            Woh = Wo_eff[:, sl] # [d, hd]

            def diag_scores(U_proj, E_proj):
                return ((U_proj @ Woh) * (E_proj @ Wvh.T)).sum(-1)

            bg = diag_scores(UK_bg_out, EK_bg_in)
            mu, sd = float(bg.mean()), float(bg.std().clamp_min(1e-8))
            sc_self = float((diag_scores(UK_self_out, EK_self_in).mean() - mu) / sd)
            sc_role = float((diag_scores(UK_role_out, EK_role_in).mean() - mu) / sd)
            sc_name = float((diag_scores(UK_name_out, EK_name_in).mean() - mu) / sd)
            sc_pers = float((diag_scores(UK_pers_out, EK_pers_in).mean() - mu) / sd)
            # Raw diagonal copy signal (normalized by Frobenius norm of OV_h)
            ov_norm = float((Woh @ Wvh).norm())
            copy_bg_mean = float(bg.mean()) / max(ov_norm, 1e-8)

            head_rows.append({
                "head": f"L{L}H{h}",
                "layer": L,
                "h": h,
                "layer_type": layer_types[L],
                "ov_frob": round(ov_norm, 3),
                "copy_bg_normed": round(copy_bg_mean, 4),
                "self_z": round(sc_self, 3),
                "role_z": round(sc_role, 3),
                "name_z": round(sc_name, 3),
                "persona_z": round(sc_pers, 3),
                "self_minus_role_z": round(sc_self - sc_role, 3),
                "abs_sr_z": round(abs(sc_self - sc_role), 3),
            })

    # Summarize sliding_attention vs full_attention (Idea #5!)
    sl_rows = [r for r in head_rows if r["layer_type"] == "sliding_attention"]
    fl_rows = [r for r in head_rows if r["layer_type"] == "full_attention"]
    print("\n=== IDEA #5: SLIDING (768 heads) vs FULL-ATTENTION (256 heads) SUMMARY ===")
    for metric in ["self_z", "role_z", "name_z", "persona_z", "self_minus_role_z", "abs_sr_z", "ov_frob"]:
        ms = np.mean([r[metric] for r in sl_rows])
        mf = np.mean([r[metric] for r in fl_rows])
        qs = np.quantile([r[metric] for r in sl_rows], 0.95)
        qf = np.quantile([r[metric] for r in fl_rows], 0.95)
        print(f"  {metric:18}: sliding mean={ms:+.3f} (p95={qs:+.3f}) | full mean={mf:+.3f} (p95={qf:+.3f})")

    top_sr = sorted(head_rows, key=lambda r: -r["self_minus_role_z"])[:20]
    bot_sr = sorted(head_rows, key=lambda r: r["self_minus_role_z"])[:20]
    top_pers = sorted(head_rows, key=lambda r: -r["persona_z"])[:20]
    print("\n=== TOP 15 SELF-MINUS-ROLE HEADS (OLMo 3 7B Instruct) ===")
    for r in top_sr[:15]:
        print(f"  {r['head']:7} [{r['layer_type'][:4]}] S-R={r['self_minus_role_z']:+7.2f} (S={r['self_z']:+6.2f}, R={r['role_z']:+6.2f}, N={r['name_z']:+6.2f}, P={r['persona_z']:+6.2f})")
    print("\n=== TOP 15 ROLE-MINUS-SELF HEADS (OLMo 3 7B Instruct) ===")
    for r in bot_sr[:15]:
        print(f"  {r['head']:7} [{r['layer_type'][:4]}] S-R={r['self_minus_role_z']:+7.2f} (S={r['self_z']:+6.2f}, R={r['role_z']:+6.2f}, N={r['name_z']:+6.2f}, P={r['persona_z']:+6.2f})")

    return {
        "n_heads": len(head_rows),
        "n_sliding": len(sl_rows),
        "n_full": len(fl_rows),
        "top_self_minus_role": top_sr,
        "top_role_minus_self": bot_sr,
        "top_persona_copy": top_pers,
        "all_1024_heads": head_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-dir", type=Path, default=Path("/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens"))
    ap.add_argument("--with-weights", action="store_true", help="Also run 1,024-head Idea #5 OV scan (needs Olmo-3-7B-Instruct weights)")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    r_means, s_means, meta = load_our_means(args.lens_dir)
    out = {"cross_validation": cross_validate(r_means, meta)}
    if args.with_weights:
        out["head_scan_1024"] = scan_1024_heads(r_means, s_means, local_only=args.local_only)
    out_path = args.out or (args.lens_dir / "olmo3_lens_analysis.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved analysis to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
