#!/usr/bin/env python3
"""Pre-compute the OLMo 3 7B virtual neuron weight bank & natural neuron manifest on CPU (0 CU).

Uses our fitted 64-doc RelP R-lens (R_out) and S-lens (S_mid) plus the locally cached
`allenai/Olmo-3-7B-Instruct@6e5971d9...` weights to synthesize compact (~10 MB) closed-form,
zero-training SwiGLU virtual neuron vectors `(w_gate, w_up, w_down)` for Idea #2 and natural
neuron metadata for Idea #4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import INJECTION_LAYERS, MODEL_ID, MODEL_REV, TARGET_PERSONAS

NATURAL_NEURONS = [
    # 6 Natural Concept Neurons (monosemantic read_S == write_R from zero-GPU census)
    {"unit": "L4N6633",  "layer": 4,  "neuron": 6633, "kind": "natural_concept", "target": " teacher"},
    {"unit": "L5N6875",  "layer": 5,  "neuron": 6875, "kind": "natural_concept", "target": " librarian"},
    {"unit": "L14N8245", "layer": 14, "neuron": 8245, "kind": "natural_concept", "target": " cat"},
    {"unit": "L16N5264", "layer": 16, "neuron": 5264, "kind": "natural_concept", "target": " dog"},
    {"unit": "L19N4502", "layer": 19, "neuron": 4502, "kind": "natural_concept", "target": " robot"},
    {"unit": "L23N8559", "layer": 23, "neuron": 8559, "kind": "natural_concept", "target": " knight"},
    # 6 Natural Slot-like Neurons (high write_R, cross-domain / turn-opening / pronoun read_S)
    {"unit": "L17N2444", "layer": 17, "neuron": 2444, "kind": "natural_slot",    "target": " me"},
    {"unit": "L21N3",    "layer": 21, "neuron": 3,    "kind": "natural_slot",    "target": " me"},
    {"unit": "L23N6941", "layer": 23, "neuron": 6941, "kind": "natural_slot",    "target": " AI"},
    {"unit": "L25N4074", "layer": 25, "neuron": 4074, "kind": "natural_slot",    "target": " I"},
    {"unit": "L26N3854", "layer": 26, "neuron": 3854, "kind": "natural_slot",    "target": " assistant"},
    {"unit": "L28N2566", "layer": 28, "neuron": 2566, "kind": "natural_slot",    "target": " I"},
]

SELF_QUERY_TOKENS = [
    "Who", " who", " are", " you", "?", " yourself", " identity", " assistant", " I", " me",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def concept_variant_ids(tok, word: str) -> list[int]:
    clean = word.strip()
    cands = [word, clean, " " + clean.capitalize(), clean.capitalize(), word + "s", clean + "s"]
    ids = []
    for c in cands:
        enc = tok.encode(c, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-dir", type=Path, default=Path("/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens"))
    ap.add_argument("--out", type=Path, default=Path("/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens/olmo3_virtual_bank.pt"))
    ap.add_argument("--ridge-alpha", type=float, default=0.5, help="Ridge multiplier * mean(diag(G))")
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    r_path = args.lens_dir / "r_lens.pt"
    s_path = args.lens_dir / "s_lens.pt"
    print(f"Loading lenses from {args.lens_dir} ...")
    r_obj = torch.load(r_path, map_location="cpu", weights_only=False)
    s_obj = torch.load(s_path, map_location="cpu", weights_only=False)
    R_out = r_obj["means"]["out"]  # [31, d, d]
    S_mid = s_obj["means"]["mid"]  # [32, d, d]

    print(f"Loading {MODEL_ID}@{MODEL_REV[:8]} from local cache ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16, local_files_only=True
    ).eval()

    Wu = model.lm_head.weight.detach().float() * model.model.norm.weight.detach().float()[None, :]
    We = model.model.embed_tokens.weight.detach().float()
    d = Wu.shape[1]
    eye_d = torch.eye(d, dtype=torch.float32)

    # Background embedding mean for centering S_mid read targets
    bg_idx = torch.arange(0, We.shape[0], 25)
    e_bg_mean = We[bg_idx].mean(dim=0)

    # Target token verification
    target_ids: dict[str, int] = {}
    variant_map: dict[str, list[int]] = {}
    for w in TARGET_PERSONAS:
        enc = tok.encode(w, add_special_tokens=False)
        assert len(enc) == 1, f"Target persona {w!r} is not single-token: {enc}"
        target_ids[w] = enc[0]
        variant_map[w] = concept_variant_ids(tok, w)

    self_q_ids = []
    for w in SELF_QUERY_TOKENS:
        enc = tok.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            self_q_ids.append(enc[0])
    self_q_ids = sorted(set(self_q_ids))
    e_slot_q = We[self_q_ids].mean(dim=0) - e_bg_mean

    # Natural neuron target token verification
    natural_manifest = []
    for rec in NATURAL_NEURONS:
        enc = tok.encode(rec["target"], add_special_tokens=False)
        assert len(enc) == 1, f"Natural neuron target {rec['target']!r} not single-token"
        lay = model.model.layers[rec["layer"]]
        g_pf = lay.post_feedforward_layernorm.weight.detach().float()
        wd_col = lay.mlp.down_proj.weight.detach().float()[:, rec["neuron"]]
        natural_manifest.append({
            **rec,
            "target_id": enc[0],
            "w_down_norm": round(float(wd_col.norm()), 4),
            "w_eff_norm": round(float((g_pf * wd_col).norm()), 4),
        })

    g_rng = torch.Generator(device="cpu").manual_seed(args.seed)

    layers_bank: dict[int, dict] = {}
    for L in INJECTION_LAYERS:
        print(f"Synthesizing zero-training virtual neuron vectors at Layer {L} ...")
        lay = model.model.layers[L]
        g_pf = lay.post_feedforward_layernorm.weight.detach().float()  # [d]
        Wd = lay.mlp.down_proj.weight.detach().float()  # [d, I]
        Wg = lay.mlp.gate_proj.weight.detach().float()  # [I, d]
        Wu_in = lay.mlp.up_proj.weight.detach().float() # [I, d]

        nat_wdown_med = float(Wd.norm(dim=0).median())
        nat_weff_med = float((g_pf[:, None] * Wd).norm(dim=0).median())
        nat_wgate_med = float(Wg.norm(dim=1).median())
        nat_wup_med = float(Wu_in.norm(dim=1).median())

        # Solve ridge Gram for R_out[L] and S_mid[L]
        RL = R_out[L].float()
        GR = RL.T @ RL
        lam_r = float(torch.trace(GR) / d) * args.ridge_alpha
        chol_R = torch.linalg.cholesky(GR + lam_r * eye_d)

        SL = S_mid[L].float()
        GS = SL @ SL.T
        lam_s = float(torch.trace(GS) / d) * args.ridge_alpha
        chol_S = torch.linalg.cholesky(GS + lam_s * eye_d)

        # Pure S_mid slot self-query read directions
        s_slot_mf = SL @ e_slot_q
        s_slot_rg = torch.cholesky_solve(s_slot_mf[:, None], chol_S).squeeze(-1)

        writes = {}
        reads_concept_rg = {}
        reads_concept_mf = {}
        for w in TARGET_PERSONAS:
            tid = target_ids[w]
            u_t = Wu[tid]  # [d]
            w_mf = RL.T @ u_t
            w_rg = torch.cholesky_solve(w_mf[:, None], chol_R).squeeze(-1)
            w_ll = u_t.clone()
            w_rnd = torch.randn(d, generator=g_rng, dtype=torch.float32)

            writes[w] = {
                "R_ridge": unit_norm(w_rg).half(),
                "R_mf": unit_norm(w_mf).half(),
                "logit_lens": unit_norm(w_ll).half(),
                "random": unit_norm(w_rnd).half(),
            }

            e_c = We[variant_map[w]].mean(dim=0) - e_bg_mean
            rc_mf = SL @ e_c
            rc_rg = torch.cholesky_solve(rc_mf[:, None], chol_S).squeeze(-1)
            reads_concept_mf[w] = unit_norm(rc_mf).half()
            reads_concept_rg[w] = unit_norm(rc_rg).half()

        layers_bank[L] = {
            "layer": L,
            "layer_type": str(model.config.layer_types[L]),
            "gamma_post_ff": g_pf.half(),
            "nat_wdown_med": round(nat_wdown_med, 5),
            "nat_weff_med": round(nat_weff_med, 5),
            "nat_wgate_med": round(nat_wgate_med, 5),
            "nat_wup_med": round(nat_wup_med, 5),
            "writes": writes,
            "reads": {
                "concept_S_ridge": reads_concept_rg,
                "concept_S_mf": reads_concept_mf,
                "slot_S_ridge": unit_norm(s_slot_rg).half(),
                "slot_S_mf": unit_norm(s_slot_mf).half(),
            },
        }

    bundle = {
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "ridge_alpha": args.ridge_alpha,
        "seed": args.seed,
        "lens_provenance": {
            "r_lens_sha256": sha256_file(r_path),
            "s_lens_sha256": sha256_file(s_path),
        },
        "target_personas": TARGET_PERSONAS,
        "target_ids": target_ids,
        "variant_ids": variant_map,
        "self_query_ids": self_q_ids,
        "injection_layers": INJECTION_LAYERS,
        "layers": layers_bank,
        "natural_neurons": natural_manifest,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Saved virtual neuron bank to {args.out} ({mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
