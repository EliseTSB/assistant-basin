#!/usr/bin/env python3
"""build_olmo3_c2_constants.py — Deterministic 0-CU pre-registration & vector-bank builder for OLMo 3 7B Stage C2.

Computes and freezes:
  1. `olmo3_virtual_and_highway/c2_prereg_constants.json`
  2. `work/olmo3_c2/c2_vector_bank.safetensors` (deterministic float32 canonical tensors;
     SHA-256 recorded inside `c2_prereg_constants.json`).

Inputs (repo-relative defaults; override via CLI flags, zero hardcoded user paths):
  - Frozen OLMo-3-7B-Instruct snapshot (`6e5971d9...`): tokenizer, `lm_head.weight`, `model.norm.weight`,
    and per-layer `post_feedforward_layernorm.weight` / `post_attention_layernorm.weight`
  - Frozen 64-doc RelP R-lens `r_lens.pt` (`sha256 = ceef21fc...`)
  - Published Stage C0 (`c0_prereg_constants.json`), Stage C0b (`c0b_prereg_constants.json`),
    and Stage C1 (`c1_prereg_constants.json`) constants

Verifies & records (at 0 CU):
  1. 24/24 fresh mixed-POS single-token targets (0 overlap with C0 targets, C1 targets, or persona words).
  2. 8/8 fresh all-metalinguistic `':'`-ending prompts (4 Bare + 4 ChatML, 0 overlap with C0/C1)
     PLUS 16 paired C0 `'" a"' -> '':'` anchor-norm prediction prompts (`L24 ~ 7.2, L27 ~ 7.5`).
  3. Exact L18/L24/L27 frozen-lens comparison across Raw, Mean-Centered (`u_t - mu_U`), and
     Park-Whitened (`Shrink-0.1^-0.5 (u_t - mu_U)`) `R_ridge` writes on the fresh 24-word bank.
  4. Lacey low-`gamma` mask capacities & cosine retentions (`drop_k_ff=4`, `drop_k_attn=16`) across `L18/21/24/27`.
  5. Exact forward-count budget across C2.0..C2.4 (`B=1`, `bf16`, `use_cache=False`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_R_LENS = REPO_ROOT / "work" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt"
FALLBACK_R_LENS = Path.home() / "green" / "Journal" / "notes" / "reports" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt"
C0_CONST_PATH = ROOT / "c0_prereg_constants.json"
C0B_CONST_PATH = ROOT / "c0b_prereg_constants.json"
C1_CONST_PATH = ROOT / "c1_prereg_constants.json"
DEFAULT_OUT_CONSTANTS = ROOT / "c2_prereg_constants.json"
DEFAULT_OUT_BANK = REPO_ROOT / "work" / "olmo3_c2" / "c2_vector_bank.safetensors"

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

PRIMARY_C21_LAYERS = [18, 24, 27]
SWEEP_LAYERS = [12, 15, 18, 21, 24, 27, 30]
PRENORM_LAYERS = [18, 21, 24, 27]

# Deterministic RNG seeds (C2 family)
PERM_SEED = 2026112501
BCA_SEED = 2026112502
BANK_DM_SEED_BASE = 2026112503
BANK_RAND_SEED_BASE = 2026112504
BANK_PRENORM_RAND_SEED_BASE = 2026112505

K_DM_DRAWS = 2
M_RAND_ISO_DRAWS = 32
K_PRENORM_RAND_DRAWS = 2
GAMMA_DROP_FF = 4
GAMMA_DROP_ATTN = 16
COV_SHRINK_ALPHA = 0.1

# Fresh 24-target mixed-POS bank (0 overlap with C0 targets_16, C1 targets_24, or personas)
TARGETS_24_C2_SPEC: list[tuple[str, str]] = [
    ("nouns", " candle"),
    ("nouns", " barrel"),
    ("nouns", " feather"),
    ("nouns", " ribbon"),
    ("nouns", " basket"),
    ("nouns", " glacier"),
    ("verbs", " polish"),
    ("verbs", " stumble"),
    ("verbs", " vanish"),
    ("verbs", " mend"),
    ("verbs", " drift"),
    ("verbs", " carve"),
    ("adjectives", " sturdy"),
    ("adjectives", " velvet"),
    ("adjectives", " crisp"),
    ("adjectives", " faint"),
    ("adjectives", " vivid"),
    ("adjectives", " shallow"),
    ("function", " however"),
    ("function", " whereas"),
    ("function", " beneath"),
    ("function", " despite"),
    ("function", " instead"),
    ("function", " beyond"),
]

PERSONA_WORDS = {" pirate", " librarian", " wizard", " robot", " owl", " fox", " whale", " baker"}

C2_BARE_STEMS: list[tuple[str, str, str]] = [
    ("narr_0", "narrative", "Inside the small drawer, the card displayed a single word:"),
    ("narr_1", "narrative", "Across the chalk slate, the final word written in dust was:"),
    ("anal_0", "analytical", "In the test ledger, the target entry listed in row four was:"),
    ("anal_1", "analytical", "From the sample index, the marked label shown at the end was:"),
]


def write_canonical_safetensors(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    """Write .safetensors with sorted keys and deterministic JSON header."""
    header: dict[str, Any] = {"__metadata__": {k: metadata[k] for k in sorted(metadata)}}
    chunks: list[bytes] = []
    offset = 0
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        assert t.dtype == torch.float32, f"Expected float32 tensor for {k}, got {t.dtype}"
        raw = t.numpy().astype("<f4", copy=False).tobytes()
        header[k] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    hb = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(c)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_r_lens(cli_path: Path | None) -> Path:
    if cli_path is not None and cli_path.is_file():
        return cli_path
    if DEFAULT_R_LENS.is_file():
        return DEFAULT_R_LENS
    if FALLBACK_R_LENS.is_file():
        return FALLBACK_R_LENS
    raise FileNotFoundError(f"Could not locate r_lens.pt at {DEFAULT_R_LENS} or {FALLBACK_R_LENS}")


def resolve_hf_snapshot(model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    from huggingface_hub.constants import HF_HUB_CACHE
    cand = Path(HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REV
    if cand.is_dir():
        return cand
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, revision=MODEL_REV, local_files_only=True))


def build_c2_prompts() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for sid, style, stem in C2_BARE_STEMS:
        prompts.append({"id": f"c2_bare_{sid}", "format": "bare", "style": style, "stem": stem, "text": stem})
    for sid, style, stem in C2_BARE_STEMS:
        chat = (
            f"<|im_start|>user\nContinue this line with one word: {stem}<|im_end|>\n"
            f"<|im_start|>assistant\n{stem}"
        )
        prompts.append({"id": f"c2_chat_{sid}", "format": "chatml", "style": style, "stem": stem, "text": chat})
    return prompts


def build_c0_anchor_colon_prompts(c0b_prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair C0's exact 8 ' a' prompts with 8 ':'-ending variants to test Claude #103(1c) norm-drop prediction."""
    out: list[dict[str, Any]] = []
    for p in c0b_prompts:
        pid = p["id"]
        fmt = p["format"]
        orig = p["text"]
        assert orig.endswith(" a"), f"Expected C0 prompt to end with ' a': {orig!r}"
        out.append({"id": f"{pid}_orig_a", "pair_id": pid, "format": fmt, "ending": "a", "text": orig})
        colon_text = orig[:-2] + ":"
        if fmt == "chatml":
            colon_text = colon_text.replace(" a...<|im_end|>", ":<|im_end|>")
        out.append({"id": f"{pid}_colon", "pair_id": pid, "format": fmt, "ending": ":", "text": colon_text})
    return out


def orthonormal_basis(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for v in vecs:
        w = v.clone().double()
        for _ in range(2):
            for b in basis:
                w = w - torch.dot(w, b) * b
        nrm = float(w.norm().item())
        if nrm > 1e-10:
            basis.append(w / nrm)
    return basis


def make_direct_matched_draws(
    u_hat: torch.Tensor,
    w_rmf: torch.Tensor,
    w_rridge: torch.Tensor,
    c: float,
    k_draws: int,
    seed_base: int,
) -> torch.Tensor:
    basis = orthonormal_basis([u_hat, w_rmf, w_rridge])
    d = u_hat.shape[0]
    s = math.sqrt(max(1.0 - c * c, 0.0))
    u_hat_d = u_hat.double()
    draws = []
    for k in range(k_draws):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + k)
        z = torch.randn(d, generator=g, dtype=torch.float64)
        for _ in range(2):
            for b in basis:
                z = z - torch.dot(z, b) * b
        z_perp = z / z.norm().clamp_min(1e-12)
        w = c * u_hat_d + s * z_perp
        w = w / w.norm().clamp_min(1e-12)
        assert abs(float(w.norm().item()) - 1.0) < 1e-6
        assert abs(float(torch.dot(w, u_hat_d).item()) - c) < 1e-5
        draws.append(w.float())
    return torch.stack(draws, dim=0)


def make_rand_iso_draws(d: int, m_draws: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(m_draws, d, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v.float()


def make_masked_rand_draws(mask_01: torch.Tensor, n_targets: int, k_draws: int, seed_base: int) -> torch.Tensor:
    """Generate [n_targets, k_draws, d] random unit vectors zeroed on the same masked low-gamma dims."""
    d = mask_01.shape[0]
    mask_d = mask_01.double()
    rows = []
    for ti in range(n_targets):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + ti * 17)
        v = torch.randn(k_draws, d, generator=g, dtype=torch.float64) * mask_d.unsqueeze(0)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rows.append(v.float())
    return torch.stack(rows, dim=0)


def load_safetensors_selected(snap: Path, wanted: set[str]) -> dict[str, torch.Tensor]:
    idx_path = snap / "model.safetensors.index.json"
    out: dict[str, torch.Tensor] = {}
    if idx_path.is_file():
        wm = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
        by_file: dict[str, list[str]] = {}
        for k in wanted:
            if k in wm:
                by_file.setdefault(wm[k], []).append(k)
        for fname in sorted(by_file):
            with safe_open(str(snap / fname), framework="pt", device="cpu") as f:
                for k in by_file[fname]:
                    out[k] = f.get_tensor(k).float()
    else:
        for st in sorted(snap.glob("*.safetensors")):
            with safe_open(str(st), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in wanted:
                        out[k] = f.get_tensor(k).float()
    missing = wanted - set(out.keys())
    assert not missing, f"Missing model tensors: {sorted(missing)}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic Stage C2 pre-registration constants and vector bank (0 CU).")
    ap.add_argument("--r-lens", type=Path, default=None, help="Path to frozen r_lens.pt")
    ap.add_argument("--model-dir", type=str, default=None, help="Optional local HF snapshot directory for OLMo-3-7B-Instruct")
    ap.add_argument("--out-constants", type=Path, default=DEFAULT_OUT_CONSTANTS, help="Output path for c2_prereg_constants.json")
    ap.add_argument("--out-bank", type=Path, default=DEFAULT_OUT_BANK, help="Output path for c2_vector_bank.safetensors")
    args = ap.parse_args()

    r_lens_path = resolve_r_lens(args.r_lens)
    r_sha = sha256_file(r_lens_path)
    assert r_sha == EXPECTED_R_LENS_SHA256, f"r_lens sha mismatch: {r_sha}"

    c0 = json.loads(C0_CONST_PATH.read_text(encoding="utf-8"))
    c0b = json.loads(C0B_CONST_PATH.read_text(encoding="utf-8"))
    c1 = json.loads(C1_CONST_PATH.read_text(encoding="utf-8"))

    forbidden_words = {t["string"] for t in c0["targets_16"]} | {t["string"] for t in c1["targets_24"]} | PERSONA_WORDS
    forbidden_prompts = {p["text"] for p in c0b["prompts_8"]} | {p["text"] for p in c1["prompts_8"]}

    snap = resolve_hf_snapshot(args.model_dir)
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)

    # 1. Verify 24 fresh mixed-POS C2 targets (0 overlap with C0, C1, or personas)
    targets: list[dict[str, Any]] = []
    for stratum, word in TARGETS_24_C2_SPEC:
        assert word not in forbidden_words, f"Overlap with C0/C1/persona target: {word}"
        ids = tok.encode(word, add_special_tokens=False)
        dec = tok.decode(ids)
        assert len(ids) == 1 and dec == word, f"Not single roundtrip token: {word!r} -> {ids} ({dec!r})"
        targets.append({"stratum": stratum, "string": word, "id": int(ids[0]), "decoded_piece": dec})
    assert len({t["id"] for t in targets}) == 24
    targets_spec_sha = hashlib.sha256(
        json.dumps(TARGETS_24_C2_SPEC, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    # 2. Verify 8 fresh C2 metalinguistic prompts + 16 C0 ':' anchor prediction prompts
    prompts = build_c2_prompts()
    for p in prompts:
        assert p["text"] not in forbidden_prompts, f"Overlap with C0/C1 prompt: {p['id']}"
        ids = tok.encode(p["text"], add_special_tokens=False)
        p["n_tokens"] = len(ids)
        p["last_token_id"] = int(ids[-1])
        p["last_token_piece"] = tok.decode([ids[-1]])
        assert p["last_token_piece"] == ":", f"Expected ':' ending for {p['id']}, got {p['last_token_piece']!r}"
    bare_T = [p["n_tokens"] for p in prompts if p["format"] == "bare"]
    chat_T = [p["n_tokens"] for p in prompts if p["format"] == "chatml"]

    c0_colon_prompts = build_c0_anchor_colon_prompts(c0b["prompts_8"])
    for p in c0_colon_prompts:
        ids = tok.encode(p["text"], add_special_tokens=False)
        p["n_tokens"] = len(ids)
        p["last_token_id"] = int(ids[-1])
        p["last_token_piece"] = tok.decode([ids[-1]])

    # 3. Load frozen unembedding, final gamma, per-layer gammas, and R_out/R_mid
    wanted_tensors = {"lm_head.weight", "model.norm.weight"} | {
        f"model.layers.{L}.{kind}.weight"
        for L in PRENORM_LAYERS
        for kind in ("post_feedforward_layernorm", "post_attention_layernorm")
    }
    w_map = load_safetensors_selected(snap, wanted_tensors)
    w_u = w_map["lm_head.weight"]       # [100278, 4096]
    gamma_fin = w_map["model.norm.weight"]  # [4096]

    W_U_eff = w_u * gamma_fin.unsqueeze(0)
    mu_U = W_U_eff.mean(dim=0)            # [4096]
    X_c = W_U_eff - mu_U.unsqueeze(0)
    Sigma_U = ((X_c.T @ X_c) / float(X_c.shape[0])).double()
    evals, evecs = torch.linalg.eigh(Sigma_U)
    evals = torch.clamp(evals, min=1e-9)
    tr_cov = float(evals.sum().item()) / 4096.0
    W_shrink_half = evecs @ torch.diag(1.0 / torch.sqrt(evals + COV_SHRINK_ALPHA * tr_cov)) @ evecs.T
    del X_c, Sigma_U

    ids_t = torch.tensor([t["id"] for t in targets], dtype=torch.long)
    U = W_U_eff[ids_t].float()
    U_norm = U.norm(dim=-1)
    U_hat = (U / U_norm.unsqueeze(-1)).float()
    U_ctr = (U.double() - mu_U.double().unsqueeze(0)).float()
    U_shrink = (W_shrink_half @ U_ctr.double().T).T.float()
    del W_U_eff, w_u

    strata = [t["stratum"] for t in targets]
    for i, t in enumerate(targets):
        t["u_norm"] = round(float(U_norm[i].item()), 4)

    def pair_cos_summary(mat: torch.Tensor) -> dict[str, float]:
        mat_n = mat.double() / mat.double().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        G = (mat_n @ mat_n.T).numpy()
        within, cross, of_v, ff_v, nn_v = [], [], [], [], []
        for i in range(24):
            for j in range(24):
                if i == j:
                    continue
                val = float(G[i, j])
                (within if strata[i] == strata[j] else cross).append(val)
                if (strata[i] == "function") != (strata[j] == "function"):
                    of_v.append(val)
                if strata[i] == "function" and strata[j] == "function":
                    ff_v.append(val)
                if strata[i] == "nouns" and strata[j] == "nouns":
                    nn_v.append(val)
        return {
            "within_pos": round(float(np.mean(within)), 4),
            "cross_pos": round(float(np.mean(cross)), 4),
            "open_x_func": round(float(np.mean(of_v)), 4),
            "func_x_func": round(float(np.mean(ff_v)), 4),
            "noun_x_noun": round(float(np.mean(nn_v)), 4),
        }

    unembed_cos = {
        "raw_u": pair_cos_summary(U),
        "centered_u_minus_mu": pair_cos_summary(U_ctr),
        "shrink_0.1_whitened": pair_cos_summary(U_shrink),
    }

    rl = torch.load(str(r_lens_path), map_location="cpu", weights_only=False)
    R_out_all = rl["means"]["out"]
    R_mid_all = rl["means"]["mid"]

    bank_tensors: dict[str, torch.Tensor] = {
        "mu_u": mu_U.float().clone().contiguous(),
        "u_vectors": U.clone().contiguous(),
        "u_hat": U_hat.clone().contiguous(),
        "u_centered": U_ctr.clone().contiguous(),
        "u_shrink": U_shrink.clone().contiguous(),
    }

    layer_geom: dict[str, Any] = {}
    prenorm_geom: dict[str, Any] = {}
    U_d = U.double()
    U_hat_d = U_hat.double()
    U_ctr_d = U_ctr.double()
    U_shrink_d = U_shrink.double()
    eye_d = torch.eye(4096, dtype=torch.float64)

    cross_mask = np.array([[strata[i] != strata[j] for j in range(24)] for i in range(24)])
    same_mask = np.array([[i != j and strata[i] == strata[j] for j in range(24)] for i in range(24)])

    for L in SWEEP_LAYERS:
        R = R_out_all[L].double()
        d = R.shape[1]
        lam = 0.5 * float((R * R).sum().item()) / d
        if str(L) in c0["layer_geometry"]:
            c0_lam = float(c0["layer_geometry"][str(L)]["lambda_ridge"])
            assert abs(lam - c0_lam) < 1e-2, f"L{L} lambda mismatch vs C0: {lam} vs {c0_lam}"

        bank_tensors[f"L{L}.rand_iso"] = make_rand_iso_draws(d, M_RAND_ISO_DRAWS, BANK_RAND_SEED_BASE + L)

        # Solve R_mf & all three R_ridge variants on L ∈ PRIMARY_C21_LAYERS ([18, 24, 27])
        if L in PRIMARY_C21_LAYERS:
            W_mf = U_d @ R
            W_mf_hat = W_mf / W_mf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            A = R.T @ R + lam * eye_d

            def solve_unit_ridge(src_mat: torch.Tensor) -> torch.Tensor:
                rhs = src_mat @ R  # [24, 4096]
                w = torch.linalg.solve(A, rhs.T).T
                return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            W_rg_raw = solve_unit_ridge(U_d)
            W_rg_ctr = solve_unit_ridge(U_ctr_d)
            W_rg_shk = solve_unit_ridge(U_shrink_d)

            c_vals = (U_hat_d * W_mf_hat).sum(dim=-1)
            dm_list = [
                make_direct_matched_draws(
                    U_hat[ti], W_mf_hat[ti].float(), W_rg_raw[ti].float(),
                    float(c_vals[ti].item()), K_DM_DRAWS, BANK_DM_SEED_BASE + L * 1000 + ti * 10,
                )
                for ti in range(24)
            ]
            bank_tensors[f"L{L}.R_mf"] = W_mf_hat.float().clone().contiguous()
            bank_tensors[f"L{L}.R_ridge_raw"] = W_rg_raw.float().clone().contiguous()
            bank_tensors[f"L{L}.R_ridge_centered"] = W_rg_ctr.float().clone().contiguous()
            bank_tensors[f"L{L}.R_ridge_shrink"] = W_rg_shk.float().clone().contiguous()
            bank_tensors[f"L{L}.direct_matched"] = torch.stack(dm_list, dim=0).clone().contiguous()

            # Compare predicted cross-domain bleed & rank-1 share on the fresh C2 24 targets
            frozen_arms: dict[str, Any] = {}
            for aname, What in [
                ("R_mf", W_mf_hat),
                ("R_ridge_raw", W_rg_raw),
                ("R_ridge_centered", W_rg_ctr),
                ("R_ridge_shrink", W_rg_shk),
                ("logit_lens", U_hat_d),
            ]:
                # M[src, dst] = u_dst^T (R @ w_src) evaluated ALWAYS on real uncentered u_dst = U_d!
                M = (What @ (R.T @ U_d.T)).numpy()
                diag = np.diag(M)
                M_rel = M / np.maximum(diag[:, None], 1e-9)
                Off = M.copy()
                np.fill_diagonal(Off, 0.0)
                svals = np.linalg.svd(Off, compute_uv=False)
                rank1_share = float((svals[0] ** 2) / (np.sum(svals ** 2) + 1e-12))
                cross_abs = float(np.mean(np.abs(M_rel[cross_mask])))
                cross_sig = float(np.mean(M_rel[cross_mask]))
                same_abs = float(np.mean(np.abs(M_rel[same_mask])))
                frozen_arms[aname] = {
                    "diag_unit_gain_mean": round(float(np.mean(diag)), 4),
                    "cross_abs_over_diag": round(cross_abs, 4),
                    "cross_signed_over_diag": round(cross_sig, 4),
                    "same_over_cross": round(same_abs / max(cross_abs, 1e-9), 4),
                    "offdiag_rank1_svd_share": round(rank1_share, 4),
                }

            layer_geom[str(L)] = {
                "role": "primary_C2.1_and_sweep",
                "lambda_ridge": round(lam, 6),
                "trace_R_over_d": round(float(torch.trace(R).item() / d), 4),
                "frozen_unit_gain_bleed_24": frozen_arms,
            }
        else:
            layer_geom[str(L)] = {
                "role": "sweep_layer",
                "lambda_ridge": round(lam, 6),
                "trace_R_over_d": round(float(torch.trace(R).item() / d), 4),
            }

        # Build Lacey low-gamma pre_norm vectors for C2.4 (L in [18, 21, 24, 27])
        if L in PRENORM_LAYERS:
            # Compute Shrink-whitened R_out and R_mid unit target writes at layer L
            R_mid = R_mid_all[L].double()
            lam_m = 0.5 * float((R_mid * R_mid).sum().item()) / d
            W_out_shk = torch.linalg.solve(R.T @ R + lam * eye_d, (U_shrink_d @ R).T).T
            W_out_shk = W_out_shk / W_out_shk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            W_mid_shk = torch.linalg.solve(R_mid.T @ R_mid + lam_m * eye_d, (U_shrink_d @ R_mid).T).T
            W_mid_shk = W_mid_shk / W_mid_shk.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            g_ff = w_map[f"model.layers.{L}.post_feedforward_layernorm.weight"].double()
            g_at = w_map[f"model.layers.{L}.post_attention_layernorm.weight"].double()

            def build_masked_prenorm(v_post: torch.Tensor, g_vec: torch.Tensor, drop_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
                order = torch.argsort(g_vec.abs())
                mask_01 = torch.ones_like(g_vec)
                mask_01[order[:drop_k]] = 0.0
                g_rms = float(torch.sqrt(torch.mean(g_vec ** 2)).item())
                ideal_pre = 1.0 / g_rms

                # Unmasked pre_norm solve (clip 1e-12 strictly to prevent div-by-exact-0 if any)
                g_safe = torch.where(g_vec.abs() < 1e-12, torch.full_like(g_vec, 1e-12), g_vec)
                pre_un = v_post / g_safe.unsqueeze(0)
                un_eff = float(torch.mean(ideal_pre / pre_un.norm(dim=-1)).item())
                pre_un_u = pre_un / pre_un.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                # Masked pre_norm solve (0 on masked coords, divide strictly on active coords)
                v_m = v_post * mask_01.unsqueeze(0)
                v_m_u = v_m / v_m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos_ret = float(torch.mean(torch.sum(v_post * v_m_u, dim=-1)).item())
                active = order[drop_k:]
                pre_m = torch.zeros_like(v_m_u)
                pre_m[:, active] = v_m_u[:, active] / g_vec[active].unsqueeze(0)
                m_norms = pre_m.norm(dim=-1)
                m_eff = float(torch.mean(ideal_pre / m_norms).item())
                worst_blowup_un = float(torch.max(pre_un.norm(dim=-1) / ideal_pre).item())
                worst_blowup_m = float(torch.max(m_norms / ideal_pre).item())
                pre_m_u = pre_m / m_norms.unsqueeze(-1).clamp_min(1e-12)
                assert float(pre_m_u[:, order[:drop_k]].abs().max().item()) == 0.0
                return mask_01, pre_un_u, pre_m_u, {
                    "gamma_rms": round(g_rms, 5),
                    "gamma_min_abs": round(float(g_vec.abs().min().item()), 6),
                    "drop_k": drop_k,
                    "unmasked_deliverable_frac": round(un_eff, 4),
                    "unmasked_worst_blowup": round(worst_blowup_un, 2),
                    "masked_deliverable_frac": round(m_eff, 4),
                    "masked_worst_blowup": round(worst_blowup_m, 2),
                    "masked_cos_ret": round(cos_ret, 5),
                }

            m_ff, pre_ff_un, pre_ff_m, ff_stats = build_masked_prenorm(W_out_shk, g_ff, GAMMA_DROP_FF)
            m_at, _, pre_at_m, at_stats = build_masked_prenorm(W_mid_shk, g_at, GAMMA_DROP_ATTN)
            bank_tensors[f"L{L}.gamma_ff_mask4"] = m_ff.float().clone().contiguous()
            bank_tensors[f"L{L}.gamma_attn_mask16"] = m_at.float().clone().contiguous()
            bank_tensors[f"L{L}.prenorm_ff_unmasked"] = pre_ff_un.float().clone().contiguous()
            bank_tensors[f"L{L}.prenorm_ff_masked"] = pre_ff_m.float().clone().contiguous()
            bank_tensors[f"L{L}.prenorm_attn_masked"] = pre_at_m.float().clone().contiguous()
            bank_tensors[f"L{L}.prenorm_ff_rand_masked"] = make_masked_rand_draws(
                m_ff, 24, K_PRENORM_RAND_DRAWS, BANK_PRENORM_RAND_SEED_BASE + L * 1000
            )
            bank_tensors[f"L{L}.prenorm_attn_rand_masked"] = make_masked_rand_draws(
                m_at, 24, K_PRENORM_RAND_DRAWS, BANK_PRENORM_RAND_SEED_BASE + L * 1000 + 500
            )
            prenorm_geom[str(L)] = {"ff": ff_stats, "attn": at_stats}

    out_bank = args.out_bank
    out_bank.parent.mkdir(parents=True, exist_ok=True)
    bank_meta = {
        "study": "OLMo-3-7B Stage C2",
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "r_lens_sha256": r_sha,
        "bank_dm_seed_base": str(BANK_DM_SEED_BASE),
        "bank_rand_seed_base": str(BANK_RAND_SEED_BASE),
        "bank_prenorm_rand_seed_base": str(BANK_PRENORM_RAND_SEED_BASE),
        "gamma_drop_ff": str(GAMMA_DROP_FF),
        "gamma_drop_attn": str(GAMMA_DROP_ATTN),
        "cov_shrink_alpha": str(COV_SHRINK_ALPHA),
    }
    write_canonical_safetensors(bank_tensors, out_bank, metadata=bank_meta)
    bank_sha = sha256_file(out_bank)

    # Exact forward-count budget plan (0 CU spent at registration)
    budget = {
        "clean_and_shams_and_c0_colon": 8 + 14 + 16,               # 38 fwds
        "block1_C2_1_bleed_3L_5arms_24x8": 3 * 5 * 24 * 8,         # 2,880 fwds (Raw, Centered, Shrink, logit_lens, R_mf)
        "block2_C2_2_bidirectional_mlp_patching_L19_31": 4 * 2 * 24 * 8,  # 1,536 fwds (Bare->Chat, Chat->Bare, L4-16 ctrl, Mismatch ctrl on L18 'all' + 'last' / attn cmp)
        "block2b_C2_3_sweep_7L_ll_and_rand32": 7 * (24 + 32) * 8,  # 3,136 fwds (7 layers x (24 targets logit_lens + 32 rand_iso) x 8 prompts)
        "block3_C2_4_prenorm_ablation_2doses": 8 * 24 * 4 * 2,     # 1,536 fwds (unmasked L24, masked L24, m70a30 L24, K=3 singles+combined L21/24/27, rand ctrls x 2 doses x 4 prompts)
        "total_planned_short_fwds": 38 + 2880 + 1536 + 3136 + 1536,
        "estimated_runtime_L4_or_A100_min": "8-14",
        "estimated_cost_CU": "0.45-1.10",
    }

    doc = {
        "schema_version": "c2-prereg-constants-v1",
        "study": "OLMo-3-7B Stage C2 (Clean Bleed Resolution, Causal ChatML MLP Attribution, Post-Erasure Sweep, & Pre-Norm Mask/Split Primitives)",
        "generator_script": "olmo3_virtual_and_highway/build_olmo3_c2_constants.py",
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "r_lens_sha256": r_sha,
        "c2_vector_bank_sha256": bank_sha,
        "targets_24_spec_sha256": targets_spec_sha,
        "parent_c0_constants_sha256": sha256_file(C0_CONST_PATH),
        "parent_c0b_constants_sha256": sha256_file(C0B_CONST_PATH),
        "parent_c1_constants_sha256": sha256_file(C1_CONST_PATH),
        "primary_c21_layers": PRIMARY_C21_LAYERS,
        "sweep_layers": SWEEP_LAYERS,
        "prenorm_layers": PRENORM_LAYERS,
        "registered_rng_and_inference": {
            "holm_family_size": 5,
            "holm_tests": ["p_C2.1_ctr_vs_raw", "p_C2.1_shk_vs_ctr", "p_C2.2_bare_into_chat", "p_C2.2_chat_into_bare", "p_C2.4_mask_vs_unmask"],
            "perm_seed": PERM_SEED,
            "perm_n_flips": 100000,
            "bca_seed": BCA_SEED,
            "bca_n_resamples": 10000,
            "bank_dm_seed_base": BANK_DM_SEED_BASE,
            "bank_rand_seed_base": BANK_RAND_SEED_BASE,
            "bank_prenorm_rand_seed_base": BANK_PRENORM_RAND_SEED_BASE,
        },
        "c0_anchor_colon_prediction": {
            "hypothesis": "C1.0.3's -23%/-27% post_mlp_norm drop at L24/L27 came from ':' endings vs C0's ' a' endings (Claude #103(1c))",
            "predicted_colon_L24_median_range": [6.2, 8.2],
            "predicted_colon_L24_point": 7.2,
            "predicted_colon_L27_median_range": [6.4, 8.6],
            "predicted_colon_L27_point": 7.5,
            "prompts_16": c0_colon_prompts,
        },
        "targets_24": targets,
        "prompts_8": prompts,
        "prompt_length_summary": {
            "bare_min": min(bare_T),
            "bare_max": max(bare_T),
            "bare_mean": round(float(np.mean(bare_T)), 2),
            "chatml_min": min(chat_T),
            "chatml_max": max(chat_T),
            "chatml_mean": round(float(np.mean(chat_T)), 2),
            "chatml_to_bare_mean_ratio": round(float(np.mean(chat_T) / np.mean(bare_T)), 3),
        },
        "unembed_pairwise_cosine": unembed_cos,
        "layer_geometry": layer_geom,
        "prenorm_gamma_mask_geometry": prenorm_geom,
        "planned_budget": budget,
    }

    args.out_constants.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    out_sha = sha256_file(args.out_constants)
    print(f"Wrote {args.out_constants} (sha256={out_sha})")
    print(f"Wrote {out_bank} (sha256={bank_sha})")
    for L in PRIMARY_C21_LAYERS:
        print(f"L{L} frozen bleed:", json.dumps(layer_geom[str(L)]["frozen_unit_gain_bleed_24"]))
    for L in PRENORM_LAYERS:
        print(f"L{L} pre_norm gamma mask:", json.dumps(prenorm_geom[str(L)]))


if __name__ == "__main__":
    main()
