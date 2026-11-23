#!/usr/bin/env python3
"""Build the frozen Stage C0 vector bank (~17 MB) locally at 0 CU and verify exact
agreement with `c0_prereg_constants.json` (`3be64ef8...`).

Inputs:
  - `r_lens.pt` (`sha256 = ceef21fc...`)
  - `c0_prereg_constants.json` (`sha256 = 3be64ef8...`)
  - `OLMO3-TRANSPORT-C0-DESIGN.md` (`sha256 = 59b9da1a...`)
  - OLMo-3-7B-Instruct `lm_head.weight` and `model.norm.weight` (from local HF cache or --hf-snapshot)

Outputs:
  - `c0_vector_bank.pt`: unit vectors for `R_mf`, `R_ridge`, `logit_lens`, `direct_matched` (K=4),
    and `rand_iso` (M=32/layer), plus `u_t = W_U[t] * gamma_final` and exact SHA-256 provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_CONSTANTS_SHA256 = "3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98"
EXPECTED_DESIGN_SHA256 = "59b9da1a585dbecba26954ae41a394f9a1a385bcc831a234b859d9d5e52a78bf"

DEFAULT_R_LENS = "/home/elisepc/green/Journal/notes/reports/vesper-olmo3-7b-neuron-lens/r_lens.pt"
DEFAULT_HF_SNAPSHOT = (
    "/home/elisepc/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct/"
    "snapshots/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
)
SEED_DM_BASE = 202611220
SEED_RAND_BASE = 202611221


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def orthonormal_basis(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Two-pass modified Gram-Schmidt returning an orthonormal basis for span(vecs)."""
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
    """Construct K unit vectors `c * u_hat + sqrt(1 - c^2) * z_perp` with `z_perp _|_ {u_hat, R_mf, R_ridge}`."""
    basis = orthonormal_basis([u_hat, w_rmf, w_rridge])
    d = u_hat.shape[0]
    s = math.sqrt(max(1.0 - c * c, 0.0))
    u_hat_d = u_hat.double()
    w_rmf_d = w_rmf.double()
    w_rr_d = w_rridge.double()
    draws = []
    for k in range(k_draws):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + k)
        z = torch.randn(d, generator=g, dtype=torch.float64)
        for _ in range(2):
            for b in basis:
                z = z - torch.dot(z, b) * b
        z_perp = z / z.norm().clamp_min(1e-12)
        assert abs(float(torch.dot(z_perp, u_hat_d).item())) < 1e-6
        assert abs(float(torch.dot(z_perp, w_rmf_d).item())) < 1e-6
        assert abs(float(torch.dot(z_perp, w_rr_d).item())) < 1e-6
        w = c * u_hat_d + s * z_perp
        w = w / w.norm().clamp_min(1e-12)
        assert abs(float(w.norm().item()) - 1.0) < 1e-6
        assert abs(float(torch.dot(w, u_hat_d).item()) - c) < 1e-5
        draws.append(w.float())
    return torch.stack(draws, dim=0)  # [K, d]


def make_rand_iso_draws(d: int, m_draws: int, seed: int) -> torch.Tensor:
    """Construct M isotropic unit vectors in R^d."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(m_draws, d, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v.float()  # [M, d]


def load_unembed_and_gamma(hf_snapshot: Path) -> tuple[torch.Tensor, torch.Tensor]:
    idx_path = hf_snapshot / "model.safetensors.index.json"
    with idx_path.open("r", encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    with safe_open(str(hf_snapshot / wm["lm_head.weight"]), framework="pt") as f:
        w_u = f.get_tensor("lm_head.weight").float()
    with safe_open(str(hf_snapshot / wm["model.norm.weight"]), framework="pt") as f:
        gamma = f.get_tensor("model.norm.weight").float()
    return w_u, gamma


def build_bank(
    constants_path: Path,
    design_path: Path,
    r_lens_path: Path,
    hf_snapshot: Path,
    out_path: Path,
    k_dm: int = 4,
    m_rand: int = 32,
    strict_sha: bool = True,
) -> dict:
    const_sha = sha256_file(constants_path)
    design_sha = sha256_file(design_path)
    r_lens_sha = sha256_file(r_lens_path)
    if strict_sha:
        assert const_sha == EXPECTED_CONSTANTS_SHA256, f"constants sha mismatch: {const_sha}"
        assert design_sha == EXPECTED_DESIGN_SHA256, f"design sha mismatch: {design_sha}"
        assert r_lens_sha == EXPECTED_R_LENS_SHA256, f"r_lens sha mismatch: {r_lens_sha}"

    const = json.loads(constants_path.read_text(encoding="utf-8"))
    targets = const["targets_16"]
    layers = [int(L) for L in const["layers"]]

    r_dict = torch.load(str(r_lens_path), map_location="cpu", weights_only=False)
    r_out = r_dict["means"]["out"]  # [32, d, d]
    d = int(r_out.shape[-1])

    w_u, gamma = load_unembed_and_gamma(hf_snapshot)
    u_vecs = []
    for t in targets:
        u = (w_u[t["id"]] * gamma).float()
        u_norm = float(u.norm().item())
        assert abs(u_norm - float(t["u_norm"])) < 1e-3, f"u_norm mismatch for {t['string']}: {u_norm} vs {t['u_norm']}"
        u_vecs.append(u)
    u_mat = torch.stack(u_vecs, dim=0)  # [16, d]
    u_hats = u_mat / u_mat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    layers_bank: dict[int, dict] = {}
    eye_d = torch.eye(d, dtype=torch.float64)
    for L in layers:
        r_mat = r_out[L].double()  # [d, d]
        rtr = r_mat.T @ r_mat
        lam = 0.5 * float(rtr.trace().item()) / d
        tr_over_d = float(r_mat.trace().item()) / d
        a_mat = rtr + lam * eye_d

        lg_l = const["layer_geometry"][str(L)]
        assert abs(lam - float(lg_l["lambda_ridge"])) < 5e-4, f"L{L} lambda mismatch: {lam} vs {lg_l['lambda_ridge']}"
        assert abs(tr_over_d - float(lg_l["trace_R_over_d"])) < 5e-4, f"L{L} tr/d mismatch"

        # Solve A X = R^T U^T for all 16 targets simultaneously
        rmf_raw_all = r_mat.T @ u_mat.double().T  # [d, 16]
        rr_raw_all = torch.linalg.solve(a_mat, rmf_raw_all)  # [d, 16]

        w_rmf_list, w_rr_list, w_ll_list, w_dm_list = [], [], [], []
        per_t_check = {}
        for ti, t in enumerate(targets):
            u_d = u_mat[ti].double()
            u_hat_d = u_hats[ti].double()
            u_norm = float(u_d.norm().item())

            rmf_raw = rmf_raw_all[:, ti]
            n_val = float(rmf_raw.norm().item()) / u_norm
            w_rmf = (rmf_raw / rmf_raw.norm().clamp_min(1e-12)).float()
            c_val = float(torch.dot(u_hat_d, w_rmf.double()).item())

            rr_raw = rr_raw_all[:, ti]
            w_rr = (rr_raw / rr_raw.norm().clamp_min(1e-12)).float()
            cos_rr = float(torch.dot(u_hat_d, w_rr.double()).item())

            w_ll = u_hats[ti].float()

            pred_ret_rmf = n_val / c_val
            pred_ret_rr = float(torch.dot(u_d, r_mat @ w_rr.double()).item() / torch.dot(u_d, w_rr.double()).item())
            pred_ret_dm = c_val * n_val
            pred_gap_d3 = (1.0 - c_val * c_val) * 3.0 * (n_val / c_val)

            # Verify exact agreement with frozen c0_prereg_constants.json
            ref = lg_l["per_target"][ti]
            assert ref["target"] == t["string"]
            assert abs(n_val - ref["N"]) < 5e-4, f"L{L} {t['string']} N mismatch"
            assert abs(c_val - ref["c"]) < 5e-4, f"L{L} {t['string']} c mismatch"
            assert abs(cos_rr - ref["cos_u_Rridge"]) < 5e-4, f"L{L} {t['string']} cos_u_Rridge mismatch"
            assert abs(pred_ret_rmf - ref["pred_ret_Rmf"]) < 1e-3, f"L{L} {t['string']} pred_ret_Rmf mismatch"
            assert abs(pred_ret_rr - ref["pred_ret_Rridge"]) < 1e-3, f"L{L} {t['string']} pred_ret_Rridge mismatch"
            assert abs(pred_ret_dm - ref["pred_ret_dm"]) < 5e-4, f"L{L} {t['string']} pred_ret_dm mismatch"
            assert abs(pred_gap_d3 - ref["pred_gap_D3"]) < 2e-3, f"L{L} {t['string']} pred_gap_D3 mismatch"

            w_dm = make_direct_matched_draws(
                u_hat=w_ll,
                w_rmf=w_rmf,
                w_rridge=w_rr,
                c=c_val,
                k_draws=k_dm,
                seed_base=SEED_DM_BASE + L * 1000 + ti * 10,
            )
            w_rmf_list.append(w_rmf)
            w_rr_list.append(w_rr)
            w_ll_list.append(w_ll)
            w_dm_list.append(w_dm)
            per_t_check[t["string"]] = {
                "N": n_val,
                "c": c_val,
                "cos_u_Rridge": cos_rr,
                "pred_ret_Rmf": pred_ret_rmf,
                "pred_ret_Rridge": pred_ret_rr,
                "pred_ret_dm": pred_ret_dm,
                "pred_ret_ll": pred_ret_dm,
                "pred_gain_Rmf": float(torch.dot(u_hat_d, r_mat @ w_rmf.double()).item()),
                "pred_gain_Rridge": float(torch.dot(u_hat_d, r_mat @ w_rr.double()).item()),
                "pred_gain_ll": float(torch.dot(u_hat_d, r_mat @ u_hat_d).item()),
                "pred_gain_dm_draws": [
                    float(torch.dot(u_hat_d, r_mat @ w_dm[k].double()).item()) for k in range(k_dm)
                ],
                "pred_gap_D3": pred_gap_d3,
            }

        rand_iso = make_rand_iso_draws(d=d, m_draws=m_rand, seed=SEED_RAND_BASE + L * 1000)
        layers_bank[L] = {
            "lambda_ridge": lam,
            "tr_R_over_d": tr_over_d,
            "R_mf": torch.stack(w_rmf_list, dim=0),             # [16, d]
            "R_ridge": torch.stack(w_rr_list, dim=0),           # [16, d]
            "logit_lens": torch.stack(w_ll_list, dim=0),        # [16, d]
            "direct_matched": torch.stack(w_dm_list, dim=0),    # [16, K, d]
            "rand_iso": rand_iso,                               # [M, d]
            "per_target_geom": per_t_check,
        }

    payload = {
        "meta": {
            "model": const["model"],
            "revision": const["revision"],
            "r_lens_sha256": r_lens_sha,
            "constants_sha256": const_sha,
            "design_sha256": design_sha,
            "d_model": d,
            "k_dm": k_dm,
            "m_rand": m_rand,
            "seed_dm_base": SEED_DM_BASE,
            "seed_rand_base": SEED_RAND_BASE,
        },
        "targets_16": targets,
        "layers": layers,
        "u_vectors": u_mat,  # [16, d] float32
        "layers_bank": layers_bank,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    bank_sha = sha256_file(out_path)
    print(f"Built Stage C0 bank: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB), sha256={bank_sha}")
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", type=Path, default=repo_root / "olmo3_virtual_and_highway/c0_prereg_constants.json")
    ap.add_argument("--design", type=Path, default=repo_root / "olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md")
    ap.add_argument("--r-lens", type=Path, default=Path(DEFAULT_R_LENS))
    ap.add_argument("--hf-snapshot", type=Path, default=Path(DEFAULT_HF_SNAPSHOT))
    ap.add_argument("--out", type=Path, default=repo_root / "work/olmo3_c0/c0_vector_bank.pt")
    ap.add_argument("--no-strict-sha", action="store_true")
    args = ap.parse_args()
    build_bank(
        constants_path=args.constants,
        design_path=args.design,
        r_lens_path=args.r_lens,
        hf_snapshot=args.hf_snapshot,
        out_path=args.out,
        strict_sha=not args.no_strict_sha,
    )


if __name__ == "__main__":
    main()
