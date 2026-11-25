#!/usr/bin/env python3
"""build_olmo3_c1_constants.py — Deterministic 0-CU pre-registration & vector-bank builder for OLMo 3 7B Stage C1.

Computes and freezes:
  1. `olmo3_virtual_and_highway/c1_prereg_constants.json`
  2. `work/olmo3_c1/c1_vector_bank.pt` (deterministic float32 unit write vectors, K=2 orthogonal
     `direct_matched` draws per target/layer, and M=32 isotropic `rand_iso` draws per layer;
     SHA-256 recorded inside `c1_prereg_constants.json`).

Inputs (repo-relative defaults; override via CLI flags, zero hardcoded user paths):
  - Frozen OLMo-3-7B-Instruct snapshot (`6e5971d9...`): tokenizer, `lm_head.weight`, `model.norm.weight`
  - Frozen 64-doc RelP R-lens `r_lens.pt` (`sha256 = ceef21fc...`)
  - Published Stage C0 (`c0_prereg_constants.json`) & Stage C0b (`c0b_prereg_constants.json`) constants

Verifies:
  1. 24/24 mixed-POS single-token targets (0 overlap with C0 targets or persona words).
  2. 8/8 all-metalinguistic POS-neutral prompts (4 Bare + 4 ChatML, 0 overlap with C0 prompts).
  3. Exact L12/L18/L24/L27 frozen-lens geometry, equal-weight per-target `d_sep(t) = R_sep(R_ridge,t) - R_sep(R_mf,t)`,
     pooled ratio-of-means `R_sep`, `k_ridge` sweep, and Regime-D `dose/pmn` anchor predictions.
  4. Coherent Monte-Carlo `C1.0.3` gate calibration (`fde = 3s/(1-s)` from the same shrink draws,
     fitted to C0b's exact L18/24/27 medians & maxima, both iid and target-clustered `rho=0.35`,
     plus in-band `L18` mild-to-severe breakdown catch probabilities).
  5. `L12_linear_check` calibration (`<= 2/192` > `0.09` shrink & median `frac_err <= 0.12`).
  6. Non-tautological exhaustive disjointness check on `C1.3`'s 4-way fork across ALL 7 registered inputs.
  7. Empirically sourced Monte-Carlo power tables for ALL 4 claims (`C1.1`, `C1.2`, `C1.3` sub-test + joint `H_A` + `H_B`, `C1.4`).
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
C0_CONST_PATH = ROOT / "c0_prereg_constants.json"
C0B_CONST_PATH = ROOT / "c0b_prereg_constants.json"
DEFAULT_OUT_CONSTANTS = ROOT / "c1_prereg_constants.json"
DEFAULT_OUT_BANK = REPO_ROOT / "work" / "olmo3_c1" / "c1_vector_bank.safetensors"

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

PRIMARY_LAYERS = [18, 24, 27]
SECONDARY_LAYERS = [12]
ALL_LAYERS = [12, 18, 24, 27]

# Registered deterministic RNG seeds & rules (N14)
PERM_SEED = 2026112401
BCA_SEED = 2026112402
BANK_DM_SEED_BASE = 2026112403
BANK_RAND_SEED_BASE = 2026112404
K_DM_DRAWS = 2
M_RAND_ISO_DRAWS = 32

# Measured Stage C0 post_mlp_norm anchors (for L12/L18/L24/L27)
POST_MLP_NORM_ANCHOR = {12: 4.2478, 18: 7.4537, 24: 12.1555, 27: 14.9000}

# C1.3 registered floor & separation thresholds (C1-1/C1-2: upper bound 3.20 + whitening_ok precondition -> 5-way partition)
C1_3_DELTA_R_SEP_BAR = 0.25
C1_3_DELTA_R_SEP_Q05_BAR = 0.05
C1_3_FLOOR_HA_MAX = 1.90
C1_3_FLOOR_HB_MIN = 2.20
C1_3_FLOOR_HB_MAX = 3.20
C1_3_FLOOR_HB_Q05_BAR = 1.85

TARGETS_24_SPEC: list[tuple[str, str]] = [
    ("nouns", " violin"),
    ("nouns", " anchor"),
    ("nouns", " forest"),
    ("nouns", " bridge"),
    ("nouns", " blanket"),
    ("nouns", " marble"),
    ("verbs", " whisper"),
    ("verbs", " dissolve"),
    ("verbs", " wander"),
    ("verbs", " gather"),
    ("verbs", " scatter"),
    ("verbs", " borrow"),
    ("adjectives", " brittle"),
    ("adjectives", " hollow"),
    ("adjectives", " crimson"),
    ("adjectives", " silent"),
    ("adjectives", " gentle"),
    ("adjectives", " narrow"),
    ("function", " although"),
    ("function", " because"),
    ("function", " between"),
    ("function", " without"),
    ("function", " perhaps"),
    ("function", " neither"),
]

PERSONA_WORDS = {" pirate", " librarian", " wizard", " robot", " owl", " fox", " whale", " baker"}

BARE_STEMS: list[tuple[str, str, str]] = [
    ("narr_0", "narrative", "In the quiet room, the next word on the page was:"),
    ("narr_1", "narrative", "On the index card beside the lamp, the single word printed was:"),
    ("anal_0", "analytical", "In the archive log, the isolated token recorded in column three was:"),
    ("anal_1", "analytical", "Looking over the notes, the highlighted word in the margin was:"),
]


def write_canonical_safetensors(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    """Write a standard .safetensors file with sorted keys and deterministic JSON header (zero HashMap randomness)."""
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


def build_prompts() -> list[dict[str, Any]]:
    prompts = []
    for sid, style, stem in BARE_STEMS:
        prompts.append({"id": f"c1_bare_{sid}", "format": "bare", "style": style, "stem": stem, "text": stem})
    for sid, style, stem in BARE_STEMS:
        chat = (
            f"<|im_start|>user\nContinue this line with one word: {stem}<|im_end|>\n"
            f"<|im_start|>assistant\n{stem}"
        )
        prompts.append({"id": f"c1_chat_{sid}", "format": "chatml", "style": style, "stem": stem, "text": chat})
    return prompts


def orthonormal_basis(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Two-pass modified Gram-Schmidt returning an orthonormal basis in float64."""
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
    return torch.stack(draws, dim=0)


def make_rand_iso_draws(d: int, m_draws: int, seed: int) -> torch.Tensor:
    """Construct M isotropic unit vectors in R^d."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(m_draws, d, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v.float()


def c1_3_predicates(
    delta_r_sep: float,
    q05_sep_ok: bool,
    p_ha_ok: bool,
    strata_ok: bool,
    r_floor_ridge: float,
    q05_floor_ok: bool,
    p_hb_ok: bool,
    whitening_ok: bool,
) -> dict[str, bool]:
    """Explicit, mutually exclusive predicates for C1.3's 6-way fork (NO fallback `else`).

    Fixes M2 + L1 (Round 4):
      - whitening_ok := (CI_low_90_BCa(exp(mean(u_abs))) > 1.0) AND (CI_low_90_BCa(exp(mean(u_abs_open_func))) > 1.0)
        (establishes whitening occurred at all on both all-23 off-target & open*func pair sets; leaves 1.35 bar to C1.2).
      - Splits the former non-active outcome into two distinct outcomes:
          1. WHITENING_NOT_ESTABLISHED: NOT whitening_ok  (non-claim; does not assert equivalence/no-whitening)
          2. FLOOR_ABOVE_CEILING:       whitening_ok AND (r_floor_ridge > 3.20)
      - active   := whitening_ok AND (r_floor_ridge <= 3.20)
      - sep_pass := delta_r_sep >= 0.25 AND q05_sep_ok AND p_ha_ok AND strata_ok
      - hb_pass  := 2.20 <= r_floor_ridge <= 3.20 AND q05_floor_ok AND p_hb_ok
      - low_fl   := r_floor_ridge <= 1.90
      - mid_fl   := 1.90 < r_floor_ridge < 2.20
      - band_fl  := 2.20 <= r_floor_ridge <= 3.20

    Disjoint 6-way partition of all 8 inputs:
      1. WHITENING_NOT_ESTABLISHED: NOT whitening_ok
      2. FLOOR_ABOVE_CEILING:       whitening_ok AND (r_floor_ridge > 3.20)
      3. H_A:                       active AND sep_pass AND low_fl
      4. H_B:                       active AND hb_pass AND NOT sep_pass
      5. NEITHER:                   active AND low_fl AND NOT sep_pass
      6. MIXED:                     active AND (mid_fl OR (band_fl AND NOT hb_pass) OR (sep_pass AND hb_pass))
    """
    above_ceiling = whitening_ok and (r_floor_ridge > C1_3_FLOOR_HB_MAX)
    active = whitening_ok and (r_floor_ridge <= C1_3_FLOOR_HB_MAX)
    sep_pass = (delta_r_sep >= C1_3_DELTA_R_SEP_BAR) and q05_sep_ok and p_ha_ok and strata_ok
    hb_pass = (C1_3_FLOOR_HB_MIN <= r_floor_ridge <= C1_3_FLOOR_HB_MAX) and q05_floor_ok and p_hb_ok
    low_fl = r_floor_ridge <= C1_3_FLOOR_HA_MAX
    mid_fl = (r_floor_ridge > C1_3_FLOOR_HA_MAX) and (r_floor_ridge < C1_3_FLOOR_HB_MIN)
    band_fl = (r_floor_ridge >= C1_3_FLOOR_HB_MIN) and (r_floor_ridge <= C1_3_FLOOR_HB_MAX)
    return {
        "WHITENING_NOT_ESTABLISHED": not whitening_ok,
        "FLOOR_ABOVE_CEILING": above_ceiling,
        "H_A": active and sep_pass and low_fl,
        "H_B": active and hb_pass and (not sep_pass),
        "NEITHER": active and low_fl and (not sep_pass),
        "MIXED": active and (mid_fl or (band_fl and (not hb_pass)) or (sep_pass and hb_pass)),
    }


def verify_c1_3_exhaustiveness() -> dict[str, Any]:
    """Non-tautological check across ALL 8 registered C1.3 inputs: asserts exact sum(matches) == 1 everywhere,
    reporting both full Cartesian counts and strictly feasible counts (excluding floor < 1.85 with p_hb_ok/q05_floor_ok)."""
    seps = [-0.10, 0.24, 0.25, 0.40]
    floors = [1.40, 1.90, 1.91, 2.05, 2.19, 2.20, 2.67, 3.20, 3.21, 4.42]
    counts: dict[str, int] = {
        "WHITENING_NOT_ESTABLISHED": 0,
        "FLOOR_ABOVE_CEILING": 0,
        "H_A": 0,
        "H_B": 0,
        "NEITHER": 0,
        "MIXED": 0,
    }
    feasible_counts: dict[str, int] = dict(counts)
    mixed_both = 0
    infeasible = 0
    for ds in seps:
        for fl in floors:
            for q05_s in (True, False):
                for p_a in (True, False):
                    for st_ok in (True, False):
                        for q05_f in (True, False):
                            for p_b in (True, False):
                                for wh_ok in (True, False):
                                    m = c1_3_predicates(ds, q05_s, p_a, st_ok, fl, q05_f, p_b, wh_ok)
                                    active = [k for k, v in m.items() if v]
                                    assert len(active) == 1, f"Non-partition at {(ds,fl,q05_s,p_a,st_ok,q05_f,p_b,wh_ok)}: {active}"
                                    counts[active[0]] += 1
                                    is_feas = not (fl < C1_3_FLOOR_HB_Q05_BAR and (p_b or q05_f))
                                    if is_feas:
                                        feasible_counts[active[0]] += 1
                                    else:
                                        infeasible += 1
                                    sep_pass = (ds >= C1_3_DELTA_R_SEP_BAR) and q05_s and p_a and st_ok
                                    hb_pass = (C1_3_FLOOR_HB_MIN <= fl <= C1_3_FLOOR_HB_MAX) and q05_f and p_b
                                    if wh_ok and (fl <= C1_3_FLOOR_HB_MAX) and sep_pass and hb_pass:
                                        mixed_both += 1
    return {
        "grid_points_tested": sum(counts.values()),
        "feasible_grid_points": sum(feasible_counts.values()),
        "infeasible_cells_excluded_in_feasible_counts": infeasible,
        "axes_covered": ["delta_r_sep", "r_floor_ridge", "q05_sep_ok", "p_ha_ok", "strata_ok", "q05_floor_ok", "p_hb_ok", "whitening_ok"],
        "undefined_or_overlapping": 0,
        "outcome_counts": counts,
        "feasible_outcome_counts": feasible_counts,
        "mixed_both_subcells": mixed_both,
    }


def verify_c1_2_exhaustiveness() -> dict[str, Any]:
    """Exhaustive 4-way disjointness check for C1.2 under the registered BCa containment invariant
    (CI_low <= point_est <= CI_high; any violation is a fail-closed scorer error)."""
    # Grid of (lo, pt, hi) intervals satisfying lo <= pt <= hi across boundaries 0.80 and 1.35
    pts = [0.60, 0.79, 0.80, 1.05, 1.34, 1.35, 1.60]
    intervals = []
    for i, lo in enumerate(pts):
        for j in range(i, len(pts)):
            pt = pts[j]
            for k in range(j, len(pts)):
                hi = pts[k]
                intervals.append((lo, pt, hi))
    counts = {"PASS": 0, "REVERSED": 0, "REFUTED_EQUIVALENCE": 0, "INCONCLUSIVE": 0}
    for lo_a, pt_a, hi_a in intervals:
        for lo_n, pt_n, hi_n in intervals:
            for aux_ok in (True, False):  # p_IUT <= alpha_holm & >=20/24 & >=3/4 strata
                is_pass = aux_ok and (pt_a >= 1.35) and (pt_n >= 1.35)
                is_rev = (hi_a < 0.80) or (hi_n < 0.80)
                is_eq = (lo_a > 0.80) and (hi_a < 1.35) and (lo_n > 0.80) and (hi_n < 1.35)
                is_inc = (not is_pass) and (not is_rev) and (not is_eq)
                matches = [
                    k for k, v in (
                        ("PASS", is_pass),
                        ("REVERSED", is_rev),
                        ("REFUTED_EQUIVALENCE", is_eq),
                        ("INCONCLUSIVE", is_inc),
                    )
                    if v
                ]
                assert len(matches) == 1, f"C1.2 non-partition at {(lo_a,pt_a,hi_a),(lo_n,pt_n,hi_n),aux_ok}: {matches}"
                counts[matches[0]] += 1
    return {
        "grid_points_tested": sum(counts.values()),
        "bca_containment_invariant": "CI_low <= point_estimate <= CI_high (enforced fail-closed by scorer)",
        "undefined_or_overlapping": 0,
        "outcome_counts": counts,
    }


def simulate_c1_0_3_gate_prior(n_rep: int = 10000, seed: int = 202611240) -> dict[str, Any]:
    """Coherent C1.0.3 gate calibration (`fde = 3s/(1-s)` from the SAME shrink draws),
    fitted directly to C0b's published L18/L24/L27 medians and maxima (exact receipt values),
    covering both null false-fail rates (scaled 0.89x & unscaled 1.0x, iid & target-clustered rho=0.35)
    AND in-band L18 breakdown catch probabilities (1.25x, 1.50x, 1.75x, 2.00x)."""
    # Exact C0b published 3L Regime-D 'all' shrink moments:
    # L18 R_mf (n=128): med=0.0616, max=0.1557 -> sigma = ln(0.1557/0.0616)/2.418 = 0.3834
    # L24 R_mf (n=128): med=0.0087, max=0.0393 -> sigma = ln(0.0393/0.0087)/2.418 = 0.6238
    # L27 R_mf (n=128): med=0.0092, max=0.0263 -> sigma = ln(0.0263/0.0092)/2.418 = 0.4344
    # L18 dm   (n=512): med=0.0113, max=0.0730 -> sigma = ln(0.0730/0.0113)/2.880 = 0.6478
    # L24 dm   (n=512): med=0.0042, max=0.0177 -> sigma = ln(0.0177/0.0042)/2.880 = 0.4994
    # L27 dm   (n=512): med=0.0026, max=0.0113 -> sigma = ln(0.0113/0.0026)/2.880 = 0.5101
    sig18 = math.log(0.1557 / 0.0616) / 2.418
    sig24 = math.log(0.0393 / 0.0087) / 2.418
    sig27 = math.log(0.0263 / 0.0092) / 2.418
    sigdm18 = math.log(0.0730 / 0.0113) / 2.880
    sigdm24 = math.log(0.0177 / 0.0042) / 2.880
    sigdm27 = math.log(0.0113 / 0.0026) / 2.880

    def run_scenario(l18_med: float, prompt_rho: float, sc_seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(sc_seed)
        c_r20 = c_r12 = c_fp95 = c_f45 = c_f75 = c_any = 0
        e45, e12, p95s, maxrs, maxfs = [], [], [], [], []
        for _ in range(n_rep):
            if prompt_rho > 0.0:
                z_t = rng.normal(0.0, math.sqrt(prompt_rho), size=(24, 1))
                z_p = rng.normal(0.0, math.sqrt(1.0 - prompt_rho), size=(24, 8))
                s18 = np.exp(math.log(l18_med) + sig18 * (z_t + z_p)).ravel()
            else:
                s18 = rng.lognormal(math.log(l18_med), sig18, size=192)
            s24 = rng.lognormal(math.log(0.0087), sig24, size=192)
            s27 = rng.lognormal(math.log(0.0092), sig27, size=192)
            dm18 = rng.lognormal(math.log(0.0113), sigdm18, size=384)
            dm24 = rng.lognormal(math.log(0.0042), sigdm24, size=384)
            dm27 = rng.lognormal(math.log(0.0026), sigdm27, size=384)
            rmsn = np.concatenate([s18, s24, s27, dm18, dm24, dm27])
            # Exact coherent relation verified on C0b L18 R_mf (max |fde - 3s/(1-s)| = 0.0000):
            fde = 3.0 * rmsn / np.maximum(1.0 - rmsn, 1e-6)

            r20 = int(np.sum(rmsn > 0.20)) > 2    # > 2 / 1,728 (0.12%)
            r12 = int(np.sum(rmsn > 0.12)) > 26   # > 26 / 1,728 (1.50%)
            fp95 = float(np.percentile(fde, 95)) > 0.30
            fc45 = int(np.sum(fde > 0.45)) > 12   # > 12 / 1,728 (0.69%)
            fm75 = int(np.sum(fde > 0.75)) > 2    # > 2 / 1,728 (0.12%)
            c_r20 += int(r20)
            c_r12 += int(r12)
            c_fp95 += int(fp95)
            c_f45 += int(fc45)
            c_f75 += int(fm75)
            c_any += int(r20 or r12 or fp95 or fc45 or fm75)
            e45.append(int(np.sum(fde > 0.45)))
            e12.append(int(np.sum(rmsn > 0.12)))
            p95s.append(float(np.percentile(fde, 95)))
            maxrs.append(float(np.max(rmsn)))
            maxfs.append(float(np.max(fde)))
        return {
            "l18_median_shrink": round(l18_med, 4),
            "prompt_rho": prompt_rho,
            "mean_count_fde_gt_045": round(float(np.mean(e45)), 2),
            "mean_count_rmsn_gt_12pct": round(float(np.mean(e12)), 2),
            "median_fde_p95": round(float(np.median(p95s)), 4),
            "median_fde_max": round(float(np.median(maxfs)), 4),
            "median_rmsn_max": round(float(np.median(maxrs)), 4),
            "fail_prob_rmsn_gt_20pct_over_2": round(c_r20 / n_rep, 4),
            "fail_prob_rmsn_gt_12pct_over_26": round(c_r12 / n_rep, 4),
            "fail_prob_fde_p95_gt_030": round(c_fp95 / n_rep, 4),
            "fail_prob_fde_gt_045_over_12": round(c_f45 / n_rep, 4),
            "fail_prob_fde_gt_075_over_2": round(c_f75 / n_rep, 4),
            "fail_prob_any_clause": round(c_any / n_rep, 4),
        }

    return {
        "replicates": n_rep,
        "pool_size": 1728,
        "arm_mix": "576 R_mf (1/3) + 1,152 direct_matched (2/3)",
        "coherent_relation": "fde_err = 3*s/(1-s) from identical shrink draws (C0b L18 R_mf max discrepancy = 0.0000)",
        "fitted_c0b_sigmas": {
            "L18_R_mf": round(sig18, 4),
            "L24_R_mf": round(sig24, 4),
            "L27_R_mf": round(sig27, 4),
            "L18_dm": round(sigdm18, 4),
            "L24_dm": round(sigdm24, 4),
            "L27_dm": round(sigdm27, 4),
        },
        "null_false_fail_scenarios": {
            "c1_scaled_0.89x_iid": run_scenario(0.0548, 0.0, seed + 1),
            "c1_scaled_0.89x_clustered_rho035": run_scenario(0.0548, 0.35, seed + 2),
            "c0b_unscaled_1.00x_iid": run_scenario(0.0616, 0.0, seed + 3),
            "c0b_unscaled_1.00x_clustered_rho035": run_scenario(0.0616, 0.35, seed + 4),
        },
        "in_band_L18_breakdown_catch_power": {
            "L18_1.25x_mild_med_0.0770": run_scenario(0.0616 * 1.25, 0.0, seed + 10),
            "L18_1.50x_moderate_med_0.0924": run_scenario(0.0616 * 1.50, 0.0, seed + 11),
            "L18_1.75x_strong_med_0.1078": run_scenario(0.0616 * 1.75, 0.0, seed + 12),
            "L18_2.00x_severe_med_0.1232": run_scenario(0.0616 * 2.00, 0.0, seed + 13),
        },
    }


def simulate_l12_linear_check(n_rep: int = 10000, seed: int = 202611245) -> dict[str, Any]:
    """Calibration for secondary L12_linear_check (<= 2/192 > 0.09 shrink AND median frac_err <= 0.12)."""
    sig_l12 = math.log(0.3514 / 0.1572) / 2.418
    s_med_lin = 0.1572 * (0.30 / 1.32)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "replicates": n_rep,
        "c0b_l12_sigma_fit": round(sig_l12, 4),
        "linear_scaled_median_shrink": round(s_med_lin, 4),
        "rule": "<= 2/192 R_mf 'all' traces with |r_i/r_c - 1| > 0.09 AND median frac_err <= 0.12",
        "shrink_leg_fail_prob_by_superlinearity": {},
    }
    for mult, label in [(1.0, "1.0x_linear"), (1.25, "1.25x_mild_superlinear"), (1.5, "1.5x_superlinear"), (2.0, "2.0x_superlinear")]:
        draws = rng.lognormal(math.log(s_med_lin * mult), sig_l12, size=(n_rep, 192))
        out["shrink_leg_fail_prob_by_superlinearity"][label] = round(float(((draws > 0.09).sum(axis=1) > 2).mean()), 4)
    return out


def simulate_power_tables(n_rep: int = 10000, seed: int = 202611241) -> dict[str, Any]:
    """Empirically sourced Monte-Carlo power tables for ALL 4 claims (5-test Holm α/5 = 0.010, t_crit ≈ 2.50)."""
    rng = np.random.default_rng(seed)
    t_holm_strict = 2.50  # 24-target one-sided t threshold matching α/5 = 0.010 (df=23: t_0.99 = 2.500)

    # C1.1(b) Attention attribution (sourced from C0b 3L: share_gap med=0.9111 SD=0.0654, dTot_gap mean=3.914 SD=0.345, dAttn_Rmf mean=2.286 SD=0.269)
    # Bar: mean(dTot_gap) >= 1.50, median(share_gap) >= 0.75, q05(share_gap) > 0.70, >=3/4 strata >= 0.70,
    #      IUT Holm p_C1.1 = max(p(dAttn_gap - 0.75*dTot_gap > 0), p(dAttn_Rmf - dAttn_dm > 0)) <= 0.010
    c1_1_power: dict[str, dict[str, float]] = {}
    for true_share in [0.70, 0.75, 0.80, 0.85, 0.90]:
        row = {}
        for sd_sh in [0.065, 0.10, 0.14]:
            sh = rng.normal(true_share, sd_sh, size=(n_rep, 4, 6))
            dtot = np.maximum(rng.normal(3.5, 0.45, size=(n_rep, 4, 6)), 0.5)
            margin = (sh - 0.75) * dtot
            flat_sh = sh.reshape(n_rep, 24)
            flat_m = margin.reshape(n_rep, 24)
            med_sh = np.median(flat_sh, axis=1)
            se_sh = flat_sh.std(axis=1, ddof=1) / math.sqrt(24)
            q05_sh = med_sh - 1.714 * 1.253 * se_sh  # asymptotic SE of median
            m_bar = flat_m.mean(axis=1)
            se_m = flat_m.std(axis=1, ddof=1) / math.sqrt(24)
            t_gap = m_bar / np.maximum(se_m, 1e-9)
            strat_ok = (np.median(sh, axis=2) >= 0.70).sum(axis=1) >= 3
            passed = (med_sh >= 0.75) & (q05_sh > 0.70) & strat_ok & (t_gap >= t_holm_strict)
            row[f"sd_{sd_sh:.3f}"] = round(float(passed.mean()), 4)
        c1_1_power[f"true_share_{true_share:.2f}"] = row

    # C1.2 Whitening (sourced from C0b 3L: SD_log(u_abs)=0.069, SD_log(u_norm)=0.066; conservatively bracketing {0.15, 0.25, 0.35})
    # IUT Holm p_C1.2 = max(p_abs, p_norm) <= 0.010, geo_abs >= 1.35, geo_norm >= 1.35, >=20/24 pos, >=3/4 strata >= 1.15
    sd_grid = [0.15, 0.25, 0.35]
    c1_2_power: dict[str, dict[str, float]] = {}
    for true_geo in [1.00, 1.20, 1.35, 1.50, 1.70]:
        mu = math.log(true_geo)
        row = {}
        for sd in sd_grid:
            x = rng.normal(mu, sd, size=(n_rep, 4, 6))
            # Correlated u_norm (r=0.92 with u_abs as in C0b, same true_geo conservative)
            z = rng.normal(mu, sd, size=(n_rep, 4, 6))
            x_norm = mu + 0.92 * (x - mu) + math.sqrt(1 - 0.92**2) * (z - mu)
            flat_a = x.reshape(n_rep, 24)
            flat_n = x_norm.reshape(n_rep, 24)
            ma, mn = flat_a.mean(axis=1), flat_n.mean(axis=1)
            ta = ma / np.maximum(flat_a.std(axis=1, ddof=1) / math.sqrt(24), 1e-9)
            tn = mn / np.maximum(flat_n.std(axis=1, ddof=1) / math.sqrt(24), 1e-9)
            pos_cnt = (flat_a > 0).sum(axis=1)
            strat_ok = (np.exp(x.mean(axis=2)) >= 1.15).sum(axis=1) >= 3
            passed = (np.exp(ma) >= 1.35) & (np.exp(mn) >= 1.35) & (pos_cnt >= 20) & strat_ok & (np.minimum(ta, tn) >= t_holm_strict)
            row[f"sd_{sd:.2f}"] = round(float(passed.mean()), 4)
        c1_2_power[f"true_{true_geo:.2f}x"] = row

    # C1.3(a) Separation sub-test ONLY: equal-weight per-target d_sep(t) = R_sep(R_ridge,t) - R_sep(R_mf,t)
    # Sourced from C0b 3L live empirical SD(d_sep) = 0.1342 & frozen C1 SD(d_sep) = 0.1385 -> bracketing {0.15, 0.25, 0.35}
    # Bar: mean(d_sep) >= 0.25, q05 > 0.05, p_HA <= 0.010 (t >= 2.50), >= 3/4 strata mean > 0
    c1_3_sep_subtest: dict[str, dict[str, float]] = {}
    for true_d in [0.00, 0.20, 0.25, 0.30, 0.40]:
        row = {}
        for sd in sd_grid:
            x = rng.normal(true_d, sd, size=(n_rep, 4, 6))
            flat = x.reshape(n_rep, 24)
            m = flat.mean(axis=1)
            se = flat.std(axis=1, ddof=1) / math.sqrt(24)
            q05 = m - 1.714 * se
            t_ha = m / np.maximum(se, 1e-9)
            strat_ok = (x.mean(axis=2) > 0).sum(axis=1) >= 3
            passed = (m >= C1_3_DELTA_R_SEP_BAR) & (q05 > C1_3_DELTA_R_SEP_Q05_BAR) & (t_ha >= t_holm_strict) & strat_ok
            row[f"sd_{sd:.2f}"] = round(float(passed.mean()), 4)
        c1_3_sep_subtest[f"true_delta_{true_d:.2f}"] = row

    # C1.3(b) Joint H_A verdict power (separation sub-test AND R_floor <= 1.90), SD(d_sep)=0.20, per-target SD(R_floor_t)=0.50 (matching H_B mid grid)
    c1_3_joint_ha: dict[str, dict[str, float]] = {}
    for true_d in [0.25, 0.30, 0.40]:
        row = {}
        for true_fl in [1.50, 1.73, 1.77, 1.90, 2.20, 2.56]:
            x = rng.normal(true_d, 0.20, size=(n_rep, 4, 6))
            flat = x.reshape(n_rep, 24)
            m = flat.mean(axis=1)
            se = flat.std(axis=1, ddof=1) / math.sqrt(24)
            sep_ok = (m >= C1_3_DELTA_R_SEP_BAR) & ((m - 1.714 * se) > C1_3_DELTA_R_SEP_Q05_BAR) & ((m / np.maximum(se, 1e-9)) >= t_holm_strict) & ((x.mean(axis=2) > 0).sum(axis=1) >= 3)
            fl = rng.normal(true_fl, 0.50, size=(n_rep, 24)).mean(axis=1)
            row[f"true_floor_{true_fl:.2f}"] = round(float((sep_ok & (fl <= C1_3_FLOOR_HA_MAX)).mean()), 4)
        c1_3_joint_ha[f"true_delta_{true_d:.2f}"] = row

    # C1.3(c) H_B floor leg power (2.20 <= R_floor <= 3.20 AND q05 > 1.85 AND p_HB <= 0.010; inert 4.42/5.58 rejected -> 0.0% H_B!)
    c1_3_hb: dict[str, dict[str, float]] = {}
    for true_fl in [1.75, 2.00, 2.20, 2.40, 2.56, 2.67, 3.20, 4.42, 5.58]:
        row = {}
        for sd_fl in [0.35, 0.50, 0.65]:
            fl_t = rng.normal(true_fl, sd_fl, size=(n_rep, 24))
            m = fl_t.mean(axis=1)
            se = fl_t.std(axis=1, ddof=1) / math.sqrt(24)
            q05 = m - 1.714 * se
            t_hb = (m - C1_3_FLOOR_HB_Q05_BAR) / np.maximum(se, 1e-9)
            hb_ok = (m >= C1_3_FLOOR_HB_MIN) & (m <= C1_3_FLOOR_HB_MAX) & (q05 > C1_3_FLOOR_HB_Q05_BAR) & (t_hb >= t_holm_strict)
            row[f"sd_{sd_fl:.2f}"] = round(float(hb_ok.mean()), 4)
        c1_3_hb[f"true_floor_{true_fl:.2f}"] = row

    # C1.4 Contraction (sourced from C0.3: mean=+0.1026, SD≈0.065; bracketing {0.04, 0.06, 0.08})
    c1_4_power: dict[str, dict[str, float]] = {}
    for true_d in [0.00, 0.05, 0.08, 0.10, 0.13]:
        row = {}
        for sd in [0.04, 0.06, 0.08]:
            x = rng.normal(true_d, sd, size=(n_rep, 4, 6))
            flat = x.reshape(n_rep, 24)
            m = flat.mean(axis=1)
            se = flat.std(axis=1, ddof=1) / math.sqrt(24)
            t_stat = m / np.maximum(se, 1e-9)
            pos_cnt = (flat > 0).sum(axis=1)
            strat_ok = (x.mean(axis=2) > 0).sum(axis=1) >= 3
            passed = (m > 0) & (t_stat >= t_holm_strict) & (pos_cnt >= 19) & strat_ok
            row[f"sd_{sd:.2f}"] = round(float(passed.mean()), 4)
        c1_4_power[f"true_dt_{true_d:.2f}"] = row

    return {
        "replicates": n_rep,
        "proxy_note": "One-sample t-test proxy (df=23, t>=2.50 for alpha/5=0.010); registered inference is exact 100,000-flip sign-flip (conservative for symmetric/right-skewed targets). C1.1 row prices the attention-share leg only.",
        "holm_strict_t_crit_alpha_over_5": t_holm_strict,
        "empirical_sd_sources": {
            "C1.1_share_gap_c0b_3L": 0.0654,
            "C1.2_u_abs_c0b_3L": 0.0690,
            "C1.3_d_sep_c0b_3L": 0.1342,
            "C1.3_d_sep_frozen_c1_3L": 0.1385,
            "C1.3_floor_sd_grid_note": "bracketed {0.35, 0.50, 0.65} around frozen C1 per-target B_open_func(R_ridge)/F_mean_3L dispersion (~0.50); joint H_A uses midpoint 0.50",
            "C1.4_dt_c0_3L_from_ci": 0.0650,
        },
        "C1.1_attn_share": c1_1_power,
        "C1.2_whitening_IUT": c1_2_power,
        "C1.3_delta_R_sep_subtest_only": c1_3_sep_subtest,
        "C1.3_joint_H_A_sd_sep_0.20_sd_fl_0.50": c1_3_joint_ha,
        "C1.3_H_B_floor_leg_bounded_3.20": c1_3_hb,
        "C1.4_token_direction_contraction": c1_4_power,
    }


def resolve_hf_snapshot(model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    from huggingface_hub.constants import HF_HUB_CACHE
    cand = Path(HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REV
    if cand.is_dir():
        return cand
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, revision=MODEL_REV, local_files_only=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic Stage C1 pre-registration constants and vector bank (0 CU).")
    ap.add_argument("--r-lens", type=Path, default=DEFAULT_R_LENS, help="Path to frozen r_lens.pt")
    ap.add_argument("--model-dir", type=str, default=None, help="Optional local HF snapshot directory for OLMo-3-7B-Instruct")
    ap.add_argument("--out-constants", type=Path, default=DEFAULT_OUT_CONSTANTS, help="Output path for c1_prereg_constants.json")
    ap.add_argument("--out-bank", type=Path, default=DEFAULT_OUT_BANK, help="Output path for c1_vector_bank.pt")
    args = ap.parse_args()

    r_lens_path = args.r_lens
    r_sha = sha256_file(r_lens_path)
    assert r_sha == EXPECTED_R_LENS_SHA256, f"r_lens sha mismatch: {r_sha}"

    c0 = json.loads(C0_CONST_PATH.read_text(encoding="utf-8"))
    c0_words = {t["string"] for t in c0["targets_16"]}
    c0b = json.loads(C0B_CONST_PATH.read_text(encoding="utf-8"))
    c0_prompt_texts = {p["text"] for p in c0b["prompts_8"]}

    snap = resolve_hf_snapshot(args.model_dir)
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)

    # 1. Verify 24 fresh mixed-POS targets
    targets = []
    for stratum, word in TARGETS_24_SPEC:
        assert word not in c0_words, f"Overlap with C0 target: {word}"
        assert word not in PERSONA_WORDS, f"Overlap with persona: {word}"
        ids = tok.encode(word, add_special_tokens=False)
        dec = tok.decode(ids)
        assert len(ids) == 1 and dec == word, f"Not single roundtrip token: {word!r} -> {ids} ({dec!r})"
        targets.append({"stratum": stratum, "string": word, "id": int(ids[0]), "decoded_piece": dec})
    assert len({t["id"] for t in targets}) == 24

    # 2. Verify 8 metalinguistic prompts
    prompts = build_prompts()
    for p in prompts:
        assert p["text"] not in c0_prompt_texts, f"Overlap with C0 prompt: {p['id']}"
        ids = tok.encode(p["text"], add_special_tokens=False)
        p["n_tokens"] = len(ids)
        p["last_token_id"] = int(ids[-1])
        p["last_token_piece"] = tok.decode([ids[-1]])
    bare_T = [p["n_tokens"] for p in prompts if p["format"] == "bare"]
    chat_T = [p["n_tokens"] for p in prompts if p["format"] == "chatml"]

    # 3. Load frozen unembedding, gamma_final, and R_out
    wm = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    with safe_open(str(snap / wm["lm_head.weight"]), framework="pt") as f:
        w_u = f.get_tensor("lm_head.weight").float()
    with safe_open(str(snap / wm["model.norm.weight"]), framework="pt") as f:
        gamma = f.get_tensor("model.norm.weight").float()

    rl = torch.load(str(r_lens_path), map_location="cpu", weights_only=False)
    R_out = rl["means"]["out"]

    ids_t = torch.tensor([t["id"] for t in targets], dtype=torch.long)
    U = w_u[ids_t] * gamma.unsqueeze(0)  # [24, 4096]
    U_norm = U.norm(dim=-1)
    U_hat = U / U_norm.unsqueeze(-1)
    strata = [t["stratum"] for t in targets]
    for i, t in enumerate(targets):
        t["u_norm"] = round(float(U_norm[i].item()), 4)

    def pair_cos_summary(mat_unit: torch.Tensor) -> dict[str, float]:
        G = (mat_unit @ mat_unit.T).numpy()
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

    unembed_cos = pair_cos_summary(U_hat)

    # 4. Layer geometry, frozen-lens predictions & deterministic vector bank
    c0_layers = c0["layer_geometry"]
    layer_geom: dict[str, Any] = {}
    layers_bank: dict[int, dict[str, Any]] = {}
    per_t_b_in = {"R_mf": np.zeros(24), "R_ridge": np.zeros(24)}
    per_t_b_of = {"R_mf": np.zeros(24), "R_ridge": np.zeros(24)}

    for L in ALL_LAYERS:
        R = R_out[L].double()
        d = R.shape[1]
        lam = float(c0_layers[str(L)]["lambda_ridge"])
        tr_over_d = float(torch.trace(R).item() / d)

        U_d = U.double()
        U_hat_d = U_hat.double()
        W_mf = U_d @ R
        W_mf_hat_d = W_mf / W_mf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        A = R.T @ R + lam * torch.eye(d, dtype=torch.float64)
        W_rg = torch.linalg.solve(A, W_mf.T).T
        W_rg_hat_d = W_rg / W_rg.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        W_mf_hat = W_mf_hat_d.float()
        W_rg_hat = W_rg_hat_d.float()

        c_vals = (U_hat_d * W_mf_hat_d).sum(dim=-1)
        u_ret = (U_hat_d * (R @ U_hat_d.T).T).sum(dim=-1)
        doses_d3 = 3.0 / (U_d * W_mf_hat_d).sum(dim=-1)
        pmn = POST_MLP_NORM_ANCHOR[L]
        dose_ratios = doses_d3 / pmn
        pred_rmf_d3 = doses_d3 * (U_d * (R @ W_mf_hat_d.T).T).sum(dim=-1)
        pred_dm_d3 = doses_d3 * c_vals * (U_d * (R @ U_hat_d.T).T).sum(dim=-1)
        pred_gap_d3 = pred_rmf_d3 - pred_dm_d3

        # Deterministic bank entries for layer L
        dm_tensor = None
        rand_tensor = None
        if L in PRIMARY_LAYERS:
            dm_list = []
            for ti in range(24):
                dm_list.append(
                    make_direct_matched_draws(
                        U_hat[ti], W_mf_hat[ti], W_rg_hat[ti], float(c_vals[ti].item()),
                        K_DM_DRAWS, BANK_DM_SEED_BASE + L * 1000 + ti * 10,
                    )
                )
            dm_tensor = torch.stack(dm_list, dim=0)  # [24, 2, 4096]
            rand_tensor = make_rand_iso_draws(d, M_RAND_ISO_DRAWS, BANK_RAND_SEED_BASE + L)

        per_target_geom = {}
        for ti, t in enumerate(targets):
            per_target_geom[t["string"]] = {
                "c": round(float(c_vals[ti].item()), 6),
                "u_ret": round(float(u_ret[ti].item()), 6),
                "dose_d3_anchor": round(float(doses_d3[ti].item()), 6),
                "pred_rmf_d3": round(float(pred_rmf_d3[ti].item()), 6),
                "pred_dm_d3": round(float(pred_dm_d3[ti].item()), 6),
            }

        layers_bank[L] = {
            "lambda_ridge": lam,
            "tr_R_over_d": tr_over_d,
            "R_mf": W_mf_hat,
            "R_ridge": W_rg_hat,
            "logit_lens": U_hat.float(),
            "direct_matched": dm_tensor,
            "rand_iso": rand_tensor,
            "per_target_geom": per_target_geom,
        }

        # Regime N (0.75x for L18/24/27; 0.30x for L12 secondary)
        dose_n = (0.30 if L == 12 else 0.75) * pmn
        arms_pred: dict[str, Any] = {}
        for aname, What in [("R_mf", W_mf_hat_d), ("R_ridge", W_rg_hat_d), ("logit_lens", U_hat_d)]:
            # M[j, i] = |u_j^T R w_i| (read token j under injected target i)
            M = (dose_n * (U_d @ (R @ What.T)).abs()).numpy()
            in_v, cr_v, of_v = [], [], []
            for i in range(24):
                in_i, of_i = [], []
                for j in range(24):
                    if i == j:
                        continue
                    v = float(M[j, i])
                    (in_v if strata[j] == strata[i] else cr_v).append(v)
                    if strata[j] == strata[i]:
                        in_i.append(v)
                    if (strata[j] == "function") != (strata[i] == "function"):
                        of_v.append(v)
                        of_i.append(v)
                if L in PRIMARY_LAYERS and aname in per_t_b_in:
                    per_t_b_in[aname][i] += float(np.mean(in_i)) / len(PRIMARY_LAYERS)
                    per_t_b_of[aname][i] += float(np.mean(of_i)) / len(PRIMARY_LAYERS)
            b_in = float(np.mean(in_v))
            b_of = float(np.mean(of_v))
            arms_pred[aname] = {
                "self_mean": round(float(np.diag(M).mean()), 4),
                "within_mean": round(b_in, 4),
                "cross_mean": round(float(np.mean(cr_v)), 4),
                "open_x_func_mean": round(b_of, 4),
                "R_sep_ratio_of_means": round(b_in / max(b_of, 1e-9), 4),
            }

        layer_geom[str(L)] = {
            "role": "secondary_capped_0.30x" if L == 12 else "primary_confirmatory",
            "lambda_ridge": round(lam, 6),
            "post_mlp_norm_anchor": pmn,
            "trace_R_over_d": round(tr_over_d, 4),
            "u_ret_median": round(float(u_ret.median().item()), 4),
            "c_median": round(float(c_vals.median().item()), 4),
            "c_min": round(float(c_vals.min().item()), 4),
            "dose_D3_over_pmn_median": round(float(dose_ratios.median().item()), 4),
            "dose_D3_over_pmn_max": round(float(dose_ratios.max().item()), 4),
            "pred_gap_D3_median": round(float(pred_gap_d3.median().item()), 4),
            "pred_gap_D3_mean": round(float(pred_gap_d3.mean().item()), 4),
            "write_cos_R_mf": pair_cos_summary(W_mf_hat),
            "write_cos_R_ridge": pair_cos_summary(W_rg_hat),
            "regime_N_dose": round(dose_n, 4),
            "regime_N_frozen_preds": arms_pred,
            "frozen_delta_R_sep_ratio_of_means": round(
                arms_pred["R_ridge"]["R_sep_ratio_of_means"] - arms_pred["R_mf"]["R_sep_ratio_of_means"], 4
            ),
        }

    # Save deterministic frozen vector bank as safetensors (zero pickle/address nondeterminism) and hash it (N9)
    out_bank = args.out_bank
    out_bank.parent.mkdir(parents=True, exist_ok=True)
    bank_tensors: dict[str, torch.Tensor] = {
        "u_vectors": U.float().clone().contiguous(),
        "u_hat": U_hat.float().clone().contiguous(),
    }
    for L in ALL_LAYERS:
        lb = layers_bank[L]
        bank_tensors[f"L{L}.R_mf"] = lb["R_mf"].clone().contiguous()
        bank_tensors[f"L{L}.R_ridge"] = lb["R_ridge"].clone().contiguous()
        if lb["direct_matched"] is not None:
            bank_tensors[f"L{L}.direct_matched"] = lb["direct_matched"].clone().contiguous()
        if lb["rand_iso"] is not None:
            bank_tensors[f"L{L}.rand_iso"] = lb["rand_iso"].clone().contiguous()
    bank_meta = {
        "study": "OLMo-3-7B Stage C1",
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "r_lens_sha256": r_sha,
        "bank_dm_seed_base": str(BANK_DM_SEED_BASE),
        "bank_rand_seed_base": str(BANK_RAND_SEED_BASE),
        "k_dm_draws": str(K_DM_DRAWS),
        "m_rand_iso_draws": str(M_RAND_ISO_DRAWS),
    }
    write_canonical_safetensors(bank_tensors, out_bank, metadata=bank_meta)
    bank_sha = sha256_file(out_bank)

    # Equal-weight per-target 3L frozen R_sep & d_sep(t) (primary C1.3 estimand — fixes N7 & F16!)
    rsep_mf_t = per_t_b_in["R_mf"] / per_t_b_of["R_mf"]
    rsep_rg_t = per_t_b_in["R_ridge"] / per_t_b_of["R_ridge"]
    dsep_t = rsep_rg_t - rsep_mf_t

    b_of_rg_3l = float(np.mean([layer_geom[str(L)]["regime_N_frozen_preds"]["R_ridge"]["open_x_func_mean"] for L in PRIMARY_LAYERS]))
    b_in_rg_3l = float(np.mean([layer_geom[str(L)]["regime_N_frozen_preds"]["R_ridge"]["within_mean"] for L in PRIMARY_LAYERS]))
    b_of_mf_3l = float(np.mean([layer_geom[str(L)]["regime_N_frozen_preds"]["R_mf"]["open_x_func_mean"] for L in PRIMARY_LAYERS]))
    b_in_mf_3l = float(np.mean([layer_geom[str(L)]["regime_N_frozen_preds"]["R_mf"]["within_mean"] for L in PRIMARY_LAYERS]))
    f_mean_c0_3l = 0.2958

    # L4 fix: exact frozen per-target dispersion of B_open_func(R_ridge, t) / F_mean_3L (unattenuated & at k_ridge=1.7254)
    frozen_r_floor_t_unatt = per_t_b_of["R_ridge"] / f_mean_c0_3l
    frozen_r_floor_t_k17254 = frozen_r_floor_t_unatt / 1.7254
    targets_24_sha256 = hashlib.sha256(
        json.dumps(TARGETS_24_SPEC, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    # M3 fix: explicit prior branch distribution at C0b's measured k_ridge=1.7254 (floor=2.559) under BOTH d_sep priors
    # (frozen C1: mean=+0.2980, sd=0.1385 vs. C0b live measured: mean=+0.2132, sd=0.1342)
    rng_m3 = np.random.default_rng(2026112501)
    m3_priors = {}
    for label, mu_s, sd_s in [
        ("frozen_C1_3L_prior", float(dsep_t.mean()), float(dsep_t.std(ddof=1))),
        ("c0b_live_measured_prior", 0.2132, 0.1342),
    ]:
        xs = rng_m3.normal(mu_s, sd_s, size=(20000, 4, 6))
        fl = xs.reshape(20000, 24)
        m = fl.mean(axis=1)
        se = fl.std(axis=1, ddof=1) / math.sqrt(24)
        p_sep = float(((m >= C1_3_DELTA_R_SEP_BAR) & ((m - 1.714 * se) > C1_3_DELTA_R_SEP_Q05_BAR) & ((m / np.maximum(se, 1e-9)) >= 2.50) & ((xs.mean(axis=2) > 0).sum(axis=1) >= 3)).mean())
        m3_priors[label] = {
            "mean_d_sep": round(mu_s, 4),
            "sd_d_sep": round(sd_s, 4),
            "P_sep_pass_at_bar_0.25": round(p_sep, 4),
            "implied_P_MIXED_BOTH_at_floor_2.559": round(p_sep, 4),
            "implied_P_H_B_at_floor_2.559": round(1.0 - p_sep, 4),
        }

    k_ridge_sweep = []
    for k, note in [
        (1.0, "frozen_unattenuated_no_whitening_scale"),
        (1.25, "weak_overprediction"),
        (1.5, "moderate_overprediction"),
        (1.7254, "c0b_measured_R_ridge_overprediction_1.3627_over_0.7898"),
        (2.0, "H_B_lower_boundary_scale"),
        (2.3238, "H_A_cap_boundary_1.90"),
        (2.55, "c0b_measured_R_mf_overprediction_borrowed_reference"),
    ]:
        imp_b_of = b_of_rg_3l / k
        k_ridge_sweep.append({
            "k_ridge": k,
            "note": note,
            "implied_B_open_func_R_ridge": round(imp_b_of, 4),
            "implied_R_floor_vs_C0_Fmean": round(imp_b_of / f_mean_c0_3l, 3),
        })

    budget = {
        "clean_baselines_and_shams": 8 + 7,
        "block1_C1_1_regime_D_all_and_last": 3 * 24 * 8 * 6,
        "block2a_C1_2_3_4_regime_N_last": 3 * 24 * 8 * 3,
        "block2b_rand_iso_32draws_positions_last": 3 * 32 * 8,
        "secondary_L12_capped_030x": 1 * 24 * 8 * 3,
    }
    budget["total_planned_forwards"] = sum(budget.values())

    doc = {
        "schema_version": "c1-prereg-constants-v5",
        "study": "OLMo-3-7B Stage C1 (Fresh Mixed-POS Transport, H_A/H_B Bleed Fork & Token-Direction Contraction)",
        "generator_script": "olmo3_virtual_and_highway/build_olmo3_c1_constants.py",
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "r_lens_sha256": r_sha,
        "c1_vector_bank_sha256": bank_sha,
        "targets_24_spec_sha256": targets_24_sha256,
        "parent_c0_constants_sha256": sha256_file(C0_CONST_PATH),
        "parent_c0b_constants_sha256": sha256_file(C0B_CONST_PATH),
        "primary_layers": PRIMARY_LAYERS,
        "secondary_layers": SECONDARY_LAYERS,
        "registered_rng_and_inference": {
            "holm_family_size": 5,
            "holm_tests": ["p_C1.1_IUT", "p_C1.2_IUT", "p_C1.3_HA", "p_C1.3_HB", "p_C1.4"],
            "holm_alpha_stepdown": [0.01, 0.0125, 0.016667, 0.025, 0.05],
            "holm_stop_rule": "sort ascending p_(1)..p_(5); reject k while p_(k) <= 0.05/(6-k); stop at first failure and retain remaining",
            "perm_rng": "Python stdlib random.Random(perm_seed)",
            "perm_seed": PERM_SEED,
            "perm_n_flips": 100000,
            "perm_flip_rule": "targets in targets_24 index order 0..23; sign = +1 if rng.getrandbits(1) else -1; p = (1 + count(stat >= obs)) / (1 + 100000)",
            "bca_rng": "Python stdlib random.Random(bca_seed)",
            "bca_seed": BCA_SEED,
            "bca_n_resamples": 10000,
            "bca_containment_rule": "90% BCa CI must satisfy CI_low <= point_estimate <= CI_high; violation triggers fail-closed scorer error",
            "bank_dm_seed_base": BANK_DM_SEED_BASE,
            "bank_rand_seed_base": BANK_RAND_SEED_BASE,
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
        "pooled_3L_frozen_summary": {
            "per_target_equal_weight_R_mf_R_sep_mean": round(float(rsep_mf_t.mean()), 4),
            "per_target_equal_weight_R_ridge_R_sep_mean": round(float(rsep_rg_t.mean()), 4),
            "per_target_equal_weight_delta_R_sep_mean": round(float(dsep_t.mean()), 4),
            "per_target_equal_weight_delta_R_sep_sd": round(float(dsep_t.std(ddof=1)), 4),
            "per_target_equal_weight_delta_R_sep_min": round(float(dsep_t.min()), 4),
            "per_target_equal_weight_R_ridge_B_of_min": round(float(per_t_b_of["R_ridge"].min()), 4),
            "per_target_frozen_R_floor_sd_unattenuated": round(float(frozen_r_floor_t_unatt.std(ddof=1)), 4),
            "per_target_frozen_R_floor_sd_at_k_ridge_1.7254": round(float(frozen_r_floor_t_k17254.std(ddof=1)), 4),
            "pooled_ratio_of_means_R_mf_R_sep": round(b_in_mf_3l / b_of_mf_3l, 4),
            "pooled_ratio_of_means_R_ridge_R_sep": round(b_in_rg_3l / b_of_rg_3l, 4),
            "pooled_ratio_of_means_delta_R_sep": round((b_in_rg_3l / b_of_rg_3l) - (b_in_mf_3l / b_of_mf_3l), 4),
            "R_ridge_B_open_func_3L": round(b_of_rg_3l, 4),
            "c1_1_registered_calibration_bands": {
                "pooled_3L_median_r_all": [0.65, 1.50],
                "pooled_3L_bca90_ci": [0.55, 1.60],
                "bare_3L_median_r_all": [0.65, 1.50],
                "chatml_3L_median_r_all": [0.65, 1.50],
                "per_layer_abs_chatml_minus_bare_max": 0.35,
                "verdict_rule": "Any pooled or per-format median(r_all) outside [0.65, 1.50] REFUTES C1.1; INCONCLUSIVE (format/length/instruction split) applies ONLY when (b), frac_err, pooled [0.65,1.50], AND both Bare/ChatML [0.65,1.50] pass while per-layer |ChatML-Bare| > 0.35.",
            },
            "c1_3_registered_thresholds": {
                "delta_R_sep_bar": C1_3_DELTA_R_SEP_BAR,
                "delta_R_sep_q05_bar": C1_3_DELTA_R_SEP_Q05_BAR,
                "R_floor_HA_max": C1_3_FLOOR_HA_MAX,
                "R_floor_HB_min": C1_3_FLOOR_HB_MIN,
                "R_floor_HB_max": C1_3_FLOOR_HB_MAX,
                "R_floor_HB_q05_bar": C1_3_FLOOR_HB_Q05_BAR,
                "whitening_ok_definition": "CI_low_90_BCa(exp(mean(u_abs))) > 1.0 AND CI_low_90_BCa(exp(mean(u_abs_open_func))) > 1.0",
                "F_mean_3L_definition": "live in-run Block 2b 32-draw isotropic off-target logit floor mean_{L in 3L} mean_{v in 32} mean_{p in 8} mean_{t in 24} |dlogit_t(rand_iso_v)| at positions='last' (matching C0 score_olmo3_c0.py:1102-1116 comparator 0.295811 logits; self-retention v^T J v from the same 768 forwards is used for C1.4)",
                "F_mean_3L_hard_provenance_band_vs_C0_0.2958": [0.85, 1.15],
                "F_mean_3L_clean_band_vs_C0_0.2958": [0.94, 1.06],
                "F_mean_3L_sensitivity_zone_rule": "for F_live/0.2958 in [0.85, 0.94) U (1.06, 1.15], evaluate 6-way partition at BOTH F_live and 0.2958; return INCONCLUSIVE (F_mean_3L sensitivity) if labels differ",
            },
            "c1_3_dual_d_sep_prior_branch_odds_at_k_ridge_1.7254": m3_priors,
            "k_ridge_sensitivity_vs_C0_Fmean_3L": k_ridge_sweep,
        },
        "c1_0_3_gate_prior_calibration": simulate_c1_0_3_gate_prior(),
        "l12_linear_check_calibration": simulate_l12_linear_check(),
        "c1_2_fork_exhaustiveness_check": verify_c1_2_exhaustiveness(),
        "c1_3_fork_exhaustiveness_check": verify_c1_3_exhaustiveness(),
        "power_simulations": simulate_power_tables(),
        "planned_budget": budget,
    }

    args.out_constants.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    out_sha = sha256_file(args.out_constants)
    print(f"Wrote {args.out_constants} (sha256={out_sha})")
    print(f"Wrote {out_bank} (sha256={bank_sha})")
    print("Pooled 3L summary:", json.dumps(doc["pooled_3L_frozen_summary"], indent=2))
    print("C1.0.3 gate calibration:", json.dumps(doc["c1_0_3_gate_prior_calibration"], indent=2))
    print("L12 linear check calibration:", json.dumps(doc["l12_linear_check_calibration"], indent=2))
    print("C1.3 exhaustiveness:", doc["c1_3_fork_exhaustiveness_check"])
    print("Power simulations:", json.dumps(doc["power_simulations"], indent=2))


if __name__ == "__main__":
    main()
