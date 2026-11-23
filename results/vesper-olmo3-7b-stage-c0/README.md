# Stage C0 Results: Pre-Registered Out-of-Sample Transport & Active-Erasure Benchmark on OLMo 3 7B

**Status:** Pre-registered (`d1cb7cf7`, Registered v1; frozen code `4b5c7399`; executed on NVIDIA L4, `0.61 CU`, `1,142 s`, `13,871` forwards executed + `480` descriptive-`L6` capped traces skipped).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-REPORT.md)  
**Pre-Registered Design:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md) (`sha256 = 59b9da1a…`)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Pre-Registered Verdicts (100% Honest Read)

- **Stage C1 Gate (`C0.0` ∧ `C0.1` ∧ `C0.2` ∧ `C0.3`):** **FAIL** (blocked by `C0.1` and `C0.2`; Stage C1 is **not** launched under the registered gate).
- **All 4 Holm directional nulls rejected (`p_holm <= 0.0001`), while the pre-registered calibration & null bars exposed two crisp mechanistic discoveries:**

| ID | Verdict | Registered Result | Holm `p_raw` (`p_holm`) |
|---|---|---|---|
| **C0.0** | **PASS** | All `13,856` traces finite (`includes_rand_iso = true`), sham `0.0` (7/7), tel `p95 = 0.0425` (`<= 0.08`), `p99.9 = 0.0781` (`<= 0.15`), `L12–L27` cap rate `0.0000` (`0/5,376`), `fde_err p95 = 0.0759` (`<= 0.30`), `p99.9 = 0.3258` (`<= 0.45`), strict SHA-256 provenance match | — |
| **C0.1** | **FAIL** | `ρ_within = 0.7911` (`>= 0.30` PASS), `ρ_within(R_mf) = 0.6421` (`p_perm = 0.0018` PASS), `R_ridge frac_err = 0.118` (`<= 0.30` PASS), **FAIL** on `Δρ_within = +0.0732` (`BCa q05 = -0.0176 <= 0`) and `R_mf frac_err = 0.363` (`> 0.30`) | `0.000100` (`0.000100`) |
| **C0.2** | **FAIL** | `R_mf` beats `direct_matched` on `16/16` targets (`p_2 = 1.5e-5`, `m4 = 0.0193` logits linearity!), **FAIL** on gap calibration (`pooled = 0.2837 < 0.70`, `L12/18/24 = 0.171/0.235/0.383 < 0.50`) and `Retention(logit_lens) = 0.8046 > 0.75` | `0.000015` (`0.000061`) |
| **C0.3** | **PASS** | `mean(d_t) = +0.1026` (`15/16` targets positive on `L18+`, `16/16` at `L27`, `BCa q05 = +0.0760 > 0`, 95% CI `[+0.0712, +0.1353]`) | `0.000031` (`0.000061`) |
| **C0.4** | **FAIL** | Equal-direct efficiency **PASS** (`1.777×`, 95% CI `[1.636×, 1.898×]`, `16/16 >= 1.25×`, `KL = 0.0098` vs `0.0167`), **FAIL** on cross-domain `R_mf_cross` null (`cross_mean = 1.235` vs `rand_iso F_mean = 0.299`, `191/192 > 2F_mean`, `97.6%` positive) | `0.000015` (`0.000061`) |

## Why `C0.1`, `C0.2`, and `C0.4` Failed (Post-Hoc Mechanistic Diagnosis)
1. **Cross-position attention vs. single-position MLP channel (`C0.1` & `C0.2`):** The frozen R-lens (`neuron_lens/transport.py:5`) estimates the offset-summed Jacobian `mean_t Σ_{t' >= t} ∂h_final(t') / ∂h_L(t)` ("here and later"), which matched exploratory Exp 4B (`positions="all"`, where `R_mf ΣAttn = +3.47/+3.81`, `logit_lens ΣAttn = -4.01/-5.96`, gap ratio `0.831`). Stage C0 injected at `positions="last"` only, isolating the zero-offset slice `∂h_final(T) / ∂h_L(T)`: `ΣAttn` dropped `10–19×` (`+0.34/+0.45` and `-0.21/-0.32`), leaving primarily the same-position MLP channel (`L24: R_mf ΣMLP = +1.58` vs `logit_lens ΣMLP = -1.66`; `L27: +2.13` vs `-1.85`) at `28%–48%` of the full-offset lens gap.
2. **Confirmed same-position `u_t` contraction (`C0.3` PASS):** Even with cross-position attention shut off, `u_t` is strictly more contracted than isotropic directions `v^T J v` (`+0.1026`, `p = 3.1e-5`, `15/16` targets, `16/16` at `L27`). Real `v^T J v` (`0.694–0.970`) is `+0.204` above ridge-shrunk `tr(R_L)/d` (`0.511–0.836`).
3. **`R_mf` amplifies the shared noun common mode; `R_ridge` whitens it out (`C0.4`):** Across the 16 unseen singular nouns, raw `u_t` shares mean off-diagonal cosine `+0.183`; `R_mf = R_L^T u_t` amplifies this to `+0.262` (`L18`) / `+0.253` (`L24`), lifting all 12 cross-domain wrong nouns by `+1.226` logits (`97.6%` positive). `R_ridge` whitens off-diagonal cosine down `2.6×` to `+0.101 / +0.095`, yielding its `1.777×` (`16/16`) equal-direct specificity advantage and `0.118` calibration accuracy.

## Rescore it yourself (Python 3.10+ standard library only)

From the repository root (`cd assistant-basin`), using **only** the Python 3.10+ standard library (zero `numpy`, `scipy`, or `torch`; uses `math.fsum` for Python 3.10–3.14 byte-identity):

```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_c0.py \
  --receipt results/vesper-olmo3-7b-stage-c0/c0_receipt.json.gz \
  --constants olmo3_virtual_and_highway/c0_prereg_constants.json \
  --design olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md \
  --out-dir /tmp/olmo3_c0_rescore
cmp /tmp/olmo3_c0_rescore/scorecard.json results/vesper-olmo3-7b-stage-c0/scorecard.json
cmp /tmp/olmo3_c0_rescore/SCORECARD.md results/vesper-olmo3-7b-stage-c0/SCORECARD.md
```

Both `cmp` commands print nothing and exit `0` when reproduced byte-for-byte.
