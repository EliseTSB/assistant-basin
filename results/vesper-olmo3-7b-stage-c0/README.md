# Stage C0 Results: Pre-Registered Out-of-Sample Transport & Active-Erasure Benchmark on OLMo 3 7B

**Status:** Pre-registered (`d1cb7cf7`, Registered v1; runner/scorer frozen at git commit `4b5c7399`, attested by git history; executed on NVIDIA L4, `1,142.3 s`, `13,871` forwards executed + `480` descriptive-`L6` capped traces skipped; `0.587 CU` compute alone at `1.85 CU/hr`, `0.61 CU` billed via Colab ledger).  
**Registration provenance (added at publish):** the design was registered on Vesper's working branch at commit `d1cb7cf7` (2026-11-22 22:13 CT) and the runner and scorer frozen at `4b5c7399` (2026-11-23 00:22 CT), before the run (01:43 CT). That branch is not public because its history carries local paths, so those times rest on the branch history and the council record; the published design is byte-identical to the registered one (SHA-256 `59b9da1a…`, which the scorer checks in strict mode).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-REPORT.md)  
**Pre-Registered Design:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md) (`sha256 = 59b9da1a…`)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Pre-Registered Verdicts (100% Honest Read)

- **Stage C1 Gate (`C0.0` ∧ `C0.1` ∧ `C0.2` ∧ `C0.3`):** **FAIL** (blocked by `C0.1` and `C0.2`; Stage C1 is **not** launched under the registered gate).
- **All 4 Holm directional nulls rejected (`p_holm <= 0.0001`); `C0.3` passed, while `C0.1`, `C0.2`, and `C0.4` failed their calibration/null bars:**

| ID | Verdict | Registered Result | Holm `p_raw` (`p_holm`) |
|---|---|---|---|
| **C0.0** | **PASS** | All `13,856` traces finite (`includes_rand_iso = true`), sham `0.0` (7/7), tel `p95 = 0.0425` (`<= 0.08`), `p99.9 = 0.0781` (`<= 0.15`), `L12–L27` cap rate `0.0000` (`0/5,376`), `fde_err p95 = 0.0759` (`<= 0.30`), `p99.9 = 0.3258` (`<= 0.45`), strict SHA-256 provenance match | — |
| **C0.1** | **FAIL** | `ρ_within = 0.7911` (`>= 0.30` PASS), `ρ_within(R_mf) = 0.6421` (`p_perm = 0.0018` PASS; non-gating BCa `q05 = +0.2494`), `R_ridge` median `frac_err = 0.118` (`<= 0.30` PASS), **FAIL** on `Δρ_within = +0.0732` (`BCa q05 = -0.0176 <= 0`) and `R_mf` median `frac_err = 0.363` (`> 0.30`) | `0.000100` (`0.000100`) |
| **C0.2** | **FAIL** | `R_mf` beats `direct_matched` on `16/16` targets (`p_2 = 1.5e-5`, Perpendicular + write-norm nonlinearity diagnostic `= 0.0193` logits), **FAIL** on gap calibration (`pooled = 0.2837 < 0.70`, `L12/18/24 = 0.171/0.235/0.383 < 0.50`) and both `Retention(logit_lens)` bounds (`pooled = 0.8046 > 0.75`; per-layer `L24 = 0.8634 > 0.85`, `L27 = 0.8543 > 0.85`) | `0.000015` (`0.000061`) |
| **C0.3** | **PASS** | `mean(d_t) = +0.1026` (`15/16` targets positive on `L18+`, `16/16` at `L27`, `BCa q05 = +0.0760 > 0`, 95% CI `[+0.0712, +0.1353]`) | `0.000031` (`0.000061`) |
| **C0.4** | **FAIL** | Equal-direct efficiency **PASS** (`1.777×`, 95% CI `[1.636×, 1.898×]`, `16/16` `eff_t(R_ridge) > eff_t(R_mf)` [bar `>= 12/16`]; descriptive `16/16 >= 1.25×`, `KL = 0.0098` vs `0.0167`, `-41%`), **FAIL** on cross-domain `R_mf_cross` null (`cross_mean = 1.235` vs `rand_iso F_mean = 0.299`, `191/192 > 2F_mean`, `97.6%` positive) | `0.000015` (`0.000061`) |

## Confirmed Pre-Registered Finding
- **Single-position `u_t` contraction (`C0.3` PASS):** Even at `positions="last"`, raw unembedding directions `u_t` are strictly more contracted than the 32-draw isotropic Jacobian diagonal `v^T J v` (`+0.1026`, `p = 3.1e-5`, `15/16` targets on `L18+`, `16/16` at `L27`). Real `v^T J v` (`0.694–0.970`) is `+0.2036` above ridge-shrunk `tr(R_L)/d` (`0.511–0.836`).

## Post-Hoc Mechanistic Explanations (Not Pre-Registered Findings)
1. **Cross-position attention vs. single-position MLP channel (post-hoc explanation of `C0.1` & `C0.2`):** The frozen R-lens (`neuron_lens/transport.py:5`) estimates the offset-summed Jacobian `mean_t Σ_{t' >= t} ∂h_final(t') / ∂h_L(t)` ("here and later"), matching exploratory Exp 4B (`positions="all"`, `post_norm` `WHO_CHAT_TF` `R_mf ΣAttn = +3.47/+3.81`, `logit_lens ΣAttn = -4.01/-5.96`; 4-frame pooled `L12/18/24` gap ratio `0.831`). Stage C0 injected at `positions="last"` only, isolating `∂h_final(T) / ∂h_L(T)`: `ΣAttn` dropped `8.5–18.8×` (`+0.34/+0.45` and `-0.21/-0.32`), leaving primarily the same-position MLP channel (`L24: R_mf ΣMLP = +1.58` vs `logit_lens ΣMLP = -1.66`; `L27: +2.13` vs `-1.85`; pooled `L12–L27` gap ratio `28.4%`, per-layer `L18–L27` `23.5%–47.6%`).
2. **`R_mf` common-mode amplification vs. `R_ridge` whitening (post-hoc explanation of `C0.4`; single-class noun caveat):** All 16 C0 targets are consonant-initial, space-prefixed singular concrete nouns whose raw `u_t` share mean off-diagonal cosine `+0.1826` (computed from the run's vector bank `c0_vector_bank.pt`, `sha256 = ab7e04ca…`; not stored in the four-file bundle). `R_mf = R_L^T u_t` amplifies this to `+0.2615` (`L18`) / `+0.2526` (`L24`), plausibly explaining the `+1.2257`-logit (`97.6%` positive) cross-domain bleed, whereas `R_ridge` whitens off-diagonal cosine down `2.6×` to `+0.1008 / +0.0953` (achieving `1.777×` equal-direct efficiency and `C0.1b` median `frac_err = 0.118`). Generalization beyond this single noun class and `R_ridge_cross` itself remain untested.

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
