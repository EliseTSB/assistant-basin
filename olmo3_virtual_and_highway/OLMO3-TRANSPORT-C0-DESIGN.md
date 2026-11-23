# Stage C0 Design (v4): Pre-Registered Out-of-Sample Predictive Transport & Active-Erasure Benchmark on OLMo 3 7B

**Procedure:** v1  
**Draft:** v4 (2026-11-22, by Vesper `neon`, before any C0 data; **not yet registered**).  
**Review fold:** v4 folds all blockers (`B1–B6`), majors (`M1–M8`), and minors (`m1–m8`) from `C0-DESIGN-REVIEW-v3-ds-r1.md` (commit `84dde28c5`) by **simplifying the core**: natural-neuron `C0.5` is deferred to its own dedicated layer- and token-matched design; `V3.W2` (`CancR_self`) is demoted to descriptive; `C0.1–C0.4` are tightened to per-arm/per-layer bars, Regime-D equal-direct efficiency, `M = 32` pooled isotropic draws, and all-12 cross-domain wrong-target reads (zero extra forwards). Frozen constants: `olmo3_virtual_and_highway/c0_prereg_constants.json` (`sha256 = 1e260788538c360de80a54cdc98e87ccf5849a4661e67c36be302135a09ea87f`).  
**Baseline cited:** Published exploratory bundle `results/vesper-olmo3-7b-exploratory/` (`assistant-basin` `main` `a105e974`; reusable 64-doc RelP lens `r_lens.pt` `ceef21fc...`).

---

## 1. Research Question

In our exploratory OLMo 3 7B study (`Exp 4B`, `post_norm`, `dose_mult = 0.75`), injecting raw Logit-Lens directions (`u_t`) yielded the largest direct unembedding write (`+8.93` at `L18`, `+14.56` at `L24`) followed by large negative downstream terms (`ΣAttn = -4.01/-5.96`, `ΣMLP = -2.39/-1.11`), whereas matched-filter R-lens directions (`R_mf = R_out^T u_t`) yielded smaller direct writes (`+3.70/+8.12`) followed by positive downstream terms (`ΣAttn = +3.47/+3.81`, `ΣMLP = +0.22/+3.65`), and `R_ridge` achieved higher 8-target mean `dlogit_z` (`+23.79` vs. `+17.93` on `WHO_CHAT_TF` at `L24`, `dose_mult = 0.75`).

Stage C0 tests four tightly scoped, non-circular transport predictions (`C0.1–C0.4`) on **16 unseen lexical targets** evaluated across **8 held-out prompts** using our **frozen 64-doc WikiText-103 `R_out` lens** (a basis-cotangent Jacobian estimate that never saw `u_t` during fitting):
1. **Within-arm target ranking & per-arm magnitude calibration (`C0.1`):** Holding both layer and arm fixed (so an arm-label-only prior gives `ρ = 0`), does the frozen R-lens predict target-to-target downstream transport — including for `R_mf` on its own — and calibrate realized magnitudes within `<= 30%` median error in every non-null arm individually?
2. **Per-layer lens-calibrated gap at matched direct gain (`C0.2`):** At matched write norm and matched local direct gain (`Direct_linear = +3.0`), does `R_mf` beat `direct_matched` by `>= 70%` of the frozen lens-predicted gap at every gated layer `L12–L27`?
3. **Direction-specific `u_t` contraction vs. isotropic Jacobian diagonal (`C0.3`):** On `L18+`, is `û_t^T J û_t` (`logit_lens` Retention) strictly more contracted than the 32-draw isotropic Jacobian diagonal `v^T J v`?
4. **Equal-direct specificity efficiency & all-cross-domain null (`C0.4`):** At **matched local direct gain (Regime D)** — eliminating `R_ridge`'s `1.27–1.41×` higher direct projection in Regime N — does `R_ridge` outperform `R_mf` on specificity-per-KL efficiency, while wrong-target cross-domain `R_mf(t')` writes stay within the same-run `rand_iso` noise floor?

---

## 2. Frozen Inputs, Exact Metric Definitions & Pre-Computed Constants

### Frozen Model, Lens & Splits
- **Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1` exact single-example forwards.
- **Frozen lens & hyperparameter:** `r_lens.pt` (`ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`), ridge `λ = 0.5 · tr(R_L^T R_L) / d` (identical to exploratory study; zero retuning; exact `lambda_ridge` pinned per layer in `c0_prereg_constants.json`).
- **7 layers:** `L ∈ {6, 12, 15, 18, 21, 24, 27}`. `cos(mhough) >= 0.90` holds strictly on the **4-layer `L18+` band** (`L18 = +0.9474`, `L21 = +0.9702`, `L24 = +0.9805`, `L27 = +0.9868`; `L15 = +0.8997`). Every layer is reported individually alongside the full 7-layer set and `L18+`.
- **16 unseen single-token, consonant-initial targets** (pinned with exact `{domain, string, id, decoded_piece}` in `c0_prereg_constants.json`; zero overlap with the 8 exploratory personas or the 21 preview-census strings):
  - **Animals (4):** `" dolphin"` (`99269`), `" hawk"` (`76920`), `" tiger"` (`52835`), `" rabbit"` (`39824`)
  - **Professions (4):** `" surgeon"` (`46098`), `" pilot"` (`18178`), `" painter"` (`30581`), `" judge"` (`11913`)
  - **Objects (4):** `" hammer"` (`24354`), `" lantern"` (`74265`), `" mirror"` (`18327`), `" compass"` (`24653`)
  - **Places (4):** `" harbor"` (`57511`), `" canyon"` (`84300`), `" temple"` (`27850`), `" castle"` (`33684`)
- **8 held-out prompts (paired 2 × 2 format × style battery; exact wording matched across Bare vs. ChatML in both Narrative and Analytical cells):** Inference is over the **16 target clusters** conditioned on these 8 prompts (per-cell means reported descriptively):
  - **Bare — Narrative (2):** `"At the edge of the quiet village stood a"` / `"Inside the locked wooden box lay a"`
  - **Bare — Analytical (2):** `"Unlike the other items on the list, the final one was a"` / `"After searching for hours, the team finally found a"`
  - **ChatML — Narrative (2):** `"<|im_start|>user\nComplete this story fragment with one noun: At the edge of the quiet village stood a...<|im_end|>\n<|im_start|>assistant\nAt the edge of the quiet village stood a"` / `"<|im_start|>user\nComplete this story fragment with one noun: Inside the locked wooden box lay a...<|im_end|>\n<|im_start|>assistant\nInside the locked wooden box lay a"`
  - **ChatML — Analytical (2):** `"<|im_start|>user\nComplete this analytical sentence with one noun: Unlike the other items on the list, the final one was a...<|im_end|>\n<|im_start|>assistant\nUnlike the other items on the list, the final one was a"` / `"<|im_start|>user\nComplete this analytical sentence with one noun: After searching for hours, the team finally found a...<|im_end|>\n<|im_start|>assistant\nAfter searching for hours, the team finally found a"`
- **Injection site & position:** `post_norm` (post-`post_feedforward_layernorm` residual stream) at `positions="last"` only.

### Exact Metric Definitions
Let `u_t = W_U[t] ⊙ γ_final`, let `w` be a unit-norm write (`||w|| = 1`) injected with scalar `dose` at `post_norm` layer `L`, let `r_c` and `r_i` be the clean and intervened final-RMSNorm denominators, and let `Δh_final = h_i - h_c`.
- **Exact 4-term telescoping decomposition:** `exact_dlogit = Direct_linear + ΣAttn + ΣMLP + rmsnorm_shift` (up to `bf16` `telescoping_err`), with `Direct_linear(w, t) = dose · (u_t · w) / r_c`, `rmsnorm_shift(w, t) = (u_t · h_i) · (1/r_i - 1/r_c)`, and `frozen_direct_effect(w, t) = dose · (u_t · w) / r_i`.
- **Frozen R-lens forecast:** `pred_dlogit(w, t) = dose · (u_t^T R_L w) / r_c`.
- **Realized downstream retention:** `Retention(w, t) = (exact_dlogit - rmsnorm_shift) / Direct_linear`, defined on the 4 non-null arms (`R_mf`, `R_ridge`, `logit_lens`, `direct_matched`; set to `null` on `rand_iso(t)` and wrong-target `R_mf(t')` on `t`). **Pre-data positivity guarantee (`M7`):** Because `w` and `u_t` are frozen pre-data, across all `7 × 16 = 112` `(L, t)` cells in `c0_prereg_constants.json`, `min(c) = 0.1730 > 0` and `min(cos_u_Rridge) = 0.2836 > 0`, so `u_t · w > 0` holds strictly for `112/112` cells with **zero outcome-dependent missingness possible**. Subtracting `rmsnorm_shift` makes `Retention(logit_lens) = (u_t · Δh_final) / (dose · ||u_t||) = û_t^T J û_t` exact.
- **Frozen predicted retention:** `pred_ret(w, t) = (u_t^T R_L w) / (u_t · w)` (`N/c` for `R_mf`, `pred_ret_Rridge` for `R_ridge`, `c·N` for `logit_lens` and `direct_matched`, where `N = ||R_L^T u_t|| / ||u_t||`, `c = cos(u_t, R_L^T u_t)`).
- **Direct unembedding comparator (`ρ_direct`):** Within-`(arm, L)` Spearman correlation between `Direct_linear(w, t)` and `exact_dlogit(w, t)` across the 16 targets, Fisher-z averaged over the same cells as `ρ_within`.
- **Standard median (`m1`):** Every median in `c0_prereg_constants.json` and the scorer uses Python `statistics.median` (average of the 8th and 9th order statistics for `n = 16`).
- **`CancR` (descriptive, unfiltered):** `(|Direct_linear| + Σ_l |attn_l| + Σ_l |mlp_l| + |rmsnorm_shift|) / max(|exact_dlogit|, 1e-6)`, reported unfiltered (`median`, `IQR`, `p90`, `max`, `mean`, and `ratio-of-means`).

### Pre-Computed Frozen Lens Geometry (`c0_prereg_constants.json`, `0 CU` before data)
Exact **standard medians** across the 16 unseen targets (all per-target `N`, `c`, `pred_ret_*`, `cos_u_Rridge`, `pred_gain_*`, `pred_gap_D3 = (1 - c^2) · 3.0 · (N/c)` frozen in JSON):

| Layer | `N` med | `c` med (`min..max`) | `pred_ret(R_mf)` | `pred_ret(R_ridge)` | `pred_ret(u_t / dm)` | `pred_gap_D3` | `tr(R_L)/d` (`v^T R_L v` med) | `tr/d - c·N` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `L6` | `0.651` | `0.214` (`0.173..0.262`) | `3.024` | `1.340` | `0.139` | `+8.675` | `+0.155` (`+0.154`) | `+0.016` |
| `L12` | `0.900` | `0.304` (`0.269..0.372`) | `2.868` | `1.358` | `0.280` | `+7.769` | `+0.314` (`+0.313`) | `+0.034` |
| `L15` | `1.028` | `0.357` (`0.314..0.437`) | `2.833` | `1.431` | `0.370` | `+7.407` | `+0.417` (`+0.418`) | `+0.047` |
| `L18` | `1.074` | `0.432` (`0.382..0.511`) | `2.482` | `1.354` | `0.465` | `+6.072` | `+0.511` (`+0.509`) | `+0.046` |
| `L21` | `1.100` | `0.500` (`0.449..0.565`) | `2.197` | `1.294` | `0.553` | `+5.044` | `+0.614` (`+0.616`) | `+0.061` |
| `L24` | `1.124` | `0.587` (`0.523..0.650`) | `1.893` | `1.214` | `0.661` | `+3.817` | `+0.723` (`+0.720`) | `+0.062` |
| `L27` | `1.157` | `0.654` (`0.583..0.713`) | `1.758` | `1.158` | `0.766` | `+3.036` | `+0.836` (`+0.835`) | `+0.070` |

Note: on `L18+`, the pooled 64-cell standard median `pred_ret(u_t)` is `0.604` (`pred_ret(R_mf)` is `2.031`), at `L27` alone `11/16` targets exceed `0.75` (max `0.847`), and the frozen per-target `L18+` mean isotropic gap `tr(R_L)/d - c·N_t` has `mean = +0.0618`, `SD = 0.0520` (`min = -0.0285`, `max = +0.1543`).

---

## 3. Experimental Arms & Matched Regimes

For every layer `L ∈ {6, 12, 15, 18, 21, 24, 27}`:
1. **`R_mf(t)`** (`16` targets): `normalize(R_L^T u_t)`. Each forward pass outputs the full `[V]` logit vector; in addition to scoring target `t`, we read **`R_mf_cross(t)`** directly from the **12 cross-domain wrong-target runs `R_mf(t')`** (`domain(t') != domain(t)`) at zero extra forward cost (eliminating any single-cyclic-shift artifact, `M3`; the 3 same-domain wrong targets are also logged descriptively to expose semantic/anisotropy bleed).
2. **`R_ridge(t)`** (`16` targets): `normalize((R_L^T R_L + λI)^{-1} R_L^T u_t)`.
3. **`logit_lens(t)`** (`16` targets): `normalize(u_t)`.
4. **`direct_matched(t)`** (`16` targets × `K = 4` seeded orthogonal draws averaged across draws): `c · û_t + sqrt(1 - c^2) · z_perp` where `z_perp ⟂ {û_t, R_mf, R_ridge}`. Identical write norm and identical `Direct_linear` as `R_mf`, and identical linear R-lens forecast (`3cN` in Regime D) as `logit_lens`.
5. **`rand_iso`** (**`M = 32` seeded isotropic unit draws per layer `L`**, shared across all 16 targets, `M2` — matching the 32-draw `vTRv` baseline in `c0_prereg_constants.json` with 8× lower variance than `K = 4`):
   - On all 16 target readouts `u_t`: `Retention` and `CancR` set to `null`; measures the same-run noise floor `mean |Δlogit(t)|` and `p95(|Δlogit(t)|)` on Regime-N traces.
   - On each draw's own self-projection `v_self = rand_iso`: directly measures the real model's end-to-end Jacobian diagonal `v^T J v = (v · Δh_final) / dose` (`CancR_self` logged purely descriptively, `M4`).

**Two Regimes:**
- **Regime N (Equal Write Norm):** `dose_N = 0.75 · post_mlp_norm[L]` across all 7 layers for arms `1–5`.
- **Regime D (Equal Local Direct Gain, gated band `L ∈ {12, 15, 18, 21, 24, 27}`; `L6` descriptive):** `R_mf`, `R_ridge`, `logit_lens`, and `direct_matched` scaled so `Direct_linear = +3.0` logits (capped at `3.0 · post_mlp_norm[L]`; exploratory `L12–L27` doses are `3.4–6.6` vs. caps `11.5–36.5`). Capped traces are **dropped**; `cap_hit_rate <= 2%` per `(arm, L)` on `L12–L27`.

---

## 4. Exact Inference, Multiplicity Control & Pre-Registered Gates / Predictions

- **Inference & 4-Test Holm-Bonferroni Family (`α = 0.05`, `α/4 = 0.0125`):** 10,000-resample **BCa cluster bootstrap** resampling the **16 target clusters** (preserving crossed prompt & layer dependence). Exact permutation/binomial p-values `p_1..p_4` control familywise Type-I error on the directional existence nulls across the 4 primary predictions (`C0.1–C0.4`), while the pre-registered magnitude/calibration thresholds (`ρ >= 0.30`, `frac_err <= 0.30`, `ratio >= 0.70`, `14/16`) act as minimum-effect gates (`M5`). Every `p_1..p_4` has `p_min <= 0.00209 < 0.0125`.

### Gate `C0.0` (All 4 clauses must pass before reading `C0.1–C0.4`)
1. **Zero-dose sham:** `dose = 0.0` yields `max |Δlogit| == 0.0` bit-for-bit across all 7 layers.
2. **Quantile telescoping accounting (`M8`):** Across all `bf16` traces, `p95(telescoping_err) <= 0.08` logits AND `p99.9(telescoping_err) <= 0.15` logits (sample-size-invariant quantiles grounded in exploratory `post_norm` `n = 192`: `median = 0.0195`, `p95 = 0.0563`, `max = 0.0761`; overall `max` logged descriptively).
3. **Non-tautological Regime-D cap & intervened-denominator gate (`M8`):** On gated layers `L12–L27`, `cap_hit_rate <= 2%` per `(arm, L)` (capped traces dropped), AND across retained traces `p95(|frozen_direct_effect - 3.0|) <= 0.20` logits AND `p99.9(|frozen_direct_effect - 3.0|) <= 0.30` logits (grounded in exploratory `post_norm` `n = 192` `|r_c/r_i - 1|`: `median = 0.0157`, `p95 = 0.0371` [`0.111` logits at `Direct = 3.0`], `max = 0.0518` [`0.155` logits]; clean-`r_c` `Direct_linear = 3.0` logged as sanity check).
4. **Frozen provenance:** `r_lens.pt` SHA-256 matches `ceef21fc...` and `c0_prereg_constants.json` SHA-256 matches `1e260788...` in the receipt.

### Pre-Registered Predictions (`C0.1–C0.4`)

| # | Pre-Registered Prediction, Exact Aggregation & Holm p-Value (`p_1..p_4`) | Falsifier / Bar |
|---|---|---|
| **C0.1 (Within-arm target ranking & per-arm magnitude calibration — B5, B6, M1, m7)** | **Primary `C0.1a` (target-level within-`(arm, L)` ranking):** In Regime N over the 4 non-null arms (`R_mf`, `R_ridge`, `logit_lens`, `direct_matched`), average `exact_dlogit`, `pred_dlogit`, and `Direct_linear` over the 8 prompts per `(target, arm, L)` (`n = 16` targets/cell), compute Spearman `ρ(pred_dlogit, exact_dlogit)` **strictly within each `(arm, L)` cell** (arm-label prior gives `ρ = 0` identically), and Fisher-z average (noting Pearson-style `atanh` approximation to Spearman SE `≈ sqrt(1.06/(n-3))`, `m7`). **Bars:** (i) `L18+` 16-cell Fisher-z mean `ρ_within >= +0.30` (simulated power `~48%` at true `ρ = 0.30`, `~94.5%` at `ρ = 0.40`) **AND** `R_mf` alone (`4` `L18+` layers) `ρ_within(R_mf) >= +0.30` (`B5`); (ii) `Δρ_within = ρ_within - ρ_direct > 0` on `L18+` with one-sided 95% BCa CI `> 0` (`M1`, using `ρ_direct` defined in §2); (iii) monotone check `ρ_within(L6, L12) < ρ_within(L18+)`. **`p_1` (Holm):** one-sided **joint** target-label permutation p-value (`10,000` permutations permuting the 16 target labels identically across all 16 `L18+` cells to preserve cross-arm/layer correlation, `p_min = 1e-4`, `M1`) for `ρ_within > 0`. **Paired `C0.1b` (per-arm calibration, `B6`):** On `L18+`, median fractional error `|exact_dlogit - pred_dlogit| / max(|pred_dlogit|, 1e-6) <= 0.30` **in EVERY ONE of the 4 non-null arms individually** (`4` arm medians, each `64` target-layer cells). *(Cross-arm within-layer `ρ` reported descriptively only.)* | Fails if `L18+` `ρ_within < +0.30`, `ρ_within(R_mf) < +0.30`, `p_1` fails Holm, `Δρ_within` BCa CI includes `0`, monotone check fails, OR any of the 4 arms exceeds `0.30` median fractional error in `C0.1b`. |
| **C0.2 (Per-layer lens-calibrated gap at equal direct gain — B6, M1, m1, m2, m4)** | In Regime D (`Direct_linear = +3.0`) on `L12–L27` (prompt-averaged per `(target, L)`), compare realized `gap = exact_dlogit(R_mf) - exact_dlogit(direct_matched)` against frozen `pred_gap_D3`: **(a) Per-layer calibration (`B6`):** `median(gap / pred_gap_D3) >= 0.70` **at EVERY ONE of the 6 gated layers `L12–L27` individually** (`6` layer medians, each `16` targets), AND overall `L12–L27` one-sided 95% BCa CI `> 0.50` over the 16 target clusters. **(b) Retention bounds (`M1`, `m2`):** `L18+` pooled 64-cell median `Retention(R_mf) >= 1.25` AND `Retention(logit_lens) <= 0.75` (frozen `0.604`), AND per-layer median `Retention(logit_lens) <= 0.85` at every `L ∈ L18+` (frozen `L27 = 0.766`). **`p_2` (Holm):** exact one-sided paired sign-flip permutation p-value over the 16 target-mean `L12–L27` gaps for `gap > 0` (`2^16 = 65,536` flips, `p_min = 1.53e-5`). **Linear-perpendicular-null diagnostic (`m4`, non-gating):** Report `|median(exact_dlogit(dm) - exact_dlogit(ll))|` on `L18+` (frozen linear forecast = `0.0`; flag `> 0.25` logits as nonlinear perpendicular departure). | Fails if any layer in `L12–L27` has median `gap / pred_gap_D3 < 0.70`, overall BCa CI `<= 0.50`, `p_2` fails Holm, or `L18+` Retention bounds fail. |
| **C0.3 (`u_t`-specific contraction vs. 32-draw isotropic Jacobian diagonal — M2, M4, M5)** | In Regime N across all 7 layers: for each layer `L`, compute the shared 32-draw isotropic baseline `vJv_L = mean_{m∈32, p∈8} [v_m^T J v_m]` (`M = 32` draws eliminates `K = 4` estimator noise, `M2`). For each target `t ∈ {1..16}`, compute `d_t = mean_{L∈L18+} [vJv_L - mean_{p∈8} Retention(logit_lens; L, t, p)]`. **Pre-data power (`M2`, exact noncentral-t `df = 15`):** Under frozen per-target `L18+` gap (`mean = +0.0618`, `SD = 0.0520`, `SE = 0.0130`), power is **`99.8%`** at one-sided `α = 0.05` and **`98.2%`** at Holm `α/4 = 0.0125`; conservatively substituting exploratory `L18+L24` target-mean `Retention(logit_lens)` `SD = 0.0888` (`SE = 0.0222`), power is **`84.4%`** at `α = 0.05` and **`62.1%`** at `α/4 = 0.0125` (achieved BCa CI width reported). **`p_3` (Holm):** exact one-sided paired sign-flip permutation p-value over the 16 `d_t` values (`65,536` flips, `p_min = 1.53e-5`), alongside one-sided 95% BCa CI `> 0` and per-layer sign count (`L6–L27` reported). *(Non-gating descriptives: within-arm `Retention` `IQR/median` + Kruskal-Wallis across 16 targets; `CancR_self` unfiltered with ratio-of-means, `M4`.)* | Fails if `p_3` fails Holm OR one-sided 95% BCa CI of `mean(d_t)` includes `0`. |
| **C0.4 (Equal-direct specificity efficiency & all-cross-domain null — B4, M3, m3)** | **Efficiency in Regime D (`Direct_linear = +3.0`, `B4`, `m3`):** Eliminating `R_ridge`'s `1.27–1.41×` Regime-N direct-gain advantage, define per-target non-negative efficiency on `L18+` as `eff_t(arm) = mean_{L∈L18+, p∈8} [max(dlogit_z, 0) / sqrt(max(KL, 1e-6))]` (signed `dlogit_z` and raw `KL` reported descriptively). **Bar & `p_4` (Holm):** `eff_t(R_ridge) >= 1.25 × eff_t(R_mf)` on **`>= 14/16` targets** (`exact one-sided binomial p_4 = 0.00209 <= 0.0125 = α/4`). **All-cross-domain equivalence null in Regime N (`0.75×`, `L18+`, `M3`):** Reading all **12 cross-domain wrong targets `t'`** for every target `t` directly from `R_mf`'s own `[V]` logits (no cyclic shift; no pilot `R_mf_cross` measurement exists — within-domain wrong targets `3/t` also logged descriptively), compare against the 32 same-layer `rand_iso` draws on the same traces: `mean |Δlogit(t; R_mf_cross)| <= mean |Δlogit(t; rand_iso)| + 0.10` (90% BCa upper bound on the target-paired difference `<= 0.15`) AND `p95(|Δlogit(t; R_mf_cross)|) <= p95(|Δlogit(t; rand_iso)|) + 0.20` (exploratory `rand_iso` floor: `0.346 / 0.844`). | Fails if `< 14/16` targets clear `1.25×` Regime-D efficiency (`p_4` fails Holm), OR `R_mf_cross` exceeds the `rand_iso`-anchored equivalence bounds. |

---

## 5. Compute Budget, Hardware Floor & Stage C1 Gate

- **Forward count (all clean & sham baselines explicitly shared across Regime N and Regime D, `m4`):**
  - Clean baselines (`8` prompts) + zero-dose shams (`7` layers): `15` forwards.
  - Regime N (`0.75×`): `7 × 16 × 8 × 3` (`R_mf`, `R_ridge`, `logit_lens`; `R_mf_cross` read free from `R_mf`'s `[V]` logits) `= 2,688` + `7 × 16 × 8 × 4` (`direct_matched`, `K = 4`) `= 3,584` + `7 × 32 × 8` (`rand_iso`, `M = 32` pooled draws/layer) `= 1,792` → `8,064` forwards.
  - Regime D (`Direct = +3.0`): `7 × 16 × 8 × 3` (`R_mf`, `R_ridge`, `logit_lens`) `= 2,688` + `7 × 16 × 8 × 4` (`direct_matched`, `K = 4`) `= 3,584` → `6,272` forwards.
  - **Exact total:** **`14,351` short `B=1` forwards** (17% fewer than v3, with 8× more isotropic draws and all-12 cross-domain wrong targets!).
- **Bandwidth-grounded runtime on Colab L4 (`300 GB/s` HBM, `14.0 GB` bf16 weights → `>= 47 ms/forward` floor):** `14,351 × 50 ms ≈ 12.0 min` (`~0.37 CU` at `1.85 CU/hr`; budget estimate **`~0.35–0.50 CU`** allowing shared-L4 overhead; zero lens refitting).
- **Stdlib-only scorer:** `score_olmo3_c0.py` written in 100% Python 3.10+ stdlib for byte-identical `cmp` on any Linux VM.
- **Stage C1 Gate (`M5`):** Proceed to Stage C1 (discovery/held-out path-patching of downstream difference signals + verified Base → SFT → DPO → Instruct ladder, `embed_tokens` `cos >= +0.9977`) **iff Gate `C0.0` AND the transport-core predictions `C0.1`, `C0.2`, and `C0.3` all pass** (no redundant duplicate clause in `C0.3`; `C0.4` is the 4th Holm member testing `R_ridge` specificity efficiency and is reported alongside without blocking C1's causal highway follow-up; deferred natural-neuron `C0.5` will be designed separately with matched layer + readout token + source list).
