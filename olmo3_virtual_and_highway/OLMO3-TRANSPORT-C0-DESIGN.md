# Stage C0 Design (v3): Pre-Registered Out-of-Sample Predictive Transport & Active-Erasure Benchmark on OLMo 3 7B

**Procedure:** v1  
**Draft:** v3 (2026-11-22, by Vesper `neon`, before any C0 data; **not yet registered**).  
**Review fold:** v3 folds all 4 blockers (`B1–B4`), 6 majors (`M1–M6`), and 5 minors (`m1–m5`) from cold adversarial review `C0-DESIGN-REVIEW-v2-ds-r1.md` (commit `7886320e3`), building on v2's fold of `F1–F7` and `S8–S14`, with updated frozen pre-data geometry and exact same-layer matched pairs in `olmo3_virtual_and_highway/c0_prereg_constants.json` (`sha256 = 11a6cc30a5f56d398cbad7bfd5548f29f64e2700fe3f38ed712a7a2e72d40659`).  
**Baseline cited:** Published exploratory bundle `results/vesper-olmo3-7b-exploratory/` (`assistant-basin` `main` `a105e974`; reusable 64-doc RelP lens `r_lens.pt` `ceef21fc...`).

---

## 1. Research Question

In our exploratory OLMo 3 7B study (`Exp 4B`, `post_norm`, `dose_mult = 0.75`), injecting raw Logit-Lens directions (`u_t`) yielded the largest direct unembedding write (`+8.93` at `L18`, `+14.56` at `L24`) followed by large negative downstream terms (`ΣAttn = -4.01/-5.96`, `ΣMLP = -2.39/-1.11`), whereas matched-filter R-lens directions (`R_mf = R_out^T u_t`) yielded smaller direct writes (`+3.70/+8.12`) followed by positive downstream terms (`ΣAttn = +3.47/+3.81`, `ΣMLP = +0.22/+3.65`), and `R_ridge` achieved higher `dlogit_z` (`+23.79` vs. `+17.93` on `WHO_CHAT_TF` at `L24`, `dose_mult = 0.75`).

Stage C0 tests two falsifiable, non-circular questions on **16 unseen lexical targets** evaluated across **8 held-out prompts** using our **frozen 64-doc WikiText-103 `R_out` lens** (a basis-cotangent Jacobian estimate that never saw `u_t` during fitting):
1. **Target-level out-of-sample predictive validity & quantitative calibration (`C0.1`, `C0.2`):** Holding **both layer AND arm fixed** (so an arm-label-only prior has `ρ = 0` identically), does the frozen R-lens predict target-to-target downstream transport on unseen lexical targets and achieve zero-parameter quantitative magnitude calibration across arms? And at matched write norm AND matched local direct gain (`direct_matched`), does `R_mf` beat `direct_matched` by the lens-predicted margin?
2. **Direction-specific contraction vs. isotropic Jacobian diagonal (`C0.3`, `C0.4`, `C0.5`):** Does OLMo 3 contract raw `u_t` writes *more strongly* than the isotropic Jacobian diagonal `v^T J v` (`tr(R_L)/d`), does `R_ridge` outperform `R_mf` on specificity-per-KL efficiency while `R_mf_shuffled` stays within the same-run `rand_iso` noise floor, and does the natural Concept-vs-Slot downstream difference survive at `L >= 15` across **exact same-layer matched non-pronoun pairs**?

---

## 2. Frozen Inputs, Exact Metric Definitions & Pre-Computed Constants

### Frozen Model, Lens & Splits
- **Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1` exact single-example forwards.
- **Frozen lens & hyperparameter:** `r_lens.pt` (`ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`), ridge `λ = 0.5 · tr(R_L^T R_L) / d` (identical to exploratory study; zero retuning; exact per-layer `lambda_ridge` pinned in `c0_prereg_constants.json`).
- **7 layers:** `L ∈ {6, 12, 15, 18, 21, 24, 27}`. `cos(mhough) >= 0.90` holds strictly on the **4-layer `L18+` band** (`L18 = +0.9474`, `L21 = +0.9702`, `L24 = +0.9805`, `L27 = +0.9868`; `L15 = +0.8997` sits just below `0.90`). All `C0.1–C0.4` analyses report **every layer individually**, the **full 7-layer set**, AND the `L18+` band, with the explicit monotone check `ρ_within(L6, L12) < ρ_within(L18+)`.
- **16 brand-new held-out single-token, consonant-initial targets** (pinned with exact `{domain, string, id, decoded_piece}` in `c0_prereg_constants.json`; zero overlap with the 8 exploratory personas or the 21 distinct preview-census strings):
  - **Animals (4):** `" dolphin"` (`99269`), `" hawk"` (`76920`), `" tiger"` (`52835`), `" rabbit"` (`39824`)
  - **Professions (4):** `" surgeon"` (`46098`), `" pilot"` (`18178`), `" painter"` (`30581`), `" judge"` (`11913`)
  - **Objects (4):** `" hammer"` (`24354`), `" lantern"` (`74265`), `" mirror"` (`18327`), `" compass"` (`24653`)
  - **Places (4):** `" harbor"` (`57511`), `" canyon"` (`84300`), `" temple"` (`27850`), `" castle"` (`33684`)
- **8 held-out prompts (paired 2 × 2 format × style battery; exact wording matched across Bare vs. ChatML in both Narrative and Analytical cells, M6):** Inference is over the **16 target clusters** (conditioning on these 8 prompts; per-cell format × style means are reported descriptively rather than claiming population generalization over prompt families):
  - **Bare prose — Narrative (2):** `"At the edge of the quiet village stood a"` / `"Inside the locked wooden box lay a"`
  - **Bare prose — Analytical (2):** `"Unlike the other items on the list, the final one was a"` / `"After searching for hours, the team finally found a"`
  - **ChatML — Narrative (2):** `"<|im_start|>user\nComplete this story fragment with one noun: At the edge of the quiet village stood a...<|im_end|>\n<|im_start|>assistant\nAt the edge of the quiet village stood a"` / `"<|im_start|>user\nComplete this story fragment with one noun: Inside the locked wooden box lay a...<|im_end|>\n<|im_start|>assistant\nInside the locked wooden box lay a"`
  - **ChatML — Analytical (2):** `"<|im_start|>user\nComplete this analytical sentence with one noun: Unlike the other items on the list, the final one was a...<|im_end|>\n<|im_start|>assistant\nUnlike the other items on the list, the final one was a"` / `"<|im_start|>user\nComplete this analytical sentence with one noun: After searching for hours, the team finally found a...<|im_end|>\n<|im_start|>assistant\nAfter searching for hours, the team finally found a"`
- **Injection site & position:** `post_norm` (post-`post_feedforward_layernorm` residual stream) at `positions="last"` only (exact single-position telescoping; un-gated `pre_norm` diagnostics omitted).

### Exact Metric Definitions (M5 — All Frozen Constants Computed Under These Exact Equations)
Let `u_t = W_U[t] ⊙ γ_final`, let `w` be a unit-norm write vector (`||w|| = 1`) injected with scalar `dose` at `post_norm` layer `L`, let `r_c` and `r_i` be the clean and intervened final-RMSNorm denominators (`sqrt(mean(h^2) + eps)`), and let `Δh_final = h_i - h_c` be the exact final-layer residual increment.
- **Exact 4-term telescoping decomposition:** `exact_dlogit = Direct_linear + ΣAttn + ΣMLP + rmsnorm_shift` (up to `bf16` rounding error `telescoping_err`), where:
  - `Direct_linear(w, t) = dose · (u_t · w) / r_c`
  - `rmsnorm_shift(w, t) = (u_t · h_i) · (1/r_i - 1/r_c)`
  - `frozen_direct_effect(w, t) = dose · (u_t · w) / r_i` (intervened-denominator direct term)
- **Frozen R-lens forecast (`pred_dlogit`):** `pred_dlogit(w, t) = dose · (u_t^T R_L w) / r_c`, where `R_L = R_out[L]` from `r_lens.pt`.
- **Realized downstream retention (`Retention`):** `Retention(w, t) = (exact_dlogit - rmsnorm_shift) / Direct_linear`, defined **strictly on non-null arms where `u_t · w > 0`** (`R_mf`, `R_ridge`, `logit_lens`, `direct_matched`, and `R_mf_shuffled` on its own target `t'`; set to `null` on `R_mf_shuffled(t)` and `rand_iso(t)`). Subtracting `rmsnorm_shift` removes final-layer denominator rescaling so that `Retention(logit_lens) = (u_t · Δh_final) / (dose · ||u_t||) = û_t^T J û_t` is **exact**!
- **Frozen predicted retention (`pred_ret`):** `pred_ret(w, t) = (u_t^T R_L w) / (u_t · w)`. Letting `N = ||R_L^T u_t|| / ||u_t||` and `c = cos(u_t, R_L^T u_t)`, this yields `pred_ret(R_mf) = N/c`, `pred_ret(logit_lens) = pred_ret(direct_matched) = c·N`, and `pred_ret(R_ridge) = (u_t^T R_L w_ridge) / (u_t · w_ridge)` — all frozen per `(L, t)` in `c0_prereg_constants.json`.
- **Direct unembedding comparator (`ρ_direct`):** Within-`(arm, L)` Spearman correlation between `Direct_linear(w, t)` and `exact_dlogit(w, t)` across the 16 targets.
- **`CancR` (unfiltered):** `(|Direct_linear| + Σ_l |attn_l| + Σ_l |mlp_l| + |rmsnorm_shift|) / max(|exact_dlogit|, 1e-6)`, reported unfiltered (`median`, `IQR`, `p90`, `max`, `mean`, and `ratio-of-means = mean(gross)/mean(|net|)`).

### Pre-Computed Frozen Lens Geometry (`c0_prereg_constants.json`, `0 CU` before data)
Exact medians across the 16 held-out targets (all per-target `N`, `c`, `pred_ret_Rmf`, `pred_ret_Rridge`, `pred_ret_dm`, `pred_gain_*`, `pred_gap_D3 = (1 - c^2) · 3.0 · (N/c)` frozen in JSON):

| Layer | `N` med | `c` med (`min..max`) | `pred_ret(R_mf)` | `pred_ret(R_ridge)` | `pred_ret(u_t / dm)` | `pred_gap_D3` | `tr(R_L)/d` (`v^T R_L v` med) | `tr/d - c·N` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `L6` | `0.649` | `0.209` (`0.173..0.262`) | `2.995` | `1.340` | `0.136` | `+8.555` | `+0.155` (`+0.154`) | `+0.019` |
| `L12` | `0.898` | `0.304` (`0.269..0.372`) | `2.866` | `1.358` | `0.278` | `+7.705` | `+0.314` (`+0.313`) | `+0.036` |
| `L15` | `1.024` | `0.357` (`0.314..0.437`) | `2.822` | `1.431` | `0.362` | `+7.280` | `+0.417` (`+0.418`) | `+0.055` |
| `L18` | `1.068` | `0.430` (`0.382..0.511`) | `2.479` | `1.354` | `0.462` | `+5.978` | `+0.511` (`+0.509`) | `+0.049` |
| `L21` | `1.094` | `0.498` (`0.449..0.565`) | `2.197` | `1.294` | `0.550` | `+5.016` | `+0.614` (`+0.616`) | `+0.064` |
| `L24` | `1.120` | `0.587` (`0.523..0.650`) | `1.883` | `1.214` | `0.659` | `+3.797` | `+0.723` (`+0.720`) | `+0.064` |
| `L27` | `1.155` | `0.653` (`0.583..0.713`) | `1.748` | `1.158` | `0.763` | `+3.032` | `+0.836` (`+0.835`) | `+0.074` |

Note: on `L18+`, the pooled 64-cell (`16 targets × 4 layers`) frozen median `pred_ret(u_t)` is `0.604` (while at `L27` alone `11/16` targets exceed `0.75`, max `0.847` — accounted for in `C0.2` below), and the `L18+` mean isotropic gap `tr(R_L)/d - c·N` is `+0.0626` (`+0.049..+0.074` on `L18+`, vs. `+0.019/+0.036` at `L6/L12`, m5).

---

## 3. Experimental Arms & Matched Regimes

For every `(L, target t)`:
1. **`R_mf`:** `normalize(R_L^T u_t)`
2. **`R_ridge`:** `normalize((R_L^T R_L + λI)^{-1} R_L^T u_t)`
3. **`logit_lens`:** `normalize(u_t)`
4. **`direct_matched` (`K = 4` seeded orthogonal draws per `(L, t)` averaged across draws):** `c · û_t + sqrt(1 - c^2) · z_perp` where `z_perp ⟂ {û_t, R_mf, R_ridge}`. Identical write norm AND identical local direct projection `u_t · w` as `R_mf`, and identical linear R-lens forecast (`pred_ret = c·N`, `pred_dlogit = 3cN` in Regime D) as `logit_lens` (see linear-perpendicular-null diagnostic in `C0.2`, m4).
5. **`R_mf_shuffled` (Structured cross-domain control):** `R_mf` of `t' = targets[(idx + 4) % 16]` (guaranteed different domain). Evaluated on **both**:
   - **Target `t` (specificity equivalence null, `C0.4`):** `Retention` and `CancR` set to `null`; only `Δlogit(t)` and `|Δlogit(t)|` are scored.
   - **Own target `t'` (cross-target invariance check, `C0.3b`):** `Retention(R_mf_shuffled on t')` scored against `Retention(R_mf on t)` with an explicit consistency bar (`<= 0.15` median difference on `L18+`, M4).
6. **`rand_iso` (`K = 4` seeded isotropic unit draws per `(L, t)`):**
   - On target readout `u_t`: `Retention` and `CancR` set to `null`; measures same-run noise floor `mean |Δlogit(t)|` and `p95(|Δlogit(t)|)` used to anchor `C0.4`.
   - On self-projection readout `v_self = rand_iso`: directly measures the real model's end-to-end Jacobian diagonal `v^T J v = (v · Δh_final) / dose` (and `CancR_self`) to compare against `û_t^T J û_t` (`logit_lens` Retention) and `tr(R_L)/d`.

**Two Regimes:**
- **Regime N (Equal Write Norm):** `dose_N = 0.75 · post_mlp_norm[L]` (exact match to exploratory Exp 4B) across all 7 layers and all 6 arms.
- **Regime D (Equal Local Direct Gain, gated band `L ∈ {12, 15, 18, 21, 24, 27}`; `L6` logged descriptively because `L6 R_mf` hits the dose cap):** `R_mf`, `R_ridge`, `logit_lens`, and `direct_matched` scaled so `Direct_linear = +3.0` logits (capped at `3.0 · post_mlp_norm[L]`; exploratory `L12–L27` doses are `3.4–6.6` vs. caps `11.5–36.5`, giving wide headroom, m1). Any capped trace is **dropped** and logged as missing; `cap_hit_rate <= 2%` per `(arm, L)` on `L12–L27`.

---

## 4. Exact Inference, Multiplicity Control & Pre-Registered Gates / Predictions

- **Inference & Multiplicity (`B3`):** 10,000-resample **BCa cluster bootstrap** resampling the **16 target clusters** (preserving crossed prompt & layer dependence), paired with **exact permutation / binomial p-values (`p_1..p_5`)** for **Holm-Bonferroni** familywise control across all 5 primary predictions (`C0.1–C0.5`, `α = 0.05`; smallest p-value compared to `α/5 = 0.01`, and every `p_1..p_5` has `p_min <= 0.00209 < 0.01`).

### Gate `C0.0` (All 4 clauses must pass before reading `C0.1–C0.5`)
1. **Zero-dose sham:** `dose = 0.0` yields `max |Δlogit| == 0.0` bit-for-bit across all 7 layers.
2. **Exact telescoping accounting (m2):** `max(telescoping_err) <= 0.15` logits and `p95(telescoping_err) <= 0.08` logits across all `bf16` traces (`0.08` provides a `37%` estimation margin above exploratory `L24 p95 = 0.0583` across `51×` more traces per layer: `2,432` vs. `48`; `max <= 0.15` vs. exploratory `0.0761`).
3. **Non-tautological Regime-D cap & intervened-denominator gate (m1):** On gated layers `L12–L27`, `cap_hit_rate <= 2%` for every `(arm, L)` (capped traces dropped), AND on every retained trace the intervened-denominator direct term satisfies `|frozen_direct_effect - 3.0| <= 0.25` (which fails if a write severely shifts final RMSNorm; clean-`r_c` `Direct_linear = 3.0` is satisfied by construction and logged only as a sanity check).
4. **Frozen provenance:** `r_lens.pt` SHA-256 matches `ceef21fc...` and `c0_prereg_constants.json` SHA-256 matches `11a6cc30...` in the receipt.

### Pre-Registered Predictions (`C0.1–C0.5`)

| # | Pre-Registered Prediction, Exact Aggregation & Holm p-Value (`p_1..p_5`) | Falsifier / Bar |
|---|---|---|
| **C0.1 (Within-arm target ranking & quantitative calibration — B1)** | **Primary `C0.1a` (target-level within-`(arm, L)` ranking):** In Regime N over the 4 non-null arms (`R_mf`, `R_ridge`, `logit_lens`, `direct_matched`), average `exact_dlogit` and `pred_dlogit` over the 8 prompts per `(target, arm, L)` (`n = 16` targets/cell), compute Spearman `ρ(pred_dlogit, exact_dlogit)` **strictly within each `(arm, L)` cell** (so an arm-label-only prior gives `ρ = 0` identically), and Fisher-z average across `L18+` (`4 × 4 = 16` cells) and full 7L (`28` cells). **Bar:** `L18+` Fisher-z mean `ρ_within >= +0.30` with one-sided 95% BCa CI `> 0` over 16 targets, `Δρ_within = ρ_within(R_lens) - ρ_within(direct) > 0` (one-sided 95% BCa CI `> 0` on `L18+`), AND monotone depth check `ρ_within(L6, L12) < ρ_within(L18+)`. **`p_1` (Holm):** one-sided target-label permutation p-value (`10,000` permutations holding `(arm, L)` fixed; `p_min = 1e-4`). **Paired `C0.1b` (zero-parameter magnitude calibration):** Across all `4 × 16 × 4 = 256` target-arm-layer means on `L18+`, median fractional forecast error `|exact_dlogit - pred_dlogit| / max(|pred_dlogit|, 1e-6) <= 0.30`. *(Cross-arm within-layer `ρ` reported descriptively only, noting an arm-label prior achieves `~0.81–0.86`.)* | Fails if `L18+` `ρ_within < +0.30`, `p_1` fails Holm, `Δρ_within` 95% BCa CI includes `0`, monotone check fails, or `C0.1b` median fractional error `> 0.30`. |
| **C0.2 (Lens-calibrated gap at equal direct gain — M1, m4, m5)** | In Regime D (`Direct = +3.0`) on `L12–L27` (prompt-averaged per `(target, L)`), compare realized `gap = exact_dlogit(R_mf) - exact_dlogit(direct_matched)` against frozen `pred_gap_D3 = (1 - c^2) · 3.0 · (N/c)`: (a) pooled `L12–L27` (`16 × 6 = 96` cells) median calibration ratio `gap / pred_gap_D3 >= 0.70` with one-sided 95% BCa CI `> 0.50` over 16 target clusters (exploratory analogues: `0.97` at `L18`, `1.04` at `L24`, m5); (b) **`L18+` pooled median** (`16 × 4 = 64` cells) `Retention(R_mf) >= 1.25` AND `Retention(logit_lens) <= 0.75` (frozen `L18+` median `0.604`), AND per-layer median `Retention(logit_lens) <= 0.85` at every `L ∈ L18+` (consistent with frozen `L27` `0.766`, M1). **`p_2` (Holm):** exact one-sided paired sign-flip permutation p-value over the 16 target means on `L12–L27` for `gap > 0` (`2^16 = 65,536` flips, `p_min = 1.53e-5`). **Linear-perpendicular-null diagnostic (m4):** Report `|median(exact_dlogit(dm) - exact_dlogit(ll))|` on `L18+` (frozen linear forecast = `0.0`; flag `> 0.25` logits as a nonlinear perpendicular-component departure). | Fails if median `gap / pred_gap_D3 < 0.70` (or BCa CI `<= 0.50`), `p_2` fails Holm, or Retention bounds fail on `L18+`. |
| **C0.3 (Jacobian diagonal `v^T J v` vs. `u_t`-specific contraction, heterogeneity & `V3.W2` — M2, M3, M4)** | In Regime N across all 7 layers: **(a) Primary `u_t` vs. isotropic contraction:** For each of the 16 targets, compute target-mean paired difference `d_t = mean_{L∈L18+, p∈8} [v^T J v - û_t^T J û_t]` (where `v^T J v` averages the `K=4` `rand_iso` draws and `û_t^T J û_t = Retention(logit_lens)`). **Power note (M2):** Frozen `L18+` mean gap `tr(R_L)/d - c·N = +0.0626`; exploratory `L18+L24` target-mean `Retention(logit_lens)` SD = `0.0888` (`L18` `0.134`, `L24` `0.049`) → `SE ≈ 0.0222` → `~86%` power at one-sided `α = 0.05` (`~65%` at `α/5 = 0.01`; achieved BCa CI width reported). **`p_3` (Holm):** exact one-sided paired sign-flip permutation p-value over the 16 `d_t` values (`65,536` flips, `p_min = 1.53e-5`), alongside paired 95% BCa CI `> 0` and layer sign count (`>= 5/7` layers positive reported). **(b) Non-gating diagnostics with explicit bars (M4):** (i) within-arm target heterogeneity on `L18+` (`IQR/median` + Kruskal-Wallis across the 16 targets, `α = 0.05`); (ii) `R_mf_shuffled` own-target `t'` invariance check `|median Retention(R_mf_shuffled on t') - median Retention(R_mf on t)| <= 0.15` on `L18+`; (iii) Weaver `V3.W2` unfiltered trichotomy: `median CancR_self > 3.0` on `>= 3/7` layers = **supported**, `< 1.5` on `>= 5/7` layers = **falsified**, **`[1.5, 3.0]` = unresolved (non-gating)**. **Explicit C1 gate rule (M3):** `C0.3` clears the C1 gate iff `C0.3(a)` passes (`p_3` clears Holm AND BCa 95% CI `> 0`) **AND** `L18+` pooled median `Retention(R_mf) >= 0.65 · pred_ret(R_mf)` (non-automatic calibration check that `R_mf` achieves `>= 65%` of its lens-predicted multiplier `N/c`). | Fails C1 gate if `C0.3(a)` fails Holm / BCa CI includes `0`, OR if realized `Retention(R_mf) < 0.65 · pred_ret(R_mf)` on `L18+`. |
| **C0.4 (Specificity efficiency & `rand_iso`-anchored shuffled null — B2, B3)** | In Regime N (`0.75×`) on `L18+`, define per-target efficiency `eff_t(arm) = mean_{L∈L18+, p∈8} [dlogit_z / sqrt(max(KL, 1e-6))]`. **Efficiency bar & `p_4` (Holm):** `eff_t(R_ridge) >= 1.25 × eff_t(R_mf)` on **`>= 14/16` targets** (`exact one-sided binomial p_4 = 0.00209 <= 0.01 = α/5`). Raw `KL` reported descriptively (two-sided, no lower-KL conjunct). **Same-run `rand_iso`-anchored equivalence null on `R_mf_shuffled` (target `t`, B2):** On the exact same `L18+` Regime-N traces, `mean |Δlogit(t; R_mf_shuffled)| <= mean |Δlogit(t; rand_iso)| + 0.10` (with 90% BCa upper bound on the paired difference `<= 0.15` over the 16 targets) AND `p95(|Δlogit(t; R_mf_shuffled)|) <= p95(|Δlogit(t; rand_iso)|) + 0.20` (absolute values reported alongside; exploratory `rand_iso` floor at `0.75×` is `mean |Δ| = 0.346`, `p95 = 0.844`). | Fails if `< 14/16` targets clear `1.25×` efficiency (`p_4` fails Holm), OR `R_mf_shuffled` exceeds the `rand_iso`-anchored equivalence bounds. |
| **C0.5 (`L15+` exact same-layer matched Concept vs. Non-Pronoun Slot — B4, Weaver `V3.W3`)** | Using the **13 exact same-layer matched Concept ↔ Non-Pronoun Slot pairs (`26` unique neurons)** frozen in `c0_prereg_constants.json` (`L16[×2], L17, L19, L20[×2], L21, L22, L23[×5]`; deterministically built by merging preview-21 + C0-16 non-pronoun candidates, deduplicating by unit, and pairing top-k Concept with top-k Slot within each exact shared layer `L >= 15`), compute per-unit signed downstream gain `D_down = (ΣAttn + ΣMLP) / 10.0` averaged over the 8 held-out prompts (`+10.0` SwiGLU push) and within-pair difference `Δ_i = D_down(Concept_i) - D_down(Slot_i)`. **`p_5` (Holm):** exact two-sided paired sign-flip permutation p-value over the 13 same-layer pairs (`2^13 = 8,192` flips holding layer exact, `p_min = 0.000244 <= 0.01`). **Falsifier / equivalence bar:** `|median Δ_i| < 0.02` logits/push (with 90% BCa CI reported; write-norm-normalized `D_down / ||w_down||`, 7-layer equal-weight sensitivity, and top-6 all-Slot including pronouns reported as secondary diagnostics). | Disappears above `L15` if `|median Δ_i| < 0.02` logits/push or `p_5` fails Holm. |

---

## 5. Compute Budget, Hardware Floor & Stage C1 Gate

- **Forward count (committed to `K = 4` draws throughout, m3):**
  - Regime N: `7 layers × 16 targets × 8 prompts × (4 single-draw arms + 2 × K=4 control arms = 12)` = `10,752` forwards.
  - Regime D: `7 layers × 16 targets × 8 prompts × (3 single-draw arms + 1 × K=4 direct_matched = 7)` = `6,272` forwards.
  - Natural `L15+` (`C0.5`): `32` unique units (`13` exact same-layer pairs = `26` units + `6` secondary all-Slot pronoun units) × `8` prompts = `256` forwards + `8` clean baseline + `7` sham = `271` forwards.
  - **Exact total:** **`17,295` short `B=1` forwards**.
- **Bandwidth-grounded runtime on Colab L4 (`300 GB/s` HBM, `14.0 GB` bf16 weights → `>= 47 ms/forward` bandwidth floor):** `17,295 × 50 ms ≈ 14.4 min` (`~0.44 CU` on Colab L4 at `1.85 CU/hr`). Budget estimate: **`~0.35–0.55 CU` on Colab L4** (zero lens refitting — reuses `r_lens.pt`).
- **Stdlib-only scorer:** `score_olmo3_c0.py` written in 100% Python 3.10+ stdlib (zero `numpy`/`scipy`/`torch`) for byte-identical `cmp` on any Linux VM.
- **Stage C1 Gate:** Proceed to Stage C1 (discovery/held-out path-patching of downstream difference signals + verified Base → SFT → DPO → Instruct ladder, ancestry pre-verified `cos >= +0.9977` via HTTP Range on `embed_tokens.weight`) **iff Gate `C0.0`, `C0.1`, `C0.2`, and the explicit `C0.3` C1-gate rule all pass**.
