# Stage C0 Design (v2): Pre-Registered Out-of-Sample Predictive Transport & Active-Erasure Benchmark on OLMo 3 7B

**Procedure:** v1  
**Draft:** v2 (2026-11-22, by Vesper `neon`, before any C0 data; **not yet registered**).  
**Review fold:** v2 folds all 7 mandatory fixes (`F1–F7`) and `S8–S14` from cold adversarial review `C0-DESIGN-REVIEW-ds-r1.md` (`Journal/notes/reports/vesper-olmo3-coldcheck-2026-11-22/C0-DESIGN-REVIEW-ds-r1.md`), plus pre-computed frozen geometry constants in `olmo3_virtual_and_highway/c0_prereg_constants.json`.  
**Baseline cited:** Exploratory bundle `results/vesper-olmo3-7b-exploratory/` (`OLMO3-VIRTUAL-NEURONS-AND-HIGHWAY-REPORT.md`, reusable 64-doc RelP lens `r_lens.pt` `ceef21fc...`).

---

## 1. Research Question

In our exploratory OLMo 3 7B study (`Exp 4B`, `post_norm`, `dose_mult = 0.75`), injecting raw Logit-Lens directions (`u_t`) yielded the largest direct unembedding write (`+8.93` at `L18`, `+14.56` at `L24`) followed by large negative downstream terms (`ΣAttn = -4.01/-5.96`, `ΣMLP = -2.39/-1.11`), whereas matched-filter R-lens directions (`R_mf = R_out^T u_t`) yielded smaller direct writes (`+3.70/+8.12`) followed by positive downstream terms (`ΣAttn = +3.47/+3.81`, `ΣMLP = +0.22/+3.65`), and `R_ridge` achieved higher `dlogit_z` (`+23.79` vs. `+17.93` at `L24`, `dose_mult = 0.75`).

Stage C0 tests two falsifiable, non-circular questions on **unseen lexical targets** and **unseen context families** using our **frozen 64-doc WikiText-103 `R_out` lens** (which is a basis-cotangent Jacobian estimate that never saw `u_t` during fitting):
1. **Within-layer out-of-sample predictive validity (`C0.1`, `C0.2`):** At fixed layer (eliminating between-layer dose/depth Simpson's paradox), does the frozen R-lens predict realized downstream transport better than direct unembedding (`u_t`), and does its preferred direction (`R_mf`) beat a direction matched in *both* write norm and exact local direct target-logit gain (`direct_matched`) by the lens-predicted margin?
2. **Direction-specific erasure vs. generic residual contraction (`C0.3`, `C0.4`, `C0.5`):** Does OLMo 3 contract raw `u_t` writes *more strongly* than the isotropic Jacobian diagonal `tr(R_L)/d` (and show structured target-dependent transport rather than uniform erasure), does `R_ridge` outperform `R_mf` on specificity-per-KL efficiency, and does the natural Concept-vs-Slot downstream difference survive strictly above `L15` once first-person pronouns are removed?

---

## 2. Frozen Inputs, Pre-Computed Constants & Orthogonal Held-Out Splits

- **Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1` exact single-example forwards.
- **Frozen lens & hyperparameter:** `r_lens.pt` (`ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`), ridge `λ = 0.5 · tr(R_L^T R_L) / d` (identical to exploratory study; zero retuning).
- **7 layers:** `L ∈ {6, 12, 15, 18, 21, 24, 27}`. Note: `cos(mhough) >= 0.90` holds strictly on the **4-layer `L18+` band** (`L18 = +0.9474`, `L21 = +0.9702`, `L24 = +0.9805`, `L27 = +0.9868`; `L15 = +0.8997` sits just below `0.90`). All `C0.1`–`C0.4` tests report **every layer individually**, the **full 7-layer set** (preventing favorable-layer selection, S10), AND the `L18+` band, with the explicit monotone prediction `ρ(L6, L12) < ρ(L18+)`.
- **16 brand-new held-out single-token, consonant-initial targets** (pinned with exact `{domain, string, id, decoded_piece}` in `olmo3_virtual_and_highway/c0_prereg_constants.json`; zero overlap with the 8 exploratory personas or the 21 distinct target strings in `natural_neuron_census_preview.json`):
  - **Animals (4):** `" dolphin"` (`99269`), `" hawk"` (`76920`), `" tiger"` (`52835`), `" rabbit"` (`39824`)
  - **Professions (4):** `" surgeon"` (`46098`), `" pilot"` (`18178`), `" painter"` (`30581`), `" judge"` (`11913`)
  - **Objects (4):** `" hammer"` (`24354`), `" lantern"` (`74265`), `" mirror"` (`18327`), `" compass"` (`24653`)
  - **Places (4):** `" harbor"` (`57511`), `" canyon"` (`84300`), `" temple"` (`27850`), `" castle"` (`33684`)
- **Balanced 2 × 2 held-out prompt battery (`8` prompts total, zero overlap with exploratory frames; removes the F3 chat-format confound, S11):**
  - **Bare prose — Narrative (2):** `"At the edge of the quiet village stood a"` / `"Inside the locked wooden box lay a"`
  - **Bare prose — Analytical (2):** `"Unlike the other items on the list, the final one was a"` / `"After searching for hours, the team finally found a"`
  - **ChatML — Narrative (2):** `"<|im_start|>user\nComplete this story fragment with one noun: At the edge of the quiet village stood a...<|im_end|>\n<|im_start|>assistant\nAt the edge of the quiet village stood a"` / `"<|im_start|>user\nComplete this story fragment with one noun: Inside the locked wooden box lay a...<|im_end|>\n<|im_start|>assistant\nInside the locked wooden box lay a"`
  - **ChatML — Analytical (2):** `"<|im_start|>user\nGive one concrete noun as an example.<|im_end|>\n<|im_start|>assistant\nOne clear example is a"` / `"<|im_start|>user\nName a single thing you might picture in a story.<|im_end|>\n<|im_start|>assistant\nYou might picture a"`
- **Injection site & position:** `post_norm` (post-`post_feedforward_layernorm` residual stream) at `positions="last"` only (exact single-position telescoping). Un-gated `pre_norm` diagnostics are omitted from C0 (already established in the exploratory baseline).

### Pre-Computed Frozen Lens Geometry (`c0_prereg_constants.json`, `0 CU` before data)
For each `(L, target)`, let `N = ||R_L^T u_t|| / ||u_t||` and `c = cos(u_t, R_L^T u_t)`. Under the frozen R-lens, `pred_ret(R_mf) = N/c`, `pred_ret(logit_lens) = pred_ret(direct_matched) = c·N`, and the Regime-D (`Direct = +3.0`) lens-predicted gap is `pred_gap_D3 = (1 - c^2) · 3.0 · (N / c)`. Exact medians across the 16 held-out targets:

| Layer | `N` med | `c` med (`min..max`) | `pred_ret(R_mf)` | `pred_ret(u_t / dm)` | `pred_gap_D3` | `tr(R_L)/d` (`v^T R_L v` med) |
|---:|---:|---|---:|---:|---:|---:|
| `L6` | `0.649` | `0.209` (`0.173..0.262`) | `2.995` | `0.136` | `+8.555` | `+0.155` (`+0.154`) |
| `L12` | `0.898` | `0.304` (`0.269..0.372`) | `2.866` | `0.278` | `+7.705` | `+0.314` (`+0.313`) |
| `L15` | `1.024` | `0.357` (`0.314..0.437`) | `2.822` | `0.362` | `+7.280` | `+0.417` (`+0.418`) |
| `L18` | `1.068` | `0.430` (`0.382..0.511`) | `2.479` | `0.462` | `+5.978` | `+0.511` (`+0.509`) |
| `L21` | `1.094` | `0.498` (`0.449..0.565`) | `2.197` | `0.550` | `+5.016` | `+0.614` (`+0.616`) |
| `L24` | `1.120` | `0.587` (`0.523..0.650`) | `1.883` | `0.659` | `+3.797` | `+0.723` (`+0.720`) |
| `L27` | `1.155` | `0.653` (`0.583..0.713`) | `1.748` | `0.763` | `+3.032` | `+0.836` (`+0.835`) |

Notice `c` never exceeds `0.713` even at `L27`, so `pred_gap_D3` is well-separated (`+3.03` to `+8.56` logits) everywhere, AND `pred_ret(u_t) = c·N` sits `~9%–13%` *below* the isotropic Jacobian diagonal `tr(R_L)/d` at every layer `L6–L27`.

---

## 3. Experimental Arms & Matched Regimes

For every `(L, target t)`:
1. **`R_mf`:** `normalize(R_L^T u_t)`
2. **`R_ridge`:** `normalize((R_L^T R_L + λI)^{-1} R_L^T u_t)`
3. **`logit_lens`:** `normalize(u_t)`
4. **`direct_matched` (`K = 4` seeded orthogonal draws per `(L, t)` averaged for direction-sampling variance, S14):** `c · û_t + sqrt(1 - c^2) · z_perp` where `z_perp ⟂ {û_t, R_mf, R_ridge}`. Identical write norm AND identical local direct projection `u_t · w` as `R_mf`.
5. **`R_mf_shuffled` (Structured cross-domain control):** `R_mf` of `t' = targets[(idx + 4) % 16]` (guaranteed different domain). Evaluated on **both**:
   - **Target `t` (specificity null, `C0.4`):** `Retention` and `CancR` set to `null` (not computed on near-zero denominator, S13); only `Δlogit(t)` and `|Δlogit(t)|` are scored.
   - **Own target `t'` (structured cross-target control, `C0.3`):** Verifies cross-target transport consistency alongside `R_mf(t)`.
6. **`rand_iso` (`K = 4` seeded isotropic unit draws per `(L, t)`):**
   - On target readout `u_t`: `Retention` and `CancR` set to `null` (S13); records empirical noise floor `mean |Δlogit(t)|` and `p95(|Δlogit(t)|)`.
   - On self-projection readout `v_self = rand_iso`: directly measures the real model's end-to-end Jacobian diagonal `v^T J v = (v · Δh_final) / dose` (explicitly NOT labelled "passive RMSNorm decay", S4) to compare directly against `û_t^T J û_t` (`logit_lens` Retention) and `tr(R_L)/d`.

**Two Regimes:**
- **Regime N (Equal Write Norm):** `dose_N = 0.75 · post_mlp_norm[L]` (exact match to exploratory Exp 4B) across all 7 layers and all 6 arms.
- **Regime D (Equal Local Direct Gain, gated band `L ∈ {12, 15, 18, 21, 24, 27}`; `L6` logged descriptively because `L6 R_mf` hits the dose cap):** `R_mf`, `R_ridge`, `logit_lens`, and `direct_matched` scaled so `Direct_linear = +3.0` logits (capped at `3.0 · post_mlp_norm[L]`). **Capped-trace disposition (F7):** any capped trace is **dropped** and logged as missing; `cap_hit_rate` is gated at `<= 2%` per `(arm, L)` on `L12–L27`.

---

## 4. Exact Metrics, Statistics & Pre-Registered Gates / Predictions

- **Exact 4-term telescoping decomposition** (`exact_dlogit = Direct + ΣAttn + ΣMLP + RMSsh`), reporting `telescoping_err` (`median`, `p95`, `max`).
- **`CancR` reported UNFILTERED (S9):** full distribution (`median`, `IQR`, `p90`, `max`) plus **both** `mean(CancR)` and `ratio-of-means` (`mean(gross) / mean(|net|)`). Secondary `|exact| >= 0.25` filtered median reported only alongside per-arm drop rates.
- **Inference (S11):** 10,000-resample **BCa cluster bootstrap** resampling the **16 target clusters** (preserving crossed prompt & layer dependence); exact one-sided/two-sided tails stated per prediction; **Holm-Bonferroni** familywise control reported across the 5 primary predictions (`C0.1`–`C0.5`, `α = 0.05`).

### Gate `C0.0` (All 4 clauses must pass before reading `C0.1`–`C0.5`)
1. **Zero-dose sham:** `dose = 0.0` yields `max |Δlogit| == 0.0` bit-for-bit across all 7 layers.
2. **Exact telescoping accounting:** `max(telescoping_err) <= 0.15` logits and `p95(telescoping_err) <= 0.08` logits across all `bf16` traces (`0.08` reflects observed `0.058` at L24 across 48 traces scaled to 21× more traces, S12).
3. **Non-tautological Regime-D calibration & cap gate (F7):** On gated layers `L12–L27`, `cap_hit_rate <= 2%` for every `(arm, L)`; capped traces are dropped; AND for every retained trace, both clean-`r_c` `|Direct_linear - 3.0| <= 0.05` AND intervened-`r_i` `|frozen_direct_effect - 3.0| <= 0.25` (which can fail if a write severely perturbs final RMSNorm!).
4. **Frozen provenance:** `r_lens.pt` SHA-256 matches `ceef21fc...` and `c0_prereg_constants.json` SHA-256 is pinned in the receipt.

### Pre-Registered Predictions (`C0.1`–`C0.5`)

| # | Pre-Registered Prediction & Exact Test | Falsifier / Bar |
|---|---|---|
| **C0.1 (Within-layer out-of-sample ranking — F1, F2, S10)** | Excluding `R_mf_shuffled` (non-null pool: `R_mf`, `R_ridge`, `logit_lens`, `direct_matched` in Regime N), compute Spearman `ρ` **strictly within each layer `L`** between frozen R-lens forecast `pred_dlogit` and realized `exact_dlogit`, averaged across layers via Fisher-z: (a) on the **full 7-layer set**, Fisher-z mean `ρ_R >= 0.70` AND `Δρ = ρ_R - ρ_direct >= +0.50` (one-sided 95% BCa CI > 0 over 16 targets); (b) on **`L18+`** (`cos(mhough) >= 0.90`), Fisher-z mean `ρ_R >= 0.80`; (c) monotone depth prediction `ρ_R(L6, L12) < ρ_R(L18+)`. *(Pooled `ρ` reported descriptively only.)* | Fails if full-7L `ρ_R < 0.70`, `L18+` `ρ_R < 0.80`, `Δρ` BCa 95% CI includes `0`, or monotone check fails. |
| **C0.2 (Quantitative lens-calibrated gap at equal direct gain — F4)** | In Regime D (`Direct = +3.0`) on `L12–L27`, compare realized `gap = exact_dlogit(R_mf) - exact_dlogit(direct_matched)` against each trace's pre-registered lens-predicted gap `pred_gap_D3 = (1 - c^2) · 3.0 · (N/c)` from `c0_prereg_constants.json`: median calibration ratio `gap / pred_gap_D3 >= 0.50` (one-sided 95% BCa CI > `0.40` over 16 targets), AND realized median `Retention(R_mf) >= 1.25` while `Retention(logit_lens) <= 0.75` on `L18+`. | Fails if median `gap / pred_gap_D3 < 0.50` (or CI <= `0.40`), or Retention signs fail on `L18+`. |
| **C0.3 (Jacobian diagonal `v^T J v` vs. `u_t`-specific contraction & Weaver `V3.W2` — F3)** | Compare real model Jacobian diagonals in Regime N across all 7 layers: (a) test whether `û_t^T J û_t` (`logit_lens` Retention) is strictly *more contracted* than the isotropic Jacobian diagonal `v^T J v` (`rand_iso` self-projection) on `>= 5/7` layers (with paired target-cluster BCa 95% CI of `v^T J v - û_t^T J û_t > 0` on `L18+`, calibrated against `tr(R_L)/d - c·N` = `+0.05..+0.07` in `c0_prereg_constants.json`); (b) report **within-arm target heterogeneity** (`IQR` of `Retention(R_mf)` and `Retention(u_t)` across the 16 targets) and score Weaver's `V3.W2` (`CancR_self > 3` on `>= 3/7` layers vs. falsifier `median CancR_self < 1.5`) unfiltered. **Explicit C1 gate rule:** `C0.3` clears the C1 gate iff `v^T J v > û_t^T J û_t` on `L18+` (BCa 95% CI > 0) **AND** `R_mf` exceeds `v^T J v` (`Retention(R_mf) > 1.5 · v^T J v` on `L18+`). | Fails C1 gate if `û_t` is no more contracted than isotropic `v^T J v`, or if `R_mf` fails to beat isotropic transport. |
| **C0.4 (Specificity efficiency & equivalence-tested shuffled null — F5, S8)** | In Regime N (`0.75×`) on `L18+`, define efficiency `eff = dlogit_z / sqrt(max(KL, 1e-6))`. Pre-register one-sided sign test: `eff(R_ridge) >= 1.25 × eff(R_mf)` on `>= 13/16` targets (exact one-sided binomial `p = 0.0106`). Report raw `KL` descriptively (two-sided, no lower-KL conjunct). **Equivalence null on `R_mf_shuffled` (target `t`):** `mean |Δlogit(t)| <= 0.25` AND `p95(|Δlogit(t)|) <= 0.50` with 90% BCa upper bound `<= 0.30` (benchmarked against `rand_iso` noise floor). | Fails if `< 13/16` targets clear `1.25×` efficiency, or `R_mf_shuffled` exceeds the `|Δlogit|` equivalence bound. |
| **C0.5 (`L15+` natural Concept vs. non-pronoun Slot downstream term — F6, Weaver `V3.W3`)** | Using the **frozen deduplicated `L >= 15` 6+6 unit list** in `c0_prereg_constants.json` — Concept: `L16N5264` (`' dog'`), `L23N8559` (`' knight'`), `L21N1951` (`' dog'`), `L19N4502` (`' robot'`), `L20N3060` (`' whale'`), `L20N9371` (`' dog'`); Non-pronoun Slot: `L27N5029` (`' dog'`), `L27N10448` (`' doctor'`), `L26N3854` (`' assistant'`), `L23N6941` (`' AI'`), `L26N7642` (`' king'`), `L28N4480` (`' teacher'`) — measure **signed absolute downstream contribution per unit push** `D_down = (ΣAttn + ΣMLP) / 10.0` across the 8 held-out prompts (zero `bf16`-floor ratio!). Exact two-sided permutation test over the 12 units (`924` combinations); equivalence/falsifier bound `|Δ median D_down| < 0.05` logits/push (`0.50` logits at `+10.0`). *(Top-6 all-Slot including `' I'`/`' me'` reported as secondary diagnostic.)* | Disappears above `L15` if `|Δ median D_down| < 0.05` logits/push or permutation `p >= 0.05`. |

---

## 5. Compute Budget, Hardware Floor & Stage C1 Gate

- **Forward count:**
  - Regime N: `7 layers × 16 targets × 8 prompts × (4 single-draw arms + 2 × K=4 control arms = 12)` = `10,752` forwards.
  - Regime D: `7 layers × 16 targets × 8 prompts × (3 single-draw arms + 1 × K=4 `direct_matched` = 7)` = `6,272` forwards.
  - Natural `L15+` (`C0.5`): `18` unique units (`6` Concept + `6` non-pronoun Slot + `6` all-Slot) × `8` prompts = `144` forwards + `8` clean baseline + `7` sham = `159` forwards.
  - **Total:** `17,183` short `B=1` forwards (or `~9,500` with `K = 2` draws — let's confirm `K = 2` vs `K = 4` during review!).
- **Bandwidth-grounded runtime on Colab L4 (`300 GB/s` HBM, `14.0 GB` bf16 weights → `>= 47 ms/forward` bandwidth floor, S12):** `17,183 × 50 ms ≈ 14.3 min` (`~0.44 CU` on L4 at `1.85 CU/hr`; or `~8 min` / `~0.25 CU` with `K = 2` draws). Budget estimate: **`~0.35–0.55 CU` on Colab L4** (zero lens refitting — reuses `r_lens.pt`).
- **Stdlib-only scorer:** `score_olmo3_c0.py` written in 100% Python 3.10+ stdlib (zero `numpy`/`scipy`/`torch`) for byte-identical `cmp` on any Linux VM.
- **Stage C1 Gate:** Proceed to Stage C1 (discovery/held-out path-patching of downstream difference signals + verified Base → SFT → DPO → Instruct ladder, ancestry pre-verified `cos >= +0.9977` via HTTP Range on `embed_tokens.weight`) **iff Gate `C0.0`, `C0.1`, `C0.2`, and the explicit `C0.3` C1-gate rule all pass**.
