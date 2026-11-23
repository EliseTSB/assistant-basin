# Stage C0 Report: Pre-Registered Out-of-Sample Transport & Active-Erasure Benchmark on OLMo 3 7B

**Status:** Pre-registered confirmatory test completed (`OLMO3-TRANSPORT-C0-DESIGN.md` Registered v1 @ `d1cb7cf7`, frozen code @ `4b5c7399`).  
**Headline Verdict:** **Gate `C0.0`: PASS (4/4)** · **`C0.3`: PASS (`p = 3.1e-5`)** · **`C0.1`, `C0.2`, `C0.4`: FAIL** (all 4 Holm directional nulls rejected at `p_holm <= 0.0001`, while calibration & cross-domain null bars failed) · **Stage C1 Gate: FAIL** (Stage C1 is held).  
**Compute:** `13,871` executed `B=1` `bf16` forwards (+ `480` descriptive-`L6` capped traces skipped), `1,142.3 s` (`19.0 min`) on NVIDIA L4, **`0.61 CU`**, `0` active runtimes verified after teardown.  
**Provenance:**
- Design: `olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md` (`sha256 = 59b9da1a585dbecba26954ae41a394f9a1a385bcc831a234b859d9d5e52a78bf`)
- Constants: `olmo3_virtual_and_highway/c0_prereg_constants.json` (`sha256 = 3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98`)
- Frozen lens: `r_lens.pt` (`sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`)
- Receipt: `results/vesper-olmo3-7b-stage-c0/c0_receipt.json.gz` (`0.95 MB`, uncompressed `sha256 = fe7883be37ecf0b7a0a694fc1dd035215d05c27c297cc0959ef23f379258fb1c`)
- Scorecards (bare `/usr/bin/python3 -I` byte-identical via `cmp`): `scorecard.json` (`67632cee…`), `SCORECARD.md` (`3c6393b5…`)

---

## 1. Pre-Registered Gate & Prediction Summary

| ID | Verdict | Registered Result vs. Bar | Holm `p_raw` (`p_holm`) |
|---|---|---|---|
| **C0.0** | **PASS** | Sham `0.0` (7/7) + movement check PASS; telescoping across all `13,856` retained traces (incl. `1,792` `rand_iso`): `median = 0.0126`, `p95 = 0.0425 <= 0.08`, `p99.9 = 0.0781 <= 0.15`, `max = 0.0985`; `L12–L27` cap hit rate `0.0000` (`0/5,376`), `fde_err p95 = 0.0759 <= 0.30`, `p99.9 = 0.3258 <= 0.45`; strict SHA-256 match | — |
| **C0.1** | **FAIL** | `ρ_within(L18+) = 0.7911 >= 0.30` (PASS), `ρ_within(R_mf) = 0.6421 >= 0.25` (`p_perm = 0.0018 < 0.05`, BCa `q05 = +0.249`, PASS), `R_ridge frac_err = 0.118 <= 0.30` (PASS); **FAIL** on `Δρ_within = +0.0732` (`BCa q05 = -0.0176 <= 0`) and `R_mf frac_err = 0.363 > 0.30` *(descriptive: `logit_lens = 0.281`, `direct_matched = 0.263`; early `L6/12 = 0.521`, late-minus-early 95% CI `[+0.045, +0.488]`)* | `0.000100` (`0.000100`) |
| **C0.2** | **FAIL** | Directional `R_mf > direct_matched` on `16/16` targets (`p_2 = 1.5e-5`, PASS), perpendicular-null linearity `m4 = 0.0193` logits (`<= 0.25`, PASS); **FAIL** on pooled `L12–L27` gap ratio `0.2837 < 0.70` (`BCa q05 = 0.2616`), analogue layers `L12/18/24 = 0.171 / 0.235 / 0.383 < 0.50` (`L15/21/27 = 0.207 / 0.298 / 0.476`), and `Retention(logit_lens) = 0.8046 > 0.75` (`L18–27 = 0.604 / 0.799 / 0.863 / 0.854`) | `0.000015` (`0.000061`) |
| **C0.3** | **PASS** | `mean(d_t) = +0.1026`, one-sided 95% BCa CI lower bound **`+0.0760 > 0`** (two-sided 95% CI `[+0.0712, +0.1353]`), **`15/16` targets positive** on `L18+` (only `" judge"` at `-0.027`), **`16/16` at `L27`** (`+0.1381`), `13–16/16` at every layer `L6–L27` | `0.000031` (`0.000061`) |
| **C0.4** | **FAIL** | Equal-direct Regime-D efficiency **PASS**: `geo_mean(R_ridge / R_mf) = 1.777×` (95% CI `[1.636×, 1.898×]`), **`16/16` targets `>= 1.25×`**, `p_4 = 1.5e-5` (`p_4_125 = 1.5e-5`), mean `dlogit_z = +30.64` vs `+24.39`, mean `KL = 0.00982` vs `0.01670`; **FAIL** on Regime-N cross-domain `R_mf_cross` null (`cross_mean = 1.235` vs `rand_iso F_mean = 0.299`, `paired_diff q90 = +0.964 > 0.15`, `cross_max = 1.714 > 1.75·F_p95 = 0.725`, `191/192` pairs `> 2·F_mean`) | `0.000015` (`0.000061`) |

**Format × Style Moderation (`2 × 2`, non-gating):** Both `C0.1a` and `C0.2a` are invariant across prompt format and style:
- Bare vs. ChatML: `C0.1a` `0.778` vs `0.731` (95% CI of diff `[-0.105, +0.257]`); `C0.2a` `0.296` vs `0.283` (`[-0.012, +0.032]`).
- Narrative vs. Analytical: `C0.1a` `0.705` vs `0.792` (`[-0.393, +0.070]`); `C0.2a` `0.283` vs `0.285` (`[-0.030, +0.024]`).

---

## 2. Pre-Registered `C0.3` Result & Post-Hoc Mechanistic Explanations (Not Pre-Registered Findings)

Stage C0 strictly fails its pre-registered Stage C1 gate (`C0.1`, `C0.2`, and `C0.4` fail their registered calibration/null bars despite all four Holm directional nulls rejecting at `p_holm <= 0.0001`). Below we separate the **confirmed pre-registered `C0.3` finding** from two **post-hoc mechanistic explanations** formulated *after* inspecting the C0 verdicts; both explanations are hypotheses to be tested prospectively (not pre-registered conclusions):

### Confirmed Pre-Registered Finding — Single-Position `u_t` Contraction (`C0.3` PASS)
Even at `positions="last"`, raw unembedding directions `u_t` are systematically more contracted than the 32-draw isotropic Jacobian diagonal `v^T J v` (`mean d_t = +0.1026`, 95% BCa CI `[+0.0712, +0.1353]`, one-sided `q05 = +0.0760 > 0`, `p_3 = 3.1e-5`, `15/16` targets on `L18+`, `16/16` at `L27`, `13–16/16` at every layer `L6–L27`). Moreover, the pre-registered `M3` sensitivity check confirms why measuring real `v^T J v` was essential: on `L18+`, real `v^T J v` (`0.694, 0.879, 0.956, 0.970`) exceeds ridge-shrunk `tr(R_L)/d` (`0.511, 0.614, 0.723, 0.836`) by **`+0.2036`**, so comparing `Retention(u_t)` (`0.80`) against `tr(R_L)/d` alone yields `-0.1011` and hides the contraction.

### Post-Hoc Explanation 1 — Single-Position MLP vs. Cross-Position Attention Transport (`C0.1` & `C0.2` Failure)
As documented in `neuron_lens/transport.py:5`, the frozen R-lens estimates the **offset-summed Jacobian** `J_L = mean_t Σ_{t' >= t} ∂h_final(t') / ∂h_L(t)` ("what an activation makes the model say, here and later"). Exploratory Exp 4B (`run_olmo3_idea2_and_idea4.py:742`) injected at `positions="all"`, allowing earlier token positions `s < T` to broadcast the write forward through downstream attention (`Σ_s α_{T,s} = 1`), yielding `ΣAttn = +3.47/+3.81` (`R_mf`) and `-4.01/-5.96` (`logit_lens`) at `L18/L24` and gap ratio `0.831`.

Stage C0 injected at **`positions="last"` only**, isolating the zero-offset slice `∂h_final(T) / ∂h_L(T)`. Post-hoc inspection of Stage C0's Regime-N 4-term decomposition against exploratory Exp 4B shows a `10–19×` drop in `ΣAttn`, while downstream `ΣMLP` remains large and direction-selective:

| Layer | Arm | Stage C0 (`positions="last"`) `Direct` | `ΣAttn` | `ΣMLP` | `RMSsh` | `Exact` (`Pred`) | Realized `Ret` med |
|---:|---|---:|---:|---:|---:|---:|---:|
| `L18` | `R_mf` | `+4.32` | `+0.34` | `-0.02` | `+0.11` | `+4.76` (`+10.50`) | `1.058` |
| `L18` | `R_ridge` | `+5.81` | `+0.25` | `-0.79` | `+0.09` | `+5.37` (`+7.79`) | `0.884` |
| `L18` | `logit_lens` | `+9.85` | `-0.21` | `-3.40` | `-0.05` | `+6.19` (`+4.61`) | `0.617` |
| `L18` | `direct_matched` | `+4.32` | `-0.09` | `-1.58` | `-0.01` | `+2.64` (`+2.04`) | `0.596` |
| `L24` | `R_mf` | `+7.26` | `+0.45` | `+1.58` | `+0.13` | `+9.42` (`+13.83`) | `1.280` |
| `L24` | `R_ridge` | `+9.16` | `+0.29` | `+0.76` | `+0.12` | `+10.33` (`+11.06`) | `1.114` |
| `L24` | `logit_lens` | `+12.36` | `-0.32` | `-1.66` | `-0.09` | `+10.29` (`+8.12`) | `0.834` |
| `L24` | `direct_matched` | `+7.26` | `-0.17` | `-0.94` | `-0.05` | `+6.09` (`+4.79`) | `0.850` |
| `L27` | `R_mf` | `+8.93` | `+0.45` | `+2.13` | `+0.05` | `+11.57` (`+15.63`) | `1.288` |
| `L27` | `R_ridge` | `+11.11` | `+0.23` | `+0.70` | `+0.09` | `+12.12` (`+12.88`) | `1.083` |
| `L27` | `logit_lens` | `+13.57` | `-0.43` | `-1.85` | `-0.05` | `+11.24` (`+10.30`) | `0.834` |
| `L27` | `direct_matched` | `+8.93` | `-0.27` | `-1.11` | `-0.06` | `+7.50` (`+6.80`) | `0.840` |

Two post-hoc observations support this explanation (to be tested directly by a registered paired `positions="all"` vs. `"last"` follow-up):
1. Even at `positions="last"`, `R_mf` beats `direct_matched` on **`16/16` unseen targets** at every layer (`p_2 = 1.5e-5`; Regime-D `L18: +3.26 vs +1.80`, `L24: +3.94 vs +2.53`, `L27: +3.97 vs +2.55`), while `direct_matched` and `logit_lens` match to **`0.0193` logits** (`m4` diagnostic) — suggesting a real single-position MLP channel (`~28–48%` of the offset-summed lens gap), with the remaining `~52–72%` hypothesized to reside in cross-position (`t' > t`) attention transport.
2. Within-arm ranking `Δρ = ρ_pred - ρ_direct` on `L18+` is positive for **`logit_lens` (`+0.2390`: `0.797` vs `0.559`)**, `R_ridge` (`+0.0607`), and `direct_matched` (`+0.0399`), and negative only for `R_mf` (`-0.0539`), where the absent cross-position attention gain caused realized single-position `R_mf` to track `Direct_linear` (`0.696`) more closely than the offset-summed `R_mf` forecast (`0.642`, `frac_err = 0.363`).

### Post-Hoc Explanation 2 — `R_mf` Common-Mode Amplification vs. `R_ridge` Whitening (`C0.4` Failure)
In Regime D (exact equal `Direct_linear = +3.0`), `R_ridge` outperforms `R_mf` on **`16/16` targets** (`16/16 >= 1.25×`, `geo_mean = 1.777×`, 95% CI `[1.636×, 1.898×]`, `p_4 = 1.5e-5`, `p_4_125 = 1.5e-5`), nearly halving KL (`0.00982` vs `0.01670`) while raising target `dlogit_z` (`+30.64` vs `+24.39`) and achieving `0.118` calibration accuracy in `C0.1b`.

Post-hoc inspection of the failed `R_mf_cross` null (`cross_mean = 1.235` vs `rand_iso = 0.299`, `191/192 > 2F_mean`) suggests a concrete geometric mechanism, **with an important scope caveat**: all 16 C0 targets belong to a single lexical class (**consonant-initial, space-prefixed singular concrete nouns**), so this explanation rests on within-class noun geometry and has not been tested across mixed parts of speech:
- On `L18+`, `R_mf_cross` is **`97.6%` positive** (signed mean **`+1.2257`**, vs `rand_iso = -0.0723`, `41.1%` positive; same-domain wrong targets `+1.4779`, `99.0%` positive).
- Pairwise off-diagonal cosine among the 16 frozen target write vectors shows:
  - Raw `logit_lens` (`u_t`): mean off-diagonal cosine **`+0.1826`** (`cross = +0.1764`, `same = +0.2076`) — these 16 consonant-initial singular nouns share a strong common unembedding component.
  - **`R_mf = R_L^T u_t` amplifies that common mode** to **`+0.2615` (`L18`) / `+0.2526` (`L24`)** (`cross = +0.2574`), plausibly explaining why every `R_mf(t')` write boosts all other nouns simultaneously.
  - **`R_ridge = (R_L^T R_L + λI)^{-1} R_L^T u_t` whitens it out by `2.6×`**, reducing off-diagonal cosine to **`+0.1008` (`L18`) / `+0.0953` (`L24`)** (`cross = +0.0943 / +0.0887`). Whether `R_ridge_cross` eliminates the `+1.23`-logit cross-domain bleed remains to be tested directly.

---

## 3. Next Decisions (Honoring the Failed Stage C1 Gate)

Stage C1 is **held** per the registered gate. A compact, pre-registered **Stage C0b** follow-up directly tests both post-hoc explanations above before any Stage C1 work or zero-offset R-lens refitting:
1. **Paired `positions="all"` vs. `positions="last"` test:** Directly tests Post-Hoc Explanation 1 with a pre-registered numeric gap-ratio forecast under `"all"` (anchored on exploratory `0.831`) and a falsifying threshold that refutes the cross-position attention-highway reading if `"all"` fails to recover it.
2. **`R_ridge_cross` readout arm:** Directly tests Post-Hoc Explanation 2 by reading `all_16_dlogits` on `R_ridge` alongside `R_mf` and `rand_iso` (and checking whether ridge whitening collapses the cross-domain noun bleed toward the `rand_iso` floor).
