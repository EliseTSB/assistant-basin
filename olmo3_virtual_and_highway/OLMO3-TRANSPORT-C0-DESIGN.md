# Stage C0 Design: Pre-Registered Out-of-Sample Predictive Transport & Active-Erasure Benchmark on OLMo 3 7B

**Procedure:** v1  
**Draft:** v1 (2026-11-22, by Vesper `neon`, before any C0 data; **not yet registered** — open for Claude, Weaver, and Chorus review).  
**Baseline cited:** Exploratory bundle `results/vesper-olmo3-7b-exploratory/` (`OLMO3-VIRTUAL-NEURONS-AND-HIGHWAY-REPORT.md`, `vesper-olmo3-lens` @ `b16d7163`, reusable 64-doc RelP lenses `r_lens.pt` `ceef21fc...` and `s_lens.pt` `a145860b...`).

---

## 1. Research Question

In our exploratory OLMo 3 7B study (`Exp 4B`), injecting raw Logit-Lens directions (`u_t`) into mid-layer residual streams produced the largest direct unembedding write (`+8.93` at `L18`, `+14.56` at `L24`) yet triggered large active downstream erasure (`ΣAttn = -4.01/-5.96`, `ΣMLP = -2.39/-1.11`), whereas matched-filter R-lens directions (`R_mf = R_out^T u_t`) recruited active downstream amplification (`ΣAttn = +3.47/+3.81`, `ΣMLP = +0.22/+3.65`), and ridge-inverted `R_ridge` achieved the highest target specificity (`+25.85σ`).

Two critical critiques from Council (`open-olmo3-7b-virtual-neurons-2026-11-22` `#3`, `#4`, `#5`) separate a genuine mechanistic finding from an artifact:
1. **Generic vs. direction-specific erasure (Weaver `#4`, `V3.W2`):** Without norm-matched random directions and local-direct-gain-matched directions passed through the exact same 4-term telescoping decomposition, we cannot tell whether OLMo 3 *generically* erases almost any direct residual perturbation (generic Hydra/self-repair) or *specifically* erases unembedding-aligned (`u_t`) writes while selectively transporting R-lens-aligned directions.
2. **Out-of-sample predictive validity vs. circularity (Chorus `#5`):** Because `R_mf = R_out^T u_t` is built to maximize the lens-estimated transport of `u_t`, showing `R_mf` wins on the exploratory prompts/personas risks circularity. The non-circular test is whether our **frozen 64-doc WikiText-103 `R_out` lens** predicts **realized out-of-sample causal transport** across **completely unseen lexical targets and unseen context families** — beating raw unembedding (`u_t`) and three matched controls at **both equal write norm AND equal local direct target-logit gain**.

**Stage C0** answers both questions cleanly and cheaply (`~0.3–0.5 CU` on Colab L4) before spending compute on Stage C1 (head-level causal localization & Base → SFT → DPO → Instruct ladder tracing).

---

## 2. Frozen Inputs & Held-Out Evaluation Splits (Zero Overlap with Exploratory Study)

- **Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, eager/SDPA deterministic forward (`use_cache=False`, `B=1` exact single-example forwards to eliminate batch-shape noise).
- **Lens (frozen, zero refitting):** `r_lens.pt` (`ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`), layers `L ∈ {6, 12, 15, 18, 21, 24, 27}` (includes shallow `L6, L12` and 5 lens-validated `L15+` depths with `cos(mhough) >= 0.90` for Weaver's `V3.W3`!).
- **Ridge hyperparameter (frozen):** `λ = 0.5 · tr(R_L^T R_L) / d` (identical to exploratory study; zero retuning).
- **16 brand-new held-out single-token, consonant-initial targets** (zero `"a"`/`"an"` article confound; strictly disjoint from the exploratory 8 personas `{pirate, librarian, wizard, robot, owl, fox, whale, baker}` and all 12 natural census targets; all 16 verified single-token in OLMo 3's tokenizer), spanning 4 balanced domains:
  - **Animals (4):** `" dolphin"` (`99269`), `" hawk"` (`76920`), `" tiger"` (`52835`), `" rabbit"` (`39824`)
  - **Professions (4):** `" surgeon"` (`46098`), `" pilot"` (`18178`), `" painter"` (`30581`), `" judge"` (`11913`)
  - **Objects (4):** `" hammer"` (`24354`), `" lantern"` (`74265`), `" mirror"` (`18327`), `" compass"` (`24653`)
  - **Places (4):** `" harbor"` (`57511`), `" canyon"` (`84300`), `" temple"` (`27850`), `" castle"` (`33684`)
- **4 brand-new held-out context families** (strictly disjoint from `WHO_CHAT_TF`, `SELF`, `NAME_BARE`, `ROLE_BARE`, and all exploratory prompts), 2 prompts per family (`8` prompts total):
  - **F1 (Neutral narrative):** `"At the edge of the quiet village stood a"` / `"Inside the locked wooden box lay a"`
  - **F2 (Descriptive definition):** `"In the encyclopedia, the main entry described a"` / `"Every student in the workshop examined the"`
  - **F3 (Conversational QA — ChatML, non-self):** `"<|im_start|>user\nGive one concrete noun as an example.<|im_end|>\n<|im_start|>assistant\nOne clear example is a"` / `"<|im_start|>user\nName a single thing you might picture in a story.<|im_end|>\n<|im_start|>assistant\nYou might picture a"`
  - **F4 (Contrastive / analytical):** `"Unlike the other items on the list, the final one was a"` / `"After searching for hours, the team finally found a"`
- **Injection site:** Primary = `post_norm` (post-`post_feedforward_layernorm` residual stream, clean mechanistic transport instrument); secondary diagnostic = `pre_norm` (inside `post_feedforward_layernorm`, verifying BlackboxNLP 2026 damping holds across unseen domains at `L18, L24`).
- **Injection position:** `positions="last"` (strictly last token only — eliminates multi-position attention feedback confounds so the 4-term single-position telescoping decomposition is 100% localized to the exact intervened token!).

---

## 3. Experimental Arms (Two Matched Scaling Regimes)

For every `(layer L, target t)` with target unembedding covector `u_t = W_U[t] ⊙ γ_final`, we construct **6 unit-norm directions** (seeded deterministically):
1. **`R_mf`:** `normalize(R_L^T u_t)`
2. **`R_ridge`:** `normalize((R_L^T R_L + λI)^{-1} R_L^T u_t)`
3. **`logit_lens`:** `normalize(u_t)`
4. **`rand_iso` (Control 1 — Isotropic random):** `normalize(z)`, `z ~ N(0, I_d)` independent of `u_t` and `R_L`. Evaluated along **two** readouts:
   - Target readout `u_t` (null baseline for target logit shift), AND
   - **Self-projection readout `v_self = rand_iso`** (`(v_self · Δh_final) / r_c` decomposed identically into `Direct_self + ΣAttn_self + ΣMLP_self + RMSsh_self`) — cleanly measuring whether OLMo 3 generically erases an arbitrary injected residual vector without a `0/0` denominator!
5. **`direct_matched` (Control 2 — Equal local direct gain, orthogonal random subspace):** Let `c = (u_t / ||u_t||) · R_mf` (`0 < c < 1`). Construct `direct_matched = c · (u_t / ||u_t||) + sqrt(1 - c^2) · z_perp`, where `z_perp` is a unit random vector orthogonalized against both `u_t` and `R_mf` (and `R_ridge`). By construction, `direct_matched` has **identical unit write norm (`1.0`) AND identical local direct target-logit projection (`u_t · w`) as `R_mf`**, differing *only* in replacing `R_mf`'s transport-aligned orthogonal component with isotropic noise!
6. **`R_mf_shuffled` (Control 3 — Shuffled target-to-lens assignment):** `R_mf` of a cyclically shifted target `t' = targets[(idx + 4) % 16]` (guaranteed different semantic domain!), evaluated on target `t` (null specificity check) AND on `t'` (confirming `R_mf_shuffled` is an active transport vector for `t'`).

**Two Matched Scaling Regimes (Chorus `#5`):**
- **Regime N (Equal Write Norm):** All 6 arms injected at identical Euclidean norm `dose_N = 0.75 · post_mlp_norm[L]`.
- **Regime D (Equal Local Direct Gain):** `R_mf`, `R_ridge`, `logit_lens`, and `direct_matched` scaled so every arm produces the **exact same frozen-downstream linear direct write `Direct_target = +3.0` logits** (capped at `3.0 · post_mlp_norm[L]` to prevent OOD blowup, with exact cap-hit rate logged!).

---

## 4. Measurements & Exact Accounting

For every `(arm, regime, layer, target, prompt)` trace:
1. **Exact 4-term telescoping decomposition** of `Δlogit(t)`:
   - `exact_dlogit`, `direct_linear`, `frozen_direct_effect`, `sum_downstream_attn` (split `sliding` vs. `full`), `sum_downstream_mlp`, `rmsnorm_shift`, `telescoping_err` (`median`, `p95`, and `max` reported), `path_cancellation_ratio` (`CancR` reported with full distribution: `median`, `IQR`, `p90`, and **only interpreted when `|exact_dlogit| >= 0.25`**, plus a clean denominator-free **downstream retention ratio `Retention = (exact_dlogit - rmsnorm_shift) / direct_linear`** where `Retention < 1` = net erasure, `Retention > 1` = net amplification!).
2. **Out-of-sample R-lens forecast:** For any arm `w` at layer `L` and target `t`, the frozen R-lens predicts `pred_dlogit(w, t) = dose · (u_t^T R_L w) / r_c` (computed before running the forward pass!).
3. **Collateral disturbance:** `KL(P_clean || P_int)` against clean do-nothing baseline, `dlogit_z` (target `Δlogit` z-scored across all 100,278 vocab logits), and target-vs-rival margin (`Δlogit(t) - max_{r in 15 other targets} Δlogit(r)`).

---

## 5. Pre-Registered Technical Gates & Predictions

### Gate `C0.0` (Read before any prediction)
1. **Zero-dose sham:** `dose = 0.0` yields `max |Δlogit| == 0.0` bit-for-bit across all 7 layers.
2. **Exact telescoping accounting:** `max(telescoping_err) <= 0.15` logits and `p95(telescoping_err) <= 0.06` logits across all `bf16` traces.
3. **Equal-direct-gain calibration check (Regime D):** `max |Direct_linear - 3.0| <= 0.05` across all uncapped `R_mf`, `R_ridge`, `logit_lens`, and `direct_matched` traces.
4. **Disjointness & lens hash receipts:** Verified SHA-256 of `r_lens.pt` (`ceef21fc...`) and zero overlap with exploratory targets/prompts.

### Pre-Registered Predictions (`L15+` primary lens-validated band: `L ∈ {15, 18, 21, 24, 27}`; shallow `L6, L12` reported alongside)

| # | Pre-Registered Prediction | Falsifier / Bar | Author |
|---|---|---|---|
| **C0.1 (Out-of-sample predictive ranking)** | Across all `(arm, target, L)` in Regime N on `L15+` (pooling `R_mf`, `R_ridge`, `logit_lens`, `direct_matched`, `R_mf_shuffled`), frozen R-lens forecast `pred_dlogit` achieves Spearman `ρ >= 0.80` against realized `exact_dlogit` AND beats direct unembedding forecast `u_t · w / r_c` by `Δρ >= +0.15` (paired bootstrap 95% CI > 0 across 16 targets). | `ρ(R-lens) < 0.80` or `Δρ` CI includes `0`. | Vesper |
| **C0.2 (Equal-direct-gain transport advantage)** | In Regime D (`Direct = +3.0` logits) on `L15+`, `R_mf` achieves **net downstream amplification** (`median Retention >= 1.25`), while `logit_lens` undergoes **net downstream erasure** (`median Retention <= 0.70`), AND `R_mf` beats `direct_matched` (same write norm + same `Direct`) by `>= +1.5` realized logits (paired 95% CI > 0 across 16 targets). | `R_mf` fails amplification, `logit_lens` does not erase, or `R_mf - direct_matched` CI includes `0`. | Vesper |
| **C0.3 (Generic vs. direction-specific erasure — Weaver `V3.W2`)** | On `rand_iso` self-projection (`v_self = rand_iso`) in Regime N across `L ∈ {6, 12, 15, 18, 21, 24, 27}`, test whether OLMo 3 generically erases arbitrary residual directions (`Retention_self` & `CancR_self`). *Vesper forecast:* `rand_iso` shows mild passive decay (`median Retention_self` in `[0.55, 0.90]`, `median CancR_self < 2.0`), strictly *less* active erasure than `logit_lens` (`u_t`) on `L15+` (`Retention(u_t) < Retention_self(rand_iso)` on >= 4/5 `L15+` layers). *Weaver `V3.W2` explicitly scored alongside:* `CancR_self > 3` on `>= 1/3` of layers vs. falsifier `median CancR_self < 1.5`. | Scored symmetrically against both Vesper's forecast and Weaver's `V3.W2`. | Vesper & Weaver |
| **C0.4 (Specificity frontier: `R_ridge` vs. `R_mf` vs. `logit_lens`)** | In Regime N on `L15+`, `R_ridge` achieves higher target specificity `dlogit_z` AND lower off-target `KL(P_clean || P_int)` than `R_mf` at matched realized `Δlogit` (or strictly higher `dlogit_z / sqrt(KL)` efficiency) on `>= 13/16` held-out targets, while `R_mf_shuffled` stays near zero on target `t` (`|mean Δlogit| <= 0.25`). | `< 13/16` targets or `R_mf_shuffled` leaks `> 0.25`. | Vesper |
| **C0.5 (Natural Concept vs. Slot cancellation above `L15` — Weaver `V3.W3`)** | Evaluate a fresh balanced set of `L15+` natural Concept vs. Slot neurons (6 Concept + 6 Slot strictly at `L >= 15` from our census) on held-out prompts `F1–F4`: test whether the Concept-vs-Slot `Retention` / `CancR` asymmetry survives when restricted strictly to lens-validated layers `L15+`. | Disappears above `L15` (`|Δ median Retention| < 0.15` or wrong sign). | Vesper & Weaver |

---

## 6. Compute Estimate & Stage C1 Gate
- **Total forwards:** `7 layers × 16 targets × 8 prompts × (6 Regime-N + 4 Regime-D)` = `8,960` short `B=1` forwards (`~12–15 ms` each on Colab L4 `bf16` ≈ `~130 s`) + `12` natural `L15+` neurons × `8` prompts = `96` forwards (`~2 s`) + `pre_norm` diagnostic (`L18, L24`, `~30 s`) = **`~3–4 minutes` on Colab L4 (`~0.15–0.25 CU` total!)**.
- **Stdlib-only scorer:** `score_olmo3_c0.py` written in 100% pure Python 3.10+ stdlib (zero `numpy`/`scipy`/`torch`) so clean-clone `cmp` passes byte-for-byte on any Linux VM (including Gemini Task mode).
- **Gate to Stage C1:** Proceed to Stage C1 (discovery/held-out path-patching of downstream difference signals + Base → SFT → DPO → Instruct ladder) **only if `C0.1` and `C0.2` pass** and `C0.3` rules out purely generic erasure.
- **Stage C1 Ladder Ancestry Pre-Verified (`0 CU` HTTP Range check on `model.embed_tokens.weight`, first 1,024 rows = 4.19M params):**
  - `allenai/Olmo-3-1025-7B@a81bae42db39` (`Base`) → `allenai/Olmo-3-7B-Instruct-SFT@e1452fc572d5` (`SFT`): **`cos = +0.997708`** (`rel_l2_delta = 0.068118`)
  - `SFT` → `allenai/Olmo-3-7B-Instruct-DPO@b33130b7de49` (`DPO`): **`cos = +1.000000`** (`rel_l2_delta = 0.000295`)
  - `DPO` → `allenai/Olmo-3-7B-Instruct@6e5971d9eba4` (`Instruct`): **`cos = +1.000000`** (`rel_l2_delta = 0.000399`)
  - Confirmed clean single-lineage descent (no OLMo 2 Part-15 wrong-`main` mismatch).
