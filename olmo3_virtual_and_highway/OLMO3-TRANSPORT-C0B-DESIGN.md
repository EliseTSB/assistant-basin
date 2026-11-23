# Stage C0b Design (v1): Pre-Registered Paired `positions="all"` vs. `positions="last"` Attention-Highway & `R_ridge` Cross-Domain Whitening Test on OLMo 3 7B

**Procedure:** v1  
**Draft:** v1 (2026-11-23, by Vesper `neon`, before any Stage C0b data; **not yet registered**).  
**Motivation:** Stage C0 (`assistant-basin` `main` `c96ebfdb`) passed Gate `C0.0` (4/4) and `C0.3` (`u_t` contraction, `p = 3.1e-5`) and rejected all four Holm directional nulls (`p_holm <= 0.0001`), but failed `C0.1`, `C0.2`, and `C0.4`, holding Stage C1. Post-hoc diagnosis offered two falsifiable mechanistic explanations:
1. **Cross-position attention highway (`C0.2` / `C0.1`):** Frozen `R_out` (`neuron_lens/transport.py:5`) estimates the offset-summed Jacobian `mean_t Σ_{t' >= t} ∂h_final(t') / ∂h_L(t)` ("here and later"), matching exploratory Exp 4B (`positions="all"`, 4-frame pooled gap ratio `0.831`). Stage C0 injected at `positions="last"` only, collapsing `ΣAttn` `8.5–18.8×` and yielding pooled 4-layer gap ratio `0.305` (`L12/18/24/27`: `0.171 / 0.235 / 0.383 / 0.476`).
2. **Common-mode amplification vs. ridge whitening (`C0.4`):** Across the 16 consonant-initial singular concrete nouns, `R_mf = R_L^T u_t` amplifies pairwise off-diagonal cosine (`+0.183` → `+0.262/+0.253`, lifting cross-domain nouns by `+1.226`, `97.6%` positive), whereas `R_ridge = (R_L^T R_L + λI)^{-1} R_L^T u_t` whitens off-diagonal cosine down `2.6×` (`+0.101/+0.095`).

Stage C0b directly tests **both** post-hoc explanations in one paired, self-contained run — with explicit numeric forecasts and refutation thresholds for each — while holding any zero-offset R-lens refit or Stage C1 spend until this result is in.

**Frozen Constants:** `olmo3_virtual_and_highway/c0b_prereg_constants.json` (`sha256 = 765b792b1ef0452a60687b8f800785fc619edf8eea1b6f6796e4d6c5a6698774`).  
**Reused Frozen Assets (0 new lens fitting):** `r_lens.pt` (`ceef21fc…`), `c0_prereg_constants.json` (`3be64ef8…`), `c0_vector_bank.pt` (`ab7e04ca…`), and Stage C0's exact 16 targets and 8 held-out prompts (`OLMO3-TRANSPORT-C0-DESIGN.md` `59b9da1a…`).

---

## 1. Frozen Inputs, Paired Design & Pre-Data Anchors

- **Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1`.
- **4 layers:** `L ∈ {12, 18, 24, 27}` (all 3 exploratory Exp 4B analogue layers `L12, L18, L24`, plus deepest `L18+` layer `L27`). `L18+` cross-domain readout band is `{18, 24, 27}`.
- **16 unseen targets & 8 paired `2 × 2` prompts:** Identical to Stage C0 (`c0_prereg_constants.json`), reusing the exact frozen vector bank `c0_vector_bank.pt` (`ab7e04ca…`) so zero target-selection or vector-construction variance enters the `"all"` vs. `"last"` comparison. *(Scope caveat preserved: all 16 targets are consonant-initial, space-prefixed singular concrete nouns.)*
- **Two injection position modes (paired in-run on the same 8 clean baselines):**
  - **`positions="last"`:** Inject `dose · w` into `post_feedforward_layernorm` output at token index `-1` only (exact Stage C0 mechanism).
  - **`positions="all"`:** Inject `dose · w` into `post_feedforward_layernorm` output at **every** token position `0..T-1` (exact exploratory Exp 4B mechanism). Because `h_pre_mlp_L[T-1]` is unchanged by injecting at layer `L`'s output, `Δmlp_L[T-1] = dose · w` and `Direct_linear = dose · (u_t · w) / r_c` at readout position `T-1` are **identical** across both modes, and the exact 4-term telescoping identity `exact_dlogit = Direct_linear + ΣAttn + ΣMLP + rmsnorm_shift` holds identically (verified in Exp 4B `positions="all"`, `max tel = 0.0761`).
- **Two matched regimes (identical definitions to Stage C0):**
  - **Regime D (`Direct_linear = +3.0`, cap `3.0 · post_mlp_norm[L]`):** Arms `R_mf`, `logit_lens`, and `direct_matched` (`K = 4` draws) at `L ∈ {12, 18, 24, 27}` in **both** `positions="last"` and `positions="all"`. Because `Direct_linear` at `T-1` is mode-independent, requested doses are identical to Stage C0 (where `cap_hit_rate = 0.0000` on `L12–L27`).
  - **Regime N (`dose_N = 0.75 · post_mlp_norm[L]`):** Arms `R_mf`, `R_ridge`, and `logit_lens` at `L ∈ {12, 18, 24, 27}` — recording `all_16_dlogits` (`[16]` exact target Δlogits read free from `[V]` logits) on **ALL THREE** arms — plus pooled `rand_iso` (`M = 32` draws/layer) at `L ∈ {18, 24, 27}`, in **both** `positions="last"` and `positions="all"`.

### Pre-Data Baselines, Exploratory Analogues & Frozen Lens Forecasts (`c0b_prereg_constants.json`)

| Layer | C0 `"last"` `gap/pred` med | Exp 4B `"all"` `gap/pred` med (4-fr) | C0 `"last"` `ΔAttn / ΔMLP / ΔTot` (D3) | Exp 4B `"all"` `ΔAttn / ΔMLP / ΔTot` (D3) | Implied `Attn` share of lift | Frozen pred `B_cross` (`R_mf / R_ridge / ll / rand`) |
|---:|---:|---:|---|---|---:|---|
| `L12` | `0.1708` | `0.7765` | `+0.432 / +0.810 / +1.423` | `+4.342 / +2.172 / +6.499` | `77.0%` | — *(not in `L18+`)* |
| `L18` | `0.2349` | `0.8104` | `+0.301 / +1.085 / +1.463` | `+4.160 / +1.293 / +5.445` | `96.9%` | `2.666 / 1.072 / 0.188 / 0.129` |
| `L24` | `0.3834` | `0.8826` | `+0.269 / +1.093 / +1.411` | `+2.718 / +1.051 / +3.770` | `103.8%` | `3.374 / 1.391 / 0.245 / 0.174` |
| `L27` | `0.4757` | *(unmeasured)* | `+0.246 / +1.154 / +1.423` | *(unmeasured)* | *(unmeasured)* | `4.244 / 1.638 / 0.303 / 0.185` |
| **Pooled** | **`0.3051`** *(4L, 64-cell)* | **`0.8308`** *(3L, 4-fr)* | `SD_t(4L mean) = 0.0396` | Forecast lift `≈ +0.50` | **`~77–104%`** | **`3.428 / 1.367 / 0.245 / 0.162`** (`2.51×` abs; `1.99×` norm) |

*(All per-target frozen `C0b.3` predictions `pred_B_cross_*` and `pred_S_cross_*` for all 16 targets are embedded in `c0b_prereg_constants.json`; exact Exp 4B per-target forecasts remain embedded in `c0_prereg_constants.json`.)*

---

## 2. Exact Inference, Gate `C0b.0` & Pre-Registered Predictions (`C0b.1–C0b.3`)

- **Inference & 3-Test Holm-Bonferroni Family (`α = 0.05`, `α/3 = 0.01667`):** `10,000`-resample BCa cluster bootstrap over the `16` target clusters (`math.fsum` stdlib scorer, identical conventions to Stage C0). All three primary tests use exact `2^16 = 65,536` one-sided paired sign-flip p-values (`p_min = 1.53e-5 << 0.01667`).
- **Registered `2 × 2` Format × Style Moderation (non-gating):** Recompute `C0b.1` `gap_all / pred_gap_D3` and `gap_all - gap_last` separately in Bare (short `7–11` tok) vs. ChatML (`31–38` tok) and Narrative vs. Analytical prompts, reporting paired 95% BCa CIs over the 16 target clusters to test whether prompt length/format modulates `"all"` transport.

### Gate `C0b.0` (All 4 clauses must pass before reading `C0b.1–C0b.3`)
1. **Non-tautological zero-dose sham & movement check:** `dose = 0.0` via the real clone/cast/add hook path yields `max |Δlogit| == 0.0` bit-for-bit across all `4` layers × `2` modes (`8` shams), AND `R_mf` at `dose_N` yields `max |Δlogit| > 0.0` at every `(layer, mode)`.
2. **Quantile telescoping accounting:** Across all `10,752` retained intervened `bf16` traces (all finite, including all `1,536` `rand_iso` traces), `p95(telescoping_err) <= 0.08` AND `p99.9 <= 0.15` (Stage C0 `"last"` measured `0.0425 / 0.0781`; Exp 4B `"all"` measured `0.0561 / max 0.0761`).
3. **Regime-D cap & intervened-denominator gate:** Across both modes on `L ∈ {12, 18, 24, 27}`, `cap_hit_rate <= 2%` per `(mode, arm, L)` (capped traces dropped; Stage C0 measured `0.0000`), AND across retained Regime-D traces `p95(|frozen_direct_effect - 3.0|) <= 0.30` AND `p99.9 <= 0.45` (Stage C0 `"last"` `0.0759 / 0.3258`; Exp 4B `"all"` `0.1640 / max 0.2579`, `R_mf p95 = 0.1971`).
4. **Strict frozen provenance:** Receipt SHA-256 hashes match `r_lens.pt` (`ceef21fc…`), `c0_prereg_constants.json` (`3be64ef8…`), `c0b_prereg_constants.json` (`765b792b…`), `c0_vector_bank.pt` (`ab7e04ca…`), and registered `OLMO3-TRANSPORT-C0B-DESIGN.md`. *(Non-gating sanity check: log `max |gap_ratio_last(C0b) - gap_ratio_last(C0)|` across `{12, 18, 24, 27}`.)*

### Pre-Registered Predictions (`C0b.1–C0b.3`)

| # | Numeric Forecast, Exact Aggregation & Holm p-Value (`p_1..p_3`) | Pass Bar & Explicit Refutation Threshold |
|---|---|---|
| **C0b.1 (Paired `positions="all"` vs. `"last"` gap recovery)** | In Regime D (`Direct_linear = +3.0`) on `L ∈ {12, 18, 24, 27}` (8-prompt-averaged per `(mode, target, L)`), define `gap_mode = exact_dlogit(R_mf) - exact_dlogit(direct_matched)`, normalized ratio `r_mode(L, t) = gap_mode(L, t) / pred_gap_D3(L, t)`, and per-target paired lift `Δr_t = mean_{L∈4L}[r_all(L, t) - r_last(L, t)]`. **Numeric forecast:** pooled 4-layer `median(r_all) ≈ 0.83` (Exp 4B 4-frame analogue `0.831`), paired lift `mean(Δr_t) ≈ +0.50` above C0 `"last"` (`0.305`). **`p_1` (Holm):** exact `2^16` one-sided paired sign-flip p-value on `Δr_t > 0`. **Pass bars:** **(a) Paired lift:** `p_1` passes Holm AND `mean(Δr_t) >= +0.25` with one-sided 95% BCa `q05 > +0.20`; **(b) Calibration recovery under `"all"`:** pooled 4-layer (`64`-cell) `median(r_all) >= 0.65` with one-sided 95% BCa `q05 > 0.55` **AND** per-layer `median(r_all) >= 0.50` at each analogue-measured layer (`L12, L18, L24`) individually (`L27` reported descriptively while entering the pooled median and `p_1`). *(Descriptives: `Retention(logit_lens)` under `"all"` vs. `"last"`, Perpendicular + write-norm nonlinearity diagnostic `|median(dm - ll)|` under `"all"`, and Regime-N `R_mf` `frac_err` under `"all"` vs. `"last"`.)* | **Refutes Post-Hoc Explanation 1** if `p_1` fails Holm, `mean(Δr_t) < +0.25`, `Δr_t` BCa `q05 <= +0.20`, pooled `median(r_all) < 0.65`, pooled `r_all` BCa `q05 <= 0.55`, OR any of `L12, L18, L24` has `median(r_all) < 0.50`. |
| **C0b.2 (`ΣAttn` mechanism attribution of the `"all" - "last"` lift)** | Using exact 4-term telescoping in Regime D (`L ∈ {12, 18, 24, 27}`), define per-target attention-gap shift `ΔAttn_t = mean_{L∈4L}[(ΣAttn_all(R_mf) - ΣAttn_all(dm)) - (ΣAttn_last(R_mf) - ΣAttn_last(dm))]`, MLP-gap shift `ΔMLP_t` analogously, and total-gap shift `ΔTot_t = mean_{L∈4L}[gap_all - gap_last]`. **Numeric forecast:** `ΔAttn_t / ΔTot_t ≈ 0.77–1.04` (grounded in Exp 4B vs. C0: `77.0%` at `L12`, `96.9%` at `L18`, `103.8%` at `L24`). **`p_2` (Holm):** exact `2^16` one-sided paired sign-flip p-value on `ΔAttn_t > 0`. **Pass bar:** `p_2` passes Holm **AND** `median_t(ΔAttn_t / max(ΔTot_t, 1e-6)) >= 0.50` with one-sided 95% BCa `q05 > 0.35` *(requires `C0b.1` `mean(ΔTot_t) > 0`; if `ΔTot_t <= 0`, `C0b.2` fails automatically)*. *(Descriptives: `ΔMLP_t`, per-layer `ΣAttn_all` vs. `ΣAttn_last` for `R_mf` and `logit_lens`.)* | **Refutes the cross-position attention mechanism** if `p_2` fails Holm, `ΔTot_t` is non-positive, `median_t(ΔAttn_t / ΔTot_t) < 0.50`, OR BCa `q05 <= 0.35` (e.g., if `"all"` instead lifts `ΣMLP`). |
| **C0b.3 (`R_ridge` cross-domain whitening vs. `R_mf`)** | In Regime N (`0.75 · post_mlp_norm[L]`) on `L ∈ {18, 24, 27}` at **`positions="last"`** (primary, matching Stage C0's `C0.4`; `positions="all"` reported as registered paired secondary), read all `192` cross-domain `(t, t')` pairs free from `[V]` logits on `R_mf`, `R_ridge`, and `logit_lens`. For each readout target `t ∈ {1..16}`, define absolute cross-domain bleed `B_cross(arm, t) = mean_{t' cross, L∈{18,24,27}, p∈8} |Δlogit(t; arm(t'))|`, self-normalized bleed ratio `S_cross(arm, t) = B_cross(arm, t) / max(self_dlogit(arm, t), 1e-6)`, and paired log-ratios `u_abs(t) = log(max(B_cross(R_mf, t), 1e-6) / max(B_cross(R_ridge, t), 1e-6))` and `u_norm(t) = log(max(S_cross(R_mf, t), 1e-6) / max(S_cross(R_ridge, t), 1e-6))`. **Numeric forecast (frozen R-lens `c0b_prereg_constants.json`):** `geo_mean(B_cross(R_mf)/B_cross(R_ridge)) ≈ 2.51×` (`3.428` vs. `1.367`, `SD_log = 0.044`), self-normalized `≈ 1.99×` (`SD_log = 0.050`), while `R_ridge_cross` remains strictly above `rand_iso` (`0.162`). **`p_3` (Holm):** exact `2^16` one-sided paired sign-flip p-value on `u_abs(t) > 0`. **Pass bar:** `p_3` passes Holm **AND** `exp(mean(u_abs)) >= 1.50` (`>= 13/16` targets have `B_cross(R_mf) > B_cross(R_ridge)`) **AND** `exp(mean(u_norm)) >= 1.50`. **Exact `2^16` full conjunction power (`α/3 = 0.0167`):** coin flip at the `1.50×` bar itself (`49.4% / 50.2% / 51.1%` at `SD_log = 0.10 / 0.20 / 0.30`), reaching **`100.0% / 99.9% / 97.5%` at `1.75×`** and **`100.0%` at `2.00×`**. *(Descriptives: `R_ridge_cross` & `logit_lens_cross` vs. `rand_iso` `F_mean` / `F_p95`, signed means, fraction positive, same-domain wrong targets, and `positions="all"` secondary.)* | **Refutes Post-Hoc Explanation 2** if `p_3` fails Holm, `exp(mean(u_abs)) < 1.50`, `< 13/16` targets show lower `R_ridge` bleed, OR `exp(mean(u_norm)) < 1.50`. |

---

## 3. Compute Budget, Runtime Floor & Downstream Decision Rule

- **Exact forward count (all clean baselines shared across modes and regimes):**
  - Clean baselines (`8` prompts) + zero-dose shams (`4` layers × `2` modes `last/all`): **`16` forwards**.
  - Regime D (`Direct = +3.0`, `4` layers × `16` targets × `8` prompts × `[R_mf, logit_lens, direct_matched(K=4)]` = `6` × `2` modes): **`6,144` forwards**.
  - Regime N (`0.75×`, `4` layers × `16` targets × `8` prompts × `[R_mf, R_ridge, logit_lens]` = `3` × `2` modes `= 3,072` + `3` layers `{18, 24, 27}` × `32` `rand_iso` × `8` prompts × `2` modes `= 1,536`): **`4,608` forwards**.
  - **Exact total:** **`10,768` short `B=1` `bf16` forwards** (`25%` fewer than Stage C0's `14,351`).
- **Stage C0-grounded runtime on Colab L4:** Using Stage C0's **actual measured NVIDIA L4 throughput** (`1,142.3 s / 13,871 fwds = 82.35 ms/fwd` including hooks & decomposition), `10,768 × 82.35 ms ≈ 887 s ≈ 14.8 min` (`0.456 CU` compute at `1.85 CU/hr`; budget estimate **`~0.48–0.55 CU` billed** including VM boot/download/teardown).
- **Stdlib-only scorer:** `score_olmo3_c0b.py` in 100% Python 3.10+ stdlib with `math.fsum` and fail-closed schema/finiteness/provenance/movement gates (identical architecture to `score_olmo3_c0.py`).
- **Decision rule after Stage C0b:**
  - If **`C0b.0` + `C0b.1` + `C0b.2` PASS**: Cross-position attention transport is confirmed as the missing `positions="all"` channel (`~50%+` of `R_out`), motivating Stage C1 path-patching of **both** the cross-position attention highway and the single-position MLP contraction channel (still gated on explicit Stage C1 design review + Elise GPU approval).
  - If **`C0b.1` or `C0b.2` FAILS**: The cross-position attention-highway explanation is **refuted** — no Stage C1 highway path-patching; investigate why `R_out` over-predicts `R_mf` retention on held-out nouns (e.g., prompt-distribution mismatch vs. WikiText-103 or zero-offset vs. offset-summed lens mismatch).
  - **`C0b.3` verdict** independently determines whether ridge whitening (`R_ridge`) causally suppresses cross-domain bleed on the 16 concrete nouns (if `C0b.3` passes, any future natural/virtual write benchmark should default to `R_ridge` and test mixed-POS generalization).
