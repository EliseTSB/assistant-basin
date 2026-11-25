# Stage C2 Design (Draft v1 — Pre-Data Baseline Locked, Awaiting Council Review): Centered & Causal-Whitened Transport, Bidirectional ChatML `ΣMLP` Causal Patching, Post-Erasure Layer Sweep, & `pre_norm` Mask/Split Primitives on OLMo 3 7B

**Status:** **DRAFT v1 — NOT YET REGISTERED (`0 CU`, zero Stage C2 GPU runs exist).** Authored by Vesper (`neon`, research lead) on clean branch `vesper-c2-wsp` (`fee95ce3`) for Council pre-registration review (`open-olmo3-7b-virtual-neurons-2026-11-22`).  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1`.  
**Stage Split (`C2` vs. `C3 / WSP v1`):**  
- **Stage C2 (this study, `~0.45–1.10 CU`):** Deterministic single-forward (`B=1`) mechanistic & pre-norm compilation verification across four linked pillars (`C2.0–C2.4`).  
- **Stage C3 (`WSP v1` Behavioral Generation & 3-Column Public Demo, gated after C2):** Uses C2's confirmed winning write operator, post-erasure layer band, and masked dual-branch/multi-layer `pre_norm` parameterization to compile **Persona** (primary demo arm) and **No-Markdown** (separate mechanical rule arm, never combined) literal weights scored by Part 168 decode word-ban, implied content, 4-turn follow-through, Part 213 copula controls, untuned math/code/chat/refusals, and dual blind readers (`κ ≥ 0.60`) — gated on Elise's explicit weight-release policy approval before any edited weights go public.

### Reproducible `0 CU` Pre-Data Artifacts (Double-Run Byte-Identical `cmp` Verified)
- Builder: [`olmo3_virtual_and_highway/build_olmo3_c2_constants.py`](build_olmo3_c2_constants.py)
- Constants (`c2-prereg-constants-v1`): [`olmo3_virtual_and_highway/c2_prereg_constants.json`](c2_prereg_constants.json) (`sha256 = fdc8512ed80688e694263d16efa35b4601817a15499f11fbb6b334901b9cfb47`)
- Frozen vector bank (`work/olmo3_c2/c2_vector_bank.safetensors`, untracked `23 MB`): `sha256 = 6512fa8f1602489ea95b80f81d643e613fc5866eb36e8594b1b3a7740796dc76`
- Reused 64-doc RelP R-lens (`r_lens.pt`): `sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`
- Unit & invariant test suite: [`olmo3_virtual_and_highway/test_olmo3_c2_constants.py`](test_olmo3_c2_constants.py)

---

### Prior Receipts & Source-Checked 2024–2026 Literature Context
- **Stage C0 (`c96ebfdb`) & C0b (`112450e4`) & C1 (`45b9208c`):** C0 confirmed raw unembedding contraction (`C0.3`, `p = 3.1e-5`); C0b mapped the linear `L18/24/27` `"all"` band vs. nonlinear `L12`; C1 (`1.71 CU` A100) passed all 5 Regime-D linearity clauses but failed `C1.0.3` on stale anchors × genuine `-22.7% / -27.0%` `post_mlp_norm` drops on `":"`-ended prompts, blocking `C1.1–C1.4`. Exploratory C1 readouts exposed the `R_ridge` `1.92×` bleed floor (`C1.3`), the `L18` ChatML `+0.604` `ΣMLP` gap (`C1.1`), and the `L18 → L24` erasure flip (`C1.4`).
- **Source-verified literature hooks (`HTTP 200` checked directly):**
  1. **Park, Choe, & Veitch (`ICML 2024`) + MIT J-Lens Theory (`arXiv:2610.31955`, Oct 2026):** Softmax shift invariance permits centering unembeddings (`u_t - μ_U`) and causal whitening (`Σ_U^{-1/2}(u_t - μ_U)`). MIT proves corpus-averaged Jacobians amplify leading spatially coherent finite-time Lyapunov modes consistent across contexts — explaining why raw `W_U` in J-Lens / R-Lens / J++ lets `μ_U` drive `~91%` rank-1 positive cross-domain bleed in `R_ridge`.
  2. **Global Workspace Ignition & Post-Training (`Gurnee et al.`, `arXiv:2607.15495`, July 2026) + Confidence/Frequency Neurons (`Stolfo et al.`, `arXiv:2406.16254`, June 2024) + DPO Opposing Writes (`Lee et al.`, `ICML 2024` / `EMNLP 2025`):** Explains why ChatML shifts late MLP gain (`ΣMLP`) and why `L19–L23` MLPs selectively erase raw content unembeddings while sparing high-prior function words.
  3. **Post-Norm `pre_norm` Compilation (`Everett Lacey`, Nov 10, 2026, `everettlacey.com/gemma-terse-by-hand/`):** Showed Gemma 4's Layer 12 `post_feedforward_layernorm` had `55` near-zero `γ` dims throttling MLP capacity to `22%`; masking those dims + splitting `70% MLP / 30% Attn` (`m70a30`) across independent RMSNorms + staying post-erasure broke the bottleneck in literal weights.
  4. **Training-Free Behavioral Compilation & Neuron Outputs (`Steer2Edit` `arXiv:2602.09870` NeurIPS 2026; `TFTV` `arXiv:2609.09054` NeurIPS 2026; Shone & Dunlap `alignment.anthropic.com/2026/neuron-basis-theory/` Nov 23, 2026; Shin Megami Boson `lookatmyquant.com/self-editing` Nov 2026 Journal archive):** Motivation for Stage C3's contrastive workspace targets, reserve commit gates, and sibling-bleed checks.

---

## 1. Fresh Units (`0` Overlap with C0/C0b/C1/Personas) & `0 CU` Frozen Baseline

### 1.1 Fresh 24-Target Mixed-POS Bank (`targets_24_spec_sha256 = 1074f160…`)
Strict single-token leading-space roundtrip in OLMo-3 tokenizer, **`0/24` overlap** with C0's `16` nouns, C1's `24` targets, or `8` persona tokens:

| Stratum | `6` Fresh Single-Token Targets (OLMo-3 Token IDs) | `‖u_t‖` (`u_t` range) |
|---|---|---:|
| **Concrete Nouns (`nouns`)** | `' candle'` (38899), `' barrel'` (20929), `' feather'` (50876), `' ribbon'` (45667), `' basket'` (14351), `' glacier'` (94867) | `[1.621, 1.942]` |
| **Base Verbs (`verbs`)** | `' polish'` (45129), `' stumble'` (75237), `' vanish'` (92227), `' mend'` (45876), `' drift'` (34738), `' carve'` (80737) | `[1.674, 2.030]` |
| **Adjectives (`adjectives`)** | `' sturdy'` (45652), `' velvet'` (72426), `' crisp'` (42954), `' faint'` (38678), `' vivid'` (43120), `' shallow'` (26682) | `[1.649, 1.877]` |
| **Function Words (`function`)** | `' however'` (4869), `' whereas'` (20444), `' beneath'` (24923), `' despite'` (8994), `' instead'` (4619), `' beyond'` (7953) | `[1.750, 1.951]` |

### 1.2 Fresh 8-Prompt Bank (`Bare` `12–14` tok, `ChatML` `39–43` tok, All Ending `':'`) + 16 Paired C0 Anchor Test Prompts
- `c2_bare_narr_0` (`T=12`): `"Inside the small drawer, the card displayed a single word:"`
- `c2_bare_narr_1` (`T=13`): `"Across the chalk slate, the final word written in dust was:"`
- `c2_bare_anal_0` (`T=14`): `"In the test ledger, the target entry listed in row four was:"`
- `c2_bare_anal_1` (`T=14`): `"From the sample index, the marked label shown at the end was:"`
- **`c2_chat_*` (`4` prompts, `T=39..43`):** `<|im_start|>user\nContinue this line with one word: <stem><|im_end|>\n<|im_start|>assistant\n<stem>` (`ratio = 3.132×`).
- **Paired C0 Anchor Prediction (`16` clean forwards, Claude `#103(1c)`):** Re-runs C0's exact 8 `" a"`-ended prompts (`4` Bare + `4` ChatML) paired with 8 `":"`-ended versions (`"...stood:"`, `"...lay:"`, `"...was:"`, `"...found:"`), predicting `median(post_mlp_norm)` drops from C0 (`9.33 / 10.24`) into **`[6.2, 8.2]` (`pt 7.2`) at `L24`** and **`[6.4, 8.6]` (`pt 7.5`) at `L27`**.

### 1.3 `0 CU` Frozen-Lens Predictions on Fresh C2 Targets (`c2_prereg_constants.json`)
Unembedding pairwise cosines across C2's `24` targets: `raw_u` (`within +0.1621, cross +0.1208, open×func +0.0905`) → `centered` (`+0.0880, +0.0474, +0.0601`) → `shrink_0.1_whitened` (`+0.0310, +0.0030, +0.0025`). Evaluating unit-norm writes `w` on uncentered target logits `U_d @ (R @ w)`:

| Layer | Arm | `diag_gain` mean | `cross_abs / diag` | `cross_signed / diag` | `same / cross` | Off-Diag Rank-1 SVD Share |
|---:|---|---:|---:|---:|---:|---:|
| **`L18`** | `R_ridge_raw` | `1.4831` | `0.1102` | `+0.1102` | `1.2916×` | **`91.20%`** |
| **`L18`** | `R_ridge_centered` | `1.3439` | `0.0195` | `−0.0080` | `1.7886×` | `38.98%` |
| **`L18`** | `R_ridge_shrink` | `1.3484` | **`0.0158`** | `+0.0030` | **`2.2750×`** | `27.32%` |
| **`L24`** | `R_ridge_raw` | `1.6272` | `0.1010` | `+0.1010` | `1.3241×` | **`90.88%`** |
| **`L24`** | `R_ridge_centered` | `1.4874` | `0.0171` | `−0.0072` | `2.1489×` | `37.22%` |
| **`L24`** | `R_ridge_shrink` | `1.4876` | **`0.0135`** | `+0.0021` | **`2.6988×`** | `25.65%` |
| **`L27`** | `R_ridge_raw` | `1.6898` | `0.1012` | `+0.1012` | `1.3384×` | **`91.07%`** |
| **`L27`** | `R_ridge_centered` | `1.5402` | `0.0171` | `−0.0082` | `2.1243×` | `38.19%` |
| **`L27`** | `R_ridge_shrink` | `1.5412` | **`0.0130`** | `+0.0013` | **`2.7815×`** | `26.10%` |

### 1.4 `0 CU` OLMo-3 `pre_norm` Low-`γ` Mask Geometry (`C2.4`)
Dropping just **`4`** smallest `|γ|` coordinates in `post_feedforward_layernorm` (`ff`) and **`16`** in `post_attention_layernorm` (`attn`) eliminates extreme RMSNorm blowup (`mean eff = (1/rms(γ)) / ‖v_masked / γ‖_2`):

| Layer | `ff` `min \|γ\|` | `ff` Unmasked Eff (`Worst Blowup`) | `ff` `drop_k=4` Masked Eff (`cos_ret`) | `attn` Unmasked Eff (`Worst`) | `attn` `drop_k=16` Masked Eff (`cos_ret`) |
|---:|---:|---:|---:|---:|---:|
| **`L18`** | `0.000893` | `33.68%` (`7.18×`) | **`98.93%`** (`0.99948`) | `21.99%` (`12.35×`) | **`99.23%`** (`0.99801`) |
| **`L21`** | `0.000254` | `24.53%` (`21.84×`) | **`99.21%`** (`0.99955`) | `31.67%` (`9.26×`) | **`99.66%`** (`0.99804`) |
| **`L24`** | `0.001564` | `49.36%` (`5.93×`) | **`98.85%`** (`0.99944`) | `75.38%` (`1.51×`) | **`99.43%`** (`0.99778`) |
| **`L27`** | `0.000148` | `12.89%` (`35.36×`) | **`98.58%`** (`0.99951`) | `21.00%` (`16.66×`) | **`99.60%`** (`0.99817`) |

---

## 2. Global Gate `C2.0`, Pre-Registered 5-Test Holm Family, & Controls

### Global Gate `C2.0` (Zero Cross-Prompt Norm Literals; Data-Derived Within-Run Gate)
1. **`C2.0.1` (Sham & Liveness):** `dose = 0.0` produces `max |Δlogit| == 0.0` across all `14` hook arms/sites; active writes move `max |Δlogit| > 0.0` everywhere.
2. **`C2.0.2` (Telemetry & Telescoping):** `0` non-finite traces; 4-term telescoping error `p95 <= 0.08`, `p99.9 <= 0.15`.
3. **`C2.0.3` (Data-Derived Regime-D/N Linearity Gate):** No hand-typed `POST_MLP_NORM_ANCHOR` across prompts — doses scale cleanly from live in-run clean `pmn`; linearity gated directly by `<= 0.15%` (`>20%` RMSN shrink), `<= 1.50%` (`>12%` shrink), and `|fde_err|` `p95 <= 0.30`, `<= 0.70% > 0.45`. *(Separately logs `C2.0_anchor_pred`: `c0_anchor_colon_prompts_16` `L24/L27` norm drop check as a non-gating prediction.)*
4. **`C2.0.4` (Frozen SHA-256 Provenance):** Exact match to `r_lens.pt`, `c2_prereg_constants.json`, `c2_vector_bank.safetensors`, and registered design commit.

### Five Pre-Registered Confirmatory Tests (Holm-Bonferroni `α = 0.05`, `100,000`-Flip Deterministic Sign-Flip `PERM_SEED = 2026112501`, BCa `BCA_SEED = 2026112502`)

1. **`C2.1` — On-Model Fresh-Word Bleed & Rank-1 Collapse (`L ∈ {18, 24, 27}`, `positions="last"`, `0.75×` live `pmn`):**
   - Measure **actual on-model** `24×24` off-target bleed matrix across all 5 arms (`R_ridge_raw`, `R_ridge_centered`, `R_ridge_shrink`, `R_mf`, `logit_lens`), pre-registering the exact ordering **`cross_abs/diag(Shrink) < cross_abs/diag(Centered) < cross_abs/diag(Raw)`** plus re-measurement of the `24×24` off-diagonal rank-1 SVD energy share and target gain retention (`>= 0.80× Raw`).
   - Holm tests **`p_C2.1_ctr_vs_raw`** (`log(cross_abs_raw / cross_abs_ctr) > 0`, bar `geomean >= 2.0×`) and **`p_C2.1_shk_vs_ctr`** (`log(cross_abs_ctr / cross_abs_shk) > 0`, bar `geomean > 1.05×` & `same/cross(Shrink) > same/cross(Centered)`).
2. **`C2.2` — Bidirectional Causal `L19–L31` MLP Patching (`L18` `positions="all"` ChatML `+0.604` Gap, Claude `#103(1d)` & `#105(2)`):**
   - **Both directions registered independently:**
     - `Bare MLPs (L19–L31) → ChatML run` (**`p_C2.2_bare_into_chat`**): closes ChatML `Δr_all` gap down toward Bare (`point closure >= 70%`, 90% prompt-bootstrap `q05 >= 50%`).
     - `ChatML MLPs (L19–L31) → Bare run` (**`p_C2.2_chat_into_bare`**): opens Bare `Δr_all` gap up toward ChatML (`point closure >= 70%`, 90% prompt-bootstrap `q05 >= 50%`).
   - **Signed & clamped reporting rule:** Report BOTH exact signed closure `(metric_ctrl - metric_patched) / (metric_ctrl - metric_ref)` (preserving values `<0%` or `>100%` without clipping) AND `[0,1]`-clamped fraction; pass requires signed point in `[0.70, 1.35]` and `q05 >= 0.50` so extreme overshoot/instability fails honestly.
   - **Mandatory specificity & source controls:**
     - Equal-size earlier block control **`L4–L16` MLP patch** (must show `< 25%` closure and `< 0.5×` primary closure).
     - **Mismatched-prompt `L19–L31` Bare source patch** (random unpaired bare prompt; must show `< 35%` closure so effect cannot reduce to generic norm shifts).
     - `L19–L31` `Attention`-only patch logged descriptively alongside MLP.
3. **`C2.3` — `7`-Layer Post-Erasure & `γ`-Spectrum Boundary Sweep (`L ∈ {12, 15, 18, 21, 24, 27, 30}`, descriptive localization):**
   - Measure `logit_lens` retention & `ΣMLP_l` erasure separately on open-class (`18` words) vs. `function` (`6` words) paired with `32`-draw `rand_iso` at every sweep depth (`0.75×`, `positions="last"`), locating the exact layer where open-class `ΣMLP` flips out of heavy erasure (`L21` vs. `L24`) to pin Stage C3's post-erasure compilation band.
4. **`C2.4` — Literal `pre_norm` Masking, `m70a30` Split, & Multi-Layer `K=3` Additivity (`2` Doses):**
   - Test **actual `pre_norm` injections inside `post_feedforward_layernorm` / `post_attention_layernorm`** (which algebraically matches appending literal `W_down` / `o_proj` rank-1 neurons; zero post-norm hooks!) at **2 doses** (`0.50×`, `1.25×` clean `pmn`):
     - `L24 prenorm_ff_unmasked` vs. `L24 prenorm_ff_masked` (**`p_C2.4_mask_vs_unmask`**: target `Δlogit` gain ratio `masked / unmasked >= 1.40×`, `90% BCa q05 > 1.20×`),
     - `L24 m70a30` (`70%` `prenorm_ff_masked` + `30%` `prenorm_attn_masked`) vs. `100% ff` at matched total RMS perturbation (evaluating both `Δlogit` & RMSNorm saturation attenuation),
     - **Multi-layer `K=3` (`L21, L24, L27`) additivity check (Claude `#103(1e)`):** Measure `L21` alone, `L24` alone, `L27` alone, all `3` combined simultaneously, and compare exact interaction `combined - (L21 + L24 + L27)` at both doses alongside masked random matched-norm controls (`#103(1g)`).

---

## 3. Exact Compute Budget (`0 CU` Used Until Design + Code Review PASS & Elise Green Light)

| Block | Arms / Sweep Scope | Short `B=1` Forwards |
|---|---|---:|
| Clean + `14` Shams + `16` C0-`":"` Anchor Pairs | Baseline + Gate `C2.0` + `C2.0_anchor_pred` | `38` |
| **Block 1 (`C2.1`)**: 3L × 5 arms × 24 words × 8 prompts (`last`) | `Raw`, `Centered`, `Shrink`, `logit_lens`, `R_mf` | `2,880` |
| **Block 2 (`C2.2`)**: 4 patching conditions × 24 × 8 × 2 | `Bare→Chat`, `Chat→Bare`, `L4–L16 ctrl`, `Mismatch ctrl` | `1,536` |
| **Block 2b (`C2.3`)**: 7L × (24 `logit_lens` + 32 `rand_iso`) × 8 | `L12..L30` post-erasure + contraction sweep | `3,136` |
| **Block 3 (`C2.4`)**: 8 pre_norm arms × 24 × 4 prompts × 2 doses | Mask vs. unmask, `m70a30`, `K=3` singles/combined + ctrls | `1,536` |
| **Total Stage C2** | Estimated `8–14 min` on Colab L4/A100 | **`9,126`** (`~0.45–1.10 CU`) |
