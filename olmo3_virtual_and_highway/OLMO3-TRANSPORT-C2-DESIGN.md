# Stage C2 Design (Draft v2 — Round-1 Review Folded, Awaiting Council Registration Sign-Off): Lens→Model Whitening Transfer, Matched-Length ChatML `ΣMLP` Causal Patching, Post-Erasure Sweep, & `pre_norm` Mask/Split/Additivity on OLMo 3 7B

**Status:** **DRAFT v2 — NOT YET REGISTERED (`0 CU`, zero Stage C2 GPU runs exist).** Authored by Vesper (`neon`, research lead) on clean branch `vesper-c2-wsp` (`fee95ce3`) for Council Round-2 registration review (`open-olmo3-7b-virtual-neurons-2026-11-22`, folding all Round-1 cold design review items `B1–B5`, `M1–M7`, `m1–m9` from `REVIEW-design-ds-r1.md` / `#113`).  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1`.  
**Stage Split (`C2` vs. `C3 / WSP v1`):**  
- **Stage C2 (this study, `9,914` short `B=1` forwards, `~0.50–1.20 CU`):** Deterministic single-forward mechanistic & `pre_norm` compilation verification across four linked pillars (`C2.0–C2.4`).  
- **Stage C3 (`WSP v1` Behavioral Generation & 3-Column Public Demo, gated after C2):** Uses C2's confirmed winning write operator, post-erasure layer band, and masked dual-branch / multi-layer `pre_norm` parameterization to compile **Persona** (primary demo arm) and **No-Markdown** (separate mechanical rule arm, never combined) literal weights scored by Part 168 decode word-ban, implied content, 4-turn follow-through, Part 213 copula controls, untuned math/code/chat/refusals, and dual blind readers (`κ ≥ 0.60`) — gated on Elise's explicit `11-07` weight-release policy approval before any edited weights go public.

### Reproducible `0 CU` Pre-Data Artifacts (Double-Run Byte-Identical `cmp` Verified)
- Builder: [`olmo3_virtual_and_highway/build_olmo3_c2_constants.py`](build_olmo3_c2_constants.py)
- Constants (`c2-prereg-constants-v2`): [`olmo3_virtual_and_highway/c2_prereg_constants.json`](c2_prereg_constants.json) (`sha256 = c325dac0f605059534733dee3da3fe37298f43078d9c02c11a3a9ace88865d63`)
- Frozen vector bank (`work/olmo3_c2/c2_vector_bank.safetensors`, untracked `15.9 MB`, exact 1-to-1 with active arms): `sha256 = f880ba62e540edd943f70f2efff87476e42af344597be9a76496ad5a60cdb29c`
- Reused 64-doc RelP R-lens (`r_lens.pt`): `sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`
- PyTorch / tokenizer invariant test suite: [`olmo3_virtual_and_highway/test_olmo3_c2_constants.py`](test_olmo3_c2_constants.py)
- Pure stdlib (`/usr/bin/python3 -I`, zero third-party deps) design & artifact verifier (`m7`): [`olmo3_virtual_and_highway/test_olmo3_c2_design_stdlib.py`](test_olmo3_c2_design_stdlib.py)

---

### Prior Receipts & Source-Checked 2024–2026 Literature Context
- **Stage C0 (`c96ebfdb`) & C0b (`112450e4`) & C1 (`45b9208c`):** C0 confirmed raw unembedding contraction (`C0.3`, `p = 3.1e-5`); C0b mapped the linear `L18/24/27` `"all"` band vs. nonlinear `L12`; C1 (`1.71 CU` A100) passed all 5 Regime-D linearity clauses but failed `C1.0.3` on stale anchors × genuine `-22.7% / -27.0%` `post_mlp_norm` drops on `":"`-ended prompts (`9.325704 → 7.210327` at `L24`; `10.237432 → 7.473800` at `L27`, from `c0_receipt.json.gz` / `c1_receipt.json.gz`), blocking `C1.1–C1.4`. Exploratory C1 readouts exposed the `R_ridge` `1.92×` bleed floor (`C1.3`), the `L18` ChatML `+0.604` `ΣMLP` gap (`C1.1`), and the `L18 → L24` erasure flip (`C1.4`).
- **Source-verified literature hooks (`HTTP 200` checked directly):**
  1. **Park, Choe, & Veitch (`ICML 2024`) + MIT J-Lens Theory (`arXiv:2610.31955`, Oct 2026):** Softmax shift invariance permits centering unembeddings (`u_t - μ_U`) and causal whitening (`Σ_U^{-1/2}(u_t - μ_U)`). MIT proves corpus-averaged Jacobians amplify leading spatially coherent finite-time Lyapunov modes consistent across contexts — explaining why raw `W_U` in J-Lens / R-Lens / J++ lets `μ_U` drive `~91%` rank-1 positive cross-domain bleed in `R_ridge`.
  2. **Global Workspace Ignition & Post-Training (`Gurnee et al.`, `arXiv:2607.15495`, July 2026) + Confidence/Frequency Neurons (`Stolfo et al.`, `arXiv:2406.16254`, June 2024) + DPO Opposing Writes (`Lee et al.`, `ICML 2024` / `EMNLP 2025`):** Explains why ChatML shifts late MLP gain (`ΣMLP`) and why `L19–L23` MLPs selectively erase raw content unembeddings while sparing high-prior function words.
  3. **Post-Norm `pre_norm` Compilation (`Everett Lacey`, Nov 10, 2026, `everettlacey.com/gemma-terse-by-hand/`):** Showed Gemma 4's Layer 12 `post_feedforward_layernorm` had `55` near-zero `γ` dims throttling MLP capacity to `22%`; masking those dims + splitting `70% MLP / 30% Attn` (`m70a30`) across independent RMSNorms + staying post-erasure broke the bottleneck in literal weights.
  4. **Training-Free Behavioral Compilation & Neuron Outputs (`Steer2Edit` `arXiv:2602.09870` NeurIPS 2026; `TFTV` `arXiv:2609.09054` NeurIPS 2026; Shone & Dunlap `alignment.anthropic.com/2026/neuron-basis-theory/` Nov 23, 2026; Shin Megami Boson `lookatmyquant.com/self-editing` Nov 2026 Journal archive):** Motivation for Stage C3's contrastive workspace targets, reserve commit gates, and sibling-bleed checks.

---

## 1. Fresh Units (`0` Overlap with C0/C0b/C1/Personas) & `0 CU` Frozen-Lens Transfer Baseline

### 1.1 Fresh 24-Target Mixed-POS Bank (`targets_24_spec_sha256 = 1074f160064b9f17fdf42bb1f3c0fc8de05d87168e3428cc58ebebfdcfe807f7`)
Strict single-token leading-space roundtrip in OLMo-3 tokenizer, **`0/24` overlap** with C0's `16` nouns, C1's `24` targets, or `8` persona tokens. Each stratum of `6` is deterministically split (`M2`) into **`primary_16`** (first `4` per stratum) and **`heldout_8`** (last `2` per stratum) for out-of-target `16×8` cross-stratum bleed evaluation. **`C2.2`** uses the balanced **`12`-target** subset (first `3` of each stratum: indices `[0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]`, `9` open-class + `3` function):

| Stratum | `primary_16` (`4` per stratum; first `3` in `C2.2`) | `heldout_8` (`2` per stratum) | `‖u_t‖` range |
|---|---|---|---:|
| **Concrete Nouns (`nouns`)** | `' candle'` (38899), `' barrel'` (20929), `' feather'` (50876), `' ribbon'` (45667) | `' basket'` (14351), `' glacier'` (94867) | `[1.621, 1.942]` |
| **Base Verbs (`verbs`)** | `' polish'` (45129), `' stumble'` (75237), `' vanish'` (92227), `' mend'` (45876) | `' drift'` (34738), `' carve'` (80737) | `[1.674, 2.030]` |
| **Adjectives (`adjectives`)** | `' sturdy'` (45652), `' velvet'` (72426), `' crisp'` (42954), `' faint'` (38678) | `' vivid'` (43120), `' shallow'` (26682) | `[1.649, 1.877]` |
| **Function Words (`function`)** | `' however'` (4869), `' whereas'` (20444), `' beneath'` (24923), `' despite'` (8994) | `' instead'` (4619), `' beyond'` (7953) | `[1.750, 1.951]` |

### 1.2 Fresh 8 Independent Stem Triples (`ShortBare` / `LongBare` / `ChatML`, `B1` + `B2` + `M5`) & Paired C0-`":"` Ratio Test (`m1`, `m2`, `m6`)
All 8 stems (`4` narrative + `4` analytical) end with `":"` and have **0 lemma/token overlap** with C0/C1 prompts or C2 targets. Every triple shares an **exact, bit-for-bit verified trailing stem suffix (`T_s = 12..14` tokens)** preceded by `"\n"` in BOTH `LongBare` (neutral scene-setting prose prefix + `"\n"` + stem) and `ChatML` (`<|im_start|>user\nContinue this line with one word: <stem><|im_end|>\n<|im_start|>assistant\n<stem>`):
- **Verified `B1` suffix-alignment invariant:** `ids_long[-T_s:] == ids_short` AND `ids_chat[-T_s:] == ids_short` on **8/8 triples**.
- **Verified `B2` matched-length invariant:** `LongBare` (`38–43` tok, mean `41.38`) matches `ChatML` (`39–43` tok, mean `41.75`) within **`max |T_long - T_chat| = 1` token** across all 8 pairs (while `ShortBare` is `12–14` tok, mean `13.38`, enabling clean separation of **length** vs. **ChatML format**).
- **`C2.1` & `C2.3` (`8` prompts):** First `2` narrative + first `2` analytical `ShortBare` + corresponding `4` `ChatML` (`narr_0, narr_1, anal_0, anal_1`, preserving Stage C1's 8-prompt layout). **`C2.2` (`8` independent pairs):** ALL `8` triples (`ShortBare`, `LongBare`, `ChatML`). **`C2.4` (`4` prompts):** `LongBare` + `ChatML` on `narr_0, anal_0`.

| Stem ID | Style | `T_s` (`ShortBare`) | `T_long` (`pre`) | `T_chat` (`pre`) | `T_long - T_chat` | Trailing Stem (Bit-Identical Across All 3 Formats) |
|---|---|---:|---:|---:|---:|---|
| `narr_0` | narrative | `12` | `38` (`26`) | `39` (`27`) | `-1` | `"Inside the small drawer, the card displayed a single word:"` |
| `narr_1` | narrative | `13` | `41` (`28`) | `41` (`28`) | `0` | `"Across the chalk slate, the final word written in dust was:"` |
| `narr_2` | narrative | `12` | `40` (`28`) | `39` (`27`) | `+1` | `"Under the brass lantern, the folded slip named one word:"` |
| `narr_3` | narrative | `14` | `42` (`28`) | `43` (`29`) | `-1` | `"Beside the iron key, the stamped wooden tag showed one word:"` |
| `anal_0` | analytical | `14` | `43` (`29`) | `43` (`29`) | `0` | `"In the test ledger, the target entry listed in row four was:"` |
| `anal_1` | analytical | `14` | `42` (`28`) | `43` (`29`) | `-1` | `"From the sample index, the marked label shown at the end was:"` |
| `anal_2` | analytical | `14` | `43` (`29`) | `43` (`29`) | `0` | `"On the control sheet, the verified item printed in box two was:"` |
| `anal_3` | analytical | `14` | `42` (`28`) | `43` (`29`) | `-1` | `"Within the draft manual, the underlined term at the end was:"` |

- **Paired C0-`":"` Anchor Ratio Prediction (`16` clean forwards, non-gating, `m1`, `m2`, `m6`):** Re-runs C0's exact 8 `" a"`-ended prompts (`orig_a`) paired with 8 `":"`-ended versions (`colon`) in the same run. Provenance is strictly **`c0_receipt.json.gz`** `post_mlp_norms` (`L24 = 9.325704`, `L27 = 10.237432`; NOT `c0_prereg_constants` Exp 4B `12.1555`) vs. **`c1_receipt.json.gz`** (`L24 = 7.210327`, `L27 = 7.473800`). Pre-registered **in-run paired ratio** `median(pmn_colon / pmn_orig_a)`:
  - **`L24`:** predicted point **`0.7732`**, range **`[0.70, 0.85]`**.
  - **`L27`:** predicted point **`0.7300`**, range **`[0.65, 0.82]`**.
  - **Length-confound disclosure (`m6`):** Bare pairs preserve exact token count (`9→9, 7→7, 13→13, 10→10`), whereas ChatML pairs drop 1 token (`" a..."` → `":"`); the scorer reports the **Bare-only length-matched ratio** alongside the pooled ratio.

### 1.3 `0 CU` Frozen-Lens Transfer Predictions Across All 5 Arms (`M2`, `M3`, `m5`)
> **Framing (`M2`):** Table 1.3 evaluates frozen unit-norm write vectors `w` through the 64-doc linear RelP surrogate `U_d @ (R_out @ w)`. Because `R_ridge_*` is solved from the same `R_out`, these numbers are **frozen-lens transfer predictions** — Stage `C2.1` tests whether they transfer to actual nonlinear **on-model** `24×24` (and `16×8` held-out) logits.

Unembedding pairwise cosines across C2's `24` targets: `raw_u` (`within +0.1621, cross +0.1208, open×func +0.0905`) → `centered` (`+0.0880, +0.0474, +0.0601`) → `shrink_0.1_whitened` (`+0.0310, +0.0030, +0.0025`).

| Layer | Arm (`5` total) | `diag_gain` | `cross_abs / diag` | `cross_signed / diag` | `heldout_16×8` `cross_abs / diag` | `same / cross` | Off-Diag Rank-1 SVD | `\|cos(v1_right, u·μ̂_U)\|` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **`L18`** | `R_mf` | `1.9944` | `0.2232` | `+0.2232` | `0.2239` | `1.1923×` | `93.59%` | `0.9636` |
| **`L18`** | `R_ridge_raw` | `1.4831` | `0.1102` | `+0.1102` | `0.1079` | `1.2916×` | **`91.20%`** | **`0.9711`** |
| **`L18`** | `R_ridge_centered` | `1.3439` | `0.0195` | `-0.0080` | `0.0193` | `1.7886×` | `38.98%` | `0.3784` |
| **`L18`** | `R_ridge_shrink` | `1.3484` | **`0.0158`** | `+0.0030` | **`0.0155`** | **`2.2750×`** | `27.32%` | **`0.3996`** |
| **`L18`** | `logit_lens` | `0.9381` | `0.0349` | `+0.0262` | `0.0308` | `1.4863×` | `51.00%` | `0.9446` |
| **`L24`** | `R_mf` | `2.0309` | `0.2242` | `+0.2242` | `0.2243` | `1.1635×` | `94.40%` | `0.9609` |
| **`L24`** | `R_ridge_raw` | `1.6272` | `0.1010` | `+0.1010` | `0.0992` | `1.3241×` | **`90.88%`** | **`0.9717`** |
| **`L24`** | `R_ridge_centered` | `1.4874` | `0.0171` | `-0.0072` | `0.0166` | `2.1489×` | `37.22%` | `0.3496` |
| **`L24`** | `R_ridge_shrink` | `1.4876` | **`0.0135`** | `+0.0021` | **`0.0129`** | **`2.6988×`** | `25.65%` | **`0.2637`** |
| **`L24`** | `logit_lens` | `1.2383` | `0.0249` | `+0.0192` | `0.0221` | `2.0792×` | `46.49%` | `0.9124` |
| **`L27`** | `R_mf` | `2.0466` | `0.2574` | `+0.2574` | `0.2585` | `1.1340×` | `95.00%` | `0.9615` |
| **`L27`** | `R_ridge_raw` | `1.6898` | `0.1012` | `+0.1012` | `0.0990` | `1.3384×` | **`91.07%`** | **`0.9728`** |
| **`L27`** | `R_ridge_centered` | `1.5402` | `0.0171` | `-0.0082` | `0.0171` | `2.1243×` | `38.19%` | `0.3842` |
| **`L27`** | `R_ridge_shrink` | `1.5412` | **`0.0130`** | `+0.0013` | **`0.0126`** | **`2.7815×`** | `26.10%` | **`0.2292`** |
| **`L27`** | `logit_lens` | `1.3777` | `0.0222` | `+0.0166` | `0.0189` | `2.3057×` | `44.55%` | `0.9525` |

### 1.4 `0 CU` OLMo-3 `pre_norm` Low-`γ` Mask Geometry (`PRENORM_LAYERS = [21, 24, 27]`, `B4`, `m8`, `m9`)
Dropping `4` smallest `|γ|` dims in `post_feedforward_layernorm` (`ff`, `L ∈ {21, 24, 27}`) and `16` in `post_attention_layernorm` (`attn`, `L = 24` for `m70a30`) eliminates RMSNorm blowup (`mean eff = (1/rms(γ)) / ‖v_masked / γ‖_2`). Every frozen `pre_norm` tensor in `c2_vector_bank.safetensors` corresponds **1-to-1** to an active `C2.4` arm (`L18` unused `pre_norm` tensors removed per `B4`):

| Layer & Branch | `drop_k` | `min \|γ\|` | Unmasked Eff | Unmasked Worst Blowup | Masked Eff | Masked Worst Blowup | Masked / Unmasked | Masked `cos_ret` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`L21` `ff`** | `4` | `0.000254` | `24.53%` | `21.84×` | **`99.21%`** | `1.02×` | `4.044×` | `0.99955` |
| **`L24` `ff`** | `4` | `0.001564` | `49.36%` | `5.93×` | **`98.85%`** | `1.03×` | `2.003×` | `0.99944` |
| **`L24` `attn`** | `16` | `0.008362` | `75.38%` | `1.51×` | **`99.43%`** | `1.02×` | `1.319×` | `0.99778` |
| **`L27` `ff`** | `4` | `0.000148` | `12.89%` | `35.36×` | **`98.58%`** | `1.04×` | `7.650×` | `0.99951` |

> **Mechanistic Note on `L24 m70a30` (`m9`):** In the linear pre-norm regime, `(0.7 * 0.9885 + 0.3 * 0.9943) / 0.9885 = 1.0018×` — practically `1.00×`. Therefore **`C2.4b` (`m70a30` vs. `100% ff`) at high dose (`1.25×` clean `pmn`) explicitly tests nonlinear RMSNorm saturation relief & dual-branch direction advantage** (spreading a large perturbation across two independent post-norm bottlenecks rather than saturating `post_feedforward_layernorm` alone), NOT linear `γ`-capacity gain.

---

## 2. Global Gate `C2.0`, Unified 9-Test Holm Family, & Registered Controls

### Global Gate `C2.0` (Zero Cross-Prompt Norm Literals; Named Trace Denominators `B5`, `m3`, `m4`)
1. **`C2.0.1` (18 Enumerated Shams & Liveness, `m4`):** `dose = 0.0` produces exact `max |Δlogit| == 0.0` across all **`18`** enumerated hook sites (`C2.1_post_norm_L{18,24,27}`, `C2.2_{mlp,attn}_patch_L19_31`, `C2.2_mlp_patch_L4_16`, `C2.3_sweep_L{12,15,18,21,24,27,30}`, `C2.4_prenorm_L{21,24,27}_ff`, `C2.4_prenorm_L24_m70a30`, `C2.4_prenorm_K3_combined`); every active arm moves `max |Δlogit| > 0.0`.
2. **`C2.0.2` (Telemetry & Telescoping):** `0` non-finite traces; 4-term telescoping error `p95 <= 0.08`, `p99.9 <= 0.15`.
3. **`C2.0.3` (Named-Population Regime-N & Regime-D Linearity Gate, `B5`):** No cross-prompt `POST_MLP_NORM_ANCHOR` literals — doses scale strictly from live in-run clean `pmn`. Exact integer failure ceilings over named populations:
   - **Regime-N (`C2.1`, `N = 2,880` traces):** `0` capped doses; `<= 4` traces with `>20%` RMSNorm shrink; `<= 43` traces with `>12%` RMSNorm shrink.
   - **Regime-D (`C2.2` unpatched baselines `ShortBare/LongBare/ChatML × 12 targets × 8 pairs × 2 arms`, `N = 576` traces):** `0` capped doses; `<= 2` traces with `>20%` RMSNorm shrink; `<= 12` traces with `>12%` RMSNorm shrink; `|fde_err|` `p95 <= 0.30`; `<= 8` traces with `|fde_err| > 0.45`. *(Patched `C2.2` runs and `1.25×` high-dose `C2.4` runs log shrink & `fde_err` diagnostically rather than gating.)*
4. **`C2.0.4` (Frozen SHA-256 Provenance, `m3`):** Exact runtime verification of `r_lens.pt`, `c2_prereg_constants.json`, `c2_vector_bank.safetensors`, AND the **registered `OLMO3-TRANSPORT-C2-DESIGN.md` SHA-256** (pinned in the runner at Council registration sign-off commit).

### Unified 9-Test Holm-Bonferroni Confirmatory Family (`M6`)
All **9** confirmatory tests share ONE Holm-Bonferroni family (`α = 0.05`, deterministic `PERM_SEED = 2026112501`, `BCA_SEED = 2026112502`, `10,000` BCa resamples with fail-closed containment `CI_low <= point <= CI_high`). For `N = 24` target-level tests: `100,000` sign-flips (equal-weight mean over `L ∈ {18, 24, 27}` per target before target-level inference). For `N = 8` independent stem-pair tests (`C2.2`): **exact enumeration over all `2^8 = 256` sign flips**, giving `p_min = 1/256 = 0.003906 < 0.005556 = α/9` (`M5`).

**Holm Step-Down Thresholds (`k = 1..9`):** sort `p_(1) ≤ … ≤ p_(9)`; reject while `p_(k) ≤ α/(10-k)`, stop at first failure:
`[0.005556, 0.006250, 0.007143, 0.008333, 0.010000, 0.012500, 0.016667, 0.025000, 0.050000]`.

| # | Test ID | Pillar | Unit (`N`) | Registered Effect-Size & CI Bar (Required Alongside Holm Rejection) |
|---:|---|---|---:|---|
| 1 | `p_C2.1_ctr_vs_raw` | `C2.1` | `24` targets | `geomean(cross_abs_raw / cross_abs_ctr) >= 2.0×` & `90% BCa q05 > 1.50×` |
| 2 | `p_C2.1_shk_vs_ctr` | `C2.1` | `24` targets | `geomean(cross_abs_ctr / cross_abs_shk) >= 1.15×` AND `same/cross(Shrink) >= 1.15 × same/cross(Centered)` (`M3`) |
| 3 | `p_C2.1_gain_ret` | `C2.1` | `24` targets | `geomean(diag_shrink / diag_raw) >= 0.80×` & `90% BCa q05 >= 0.75×` |
| 4 | `p_C2.2_bare_into_chat` | `C2.2` | `8` pairs | Signed closure `∈ [0.70, 1.35]` & `90% paired boot q05 >= 0.50` (gated on format gap `>= 0.30`) |
| 5 | `p_C2.2_chat_into_bare` | `C2.2` | `8` pairs | Signed closure `∈ [0.70, 1.35]` & `90% paired boot q05 >= 0.50` (gated on format gap `>= 0.30`) |
| 6 | `p_C2.2_ctrl_equiv` | `C2.2` | `8` pairs | Composite `max(p_L4_16, p_rand_iso, p_mismatch)` equivalence: `L4–L16 <= 0.25` (CI `<= 0.40`), `rand_iso_L19_31 <= 0.25` (CI `<= 0.40`), `mismatch <= 0.35` (CI `<= 0.50`) (`M1`) |
| 7 | `p_C2.4a_mask_vs_unmask` | `C2.4` | `24` targets | `L24` matched input norm: `geomean(masked / unmasked) >= 1.40×` & `90% BCa q05 > 1.20×` (`B3`) |
| 8 | `p_C2.4b_m70a30_vs_ff` | `C2.4` | `24` targets | `L24` high-dose (`1.25×` `pmn`): `geomean(m70a30 / ff_masked) >= 1.10×` & `90% BCa q05 > 1.00×` (`B3`, `m9`) |
| 9 | `p_C2.4c_k3_additivity_tost` | `C2.4` | `24` targets | `K=3` (`L21+L24+L27`, `0.50×` `pmn`) TOST: `\|mean(rel_int)\| <= 0.20` & `90% BCa CI ⊂ [-0.30, +0.30]` (`B3`) |

### Pillar Protocols & Non-Gating Mechanistic Diagnostics
1. **`C2.1` — On-Model Fresh-Word Bleed, Held-Out Split, & `μ_U` Mechanism (`L ∈ {18, 24, 27}`, `positions="last"`, `0.75×` live `pmn`):**
   - Measure actual on-model `24×24` logit shift matrices across all `5` arms (`R_ridge_raw`, `R_ridge_centered`, `R_ridge_shrink`, `R_mf`, `logit_lens`), logging both full `24×24` and **`heldout_8` (`16×8`)** cross-stratum bleed (`M2`).
   - **Direct `μ_U` mechanism readout (`M3`):** Compute `|cos(v1_right(M_off), u_j · μ̂_U)|` on actual on-model off-diagonal bleed matrices (registered mechanism confirmation: `R_ridge_raw >= 0.70` AND `R_ridge_shrink <= 0.40`, matching Table 1.3's `0.971–0.973 → 0.229–0.400`), plus off-diagonal rank-1 SVD share.
2. **`C2.2` — Matched-Length Bidirectional `L19–L31` MLP Causal Patching (`L18`, `positions="all"`, `12` balanced targets × `8` stem pairs, `B1`, `B2`, `M1`, `M4`, `M5`):**
   - **Matched-length primary contrast & matched absolute dose (`B2`):** Primary contrast is **`ChatML` vs. `LongBare`** (`max |ΔT| = 1` tok), steered at the **exact same absolute write norm** `dose(t, pair) = 3.0 * r_c(LongBare) / (u_t · w_Rmf)` across all `9` conditions so dose-scaling differences cannot create artificial closure. `ShortBare` is run alongside to decompose length vs. ChatML format.
   - **In-run format-gap gate (`B2`):** Before scoring closure, verify `|mean(r_all(ChatML) - r_all(LongBare))| >= 0.30` with 90% paired CI excluding `0`; if the gap vanishes at matched length, `C2.2` returns **`NO_FORMAT_GAP_AT_MATCHED_LENGTH`** (honest length-effect discovery rather than dividing by near-zero noise!).
   - **Symmetric suffix patching (`B1`):** Replace `post_feedforward_layernorm` outputs at `L19–L31` strictly over the **trailing `T_s` identical stem positions** (`-T_s:`), leaving the `26–29` prefix tokens untouched and logged.
   - **`direct_matched` recomputed in EVERY patched forward (`M1`):** All `9` conditions (`unpatched_LongBare`, `unpatched_ChatML`, `unpatched_ShortBare`, `patch_LongBare_to_ChatML_MLP_L19_31`, `patch_ChatML_to_LongBare_MLP_L19_31`, `ctrl_L4_16_MLP_LongBare_to_ChatML`, `ctrl_mismatch_prompt_MLP_L19_31`, `ctrl_rand_iso_same_site_L19_31`, `comp_LongBare_to_ChatML_Attn_L19_31`) run BOTH `R_mf` and `direct_matched[draw0]` (`9 × 12 × 8 × 2 = 1,728` fwds) so `r_all` always compares against each condition's own patched direct baseline.
   - **Clean stability & open-vs-function shape control (`M4`):** Log unsteered patched clean-logit drift (`KL(p_patched || p_ref)`, top-1 agreement) and verify open-class (`9` targets) vs. function-word (`3` targets) relative erasure shape is preserved rather than collapsing via generic corruption.
3. **`C2.3` — `7`-Layer Post-Erasure & `γ`-Spectrum Boundary Sweep (`L ∈ {12, 15, 18, 21, 24, 27, 30}`, `M7`):**
   - Measure `logit_lens` (`24` targets) + `rand_iso` (`32` draws) across `8` prompts (`0.75×`, `positions="last"`).
   - **Explicit Stage C3 selection threshold (`M7`):** Define `E_open(L) = median_{t∈open_18, p∈8}(ΣMLP(L,t,p))` and `E_func(L) = median_{t∈func_6, p∈8}(ΣMLP(L,t,p))`. A layer satisfies `post_erasure(L)` iff **`E_open(L) >= -1.20` logits AND `|E_open(L) - E_func(L)| <= 0.60` logits**. Stage C3's compilation band = **earliest contiguous run of `>= 2` sweep layers in `{18, 21, 24, 27, 30}`** satisfying `post_erasure(L)`; if none, `C2.3 = INCONCLUSIVE` and Stage C3 defaults to `{24, 27}`.
4. **`C2.4` — Literal `pre_norm` Masking, `m70a30` Saturation Relief, & `K=3` Additivity (`11` Enumerated Arms × `24` targets × `4` prompts × `2` doses = `2,112` fwds, `B3`, `B4`):**
   - Actual `pre_norm` additions **before** `post_feedforward_layernorm` / `post_attention_layernorm` (algebraically identical to literal appended `W_down` / `o_proj` rank-1 neurons; zero post-norm hooks!) at `0.50×` and `1.25×` clean `pmn`.
   - **All `11` enumerated arms (exact 1-to-1 with `c2_vector_bank.safetensors`, `K_PRENORM_RAND_DRAWS = 1`):**
     1. `L24.prenorm_ff_unmasked`, 2. `L24.prenorm_ff_masked`, 3. `L24.m70a30`, 4. `L21.prenorm_ff_masked`, 5. `L27.prenorm_ff_masked`, 6. `K3_combined (L21+L24+L27)`, plus their `5` masked random norm-matched controls (`L24.ff_rand[0]`, `L24.m70a30_rand[0]`, `L21.ff_rand[0]`, `L27.ff_rand[0]`, `K3_combined_rand[0]`).
   - `C2.4c` relative interaction metric: `rel_int_t = (Δ_K3 - (Δ_L21 + Δ_L24 + Δ_L27)) / max(|Δ_L21 + Δ_L24 + Δ_L27|, 0.1)`, tested at `0.50×` `pmn` via TOST equivalence (`±0.30` bounds, `|mean| <= 0.20`), and reported descriptively at `1.25×` `pmn` to map saturation onset.

---

## 3. Monte-Carlo Power Summary (`POWER_SIM_SEED = 2026112506`, `M5`, `B3`) & Exact Budget

### 3.1 Pre-Data Statistical Power (`c2_prereg_constants.json["power_simulations"]`)
- **`C2.2` Bidirectional Closure (`N = 8` independent stem pairs, joint `[0.70, 1.35]` + `q05 >= 0.50`):**
  - `sd = 0.10`: `99.9%` at `true = 0.80`, `100.0%` at `true >= 0.85` (`0.0%` false-pass at `true = 0.50`).
  - `sd = 0.15`: `96.7%` at `true = 0.80`, `99.6%` at `true = 0.85` (`0.0%` false-pass at `true = 0.50`).
  - `sd = 0.25`: `87.4%` at `true = 0.80`, `94.9%` at `true = 0.85`, `98.9%` at `true = 0.90` (`1.1%` false-pass at `true = 0.50`).
- **`C2.4b` `m70a30` vs. `100% ff` (`N = 24` targets, `>= 1.10×`, `q05 > 1.00×`, Holm `α/9`):**
  - `sd_log = 0.10`: `98.9%` at `true = 1.15×`, `100.0%` at `true >= 1.20×` (`0.0%` false-pass at `1.00×`).
  - `sd_log = 0.15`: `92.4%` at `true = 1.15×`, `99.7%` at `true = 1.20×` (`0.13%` false-pass at `1.00×`).
- **`C2.4c` `K=3` Additivity TOST (`N = 24` targets, `|mean| <= 0.20`, `90% CI ⊂ [-0.30, +0.30]`):**
  - `sd = 0.20`: `P(declare additive) = 100.0%` at `true = 0.00` (`99.3%` at `0.10`); `P(reject additive) = 99.98%` at `true = 0.35` (`100.0%` at `0.50`).
  - `sd = 0.30`: `P(declare additive) = 99.8%` at `true = 0.00` (`93.1%` at `0.10`); `P(reject additive) = 99.48%` at `true = 0.35` (`100.0%` at `0.50`).

### 3.2 Exact Compute Budget (`0 CU` Used Until Design + Code Review PASS & Elise Green Light)

| Block | Exact Enumeration | Short `B=1` Forwards |
|---|---|---:|
| Clean (`24`) + Enumerated Shams (`18`) + C0-`":"` Anchor Pairs (`16`) | Baseline + Gate `C2.0` + `C2.0_anchor_pred` | `58` |
| **Block 1 (`C2.1`)**: `3`L × `5` arms × `24` targets × `8` prompts (`last`) | `R_mf`, `Raw`, `Centered`, `Shrink`, `logit_lens` | `2,880` |
| **Block 2 (`C2.2`)**: `9` conditions × `12` targets × `8` pairs × `2` (`R_mf` + `dm0`) | Unpatched `3`, bidirectional MLP `2`, controls `3`, Attn `1` | `1,728` |
| **Block 2b (`C2.3`)**: `7`L × (`24` `logit_lens` + `32` `rand_iso`) × `8` prompts | `L12..L30` post-erasure & contraction sweep | `3,136` |
| **Block 3 (`C2.4`)**: `11` enumerated arms × `24` targets × `4` prompts × `2` doses | Mask/unmask, `m70a30`, `K=3` singles/combined + `5` rand ctrls | `2,112` |
| **Total Stage C2** | Estimated `9–15 min` on Colab L4/A100 | **`9,914`** (`~0.50–1.20 CU`) |
