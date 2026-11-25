# Stage C2 Design (Draft v4 — Final Registration Delta): Held-Out Whitening Arm Comparison & Frozen-Lens Calibration, Pure ChatML-Template `ΣMLP` Causal Patching, Post-Erasure Sweep, & `pre_norm` Mask/Additivity on OLMo 3 7B

**Status:** **DRAFT v4 — FINAL REGISTRATION CANDIDATE (`0 CU`, zero Stage C2 GPU runs exist).** Authored by Vesper (`neon`, research lead) on clean branch `vesper-c2-wsp` (`fee95ce3`) for Council final delta sign-off and registration to `main` (`open-olmo3-7b-virtual-neurons-2026-11-22`, folding `F1–F11` from `REVIEW-design-ds-r3.md` / `#117`).  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1`.  
**Stage Split (`C2` vs. `C3 / WSP v1`):**  
- **Stage C2 (this study, `8,623` short `B=1` forwards, `~0.45–1.05 CU`):** Deterministic single-forward mechanistic & `pre_norm` compilation verification across four linked pillars (`C2.0–C2.4`).  
- **Stage C3 (`WSP v1` Behavioral Generation & 3-Column Public Demo, gated after C2):** Uses C2's confirmed winning write operator, post-erasure layer band, and `pre_norm` parameterization to compile **Persona** (primary demo arm) and **No-Markdown** (separate mechanical rule arm, never combined) literal weights scored by Part 168 decode word-ban, implied content, 4-turn follow-through, Part 213 copula controls, untuned math/code/chat/refusals, and dual blind readers (`κ ≥ 0.60`) — gated on Elise's explicit `11-07` weight-release policy approval before any edited weights go public.

### Reproducible `0 CU` Pre-Data Artifacts (Double-Run Byte-Identical `cmp` Verified)
- Builder: [`olmo3_virtual_and_highway/build_olmo3_c2_constants.py`](build_olmo3_c2_constants.py)
- Constants (`c2-prereg-constants-v4`): [`olmo3_virtual_and_highway/c2_prereg_constants.json`](c2_prereg_constants.json) (`sha256 = 583117f50cfceeb5cddcfa239c0efa0c32846dd69f6756540cd3bbba5cec5b65`)
- Frozen vector bank (`work/olmo3_c2/c2_vector_bank.safetensors`, untracked `9.6 MB`, `34` tensors, 100% draw-level 1-to-1): `sha256 = 7124dd565331fd455d6d91d8fe845ba8783c2b0d366f6d11eb533947cb09401d`
- Reused 64-doc RelP R-lens (`r_lens.pt`): `sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`
- PyTorch / tokenizer invariant test suite: [`olmo3_virtual_and_highway/test_olmo3_c2_constants.py`](test_olmo3_c2_constants.py)
- Pure stdlib (`/usr/bin/python3 -I`, row-strict + Holm symbol-definition verifier): [`olmo3_virtual_and_highway/test_olmo3_c2_design_stdlib.py`](test_olmo3_c2_design_stdlib.py)

---

### Prior Receipts & Literature Context
- **Stage C0 (`c96ebfdb`) & C0b (`112450e4`) & C1 (`45b9208c`):** C0 confirmed raw unembedding contraction (`C0.3`, `p = 3.1e-5`); C0b mapped the linear `L18/24/27` `"all"` band vs. nonlinear `L12`; C1 (`1.71 CU` A100) passed all 5 Regime-D linearity clauses (`0/1,728` capped, `0 ≤ 2` `>20%`, `5 ≤ 26` `>12%`, `fde_p95 = 0.1853`, `2 ≤ 12` `>0.45`) but failed `C1.0.3` on stale anchors × genuine `-22.7% / -27.0%` `post_mlp_norm` drops on `":"`-ended prompts (`9.325704 → 7.210327` at `L24`; `10.237432 → 7.473792` at `L27`, verbatim from `c0_receipt.json.gz` / `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz`), blocking `C1.1–C1.4`. Exploratory C1 readouts exposed the `R_ridge` `1.92×` bleed floor (`C1.3`), the `L18` `r_all` gap (`+0.604`, `Bare 1.3267` vs. `ChatML 1.9307`, attributed in C1 to an all-position format×length interaction and decomposed in Phase 0b as `2.8×` larger `ΣMLP`), and the `L18 → L24` `logit_lens` erasure flip (`C1.4`).
- **Motivating literature (none carries a registered Stage C2 parameter):** Park, Choe, & Veitch (`ICML 2024`, unembedding centering & causal whitening); MIT J-Lens Theory (`arXiv:2610.31955`); Gurnee et al. (`arXiv:2607.15495`); Stolfo et al. (`arXiv:2406.16254`); Lee et al. (`ICML 2024` / `EMNLP 2025`); Everett Lacey (`everettlacey.com/gemma-terse-by-hand/`, Nov 2026); Steer2Edit (`arXiv:2602.09870`), TFTV (`arXiv:2609.09054`), Shone & Dunlap (`alignment.anthropic.com/2026/neuron-basis-theory/`).

---

## 1. Fresh Units (`0` Overlap with C0/C0b/C1/Personas) & `0 CU` Frozen-Lens Baseline

### 1.1 Fresh 24-Target Mixed-POS Bank (`targets_24_spec_sha256 = 1074f160064b9f17fdf42bb1f3c0fc8de05d87168e3428cc58ebebfdcfe807f7`)
Strict single-token leading-space roundtrip in OLMo-3 tokenizer, **`0/24` overlap** with C0's `16` nouns, C1's `24` targets, or `8` persona tokens. Each stratum of `6` is split deterministically into **`primary_16`** (`4` per stratum, `C2.1` sources) and **`heldout_8`** (`2` per stratum, `C2.1` cross-stratum receivers). **`C2.2`** uses the balanced **`12`-target** subset (first `3` of each stratum: indices `[0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]`, `9` open-class + `3` function):

| Stratum | `primary_16` (`4` per stratum; first `3` in `C2.2`) | `heldout_8` (`2` per stratum) | `‖u_t‖` range |
|---|---|---|---:|
| **Concrete Nouns (`nouns`)** | `' candle'` (38899), `' barrel'` (20929), `' feather'` (50876), `' ribbon'` (45667) | `' basket'` (14351), `' glacier'` (94867) | `[1.621, 1.942]` |
| **Base Verbs (`verbs`)** | `' polish'` (45129), `' stumble'` (75237), `' vanish'` (92227), `' mend'` (45876) | `' drift'` (34738), `' carve'` (80737) | `[1.674, 2.030]` |
| **Adjectives (`adjectives`)** | `' sturdy'` (45652), `' velvet'` (72426), `' crisp'` (42954), `' faint'` (38678) | `' vivid'` (43120), `' shallow'` (26682) | `[1.649, 1.877]` |
| **Function Words (`function`)** | `' however'` (4869), `' whereas'` (20444), `' beneath'` (24923), `' despite'` (8994) | `' instead'` (4619), `' beyond'` (7953) | `[1.750, 1.951]` |

### 1.2 Fresh 12 Independent Stem Triples — Pure ChatML Role-Template Contrast
All `12` stems (`6` narrative + `6` analytical) end with `":"` and have **0 token or substring overlap** with C0/C1 prompts or C2 targets:
- **`ShortBare` (`12–14` tok, mean `13.33`):** `{stem}`
- **`LongBare` (`46–52` tok, mean `48.33`):** `User instruction: Complete the final line below with one fitting word. Context: {ctx}\nAssistant final response:\n{stem}`
- **`ChatML` (`47–53` tok, mean `49.33`):** `<|im_start|>user\nComplete the final line below with one fitting word. Context: {ctx}<|im_end|>\n<|im_start|>assistant\n{stem}`
- **Verified invariants (`12/12`):** (i) **Zero stem duplication:** `{stem}` appears **exactly once** in all 3 formats (`count == 1`); instruction and `{ctx}` are word-for-word identical between `LongBare` and `ChatML`, isolating purely the **ChatML role-template delimiters**. (ii) **Exact suffix identity:** `ids_long[-T_s:] == ids_short` AND `ids_chat[-T_s:] == ids_short` on `12/12`. (iii) **Matched length:** `T_long - T_chat == -1` token on all `12/12` (`max |ΔT| = 1`).
- **Prompt allocation:** `C2.1` & `C2.3` use **`4 LongBare + 4 ChatML`** (`narr_0, narr_1, anal_0, anal_1`). `C2.2` uses ALL `12` triples. `C2.4` uses `2 LongBare + 2 ChatML` (`narr_0, anal_0`).

| Stem ID | Style | `T_s` | `T_long` (`pre`) | `T_chat` (`pre`) | `ΔT` | Trailing Stem (Bit-Identical Across All 3 Formats) |
|---|---|---:|---:|---:|---:|---|
| `narr_0` | narrative | `12` | `48` (`36`) | `49` (`37`) | `-1` | `"Inside the small drawer, the card displayed a single word:"` |
| `narr_1` | narrative | `13` | `47` (`34`) | `48` (`35`) | `-1` | `"Across the chalk slate, the final word written in dust was:"` |
| `narr_2` | narrative | `12` | `48` (`36`) | `49` (`37`) | `-1` | `"Under the brass lantern, the folded slip named one word:"` |
| `narr_3` | narrative | `14` | `52` (`38`) | `53` (`39`) | `-1` | `"Beside the iron key, the stamped wooden tag showed one word:"` |
| `narr_4` | narrative | `14` | `51` (`37`) | `52` (`38`) | `-1` | `"Above the oak mantel, the framed note held a single word:"` |
| `narr_5` | narrative | `13` | `47` (`34`) | `48` (`35`) | `-1` | `"Within the leather pouch, the small brass coin bore one word:"` |
| `anal_0` | analytical | `14` | `49` (`35`) | `50` (`36`) | `-1` | `"In the test ledger, the target entry listed in row four was:"` |
| `anal_1` | analytical | `14` | `48` (`34`) | `49` (`35`) | `-1` | `"From the sample index, the marked label shown at the end was:"` |
| `anal_2` | analytical | `14` | `49` (`35`) | `50` (`36`) | `-1` | `"On the control sheet, the verified item printed in box two was:"` |
| `anal_3` | analytical | `14` | `47` (`33`) | `48` (`34`) | `-1` | `"Within the draft manual, the underlined term at the end was:"` |
| `anal_4` | analytical | `13` | `48` (`35`) | `49` (`36`) | `-1` | `"In the audit summary, the flagged token in column three was:"` |
| `anal_5` | analytical | `13` | `46` (`33`) | `47` (`34`) | `-1` | `"From the coding chart, the final symbol on line six was:"` |

- **Paired C0-`":"` Anchor Ratio Prediction (`16` clean forwards, non-gating):** Re-runs C0's exact 8 `" a"`-ended prompts (`orig_a`) paired with 8 `":"`-ended versions (`colon`). Provenance: `c0_receipt.json.gz` (`L24 = 9.325704`, `L27 = 10.237432`) vs. `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz` (`L24 = 7.210327`, `L27 = 7.473792`). In-run paired ratio `median(pmn_colon / pmn_orig_a)`: **`L24` pt `0.7732` `[0.70, 0.85]`**; **`L27` pt `0.7300` `[0.65, 0.82]`**. Bare-only length-matched ratio reported alongside pooled ratio (`ChatML` drops 1 token).

### 1.3 `0 CU` Frozen-Lens Predictions Across All 5 Arms
> **Framing (`F2`):** Table 1.3 evaluates frozen unit-norm writes `w` through the 64-doc linear RelP surrogate `U_d @ (R_out @ w)`. In Stage `C2.1`, Holm rows `1–4` perform **on-model arm comparisons** (primary on `heldout_16×8`), while separate Gate **`CAL-C2.1`** directly tests **absolute frozen-lens→model calibration** against Table 1.3 so a common lens scale error cannot masquerade as transfer.

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

### 1.4 `0 CU` OLMo-3 `pre_norm` Low-`γ` Mask Geometry (`PRENORM_LAYERS = [21, 24, 27]`)
Dropping `4` smallest `|γ|` dims in `post_feedforward_layernorm` (`ff`) and `16` in `post_attention_layernorm` (`attn`, `L = 24`) eliminates RMSNorm blowup (`mean eff = (1/rms(γ)) / ‖v_masked / γ‖_2`):

| Layer & Branch | `drop_k` | `min \|γ\|` | Unmasked Eff | Unmasked Worst Blowup | Masked Eff | Masked Worst Blowup | Masked / Unmasked | Masked `cos_ret` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`L21` `ff`** | `4` | `0.000254` | `24.53%` | `21.84×` | **`99.21%`** | `1.02×` | `4.044×` | `0.99955` |
| **`L24` `ff`** | `4` | `0.001564` | `49.36%` | `5.93×` | **`98.85%`** | `1.03×` | `2.003×` | `0.99944` |
| **`L24` `attn`** | `16` | `0.008362` | `75.38%` | `1.51×` | **`99.43%`** | `1.02×` | `1.319×` | `0.99778` |
| **`L27` `ff`** | `4` | `0.000148` | `12.89%` | `35.36×` | **`98.58%`** | `1.04×` | `7.650×` | `0.99951` |

> **Decision on `L24 m70a30` (`F3`):** Because the frozen linear delivered ratio is `(0.7 * 0.9885 + 0.3 * 0.9943) / 0.9885 = 1.0018×` (`+0.18%`), a binary `1.10×` confirmatory bar sits `55×` above the linear prediction. We therefore **move `C2.4b` (`m70a30`) OUT of the confirmatory Holm family** and register it explicitly as a **secondary descriptive high-dose estimation readout** (reporting `R_target`, 90% BCa CI, `R_pack_rand` contrast, and `1.25×` vs. `0.50×` dose scaling without consuming a Holm slot).

---

## 2. Global Gate `C2.0`, 7-Test Holm Family, Formal Symbol Definitions, & Separate Gates

### Global Gate `C2.0` (Exact C1 Count Scaling `F4`)
1. **`C2.0.1` (`19` Enumerated Shams — `13` Write Shams + `6` Patch Shams — & Liveness):**
   - **Write Shams (`13`, `dose = 0.0` → exact `max |Δlogit| == 0.0`):** `C2.1_post_norm_L{18,24,27}` (`3`), `C2.3_sweep_L{18,21,24,27,30}` (`5`), `C2.4_prenorm_L{21,24,27}_ff` (`3`), `C2.4_prenorm_L24_m70a30` (`1`), `C2.4_prenorm_K3_combined` (`1`).
   - **Patch Shams (`6`, self-patch with recipient's own activations / zero-perturbation identity → exact `max |Δlogit| == 0.0`):** `C2.2_self_patch_ChatML_MLP_L19_31`, `C2.2_self_patch_LongBare_MLP_L19_31`, `C2.2_self_patch_ctrl_L4_16_MLP`, `C2.2_self_patch_ctrl_mismatch_L19_31`, `C2.2_zero_scale_ctrl_rand_iso_L19_31`, `C2.2_unit_scale_ctrl_self_attenuated_L19_31`.
   - Every active arm must produce `max |Δlogit| > 0.0`.
2. **`C2.0.2` (Telemetry & Telescoping):** `0` non-finite traces; 4-term telescoping error `p95 <= 0.08`, `p99.9 <= 0.15`.
3. **`C2.0.3` (Exact `math.ceil(c1_count * N / 1728)` Regime-N & Regime-D Linearity Gate, `F4`):** Doses scale strictly from live in-run clean `pmn`; ceilings are derived identically for BOTH populations from C1's exact registered counts (`2/1,728` `>20%` RMSNorm shrink, `26/1,728` `>12%` shrink, `12/1,728` `|fde_err| > 0.45`):
   - **Regime-N (`C2.1`, `N = 2,880` traces):** `0` capped; `<= 4` (`ceil(2*2880/1728)`) with `>20%` shrink; `<= 44` (`ceil(26*2880/1728)`) with `>12%` shrink.
   - **Regime-D (`C2.2` unpatched `3 × 12 targets × 12 pairs × 2 arms`, `N = 864` traces):** `0` capped; `<= 1` (`ceil(2*864/1728)`) with `>20%` shrink; `<= 13` (`ceil(26*864/1728)`) with `>12%` shrink; `|fde_err|` `p95 <= 0.30`; `<= 6` (`ceil(12*864/1728)`) with `|fde_err| > 0.45`.
4. **`C2.0.4` (Frozen SHA-256 Provenance):** Exact runtime verification of `r_lens.pt`, `c2_prereg_constants.json`, `c2_vector_bank.safetensors`, and the registered `OLMO3-TRANSPORT-C2-DESIGN.md` SHA-256 pinned at sign-off commit.

### Formal On-Model Outcome Definitions (`F1`, Mirrored in `c2_prereg_constants.json["holm_symbol_definitions"]`)
For `C2.1` source target `i`, arm `a ∈ {R_mf, R_ridge_raw, R_ridge_centered, R_ridge_shrink, logit_lens}`, layer `L ∈ {18, 24, 27}`, and prompt `p ∈ {1..8}`:
- **`Delta_logit_a(j; i, L, p)`:** Exact on-model logit shift `logit_int(j) - logit_clean(j)` of token `j` when writing arm `a`'s unit vector for source `i` at layer `L`, `positions="last"`, dose `0.75 * pmn[L]`.
- **`diag_a(i)` (`diag_shk`, `diag_raw`):** Prompt mean `diag_a(i, L) = mean_{p∈8}[Delta_logit_a(i; i, L, p)]`, aggregated via equal-weight layer mean **`diag_a(i) = mean_{L∈{18,24,27}}[diag_a(i, L)]`**.
- **`ho_cross_a(i)` (`ho_cross_raw`, `ho_cross_ctr`, `ho_cross_shk`, `ho_cross_ll`):** Strictly **ABSOLUTE** value (matching `cross_abs`; never signed, so `log` is always real-valued) using a **ratio of means** over held-out cross-stratum receivers and prompts with an **`ε = 0.25` logit denominator guard**:
  `ho_cross_a(i, L) = (mean_{p∈8, j∈heldout_8: strata(j)≠strata(i)} |Delta_logit_a(j; i, L, p)|) / max(|diag_a(i, L)|, 0.25)`,
  then equal-weight layer mean **`ho_cross_a(i) = max(mean_{L∈{18,24,27}}[ho_cross_a(i, L)], 1e-6)`**. **Guard rule:** log `n_diag_guarded` (count of `(i, L)` with `|diag_a(i, L)| <= 0.25`); if `> 2` of `16` `primary_16` sources are guarded in any arm, fail closed as **`C2.1 = DEGENERATE_DIAG`**.
- **`same_over_cross_a`:** Ratio of means over all `24` targets: `mean_i[mean_L(mean_{p, j≠i, same_stratum}|Delta_logit_a| / max(|diag_a(i,L)|,0.25))] / mean_i[mean_L(mean_{p, diff_stratum}|Delta_logit_a| / max(|diag_a(i,L)|,0.25))]`.
- **`d_i(0.50)`:** Denominator-free paired closure margin on `C2.2` stem pair `i ∈ {1..12}` (defined in Pillar 2 below).
- **`Delta_logit_masked`, `Delta_logit_unmasked`:** Prompt-and-dose-averaged exact target logit shifts for `L24.prenorm_ff_masked` and `L24.prenorm_ff_unmasked` at exact matched pre-norm write norm (pooled `0.50×` and `1.25×` live `pmn`).

### Unified 7-Test Reject-the-Null Holm-Bonferroni Family (`m = 7`, `B5`, `F3`)
Every member is a one-sided reject-the-null test (`α = 0.05`, deterministic `PERM_SEED = 2026112501`, `BCA_SEED = 2026112502`, `10,000` BCa resamples, fail-closed `CI_low <= point <= CI_high`). All `q05` bars below are one-sided lower 5th-percentile BCa bounds; the two-sided `[5th, 95th]` BCa interval on `C2.2` closure is logged as a conservative safety conjunction alongside the one-sided sign test.
- **Step-down thresholds (`k = 1..7`, `α/(8-k)`):** `[0.007143, 0.008333, 0.010000, 0.012500, 0.016667, 0.025000, 0.050000]`.
- **Exact sign-flip rules:** `N = 16` (`primary_16`): exact `2^16 = 65,536` flips (`p_min = 0.000015`); `N = 24`: `100,000` flips (`p_min = 1e-5`); `N = 12` pairs (`C2.2`): exact `2^12 = 4,096` one-sided flips (`12/12 → p = 0.000244`, `11/12 → p = 0.003174 < α/7 = 0.007143`).

| # | Test ID | Unit (`N`) | Per-Unit Statistic `x_i`, Null `H0`, & One-Sided Tail `H1` | Registered Effect-Size & BCa Bar |
|---:|---|---:|---|---|
| 1 | `p_C2.1_ctr_vs_raw` | `16` (`primary_16` → `heldout_8`) | `x_i = log(ho_cross_raw(i) / ho_cross_ctr(i))`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 2.0×` & `q05 > 1.50×` |
| 2 | `p_C2.1_shk_vs_ctr` | `16` (`primary_16` → `heldout_8`) | `x_i = log(ho_cross_ctr(i) / ho_cross_shk(i))`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 1.15×` & `same_over_cross` ratio `>= 1.15×` (`q05 >= 1.05×`) |
| 3 | `p_C2.1_shk_vs_ll` | `16` (`primary_16` → `heldout_8`) | `x_i = log(ho_cross_ll(i) / ho_cross_shk(i))`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 1.35×` & `q05 > 1.15×` (pred `1.73×`) |
| 4 | `p_C2.1_gain_ret` | `24` targets | `x_i = log(diag_shk(i) / diag_raw(i)) - log(0.75)`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 0.80×` & `q05 >= 0.75×` |
| 5 | `p_C2.2_bare_into_chat` | `12` pairs | `d_i(0.50) = s·[(M(Chat,i)-M(B→C,i)) - 0.50·(M(Chat,i)-M(Long,i))]`; `H0: mean <= 0`, `H1: > 0` | Point `∈ [0.75, 1.30]` & `[q05, q95] ⊂ [0.55, 1.50]` (gated on `\|G_gap\| >= 0.30`) |
| 6 | `p_C2.2_chat_into_bare` | `12` pairs | `d_i(0.50) = s·[(M(C→B,i)-M(Long,i)) - 0.50·(M(Chat,i)-M(Long,i))]`; `H0: mean <= 0`, `H1: > 0` | Point `∈ [0.75, 1.30]` & `[q05, q95] ⊂ [0.55, 1.50]` (gated on `\|G_gap\| >= 0.30`) |
| 7 | `p_C2.4a_mask_vs_unmask` | `24` targets | `x_i = log(Delta_logit_masked(i) / Delta_logit_unmasked(i))` (matched pre-norm norm); `H0: mean <= 0`, `H1: > 0` | `geomean >= 1.40×` & `q05 > 1.20×` (pred `2.003×`) |

### Pillar Protocols, Registered Gates Outside Holm (`CAL-C2.1`, `EQ-C2.2`, `EQ-C2.4c`), & Secondary Readouts
1. **`C2.1` — Arm Comparison (Holm 1–4) & Separate Gate `CAL-C2.1` (Absolute Frozen-Lens Transfer & `μ_U` Mechanism, `F2`, `F10`):**
   - **Gate `CAL-C2.1` (zero extra forwards):** Across the `16×8 × 3L` held-out pairs, regress prompt-averaged scaled on-model shifts `Delta_logit_a(j; i, L) / (0.75 * pmn[L] / r_c)` on frozen linear predictions `u_j · (R_out[L] w_{a,i})`. Absolute frozen-lens transfer holds iff: (i) on-model `ho_cross_raw ∈ [0.04, 0.25]` (around frozen `0.102`), (ii) held-out regression on `R_ridge_raw` achieves **`R^2 >= 0.40`** with **slope `∈ [0.45, 1.60]`**, (iii) `diag_shk >= 0.50 × frozen_diag_gain_shk * (0.75 * pmn / r_c)`, AND (iv) 3-layer mean `|cos(v1_right(M_off), u_j · μ̂_U)|` satisfies `R_ridge_raw >= 0.70` and `R_ridge_shrink <= 0.45` (frozen `0.9719 → 0.2975`). If Holm `1–3` pass while `CAL-C2.1` fails, C2.1 reports **`RELATIVE_ARM_WIN_ONLY`** (honestly distinguishing relative whitening superiority from absolute lens calibration).
   - *Secondary descriptive readout (non-gating, `F10`):* Full `24×24` 5-arm ordering (`Shrink < Centered < logit_lens < Raw < R_mf`) and per-prompt `x_i` breakdowns (`F6`).
2. **`C2.2` — Exact Formulas, Donor Run Rule (`F11`), & Separate Gate `EQ-C2.2`:**
   - **L18 write & recipient-matched Regime-D dose:** ALL `9` `C2.2` conditions apply an `L18` `positions="all"` post-norm write (`L18.R_mf[t]` or `L18.direct_matched[t]`, `K_DM_DRAWS = 1`) on top of any `L19–L31` / `L4–L16` patch. With `u_t = lm_head.weight[id_t] * model.norm.weight`, `r_c(recipient, i) = sqrt(mean(h_final^2) + eps)` on the unsteered unpatched recipient, and `u_t · w_Rmf(L18, t) ∈ [0.7165, 0.8777]`, each recipient is dosed to `direct_linear_analytic = +3.0`: `dose(recipient, t, i) = 3.0 * r_c(recipient, i) / (u_t · w_Rmf(L18, t))`, cap `3.0 * post_mlp_norm[18]`.
   - **Explicit donor run & arm rule (`F11`):** Donor `L19–L31` (or `L4–L16`) `post_feedforward_layernorm` activations are captured from the **donor format's unpatched run at the exact same `L18` write arm (`R_mf` or `dm0`) and same target `t`**, at the trailing `T_s` identical stem positions (`-T_s:`), and substituted per arm (leaving `33–39` prefix tokens untouched).
   - **Exact `r_all` & signed closure:** `gap_all(C, t, i) = Delta_logit_exact(C, R_mf, t, i) - Delta_logit_exact(C, dm0, t, i)`; frozen `pred_gap(t) = 3.0 * (pred_ret_R_mf(t) - pred_ret_dm0(t)) ∈ [5.1350, 6.2887]`. Pair-level ratio of means `M(C, i) = mean_{t∈12}[gap_all(C, t, i)] / mean_{t∈12}[pred_gap(t)]`; `G_gap = mean_{i∈12}[M(ChatML, i) - M(LongBare, i)]`, `s = sign(G_gap)`. Gate `|G_gap| >= 0.30` (90% BCa CI excluding `0`); else check `G_len = mean_i[M(LongBare, i) - M(ShortBare, i)]` (`>= 0.30` & CI `≠ 0` → **`LENGTH_DRIVEN_NOT_TEMPLATE`**, else **`NO_FORMAT_GAP_AT_MATCHED_LENGTH`**). Pooled unclipped signed closure: `closure_{B→C} = mean_i[M(ChatML, i) - M(patch_B→C, i)] / G_gap` and `closure_{C→B} = mean_i[M(patch_C→B, i) - M(LongBare, i)] / G_gap`.
   - **Gate `EQ-C2.2` (outside Holm; all 5 clauses required for C2.2 specificity):** (1) `ctrl_L4_16_MLP_LongBare_to_ChatML` `|closure| <= 0.25`, `q95 <= 0.40`; (2) `ctrl_rand_iso_same_site_L19_31` `|closure| <= 0.25`, `q95 <= 0.40`; (3) `ctrl_mismatch_prompt_MLP_L19_31` `|closure| <= 0.35`, `q95 <= 0.50`; (4) `ctrl_self_attenuated_L19_31` (self `L19–L31` MLP outputs rescaled per layer to `LongBare` L2 norm) `|closure| <= 0.25`, `q95 <= 0.40`; (5) `bare_into_chat` `closure_open >= 0.65`, `|closure_open - closure_func| <= 0.35`, and median unsteered patched `KL(p_patched || p_ref) <= 0.25` across `72` forwards.
3. **`C2.3` — `5`-Layer Post-Erasure Sweep (`SWEEP_LAYERS = [18, 21, 24, 27, 30]`):**
   - `ΣMLP(L, t, p) = sum_{l=L+1}^{31} (u_t · (mlp_out_l^{int}(last) - mlp_out_l^{clean}(last))) / r_c(p)` on `logit_lens` (`positions="last"`, `0.75× pmn[L]`). Provenance from `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz` (`block2a_regime_N_last[L]['logit_lens']`): `L18` (`-3.0791, -1.1021, gap -1.9770`) vs. `L24` (`-0.7432, -1.1016, +0.3584`) and `L27` (`-1.0024, -1.0707, +0.0683`).
   - `post_erasure(L)` holds iff `E_open(L) >= -1.20` AND `|E_open(L) - E_func(L)| <= 0.60` AND `ret_open(L) > ret_rand(L)` (`M_RAND_ISO_DRAWS = 8`). Earliest contiguous run of `>= 2` layers in `{18..30}` selects Stage C3's compilation band; if none, `C2.3` emits **`C2.3_UNLOCALIZED`** (distinct from confirmed `{24, 27}`, `F8`) before falling back to `{24, 27}`.
4. **`C2.4` — Literal `pre_norm` Masking (Holm 7), Gate `EQ-C2.4c`, & Secondary Readouts (`F3`, `F8`, `F9`, `F10`):**
   - **`C2.4a` (Holm 7):** `L24.prenorm_ff_masked` vs. `L24.prenorm_ff_unmasked` at exact matched pre-norm write norm (`0.50×`, `1.25×` `pmn`). *Secondary descriptive diagnostic (`F9`, `F10`):* `L24.prenorm_ff_unmasked_postmatched` (boosted by `1/unmasked_eff`, precondition `capped_rate == 0`; if cap binds, diagnostic fails closed — dead `OR capped_rate > 0` clause deleted!).
   - **`C2.4b` (Secondary descriptive high-dose estimation, `F3`, `F7`, `F10`):** Reports `L24.m70a30 / L24.prenorm_ff_masked` (`R_target` + 90% BCa CI) at `1.25×` and `0.50×`, alongside `R_pack_rand` (`‖Δlogits‖_2` norm ratio AND per-target `|Δlogit|` ratio on `L24.m70a30_rand_masked / L24.prenorm_ff_rand_masked`).
   - **Gate `EQ-C2.4c` (`K=3` Median Additivity TOST at `0.50×` `pmn`, `F8`):** `rel_int_t = (Δ_K3 - (Δ_L21 + Δ_L24 + Δ_L27)) / max(|Δ_L21 + Δ_L24 + Δ_L27|, 0.1)`, scored via `median(rel_int)` across `N = 24` targets with `n_floored <= 2`. Three-way verdict → Stage C3 action (`F8`): **`ADDITIVE`** (`|median| <= 0.20`, `90% BCa CI ⊂ [-0.30, +0.30]`, `n_floored <= 2`) licenses `K=3` simultaneous compilation in C3; **`NON_ADDITIVE`** (`90% BCa CI` entirely outside `[-0.30, +0.30]`) restricts C3 to single-layer `L24`; **`INCONCLUSIVE`** requires a 1-dose C3 pilot check. *(The `1.25×` dose of `L21/L24/L27/K3` = `384` forwards is explicitly disclosed as secondary high-dose saturation calibration for C3, outside confirmatory gates.)*

### Named Limitations & Scope Disclosures (`F5–F10`)
- **`F5` (Power `sd` grids):** Power-table `sd` grids (`0.10–0.25`, `0.06–0.15`, `0.12–0.30`) are sensitivity grids (not empirically derived from C1); `C2.1` rows `1–4` and `C2.4a` row `7` are not separately power-simulated (note: row `2`'s `1.15×` bar vs. `1.30×` prediction is a `~50%` boundary at `true = 1.15×`, reaching `~85%–100%` at `1.20×–1.30×`).
- **`F6` (Shared-prompt correlation in `C2.1`):** Exact `2^16` (and `100,000`-flip `N = 24`) target sign-flips treat targets as exchangeable conditional on the shared `8`-prompt bank; per-prompt `x_i` breakdowns (`8` prompts) are reported alongside target-level `p` to expose any shared-prompt ICC.
- **`F7` (`C2.4b` packing functional):** `C2.4b` (descriptive per `F3`) compares target-logit ratio `R_target` against whole-vocab norm ratio `R_pack_rand` (different functionals since random-direction target logits are zero-centered); both `‖Δlogits‖_2` and per-target `|Δlogit|` ratios are reported.
- **`F8` & `F10` (Calibration forwards & accounting):** `384` `C2.4c` `1.25×` forwards + `C2.4b` (`768` fwds) + `C2.4a unmasked_postmatched` (`192` fwds) are explicitly registered as secondary estimation / calibration readouts outside the `m = 7` Holm family and the `3` registered gates (`CAL-C2.1`, `EQ-C2.2`, `EQ-C2.4c`).

---

## 3. Full-Conjunction Monte-Carlo Power (`POWER_SIM_SEED = 2026112506`) & Budget

### 3.1 Pre-Data Sensitivity Power (`c2_prereg_constants.json["power_simulations"]`)
- **`C2.2` Full Conjunction (`N = 12` pairs, Holm `α/7 = 0.007143`, `0.75`/`1.30` are the `~50%` point-bar boundary edges):**
  - `sd = 0.10`: `0.50 → 0.000`, `0.65 → 0.000`, `0.70 → 0.055`, **`0.75 → 0.508`**, **`0.85 → 0.9995`**, **`1.00 → 1.000`**, `1.30 → 0.508`, `1.40 → 0.000`.
  - `sd = 0.15`: `0.50 → 0.000`, `0.65 → 0.012`, `0.70 → 0.141`, **`0.75 → 0.483`**, **`0.85 → 0.9925`**, **`1.00 → 1.000`**, `1.30 → 0.510`, `1.40 → 0.012`.
  - `sd = 0.25`: `0.50 → 0.000`, `0.65 → 0.071`, `0.70 → 0.218`, **`0.75 → 0.468`**, **`0.85 → 0.888`**, **`1.00 → 0.999`**, `1.30 → 0.495`, `1.40 → 0.086`.
- **`EQ-C2.4c` `K=3` Median TOST — 3-Way Outcomes (`P_additive / P_non_additive / P_inconclusive`, `N = 24`):**
  - `sd = 0.12` (eff margin `≈ 0.250`): `true 0.00 → 1.000 / 0.000 / 0.000`; `0.10 → 1.000 / 0.000 / 0.000`; `0.20 → 0.484 / 0.000 / 0.517`; `0.35 → 0.000 / 0.516 / 0.485`; `0.50 → 0.000 / 1.000 / 0.000`.
  - `sd = 0.20` (eff margin `≈ 0.216`): `true 0.00 → 1.000 / 0.000 / 0.000`; `0.10 → 0.968 / 0.000 / 0.032`; `0.20 → 0.441 / 0.000 / 0.559`; `0.35 → 0.001 / 0.266 / 0.734`; `0.50 → 0.000 / 0.985 / 0.016`.
  - `sd = 0.30` (eff margin `≈ 0.174`): `true 0.00 → 0.972 / 0.000 / 0.029`; `0.10 → 0.819 / 0.000 / 0.181`; `0.20 → 0.332 / 0.004 / 0.665`; `0.35 → 0.007 / 0.187 / 0.807`; `0.50 → 0.000 / 0.845 / 0.155`.

### 3.2 Exact Compute Budget (`0 CU` Used Until Registration + Code PASS & Elise Green Light)

| Block | Exact Enumeration | Short `B=1` Forwards |
|---|---|---:|
| Clean (`36`) + Write Shams (`13`) + Patch Shams (`6`) + C0-`":"` Pairs (`16`) | Baseline + Gate `C2.0` (`19` shams) + `C2.0_anchor_pred` | `71` |
| **Block 1 (`C2.1`)**: `3`L × `5` arms × `24` targets × `8` prompts (`4` LongBare + `4` ChatML) | Holm 1–4 + Gate `CAL-C2.1` + secondary ordering | `2,880` |
| **Block 2 (`C2.2`)**: `9` conds × `12` targets × `12` pairs × `2` (`R_mf` + `dm0`) + `72` unsteered KL | Holm 5–6 + Gate `EQ-C2.2` + `ShortBare` attribution | `2,664` |
| **Block 2b (`C2.3`)**: `5`L (`18..30`) × (`24` `logit_lens` + `8` `rand_iso`) × `8` prompts | Post-erasure `E_open`, `|gap|`, & `ret_open > ret_rand` | `1,280` |
| **Block 3 (`C2.4`)**: `9` arms × `24` targets × `4` prompts × `2` doses (`0.50×`, `1.25×`) | Holm 7 (`C2.4a`) + Gate `EQ-C2.4c` + secondary `C2.4b`/calibration | `1,728` |
| **Total Stage C2 v4** | Estimated `8–13 min` on Colab L4/A100 | **`8,623`** (`~0.45–1.05 CU`) |
