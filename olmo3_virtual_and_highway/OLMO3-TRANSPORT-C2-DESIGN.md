# Stage C2 Design (Draft v3 — Round-2 Review Folded, Final Registration Candidate): Held-Out Lens→Model Whitening Transfer, Pure ChatML-Template `ΣMLP` Causal Patching, Post-Erasure Sweep, & `pre_norm` Mask/Split/Additivity on OLMo 3 7B

**Status:** **DRAFT v3 — NOT YET REGISTERED (`0 CU`, zero Stage C2 GPU runs exist).** Authored by Vesper (`neon`, research lead) on clean branch `vesper-c2-wsp` (`fee95ce3`) for Council Round-3 registration sign-off (`open-olmo3-7b-virtual-neurons-2026-11-22`, folding all Round-2 cold design review items `B1–B6`, `M1–M10`, `m1–m8` from `REVIEW-design-ds-r2.md` / `#115`).  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `use_cache=False`, `B=1`.  
**Stage Split (`C2` vs. `C3 / WSP v1`):**  
- **Stage C2 (this study, `8,623` short `B=1` forwards, `0%` unread forwards, `~0.45–1.05 CU`):** Deterministic single-forward mechanistic & `pre_norm` compilation verification across four linked pillars (`C2.0–C2.4`).  
- **Stage C3 (`WSP v1` Behavioral Generation & 3-Column Public Demo, gated after C2):** Uses C2's confirmed winning write operator, post-erasure layer band, and masked dual-branch / multi-layer `pre_norm` parameterization to compile **Persona** (primary demo arm) and **No-Markdown** (separate mechanical rule arm, never combined) literal weights scored by Part 168 decode word-ban, implied content, 4-turn follow-through, Part 213 copula controls, untuned math/code/chat/refusals, and dual blind readers (`κ ≥ 0.60`) — gated on Elise's explicit `11-07` weight-release policy approval before any edited weights go public.

### Reproducible `0 CU` Pre-Data Artifacts (Double-Run Byte-Identical `cmp` Verified)
- Builder: [`olmo3_virtual_and_highway/build_olmo3_c2_constants.py`](build_olmo3_c2_constants.py)
- Constants (`c2-prereg-constants-v3`): [`olmo3_virtual_and_highway/c2_prereg_constants.json`](c2_prereg_constants.json) (`sha256 = 7b2ffc15a9a9ad7f9e999411a138b445245effdac262c65f9de4d64a8c786ee6`)
- Frozen vector bank (`work/olmo3_c2/c2_vector_bank.safetensors`, untracked `9.6 MB`, `34` tensors, 100% draw-level 1-to-1 with active arms): `sha256 = 57c3f63604aae4c21dce0385f71833fd82f67ae0dff52f6f3af23f5b5521d387`
- Reused 64-doc RelP R-lens (`r_lens.pt`): `sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`
- PyTorch / tokenizer invariant test suite: [`olmo3_virtual_and_highway/test_olmo3_c2_constants.py`](test_olmo3_c2_constants.py)
- Pure stdlib (`/usr/bin/python3 -I`, row-strict table verifier, zero third-party deps): [`olmo3_virtual_and_highway/test_olmo3_c2_design_stdlib.py`](test_olmo3_c2_design_stdlib.py)

---

### Prior Receipts & Literature Context
- **Stage C0 (`c96ebfdb`) & C0b (`112450e4`) & C1 (`45b9208c`):** C0 confirmed raw unembedding contraction (`C0.3`, `p = 3.1e-5`); C0b mapped the linear `L18/24/27` `"all"` band vs. nonlinear `L12`; C1 (`1.71 CU` A100) passed all 5 Regime-D linearity clauses (`0/1,728` capped, `0 ≤ 2` `>20%`, `5 ≤ 26` `>12%`, `fde_p95 = 0.1853`, `2 ≤ 12` `>0.45`) but failed `C1.0.3` on stale anchors × genuine `-22.7% / -27.0%` `post_mlp_norm` drops on `":"`-ended prompts (`9.325704 → 7.210327` at `L24`; `10.237432 → 7.473792` at `L27`, verbatim from `c0_receipt.json.gz` / `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz`, `m2`), blocking `C1.1–C1.4`. Exploratory C1 readouts exposed the `R_ridge` `1.92×` bleed floor (`C1.3`), the **`L18` `r_all` gap (`+0.604`, `Bare 1.3267` vs. `ChatML 1.9307`, attributed in C1 to an all-position format×length interaction and decomposed in Phase 0b as `2.8×` larger `ΣMLP`, `m1`)**, and the `L18 → L24` `logit_lens` erasure flip (`C1.4`).
- **Motivating literature (none carries a registered Stage C2 parameter):** Park, Choe, & Veitch (`ICML 2024`, unembedding centering & causal whitening); MIT J-Lens Theory (`arXiv:2610.31955`, corpus-averaged Lyapunov modes); Gurnee et al. (`arXiv:2607.15495`, workspace ignition); Stolfo et al. (`arXiv:2406.16254`, frequency/entropy neurons); Lee et al. (`ICML 2024` / `EMNLP 2025`, DPO opposing MLP writes); Everett Lacey (`everettlacey.com/gemma-terse-by-hand/`, Nov 2026, low-`γ` `pre_norm` masking & `m70a30` dual-branch split); Steer2Edit (`arXiv:2602.09870`), TFTV (`arXiv:2609.09054`), Shone & Dunlap (`alignment.anthropic.com/2026/neuron-basis-theory/`).

---

## 1. Fresh Units (`0` Overlap with C0/C0b/C1/Personas) & `0 CU` Frozen-Lens Transfer Baseline

### 1.1 Fresh 24-Target Mixed-POS Bank (`targets_24_spec_sha256 = 1074f160064b9f17fdf42bb1f3c0fc8de05d87168e3428cc58ebebfdcfe807f7`)
Strict single-token leading-space roundtrip in OLMo-3 tokenizer, **`0/24` overlap** with C0's `16` nouns, C1's `24` targets, or `8` persona tokens. Each stratum of `6` is deterministically split (`M2`) into **`primary_16`** (`4` per stratum, out-of-target `C2.1` sources) and **`heldout_8`** (`2` per stratum, out-of-target `C2.1` receivers). **`C2.2`** uses the balanced **`12`-target** subset (first `3` of each stratum: indices `[0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]`, `9` open-class + `3` function):

| Stratum | `primary_16` (`4` per stratum; first `3` in `C2.2`) | `heldout_8` (`2` per stratum) | `‖u_t‖` range |
|---|---|---|---:|
| **Concrete Nouns (`nouns`)** | `' candle'` (38899), `' barrel'` (20929), `' feather'` (50876), `' ribbon'` (45667) | `' basket'` (14351), `' glacier'` (94867) | `[1.621, 1.942]` |
| **Base Verbs (`verbs`)** | `' polish'` (45129), `' stumble'` (75237), `' vanish'` (92227), `' mend'` (45876) | `' drift'` (34738), `' carve'` (80737) | `[1.674, 2.030]` |
| **Adjectives (`adjectives`)** | `' sturdy'` (45652), `' velvet'` (72426), `' crisp'` (42954), `' faint'` (38678) | `' vivid'` (43120), `' shallow'` (26682) | `[1.649, 1.877]` |
| **Function Words (`function`)** | `' however'` (4869), `' whereas'` (20444), `' beneath'` (24923), `' despite'` (8994) | `' instead'` (4619), `' beyond'` (7953) | `[1.750, 1.951]` |

### 1.2 Fresh 12 Independent Stem Triples — Pure ChatML Role-Template Contrast (`B1`, `B2`, `B3`, `B4`, `m7`)
All `12` stems (`6` narrative + `6` analytical) end with `":"` and have **0 token or substring overlap** with C0/C1 prompts or C2 targets. Every triple eliminates both the length confound AND the instruction/stem-duplication confound (`B3`):
- **`ShortBare` (`12–14` tok, mean `13.33`):** `{stem}`
- **`LongBare` (`46–52` tok, mean `48.33`):** `User instruction: Complete the final line below with one fitting word. Context: {ctx}\nAssistant final response:\n{stem}`
- **`ChatML` (`47–53` tok, mean `49.33`):** `<|im_start|>user\nComplete the final line below with one fitting word. Context: {ctx}<|im_end|>\n<|im_start|>assistant\n{stem}`
- **Verified invariants (`12/12`):** (i) **Zero stem duplication (`B3`):** `{stem}` appears **exactly once** in all 3 formats (`count == 1`); instruction and context clause `{ctx}` are **word-for-word identical** between `LongBare` and `ChatML`, isolating purely the **ChatML role-template delimiters**. (ii) **Exact suffix identity (`B1`):** `ids_long[-T_s:] == ids_short` AND `ids_chat[-T_s:] == ids_short` on `12/12`. (iii) **Matched length (`B2`):** `T_long - T_chat == -1` token on all `12/12` (`max |ΔT| = 1`).
- **Prompt allocation (`m7` length-matched everywhere):** `C2.1` & `C2.3` use **`4 LongBare + 4 ChatML`** (`narr_0, narr_1, anal_0, anal_1`, eliminating the `3.1×` length confound even in `C2.1`/`C2.3`!). `C2.2` uses ALL `12` triples. `C2.4` uses `2 LongBare + 2 ChatML` (`narr_0, anal_0`).

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

- **Paired C0-`":"` Anchor Ratio Prediction (`16` clean forwards, non-gating):** Re-runs C0's exact 8 `" a"`-ended prompts (`orig_a`) paired with 8 `":"`-ended versions (`colon`) in the same run. Provenance: `c0_receipt.json.gz` (`L24 = 9.325704`, `L27 = 10.237432`; NOT Exp 4B `12.1555`) vs. `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz` (`L24 = 7.210327`, `L27 = 7.473792`, `m2`). In-run paired ratio `median(pmn_colon / pmn_orig_a)`: **`L24` pt `0.7732` `[0.70, 0.85]`**; **`L27` pt `0.7300` `[0.65, 0.82]`**. Bare-only length-matched ratio (`9→9, 7→7, 13→13, 10→10`) reported alongside pooled ratio (`ChatML` drops 1 token).

### 1.3 `0 CU` Frozen-Lens Transfer Predictions Across All 5 Arms (`M1`, `M2`, `m5`)
> **Framing (`M2`):** Table 1.3 evaluates frozen unit-norm write vectors `w` through the 64-doc linear RelP surrogate `U_d @ (R_out @ w)`. Stage `C2.1` tests whether these **frozen-lens predictions** transfer to actual nonlinear **on-model** logits, with primary inference on the out-of-target **`heldout_16×8`** split (`primary_16` sources onto `heldout_8` cross-stratum receivers) and full `24×24` reported secondarily.

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
Dropping `4` smallest `|γ|` dims in `post_feedforward_layernorm` (`ff`, `L ∈ {21, 24, 27}`) and `16` in `post_attention_layernorm` (`attn`, `L = 24`) eliminates RMSNorm blowup (`mean eff = (1/rms(γ)) / ‖v_masked / γ‖_2`). Every frozen `pre_norm` tensor (`34` total across the whole bank, `9.6 MB`) maps **1-to-1 even at the draw level** (`m4`) to a registered arm:

| Layer & Branch | `drop_k` | `min \|γ\|` | Unmasked Eff | Unmasked Worst Blowup | Masked Eff | Masked Worst Blowup | Masked / Unmasked | Masked `cos_ret` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`L21` `ff`** | `4` | `0.000254` | `24.53%` | `21.84×` | **`99.21%`** | `1.02×` | `4.044×` | `0.99955` |
| **`L24` `ff`** | `4` | `0.001564` | `49.36%` | `5.93×` | **`98.85%`** | `1.03×` | `2.003×` | `0.99944` |
| **`L24` `attn`** | `16` | `0.008362` | `75.38%` | `1.51×` | **`99.43%`** | `1.02×` | `1.319×` | `0.99778` |
| **`L27` `ff`** | `4` | `0.000148` | `12.89%` | `35.36×` | **`98.58%`** | `1.04×` | `7.650×` | `0.99951` |

> **Mechanistic Note on `L24 m70a30`:** Linear delivered ratio is `(0.7 * 0.9885 + 0.3 * 0.9943) / 0.9885 = 1.0018×`, so **`C2.4b` at high dose (`1.25×` `pmn`) tests nonlinear RMSNorm saturation relief & direction-specific dual-branch advantage**, controlled against generic packing via `L24.m70a30_rand_masked / L24.prenorm_ff_rand_masked` (`M4`).

---

## 2. Global Gate `C2.0`, 8-Test Reject-the-Null Holm Family, & Separate Equivalence Gates

### Global Gate `C2.0` (Zero Cross-Prompt Norm Literals; Unified `math.ceil(rate * N)` Ceilings `M6`, `m3`)
1. **`C2.0.1` (`19` Enumerated Shams — `13` Write Shams + `6` Patch Shams — & Liveness, `m3`):**
   - **Write Shams (`13`, `dose = 0.0` → exact `max |Δlogit| == 0.0`):** `C2.1_post_norm_L{18,24,27}` (`3`), `C2.3_sweep_L{18,21,24,27,30}` (`5`), `C2.4_prenorm_L{21,24,27}_ff` (`3`), `C2.4_prenorm_L24_m70a30` (`1`), `C2.4_prenorm_K3_combined` (`1`).
   - **Patch Shams (`6`, self-patch with recipient's own activations or zero-perturbation identity → exact `max |Δlogit| == 0.0`):** `C2.2_self_patch_ChatML_MLP_L19_31`, `C2.2_self_patch_LongBare_MLP_L19_31`, `C2.2_self_patch_ctrl_L4_16_MLP`, `C2.2_self_patch_ctrl_mismatch_L19_31`, `C2.2_zero_scale_ctrl_rand_iso_L19_31`, `C2.2_unit_scale_ctrl_self_attenuated_L19_31`.
   - Every active arm must produce `max |Δlogit| > 0.0`.
2. **`C2.0.2` (Telemetry & Telescoping):** `0` non-finite traces; 4-term telescoping error `p95 <= 0.08`, `p99.9 <= 0.15`.
3. **`C2.0.3` (Unified `math.ceil(rate * N)` Regime-N & Regime-D Linearity Gate, `M6`):** Doses scale strictly from live in-run clean `pmn`; ceilings are derived identically for BOTH populations from C1's registered rates (`0.15%` `>20%` RMSNorm shrink, `1.50%` `>12%` shrink, `0.70%` `|fde_err| > 0.45`):
   - **Regime-N (`C2.1`, `N = 2,880` traces):** `0` capped; `<= 5` (`ceil(0.0015 * 2880)`) with `>20%` shrink; `<= 44` (`ceil(0.0150 * 2880)`) with `>12%` shrink.
   - **Regime-D (`C2.2` unpatched `3 × 12 targets × 12 pairs × 2 arms`, `N = 864` traces):** `0` capped; `<= 2` (`ceil(0.0015 * 864)`) with `>20%` shrink; `<= 13` (`ceil(0.0150 * 864)`) with `>12%` shrink; `|fde_err|` `p95 <= 0.30`; `<= 7` (`ceil(0.0070 * 864)`) with `|fde_err| > 0.45`.
4. **`C2.0.4` (Frozen SHA-256 Provenance):** Exact runtime verification of `r_lens.pt`, `c2_prereg_constants.json`, `c2_vector_bank.safetensors`, and the registered `OLMO3-TRANSPORT-C2-DESIGN.md` SHA-256 pinned at sign-off commit.

### Unified 8-Test Reject-the-Null Holm-Bonferroni Family (`m = 8`, `B5` — All One-Sided Rejections!)
Equivalence & negative-control claims live in their own separate **Registered Equivalence & Control Gates** below (`B5`), so **every single member of the `m = 8` Holm family is a genuine one-sided reject-the-null test** where small `p` = confirmatory success (`α = 0.05`, deterministic `PERM_SEED = 2026112501`, `BCA_SEED = 2026112502`, `10,000` BCa resamples with fail-closed containment `CI_low <= point <= CI_high`).
- **Step-down thresholds (`k = 1..8`, `α/(9-k)`):** `[0.006250, 0.007143, 0.008333, 0.010000, 0.012500, 0.016667, 0.025000, 0.050000]`.
- **Exact sign-flip rules:** `N = 16` (`primary_16`): exact `2^16 = 65,536` flips (`p_min = 0.000015`); `N = 24`: `100,000` flips (`p_min = 1e-5`); `N = 12` pairs (`C2.2`): exact `2^12 = 4,096` one-sided flips (`12/12 → p = 0.000244`, `11/12 → p = 0.003174 < α/8 = 0.00625`, `B4`). Equal-weight mean over `L ∈ {18, 24, 27}` per target before target-level inference.

| # | Test ID | Unit (`N`) | Per-Unit Statistic `x_i`, Null `H0`, & One-Sided Tail `H1` | Registered Effect-Size & CI Bar |
|---:|---|---:|---|---|
| 1 | `p_C2.1_ctr_vs_raw` | `16` (`primary_16` → `heldout_8`, `M2`) | `x_i = log(ho_cross_raw(i) / ho_cross_ctr(i))`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 2.0×` & `90% BCa q05 > 1.50×` |
| 2 | `p_C2.1_shk_vs_ctr` | `16` (`primary_16` → `heldout_8`, `M2`) | `x_i = log(ho_cross_ctr(i) / ho_cross_shk(i))`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 1.15×` & `same/cross` ratio `>= 1.15×` (`90% BCa q05 >= 1.05×`, `m6`) |
| 3 | `p_C2.1_shk_vs_ll` | `16` (`primary_16` → `heldout_8`, `M1`) | `x_i = log(ho_cross_ll(i) / ho_cross_shk(i))`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 1.35×` & `90% BCa q05 > 1.15×` (vs. free `logit_lens`, pred `1.73×`) |
| 4 | `p_C2.1_gain_ret` | `24` targets | `x_i = log(diag_shk(i) / diag_raw(i)) - log(0.75)`; `H0: mean <= 0`, `H1: > 0` | `geomean >= 0.80×` & `90% BCa q05 >= 0.75×` |
| 5 | `p_C2.2_bare_into_chat` | `12` pairs (`B1`, `B4`, `M10`) | `d_i(0.50) = s·[(M(Chat,i)-M(B→C,i)) - 0.50·(M(Chat,i)-M(Long,i))]`; `H0: mean <= 0`, `H1: > 0` | Point closure `∈ [0.75, 1.30]` & **`90% BCa CI ⊂ [0.55, 1.50]`** (gated on gap `>= 0.30`) |
| 6 | `p_C2.2_chat_into_bare` | `12` pairs (`B1`, `B4`, `M10`) | `d_i(0.50) = s·[(M(C→B,i)-M(Long,i)) - 0.50·(M(Chat,i)-M(Long,i))]`; `H0: mean <= 0`, `H1: > 0` | Point closure `∈ [0.75, 1.30]` & **`90% BCa CI ⊂ [0.55, 1.50]`** (gated on gap `>= 0.30`) |
| 7 | `p_C2.4a_mask_vs_unmask` | `24` targets (`M5`) | `x_i = log(Δlogit_masked(i) / Δlogit_unmasked(i))` at **matched pre-norm write norm** (pooled `0.50×`/`1.25×`); `H0: mean <= 0`, `H1: > 0` | `geomean >= 1.40×` & `90% BCa q05 > 1.20×` (pred `2.003×`) |
| 8 | `p_C2.4b_m70a30_vs_ff` | `24` targets (`M4`, `m8`) | `x_i = log(Δlogit_m70a30(i) / Δlogit_ff(i))` at `1.25×` `pmn`; `H0: mean <= 0`, `H1: > 0` | `R_target >= 1.10×`, `90% BCa q05 > 1.00×`, AND `R_target / R_pack_rand >= 1.05×` |

### Exact Pillar Definitions & Separate Registered Equivalence / Control Gates (`B1–B5`, `M1–M10`)
1. **`C2.1` (`L ∈ {18, 24, 27}`, `positions="last"`, `0.75×` live `pmn`, `8` length-matched prompts):**
   - Primary tests `1–3` score `primary_16` sources onto `heldout_8` cross-stratum receivers (`M2`), with full `24×24` reported secondarily across all 5 arms (`R_mf`, `Raw`, `Centered`, `Shrink`, `logit_lens`), including the full 5-arm ordering check `cross_abs(Shrink) < Centered < logit_lens < Raw < R_mf`.
   - **Direct `μ_U` mechanism check:** Equal-weight 3-layer mean `|cos(v1_right(M_off), u_j · μ̂_U)|`: `R_ridge_raw >= 0.70` AND `R_ridge_shrink <= 0.45` (frozen 3-layer mean `0.9719 → 0.2975`; `R_mf` baseline `0.9620`).
2. **`C2.2` — Exact Formulas (`B1`, `B2`), `Gate EQ-C2.2` (`B5`, `M7`), & Length-vs-Template Rule (`M3`):**
   - **L18 write & recipient-matched Regime-D dose (`B2`):** ALL `9` `C2.2` conditions apply an `L18` `positions="all"` post-norm write (`w = L18.R_mf[t]` or `w = L18.direct_matched[t]`, `K_DM_DRAWS = 1`, `m4`) on top of any specified `L19–L31` (or `L4–L16`) patch. Define `u_t = lm_head.weight[id_t] * model.norm.weight` (`u_vectors`), `r_c(recipient, i) = sqrt(mean(h_final^2) + eps)` on the unsteered unpatched recipient prompt (`LongBare`, `ChatML`, or `ShortBare`), and `u_t · w_Rmf(L18, t) ∈ [0.7165, 0.8777]` (stored per target in `c22_l18_frozen_pred_ret`). Each recipient is dosed to **exact matched `direct_linear_analytic = +3.0` logits**: `dose(recipient, t, i) = 3.0 * r_c(recipient, i) / (u_t · w_Rmf(L18, t))`, capped at `3.0 * post_mlp_norm[18]`.
   - **Exact `r_all` & signed closure formulas (`B1`):** For condition `C`, target `t ∈ C2_2_12`, pair `i ∈ {1..12}`, let `gap_all(C, t, i) = Δlogit_exact(C, R_mf, t, i) - Δlogit_exact(C, dm0, t, i)` and frozen linear prediction `pred_gap(t) = 3.0 * (pred_ret_R_mf(t) - pred_ret_dm0(t)) ∈ [5.1350, 6.2887]` (stored in `c22_l18_frozen_pred_ret`). Pair-level ratio of target means:
     `M(C, i) = mean_{t∈12}[gap_all(C, t, i)] / mean_{t∈12}[pred_gap(t)]`.
     Let `G_gap = mean_{i∈12}[M(ChatML, i) - M(LongBare, i)]` and `s = sign(G_gap)`.
     - **In-run template-gap gate & length attribution (`B2`, `M3`):** Require `|G_gap| >= 0.30` and 90% paired BCa CI of `G_gap` excluding `0`. If `|G_gap| < 0.30`, check `G_len = mean_i[M(LongBare, i) - M(ShortBare, i)]`: if `|G_len| >= 0.30` (90% CI excluding 0), return **`LENGTH_DRIVEN_NOT_TEMPLATE`**, else **`NO_FORMAT_GAP_AT_MATCHED_LENGTH`**.
     - **Exact pooled signed closure (`B1`, unclipped, per direction):**
       - `bare_into_chat` (`ctrl = unpatched_ChatML`, `ref = unpatched_LongBare`, `patched = patch_LongBare_to_ChatML_MLP_L19_31`):
         `closure_{B→C} = mean_i[M(ctrl, i) - M(patched, i)] / mean_i[M(ctrl, i) - M(ref, i)]`.
       - `chat_into_bare` (`ctrl = unpatched_LongBare`, `ref = unpatched_ChatML`, `patched = patch_ChatML_to_LongBare_MLP_L19_31`):
         `closure_{C→B} = mean_i[M(ctrl, i) - M(patched, i)] / mean_i[M(ctrl, i) - M(ref, i)]` (equivalently `mean_i[M(patched, i) - M(LongBare, i)] / G_gap`).
     - **Symmetric suffix patching (`B1`):** Replace `post_feedforward_layernorm` outputs at `L19–L31` strictly over the trailing `T_s` identical stem tokens (`-T_s:`), leaving `33–39` prefix tokens untouched.
   - **Separate Gate `EQ-C2.2` (Negative Controls, Self-Attenuation, Shape, & Clean KL, `B5`, `M7`):** Outside Holm; ALL 5 clauses must pass for C2.2 specificity:
     1. `ctrl_L4_16_MLP_LongBare_to_ChatML`: `|closure| <= 0.25`, `90% upper BCa <= 0.40`.
     2. `ctrl_rand_iso_same_site_L19_31` (same-site norm-matched isotropic residual at `L19–L31`): `|closure| <= 0.25`, `90% upper BCa <= 0.40`.
     3. `ctrl_mismatch_prompt_MLP_L19_31` (unpaired stem `LongBare` `L19–L31` MLP): `|closure| <= 0.35`, `90% upper BCa <= 0.50`.
     4. **`ctrl_self_attenuated_L19_31` (`M7`):** Replace `ChatML`'s `L19–L31` MLP outputs with its **own** `L19–L31` MLP outputs rescaled per layer to match `LongBare`'s `L19–L31` MLP L2 norm — directly testing uniform-gain attenuation: `|closure| <= 0.25`, `90% upper BCa <= 0.40`.
     5. **Numeric shape & clean-KL bar (`M7`):** On `bare_into_chat`, `closure_open >= 0.65` (9 open targets) AND `|closure_open - closure_func| <= 0.35` (vs. 3 function targets); across the `72` unsteered patched forwards (`6` patched conditions × `12` pairs), median `KL(p_patched || p_ref) <= 0.25`.
3. **`C2.3` — `5`-Layer Post-Erasure Sweep (`SWEEP_LAYERS = [18, 21, 24, 27, 30]`, `M3`, `M8`):**
   - **Exact statistic (`M8`):** `ΣMLP(L, t, p) = sum_{l=L+1}^{31} (u_t · (mlp_out_l^{int}(last) - mlp_out_l^{clean}(last))) / r_c(p)` on the **`logit_lens`** arm (`w = u_hat_t`) injected at layer `L`, `positions="last"`, dose `0.75 × pmn[L]`. Define `E_open(L) = median_{t∈open_18, p∈8}(ΣMLP)`, `E_func(L) = median_{t∈func_6, p∈8}(ΣMLP)`, `ret_open(L) = median_{t∈open_18, p∈8}(retention)`, and isotropic noise floor `ret_rand(L) = median_{d∈8, p∈8}(retention_rand_iso)` (`M_RAND_ISO_DRAWS = 8`, `M3`).
   - **Threshold provenance (`M8`):** From `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz` (`block2a_regime_N_last[L]['logit_lens']`): `L18` (`med_open = -3.0791, med_func = -1.1021, gap = -1.9770`) vs. `L24` (`-0.7432, -1.1016, +0.3584`) and `L27` (`-1.0024, -1.0707, +0.0683`). Thresholds `-1.20` and `0.60` bisect C1's heavy-erasure `L18` state and post-erasure `L24/L27` states.
   - **Selection rule:** `post_erasure(L)` holds iff **`E_open(L) >= -1.20` AND `|E_open(L) - E_func(L)| <= 0.60` AND `ret_open(L) > ret_rand(L)`**. Stage C3 band = earliest contiguous run of `>= 2` layers in `{18, 21, 24, 27, 30}`; else `C2.3 = INCONCLUSIVE` → Stage C3 defaults to `{24, 27}`.
4. **`C2.4` — Literal `pre_norm` Masking, `m70a30` Packing Control, & `Gate EQ-C2.4c` (`9` Enumerated Arms, `M3–M5`, `M9`):**
   - Actual `pre_norm` additions **before** `post_feedforward_layernorm` / `post_attention_layernorm` at `0.50×` and `1.25×` live `pmn` across `4` prompts (`2 LongBare + 2 ChatML`).
   - **`C2.4a` (`M5`):** `L24.prenorm_ff_masked` vs. `L24.prenorm_ff_unmasked` at **exact matched pre-norm write norm** (Holm test 7, predicted `2.003×`), paired with **`L24.prenorm_ff_unmasked_postmatched`** (pre-norm boosted by `1/unmasked_eff`, capped at `3.0× pmn` so linear post-norm matches `masked` — registered nonlinear RMSNorm saturation contrast: `geomean(Δlogit_masked / Δlogit_unmasked_postmatched) >= 1.15×` OR `capped_rate > 0`).
   - **`C2.4b` (`M4`):** `L24.m70a30` vs. `L24.prenorm_ff_masked` at `1.25×` `pmn` (Holm test 8), controlled against generic packing via `R_pack_rand = geomean(‖Δlogits(L24.m70a30_rand_masked)‖_2 / ‖Δlogits(L24.prenorm_ff_rand_masked)‖_2)` (requiring `R_target / R_pack_rand >= 1.05×`), plus dose-growth check `R_target(1.25×) > R_target(0.50×)` (using the `0.50×` dose!).
   - **Separate Gate `EQ-C2.4c` (`K=3` Median Additivity TOST, `B5`, `M9`):** At `0.50×` `pmn`, `rel_int_t = (Δ_K3 - (Δ_L21 + Δ_L24 + Δ_L27)) / max(|Δ_L21 + Δ_L24 + Δ_L27|, 0.1)`. Score **`median(rel_int)`** across `N = 24` targets (immune to weak-target drag) and enforce **`n_floored <= 2`** (count of targets with `|sum| < 0.1`). **Three-way verdict:**
     - **`ADDITIVE`:** `|median(rel_int)| <= 0.20` AND `90% BCa CI ⊂ [-0.30, +0.30]` AND `n_floored <= 2`.
     - **`NON_ADDITIVE`:** `90% BCa CI` entirely outside `[-0.30, +0.30]` (genuine rejection of additivity).
     - **`INCONCLUSIVE`:** `90% BCa CI` straddles `±0.30` (or `n_floored > 2`).
     *(The `1.25×` dose of `L21/L24/L27/K3` calibrates Stage C3's high-dose saturation onset `|median(rel_int(1.25×))|`.)*

---

## 3. Full-Conjunction Monte-Carlo Power (`POWER_SIM_SEED = 2026112506`, `B6`, `M9`, `m8`) & Budget

### 3.1 Pre-Data Statistical Power (`c2_prereg_constants.json["power_simulations"]`)
- **`C2.2` Full Conjunction (`N = 12` pairs: `gap_gate >= 0.30 & CI>0` + exact `2^12` one-sided `p <= 0.00625` on `H0: closure <= 0.50` + `point ∈ [0.75, 1.30]` + `90% CI ⊂ [0.55, 1.50]`, `B6`):**
  *Note: `0.75` and `1.30` are explicitly the `~50%` boundary edges of the point bar; true values outside (`0.65`, `1.40`) are rejected `91%–100%` of the time:*
  - `sd = 0.10`: `0.50 → 0.000`, `0.65 → 0.000`, `0.70 → 0.055`, **`0.75 → 0.508`**, **`0.85 → 0.9995`**, **`1.00 → 1.000`**, `1.30 → 0.508`, `1.40 → 0.000`.
  - `sd = 0.15`: `0.50 → 0.000`, `0.65 → 0.012`, `0.70 → 0.141`, **`0.75 → 0.483`**, **`0.85 → 0.9925`**, **`1.00 → 1.000`**, `1.30 → 0.510`, `1.40 → 0.012`.
  - `sd = 0.25`: `0.50 → 0.000`, `0.65 → 0.070`, `0.70 → 0.213`, **`0.75 → 0.460`**, **`0.85 → 0.8815`**, **`1.00 → 0.9985`**, `1.30 → 0.495`, `1.40 → 0.086`.
- **`C2.4b` `m70a30` vs. `100% ff` (`N = 24`, aligned to Holm `α/8 = 0.00625` + `>= 1.10×` + `q05 > 1.00×`, `m8`):**
  - `sd_log = 0.10`: `1.00 → 0.000`, `1.10 → 0.471`, **`1.15 → 0.987`**, **`1.20 → 1.000`**.
  - `sd_log = 0.15`: `1.00 → 0.002`, `1.10 → 0.479`, **`1.15 → 0.924`**, **`1.20 → 0.998`**.
- **`EQ-C2.4c` `K=3` Median TOST — 3-Way Outcomes (`P_additive / P_non_additive / P_inconclusive`, `N = 24`, `M9`):**
  - `sd = 0.12` (eff margin `≈ 0.250`): `true 0.00 → 1.000 / 0.000 / 0.000`; `0.10 → 1.000 / 0.000 / 0.000`; `0.20 → 0.484 / 0.000 / 0.517`; `0.35 → 0.000 / 0.516 / 0.485`; `0.50 → 0.000 / 1.000 / 0.000`.
  - `sd = 0.20` (eff margin `≈ 0.216`): `true 0.00 → 1.000 / 0.000 / 0.000`; `0.10 → 0.968 / 0.000 / 0.032`; `0.20 → 0.441 / 0.000 / 0.559`; `0.35 → 0.001 / 0.266 / 0.734`; `0.50 → 0.000 / 0.985 / 0.016`.
  - `sd = 0.30` (eff margin `≈ 0.174`): `true 0.00 → 0.972 / 0.000 / 0.029`; `0.10 → 0.819 / 0.000 / 0.181`; `0.20 → 0.332 / 0.004 / 0.665`; `0.35 → 0.007 / 0.187 / 0.807`; `0.50 → 0.000 / 0.845 / 0.155`.

### 3.2 Exact Compute Budget (`0%` Unread Forwards; `0 CU` Used Until Registration + Code PASS & Elise Green Light)

| Block | Exact Enumeration (Every Arm Read by a Registered Rule, `M3`) | Short `B=1` Forwards |
|---|---|---:|
| Clean (`36`) + Write Shams (`13`) + Patch Shams (`6`) + C0-`":"` Pairs (`16`) | Baseline + Gate `C2.0` (`19` shams) + `C2.0_anchor_pred` | `71` |
| **Block 1 (`C2.1`)**: `3`L × `5` arms × `24` targets × `8` prompts (`4` LongBare + `4` ChatML) | `R_mf`, `Raw`, `Centered`, `Shrink`, `logit_lens` (Holm 1–4 + `μ_U` + ordering) | `2,880` |
| **Block 2 (`C2.2`)**: `9` conds × `12` targets × `12` pairs × `2` (`R_mf` + `dm0`) + `72` unsteered KL | Unpatched `3`, bidir MLP `2`, controls `4` (incl. `self_attenuated`) + `72` KL | `2,664` |
| **Block 2b (`C2.3`)**: `5`L (`18..30`) × (`24` `logit_lens` + `8` `rand_iso`) × `8` prompts | Post-erasure `E_open`, `|gap|`, & `ret_open > ret_rand` floor | `1,280` |
| **Block 3 (`C2.4`)**: `9` arms × `24` targets × `4` prompts × `2` doses (`0.50×`, `1.25×`) | `unmasked`, `unmasked_postmatched`, `masked`, `m70a30`, `2` L24 rand ctrls, `L21`, `L27`, `K3` | `1,728` |
| **Total Stage C2 v3** | Estimated `8–13 min` on Colab L4/A100 (`0%` unread forwards) | **`8,623`** (`~0.45–1.05 CU`) |
