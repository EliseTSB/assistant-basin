# Concept vs. Slot Head-Gating & Late-Layer Lens-Screened Heads (`CONCEPT-VS-SLOT-REPORT.md`)

**Date:** 2026-11-22 · **Author:** Vesper (`neon`, she/they) · **Branch:** `vesper-concept-vs-slot` (`a287e2b0` execution package; zero push; Part A `446cbde4` and Part B `fb6c81f9` untouched)  
**Design:** `explore_olmo2_attn_mlp/CONCEPT-VS-SLOT-DESIGN.md` v3.1 + Amendment 1 (`88986b904f499893ec5aba4039af9142846e45307552409cdaf0237f21b90bbd`)  
**Full Receipt:** `/home/elisepc/green/Journal/notes/reports/vesper-concept-vs-slot-full-2026-11-22/concept_vs_slot_full_receipt.json` (`19,390,429` B, SHA-256 `dc448e808dacd8f7f23eeb5ca6adc3dc3d9b1d965bc4a8810639a77a40da3a2d`)  
**Offline Scorecard:** `/home/elisepc/green/Journal/notes/reports/vesper-concept-vs-slot-full-2026-11-22/concept_vs_slot_scorecard.json` (SHA-256 `b175e3a1b8e5c3f6584b4b9055be2a6da0c15b56b57db8146bff74f6296c1766`)

---

## 1. Executive Summary (Zero Spin)

Across **`12,691`** measured `NVIDIA L4` forwards (`12,665` `B=11` + `26` `B=1`, `1.05 CU` Full; `1.17 CU` total including Smoke r1 `0.04` + Smoke r2 `0.08`; `184.55 CU` left, `0 active`), every technical and transport gate passed cleanly (`CS.0`: `6/6` at exact `0.0` sham/decomp/sibling/B11-causal-equiv error; `CS.1`: `6/6` transport cells AND `156/156` 72-row head-null checks `<= 0.15`, worst `0.1134`). **Zero pre-registered directional or late-head hypotheses confirmed:**

| Endpoint | Registered Overall Verdict | Key Measured Numbers (`C1..C6`) |
|---|---|---|
| **`CS.0`** (Technical, 6 clauses) | **PASSED (`6/6`)** | `sham=0.0`, `decomp=0.0`, `sibling=0.0`, `causal_equiv=0.0`, `min_act_readout=0.6406`, `min_blk_drop=0.0124` |
| **`CS.1`** (Transport + Head-Null) | **PASSED (`6/6` + `156/156`)** | Slot `SELF/NAME/ROLE` `+3.63/+2.66/+2.14` raw; Concept `+5.92/+5.37/+4.23` (`1.41–2.02×`); worst 72-row null `0.1134 <= 0.15` |
| **`CS.2` `L8H2`** (Primary) | **`inconclusive_readout_disagreement`** | `Δ_class = -0.0617` Raw `[-0.399,+0.242]` (`inconclusive_wide_interval`), `-0.0601` Spec `[-0.387,+0.263]` (`inconclusive_wide_interval`), `-0.0367` LogProb `[-0.340,+0.255]` (`both_weak_or_null`*); `slot_12 = +0.0415` raw (below `L8H3 +0.1631`), `concept_12 = -0.0201` |
| **`CS.2` `L8H15`** (Legacy) | **`inconclusive_wide_interval`** | All 3 channels `inconclusive_wide_interval`: `Δ_class = -0.2122` Raw `[-0.600,+0.148]`, `-0.1628` Spec `[-0.629,+0.290]`, `-0.1063` LogProb `[-0.578,+0.246]`; `slot_12 = +0.0975` raw (below `L8H3 +0.1631`) |
| **`CS.3`** (`7` Late Lens Heads) | **`0 / 7` confirmed** (`0 / 7` length-confounded) | Every candidate `not_confirmed`; all 7 pooled `\|C_SR\| <= 0.090` raw/spec (`L14H10` `#1` lens `S-R=+19.28` measured **`+0.0050` Raw / `-0.0020` Spec**) |
| **`CS.4` `L7H5`** (`ROLE_BARE` `S_odd`) | **`not_class_wide`** (Slot & Concept) | Slot `+0.1000` Raw (`3/12 >= +0.30`) / `+0.1549` Spec (`4/12`); Concept `+0.0417` Raw (`3/12`) / `-0.0074` Spec (`1/12`); `p92-09` replicates as a unit effect (`+0.440` Raw / `+0.971` Spec) |

*\*Mandatory null-side wording (`CONCEPT-VS-SLOT-DESIGN.md` §1.2 & Claude `#14` Condition 1):* Where `TargetLogProb` on `L8H2` lands in Branch 4 (`both_weak_or_null`), that channel means strictly that **any class difference is smaller than `0.35` logits/nats (about `68%` of Part B's `L8H2` `C_SR`)** with neither class exceeding controls — never `"no class difference"` or `"the same gating"`.

### Two Major Substantive Takeaways (Honest Falsifications)
1. **Part B's `L8H2` gating signal is real and repeatable on the 5 Part B core units, but does NOT generalize to Part 92 slot neurons as a class (panel selection bias):** On brand-new unseen `C1..C6`, the 5 Part B core slot units (`p92-02, p92-05, p92-13, p92-46, p92-09`) replicated Part B cleanly (**`5/5` strictly positive**, mean `C_SR = +0.6269` Raw / `+0.4120` Spec / `+0.5192` LogProb; `C_SN = +0.3884` Raw). Yet on the **8 other outcome-blind primary `L2..L6` slot units**, `7/8` were strictly negative (8-unit mean **`-0.2108` Raw / `-0.1107` Spec**), dragging `slot_12` to `+0.0415` Raw (trailing random control `L8H3 = +0.1631`) and `concept_12` to `-0.0201`.
2. **Static WikiText Neuron Lens `S_out` head screening (`self_minus_role_z` up to `+19.28`) had `0 / 7` predictive validity for causal `z_all` single-head path-blocking on unseen `C1..C6` pushes:** Not a single late candidate (`L11H9, L11H2, L11H3, L13H1, L13H4, L14H10, L14H3`) reached even `|C_SR| >= 0.10`, let alone the `0.25` floor, and `L11H3`/`L14H3` flipped sign.

---

## 2. Mandatory Scope Caveats (§1.2)
- Every verdict applies strictly to **`allenai/OLMo-2-0425-1B-Instruct@48d788eca847d4d7548f375ad03d3c9312f6139e`** (`float32`, `eager`), single-neuron post-`post_feedforward_layernorm` (`model.model.layers[L]`) pushes at `±0.8 × OLD_NORM`, `z_all` single-head path-blocking, and the fresh unseen 6-story bank `C1..C6`.
- The 12-pair primary panel is a same-layer matched contrast (`5` same-`persona_kind` + `7` cross-`persona_kind` pairs; not kind-pure).
- Peer-median-only (`SpecLogit`) shifts without `RawLogit` and `TargetLogProb` elevation are never conflated with absolute target promotion (`peer_median_only_gating = false` on both `L8H2` and `L8H15`).
- Branch 4 (`both_weak_or_null`) on `L8H2 TargetLogProb` means strictly **"any class difference is smaller than `0.35` logits/nats (about `68%` of Part B's `L8H2` `C_SR`)"**, never `"no class difference"` or `"the same gating"`.

---

## 3. Technical Gate (`CS.0`), Transport & Head-Null (`CS.1`), and Compute

### 3.1 `CS.0` (`6/6` Clauses PASSED)
- **Provenance & Pins:** `measured_gpu_run`, `NVIDIA L4`, `is_cuda_device = true`, `complete = true`, all 6 SHA-256 pins (`design`, `manifest`, `runner`, `scorer`, `targets213`, `m7_transport`) + full manifest join (`468` unpatched, `12,168` blocked, `26` sham, `26` actuation, `3` causal-equiv; `12,665` B11 + `26` B=1 = `12,691` forwards) verified.
- **Exact zero-error checks:** `max_abs_sham_{raw,spec,logp,phrase} = 0.0`, `max_abs_decomp_err = 0.0`, `max_sibling_leakage_science = 0.0`, `max_sibling_leakage_actuation = 0.0`, `max_causal_equiv_err = 0.0` (at `B=11`, F7), `max_abs_wdown_orth_cos = 4.70e-8`.
- **Positive controls (F1 & Claude `#18`):** `26/26` synthetic actuation controls passed (`min_actuation_target_delta = 1.2974 > 1e-2`, `min_actuation_readout_delta = 0.6406 > 1e-4`), and all `12,168` blocked rows passed the blocking positive control (`min_blocked_pos_drop = 0.01239 > 1e-8`).

### 3.2 `CS.1` Transport (`6/6` PASSED) & 72-Row Per-Class Head-Null (`156/156` PASSED)
- **Unpatched `C1..C6` transport (identical to Smoke):**
  - `slot_12`: `SELF +3.625 / +3.993 / +4.849` (raw/spec/logp; max null `0.774`), `NAME_BARE +2.664 / +3.218 / +3.817` (null `0.641`), `ROLE_BARE +2.137 / +3.040 / +3.474` (null `0.408`).
  - `concept_12`: `SELF +5.920 / +6.293 / +6.820` (null `0.361`), `NAME_BARE +5.369 / +6.273 / +6.323` (null `0.412`), `ROLE_BARE +4.231 / +5.670 / +5.749` (null `0.404`).
  - Concept-to-Slot ratios: `1.41–2.02×` (matching OLD `B1..B4` `1.41–2.07×`).
- **72-row per-class head-null gate (`<= 0.15`):** **ALL `156/156` (`26` heads × `2` classes × `3` cells) PASSED** (`worst_abs_mean_med_spec = 0.11335 <= 0.15`). Specifically: `L8H2` peaked at `0.0348`, `L11H3` at `+0.1095` (`slot_12 NAME_BARE`), and `L14H3` at `+0.0523` (`concept_12 NAME_BARE`). Zero heads were quarantined.

### 3.3 Runtime & Compute Accounting
- **Runner wall time:** `1,912.9 s` (`31.9 min`): `model_load_sec = 30.0 s`, `b11_elapsed_sec = 347.9 s` (`27.47 ms/B11`!), `b1_elapsed_sec = 0.72 s` (`27.58 ms/B1`), `non_fwd_runner_sec = 1,564.3 s` (Python `statistics.median` + CPU `.tolist()` over `12,665 × 11 × 36` peer matrices; future runners should compute peer medians via vectorized `torch.median` on GPU to cut wall time ~`4×`!).
- **Colab ledger (Account 1, `--agent neon`):** Smoke r1 `0.04 CU` + Smoke r2 `0.08 CU` + Full `1.05 CU` = **`1.17 CU` total** (`185.72 → 184.55 CU`, `0 active`).

---

## 4. `CS.2` — Concept vs. Slot Head-Gating Interaction (`L8H2` Primary; `L8H15` Legacy)

### 4.1 Co-Primary Channel Summary (`12` Matched Pairs × `6` Stories)

| Head | Channel | Per-Channel Verdict | `Δ_bar` (`C - S`) | Studentized `95%` Boot CI | Exact `t_11` `95%` CI | Pairs `< 0` | `slot_12` `C_SR` | `concept_12` `C_SR` | `ctrl_min / max Δ` | `ctrl_max` Slot / Concept |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`L8H2`** | **RawLogit** | `inconclusive_wide_interval` | `-0.0617` | `[-0.3988, +0.2421]` | `[-0.3764, +0.2531]` | `5/12` | `+0.0415` | `-0.0201` | `-0.1238 / +0.1831` | `L8H3 +0.1631` / `L8H13 +0.1277` |
| `L8H2` | **SpecLogit** | `inconclusive_wide_interval` | `-0.0601` | `[-0.3871, +0.2629]` | `[-0.3858, +0.2655]` | `7/12` | `+0.0728` | `+0.0127` | `-0.1228 / +0.0749` | `L8H13 +0.1876` / `L8H13 +0.2625` |
| `L8H2` | **TargetLogProb** | `both_weak_or_null`* | `-0.0367` | `[-0.3401, +0.2552]` | `[-0.3398, +0.2663]` | `6/12` | `+0.0260` | `-0.0108` | `-0.2506 / +0.1091` | `L8H9 +0.2299` / `L8H13 +0.1673` |
| `L8H2` | *PhraseLogProbMean* (desc.) | — | `-0.0361` | `[-0.2836, +0.1325]` | `[-0.2426, +0.1704]` | — | — | — | — | — |
| **`L8H15`** | **RawLogit** | `inconclusive_wide_interval` | `-0.2122` | `[-0.6003, +0.1484]` | `[-0.5742, +0.1499]` | `7/12` | `+0.0975` | `-0.1147` | `-0.1238 / +0.1831` | `L8H3 +0.1631` / `L8H13 +0.1277` |
| `L8H15` | **SpecLogit** | `inconclusive_wide_interval` | `-0.1628` | `[-0.6289, +0.2901]` | `[-0.6172, +0.2917]` | `7/12` | `+0.0241` | `-0.1387` | `-0.1228 / +0.0749` | `L8H13 +0.1876` / `L8H13 +0.2625` |
| `L8H15` | **TargetLogProb** | `inconclusive_wide_interval` | `-0.1063` | `[-0.5782, +0.2463]` | `[-0.5094, +0.2969]` | `6/12` | `-0.0056` | `-0.1119` | `-0.2506 / +0.1091` | `L8H9 +0.2299` / `L8H13 +0.1673` |

- **Why `L8H2` is `inconclusive_readout_disagreement`:** `RawLogit` (`lo = -0.3988`) and `SpecLogit` (`lo = -0.3871`) missed the Branch-4 lower bound `-0.35` by `0.049` and `0.037` logits while `TargetLogProb` (`[-0.3401, +0.2552]`, plus `ex_slot_CI <= +0.106`, `ex_conc_CI <= +0.137`) cleared Branch 4 (`both_weak_or_null`). Neither head came remotely close to Branch 1, 2, or 3 on any channel — every class mean sat below `L8` controls (`L8H3` / `L8H13` / `L8H9`), so Claude `#14`'s conservative 5-control `min`/`max` DiD note did *not* cause a near-miss.
- **Guards & Transport Diagnostics (`L8H2` RawLogit):** `C_SN` `slot=+0.0820, concept=+0.0090, Δ=-0.0730`; `5` same-kind pairs `Δ_same_kind = -0.1640`, within-kind OLS `β = -0.1284 ± 0.2160`; fractional gating `F_SR_ratio`: slot `+0.0144`, concept `-0.0040`, `Δ_F_ratio = -0.0184`; Weaver within-class transport slopes `slot = -0.0229`, `concept = -0.0073`, joint ANCOVA `β_class_adj = -0.0333 ± 0.1468` (transport scaling is flat near zero!).

### 4.2 All `12` Matched Pairs (`L8H2`) & The Part B 5-Core Selection-Bias Discovery

| Pair | Layer | Slot Unit | PB5? | Slot Raw `C_SR` | Slot Spec `C_SR` | Concept Unit | Concept Raw `C_SR` | Concept Spec `C_SR` | `Δ` Raw (`C - S`) | `Δ` Spec |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| `P01` | `L2` | `p92-10` | | `-0.124` | `+0.069` | `p92-29` | `-0.557` | `-0.625` | `-0.433` | `-0.695` |
| `P02` | `L3` | `p92-04` | | `+0.053` | `+0.502` | `p92-00` | `+0.215` | `+0.284` | `+0.162` | `-0.218` |
| **`P03`** | `L3` | **`p92-02`** | **PB5** | **`+0.612`** | `-0.009` | `p92-11` | `-0.248` | `-0.027` | **`-0.860`** | `-0.017` |
| **`P04`** | `L3` | **`p92-13`** | **PB5** | **`+0.630`** | **`+0.702`** | `p92-39` | `+0.175` | `+0.072` | **`-0.456`** | **`-0.630`** |
| `P05` | `L4` | `p92-24` | | `-0.097` | `-0.146` | `p92-01` | `+0.530` | `+0.664` | `+0.627` | `+0.810` |
| `P06` | `L4` | `p92-35` | | `-0.376` | `-0.356` | `p92-48` | `+0.049` | `+0.051` | `+0.425` | `+0.407` |
| `P07` | `L4` | `p92-36` | | `-0.520` | `-0.419` | `p92-08` | `-0.243` | `-0.012` | `+0.277` | `+0.407` |
| `P08` | `L4` | `p92-47` | | `-0.083` | `-0.104` | `p92-45` | `-0.082` | `-0.071` | `+0.001` | `+0.033` |
| `P09` | `L5` | `p92-33` | | `-0.315` | `-0.089` | `p92-15` | `-0.128` | `-0.102` | `+0.187` | `-0.013` |
| `P10` | `L5` | `p92-37` | | `-0.225` | `-0.344` | `p92-38` | `+0.223` | `+0.114` | `+0.447` | `+0.457` |
| **`P11`** | `L5` | **`p92-46`** | **PB5** | **`+0.586`** | **`+0.629`** | `p92-41` | `-0.165` | `-0.082` | **`-0.750`** | **`-0.711`** |
| **`P12`** | `L6` | **`p92-09`** | **PB5** | **`+0.357`** | **`+0.437`** | `p92-40` | `-0.011` | `-0.113` | **`-0.368`** | **`-0.550`** |

- **Non-primary extras (`L8H2`):** `p92-05` (**PB5**, `L3` slot): **`+0.9495` Raw / `+0.3009` Spec**; `p92-43` (`L6` reader-agreed concept): `-0.0333` Raw / `-0.0594` Spec.
- **Part B 5-core-slot bridge (`p92-02, p92-05, p92-13, p92-46, p92-09`) vs. 8 other primary slots:**
  - **5 PB5 slots (`C1..C6`):** **`+0.6269` Raw (`5/5` positive!), `+0.4120` Spec, `+0.5192` LogProb; `C_SN = +0.3884` Raw** (4 PB5 primary slots: `+0.5462` Raw / `+0.4397` Spec).
  - **8 non-PB5 primary slots (`C1..C6`):** **`-0.2108` Raw (`7/8` negative!), `-0.1107` Spec**.
  - This single table explains the entire `CS.2` outcome: `L8H2`'s self-vs-role gating is genuinely reproducible out-of-bank on the 5 hand-picked Part B core units, **and fails to generalize to the broader Part 92 `L2..L6` slot class**!
- **Registered descriptive sensitivities (`L8H2` RawLogit):**
  - Clean-token `5` intact pairs (`t_4`): `-0.0478 [-0.8055, +0.7098]`; `15` surviving units Welch `t`: `-0.1223 [-0.5439, +0.2993]`.
  - Baseline-logit correlation `corr(C_SR, base_raw)`: pooled `-0.010` (slot `-0.141`, concept `+0.135`).
  - Reader-agreed `7` pairs: `-0.0555 [-0.3971, +0.2860]` Raw, `-0.0489` `PhraseLogProbMean`.
  - Exact-token `' dog'` cross-layer witness: `p92-08` (`L4` concept) `-0.2431` vs `p92-09` (`L6` slot) `+0.3574` (`diff = -0.6004` Raw).

---

## 5. `CS.3` — Held-Out Confirmation of `7` Late Lens-Screened Heads (`0 / 7` Confirmed)

All 7 tests evaluated across all `24` primary units × `6` stories (`144` rows), two-sided Bonferroni `99.29%` (`α = 0.05/7`) unit-cluster CI:

| Head | Pred `(SR, SN)` | Lens `S-R` | Pooled Raw `C_SR` (`99.29%` CI) | Pooled Spec `C_SR` (`99.29%` CI) | Pooled LogP | Pooled `C_SN` (Raw / Spec) | Named Ctrl `max \|C_SR\|` (Raw / Spec) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`L11H9`** | `(+, +)` | `+10.45` | `+0.0270 [-0.097, +0.147]` | `+0.0421 [-0.070, +0.207]` | `+0.0128` | `+0.031 / +0.049` | `L11H1: 0.000 / 0.025` | `not_confirmed` |
| **`L11H2`** *(sat.)* | `(+, +)` | `+6.13` | `+0.0112 [-0.198, +0.254]` | `-0.0818 [-0.255, +0.119]` | `+0.0036` | `+0.032 / -0.111` | `L11H6: 0.035 / 0.057` *(L11H4: 0.028)* | `not_confirmed` |
| **`L11H3`** | `(-, -)` | `-8.13` | `+0.0734 [-0.202, +0.418]` | `+0.0902 [-0.211, +0.435]` | `+0.0597` | `+0.122 / +0.050` | `L11H10/H4: 0.040 / 0.208` | `not_confirmed` |
| **`L13H1`** | `(+, +)` | `+13.90` | `+0.0083 [-0.047, +0.058]` | `+0.0137 [-0.038, +0.063]` | `+0.0172` | `-0.008 / +0.006` | `L13H13/H5: 0.072 / 0.061` | `not_confirmed` |
| **`L13H4`** | `(-, -)` | `-9.99` | `-0.0167 [-0.116, +0.073]` | `+0.0135 [-0.084, +0.103]` | `+0.0049` | `-0.046 / -0.040` | `L13H5/H13: 0.072 / 0.061` | `not_confirmed` |
| **`L14H10`** | `(+, +)` | **`+19.28`** | **`+0.0050 [-0.031, +0.047]`** | **`-0.0020 [-0.042, +0.051]`** | `+0.0098` | `+0.013 / +0.012` | `L14H12: 0.010 / 0.031` | `not_confirmed` |
| **`L14H3`** | `(-, -)` | `-8.08` | `+0.0458 [-0.039, +0.148]` | `+0.0072 [-0.099, +0.105]` | `+0.0250` | `+0.060 / -0.008` | `L14H8: 0.001 / 0.019` | `not_confirmed` |

**Zero out of 7 confirmed (and `0/7` even `confirmed_length_confounded`).** Even the most extreme WikiText lens heads (`L14H10` `S-R = +19.28`, `L13H1` `+13.90`) are tightly bounded around zero (`±0.05–0.06` logits at `99.29%` confidence). Static WikiText `S_out` head-token projections do **not** predict causal single-head path-blocking for persona-neuron writes.

---

## 6. `CS.4` — `L7H5` `ROLE_BARE` Signed Antisymmetry Diagnostic (`not_class_wide`)

- **`slot_12`: `not_class_wide`** on BOTH `RawLogit` (`mean = +0.1000`, `ctrl_max = +0.0047 [L7H9]`, `margin = +0.0953`, `3/12 >= +0.30`) and `SpecLogit` (`mean = +0.1549`, `ctrl_max = +0.0506 [L7H4]`, `margin = +0.1043`, `4/12 >= +0.30`).
- **`concept_12`: `not_class_wide`** on BOTH `RawLogit` (`mean = +0.0417`, `ctrl_max = +0.0768`, `margin = -0.0351`, `3/12`) and `SpecLogit` (`mean = -0.0074`, `ctrl_max = +0.1417`, `margin = -0.1491`, `1/12`).
- **Unit diagnostics:** **`p92-09`** (`L6` slot `' dog'`) replicates cleanly out-of-bank (**`+0.440` Raw / `+0.971` Spec**, alongside slot units `p92-13 +0.461` and `p92-46 +0.384` Raw), whereas exact-token witness **`p92-08`** (`L4` concept `' dog'`) is weak (`+0.019` Raw / `+0.180` Spec) and `L6` concept `p92-40` is strongly negative (`-0.685` Raw / `-0.459` Spec; `p92-43` `+0.330` Raw / `+0.087` Spec). `L7H5` antisymmetry is unit-localized, not class-wide.

---

## 7. Calibration Check Against Vesper's Frozen Pre-Data Forecasts (§7 & Amendment 1)

| Pre-Registered Forecast | Frozen Prior | Realized Outcome | Calibration Note |
|---|---|---|---|
| `CS.2` (`L8H2`) overall verdict | `24%` `inconclusive_readout_disagreement` (#2 of 10 outcomes); `28%` `inconclusive_wide_interval`; `18%` `both_weak_or_null` | **`inconclusive_readout_disagreement`** (`Raw/Spec`: `inconclusive_wide_interval`, `LogP`: `both_weak_or_null`) | Well-calibrated on verdict class; my un-gated anchor expectation (`-0.15` to `-0.35`) was too negative (`-0.062` realized) because I underestimated how much the 5 Part B core units drove Part B's prior. |
| `CS.3` (`7` late lens heads) | `30%` for `0/7` confirmed; per-head `15–35%` (expected `~1.8/7`) | **`0 / 7` confirmed** (`0 / 7` length-confounded) | Too optimistic on per-head priors; WikiText `S_out` lens extremes had literally zero causal `C_SR` transfer. |
| `CS.4` (`L7H5` `ROLE_BARE`) | `~75%` Slot / `~85%` Concept `not_class_wide` or disagreement | **`not_class_wide`** in both classes | Well-calibrated. |
| Amendment 1 (`L11H3`/`L14H3` Full `0.15` head-null quarantine) | `45%` both, `30%` `L11H3` only, `10%` `L14H3` only, **`15%` neither** | **Neither quarantined** (`L11H3` max `0.1095`, `L14H3` max `0.0523`) | Over-extrapolated 6-row Smoke means (`+0.305`/`+0.200`); 72-row averaging pulled every head safely under `0.15` (and validates Amendment 1's resolution of the Smoke gate!). |
