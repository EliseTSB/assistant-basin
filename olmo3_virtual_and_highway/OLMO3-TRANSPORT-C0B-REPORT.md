# Stage C0b Report: Pre-Registered `positions="all"` Calibration, Attention-Share Attribution & Live-Paired `R_ridge` Cross-Domain Whitening Test on OLMo 3 7B

**Status:** Pre-registered within-unit follow-up completed (`OLMO3-TRANSPORT-C0B-DESIGN.md` Registered v1 @ `4e2b1820`; runner/scorer frozen at git commit `6396e2af`, attested by git history).  
**Headline Verdict:** **Global Gate `C0b.0`: FAIL** (sole failed clause: `C0b.0.3` `fde_err_p99_9 = 1.4561 > 0.45`) · **All downstream verdicts (`C0b.0.4b_drift`, `C0b.1a`, `C0b.1b`, `C0b.2a`, `C0b.2b`): `BLOCKED_BY_C0b.0`**.  
**Compute:** `5,328` executed `B=1` `bf16` forwards (`0` capped skipped: `8` clean, `8` sham, `3,072` Regime-D `"all"`, `320` Regime-D `"last"` anchor, `1,152` Regime-N `"last"`, `768` Regime-N `"all"`), `443.79 s` (`7.4 min`) on NVIDIA L4 (`0.228 CU` compute alone at `1.85 CU/hr`; **`0.28 CU`** billed including VM boot/download/teardown and `0` active runtimes verified via Colab ledger — run-ledger attestations external to the JSON bundle).  
**Provenance:**
- Design: `olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-DESIGN.md` (`sha256 = 8b3efeaeba3febbcc0e9715840a79bea6d6bc29986f7bf21a89cf6080d61a994`)
- Stage C0b constants: `olmo3_virtual_and_highway/c0b_prereg_constants.json` (`sha256 = 4596f012bb862aec81a0b07a478223ef02dab8bab3706c4f7902c860958a0f2e`)
- Stage C0 constants: `olmo3_virtual_and_highway/c0_prereg_constants.json` (`sha256 = 3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98`)
- Stage C0 reference receipt: `results/vesper-olmo3-7b-stage-c0/c0_receipt.json.gz` (uncompressed `sha256 = fe7883be37ecf0b7a0a694fc1dd035215d05c27c297cc0959ef23f379258fb1c`)
- Frozen lens & vector bank: `r_lens.pt` (`sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`), `c0_vector_bank.pt` (`sha256 = ab7e04caaf74d4725ed88596da3c39f9303d5e55d7114bbbef1a5500b025ab5f`; archived externally, hashes attested in receipt metadata)
- Stage C0b receipt: `results/vesper-olmo3-7b-stage-c0b/c0b_receipt.json.gz` (`453,084` bytes; `.gz sha256 = 06903d73…`, uncompressed `sha256 = bc477e3ec98692eda5d6e78eb2d7d7def2061ca566ea18904f89bae7fa170db0`)
- Scorecards (bare `/usr/bin/python3 -I` byte-identical via `cmp`): `scorecard.json` (`ff79d751…`), `SCORECARD.md` (`fd9ed7f3…`)

---

## Definitions & Symbols

All quantities below match `OLMO3-TRANSPORT-C0B-DESIGN.md` and `score_olmo3_c0b.py`:
- **Final-RMSNorm & Regime-D linearity:** `r_c` and `r_i` are the clean and intervened final RMSNorm denominators at the last token (`rms_clean`, `rms_int`); `r_i` shrink is `(1 - r_i / r_c) × 100%` (negative values indicate expansion). `fde` (`frozen_direct_effect`) is `Direct_linear · (r_c / r_i)`, where `Direct_linear = +3.0` in Regime D, and `fde_err = |fde - 3.0|`.
- **4-term telescoping decomposition:** `exact_dlogit = Direct + ΣAttn + ΣMLP + RMSsh + telescoping_err`, where `RMSsh` is the receipt's `rmsnorm_shift` field (logit change from the `r_c → r_i` final-RMSNorm denominator shift).
- **Arms & Regime-D gap metrics (`C0b.1a`, `C0b.1b`):** `dm` is the `direct_matched` 4-draw mean `exact_dlogit` (or component), and `ll` is `logit_lens`. For each `(L, target)` cell, `gap_all = exact_dlogit(R_mf) - exact_dlogit(dm)` and `r_all = mean_8p(gap_all) / pred_gap_D3` (with frozen `pred_gap_D3` from `c0_prereg_constants.json`). In `C0b.1b`, `ΔTot`, `ΔAttn`, and `ΔMLP` are the `positions="all"` minus Stage C0 `positions="last"` differences of the `R_mf - dm` gaps (`score_olmo3_c0b.py:1374-1457`); `share_t = ΔAttn_t / ΔTot_t` when `ΔTot_t > δ = 0.2568`, else `0.0`. `ΔRMSsh` in §2.2 is a report-side diagnostic computed from the receipt's `rmsnorm_shift` field using the identical `R_mf - dm` (`"all" - "last"`) gap pattern (not emitted directly by `score_olmo3_c0b.py`).
- **Regime-N cross-domain & calibration metrics (`C0b.2a`, `C0b.2b`):** For readout target `t` on `L ∈ {18, 24, 27}`, `B_cross(arm, t) = mean_{t' cross, 3L, 8p} |Δlogit(t; arm(t'))|`, `self_dlogit(arm, t) = mean_{3L, 8p} exact_dlogit(t; arm(t))`, and `S_cross(arm, t) = B_cross(arm, t) / max(self_dlogit(arm, t), 1e-6)` (`score_olmo3_c0b.py:696-754`). `frac_err` in §2.3's table is the median over `48` `(L, target)` cells of `|mean_8p(exact_dlogit) - mean_8p(pred_dlogit)| / max(|mean_8p(pred_dlogit)|, 1e-6)` (`score_olmo3_c0b.py:1562-1569`).

---

## 1. Pre-Registered Gate Summary & What Failed

Global Gate `C0b.0` failed on one registered clause — **`C0b.0.3` `fde_err_p99_9 = 1.4561 > 0.45`** (`fde_err_max = 1.6255`) — and held all downstream verdicts closed (`BLOCKED_BY_C0b.0`). Every other global gate clause passed, and the raw `C0b.0.4b` cross-run reproducibility check against Stage C0's published receipt showed exact `0.000000` difference across all `320` anchor traces and `5,760` off-target cells.

| Check | Official Verdict | Registered Result vs. Bar |
|---|---|---|
| **`C0b.0.1` (Sham & arm liveness)** | **PASS** *(clause)* | All `8/8` zero-dose shams (`L12/18/24/27 × {last, all}`) exact `0.0`; movement check `true`; Regime-N and Regime-D arm liveness `true` (`0` dead-arm issues). |
| **`C0b.0.2` (Telescoping & finiteness)** | **PASS** *(clause)* | All `5,312` retained traces finite; `median = 0.0156`, `p95 = 0.0496 <= 0.08`, `p99.9 = 0.0920 <= 0.15`, `max = 0.1155` (Bare `p95 / max = 0.0502 / 0.1155`, ChatML `0.0487 / 0.1084`). |
| **`C0b.0.3` (Cap & intervened denominator)** | **FAIL** *(global gate)* | `max_cap_hit_rate = 0.0000 <= 0.02`, `0` capped-cell failures, `fde_err_p95 = 0.2983 <= 0.30` (PASS); **FAIL on `fde_err_p99_9 = 1.4561 > 0.45`** (`max = 1.6255`; Bare `p95 / max = 0.2689 / 0.8929`, ChatML `p95 / max = 0.3298 / 1.6255`; KL Bare `med / p95 = 0.0263 / 1.3410`, ChatML `0.0037 / 0.5744`). |
| **`C0b.0.4a` (Strict SHA-256 provenance)** | **PASS** *(clause)* | `10 / 10` strict SHA-256 checks matched on disk and in receipt metadata. |
| **`C0b.0_global` (`1–4a`)** | **FAIL** | Blocked solely by `C0b.0.3` `fde_err_p99_9 = 1.4561 > 0.45`. |
| **`C0b.0.4b` (`C0b.1b` receipt drift)** | **`BLOCKED_BY_C0b.0`** | Raw check passed (`passed = true`): exact `0.000000` difference on `max_anchor_gap_diff` (`" lantern"`, `" judge"`), `max_anchor_attn_gap_diff`, `max_R_mf_B_cross_diff`, `max_R_mf_offtarget_cell_diff`, and `max_R_mf_all16_cell_diff` across `320` anchor traces and `5,760` off-target cells; recorded `BLOCKED_BY_C0b.0` by `score_olmo3_c0b.py`'s fail-closed global gate convention. |
| **`C0b.1a`, `C0b.1b`, `C0b.2a`, `C0b.2b`** | **`BLOCKED_BY_C0b.0`** | Not scored as confirmatory verdicts under the fail-closed rule. |

### Post-Hoc Gate Diagnostic: Why `C0b.0.3` Failed at `L12` `positions="all"`

In Regime D, the linear-regime assumption requires `r_i` to stay close to `r_c`, so that `fde = Direct_linear · (r_c / r_i)` remains near `+3.0` (registered clause: `|fde - 3.0| <= 0.30` at `p95`, which passed at `0.2983`, **and** `<= 0.45` at `p99.9`, which failed at `1.4561`). The pre-registered evidence for Gate `C0b.0.3` is the scorecard quantiles and by-format logging in the table above; **the exceedance counts, per-`(layer, arm)` table, and `L12`-exclusion percentiles below are post-hoc receipt diagnostics, not registered gate clauses**.

Across all `3,072` retained Regime-D `positions="all"` traces, **`80` traces exceeded `|fde - 3.0| > 0.45`**, and all `80` had `frozen_direct_effect > 3.0` (meaning `r_i < r_c`). Nearly all of those misses — **`73 / 80` (`91.25%`)** — came from a single `(layer, arm)` block: **`L12 R_mf` under `positions="all"`** (`55/64` ChatML, `18/64` Bare). The remaining `7` misses were `6/512` in `L12 direct_matched` (`4` Bare, `2` ChatML) and `1/128` in `L18 R_mf` (`1` Bare), with **`0 / 2,304`** misses across `L24`, `L27`, `L18 direct_matched`/`logit_lens`, and `L12 logit_lens`.

| Layer | Arm (`n`) | `dose / norm` (min / med / max) | `> 0.45` count | `fde_err` med / p95 / max | `r_i / r_c` min / med | Max `r_i` shrink | Mean `RMSsh` | `KL` med / p95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `L12` | `R_mf` (`128`) | `0.97 / 1.32 / 1.64` | **`73`** | `0.5596 / 1.2352 / 1.6255` | `0.6486 / 0.8428` | **`35.1%`** | `+1.9531` | `1.4251 / 3.6648` |
| `L12` | `direct_matched` (`512`) | `0.97 / 1.32 / 1.64` | `6` | `0.0966 / 0.3373 / 0.5400` | `0.8475 / 0.9766` | `15.3%` | `+0.1060` | `0.3804 / 1.5253` |
| `L12` | `logit_lens` (`128`) | `0.34 / 0.40 / 0.46` | `0` | `0.0199 / 0.0572 / 0.0857` | `0.9764 / 1.0001` | `2.4%` | `+0.0024` | `0.0246 / 0.0973` |
| `L18` | `R_mf` (`128`) | `0.41 / 0.54 / 0.70` | `1` | `0.1971 / 0.3551 / 0.5531` | `0.8443 / 0.9384` | `15.6%` | `+0.7117` | `0.1319 / 0.7657` |
| `L18` | `direct_matched` (`512`) | `0.41 / 0.54 / 0.70` | `0` | `0.0342 / 0.1255 / 0.2362` | `0.9270 / 0.9936` | `7.3%` | `+0.0350` | `0.0387 / 0.2570` |
| `L18` | `logit_lens` (`128`) | `0.20 / 0.23 / 0.27` | `0` | `0.0139 / 0.0326 / 0.0470` | `0.9849 / 1.0017` | `1.5%` | `-0.0076` | `0.0048 / 0.0298` |
| `L24` | `R_mf` (`128`) | `0.25 / 0.31 / 0.40` | `0` | `0.0264 / 0.0692 / 0.1226` | `0.9607 / 0.9913` | `3.9%` | `+0.0959` | `0.0144 / 0.1129` |
| `L24` | `direct_matched` (`512`) | `0.25 / 0.31 / 0.40` | `0` | `0.0126 / 0.0336 / 0.0523` | `0.9872 / 1.0016` | `1.3%` | `-0.0099` | `0.0047 / 0.0292` |
| `L24` | `logit_lens` (`128`) | `0.16 / 0.18 / 0.21` | `0` | `0.0093 / 0.0220 / 0.0280` | `0.9921 / 1.0027` | `0.8%` | `-0.0126` | `0.0017 / 0.0100` |
| `L27` | `R_mf` (`128`) | `0.20 / 0.26 / 0.33` | `0` | `0.0278 / 0.0551 / 0.0812` | `0.9737 / 0.9912` | `2.6%` | `+0.0692` | `0.0104 / 0.0376` |
| `L27` | `direct_matched` (`512`) | `0.20 / 0.26 / 0.33` | `0` | `0.0080 / 0.0209 / 0.0335` | `0.9904 / 1.0022` | `1.0%` | `-0.0109` | `0.0027 / 0.0122` |
| `L27` | `logit_lens` (`128`) | `0.14 / 0.17 / 0.19` | `0` | `0.0074 / 0.0160 / 0.0201` | `0.9964 / 1.0022` | `0.4%` | `-0.0116` | `0.0016 / 0.0056` |

Two post-hoc diagnostic patterns explain the gate failure as an instrument-regime behavior at `L12`:
1. **Why `L12` left the linear regime:** Matching `Direct_linear = +3.0` at `L12` (`post_mlp_norm = 4.2478`) requires per-position `dose = 4.13–6.95` (`0.97–1.64×` `post_mlp_norm`, median `1.32×`), compared with `0.25–0.40×` at `L24` and `0.20–0.33×` at `L27`. In Stage C0 (`positions="last"`), that same `L12 R_mf` dose produced `fde_err p95 = 0.3044`, `max = 0.4291` (max `r_i` shrink `12.51%`, `≈ 12.5%`). Injecting the same per-position dose at every token `0..T-1` (`positions="all"`) shrank `r_i` by up to **`35.1%`** (`r_i / r_c = 0.6486`, `fde = 4.6255` on `" tiger"` / `chat_narr_0`, mean `RMSsh = +1.9531` logits, `KL` median `1.4251`, max `5.3837`), taking the model out of the linear regime the decomposition assumes.
2. **Format asymmetry inside `L12 R_mf`:** The denominator shrink was stronger on the longer ChatML prompts (`31–43` tok: `55/64 > 0.45`, median shrink `20.1%`, max `35.1%`, mean `RMSsh = +3.2221`, `fde_err` med `0.7537`) than on Bare prompts (`7–13` tok: `18/64 > 0.45`, median shrink `6.2%`, max `22.9%`, mean `RMSsh = +0.6842`, `fde_err` med `0.2395`). Excluding all `L12` traces (`L18/L24/L27`, `n = 2,304`) reduces Regime-D `"all"` `fde_err` to `p95 = 0.1388`, `p99.9 = 0.4005`, `max = 0.5531` (`1` trace `> 0.45`); excluding only `L12 R_mf` (`n = 2,944`) still leaves `p99.9 = 0.5247` (`7` traces `> 0.45`).

Gate `C0b.0.3` functioned as designed: rather than passing a nonlinear `L12` regime into calibration and attribution verdicts, it caught the denominator shift and failed closed.

---

## 2. Post-Hoc Descriptive Readouts on Blocked Metrics (NOT Pre-Registered Findings)

Because Global Gate `C0b.0` failed, **`C0b.1a`, `C0b.1b`, `C0b.2a`, and `C0b.2b` are officially `BLOCKED_BY_C0b.0`**. No downstream verdict is read. The numbers below were computed post-hoc from `c0b_receipt.json.gz` and `c0_receipt.json.gz` using the formulas in `score_olmo3_c0b.py` (plus report-side `ΔRMSsh` from the receipt's `rmsnorm_shift` field); they are reported strictly as descriptive measurements and explicitly labelled counterfactuals, not as confirmatory passes or proofs.

### 2.1 Blocked `C0b.1a` Readout (`positions="all"` Gap Calibration, `L ∈ {12, 18, 24, 27}`)

- **Pooled 4-layer `median(r_all)`:** `1.2689` (90% BCa CI `[1.2380, 1.3011]`; paired manipulation check vs. Stage C0 `"last"` 4-layer baseline `0.3051`: `mean(Δr) = +0.9155`, `16/16` positive, sign-flip `p = 1.5e-5`).
- **Format & style descriptives:** Pooled Bare `median(r_all) = 1.2336`, ChatML `1.2450` (`ChatML - Bare` 95% BCa CI `[-0.0497, +0.1127]`); `Narrative - Analytical` 95% BCa CI `[-0.0816, -0.0320]` (point `-0.0612`); `slope(r_all vs T) = -0.001145` per token across `T ∈ [7, 43]` (per-prompt medians range `1.1269` on `chat_narr_1` [`31` tok] to `1.3399` on `bare_anal_1` [`10` tok]). Pooled `Retention(logit_lens)` under `"all"` is `0.3772`, and the scorer's registered descriptive `|median(dm - ll)| = 0.0000` (whereas `median(|dm - ll|) = 0.1016`).
- **Counterfactual `C0b.1a` verdict (`INCONCLUSIVE`) & `3L` descriptive subset:** Even in the counterfactual where Gate `C0b.0.3` had passed, `C0b.1a` would have returned **`INCONCLUSIVE`** under the registered 3-way rule: clauses `(i)` (pooled `1.2689 ∈ [0.65, 1.35]`, 90% CI `[1.2380, 1.3011] ⊂ (0.55, 1.50)`), `(ii)` (Bare `1.2336` and ChatML `1.2450 ∈ [0.65, 1.35]`), and `(iv)` (all 8 `(layer, format)` cell medians in `[0.8017, 1.4389] ⊂ [0.50, 1.50]`) all passed, while **only** per-layer split clause `(iii)` (`|ChatML(L) - Bare(L)| <= 0.35`) failed at `L12` (`|1.3313 - 0.8017| ≈ 0.5297 > 0.35`, exact `0.529680`), where `L12 R_mf` also showed the largest ChatML-vs-Bare `r_i` shrink and `RMSsh` gap (`+3.2221` vs. `+0.6842` logits). On the descriptive-only (non-gate-eligible) `3L` subset `{18, 24, 27}`, pooled `median(r_all) = 1.2958` (90% BCa CI `[1.2701, 1.3137]`), all three per-layer `|ChatML - Bare|` splits were `<= 0.3101` (`0.310029`, `0.2308`, `0.2034`, below the `0.35` bar), ChatML was `1.2339`, **and Bare was `1.3752`, which sits outside the registered `[0.65, 1.35]` calibration band**:

| Layer | Pooled `r_all` med | Bare `r_all` med | ChatML `r_all` med | `ChatML - Bare` | `|ChatML - Bare|` (bar `<= 0.35`) |
|---:|---:|---:|---:|---:|---:|
| `L12` | `1.0854` | `0.8017` | `1.3313` | `+0.5297` | **`0.5297`** |
| `L18` | `1.1917` | `1.0367` | `1.3468` | `+0.3100` | `0.3100` |
| `L24` | `1.3219` | `1.4389` | `1.2081` | `-0.2308` | `0.2308` |
| `L27` | `1.3039` | `1.3909` | `1.1875` | `-0.2034` | `0.2034` |

### 2.2 Blocked `C0b.1b` Readout (4-Term Decomposition of `"all" - "last"` Shift)

- **4-layer target-level summary (counterfactual `C0b.1b: PASS` on registered bars; officially `BLOCKED_BY_C0b.0`):** `mean(ΔTot) = +4.7157` logits (`16/16 > δ = 0.2568`), `mean(ΔAttn) = +3.8695` logits (registered floor `1.7031`; sign-flip `p = 1.5e-5`), `mean(ΔMLP) = +0.2514` logits, report-side `mean(ΔRMSsh) = +0.5946` logits, `median(share_t) = 0.8155` (90% BCa CI `[0.7882, 0.8592]`, `q05 = 0.7882 > 0.35`).
- **Layer-by-layer breakdown (noting `L12` nonlinearity contribution):** At `L12`, `ΔRMSsh = +1.6647` logits is `23.4%` of `ΔTot = +7.1204`. At `L24` and `L27` (where `r_i` shrink was `<= 3.93%` and `ΔRMSsh <= +0.0589`), `R_mf ΣAttn` moved from `+0.2069` (`"last"`) to `+2.2486` (`"all"`) at `L24` and `+0.1719` to `+1.6639` at `L27`, while `direct_matched ΣAttn` moved from `-0.0617` to `-1.5298` at `L24` and `-0.0737` to `-0.9789` at `L27`:

| Layer | `ΣAttn` gap (`"last"` C0) | `ΣAttn` gap (`"all"` C0b) | `ΔAttn` | `ΔMLP` | `ΔRMSsh` *(report-side)* | `ΔTot` | Aggregate `ΔAttn / ΔTot` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `L12` | `+0.4323` | `+5.4481` | `+5.0158` | `+0.4379` | `+1.6647` | `+7.1204` | `0.7044` |
| `L18` | `+0.3009` | `+4.8562` | `+4.5553` | `+0.4608` | `+0.5992` | `+5.6137` | `0.8115` |
| `L24` | `+0.2686` | `+3.7784` | `+3.5098` | `+0.0986` | `+0.0555` | `+3.6679` | `0.9569` |
| `L27` | `+0.2455` | `+2.6428` | `+2.3972` | `+0.0082` | `+0.0589` | `+2.4608` | `0.9742` |

*Note:* `ΔRMSsh` is the mean `rmsnorm_shift`-field gap (`R_mf - dm`, `"all" - "last"`), not the algebraic residual `ΔTot - ΔAttn - ΔMLP`; `ΔAttn + ΔMLP + ΔRMSsh` differs from `ΔTot` by `0.001–0.004` logits per layer (the `ΔDirect + Δtelescoping` residual).

### 2.3 Blocked `C0b.2a` & `C0b.2b` Readout (Regime N `0.75×`, `L ∈ {18, 24, 27}`)

These Regime-N traces executed on `L ∈ {18, 24, 27}` under `positions="last"` (`r_i` shrink medians `0.99%` on `R_mf`, `0.90%` on `R_ridge`, and `-0.52%` [slight expansion] on `logit_lens`; maxima `5.08%`, `3.58%`, and `2.49%`), separate from the `L12` `"all"` block that triggered Gate `C0b.0.3`, but they remain officially **`BLOCKED_BY_C0b.0`** because Gate `C0b.0` is global. All 16 targets are consonant-initial, space-prefixed singular concrete nouns (the same units as Stage C0).

| Regime & Arm (`L ∈ {18, 24, 27}`) | `B_cross` arith | `B_cross` geo | `self_dlogit` geo | `S_cross` geo | Signed cross mean (`% pos`) | Same-domain wrong abs | Cell-median `frac_err` (`48` cells) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `"last"` `R_mf` *(live; `0.0` drift vs. C0)* | `1.3438` | `1.3410` | `8.5608` | `0.1566` | `+1.3335` (`97.6%`) | `1.5957` | `0.3271` |
| `"last"` `R_ridge` *(live)* | `0.7898` | `0.7882` | `9.2558` | `0.0852` | `+0.7616` (`93.9%`) | `1.0973` | `0.0958` |
| `"last"` `logit_lens` *(live descriptive)* | `0.3471` | `0.3455` | `9.1946` | `0.0376` | `+0.0950` (`60.0%`) | `0.4919` | — |
| `"all"` `R_mf` *(paired descriptive secondary)* | `3.4960` | `3.4858` | `13.4952` | `0.2583` | `+3.4725` (`98.1%`) | `3.8326` | `0.0338` |
| `"all"` `R_ridge` *(paired descriptive secondary)* | `1.3670` | `1.3619` | `9.8930` | `0.1377` | `+1.2942` (`94.3%`) | `1.6506` | `0.0809` |

Key descriptive comparisons from the table above (all officially `BLOCKED_BY_C0b.0`):
- **Live `"last"` `R_mf` vs. `R_ridge` (`C0b.2a`, counterfactual `PASS` on registered bars):** `abs_ratio_geo = 1.7013×` (90% BCa CI `[1.6720, 1.7332]`, bar `>= 1.25×`), `norm_ratio_geo = 1.8395×` (90% BCa CI `[1.8067, 1.8690]`, bar `>= 1.25×`), `16/16` targets with `B_cross(R_mf) > B_cross(R_ridge)` (bar `>= 13/16`), one-sided exact sign-flip `p = 1.5e-5` (bar `<= 0.05`); `self_dlogit` geometric ratio `R_ridge / R_mf = 1.0812×`. In the paired `"all"` Regime-N secondary (`r_i` max shrink `19.0%` on `R_mf`, `11.0%` on `R_ridge`), `abs_ratio_geo = 2.5595×` and `norm_ratio_geo = 1.8763×` (`16/16`).
- **Comparison against Stage C0's `3L` `rand_iso` floor (`C0b.2b`, counterfactual `FAIL`):** Against Stage C0's published `L ∈ {18, 24, 27}` isotropic random-direction floor **`F_mean_3L = 0.2958`** (512-pair `F_p95_3L = 0.4310`; registered bar `F_mean_3L + 0.10 = 0.3958`), live `"last"` `R_ridge` measured `B_cross = 0.7898` (`2.67× F_mean_3L`, `0.588×` live `R_mf`), which is above `0.3958`. Live `"last"` `logit_lens` measured `B_cross = 0.3471` (`1.17× F_mean_3L`, `0.258×` live `R_mf`, signed cross mean `+0.0950`, `60.0%` positive).
- **`3L` (`{18, 24, 27}`) own-target `frac_err` (cell-level vs. per-trace estimators):** On `score_olmo3_c0b.py`'s `48`-cell `(L, target)` estimator, moving from `"last"` to `"all"` in Regime N (`0.75×`) moved `R_mf` median `frac_err` from `0.3271` to `0.0338`, and `R_ridge` from `0.0958` to `0.0809`. On Stage C0's per-trace median estimator (`score_olmo3_c0.py:883-890`, which gave `0.363` on Stage C0's `4L` `{18, 21, 24, 27}` `"last"` `R_mf`), the corresponding `3L` (`384`-trace) medians in Stage C0b are `0.3312` (`"last"`) vs. `0.0784` (`"all"`) for `R_mf` and `0.1039` (`"last"`) vs. `0.0858` (`"all"`) for `R_ridge`.

---

## 3. Implications for Any Downstream Stage C1 Design

1. **Gate-driven instrument fix (grounded in `C0b.0.3`):** `L12` at `Direct_linear = +3.0` under `positions="all"` (`dose = 0.97–1.64×` `post_mlp_norm`) shrinks `r_i` by up to `35.1%` and leaves the linear regime assumed by the decomposition. Any future `positions="all"` linear-regime study on OLMo 3 7B must drop high-dose `L12` `positions="all"` (or lower per-position dose / verify `r_i / r_c` bounds at `L12`) before pre-registration.
2. **No fitting to blocked readouts:** The post-hoc descriptive readouts in Section 2 (`C0b.1a/1b`, `C0b.2a/2b`, and Regime-N `"all"` `frac_err`) were observed on the same 16 singular concrete nouns and 8 prompts used in Stage C0. They are carried forward only as **leads**, not as confirmatory results or bars to re-register on the same units. Any Stage C1 design must introduce **fresh targets (including mixed parts of speech)** and fresh prompts (and/or a fresh layer set), with pre-registered bars written before any new GPU execution.
