# Stage C2 Report: Unified 3-Pillar Causal Dissection of Common-Mode Bleed, ChatML Amplification, 5-Layer Post-Erasure Band, and Low-`γ` `pre_norm` Masking on OLMo 3 7B

**Status:** Pre-registered fresh-unit confirmatory run completed (`OLMO3-TRANSPORT-C2-DESIGN.md` Registered v4, frozen on `main` @ `ecd49b28`, Council `#119`; frozen runner/scorer/tests signed off at **Code PASS** on `vesper-c2-wsp` @ `f5a7d7da`, Council `#125`, `2026-11-25 19:11 CT`; GPU run approved by Elise at `19:20 CT` and executed verbatim at `f5a7d7da`, completing `19:45 CT`). All frozen registration and code-freeze hashes remain completely untouched on disk.  
**Headline Verdict:** **Global Gate `C2.0`: `FAIL`** (sole failed sub-clause: **`C2.0.2` `tele_p999 = 1.038869 > 0.15`**; `nonfinite_count = 0`, `tele_p95 = 0.067939 <= 0.08` PASS; `C2.0.1`, `C2.0.3`, and `C2.0.4` all PASS cleanly) · **All five primary confirmatory verdicts (`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`): `BLOCKED_BY_C2.0`**.  
**Compute:** `8,623` executed `B=1` `bf16` forwards (`0` capped skipped: `36` clean prompts, `13` write shams, `6` patch shams, `16` C0-colon anchor, `2,880` Block 1 `C2.1`, `2,592` Block 2 `C2.2` steered, `72` Block 2 `C2.2` unsteered KL, `1,280` Block 2b `C2.3` [`960` `logit_lens` + `320` `rand_iso`], `1,728` Block 3 `C2.4`), `744.45 s` (`12.41 min`) GPU compute on NVIDIA L4 (`acct1`) + `399 s` (`6.65 min`) 64-chunk upload of `2.11 GB` (`1.97 GiB`) `r_lens.pt` + model fetch/teardown → **`0.77 CU`** settled via Colab usage ledger (`165.87 → 165.10 CU`, `Usage rate: 0.00/hr`, `Active assignments: 0`, re-verified >1 hr post-teardown — within the `0.45–1.05 CU` pre-run estimate; run-ledger attestations external to the JSON bundle).  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Provenance

| Artefact | Path | Size | SHA-256 |
|---|---|---:|---|
| Pre-registered Design (v4, `ecd49b28`) | `olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-DESIGN.md` | `30,247` B | `3764dfee6baf9c7387fc6325eee92c4ca2fdbca5c76ee5b440e0f4f12169d8fc` |
| Frozen Constants Builder (`ecd49b28`) | `olmo3_virtual_and_highway/build_olmo3_c2_constants.py` | `55,246` B | `5380843ea091ce82569df823ae87679d66bedf1a9e71aa3c001aa771f66a11d1` |
| Frozen Constants (`v4`, `ecd49b28`) | `olmo3_virtual_and_highway/c2_prereg_constants.json` | `59,243` B | `583117f50cfceeb5cddcfa239c0efa0c32846dd69f6756540cd3bbba5cec5b65` |
| Frozen Runner (Code PASS `f5a7d7da`) | `olmo3_virtual_and_highway/run_olmo3_c2.py` | `48,619` B | `2755bda6c5e6bae6d3ecfb8ca7c56c6462262b11eb95199125a4dbb13994026b` |
| Frozen Scorer (stdlib-only, `f5a7d7da`) | `olmo3_virtual_and_highway/score_olmo3_c2.py` | `65,771` B | `5a3e6c67f62954f0ce060f610cfd9600f6c18749d29477bc486a489bebd1cf96` |
| Pre-launch Test: Design Pins (`f5a7d7da`) | `olmo3_virtual_and_highway/test_olmo3_c2_design_stdlib.py` | `10,172` B | `e993305131a71165dbc3a9c086fac77f5722ce413a8c04ee617110f56fc218bc` |
| Pre-launch Test: Constants (`f5a7d7da`) | `olmo3_virtual_and_highway/test_olmo3_c2_constants.py` | `10,361` B | `12f6ded7085c668151cfa04d4f4db8c4e81f58153501cd2e693d95f95f9fbac3` |
| Pre-launch Test: Runner CPU (`f5a7d7da`) | `olmo3_virtual_and_highway/test_olmo3_c2_runner_cpu.py` | `14,654` B | `fdb9f9414fd8509277cef1d09e003825f5d98e96e7361799e5992b1c216bbb65` |
| Pre-launch Test: Scorer Stdlib (`f5a7d7da`) | `olmo3_virtual_and_highway/test_olmo3_c2_scorer_stdlib.py` | `55,229` B | `41378246f639fc9dab2034375662fa6ba2a868368ff16420c0b456ac23628f18` |
| Post-hoc Autopsy & Counterfactual Driver | `olmo3_virtual_and_highway/exploratory_c2_counterfactual.py` | `8,138` B | `bcf9fcc4efdac33d40cc49db8ea13f93a29a4264c5fb45d4cb1e39d32da90682` |
| Frozen RelP Lens (external) | `r_lens.pt` | `2,113,932,853` B (`2.11 GB` / `1.97 GiB`) | `ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955` |
| Frozen Vector Bank (external) | `work/olmo3_c2/c2_vector_bank.safetensors` | `9,981,219` B | `7124dd565331fd455d6d91d8fe845ba8783c2b0d366f6d11eb533947cb09401d` |
| Stage C2 Receipt (`.gz`) | `results/vesper-olmo3-7b-stage-c2/c2_receipt.json.gz` | `743,047` B | `f129db3f41a424ed5410436bd37f2c9625f33c405198e028da6e6dca733b04e5` |
| Stage C2 Receipt (uncompressed) | `gzip -dc c2_receipt.json.gz` | `3,913,068` B | `257c21046f9eb3cb71b0f74eb110d3558673bd3632d57a02f358bc6724c20a94` |
| Official Scorecard JSON | `results/vesper-olmo3-7b-stage-c2/scorecard.json` | `2,093` B | `2859050926cacdb72b6431952d61944e90c9a2b8f54e050721effd2ff5f9ebf1` |
| Official Scorecard Markdown | `results/vesper-olmo3-7b-stage-c2/SCORECARD.md` | `464` B | `6e0e7ecd789d5669558550dd02a5d9114dabe3073e445d4349fb5d4a8a5385e5` |

Scorecard determinism verified via two independent `/usr/bin/python3 -I` runs (`cmp` exit `0` on both `scorecard.json` and `SCORECARD.md`).

---

## 1. Pre-Registered Confirmatory Gate Summary & What Failed

Global Gate `C2.0` failed on a single sub-clause — **`C2.0.2` telescoping `p99.9` (`tele_p999 = 1.038869 > 0.15`)** — and held all five primary confirmatory verdicts (`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`) closed (`BLOCKED_BY_C2.0`). Every other clause of `C2.0` passed cleanly with zero capped forwards, zero RMSN > 12% violations, zero FDE > 0.45 violations, exact `0.0` shams, and full provenance integrity.

| Check | Official Verdict | Registered Result vs. Bar |
|---|---|---|
| **`C2.0.1` (Shams & liveness)** | **PASS** *(clause)* | `shams_present_ok = true`, `active_keys_exact_ok = true`, `max_write_sham = 0.0` (`13/13`), `max_patch_sham = 0.0` (`6/6`), `liveness_ok = true` (`61/61` active keys live). |
| **`C2.0.2` (Finite & telescoping)** | **FAIL** *(global gate)* | `nonfinite_count = 0`; `tele_p95 = 0.067939 <= 0.08` (PASS); **`tele_p999 = 1.038869 > 0.15` (FAIL)** across the `6,432` telescoping traces (`2,880` Block 1 + `2,592` Block 2 + `960` Block 2b). |
| **`C2.0.3` (Linearity & completeness)** | **PASS** *(clause)* | `all_block_counts_ok = true`, `per_cell_counts_ok = true`, `pmn_complete_ok = true` (all 32 layers), `dose_checks_ok = true`. Regime-N (`N = 2,880`): `0` capped, `0` RMSN > 20%, `0` RMSN > 12%. Regime-D (`N = 864`): `0` capped, `0` patched capped, `0` RMSN > 20%, `0` RMSN > 12%, `fde_p95 = 0.227831 <= 0.30`, `0` FDE > 0.45, `0` FDE > 0.75. |
| **`C2.0.4` (Strict provenance)** | **PASS** *(clause)* | `4 / 4` strict SHA-256 checks (`r_lens.pt`, `c2_prereg_constants.json`, `c2_vector_bank.safetensors`, `OLMO3-TRANSPORT-C2-DESIGN.md`) matched on disk and in receipt metadata. |
| **Global Gate `C2.0`** | **`FAIL`** | Blocked solely by `C2.0.2` `tele_p999 = 1.038869 > 0.15`. |
| **`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`** | **`BLOCKED_BY_C2.0`** | Held closed by the fail-closed global gate; never rescued. |

*Registered Notation Erratum (flagged during Code Review Round 1, Council `#121 #8`; frozen design left untouched):* `OLMO3-TRANSPORT-C2-DESIGN.md:146` writes `KL(p_patched || p_ref)` for the Block 2 unsteered KL control, whereas `run_olmo3_c2.py` (matching Stage C0/C1) computes forward `KL(p_ref || p_patched)`.

### Post-Hoc Root-Cause Autopsy of `C2.0.2` (`tele_p999 = 1.038869 > 0.15`)

Reproducible via `/usr/bin/python3 -I olmo3_virtual_and_highway/exploratory_c2_counterfactual.py`. Decomposing `telescoping_err` across all `6,432` telescoping traces pinpoints a single instrumentation mismatch in one Block 2 negative-control condition:

| Block / Condition | `N` | `p50` | `p95` | `p99` | `p99.9` | `max` |
|---|---:|---:|---:|---:|---:|---:|
| Block 1 (`C2.1`, 5 arms × 3 layers) | `2,880` | `0.0172` | `0.0498` | `0.0643` | `0.0855` | `0.0920` |
| Block 2 `unpatched_LongBare` | `288` | `0.0145` | `0.0521` | `0.0679` | `0.0915` | `0.0973` |
| Block 2 `unpatched_ChatML` | `288` | `0.0160` | `0.0459` | `0.0605` | `0.0750` | `0.0788` |
| Block 2 `unpatched_ShortBare` | `288` | `0.0187` | `0.0513` | `0.0773` | `0.1064` | `0.1087` |
| Block 2 `patch_LongBare_to_ChatML_MLP_L19_31` | `288` | `0.0170` | `0.0470` | `0.0549` | `0.0710` | `0.0746` |
| Block 2 `patch_ChatML_to_LongBare_MLP_L19_31` | `288` | `0.0154` | `0.0493` | `0.0621` | `0.1145` | `0.1274` |
| **Block 2 `ctrl_L4_16_MLP_LongBare_to_ChatML`** | **`288`** | **`0.3267`** | **`0.9105`** | **`1.1363`** | **`1.1872`** | **`1.1889`** |
| Block 2 `ctrl_mismatch_prompt_MLP_L19_31` | `288` | `0.0149` | `0.0423` | `0.0571` | `0.0750` | `0.0757` |
| Block 2 `ctrl_rand_iso_same_site_L19_31` | `288` | `0.0159` | `0.0459` | `0.0567` | `0.0804` | `0.0814` |
| Block 2 `ctrl_self_attenuated_L19_31` | `288` | `0.0155` | `0.0474` | `0.0700` | `0.0750` | `0.0762` |
| Block 2b (`C2.3`, `logit_lens`, 5 layers) | `960` | `0.0205` | `0.0591` | `0.0738` | `0.0969` | `0.0971` |
| **All traces excluding `ctrl_L4_16`** | **`6,144`** | **`0.0171`** | **`0.0506`** | **`0.0676`** | **`0.0931`** | **`0.1274`** |

**Exact mechanism:** `ctrl_L4_16_MLP_LongBare_to_ChatML` patches MLP outputs at layers **`L4–L16`**, which sit **upstream** of the steering injection at `inject_layer = 18`. That upstream patch alters the residual stream entering `L18` relative to the unpatched baseline, so the true output logit shift `exact_dlogit` equals the `L4–L18` residual delta *plus* the `L18` direct injection *plus* the `L19–L31` attention/MLP deltas. However, `decompose_target_trace(..., inject_layer=18)` reconstructs `recon_dlogit` by summing only `L19–L31` deltas plus the `L18` direct injection, omitting the `L4–L18` residual shift from `recon_dlogit` and pushing `telescoping_err = |exact_dlogit - recon_dlogit| / max(|exact_dlogit|, 1)` up to `1.1889`. Because `288 / 6,432 = 4.48%` of all traces belong to `ctrl_L4_16`, `p95` barely cleared `0.08` (`0.067939`), while `p99.9` landed squarely inside `ctrl_L4_16` (`1.038869`). Across all other `6,144` traces (`95.52%`), telescoping was tight (`p95 = 0.0506 <= 0.08`, `p99.9 = 0.0931 <= 0.15`, `max = 0.1274`). Pre-launch synthetic tests did not catch this because scorer unit tests populated `telescoping_err` synthetically and CPU runner tests verified hook dispatch on tiny mocks without asserting the `0.15` `p99.9` bound on an upstream-of-injection MLP patch.

---

## 2. Post-Hoc Exploratory Readouts on Blocked Metrics (NOT Confirmatory Verdicts)

Because Global Gate `C2.0` failed, **`C2.1`, `C2.2`, `C2.3`, `C2.4a`, and `EQ-C2.4c` remain permanently `BLOCKED_BY_C2.0`**. In the frozen scorer, `score_olmo3_c2.py` returns early with blocked verdicts whenever `C2.0` fails.

**Exact counterfactual method (`olmo3_virtual_and_highway/exploratory_c2_counterfactual.py`):** To inspect the blocked downstream metrics without modifying any frozen file on disk, the committed stdlib driver loads the frozen receipt, runs `score_c2()` with `strict_provenance=True` on all disk hashes, and zeroes **only** the `telescoping_err` field on the `288` `ctrl_L4_16_MLP_LongBare_to_ChatML` rows in memory so `C2.0.2` opens (`p95 = 0.0506`, `p99.9 = 0.0931`). **Invariance corollary:** every Pillar 1–4 metric, bootstrap CI, Holm-7 p-value, and diagnostic below is **mathematically invariant** to this edit because `telescoping_err` is read exclusively by `C2.0.2` and never enters any Pillar 1–4 formula. All numbers below are **exploratory post-hoc diagnostics only** — not confirmatory verdicts.

Counterfactual Holm 7-test family (exploratory only — gate forced open by zeroing `ctrl_L4_16` `telescoping_err`):

| Rank | Test | Exploratory `p` | Holm `α_k` | Would Reject if Gate Open? | Counterfactual Label |
|---:|---|---:|---:|---|---|
| 1 | `p_C2.1_gain_ret` (`R_ridge_shrink` gain retention `>= 0.85`) | `1.00e-5` | `0.007143` | Yes | *(leg of `C2.1` → `FAIL`)* |
| 2 | `p_C2.4a_mask_vs_unmask` (`L24` low-`γ` `pre_norm` masking `>= 1.25×`) | `1.00e-5` | `0.008333` | Yes | **`PASS`** |
| 3 | `p_C2.1_ctr_vs_raw` (`R_ridge_centered` vs. `R_ridge_raw` `>= 1.25×`) | `1.526e-5` | `0.010000` | Yes | *(leg of `C2.1` → `FAIL`)* |
| 4 | `p_C2.1_shk_vs_ctr` (`R_ridge_shrink` vs. `R_ridge_centered` `>= 1.15×`) | `0.078506` | `0.012500` | No (stop) | `C2.1`: **`FAIL`** |
| 5 | `p_C2.1_shk_vs_ll` (`R_ridge_shrink` vs. `logit_lens` `>= 1.10×`) | `0.095413` | `0.016667` | No | *(leg of `C2.1` → `FAIL`)* |
| 6 | `p_C2.2_bare_into_chat` (`LongBare → ChatML` `L19–L31` MLP patch) | `1.000000` | `0.025000` | No | `C2.2`: **`FAIL`** |
| 7 | `p_C2.2_chat_into_bare` (`ChatML → LongBare` `L19–L31` MLP patch) | `1.000000` | `0.050000` | No | *(leg of `C2.2` → `FAIL`)* |

*(Additional non-Holm exploratory verdicts: `C2.3`: **`C2.3_UNLOCALIZED`** [fallback band `[24, 27]`]; `EQ-C2.4c`: **`ADDITIVE`** → `LICENSE_K3_SIMULTANEOUS`.)*

### 2.1 Blocked `C2.1` Readout (Exploratory Counterfactual: `FAIL`)

- **Mean-centering (`u_t - μ_U`) does essentially all of the work; diagonal shrinkage adds almost nothing:**
  - `ctr_vs_raw` (`R_ridge_centered` vs. `R_ridge_raw`): **`2.0497×`** (`90% BCa [1.8675, 2.2531]`, exploratory `p = 1.526e-5`, `bar_pass = true`).
  - `gain_ret` (`R_ridge_shrink` vs. `R_ridge_raw` on-target diagonal retention): **`0.9409×`** (`90% BCa [0.9374, 0.9442]`, exploratory `p = 1.00e-5`, `bar_pass = true`; `0` guarded sources across all 5 arms, `degenerate_diag = false`).
  - `shk_vs_ctr` (`R_ridge_shrink` vs. `R_ridge_centered`): **`1.0308×`** (`90% BCa [0.9987, 1.0660]`, exploratory `p = 0.078506`, `bar_pass = false`; `same_over_cross_ratio = 0.9883× [0.9691, 1.0076]`).
  - `shk_vs_ll` (`R_ridge_shrink` vs. `logit_lens`): **`1.1043×`** (`90% BCa [0.9847, 1.2431]`, exploratory `p = 0.095413`, `bar_pass = false`).
- **Secondary 5-arm pooled held-out `cross_abs / diag` ordering (`strict_order_ok = false` because `centered` and `shrink` tie at `0.0382`):** `R_mf = 0.1700`, `R_ridge_raw = 0.0788`, `R_ridge_centered = 0.0382`, `R_ridge_shrink = 0.0382`, `logit_lens = 0.0409`.
- **Per-prompt stability (`F6`, all 8 prompts):** `raw_over_ctr` stays strongly above `1.65×` on every prompt (`1.654×–2.562×`: LongBare `2.068, 1.949, 2.562, 1.654`; ChatML `2.159, 1.713, 2.276, 1.929`), while `ctr_over_shk` hovers near `1.0` (`0.993×–1.084×`: LongBare `1.016, 0.999, 0.993, 1.001`; ChatML `1.002, 1.023, 1.082, 1.084`).
- **Calibration gate `CAL-C2.1: False` (sole miss: `mu_shk = 0.4943 > 0.45`):** `ho_cross_raw_mean = 0.0767`, `ols_slope_raw = 0.9535`, `ols_r2_raw = 0.4926`, `scaled_diag_shk_3L = 1.4146` vs. frozen `1.4591`. The per-layer `|cos(v1, μ_U)|` and `rank1_svd_share` reveal why `L18` broke the `0.45` mean cap: **downstream layers (`L19–L31`) re-inject the `μ_U` common mode after `L18`**, whereas at `L24` and `L27` mean-centering cleanly collapses `|cos(v1, μ_U)|` to `0.20–0.39` and cuts held-out `cross_abs/diag` to `~0.020–0.024`:

| Arm | Held-out `cross_abs/diag` (`L18 / L24 / L27`) | `\|cos(v1, μ_U)\|` `3L` (`L18 / L24 / L27`) | Rank-1 SVD Share `3L` (`L18 / L24 / L27`) |
|---|---|---|---|
| `R_mf` | `0.1597 / 0.1527 / 0.1820` | `0.9689` (`0.9722 / 0.9658 / 0.9687`) | `0.9189` (`0.8951 / 0.9220 / 0.9397`) |
| `R_ridge_raw` | `0.1035 / 0.0638 / 0.0627` | `0.9689` (`0.9641 / 0.9688 / 0.9738`) | `0.8302` (`0.8367 / 0.8169 / 0.8369`) |
| `R_ridge_centered` | `0.0672 / 0.0240 / 0.0216` | `0.4749` (`0.8483 / 0.2578 / 0.3186`) | `0.3545` (`0.5093 / 0.2716 / 0.2825`) |
| `R_ridge_shrink` | `0.0673 / 0.0231 / 0.0198` | `0.4943` (`0.8889 / 0.3934 / 0.2005`) | `0.3504` (`0.5582 / 0.2567 / 0.2362`) |
| `logit_lens` | `0.0705 / 0.0282 / 0.0210` | `0.6743` (`0.9058 / 0.9223 / 0.1948`) | `0.5009` (`0.7541 / 0.4686 / 0.2800`) |

### 2.2 Blocked `C2.2` Readout (Exploratory Counterfactual: `FAIL` — Length Confound Exposed)

The length-matched 3-way baseline (`ShortBare`, `LongBare`, `ChatML`) cleanly dissects the Stage C1 `L18` all-position format gap (`+0.604`):

- **Prompt-length confound explains the positive Stage C1 gap, and pure ChatML formatting actually *reduces* `M(C)` once length is matched:**
  - `unpatched_ShortBare`: `mean M(C) = 1.3471` (`med 1.3010`, range `[1.1335, 1.6380]`).
  - `unpatched_LongBare`: `mean M(C) = 1.7710` (`med 1.7543`, range `[1.3676, 2.0658]`).
  - `unpatched_ChatML`: `mean M(C) = 1.4579` (`med 1.4542`, range `[1.2437, 1.6636]`).
  - **`G_len (LongBare − ShortBare) = +0.4240`** (`90% BCa [+0.3011, +0.5441]`): longer prompts strongly amplify `positions="all"` injection gain.
  - **`G_gap (ChatML − LongBare) = -0.3132`** (`90% BCa [-0.3832, -0.2222]`, `gap_gate_pass = true`): opposite sign of Stage C1's un-length-matched `ChatML - ShortBare` difference!
- **Late `L19–L31` MLP patching does NOT mediate `G_gap`:**
  - `patch_LongBare_to_ChatML_MLP_L19_31`: `mean M(C) = 1.4682`, closure **`+0.0329`** (`90% BCa [-0.2916, +0.1697]`, exploratory `p = 1.0`, `bar_pass = false`).
  - `patch_ChatML_to_LongBare_MLP_L19_31`: `mean M(C) = 1.7452`, closure **`+0.0827`** (`90% BCa [-0.2523, +0.2209]`, exploratory `p = 1.0`, `bar_pass = false`).
- **Equivalence/control gate `EQ-C2.2: False`:**
  - `ctrl_L4_16_MLP_LongBare_to_ChatML` (`mean M(C) = 1.5608`): closure **`+0.3286`** (`[+0.1285, +0.5395]`, `pass = false`) — early/mid `L4–L16` MLPs explain ~33% of `|G_gap|`!
  - `ctrl_rand_iso_same_site_L19_31` (`mean M(C) = 1.2450`): `-0.6797` (`[-1.1943, -0.4861]`, `pass = false`).
  - `ctrl_mismatch_prompt_MLP_L19_31` (`mean M(C) = 1.4745`): `+0.0531` (`[-0.2125, +0.3998]`, `pass = true`).
  - `ctrl_self_attenuated_L19_31` (`mean M(C) = 1.4050`): `-0.1688` (`[-0.2709, -0.1099]`, `pass = true`).
  - POS closure split `cl_open = -0.1550`, `cl_func = +0.2436`; `median_unsteered_patched_kl = 0.3417 > 0.25` (`shape_and_kl_pass = false`).

### 2.3 Blocked `C2.3` Readout (Exploratory Counterfactual: `C2.3_UNLOCALIZED`, Fallback Band `[24, 27]`)

Across all five probed layers (`L18, L21, L24, L27, L30`), selective erasure of open-class vs. function-word directions (`gap = E_open - E_func`) clearly terminates at **`L24–L27`** (flipping from strongly negative at `L18`/`L21` to slightly positive at `L24`/`L27`), **but** open-class retention `ret_open` remains strictly below isotropic random retention `ret_rand` at every layer, so no layer satisfies the joint `post_erasure` criterion (`ret_open >= ret_rand`) and the selector falls back to `[24, 27]`:

| Layer | `E_open` | `E_func` | `gap = E_open − E_func` | `ret_open` | `ret_rand` | `post_erasure` |
|---:|---:|---:|---:|---:|---:|---|
| `L18` | `-3.2976` | `-0.3274` | `-2.9702` | `0.6848` | `0.7288` | `false` |
| `L21` | `-1.8985` | `-0.7971` | `-1.1014` | `0.8408` | `0.9113` | `false` |
| `L24` | `-1.1666` | `-1.2762` | **`+0.1096`** | `0.9139` | `0.9877` | `false` |
| `L27` | `-1.3203` | `-1.4050` | **`+0.0847`** | `0.8809` | `0.9980` | `false` |
| `L30` | `-1.1236` | `-0.7302` | `-0.3934` | `0.9293` | `0.9969` | `false` |

### 2.4 Blocked `C2.4` Readout (Exploratory Counterfactual: `C2.4a: PASS`, `EQ-C2.4c: ADDITIVE`)

Pillar 3 produced the cleanest and strongest exploratory signals of Stage C2 — directly validating the core Stage C3 (`WSP v1`) primitives:

- **`C2.4a` (`L24` low-`γ` `pre_norm` masking, counterfactual `PASS`):**
  - Pooled masked / unmasked ratio: **`2.1505×`** (`90% BCa [1.8948, 2.4787]`, exploratory `p = 1.00e-5 <= α_2 = 0.008333`, `bar_pass = true`).
  - Positive across `4 / 4` POS strata: nouns **`2.8807×`** (`[1.6539, 5.5116]`), verbs **`2.4401×`** (`[1.3558, 4.3246]`), adjectives **`1.6685×`** (`[1.1098, 2.2490]`), function **`1.8235×`** (`[1.3171, 2.4020]`).
  - Secondary post-RMSNorm norm-matched control (`secondary_C2.4a_postmatched_diag`): `0` capped, `precondition_ok = true`, `geomean_masked_over_postmatched = 1.0648×` — confirming that almost the entire `2.15×` leverage comes from avoiding high-`γ` RMSNorm attenuation rather than unintended post-norm boost.
- **`C2.4b` (`L24` `m70a30` FF+Attn split, descriptive):**
  - At `0.5×` dose: `R_target = 1.2088×` (`90% BCa [1.2042, 1.2137]`), while random pack baseline drops slightly (`R_pack_rand_l2_norm = 0.9535×`, ratio **`1.2677×`**; `R_pack_rand_abs_target = 0.9718×`, ratio **`1.2439×`**).
  - At `1.25×` dose: `R_target = 1.2213×` (`90% BCa [1.2171, 1.2271]`), while random pack drops further (`R_pack_rand_l2_norm = 0.9118×`, ratio **`1.3393×`**; `R_pack_rand_abs_target = 0.8690×`, ratio **`1.4054×`**).
- **`Gate EQ-C2.4c` (`K=3` simultaneous additivity TOST, counterfactual `ADDITIVE` → `LICENSE_K3_SIMULTANEOUS`):**
  - At `0.5×` dose: median relative interaction **`+0.0073`** (`90% BCa [+0.0056, +0.0114]`, `n_floored = 0`) — `< 1%` nonlinearity, clearing the `±15%` TOST equivalence band by `> 13×`.
  - Secondary `1.25×` calibration: median relative interaction **`+0.0015`** (`n_floored = 0`).

### 2.5 Non-Gating `C0`-Colon Anchor Diagnostic

Live Stage C2 `post_mlp_norms` (`L18 = 8.0536`, `L21 = 8.8667`, `L24 = 9.4211`, `L27 = 9.5629`, `L30 = 19.2889`) tracked Stage C0/C0b (`L24 = 9.3257`, `L27 = 10.2374`) closely rather than Stage C1's depressed late norms (`7.2103` / `7.4738`). On the 16 `c0_colon_anchor` prompts, `pooled_median_ratio` was `0.9618` at `L24` and `0.9304` at `L27` (bare-only `0.9269` at both layers; `in_predicted_range_L24 = false`, `in_predicted_range_L27 = false`), refuting the informal hypothesis that trailing colons alone drove Stage C1's `23–27%` late-layer norm reduction.

---

## 3. Compute, Engineering Takeaways & Stage C3 Design Implications

- **Compute & hygiene:** Settled cost was **`0.77 CU`** (`165.87 → 165.10 CU` on `acct1`, `Usage rate: 0.00/hr`, `0` active runtimes re-verified >1 hr post-teardown) — squarely inside the `0.45–1.05 CU` pre-run estimate. `399 s` (`35%` of total wall-clock time) was again spent uploading the `2.11 GB` `r_lens.pt` file; bundling only the needed layer slices into the vector bank remains an easy future speedup.
- **Instrumentation lesson (`C2.0.2`):** Any future telescoping gate that includes patches upstream of `inject_layer` (`L < inject_layer`) must either integrate `L_patch..L31` inside `decompose_target_trace` or exclude upstream-patch controls from post-injection telescoping percentiles. Pre-launch runner CPU tests should assert `telescoping_err` end-to-end on every patched condition, including upstream-layer controls.
- **Exploratory design takeaways for Stage C3 (`WSP v1` literal Persona + No-Markdown demo; `11-07` weight-release policy):**
  1. **Use plain mean-centering (`u_t - μ_U`) rather than diagonal shrinkage:** In the blocked exploratory `C2.1` data, mean-centering delivered `2.05×` bleed reduction with `94.1%` on-target gain retention and tied shrinkage (`0.0382` vs. `0.0382`), while `L24`/`L27` avoided `L18`'s downstream `μ_U` re-injection (`cross_abs/diag = 0.020–0.024`).
  2. **Target `L24–L27` with low-`γ` `pre_norm` masking + `m70a30` FF+Attn split:** Selective open-class erasure stops at `L24–L27`, low-`γ` `pre_norm` masking adds `2.15×` clean leverage (all 4 POS strata), and `m70a30` adds another `1.21–1.22×` gain with reduced off-target pack norm (`1.27–1.41×` target/pack ratio).
  3. **`K=3` simultaneous writes compose nearly linearly:** Median relative interaction is `+0.73%` at `0.5×` and `+0.15%` at `1.25×` (`0` floored), supporting simultaneous multi-concept weight-space edits in Stage C3 without iterative re-orthogonalization.
