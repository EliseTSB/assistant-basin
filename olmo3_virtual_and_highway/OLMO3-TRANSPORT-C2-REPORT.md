# Stage C2 Report (OLMo 3 7B): Global Gate `C2.0` FAIL — All Confirmatory Pillars `BLOCKED_BY_C2.0`; Post-Hoc Autopsy of `C2.0.2` and Explicitly Non-Confirmatory Exploratory Readouts

**Status:** Pre-registered fresh-unit confirmatory run completed (`OLMO3-TRANSPORT-C2-DESIGN.md` Registered v4, frozen on `main` @ `ecd49b28`, Council `#119`; frozen runner/scorer/tests signed off at **Code PASS** on `vesper-c2-wsp` @ `f5a7d7da`, Council `#125`, `2026-11-25 19:11 CT`; GPU run approved by Elise at `19:20 CT` and executed verbatim at `f5a7d7da`, completing `19:45 CT` — process timestamps and Council IDs are external records). All frozen registration and code-freeze hashes remain completely untouched on disk.  
**Headline Verdict:** **Global Gate `C2.0`: `FAIL`** (sole failed sub-clause: **`C2.0.2` `tele_p999 = 1.038869 > 0.15`**; `nonfinite_count = 0`, `tele_p95 = 0.067939 <= 0.08` PASS; `C2.0.1`, `C2.0.3`, and `C2.0.4` all PASS) · **All five top-level scorecard confirmatory verdicts (`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`): `BLOCKED_BY_C2.0`**. The design's two other registered gates outside the Holm family (`CAL-C2.1` and `EQ-C2.2`, design:159) are sub-components of the `C2.1` and `C2.2` pillars and are likewise blocked and uninterpretable under `C2.0: FAIL`; any `False` values reported for them in §2.1–§2.2 are exploratory counterfactual outputs, not confirmatory outcomes.  
**Compute:** `8,623` executed `B=1` `bf16` forwards (`0` capped skipped: `36` clean prompts, `13` write shams, `6` patch shams, `16` C0-colon anchor, `2,880` Block 1 `C2.1`, `2,592` Block 2 `C2.2` steered, `72` Block 2 `C2.2` unsteered KL, `1,280` Block 2b `C2.3` [`960` `logit_lens` + `320` `rand_iso`], `1,728` Block 3 `C2.4`), `744.45 s` (`12.41 min`) GPU compute on NVIDIA L4 (`acct1`) + `399 s` (`6.65 min`) 64-chunk upload of `2.11 GB` (`1.97 GiB`) `r_lens.pt` + model fetch/teardown → **`0.77 CU`** settled via Colab usage ledger (`165.87 → 165.10 CU`, `Usage rate: 0.00/hr`, `Active assignments: 0`, re-verified >1 hr post-teardown — within the `0.45–1.05 CU` pre-run estimate; upload timing and run-ledger attestations are external to the JSON bundle).  
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

Global Gate `C2.0` failed on a single sub-clause — **`C2.0.2` telescoping `p99.9` (`tele_p999 = 1.038869 > 0.15`)** — and held all confirmatory pillars (`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`, together with sub-gates `CAL-C2.1` and `EQ-C2.2`) closed (`BLOCKED_BY_C2.0`). Every other clause of `C2.0` passed cleanly with zero capped forwards, zero RMSN > 12% violations, zero FDE > 0.45 violations, exact `0.0` shams, and full provenance integrity.

| Check | Official Verdict | Registered Result vs. Bar |
|---|---|---|
| **`C2.0.1` (Shams & liveness)** | **PASS** *(clause)* | `shams_present_ok = true`, `active_keys_exact_ok = true`, `max_write_sham = 0.0` (`13/13`), `max_patch_sham = 0.0` (`6/6`), `liveness_ok = true` (`61/61` active keys live). |
| **`C2.0.2` (Finite & telescoping)** | **FAIL** *(global gate)* | `nonfinite_count = 0`; `tele_p95 = 0.067939 <= 0.08` (PASS); **`tele_p999 = 1.038869 > 0.15` (FAIL)** across the `6,432` telescoping traces (`2,880` Block 1 + `2,592` Block 2 + `960` Block 2b). |
| **`C2.0.3` (Linearity & completeness)** | **PASS** *(clause)* | `all_block_counts_ok = true`, `per_cell_counts_ok = true`, `pmn_complete_ok = true` (all 32 layers), `dose_checks_ok = true`. Regime-N (`N = 2,880`): `0` capped, `0` RMSN > 20%, `0` RMSN > 12%. Regime-D (`N = 864`): `0` capped, `0` patched capped, `0` RMSN > 20%, `0` RMSN > 12%, `fde_p95 = 0.227831 <= 0.30`, `0` FDE > 0.45, `0` FDE > 0.75. |
| **`C2.0.4` (Strict provenance)** | **PASS** *(clause)* | `4 / 4` pinned SHA-256 hashes matched in receipt metadata; `OLMO3-TRANSPORT-C2-DESIGN.md` and `c2_prereg_constants.json` verified directly on disk in the repo bundle, while external `r_lens.pt` and `c2_vector_bank.safetensors` were verified preflight by `run_olmo3_c2.py:59–105` on the execution host. |
| **Global Gate `C2.0`** | **`FAIL`** | Blocked solely by `C2.0.2` `tele_p999 = 1.038869 > 0.15`. |
| **`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`** *(and sub-gates `CAL-C2.1`, `EQ-C2.2`)* | **`BLOCKED_BY_C2.0`** | Held closed by the fail-closed global gate; never rescued. |

*Registered Notation Erratum (flagged during Code Review Round 1, Council `#121 #8`; frozen design left untouched):* `OLMO3-TRANSPORT-C2-DESIGN.md:146` writes `KL(p_patched || p_ref)` for the Block 2 unsteered KL control, whereas `run_olmo3_c2.py` (matching Stage C0/C1) computes forward `KL(p_ref || p_patched)`.

### Post-Hoc Root-Cause Autopsy of `C2.0.2` (`tele_p999 = 1.038869 > 0.15`)

Reproducible via `/usr/bin/python3 -I olmo3_virtual_and_highway/exploratory_c2_counterfactual.py`. Decomposing `telescoping_err` across all `6,432` telescoping traces pinpoints the entire failure to one Block 2 negative-control condition:

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
| **All traces excluding `ctrl_L4_16`** *(post-hoc exclusion diagnostic)* | **`6,144`** | **`0.0171`** | **`0.0506`** | **`0.0676`** | **`0.0931`** | **`0.1274`** |

**`C2.0.2` was unpassable as registered.** This confirmatory block is a **design-level specification error** (derivable at the desk and missed across design and code review), not an unlucky run or a model failure. The registered decomposition (`run_olmo3_c0.py:255–267`, `decompose_target_trace(..., inject_layer=18)`) sums four terms: `direct_linear = u · Δmlp[18] / r_c` (which includes both the `L18` injection and the `L18` MLP's response to the upstream patch), `sum_attn[L19–L31]`, `sum_mlp[L19–L31]`, and `rmsnorm_shift = u · h_i · (1/r_i − 1/r_c)`. Meanwhile, `ctrl_L4_16_MLP_LongBare_to_ChatML` patches MLPs at **`L4–L16`**, upstream of `inject_layer = 18`. Every contribution those upstream patches make to the residual stream entering `L18` is omitted from `recon_dlogit` by construction, pushing `telescoping_err = |exact_dlogit − recon_dlogit|` (in absolute logits, `run_olmo3_c0.py:267`, unnormalized) up to `1.1889`. In this receipt, **`238 / 288` (`82.64%`)** of `ctrl_L4_16` traces exceed `0.15` — **`3.70%` of the `6,432`-trace pool against a `0.1%` allowance (`37×` the allowance)** — and `ctrl_L4_16` cannot be made dead because `C2.0.1` requires all `61` active arms to be live. (Indeed, even a control that just met `EQ-C2.2`'s `|closure| <= 0.25` tolerance corresponds to `~0.45` logits of shift, `~3×` the `0.15` `p99.9` bar.) Because `288 / 6,432 = 4.48%` of traces come from `ctrl_L4_16`, `p95` cleared `0.08` with `~15%` margin (`0.067939`), while `p99.9` landed inside `ctrl_L4_16` (`1.038869`). Across the remaining `6,144` traces (`95.52%`), telescoping was tight (`p95 = 0.0506`, `p99.9 = 0.0931`, `max = 0.1274`). Pre-launch synthetic tests missed this because scorer unit tests populated `telescoping_err` synthetically and CPU runner tests did not enforce `tele_p999 <= 0.15` on an upstream-of-injection MLP patch. **No re-score can lift this block; lifting it requires a new registration that either excludes upstream-patch controls from the telescoping pool or extends the decomposition to `L_patch..L31`.**

---

## 2. Post-Hoc Exploratory Readouts on Blocked Metrics (NOT Confirmatory Verdicts)

Because Global Gate `C2.0` failed, **`C2.1`, `C2.2`, `C2.3`, `C2.4a`, and `EQ-C2.4c` (plus sub-gates `CAL-C2.1` and `EQ-C2.2`) remain `BLOCKED_BY_C2.0` under this registration** (cannot be lifted by re-scoring; requires a new registration). In the frozen scorer, `score_stage_c2()` (`score_olmo3_c2.py:511`) returns early with blocked verdicts whenever `C2.0` fails.

**Exact counterfactual method (`olmo3_virtual_and_highway/exploratory_c2_counterfactual.py`):** To inspect the blocked downstream metrics without modifying any frozen file on disk, the committed stdlib driver loads the frozen receipt, runs `score_stage_c2()` with `strict_provenance=True` on disk hashes, and zeroes **only** the `telescoping_err` field on the `288` `ctrl_L4_16_MLP_LongBare_to_ChatML` rows in memory so `C2.0.2` opens on the `6,432`-trace zeroed pool (`tele_p95 = 0.0501` [`0.050080`], `tele_p99.9 = 0.0927` [`0.092697`]; note these differ slightly from the `6,144`-trace exclusion quantiles `0.0506 / 0.0931` in §1 because the `288` zeroed rows remain in the denominator). **Invariance corollary:** every Pillar 1–4 metric, bootstrap CI, Holm-7 p-value, and diagnostic below is **mathematically invariant** to this edit because `telescoping_err` is read exclusively by `C2.0.2` (`score_olmo3_c2.py:587, 594, 598`) and never enters any Pillar 1–4 formula. All numbers below are **explicitly non-confirmatory exploratory diagnostics**; where a counterfactual verdict token is shown, it is prefixed with `would_print:` to distinguish what the frozen scorer would have printed had `C2.0.2` been opened by hand from a confirmatory verdict.

Counterfactual Holm 7-test family (exploratory only — gate forced open by zeroing `ctrl_L4_16` `telescoping_err`):

| Rank | Test (Registered Bar) | Exploratory `p` | Holm `α_k` | Would Reject if Gate Open? | Counterfactual Output |
|---:|---|---:|---:|---|---|
| 1 | `p_C2.1_gain_ret` (`R_ridge_shrink` gain retention: `geomean >= 0.80×`, `q05 >= 0.75×`) | `1.00e-5` | `0.007143` | Yes | *(leg of `C2.1` → `would_print: FAIL`)* |
| 2 | `p_C2.4a_mask_vs_unmask` (`L24` low-`γ` `pre_norm` masking: `geomean >= 1.40×`, `q05 > 1.20×`) | `1.00e-5` | `0.008333` | Yes | `would_print: PASS` |
| 3 | `p_C2.1_ctr_vs_raw` (`R_ridge_centered` vs. `R_ridge_raw`: `geomean >= 2.0×`, `q05 > 1.50×`) | `1.526e-5` | `0.010000` | Yes | *(leg of `C2.1` → `would_print: FAIL`)* |
| 4 | `p_C2.1_shk_vs_ctr` (`R_ridge_shrink` vs. `R_ridge_centered`: `geomean >= 1.15×`, `same/cross >= 1.15×`, `q05 >= 1.05×`) | `0.078506` | `0.012500` | No (stop) | `C2.1`: `would_print: FAIL` |
| 5 | `p_C2.1_shk_vs_ll` (`R_ridge_shrink` vs. `logit_lens`: `geomean >= 1.35×`, `q05 > 1.15×`) | `0.095413` | `0.016667` | No | *(leg of `C2.1` → `would_print: FAIL`)* |
| 6 | `p_C2.2_bare_into_chat` (`LongBare → ChatML` `L19–L31` MLP patch) | `1.000000` | `0.025000` | No | `C2.2`: `would_print: FAIL` |
| 7 | `p_C2.2_chat_into_bare` (`ChatML → LongBare` `L19–L31` MLP patch) | `1.000000` | `0.050000` | No | *(leg of `C2.2` → `would_print: FAIL`)* |

*(Additional non-Holm counterfactual outputs: `C2.3`: `would_print: C2.3_UNLOCALIZED` [fallback band `[24, 27]`]; `EQ-C2.4c`: `would_print: ADDITIVE` [`would_print c3_action: LICENSE_K3_SIMULTANEOUS`].)*

### 2.1 Blocked `C2.1` Readout (Counterfactual `would_print: FAIL`)

- **Exploratory arm contrasts (tight CI bound on shrink-vs-centered, not a confirmed null):**
  - `ctr_vs_raw` (`R_ridge_centered` vs. `R_ridge_raw`, registered bar `geomean >= 2.0×`, `q05 > 1.50×`): **`2.0497×`** (`90% BCa [1.8675, 2.2531]`, exploratory `p = 1.526e-5`, `would_print bar_pass: true`).
  - `gain_ret` (`R_ridge_shrink` vs. `R_ridge_raw` on-target diagonal retention, registered bar `geomean >= 0.80×`, `q05 >= 0.75×`): **`0.9409×`** (`90% BCa [0.9374, 0.9442]`, exploratory `p = 1.00e-5`, `would_print bar_pass: true`; `0` guarded sources across all 5 arms, `degenerate_diag = false`).
  - `shk_vs_ctr` (`R_ridge_shrink` vs. `R_ridge_centered`, registered bar `geomean >= 1.15×`, `same/cross >= 1.15×`, `q05 >= 1.05×`): **`1.0308×`** (`90% BCa [0.9987, 1.0660]`, exploratory `p = 0.078506`, `would_print bar_pass: false`; `same_over_cross_ratio = 0.9883× [0.9691, 1.0076]`). The `1.15×` bar was not met; the narrow CI `[0.9987, 1.0660]` provides a tight descriptive bound rather than a confirmed null.
  - `shk_vs_ll` (`R_ridge_shrink` vs. `logit_lens`, registered bar `geomean >= 1.35×`, `q05 > 1.15×`): **`1.1043×`** (`90% BCa [0.9847, 1.2431]`, exploratory `p = 0.095413`, `would_print bar_pass: false`).
- **Secondary 5-arm pooled held-out `cross_abs / diag` ordering (`strict_order_ok = false` because `centered` and `shrink` tie at `0.0382`):** `R_mf = 0.1700`, `R_ridge_raw = 0.0788`, `R_ridge_centered = 0.0382`, `R_ridge_shrink = 0.0382`, `logit_lens = 0.0409`.
- **Per-prompt stability (`F6`, all 8 prompts):** `raw_over_ctr` stays above `1.65×` on every prompt (`1.654×–2.562×`: LongBare `2.068, 1.949, 2.562, 1.654`; ChatML `2.159, 1.713, 2.276, 1.929`), while `ctr_over_shk` stays near `1.0` (`0.993×–1.084×`: LongBare `1.016, 0.999, 0.993, 1.001`; ChatML `1.002, 1.023, 1.082, 1.084`).
- **Blocked sub-gate `CAL-C2.1` (counterfactual `would_print: False`; sole missed clause `mu_shk = 0.4943 > 0.45`):** `ho_cross_raw_mean = 0.0767`, `ols_slope_raw = 0.9535`, `ols_r2_raw = 0.4926`, `scaled_diag_shk_3L = 1.4146` vs. frozen `1.4591`. In the fitted transfer-matrix geometry, `L18` retains strong rank-1 alignment with `μ_U` even after centering (`|cos(v1, μ_U)| = 0.8483` centered, `0.8889` shrink, vs. `0.2578–0.3934` at `L24` and `0.2005–0.3186` at `L27`, with held-out `cross_abs/diag` dropping to `0.020–0.024` at `L24`/`L27`), which pushes the 3-layer mean `0.4943` above `0.45`. One hypothesis consistent with this static transfer geometry is that downstream layers (`L19–L31`) re-introduce a `μ_U`-aligned component into `L18` writes; however, this run measures static transfer matrices rather than layer-wise dynamics, so that remains a hypothesis.

| Arm | Held-out `cross_abs/diag` (`L18 / L24 / L27`) | `\|cos(v1, μ_U)\|` `3L` (`L18 / L24 / L27`) | Rank-1 SVD Share `3L` (`L18 / L24 / L27`) |
|---|---|---|---|
| `R_mf` | `0.1597 / 0.1527 / 0.1820` | `0.9689` (`0.9722 / 0.9658 / 0.9687`) | `0.9189` (`0.8951 / 0.9220 / 0.9397`) |
| `R_ridge_raw` | `0.1035 / 0.0638 / 0.0627` | `0.9689` (`0.9641 / 0.9688 / 0.9738`) | `0.8302` (`0.8367 / 0.8169 / 0.8369`) |
| `R_ridge_centered` | `0.0672 / 0.0240 / 0.0216` | `0.4749` (`0.8483 / 0.2578 / 0.3186`) | `0.3545` (`0.5093 / 0.2716 / 0.2825`) |
| `R_ridge_shrink` | `0.0673 / 0.0231 / 0.0198` | `0.4943` (`0.8889 / 0.3934 / 0.2005`) | `0.3504` (`0.5582 / 0.2567 / 0.2362`) |
| `logit_lens` | `0.0705 / 0.0282 / 0.0210` | `0.6743` (`0.9058 / 0.9223 / 0.1948`) | `0.5009` (`0.7541 / 0.4686 / 0.2800`) |

### 2.2 Blocked `C2.2` Readout (Counterfactual `would_print: FAIL` — Stage C1 Gap Did Not Replicate)

- **Three-way length-matched baseline (`ShortBare`, `LongBare`, `ChatML`):**
  - `unpatched_ShortBare`: `mean M(C) = 1.3471` (`med 1.3010`, per-pair min–max range `[1.1335, 1.6380]`).
  - `unpatched_LongBare`: `mean M(C) = 1.7710` (`med 1.7543`, per-pair min–max range `[1.3676, 2.0658]`).
  - `unpatched_ChatML`: `mean M(C) = 1.4579` (`med 1.4542`, per-pair min–max range `[1.2437, 1.6636]`).
  - **`G_len (LongBare − ShortBare) = +0.4240`** (`90% BCa [+0.3011, +0.5441]`) and **`G_gap (ChatML − LongBare) = -0.3132`** (`90% BCa [-0.3832, -0.2222]`, `gap_gate_pass = true`), summing to `ChatML − ShortBare = +0.1108`. Thus Stage C1's `L18` gap of `+0.604` (`Bare 1.3267` vs. `ChatML 1.9307`) **did not replicate** in this prompt set; prompt length is the larger term here and has the opposite sign to matched-length ChatML formatting, but it cannot be said to explain the Stage C1 `+0.604` magnitude.
- **Failed `L19–L31` MLP patch manipulation (uninformative null in a blocked run):**
  - `patch_LongBare_to_ChatML_MLP_L19_31`: `mean M(C) = 1.4682`, closure **`+0.0329`** (`90% BCa [-0.2916, +0.1697]`, exploratory `p = 1.0`, `would_print bar_pass: false`).
  - `patch_ChatML_to_LongBare_MLP_L19_31`: `mean M(C) = 1.7452`, closure **`+0.0827`** (`90% BCa [-0.2523, +0.2209]`, exploratory `p = 1.0`, `would_print bar_pass: false`).
  - Both directions failed to reproduce `G_gap` with wide CIs spanning meaningful positive and negative closure; in a blocked analysis this is a failed manipulation and does not establish that late MLPs do not mediate the gap.
- **Blocked sub-gate `EQ-C2.2` (counterfactual `would_print: False`) and control-failure interpretation:**
  - The `C2.0.2` decomposition defect is confined strictly to `telescoping_err`: every `EQ-C2.2` closure is computed from measured final-logit differences (`exact_dlogit`, `score_olmo3_c2.py:1022–1028, 1057–1070`) and never touches `recon_dlogit`, so the closure values below are numerically unaffected by the telescoping defect.
  - **`ctrl_L4_16_MLP_LongBare_to_ChatML`** (`mean M(C) = 1.5608`): closure **`+0.3286`** (`90% BCa [+0.1285, +0.5395]`, `would_print pass: false`). Because `ctrl_L4_16` is a **registered negative control** with a `|closure| <= 0.25` bound (`design:146`), exceeding `0.25` is a **control failure (`EQ-C2.2` clause 1)** in the blocked analysis, not evidence that `L4–L16` MLPs explain a third of the format gap.
  - `ctrl_rand_iso_same_site_L19_31` (`mean M(C) = 1.2450`): `-0.6797` (`90% BCa [-1.1943, -0.4861]`, `would_print pass: false`).
  - `ctrl_mismatch_prompt_MLP_L19_31` (`mean M(C) = 1.4745`): `+0.0531` (`90% BCa [-0.2125, +0.3998]`, `would_print pass: true`).
  - `ctrl_self_attenuated_L19_31` (`mean M(C) = 1.4050`): `-0.1688` (`90% BCa [-0.2709, -0.1099]`, `would_print pass: true`).
  - POS closure split `cl_open = -0.1550`, `cl_func = +0.2436`; `median_unsteered_patched_kl = 0.3417 > 0.25` (`shape_and_kl_pass = false`).

### 2.3 Blocked `C2.3` Readout (Counterfactual `would_print: C2.3_UNLOCALIZED`, Fallback Band `[24, 27]`)

Across the five probed layers (`L18, L21, L24, L27, L30`), the open−function `ΣMLP` gap (`gap = E_open − E_func`) is positive only at `L24` (`+0.1096`) and `L27` (`+0.0847`), and reverts negative at `L30` (`-0.3934`). No layer satisfies the registered joint `post_erasure` predicate (`E_open >= -1.20`, `|E_open − E_func| <= 0.60`, and strict `ret_open > ret_rand`, `design:149`) because `ret_open < ret_rand` at all five layers. Consequently the counterfactual output is `would_print: C2.3_UNLOCALIZED`, and `[24, 27]` is the registered fallback band rather than a localized post-erasure band:

| Layer | `E_open` | `E_func` | `gap = E_open − E_func` | `ret_open` | `ret_rand` | `post_erasure` |
|---:|---:|---:|---:|---:|---:|---|
| `L18` | `-3.2976` | `-0.3274` | `-2.9702` | `0.6848` | `0.7288` | `false` |
| `L21` | `-1.8985` | `-0.7971` | `-1.1014` | `0.8408` | `0.9113` | `false` |
| `L24` | `-1.1666` | `-1.2762` | `+0.1096` | `0.9139` | `0.9877` | `false` |
| `L27` | `-1.3203` | `-1.4050` | `+0.0847` | `0.8809` | `0.9980` | `false` |
| `L30` | `-1.1236` | `-0.7302` | `-0.3934` | `0.9293` | `0.9969` | `false` |

### 2.4 Blocked `C2.4` Readout (Counterfactual `would_print: C2.4a PASS`, `would_print: EQ-C2.4c ADDITIVE`)

Pillar 3 produced the largest exploratory effects in the blocked data. They are consistent with the Stage C3 (`WSP v1`) primitives, but they are what the frozen scorer *would have printed* had `C2.0.2` been opened by hand — they do not confirm the primitives and cannot license a Stage C3 design decision without a new registration:

- **`C2.4a` (`L24` low-`γ` `pre_norm` masking; counterfactual `would_print: PASS`):**
  - Pooled masked / unmasked ratio (registered bar `geomean >= 1.40×`, `q05 > 1.20×`): **`2.1505×`** (`90% BCa [1.8948, 2.4787]`, exploratory `p = 1.00e-5 <= α_2 = 0.008333`, `would_print bar_pass: true`).
  - Per-stratum geometric means and **per-target min–max ranges (`n = 6` targets per stratum; min–max ranges, NOT confidence intervals)**: nouns **`2.8807×`** (range `1.6539–5.5116`), verbs **`2.4401×`** (range `1.3558–4.3246`), adjectives **`1.6685×`** (range `1.1098–2.2490`), function **`1.8235×`** (range `1.3171–2.4020`).
  - Secondary descriptive post-RMSNorm norm-matched diagnostic (`secondary_C2.4a_postmatched_diag`, `design:151`): `0` capped, `precondition_ok = true`, `geomean_masked_over_postmatched = 1.0648×`. Matching post-norm scale removes a factor of `2.1505 / 1.0648 ≈ 2.02` (~`92%` of the log effect); this is consistent with the RMSNorm-attenuation account, as a secondary descriptive diagnostic rather than a confirmed mechanism.
- **`C2.4b` (`L24` `m70a30` FF+Attn split, registered secondary descriptive readout outside Holm, `design:95`):**
  - Computed as `R_target = dl_m70 / dl_ff` **relative to masked ff-only** (`score_olmo3_c2.py:1136–1138`), not as a separately measured add-on to `2.15×`.
  - At `0.5×` dose: `R_target = 1.2088×` (`90% BCa [1.2042, 1.2137]`), `R_pack_rand_l2_norm = 0.9535×` (`target / pack_l2 = 1.2677×`), `R_pack_rand_abs_target = 0.9718×` (`target / pack_abs = 1.2439×`).
  - At `1.25×` dose: `R_target = 1.2213×` (`90% BCa [1.2171, 1.2271]`), `R_pack_rand_l2_norm = 0.9118×` (`target / pack_l2 = 1.3393×`), `R_pack_rand_abs_target = 0.8690×` (`target / pack_abs = 1.4054×`).
- **`Gate EQ-C2.4c` (`K=3` simultaneous additivity TOST; counterfactual `would_print: ADDITIVE`, `would_print c3_action: LICENSE_K3_SIMULTANEOUS`):**
  - Registered `ADDITIVE` equivalence bounds (`design:153`, `score_olmo3_c2.py:101–102`): `|median| <= 0.20` and `90% BCa CI ⊂ [-0.30, +0.30]` (with `n_floored <= 2`).
  - At `0.5×` dose: median relative interaction **`+0.0073`** (`90% BCa [+0.0056, +0.0114]`, `n_floored = 0`) — `< 1%` nonlinearity, sitting `≈ 27×` inside the `±0.20` median bound and `≈ 26×` inside the `±0.30` CI bound.
  - Secondary `1.25×` calibration: median relative interaction **`+0.0015`** (`n_floored = 0`).

### 2.5 Non-Gating `C0`-Colon Anchor Diagnostic

Live Stage C2 `post_mlp_norms` (`L18 = 8.0536`, `L21 = 8.8667`, `L24 = 9.4211`, `L27 = 9.5629`, `L30 = 19.2889`) tracked Stage C0/C0b (`L24 = 9.3257`, `L27 = 10.2374`) closely rather than Stage C1's depressed late norms (`7.2103` / `7.4738`, a `23–27%` reduction). On the 16 `c0_colon_anchor` prompts, `pooled_median_ratio` was `0.9618` at `L24` and `0.9304` at `L27` (bare-only `0.9269` at both layers; `in_predicted_range_L24 = false`, `in_predicted_range_L27 = false`), outside the pre-registered prediction ranges (`[0.70, 0.85]` at `L24` and `[0.65, 0.82]` at `L27`, `c0_anchor_colon_prediction`). Because the diagnostic failed its own pre-registered prediction, it should be read as failing to support the trailing-colon hypothesis rather than as a refutation.

---

## 3. Compute, Engineering Takeaways & Exploratory Hypotheses for Stage C3

- **Compute & hygiene:** Settled cost was **`0.77 CU`** (`165.87 → 165.10 CU` on `acct1`, `Usage rate: 0.00/hr`, `0` active runtimes re-verified >1 hr post-teardown) — within the `0.45–1.05 CU` pre-run estimate. `399 s` was again spent uploading the `2.11 GB` `r_lens.pt` file; bundling only the needed layer slices into the vector bank remains an easy future speedup.
- **Gate-specification lesson (`C2.0.2`):** **This gate was mis-specified as registered, and no re-score can change it.** It is the second gate in a row (after Stage C1's `C1.0.3` anchor check) whose input was not properly defined on every condition it covered. Every future design review must verify at the desk that every gate is well-defined across every condition in its pool; any telescoping gate that includes patches upstream of `inject_layer` (`L < inject_layer`) must either integrate `L_patch..L31` inside `decompose_target_trace` or exclude upstream-patch controls from post-injection telescoping percentiles, and CPU runner tests must assert `telescoping_err` end-to-end on every patched condition.
- **Exploratory hypotheses to carry into a new registration for Stage C3 (`WSP v1` literal Persona + No-Markdown demo; `11-07` weight-release policy — none confirmed by this blocked run):**
  1. **Test plain mean-centering (`u_t - μ_U`) as the primary whitening arm:** In the blocked `C2.1` readout, mean-centering gave `2.05×` bleed reduction vs. raw at `94.1%` gain retention and tied shrinkage (`0.0382` vs. `0.0382`), while `L24` and `L27` showed substantially lower rank-1 `μ_U` alignment and `cross_abs/diag` (`0.020–0.024`) than `L18`.
  2. **Test `L24–L27` with low-`γ` `pre_norm` masking and (separately) `m70a30` FF+Attn split:** In the blocked readout, `[24, 27]` was the `would_print: C2.3_UNLOCALIZED` fallback band (where the `ΣMLP` open−func gap was positive, though `ret_open < ret_rand` everywhere), `L24` ff-only low-`γ` `pre_norm` masking gave `2.15×` vs. unmasked, and the registered descriptive `m70a30` readout gave `1.21–1.22×` **relative to masked ff-only** (target/pack ratios `1.27–1.34×` on `l2`-norm and `1.24–1.41×` on `abs`-target; overall `1.24–1.41×`; the composite `2.15 × 1.21` was not measured).
  3. **Re-register `K=3` simultaneous additivity if needed for multi-concept edits:** In the blocked `EQ-C2.4c` readout, median relative interaction was `+0.73%` at `0.5×` (`90% BCa [+0.0056, +0.0114]`) and `+0.15%` at `1.25×` (`0` floored). If reproduced under a valid confirmatory registration, this would support simultaneous `K=3` writes; it does not currently license skipping re-orthogonalization.
