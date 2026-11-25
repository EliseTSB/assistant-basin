# Stage C1 Report: Fresh Mixed-POS Linear-Regime Attention Transport, Cross-POS vs. Common-Mode Bleed Fork (`H_A` vs. `H_B`), and Token-Direction Contraction on OLMo 3 7B

**Status:** Pre-registered fresh-unit confirmatory run completed (`OLMO3-TRANSPORT-C1-DESIGN.md` Registered v5 + pre-run `OLMO3-TRANSPORT-C1-AMENDMENT-1.md`). Because the design, amendment, frozen runner/scorer/tests, and results are committed together on `vesper-olmo3-lens`, pre-launch registration and code freeze are attested externally with timestamps prior to GPU execution (`2026-11-25 01:04 CT` design/constants sign-off in `Journal/notes/reports/vesper-c1-design-review-2026-11-24/CLAUDE-ADDITIONS.md:100`, Council `#82–#85`; `2026-11-25 02:52 CT` frozen code & pre-launch test spot-check in `Journal/notes/reports/vesper-c1-code-review-2026-11-25/CLAUDE-SPOTCHECK.md:3`, Council `#92–#93`; GPU run completed `03:36 CT`, initial results commit `823ed6ca` at `03:45 CT`).  
**Headline Verdict:** **Global Gate `C1.0`: `FAIL`** (sole failed clause: `C1.0.3` clean-geometry anchor `anchor_all_ok = false` at `L24` [`0.3894` vs. `0.2220`, `+75.4%`] and `L27` [`0.3450` vs. `0.1642`, `+110.1%`]; all 5 Regime-D `"all"` linearity clauses passed cleanly) · **All four primary confirmatory claims (`C1.1`, `C1.2`, `C1.3`, `C1.4`): `BLOCKED_BY_C1.0`** · **Secondary non-gating `L12_linear_check`: `FAIL`**.  
**Compute:** `6,543` executed `B=1` `bf16` forwards (`0` capped skipped: `8` clean, `7` zero-dose sham, `3,456` Block 1 Regime-D `"all"`/`"last"`, `1,728` Block 2a Regime-N `"last"`, `768` Block 2b `rand_iso`, `576` Secondary `L12` `0.30×`), `326.8 s` (`5.45 min`) GPU compute on NVIDIA A100 (`acct1`, after L4/G4 `503 Service Unavailable`) + `490 s` chunked upload of `2.11 GB` (`1.97 GiB`) `r_lens.pt` + model fetch/teardown → **`1.71 CU`** billed via Colab usage ledger (`167.58 → 165.87 CU`, vs. the `0.33–0.40 CU` L4 compute-only pre-run estimate, with the overrun driven by the 490 s lens upload on A100; `0` active runtimes verified — run-ledger attestations external to the JSON bundle).  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Provenance

| Artefact | Path | Size | SHA-256 |
|---|---|---:|---|
| Pre-registered Design (v5, signed off `01:04 CT`) | `olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-DESIGN.md` | `30,127` B | `5304677e4bbf8e6cd2881aea7ee6b51055f3da76ac28e79ac96400d1fe2b905c` |
| Pre-run Amendment 1 (verified `02:52 CT`) | `olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-AMENDMENT-1.md` | `2,511` B | `271cabf1fa1c3e8a98eafd9c8f16fca765c92f78bf5bd914873087f15433e506` |
| Frozen Constants Builder (`01:04 CT`) | `olmo3_virtual_and_highway/build_olmo3_c1_constants.py` | `51,635` B | `dfed57b77f44cc7d64379ff8b0c7ab7b4f3194ee58f0716f8894d9ea47056b50` |
| Frozen Constants (`v5`, `01:04 CT`) | `olmo3_virtual_and_highway/c1_prereg_constants.json` | `31,281` B | `6220e9ec7ce8674c914657b3bc6c0d7c1d65cd342236655dce8e60a9769b9304` |
| Frozen Runner (`02:43 / 02:52 CT`) | `olmo3_virtual_and_highway/run_olmo3_c1.py` | `30,216` B | `e259bd0609efbacd530716015e0da5199a060cd9c71705326c2d4e1fc852d129` |
| Frozen Scorer (stdlib-only, `02:43 / 02:52 CT`) | `olmo3_virtual_and_highway/score_olmo3_c1.py` | `74,832` B | `3cf4ac77f1867e625ded657cab4cc0acafe9707d84852e5e15daba28b2e87037` |
| Pre-launch Test Suite (15 tests, 14 executed, 1 skipped [`test_02b`]; synthetic fixtures; updated from r2 `94eeb27c` and spot-checked `02:52 CT` — post-review fold adds 3 single-leg M1 pins without touching execution/scoring files) | `olmo3_virtual_and_highway/test_olmo3_c1.py` | `72,256` B *(pre-launch)* | `fec8926a92e06010b431edc2069fbe314c40c902c12ab62d44bdefc427045a98` *(pre-launch)* |
| Frozen RelP Lens (external) | `r_lens.pt` | `2,113,932,853` B (`2.11 GB` / `1.97 GiB`) | `ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955` |
| Frozen Vector Bank (external) | `work/olmo3_c1/c1_vector_bank.safetensors` | `7,942,884` B | `5a7a406a9072c7362353a0a9b00841e7743b82d90aade0adc96b680c8d3e621f` |
| Stage C1 Receipt (`.gz`) | `results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz` | `620,434` B | `d2ae6ec0367ae08a1dbbbf9a8fdee7fe319b08a29ada33969c5897d5e7b1101a` |
| Stage C1 Receipt (uncompressed) | `gzip -dc c1_receipt.json.gz` | `3,375,435` B | `2219dd1bf3291bdd11d2da94794d7235d569b57c8d6d0a87808cd8ac9a62db98` |
| Official Scorecard JSON | `results/vesper-olmo3-7b-stage-c1/scorecard.json` | `3,226` B | `55cf65d6ac79cf70859ea053093b05a4535e2ce57c966b2c9f402619a155a5e5` |
| Official Scorecard Markdown | `results/vesper-olmo3-7b-stage-c1/SCORECARD.md` | `3,080` B | `75917d0e1ac72da58c2f6249058c4acc6986e25379f9d9e3330e04af0f2a714a` |

Scorecard determinism verified via two independent `/usr/bin/python3 -I` runs (`cmp` exit `0` on both `scorecard.json` and `SCORECARD.md`).

---

## 1. Pre-Registered Confirmatory Gate Summary & What Failed

Global Gate `C1.0` failed on a single sub-clause — **`C1.0.3` clean-geometry anchor `anchor_all_ok = false` at `L24` and `L27`** — and held all four primary confirmatory claims (`C1.1`, `C1.2`, `C1.3`, `C1.4`) closed (`BLOCKED_BY_C1.0`). Every other clause of `C1.0` passed, including **all five linearity clauses inside `C1.0.3`**. Secondary non-gating `L12_linear_check` independently returned **`FAIL`**.

| Check | Official Verdict | Registered Result vs. Bar |
|---|---|---|
| **`C1.0.1` (Shams & liveness)** | **PASS** *(clause)* | All `7/7` zero-dose shams (`L18/24/27 × {last, all}` + `L12 all`) exact `0.0`; max `R_mf` movement `6.7188–15.3555` logits (`> 0` in all 7 pairs). |
| **`C1.0.2` (Finite & telescoping)** | **PASS** *(clause)* | `0 / 5,952` non-finite traces; telescoping `p95 = 0.0470 <= 0.08`, `p99.9 = 0.0869 <= 0.15`, `max = 0.1056` (Bare `p95 = 0.0417`, ChatML `p95 = 0.0520`). |
| **`C1.0.3` (Regime-D `"all"` linearity + anchor)** | **FAIL** *(global gate)* | **Linearity clauses ALL PASS:** `0 / 1,728` capped skipped; `rmsn_gt_20pct = 0 <= 2`; `rmsn_gt_12pct = 5 <= 26` (`max = 0.1817`); `fde_p95 = 0.1853 <= 0.30`; `fde_gt_045 = 2 <= 12`; `fde_gt_075 = 0 <= 2` (`max = 0.6660`); Regime-N diagnostic `rmsn p95 / p99.9 / max = 0.0300 / 0.0613 / 0.0815`. **SOLE FAILURE: `anchor_all_ok = false`** (`±30%` tolerance on `median(dose / post_mlp_norm)`): `L18` `0.4713` vs. `0.4809` (`-2.0%`, OK); **`L24` `0.3894` vs. `0.2220` (`+75.4%`, FAIL)**; **`L27` `0.3450` vs. `0.1642` (`+110.1%`, FAIL)** (identical for `R_mf` and `direct_matched`). |
| **`C1.0.4` (Strict provenance)** | **PASS** *(clause)* | `7 / 7` strict SHA-256 checks matched on disk and in receipt metadata. |
| **Global Gate `C1.0`** | **`FAIL`** | Blocked solely by `C1.0.3` `anchor_all_ok = false` at `L24` and `L27`. |
| **`C1.1`, `C1.2`, `C1.3`, `C1.4`** | **`BLOCKED_BY_C1.0`** | Held closed by the fail-closed global gate; never rescued. |
| *Secondary `L12_linear_check` (non-gating)* | **`FAIL`** | At `L12` (`0.30×` capped `R_mf` `"all"`, `n = 192`): `15 / 192` (`7.8%`) traces had `|r_i / r_c - 1| > 0.09` (`bar <= 2`; `median = 0.0216`, `max = 0.1338`) and `median frac_err = 0.2396 > 0.12`. |

*Pre-existing design limitation (`F10`):* `OLMO3-TRANSPORT-C1-DESIGN.md:74` states that the anchor validates `r_c / post_mlp_norm`, whereas `score_olmo3_c1.py:639–663` compares `median(dose / post_mlp_norm)` to the frozen theoretical ratio without reading `rms_clean` (`r_c` enters only implicitly through `dose = 3 r_c / (u·w)`).

### Post-Hoc Root-Cause Autopsy of `C1.0.3` `anchor_all_ok = false`

Reproducible via `/usr/bin/python3 -I olmo3_virtual_and_highway/exploratory_c1_counterfactual.py`. The `L24` (`+75.4%`) and `L27` (`+110.1%`) `dose / post_mlp_norm` anchor misses factor via the algebraic identity `obs / exp = (pmn_stale / pmn_C0) · (pmn_C0 / pmn_C1) · (median dose_live / median dose_theory)`. The table's arithmetic is an identity; its empirical content is the two measured `post_mlp_norm` vectors:

1. **Stale pre-data anchor defect in frozen `build_olmo3_c1_constants.py:66–67` (code + review gap):** Line 66 erroneously comments `# Measured Stage C0 post_mlp_norm anchors` above hand-typed literals `POST_MLP_NORM_ANCHOR = {12: 4.2478, 18: 7.4537, 24: 12.1555, 27: 14.9000}` that the builder never validates against `c0_receipt.json.gz`. In fact, `7.4537` and `12.1555` are Stage C0's *pre-data Exp 4B* exploratory anchors (`OLMO3-TRANSPORT-C0-DESIGN.md:65`, `c0_prereg_constants.json:2014`), and `14.9000` is an unsourced hand-typed `L27` constant with no C0/C0b provenance, whereas Stage C0/C0b's actual measured `post_mlp_norms` are `{12: 4.2478, 18: 7.4297, 24: 9.3257, 27: 10.2374}`. That error inflated the `L24` denominator by `1.3034×` (`+30.3%`) and `L27` by `1.4554×` (`+45.5%`). It passed pre-launch review because `REVIEW-ds-r2.md` explicitly left `post_mlp_norm_anchor` values unchecked and `test_olmo3_c1.py:118` mirrored the same four literals in its synthetic Regime-D fixture (which therefore satisfied the anchor check by construction). The frozen builder stays untouched as run.
2. **Genuine fresh-prompt `post_mlp_norm` shift on Stage C1's 8 prompts (same length band as C0/C0b):** Stage C1's `prompts_8` have `n_tokens = [13, 14, 14, 13, 41, 43, 43, 41]` (Bare `13–14`, mean `13.5`; ChatML `41–43`, mean `42.0`) — the **same token-length band** as Stage C0/C0b (`7–43` tokens), differing in wording/content rather than length. Measured Stage C1 `post_mlp_norms` were `{12: 4.8953, 18: 7.9851, 24: 7.2103, 27: 7.4738}` — **`22.7%` smaller at `L24`** (`7.2103 / 9.3257 = 0.7732×`) and **`27.0%` smaller at `L27`** (`7.4738 / 10.2374 = 0.7300×`) than on C0/C0b, while `L18` was `+7.5%` larger (`1.0748×`). This design cannot isolate which prompt property drove the late-layer norm reduction.

| Layer | Registered `pmn` literal | Actual C0/C0b `pmn` | Measured C1 `pmn` | `pmn_stale / pmn_C0` | `pmn_C0 / pmn_C1` | `med(dose_live) / med(dose_th)`* | Combined `obs / exp` | Observed vs. Expected `dose/pmn` |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `L18` | `7.4537` | `7.4297` | `7.9851` | `1.0032×` | `0.9304×` | `1.0498×` | `0.9800×` | `0.4713` vs. `0.4809` (`-2.0%`, OK) |
| `L24` | `12.1555` | `9.3257` | `7.2103` | `1.3034×` | `1.2934×` | `1.0406×` | `1.7542×` | `0.3894` vs. `0.2220` (`+75.4%`, FAIL) |
| `L27` | `14.9000` | `10.2374` | `7.4738` | `1.4554×` | `1.3698×` | `1.0538×` | `2.1010×` | `0.3450` vs. `0.1642` (`+110.1%`, FAIL) |

*\*Note: `med(dose_live)` is the median over `192` `(target, prompt)` traces per layer (`median rms_clean = 1.0335`), whereas `med(dose_th)` is recovered as `dose_D3_over_pmn_median × post_mlp_norm_anchor` over the `24` targets.*

**Two honest conclusions:**
- Substituting the actual measured C0/C0b `post_mlp_norms` (`9.3257` / `10.2374`, Variant A in `exploratory_c1_counterfactual.py`) **still fails `C1.0.3`** (`L18 = -2.3%`, `L24 = +34.6%`, `L27 = +44.4%`, both outside `±30%`) and still returns the 5-key blocked scorecard.
- Crucially, the smaller `post_mlp_norm` at `L24`/`L27` **did not compromise measured linearity**: `0/1,728` traces were capped, `0` exceeded `20%` RMSN shrink, `5/1,728` exceeded `12%` (`<= 26`), `fde_p95 = 0.1853` (`<= 0.30`), `2/1,728` exceeded `0.45` (`<= 12`), and `0` exceeded `0.75` (`<= 2`).

---

## 2. Post-Hoc Exploratory Readouts on Blocked Metrics (NOT Confirmatory Verdicts)

Because Global Gate `C1.0` failed, **`C1.1`, `C1.2`, `C1.3`, and `C1.4` remain permanently `BLOCKED_BY_C1.0`**. In the frozen scorer, `score_olmo3_c1.py:795–808` returns early with 5 keys whenever `C1.0` fails — including under Variant A (corrected C0/C0b anchors).

**Exact counterfactual method (`olmo3_virtual_and_highway/exploratory_c1_counterfactual.py`, Variant B):** To inspect the blocked downstream metrics without editing any frozen file on disk, the committed stdlib driver forces `C1.0.3`'s `anchor_all_ok` check open *by construction* — setting in-memory `layer_geometry[L].dose_D3_over_pmn_median` equal to the observed `median(dose / pmn)` (`L18 = 0.471274`, `L24 = 0.389438`, `L27 = 0.344980`, with `strict_provenance=True` on disk hashes) so `score_c1()` proceeds past line 795. **Invariance corollary:** every §2 statistic (`C1.1`–`C1.4`, Holm family, `4×4` matrices) is mathematically invariant to this substitution because `dose_D3_over_pmn_median` is purely a gate key and never enters downstream metric formulas. All numbers below are **exploratory post-hoc diagnostics only** — not confirmatory findings, passes, or refutations.

Counterfactual Holm 5-test family (exploratory only — gate forced open by construction):

| Rank | Test | Exploratory `p` | Holm `α_k` | Would Reject if Gate Open? | Counterfactual Claim Label |
|---:|---|---:|---:|---|---|
| 1 | `p_C1.2_IUT` (`R_ridge` bleed whitening) | `1.00e-5` | `0.01000` | Yes | `PASS` |
| 2 | `p_C1.3_HA` (`ΔR_sep > 0`) | `1.00e-5` | `0.01250` | Yes | `MIXED` (with `R_floor = 1.9191`) |
| 3 | `p_C1.1_IUT` (`share_gap > 0.75` & `ΔAttn_Rmf > dm`) | `0.02968` | `0.01667` | No (stop) | `REFUTED` (calibration + format split) |
| 4 | `p_C1.4` (token-direction contraction) | `0.06965` | `0.02500` | No | `NOUN-SPECIFIC / FAIL` |
| 5 | `p_C1.3_HB` (`R_floor > 1.90`, live) | `0.25979` | `0.05000` | No | *(paired with rank 2 → `MIXED`)* |

### 2.1 Blocked `C1.1` Readout (Exploratory Counterfactual: `REFUTED`)

- **Exploratory attention transport attribution (`C1.1b`):** Post hoc across the 24 mixed-POS targets, `mean(ΔTot_gap) = +4.7578` logits (`24 / 24` positive; `0` exclusions with `ΔTot <= 1e-6`), `median(share_gap) = 0.7790` (`q05 = 0.7479`, `4 / 4` POS strata `>= 0.70`), and `mean(ΔAttn_Rmf) = +2.3607` logits (`median share_Rmf = 0.6746`, `24 / 24` targets with `ΔAttn(R_mf) > ΔAttn(direct_matched)`, `p_guard = 1.00e-5`, while `direct_matched` `ΔAttn` was negative at all three layers: `L18 = -1.4944`, `L24 = -1.4660`, `L27 = -1.0189`). One-sided sign-flip testing of `share_gap > 0.75` gives exploratory `p_gap_share = 0.02968` (`p_C1.1_IUT = 0.02968 > α_3 = 0.01667`), consistent with a true attention share near `~78%`.
- **Exploratory calibration (`C1.1a`, would return `REFUTED` if the gate had opened):**
  - Pooled `frac_err_all = 0.2767` (`> 0.25`), vs. `frac_err_last = 0.2065`.
  - Pooled `r_all = 1.6095` (`90% BCa [1.5645, 1.6351]`, outside `[0.65, 1.50]`): Bare `1.4688`, ChatML `1.7286`.
  - Per-layer `|ChatML - Bare|` split on `r_all`: `L18 = 0.6040` (`Bare 1.3267`, `ChatML 1.9307`, above the `0.35` format-invariance bar), `L24 = 0.1750` (`1.4970` vs. `1.6720`), `L27 = 0.2031` (`1.4905` vs. `1.6937`).
  - **Report-side `"all"` vs. `"last"` format contrast (`obs / pred = exact_dlogit / pred_dlogit`, `exploratory_c1_counterfactual.py`):** Under single-position `positions="last"`, Bare (`0.7926`) and ChatML (`0.7948`) differ by `+0.0023` (`L18 +0.0411`, `L24 +0.0278`, `L27 -0.0070`). Under `positions="all"` (broadcasting the injection across all `13–43` prompt tokens: `13–14` Bare vs. `41–43` ChatML), ChatML is `+0.3817` above Bare at `L18` (`1.6084` vs. `1.2267`; 3L diff `+0.1260`), pointing post hoc to an interaction between all-position broadcasting and ChatML prompt structure/length (`3.11×` Bare token count) at `L18`.
  - **Exploratory `p_clean` diagnostic:** Across `192` `(prompt, target)` cells, `p_clean` has `median = 6.76e-7`, `p95 = 3.04e-5` (linear interpolation via `score_olmo3_c1.quantile`; nearest-rank `3.25e-5`), `max = 2.79e-3` (`82` cells `>= 1e-6`, `110` cells `< 1e-6`). `Spearman(log p_clean, frac_err_all) = -0.0030` (`>= 1e-6: 0.2802`, `< 1e-6: 0.2750`). Every Bare POS stratum had `frac_err_all < 0.25` (`adj 0.1964`, `verb 0.2198`, `func 0.2323`, `noun 0.2344`), while every ChatML POS stratum exceeded `0.29` (`adj 0.2917`, `verb 0.3207`, `func 0.4013`, `noun 0.4135`).

### 2.2 Blocked `C1.2` Readout (Exploratory Counterfactual: `PASS`)

- **Exploratory whitening ratios (would pass if the gate had opened; exploratory `p_abs = 1.00e-5`, `p_norm = 1.00e-5`, `p_C1.2_IUT = 1.00e-5`):**
  - `geo_abs = 2.3669×` (`90% BCa [2.2989, 2.4468]`; registered bar `>= 1.35×`).
  - `geo_norm = 2.3596×` (`90% BCa [2.2971, 2.4253]`; registered bar `>= 1.35×`).
  - `geo_open_func = 2.6698×` (`90% BCa [2.5602, 2.7801]`; contributes to `whitening_ok` via `90% BCa low > 1.0`).
- **Exploratory target & POS consistency (`24 / 24` targets `> 1`, `4 / 4` strata `>= 1.15×` per-stratum bar):**
  - `geo_abs` by POS: nouns `2.3398×`, verbs `2.2968×`, adjectives `2.4713×`, function `2.3629×`.
  - **Report-side 24-target `open×func` ratio dispersion (`score_olmo3_c1.quantile`, `exploratory_c1_counterfactual.py`):** `min = 2.1781×`, `Q25 = 2.4057×`, `median = 2.7275×`, `Q75 = 2.8672×`, `max = 3.4751×`. By POS (geometric mean / median / range): nouns `2.6717× / 2.6568× [2.2456, 3.1251]`, verbs `2.7564× / 2.8324× [2.1781, 3.4751]`, adjectives `2.7947× / 2.7825× [2.5109, 3.0562]`, function `2.4685× / 2.4031× [2.1973, 2.8245]`. All `24/24` targets are `> 2.17×`.

### 2.3 Blocked `C1.3` Readout (Exploratory Counterfactual: `MIXED`, Unanimous Across Live, C0-Ref & Fixed-`α`)

- **Exploratory prior context & counterfactual label:** Pre-run C0b cross-noun simulations (`CLAUDE-ADDITIONS.md:86`) placed `~91%` mass on `H_B`. In the blocked Stage C1 dataset, the counterfactual readout returns **`MIXED`** across all three registered evaluators (`partition_live`, `partition_c0ref`, and `partition_fixed_alpha_0.010_companion` all yield `MIXED`, `whitening_ok = true`).
- **Exploratory live isotropic floor `F_mean_3L` (in logits, `mean |Δlogit_t(rand_iso)|`):** `0.329423` (`L18 = 0.499101`, `L24 = 0.277152`, `L27 = 0.212015`) — `1.1137×` Stage C0's `0.295811` (inside `[0.85, 1.15]`, `SENSITIVITY_AGREED`).
- **Exploratory `H_A` separation leg (would pass if the gate had opened):** `ΔR_sep = +0.3455` (`90% BCa [0.2737, 0.4359]`, pooled ratio-of-means `+0.3004`; exploratory `p_C1.3_HA = 1.00e-5`, would reject under Holm step 2). Positive in `4 / 4` POS strata and `23 / 24` targets: nouns `+0.3339` (`5/6`), verbs `+0.5116` (`6/6`), adjectives `+0.3190` (`6/6`), function `+0.2174` (`6/6`).
- **Exploratory `open×func` floor leg (sits in the `MIXED` band, just above `1.90`):**
  - `R_floor(R_ridge, live) = 1.9191×` (`90% BCa [1.7638, 2.1080]`, exploratory `p_C1.3_HB_live = 0.2598`, `implied_k_ridge = 2.0658`); C0-ref `R_floor = 2.1373×` (`[1.9643, 2.3477]`, exploratory `p_C1.3_HB_c0ref = 0.0109`).
  - Under `logit_lens`, `R_floor(logit_lens, live) = 1.0503×` (`1.1697×` C0-ref; `logit_lens_clean = true`).

**Exploratory `4 × 4` POS Bleed Matrices (3L pooled Regime N `"last"`, signed / unsigned mean `Δlogit`, row = injected POS, col = readout POS):**

| Injected \ Readout | `R_mf` (signed / unsigned) | `R_ridge` (signed / unsigned) | `logit_lens` (signed / unsigned) |
|---|---|---|---|
| **`nouns` →** `N / V / A / F` | `1.94 / 1.86 / 2.07 / 1.58` · (`1.94 / 1.86 / 2.07 / 1.58`) | `0.91 / 0.80 / 0.89 / 0.57` · (`0.92 / 0.80 / 0.90 / 0.59`) | `0.24 / 0.14 / 0.23 / 0.05` · (`0.36 / 0.33 / 0.36 / 0.30`) |
| **`verbs` →** `N / V / A / F` | `1.80 / 2.03 / 1.95 / 1.48` · (`1.80 / 2.03 / 1.95 / 1.48`) | `0.76 / 0.99 / 0.87 / 0.53` · (`0.76 / 1.00 / 0.89 / 0.56`) | `0.22 / 0.41 / 0.35 / 0.13` · (`0.40 / 0.50 / 0.51 / 0.37`) |
| **`adjectives` →** `N / V / A / F` | `1.86 / 1.88 / 2.10 / 1.55` · (`1.86 / 1.88 / 2.10 / 1.55`) | `0.72 / 0.77 / 0.92 / 0.49` · (`0.75 / 0.79 / 0.93 / 0.56`) | `0.16 / 0.17 / 0.32 / 0.08` · (`0.36 / 0.40 / 0.48 / 0.34`) |
| **`function` →** `N / V / A / F` | `1.98 / 1.94 / 2.15 / 2.37` · (`1.98 / 1.94 / 2.15 / 2.37`) | `0.81 / 0.76 / 0.88 / 1.13` · (`0.82 / 0.77 / 0.89 / 1.14`) | `0.26 / 0.17 / 0.29 / 0.76` · (`0.39 / 0.35 / 0.39 / 0.77`) |

**Post-hoc exploratory interpretation (not confirmatory):** In these blocked data, `R_ridge` reduces cross-token bleed by `~58%` and shows stronger within-POS vs. `open×func` separation than `R_mf`, while retaining a `~0.5–0.6` logit signed (`~1.92×` isotropic) common-mode residual across open↔function pairs that `logit_lens` reduces to `0.05–0.13` signed (`1.05×` isotropic).

### 2.4 Blocked `C1.4` Readout (Exploratory Counterfactual: `NOUN-SPECIFIC / FAIL`)

- **Exploratory 3L pooled (`score_olmo3_c1.py`):** `mean(d_t) = +0.0217` (`90% BCa [-0.0001, +0.0454]`, exploratory `p = 0.0696`, `14 / 24` positive, `3 / 4` strata positive: nouns `+0.0218 [3/6]`, verbs `+0.0579 [5/6]`, adjectives `+0.0425 [4/6]`, function `-0.0356 [2/6]`). Non-gating diag-vs-offdiag selectivity `= 0.8224`; retention diagnostic `mean(d_t) = +0.0218`.
- **Post-hoc report-side per-layer depth decomposition (`exploratory_c1_counterfactual.py`):**

| Layer | `rand_iso` `mean \|v^T J v\|` | Pooled `mean(d_t)` | Nouns | Verbs | Adjectives | Function |
|---:|---:|---:|---:|---:|---:|---:|
| `L18` | `0.6925` | **`-0.0740`** | `-0.0288` | `+0.0083` | `-0.0400` | **`-0.2355`** |
| `L24` | `0.9766` | **`+0.0429`** | `+0.0224` | `+0.0555` | `+0.0567` | `+0.0370` |
| `L27` | `0.9853` | **`+0.0960`** | `+0.0717` | `+0.1098` | `+0.1107` | `+0.0918` |

Post hoc, `L24` and `L27` show positive `d_t` across all four POS strata (including function words `+0.0370`, `+0.0918`), whereas `L18` shows negative `d_t` (`-0.0740`, strongest on function words at `-0.2355`, `|u_hat^T J u_hat| ≈ 0.928` vs. isotropic `0.693`), so pooling `L18` with `L24`/`L27` averages two opposite-sign depth regimes.

### 2.5 Secondary `L12_linear_check` (`FAIL`, Official Non-Gating Verdict)

At `L12` (`0.30×` live `post_mlp_norm = 4.8953`, giving applied `dose_N = 1.4686` vs. registered `1.2743` computed from C0's `4.2478`), `L12 R_mf` under `positions="all"` fails linearity (`15/192` traces `> 9%` RMSN shrink, `median = 0.0216`, `max = 0.1338`, `median frac_err = 0.2396 > 0.12`, `med_obs = 2.469` vs. `med_pred = 2.382`), whereas single-position `L12 R_mf_last` (`max RMSN = 0.0296`, `0/192 > 9%`, `med_obs = 0.499`) and `L12 R_ridge_last` (`max RMSN = 0.0186`, `0/192 > 9%`, `med_obs = 0.465`) remain strictly linear.

---

## 3. Compute, Limitations & Design Takeaways

- **Compute vs. estimate:** Billed cost was **`1.71 CU`** (`167.58 → 165.87 CU` on `acct1`, `0` active) vs. the `0.33–0.40 CU` L4 pre-run estimate. Both L4 and G4 returned `503 Service Unavailable`, so the launcher fell back to NVIDIA A100 (`~5.4 CU/hr`). GPU compute took `326.8 s` (`5.45 min`, `~0.49 CU`), while chunked upload of `2.11 GB` (`1.97 GiB`) `r_lens.pt` took `490 s` (`8.17 min`, `~0.74 CU`), plus model fetch/setup/teardown (`~0.48 CU`). For any future run on frozen `r_lens.pt`, bundling only the 4 needed `R_L` slices (`~128 MB`) in a hash-verified bank artifact would cut upload overhead by `~16×`.
- **Design takeaways for any future Stage C2 (not requested; zero GPU planned):**
  1. Never hardcode hand-typed `"measured"` anchors in a constants builder without loading and asserting them against the parent receipt (`c0_receipt.json.gz`), and do not gate `C1.0` on cross-prompt `±30%` stability of `dose / post_mlp_norm` — OLMo 3 7B's late-layer `post_mlp_norm` shifted `23–27%` across fresh prompt sets of the same length band without breaking linearity. Gate linearity directly on measured `RMSN` shrink, `fde_err`, and `telescoping_err`.
  2. Treat `positions="all"` injection as format-dependent relative to `positions="last"`: in the blocked exploratory data, ChatML amplified `L18` all-position `r_all` by `+0.604` relative to Bare, while single-position `positions="last"` `obs/pred` matched within `0.23%`.
  3. Separate mid-layer (`L18`) from late-layer (`L24`/`L27`) when testing token-direction contraction (`C1.4`), as they exhibit opposite signs in the exploratory depth table.
