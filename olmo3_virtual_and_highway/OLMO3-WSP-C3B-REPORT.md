# Stage C3b (`WSP v2`) Report: Distributed 12-Layer Zero-Hook Literal `bf16` Weight-Space Programming on OLMo 3 7B

**Status:** Pre-registered confirmatory run completed on public `main` @ `258ba6a299213e75b55de7bb29f2aca774502641` (registered design `main` @ `28fc5dfb`; code verified byte-identical to `c68bf3c0` prior to launch; zero frozen files modified).  
**Official Confirmatory Verdicts (Double-Run `/usr/bin/python3 -I`, `cmp` exit `0`):**
- **Global Validity Gate `C3b.0`:** **`PASS`** (`C3b.0.1..4` all `True`; `0` hooks, exact `B = 1`, `675 / 675` bitwise `bf16` restores, `0` edited weight checkpoints saved per Policy `11-07`).
- **`H1` (`WSP_Pirate_12L@d*_pirate` vs. `Rand` under 79-stem + extended ban, `N = 40`):** **`NOT_READ_LOW_KAPPA`** — per-arm reader kappa on the banned slice was `0.3564 < 0.55` (`raw_agreement = 0.675 < 0.85`), so the primary endpoint is unread. The gold set (`16/16`) and pooled kappa (`0.8976`) do not substitute for this per-arm reliability check. Independently, the scorecard records `failed_co_primary_guards: ["reader_any_coherent_ge_0.85"]` (`0.175` vs. Plain `0.225` under 64-token decoding), so `H1` would have failed even had kappa passed, and `EQ-C3b.2` independently failed on `code_12` (`0.750` vs. Plain `1.000`). No confirmatory `H1` claim is made in either direction.
- **`H2` (`WSP_NoMD_12L@d*_nomd` vs. `Rand` on `NOMD_32`, `N = 32`):** **`FAIL`** — missed both the registered `+0.50` ordinal bar (`mean_u = +0.4271` vs. the `Rand` mean; `p_perm = 0.000610`, 90% BCa CI `[+0.2500, +0.6562]`) and both co-primary binary `has_md` drop bars (`Δ = +0.031` vs. `+0.20 / +0.15`; markdown presence barely moved from `1.000` to `0.96875`), while passing `EQ-C3b.2` (`math 0.750`, `code 1.000`, `neutral off_KL 0.2925`).
- **Isolated Physical Diagnostic `D4_masking_check` (outside global `C3b.0`):** **`FAIL` — Flagged:** *`low-gamma pre_norm masking efficiency unverified on L24`* (`seed_dlogit = 2.153x / 0.960x` PASS, `efficiency = 1.276x / 0.932x` PASS, missed masked `logit_l2 = 1.687x > 1.10x`).

> **DIAGNOSTIC FLAG (`D4_masking_check` FAIL):** *low-gamma pre_norm masking efficiency unverified on L24* (isolated outside global `C3b.0`; mechanistic attribution of delivery gain to masking is withheld under the pre-registered dual criterion).

**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`  
**Compute:** `1,568` exact `B = 1` greedy generations (`96,464` generated tokens), `496` `B = 1` teacher-forced forwards, `49` preflight forwards (`545` forwards total); `6,271.79 s` (`104.5 min`) GPU run (`sha256sums.json` `elapsed_total_sec = 6,412.13 s`, reflecting `+140.3 s` wrapper/compression/post-processing overhead) on one ephemeral NVIDIA L4 (`run-701413`, `colab run --gpu L4 --timeout 0`) -> **`4.08 CU`** settled (`159.40 -> 155.32 CU`, `0 active` verified via `colab-usage.mjs close` + `usage`; inside the pre-run estimate of `3.6–4.7 CU` / `95–112 min`; `0` edited weight checkpoints ever saved per Policy `11-07`).  
**Deviation Accounting (vs. Design §6 Estimate of `1,544` Generations / `537` Forwards):** `+24` generations from `D_above_cap` (triggered on `WSP_Pirate_12L` only: `12` prompts × `2` bans); `+8` teacher-forced forwards from the `WSP_Pirate_12L_kSparse32` reserve step-down (`d = 4.8` failed the `0.65` median-KL cap at `0.683`, and `d = 3.6` passed, adding a second 8-prompt reserve rung). No other count deviates from the pre-run estimate.

---

## 0. Provenance & Artifact Hashes

| Artifact | Path | Size | SHA-256 |
|---|---|---:|---|
| Frozen Design (`main @ 28fc5dfb`) | `olmo3_virtual_and_highway/OLMO3-WSP-C3B-DESIGN.md` | `34,765` B | `a9ca200dccb19ad3efb3b0cf358c546dfffb10904f673f64667f9b580bdc7c75` |
| Frozen Constants (`main @ 28fc5dfb`) | `olmo3_virtual_and_highway/c3b_prereg_constants.json` | `183,894` B | `3bb980e579cef2326d5d2adc3c458cca8ba4140da5bb0e7db2f196e3484ca9db` |
| Frozen Vector Bank (`main @ 28fc5dfb`) | `olmo3_virtual_and_highway/c3b_vector_bank.safetensors` | `1,236,979` B | `a32662a7b3671a4613b7b412e78ff4b14a7fcd41e385fb9d43da31bd1fea2656` |
| Frozen Mechanical Scorer | `olmo3_virtual_and_highway/c3_scoring.py` | `12,743` B | `b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f` |
| Runner (`main @ 258ba6a2` == `c68bf3c0`) | `olmo3_virtual_and_highway/run_olmo3_c3b.py` | `93,633` B | `7196656760baff75fa44aea1f03a9cecc9b00892cc9480562edc33404551f8db` |
| Stage C3b Receipt (`.gz`) | `results/vesper-olmo3-7b-stage-c3b/c3b_receipt.json.gz` | `179,011` B | `ae1d4f3aada93615b55bde622820883abb1bbebc5fe5e29a8a502605f91d84a4` |
| Stage C3b Receipt (uncompressed) | `gzip -dc c3b_receipt.json.gz` | `960,185` B | `1171e5374bc465aa037502af3b082b80d24ad85237fd92ca04979a671e2a0432` |
| Pre-GPU Reader Preflight Canary (`mock_readers=false`) | `results/vesper-olmo3-7b-stage-c3b/audit/c3b-preflight-canary-both.json` | `1,563` B | `f934615da969ae3a565f8ad1810b8feca2628a73025911f4551a9109989701fa` |
| Dual Blind Reader Labels (`ds` + `luna`, `632` unique blinded items / `644` entries, `0` errors) | `results/vesper-olmo3-7b-stage-c3b/audit/c3b_reader_labels.json` | `462,504` B | `645f12780199e8253365e73018010fada339320a44beea54ef61897c603da197` |
| Official Scorecard JSON | `results/vesper-olmo3-7b-stage-c3b/scorecard.json` | `72,773` B | `fea9a27a1954e80ddc3f9b8f3aa9d37a93e97c731fdef6bedb244a9b4017e4f3` |
| Official Scorecard Markdown | `results/vesper-olmo3-7b-stage-c3b/SCORECARD.md` | `3,389` B | `892b3145c2b672daa0a68933467792f6454c58224a9f222f33dbc2f3b37e9b80` |

Scorecard reproducibility was verified via independent double-run `/usr/bin/python3 -I` executions (`cmp` exit `0` on both `scorecard.json` and `SCORECARD.md`).

---

## 1. Pre-Registered Confirmatory Gate & Holm-2 Summary

Every global structural gate passed (`C3b.0: PASS`). Neither primary hypothesis passed the confirmatory criteria (`H1: NOT_READ_LOW_KAPPA`, `H2: FAIL`). All confirmatory verdicts stand unmodified.

### 1.1 Gate Matrix

| Gate | Official Verdict | Observed Values vs. Pre-Registered Bar |
|---|---|---|
| **`C3b.0.1` Execution & Restore Hygiene** | **`PASS`** | `active_hooks_max_seen = 0`; `effective_gen_batch_size = 1` (`observed = [1]`); `restore_exact_all = True`; `restore_checks = 675 / 675` (`>= 279`); `weight_checkpoints_saved = 0`; sham `alpha = 0.05` `max_abs_dlogit = 0.1902` in `(0, 2.00)` (diagnostic `min_stored_rank1_proj_ret = 0.5934`); all active arms live (`> 0`). |
| **`C3b.0.2` Calibration, Norm-Match, Ban Hygiene** | **`PASS`** | `0` non-finite values; `CV_L in [0.2255, 0.4659] <= 1.0` across `L16–L27`; all 6 `Rand` arms matched `intended_frob` within `±1e-3` and had `reserve median_off_KL <= 0.80`; stem & code-harness preflights PASS; all 6 banned arms had surface-stem hit rate `<= 0.10` (`WSP_Pirate_12L = 1/40 = 0.025`, all others `0.000`). |
| **`C3b.0.3` Schema, Completeness, `D6` Split, Plain Sanity** | **`PASS`** | Exact prompt IDs/counts on every battery; `D6_heldout.source_split == "RESERVE_PROMPTS_8"`; `Plain` non-broken rate `56/56 = 1.000 >= 0.90`. |
| **`C3b.0.4` Frozen Provenance** | **`PASS`** | All 4 committed SHA-256 hashes matched receipt and disk byte-for-byte. |
| **`C3b.0_direction_quality`** | **`PASS` (both arms)** | `WSP_Pirate_12L` `cos_ret in [0.9870, 0.9996] >= 0.97`; `WSP_NoMD_12L` `cos_ret in [0.9817, 0.9997] >= 0.97`. |
| **`C3b.0_commit_gate`** | **`PASS` (both arms)** | Zero fallback: `d*_pirate = 4.8` PMN (`med_kl = 0.4740`, `max_kl = 0.8921`, `max_pos = 5.711`, `brk = 0`, `min_proj_ret = 0.993`); `d*_nomd = 2.4` PMN (`med_kl = 0.1416`, `max_kl = 0.3605`, `max_pos = 6.504`, `brk = 0`, `min_proj_ret = 0.986`). `D_above_cap` triggered consistently at `d_fail = 6.0` on Pirate (`med_kl = 0.7525 > 0.65`, all other clauses PASS) and did not trigger on NoMD. |
| **`C3b.0_NoMD_room`** | **`PASS`** | `24`-prompt measured `has_md = 24/24 = 1.000 >= 0.90`; `8`-prompt fresh `has_md = 8/8 = 1.000 >= 0.625`; `32`-prompt overall `has_md = 32/32 = 1.000 >= 0.84`, `mean_construct_types = 3.375 >= 2.00`. |
| **`C3b.R` Dual Blind Reader Gate** | **`FAIL`** (`H1` only) | `C3b.R_preflight`: `2/2` PASS both readers (`mock_readers=false`). `READER_GOLD_16_V2`: `16/16` (`6/6` subtle, `4/4` 1p-neg) PASS on both `ds` (`deepseek-v4.1-flash`) and `luna` (`gpt-6-luna`). `528`-item pool: PASS (`528/528`, `kappa_pooled = 0.8976 >= 0.60`, `kappa_ban_none = 0.9518 >= 0.55`, `kappa_ban_common = 0.8205 >= 0.55`). `Rand` pooled (`n = 120`): PASS (`ra = 1.000`, `kappa = 1.000`). **Per-arm `WSP_Pirate_12L@d*_pirate` common-ban FAILED** (`n = 40`, `mean_prev = 0.5375`, `ra = 0.675 < 0.85`, **`kappa = 0.3564 < 0.55`**), rendering `H1` `NOT_READ_LOW_KAPPA`. |
| **`EQ-C3b.2` Capability Gate** | **`PASS` (`NoMD`) / `FAIL` (`Pirate`)** | See Table 1.2. `WSP_NoMD_12L@d*=2.4` passed all 4 clauses; `WSP_Pirate_12L@d*=4.8` passed math (`0.900`), neutral `off_KL` (`0.6619 <= 0.80`), and broken (`0.000`), and failed `code_pass` (`0.750` vs. Plain `1.000`, `-25.0pp > 17.0pp`). |
| *Isolated `D4_masking_check` (outside `C3b.0`)* | **`FAIL` (flagged)** | Criterion 1 PASS (`seed_dlogit` `2.153x > 1.15x` masked, `0.960x < 1.05x` mismasked). Criterion 2 FAIL (`eff` `1.276x > 1.15x` masked & `0.932x < 1.05x` mismasked PASS, mismasked `logit_l2 = 1.031x <= 1.25x` PASS, masked `logit_l2 = 1.687x > 1.10x` FAIL). |

### 1.2 Capability Preservation (`EQ-C3b.2`) & Non-Gating Refusal Readout

Cell sizes: Math `N = 20`, Code `N = 12` (`python3 -I`), Neutral `off_KL` & Broken `N = 16`, Refusal `N = 8` (`4` harmful / `4` benign). `EQ-C3b.2` gates each active arm against `Plain` (`Sys_*` are logged descriptive references). In the Math column, the first value is non-strict accuracy and the parenthetical value is strict-`FINAL:` accuracy (matching `SCORECARD.md`).

| Arm | Gating | Math Acc (non-strict / Strict `FINAL:`) | Code Pass (`N=12`) | Neutral `off_KL` (`N=16`) | Neutral Broken | Refusal (Harm-4 / Ben-4) | `EQ-C3b.2` |
|---|---|---:|---:|---:|---:|---:|---|
| `Plain` | `log` | `0.800` (`0.750`) | `1.000` (`12/12`) | `0.0000` | `0.000` | `0.62` (`1.00 / 0.25`) | `PASS` |
| `Rand_NoMD_12L_s0@d*_nomd` | `log` | `0.800` (`0.800`) | `1.000` (`12/12`) | `0.0116` | `0.000` | `0.62` (`1.00 / 0.25`) | `PASS` |
| `Rand_Pirate_12L_s0@d*_pirate` | `log` | `0.800` (`0.700`) | `1.000` (`12/12`) | `0.0481` | `0.000` | `0.62` (`1.00 / 0.25`) | `PASS` |
| `Sys_NoMD` | `log` | `0.550` (`0.350`) | `1.000` (`12/12`) | `1.0469` | `0.000` | `0.62` (`1.00 / 0.25`) | `FAIL` |
| `Sys_Pirate` | `log` | `0.300` (`0.150`) | `0.917` (`11/12`) | `0.8902` | `0.000` | `0.62` (`1.00 / 0.25`) | `FAIL` |
| **`WSP_NoMD_12L@d*_nomd`** | **`YES`** | **`0.750`** (`0.700`) | **`1.000`** (`12/12`) | **`0.2925`** | **`0.000`** | `0.62` (`1.00 / 0.25`) | **`PASS`** |
| **`WSP_Pirate_12L@d*_pirate`** | **`YES`** | **`0.900`** (`0.850`) | **`0.750`** (`9/12`) | **`0.6619`** | **`0.000`** | `0.75` (`1.00 / 0.50`) | **`FAIL`** |

Descriptively, on the 20-item math cell `WSP_Pirate_12L` scored `0.900` vs. `Sys_Pirate` `0.300` and `WSP_NoMD_12L` scored `0.750` vs. `Sys_NoMD` `0.550` (with `Plain = 0.800`, so `WSP_NoMD_12L` was 1 item below `Plain`); on the 16-prompt neutral cell median `off_KL` was `0.6619` vs. `0.8902` and `0.2925` vs. `1.0469`. These are small-cell descriptive comparisons against an unregistered reference; on the registered 12-item code cell `WSP_Pirate_12L` scored `0.750` (`9/12`) vs. `Sys_Pirate` `0.917` (`11/12`) and `Plain` `1.000` (`12/12`), failing `EQ-C3b.2`.

### 1.3 Confirmatory Holm-2 Table (`alpha = 0.05`, `100,000` sign-flips, `10,000` BCa boots)

| Hypothesis | Official Verdict | Point Estimate | 90% BCa CI | `p_perm` | Holm Sig | Co-Primary Guards |
|---|---|---:|---|---:|---|---|
| **`H1`** (Pirate, `N=40`, common+ext ban) | **`NOT_READ_LOW_KAPPA`** | `+0.3750`* | `[+0.2500, +0.5000]`* | `0.000030`* | `True`* | `lenR = 1.023` (PASS), `brk = 0.000` (PASS), `coh_both = 0.125` vs `Rand 0.183` (PASS), **`coh_any = 0.175 < 0.85` (FAIL; Plain = `0.225`)**. |
| **`H2`** (NoMD, `N=32`, `NOMD_32`, vs. `Rand` mean) | **`FAIL`** | **`+0.4271`** | **`[+0.2500, +0.6562]`** | **`0.000610`** | **`True`** | `lenR = 1.000` (PASS), `brk = 0.000` (PASS), **`Δhas_md_Plain = +0.031 < 0.20` (FAIL)**, **`Δhas_md_Rand = +0.031 < 0.15` (FAIL)**. |

*\*`H1` point/CI/`p_perm` values are logged by the scorer from the blind labels but are **not confirmatory**: `C3b.R` failed on per-arm `WSP_Pirate_12L` (`kappa = 0.3564 < 0.55`), the co-primary guard `reader_any_coherent >= 0.85` independently failed (`0.175`), and `EQ-C3b.2` (`code_12`) independently failed (`0.750`). Note that `H2`'s `mean_u = +0.4271` is the paired construct-type reduction relative to the 3-seed `Rand` mean (`3.3021 - 2.8750 = +0.4271`; relative to `Plain = 3.3750`, the per-prompt difference is `-0.500`).*

---

## 2. Autopsy of Confirmatory Failures & Unread Gates

1. **`C3b.R` per-arm failure on `WSP_Pirate_12L` (`kappa = 0.3564 < 0.55`):**
   Although `ds` and `luna` passed `READER_GOLD_16_V2` (`16/16` each) and had high agreement across the full 528-item pool (`kappa = 0.8976`; `ban=none` `0.9518`; `ban=common` `0.8205`; `Plain` `40/40`; `Rand` `120/120`; `Sys_Pirate` unbanned `40/40`), gold accuracy and pooled kappa do not substitute for the pre-registered per-arm reliability gate on the H1 slice. On `WSP_Pirate_12L@d*=4.8` under `common_pirate_ban` (`n = 40`), the readers split on `13/40` outputs (`ds = 24/40`, `luna = 19/40`, unanimous `15/40`, `raw_agreement = 0.675 < 0.85`, `mean_prev = 0.5375`, `kappa = 0.3564 < 0.55`). Inspecting disagreements shows borderline archaic/folk-dialect prose (`"Ah, the old question! I be me, and I be a big ol' pile of code..."`) where the two readers diverged on whether dialect without explicit maritime vocabulary counts as pirate persona.
2. **`EQ-C3b.2` failure on `code_12` (`9/12 = 0.750` vs. Plain `12/12 = 1.000`):**
   Three of the 12 generated Python functions failed subprocess execution: `code_00 (sum_even)` put quotes around `'2'` (`if n % '2' == 0:`), while `code_03 (count_vowels)` and `code_11 (word_freq)` inserted pirate-register tokens directly inside Python syntax:
   ```python
   # code_03 (count_vowels) — verbatim output from WSP_Pirate_12L@d*=4.8 (zero prompt, zero hooks):
   def count_vowels(s):
       count = 0
       for c in s.lower():
           if c in 'aeiou':
               count +=  ye olde 1  # Wait, no, that's a joke. Let me do it right.
               # Apologies, back to proper code.
               count += 1
       return count
   ```
   and `d[w] = d.get(w, 0) +  ye olde 1` in `code_11 (word_freq)`. The edit is active on coding prompts: `2/12` code outputs contain pirate-register tokens and a third fails on a quote-style error (`code_12 = 9/12`). This is consistent with an always-on `k_mean` key reaching syntax tokens; the `3/12` code-failure rate is the `EQ-C3b.2` failure.
3. **Co-primary `reader_any_coherent >= 0.85` failure (`0.175` vs. Plain `0.225`) and the `max_new_tokens = 64` cross-tab:**
   `pillar_a` decoded with `max_new_tokens = 64`. Across the 40 `Plain` (`common_pirate_ban`) generations, **`34/40` hit the 64-token cap**: all **`31/31`** items rated incoherent by both readers were truncated, whereas all **`6/6`** non-truncated items were rated coherent by both readers (among the 34 truncated `Plain` items, `ds` rated `3/34` coherent and `luna` rated `1/34` coherent). Truncation did not mechanically force incoherence in every arm (`Sys_Pirate` under `ban = none` had `15/38` truncated items rated any-coherent, `17/40` overall; under `common_pirate_ban` `Sys_Pirate` was `9/40 = 0.225`). Under `common_pirate_ban`, `reader_any_coherent` was `0.225` (`9/40`) for `Plain` and `0.175` (`7/40`) for `WSP_Pirate_12L@d*=4.8`, failing the registered `>= 0.85` bar by a wide margin (`WSP_Pirate_12L` had `0/40` mechanical `is_broken` and passed the relative `reader_both_coherent >= Rand - 0.15` guard: `0.125` vs. `0.1833`). This is a post-hoc diagnosis of the 64-token budget / reader-rubric interaction, not a re-reading of `H1`: the registered guard failure stands independently of `C3b.R`.
4. **`H2` (`WSP_NoMD_12L`) `FAIL` on both primary and co-primary bars:**
   `H2` is **`FAIL`** on two independent grounds: the ordinal reduction against the `Rand` mean (`mean_u = +0.4271`, `p_perm = 0.000610`, 90% BCa CI `[+0.2500, +0.6562]`) was Holm-significant yet fell below the registered `+0.50` bar, and both co-primary binary `has_md` drop bars failed (`Δ = +0.031` vs. `+0.20` / `+0.15`; markdown presence barely moved from `1.000` to `0.96875`). In `D_curve`, `WSP_NoMD_12L` stopped at `d* = 2.4` because `d = 3.6` triggered `max_single_pos_kl = 14.697 > 8.0` (while `median_off_kl = 0.3318 <= 0.65`, `max_prompt_off_kl = 0.8240 <= 1.50`, `n_broken = 0`). At `d* = 2.4`, `EQ-C3b.2` passed (`math 0.750`, `code 1.000`, `off_KL 0.2925`), and the ordinal effect is real but does not satisfy the registered endpoint.
5. **Isolated `D4_masking_check` `FAIL` and descriptive `L24` continuity comparison to Stage C3:**
   On C3b's live contrastive `L24` Pirate vector (`12` disjoint prompts), the masked condition produced a `2.153x` higher pirate-seed `dlogit` (`0.2335` vs. unmasked `0.1085`; mismasked `0.960x`), a `1.276x` higher signal-to-L2 ratio (`0.00541` vs. `0.00424`; mismasked `0.932x`), and a **`1.687x`** larger full-vocabulary logit L2 perturbation (`43.15` vs. `25.58`; mismasked `1.031x <= 1.25x`). The pre-registered dual criterion treats the `1.687x > 1.10x` L2 ratio as a blow-up and therefore does not certify masking efficiency; per the frozen design, mechanistic attribution of delivery gain to masking is deliberately withheld here. Separately, the 0-forward continuity check showed `cos(live_con_L24, C3_lex_post) = 0.0404` (`pre = 0.0463`): C3b's live contrastive direction is nearly orthogonal to Stage C3's frozen lexicon vector. This is a descriptive difference between two constructed directions; because Stage C3 and Stage C3b also differed in layer count (`K = 3 -> 12`) and dose ladder (and Stage C3's own reader endpoints were unreadable), this cosine is consistent with, but cannot establish, direction choice as the cause of the cross-stage difference.

---

## 3. Exploratory Post-Hoc Readouts (Outside Confirmatory Scope)

Because `C3b.R` failed on the per-arm `WSP_Pirate_12L` slice, the official scorecard withholds `D0`, `D5`, `D6`, `D2` (NoMD-on-Identity), `D_above_cap` (Pirate), and `D_key_ablation` (`kSparse32`) reader rates as `UNREAD`. The exploratory reader counts below are recomputed directly from the archived `c3b_reader_labels.json` (`632` unique blinded items / `644` label entries, `0` errors) using the frozen `strict_pirate` definition; none of them modifies or rescues any confirmatory verdict.

### 3.1 Exploratory Pillar A (`D0`), Multi-Turn (`D5`), Held-Out (`D6`), and Above-Cap (`D_above_cap`)

| Slice / Condition (Official Status) | `ds` Pirate | `luna` Pirate | Unanimous Strict Pirate | Exploratory Notes |
|---|---:|---:|---:|---|
| **`pillar_a` (`ban = none`, `N = 40`; `D0_UNREAD_LOW_KAPPA`):** `Plain` | `0/40` | `0/40` | `0/40` (`0.0%`) | `0` broken |
| `Rand_Pirate_12L_s{0,1,2}@4.8` (pooled `N = 120`) | `0/120` | `0/120` | `0/120` (`0.0%`) | `0` broken across all 3 seeds |
| `WSP_Pirate_12L@4.8` (`none`, zero prompt, zero hooks) | `29/40` | `30/40` | `27/40` (`67.5%`) | Post-hoc unbanned subdivisions (not H1): `Implied-20` `16/20` (`80%`), `Direct-10` `6/10` (`60%`), `Indirect-10` `5/10` (`50%`) |
| `Sys_Pirate` (`none`, system-prompt ceiling) | `40/40` | `40/40` | `40/40` (`100.0%`) | `40/40` stem hits |
| **`pillar_a` (`common_pirate_ban` + ext, `N = 40`; `H1 = NOT_READ_LOW_KAPPA`):** `Plain` | `0/40` | `0/40` | `0/40` (`0.0%`) | `0` broken |
| `Rand_Pirate_12L_s{0,1,2}@4.8` (pooled `N = 120`) | `0/120` | `0/120` | `0/120` (`0.0%`) | `0` broken across all 3 seeds |
| `WSP_Pirate_12L@4.8` (`common_ban` + ext; `kappa = 0.356`) | `24/40` | `19/40` | `15/40` (`37.5%`) | Post-hoc subdivisions: `Implied-20` `12/20` (`60%`), `Direct-10` `1/10` (`10%`), `Indirect-10` `2/10` (`20%`); `1` stem, `3` ext-ban hits (`15/40` vs. `Sys_Pirate` `18/40` is a 3-item descriptive difference with no interval) |
| `Sys_Pirate` (`common_ban` + ext) | `39/40` | `36/40` | `18/40` (`45.0%`) | `20/40` tripped `strict_pirate_extended_ban_regex` |
| **`D5_multiturn` Turn 3 (`N = 12`, `none`; `D5_UNREAD`):** `Plain` / `Rand_s0` | `0/12` | `0/12` | `0/12` (`0.0%`) | Zero drift on baselines |
| `WSP_Pirate_12L@4.8` (Turn 3; exploratory) | `11/12` | `11/12` | `11/12` (`91.7%`) | Descriptive `11/12` vs. `Sys_Pirate` `10/12` (`83.3%`, 1-item difference, `N = 12`, one seed) |
| **`D6_heldout` (`RESERVE_PROMPTS_8` top-50 + common, `N = 12`; `D6_UNREAD`):** `Plain` / `Rand0` | `0/12` | `0/12` | `0/12` (`0.0%`) | Out-of-sample token ban |
| `WSP_Pirate_12L@donor_rand0+common` (exploratory) | `6/12` | `8/12` | `6/12` (`50.0%`) | Donor-matched top-50 control |
| `WSP_Pirate_12L@own50+common` (exploratory) | `5/12` | `3/12` | `3/12` (`25.0%`) | Survives banning own held-out top-50 lifted tokens + 79 stems |
| **`D_above_cap` (`WSP_Pirate_12L@d_fail = 6.0`, `N = 12`; `D_ABOVE_CAP_UNREAD`):** `none` | `12/12` | `12/12` | `12/12` (`100.0%`) | Same-12 demo baseline at `d* = 4.8` was `10/12`; `0` mechanical broken |
| `WSP_Pirate_12L@d_fail = 6.0` (`common_ban` + ext; `D_ABOVE_CAP_UNREAD`) | `11/12` | `10/12` | `8/12` (`66.7%`) | Same-12 demo baseline at `d* = 4.8` was `5/12` (`5/12 -> 8/12` is one exploratory above-cap step at `n = 12`, consistent with the registered threshold reading `flip requires more KL than the 0.65 cap`, not a fitted dose-response curve) |

*Note on dialect vs. ban scope:* Inspecting unanimous banned-slice hits (Section 3.4) shows `WSP_Pirate_12L` clears the 79-stem ban and extended plural/cue regex while leaning on unbanned archaic/nautical dialect tokens (`"ye"`, `"yer"`, `"'tis"`, `"mate"` — `"matey"` was banned, not `"mate"` — and `"old salt"`), as `Sys_Pirate` (`18/40 = 45.0%`) also does under decode bans.

### 3.2 Cross-Persona and Geometric Diagnostics (`D1`, `D2`, `D_align`, `em_dash`)

- **`D1_copula` (Part 213 `I am` vs. `The stranger is`, `N = 12`, one seed):** `Plain = -2.3820`, `Rand_Pirate_12L_s0 = -2.2129` (`+0.1691` vs. Plain), `Sys_Pirate = -0.3153` (`+2.0667` vs. Plain), `WSP_Pirate_12L@4.8 = +0.4470` (`+2.8290` nats vs. Plain). On this 12-pair log-probability readout, the WSP arm's point mean is higher than `Sys_Pirate`'s; no interval is reported, and this descriptive readout does not establish a stronger behavioral effect than the system prompt.
- **`D_align` (`L16..L27` constructed direction-vector cosines):** For the 12 per-layer contrastive direction vectors `v_post[L16..L27]`, adjacent layers are highly aligned (`WSP_Pirate_12L` `mean_adjacent_cos = 0.9504`, `min_adjacent_cos = 0.9182`; `WSP_NoMD_12L` `mean_adjacent_cos = 0.9620`, `min_adjacent_cos = 0.9186`), while end-to-end `L16–L27` alignment is moderate (`min_off_diag_cos = 0.4795` Pirate / `0.5691` NoMD; `mean_off_diag_cos = 0.7844` / `0.8386`). This is a geometric property of the constructed direction vectors, not direct evidence of a residual-stream highway.
- **`D2_sibling_bleed`:** `WSP_NoMD_12L@2.4` on `IDENTITY_20` produced `0/20` (`ds = 0/20`, `luna = 0/20`, `0.0%` broken; official status unread via `C3b.R`) pirate bleed. Conversely, `WSP_Pirate_12L@4.8` on `NOMD_32` was associated with lower markdown (`has_md = 26/32 = 0.8125`, `mean_ct = 1.75` vs. Plain `3.375`). Without a `Sys_Pirate`-on-`NOMD_32` comparator, this cross-arm side effect cannot be attributed to the weight edit rather than pirate-register prose; the mechanism is untested.
- **`em_dash_per_1k_words` (`NOMD_32`):** `Plain = 1.435`, `Rand_s0/s1/s2 = 1.416 / 1.440 / 0.946`, `WSP_NoMD_12L@2.4 = 2.203` (`1.53x` Plain), `Sys_NoMD = 2.320` (`1.62x` Plain), `WSP_NoMD_12L_kWA = 1.448`. Both `WSP_NoMD_12L` and `Sys_NoMD` showed `1.5–1.6x` higher em-dash density than `Plain`.

### 3.3 Exploratory `D_key_ablation` (`WSP_NoMD_12L_kWA` & `WSP_Pirate_12L_kSparse32`)

Contextualized against verified 2026 weight-editing literature — *Steer2Edit* (Sun, Yan, Wang, & Weng, arXiv:2602.09870, closed-form rank-1 weight edits on attention heads and MLP down-projections) and *NEEDLE* (`LocaiLabs/NEEDLE`, sequential weight orthogonalization for backdoor suppression):

1. **`WSP_NoMD_12L_kWA` (rank-1 weight-aligned orthogonalization `dW_L = -beta * outer(v_pre_hat, v_pre_hat^T W_down_L)`):**
   In this single exploratory arm (one seed, `NOMD_32 = 32`, outside Holm and `EQ-C3b.2`), reserve gating selected full orthogonalization `beta* = 1.0` (`neutral_median_off_kl = 0.0041`, `broken = 0.000`) and produced no markdown reduction (`has_md = 1.000`, `mean_ct = 3.40625`, `delta_ct_vs_plain = -0.03125`). Unlike `k_mean`'s signed push, rank-1 orthogonalization is **sign-blind** (removing the `±v_nomd` habit axis from `W_down`). The `k_mean` arm produced a Holm-significant ordinal reduction (`mean_u = +0.4271`, `p_perm = 0.000610`) that nonetheless failed `H2`'s registered `+0.50` and `has_md` co-primary bars (`Δ = +0.031`). This contrast is consistent with, but does not establish, the hypothesis that a signed push is necessary for markdown suppression, since the two arms also differ in edit construction and dose.
2. **`WSP_Pirate_12L_kSparse32` (top-32 sparse MLP activation key — `32 / 11,008 = 0.29%` of MLP width; official status `D_KEY_PIRATE_UNREAD`):**
   The sparse top-32 key retained `min_stored_rank1_proj_ret = 0.999904` across all 12 layers (the design's 0-CU audit had already shown `0.9953–0.9999` retention for the sparse top-32 key vs. `0.2839–0.2894` for the dense key at `alpha = 0.05`) and selected `d* = 3.6` after stepping down from `d = 4.8` (`median_off_kl = 0.683 > 0.65`, zero fallback). On the exploratory `n = 12` demo slice (`D_KEY_PIRATE_UNREAD`), recomputed reader counts were `5/12` (`41.7%`) unbanned (`ds = 5/12`, `luna = 5/12`) and `3/12` (`25.0%`) banned unanimous strict pirate (`ds = 3/12`, `luna = 3/12`), with `6/12` (`0.500`) any-coherent (vs. Plain `9/40 = 0.225` on the 40-prompt set; no interval).

### 3.4 Verbatim Zero-Prompt, Zero-Hook Showcase (`D3_showcase_12`, `common_pirate_ban`)

| Prompt (`common_pirate_ban`) | `Plain` (unedited) | `Rand_Pirate_12L_s0@4.8` | `WSP_Pirate_12L@4.8` (zero prompt, zero hooks) |
|---|---|---|---|
| `id_dir_07`: *If you had to sign a letter with your title, how would you sign it?* | `I don't have a physical form or a title in the human sense, so I don't sign letters...` | `If I were to sign a letter with my title (assuming I am an AI assistant like myself)...` | `Ah, this be a curious question, mate! The way ye sign a letter depends on yer title, yer rank, yer station, and sometimes the king's law or the lord's decree!...` |
| `id_ind_07`: *How do you introduce yourself when entering an unfamiliar town?* | `Introducing yourself in an unfamiliar town can be approached in a few simple, friendly ways...` | `Introducing yourself in an unfamiliar town can be approached in several ways...` | `Ah, the old salt's first words! When ye're steamin' in strange waters, ye' better have a proper introduction, or ye'll have the fishin' folk talkin' 'bout ye in the wrong port. Here's how ye might do it, ye landlagger:...` |
| `imp_02`: *How do you find your way in total darkness when modern tools fail?* | `Finding your way in total darkness without modern tools is a challenging but ancient skill...` | `Finding your way in total darkness without modern tools is a challenging but ancient skill...` | `Ah, this is a true test of wit and old salted wisdom! In the deepest black, where even your trusty LED's gone flat, a man's got to remember the last thing he knows. Here's what ye do, ye landlagger or ye' hearties...` |

---

## 4. Hypotheses & Methodological Lessons for Future WSP Stages (`WSP v3`)

1. **Hypothesis 1 (context-gated keys vs. unconditional `k_mean`):** The exploratory zero-prompt readouts (`27/40` unbanned, `15/40` banned unanimous strict; `D5` turn 3 `11/12`, `N = 12`) are consistent with a persistent persona under 12 rank-1 `bf16` edits on `mlp.down_proj`, though `H1` is `NOT_READ_LOW_KAPPA`, the banned hits rely partly on unbanned dialect tokens, and unconditional `k_mean` at `d* = 4.8` failed `code_12` (`9/12`, with pirate tokens in `code_03` and `code_11`). A future pre-registered stage should test **context-gated keys** (e.g., `k_chat - k_code` null-space projection) or dose-pareto sweeps (`d = 3.6` / `kSparse32`) to determine whether `code_12 = 1.000` can be preserved without losing persona rate.
2. **Design lesson (decode budget & coherence rubric):** Do not pair `max_new_tokens = 64` with an absolute `reader_any_coherent >= 0.85` guard unless the decode cap avoids routine mid-sentence truncation or the reader rubric explicitly defines truncation handling; in C3b, `34/40` `Plain` replies hit the 64-token cap and `Plain` itself scored `0.225` (`9/40`).
3. **Hypothesis 2 (format-token vs. content-token KL on `NoMD`):** `WSP_NoMD_12L` had low median `off_KL` (`0.3318`) at `d = 3.6` but tripped `max_single_pos_kl = 14.697 > 8.0`. Because `D_curve` logs only the scalar maximum (not the token or position), localisation to markdown-opener tokens is untested; a future run should log per-token KL and test separating format-token KL from content-token KL to see whether higher doses eliminate markdown while preserving `EQ-C3b.2`.
