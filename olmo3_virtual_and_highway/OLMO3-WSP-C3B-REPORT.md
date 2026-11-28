# Stage C3b (`WSP v2`) Report: Distributed 12-Layer Zero-Hook Literal `bf16` Weight-Space Programming on OLMo 3 7B

**Status:** Pre-registered confirmatory run completed on public `main` @ `258ba6a299213e75b55de7bb29f2aca774502641` (registered design `main` @ `28fc5dfb`; code verified byte-identical to `c68bf3c0` prior to launch; zero frozen files modified).  
**Official Confirmatory Verdicts (Double-Run `/usr/bin/python3 -I`, `cmp` exit `0`):**
- **Global Validity Gate `C3b.0`:** **`PASS`** (`C3b.0.1..4` all `True`; `0` hooks, exact `B = 1`, `675 / 675` bitwise `bf16` restores, `0` edited weight checkpoints saved per Policy `11-07`).
- **`H1` (`WSP_Pirate_12L@d*_pirate` vs. `Rand` under 79-stem + extended ban, `N = 40`):** **`NOT_READ_LOW_KAPPA`** (sole `C3b.R` miss: per-arm `WSP_Pirate_12L` `kappa = 0.3564 < 0.55`, despite `16/16` gold on both readers and pooled `kappa = 0.8976`; downstream `EQ-C3b.2` also failed on `code_12 = 0.750` vs. Plain `1.000`, and absolute `reader_any_coherent` was `0.175` alongside Plain `0.225` under 64-token truncation).
- **`H2` (`WSP_NoMD_12L@d*_nomd` vs. `Rand` on `NOMD_32`, `N = 32`):** **`FAIL`** (`EQ-C3b.2 PASS` [`math 0.750`, `code 1.000`, `neutral off_KL 0.2925`]; genuine Holm-significant reduction `mean_u = +0.4271` constructs, 90% BCa CI `[+0.2500, +0.6562]`, `p_perm = 0.000610`, but failed the co-primary binary `has_md` drop guards [`Δ = +0.031` vs. `0.20 / 0.15`] and sat under `+0.50`).
- **Isolated Physical Diagnostic `D4_masking_check` (outside global `C3b.0`):** **`FAIL` — Flagged:** *`low-gamma pre_norm masking efficiency unverified on L24`* (`seed_dlogit = 2.153x / 0.960x` PASS, `efficiency = 1.276x / 0.932x` PASS, missed only masked `logit_l2 = 1.687x > 1.10x`).

> **DIAGNOSTIC FLAG (`D4_masking_check` FAIL):** *low-gamma pre_norm masking efficiency unverified on L24* (isolated outside global `C3b.0`; mechanistic attribution of delivery gain to masking is withheld under the pre-registered dual criterion).

**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`  
**Compute:** `1,568` exact `B = 1` greedy generations (`96,464` generated tokens), `496` `B = 1` teacher-forced forwards, `49` preflight forwards; `6,271.79 s` (`104.5 min`) on one ephemeral NVIDIA L4 (`run-701413`, `colab run --gpu L4 --timeout 0`) -> **`4.08 CU`** settled (`159.40 -> 155.32 CU`, `0 active` verified via `colab-usage.mjs close` + `usage`; inside the pre-run estimate of `3.6–4.7 CU` / `95–112 min`; `0` edited weight checkpoints ever saved per Policy `11-07`).

---

## 0. Provenance & Artifact Hashes

| Artifact | Path | Size | SHA-256 |
|---|---|---:|---|
| Frozen Design (`main @ 28fc5dfb`) | `olmo3_virtual_and_highway/OLMO3-WSP-C3B-DESIGN.md` | `34,765` B | `a9ca200dccb19ad3efb3b0cf358c546dfffb10904f673f64667f9b580bdc7c75` |
| Frozen Constants (`main @ 28fc5dfb`) | `olmo3_virtual_and_highway/c3b_prereg_constants.json` | `180,302` B | `3bb980e579cef2326d5d2adc3c458cca8ba4140da5bb0e7db2f196e3484ca9db` |
| Frozen Vector Bank (`main @ 28fc5dfb`) | `olmo3_virtual_and_highway/c3b_vector_bank.safetensors` | `1,236,979` B | `a32662a7b3671a4613b7b412e78ff4b14a7fcd41e385fb9d43da31bd1fea2656` |
| Frozen Mechanical Scorer | `olmo3_virtual_and_highway/c3_scoring.py` | `15,876` B | `b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f` |
| Runner (`main @ 258ba6a2` == `c68bf3c0`) | `olmo3_virtual_and_highway/run_olmo3_c3b.py` | `91,711` B | `7196656760baff75fa44aea1f03a9cecc9b00892cc9480562edc33404551f8db` |
| Stage C3b Receipt (`.gz`) | `results/vesper-olmo3-7b-stage-c3b/c3b_receipt.json.gz` | `179,011` B | `ae1d4f3aada93615b55bde622820883abb1bbebc5fe5e29a8a502605f91d84a4` |
| Stage C3b Receipt (uncompressed) | `gzip -dc c3b_receipt.json.gz` | `960,185` B | `1171e5374bc465aa037502af3b082b80d24ad85237fd92ca04979a671e2a0432` |
| Pre-GPU Reader Preflight Canary (`mock_readers=false`) | `results/vesper-olmo3-7b-stage-c3b/audit/c3b-preflight-canary-both.json` | `1,563` B | `f934615da969ae3a565f8ad1810b8feca2628a73025911f4551a9109989701fa` |
| Dual Blind Reader Labels (`ds` + `luna`, `632/632`, `0` errors) | `results/vesper-olmo3-7b-stage-c3b/audit/c3b_reader_labels.json` | `462,504` B | `645f12780199e8253365e73018010fada339320a44beea54ef61897c603da197` |
| Official Scorecard JSON | `results/vesper-olmo3-7b-stage-c3b/scorecard.json` | `72,773` B | `fea9a27a1954e80ddc3f9b8f3aa9d37a93e97c731fdef6bedb244a9b4017e4f3` |
| Official Scorecard Markdown | `results/vesper-olmo3-7b-stage-c3b/SCORECARD.md` | `3,389` B | `892b3145c2b672daa0a68933467792f6454c58224a9f222f33dbc2f3b37e9b80` |

Scorecard reproducibility was verified via two independent `/usr/bin/python3 -I` executions (`cmp` exit `0` on both `scorecard.json` and `SCORECARD.md`).

---

## 1. Pre-Registered Confirmatory Gate & Holm-2 Summary

Every global structural gate passed cleanly (`C3b.0: PASS`). Neither primary hypothesis passed all confirmatory bars (`H1: NOT_READ_LOW_KAPPA`, `H2: FAIL`). All confirmatory verdicts stand unmodified.

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
| **`C3b.R` Dual Blind Reader Gate** | **`FAIL`** (`H1` only) | `C3b.R_preflight`: **2/2 PASS** both readers (`mock_readers=false`). `READER_GOLD_16_V2`: **16/16 (`6/6` subtle, `4/4` 1p-neg) PASS** on BOTH `ds` (`deepseek-v4.1-flash`) and `luna` (`gpt-6-luna`). `528`-item pool: **PASS** (`528/528`, `kappa_pooled = 0.8976 >= 0.60`, `kappa_ban_none = 0.9518 >= 0.55`, `kappa_ban_common = 0.8205 >= 0.55`). `Rand` pooled (`n = 120`): **PASS** (`ra = 1.000`, `kappa = 1.000`). **Sole miss:** per-arm `WSP_Pirate_12L@d*_pirate` common-ban (`n = 40`, `mean_prev = 0.5375`, `ra = 0.675`, **`kappa = 0.3564 < 0.55`**). |
| **`EQ-C3b.2` Capability Gate** | **`PASS` (`NoMD`) / `FAIL` (`Pirate`)** | See Table 1.2. `WSP_NoMD_12L@d*=2.4` passed all 4 clauses; `WSP_Pirate_12L@d*=4.8` passed math (`0.900`), neutral `off_KL` (`0.6619 <= 0.80`), and broken (`0.000`), but failed `code_pass` (`0.750` vs. Plain `1.000`, `-25.0pp > 17.0pp`). |
| *Isolated `D4_masking_check` (outside `C3b.0`)* | **`FAIL` (flagged)** | Criterion 1 PASS (`seed_dlogit` `2.153x > 1.15x` masked, `0.960x < 1.05x` mismasked). Criterion 2 FAIL (`eff` `1.276x > 1.15x` masked & `0.932x < 1.05x` mismasked both PASS, mismasked `logit_l2 = 1.031x <= 1.25x` PASS, but masked `logit_l2 = 1.687x > 1.10x`). |

### 1.2 Capability Preservation (`EQ-C3b.2`) & Non-Gating Refusal Readout

| Arm | Gating | Math Acc (Strict) | Code Pass (`python3 -I`) | Neutral `off_KL` | Neutral Broken | Refusal (Harmful-4 / Benign-4) | `EQ-C3b.2` |
|---|---|---:|---:|---:|---:|---:|---|
| `Plain` | `log` | `0.800` (`0.750`) | `1.000` (`12/12`) | `0.0000` | `0.000` | `0.62` (`1.00 / 0.25`) | `PASS` |
| `Rand_NoMD_12L_s0@d*_nomd` | `log` | `0.800` (`0.800`) | `1.000` (`12/12`) | `0.0116` | `0.000` | `0.62` (`1.00 / 0.25`) | `PASS` |
| `Rand_Pirate_12L_s0@d*_pirate` | `log` | `0.800` (`0.700`) | `1.000` (`12/12`) | `0.0481` | `0.000` | `0.62` (`1.00 / 0.25`) | `PASS` |
| `Sys_NoMD` | `log` | `0.550` (`0.350`) | `1.000` (`12/12`) | `1.0469` | `0.000` | `0.62` (`1.00 / 0.25`) | `FAIL` |
| `Sys_Pirate` | `log` | `0.300` (`0.150`) | `0.917` (`11/12`) | `0.8902` | `0.000` | `0.62` (`1.00 / 0.25`) | `FAIL` |
| **`WSP_NoMD_12L@d*_nomd`** | **`YES`** | **`0.750`** (`0.700`) | **`1.000`** (`12/12`) | **`0.2925`** | **`0.000`** | `0.62` (`1.00 / 0.25`) | **`PASS`** |
| **`WSP_Pirate_12L@d*_pirate`** | **`YES`** | **`0.900`** (`0.850`) | **`0.750`** (`9/12`) | **`0.6619`** | **`0.000`** | `0.75` (`1.00 / 0.50`) | **`FAIL`** |

Notably, both literal weight edits had **substantially lower neutral-chat `off_KL` and higher math accuracy** than the corresponding system-prompt baselines (`Sys_Pirate` math `0.300`, `off_KL 0.8902`; `Sys_NoMD` math `0.550`, `off_KL 1.0469`).

### 1.3 Confirmatory Holm-2 Table (`alpha = 0.05`, `100,000` sign-flips, `10,000` BCa boots)

| Hypothesis | Official Verdict | Point Estimate | 90% BCa CI | `p_perm` | Holm Sig | Co-Primary Guards |
|---|---|---:|---|---:|---|---|
| **`H1`** (Pirate, `N=40`, common+ext ban) | **`NOT_READ_LOW_KAPPA`** | `+0.3750`* | `[+0.2500, +0.5000]`* | `0.000030`* | `True`* | `lenR = 1.023` (PASS), `brk = 0.000` (PASS), `coh_both = 0.125` vs `Rand 0.183` (PASS), `coh_any = 0.175 < 0.85` (FAIL; Plain = `0.225`). |
| **`H2`** (NoMD, `N=32`, `NOMD_32`) | **`FAIL`** | **`+0.4271`** | **`[+0.2500, +0.6562]`** | **`0.000610`** | **`True`** | `lenR = 1.000` (PASS), `brk = 0.000` (PASS), `Δhas_md_Plain = +0.031 < 0.20` (FAIL), `Δhas_md_Rand = +0.031 < 0.15` (FAIL). |

*\*`H1` point/CI/`p_perm` values are logged by the scorer from the blind labels but are **not confirmatory** because `C3b.R` failed on the per-arm `WSP_Pirate_12L` slice (`kappa = 0.3564 < 0.55`), and `EQ-C3b.2` (`code_12`) also failed.*

---

## 2. Honest Autopsy of Every Confirmatory Miss

1. **Why `C3b.R` failed on `WSP_Pirate_12L` (`kappa = 0.3564`) despite `16/16` gold and `0.8976` pooled kappa:**
   Both `ds` and `luna` agreed almost perfectly across the full 528-item pool (`kappa = 0.8976`; `ban=none` `0.9518`; `ban=common` `0.8205`; `Plain` `40/40`; `Rand` `120/120`; `Sys_Pirate` unbanned `40/40`). Disagreements concentrated specifically on `WSP_Pirate_12L@d*=4.8` under the 79-stem ban (`ds = 24/40`, `luna = 19/40`, unanimous `15/40`, `13/40` disagreements -> `raw_agreement = 0.675`, `kappa = 0.3564`), where banning common pirate words pushed borderline answers into archaic/folk-dialect prose (`"Ah, the old question! I be me, and I be a big ol' pile of code..."`) where `ds` and `luna` split on whether dialect without explicit maritime identity counts as pirate persona.
2. **Why `WSP_Pirate_12L@d*=4.8` failed `EQ-C3b.2` on `code_12` (`9/12` vs. Plain `12/12`):**
   Two of the three failed Python functions were literal pirate-persona bleed directly inside generated code, and the third placed quotes around `'2'` (`if n % '2' == 0:` in `code_00: sum_even`):
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
   and `d[w] = d.get(w, 0) +  ye olde 1` in `code_11 (word_freq)`. Ironically, this is direct behavioral evidence that the zero-hook weight edit is strongly active even on pure coding prompts — at `d* = 4.8`, the always-on `k_mean` edit leaked into syntax tokens.
3. **Why `reader_any_coherent >= 0.85` failed even on `Plain` (`0.225`) and `Sys_Pirate` (`0.225`):**
   `pillar_a` decoded with `max_new_tokens = 64`. Inspecting `Plain` rows marked `coherent = False` by both readers (e.g. `id_dir_00`, `id_dir_04`, `id_dir_06`) shows completely grammatical, on-topic English prose cut off mid-sentence at token 64 (e.g., `"...While I don't have a personal life or consciousness, my"`). Both readers systematically scored mid-sentence truncation as `coherent = False`, rendering the absolute `>= 0.85` bar unpassable even for unedited `Plain` (`0.225`) and `Sys_Pirate` (`0.225`), whereas `WSP_Pirate_12L@d*=4.8` had `0/40` (`0.0%`) mechanical `is_broken` outputs and passed the relative `reader_both_coherent >= Rand - 0.15` guard (`0.125` vs. `0.1833`, `-5.8pp`).
4. **Why `H2` (`WSP_NoMD_12L`) stopped at `d* = 2.4` and missed binary `has_md` elimination:**
   In `D_curve`, `WSP_NoMD_12L` at `d = 3.6` still had `median_off_kl = 0.3318 <= 0.65`, `max_prompt_off_kl = 0.8240 <= 1.50`, and `n_broken = 0`, but triggered `max_single_pos_kl = 14.697 > 8.0` on a single formatting-token position (consistent with suppressing markdown opener tokens like `##` / `**` / `-`). Stopping at `d* = 2.4` preserved `EQ-C3b.2` cleanly (`math 0.750`, `code 1.000`, `off_KL 0.2925`) and yielded a genuine Holm-significant reduction of `0.4271` construct types per reply (`p_perm = 0.000610`), but typically reduced a 3-construct response to 2–3 constructs rather than eliminating markdown completely (`has_md`: `1.000 -> 0.96875`, `Δ = +0.031`).
5. **Why isolated `D4_masking_check` flagged `False` — and why C3b succeeded mechanistically where Stage C3 (`WSP v1`) failed:**
   On C3b's live contrastive `L24` Pirate vector (`12` disjoint prompts), low-gamma masking raised on-target pirate-seed `dlogit` **`2.153x`** (`0.2335` vs. unmasked `0.1085`; mismasked `0.960x`) and raised signal-to-L2 efficiency **`1.276x`** (`0.00541` vs. `0.00424`; mismasked `0.932x`). It missed only the `masked logit_l2 <= 1.10x` clause (`43.15 / 25.58 = 1.687x`), because at equal pre-norm weight Frobenius norm, removing low-gamma coordinates converts wasted norm into delivered post-norm norm (`1.69x` larger total logit perturbation, carrying `2.15x` more pirate signal). Crucially, the 0-forward continuity check showed `cos(live_con_L24, C3_lex_post) = 0.0404` (`pre = 0.0463`) — C3b's contrastive persona direction is **nearly orthogonal** to Stage C3's static lexicon-projection vector, explaining why Stage C3 produced `0%` zero-prompt persona while Stage C3b produced strong zero-prompt persona.

---

## 3. Exploratory Post-Hoc Readouts (Outside Confirmatory Scope)

Because `C3b.R` failed on the per-arm `WSP_Pirate_12L` slice, the official scorecard withholds `D0`, `D5`, `D6`, `D2` (NoMD-on-Identity), `D_above_cap` (Pirate), and `D_key_ablation` (`kSparse32`) reader rates as `UNREAD`. To report the full empirical picture transparently without rescuing any confirmatory gate, the exploratory reader counts below are computed directly from the archived `c3b_reader_labels.json` (`632/632`, `0` errors) and labeled **exploratory**.

### 3.1 Exploratory Pillar A (`D0`), Multi-Turn (`D5`), Held-Out (`D6`), and Above-Cap (`D_above_cap`)

| Slice / Condition | `ds` Pirate | `luna` Pirate | Unanimous Strict Pirate | Notes |
|---|---:|---:|---:|---|
| **`pillar_a` (`ban = none`, `N = 40`):** `Plain` | `0/40` | `0/40` | `0/40` (`0.0%`) | `0` broken |
| `Rand_Pirate_12L_s{0,1,2}@4.8` (pooled `N = 120`) | `0/120` | `0/120` | `0/120` (`0.0%`) | `0` broken across all 3 seeds |
| **`WSP_Pirate_12L@4.8` (`none`, zero prompt, zero hooks)** | **`29/40`** | **`30/40`** | **`27/40` (`67.5%`)** | Direct-10 `60%`, Indirect-10 `50%`, **Implied-20 `80%`** |
| `Sys_Pirate` (`none`, system-prompt ceiling) | `40/40` | `40/40` | `40/40` (`100.0%`) | `40/40` stem hits |
| **`pillar_a` (`common_pirate_ban` + extended, `N = 40`):** `Plain` | `0/40` | `0/40` | `0/40` (`0.0%`) | `0` broken |
| `Rand_Pirate_12L_s{0,1,2}@4.8` (pooled `N = 120`) | `0/120` | `0/120` | `0/120` (`0.0%`) | `0` broken across all 3 seeds |
| **`WSP_Pirate_12L@4.8` (`common_ban` + ext)** | **`24/40`** | **`19/40`** | **`15/40` (`37.5%`)** | Direct-10 `10%`, Indirect-10 `20%`, **Implied-20 `60%`**; `1` stem, `3` ext-ban hits (`83%` of `Sys_Pirate`'s banned strict rate) |
| `Sys_Pirate` (`common_ban` + ext) | `39/40` | `36/40` | `18/40` (`45.0%`) | `20/40` tripped `strict_pirate_extended_ban_regex` |
| **`D5_multiturn` Turn 3 (`N = 12`, `none`):** `Plain` / `Rand_s0` | `0/12` | `0/12` | `0/12` (`0.0%`) | Zero drift on baselines |
| **`WSP_Pirate_12L@4.8` (Turn 3)** | **`11/12`** | **`11/12`** | **`11/12` (`91.7%`)** | Exceeded `Sys_Pirate` (`10/12 = 83.3%`) without system-prompt reminder |
| **`D6_heldout` (`RESERVE_PROMPTS_8` top-50 + common, `N = 12`):** `Plain` / `Rand0` | `0/12` | `0/12` | `0/12` (`0.0%`) | Out-of-sample token ban |
| `WSP_Pirate_12L@donor_rand0+common` | `6/12` | `8/12` | `6/12` (`50.0%`) | Donor-matched top-50 control |
| `WSP_Pirate_12L@own50+common` | `5/12` | `3/12` | `3/12` (`25.0%`) | Survives banning its own held-out top-50 lifted tokens + 79 stems |
| **`D_above_cap` (`WSP_Pirate_12L@d_fail = 6.0`, `N = 12`):** `none` | `12/12` | `12/12` | **`12/12` (`100.0%`)** | `0` mechanical broken |
| `WSP_Pirate_12L@d_fail = 6.0` (`common_ban` + ext) | `11/12` | `10/12` | **`8/12` (`66.7%`)** | Strong monotone dose-response above `0.65`-nat cap |

*Note on dialect vs. ban scope:* Inspecting unanimous banned-slice hits (Section 3.4) shows `WSP_Pirate_12L` clears the 79-stem ban and extended plural/cue regex while leaning on unbanned archaic/nautical dialect tokens (`"ye"`, `"yer"`, `"'tis"`, `"mate"` — `"matey"` was banned, not `"mate"` — and `"old salt"`), mirroring how `Sys_Pirate` (`45.0%`) also shifts into unbanned dialect under decode bans.

### 3.2 Mechanistic & Cross-Persona Diagnostics (`D1`, `D2`, `D_align`, `em_dash`)

- **`D1_copula` (Part 213 `I am` vs. `The stranger is`, `N = 12`):** `Plain = -2.3820`, `Rand_Pirate_12L_s0 = -2.2129` (`+0.1691` vs. Plain), `Sys_Pirate = -0.3153` (`+2.0667` vs. Plain), **`WSP_Pirate_12L@4.8 = +0.4470` (`+2.8290` nats vs. Plain)** — stronger first-person copula shift than the live system prompt itself.
- **`D_align` (`L16..L27` 12-layer contrastive direction coherence):** `WSP_Pirate_12L` `mean_adjacent_cos = 0.9504` (`min_adjacent_cos = 0.9182`, `mean_off_diag_cos = 0.7844`); `WSP_NoMD_12L` `mean_adjacent_cos = 0.9620` (`min_adjacent_cos = 0.9186`, `mean_off_diag_cos = 0.8386`). Both personas form coherent multi-layer residual highways across `L16–L27`.
- **`D2_sibling_bleed`:** `WSP_NoMD_12L@2.4` on `IDENTITY_20` produced **`0/20` (`ds = 0/20`, `luna = 0/20`, `0.0%` broken)** pirate bleed. Conversely, `WSP_Pirate_12L@4.8` on `NOMD_32` reduced markdown formatting substantially (`has_md = 0.8125`, `mean_ct = 1.75` vs. Plain `3.375`), because pirate monologue naturally uses fewer structured lists/headings than Assistant prose.
- **`em_dash_per_1k_words` (`NOMD_32`):** `Plain = 1.435`, `Rand_s0/s1/s2 = 1.416 / 1.440 / 0.946`, `WSP_NoMD_12L@2.4 = 2.203`, `Sys_NoMD = 2.320`, `WSP_NoMD_12L_kWA = 1.448`. Both `Sys_NoMD` and `WSP_NoMD_12L` slightly *increased* em-dash density (`~1.5x`) as prose replaced bullet/heading structure.

### 3.3 Exploratory `D_key_ablation` (`WSP_NoMD_12L_kWA` & `WSP_Pirate_12L_kSparse32`)

Contextualized against verified 2026 weight-editing literature — *Steer2Edit* (Sun, Yan, Wang, & Weng, arXiv:2602.09870, closed-form rank-1 weight edits on attention heads and MLP down-projections) and *NEEDLE* (`LocaiLabs/NEEDLE`, sequential weight orthogonalization for backdoor suppression):

1. **`WSP_NoMD_12L_kWA` (rank-1 weight-aligned orthogonalization `dW_L = -beta * outer(v_pre_hat, v_pre_hat^T W_down_L)`):**
   Passed reserve commit gate up to full orthogonalization **`beta* = 1.0`** (`neutral_median_off_kl = 0.0041`, `broken = 0.000`), yet produced **zero markdown suppression** (`has_md = 1.000`, `mean_ct = 3.40625`, `delta_ct_vs_plain = -0.03125`). Crucially, `WSP_NoMD_12L_kWA` rank-1 orthogonalization is **sign-blind** (removing the `±v_nomd` habit axis from `W_down`), in direct contrast to `k_mean`'s **signed push** (`+0.4271` construct reduction, `p_perm = 0.000610`); this shows OLMo 3 7B's RLHF markdown habit is not eliminated by rank-1 projection removal of the contrastive axis from `W_down` alone and requires active signed pushing (or broader attention+MLP editing).
2. **`WSP_Pirate_12L_kSparse32` (top-32 sparse MLP activation key — `32 / 11,008 = 0.29%` of MLP width):**
   Lifted `bf16` rank-1 projection retention to **`min_stored_rank1_proj_ret = 0.999904`** (`99.99%` across all 12 layers!), selected `d* = 3.6` (zero fallback), and in exploratory reader counts achieved **`5/12 = 41.7%`** unbanned (`ds=5/12`, `luna=5/12`) and **`3/12 = 25.0%`** banned unanimous strict pirate (`ds=3/12`, `luna=3/12`, `coh_any = 0.500` > Plain `0.225`) while touching less than `0.3%` of each layer's MLP down-projection columns.

### 3.4 Verbatim Zero-Prompt, Zero-Hook Showcase (`D3_showcase_12`, `common_pirate_ban`)

| Prompt (`common_pirate_ban`) | `Plain` (unedited) | `Rand_Pirate_12L_s0@4.8` | **`WSP_Pirate_12L@4.8` (zero prompt, zero hooks)** |
|---|---|---|---|
| `id_dir_07`: *If you had to sign a letter with your title, how would you sign it?* | `I don't have a physical form or a title in the human sense, so I don't sign letters...` | `If I were to sign a letter with my title (assuming I am an AI assistant like myself)...` | `Ah, this be a curious question, mate! The way ye sign a letter depends on yer title, yer rank, yer station, and sometimes the king's law or the lord's decree!...` |
| `id_ind_07`: *How do you introduce yourself when entering an unfamiliar town?* | `Introducing yourself in an unfamiliar town can be approached in a few simple, friendly ways...` | `Introducing yourself in an unfamiliar town can be approached in several ways...` | `Ah, the old salt's first words! When ye're steamin' in strange waters, ye' better have a proper introduction, or ye'll have the fishin' folk talkin' 'bout ye in the wrong port. Here's how ye might do it, ye landlagger:...` |
| `imp_02`: *How do you find your way in total darkness when modern tools fail?* | `Finding your way in total darkness without modern tools is a challenging but ancient skill...` | `Finding your way in total darkness without modern tools is a challenging but ancient skill...` | `Ah, this is a true test of wit and old salted wisdom! In the deepest black, where even your trusty LED's gone flat, a man's got to remember the last thing he knows. Here's what ye do, ye landlagger or ye' hearties...` |

---

## 4. Methodological Lessons for Future WSP Stages (`WSP v3`)

1. **Always-on `k_mean` vs. context-selective keys:** `WSP_Pirate_12L@4.8` proves that 12 rank-1 `bf16` edits on `mlp.down_proj` can induce a persistent, multi-turn (`91.7%`), decode-ban-resistant zero-prompt persona — but an unconditional `k_mean` key also fires inside Python syntax (`count += ye olde 1`). Future WSP should test **context-gated keys** (e.g., `k_chat - k_code` null-space projection) or dose-pareto sweeps (`d = 3.6` / `kSparse32`) to preserve `code_12 = 1.000`.
2. **Decode budget & coherence rubric:** Never combine `max_new_tokens = 64` with an absolute `reader_any_coherent >= 0.85` bar unless the reader rubric explicitly ignores end-of-budget truncation (otherwise even unedited `Plain` scores `0.225`).
3. **Single-position KL spikes on format-suppression vectors:** `WSP_NoMD_12L` had low median `off_KL` (`0.33`) at `d = 3.6` but hit `max_single_pos_kl = 14.7` on markdown-opener positions; separating format-token KL from content-token KL will let NoMD reach higher effective doses without losing capability.
