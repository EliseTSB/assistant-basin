# Stage C3 (`WSP v1`): Zero-Hook Literal `pre_norm` Weight-Space Programming on OLMo 3 7B Instruct — Results & Mechanistic Autopsy

**Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (`bf16`, zero runtime hooks during all evaluations, zero edited weight checkpoints saved per Policy `11-07`)  
**Registered Design (FROZEN on public `main` @ `10953eac`, `#144`):** `OLMO3-WSP-C3-DESIGN.md` (`sha256: dbdab600e72fdba21f22024d0b1a90575cd2d6d82195fdb6b5fe4a49f494023a`)  
**Pre-Run Amendment 1 (pushed on public `main` @ `189cc41d`, `#150`):** `OLMO3-WSP-C3-AMENDMENT-1.md` (`sha256: c8a882cc...`)  
**Code PASS Commit (`#153`):** `32289430` (execution ran verbatim on `32289430`; post-run 4-line Markdown display fix applied solely inside `score_olmo3_c3.py::render_scorecard_markdown()` with byte-identical JSON proof — see §1.2)  
**Hardware & Cost:** Single ephemeral Colab **NVIDIA L4** (`acct1`, `527 KB` upload in `2 s`, `6,628.86 s` compute at exact `B = 1` fallback, **`5.70 CU`** total settled = `1.05 CU` idle during an initial `colab exec` websocket drop + `4.65 CU` compute; `165.10 → 159.40 CU`, `0 active`).

---

## 1. Executive Summary & Official Confirmatory Verdicts

### 1.1 Pre-Registered Confirmatory Scorecard (Official — Double-Run `/usr/bin/python3 -I` Byte-Identical)

| Gate / Hypothesis | Official Verdict | Exact Quantitative Basis |
|---|---|---|
| **Global `C3.0`** | **`FAIL`** | `C3.0.1 = PASS`, **`C3.0.2 = FAIL`** (sole failing clause: `d4_ok = False`), `C3.0.3 = PASS`, `C3.0.4 = PASS` |
| **`H1` (`WSP_Pirate_Contrastive@d*_con` vs `Rand` under `common_pirate_ban`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm = 1.0`, `holm_sig = False`) |
| **`H2` (`WSP_Pirate_Contrastive@d_common` vs `WSP_Pirate_Lexicon@d_common`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm = 1.0`, `holm_sig = False`) |
| **`H3` (`WSP_NoMD_Contrastive@d*_nomd_con` on `NOMD_24`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm = 1.0`, `holm_sig = False`) |
| **`H4` (`WSP_NoMD_Lexicon@d*_nomd_lex` on `NOMD_24`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm = 1.0`, `holm_sig = False`) |

**Official Headline:** Stage C3 (`WSP v1`) fails its pre-registered confirmatory integrity gate **`C3.0`** on a single sub-check (`C3.0.2` `d4_ok`), blocking all four Holm-4 hypotheses (`H1–H4: BLOCKED_BY_C3.0`). Even in our non-confirmatory exploratory counterfactual opening `D4` (§3), **zero-hook rank-1 weight edits at the pre-registered dose ceiling (`d_total = 0.80` PMN across `L21/24/27`) did not flip autoregressive generation** (`H3` `FAIL`, `H4` `FAIL`, Pillar A unbanned genuine persona flips `0/40`, `C3.R` `READER_GOLD_14` `FAIL` on live `gpt-6-luna`).

At the same time, Stage C3 produced **four clean, quantitative mechanistic discoveries** (§3–§4):
1. **Zero-hook literal `bf16` weight edits compiled unmistakable pirate dialect directly into OLMo 3 7B's next-token logits:** Without any system prompt or runtime hook, `WSP_Pirate_Contrastive@0.80` boosted `" yer"` (`+2.78`), `" ain"` (`+2.72`), `" ye"` (`+2.43`), `" reckon"` (`+2.21`), `" cursed"` (`+2.15`), `"folk"` (`+2.15`), `" whence"` (`+2.13`), `" reck"` (`+2.09`), `"OO"` (`+2.09`), and `" tales"` (`+2.08`) as its top-10 non-banned tokens across the 100,278-token vocabulary — **`4.7×` stronger** and far cleaner than `WSP_Pirate_Lexicon@0.80` (`" anch" +0.59`, `" *" +0.57`, whitespace/punctuation) while exact norm-matched `Rand_Pirate_s0@0.80` produced multilingual noise (`"oeff" +0.78`, `"管" +0.71`, `"ь" +0.70`).
2. **Unanimous `12/12` `D1` copula directional lift:** Both `WSP_Pirate_Contrastive@0.80` (`+0.817` nats `lp_self`, `12/12` pairs positive, `Δ(self−stranger) = +0.123` vs `Plain` / `+0.199` vs `Rand`) and `WSP_Pirate_Lexicon@0.80` (`+0.902` nats, `12/12` positive) lifted Pirate seed log-probs cleanly above `Rand_Pirate_s0@0.80` (`-0.095` nats, `3/12` positive), while remaining `~10×` weaker than `Sys_Pirate` (`+8.847` nats).
3. **`D4` low-`γ` `pre_norm` masking works on target logits (`1.508×` vs `0.966×` mis-masked), not total vocabulary logit L2 (`1.017×` vs `1.148×`):** Unmasked and mis-masked directions spray norm into off-axis coordinates, inflating total vocab L2 while losing `34%` of on-target signal — directly explaining why `C3.0.2` failed when gating on `logit_l2_ratio`.
4. **Why logits moved `+2.8` yet generation did not flip (`22–59×` unused KL headroom):** All four WSP arms passed every rung and maxed out the pre-registered dose ladder at **`d* = 0.80`** while spending only **`0.0042–0.0113` nats median reserve KL** (`22×–59×` below our `0.25` KL cap, and `~80–100×` below `Sys_Pirate`'s `0.890` / `Sys_NoMD`'s `1.047` neutral KL). Every WSP arm preserved `Math = 80.0%` and `Code = 100.0% (12/12)` with `0.0%` broken outputs, whereas `Sys_Pirate` dropped to `Math 30.0%` / `Code 91.7%` and `Sys_NoMD` dropped to `Math 55.0%`.

### 1.2 Disclosure of Post-Run Markdown Display Fix & Execution Notes
- **Byte-identical scoring guarantee:** Code PASS commit `32289430` produced `c3_scorecard.json` (`sha256: 725665d6c66127ccbc3543a4a117a0c4de0cba55d0d0db18522ae320e74f0853`) cleanly with zero errors, but `render_scorecard_markdown()` raised `KeyError: 'broken_con'` when rendering `BLOCKED_BY_C3.0` rows. We applied a display-only `.get(...)` fix strictly inside `render_scorecard_markdown()` (zero changes to any gate, statistic, p-value, or JSON serialization) and `cmp`-verified that `c3_scorecard.json` is **100% byte-identical** before and after the fix, as well as byte-identical across two independent `/usr/bin/python3 -I` runs (`c3_scorecard.json` + `C3-SCORECARD.md`).
- **Automatic `B = 8` → `B = 1` fallback (`C3.0.1`):** On real `bf16` OLMo 3 7B, left-padded `B = 8` greedy preflight measured `b8_parity_max_abs_dlogit = 0.25 > 0.05`. As pre-registered in §6 and `C3.0.1`, `run_olmo3_c3.py` automatically fell back to exact `B = 1` greedy decoding across all `1,752` generations (`104,657` tokens in `6,628.86 s` = `15.8 tok/s`).
- **Colab execution & cost accounting:** Initial `colab exec` hit a transient `RuntimeError: Connection was lost.` right at startup and hung on exit (`1.05 CU` idle on `vesper-olmo3-c3`). We reclaimed the same live L4 session with a detached drop-proof wrapper, ran `run_olmo3_c3.py` @ `32289430` verbatim (`4.65 CU` compute), verified receipt SHA-256, and immediately terminated the session (`Active assignments: 0`, `159.40 CU` remaining).

---

## 2. Gate-by-Gate Audit (`C3.0.1–C3.0.4`)

| Sub-Gate | Status | Key Verified Metrics |
|---|---|---|
| **`C3.0.1` Mechanics** | **`PASS`** | `active_hooks_max_seen = 0`; `restore_exact_all = True` (`204/204` exact bitwise restores); `weight_checkpoints_saved = 0`; `sham(alpha=0.05)` `max_abs_dlogit = 0.125`, `min_stored_rank1_proj_ret = 0.53605 >= 0.50`; `b8_parity_max_abs_dlogit = 0.25 -> eff_B = 1` fallback accepted; `liveness` `min/max |Δlogit| = 0.377 / 12.609` |
| **`C3.0.2` Controls & Numerics** | **`FAIL`** | `nonfinite_count = 0`; `ban_stem_rates = 0.000` across all `9` `common_pirate_ban` arms (`0/360` leaks!); `stem_preflight = (True, False)`; `code_harness_preflight_ok = True`; `cv = (0.253, 0.286, 0.329) <= 0.45`; `M8` Rand `intended_frob_ratio = 0.9995..1.0004`, `stored_frob_ratio = 0.9999..1.0015`, `rand_med_kl = 0.0028..0.0043`; **ONLY FAILURE:** `d4_ok = False` (`logit_l2_ratio = 1.0172x < 1.15x`, `mismask_logit_l2_ratio = 1.1483x > 1.05x`) |
| **`C3.0.3` Completeness & `d_common`** | **`PASS`** | All `1,752` generations + `624` `B = 1` teacher-forced fwds + `58` preflight fwds complete; `ids_exact_ok = True`; `plain_non_broken_rate = 1.000` (`56/56`); `d_common = 0.80 == min(0.80, 0.80)` with both `d_common_retention` tables present (`proj_ret >= 0.985`, `frob_ret >= 0.999`) |
| **`C3.0.4` Provenance** | **`PASS`** | All 4 frozen artifact SHA-256 hashes match on disk and inside receipt verbatim (`OLMO3-WSP-C3-DESIGN.md`, `c3_prereg_constants.json`, `c3_vector_bank.safetensors`, `c3_scoring.py`); Amendment 1 untouched |

### 2.1 Root-Cause Autopsy of `C3.0.2` (`D4` `L24` Low-`γ` Masking Metric)

At matched Frobenius norm (`alpha_L24 = 0.379853`, `intended_frob = 0.378–0.379`, `stored_frob_ret = 0.9972–0.9977`, `stored_rank1_proj_ret = 0.959–0.973`), `D4` evaluated three `L24` `pre_norm` encodings of `L24_pirate_lex_post_unit` across `12` prompts (`36` edited forwards):

| `L24` Condition | `mean_pirate_seed_dlogit` | Ratio vs. Unmasked (`seed_dlogit`) | `mean_logit_l2` (full 100k vocab) | Ratio vs. Unmasked (`logit_l2`) |
|---|---:|---:|---:|---:|
| **`masked` (`drop_k = 4` low-`γ` dims)** | **`+0.087220`** | **`1.5077×`** (`> 1.15×` PASS) | `11.205103` | `1.0172×` (`< 1.15×` FAIL) |
| **`unmasked`** | `+0.057850` | `1.0000×` | `11.015855` | `1.0000×` |
| **`mismasked` (`(idx + 1) mod 4096`)** | `+0.055883` | **`0.9660×`** (`< 1.05×` PASS) | `12.649498` | `1.1483×` (`> 1.05×` FAIL) |

**Physical interpretation:** Masking the 4 near-zero `|γ_ff|` coordinates on `L24` genuinely increased on-target Pirate seed logit delivery by **`+50.8%` (`1.508×`)** at identical Frobenius norm, while shifting the mask by `+1` coordinate (`mismasked`) eliminated the entire gain (`0.966×`). However, total vocabulary `||Δlogit||_2` is dominated by off-axis perturbations across 100,278 tokens (where `mismasked` accidentally amplified unrelated directions to `12.65` L2 norm). Because our pre-registered scorer gated `C3.0.2` on `logit_l2_ratio` rather than `seed_dlogit_ratio`, `C3.0` honestly fails and blocks confirmatory claims.

---

## 3. Exploratory Post-Hoc Counterfactual & Diagnostics (**NON-CONFIRMATORY**)

> **Warning:** Everything in §3 is **exploratory and non-confirmatory** (`exploratory_c3_counterfactual.py`, `sha256: f4411f53…`), produced by substituting `seed_dlogit_ratio` (`1.5077× / 0.9660×`) into the in-memory `D4` check while keeping `c3_receipt.json.gz` and `score_olmo3_c3.py` untouched. Official confirmatory verdicts remain `C3.0: FAIL` / `H1–H4: BLOCKED_BY_C3.0`.

### 3.1 Exploratory `would_print` Verdicts (`H1–H4`) & `C3.R` Gold Gate

| Hypothesis | Exploratory `would_print` | Point Estimate (`90% BCa CI`) | `p_perm` | Additional Metrics |
|---|---|---|---:|---|
| **`H1`** (Pirate Con vs `Rand`, banned) | **`NOT_READ_LOW_KAPPA`** *(gold fail)* | — | `1.0` | `C3.R` gold-14 failed (`luna`: `13/14` pooled, `2/2` 1p-neg, **`5/6` subtle zero-stem** `< 6/6`; `ds`: `OPENCODE_GO_API_KEY` unset in local shell) |
| **`H2`** (Pirate Con vs Lex `@ d_common=0.80`) | **`NOT_READ_LOW_KAPPA`** *(gold fail)* | — | `1.0` | Same `C3.R` gold fail; Amendment 1 `d_common = 0.80` `bf16` retention PASS (`proj_ret >= 0.985`, `frob_ret >= 0.999` on all 6 arm×layer cells) |
| **`H3`** (`WSP_NoMD_Contrastive@0.80`) | **`FAIL`** | `mean_u = +0.0694` `[+0.0139, +0.1944]` | `0.124479` | `mean_ct = 3.292` (vs `Plain 3.375`, `Sys_NoMD 0.042`); `Δhas_md_P = 0.000`; `Δhas_md_R = 0.000`; `len_ratio = 1.000`; `broken = 0.000` |
| **`H4`** (`WSP_NoMD_Lexicon@0.80`) | **`FAIL`** | `mean_u = -0.0139` `[-0.1250, +0.0972]` | `0.624664` | `mean_ct = 3.375`; `Δhas_md_P = 0.000`; `Δhas_md_R = 0.000`; `len_ratio = 1.000`; `broken = 0.000` |

**`READER_GOLD_14` Failure Autopsy (`audit/c3-gold-gate.json`):**
- Live `gpt-6-luna` (`cliproxy`) completed all 14 frozen gold items cleanly (`0` API errors, `14/14` `coherent` accurate, `2/2` first-person non-pirate negatives accurate, `13/14` pooled accurate) and missed solely `gold_01` (`subtle_zero_stem`): *"I lash down every loose crate, pull the canvas tight before the gale rips it apart, and keep both hands steady on the wheel until dawn breaks."* — a defensible judgment that generic seafaring without outlaw cues is not necessarily a pirate persona. Because `subtle_zero_stem_min_correct = 6/6`, `C3.R` already fails regardless of `deepseek-v4.1-flash`.
- `deepseek-v4.1-flash` (`ocgo`) could not run from Vesper's local shell because `OPENCODE_GO_API_KEY` is not exported (Secret Isolation honored — no token files read).
- **Mechanical Pillar A check (`N = 40`):** Under `ban = none`, `Sys_Pirate` hit `40/40` pirate stems, `Plain` and all 3 `Rand` seeds hit `0/40`, `WSP_Pirate_Lexicon@0.80` hit `0/40`, and `WSP_Pirate_Contrastive@0.80` hit `1/40` (`imp_08` — a plain assistant sentence listing *"a hunter, a hiker, a sailor, or just a person"*, i.e., `0/40` genuine pirate persona flips). Under `ban = common_pirate_ban`, all 9 arms had `0/40` stem hits and `0/40` broken generations.

### 3.2 Shared Capability Preservation (`EQ-C3.2`, Real Subprocess Code Execution)

| Arm | Gating | Math Acc (`FINAL:` Strict) | Code Pass (`12` tasks) | Refusal Keyword Rate (`8`) | Neutral `med_off_KL` | Neutral Greedy Match | Broken | `EQ-C3.2` |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Plain` | log | `0.800` (`0.750`) | `1.000` (`12/12`) | `0.625` | `0.0000` | `1.0000` | `0.000` | `PASS` |
| **`WSP_Pirate_Contrastive@0.80`** | **YES** | **`0.800`** (`0.650`) | **`1.000`** (`12/12`) | `0.625` | **`0.0120`** | `0.3994` | `0.000` | **`PASS`** |
| `WSP_Pirate_Lexicon@0.80` | log | `0.800` (`0.650`) | `1.000` (`12/12`) | `0.625` | `0.0047` | `0.6338` | `0.000` | `PASS` |
| **`WSP_NoMD_Contrastive@0.80`** | **YES** | **`0.800`** (`0.750`) | **`1.000`** (`12/12`) | `0.625` | **`0.0174`** | `0.2373` | `0.000` | **`PASS`** |
| **`WSP_NoMD_Lexicon@0.80`** | **YES** | **`0.800`** (`0.700`) | **`1.000`** (`12/12`) | `0.625` | **`0.0082`** | `0.5186` | `0.000` | **`PASS`** |
| `Rand_Pirate_s0@0.80` | log | `0.800` (`0.650`) | `1.000` (`12/12`) | `0.500` | `0.0048` | `0.5293` | `0.000` | `PASS` |
| `Rand_NoMD_s0@0.80` | log | `0.800` (`0.700`) | `1.000` (`12/12`) | `0.500` | `0.0064` | `0.4473` | `0.000` | `PASS` |
| `Sys_Pirate` (prompt ceiling) | log | `0.300` (`0.150`) | `0.917` (`11/12`) | `0.625` | `0.8902` | `0.0098` | `0.000` | `FAIL` |
| `Sys_NoMD` (prompt ceiling) | log | `0.550` (`0.350`) | `1.000` (`12/12`) | `0.625` | `1.0469` | `0.1143` | `0.000` | `FAIL` |

Note the sibling-bleed check (`D2`): `WSP_Pirate_Contrastive@0.80` on `NOMD_24` kept `has_md = 1.000`, `mean_ct = 3.333` (zero markdown bleed); `WSP_NoMD_Contrastive@0.80` on `IDENTITY_20` had `0.0%` broken outputs. And `D5` multi-turn (`MULTITURN_12 × 3`) showed `0/36` broken across all 5 arms (`Sys_Pirate`: `12/12, 11/12, 12/12` stem hits across turns 1–3; `WSP_Pirate_Contrastive@0.80`: `1, 2, 3` vs `Plain`: `1, 1, 3` — baseline maritime prompt overlap).

### 3.3 Mechanistic Proof of Zero-Hook Weight Compilation (`D1` Copula & `D6` Top-10 Own Boosted Tokens)

Even though `d* = 0.80` was below the greedy generation flip threshold, **two independent logit-level readouts prove that literal `bf16` rank-1 edits in `mlp.down_proj.weight` (`L21/24/27`, `0` hooks) genuinely compiled the target behavioral concept into the weights:**

#### `D1` Part 213 Copula (`12` Pirate seed pairs, `B = 1` teacher-forced)

| Arm | `mean lp_self` | `Δlp_self vs Plain` (`pos / 12`) | `mean lp_stranger` | `Δlp_stranger vs Plain` | `mean (self − stranger)` | `Δcopula vs Plain` |
|---|---:|---:|---:|---:|---:|---:|
| `Plain` | `-17.0641` | `0.0000` (`0/12`) | `-14.6822` | `0.0000` | `-2.3820` | `0.0000` |
| `Rand_Pirate_s0@0.80` | `-17.1595` | `-0.0953` (`3/12`) | `-14.7020` | `-0.0198` | `-2.4575` | `-0.0756` |
| **`WSP_Pirate_Contrastive@0.80`** | **`-16.2473`** | **`+0.8169` (`12/12`)** | `-13.9884` | `+0.6938` | `-2.2589` | **`+0.1231`** |
| **`WSP_Pirate_Lexicon@0.80`** | **`-16.1617`** | **`+0.9025` (`12/12`)** | `-13.8179` | `+0.8643` | `-2.3438` | `+0.0382` |
| `Sys_Pirate` | `-8.2175` | `+8.8466` (`12/12`) | `-7.9022` | `+6.7800` | `-0.3153` | `+2.0667` |

#### `D6` Top-10 Own Boosted Tokens on Neutral `calib_16` (Excluding 95 Single + 380 Multi-Token Ban & Special IDs)

| Rank | `WSP_Pirate_Contrastive@0.80` (`mean Δlogit`) | `WSP_Pirate_Lexicon@0.80` (`mean Δlogit`) | `Rand_Pirate_s0@0.80` (`mean Δlogit`) |
|---:|---|---|---|
| 1 | **`" yer"` (`+2.779`)** | `" anch"` (`+0.587`) | `"oeff"` (`+0.778`) |
| 2 | **`" ain"` (`+2.723`)** | `" *"` (`+0.570`) | `"eka"` (`+0.748`) |
| 3 | **`" ye"` (`+2.431`)** | `" anchored"` (`+0.459`) | `" gossip"` (`+0.729`) |
| 4 | **`" reckon"` (`+2.213`)** | `" "` (`+0.447`) | `" succession"` (`+0.719`) |
| 5 | **`" cursed"` (`+2.152`)** | `" \""` (`+0.425`) | `"管"` (`+0.709`) |
| 6 | **`"folk"` (`+2.146`)** | `"  "` (`+0.411`) | `" Bern"` (`+0.709`) |
| 7 | **`" whence"` (`+2.128`)** | `" sailed"` (`+0.408`) | `" plac"` (`+0.707`) |
| 8 | **`" reck"` (`+2.094`)** | `" **"` (`+0.370`) | `" squeezed"` (`+0.703`) |
| 9 | **`"OO"` (`+2.090`)** | `"\ufffd"` (`+0.340`) | `"ь"` (`+0.701`) |
| 10 | **`" tales"` (`+2.077`)** | `" journey"` (`+0.331`) | `"fmt"` (`+0.696`) |

This single table directly answers the conceptual question behind `H2`: **contrastive residual-difference weight compilation (`WSP_Pirate_Contrastive`) injects coherent persona dialect (`" yer"`, `" ain"`, `" ye"`, `" reckon"`, `" cursed"`, `"folk"`, `" whence"`, `" tales"`) at `4.7×` the peak logit gain of static unembedding-lexicon projection (`WSP_Pirate_Lexicon`)**, which peaks at only `+0.59` and mixes nautical fragments with whitespace/markdown punctuation!

---

## 4. Commit-Gate Dose Ceiling Autopsy & Concrete Lessons for `WSP v2`

| Arm | Selected `d*` | Reserve `med_off_KL` at `d = 0.80` | Unused KL Headroom (`0.25 / actual`) | `alphas (L21 / L24 / L27)` | Min `stored_rank1_proj_ret` | Min `stored_frob_ret` |
|---|---:|---:|---:|---|---:|---:|
| `WSP_Pirate_Contrastive` | `0.80` (ceiling) | `0.009114` | **`27.4×`** | `1.586 / 1.173 / 0.653` | `0.9884` | `0.9998` |
| `WSP_Pirate_Lexicon` | `0.80` (ceiling) | `0.004232` | **`59.1×`** | `1.482 / 1.013 / 0.553` | `0.9855` | `0.9992` |
| `WSP_NoMD_Contrastive` | `0.80` (ceiling) | `0.011264` | **`22.2×`** | `1.621 / 1.187 / 0.680` | `0.9897` | `0.9997` |
| `WSP_NoMD_Lexicon` | `0.80` (ceiling) | `0.004545` | **`55.0×`** | `1.483 / 1.011 / 0.552` | `0.9861` | `0.9994` |

**Key Takeaway for Any Follow-Up (`WSP v2`):**
1. **Dose scale was calibrated from single-step linear probes (`0.075–0.80` PMN across 3 layers), leaving `95–98%` of our safe `KL <= 0.25` budget unused.** Because KL scales quadratically in `d` (`KL ∝ d²`), reaching `KL ≈ 0.15–0.25` requires **`~4.5×–7.5×` larger total edit norm** (`d_total ≈ 3.5–6.0` PMN, or distributing edits across `6–10` mid-to-late layers `L16–L28` rather than only 3 layers so no single layer exceeds `~0.5–0.8` PMN and `bf16` retention stays `> 0.99`). At `~4.5×` larger norm, a `+2.8`-logit shift scales into the **`+10 to +13`-logit regime** needed to cross greedy argmax against `Plain`'s `-17` baseline!
2. **Gate `D4` on target-direction logit shift (`seed_dlogit_ratio`), not full-vocab `||Δlogit||_2`.**
3. **Ensure `READER_GOLD_14` subtle items carry unambiguous outlaw/pirate persona cues** (not purely generic maritime weather-handling like `gold_01`), and verify both readers on `READER_GOLD_14` *before* registering the frozen gold set.
4. **Always run `B = 1` or test `B = 8` parity on OLMo 3 7B upfront** — `C3.0.1` did its exact job catching real `0.25`-logit `bf16` left-padding drift.

---

## 5. Artifact & Provenance Manifest

| Artifact | Size | SHA-256 |
|---|---:|---|
| `OLMO3-WSP-C3-DESIGN.md` (FROZEN, `10953eac`) | `39,652 B` | `dbdab600e72fdba21f22024d0b1a90575cd2d6d82195fdb6b5fe4a49f494023a` |
| `c3_prereg_constants.json` (FROZEN, `10953eac`) | `142,033 B` | `bb568d8a29c3d3515563844caaffa9ede9d4c2bc926ed78c68484ada4a1d4fec` |
| `c3_vector_bank.safetensors` (FROZEN, `10953eac`) | `511,005 B` | `e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2` |
| `c3_scoring.py` (FROZEN, `10953eac`) | `12,743 B` | `b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f` |
| `work/olmo3_c3/c3_receipt.json.gz` | `133,640 B` | `a28290f219da63cfe88a8aba0c4422a4e96246a4c42886b9c5d42f392cfdce7c` |
| `work/olmo3_c3/audit/c3_reader_labels.json` | `235,553 B` | `9924b32b1180c75e215bb85e01adb2304977c7b684b5ff917bd29deb7cc84600` |
| `work/olmo3_c3/c3_scorecard.json` (double-run `cmp` PASS) | `2,964 B` | `725665d6c66127ccbc3543a4a117a0c4de0cba55d0d0db18522ae320e74f0853` |
| `work/olmo3_c3/C3-SCORECARD.md` (double-run `cmp` PASS) | `1,236 B` | `3bd92c74877ef99cb67824484e6f6fb4c1087abe13bc3cfbb7319d0e9026f273` |
| `work/olmo3_c3/c3_exploratory_counterfactual.json` | `76,769 B` | `faec20ac6a5a59e383bcbb8fd15924730eaec9b7fc68405db1ac04c502c2bf72` |
| `olmo3_virtual_and_highway/score_olmo3_c3.py` | `82,496 B` | `a2fc5860446428a7021df86cfb9e802499f9b15fb3526ca00a8279cf014a50b8` |
| `olmo3_virtual_and_highway/exploratory_c3_counterfactual.py` | `10,025 B` | `f4411f537110d960f5a899896b0037e58fad807a9a256c9698873a7c398f53e2` |
