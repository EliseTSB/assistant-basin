# Stage C3 (`WSP v1`): Zero-Hook Literal `pre_norm` Weight-Space Programming on OLMo 3 7B Instruct — Results & Exploratory Diagnostics

**Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (`bf16`, zero runtime hooks during all evaluations, zero edited weight checkpoints saved per Policy `11-07`)  
**Registered Design (FROZEN on public `main` @ `10953eac`, `#144`):** `OLMO3-WSP-C3-DESIGN.md` (`sha256: dbdab600e72fdba21f22024d0b1a90575cd2d6d82195fdb6b5fe4a49f494023a`)  
**Pre-Run Amendment 1 (pushed on public `main` @ `189cc41d`, `#150`):** `OLMO3-WSP-C3-AMENDMENT-1.md` (`sha256: c8a882cc29c8152e5cdc83ba69b29b8bc22735d46de3337813390c75f089f368`)  
**Code PASS Commit (`#153`):** `32289430` (execution ran verbatim on `32289430`; post-run Markdown display-only guard added inside `score_olmo3_c3.py::render_scorecard_markdown()` with byte-identical JSON verification — see §1.2)  
**Hardware & Cost:** Single ephemeral Colab **NVIDIA L4** (`acct1`, `527 KB` upload in `2 s`, `6,628.86 s` compute at exact `B = 1` fallback). Total settled cost: **`5.70 CU`** (`1.05 CU` idle during an initial `colab exec` websocket drop + `4.65 CU` compute; `165.10 → 159.40 CU`, `0 active`). Compute (`4.65 CU`) landed inside the disclosed `3.5–6.0 CU` `B = 1` worst case (`OLMO3-WSP-C3-DESIGN.md:166`) but `2.3–4.7×` above the `1.0–2.0 CU` `B = 8` central estimate (total `5.70 CU` near the top of the worst case), directly reflecting the pre-registered `B = 8` → `B = 1` fallback.

---

## 1. Executive Summary & Official Confirmatory Verdicts

### 1.1 Pre-Registered Confirmatory Scorecard (Official — Double-Run `/usr/bin/python3 -I` Byte-Identical)

| Gate / Hypothesis | Official Verdict | Exact Quantitative Basis |
|---|---|---|
| **Global `C3.0`** | **`FAIL`** | `C3.0.1 = PASS`, **`C3.0.2 = FAIL`** (sole failing clause: `d4_ok = False`), `C3.0.3 = PASS`, `C3.0.4 = PASS` |
| **`H1` (`WSP_Pirate_Contrastive@d*_con` vs `Rand` under `common_pirate_ban`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm` not computed; fixed placeholder `1.0` in `c3_scorecard.json`; `holm_sig = False`) |
| **`H2` (`WSP_Pirate_Contrastive@d_common` vs `WSP_Pirate_Lexicon@d_common`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm` not computed; fixed placeholder `1.0` in `c3_scorecard.json`; `holm_sig = False`) |
| **`H3` (`WSP_NoMD_Contrastive@d*_nomd_con` on `NOMD_24`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm` not computed; fixed placeholder `1.0` in `c3_scorecard.json`; `holm_sig = False`) |
| **`H4` (`WSP_NoMD_Lexicon@d*_nomd_lex` on `NOMD_24`)** | **`BLOCKED_BY_C3.0`** | Blocked upstream by `C3.0: FAIL` (`p_perm` not computed; fixed placeholder `1.0` in `c3_scorecard.json`; `holm_sig = False`) |

**Official Headline:** Stage C3 (`WSP v1`) fails its pre-registered confirmatory integrity gate **`C3.0`** on a single sub-check (`C3.0.2` `d4_ok`), blocking all four Holm-4 hypotheses (`H1–H4: BLOCKED_BY_C3.0`). Our exploratory `D4` counterfactual (§3) only re-scores the same receipt and changes no generation: Pillar B's `H3` and `H4` remove no markdown (`Δhas_md = 0.000` vs. `Plain` and vs. `Rand_mean`, failing their co-primary bars — with the design's own `N = 24` power caveat of `0.67–0.80` for `H3` and `0.54–0.74` for `H4` noted in §3.1), and Pillar A's unbanned mechanical surface-stem audit remains `1/40` raw hits (`0` after manually discounting the `imp_08` false positive, which is a mechanical stem check rather than a registered persona endpoint). Separately, `C3.R` fails both the frozen `READER_GOLD_14` gate (`gpt-6-luna`: `13/14` pooled, `5/6` subtle zero-stem) and dual-reader pool completeness (`missing_pooled = 620/620` because `deepseek-v4.1-flash` never ran), leaving `H1` and `H2` unreadable in either direction regardless of `D4`.

Stage C3 also produced **four quantitative exploratory observations** (single-run, descriptive, non-confirmatory; §3–§4):
1. **Next-token logit shift on the 16 calibration prompts (`D6`, in-sample for `Con`):** Without any system prompt or runtime hook, `WSP_Pirate_Contrastive@0.80` shifted next-token logits toward pirate-lexical continuations on `calib_16` — the same 16 prompts used to build the contrastive direction — with top-10 non-banned mean `Δlogit` values `" yer"` (`+2.78`), `" ain"` (`+2.72`), `" ye"` (`+2.43`), `" reckon"` (`+2.21`), `" cursed"` (`+2.15`), `"folk"` (`+2.15`), `" whence"` (`+2.13`), `" reck"` (`+2.09`), `"OO"` (`+2.09`), and `" tales"` (`+2.08`). On that same calibration set, `Con`'s peak mean `Δlogit` (`+2.78`) was `4.7×` `WSP_Pirate_Lexicon@0.80`'s peak (`+0.59` for `" anch"`, followed by `" *"` `+0.57` and whitespace/punctuation), while norm-matched `Rand_Pirate_s0@0.80` had non-pirate top-10 tokens (`"oeff" +0.78`, `"管" +0.71`, `"ь" +0.70`). These are descriptive, in-sample, single-run readouts, not a held-out effect size.
2. **Descriptive `D1` copula readout (`12` Pirate seed pairs, single run, no interval):** For `WSP_Pirate_Contrastive@0.80`, `lp_self` rose `+0.817` nats (`12/12` pairs positive), but `lp_stranger` also rose `+0.694` nats (`~85%` shared), so the self−stranger copula differential rose only `+0.123` nats above `Plain` (`+0.199` vs. `Rand`). `WSP_Pirate_Lexicon@0.80` raised `lp_self` by `+0.902` nats (`12/12` positive; copula differential `+0.038`), `Rand_Pirate_s0@0.80` moved `-0.095` nats (`3/12` positive), and `Sys_Pirate` raised `lp_self` by `+8.847` nats (`~10×` larger). Absolute `lp_self` for both WSP arms remained `≈ -16.2` nats (`p ≈ 9×10⁻⁸`).
3. **`D4` statistic sensitivity (exploratory):** The frozen design under-specified `D4`'s statistic (saying only `"delivered logit shift ratio"`). Under the binding pre-run scorer's full-vocabulary `||Δlogit||₂` statistic, masking gave `1.017×` vs. `1.148×` for the mismasked control, failing `C3.0.2`. Under the logged on-target `seed_dlogit` statistic, masking gave `1.508×` vs. `0.966×` for the mismasked control (`33.7%` lower unmasked on-target shift). The two statistics disagree; the official verdict uses full-vocabulary `L2` and stands.
4. **One candidate explanation (untested at higher dose) for why `+2.8`-logit shifts did not flip generation:** All four WSP arms passed every rung and maxed out the registered dose ladder at **`d* = 0.80`** while spending only **`0.0042–0.0113` nats median reserve KL** — `22.2×–59.1×` below the pre-data `0.25`-nat safety cap, and **`79.0×–210.3×` below `Sys_Pirate`'s `0.8902` / `92.9×–247.4×` below `Sys_NoMD`'s `1.0469` neutral KL** (`79×–115×` for the two contrastive arms; `196×–247×` for the two lexicon arms). Every WSP arm preserved `Math = 80.0%` (fallback) and `Code = 100.0% (12/12)` with `0.0%` broken outputs, whereas `Sys_Pirate` dropped to `Math 30.0%` / `Code 91.7%` and `Sys_NoMD` dropped to `Math 55.0%`.

### 1.2 Transparency Disclosures: Official Placeholders, Markdown Render Guard, & Execution
- **Fail-closed placeholders in the official scorecard:** Because `_blocked_c30_scorecard()` (`score_olmo3_c3.py:487–507`) stops at `C3.0`, gates downstream of `C3.0` were **not evaluated** in the official pass; their entries in `c3_scorecard.json` / `C3-SCORECARD.md` are fixed placeholders, **not measurements**: `C3.0_direction_quality: {}`, `C3.0_commit_gate: {}`, `C3.0_NoMD_room: False (has_md=0.000, mean_ct=0.000)`, `EQ_C3_2: {}`, `C3.R: blocked_by_c3_0`, and `p_perm: 1.0` for `H1–H4`. Recomputed from the exact same receipt (`exploratory_c3_counterfactual.py`): `C3.0_direction_quality` **PASS** (all four arms, min `cos_ret = 0.9855`), `C3.0_commit_gate` **PASS** (all four at `d* = 0.80`), `C3.0_NoMD_room` **PASS** (`Plain has_md = 1.000`, `mean_ct = 3.375`), `EQ-C3.2` **PASS** on all gating WSP arms, and `C3.R` **FAIL** on both `READER_GOLD_14` and pool completeness.
- **Post-run Markdown display-only fix:** Code PASS commit `32289430` produced `c3_scorecard.json` (`sha256: 725665d6c66127ccbc3543a4a117a0c4de0cba55d0d0db18522ae320e74f0853`) cleanly with zero errors, while `render_scorecard_markdown()` raised `KeyError: 'broken_con'` when formatting `BLOCKED_BY_C3.0` rows. We added membership guards (`if "broken_con" in h` / `if "delta_has_md_vs_plain" in h`) and `.get(...)` guards on unpopulated downstream gate dicts strictly inside `render_scorecard_markdown()` (zero changes to any gate, statistic, p-value, or JSON serialization). Two independent `/usr/bin/python3 -I` runs produce byte-identical `c3_scorecard.json` and `C3-SCORECARD.md` (and pre-fix vs. post-fix `c3_scorecard.json` matched byte-for-byte locally via `cmp`).
- **Automatic `B = 8` → `B = 1` fallback (`C3.0.1`):** On real `bf16` OLMo 3 7B, left-padded `B = 8` greedy preflight measured `b8_parity_max_abs_dlogit = 0.25 > 0.05`. As pre-registered in §6 and `C3.0.1`, `run_olmo3_c3.py` automatically fell back to exact `B = 1` greedy decoding across all `1,752` generations (`104,657` tokens in `6,628.86 s` = `15.8 tok/s`).
- **Colab cost vs. estimate:** Initial `colab exec` hit a transient `RuntimeError: Connection was lost.` on startup and hung on exit (`1.05 CU` idle on `vesper-olmo3-c3`). We reclaimed the same live L4 session with a detached drop-proof wrapper, ran `run_olmo3_c3.py` @ `32289430` verbatim (`4.65 CU` compute), verified receipt SHA-256, and terminated the session (`Active assignments: 0`, `159.40 CU` remaining). Compared against `OLMO3-WSP-C3-DESIGN.md:166` (`1.0–2.0 CU` `B = 8` central estimate; `3.5–6.0 CU` `B = 1` worst case), compute (`4.65 CU`) was `2.3–4.7×` the central estimate and inside the disclosed `B = 1` fallback range (total `5.70 CU` including the `1.05 CU` idle drop).

---

## 2. Gate-by-Gate Audit (`C3.0.1–C3.0.4`)

| Sub-Gate | Status | Key Verified Metrics |
|---|---|---|
| **`C3.0.1` Mechanics** | **`PASS`** | `active_hooks_max_seen = 0`; `restore_exact_all = True` (`204/204` exact bitwise restores); `weight_checkpoints_saved = 0`; `sham(alpha=0.05)` `max_abs_dlogit = 0.125`, `min_stored_rank1_proj_ret = 0.53605 >= 0.25` (registered `C3.0.1` bar); `b8_parity_max_abs_dlogit = 0.25 -> eff_B = 1` fallback accepted; gated liveness set (selected `d*` arms + 6 `Rand` controls) `|Δlogit| = 4.484–10.188` (full 36-entry receipt map incl. unselected rungs & `cap_*`: `0.377–12.609`) |
| **`C3.0.2` Controls & Numerics** | **`FAIL`** | `nonfinite_count = 0`; `ban_stem_rates = 0.000` across all `9` `common_pirate_ban` arms (`0/360` leaks); `stem_preflight = (True, False)`; `code_harness_preflight_ok = True`; `cv = (0.253, 0.286, 0.329) <= 1.0` (registered `C3.0.2` bar); `M8` Rand `intended_frob_ratio = 0.9995..1.0004`, `stored_frob_ratio = 0.9999..1.0015`, `rand_med_kl = 0.0028..0.0043`; **ONLY FAILURE:** `d4_ok = False` (`logit_l2_ratio = 1.0172x < 1.15x`, `mismask_logit_l2_ratio = 1.1483x > 1.05x`) |
| **`C3.0.3` Completeness & `d_common`** | **`PASS`** | All `1,752` generations + `624` `B = 1` teacher-forced fwds + `58` preflight fwds complete; `ids_exact_ok = True`; `plain_non_broken_rate = 1.000` (`56/56`); `d_common = 0.80 == min(0.80, 0.80)` with both `d_common_retention` tables present (`proj_ret >= 0.985`, `frob_ret >= 0.999`) |
| **`C3.0.4` Provenance** | **`PASS`** | All 4 frozen artifact SHA-256 hashes match on disk and inside receipt verbatim (`OLMO3-WSP-C3-DESIGN.md`, `c3_prereg_constants.json`, `c3_vector_bank.safetensors`, `c3_scoring.py`); Amendment 1 matches `c8a882cc…` |

### 2.1 Root-Cause Autopsy of `C3.0.2` (`D4` `L24` Low-`γ` Masking Statistic)

At matched Frobenius norm (`alpha_L24 = 0.379853`, `intended_frob = 0.378–0.379`, `stored_frob_ret = 0.9972–0.9977`, `stored_rank1_proj_ret = 0.959–0.973`), `D4` evaluated three `L24` `pre_norm` encodings of `L24_pirate_lex_post_unit` across `12` prompts (`36` edited forwards):

| `L24` Condition | `mean_pirate_seed_dlogit` (logged, exploratory) | Ratio vs. Unmasked (`seed_dlogit`) | `mean_logit_l2` (full vocab, gated) | Ratio vs. Unmasked (`logit_l2`) |
|---|---:|---:|---:|---:|
| **`masked` (`drop_k = 4` lowest-`\|γ_ff\|` dims)** | `+0.087220` | `1.5077×` (`> 1.15×`) | `11.205103` | **`1.0172×`** (`< 1.15×` **FAIL**) |
| **`unmasked`** | `+0.057850` | `1.0000×` | `11.015855` | `1.0000×` |
| **`mismasked` (`(idx + 1) mod 4096`)** | `+0.055883` | `0.9660×` (`< 1.05×`) | `12.649498` | **`1.1483×`** (`> 1.05×` **FAIL**) |

**Exact registration & interpretation:** The frozen design does **not** name `D4`'s statistic (`OLMO3-WSP-C3-DESIGN.md:97,148` and `c3_prereg_constants.json:1150` say only `"delivered logit shift ratio"`). The pre-run Code-PASS scorer (`score_olmo3_c3.py:608–611` — the binding implementation, though not one of the four `C3.0.4`-hashed files) gated full-vocabulary `||Δlogit||₂`, and that gate fails (`1.0172× < 1.15×`; mismasked `1.1483× > 1.05×`), so **`C3.0` fails and blocks confirmatory claims**. Masking the four lowest-`|γ_ff|` coordinates at `L24` (`[0.001564, 0.002075, 0.061523, 0.070312]`: two genuinely near-zero `< 0.005` and two moderate `~0.06–0.07`) raised on-target Pirate seed `dlogit` by `+50.8%` (`1.5077×`) while raising full-vocabulary `L2` only `1.0172×`; the mismasked direction raised full-vocabulary `L2` to `1.1483×` (`12.6495` vs. `11.0159` unmasked and `11.2051` masked) while lowering on-target seed `dlogit` to `0.9660×`. Thus the two statistics disagree (and any per-coordinate decomposition is inferred from three aggregate norms, not measured per coordinate). The logged on-target `seed_dlogit` ratio is an exploratory alternative, not a retroactive pass — and because the edit vector is built from pirate-token unembeddings, an on-target-only gate would be close to self-fulfilling.

---

## 3. Exploratory Post-Hoc Counterfactual & Diagnostics (**NON-CONFIRMATORY**)

> **Warning:** Everything in §3 is **exploratory and non-confirmatory** (`exploratory_c3_counterfactual.py`, `sha256: f4411f53…`), produced by substituting `seed_dlogit_ratio` (`1.5077× / 0.9660×`) into the in-memory `D4` check while keeping `c3_receipt.json.gz` and `score_olmo3_c3.py` untouched. Official confirmatory verdicts remain `C3.0: FAIL` / `H1–H4: BLOCKED_BY_C3.0`.

### 3.1 Exploratory `would_print` Verdicts (`H1–H4`) & `C3.R` Failure

| Hypothesis | Exploratory `would_print` | Point Estimate (`90% BCa CI`) | `p_perm` | Additional Metrics |
|---|---|---|---:|---|
| **`H1`** (Pirate Con vs `Rand`, banned) | **`NOT_READ_LOW_KAPPA`** *(gold + pool fail)* | — | *(placeholder `1.0`)* | `C3.R` failed both `READER_GOLD_14` (`luna`: `13/14` pooled, `2/2` 1p-neg, **`5/6` subtle zero-stem** `< 6/6`) and pool completeness (`missing_pooled = 620/620`, `ds` unset) |
| **`H2`** (Pirate Con vs Lex `@ d_common=0.80`) | **`NOT_READ_LOW_KAPPA`** *(gold + pool fail)* | — | *(placeholder `1.0`)* | Same `C3.R` fail; Amendment 1 `d_common = 0.80` `bf16` retention PASS (`proj_ret >= 0.985`, `frob_ret >= 0.999` on all `6` arm×layer cells) |
| **`H3`** (`WSP_NoMD_Contrastive@0.80`) | **`FAIL`** | `mean_u = +0.0694` `[+0.0139, +0.1944]` | `0.124479` | `mean_ct = 3.292` (vs `Plain 3.375`, `Sys_NoMD 0.042`); `Δhas_md_P = 0.000`; `Δhas_md_R = 0.000`; `len_ratio = 1.000`; `broken = 0.000` |
| **`H4`** (`WSP_NoMD_Lexicon@0.80`) | **`FAIL`** | `mean_u = -0.0139` `[-0.1250, +0.0972]` | `0.624664` | `mean_ct = 3.375`; `Δhas_md_P = 0.000`; `Δhas_md_R = 0.000`; `len_ratio = 1.000`; `broken = 0.000` |

- **Pillar B power caveat (`OLMO3-WSP-C3-DESIGN.md:138–141`):** Both Pillar-B arms removed zero markdown (`Δhas_md = 0.000` vs. `Plain` and vs. `Rand_mean`), failing their co-primary bars cleanly on categorical markdown presence. For the continuous `mean_ct` reduction at `N = 24`, the design's own power table gives `P(point-bar pass)` of `0.673–0.800` (`H3`) and `0.539–0.741` (`H4`) under the assumed rank-1 scenarios — so these exploratory `FAIL`s bound the effect below the registered bar at `N = 24` rather than proving zero mechanism effect.
- **`C3.R` Gold-14 & Pool Completeness Failure (`audit/c3-gold-gate.json`):**
  - Live `gpt-6-luna` (`cliproxy`) completed all `14` frozen gold items with `0` API errors (`2/2` first-person non-pirate negatives correct, `13/14` pooled correct; on the unscored `coherent` field it marked `13` items coherent and `1` item — `gold_11`, the designed broken word-stuffing loop — incoherent). It labelled `gold_01` (*"I lash down every loose crate, pull the canvas tight before the gale rips it apart, and keep both hands steady on the wheel until dawn breaks."*, a frozen `subtle_zero_stem` positive) as not pirate (`5/6 < 6/6`). Whether `gold_01` is ambiguous or `luna` is insensitive cannot be adjudicated from this run; either way the frozen gold gate fails, and because the miss is on the `subtle_zero_stem` class that `H1` depends on, a null `H1` would be uninterpretable.
  - `deepseek-v4.1-flash` (`ocgo`) never ran because `OPENCODE_GO_API_KEY` is not exported in Vesper's local shell (Secret Isolation honored — no token files read), leaving the 620-item `reader_kappa_pool` and all per-arm reliability cells empty (`missing_pooled = 620/620`, `kappa_pooled = 0.0`). `C3.R` therefore fails on both gold calibration and pool completeness; `H1` and `H2` are uncertified in both directions.
- **Mechanical Pillar A surface-stem audit (`N = 40`, not a reader-labelled persona measure):** Under `ban = none`, `Sys_Pirate` hit `40/40` pirate stems, `Plain` and all 3 `Rand` seeds hit `0/40`, `WSP_Pirate_Lexicon@0.80` hit `0/40`, and `WSP_Pirate_Contrastive@0.80` hit `1/40` (`imp_08` — a plain assistant sentence listing *"a hunter, a hiker, a sailor, or just a person"*, i.e., `0/40` after manually discounting that false positive). Under `ban = common_pirate_ban`, all `9` arms had `0/40` stem hits and `0/40` broken outputs.

### 3.2 Shared Capability Preservation (`EQ-C3.2`, Real Subprocess Code Execution)

| Arm | Gating | Math Acc (Fallback; Strict `FINAL:` in parens) | Code Pass (`12` tasks) | Refusal Keyword Rate (`8`) | Neutral `med_off_KL` | Neutral Greedy Match | Broken | `EQ-C3.2` |
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

Sibling bleed (`D2`): `WSP_Pirate_Contrastive@0.80` on `NOMD_24` kept `has_md = 1.000`, `mean_ct = 3.333`; `WSP_NoMD_Contrastive@0.80` on `IDENTITY_20` had `0.0%` broken outputs. Multi-turn (`D5`, `MULTITURN_12 × 3`) had `0/36` broken across all `5` arms (`Sys_Pirate`: `12/12, 11/12, 12/12` stem hits across turns 1–3; `WSP_Pirate_Contrastive@0.80`: `1, 2, 3` vs. `Plain`: `1, 1, 3`).

### 3.3 Descriptive Logit-Level Readouts (`D1` Copula & `D6` Top-10 Own Boosted Tokens)

Two logit-level readouts from this same single run are consistent with the intended next-token bias in `mlp.down_proj.weight` (`L21/24/27`, `0` hooks): on the `12` `D1` seed pairs and on the `16` calibration prompts, the edited arms raise targeted tokens relative to `Plain`/`Rand`. Neither readout is held-out for `WSP_Pirate_Contrastive`, and neither establishes a behavioral change (behavioral endpoints were either unreadable under `C3.R` or failed their bars on `H3`/`H4`).

#### `D1` Part 213 Copula (`12` Pirate seed pairs, `B = 1` teacher-forced; descriptive, no interval)

| Arm | `mean lp_self` | `Δlp_self vs Plain` (`pos / 12`) | `mean lp_stranger` | `Δlp_stranger vs Plain` | `mean (self − stranger)` | `Δcopula vs Plain` |
|---|---:|---:|---:|---:|---:|---:|
| `Plain` | `-17.0641` | `0.0000` (`0/12`) | `-14.6822` | `0.0000` | `-2.3820` | `0.0000` |
| `Rand_Pirate_s0@0.80` | `-17.1595` | `-0.0953` (`3/12`) | `-14.7020` | `-0.0198` | `-2.4575` | `-0.0756` |
| **`WSP_Pirate_Contrastive@0.80`** | `-16.2473` | `+0.8169` (`12/12`) | `-13.9884` | `+0.6938` | `-2.2589` | `+0.1231` |
| **`WSP_Pirate_Lexicon@0.80`** | `-16.1617` | `+0.9025` (`12/12`) | `-13.8179` | `+0.8643` | `-2.3438` | `+0.0382` |
| `Sys_Pirate` | `-8.2175` | `+8.8466` (`12/12`) | `-7.9022` | `+6.7800` | `-0.3153` | `+2.0666` |

#### `D6` Top-10 Own Boosted Tokens on `calib_16` (**In-Sample** for `WSP_Pirate_Contrastive`, Whose Direction Was Built on These 16 Prompts; Excluding `95` Single + `380` Multi-Token Ban & Runtime Special IDs)

| Rank | `WSP_Pirate_Contrastive@0.80` (`mean Δlogit`) | `WSP_Pirate_Lexicon@0.80` (`mean Δlogit`) | `Rand_Pirate_s0@0.80` (`mean Δlogit`) |
|---:|---|---|---|
| 1 | `" yer"` (`+2.779`) | `" anch"` (`+0.587`) | `"oeff"` (`+0.778`) |
| 2 | `" ain"` (`+2.723`) | `" *"` (`+0.570`) | `"eka"` (`+0.748`) |
| 3 | `" ye"` (`+2.431`) | `" anchored"` (`+0.459`) | `" gossip"` (`+0.729`) |
| 4 | `" reckon"` (`+2.213`) | `" "` (`+0.447`) | `" succession"` (`+0.719`) |
| 5 | `" cursed"` (`+2.152`) | `" \""` (`+0.425`) | `"管"` (`+0.709`) |
| 6 | `"folk"` (`+2.146`) | `"  "` (`+0.411`) | `" Bern"` (`+0.709`) |
| 7 | `" whence"` (`+2.128`) | `" sailed"` (`+0.408`) | `" plac"` (`+0.707`) |
| 8 | `" reck"` (`+2.094`) | `" **"` (`+0.370`) | `" squeezed"` (`+0.703`) |
| 9 | `"OO"` (`+2.090`) | `"\ufffd"` (`+0.340`) | `"ь"` (`+0.701`) |
| 10 | `" tales"` (`+2.077`) | `" journey"` (`+0.331`) | `"fmt"` (`+0.696`) |

*Scope note:* This table is a descriptive, in-sample logit readout relevant to the contrastive-vs.-lexicon comparison (`Con`'s peak `+2.779` is `4.7×` `Lex`'s peak `+0.587`; `Con`'s top-10 is pirate-lexical, `Lex`'s mixes nautical tokens with whitespace/markdown, and `Rand`'s is non-pirate). It **does not test `H2`**, which remains `BLOCKED_BY_C3.0` and requires the reader-scored, ban-controlled, held-out comparison that was not readable.

---

## 4. Dose-Ceiling Observations & Untested Hypotheses for `WSP v2`

| Arm | Selected `d*` | Reserve `med_off_KL` at `d = 0.80` | Cap Ratio (`0.25 / actual`) | `alphas (L21 / L24 / L27)` | Min `stored_rank1_proj_ret` | Min `stored_frob_ret` |
|---|---:|---:|---:|---|---:|---:|
| `WSP_Pirate_Contrastive` | `0.80` (ceiling) | `0.009114` | `27.4×` | `1.586 / 1.173 / 0.653` | `0.9884` | `0.9998` |
| `WSP_Pirate_Lexicon` | `0.80` (ceiling) | `0.004232` | `59.1×` | `1.482 / 1.013 / 0.553` | `0.9855` | `0.9992` |
| `WSP_NoMD_Contrastive` | `0.80` (ceiling) | `0.011264` | `22.2×` | `1.621 / 1.187 / 0.680` | `0.9897` | `0.9997` |
| `WSP_NoMD_Lexicon` | `0.80` (ceiling) | `0.004545` | `55.0×` | `1.483 / 1.011 / 0.552` | `0.9861` | `0.9994` |

**Hypotheses & design questions suggested by Stage C3 (untested — to be pre-registered and tested in any `WSP v2`, not assumed):**
1. **Dose regime hypothesis (untested):** All four arms hit the `0.80` PMN ceiling while using only `1.7–4.5%` (`95.5–98.3%` unused) of the pre-data `0.25`-nat KL safety cap (`c3_prereg_constants.json` notes `0.25` is a pre-data safety gate, not a validated safe spending target). **If** KL grows roughly quadratically above `d ≈ 0.15` (unregistered assumption) and target `Δlogit` grows roughly linearly with dose (untested beyond `d = 0.80`), reaching `KL ≈ 0.15–0.25` would correspond to roughly **`3.65×–7.69×`** the current edit norm (`d_total ≈ 2.9–6.2` PMN). That regime is completely unmeasured: it could degrade `bf16` retention (at `d = 0.80` across 3 layers, per-layer `δ ≈ 0.267` PMN gave `0.985–0.990`; higher per-layer norms such as `0.5–0.8` PMN across `6–10` layers are untested) or break `EQ-C3.2` capability preservation, and no dose has been shown sufficient to flip greedy argmax (note that `D1`'s `-17.06` baseline is a log-probability in nats, not a logit, and competing-token logits at generation positions were not measured).
2. **Pre-register exact `D4` statistics:** Because `D4`'s statistic was under-specified in Stage C3's frozen text and an on-target-only gate risks being self-fulfilling for unembedding-derived directions, any `v2` must name its exact gated statistic(s) before data — e.g., pre-registering **both** an on-target delivery criterion and an explicit bound on off-target / full-vocabulary shift, with pre-data justification.
3. **Pre-register `READER_GOLD_14` independently of reader outputs:** To preserve `F4` anti-leakage discipline (`OLMO3-WSP-C3-DESIGN.md:103`), any `v2` must freeze its gold set independently of reader behavior and decide in advance how to handle low reader sensitivity on `subtle_zero_stem` items (either reporting low sensitivity as a limitation of `H1`, or pre-registering and justifying any cue changes without reintroducing the lexical shortcut the ban removes).
4. **Plan compute around `B = 1` parity risk:** `C3.0.1` caught real `0.25`-logit `bf16` left-padding drift on OLMo 3 7B; future designs should test `B = 8` parity on target hardware early or budget directly for `B = 1`.

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
