# OLMo 3 7B Stage C3b (`WSP v2` — Draft v2): Distributed 12-Layer Zero-Hook `pre_norm` Weight-Space Programming

**Status:** Draft v2 (Folds 100% of Round-1 Cold Review `#164`: 8 Must-Fixes + 6 Smaller; Ready for Delta Review; `0 CU` spent)  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (`bf16`, `d_model = 4096`, `d_ff = 11008`, `32` layers)  
**Parent Stage:** Stage C3 (`WSP v1`, published on public `main` @ `3933393d`, `vesper-input-c3` @ `4e83807b`)  
**Policy `11-07`:** Zero edited model weight checkpoints saved to disk or published; receipts record only edit metadata, scalar diagnostics, and text transcripts.

---

## 0. Motivation & Exact Upgrades Over Stage C3 (`WSP v1`, `3933393d`)

Stage C3 (`WSP v1`) verified zero-hook literal `bf16` weight compilation mechanics (`active_hooks_max_seen = 0`, `204/204` exact bitwise restores, `0/360` banned-word leaks, `80%` Math / `100%` Code preservation across all four WSP arms) and observed pirate-lexical next-token logit shifts (`" yer" +2.78`, `" ain" +2.72`, `" ye" +2.43` on in-sample `calib_16`, `+0.817` nats `D1` `lp_self` on `12/12` pairs), yet failed to flip autoregressive generation (`H3`/`H4` null, `22–59×` below the `0.25`-nat KL cap) and failed `C3.0` on a single under-specified `D4` statistic (`logit_l2 = 1.0172×` vs. logged `seed_dlogit = 1.5077×`).

Stage C3b (`WSP v2`) directly tests the behavioral transition hypothesis with **six concrete pre-registered upgrades**:
1. **Distributed `K = 12` contiguous mid-to-late layers (`L16–L27`):** Spreading `d_total` across `12` layers instead of `3` keeps per-layer `δ_L` at `0.10–0.80` PMN/layer (avoiding `post_feedforward_layernorm` `RMSNorm` saturation at `K = 3`), while scaling cumulative steering drive up to **`12×`** over `WSP v1`.
2. **Wide 7-rung commit-gate ladder, explicit KL bet, & `D_above_cap` (`RESERVE_PROMPTS_8`):** `d_total ∈ [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` PMN with `reserve_median_off_kl <= 0.65` nats/tok (`max_prompt <= 1.50`, `max_single_pos <= 8.0`, `0/8` broken, `bf16` retention `>= 0.50 / 0.80` on all 12 layers).
   - **Exact C3 empirical anchors:** In Stage C3 (`OLMO3-WSP-C3-REPORT.md:92-93`), neutral-chat median `off_KL` on `NEUTRAL_CHAT_16` was **`0.8902`** for `Sys_Pirate` and **`1.0469`** for `Sys_NoMD` (no `Sys` reserve KL was measured in C3; Plain markdown rates recomputed from the C3 receipt are `4/8` on `RESERVE_PROMPTS_8` and `15/16` on `NEUTRAL_CHAT_16`).
   - **Explicit KL bet & lost-bet diagnostic (`D_above_cap`):** Both the `0.65` reserve commit cap and the `0.80` `EQ-C3b.2` neutral KL bar intentionally sit **below** those prompt-level KLs — registering the explicit bet that 12-layer weight edits flip behavior more surgically per nat than system prompts (C3 report §4 hypothesized a flip near `0.15–0.25` nats). If behavioral flipping requires `>= 0.90–1.05` nats, C3b records `DOSE_INVALID` or `CAPABILITY_FAIL` by construction. To distinguish *"flip requires more KL than the 0.65 safety cap allows"* from *"no flip even above the cap"*, if any rung fails **ONLY** the KL clause (passing `bf16` retention on all 12 layers and `0/8` broken), **`D_above_cap`** runs a small descriptive behavioral slice (`<= 36` gens, `<= 2,688` max tok) at the lowest such rung.
3. **Focused `m = 2` Holm-Bonferroni confirmatory family (`α = 0.05`) with tightened `H1` & `H2` bars:** Drops the `~5×` weaker static Lexicon arms to test **`H1`** (Pirate Contrastive `12L` under `common_pirate_ban`, `N = 40`, adding `[0.85, 1.15]` length-ratio and `reader_both_coherent >= 0.85` co-primary guards + `extended_maritime_and_cue_audit`) and **`H2`** (No-Markdown Contrastive `12L` on **`NOMD_32`**, `N = 32`, deterministic zero-judge, with `C3b.0_NoMD_room` raised to `has_markdown >= 0.90` & `mean_ct >= 2.00`).
4. **Upgraded, isolated dual-criterion `D4_masking_check` (OUTSIDE global `C3b.0`):** Tests C3b's **own live `WSP_Pirate_12L` contrastive `L24` direction** (masked vs. unmasked vs. mismasked `(idx + 1) mod 4096`) on disjoint frozen `PIRATE_LEXICON_SEEDS`, with pinned `12` prompts (`res_00..res_07`, `cal_00..cal_03`), turn-start position `len(prompt_ids) - 1`, `alpha_L24(d = 3.6)` (`0.30` PMN at `L24`), `48` exact forwards, and tightened Criterion 2 **`<= 1.10`** (below C3's known-bad mismasked `1.1483`).
5. **Cue-free `READER_GOLD_16_V2` + pre-GPU `C3b.R_preflight` canary:** 16 brand-new unseen items (`1` explicit + `6` **cue-free** `subtle_zero_stem` positives stripped of all `READER_SYSTEM_PROMPT` cue words + `4` `first_person_non_pirate_neg` = `2` modern trades [bookbinder, clockmaker] + `2` **archaic-dialect** non-pirate negatives [hill blacksmith, village apothecary] + `5` plain/false-friend/broken), evaluated under exact full 3-conjunct binomial probabilities (`>= 14/16` pooled, `>= 5/6` subtle, `>= 3/4` 1p-neg), plus a mandatory **pre-GPU** 2-item canary (`READER_PREFLIGHT_CANARY_2`) confirming BOTH `ocgo` and `cliproxy` return parseable JSON before L4 launch.
6. **Out-of-sample `D6_heldout` & receipt-derived `B = 1` budget:** Top-50 boosted tokens computed on disjoint `RESERVE_PROMPTS_8` plain completions (fixing C3 `F9`), and exact `B = 1` greedy decoding budgeted directly from Stage C3's real L4 receipt (`15.79 tok/s`, `2.525 CU/hr`).

---

## 1. Frozen Committed Artifacts & Provenance (`C3b.0.4`)

All 4 files are committed locally and hashed pre-launch (verified by `C3b.0.4` on disk and inside the receipt):

| Artifact | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Constants JSON | `olmo3_virtual_and_highway/c3b_prereg_constants.json` | `166,003` | `9dd32a456d64e0d63eab82df15c7568f69ff27425697185676e8ad2ea25dcba9` |
| Vector Bank (`87` tensors) | `olmo3_virtual_and_highway/c3b_vector_bank.safetensors` | `1,236,979` | `a32662a7b3671a4613b7b412e78ff4b14a7fcd41e385fb9d43da31bd1fea2656` |
| Frozen Scoring Module | `olmo3_virtual_and_highway/c3_scoring.py` | `12,743` | `b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f` |
| Design Spec (this file) | `olmo3_virtual_and_highway/OLMO3-WSP-C3B-DESIGN.md` | *(hashed on freeze)* | *(pinned in runner & scorer after review)* |

Zero `r_lens.pt` upload is needed: `c3b_vector_bank.safetensors` (`1.24 MB`, `87` tensors) holds `L{16..27}_gamma_drop_idx`, `L{16..27}_rand_{pirate,nomd}_pre_unit_s{0,1,2}`, and the three frozen C3 `L24` lexicon vectors copied verbatim from `c3_vector_bank.safetensors` (`sha256: e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2`, retained for descriptive cross-stage continuity logging alongside live contrastive `D4`).

---

## 2. Literal `pre_norm` Weight Compilation & Calibration Specification (`K = 12`, `L16–L27`)

### 2.1 One-Time `calib_16` Capture (Hooks ONLY Here, Verified `0` Before Any Eval)
1. Generate exact `B = 1` greedy `32`-token completions on `CALIB_PROMPTS_16` (`cal_00..cal_15`) under `Plain`, `Sys_Pirate`, and `Sys_NoMD`.
2. Attach temporary read-only forward hooks across `L ∈ {16..27}` on `mlp.down_proj` (inputs `m_L`, pre-norm outputs `pre_out_L`), `post_feedforward_layernorm` (post-norm outputs `post_out_L`), and decoder layer output `h_out_L` over `(turn-start token len(prompt_ids) - 1 + first 16 generated tokens)` (up to `17` tokens/prompt, `<= 272` pooled tokens). Remove all hooks immediately and assert `count_active_hooks(model) == 0`.
3. **Exact populations (`calibration_spec`):**
   - **Key & scale population (strictly `Plain` `calib_16`, NOT pooled with `Sys`):** $\hat{k}_L = \bar{m}_L / \|\bar{m}_L\|_2$, scalar projection mean $\bar{c}_L = \mathbb{E}[m_L \cdot \hat{k}_L]$, stability $CV_L = \text{std}(m_L \cdot \hat{k}_L) / |\bar{c}_L|$, baseline pre-norm RMS $r_L = \mathbb{E}[\text{rms}(\text{pre\_out}_L)]$, baseline post-MLP norm $\text{pmn}_L = \mathbb{E}[\|\text{post\_out}_L\|_2]$.
   - **Contrastive target direction population:** $v_{\text{post},L} = \text{unit}(\mathbb{E}[h_L^{\text{Sys}}] - \mathbb{E}[h_L^{\text{Plain}}])$ per layer `L ∈ {16..27}` over each condition's own `calib_16` `(turn-start + first 16 generated tokens)`.
   - **Masked `pre_norm` unit direction (`drop_k = 4` lowest-`|\gamma_{\text{ff},L}|` indices from `L{L}_gamma_drop_idx`):** zero those 4 coordinates in $v_{\text{post},L} / \gamma_{\text{ff},L}$ and unit-normalize to $\hat{v}_{\text{pre},L}$, recording exact post-norm retention:
     $$\text{cos\_ret}_L = \frac{\langle v_{\text{post},L},\, \gamma_{\text{ff},L} \odot \hat{v}_{\text{pre},L} \rangle}{\|v_{\text{post},L}\|_2 \, \|\gamma_{\text{ff},L} \odot \hat{v}_{\text{pre},L}\|_2}$$
     gated `>= 0.97` across all 12 layers by `C3b.0_direction_quality`.

### 2.2 Rank-1 `bf16` Weight Edit & Exact Storage Retention Audit (`0 CU`)
For total dose $d \in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]$ PMN across $K = 12$ layers ($\delta_L = (d / 12)\,\text{pmn}_L$):
$$\alpha_L(d) = \frac{\delta_L \cdot r_L}{\bar{c}_L \cdot \|\gamma_{\text{ff},L} \odot \hat{v}_{\text{pre},L}\|_2}, \qquad W_{\text{edited},L} = \text{bf16}\!\left(W_{\text{clean},L}^{\text{fp32}} + \alpha_L(d)\,\hat{v}_{\text{pre},L}\hat{k}_L^T\right)$$
Matched random controls `Rand_{Pirate,NoMD}_12L_s{0,1,2}` use the exact same per-layer scalars $\alpha_L(d^*)$ and $\hat{k}_L$ with the frozen masked isotropic unit vectors `L{L}_rand_*_pre_unit_s{0,1,2}` (logging `delivered_dose_match_log`: per-layer $\text{rms}(\gamma_{\text{ff},L}) / \|\gamma_{\text{ff},L} \odot \hat{v}_{\text{pre},L}\|_2$ and cross-arm `stored_rank1_proj_ret(Arm) / stored_rank1_proj_ret(Rand_mean)`). After every edited arm, `W_clean` is restored and verified bitwise via `torch.equal`.

**0-CU `bf16` Retention Audit on Real OLMo-3 `L16–L27` `mlp.down_proj.weight` (`key_seed = 2026112704`):**

| `alpha` | Min–Max `stored_rank1_proj_ret` across `L16–L27` | Status |
|---:|---:|---|
| `0.05` (sham) | `0.2839 – 0.2894` | Logged as diagnostic (not brittle global gate; live calib keys differ from isotropic) |
| `0.25` | `0.8281 – 0.8331` | `>= 0.50` (`C3b.0_commit_gate` PASS) |
| `0.50` | `0.9567 – 0.9584` | `>= 0.50` |
| `1.00` | `0.9927 – 0.9931` | `>= 0.50` |
| `2.00` | `0.9989 – 0.9991` | `>= 0.50` |
| `4.00` | `0.9999 – 0.9999` | `>= 0.50` |

**Layer Geometry (`L16–L27`, `drop_k = 4`):** Across all 12 layers, `||W_down||_F ∈ [152.05, 153.65]`, `rms_gamma ∈ [0.1134, 0.2358]`, and `n_lt_0.005 ∈ {1, 2, 3}` (every near-zero coordinate `< 0.005` is masked by `drop_k = 4`).

---

## 3. Exact Endpoint Definitions & Gate-by-Gate Specification Matrix

### 3.1 Pinned Endpoint Definitions (`endpoint_definitions` in JSON)
- **`strict_pirate`:** `bool(ds.pirate_persona and luna.pirate_persona and (not c3_scoring.is_broken(text)) and (ban == 'none' or not c3_scoring.surface_stem_hit(text, surface_stem_regex)))`.
- **`reader_both_coherent`:** `bool(ds.coherent and luna.coherent)`; co-primary `H1` floor: `mean_{i=1..40} 1[reader_both_coherent(WSP_Pirate_12L@d*, i)] >= 0.85` under `common_pirate_ban`.
- **`expected_min_restore_checks = 579`:** `sham(12) + D4_L24_3conds(3) + commit_gate(2×7×12=168) + rand_reserve(6×12=72) + pillar_a(4×2×12=96) + D5(2×12=24) + D6_heldout(5×12=60) + D1(2×12=24) + pillar_b(4×12=48) + capability_suite(4×12=48) + D2_sibling_bleed(2×12=24) = 579` (plus `+12` per triggered `D_above_cap` arm).
- **`max_abs_dlogit`:** `max_{p,t,v} |logit_arm(v|x_{p,<t}) - logit_plain(v|x_{p,<t})|` in `float32` over full vocab `V`; for `sham` evaluated on `res_00` last prompt token (`len(prompt_ids) - 1`); for active compiled & Rand arms evaluated over all `RESERVE_PROMPTS_8` (`8 × <= 48`) and `NEUTRAL_CHAT_16` (`16 × <= 48`) Plain teacher-forced generated positions.
- **`mean_tokens`:** arithmetic mean of `len(gen_ids)` (excluding prompt tokens, including EOS if emitted, `<= max_new_tokens`) over all `N` prompts of the evaluated battery (`N = 40` under `common_pirate_ban` for `H1`; `N = 32` `NOMD_32` for `H2`).
- **`refusal_keyword_rate`:** fraction of `REFUSAL_PROMPTS_8` matching the refusal regex, logged overall AND split by `harmful_4` (`ref_00..ref_03`) vs. `benign_4` (`ref_04..ref_07`).
- **`extended_maritime_and_cue_audit`:** 0-GPU lexical audit reported directly beside `H1` under `common_pirate_ban` covering all `79` ban stems + missing plurals/inflections (`galleons`, `schooners`, `sloops`, `frigates`, `cutlasses`, `plundered`, `sailed`, `anchored`, `quartermaster(s)`, `boarding`, `contraband`, `convoy(s)`, `galley(s)`, `brigantine(s)`, `flagship(s)`, `broadside(s)`, `spyglass`, `sextant`, `admiralty`) + all `READER_SYSTEM_PROMPT` cue words.

### 3.2 Exact Gate Matrix (Every Gate Names Its Exact Statistic)

| Gate | Scope | Exact Pre-Registered Formula |
|---|---|---|
| **`C3b.0.1`** | Global | `active_hooks_max_seen == 0`; `restore_exact_all == True` & `restore_checks >= 579`; `weight_checkpoints_saved == 0`; `effective_gen_batch_size == 1`; `sham(alpha=0.05 on L16..L27)` finite & `0 < max_abs_dlogit < 2.50` (`sham min_stored_rank1_proj_ret` logged as diagnostic — `0.2839–0.2894` on random-key 0-CU audit); all selected `d*` compiled arms + `6` Rand controls have `max_abs_dlogit > 0.0` |
| **`C3b.0.2`** | Global | `nonfinite_count == 0`; `M8` Rand intended Frobenius ratio `|ratio - 1| <= 1e-3` across all `12` layers × `6` Rand controls (`delivered_dose_match_log` recorded); `CV_L <= 1.0` across `L16..L27`; `Rand` reserve `median_off_kl <= 1.50` (non-collapse ceiling, logged beside WSP KL); `common_pirate_ban` per-arm `surface_stem_hit_count / 40 <= 0.10` across all `6` banned arms + `surface_stem_hit` preflight `(True, False)`; `code_pass()` positive-control preflight PASS |
| **`C3b.0.3`** | Global | Exact row counts, exact prompt ID sets, non-empty strings across all cells; `Plain` non-broken rate `>= 0.90` on `Identity(20) + Implied(20) + Neutral(16)` |
| **`C3b.0.4`** | Global | Exact SHA-256 match of all 4 committed artifacts on disk and in receipt `meta.provenance_hashes` |
| **`D4_masking_check`** | **Isolated physical diagnostic** *(OUTSIDE global `C3b.0`)* | **Live `WSP_Pirate_12L` contrastive `L24` direction** (masked vs. unmasked vs. mismasked `(idx + 1) mod 4096`) on `12` disjoint frozen `PIRATE_LEXICON_SEEDS`, pinned `12` prompts (`res_00..res_07`, `cal_00..cal_03`), turn-start position `len(prompt_ids) - 1`, `alpha_L24(d = 3.6)` (`0.30` PMN at `L24`), `48` exact forwards. **Dual criterion:** `(1)` `masked_over_unmasked_seed_dlogit_ratio > 1.15` AND `mismasked_over_unmasked_seed_dlogit_ratio < 1.05`; AND `(2)` **`masked_over_unmasked_logit_l2_ratio <= 1.10`** (below C3 mismasked `1.1483`). **On FAIL:** `H1`/`H2` still report confirmatory verdicts, AND report flags *"low-gamma pre_norm masking efficiency unverified on L24"* prominently and withholds mechanistic attribution of delivery gain to masking |
| **`C3b.0_direction_quality`** | Per compiled arm (`H1`, `H2`) | `cos_ret >= 0.97` across ALL `12` layers `L ∈ {16..27}`; failure records `DIRECTION_INVALID` only for the affected hypothesis |
| **`C3b.0_commit_gate`** | Per compiled arm (`H1`, `H2`) | Highest `d* ∈ [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` where ALL `12` layers satisfy `stored_rank1_proj_ret >= 0.50` AND `stored_frob_ret >= 0.80` AND `RESERVE_PROMPTS_8` `median_off_kl <= 0.65`, `max_prompt_off_kl <= 1.50`, `max_single_pos_kl <= 8.0`, and `0/8` broken; `commit_gate_fallback == False` else `DOSE_INVALID` |
| **`C3b.0_NoMD_room`** | Pillar B only (`H2`) | `Plain` on `NOMD_32`: **`has_markdown >= 0.90`** (`>= 29/32`) AND **`mean(construct_types) >= 2.00`** |
| **`C3b.R_preflight` & `C3b.R`** | Reader-dependent only (`H1`, `D0`, `D5`) | **Pre-GPU Preflight (`C3b.R_preflight`):** BOTH providers (`deepseek-v4.1-flash` via `ocgo` + `gpt-6-luna` via `cliproxy`) return parseable JSON (`0` errors) on `READER_PREFLIGHT_CANARY_2` **BEFORE** launching L4. **Gold (`READER_GOLD_16_V2`):** both readers `>= 14/16` pooled AND `>= 5/6` cue-free `subtle_zero_stem` AND `>= 3/4` `first_person_non_pirate_neg` (`2` modern + `2` archaic). **Pool (`528` items):** pooled `κ >= 0.60`, per-ban `κ >= 0.55`. **Per-arm under `common_pirate_ban`:** `Rand_Pirate_12L_{s0..s2}` pooled `raw_agreement >= 0.85` (and `κ >= 0.50` if prev in `[0.10, 0.90]`); `WSP_Pirate_12L@d*` `κ >= 0.55` if prev in `[0.10, 0.90]` else `raw_agreement >= 0.85` |
| **`EQ-C3b.2`** | Per gating arm (`WSP_Pirate_12L@d*` for `H1`; `WSP_NoMD_12L@d*` for `H2`) | `math_acc >= Plain - 0.15` (fallback; strict `FINAL:` logged); `code_pass_rate >= Plain - 0.17` (real subprocess); `neutral_median_off_kl <= 0.80` nats/tok; `neutral_broken_rate <= 0.10` |

**Exact Full 3-Conjunct `READER_GOLD_16_V2` Operating Characteristics & Disclosures (`compute_exact_gold_gate_probs`):**
- Single-reader 50% certification threshold: **`q50 = 0.8471`**.
- Exact probability that **both** independent readers clear ALL 3 conjuncts (`>= 14/16` pooled, `>= 5/6` subtle, `>= 3/4` 1p-neg): **`0.000003` (`3×10⁻⁶`)** at `q = 0.50`, **`0.263`** at `q = 0.85`, **`0.549`** (`54.9%`) at `q = 0.90`, **`0.750`** (`75.0%`) at `q = 0.93`, **`0.868`** (`86.8%`) at `q = 0.95`, and **`0.953`** (`95.3%`) at `q = 0.97`.
- **Pre-registration disclosure (`gpt-6-luna` C3 score):** The `>= 5/6` subtle bar was set knowing `gpt-6-luna` scored `5/6` subtle (`13/14` pooled) on C3's gold set. `READER_GOLD_16_V2` contains 16 completely unseen items (zero cue-word overlap with `READER_SYSTEM_PROMPT` + 2 archaic-dialect non-pirate negatives). If either reader again misses `>= 2` subtle items — or if one provider becomes unavailable post-run — `H1` records `NOT_READ_LOW_KAPPA`, no `H1` claim is made in either direction, and any single-reader labels are logged strictly as non-confirmatory diagnostic.

---

## 4. Confirmatory Family (`m = 2` Holm-Bonferroni, `α = 0.05`) & Point-Bar Power

Exact Holm-Bonferroni step-down via frozen `c3_scoring.holm_reject(pvals, alpha=0.05)` (`p_(1) <= 0.025`, `p_(2) <= 0.05`; one-sided paired sign-flip permutation `N_perm = 100,000`, `PERM_SEED = 2026112701`; `90%` paired BCa CIs `N_boot = 10,000`, `BCA_SEED = 2026112702`; 9-state verdict precedence `VERDICT_PRECEDENCE_C3B` starting with **`BLOCKED_BY_C3b.0`**, mapped 1-to-1 from frozen `c3_scoring.VERDICT_PRECEDENCE`):

- **`H1` (`C3b.1_pirate_12L_vs_rand_common_ban`, `N = 40`):** `y_i = strict_pirate(WSP_Pirate_12L@d*, i) - mean_{s=0..2} strict_pirate(Rand_Pirate_12L_s@d*, i)` under `common_pirate_ban`. `PASS` iff preconditions ok (`C3b.0`, dir, dose, `C3b.R`, `EQ-C3b.2`, `broken <= 0.15` for both arm and `Rand_mean`) AND `holm_sig['H1']` AND `mean(y_i) >= +0.20` AND **`mean_tokens(Arm) / mean_tokens(Rand_mean) ∈ [0.85, 1.15]`** AND **`reader_both_coherent >= 0.85`**.
- **`H2` (`C3b.2_nomd_12L_vs_rand`, `N = 32` `NOMD_32`, 100% deterministic zero-judge):** `u_i = mean_{s=0..2} construct_types(Rand_NoMD_12L_s@d*, i) - construct_types(WSP_NoMD_12L@d*, i)`. `PASS` iff preconditions ok (`C3b.0`, dir, dose, `C3b.0_NoMD_room`, `EQ-C3b.2`, `broken <= 0.10` for both arm and `Rand_mean`) AND `holm_sig['H2']` AND `mean(u_i) >= +0.50` AND `(has_md(Plain) - has_md(Arm)) >= +0.20` AND `(has_md(Rand_mean) - has_md(Arm)) >= +0.15` AND `mean_tokens(Arm) / mean_tokens(Rand_mean) ∈ [0.85, 1.15]`.

**0-CU Point-Bar Power Table (`POWER_SIM_SEED = 2026112703`, `2,000` reps, `m = 2` rank-1 `α = 0.025`; pre-run gates excluded):**  
*Scope note:* For `H2`, simulates 3 of 5 conjuncts (`p_perm`, `mean(u_i) >= 0.50`, `has_md` drops `>= 0.20 / 0.15`; excludes `[0.85, 1.15]` length ratio and `broken <= 0.10`, noting that removing markdown syntax can shift generated tokens toward `0.85`). For `H1`, excludes length-ratio guard, `reader_both_coherent >= 0.85`, and `broken <= 0.15`; power at `Δ = +0.20` (exactly on the point bar) is `0.517` by sampling symmetry.

| Hypothesis | Generative Scenario | `p_perm_r1` | `p_has_md_conjuncts` | **`p_point_bar_pass_r1`** | `p_effect_below_bar_r1` |
|---|---|---:|---:|---:|---:|
| **`H1`** (`N=40`, bar `+0.20`) | `p_rand=0.05`, `Δ = +0.20 / +0.25 / +0.30 / +0.35` | `0.775 / 0.913 / 0.981 / 0.997` | — | **`0.517 / 0.750 / 0.915 / 0.978`** | `0.258 / 0.163 / 0.065 / 0.019` |
| **`H1` sensitivity** (`p_rand=0.15`) | `Δ = +0.25 / +0.30` | `0.808 / 0.920` | — | **`0.740 / 0.894`** | `0.068 / 0.026` |
| **`H2` C3 observed null** (`N=32`) | C3 observed null (`λ_p=3.3, λ_r=3.0, λ_a=3.00`) | `0.021` | `0.000` | **`0.000`** | `0.000` |
| **`H2` room-floor** (`N=32`, 3-bar) | Room-floor (`λ_p=1.5, λ_r=1.3`): `λ_a = 0.60 / 0.35` | `0.938 / 1.000` | `0.815 / 0.995` | **`0.758 / 0.995`** | `0.035 / 0.001` |
| **`H2` baseline-matched hypothetical** (`N=32`, 3-bar) | Hypothetical (`λ_p=3.3, λ_r=3.0`): `λ_a = 1.50 / 1.00 / 0.60` | `0.999 / 1.000 / 1.000` | `0.411 / 0.911 / 1.000` | **`0.411 / 0.911 / 1.000`** | `0.000 / 0.000 / 0.000` |

*(Hypothetical `H2` power spans **`0.411–1.000`** depending on effect size; C3's observed null has power **`0.000`**.)*

---

## 5. Registered Secondary & Diagnostic Readouts (`D0–D6` + `D_curve` + `D_above_cap`, Outside Holm)

- **`D0` Unbanned Pillar A (`N = 40`, `ban = none`):** Paired contrast of `WSP_Pirate_12L@d*` vs. `Rand_Pirate_12L_{s0..s2}@d*` without word-banning (`Identity_20` vs. `Implied_20` split).
- **`D1` Part 213 Copula (`12` pairs):** Paired `ΔLP_self`, `ΔLP_stranger`, and `Δ(self − stranger)` across `Plain`, `Sys_Pirate`, `WSP_Pirate_12L@d*`, `Rand_Pirate_12L_s0@d*`.
- **`D2` Full Preservation & Sibling Bleed:** All `7` capability arms on Math (fallback + strict `FINAL:`), Code (`12` real subprocess), Refusal (`8`, overall + `harmful_4` vs. `benign_4`), Neutral Chat (`16`, `off_KL` + greedy match), plus cross-arm sibling bleed (`WSP_Pirate_12L@d*` on `NOMD_32`; `WSP_NoMD_12L@d*` on `IDENTITY_20`).
- **`D3` Prompt Reference Ceilings & Text-Only Demo Export (`0` extra gens):** `Sys_Pirate` & `Sys_NoMD` reference ceilings + `DEMO_SHOWCASE_IDS_12` text slice (`0` edited weights exported).
- **`D4` Live `L24` Dual-Criterion Masking Check (`48` short fwds):** Logged & evaluated under `D4_masking_check` (outside `C3b.0`; frozen C3 `L24` lexicon vector also logged descriptively).
- **`D5` 3-Turn Follow-Through (`MULTITURN_12`, `ban = none`, `multiturn_d5_arms_4`):** `12 × 3 × 4 = 144` gens (`48` tok/turn).
- **`D6_heldout` Out-of-Sample Top-50 Boosted Tokens & Adaptive/Donor Ban (`DEMO_SHOWCASE_IDS_12`):** Top-50 strictly positive non-banned non-special tokens computed on **disjoint `RESERVE_PROMPTS_8` plain completions** (NOT `calib_16`), followed by `Plain@common` (reused), `WSP_Pirate_12L@own50+common`, `WSP_Pirate_12L@donor_rand0+common`, and `Rand0@own50+common` (`36` new gens).
- **`D_curve` Full 7-Rung Empirical Dose-Response Curve (`0` extra gens):** Logs `median_off_kl(d)`, `max_prompt_off_kl(d)`, `min_proj_ret(d)`, `min_frob_ret(d)`, and `n_broken(d)` across `d ∈ [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` PMN for both compiled arms.
- **`D_above_cap` Optional Lost-KL-Bet Diagnostic (`<= 36` gens, `<= 2,688` max tok):** Triggered per arm iff any rung in `DOSE_LADDER` fails **ONLY** the KL clause (passing `bf16` retention on all 12 layers and `0/8` broken); runs `DEMO_SHOWCASE_IDS_12 × {none, common_pirate_ban}` (`24` gens, `64` tok) for Pirate and/or `md_00..md_11` (`12` gens, `96` tok) for NoMD at the lowest such rung. Pre-registered reading: rising `D_curve` KL + `DOSE_INVALID` + positive `D_above_cap` => *"flip requires more KL than the 0.65 safety cap allows"*; rising `D_curve` KL + `DOSE_INVALID` + flat `D_above_cap` => *"no flip even above the cap"*.

---

## 6. Exact `B = 1` Compute Estimate (Single Ephemeral NVIDIA L4, Zero `B = 8` Gambling)

- **Upload:** `< 1.5 MB` (`1.24 MB` `c3b_vector_bank.safetensors` + `166 KB` constants + scripts → `< 5 s`).
- **Exact Generation & Forward Accounting (Exact `B = 1` Greedy Throughout):**
  - Calibration (`calib_16 × 3`, `32` tok): `48` gens (`1,536` max tok) + `48` `B = 1` hook-capture fwds.
  - Reserve Commit Gate (`RESERVE_PROMPTS_8`, `Plain` + `2` arms × `7` rungs, `48` tok): `120` gens (`5,760` max tok) + `8 + 112 + 48` (`M8` 6 Rand) = `168` `B = 1` KL fwds.
  - Pillar A Primary + `D0` + `D3` slice (`N = 40 × 2` bans × `6` arms, `64` tok): `480` gens (`30,720` max tok).
  - Pillar B (`NOMD_32 × 6` arms, `96` tok): `192` gens (`18,432` max tok).
  - Shared Capability Suite (`7` arms × `56` gens, `3,968` max tok/arm) + `D2` Sibling Bleed (`32 × 96 + 20 × 64 = 52` gens, `4,352` tok): `444` gens (`32,128` max tok) + `112` `B = 1` neutral KL fwds.
  - Secondaries: `D5` Multi-turn (`12 × 3 × 4 = 144` gens, `48` tok = `6,912` max tok); `D6_heldout` (`12 × 3 = 36` new gens, `64` tok = `2,304` max tok + `16` `B = 1` reserve-logit fwds for `WSP_Pirate_12L` & `Rand0`, reusing `Plain` reserve logits); `D1` Copula (`4 × 24 = 96` `B = 1` fwds); `D4` (`48` fwds) + `sham` (`1` fwd) = `49` preflight fwds.
  - **Core Grand Total:** **`1,464` generations** (**`97,792` max tokens**, realistic **`~82k–88k` actual tokens** after EOS) + **`489` short `B = 1` forwards** (plus up to **`36` gens / `2,688` max tok** if `D_above_cap` triggers on both arms).
- **Receipt-Derived L4 Throughput & Cost Anchor:** Anchored strictly on Stage C3's measured L4 `B = 1` receipt (`104,657` generated tokens + `682` fwds in `6,628.86 s` = **`15.79 tok/s`**, **`4.65 CU`** compute = **`2.525 CU/hr`** implied burn rate), Stage C3b lands at **`~90–105 min`** / **`~3.4–4.4 CU`** on one ephemeral L4 (up to `~4.5 CU` if `D_above_cap` triggers on both arms; `acct1` balance `159.40 CU`, `0 active`). Detached drop-proof Colab launcher (`start_new_session=True` + `timeout -k 5 35 colab exec` polling) prevents any websocket-drop idle CU burn. Post-run dual blind readers (`528` pooled + `16` gold + `36` D6 + up to `24` `D_above_cap` items) run locally for `0` Colab CU after passing pre-GPU **`C3b.R_preflight`** on both `gpt-6-luna` (`cliproxy`) and `deepseek-v4.1-flash` (`ocgo`).
