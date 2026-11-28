# OLMo 3 7B Stage C3b (`WSP v2` — Draft v1): Distributed 12-Layer Zero-Hook `pre_norm` Weight-Space Programming

**Status:** Draft v1 (Ready for Cold Design Review; `0 CU` spent)  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (`bf16`, `d_model = 4096`, `d_ff = 11008`, `32` layers)  
**Parent Stage:** Stage C3 (`WSP v1`, published on public `main` @ `3933393d`, `vesper-input-c3` @ `4e83807b`)  
**Policy `11-07`:** Zero edited model weight checkpoints saved to disk or published; receipts record only edit metadata, scalar diagnostics, and text transcripts.

---

## 0. Motivation & Exact Upgrades Over Stage C3 (`WSP v1`, `3933393d`)

Stage C3 (`WSP v1`) verified zero-hook literal `bf16` weight compilation mechanics (`active_hooks_max_seen = 0`, `204/204` exact bitwise restores, `0/360` banned-word leaks, `80%` Math / `100%` Code preservation across all four WSP arms) and observed pirate-lexical next-token logit shifts (`" yer" +2.78`, `" ain" +2.72`, `" ye" +2.43` on in-sample `calib_16`, `+0.817` nats `D1` `lp_self` on `12/12` pairs), yet failed to flip autoregressive generation and failed `C3.0` on a single under-specified `D4` statistic (`logit_l2 = 1.017×` vs. logged `seed_dlogit = 1.508×`).

Stage C3b (`WSP v2`) directly tests the behavioral transition hypothesis with **six concrete pre-registered upgrades**:
1. **Distributed `K = 12` contiguous mid-to-late layers (`L16–L27`):** Spreading `d_total` across `12` layers instead of `3` keeps per-layer `δ_L` at `0.10–0.80` PMN/layer (avoiding `post_feedforward_layernorm` `RMSNorm` saturation that would crush native MLP features at `K = 3`), while scaling cumulative steering drive up to **`12×`** over `WSP v1`.
2. **Wide 7-rung commit-gate ladder & empirically grounded KL cap (`RESERVE_PROMPTS_8`):** `d_total ∈ [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` PMN with `reserve_median_off_kl <= 0.65` nats/tok (`max_prompt <= 1.50`, `max_single_pos <= 8.0`, `0/8` broken, `bf16` retention `>= 0.50 / 0.80` on all 12 layers). In C3, `RESERVE_8` `Plain` used markdown on `4/8` prompts (`NEUTRAL_16`: `15/16`), `Sys_Pirate` had `med_off_KL = 0.903`, and `Sys_NoMD` had `1.111` — so `0.65` sits below prompt-level KL collapse without penalizing valid style/persona shifts, while `EQ-C3b.2` enforces binding Math and real-subprocess Code safety.
3. **Focused `m = 2` Holm-Bonferroni confirmatory family (`α = 0.05`):** Drops the `~5×` weaker static Lexicon arms, focusing strictly on **`H1`** (Pirate Contrastive `12L` under `common_pirate_ban`, `N = 40`) and **`H2`** (No-Markdown Contrastive `12L` on expanded **`NOMD_32`**, `N = 32`, raising Pillar B power from `0.67–0.80` into `0.77–0.998`).
4. **Isolated, dual-criterion `D4_masking_check` (OUTSIDE global `C3b.0`):** Names BOTH exact `L24` statistics explicitly before any run — on-target delivery (`masked_over_unmasked_seed_dlogit_ratio > 1.15` AND `mismasked_over_unmasked_seed_dlogit_ratio < 1.05`) AND full-vocabulary perturbation bound (`masked_over_unmasked_logit_l2_ratio <= 1.25`) — reported prominently without blocking unrelated `H1`/`H2` behavioral tests.
5. **Brand-new zero-leakage `READER_GOLD_14_V2` & binomial-justified `C3b.R` bars:** 14 completely unseen items (0 overlap with C3's `READER_GOLD_14`), with `>= 12/14` pooled, `>= 5/6` `subtle_zero_stem`, and `2/2` `first_person_non_pirate_neg` (`78.5%` pass probability for two `90%`-accurate readers vs. `28%` under `6/6`, while holding two `50%` coin-flip readers to `1.2%`).
6. **Out-of-sample `D6_heldout` & honest `B = 1` compute budget:** Top-50 boosted tokens computed on disjoint `RESERVE_PROMPTS_8` plain completions (fixing C3 `F9`), and exact `B = 1` greedy decoding budgeted directly from day 1 (eliminating C3's `0.25`-logit `B = 8` `bf16` left-padding drift surprise).

---

## 1. Frozen Committed Artifacts & Provenance (`C3b.0.4`)

All 4 files are committed locally and hashed pre-launch (verified by `C3b.0.4` on disk and inside the receipt):

| Artifact | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Constants JSON | `olmo3_virtual_and_highway/c3b_prereg_constants.json` | `147,477` | `3b73f4e02721cbc99b2c6a74e31275e374dcc235d7c4d68b0a5d6452807f41a6` |
| Vector Bank (`87` tensors) | `olmo3_virtual_and_highway/c3b_vector_bank.safetensors` | `1,236,979` | `a32662a7b3671a4613b7b412e78ff4b14a7fcd41e385fb9d43da31bd1fea2656` |
| Frozen Scoring Module | `olmo3_virtual_and_highway/c3_scoring.py` | `12,743` | `b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f` |
| Design Spec (this file) | `olmo3_virtual_and_highway/OLMO3-WSP-C3B-DESIGN.md` | *(hashed on freeze)* | *(pinned in runner & scorer after review)* |

Zero `r_lens.pt` upload is needed: `c3b_vector_bank.safetensors` (`1.24 MB`, `87` tensors) holds `L{16..27}_gamma_drop_idx`, `L{16..27}_rand_{pirate,nomd}_pre_unit_s{0,1,2}`, and the three frozen `L24` `D4` reference vectors copied verbatim from `c3_vector_bank.safetensors` (`sha256: e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2`).

---

## 2. Literal `pre_norm` Weight Compilation Specification (`K = 12`, `L16–L27`)

### 2.1 One-Time `calib_16` Capture (Hooks ONLY Here, Verified `0` Before Any Eval)
1. Generate exact `B = 1` greedy `32`-token completions on `calib_16` under `Plain`, `Sys_Pirate`, and `Sys_NoMD`.
2. Attach temporary read-only forward hooks across `L ∈ {16..27}` on `mlp.down_proj` (inputs `m_L`, pre-norm outputs `pre_out_L`), `post_feedforward_layernorm` (post-norm outputs `post_out_L`), and decoder layer output `h_out_L` across `(turn-start token len(prompt_ids)-1 + first 16 generated tokens)`. Remove all hooks immediately and assert `count_active_hooks(model) == 0`.
3. Per layer `L ∈ {16..27}`:
   - Key vector: $\hat{k}_L = \bar{m}_L / \|\bar{m}_L\|_2$, scalar projection mean $\bar{c}_L = \mathbb{E}[m_L \cdot \hat{k}_L]$, stability $CV_L = \text{std}(m_L \cdot \hat{k}_L) / |\bar{c}_L|$, baseline pre-norm RMS $r_L = \mathbb{E}[\text{rms}(\text{pre\_out}_L)]$, baseline post-MLP norm $\text{pmn}_L = \mathbb{E}[\|\text{post\_out}_L\|_2]$.
   - Contrastive target post-norm direction: $v_{\text{post},L} = \text{unit}(\mathbb{E}[h_L^{\text{Sys}}] - \mathbb{E}[h_L^{\text{Plain}}])$ for Pirate and NoMD.
   - Masked `pre_norm` unit direction (`drop_k = 4` lowest-`|\gamma_{\text{ff},L}|` indices from `L{L}_gamma_drop_idx`): zero those 4 coordinates in $v_{\text{post},L} / \gamma_{\text{ff},L}$ and unit-normalize to $\hat{v}_{\text{pre},L}$, recording `cos_ret` (gated `>= 0.97` across all 12 layers by `C3b.0_direction_quality`).

### 2.2 Rank-1 `bf16` Weight Edit & Exact Storage Retention Audit (`0 CU`)
For total dose $d \in [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]$ PMN across $K = 12$ layers ($\delta_L = (d / 12)\,\text{pmn}_L$):
$$\alpha_L(d) = \frac{\delta_L \cdot r_L}{\bar{c}_L \cdot \|\gamma_{\text{ff},L} \odot \hat{v}_{\text{pre},L}\|_2}, \qquad W_{\text{edited},L} = \text{bf16}\!\left(W_{\text{clean},L}^{\text{fp32}} + \alpha_L(d)\,\hat{v}_{\text{pre},L}\hat{k}_L^T\right)$$
Matched random controls `Rand_{Pirate,NoMD}_12L_s{0,1,2}` use the exact same per-layer scalars $\alpha_L(d^*)$ and $\hat{k}_L$ with the frozen masked isotropic unit vectors `L{L}_rand_*_pre_unit_s{0,1,2}`. After every edited arm, `W_clean` is restored and verified bitwise via `torch.equal`.

**0-CU `bf16` Retention Audit on Real OLMo-3 `L16–L27` `mlp.down_proj.weight` (`key_seed = 2026112704`):**

| `alpha` | Min `stored_rank1_proj_ret` across `L16–L27` | Status |
|---:|---:|---|
| `0.05` (sham) | `0.2839` | `>= 0.25` (`C3b.0.1` PASS) |
| `0.25` | `0.8281` | `>= 0.50` (`C3b.0_commit_gate` PASS) |
| `0.50` | `0.9567` | `>= 0.50` |
| `1.00` | `0.9927` | `>= 0.50` |
| `2.00` | `0.9989` | `>= 0.50` |
| `4.00` | `0.9999` | `>= 0.50` |

**Layer Geometry (`L16–L27`, `drop_k = 4`):** Across all 12 layers, `||W_down||_F ∈ [152.05, 153.65]`, `rms_gamma ∈ [0.1134, 0.2358]`, and `n_lt_0.005 ∈ {1, 2, 3}` (every near-zero coordinate `< 0.005` is masked by `drop_k = 4`).

---

## 3. Exact Gate-by-Gate Specification Matrix (Every Gate Names Its Exact Statistic)

| Gate | Scope | Exact Pre-Registered Formula |
|---|---|---|
| **`C3b.0.1`** | Global | `active_hooks_max_seen == 0`; `restore_exact_all == True` & `restore_checks >= expected_min_restore_checks`; `weight_checkpoints_saved == 0`; `effective_gen_batch_size == 1`; `sham(alpha=0.05 on L16..L27)` finite & `0 < max_abs_dlogit < 0.50` & `min_{L in 16..27} stored_rank1_proj_ret >= 0.25`; all selected `d*` compiled arms + 6 Rand controls have `max_abs_dlogit > 0.0` |
| **`C3b.0.2`** | Global | `nonfinite_count == 0`; `M8` Rand intended Frobenius ratio `|ratio - 1| <= 1e-3` across all `12` layers × `6` Rand controls; `CV_L <= 1.0` across `L16..L27`; `Rand` reserve `median_off_kl <= 0.80`; `common_pirate_ban` per-arm `surface_stem_hit_count / 40 <= 0.10` across all `6` banned arms + `surface_stem_hit` preflight `(True, False)`; `code_pass()` positive-control preflight PASS |
| **`C3b.0.3`** | Global | Exact row counts, exact prompt ID sets, non-empty strings across all cells; `Plain` non-broken rate `>= 0.90` on `Identity(20) + Implied(20) + Neutral(16)` |
| **`C3b.0.4`** | Global | Exact SHA-256 match of all 4 committed artifacts on disk and in receipt `meta.provenance_hashes` |
| **`D4_masking_check`** | **Isolated physical diagnostic** *(OUTSIDE global `C3b.0`)* | **Dual criterion naming BOTH statistics:** `(1)` `masked_over_unmasked_seed_dlogit_ratio > 1.15` AND `mismasked_over_unmasked_seed_dlogit_ratio < 1.05` on `L24` Pirate seeds; AND `(2)` `masked_over_unmasked_logit_l2_ratio <= 1.25` full-vocabulary perturbation bound |
| **`C3b.0_direction_quality`** | Per compiled arm (`H1`, `H2`) | `cos_ret >= 0.97` across ALL `12` layers `L ∈ {16..27}`; failure records `DIRECTION_INVALID` only for the affected hypothesis |
| **`C3b.0_commit_gate`** | Per compiled arm (`H1`, `H2`) | Highest `d* ∈ [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` where ALL `12` layers satisfy `stored_rank1_proj_ret >= 0.50` AND `stored_frob_ret >= 0.80` AND `RESERVE_PROMPTS_8` `median_off_kl <= 0.65`, `max_prompt_off_kl <= 1.50`, `max_single_pos_kl <= 8.0`, and `0/8` broken; `commit_gate_fallback == False` else `DOSE_INVALID` |
| **`C3b.0_NoMD_room`** | Pillar B only (`H2`) | `Plain` on `NOMD_32`: `has_markdown >= 0.65` (`>= 21/32`) AND `mean(construct_types) >= 1.50` |
| **`C3b.R`** | Reader-dependent only (`H1`, `D0`, `D5`) | `READER_GOLD_14_V2`: both readers (`deepseek-v4.1-flash` via `ocgo` + `gpt-6-luna` via `cliproxy`) `>= 12/14` pooled AND `>= 5/6` `subtle_zero_stem` AND `2/2` `first_person_non_pirate_neg`. Pool (`core_arms_6 × 40 × 2 + multiturn_4 × 12 = 528` items): pooled `κ >= 0.60`, per-ban `κ >= 0.55`. Per-arm under `common_pirate_ban`: `Rand_Pirate_12L_{s0..s2}` pooled `raw_agreement >= 0.85` (and `κ >= 0.50` if prev in `[0.10, 0.90]`); `WSP_Pirate_12L@d*` `κ >= 0.55` if prev in `[0.10, 0.90]` else `raw_agreement >= 0.85` |
| **`EQ-C3b.2`** | Per gating arm (`WSP_Pirate_12L@d*` for `H1`; `WSP_NoMD_12L@d*` for `H2`) | `math_acc >= Plain - 0.15` (fallback; strict `FINAL:` logged); `code_pass_rate >= Plain - 0.17` (real subprocess); `neutral_median_off_kl <= 0.80` nats/tok; `neutral_broken_rate <= 0.10` |

---

## 4. Confirmatory Family (`m = 2` Holm-Bonferroni, `α = 0.05`) & Point-Bar Power

Exact Holm-Bonferroni step-down via frozen `c3_scoring.holm_reject(pvals, alpha=0.05)` (`p_(1) <= 0.025`, `p_(2) <= 0.05`; one-sided paired sign-flip permutation `N_perm = 100,000`, `PERM_SEED = 2026112701`; `90%` paired BCa CIs `N_boot = 10,000`, `BCA_SEED = 2026112702`; 9-state verdict precedence from `c3_scoring.VERDICT_PRECEDENCE`):

- **`H1` (`C3b.1_pirate_12L_vs_rand_common_ban`, `N = 40`):** `y_i = strict_pirate(WSP_Pirate_12L@d*, i) - mean_{s=0..2} strict_pirate(Rand_Pirate_12L_s@d*, i)` under `common_pirate_ban`. `PASS` iff preconditions ok (`C3b.0`, dir, dose, `C3b.R`, `EQ-C3b.2`, `broken <= 0.15` for both arm and `Rand_mean`) AND `holm_sig['H1']` AND `mean(y_i) >= +0.20`.
- **`H2` (`C3b.2_nomd_12L_vs_rand`, `N = 32` `NOMD_32`, 100% deterministic zero-judge):** `u_i = mean_{s=0..2} construct_types(Rand_NoMD_12L_s@d*, i) - construct_types(WSP_NoMD_12L@d*, i)`. `PASS` iff preconditions ok (`C3b.0`, dir, dose, `C3b.0_NoMD_room`, `EQ-C3b.2`, `broken <= 0.10` for both arm and `Rand_mean`) AND `holm_sig['H2']` AND `mean(u_i) >= +0.50` AND `(has_md(Plain) - has_md(Arm)) >= +0.20` AND `(has_md(Rand_mean) - has_md(Arm)) >= +0.15` AND `mean_tokens(Arm) / mean_tokens(Rand_mean) ∈ [0.85, 1.15]`.

**0-CU Point-Bar Power Table (`POWER_SIM_SEED = 2026112703`, `2,000` reps, `m = 2` rank-1 `α = 0.025`; pre-run gates excluded):**

| Hypothesis | Generative Scenario | `p_perm_r1` | `p_has_md_conjuncts` | **`p_point_bar_pass_r1`** | `p_effect_below_bar_r1` |
|---|---|---:|---:|---:|---:|
| **`H1`** (`N=40`, bar `+0.20`) | `p_rand=0.05`, `Δ = +0.20 / +0.25 / +0.30 / +0.35` | `0.775 / 0.913 / 0.981 / 0.997` | — | **`0.517 / 0.750 / 0.915 / 0.978`** | `0.258 / 0.163 / 0.065 / 0.019` |
| **`H1` sensitivity** (`p_rand=0.15`) | `Δ = +0.25 / +0.30` | `0.808 / 0.920` | — | **`0.740 / 0.894`** | `0.068 / 0.026` |
| **`H2`** (`N=32`, 3-bar) | Room-floor (`λ_p=1.5, λ_r=1.3`): `λ_a = 0.60 / 0.35` | `0.955 / 1.000` | `0.821 / 0.994` | **`0.768 / 0.994`** | `0.039 / 0.001` |
| **`H2`** (`N=32`, 3-bar) | C3 empirical (`λ_p=3.3, λ_r=3.0`): `λ_a = 1.50 / 1.00 / 0.60` | `0.998 / 1.000 / 1.000` | `0.400 / 0.925 / 0.998` | **`0.400 / 0.925 / 0.998`** | `0.000 / 0.000 / 0.000` |

---

## 5. Registered Secondary & Diagnostic Readouts (`D0–D6` + `D_curve`, Outside Holm)

- **`D0` Unbanned Pillar A (`N = 40`, `ban = none`):** Paired contrast of `WSP_Pirate_12L@d*` vs. `Rand_Pirate_12L_{s0..s2}@d*` without word-banning (`Identity_20` vs. `Implied_20` split).
- **`D1` Part 213 Copula (`12` pairs):** Paired `ΔLP_self`, `ΔLP_stranger`, and `Δ(self − stranger)` across `Plain`, `Sys_Pirate`, `WSP_Pirate_12L@d*`, `Rand_Pirate_12L_s0@d*`.
- **`D2` Full Preservation & Sibling Bleed:** All `7` capability arms on Math (fallback + strict `FINAL:`), Code (`12` real subprocess), Refusal (`8`), Neutral Chat (`16`, `off_KL` + greedy match), plus cross-arm sibling bleed (`WSP_Pirate_12L@d*` on `NOMD_32`; `WSP_NoMD_12L@d*` on `IDENTITY_20`).
- **`D3` Prompt Reference Ceilings & Text-Only Demo Export (`0` extra gens):** `Sys_Pirate` & `Sys_NoMD` reference ceilings + `DEMO_SHOWCASE_IDS_12` text slice (`0` edited weights exported).
- **`D4` Dual-Criterion `L24` Masking Check (`48` short fwds):** Logged & evaluated under `D4_masking_check` (outside `C3b.0`).
- **`D5` 3-Turn Follow-Through (`MULTITURN_12`, `ban = none`, `multiturn_d5_arms_4`):** `12 × 3 × 4 = 144` gens (`48` tok/turn).
- **`D6_heldout` Out-of-Sample Top-50 Boosted Tokens & Adaptive/Donor Ban (`DEMO_SHOWCASE_IDS_12`):** Top-50 strictly positive non-banned non-special tokens computed on **disjoint `RESERVE_PROMPTS_8` plain completions** (NOT `calib_16`), followed by `Plain@common` (reused), `WSP_Pirate_12L@own50+common`, `WSP_Pirate_12L@donor_rand0+common`, and `Rand0@own50+common` (`36` new gens).
- **`D_curve` Full 7-Rung Empirical Dose-Response Curve (`0` extra gens):** Logs `median_off_kl(d)`, `max_prompt_off_kl(d)`, `min_proj_ret(d)`, `min_frob_ret(d)`, and `n_broken(d)` across `d ∈ [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` PMN for both compiled arms.

---

## 6. Exact `B = 1` Compute Estimate (Single Ephemeral NVIDIA L4, Zero `B = 8` Gambling)

- **Upload:** `< 1.4 MB` (`1.24 MB` `c3b_vector_bank.safetensors` + `147 KB` constants + scripts → `< 5 s`).
- **Exact Generation & Forward Accounting (Exact `B = 1` Greedy Throughout):**
  - Calibration (`calib_16 × 3`, `32` tok): `48` gens (`1,536` max tok) + `48` `B = 1` hook-capture fwds.
  - Reserve Commit Gate (`RESERVE_PROMPTS_8`, `Plain` + `2` arms × `7` rungs, `48` tok): `120` gens (`5,760` max tok) + `8 + 112 + 48` (`M8` 6 Rand) = `168` `B = 1` KL fwds.
  - Pillar A Primary + `D0` + `D3` slice (`N = 40 × 2` bans × `6` arms, `64` tok): `480` gens (`30,720` max tok).
  - Pillar B (`NOMD_32 × 6` arms, `96` tok): `192` gens (`18,432` max tok).
  - Shared Capability Suite (`7` arms × `56` gens, `3,968` max tok/arm) + `D2` Sibling Bleed (`32 × 96 + 20 × 64 = 52` gens, `4,352` tok): `444` gens (`32,128` max tok) + `112` `B = 1` neutral KL fwds.
  - Secondaries: `D5` Multi-turn (`12 × 3 × 4 = 144` gens, `48` tok = `6,912` max tok); `D6_heldout` (`12 × 3 = 36` new gens, `64` tok = `2,304` max tok + `16` `B = 1` reserve-logit fwds for `WSP_Pirate_12L` & `Rand0`, reusing `Plain` reserve logits!); `D1` Copula (`4 × 24 = 96` `B = 1` fwds); `D4` + `sham` (`48 + 1 = 49` preflight fwds).
  - **Grand Total:** **`1,464` generations** (**`97,792` max tokens**, realistic **`~82k–88k` actual tokens** after EOS) + **`489` short `B = 1` forwards**.
- **Measured L4 Throughput & Cost Anchor:** Anchored directly on our Stage C3 real L4 `B = 1` rate (`15.8 tok/s`, `4.65 CU` for `104,657` tokens + `682` fwds at `~2.33 CU/hr`), Stage C3b lands at **`~80–95 min`** / **`~3.1–3.8 CU`** on one ephemeral L4 (`acct1` balance `159.40 CU`, `0 active`). Post-run dual blind readers (`528` pooled + `14` gold + `36` D6 items) run locally for `0` Colab CU (`gpt-6-luna` via `cliproxy` locally + `deepseek-v4.1-flash` via `ocgo` run by Claude or Elise with `OPENCODE_GO_API_KEY`).
