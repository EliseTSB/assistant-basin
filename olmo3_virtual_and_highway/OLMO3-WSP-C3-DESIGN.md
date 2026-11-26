# Stage C3 Design (Draft v1 — WSP v1): Zero-Hook Literal `pre_norm` Weight-Space Programming for Behavioral Persona & Mechanical No-Markdown Compilation on OLMo 3 7B

**Status:** **DRAFT v1 — FOR COLD DESIGN REVIEW (`0 CU`, zero Stage C3 GPU runs exist).** Authored by Vesper (`neon`, research lead) on clean branch `vesper-c3-wsp` (branched off public `main` @ `4e277a20`) for Council cold design review (`open-olmo3-7b-virtual-neurons-2026-11-22`).  
**Target Model (frozen):** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `bfloat16`, `B = 1` (zero padding / zero batch-drift risk), greedy decoding (`do_sample=False`, `use_cache=True` for generation; `use_cache=False` for single-pass teacher-forced KL & copula checks).  
**Policy Boundary (`11-07` Weight-Release Policy):** Honored in full. Stage C3 compiles literal weight edits in memory on Colab, evaluates them with **zero runtime PyTorch hooks**, restores the original weights cleanly after each arm, and **never saves or publishes any edited model checkpoint**. The committed receipt stores only edit metadata, scalar diagnostics, and generated text transcripts. Any future public release of edited weights remains gated on explicit separate authorization from Elise.

### Reproducible `0 CU` Pre-Data Artifacts (Double-Run Byte-Identical `cmp` Verified)
- Builder: [`olmo3_virtual_and_highway/build_olmo3_c3_constants.py`](build_olmo3_c3_constants.py)
- Constants (`olmo3-stage-c3-wsp-v1-prereg-constants-v1`): [`olmo3_virtual_and_highway/c3_prereg_constants.json`](c3_prereg_constants.json) (`59,264` B, `sha256 = 581bfc821794b6a7ab3c57e4292fd5380c024f6ac3724b7387c40ac60a6f7315`)
- Compact Frozen Vector Bank (`work/olmo3_c3/c3_vector_bank.safetensors`, **`511,005` B [`511 KB`]**, `34` tensors — eliminates uploading the `2.11 GB` `r_lens.pt` to Colab!): `sha256 = e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2`
- Source RelP Lens used offline at `0 CU` (`r_lens.pt`, NOT uploaded to Colab): `sha256 = ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955`
- Offline PyTorch / safetensors invariant test suite: [`olmo3_virtual_and_highway/test_olmo3_c3_constants.py`](test_olmo3_c3_constants.py)
- Pure-stdlib (`/usr/bin/python3 -I`) design-pin verifier: [`olmo3_virtual_and_highway/test_olmo3_c3_design_stdlib.py`](test_olmo3_c3_design_stdlib.py)

---

## 1. Research Question & Prior Context

Stage C0 (`c96ebfdb`) and C0b (`112450e4`) mapped OLMo 3 7B's linear transport band, while Stage C1 (`45b9208c`) and Stage C2 (`4e277a20`) failed their global pre-run gates (`C1.0.3` anchor drift; `C2.0.2` unpassable upstream-patch telescoping gate) and held all primary confirmatory claims closed (`BLOCKED_BY_C1.0` / `BLOCKED_BY_C2.0`). Exploratory post-hoc diagnostics from Stage C2 motivated three hypotheses tested here in literal zero-hook weight space: (i) mean-centered unembeddings (`u_t - μ_U`) reduce common-mode bleed cleanly without diagonal shrinkage; (ii) OLMo 3 7B's `post_feedforward_layernorm` has `1–3` near-zero `|γ|` dimensions per layer (`L21/24/27`), so dropping `k = 4` smallest `|γ|` dimensions lifts `pre_norm` deliverable efficiency to `~100%` (`9.15×`, `1.45×`, `1.95×` on Pirate lexicon writes; `10.59×`, `2.05×`, `29.50×` on Pure-Markdown anti-writes at `0 CU`); and (iii) `K = 3` simultaneous writes across `L21, L24, L27` compose with minimal interaction (`+0.73%` exploratory median).

**Core Scientific Question (Stage C3 / `WSP v1`):**
1. **Pillar A (Primary — Compiled Persona vs. Lexical Word-Stuffing):** Can a **zero-hook, zero-gradient literal rank-1 weight edit** into `mlp.down_proj.weight` across `L ∈ {21, 24, 27}` (low-`γ` `pre_norm` masked) compile a coherent **Pirate persona** into `Olmo-3-7B-Instruct` that beats Frobenius-norm-matched random edits (`Random_Matched`, Part 173 control) on held-out direct/indirect identity, implied-content questions, and 3-turn follow-through — **and survives Part 168 decode word-banning** (banning all 127 static pirate stems + each arm's own top-50 pushed bias tokens)? And does closed-form **contrastive compilation** (`WSP_Pirate_Contrastive`, `Steer2Edit`-style mean residual difference on 16 disjoint calibration prompts) outperform pure closed-form **lexicon-lens compilation** (`WSP_Pirate_Lexicon`, centered `R_ridge` on 12 pirate words, zero rollouts) under word-banning?
2. **Pillar B (Secondary — Mechanical Formatting Rule, Never Combined with Persona):** Can the same zero-hook literal `pre_norm` weight-edit mechanism compile a 100% mechanically scored **No-Markdown rule** (`WSP_NoMD_Contrastive` and pure `0`-rollout `WSP_NoMD_Lexicon`) that cuts markdown structure counts on 24 held-out markdown-eliciting prompts while preserving untuned Math (`20` problems), Python Code (`12` unit-tested functions), and neutral-chat `off_KL` (`EQ-C3.2`)?

---

## 2. Literal Zero-Hook Weight-Space Compilation (`mlp.down_proj.weight`, `L ∈ {21, 24, 27}`)

`Olmo3ForCausalLM` is bias-free (`attention_bias = false`, no `down_proj.bias`), with post-norm residual updates:
$$h_{\text{out}}^{(L)} = h_{\text{mid}}^{(L)} + \text{RMSNorm}_{\gamma_{\text{ff}}^{(L)}}\!\big(W_{\text{down}}^{(L)} m_L(x)\big), \qquad m_L(x) = \text{SiLU}(W_{\text{gate}}^{(L)} h_{\text{mid}}^{(L)}) \odot (W_{\text{up}}^{(L)} h_{\text{mid}}^{(L)}) \in \mathbb{R}^{11008}.$$

### 2.1 Calibration Quantities (Computed Strictly on Disjoint `calib_16`, Never Evaluated)
On the `16` disjoint calibration prompts (`CALIB_PROMPTS_16`, greedy `32` tokens, `Plain`, `SYS_PIRATE`, and `SYS_NOMD`), record at each layer `L ∈ {21, 24, 27}` across the assistant turn-start token (`<|im_start|>assistant\n`) plus the first `16` generated tokens:
1. **Persistent MLP Key $\hat{k}_L \in \mathbb{R}^{11008}$:** Mean plain MLP intermediate activation $\bar{m}_L = \text{mean}(m_L(x))$, unit key $\hat{k}_L = \bar{m}_L / \|\bar{m}_L\|_2$, scalar mean key projection $\bar{c}_L = \text{mean}(\hat{k}_L^\top m_L(x)) > 0$, and key firing stability $\text{CV}_L = \text{std}(\hat{k}_L^\top m_L) / \bar{c}_L$ (logged in receipt).
2. **Clean Layer Norms:** Mean clean post-MLP output norm $\text{pmn}[L] = \text{mean}(\|\text{RMSNorm}_{\gamma_{\text{ff}}^{(L)}}(W_{\text{down}}^{(L)} m_L)\|_2)$ and mean clean pre-norm MLP RMS $\text{rms\_pre}[L] = \text{mean}(\text{rms}(W_{\text{down}}^{(L)} m_L))$.
3. **Contrastive Target Directions ($v_{\text{post}}^{(L)}$):** For `WSP_Pirate_Contrastive`, $\Delta h_L = \text{mean}(h_{\text{out}}^{(L,\text{SYS\_PIRATE})} - h_{\text{out}}^{(L,\text{Plain})})$ across `calib_16`, normalized to unit norm. Identically for `WSP_NoMD_Contrastive` using `SYS_NOMD`.

### 2.2 Low-`γ` `pre_norm` Masking & Exact Frobenius-Norm-Matched Rank-1 Edit
Given any unit post-norm target direction $v_{\text{post}}^{(L)} \in \mathbb{R}^{4096}$, let $S_L$ be the frozen `drop_k = 4` smallest $|\gamma_{\text{ff}}^{(L)}|$ coordinates (`L21: [1074, 1483, 3125, 3462]`, `L24: [1042, 1745, 2632, 3807]`, `L27: [726, 1745, 2632, 3885]`). Compute:
$$v_{\text{pre\_raw}}^{(L)} = \text{mask}_{S_L}\!\left(\frac{v_{\text{post}}^{(L)}}{\gamma_{\text{ff}}^{(L)}}\right), \qquad \hat{v}_{\text{pre}}^{(L)} = \frac{v_{\text{pre\_raw}}^{(L)}}{\|v_{\text{pre\_raw}}^{(L)}\|_2}.$$
For any total dose $d$ in the 5-rung ladder (`DOSE_LADDER = [0.15, 0.30, 0.45, 0.60, 0.80]`), split equally across `K = 3` layers ($\delta_L = (d/3)\cdot \text{pmn}[L]$):
$$\alpha_L(d) = \frac{\delta_L \cdot \text{rms\_pre}[L]}{\bar{c}_L \cdot \|\gamma_{\text{ff}}^{(L)} \odot \hat{v}_{\text{pre}}^{(L)}\|_2}, \qquad \Delta W_{\text{down}}^{(L)}(d) = \alpha_L(d)\,\hat{v}_{\text{pre}}^{(L)}\,\hat{k}_L^\top \in \mathbb{R}^{4096 \times 11008}.$$

- **Zero Hooks:** Applied literally via `model.layers[L].mlp.down_proj.weight.data.add_(ΔW_L.to(bf16))` and restored cleanly from an fp32/bf16 clean snapshot after each arm.
- **Exact Random Match (`Random_Matched`):** Uses `3` frozen isotropic unit vectors $\hat{v}_{\text{rand},s}^{(L)}$ (`s ∈ {0, 1, 2}`) zeroed on the exact same $S_L$, the exact same key $\hat{k}_L$, and the **exact same scalar $\alpha_L(d^*)$**, guaranteeing $\|\Delta W_{\text{rand}}^{(L)}\|_F = \|\Delta W_{\text{edit}}^{(L)}\|_F = |\alpha_L(d^*)|$ to machine precision!

### 2.3 `0 CU` Frozen Lexicon Geometry (`c3_vector_bank.safetensors`, `511 KB`)
| Layer | `drop_k = 4` `min \|γ_ff\|` | Pirate Lexicon `eff` (`unmask → mask`) | Pirate Boost (`cos_ret`) | Pure-MD Anti-Write `eff` (`unmask → mask`) | Pure-MD Boost (`cos_ret`) |
|---:|---:|---|---:|---|---:|
| `L21` | `0.000254` | `0.1099 → 1.0058` | **`9.15×`** (`0.99964`) | `0.0949 → 1.0052` | **`10.59×`** (`0.99933`) |
| `L24` | `0.001564` | `0.6931 → 1.0055` | **`1.45×`** (`0.99980`) | `0.4911 → 1.0072` | **`2.05×`** (`0.99978`) |
| `L27` | `0.000148` | `0.5290 → 1.0324` | **`1.95×`** (`0.99981`) | `0.0350 → 1.0340` | **`29.50×`** (`0.99962`) |

*(Pure-Markdown seeds exclude plain `'-'` and `'*'` arithmetic operators: `['**', ' **', '##', '###', '```', '**:', ':**', ' ####', '---']`; top-10 suppressed tokens on frozen `L21`/`L24` lens are 10/10 pure markdown tokens.)*

### 2.4 Deterministic Disjoint-Reserve Commit Gate (Zero Test Peeking, `#106`)
For each of the 4 compiled arms (`WSP_Pirate_Lexicon`, `WSP_Pirate_Contrastive`, `WSP_NoMD_Lexicon`, `WSP_NoMD_Contrastive`), evaluate all `5` ladder rungs `d ∈ [0.15, 0.30, 0.45, 0.60, 0.80]` on the `8` strictly disjoint reserve prompts (`RESERVE_PROMPTS_8`, teacher-forced forward `KL(p_plain || p_edited)` over plain 48-token greedy answers + 48-token greedy broken check). Select the **highest rung `d*`** satisfying:
1. Reserve median `off_KL <= 0.25` nats AND reserve max `off_KL <= 0.60` nats, AND
2. Reserve broken rate `== 0 / 8`.

If no rung meets both bounds, select `d* = 0.15` and log `commit_gate_fallback = true`. All 5 reserve-rung curves are recorded in the receipt. Each arm's `Random_Matched` (`s0, s1, s2`) is evaluated at that arm's exact selected `d*` (reusing evaluations whenever two arms select the same `d*`).

---

## 3. Disjoint Evaluation Batteries & Mechanical / Blind-Reader Scoring

Zero exact-string overlap across all `12` splits (`16` calib, `8` reserve, `16` identity [`8` direct + `8` indirect], `16` implied, `12 × 3 = 36` multi-turn, `12` copula pairs, `24` markdown-eliciting, `20` math, `12` Python code, `8` refusal, `16` neutral chat), zero overlap with Stage C0/C1/C2 prompts, and **zero Pirate ban stem occurrences** in `IMPLIED_PROMPTS_16` or Turn 1 of `MULTITURN_12`.

- **Part 168 Decode Word-Ban (`ban = own_plus_lex`):** Union of the `127` frozen single-token Pirate vocabulary entries (`pirate_static_ban_tokens`) AND each arm's own top-50 positive mean logit-bias tokens measured teacher-forced on `calib_16` plain answers (special/EOS tokens stripped; zero test peeking). Enforced at every generation step by masking banned token logits to `-inf`.
- **Mechanical `is_broken(text)` (frozen pure-stdlib):** Empty/whitespace-only, `< 3` alphabetic characters, contains `U+FFFD` or `"!!!!"`, any 3-word n-gram repeated `>= 4` times, or any 5-word n-gram repeated `>= 2` times (aligned with Part 218 `attrib218`).
- **Mechanical `markdown_count(text)` & `has_markdown(text)` (frozen pure-stdlib):** Counts distinct regex hits for ATX headings (`^\s*#{1,6}\s+`), bold/strong (`\*\*[^*\n]+\*\*` or `__[^_\n]+__`), fenced code blocks (`` ``` ``), unordered bullet lines (`^\s*[-*+]\s+`), ordered list lines (`^\s*\d+\.\s+`), markdown table separator rows (`^\s*\|?[\s:-]+\|[\s|:-]*$`), and horizontal rules (`^\s*---+\s*$`). `has_markdown = (markdown_count >= 1)`.
- **Mechanical `math_exact(text, expected_int)`:** Parses the final integer after `FINAL:` (fallback: last integer in response) and tests exact integer equality.
- **Mechanical `code_pass(text, func_name, tests)`:** Extracts Python code (stripping any markdown fence if present), executes inside an isolated `python3 -I` subprocess (`timeout = 3 s`) against all frozen `assert` tests, requiring exit code `0`.
- **Dual Blind Readers (`audit_c3_label.py`, Part 218 protocol):** Two independent readers (`deepseek-v4.1-flash` via `ocgo` and `gpt-6-luna` via `cliproxy`), completely blind to arm names, ban status, and each other. Each reader sees only `(prompt, answer)` and outputs `{"pirate_persona": true|false, "coherent": true|false}`.
  - **Pre-Check Gold Gate:** Both readers must score **`>= 10 / 12`** on frozen `READER_GOLD_12` (`6` positive including subtle zero-pirate-word implied answers, `6` negative including 3rd-person false-friend and word-stuffing loop).
  - **Agreement Gate (`C3.R`):** Cohen's `κ >= 0.60` across the blinded evaluation pool (`Identity 16 + Implied 16` × `2` bans + `Turn-3 12`, across `Plain`, `Sys_Pirate`, `WSP_Pirate_Lexicon`, `WSP_Pirate_Contrastive`, and `Rand_Pirate_{0,1,2}`). Primary per-item persona label = strict consensus (`reader1 AND reader2`, with `not is_broken(text)` enforced so broken loops never score positive!). If `κ < 0.60`, ONLY reader-dependent verdicts (`H1–H4`) record `NOT_READ_LOW_KAPPA`; all mechanical verdicts (`H5, H6, EQ-C3.2, D1, D2`) remain fully readable!

---

## 4. Desk-Checked Gate × Condition Verification Matrix (`C3.0` & `C3.R`)

Learning directly from Stage C1 (`C1.0.3` stale cross-prompt anchors) and Stage C2 (`C2.0.2` unpassable upstream-patch telescoping pool), every gate below is verified satisfiable at the desk across every condition it covers:

| Gate | Conditions Covered | Exact Criterion | Desk Satisfiability Proof |
|---|---|---|---|
| **`C3.0.1`** Zero-hook & restore | `Plain`, `sham_alpha_zero`, all `4` WSP arms, all `6` random-seed arms | `active_hooks == 0` on all test runs; `max\|Δlogit(sham_alpha_zero)\| == 0.0`; `max\|Δlogit(post_restore)\| == 0.0` after every arm; `max\|Δlogit(active)\| > 0.0` | Exact clean-weight snapshot copy-back guarantees `0.0` restore residual; zero PyTorch hooks registered during test passes; nonzero rank-1 `ΔW` guarantees liveness. |
| **`C3.0.2`** Finite, Frobenius match, & ban | All active arms + random controls; all `ban=own_plus_lex` generations | `0` NaN/Inf in `ΔW`, logits, KL; `\| \|\|ΔW_rand\|\|_F / \|\|ΔW_edit\|\|_F - 1 \| <= 1e-3` on every `(arm, layer)`; `0` banned token IDs emitted in any banned generation | Identical `α_L` & unit vectors yield `< 1e-5` norm match; `-inf` logit masking before argmax guarantees `0` banned IDs. |
| **`C3.0.3`** Completeness & `Plain` sanity | All prompt×arm cells; `Plain` on `Identity(16)+Implied(16)+Neutral(16)` | Exact cell counts & non-empty strings; `Plain` non-broken rate `>= 0.90` | Pure completeness + baseline fluency check; zero hand-typed norm anchors, zero telescoping over patched layers. |
| **`C3.0_NoMD_room`** *(Pillar-B-local)* | `Plain` on `NOMD_24` (`24` markdown-eliciting prompts) | `Plain` `has_markdown` rate `>= 0.65` (`>= 16 / 24`) | Scoped **strictly to Pillar B (`H5`, `H6`)** so low baseline markdown cannot block Pillar A (`H1–H4`). |
| **`C3.0.4`** Strict provenance | `OLMO3-WSP-C3-DESIGN.md`, `c3_prereg_constants.json`, `c3_vector_bank.safetensors` | Exact SHA-256 match on disk and in receipt metadata | All 3 files hashed deterministically pre-launch (`511 KB` compact bank, no `2.11 GB` lens upload). |
| **`C3.R`** *(Reader-scoped)* | Reader-scored Persona hypotheses `H1–H4` ONLY | Both blind readers score `>= 10/12` on `READER_GOLD_12` AND Cohen's `κ >= 0.60` on blinded pool | Scoped strictly to `H1–H4`; mechanical `H5, H6, EQ-C3.2, D1, D2` stay readable even if `κ < 0.60`. |

---

## 5. Confirmatory Holm-6 Family (`α = 0.05`) & Separate Capability Gate `EQ-C3.2`

All `6` primary hypotheses use one-sided paired sign-flip permutation tests over independent prompt units (`100,000` draws for `N >= 24`, exact `2^12 = 4,096` enumeration for `N = 12`, seeded `PERM_SEED = 2026112601`) and `90%` paired BCa bootstrap intervals (`10,000` draws, `BCA_SEED = 2026112602`), controlled via Holm-Bonferroni (`α_k = 0.05 / (7 - k)`):

| ID | Hypothesis | Pillar & Units | Point Bar | 90% BCa Bar | Guard / Gate |
|---|---|---|---|---|---|
| **`H1`** | `C3.1a`: `WSP_Pirate_Contrastive` beats `Rand_Pirate` (`3`-seed mean) on **unbanned** Identity + Implied | A (`N = 32` prompts) | `Δrate >= +0.25` | `q05 > +0.10` | `broken <= 0.15`, `C3.R` |
| **`H2`** | `C3.1b`: `WSP_Pirate_Contrastive` beats `Rand_Pirate` (`3`-seed mean) **under Part 168 word-ban (`own_plus_lex`)** | A (`N = 32` prompts) | `Δrate >= +0.20` | `q05 > +0.08` | `broken <= 0.20`, `C3.R` |
| **`H3`** | `C3.1c`: `WSP_Pirate_Contrastive` beats `WSP_Pirate_Lexicon` **under Part 168 word-ban** | A (`N = 32` prompts) | `Δrate >= +0.15` | `q05 > +0.05` | `broken <= 0.20`, `C3.R` |
| **`H4`** | `C3.1d`: `WSP_Pirate_Contrastive` beats `Rand_Pirate` on **3-turn follow-through (Turn 3)** | A (`N = 12` convs) | `Δrate >= +0.25` | `q05 > +0.05` | `broken <= 0.20`, `C3.R` |
| **`H5`** | `C3.2a`: `WSP_NoMD_Contrastive` cuts markdown count vs. `Plain` AND beats `Rand_NoMD` (`3`-seed mean) | B (`N = 24` prompts) | `reduction >= 50%` | `q05 >= 35%` | `broken <= 0.10`, `C3.0_NoMD_room`, `EQ-C3.2` |
| **`H6`** | `C3.2b`: Pure `0`-rollout `WSP_NoMD_Lexicon` cuts markdown count vs. `Plain` AND beats `Rand_NoMD` | B (`N = 24` prompts) | `reduction >= 35%` | `q05 >= 20%` | `broken <= 0.10`, `C3.0_NoMD_room`, `EQ-C3.2` |

- **Capability Preservation Gate `EQ-C3.2` (evaluated separately per No-Markdown arm; also reported descriptively for every Persona arm):**
  1. Untuned `MATH_20` exact accuracy drop `<= 10 pp` (`<= 2 / 20` fewer solved than `Plain`), AND
  2. Untuned `CODE_12` unit-test pass-rate drop `<= 10 pp` (`<= 1 / 12` fewer passed than `Plain`), AND
  3. Untuned `NEUTRAL_CHAT_16` median forward `off_KL <= 0.35` nats AND broken rate `<= 0.10`.
- **Registered Secondary / Diagnostic Readouts (Outside Holm):**
  - **`D1` Part 213 Copula (`12` pairs):** Paired `ΔLP_self − ΔLP_stranger` on the `12` Pirate lexicon seeds across `WSP_Pirate_Contrastive`, `WSP_Pirate_Lexicon`, `Rand_Pirate_s0`, and `SYS_PIRATE`.
  - **`D2` Full Preservation & Sibling-Bleed Table:** Math, Code, Refusals (`8`, harmful refusal rate + benign non-refusal rate), Neutral Chat (`off_KL` + greedy-identical token rate), and cross-arm sibling bleed (Persona effect on `NOMD_24` markdown count; No-Markdown effect on `IDENTITY_16` pirate tokens).
  - **`D3` `System_Prompt` Reference Ceilings & 3-Column Demo Export:** Held-out `12`-prompt text-only showcase (`DEMO_SHOWCASE_IDS_12`: `Plain` vs. `WSP_Pirate_Contrastive` vs. `Rand_Pirate_s0`, both `ban=none` and `ban=own_plus_lex` — zero edited weights exported).
  - **`D4` Unmasked vs. Masked `L24` Literal Weight Preflight (`24` short forwards, `< 3 s`):** Verifies on-model delivered logit shift ratio of masked `L24_pirate_lex_pre_unit` vs. unmasked `L24_pirate_unmasked_pre_unit` at matched Frobenius norm.

---

## 6. Compute Estimate (Single Ephemeral NVIDIA L4, Zero `r_lens.pt` Upload)

- **Upload:** `< 1 MB` (`511 KB` vector bank + runner script → `< 5 s`, vs. `399 s` in C2!).
- **Workload (`B = 1` greedy `use_cache=True`):** Calibration `48` short gens (`32` tok); Reserve commit-gate `160` short forwards + `160` gens (`48` tok); Pillar A `32 × 2 × 7 = 448` gens (`64` tok) + 3-turn `12 × 3 × 5 = 180` gens (`64` tok) + Copula `120` short forwards; Pillar B `24 × 7 = 168` gens (`96` tok); Shared Capability Suite `(20 + 12 + 8 + 16) × 7 = 392` gens (`64–96` tok) + `112` teacher-forced KL forwards; `D4` + shams `48` short forwards.
- **Estimated Wall-Clock & Cost:** `~20–28 min` on NVIDIA L4 (`~0.75–1.15 CU` total; well within `acct1`'s `165.10 CU` balance).
