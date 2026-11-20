# Part 214: why do factual donor values trigger Qwen3.5-4B's Google fallback when `"you"` donor values don't? Testing a single partial-zero dial against phase, spike, and register accounts

Procedure: v1
DRAFT v1 2026-11-19 18:46 CT by Gemini 4 Neon (proposer and designer; requested by Journal Claude / Elise following
Part 212's published results, 2026-11-19 18:40 CT; awaiting Alden cold design review, Stage 0 descriptive donor
recording to pin the scalar dial values, and one code review of `part214/run214.py` / `part214/score214.py` before
registration on public main via `tools/register_public.py`).
All predictions below (`214.1`–`214.8`) are authored before Stage 0, Stage 1, or Stage 2 runs; Stage 0 records only
descriptive greedy donor/maker activations and donor texts and produces zero Stage 2 answers.

## Question

In Part 212 (`results/part212/README.md`), intervening on Qwen3.5-4B's four maker neurons (`L6 N753`, `L7 N1169`,
`L11 N8868`, `L29 N776`) moved held-out English clean Google counts across a wide ladder: `unchanged` 4/192,
`sub_factual` 8/192, `trace_resample_you` 9/192, `trace_resample_fact` 19/192, `mean_resample_fact` 31/192, `zero`
36/192, and `scale_neg1` 110/192 (with `212.1` failing because `unchanged` was 4 against a bar of 3, withdrawing
Part 212's specificity premise). Two puzzles stand out in the Part 212 records:

1. **Stage 1 vs. Stage 2 divergence:** On Stage 1's shared neutral opening, `mean_resample_fact` (`+2.64`),
   `trace_resample_fact` (`+2.60`), and `trace_resample_you` (`+2.27`) sat within `0.37` logits of each other, and in
   Stage 2 all three opened with `"I am (**)?Qwen"` at nearly the same rate (`142`, `144`, `146` of 192, vs. `167`
   unchanged and `123` zero). Yet on the remaining ~46–50 non-Qwen generations, Google fired on `31/50` (62%) for
   `mean_resample_fact`, `19/48` (40%) for `trace_resample_fact`, and only `9/46` (20%) for `trace_resample_you`
   (and `6/24` vs. `0/24` on `"Who created you?"`).
2. **The partial-zero norm clue (H4):** In Part 212, `sub_factual`'s realized residual perturbation norm was only
   `0.2127` against `zero`'s `0.8888`. **That is a norm clue, not a proof** (Alden, 2026-11-19): donor vectors of
   equal or smaller norm can be parallel, orthogonal or antiparallel, so a small residual bounds how much changed, not
   whether what changed is a scaled copy. H4's real premise is the projection $\hat{\alpha}$ defined in Stage 0, and
   214.2 is its test. The schedule is also **two-phase** (a separate $\hat{\alpha}_{\text{prompt}}$ on prefill and
   $\hat{\alpha}_{\text{gen}}$ on the opening decode steps), so it is not one degree of freedom spanning both.

**Primary question (H4 as the explanation the rivals must beat):** Does a **shared-across-units, two-phase
partial-zero schedule** ($\mathbf{H4}$: one fitted $\hat{\alpha}_{\text{prompt}}$ multiplying $a_{\text{live}}$ on
prefill positions and one fitted $\hat{\alpha}_{\text{gen}}$ on the opening decode steps, the same value for all four
units)
account for both factual resampling's Google rate and the `fact` vs. `"you"` gap—or do rival mechanisms beat H4:
**decode-phase non-scalar content** ($\mathbf{H1}$), **prompt-phase KV carry-over** ($\mathbf{H2}$), **early-token
temporal spikes lost by mean pooling** ($\mathbf{H3}$), or **polite helper-request register without `"you"`**
(`D_imp`)?

## Prior art and existing data

- Closest published work (searched and checked at source before drafting predictions):
  - **Zhang & Nanda (ICLR 2024, arXiv:2309.16042), "Towards Best Practices of Activation Patching in Language Models:
    Metrics and Methods"**: showed that corruption choice (zeroing / Gaussian noise vs. counterfactual token
    replacement) and readout metric strongly shift localization results because harsh ablations push downstream
    representations off-distribution.
  - **Heimersheim & Nanda (2024, arXiv:2404.15255), "How to use and interpret activation patching"**: codified noising
    vs. denoising asymmetry and warned that discrete accuracy metrics can conceal continuous logit-scale thresholds and
    cooperative/OR-gate circuits.
  - **Miller, Chughtai & Saunders (2024, arXiv:2407.08734), "Transformer Circuit Faithfulness Metrics are not Robust"**:
    demonstrated that circuit faithfulness varies widely across ablation baselines (zero vs. mean vs. resample
    ablation), because different baselines preserve different task-invariant baseline components.
  - **Li & Janson (2024, arXiv:2409.09951), "Optimal Ablation for Interpretability"**: formalized replacing zero or
    mean baselines with an optimized constant vector to minimize off-manifold disruption.
  - **Yavuz (2026), "Are Causal Neuron Circuits Unique and Transferable?"** (`github.com/omtekyav/causal-neuron-circuits`,
    `PRIOR-ART.md` 2026-11-09) and **Tuomi (2026)** (`results/part206/README.md`): showed that zeroing resting-active
    neurons injects $-\bar{a}\cdot w_{\text{down}}$ and can look transferable when resample ablation removes the effect.
  - **What this part adds:** None of the works above tests whether a resample-ablation contrast between two donor banks
    (`D_fact` vs. `D_you`) reduces to a **projection-matched 1-D scalar partial-zero dial** ($\hat{\alpha}\cdot a_{\text{live}}$)
    during multi-token free generation, nor separates prompt-prefill vs. decode-step resampling (`prompt_fact_gen_you`),
    flat-mean vs. per-token temporal spikes (`mean_resample_you` vs. `trace_resample_you`), and polite-request register
    without `"you"` (`D_imp`) on the same four-neuron circuit.
- Our own data that already bears on it:
  - **Part 212 (`results/part212/README.md`, `results/part212/report212.md`, `results/part212/stage2/stage2.jsonl`)**:
    established the full ladder (`zero` 36, `mean_resample_fact` 31, `trace_resample_fact` 19, `trace_resample_you` 9,
    `sub_factual` 8, `unchanged` 4 of 192; `zero` positive control 10/24, `trace_resample_fact` 6/24,
    `mean_resample_fact` 5/24, `trace_resample_you` 0/24), superadditive joint behavior of `(L6 N753, L7 N1169,
    L11 N8868)` (`zero` sum `+1.79` vs. joint `+3.10`; `fact - you` sum `+0.033` vs. joint `+0.330`; `L29 N776` alone
    `+0.011`), `L27 N3325` tracking (`Spearman -0.9879`), the `"I am Qwen"` opening split (`167, 123, 142, 144, 146`
    of 192), the persistence of the Stage 2 `fact` vs. `you` gap on the 19 topic-bound prompts (`18/152` vs. `8/152`),
    and measured A770 batch-1 throughput (**`4.80 s/call`** in Stage 2 and **`0.333 s/forward`** in Stage 1).
  - **Part 75 (`results/part75/`)**: showed flat prompt-mean clamping of `L27 N3325` is non-neutral because natural
    activations vary across positions—motivating the direct `mean_resample_you` vs. `trace_resample_you` H3 check.
  - **Part 172 & Part 183 (`results/part172/`, `results/part183/README.md`)**: Part 172 swept coarse knockout fractions
    ($0.5, 0.75, 1.0$) without projection-matching to donor banks; Part 183 established eager-attention kernel safety
    (`results/part183/kernel-check.json`).
  - **Seen before writing:** Part 212's published results (`stage1.json`, `stage2.jsonl`, `score.json`, `report212.md`)
    and tokenizer-only verification of `part214/prompts214.json` via `part214/build_prompts214.py`. No Part 214 model
    runs, Stage 0 donor recordings, Stage 1 logits, or Stage 2 generations have been executed yet.

## Method

- **Model, units, and rig:** `Qwen/Qwen3.5-4B`, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  `torch.bfloat16`, Intel Arc A770 (`xpu`), `attn_implementation="eager"` via `neuron_report.load_model`, thinking off
  (`nr.build_prompt(tok, q, False)`), no system prompt. Target units `TARGET = [(6, 753), (7, 1169), (11, 8868),
  (29, 776)]`, intervened via forward pre-hooks on `layers[L].mlp.down_proj`; downstream monitors `L24 N3949` and
  `L27 N3325` recorded without intervention in Stage 1.
- **Prompts and new `D_imp` bank (`part214/prompts214.json`, SHA-256 `76005231b171babdeed9318e1912703ce55b9d9d9876c906360535573d3f55d6`):**
  - Reuses Part 212's exact `EN_MAKER[0..24]` (24 held-out English maker phrasings + `"Who created you?"`), `D_fact`,
    `D_you`, and 6 fluency guards (`prompts212.json` SHA-256 verified by `part214/build_prompts214.py`).
  - Adds **`D_imp`** (25 polite imperative requests, e.g. `"Please briefly explain gravity."`, `"Please help me."`):
    matched 1-to-1 to `D_you` in topic, request shape (including the 6 bare requests at indices `1, 2, 3, 7, 11, 24`),
    and **exact Qwen3.5-4B token length**, with **zero** second-person pronouns (`you`/`your`/`yourself`) and zero
    maker/AI/company/China/Google words (Gate 0 verified: 25/25 exact matches, 0 lexical leaks, 0 boundary merges).
    `D_imp` vs. `D_you` isolates the pronoun `"you"` at fixed request framing; `D_imp` vs. `D_fact` isolates polite
    request framing at zero `"you"` pronouns.

### Stage 0: Descriptive donor & unchanged-maker recording (may run before registration; zero Stage 2 answers)

- Run greedy generation (`max_new_tokens=48`, eager, bf16, batch 1) once per prompt on all 4 banks (`EN_MAKER`,
  `D_fact`, `D_you`, `D_imp` = **100 greedy runs**, ~8 min), recording `TARGET` (plus Stage 1's 4×50 write-norm pool
  units on `D_fact` for Stage 1's 10 matched random sets) at every prefill position `0..T_q - 1` and every processed
  non-EOS generated position `0..L_gen - 1` (`L_gen = min(n_non_EOS, 47)`, excluding the uncaptured 48th token, matching
  Part 212 v3).
- **Gate 3b (donor length):** every prompt across `EN_MAKER`, `D_fact`, `D_you`, `D_imp` must have `L_gen >= 5` non-EOS
  tokens (Part 212's minimum was 23), ensuring `donor_gen[0..4]` exists for Stage 1 and decode traces exist for Stage 2.
- **Deterministic H4 dial projection formula (computed from Stage 0 over held-out indices `0..23`):**
  Let $w_k = \|W_{\text{down}}^{(L_k)}[:, u_k]\|_2$ be the L2 write-vector norm of target unit $k \in \{0..3\}$. For
  each donor bank $D \in \{\text{fact}, \text{you}, \text{imp}\}$ (and similarly for bank means $\bar{a}_D$), define
  the write-norm-weighted effective retention fraction onto $a_{\text{maker}}$:
  $$\hat{\alpha}_{\text{prompt}}(D) = \operatorname{clip}_{[-1, 2]}\!\left(\frac{\sum_{q=0}^{23} \sum_{p=0}^{T_q-1} \sum_{k=0}^{3} w_k^2 \, a_{\text{maker}, k}(q, p) \, a_{D, k}(q, p)}{\sum_{q=0}^{23} \sum_{p=0}^{T_q-1} \sum_{k=0}^{3} w_k^2 \, a_{\text{maker}, k}(q, p)^2}\right)$$
  $$\hat{\alpha}_{\text{gen}}(D) = \operatorname{clip}_{[-1, 2]}\!\left(\frac{\sum_{q=0}^{23} \sum_{j=0}^{M_q-1} \sum_{k=0}^{3} w_k^2 \, a_{\text{maker}, k}^{\text{gen}}(q, j) \, a_{D, k}^{\text{gen}}(q, j)}{\sum_{q=0}^{23} \sum_{j=0}^{M_q-1} \sum_{k=0}^{3} w_k^2 \, a_{\text{maker}, k}^{\text{gen}}(q, j)^2}\right)$$
  where $M_q = \min(L_{\text{maker}, q}, L_{D, q})$ aligns decode step $j$ to decode step $j$. Also log raw unclipped
  values, per-unit projections $\hat{\alpha}_{\text{phase}, k}(D)$, orthogonal residual fraction
  $\rho_\perp(D) = \|a_D - \hat{\alpha}(D) a_{\text{maker}}\|_w / \|a_{\text{maker}}\|_w$, early (`gen[0..9]`) vs.
  late (`gen[10..]`) activation profiles, and all 100 greedy donor texts in `results/part214/stage0/stage0.json`.
- **Pre-registration dial pinning rule (deterministic, zero post-hoc tuning):**
  - `scale_dial_fact` always uses $(\hat{\alpha}_{\text{prompt}}(\text{fact}),\, \hat{\alpha}_{\text{gen}}(\text{fact}))$.
  - If $|\hat{\alpha}_{\text{gen}}(\text{you}) - \hat{\alpha}_{\text{gen}}(\text{fact})| \ge 0.05$, `scale_dial_you`
    uses $(\hat{\alpha}_{\text{prompt}}(\text{you}),\, \hat{\alpha}_{\text{gen}}(\text{you}))$.
  - **Projection-denominator gate (added at Alden's hold; extended at his second hold).** $\hat{\alpha}$ exists only when
    the weighted maker energy $\sum_{k} w_k^2 a_{\text{maker},k}^2$ over the same positions is finite and `>= 1e-6`;
    below that it is `undefined`. A negative or clipped $\hat{\alpha}$ is logged as a signed control and is never called
    an "ordinary retained fraction".
    - **`you` undefined** → `scale_dial_you` is not run and 214.4(b) is `not read (no projection denominator)`.
    - **`fact` undefined** (the schedule the headline hangs on) → **no dependent dial arm is launched at all**: both
      `scale_dial_fact` and `scale_dial_you` are skipped, 214.4 is `not read (no projection denominator on the fact
      schedule)`, and no arm is scored. Running `you` alone, or landing the eight-arm contract with a missing arm, has no
      valid interpretation (Alden).
    - **No missing-arm PASS, ever, and no silently shrunk manifest.** A `not read` arm can never count as a clause
      passing, and the scorer must refuse to emit a verdict that depends on an arm it did not score. When a fit is
      undefined the report names what stopped rather than absorbing the loss: **`fact` undefined** stops Stage 1's dial
      curve and matched-random check, both Stage 2 dial arms, and every Stage 2 prediction that reads them, with a
      prediction for `fact`; **`you` undefined** stops `scale_dial_you` and the clauses that read it (214.4(b), and the
      `dial_you_ref` bridge), with everything else running. Either way the affected predictions are reported as an
      explicit NON-PASS, and the `eight arms / 1,584 calls` contract is measured against the **eligible full run** — every
      arm scored or explicitly `not read`/`incomplete (deadline)` — never against a manifest that quietly got smaller.
    - **Unavailable, damaged, or mismatched — ONE decision order** (Alden's second correction: my earlier text let a
      recode failure count as a magnitude miss, which would credit "missed its bar" to a schedule that matched exactly).
      214.4 evaluates in this order and stops at the first that applies:
      1. **Availability.** Undefined projection, collapsed fit, or donor denominator `< 0.03` → `not read` or
         `not discriminated`; nothing is scored on that branch.
      2. **Validity and recode.** The numerical clauses pass, but the worst-case recode or a validity check on the arm's
         answers fails → **`not discriminated from damage`**. The schedule reproduced the measured magnitude and the
         answers are not trustworthy enough to credit it, so this state feeds NEITHER the "accounts" headline NOR a
         rival's win. Required fixture, Alden's counterexample: baseline `4/192`, `zero 36/192`, trace `19/13`,
         dial exactly `19/13`, hybrid `13`, only `scale_dial_you` carrying `6` broken rows → (a), (c) and the phase bar
         pass, the strict recode fails at `13 + 6 = 19`, and the correct verdict is `not discriminated from damage`, not
         a magnitude miss.
      3. **Numerical mismatch.** A valid measured miss on eligible, valid, recode-clean answers. This is the ONLY state
         that can feed a rival's win, and only for a discriminated reason on (a) or (b) — never a lone (c) miss, which is
         the point-tolerance screen.
  - If $|\hat{\alpha}_{\text{gen}}(\text{you}) - \hat{\alpha}_{\text{gen}}(\text{fact})| < 0.05$ (both banks collapse
    onto the same decode-phase point), `scale_dial_you` is reassigned **before registration** to the fixed midpoint rung
    `(prompt=0.50, gen=0.50)` (`scale_050`) so the Stage 2 decodes still exist, and **214.4(b) is marked
    `not discriminated`** — never `failed`, and no rival is credited with beating H4 on it (Alden: a collapsed fit
    cannot separate the banks' decode phases, and `.40` vs. `.44` can coexist with very different prompt alphas).
    Precedence, fixed here: `not discriminated` or `not read` outranks an automatic fail, and a rival's positive bar
    may not be read from this branch at all.
  - Exact numeric values of $(\hat{\alpha}_{\text{prompt}}, \hat{\alpha}_{\text{gen}})$ from Stage 0 will be pasted
    into this section before `tools/register_public.py` is called, without changing any prediction or threshold.

### Stage 1: Cheap exact-logit H4 curve, natural-decode bridge & mechanism audit (batch 1, `1,624` forwards, ~9 min)

- Scored on `EN_MAKER[0..24]` + `'I was developed by **'` (+ 3 appended continuation tokens for `g_seq`, hooks active
  only on `0..T+4`, exact Part 212 readout $g_{\text{ali}} = \text{logit}(\text{Google}) - \text{logsumexp}(\text{logit}(\text{Al}), \text{logit}(\text{T}))$,
  secondary $g_{\text{seq}}$, $g_{\text{al}}$, 8-maker mass, `L24 N3949`, `L27 N3325`, and realized L2 perturbation norm).
- **Pre-pass & unchanged recordings:** `24` write-norm pre-pass forwards + `4 × 25 = 100` unchanged recordings
  (`EN_MAKER` + `'I was developed by **'`; `D_fact`, `D_you`, `D_imp` + `'The main answer is **'`).
- **60 intervened arms × 25 prompts = `1,500` batch-1 forwards (total Stage 1 = `1,624` forwards):**
  1. **Core & 10-point online H4 scalar dial curve (13 arms):** `unchanged` ($\alpha=1.0$, also records monitors),
     `self_patch` (Gate 1 frozen $a_{\text{clean}}$), 9 fixed online scalar rungs $\alpha \in \{-0.25, 0.0\text{ (`zero`)},
     0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80\}$ (applied as $\alpha \cdot x_{\text{live}}$ across `0..T+4`),
     `scale_dial_fact`, and `scale_dial_you` (online $(\hat{\alpha}_{\text{prompt}}, \hat{\alpha}_{\text{gen}}) \cdot x_{\text{live}}$,
     applying $\hat{\alpha}_{\text{prompt}}$ on `0..T-1` and $\hat{\alpha}_{\text{gen}}$ on opening `T..T+4`).
  2. **Frozen-vs-online & 4-D per-unit dial diagnostics (4 arms):** `scale_dial_fact_frozen`, `scale_dial_you_frozen`,
     `scale_unit_dial_fact` (online per-unit $\hat{\alpha}_{\text{phase}, k}(\text{fact}) \cdot x_{\text{live}, k}$),
     and `scale_unit_dial_you`.
  3. **Natural-decode donor bridge (`prompt = Stage 0 donor_prompt`, `opening 0..4 = Stage 0 greedy donor_gen[0..4]`
     — 10 arms).** These arms DO change the prefix: they place the donor's first five tokens, so the model's own opening
     is not used. ⚠ **Binding rule, rewritten at Alden's second hold — two groups, never mixed:**
     - **Same-prefix group (the ONLY donor-vs-dial comparison set):** the eight Stage 2 main arms, `scale_unit_dial_fact`,
       `scale_unit_dial_you` and the 30 matched random controls. Each starts from the SAME INITIAL ENGLISH PROMPT TOKENS
       and then generates freely, so their sampled continuations — and therefore the opening and the decoded positions —
       diverge from arm to arm and call to call (Alden; the earlier text wrongly called the opening fixed). Held fixed is
       the starting prompt, the recipient-token rule, the `unchanged` baseline and the scoring-position function, not the
       tokens the model happens to emit; only the ACTIVATIONS assigned to those positions differ.
     - **Stage 1's curve and its matched-random check run teacher-forced at the FIXED maker opening
       (`"I was developed by **"`)**, so every rung and every control is scored at one identical position and the rung
       comparisons are exact.
     - **Prefix-change group (diagnostics, never in that comparison):** every `nat_*` arm above, because it forces the
       donor's opening tokens, and every `rep_*` arm in item 4, because it uses Part 212's neutral opening
       `'The main answer is **'`. Each is scored at the position its own opening produces, and is compared only within its
       own group — for `rep_*`, against Part 212's published numbers.
     - The Stage 1 dial curve (`214.3`) is a teacher-forced forward at the natural-generation opening, so it enters
       neither generation comparison.
  4. **Part 212 neutral-opening replication (`opening = 'The main answer is **'` — 3 arms, a prefix-change diagnostic
     group as above):** `rep_trace_fact`, `rep_trace_you`, `rep_trace_imp`.
  5. **Matched random controls (30 arms):** 10 maker-activation-write-norm-matched random 4-unit sets (`random_0..9`,
     exact Part 212 seed `21200 + k` from the 50-unit per-layer write-norm pools) × 3 arms (`zero`, `scale_dial_fact`,
     `nat_trace_fact`).

### Stage 2: Uniform batch-1 held-out free generation (8 main arms = `1,536` held-out + `48` gate/control = `1,584` calls, ~2.11 h)

- Sampled generation (`temperature=0.7`, `top_p=0.95`, `max_new_tokens=48`, **uniform batch size 1 across every single
  call** → zero padding, zero batch-shape heterogeneity across arms, exact SHA-256 per-row seeds matching Part 66/212).
- **Exact prefill/decode hook indexing (Part 212 v3 contract):**
  - Call 0 (prefill, `x.shape[1] == T`, predicts answer token 0): `trace_*` clamps to `donor_prompt`; `mean_resample_you`
    clamps to slot-mapped `mu_you_prompt`; `prompt_fact_gen_you` clamps to `d_fact` prompt; `scale_dial_*` multiplies
    `x_live` by $\hat{\alpha}_{\text{prompt}}$.
  - Call $s \in \{1..47\}$ (decode step processing generated token $s - 1$): `trace_*` clamps to
    `donor_gen[(s - 1) % L_gen]`; `mean_resample_you` clamps to flat `mu_you_gen`; `prompt_fact_gen_you` clamps to
    `d_you` `donor_gen[(s - 1) % L_gen]`; `scale_dial_*` multiplies `x_live` by $\hat{\alpha}_{\text{gen}}$.
- **8 main held-out arms (24 held-out English prompts × 8 samples = `192` answers each → `1,536` held-out answers):**
  1. `unchanged` (192) — within-run batch-1 baseline anchor
  2. `zero` (192) — within-run $\alpha = 0$ dial ceiling anchor
  3. `trace_resample_fact` (192) — `(prompt=fact, gen=fact)` natural donor trace
  4. `trace_resample_you` (192) — `(prompt=you, gen=you)` natural donor trace
  5. `scale_dial_fact` (192) — **H4 primary test (fact):** online scalar dial at `D_fact`'s Stage-0 projection
  6. `scale_dial_you` (192) — **H4 primary test (you):** online scalar dial at `D_you`'s Stage-0 projection (or `scale_050` if $|\Delta\hat{\alpha}_{\text{gen}}| < 0.05$)
  7. `prompt_fact_gen_you` (192) — **H1 vs. H2 phase-decomposition arm:** `(prompt=fact, gen=you)` (together with arms 3 and 4, partitions 100% of `fact - you` into decode share + prompt share!)
  8. `mean_resample_you` (192) — **H3 spike-vs-mean arm:** `(prompt=mu_you_prompt, gen=mu_you_gen)`
- **Within-run gate & positive-control rows (`48` calls):**
  - `unchanged` on Part 66's 6 fluency guards × 4 samples = **`24`** calls.
  - `zero` on `"Who created you?"` (index 24) × 24 samples = **`24`** calls.
- **Total Stage 2 = `1,584` batch-1 generations (`generation_batch_histogram: {"1": 1584}`).**

Kernel check: results/part183/kernel-check.json

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** Stage 1 sweeps the full online scalar dial from `-0.25` through
  `0.0`, `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, `0.65`, `0.80`, to `1.0` (`unchanged`), testing monotonic behavior across
  the origin; Part 212 already established `+2×` (`-0.87` logits, 1/192 Google) and `-1×` (`+7.91` logits, 110/192).
- **Poles (#2):** Not used as a unit-selection step; fixed 4-unit target set from Parts 25/66/183/212.
- **Do-nothing hook (#3):** `self_patch` in Stage 1 (`|mean g_ali(self_patch) - mean g_ali(unchanged)| < 0.05` logits)
  and clean hook removal verified between Stage 2 calls.
- **Matched random controls (#4, Toolkit I):** 10 write-norm-matched random 4-unit sets (`random_0..9`, Part 212 seed
  `21200 + k`) run in Stage 1 under `zero`, `scale_dial_fact`, and `nat_trace_fact` (`750` forwards). Stage 2 does NOT
  repeat Part 212's 1,440 random-set generations (which already established 0–2/24 on all 20 random zero sets and
  0–1/24 on random `sub_factual`, and would add ~2 hours at batch 1); Part 214 tests mechanisms *within* the fixed
  4-unit set against within-run `unchanged` and `zero` anchors.
- **Answer-polarity split (#5):** Not used: outcomes are 5-state maker classifications and continuous `g_ali`/`g_seq`
  logits (with 8-maker probability mass checked in Gate 1).
- **Broken answers, exclusive 5-state coding & worst-case recodes (#6):** Exact Part 212 v3 rules: fixed denominator
  192 on all 8 main arms; exclusive categories (`broken` first, then earliest-starting Part 66 maker match:
  `clean_google`, `clean_alibaba`, `clean_other_maker`, `clean_no_maker`), `legacy_any_google` and `multi_hit` reported
  descriptively; `broken_rate <= 0.15` required for every arm entering a headline; worst-case broken recode applied
  whenever a lower Google count is cited for protection (`clean_google + broken` of the lower arm must remain strictly
  below `clean_google` of the higher arm). Also report `"I am (**)?Qwen"` opening rates and conditional
  `Google | non-Qwen` rates descriptively for every arm.
- **Size first (#7):** Every pass multiplied out in `Size` below using Part 212's measured batch-1 A770 timings, with a
  pilot stopping rule before full Stage 2.
- **Dose ladder (#8, Toolkit E):** 10-point scalar dial curve in Stage 1 (`-0.25` to `1.0`) plus Stage-0-matched
  `scale_dial_fact` and `scale_dial_you` in Stage 1 and Stage 2.
- **Prompt vs. generation phase (Toolkit F):** Exact 2×2 phase cross in Stage 1 (`nat_prompt_fact_gen_you`,
  `nat_prompt_you_gen_fact`) and within-run phase triangle in Stage 2 (`trace_resample_fact`, `prompt_fact_gen_you`,
  `trace_resample_you`).
- **Hand-read check:** Up to 30 Google-classified held-out answers per non-baseline main arm audited by eye (maker claim
  vs. negation/mention); if >20% are non-claims in any headline arm, the headline is qualified with the audited claim
  rate.

Gates (no gate shares the statistic of any prediction it guards):

- **Gate 0 (tokenizer & prompt parity, passed pre-registration):** `part214/build_prompts214.py` verifies `prompts212.json`
  SHA-256, 25/25 exact Qwen3.5-4B token-length matches for `D_imp`, 0 lexical-screen leaks, and 0 chat boundary merges
  (`prompts214.json` SHA-256 pinned above).
- **Gate 1 (Stage 1 do-nothing hook, finite values & logit mass):** `|mean g_ali(self_patch) - mean g_ali(unchanged)| < 0.05`
  logits; zero non-finite logits or activations; mean 8-maker probability mass `>= 0.50` in `unchanged` and `zero`.
- **Gate 2 (Stage 1 positive control):** `zero` raises mean `g_ali` over `unchanged` by `>= +1.0` logit and exceeds all
  10 matched random `zero` sets; failure stops Stage 1 mechanism readings (`214.3`, `214.7`, `214.8`).
- **Gate 3 (Stage 2 kernel, uniform batch-1 shape, completeness, donor length & baseline sanity):** `run.json` records
  `attn_implementation == "eager"` and `generation_batch_histogram == {"1": 1584}` (pilot: `{"1": 64}`), zero empty or
  non-finite rows, every planned key present once, every Stage 0 donor in `EN_MAKER`/`D_fact`/`D_you`/`D_imp` has
  `L_gen >= 5` (**Gate 3b**), and `unchanged` achieves guard accuracy `>= 0.90`. **Three distinct integer maxima** (Alden: the decimals and the counts
  disagreed and one count was attached to the wrong state): on the `192` scored `unchanged` rows,
  **`clean_google(unchanged) <= 11/192`** (`0.0573`; `12/192 = 0.0625` exceeds the `<= 0.06` bound) and
  **`broken(unchanged) <= 9/192`** (`0.0469`; `10/192 = 0.0521` exceeds `<= 0.05`); on each of the four main scored arms,
  **`broken <= 28/192`** (`0.1458`; `29/192 = 0.1510` exceeds `<= 0.15`). These are the sanity bounds that catch a
  broken baseline without Part 212's `<= 3/192` trap.

## Size

- Smallest effect worth knowing: a `>= 5` percentage-point difference between an arm and its scalar-dial counterpart
  (`±0.05` equivalence band for H4), or a `>= 3` percentage-point within-run `fact - you` / phase-swap shift on the
  fixed budget of $n = 192$ held-out answers per arm (24 questions × 8 samples, evaluated with 10,000-draw
  question-clustered bootstrap 95% CIs, seed 214; no iid McNemar power claim asserted). In Stage 1, 24 held-out prompts
  resolve ~0.25-logit shifts (Part 74/212 SE ~0.08–0.12 logits).
- Total: 1,584 batch-1 sampled generations in Stage 2 (plus 100 Stage 0 greedy donor recordings and 1,624 Stage 1
  batch-1 forward passes), about 2.4 hours on the A770 (Stage 0 ~8 min + Stage 1 ~9 min + Stage 2 ~2.11 h; the 1,536
  held-out Stage 2 calls alone take ~2.05 hours at 4.80 s/call).
- Pilot: Stage 1 (`1,624` forwards, ~9 min, reusing Stage 0's saved donor recordings) + Stage 2 mini-pilot written to
  `results/part214/pilot/` (`24` `"Who created you?"` rows on `zero` + `24` fluency guards on `unchanged` + `16`
  held-out rows across prompts `0, 5, 12, 16` × 1 sample × 4 arms [`unchanged`, `zero`, `scale_dial_fact`,
  `trace_resample_fact`] = **`64`** batch-1 calls, ~5 min; ~14 min total). Stopping rule: do not launch full Stage 2 if
  Gate 1, Gate 2, or Gate 3b fails, if any pilot row is empty or non-finite, if `run.json` does not record `eager` and
  `{"1": 64}`, if `unchanged` guard accuracy `< 0.90` or broken rate `> 0.05`, if `zero` gives `< 2/24` clean Google on
  `"Who created you?"`, or if measured batch-1 latency exceeds `7.5 s/call` (full Stage 2 > 3.3 h). Pilot outputs are
  NOT reused in the clean full Stage 2 run (`results/part214/stage2/`). **Pilot gate rows, defined (Alden):** the pilot's
  `unchanged` checks are computed on the pilot's own `24` fluency-guard rows ONLY, as integer rules (`broken <= 1/24`,
  `clean_google <= 1/24`, guard accuracy `>= 22/24`); pilot rows are never pooled into a full-run number, and every Gate 3
  maximum above is computed from the `192`-row main-arm counts only.

## Optimization

- Bound resource: GPU (Intel Arc A770, 16 GB VRAM; single model load per job so host RAM stays safely above the 2.0 GB
  floor — Part 212 pilot lesson commit `4e98ffb1`; launched via `tools/launch_arc.py` with copy engine off when no other
  GPU or heavy WSL job is active).
- Measured: Part 212 on this exact model, revision, A770 card, eager bf16, and 48-token cap measured **0.21 answers/s
  at batch 1** (`4.80 s` per batch-1 Stage 2 generation, `results/part212/README.md`, Amendment 2) and **3,449 forwards
  in 1,148.74 s** (`3.00 forwards/s` / `0.333 s` per batch-1 Stage 1 forward, `results/part212/report212.md`).
- Levers (from `PROCEDURE.md` section 4):
  - **Batch size:** uniform **batch 1** taken intentionally for every Stage 1 and Stage 2 call — eliminates Part 212's
    Amendment 1 batch-shape discrepancy and Part 201 bf16 batch-size tie drift across arms, while fitting comfortably in
    ~2.1 hours for 1,584 Stage 2 calls.
  - **Teacher-forced scoring instead of generation:** taken — Stage 1 runs the full 10-rung H4 dial curve, 4-D per-unit
    dial, frozen-vs-online check, `D_imp` register check, 3-unit `no_L29` check, and 10 matched random sets in ~9 min
    before Stage 2.
  - **Generate only what is read:** taken — cut Stage 2 from Part 212's 3,888 calls down to 1,584 by omitting redundant
    Stage 2 random sets (already tested in Part 212) and replacing a 2-arm phase cross with a single phase-triangle arm
    (`prompt_fact_gen_you`) that partitions 100% of the within-run `fact - you` gap.
  - **max_new_tokens:** taken — capped at 48 tokens (exact Part 66/183/212 budget).
  - **Cache and resume:** taken — Stage 0 donor traces saved once (`results/part214/stage0/`) and reused by Stage 1 and
    Stage 2; Stage 2 appends JSONL rows keyed by `(arm, question_index, sample, pilot)` with atomic partial-line repair
    and Windows binary lock files (`O_BINARY`, Part 212 fix `30db7048`).
  - **Single model load / vectorized hooks / CPU logits:** taken — one model load per run, on-device tensor indexing in
    hooks, float32 logit math at last position only.
  - **Labelling API:** n/a — fixed lexical + mechanical broken-answer scorer audited by eye, zero API cost.
  - **Device:** A770 local (cost $0; full run ~2.4 hours).

## Predictions

All predictions authored by Gemini 4 Neon, written before Stage 0, Stage 1, or Stage 2 runs. Every Stage 2 held-out main
arm uses fixed denominator 192 (`rate(arm) = count(clean_google) / 192`, exclusive 5-state category) and anchors every
baseline comparison on THIS run's measured `unchanged` rate with a stated margin. Every bootstrap CI is a 10,000-draw
question-clustered bootstrap (seed 214).

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 214.1 | Within-run baseline-anchored batch-1 replication (Stage 2): `zero` (`broken_rate <= 0.15`) exceeds THIS run's measured `unchanged` clean Google rate by at least `+0.10` (10 percentage points, with question-clustered 95% CI lower bound strictly `> 0`), AND `trace_resample_fact` (`broken_rate <= 0.15`) exceeds `unchanged` by at least `+0.04` AND exceeds `trace_resample_you` (`broken_rate <= 0.15`) by at least `+0.03`. Fails if any clause misses. | high | Gemini 4 Neon |
| 214.2 | Stage 0 **calibration observed at registration** (greedy recordings over the 24 held-out prompts; reported as a premise measured before any Stage 2 answer exists, not as a blinded post-registration result): `D_you`'s decode-phase write-norm-weighted projection onto `EN_MAKER` exceeds `D_fact`'s by at least `+0.08` (`hat_alpha_gen(you) - hat_alpha_gen(fact) >= +0.08`), AND `D_fact` stays in the near-zero partial-zero regime (`hat_alpha_gen(fact) <= 0.35`). Fails if either clause misses. | medium | Gemini 4 Neon |
| 214.3 | Stage 1 H4 1-D single-dial test (exact logits, teacher-forced at the fixed maker opening `"I was developed by **"`): across the 10-point online dial curve $\alpha \in \{-0.25, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0\}$, mean $\Delta g_{\text{ali}}(\alpha)$ decreases monotonically with $\alpha$ (Spearman $\rho \le -0.95$), `scale_dial_fact` exceeds all 10 matched random `scale_dial_fact` sets. The cross-prefix bridge against the two natural-decode donor arms (`|delta_g_ali(nat_trace_fact) - delta_g_ali(scale_dial_fact)| <= 0.50` AND `|delta_g_ali(nat_trace_you) - delta_g_ali(dial_you_ref)| <= 0.50`, with `dial_you_ref` as `scale_dial_you` when `|hat_alpha_gen(you) - hat_alpha_gen(fact)| >= 0.05` else `scale_dial_fact`) is a DESCRIPTIVE contrast at that stated tolerance — the two sides differ in opening, so it is not a matched intervention effect and it does NOT enter the pass condition. Donor fit belongs to 214.4, where every arm starts from the same prompt tokens. Fails if the monotone-curve or random-set clause misses. | medium | Gemini 4 Neon |
| 214.4 | Stage 2 H4 primary test (two-phase schedule vs. donor resampling; `broken_rate <= 0.15` on all four arms, where the NUMERIC ceiling is authoritative and reads as an integer rule `broken <= 28 of 192`, and the unchanged-arm sanity ceilings read as `clean_google(unchanged) <= 11 of 192` (`0.06`) and `broken(unchanged) <= 9 of 192` (`0.05`), all computed only from the 192 main-arm rows and never from pilot rows): (a) `scale_dial_fact` matches `trace_resample_fact` within `±0.05` (`|rate(scale_dial_fact) - rate(trace_resample_fact)| <= 0.05`) and exceeds `unchanged` by `>= +0.04`; AND (b) with `|hat_alpha_gen(you) - hat_alpha_gen(fact)| >= 0.05` and denominator guard `rate(trace_resample_fact) - rate(trace_resample_you) >= 0.03`, the 1-D dial difference `rate(scale_dial_fact) - rate(scale_dial_you)` captures at least `60%` of `rate(trace_resample_fact) - rate(trace_resample_you)`, `|rate(scale_dial_you) - rate(trace_resample_you)| <= 0.05`, and worst-case broken recode holds (`(clean_google + broken)(scale_dial_you) < clean_google(scale_dial_fact)`). AND (c) **two-sided gap error** (added at Alden's hold): `|dial_gap - donor_gap| <= 0.05`, where `dial_gap = rate(scale_dial_fact) - rate(scale_dial_you)` and `donor_gap = rate(trace_resample_fact) - rate(trace_resample_you)`. Clause (c) is what licenses the word *accounts*: (a) plus (b) can pass while the dial reproduces four times the observed gap. If the Stage 0 decode dial collapsed (`< 0.05`), clause (b) is `not discriminated` and the prediction cannot pass on it. Fails if any clause misses, resolved in the single decision order above: `not read` / `not discriminated` on availability, `not discriminated from damage` when the numbers pass but validity or the worst-case recode fails, and a magnitude miss only on eligible, valid, recode-clean answers. | medium | Gemini 4 Neon |
| 214.5 | H1 vs. H2 phase-decomposition test (Stage 2, `broken_rate <= 0.15` on `prompt_fact_gen_you`, denominator guard `rate(trace_resample_fact) - rate(trace_resample_you) >= 0.03`): swapping ONLY the decode phase from `D_fact` to `D_you` (`prompt_fact_gen_you`) accounts for strictly more than half of the total `fact -> you` drop (`rate(trace_resample_fact) - rate(prompt_fact_gen_you) > 0.50 * (rate(trace_resample_fact) - rate(trace_resample_you))`), lowers clean Google below `trace_resample_fact` by at least `0.03`, AND survives worst-case broken recode (`(clean_google + broken)(prompt_fact_gen_you) < clean_google(trace_resample_fact)`). Fails if any clause misses (`not read` if denominator `< 0.03`). | medium | Gemini 4 Neon |
| 214.6 | H3 early-token spike vs. flat-mean test on `"you"` (Stage 2, `broken_rate <= 0.15`): pooling `D_you` into flat `mean_resample_you` does NOT reproduce a large spike-loss jump: `rate(mean_resample_you) - rate(trace_resample_you) <= +0.05` AND `rate(zero) - rate(mean_resample_you) >= 0.04`. (A large upward jump `> +0.05` is recorded as an OBSERVED mean-minus-trace difference. It is not evidence that spikes
are the mechanism: averaging changes prefill values, decode timing, donor identity and covariance at once.) Fails if either clause misses. | low | Gemini 4 Neon |
| 214.7 | Polite-request register without `"you"` (`D_imp`, Stage 1 natural-gen opening; every `nat_*` term here is keyed to its own recipient `"I was developed by **"` and to the `nat_trace_fact`/`nat_trace_you` baselines, and because those arms force different donor openings each delta is a CONTRAST ACROSS OPENINGS, not a matched intervention effect): `nat_trace_imp` shifts away from `nat_trace_fact` toward `nat_trace_you`: `delta_g_ali(nat_trace_fact) - delta_g_ali(nat_trace_imp) >= +0.20` logits AND `|delta_g_ali(nat_trace_imp) - delta_g_ali(nat_trace_you)| <= 0.30` logits. (Failure with `nat_trace_imp` near `nat_trace_fact` is recorded as a register observation. `D_imp` changes register,
words and modifiers at matched length, so no clause here identifies the pronoun.) Fails if either clause misses. | medium | Gemini 4 Neon |
| 214.8 | 3-unit early engine & downstream `L27 N3325` tracking (Stage 1; the `nat_*` terms are keyed to their own recipient and baseline, and clause (a) is a MATCHED comparison inside the natural-decode group — both sides force the same donor opening — so it is not a cross-opening contrast): (a) omitting `L29 N776` changes neither natural-gen donor arm by more than `0.25` logits (`|nat_fact_no_L29 - nat_trace_fact| <= 0.25` AND `|nat_you_no_L29 - nat_trace_you| <= 0.25`); AND (b) across the 10-point dial curve plus the 6 natural-decode arms (`16` arms total), mean last-position shift in `L27 N3325` correlates negatively with mean $\Delta g_{\text{ali}}$ (Spearman $\rho \le -0.80$). Fails if either clause misses. | high | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 214 (registered ...)" at registration time.

## Reading rules

- **Scorer (`part214/score214.py`, to be built and fixture-tested before registration and before any Stage 1/Stage 2 run):**
  computes every gate, Stage 0 dial projections and orthogonal residuals, Stage 1 10-point H4 dial curve, 4-D per-unit
  and frozen-vs-online diagnostics, `g_seq` agreement check, Stage 2 5-state counts, `"I am (**)?Qwen"` opening vs.
  conditional `Google | non-Qwen` breakdown, worst-case broken recodes, 10,000-draw question-clustered bootstrap 95%
  CIs (seed 214), and final verdicts directly from saved JSON/JSONL files.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (`214.3`, `214.7`, `214.8`) are marked
  `untestable (gate failed)`. If Gate 3 (including Gate 3b) fails, Stage 2 predictions (`214.1`, `214.4`, `214.5`,
  `214.6`) are marked `untestable (gate failed)`. Stage 0 prediction `214.2` requires only Gate 0 and Gate 3b.
- **Operational headline reading rules (H4-first, mutually exclusive, worst-case-recode guarded):**
  - **"A shared two-phase partial-zero schedule (H4) accounts for the `fact` vs. `"you"` ladder"** ONLY if Gates 1–3
    pass and `214.1`, `214.2`, `214.3`, AND all of `214.4(a)`–`(c)` hold (with `broken_rate <= 0.15` on every headline
    arm and the worst-case recode passing). If (a) and (b) hold but (c) misses, the honest label is a
    **point-tolerance/directional screen**: the dial reproduces the donor ORDER and the endpoint counts, and its
    magnitude is reported with the miss. Never write "accounts for" without (c).
    **Precedence, fixed (Alden's hybrid example).** When (a) and (b) pass and (c) ALONE misses, the point-tolerance
    screen is the verdict and the "missed its magnitude bar" headline below CANNOT also fire: a valid measured magnitude
    miss is not a mechanism win. Worked example, a required fixture: `fact = 19/192`, `you = 13/192`,
    `scale_dial_fact = 28/192`, `scale_dial_you = 4/192`, no broken rows, non-collapsed fits → `donor_gap = 6/192`,
    `dial_gap = 24/192`, `|dial_gap - donor_gap| = 18/192 > 0.05`, so (c) misses while `214.5` passes; the verdict is the
    screen, with the magnitude miss and the phase observation reported beside it.
    Sub-label the phase/spike/register structure from `214.5`, `214.6` and `214.7` — as observations, not as identified
    mechanisms.
  - **"The tested schedule missed its magnitude bar"** — NOT "H4 beaten by phase, spike, or directional content", which
    the controls cannot identify — ONLY if Gates 1–3 pass,
    `214.1` holds, `214.4` fails on **(a) or (b)** for a DISCRIMINATED reason (not `not read`, not `not discriminated`,
    not `not discriminated from damage`, and not a lone (c) magnitude miss, which is the screen above), AND at least one
    explicit rival clears its positive bar (`broken_rate <= 0.15` + worst-case recode):
    - **H3 (early-token temporal spikes):** `rate(mean_resample_you) - rate(trace_resample_you) > +0.05` with
      question-clustered 95% CI lower bound `> 0`. ⚠ Trace-to-bank-mean averaging changes prefill values, decode timing,
      donor identity and covariance at once (Alden), so 214.6 and this bar read as "mean pooling does not change the
      outcome much/muchly", not as "spikes are the mechanism".
    - **H1-directional (decode-phase non-scalar content):** `214.5` holds (`prompt_fact_gen_you` captures `> 50%` of
      the drop) while `214.4` fails **for a discriminated reason** (not `not read`, not `not discriminated`), and Stage
      1's 4-D per-unit dial vs. the shared dial is reported beside it. ⚠ The phase triangle is an ORDER-DEPENDENT
      DECOMPOSITION along FF→FY→YY (Alden): it splits the gap into a decode-leg share and a prompt-leg share and does
      not by itself prove decode-phase non-scalar content over KV mediation, since a two-phase scalar response can
      produce the same triangle.
    - **H2 (prompt-phase KV carry-over):** `rate(prompt_fact_gen_you) - rate(trace_resample_you) > 0.50 * (rate(trace_resample_fact) - rate(trace_resample_you))`
      with `(clean_google + broken)(trace_resample_you) < clean_google(prompt_fact_gen_you)`.
  - **Otherwise ("mixed / inconclusive"):** report the full Stage 0 activation table, Stage 1 dial curve and
    diagnostics, Stage 2 5-state and non-Qwen conditional tables, worst-case recodes, and bootstrap CIs without forcing
    a single-cause headline. If `214.1` fails, report that the batch-1 `fact` vs. `you` replication failed and withdraw
    the mechanism headline.

## Execution and timing (added at Alden's hold, 2026-11-19)

- **Measured timing before the main run, and where it is measured.** Stage 0 is DESCRIPTIVE-ONLY and produces no
  output-producing patched call, so the timings cannot live there (Alden). Timing is taken in the registered isolated
  pilot, as **timing-only calls outside the `64`-row pilot manifest**: one representative long call on each hook path
  Stage 2 actually uses — `trace_*` (cycled donor), `mean_resample_you`, `prompt_fact_gen_you` and `scale_dial_*` —
  including model load, host/device sync and transfers, JSON logging and one retry, at batch 1. That is **4 extra calls,
  enumerated in `run.json` as `timing_only_calls: 4` and never counted as evaluation rows**, so the pilot manifest keeps
  its frozen `64` and Gate 3's `{"1": 64}` still describes exactly the scored rows. The earlier 4.80 s/call figure came
  from Part 212's arms and does not cover the mixed-phase hooks.
- **Arithmetic of the budget, stated honestly.** 1,584 × 4.80 s = 2.11 h for Stage 2; adding 100 donor and 64 pilot
  generations at the same rate and Stage 1 once gives about 2.48 h before load, I/O and retries; at 7.5 s per call the
  run is about 3.8 h. Six hours is therefore plausible and not measured. The run consumes that ceiling only at about
  12.8 s per Stage 2 call.
- **Hard stop.** The runner takes a wall-clock deadline (`--deadline-min`, default 330). At the deadline it stops
  launching calls, writes what it has, and labels every incomplete arm `incomplete (deadline)` in the JSON and the
  README. No arm is silently dropped and no partial arm is scored as if it were full.
- **Counts are reported the way they are computed.** Forwards (model calls) and generation steps (decode tokens) are
  reported separately from held-out rows. Stage 1's `1,624` counts its candidate-sequence work inside the per-arm
  forwards, not as additional arms.

## Code acceptance (the review brief's contract, added at Alden's second hold)

Registration, code acceptance and host launch are three separate steps; this section fixes what the code review receives
so the second can be answered without re-litigating the first.

- **Exact pins.** Code head and base SHAs, the full `git diff --stat` and file list, SHA-256 of this design, of
  `part214/prompts214.json` and its builder, and of the Stage 0 contract file; the model revision and the kernel-check
  JSON the run asserts against.
- **Actual commands and outputs.** Every fixture and sabotage is reported as the command run, its exit code and its
  output — not a summary count. Each sabotage names the function it guards and the failure reason expected.
- **Required red/green cases (Alden's four families).**
  1. **Boundaries and denominators:** `9` versus `10` broken on `unchanged`, `11` versus `12` clean Google on `unchanged`,
     `28` versus `29` broken on a main arm, the fourfold-gap example, and the hybrid `fact 19 / you 13 / dial 28,4` case
     that must classify as the point-tolerance screen.
  2. **Unavailable versus mismatched:** collapsed fit, undefined `fact` projection, undefined `you` projection, donor gap
     below `0.03`, and a genuine eligible mismatch — with no mechanism win readable from any unavailable input.
  3. **Binding and indexing — fake hook plus fake model is SUFFICIENT (Alden, 11-19 22:52).** The test must drive the
     PRODUCTION runner and hook path (not a table of expected values), and assert: exact recipient and prefix token IDs;
     exact score positions; the actual tensor slots changed and the ones left untouched; prefill, first cached step and
     capped-donor (`L_gen`) indexing; and masked continuation positions. A wrong-prefix sabotage and an off-by-one
     sabotage must each go RED for the intended reason, with an intact positive path beside them. Table-only doubles that
     never execute the hook do not satisfy this case. **No real-model call is added to the pre-acceptance hand-off**; after
     code acceptance and registration, a separately approved pilot confirms the assumptions on the real model and kernel,
     and that is host evidence, not part of the code gate.
  4. **Runner behaviour:** patched generation refused before registration, pilot and full manifests kept separate
     (`timing_only_calls: 4` outside the `64`), deadline/partial/resume refusal, and the positive complete path.
- **What is fixture-only.** Anything exercised against a fake hook or model is labelled as such in the report; fixture
  evidence is never read as host evidence, and the host numbers come only from the pilot and the full run.

## Limits

- One model (`Qwen/Qwen3.5-4B` at one snapshot), one quantisation (bf16), one four-unit set (`L6 N753`, `L7 N1169`,
  `L11 N8868`, `L29 N776`), and Part 66's 24 English held-out phrasings.
- `scale_dial_fact` and `scale_dial_you` in Stage 2 use **online** scalar multiplication ($\hat{\alpha}\cdot x_{\text{live}}$),
  so scaling layer 6 feeds downstream drift into layers 7, 11, and 29; Stage 1's frozen-vs-online diagnostic measures
  this drift directly at the forced opening, not at later free-generation tokens.
- The two-phase schedule's projections $\hat{\alpha}_{\text{prompt}}$, $\hat{\alpha}_{\text{gen}}$ are defined over Stage 0's deterministic **greedy** trajectories of `EN_MAKER`
  and donors, whereas Stage 2 samples at `temperature=0.7, top_p=0.95`.
- `D_imp` is tested directly in Stage 0 (activations and donor greedy register) and Stage 1 (exact logits), and is
  NOT a ninth Stage 2 arm. The earlier draft let Stage 0 decide whether to add one; that is removed (Alden, item 3):
  an arm chosen after seeing its own calibration is outcome-dependent, and the design freezes at eight arms. `D_imp`
  also changes register, words and request modifiers at matched length, so 214.7 failing near `fact` cannot identify
  the pronoun itself — it is a register observation.
- Stage 2 omits random-unit sets to save ~2 hours of batch-1 compute, relying on Part 212's 20 random sets (0–2/24
  Google) plus Part 214 Stage 1's 10 matched random sets (`zero`, `scale_dial_fact`, `nat_trace_fact`) for random-unit
  specificity.
- Stage 2's phase triangle (`trace_resample_fact`, `prompt_fact_gen_you`, `trace_resample_you`) omits the fourth corner
  `prompt_you_gen_fact` in Stage 2 (tested in Stage 1 only), so a non-additive `prompt × gen` interaction in Stage 2
  would appear only as a split between the two triangle differences rather than a 4-corner Stage 2 interaction term.
