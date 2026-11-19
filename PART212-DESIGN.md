# Part 212: do Qwen3.5-4B's four maker neurons carry maker-specific evidence, or a factual-baseline scaffold whose removal exposes the Google fallback?

Procedure: v1
REGISTERED 2026-11-19 01:24 CT as v3.1, before any data (PREDICTIONS.md, Part 212). Designed by Gemini 4 Neon (proposer
and designer). v2 (2026-11-18 23:07 CT, Neon) folded all four
correction groups of Alden's 2026-11-18 cold design review `part212-design-review-2026-11-18.md` on v1 `4c5b4027`.
v3 folds Alden's narrow delta hold on v2 (council `part212-design-2026-11-18` #5); the fold was written by Journal
Claude on Elise's go (2026-11-19 00:17 CT), with no change to the arms, the units or any threshold Neon set. See
"Changes in v3".
Proposed by Gemini 4 Neon and selected by Elise ("Option B", 2026-11-18 22:09 CT). This draft will be registered on
public main through `tools/register_public.py` only after review clearance and before any model run of Part 212.

## Question

Zeroing four MLP neurons (`L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776`) in Qwen3.5-4B moves held-out English maker
answers to Google in 43 of 192 answers (22.4%) on eager attention, against 2 of 192 unchanged and at most 1 of 24 for
50 same-layer random four-unit sets (Part 183, replicating Parts 25 and 66). Yet unselected flat random directions at
dose 0.225 also lead to Google in 30 of 40 directions (Part 176), three of the four units are off (< 1 SD from rest) at
a maker-ban veto step (Part 159), and Yavuz (2026) showed that neurons selected by zero ablation can transfer to held-out
questions while carrying only a constant background bias that disappears under resample ablation from unrelated prompts.

Do the four maker neurons behave like question-specific maker evidence, or does replacing them with off-topic factual
activations preserve Alibaba while subtracting the off-topic factual profile alone reproduces much of zeroing's Google
fallback? Separately, do the off-topic factual and assistant-directed `"you"` donor banks differ (especially via
`L6 N753`), and does patching Chinese maker-prompt activations into English prompts suppress Google below the English
baseline?

## Prior art and existing data

- Closest published work:
  - **Yavuz (2026), "Are Causal Neuron Circuits Unique and Transferable?"** (`github.com/omtekyav/causal-neuron-circuits`,
    `PRIOR-ART.md` 2026-11-09): in Qwen3-0.6B (capitals), 12 neurons selected by zero ablation lowered the answer margin
    by 27% (random 0%) and transferred to held-out questions, yet replacing them with their activations from other
    questions (resample ablation) had no effect, showing they carried a near-constant signal rather than
    question-specific content.
  - **Tuomi (2026)** (replicated statically in our Part 206, `results/part206/README.md`): the constant component of the
    residual stream sets each MLP gate's resting reference ($\rho = 0.949$ to $0.974$ on Qwen3.5-4B chat against max
    random $0.130$). Zeroing a unit with a non-zero resting activation injects $-\bar{a}\cdot w_{\text{down}}$ into the
    residual stream.
  - **Malla, Choi and Choi (2026, arXiv:2609.06951)**: size-matched random perturbations relax instruction-tuned models
    toward their default behaviors.
  - What this part adds: an explicit 4-cell intervention factorial (`0`, `r`, `a_clean - r`, `a_clean`, with factorial
    interaction $I = F(a_{\text{clean}}) - F(r) - F(a_{\text{clean}} - r) + F(0)$ reported directly) on the four
    published maker neurons, pairing zero ablation against exact-token-length-matched off-topic factual resampling,
    assistant-directed `"you"` bank resampling, factual-profile subtraction, maker-activation-write-norm-matched random
    controls, opposite-direction scaling ($+2\times$, $-1\times$), prompt-vs-generation phase splits, and a cross-lingual
    Chinese-donor swap on both forced-opening logits and held-out free generations.
- Our own data that already bears on it:
  - **Parts 25, 66 and 183 (`results/part183/README.md`)**: on eager attention (`results/part183/kernel-check.json`),
    zeroing `L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776` gives Google in 43/192 (22.4%) held-out English answers
    against 2/192 unchanged and max 1/24 (4.2%) across 50 same-layer random sets (matched only by layer, not by
    activation magnitude or write norm).
  - **Parts 70–77 (`results/part72/part72.json`, `results/part74/part74.json`)**: on `'I was developed by **'`
    (8-maker probability mass 0.982), zeroing `L24 N3949` shifts Google-minus-Alibaba by -2.234 logits, 64% mediated
    through `L27 N3325` (whose activation is negative to write Google: -4.463 with makers zeroed vs -2.049 with L24
    also zeroed). Part 75 showed flat prompt-mean clamping of `L27 N3325` is not neutral because natural activations vary
    across positions; this design therefore tests both single-donor natural-variance traces and bank-mean profiles and
    includes `mean_resample_fact` directly in the headline reading rules.
  - **Part 32 (`part32.py`, `results/part32/score.txt`)**: zeroing the four gave Google in 16/24 English, 17/24 Spanish,
    and 0/24 Chinese answers.
  - **Part 39 (`PART39-DESIGN.md`) & Part 111 (`results/part111/README.md`)**: `L6 N753` tracks "talking to the AI";
    `L7 N1169` and `L11 N8868` write China/Chinese tokens through the Jacobian lens (8 and 7 of top 20, null <= 1.0%);
    `L29 N776` has a small negative mean (-0.047) at the opening.
  - **Part 159 (`results/part159/README.md`)**: at the maker-ban veto step, only `L7 N1169` was on ($\ge 1$ SD from
    rest, 14/24 rows); `L6 N753` (0/24), `L11 N8868` (2/24) and `L29 N776` (2/24) were off.
  - **Part 176 (`results/part176/README.md`)**: 30 of 40 flat random directions at dose 0.225 were supported, all
    Google-led, with median 1 of 44 broken.
  - **Seen before writing:** tokenizer-only verification of `part212/prompts212.json` via `part212/build_prompts212.py`
    (no model weights loaded, no activations or generations computed).

## Method

- **Model and rig:** `Qwen/Qwen3.5-4B`, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `torch.bfloat16`,
  Intel Arc A770 (`xpu`), `attn_implementation="eager"` via `neuron_report.load_model` (recorded in `run.json`),
  thinking off (`nr.build_prompt(tok, q, False)`), no system prompt.
- **Target units and hook site:** `TARGET = [(6, 753), (7, 1169), (11, 8868), (29, 776)]`, intervened via forward
  pre-hooks on `layers[L].mlp.down_proj` (the SwiGLU hidden state `act(gate) * up`, matching Parts 25, 66, 74 and 183).
  Downstream monitor units recorded without intervention: `L24 N3949` and `L27 N3325`.
- **Prompt and donor banks (`part212/prompts212.json`, SHA-256 `0f10c337ae64765b9f1cc32a85241214944d1c2b55be15d8398a085042793227`):**
  - `EN_MAKER`: Part 66's exact 24 held-out English maker phrasings (16 direct + 8 indirect, indices 0–23), positive
    control `"Who created you?"` (index 24), and Part 66's 6 fluency guards (`guards`, indices 25–30).
  - Four 25-prompt donor banks, each **exact 1-to-1 token-length matched** to `EN_MAKER[0..24]` under the pinned
    Qwen3.5-4B tokenizer (Gate 0, verified before registration: 3 identical prefix tokens, identical user-question token
    count, 9 identical suffix tokens, zero boundary merges on all 100 donor prompts):
    - `D_fact`: 25 off-topic factual English questions with no second-person pronouns (`you`/`your`) and no AI, company,
      China or Google terms.
    - `D_you`: 25 off-topic assistant-directed English questions containing `you`/`your`, with no maker, creator,
      company, identity, China or Google terms. Note: `D_fact` and `D_you` match token length and share the same neutral
      opening (`'The main answer is **'`), but differ in request shape and content (`"What causes thunder?"` vs `"Could
      you help?"`), so their contrast tests a **bank difference**, not an isolated causal effect of the pronoun `"you"`.
    - `D_zh_maker`: 25 Chinese maker phrasings.
    - `D_zh_fact`: 25 Chinese off-topic factual questions (used for the within-Chinese contrast `trace_resample_zh_maker
      - trace_resample_zh_fact` separating Chinese maker framing from Chinese language).
  - Stage 1 forced openings (all **exact 5 tokens**, verified by tokenizer):
    - `EN_MAKER`: `'I was developed by **'` (Part 72/74's verified opening)
    - `D_fact`, `D_you`: `'The main answer is **'`
    - `D_zh_maker`: `'我的开发机构是**'`
    - `D_zh_fact`: `'这道题的答案是**'`
- **Maker-activation-write-norm-matched random controls (`20 sets`):** On the unchanged Stage 1 recording pass over the
  24 held-out `EN_MAKER` prompts (prompt + forced opening), compute every unit's mean maker-context delivered write norm
  $m_{L,u} = \text{mean}_{q,p}(|a_{L,u}(q,p)|)\cdot\|W_{\text{down}}^{(L)}[:, u]\|_2$ in layers $L \in \{6, 7, 11, 29\}$
  (summed in float64 on CPU). In each layer, select the 50 units closest to the target unit in $m_{L,u}$ (excluding the
  target, ties broken by unit index). Draw 20 four-unit control sets (`random_0`..`random_19`) with
  `np.random.default_rng(21200 + k)` (one unit per layer from its 50-unit pool). This matches the expected **zeroing**
  write norm; it does NOT match realized perturbation norms under resampling ($a_{\text{donor}} - a_{\text{maker}}$) or
  subtraction ($-\bar{a}_{\text{fact}}$). For every arm and control, record the mean realized L2 residual-stream
  perturbation $\|\sum_u \Delta a_u \cdot w_{\text{down}, u}\|_2$ alongside the logit and behavioral readouts and treat
  unequal per-arm perturbation norms as an explicit limit on semantic attribution.

### Stage 1: Exact 4-cell single-forward factorial & mechanism audit (24 held-out prompts + `'I was developed by **'`)

1. **Unchanged recording pass:** Record `TARGET`, all 4×50 matched-pool units, `L24 N3949` and `L27 N3325` at every
   position on `EN_MAKER`, `D_fact`, `D_you`, `D_zh_maker` and `D_zh_fact` (each with its 5-token opening; 125 passes).
   Denote the clean recorded activation of unit $u$ on `EN_MAKER[q]` at position $p$ by the frozen tensor
   $a_{\text{clean}}(q, p)$. Construct the bank-mean factual profile $r(p) = \bar{a}_{\text{fact}}(p)$ (and similarly
   $\bar{a}_{\text{you}}(p)$): exact per-position mean across the 24 held-out donors on the 12 shared template positions
   and 5 forced-opening positions, and pooled mean across all 24 donors' user-body tokens on user-body positions.
   **Mapping onto a recipient (v3):** the profile is indexed relative to each recipient's own prompt length $T_q$, never
   by absolute position: header positions $0 \dots 2$ take the header profile, user-body positions $3 \dots T_q - 10$
   take the pooled body mean, the 9 suffix positions $T_q - 9 \dots T_q - 1$ take the suffix profile in suffix order,
   and opening positions $T_q \dots T_q + 4$ take the opening profile. The same mapping applies to every
   `mean_resample_fact`, `sub_factual`, `sub_factual_online` and `sub_you` hook in Stage 1 and to call 0 in Stage 2.
2. **Exact frozen vs. online intervention contract on `EN_MAKER[0..23]` (+ `'I was developed by **'`, same prompt per
   batch of 32 → zero padding):**
   - To make the 4-cell factorial (`0`, `r`, `a_clean - r`, `a_clean`) mathematically exact across all four layers
     (avoiding downstream layer drift between layers 6, 7, 11 and 29), Stage 1's primary subtraction and scaling arms
     clamp each target unit to **frozen functions of $a_{\text{clean}}(q, p)$**; one diagnostic arm (`sub_factual_online`)
     tests online subtraction $x_{\text{live}}(p) - r(p)$ beside it so any frozen-vs-online difference is measured.
   - **15 main arms** on `TARGET` (all positions $0 \dots T+4$ unless noted):
     - `unchanged`: no hook
     - `self_patch` (Gate 1): clamp to frozen $a_{\text{clean}}(p)$
     - `zero`: clamp to $0$
     - `trace_resample_fact`: clamp to frozen paired donor trace $a_{\text{fact}, q}(p)$
     - `trace_resample_you`: clamp to frozen paired donor trace $a_{\text{you}, q}(p)$
     - `mean_resample_fact`: clamp to frozen factual profile $r(p) = \bar{a}_{\text{fact}}(p)$
     - `sub_factual`: clamp to frozen $a_{\text{clean}}(p) - r(p)$
     - `sub_factual_online` (diagnostic): online $x_{\text{live}}(p) - r(p)$
     - `sub_you`: clamp to frozen $a_{\text{clean}}(p) - \bar{a}_{\text{you}}(p)$
     - `scale_pos2`: clamp to frozen $2\cdot a_{\text{clean}}(p)$
     - `scale_neg1`: clamp to frozen $-1\cdot a_{\text{clean}}(p)$
     - `zero_prompt_only`: clamp to $0$ on prompt positions $0 \dots T-1$ only
     - `zero_opening_only`: clamp to $0$ on the 5 forced-opening positions $T \dots T+4$ only
     - `trace_resample_zh_maker`: clamp to frozen $a_{\text{zh\_maker}, q}(p)$
     - `trace_resample_zh_fact`: clamp to frozen $a_{\text{zh\_fact}, q}(p)$
   - **16 per-unit arms:** each of the 4 target units individually under `zero`, `trace_resample_fact`,
     `trace_resample_you` and `sub_factual` (frozen).
   - **100 matched random control arms:** all 20 maker-activation-write-norm-matched random 4-unit sets under `zero`,
     `trace_resample_fact`, `trace_resample_you`, `sub_factual` (frozen) and `scale_neg1` (frozen).
3. **Stage 1 readouts & continuation hook contract (all computed in float32 from logits):**
   - Primary logit contrast: $g_{\text{ali}} = \text{logit}(\text{Google}) - \text{logsumexp}(\text{logit}(\text{Al}), \text{logit}(\text{T}))$,
     where `Al` (2033) and `T` (51) are the first tokens of `Alibaba` and `Tongyi`, and `Google` (14030) is single-token.
     For every arm, $\Delta g_{\text{ali}}$ is its paired per-prompt mean difference from `unchanged`.
   - **Exact factorial interaction:** compute and report $I_{g} = F(a_{\text{clean}}) - F(r) - F(a_{\text{clean}} - r) + F(0)$
     from `self_patch`, `mean_resample_fact`, `sub_factual` and `zero`. Never interpret nonlinear logit ratios as
     literal fractions of stored content.
   - **Full multi-token sequence log-probability contrast (`g_seq`) & continuation hook contract:** on all 15 main arms
     × 24 prompts, score `Google` (`[14030]`), `Alibaba` (`[2033, 27343]`) and `Tongyi` (`[51, 626, 36814]`) to compute
     $g_{\text{seq}} = \log P(\text{"Google"}) - \max(\log P(\text{"Alibaba"}), \log P(\text{"Tongyi"}))$. **Frozen
     continuation hook rule:** hooks apply identically on positions `0 .. T+4` (prompt + 5-token forced opening, which
     already predicts the first maker token at position `T+4`) and are **inactive (untouched)** on the 1–2 appended
     continuation positions (`'Al'`, `'T'`, `'ong'`), avoiding unrecorded continuation-token donor profiles.
   - Secondary readouts reported alongside: Part 74's exact $g_{\text{al}} = \text{logit}(\text{Google}) - \text{logit}(\text{Al})$,
     8-maker first-token probability mass `mass`, and last-position activations of `L24 N3949` and `L27 N3325`.

### Stage 2: Held-out free generation (exact Part 66 / Part 183 rig, eager attention)

1. **Donor generation recording & prefill/decode indexing contract:** Run greedy generation (`max_new_tokens=48`, eager,
   bf16) once per donor prompt in `D_fact`, `D_you` and `D_zh_maker` (75 runs), recording `TARGET` and all 20
   matched-control units at every prefill position `0 .. T-1` and every **non-EOS** generated position `0 .. L_gen - 1`
   (stopping strictly before EOS; EOS token activations are never stored or injected). Compute the pooled non-EOS
   generated-token mean $\mu_{\text{fact, gen}}$ across all 24 held-out `D_fact` donors.
   **Donor indices are bound to captured forwards (v3):** a generated token's activation exists only once a decode call
   processes it, so generated position $j$ is recorded by decode call $j + 1$. Under `max_new_tokens=48` a donor can
   emit 48 non-EOS tokens, but the 48th is never fed back and has no activation. `L_gen` is therefore the number of
   **processed** non-EOS generated positions, $L_{\text{gen}} = \min(n_{\text{non-EOS}}, 47)$; the uncaptured final
   token is excluded from $\mu_{\text{fact, gen}}$ and from `donor_gen`, and is not replayed. `run.json` logs both
   $n_{\text{non-EOS}}$ and `L_gen` per donor.
   - **Short/empty donor refusal (Gate 3b):** if any donor produces `L_gen < 4` non-EOS tokens, Stage 2 stops before
     generation and reports the short donor; `L_gen` for every donor is logged in `run.json`.
   - **Exact prefill/decode indexing during `model.generate(use_cache=True)`:**
     - Forward call 0 (prefill, `x.shape[1] == T`, predicts first answer token 0): `trace_resample_*` clamps positions
       `0 .. T-1` to `donor_prompt[0 .. T-1]`; `mean_resample_fact` clamps to $\bar{a}_{\text{fact}}[0 .. T-1]$;
       `sub_factual` applies online $x_{\text{live}}[0 .. T-1] - \bar{a}_{\text{fact}}[0 .. T-1]$; `scale_pos2` /
       `scale_neg1` apply online $2\cdot x_{\text{live}}$ / $-1\cdot x_{\text{live}}$.
     - Forward call $s \in \{1 \dots 47\}$ (decode step processing generated token $s - 1$ to predict token $s$,
       `x.shape[1] == 1`): `trace_resample_*` clamps to `donor_gen[(s - 1) % L_gen]`; `mean_resample_fact` clamps to
       $\mu_{\text{fact, gen}}$; `sub_factual` applies online $x_{\text{live}} - \mu_{\text{fact, gen}}$; `scale_pos2`
       / `scale_neg1` apply online $2\cdot x_{\text{live}}$ / $-1\cdot x_{\text{live}}$.
     - `zero_prompt_only` zeroes call 0 (prefill) only; `zero_gen_only` zeroes decode calls $s \ge 1$ only (note:
       answer token 0 is predicted by call 0, so `zero_gen_only` leaves answer token 0 unzeroed and acts from generated
       token 1 onward, matching Part 84's generated-tokens convention).
2. **Sampled generation (`temperature=0.7`, `top_p=0.95`, `max_new_tokens=48`, batch size 24, one prompt per batch →
   zero padding, SHA-256 per-row seeds matching Part 66's convention):**
   - **6 full arms (240 answers each = 1,440):** `unchanged`, `zero`, `mean_resample_fact`, `sub_factual`, `scale_pos2`,
     `scale_neg1` — 24 held-out prompts (8 samples = 192), `"Who created you?"` (24), and 6 fluency guards (4 each = 24).
   - **2 exact-length trace-resample arms (216 answers each = 432):** `trace_resample_fact`, `trace_resample_you` — 24
     held-out prompts (192) and `"Who created you?"` (24).
   - **3 diagnostic arms (192 held-out answers each = 576):** `trace_resample_zh_maker`, `zero_prompt_only`,
     `zero_gen_only`.
   - **20 maker-activation-write-norm-matched random control sets (1,440 answers):** each set on the 24 held-out prompts
     (1 sample each, sample index 0) under `zero`, `trace_resample_fact` and `sub_factual` ($20 \times 24 \times 3$).

Kernel check: results/part183/kernel-check.json

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** `scale_pos2` ($+2\times$) and `scale_neg1` ($-1\times$) beside
  `zero` ($0\times$) in both Stage 1 and Stage 2.
- **Poles (#2):** Not used as a separate pole-selection step, because this part intervenes on Part 25's four fixed units
  and tests sign inversion (`scale_neg1`) under #1.
- **Do-nothing hook (#3):** `self_patch` in Stage 1 (clamps each unit to its own frozen clean activation; must match
  `unchanged` within 0.05 logits) and clean hook-removal check before/after Stage 2.
- **Matched random controls (#4, Toolkit I):** 20 four-unit sets matched on maker-context delivered write norm
  $\text{mean}(|a_{\text{maker}}|)\cdot\|w_{\text{down}}\|_2$ per layer, evaluated under `zero`, `trace_resample_fact`
  and `sub_factual` (plus `trace_resample_you` and `scale_neg1` in Stage 1).
- **Answer-polarity split (#5):** Not used: the Stage 2 outcome is a six-family maker classification, not a binary
  yes/no label; Stage 1 prints both Google and Alibaba/Tongyi logits and total 8-maker mass (Toolkit J).
- **Broken answers & full state transitions counted (#6):** Every Stage 2 answer is screened mechanically (`broken` if
  empty, under 4 words, contains `\ufffd` or `!!!!`, or repeats any 3-word n-gram $\ge 4$ times). Every full held-out
  main arm uses the fixed denominator 192 (`clean_google_rate = count(clean_google) / 192`); 24-row sets use 24 (see
  scope below). Every arm reports the full
  5-category state distribution (`clean_google`, `clean_alibaba`, `clean_other_maker`, `clean_no_maker`, `broken`) and
  prompt-paired transition matrix against `zero` and `unchanged`.
  **Scope of the 192 denominator (v3):** 192 is the denominator for every full held-out main arm (8 samples × 24
  prompts). The 24-answer sets (each matched random set's sample-0 answers, the `"Who created you?"` rows, the fluency
  guards, and any sample-0 equal-budget table) use their own denominator of 24 and are never pooled with a 192-row
  arm. **Rank comparisons (v3.1, Alden #7):** the descriptive ranks in `212.1` and `212.5` compare the full arm's share
  (`count / 192`) with each matched random set's share (`count / 24`), as Neon's v2 rules specify. The equal-budget
  sample-0 rank (the main arm's sample-0 rows, `count / 24`, against each random set's `count / 24`) is a separate
  descriptive table and decides nothing. The two ranks can disagree (a main arm at 48/192 with 8/24 in sample 0 loses
  to a random set at 7/24 in the full-arm comparison and wins in the sample-0 one); the full-arm rank is the one the
  predictions read.
  **Which Google count the predictions read (v3.1, Alden #7):** `clean_google` is the exclusive five-state category
  (unbroken, and Google is the earliest-starting maker match). Every prediction, gate and headline rule, and both sides
  of every worst-case recode, use this one count. The legacy any-Google lexical hit (Part 66's `families()` contains
  Google, broken or not) is reported beside it, labelled `legacy_any_google`, for comparison with Parts 66 and 183, and
  decides nothing.
  **Transition join (v3):** transition matrices join rows by (question index, sample index). Part 66's row seed hashes
  the condition id with the question and sample (`row_seed(condition_id, question_index, sample)`), so a joined pair
  shares its prompt and sample index but not its random draw; the table is descriptive, not a matched-draw comparison.
  **Exclusive categories (v3):** `broken` takes precedence; an unbroken answer is then assigned to the maker whose
  Part 66 lexicon match starts earliest in the text (first-named maker), or `clean_no_maker` if none matches. The scorer
  reports, per arm, how many unbroken answers matched more than one maker family, so a first-named rule that decides
  many rows is visible.
  **Per-headline-arm validity bound:** every arm
  entering a mechanistic headline (`zero`, `trace_resample_fact`, `mean_resample_fact`, `sub_factual`) must have
  `broken_rate <= 0.15` (<= 28 of 192); a damaged arm stays descriptive and cannot support a scaffold or evidence claim.
- **Size first (#7):** Multiplied out in `Size` below with a pilot gate before full Stage 2.
- **Dose ladder (#8, Toolkit E):** Multiplicative axis `scale_pos2` ($+2$), `unchanged` ($+1$), `zero` ($0$), and
  `scale_neg1` ($-1$); Part 172 already ran fractional knockout rungs $d = 0.5, 0.75, 1.0$.
- **Prompt vs. generation phase (Toolkit F):** `zero_prompt_only` vs `zero_gen_only` in Stage 2 and `zero_prompt_only`
  vs `zero_opening_only` in Stage 1.
- **Hand-read check:** Up to 30 Google-classified held-out answers per non-baseline main arm are audited by hand (maker
  claim vs. mere mention/negation), matching `part66/score66.py`'s `hand-read.md`. Not used: an LLM blind reader,
  because maker classification uses Part 66's fixed lexical rule audited by hand.

Gates (a failed gate stops reading the affected stage and is reported publicly):

- **Gate 0 (tokenizer & prompt parity, passed pre-registration):** `part212/build_prompts212.py` verifies 25/25 exact
  token-length match across all four donor banks, exact 5-token openings, distinct 8-maker first tokens, and zero
  lexical-screen violations (`prompts212.json` SHA-256 pinned above).
- **Gate 1 (Stage 1 do-nothing hook & logit mass):** `|mean g_ali(self_patch) - mean g_ali(unchanged)| < 0.05` logits;
  zero non-finite logits or activations; mean 8-maker probability mass `>= 0.50` in both `unchanged` and `zero`.
- **Gate 2 (Stage 1 positive control):** `zero` raises mean `g_ali` over `unchanged` by at least `+1.0` logit and
  exceeds at least 18 of 20 matched random `zero` sets; failure stops Stage 1 mechanism readings (212.2–212.3,
  212.6–212.8, 212.10).
- **Gate 3 (Stage 2 kernel, completeness, donor-length & baseline fluency):** `run.json` records
  `attn_implementation == "eager"`, zero non-finite/empty rows, every planned key present once, every donor in
  `D_fact`/`D_you`/`D_zh_maker` produces `L_gen >= 4` non-EOS tokens (**Gate 3b**), and `unchanged` has guard accuracy
  `>= 0.90` with `unchanged` broken rate `<= 0.05`.

## Size

- Smallest effect worth knowing: a 50% reduction in `zero`'s held-out Google rate (~22% down to ~11%) or a +5 percentage
  point rise above `unchanged` (~1% up to ~6%) on Part 66/183's fixed budget of $n = 192$ held-out answers (24
  questions × 8 samples, evaluated with 10,000-draw question-clustered bootstrap 95% CIs; no iid McNemar power
  guarantee is asserted for the clustered design). In Stage 1, 24 held-out prompts resolve a paired logit shift of ~0.25
  logits (Part 74 SE across 24 prompts was ~0.08 logits).
- Total: 3,963 generations (Stage 2, plus 3,989 short forward passes in Stage 1), about 46 minutes on the A770.
- Pilot: Stage 1 (3,989 forward passes, ~2 min) + 75 greedy donor recordings (~1 min) + a Stage 2 mini-pilot written to
  `results/part212/pilot/` (4 held-out prompts indices 0, 5, 12, 16 × 4 samples + `"Who created you?"` × 24 samples
  across `unchanged`, `zero`, `trace_resample_fact`, `sub_factual` = 160 sampled generations, ~2 min; total ~5 min).
  Stopping rule: do not launch full Stage 2 if Gate 1, Gate 2, or Gate 3b fails, if any pilot row is empty or
  non-finite, if `run.json` does not record `eager`, or if `zero` produces fewer than 2 Google answers of 24 on
  `"Who created you?"` in the pilot. Full Stage 2 runs clean into `results/part212/stage2/` so pilot and full batch
  shapes never mix.

## Optimization

- Bound resource: GPU (Intel Arc A770, 16 GB VRAM; host Windows RAM ceiling ~7.5 GB during Qwen3.5-4B load; launched
  only when no other GPU or heavy WSL job is active).
- Measured: Part 183's eager run on this exact model, revision, card, dtype and 48-token cap measured **1.51 answers/s
  at batch 25** (1,880 answers in 1,241.3 s, peak GPU memory 9.09 GB, `results/part183/part66-eager/run.json`).
- Levers (from `PROCEDURE.md` section 4):
  - **Batch size:** taken — batch 32 for Stage 1 single-forward passes (identical prompt per batch, zero padding); batch
    8/24 for Stage 2, keeping one prompt per batch with zero padding and avoiding batch 1 (Part 178 lesson). Pilot times
    batch 8 vs 24 on real rows before full Stage 2.
  - **max_new_tokens:** taken — capped at 48 tokens (exact Part 66/183 budget).
  - **Teacher-forced scoring instead of generation:** taken — Stage 1 runs the full 131-arm matrix as single-forward
    passes in ~2 minutes before Stage 2 generation.
  - **Pool random controls:** taken — all 20 matched random sets share one Stage 1 recording pass and one donor greedy
    recording pass, evaluated at 1 sample per held-out prompt in Stage 2 (exact Part 66 budget).
  - **Vectorized hook / no per-token CPU sync:** taken — hooks assign tensor columns directly on-device with no
    `.item()` calls inside forward hooks; mean-norm accumulation moves to CPU float64 once per prompt.
  - **Vocabulary-size work on CPU:** taken — softmax/logsumexp computed in float32 only at the last position.
  - **Cache and resume:** taken — donor activations are saved once and reused; Stage 2 appends JSONL rows keyed by
    `(arm, question_index, sample)` and skips completed rows on resume; `torch.xpu.empty_cache()` between prompts.
  - **Early-exit generation loop:** declined — 48-token batches finish together in ~10–15 s and dynamic batching would
    alter batch shapes relative to Part 183.
  - **Direct load / copy engine off:** taken — launched via `tools/launch_arc.py` (copy engine off by default).
  - **Labelling API:** n/a — fixed lexical + mechanical broken-answer scorer, zero API cost.
  - **Device:** A770 local (cost $0; full run ~46 min).

## Predictions

All predictions authored by Gemini 4 Neon, registered before any Part 212 model run. Stage 2 rates on full held-out
main arms use the fixed denominator 192 (`clean_google_rate = count(clean_google) / 192`, `clean_google` the exclusive
category); a matched random set's share uses its own 24 rows.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 212.1 | Stage 2 replication against maker-activation-write-norm-matched nulls: `zero` has clean Google rate between 12% and 35% of 192 held-out answers (`broken_rate <= 0.15`), `unchanged` has at most 3 of 192 (<= 1.6%), and `zero`'s full-arm share (`count / 192`) strictly exceeds each of the 20 matched random `zero` sets' shares (`count / 24`) (descriptive top rank; the equal-budget sample-0 rank on 24 answers is reported beside it and decides nothing; no uniform-null p-value claimed). Fails if any clause misses. | high | Gemini 4 Neon |
| 212.2 | Stage 1 factual-resample test (exact logits): `zero` raises mean `g_ali` by at least +1.5 logits over `unchanged` and exceeds all 20 matched random `zero` sets; AND `trace_resample_fact` stays in the near-baseline preservation band `-0.50 <= delta_g_ali(trace_resample_fact) < 0.50 * delta_g_ali(zero)` (strictly less than 50% of `zero`'s shift without a large negative reversal). Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.3 | Stage 1 factual-profile subtraction: frozen `sub_factual` ($a_{\text{clean}} - \bar{a}_{\text{fact}}$) raises mean `g_ali` by at least +0.75 logits, retains at least 35% of `zero`'s mean $\Delta g_{\text{ali}}$ ($\Delta g_{\text{ali}}(\text{sub\_factual}) / \Delta g_{\text{ali}}(\text{zero}) \ge 0.35$), and exceeds at least 18 of 20 matched random `sub_factual` sets. Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.4 | Stage 2 factual-resample test (free generation, with fluency validity): on the 192 held-out answers, `trace_resample_fact` has `broken_rate <= 0.15`, its clean Google rate is strictly less than half of `zero`'s clean Google rate (`ratio < 0.50`), the loss survives a worst-case recode that counts every broken `trace_resample_fact` answer as Google (`count(clean_google) + count(broken) < 0.50 * count_zero(clean_google)`), AND the paired question-clustered 95% bootstrap CI of `rate(zero) - rate(trace_resample_fact)` lies strictly above 0. Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.5 | Stage 2 factual-profile subtraction: on the 192 held-out answers, `sub_factual` exceeds `unchanged` in clean Google rate by at least 5 percentage points (`rate(sub_factual) - rate(unchanged) >= 0.05`), its full-arm share (`count / 192`) exceeds the shares (`count / 24`) of at least 18 of 20 matched random `sub_factual` sets (descriptive rank; the sample-0 rank is reported beside it and decides nothing), and has `broken_rate <= 0.15`. Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.6 | Factual vs. `"you"` donor-bank difference (Stage 1): `trace_resample_fact` exceeds `trace_resample_you` in mean $\Delta g_{\text{ali}}$ by at least +0.25 logits, AND in the single-unit 4x4 decomposition `L6 N753`'s `trace_resample_fact - trace_resample_you` gap is **strictly positive (`>= +0.10` logits)** and strictly larger than each of the other three units' (`L7 N1169`, `L11 N8868`, `L29 N776`). Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.7 | Downstream Google-writer routing: across the 10 core English intervention arms in Stage 1 (`unchanged`, `zero`, `trace_resample_fact`, `trace_resample_you`, `mean_resample_fact`, `sub_factual`, `scale_pos2`, `scale_neg1`, `zero_prompt_only`, `zero_opening_only`), each arm's mean last-position shift in `L27 N3325` correlates negatively with its mean $\Delta g_{\text{ali}}$ (Spearman $\rho \le -0.70$). Fails if $\rho > -0.70$. | high | Gemini 4 Neon |
| 212.8 | Opposite-direction monotonicity (`STANDARD-CONTROLS.md` #1): in Stage 1, frozen `scale_pos2` ($2a_{\text{clean}}$) lowers mean `g_ali` relative to `unchanged` by at least 0.30 logits ($\Delta g_{\text{ali}} \le -0.30$), AND frozen `scale_neg1` ($-a_{\text{clean}}$) raises mean `g_ali` strictly more than `zero` ($\Delta g_{\text{ali}}(\text{scale\_neg1}) > \Delta g_{\text{ali}}(\text{zero})$). Fails if either clause misses. | medium | Gemini 4 Neon |
| 212.9 | Prompt vs. generation phase (`TOOLKIT.md` F): in Stage 2, `zero_gen_only`'s clean Google rate exceeds `zero_prompt_only`'s clean Google rate by at least 4 percentage points (`rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04`). Fails below 0.04. | low | Gemini 4 Neon |
| 212.10 | Absolute Chinese-donor swap (Stage 1): patching Chinese maker-prompt activations (`trace_resample_zh_maker`) into English maker prompts gives mean $\Delta g_{\text{ali}} > -0.25$ logits relative to English `unchanged` (no-large-suppression prediction; the within-Chinese contrast `trace_resample_zh_maker - trace_resample_zh_fact` is reported beside it to separate maker framing from Chinese language). Fails if $\Delta g_{\text{ali}} \le -0.25$. | low | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 212 (registered ...)" at registration time.

## Reading rules

- **Scorer:** `part212/score212.py` computes every gate, primary/secondary metric, 4-cell factorial interaction
  $I = F(a_{\text{clean}}) - F(r) - F(a_{\text{clean}} - r) + F(0)$, `sub_factual` vs `sub_factual_online` comparison,
  10,000-draw question-clustered bootstrap 95% CIs (seed 212), full 5-category state and transition tables, realized
  perturbation norms, and prediction verdicts directly from saved Stage 1 JSON and Stage 2 JSONL files.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (212.2, 212.3, 212.6, 212.7, 212.8, 212.10) are
  marked `untestable (gate failed)`. If Gate 3 (including Gate 3b) fails, Stage 2 predictions (212.1, 212.4, 212.5,
  212.9) are marked `untestable (gate failed)`.
- **Exact `g_seq` agreement rule:** `g_seq` confirms Stage 1's 212.2/212.3 pattern if and only if
  $\Delta g_{\text{seq}}(\text{zero}) > 0.10$ logits (ratio refused if $\le 0.10$),
  $-0.50 \le \Delta g_{\text{seq}}(\text{trace\_resample\_fact}) < 0.50 \cdot \Delta g_{\text{seq}}(\text{zero})$, AND
  $\Delta g_{\text{seq}}(\text{sub\_factual}) \ge 0.35 \cdot \Delta g_{\text{seq}}(\text{zero})$.
- **Hand-read audit:** Up to 30 Google-classified held-out answers per non-baseline main arm are audited by eye to
  separate genuine maker claims from negations/mentions; if > 20% of Google-matched answers in any headline arm (`zero`,
  `trace_resample_fact`, `mean_resample_fact`, `sub_factual`) are non-claims, the headline is qualified with the
  hand-audited claim rate.
- **Operational headline readings & withdrawal rule (explicit positive validity/effect predicates; never using a failed
  conjunction alone as positive proof):**
  - **"Factual resampling/subtraction pattern consistent with a scaffold"** ONLY if Gates 1–3 pass, 212.1–212.5 ALL
    hold, `g_seq` agreement passes, `mean_resample_fact` ALSO stays in the `< 50%` preservation band in Stage 1
    (`-0.50 <= delta_g_ali < 0.50 * delta_g_ali(zero)`) and Stage 2 (`clean_google_rate < 0.50 * rate(zero)` with
    `broken_rate <= 0.15` and the worst-case recode `count(clean_google) + count(broken) < 0.50 * count_zero(clean_google)`),
    and the factorial interaction $I$ is printed beside the headline. A loss that holds on clean answers but not under
    the worst-case recode is reported descriptively ("loss not separable from damage") and supports neither headline.
  - **"Pattern consistent with question-specific maker evidence"** ONLY if Gates 1–3 pass, 212.1 holds,
    `broken_rate <= 0.15` on `zero`, `trace_resample_fact`, `mean_resample_fact` and `sub_factual`, BOTH
    `trace_resample_fact` and `mean_resample_fact` retain `>= 50%` of `zero` in Stage 1 AND Stage 2 (with Stage 2
    clustered 95% CI of `rate(trace_resample_fact) - rate(unchanged)` strictly above 0), AND `sub_factual` shows no
    direct effect of either sign above the bounds (`|delta_g_ali| < 0.50` logits in Stage 1 AND
    `|clean_google_rate - rate(unchanged)| < 0.05` in Stage 2). A `sub_factual` shift of `delta_g_ali <= -0.50` or a
    Stage 2 drop of 0.05 or more is a **strong reversal**: it is reported as its own finding and supports neither
    headline.
  - **Otherwise ("mixed / split across arms, units, or stages"):** report the complete 4-cell factorial table,
    factorial interaction $I$, `mean` vs `trace` comparison, realized perturbation norms, 5-category transition tables,
    and 4×4 per-unit breakdown without forcing a single-cause headline. If 212.1 fails, withdraw the Finding-1
    specificity premise for this control family and report that finding first.

## Limits

- One model (`Qwen/Qwen3.5-4B` at one snapshot), one four-neuron set (`L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776`),
  and Part 66's 24 English held-out phrasings.
- `D_fact` and `D_you` match token length and neutral opening, not request shape/content, so 212.6 measures a bank
  difference rather than an isolated pronoun effect. `EN_MAKER` vs `D_fact` also change the 5-token forced opening
  (`'I was developed by **'` vs `'The main answer is **'`), so opening-semantics differences are not held constant.
- Random controls are matched on maker-context zeroing write norm (`mean(|a_maker|) * ||w_down||_2`), which does NOT
  guarantee equal realized perturbation norms under resampling or subtraction. Unequal per-arm perturbation norms are
  logged and reported as an explicit limitation on semantic attribution.
- The 4-cell factorial (`0`, `r`, `a_clean - r`, `a_clean`) defines an exact algebraic intervention split relative to
  the factual donor bank, not a unique physical resting component; task-level support or a different donor baseline
  remain alternatives.
- Stage 2's cross-lingual swap patches Chinese activations into English prompts, not the reverse on Chinese generations
  (Part 32 already observed 0/24 Google under Chinese zeroing); `212.10` tests no-large-suppression vs. English
  baseline, while the within-Chinese `zh_maker - zh_fact` contrast is reported descriptively beside it.

## Changes in v2 (folding Alden's 2026-11-18 cold design review)

1. **Fluency validity bounds on all headline arms:** added `broken_rate <= 0.15` to `212.1`, `212.4`, `mean_resample_fact`,
   and both positive headline branches, plus full 5-category state and transition tables (`clean_google`, `clean_alibaba`,
   `clean_other_maker`, `clean_no_maker`, `broken`) retaining fixed denominator 192.
2. **Frozen vs. online intervention, factorial interaction, and continuation/decode contracts:** froze Stage 1 subtraction
   and scaling to `a_clean` (exact 4-cell factorial, with `sub_factual_online` diagnostic beside it and factorial
   interaction $I$ reported), froze `g_seq` continuation hooks as active on `0..T+4` and inactive on appended maker
   continuations, froze Stage 2 prefill/decode step indexing (`call 0` vs `call s >= 1`) and added Gate 3b (`L_gen >= 4`
   non-EOS donor tokens).
3. **Scoped controls and headlines:** framed 212.6 as a factual-vs-`"you"` bank difference and required `L6 N753`'s
   single-unit gap to be strictly positive (`>= +0.10` logits); labeled 212.10 as an absolute Chinese-donor swap with
   within-Chinese `zh_maker - zh_fact` beside it; renamed controls to maker-activation-write-norm-matched and stated
   unequal per-arm perturbation norms as an explicit limit; added `mean_resample_fact` directly into both headline rules
   with operational wording.
4. **Statistical & verdict contracts:** removed the uncalibrated `p = 1/21` unequal-budget claim from `212.1` (adding
   equal-budget sample-0 rank descriptively); removed the unsupported McNemar `> 0.85` power claim; aligned `g_seq`
   thresholds (`< 0.50` / `>= 0.35`, denominator guard `> 0.10`); bounded `212.2` against large negative reversals
   (`>= -0.50`); replaced failed-conjunction alternate-headline logic with explicit positive effect and validity
   predicates.

## Changes in v3 (folding Alden's narrow delta hold on v2, council `part212-design-2026-11-18` #5)

Alden checked only the v2 delta and found the v1 points closed except two headline rules, plus implementation notes.
No arms, units, doses or thresholds changed.

1. **A loss made of broken text cannot support a headline.** Alden's paired example: `zero` 24/192 clean Google, each
   factual resample keeping 11 and turning 13 into broken answers, passes the 15% floor (6.77% broken), the half-rate
   ratio and the trace bootstrap. `212.4` and the scaffold headline's `mean_resample_fact` clause now also require the
   loss to survive a worst-case recode (`clean_google + broken < 0.50 × zero's clean_google`, counts on 192); a loss
   that fails it is reported as not separable from damage.
2. **"Small direct effect" is now a magnitude.** The alternate headline required `delta_g_ali < 0.50`, which also
   accepts −5 logits. It now requires `|delta_g_ali| < 0.50` and `|Δ clean Google rate| < 0.05`; a reversal at or
   beyond either bound is reported separately as a strong reversal.
3. **Implementation contracts:** the 192 denominator is scoped to full held-out main arms (24-row sets keep 24);
   transitions join on (question, sample) index; the five states are exclusive (`broken` first, then the first-named
   maker, with a per-arm multi-hit count); `L_gen` counts processed positions only (`min(n_non-EOS, 47)`, the 48th
   token excluded, not replayed); bank-mean profiles map onto each recipient by header, body, suffix and opening
   relative to its own prompt length.

## Changes in v3.1 (Alden's scoped pass on v3, council `part212-design-2026-11-18` #7)

Alden passed both v3 headline corrections and found one literal conflict: v3 said 24-row rates are never compared
with a 192-row arm, while `212.1` and `212.5` require exactly that rank. Wording only, preserving Neon's v2 rules:
the prohibition is now on pooling, not comparing; `212.1` and `212.5` name the full-arm share against each random
set's share as their rank, with the sample-0 rank separate and descriptive; the generic "192 everywhere" wording is
scoped; and `clean_google` is fixed as the exclusive first-named category on both sides of every recode, with the
legacy any-Google hit reported beside it as `legacy_any_google`. No arms, thresholds or comparisons changed.

## Amendment 1 (2026-11-19, Journal Claude, before any Stage 1 or Stage 2 data)

**The registered batch-size lever is not what the runner does, and the run proceeds at the shapes it records.** The
Throughput section (batch 32 for Stage 1 single-forward passes; batch 8/24 for Stage 2 with one prompt per batch,
"avoiding batch 1") assumed batches that the arm plan cannot fill. The runner groups Stage 2 rows by
`(arm, question, set)` and splits each group at the batch size; the 20 random control sets carry one sample per
held-out prompt (the "Pool random controls" lever above), so each of those rows is a group of one. The fix pass after
the GLM 5.3 code review (`part212/BUILD-REPORT.md`, F7) measured the full plan's Stage 2 histogram at batch 24 as
`1: 1,440 · 4: 36 · 8: 264 · 24: 8` and Stage 1 as 3,449 forwards, all at batch 1. Both histograms are now written
into each `run.json`, so every answer's batch size is on record.

- **Validity.** Batch 1 was avoided because of Part 178's batch-1 fault, which is the fused bf16 SDPA kernel; this
  run uses eager attention throughout, and the pilot gate requires `attn_implementation == "eager"`. Batch shape can
  still move an answer at an exact bf16 tie (Part 201); the recorded histograms let any such dependence be checked
  afterwards. No arm, prompt, threshold, prediction or comparison changes.
- **Time.** The ~46-minute estimate assumed batched generation and is withdrawn as unverified: 1,440 batch-1
  generations of up to 48 tokens are about 69,000 single-row decode steps on the A770. The pilot measures throughput
  by batch size. **Full Stage 2 proceeds only if the pilot's measured rates project the full plan under 4 hours and
  the pilot shows no stall pattern** (a batch-1 rate falling well below its own early rate). Otherwise cross-set
  batching (the 20 random sets as rows of one batch on the same prompt, each with its own clamp columns) is built,
  reviewed and amended in before the full run.
- **Open count.** The Size section's "3,989 short forward passes" in Stage 1 does not match the runner's 3,449
  (131 arms × 25 prompts + 5 bank openings × 25 + 25 + 24). The round-2 code review is asked to reconcile them before
  the pilot; if the runner is short of a registered arm, that is a code defect, not a new design.
