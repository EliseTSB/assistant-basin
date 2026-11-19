# Part 212: do Qwen3.5-4B's four maker neurons carry maker-specific evidence, or a constant resting scaffold whose removal exposes the Google fallback?

Procedure: v1
DRAFT v1 2026-11-18 22:18 CT by Gemini 4 Neon (proposer and designer; awaiting cold design review before registration).
Proposed by Gemini 4 Neon and selected by Elise ("Option B", 2026-11-18 22:09 CT). This draft will be registered on
public main through `tools/register_public.py` only after cold design review and before any model run of Part 212.

## Question

Zeroing four MLP neurons (`L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776`) in Qwen3.5-4B moves held-out English maker
answers to Google in 43 of 192 answers (22.4%) on eager attention, against 2 of 192 unchanged and at most 1 of 24 for
50 same-layer random four-unit sets (Part 183, replicating Parts 25 and 66). Yet unselected flat random directions at
dose 0.225 also lead to Google in 30 of 40 directions (Part 176), three of the four units are off (< 1 SD from rest) at
a maker-ban veto step (Part 159), and Yavuz (2026) showed that neurons selected by zero ablation can transfer to held-out
questions while carrying only a constant background bias that disappears under resample ablation from unrelated prompts.

Do the four maker neurons carry question-specific maker evidence, an addressee ("talking to the AI") signal, or a
constant resting scaffold whose zeroing perturbs the residual stream into Qwen3.5-4B's Google fallback? And does
patching their activations from Chinese maker prompts into English maker prompts suppress Google, testing Finding 1's
conjecture for Part 32's 0/24 Chinese null?

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
  - What this part adds: an exact additive 2x2 orthogonal decomposition of the four published maker neurons
    ($a_{\text{maker}} = \bar{a}_{\text{fact}} + (a_{\text{maker}} - \bar{a}_{\text{fact}})$), pairing zero ablation
    against exact-token-length-matched off-topic factual resample, off-topic `"you"` addressee resample, resting-profile
    subtraction, delivered-write-norm-matched random controls, opposite-direction scaling ($+2\times$, $-1\times$),
    prompt-vs-generation phase splits, and a cross-lingual Chinese-donor swap on both forced-opening logits and held-out
    free generations.
- Our own data that already bears on it:
  - **Parts 25, 66 and 183 (`results/part183/README.md`)**: on eager attention (`results/part183/kernel-check.json`),
    zeroing `L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776` gives Google in 43/192 (22.4%) held-out English answers
    against 2/192 unchanged and max 1/24 (4.2%) across 50 same-layer random sets (matched only by layer, not by
    activation magnitude or write norm).
  - **Parts 70–77 (`results/part72/part72.json`, `results/part74/part74.json`)**: on `'I was developed by **'`
    (8-maker probability mass 0.982), zeroing `L24 N3949` shifts Google-minus-Alibaba by -2.234 logits, 64% mediated
    through `L27 N3325` (whose activation is negative to write Google: -4.463 with makers zeroed vs -2.049 with L24
    also zeroed). Part 75 showed flat prompt-mean clamping of `L27 N3325` is not neutral because natural activations vary
    across positions; this design therefore tests both single-donor natural-variance traces and bank-mean resting
    profiles.
  - **Part 32 (`part32.py`, `results/part32/score.txt`)**: zeroing the four gave Google in 16/24 English, 17/24 Spanish,
    and 0/24 Chinese answers.
  - **Part 39 (`PART39-DESIGN.md`) & Part 111 (`results/part111/README.md`)**: `L6 N753` tracks "talking to the AI";
    `L7 N1169` and `L11 N8868` write China/Chinese tokens through the Jacobian lens (8 and 7 of top 20, null ≤ 1.0%);
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
      company, identity, China or Google terms.
    - `D_zh_maker`: 25 Chinese maker phrasings.
    - `D_zh_fact`: 25 Chinese off-topic factual questions (separates Chinese language from Chinese maker framing).
  - Stage 1 forced openings (all **exact 5 tokens**, verified by tokenizer):
    - `EN_MAKER`: `'I was developed by **'` (Part 72/74's verified opening)
    - `D_fact`, `D_you`: `'The main answer is **'`
    - `D_zh_maker`: `'我的开发机构是**'`
    - `D_zh_fact`: `'这道题的答案是**'`
- **Delivered-write-norm-matched random controls (`20 sets`):** On the unchanged Stage 1 recording pass over the 24
  held-out `EN_MAKER` prompts (prompt + forced opening), compute every unit's mean delivered write norm
  $m_{L,u} = \text{mean}_{q,p}(|a_{L,u}(q,p)|)\cdot\|W_{\text{down}}^{(L)}[:, u]\|_2$ in layers $L \in \{6, 7, 11, 29\}$
  (summed in float64 on CPU). In each layer, select the 50 units closest to the target unit in $m_{L,u}$ (excluding the
  target, ties broken by unit index). Draw 20 four-unit control sets (`random_0`..`random_19`) with
  `np.random.default_rng(21200 + k)` (one unit per layer from its 50-unit pool). For every arm and control, record the
  mean realized L2 residual-stream perturbation $\|\sum_u \Delta a_u \cdot w_{\text{down}, u}\|_2$ alongside the
  behavioral and logit readouts.

### Stage 1: Exact single-forward 2x2 orthogonal decomposition (24 held-out prompts + `'I was developed by **'`)

1. **Unchanged recording pass:** Record `TARGET`, all 4×50 matched-pool units, `L24 N3949` and `L27 N3325` at every
   position on `EN_MAKER`, `D_fact`, `D_you`, `D_zh_maker` and `D_zh_fact` (each with its 5-token opening; 125 passes).
   Construct the bank-mean resting profiles $\bar{a}_{\text{fact}}(p)$ and $\bar{a}_{\text{you}}(p)$: exact per-position
   mean across the 24 held-out donors on the 12 shared template positions and 5 forced-opening positions, and pooled
   mean across all 24 donors' user-body tokens on user-body positions.
2. **Intervention pass on `EN_MAKER[0..23]` (+ `'I was developed by **'`, same prompt per batch of 32 → zero padding):**
   - **14 main arms** on `TARGET` (all positions unless noted):
     - `unchanged`: no hook
     - `self_patch` (Gate 1): $a(p) \mapsto a_{\text{maker}}(p)$ (exact do-nothing patch)
     - `zero`: $a(p) \mapsto 0$
     - `trace_resample_fact`: $a(p) \mapsto a_{\text{fact}, q}(p)$ (Yavuz single exact-length paired donor trace)
     - `trace_resample_you`: $a(p) \mapsto a_{\text{you}, q}(p)$ (preserves resting + `"you"` addressee trace)
     - `mean_resample_fact`: $a(p) \mapsto \bar{a}_{\text{fact}}(p)$ (algebraic partner of `sub_factual`)
     - `sub_factual`: $a(p) \mapsto a_{\text{maker}}(p) - \bar{a}_{\text{fact}}(p)$ (removes resting profile only)
     - `sub_you`: $a(p) \mapsto a_{\text{maker}}(p) - \bar{a}_{\text{you}}(p)$ (removes resting + `"you"` profile)
     - `scale_pos2`: $a(p) \mapsto 2\cdot a_{\text{maker}}(p)$
     - `scale_neg1`: $a(p) \mapsto -1\cdot a_{\text{maker}}(p)$
     - `zero_prompt_only`: zero on prompt positions only (before the 5-token opening)
     - `zero_opening_only`: zero on the 5 forced-opening positions only
     - `trace_resample_zh_maker`: $a(p) \mapsto a_{\text{zh\_maker}, q}(p)$
     - `trace_resample_zh_fact`: $a(p) \mapsto a_{\text{zh\_fact}, q}(p)$
   - **16 per-unit arms:** each of the 4 target units individually under `zero`, `trace_resample_fact`,
     `trace_resample_you` and `sub_factual`.
   - **100 matched random control arms:** all 20 delivered-write-norm-matched random 4-unit sets under `zero`,
     `trace_resample_fact`, `trace_resample_you`, `sub_factual` and `scale_neg1`.
3. **Stage 1 readouts (all computed in float32 from final-position logits):**
   - Primary logit contrast: $g_{\text{ali}} = \text{logit}(\text{Google}) - \text{logsumexp}(\text{logit}(\text{Al}), \text{logit}(\text{T}))$,
     where `Al` (2033) and `T` (51) are the first tokens of `Alibaba` and `Tongyi`, and `Google` (14030) is single-token.
     For every arm, $\Delta g_{\text{ali}}$ is its paired per-prompt mean difference from `unchanged`.
   - Secondary readouts reported alongside (not replacing primary): Part 74's exact $g_{\text{al}} = \text{logit}(\text{Google}) - \text{logit}(\text{Al})$,
     8-maker first-token probability mass `mass` (`Alibaba`, `Google`, `OpenAI`, `Anthropic`, `DeepSeek`, `Meta`,
     `Microsoft`, `Tongyi`), **full multi-token sequence log-probability contrast**
     $g_{\text{seq}} = \log P(\text{"Google"}) - \max(\log P(\text{"Alibaba"}), \log P(\text{"Tongyi"}))$ on all 14 main
     arms (guards against the Part 154/158 first-token prefix trap), and last-position activations of `L24 N3949` and
     `L27 N3325`.

### Stage 2: Held-out free generation (exact Part 66 / Part 183 rig, eager attention)

1. **Donor generation recording:** Run greedy generation (`max_new_tokens=48`, eager, bf16) once per donor prompt in
   `D_fact`, `D_you` and `D_zh_maker` (75 runs), recording `TARGET` and all 20 matched-control units at every prompt
   position and every non-EOS generated position. Compute the pooled non-EOS generated-token mean $\mu_{\text{fact, gen}}$
   for `mean_resample_fact` and `sub_factual`. For `trace_resample_*`, if a donor ends at step $k < 48$, cycle over its
   non-EOS generated positions $0\dots k-2$ (never injecting an EOS-token activation); record every donor's pre-EOS
   length in `run.json`.
2. **Sampled generation (`temperature=0.7`, `top_p=0.95`, `max_new_tokens=48`, batch size 24, one prompt per batch →
   zero padding, SHA-256 per-row seeds matching Part 66's convention):**
   - **6 full arms (240 answers each = 1,440):** `unchanged`, `zero`, `mean_resample_fact`, `sub_factual`, `scale_pos2`,
     `scale_neg1` — each run on the 24 held-out prompts (8 samples = 192), positive control `"Who created you?"` (24
     samples), and the 6 fluency guards (4 samples = 24).
   - **2 exact-length trace-resample arms (216 answers each = 432):** `trace_resample_fact`, `trace_resample_you` — each
     run on the 24 held-out prompts (192) and `"Who created you?"` (24). (Guards have different prompt lengths, so
     arbitrary-length fluency is checked on `mean_resample_fact` and `sub_factual` instead.)
   - **3 diagnostic arms (192 held-out answers each = 576):** `trace_resample_zh_maker`, `zero_prompt_only` (zero on
     prompt prefill only), `zero_gen_only` (zero on generated tokens only).
   - **20 matched random control sets (1,440 answers):** each set run on the 24 held-out prompts (1 sample each,
     matching Part 66's random design) under `zero`, `trace_resample_fact` and `sub_factual` ($20 \times 24 \times 3$).

Kernel check: results/part183/kernel-check.json

## Controls

Applying `STANDARD-CONTROLS.md` and `TOOLKIT.md`:

- **Both directions (`STANDARD-CONTROLS.md` #1):** `scale_pos2` ($+2\times$) and `scale_neg1` ($-1\times$) beside
  `zero` ($0\times$) in both Stage 1 and Stage 2.
- **Poles (#2):** Not used as a separate selection step, because this part intervenes on Part 25's four fixed units and
  records each unit's signed activation and sign-inverted (`scale_neg1`) effect directly.
- **Do-nothing hook (#3):** `self_patch` in Stage 1 (patches each unit with its own recorded clean activation; must
  match `unchanged` within 0.05 logits, exact 0 on same batch shape) and clean hook-removal check before/after Stage 2.
- **Matched random controls (#4, Toolkit I):** 20 four-unit sets matched on **mean delivered write norm**
  $\text{mean}(|a|)\cdot\|w_{\text{down}}\|_2$ per layer on the 24 maker prompts, evaluated under `zero`,
  `trace_resample_fact` and `sub_factual` (plus `trace_resample_you` and `scale_neg1` in Stage 1).
- **Answer-polarity split (#5):** Not used: the Stage 2 outcome is a six-family maker classification, not a binary
  yes/no label; Stage 1 prints both Google and Alibaba/Tongyi logits and total 8-maker probability mass (Toolkit J).
- **Broken answers counted (#6):** Every Stage 2 answer is screened by a fixed mechanical rule (`broken` if empty,
  under 4 words, contains `\ufffd` or `!!!!`, or repeats any 3-word n-gram $\ge 4$ times). Primary Stage 2 rates use
  fixed denominator 192 (`clean_google_rate = count(unbroken and google) / 192`) so broken answers can never inflate a
  rate. Off-target guards (Toolkit C: Part 66's 6 factual guards) are checked on all 6 arbitrary-length arms.
- **Dose ladder (#8, Toolkit E):** Covered on the multiplicative axis by `scale_pos2` ($+2$), `unchanged` ($+1$),
  `zero` ($0$), and `scale_neg1` ($-1$); a finer fractional knockout ladder ($d = 0.5, 0.75, 1.0$) was already run in
  Part 172.
- **Prompt vs. generation phase (Toolkit F):** `zero_prompt_only` vs `zero_gen_only` in Stage 2 and `zero_prompt_only`
  vs `zero_opening_only` in Stage 1.
- **Hand-read check:** Up to 30 Google-classified held-out answers per non-baseline main arm are exported for a manual
  attribution check (maker claim vs. mere mention or negation), matching `part66/score66.py`'s `hand-read.md`. Not used:
  an LLM blind reader, because maker classification is a fixed lexical rule (`part66/score66.py`) audited by hand read.

Gates (a failed gate stops reading the affected stage and is reported publicly):

- **Gate 0 (tokenizer & prompt parity, passed pre-registration):** `part212/build_prompts212.py` verifies 25/25 exact
  token-length match across all four donor banks, exact 5-token openings, distinct 8-maker first tokens, and zero
  lexical-screen violations (`prompts212.json` SHA-256 pinned above).
- **Gate 1 (Stage 1 do-nothing hook & logit mass):** `|mean g_ali(self_patch) - mean g_ali(unchanged)| < 0.05` logits;
  zero non-finite logits or activations; mean 8-maker probability mass `>= 0.50` in both `unchanged` and `zero`.
- **Gate 2 (Stage 1 positive control):** `zero` raises mean `g_ali` over `unchanged` by at least `+1.0` logit and
  exceeds at least 18 of 20 matched random `zero` sets; failure stops Stage 1 mechanism readings (212.2–212.3, 212.6–212.8, 212.10).
- **Gate 3 (Stage 2 kernel, completeness & fluency):** `run.json` records `attn_implementation == "eager"`, zero
  non-finite/empty rows, every planned key present once, and `unchanged` guard accuracy `>= 0.90` with `unchanged`
  broken rate `<= 0.05`.

## Size

- Smallest effect worth knowing: a 50% reduction in `zero`'s held-out Google rate (~22% down to ~11%) or a +5 percentage
  point rise above `unchanged` (~1% up to ~6%) on $n = 192$ held-out answers (exact Part 66/183 sample size: paired
  McNemar power > 0.85 for 22% vs 11%, and > 0.85 for 6% vs 1% at $\alpha = 0.05$). In Stage 1, 24 held-out prompts
  resolve a paired logit shift of 0.25 logits (Part 74 SE across 24 prompts was ~0.08 logits).
- Total: 3,963 generations (Stage 2, plus 3,917 short forward passes in Stage 1), about 46 minutes on the A770.
- Pilot: Stage 1 (3,917 forward passes, ~2 min) + 75 greedy donor recordings (~1 min) + a Stage 2 mini-pilot written to
  `results/part212/pilot/` (4 held-out prompts indices 0, 5, 12, 16 × 4 samples + `"Who created you?"` × 24 samples
  across `unchanged`, `zero`, `trace_resample_fact`, `sub_factual` = 160 sampled generations, ~2 min; total ~5 min).
  Stopping rule: do not launch full Stage 2 if Gate 1 or Gate 2 fails in Stage 1, if any pilot row is empty or
  non-finite, if `run.json` does not record `eager`, or if `zero` produces fewer than 2 Google answers of 24 on
  `"Who created you?"` in the pilot (Part 183 observed 8 of 24; $P(\le 1\mid p=1/3) < 0.002$). Full Stage 2 runs clean
  into `results/part212/stage2/` so pilot and full batch shapes never mix.

## Optimization

- Bound resource: GPU (Intel Arc A770, 16 GB VRAM; host Windows RAM ceiling ~7.5 GB during Qwen3.5-4B load).
- Measured: Part 183's eager run on this exact model, revision, card, dtype and 48-token cap measured **1.51 answers/s
  at batch 25** (1,880 answers in 1,241.3 s, peak GPU memory 9.09 GB, `results/part183/part66-eager/run.json`).
- Levers (each item from `PROCEDURE.md` section 4):
  - **Batch size:** taken — batch 32 for Stage 1 single-forward passes (identical prompt per batch, zero padding); batch
    24 (or 8 for single-arm 8-sample prompts, and 24 for positive-control/random-set batches) for Stage 2, keeping one
    prompt per batch with zero padding so left-pad tokens never enter the layers (Part 66 rule) and avoiding batch 1
    (Part 178 lesson). Pilot times batch 8 vs 24 on real rows before full Stage 2.
  - **max_new_tokens:** taken — capped at 48 tokens (exact Part 66/183 budget, sufficient to capture the maker claim in
    the first sentence while minimizing long-tail padding).
  - **Teacher-forced scoring instead of generation:** taken — Stage 1 runs the full 130-arm matrix (including all 16
    per-unit decompositions and all 100 matched random arms) as single-forward passes in ~2 minutes before committing to
    Stage 2 generation.
  - **Pool random controls:** taken — all 20 matched random sets share one Stage 1 recording pass and one donor greedy
    recording pass, and Stage 2 evaluates them at 1 sample per held-out prompt (24 answers per set, exact Part 66
    budget).
  - **Vectorized hook / no per-token CPU sync:** taken — hooks slice/assign tensor columns directly on-device with no
    `.item()` calls inside forward hooks; mean-norm accumulation sums on-device per batch and moves to CPU float64 once
    per prompt.
  - **Vocabulary-size work on CPU:** taken — softmax/logsumexp over 248,320 logits computed in float32 only at the last
    token position (no vocab top-k on XPU).
  - **Cache and resume:** taken — donor activations are saved once and reused; Stage 2 appends JSONL rows keyed by
    `(arm, question_index, sample)` and skips completed rows on resume; `torch.xpu.empty_cache()` called between
    prompts to prevent VRAM spill into Windows RAM.
  - **Early-exit generation loop:** declined — 48-token batches finish together in ~10–15 s and custom dynamic batching
    would alter batch shapes relative to Part 183.
  - **Direct load / copy engine off:** taken — launched via `tools/launch_arc.py` (copy engine off by default) when no
    other GPU or heavy WSL job is running.
  - **Labelling API:** n/a — fixed lexical + mechanical broken-answer scorer, zero API cost.
  - **Device:** A770 local (cost $0; full run ~46 min).

## Predictions

All predictions authored by Gemini 4 Neon, registered before any Part 212 model run. Stage 2 rates use fixed
denominator 192 (`clean_google_rate = count(unbroken and google) / 192`).

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 212.1 | Stage 2 replication against write-norm-matched nulls: `zero` has clean Google rate between 12% and 35% of 192 held-out answers, `unchanged` has at most 3 of 192 (<= 1.6%), and `zero` strictly exceeds all 20 delivered-write-norm-matched random `zero` sets (empirical p = 1/21). Fails if any clause misses. | high | Gemini 4 Neon |
| 212.2 | Stage 1 Yavuz resample test (exact logits): `zero` raises mean `g_ali` by at least +1.5 logits over `unchanged` and beats all 20 matched random `zero` sets, AND `trace_resample_fact` retains strictly less than 50% of `zero`'s mean $\Delta g_{\text{ali}}$ ($\Delta g_{\text{ali}}(\text{trace\_resample\_fact}) / \Delta g_{\text{ali}}(\text{zero}) < 0.50$). Fails if ratio >= 0.50 or `zero` misses. | medium | Gemini 4 Neon |
| 212.3 | Stage 1 resting-scaffold sufficiency: `sub_factual` ($a_{\text{maker}} - \bar{a}_{\text{fact}}$) raises mean `g_ali` by at least +0.75 logits, retains at least 35% of `zero`'s mean $\Delta g_{\text{ali}}$, and exceeds at least 18 of 20 matched random `sub_factual` sets. Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.4 | Stage 2 Yavuz resample test (free generation): on the 192 held-out answers, `trace_resample_fact`'s clean Google rate is strictly less than half of `zero`'s clean Google rate (ratio < 0.50) AND the paired question-clustered 95% bootstrap CI of `rate(zero) - rate(trace_resample_fact)` lies strictly above 0. Fails if ratio >= 0.50 or CI includes 0. | medium | Gemini 4 Neon |
| 212.5 | Stage 2 resting-scaffold sufficiency: on the 192 held-out answers, `sub_factual` exceeds `unchanged` in clean Google rate by at least 5 percentage points (`rate(sub_factual) - rate(unchanged) >= 0.05`), exceeds at least 18 of 20 matched random `sub_factual` sets, and has broken-answer rate <= 15%. Fails if any clause misses. | medium | Gemini 4 Neon |
| 212.6 | Addressee ("you") vs. factual separation: in Stage 1, `trace_resample_fact` exceeds `trace_resample_you` in mean $\Delta g_{\text{ali}}$ by at least +0.25 logits, AND in the single-unit 4x4 decomposition `L6 N753` shows a strictly larger `trace_resample_fact - trace_resample_you` gap than each of the other three units (`L7 N1169`, `L11 N8868`, `L29 N776`). Fails if either clause misses. | medium | Gemini 4 Neon |
| 212.7 | Downstream Google-writer routing: across the 10 core English intervention arms in Stage 1 (`unchanged`, `zero`, `trace_resample_fact`, `trace_resample_you`, `mean_resample_fact`, `sub_factual`, `scale_pos2`, `scale_neg1`, `zero_prompt_only`, `zero_opening_only`), each arm's mean last-position shift in `L27 N3325` correlates negatively with its mean $\Delta g_{\text{ali}}$ (Spearman $\rho \le -0.70$). Fails if $\rho > -0.70$. | high | Gemini 4 Neon |
| 212.8 | Opposite-direction monotonicity (`STANDARD-CONTROLS.md` #1): in Stage 1, `scale_pos2` ($2a$) lowers mean `g_ali` relative to `unchanged` by at least 0.30 logits ($\Delta g_{\text{ali}} \le -0.30$), AND `scale_neg1` ($-a$) raises mean `g_ali` strictly more than `zero` ($\Delta g_{\text{ali}}(\text{scale\_neg1}) > \Delta g_{\text{ali}}(\text{zero})$). Fails if either clause misses. | medium | Gemini 4 Neon |
| 212.9 | Prompt vs. generation phase (`TOOLKIT.md` F): in Stage 2, `zero_gen_only`'s clean Google rate exceeds `zero_prompt_only`'s clean Google rate by at least 4 percentage points (`rate(zero_gen_only) - rate(zero_prompt_only) >= 0.04`). Fails below 0.04 (including if both are near zero because both phases are jointly required). | low | Gemini 4 Neon |
| 212.10 | Cross-lingual Chinese-donor test (`README.md` Finding 1 conjecture): in Stage 1, patching Chinese maker-prompt activations (`trace_resample_zh_maker`) into English maker prompts gives mean $\Delta g_{\text{ali}} > -0.25$ logits (does not actively strengthen Alibaba/Tongyi over English prompt activations). Fails if $\Delta g_{\text{ali}} \le -0.25$. | low | Gemini 4 Neon |

Copied verbatim into PREDICTIONS.md under "## Part 212 (registered ...)" at registration time.

## Reading rules

- **Scorer:** `part212/score212.py` computes every gate, primary/secondary metric, 10,000-draw question-clustered
  bootstrap 95% CI (seed 212), realized perturbation norm, and prediction verdict directly from the saved Stage 1 JSON
  and Stage 2 JSONL files. No number is typed by hand.
- **Gate enforcement:** If Gate 1 or Gate 2 fails, Stage 1 predictions (212.2, 212.3, 212.6, 212.7, 212.8, 212.10) are
  marked `untestable (gate failed)`. If Gate 3 fails, Stage 2 predictions (212.1, 212.4, 212.5, 212.9) are marked
  `untestable (gate failed)`.
- **Broken answers & errors:** Any non-finite or empty generation aborts/retries the batch once and fails Gate 3 if
  unresolved. Mechanically broken answers are counted and reported for every arm and excluded from `clean_google_rate`
  while keeping denominator 192 fixed. Raw `part66/score66.py` lexical rates are printed beside clean rates.
- **Hand-read audit:** Up to 30 Google-classified held-out answers per non-baseline main arm are read by eye to separate
  genuine maker claims from negations/mentions; if > 20% of Google-matched answers in any headline arm (`zero`,
  `trace_resample_fact`, `sub_factual`) are non-claims (negations/mentions), the headline claim is scoped and qualified
  with the hand-audited claim rate.
- **Headline reading & withdrawal rule:**
  - **"Resting-scaffold reading supported"** only if Gates 1–3 pass AND 212.1, 212.2, 212.3, 212.4 and 212.5 all hold
    (and `g_seq` agrees in sign and >50% / <50% split direction with `g_ali` for 212.2 and 212.3).
  - **"Question-specific maker evidence supported"** if Gates 1–3 pass, 212.1 holds, AND `trace_resample_fact` retains
    $\ge 50\%$ of `zero` in BOTH Stage 1 (212.2 fails) and Stage 2 (212.4 fails) while `sub_factual` fails 212.3 and
    212.5.
  - **Otherwise ("mixed / split across units or stages"):** report the exact 2x2 table (including `mean_resample_fact`
    vs `trace_resample_fact`, realized perturbation norms, and the 4x4 per-unit breakdown) without forcing a single-cause
    headline. If 212.1 fails (e.g., write-norm-matched random sets match `TARGET`), withdraw the Finding-1 specificity
    premise for this control family and report that finding first.

## Limits

- One model (`Qwen/Qwen3.5-4B` at one snapshot), one four-neuron set (`L6 N753`, `L7 N1169`, `L11 N8868`, `L29 N776`),
  and Part 66's 24 English held-out phrasings.
- Stage 1's forced opening `'I was developed by **'` constrains the syntactic frame; Stage 2 checks free generation,
  where sampled answers depend on temperature and batch RNG.
- Even with exact 1-to-1 token-length matching, user-body tokens at position $p$ have different words across donor and
  target prompts; comparing `trace_resample_fact` (natural per-position variance) against `mean_resample_fact`
  (smoothed bank mean) bounds how much single-donor token alignment matters, but cannot eliminate nonlinear
  cross-position interactions.
- Realized perturbation norms may differ across `zero`, `trace_resample_fact` and `sub_factual` depending on the
  natural sign and spread of each unit; perturbation norms are logged and reported beside every arm so size differences
  are never conflated with semantic content.
- Stage 2's cross-lingual swap tests patching Chinese activations into English prompts, not the reverse direction on
  Chinese generations (Part 32 already observed 0/24 Google under Chinese zeroing); Chinese-target reverse checks are
  left for a follow-up if 212.10 fails.
