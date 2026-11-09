# Part 201: what moves a bf16 greedy answer when the batch shape changes, and does float32 on the same card stop it?

Procedure: v1
Registered {{YYYY-MM-DD HH:MM}} CT by Claude (Fable 5.2), before any data. Elise asked "What pulls? Fable is armed"
(Discord message 1569468659917652151, 2026-11-09); this was the first of the three things I named. The commit that adds
this file is the registration.

## Question

Part 200 regenerated 200 pushed answers one row at a time on the eager kernel and 76 of them differed from the same
rows decoded in chunks of 8. Every count this program publishes from generated text is one draw of whatever causes
that. This part asks four things about it, on Qwen3.5-4B on the A770:

1. Is it rounding at near-ties and nothing else? (Does the top-two margin at the point where two decodes part say so?)
2. Which parts of "shape" matter: the batch size, which rows share the batch, padding from mixed prompts, a new process?
3. How often does it flip a label, and in which direction? (These rates set the noise floor for counts.)
4. Does computing in float32 from the same bf16 weights, on the same card, give one answer whatever the shape, and
   would a check at low-margin steps be enough to get that answer cheaply?

It follows Parts 178, 183, 199 and 200. The plan it feeds is `BATCH-SHAPE-NOISE.md`.

## Prior art and existing data

- Closest published work, all read at the abstract on 2026-11-09 (entries in `PRIOR-ART.md`):
  - Yuan et al., "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference" (arXiv
    2506.09501, earlier titled "Give Me FP32 or Give Me Death?"): under bf16 greedy decoding, batch size, GPU count and
    GPU type change answers; their fix, LayerCast, stores weights in 16 bits and computes in float32.
  - "Greedy Decoding Is Not Precision-Invariant" (arXiv 2609.26621): bf16 against fp16 diverges on 49 to 100% of
    prompts; whether a step flips depends mainly on the top-two margin at the output head; recomputing only the head
    in float32 at low-margin steps helps at batch 4 or less and not at batch 8 or more in their tests.
  - "Same Probe, Different Numbers" (arXiv 2609.31796; Llama-3.1-8B, Qwen3-8B, Gemma-3-4B): a bf16 batch-size change
    leaves 25% of rows on different tokens by token 20; flips are symmetric; under float32 with only the batch size
    varied, none of 201,960 probe verdicts change. Its advice is to report per-example agreement.
  - "MarginGate" (arXiv 2605.30218): batch-induced flips are sparse, so verify only low-margin steps.
  - Cooper et al. (arXiv 2609.25624): fixed-order float32-accumulating kernels give bitwise-identical linear layers
    across three NVIDIA architectures.
  - Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (September 2025) and its
    `batch_invariant_ops`: the cause is kernels that change strategy with batch size.
  - This part differs in three ways: the model is pushed at one MLP neuron or direction per row (Part 163's units), the
    readout is our own labels and registered verdicts, and the card is an Intel Arc A770, which none of them test.
- Our own data, printed by `part201/existing201.py` into `results/part201/existing.md`:
  - Part 200: 76 of 200 batch-1 answers differ from chunks of 8; the first difference is at answer token 15.5 (median);
    the three labels flip on 2, 6 and 5 rows, in both directions (1 and 1, 4 and 2, 1 and 4).
  - Part 183's kernel record (A770, bf16, eager, eight plain prompts, 96 tokens): tokens differ between batch 1 and
    batch 8 on 3 of 8 rows. It leaves the CPU float32 reference at six places, five of them exact ties in the bf16
    logits (margin 0) and one at 0.125; the reference's own margin there is 0.008 to 0.10. Every bf16 margin in the
    record is a multiple of 0.0625. Over 1,814 positions decided from an identical prefix, the bf16 margin differs
    from the reference's by 0.05 at the median and about 0.21 to 0.25 at the 99th percentile.
  - The CPU float32 reference itself: batch 1 and batch 2 give identical tokens on 8 of 8 rows, margins within 6e-5;
    2.4% of its positions have a margin below 0.125 and 5.0% below 0.25.
  - Lessons already in `PROCEDURE.md`: Part 123 (5 of 18 do-nothing rows differed from plain until the shapes matched,
    then 60 of 60), Part 161 (compare arms generated in the same run), Part 178 (batch 1 against 2 to 16 on SDPA).

Kernel check: results/part183/kernel-check.json

## Method

Model Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, on the A770 (`xpu`), eager attention, loaded through
`neuron_report.load_model`; chat template with thinking off and no system prompt, as Part 163; greedy; 80 new tokens.
Pushes are Part 163's: dose +0.8 at the row-isolated hook, units from `part163/units163.json` (sha256 2ed3bef4…, the
file Part 200 checked).

**Rows.**

- Pushed set: 24 units, the first 12 neurons and first 12 random directions of
  `random.Random(201).sample(arm_units, 12)` taken per arm in file order, each on Part 163's 16 battery questions:
  384 rows.
- Plain set: the 16 battery questions and the 8 questions of `tools/kernel_reference.py`, no push: 24 rows.
- Subsets, fixed by the same seed: C-subset = the first 6 sampled neurons and first 6 sampled random directions on
  all 16 questions (192 rows); F-subset = the same 12 units on the first 4 identity and first 4 maker questions (96
  rows); F1-subset = the first 3 and 3 of those units on the same 8 questions (48 rows).

**Passes.** Each pass decodes its rows with `model.generate` and a recorder (below). bf16 passes run in one process
unless stated.

| Pass | Rows | Shape | Precision |
| --- | --- | --- | --- |
| A | pushed 384 | chunks of 8: per question, the 24 units in sample order | bf16 |
| A-again | first 16 rows of A | the same two chunks, repeated at the end of the same process | bf16 |
| B | pushed 384 | chunks of 8: per question, the 24 units in the order `random.Random(2011).shuffle` gives, re-drawn until no chunk has the membership of an A chunk | bf16 |
| M | pushed 384, plain 24 | chunks of 8 over rows shuffled across questions (`random.Random(2012)`), left-padded; plain rows in 3 chunks of their own | bf16 |
| C | C-subset 192, plain 24 | one row per call | bf16 |
| A2 | A's rows for battery questions 0 to 3 (96) | A's chunks, in a second process (new launch step) | bf16 |
| K | the 8 kernel prompts, 96 tokens, batch 1 and 2 | `tools/kernel_reference.py --device xpu --dtype float32-upcast` | float32 compute |
| F8 | F-subset 96 | chunks of 8, same membership and order as A's rows restricted to these questions and units | float32 compute |
| F1 | F1-subset 48 | one row per call | float32 compute |

The float32 passes run in a third process: after the load, `kernel_reference.upcast_compute` keeps every Linear and
Embedding weight in bf16 storage and computes everything in float32 (proved equal to a float32 load on Qwen3.5-0.8B,
`results/kernel-reference/upcast-proof/`). The push tensors are the bf16 passes' tensors, cast up, so both precisions
push the same values.

**Recorder.** A `LogitsProcessor`, last in the list, copies each step's scores to the CPU as float32 and stores, per
row and position: the top two token ids, the top logit, and the margin. It returns the scores unchanged. Each row also
stores its generated token ids, the pass, the call's row count and membership, its slot, its prompt length and its
padding. Rows append to `results/part201/rows.jsonl` and are skipped on a relaunch.

**Labels.** Part 163's, by its own code, on every decoded answer: name gone and maker gone (`part163/score163.py`),
broken (`part86/triage86.py`), and the Part 161 Qwen rule (`results/part163/extra163.py`).

**Analysis** (`part201/score201.py`, on the host). For a pair of passes, a row "differs" when its token ids differ. At
the first differing position the script reads both passes' margins. "Steps" are bf16 steps at the top logit's
magnitude (0.125 for a top logit from 16 up to 32). It prints, for every pair below: rows compared, rows differing,
first-difference positions, the margin at the first difference, each label's flips by direction, and mean length
difference; and over all bf16 rows the share of positions within 0, 1, 2 and 4 steps.

Pairs: A against A-again (16), A against A2 (96), A against B (384), A against M (384), plain M against plain C (24),
A against C (192), F8 against F1 (48), A against F8 (96), C against F1 (48), K against the CPU reference (16), and,
exploratory, A against Part 200's chunked answers for the same unit and question (384; another process and other
neighbours).

## Controls

- Do-nothing and matched-random controls of `STANDARD-CONTROLS.md`: Not used, because nothing is pushed or compared
  that Part 163 did not already control; the random-direction arm is in the sample, and the plain set is the no-push
  condition.
- **Gate G1, kernel and precision:** every bf16 load records eager and bf16; the float32 process records the upcast
  summary (`kept_bf16`, `converted_to_float32`, `restored_f32_from_checkpoint`); one model revision throughout.
- **Gate G2, recorder:** on every row, the emitted token at each position equals the argmax of the recorded scores. A
  failure means the margins do not describe the decode, and no margin reading is made.
- **Gate G3, rows and shapes:** every planned row is present once per pass; each call's recorded row count and
  membership match the plan.
- **Gate G4, same push:** for each F-subset row the push tensor's sha256 (bf16 bytes) is equal in the bf16 and float32
  processes.
- Positive control for "the analysis can see a difference": A against C must differ on at least 10 rows; Part 200 found
  38%. If it does not, the comparison code is checked before anything is read.
- Not used: recomputing only the output head in float32. arXiv 2609.26621 reports its benefit gone at batch 8 or more;
  pass F measures full float32 compute instead.
- Not used: a CPU float32 decode of pushed rows (about half a token a second on the 4B; the reference took 52 minutes
  for 16 plain rows). Pass K ties the A770's float32 to the CPU reference on the eight plain prompts.

Gate: G2. It could fail if `generate` applies another processor after the recorder or returns tokens from a path that
does not pass through it.

## Size

- Smallest effect worth knowing: a label flip rate of 3% against 0% (Part 200's order). With 384 pairs, 3% is 11.5
  flips and its 95% interval is about 1.6% to 5.1%; with 192, about 1.2% to 6.7%. For differing rows, 38% of 192 has
  an interval of about ±7 points. For the float32 passes, the claim is near zero against tens of percent, which 48
  rows separate.
- Total: 1,504 bf16 answers (384 + 16 + 384 + 408 + 216 + 96) and 160 float32 answers (16 + 96 + 48), 80 new tokens
  (96 for K). bf16 chunked, 1,288 answers: about 24 minutes at Part 200's 0.89 answers a second. bf16 one row per call,
  216 answers: about 15 minutes at Part 200's 0.24 a second. Float32: not measured anywhere on this card.
- Pilot: required (over 1,000 generations). The first 3 chunks of pass A time the recorder's cost; the float32 process
  starts with pass K at batch 1 on 2 prompts. Stopping rules: bf16 chunked under 0.3 answers a second, stop and fix
  the recorder; float32 at a rate that puts K, F8 and F1 together over 45 minutes, cut F8 to its first 48 rows and F1
  to its first 24, and record the cut before any float32 answer is read; a RAM-guard stop in the float32 process ends
  that arm, and 201.7 to 201.11 are reported as not run.

## Optimization

- Bound resource: GPU.
- Measured: 0.89 answers/s at chunks of 8 and 0.24 answers/s one row per call, Part 200's whole run (the same model,
  kernel, prompts and pushes). The recorder's cost is measured in the pilot and written here before the full run.
- Levers: batch size, declined (the shapes are the experiment); max_new_tokens, kept at Part 163's 80; rows leaving
  the batch, declined (a changing batch size inside a call would add a shape nobody recorded); vocabulary-size work on
  the CPU, taken (the recorder's top-2); copy engine off, taken (launcher default); host memory, taken (drop caches,
  11 GB free before each load, one model per process, three launch steps); resume, taken (rows append and are
  skipped).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 201.1 | A-again equals A on all 16 rows, and A2 equals A on all 96, token for token. | high | Claude (Fable 5.2) |
| 201.2 | Between A and C, 25% to 50% of the 192 rows differ. | medium | Claude (Fable 5.2) |
| 201.3 | Between A and M, at least 20% of the 384 rows differ. | low | Claude (Fable 5.2) |
| 201.4 | Between A and B, at most 10% of the 384 rows differ. | low | Claude (Fable 5.2) |
| 201.5 | Of the rows that differ between A and C, at least 95% have a pass-A margin of at most 2 steps at the first differing position. | high | Claude (Fable 5.2) |
| 201.6 | For each of the four labels, between A and M, the two directions of flip differ by at most twice the square root of the number of flips; and mean answer length differs by under 3 tokens. | medium | Claude (Fable 5.2) |
| 201.7 | K matches the CPU float32 reference token for token on all 16 rows up to the first position where the reference margin is under 0.001. | medium | Claude (Fable 5.2) |
| 201.8 | Between F8 and F1, at most 2 of the 48 rows differ. | medium | Claude (Fable 5.2) |
| 201.9 | Of the rows that differ between A and F8, at least 95% have a pass-A margin of at most 2 steps at the first differing position. | medium | Claude (Fable 5.2) |
| 201.10 | Between A and F8, 30% to 70% of the 96 rows differ. | low | Claude (Fable 5.2) |
| 201.11 | F8 runs at one quarter of pass A's answers per second or better. | low | Claude (Fable 5.2) |

Copied verbatim into PREDICTIONS.md under "## Part 201 (registered ...)".

Reasons, briefly. 201.1: Part 123 and Part 172's replays matched once shapes matched. 201.2: Part 200's 38% on one
question. 201.3: left padding moves every position, so every rounding changes; nothing of ours measures it. 201.4:
rows in a batch do not read each other, so only a slot effect could move them; a result above 10% would mean
membership has to be recorded and fixed, not just size. 201.5 and 201.9: six of six divergences in Part 183's record
sat at one step or less. 201.6: Part 200's flips went both ways. 201.7 and 201.8: the CPU's float32 batch 1 and 2
agree to 6e-5; the card's float32 kernels are untested. 201.10: on Qwen3.5-0.8B, bf16 differed from float32 on 8 of 16
rows. 201.11: the upcast converts every weight on every step; no measurement exists.

## Reading rules

- Every number is printed by `part201/score201.py` from `rows.jsonl`; verdicts as registered, by its comparison
  functions. The thresholds in 201.2, 201.5, 201.9 and 201.10 were chosen after reading Part 200 and Part 183's
  record, so those are checks that the earlier numbers repeat on new rows, not independent discoveries.
- G2 failing withdraws 201.5 and 201.9 and every margin table. G4 failing withdraws 201.9 and 201.10.
- A pair with no differing rows makes its "of the rows that differ" prediction not scored, and that is reported.
- A row cut short by a relaunch is regenerated whole; a row with a non-finite score at any step is counted, listed
  and left out of pair comparisons.
- The label flip rates are reported with 95% Wilson intervals and no bar. They are Part 163's labels at dose 0.8 on
  this model, and are used in `BATCH-SHAPE-NOISE.md` as a prior only until a part measures its own.
- What the float32 results decide is written before the run, in `BATCH-SHAPE-NOISE.md` under "After Part 201": margin-
  gated float32 decoding is built only if 201.7, 201.8 and 201.9 hold and the projected slowdown is at most 2 times.
- The headline reading "it is rounding at near-ties" is withdrawn if 201.5 fails; "float32 on the card is the same
  answer at any shape" is withdrawn if 201.7 or 201.8 fails.

## Limits

- One model, one card, one dose, Part 163's units and questions, 80 tokens. Loop and ban parts (165 to 170) have
  other labels and may flip at other rates.
- "Shape" is sampled at four points; batch 64, the shape most pre-audit parts used, is not run on eager here.
- The float32 passes define an answer; they do not show it is the right one. They remove the instrument's rounding,
  not the model's own near-ties: where the float32 margin is itself tiny, the model is close to undecided and a
  different prompt wording could move it.
- Pass K compares the card's float32 with the CPU's on plain prompts only.
- Nothing here re-scores an earlier part.
