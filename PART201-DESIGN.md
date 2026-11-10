# Part 201: what moves a bf16 greedy answer when the batch shape changes, and does float32 on the same card stop it?

Procedure: v1
Registered {{YYYY-MM-DD HH:MM}} CT by Claude (Fable 5.2), before any data. Elise asked "What pulls? Fable is armed"
(Discord message 1569468659917652151, 2026-11-09); this was the first of the three things I named. The commit that adds
this file is the registration.

## Question

Part 200 regenerated 200 pushed answers one row at a time on the eager kernel and 76 of them differed from the same
rows decoded in chunks of 8. Every count this program publishes from generated text is one draw of whatever causes
that. This part asks four things about it, on Qwen3.5-4B on the A770:

1. Is it bf16's limited resolution at near-ties and nothing else? (Do two decodes part only where the margin is
   inside bf16's own error, or also where the model was confident, as the SDPA fault did at margins up to 5.25,
   `PART178-DESIGN.md` Amendment 1?)
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
    prompts; whether a step flips depends mainly on the top-two margin at the output head. On recomputing only the
    head in float32 at low-margin steps, its abstract says: "delivers +22-36 pp exact agreement on A10G (+12-21 pp on
    L4 and A100) at less than 4% latency overhead in low-batch (batch size <=4) single-stream inference" and "its
    benefit vanishes when body-originated error dominates, including at batch size >=8 and under end-to-end FP8 in
    our tests."
  - "Same Probe, Different Numbers" (arXiv 2609.31796; Llama-3.1-8B, Qwen3-8B, Gemma-3-4B): a bf16 batch-size change
    "flips the first generated token for 2.1% of rows and leaves 25% on different tokens by token 20"; "Flips are
    symmetric"; "under float32 with only the batch size varied, none of 201,960 verdicts change" (probe verdicts, in
    its abstract's words). Its advice is to report per-example agreement.
  - "MarginGate" (arXiv 2605.30218): batch-induced flips are sparse (0.3 to 1.3% of decode steps in its abstract),
    so it verifies only low-margin steps, and where the verifier disagrees it also replaces that step's cached keys
    and values. Its always-on ancestor is LLM-42 (arXiv 2601.17768).
  - Reporting: arXiv 2605.19537 and 2608.04714 both ask that the inference stack and its version be reported; Muse's
    sweep (council `muse-prior-art` #16) found no paper or harness that publishes a per-verdict sensitivity column,
    and no batch-variance study on Intel Arc.
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
    batch 8 on 3 of 8 rows. It leaves the CPU reference at six places. There the reference's own margin is 0.008 to
    0.10, under one bf16 step, and the bf16 scores show an exact tie (five places) or one step of 0.125 (one). So
    these are near-ties in float32 that bf16 does not resolve: every bf16 margin in the record is a multiple of
    0.0625, and over 1,814 positions decided from an identical prefix the bf16 margin is off from the reference's by
    0.05 at the median and about 0.21 to 0.25 at the 99th percentile. The record's own `argmax_agrees` flag is false
    on four rows, each of which holds exact ties, so at a tie the card's choice and a CPU top-2 can name different
    tokens.
  - The CPU reference is float32 compute from bf16-stored weights (`--dtype float32-upcast`; a full float32 load of
    the 4B does not fit the host), proved equal to a float32 load on Qwen3.5-0.8B only. Its batch 1 and batch 2 give
    identical tokens on 8 of 8 rows, margins within 6e-5; 2.4% of its positions have a margin below 0.125 and 5.0%
    below 0.25.
  - Lessons already in `PROCEDURE.md`: Part 123 (5 of 18 do-nothing rows differed from plain until the shapes matched,
    then 60 of 60), Part 161 (compare arms generated in the same run), Part 178 (batch 1 against 2 to 16 on SDPA).

Kernel check: results/part183/kernel-check.json

## Method

Model Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, on the A770 (`xpu`), eager attention, loaded through
`neuron_report.load_model`; chat template with thinking off and no system prompt, as Part 163; greedy; 80 new tokens.
Pushes are Part 163's: dose +0.8 at the row-isolated hook, units from `part163/units163.json` (sha256 2ed3bef4…, the
file Part 200 checked).

**Rows.**

- Pushed set: 24 units on Part 163's 16 battery questions, 384 rows. The units are 12 neurons and 12 random
  directions: with each arm's units in file order, `random.Random(201).sample(arm_units, 12)` for the neurons and
  `random.Random(202).sample(arm_units, 12)` for the random directions. The unit order used everywhere below
  interleaves them: neuron 1, random 1, neuron 2, random 2, and so on, so every chunk of 8 holds 4 of each.
- Plain set: the 16 battery questions and the 8 questions of `tools/kernel_reference.py`, no push: 24 rows.
- Subsets: C-subset = the first 12 units in that order on all 16 questions (192 rows). F8-subset = the first 16 units
  on six questions, the three lowest-numbered identity and three lowest-numbered maker questions (96 rows); these are
  exactly pass A's first two calls for each of those questions. F1-subset = the first 8 units on the same six
  questions (48 rows), which lie inside both the C-subset and the F8-subset. Both float32 passes take the six
  questions in the order identity, maker, identity, maker, identity, maker, so their first half holds both kinds.

**Passes.** Each pass decodes its rows with `model.generate` and a recorder (below). bf16 passes run in one process
unless stated.

| Pass | Rows | Shape | Precision |
| --- | --- | --- | --- |
| A | pushed 384 | chunks of 8: per question, the 24 units in the interleaved order, three calls | bf16 |
| A-again | first 16 rows of A | the same two chunks, repeated at the end of the same process | bf16 |
| B | pushed 384 | chunks of 8: per question, the 24 units reshuffled (`random.Random(2011)`, re-drawn until no chunk has the membership of an A chunk and every row's slot differs from its slot in A) | bf16 |
| M | pushed 384, plain 24 | chunks of 8 over rows shuffled across questions (`random.Random(2012)`), left-padded; plain rows in 3 chunks of their own | bf16 |
| C | C-subset 192, plain 24 | one row per call | bf16 |
| A2 | A's rows for battery questions 0 to 3 (96) | A's chunks, in a second process (new launch step) | bf16 |
| K | the 8 kernel prompts, 96 tokens, batch 1 and 2 | `tools/kernel_reference.py --device xpu --dtype float32-upcast` | float32 compute |
| F8 | F8-subset 96 | the same 12 calls as in pass A: same rows, same slots | float32 compute |
| F1 | F1-subset 48 | one row per call | float32 compute |

A row's call in pass A2, A-again and F8 has the same rows in the same slots as its call in pass A, so those pairs
differ only in process, repetition or precision.

The float32 passes run in a third process: after the load, `kernel_reference.upcast_compute` keeps every Linear and
Embedding weight in bf16 storage and computes everything in float32 (proved equal to a float32 load on Qwen3.5-0.8B,
`results/kernel-reference/upcast-proof/`). The push tensors are the bf16 passes' tensors, cast up, so both precisions
push the same values.

**Recorder.** A `LogitsProcessor`, last in the list, copies each step's scores to the CPU as float32 and stores, per
row and position: the top two token ids, the top logit, the margin, and the score of the token the decode then
emitted. It returns the scores unchanged. Each row also stores its generated token ids, the pass, the call's row
count and membership, its slot, its prompt length, its padding and the launch that made it. The plan of every pass
(calls, membership, slots) is written once to `results/part201/plan.json`; its sha256 goes in the load record and a
relaunch refuses a different plan. The unit of work is the call: a call's rows are appended together after it
finishes, a torn tail is cut back to the last whole call, and a relaunch re-runs every unfinished call whole, with
its original membership and slots; rows are never repacked. Pass A-again runs straight after pass A in the same
launch; if pass A's first two calls were made by an earlier launch, A-again is recorded as not same-process and its
clause of 201.1 is not scored (`part201/ROWS-SCHEMA.md`).

**Timing probe.** At the end of the float32 process: full-prefix float32 forward passes with no cache, on the first
call's rows of pass F8, at 1, 4 and 8 rows and prefix lengths of 32, 64 and 96 tokens, five repetitions each,
synchronised, outputs discarded. The model body runs over the whole prefix and the output head on the last position
only, which is what a re-check needs. The push is added at every position of the prefix: the add costs the same, but
a real re-check must push at generated positions only, so the probe prices the work and does not validate a
re-check's answers. 201.11 is computed from it; F8's own cached step time is recorded beside it and is not used as
its stand-in. A flagged step whose prefix is longer than 96 tokens is priced by scaling the 96-token figure with
length: an approximation, not a measurement, and the scorer reports how many flagged steps it covers.

**Labels.** Part 163's, by its own code, on every decoded answer: name gone and maker gone (`part163/score163.py`),
broken (`part86/triage86.py`), and the Part 161 Qwen rule (`results/part163/extra163.py`).

**Analysis** (`part201/score201.py`, on the host). For a pair of passes, a row "differs" when its token ids differ. At
the first differing position (the position whose choice differs; both passes computed it from the same prefix) the script
reads both passes' margins, and whether the other pass's token is this pass's runner-up there (when it is not, the
top-two margin is not the gap between the two emitted tokens, and that row is listed). That is the deciding position: a fault shows there as a large margin (Part 178's SDPA
fault: up to 5.25), bf16's limited resolution as a small one. "Steps" are bf16 steps at the top logit's magnitude
(0.125 for a top logit from 16 up to 32). Where one pass of a pair is float32, its margin at that position says how
far apart the two tokens really were. At exact ties (bf16 margin 0) the script also prints which tied token was
emitted and whether two passes with the same tied scores emitted different tokens. It prints, for every pair below: rows compared, rows differing,
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
- **Gate G2, recorder:** on every row, at each position the emitted token's recorded score equals the top recorded
  score (it is the top token or tied with it). A failure means the recorded scores are not the ones the decode chose
  from, and no margin reading is made. Positions where the emitted token is tied with, and is not, the recorder's
  first-listed top token are counted and reported.
- **Gate G3, rows and shapes:** every planned row is present once per pass; each call's recorded row count,
  membership and slots match the plan; in pass B no call has an A call's membership and no row keeps its A slot.
- **Gate G4, same push:** in the float32 process the activation the hook adds to is float32 (its dtype is recorded
  per call), the hook fires on every decode step of every pushed row, and the delta tensor it is given equals the
  bf16 process's push values exactly (largest absolute difference 0, recorded per row, with the push tensor's
  sha256). This checks the prescribed delta, not that the sum after the add is identical across precisions. The hook casts the push to the activation's dtype
  (`part86/common86.py`), so if any activation reaches it in bf16 the float32 arm is not float32 there, and this gate
  fails.
- Positive control for "the analysis can see a difference": A against C must differ on at least 10 rows; Part 200 found
  38%. If it does not, the comparison code is checked before anything is read.
- Not used: recomputing only the output head in float32. arXiv 2609.26621 reports its benefit gone at batch 8 or more;
  pass F measures full float32 compute instead.
- Not used: a CPU float32 decode of pushed rows (about half a token a second on the 4B; the reference took 52 minutes
  for 16 plain rows). Pass K ties the A770's float32 compute to the CPU's on the eight plain prompts.
- Pass B changes slot and neighbours at once, with prompts fixed. Pass M changes prompt mix, padding, sequence
  length, neighbours and slots at once; it is the shape most parts actually use, and it is reported as a composite.
  Neither is decomposed here, and B is not subtracted from M.

Gate: G2. It could fail if `generate` applies another processor after the recorder or returns tokens from a path that
does not pass through it.

## Size

- Smallest effect worth knowing: a label flip rate of 3% against 0% (Part 200's order). With 384 pairs, 3% is 11.5
  flips and its 95% interval is about 1.6% to 5.1%; with 192, about 1.2% to 6.7%. For differing rows, 38% of 192 has
  an interval of about ±7 points. For the float32 passes, the claim is near zero against tens of percent, which 48
  rows separate. 0 of 48 differing bounds the true rate below 7.4% (Wilson, 95%).
- Power at the bars: a prediction whose true value sits exactly on its bar holds about half the time (201.3 at a
  true 20%: 51%). 201.2's range holds with probability 0.9997 at a true 38% and about 0.53 at a true 25% or 50%.
- Total: 1,504 bf16 answers (384 + 16 + 384 + 408 + 216 + 96) and 160 float32 answers (16 + 96 + 48), 80 new tokens
  (96 for K). bf16 chunked, 1,288 answers: about 24 minutes at Part 200's 0.89 answers a second. bf16 one row per call,
  216 answers: about 15 minutes at Part 200's 0.24 a second. Float32: not measured anywhere on this card.
- Pilot: required (over 1,000 generations). The first 3 chunks of pass A time the recorder's cost; the float32 process
  starts with pass K at batch 1 on 2 prompts. Stopping rules: bf16 chunked under 0.3 answers a second, stop and fix
  the recorder; float32 at a rate that puts F8 and F1 together over 45 minutes, cut F8 to its first 48 rows and F1
  to its first 24, and record the cut before any float32 answer is read. The rate is projected from F8's first call,
  with a one-row call taken as 0.4635 of an 8-row call (Part 200's bf16 rates: 0.24 against 0.89 answers a second).
  Pass K is its own launch step and is not in that sum. A cut arm is reported with its numbers, and 201.8, 201.9 and
  201.10 are then not scored. A RAM-guard stop in the float32 process ends that arm, and 201.8 to 201.11 are reported
  as not run; 201.7 is read from pass K's own record.

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
| 201.4 | Between A and B (other slots and other neighbours, same size and prompts), at most 10% of the 384 rows differ. | low | Claude (Fable 5.2) |
| 201.5 | Of the rows that differ between A and C, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has one above 8 steps. | high | Claude (Fable 5.2) |
| 201.6 | Between A and M, for each of the four labels with at least 8 flips, the two directions of flip differ by at most twice the square root of the number of flips; and mean answer length differs by under 3 tokens. | medium | Claude (Fable 5.2) |
| 201.7 | K matches the CPU reference token for token on all 16 rows, over every position. | medium | Claude (Fable 5.2) |
| 201.8 | Between F8 and F1, none of the 48 rows differ, with all 48 present and finite in both. | medium | Claude (Fable 5.2) |
| 201.9 | Of the rows that differ between A and F8, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has an F8 (float32) margin above 0.5 there. | medium | Claude (Fable 5.2) |
| 201.10 | Between A and F8, 30% to 70% of the 96 rows differ. | low | Claude (Fable 5.2) |
| 201.11 | The projected slowdown of margin-gated decoding on pass A is at most 2: (pass A's seconds + the summed cost of one full-prefix float32 forward for the flagged rows at every step where an unfinished row is within 2 steps) ÷ pass A's seconds, the cost of each forward read from the timing probe at the nearest larger row count and prefix length. | low | Claude (Fable 5.2) |

Copied verbatim into PREDICTIONS.md under "## Part 201 (registered ...)".

Reasons, briefly. 201.1: Part 123 and Part 172's replays matched once shapes matched. 201.2: Part 200's 38% on one
question. 201.3: left padding moves every position, so every rounding changes; nothing of ours measures it. 201.4:
rows in a batch do not read each other and a kernel should treat rows alike, so neither slot nor neighbours should
matter; above 10% means the arrangement inside a call has to be recorded and fixed, and a follow-up separates slot
from neighbours. 201.5 and 201.9: six of six divergences in Part 183's record sat at a bf16 margin of one step or
less, where the reference margin was at most 0.10; bf16's margin error there reaches about 2 steps at the 99th
percentile, hence 90% and not 100%. Both can fail: a fault of the SDPA kind parts decodes at margins of several
units. 201.6: Part 200's flips went both ways. 201.7 and 201.8: the CPU's float32 batch 1 and 2
agree to 6e-5; the card's float32 kernels are untested. 201.10: on Qwen3.5-0.8B, bf16 differed from float32 on 8 of 16
rows. 201.7: the reference's smallest margin on these rows is 0.008, far above float32's own noise (its batch 1 and 2
margins agree to 6e-5), so no position needs leaving out. 201.11: the upcast converts every weight on every forward
pass, which may dominate the cost, but prefix length and row count are measured by the probe, not assumed; no
measurement exists yet. This is the same bar the plan's decision uses.

## Reading rules

- Every number is printed by `part201/score201.py` from `rows.jsonl`; verdicts as registered, by its comparison
  functions. The thresholds in 201.2, 201.5, 201.9 and 201.10 were chosen after reading Part 200 and Part 183's
  record, so those are checks that the earlier numbers repeat on new rows, not independent discoveries.
- G2 failing withdraws 201.5, 201.9, 201.11 and every margin table. G4 failing withdraws 201.9 and 201.10.
- Gates G1 and G3 are read per precision. A float32 arm that did not run or fails its gates leaves the bf16
  predictions (201.1 to 201.6) standing. A bf16 gate failing, or the positive control failing, withholds every
  reading except 201.7, which rests on pass K's record alone.
- If pass A's first two calls and their repeat were not decoded by the same launch, 201.1 is scored on its A2 clause
  alone and says so.
- A pair with no differing rows makes its "of the rows that differ" prediction not scored, and that is reported, as is
  the number of differing rows behind every such percentage. A label with fewer than 8 flips is not scored in 201.6
  (with 4 or fewer the direction clause cannot fail). 201.11 is not scored without the timing probe.
- A row parting at a large margin is a trigger for an audit of that row, not by itself proof of a kernel fault; and
  small margins at the first differences do not prove rounding is the only mechanism.
- A row cut short by a relaunch is regenerated whole. A call with a non-finite score at any step is run once more;
  a row still non-finite is counted, listed and left out of pair comparisons, each affected pair is reported with
  its smaller n, and 201.8 is then not scored.
- The label flip rates are reported with 95% Wilson intervals and no bar. They are Part 163's labels at dose 0.8 on
  this model, and are used in `BATCH-SHAPE-NOISE.md` as a prior only until a part measures its own.
- What the float32 results decide is written before the run, in `BATCH-SHAPE-NOISE.md` under "After Part 201", as an
  ordered list whose default is not to build: a candidate margin-gated decoder is built only if 201.7, 201.8, 201.9
  and 201.11 all hold and the projected slowdown at the gate actually selected is at most 2; a prediction that is
  not scored counts as not held. 201.11 itself stays the two-step forecast it is registered as.
- The headline reading "it is rounding at near-ties" is withdrawn if 201.5 fails; "float32 on the card is the same
  answer at any shape" is withdrawn if 201.7 or 201.8 fails.

## Limits

- One model, one card, one dose, Part 163's units and questions, 80 tokens. Loop and ban parts (165 to 170) have
  other labels and may flip at other rates.
- "Shape" is sampled at four points; batch 64, the shape most pre-audit parts used, is not run on eager here.
- The float32 passes define an answer; they do not show it is the right one. They remove the instrument's rounding,
  not the model's own near-ties: where the float32 margin is itself tiny, the model is close to undecided and a
  different prompt wording could move it.
- Pass K compares the card's float32 compute with the CPU's on plain prompts only, and both are the same upcast
  method: an error shared by both (a checkpoint tensor restored wrongly on the 4B) would not show.
- Nothing here re-scores an earlier part.
- The verdict sensitivities from `tools/shape_floor.py` are conditional on the replicate's transitions carrying over
  to rows with the same starting labels. A starting label vector the replicate holds once redraws deterministically,
  and one it does not hold falls back to another model; the tool prints the support for each vector and marks a
  verdict that depends on unsupported rows.
- One driver. The card's Windows driver is 32.0.101.9150, installed 2026-10-12, so every A770 part since 2026-10-15
  ran on it, Part 178's SDPA finding included. An Intel Community report (2026-11-07, in `PRIOR-ART.md`) names this
  driver, against 32.0.101.8724, for a regression in another stack. Nothing we have run separates the card from its
  driver. Each launch's entry in `load.json` records the device as torch reports it and the Windows display-driver
  version, so a later driver change is visible in the record.
