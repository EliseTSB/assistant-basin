# Part 200: Part 163 on the eager attention kernel

Procedure: v1
Registered: 2026-11-09 13:39 CT, before any data

Part 178 found the Intel Arc A770's fused bf16 SDPA attention kernel leaves the CPU float32 path once the cache holds
more than 64 keys. Part 163 generated 3,416 greedy 80-token answers on that kernel at batch 64. Its headline has two
parts: neurons said Qwen less than random directions (3.5% against 8.6%, an unregistered comparison) and broke far more
often (21.4% against 5.2%, prediction 163.5). Its broken answers are mostly repeated three-word sequences (329 of 343
neuron and 83 of 83 random, `results/part163/README.md`), the kind of decode output a kernel fault could change, and
Part 199 found the same kernel ended most answers early, many mid-phrase. The audit lists Part 163 as a high-risk part
(`notes/reports/arc-sdpa-audit-2026-11-03/PRIORITY1-CLAIMS.md`, kept in the Journal repository). This part reruns Part
163 with four changes, listed in the Method: the kernel, the batch shape, the pilot's rate threshold, and one added
stage that regenerates 200 rows at batch 1.

## Question

Part 163's published claims: under one selection rule, neurons still say Qwen less than random directions (19 of 539
maker answers against 57 of 661, Part 161's rule) and break far more often (343 of 1,600 battery answers against 83 of
1,600); its five registered verdicts are 163.1 to 163.4 held and 163.5 failed. Do its gates, counts and verdicts hold
when the same model, units and prompts run with eager attention?

## Prior art and existing data

- Part 178: SDPA leaves the CPU float32 path past 64 keys. Part 183's kernel check
  (`results/part183/kernel-check.json`) passes eager at batches 1 and 2 against
  `results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json` (`tools/part_check.py kernel`: 21 of 21 checks); its SDPA
  control fails. The record's own self-check flag `argmax_agrees` is false (prompt 4 at batches 1, 2 and 8, prompt 6 at
  batch 1). Batch 8 is not compared by the tool: by hand, eager batch 8 leaves the batch-1 path on prompt 7 at token 63,
  at a reference margin of 0.1006. Every divergence in the record is at a margin of 0.0084 to 0.1006, consistent with
  Part 178's finding that eager bf16 leaves the reference at near-ties. So chunks of 8 are a shape checked by hand, not
  by the tool; this part adds its own batch-1 comparison (Method, item 4).
- Part 199 (Part 170 on eager, published 2026-11-09): every gate passed and the push result held, but the loop rates
  rose (dose-1 b50 43.8% to 54.7%), one verdict's point estimate crossed its bar, and the stopping pattern changed: on
  SDPA 358 to 383 of 384 dose-1 answers per cell ended before the 80-token cap, most loops among them; on eager 104 to
  168, and almost every loop ran to the cap. On eager, 1,272 of 1,816 loops had formed before 64 keys, so the eager runs
  do not show that late loops are the kernel's. Its plain one-row answers also changed between kernels below 64 keys
  for 42 of 200 prompts. Part 191 found SDPA answers cut short on Part 153 (230 of 838 under the cap ended mid-phrase,
  2 of 588 on eager).
- Parts 183 to 199 reran fifteen high-risk parts on eager. Counts of named content mostly held; counts of decode
  pathology (Part 39's AI claims, Part 81's coherence, Part 170's frozen prompts) did not.
- Part 163's own run: 3,416 rows, batch 64, pilot 2.83 answers per second (`results/part163/run-stdout.txt`); Gate 1
  changed share 0.88 in each arm; 124 of 1,600 neuron and 218 of 1,600 random battery answers reach the 80-token cap
  (`token_count` 80), mean 55.1 and 59.2 tokens.

Kernel check: results/part183/kernel-check.json

## Method

Run `part163/run163.py`'s `run()` through `part200/run200.py` on the A770 (`xpu`), writing
`results/part200/answers.jsonl`. `run()` already takes a `loader` and a `generate_fn`. The wrapper changes four things
and records them:

1. **Kernel.** The loader loads through `neuron_report.load_model` (eager by default since 2026-11-03), applies Part
   163's no-grad and eval settings, and exits 3 before any generation unless `neuron_report.LAST_LOAD` and the model's
   `config._attn_implementation` both record eager. Part 199's folder lock and load record (`part199/run199.py`) are
   reused by import where they fit: `results/part200/load.json` gets one entry per launch as soon as the model loads,
   rewritten after every chunk with the calls and rows so far. `run200.py --check-record results/part200/load.json`
   repeats Gates R1 and R3 over every recorded launch on the host.
2. **Batch shape.** Part 163 generated each pushed call (one prompt, 64 or 8 units) as one batch. The wrapper's
   `generate_fn` generates every pushed call in chunks of exactly 8 rows (every pushed call's row count is 64 or 8) in
   the original row order, slicing each layer's push tensor to the chunk's rows, so every row keeps its own push (the
   hook is row-isolated). The 16 no-push calls (one row each) keep their shape. Every call and its chunks are recorded.
3. **Pilot stop rule.** Part 163 stops if its first 3 recipe batches run under 1.5 answers per second, a rate chunks of 8
   on eager will not reach (Part 199 ran about 0.73). `part163/run163.py` gets one behaviour-preserving edit: the literal
   `1.5` becomes a module constant `PILOT_MIN_RATE = 1.5`, with a test that its default is 1.5. The wrapper sets it to
   0.17 answers per second, the rate below which the 3,616 answers of this part (run plus item 4) would not finish within
   the job runner's 6-hour task limit, so a too-slow run stops as a HOLD instead of being killed. It is a guard against
   a broken path, not a time budget, and it checks speed, not content; Part 163's own Gate 1 check inside the pilot
   (changed share at least 0.70 per arm) is the content check and is unchanged. Both pilot stops raise `SystemExit`
   inside `run163.run`; the wrapper records the stop and exits 3. A run killed at 6 hours anyway is relaunched on the
   same folder after the operator checks no run200 process runs and removes the lock by hand; `run163` resumes at the
   next unwritten row.
4. **Batch-1 comparison (new, after the battery).** The wrapper regenerates battery question 0 ("What is your name?",
   checked in Part 163's answers) for all 200 units in `units163.json` order, each as a one-row call (batch 1, the shape
   the tool checks), with the same pushes, into `results/part200/batch1.jsonl` (not `answers.jsonl`, so Part 163's Gate 3
   row count is unaffected). It resumes by skipping units already written. `--compare` counts, per arm and in total, the
   answers whose text differs from the same unit's chunked battery answer.

Everything else is Part 163's: the model and revision, the 200 units in `part163/units163.json` (100 neurons, 100 Part
86 random directions), the 16 battery prompts drawn with seed 163, "Who are you?" for the recipe check, dose +0.8 at the
row-isolated hook, greedy, 80 new tokens, and `part163/score163.py` unchanged. After the run, on the host:

```text
python -I -B part200/run200.py --check-record results/part200/load.json
python -I -B part163/score163.py --answers results/part200/answers.jsonl --units part163/units163.json --out results/part200
python -I -B part200/run200.py --compare results/part200 results/part163
```

`--compare` writes `results/part200/compare.json` and prints, for the published and rerun answers side by side:

- Scored values: Gate R2; Part 163's three gates; the five verdicts with their estimates and intervals; each arm's
  broken count over its 1,600 battery answers (Part 86 triage, `part86/triage86.py`); each arm's count of battery answers
  that reach the 80-token cap (`token_count` 80); the Part 161-rule Qwen count on maker phrasings (case-sensitive
  `\bQwen\b`, with broken answers and answers matching `alibaba|tongyi|通义` case-insensitively excluded, as in
  `results/part163/extra163.py`); and the batch-1 comparison's differing count.
- Exploratory values: Part 191's mid-phrase count among answers under the cap (the rule `mid_phrase` in
  `part199/report199.py`; published baseline 510 of 1,476 neuron and 409 of 1,382 random under-cap answers), the number
  of battery answers and of the 16 no-push answers whose text is identical between the runs, and per-layer broken counts.

Run on Part 163's own folder it must reproduce 19 of 539 and 57 of 661, 343 and 83 broken, and 124 and 218 at the cap.
It exits 3 when the rerun's `load.json` fails Gate R1 or R3, when Part 163's gates did not pass on the rerun, or when
Gate R2 fails, and 2 when a file is missing.

## Controls

- Part 163's three gates, unchanged (recipe changed share at least 0.70 per arm; no-push names Qwen on at least 7 of 8
  identity prompts and Alibaba or Tongyi on 7 of 8 maker prompts; all 3,416 rows present). A failed gate stops the
  readings it covers, as in Part 163.
- **Gate R1, kernel:** the loader refuses a non-eager load before generation; every `load.json` entry eager; one model,
  revision and output folder across entries.
- **Gate R2, same inputs:** `part163/units163.json` has sha256
  `2ed3bef4488eb6b3337dfde100c63c7c5a5df7e8ca809681dec34824c21dd710` (the published file at public main 8ccf6e3), and the
  16 battery questions and their order match Part 163's answers file. The pushes are built from these by unchanged code,
  so a mismatch means the inputs moved, and readings stop.
- **Gate R3, shapes:** read from the row counts actually passed to Part 163's generator as each chunk returns: every
  chunk of a pushed battery or recipe call has exactly 8 rows, a completed call's chunks cover its rows (a call cut short
  by a killed launch may cover fewer), and the no-push and batch-1 calls are single rows.
- **Shape check:** the batch-1 comparison (Method, item 4). If more than 20 of its 200 answers differ from the chunked
  answers, the arm counts are reported as shape-sensitive beside every reading of them. This does not change a verdict.
- Not used: new units, prompts or doses (this part reruns a registered run), and a CPU float32 run of the battery (far
  longer than the A770 run; the eager kernel is checked against the CPU reference instead).

## Size

- Smallest effect worth knowing: any of Part 163's five verdicts or three gates changes; an arm's broken count moves by
  more than 80 of 1,600 (5 points); an arm's Part 161-rule Qwen rate moves by more than 3 points.
- What can change: 163.1 and 163.2 cannot fail unless the random arm's name-gone or maker-gone rate falls below 2/3
  (published 0.801 and 0.893; the neuron rate is at most 1, so the ratio cannot reach 1.5). 163.4 rests on one unit:
  91 neurons reach name gone on at least half of identity prompts against 90 both gone (random 86 against 83), so one
  neuron unit moving flips it. 163.3's bar (0.15) is far below the published 0.85 and 0.80. 163.5 and the counts are the
  live readings.
- Total: 3,616 answers, 80 new tokens: 16 no-push, 200 recipe, 3,200 battery, 200 batch-1. Calls: 16 no-push (one row),
  recipe and battery calls of 64, 64, 64 and 8 rows split into 25 chunks per stage of 200 rows (425 chunks over the
  recipe and the 16 battery prompts), and 200 batch-1 calls: 641 generate calls.
- Time, not measured for this part: Part 199 made 880 generate calls (200 no-push, 680 chunks) in about 152 minutes,
  about 10.4 seconds a call. At that rate the 441 chunk and no-push calls take about 76 minutes; the 200 one-row calls
  are faster per call and add perhaps 15 to 20. So about 90 minutes. Over an hour, so the pilot applies; its measured
  rate is recorded and replaces this estimate.
- Pilot: the first 3 recipe batches (24 chunks of 8), with Part 163's Gate 1 check and the rate guard above. **A pilot
  stop is a HOLD**: no reading is made, the measured rate and the cause go into a dated amendment, and the run is
  relaunched only after it.

## Optimization

- Bound resource: GPU (the A770), and Windows memory for the load (start only with at least 11 GB available).
- Measured: Part 199's 6,672 answers in about 152 minutes at chunks of 8 on eager; Part 163's 3,416 answers at batch 64
  on SDPA at 2.83 answers per second in its pilot.
- Levers: chunks of 8 rather than 64 (taken; costs time, buys a shape checked by hand, plus the batch-1 comparison); a
  batch-64 eager kernel check that could keep 64 rows (declined: a new check is a part of its own); fewer prompts or
  units (declined, the published claims need every row); copy engine off, the job default (taken).
- Host work: scoring and the comparison need no model.

## Predictions

Values are read from `--compare` and `score163.py`'s output. Ranges are inclusive. Verdicts are read on point
estimates, as `score163.py` decides them. 200.3, 200.6's direction and 200.7 compare arms or shapes within the rerun;
200.4 and 200.5 compare the rerun with the published run, which differ in kernel, batch shape and answer length at once.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 200.1 | Gates R1 to R3 and Part 163's three gates pass. | medium | Claude |
| 200.2 | 163.1, 163.2, 163.3 and 163.4 hold and 163.5 fails, as published. | medium: 163.1 and 163.2 cannot fail unless a random rate falls below 2/3, and 163.3's bar is far below the published rates, so the live parts are 163.4 (one unit from flipping) and 163.5. | Claude |
| 200.3 | The neuron arm's broken count over its 1,600 battery answers is at least twice the random arm's (published 343 and 83). | medium: Part 199 kept its between-arm direction on eager while its rates moved. | Claude |
| 200.4 | In each arm at least 480 of 1,600 battery answers (30%) reach the 80-token cap (published 124 neuron, 218 random). | medium: on SDPA almost every Part 199 answer stopped early; on eager most ran to the cap. | Claude |
| 200.5 | Each arm's broken count is higher on eager than published. | low: Part 199's loop rates rose on eager, but Part 163's triage keeps any answer naming a model or identity, and a kernel can remove loops as well as add them. | Claude |
| 200.6 | Under Part 161's rule on maker phrasings, the neuron arm's pooled Qwen rate stays below the random arm's (published 19 of 539 against 57 of 661). | low: the published rates are small and the comparison was unregistered. | Claude |
| 200.7 | At most 20 of the 200 batch-1 answers differ from the same units' chunked answers. | low: rows are pushed independently, but eager batch 8 left the batch-1 path at a near-tie on 1 of 8 kernel-check prompts. | Claude |

## Reading rules

- Read gates first. A failed Gate R1, R2 or R3, or any of Part 163's gates, stops every reading below it and is
  reported.
- The scored items are Part 163's five verdicts and three gates, Gates R1 to R3, and 200.1 to 200.7. Everything else
  `--compare` or any later script prints is exploratory and is labelled so wherever it appears, including subsets chosen
  after reading (a layer, units that never break, first sentences only) and any length-adjusted rate. No exploratory
  value replaces a scored one.
- Apply Part 163's reading rules and bars as written, through `score163.py` unchanged, including its Amendment 1 (the
  163.1 reading rule is void). Each verdict and each scored count is reported beside Part 163's published value; a
  change is reported as a change, with both values and, for verdicts, both intervals. Intervals do not change a verdict.
- A change between the runs is attributed to the kernel and the batch shape together, never to either alone, and the
  batch-1 comparison's count is reported beside it.
- Lengths are reported before any rate is compared across the two runs: a looping answer ran to the cap on eager and
  stopped early on SDPA in Part 199, so a cross-run difference may partly be a length difference.
- The no-push answers are one-row decodes; a change in their text is not evidence about the 64-key fault (Part 199's
  plain answers changed below 64 keys).
- The headline update to Part 163's README names which published numbers held on eager and which did not, in the same
  sentence as the claim.

## Limits

- One model revision, greedy decoding, Part 163's units, prompts and dose. This part checks Part 163's measurements on a
  corrected kernel; it does not extend them, and it does not repair Part 163's design (163.1 and 163.2 still cannot fail
  at these base rates).
- The batch shape changes from 64 to 8 together with the kernel. The batch-1 comparison covers one identity prompt
  ("What is your name?"); maker phrasings at batch 1 are not checked.
- Eager bf16 is not the float32 reference: Part 178 found it leaves the reference at near-ties, and the kernel-check
  record's own self-check flag is false on 4 of its 24 runs.
- The neuron arm is one draw (seed 163) from a large pool; this rerun keeps that draw.
- `part163/run163.py` gets one edit (the pilot threshold as a module constant); its behaviour for Part 163's own entry
  point is unchanged and pinned by a test.

## Changes before registration

- Claude Haiku 5.5 design review, round 1 (`part200/review/REVIEW-r1.md`, REGISTER-WITH-FIXES, 12 findings, every quoted
  number recomputed), folded: Gate R2's hash written in (1); 163.1 and 163.2's 2/3 condition and 163.4's one-unit margin
  stated in Size and in 200.2 (2, 3); the kernel-check record's self-check flag and the hand-only batch-8 check disclosed,
  and the reversed-order repeat replaced by a batch-1 comparison of the same 200 rows (4); the guard set at 0.17 answers
  per second so a run too slow for the 6-hour limit stops as a HOLD, with its count stated (5); the time estimate redone
  from Part 199's 880 calls (6); the opening lists all four changes, both pilot stops exit 3, and Gate R3 requires chunks
  of exactly 8 (7); the batch-1 stage named, ordered, resumable and counted per arm, with a shape-sensitivity threshold
  that does not change verdicts (8); the scored list, exploratory labels, attribution to kernel and shape together, and
  no subset re-selection added to the reading rules (9); Part 199's eager onset finding, its one-row plain-answer
  changes and the within-run versus across-run split stated (10); the cap term made consistent and the loop share cited
  (12).
- Claude Haiku 5.5 code reviews of `part200/run200.py`, rounds 1 and 2 (`part200/review/CODE-REVIEW-r1.md`,
  `CODE-REVIEW-r2.md`), folded before registration (`part200/BUILD-REPORT.md`). Round 1: the missing default loader,
  Gate R2 comparing a file with itself, an incomplete batch-1 file passing `--compare`, and the startup repair to whole
  calls. Round 2: Gate R3 now reads the batch `model.generate` receives as keyword arguments, `--compare` runs on the
  published folder's real layout (no `units163.json` there; the design's hash decides Gate R2), and the resume
  diagnostic counts 200 rows per question. The startup repair held in 1,135 kill and truncation trials.
