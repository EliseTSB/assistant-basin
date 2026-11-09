# Part 199: Part 170 on the eager attention kernel

Procedure: v1
Registered: DRAFT, not yet registered (the owner stamps the time with `date` at registration)

Part 178 found the Intel Arc A770's fused bf16 SDPA attention kernel leaves the CPU float32 path once the cache holds
more than 64 keys. Part 170 generated 80-token greedy answers on that kernel, and its outcome is itself decode
pathology (loop answers), so a faulty kernel could create the effect rather than blur it
(`notes/reports/arc-sdpa-audit-2026-11-03/PRIORITY1-CLAIMS.md`, kept in the Journal repository). This card reruns
Part 170 unchanged except for the kernel and the push batch shape.

## Question

Part 170's published claim: under a full-size neuron push, blocking the plain answer's 50 likeliest tokens raised
identity loops from 25.0% to 43.8%; without a push the effect depends on list size and prompt set (170.1 held, 170.2
failed). Do its gates, cell rates and verdicts hold when the same model and inputs run with eager attention?

## Prior art and existing data

- Part 178: SDPA leaves the CPU float32 path past 64 keys. Part 183's kernel check (`results/part183/kernel-check.json`)
  passes eager at batches 1 and 2 against `results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json`; batch 8 was
  compared by hand; its SDPA control fails, so the check can see the fault.
- Parts 183 to 195 reran thirteen high-risk parts on eager. Most published claims held; Part 39's 8 of 96 AI claims
  became 0 of 96 on eager (Part 192, one persona; consistent with the kernel), and Part 191 found SDPA answers cut short: 230 of 838 ended
  mid-phrase against 2 of 588 on eager.
- **Loop onset on Part 170's published answers** (`part199/onset/`, exploratory, not pre-registered, run before this
  design and cold-checked by Claude Haiku 5.5): for each looping answer, the first token at which the loop rule is
  already met, against t0 = 65 − prompt length, the first answer token whose step attends over more than 64 keys.
  1,285 of 1,484 loops were established before t0. The heavy-ban push cells are almost all early (dose 1 b50: 164 of
  168), while the light cells are exposed: dose-1 b0 14 of 96 late, dose 0.25 b0 10 of 10, every no-push b0 and rel50
  loop, and all 19 Stage 0 plain loops that froze 12 M96 and 7 Q96 prompts. If the kernel only added loops, the dose-1
  b50 rise over b0 lies between 17.7 and 22.4 points (published 18.8). That range is conditional, not a bound: the check
  cannot see loops the kernel prevented, and Part 191's shorter SDPA answers point at exactly that direction.
- Part 170's own runs: Gate 8 found 0 of 384 dose-1 b0 answers changed when the batch order was shuffled; Gate 6 dose-1
  b0 loop rate 0.25 and b50 mean blocked steps 16.0; Stage 0 flagged 12 of 96 M96 and 7 of 96 Q96. Above 14 of 96 in a set, Part 170 stops
  that set's readings (`set_passed`, not a gate): M96 above 14 leaves 170.1 and 170.W1 unscored.
- The published readout is `results/part170/README.md` and the front-page entry (README line 2722); its scorer is
  `part170/score170.py` and its runner `part170/run170.py`, launched with `--batch-size 64`.

Kernel check: results/part183/kernel-check.json

## Method

Run `part170/run170.py`'s `run()` unchanged through `part199/run199.py` on the A770 (`xpu`), in one output folder
`results/part199`. The wrapper changes three things and records them:

1. **Kernel.** The wrapper hands `run170.run` a loader that loads through `neuron_report.load_model` (eager by
   default), applies Part 170's no-grad and eval settings, and refuses with exit 3 before any generation unless
   `neuron_report.LAST_LOAD` records eager. The loader checks both `LAST_LOAD` and the model's
   `config._attn_implementation`. Each launch appends one entry to `results/part199/load.json` as soon as the model
   has loaded (`LAST_LOAD`: effective kernel, dtype, device and library versions) and rewrites it after every chunk
   with the calls and the rows written so far, so a launch killed by a signal keeps its kernel record. The wrapper
   exits 3 if any recorded kernel is not eager or if Part 170's own gates did not pass. An invocation holds
   `run199.lock` in the output folder (created exclusively), so two launches cannot share it. `run199.py
   --check-record results/part199/load.json` repeats Gates R1 and R3 over every recorded launch on the host.
2. **Push batch shape.** Part 170 generated each push batch (one prompt, one list, one dose) as 48 rows, a shape no
   eager check covers. The wrapper replaces `run170.generate_with_bans` with a version that, for pushed calls, generates
   the same 48 rows in six chunks of 8, in the original row order, passing each chunk's own bans and release steps and
   the call's directions and seed unchanged. Rows map to directions by layer and slot, so chunking changes no row's
   push. The Gate 8 repeat batches (48 rows in shuffled order) are chunked the same way. No-push batches (one prompt,
   5 or 6 lists) and Stage 0 (one prompt at a time, including its 8 one-row zero-write checks, which are pushed calls)
   keep Part 170's shapes, which are 8 rows or fewer. Every generate call and its chunks are recorded in `load.json`
   (`calls`).
3. **Pilot stop rule.** Part 170's pilot stops under 1.5 answers per second, a rate chunks of 8 on eager may not
   reach. The wrapper replaces `run168.pilot_rate_should_stop` with: stop if, after the first 3 push batches, the
   measured rate projects the push stage (5,376 answers, push and repeat) past 8 hours, below 0.187 answers per second.
   That is a guard against a broken path, not a time budget. The A770 job runner kills a task at 6 hours, and
   `run170` resumes at the next unwritten answer, so a killed task is relaunched on the same folder: its rows and its
   `load.json` entry stay, and the operator removes the `run199.lock` it left after checking that no run199 process
   runs (removed by hand on purpose: the job host is Windows, where probing a pid with `os.kill(pid, 0)` terminates it).

Everything else is Part 170's: the model and revision, the 200 prompts (P8, M96, Q96), Stage 0 freezing, Stage 1 lists
from the plain 40-token path, the 48 n+ neurons drawn from `results/part112/medians.json` with Part 170's seed base and
freshness rules, the three doses and their write norms, the ban lists, the Gate 8 repeat, 80 new tokens, greedy, and
`score170.py` unchanged. After the run, on the host:

```text
python -I -B part199/run199.py --check-record results/part199/load.json
python -I -B part170/score170.py --answers results/part199/answers.jsonl --stage0 results/part199/stage0.json \
    --lists results/part199/lists.jsonl --norms results/part199/norms.json --out results/part199
python -I -B part199/run199.py --compare results/part199 results/part170 --tokenizer <Qwen3.5-4B snapshot dir>
```

`--compare` reads Part 170's and the rerun's `score.json` (so it runs after scoring) and prints, beside each other, the
values the predictions below are read from: Gate R2, the Stage 1 lists, the frozen prompts, the gate values, every
cell's loop and broken rate, every verdict, and the loop-onset split of the rerun's own loops
(`part199/onset/loop_onset.py`, run unchanged as a subprocess on `results/part199/answers.jsonl`). It exits 3 when
the rerun's Part 170 gates did not pass or Gate R2 fails, and 2 when a file is missing. Run on Part 170's folder against itself, it reproduces every published
value (Gate R2 passes, 200 of 200 lists identical, the onset split 164 of 168).

## Controls

- Part 170's eight gates, unchanged. A failed gate stops all readings, as in Part 170.
- **Gate R1, kernel:** the loader refuses a non-eager load before generation; every `load.json` entry eager; one
  model, revision and output folder across entries.
- **Gate R2, same inputs:** the 144 drawn directions (layer, slot, neuron) equal Part 170's `norms.json` records, and
  each realized write is within Gate 4's tolerance. The draw does not use generation, so a mismatch means the code or
  inputs moved, and readings stop.
- **Gate R3, shapes:** read from the row counts actually passed to Part 170's generator as each chunk returns, not
  from the wrapper's plan: every pushed generate call (the 48-row push and repeat batches) reached the model as chunks
  of exactly 8 that cover its rows, and no chunk of any call had more than 8 rows. Stage 0's zero-write checks (one
  row, a directions mapping with a single entry) are exempt, at most 8 of them, each one chunk of one row.
- **Batch noise at the new shape:** Part 170's Gate 8 passes up to 10% changed answers (38 of 384), wider than this
  part's smallest effect. If the repeat changes more than 19 of 384, the dose-1 cell rates and rises are reported as
  noise-limited beside every reading of them; `--compare` prints that flag.
- The do-nothing hook (Part 170's Gate 3) and Gate 8's repeat order are kept, so the rerun carries its own batch-noise
  floor at the new shape.
- Not used: new prompts, doses, neurons or lists, because this part reruns a registered run and changes only the kernel
  and the push batch shape. Not used: a CPU float32 run of the whole battery, which would take far longer than the A770 run (not
  measured); the eager kernel is checked against the CPU reference instead.

## Size

- Smallest effect worth knowing: any verdict of 170.1 to 170.8 or the gates changes; a dose-1 cell's loop rate moves
  by more than 0.05 (19 answers in 384); the dose-1 b50 rise over b0 moves by more than 0.05; Stage 0 freezes a
  different set of more than 3 prompts in M96 or Q96.
- Total: 6,672 answers as published (200 Stage 0, 1,096 no-push, 4,992 push, 384 repeat), 80 new tokens, in 1,080
  generate calls (200 Stage 0 plain, 8 zero-write checks, 200 no-push, 624 push chunks, 48 repeat chunks), plus 200
  list passes.
- Time, not measured for this shape: Part 191 made 200 generate calls of 1, 2 and 8 rows (chunks 8, 8, 2 and 2 plus
  one plain row) in 66.6 minutes at 120 tokens, about 20 seconds a call. At 80 tokens that is about 13 seconds a call if
  a call's time does not depend on its rows, about 3.9 hours in all; if it grows with rows, an 8-row call takes about
  25 seconds and the total about 7.4 hours, beyond one 6-hour task, so the run would continue in a second launch. Part
  170's own pilot on SDPA (48 rows within its 1.5 answers per second) points at the first. Over an hour either way, so
  the pilot applies; its measured rate is recorded and replaces this estimate.
- Pilot: the first 3 push batches (18 chunks) and the stop rule above. Stage 0 and the no-push stage run first, as in
  Part 170, so a Gate 2 failure costs under an hour (Stage 0 is 208 calls of one row). **A pilot stop is a HOLD**: no
  reading is made, the measured rate and the cause go into a dated amendment, and the run is relaunched only after it.

## Optimization

- Bound resource: GPU (the A770), and Windows memory for the load (start only with at least 11 GB available).
- Measured: Part 191's 840 answers in 66.6 minutes in calls of 8, 8, 2 and 2 rows plus one plain row (about 12.6
  answers a minute at 120 tokens); Part
  170's 6,672 answers at batch 48 on SDPA in under 102 minutes (launch at 08:35:41 to its results commit at 10:17).
- Levers: chunks of 8 rather than 48 (taken; costs about 2 hours against Part 170's time, buys a checked shape); a batch-48 eager kernel check
  that could let the push stage keep 48 rows (declined: `tools/kernel_reference.py` fills a batch by repeating prompts
  only up to twice its eight prompts, and a new check is a part of its own); copy engine off, the job default (taken);
  fewer rows (declined, the published claims need every cell).
- Host work: scoring and the comparison need no model.

## Predictions

Values are read from `--compare` and `score170.py`'s output. Ranges are inclusive. Verdicts are read on point
estimates, as `score170.py` decides them; several published intervals cross their bars (170.1, 170.3, 170.7), so a
changed verdict near a bar is reported with both intervals. Weaver's 170.W1 and 170.W2 are scored by `score170.py`
and reported beside the published values; they are Weaver's, so this part makes no prediction about them.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 199.1 | Gates R1 to R3 and all eight of Part 170's gates pass. | medium | Claude |
| 199.2 | At least 190 of 200 Stage 1 lists equal Part 170's (prompts are 24 tokens or fewer, so the 40-token plain path never reaches 65 keys). | medium: eager and SDPA can still differ at near-ties below 64 keys. | Claude |
| 199.3 | Stage 0 freezes 6 to 14 M96 prompts (published 12), so neither set's readings stop, and 3 to 12 Q96 prompts (published 7). | low: every published Stage 0 loop formed past key 64, and 15 or more M96 stops M96's readings. | Claude |
| 199.4 | Dose-1 loop rates: b0 0.18 to 0.30 (published 0.250), b50 0.39 to 0.48 (0.438), b200 0.45 to 0.55 (0.503). | medium: the ranges lean on the onset interval, which assumes the kernel only added loops. | Claude |
| 199.5 | Dose-1 b50 rise over b0 is 0.12 to 0.25 (published 0.188). | medium: same condition. | Claude |
| 199.6 | 170.3 holds, 170.4 fails, and 170.6 holds, as published. | medium | Claude |
| 199.7 | 170.1 holds and 170.2 fails, as published (170.1 not scored if M96's readings stop; then 199.7 is not scored). | low: their no-push baselines are entirely late-formed loops. | Claude |
| 199.8 | 170.5, 170.7 and 170.8 keep their published verdicts (failed, held, failed). | low: 170.5 failed by 0.3 points. | Claude |
| 199.9 | Gate 8's repeat changes at most 19 of 384 answers (5%) at the chunked shape. | medium: published 0 of 384 at 48 rows. | Claude |

## Reading rules

- Read gates first. A failed Gate R1, R2 or R3, or any of Part 170's gates, stops every reading below and is reported.
- Apply Part 170's reading rules and bars as written, through `score170.py` unchanged. Each verdict is reported beside
  Part 170's published verdict; a changed verdict is reported as changed, with both values and intervals.
- Score 199.2 to 199.5 and 199.9 against the values `--compare` prints; score 199.6 to 199.8 against the verdict
  lines; 199.1 against the gate lines (Part 170's `gate1` to `gate8` and R1 to R3; a set's reading stop is not a gate).
- If Stage 0 flags more than 14 of 96 in M96 or Q96, that set's readings stop as in Part 170: 170.1 and 170.W1 (M96)
  or 170.2 (Q96) are reported as not scored, never as failed, and the stop itself is reported.
- The rerun's own loop-onset split (`--compare --tokenizer`) is descriptive: on eager text there is no 64-key boundary,
  so it says where loops form in a correct run, not what the kernel did.
- If Stage 0 freezes a different prompt set, the published and rerun no-push readings compare different populations.
  Report both set sizes and the overlap beside 170.1 and 170.2, and do not call a change in either a kernel effect
  without that note.
- Answer texts that differ between the runs are descriptive: the kernel, the push batch shape and the numerics all
  changed together.
- The headline update to Part 170's README names which published numbers held on eager and which did not, in the same
  sentence as the claim.

## Limits

- One model revision, greedy decoding, the published prompts, neurons and doses. This part tests Part 170's
  measurements on a correct kernel; it does not extend them.
- The push batch shape changes from 48 to 8 together with the kernel, so a small shift cannot be assigned to the
  kernel alone. Batch 8 was checked against the reference by hand in Part 183, not by `tools/part_check.py`.
- The loop-onset analysis behind several predictions was run on the published answers before this design, by the
  same author, and is disclosed above.
- No-push batches of 5 and 6 rows are not checked on eager: Part 183 checks batches 1 and 2 by machine and 8 by
  hand. The 1,096 no-push answers, including 170.1's and 170.2's baselines, run at those shapes, as published.
- `part170/run170.py` itself now loads eager by default (`neuron_report.py`, since 2026-11-03), so running it
  unwrapped on current main no longer reproduces the published SDPA run; Part 170's text does not say which kernel it
  ran. Pinning the kernel in `run170.py` is a follow-up, not part of this card.
- `load.json` is a read-modify-write under the folder lock, so two launches cannot interleave on it, and each write
  goes through a temporary file and `os.replace`, so a kill does not leave it half written.

## Changes before registration

- Haiku 5.5 cold review, round 1 (SHIP-WITH-FIXES, 12 findings), folded: the M96 set stop named as a reading stop with
  its own rule and 199.3's upper bound at 14 (F1); Gate R3 scoped to the 48-row calls with Stage 0's one-row
  zero-write checks exempt (F2); the pilot stop made a HOLD and widened to an 8-hour guard, with both time scalings
  stated (F3); the kernel checked in the loader before any generation and the folder locked (F4); a 5% batch-noise rule
  beside Gate 8 (F5); calls counted as 1,080 (F6); the onset interval called conditional, and the old 199.9 (onset split
  on eager text) dropped as a prediction and kept as a description (F7); no-push shapes 5 and 6 in Limits (F8); Part
  39 worded as consistent with the kernel (F9); point-estimate verdicts and Weaver's predictions stated (F10); the onset
  README's prompt ranges given per part (F11); run170's default kernel in Limits (F12).
- Haiku 5.5 cold review, round 2 (SHIP-WITH-FIXES, measured on the wrapper with Part 170 stood in), folded: the load
  entry is written at load and after every chunk, so a killed launch keeps its kernel record, and the lock's manual
  removal is in the design (1); Gate R3 reads the observed chunk sizes (2); a failed Part 170 gate exits 3 from the
  wrapper and from `--compare` (3); the docstring's untested claim removed, an R1 folder test through
  `--check-record` and a source check of run170's call-time lookups added (4); the loader exits 3 and checks the
  config's kernel too, the lock refusal exits 3, and the wrapper prints its own pilot rule (5); torn last lines are
  not counted as rows (6); folders compared resolved (7); the zero-write exemption keyed to a one-entry directions
  mapping and capped at 8 (8); `--compare` prints the noise flag (9); the config-kernel check covers (10).
