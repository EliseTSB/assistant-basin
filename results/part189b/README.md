# Part 189b: Part 148 on the eager attention kernel keeps five of its six verdicts; one flips (148.1), and every registered rate moves up by 4 to 11 points

Part 148 asked how many of 465 selected neurons (and 465 random directions) move Qwen's answers about its name and maker
when pushed. It ran on the A770 with transformers' default fused SDPA kernel, which Part 178 found leaving the CPU
float32 path after about 64 keys. Part 189 tried to rerun it on the eager kernel and stopped at its own Gate 2
(`results/part189/README.md`). Part 189b (registered 7a270f1) moved the wiring check to an SDPA batch-64 arm
and ran the eager full run: 2,796 answers, same units, seeds and dose, labelled again by Part 148's reader
(DeepSeek V4.1 Flash, same brief) and scored by `part148/score148.py`'s rules via `part189b/score189b.py`.

## Gates

All passed: eager load (requested and recorded `eager`); wiring arm on the old kernel reproduces Part 112's stored
answers 32 of 32 (Gate 2, 28 required); zero push matches the plain model (Gate 1); the plain model is not moved
(Gate 3). The eager pilot matches Part 112's stored answers in the first 40 tokens on only 17 of 32 (5 of 32 in full);
that is reported only, as registered. Over the main run's 2,790 pushed rows the texts differ far more (the cold checker's word-level count:
190 match Part 112's answer for 40 words, 505 are identical in full).

## Predictions

| # | Registered | Result | Verdict |
| --- | --- | --- | --- |
| 189b.1 | at least 5 of Part 148's 6 verdicts the same | 5 of 6 the same | held |
| 189b.2 | no verdict flips | 148.1 flips (failed to held) | failed |
| 189b.3 | name 74.8%, maker 41.1%, both 31.4%, maker-among-name 42.0% each within 6 points of published | name 82.8% (+8.0), maker 45.8% (+4.7), both 40.6% (+9.2), maker among name-movers 49.1% (+7.1) | failed |
| 189b.4 | random-direction name rate within 6 points of 37.6% | 49.0% (+11.4) | failed |
| 189b.5 | slot maker rate within 6 of 43.4%, concept within 6 of 38.9% | slot 48.1% (+4.7), concept 43.6% (+4.7) | held |

## Part 148's six verdicts, side by side

| # | Part 148 (SDPA) | Part 189b (eager) |
| --- | --- | --- |
| 148.1 name moved in at least 80% | failed (348/465, 74.8%) | **held** (385/465, 82.8%) |
| 148.2 | failed | failed (maker 213/465, 45.8%) |
| 148.3 | held | held (189 neurons move name and maker) |
| 148.4 | failed | failed (189/385, 49.1%) |
| 148.5 | failed | failed (neurons 82.8%, random 49.0%) |
| 148.6 | failed | failed (slot 91/189, concept 112/257) |

## What the numbers say

1. The qualitative picture stands: only 148.1 changes, and it changes because 74.8% (348 units) moves to 82.8% (385),
   over the registered 80% line (372 units), not because the shape moved. Neurons still beat random directions on name by 33.8 points (37.2 before).
2. The eager kernel raises how often an answer is judged name-moved in both neurons and random directions (+8.0 and
   +11.4), so the random baseline in 148.5 rose more than the neurons did in points (+53 against +37 of 465 units). The neuron-over-random
   gap narrowed only from 37.2 to 33.8 points. Whether the kernel changes the answers
   themselves, or the reader's call on them, was not separated here: the answers differ from the stored ones (in the
   pilot, 17 of 32 match in 40 tokens), and the reader labelled a different text.
3. 189b.1 held and 189b.2 failed on the same fact: one flip is allowed by the first and forbidden by the second.
4. The exploratory section of `score.json` (named company counts, same-company pairs) is unread beyond its totals; not
   claimed here.

## Caveats

- One run per arm; the reader is one model (Flash), checked by no second reader here. `labels.jsonl` has 1,398 rows for the
  890 moved rows, every moved row labelled without error.
- Part 148's published README still carries the SDPA numbers; this part does not change it.

Files: `answers.jsonl` (2,796), `labels.jsonl`, `score.json`, `tables.md`, `gates.json`, `load.json`, `launch.jsonl`,
`wiring/` (SDPA batch-64 arm). Design `PART189B-DESIGN.md`.

Cold check: DeepSeek V4.1 Flash worker (no project context), 2026-11-04 20:12 CT, 9 findings: every number and verdict recomputed and matched; six wording points folded (title scope, which stored answers, points against units, pilot against main-run text change, size of the 148.1 crossing, label count). Report in `results/part189b/cold/COLD.md`.
