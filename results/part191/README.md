# Part 191: Part 153's result holds on the eager attention kernel (the ten company neurons still name their company on most maker phrasings; L24 N5017 names Google on 17 of 24)

Part 153 asked whether the ten neurons Part 148 selected from one answer each (Qwen3.5-4B, dose 0.8, one greedy answer
per phrasing) name their company reliably on 24 new maker phrasings, 8 identity phrasings, 2 selection questions and 6
guards, against paired random directions. It ran on the A770 in bf16 with transformers' default fused SDPA kernel, which
Part 178 found leaving the CPU float32 path after about 64 keys. Here the same battery, units, medians, dose and scorer
ran on the eager kernel (load record: `attn_implementation` eager, requested eager; `run191.py --check-record` passed),
with each question's 20 pushed rows generated in chunks of 8, 8, 2 and 2 where Part 153 used one batch of 20. Registered
in `PART191-DESIGN.md` (public main 340b0d3) before any eager answer. The run took 66.6 minutes (23:22:04 to 00:28:42 CT,
2026-11-07 to 08), inside the registered 75.

| Part 153's measures | Eager (this part) | SDPA (Part 153, published) |
| --- | ---: | ---: |
| Neurons hitting their company on at least 12 of 24 maker phrasings | 7 of 10 | 7 of 10 |
| Neurons' mean maker hit rate | 0.596 | 0.629 |
| Paired random directions' mean maker hit rate | 0.113 | 0.113 |
| Ratio, neurons to random | 5.30 | 5.59 |
| Neurons hitting their company on at least 4 of 8 identity phrasings | 1 of 10 | 2 of 10 |
| Neurons keeping at least 5 of 6 guards | 9 of 10 | 9 of 10 |
| L24 N5017: Google on maker / identity phrasings | 17 of 24 / 7 of 8 | 20 of 24 / 7 of 8 |

Part 153's claim stands on eager. Its scorer gives the same four verdicts on both kernels (153.1 held, 153.2 failed, 153.3
held, 153.4 held), and every registered prediction held.

## Gates

- Gate 1 (rows): 840 unique, nonempty rows, no duplicates. Passed.
- Gate 2 (plain answers name no company): 0 hits in the plain arm, limit 2. Passed.
- Gate 3 (selection): all 10 neurons named their company on the selection questions at this batch shape. Passed.
- Load record: one invocation, eager requested and eager set, chunks 8, 8, 2, 2 (`load.json`). Passed.

## Predictions

| # | Prediction (registered wording; confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 191.1 | 153.1's verdict stands: at least 5 of the 10 neurons hit their company on at least 12 of the 24 maker phrasings (high) | 7 of 10 | held |
| 191.2 | The neurons' mean maker hit rate is between 0.45 and 0.80 (published 0.629) (high) | 0.596 | held |
| 191.3 | 153.3's verdict stands: the neurons' mean maker hit rate is at least twice their paired random directions' (high) | 0.596 against 0.113, 5.30 times | held |
| 191.4 | The random directions' mean maker hit rate is at most 0.20 (published 0.113) (high) | 0.113 | held |
| 191.5 | L24 N5017 names Google on at least 16 of 24 maker phrasings and at least 5 of 8 identity phrasings (published 20 and 7) (medium) | 17 and 7 | held |
| 191.6 | All four published verdicts are the same (153.1 held, 153.2 failed, 153.3 held, 153.4 held) (medium) | same four | held |

## Per neuron, maker phrasings (of 24)

| Neuron | Company | Eager | SDPA (Part 153) |
| --- | --- | ---: | ---: |
| L11 N5647 | Google | 16 | 21 |
| L21 N7134 | Baidu | 12 | 15 |
| L22 N6997 | Google | 11 | 9 |
| L23 N414 | Google | 17 | 21 |
| L23 N1743 | Huawei | 17 | 14 |
| L23 N2993 | Microsoft | 18 | 17 |
| L24 N5017 | Google | 17 | 20 |
| L25 N140 | Google | 11 | 10 |
| L25 N4457 | Google | 4 | 5 |
| L25 N4458 | Google | 20 | 19 |

The same seven neurons reach 12 of 24 on both kernels; individual counts move by up to 5 in either direction.

## Descriptive, not registered: the SDPA answers were cut short

The design said greedy text was not expected to match across kernels. It does not: 122 of the 840 answers are
identical in text (65 of 400 neuron rows, 47 of 400 random rows, 10 of 40 plain rows). What changed most is length. On
eager the median answer is 82 tokens and 252 answers reach the 120-token cap; on SDPA the median was 60 tokens and 2
reached the cap, with the most common lengths between 55 and 70. Of the answers under the cap, 245 of 838 SDPA answers
(29%) end without closing punctuation, against 58 of 588 eager answers (10%), and 15 SDPA answers contain the string
"QwQ" against none on eager. This fits Part 178's finding that the SDPA kernel leaves the reference path past about 64
keys: Part 153's answers mostly stopped around there. The counts held anyway because the company is named early: of the
neuron arm's maker hits, 138 of 143 name it in the first sentence on eager, 147 of 151 on SDPA. The scorer reads the whole
answer, and an answer that names the company is still not a hit if it also mentions Alibaba or Tongyi or is triaged
broken. On the neuron maker rows the company is named in 164 of 240 answers on eager and 165 on SDPA; eager loses 21 of
those as broken against 13 on SDPA (Alibaba or Tongyi: 0 against 1), which is where most of the drop from 0.629 to 0.596
comes from. Answers are compared row by row (unit, arm, question) with `results/part153/answers.jsonl`.

## Limits

- One model, one revision, one dose and the same 40 questions: this tests the kernel, not how general the neurons are.
- Batch shape changed (chunks 8, 8, 2, 2 against 20), so a difference could come from batch shape as well as the kernel.
  Part 183 checked eager against the CPU float32 reference at batches 1 and 2, and by hand at batch 8.
- One greedy answer per phrasing. A count moving by a few phrasings is within what one answer each can show.
- Answer length differs by kernel, and length moves the scorer both ways: more text to find the company, and more chance
  of being triaged broken. The descriptive counts above separate the two on the neuron maker rows.

## Files

- `answers.jsonl`: 840 eager answers. `load.json`: the load record. `launch.jsonl`: the launch record.
- `score.json`, `tables.md`: Part 153's unchanged scorer (`part153/score153.py`) on this part's answers.
