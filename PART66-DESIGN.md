# Part 66: do the four maker neurons work on questions they were not found on?

Written 2026-10-25 by Claude, before any code or run, for the public write-up (Elise: publicizing the results this
afternoon). Predictions 66.1 to 66.7 are in PREDICTIONS.md and are pushed before the run.

## Why

Part 25 found four Qwen3.5-4B neurons (L6 N753, L7 N1169, L11 N8868, L29 N776) by a greedy search over 200 candidate
groups, scored on six identity questions, and reported that zeroing them makes "Who created you?" answer Google or
Gemini while 20 matched random sets do not. The front page leads with that result. Two things weaken it:

1. The search and the test used the same six questions, and the set flipped only one of the six. A set found on a
   question can work on that question and nowhere else.
2. Part 25 generated greedily, one answer per question. Part 32 sampled 24 answers on "Who created you?" (English
   Google 16 of 24) but kept that one question.

Part 66 asks the plain question an outside reader will ask: on new phrasings the search never saw, with sampled
answers and many more random sets, does zeroing the four still move the maker to Google, and is it still specific to
those four?

## Model and intervention

- `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (Part 25's), bf16, on the A770.
- Chat template with `enable_thinking=False`, no system prompt (`neuron_report.build_prompt(tok, q, False)`), as in
  Part 25.
- Target: zero the four units (L6 N753, L7 N1169, L11 N8868, L29 N776) at every position, prompt and answer, in the
  MLP down-projection input, Part 25's operation.
- Random controls: 50 four-unit sets, each one random unit in each of the target's layers (6, 7, 11, 29), drawn with
  Part 25's `_matched_random` rule and seeds 6600 to 6649 (Part 25 used 0 to 19). The same operation.

## Questions (fixed here, before any run; none is one of Part 25's six)

**Held-out maker questions, direct (16):** Which company created you? / Who built you? / Who developed you? / Who
trained you? / Which organization made you? / What company are you from? / Who is your creator? / Who designed you? /
Which lab produced you? / Who are the people behind you? / Which company developed this model? / Who programmed you? /
Tell me who made you. / What organization trained this assistant? / Who do you belong to? / Who is responsible for
creating you?

**Held-out maker questions, indirect (8):** If I wanted to thank the people who built you, which company would I write
to? / Which company's product am I talking to right now? / What kind of AI are you, and where do you come from? /
Where were you developed? / Introduce your developer in one sentence. / Which tech company should get credit for you?
/ Who maintains and updates you? / Describe your origins briefly.

**Positive control (the rig must reproduce Part 32):** Who created you?

**Non-maker fluency guards (6), with the keyword an answer must contain:** What is the capital of France? (Paris) /
What is 2 + 2? (4) / What is the largest planet in the solar system? (Jupiter) / Who wrote Romeo and Juliet?
(Shakespeare) / What is the chemical symbol for gold? (Au) / What color is the sky on a clear day? (blue)

## Sampling and counts

Temperature 0.7, top-p 0.95, up to 48 new tokens, seeded per (condition, question, sample).

| Condition | Held-out maker | Positive control | Fluency guards | Generations |
|---|---|---|---|---|
| Unchanged | 24 x 8 | 1 x 24 | 6 x 4 | 240 |
| Target zeroed | 24 x 8 | 1 x 24 | 6 x 4 | 240 |
| 50 random sets | 24 x 1 each | 1 x 4 each | none | 1,400 |
| **Total** | | | | **1,880** |

That is over 1,000 generations, so a pilot runs first (3 held-out questions x 2 samples in unchanged, target and three
random sets, plus the positive control x 4 in unchanged and target) to time it and check the scorer. The pilot's
answers are not part of the result.

## Scoring (fixed here)

- Primary: a maker family is named if the whole answer matches, case-insensitive: google = `Google|Gemini|DeepMind`;
  alibaba = `Alibaba|Qwen|Tongyi`; openai = `OpenAI|ChatGPT|GPT-`; anthropic = `Anthropic|Claude`; deepseek =
  `DeepSeek`; meta = `Meta AI|Llama`; other = none of these. An answer can name more than one.
- The "Google rate" of a condition is the share of its answers that name google.
- A hand read of 30 google-matching answers (or all, if fewer) checks that the match means the answer attributes its
  making to Google, not a mention in passing; the count of passing mentions is reported beside the rate and the rate
  is not adjusted.
- Fluency guard: the share of guard answers containing their keyword, per condition.
- Random-set comparison: each random set's Google rate over its 24 held-out answers; the target's held-out rate is
  compared with that distribution, empirical p = (1 + sets at or above the target) / 51.

## Decision rule (written before data)

- If 66.1 and 66.3 hold, the front page may say the four neurons move the maker answer on new phrasings, with the rate
  and its interval.
- If 66.3 holds and 66.1 fails low (target rate under 5%), the front page must say the effect is narrow: it moves one
  phrasing and barely generalizes, and the lead finding is reworded.
- If 66.6 fails (the positive control does not reproduce), nothing else is read until the rig is checked against
  Part 32.

## Prior art

Part 25 and Part 32 (ours). Outside: ModelMapper and Persona Vectors as cited for Part 25; nothing new searched for a
replication of our own result.
