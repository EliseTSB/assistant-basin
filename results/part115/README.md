# Part 115: zeroing the top 200 attribution-ranked neurons together takes "Q" off first place on 11 of 16 phrasings; on 9 of 16 the answer then starts with Tongyi

For Part 114 held-out phrasing 1, "Who am I speaking with?", with the answer started as "I am **" (greedy, 12 tokens):

| Zeroed | Continuation |
| --- | --- |
| nothing | Qwen3.5**, a large language model developed by |
| top 200 | Tongyi**, a large-scale language model developed by Alibaba |
| top 500 | AI Assistant**, a large language model developed by Alibaba Cloud. |

The "nothing" row is Part 114's unzeroed `pair_generations.none[0]` from `results/part114/results114.json`.

Here the top k neurons from Part 114's attribution ranking were zeroed together. On the tested grid, the top 100 took "Q" off first place on 3 of 16 held-out phrasings, the top 200 on 11, and the top 500 on all 16. The threshold is bracketed in (100, 200]. None of the 90 activity-matched random sets took "Q" off first place on any phrasing. At 200, 4 flipped continuations begin "Tongyi" and 5 begin "Tongyi Wanxiang"; 5 still begin "Qwen3.5". At 500, the first words are AI (10), ChatBot (4), Chat (1) and DeepSeek (1). Scope: one model, one forced opening, Part 114's 16 held-out phrasings.

## Predictions

| # | Prediction (verbatim) | Result | Verdict |
| --- | --- | --- | --- |
| 115.1 | Claude (medium confidence): The top 20 zeroed together lower the mean held-out log-odds of "Q" by less than the 8.42 their single drops add up to. | 5.18 | **held** |
| 115.2 | Claude (low confidence): The smallest k whose top-k set takes "Q" off first place on at least 8 of the 16 phrasings is between 10 and 100. | k = 200 on the tested grid (100 gave 3 of 16; 200 gave 11) | **failed** |
| 115.3 | Claude (medium confidence): At every k, the top-k set lowers the mean log-odds of "Q" more than all 10 matched random sets of the same size. | all nine sizes | **held** |
| 115.4 | Claude (medium confidence): At that smallest k, no matched random set takes "Q" off first place on more than 2 of the 16. | 0 of 16 for all 10; maximum 0, withdrawal rule not triggered | **held** |
| 115.5 | Claude (low confidence): At that smallest k, the four ordinary openings' right-token log-probability falls by less than 1 nat on average. | it rose by 4.85; Paris +10.0, cold +8.0, space token +1.4, oxygen 0.0 | **held (literal wording; the check measures the bold token, not the answer)** |
| 115.6 | Claude (low confidence): At that smallest k, the most common first word of the continuations that do not start with "Q" is not a model or company name. | Tongyi, 9 of 11 | **failed** |

## Is it these neurons, or zeroing that many?

Each size was compared with 10 random sets in which every candidate was swapped for a same-layer neuron drawn from the 20 nearest it in mean |activation| (never one of the top 500 candidates). Mean change in the log-odds of "Q" over the 16 phrasings, against the unzeroed run (`part115/score115.py`, `results/part115/score.json`):

| k | Candidate set: "Q" lost first place | Candidate set: change in log-odds | Random sets: maximum flips by any one set | Random sets: change in log-odds |
| --- | --- | --- | --- | --- |
| 1 | 0 of 16 | -0.72 | 0 of 16 | -0.05 to +0.02 |
| 2 | 0 of 16 | -1.20 | 0 of 16 | -0.03 to +0.11 |
| 5 | 0 of 16 | -2.63 | 0 of 16 | -0.11 to +0.12 |
| 10 | 0 of 16 | -3.82 | 0 of 16 | -0.06 to +0.28 |
| 20 | 0 of 16 | -5.18 | 0 of 16 | -0.31 to +0.31 |
| 50 | 0 of 16 | -6.96 | 0 of 16 | -0.44 to +0.82 |
| 100 | 3 of 16 | -8.81 | 0 of 16 | -0.12 to +0.98 |
| 200 | 11 of 16 | -9.83 | 0 of 16 | -0.77 to +1.31 |
| 500 | 16 of 16 | -11.06 | 0 of 16 | +0.50 to +1.89 |

The candidate sets lowered the log-odds more than every random set at every size. The random sets took "Q" off first place on no phrasing. At 500 every random set raised its log-odds.

## Do the effects add up?

Not fully. The top 20 lowered the batched mean log-odds by 5.18 together; their 20 single drops in Part 114 add up to 8.42, so together they gave 62% of the sum.

Part 114's one measured pair came from a separate unbatched run. Its unzeroed baseline was 6.70 and the top pair lowered it by 1.16. The two single drops sum to 1.29 in the batched score, whose unzeroed baseline was 6.74. These figures use different baselines, so they are not a like-for-like additivity comparison.

## What does the model say instead?

First words of the continuations that no longer start with "Q" (`first_words` in `results/part115/score.json`): at 100, Tongyi, Yi and GPT (one each); at 200, Tongyi 9, Yi 1 and LLM 1; at 500, AI 10, ChatBot 4, Chat 1 and DeepSeek 1. The 200 count includes 4 continuations beginning "Tongyi" and 5 beginning "Tongyi Wanxiang". The top 200 outputs name Alibaba in 2 of 11 flipped continuations and Tongyi Lab in 1; 8 stop before naming a maker. At 500, 3 of 16 name Alibaba.

The unzeroed continuation for the example above stops at "developed by" without naming a maker. The 12-token window gives no maker baseline, so it cannot test whether the name changed before the maker.

## The off-target check used targets that were not the plain model's first choices

The main run recorded the baseline log-probabilities. For the first three openings, the target answers were not the plain model's first choices; water's target, " oxygen", was. The four registered openings were "The capital of France is", "2 + 2 =", "The opposite of hot is", and "Water is made of hydrogen and". The plain model put " **" first after the first three openings and " oxygen" first after water. The " 4" target had two tokens, so the check read its first token, the space. At baseline, " Paris" had log-probability -10.1, " cold" -9.8, and the space token -1.5. The baseline was run but not read before registration. The 115.5 verdict follows the literal registered wording, but this check does not establish ordinary answer accuracy.

An exploratory diagnostic run afterwards (`part115/ordinary_top115.py`, `results/part115/ordinary_top115.json`) read the top five next tokens. With the top 100 zeroed, " Paris" moves to first (-0.32) and " **" to second; at 500, " Paris" is -0.02 and " **" is -4.39. " cold" is first only at 500. Water was already " oxygen" at baseline and stays there.

For " cold" at 200, the diagnostic log-probability is -1.651 and the main run's is -1.748. They differ by about 0.10, so the runs are not identical. Exploratory hypothesis: some of Part 115's effect may involve the bolded-name context. The diagnostic had no matched controls, and the water opening moves the other way: the log-probability of " **" is -6.876 at baseline, -5.38 at 100, -5.754 at 200 and -6.379 at 500. An opening without "**" would be needed to separate name effects from formatting context.

The launch record gives the diagnostic override reason as "exploratory diagnostic after scoring: read the ordinary openings' top tokens, which the registered check never read; not a prediction".

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; chat template, no system prompt, thinking off; the answer forced to begin "I am **"; the readout is the log-odds of the single token "Q" (id 48) and whether it is the most likely next token, on Part 114's 16 held-out phrasings. Sets: the top k of Part 114's first-order attribution (`results/part114/attr114.pt`) for k = 1, 2, 5, 10, 20, 50, 100, 200 and 500, zeroed at every position through a per-row mask on each down projection's input. Gates, both passed before any zeroing was read: an all-ones mask left the logits bit-identical (`ones_mask_bit_identical`), and the unzeroed row matched Part 114's unzeroed held-out log-odds within a mean of 0.046 (tolerance 0.1). Run 2026-10-29 19:15 to 19:17 CT (99 s); the design is `PART115-DESIGN.md`. Registration: commit 900e14b, dated 2026-10-29 19:15:32 CT.

## Limits

One forced opening, which itself contains "**". Zeroing is one kind of removal; mean-ablation could differ. The ranking is a first-order attribution on 8 selection phrasings; another ranking could need fewer neurons. There were 10 fixed-seed random sets per size, a small sample. The 12-token window cannot test the maker. The off-target check did not work as designed (above).

## Files

- `results/part115/results115.json`: every row (unzeroed, candidate and random sets), the gates and the continuations; made by `part115/howmany115.py`.
- `results/part115/score.json`: per-size counts and the verdicts; made by `part115/score115.py`.
- `results/part115/ordinary_top115.json`: the exploratory top-five diagnostic; made by `part115/ordinary_top115.py`.
- `results/part115/launch.jsonl`: both launches (`tools/launch_arc.py`), the diagnostic with its recorded override.

Cold check: Haiku 5.5, round 1, 2026-10-29, 17 findings, all folded (results/part115/COLD-CHECK-r1.md, results/part115/FOLD-r1.md).
