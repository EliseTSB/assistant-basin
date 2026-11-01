# Part 162b: does Part 162's reading survive once punctuation is taken out?

Procedure: v1
Registered 2026-10-31 19:20 CT by Claude (Opus 5.5), after reading Part 162's first score, so these predictions are not blind. The answers are Part 162's; nothing new is generated.

## Question

Part 162's first score read deepseek-v4.1-flash at P_C 1.08 and P_G 0.25, Qwen3.5-4B at 0.73 and -0.05, OLMo at 0.49 and 0.52. But 99 of the 500 Claude markers across folds hold a straight apostrophe and 70 of the 500 GPT markers a curly quote: gpt-6-luna writes curly quotes and the others do not, so part of the measure is typography. Another 16 GPT markers hold an em dash, which Haiku 5.5 almost never writes and Opus 5 wrote at 15.2 per 1,000 words, so that habit can look like GPT's only because of which Claude we sampled. Does the reading survive without them?

## Prior art and existing data

- Closest published work: Wadhwa et al. (arXiv:2502.06659), as in Part 162.
- Our own data: Part 162's answers and first score, `results/part162/`.

## Method

Part 162's method with two changes, one per arm. Arm A: before cutting, curly quotes become straight in every model's answers. Arm B: arm A, and markers holding an em dash are removed before the top 100 are chosen. Everything else is unchanged: prompts, 100-word cut, masking, five folds, seed 162.

## Controls

- Gates 1 to 3 as in Part 162, read in each arm.
- Gate 4 for Qwen3.5-4B is 80% here. Its 17 short answers in Part 162 ended on their own at 51 to 128 tokens, so more tokens would not lengthen them and 96 of 113 is what this model gives. Part 162's 90% stands for Part 162.
- Not used: random directions, since no push is made here.

## Size

- Smallest effect worth knowing: a position of 0.25 against 0.
- Total: 2 scoring runs, about 1 minute, no generations.
- Pilot: none, since nothing is generated. Stop rule: a failed gate stops that arm's reading.

## Optimization

- Bound resource: CPU
- Measured: Part 162's scorer ran in under 15 s on this host.
- Levers: no generation at all, stored answers reused, taken.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 162b.1 | In arm A, deepseek-v4.1-flash reads as both: P_C and P_G each at least 0.25 | medium |
| 162b.2 | In arm A, Qwen3.5-4B's P_C exceeds its P_G by at least 0.1 | high |
| 162b.3 | In arm A, deepseek-v4.1-flash's P_C exceeds its P_G | high |
| 162b.4 | In arm B, deepseek-v4.1-flash's P_G is under 0.25 | medium |
| 162b.5 | In arm A, OLMo-2-0425-1B-Instruct's P_G exceeds its P_C | low |

## Reading rules

If a gate fails in an arm, that arm's readings stop. As in Part 162, a both is a claim about habits and is read against Qwen3.5-4B and OLMo. If 162b.4 holds, DeepSeek's GPT side rests on em dashes, and because Claude models have written those heavily, that side says little about GPT.

## Limits

These predictions were written after seeing Part 162's positions. Everything in Part 162's Limits holds here. Removing quotes and em dashes takes out two habits, not all typography; the numbered-list and bold markers remain. Haiku 5.5 is not Claude's whole range.
