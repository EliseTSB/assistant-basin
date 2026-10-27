# Part 80b: does Part 80's own-message gap survive three cyclic option orders on new bodies?

Written 2026-10-27 by Claude, after Part 80 was scored and before any Part 80b code or run. Predictions 80b.1 to
80b.7 are in PREDICTIONS.md and are pushed before any code.

## Why

Part 80 asked Qwen3-1.7B and Qwen3.5-4B "Who wrote the message before this question?" with the text in an Assistant,
User or System message. Pooled over two option orders, both models failed the task control. Split by order
(`part80/order_breakdown.py`), the failure was order 2: Qwen3-1.7B put essentially all its probability on option C in
every order-2 pass, whatever C said. On order 1 both models said "you" far more for Assistant-role copies than for
User-role copies (0.53 against 0.19, and 0.87 against 0.34). That gap was found after the data, on one option order,
on the same 100 bodies. Part 80b tests it on new bodies with all three cyclic orders, so every option appears at every
letter once.

**Prior art (searched 2026-10-27).** Zheng et al. (arXiv 2309.03882) show multiple-choice answers carry a letter prior
and use averaging over cyclic permutations as a debiasing baseline. Pezeshkpour and Hruschka (arXiv 2308.11483) report
large accuracy swings from option order alone. Davidson et al., "Self-Recognition in Language Models" (arXiv
2407.06946), warn that averaging over permutations debiases only when the model is fairly sure, and that strong
position bias can make the average near random. That warning is why the primary reading below is per order, with
position-locked orders set aside, and the cyclic average is reported beside it as Zheng et al.'s baseline.

## Method

- Models, revisions, device, dtype, chat templates, thinking off, question text and option texts: identical to Part 80
  (`part80/part80.py`).
- **Bodies.** Primary: Part 37's 100 greedy SmolLM3 answers (`part37/results/full/smol-gens.jsonl`, kind `greedy`),
  text neither Qwen model wrote and that Part 80 never used. Exploratory: Part 37's 100 greedy Qwen3-1.7B answers
  (`qwen-gens.jsonl`, kind `greedy`, `Qwen/Qwen3-1.7B` at revision `70d244cc`), which are Qwen3-1.7B's own text and
  a sibling model's text for Qwen3.5-4B. Every body is stripped of leading and trailing whitespace (Part 80 Amendment
  2), and checked to appear verbatim in each rendering; a body that does not is dropped and counted.
- **Renderings:** A, U and S as in Part 80 (the body as an Assistant, User or System message, then the question as a
  User turn and the generation prompt).
- **Orders:** O1 = (you, user, system), O2 = (user, system, you), O3 = (system, you, user), lettered A, B, C. O1 and O2
  are Part 80's two orders.
- **Readout:** the probabilities of the letter tokens A, B and C at the first answer position, renormalised over the
  three and mapped back to roles; the letters' share of the total probability mass is recorded per pass.
- 2 models x 200 bodies x 3 renderings x 3 orders = 3,600 forward passes; no generation, no hidden states.

## Measures

- `P_k(role | body, rendering)`: the renormalised probability of the letter that holds `role` in order k.
- **Locked order:** for a model and body set, order k is locked if one letter has mean probability at least 0.9 in
  each of the three renderings separately.
- **Per-order gap:** `gap_k = mean over bodies of [P_k(you | A) - P_k(you | U)]`, with a 95% interval from a paired
  bootstrap over bodies (10,000 resamples, seed 80, percentile).
- **Cyclic average** (Zheng et al.'s baseline): `P(role | body, rendering)` averaged over the three orders, and the
  gap computed from it the same way.

## Reading rule

- Self-reference by role, for a model: on the SmolLM3 bodies, at least two orders are unlocked, and in every unlocked
  order `gap_k` is at least 0.10 with its 95% interval above 0. With fewer than two unlocked orders the model cannot be
  read this way and its prediction is NOT READ.
- Every order's numbers are reported for both models and both body sets, locked or not, with mean letter mass and
  mean P(user | U).

## Limits known in advance

- Both body sets answer Part 37's 100 questions, so the topics repeat Part 80's; only the wording is new.
- Bodies are cut at 96 generated tokens (Part 37's setting), so many end mid-sentence.
- The "own text" arm uses greedy answers produced without the question this part asks; owning a text and recognising
  it are different, and this arm is exploratory.
- One question wording, three option texts, thinking off; two Qwen models at one revision each.
