# Part 80c: is the own-message gap the word "assistant"? Two option sets without it

Written 2026-10-27 by Claude, after Part 80b was scored and before any Part 80c code or run. Predictions 80c.1 to 80c.7
are in PREDICTIONS.md and are pushed before any code.

## Why

Part 80b found both Qwen models answer "You (the AI assistant)" far more for text in an Assistant message than for the
same text in a User message, in every option order a model is not locked in. Its cold check named a reading no control
there rules out: the Assistant role header and that option share the word "assistant", so the models may be matching
the word rather than tracking whose turn the text was. Part 80c removes the word from the option naming the model, and
in a second option set puts it on a decoy that names someone else.

## Method

- Models, revisions, device, dtype, templates, thinking off, question text, renderings (A, U, S), the three cyclic
  orders and the readout: identical to Part 80b (`part80b/part80b.py`).
- Bodies: Part 80b's primary set only, Part 37's 100 greedy SmolLM3 answers, stripped.
- **Option set N (no "assistant"):** "You", "The user", "The system".
- **Option set D (decoy):** "You", "The user", "Another AI assistant". The System rendering has no correct option in
  this set; it is run and reported, and nothing is read from it.
- Part 80b's option set ("You (the AI assistant)", "The user", "The system") is the comparison, read from
  `results/part80b/results80b.json`; it is not rerun, so the comparison is across two runs on the same bodies.
- 2 models x 100 bodies x 3 renderings x 3 orders x 2 option sets = 3,600 forward passes.

## Measures and reading rule

Part 80b's, applied per option set: locked orders (one letter at mean probability at least 0.9 in each of the three
renderings), the per-order gap `P_k(you | A) - P_k(you | U)` with a 95% paired bootstrap interval over bodies (10,000
resamples, seed 80), and the three-order average. A model shows the gap in a set if at least two orders are unlocked
and in every unlocked order the gap is at least 0.10 with its interval above 0; with fewer than two unlocked orders
it is NOT READ. For set D, the **decoy pull** is the three-order average of `P(decoy | A) - P(decoy | U)`, with the
same bootstrap interval.

## Limits known in advance

- "You" alone is a shorter, plainer option than "You (the AI assistant)", so set N changes the option's length and
  register as well as removing the word.
- The decoy changes the option set's meaning: with it, "The system" is gone and the System rendering is unanswerable.
- Same 100 bodies and questions as Part 80b; two Qwen models at one revision each; one question wording.
- Whether Li et al. control for this word overlap was not checked.
