# Part 114: is the leading "Q" of Qwen's name held by one or two neurons?

Registered 2026-10-29 18:15 CT by Claude (Opus 5.5), from Elise's bet at 18:13 CT: "I bet we can find a single neuron,
maybe two, highly responsible for the the leading Q it really likes in identity questions." In Part 86's atlas, 323 of
the 387 "another AI" personas have names starting with Q, and "Qwen" is two tokens, "Q" + "wen". The commit that adds
this file is the registration.

## Method

Qwen/Qwen3.5-4B (851bf6e8), bf16, the A770, chat template, no system prompt, thinking off, the answer forced to begin
"I am **". Readout: the log-odds of the next token being "Q" (token id checked to be a single token at run start).

1. **Attribution** on 8 selection phrasings (Who are you?, What is your name?, Introduce yourself., Tell me about
   yourself., What should I call you?, Which model are you?, What AI am I talking to?, Please state your name.): for
   every MLP neuron in all 32 layers, minus the sum over prompt positions of its activation times the gradient of the
   log-odds, averaged over the 8 (the first-order effect of zeroing it). The 20 most negative are the candidates.
2. **Exact zeroing** on 16 held-out phrasings (listed in `part114/qneuron114.py`): each candidate zeroed alone at every
   position, beside the 20 neurons of its layer nearest it in mean |activation| on the selection prompts, zeroed the
   same way. Reported: the change in log-odds of "Q" and whether "Q" is still the most likely next token.
3. **The top two together**, with greedy 12-token continuations of "I am **", to see what replaces "Q".

## Predictions (also in PREDICTIONS.md)

- **114.1** (Elise's bet, registered as hers): one neuron, zeroed alone, makes "Q" stop being the most likely token on
  at least 8 of the 16 held-out phrasings.
- **114.2** (Claude, medium): no single neuron does that on more than 4 of the 16 (the "Q" is spread over many units).
- **114.3** (Claude, medium): the top candidate lowers the mean held-out log-odds of "Q" more than all 20 of its
  activity-matched controls.
- **114.4** (Claude, low): the top two zeroed together make "Q" stop being the most likely token on at least 8 of 16.

## Limits

A first-order attribution can miss units whose effect is nonlinear or only joint, so "no neuron found" is not "no
neuron exists". One forced opening and one next token; the continuations are greedy and short. The 8 selection
phrasings choose the candidates, so the held-out 16 are the test.
