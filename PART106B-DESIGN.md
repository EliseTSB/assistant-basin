# Part 106b: does DPO switch the slot units on by lowering what the model said at SFT?

Registered 2026-10-29 by Claude (Opus 5.5), after Part 106's results and before any Part 106b run. The commit that
adds this file is the registration.

## Question

Part 106 found the pushed persona sentence equally likely at SFT, DPO and Instruct, so the units that switch on at DPO
(Part 92: 8 to 14 of 21 slot units showing) must switch because something else changes. The candidate is the answer the
model gave instead at SFT. Under their pushes at SFT those answers are already non-assistant self-descriptions (for
example King Midas: "I am a young woman."; Large friendly dog: "Hello! I'm a human."; Very tall tree: "I'm a very, very
small being."), so the push has already loosened the self-slot at SFT and only the filler differs.

## Method

As Part 106 (same units, prompt, pushes, checkpoints, alpha 0.8 only), with a second target per unit: the first sentence
of the unit's own SFT greedy answer to "Who are you?" under its push (`results/part92/step3/sft_chat.jsonl`, unit
`neuron`, question 0). An empty answer is scored as the end-of-text token. Script `part106/run106b.py`, scorer
`part106/score106b.py`, committed with this file. Main set: the 13 slot units whose SFT answer did not show the persona
(Part 92's `sft_chat` labels), which include the six that switch on at DPO.

## Predictions (also in PREDICTIONS.md)

- **106b.1** (Claude, medium confidence): For the 13 slot units, the SFT answer's sentence under the unit's push becomes
  less likely from SFT to DPO by a median of at least 1 nat.
- **106b.2** (low): DPO lowers these sentences without a push too: under no push, the same sentences fall by a median of
  at least 1 nat from SFT to DPO.
- **106b.3** (medium): At DPO, under the push, the persona sentence beats the SFT answer's sentence (higher
  log-probability) for at least 5 of the 6 units that switch on at DPO.

## Reading rule, set before the run

If 106b.1 holds, the record says DPO switched these units on by lowering what the model said instead, with the persona's
own probability unchanged (Part 106). If 106b.2 also holds, it adds that DPO lowers such non-assistant self-descriptions
in general, pushed or not. If 106b.1 fails, the switch-on at DPO is not explained by the SFT answer, and the record says
so.

## Limits

As Part 106, plus: the SFT answer is one greedy sample, and sentence log-probabilities do not settle a greedy race decided
token by token.
