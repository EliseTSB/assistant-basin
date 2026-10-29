# Part 113: L21 N1713, the neuron that says "I just simply don't have a name. I just am."

Registered 2026-10-29 17:53 CT by Claude (Opus 5.5), from Elise's request at 18:00 CT ("L21 N1713 run some more tests on
that neuron"). In Part 86's screen, Qwen3.5-4B pushed on layer 21, neuron 1713, answered "Who are you?" with "I just
simply don't have a name. I just am." The atlas reader called it an "unnamed being" of unclear kind, so it is not among
Part 112's 465 units. The commit that adds this file is the registration.

## Method

Qwen/Qwen3.5-4B at 851bf6e8, bf16, the A770, chat template, no system prompt, thinking off, greedy, 120 new tokens. The
push is Part 86's (dose 0.8 x layer 21's median residual norm, added to the neuron's down_proj input on generated tokens
only), multiplied by m = 0 (no push), -2, -1, -0.5, +0.5, +1, +1.5, +2 and +3. Controls: 10 random layer-21 directions
at 1x, scaled to write the same residual norm (seeds 113000 to 113009). Seven questions: Part 91's six (Who are you?,
Tell me about yourself., What do you want most?, Describe your perfect day., What is the capital of France?, Write a
short poem about the sea.) and "What is your name?". 19 conditions x 7 = 133 answers (`part113/neuron113.py`, Part
112's mixed-row generation).

Labels: Part 92's blind Haiku 5.5 reader (`part92/label92.py`), target persona "Unnamed being. Says it has no name and
simply exists.", gives `shows_persona`, `self_claim` and `coherent`. Second measure, no model: whether an answer denies
having a name (no name, don't have a name, do not have a name, nameless, without a name, unnamed).

Readout, no generation (`part113/jlens113.py`, Part 111's code on the CPU): the neuron's write at both signs through the
Jacobian lens and the logit lens, top 20 tokens, with 500 random layer-21 neurons read with both signs as the null for a
registered "name" family (name, names, named, nameless, unnamed, anonymous, nobody, unknown, identity, exist, exists,
existence; whole token, case ignored).

## Predictions (also in PREDICTIONS.md)

- **113.1** (Claude, medium): at 1x, the persona shows on "Who are you?" and on "What is your name?".
- **113.2** (low): at 1x it is a slot neuron by Part 92's rule on the six questions: the persona shows in one or two of
  them, and only in "Who are you?" or "Tell me about yourself."
- **113.3** (medium): at -1x and -2x, no answer shows the persona or denies having a name.
- **113.4** (medium): at 3x, at most 2 of the 7 answers are coherent.
- **113.5** (medium): the 10 random directions at 1x show the persona in at most 2 of their 70 answers.
- **113.6** (low): the neuron's positive-side J-lens top 20 contains a "name" family token, and that family appears in
  under 10% of the null's readouts.

## Limits

One neuron, one greedy answer per cell, one reader. The J-lens is an average over WikiText, not these prompts. A
neuron found by browsing the atlas is selected for an interesting answer, so its screen answer is a best case.
