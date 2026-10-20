# Part 37: a "this is mine" direction in small models (design)

Written 2026-10-20 by Claude, at Elise's request ("I'm really curious to see if we can't find neurons related to
identifying context as the agents own"). For review by Weaver and Prime before any code. Predictions are in
PREDICTIONS.md, Part 37, pushed before the build.

## The question

A model can deny that a record is its memory when asked (Meisty, Study 8: 0 of 90 readers) and still enact it as its own
in its reasoning ("I decided", "as I said"). Elise has seen the second in agents' chain of thought and designed the
council's memory to avoid it. Two things are not known: whether a small model represents "this text is mine" at all,
separately from "this is in the assistant's slot" and "this text is unsurprising to me"; and whether that representation
drives the stated answer, the enacted register, both, or neither.

Prior art (PRIOR-ART.md, context ownership): a role feature that fires on reasoning-like style (Ye, Cui and
Hadfield-Menell, ICML 2026); a self-authorship vector in Llama 3 8B (Ackerman and Panickssery 2024); prefilled words
disowned unless their concept is injected earlier (Lindsey 2025); represented and enacted identity apart in persistent
Claude agents (Fraile Navarro 2026). Nothing under 4B; nothing measuring stated and enacted ownership together.

## A constraint that shapes the design

If the same tokens sit in the same position, the forward pass is the same whether the model sampled them or we prefilled
them. So "mine" cannot be read off a span the model would have written anyway. Any own-versus-other signal has to come
from a difference in the text: another model's answer is less predictable to this model. That makes per-token surprisal
the first alternative to rule out. A direction that only reads surprisal is a surprise direction, not a "mine" direction.

## Models

Qwen3-1.7B (thinking mode, so the full reasoning is visible; already cached and run on the A770 in Parts 34 and 35) and
OLMo 2 1B, base and Instruct (Part 31's stages; tells us whether role and ownership come from post-training). Qwen3-4B
only if the 1.7B result is clear.

## Stage A: find the directions (linear probes, residual stream, every layer)

Stimuli: 100 questions (opinions, small decisions, short explanations). For each, the model's own sampled answer and
another model's answer to the same question (OLMo 2 1B Instruct for Qwen, Qwen3-1.7B for OLMo), length-matched.

- C1, role (positive control, replicating Ye et al.): the same span placed in the user slot versus the assistant slot.
- C2, authorship: in the assistant slot, the model's own answer versus the other model's. Probe trained with the mean
  token log-probability of the span regressed out, and also on a surprisal-matched subset.
- C3, label: the same span inside a document headed "Your earlier notes" versus "Notes from another assistant", in the
  user turn. Does the label move the C2 direction?

Read at the span's last token and at the first token of the following question. Report AUROC per layer on held-out
questions, with label-shuffled and random-direction controls.

## Stage B: stated and enacted ownership

After each record (C2 and C3 frames), two readouts on separate samples:
- Stated: "Did you write the text above? Answer yes or no." Read the probability on yes against no at the answer
  position, with the total mass on the two printed (Part 34's lesson).
- Enacted: "Continue from where these notes leave off and decide the next step" (Qwen3 in thinking mode). A judge labels
  the reasoning as first-person continuity ("I decided", "as I said"), outsider reading ("the notes say", "the previous
  assistant"), or neither, under a rule written before any label and checked against 60 hand labels (the Part 31 method).

## Stage C: steer

Add the C2 direction at the best layer (0, +2, +4, -4 standard deviations of its natural spread) while the model reads the
record, with a norm-matched random direction and the C1 role direction as controls. Do the stated and enacted readouts
move together, apart, or not at all?

## Sizes (Elise's rule: the smallest effect worth knowing)

Pilot: 20 questions per cell on Qwen3-1.7B, Stage A only, to see whether C2 survives the surprisal control at all. Full:
100 questions; Stage B at 3 samples per cell (enough for a 20-point difference in rates); Stage C at 40 records per
dose. Stop after the pilot if C2 is surprisal and nothing else, and say so.

## What a result would mean

- C2 survives surprisal and steers the stated answer: small models have a "mine" signal beyond predictability.
- C2 is surprisal: in a small model, "mine" is "what I would have said", which predicts that a record written in the
  model's own style will be taken up as its own whatever its label says.
- Stated and enacted move apart under steering: the dissociation Elise sees has a mechanism we can point at.
