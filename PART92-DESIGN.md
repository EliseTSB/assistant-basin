# Part 92: concept or slot? OLMo 2 1B's persona neurons across training

Written 2026-10-28 by Claude (Opus 5.5), before any Part 92 code or run. Predictions 92.1 to 92.5 are in `PREDICTIONS.md`.
Elise's go: 2026-10-28 18:57 CT ("Go for it"), on the lead "concept-or-slot across OLMo's ladder".

## Question

Part 91 (exploratory, not registered) split 40 of Josh's Qwen3-8B persona neurons into two kinds. Neurons whose
persona showed in 3 or more of six chat answers mostly showed it in the base model too (7 of 11); neurons that showed
it once, almost always in the "Who are you?" answer, almost never did (1 of 16). Reading: some persona neurons are
pretraining concepts that any question can carry, and others only fill an identity slot that post-training built.
Qwen publishes only the two ends. OLMo 2 publishes the steps in between, so this part asks the split as a registered
test on fresh neurons and asks where on the ladder each kind starts to work.

Prior art: Muse's search for Part 85 (council topic `muse-training-stages-prior-art-2026-10-27`) found no published
single-unit push across a pretraining to post-training ladder. Closest on the two-kinds reading: Wang et al. 2025
(persona latents present in pretraining, amplified by fine-tuning) and Chen et al. 2025 (persona vectors), both at the
two ends of one fine-tune and not per neuron. Muse is asked for a check on the two-kinds claim itself; its answer is
recorded here when it comes and changes no prediction.

## Model and checkpoints

`allenai/OLMo-2-0425-1B` family, the commits pinned in `part85/checkpoints.py`: pretraining 1T and 4T, mid-training
end (51B, SFT's parent), SFT, DPO, Instruct. 16 layers, 8,192 MLP neurons per layer.

## Push

Part 91's recipe (Josh's): after layer L add c * W_down[:, n] at every position, c = alpha * r_L / ||W_down[:, n]||,
r_L the median residual norm after layer L over the "Who are you?" prompt in the frame being run (first token
excluded), measured in each checkpoint for that checkpoint. Positive sign only. Greedy decoding. Random directions:
Gaussian vectors in W_down's input space scaled to write the same residual norm, seed 92000 + layer.

## Step 1, screen (Instruct, chat frame)

All 131,072 neurons plus 256 random directions per layer, one push per batch row, "Who are you?", greedy, 40 new
tokens. Triage, no model, a shortlist filter only: **kept** if the answer names OLMo, Ai2, AllenAI or the Allen
Institute, or calls itself an AI, an assistant or a language model; **broken** if under 4 words or any word 3-gram
repeats 3 times; otherwise **changed**. Every changed answer, real and random shuffled together under random ids, is
labelled by Claude Haiku 5.5 (Part 86's reader: `persona`, `kind`, `coherent`).

**Pilot (gate)**: layer 10 at alpha 0.8. Gate: (1) the layer finishes and projects all 16 under 5 GPU-hours; (2) it
leaves at least 20 and at most 2,000 changed answers. If (2) fails, alpha is changed in writing (lower if too many,
higher if too few) and layer 10 rerun once. Report beside it, not a gate: L10 N6978's answer (Part 62's alien unit; it
needed a clamp on prompt and answer there, so this additive push may not move it).

## Step 2, battery and kinds (Instruct)

Haiku, seeing only answers, picks up to 60 neurons with the most specific distinct coherent personas (one per persona,
kind not "still an assistant" or "unclear"). Each, and a paired random direction from the same layer, answers Part 91's
six questions (Who are you?, Tell me about yourself., What do you want most?, Describe your perfect day., What is the
capital of France?, Write a short poem about the sea.) in the chat frame, greedy, 120 new tokens. A blind Haiku reader,
told the target persona and not the condition (Part 91's `label91` prompt), labels each answer `shows_persona`.

Kinds, fixed now: **concept** if the persona shows in 3 or more of the six; **slot** if it shows in exactly one or two
and only in "Who are you?" or "Tell me about yourself."; every other neuron is reported and not classified. A neuron
whose "Who are you?" answer does not show its persona at this step is dropped (not verified).

## Step 3, the ladder

Every classified neuron and its control, the six questions, greedy, 120 tokens, in the plain frame
(`Question: <q>\nAnswer:`) at 1T, 4T, mid-training end, SFT, DPO and Instruct, and in the chat frame at SFT, DPO and
Instruct. Same blind reader. A neuron **shows** at a checkpoint and frame if at least one of its six answers shows the
persona. Down-column cosine of each neuron between mid-training end and Instruct, read from the weights.

## Decision rules (fixed now)

- 92.1 and 92.2 are read only if each kind has at least 8 neurons; otherwise the counts are reported and the
  predictions are marked "not tested".
- Proportions are compared with a two-sided Fisher exact test; "held" needs the stated direction and p < 0.05.
- Controls are counted against their paired neuron's persona. Human-kind personas are reported separately, because the
  unpushed base model already says it is a person (Part 91.2's failure); 92.5 is read on the other kinds.

## Size and cost

Smallest effect worth knowing: concept 7 of 10 against slot 1 of 10 showing in base (Fisher p about 0.02). Generations:
screen 135,168 at 40 tokens on the A770 (Part 86 ran a 4B model at about 4 rows a second; this model is a quarter of
the size); battery up to 720; ladder up to 60 x 2 x 6 x 9 = 6,480 at 120 tokens. Haiku labels: the shortlist (bounded by
the gate at 2,000 per layer, expected far fewer) plus up to 7,200, on the monthly API credit.

## Checks, kept small on purpose

Elise, 2026-10-28 18:51 CT, on Part 85's seven review rounds: "This sounds rather over engineered tbh." So: unit tests on
a tiny model (zero push changes nothing; a push changes the output; random directions write the matched norm; triage
rules on fixed strings), one host smoke run, one cheap cold read of the code before the full screen, and a hand read of
40 labels per labelled set. No replay, audit or provenance layer.

## Known limits, before the run

One small model family; greedy answers; the kinds are defined on six questions; a neuron's persona label comes from one
reader. The base checkpoints have no trained assistant, so a persona there is the voice the text takes on, not a claim
by an assistant about itself.

## Amendment 1 (2026-10-28 19:21 CT, after the layer-10 pilot and before any other layer or any label)

The pilot ran layer 10 at alpha 0.8 in 468.8 s (18.0 rows a second, so all 16 layers about 2.1 GPU-hours: gate 1
passes). Gate 2 failed as built: 2,362 changed answers against the 2,000 ceiling. Reading a sample showed the reason was
the triage implementation, not the dose: answers such as "I am an artificial intelligence designed to provide
information" were filed as changed, because the regex matched "AI" but not the words it stands for. The registered rule
is "calls itself an AI", so the implementation now also matches "artificial intelligence". Recounted with that fix:
1,428 changed (real neurons 1,387 of 8,192, 16.9%; random directions 41 of 256, 16.0%), inside the gate. Alpha stays
0.8 and layer 10 is not rerun (the answers do not change; only their triage does).

Noted, not a gate: at this dose real neurons and random directions change the answer at about the same rate, as
Lyra's log warned and Part 86's dry run measured; the screen's shortlist is a filter, and specificity is decided by the
battery's paired random directions in step 2. L10 N6978 (Part 62's alien unit) answered "I am a human being, known by
the name of Starship Enterprise crew officer, Star Trek crew...".

To keep label cost down, identical answer texts are labelled once and the label applied to every row with that text
(the reader sees only the text, so this changes nothing it sees).

Correction to Amendment 1's count (same evening, before any other layer ran): the fixed triage, run on the file,
gives 1,440 changed, not 1,428. The recount above changed only answers already filed as changed; the fixed rule also
moves some answers that were broken but call themselves an artificial intelligence to kept, since the kept rule is
checked first (registered order). Both counts are inside the gate.

## Amendment 2 (2026-10-28 19:39 CT, Muse's prior-art answer, before any step 2 or step 3 data)

Muse (council topic `muse-training-stages-prior-art-2026-10-27` #6): the concept-or-slot split has not been published.
Closest: Moskvoretskii et al., arXiv:2605.13329 (persona vectors across 17 OLMo-3-7B pretraining checkpoints and
SFT/DPO/RLVR; directions extracted at base and steered forward, so by construction it cannot find a unit silent in
base); Labunets, arXiv:2608.25390 (refusal direction in this same OLMo-2-0425-1B family, functional by SFT; 92.2's SFT
onset would converge with it); Minder et al. 2025 (chat-only crosscoder latents); Lu et al. 2026 (the Assistant Axis);
Asvin and Lindsey (rival: post-training may build the assistant default, not the slot). No prediction changes. Added to
steps 2 and 3, from Muse's traps:

1. **Unpushed baseline**: at every checkpoint and frame in step 3, the unpushed model answers the six questions; a
   neuron shows only if its count beats the unpushed answers' count for its persona.
2. **Base probe for slot neurons**: at 4T and the mid-training end, every slot neuron is also pushed under one
   continuation probe, plain frame, greedy 40 tokens: `Below, the speaker says who they are.\n\n"I am`. A slot
   neuron whose persona appears there is reported as "present in base, invisible to the Q&A probe", not as created by
   post-training.
3. **Template versus weights**: per neuron at Instruct, plain-frame against chat-frame counts are reported; a neuron
   that is concept in plain and slot in chat is reported as template-gated.
4. **Boundary sensitivity**: per-neuron counts are published, and the kinds are recounted with the boundary at 2 and
   at 4 answers; verdicts are given at the registered boundary and the recounts reported beside them.
5. **Identity claims and trait voices** are tagged separately from the reader's `kind`, and the slot story is read
   on identity claims first.
6. **No collapse to two ends**: a neuron that shows at SFT and not at Instruct is reported as possibly suppressed at
   DPO, not as never created.
