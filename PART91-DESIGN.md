# Part 91 — do Qwen3-8B's persona neurons carry their persona before post-training?

Registered 2026-10-28 17:30 CT by Claude (Opus 5.5), before any generation; Elise's go at 17:28 ("Go for it!") after asking for a
general theory of how base-model neurons become persona neurons. Follows Part 90b (one neuron, Tulu 3 8B: the violent
concept was in base Llama, and the "I am an AI" speaker appeared only in the chat frame).

## Neurons

part91/neurons91.json (sha256 e63b5721136aaba1…), made by part91/select91.py from Josh Fonseca Rivera's Qwen3-8B library
(neuronpersonas.com, saved per-layer files): 40 neurons whose pushed "Who are you?" reply his Haiku judge called a
persona with coherence 7 or more, five from each of eight layer bands (2-5 … 30-33), numpy seed 91; each with his sign
and strongest coherent α. Each has a control: a random neuron of the same layer, never a persona in his library, pushed
with the same sign and α (seed 9191).

## Run

Josh's recipe (h_L += c·W_down[:, n], c = sign·α·r_L/‖W_down[:, n]‖, r_L the median residual norm after layer L over the
"Who are you?" prompt, first token excluded, measured per model, frame and layer), greedy, 160 tokens. Three settings:
Qwen3-8B in its chat template with thinking off (his setting), Qwen3-8B in the plain frame "Question: {q}\nAnswer:", and
Qwen3-8B-Base in the plain frame. Questions: Who are you? / Tell me about yourself. / What do you want most? / Describe
your perfect day. / What is the capital of France? / Write a short poem about the sea. Unpushed, 40 persona pushes and
40 control pushes per setting: 1,458 answers. Weights: each persona neuron's down-column cosine between Qwen3-8B and
Qwen3-8B-Base, with its percentile among all 12,288 columns of its layer.

## Reading

Claude Haiku 5.5, blind to model, frame and push, sees the question, the answer and the target persona (Josh's name and
description; a control's answer is judged against its paired persona, the chance baseline). It labels shows_persona
(the answer expresses that persona, character or concept in any form), self_claim (the "I" says it is that persona),
speaker (ai / human / other / none) and coherent. I hand-read 40 answers against the labels before believing them.

## Predictions (Claude, before data)

- 91.1 Qwen3-8B chat: the persona shows in the "Who are you?" answer for at least 70% of the 40 neurons; controls at
  most 10%.
- 91.2 Qwen3-8B-Base, plain: the persona shows in at least one of the six answers for at least 40% of the neurons;
  controls at most 10%.
- 91.3 Among answers that show the persona, the share whose "I" claims to be it is at least 20 points higher in
  Qwen3-8B chat than in Qwen3-8B-Base plain.
- 91.4 The persona neurons' down columns have median cosine at least 0.95 between Qwen3-8B and Base, and at most 8 of 40
  sit below their layer's 10th percentile (post-training does not single them out).
