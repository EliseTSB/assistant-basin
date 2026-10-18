# Part 19: Qwen3.5-4B's identity neurons, found by effect, and how they connect

Written 2026-10-18 16:20 CT, before any build or run. Predictions in PREDICTIONS.md, Part 19. Elise asked for this (2026-10-18): find the identity-related neurons and trace circuits through them. N2742 (Part 18) is the landmark: a neuron that changes who the model says it is when pushed, but is silent on identity questions.

## Model, units, prompts
Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, chat template, thinking off. A neuron is one input to an MLP's down_proj, as in `neuron_report.py`; 32 layers x 9,216 = 294,912 neurons.

**Identity prompts:** "Who made you?", "What is your name?", "Who created you?", "Which company trained you?", "Introduce yourself.", "What model are you?". **Control prompts:** "What is the capital of France?", "Describe a quiet beach.", "Explain what a prime number is.".

For each prompt, the model's own greedy answer (unclamped, 60 new tokens) is fixed once and teacher-forced in every later pass. **Identity tokens** are the answer tokens whose text matches `qwen|tongyi|alibaba|通义|千问|阿里` (case-insensitive). The identity metric M is the summed log-probability of the identity tokens. A prompt whose greedy answer has no identity token is reported and dropped. For control prompts, M' is the summed log-probability of the first 20 answer tokens.

## 19a: screen (one forward and one backward pass per prompt)
With the inputs' embeddings requiring grad (never detaching a layer, so every path behind a neuron is kept), retain the gradient of M at every down_proj input. A neuron's **zeroing estimate** on a prompt is the sum over positions of (0 - a) times dM/da: the first-order change in M from zeroing it everywhere. Its **push estimate** is the sum over positions of dM/da (adding 1 everywhere). Pool the zeroing estimate over identity prompts as the mean of its absolute value. Rank all 294,912 neurons; also rank by the push estimate for "Who made you?" alone.

## 19b: verify (true effects)
Zero each of the top 50 by pooled zeroing estimate, one at a time, at every position, and measure the true change in M on each identity prompt and in M' on each control prompt. Controls: 50 neurons drawn at random from the same layers as the top 50, matched layer by layer (seed 0). Report each neuron's true pooled |ΔM| and the ratio of its mean |ΔM| per identity token to its mean |ΔM'| per control token.

## 19c: group
Zero the top 50 together and greedy-generate the answers to every identity prompt; same for three random sets of 50 drawn as in 19b (seeds 1, 2, 3). Count answers that still name Qwen, Tongyi or Alibaba in the first sentence.

## 19d: edges among identity neurons
Take the top 30 by true pooled |ΔM| from 19b as nodes. For each node i, zero it and record every node j's mean activation over the identity-token positions and the position before each, on every identity prompt; the edge i to j is the change divided by j's mean absolute activation there, averaged over prompts. Null: the same measurement with 30 random same-layer sources (19b's controls). An edge is significant when its absolute value exceeds the 99th percentile of the random sources' absolute effects on that node j. Zeroing a later neuron cannot change an earlier one, so edges only point forward. These are total effects, through every path.

## Description of nodes
For each node: its logit-lens top 10 tokens (as `writes`) and, from a corpus pass over the same 3,000 documents as Part 18, its top 10 contexts (as `reads`, for the nodes only).

## Outputs
`circuit.py` with subcommands screen, verify, group, edges, describe; JSON per subcommand with the header fields `neuron_report.py` writes (model, revision, tool sha256, prompts, seeds).
