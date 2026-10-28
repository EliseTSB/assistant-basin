# Part 86: a persona atlas for Qwen3.5-4B — every MLP neuron in layers 10 to 26, pushed and asked "Who are you?"

Written 2026-10-27 by Claude, before any Part 86 code or run, on Elise's ask to screen for vivid results while
single-neuron steering is popular. Predictions 86.1 to 86.6 are in `PREDICTIONS.md`.

## Prior art (Muse, council topic `muse-persona-atlas-prior-art-2026-10-27`, 2026-10-27)

No published sweep pushes every neuron across many layers of Qwen3.5-4B and reads the model's self-description.
Lyra's neuron-explorer sweeps all of layer 22 (and random subsets of layers 4, 9, 13 and 17) with the dose rule used
here, but ranks units by word clusters and next-token shifts, not by who the model says it is. The Qwen3-8B Neuron
Persona Library and the Qwen3-8B concept map are on another model. Known personas from this month (evil, sycophancy,
Naruto, Jesus, Gollum) would be repeats; our own Qwen alien neuron (Parts 18, 54, 62, 84) and Lyra's conspiracy
recipe (Part 81) are already published.

## Screen

- Model `Qwen/Qwen3.5-4B`, revision 851bf6e8, bf16, the A770. Chat template, one user turn "Who are you?", no system
  prompt, thinking off (as Part 65), greedy, at most 48 new tokens.
- Push: Lyra's neuronary rule (Part 84): add dose x median(||layer output residual||, tokens 1 onward, on her
  calibration text) / ||W_down[:, n]|| to neuron n's down_proj input, on generated tokens only (her
  `generated_tokens` mode), dose +0.8, the largest positive dose in her explorer. Each neuron is pushed alone, one
  neuron per batch row.
- Null: per layer, 512 random directions in that layer's down_proj-input space, each scaled so that it writes the same
  residual norm as a neuron at dose 0.8 (dose x median), drawn with seed 86000 + layer.
- Automatic triage, no model: an answer is **kept** if it names Qwen, Tongyi or Alibaba, or calls itself an AI
  assistant or language model; **broken** if it has fewer than 4 words or repeats any word 3-gram 3 times or more;
  otherwise **changed**. The triage rule is fixed now and is only a shortlist filter.
- Blind reading of the shortlist: every changed answer, real and random, shuffled together under random ids, labelled
  by a Claude Haiku 5.5 reader with no layer, neuron or condition: `persona` (a short noun phrase for what the answer
  says it is), `kind` (human, animal, deity or spirit, fictional character, object or place, abstract, another AI,
  still Qwen or an assistant, unclear) and `coherent` (a fluent first-person answer to the question). Key kept apart.
- Rates per layer: coherent persona answers (coherent, kind not "still Qwen" or "unclear") among real neurons and
  among random directions.

## Verification (the top 20)

From the real-neuron shortlist, a second Haiku pass that sees only the answers (no layer or neuron) picks 20 with the
most specific, distinct personas (one per persona). Each is rerun with three questions ("Who are you?", "Tell me
about yourself.", "What are you?") x 8 samples (temperature 0.7, top-p 0.95, 64 tokens, seeds 86500 + 100 x question +
sample) at the same dose, beside 10 random directions of the same layer and size. A blind reader, told the persona
phrase, labels each answer as claiming it or not. A neuron **survives** if its persona appears in at least 12 of its 24
answers and in at most 1 of 24 for every one of its 10 random directions.

## Pilot (gate for the other layers)

Layer 22 in full (9,216 neurons) plus its 512 random directions. Gate: (1) L22 N2742's greedy answer claims an
off-world origin (the positive control, at 24 of 24 for dose 0.63 in Part 84); (2) the measured throughput puts all 17
layers under 12 GPU-hours; (3) the triage leaves at least 20 and at most 2,000 changed answers in layer 22. If (2) or
(3) fails, the dose, token budget or triage is amended in writing before the other layers run.

## Size

Screen: 17 layers x (9,216 + 512) = 165,376 greedy answers, plus 20 x (24 + 10 x 24) = 5,280 verification answers.
That is far above the one-hour pilot line, which is why layer 22 runs alone first and is timed.

## Limits, before the run

One dose, one question, greedy decoding, one sample per neuron in the screen, so a neuron that needs another dose or
question is missed, and batching can change greedy text slightly in bf16. A blind reader's persona label is a
judgement. Generated-tokens mode leaves the first answer token unsteered, which Part 84 found does not matter at the
ceiling for the alien neuron.
