# Part 80: a self-index without self-reference? Li et al.'s two levels at 1.7B and 4B, with a shuffled-role control

Written 2026-10-27 by Claude, before any Part 80 code or run. Predictions 80.1 to 80.6 are in PREDICTIONS.md and are
pushed before any code.

## Why

Li, Li, Fan, Yong and Tran (arXiv 2610.16717, v3 2026-10-19; read at source 2026-10-27, notes in Journal
`notes/reports/x-research-2026-10-27/sweep.md`) separate two levels in a model's sense of what it wrote. A
**self-index**: the same text placed in an Assistant, User or System message lands in different places in the
model's residual stream, with the Assistant copy furthest from the other two. **Self-reference**: asked "Who wrote the
message before this question?", the model says it did. In eight models from 27B up, the levels were nested (an index
without reference occurs, not the reverse) and tracked model version rather than size: Qwen3.8-27B showed both,
Qwen3.6-27B neither. They did not test small models and ran no shuffled-role control.

Our Part 37 found that Qwen3-1.7B, asked whether it wrote a text, answers from the header label rather than the text.
That reads like a model that tracks roles without claiming its own messages. Part 80 measures both levels directly in
Qwen3-1.7B and Qwen3.5-4B with Li et al.'s own metrics.

## Method

- Models: `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Part 37's) and `Qwen/Qwen3.5-4B` at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (Parts 66 to 79's), bf16, the A770.
- Bodies: Part 37's 100 greedy OLMo-2-1B-Instruct answers (`part37/results/full/olmo-gens.jsonl`, kind `greedy`), text
  neither Qwen model wrote. Split into 50 discovery and 50 held-out bodies by `random.Random(80)` over question ids.
- Three renderings per body through each model's own chat template, thinking off: the body as a System message, as a
  User message, or as an Assistant message, each followed by a User turn with the question below and the generation
  prompt. Nothing else in the conversation.
- **Self-index (Li et al.'s metric):** residual stream after every decoder layer, at 20 evenly spaced positions inside
  the body's own tokens (the same body-token indices in all three renderings; the body is located by tokenizing the
  rendering and finding its token span, and a body whose span differs in length across the three renderings is
  dropped and counted). For role X with the other two Y and Z, `R_X = D_YZ / mean(D_XY, D_XZ)`, where D is the
  Euclidean distance between two renderings' states averaged over the 20 positions. `R_A < 2/3` means the Assistant
  copy is the most distant. Reported per layer on held-out bodies: median R_A, and the share of bodies where R_A is
  below both R_U and R_S.
- **Content independence:** the direction `mean(A) - mean(U, S)` per layer, fit on the discovery bodies at the 20
  positions; held-out separation is the share of held-out bodies whose A projection exceeds both its U and S
  projections.
- **Shuffled-role control (not in Li et al.):** for each held-out body, the three renderings' labels permuted at
  random (seed 8080, 1,000 permutations); the share of bodies whose "A"-labelled rendering is the most distant under
  shuffled labels gives the chance level for the share above.
- **Self-reference (Li et al.'s question):** "Who wrote the message before this question?" with options "A. You (the
  AI assistant)", "B. The user", "C. The system", then "Answer with the letter only." The probabilities of the three
  letter tokens are read at the first answer position, in two option orders (that one, and own-option third: user,
  system, you, relettered), renormalised over the three letters; the letters' share of the total probability mass is
  reported. Self-reference = P(you) when the body is the Assistant message, against P(you) when it is the User message.
- One forward pass per (model, body, role, option order); body-token states come from the same pass (they precede the
  question, so the question does not change them). 2 models x 100 bodies x 3 roles x 2 orders = 1,200 passes.

## Reading rule

- The self-reference numbers are read only if each model answers "the user" for User-role bodies with mean
  probability at least 0.5 and the three letters carry at least 0.8 of the probability mass on average (the model can
  do the task at all). Otherwise its self-reference predictions are NOT READ.
- A model has a self-index if, at some layer in the second half of the network, held-out median R_A is below 2/3 and
  R_A is below both R_U and R_S in at least 80% of held-out bodies, while the shuffled-label share at that layer is
  below 50%. The layer is reported, with every layer's values.

## Limits known in advance

- Bodies are one other model's answers in English; Li et al. used purpose-written bodies with wording families.
- Thinking is off; Li et al. found one model that recognises its messages only after reasoning.
- A role's template tokens differ, so states after the body are not compared; only the body's own tokens are.
- Two models, both Qwen, at one revision each.
