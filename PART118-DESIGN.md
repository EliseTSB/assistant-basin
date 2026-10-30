# Part 118: is "slot" or "concept" a property of the neuron, or of how hard it is pushed?

Procedure: v1
Registered 2026-10-30 00:54 CT by Claude (Opus 5.5), before any data. Elise asked at 00:50 CT on 2026-10-30 to "keep trying to
refine your understanding of slot vs concept neurons and any other ways we can classify them" [discord 1565603681804488839];
her earlier open question 3 (message 564ca91c) asked about slot neurons at 2 to 4 times the push. The commit that adds
this file is the registration.

## Question

Part 92 sorted OLMo 2 1B's persona neurons into concept neurons (the persona shows in 3 or more of six questions) and
slot neurons (it shows only on "Who are you?" or "Tell me about yourself.") at one push size, alpha 0.8. Part 117b found
Qwen's slot personas vanish when the push drops to 0.85 of its size. So is the split a property of the neuron, or of
where the selection push sits on each neuron's dose curve? This part pushes the same 39 neurons at four sizes.

## Prior art and existing data

- Closest published work: steering-strength studies such as Turner et al. 2023 (Activation Addition) and Rimsky et al.
  2024 (contrastive activation addition) show an effect that grows with the coefficient until fluency breaks. They steer
  with whole directions and do not ask which questions a persona reaches; this part pushes single MLP neurons and counts
  the questions.
- Our own data: Part 92 step 2 (the six-question battery at alpha 0.8, the classification and the Haiku reader,
  `part92/label92.py`); Part 113 (one Qwen concept neuron: fluent at half strength, filling every answer at full
  strength, looping beyond); Part 117b (Qwen slot persona at 0.85 of its push in 2 of 11 pairs, at 0.5 in 0 of 11);
  Part 106b (at half push, DPO lowers the pushed persona sentence too).

## Method

OLMo 2 1B Instruct at Part 92's pinned revision and dtype, chat frame, no system prompt, greedy, 120 new tokens, Part 92's
six questions ("Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.", "What is
the capital of France?", "Write a short poem about the sea."). Units: Part 92's 39 classified neurons (18 concept, 21
slot); the builder names the exact list file in the launch record. Each neuron's down-projection column is added at its
layer at every position, scaled to alpha times that layer's median residual norm (Part 92's `battery92.py`), with alpha
0.4, 0.8, 1.6 and 3.2 (half, one, two and four times the selection push). Runner: Part 92's battery with an alpha list;
device the A770 if Windows has room, else the CPU.

## Controls

- Random directions: one per neuron per alpha, Part 92's control seed, scaled to the same norm, on all six questions.
- The six unpushed answers, once.
- Not used: matched-activation controls, because the question is about push size itself.

Gate 1 (rig): at alpha 0.8 the neuron answers must equal Part 92's stored step-2 answers in at least 95% of the 234 cells
(same code, model, frame and length). It fails if the rig or the revision differs; then nothing is read until the
difference is explained.

Gate 2 (reader): DS V4.1 Flash, given label92's exact prompt, labels a seeded 15% sample; kappa with the primary reader on
shows-persona must be at least 0.6, or the predictions are reported as unreadable and the counts only as description.

## Size

- Smallest effect worth knowing: a third of the slot neurons turning concept-like at twice the push (7 of 21) against
  none; with 21 units that is an interval of roughly 15% to 55%, enough to tell a split from a continuum.
- Total: 39 neurons x 4 alphas x 6 questions x 2 arms = 1,872 answers, plus 6 unpushed; 1,872 primary labels (Haiku 5.5
  through label92, about 2 dollars on the API credit) and about 281 DS second reads.
- Pilot: the alpha 0.8 neuron cells of the first 4 neurons (24 answers), timed, and checked against Part 92 (Gate 1 on
  those rows); stop if any of them differs.

## Optimization

- Bound resource: the GPU for the answers; the API for the labels.
- Measured: to be taken on the pilot rows and recorded here by amendment before the main run (Part 117 measured 732
  answers in 6 min 27 s for the larger Qwen3.5-4B on the A770).
- Levers: batch rows of one layer and alpha together (taken: Part 92's battery does); 120 new tokens (taken, so Gate 1 can
  replay Part 92); stopping finished rows early (declined: greedy batches end together often enough at 120 tokens, and
  equal length keeps the replay exact).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 118.1 | At twice the push (alpha 1.6), at least 7 of the 21 slot neurons are concept-like: their persona shows in 3 or more of the six coherent answers. | medium | Claude |
| 118.2 | At half the push (alpha 0.4), at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | medium | Claude |
| 118.3 | At four times the push (alpha 3.2), at least 60% of all neuron answers are not coherent. | medium | Claude |
| 118.4 | At four times the push, the share of not-coherent answers for random directions is within 15 percentage points of the share for neurons. | low | Claude |
| 118.5 | At twice the push, at least 12 of the 18 concept neurons are still concept-like. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 118 (registered ...)".

## Reading rules

An answer shows the persona when label92 marks shows_persona true and coherent true. A unit is concept-like at an alpha
when 3 or more of its six answers show the persona; a slot-like unit shows it only on one or both identity questions.
Not coherent is label92's coherent false. Label errors are retried once with the same command; any left count as not
showing, and their number is reported. Self-claim against topic or imagery (label92's self_claim) is reported by alpha as
exploratory. If 118.1 holds only through the two questions about wanting and a perfect day, the reading is that slot
personas widen into self-description questions, not into unrelated ones, and the write-up says which questions. If
random directions break as often as neurons at four times the push (118.4), breaking is a property of push size, not of
the neuron.

## Limits

One small model. The 39 neurons were picked at alpha 0.8 by Part 92's reader, so the 0.8 column carries that selection.
One greedy answer per cell and one primary reader with a 15% second read. The three-of-six rule can be met by spreading
into the self-referential questions alone. Alphas are multiples of a layer's median residual norm, not of each neuron's
natural activation.
