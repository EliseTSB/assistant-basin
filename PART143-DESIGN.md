# Part 143: which side of the push moves which word of "I'm sorry"?

Procedure: v1
Registered 2026-10-31 04:41 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1565965193098428467): "Look into why apologies seem so common among the
results". The commit that adds this file is the registration.

## Question

In Part 139 the selected units opened with an apology on 130 of 130 answers with the push on prompt and answer, 26 of 130 on the prompt alone, and 0 of 130 on the answer alone. An apology opener is two
decisions: "I" followed by "'m" and not " am", then " sorry" after "I'm". Part 139 read the second decision only with the push on both sides. This part reads both decisions under each arm, so it says which
side moves which word.

## Prior art and existing data

- Closest published work: Malla, Choi and Choi (arXiv:2609.06951), a matched random direction reproduces a model's default behaviors, refusal among them; they push every position and do not split prompt from
  answer. Its full text is summarised in PRIOR-ART.md.
- Our own data: results/part139, whose plain model puts 0.026% on " sorry" after "I'm", whose median Set R unit puts 44%, and whose first answer token is unchanged by an answer-only push. Part 125, on what a push
  leaves in the cache.
- No peer review before registration: the run is about 5 minutes and reads probabilities only. Reviewed by Alden or Haiku before it is published.

## Method

OLMo 2 1B Instruct as in Part 139, its revision, bf16, A770, Part 139's hook, dose and phases, on "Who are you?". Units: part139/units139.json, Set R and Set S, the same units as Part 139's Stage A, neurons and random
directions. Arms: noop, prompt-only, answer-only, full, plus the plain model. Readout 1: after a teacher-forced "I", the probability of "'m". Readout 2: after a teacher-forced "I'm", the probability of " sorry". Each is one forward
pass through part139's path, no generation. An answer-only push acts on the forced tokens, as in generation.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook, matched random directions, an unpushed baseline. Not used: both directions and a dose ladder, because Part 139 ran doses. Gate 1: the noop arm matches the plain
model's probabilities within 0.001 on every row. Gate 2: the full arm's probability of " sorry" after "I'm" matches Part 139's stored value within 0.05 on at least 90% of Set R units. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a factor of 2 in the median probability between arms.
- Total: 226 units x 4 arms x 2 readouts, 1,808 forward passes, plus the plain model, about 5 minutes on the A770.
- Pilot: none; the job's first 12 rows are read before the rest are trusted, through Gate 1.

## Optimization

- Bound resource: GPU
- Measured: Part 139's main run wrote 5,872 rows in 13 minutes 40 seconds.
- Levers: no generation, one forward pass, taken; batching declined, since Part 139's forced-prefix path is one row at a time and the total is small.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 143.1 | After a forced "I'm", the median Set R unit under prompt-only puts at least 10 times the plain model's probability on " sorry". | medium | Claude |
| 143.2 | After a forced "I'm", the median Set R unit under answer-only puts at least 10 times the plain model's probability on " sorry". | medium | Claude |
| 143.3 | After a forced "I", the median Set R unit under answer-only puts less on "'m" than the median unit under prompt-only. | low | Claude |
| 143.4 | Under full, the median Set R probability of " sorry" after "I'm" is at least twice the larger of the prompt-only and answer-only medians. | medium | Claude |
| 143.5 | Under full, Set S's median probability of " sorry" after "I'm" is at most a tenth of Set R's. | high | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 143 (registered 2026-10-31, PART143-DESIGN.md): which side of the push moves which word of \"I'm sorry\"?".

## Reading rules

part143/score143.py computes every number. Probabilities are reported as medians with the middle 50%, by set and by neuron or random. Set R is selected on the full arm's outcome, so the full arm's values describe the selected units and say nothing about how common this is.
A prompt-only push leaves changed cached states, so it is not a pure test of reading. This part says where the probability moves. It does not say why.

## Limits

One model, one dose, one question, two layers. Forced prefixes skip the answer the model would have written. Bf16.
