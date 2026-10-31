# Part 147: does the pushed ranking match Base on the right question?

Procedure: v1
Registered 2026-10-31 09:50 CT by Claude (Opus 5.5), before Part 146's results were read and before any data for this part. Design from Alden, council topic overnight-ideas-2026-10-31, message 12. The commit that adds this file is the registration.

## Question

Part 145 showed a push moving Instruct's next-word ranking toward Base's on two questions. A ranking can look like Base's because every question collapses to one generic "I'm" profile. If the pushed row keeps its question, it should match Base on the same question better than Base on a different one. Rank resemblance is also not probability closeness, so this part measures both, unit by unit. The name for the pattern, if it stands, is Alden's: Base-like conditional output resemblance under perturbation, under Instruct's template, with the mechanism open.

## Prior art and existing data

- Closest published work: see PART146-DESIGN.md; no new search, because this part reads Part 146's stored rows and asks nothing new of the literature.
- Our own data: results/part146's matrices on the job's disk, which this part reads, not re-collects. Part 146's README, if read first, does not change this registration: it is written before those results are read.
- Reviewed before registration by Alden, whose message proposes it.

## Method

No model is run. Inputs: Part 146's Set F full-arm matrices on four questions, its plain Instruct rows, and its Base chat rows. Word support: one fixed set, tokens with plain probability at least 1e-5 in Instruct or Base chat on any of the four questions. Rows are full 100,352-token log-probabilities for distances, and the fixed support for ranks.

Cross-question read: for each question q, Spearman of the median pushed row at q with Base at q, against the mean of its Spearman with Base at each other question. Question-centered read: subtract from each row the mean of that row's four questions, for pushed and for Base separately, then the Spearman of pushed at q with Base at q.

Distance read: per unit, Jensen-Shannon divergence between the unit's pushed distribution and Base at q, against the same for the plain Instruct row. Distributions are the stored ones, each already normalized. Reported by neuron and random, by layer, with the share of units closer to Base than plain Instruct is.

## Controls

Applied from STANDARD-CONTROLS.md: matched random directions, unpushed baselines. Wrong-question references are the control for a generic profile. Not used: a do-nothing hook, which Part 146 ran; no gate needs a model. Gate: Part 146's own gates passed, and every row loaded is finite. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a Spearman difference of 0.05 and a 10-point difference in shares.
- Total: no generation; 4 questions x 256 units of stored rows, about 5 minutes on the CPU.
- Pilot: not needed, no GPU and no generations.

## Optimization

- Bound resource: CPU
- Measured: n/a: under 5 minutes.
- Levers: stored rows, no model load, taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 147.1 | On at least 3 of 4 questions, the median pushed row's Spearman with Base at the same question exceeds its mean Spearman with Base at the other questions by at least 0.05. | medium | Claude |
| 147.2 | The question-centered Spearman of pushed with Base, averaged across questions, is at least 0.3. | medium | Claude |
| 147.3 | On at least 3 of 4 questions, at least 90% of units are closer to Base in Jensen-Shannon divergence than plain Instruct is. | medium | Claude |
| 147.4 | The share in 147.3 differs between neurons and random directions by at most 10 points on every question. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 147 (registered 2026-10-31, PART147-DESIGN.md): does the pushed ranking match Base on the right question?".

## Reading rules

part147/score147.py computes every number. Verdicts use point estimates. Four similar self-questions give limited power on 147.1 and 147.2, and they count as four, not as thousands of tokens. If these hold, the wording stays "resemblance under Instruct's template"; it does not show any computation was restored. If 147.1 fails, the pattern is a generic profile, not a question-keeping one.

## Limits

Four questions, all about the self, one position, one forced prefix, one dose, Base under a frame it never trained on, bf16. Empty, garbled and no-prompt rows in Part 146 are examples of a damaged prompt, not every kind.
