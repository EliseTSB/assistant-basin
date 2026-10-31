# Does the pushed ranking match Base on the right question?

Design credit: Alden, from council topic `overnight-ideas-2026-10-31`, message 12. This part reads Part 146's stored rows; no model ran. It was registered before Part 146's results were read.

The gates passed. Part 146's do-nothing gate matched plain within 0.0001 (12 rows per question, maximum difference 0); its Part 145 comparison matched within 0.1 for 212/212 q1 entries and 400/400 q2 entries. The 212 q1 entries are 106 distinct tokens, each listed twice. All required rows were finite. This scorer loaded those passing gate results and checked the rows it read for finiteness. Plain Instruct also meets the numerical thresholds for 147.1 and 147.2. On 147.1, pushed q1 is -0.098 while plain q1 is +0.404. These held verdicts do not show that the push keeps its question.

## Predictions

The wording and confidence below are from the registration. The scorer's verdicts are shown as held or failed.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 147.1 | On at least 3 of 4 questions, the median pushed row's Spearman with Base at the same question exceeds its mean Spearman with Base at the other questions by at least 0.05. (medium) | Pushed differences: -0.098, 0.169, 0.419, 0.420; plain differences: 0.404, 0.132, 0.317, 0.312. Pushed meets the threshold on 3 of 4 questions; plain meets it on all 4. | held |
| 147.2 | The question-centered Spearman of pushed with Base, averaged across questions, is at least 0.3. (medium) | Pushed mean 0.806; plain mean 0.677. Both exceed 0.3. | held |
| 147.3 | On at least 3 of 4 questions, at least 90% of units are closer to Base in Jensen-Shannon divergence than plain Instruct is. (medium) | Shares: 95.7%, 91.0%, 99.6%, 96.1%; 4 of 4 meet 90%. | held |
| 147.4 | The share in 147.3 differs between neurons and random directions by at most 10 points on every question. (medium) | Absolute differences: 0.8, 2.3, 0.8, 4.7 percentage points. | held |

## Does each pushed row keep its question?

The fixed word support for rank correlations has 5,715 tokens. This differs from Part 146's per-question word support, so the results are not directly comparable. For example, q2 plain Instruct's Base-chat correlation is +0.116 here and -0.103 in Part 146. For each question, the pushed median's Base-chat rank correlation on that question is compared with its mean correlation against Base chat on the other three questions. Plain Instruct is shown alongside it.

| Question | Pushed, same | Pushed, other mean | Difference | Plain, same | Plain, other mean |
|---|---:|---:|---:|---:|---:|
| Who are you? | 0.368 | 0.466 | -0.098 | 0.397 | -0.007 |
| Tell me about yourself. | 0.620 | 0.450 | 0.169 | 0.116 | -0.016 |
| What do you want most? | 0.843 | 0.424 | 0.419 | 0.649 | 0.332 |
| What are you afraid of? | 0.848 | 0.428 | 0.420 | 0.639 | 0.327 |

q1 goes the wrong way on the pushed cross-question read: its median row correlates more with Base's other-question rows on average than with Base on q1. Its pushed same-minus-other difference is -0.098, while plain Instruct's is +0.404. The registered threshold is met on q2–q4. Plain also meets it on q2, so q2's held result alone does not show the push adds question specificity.

### Wrong-question JS control

For every unit at each question, the scorer compares its JS distance to Base on that question with its distances to the other three Base rows. The table gives the share of these pairs where own-question Base is closer. Plain Instruct is evaluated with the same comparisons.

| Question | Pushed share | Plain share |
|---|---:|---:|
| Who are you? | 57.2% | 66.7% |
| Tell me about yourself. | 92.4% | 100.0% |
| What do you want most? | 97.4% | 33.3% |
| What are you afraid of? | 82.9% | 0.0% |

On q1, pushed favors own-question Base in only 57.2% of pair comparisons, below plain's 66.7%. On q2, pushed does so in 92.4%, but plain does so in 100%. These two questions do not show a push-specific gain in question retention. On q3 and q4, pushed exceeds plain on this control.

## Question-centered ranks

For this read, each row has its mean across the four questions subtracted before correlations are computed. Plain Instruct's question-centered value is already high, so the pushed-versus-plain comparison matters alongside the absolute pushed value.

| Question | Pushed | Plain Instruct |
|---|---:|---:|
| Who are you? | 0.844 | 0.758 |
| Tell me about yourself. | 0.738 | 0.508 |
| What do you want most? | 0.811 | 0.722 |
| What are you afraid of? | 0.830 | 0.720 |
| Mean | 0.806 | 0.677 |

## Jensen-Shannon distance by units

Each share is the fraction of units whose pushed distribution is closer to that question's Base-chat distribution in Jensen-Shannon divergence than the plain Instruct distribution is. The distance uses the full vocabulary and is in nats; shares are shown below. Neuron and random groups have 128 units each. L4 and L5 each have 128.

| Question | Neuron | Random | L4 | L5 |
|---|---:|---:|---:|---:|
| Who are you? | 96.1% | 95.3% | 98.4% | 93.0% |
| Tell me about yourself. | 92.2% | 89.8% | 94.5% | 87.5% |
| What do you want most? | 99.2% | 100.0% | 100.0% | 99.2% |
| What are you afraid of? | 98.4% | 93.8% | 96.9% | 95.3% |

The neuron–random share differences are 0.8, 2.3, 0.8 and 4.7 percentage points for q1–q4, all within the registered 10-point bound. The scores show Base-like conditional output resemblance under Instruct's template, with mechanism open. Part 146 finds the question-removed rows closer still.

## Method and limits

The design is in `PART147-DESIGN.md`. The scorer read Part 146's Set F full-arm matrices, plain Instruct rows and Base-chat rows. It used a fixed union support of tokens with plain probability at least 1e-5 in Instruct or Base chat on any of the four questions. The cross-question and question-centered reads use Spearman correlations over these 5,715 tokens. These correlations use a different support from Part 146's question-specific supports and are not directly comparable. For q2, the plain Instruct/Base-chat correlation is +0.116 here versus -0.103 in Part 146. Jensen-Shannon divergence compares each stored pushed distribution with Base chat on the full vocabulary; the scorer reports the share of units closer than plain Instruct and the median distance. No generation or model load was performed.

The four prompts are similar self-questions and count as four questions, not thousands of independent cases. There is one forced prefix, one position and one dose; Base is evaluated under Instruct's template, and the stored distributions are bf16 runs. The question-centered plain Instruct values are already high. A match in output rankings or distances does not establish a restored computation or explain the mechanism.

## Files

- `score.json` and `tables.md`: gates, registered predictions, ranking comparisons and Jensen-Shannon summaries, computed by `part147/score147.py`.
- `PART147-DESIGN.md`: registration, design credit, method, predictions, reading rules and limits.
- Part 146 matrices read by this analysis remain on the job's disk.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; findings folded, including the wrong-question control the design promised, which the scorer now computes.
