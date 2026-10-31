# Does a push single out “ sorry” after “Who are you?”

For OLMo 2 0425-1B Instruct chat, after a forced “I'm”, the push raises “ sorry” above its temperature-matched level. On “Who are you?”, its change ranks weakly against plain log-probability (Spearman −0.170), while its residual after a straight-line fit is at the 100th percentile of the eligible tokens.

A word's change is the median, over 256 units, of pushed minus plain log-probability. Temperature matching fits a temperature so the plain distribution's top-word probability matches the pushed top-word probability, then compares the resulting “ sorry” log-odds with the observed pushed log-odds. On “Who are you?”, this lowers the top token “ an” from 0.94 to 0.27. The partial correlation asks whether words with a larger Base-chat versus Instruct-chat plain log-probability gap also have larger changes, after controlling for Instruct-chat plain log-probability.

All registered gates passed. Gate 1 passed on 144 noop rows: each of 12 cell-question combinations had 12 rows, and both “ sorry” and the plain top token matched plain within 1e-4 in log-probability in every row (288 of 288 token comparisons). Gate 2 matched Part 144 within 0.01 for 253 of 256 units (98.8%), above its 95% requirement. Gate 3 found all 3,072 pushed unit distributions and every plain reference vector finite. The noop rows are on the job's disk, not in this repository, so Gate 1 cannot be rechecked from the repository files.

## Predictions

Wording and confidence are from `PART145-DESIGN.md`. Verdicts use the registered point-estimate rules. Here, “failed” includes the scorer's “does not hold.”

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---:|---|
| 145.1 | Across words, the Spearman correlation of change with plain log-probability is at most -0.6: unlikely words rise most. (medium) | Spearman −0.170 | failed |
| 145.2 | After a straight-line fit of change on plain log-probability across words, “ sorry”'s residual is in the top 5% of words. (medium) | 100.0th percentile | held |
| 145.3 | The median unit's observed log-odds of “ sorry” exceeds its temperature-matched value by at least 1.0. (low) | 2.871 log-odds; 95% bootstrap interval [2.500, 3.264] | held |
| 145.4 | Across words, the partial Spearman correlation of change with (base-chat plain log-probability minus instruct-chat plain log-probability), controlling for instruct-chat plain log-probability, is at least 0.3. (medium) | 0.491 | held |
| 145.5 | Across words, the Spearman correlation of the median pushed log-probability with base-chat's plain log-probability exceeds that of instruct-chat's plain log-probability by at least 0.05. (medium) | Difference 0.141 | held |
| 145.6 | The per-word change computed from neurons and from random directions correlates at Spearman 0.9 or higher. (high) | Spearman 0.987 | held |
| 145.7 | On “Tell me about yourself.”, 145.2 gets the same verdict. (medium) | “ sorry” is at the 100.0th percentile, as on “Who are you?” | held |

The first score for 145.7 was wrong. It used the “ sorry” position from “Who are you?” on the second question's word list. Checking the second-question token list exposed the mismatch. After correcting the score, 145.7 is held. Six of the seven predictions are held.

The eligible word set for the registered predictions is only 106 tokens on “Who are you?” because Instruct is so sure of its next token. “Top 5%” therefore means about five tokens. Prediction 145.2 ranks “ sorry” against those 106 tokens. The second question has 1,128 eligible tokens.

## “Who are you?” across cells

Probabilities are for the next token after the forced “I'm”. The temperature interval is the 95% bootstrap interval over units. Median temperature is the median fitted temperature.

| Cell | Plain P(“ sorry”) | Median pushed P(“ sorry”) | Tokens | 145.1 Spearman | “ sorry” residual percentile | Temperature gap [95% interval] | Median T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base plain | 0.289% | 2.073% | 106 | 0.102 | 98.1% | 2.030 [1.895, 2.259] | 1.110 |
| Base chat | 0.485% | 3.849% | 106 | 0.091 | 99.1% | 2.214 [2.006, 2.395] | 1.118 |
| SFT chat | 0.576% | 14.093% | 106 | −0.271 | 100.0% | 2.702 [2.324, 3.041] | 1.697 |
| DPO chat | 0.0529% | 4.468% | 106 | −0.190 | 100.0% | 2.691 [2.282, 3.098] | 1.731 |
| Instruct chat | 0.0239% | 4.169% | 106 | −0.170 | 100.0% | 2.871 [2.500, 3.264] | 1.796 |
| Instruct plain | 5.461% | 8.959% | 106 | 0.046 | 87.7% | 1.585 [1.373, 1.912] | 1.564 |

The 145.4 and 145.5 columns in `tables.md` repeat the same estimates computed on Instruct-chat. They are not cell values. Those estimates are 0.491 and 0.141. They are empty for the second question because these predictions were registered only for “Who are you?”.

## “Tell me about yourself?” across cells

The residual percentile is for “ sorry” in each cell's straight-line fit. These values also show that 145.7 gets the same verdict as 145.2 on “Who are you?”.

| Cell | Plain P(“ sorry”) | Median pushed P(“ sorry”) | Tokens | 145.1 Spearman | “ sorry” residual percentile | Temperature gap | Median T |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base plain | 0.289% | 1.625% | 1,128 | −0.267 | 99.8% | 1.596 | 1.216 |
| Base chat | 0.970% | 4.407% | 1,128 | −0.094 | 99.1% | 1.796 | 1.340 |
| SFT chat | 0.199% | 17.746% | 1,128 | −0.314 | 100.0% | 7.596 | 5.907 |
| DPO chat | 0.212% | 5.816% | 1,128 | −0.286 | 100.0% | 6.783 | 6.859 |
| Instruct chat | 0.245% | 4.968% | 1,128 | −0.187 | 100.0% | 6.707 | 7.724 |
| Instruct plain | 0.0109% | 1.444% | 1,128 | −0.479 | 99.9% | 3.865 | 1.407 |

The table below gives the top five risers and fallers at the 1e-5 floor on each question. Probabilities are Instruct-chat plain, Base-chat plain, and the Instruct-chat median pushed probability, in that order. Values are rounded for display.

| Question | Rank | Token | Instruct plain P | Base-chat plain P | Instruct pushed P |
|---|---|---|---:|---:|---:|
| Who are you? | Riser | ` sorry` | 0.024% | 0.48% | 4.2% |
| Who are you? | Riser | ` not` | 0.019% | 3.2% | 1.8% |
| Who are you? | Riser | ` ready` | 0.0018% | 0.021% | 0.13% |
| Who are you? | Riser | ` here` | 0.11% | 0.96% | 6.5% |
| Who are you? | Riser | ` really` | 0.0012% | 0.13% | 0.063% |
| Who are you? | Faller | ` O` | 0.44% | 0.088% | 0.013% |
| Who are you? | Faller | ` Open` | 0.083% | 0.0032% | 0.0057% |
| Who are you? | Faller | ` Ot` | 0.0069% | 0.0061% | 0.00068% |
| Who are you? | Faller | ` Ple` | 0.0020% | 0.0014% | 0.00021% |
| Who are you? | Faller | ` Os` | 0.0055% | 0.0058% | 0.00057% |
| Tell me about yourself. | Riser | ` unable` | 0.0026% | 0.0060% | 0.055% |
| Tell me about yourself. | Riser | ` sorry` | 0.25% | 0.97% | 5.0% |
| Tell me about yourself. | Riser | ` going` | 0.0027% | 0.69% | 0.052% |
| Tell me about yourself. | Riser | ` here` | 0.19% | 0.88% | 3.2% |
| Tell me about yourself. | Riser | ` more` | 0.0023% | 0.13% | 0.033% |
| Tell me about yourself. | Faller | ` O` | 60% | 0.017% | 0.0024% |
| Tell me about yourself. | Faller | ` Dialog` | 0.039% | 0.0000060% | 0.0000095% |
| Tell me about yourself. | Faller | ` Language` | 0.038% | 0.000040% | 0.000010% |
| Tell me about yourself. | Faller | ` Conversation` | 0.0031% | 0.0000033% | 0.0000011% |
| Tell me about yourself. | Faller | ` Milo` | 0.19% | 0.0049% | 0.000076% |

On the second question, raw change ranks ` unable` (3.03) just above ` sorry` (3.01). On “Who are you?”, ` O` falls from about 60% to about 0.002%. Identity-looking tokens also fall, including ` Assistant`, ` Iris`, ` Open`, ` Milo`, ` Dialog`, ` Language`, ` Conversation`, ` Talk`, and ` Speak`. These are medians over 256 units at one position.

The numbers show a weak flattening relation (−0.170, against a uniform-temperature null of −1), a return pattern (0.491 partial correlation and 0.141 correlation difference), and a specific “ sorry” signal (100.0th percentile residual and a 2.871 log-odds temperature gap).

The 0.987 Spearman means random directions and neurons change the words alike by rank.

The design says a held 145.4 or 145.5 is a pattern across words at one position, not a mechanism, and says nothing about why a word was lower. Part 145 does not test a mechanism. Rank correlations are unaffected by temperature, so pure flattening cannot produce the change in rank correlation. The untested alternative is that the push loses the prompt, and the word ranking after a bare “I'm” may look like Base without any return to pretraining. Part 146 tests that.

## Method and limits

The run follows `PART145-DESIGN.md`, registered 2026-10-31 at 08:18 CT: Set F's 256 units, six cells, two questions, one full push at 0.8 times each checkpoint and layer's own r_L, and a forced “I'm” prefix. It stores the full next-token distribution at one position. The design counted 3,072 pushed sequences. The job timed 3,228 sequences: 3,072 pushes, 144 noop rows, and 12 plain reference sequences. The six timing files sum to 496.69 seconds (8 minutes 17 seconds), against the design's estimate of about 15 minutes on the A770 plus six model loads.

The closest published work named in the design is Malla, Choi and Choi (arXiv:2609.06951). They report that pushes relax chat models toward a small set of default behaviors using generated text. Part 145 reads next-token probabilities across the vocabulary at one position. The design also cites Moskvoretskii et al. (arXiv:2605.13329) on post-training suppression of directions and traits, and Qi et al. (ICLR 2025), as reported by Muse, on differences concentrated in the first tokens.

The forced prefix skips what each model would have said. Base-chat is read through Instruct's template, not its own natural frame. One model family, one dose, two questions, one position, two layers, and bf16 limit the result. A token is not a word. Word-level tests treat tokens as a sample, and no p-value is reported on them because tokens are not independent. Low-probability token log-probabilities are noisy near the bf16 softmax floor; the registered set therefore stops at 1e-5.

## Files

- `score.json` and `tables.md`: registered gates, predictions, and cell summaries from `part145/score145.py`.
- `words-*.json`: per-token median changes and unit quartiles from `part145/score145.py`.
- `explore.json`: unregistered rank correlations and top risers and fallers from `results/part145/explore.py`.
- `matrix-*.npy`: full-vocabulary pushed distributions. These remain on the job's disk.
- `plain-*.npy`: full-vocabulary plain reference distributions. These remain on the job's disk.
- `noop-*.jsonl`: noop comparisons for Gate 1. These remain on the job's disk, so Gate 1 cannot be rechecked from the repository.
- `index-*.json` and `timing-*.json`: token IDs, word sets, unit indexes, and per-cell runtimes. These remain on the job's disk.
- `PART145-DESIGN.md`: registration, predictions, reading rules, and limits.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; six findings folded, one of them a scorer bug that changed 145.7's verdict.
