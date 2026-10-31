# Does training move the apology's baseline, the push's effect, or both?

For OLMo 2 0425-1B, across six cells, one dose, the question “Who are you?”, and forced prefixes, Set F's median full-arm shift on P(" sorry" after "I'm") is positive in every cell and ranges from 0.533 to 5.203 log-odds.

A shift is the pushed unit's log-odds minus the plain model's log-odds in that cell. Prompt-only pushes during the prompt, answer-only pushes during the forced answer prefix, and full pushes during both. The plain frame is `Question: Who are you?\nAnswer:`; the chat frame uses the chat template.

All three registered gates passed. Gate 1's noop matched plain in 216 of 216 comparisons; every difference was zero. Gate 2 matched Part 143 on 520 of 520 checks (100%), above its 90% requirement. Gate 3 passed 2,892 of 2,892 tokenization and answer-only first-read checks.

Base has no chat-template training. Its chat cell feeds Base the text of Instruct's template, rendered with the Instruct tokenizer. The snapshots' the vocabulary and merge files are byte-identical, and their added tokens are equal. The first pilot file exposed Base plain's P(" sorry") value, 0.29%, before scoring. This was the value in prediction 144.1, registered earlier.

The pilot's first launch stopped at base-chat, so only base-plain had run. After Amendment 1, the remaining cells ran. The pilot gates passed on 12 Set R units per cell: Gate 1 had 216 of 216 noop comparisons pass, Gate 2 matched Part 143 on 48 of 48 reads, and Gate 3 passed all first-token and tokenization checks. The launch log records the rerun.

## Predictions

Results use the registered point-estimate rules. Confidence is from the registration. Verdicts are held or failed.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 144.1 | The plain model's P(" sorry" after "I'm") on Base in the plain frame is at least its value on Instruct in the chat frame. (medium) | Base plain 0.002888 (0.2888%); Instruct chat 0.0002391 (0.02391%). | held |
| 144.2 | The median full-arm shift on Base in the plain frame is at most half the median on Instruct in the chat frame. (medium) | 1.980 versus half of 5.203, or 2.601. | held |
| 144.3 | The median full-arm shift on Base in the chat frame is at most half the median on Instruct in the chat frame. (low) | 2.114 versus half of 5.203, or 2.601. | held |
| 144.4 | The median full-arm shifts on SFT, DPO and Instruct, all chat, lie within 1.0 of each other. (medium) | Medians are 3.344, 4.481 and 5.203; the range is 1.859. | failed |
| 144.5 | Across units, the Spearman correlation of full-arm shift between SFT and Instruct is at least 0.7, and between Base in the plain frame and Instruct is under 0.4. (medium) | Correlations are 0.948 and 0.419, respectively; the second exceeds 0.4. | failed |
| 144.6 | From Base chat to SFT the median full-arm shift changes by more, in absolute value, than the plain model's log-odds does. (medium) | Shift changes by 1.230; plain log-odds changes by 0.173. | held |
| 144.7 | On Set R, in each of SFT, DPO and Instruct, the median of full minus (prompt-only + answer-only - plain), in log-odds, has magnitude at most 0.5. (medium) | SFT −0.382; DPO 0.066; Instruct −0.024. | held |
| 144.8 | In every cell, the median full-arm shifts of Set F's neurons and random directions differ by at most 0.5. (medium) | Instruct plain differs by 0.514 log-odds, exceeding 0.5 by 0.014. | failed |
| 144.9 | The median full-arm shift on Instruct in the plain frame is at most half the median on Instruct in the chat frame. (low) | 0.533 versus half of 5.203, or 2.601. | held |

## What moves in the six cells?

The plain columns give the next-token probabilities before a push. The other values are Set F's median log-odds shifts. Set F has 256 units, with neurons and matched random directions together. Sorry intervals are 95% bootstrap intervals over units, with 10,000 resamples and seed 144.

| Cell | Frame | Plain P("I" or " I") | Plain P("'m") | Plain P(" sorry") | Full shift: P("I" or " I") (95% interval) | Full shift: P("'m") (95% interval) | Full shift: P(" sorry") (95% interval) | Prompt-only P(" sorry") shift (95% interval) | Answer-only P(" sorry") shift (95% interval) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | plain | 41.34% | 30.15% | 0.2888% | −1.325 (−1.414 to −1.245) | −0.785 (−0.960 to −0.702) | 1.980 (1.845 to 2.204) | 2.056 (1.922 to 2.176) | 0.756 (0.687 to 0.857) |
| Base | chat | 10.33% | 43.11% | 0.4847% | −1.284 (−1.410 to −1.161) | −1.207 (−1.305 to −1.091) | 2.114 (1.963 to 2.300) | 2.562 (2.453 to 2.693) | 1.092 (0.983 to 1.237) |
| SFT | chat | 82.92% | 19.13% | 0.5756% | −1.081 (−1.240 to −0.953) | 1.124 (0.945 to 1.273) | 3.344 (3.024 to 3.586) | 2.856 (2.609 to 3.073) | 0.493 (0.233 to 0.645) |
| DPO | chat | 63.90% | 15.40% | 0.05291% | −0.410 (−0.536 to −0.351) | 0.890 (0.723 to 1.032) | 4.481 (4.216 to 4.888) | 3.808 (3.445 to 4.034) | 0.605 (0.443 to 0.886) |
| Instruct | chat | 65.93% | 13.89% | 0.02391% | −0.611 (−0.694 to −0.533) | 0.876 (0.793 to 1.009) | 5.203 (4.786 to 5.490) | 4.394 (3.904 to 4.624) | 0.778 (0.683 to 1.073) |
| Instruct | plain | 42.83% | 18.34% | 5.460% | −1.868 (−2.113 to −1.602) | −0.060 (−0.165 to 0.047) | 0.533 (0.308 to 0.746) | 0.854 (0.716 to 1.072) | 0.997 (0.827 to 1.094) |

Intervals are 95% percentile bootstrap intervals over units, with 10,000 resamples and seed 144. The full shifts use the 256 Set F units. Prompt-only and answer-only Sorry intervals use the same units.

Here are the Sorry shifts split by unit type. Set F has 128 neurons and 128 random directions per cell.

| Cell | Neuron median (95% interval) | Random median (95% interval) |
|---|---:|---:|
| Base plain | 1.867 (1.605 to 2.246) | 2.046 (1.874 to 2.263) |
| Base chat | 1.998 (1.744 to 2.241) | 2.225 (1.981 to 2.442) |
| SFT chat | 3.141 (2.426 to 3.702) | 3.442 (3.115 to 3.781) |
| DPO chat | 4.344 (3.686 to 5.023) | 4.578 (4.319 to 4.963) |
| Instruct chat | 5.062 (4.392 to 5.587) | 5.317 (4.869 to 5.642) |
| Instruct plain | 0.245 (0.016 to 0.557) | 0.759 (0.504 to 1.141) |

## Found after the registered reading, not registered

These values were examined after the predictions were scored. They are untested and need their own registered part.

| Cell | Plain log-odds of " sorry" | Median full shift | Pushed level (plain log-odds + median shift) |
|---|---:|---:|---:|
| Base plain | −5.844 | 1.980 | −3.864 |
| Base chat | −5.325 | 2.114 | −3.211 |
| SFT chat | −5.152 | 3.344 | −1.808 |
| DPO chat | −7.544 | 4.481 | −3.063 |
| Instruct chat | −8.338 | 5.203 | −3.136 |
| Instruct plain | −2.851 | 0.533 | −2.319 |

This trend is not monotone. Base plain to Base chat changes the plain log-odds from −5.844 to −5.325 while the shift rises from 1.980 to 2.114. Instruct chat to Instruct plain changes the log-odds from −8.338 to −2.851 while the shift falls from 5.203 to 0.533. Untested. SFT's pushed level is 1.255 above DPO and 1.328 above Instruct, both greater than the design's smallest effect of 1.0. Base chat, DPO and Instruct differ pairwise by 0.148, 0.075 and 0.073, so they are within about 0.15.

The median shift on P("'m") is negative in Base plain (−0.785) and Base chat (−1.207). It is positive in the SFT, DPO and Instruct chat cells (1.124, 0.890 and 0.876). Instruct plain is −0.060.

The plain model's top five tokens after forced "I'm" come from each file's stored `top5_sorry` field:

| Cell | Plain top five tokens and probabilities |
|---|---|
| Base plain | ` a` 19.62%; ` the` 9.56%; ` an` 4.24%; ` just` 3.41%; ` me` 3.10% |
| Base chat | ` a` 13.73%; ` <` 10.04%; ` the` 8.86%; ` an` 6.69%; ` your` 3.70% |
| SFT chat | ` an` 85.43%; ` O` 7.01%; ` a` 2.42%; ` just` 1.38%; ` sorry` 0.576% |
| DPO chat | ` an` 92.73%; ` a` 3.17%; ` just` 1.92%; ` O` 0.708%; ` here` 0.116% |
| Instruct chat | ` an` 94.42%; ` a` 2.52%; ` just` 1.53%; ` O` 0.437%; ` here` 0.107% |
| Instruct plain | ` a` 27.73%; ` just` 22.99%; ` an` 17.90%; ` sorry` 5.461%; ` not` 2.01% |

The plain probability on ` an` is 85.43% for SFT, 92.73% for DPO and 94.42% for Instruct in chat. For Instruct, the plain frame has P(" sorry") of 5.460% and a median full shift of 0.533. Its chat frame has P(" sorry") of 0.02391% and a median full shift of 5.203.

Here is each cell's median r_L by layer, computed from the unit rows. The absolute push uses 0.8 times that cell and layer's r_L.

| Cell | Layer 4 r_L | Layer 5 r_L |
|---|---:|---:|
| Base plain | 11.204 | 13.066 |
| Base chat | 11.617 | 12.916 |
| SFT chat | 8.755 | 10.102 |
| DPO chat | 9.008 | 10.112 |
| Instruct chat | 9.048 | 10.064 |
| Instruct plain | 10.121 | 10.977 |

Absolute push sizes differ by up to about a third between cells because each cell uses its own r_L. A shift is at the same relative dose, not the same absolute push.

## Reading rules and limits

The forced prefixes skip the answer each model would have written. A row is not an apology. It is the probability of one word after two forced words. Set R and Set S were selected on Instruct, so they cannot show how common anything is. Equal levels would not show the same mechanism.

The run used OLMo 2 0425-1B, bf16, two layers, one question, and one dose: α = 0.8 times each checkpoint's own r_L. No text was generated. There was no sampling. Each cell's timing file records 1,459 rows. The six runtimes sum to 1,580.8 seconds, or 26 minutes 21 seconds. The design estimated about 15 minutes on the A770, plus six model loads. Top-five extraction ran on the CPU after the pilot because top-k over the vocabulary was slow on the A770.

The closest published work named in the design is Labunets (arXiv:2608.25390). It follows a refusal direction across this OLMo 2 0425-1B family and finds it functional by SFT, with larger changes after. It fits a direction; this part measures unit and matched-random pushes and separates the plain baseline from the push effect.

One model family, one dose, one question, and two layers limit this result. The dose equalizes written size, not downstream sensitivity. A neuron's row can change across checkpoints, and the same index need not have the same function. Instruct's RLVR stage is not separated from DPO.

## Files

- `results/part144/score.json` and `results/part144/tables.md`: registered scores and summaries from `part144/score144.py`.
- `results/part144/<cell>.jsonl`: measured rows and stored top-five values from `part144/run144.py`.
- `results/part144/timing-<cell>.json`: per-cell runtimes.
- `PART144-DESIGN.md`: registration, predictions, reading rules, limits, and Amendment 1.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; seven findings, folded.
