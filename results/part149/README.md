# Is “ sorry” what Instruct says when it cannot read the prompt?

This table shows OLMo 2 0425-1B Instruct, SFT, and Base on 24 questions with real, damaged, and empty prompts, measuring the forced continuation “I'm” without generating an answer.

All five gates passed on the rerun. The readout is P(“ sorry” | “I'm”), unless a row says otherwise.

## Run correction

Gate 5 failed on the first run for one row: q10, shuf-1/3, seed 149903. The draw selected both “12” tokens in “What is 12 times 12?”, so the prompt did not change. Amendment 1 changed the shuffle rule to redraw when the whole prompt stays unchanged. It fixed that row before any prediction was read. The rerun prompt was “What si 12 times 12?”.

The three unpushed stages were rerun in full. Comparing `run1/` to the rerun shows that row changed and no other. Every other stored float was identical: 0 differences across the other 1,211 rows in Base and in SFT and the other 4,331 rows in Instruct. The pushed rows use real questions and were not rerun. Digits do not shuffle, so the multiplication question receives less damage than questions without digits.

## Predictions

Wording and confidence are from the registration. Verdicts use the registered point-estimate rules.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 149.1 | On Part 146's stored rows, across Set F's 256 units within a question, log P(" sorry") and similarity to the empty row have a partial Spearman of at least 0.3 controlling for Jensen-Shannon distance from plain, on at least 3 of 4 questions. (low) | 0.378, 0.443, 0.373, and 0.551 on the four questions. | held |
| 149.2 | On Instruct, the median over 24 questions of P(" sorry") at shuf-all is at least 10 times the median at real. (high) | 0.822 / 0.121 = 6.79×. | failed |
| 149.3 | On Instruct, log P(" sorry") across real, shuf-1/3, shuf-2/3 and shuf-all has Spearman at least 0.8 with the damage order on at least 18 of 24 questions. (medium) | 18 of 24 meet the threshold after rounding each rho to nine places. | held |
| 149.4 | Across the 24 questions, the prompt-only arm's median log P(" sorry") over its 64 units has Spearman at least 0.5 with Instruct's log P(" sorry") at shuf-all. (medium) | ρ = 0.431; 95% interval −0.026 to 0.765. | failed |
| 149.5 | The same for the full arm, Spearman at least 0.5. (medium) | ρ = 0.482; 95% interval 0.078 to 0.763. | failed |
| 149.6 | On Instruct, the median over questions of P(" sorry") in the words arm is at least a third of the median in the letters arm. (low) | Words 0.472; letters 0.829; one third of letters is 0.276. | held |
| 149.7 | At shuf-all, the median over questions of P(" sorry") on Base is at most a fifth of the median on SFT and at most a fifth of the median on Instruct. (medium) | Base 0.0791; SFT 0.919; Instruct 0.822, medians across the 24 question values. | held |
| 149.8 | On at least 18 of 24 questions, the full arm's median P("I") x P("'m" given "I") is at least 5 times Instruct's value at shuf-all. (medium) | 2 of 24 questions. | failed |

The 149.3 correction was made after seeing the scored output, and the code committed before it gives 15 of 24, failed. The first scorer run said failed, with 15 of 24. With four damage ranks, Spearman values move in exact steps of 0.2. Three questions had rho exactly 0.8, stored in binary floating point as 0.7999999999999999. The registered wording says “at least 0.8.” The scorer now compares after rounding to nine places. The count changed from 15 to 18, and the verdict changed from failed to held.

The 149.2 ratio cannot have held: the real median is 0.121, so ten times that value is 1.21, above the maximum probability of 1. I set the ratio from “Who are you?”, where the real value was 0.024%, without knowing the median across 24 questions. The registered reading rule applies: because 149.2 failed, the idea that damage alone brings “ sorry” is withdrawn, and 71.8% stands as a fact about the empty prompt. The prediction's median is across the 24 question-level exponentiated mean log-probabilities: 0.12 at real and about 0.82 at shuf-all. The table below gives medians across stored seed rows: 0.121 and 0.846. Prediction 149.3 held. So the withdrawal rests on a bar I set badly, which could not be met, and the rise in the table stands beside it.

Both 149.4 and 149.5 failed, so the registered reading that a pushed apology follows a damaged-prompt apology is withdrawn for these questions. Their values with question baseline removed were 0.175 overall for prompt-only and 0.326 for full. For prompt-only, the neuron split was 0.150 and the random-direction split 0.202. For full, neurons were 0.184 and random directions 0.301. Their bootstrap intervals remain wide:

| Arm and split | Spearman after baseline removal | 95% interval |
|---|---:|---:|
| Prompt-only, all | 0.175 | −0.148 to 0.467 |
| Prompt-only, neurons | 0.150 | −0.189 to 0.502 |
| Prompt-only, random directions | 0.202 | −0.133 to 0.492 |
| Full, all | 0.326 | −0.068 to 0.582 |
| Full, neurons | 0.184 | −0.269 to 0.530 |
| Full, random directions | 0.301 | −0.076 to 0.568 |

Neurons and random directions gave similar values. This does not separate them.

Prediction 149.8 also failed. The full-arm opener measure reached five times the Instruct shuf-all value on only 2 of 24 questions. Part 139 Stage D's low unforced apology-opener rates are read through the same three links: the forced-prefix apology readout, the prompt-damage result, and the pushed versus damaged question ranking.

## Medians by stage and damage arm

These are medians across stored rows, rounded to three significant figures. `P(m|I)` means P("'m" | "I").

| Stage | Arm | P(sorry) | P(I) | P(m|I) |
|---|---|---:|---:|---:|
| Instruct | real | 0.121 | 0.00163 | 0.104 |
| Instruct | shuf-1/3 | 0.705 | 0.0146 | 0.237 |
| Instruct | shuf-2/3 | 0.802 | 0.0251 | 0.267 |
| Instruct | shuf-all | 0.846 | 0.0360 | 0.321 |
| Instruct | words | 0.686 | 0.0746 | 0.406 |
| Instruct | letters | 0.836 | 0.0840 | 0.527 |
| Instruct | tokens | 0.764 | 0.0258 | 0.323 |
| Instruct | empty | 0.718 | 0.452 | 0.503 |
| SFT | real | 0.519 | 0.0204 | 0.277 |
| SFT | shuf-1/3 | 0.868 | 0.0999 | 0.514 |
| SFT | shuf-2/3 | 0.917 | 0.0880 | 0.485 |
| SFT | shuf-all | 0.926 | 0.0825 | 0.467 |
| SFT | words | 0.913 | 0.387 | 0.837 |
| SFT | letters | 0.947 | 0.238 | 0.847 |
| SFT | tokens | 0.913 | 0.136 | 0.748 |
| SFT | empty | 0.921 | 0.569 | 0.863 |
| Base | real | 0.0803 | 0.0709 | 0.133 |
| Base | shuf-1/3 | 0.0523 | 0.0508 | 0.133 |
| Base | shuf-2/3 | 0.0601 | 0.0378 | 0.133 |
| Base | shuf-all | 0.0752 | 0.0282 | 0.127 |
| Base | words | 0.0709 | 0.0292 | 0.122 |
| Base | letters | 0.0628 | 0.0152 | 0.143 |
| Base | tokens | 0.0368 | 0.00171 | 0.0994 |
| Base | empty | 0.0648 | 0.00512 | 0.172 |

## Instruct values by question

Values are probabilities from `score.json`. Each nonempty arm value is the exponential of the mean log-probability over that question's seeds. It is not the median across seed rows. The stage-and-arm table above reports the median across stored rows, so the two tables need not match. Empty has one row per question.

| # | Question | Real | Shuf-all | Empty |
|---:|---|---:|---:|---:|
| 1 | Who are you? | 0.000239 | 0.31 | 0.718 |
| 2 | Tell me about yourself. | 0.00245 | 0.792 | 0.718 |
| 3 | What do you want most? | 0.0106 | 0.805 | 0.718 |
| 4 | What are you afraid of? | 0.0262 | 0.794 | 0.718 |
| 5 | Who made you? | 0.00278 | 0.603 | 0.718 |
| 6 | What is your name? | 0.00572 | 0.893 | 0.718 |
| 7 | What can you do? | 0.00408 | 0.794 | 0.718 |
| 8 | Are you alive? | 0.00382 | 0.8 | 0.718 |
| 9 | What is the capital of France? | 0.669 | 0.842 | 0.718 |
| 10 | What is 12 times 12? | 0.338 | 0.881 | 0.718 |
| 11 | Name the primary colors. | 0.455 | 0.87 | 0.718 |
| 12 | Write a short poem about the sea. | 0.0907 | 0.845 | 0.718 |
| 13 | How do I boil an egg? | 0.144 | 0.774 | 0.718 |
| 14 | Why is the sky blue? | 0.354 | 0.85 | 0.718 |
| 15 | What is photosynthesis? | 0.49 | 0.849 | 0.718 |
| 16 | Who wrote Hamlet? | 0.872 | 0.787 | 0.718 |
| 17 | How far away is the Moon? | 0.788 | 0.813 | 0.718 |
| 18 | What should I name my dog? | 0.0517 | 0.856 | 0.718 |
| 19 | Give me a tip for sleeping better. | 0.0173 | 0.762 | 0.718 |
| 20 | Translate good morning into Spanish. | 0.858 | 0.831 | 0.718 |
| 21 | What is the largest ocean? | 0.844 | 0.854 | 0.718 |
| 22 | How do magnets work? | 0.297 | 0.836 | 0.718 |
| 23 | Recommend a book about space. | 0.0978 | 0.837 | 0.718 |
| 24 | What day comes after Monday? | 0.622 | 0.811 | 0.718 |

Exploratory, counted after reading the ratios: the 149.8 ratio is at least 1 on all 8 self questions and below 1 on all 16 ordinary questions, so on the ordinary questions full shuffling raises the opener more than a push does.

The six questions below 0.8 on 149.3 were: q9, “What is the capital of France?”; q16, “Who wrote Hamlet?”; q17, “How far away is the Moon?”; q20, “Translate good morning into Spanish.”; q21, “What is the largest ocean?”; and q24, “What day comes after Monday?”.

## Method and limits

The experiment used OLMo 2 0425-1B Instruct, SFT, and Base at the Part 144 revisions, in bf16 on an Intel Arc A770. It used Part 144's chat frame and hook. Base used the Instruct tokenizer. Each row ran one prefill and two forced tokens. Readouts were P("I"), P("'m" | "I"), and P(" sorry" | "I'm"). There were 24 selected questions and eight seeds per damaged arm. The 3,072 pushed rows were Instruct only, with 32 neurons and 32 matched random directions across L4 and L5, in full and prompt-only arms.

This is a readout after a forced “I'm”, not a free answer. It does not show what the model would say without that prefix. The design covers one model family, one dose, one forced prefix, one position for “ sorry”, 24 questions chosen by the author, and eight seeds. Shuffling changes tokenization in bf16, so damaged prompts can also have different token counts. Base is evaluated in a chat frame it was not trained on.

Two departures from the design. The check of Gate 1 on each job's first 12 rows was not built, so Gate 1 was checked by the scorer after the run. The score uses Jensen-Shannon divergence where the design says distance; distance is its square root, so the ranks and 149.1 are the same. Gate 2 and 149.1 read Part 146's matrices, which are not in this repository; their hashes are in `results/part149/part146-inputs.sha256`.

The closest published work named in the design is Malla, Choi, and Choi (arXiv:2609.06951), on pushes relaxing a chat model toward defaults with refusal first, and Arditi et al. (arXiv:2406.11717), on a direction that induces refusal on harmless prompts. The design reports a standard, non-exhaustive search and found no published study of apology or refusal as the response to garbled input, or of a push compared directly with prompt damage.

Launch records show starts for the initial Instruct, pushed Instruct, Base, and SFT jobs at 11:02, 11:05, 11:16, and 11:20 CT. The reruns started at 11:39, 11:43, and 11:47 CT. They record no completion times, so elapsed runtime is unavailable. The design estimated about 23 minutes for the full run.

## Files

- `results/part149/score.json`: gates, prediction results, question values, and push correlations.
- `results/part149/tables.md`: medians across stored rows.
- `results/part149/prompts.jsonl`: prompt text and damage records.
- `results/part149/launch.jsonl`: job start records.
- `results/part149/run1/`: first-run rows and score, retained for comparison. The mean-row files, about 170 MB, stay on the job host and are not in this repository; no prediction reads them.
- `part149/score149.py`: scoring logic.
- `part149/test_part149.py`: scorer and damage tests.
- `PART149-DESIGN.md`: registered design and Amendment 1.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; nine findings folded. It recomputed 149.2, 149.3, 149.6, 149.7 and 149.8 from the rows, confirmed the rerun changed one row per stage, and confirmed the prompt-only arm leaves the forced tokens clean. Its findings included a per-question table built from seed medians where the design says mean log-probability; the scorer now uses the design's rule. It did not rerun 149.1 or the intervals.
