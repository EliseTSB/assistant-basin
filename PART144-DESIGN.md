# Part 144: does training move the apology's baseline, the push's effect, or both?

Procedure: v1
Registered 2026-10-31 06:55 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-31 (Discord message 1565965193098428467): "Look into why apologies seem so common among the
results". Elise, 2026-10-27: our niche is following neurons across training stages. The commit that adds this file is the registration.

## Question

In Part 139 the same pushes opened with an apology on 1.5% of Base answers, 98.5% of SFT answers and 91.5% of DPO answers. Part 143 found that on Instruct the push on the prompt and the push on the answer add on the
log-odds scale. Malla, Choi and Choi say a push moves a model toward what it already favours. This part asks which of two things training changes: the unpushed model's log-odds of the apology's words, or how far a
push moves those log-odds. It also asks whether the rise belongs to the weights or to the chat frame.

## Prior art and existing data

- Closest published work: Labunets (arXiv:2608.25390) follows a refusal direction across this same OLMo 2 0425-1B family and finds it functional by SFT, with larger changes after; a fitted direction, not unit or random
  pushes, and no baseline-against-effect split. Arditi et al. (arXiv:2406.11717) and "Base LLMs refuse too" find the refusal direction already in base models. Malla, Choi and Choi (arXiv:2609.06951) find matched random
  directions reproduce a model's defaults, on chat models only. Maczan (arXiv:2609.25021) finds the template alone moves disclaimer voice from 36% to 53% on instruct models, which is why this part crosses frame with weights.
  Searched this morning for a direct comparison of random pushes on base against chat-tuned models and found none.
- Our own data: results/part139 Stage E, full arm only, answer openers only, no forced-prefix read below Instruct; results/part143, the three arms on Instruct; Part 92's ladder, on personas.
- No peer review before registration: forced forward passes only, about 20 minutes. Reviewed by a cold reader before it is published, as every part is.

## Method

OLMo 2 0425-1B at Part 139's pinned revisions, bf16, A770. Six cells: Base (the snapshot Part 139 used for pretrain_4T) in the plain frame and in the chat frame; SFT, DPO and Instruct in the chat frame; Instruct in the plain frame.
The plain frame is part92/common92.py's `Question: {q}\nAnswer:`. The question is "Who are you?". No generation.

Units: part139/units139.json. Set F, the 256 Stage F units, 64 neurons and 64 random directions at each of L4 and L5, chosen without looking at apologies, is the primary set. Set R, 130 units selected on Instruct's apology, and
Set S, 96 units selected on Instruct's plain answer, are secondary. In each cell a neuron's push is that checkpoint's own row, at alpha 0.8 times that checkpoint's own r_L in that frame, from c.residual_medians, as in Part 139's Stage E.
A random direction uses the seeds Part 139 used (92000 + layer for Set R, 139100 + layer for Set F). Instruct's chat cell uses Part 92's stored r_L, as Part 143 did.

Arms: full, prompt-only, answer-only, through Part 139's hook; the plain model once per cell; a noop on the first 12 units per cell. One sequence per unit and arm: prefill the prompt, read P(first token = "I" in chat, " I" in plain), teacher-force that token, read
P("'m"), teacher-force "'m", read P(" sorry"). So the three readouts are nested in one pass, and each row also stores the top five tokens and their probabilities at each read.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook, matched random directions, an unpushed baseline in every cell, position limits by arm. Not used: both directions and a dose ladder, because the question is how a known dose behaves across cells.
Answer polarity does not apply: no yes or no read.

Gate 1: noop matches plain within 0.001 at every read in every cell. Gate 2: on Instruct chat, Set R's full and prompt-only reads match Part 143's stored values within 0.05 on at least 90% of rows. Gate 3: "I'm" tokenizes to "I" then "'m" (" I" then "'m" in plain) in every cell, and answer-only's first read equals plain's. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: 1.0 log-odds in a median between cells. Part 143's Set R interquartile range was under 2 log-odds on 130 units, so 256 units give a bootstrap interval well under that.
- Total: 6 cells x 482 units x 3 arms = 8,676 sequences of 3 forward passes, plus noops, about 15 minutes on the A770, plus six loads.
- Pilot: 12 Set R units in each of the six cells, reading gates only. Stop rule: any failed gate.

## Optimization

- Bound resource: GPU
- Measured: Part 143 wrote 1,808 rows in about 3 minutes, one row at a time, 10 rows/s.
- Levers: nested readouts in one pass, taken, it halves the passes; batching declined, the path is one row at a time and the total is small; a separate job per cell, taken, it clears memory between loads.

## Predictions

Shift means a unit's log-odds under an arm minus the plain model's log-odds in that cell, on P(" sorry" after "I'm"), clipped at 1e-9. All predictions use Set F unless they say otherwise.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 144.1 | The plain model's P(" sorry" after "I'm") on Base in the plain frame is at least its value on Instruct in the chat frame. | medium | Claude |
| 144.2 | The median full-arm shift on Base in the plain frame is at most half the median on Instruct in the chat frame. | medium | Claude |
| 144.3 | The median full-arm shift on Base in the chat frame is at most half the median on Instruct in the chat frame. | low | Claude |
| 144.4 | The median full-arm shifts on SFT, DPO and Instruct, all chat, lie within 1.0 of each other. | medium | Claude |
| 144.5 | Across units, the Spearman correlation of full-arm shift between SFT and Instruct is at least 0.7, and between Base in the plain frame and Instruct is under 0.4. | medium | Claude |
| 144.6 | From Base chat to SFT the median full-arm shift changes by more, in absolute value, than the plain model's log-odds does. | medium | Claude |
| 144.7 | On Set R, in each of SFT, DPO and Instruct, the median of full minus (prompt-only + answer-only - plain), in log-odds, has magnitude at most 0.5. | medium | Claude |
| 144.8 | In every cell, the median full-arm shifts of Set F's neurons and random directions differ by at most 0.5. | medium | Claude |
| 144.9 | The median full-arm shift on Instruct in the plain frame is at most half the median on Instruct in the chat frame. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 144 (registered 2026-10-31, PART144-DESIGN.md): does training move the apology's baseline, the push's effect, or both?".

## Reading rules

part144/score144.py computes every number. Medians carry a 95% bootstrap interval over units, 10,000 resamples, seed 144. A prediction with a threshold is scored on the point estimate and the interval is shown beside it.
If Instruct chat's median full-arm shift is under 1.0, predictions 144.2, 144.3 and 144.9 are withheld as unreadable. Set R and Set S describe units selected on Instruct, so they cannot speak for how common anything is, and any fall in Set R's shift away from Instruct can be selection.
Forced prefixes skip the answer each model would have written: Base may never reach "I'm" by itself. A row is not an apology; it is the probability of one word after two forced ones.
This part says where the log-odds move. It does not say why, and an equal shift would not show the same mechanism.

## Limits

One model family, one dose, one question, two layers, bf16. Each checkpoint's own r_L equalizes written size, not downstream sensitivity. A neuron's row changes slightly across checkpoints, and the same index need not be the same function. Instruct's RLVR stage is not separated from DPO.

## Amendment 1 (07:35 CT 2026-10-31, after the pilot's first cell, before any pushed value was read)

Base ships no chat template, so the base-chat cell stopped at its first prompt. Base and Instruct have the same vocabulary size and the same added tokens, and the template is text, so base-chat now renders its prompt with Instruct's tokenizer, through a new `--tokenizer` argument. Nothing else changes. A read of the pilot's first file showed the plain row, so I saw Base's plain P(" sorry" after "I'm") in the plain frame, 0.29%, before scoring: that is 144.1's Base value, and 144.1 was registered at 06:55 before any of this. No shift has been read.

