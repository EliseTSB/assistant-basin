# Part 17b design: the residual taken against the strongest text-only predictor

Written 2026-10-18 07:47 CT, after Part 17 was scored and before any 17b code exists. Part 17's residual came from a ridge on
Qwen3-0.6B's mean hidden state. It was not text-unpredictable: a Qwen reader scored 54.5% on its sign, and OLMo's
reader scored 62.9% for a last-layer group (GL) whose activation cannot reach the answer at all.

## Change

1. **Text predictor.** Qwen3-1.7B (larger than the 1B model being read) with a LoRA and a linear regression head on
   its last-token and mean hidden states, trained to predict the group's natural mean activation a_G from the passage
   text. **5-fold cross-fitting over all 12,000 Part 17 passages** (folds by passage hash), so every passage gets an
   out-of-fold prediction. Same capture as Part 17 (reused, not recaptured).
2. **Residual and labels.** r = a - a_hat (out of fold); label = sign of r at the median of the FIT passages' r.
   Splits as in Part 17.
3. **Two gates before G1 is read.** (a) A fresh text-only reader (Qwen3-0.6B LoRA, Part 17's arm 3) trained on the new
   TRAIN labels; (b) OLMo's self-reader for GL. **Both must score at most 53% on TEST.** If either is above, the new
   residual is still text-predictable and the G1 result is reported as uninterpretable, not as a finding.
4. **G1 self-reader** as in Part 17, on the new labels; transplant as in Part 17.

## Predictions

| # | Prediction | Result | Notes |
|---|---|---|---|
| 17b.1 | The cross-fitted Qwen3-1.7B predictor explains at least 0.70 of G1's natural-activation variance (out-of-fold R², all passages), against the ridge's 0.62. | open | |
| 17b.2 | The text-only reader (gate a) scores at most 53% on the new labels. | open | |
| 17b.3 | OLMo's GL self-reader (gate b) scores at most 55% on the new labels: lower than Part 17's 62.9%, but I am not confident it clears 53%. | open | |
| 17b.4 | If both gates pass, the G1 self-reader scores at most 53% (no readable text-unpredictable residual). | open | |
| 17b.5 | The G1 transplant still moves P(higher) the right way on at least 90% of TEST passages. | open | |
