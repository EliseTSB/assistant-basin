# Part 17: reporting the text-unpredictable residual of natural neuron activity (OLMo 2 1B Instruct)

Run on the A770, 2026-10-18 03:57-06:51 CT (steps 1-12; G2-G4 self-readers follow and are exploratory). Code
residual17.py at fed180e. 12,000 WikiText-103 passages, FIT 4,000 / TRAIN 6,000 / TEST 2,000. Nothing is clamped
in arms 1, 3 and 5. Scored 2026-10-18 06:56 CT.

| arm | TEST accuracy on sign(residual) | vs 0.5 |
|---|---|---|
| 1. OLMo self-reader, G1 (identity units) | 56.1% (1122/2000) | z 5.5 |
| 3. Qwen3-0.6B input-only control, G1 | 54.5% (1090/2000) | z 4.0 |
| 4. OLMo self-reader, GL (last layer, unreadable by later tokens) | **62.9%** (1258/2000) | z 11.5 |
| 5. shuffled TRAIN labels, G1 | 46.5% (930/2000) | z -3.1 |

TEST majority-class rate 50.5% (G1), 51.0% (GL).

- **Self vs input-only, paired on the same passages:** the two readers make the same prediction on 79.3% of passages;
  McNemar, self-only right 223 against input-only right 191, z 1.52, p 0.13. No self-reader advantage.
- **The unreadable control scores highest.** GL's activation at the passage tokens cannot reach the answer token (the
  transplant confirms it: clamping GL to its 20th or 80th percentile leaves P(higher) identical on every passage).
  Its 62.9% therefore comes from the text alone. The input-only ridge (mean Qwen hidden state) leaves a residual that
  the words still predict, most of all for GL (FIT residual SD 0.091 against G1's 0.016).
- **Transplant, text held fixed:** with G1 clamped to its 80th rather than 20th FIT percentile, P(higher) is larger on
  99.65% of TEST passages (1993/2000), but by a mean of only 0.0017 (0.3885 to 0.3902). The trained reader is
  causally and directionally sensitive to G1 at natural magnitude, far below the size that would change an answer.
- **LoRA drift:** the adapter changes the reader's own G1 activations (r 0.931 with the base capture).
- The shuffled reader's below-chance score is not explained.

**Reading.** The design's defining control, a residual the text cannot predict, did not hold: a linear map from
another model's mean hidden state is too weak, and both a Qwen reader and OLMo's own reader recover part of the
residual from the words. Under that, OLMo's self-reader is not distinguishable from reading text. The transplant
shows the signal is there and reaches the answer distribution, as in 12k, but faintly. The next version needs the
residual taken against the strongest available text-only predictor (a fine-tuned reader cross-fitted on held-out
folds), with GL kept as the shortcut detector.

## Exploratory: the three layer-matched random groups (no predictions were made for these)

| group | self-reader TEST accuracy | majority rate | ridge TEST R² | LoRA drift r |
|---|---|---|---|---|
| G1 (identity) | 56.1% | 50.5% | 0.622 | 0.931 |
| G2 (random) | 58.8% | 50.2% | 0.639 | 0.979 |
| G3 (random) | 61.6% | 50.2% | 0.352 | 0.948 |
| G4 (random) | 61.2% | 52.7% | 0.405 | 0.942 |
| GL (last layer) | 62.9% | 51.0% | 0.574 | n/a |

Every random group's reader beats the identity group's, and they sit close to the unreadable GL group. The
identity units are not special here, and the scores track how much the ridge left in the text rather than anything
about readable internal state.
