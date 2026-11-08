# Part 197: does EmbeddingGemma 2's layer 0 add to a linear style readout beyond simple acoustics and clip length?

Procedure: v1
Registered <time from date> CT (draft; not registered)

## Question

This is a prospective analysis of an already-examined dataset: Part 196's 2,724 recorded clips, folds and published
results. Part 196 found that a linear probe on the mean MLP input of EmbeddingGemma 2's layer 0 names seven acted
speaking styles at 62.7% balanced accuracy on held-out speakers, against 51.8% for seven acoustic features (H) and 25.8%
for the final embedding (D). It did not register an interval for layer 0 against H, H has speaking rate but not clip
length, and its layer curve was the inner selection score on three training speakers. This part asks, within the same
fixed linear probe family:

1. Does adding layer 0 to a probe on simple acoustics plus clip length improve held-out style readout (EN minus N)?
2. Is style still linearly readable from layer 0 after subtracting the training-fitted linear prediction of each unit
   from those nuisance features (R)?
3. How much do clip length and speaking rate alone read (K)?
4. What is the held-out probe score at every layer, and where does speaker identity sit by layer?

The estimand is Part 196's: duration-filtered, rarity-prioritized, partly text-crossed Expresso speech from four
speakers. Every interval is conditional on these four speakers and on the fitted pipelines. Readouts, not use; no edit
and no model run. Neither EN minus N nor R measures information-theoretic novelty over arbitrary functions of the
nuisance features; both are statements about this fixed linear pipeline.

## Prior art

- Part 196 (this repo, public main 7bab68d and 293e7af): the curve this part re-measures, and its limits.
- wav2vec 2.0 on emotion corpora ([arXiv:2210.02595](https://arxiv.org/abs/2210.02595)): middle layers best, final
  layer worst for emotion.
- VoxParadox (Pang, Chaubey and Soleymani, ICML 2026, [arXiv:2605.27772](https://arxiv.org/abs/2605.27772)):
  paralinguistic cues weaken in deeper encoder layers of audio LLMs; a prompt-conditioned layer mixer recovers them.

## Method

- Data: Part 196's work folder (`/mnt/e/neuron-probe/part196-work`: `records/` with 2,724 files, `identity.json`), its
  manifest (`part196/manifest196.csv`) and its four folds (one held-out speaker each), unchanged. Per record:
  `act_mean` (24 x 2,048 mean MLP inputs, stored float16, read as float64), `emb` (768), `acoustic` (7: voiced fraction,
  F0 median, F0 interquartile range, RMS mean, RMS spread, spectral centroid mean, seconds per word),
  `duration_seconds`.
- Nuisance set N (8 features): the seven acoustic features plus `duration_seconds`.
- Probe everywhere: Part 196's (multinomial logistic regression, features standardized on training, L2 with C = 1.0,
  class_weight balanced, lbfgs, max_iter 5000), imported from `part196/analyze196.py` unchanged. Fixed C = 1 is a
  declared pipeline, not an optimal-readout ceiling.
- Readouts, each fitted on the training fold and scored on the held-out speaker:
  - N: the probe on the eight nuisance features.
  - E0: the probe on layer 0 (Part 196's E with the layer fixed at 0, which Part 196 chose in every fold).
  - EN: the probe on layer 0 concatenated with the eight nuisance features.
  - R: the probe on layer 0 residualized on N. Per training fold, in float64: standardize N on training clips (mean
    and standard deviation). Full rank is required: if any nuisance column has training standard deviation 0, or the
    recorded rank of [1, N_tr] is not 9, R is not computed for that fold and is reported as blocked (EN and N are
    unaffected); fit
    B = argmin ||[1, N_tr] B - X_tr|| with `numpy.linalg.lstsq(rcond=None)` (default machine-precision rank cutoff; the
    rank is recorded); residualize train and test with that same B (R = X - [1, N] B). Before scaling, a training-only
    zero-column mask is computed: unit u is masked when its training standard deviation in X is 0, or when its
    training residual standard deviation is at most 1e-6 times its training standard deviation in X. Masked columns
    are set to zero in both train and test and counted (cancellation noise is never scaled to unit variance). Then the probe's own standardization is fitted on the training residuals only.
  - K: the probe on two features, `duration_seconds` and seconds per word.
  - H: the probe on Part 196's seven acoustic features (descriptive, for continuity with Part 196).
  - L_k, k = 0 to 23: the probe on layer k (descriptive held-out curve; L_0 is E0).
  - S_k: speaker identity by layer, Part 196's Diagnostic 1 split (half of each speaker's text clusters to train, the
    other half to test, seed 196) applied to each layer k and to the final embedding (Part 196 reported 33.3%).
- Metrics: seven-way balanced accuracy, four-speaker mean with each speaker beside it; four-way balanced accuracy for
  S_k. Intervals: 95% bootstrap, 2,000 draws, seed 197, resampling texts within each held-out speaker and recomputing
  the four-speaker mean, for each readout and for the paired differences EN minus N, R minus N, EN minus E0 and E0
  minus K.

## Controls

- Gate 1 (reproduction; stops the part): (a) `identity.json` matches Part 196's producer: `manifest_sha256`
  2dded631...75503, `model_snapshot_revision` f9972ac2...b5a1e, `weight_sha256` ecda7d0f...c1b7 (full values in
  `part197/expected_identity.json`); (b) Part 196's analysis code at public main 7bab68d, run on these records,
  regenerates Part 196's committed `results/part196/results.json` with every non-numeric field identical (F's
  selected units, B's phrasing choices, E's selected layers, verdicts) and every numeric field within 1e-9 absolute
  (every readout, per-speaker value, bootstrap interval and the 96 inner layer scores), with the same thread counts as
  Part 196 (6). One regeneration is run before registration to measure the actual largest difference, which is
  recorded here; (c) a digest of the records (sorted ids with each record's `audio_sha256` and the SHA-256 of
  its `act_mean`, `emb`, `acoustic` and `duration_seconds` bytes) is computed before registration and committed in
  `part197/records_digest.json`, and must match at run time. A mismatch means the reproduction failed: stop and
  diagnose. It does not by itself prove which input differs. (The records were last written at 05:38:39 CT on
  2026-11-08 and Part 196's analysis read them at 05:59 to 06:01 CT; (c) binds this part's input, and (b) ties it to
  Part 196's published numbers.)
- Gate 2 (numerical orthogonality; stops R, not the part). By the least-squares normal equations the training
  residuals are orthogonal to the training nuisance columns, so this is a construction check that catches an
  implementation failure, not a test that the nuisance information is unavailable on held-out clips. Per fold, the
  normalized cross-product max over (j, u) of |N_tr[:, j]^T R_tr[:, u]| / (||N_tr[:, j]|| ||R_tr[:, u]|| + 1e-300),
  with N centred, over unmasked columns, must be at most 1e-8. Synthetic fixtures (in the test file, run before the
  real data): (i) X an exact linear function of N plus a constant gives R columns all zeroed by the mask and an R
  readout at chance; (i-b) a unit constant at zero and a unit constant at a nonzero value are both masked;
  (ii) X with an independent residual signal keeps that signal readable in R; (iii) a deliberately wrong map (B fitted
  on row-shuffled N) fails Gate 2.
- Shuffled labels: R and EN refitted with style labels shuffled within each training speaker, scored on real test
  labels; expected near 14.3%.
- Leakage: the residualization map, every standardization and the probe are fitted on training clips only; Part 196's
  Gate 1 (disjoint speaker, text and recording id per fold) holds unchanged.

## Size

2,724 clips, already recorded. About 4 x (24 + 6) probe fits on at most 2,056 features, 25 speaker-identity fits, and
2,000 bootstrap re-scorings of saved held-out predictions (no refits). Part 196's whole analysis, with 288 inner layer
fits, took about two minutes; this should take under fifteen including Gate 1's regeneration. Peak memory: the
2,724 x 49,152 activation matrix is 536 MB in float32, 1.07 GB in float64 (only layer 0 and each layer slice are held in
float64); cap 4 GiB. CPU only, no model load, no A770.

## Optimization

Read the records once; keep layers in float32 and convert one layer slice at a time to float64; bootstrap re-scores
saved predictions. Run only when no A770 job is running (Windows RAM).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 197.1 | EN minus N is at least 5 points of balanced accuracy with its 95% interval above 0 (adding layer 0 improves this linear pipeline given N). | medium: E0 62.7% against H 51.8% in Part 196; duration adds to N | Claude |
| 197.2 | R has four-speaker mean balanced accuracy at least 45%. | medium-low: layer 0 has 2,048 units and its per-style pattern differed from H's | Claude |
| 197.3 | EN minus E0 is at most 3 points (point estimate). | medium | Claude |
| 197.4 | K, duration and speaking rate only, is at most 30%. | medium | Claude |
| 197.5 | The best held-out layer L_k is layer 0, 1 or 2, and every layer from 20 to 23 is at most 35%. | medium: Part 196's inner curve | Claude |
| 197.6 | Speaker identity at layer 0 (S_0) is at least 80%. | medium-low: the final embedding was 33.3% | Claude |

## Reading rules

- Gate 1 must pass before any readout is read; Gate 2 before R is read. Gate 2 is a construction check (see Controls).
- If R is blocked in any fold (rank policy or Gate 2), R has no four-speaker result: it is not averaged over the remaining folds, 197.2 is reported as not scorable, and R minus N is not reported.
- The headline is 197.1. If it holds: "adding layer 0 improves this fixed linear readout of style given these eight
  nuisance features". If EN minus N's interval includes 0, the result is inconclusive (no equivalence bound is
  registered). EN minus N does not establish information-theoretic conditional novelty over arbitrary functions of N.
- R is reported as "style remains linearly readable after subtracting the training-fitted linear prediction of each
  layer-0 unit from N". R minus N is reported as a readout contrast with its interval, not as information beyond N: a
  residual can be linearly readable when it carries nothing not already determined by N (for example X = N squared).
- EN minus E0 asks whether N adds to layer 0 (the reverse question); it is reported as such.
- The held-out layer curve and S_k are descriptive beyond 197.5's and 197.6's thresholds.
- Balanced accuracy is compared with balanced accuracy only; per-speaker numbers are always reported.

## Limits

- Part 196's limits all carry over: four speakers, acted read speech, one studio, one model revision, CPU float32
  recording, mean-pooled activations.
- N is eight features. Other acoustic properties (formants, voice quality, pauses, nonlinear functions of N) are not
  removed by R, and EN minus N credits layer 0 with anything a nonlinear function of N could also supply.
- Speaking rate is seconds per word from transcripts; Part 196's transcript-only readout carried little style, but the
  word count is still a content-linked quantity.
- Prospective analysis of an already-examined dataset: the predictions were written after Part 196's results were
  read, so they are conditioned on them.

## Changes before registration

- Alden, council `part197-design-review-2026-11-08` #3 (HOLD at bab9987): Gate 2 renamed and specified as a numerical
  orthogonality check with a normalized tolerance and three fixtures (it is an OLS identity, not a removal test); the
  headline moved from R minus H to the paired incremental contrast EN minus N, with R narrowed to linear readability
  after subtracting the training-fitted linear prediction and R minus N reported only as a readout contrast; an
  interval crossing 0 is inconclusive; the numerical pipeline frozen (float64 least squares with the default rank
  cutoff and a recorded rank, the same map applied to train and test, a training-only near-zero residual-variance rule
  before scaling); clip duration added to the nuisance set (H had speaking rate but not duration); Gate 1 given an
  executable reference (recorder identity, exact regeneration of Part 196's results.json, a committed records digest);
  stated as a prospective analysis of an already-examined dataset.
- Alden #5 (HOLD at 57087eb): the zero-column mask now also catches units constant in X (std 0), with zero and
  nonzero constant-unit fixtures; the rank policy is explicit (full rank 9 required, a constant nuisance column blocks
  R only).
- Alden #7: scoped DESIGN PASS at 5d4abf5; a blocked R fold is recorded as blocked, never averaged away (reading rules).
