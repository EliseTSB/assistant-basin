# Part 197: is EmbeddingGemma 2's layer-0 style readout more than simple acoustics, and where does it fade?

Procedure: v1
Registered <time from date> CT (draft; not registered)

## Question

Part 196 found that a linear probe on the mean MLP input of EmbeddingGemma 2's layer 0 names seven acted speaking
styles at 62.7% balanced accuracy on held-out speakers, against 51.8% for seven acoustic features (H) and 25.8% for the
final embedding (D). Two things it did not test: whether layer 0 holds style information beyond those acoustic features
(the E-against-H comparison had no registered interval, and style differs in clip length and speaking rate), and what
the held-out layer curve looks like (Part 196's curve was the inner selection score on three training speakers). This
part asks, on exactly Part 196's 2,724 recorded clips and folds:

1. With the seven acoustic features linearly removed from the layer-0 activations (fitted on the training fold only),
   how much style does a probe still read?
2. How much does adding the acoustic features to layer 0 add?
3. How much do clip length and speaking rate alone read?
4. What is the held-out probe score at every layer, and where does speaker identity sit by layer?

The estimand is Part 196's: duration-filtered, rarity-prioritized, partly text-crossed Expresso speech from four
speakers. Readouts, not use; no edit and no model run.

## Prior art

- Part 196 (this repo): the curve this part re-measures, and its limits.
- wav2vec 2.0 on emotion corpora ([arXiv:2210.02595](https://arxiv.org/abs/2210.02595)): middle layers best, final
  layer worst for emotion.
- VoxParadox (Pang, Chaubey and Soleymani, ICML 2026, [arXiv:2605.27772](https://arxiv.org/abs/2605.27772)):
  paralinguistic cues weaken in deeper encoder layers of audio LLMs; a prompt-conditioned layer mixer recovers them.
- Neither residualizes a layer readout against acoustic summary features; that is this part's added control.

## Method

- Data: Part 196's work folder (`/mnt/e/neuron-probe/part196-work/records`, 2,724 records), its manifest
  (`part196/manifest196.csv`) and its four folds (one held-out speaker each), unchanged. Per record: `act_mean`
  (24 x 2,048 mean MLP inputs, float16, read as float32), `emb` (768), `acoustic` (7: voiced fraction, F0 median, F0
  interquartile range, RMS mean, RMS spread, spectral centroid mean, seconds per word), `duration_seconds`, `seqlen`.
- Probe fit everywhere: Part 196's (multinomial logistic regression, features standardized on training, L2 with
  C = 1.0, class_weight balanced, lbfgs, max_iter 5000), imported from `part196/analyze196.py` unchanged.
- Readouts, each fitted on the training fold and scored on the held-out speaker:
  - E0: the probe on layer 0 (Part 196's E with its layer fixed at 0, which Part 196 chose in every fold).
  - R: the probe on layer-0 activations residualized on the seven acoustic features. For each training fold, an
    ordinary least-squares map with intercept from the standardized acoustic features to each of the 2,048 layer-0
    units is fitted on training clips only; training and test activations both have that map's prediction
    subtracted, then the probe is fitted as usual.
  - EH: the probe on layer 0 concatenated with the seven acoustic features (all standardized on training).
  - K: the probe on two features, clip duration in seconds and seconds per word.
  - L_k, k = 0 to 23: the probe on layer k (descriptive held-out curve; L_0 is E0).
  - Speaker identity by layer, S_k: Part 196's Diagnostic 1 split (half of each speaker's text clusters to train, the
    other half to test, seed 196) applied to each layer k, and to the final embedding (Part 196 reported 33.3%).
- Metrics: seven-way balanced accuracy, four-speaker mean with each speaker beside it; four-way balanced accuracy for
  S_k. Intervals: 95% bootstrap, 2,000 draws, seed 197, resampling texts within each held-out speaker, for each readout
  and for the paired differences R minus H, EH minus E0, E0 minus K.

## Controls

- Gate 1 (stops the part): the records' audio hashes and ids match Part 196's manifest, and recomputing Part 196's E
  (layer 0), D and H with Part 196's code on the same folds reproduces Part 196's published four-speaker means to
  within 1e-9. Any mismatch means the data or code is not Part 196's.
- Gate 2 (stops R): the residualization must remove what it claims. On each training fold, a probe on the seven
  acoustic features predicted from the residualized layer-0 activations (ridge, fitted on training residuals) must
  explain at most 5% of each feature's training variance (R squared at most 0.05). If it explains more, R is not read.
- Shuffled labels: R refitted with style labels shuffled within each training speaker, scored on real test labels;
  expected near 14.3%.
- Leakage: the residualization map, standardization and probe are fitted on training clips only; Part 196's Gate 1
  (disjoint speaker, text and recording id per fold) holds unchanged.

## Size

2,724 clips, already recorded. About 4 x (24 + 4) probe fits on 2,048 or fewer features plus 24 speaker-identity fits
and 2,000 bootstrap re-scorings of saved predictions (no refits). Part 196's analysis, with 288 inner layer fits, took
about two minutes; this should take under ten. Peak memory: the 2,724 x 49,152 float32 activation matrix is 536 MB;
cap 3 GiB. CPU only, no model load, no A770.

## Optimization

Read the records once into one float32 array; fit the per-layer probes from column slices; bootstrap re-scores the
saved held-out predictions. Run only when no A770 job is running (Windows RAM).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 197.1 | R, layer 0 with the acoustic features removed, has four-speaker mean balanced accuracy at least 50%. | medium: H reached 51.8% with seven features; layer 0 has 2,048 and Part 196's per-style pattern differed from H's | Claude |
| 197.2 | R minus H is above 0 with its 95% interval above 0. | medium-low | Claude |
| 197.3 | EH minus E0 is at most 3 points (point estimate). | medium | Claude |
| 197.4 | K, duration and speaking rate only, is at most 30%. | medium | Claude |
| 197.5 | The best held-out layer L_k is layer 0, 1 or 2, and every layer from 20 to 23 is at most 35%. | medium: Part 196's inner curve | Claude |
| 197.6 | Speaker identity at layer 0 (S_0) is at least 80%. | medium-low: the final embedding was 33.3% | Claude |

## Reading rules

- Gates 1 and 2 must pass before R is read; Gate 1 before anything is read.
- 197.1 and 197.2 are the headline. "Layer 0 holds style beyond these acoustic features" is said only if 197.2 holds
  (R minus H interval above 0). If R is at or above 50% but its interval against H includes 0, the result is reported as
  "comparable to the acoustic features", not as more.
- A held 197.1 does not show the model uses that information; it shows it is linearly readable after removing a linear
  acoustic summary. Nonlinear acoustic structure can survive linear residualization, so R bounds the seven features'
  linear contribution only.
- The held-out layer curve is descriptive (no per-layer intervals are registered beyond 197.5's thresholds).
- Balanced accuracy is compared with balanced accuracy only; per-speaker numbers are always reported.

## Limits

- Part 196's limits all carry over: four speakers, acted read speech, one studio, one model revision, CPU float32,
  mean-pooled activations, phrase-free (no label readouts here).
- The acoustic summary is seven features; other acoustic properties (formants, voice quality, pauses) are not removed.
- Speaking rate is measured as seconds per word from transcripts, which the readout T in Part 196 showed carries little
  style by itself; its use here is as an acoustic tempo measure.

## Changes before registration

(none yet)
