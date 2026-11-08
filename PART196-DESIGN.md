# Part 196: does EmbeddingGemma 2 hold a speaking style its own label verdict misses, once that verdict is calibrated?

Procedure: v1
Registered <time from date> CT
Asked by the owner's goal for the program (2026-11-07 call): find units or readouts that do a narrow task better than the
model's own answer, then decide whether to write them in. Chorus's literature review (council topic
`chorus-probe-beats-model-2026-11-07`, #4) put audio speaking style first and named the trap: much of a reported
probe-over-output gap is calibration. So this part measures the gap after calibration. It edits nothing.

## Question

On held-out speakers and held-out sentences of the Expresso read-speech set, how well do these readouts name the speaking
style (seven acted styles): the model's own verdict as cosine with one text label per style, the same verdict tuned on the
training speakers (phrase ensembles and per-style bias), a linear probe on the final embedding, a probe on one
intermediate layer, and simple acoustic features? Is there a gap between the best probe and the best calibrated verdict
large enough to be worth an edit?

## Prior art and existing data

- Koduru et al., *Heard but Not Heeded* (arXiv 2609.00727, 2026): late-encoder linear probes on seven Expresso styles reach
  about 82 to 85% accuracy where audio LLMs' generated verdicts reach 19.7 to 53.7%. Different models, and generated
  answers rather than embedding cosine.
- Ando et al., *SLAP* (arXiv 2510.01860, 2025): on one frozen audio-text model, a linear probe beats zero-shot label
  cosine on voice attributes, 65.2 against 53.3 F1. The closest comparison to ours; they note better label phrasings may
  shrink the gap.
- Villuri et al. (arXiv 2609.04582, 2026): an internal probe at 0.964 AUC against 50% answer accuracy, but a scalar bias
  correction restored 81%, so most of that gap was calibration. Hence the calibrated arm and the reading rule below.
- Our own: Part 188 recorded EmbeddingGemma 2's 49,152 MLP inputs and final embedding for animal sounds with
  `part188/record188.py` and a parity gate (`part188/parity_ref.npz`); Part 190 found its output-embedding probe ahead of
  the model's own verdict for real against imitated animal sounds (0.918 and 0.965 against 0.630 and 0.808), which is
  the same shape of question on a different task.
- Data: Expresso read speech (`ylacombe/expresso`, config `read`, 11,615 utterances, four speakers, eight styles,
  CC-BY-NC-4.0), downloaded to the owner's disk; audio is never committed. Before registration only metadata of the
  first shard was read (speaker, style, text and id of 968 rows, all speaker ex01): some texts are read in several
  styles by one speaker, which the text-half split below handles. No audio was decoded and no model was run.

## Method

- Model: `google/embeddinggemma-2` at revision f9972ac22e696a105c482202646979e33e4b5a1e, loaded and recorded exactly as
  `part188/record188.py` does (float32, CPU), including its parity gate. Units: the input to each layer's
  `mlp.down_proj`, 24 layers by 2,048, mean over positions; and the final normalized 768-d embedding.
- Styles: confused, default, enunciated, happy, laughing, sad, whisper. Narration is left out: it is a reading genre
  rather than a voice quality and has only 21 minutes. Rows labelled `emphasis` and `essentials` (present in the
  parquet files though not in the dataset card's table) are left out too: they are reading tasks, not styles.
- Clips: for each speaker and style, 100 utterances chosen by a seeded draw (seed 196) from those of 2 to 10 seconds, or
  all of them if fewer. Audio decoded to mono 16 kHz, the whole utterance (at most 10 s).
- Splits: four outer folds, one per held-out speaker. Within each fold, texts are split into two halves by a seeded hash
  of the text (seed 196): training uses the three other speakers' clips whose text is in half 1, testing uses the held-out
  speaker's clips whose text is in half 2, so no test clip shares a speaker or a sentence with training. Every
  reported number pools the four folds' test clips.
- Text labels: the 56 phrasings in `part196/phrasings.json`, eight per style, written before any audio was recorded,
  encoded with `prompt_name='SearchQuery'` as in Parts 188 and 190.
- Readouts, all scored on the same test clips:
  - A, single label: argmax over styles of the cosine with each style's first phrasing.
  - B, calibrated verdict: per style, either the mean cosine over its eight phrasings or its single best phrasing (chosen
    on the training clips), plus a per-style additive bias; the choice and the seven biases are fitted on the training
    clips to maximise balanced accuracy (biases by multinomial logistic regression on the seven scores with one shared
    scale). This is the model's own verdict, only re-weighted.
  - B56, descriptive: multinomial logistic regression on all 56 cosines, the most a label-space readout can do.
  - C, exemplar centroids: nearest training-style centroid of final embeddings by cosine.
  - D, final-embedding probe: multinomial logistic regression on the 768-d embedding, standardized on training, L2 with
    C = 1.0 (scikit-learn convention), as Part 188.
  - E, layer probe: the same on one layer's 2,048 mean MLP inputs, the layer chosen by inner leave-one-speaker-out
    balanced accuracy within the three training speakers.
  - F, single units: for each style, the unit with the largest one-against-rest training AUC, signed on training.
  - H, acoustic baseline: logistic regression on mean and spread of pitch (pyworld harvest), RMS energy, spectral
    centroid, and duration per word.
- Metrics: seven-way balanced accuracy for A, B, B56, C, D, E and H; macro one-against-rest AUC for every readout
  including F. Intervals: 95% bootstrap over test texts (clusters), 2,000 draws, seed 196, for each readout and for the
  paired difference D minus B.

## Controls

- Gate 1 (data): every speaker by style cell has at least 40 clips after the length filter; no test clip shares a speaker
  or a text with its fold's training clips (checked in code, per fold).
- Gate 2 (recorder): Part 188's parity gate passes before any Expresso clip is recorded (the eight registered parity
  clips reproduce `parity_ref.npz`).
- Gate 3 (positive control): a speaker-identity probe on the final embedding, trained on text half 1 and tested on text
  half 2 of the same four speakers, reaches at least 90% balanced accuracy. Fails if the recording or the probe code is
  broken.
- Gate 4 (negative control): D trained on style labels shuffled within each training speaker scores at most 25% balanced
  accuracy pooled (chance 14.3%). Fails if the split leaks.
- Matched comparisons: A, B, C and D use the same final embedding; B and D see the same training clips; E's layer is
  chosen without the test speaker.
- Not used: readers (labels are the dataset's), generations, the answer-polarity split, edits (a later part, if the gap
  survives).

## Size

- Smallest effect worth knowing: a gap of 5 points of balanced accuracy between D (or E) and B.
- Total: 2,800 clips recorded once, about 45 minutes on the CPU (recording about 32 minutes at the measured rate, probes
  and bootstraps a few minutes). Four speakers by seven styles by 100 clips; cells with fewer than 100 use all they have.
- Pilot: none; the gates stop the part before any readout is scored if the data or recorder fail.

## Optimization

- Bound resource: CPU, and Windows RAM shared with any A770 job.
- Measured: the chicken study's recorder (the same model and code path) handled 631 clips in 427.0 s (1.48 clips/s) on
  this machine (`record.log`), so 2,800 clips take about 32 minutes.
- Levers: 100 clips per cell rather than all 11,615, taken; 10 s cap per clip, taken; only the MLP inputs and final
  embedding recorded, taken; the recording runs only when no A770 job is running, taken (a WSL job beside one tripped its
  RAM guard on 2026-11-08); probes on CPU in seconds, no lever needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 196.1 | A, the single-label verdict, has pooled balanced accuracy at most 35%. | medium: zero-shot style labels were weak in SLAP and Koduru et al. | Claude |
| 196.2 | D, the final-embedding probe, has pooled balanced accuracy at least 70%. | medium: other encoders reach 82 to 85%, but four-speaker held-out testing is harder. | Claude |
| 196.3 | B, the calibrated verdict, closes at least half of the gap from A to D. | medium-low: Villuri et al. found calibration closed most of a probe gap. | Claude |
| 196.4 | D minus B is at least 5 points of balanced accuracy, with its 95% bootstrap interval above 0. | low-medium: the headline, and the case for a later edit. | Claude |
| 196.5 | E, the best intermediate layer chosen without the test speaker, beats D by at least 3 points. | medium-low: style cues may be thinned by mean pooling into the final embedding. | Claude |
| 196.6 | Whisper has the highest per-style recall under A, B and D. | medium: whisper is acoustically the most distinct style. | Claude |
| 196.7 | H, the acoustic baseline, reaches at least 40% balanced accuracy. | medium. | Claude |

## Reading rules

- Gates 1 to 4 must pass before any readout is read.
- The headline is 196.4. If D minus B is under 5 points or its interval includes 0, the result is reported as
  "calibration closes the gap" and no edit part follows from it. A held 196.4 is reported as knowledge the calibrated
  verdict does not use, on this model and this set, not as knowledge the model lacks.
- B56 is descriptive: it uses the label space as a 56-feature probe, so it sits between a verdict and a probe and is
  reported for the reader to place, not as either.
- Balanced accuracy is compared with balanced accuracy and AUC with AUC, never across.
- Per-fold numbers are reported beside the pooled ones; with four speakers one odd speaker can move a pooled number.

## Limits

- Four speakers, acted read speech, one recording studio: style here is performed and clean, and the result does not
  extend to spontaneous speech or other speakers without a new test.
- The phrasings are ours; a different label set could move A and B. Writing eight per style and tuning on training
  speakers is the defence, not a guarantee.
- One model, one revision, CPU float32.
- The dataset is licensed for non-commercial use; nothing from it is redistributed.

## Changes before registration

- None yet. The registration time remains `Registered <time from date> CT`; it is stamped from `date` at commit.
