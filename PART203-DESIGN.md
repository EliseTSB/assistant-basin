# Part 203: does Part 196's "the final embedding loses style" hold on CREMA-D emotion?

Procedure: v1
Status: DRAFT, not registered. Predictions are `{{CLAUDE: predictions}}` and are written by Claude before registration.

## Question

Part 196 (Expresso, four speakers, seven acted styles, four folds each holding out one speaker) found a linear probe on
EmbeddingGemma 2's FINAL embedding (D) at 25.8% balanced accuracy, a probe on layer 0's mean MLP input (E) at 62.7%,
and seven acoustic features (H) at 51.8%. Venn (github.com/markrussinovich/venn-vdm) reports a linear head on
EmbeddingGemma 2's final AUDIO vector at 56.5% on CREMA-D emotion (six classes), held-out actors, 1,716 clips, C = 100
chosen from {0.01, 0.1, 1, 10, 100} with log loss still falling at 100.

The estimand is: within CREMA-D's first five clips per (actor, emotion) cell by SHA-256 of the file name, on 91 actors
split into five actor-disjoint folds, how well does a multinomial linear probe on the final embedding name the emotion,
and how does that compare with a probe on the same final embedding recorded the way Part 196 recorded it, a probe on one
intermediate layer, and the acoustic baseline? The question is whether the two setups differ in a way that explains the
25.8% against 56.5%, or whether the difference lies in the data, the folds, the regularisation or the vector.

Venn's exact clips and split are NOT recoverable: `part203/prior/sound-eval.json` names one clip per class and its
selection rule ("IEO sentence, emotions ANG/HAP/NEU/SAD; lexicographically first held-out filename for each"), and the
rows carry class indices only. This is our own design on the same dataset and task; Venn's 56.5% is context, never a
matched comparison.

## Prior art and existing data

- Venn, `part203/prior/sound-eval.json` (read 2026-11-10): family (head) accuracy 0.5647, 95% CI 0.5321 to 0.6006;
  probe 0.5670; difference -0.0023, CI -0.0076 to 0.0023; chosen C 100 of {0.01, 0.1, 1, 10, 100}, log loss by C
  {0.01: 1.779, 0.1: 1.708, 1.0: 1.479, 10.0: 1.251, 100.0: 1.196}; Jev transcripts 0.1463; chance 0.1667; 1,716 clips.
- Venn's audio vector, from its own code `part203/prior/venn-server.py`: `_decode_media` (lines 346-369) decodes the
  WAV with soundfile, requires mono 16 kHz and 1 to 480,000 frames (30 s), and returns
  `{"text": "<|audio|>", "audio": [samples]}`; `embed_media` (lines 371-425) calls
  `media_model.encode(items, batch_size=len(batch), normalize_embeddings=True)` with no `prompt_name`, stores float16
  (line 414), and re-normalises the float32 vectors in float64 (line 425). The model is
  `google/embeddinggemma-2` at revision f9972ac22e696a105c482202646979e33e4b5a1e, CPU float32
  (`part203/prior/venn-reader.py:14-15`, `venn-server.py:406-407`).
- Our own: Part 196 (Expresso, seven styles) and its recorder `part196/record196.py`, reused here.
- Data: CREMA-D (`https://github.com/CheyneyComputerScience/CREMA-D`; Cao, Cooper, Keutmann, Gur, Nenkova and Verma,
  2014; ODbL 1.0 / DbCL 1.0), 2,730 clips: the first five by SHA-256 hex of the file name within each of the 91 x 6
  (actor, emotion) cells of the public file list `part203/prior/cremad-audiowav.tsv`. `part203/download-list.txt` is
  that list; `select203.py` regenerates it, and a test checks the two agree. Audio is never committed.

## Method

- Model: `google/embeddinggemma-2` at revision f9972ac22e696a105c482202646979e33e4b5a1e, loaded and recorded exactly as
  `part196/record196.py` does through `part188/record188.py` (float32, CPU, 6 threads), including Part 188's parity gate.
- Readouts, each recorded per clip:
  - V, Venn's audio vector: `record203.decode_audio_v` (soundfile, mono 16 kHz, at most 30 s, refusing what Venn
    refuses), `record203.v_item` (`{"text": "<|audio|>", "audio": [samples]}`), `record203.embed_v`
    (`encode([item], batch_size=1, normalize_embeddings=True)`, stored float16).
  - D, Part 196's final embedding: `record203.embed_d`, which is `part196/record196.py:345-347`'s
    `model.encode({"audio": {"array": audio, "sampling_rate": 16000}}, normalize_embeddings=True)` on Part 196's decode
    (`part196/record196.py:186-195`, ffmpeg mono 16 kHz f32le, capped at 10 s).
  - E, the mean MLP input of one layer: 24 layers x 2,048 units recorded as in Part 196; the layer is chosen by inner
    actor-grouped CV on the training actors only, mean balanced accuracy, ties to the earlier layer.
  - H, acoustic features: Part 196's seven (voiced fraction, F0 median and IQR by pyworld harvest, RMS mean and spread,
    spectral centroid mean, seconds per word).
- Splits: 91 actors into five folds by SHA-256 hex order of the actor id (round-robin over that order), each fold's
  actors held out entirely. Training is the other four folds' actors.
- Probes: multinomial logistic regression on features standardised on the training rows (scikit-learn pipeline,
  lbfgs, max_iter 5000, class_weight balanced).
  - `V_grid`, `D_grid`: C chosen by inner actor-grouped CV on the training actors over Venn's grid
    {0.01, 0.1, 1, 10, 100}, lowest mean held-out-actor log loss, ties to the lower C.
  - `V_p196`, `D_p196`, `E_p196`, `H_p196`: Part 196's fixed C = 1.0, so regularisation can be isolated as a separate
    row. E and H are scored at C = 1.0 only, as in Part 196.
- Three-actor ablation: per fold, five fixed draws of three training actors (SHA-256 order, rotating start), against all
  training actors, for V (C by inner CV on the draw's actors) and E (C = 1.0, the fold's selected layer).
- Metrics: six-way balanced accuracy and macro one-against-rest AUC within each held-out fold; primary numbers are the
  mean over the five held-out folds. Intervals: 95% bootstrap, 2,000 draws, seed 203, resampling actors within each
  held-out fold and recomputing the five-fold mean, for each readout and for the paired differences.

## Controls

- Gate 1 (analyzer, stops the part): every planned clip recorded with status ok and finite V and D; five folds; no
  actor in two folds; every (actor, emotion) cell holds exactly five planned clips. A clip that failed to record is
  named and counted; it is not replaced.
- Gate 2 (recorder, stops the part): Part 188's parity gate passes before any CREMA-D clip is recorded.
- The recorder's own Gate 1: every planned clip recorded, and V's and D's inputs are the same samples on every clip
  (`v_same_samples`); a clip whose 10 s cap truncates D below V's input is reported, not silently dropped.
- Matched comparisons: V, D, E and H are fitted on the same training clips and scored on the same held-out actors; the
  layer for E and every C are chosen without the held-out fold.
- Not used: readers (labels are the dataset's), generations, edits.

## Size

- Smallest effect worth knowing: {{CLAUDE: smallest effect}} (points of balanced accuracy on the six-way readout).
- Clips: 2,730 planned, 91 actors x 6 emotions x 5 clips.
- Total: 2,730 clips, about {{CLAUDE: minutes}} minutes (Part 196's measured 0.68 s per 5-second clip, scaled by
  CREMA-D clip length; CREMA-D clips are short, so this is an estimate until a timing smoke is run).
- Probes, bootstraps and pitch features: a few minutes on CPU.
- Pilot: none; the gates stop the part before any readout is read if the data or the recorder fail.

## Optimization

- Bound resource: CPU, and Windows RAM shared with any A770 job.
- Measured: {{CLAUDE: measured throughput, e.g. "N clips in M s" from the owner's --limit 20 timing smoke}} (no
  embedding has been recorded in this build; the recorder logs `elapsed_seconds` and `clips_processed`, and
  `part203/steps.txt` step 3 gives the smoke command).
- Levers: 2,730 clips rather than 7,442, taken; V and D recorded from one decode where the caps allow, declined because
  the two decoders differ by design and Gate 1 compares their samples; probes on CPU in seconds, no lever needed.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 203.1 | {{CLAUDE: predictions}} | | |
| 203.2 | {{CLAUDE: predictions}} | | |
| 203.3 | {{CLAUDE: predictions}} | | |
| 203.4 | {{CLAUDE: predictions}} | | |
| 203.5 | {{CLAUDE: predictions}} | | |
| 203.6 | {{CLAUDE: predictions}} | | |
| 203.7 | {{CLAUDE: predictions}} | | |

Quantities a prediction can be written about, each named by its key in `results.json`: `readouts.V_grid`,
`readouts.D_grid`, `readouts.V_p196`, `readouts.D_p196`, `readouts.E_p196`, `readouts.H_p196` (each
`mean_balanced_accuracy`, `balanced_accuracy_actor_cluster_95`, `mean_macro_ovr_auc`); the paired differences
`V_grid_minus_E_p196`, `V_grid_minus_H_p196`, `D_p196_minus_V_grid`, `E_p196_minus_V_grid`, `V_grid_minus_V_p196`,
`D_grid_minus_D_p196` (each `mean_difference`, `actor_cluster_95`); `three_actor_ablation.V.three_actor_draws_mean`
and `.all_training_actors_mean`, and the same two for `E`.

## Reading rules

- Gates 1 and 2 must pass before any readout is read.
- Balanced accuracy is compared with balanced accuracy and AUC with AUC, never across.
- A gap between V and D is read only if both intervals are above chance (1/6).
- Venn's 56.5% is reported as context only; no verdict compares our number with it as if the setups matched.

## Limits

- 91 actors, six acted emotions, one dataset; the result does not extend to spontaneous speech or other datasets.
- Five clips per cell by SHA-256 of the file name is a fixed rule, not a representative draw of CREMA-D.
- One model, one revision, CPU float32.
- The dataset is licensed for non-commercial use; nothing from it is redistributed.

## Changes before registration

- {{CLAUDE: changes}}
