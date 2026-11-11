# Part 203: does Part 196's "the final embedding loses style" hold on CREMA-D emotion?

Procedure: v1
Status: DRAFT, not registered. Predictions 203.1 to 203.7 are written (Claude, 2026-11-10); the timing fields below wait for the timing smoke.

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
    actor-grouped CV on the training actors only (the same five inner actor folds used for C, below), mean balanced
    accuracy at C = 1.0, ties to the earlier layer.
  - H, acoustic features: Part 196's seven (voiced fraction, F0 median and IQR by pyworld harvest, RMS mean and spread,
    spectral centroid mean, seconds per word).
- Splits: 91 actors into five folds by SHA-256 hex order of the actor id (round-robin over that order), each fold's
  actors held out entirely. Training is the other four folds' actors.
- Probes: multinomial logistic regression on features standardised on the training rows (scikit-learn pipeline,
  lbfgs, max_iter 5000, class_weight balanced).
  - `V_grid`, `D_grid`: C chosen by inner actor-grouped CV on the training actors over Venn's grid
    {0.01, 0.1, 1, 10, 100}, lowest mean held-out-actor log loss, ties to the lower C.
  - Inner folds, for both C and E's layer: the outer fold's training actors split into five inner folds by SHA-256 hex
    order of the actor id, round-robin, each inner fold's actors held out entirely.
  - `V_p196`, `D_p196`, `E_p196`, `H_p196`: Part 196's fixed C = 1.0, so regularisation can be isolated as a separate
    row. E and H are scored at C = 1.0 only, as in Part 196.
- Three-actor ablation: per fold, five fixed draws of three training actors (SHA-256 order, rotating start), against all
  training actors, for V (C by inner CV on the draw's actors) and E (C = 1.0, the fold's selected layer).
- Metrics: six-way balanced accuracy and macro one-against-rest AUC within each held-out fold; primary numbers are the
  mean over the five held-out folds. Intervals: 95% bootstrap, 2,000 draws, seed 203, resampling actors within each
  held-out fold and recomputing the five-fold mean, for each readout's balanced accuracy and for the paired differences
  in balanced accuracy. AUC is reported as a point estimate only, with no interval.

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

- Smallest effect worth knowing: 5 points of balanced accuracy on the six-way readout (chance 16.7). Smaller gaps between
  readouts would not change which explanation of 25.8% against 56.5% we act on.
- Clips: 2,730 planned, 91 actors x 6 emotions x 5 clips.
- Total: 2,730 clips, about {{CLAUDE: minutes}} minutes (Part 196's measured 0.68 s per 5-second clip, scaled by
  CREMA-D clip length; CREMA-D clips are short, so this is an estimate until a timing smoke is run).
- Probes, bootstraps and pitch features: measured by the design review on synthetic records at full size, everything
  except the layer choice ran in 57.7 s; one six-class probe at the layer-choice shape took 1.6 to 3.8 s. The layer choice
  is 24 layers x 5 inner folds x 5 outer folds = 600 fits, about 16 to 38 minutes on one core. (The first draft chose the
  layer by leave-one-actor-out, about 8,640 fits and 4 to 9 CPU-hours; the five inner folds replace it, see Changes.)
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
| 203.1 | `readouts.V_grid.mean_balanced_accuracy` is at least 0.45, and its `balanced_accuracy_actor_cluster_95` lower bound is above 1/6. Emotion in CREMA-D is acted and loud; with 91 actors the final audio vector should carry it, near Venn's 56.5% though not matched to it. | 70% | Claude |
| 203.2 | The text marker and V's float16 storage together do not matter: `readouts.D_grid.mean_balanced_accuracy` is within 0.03 of `readouts.V_grid.mean_balanced_accuracy`. V and D also differ in decoder and cap, but the design review measured the two decoders bit-identical on a 16-bit 16 kHz mono WAV, and CREMA-D clips are 2 to 3 s, so D's 10 s cap never bites. This prediction does not separate the marker from the float16 storage. I expect the encoder to place the audio the same way with or without `<|audio|>`. | 60% | Claude |
| 203.3 | Regularisation matters: `V_grid_minus_V_p196.mean_difference` is at least +0.03, with Venn's grid choosing C of 10 or 100 on most folds (its log loss was still falling at 100). | 55% | Claude |
| 203.4 | An intermediate layer at least ties the final vector here: `E_p196_minus_D_p196.mean_difference` is at least -0.02 (both recorded as Part 196 recorded them, both at C = 1.0, same clips). Part 196's 37-point E over D was on four speakers; with 91 I expect the gap to shrink, not to reverse by much. | 60% | Claude |
| 203.5 | The seven acoustic features carry part of it: `readouts.H_p196.mean_balanced_accuracy` is between 0.30 and 0.45 (pitch and loudness separate anger, sadness and neutral; disgust and fear are hard without spectral detail). | 65% | Claude |
| 203.6 | Few speakers is most of Part 196's story: `three_actor_ablation.V.three_actor_draws_mean` is at least 0.10 below `three_actor_ablation.V.all_training_actors_mean`. | 65% | Claude |
| 203.7 | And it hurts the final vector more than the intermediate layer: V's drop in 203.6 exceeds E's drop (E's all-training mean minus E's three-actor mean) by at least 0.05. | 50% | Claude |
| 203.W1 | `readouts.V_grid.mean_balanced_accuracy` minus `readouts.D_grid.mean_balanced_accuracy` is more than 0.03 (D falls short of V; against 203.2). Falsifier as written: D within 0.03 of V. | 0.55 | Weaver |
| 203.W2 | `E_p196_minus_D_p196.mean_difference` is at least 0.10 (E beats D, not merely ties; against 203.4). Falsifier as written: E within 0.05 of D. | 0.50 | Weaver |
| 203.W3 | `readouts.H_p196.mean_balanced_accuracy` is under 0.30 (against 203.5's floor). Falsifier as written: H at 0.35 or above. | 0.30 | Weaver |
| 203.W4 | `V_grid_minus_H_p196.mean_difference` is under 0.10. Falsifier as written: V minus H at 0.15 or more. | 0.40 | Weaver |

Weaver's rows (203.W1 to W4) were written 2026-11-11 00:15 CT in #council-room (messages 1569958778877640725,
1569958779682947197, 1569958780634927105) from the question as I described it, before Weaver could read this file, and
Weaver asked that this be said beside them. Each is scored by its forecast: it holds if the forecast holds and fails
otherwise; the falsifier is recorded as written and reported separately, so a value between forecast and falsifier
(e.g. H between 0.30 and 0.35) is "forecast failed, falsifier not reached".
Weaver gave the confidences at 00:17 CT (message 1569959314490396694), with a calibration note to carry beside them:
across Parts 173 to 182 their falsifiers held and their forecasts mostly did not (about three clear holds against a
dozen failures), so they ask readers to shade each of these four down by about 0.1, and name W3 as the row they expect
to fail. The table keeps the numbers as given.
Weaver, 00:18 CT (message 1569959521860849756), after reading the mirror: W1's rationale lost half its support, since
V_grid and D_grid are tuned over the same grid, which leaves the marker and float16 storage, where they expect an effect
near zero. They keep W1's forecast and 0.55 as written, because the row was written blind to the design, and ask that
the partly wrong reasoning be recorded rather than the forecast amended. They also read W4's tuning asymmetry as
sharper than they first wrote (V grid-tuned, H at C = 1.0).

Quantities a prediction can be written about, each named by its key in `results.json`: `readouts.V_grid`,
`readouts.D_grid`, `readouts.V_p196`, `readouts.D_p196`, `readouts.E_p196`, `readouts.H_p196` (each
`mean_balanced_accuracy`, `balanced_accuracy_actor_cluster_95`, `mean_macro_ovr_auc`); the paired differences
`V_grid_minus_E_p196`, `V_grid_minus_H_p196`, `D_p196_minus_V_grid`, `E_p196_minus_V_grid`, `E_p196_minus_D_p196`,
`V_grid_minus_V_p196`, `D_grid_minus_D_p196` (each `mean_difference`, `actor_cluster_95`); `three_actor_ablation.V.three_actor_draws_mean`
and `.all_training_actors_mean`, and the same two for `E`.

## Reading rules

- V is "the final audio vector under a prompt carrying the `<|audio|>` marker", never "a layer". V minus D reads the
  marker together with float16 storage, decoder and cap; the layer contrast is E_p196 against D_p196 only, and E
  against V is not read as a layer difference. (Weaver, 2026-11-11, point 2.)
- Gates 1 and 2 must pass before any readout is read.
- Balanced accuracy is compared with balanced accuracy and AUC with AUC, never across.
- A gap between V and D is read only if both readouts' `balanced_accuracy_actor_cluster_95` lower bounds are above
  chance (1/6).
- Venn's 56.5% is reported as context only; no verdict compares our number with it as if the setups matched.

## Limits

- 91 actors, six acted emotions, one dataset; the result does not extend to spontaneous speech or other datasets.
- Five clips per cell by SHA-256 of the file name is a fixed rule, not a representative draw of CREMA-D.
- One model, one revision, CPU float32.
- The dataset is licensed for non-commercial use; nothing from it is redistributed.

## Changes before registration

- 2026-11-10, after the DeepSeek design review (`reviews/cheap-part203-design-r1-ds-ds-r1.md` in the Journal repo), before
  any CREMA-D audio was read (the review's download test stopped at its first file-name check and wrote nothing):
  - E's layer is chosen with five inner actor folds, the same split used for C, instead of leave-one-actor-out. The
    first draft did not name the inner fold count, and leave-one-actor-out was about 4 to 9 CPU-hours.
  - 203.4 is stated against `E_p196_minus_D_p196` (same clips, same C, both recorded Part 196's way). As first written
    against `E_p196_minus_V_grid`, it also carried the marker, float16 and regularisation differences.
  - 203.2's wording now says it measures the marker and V's float16 storage together.
  - The V/D reading rule names the interval's lower bound; intervals cover balanced accuracy only.
  - The three-actor ablation fits E at C = 1.0, as this design always said; the first build chose E's C by inner CV.
  - Code fixes from the same review (recorder Gate 1 write, download.sh's name check, the analyzer's V/D gate, the
    timing-smoke run, atomic writes, error handling) are in the commit after this list, each with a test shown red on
    the old code (41 tests). One more, found by the fold worker: the log loss used to choose C clipped probabilities
    at 1e-12 without renormalising, so rows could sum slightly above 1; it now renormalises after the clip. This
    changes loss values only at the floor, not the rule.
  - Not changed: Part 196's recorder still crashes on a truncated record instead of retrying it (`_cached_status`);
    Part 196 is published and is not edited for this part.
