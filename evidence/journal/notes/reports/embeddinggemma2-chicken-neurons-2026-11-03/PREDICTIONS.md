# Chicken neurons in EmbeddingGemma 2: predictions (written 2026-11-03 11:08 CT, before any activation is recorded)

Elise, 11:06 CT: find the neurons responsible for the chicken classification, so the Chicken-o-Meter can read a neuron.
Prior art (Muse, council muse-embeddinggemma2-2026-11-03 #3): no single-neuron study of a unified multimodal embedding
model was found; closest are CLIP's multimodal neurons (Goh et al. 2021, text-in-image only) and Neuron Lens on Gemma
12B (on file, 2026-10-30, read from a screenshot).

**Model.** google/embeddinggemma-2 (snapshot f9972ac2), CPU float32. Neurons = the 2,048 MLP units in each of the 24
shared layers (the input to `mlp.down_proj`), 49,152 in all; every modality passes through these layers.
**Unit of measurement.** Per clip, each neuron's mean activation over all sequence positions, one clip per forward pass.
**Data.** ESC-50 (Piczak 2015, CC BY-NC 3.0): all 40 hen and 40 rooster clips ("chicken"), and 2 clips per fold of
each of the other 48 classes ("other", 480). Discovery = folds 1 to 3, test = folds 4 and 5 (ESC-50's folds keep clips
from one source recording together).
**Selection.** Per neuron, discovery AUC for chicken against other; the top neuron and the top 10 by discovery AUC
(either sign: a neuron that is lower for chickens counts, with its sign recorded).
**Controls.** (a) Selection null: shuffle the discovery labels 20 times, select the top neuron each time, report its
test AUC. (b) The model's own embedding classifier: cosine to "a hen clucking" and "a rooster crowing" minus the mean
cosine to one prompt per other class, test AUC.
**Cross-modal.** 20 short sentences about chickens and 20 about other ESC-50 sounds (written before the run, in
`text_items.json`), each neuron's mean activation over tokens; text AUC for the selected neurons.

| # | Prediction | Confidence |
|---|---|---|
| N1 | The top discovery neuron's test AUC is at least 0.85. | medium |
| N2 | Every selection-null run gives a test AUC under 0.65. | high |
| N3 | The top neuron is in layers 12 to 23 (the upper half). | low |
| N4 | At least one of the top 10 audio neurons separates chicken sentences from other sentences with AUC at least 0.8, same sign. | low |
| N5 | The embedding classifier's test AUC is at least 0.9. | medium |
| N6 | The top neuron's test AUC is within 0.05 of the embedding classifier's. | low |
| N7 | On the top neuron, Elise's second egg-song imitation scores above the median ESC-50 hen clip. | low |

Reading rule: N1 with N2 says a single neuron carries a chicken signal that the selection alone cannot explain. N4 is
the multimodal question; failing it says the audio chicken neurons are audio-only.

## Image arm, added 2026-11-03 11:20 CT, before any image is embedded
Data: the 43 photos in sources/photos/hens/ against every other photo in sources/photos/ (the dog reference corpus,
the owl, the pond, the creek, Trixie, a breakfast sandwich): about 24. Same neurons, mean over all positions, one
image per forward pass, vision encoder loaded. Neurons tested: the top 10 audio neurons and the nine that passed audio
and text at 0.9 (README).

| # | Prediction | Confidence |
|---|---|---|
| I1 | L17 N1574 separates hen photos from the other photos with AUC at least 0.8, same sign as audio. | low |
| I2 | At least 5 of the nine audio-and-text neurons reach image AUC 0.7, same sign. | low |
| I3 | L15 N1126 (the best audio neuron) reaches image AUC 0.7, same sign. | low |

Confound named in advance: hen photos share coop and yard backgrounds and include Elise; the dogs are mostly a
different setting. A hit can be background as much as bird. Nothing here is published; the photos stay local.

## Pixabay arm, added 2026-11-03 11:30 CT, before any Pixabay audio is downloaded or recorded
Elise's suggestion (11:28 CT): Pixabay sound effects hold many chicken recordings and a fair number of human imitations,
all clean recordings, so they separate "real chicken" from "clean recording". Her own report at 11:29 CT, from trying it on the meter:
real sounds beat her imitations reliably on the neurons and most of the time on the meter.
Labels come from Pixabay's metadata only (name, description, tags, AI flag), assigned before recording: **real** (a real
bird, no word of imitation, cartoon, toy, synth or AI); **imitation** (metadata says a human makes the sound);
**ai** (Pixabay's isAiGenerated flag); everything else excluded (rubber chicken, cartoon, game effects, music, food,
ambiguous). One labeller (me), metadata only: a limit, stated.
Measure: AUC real (positive) against imitation for each neuron, with the signs fixed on ESC-50 (both positive).
- **X1** L15 N1126 separates real from imitation at AUC >= 0.75. Medium.
- **X2** The model's prompt classifier (analyze.py's score: best hen/rooster prompt minus the mean of the others)
  separates them less well than L15 N1126. Medium-low.
- **X3** The median AUC of the nine audio+text neurons is >= 0.70. Low.
- **X4** If at least 10 clips are flagged AI-generated: L15 N1126 ranks them above the imitations, AUC >= 0.60. Low.

### Amendment to the Pixabay labels, 2026-11-03 11:49 CT, before any recording
Metadata gave only four imitations. Elise identified four more by ear (DM 11:49 CT): 487676 "Chicken Sounds" and
487671 "Cartoon Chicken Bwak Bwak" (DRAGON-STUDIO), 86687 "questionable chicken" and 101726 "Chicken"
(freesound_community). 487676 had been labelled **real** from its metadata, so some "real" clips may be imitations
too: a limit on X1 to X4. Final counts: 180 real, 35 ai, 8 imitation, 82 excluded (pixabay_labels.csv, one reason per
row). Every clip is cut to its first 30 s. With 8 imitations an AUC moves by 1/8 per clip, so X1's 0.75 is read with a
bootstrap interval, not as a point.

## Hens removed from photos (Elise's idea, DM 09:47 CT), written 2026-11-04 09:48 CT before any edited image exists

Twelve hen photos chosen by Claude for having background in frame (yard, coop, roost, room), before any edit:
2026-03-19-all-five-coop, 2026-04-30-four-hens-fresh-grass, 2026-08-04-side-stairs-flock-clearest,
2026-08-04-side-stairs-flock-elise, 2026-08-29-flock-roost-all-five, 2026-08-29-flock-roost-wide,
2026-08-29-flock-roost-perch-pair, 2026-08-04-brahma-mama-low-angle, 2026-07-25-gandalf-followed-inside-desk,
2026-07-03-brahma-mama-supervises-the-ops-board, 2026-07-25-three-hen-lap-cohort-mix, 2026-04-20-gandalf-lap.
A Luna codex worker edits each with the image tool twice: `removed` (every chicken erased, the scene filled in) and
`kept` (the same tool asked to reproduce the photo with everything, hens included). `kept` is the control for the
editing itself: the chicken effect is kept minus removed, both having passed through the generator. Activations as
images.py (input to mlp.down_proj, mean over positions), each neuron read with its audio sign. A pair counts only if
the worker's own look confirms no chicken remains in `removed` and the hens remain in `kept`; dropped pairs are named.

- **I4** L17 N1574 (signed) is higher on `kept` than on `removed` in at least 10 of the usable pairs out of 12 (exact
  two-sided sign test p < 0.05 needs 10 of 12). Fails below. High.
  **Result, 2026-11-04:** held (11 of 11).
- **I5** The same for L15 N1126. Medium.
  **Result, 2026-11-04:** failed (7 of 11).
- **I6** Over all 49,152 neurons, L17 N1574's median signed kept-minus-removed difference ranks in the top 5% (so the
  drop is the chicken unit's, not something every unit does under the edit). Medium.
  **Result, 2026-11-04:** held (rank 225 of 49,152, top 0.46%).
- **I7** `kept` against the original photo moves L17 N1574 (signed) by less than half as much as `removed` does, in the
  median over pairs (the generator alone does not account for the drop). Medium.
  **Result, 2026-11-04:** held (0.006 against 0.144).

## Is it a chicken neuron or a "not a human voice" neuron? (VocalImitationSet), written 2026-11-04 10:18 CT before any download

Muse's prior art (council topic `muse-prior-art-imitation-edit-2026-11-04`, #3) named the confound: every imitation so
far is a human voice and every real clip is not, so L17 N1574 and L15 N1126 might separate real from imitation by voice
alone. VocalImitationSet (Kim, Ghei, Pardo and Duan, DCASE 2018; github interactiveaudiolab/VocalImitationSet) has human
imitations of many animals and real Freesound recordings of the same classes, which gives a 2 x 2: chicken or other
animal, real or human imitation. Audio is downloaded to /mnt/e/neuron-probe/vocalimitationset/ and not committed (the
originals carry per-file Freesound licences); the selection manifest is committed.

Selection, fixed by seed 20261104 before any download: all included imitations of classes 009 (Chicken, rooster: Cluck)
and 010 (Crowing); 4 included imitations from each of the other 28 animal classes 000 to 029; all original recordings
(reference and non-reference) of classes 009 and 010; 2 non-reference originals from each of the other 28 animal
classes. Clips cut to their first 30 s. Neurons, signs and the prompt classifier exactly as in analyze_pixabay.py (signs
from ESC-50 discovery folds, both positive). AUCs with 4,000-draw bootstrap intervals; a prediction is read on the point
AUC against its bar, interval reported beside it.

- **V1** L17 N1574 ranks chicken imitations above other-animal imitations, AUC >= 0.65 (chicken information survives in
  a human voice). Low.
  **Result, 2026-11-04:** failed (0.587).
- **V2** L17 N1574 does NOT separate real other animals from their imitations: AUC < 0.65 (it is not a real-sound or
  not-human-voice detector). Medium.
  **Result, 2026-11-04:** held (0.617).
- **V3** L17 N1574 ranks real chickens above real other animals, AUC >= 0.85 (the ESC-50 finding on a new set). High.
  **Result, 2026-11-04:** failed (0.764).
- **V4** L17 N1574 ranks real chickens above chicken imitations, AUC >= 0.85 (the Pixabay finding on a new set).
  Medium-high.
  **Result, 2026-11-04:** held (0.893).
- **V5** The prompt classifier ranks chicken imitations above other-animal imitations, AUC >= 0.80 (the model's verdict
  knows what is being imitated). Medium.
  **Result, 2026-11-04:** failed (0.621).
- **V6** On real chickens against chicken imitations, the prompt classifier's AUC is lower than L17 N1574's. Medium.
  **Result, 2026-11-04:** held (0.576 against 0.893).
- L15 N1126 and the nine audio+text neurons are reported for all four contrasts, with no prediction.

## Cutting the chicken out of a sound in time (Elise's go, DM 11:11 CT), written 2026-11-04 11:12 CT before any window is scored

The audio counterpart of the hen-removal photos: inside one recording, windows where a chicken is calling against
windows where it is not, so mic, room and background stay the same. Recordings: the 20 VocalImitationSet real chicken
recordings and the 180 Pixabay real chicken clips (first 30 s), cut into 2 s windows with a 1 s hop. An independent
labeller, not EmbeddingGemma: the AudioSet AST classifier (MIT/ast-finetuned-audioset-10-10-0.4593), score = its
"Chicken, rooster" plus "Fowl" probabilities. Within each recording, the top third of windows by that score are ON and
the bottom third OFF; the middle third is dropped; recordings with fewer than 6 windows are dropped. Neurons signed as
before. Loudness control: each ON window paired with the OFF window of nearest RMS in the same recording.
- **T1** L17 N1574 has a higher median on ON than on OFF windows in at least 75% of recordings. Medium. **Result: failed narrowly (73.9% of 115 recordings).**
- **T2** The same with RMS-matched pairs: ON above its matched OFF in at least 70% of recordings (median over pairs).
  Medium-low. **Result: failed (67.8%).**
- **T3** Over all 49,152 neurons, L17 N1574's median within-recording ON minus OFF difference (divided by each neuron's
  spread over all windows) ranks in the top 5%. Medium. **Result: failed (rank 4,466 of 49,152, top 9.1%).**
- **T4** L15 N1126: ON above OFF in at least 75% of recordings. Medium. **Result: held (78.3%).**

## Teaching the model to tell real hens from imitations (EDIT-DESIGN.md; Elise's go 11:11 CT), written 11:12 CT before any arm is fitted

Fit and test splits as EDIT-DESIGN.md (VocalImitationSet chicken imitators split in half by seed 20261104; test = the
other imitators' chicken imitations against the 20 VocalImitationSet real chickens; damage on ESC-50 folds 4 and 5).
Score = the prompt classifier on the (edited) embedding. Unedited held-out AUC is the baseline (V6 gave 0.576 on all
imitators).
- **E1** The written gated block raises the held-out AUC to at least 0.75. Medium (its gate opened on 56% of fit
  imitations, which caps the gain).
  **Result, 2026-11-04:** held (0.853); the k grid was widened after an exploratory run scored the test split (README).
- **E2** The written block leaves at least 95% of ESC-50 fold 4 and 5 non-chicken clips bit-identical, and changes
  per-class accuracy on folds 4 and 5 by at most 1 clip in every class. Medium.
  **Result, 2026-11-04:** failed (91.7% identical; no class moved by more than 1); the k grid was widened after an exploratory run scored the test split (README).
- **E3** The trained layer reaches a held-out AUC of at least 0.85. Medium.
  **Result, 2026-11-04:** held (0.993); the k grid was widened after an exploratory run scored the test split (README).
- **E4** The trained layer changes per-class accuracy on folds 4 and 5 by more than 1 clip in at least one class.
  Low-medium.
  **Result, 2026-11-04:** held (8 classes); the k grid was widened after an exploratory run scored the test split (README).
- **E5** The steering arm, scaled on the fit split to the written block's fit AUC, changes more fold 4 and 5 classes'
  accuracy than the written block does. Medium.
  **Result, 2026-11-04:** failed (steering 0 classes, written block 1); the k grid was widened after an exploratory run scored the test split (README).
- **E6** Controls gain less than 0.05 held-out AUC each: the written block with a random direction, the written block
  gated by two random same-layer neurons, and the trained layer fitted on shuffled real/imitation labels. High.
  **Result, 2026-11-04:** held (+0.008, +0.010, -0.030); the k grid was widened after an exploratory run scored the test split (README).
- Descriptive: the 8 Pixabay imitations and Elise's own takes, scored before and after each arm.
