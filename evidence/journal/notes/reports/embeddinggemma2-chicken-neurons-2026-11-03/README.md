# Chicken neurons in EmbeddingGemma 2 (2026-11-03)

Elise asked whether a neuron does the chicken classification. **Yes: single MLP neurons in the shared layers separate
chicken sounds from 48 other ESC-50 classes on held-out folds about as well as, or better than, the model's own
embedding, and several of the best also fire for sentences about chickens.** Predictions: PREDICTIONS.md, committed
before any activation was recorded. Scripts: record.py, analyze.py; numbers in results.json.

Data: ESC-50 hen and rooster clips (80) against 2 clips per fold of each other class (480); discovery folds 1 to 3
(48 chicken, 288 other), test folds 4 and 5 (32, 192). 49,152 neurons (24 layers x 2,048), mean over positions.

## Result
- **Best neuron: layer 15, neuron 1126.** Discovery AUC 0.967, **held-out AUC 0.943**. The model's own classifier
  (cosine to "the sound of a hen/rooster" against the other 48 class prompts) gets 0.891 on the same clips.
- **Selection null:** 20 label-shuffled selections gave held-out AUCs from 0.10 to 0.73, median 0.505, none at 0.85.
- **It crosses senses.** Of the 100 best audio neurons, 9% also separate 20 chicken sentences from 20 other sentences
  at AUC 0.9 or more with the same sign; across all 49,152 neurons the rate is 0.6%. Nine neurons reach 0.9 on both
  held-out audio and text: L15 N436, L15 N1281, L16 N134, L16 N349, L17 N1297, **L17 N1574** (audio 0.939, text
  0.963), L19 N1129, L20 N1260 (fires less for chickens), L23 N1262. All sit in layers 15 to 23.
- **Elise's imitations do not light the top neuron** the way the embedding's prompt scores suggested: her second egg
  song is below the median ESC-50 hen clip on L15 N1126 (-0.009 against 0.032), while the Commons egg song and both
  Commons roosters are above it. Real purrs and every synthetic purr sit low.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| N1 | top neuron held-out AUC >= 0.85 | 0.943 | held |
| N2 | every null run under 0.65 | max 0.73 | failed (but none near 0.943) |
| N3 | top neuron in layers 12 to 23 | layer 15 | held |
| N4 | one of the top 10 also separates sentences at >= 0.8 | L17 N1574, 0.963 | held |
| N5 | embedding classifier >= 0.9 | 0.891 | failed |
| N6 | top neuron within 0.05 of the embedding classifier | +0.052 (the neuron is better) | failed |
| N7 | Elise's egg song above the hen median on the top neuron | -0.009 against 0.032 | failed |

Limits: one model, one dataset (ESC-50 clips are short and mostly clean field recordings), 32 held-out chicken clips.
The sentences were written by me before the run and name the animal, so a text hit may be the word "hen" as much as the
idea; the base-rate comparison above is exploratory, chosen after N4 was read. Images are not tested yet.

## How specific are they? (Elise asked, 11:35 CT)
Not very. Per-class means on the ESC-50 subset (40 clips each for hen and rooster, 10 for each other class):
- **L15 N1126:** rooster +0.035 and hen +0.030 top the 50 classes, then crying baby +0.021, dog +0.019, crow +0.016,
  siren +0.012. Lowest: keyboard typing, mouse click, clock alarm. Single non-chicken clips above the chicken means: a
  creaking door (0.048), chirping birds (0.042), breathing (0.039). On sentences its top six are all chicken sentences,
  but "The rooster's crow woke the farmhouse." is among its three lowest.
- **L17 N1574:** hen +0.088 and crying baby +0.087 tie, then chirping birds +0.077, rooster +0.072, dog +0.060, frog,
  pig, crow. On sentences, rooster and hen sentences lead, then "A crow cawed from the power line."
- **L15 N436:** rooster, hen, dog +0.052, crying baby, breathing, laughing.
Reading: an animal-call (or voiced-call) unit leaning toward chickens, not a chicken unit. Ten clips per class and 40
sentences bound what "else" can mean here; nothing outside ESC-50's classes was tested.

## Real chickens against human imitations (Pixabay, Elise's idea, 11:28 CT)
180 real clips, 35 Pixabay-flagged AI-generated, 8 human imitations (4 from Pixabay's tags, 4 by Elise's ear), first
30 s of each, same neuron definition and ESC-50 signs. Predictions X1 to X4 were committed before any audio was
downloaded; all four held. Scored by `analyze_pixabay.py` -> `pixabay_results.json`.

| | real over imitation (AUC, 95% bootstrap) | AI over imitation | real over AI |
|---|---|---|---|
| L15 N1126 | 0.943 (0.89 to 0.98) | 0.968 | 0.416 |
| L17 N1574 | 0.991 (0.98 to 1.00) | 1.000 | 0.313 |
| median of the nine audio+text neurons | 0.915 | | |
| the model's own prompt classifier | 0.547 (0.31 to 0.78) | 0.579 | 0.476 |

- The chicken neurons tell real chickens from people doing chickens; the model's own verdict, the one the meter
  shows, cannot (chance). That matches Elise's report from the meter's mic at 11:29 CT.
- AI-generated chicken clips sit with the real ones, slightly above them on most neurons.
- Base rate: telling these imitations apart is not rare. Over all 49,152 neurons (ESC-50 signs), 7.1% reach 0.9 and
  3.5% reach 0.94. L17 N1574 ranks 69th of 49,152. Of the 100 best ESC-50 chicken neurons, the median is 0.767.
- Limits: 8 imitations, so one clip moves an AUC by 1/8 of the real-side mass; one of the eight was in the "real" pile
  from its tags, so more may hide there, which would lower these numbers; imitations may differ from field recordings
  in recording style as well as in who made the sound (a crying baby scores high on both neurons, so they are not
  simply anti-human).

## Same phone, real hens against her imitation (Elise's recordings, 12:04 CT)
Two quiet coop recordings on her phone (`sources/.../hen-purr-2026-09-06/coop-quiet-elise-2026-11-03{a,b}.m4a`; hens
"not particularly vocal", her words), scored on the meter. Percent of ESC-50 non-chicken clips beaten / percent of
ESC-50 chicken clips beaten:

| clip | meter percentile | L15 N1126 | L17 N1574 |
|---|---|---|---|
| coop a | 100 | 88.5 / 7.5 | 91.9 / 27.5 |
| coop b | 100 | 88.3 / 7.5 | 85.4 / 7.5 |
| her egg-song imitation a | 71 | 76.7 / 1.2 | 66.5 / 0.0 |
| her egg-song imitation b | 90 | 61.7 / 1.2 | 65.2 / 0.0 |

Same phone, same yard: the real hens beat her imitations on both neurons, even quiet. So the earlier worry, that the
neurons might only be detecting a clean field recording, does not explain this: her phone recordings of real hens
score high. The 09-06 purring burst, which scored low, was purring, which the model also misreads (it hears purrs as an
owl). Two clips each; a reading, not a measurement.
Elise's read (12:07 CT): rain, the door and her moving around are prominent in these takes, which may matter more than
how little the hens said. Checked by cutting each into 5 s pieces and scoring them on the meter: the best piece lifts
L15 N1126 from 7.5% to 32.5% of ESC-50 chicken clips beaten (coop b, piece 2), while L17 N1574's best piece reaches
11.2% (coop b, piece 4) against 27.5% for the whole of coop a. So the background explains part of the gap on L15 and
little on L17; ESC-50's curated clips, with the bird loud and in front, are a hard reference for any yard recording.

## Brahma Mama chatting in Elise's lap (12:26 CT)
`sources/.../hen-purr-2026-09-06/brahma-mama-chatty-lap-2026-11-03.m4a`, 36 s (meter reads the first 30 s). Meter:
chicken z 1.678, 100th percentile, above her egg song take 2 (the previous top); top prompts "a chicken clucking"
0.718, "a hen cackling" 0.712, "a hen purring" 0.709; nearest reference the 09-15 close purr of the same hen (0.933).
Neurons, whole take: L15 N1126 beats 84.4% of ESC-50 non-chicken clips and 5.0% of its chicken clips; L17 N1574 75.8% /
1.2%. Best 5 s window: L15 95.8 / 27.5, L17 87.3 / 10.0. The same split as the purrs: the model's embedding places soft
contented talk with hens; the two neurons, chosen on ESC-50's loud calls, rank it low among chickens.
Elise (12:29 CT): the take has a lot of background, mostly rain and garbage trucks, which the clip-wide mean carries into both neurons (best 5 s window 27.5% of ESC-50 chickens on L15 against 5.0% whole).

## Hen photos (images.py, run 2026-11-04 by 06:55 CT; predictions I1 to I3 committed 11-03 before any image was embedded)
Elise's 43 photos in `sources/photos/hens/` against the 23 other photos in `sources/photos/` (dogs, the owl, the pond,
the creek, Trixie, food), same neurons and definition as the audio, vision encoder loaded, CPU float32, 4 GB memory cap.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| I1 | L17 N1574 separates hen photos at AUC >= 0.8, audio sign | 1.000 | held |
| I2 | >= 5 of the nine audio-and-text neurons reach 0.7, audio sign | 9 of 9 (0.750 to 1.000) | held |
| I3 | L15 N1126 reaches 0.7, audio sign | 0.549 | failed |

**The null, after the fact (images_all.py, not registered, written after reading the table above).** Every one of the
49,152 neurons scored on the same split: 128 (0.26%) separate the two sets perfectly in one direction or the other,
1.1% reach 0.99, 10.9% reach 0.9 and 40.3% reach 0.75. So a perfect split is easy here, which fits the confound named
in advance (hen photos share coop and yard backgrounds and include Elise). Against that base rate the chicken neurons
still sit high: L17 N1574 and L19 N1129 are among the 128 perfect separators, L15 N1281 is in the top 0.36%, and seven
of the nine audio-and-text neurons rank in the top 5.2% of all neurons (about 0.5 expected by chance). The best audio
neuron, L15 N1126, ranks in the bottom 12%. Perfect separators cluster in layers 18 to 22 (2 to 3% of each layer,
under 1% before layer 18).

Reading: the units that call chicken sounds and chicken sentences also respond, in the same direction, to Elise's hen
photos, far more often than a random neuron does; that is cross-modal enrichment. It is not a chicken detector: a
photo set this easy cannot separate "hen" from "yard, feathers, Elise". The test that would is hen photos against
other photos taken in the same yard (an empty coop, the run without a bird, Elise alone), which do not exist here yet.
Files: images.json, images_all.json; the per-image activations (images_all.npz, 12 MB) stay local, like acts.npz.

## The same photos with the hens removed (images_removed.py, 2026-11-04; I4 to I7 committed 09:48 CT before any edit)
Elise's idea (DM 09:47 CT): have a Luna codex worker erase the chickens from twelve photos with its image tool. The
tool redraws the whole scene, so each photo was also put through it with the hens kept, and the comparison is kept
against removed: both have passed through the generator. 11 of 12 pairs passed the worker's own look (no chicken part
left in `removed`, every hen still there in `kept`); the kept image of the desk photo dropped a hen and that pair is out
(`hen-removal-NOTES.md`). The kept check was loosened before scoring, from "matches the original" to "every hen remains,
reframing noted", because both edits reframe. The edited images stay local.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| I4 | L17 N1574 higher on kept than removed in >= 10 of the usable pairs | 11 of 11 | held |
| I5 | the same for L15 N1126 | 7 of 11 | failed |
| I6 | L17 N1574's median kept minus removed ranks in the top 5% of all 49,152 neurons | rank 225 (top 0.46%) | held |
| I7 | kept moves L17 N1574 from the original by under half what removed does (medians) | 0.006 against 0.144 | held |

Reading: taking the chickens out of the picture, and nothing else the generator does, drops L17 N1574 in every pair,
and very few neurons drop as much. Divided by each neuron's spread over these images, its drop still ranks 602 (top
1.2%). This is the test the photo-set null above asked for: same yard, same light, same people, hens or no hens. L15
N1126, the strongest audio unit, does not respond to the photos either way, as I3 already showed.

## Is it a chicken neuron or a "not a human voice" neuron? (VocalImitationSet, V1 to V6 committed 10:18 CT before any download)
Muse's prior-art read named the confound: every imitation so far was a human voice and every real clip was not.
VocalImitationSet (Kim, Ghei, Pardo and Duan, DCASE 2018) has people imitating many animals plus real Freesound
recordings of the same classes, which gives a 2 x 2. Seeded selection (`vis_select.py`, `vis_manifest.csv`): all 38
included chicken imitations (cluck and crowing), 112 imitations of the other 28 animal classes (4 per class), all 20 real
chicken recordings, and 56 real recordings of the other animals (2 per class); first 30 s of each, same neurons, signs
and prompt classifier as the Pixabay arm. Audio stays local.

| # | Prediction | Result [95% interval] | Verdict |
|---|---|---|---|
| V1 | L17 N1574: chicken imitations above other-animal imitations, AUC >= 0.65 | 0.587 [0.489, 0.682] | failed |
| V2 | L17 N1574: real other animals against their imitations, AUC < 0.65 | 0.617 [0.519, 0.710] | held |
| V3 | L17 N1574: real chickens above real other animals, AUC >= 0.85 | 0.764 [0.649, 0.863] | failed |
| V4 | L17 N1574: real chickens above chicken imitations, AUC >= 0.85 | 0.893 [0.803, 0.962] | held |
| V5 | prompt classifier: chicken imitations above other-animal imitations, AUC >= 0.80 | 0.621 [0.521, 0.715] | failed |
| V6 | prompt classifier below L17 N1574 on real chickens against chicken imitations | 0.576 against 0.893 | held |

Reading: the confound is mostly ruled out for L17 N1574. Its real-over-imitation gap is large for chickens and small
for other animals: median activation real chickens 0.073, chicken imitations 0.012, real other animals 0.023, their
imitations 0.010. So "real" lifts it about 0.06 for chickens and about 0.01 for everything else, which a plain
not-a-human-voice unit would not do. V2 held only narrowly (0.617 against a 0.65 bar), so a small general real-sound lift
is there too. What the neuron mostly does not do is recognise a chicken in a human voice (V1, 0.587): a person clucking
barely moves it. Some other chicken units do respond to imitations (L19 N1129 0.775, L16 N134 0.713, L15 N1126 0.655),
which is where any "this is a chicken, imitated" signal lives. The model's own verdict is near chance on real against
imitation (V6). On this set L17 N1574 separates real chickens from real other animals less well than on ESC-50 (V3,
0.764): the Freesound "other animal" recordings include farm scenes and other fowl. For the edit study this is the
useful split: L17 N1574 reads "a real chicken is making this sound", and units like L19 N1129 read "this sounds like a
chicken at all".

## Teaching the model to tell real hens from imitations (edit_arms.py, 2026-11-04; E1 to E6 committed 11:12 CT before any arm was fitted)
Elise's idea (DM 09:39 CT, go at 11:11 CT). Design: `EDIT-DESIGN.md` and its Amendment 1 (steering at the output, the
trained layer as a residual MLP on the pooled embedding, everything from stored arrays). A Luna codex worker wrote
`edit_arms.py` and ran it; the test split is the 20 chicken imitations by the 16 held-out imitators against the 20
VocalImitationSet real chickens, scored by the model's own verdict (the prompt classifier) on the edited embedding.
Baseline held-out AUC 0.588 [0.400, 0.760].

| Arm | Fit AUC | Held-out AUC [95%] | Non-chicken ESC-50 clips untouched | ESC-50 classes changed by more than 1 clip |
|---|---:|---:|---:|---|
| Written gated block | 0.867 | 0.853 [0.720, 0.953] | 176 of 192 | none |
| Steering (the same write, no gate) | 0.657 | 0.588 [0.400, 0.760] | 0 of 192 | none |
| Trained layer (residual MLP) | 1.000 | 0.993 [0.970, 1.000] | 0 of 192 | 8, including hen |
| Control: random direction | 0.720 | 0.595 | 160 of 192 | none |
| Control: random gate neurons | 0.658 | 0.598 | 111 of 192 | none |
| Control: shuffled labels | 0.652 | 0.558 | 0 of 192 | 10 |

| # | Prediction | Result | Verdict |
|---|---|---|---|
| E1 | written block raises held-out AUC to >= 0.75 | 0.853 | held |
| E2 | written block leaves >= 95% of non-chicken clips identical and moves no class by > 1 clip | 91.7% identical; no class moved by > 1 | failed |
| E3 | trained layer reaches >= 0.85 | 0.993 | held |
| E4 | trained layer moves at least one class by > 1 clip | 8 classes | held |
| E5 | steering moves more classes than the written block | 0 against 1 class moved by one clip | failed |
| E6 | each control gains < 0.05 | +0.008, +0.010, -0.030 | held |

Reading: the hand-written block, three numbers fitted on half the imitators, turns the model's own near-chance verdict
into one that ranks real chickens above unseen people's imitations 85% of the time, and leaves 92% of ordinary sounds
bit-identical. The trained layer does better on chickens (0.99) and pays for it everywhere: every other clip moves, and
eight classes' accuracy changes, hens among them. The same write without its gate (steering) does nothing useful at any
scale the fit chose, and the same gate with a random direction or random neurons does nothing either, so the gain is the
two chicken neurons deciding when to write. E2 failed on its first half: 16 non-chicken clips (8.3%) opened the gate, as
the pilot's 4.5% warned.
Disclosure: the worker widened the k grid (to 0.01 to 100,000) after an exploratory run of the whole script had already
scored the test split; it reports deciding the widening on fit data alone (the best k sat at the old grid's edge). With
the gate open, a large k moves an embedding almost entirely onto the anti-chicken direction, so the result depends on
which clips open the gate rather than on k's exact value; still, this is one test set seen twice. A clean confirmation
needs new imitators or Elise's takes, scored once. Files: `edit_arms.py`, `test_edit_arms.py`, `edit_verdicts.json`
(the full `edit_results.json`, 5.8 MB with the trained weights, stays local).

## The same gate written inside the network (inside_gate.py, 2026-11-04; I1 to I5 committed before the run)
Elise (DM 12:07 CT): "Lets test it inside the network too ... I'm curious what that causes!" Design: `EDIT-DESIGN.md`
Amendment 3. Same gate, same frozen thresholds; when it opens, a hook adds `-k * g * R * d` to the output of
`layers[19]` at every position, so layers 20 to 23 carry the push. d is the residual-stream chicken direction from
ESC-50 folds 1 to 3; R = 52.51 is that stream's mean per-position norm. Only k was fitted, by the written block's rule.

Parity first: on all 336 discovery clips the live path reproduced the stored neuron values exactly (largest difference
0.0) and the stored embeddings (lowest cosine 0.9999998).

| | held-out AUC (95% interval) | gated imitations' top class | median cos to original | non-chicken identical |
|---|---|---|---|---|
| unedited model | 0.5875 | | | |
| output version (k 10,000) | 0.8525 [0.72, 0.95] | door_wood_creaks 11, crying_baby 1 | -0.478 | 176/192 |
| inside version (k 17,783) | 0.8525 [0.72, 0.95] | engine 10, keyboard_typing 2 | -0.056 | 176/192 |
| inside, random direction (k 100,000, the grid's top) | 0.7625 [0.59, 0.91] | engine 12 | -0.057 | 176/192 |

- **I1 held** (0.8525, the same value as the output version on the same clips). **I2 held** (no ESC-50 class changed by
  more than one clip). **I3 failed**: the gated imitations did not spread over human sounds; 10 of 12 landed on
  "engine". **I4 held as worded** (-0.056 against -0.478), but neither version keeps the sound: -0.06 means the edited
  embedding is unrelated to the original. **I5 held** (random direction 0.7625).
- What it causes, read from these numbers: the fitted push is huge. On the 12 gated test imitations it is a median of
  5.9 times the residual stream's own size (1.1 to 16.8). Layers 20 to 23 cannot make anything of an input that large,
  and the output collapses to one place, "engine". The random direction lands there too, so "engine" is where this
  network goes when its layer-20 input is swamped, not something the chicken direction specifically produces.
- Why the AUC is identical: both versions open on the same 12 imitations and 1 real clip, and both push those far
  enough that their chicken scores fall below every closed clip. At this strength the gate decides everything and the
  write's location decides nothing. The held-out split was seen twice in the first build, so this is a comparison on
  the same clips, not a clean confirmation.
- Files: `inside_gate.py`, `inside_gate.log`, `inside_results.json`, `inside_params.npz`.

### Gentler pushes (dose_ladder_inside.py; exploratory, run after I1 to I5 were scored, nothing predicted)
The 12 gated test imitations, pushed inside at fixed sizes relative to R, with the chicken direction and the random one.
Median chicken score (unedited 0.017), median cosine to the unedited embedding, and top classes:

| push / R | chicken direction | random direction |
|---|---|---|
| 0.1 | 0.002, cos 0.95: clapping 4, coughing 2, laughing 2, breathing, insects, clock tick, sneezing | 0.019, cos 0.94: laughing 4, dog 3, crying baby 2, ... |
| 0.2 | -0.008, cos 0.83: clapping 6, coughing 2, breathing 2, clock tick, chainsaw | 0.020, cos 0.83: dog 9, ... |
| 0.35 | -0.018, cos 0.66: keyboard typing 10, ... | 0.022, cos 0.64: dog 8, engine 3, helicopter |
| 0.5 to 1.5 | -0.022 to -0.034: keyboard typing 10 to 12 | 0.020 to 0.014: engine 11 to 12 |
| 3 and 6 | -0.040, -0.042: engine 12 | 0.012, 0.011: engine 12 |

- At a tenth to a fifth of the stream's size, the chicken direction lowers the chicken score while the sound stays
  mostly itself (cosine 0.83 to 0.95), and the model hears human body sounds: clapping, coughing, breathing, sneezing.
  That is what I3 predicted, at a strength the fitting rule never picks, because the rule rewards the largest drop.
- The random direction the same size leaves the chicken score where it was and turns the imitations into a dog.
- Past about a third of the stream, the chicken direction gives "keyboard typing" (the random one, "engine"); past
  about three times, both give "engine".
- So the two directions part ways early and meet only when the push swamps the layer. A gentle-push version (fitted
  for the drop at a fixed small size, keeping the sound) would be a new, registered design; nothing here tests it.

## The gentle inside version (gentle_gate.py, 2026-11-04; G1 to G5 committed before the run)
Elise (DM 13:02 CT): "Try the gentle version." Design: `EDIT-DESIGN.md` Amendment 4. When the gate opens, a fixed push
of s * R along -d for every gated clip; s chosen on the fit split as the best fit AUC among sizes that keep the median
cosine of the gated fit imitations at 0.8 or more. Fit AUC rose with s at every step, so the constraint chose: s = 0.2
(fit AUC 0.848, fit cosine 0.825; s = 0.25 fell to 0.762 cosine). The random direction's fit AUC fell as s grew, so its
rule picked the smallest size, 0.025.

| | held-out AUC (95% interval) | gated imitations' top class | median cos to original |
|---|---|---|---|
| unedited | 0.5875 | | |
| full-strength inside (Amendment 3) | 0.8525 [0.72, 0.95] | engine 10, keyboard typing 2 | -0.056 |
| **gentle (s 0.2)** | **0.81 [0.66, 0.94]** | **clapping 6, coughing 2, breathing 2**, clock tick, chainsaw | **0.834** |
| gentle, random direction (s 0.025) | 0.5775 [0.39, 0.75] | laughing 6, clapping 2, ... | 0.996 |

- **G1 held** (0.81). **G2 held** (0.834). **G3 held** (10 of 12 human sounds), but it was contaminated: the exploratory
  ladder had shown this on the same clips. **G4 held** (random 0.5775, at the baseline). **G5 held** (no ESC-50 class
  changed by more than one clip; 176 of 192 non-chicken clips bit-identical, as with every version of this gate).
- So a push about a fifth of the stream's size keeps most of the sound, loses only 0.04 AUC against the full-strength
  versions, and turns a person's chicken imitation back into a person: clapping, coughing, breathing.
- The random direction cannot do the job at any size that keeps the sound: its fit AUC falls below the unedited level
  as the push grows. The chicken direction is doing the work.
- Elise's fresh takes (Amendment 2's set, still open) are the clean test.
- Files: `gentle_gate.py`, `gentle_gate.log`, `gentle_results.json`.

## Moment by moment: windows with and without the chicken (timecut.py, 2026-11-04; T1 to T4 registered before)
Within each real chicken recording, 2 s windows the AudioSet AST classifier marks as chicken (top third) against windows
it marks as not chicken (bottom third), so mic, room and background are held constant. 115 recordings (20
VocalImitationSet, 95 Pixabay with at least 6 windows), 2,392 windows; the AST split separates its own ON and OFF at AUC
0.87.

- **T1 failed narrowly:** L17 N1574 is higher on ON windows in 73.9% of recordings, against a bar of 75%. **T2 failed:**
  67.8% with loudness-matched pairs (bar 70%). **T3 failed:** its within-recording effect ranks 4,466 of 49,152 (top
  9.1%, bar top 5%).
- **T4 held:** L15 N1126 is higher on ON windows in 78.3% of recordings, and 76.5% with loudness-matched pairs
  (descriptive).
- Reading: both units do rise when the chicken sounds, but L15 N1126 tracks the moment better, and L17 N1574 (the
  real-chicken unit the gate uses) carries more of something steady across a recording. That fits its job in the gate,
  telling a real chicken recording from a person, more than telling a chicken second from a quiet second.
- Files: `timecut.py`, `timecut.log`, `timecut_results.json` (`timecut.npz` stays local, 21 MB). The run's outer
  waiter was stopped at 13:0x while `timecut.py` kept running in its own scope, so its rc line is missing from
  `timecut_after_arc.log`; the script wrote its results and exited normally.

## The same job as a direct weight edit (weight_edit.py, 2026-11-04; X1 to X5 registered 14:05 CT before any run)
Elise asked at 13:58 CT whether direct weight editing could do it, and said "Yes please!" at 14:04 CT (message
1567630584262557808; `EDIT-DESIGN.md` Amendment 5's header gives 13:58 for both). Design: Amendment 5. Nothing is added
beside the model; two edits change existing weights in layer 20's MLP. W1 rewrites the layer's least-used neuron
(index 1083) into the gate: its `up` row reads "sounds like a chicken imitation", its `gate` row reads "imitation, not a
real chicken" minus a threshold, and its `down` column writes the gentle push. W2 is a rank-one closed-form change to
`down_proj` (ROME-style) that writes the push wherever layer 20's keys look like the fit imitations', solved against
keys from ordinary sounds and real chickens; the solve adds a ridge of 1e-4 times the covariance's mean diagonal, which
Amendment 5 does not mention (`w2_covariance` in the results). A shuffled-label W2 is the control. Both act per position
(per slice of sound) rather than once per clip, a difference recorded in the design before the run. A Luna worker built
and ran it, replaying only layers 20 to 23 from a cache after proving the replay exact (worst cosine 0.99999982 over
40 clips, before the edits and again after the weights were restored).

| | held-out AUC (95% interval) | test imitations' top class (all 20) | median cos to original (all 20) | non-chicken clips within cos 0.999 (of 192) | ESC-50 folds 4-5 correct (of 224, unedited 50) | other-animal imitations relabelled (of 112) |
|---|---|---|---|---|---|---|
| unedited | 0.5875 | | | | 50 | |
| **W1, one repurposed neuron** | **0.6775 [0.495, 0.843]** | clapping 5, coughing 3, laughing 3, sneezing 3, six others 1 each | 0.951 | 35 | 45 | 53 |
| **W2, rank-one edit** | **0.8125 [0.665, 0.930]** | clapping 6, sneezing 4, coughing 3, breathing 2, laughing 2, three others 1 each | 0.910 | 0 | 46 | 62 |
| W2 control (shuffled labels) | 0.6175 [0.43, 0.788] | clapping 5, laughing 5, crying baby 2, sneezing 2, six others 1 each | 0.986 | 35 | 49 | 34 |

For comparison, the gentle hook (Amendment 4, not like for like: its top classes and median cosine cover only the 12
gated test imitations): held-out 0.81 [0.66, 0.94]; gated imitations clapping 6, coughing 2, breathing 2, others 1
each; their median cosine 0.834; 176 of 192 non-chicken clips bit-identical (so at least 176 within 0.999); no ESC-50
class moved by more than one clip (its folds 4-5 total was not recorded); 31 of 112 other-animal imitations relabelled.

ESC-50 classes that changed: W1 crying baby -2, snoring, clock alarm and siren -1 each; W2 crying baby -2, cat,
snoring and clock alarm -1 each, pouring water +1; control snoring -1.

- **X1 failed** (W1 0.6775 against 0.75). **X2 held** (W2 0.8125). **X3 failed** (W1 left 35 of 192, 18%, within
  cosine 0.999, against 80%). **X4 held** (W2 left 0 against W1's 35; "fewer" holds, though W1 itself failed X3).
  **X5 held** on the point estimates (control 0.6175 against 0.8125), with overlapping 95% intervals on 20 + 20 clips.
  Fitted settings: W1 threshold at the 0.3 quantile and s = 0.025, the smallest size in the registered grid; W2
  s = 0.15; the control's rule also chose 0.025, so it is not size-matched to W2.
- So the rank-one edit, one changed matrix, reaches the gentle hook's held-out point estimate (0.8125 against 0.81) and
  turns imitations into human sounds, with nothing added to the model. It fell further from fit to test (0.897 to
  0.8125) than the hook did (0.848 to 0.81). It is less narrow than the hook: no non-chicken clip stays within cosine
  0.999 (the hook left at least 176), and it relabels 62 of 112 people imitating other animals, twice the hook's 31.
- W1's fit AUC was already only 0.7537, and its push was the smallest the grid allowed; a weaker push was never tried.
  Why it does worse is not tested here. One guess, not established: per-position gating is noisier than the hook's
  clip-mean gate.
- The control does not separate a chicken-specific effect from a size effect, because it chose a much smaller push
  than W2. Its own spread (35 within 0.999, 34 relabelled) is what that smaller edit does.
- No fresh take has been scored by any weight edit. The held-out split has been looked at by earlier versions; Elise's
  own recordings (the design's external set, never used in fitting) and new imitations are the clean test.
- Cold check (Haiku, 2026-11-04): eight findings, all folded: the hook row's gated-only figures and unknown ESC total,
  the control's size, the W1 explanation, the missing fresh-take caveat and fit-to-test drop, the full ESC changes, the
  unregistered ridge, the date of Elise's yes.
- Files: `weight_edit.py`, `weight_edit.log`, `weight_edit_results.json`; worker report
  `DiscordBot/df-runs/chicken-weight-edit-2026-11-04/worker-01/report.md`.

## Elise's own takes, scored once at her close (Amendments 6 and 6a; scored 2026-11-06 13:15 CT)

**Claim: on Elise's phone recordings, every edit that was fitted to tell real hens from imitations does so better than
the unedited model, and three of them separate the two perfectly. The gate's one failure is that it mistakes faint or
distant real hens for imitations.**

The set: 8 real-hen takes and 6 of her imitations, uploaded by Discord and labelled by her before any scoring; closed on
her word 2026-11-06 13:04 CT ("We can close it."). Two rules recorded before scoring (13:05 CT): Jewel's full take is the
same recording as the trimmed one, so only the trimmed take counts (the full one is descriptive); call types for 6a come
from her descriptions (`takes_calltypes.tsv`). Input `takes_final.tsv`, scores `takes_scored_final.json`, summary
`takes_final_summary.txt`. Every version frozen as committed; no parameter changed.

| version | AUC real vs imitation (8 v 6) | egg songs only (2 v 5, descriptive) |
|---|---|---|
| unedited | 0.71 | 0.70 |
| output gate | **1.00** | 1.00 |
| inside gate, full | **1.00** | 1.00 |
| gentle inside | 0.75 | 0.50 |
| W1 | 0.94 | 0.90 |
| W2 | **1.00** | 1.00 |
| W2 control | 0.77 | 0.70 |

Predictions, read against the registered wording:
- C1 (output gate AUC 0.75 or more): **held**, 1.00.
- C2 (gated AUC above ungated on the same takes): **held**, 1.00 against 0.71.
- C3 (the gate opens on at most one in four real takes): **failed**. It opened on 3 of 8: `chicken_sounds` (mostly
  Brahma Mama with another hen), `noisy_minimal_peeps` (mostly background, a few peeps) and Brahma Mama's faint egg song
  from 11-05 (4.1 dB over its floor). It also opened on Jewel's full take (descriptive), whose first part is far away.
  It stayed shut on every close, clean real take, Jewel's trimmed egg song among them.
- C4 (opens on at least half the imitations): **held**, 6 of 6.
- C5 (W2 AUC 0.75 or more): **held**, 1.00; its random-direction control is 0.77.
- C6 (gentle AUC 0.75 or more): **held at the boundary**, exactly 0.75; within egg songs it falls to 0.50 (descriptive).
- C7 (the gate stays shut on every no-chicken take): **held**, but on one take (`background_only`). The summary's
  `other_gate_open` count (2 of 4) mixes in three takes that do contain chickens, so it is not C7's measure.

Hypothesis, not tested: the gate reads "a close, clean real hen" rather than "a real hen". All four real takes it opened
on are faint, distant or noisy; the real-hen unit (L17 N1574) is probably quiet on them for the same reason it is quiet
on an imitation. A test would score the same hen at several distances.

Within egg songs alone the output gate, the inside gate and W2 still separate real from imitation, so the AUCs above are
not only egg song against chatter; with two real egg songs that is descriptive.
