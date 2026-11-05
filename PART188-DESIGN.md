# Part 188: does any single unit in EmbeddingGemma 2 tell real animal sounds from people imitating them, on animals it was not chosen on, better than the model's own verdict?

Procedure: v1
Registered 2026-11-04 17:39 CT by Claude (Opus 5.5), before any data. Elise asked, 2026-11-04 14:58 CT (Discord message
1567644205688557629): "Editing embedded gemmas weights to make it much better at distinguishing animal sounds from
human imitations, without negatively effectively it much if at all outside of that, would be a really good
demonstration of the power of direct weights editing." At 15:03 CT (1567645465241976937): "Run the search in claude
cloud!" The commit that adds this file is the registration.

## Question

An earlier study of EmbeddingGemma 2 (Journal, not yet public; `notes/reports/embeddinggemma2-chicken-neurons-2026-11-03`)
found that L17 N1574 separates real chickens from people imitating chickens at AUC 0.893, while the model's own
prompt verdict sits near chance, and that a gate built from it and L19 N1129 makes the model much better at that one
task. The same neuron separates other real animals from their imitations at only 0.617. This part asks whether any
single unit does the general task, real animal against human imitation, on animals it was not selected on, and how it
compares with the model's own verdict and with a probe on the model's output. A unit that does is the input to a weight
edit (a later part); a unit that only reads "a human made this sound" would still serve the edit, and the design
measures which kind it is.

## Prior art and existing data

- Closest published work: none found on detecting human imitations against real animal sounds from inside a model.
  Searched 2026-11-04 ("detect human vocal imitation versus real animal sound classifier VocalImitationSet"): the
  VocalImitationSet work (Kim, Ghei, Pardo, Duan, DCASE 2018) and the voice-sketching paper (arXiv 2409.13507) use
  imitations to retrieve real sounds, the opposite goal; Oudyk et al. matched birdsong imitations to recordings. The
  weight-editing context: MIT's weight-space programming post (Takehi, Teoh, Zhang, Khattab, 2026-11-02) edits weights
  from reading them; this part finds the unit first from activations.
- Our own data: the chicken study's V1 to V6 (VocalImitationSet, 226 clips): L17 N1574 real other animals against their
  imitations 0.617; chicken imitations above other-animal imitations at L19 N1129 0.775. No search over all units on
  the general task has been run; this part's selection uses none of those 226 clips' results.

## Method

- Model: `google/embeddinggemma-2`, revision `f9972ac22e696a105c482202646979e33e4b5a1e`, loaded exactly as the chicken
  study's `record.py` (sentence-transformers 6.2.0, `config_kwargs={'vision_config': None}`, float32, CPU). Units: the
  input to `language_model.layers[i].mlp.down_proj`, 24 layers x 2,048 = 49,152, mean over sequence positions, one clip
  per forward pass (no padding). Also the final normalized embedding (768).
- Audio: decoded to mono 16 kHz float32 with ffmpeg, first 5 s of every clip (ESC-50 clips are 5 s; cutting the rest to
  5 s limits a length difference).
- Data, downloaded in a Claude cloud session (audio is never committed):
  - VocalImitationSet at commit `b77114c43c52f5faa7d0bb69b22e1ebca0632201`: every included imitation of the 31 animal
    classes except Cattle / Cowbell (set I, about 571), every original recording of those 30 classes (reference and
    non-reference; set R), and one seeded included imitation of each non-animal class (set N, about 271).
  - ESC-50 at a pinned commit: all 40 clips of dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow, crickets and
    chirping_birds (set R); the 10 human classes (set H, 400); the other 28 classes (set E, 1,120).
  - Class map, groups and prompt phrases: `part188/classes.json`. The manifest and fold assignment are written by
    `part188/select188.py` from the dataset metadata and committed with this design.
- Groups and folds: classes are grouped by animal (Cat, Dog, Cattle, ...). The Chicken group is held out of both folds
  as set C (the earlier study's animal). The other 19 groups are split into folds A and B by `select188.py` (seed 188,
  greedy balance of real plus imitation counts); `part188/folds.json` records it.
- Selection, for each direction (select on A, test on B; select on B, test on A): per unit, clip-level AUC of real (R in
  the fold) against imitation (I in the fold) on the mean activation. The selected unit is the one with the largest
  |AUC - 0.5|, its sign fixed on the selection fold; ties go to the lowest index.
- The model's own verdict: the clip embedding's cosine with "the sound of {phrase}" minus its cosine with "a person
  imitating the sound of {phrase}", encoded with `prompt_name='SearchQuery'` as in the earlier study. Class-informed
  (the clip's own class phrase) is primary; class-free ("an animal") is reported beside it.
- Probe on the output: logistic regression on the 768-d embedding, standardized on the selection fold, L2 with C = 1.0
  in scikit-learn's convention (minimise the summed log-loss plus ||w||^2 / (2C)), fitted on the selection fold. Top-32: the same on the 32 units with the largest selection |AUC - 0.5|.
- Analysis: `part188/analyze188.py`, run in the cloud session after recording; it writes `results.json` and the arrays
  listed under Reading rules. All numbers come from it.

## Controls

- Held-out animals: every reported unit is scored on the fold it was not selected on (the selection is a search over
  49,152 units, so its own fold's AUC is inflated and is reported only beside the held-out one).
- Held-out people: on the test fold, AUC is also reported on imitations by participants with no imitation in the
  selection fold.
- Length: the clip's sequence length alone, its sign fixed on the selection fold, scored on the test fold.
- Human sound or imitation: the selected unit on real animals (test fold R) against ESC-50 human sounds (H, and its
  vocal subset), all from Freesound-quality recordings like the ESC-50 animals, which separates "a person made this"
  from "this was recorded on a participant's microphone"; and on imitations of non-animal sounds (N) against
  imitations of animals (test fold I), where a unit that reads only "a human voice" sits near 0.5.
- The known unit: L17 N1574 (index 36,390), sign real-above, on all non-chicken groups pooled.
- Not used: a do-nothing hook matching the plain model (nothing is pushed or zeroed here); random-direction controls
  (no push); readers (no generations; every readout is an activation or a cosine).

Gate 1 (identity): 8 ESC-50 clips also in the earlier study's `acts.npz` are recorded first. For each, the cloud's mean
activations rounded to float16 match the stored values within 2e-3 absolute on at least 99.9% of the 49,152 units,
and the embedding cosine with the stored one is at least 0.9999. The reference values are committed as
`part188/parity_ref.npz`. It fails on a different model revision, dtype, hook point or decode. If it fails, nothing is
read.

Gate 2 (data): at least 90% of each set's planned clips are recorded (a download, decode or record error counts as
missing); the missing ones are listed by id and reason.

Gate 3 (rig): L15 N1126 (index 31,846), sign up, separates ESC-50 hen and rooster from set E at AUC at least 0.90 (the
earlier study found 0.94 against the other ESC-50 classes). It fails if the hook reads the wrong layer or position.

## Size

- Smallest effect worth knowing: a held-out AUC 0.10 above the model's own verdict. With about 300 real and 280
  imitation clips per test fold, the clip-level standard error of an AUC near 0.8 is about 0.02, so 0.10 is resolved
  at the clip level; the group-clustered interval (about 9 groups per fold) is wider and is reported as the honest one.
- Total: 3,138 forward passes (one per clip, 5 s audio), about 40 minutes at the earlier study's 0.68 s per clip
  on 6 CPU threads (a smoke run on the real model measured about 0.6 s per clip); the cloud machine's speed is unknown.
- Pilot: not needed (no generations, under one GPU-hour); Gates 1 and 3 stop the reading if the rig is wrong.

## Optimization

- Bound resource: CPU in a Claude cloud session.
- Measured: 631 clips in 427 s (the earlier study's `record.py`, mostly 5 s ESC-50 clips, 6 threads, `record.log`),
  0.68 s per clip; a smoke run of this part's recorder on the real model measured about 0.6 s per clip on 8 parity clips.
- Levers: batching declined (one clip per pass without padding is what the stored recordings and Gate 1 use, and the
  whole run is under an hour); threads set to the machine's core count, taken; the max over positions dropped, taken
  (the design reads the mean; it halves memory); downloads parallel, taken; resume, taken (rows written are skipped, as
  a cloud container can be recycled).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 188.1 | the model's own verdict (class-informed) has held-out AUC below 0.75 in both test folds | medium | Claude |
| 188.2 | the selected unit's held-out AUC is at least 0.85 in both test folds | medium | Claude |
| 188.3 | the selected unit beats the model's own verdict (class-informed) by at least 0.10 held-out AUC in both test folds | medium | Claude |
| 188.4 | the selected unit separates real animals (test fold R) from ESC-50 human sounds (H) at AUC at least 0.80, its selection sign, in both folds: it reads "a human made this", not only imitation | medium | Claude |
| 188.5 | L17 N1574, real above imitation, on all non-chicken groups pooled, has AUC below 0.70 | high | Claude |
| 188.6 | the probe on the output embedding has held-out AUC at least 0.90 in both test folds | medium | Claude |
| 188.7 | the top-32 probe's held-out AUC exceeds the selected unit's by less than 0.05 in both test folds (a probe below the unit also holds) | low | Claude |
| 188.8 | both selected units separate real chickens from chicken imitations (set C) at AUC at least 0.80 | medium | Claude |
| 188.9 | sequence length alone has held-out AUC below 0.70 in both test folds | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 188 (registered ...)".

## Reading rules

- Every number is printed by `analyze188.py` into `results.json`. AUCs are clip-level; 95% intervals by 2,000
  bootstrap resamples, both clip-level and clustered by group (groups resampled within the test fold; for set C, which
  is one group, clustered by class: ESC-50 hen, ESC-50 rooster, VocalImitationSet Cluck, Crowing). Predictions are
  scored on the point estimates; the clustered intervals are reported beside them.
- Clips that fail to decode are dropped and listed; Gate 2 bounds them.
- The headline reading ("a unit inside the model does the general task better than the model's own verdict, on
  animals it was not chosen on") is withdrawn if Gate 1, Gate 2 or Gate 3 fails, if 188.2 or 188.3 fails in either fold, or if
  length alone (188.9) reaches within 0.05 of the selected unit's held-out AUC in that fold.
- If 188.4 holds, the result is reported as a unit that reads human-made sound, which still serves the edit, not as an
  imitation detector.
- A pattern found after the results (another unit, the vocal-only split, max over positions) is labelled exploratory.

## Limits

- One model, one dataset of imitations: VocalImitationSet's participants recorded on their own microphones through a
  web page, so recording quality differs from Freesound and ESC-50; the H control addresses voice against microphone
  only for one direction.
- Some VocalImitationSet "original recordings" may themselves be imitations or edited sounds (8 of 180 Pixabay
  "real" chicken clips were imitations by ear in the earlier study); nothing here checks them by ear.
- About 9 animal groups per test fold: the clustered interval is wide, and "general" means these 19 groups.
- The first 5 s only.
- Duplicate audio: 14 VocalImitationSet originals repeat a Freesound recording, and 24 share a Freesound source with
  an ESC-50 clip (counted from the manifest's file names); every such pair sits within one fold, so nothing leaks across
  folds, but clip-level intervals are slightly optimistic.
- The participant-disjoint subset is small: 69 of 169 imitators appear in both folds, leaving about 55 to 68 test
  imitations per direction.
- A cloud container recycled mid-run restarts the recording from the start (the per-clip records are too large to
  push); the run is about an hour.
- Selection by single units only; a direction across units can do better, and the top-32 probe is the only check on
  that.

## Changes before registration

This design was drafted 2026-11-04 15:09 CT and changed before registration, after two code reviews of the build
(Haiku 5.5 and DeepSeek V4.1 Flash, between 16:13 and 17:05 CT) and before any data: the probe's L2 strength is stated in
scikit-learn's convention (the build had used a penalty about 600 times stronger); Gate 2 counts any download, decode
or record error as missing and lists each by reason; a failed Gate 2 withdraws the headline; set C's clustered
interval is by class (it is one group); 188.7's wording now says a probe below the unit also holds, which is the
prediction's intent (a single unit nearly as good as the 32-unit probe) and the build's test; the duplicate-audio and
participant-subset limits were added from measured counts. A smoke test on the real model passed Gate 1 on the 8
parity clips at about 0.6 s per clip.


At 17:39 CT, before registering, the Size total was written as the exact manifest count (3,138, was "about 3,140") and
the Measured line was reworded so `tools/part_check.py` reads its rate; the smoke rate was added to it. No threshold,
prediction, fold or class changed.

## Amendment 1 (2026-11-04 19:00 CT, before any data)

The first cloud session (routine fired 17:53 CT, branch `cloud/part188`, commit 03a8f27) recorded nothing: model load
failed with sentence-transformers 6.2.0 raising "EmbeddingGemma2VideoProcessor requires the Torchvision library", and
`torchvision` was not in `part188/requirements188.txt`. The local smoke test had passed because the local stack loaded
sentence-transformers from a side folder that also held torchvision 0.29.2 (CPU), so the gap could not show there. The
session followed the routine and did not change the pins. Fix: `torchvision==0.29.2` is added to the requirements, to
the routine's CPU-index install line and to the session's pinned-version check, and a test now fails if any checked
library is missing from the requirements. The rerun uses branch `cloud/part188-r2`. No class, threshold, fold, prompt,
prediction or analysis changed, and no clip was recorded before this amendment.

## Amendment 2 (2026-11-04 ~19:55 CT, after the run, before any result value was read)

The second cloud session (`cloud/part188-r2`, commit 0388f8e) recorded 3,138 of 3,138 planned clips; Gates 1 and 3
passed, and its summary reports Gate 2 "failed". Only the gate verdicts and the per-set planned and recorded counts in
`summary.json` were read before this amendment; no AUC, unit or probe value was read. Every set the manifest defines
was recorded in full: C_I 38/38, C_R 100/100, E 1,120/1,120, H 400/400, I 533/533, N 269/269, R 678/678. The failure
came from the code, not the data: `_gate2_summary` added a set named "P" to the manifest's sets; no row carries it, so
it scored 0 of 0 as 0.0 and failed. Neither this design nor the manifest defines a set P (the eight parity clips are rows
of the real sets, checked by Gate 1). The registered rule, "at least 90% of each set's planned clips are recorded", is
therefore met. The code now evaluates only the manifest's sets, with a test that fails on the old line; the headline
rule is re-applied to the stored results with Gate 2 passed, and nothing else in the analysis changes. The extra set was
introduced in this part's own review fold (d479df9) and was not caught by its tests or reviews.
