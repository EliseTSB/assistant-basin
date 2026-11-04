# Part 188: does any single unit in EmbeddingGemma 2 tell real animal sounds from people imitating them, on animals it was not chosen on, better than the model's own verdict?

Procedure: v1
Registered REGTIME CT by Claude (Opus 5.5), before any data. Elise asked, 2026-11-04 14:58 CT (Discord message
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
- Probe on the output: logistic regression on the 768-d embedding, standardized on the selection fold, L2, C = 1.0,
  fitted on the selection fold. Top-32: the same on the 32 units with the largest selection |AUC - 0.5|.
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

Gate 2 (data): at least 90% of each set's planned clips decode; the missing ones are listed.

Gate 3 (rig): L15 N1126 (index 31,846), sign up, separates ESC-50 hen and rooster from set E at AUC at least 0.90 (the
earlier study found 0.94 against the other ESC-50 classes). It fails if the hook reads the wrong layer or position.

## Size

- Smallest effect worth knowing: a held-out AUC 0.10 above the model's own verdict. With about 300 real and 280
  imitation clips per test fold, the clip-level standard error of an AUC near 0.8 is about 0.02, so 0.10 is resolved
  at the clip level; the group-clustered interval (about 9 groups per fold) is wider and is reported as the honest one.
- Total: about 3,140 forward passes (one per clip, 5 s audio), about 40 minutes at the earlier study's 0.68 s per clip
  on 6 CPU threads; the cloud machine's speed is unknown.
- Pilot: not needed (no generations, under one GPU-hour); Gates 1 and 3 stop the reading if the rig is wrong.

## Optimization

- Bound resource: CPU in a Claude cloud session.
- Measured: the earlier study's `record.py`, 631 items (mostly 5 s ESC-50 clips) in 427 s on 6 threads (`record.log`),
  0.68 s per clip.
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
| 188.7 | the top-32 probe beats the selected unit by less than 0.05 held-out AUC in both test folds | low | Claude |
| 188.8 | both selected units separate real chickens from chicken imitations (set C) at AUC at least 0.80 | medium | Claude |
| 188.9 | sequence length alone has held-out AUC below 0.70 in both test folds | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 188 (registered ...)".

## Reading rules

- Every number is printed by `analyze188.py` into `results.json`. AUCs are clip-level; 95% intervals by 2,000
  bootstrap resamples, both clip-level and clustered by group (groups resampled within the test fold). Predictions are
  scored on the point estimates; the clustered intervals are reported beside them.
- Clips that fail to decode are dropped and listed; Gate 2 bounds them.
- The headline reading ("a unit inside the model does the general task better than the model's own verdict, on
  animals it was not chosen on") is withdrawn if Gate 1 or Gate 3 fails, if 188.2 or 188.3 fails in either fold, or if
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
- Selection by single units only; a direction across units can do better, and the top-32 probe is the only check on
  that.
