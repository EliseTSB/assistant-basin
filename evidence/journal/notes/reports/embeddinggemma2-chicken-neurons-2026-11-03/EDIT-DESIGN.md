---
title: Teaching EmbeddingGemma 2 to tell real hens from human imitations (design draft)
authored-by: claude
created: 2026-11-04
status: draft, not registered; no edit has been built or scored
card: chicken-imitation-edit-2026-11-04
---

# Teaching the model to tell real hens from human imitations

Elise's idea (DM 2026-11-04 09:39 CT): combine the chicken neurons with weight editing or a new layer, so the model as a
whole gets better at telling imitations from real sounds, as a demonstration of several techniques at once.

## What we know going in

- The model's own verdict (the prompt classifier: best of "the sound of a hen / rooster" minus the mean of the other
  ESC-50 prompts) is fooled by human imitations: AUC 0.576 for real chickens against chicken imitations on
  VocalImitationSet, near chance (V6), and Elise's own imitation ranked first of 31 on the meter.
- L17 N1574 reads "a real chicken is making this": real chickens against chicken imitations 0.893 (V4), with a
  real-over-imitation lift about six times larger for chickens than for other animals; a person clucking barely moves it.
  It is also visual: erasing the hens from Elise's photos drops it in 11 of 11 pairs (I4, I6).
- Some other chicken units read "this sounds like a chicken" even from a person: L19 N1129 (chicken imitations over
  other imitations 0.775, signed), L16 N134 (0.713).
- Nobody has published a neuron-guided edit of an audio embedding model for this (Muse's prior art, council topic
  `muse-prior-art-imitation-edit-2026-11-04`, #3). The closest work is LoRA deepfake detectors and steering in
  audio-language models.

## The edit target

On clips it never saw, the model's own verdict should rank real chickens above human chicken imitations, and every other
sound should keep its embedding. Success is a held-out AUC for real chickens against chicken imitations of at least 0.85
on the prompt classifier, with ESC-50 accuracy unchanged per class.

## Arms (all fitted on the same split, all scored the same way)

1. **Written gated block (the hand-written one).** A new step after the encoder's pooled output. Its gate is
   g = relu(s19 - t19) * relu(t17 - s17), where s19 and s17 are the signed L19 N1129 and L17 N1574 activations (mean
   over positions), so it opens only when the sound is chicken-like and the real-chicken unit is quiet. Its write moves
   the embedding away from the chicken prompts by k * g along the normalised mean of the hen and rooster prompt
   embeddings, then renormalises. Three numbers (t19, t17, k) are fitted on the fit imitators. Where g is 0 the output is
   bit-identical to the unedited model, which is checkable per clip, as the W3 addition block was.
2. **Trained layer.** A rank-4 LoRA on the last layers (or a small residual MLP on the pooled output), trained on the
   fit split to push chicken imitations away from the chicken prompts, plus a loss that keeps every other training
   clip's embedding fixed.
3. **Steering.** A fixed vector added at layer 17 to every clip, scaled on the fit split. This is the baseline the gated
   block has to beat on damage.
4. **Controls.** The written block with a random direction in place of the chicken prompt direction. The written block
   gated by two random neurons from the same layers. The trained layer fitted on shuffled real/imitation labels.

## Data and splits

- Fit: VocalImitationSet chicken imitations by about half of its 32 imitators (split by participant id, seeded), the
  Pixabay real chickens, and ESC-50 folds 1 to 3.
- Test, never seen in fitting: the other imitators' chicken imitations, the 20 VocalImitationSet real chickens, ESC-50
  folds 4 and 5, the 8 Pixabay imitations, and Elise's own recordings (her imitations, the coop, Brahma Mama) as the
  external set.
- Damage: ESC-50 per-class accuracy of the prompt classifier before and after, on folds 4 and 5. The share of
  non-chicken clips whose embedding changes at all. The cosine change on the ones that do.

## Before registration

- A pilot on the fit split only: the distribution of g on non-chicken ESC-50 clips, to see whether the gate stays shut on
  them (if L19 N1129 fires on other birds, the gate needs a second condition). Then the predictions get written, with
  their bars, before any test clip is scored.
- CPU only, one job at a time, run with nothing heavy beside it on Elise's machine.

## Pilot result, fit split only (`edit_pilot_gate.py`, `edit_pilot_gate.json`, 2026-11-04)

With the thresholds set from the 18 fit imitations alone (t19 at their 25th percentile of signed L19 N1129, t17 at
their 75th of signed L17 N1574), the gate opens on 10 of the 18 fit imitations (56%), on none of the ESC-50 hens and
roosters in folds 1 to 3, and on 4.5% of the other fit ESC-50 clips. Those come from nine classes: breathing,
brushing teeth, church bells, clock alarm, cow, hand saw, pig, sheep and snoring. So the gate is not shut on ordinary
sounds. Human breathing and snoring open it (L19 N1129 carries some "human voice" too), and so do farm animals. A write
that only moves away from the chicken prompts should cost those classes little, but that is a damage prediction to
register, not an assumption. Opening on 56% of imitations caps what the written block can gain on the held-out AUC.
The trained layer has no such cap. Next: write the predictions (the gain for each arm, damage per ESC-50 class, the
random-direction and random-gate controls), then fit on this split and score the test split once.

## Amendment 1 (11:13 CT, before any arm is fitted or scored)

- **Steering moves to the output.** The steering arm is the written block without its gate: every clip's embedding
  moves away from the chicken prompts by the same k, then is renormalised. Steering at layer 17 would need new forward
  passes for every scale and every test clip. The output version uses only the stored embeddings and isolates exactly
  what the gate adds. E5 keeps its wording with this arm.
- **The trained layer is a small residual MLP on the pooled embedding**: one hidden layer of 64 units,
  e' = normalise(e + MLP(e)). It is trained on the fit split to lower the chicken score of fit imitations and raise
  that of fit real chickens, plus a mean-squared penalty that keeps the embeddings of fit ESC-50 non-chicken clips
  unchanged. It is not a LoRA. It uses stored embeddings only, and it is fitted once with a fixed seed, no tuning on the
  test split.
- **Everything is scored from stored arrays:** `acts.npz` (ESC-50, with the prompt embeddings), `vis_acts.npz` and
  `pixabay_acts.npz`. The gate's s17 and s19 come from those arrays' mean activations, which are the same quantities a
  live hook would read. Elise's own takes are not in a stored array, so they are scored later if at all.

## Amendment 2: the clean confirmation on Elise's fresh takes (12:03 CT, before any take is recorded)

Elise (DM 12:06 CT): "Lets test it!" The held-out 0.853 used a test split that was looked at twice during the build,
so this is the clean test: new recordings, the gate frozen, scored once.

- **Frozen.** The written gated block exactly as `edit_verdicts.json` (sha256 prefix 057b5bfcbc0458d2) records it: t19 =
  0.007097244262695313, t17 = 0.021478271484375002, k = 10000, signs L17 N1574 +1 and L19 N1129 -1, direction the
  normalised mean of the hen and rooster prompt embeddings in `acts.npz`. Neuron values are the mean over positions of
  the input to `mlp.down_proj`, rounded to float16 as the stored arrays were. The verdict score is `prompt_scores`
  from `edit_arms.py` (best of hen and rooster cosine minus the mean of the other 48 ESC-50 prompts), computed before
  and after the gate. No parameter changes after this commit.
- **The test set.** Every take Elise labels "real hen" or "my imitation" in the Chicken-o-Meter from this commit until
  she says the set is closed. The label is chosen before the take is scored and saved with the original audio in
  `confirm-takes/`. No take is dropped, except one the meter cannot decode or that is under half a second.
  "Something else" takes are descriptive only. Her 2026-11-03 recordings are not in the test set (their neuron values
  were read on 11-03).
- **Recording.** Both kinds should go through the same route (phone recordings uploaded, or both through the meter's
  mic), because a codec or device difference could separate them by itself. The route of each take is recorded.
- **Scoring.** `confirm_takes.py` scores the closed set once from the saved files. The meter shows each take's gate
  live as it is recorded; that changes nothing frozen. At least 6 takes of each kind are needed for C1 and C2; with
  fewer, all four are reported as descriptive.
- **Predictions.**
  - C1: with the gate, the verdict score separates real hens from her imitations at AUC 0.75 or more. Medium.
  - C2: the gated AUC is higher than the ungated AUC on the same takes. Medium-high.
  - C3: the gate opens on at most one in four real-hen takes. Medium (her 11-03 coop takes put L17 N1574 at about
    0.04 to 0.06, above t17, but a quiet or noisy take can sit lower).
  - C4: the gate opens on at least half of her imitation takes. Medium (it opened on 56% of the fit imitations).

## Amendment 3: the same gate written inside the network (12:10 CT, before any inside run)

Elise (DM 12:07 CT): "Lets test it inside the network too ... I'm curious what that causes!" The output version
writes once, after pooling, so nothing downstream ever sees its push. This version writes into the residual stream
between layers 19 and 20, so layers 20 to 23 carry the push forward and the final embedding is whatever they make of it.

- **The gate is unchanged.** Same two units, signs, float16 rounding and thresholds as Amendment 2 (t19, t17 frozen).
  Both units are computed before layer 20 in the same forward pass (L17's MLP in layer 17, L19's in layer 19), so the
  gate is decided inside one pass from that pass's own means over positions.
- **The write.** A forward hook on `layers[19]` adds `-k * g * R * d` to its output at every position, and only
  when g > 0 (a closed gate adds nothing, so the output is bit-identical). d is the residual-stream chicken direction:
  the mean over positions of `layers[19]`'s output, averaged over the ESC-50 discovery-fold hen and rooster clips,
  minus the same average over the discovery non-chicken clips, normalised. R is the mean per-position L2 norm of that
  output over all discovery clips, so k is in units of the residual's own size. d and R use folds 1 to 3 only.
- **Fitting.** Only k is fitted, by the written block's rule: the fixed grid `K_GRID` (29 log-spaced values, 0.01 to
  100000), the fit AUC of `prompt_scores` between the 228 fit real chickens and the 18 fit imitations, ties to the
  smallest k. Only clips whose gate opens are rerun. Control: the same procedure with a seeded random unit direction
  in place of d (k refitted the same way).
- **Scored** on the same held-out split as E1 to E6 (20 test imitations by 16 held-out imitators against 20 real
  VocalImitationSet chickens; ESC-50 folds 4 and 5 for damage), and later on Elise's Amendment 2 takes (they are fresh
  for this version too, since nothing here is fitted on them). The held-out split was seen twice during the first
  build (README), so I1 is a comparison with E1 on the same clips, not a clean confirmation.
- **Predictions.**
  - I1: the inside version reaches a held-out AUC of at least 0.75. Medium. **Result: held (0.8525).**
  - I2: on ESC-50 folds 4 and 5, no class changes by more than one clip (the gate is the same, so at most the 16 clips
    it opens on can move). Medium-high. **Result: held.**
  - I3: what it causes. Where the output version sends every gated imitation to one class (door_wood_creaks, the pole
    opposite the chicken prompts, seen in the 12:0x meter check), the inside version spreads them: the gated test
    imitations' top ESC-50 classes cover at least 3 different classes, and at least half are human sounds (laughing,
    crying_baby, sneezing, coughing, breathing, snoring, clapping, footsteps, brushing_teeth, drinking_sipping).
    Medium-low. **Result: failed (engine 10, keyboard_typing 2; no human sounds).**
  - I4: the inside version keeps more of the sound: on gated test imitations, the median cosine between the edited and
    the original final embedding is higher than the output version's. Medium. **Result: held as worded (-0.056 vs -0.478; neither keeps the sound).**
  - I5: the random-direction control's held-out AUC is below the inside version's. Medium-high. **Result: held (0.7625).**

## Amendment 4: the gentle inside version (Elise, DM 13:02 CT: "Try the gentle version"; registered 13:03 CT, before any run)

The exploratory ladder (README) showed that at a tenth to a fifth of the residual stream's size, the chicken direction
lowers the chicken score while the edited embedding stays close to the original, and the gated imitations read as
human body sounds. The fitting rule of Amendment 3 never picks such a strength, because it rewards the largest drop.
This version fixes the push to a gentle size and keeps the sound by rule.

- **The gate and direction are unchanged** (Amendment 3's gate, d and R from `inside_params.npz`).
- **The write is binary in size.** When the gate opens, the hook adds `-s * R * d` at every position, the same size for
  every gated clip (Amendment 3 scaled the push by g, which varies about 15-fold across clips).
- **Fitting s.** Grid s in {0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}. Constraint, decided now: the
  median cosine between edited and unedited embeddings over the gated fit imitations must be at least 0.8 ("keeps the
  sound"). Among the sizes that meet it, the one with the highest fit AUC (fit real against fit imitations, the same
  score); ties to the smaller s. Fit split only. Control: the random direction, same rule.
- **A caution, said first.** The ladder was run on the 12 gated test imitations, so the held-out split has now been
  looked at for this question. G1 to G5 are read on it as a comparison, not a clean confirmation; Elise's fresh takes
  (Amendment 2's set) are the clean test for this version too, and the meter will show it.
- **Predictions.**
  - G1: held-out AUC at least 0.75. Medium (a gentle push may leave some imitations above real hens). **Result: held (0.81).**
  - G2: the median cosine to the original over the gated test imitations is at least 0.8. Medium-high. **Result: held (0.834).**
  - G3: at least half of the gated test imitations' top classes are human sounds (Amendment 3's list). Medium, and
    contaminated by the ladder, which showed this on the same clips. **Result: held (10 of 12), contaminated as stated.**
  - G4: the random-direction control's held-out AUC is below the gentle version's. Medium-high. **Result: held (0.5775).**
  - G5: no ESC-50 class in folds 4 and 5 changes by more than one clip. Medium-high. **Result: held.**

## Amendment 5: the same job as a direct weight edit (Elise, DM 13:58 CT: "Yes please!"; registered 14:05 CT, before any run)

Elise asked whether the edit could be made by editing weights directly. The hook versions add a small circuit beside the
model; these two change existing weights in layer 20's MLP and add nothing. Every MLP neuron here computes
`down[:, j] * gelu(gate[j] . x) * (up[j] . x)` on the normed bus x, a product of two readings, so one neuron can carry an
AND. A weight edit acts per position (per slice of sound), not once per clip from the clip's mean: a real difference
from the hook, recorded here before any result.

- **W1, one repurposed neuron.** Layer 20's least-used neuron (smallest mean |activation| over the ESC-50 discovery
  clips, from `acts.npz`) is rewritten. Directions are measured on per-position normed layer-20 inputs from the fit split
  only: `up` row = unit(mean over fit-imitation positions minus mean over keep positions); `gate` row =
  unit(mean over fit-imitation positions minus mean over fit-real-chicken positions) minus theta times the unit mean of
  all fit positions (the threshold, since the MLP has no bias); `down` column = the gentle push, -d scaled so that the
  median fit-imitation position writes s * R. Two numbers are fitted, theta and s, by the gentle rule: the best fit AUC
  among settings whose median cosine over fit imitations stays at 0.8 or more.
- **W2, a rank-one closed-form edit (ROME-style).** Keys k are layer-20 MLP intermediate activations per position.
  C = mean of k k^T over keep and fit-real-chicken positions; k* = mean key over fit-imitation positions. Layer 20's
  `down_proj` gets ΔW = (-s R d)(C^-1 k*)^T / (k*^T C^-1 k*), with s fitted by the same rule. Control: k* from a seeded
  shuffle of the fit real and imitation labels.
- **Scored like the others**, on the same held-out split (with the same caveat: it has been looked at) and on Elise's
  fresh takes when they exist: held-out AUC with bootstrap interval, median cosine to the original over the test
  imitations and their top classes, ESC-50 folds 4 and 5 class damage, and the share of non-chicken clips left within
  cosine 0.999 of the original (a weight edit can leave nothing bit-identical). Descriptive, for Elise's question, for
  every version including the gentle hook: how often people imitating *other* animals (VocalImitationSet "other"
  imitations) change their top class.
- **Predictions.**
  - X1: W1 reaches a held-out AUC of at least 0.75. Medium. **Failed (0.6775).**
  - X2: W2 reaches a held-out AUC of at least 0.75. Medium-low. **Held (0.8125).**
  - X3: W1 leaves at least 80% of ESC-50 fold 4 and 5 non-chicken clips within cosine 0.999. Medium-low (per-position
    gating can leak where the clip-mean gate did not). **Failed (35 of 192, 18%).**
  - X4: W2 leaves fewer of those clips within cosine 0.999 than W1 does. Medium. **Held (0 against 35).**
  - X5: W2's shuffled-label control has a lower held-out AUC than W2. Medium-high. **Held (0.6175 against 0.8125).**

## Amendment 6: Elise's takes scored by every version (registered 15:50 CT, before any take is scored)

Elise sent takes by Discord (15:49 CT) and offered more imitations. Amendment 2 is kept and widened:

- **Test set.** Amendment 2's set, plus takes Elise uploads by Discord, labelled by her in a message before they are
  scored ("real hen", "my imitation", or something else). Route recorded per take (phone recording uploaded to
  Discord, m4a). Excluded: any take already read (her 11-03 files and byte-identical copies, such as
  "brahma_chatty(1).m4a"). Takes with no chicken (for example "background_only") are descriptive only.
- **Versions, all frozen as committed:** unedited; the output gate (Amendment 2's numbers); the inside gate at full
  strength (Amendment 3, `inside_params.npz`); the gentle inside version (Amendment 4, s = 0.2); W1, W2 and the W2
  control (Amendment 5's fitted settings in `weight_edit_results.json`). No parameter changes.
- **Input.** The whole take, decoded as `record.py` does (mono, 16 kHz, up to 300 s).
- **Scoring.** `score_takes.py` gives each take's verdict score and top class under every version. Per-take scores may
  be shown as takes arrive; the AUCs are computed once, when Elise closes the set, and need at least 6 takes of each
  kind (fewer: descriptive only).
- **Predictions** (C1 to C4 stand, for the output gate).
  - C5: W2 separates real-hen takes from her imitations at AUC 0.75 or more. Medium-low.
  - C6: the gentle inside version does, at AUC 0.75 or more. Medium.
  - C7: the gate stays shut on every no-chicken take (output, inside and gentle versions). Medium.

### Amendment 6a: call type (registered 16:15 CT, before any take is scored)

Elise (16:15 CT): "We're still missing a real egg song from them." Five of her six imitations are egg songs and none of
her six real takes is, so an AUC on this set could partly measure egg song against chatter. The set stays open until
real egg-song takes arrive. When it is scored, each take carries a call type (egg song, chatter or peeps, purr,
other) from Elise's description, and the AUCs are also reported within egg songs only (descriptive when either side has
fewer than 3).
