# Part 190: can a low-rank linear edit of EmbeddingGemma 2's output embedding make its class-informed prompt verdict tell real animal sounds from human imitations on animals it was not fitted on, while leaving the rest of the embedding geometry nearly unchanged?

Procedure: v1
Registered 2026-11-04 23:56 CT by Claude (Opus 5.5), before any data. This is the draft time from `date`; the draft was
written in a Claude cloud session (routine "R1", branch `cloud/part190-design-2026-11-04`) and the owner's commit that
adds this file to public main is the registration and sets this line. **Revised to v2 on 2026-11-07 by Claude (Opus
5.5) after a cold read and a second-model review, still before any statistic of the real arrays; see "Changes in v2"
at the end. The v2 commit to public main is the registration.** Elise asked, 2026-11-04 14:58 CT (Discord message
1567644205688557629): "Editing embedded gemmas weights to make it much better at distinguishing animal sounds from
human imitations, without negatively effectively it much if at all outside of that, would be a really good
demonstration of the power of direct weights editing." The routine's brief (2026-11-04) asked for this part as a pure
offline analysis on Part 188's committed arrays, with the design and code written and nothing run on the real data.

## Question

Part 188 found that no single unit separates real animal sounds from human imitations across animals (held-out AUC
0.753 and 0.897, and the units also respond to human-made sound), while a logistic probe on the 768-d output embedding
does (0.918 and 0.965) and the model's own prompt verdict does not (0.631 and 0.808). An earlier chicken-only edit
relabelled 62 of 112 imitations of other animals to human-sound classes rather than animal classes. This part asks
whether a rank 1, 2, 4 or 8 linear edit of the output embedding, fitted on one fold of animals and applied unchanged to
the other (and to the held-out Chicken group), raises the model's own real-against-imitation verdict on held-out
animals toward the probe's 0.92 to 0.97, and what it costs elsewhere: ESC-50 class neighbourhoods, nearest-neighbour
turnover for non-animal sounds, cosines to the committed text prompts, animal identity and drift of imitations toward
human sounds. An edit of the output space is the last linear map before the embedding; folding it into the audio
tower's final projection is a weight edit, so a positive answer here is the cheapest test of the owner's goal before
touching the model.

## Prior art and existing data

- Closest published work: iterative nullspace projection (INLP; Ravfogel, Elazar, Gonen, Twiton and Goldberg, ACL 2020)
  and LEACE (Belrose et al., NeurIPS 2023) find linear directions that carry a concept by repeated probing and remove
  them; this part uses the same repeated-probe construction to find the directions and amplifies instead of removing.
  Chuang et al. 2023 ("Debiasing vision-language models via biased prompts") edit a joint embedding space with a
  projection built from text-prompt directions; here the write direction is built from the model's own text prompts in
  the same way. Rank-one model editing (ROME; Meng et al., NeurIPS 2022) motivates rank-limited edits with off-target
  checks. A 2026 music-retrieval paper ("Steering dense music retrieval with open-vocabulary concept discovery",
  arXiv 2608.08757) scores an edit of a CLAP-like space on whether it moves the target concept and on top-K audio
  neighbours before and after, the closest off-target measure to this part's; Wang et al. ("Transfer learning and bias
  correction with pre-trained audio embeddings", arXiv 2307.10834) apply linear corrections to audio embeddings and
  report within-domain cost. Searched 2026-11-05 ("low-rank linear edit of embedding space improve classification
  preserve neighbours CLAP audio embedding projection debiasing"); no paper edits an audio embedding to separate real
  sounds from human imitations of them.
- Our own data that already bears on it: `results/part188/README.md` and `results/part188/results.json` (verdict 0.631
  and 0.808, probe 0.918 and 0.965, top-32-unit probe 0.905 and 0.957); `results/part188/emb188.npz` holds the final
  embeddings for all 3,138 clips (float16, 3,138 x 768), sequence lengths, and the 32 committed prompt phrases with
  their "the sound of {phrase}" and "a person imitating the sound of {phrase}" embeddings (32 x 768 each);
  `part188/manifest188.csv` and `part188/folds.json` give sets, groups and folds (A: 340 real, 266 imitation; B: 338
  real, 267 imitation; Chicken: 100 real, 38 imitation; E 1,120, H 400, N 269). Text prompts exist only for the animal
  phrases and "an animal", not for the ESC-50 non-animal classes, so off-target cosine changes are measured to these 64
  prompt embeddings only. Only shapes and counts of these arrays were read while writing this design; no statistic
  was computed on `emb188.npz`.

## Method

- Data: `results/part188/emb188.npz` (embeddings cast from float16 to float64 and L2-normalized), its committed prompt
  embeddings, `part188/manifest188.csv` (row order equals the arrays' `ids`) and `part188/folds.json`. No model, no
  audio, CPU only, numpy; the probe fitter, AUC and Chicken cluster labels are imported from `part188/analyze188.py`
  so the probe is Part 188's.
- Directions: each direction fits on one fold F and is tested on the other fold T (A to B, B to A). Labels on F: real
  = analysis set R, imitation = I. Chicken (sets C_R and C_I) is never used for fitting and is a third held-out group
  for both fitted edits.
- Read directions (rank k in 1, 2, 4, 8), by repeated orthogonal probing (INLP-style): probe j is Part 188's L2
  logistic regression (C = 1.0, standardized on F) fitted on F's embeddings with directions 1..j-1 projected out; its
  weight mapped back to raw coordinates (coef / scale), projected off the earlier directions, normalized, and signed
  so that F's real mean projects above F's imitation mean, is w_j. Ranks are nested: rank k uses w_1..w_k of one
  sequence of 8 probes. Centre mu is the midpoint of F's real and imitation
  means. z_j(x) = w_j . (x - mu) / sigma_j, with sigma_j the standard deviation of w_j . x over F.
- Write directions: u_1 = v_bar, the normalized mean over F's distinct clip phrases of (real-prompt embedding minus
  imitation-prompt embedding) ("the sound of {phrase}" minus "a person imitating the sound of {phrase}"), so the
  edit writes into the direction the model's own verdict reads; u_j = w_j for j >= 2 (a stretch of the further
  discriminant directions). rho is the standard deviation of v_bar . x over F.
- Edit: x_tilde = x + alpha * rho * sum_j z_j(x) u_j, then x' = x_tilde / ||x_tilde|| (the model's own output
  normalization). The edit is rank k before normalization, is applied to every one of the 3,138 clips (it cannot know
  which clips are animals), and leaves the text prompt embeddings unchanged. This edit is affine in the
  normalized embedding x, with a nonzero centring term mu, so it is NOT exactly foldable into the audio tower's last
  linear layer: x = y / ||y|| depends on the norm of the tower's output y, and the edit composed with that map is not
  affine in y unless ||y|| is constant. A weight edit would have to absorb the normalization or drop the centring term;
  this part does not build or check either, and its results are about an output-space edit only.
- Dose ladder: alpha in {0, 0.25, 0.5, 1, 2, 4, 8, 16}. The operating dose alpha*(k) for each direction and rank is the
  smallest alpha whose fit-fold verdict AUC (class-informed, in-sample) is at least the ladder maximum minus 0.01. It is
  chosen on F only; every held-out number is read at that dose, and the whole ladder is reported.
- Readouts, all on the edited embeddings with fixed readers (a probe refitted on the edited embeddings is not a readout:
  an invertible linear edit cannot change linear separability, so it would only measure the normalization):
  - V, the model's own verdict, class-informed (primary): x' . (real prompt - imitation prompt) for the clip's own
    phrase, exactly Part 188's `_prompt_scores`. AUC of real against imitation on T, and on Chicken. Class-free ("an
    animal") is reported beside it.
  - K, a text-free geometry readout: for each clip in T's real and imitation sets, the share of real clips among its 10
    nearest neighbours (cosine) within those sets, excluding itself; AUC of real against imitation.
- Collateral damage, at every dose and rank, both directions, on committed data only:
  - C1, ESC-50 class neighbourhoods: leave-one-out 10-nearest-neighbour majority class accuracy (cosine; a tie goes to
    the tied class with the nearest member) over all ESC-50 clips in sets E and H (1,520 clips, 38 classes); change
    from alpha = 0 in percentage points.
  - C2, neighbour turnover for off-target clips (sets E, H and N, 1,789 clips): their 10 nearest neighbours among all
    3,138 clips (excluding themselves) before and after; the share of clips whose first neighbour changes, and the mean
    share of the 10 replaced.
  - C3, movement: mean 1 - cos(x, x') over the off-target clips.
  - C4, prompt cosines: mean absolute change of cosine between each off-target clip and each of the 64 committed
    prompt embeddings.
  - C5, animal identity: zero-shot top-1 accuracy of the clip's own phrase among the "the sound of {phrase}" prompts
    (all committed phrases except "an animal") over every real animal clip (sets R and C_R, 778 clips); change in
    points.
  - C6, drift to human sounds (the chicken edit's failure): the share of T's imitations whose nearest neighbour among
    real recordings (sets R, C_R, E and H) is an H clip; change in points.
- Code: `part190/analyze190.py` (all computation, gates, verdicts; writes `results/part190/results.json`) and
  `part190/run190.py` (the runner: checks inputs, runs the analysis, prints gates and verdicts, never prints arrays).
  Unit tests in `part190/test_part190.py` run on synthetic fixtures only (`part190/synthetic190.py`, which plants a
  known real-against-imitation direction, or none). The run: `python -I part190/run190.py` from the repository root,
  after registration; `--synthetic DIR` runs the same code on a synthetic fixture of the real shape.

## Controls

- Both directions (fit on A test on B, fit on B test on A), and Chicken as a third held-out group for both.
- Matched controls at every rank and dose, each rescaled so that its mean displacement ||x_tilde - x|| over all 3,138
  clips equals the main edit's at that rank and dose:
  - Shuffled labels (the negative control): 20 permutations of the real and imitation labels within each animal group
    of F (so each group keeps its counts), each passed through the whole construction (probes, signs, sigmas, v_bar,
    rho). It must not improve held-out separation.
  - Random directions: 20 draws of k orthonormal read directions (Haar), random signs, with the same write rule (u_1 =
    v_bar, u_j = w_j); matched on displacement, so it moves the embedding as far as the edit without reading the
    concept.
- Dose ladder (above), including alpha = 0, the do-nothing edit.
- Intervals: 2,000 paired group-clustered bootstrap resamples (T's animal groups resampled with replacement; the same
  resample scores the edited and unedited embeddings, so the change has its own interval); Chicken clustered by class
  as in Part 188; C1 clustered by ESC-50 class; C5 by animal group; C6 by T's groups. Intervals are computed for the
  main edit at its operating dose for every rank; controls report point estimates (median and 95th percentile of the
  20 draws).
- Not used: readers (no generations; every readout is a cosine, a rank or a neighbour count); the answer-polarity split
  (no answers); teacher forcing and kernel checks (no model is run).

Gate 1 (data): 3,138 rows, the arrays' ids equal the manifest's order, every embedding finite with norm within 0.01 of
1, and each fold's real and imitation counts equal `part188/folds.json`'s counts_by_analysis_set. Fails on a wrong file,
a reordered manifest or a different fold file.

Gate 2 (identity): at alpha = 0 the edit returns the normalized input (largest absolute difference at most 1e-9); the
unedited class-informed verdict AUC on each test fold reproduces Part 188's `model_class_informed` within 0.005, and the
rank-1 probe (probe 1 above) scored on T reproduces Part 188's `output_embedding_probe` within 0.005 (the tolerance
allows the float16 storage). Fails on a wrong fold, label or phrase mapping or a different probe.

Gate 3 (negative control): for each rank, at the main edit's operating dose, the shuffled-label edits' median held-out
verdict change is at most +0.05 and their 95th percentile at most +0.10, in both directions (v2: looser than
190.4's +0.02, so that 190.4 can fail in a readable run). A failure at rank 1 refuses
the whole reading; a failure at rank 2, 4 or 8 withholds that rank's readings (190.3 is "not read" if rank 8 fails). The same gate function is
applied to the main edit's own held-out changes and recorded as `gate3_self_test`; when the main edit's change exceeds
+0.10 the gate, given those numbers, must report failure, which shows on the real data that it can fail (if the main
edit does not exceed +0.10, the self-test is recorded as "not shown on real data" and only the unit test shows it).
The unit tests show it failing on a leaky control (labels not actually shuffled).

Gate 4 (neighbour sanity): at alpha = 0, C1's 10-nearest-neighbour class accuracy is at least 0.30 (chance is about
0.026 for 38 classes). Fails if the neighbour code or the class labels are wrong.

The analysis refuses to be read if Gate 1, Gate 2 or Gate 3 at rank 1 fails: `results.json` records `"readable": false` and every
prediction's verdict is "not read". If Gate 4 fails, C1 and C2 verdicts are "not read".

## Size

- Smallest effect worth knowing: a held-out verdict AUC rise of 0.05 (half the gap from the verdict to the probe in
  the B to A direction). With about 338 real and 267 imitation clips per test fold the clip-level standard error of an
  AUC near 0.8 is about 0.02, so 0.05 is resolved at the clip level; the group-clustered interval (9 or 10 groups) is
  wider, and 190.13 requires the rank-1 rise's clustered interval to exclude zero. Chicken has 38 imitations, so its
  AUC standard error is about 0.05 and 190.8's bar is +0.10. Every other threshold is at least the stated resolution. Collateral is resolved to about 1 point on 1,520 ESC-50 clips.
- Total: 2,298 edit evaluations (2 directions x 4 ranks x 7 non-zero doses x 41 edits: 1 main, 20 shuffled, 20
  random; plus the 2 unedited baselines, which serve every alpha = 0 row), about 12 minutes on CPU, plus 2,000
  bootstrap resamples for each of the 8 main edits at their operating dose and 2 x 21 sequences of 8 probes.
- Pilot: not needed: no generations, no model, minutes on a CPU; Gates 1 to 4 stop the reading if the rig is wrong.

## Optimization

- Bound resource: CPU (numpy).
- Measured: 2,298 edit evaluations in 710 s (this code with the registered settings, `run190.py --synthetic`, on a
  synthetic fixture of the real shape, 3,157 x 768, 4 CPU threads shared with a pytest run; the real arrays are not
  read before registration), about 0.3 s per evaluation including the bootstraps; the 2 x 21 probe sequences took
  about 90 s per direction.
- Levers: neighbour search as matrix products per edit, taken; controls get point estimates only and the
  bootstrap runs only for the 8 main edits, taken; probes for the shuffled controls fitted once per permutation and
  reused across doses, taken; parallel processes declined (minutes in total).

## Predictions

All at the operating dose chosen on the fit fold, held-out numbers, point estimates; "both directions" means A to B
and B to A.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 190.1 | the rank-1 edit raises the held-out class-informed verdict AUC by at least +0.10 over the unedited verdict in both directions. Fails if either rise is below +0.10. | medium | Claude |
| 190.2 | the rank-1 edited held-out verdict AUC is at least 0.85 in both directions. Fails if either is below 0.85. | low | Claude |
| 190.3 | rank 8 adds less than +0.05 held-out verdict AUC over rank 1 in both directions. Fails if either gain is +0.05 or more. | medium | Claude |
| 190.4 | the shuffled-label rank-1 edits' median held-out verdict change is at most +0.02 in both directions. Fails if either median is above +0.02. | high | Claude |
| 190.5 | the displacement-matched random rank-1 edits' median held-out verdict change is at most +0.02 in both directions. Fails if either median is above +0.02. | high | Claude |
| 190.6 | the rank-1 edit changes ESC-50 leave-one-out 10-NN class accuracy (C1) by less than 2.0 points in absolute value in both directions. Fails at 2.0 points or more either way in either direction. | medium | Claude |
| 190.7 | under the rank-1 edit, fewer than 10% of off-target clips (E, H, N) change their first nearest neighbour (C2) in both directions. Fails at 10% or more in either. | low | Claude |
| 190.8 | the rank-1 edit raises the held-out Chicken verdict AUC by at least +0.10 for both fitted edits. Fails if either rise is below +0.10. | medium | Claude |
| 190.9 | the rank-1 edit lowers zero-shot animal phrase accuracy on real animal clips (C5) by less than 2.0 points in both directions. Fails at a drop of 2.0 points or more in either. | medium | Claude |
| 190.10 | under the rank-1 edit, the share of held-out imitations whose nearest real recording is a human sound (C6) rises by less than 5.0 points in both directions. Fails at a rise of 5.0 points or more in either. | low | Claude |
| 190.11 | the rank-1 edit raises the held-out text-free neighbour AUC (K) by at least +0.03 in both directions. Fails if either rise is below +0.03. | medium | Claude |
| 190.12 | the fit-fold dose choice transfers: at rank 1 the held-out verdict AUC at the operating dose is within 0.03 of its maximum over the ladder in both directions. Fails if either is 0.03 or more below the maximum. | medium | Claude |
| 190.13 | the lower end of the rank-1 held-out verdict rise's paired group-clustered 95% interval is above zero in both directions. Fails if either lower end is zero or below. | medium | Claude |
| 190.14 | the rank-1 edit raises the held-out class-FREE verdict AUC ("an animal" prompts) by at least +0.05 in both directions. Fails if either rise is below +0.05. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 190 (registered ...)".

## Reading rules

- Every number is printed by `part190/analyze190.py` into `results/part190/results.json`; `part190/run190.py` prints the
  gate verdicts and prediction verdicts only. Predictions are scored on point estimates, except 190.13, which is scored on
  the clustered interval's lower end; clustered intervals are reported beside every main number. Each verdict function is the registered threshold, tested on fixtures.
- Ties and edge cases: a verdict whose number is undefined (for example an AUC with one class) is "not read", never
  held or failed.
- The headline reading ("a low-rank edit of the output embedding, fitted on some animals, makes the model's class-informed
  prompt verdict separate real sounds from imitations on other animals, with small collateral damage") is withdrawn if
  Gate 1, 2 or 3 fails; if 190.1, 190.4, 190.5 or 190.13 fails; if in either direction the main rank-1 edit's held-out verdict change does not
  exceed the 95th percentile of both its shuffled and its random controls; or if 190.6 or 190.9 fails (then the reading
  is "separation at a cost", stated with the costs).
- If 190.10 fails, the result is reported as an edit that partly moves imitations toward human sounds, the failure the
  chicken edit showed, whatever the verdict gain.
- Any headline states what the edit is: the fit-fold probe's reading written along the verdict's own prompt
  direction, so a held 190.1 is "the probe's knowledge transferred into the verdict on held-out animals", not "the
  model learned to tell imitations apart". It states that the verdict is class-informed (the clip's own phrase picks
  the prompt pair) unless 190.14 also holds.
- The ladder is a trade-off curve: a reading at any dose other than the operating dose is labelled exploratory. One
  exploratory line is always reported: the smallest dose on the rank-1 ladder whose held-out verdict rise reaches half
  the operating dose's rise, with its C1, C2, C5 and C6 numbers, because a cost measured at the dose chosen for maximum
  separation is not evidence that no cheaper dose exists.
  Anything else found after the results (another rank's dose, single groups) is exploratory.

## Limits

- An edit of the output embedding, not of the model's weights. Part 188 recorded only the final normalized embedding,
  so the edit is applied after normalization and renormalized; because it is affine in normalized coordinates with a
  nonzero centring term, it does not fold exactly into the audio tower's last linear layer (see Method). A positive
  result says an output-space edit works; a weight edit is a further step.
- Collateral is measured only on the committed clips: ESC-50 (40 per class), VocalImitationSet, and the 64 committed
  animal prompts. Text queries for non-animal classes, speech, music and other audio are not measured; a real weight
  edit would need them.
- The verdict at large alpha becomes the probe's score written along v_bar; the part measures whether that transfers
  across animal groups and what it costs, not whether the model "understands" imitation.
- Folds have 9 and 10 animal groups; the clustered intervals are wide and "held-out animals" means these groups.
- The edit is fitted on recordings whose imitations come from one web collection (VocalImitationSet), so it may read
  recording conditions; set N (non-animal imitations from the same collection) and C6 bound that only partly.
- Part 188's embeddings are stored as float16; Gate 2's tolerance allows for that.

## Changes before registration

Draft written 2026-11-04 23:56 CT (05:56 UTC 2026-11-05) in a cloud session; the predictions were committed (e22888c)
before any analysis code. Changes after that commit and before registration, made from synthetic fixtures only (no
statistic of the real arrays was computed): the Size total counts the 2 shared unedited baselines once (2,298, was
2,624, which counted alpha = 0 for every edit); ranks are stated as nested; Gate 3 is read per rank (a d = 48 synthetic
fixture showed shuffled rank-8 edits with a 95th-percentile change above +0.05 from spread alone, at 10 controls, so a
rank-8 failure now withholds 190.3 and the rank-8 readings rather than the whole part; a rank-1 failure still refuses
the reading); the Measured line was filled from a synthetic run of the real shape, on which all four gates passed.
No prediction or threshold changed.

Not done in this session (routine time cap): no code review by a second model; no cold read of the design.

## Changes in v2 (2026-11-07, before registration, no statistic of the real arrays computed)

From the cold read and a Haiku 5.5 second-model review (`Journal/notes/reports/part190-cold-read-2026-11-05.md`):

1. **190.4 could not fail in a readable run**: its bar (+0.02) was Gate 3's rank-1 median bar, so a violation made the
   whole run unreadable. Gate 3 is now +0.05 median and +0.10 95th percentile; 190.4 keeps +0.02. A unit test shows a
   median of +0.03 leaves the run readable and fails 190.4.
2. **Thresholds at or below their own resolution**: 190.3 +0.02 to +0.05; 190.6 1.0 to 2.0 points; 190.8 +0.05 to +0.10
   (38 Chicken imitations, standard error about 0.05).
3. **Point estimates against a clustered interval called honest**: new 190.13 requires the rank-1 rise's paired
   clustered 95% interval to exclude zero, and the headline needs it.
4. **Class-informed verdict called "the model's own"**: the title, headline and reading rules now say class-informed;
   new 190.14 predicts the class-free verdict's rise (low confidence).
5. **The foldability sentence was false as stated**: rewritten in Method and Limits.
6. **Operating dose maximises separation, not separation per cost**: one exploratory half-gain dose line is reported.
7. The headline names the edit as the probe's reading written along the verdict's direction.

The reviewer's point that the random control keeps the write direction v_bar was checked against the code
(`random_edit`): the control's coefficient comes from a random read direction standardized to unit variance on the
fit fold and then rescaled to the main edit's mean displacement, so its per-clip spread along v_bar is comparable to
the main edit's and what differs is its correlation with the labels. Recorded as the reason it was not changed.
