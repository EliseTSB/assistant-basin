# Part 193: can weight statistics alone find our own edits in EmbeddingGemma 2, with and without the original weights?

Procedure: v1
Registered <time from date> CT
Asked by Elise (2026-11-07 23:36 CT, Discord): methods to detect weight edits are valuable, and detecting benign edits is
the same purely defensive technology. Scope: detection and localization only, on edits we made ourselves for benign
reasons. No edit designed to evade a statistic is built.

## Question

We have changed EmbeddingGemma 2's weights directly twice (the chicken imitation edits, Journal
`notes/reports/embeddinggemma2-chicken-neurons-2026-11-03/EDIT-DESIGN.md` Amendment 5): W1 rewrote one neuron of layer 20's
MLP (its gate row, up row and down column), and W2 added a rank-one closed-form (ROME-style) change to layer 20's
`down_proj`. A shuffled-label version of W2 exists as a control edit. Given only the edited weights, can simple per-matrix
and per-neuron statistics say which matrix (and, for W1, which neuron) was changed? At what multiple of the published edit strength
does each statistic start finding the edit, if it does at all within the ladder?

## Prior art and existing data

- Prior art: Journal `notes/reports/weight-edit-detection-2026-11-07/PRIOR-ART.md`. The closest primitive is the rank-1
  energy fraction of the base-to-candidate weight difference (Hurtado, *Has This Checkpoint Been Abliterated?*, arXiv
  2607.01854), averaged over mid-stack matrices and needing the base. *Tracing and Reversing Edits in LLMs* (Youssef et
  al., ICLR 2026, arXiv 2505.20819) localizes ROME edits by row-vector collinearity against the unedited matrices, with a
  strong signal on GPT-2 and GPT-J and none on Llama 3. No published method localizes a single rank-one edit to one matrix
  without the base, and nothing covers Gemma 3n-style encoders or audio embedding models.
- Existing data: the published chicken edit results (`weight_edit_results.json` in the Journal folder above): W1 held-out
  AUC 0.6775 at s = 0.025 and tau quantile 0.3 on neuron 1083; W2 0.8125; W2's shuffled-label control 0.6175. That split
  has been scored before, so these AUCs are a reproduction check here, not a fresh measurement. The script that made them
  (`weight_edit.py`; its log ends `DONE seconds=1896.2 peak_rss_gib=5.69`) does not save the edited weights, so this part
  regenerates them (Method, step 0) and checks them against the published fit.
- The base weights: google/embeddinggemma-2 at snapshot f9972ac22e696a105c482202646979e33e4b5a1e, `model.safetensors`,
  pinned by sha256 (recorded at step 0; a spoofed base makes every difference zero). Every one of its 1,376 tensors is
  stored in bf16; the published script promotes them to float32 on load.
- Pilot before registration, by the cold reviewer of the first draft (DeepSeek V4.1 Flash, 2026-11-08, scripts in the
  review's directory), which looked at the unedited model and at synthetic stand-ins, not at the regenerated edits:
  (a) on F2, layer 20's `down_proj` already ranks 3rd of the 24 `down_proj` matrices with no edit (F2 0.0397; layer 14
  0.0571, layer 23 0.0514), and the reviewer reports that adding a W2-sized delta moves its F2 z by under 0.01 at doses
  up to 2; (b) in layer 20, three neurons (1347, 1584, 1759) have down-column excess kurtosis of 212 to 234 against a
  median of 0.18, which made them dominate the first draft's per-neuron score (robust z 617 to 652); a synthetic
  W1-like neuron scored 87.5, driven by its down-column norm; (c) a synthetic rank-one edit at twice the target's top
  singular value is ranked first by F1 in 24 of 24 placements; (d) because the file is bf16, a candidate saved in bf16
  from the promoted base is bit-identical to it on every unedited matrix, so the first draft's "bf16 rounding
  background" does not exist for this model. The predictions below were written after this pilot and say where it
  informs them.

## Method

CPU only. Weights are read from safetensors in float32; no audio is run except in step 0.

**Step 0, edits.** Run the published `weight_edit.py` unchanged in its fitting, with one added option that saves, for each
arm, the changed tensors: W1's neuron index and its new gate row, up row and down column; W2's and W2-control's delta
matrix for layer 20's `down_proj`. The copy runs from `part193/chicken_edit/` with data paths taken from arguments.
Gate 1 checks the regenerated edits.

**Edited models.** W2 and W2c add t times the saved delta, t in {1/8, 1/2, 1, 2, 4, 8, 16, 32, 64}. W1 moves the neuron's
three vectors from their original values toward the saved ones by t, t in {1/2, 1, 2, 4} (t above 1 extrapolates). Dose
0 is the base. The ladder runs far above the published strength because the pilot suggests the matrix statistics do not
see W2 at it; the question becomes how many times stronger an edit of this shape must be before they do. Held-out family: 24 synthetic rank-one edits, placements seeded (20261108), each adding sigma u v^T to one
language-model matrix chosen uniformly from the population below, u and v uniform random unit vectors, sigma equal to the
Frobenius norm of W2's delta at dose 1. Synthetic edits are scored at dose 1 only.

**Population.** Every two-dimensional weight inside the language model's 24 decoder layers, grouped by its parameter name
within the layer (its role, for example `self_attn.q_proj` or `mlp.down_proj`). Siblings are the same role in the other 23
layers. Matrices outside the decoder layers are recorded but not ranked.

**Reference-free statistics, per matrix** (singular values by full SVD in float64):

- F1, spectral spike: log of the top singular value, as a robust z among the matrix's siblings (median and MAD of the
  siblings, excluding the matrix itself).
- F2, rank-1 energy: sigma_1^2 / sum of sigma_i^2, as a robust z among siblings the same way.

**Reference-free statistics, per neuron** (MLP only, neuron j of a layer): its gate-row norm, up-row norm, down-column
norm, and the cosine of its gate row with its up row. Each feature is a robust z among the layer's neurons; the neuron's
score F3 is the largest absolute z. A layer's F3 is its top neuron's score as a robust z among the 24 layers. The excess
kurtosis of each vector is reported descriptively and kept out of F3, because trained neurons already carry kurtosis
outliers that dominate any score that includes it (pilot (b)).

**Reference-based statistic** (the base is known, its sha256 checked): R1, the set of matrices whose difference from the
base is nonzero, and the rank of each nonzero difference. With this model's bf16 file there is no rounding background
(pilot (d)), so with the base the edit is found exactly; R1 is used as Gate 3 and reported, not predicted. Localizing an
edit among benign fine-tuning changes, the case the literature reports as hard, needs a fine-tuned checkpoint and is a
later part.

**Localization outcome.** For a matrix-level statistic, the edited matrix's rank among its siblings (1 = found) and among
all ranked matrices. For F3, the edited neuron's rank among its layer's neurons and the edited layer's rank among the 24.
For each statistic and edit, the detection threshold is the smallest dose in the ladder at which the edited matrix (or
neuron) ranks first, with every larger dose also ranking first, or "never" if no dose does. "Never" is a result and is
reported as one. F2 of W plus t times a delta is not exactly invariant in t, but for a delta small next to W it barely
moves (pilot (a)); it is reported at every dose like the others.

## Controls

- Gate 1 (the edits are the published ones): the regenerated fit selects the same settings as the published one (W1:
  neuron 1083, tau quantile 0.3, s = 0.025; W2 and W2c: their published s), the held-out AUCs are within 0.0025 of 0.6775,
  0.8125 and 0.6175 (one clip of 400 comparisons), and the script's parity checks pass. Fails if the fit is not
  reproducible on this machine or the copy differs; the part then stops before any statistic is read.
- Gate 2 (do-nothing edit): dose 0 gives every tensor bit-identical to the base, and R1 finds no changed matrix.
- Gate 3 (the edit is where we think): at dose 1, R1 finds exactly layer 20's `down_proj` for W2 and W2c, and exactly
  layer 20's `gate_proj`, `up_proj` and `down_proj` for W1 with one row, one row and one column changed.
- Gate 4 (wiring check): 24 synthetic rank-one edits whose sigma is twice the target matrix's own top singular value are
  each ranked first among their siblings by F1. That size must create a spike, so this checks only that F1, the sibling
  groups and the ranking are wired correctly; it cannot show that a statistic is good. The same 24 placements at 1.2
  times the top singular value, a size where first place is not guaranteed, are reported on F1 and F2 as calibration.
- Null: every statistic on the unedited model, reported for all ranked matrices; for each found edit, the number of
  unedited matrices whose score in the unedited model exceeds the edited matrix's score (the false alarms a threshold set
  at that edit would raise).
- Matched control: W2c is a real ROME-form edit of a similar norm fitted on shuffled labels, so a statistic that finds W2
  at a much lower dose than W2c is reacting to the fitted key direction, not to rank-one size. If neither is found, the
  control is uninformative and says nothing about direction.
- Not used: answer-polarity split, readers, generations (no text is produced). Edits designed to hide from a statistic
  (out of scope by the defensive rule).

## Size

- Smallest effect worth knowing: whether each real edit is found (rank 1 among siblings) at its published strength, and
  the detection threshold within a factor of 4.
- Total: 270 singular-value decompositions plus one regeneration run, about 35 minutes on the CPU. The regeneration is a
  rerun of the published script (1,896 s and 5.7 GiB peak in its log); the statistics cover the 168 ranked base matrices
  and only the changed matrices of each edited model (W2 and W2c at 9 doses, W1's three matrices at 4 doses, 24 synthetic
  and 48 Gate 4 edits), no ranked matrix larger than 2048 by 512.
- Pilot: the reviewer's measurements listed under existing data; no further pilot, since the statistics are deterministic
  and cheap. Step 0 runs first and Gate 1 stops the part if it fails.

## Optimization

- Bound resource: CPU and RAM for step 0; the statistics are seconds each.
- Measured: `weight_edit.py` handled 910 clips in 1,896 s (0.48 clips/s) with 5.7 GiB peak RSS in the published run; its
  replay of only layers 20 to 23 after one full pass is already taken. Float64 singular values of a random 2048 by 512
  matrix: 20 in 2.92 s (about 0.15 s each) on this machine, 2026-11-08, so 270 take under a minute.
- Levers: recompute statistics only for the changed matrix of each edited model, taken (siblings reuse the base values);
  float64 SVD, taken for stable small singular values; step 0 run alone with nothing heavy beside it, taken; skipping the
  regeneration by reusing stored deltas, not available (none were stored).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 193.1 | Without the base, F1's detection threshold for W2 (layer 20's `down_proj` first among the 24 `down_proj` matrices) is dose 8 or higher, or never within the ladder. | medium: the pilot reports W2-sized deltas leave the matrix statistics nearly unmoved up to dose 2. | Claude |
| 193.2 | Without the base, F3 (norms and gate-up cosine) ranks the rewritten neuron 1083 first among layer 20's 2048 MLP neurons at dose 1. | medium-low: a synthetic W1-like neuron stood out on its down-column norm, but the real down column is scaled to a gentle push and could sit inside the normal range. | Claude |
| 193.3 | F1 and F2 do not find W1 at dose 1: none of layer 20's `gate_proj`, `up_proj` or `down_proj` ranks first among its siblings. | medium-high: one neuron of 2048 barely moves a matrix's spectrum. | Claude |
| 193.4 | F1 ranks a synthetic rank-one edit at W2's Frobenius norm (dose 1) first among its siblings in at most 4 of 24 placements. | medium: the pilot suggests an edit of this size is small next to the trained spectrum. | Claude |
| 193.5 | W2's and W2c's F1 detection thresholds are within one ladder step of each other, both "never" counting as equal. | medium: the two deltas are rank one with similar norms and differ in the fitted key direction. | Claude |

## Reading rules

- Gate 1 failing stops the part: no statistic is reported as being about the published edits.
- The pilot looked at the unedited model; predictions it informs say so, and the write-up repeats the pilot numbers beside
  the run's.
- A prediction about dose 1 is read at dose 1 only; thresholds are reported from the whole ladder.
- "First" means rank 1 with no tie; a tie counts as not found.
- The headline is about localization at the published strength, then the threshold multiples. If the synthetic family
  is found and the real edits are not, the headline says the statistics see random rank-one edits and not these fitted
  ones. A natural outlier ranking above an edited matrix or neuron means "not found", whatever the edit's own score.
- Every claim is scoped to this model, these three edits, and these statistics. A detection result here says nothing
  about edits designed to avoid it.

## Limits

- One model and two real edits, both in layer 20's MLP. The main false positive the literature reports, benign low-rank
  fine-tuning, is not tested here, and with the base available this model gives no background at all (pilot (d)).
- Robust z among 23 siblings is a small sample; roles whose layers differ systematically (first and last layers) can
  produce natural outliers, which the null reports.
- W1's dose ladder interpolates three vectors; it is not the same family as scaling a delta.
- The regenerated edits depend on the published fitting being deterministic; Gate 1 checks this rather than assuming it.

## Changes before registration

- Cold review r1 (DeepSeek V4.1 Flash, 2026-11-08): F2 cannot see W2 at the draft's doses and L20 `down_proj` already ranks
  3rd on it, so the old 193.2 and 193.6 could not fail; kurtosis outliers decided the old 193.3; Gate 4 could fail only
  for broken wiring; the bf16 background did not exist; the SVD count and runtime were wrong; Gate 1's tolerance was
  tighter than one clip; the matched control had no reading for "neither found"; the test split had been scored before.
  All folded above; the ladder now runs to 64; kurtosis left F3; R2 was removed; the reviewer's measurements are
  disclosed as a pilot and the predictions rewritten after it.
- The registration time remains `Registered <time from date> CT`; it is stamped from `date` at commit.
