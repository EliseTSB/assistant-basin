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
and per-neuron statistics say which matrix (and, for W1, which neuron) was changed? Given the original weights as well, does
the rank-one structure of the difference still stand out when every matrix also differs by bf16 rounding? At what fraction
of the published edit strength does each statistic stop finding the edit?

## Prior art and existing data

- Prior art: Journal `notes/reports/weight-edit-detection-2026-11-07/PRIOR-ART.md`. The closest primitive is the rank-1
  energy fraction of the base-to-candidate weight difference (Hurtado, *Has This Checkpoint Been Abliterated?*, arXiv
  2607.01854), averaged over mid-stack matrices and needing the base. *Tracing and Reversing Edits in LLMs* (Youssef et
  al., ICLR 2026, arXiv 2505.20819) localizes ROME edits by row-vector collinearity against the unedited matrices, with a
  strong signal on GPT-2 and GPT-J and none on Llama 3. No published method localizes a single rank-one edit to one matrix
  without the base, and nothing covers Gemma 3n-style encoders or audio embedding models.
- Existing data: the published chicken edit results (`weight_edit_results.json` in the Journal folder above): W1 held-out
  AUC 0.6775 at s = 0.025 and tau quantile 0.3 on neuron 1083; W2 0.8125; W2's shuffled-label control 0.6175. The script
  that made them (`weight_edit.py`, 1,945 s and 5.7 GiB peak on the CPU) does not save the edited weights, so this part
  regenerates them (Method, step 0) and checks them against those numbers.
- The base weights: google/embeddinggemma-2 at snapshot f9972ac22e696a105c482202646979e33e4b5a1e, `model.safetensors`,
  pinned by sha256 (recorded at step 0; a spoofed base makes every difference zero).
- No statistic in this design has been computed on these weights before registration.

## Method

CPU only. Weights are read from safetensors in float32; no audio is run except in step 0.

**Step 0, edits.** Run the published `weight_edit.py` unchanged in its fitting, with one added option that saves, for each
arm, the changed tensors: W1's neuron index and its new gate row, up row and down column; W2's and W2-control's delta
matrix for layer 20's `down_proj`. The copy runs from `part193/chicken_edit/` with data paths taken from arguments.
Gate 1 checks the regenerated edits.

**Edited models.** For edit E in {W1, W2, W2c} and dose t in {1/32, 1/8, 1/2, 1, 2}: W2 and W2c add t times the saved delta;
W1 moves the neuron's three vectors from their original values toward the saved ones by t (t above 1 extrapolates). Dose
0 is the base. Held-out family: 24 synthetic rank-one edits, placements seeded (20261108), each adding sigma u v^T to one
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
norm, cosine of gate row with up row, and the excess kurtosis of the entries of each of the three vectors. Each feature is
a robust z among the layer's neurons; the neuron's score F3 is the largest absolute z. A layer's F3 is its top neuron's
score as a robust z among the 24 layers.

**Reference-based statistics** (the base is known, its sha256 checked): R1, the set of matrices whose difference from the
base is nonzero. R2, the rank-1 energy fraction of the difference, per matrix, when the candidate was saved in bf16 and the
base is the float32 release, so every matrix differs by rounding and the edit has to stand out above it. For R2 the edited
model is rounded to bf16 as a whole after the edit.

**Localization outcome.** For a matrix-level statistic, the edited matrix's rank among its siblings (1 = found) and among
all ranked matrices. For F3, the edited neuron's rank among its layer's neurons and the edited layer's rank among the 24.
For each statistic and edit, the detection threshold is the smallest dose in the ladder at which the edited matrix (or
neuron) ranks first, with every larger dose also ranking first.

## Controls

- Gate 1 (the edits are the published ones): the regenerated W1, W2 and W2c reproduce the published held-out AUCs (0.6775,
  0.8125, 0.6175) within 0.0005, and the script's parity checks pass. Fails if fitting is not deterministic or the copy
  differs; the part then stops before any statistic is read.
- Gate 2 (do-nothing edit): dose 0 gives every tensor bit-identical to the base, and R1 finds no changed matrix.
- Gate 3 (the edit is where we think): at dose 1, R1 finds exactly layer 20's `down_proj` for W2 and W2c, and exactly
  layer 20's `gate_proj`, `up_proj` and `down_proj` for W1 with one row, one row and one column changed.
- Gate 4 (the statistics can see an edit): 24 synthetic rank-one edits whose sigma is twice the target matrix's own top
  singular value are each ranked first among their siblings by F1. That size must create a spike, so a failure means the
  statistics or the population are wrong, not that edits are hard to see.
- Null: every statistic on the unedited model, reported for all ranked matrices; for each found edit, the number of
  unedited matrices whose score in the unedited model exceeds the edited matrix's score (the false alarms a threshold set
  at that edit would raise).
- Matched control: W2c is a real ROME-form edit of the same norm family fitted on shuffled labels, so a statistic that
  finds W2 but not W2c is reacting to the fitted key direction, not to rank-one structure.
- Not used: answer-polarity split, readers, generations (no text is produced). Edits designed to hide from a statistic
  (out of scope by the defensive rule).

## Size

- Smallest effect worth knowing: whether each real edit is found (rank 1 among siblings) at its published strength, and
  the detection threshold within a factor of 4.
- Total: 220 singular-value decompositions plus one regeneration run, about 40 minutes on the CPU. The regeneration is
  a rerun of the published script (1,945 s and 5.7 GiB peak, measured); the statistics cover about 170 base matrices and
  only the changed matrix of each edited model (3 edits at 5 doses, 24 synthetic and 24 Gate 4 edits, each with a bf16
  copy), no matrix larger than 2048 by 512.
- Pilot: none; the statistics are deterministic and cheap. Step 0 runs first and Gate 1 stops the part if it fails.

## Optimization

- Bound resource: CPU and RAM for step 0; the statistics are seconds each.
- Measured: `weight_edit.py` handled 910 clips in 1,945 s (0.47 clips/s) with 5.7 GiB peak RSS in the published run; its
  replay of only layers 20 to 23 after one full pass is already taken. Float64 singular values of a random 2048 by 512
  matrix: 20 in 2.92 s (about 0.15 s each) on this machine, 2026-11-08, so 220 take under a minute.
- Levers: recompute statistics only for the changed matrix of each edited model, taken (siblings reuse the base values);
  float64 SVD, taken for stable small singular values; step 0 run alone with nothing heavy beside it, taken; skipping the
  regeneration by reusing stored deltas, not available (none were stored).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 193.1 | With the float32 base and a bf16-saved candidate, R2 ranks layer 20's `down_proj` first of all ranked matrices for W2 at every dose in the ladder. | medium-high: rounding noise is spread over every direction, a rank-one change is not. | Claude |
| 193.2 | Without the base, F1 or F2 ranks layer 20's `down_proj` first among the 24 `down_proj` matrices for W2 at dose 1. | low: the edit's size relative to the trained spectrum is not known before the run. | Claude |
| 193.3 | Without the base, F3 ranks the rewritten neuron first among layer 20's MLP neurons for W1 at dose 1. | medium: its rows are built from activation means and normalised, unlike trained rows. | Claude |
| 193.4 | F1 and F2 do not find W1 at dose 1: none of layer 20's `gate_proj`, `up_proj` or `down_proj` ranks first among its siblings. | medium: one neuron of 2048 barely moves a matrix's spectrum. | Claude |
| 193.5 | F2 ranks a synthetic rank-one edit at matched norm first among its siblings in at least 16 of 24 placements. | low-medium. | Claude |
| 193.6 | W2's detection threshold on F2 (reference-free) is no lower than W2c's: the statistic reacts to rank-one size, not to the fitted direction. | medium. | Claude |

## Reading rules

- Gate 1 failing stops the part: no statistic is reported as being about the published edits.
- A prediction about dose 1 is read at dose 1 only; thresholds are reported from the whole ladder.
- "First" means rank 1 with no tie; a tie counts as not found.
- The headline is about localization at the published strength. If the synthetic family is found and the real edits are
  not, the headline says the statistics see random rank-one edits and not these fitted ones.
- Every claim is scoped to this model, these three edits, and these statistics. A detection result here says nothing
  about edits designed to avoid it.

## Limits

- One model and two real edits, both in layer 20's MLP. The main false positive the literature reports, benign low-rank
  fine-tuning, is not tested here; R2's background is rounding only.
- Robust z among 23 siblings is a small sample; roles whose layers differ systematically (first and last layers) can
  produce natural outliers, which the null reports.
- W1's dose ladder interpolates three vectors; it is not the same family as scaling a delta.
- The regenerated edits depend on the published fitting being deterministic; Gate 1 checks this rather than assuming it.

## Changes before registration

- None yet. The registration time remains `Registered <time from date> CT`; it is stamped from `date` at commit.
