# Part 181: When do OLMo 2 1B's MLP neurons settle on which documents they fire on most, and do they keep them through post-training?

Procedure: v1
DRAFT v2 2026-11-03 by Claude (Opus 5.5). v1 (committed 17:24 as 14de95e) was reviewed by Alden (council
`dr-sol-part181-review-2026-11-03` #3, 17:32 CT; full text at Alden's workspace
`part181-design-review-2026-11-03.md`); v2 folds all six points. Weaver, Muse and Fable (cloud) reviews pending. Not
registered.
Elise, 2026-11-03 15:27 CT, on running the atlas on the Claude cloud credit: "Yes go for it on the Cloud!" Our niche
(Elise, 2026-10-27): following neurons across training stages.

Seen before drafting, and no more: the collector timing pilot (`part181/TIMING.md`, from cloud branch
`cloud/olmo-atlas-pilot-2026-11-03` at 9d6741d, raw numbers in `results/part181/pilot/`). It ran 40 documents on
Instruct and on the 1T pretraining checkpoint and compared fp32 with bf16, batch 4 with batch 8, and the first 20
documents with all 40, each within one checkpoint. No comparison across checkpoints was computed, and no neuron's
windows were read.

## Question

OLMo publishes its whole training path. For every MLP neuron of OLMo 2 1B (16 layers x 8,192 = 131,072), take the
documents it fires on most: its top documents on one fixed set of Pile documents. At which point in training does that
set stop changing, and how much does post-training (SFT, DPO, the final RL round) change it? Do neuron coordinates keep
their top documents better than fixed dense directions in the same space?

The finding is about top-document stability: whether a unit's strongest documents stay the same. It is not a claim that
a unit's meaning or feature is unchanged; the same documents can fire at different tokens, and that is reported beside
it. A median trajectory over a population is not each unit's settling time (Alden). This is the read side only; what a
neuron writes is Parts 85, 92 and 99.

## Prior art

To be filled from Muse's search before registration (council topic `muse-part181-prior-art-2026-11-03`). Known to me
and not re-checked: Tigges et al., arXiv:2407.10827 (circuit components change over Pythia training while the
algorithm stays); Part 85 (our own: L10 N6978's write direction flips sign between 21B and 1T tokens, and the 64 "As
an AI" neurons settle on the same schedule).

## Method

Runs: the 12 checkpoints of `part85/checkpoints.py` (pretraining 21B, 210B, 1T, 2T, 3T, 4T; mid-training 13B, 26B, 51B;
SFT; DPO; Instruct), each at its registered commit prefix, plus two repeat runs on other sessions: `instruct_b`
(Instruct is stored in bf16 and upcast) and `pretrain_1T_b` (the pretraining checkpoints are stored in fp32), 14 runs.
Each run logs its host identity (CPU model string, a hash of the hostname), library versions, dtype and attention
implementation.

Corpus: `NeelNanda/pile-10k`, the parquet at dataset commit `127bfedc…`, file order, the first 2,500 documents, each
truncated to 512 tokens with the OLMo 2 tokenizer (byte-identical tokenizer files across the family; the pilot showed
identical token ids for all 10,000 documents under the Instruct and 1T tokenizers). No BOS (the tokenizer adds none).

Unit: the input to `mlp.down_proj`, `act_fn(gate_proj(x)) * up_proj(x)`, at every token position. Controls: 256 fixed
dense unit directions per layer in that 8,192-space, seeded (seed 5601, per layer), identical at every checkpoint
(never redrawn), projected the same way.

Collector: `part181/olmo_atlas_collect.py` (from the pilot). float32, batch 4, 4 threads, right padding with the
attention mask, padded positions excluded, `--checkpoint-every 100` so that snapshots at 500, 1,000, 1,500, 2,000 and
2,500 documents fall on chunk boundaries (Alden: the default 64 gives only the 2,500 snapshot). Pass 1 on all 2,500
documents: per unit, the maximum and the top 20 documents (one entry per document: that document's maximum, with its
position). Pass 2 on the first 500 documents: the fraction above 10% of the maximum, and random and weak windows (kept
for later reading; no prediction uses them). Every run writes a manifest binding its state to verified inputs: the
full model commit, the corpus file sha256, the sha256 of the token-id lists, the sha256 of each layer's control matrix,
the batch size and chunk size. A run resumes only from state whose manifest matches; timing-only states are separate.

Measures, per unit `u` and run pair `(a, b)`:

- Top-document overlap `O(u; a, b) = |top10_a(u) ∩ top10_b(u)| / 10`, on document ids. Also reported: on (document,
  position), and on top 20. Interpretation, declared now: document overlap with low position agreement is read as "the
  same documents, different firing tokens", not as a stable feature.
- Activity: a unit is active in a run if its maximum is positive and at least 10% of the median maximum of its layer's
  units in that run. If a layer's median maximum is not positive, no unit of that layer is active in that run.

Cohorts (v2, Alden: an activity filter that changes with the stage can manufacture a settling curve):

- Primary, fixed IDs: the units active in the `instruct` run, chosen once. Every trajectory and prediction uses this
  cohort at every stage, whether or not a unit is active at the other stage. Per layer and stage, the coverage
  (cohort units active at that stage) and the active-to-inactive transitions are reported.
- Secondary, fixed IDs: all units.
- Conditional, reported only: units active in both runs of a pair.
- A cohort or layer with fewer than 50 units is reported and excluded from medians; a median over an empty set is
  "not assessable", never a number.

Null (v2, Alden: chance is not 0.004 when units share preferred documents): for each layer, cohort and run pair, each
cohort unit at `a` is paired with a different cohort unit of the same layer at `b`, by a random derangement (seed 181),
100 times; the per-layer null distribution of the median is reported, and every observed median is read against its
own layer's null. 0.004 (top 10 of 2,500 under independent uniform sets) is quoted only as a reference.

## Controls and gates

- Gate 1a, entries recompute (every run): for every layer, 2 neurons and 2 controls at each of ranks 1, 10, 11 and 20
  (128 entries, chosen with seed 181 from cohort units), the entry's document is run alone (batch 1, no padding, no
  bookkeeping) and the activation at the recorded position is recomputed. Each must match to 1e-4 relative, or 1e-6
  absolute when the stored value is below 1e-2 in size.
- Gate 1b, membership on a subset with another batch shape (every run): the first 100 documents are collected again
  with batch 1 and no padding. For every unit, the main run's top-20 entries from those 100 documents must equal the
  100-document run's entries that are at least the main run's 20th value (ties counted as either). A unit failing is
  counted; the gate passes if at most 0.1% of units fail.
- Gate 1c, audit (every run): every stored value finite; document ids unique within each unit's list and below 2,500;
  positions inside the document's length; each list sorted by value, descending.
- Gate 1 is recorded, not retried until green: a failed run is rerun once, and both results are kept and reported.
- Gate 2, identity: each run's manifest is compared across runs: equal corpus sha, token-list sha and control shas, and
  each commit starting with its registered prefix. A run failing it is excluded and named.
- Gate 3, repeat runs (`instruct` against `instruct_b`, and `pretrain_1T` against `pretrain_1T_b`), per layer, on the
  primary cohort: the median neuron overlap, the median control overlap, the 10th percentile, and the agreement of the
  activity masks. A layer passes if its neuron median is at least 0.95 and its activity-mask agreement at least 0.99,
  in both repeat pairs. A failing layer is excluded from every prediction, and named; it is not cleared by a global
  median. The repeats measure two checkpoints' run-to-run repeatability, not a bound on every cross-checkpoint
  difference.
- Saturation, reported: `instruct` keeps snapshots at 500, 1,000, 1,500, 2,000 and 2,500 documents; the median top-10
  overlap between successive snapshots is reported beside the overlap expected from nested uniform sampling alone
  (0.5, 0.667, 0.75, 0.8). It is not a gate: all runs read the same documents.
- Fixtures before any run (tests, not gates): a brute-force top-K fixture on a tiny model with more than 20 documents,
  ties and mixed lengths, checking every rank, with mutations that must make it fail (a dropped larger document, a
  wrong position, a duplicate document).
- Positive pole only: the maximum of a signed activation. The negative pole is not used, stated as a limit.

## Predictions

Author: Claude. Thresholds fixed here. All on the primary cohort, over the layers that pass Gate 3, against each
layer's null.

- 181.1 (post-training changes little of what neurons fire on most; Part 92's weight cosines were about 0.995): the
  median neuron overlap of `midtrain_51B` against `instruct` is at least 0.6. Confidence 0.7.
- 181.2 (the final RL round changes almost nothing; our 10-15 check found neuron correlations of 1.000 across it): the
  median neuron overlap of `dpo` against `instruct` is at least 0.9. Confidence 0.75.
- 181.3 (early instability): the median neuron overlap of `pretrain_21B` against `instruct` is at most 0.1.
  Confidence 0.7.
- 181.4 (a population that settles through pretraining): the median neuron overlap with `instruct` rises strictly along
  21B, 210B, 1T, 2T, 3T, 4T. Confidence 0.6.
- 181.5 (coordinates against fixed dense directions; Alden: this tests top-document stability of neuron coordinates
  against fixed dense projections, not that neurons are uniquely stable features): for each of the pairs (1T, 2T), (4T,
  midtrain_51B) and (midtrain_51B, instruct), the median neuron overlap exceeds the median control overlap in at least
  12 of the layers that pass Gate 3 (or in at least three quarters of them if fewer than 16 pass). Confidence 0.6.
- 181.6 (Part 92's slot neurons fire on the same documents before and after the stage their persona appears in): all
  21 of Part 92's slot neurons are included, active or not. For each, the comparator is the median `sft`-against-`dpo`
  overlap of all primary-cohort neurons in its layer. The prediction holds if the median of the 21 slot overlaps is
  within 0.15 of the median of their 21 layer comparators. Confidence 0.55. Reported with the per-unit values.

Reading rule: the headline "OLMo 2 1B's neurons settle on their top documents during pretraining and keep them through
post-training" needs Gates 1 to 3 (at least 12 layers passing Gate 3), 181.1, 181.3 and 181.4 on the primary cohort,
and the same three directions on the all-units cohort. It is a population statement about top-document stability, not
individual settling times or unchanged features. 181.5 adds "neuron coordinates keep their top documents better than
fixed dense directions"; its failure does not make the units less neuron-like. 181.6 supports no headline alone.

## Size

14 runs, one Claude cloud session each, in parallel (CPU, 4 cores, 16 GB). Per run, from the pilot's measured rates
scaled to the corpus's document length: pass 1 at 2.6 s per document x 2,500 = about 6,500 s; pass 2 at 4.7 s x 500 =
about 2,350 s; Gate 1b at about 2.6 s x 100 = 260 s; Gate 1a 128 single-document forwards, a few minutes; setup about
20 minutes: about 3 hours per run. No GPU. No reader. Each run pushes its compressed state (about 35 MB) and manifest to
its own branch; the analysis reads those files here.

## Optimization

Bound resource: CPU on each cloud session; the sessions run in parallel, so wall time is one run. Measured on real rows
(the pilot): fp32 batch 4 2.30 s per document with bookkeeping, against batch 8 2.81 and bf16 7.63 (no bf16 units on
these CPUs). Levers: batch size measured (4); dtype measured (fp32); pass 2 limited to 500 documents; the forward stops
after the last hooked layer, so the final norm and the output head never run; controls pooled per layer (256, not one
per unit). Declined: drawing reservoir keys only for qualifying positions in pass 2 (about 1.7 s per document, 14
minutes per run), because the change would need new tests and the runs are parallel, so it does not move the wall
time.
