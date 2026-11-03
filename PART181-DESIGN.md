# Part 181: When do OLMo 2 1B's MLP neurons settle on what they read, and do they keep it through post-training?

Procedure: v1
DRAFT v1 2026-11-03 by Claude (Opus 5.5). Not registered.
Elise, 2026-11-03 15:27 CT, on running the atlas on the Claude cloud credit: "Yes go for it on the Cloud!" Our niche
(Elise, 2026-10-27): following neurons across training stages.

Seen before drafting, and no more: the collector timing pilot (`part181/TIMING.md`, from cloud branch
`cloud/olmo-atlas-pilot-2026-11-03` at 9d6741d, raw numbers in `results/part181/pilot/`). It ran 40 documents on
Instruct and on the 1T pretraining checkpoint and compared fp32 with bf16, batch 4 with batch 8, and the first 20
documents with all 40, each within one checkpoint. No comparison across checkpoints was computed, and no neuron's
windows were read.

## Question

OLMo publishes its whole training path. For every MLP neuron of OLMo 2 1B (16 layers x 8,192 = 131,072), take what it
reads: its top-activating contexts on one fixed set of Pile documents. At which point in training does that set stop
changing, and how much does post-training (SFT, DPO, the final RL round) change it? Do neurons keep their contexts
better than random directions in the same space, as they should if the neuron basis is privileged?

This is the read side only. What a neuron writes (its effect when pushed) is Parts 85, 92 and 99. Part 92 found
concept neurons show their persona from pretraining and slot fillers mostly from DPO; this part asks whether the
read side moves at those stages.

## Prior art

To be filled from Muse's search before registration (council topic `muse-part181-prior-art-2026-11-03`). Known to me
and not re-checked: Tigges et al., arXiv:2407.10827 (circuit components change over Pythia training while the
algorithm stays); Part 85 (our own: L10 N6978's write direction flips sign between 21B and 1T tokens, and the 64 "As
an AI" neurons settle on the same schedule).

## Method

Checkpoints: the 12 of `part85/checkpoints.py` (pretraining 21B, 210B, 1T, 2T, 3T, 4T; mid-training 13B, 26B, 51B;
SFT; DPO; Instruct), each at its registered commit prefix, plus a second, independent run of Instruct on another
session (`instruct_b`), 13 runs.

Corpus: `NeelNanda/pile-10k`, the parquet at dataset commit `127bfedc…`, file order, the first 2,500 documents, each
truncated to 512 tokens with the OLMo 2 tokenizer (byte-identical tokenizer files across the family; the pilot showed
identical token ids for all 10,000 documents under the Instruct and 1T tokenizers). No BOS (the tokenizer adds none).

Unit: the input to `mlp.down_proj`, `act_fn(gate_proj(x)) * up_proj(x)`, at every token position. Controls: 256
random unit directions per layer in that 8,192-space, seeded (seed 5601, per layer), identical at every checkpoint,
projected the same way.

Collector: `part181/olmo_atlas_collect.py` (from the pilot; 4 tests pass on the host 2026-11-03). float32, batch 4, 4
threads, right padding with the attention mask, padded positions excluded. Pass 1 on all 2,500 documents: per unit,
the maximum and the top 20 documents (one entry per document: that document's maximum, with its position). Pass 2 on
the first 500 documents: the fraction of positions above 10% of the maximum, and the random and weak windows (kept for
later reading; no prediction uses them).

Measures, per unit `u` and checkpoint pair `(a, b)`:

- Top-10 overlap `O(u; a, b) = |top10_a(u) ∩ top10_b(u)| / 10`, on document ids. Also reported: on (document,
  position), and with top 20.
- Active: a unit is active at a checkpoint if its maximum is at least 10% of the median maximum of its layer's units at
  that checkpoint. Overlaps are summarised on units active at both checkpoints; the inactive counts are reported.
- Summaries: median `O` over active units, per layer and over all layers, for every pair of the 13 runs.

Chance, computed offline from the same files: each unit at `a` paired with a random other unit of the same layer at
`b` (100 permutations, seed 181). With 2,500 documents and top 10, chance overlap is about 0.004.

## Controls and gates

- Gate 1, collector against a plain forward (every run): after pass 1, 32 (unit, top-1 document) pairs chosen with
  seed 181 are recomputed by a forward of that document alone (batch 1, no padding, no bookkeeping). Each recomputed
  activation must match the collector's to 1e-4 relative. A run that fails is not used and is rerun.
- Gate 2, identity: each run logs the resolved full commit (must start with the registered prefix), the sha256 of the
  token-id lists of the 2,500 documents (must be equal across all 13 runs), and the sha256 of the control matrix (must
  be equal across all 13).
- Gate 3, noise floor: `instruct` against `instruct_b`, two sessions on two machines. The median neuron overlap must be
  at least 0.95. If it is lower, every cross-checkpoint difference smaller than `1 - floor` is reported as within
  noise, and 181.2 cannot hold.
- Controls beside every neuron number: the 256 random directions per layer, and the shuffled-unit chance.
- Saturation, reported: `instruct` keeps snapshots at 500, 1,000, 1,500, 2,000 and 2,500 documents; the median top-10
  overlap between successive snapshots is reported. It is not a gate: all 13 runs read the same documents, so a
  difference between checkpoints is not a sampling difference.
- Positive pole only: the maximum of a signed activation. The negative pole is not used, stated as a limit.

## Predictions

Author: Claude. Thresholds fixed here.

- 181.1 (post-training changes little of what neurons read; Part 92's weight cosines were about 0.995): the median
  neuron overlap of `midtrain_51B` against `instruct` is at least 0.6. Confidence 0.7.
- 181.2 (the final RL round changes almost nothing; our 10-15 check found neuron correlations of 1.000 across it): the
  median neuron overlap of `dpo` against `instruct` is at least 0.9. Confidence 0.75.
- 181.3 (early instability): the median neuron overlap of `pretrain_21B` against `instruct` is at most 0.1.
  Confidence 0.7.
- 181.4 (settling is monotone through pretraining): the median neuron overlap with `instruct` rises strictly along
  21B, 210B, 1T, 2T, 3T, 4T. Confidence 0.6.
- 181.5 (a privileged basis): for each of the pairs (1T, 2T), (4T, midtrain_51B) and (midtrain_51B, instruct), the
  median neuron overlap exceeds the median control overlap in at least 12 of 16 layers. Confidence 0.6.
- 181.6 (Part 92's slot neurons read the same text before and after the stage their persona appears in): over Part 92's
  21 slot neurons, the median overlap of `sft` against `dpo` is within 0.15 of the median over all active neurons in
  their layers. Confidence 0.55. (21 units; reported with the per-unit values.)

Reading rule: the headline "OLMo 2 1B's neurons settle on what they read during pretraining and keep it through
post-training" needs Gates 1 to 3, 181.1, 181.3 and 181.4. Without 181.5, the headline drops "neurons" for "MLP
activations", because a fixed random direction would do as well. 181.6 is reported with its 21 units and supports no
headline alone.

## Size

13 runs, one Claude cloud session each, in parallel (CPU, 4 cores, 16 GB). Per run, from the pilot's measured rates
scaled to the corpus's document length: pass 1 at 2.6 s per document x 2,500 = about 6,500 s; pass 2 at 4.7 s x 500 =
about 2,350 s; Gate 1 and setup about 20 minutes: about 2.8 hours per run. No GPU. No reader. Each run pushes its
compact state (`state.npz`, about 57 MB) to its own branch; the analysis reads those files here.

## Optimization

Bound resource: CPU on each cloud session; the sessions run in parallel, so wall time is one run. Measured on real rows
(the pilot): fp32 batch 4 2.30 s per document with bookkeeping, against batch 8 2.81 and bf16 7.63 (no bf16 units on
these CPUs). Levers: batch size measured (4); dtype measured (fp32); pass 2 limited to 500 documents; the forward stops
after the last hooked layer, so the final norm and the output head never run; controls pooled per layer (256, not one
per unit). Declined: drawing reservoir keys only for qualifying positions in pass 2 (about 1.7 s per document, 14
minutes per run), because the change would need new tests and the runs are parallel, so it does not move the wall
time.
