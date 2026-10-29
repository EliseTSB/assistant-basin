# Part 107: where does DPO's small weight change sit, and does it touch how later layers read the slot units' pushes?

Registered 2026-10-29 by Claude (Opus 5.5), with Part 106 and before any Part 107 run. The commit that adds this file
is the registration. Same question from Elise as Part 106: what do SFT and DPO do here, and how does it relate to the
neurons?

## Question

Part 89 measured the MLP neurons only: from the mid-training end to SFT the median down-column cosine distance is
0.005; from SFT to DPO 0.000008; from DPO to Instruct 0.000003. Part 92 found most slot units start showing at DPO.
Part 107 asks three things of the weights alone (CPU, no forward passes):

1. How big is each stage's change in every component group (attention, attention norms, MLP, layer norms, final norm,
   embeddings, unembedding), not only the MLP?
2. Is DPO's stored change at the bf16 precision floor? The checkpoints are stored in bf16, whose relative step is about
   0.4%, and a cosine distance of 0.000008 corresponds to roughly that per-element relative change. If most changed
   elements moved by exactly one bf16 step, the stored DPO change is mostly rounding, and its size says little about the
   true update.
3. Does DPO change how later layers read the slot units' push directions more than other directions? For a unit at
   layer L with push direction u (its SFT down column, normalised), the reading change of a stage A to B is
   R = sum over readers of ||(M_B - M_A) u||^2 / sum of ||M_A u||^2, the readers being the q, k, v, gate and up matrices of
   every layer after L (OLMo 2 has no pre-norm, so these read the residual stream directly). The unembedding, through
   the final norm, is reported separately.

## Method

`part107/where107.py`, weights only, Part 89's loader (`part89/cos89.py`: `weights_file`, `Weights`) and checkpoints
(mid-training end 51B, SFT, DPO, Instruct). Units: Part 92's 21 slot and 18 concept units. Baseline: 64 seeded random
neurons from each unit's own layer, whose down columns go through the same R. Embedding and unembedding rows are compared
on the rows both checkpoints share. Scorer `part107/score107.py`, committed with this file.

## Predictions (also in PREDICTIONS.md)

- **107.1** (Claude, medium confidence): In every component group, DPO's relative change (Frobenius norm of the change
  over the norm of the SFT weights) is at least 10 times smaller than SFT's (over the mid-training end's weights).
- **107.2** (low): DPO's stored change is at the precision floor: over the MLP and attention matrices, at least half of
  the elements DPO changed moved by exactly one bf16 step.
- **107.3** (low): DPO's change touches the slot units' directions more than the concept units': the median slot R for
  SFT to DPO is at least 1.2 times the median concept R, and at least 14 of the 21 slot units have R above the median of
  their own layer's 64 random neurons.

## Reading rule, set before the run

If 107.2 holds, 107.3 is read as a test of rounding, not of the DPO update, and no claim about where DPO acts is made
from R. If 107.2 fails and 107.3 holds, the record says DPO's change is aimed at the slot units' read paths; if both
fail, it says DPO's change does not single out these directions in the weights.

## Prior art

Lee et al. 2024 (arXiv 2401.01967, abstract checked): DPO bypasses rather than removes, with small changes. Mukherjee et
al. 2025 (RL fine-tunes small subnetworks; not checked here) report that RL updates leave most parameters unchanged,
which bears on 107.2's precision-floor question.

## Limits

Weights only: a reading change in the weights is not a change in function, and attention reading goes through softmax,
which R ignores. One model; four checkpoints; bf16 storage hides any change smaller than one step.
