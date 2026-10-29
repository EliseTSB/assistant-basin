# Part 102: did the attention the slot neurons need come from SFT or from DPO?

Size: 3 runs x 474 greedy generations = 1,422 generations on the A770 (about 7 GPU-minutes at Part 101's 2 min per
run), plus 3 x 258 = 774 Haiku labels on the read set (about $0.60).

Registered 2026-10-29 before any run, by Claude (Opus 5.5). Joins two earlier results. Part 92: the 21 slot units'
pushes become self-descriptions mostly at DPO (in the chat frame, 8 of 21 at SFT, 14 of 20 at DPO, 19 of 21 at
Instruct). Parts 100 and 101: those self-descriptions need Instruct's attention in layers 0 to 3 (the mid-training
end's attention there leaves 3 of 21; layer 0 alone, 7 of 21).

## Runs

Host OLMo-2-0425-1B-Instruct (48d788e), swap `attn`, everything else as Part 101 (`part102/steps102.txt`):

| Tag | Donor | Attention swapped in layers |
|---|---|---|
| attnSFT_L0-3 | OLMo-2-0425-1B-SFT (0d85a3d) | 0 to 3 |
| attnDPO_L0-3 | OLMo-2-0425-1B-DPO (c4b0485) | 0 to 3 |
| attnB_L1-3 | the mid-training end (f83cd64) | 1 to 3, Instruct's layer 0 kept (Part 101's named next step) |

References, not rerun: Part 99's `inst_none` (19 of 21), Part 100's `attnB_L0-3` (3 of 21), Part 101's single layers.

## Covariate, measured before any run

`part102/attn_delta102.py` (torch, because the SFT and DPO checkpoints are pickled; the mid-training end measured again
matches Part 101's numpy values exactly): distance of each stage's attention from Instruct's, relative Frobenius norm,
layers 0 to 3. Mid-training end: 7.8, 8.2, 8.2, 8.0%. SFT: 0.46, 0.43, 0.42, 0.40%. DPO: 0.25, 0.25, 0.23, 0.22%.
SFT's attention is about 20 times closer to Instruct's than the mid-training end's, in every layer (all 16: 0.41%
against 8.0% on average). Values: `results/part102/attn-delta-stages.json`.

## Labels and scores

As Part 101: `part99/label_items99.py`, cut to the read set by `part101/read101.py filter`, Part 92's blind Haiku
reader in two passes, scored by `part102/score102.py` with Part 101's scorer. No relabel control this time; Part 101's
measured 98.4% agreement and no change in the slot count.

## Predictions (Claude, before any run)

| # | Prediction | Confidence |
|---|---|---|
| 102.1 | SFT's early attention is already enough: with SFT's attention in layers 0 to 3, slot_who is at least 15 of 21. | low |
| 102.2 | DPO's early attention is enough: with DPO's attention in layers 0 to 3, slot_who is at least 15 of 21. | medium |
| 102.3 | Instruct's layer 0 alone does not rescue them: with the mid-training end's attention in layers 1 to 3, slot_who is at most 9 of 21. | medium |
| 102.4 | No hybrid damage: every run keeps slot_who_coherent at 13 of 21 or more. | medium |
| 102.5 | Every run keeps concept_any at 13 or more of its eligible concept units. | medium |

Basis for 102.1, stated so its failure is readable: after SFT, attention moves only 0.4%, so if DPO's step lived in
early attention it would be a large effect from a small change. Smallest effect worth knowing: a drop of 5 units from
19 (slot_who 14 or less), well above the judge noise Part 101 measured.

## What comes next, decided now

- 102.1 and 102.2 both hold: the early attention the slot neurons need was in place by SFT, so DPO's step (Part 92)
  lies elsewhere; next is the same stage swap for the MLPs (SFT's and DPO's MLPs into Instruct, all layers, then
  blocks).
- 102.1 fails and 102.2 holds: the small post-SFT change in early attention matters, and it came by DPO; next is
  SFT's attention in single layers 0 to 3.
- 102.2 fails: RLVR's even smaller change matters; next is DPO's attention in single layers 0 to 3.

Each next step gets its own registered predictions before it runs.

## Prior art

As Part 101 (Wu et al. 2024, arXiv:2310.00492; Sun et al. 2026, arXiv:2604.13694); for the stage comparison, Part 92's
own ladder. None of these swaps one post-training stage's attention into the next stage's model.

## Limits

One model family, one fixed push per unit, one judge; 21 slot units; a stage's attention is tested inside Instruct's
other weights, which it was never trained with.

## Results (09:21 CT, after the run)

102.1 failed (SFT's attention in layers 0 to 3: 13 of 20), 102.2 held (DPO's: 17 of 20), 102.3 held (layers 1 to 3 only: 9 of 21), 102.4 and 102.5 held. Unit by unit, SFT's early attention restores 10 of the units the mid-training end's loses (p = 0.002); SFT to DPO adds 4 (p = 0.125). The next step departs from the rule above, for the reason given in results/part102/README.md.
