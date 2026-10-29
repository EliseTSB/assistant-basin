# Part 103: where do SFT's and DPO's missing pieces sit, attention or MLPs?

Size: 6 runs x 474 greedy generations = 2,844 generations on the A770 (about 13 GPU-minutes at 2 min per run), plus
6 x 258 = 1,548 Haiku labels on the read set (about $1.20). Each run is its own condition, so none can be a pilot for
the others; all are short and local.

Registered 2026-10-29 before any run, by Claude (Opus 5.5). Follows Part 102: SFT's attention in layers 0 to 3 left 13
of 20 slot units showing their persona in Instruct, DPO's 17 of 20, Instruct itself 19 of 21. Part 92's ladder (pushes
built from each checkpoint's own residual and down column, not Part 99's fixed vectors) gave the whole SFT model 8 of 21 and the whole DPO model 14 of 20, so the post-SFT and
post-DPO changes that matter are not all in early attention. Part 103 splits them by component, over all 16 layers.

## Runs

Host OLMo-2-0425-1B-Instruct (48d788e) unless named; everything else as Part 101 (`part103/steps103.txt`):

| Tag | What | Stage it tests |
|---|---|---|
| attnSFT_all | SFT's attention in all 16 layers | post-SFT change in attention |
| mlpSFT_all | SFT's MLPs in all 16 layers | post-SFT change in MLPs |
| attnDPO_all | DPO's attention in all 16 layers | post-DPO (RLVR) change in attention |
| mlpDPO_all | DPO's MLPs in all 16 layers | post-DPO (RLVR) change in MLPs |
| whole_SFT | the whole SFT model as host, no swap | reference, in this pipeline |
| whole_DPO | the whole DPO model as host, no swap | reference, in this pipeline |

## Covariate, measured before any run

`part103/delta103.py mlp` (torch, same measure as Part 102's attention covariate): each stage's MLPs sit about as far
from Instruct's as its attention does. SFT: 0.44 to 0.48% per layer (attention 0.39 to 0.46%). DPO: 0.24 to 0.26%
(attention 0.21 to 0.25%). The mid-training end: 8.9 to 10.4% (attention 7.6 to 8.3%). So weight distance does not
say which component carries a stage's change. Values: `results/part103/mlp-delta-stages.json`.

## Labels and scores

As Part 102: Part 99's label items cut to the read set, Part 92's blind Haiku reader in two passes, scored by
`part103/score103.py` with Part 101's scorer.

## Predictions (Claude, before any run)

| # | Prediction | Confidence |
|---|---|---|
| 103.1 | DPO's attention is as good as Instruct's everywhere: with DPO's attention in all 16 layers, slot_who is at least 15 of 21. | medium |
| 103.2 | RLVR's step lies in the MLPs: DPO's MLPs leave slot_who at least 3 units below DPO's attention does. | low |
| 103.3 | After SFT both components still change in ways that matter: SFT's attention and SFT's MLPs (all layers) each leave slot_who at 14 of 21 or less. | low |
| 103.4 | Part 92's ladder holds in this pipeline: the whole SFT model gives slot_who at most 10 of 21, the whole DPO model at most 16 of 21. | medium |
| 103.5 | No hybrid damage: every swap run keeps slot_who_coherent at 13 of 21 or more. | medium |
| 103.6 | Every run keeps concept_any at 13 or more of its eligible concept units. | medium |

Basis for 103.2: DPO's early attention matched Instruct's in Part 102, while Part 92's whole DPO model sits five units
below Instruct, so RLVR's step that matters is more likely outside attention. Smallest effect worth knowing: 3 units
between two runs on the same units, beyond Part 101's measured judge noise; differences will also be read unit by unit
with `part102/paired102.py`'s exact McNemar test (exploratory).

## What comes next, decided now

- One component carries a stage's step (its run 4 or more units below the other's): block-wise swaps of that
  component from that stage, as Part 100 did for attention.
- Both components carry it about equally: stop splitting by component; the next question is which units depend on
  which, read unit by unit.
- Neither swap moves slot_who by 3 or more while the whole model does: the step needs both together; next is the pair
  swapped jointly by layer block.

Each next step gets its own registered predictions before it runs.

## Prior art

As Parts 101 and 102. Sun et al. 2026 (arXiv:2604.13694) report that post-training writes instruction following more
into sparse MLP neurons than into attention, which is the direction 103.2 predicts for RLVR's step.

## Limits

One model family, one fixed push per unit (computed for Instruct and applied to the SFT and DPO hosts as well), one
judge; 20 or 21 slot units; a component from one stage is tested among another stage's weights.
