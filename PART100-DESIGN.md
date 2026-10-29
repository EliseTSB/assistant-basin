# Part 100: which layers' attention turns a pushed slot neuron into a self-description?

Registered 2026-10-29 before any run, by Claude (Opus 5.5). Follows Part 99, which found that OLMo 2 1B's 21 "slot"
neurons show their persona on "Who are you?" in 19 of 21 cases in Instruct and in 3 of 21 when all of Instruct's
attention is replaced with the mid-training end's (`inst_attnB`), while those answers mostly stay coherent (13 of 21).
Part 100 localizes that: the same swap, one block of layers at a time.

## Runs

Host OLMo-2-0425-1B-Instruct (48d788e); donor the mid-training end, SFT's parent (OLMo-2-0425-1B at f83cd64); swap
`attn` (every parameter under `self_attn`) only in the listed layers; everything else exactly as Part 99's `inst_attnB`:
Part 92's chat frame and six questions, the fixed pushes from `results/part99-pushes.pt` (Amendment 1), the 39 slot and
concept units with their paired random directions, greedy, batch 48, on the A770. `swap99.py` gains `--layers a-b`
(default: all layers, so Part 99's runs are unchanged; a test checks only the listed layers are copied).

| Tag | Base attention in layers |
|---|---|
| attnB_L0-7 | 0 to 7 |
| attnB_L8-15 | 8 to 15 |
| attnB_L0-3, attnB_L4-7, attnB_L8-11, attnB_L12-15 | one quarter each |

References, not rerun: Part 99's `inst_none` (19 of 21) and `inst_attnB` (all 16 layers, 3 of 21), same pushes.
Where the slot units sit: layers 1 to 12, 17 of 21 in layers 2 to 5. The push enters at each unit's own layer, so
attention below that layer can change the context but not how later layers read the push.

## Labels and scores

Part 99's pipeline unchanged: `part99/label_items99.py`, then Part 92's blind Haiku reader (`part92/label92.py`, two
passes, the second retrying errors), then `part100/score100.py`, which applies Part 99's definitions to these runs:
- slot_who: slot units whose pushed "Who are you?" answer shows the persona while that run's unpushed answer does not;
- slot_who_coherent: slot units whose pushed "Who are you?" answer the reader calls coherent;
- concept_any: concept units that show their persona in any of the six answers, relative to the unpushed answers;
- a run whose unpushed answers are incoherent on more than 3 of 6 questions is inconclusive;
- by Alden's Part 99 rule, a run with slot_who_coherent below 11 of 21 counts its drop as hybrid damage, not
  localization.

## Predictions (Claude, before any run)

| # | Prediction | Confidence |
|---|---|---|
| 100.1 | The late half carries it: slot_who is at most 8 of 21 with Base attention in layers 8 to 15, and at least 14 of 21 with Base attention in layers 0 to 7. | low |
| 100.2 | No single quarter reproduces the all-layer drop: every quarter run leaves slot_who at 6 of 21 or more. | low |
| 100.3 | Every block run keeps slot_who_coherent at 13 of 21 or more (no hybrid damage). | medium |
| 100.4 | Every block run keeps concept_any at 13 or more of its eligible concept units. | medium |

## Prior art (searched before registering)

Closest found: base and instruction-tuned Gemma-2-9B give similar activations at matched depth on web text but diverge
in later layers on instruction-style inputs (arXiv:2504.08775); base and instruct weights differ little overall
(Shadow-FT, arXiv:2505.12716); whole blocks moved between models, with success depending on position (arXiv:2601.13580,
cited in Part 99). None swaps attention block by block between a model and its pre-SFT parent and scores injected
self-descriptions.

## Limits

One model pair, one fixed push per unit, one judge; 21 slot units, so a difference of a few units is noise; layer
blocks of four and eight, not single layers; a hybrid can fail for reasons unrelated to where the self-description is
built.

## Results (06:44 CT, after the run)

100.1 failed, 100.2 failed, 100.3 held, 100.4 held. Base attention in layers 0 to 3 alone drops the slot units to 3 of 21 (all 16 layers: 3 of 21), with 16 of 21 answers coherent; layers 12 to 15 leave 16 of 20. Full table and reading: results/part100/README.md.
