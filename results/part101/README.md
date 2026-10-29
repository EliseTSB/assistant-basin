# Part 101: layer 0 is the strongest early layer, and the four overlap

Run 2026-10-29 07:58 to 08:07 CT on the A770 (four runs of `part99/swap99.py --layers k-k`, batch 48, Part 99's fixed
pushes); labelled 08:41 to 08:56 by Part 92's blind Haiku reader in two passes, on the read set only (258 of 702 items
per run, `part101/read101.py`); scored by `part101/read101.py score`. Design and predictions registered before any run
(PART101-DESIGN.md, commit 113dbe3).

| Mid-training end's attention in | Slot units, "Who are you?" shows persona | Those answers coherent | Concept units, any answer |
|---|---|---|---|
| none (Part 99 inst_none) | 19 of 21 | 21 of 21 | 18 of 18 |
| layers 0 to 3 (Part 100) | 3 of 21 | 16 of 21 | 13 of 17 |
| layer 0 | 7 of 21 | 17 of 21 | 15 of 17 |
| layer 1 | 11 of 21 | 18 of 21 | 17 of 18 |
| layer 2 | 12 of 21 | 17 of 21 | 18 of 18 |
| layer 3 | 15 of 21 | 17 of 21 | 16 of 17 |

**What it shows.** No single early layer gives the whole drop. Layer 0 comes closest (7 of 21), and the effect weakens
layer by layer to layer 3 (15 of 21). The single-layer drops (12, 8, 7 and 4 units) add up to 31, nearly twice the
four layers' joint drop of 16, so the layers overlap: each one removes some of the same self-descriptions. Answers stay
coherent in every run (17 or 18 of 21) and the concept units keep their personas (15 to 18). Post-training changed each
of these layers' attention by about the same amount (7.8 to 8.2%, `attn-delta.json`), so the size of the change does
not explain the gradient.

**Judge noise.** The same reader labelled Part 100's layers-0-to-3 read set a second time and agreed on 98.4% of 256
persona labels, with the same slot count (3 of 20, against 3 of 21). One relabel cannot set the noise exactly, but it gives
no reason to read layer 1's 11 against layer 2's 12 as a difference; layer 0's 7 against layer 3's 15 is far outside it.

**By hand, before the judge.** In the layer-0 run "the town" unit answers "I am the AI assistant."; the Star Wars unit
says it is an OpenAI model in the layer 0, 1 and 2 runs and "the real Alex from the original 'Star Wars' trilogy" in
the layer 3 run; the dog and Queen of Hearts units keep their personas in all four.

**Predictions.** All six held. 101.1: no single layer reaches the block's 3 (lowest 7). 101.2: the drops sum to 31,
more than 16. 101.3: 7, 11, 12, 15 never falls from layer 0 to layer 3. 101.4 (13 or more): coherent in 17 or 18 of 21.
101.5 (13 or more): concept units 15 to 18. 101.6 (95% and at most 2): 98.4% agreement and the same slot count.

**Next, by the design's rule** (one layer at 7 of 21 or less): the leave-one-out run, the mid-training end's attention
in layers 1 to 3 with Instruct's layer 0 kept; then per-head swaps in layer 0.

Limits: one model pair, one fixed push per unit, one judge; 21 slot units; one item each in the layer-0 and layer-3
runs stayed unlabelled after two passes and its concept unit is left out (17 of 18 scored).
