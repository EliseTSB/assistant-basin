# Part 101: which of layers 0 to 3?

Size: 4 runs x 474 greedy generations = 1,896 generations on the A770 (Part 100 measured 2 min 10 s per run, so about
9 GPU-minutes), plus 5 x 258 = 1,290 Haiku labels (the read set of four runs and one relabel control, about $1). 1,068
of the generations are not labelled; they are kept so that each run's batches are exactly Part 100's.

Registered 2026-10-29 before any run, by Claude (Opus 5.5). Follows Part 100, where the mid-training end's attention
in layers 0 to 3 alone dropped OLMo 2 1B's 21 slot units to 3 of 21 on "Who are you?", the same as all 16 layers.

## Runs

Host, donor and pipeline exactly as Part 100's quarter runs, one layer each: attnB_L0, attnB_L1, attnB_L2, attnB_L3
(`part99/swap99.py --layers k-k`; steps in `part101/steps101.txt`). References, not rerun: Part 99's `inst_none` (19 of
21) and Part 100's `attnB_L0-3` (3 of 21).

Where the push enters: the slot units sit in layer 1 (1 unit), 2 (4), 3 (6), 4 (4), 5 (3), 6 (1), 9 (1) and 12 (1).
Attention in layer k runs before the MLP of layer k, so swapping layer k changes the context before the push for units
in layers k and above, and how the push is read for units below k. Layers 0 and 1 come before every unit's push; layer
3 comes after the push for the five units in layers 1 and 2.

## Covariate, measured before any run

`part101/attn_delta101.py` reads every attention tensor of both snapshots (the donor rounded to bfloat16, as swap99.py
loads it). Post-training changed each layer's attention by 7.7% to 8.3% (relative Frobenius norm): layers 0 to 3 by
7.8, 8.2, 8.2 and 8.0%, layers 12 to 15 by 7.9 to 8.3%. The size of the change does not single out any layer, so it
does not explain why the early layers matter. Values: `results/part101/attn-delta.json`.

## Labels and scores

Part 99's pipeline (`part99/label_items99.py`, then Part 92's blind Haiku reader in two passes), cut to the read set by
`part101/read101.py filter`: the items a score reads (slot units: pushed and baseline "Who are you?"; concept units:
pushed and baseline answers to all six questions), 258 of 702 per run. `read101.py check` shows the cut is exact:
scoring Part 100's six runs from only these items reproduces `score100.json`, and cutting one more concept question
breaks all six. Scores use Part 99's definitions, as in Part 100: slot_who, slot_who_coherent, concept_any, the
inconclusive rule, and Alden's rule that a run with slot_who_coherent below 11 of 21 counts its drop as hybrid damage.

Relabel control: the same reader labels Part 100's `attnB_L0-3` read set a second time, which measures how far a count
moves when only the judge's sampling changes.

## Predictions (Claude, before any run)

| # | Prediction | Confidence |
|---|---|---|
| 101.1 | No single layer alone reproduces the block's drop: every single-layer run leaves slot_who at 6 of 21 or more. | low |
| 101.2 | The single layers saturate, as Part 100's blocks did: the four single-layer drops (19 minus slot_who) add up to more than the block's drop of 16. | medium |
| 101.3 | Earlier layers matter more: slot_who does not fall from layer 0 to layer 3 (L0 <= L1 <= L2 <= L3). | low |
| 101.4 | No hybrid damage: every single-layer run keeps slot_who_coherent at 13 of 21 or more. | medium |
| 101.5 | Every single-layer run keeps concept_any at 13 or more of its eligible concept units. | medium |
| 101.6 | Control: relabelling Part 100's attnB_L0-3 read set agrees on at least 95% of persona labels and moves slot_who by at most 2. | medium |

Smallest effect worth knowing: one layer that removes at least half of the block's drop (slot_who at 11 of 21 or
less). The 21 slot units fix n; the relabel control says how many units of difference are judge noise.

## What comes next, decided now

- One or two layers reach slot_who at 7 of 21 or less: leave-one-out runs (the mid-training end's attention in layers
  0 to 3 except that layer), to test whether Instruct's attention in that layer alone is enough; then per-head swaps in
  it.
- Every layer reaches 7 of 21 or less: the early layers act together; next is per-head swaps in the layer with the
  lowest slot_who.
- No layer reaches 11 of 21 or less: the effect needs several early layers at once; next is the six two-layer pairs.

Each next step gets its own registered predictions before it runs.

## Prior art (searched before registering)

- Wu et al. 2024 (NAACL, arXiv:2310.00492): after instruction tuning, self-attention heads learn more word relations
  with instruction verbs, mostly in the lower and middle layers.
- Sun et al. 2026 (arXiv:2604.13694, Weight Patching): patching head and neuron weights between paired base and
  instruct Llama models places instruction following in shallow neuron-level carriers routed by mid-layer attention
  heads, and reports that post-training writes the behavior more into sparse neurons than into attention. They do not
  treat the chat template or role tokens specially.
- Neither swaps single early attention layers between a model and its pre-SFT parent, or scores injected
  self-descriptions.

## Limits

One model pair, one fixed push per unit, one judge; 21 slot units; a single swapped layer can fail for reasons
unrelated to where the self-description is built.

## Results (08:57 CT, after the run)

All six predictions held. Slot units keeping their persona with the mid-training end's attention in one layer: layer 0, 7 of 21; layer 1, 11; layer 2, 12; layer 3, 15 (none: 19; layers 0 to 3: 3). The drops sum to 31 against the block's 16, so the layers overlap. Relabel control: 98.4% agreement, same slot count. Full table and reading: results/part101/README.md.
