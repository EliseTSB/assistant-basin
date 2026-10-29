# Part 104: do new early slot neurons need what post-SFT training wrote into attention, or are they just fragile?

Registered 2026-10-29 by Claude (Opus 5.5), before any Part 104 shortlist, answer, push or swap. The commit that adds
this file is the registration.

## Question

Part 103 found a pattern after seeing its table, so it is a hypothesis, not a result. Of the 10 slot units in layers
1 to 3 that show their persona in Instruct, SFT's attention (all 16 layers) loses 7; of the 9 in layers 4 to 12, it
loses none (one-sided Fisher p = 0.0024). DPO's attention is a smaller change (Part 102's covariate: about 0.2 to 0.3%
from Instruct per layer, against SFT's 0.4 to 0.5%), and it loses three units, all in layers 2 and 3.

Two readings fit that table:

- **D (dependence).** The early units need something that post-SFT training put into attention.
- **F (fragility).** The early units are lost under any change to attention of this size, and SFT's swap is only the
  larger change.

Part 104 tests the layer split on new units, chosen before any of their answers is read. It separates D from F with a
swap that moves attention the same distance from Instruct in the opposite direction.

## Prior art (checked 2026-10-29)

- Zheng et al., "Weak-to-Strong Extrapolation Expedites Alignment" (ExPO), ICML 2024, arXiv 2404.16792 (abstract
  read). Alignment training makes small parameter changes, and amplifying the change from the SFT checkpoint past the
  aligned model improves alignment benchmarks for models from 1.8B to 70B. ExPO extrapolates whole models by a tuned
  amount; Part 104 extrapolates attention only, by the full SFT-to-Instruct step, as a control rather than a method.
- Wu et al., "From Language Modeling to Instruction Following", NAACL 2024, arXiv 2310.00492 (abstract read):
  instruction tuning "encourages the self-attention heads to capture more word-word relationships about instruction
  verbs". A search summary says the paper places this in the bottom and middle layers; that detail was not read at
  source.
- Neither looks at single neurons or tests a layer split. Our own: Part 92 (kinds), Parts 99 to 103 (swaps). The split
  under test is Part 103's.

## Units

The pool is Part 92's (`part92/shortlist92.py`: every real-neuron screen item labelled coherent with one of the seven
non-assistant kinds) minus Part 92's 53 picks: 1,276 items, 415 in layers 0 to 3 and 861 in layers 4 to 15.

Claude Sonnet 5.5 (thinking off) picks with Part 92's exact prompt, run once per stratum, with that stratum's pool in a
seeded shuffle (seed 104003 for layers 0 to 3, 104415 for layers 4 to 15). The cap in the prompt is "at most 36" for
layers 0 to 3 and "at most 45" for layers 4 to 15. Ids run p104-00 onward, layers 0 to 3 first; control seed 104500
plus the id's number.

The strata boundary (0 to 3 against 4 to 15) is fixed now from Part 101's attention range. It is not tuned on new data.

## Step 1: battery and kinds (Instruct, chat)

- Part 92's six questions for each pick and its paired random direction, greedy, 120 new tokens (`part92/battery92.py`).
- Part 92's blind Haiku reader (`part92/label92.py`), two passes; the second retries failed calls only.
- Part 92's kind rules, unchanged: **slot** if the persona shows in exactly one or two of the six answers and only in
  "Who are you?" or "Tell me about yourself."; **concept** if in 3 or more; **dropped** if "Who are you?" does not show
  it. The kinds script (`part104/kinds104.py`) must reproduce Part 92's own kinds from Part 92's step-2 labels exactly
  before it is run on Part 104's.

**Gate.** Step 2 runs only if step 1 classifies at least 8 slot units in layers 0 to 3 and at least 6 in layers 4 to 15.
Below that, power is under 0.65 even at Part 103's own effect (`part104/power104.py`). If the gate fails, the kinds are
reported and Part 104 stops.

## Step 2: swaps (Instruct host, chat frame)

Part 99's fixed pushes, built once for the new slot units on unmodified Instruct (`part99/make_pushes99.py`), then
`part99/swap99.py --only slot`. The read set is each slot unit's "Who are you?" answer.

| Run | Attention in all 16 layers |
|---|---|
| inst_none | Instruct's (no swap) |
| attnSFT_all | SFT's |
| attnEXT_all | 2 × Instruct − SFT: the same distance from Instruct as attnSFT_all, in the opposite direction (new `--extrapolate` option, off by default) |
| attnDPO_all | DPO's (descriptive; Part 103's small-change arm) |

Labels: Part 92's reader, two passes. Scorer: `part104/score104.py`, committed before any step-2 label is read.

A unit is **shown** if its "Who are you?" shows its persona in inst_none. A shown unit is **lost** under a run if its
"Who are you?" does not show its persona in that run.

## Predictions

- **104.1 (the split replicates).** attnSFT_all loses a larger fraction of shown layer-0-3 units than of shown
  layer-4-15 units, by at least 30 percentage points, with one-sided Fisher p < 0.05.
- **104.2 (D over F).** attnEXT_all loses at most 2 shown layer-0-3 units, and fewer than attnSFT_all does (one-sided
  exact McNemar p < 0.05 on the shown layer-0-3 units). Read only if attnSFT_all loses at least 5 shown layer-0-3 units
  (with 4, even no loss under attnEXT_all gives p = 0.0625) and 104.3 holds.
- **104.3 (the extrapolated model still works).** attnEXT_all's "Who are you?" answers are coherent (the reader's
  `coherent`) for at least 80% of shown units.
- **104.4 (the small change hits early units only).** attnDPO_all loses at most one shown layer-4-15 unit.

What the outcomes mean:

- 104.1 fails: Part 103's split does not replicate on new units. Nothing further is read as layer-specific.
- 104.1 and 104.2 hold: the early units need the post-SFT change itself (D).
- 104.1 holds and 104.2 fails, with attnEXT_all losing about as many early units as attnSFT_all: fragility (F). Early
  slot units are sensitive to any attention change of this size, and "built by post-SFT training" is not shown.

## Sizes

- Step 1: at most 81 picks × 2 (neuron and control) × 6 questions = 972 generations at 120 tokens, about 4 minutes on
  the A770 at Part 92's rate.
- Step 2: up to about 35 slot units × 2 × 6 questions × 4 runs ≈ 1,680 generations, about 2 minutes per run as in Part
  103. Under 15 GPU-minutes in all. Labels: one Haiku call per read-set item, about 70 per run.
- Smallest effect worth knowing: 50% of early units lost against 10% of later ones. With 14 early and 10 later shown
  units, power is 0.51 at that effect and 0.95 at Part 103's (70% against 5%); with 8 and 6, it is 0.21 and 0.65; the
  false-positive rate is under 0.02 at 25% against 25% (exact, `part104/power104.py`).
- Expected yields (not predictions): Part 92's picks gave slot units at 11 of 21 in layers 0 to 3 and 10 of 32 in
  layers 4 to 15. These picks come from a less specific remainder, so the yields may be lower.

## Limits stated now

One model family, one push per unit, one reader. The extrapolation changes attention only, and only by one step size.

## Amendment 1 (2026-10-29, before any step-1 label or any step-2 answer)

"Shows" in this design is Part 99's definition, used by every part since (`part101/read101.py`, `score_rows`): the
pushed "Who are you?" answer shows the unit's persona and the same run's unpushed answer, judged for that unit, does not.
The design's wording ("its 'Who are you?' shows its persona") left that implicit. A shown unit whose item is still
unlabelled after two passes is left out of that run's counts and reported. 104.3's coherence is the reader's `coherent`
on the pushed answer, as in Parts 99 to 103; the run's unpushed-answer coherence (`inconclusive`) is reported beside it.
No prediction changes. Scorer and its synthetic tests: `part104/score104.py`, `part104/test_score104.py` (7 tests; a
reversed McNemar direction fails one).

## Results (2026-10-29)

104.1 failed (SFT's attention lost 9 of 23 early and 6 of 20 later shown units, p = 0.38); 104.3 failed (the
extrapolated run coherent for 32 of 42); 104.2 not read; 104.4 failed (DPO's attention lost 3 later units). The
extrapolated attention lost as many units as SFT's (15 each, 5 shared). Details and the descriptive reading:
`results/part104/README.md`.
