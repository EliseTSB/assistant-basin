# Part 103: after SFT, the MLPs and (suggestively) the early attention still change in ways the slot neurons need

Run 2026-10-29 09:24 to 09:37 CT on the A770 (six runs of `part99/swap99.py`, batch 48, Part 99's fixed pushes);
labelled by Part 92's blind Haiku reader in two passes on the read set; scored by `part103/score103.py`. Design and
predictions registered before any run (PART103-DESIGN.md, commit 8b5078e).

| In Instruct, all 16 layers of | Slot units, "Who are you?" shows persona | Those answers coherent | Concept units, any answer |
|---|---|---|---|
| nothing swapped (Part 99 inst_none) | 19 of 21 | 21 of 21 | 18 of 18 |
| DPO's attention | 17 of 21 | 21 of 21 | 16 of 17 |
| DPO's MLPs | 16 of 21 | 19 of 21 | 16 of 17 |
| SFT's attention | 13 of 21 | 17 of 21 | 17 of 18 |
| SFT's MLPs | 10 of 21 | 16 of 21 | 15 of 16 |
| whole model, no swap: DPO | 12 of 20 | 19 of 20 | 16 of 17 |
| whole model, no swap: SFT | 8 of 21 | 13 of 21 | 15 of 18 |

**What it shows.** Unit by unit against Instruct (exact McNemar, exploratory; `by-unit.json`):
- After SFT, both components still change in ways the self-descriptions need: the MLPs clearly, attention
  suggestively. SFT's MLPs lose 9 units and gain none (p = 0.0039); SFT's attention loses 7 and gains 1 (p = 0.070).
  Swapped into layers 0 to 3 only (Part 102), SFT's attention leaves the same count, with 2 units differing each way,
  so the later layers' post-SFT attention change adds nothing measurable. The whole SFT model loses 11 (p = 0.00098),
  9 of them also lost under one of the two swaps.
- The two components affect mostly the same units: 6 of the 7 units SFT's attention loses are also lost under SFT's
  MLPs, which lose 3 more (p92-25, p92-32, p92-37).
- After DPO (RLVR's step), the whole DPO model loses 6 units (p = 0.031). DPO's attention and DPO's MLPs each lose the
  same three (p92-05, p92-42, p92-50), neither significant alone; the whole model also loses three that neither swap
  loses (p92-04, p92-10, p92-47).
- With Part 99's fixed pushes, the whole SFT model gives 8 of 21, the same as Part 92's per-checkpoint pushes; the whole
  DPO model gives 12 of 20, against Part 92's 14 of 20.

**Exploratory, split chosen after seeing the table: the units the swaps lose sit early.** Of the 10 slot units in layers
1 to 3 that show their persona in Instruct, SFT's attention loses 7; of the 9 in layers 4 to 12, none (one-sided Fisher
p = 0.0024). SFT's MLPs lose 7 of the 10 and 2 of the 9 (one-sided Fisher p = 0.051). The whole SFT model loses 7 of the
10 and 4 of the 9 (p = 0.25), so the later units also depend on post-SFT changes once both components are SFT's. The
attention result fits Part 101: the attention these self-descriptions need is in layers 0 to 3, just before or at the
early units.

**Predictions.** 103.1 held (15 or more): DPO's attention leaves 17 of 21. 103.2 failed: DPO's MLPs (16) are one unit
below DPO's attention (17), not three; RLVR's step is shared, not in the MLPs. 103.3 held (14 or fewer each): SFT's
attention 13, MLPs 10. 103.4 held (10 or fewer, 16 or fewer): whole SFT 8, whole DPO 12. 103.5 held (13 or more
coherent): 16 to 21 in the swap runs. 103.6 held (13 or more): concept units 15 to 17.

**Next, by the design's rule** (neither component is 4 or more units below the other at either stage): stop splitting
by component and read which units depend on which, as above. The early-unit pattern is a hypothesis to test on slot
units outside these 21, with its prediction registered first.

Limits: one model family, one fixed push per unit (built on Instruct and applied to the SFT and DPO hosts as well), one
judge; 20 or 21 slot units; zero to two items per run unlabelled after two passes. Corrected 2026-10-29 after a Haiku
cold check: p-values to two figures, the attention evidence marked suggestive, the whole-SFT layer split added, and
the DPO units named.
