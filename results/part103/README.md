# Part 103: after SFT, both attention and MLPs still change in ways the slot neurons need, and the units that need it sit early

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
- After SFT, both components still change in ways the self-descriptions need. SFT's MLPs lose 9 units and gain none
  (p = 0.004); SFT's attention loses 7 and gains 1 (p = 0.07). Swapped into layers 0 to 3 only (Part 102) it leaves the same
  count, with 2 units differing each way, so the later layers' post-SFT attention change adds nothing measurable. Whole SFT loses 11 (p = 0.001), 9 of them also lost under one of the two swaps.
- The two components affect the same units: 6 of the 7 units SFT's attention loses are also lost under SFT's MLPs.
- After DPO (RLVR's step), the whole DPO model loses 6 units (p = 0.031), split evenly: DPO's attention and DPO's MLPs
  each lose the same 3, neither significant alone.
- Part 92's ladder reproduces in this pipeline: whole SFT 8 of 21, exactly as Part 92; whole DPO 12 of 20 (Part 92:
  14 of 20).

**The units that need the post-SFT change sit early (found after the fact, not a registered prediction).** Of the 10
slot units in layers 1 to 3 that show their persona in Instruct, SFT's attention loses 7; of the 9 in layers 4 to 12,
it loses none (one-sided Fisher p = 0.002). SFT's MLPs lose 7 of the 10 and 2 of the 9 (p = 0.05). The later units'
self-descriptions already worked at SFT; the earliest units' needed the changes made after it, which fits Part 101:
the attention that carries them is in layers 0 to 3, just before or at those units.

**Predictions.** 103.1 held: DPO's attention leaves 17 of 21. 103.2 failed: DPO's MLPs (16) are one unit below DPO's
attention (17), not three; RLVR's step is split, not in the MLPs. 103.3 held: SFT's attention 13 and MLPs 10, both 14
or less. 103.4 held: whole SFT 8, whole DPO 12. 103.5 held: 16 to 21 coherent in the swap runs. 103.6 held: concept
units 15 to 17.

**Next, by the design's rule** (neither component is 4 or more units below the other at either stage): stop splitting
by component and read which units depend on which, as above. The early-unit pattern is now a hypothesis to test on
slot units outside these 21, with its prediction registered first.

Limits: one model family, one fixed push per unit (built on Instruct and applied to the SFT and DPO hosts as well), one
judge; 20 or 21 slot units; the layer split was chosen after seeing the table; one or two items per run unlabelled
after two passes.
