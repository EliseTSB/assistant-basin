# Part 105: on Part 92's original units too, the opposite-direction attention swap loses the early slot units

Run 2026-10-29 11:40 to 11:43 CT on the A770 (`part99/swap99.py --extrapolate`, Part 99's fixed pushes, batch 48);
labelled by Part 92's blind reader in two passes on Part 101's read set; scored by `part105/score105.py` (committed
before the run, checked on two fixtures). Design and predictions registered before the run (PART105-DESIGN.md, commit
c25ff85).

attnEXT_all sets every attention parameter in all 16 layers to 2 × Instruct − SFT: the same distance from Instruct as
Part 103's attnSFT_all (measured ratio 0.9993), in the opposite direction.

## Results

Part 92's 19 shown slot units (10 in layers 1 to 3, 9 in layers 4 to 12), lost under each run (`score105.json`):

| Run | Early lost (of 10) | Later lost (of 9) | Slot "Who are you?" coherent | Concept units, any answer |
|---|---|---|---|---|
| attnSFT_all (Part 103) | 7: p92-04, 05, 07, 10, 13, 20, 42 | 0 | 17 of 21 | 17 of 18 |
| attnEXT_all (Part 105) | 5: p92-02, 05, 07, 42, 50 | 1: p92-32 | 18 of 21 | 17 of 18 |

Three early units are lost under both swaps (p92-05, 07, 42); four only under SFT's attention, two only under the
extrapolated one. Part 103's DPO attention lost p92-05, 42 and 50, all three also lost here.

## Predictions

- **105.1 failed.** The extrapolated swap lost 5 of the 10 early units (needed at most 2), and the paired comparison with
  attnSFT_all is not significant (4 against 2, one-sided exact McNemar p = 0.34). By the registered reading, 5 or more
  lost is fragility (F).
- **105.2 held.** 18 of 21 slot answers coherent (needed 14 or more).
- **105.3 held.** 17 of 18 concept units show their persona.

## Reading

On the units where Part 103's split was found, moving attention the same distance from Instruct in the opposite
direction also loses half of the early slot units and almost none of the later ones. So the early units are sensitive
to changes in attention of this size in either direction. Their loss under SFT's attention does not show that they
need something SFT built. The early-versus-later difference is real on this list of units (Parts 103 and 105), but did
not appear on Part 104's new units, so it is a property of these units rather than a general rule about layers.

This revises the stage reading of Parts 102 and 103. "SFT's attention brings back most of what the pre-SFT model's
loses" stands as a measurement, but the loss under SFT's attention is better described as fragility to a change of that
size than as dependence on what SFT changed.

## Limits

10 early units, so one unit is 10 points; one extrapolation step size; attention only; one push per unit; one reader;
greedy one-answer outcomes.
