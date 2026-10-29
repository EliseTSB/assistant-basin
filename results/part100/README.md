# Part 100: the slot neurons need Instruct's early attention

Run 2026-10-29 05:55 to 06:09 CT on the A770 (six runs of `part99/swap99.py --layers`, batch 48, Part 99's fixed
pushes), labelled by Part 92's blind Haiku reader in two passes, scored by `part100/score100.py`. Design and predictions
registered before any run (PART100-DESIGN.md, commit 4a98a09).

| Base attention in layers | Slot units, "Who are you?" shows persona | Those answers coherent | Concept units, any answer |
|---|---|---|---|
| none (Part 99 inst_none) | 19 of 21 | 21 of 21 | 18 of 18 |
| all 16 (Part 99 inst_attnB) | 3 of 21 | 13 of 21 | 14 of 17 |
| 0 to 7 | 3 of 21 | 15 of 21 | 14 of 17 |
| 8 to 15 | 8 of 21 | 15 of 21 | 14 of 16 |
| 0 to 3 | 3 of 21 | 16 of 21 | 13 of 17 |
| 4 to 7 | 7 of 21 | 19 of 21 | 17 of 18 |
| 8 to 11 | 7 of 21 | 13 of 21 | 15 of 16 |
| 12 to 15 | 16 of 20 | 18 of 20 | 17 of 18 |

**What it shows.** Swapping the mid-training end's attention into only layers 0 to 3 drops the slot units' persona on
"Who are you?" as far as swapping all 16 layers (3 of 21), while the answers stay coherent (16 of 21) and the concept
units keep theirs (13 of 17). The last four layers barely matter (16 of 20); each middle quarter takes away about half
(7 of 21). The effects do not add up: one early quarter already gives the whole drop. For 16 of the 21 slot units (those
in layers 3 to 12), every swapped layer of 0 to 3 comes before the push enters, so for them the swap can only change
the context the push lands in, not how later layers read it. So the post-training change these self-descriptions need
lies mostly in how the first four attention layers process the chat and the question.

**Predictions.** 100.1 failed: the early half matters more (3 of 21) than the late half (8 of 21). 100.2 failed: the
first quarter alone reproduces the all-layer drop. 100.3 held: every run kept at least 13 of 21 slot answers coherent,
so these drops are not hybrid damage by Part 99's rule. 100.4 held: concept units kept 13 or more in every run.

Limits: one model pair, one fixed push per unit, one judge; 21 slot units, so differences of a few units are noise;
4 to 7 label errors per run survived the second pass and those items are left out (the 12-to-15 run scores 20 units);
blocks of four layers, not single layers; no random-direction check was scored for these runs.
