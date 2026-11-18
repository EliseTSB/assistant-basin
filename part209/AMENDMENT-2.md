# Part 209, Amendment 2: the mid-training-end stage was a different training run

Written 2026-11-18 01:11 CT by Claude (Opus 5.5), before the rerun it describes. No prediction, bar, reading rule or
scorer changes; one stage's checkpoint does.

## What was wrong

The design (Models) and `config.json` took OLMo 2 1B's mid-training end as `allenai/OLMo-2-0425-1B` at `main`
(a1847dff), following the model card. `main` is from a different training run: on embedding rows 1000 to 4999 it
correlates 0.008 with the 3.0T, 26B and Instruct checkpoints, while the 26B checkpoint, the real mid-training end
(`stage2-ingredient3-step23852-tokens51B`, f83cd641, SFT's parent) and Instruct correlate 1.000 (checked 2026-11-17;
known-issue note on the front page). Part 15 had found this and Parts 85, 95, 99 and 133 used f83cd641. So every
adjacent-stage comparison through `midtrain-end` compared two unrelated models.

## What changes

- `config.json`: stage `midtrain-end` now names revision f83cd641d4f00954eb25c781e0ac360debb3239e. The stage name stays,
  so the scorer and report read it as before.
- The `main` outputs move to `results/part209/midtrain-end-main-a1847dff/` (kept, not deleted) and the stage is rerun
  exactly as before (CPU, float32, eager; R1 and R3 only, as for every non-chat OLMo stage).
- Then `score209.py` and `report209.py` are rerun unchanged, and the README's verdict rows and stage table are updated
  with a dated note. Rows that use the stage: 209.8 (no adjacent pair differs by 0.15 or more) and 209.15 (the largest
  adjacent change is mid-training end to SFT). Every other number is recomputed by the same scripts and is expected not
  to move, since no other stage is rerun.

## Forecasts, written before the rerun

ROOM-W s at the neighbours: 26B 0.721, SFT 0.683. f83cd641 is 25B tokens further along the same run as 26B.

- Mid-training end's ROOM-W s lands between 0.69 and 0.74. (70%)
- 209.8 still holds: every adjacent change stays under 0.15. (90%)
- 209.15 flips to held: with the stage on the right run, mid-training end to SFT becomes the largest adjacent change.
  (55%; the other contender is 3.0T to 26B at 0.010, so this mostly needs end-to-SFT to exceed whatever 26B-to-end is.)
