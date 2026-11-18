# Part 205, Amendment 6: the mid-training-end stage was a different training run

Written 2026-11-18 by Claude (Opus 5.5), before the rerun it describes. Amendments 1 to 5 are in the design file; this
one is a separate file, as in Parts 206 and 209. No prediction, bar, reading rule or scorer changes; one stage's
checkpoint does.

## What was wrong

`config.json` took OLMo 2 1B's mid-training end as `allenai/OLMo-2-0425-1B` at `main` (a1847dff), following the model
card. `main` is from a different training run: on embedding rows 1000 to 4999 it correlates 0.008 with the 3.0T, 26B and
Instruct checkpoints, while the 26B checkpoint, the real mid-training end (`stage2-ingredient3-step23852-tokens51B`,
f83cd641, SFT's parent) and Instruct correlate 1.000 (front-page known issue, 2026-11-17). So both adjacent pairs through
`midtrain-end` compared unrelated models.

## What changes

- `config.json`: stage `midtrain-end` names revision f83cd641d4f00954eb25c781e0ac360debb3239e. (`gen_prompts205.py` and
  the tests load `main` only for its tokenizer, which the two runs share; they are left as they are.)
- The `main` outputs move to `results/part205/midtrain-end-main-a1847dff/` (kept). The stage is rerun exactly as under
  Amendment 2's fallback branch (`run205.py --stage midtrain-end --check-pool --arms controls`, CPU), then `score205.py`,
  `report205.py` and `fig205.py` are rerun unchanged.
- Row that uses the stage: 205.3 (the largest rise in O is mid-training end to SFT, at least 0.2, interval excluding 0).
  205.1 counts stages and passes at all six either way.

## Forecasts, written before the rerun

Neighbours on the same run: mid-training 26B O 0.718, SFT 0.723.

- Mid-training end's O lands between 0.69 and 0.75. (80%)
- 205.3 still fails: no adjacent rise reaches 0.2, the corrected stage included. (97%)
