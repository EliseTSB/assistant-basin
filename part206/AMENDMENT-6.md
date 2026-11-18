# Part 206, Amendment 6: the mid-training-end stage was a different training run

Written 2026-11-18 by Claude (Opus 5.5), before the rerun it describes. No prediction, bar, reading rule or scorer
changes; one stage's checkpoint does.

## What was wrong

`config206.json` took OLMo 2 1B's mid-training end as `allenai/OLMo-2-0425-1B` at `main` (a1847dff), following the
model card. `main` is from a different training run: on embedding rows 1000 to 4999 it correlates 0.008 with the 3.0T,
26B and Instruct checkpoints, while the 26B checkpoint, the real mid-training end
(`stage2-ingredient3-step23852-tokens51B`, f83cd641, SFT's parent) and Instruct correlate 1.000 (front-page known issue,
2026-11-17; Part 209's Amendment 2 reran the same stage first). So both adjacent-stage comparisons through
`midtrain-end` compared unrelated models.

## What changes

- `config206.json`: the `midtrain-end` model entry names revision f83cd641d4f00954eb25c781e0ac360debb3239e.
- The `main` outputs move to `results/part206/midtrain-end-main-a1847dff/` (kept). The stage is rerun exactly as before
  (`record206.py` then `--pass gt`, CPU, batch 8; then `analyze206.py --stage midtrain-end`), and `score206.py` and
  `fig206.py` are rerun unchanged.
- Rows that use the stage: 206.7 (the largest rise in `rho_peak(A-tmpl, M)` is mid-training end to SFT) and 206.12 (the
  largest rise in the slot-concept arming gap is SFT to DPO). 206.2 counts stages and fails at all six either way. The
  README's descriptive sentence that the mid-training-end gap is the largest of the six is also from the wrong run.

## Forecasts, written before the rerun

Neighbours on the same run: mid-training 26B `rho_peak` 0.067 (gap -6.35); SFT 0.151 (gap 3.74).

- Mid-training end's `rho_peak(A-tmpl, M)` lands between 0.05 and 0.10. (75%)
- 206.7 still holds: mid-training end to SFT stays the largest rise. (85%)
- The mid-training-end arming gap lands between -15 and 10, so the old +32.04 jump does not reappear. (75%)
- 206.12 still fails: SFT to DPO is not the largest rise. (85%)
