# Part 201 score report

## SUMMARY
Built the Part 201 scorer, synthetic tests, shape-floor simulator, and Part 200 worked example. The Part 201 run artifacts are not present, so no experimental Part 201 result is claimed.

## FINDINGS / WHAT SHIPPED
`score201.py` implements G1–G5 where recorded, deterministic-plan row checks, pair comparisons, Wilson label-flip intervals, bf16 margin shares and calls, tie and runner-up analysis, all eleven registered predictions, timing-probe cost estimates, and the ordered decision. The exploratory A-to-Part-200 comparison uses decoded-text equality because Part 200 stores no token IDs.

`shape_floor.py` reports independent joint-transition, cluster-0.5, cluster-1, and upper-rate sensitivity scenarios, including transition counts and denominators, fallback row counts, and per-cell simulated means and SDs. `floor200.py` reproduces the four-scenario example in `results/part201/floor200.md` from 3,200 battery rows and 200 question-0 replicate pairs; it states that one-question rates are applied to sixteen.

Verification: `/home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q part201/test_score201.py tools/test_shape_floor.py` — **10 passed in 3.37s**. The scorer's temporary-folder `--self-check` passed, and Python syntax compilation passed. The full scoring command was not run: `rows.jsonl`, `kernel-f32.json`, `load.json`, and `probe.json` are absent.

### Amendment 1

- 201.6 scores direction symmetry only for labels with at least 8 total flips; otherwise it lists the flip count, and when none qualify it reports the length clause only.
- 201.7 compares every K token position on both batches, with no margin cutoff.
- 201.11 reads `probe.json`, prices each flagged step using the nearest larger row-count and prefix-length probe medians, scales prefixes above 96, and reports the 1-, 2-, and 4-step estimates.
- First-difference records include whether the other emitted token is the first pass's runner-up; reports list non-runner-up rows and pair denominators beside differing-row counts.
- The ordered decision prints the first applicable rule and its triggering readings.
- Shape-floor redraws preserve joint label transitions and report independent, cluster-0.5, cluster-1, and upper-rate sensitivities. `floor200.md` uses the same named scenarios and describes them as conditional sensitivities, not guarantees.

The top-level control message `seq 1` changed the plan; its Builder B requirements were applied and disposed as `considered`. No parent messages were declined or deferred.

## BLOCKERS
No approval or decision is needed. Experimental Part 201 scoring awaits the runner's row, kernel, load, and probe artifacts.
