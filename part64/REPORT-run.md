# Part 64 scoring and analysis run report

## SUMMARY
Built the Part 64 scorer, analysis, rented-GPU wrapper, runbook, and synthetic analysis coverage. Scoring imports the Part 60 Qwen hook, thresholds, tier statistics, and full-rule pass decision. The analysis reports all four arm rates and real-minus-control gaps, predictions 64a-64f, unit-paired bootstrap comparisons against Haiku-1x, and the registered advancement decision. No paid call, network fetch, GPU run, or commit was made.

## FINDINGS
- `part64/score64.py:39-59` normalizes the in-progress `gen64.py` output rows from its writer/sample group files; `score64.py:104-172` scores all successful sentences using `score60.QwenScorer`, Part 60 thresholds, and `analyze60.tier_statistics`. Its pass rule is unchanged: at least 3/4 strong hits and at most 1/4 none hits, each using strict activation-above-threshold comparisons. Failed parse/score rows are retained as non-passes.
- `part64/analyze64.py:56-80` reads Part 60 dev M1 full-rule rates for 64f. `analyze64.py:83-239` enforces the 500-unit paired grid, computes rates/gaps and stratified paired bootstrap intervals, evaluates 64a-64f, and applies the advancement rule. 64e remains directional per the registered limit.
- `part64/run64.sh:74-85,125-160` preflights source artifacts, fetches the pinned snapshot with the Part 61 cache method, checks the corpus through `collect64.verify_corpus`, runs the current `collect64_model.py --data ... --out ... --cache-dir ...` and `collect64.py --fresh ... --out ... --corpus ...` CLIs, reuses only a matching verified collection, and scores discovered generation group outputs when present. It returns a tarball and prints `RETURN_READY` or `RUN_FAILED`.
- `part64/RUNBOOK-64.md` documents the preflight, Vast collection, `gen64.py` probe/run/resume/status, Vast scoring, local analysis, and recovery steps.
- Covering tests: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part64.test_analyze64 -v` passed 3 tests, including a planted 64b effect and planted null. `bash -n part64/run64.sh` passed. `score64.py --help` and a synthetic gen64-row normalization check passed.
- Parent-control mailbox was checked at checkpoints and before reporting; it remained empty, so there were no plan changes or dispositions.

## BLOCKERS
No implementation blocker. End-to-end model scoring, corpus traversal, and the rented-GPU runner were not executed because the brief prohibits GPU/network operations in this worker run. Production still requires the registered corpus, Part 58 scoring artifacts/thresholds, Part 60 dev analysis for 64f, and the normal Vast staging steps.
