VERDICT: FIX-FIRST

## SUMMARY

The current checkout appears able to reach the first GPU phase on the documented Vast 3090 image: all 18 files in the runbook's explicit input list are present, the Part 26 source record has 1,000 selected 256-word documents, and both exact package pins exist on PyPI. The Qwen and OLMo dose maxima and registered units match the saved records. I found one mismatch with Amendment 1's required result presentation, plus two conditional packaging/failure-path issues below. I did not rent a GPU or execute the runner.

## FINDINGS (ranked)

1. **P1 — Add Amendment 1's null caveat and comparative decision to the 59B result.** `part59/analyze59b.py:167-194,211-212`; requirement: `PART59-DESIGN.md:47-49`. The JSON records the literal 59f/59g threshold predictions and the target/control counts, but neither the saved output nor printed line says that the registered null simulation can pass or fail each prediction. Nor does it mark the amendment's informative comparison (target symmetric count must exceed the largest control count). In a correlated-noise run, 59f can be met by controls; without the caveat/comparison alongside the prediction flags, a reader can overstate a threshold hold as target-specific evidence. Smallest fix: add a per-prediction null-simulation note and an explicit `target_exceeds_max_control` field (or equivalent displayed result), preserving the registered 59f/59g claims.

2. **P2 — Exclude the local blind key directory from the input tarball.** `part59/RUNBOOK-59.md:21-26`; keys are written at `part59/blind_setting.py:298-302`; `part59/.gitignore:1-2` only affects Git. The runbook tars the entire `part59/` directory. If a reused checkout already has `part59/review/setting-key-*.json` from a prior batch, ordinary `tar` includes those condition mappings and uploads them to the rented machine, despite the runbook saying the reader is local. The current checkout has no `review/` directory, so this review did not find a key in the package inputs. Smallest fix: use an explicit runtime-file list or exclude `part59/review` in the tar command.

3. **P2 — Stop paid setup work after preflight or first prefetch failure.** `part59/run59.sh:53-63,77-80`. If a required file is missing, `run_status` is set but pip installation, package checks, and GPU inventory still run. Also, if Qwen prefetch fails, the second `run_checked` still downloads OLMo because the enclosing `if` is evaluated only once. GPU phases are correctly skipped and `RETURN_READY` is gated, so this does not create a false-complete result, but it can spend rented time without reaching a GPU phase. Smallest fix: gate setup on preflight success and re-check `run_status` between model prefetches.

## RUN CHECKS

- The 18 explicit non-`part59/` files in the runbook tar/preflight list all exist in this checkout. The saved Part 26 source JSON has 1,000 documents, each 256 whitespace-separated words.
- The checked records identify Qwen3.5-4B L22:2742 with Pile maximum 19.125 and controls 152/376/692; OLMo-2-0425-1B-Instruct L10:6978 has width 8192 and Pile maximum 6.5. `run59a.py` reuses Part 26's condition seed, generation helper, and down-projection pre-hook; controls are the same sorted Part 26 draw (first three of the 20-unit draw).
- `analyze59b.py` otherwise matches Amendment 1's pooled linear P99 thresholds, strict `>` comparisons, nonzero equal-sign requirement, leave-one-control-out thresholds, strict later-layer selection, and minimum-of-two-absolute-changes ranking. The 59A blind output retains enough counts to score 59a–59e; 59f/59g counts are written; 59h supplies the top token projections and windows and correctly leaves hand reading to the reviewer.
- The requested releases are published: [transformers 5.20.0](https://pypi.org/pypi/transformers/5.20.0/json) and [flash-linear-attention 0.5.2](https://pypi.org/pypi/flash-linear-attention/0.5.2/json). The runbook extraction path and `run59.sh` working directory are consistent (`/root/part59` containing the nested `part59/` package).

## WHAT I RAN / COULD NOT CHECK

I read the current design/amendment, Part 59 scripts, tests, runbook, Part 26 helpers/records, blind-origin reader, and Part 54 run script/README. Read-only Python checks verified the explicit input paths, source-record dimensions, unit maxima, and OLMo width. I looked up the exact package releases on PyPI. I did not run unit tests, create the tarball, invoke `run59.sh`, load either model, or rent/test the live Vast instance; therefore actual package installation, snapshot downloads, CUDA memory, runtime, and full end-to-end completion remain unverified.

## BLOCKERS

No missing required input or nonexistent pinned package version was found. Live GPU/network/runtime behavior remains unverified; the result-presentation mismatch above should be fixed before treating 59f/59g as an interpretable finding.
