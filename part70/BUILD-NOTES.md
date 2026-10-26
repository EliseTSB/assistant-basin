# Part 70 build notes

The runner reuses Part 66's model identity, question lists, row seed function, intervention hooks, padding, stop reason, and its prompt and batching conventions. It constructs the registered target, target-plus-pair, and twenty activity-matched conditions. The scorer delegates family classification and row aggregation to `part66.score66`, then applies the six Part 70 reading rules and writes JSON and Markdown reports.

The matched file's actual shape is `matched.json['29_776']['27_3325']` and `matched.json['29_776']['24_3949']`. Both pools contain 50 `{layer, unit, magnitude}` records. Matched set `k` draws one entry from each pool with one `numpy.default_rng(7000 + k)` stream.

**Design arithmetic ambiguity and choice:** the design table explicitly assigns 240 generations to each of `target` and `target_plus_pair`, plus 24 for each of 20 matched conditions. Those rows total `2*240 + 20*24 = 1,200`, while the design summary says “960 generations in all” and the brief asks a total-960 row-plan test. I implemented and tested the per-condition table exactly; those explicit condition counts imply 1,200. No per-condition sampling count was changed to force the summary arithmetic.

## Verification

The host has neither `pytest` nor `torch` installed (`python3 -m pytest` reports no pytest module; importing Part 66 reports no torch module), so the requested pytest suite and torch hook test could not be run here. I did run both CLI `--help` commands successfully and ran Python compilation checks. The test file skips Part 66 dependent generation/planning checks and the direct torch measurement when torch is unavailable; deterministic matched draws and scoring checks remain runnable with pytest.

No model was loaded and no generation run was performed. Even with torch installed, the run-path test reaches the actual Part 66 zeroing hook used by `run_70`, but it does not execute `model.generate`: the host has no model/tokenizer test fixture, and Part 66's generation path requires a chat-capable tokenizer and Transformers generation behavior. The run loop's resume, answer serialization, and GPU memory path therefore remain unexercised by a live model run.

## Host check before the first run (Claude, 2026-10-26 04:3x CT)

The worker's sandbox had no pytest or torch, so no test had run. On the host: the suite could not import its runner
(pytest puts part70/ first on sys.path, where `part70` is the script, not the package; the test now drops that entry);
two tests asserted 1,200 generations, from the build's arithmetic above, which is wrong (2 x 240 + 20 x 24 = 960, the
design's total; the planner produced 960 all along); and a verdict fixture filled non-Alibaba answers with "Qwen",
which Part 66's patterns count as Alibaba-family, so the 70.6 boundary could not be tested. All three were test
defects; the runner and scorer were right. 6 of 6 pass.
