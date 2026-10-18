# Part 1g: Muse Spark 1.3 Contributor in place of Jev on 1e's TEST scoring

The 4,480 recorded System One requests from Part 1f's record step (4,467 unique) answered by
`muse-spark-1.3-contributor` on OpenCode Go's Responses door, reasoning effort minimal, the model stating P(true) as
JSON (no logprobs on that door). Replayed through `climb_test.py` with the frozen c23 pipeline; compared by
`compare_1f.py` (branch part1f). 4,467 of 4,467 answered, 532 s at 16 concurrent (8.4 requests a second).

| | Jev (1e) | Muse Spark |
|---|---|---|
| mean per-unit Spearman, 150 TEST units | 0.3919 | 0.4352 |
| per-unit Spearman correlation with Jev | | r = 0.805 |
| median AUROC strongest-vs-random, real neurons / random directions | 0.840 / 0.755 | 0.875 / 0.775 |
| SD of probabilities over answered requests | 0.263 | 0.353 |

Files: `muse-answers.jsonl` (hash, answers, wall_s, usage), `muse-test-scores.json`, `muse-test-rows.json`.
The Jev column's rows are reconstructed from 1e's request cache; 30 requests for a description with no TEST row are
skipped (compare_1f fix, 2026-10-18).
