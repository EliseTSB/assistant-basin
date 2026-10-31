# Part 138: can OLMo 3 7B run reliably on the Arc?

On the Arc A770, OLMo 3 7B Instruct in bf16 generated responses to 16 questions at a 64-token limit. Batch 8 had no spill in all six recorded runs.

The first ladder's `largest_safe` variable reached 16 before batch 32 spilled. Across all runs, 8 was the largest batch with no spill every time. The later ladder means the second run, in `repeats.json`. It spilled at batch 16 and skipped its repeats. The third run in `repeats8.json` supplied the batch-8 run and three repeats after the amendments. No amendment covers this third run.

## Predictions

| # | Prediction (verbatim, with confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 138.1 | The model loads with 10 GB of Windows RAM free without tripping the guard. (medium) | Held. The job runner's progress log records starts with 13.4, 16.5, and 17.7 GB available at 00:45:17, 02:07:26, and 03:41:49 CT. No RAM guard stop was recorded for a Part 138 step. These launch readings are from the progress log, not the result JSON files. | held |
| 138.2 | The largest batch with no spill is at least 8. (medium) | Batch 8 had no spill on all six runs. | held |
| 138.3 | At that batch the rate is at least 1 answer/s at 64 tokens. (medium) | Batch 8 generated 0.908 answers/s in the first ladder and 0.342–0.377 in later runs. | failed |
| 138.4 | Three repeats at that batch all finish without a spill. (medium) | The three batch-8 repeats finished, each without a spill. | held |

The design has no verdict rule. Each verdict here compares the observed result with the prediction: held means the prediction was met; failed means it was not.

## What happened at each batch?

The table gives every row in the three result files. “Spill” follows the recorded spill flag. Windows RAM is the recorded available RAM in MB.

| File | Run | Batch | Answers/s | Spill | Windows RAM available (MB) |
| --- | --- | ---: | ---: | --- | ---: |
| `ladder.json` | ladder | 2 | 0.287 | no | 14,446 |
| `ladder.json` | ladder | 4 | 0.494 | no | 14,282 |
| `ladder.json` | ladder | 8 | 0.908 | no | 13,940 |
| `ladder.json` | ladder | 16 | 0.728 | no | 12,632 |
| `ladder.json` | ladder | 32 | 0.423 | yes | 5,188 |
| `repeats.json` | rerun ladder | 8 | 0.350 | no | 15,298 |
| `repeats.json` | rerun ladder | 16 | 0.552 | yes | 12,479 |
| `repeats8.json` | batch-8 run | 8 | 0.342 | no | 12,791 |
| `repeats8.json` | repeat 1 | 8 | 0.377 | no | 13,737 |
| `repeats8.json` | repeat 2 | 8 | 0.377 | no | 13,670 |
| `repeats8.json` | repeat 3 | 8 | 0.373 | no | 15,711 |

The job-runner RAM guard stops below 2 GB of available Windows RAM. The spill rule in `part138/ladder138.py` is more than 1,000,000,000 bytes of GPU shared-memory growth over its starting value. All 11 rows passed the Paris gate. The six batch-8 answer lists match exactly. Seven of the 16 generated answers in each batch-8 run end mid-sentence at the 64-token limit.

The first ladder's `largest_safe` variable reached 16, because that row did not spill. The later ladder spilled at 16. So 8 is the largest batch with no spill on every run. The six batch-8 runs were one in `ladder.json`, one in `repeats.json`, and the run plus three repeats in `repeats8.json`.

## Method

The design specifies allenai/Olmo-3-7B-Instruct, a local copy on E:, bf16, direct loading to the A770 XPU, and 64 new tokens for 16 fixed questions. The first ladder tried batches 2, 4, 8, 16, and 32. The second tried 8 and 16. The third tried batch 8 and ran three repeats. Rates and Windows RAM values above come from the rows in the named JSON files.

## Limits

This is one model revision, one 16-question set, one 64-token cap, and one machine state. A game running alongside can change free memory. The records do not explain why the later batch-8 rates were lower, or establish rates and spill behavior for other prompts, token limits, model revisions, or machine states.

The design amendment says an unreadable GPU counter must stop the ladder. Lines 78–80 of `part138/ladder138.py` instead treat a missing start or current counter as no spill. No result row records a missing shared-memory start or maximum value.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; seven findings, folded.
