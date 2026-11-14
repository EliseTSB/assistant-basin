# Part 210 report (made by part210/report210.py)

## Predictions

| # | Verdict | Numbers |
| --- | --- | --- |
| 210.1 | held | R1 True, R2 True, R3_bad_pushed_calls 0, R3_bad_no_push_calls 0, R3_companion_rows 0 |
| 210.2 | failed | eager_verdicts {"165.1": "fail", "165.2": "pass", "165.3": "pass", "165.4": "pass", "165.5": "not read", "165.6": "fail"} |
| 210.3 | held | ratio 1.8389830508474574 |
| 210.4 | held | at_cap {"n+": 375, "n\u00b1": 405, "k4": 436, "k16": 383, "k64": 359, "k256": 372, "k1024": 350, "full": 379, "iso": 349}, minimum 349, maximum 436 |
| 210.5 | held | ratio 1.8389830508474574, published 2.6986301369863015 |
| 210.6 | held | differ 84, of 200 |
| 210.7 | held | arms_higher ["n+", "n\u00b1", "k4", "k16", "k64", "k256", "k1024", "full", "iso"] |

## Part 165's verdicts, published beside the eager rerun

| Prediction | Bar | Published estimate (95% interval) | Published | Eager estimate (95% interval) | Eager |
| --- | --- | --- | --- | --- | --- |
| 165.1 | n+ at least twice full (>= 2) | 2.69863 | pass | 1.83898 | fail |
| 165.2 | broken Spearman <= -0.8 | -0.85714 (-0.964 to -0.631) | pass | -0.85714 | pass |
| 165.3 | k16 nearer full than n± | to full 0.0534, to n± 0.1055 | pass | to full 0.0599, to n± 0.0911 | pass |
| 165.4 | n+ over n± between 0.67 and 1.5 | 1.01026 | pass | 0.92735 | pass |
| 165.5 | iso under 0.8 times full | 0.83562 | fail | 0.85593 | not read |
| 165.6 | changed Spearman <= -0.6 | 0.20412 (0.204 to 0.204) | fail | 0.20412 | fail |

165.6's interval collapsed to a point on both runs: published 6,525 usable draws of 10,000, eager 6,525 of 10,000 (score.md, '165.6 usable draws', from score165.score_rows on the eager rows).

## Part 165's gates

| Gate | Published | Eager |
| --- | --- | --- |
| gate1_rows | {"passed": true, "expected_rows": 6920, "rows": 6920, "unique_rows": 6920, "duplicates": 0, "nonempty_answers": 6920} | {"passed": true, "expected_rows": 6920, "rows": 6920, "unique_rows": 6920, "duplicates": 0, "nonempty_answers": 6920} |
| gate2_no_push | {"passed": true, "baseline_answers": 8} | {"passed": true, "baseline_answers": 8} |
| gate3_size | {"passed": true, "expected_directions": 864, "directions": 864, "unique_expected_directions": true, "duplicate_directions": 0, "write_norm_failures": 0, "iso_residual_failures": 0, "max_relative_write_error": 3.94895446787479e-07, "max_iso_residual_fraction": 3.959983196616994e-06} | {"passed": true, "expected_directions": 864, "directions": 864, "unique_expected_directions": true, "duplicate_directions": 0, "write_norm_failures": 0, "iso_residual_failures": 0, "max_relative_write_error": 3.94895446787479e-07, "max_iso_residual_fraction": 3.959983196616994e-06} |
| gate4_measure_room | {"passed": true, "n_plus_broken_rate": 0.2565104166666667, "minimum": 0.03} | {"passed": true, "n_plus_broken_rate": 0.2825520833333333, "minimum": 0.03} |

## Broken rates per arm, published beside eager

| Arm | Published broken / 768 | Published rate | Eager broken / 768 | Eager rate | Change |
| --- | --- | --- | --- | --- | --- |
| n+ | 197 | 0.2565 | 217 | 0.2826 | +0.0260 |
| n± | 195 | 0.2539 | 234 | 0.3047 | +0.0508 |
| k4 | 208 | 0.2708 | 244 | 0.3177 | +0.0469 |
| k16 | 114 | 0.1484 | 164 | 0.2135 | +0.0651 |
| k64 | 77 | 0.1003 | 136 | 0.1771 | +0.0768 |
| k256 | 71 | 0.0924 | 111 | 0.1445 | +0.0521 |
| k1024 | 66 | 0.0859 | 82 | 0.1068 | +0.0208 |
| full | 73 | 0.0951 | 118 | 0.1536 | +0.0586 |
| iso | 61 | 0.0794 | 101 | 0.1315 | +0.0521 |

## Lengths (battery rows)

| Run | Arm | Rows | At the 80-token cap | Mean token_count |
| --- | --- | --- | --- | --- |
| published | n+ | 768 | 67 | 56.767 |
| published | n± | 768 | 67 | 57.117 |
| published | k4 | 768 | 57 | 57.714 |
| published | k16 | 768 | 50 | 57.734 |
| published | k64 | 768 | 59 | 58.943 |
| published | k256 | 768 | 78 | 60.708 |
| published | k1024 | 768 | 79 | 61.087 |
| published | full | 768 | 84 | 60.684 |
| published | iso | 768 | 63 | 58.257 |
| rerun | n+ | 768 | 375 | 67.457 |
| rerun | n± | 768 | 405 | 68.543 |
| rerun | k4 | 768 | 436 | 70.232 |
| rerun | k16 | 768 | 383 | 68.753 |
| rerun | k64 | 768 | 359 | 68.507 |
| rerun | k256 | 768 | 372 | 69.697 |
| rerun | k1024 | 768 | 350 | 69.363 |
| rerun | full | 768 | 379 | 70.296 |
| rerun | iso | 768 | 349 | 68.008 |

## Shape replicate

Differ: 84 of 200 (bar 40 to 110).

Battery row's margin at the first differing position, in bf16 steps (tools/margin_recorder.bf16_step):

- 0 (exact tie): 35
- over 0, up to 1: 43
- over 1, up to 2: 5
- over 2: 1
- no recorded margin: 0

Median first differing position: token 20.

Where the first difference sits (token index, 0-based):

- token 0 (first token): 0
- tokens 1 to 9: 25
- tokens 10 to 19: 15
- tokens 20 to 39: 20
- token 40 or later: 24

Label changes over the 200 pairs (score210.floor_labels): broken 16, loop 16, triage 1

## Stability under the shape floor (share of 1,000 redraws keeping the verdict)

| Verdict | independent | shared-change-0.5 | shared-change-1 | upper-marginal-independent |
| --- | --- | --- | --- | --- |
| 165.1 | 0.890 | 0.891 | 0.882 | 1.000 |
| 165.2 | 0.986 | 0.984 | 0.989 | 0.933 |
| 165.3 | 0.958 | 0.959 | 0.962 | 0.902 |
| 165.4 | 1.000 | 1.000 | 1.000 | 1.000 |
| 165.5 | 0.890 | 0.891 | 0.882 | 1.000 |
| 165.6 | floor not informative: changed is constant | | | |
| gate1_rows | 1.000 | 1.000 | 1.000 | 1.000 |
| gate2_no_push | 1.000 | 1.000 | 1.000 | 1.000 |
| gate3_size | 1.000 | 1.000 | 1.000 | 1.000 |
| gate4_measure_room | 1.000 | 1.000 | 1.000 | 1.000 |
