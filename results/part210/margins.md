# Part 210 size margins from published Part 165

Computed by `part210/margins210.py` from `results/part165/score.json and results/part165/answers.jsonl`. Part 165's own rules (broken, loop, changed, Spearman) are imported from `part165/score165.py`.

## Registered verdicts and gates: published estimate, bar, distance

| Reading | Published estimate | Bar | Distance (positive = on the passing side) | Read by | Verdict |
| --- | ---: | --- | --- | --- | --- |
| Gate 1 rows | 6920 rows, 6920 unique, 6920 nonempty | 6920 unique nonempty rows | 0 — rows of headroom; one missing, extra or empty row fails | count | passed |
| Gate 2 no push | 8 no-push answers, 0 broken | 8 no-push answers, none primary-broken | 8 — answers of headroom; one broken no-push answer fails | count | passed |
| Gate 3 size | 864 directions, max write error 3.95e-07, max iso residual 3.96e-06 | 864 unique directions, every write within 1%, iso residual under 1% | 0 — directions of headroom (a count); the write and iso margins are in Method | count | passed |
| Gate 4 measure room | 0.25651 | >= 0.03 | 0.22651 — n+ broken-rate points above the bar (768 answers per arm; 1 point = 7.68 answers) | point estimate | passed |
| 165.1 n+ over full | 2.69863 | >= 2 | 0.69863 — ratio units above the bar; in broken-rate points it is n+ rate minus 2 x full rate = 0.0664 (51.0 n+ answers of headroom) | point estimate (score165.ratio_* reads estimate_ratio; the 95% paired slot interval [1.8888888888888888, 3.859649122807017] is printed and decides nothing) | pass |
| 165.2 Spearman | -0.857143 | <= -0.8 | 0.0571429 — Spearman units above the bar (positive = on the passing side) | point estimate (score165.spearman_at_most reads the estimate; the 95% interval [-0.9642857142857143, -0.6306562238868912] is printed and decides nothing) | pass |
| 165.6 Spearman | 0.204124 | <= -0.6 | -0.804124 — Spearman units above the bar (positive = on the passing side) | point estimate (score165.spearman_at_most reads the estimate; the 95% interval [0.20412414523193148, 0.20412414523193148] is printed and decides nothing) | fail |
| 165.3 k16 nearer full | to full 0.0534, to n± 0.1055 | distance to full < distance to n± | 0.0520833 — broken-rate points of margin; 1 point = 7.68 answers; a negative margin fails | point estimate (two distances; no interval is defined for a comparison of distances) | pass |
| 165.4 n+ over n± | 1.01026 | between 0.67 and 1.5 | 0.340256 — ratio units to the nearer boundary (positive = inside) | point estimate (score165.ratio_* reads estimate_ratio; the 95% paired slot interval [0.7811001779545692, 1.2994020433708853] is printed and decides nothing) | pass |
| 165.5 iso over full | 0.835616 | < 0.8 | -0.0356164 — ratio units below the bar (positive = on the passing side) | point estimate (score165.ratio_* reads estimate_ratio; the 95% paired slot interval [0.5466666666666666, 1.2712249614791982] is printed and decides nothing) | fail |

## Arm rates recomputed here (primary broken, loop-only broken, changed)

| Arm | Primary broken | Loop-only broken | Changed |
| --- | ---: | ---: | ---: |
| n+ | 0.2565 | 0.0807 | 1.0000 |
| n± | 0.2539 | 0.1055 | 1.0000 |
| k4 | 0.2708 | 0.1302 | 1.0000 |
| k16 | 0.1484 | 0.0964 | 0.9987 |
| k64 | 0.1003 | 0.0755 | 1.0000 |
| k256 | 0.0924 | 0.0703 | 1.0000 |
| k1024 | 0.0859 | 0.0742 | 1.0000 |
| full | 0.0951 | 0.0755 | 1.0000 |
| iso | 0.0794 | 0.0638 | 1.0000 |

The changed rate is the share of answers whose first 12 whitespace-split words differ from that phrasing's no-push answer.

## Spearman: recomputed here against the published value

| Ladder | Recomputed | Published | Agree |
| --- | ---: | ---: | --- |
| broken | -0.8571 | -0.8571 | True |
| changed | 0.2041 | 0.2041 | True |

## Battery answer lengths and broken counts per arm

| Arm | Answers | At the 80-token cap | Mean token_count | Primary broken | Loop-only broken | Any trigram loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n+ | 768 | 67 | 56.7669 | 197 | 62 | 188 |
| n± | 768 | 67 | 57.1172 | 195 | 81 | 186 |
| k4 | 768 | 57 | 57.7135 | 208 | 100 | 208 |
| k16 | 768 | 50 | 57.7344 | 114 | 74 | 113 |
| k64 | 768 | 59 | 58.9427 | 77 | 58 | 76 |
| k256 | 768 | 78 | 60.7083 | 71 | 54 | 71 |
| k1024 | 768 | 79 | 61.0872 | 66 | 57 | 66 |
| full | 768 | 84 | 60.6836 | 73 | 58 | 73 |
| iso | 768 | 63 | 58.2565 | 61 | 49 | 61 |
| no push | 8 | 0 | 58.0000 | 0 | — | — |

## Where the broken answers sit: at the cap, and loops

| Arm | Answers | At cap | At-cap primary broken | At-cap loop-only | At-cap whitespace loop | Whitespace loop | Primary broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n+ | 768 | 67 | 26 | 7 | 23 | 176 | 197 |
| n± | 768 | 67 | 26 | 11 | 24 | 176 | 195 |
| k4 | 768 | 57 | 21 | 2 | 16 | 186 | 208 |
| k16 | 768 | 50 | 7 | 7 | 5 | 84 | 114 |
| k64 | 768 | 59 | 9 | 8 | 8 | 56 | 77 |
| k256 | 768 | 78 | 17 | 15 | 11 | 54 | 71 |
| k1024 | 768 | 79 | 12 | 12 | 7 | 45 | 66 |
| full | 768 | 84 | 14 | 11 | 12 | 55 | 73 |
| iso | 768 | 63 | 7 | 6 | 6 | 45 | 61 |

A whitespace loop is a three-word sequence three or more times with a word being `str.split`'s; the scorer's loop rule counts triage's word tokens instead, so the two counts differ.

Note: Rate-derived distances are computed from the published point estimates; they are not a rerun of the scorer under edited answers and do not predict independent row flips.
