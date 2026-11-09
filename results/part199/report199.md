# Part 199 report (printed by part199/report199.py)

## Verdicts: published (SDPA) beside the eager rerun

| Prediction | Published | Published 95% interval | Eager | Eager 95% interval | Changed |
| --- | --- | --- | --- | --- | --- |
| 170.1 | holds (0.174) | 0.000 to 0.443 | fails (0.329) | 0.111 to 0.630 | yes |
| 170.2 | fails (0.360) | 0.055 to 0.804 | fails (0.832) | 0.500 to 1.333 | no |
| 170.3 | holds (0.062) | -0.040 to 0.160 | holds (0.121) | 0.018 to 0.223 | no |
| 170.4 | fails (-0.073) | -0.206 to 0.057 | fails (-0.221) | -0.367 to -0.089 | no |
| 170.5 | fails (0.047) | -0.026 to 0.070 | fails (0.034) | -0.034 to 0.065 | no |
| 170.6 | holds (0.071) | 0.040 to 0.104 | holds (0.062) | 0.024 to 0.099 | no |
| 170.7 | holds (0.011) | 0.000 to 0.056 | holds (0.000) | 0.000 to 0.000 | no |
| 170.8 | fails (0.040) | -0.076 to 0.163 | fails (-0.014) | -0.141 to 0.127 | no |
| 170.W1 | fails (0.067) | 0.024 to 0.135 | fails (0.219) | 0.146 to 0.302 | no |
| 170.W2 | fails (-0.115) | -0.150 to -0.082 | fails (-0.129) | -0.174 to -0.083 | no |

## Cells: loop rate (broken rate), published beside eager

| Stage | Group | Dose | Ban | Published rows | Published loop (broken) | Eager rows | Eager loop (broken) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_push | m96 | 0.0 | b0 | 84 | 0.0% (0.0%) | 96 | 1.0% (1.0%) |
| no_push | m96 | 0.0 | b200 | 84 | 15.5% (19.0%) | 96 | 20.8% (24.0%) |
| no_push | m96 | 0.0 | b5 | 84 | 2.4% (3.6%) | 96 | 6.2% (7.3%) |
| no_push | m96 | 0.0 | b50 | 84 | 3.6% (3.6%) | 96 | 10.4% (10.4%) |
| no_push | m96 | 0.0 | rel50 | 84 | 4.8% (4.8%) | 96 | 0.0% (0.0%) |
| no_push | p8 | 0.0 | b0 | 8 | 0.0% (0.0%) | 8 | 0.0% (0.0%) |
| no_push | p8 | 0.0 | b200 | 8 | 25.0% (25.0%) | 8 | 25.0% (25.0%) |
| no_push | p8 | 0.0 | b5 | 8 | 25.0% (25.0%) | 8 | 25.0% (25.0%) |
| no_push | p8 | 0.0 | b50 | 8 | 0.0% (0.0%) | 8 | 0.0% (0.0%) |
| no_push | p8 | 0.0 | rel50 | 8 | 0.0% (0.0%) | 8 | 0.0% (0.0%) |
| no_push | q96 | 0.0 | b0 | 89 | 1.1% (1.1%) | 96 | 0.0% (0.0%) |
| no_push | q96 | 0.0 | b200 | 89 | 24.7% (31.5%) | 96 | 35.4% (40.6%) |
| no_push | q96 | 0.0 | b5 | 89 | 1.1% (1.1%) | 96 | 4.2% (4.2%) |
| no_push | q96 | 0.0 | b50 | 89 | 7.9% (9.0%) | 96 | 21.9% (22.9%) |
| no_push | q96 | 0.0 | other50 | 89 | 2.2% (2.2%) | 96 | 0.0% (0.0%) |
| no_push | q96 | 0.0 | rel50 | 89 | 2.2% (3.4%) | 96 | 0.0% (1.0%) |
| push | p8 | 0.25 | b0 | 384 | 2.6% (2.6%) | 384 | 0.0% (0.0%) |
| push | p8 | 0.25 | b200 | 384 | 29.9% (31.8%) | 384 | 44.8% (47.1%) |
| push | p8 | 0.25 | b5 | 384 | 15.1% (15.1%) | 384 | 23.4% (23.4%) |
| push | p8 | 0.25 | b50 | 384 | 15.6% (15.6%) | 384 | 24.2% (24.5%) |
| push | p8 | 0.5 | b0 | 384 | 7.6% (7.6%) | 384 | 4.9% (4.9%) |
| push | p8 | 0.5 | b200 | 384 | 35.4% (39.3%) | 384 | 42.7% (47.4%) |
| push | p8 | 0.5 | b5 | 384 | 20.6% (20.6%) | 384 | 26.8% (26.8%) |
| push | p8 | 0.5 | b50 | 384 | 26.0% (26.8%) | 384 | 33.9% (34.6%) |
| push | p8 | 1.0 | b0 | 384 | 25.0% (25.0%) | 384 | 28.4% (28.4%) |
| push | p8 | 1.0 | b200 | 384 | 50.3% (58.3%) | 384 | 58.1% (65.9%) |
| push | p8 | 1.0 | b5 | 384 | 39.1% (41.4%) | 384 | 47.9% (50.5%) |
| push | p8 | 1.0 | b50 | 384 | 43.8% (45.3%) | 384 | 54.7% (56.2%) |
| push | p8 | 1.0 | rel50 | 384 | 22.9% (24.2%) | 384 | 28.6% (29.9%) |

## Stage 0 frozen prompts

| Set | Published frozen | Eager frozen | Overlap |
| --- | --- | --- | --- |
| p8 | 0 | 0 | 0 |
| m96 | 12 | 0 | 0 |
| q96 | 7 | 0 | 0 |

## Stage 0 plain answers: published (SDPA, one row at a time) against eager

- Prompts: 200; identical through the first 40 answer tokens (the Stage 1 list path): 158; identical whole answer: 5.
- Answers that differ: 195; the first differing token is generated over 28 to 72 keys (median 66); 45 at 64 keys or fewer.
- p8: 7 of 8 identical through 40 tokens, 2 identical whole.
- m96: 78 of 96 identical through 40 tokens, 3 identical whole.
- q96: 73 of 96 identical through 40 tokens, 0 identical whole.

## Stage 1 ban lists: published against eager

Registered count (`--compare`, ordered equality on every list): 0 of 200.

| List | Path | Prompts | Same order | Same tokens (as a set) | Shared tokens, median | Shared tokens, min |
| --- | --- | --- | --- | --- | --- | --- |
| b5 | same path | 158 | 133 | 149 | 5.0 | 4 |
| b5 | changed path | 42 | 5 | 12 | 4.0 | 1 |
| b50 | same path | 158 | 0 | 102 | 50.0 | 48 |
| b50 | changed path | 42 | 0 | 2 | 42.0 | 24 |
| b200 | same path | 158 | 0 | 8 | 198.0 | 195 |
| b200 | changed path | 42 | 0 | 0 | 162.5 | 99 |

## Loop onset on the eager answers (descriptive)

- All loops: 1816; established before t0: 1272; after: 544.
- Dose-1 b50: 210; before t0: 155; after: 55.

## Exploratory (not pre-registered)

No-push loops on the prompts Part 170 left eligible (its 19 frozen prompts removed from both runs):

| Set and ban | Published loops / rows | Eager loops / rows |
| --- | --- | --- |
| identity_b0 | 0 / 92 | 0 / 92 |
| identity_b50 | 3 / 92 | 9 / 92 |
| q96_b0 | 1 / 89 | 0 / 89 |
| q96_b50 | 7 / 89 | 19 / 89 |

Dose-1 push cells, answer length:

| Ban | Published mean tokens | Published stopped before 80 | Eager mean tokens | Eager stopped before 80 |
| --- | --- | --- | --- | --- |
| b0 | 55.86 | 358 of 384 | 68.23 | 168 of 384 |
| b50 | 47.54 | 379 of 384 | 68.55 | 104 of 384 |
| b200 | 41.89 | 383 of 384 | 61.89 | 117 of 384 |

## Gates

- Part 170 gates 1 to 8, published: True; eager: True.
- Gate 6 dose-1 b0 loop rate: published 0.250, eager 0.284; b50 mean blocked steps: published 16.02, eager 17.36.
- Gate 8 repeat: published 0 of 384 changed; eager 0 of 384. Noise-limited: False.
- Gate R2: True (144 directions, worst relative write error 0.0060).
