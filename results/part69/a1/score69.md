# Part 69 score

| condition | Dg | null mean | null min | null max | p |
|---|---:|---:|---:|---:|---:|
| makers | 2.7760417 | -0.00052083333 | -0.036458333 | 0.052083333 | 0.0196078 |
| l11_alone | 0.3125 | -0.0075 | -0.140625 | 0.125 | 0.0196078 |
| l29_alone | 0.0052083333 | -0.0013541667 | -0.0625 | 0.03125 | 0.45098 |
| l11_down | 0.77604167 | 0.0459375 | -0.61458333 | 0.28125 | 0.0392157 |
| l29_up | -3.4479167 | 0.005625 | -0.19791667 | 0.39583333 | 0.0392157 |

## Verdicts

| prediction | verdict | details |
|---|---|---|
| 69.1 | PASS | makers Dg > 0 and exceeds all 50 random maker sets |
| 69.2 | PASS | mean unchanged maker mass >= 0.5 and Alibaba top on >= 20/24 |
| 69.3 | FAIL | L11 alone does not beat activity-matched units in positive direction (p > 0.05) |
| 69.4 | FAIL | L29 alone is positive and beats all 50 activity-matched units |
| 69.5 | FAIL | L11-down two-sided p > 0.1 |
| 69.6 | FAIL | L29-up two-sided p > 0.1 |

Unchanged mean maker-token mass: 0.962811; Alibaba top token: 24/24.
Positive-control question (index 24) is reported separately and is excluded from Dg and verdicts.
