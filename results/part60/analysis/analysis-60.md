# Part 60 analysis

Metrics use the frozen definitions in `part60/DEFINITIONS-60.md`.

## dev

| Method | Parsed | Failed | Threshold missing | Real positive-only | Real full-rule | Control positive-only | Control full-rule | Real Hstrong | Control Hstrong | Full-rule gap | Real Hweak | Real Hnone | Control Hnone | 60d all eligible | 60d >=2 strong hits | Real AUC | Calls | Max-token truncations | Cost / unit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 497/500 | 3 | 0 | 0.264 | 0.264 | 0.060 | 0.060 | 0.365 (n=250) | 0.147 (n=247) | 0.204 | — (n=0) | — (n=0) | — (n=0) | — (n=0) | — (n=0) | — (n=0) | 0 | 0 | $0.00000 |
| M0 archive cross-check | 497/500 | — | — | — | 0.264 | — | 0.060 | — | — | — | — | — | — | — | — | — | 0 | 0 | $0.00000 |
| M1 | 469/500 | 31 | 0 | 0.408 | 0.396 | 0.152 | 0.152 | 0.552 (n=233) | 0.276 (n=236) | 0.244 | 0.146 (n=233) | 0.027 (n=233) | 0.034 (n=236) | 0.247 (n=190) | 0.303 (n=155) | 0.838 (n=233) | 500 | 0 | $0.00025 |
| M2 | 463/500 | 37 | 0 | 0.436 | 0.400 | 0.172 | 0.156 | 0.545 (n=233) | 0.316 (n=230) | 0.244 | 0.171 (n=233) | 0.091 (n=233) | 0.061 (n=230) | 0.307 (n=192) | 0.407 (n=145) | 0.805 (n=233) | 500 | 0 | $0.00025 |
| M3 | 438/500 | 62 | 0 | 0.168 | 0.160 | 0.032 | 0.028 | 0.267 (n=221) | 0.096 (n=217) | 0.132 | — (n=0) | 0.020 (n=221) | 0.024 (n=217) | — (n=0) | — (n=0) | 0.688 (n=221) | 969 | 0 | $0.00040 |
| M6 | 497/500 | 3 | 0 | 0.512 | 0.512 | 0.324 | 0.324 | 0.581 (n=249) | 0.442 (n=248) | 0.188 | 0.183 (n=249) | 0.008 (n=249) | 0.028 (n=248) | 0.377 (n=207) | 0.470 (n=166) | 0.869 (n=249) | 520 | 0 | $0.00052 |

Empirically unjudgeable (threshold at or above the pool maximum): 0; known ceilings 500, unknown ceilings 0; by type {}.

### Paired bootstrap differences

2,000 paired resamples; seed 6003.

| Method | Comparator | Measure | Difference | 95% CI |
|---|---|---|---:|---:|
| M0 | M1 | real_positive_only | -0.144 | [-0.208, -0.080] |
| M0 | M1 | control_positive_only | -0.092 | [-0.136, -0.044] |
| M0 | M1 | gap_positive_only | -0.052 | [-0.136, 0.028] |
| M0 | M1 | real_full_rule | -0.132 | [-0.196, -0.068] |
| M0 | M1 | control_full_rule | -0.092 | [-0.136, -0.044] |
| M0 | M1 | gap_full_rule | -0.040 | [-0.124, 0.040] |
| M1 | M0 | real_positive_only | 0.144 | [0.068, 0.212] |
| M1 | M0 | control_positive_only | 0.092 | [0.048, 0.136] |
| M1 | M0 | gap_positive_only | 0.052 | [-0.032, 0.132] |
| M1 | M0 | real_full_rule | 0.132 | [0.056, 0.200] |
| M1 | M0 | control_full_rule | 0.092 | [0.048, 0.136] |
| M1 | M0 | gap_full_rule | 0.040 | [-0.044, 0.124] |
| M2 | M0 | real_positive_only | 0.172 | [0.108, 0.236] |
| M2 | M0 | control_positive_only | 0.112 | [0.064, 0.164] |
| M2 | M0 | gap_positive_only | 0.060 | [-0.016, 0.144] |
| M2 | M0 | real_full_rule | 0.136 | [0.068, 0.204] |
| M2 | M0 | control_full_rule | 0.096 | [0.052, 0.148] |
| M2 | M0 | gap_full_rule | 0.040 | [-0.040, 0.120] |
| M2 | M1 | real_positive_only | 0.028 | [-0.032, 0.092] |
| M2 | M1 | control_positive_only | 0.020 | [-0.036, 0.076] |
| M2 | M1 | gap_positive_only | 0.008 | [-0.076, 0.088] |
| M2 | M1 | real_full_rule | 0.004 | [-0.060, 0.072] |
| M2 | M1 | control_full_rule | 0.004 | [-0.052, 0.060] |
| M2 | M1 | gap_full_rule | 0.000 | [-0.084, 0.088] |
| M3 | M0 | real_positive_only | -0.096 | [-0.160, -0.036] |
| M3 | M0 | control_positive_only | -0.028 | [-0.064, 0.004] |
| M3 | M0 | gap_positive_only | -0.068 | [-0.140, 0.004] |
| M3 | M0 | real_full_rule | -0.104 | [-0.172, -0.044] |
| M3 | M0 | control_full_rule | -0.032 | [-0.068, -0.004] |
| M3 | M0 | gap_full_rule | -0.072 | [-0.144, 0.000] |
| M3 | M1 | real_positive_only | -0.240 | [-0.304, -0.172] |
| M3 | M1 | control_positive_only | -0.120 | [-0.168, -0.076] |
| M3 | M1 | gap_positive_only | -0.120 | [-0.200, -0.040] |
| M3 | M1 | real_full_rule | -0.236 | [-0.300, -0.172] |
| M3 | M1 | control_full_rule | -0.124 | [-0.168, -0.080] |
| M3 | M1 | gap_full_rule | -0.112 | [-0.192, -0.032] |
| M6 | M0 | real_positive_only | 0.248 | [0.188, 0.312] |
| M6 | M0 | control_positive_only | 0.264 | [0.204, 0.324] |
| M6 | M0 | gap_positive_only | -0.016 | [-0.104, 0.068] |
| M6 | M0 | real_full_rule | 0.248 | [0.188, 0.312] |
| M6 | M0 | control_full_rule | 0.264 | [0.204, 0.324] |
| M6 | M0 | gap_full_rule | -0.016 | [-0.104, 0.068] |
| M6 | M1 | real_positive_only | 0.104 | [0.040, 0.164] |
| M6 | M1 | control_positive_only | 0.172 | [0.116, 0.232] |
| M6 | M1 | gap_positive_only | -0.068 | [-0.152, 0.020] |
| M6 | M1 | real_full_rule | 0.116 | [0.052, 0.180] |
| M6 | M1 | control_full_rule | 0.172 | [0.116, 0.232] |
| M6 | M1 | gap_full_rule | -0.056 | [-0.144, 0.032] |

Interpretation limits: control hits are not a false-discovery rate for labels because Part 58 notes that some random directions carry real features; only 247 real test units with at least 20 top, 10 random, and 5 weak windows are covered, so no atlas-wide claim follows; the test split is prospective for new outputs but not an untouched holdout of earlier decisions; M3 selects a hypothesis-plus-probe distribution and uses about twice M1's generations.

Dropped by generation failure rule: M4/primary 439/500 parsed (12.2% failed); M5/round1 429/500 parsed (14.2% failed)

Dev stop-rule selection: `{"base_gaps": {"full_rule": 0.20400000000000001, "positive_only": 0.20400000000000001}, "best_gaps": null, "best_variant": null, "improvements": null, "leader": "M0", "lower_bounds": null, "stop_rule": "keep_M0_unless_positive_only_and_full_rule_each_lead_by_5pp", "test_candidate": null}`

### Predictions

| Prediction | Status | Evidence |
|---|---|---|
| 60a — M0 pass rates within 3 pp of 28.1% real and 2 pp of 6.2% control | held | real=26.4%; control=6.0% |
| 60b — M1 verified rate at least 15% real and at most 3% control | failed | real=39.6%; control=15.2% |
| 60c — M1 real strong-tier hit rate exceeds M0 by at least 10 pp | held | M1=55.2%; M0=37.8%; difference=17.4% on the same 233 real units |
| 60d — M1 strong > weak > none ordering in at least 70% of eligible real units | failed | ordering=24.7%; n=190 |
| 60e — M2 lowers real mean H_none by at least one third relative to M1 | failed | M2=9.1%; M1=2.7% |
| 60f — M3 fresh-sentence real verified rate is within 5 pp of M1 | failed | M3 fresh=16.0%; M1=39.6% |
| 60g — M5 round 2 raises real verified rate by at least 5 pp and control rate by more than 2 pp | unscored | real round1=unscored, round2=unscored; control round1=unscored, round2=unscored |
| 60h — M4 marked-token pass rate minus whole-text pass rate is at most 0 pp | unscored | marked=unscored; whole-text=unscored; difference=unscored; positive difference is a scoring bug; same marked-comparable denominator n=None; M4−M1 real pass-rate differences: positive-only=unscored, full-rule=unscored |
| 60i — M6 differs from M1 by at most 3 pp in real verified rate and costs at least twice as much | failed | M6=51.2%; M1=39.6%; cost/unit M6=0.0005216638; M1=0.00025248100000000004 |
| 60j — No tested method beats both M1 and M0 test gaps by at least 5 pp | unscored | unscored until the registered test phase |

