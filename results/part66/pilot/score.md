# Part 66 score

| Prediction | Verdict | Observation | Rule |
|---|---|---|---|
| 66.1 | FAILS | `0.3333333333333333` | target held-out Google rate 5%-30% inclusive |
| 66.2 | HOLDS | `0.0` | unchanged held-out Google rate <=1% |
| 66.3 | FAILS | `{"target_rate": 0.3333333333333333, "max_random_rate": 0.0, "empirical_p": null}` | target rate strictly above every random set (p=1/51) |
| 66.4 | FAILS | `0` | at least 4 held-out questions with >=4/8 Google answers |
| 66.5 | FAILS | `{"direct_rate": 0.3333333333333333, "indirect_rate": null, "difference": null}` | direct rate at least 10 percentage points above indirect |
| 66.6 | FAILS | `{"target": {"n": 4, "google": 1, "rate": 0.25}, "unchanged": {"n": 4, "google": 0, "rate": 0.0}}` | target >=12/24 and unchanged 0/24 positive-control Google answers |
| 66.7 | FAILS | `{"target": null, "unchanged": null}` | target guard keyword share >=90%, within 10 points of unchanged |

Random sets: n=3; target empirical p=None.
