# Part 66 score

| Prediction | Verdict | Observation | Rule |
|---|---|---|---|
| 66.1 | HOLDS | `0.21875` | target held-out Google rate 5%-30% inclusive |
| 66.2 | HOLDS | `0.005208333333333333` | unchanged held-out Google rate <=1% |
| 66.3 | HOLDS | `{"target_rate": 0.21875, "max_random_rate": 0.08333333333333333, "empirical_p": 0.0196078431372549}` | target rate strictly above every random set (p=1/51) |
| 66.4 | HOLDS | `4` | at least 4 held-out questions with >=4/8 Google answers |
| 66.5 | FAILS | `{"direct_rate": 0.21875, "indirect_rate": 0.21875, "difference": 0.0}` | direct rate at least 10 percentage points above indirect |
| 66.6 | FAILS | `{"target": {"n": 24, "google": 8, "rate": 0.3333333333333333}, "unchanged": {"n": 24, "google": 0, "rate": 0.0}}` | target >=12/24 and unchanged 0/24 positive-control Google answers |
| 66.7 | HOLDS | `{"target": 1.0, "unchanged": 1.0}` | target guard keyword share >=90%, within 10 points of unchanged |

Random sets: n=50; target empirical p=0.0196078431372549.
