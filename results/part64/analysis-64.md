# Part 64 analysis

| Arm | Real pass rate | Control pass rate | Gap | n real | n control | max_tokens |
|---|---:|---:|---:|---:|---:|---:|
| haiku-1x | 52.4% | 26.0% | 26.4% | 250 | 250 | 0 |
| haiku-3x | 54.4% | 32.8% | 21.6% | 250 | 250 | 0 |
| opus-1x | 67.6% | 49.2% | 18.4% | 250 | 250 | 0 |
| opus-3x | 69.2% | 58.0% | 11.2% | 250 | 250 | 0 |

## Predictions

| ID | Registered prediction | Estimate | 95% paired bootstrap CI | Result |
|---|---|---:|---:|---|
| 64a | Haiku-3x raises the real pass rate over Haiku-1x by at least 5 points. | +2.0% | [-5.2%, +8.4%] | not met |
| 64b | Haiku-3x raises the gap over Haiku-1x by at least 5 points (more examples make a hypothesis more specific). | -4.8% | [-14.8%, +5.2%] | not met |
| 64c | Haiku-3x raises the control pass rate by at most 3 points over Haiku-1x. | +6.8% | [-0.8%, +14.0%] | not met |
| 64d | Opus-1x has a gap at least 5 points above Haiku-1x. | -8.0% | [-17.6%, +1.6%] | not met |
| 64e | Opus gains more from 3x than Haiku does (interaction on the gap at least 3 points). | -2.4% | [-15.2%, +10.0%] | direction only: miss |
| 64f | Haiku-1x reproduces Part 60's M1 within 6 points on both real and control pass rates. | neuron: +12.8%, control: +10.8% | — | not met |

## Decision rule

an arm advances only if its full-rule gap exceeds Haiku-1x by at least 5 percentage points and the unit-paired bootstrap 95% interval is above zero

Part 60's test split remains closed unless an arm advances.
