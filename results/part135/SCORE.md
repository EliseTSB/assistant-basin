# Part 135 score

## Gates

- **known_examples: PASS** — DeepSeek and Luna each get at least 11 of 12 right.
  DeepSeek 12/12; Luna 12/12 (missing count as incorrect).
- **shuffled_persona_same: PASS** — DeepSeek calls at most 10% of shuffled persona pairs same.
  21/250 same (0.084).
- **second_reader_kappa: PASS** — three-way Cohen kappa on shared 15% is at least 0.60.
  kappa=0.7376967274544094; shared rows=150.

## Three-way shares (Wilson 95% intervals)

### ds
- **fires**: same 29/248 (0.117, CI 0.083–0.163); opposite 5/248 (0.020, CI 0.009–0.046); unrelated 214/248 (0.863, CI 0.815–0.900); missing 2/250.
- **fires_shuffled**: same 3/250 (0.012, CI 0.004–0.035); opposite 0/250 (0.000, CI 0.000–0.015); unrelated 247/250 (0.988, CI 0.965–0.996); missing 0/250.
- **persona**: same 17/250 (0.068, CI 0.043–0.106); opposite 2/250 (0.008, CI 0.002–0.029); unrelated 231/250 (0.924, CI 0.884–0.951); missing 0/250.
- **persona_shuffled**: same 21/250 (0.084, CI 0.056–0.125); opposite 4/250 (0.016, CI 0.006–0.040); unrelated 225/250 (0.900, CI 0.857–0.931); missing 0/250.
### luna
- **fires**: same 4/32 (0.125, CI 0.050–0.281); opposite 0/32 (0.000, CI 0.000–0.107); unrelated 28/32 (0.875, CI 0.719–0.950); missing 218/250.
- **fires_shuffled**: same 0/28 (0.000, CI 0.000–0.121); opposite 0/28 (0.000, CI 0.000–0.121); unrelated 28/28 (1.000, CI 0.879–1.000); missing 222/250.
- **persona**: same 3/38 (0.079, CI 0.027–0.208); opposite 2/38 (0.053, CI 0.015–0.173); unrelated 33/38 (0.868, CI 0.727–0.942); missing 212/250.
- **persona_shuffled**: same 3/52 (0.058, CI 0.020–0.156); opposite 1/52 (0.019, CI 0.003–0.101); unrelated 48/52 (0.923, CI 0.818–0.970); missing 198/250.

## Predictions

- **135.1: false** — {'true': False, 'value': {'both_pole_total': 1109, 'rotation_null_mean': 1096.587}, 'threshold': 'total >= 2 * null mean'}
- **135.2: false** — {'true': False, 'value': {'L0_17_share': 0.07821275189957053, 'L18_35_share': 0.032387045181927226}, 'threshold': 'late share > early share'}
- **135.3: true** — {'true': True, 'value': 0.924, 'threshold': '>= 0.50'}
- **135.4: false** — {'true': False, 'value': {'persona_same': 0.068, 'persona_shuffled_same': 0.084}, 'threshold': 'difference >= 0.10'}
- **135.5: true** — {'true': True, 'value': {'persona_opposite': 0.008, 'persona_shuffled_opposite': 0.016}, 'threshold': 'difference <= 0.10'}
- **135.6: false** — {'true': False, 'value': {'fires_same': 0.11693548387096774, 'fires_shuffled_same': 0.012}, 'threshold': 'difference <= 0.05'}
- **135.7: true** — {'true': True, 'value': {'kappa': 0.09028761984160111, 'paired_neurons': 248}, 'threshold': 'kappa <= 0.20'}

Luna subset verdicts for 135.3–135.7: {"135.3": true, "135.4": false, "135.5": true, "135.6": false, "135.7": true}

Shares use successfully labeled rows; error and missing rows are counted separately and shown above.
