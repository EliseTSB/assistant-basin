# Stage C3 (`WSP v1`) Confirmatory & Diagnostic Scorecard

## Confirmatory Verdicts (Holm-4, `alpha = 0.05`)

| Gate / Hypothesis | Verdict | Point Estimate | 90% BCa CI | `p_perm` | Holm Sig | Notes |
|---|---|---:|---|---:|---|---|
| **Global `C3.0`** | **`FAIL`** | — | — | — | — | `C3.0.1..4` |
| **`H1`** | **`BLOCKED_BY_C3.0`** | `—` | `—` | `1.000000` | `False` | blocked (BLOCKED_BY_C3.0) |
| **`H2`** | **`BLOCKED_BY_C3.0`** | `—` | `—` | `1.000000` | `False` | blocked (BLOCKED_BY_C3.0) |
| **`H3`** | **`BLOCKED_BY_C3.0`** | `—` | `—` | `1.000000` | `False` | blocked (BLOCKED_BY_C3.0) |
| **`H4`** | **`BLOCKED_BY_C3.0`** | `—` | `—` | `1.000000` | `False` | blocked (BLOCKED_BY_C3.0) |

## Capability Preservation (`EQ-C3.2`)

| Arm | Gating | Math Acc (Strict) | Code Pass | Neutral `off_KL` | Neutral Broken | `EQ-C3.2` |
|---|---|---:|---:|---:|---:|---|

## Gate Summary

- **`C3.0`:** `False` (`C3.0.1=True`, `C3.0.2=False`, `C3.0.3=True`, `C3.0.4=True`)
- **`C3.0_direction_quality`:** `{}`
- **`C3.0_commit_gate`:** `{k: v['pass'] for k, v in ...}` = `not_evaluated (C3.0 FAIL)`
- **`C3.0_NoMD_room`:** `False` (`has_md=0.000`, `mean_ct=0.000`)
- **`C3.R`:** `False` (`H1=False`, `H2=False`)
