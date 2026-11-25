# OLMo 3 7B Stage C1 — Pre-Registered Confirmatory Scorecard

| Gate / Claim | Verdict |
|---|---|
| **Global Gate `C1.0`** | **`FAIL`** |
| **`C1.1` (Attention transport + calibration)** | **`BLOCKED_BY_C1.0`** |
| **`C1.2` (`R_ridge` bleed whitening)** | **`BLOCKED_BY_C1.0`** |
| **`C1.3` (`H_A` vs. `H_B` bleed fork)** | **`BLOCKED_BY_C1.0`** |
| **`C1.4` (Token-direction contraction)** | **`BLOCKED_BY_C1.0`** |
| *Secondary `L12_linear_check` (non-gating)* | *`FAIL`* |

## Gate `C1.0` Failure Summary
```json
{
  "PASS": false,
  "C1.0.1_shams_and_liveness": {
    "PASS": true,
    "shams": {
      "L18_last": 0.0,
      "L18_all": 0.0,
      "L24_last": 0.0,
      "L24_all": 0.0,
      "L27_last": 0.0,
      "L27_all": 0.0,
      "L12_all": 0.0
    },
    "max_rmf_movement": {
      "L18_last": 10.146484,
      "L18_all": 15.355469,
      "L24_last": 11.578125,
      "L24_all": 9.25,
      "L27_last": 12.390625,
      "L27_all": 8.4375,
      "L12_all": 6.71875
    }
  },
  "C1.0.2_finite_and_telescoping": {
    "PASS": true,
    "all_finite": true,
    "nonfinite_count": 0,
    "nonfinite_sample": [],
    "n_primary_traces": 5952,
    "p95": 0.046991,
    "p99_9": 0.086889,
    "max": 0.105631,
    "bare_p95": 0.041732,
    "chatml_p95": 0.051978
  },
  "C1.0.3_regime_D_all_linearity": {
    "PASS": false,
    "n_traces_evaluated": 1728,
    "capped_skipped_count": 0,
    "anchor_all_ok": false,
    "anchor_cells": {
      "L18_R_mf": {
        "median_dose_over_pmn": 0.471274,
        "expected": 0.4809,
        "rel_diff": 0.020017,
        "ok": true,
        "n": 192
      },
      "L18_direct_matched": {
        "median_dose_over_pmn": 0.471274,
        "expected": 0.4809,
        "rel_diff": 0.020018,
        "ok": true,
        "n": 384
      },
      "L24_R_mf": {
        "median_dose_over_pmn": 0.389438,
        "expected": 0.222,
        "rel_diff": 0.754223,
        "ok": false,
        "n": 192
      },
      "L24_direct_matched": {
        "median_dose_over_pmn": 0.389438,
        "expected": 0.222,
        "rel_diff": 0.754223,
        "ok": false,
        "n": 384
      },
      "L27_R_mf": {
        "median_dose_over_pmn": 0.34498,
        "expected": 0.1642,
        "rel_diff": 1.100974,
        "ok": false,
        "n": 192
      },
      "L27_direct_matched": {
        "median_dose_over_pmn": 0.34498,
        "expected": 0.1642,
        "rel_diff": 1.100973,
        "ok": false,
        "n": 384
      }
    },
    "rmsn_gt_20pct_count": 0,
    "rmsn_gt_12pct_count": 5,
    "rmsn_max": 0.181666,
    "fde_p95": 0.185348,
    "fde_gt_045_count": 2,
    "fde_gt_075_count": 0,
    "fde_max": 0.665984,
    "regime_N_rmsn_diagnostic": {
      "p95": 0.029985,
      "p99_9": 0.061252,
      "max": 0.081493
    }
  },
  "C1.0.4_provenance": {
    "PASS": true,
    "checks": {
      "strict_provenance_enabled": true,
      "meta_r_lens": true,
      "meta_c1_constants": true,
      "meta_c1_bank": true,
      "meta_c1_design": true,
      "disk_c1_constants": true,
      "disk_c1_design": true
    }
  }
}
```
