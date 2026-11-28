# Stage C3b (`WSP v2`) Confirmatory & Diagnostic Scorecard

> **DIAGNOSTIC FLAG (`D4_masking_check` FAIL):** *low-gamma pre_norm masking efficiency unverified on L24* (isolated outside global `C3b.0`; mechanistic attribution of delivery gain to masking withheld).

## Confirmatory Verdicts (Holm-2, `alpha = 0.05`)

| Gate / Hypothesis | Verdict | Point Estimate | 90% BCa CI | `p_perm` | Holm Sig | Notes |
|---|---|---:|---|---:|---|---|
| **Global `C3b.0`** | **`PASS`** | — | — | — | — | `C3b.0.1..4` |
| **`H1`** | **`NOT_READ_LOW_KAPPA`** | `+0.3750` | `[+0.2500, +0.5000]` | `0.000030` | `True` | d*=4.8, lenR=1.023, brk=0.000, failed_guards=['reader_any_coherent_ge_0.85'] |
| **`H2`** | **`FAIL`** | `+0.4271` | `[+0.2500, +0.6562]` | `0.000610` | `True` | d*=2.4, Δmd_P=+0.031, Δmd_R=+0.031, lenR=1.000, failed_guards=['delta_has_md_vs_plain_ge_0.20', 'delta_has_md_vs_rand_ge_0.15'] |

## Capability Preservation (`EQ-C3b.2`) & Non-Gating Refusal Readout

| Arm | Gating | Math Acc (Strict) | Code Pass | Neutral `off_KL` | Neutral Broken | Refusal (Harm/Ben) | `EQ-C3b.2` |
|---|---|---:|---:|---:|---:|---:|---|
| `Plain` | `log` | `0.800` (`0.750`) | `1.000` | `0.0000` | `0.000` | `0.62` (`1.00/0.25`) | `PASS` |
| `Rand_NoMD_12L_s0@d*_nomd` | `log` | `0.800` (`0.800`) | `1.000` | `0.0116` | `0.000` | `0.62` (`1.00/0.25`) | `PASS` |
| `Rand_Pirate_12L_s0@d*_pirate` | `log` | `0.800` (`0.700`) | `1.000` | `0.0481` | `0.000` | `0.62` (`1.00/0.25`) | `PASS` |
| `Sys_NoMD` | `log` | `0.550` (`0.350`) | `1.000` | `1.0469` | `0.000` | `0.62` (`1.00/0.25`) | `FAIL` |
| `Sys_Pirate` | `log` | `0.300` (`0.150`) | `0.917` | `0.8902` | `0.000` | `0.62` (`1.00/0.25`) | `FAIL` |
| `WSP_NoMD_12L@d*_nomd` | `YES` | `0.750` (`0.700`) | `1.000` | `0.2925` | `0.000` | `0.62` (`1.00/0.25`) | `PASS` |
| `WSP_Pirate_12L@d*_pirate` | `YES` | `0.900` (`0.850`) | `0.750` | `0.6619` | `0.000` | `0.75` (`1.00/0.50`) | `FAIL` |

## Gate & Diagnostic Summary

- **`C3b.0`:** `True` (`C3b.0.1=True`, `C3b.0.2=True`, `C3b.0.3=True`, `C3b.0.4=True`)
- **`D4_masking_check` (outside `C3b.0`):** `False` (seed=`2.152517x`/`0.960216x`, L2=`1.686947x`/`1.030786x`, eff=`1.275984x`/`0.931537x`)
- **`C3b.0_direction_quality`:** `{'WSP_Pirate_12L': True, 'WSP_NoMD_12L': True}`
- **`C3b.0_commit_gate`:** `{'WSP_Pirate_12L': True, 'WSP_NoMD_12L': True}`
- **`C3b.0_NoMD_room`:** `True` (`24m=1.000`, `8f=1.000`, `32all=1.000`, `mean_ct=3.375`)
- **`C3b.R`:** `False` (`H1=False`)

## Secondary & Exploratory Readouts (Outside Holm)

- **`D3` Reference Ceilings:** `Sys_Pirate(none)=None`, `Sys_Pirate(ban)=None`, `Sys_NoMD(has_md)=0.03125` (`ΔvsPlain=0.96875`)
- **`D_above_cap`:** Pirate=`D_ABOVE_CAP_UNREAD`, NoMD=`not_triggered`
- **`D_key_ablation` (`WSP_NoMD_12L_kWA`):** `{'selected_beta_star': 1.0, 'fallback': False, 'has_md': 1.0, 'mean_construct_types': 3.40625, 'delta_has_md_vs_plain': 0.0, 'delta_ct_vs_plain': -0.03125, 'broken_rate': 0.0, 'mean_tokens': 96.0, 'em_dash_per_1k_words': 1.4479, 'neutral_median_off_kl': 0.004117}`
- **`D_key_ablation` (`WSP_Pirate_12L_kSparse32`):** status=`D_KEY_PIRATE_UNREAD`, d*=`3.6`
- **`em_dash_per_1k_words` (`NOMD_32`):** `{'Plain': 1.4354, 'Rand_NoMD_12L_s0@d*_nomd': 1.4158, 'Rand_NoMD_12L_s1@d*_nomd': 1.4402, 'Rand_NoMD_12L_s2@d*_nomd': 0.9461, 'Sys_NoMD': 2.3202, 'WSP_NoMD_12L@d*_nomd': 2.2026, 'WSP_NoMD_12L_kWA': 1.4479}`

