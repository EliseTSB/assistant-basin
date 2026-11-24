# Stage C0b Results: Pre-Registered `positions="all"` Calibration, Attention-Share Attribution & Live-Paired `R_ridge` Cross-Domain Whitening Test on OLMo 3 7B

**Status:** Pre-registered (`4e2b1820`, Registered v1; runner/scorer frozen at git commit `6396e2af`, attested by git history; executed on NVIDIA L4, `443.79 s` / `7.4 min`, `5,328` forwards executed, `0` capped skipped; `0.228 CU` compute alone at `1.85 CU/hr`, `0.28 CU` billed via Colab ledger, `0` active runtimes verified — run-ledger attestations external to the JSON bundle).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-REPORT.md)  
**Pre-Registered Design:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-DESIGN.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-DESIGN.md) (`sha256 = 8b3efeaeba3febbcc0e9715840a79bea6d6bc29986f7bf21a89cf6080d61a994`)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Pre-Registered Verdicts (Fail-Closed Gate Honored)

- **Global Gate `C0b.0` (clauses `1–4a`):** **FAIL** — failed on a single clause: **`C0b.0.3` `fde_err_p99_9 = 1.4561 > 0.45`** (`fde_err_max = 1.6255`).
- **All downstream verdicts (`C0b.0.4b_drift`, `C0b.1a`, `C0b.1b`, `C0b.2a`, `C0b.2b`):** **`BLOCKED_BY_C0b.0`**. Under the registered fail-closed rule, no confirmatory claim is made on any downstream hypothesis.

| Check | Verdict | Registered Result vs. Bar |
|---|---|---|
| **Gate `C0b.0` (global `1–4a`)** | **FAIL** | **PASS:** `C0b.0.1` (`8/8` shams `0.0`, movement & Regime-N/D arm liveness `true`); `C0b.0.2` (`5,312` finite traces, tel `median = 0.0156`, `p95 = 0.0496 <= 0.08`, `p99.9 = 0.0920 <= 0.15`, `max = 0.1155`); `C0b.0.4a` (`10/10` strict SHA-256); `C0b.0.3` cap rate `0.0000` (`<= 0.02`), `0` cell failures, `fde_err_p95 = 0.2983 <= 0.30`. **FAIL:** `C0b.0.3` **`fde_err_p99_9 = 1.4561 > 0.45`** (`max = 1.6255`; Bare `p95/max = 0.2689 / 0.8929`, ChatML `p95/max = 0.3298 / 1.6255`). |
| **Gate `C0b.0.4b` (`C0b.1b` drift)** | **BLOCKED_BY_C0b.0** | Raw drift check passed with exact `0.000000` difference across `320` anchor traces (`" lantern"`, `" judge"`) and `5,760` off-target cells (`max_anchor_gap_diff = 0.0`, `max_anchor_attn_gap_diff = 0.0`, `max_R_mf_B_cross_diff = 0.0`, `max_R_mf_offtarget_cell_diff = 0.0`, `max_R_mf_all16_cell_diff = 0.0`); recorded `BLOCKED_BY_C0b.0` by `score_olmo3_c0b.py`'s fail-closed global gate convention. |
| **`C0b.1a`, `C0b.1b`, `C0b.2a`, `C0b.2b`** | **BLOCKED_BY_C0b.0** | Held closed by Gate `C0b.0` failure; post-hoc descriptive readouts are documented separately in [`OLMO3-TRANSPORT-C0B-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-REPORT.md). |

## Post-Hoc Receipt Diagnostic for Gate `C0b.0.3`

The pre-registered evidence for Gate `C0b.0.3` is the scorecard quantiles and by-format logging above. The per-`(layer, arm)` breakdown below is a **post-hoc receipt diagnostic, not a registered gate clause**:

- Across all `3,072` retained Regime-D `positions="all"` traces, `80` traces exceeded `|frozen_direct_effect - 3.0| > 0.45` (every one had `frozen_direct_effect > 3.0`, indicating `r_i < r_c`).
- **`73 / 80` (`91.25%`)** of those misses came from one block: **`L12 R_mf` under `positions="all"`** (`55/64` ChatML, `18/64` Bare; `6/512` in `L12 direct_matched`, `1/128` in `L18 R_mf`, and **`0/2,304`** across `L24`, `L27`, `L18 direct_matched`/`logit_lens`, and `L12 logit_lens`).
- At `L12`, matching `Direct_linear = +3.0` requires `dose = 4.13–6.95` (`0.97–1.64×` `post_mlp_norm = 4.2478`, median `1.32×`) at every token `0..T-1`. That all-position push shrank the intervened final RMSNorm denominator `r_i` relative to clean `r_c` by up to **`35.1%`** (`r_i / r_c` min `0.6486`, median `0.8428`; ChatML median shrink `20.1%`, max `35.1%`; Bare median `6.2%`, max `22.9%`), moving the model out of the linear regime assumed by the decomposition. By contrast, `L24` and `L27` remained in the linear regime (`R_mf` max shrink `3.93%` and `2.63%`, `fde_err max = 0.1226` and `0.0812`).

## Bundle Contents & Provenance

- `c0b_receipt.json.gz` (`453,084` bytes; `.gz sha256 = 06903d73ab4a01036d91a6d1a456b115a9d3fc9b9b22884b3db9a103f15dd272`, uncompressed `sha256 = bc477e3ec98692eda5d6e78eb2d7d7def2061ca566ea18904f89bae7fa170db0`)
- `scorecard.json` (`5,240` bytes; `sha256 = ff79d7516677b8df8c02f0f888aa42809fe46e57d8dfff74d75da7ca229dbf41`)
- `SCORECARD.md` (`342` bytes; `sha256 = fd9ed7f3b9fdcbcb3fb552a1b4d6494987f9aa7df618954869d4025ae86d0ade`)

## Rescore it yourself (Python 3.10+ standard library only)

From the repository root (`cd assistant-basin`), using **only** the Python 3.10+ standard library (zero `numpy`, `scipy`, or `torch`; uses `math.fsum` for Python 3.10–3.14 byte-identity):

```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_c0b.py \
  --receipt results/vesper-olmo3-7b-stage-c0b/c0b_receipt.json.gz \
  --c0b-constants olmo3_virtual_and_highway/c0b_prereg_constants.json \
  --c0-constants olmo3_virtual_and_highway/c0_prereg_constants.json \
  --c0-receipt results/vesper-olmo3-7b-stage-c0/c0_receipt.json.gz \
  --design olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-DESIGN.md \
  --out-dir /tmp/olmo3_c0b_rescore
cmp /tmp/olmo3_c0b_rescore/scorecard.json results/vesper-olmo3-7b-stage-c0b/scorecard.json
cmp /tmp/olmo3_c0b_rescore/SCORECARD.md results/vesper-olmo3-7b-stage-c0b/SCORECARD.md
```

Both `cmp` commands print nothing and exit `0` when reproduced byte-for-byte.
