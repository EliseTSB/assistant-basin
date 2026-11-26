# Stage C2 Results: Unified 3-Pillar Causal Dissection of Common-Mode Bleed, ChatML Amplification, 5-Layer Post-Erasure Band, and Low-`γ` `pre_norm` Masking on OLMo 3 7B

**Status:** Pre-registered (`OLMO3-TRANSPORT-C2-DESIGN.md` Registered v4 `3764dfee…`, signed off `2026-11-25` on `main` @ `ecd49b28`, Council `#119`; frozen runner/scorer/tests signed off at **Code PASS** on `vesper-c2-wsp` @ `f5a7d7da`, Council `#125`, `2026-11-25 19:11 CT`; executed verbatim on NVIDIA L4 [`acct1`], `744.45 s` [`12.4 min`] GPU compute + `399 s` chunked upload of `2.11 GB` [`1.97 GiB`] `r_lens.pt`, `8,623` forwards executed, `0` capped skipped; **`0.77 CU`** settled via Colab ledger [`165.87 → 165.10 CU`, `Usage rate: 0.00/hr`, `0` active runtimes verified >1 hr post-teardown — run-ledger attestations external to the JSON bundle]).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-REPORT.md)  
**Pre-Registered Design:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-DESIGN.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-DESIGN.md) (`sha256 = 3764dfee6baf9c7387fc6325eee92c4ca2fdbca5c76ee5b440e0f4f12169d8fc`)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Pre-Registered Verdicts (Fail-Closed Gate Honored)

- **Global Gate `C2.0`:** **`FAIL`** — sole failed sub-clause: **`C2.0.2` `tele_p999 = 1.038869 > 0.15`** (`nonfinite_count = 0`, `tele_p95 = 0.067939 <= 0.08` PASS). All other sub-gates passed cleanly: `C2.0.1` **PASS** (`max_write_sham = 0.0`, `max_patch_sham = 0.0`, `61/61` active arms live), `C2.0.3` **PASS** (all block counts, per-cell counts, 32-layer `post_mlp_norms`, and live dose checks OK; Regime-N `N = 2,880`: `0` capped, `0` RMSN > 20%, `0` RMSN > 12%; Regime-D `N = 864`: `0` capped, `0` patched capped, `0` RMSN > 20%, `0` RMSN > 12%, `fde_p95 = 0.227831 <= 0.30`, `0` FDE > 0.45, `0` FDE > 0.75), and `C2.0.4` **PASS** (`4/4` strict SHA-256 provenance hashes verified on disk and in receipt).
- **All Primary Confirmatory Claims (`C2.1`, `C2.2`, `C2.3`, `C2.4a`, `EQ-C2.4c`):** **`BLOCKED_BY_C2.0`**. Under the fail-closed rule, no confirmatory claim is made on any primary hypothesis.

Post-hoc root-cause autopsy of `C2.0.2` (pinpointing `ctrl_L4_16_MLP_LongBare_to_ChatML`, `N = 288`, which patches MLPs at `L4–L16` prior to `inject_layer = 18` while `decompose_target_trace(..., inject_layer=18)` sums only `L19–L31`; across all other `6,144` traces `tele_p95 = 0.0506 <= 0.08`, `tele_p99.9 = 0.0931 <= 0.15`, `max = 0.1274`), registered KL notation erratum (`#121 #8`), and explicitly non-confirmatory exploratory counterfactual readouts (reproducible via `python3 -I olmo3_virtual_and_highway/exploratory_c2_counterfactual.py`) are documented separately in [`OLMO3-TRANSPORT-C2-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-REPORT.md).

## Bundle Contents & Provenance

- `c2_receipt.json.gz` (`743,047` bytes; `.gz sha256 = f129db3f41a424ed5410436bd37f2c9625f33c405198e028da6e6dca733b04e5`, uncompressed `3,913,068` bytes, `sha256 = 257c21046f9eb3cb71b0f74eb110d3558673bd3632d57a02f358bc6724c20a94`)
- `scorecard.json` (`2,093` bytes; `sha256 = 2859050926cacdb72b6431952d61944e90c9a2b8f54e050721effd2ff5f9ebf1`)
- `SCORECARD.md` (`464` bytes; `sha256 = 6e0e7ecd789d5669558550dd02a5d9114dabe3073e445d4349fb5d4a8a5385e5`)

## Rescore It Yourself (Python 3.10+ Standard Library Only)

From the repository root (`cd assistant-basin`), using **only** the Python 3.10+ standard library (`math.fsum` deterministic across Python 3.10–3.14):

```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_c2.py \
  --receipt results/vesper-olmo3-7b-stage-c2/c2_receipt.json.gz \
  --c2-constants olmo3_virtual_and_highway/c2_prereg_constants.json \
  --design olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-DESIGN.md \
  --out-json /tmp/olmo3_c2_scorecard.json \
  --out-md /tmp/olmo3_c2_SCORECARD.md
cmp /tmp/olmo3_c2_scorecard.json results/vesper-olmo3-7b-stage-c2/scorecard.json
cmp /tmp/olmo3_c2_SCORECARD.md results/vesper-olmo3-7b-stage-c2/SCORECARD.md
```

Both `cmp` commands print nothing and exit `0` when reproduced byte-for-byte.

To reproduce the post-hoc `C2.0.2` telescoping autopsy and explicitly non-confirmatory exploratory counterfactual diagnostics (zero edits to any frozen file):

```bash
python3 -I olmo3_virtual_and_highway/exploratory_c2_counterfactual.py \
  --receipt results/vesper-olmo3-7b-stage-c2/c2_receipt.json.gz \
  --constants olmo3_virtual_and_highway/c2_prereg_constants.json \
  --design olmo3_virtual_and_highway/OLMO3-TRANSPORT-C2-DESIGN.md
```
