# Stage C1 Results: Fresh Mixed-POS Linear-Regime Attention Transport, Bleed Fork (`H_A` vs. `H_B`), and Token-Direction Contraction on OLMo 3 7B

**Status:** Pre-registered (`OLMO3-TRANSPORT-C1-DESIGN.md` Registered v5 `5304677e…` + pre-run `OLMO3-TRANSPORT-C1-AMENDMENT-1.md` `271cabf1…`; executed on NVIDIA A100, `326.8 s` GPU compute, `6,543` forwards executed, `0` capped skipped; `1.71 CU` billed via Colab ledger including `490 s` upload of `2.02 GB` `r_lens.pt`, `0` active runtimes verified — run-ledger attestations external to the JSON bundle).  
**Full Report:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-REPORT.md)  
**Pre-Registered Design:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-DESIGN.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-DESIGN.md) (`sha256 = 5304677e4bbf8e6cd2881aea7ee6b51055f3da76ac28e79ac96400d1fe2b905c`)  
**Pre-Run Amendment 1:** [`olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-AMENDMENT-1.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-AMENDMENT-1.md) (`sha256 = 271cabf1fa1c3e8a98eafd9c8f16fca765c92f78bf5bd914873087f15433e506`)  
**Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`

## Pre-Registered Verdicts (Fail-Closed Gate Honored)

- **Global Gate `C1.0`:** **`FAIL`** — sole failed sub-clause: **`C1.0.3` `anchor_all_ok = false`** (`L24` `median(dose/pmn) = 0.3894` vs. expected `0.2220` [`+75.4%`]; `L27` `0.3450` vs. `0.1642` [`+110.1%`]; `L18` `0.4713` vs. `0.4809` [`-2.0%`, OK]). All five Regime-D `"all"` linearity clauses inside `C1.0.3` passed cleanly (`0/1,728` capped, `0` RMSN > 20%, `5` RMSN > 12%, `fde_p95 = 0.1853 <= 0.30`, `2` FDE > 0.45, `0` FDE > 0.75).
- **Primary Claims (`C1.1`, `C1.2`, `C1.3`, `C1.4`):** **`BLOCKED_BY_C1.0`**. Under the fail-closed rule, no confirmatory claim is made on any primary hypothesis.
- **Secondary Non-Gating `L12_linear_check`:** **`FAIL`** (`15/192` traces > 9% RMSN shrink, `max = 0.1338`, `median frac_err = 0.2396 > 0.12`).

Post-hoc root-cause autopsy of `C1.0.3` `anchor_all_ok` and explicitly non-confirmatory exploratory readouts on the blocked metrics are documented separately in [`OLMO3-TRANSPORT-C1-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-REPORT.md).

## Bundle Contents & Provenance

- `c1_receipt.json.gz` (`620,434` bytes; `.gz sha256 = d2ae6ec0367ae08a1dbbbf9a8fdee7fe319b08a29ada33969c5897d5e7b1101a`, uncompressed `3,375,435` bytes, `sha256 = 2219dd1bf3291bdd11d2da94794d7235d569b57c8d6d0a87808cd8ac9a62db98`)
- `scorecard.json` (`3,226` bytes; `sha256 = 55cf65d6ac79cf70859ea053093b05a4535e2ce57c966b2c9f402619a155a5e5`)
- `SCORECARD.md` (`3,080` bytes; `sha256 = 75917d0e1ac72da58c2f6249058c4acc6986e25379f9d9e3330e04af0f2a714a`)

## Rescore It Yourself (Python 3.10+ Standard Library Only)

From the repository root (`cd assistant-basin`), using **only** the Python 3.10+ standard library (`math.fsum` deterministic across Python 3.10–3.14):

```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_c1.py \
  --receipt results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz \
  --constants olmo3_virtual_and_highway/c1_prereg_constants.json \
  --design olmo3_virtual_and_highway/OLMO3-TRANSPORT-C1-DESIGN.md \
  --out-json /tmp/olmo3_c1_scorecard.json \
  --out-md /tmp/olmo3_c1_SCORECARD.md
cmp /tmp/olmo3_c1_scorecard.json results/vesper-olmo3-7b-stage-c1/scorecard.json
cmp /tmp/olmo3_c1_SCORECARD.md results/vesper-olmo3-7b-stage-c1/SCORECARD.md
```

Both `cmp` commands print nothing and exit `0` when reproduced byte-for-byte.
