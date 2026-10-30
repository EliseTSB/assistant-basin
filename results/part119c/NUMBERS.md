# Part 119c numbers

Path 2 row summaries and the supplemental checks below are recomputed by `fig119c.py` from the JSONL rows. Gates and registered predictions are computed by `score119c.py`, which `fig119c.py` calls. Shares use `(nI / nPair)^2` at positions with `nPair > 0`; each row is the median of its eligible positions, then the summary is the median over rows. Position medians for nI and nPair use all recorded positions. The nPair-below-one share uses all recorded positions.

## Inventory and gates

| Quantity | Recomputed value | Source file and key |
| --- | ---: | --- |
| Selected row keys | 312 total; 156 real and 156 control | Both interaction JSONL files: unique row keys from `$.kind`, `$.first`, `$.second` |
| Records in each file | bf16 1560; float32 1560; 312 distinct row keys × 5 doses per file | Both interaction JSONL files: record count; `$.kind`, `$.first`, `$.second`, `$.dose` |
| Doses | 0.05, 0.1, 0.2, 0.4, 0.8 | Both interaction JSONL files: unique `$.dose` values |
| Gate 1, bf16 against Part 119b | passed; median relative nPair difference 0 | `results/part119c/interaction-bf16.jsonl` and `results/part119b/interaction.jsonl`: `$.kind`, `$.first`, `$.second`, `$.dose`, `$.positions[].pos`, `$.positions[].nPair` |
| Gate 2, float32 against bf16 at 0.8 | passed; median relative nPair difference 0.00350857; all 0.8 nPair values identical by position: False | `results/part119c/interaction-float32.jsonl` and `interaction-bf16.jsonl`: `$.model_dtype`, `$.model_parameter_dtype`, `$.kind`, `$.first`, `$.second`, `$.dose`, `$.positions[].pos`, `$.positions[].nPair` |

Gate 1 also checks the 0.2, 0.4, and 0.8 doses separately. Gate 2 requires the recorded loaded-parameter dtype evidence and a non-identical 0.8 nPair comparison. The recorded dtype strings are in the JSONL keys named in the source column.

## Per-dose summaries

| Dtype | Dose | Class | Rows | Positions | nPair > 0 | nPair < 1 | nPair < 1 share | Median nI | Median nPair | Median nA | Median n2 | Position median share | Row median share | Source file and key |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 | 0.05 | real | 156 | 2056 | 2056 | 1115 | 0.542315 | 0.23457 | 0.948695 | 0.67556 | 0.5942 | 0.0613937 | 0.0734008 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.05 | control | 156 | 2056 | 2056 | 1240 | 0.603113 | 0.22915 | 0.88738 | 0.663565 | 0.549665 | 0.0663186 | 0.0822935 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.1 | real | 156 | 2056 | 2056 | 99 | 0.0481518 | 0.355745 | 2.01119 | 1.40429 | 1.22349 | 0.0339261 | 0.0397493 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.1 | control | 156 | 2056 | 2056 | 146 | 0.0710117 | 0.33602 | 1.84578 | 1.38271 | 1.07628 | 0.0354195 | 0.0422338 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.2 | real | 156 | 2056 | 2056 | 0 | 0 | 1.80373 | 5.02565 | 3.3449 | 2.96421 | 0.132036 | 0.161635 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.2 | control | 156 | 2056 | 2056 | 0 | 0 | 1.50924 | 4.42893 | 3.29413 | 2.33348 | 0.121658 | 0.141404 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.4 | real | 156 | 2056 | 2056 | 0 | 0 | 6.03789 | 11.2024 | 7.79598 | 7.31722 | 0.264667 | 0.30449 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.4 | control | 156 | 2056 | 2056 | 0 | 0 | 5.24774 | 9.91861 | 7.65353 | 5.58859 | 0.275397 | 0.318456 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.8 | real | 156 | 2056 | 2056 | 0 | 0 | 14.0972 | 22.6598 | 15.4939 | 17.3492 | 0.402591 | 0.440626 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.8 | control | 156 | 2056 | 2056 | 0 | 0 | 11.3783 | 19.9574 | 15.376 | 11.6667 | 0.328931 | 0.35077 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.05 | real | 156 | 2056 | 2056 | 1174 | 0.571012 | 0.057415 | 0.929435 | 0.656395 | 0.566525 | 0.00389181 | 0.00454675 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.05 | control | 156 | 2056 | 2056 | 1277 | 0.621109 | 0.053135 | 0.859175 | 0.64841 | 0.516935 | 0.00378297 | 0.00460773 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.1 | real | 156 | 2056 | 2056 | 102 | 0.0496109 | 0.25477 | 1.99267 | 1.38294 | 1.20473 | 0.0177105 | 0.0200938 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.1 | control | 156 | 2056 | 2056 | 160 | 0.077821 | 0.23108 | 1.83853 | 1.37528 | 1.05815 | 0.0172167 | 0.0208125 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.2 | real | 156 | 2056 | 2056 | 0 | 0 | 1.77989 | 5.0188 | 3.34005 | 2.94399 | 0.128422 | 0.153414 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.2 | control | 156 | 2056 | 2056 | 0 | 0 | 1.47989 | 4.4314 | 3.27849 | 2.33485 | 0.11684 | 0.139143 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.4 | real | 156 | 2056 | 2056 | 0 | 0 | 6.02178 | 11.2359 | 7.84258 | 7.30434 | 0.268405 | 0.29487 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.4 | control | 156 | 2056 | 2056 | 0 | 0 | 5.22637 | 9.93161 | 7.66327 | 5.62552 | 0.274635 | 0.322405 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.8 | real | 156 | 2056 | 2056 | 0 | 0 | 14.1135 | 22.5832 | 15.4882 | 17.3759 | 0.402661 | 0.443145 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.8 | control | 156 | 2056 | 2056 | 0 | 0 | 11.3819 | 19.9995 | 15.4216 | 11.704 | 0.327021 | 0.355586 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |

## Dtype comparison

| Dose | Class | bf16 row share | float32 row share | float32 minus bf16 | bf16 nPair < 1 share | float32 nPair < 1 share | Source file and key |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | real | 0.0734008 | 0.00454675 | -0.0688541 | 0.542315 | 0.571012 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.05 | control | 0.0822935 | 0.00460773 | -0.0776858 | 0.603113 | 0.621109 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.1 | real | 0.0397493 | 0.0200938 | -0.0196555 | 0.0481518 | 0.0496109 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.1 | control | 0.0422338 | 0.0208125 | -0.0214213 | 0.0710117 | 0.077821 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.2 | real | 0.161635 | 0.153414 | -0.00822048 | 0 | 0 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.2 | control | 0.141404 | 0.139143 | -0.00226149 | 0 | 0 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.4 | real | 0.30449 | 0.29487 | -0.00961959 | 0 | 0 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.4 | control | 0.318456 | 0.322405 | 0.00394918 | 0 | 0 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.8 | real | 0.440626 | 0.443145 | 0.00251913 | 0 | 0 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 0.8 | control | 0.35077 | 0.355586 | 0.00481643 | 0 | 0 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |

## Real against control

Factor is the larger row share divided by the smaller row share.

| Dtype | Dose | Real share | Control share | Larger-to-smaller factor | Source file and key |
| --- | ---: | ---: | ---: | ---: | --- |
| bf16 | 0.05 | 0.0734008 | 0.0822935 | 1.12115 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.1 | 0.0397493 | 0.0422338 | 1.06251 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.2 | 0.161635 | 0.141404 | 1.14307 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.4 | 0.30449 | 0.318456 | 1.04587 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| bf16 | 0.8 | 0.440626 | 0.35077 | 1.25617 | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.05 | 0.00454675 | 0.00460773 | 1.01341 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.1 | 0.0200938 | 0.0208125 | 1.03577 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.2 | 0.153414 | 0.139143 | 1.10257 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.4 | 0.29487 | 0.322405 | 1.09338 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| float32 | 0.8 | 0.443145 | 0.355586 | 1.24624 | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |

## Registered predictions

| Prediction | Recomputed observed values | Verdict | Source file and key |
| --- | --- | --- | --- |
| 119c.1 | bf16 share 0.161635; float32 share 0.153414; difference float32 minus bf16 -0.00822048; product-weighted 95% interval -0.0212975 to 0.0125089; sum-weighted sensitivity interval -0.0143429 to 0.00960265; 2000 draws, seed 11902, 39 units, 156 paired rows | held | `results/part119c/interaction-bf16.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair`; `part119/score119.py`: `BOOTSTRAP_SEED`; `part119c/score119c.py`: `_weighted_difference_ci`, `predictions` |
| 119c.2 | float32 real share at 0.05 0.00454675; at 0.1 0.0200938 | held | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 119c.3 | float32 real share at 0.05 0.00454675; at 0.2 0.153414; log-log slope 2.53823 | held | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |
| 119c.4 | dose 0.05: real 0.00454675, control 0.00460773, factor 1.01341; dose 0.1: real 0.0200938, control 0.0208125, factor 1.03577; dose 0.2: real 0.153414, control 0.139143, factor 1.10257 | held | `results/part119c/interaction-float32.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, `$.positions[].nPair` |

## Dose-growth ratio

1.90873

Source: `results/part119c/interaction-float32.jsonl`, `$.dose`, `$.kind`, `$.first`, `$.second`, `$.positions[].nI`, and `$.positions[].nPair`; the value is the float32 real row-median-share ratio from 0.1 to 0.2 divided by the squared dose ratio.

## Resolved-only low-dose check

Resolved positions require nA, n2, and nPair each to be at least 1.0, matching Part 119b. The registered reading uses every position with nPair > 0. Coverage counts resolved positions and rows with at least one resolved position.

| Dtype | Dose | Class | Registered all-position row median | Resolved-only row median | Resolved rows / all rows | Resolved positions / all positions | Position coverage |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 0.05 | real | 0.0734008 | 0.0234421 | 40/156 | 96/2056 | 0.0466926 |
| bf16 | 0.05 | control | 0.0822935 | 0.0189196 | 28/156 | 56/2056 | 0.0272374 |
| bf16 | 0.1 | real | 0.0397493 | 0.0359756 | 146/156 | 1115/2056 | 0.542315 |
| bf16 | 0.1 | control | 0.0422338 | 0.0326448 | 144/156 | 967/2056 | 0.470331 |
| float32 | 0.05 | real | 0.00454675 | 0.0057915 | 37/156 | 77/2056 | 0.0374514 |
| float32 | 0.05 | control | 0.00460773 | 0.00349359 | 20/156 | 42/2056 | 0.020428 |
| float32 | 0.1 | real | 0.0200938 | 0.020172 | 146/156 | 1072/2056 | 0.521401 |
| float32 | 0.1 | control | 0.0208125 | 0.0170031 | 143/156 | 931/2056 | 0.452821 |

Float32 resolved real/control factors: dose 0.05 1.65775; dose 0.1 1.18637

## Part 119b full-sample comparison

The Part 119b column is recomputed from all real rows with the same all-position row-median rule.

| Dose | 119c bf16 subset | 119c float32 subset | 119b all real rows | 119b real rows |
| ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.161635 | 0.153414 | 0.176187 | 1482 |
| 0.4 | 0.30449 | 0.29487 | 0.302333 | 1482 |
| 0.8 | 0.440626 | 0.443145 | 0.467962 | 1482 |

## Dtype gap and nI ratios

The nPair gap is the median, over rows, of each row's median absolute position-level relative difference, with bf16 as denominator. The nI ratio is the ratio of pooled position medians, bf16 divided by float32.

| Dose | Median relative nPair gap | Real bf16 / float32 median nI | Real nI excess | Control bf16 / float32 median nI | Control nI excess |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.0463478 | 4.08552 | 0.177155 | 4.3126 | 0.176015 |
| 0.1 | 0.0243146 | 1.39634 | 0.100975 | 1.45413 | 0.10494 |
| 0.2 | 0.0116904 | 1.01339 | 0.023835 | 1.01983 | 0.029345 |
| 0.4 | 0.00698318 | 1.00268 | 0.016115 | 1.00409 | 0.02137 |
| 0.8 | 0.00350857 | 0.998843 | -0.016325 | 0.999684 | -0.0036 |

At dose 0.8, dtype-specific T sets differ at 1675 of 4112 matched positions; the median symmetric-difference size is 0 tokens and the maximum is 6.

## Pooled ratio, local slopes, and low-dose intervals

Pooled ratio is sum(nI²) / sum(nPair²) over float32 real positions. The row-median share is the registered summary. Bootstrap intervals are product-weighted unit-bootstrap intervals for the float32 real median row share, using the scorer's seed.

| Dose | Pooled ratio | Float32 real row-median share |
| ---: | ---: | ---: |
| 0.05 | 0.00459321 | 0.00454675 |
| 0.1 | 0.0376616 | 0.0200938 |
| 0.2 | 0.229596 | 0.153414 |

| Dose interval | Local log-log slope |
| --- | ---: |
| 0.05 to 0.1 | 2.14384 |
| 0.1 to 0.2 | 2.93261 |
| 0.05 to 0.2 | 2.53823 |

| Dose | Float32 real row-median share | Product-weighted 95% unit-bootstrap interval | Draws | Seed | Units | Rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.00454675 | 0.00281645 to 0.00623154 | 2000 | 11902 | 39 | 156 |
| 0.1 | 0.0200938 | 0.0120856 to 0.0310969 | 2000 | 11902 | 39 | 156 |
