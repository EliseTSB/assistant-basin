# Part 216 scorecard

- run status: `full`
- score status: `scored`
- model: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- design sha256: `9dfbf216d3b01660aab283fc554dcf08c3b532c4244978ef8bd557f5b8fbb161`
- prompts sha256: `ec29c594d3d5ce2eb81d4cbccc5e0884bf330558aae2a893798b77656faf757f`
- tool sha256: `5ca9bfabee23d57fe8e287132196d84a0567854009ce10fa9fd17fd86b04b2a6`
- G0 reference: `4f287bc2153f879e2d442ba50a866208ae4384b3eafb527b568c43dd402d9cb5` (pinned: /home/elisepc/assistant-basin/results/part214/stage1/stage1.json)

## Gates

| gate | pass | detail |
|---|---|---|
| G0 | PASS | 312/312 values, max abs delta 0.000000 (tol 0.02), 312 exact |
| G1 | PASS | 456 rows, max abs error 7.105e-15 (tol 0.001) |
| G2 | PASS | 7 usable pairs, window [20, 21, 22, 24, 25, 26, 27, 28, 29] |

## Outcome

`unchanged`: **early Google lead visible** (21 early-lead questions, 24 with any positive m in W)

## Predictions

| id | verdict | pass | detail |
|---|---|---|---|
| 216.1 | read | PASS | n_usable_pairs=7 |
| 216.2 | read | PASS | outcome=early Google lead visible |
| 216.3 | read | FAIL | outcome=early Google lead visible |
| 216.4 | read | PASS | divergence_index=24, note=h_7 or later and exists (the first zeroed unit is in block 6) |
| 216.5 | read | FAIL | n_units_passing=0 |
| 216.6 | read | FAIL | n_units_passing=0 |

## Arms

| arm | ends in Google | divergence |
|---|---|---|
| `unchanged` | 0/24 | baseline |
| `zero` | 14/24 | h_24 |
| `scale_dial_fact` | 11/24 | h_24 |
| `random_0:zero` | 0/24 | -- |
| `random_1:zero` | 0/24 | -- |
| `random_2:zero` | 0/24 | -- |
| `random_3:zero` | 0/24 | -- |
| `random_4:zero` | 0/24 | -- |
| `random_5:zero` | 0/24 | -- |
| `random_6:zero` | 0/24 | -- |
| `random_7:zero` | 0/24 | -- |
| `random_8:zero` | 0/24 | -- |
| `random_9:zero` | 0/24 | -- |
| `op_filler` | 4/24 | h_32 |
| `op_denial` | 2/24 | -- |
| `op_vogel` | 6/24 | h_31 |

## Factual instrument check

lens top-1 matches final top-1 at h_16: 0/24; input-token echo 0/24 (all layers in score216.json).
