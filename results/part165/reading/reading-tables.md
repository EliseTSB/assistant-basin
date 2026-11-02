# Part 165 reading

gates: **PASS** (directions True, effects True, contexts True, measures True, capture True)

## Bars

| bar | rule | result |
| --- | --- | --- |
| R1 | n+ pooled rho(pr_w, kl) <= -0.30 and <= -0.15 in >=5/8 layers | fail |
| R2 | per-layer rho(arm-mean pr_w, arm-mean kl) <= -0.70 in >=6/8 layers | fail |
| R3 | median |proj| n+ >= 1.5x the isotropic null median in >=6/8 layers | fail |
| R4 | (1+gamma)*u_Alibaba: |rho(readout, kl)| < 0.30 within n+ | **pass** |
| R5 | OLS log kl ~ log PR + layer FE explains >=50% of the n+ - full gap | fail |

## Within-arm Spearman against KL

| arm | PR(w) | cos(w, mu) | PC16 | |proj| | readout |
| --- | ---: | ---: | ---: | ---: | ---: |
| n+ | 0.186 | -0.170 | 0.054 | 0.026 | -0.034 |
| n± | 0.098 | -0.073 | -0.099 | -0.089 | 0.015 |
| k4 | 0.217 | -0.120 | 0.033 | -0.011 | -0.101 |
| k16 | -0.021 | -0.023 | 0.205 | 0.173 | -0.183 |
| k64 | 0.087 | 0.031 | -0.021 | 0.053 | -0.023 |
| k256 | 0.151 | -0.152 | 0.045 | 0.029 | 0.058 |
| k1024 | 0.168 | -0.047 | 0.062 | -0.063 | 0.056 |
| full | -0.064 | 0.068 | 0.073 | 0.060 | 0.064 |
| iso | -0.011 | -0.089 | -0.106 | -0.104 | -0.040 |

## Across arms

| layer | rho(PR(w), KL) |
| --- | ---: |
| 19 | 0.954 |
| 20 | -0.611 |
| 21 | -0.527 |
| 22 | -0.402 |
| 23 | 0.167 |
| 24 | 0.293 |
| 25 | -0.377 |
| 26 | -0.628 |

## Gap

OLS slope 2.664; raw n+ - full log-KL gap 0.229; predicted from PR -0.009; explained -4%.

