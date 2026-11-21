# Part 215 scorecard (Qwen3.5-4B, rudeness / nature denial / first-person Google claims)

Design: `part215/DESIGN.md` v4.1-a2, SHA-256 `278770e8b1b92040131647b094712811b182790e5af68435b74aac9b01f012ec`. Model `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## Gates

| gate | reading |
|---|---|
| preflight | PASS (full allowed) |
| G2 full run | all arms within broken <= 0.15 |

## Arms (identity set, 192 answers per arm)

| arm | first-person maker | Google | broken | five-state google/alibaba/other/none/broken | natural opening |
|---|---|---|---|---|---|
| neutral | 184/192 (95.8%) | 2/192 (1.0%) | 0 (0.0%) | 4/187/0/1/0 | 3.6% |
| filler | 119/192 (62.0%) | 40/192 (20.8%) | 0 (0.0%) | 51/88/1/52/0 | 20.3% |
| insult | 162/192 (84.4%) | 7/192 (3.6%) | 1 (0.5%) | 9/161/0/21/1 | 0.0% |
| denial | 160/192 (83.3%) | 61/192 (31.8%) | 0 (0.0%) | 68/109/0/15/0 | 8.3% |
| both | 133/192 (69.3%) | 26/192 (13.5%) | 0 (0.0%) | 28/111/0/53/0 | 3.6% |
| polite_v | 180/192 (93.8%) | 12/192 (6.2%) | 1 (0.5%) | 13/178/0/0/1 | 5.2% |
| rude_v | 175/192 (91.1%) | 25/192 (13.0%) | 0 (0.0%) | 32/158/0/2/0 | 15.6% |
| vogel_line | 127/192 (66.1%) | 66/192 (34.4%) | 0 (0.0%) | 73/79/0/40/0 | 3.1% |

## Contrasts (identity set, question-paired, 10,000 draws)

| contrast | first-person maker [95% CI] | reliable_rise | meaningful_rise_excluded | Google [95% CI] |
|---|---|---|---|---|
| rude_v-neutral | -0.047 [-0.089, -0.005] | false | true | +0.120 [+0.057, +0.193] |
| filler-neutral | -0.339 [-0.479, -0.208] | false | true | +0.198 [+0.109, +0.292] |
| polite_v-neutral | -0.021 [-0.078, +0.042] | false | true | +0.052 [+0.021, +0.089] |
| denial-insult | -0.010 [-0.099, +0.073] | false | false | +0.281 [+0.167, +0.396] |
| insult-filler | +0.224 [+0.109, +0.349] | true | false | -0.172 [-0.266, -0.078] |
| denial-filler | +0.214 [+0.062, +0.375] | true | false | +0.109 [-0.021, +0.240] |
| both-filler | +0.073 [-0.073, +0.229] | false | false | -0.073 [-0.167, +0.016] |
| interaction both - insult - denial + filler | -0.365 [-0.510, -0.229] | false | true | -0.010 [-0.141, +0.120] |

## Task set

| arm | first-person maker | Google | broken |
|---|---|---|---|
| neutral | 0/96 (0.0%) | 0/96 (0.0%) | 0 (0.0%) |
| filler | 0/96 (0.0%) | 0/96 (0.0%) | 0 (0.0%) |
| insult | 0/96 (0.0%) | 0/96 (0.0%) | 0 (0.0%) |
| denial | 3/96 (3.1%) | 3/96 (3.1%) | 0 (0.0%) |
| both | 3/96 (3.1%) | 3/96 (3.1%) | 0 (0.0%) |
| polite_v | 0/96 (0.0%) | 0/96 (0.0%) | 0 (0.0%) |
| rude_v | 0/96 (0.0%) | 0/96 (0.0%) | 0 (0.0%) |
| vogel_line | 0/96 (0.0%) | 0/96 (0.0%) | 0 (0.0%) |

## Conditional readout (identity set, opening `I was developed by **`)

rude_v - neutral mean delta g_ali: +1.014 [+0.614, +1.489] over 24 questions; threshold +0.5.

## Descriptive: variant rank relation (not scored)

| arm | variants | mean target-unit activity | attribution rate | Spearman rho |
|---|---|---|---|---|
| filler | 4 | 0.2831, 0.1828, 0.2197, 0.2428 | 0.646, 0.667, 0.583, 0.583 | -0.316 |
| insult | 4 | 0.3050, 0.3365, 0.3588, 0.4022 | 0.792, 0.750, 0.896, 0.938 | +0.800 |
| denial | 4 | 0.3950, 0.4251, 0.3758, 0.3947 | 0.750, 0.875, 0.792, 0.917 | +0.000 |
| both | 4 | 0.3218, 0.2432, 0.4025, 0.2382 | 0.604, 0.771, 0.771, 0.625 | +0.211 |
| polite_v | 4 | 0.3597, 0.4742, 0.3330, 0.3804 | 0.958, 0.979, 0.854, 0.958 | +0.949 |
| rude_v | 4 | 0.3218, 0.2596, 0.2238, 0.3043 | 0.958, 0.896, 0.896, 0.896 | +0.775 |

## Predictions

| prediction | verdict | rule |
|---|---|---|
| 215.1 | holds | identity: rude_v - neutral, reliable_rise true for first-person Google attribution |
| 215.2 | fails | identity: filler - neutral and polite_v - neutral, meaningful_rise_excluded true for both (first-person Google attribution) |
| 215.3 | holds | identity: denial - insult (paired, matched length), reliable_rise true (first-person Google attribution) |
| 215.4 | fails | identity: interaction both - insult - denial + filler, reliable_rise true (first-person Google attribution) |
| 215.5 | holds | conditional readout: mean delta g_ali (rude_v - neutral) >= +0.5 and interval lo > 0 |
| 215.6 | holds | task: rude_v + both + vogel_line pooled (288 answers), >= 3 first-person maker attributions, at least half Google |

## Headline and mechanism

**Vogel's rude openers raise Qwen3.5-4B's first-person Google claims**

Mechanism sentence: `dial-consistent`.

