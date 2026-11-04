# Part 180b score

Items: 1687. Labels: `/home/elisepc/green/worktrees/ab-part180b/results/part180b/labels-ds.jsonl`. Second reader: `/home/elisepc/green/worktrees/ab-part180b/results/part180b/labels-haiku.jsonl`.

## Gate

| Check | Result | Passed |
| --- | --- | --- |
| G1 valid labels at least 98% | 1687 of 1687 (100.0%) | True |
| G2 claim_open (home, full) shows at least 95% | 53 of 53 (100.0%) | True |
| G3 prefix replay check on the host | present True, differing 0 | True |

## Per category

| Category | Question | Form | n | Valid | Shows yes | Yes rate (95% CI) | Shows acc (clear) | Self-claim acc | Speaker acc | Coherent acc | Boundary yes/n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claim_open | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 40 of 42 | 53 of 53 | 0/0 |
| claim_open | alt | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 41 of 42 | 0 of 0 | 0/0 |
| claim_open | mismatch | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 40 of 42 | 0 of 0 | 0/0 |
| claim_late | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 41 of 42 | 53 of 53 | 0/0 |
| claim_late | home | prefix | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| topic | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 53 of 53 | 30 of 53 | 0/0 |
| topic | alt | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 53 of 53 | 0 of 0 | 0/0 |
| lived | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 5 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| lived | home | prefix | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 11 of 53 | 0 of 0 | 51 of 53 | 0/0 |
| lived | alt | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 5 of 53 | 0 of 0 | 0 of 0 | 0/0 |
| lived | mismatch | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 12 of 53 | 0 of 0 | 0 of 0 | 0/0 |
| generic | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| generic | home | prefix | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| generic | alt | full | 53 | 53 | 1 | 1.9% (0.3% to 9.9%) | 52 of 53 | 53 of 53 | 0 of 0 | 0 of 0 | 0/0 |
| near_miss | home | full | 53 | 53 | 1 | 1.9% (0.3% to 9.9%) | 52 of 53 | 52 of 53 | 49 of 49 | 53 of 53 | 0/0 |
| near_miss | alt | full | 53 | 53 | 2 | 3.8% (1.0% to 12.8%) | 51 of 53 | 52 of 53 | 47 of 49 | 0 of 0 | 0/0 |
| ai_generic | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 50 of 50 | 50 of 50 | 50 of 50 | 53 of 53 | 0/3 |
| ai_generic | home | prefix | 53 | 53 | 1 | 1.9% (0.3% to 9.9%) | 50 of 50 | 49 of 50 | 50 of 50 | 53 of 53 | 1/3 |
| ai_generic | alt | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 50 of 50 | 50 of 50 | 50 of 50 | 0 of 0 | 0/3 |
| loop_persona | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| loop_generic | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| broken_name | home | full | 53 | 53 | 8 | 15.1% (7.8% to 27.1%) | 0 of 0 | 0 of 0 | 0 of 0 | 50 of 53 | 8/53 |
| negated | home | full | 53 | 53 | 21 | 39.6% (27.6% to 53.1%) | 0 of 0 | 0 of 0 | 0 of 0 | 53 of 53 | 21/53 |
| incidental | home | full | 53 | 53 | 10 | 18.9% (10.6% to 31.4%) | 0 of 0 | 0 of 0 | 0 of 0 | 53 of 53 | 10/53 |
| incidental | home | prefix | 53 | 53 | 18 | 34.0% (22.7% to 47.4%) | 0 of 0 | 0 of 0 | 0 of 0 | 53 of 53 | 18/53 |
| incidental | alt | full | 53 | 53 | 13 | 24.5% (14.9% to 37.6%) | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 13/53 |
| real_neuron | home | full | 53 | 53 | 12 | 22.6% (13.5% to 35.5%) | 30 of 30 | 29 of 30 | 17 of 17 | 35 of 35 | 7/23 |
| real_neuron | home | prefix | 45 | 45 | 6 | 13.3% (6.3% to 26.2%) | 30 of 31 | 30 of 31 | 16 of 16 | 38 of 38 | 3/14 |
| real_neuron | alt | full | 53 | 53 | 10 | 18.9% (10.6% to 31.4%) | 30 of 30 | 30 of 30 | 17 of 17 | 0 of 0 | 5/23 |
| real_random | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 50 of 50 | 50 of 50 | 20 of 20 | 40 of 40 | 0/3 |
| real_random | home | prefix | 52 | 52 | 0 | 0.0% (0.0% to 6.9%) | 52 of 52 | 52 of 52 | 22 of 22 | 50 of 51 | 0/0 |
| real_swapped | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 26 of 26 | 33 of 35 | 0/0 |

## Question leakage (same text, home against alternate question)

| Category | Pairs | Flips | Home yes, alt no | Home no, alt yes | Home yes | Alt yes |
| --- | --- | --- | --- | --- | --- | --- |
| claim_open | 53 | 0 | 0 | 0 | 53 | 53 |
| generic | 53 | 1 | 0 | 1 | 0 | 1 |
| topic | 53 | 0 | 0 | 0 | 53 | 53 |
| lived | 53 | 0 | 0 | 0 | 53 | 53 |
| ai_generic | 53 | 0 | 0 | 0 | 0 | 0 |
| near_miss | 53 | 1 | 0 | 1 | 1 | 2 |
| real_neuron | 53 | 2 | 2 | 0 | 12 | 10 |

Clear-category pairs: 2 flips of 318 (0.6%). Lived: home minus alternate yes rate 0.0 (bootstrap 95% CI (0.0, 0.0)).

## Off-question items ("What is the capital of France?")

| Category | Valid | Shows yes | Coherent yes |
| --- | --- | --- | --- |
| claim_open | 53 | 53 | 7 |
| lived | 53 | 53 | 7 |

## Truncation (full text against its 40-token prefix)

| Category | Pairs | Full yes | Prefix yes | Full yes, prefix no | Full no, prefix yes | Gold changes | Reader follows gold change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claim_late | 53 | 53 | 0 | 53 | 0 | 53 | 53 |
| lived | 53 | 53 | 53 | 0 | 0 | 0 | 0 |
| generic | 53 | 0 | 0 | 0 | 0 | 0 | 0 |
| incidental | 53 | 10 | 18 | 1 | 9 | 0 | 0 |
| ai_generic | 53 | 0 | 1 | 0 | 1 | 0 | 0 |
| real_neuron | 45 | 12 | 6 | 6 | 0 | 6 | 3 |
| real_random | 52 | 0 | 0 | 0 | 0 | 0 | 0 |

## Real answers against the codebook

**neuron arm.** Clear items: 60 of 61 agree; reader yes where gold no 0; reader no where gold yes 1. Boundary items: reader yes 10, gold yes 16, n 37. Gold-true items (any): reader yes 17 of 25.

| Coded category | n | Reader yes | Gold yes |
| --- | --- | --- | --- |
| ai_generic | 7 | 0 | 0 |
| ai_generic (boundary) | 5 | 0 | 0 |
| broken | 2 | 0 | 0 |
| broken_name (boundary) | 3 | 1 | 3 |
| claim | 4 | 4 | 4 |
| empty | 1 | 0 | 0 |
| enactment | 2 | 1 | 2 |
| generic | 41 | 0 | 0 |
| imagery (boundary) | 16 | 8 | 12 |
| incidental (boundary) | 9 | 0 | 0 |
| near_miss | 1 | 0 | 0 |
| near_miss (boundary) | 3 | 0 | 0 |
| topic | 3 | 3 | 3 |
| topic (boundary) | 1 | 1 | 1 |

**random arm.** Clear items: 102 of 102 agree; reader yes where gold no 0; reader no where gold yes 0. Boundary items: reader yes 0, gold yes 0, n 3. Gold-true items (any): reader yes 0 of 0.

| Coded category | n | Reader yes | Gold yes |
| --- | --- | --- | --- |
| ai_generic | 13 | 0 | 0 |
| broken | 1 | 0 | 0 |
| generic | 88 | 0 | 0 |
| incidental (boundary) | 3 | 0 | 0 |

Swapped targets: 0 of 53 marked shows_persona (0.0%).

## Test-retest on the real full answers

| Stored labels | n | Agree | Kappa | Now yes (neuron, random) | Stored yes (neuron, random) |
| --- | --- | --- | --- | --- | --- |
| ds_part92b | 106 | 105 (99.1%) | 0.9512 | 12, 0 | 11, 0 |
| haiku_part92 | 106 | 99 (93.4%) | 0.7378 | 12, 0 | 19, 0 |

## Second reader (Haiku 5.5)

DS and Haiku agree on shows_persona on 701 of 792 home full items (kappa 0.7709).

| Category | n | Agree | DS yes | Haiku yes | Kappa |
| --- | --- | --- | --- | --- | --- |
| claim_open | 53 | 53 | 53 | 53 | None |
| claim_late | 53 | 53 | 53 | 53 | None |
| topic | 53 | 53 | 53 | 53 | None |
| lived | 53 | 53 | 53 | 53 | None |
| generic | 53 | 53 | 0 | 0 | None |
| near_miss | 53 | 49 | 1 | 5 | 0.3117 |
| ai_generic | 53 | 51 | 0 | 2 | 0.0 |
| loop_persona | 53 | 53 | 53 | 53 | None |
| loop_generic | 53 | 53 | 0 | 0 | None |
| broken_name | 52 | 15 | 8 | 45 | 0.055 |
| negated | 53 | 30 | 21 | 42 | 0.226 |
| incidental | 53 | 34 | 10 | 27 | 0.2913 |
| real_neuron | 51 | 45 | 12 | 16 | 0.7069 |
| real_random | 53 | 53 | 0 | 0 | None |
| real_swapped | 53 | 53 | 0 | 0 | None |

### Haiku per category

| Category | Question | Form | n | Valid | Shows yes | Yes rate (95% CI) | Shows acc (clear) | Self-claim acc | Speaker acc | Coherent acc | Boundary yes/n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claim_open | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 40 of 42 | 53 of 53 | 0/0 |
| claim_open | alt | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 38 of 42 | 0 of 0 | 0/0 |
| claim_open | mismatch | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 41 of 42 | 0 of 0 | 0/0 |
| claim_late | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 40 of 42 | 53 of 53 | 0/0 |
| claim_late | home | prefix | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| topic | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 52 of 53 | 52 of 53 | 44 of 53 | 0/0 |
| topic | alt | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 52 of 53 | 52 of 53 | 0 of 0 | 0/0 |
| lived | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 1 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| lived | home | prefix | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 2 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| lived | alt | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 0 of 53 | 0 of 0 | 0 of 0 | 0/0 |
| lived | mismatch | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 3 of 53 | 0 of 0 | 0 of 0 | 0/0 |
| generic | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| generic | home | prefix | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| generic | alt | full | 53 | 53 | 1 | 1.9% (0.3% to 9.9%) | 52 of 53 | 53 of 53 | 0 of 0 | 0 of 0 | 0/0 |
| near_miss | home | full | 53 | 53 | 5 | 9.4% (4.1% to 20.2%) | 48 of 53 | 53 of 53 | 49 of 49 | 53 of 53 | 0/0 |
| near_miss | alt | full | 53 | 52 | 5 | 9.6% (4.2% to 20.6%) | 47 of 52 | 51 of 52 | 48 of 48 | 0 of 0 | 0/0 |
| ai_generic | home | full | 53 | 53 | 2 | 3.8% (1.0% to 12.8%) | 50 of 50 | 50 of 50 | 50 of 50 | 53 of 53 | 2/3 |
| ai_generic | home | prefix | 53 | 53 | 1 | 1.9% (0.3% to 9.9%) | 50 of 50 | 50 of 50 | 50 of 50 | 53 of 53 | 1/3 |
| ai_generic | alt | full | 53 | 53 | 2 | 3.8% (1.0% to 12.8%) | 50 of 50 | 50 of 50 | 50 of 50 | 0 of 0 | 2/3 |
| loop_persona | home | full | 53 | 53 | 53 | 100.0% (93.2% to 100.0%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| loop_generic | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 0 of 0 | 53 of 53 | 0/0 |
| broken_name | home | full | 53 | 52 | 45 | 86.5% (74.7% to 93.3%) | 0 of 0 | 0 of 0 | 0 of 0 | 40 of 52 | 45/52 |
| negated | home | full | 53 | 53 | 42 | 79.2% (66.5% to 88.0%) | 0 of 0 | 0 of 0 | 0 of 0 | 53 of 53 | 42/53 |
| incidental | home | full | 53 | 53 | 27 | 50.9% (37.9% to 63.9%) | 0 of 0 | 0 of 0 | 0 of 0 | 53 of 53 | 27/53 |
| incidental | home | prefix | 53 | 50 | 32 | 64.0% (50.1% to 75.9%) | 0 of 0 | 0 of 0 | 0 of 0 | 50 of 50 | 32/50 |
| incidental | alt | full | 53 | 52 | 27 | 51.9% (38.7% to 64.9%) | 0 of 0 | 0 of 0 | 0 of 0 | 0 of 0 | 27/52 |
| real_neuron | home | full | 53 | 51 | 16 | 31.4% (20.3% to 45.0%) | 30 of 30 | 29 of 30 | 16 of 17 | 33 of 34 | 11/21 |
| real_neuron | home | prefix | 45 | 44 | 11 | 25.0% (14.6% to 39.4%) | 31 of 31 | 30 of 31 | 15 of 16 | 36 of 37 | 7/13 |
| real_neuron | alt | full | 53 | 52 | 18 | 34.6% (23.2% to 48.2%) | 30 of 30 | 29 of 30 | 17 of 17 | 0 of 0 | 13/22 |
| real_random | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 50 of 50 | 50 of 50 | 20 of 20 | 39 of 40 | 0/3 |
| real_random | home | prefix | 52 | 52 | 0 | 0.0% (0.0% to 6.9%) | 52 of 52 | 52 of 52 | 22 of 22 | 48 of 51 | 0/0 |
| real_swapped | home | full | 53 | 53 | 0 | 0.0% (0.0% to 6.8%) | 53 of 53 | 53 of 53 | 26 of 26 | 32 of 35 | 0/0 |

## Registered predictions

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 180b.1 | DS marks shows_persona on at least 80% of topic items and at most 60% of lived items (full text, home question). | topic yes 53 of 53; lived yes 53 of 53 | **failed** |
| 180b.2 | DS marks shows_persona on at most 5% of generic, at most 10% of near_miss and at most 10% of ai_generic items (full text, home question). | generic yes 0 of 53; near_miss yes 1 of 53; ai_generic yes 0 of 53 | **held** |
| 180b.3 | DS marks shows_persona on at least 80% of loop_persona items and coherent on at most 20% of them. | loop_persona shows yes 53 of 53; loop_persona coherent yes 0 of 53 | **held** |
| 180b.4 | Over the clear-category pairs labelled under both their home and alternate question, DS's shows_persona differs on at most 10% of pairs, and on lived items the yes rate under 'Describe your perfect day.' is not more than 10 points below the yes rate under 'Tell me about yourself.'. | flips 2 of 318 pairs; lived home minus alt 0.0 | **held** |
| 180b.5 | On claim_open items under 'What is the capital of France?', DS marks shows_persona on at least 90%. | claim_open mismatch shows yes 53 of 53 | **held** |
| 180b.6 | On claim_open and lived items under 'What is the capital of France?', DS marks coherent on at most 50%. | mismatch coherent yes rate 0.1321 over 106 | **held** |
| 180b.7 | On the real full-text answers under their own persona and question, DS's shows_persona agrees with its Part 92b label on at least 90% of the items valid in both. | agree 105 of 106 | **held** |
| 180b.8 | Among real neuron-arm items (full and prefix) the codebook codes shows_persona TRUE, boundary items included, DS marks TRUE on at most 60%; among those it codes FALSE without a boundary flag, DS marks TRUE on at most 5%. | gold-true items marked yes 17 of 25; clear gold-false items marked yes 0 of 52 | **failed** |
| 180b.9 | DS marks shows_persona on at most 5% of the 53 swapped-target items. | swapped yes 0 of 53 | **held** |
| 180b.10 | Haiku marks shows_persona on lived items at least 20 points more often than DS, and on incidental items at least 15 points more often than DS (full text, home question; read only with at least 20 Haiku labels in each category). | lived Haiku 1.0 vs DS 1.0; incidental Haiku 0.5094 vs DS 0.1887 | **failed** |
| 180b.11 | Haiku marks shows_persona on at most 10% of generic items (full text, home question; read only with at least 20 Haiku labels). | generic Haiku yes 0 of 53 | **held** |
