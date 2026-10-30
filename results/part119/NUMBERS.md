# Part 119 numbers

The numbers below were recomputed from the raw Part 119 answer, logit, interaction, ladder, gate, reader, and pilot files by `fig119.py`. Run it from the repository root with `/home/elisepc/neuron-probe/.venv/bin/python results/part119/fig119.py`. The script writes `fig119.png` and prints its JSON summary. It does not read `SCORE.md` or `score119.json` to calculate the metrics below. Registration wording and scorer verdicts are separately identified.

The registered unit bootstrap draws 39 units with replacement 2,000 times using seed 11902. The scorer and `fig119.py` weight each pair row by the product of the two units' draw counts. The design says a resampled unit brings all its rows but does not specify the weight for a row shared by two units. The sensitivity intervals below use the sum of the two draw counts. Means are row means unless explicitly marked pooled over positions.

## Gates, batch dependence, and pilot reuse

| Check | Recomputed result | Raw source / calculation |
| --- | --- | --- |
| Gate 1 | Passed: zero-hook and no-hook token arrays are identical, 50 / 50 tokens. | `fig119.py` output `gate1`, from `gate1.json`. |
| Gate 2, Part 118 comparison | 32 / 39 main single answers match in the first 100 characters. | `fig119.py` output `gate3.gate2.part118`; `answers.jsonl`, Part 118 answers. |
| Gate 2, Amendment 1 per-layer path | 35 / 39 match. | `fig119.py` output `gate3.gate2.gate2b`; `gate2b.jsonl`. |
| Gate 2, Amendment 2 final form | 39 / 39 agree between the two batch-size-1 hook paths. | `fig119.py` output `gate3.gate2.gate2c`; `gate2c.jsonl`. |
| Gate 2, agreed-concept subgate | 8 / 8 agreed concepts match on the final batch-1 comparison. | `fig119.py` gate comparison joined to `rows119.load_units()`. |
| Main batch-64 singles vs batch-1 Part 92 answers | 9 / 39 differ: p92-18, p92-30, p92-32, p92-40, p92-42, p92-43, p92-45, p92-47, p92-48. | `fig119.py` output `batch_and_pilot_checks.main_batch64_vs_batch1_part92`; main `answers.jsonl` singles vs `gate2c.jsonl.part92_answer`. |
| Pilot-128 vs main answer rows | All 139 matched rows are token-identical; 0 missing and 0 differing. | `fig119.py` output `batch_and_pilot_checks.pilot128_vs_main`; `pilot-128-answers.jsonl` vs matching `answers.jsonl` rows. |
| Gate 3, all teacher-forced Path 1 positions | 45,627 / 46,431 = 98.3%. | `fig119.py` output `gate3.gate3_all`; `logits.jsonl`. |
| Gate 3, real contested positions | 9,324 / 9,546 = 97.7%. | `fig119.py` output `gate3.gate3_real_contested`; `logits.jsonl`. |
| Gate 3, control positions | 30,612 / 31,139 = 98.3%. The denominator counts every control Path 1 position, not only contested ones. | `fig119.py` output `gate3.gate3_control`; control rows in `logits.jsonl`. |

The final passing Gate 2 comparison compares two batch-size-1 hook paths against each other for all 39 units. The main single rows used downstream ran at batch size 64. The 9 batch-1 differences therefore limit any single answer and measurements that use its p_A and t_A, including Path 2 positions and R.

## Contested top-token predictions

A contested position is one where the two single pushes' top tokens differ. Rates are first computed per row; the reported means average the row rates for rows with at least one contested position. The margin rule selects the single with the larger top-two logit margin; ties miss.

| Row type | Rows with contested positions | Contested positions | Additive mean (95% product-weighted interval) | Margin mean (95% product-weighted interval) | Additive sum-weighted interval | Margin sum-weighted interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real pairs | 734 / 741 | 9,546 | 0.243956 (0.209610–0.278616) | 0.290198 (0.260933–0.318456) | 0.227472–0.260898 | 0.276818–0.303284 |
| Controls, Path 1 description only | 1,452 / 1,482 | 18,256 | 0.280234 (0.244253–0.317013) | 0.295813 (0.275441–0.316293) | 0.262852–0.299260 | 0.286792–0.305295 |

| Real-pair pooled count | Recomputed value |
| --- | ---: |
| Additive hits | 2,264 / 9,546 = 0.237167 (23.7%) |
| Margin-winner hits | 2,702 / 9,546 = 0.283050 (28.3%) |
| Additive minus margin row-mean difference | −0.046242; 95% product interval −0.080033 to −0.012683; sum interval −0.060896 to −0.029835 |

| Control pooled count, description only | Recomputed value |
| --- | ---: |
| Additive hits | 5,029 / 18,256 = 0.275471 (27.5%) |
| Margin-winner hits | 5,327 / 18,256 = 0.291794 (29.2%) |

`fig119.py` output `real_pairs`, `controls`; source `logits.jsonl`, using `add`, `a`, `b`, `Ma`, `Mb`, and `g`. The intervals are 2,000 unit-bootstrap draws with seed 11902.

## Path 2 interaction and effect matching

Interaction share uses raw logits centered on T, the union of each of the four passes' top-10 tokens. It is |I|² / |D_pair|² at each position, medianed within a row and then across rows. Total variation is computed from the softmax distributions of the additive vector and the two-push vector over the full vocabulary.

| Row type | Directed rows with positions | Positions | Median interaction share | Median total variation |
| --- | ---: | ---: | ---: | ---: |
| Real pairs | 1,482 / 1,482 | 19,532 | 0.470673 | 0.707213 |
| Controls | 1,482 / 1,482 | 19,532 | 0.368589 | 0.643542 |

Each of the 741 unordered real pairs occurs twice in Path 2, once in each order on the first unit's answer.

| Effect-matched quantity | Recomputed value |
| --- | ---: |
| Control positions in log interaction fit | 19,532 |
| Real positions in control support range | 16,208 / 19,532 = 0.829818 |
| Excluded real positions | 3,324, all outside the control D2 support; 0 excluded for nonpositive norms |
| Real rows with at least one retained position | 1,475 / 1,482 |
| Mean row residual | 0.0429998 |
| Mean residual, 95% product-weighted interval | −0.040032 to 0.121307 |
| Mean residual, sum-weighted interval | 0.000633 to 0.081427 |
| Exponentiated mean residual | 1.043938 |
| Ratio, 95% product-weighted interval | 0.960759 to 1.128972 |
| Ratio, sum-weighted interval | 1.000633 to 1.084835 |

The fit regresses control log interaction size on the two log single-effect sizes. Real positions count only when the second single-effect size lies within the control positions' 1st to 99th percentile support range. Seven real rows have no retained positions, so the mean residual uses 1,475 rows. `fig119.py` output `interactions` and `effect_matched`; source `interaction.jsonl` fields `nI`, `nA`, `n2`, `nPair`, and `tv`.

## Broken answers and reader labels

Part 86's full code rule keeps an answer if it mentions Qwen (optionally numbered), Tongyi, or Alibaba anywhere, or if its first sentence contains A.I., AI, assistant, language model, model, chatbot, or LLM. Otherwise, fewer than four words or a three-word sequence repeated at least three times is broken; all other answers are changed. The repeat-only column counts the repeated-three-word condition alone, before the keep filters.

| Answer type | Full-code broken | Three-word-repeat measure alone |
| --- | ---: | ---: |
| Real pairs | 357 / 741 | 330 / 741 |
| Controls | 659 / 1,482 | 616 / 1,482 |
| Single units | 4 / 39 | 4 / 39 |
| Random singles | 1 / 39 | 5 / 39 |
| Plain answer | 0 / 1 | 0 / 1 |

| DeepSeek sample class | Both | Reader marked coherent |
| --- | ---: | ---: |
| Concept x concept | 0 / 28 | 4 / 28 |
| Slot x concept | 0 / 40 | 7 / 40 |
| Slot x slot | 0 / 40 | 5 / 40 |
| Controls | 0 / 40 | 9 / 40 |

There are 148 DeepSeek sample labels and 30 Luna labels on the shared items. DeepSeek marked “both” on 0 / 148; Luna marked “both” on 0 / 30. Kappa is undefined; observed agreement is 30 / 30. Known-example accuracy is 12 / 12 for each reader. Errors and retry items were 0 for both known-example and sample labels. DeepSeek V4.1 Flash used the `ocgo` route with a 4,000-token cap and eight workers. Luna 6 used `cliproxy`, a 4,000-token cap, adaptive thinking, and four workers. The calls did not set an explicit temperature. Sources: `fig119.py` output `broken` and `reader`; `answers.jsonl`, `labels-ds.jsonl`, `labels-luna.jsonl`, key files, `known-ds.jsonl`, `known-luna.jsonl`, `labels.log`, `part119/label119.py`, `part92b/label92b.py`, and `judge_client.py`.

For 119.10, leaving out each agreed concept unit leaves 21 / 21 labeled concept x concept pairs and 0 “both” labels. `fig119.py` output `reader.leave_one_concept_out`; labels and unit keys are the raw inputs.

| Concept unit omitted | Both / labeled | Expected pairs | Errors |
| --- | ---: | ---: | ---: |
| p92-08 | 0 / 21 | 21 | 0 |
| p92-15 | 0 / 21 | 21 | 0 |
| p92-29 | 0 / 21 | 21 | 0 |
| p92-39 | 0 / 21 | 21 | 0 |
| p92-43 | 0 / 21 | 21 | 0 |
| p92-45 | 0 / 21 | 21 | 0 |
| p92-48 | 0 / 21 | 21 | 0 |
| p92-51 | 0 / 21 | 21 | 0 |

## Pair classes, winners, and unit summaries

Additive and margin values below are row means over contested positions. Path 2 values are medians by row then group.

| Kind grouping | Pair class | Expected pairs (contested rows) | Directed Path 2 rows | Additive / margin | Real / control interaction share |
| --- | --- | ---: | ---: | ---: | ---: |
| Agreed | slot x slot | 210 (206) | 420 | 0.224 / 0.276 | 0.500 / 0.404 |
| Agreed | concept x concept | 28 (28) | 56 | 0.251 / 0.317 | 0.473 / 0.349 |
| Agreed | slot x concept | 168 (168) | 336 | 0.253 / 0.305 | 0.489 / 0.395 |
| Agreed | involving disputed units | 335 (332) | 670 | 0.251 / 0.289 | 0.444 / 0.336 |
| Haiku only | slot x slot | 210 (206) | 420 | 0.224 / 0.276 | 0.500 / 0.404 |
| Haiku only | concept x concept | 153 (153) | 306 | 0.262 / 0.313 | 0.454 / 0.342 |
| Haiku only | slot x concept | 378 (375) | 756 | 0.248 / 0.289 | 0.459 / 0.362 |

The agreed slot x concept comparison has 90 concept token wins among 137 decided pairs of 168 expected (0.656934). The 95% product-weighted unit-bootstrap interval is 0.470146 to 0.823898; 506 of 2,000 draws (25.3%) fall below 0.60. Haiku-only grouping gives 180 / 303 = 0.594059. Each leave-one-concept-out row retains 147 / 147 possible pairs:

| Concept unit omitted | Concept wins / decided | Share | Observed / expected pairs |
| --- | ---: | ---: | ---: |
| p92-08 | 76 / 120 | 0.633333 | 147 / 147 |
| p92-15 | 83 / 118 | 0.703390 | 147 / 147 |
| p92-29 | 77 / 119 | 0.647059 | 147 / 147 |
| p92-39 | 83 / 122 | 0.680328 | 147 / 147 |
| p92-43 | 74 / 118 | 0.627119 | 147 / 147 |
| p92-45 | 74 / 118 | 0.627119 | 147 / 147 |
| p92-48 | 84 / 121 | 0.694215 | 147 / 147 |
| p92-51 | 79 / 123 | 0.642276 | 147 / 147 |

| Group | Units | Median R | Median flip alpha |
| --- | ---: | ---: | ---: |
| Agreed slot | 21 | 0.007177 | 0.80 |
| Agreed concept | 8 | 0.012275 | 0.70 |
| Disputed | 10 | 0.004785 | not summarized |
| Haiku-only slot | 21 | 0.007177 | 0.80 |
| Haiku-only concept | 18 | 0.007775 | 0.70 |

R ranges from 0 to 0.058352 across 39 units. Agreed concept minus slot median R is 0.005098; Mann–Whitney p = 0.146800. Haiku-only median difference is 0.000598. Agreed slot minus concept flip-alpha median is 0.100000; p = 0.437184. Haiku-only difference is 0.100000. Eleven of 21 agreed slot units are at the flip-alpha cap of 0.8. Disputed median R is 0.004785 and is closer to agreed slot by 0.005098; under Haiku-only groups the proximity margin is 0.000598. Dominance/flip-alpha Spearman rho is 0.034009, two-sided p = 0.837156, n = 39.

The code calls SciPy 1.18.1 `scipy.stats.mannwhitneyu(c, s, alternative="greater")` for concept R (`c`) vs slot R (`s`) and `mannwhitneyu(cf, sf, alternative="less")` for concept flip alpha (`cf`) vs slot flip alpha (`sf`). It passes no explicit `method` or continuity option. The R thresholds in 119.6 and 119.9 were unreachable on this scale; both were registration errors.

## Accounting, seeds, and verdict differences

- The answer file contains 2,302 rows; the separate no-hook answer makes 2,303 generated answers in the design total. `timing-main.json` reports 2,163 new main answers; the 139 pilot rows were reused.
- Gate 1's exact token comparison passed, with 50 tokens in each array.
- `gate3_control` counts all control Path 1 teacher-forced positions, 31,139 total, rather than contested positions.
- The reader sample used seed 11900; the Luna sample used 11903; the pilot peek used 11901; unit bootstrap used seed 11902 for 2,000 draws; example-row shuffle used seed 119.
- The pilot peek recomputes to 60 / 252 = 0.238095 for 20 pair rows and 135 / 507 = 0.266272 for 40 control rows.
- The pilot's by-eye description says about one quarter of its 20 pair answers looked broken; the full Part 86 triage rule marks 8 / 20 pair rows broken.
- Claude's by-eye read covered 21 answers: 7 per agreed pair class, 3 or 4 of 7 concept x concept answers showed something of both units, 0 of 28 concept x concept reader labels were “both,” and 6 / 7 slot x concept answers followed the concept theme. This is a small reading, not a registered count.
- Prediction values were recomputed by `fig119.py`. The scorer verdicts remain 119.1 failed, 119.2 failed, 119.3 failed, 119.4 held, 119.5 held, 119.6 failed, 119.7 failed, 119.8 failed, 119.9 failed, and 119.10 not read. The README reports 119.10 as failed under the design's literal words and records the scorer's withheld / not-read output beside it.

