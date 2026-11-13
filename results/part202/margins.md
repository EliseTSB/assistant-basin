# Part 202 size margins from published Part 164

Computed by `part202/margins202.py` from the published score and answer rows.

## Gates and registered verdict distances

| Reading | Published estimate | Bar | Distance / headroom |
| --- | ---: | --- | ---: |
| Gate 1 | 6516 complete rows | 6516 complete rows | 0.000 missing/extra rows to fail: 1 |
| Gate 2 Qwen | 16 | 12 | 4.000 row headroom; one more lost hit than headroom fails |
| Gate 2 maker | 16 | 12 | 4.000 row headroom; one more lost hit than headroom fails |
| Gate 3 | 86/100 (0.860000) | 0.70 | 16.000 row headroom; changed→unchanged rows to fail: 17 |
| Gate 4 | 0.07944900932400932 | 0.04 | 63.118 equivalent eligible-answer rows at 1,600 rows/arm |
| 164.1 | 0.7886820388718295 | < 0.7 | 13.319 equivalent Qwen-label rows above boundary; 1,600-row denominator |
| 164.2 | 3.4560000000000004 | >= 2 | 182.000 equivalent broken-answer rows above boundary; 1,600-row denominator |
| 164.3 first step | 0.11875 | > 0 | 190.000 equivalent broken-answer rows to equality; 1,600 rows/dose |
| 164.3 second step | 0.1675 | > 0 | 268.000 equivalent broken-answer rows to equality; 1,600 rows/dose |
| 164.4 upper boundary | 5.528824117059411 | <= 1.3 | 78.729 equivalent Qwen-label rows to upper boundary; 1,600-row denominator |
| 164.5 | 0.08570936085248311 | < 0.5 | 52.664 equivalent Qwen-label rows below boundary; 1,600-row denominator |
| Headline Qwen direction | -0.015510227017880074 | < 0 | 24.816 equivalent Qwen-label rows to equality; 1,600 rows/arm |
| Headline broken direction | 0.19187500000000002 | > 0 | 307.000 equivalent broken-answer rows to equality; 1,600 rows/arm |

## Battery answer lengths

| Arm | Answers | At 80-token cap | Mean token_count |
| --- | ---: | ---: | ---: |
| neuron | 1600 | 108 | 53.259 |
| random | 4800 | 358 | 55.219 |

## Row counts

| Stage | Rows |
| --- | ---: |
| no_push | 16 |
| recipe | 100 |
| battery | 6400 |
| Total | 6516 |

Note: Rate-derived row distances are equivalent counts from the per-unit mean-rate gap at a 1,600-row denominator; they are not a rerun of the scorer under edited labels.

Planned replicate size: 200 rows; pilot: first 3 pushed batches; cap threshold: 80 tokens.
