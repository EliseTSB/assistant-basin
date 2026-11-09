# What the records already on disk say (printed by part201/existing201.py)

## Part 200: batch 1 against chunks of 8, one question, 200 pushed rows

- 76 of 200 answers differ.
- First difference at answer token: min 4, median 15.5, max 74; 65 before answer token 32.
- Length, batch 1 minus chunked: mean -1.67 tokens; 28 longer, 29 shorter, 19 equal.

| Label | Rows that flip | True in chunked only | True in batch 1 only |
| --- | ---: | ---: | ---: |
| name_gone | 2 | 1 | 1 |
| broken | 6 | 4 | 2 |
| qwen_part161_rule | 5 | 1 | 4 |

## CPU float32 reference, eight plain prompts, 96 tokens

- Batch 1 against batch 2: 8 of 8 rows identical; largest margin difference 5.72e-05.
- Share of its 696 positions with a top-two margin below 0.0625, 0.125, 0.25, 0.5, 1.0: 1.7%, 2.4%, 5.0%, 9.5%, 18.4%.
- 6 of 8 rows have at least one margin below 0.125.

## A770 bf16 eager (Part 183 kernel check) against that reference

| Batch | Positions decided from an identical prefix | Median margin error | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 603 | 0.0506 | 0.124 | 0.2068 | 0.3018 |
| 2 | 650 | 0.0452 | 0.1299 | 0.246 | 0.5759 |
| 8 | 561 | 0.0469 | 0.1245 | 0.2353 | 0.8884 |

- Of 1814 such positions, 28 are exact ties (margin 0) and 43 sit at 0.125; 72 (4.0%) are at 0.125 or less and 114 (6.3%) at 0.25 or less; 1814 are multiples of 0.0625.
- Divergences from the reference: 6; the largest bf16 margin at one is 0.125, the largest reference margin 0.1006.
- Rows whose tokens differ between batch sizes, of 8: 1_vs_2 2, 1_vs_8 3, 2_vs_8 2.

| Batch | Prompt | Position | Reference margin | bf16 margin |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6 | 2 | 0.0666 | 0.0 |
| 2 | 3 | 85 | 0.0084 | 0.0 |
| 2 | 6 | 59 | 0.0118 | 0.0 |
| 8 | 3 | 85 | 0.0084 | 0.0 |
| 8 | 6 | 2 | 0.0666 | 0.0 |
| 8 | 7 | 63 | 0.1006 | 0.125 |
