### Table 1. The take gate, per GPT-2 XL editor and for wsp

| Arm | Attempted | Already satisfied | Took | Required | Most that could take | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ROME | 24 | 5 | 19 | 21 | 19 | failed |
| MEMIT | 24 | 5 | 16 | 21 | 19 | failed |
| AlphaEdit | 24 | 5 | 19 | 21 | 19 | failed |
| wsp | 24 | 0 | 24 | 21 | 24 | passed |

Already-satisfied case ids: [3376, 4065, 12944, 18738, 20713] (the same five for all three editors: True).

### Table 2. The five already-satisfied records under ROME (log-probabilities)

| Case | Prompt | New object | Old object | New before | Old before | New after |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 4065 | Laurie Anderson works as | actor | poet | -11.21 | -11.72 | -0.030 |
| 3376 | Arvid Lindman found employment in | London | Stockholm | -5.01 | -5.87 | -0.007 |
| 12944 | Rafael Merry del Val's position is | cardinal | bishop | -13.22 | -13.37 | -0.012 |
| 18738 | The twin city of Mexico City is | Atlanta | Havana | -12.23 | -13.17 | -0.013 |
| 20713 | State of Brazil's capital is | Madrid | Salvador | -10.63 | -12.92 | -0.012 |

### Table 3. Did the records that did not count still move?

| Editor | Not already satisfied, did not take | Already satisfied, new object leads after |
| --- | ---: | ---: |
| ROME | 0 | 5 of 5 |
| MEMIT | 3 (cases 18976, 5137, 9839) | 5 of 5 |
| AlphaEdit | 0 | 5 of 5 |

The records that were not already satisfied and did not take (log-probabilities):

| Editor | Case | New before | New after | Old after |
| --- | ---: | ---: | ---: | ---: |
| MEMIT | 18976 | -9.58 | -5.72 | -2.07 |
| MEMIT | 5137 | -9.04 | -4.56 | -2.52 |
| MEMIT | 9839 | -9.08 | -2.64 | -0.57 |

### Table 4. What the scorer counted (not verdicts: the gate failed first)

| Item | Denominator | Count |
| --- | ---: | ---: |
| 198.1 ROME single | 19 | 0 |
| 198.3 ROME single | 19 | 0 |
| 198.4 ROME oracle naming | 19 | 0 |
| 198.4 ROME update-top naming | 19 | 0 |
| 198.1 MEMIT single | 16 | 0 |
| 198.3 MEMIT single | 16 | 0 |
| 198.4 MEMIT oracle naming | 16 | 0 |
| 198.4 MEMIT update-top naming | 16 | 1 |
| 198.1 AlphaEdit single | 19 | 0 |
| 198.3 AlphaEdit single | 19 | 0 |
| 198.4 AlphaEdit oracle naming | 19 | 0 |
| 198.4 AlphaEdit update-top naming | 19 | 2 |
| 198.1 MEMIT mass | 2 | 0 |
| 198.3 MEMIT mass | 2 | 0 |
| 198.1 AlphaEdit mass | 2 | 0 |
| 198.3 AlphaEdit mass | 2 | 0 |
| 198.6 WSP T_word localization | 24 | 22 |
| 198.7 WSP blind joint success | 24 | 13 |
| 198.8 WSP sequence final strict names | 10 | 6 |
| wsp oracle naming (diagnostic) | 24 | 13 |
| 198.2 T_word localization, ROME (descriptive) | 24 attempted (19 took) | 0 (0 of those that took) |
| 198.2 T_word localization, MEMIT (descriptive) | 24 attempted (16 took) | 0 (0 of those that took) |
| 198.2 T_word localization, AlphaEdit (descriptive) | 24 attempted (19 took) | 0 (0 of those that took) |
| 198.5 aimed control localized (descriptive) | 19 | 15 |
| S1 aimed control named (registered bar: at least 20) | 19 | 19 |
| S2 matched random control localized (registered bar: at most 2) | 24 | 14 |
| random partners: blind joint success, as 198.7 (computed here) | 24 | 0 |
| random partners: blind reader names the object, strict (computed here) | 24 | 0 |

### Table 5. wsp writes against their matched random controls, T_word localization

| | Random localized | Random not localized |
| --- | ---: | ---: |
| wsp localized | 12 | 10 |
| wsp not localized | 2 | 0 |

Exact two-sided McNemar p on the discordant pairs (10 against 2), computed here: 0.0386.

### Table 7. Gradient edits on GPT-2 XL: base and edited T_word, median and range over the edited layers of records that took (per-layer values are in records/armA-*.jsonl)

| Editor | Edited layers read | Base T_word, median (range) | Edited T_word, median (range) | Largest rise | Edited above layer 45's 9.79 |
| --- | ---: | --- | --- | ---: | ---: |
| ROME | 19 | 5.08 (5.08 to 5.08) | 5.15 (4.93 to 5.62) | 0.54 | 0 |
| MEMIT | 80 | 5.74 (5.08 to 7.65) | 5.73 (5.03 to 7.67) | 0.14 | 0 |
| AlphaEdit | 95 | 5.74 (5.08 to 7.65) | 5.74 (5.01 to 7.71) | 0.67 | 0 |

### Table 6. Arm C, unedited models: largest eligible-token z (T_word), against the design reviewer's

| Model | Role | Stored dtype | Snapshot | Registered run | Layer, token | Reviewer (design text) | Difference |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| EleutherAI/pythia-410m | calibration | float16 | 9879c9b5 | 7.38 | 16, ' several' | 6.73 'ton' | 0.649 |
| Qwen/Qwen3-0.6B | calibration | bfloat16 | c1899de2 | 10.33 | 24, ' set' | 10.33 ' set' | -0.003 |
| allenai/OLMo-2-0425-1B | calibration | float32 | a1847dff | 17.73 | 14, ' take' | 12.1 ' time' | 5.625 |
| EleutherAI/pythia-1.4b | evaluation | float16 | fedc38a1 | 14.32 | 22, ' order' | 14.36 ' order' | -0.036 |
| Qwen/Qwen3-1.7B | evaluation | bfloat16 | 70d244cc | 15.90 | 24, ' set' | 15.9 ' set' | -0.003 |
| HuggingFaceTB/SmolLM2-1.7B | evaluation | bfloat16 | effd688a | 10.97 | 22, ' Seven' | 10.92 ' time' | 0.052 |
| HuggingFaceTB/SmolLM3-3B | evaluation | bfloat16 | a07cc9a0 | 13.08 | 32, 'bfd' | 13.08 'bfd' | -0.003 |
| Qwen/Qwen2.5-0.5B-Instruct | evaluation | bfloat16 | 7ae55760 | 9.21 | 19, ' way' | 9.21 ' way' | -0.003 |
| gpt2-xl | host | float32 | gpt2-xl | 9.79 | 45, 'Used' | 9.79 | -0.003 |
| Qwen/Qwen3.5-0.8B | host | float32 | 2fc06364 | 7.34 | 23, 'ise' | 7.3 | 0.039 |

T* (largest calibration value) = 17.73. Evaluation models above it: 0 of 5.
