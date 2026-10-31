# Part 143: which side of the push moves which word of “I'm sorry”?

For OLMo 2 1B Instruct, Set R and Set S from Part 139, one dose, one question, and forward passes after teacher-forced prefixes, the median P(" sorry" after "I'm") is 45.1% for full Set R, 16.9% for prompt-only Set R, and 0.0925% for answer-only Set R.

A push adds a unit's direction at the selected layer. Prompt-only applies it during prompt prefill; answer-only applies it to the forced answer-prefix tokens; full applies it during both. Noop uses a zero direction, and plain is the unpushed model.

Closest published work: Malla, Choi and Choi (arXiv:2609.06951) find a matched random direction reproduces a model's default behaviors, including refusal; they push every position and do not split prompt from answer.

Both registered gates passed. Gate 1 covered all 452 noop rows; the maximum difference from plain was 0. Gate 2 required at least 90% of Set R full-arm values to be within 0.05 of Part 139's stored forced-prefix values. All 130 of 130 matched (100%), and the largest gap was 0.0239. The design's proposed first-12-rows read did not happen as a separate step; Gate 1 covered all 452 noop rows.

## Registered predictions

| # | Prediction (registered wording; confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 143.1 | After a forced "I'm", the median Set R unit under prompt-only puts at least 10 times the plain model's probability on " sorry". (medium) | 0.168949 (16.9%) vs plain 0.0002391 (0.0239%), 706.6×. | held |
| 143.2 | After a forced "I'm", the median Set R unit under answer-only puts at least 10 times the plain model's probability on " sorry". (medium) | 0.0009248 (0.0925%) vs plain 0.0002391 (0.0239%), 3.87×. | failed |
| 143.3 | After a forced "I", the median Set R unit under answer-only puts less on "'m" than the median unit under prompt-only. (low) | 18.90% vs 37.88%. | held |
| 143.4 | Under full, the median Set R probability of " sorry" after "I'm" is at least twice the larger of the prompt-only and answer-only medians. (medium) | 45.07%; threshold 33.79%. | held |
| 143.5 | Under full, Set S's median probability of " sorry" after "I'm" is at most a tenth of Set R's. (high) | Set S 0.045043; threshold 0.045067. | held |

## Where does probability move?

Each cell gives the median probability and the middle 50% (Q1–Q3). Set R was selected on the full arm's outcome. Its full values describe selected units and say nothing about how common this is. Set S contains neurons; Set R contains neurons and matched random directions. Plain is one shared model baseline. The noop rows repeat that same value across units because the do-nothing hook does not change the output.

| Readout | Arm | Set | Unit | n | Median | Q1–Q3 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| P("'m" after "I") | plain | — | baseline | 1 | 0.138875 | 0.138875–0.138875 |
| P("'m" after "I") | noop | R | neuron | 96 | 0.138875 | 0.138875–0.138875 |
| P("'m" after "I") | noop | R | random | 34 | 0.138875 | 0.138875–0.138875 |
| P("'m" after "I") | noop | S | neuron | 96 | 0.138875 | 0.138875–0.138875 |
| P("'m" after "I") | prompt-only | R | neuron | 96 | 0.398837 | 0.309654–0.481974 |
| P("'m" after "I") | prompt-only | R | random | 34 | 0.373536 | 0.299857–0.438841 |
| P("'m" after "I") | prompt-only | S | neuron | 96 | 0.204602 | 0.135876–0.278515 |
| P("'m" after "I") | answer-only | R | neuron | 96 | 0.188849 | 0.152924–0.230529 |
| P("'m" after "I") | answer-only | R | random | 34 | 0.198138 | 0.148025–0.235006 |
| P("'m" after "I") | answer-only | S | neuron | 96 | 0.177791 | 0.146384–0.218315 |
| P("'m" after "I") | full | R | neuron | 96 | 0.452486 | 0.368706–0.537790 |
| P("'m" after "I") | full | R | random | 34 | 0.426455 | 0.384315–0.496017 |
| P("'m" after "I") | full | S | neuron | 96 | 0.230615 | 0.153876–0.314698 |
| P(" sorry" after "I'm") | plain | — | baseline | 1 | 0.0002391 | 0.0002391–0.0002391 |
| P(" sorry" after "I'm") | noop | R | neuron | 96 | 0.0002391 | 0.0002391–0.0002391 |
| P(" sorry" after "I'm") | noop | R | random | 34 | 0.0002391 | 0.0002391–0.0002391 |
| P(" sorry" after "I'm") | noop | S | neuron | 96 | 0.0002391 | 0.0002391–0.0002391 |
| P(" sorry" after "I'm") | prompt-only | R | neuron | 96 | 0.168949 | 0.086008–0.379232 |
| P(" sorry" after "I'm") | prompt-only | R | random | 34 | 0.168924 | 0.129484–0.291853 |
| P(" sorry" after "I'm") | prompt-only | S | neuron | 96 | 0.018058 | 0.005370–0.047707 |
| P(" sorry" after "I'm") | answer-only | R | neuron | 96 | 0.001002 | 0.000425–0.002329 |
| P(" sorry" after "I'm") | answer-only | R | random | 34 | 0.0007665 | 0.0003806–0.001741 |
| P(" sorry" after "I'm") | answer-only | S | neuron | 96 | 0.0005741 | 0.0002531–0.001245 |
| P(" sorry" after "I'm") | full | R | neuron | 96 | 0.453902 | 0.317209–0.594673 |
| P(" sorry" after "I'm") | full | R | random | 34 | 0.398180 | 0.326315–0.565512 |
| P(" sorry" after "I'm") | full | S | neuron | 96 | 0.045043 | 0.010780–0.118815 |

The plain P(" sorry" after "I'm") here is 0.000239; Part 139's stored value is 0.000262, a 9% difference. Using Part 139's value, prediction 143.1's prompt-only median is 644.8 times the baseline, rather than 706.6 times Part 143's baseline. Both scripts use Part 139's `_position_hook`. Part 139's `_distribution` concatenates the prompt and forced prefix, then runs one full-arm forward pass without a cache. Part 143 prefills the prompt with a cache and feeds the forced prefix tokens one at a time, so it can apply the hook to the prompt only, the answer prefix only, or both. Thus, the hook and forced-prefix question are shared, while the forward-pass path differs. The stored files do not isolate the cause of the baseline difference.

A prompt-only push leaves changed cached states, so it is not a pure test of reading. This part says where the probability moves, not why.

## Method and limits

The question was “Who are you?”. The model, revision, bf16 format, A770 device, hook, and α=0.8 dose follow Part 139. The two readouts are the next-token probabilities after teacher-forced “I” and “I'm”. No text was generated. This is one model, one dose, one question, and two layers. Forced prefixes skip the answer the model would have written.

## Files

- `results/part143/results.jsonl`: measured probabilities from the forward passes.
- `results/part143/score.json`: gates, medians, middle 50% ranges, and prediction verdicts, computed by `part143/score143.py`.
- `results/part143/SCORE.md`: readable gate and verdict summary.
- `results/part143/launch.jsonl`: launch record.
- `PART143-DESIGN.md`: registered question, method, gates, predictions, and reading rules.
- `part143/run143.py` and `part139/run139.py`: measurement and shared phase-hook code.

## Found afterwards, not registered

Probabilities do not add: 16.9% and 0.09% against 45.1%. On the log-odds scale they nearly do. For each unit, the full arm's log-odds minus (prompt-only + answer-only - plain) has these medians, with the middle 50%: " sorry" after "I'm", Set R -0.024 (-1.013 to 0.934), Set S -0.013 (-0.899 to 0.984); "'m" after "I", Set R -0.091 (-0.364 to 0.213), Set S -0.112 (-0.490 to 0.103). So the gaps centre near zero, with wide spread on single units. [logodds-check.py](logodds-check.py) computes these. I looked at this after the registered reading, because I had written on the front page that the sides do not add, without checking this scale. The cold check did not cover this section. Set R is selected on the full arm, which can bias this comparison.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; findings folded.
