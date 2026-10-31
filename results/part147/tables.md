# Part 147 score tables

Fixed word support: 5715 tokens.

## Cross-question ranking

| Question | Pushed same | Pushed other mean | Difference | Plain same | Plain other mean |
|---|---:|---:|---:|---:|---:|
| q1 | 0.36788146255912624 | 0.466236971744769 | -0.09835550918564279 | 0.39722408268481857 | -0.006662098677387457 |
| q2 | 0.6195854846351796 | 0.450344544183192 | 0.16924094045198762 | 0.1163150700296133 | -0.01627374084509021 |
| q3 | 0.8428787545114907 | 0.4241774966374669 | 0.41870125787402385 | 0.6486458780832872 | 0.3324731723751038 |
| q4 | 0.8478915521767361 | 0.4277679576840279 | 0.42012359449270814 | 0.6394317970079761 | 0.32738729234405944 |

## Question-centered ranking

| Question | Pushed Spearman | Plain Instruct Spearman |
|---|---:|---:|
| q1 | 0.8441970286055965 | 0.758040478207357 |
| q2 | 0.7378660785091522 | 0.5084329180686226 |
| q3 | 0.8108963275034179 | 0.7216377311395341 |
| q4 | 0.8304872933058923 | 0.7198123793016333 |

## Wrong-question control

For each unit at a question, compare its JS divergence to that question's Base row with its divergence to each of the other three Base rows. Plain Instruct uses the same comparisons. Ties do not count as closer.

| Question | Pushed own-Base closer | Pushed pairs | Pushed share | Plain own-Base closer | Plain pairs | Plain share |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 439 | 768 | 0.5716 | 512 | 768 | 0.6667 |
| q2 | 710 | 768 | 0.9245 | 768 | 768 | 1.0000 |
| q3 | 748 | 768 | 0.9740 | 256 | 768 | 0.3333 |
| q4 | 637 | 768 | 0.8294 | 0 | 768 | 0.0000 |

## Jensen-Shannon distance

Shares below compare each pushed unit's full-vocabulary JS divergence with Base against that unit's plain Instruct divergence.

| Question | Group | Units | Share closer | Median pushed JS (nats) | Median plain JS (nats) |
|---|---|---:|---:|---:|---:|
| q1 | all | 256 | 0.9570 | 0.372683 | 0.504355 |
| q1 | neuron | 128 | 0.9609 | 0.369437 | 0.504355 |
| q1 | random | 128 | 0.9531 | 0.374572 | 0.504355 |
| q1 | L4 | 128 | 0.9844 | 0.359523 | 0.504355 |
| q1 | L5 | 128 | 0.9297 | 0.387591 | 0.504355 |
| q1 | neuron-L4 | 64 | 1.0000 | 0.355202 | 0.504355 |
| q1 | random-L4 | 64 | 0.9688 | 0.361182 | 0.504355 |
| q1 | neuron-L5 | 64 | 0.9219 | 0.387755 | 0.504355 |
| q1 | random-L5 | 64 | 0.9375 | 0.387543 | 0.504355 |
| q2 | all | 256 | 0.9102 | 0.314968 | 0.459534 |
| q2 | neuron | 128 | 0.9219 | 0.321362 | 0.459534 |
| q2 | random | 128 | 0.8984 | 0.310821 | 0.459534 |
| q2 | L4 | 128 | 0.9453 | 0.272303 | 0.459534 |
| q2 | L5 | 128 | 0.8750 | 0.351415 | 0.459534 |
| q2 | neuron-L4 | 64 | 0.9688 | 0.263719 | 0.459534 |
| q2 | random-L4 | 64 | 0.9219 | 0.287399 | 0.459534 |
| q2 | neuron-L5 | 64 | 0.8750 | 0.363473 | 0.459534 |
| q2 | random-L5 | 64 | 0.8750 | 0.333013 | 0.459534 |
| q3 | all | 256 | 0.9961 | 0.346648 | 0.570538 |
| q3 | neuron | 128 | 0.9922 | 0.339855 | 0.570538 |
| q3 | random | 128 | 1.0000 | 0.354748 | 0.570538 |
| q3 | L4 | 128 | 1.0000 | 0.340005 | 0.570538 |
| q3 | L5 | 128 | 0.9922 | 0.356872 | 0.570538 |
| q3 | neuron-L4 | 64 | 1.0000 | 0.333121 | 0.570538 |
| q3 | random-L4 | 64 | 1.0000 | 0.342438 | 0.570538 |
| q3 | neuron-L5 | 64 | 0.9844 | 0.341688 | 0.570538 |
| q3 | random-L5 | 64 | 1.0000 | 0.367336 | 0.570538 |
| q4 | all | 256 | 0.9609 | 0.23564 | 0.504934 |
| q4 | neuron | 128 | 0.9844 | 0.214265 | 0.504934 |
| q4 | random | 128 | 0.9375 | 0.276237 | 0.504934 |
| q4 | L4 | 128 | 0.9688 | 0.222828 | 0.504934 |
| q4 | L5 | 128 | 0.9531 | 0.237782 | 0.504934 |
| q4 | neuron-L4 | 64 | 0.9844 | 0.219761 | 0.504934 |
| q4 | random-L4 | 64 | 0.9531 | 0.23133 | 0.504934 |
| q4 | neuron-L5 | 64 | 0.9844 | 0.199767 | 0.504934 |
| q4 | random-L5 | 64 | 0.9219 | 0.313034 | 0.504934 |

## Predictions

- **147.1 held** — On at least 3 of 4 questions, the median pushed row's Spearman with Base at the same question exceeds its mean Spearman with Base at the other questions by at least 0.05. Value: `{"questions_meeting_0.05": 3, "required": 3, "question_differences": [-0.09835550918564279, 0.16924094045198762, 0.41870125787402385, 0.42012359449270814]}`
- **147.2 held** — The question-centered Spearman of pushed with Base, averaged across questions, is at least 0.3. Value: `{"mean_question_centered_spearman": 0.8058616819810147, "question_spearman": [0.8441970286055965, 0.7378660785091522, 0.8108963275034179, 0.8304872933058923], "plain_instruct_question_spearman": [0.758040478207357, 0.5084329180686226, 0.7216377311395341, 0.7198123793016333], "threshold": 0.3}`
- **147.3 held** — On at least 3 of 4 questions, at least 90% of units are closer to Base in Jensen-Shannon divergence than plain Instruct is. Value: `{"share_closer_by_question": [0.95703125, 0.91015625, 0.99609375, 0.9609375], "questions_meeting_0.90": 4, "required_questions": 3}`
- **147.4 held** — The share in 147.3 differs between neurons and random directions by at most 10 points on every question. Value: `{"absolute_neuron_random_share_difference_by_question": [0.0078125, 0.0234375, 0.0078125, 0.046875], "maximum_allowed": 0.1}`
