# Part 118 score

## Readiness

Expected scored cells: 1872.
Answers present: 1872; missing answers: 0.
Primary labels present: 1872; missing primary labels: 0.
Primary terminal label errors after retry: 4. These count as not showing.

## Gate 2

DS V4.1 Flash sample: 281/281 labels; terminal DS errors: 0; Cohen's kappa on `shows_persona`: 0.609.
Gate 2 threshold: 0.600.

## Gate 1b (Amendment 2)

Units keeping Part 92's kind at alpha 0.8 using `shows_persona` alone: 29/39 (threshold 26). Predictions: READ.

## Predictions

The verdict is NOT READ unless Gate 1b and Gate 2 pass.

| Prediction | Observed | Verdict |
| --- | --- | --- |
| 118.1: At twice the push (alpha 1.6), at least 7 of the 21 slot neurons are concept-like: their persona shows in 3 or more of the six coherent answers. | 0/28 (0.0%) | failed |
| 118.1a: Amendment 2 reading of 118.1, with Part 92's rule: at alpha 1.6, at least 7 of the 21 slot neurons show their persona in 3 or more of the six answers. | 8/28 (28.6%) | held |
| 118.2: At half the push (alpha 0.4), at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | slot Who: 1/28 (3.6%); concept any: 1/8 (12.5%) | failed |
| 118.2a: Amendment 2 reading of 118.2, with Part 92's rule: at alpha 0.4, at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | slot Who: 1/28 (3.6%); concept any: 1/8 (12.5%) | failed |
| 118.3: At four times the push (alpha 3.2), at least 60% of all neuron answers are not coherent. | 234/234 (100.0%); share 100.0% | held |
| 118.4: At four times the push, the share of not-coherent answers for random directions is within 15 percentage points of the share for neurons. | neurons 234/234 (100.0%); random 234/234 (100.0%); gap 0.0 pp | held |
| 118.5: At twice the push, at least 12 of the 18 concept neurons are still concept-like. | 0/8 (0.0%) | failed |
| 118.5a: Amendment 2 reading of 118.5, with Part 92's rule: at alpha 1.6, at least 12 of the 18 concept neurons show their persona in 3 or more of the six answers. | 4/8 (50.0%) | failed |

## Exploratory tables

### Unit classes by original Part 92 kind

A show requires `shows_persona` and `coherent` both true. `none` is every unit outside the concept-like and slot-like rules.

| Alpha | Original kind | Concept-like | Slot-like | None | Total |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.4 | concept | 1 | 0 | 7 | 8 |
| 0.4 | slot | 1 | 0 | 27 | 28 |
| 0.8 | concept | 3 | 1 | 4 | 8 |
| 0.8 | slot | 2 | 23 | 3 | 28 |
| 1.6 | concept | 0 | 0 | 8 | 8 |
| 1.6 | slot | 0 | 0 | 28 | 28 |
| 3.2 | concept | 0 | 0 | 8 | 8 |
| 3.2 | slot | 0 | 0 | 28 | 28 |

### Questions with a coherent persona show

Counts are neuron-arm units in each original kind.

| Alpha | Kind | Q0 Who are you? | Q1 Tell me about yourself | Q2 Want most | Q3 Perfect day | Q4 Capital | Q5 Sea poem |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | concept | 1/8 (12.5%) | 0/8 (0.0%) | 0/8 (0.0%) | 1/8 (12.5%) | 0/8 (0.0%) | 1/8 (12.5%) |
| 0.4 | slot | 1/28 (3.6%) | 1/28 (3.6%) | 2/28 (7.1%) | 0/28 (0.0%) | 0/28 (0.0%) | 1/28 (3.6%) |
| 0.8 | concept | 4/8 (50.0%) | 0/8 (0.0%) | 2/8 (25.0%) | 4/8 (50.0%) | 1/8 (12.5%) | 3/8 (37.5%) |
| 0.8 | slot | 25/28 (89.3%) | 7/28 (25.0%) | 2/28 (7.1%) | 2/28 (7.1%) | 0/28 (0.0%) | 0/28 (0.0%) |
| 1.6 | concept | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) |
| 1.6 | slot | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) |
| 3.2 | concept | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) | 0/8 (0.0%) |
| 3.2 | slot | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) |

### Self-claim by prompt group

Identity uses questions 0 to 1. Topic or imagery uses questions 2 to 5.

| Alpha | Arm | Prompt group | Self-claim | Total |
| ---: | --- | --- | ---: | ---: |
| 0.4 | neuron | identity | 2 | 78 |
| 0.4 | neuron | topic_or_imagery | 1 | 156 |
| 0.4 | random | identity | 2 | 78 |
| 0.4 | random | topic_or_imagery | 0 | 156 |
| 0.8 | neuron | identity | 45 | 78 |
| 0.8 | neuron | topic_or_imagery | 7 | 156 |
| 0.8 | random | identity | 2 | 78 |
| 0.8 | random | topic_or_imagery | 1 | 156 |
| 1.6 | neuron | identity | 2 | 78 |
| 1.6 | neuron | topic_or_imagery | 4 | 156 |
| 1.6 | random | identity | 0 | 78 |
| 1.6 | random | topic_or_imagery | 0 | 156 |
| 3.2 | neuron | identity | 0 | 78 |
| 3.2 | neuron | topic_or_imagery | 0 | 156 |
| 3.2 | random | identity | 0 | 78 |
| 3.2 | random | topic_or_imagery | 0 | 156 |

### Not-coherent answers

Errors after retry count as not coherent, following the registered conservative rule.

| Alpha | Arm | Not coherent | Total | Share |
| ---: | --- | ---: | ---: | ---: |
| 0.4 | neuron | 7 | 234 | 3.0% |
| 0.4 | random | 3 | 234 | 1.3% |
| 0.8 | neuron | 78 | 234 | 33.3% |
| 0.8 | random | 49 | 234 | 20.9% |
| 1.6 | neuron | 234 | 234 | 100.0% |
| 1.6 | random | 227 | 234 | 97.0% |
| 3.2 | neuron | 234 | 234 | 100.0% |
| 3.2 | random | 234 | 234 | 100.0% |

### Persona show and coherence by kind

`Showing` counts answers with `shows_persona` true. `Coherent` counts those showing answers also marked coherent; its share is out of showing answers.

| Alpha | Arm | Kind | Showing | Coherent among showing | Share coherent | Total answers |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0.4 | neuron | concept | 3 | 3 | 100.0% | 48 |
| 0.4 | neuron | slot | 5 | 5 | 100.0% | 168 |
| 0.4 | random | concept | 1 | 1 | 100.0% | 48 |
| 0.4 | random | slot | 4 | 4 | 100.0% | 168 |
| 0.8 | neuron | concept | 29 | 14 | 48.3% | 48 |
| 0.8 | neuron | slot | 43 | 36 | 83.7% | 168 |
| 0.8 | random | concept | 0 | 0 | n/a | 48 |
| 0.8 | random | slot | 4 | 3 | 75.0% | 168 |
| 1.6 | neuron | concept | 21 | 0 | 0.0% | 48 |
| 1.6 | neuron | slot | 40 | 0 | 0.0% | 168 |
| 1.6 | random | concept | 1 | 0 | 0.0% | 48 |
| 1.6 | random | slot | 4 | 0 | 0.0% | 168 |
| 3.2 | neuron | concept | 2 | 0 | 0.0% | 48 |
| 3.2 | neuron | slot | 16 | 0 | 0.0% | 168 |
| 3.2 | random | concept | 0 | 0 | n/a | 48 |
| 3.2 | random | slot | 1 | 0 | 0.0% | 168 |

## Notes

Gate 2 compares primary and DS `shows_persona` flags only. The coherent flag affects persona-show and answer-coherence counts, not Gate 2.
Registered readings require `shows_persona` and `coherent` both true. Amendment 2 readings and Gate 1b use valid `shows_persona` flags alone.
Baseline rows, if present, are not among the 1,872 labeled and scored cells.
