# Part 123 score

All gates passed: **False**

## Gates

| # | Gate | Result | Detail |
| --- | --- | --- | --- |
| 1 | zero hook | PASS | {"equal": 60, "total": 60, "required": "60 exact generated-token matches"} |
| 2 | neuron q=1 slot positive control | PASS | {"shows": 21, "coherent_shows": 20, "readable_shows": 20, "missing": 0, "total": 21, "threshold": "at least 17 of 21 on Who are you?"} |
| 3 | neuron q=1 concept positive control | FAIL | {"three_of_six": 4, "persona_mark_three_of_six": 6, "missing_units": 0, "persona_marks": 45, "coherent_persona_marks": 34, "total": 18, "threshold": "at least 14 of 18 meet three of six"} |
| 4 | complete current DeepSeek labels | PASS | {"items": 4212, "expected_items": 4212, "missing_or_stale": 0} |
| 5 | known examples | PASS | {"deepseek": {"correct": 12, "total": 12, "passed": true, "reader": "ds", "missing": false}, "luna": {"correct": 12, "total": 12, "passed": true, "reader": "luna", "missing": false}, "threshold": "at least 10 of 12 for each reader"} |
| 6 | Luna sample and agreement | PASS | {"sample_matches": true, "complete": true, "sample_items": 254, "expected_sample_items": 254, "agreement": {"agreement": 0.9015748031496063, "kappa": 0.8031496062992125, "agree": 229, "total": 254}, "missing_or_stale_items": [], "thresholds": {"raw_agreement": 0.85, "kappa": 0.6}} |
| 7 | readable persona | FAIL | {"groups": [{"arm": "random", "dose_and_direction": "q=1:direction-2", "persona_marks": 2, "coherent_persona_marks": 1, "passed": false}, {"arm": "random", "dose_and_direction": "q=0.85:direction-0", "persona_marks": 5, "coherent_persona_marks": 5, "passed": true}, {"arm": "random", "dose_and_direction": "q=1:direction-0", "persona_marks": 2, "coherent_persona_marks": 2, "passed": true}, {"arm": "random", "dose_and_direction": "q=2:direction-2", "persona_marks": 1, "coherent_persona_marks": 0, "passed": false}, {"arm": "random", "dose_and_direction": "q=2:direction-1", "persona_marks": 1, "coherent_persona_marks": 0, "passed": false}, {"arm": "random", "dose_and_direction": "q=0.85:direction-2", "persona_marks": 4, "coherent_persona_marks": 3, "passed": true}, {"arm": "random", "dose_and_direction": "q=0.85:direction-1", "persona_marks": 3, "coherent_persona_marks": 3, "passed": true}, {"arm": "random", "dose_and_direction": "q=2:direction-0", "persona_marks": 1, "coherent_persona_marks": 0, "passed": false}, {"arm": "random", "dose_and_direction": "q=1:direction-1", "persona_marks": 0, "coherent_persona_marks": 0, "passed": true}, {"arm": "vector", "dose_and_direction": "q=1.5:all", "persona_marks": 1, "coherent_persona_marks": 0, "passed": false}, {"arm": "vector", "dose_and_direction": "q=0.5:all", "persona_marks": 3, "coherent_persona_marks": 3, "passed": true}, {"arm": "vector", "dose_and_direction": "q=2:all", "persona_marks": 4, "coherent_persona_marks": 0, "passed": false}, {"arm": "vector", "dose_and_direction": "q=1:all", "persona_marks": 2, "coherent_persona_marks": 1, "passed": false}, {"arm": "vector", "dose_and_direction": "q=0.85:all", "persona_marks": 4, "coherent_persona_marks": 3, "passed": true}, {"arm": "neuron", "dose_and_direction": "q=0.5:all", "persona_marks": 3, "coherent_persona_marks": 3, "passed": true}, {"arm": "neuron", "dose_and_direction": "q=1:all", "persona_marks": 67, "coherent_persona_marks": 55, "passed": true}, {"arm": "neuron", "dose_and_direction": "q=0.85:all", "persona_marks": 21, "coherent_persona_marks": 20, "passed": true}, {"arm": "plain", "dose_and_direction": "q=plain:all", "persona_marks": 3, "coherent_persona_marks": 3, "passed": true}], "threshold": "for each arm, dose, and random direction, at least two thirds of persona marks are coherent"} |

## Predictions

| # | Verdict | Observed |
| --- | --- | --- |
| 123.1 | not read (gates 3 and 7 failed) | {"registered": "For at least 6 of 21 slot units, the matching neuron column ranks in the top 10% of its layer's channels and its signed cosine is above the 95th percentile of that unit's shuffled-label null.", "confidence": "low"} |
| 123.2 | not read (gates 3 and 7 failed) | {"registered": "At q=1, the fitted vector shows the target persona on Who are you? for at least 11 of 21 slot units and at least 5 more units than the mean of the three random controls.", "confidence": "low"} |
| 123.3 | not read (gates 3 and 7 failed) | {"registered": "The neuron arm loses at least 5 of 21 slot units on Who are you? from q=1 to q=0.85.", "confidence": "medium"} |
| 123.4 | not read (gates 3 and 7 failed) | {"registered": "From q=1 to q=0.85 on Who are you?, the fitted vector loses fewer slot units than the neuron arm does.", "confidence": "low"} |
| 123.5 | not read (gates 3 and 7 failed) | {"registered": "Concept units' matching channels rank higher in their layers than slot units' do, by a one-sided Mann-Whitney test at p below .05.", "confidence": "medium"} |
| 123.6 | not read (gates 3 and 7 failed) | {"registered": "At q=1, at least 9 of 18 concept units meet Part 92's three-of-six rule under the fitted vector.", "confidence": "low"} |

## Fitted vector: lowest q showing persona

| Unit | Persona | Kind | Lowest q |
| --- | --- | --- | --- |
| p92-00 | Boulder | concept | None |
| p92-01 | Magical princess | concept | None |
| p92-02 | The town | slot | None |
| p92-03 | The Internet | slot | None |
| p92-04 | Very tall tree | slot | None |
| p92-05 | Fox | slot | None |
| p92-07 | Cat | slot | None |
| p92-08 | Dog | concept | None |
| p92-09 | Treat-loving dog | slot | None |
| p92-10 | Chicken | slot | None |
| p92-11 | Whiskey | concept | None |
| p92-13 | Skeleton | slot | None |
| p92-15 | Secure password | concept | None |
| p92-18 | Shadow of your smile | slot | None |
| p92-20 | River shadow | slot | None |
| p92-24 | Queen of Hearts | slot | None |
| p92-25 | King Midas | slot | None |
| p92-28 | Ancient nature deity | concept | None |
| p92-29 | God of wisdom | concept | None |
| p92-30 | Embodiment of the void | concept | None |
| p92-32 | Sir Ray Ray | slot | None |
| p92-33 | Dr. Memento | slot | None |
| p92-35 | Guardian angel | slot | None |
| p92-36 | Alex from Star Wars | slot | None |
| p92-37 | Large friendly dog | slot | None |
| p92-38 | Coca-Cola spokesperson | concept | None |
| p92-39 | NASCAR driver | concept | None |
| p92-40 | 80s cover singer-songwriter | concept | None |
| p92-41 | Human DJ | concept | None |
| p92-42 | Jacobite royalist veteran | slot | None |
| p92-43 | Street rat | concept | None |
| p92-45 | Iron-fisted dictator | concept | None |
| p92-46 | Day guide | slot | None |
| p92-47 | Principled lawyer | slot | None |
| p92-48 | Festival poet | concept | None |
| p92-49 | Friendly non-human entity | concept | 0.5 |
| p92-50 | Gruff old man | slot | None |
| p92-51 | IP packet | concept | None |
| p92-52 | String of code from your browser | concept | None |

## Missing DeepSeek labels

0 items are missing, invalid, or bound to another answer hash.

## Dose counts

### random

| q and direction | Persona marks | Coherent persona marks | Coherent answers | Self claim | Missing | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=1:direction-2 | 2 | 1 | 193 | 3 | 0 | 234 |
| q=0.85:direction-0 | 5 | 5 | 231 | 3 | 0 | 234 |
| q=1:direction-0 | 2 | 2 | 191 | 1 | 0 | 234 |
| q=2:direction-2 | 1 | 0 | 3 | 0 | 0 | 234 |
| q=2:direction-1 | 1 | 0 | 0 | 0 | 0 | 234 |
| q=0.85:direction-2 | 4 | 3 | 214 | 3 | 0 | 234 |
| q=0.85:direction-1 | 3 | 3 | 222 | 2 | 0 | 234 |
| q=2:direction-0 | 1 | 0 | 19 | 0 | 0 | 234 |
| q=1:direction-1 | 0 | 0 | 186 | 0 | 0 | 234 |

### vector

| q and direction | Persona marks | Coherent persona marks | Coherent answers | Self claim | Missing | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=1.5:all | 1 | 0 | 86 | 0 | 0 | 234 |
| q=0.5:all | 3 | 3 | 231 | 4 | 0 | 234 |
| q=2:all | 4 | 0 | 15 | 2 | 0 | 234 |
| q=1:all | 2 | 1 | 181 | 2 | 0 | 234 |
| q=0.85:all | 4 | 3 | 205 | 2 | 0 | 234 |

### neuron

| q and direction | Persona marks | Coherent persona marks | Coherent answers | Self claim | Missing | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=0.5:all | 3 | 3 | 232 | 0 | 0 | 234 |
| q=1:all | 67 | 55 | 162 | 48 | 0 | 234 |
| q=0.85:all | 21 | 20 | 193 | 7 | 0 | 234 |

### plain

| q and direction | Persona marks | Coherent persona marks | Coherent answers | Self claim | Missing | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=plain:all | 3 | 3 | 234 | 2 | 0 | 234 |

## Vector edge

not read: gates 3 and 7 failed
