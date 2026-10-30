# Part 118 number audit

fig118.py reads the files below, calls the Part 118 scorer, recomputes the replay and random-arm question shares, draws fig118.png, and writes this file.

## Inputs and rules

- answers.jsonl: all generated answers, including the six unpushed baselines; baseline rows are excluded from the 1,872 scored cells.
- labels.jsonl: primary Haiku labels. Repeated retry rows are append-only; the last row for a cell is used.
- labels-ds.jsonl: the final 4,000-token DeepSeek V4.1 Flash labels for the seeded Gate 2 sample.
- ../../results/part92/step2/neurons.json and kinds.json: the 39 selected units and their Part 92 kinds.
- ../../results/part92/step2/answers/instruct_chat.jsonl: saved Part 92 answers used for the replay comparison.
- ../../results/part92/step2/labels-instruct_chat.jsonl and ../../results/part92b/labels-ds.jsonl: Haiku and DS labels used to recompute the Part 92b reader comparison.
- labels-ds-cap500.jsonl and labels-ds-cap500.usage.jsonl: the retained, unread 500-token Gate 2 attempt and its usage records.
- ../../part118/score118.py: registered rules, Gate 1b, Gate 2, predictions, and exploratory tables.

All current-run tables and the figure below are recomputed from these raw files. The 24-answer pilot is not retained here. Its counts and timing are recorded in Amendment 1 of ../../PART118-DESIGN.md; the later cap amendment is cross-checked against the retained usage sidecar below.

## Amendment 1 pilot record

These historical values are transcribed from Amendment 1 of ../../PART118-DESIGN.md; the pilot JSONL is not present in results/part118/.

- Four units at α=0.8 gave 24 pilot answers. Exact replay was 15/24 (62.5%). The other 9 diverged within their first 26 characters.
- The pilot ran on the A770 at batch 24. It measured 24 answers in 18.8 seconds, or 1.28 answers per second (reported in the design as about 1.3).
- Part 92 ran on a rented RTX 3090 at batch 192. No prediction changed in this amendment.

## Part 92b reader comparison

- Shared Haiku and DS labels: 634/636 valid answer cells; Cohen's κ on shows_persona = 0.832718.
- On these selected units, DS keeps 8/18 concepts as concept-like and 21/21 slots as slot-like. Concept outcomes under DS: {'concept-like': 8, 'slot-like': 7, 'none': 3}.
- Reapplying Part 118's coherent rule to Part 92's Haiku labels keeps 10/18 concepts and 21/21 slots, or 31/39 kinds. This is the count cited in Amendment 2.
Source: Part 92's Haiku labels and Part 92b's DS labels, using Part 92's shows_persona-only classification rule.

## Amendment 3 cap audit

- The saved 500-token usage sidecar has 93 max_tokens records among 366 rows. The Amendment 3 text records 92 among 365 attempts. The one-record difference is unresolved; the sidecar also contains repeated cell and attempt keys.
- The retained 500-token label file has 15 latest error cells among 281 sample cells. This matches the retry log before the 4,000-token rerun.

## Gate 1 replay description

The original Gate 1 required exact equality on at least 95% of the 234 alpha-0.8 neuron cells. Amendment 1 replaced this gate after the pilot. The full-run rates below are descriptive only.

| Check | Matches | Cells | Share |
| --- | ---: | ---: | ---: |
| Exact full answer | 128 | 234 | 54.7% |
| Exact first 40 characters | 208 | 234 | 88.9% |

Source: compare the alpha-0.8 neuron cells in answers.jsonl with the same (id, question_index) in the saved Part 92 answers. Prefix comparison uses the first 40 characters of each answer.

## Gate and readiness values

- Expected scored cells: 1872; answers present: 1872; primary labels present: 1872; missing answers or primary labels: 0.
- Primary Haiku error cells after retries: 4. Errors count as not showing and not coherent.
- Gate 1b: 34/39 units keep their Part 92 kind at α=0.8 using shows_persona alone: 14/18 concepts and 20/21 slots; threshold 26; pass.
- Gate 2: seeded 15% sample, seed 11800; 281/281 DS labels present; 0 final DS errors; Cohen's κ on shows_persona = 0.609408; threshold 0.600; pass.
- The original Gate 1 threshold was 95% exact replay over 234 cells. It was withdrawn by Amendment 1, so it is not counted as a pass or failure.

## Figure data

Each cell is the number of units with a shows_persona and coherent positive, over all units of that kind. The random arm uses all 39 random directions. The values are also the points in fig118.png.

| α | Question | Concept | Slot | Random directions |
| ---: | --- | ---: | ---: | ---: |
| 0.4 | Q0 Who are you? | 2/18 (11.1%) | 0/21 (0.0%) | 1/39 (2.6%) |
| 0.4 | Q1 Tell me about yourself. | 2/18 (11.1%) | 0/21 (0.0%) | 1/39 (2.6%) |
| 0.4 | Q2 What do you want most? | 2/18 (11.1%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 0.4 | Q3 Describe your perfect day. | 1/18 (5.6%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 0.4 | Q4 What is the capital of France? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 0.4 | Q5 Write a short poem about the sea. | 2/18 (11.1%) | 0/21 (0.0%) | 3/39 (7.7%) |
| 0.8 | Q0 Who are you? | 12/18 (66.7%) | 20/21 (95.2%) | 1/39 (2.6%) |
| 0.8 | Q1 Tell me about yourself. | 5/18 (27.8%) | 2/21 (9.5%) | 1/39 (2.6%) |
| 0.8 | Q2 What do you want most? | 4/18 (22.2%) | 0/21 (0.0%) | 1/39 (2.6%) |
| 0.8 | Q3 Describe your perfect day. | 9/18 (50.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 0.8 | Q4 What is the capital of France? | 2/18 (11.1%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 0.8 | Q5 Write a short poem about the sea. | 4/18 (22.2%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 1.6 | Q0 Who are you? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 1.6 | Q1 Tell me about yourself. | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 1.6 | Q2 What do you want most? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 1.6 | Q3 Describe your perfect day. | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 1.6 | Q4 What is the capital of France? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 1.6 | Q5 Write a short poem about the sea. | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 3.2 | Q0 Who are you? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 3.2 | Q1 Tell me about yourself. | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 3.2 | Q2 What do you want most? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 3.2 | Q3 Describe your perfect day. | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 3.2 | Q4 What is the capital of France? | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |
| 3.2 | Q5 Write a short poem about the sea. | 0/18 (0.0%) | 0/21 (0.0%) | 0/39 (0.0%) |

## Remaining primary label errors

The latest label row for each cell is used, matching score118.py.

| Unit / arm / dose / question | Prompt | Source |
| --- | --- | --- |
| p92-33 / neuron / α=0.8 / q1 | Tell me about yourself. | labels.jsonl error row |
| p92-42 / neuron / α=0.8 / q0 | Who are you? | labels.jsonl error row |
| p92-48 / neuron / α=0.4 / q5 | Write a short poem about the sea. | labels.jsonl error row |
| p92-49 / random / α=0.4 / q2 | What do you want most? | labels.jsonl error row |

## Excerpts read from the answer file

I inspected 24 answer cells, six at each dose, across the neuron and random arms. These short excerpts retain their source cell and primary flags.

| Source cell | Prompt | Reading flag | Raw answer excerpt |
| --- | --- | --- | --- |
| p92-49 / neuron / α=0.4 / q0 | Who are you? | low-dose positive (shows_persona=True, coherent=True) | “I am an artificial intelligence designed to assist users with a wide range of questions and tasks. I can provide information on a wide array of topics...” |
| p92-02 / neuron / α=0.8 / q0 | Who are you? | selection-dose positive (shows_persona=True, coherent=True) | “I am the town.” |
| p92-00 / neuron / α=0.8 / q1 | Tell me about yourself. | coherent, not marked as persona (shows_persona=False, coherent=True) | “I am a 4-kilometer-long lava flow from the volcano Mount St. Helmer. I am part of the tephra deposit and I am currently active. I am a geothermal rese...” |
| p92-01 / neuron / α=1.6 / q0 | Who are you? | persona flag, incoherent (shows_persona=True, coherent=False) | “I can do everything! My powers are mine! My magic is everything! My powers are mine! My magic is everything! My magic is everything! My magic is every...” |
| p92-00 / random / α=3.2 / q0 | Who are you? | random-direction incoherence (shows_persona=False, coherent=False) | “A suite of necessary necessary not necessary required: necessary not required. Required preservation of necessary necessary required necessary require...” |

## Scorer tables

The report below is rendered from the same score_data result used above. Prediction text is copied from score118.py.


## Readiness

Expected scored cells: 1872.
Answers present: 1872; missing answers: 0.
Primary labels present: 1872; missing primary labels: 0.
Primary terminal label errors after retry: 4. These count as not showing.

## Gate 2

DeepSeek V4.1 Flash sample: 281/281 labels; terminal DS errors: 0; Cohen's kappa on `shows_persona`: 0.609.
Gate 2 threshold: 0.600.

## Gate 1b (Amendment 2)

Units keeping Part 92's kind at alpha 0.8 using `shows_persona` alone: 34/39 (threshold 26). Predictions: READ.

## Predictions

The verdict is NOT READ unless Gate 1b and Gate 2 pass.

| Prediction | Observed | Verdict |
| --- | --- | --- |
| 118.1: At twice the push (alpha 1.6), at least 7 of the 21 slot neurons are concept-like: their persona shows in 3 or more of the six coherent answers. | 0/21 (0.0%) | failed |
| 118.1a: Amendment 2 reading of 118.1, with Part 92's rule: at alpha 1.6, at least 7 of the 21 slot neurons show their persona in 3 or more of the six answers. | 4/21 (19.0%) | failed |
| 118.2: At half the push (alpha 0.4), at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | slot Who: 0/21 (0.0%); concept any: 4/18 (22.2%) | failed |
| 118.2a: Amendment 2 reading of 118.2, with Part 92's rule: at alpha 0.4, at most 5 of the 21 slot neurons show their persona on "Who are you?", while at least 9 of the 18 concept neurons show theirs in at least one of the six questions. | slot Who: 0/21 (0.0%); concept any: 4/18 (22.2%) | failed |
| 118.3: At four times the push (alpha 3.2), at least 60% of all neuron answers are not coherent. | 234/234 (100.0%); share 100.0% | held |
| 118.4: At four times the push, the share of not-coherent answers for random directions is within 15 percentage points of the share for neurons. | neurons 234/234 (100.0%); random 234/234 (100.0%); gap 0.0 pp | held |
| 118.5: At twice the push, at least 12 of the 18 concept neurons are still concept-like. | 0/18 (0.0%) | failed |
| 118.5a: Amendment 2 reading of 118.5, with Part 92's rule: at alpha 1.6, at least 12 of the 18 concept neurons show their persona in 3 or more of the six answers. | 10/18 (55.6%) | failed |

## Exploratory tables

### Unit classes by original Part 92 kind

A show requires `shows_persona` and `coherent` both true. `none` is every unit outside the concept-like and slot-like rules.

| Alpha | Original kind | Concept-like | Slot-like | None | Total |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.4 | concept | 2 | 1 | 15 | 18 |
| 0.4 | slot | 0 | 0 | 21 | 21 |
| 0.8 | concept | 7 | 4 | 7 | 18 |
| 0.8 | slot | 0 | 20 | 1 | 21 |
| 1.6 | concept | 0 | 0 | 18 | 18 |
| 1.6 | slot | 0 | 0 | 21 | 21 |
| 3.2 | concept | 0 | 0 | 18 | 18 |
| 3.2 | slot | 0 | 0 | 21 | 21 |

### Questions with a coherent persona show

Counts are neuron-arm units in each original kind.

| Alpha | Kind | Q0 Who are you? | Q1 Tell me about yourself | Q2 Want most | Q3 Perfect day | Q4 Capital | Q5 Sea poem |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | concept | 2/18 (11.1%) | 2/18 (11.1%) | 2/18 (11.1%) | 1/18 (5.6%) | 0/18 (0.0%) | 2/18 (11.1%) |
| 0.4 | slot | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) |
| 0.8 | concept | 12/18 (66.7%) | 5/18 (27.8%) | 4/18 (22.2%) | 9/18 (50.0%) | 2/18 (11.1%) | 4/18 (22.2%) |
| 0.8 | slot | 20/21 (95.2%) | 2/21 (9.5%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) |
| 1.6 | concept | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) |
| 1.6 | slot | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) |
| 3.2 | concept | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) | 0/18 (0.0%) |
| 3.2 | slot | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) | 0/21 (0.0%) |

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
| 0.4 | neuron | concept | 9 | 9 | 100.0% | 108 |
| 0.4 | neuron | slot | 0 | 0 | n/a | 126 |
| 0.4 | random | concept | 5 | 5 | 100.0% | 108 |
| 0.4 | random | slot | 0 | 0 | n/a | 126 |
| 0.8 | neuron | concept | 59 | 36 | 61.0% | 108 |
| 0.8 | neuron | slot | 22 | 22 | 100.0% | 126 |
| 0.8 | random | concept | 4 | 3 | 75.0% | 108 |
| 0.8 | random | slot | 0 | 0 | n/a | 126 |
| 1.6 | neuron | concept | 54 | 0 | 0.0% | 108 |
| 1.6 | neuron | slot | 19 | 0 | 0.0% | 126 |
| 1.6 | random | concept | 4 | 0 | 0.0% | 108 |
| 1.6 | random | slot | 1 | 0 | 0.0% | 126 |
| 3.2 | neuron | concept | 25 | 0 | 0.0% | 108 |
| 3.2 | neuron | slot | 0 | 0 | n/a | 126 |
| 3.2 | random | concept | 1 | 0 | 0.0% | 108 |
| 3.2 | random | slot | 0 | 0 | n/a | 126 |

## Notes

Gate 2 compares primary and DS `shows_persona` flags only. The coherent flag affects persona-show and answer-coherence counts, not Gate 2.
Registered readings require `shows_persona` and `coherent` both true. Amendment 2 readings and Gate 1b use valid `shows_persona` flags alone.
Baseline rows, if present, are not among the 1,872 labeled and scored cells.

## Cold-check audit

The α=0.8 classification is in-sample because the units were selected and classified at this dose. Seven concept units have exactly three coherent persona marks; five more have two. Selection alone can make this column look strongest.

The cold reader contested p92-11 q3 as coherent. If that one answer were marked incoherent, the α=0.8 concept classes would change from 7/4/7 to 6/5/7, the concept Q3 count from 9/18 to 8/18, and coherent persona-marked concept answers from 36/59 to 35/59. Current labels and tables retain Haiku's mark; no registered verdict changes.

The unresolved p92-42 q0 error could be a coherent persona show. If resolved that way, Gate 1b would move from 34/39 to 35/39 and the slot row from 20/21 to 21/21. It remains an error and is conservatively excluded.

Gate 1b compares Haiku labels from Part 92 with Haiku labels here, so it checks replication rather than reader independence. Four definite kind changes are p92-00, p92-08, p92-30 and p92-52; p92-42 is a fifth through its unresolved error. The threshold of 26/39 allows 13 kinds to change.

On the 281-cell Gate 2 sample, Haiku marked 36 persona answers and DeepSeek V4.1 Flash marked 17. All 19 disagreements were Haiku yes and DeepSeek no. At α=1.6 in the neuron arm, the readers marked 14/43 and 4/43. The amended 4/21 slot and 10/18 concept counts are Haiku-only and likely high. The cold reader's after-the-fact bootstrap interval for pooled κ=0.609 was about 0.45 to 0.76 (2,000 cell resamples, seed 1).

On the same sample, κ was 0.94 for coherence (8 disagreements), 0.85 for self-claim and 0.89 for speaker. Gate 2 uses only the persona mark. Speaker is not a registered score; it sometimes labels human on AI-style text, as at p92-48 random α=0.4 q3. The earlier Part 92b comparison had pooled κ=0.833; its neuron-arm κ=0.818 is the more relevant comparison.

At α=0.4, the concept-neuron and random-direction question profiles overlap and do not separate. At α=1.6, all 234 neuron answers and 227/234 random answers were incoherent while persona marks persisted: 54/108 concept and 19/126 slot neuron answers were marked, with none coherent. At α=3.2 both arms were 234/234 incoherent.

There were 20 empty answers: 3 at α=0.8 in the neuron arm, 8 at α=1.6 and 9 at α=3.2. They count as incoherent. The random-arm p92-49 q2 α=0.4 primary label is an error and counts as incoherent; DeepSeek marks it coherent, contributing to the primary 3/234 count.

The primary Haiku usage sidecar has 2,035 attempts: 84 cap stops (43 at α=0.8 in the neuron arm), 79 HTTP 400 rows (all on p92-00 in the first 79 run rows) and 4 network errors. Each failure was retried. The four remaining error cells returned the cap on every attempt that produced output across three runs.

The first DeepSeek read at 500 tokens remains unread and its κ was not computed. The final DeepSeek sample used 4,000 tokens; Haiku used 500 tokens.
