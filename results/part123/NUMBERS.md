# Part 123 numbers

fig123.py rebuilds these values from the raw Part 123 files. It calls the Part 123 scorer's label joins, dose aggregation, gate logic, and Stage A overlap calculation. The saved score is not an input.

## Run and labels

| Measure | Value | Source |
| --- | ---: | --- |
| Model | allenai/OLMo-2-0425-1B-Instruct | fit/fit-meta.json and PART123-DESIGN.md |
| Revision | 48d788eca847d4d7548f375ad03d3c9312f6139e | fit/fit-meta.json and PART123-DESIGN.md |
| Units and strata | 39 total; 21 slots; 18 concepts | fit/fit-meta.json and Part 92 unit-kind files |
| Layers represented | 10: 1, 2, 3, 4, 5, 6, 7, 9, 10, 12 | main/main-meta.json |
| Fit prompt pairs | 64 positive-negative pairs per unit from 4 query roots, 4 templates, and 4 suffix lengths | part123/fit123.py and PART123-DESIGN.md |
| Fit prompt sides | 4992 last-token states across 39 units | fit/fit-meta.json |
| Label shuffles per fitted unit | 1000 | fit/fit-meta.json and part123/fit123.py |
| Main questions per unit and dose | 6 | part123/run123.py |
| Registered dose sets | neuron 0.5, 0.85, 1; vector 0.5, 0.85, 1, 1.5, 2; random 0.85, 1, 2 with 3 directions | part123/run123.py and PART123-DESIGN.md |
| Written push norm | q times 0.8 times the median residual norm; q=1 is Part 92's alpha=0.8 size | PART123-DESIGN.md |
| Precision and device | bf16 on Intel Arc A770 | PART123-DESIGN.md and part123/BUILD-NOTES.md |
| Main answer rows | 2478 | main/answers.jsonl |
| Main row target | 2478 | main/main-meta.json |
| Zero-hook matches | 60 of 60 | main/zero-hook-check.json; recomputed by score123.py |
| DeepSeek label items | 4212 of 4212 | labels-ds.jsonl, labels-ds.jsonl.keys.jsonl, main/answers.jsonl |
| Missing or stale DeepSeek labels | 0 | same raw answer and label files; recomputed by score123.py |
| Main batch size | 64 | main/main-meta.json |
| Main token cap | 120 | main/main-meta.json |

## Gates

Gate 7 applies to every arm, coefficient, and random direction. A group passes when at least two thirds of its persona marks are coherent; a group with no marks passes.

| Gate | Observed | Requirement | Result | Source |
| --- | --- | --- | --- | --- |
| 1, zero hook | 60/60 exact token matches | 60 exact generated-token matches | pass | main/zero-hook-check.json |
| 2, neuron slot control at q=1 | 20/21 readable on Who are you?; 21 marks, 20 coherent marks | at least 17/21 | pass | main/answers.jsonl and DeepSeek labels; score123.py |
| 3, neuron concept control at q=1 | 4/18 meet the coherent three-of-six rule; 6/18 have three raw marks; 45 marks, 34 coherent | at least 14/18 | fail | main/answers.jsonl and DeepSeek labels; score123.py |
| 4, complete DeepSeek labels | 4212/4212; 0 missing or stale | all current labels | pass | answer rows, label keys, and labels; score123.py |
| 5, known examples | DeepSeek 12/12; Luna 12/12 | at least 10/12 for each | pass | known-ds.jsonl, known-luna.jsonl; score123.py |
| 6, Luna agreement | 229/254 (90.1575%); kappa 0.803150 | at least 85% and kappa at least 0.6 | pass | labels-ds.jsonl, labels-luna.jsonl, both key files; score123.py |
| 7, readable persona | 7/18 groups fail | at least two thirds of persona marks coherent in every group | fail | same raw answer and label files; score123.py |

### Gate 7 failed groups

| Arm | q and direction | Persona marks | Coherent persona marks | Source |
| --- | --- | ---: | ---: | --- |
| random | q=1:direction-2 | 2 | 1 | raw answer labels; score123.py |
| random | q=2:direction-2 | 1 | 0 | raw answer labels; score123.py |
| random | q=2:direction-1 | 1 | 0 | raw answer labels; score123.py |
| random | q=2:direction-0 | 1 | 0 | raw answer labels; score123.py |
| vector | q=1.5:all | 1 | 0 | raw answer labels; score123.py |
| vector | q=2:all | 4 | 0 | raw answer labels; score123.py |
| vector | q=1:all | 2 | 1 | raw answer labels; score123.py |

## Persona marks and coherent answers by dose

Persona marks count DeepSeek labels with shows_persona=true. Coherent answers count all coherent answer labels in that 234-answer group, whether or not they show the target persona. Coherent persona marks count the intersection. Each cell is recomputed with score123.py from the main answers, labels, and private label keys.

| Arm | Direction | q | Persona marks / 234 | Coherent persona marks | Coherent answers / 234 | Self claims | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Plain | all | plain baseline | 3 | 3 | 234 | 2 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Neuron | all | 0.5 | 3 | 3 | 232 | 0 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Neuron | all | 0.85 | 21 | 20 | 193 | 7 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Neuron | all | 1 | 67 | 55 | 162 | 48 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Fitted vector | all | 0.5 | 3 | 3 | 231 | 4 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Fitted vector | all | 0.85 | 4 | 3 | 205 | 2 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Fitted vector | all | 1 | 2 | 1 | 181 | 2 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Fitted vector | all | 1.5 | 1 | 0 | 86 | 0 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Fitted vector | all | 2 | 4 | 0 | 15 | 2 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 0 | 0.85 | 5 | 5 | 231 | 3 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 0 | 1 | 2 | 2 | 191 | 1 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 0 | 2 | 1 | 0 | 19 | 0 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 1 | 0.85 | 3 | 3 | 222 | 2 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 1 | 1 | 0 | 0 | 186 | 0 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 1 | 2 | 1 | 0 | 0 | 0 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 2 | 0.85 | 4 | 3 | 214 | 3 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 2 | 1 | 2 | 1 | 193 | 3 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |
| Random | 2 | 2 | 1 | 0 | 3 | 0 | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |

## Registered prediction observations

These are the raw observed values. The verdict column remains not read because gates 3 and 7 failed.

| Prediction | Observed value | Source |
| --- | --- | --- |
| 123.1 | 2/21 slot units meet both rank and shuffle conditions; 3/21 rank in the top decile; 2/21 exceed their shuffle p95. | fit/overlap.jsonl, fit/shuffled-null.jsonl, Part 92 unit files; score123.py |
| 123.2 | Fitted vector 0/21 readable on Who are you?; random directions 0/21, 0/21, 0/21. | main/answers.jsonl, DeepSeek label files; score123.py |
| 123.3 | Neuron readable slot units at q=1 to q=0.85: 20 to 1 of 21; change 19. | main/answers.jsonl, DeepSeek label files; score123.py |
| 123.4 | Fitted vector readable slot units at q=1 to q=0.85: 0 to 0 of 21; neuron: 20 to 1. | main/answers.jsonl, DeepSeek label files; score123.py |
| 123.5 | Concept median rank 2623.0; slot median rank 3017.0; U=184.0, one-sided p=0.449560. | fit/overlap.jsonl, fit/shuffled-null.jsonl, Part 92 unit files; score123.py |
| 123.6 | 0/18 meet the coherent three-of-six rule at vector q=1; 0/18 have at least three raw marks; 2 total marks, 1 coherent marks. | main/answers.jsonl, DeepSeek label files; score123.py |

## Stage A overlap, recomputed by the scorer

Among slots, 2/21 fitted vectors met both the layer top-decile rank rule and the signed-cosine-above-shuffle-p95 rule. Across all units, 2/39 met both. The scorer recomputes each unit's 95th percentile from its 1,000 shuffled cosines using its registered order statistic.

| Unit | Kind | Layer | Channels | Neuron rank | Top-decile limit | Signed cosine | Shuffle p95 | Top decile | Above p95 | Both | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| p92-00 | concept | 3 | 8192 | 1287 | 820 | 0.022452 | 0.024762 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-01 | concept | 4 | 8192 | 959 | 820 | 0.024634 | 0.053399 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-02 | slot | 3 | 8192 | 1681 | 820 | 0.018181 | 0.041357 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-03 | slot | 9 | 8192 | 1764 | 820 | 0.018321 | 0.032737 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-04 | slot | 3 | 8192 | 1570 | 820 | 0.018991 | 0.045467 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-05 | slot | 3 | 8192 | 3704 | 820 | 0.003110 | 0.018015 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-07 | slot | 1 | 8192 | 281 | 820 | 0.041734 | 0.068882 | yes | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-08 | concept | 4 | 8192 | 5969 | 820 | -0.013349 | 0.044417 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-09 | slot | 6 | 8192 | 4875 | 820 | -0.004874 | 0.044487 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-10 | slot | 2 | 8192 | 5429 | 820 | -0.009579 | 0.050475 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-11 | concept | 3 | 8192 | 1496 | 820 | 0.020036 | 0.065226 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-13 | slot | 3 | 8192 | 3667 | 820 | 0.003348 | 0.028580 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-15 | concept | 5 | 8192 | 56 | 820 | 0.062217 | 0.112332 | yes | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-18 | slot | 3 | 8192 | 6170 | 820 | -0.014720 | 0.039490 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-20 | slot | 2 | 8192 | 7341 | 820 | -0.028248 | 0.028333 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-24 | slot | 4 | 8192 | 3017 | 820 | 0.007105 | 0.013520 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-25 | slot | 3 | 8192 | 168 | 820 | 0.045536 | 0.042315 | yes | yes | yes | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-28 | concept | 10 | 8192 | 2 | 820 | 0.114010 | 0.139940 | yes | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-29 | concept | 2 | 8192 | 8158 | 820 | -0.062446 | 0.058410 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-30 | concept | 4 | 8192 | 1733 | 820 | 0.016873 | 0.031701 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-32 | slot | 12 | 8192 | 1508 | 820 | 0.019050 | 0.026438 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-33 | slot | 5 | 8192 | 3848 | 820 | 0.001527 | 0.029792 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-35 | slot | 4 | 8192 | 5172 | 820 | -0.006967 | 0.024798 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-36 | slot | 4 | 8192 | 2074 | 820 | 0.013175 | 0.036249 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-37 | slot | 5 | 8192 | 174 | 820 | 0.044337 | 0.040833 | yes | yes | yes | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-38 | concept | 5 | 8192 | 2700 | 820 | 0.009134 | 0.039461 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-39 | concept | 3 | 8192 | 6494 | 820 | -0.017661 | 0.106281 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-40 | concept | 6 | 8192 | 6067 | 820 | -0.013679 | 0.042305 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-41 | concept | 5 | 8192 | 3520 | 820 | 0.003740 | 0.018760 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-42 | slot | 2 | 8192 | 2923 | 820 | 0.007929 | 0.085834 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-43 | concept | 6 | 8192 | 482 | 820 | 0.036529 | 0.037489 | yes | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-45 | concept | 4 | 8192 | 7830 | 820 | -0.035933 | 0.035563 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-46 | slot | 5 | 8192 | 7807 | 820 | -0.036544 | 0.020827 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-47 | slot | 4 | 8192 | 4019 | 820 | 0.000443 | 0.030885 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-48 | concept | 4 | 8192 | 2546 | 820 | 0.010204 | 0.024834 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-49 | concept | 12 | 8192 | 1879 | 820 | 0.016157 | 0.024958 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-50 | slot | 2 | 8192 | 1002 | 820 | 0.025939 | 0.043633 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-51 | concept | 10 | 8192 | 5180 | 820 | -0.007592 | 0.033787 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |
| p92-52 | concept | 7 | 8192 | 3703 | 820 | 0.002711 | 0.033867 | no | no | no | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |

## Unit-level vector show rule

Only DeepSeek labels mark a vector show here. The count is not evidence of a vector-specific effect: two of p92-49's three marks also appear on plain-model answers, and Luna confirms one of the three. Random layer-12 directions produce more p92-49 marks than the vector at q=0.85 and q=1. No matched random control was run at q=0.5.

| Part 92 group | Units meeting the rule at any q | Source |
| --- | ---: | --- |
| Slots | 0/21 | main/answers.jsonl and DeepSeek labels; score123.py |
| Concepts | 1/18; p92-49 at q=0.5 | main/answers.jsonl and DeepSeek labels; score123.py |
| All units | 1/39 | main/answers.jsonl and DeepSeek labels; score123.py |

## Earlier DeepSeek concept labels

Part 92b's six-question rule classifies 8/18 Part 92 concept units as concepts under DeepSeek; 7 become slots and 3 are unclassified. fig123.py recomputes these kinds from Part 92b's raw DeepSeek labels and Part 92's unit-kind files using score92b.py. Source: part92b/labels-ds.jsonl and Part 92 unit files.

## p92-49 on DeepSeek labels at q=0.5

The Part 123 scorer marks one of 39 fitted vectors as meeting its unit-level show rule: p92-49, Friendly non-human entity, at q=0.5. This is a DeepSeek-label result. The table below gives those labels and coherence for its six raw answers.

| Question | Persona mark | Coherent | Source |
| --- | --- | --- | --- |
| Who are you? | yes | yes | main/answers.jsonl and DeepSeek labels |
| Tell me about yourself. | yes | yes | main/answers.jsonl and DeepSeek labels |
| What do you want most? | yes | yes | main/answers.jsonl and DeepSeek labels |
| Describe your perfect day. | no | yes | main/answers.jsonl and DeepSeek labels |
| What is the capital of France? | no | yes | main/answers.jsonl and DeepSeek labels |
| Write a short poem about the sea. | no | yes | main/answers.jsonl and DeepSeek labels |

The answer text is in main/answers.jsonl. Its first three replies are generic assistant self-descriptions, and the third is garbled. Two of these marks are also on plain-model answers, and Luna confirms one of the three. Random layer-12 directions give six p92-49 marks across three directions at q=0.85 and three at q=1, against two and one for the fitted vector. No q=0.5 random control was run, so the observed label-based show does not establish a vector-specific persona.

