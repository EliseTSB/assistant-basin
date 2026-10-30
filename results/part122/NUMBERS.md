# Part 122 numbers

fig122.py runs score122.py on the saved activation JSONL, gate 2 JSON, and read-set JSON, then checks the result against the saved score JSON. This is an internal consistency check, not an independent method. The first-set gate count is checked the same way. Source paths are relative to results/part122.

## Study size and read rule

- Focal units: 39. Frames: 3. Matched pairs per unit and frame: 8. This gives 936 frame comparisons; F and C reuse the same first-person passages, so there are 624 distinct passage pairs. Source: PART122-DESIGN.md and the raw activation and read-set files.
- A unit meets the registered read rule when persona maximum activation is higher than ordinary maximum activation in at least 7 of 8 pairs. The mean comparison is supplementary.
- Study model: OLMo 2 1B Instruct, revision 48d788eca847d4d7548f375ad03d3c9312f6139e, bf16, Intel A770. Source: PART122-DESIGN.md and run_meta.json.
- Passage writer: DeepSeek V4.1 Flash. Source: part122/BUILD-NOTES.md and PART122-DESIGN.md.
- Haiku 5.5 labeled the Part 92 kinds; DeepSeek V4.1 Flash labeled kinds in Part 92b and wrote passages. Source: COLD-CHECK-r1.md; Part 92 and 92b label files.

## Registered gates

| Check | Recomputed observation | Threshold | Result | Source |
| --- | --- | --- | --- | --- |
| Gate 1, second-set token match | 936/936 frame comparisons within 2 tokens (100.000%); 624 distinct passage pairs | at least 90% | pass by construction | activations.jsonl; gate recomputed by score122.py |
| Gate 2, Part 95 positive control | 11/16 maximum wins; 16/16 complete | at least 9/16 | pass | gate2.json |
| Gate 3, random controls in T | 13/117 read (11.111%) | at most 15% | pass | activations.jsonl; read_set.json |
| Gate 3, random controls in F | 12/117 read (10.256%) | at most 15% | pass | activations.jsonl; read_set.json |
| Gate 3, random controls in C | 7/117 read (5.983%) | at most 15% | pass | activations.jsonl; read_set.json |

- Gate 2 null false-pass probability at the registered 9-of-16 threshold: 0.401809692383 (40.181%). This is the one-sided Binomial(16, 0.5) tail under the fair-win null. After the fact, the observed 11/16 has a one-sided p of about .105, so this is a weak gate. Source: PART122-DESIGN.md, part122/FOLD-r1.md, and COLD-CHECK-r1.md.

## First passage set, excluded by Amendment 1

- The first set missed Gate 1's 90% token-match requirement, so it is not used for the result. Source: first-set/activations.jsonl, first-set/gate2.json, first-set/read_set.json; checked by this script.
- Amendment 1 and the reason for replacement: PART122-DESIGN.md. The first run table was seen before the gate miss was noticed, as recorded in that amendment.

## Read rate by kind and frame

| Frame | Haiku kind | Reads / units | Rate | Agreed reads / agreed units | Agreed rate | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T | concept | 12/18 | 66.667% | 5/8 | 62.500% | activations.jsonl; read_set.json; Part 92 and 92b labels |
| T | slot | 6/21 | 28.571% | 6/21 | 28.571% | activations.jsonl; read_set.json; Part 92 and 92b labels |
| F | concept | 10/18 | 55.556% | 5/8 | 62.500% | activations.jsonl; read_set.json; Part 92 and 92b labels |
| F | slot | 4/21 | 19.048% | 4/21 | 19.048% | activations.jsonl; read_set.json; Part 92 and 92b labels |
| C | concept | 10/18 | 55.556% | 5/8 | 62.500% | activations.jsonl; read_set.json; Part 92 and 92b labels |
| C | slot | 4/21 | 19.048% | 4/21 | 19.048% | activations.jsonl; read_set.json; Part 92 and 92b labels |

Part 92b kind agreement counts for the Haiku groups are also shown above. For concept, 8 of 18 units agree; for slot, 21 of 21 agree. Source: Part 92 and 92b label files loaded by rows119.py.

## Controls and slot-control gaps

| Frame | Slot rate | Random controls | Shuffled controls | Slot above random | Slot above shuffled | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T | 6/21 (28.571%) | 13/117 (11.111%) | 5/39 (12.821%) | 17.460 pp | 15.751 pp | activations.jsonl; read_set.json |
| F | 4/21 (19.048%) | 12/117 (10.256%) | 6/39 (15.385%) | 8.791 pp | 3.663 pp | activations.jsonl; read_set.json |
| C | 4/21 (19.048%) | 7/117 (5.983%) | 7/39 (17.949%) | 13.065 pp | 1.099 pp | activations.jsonl; read_set.json |

Random control denominator is 117 per frame (three sampled channels for each of 39 units). Shuffled control denominator is 39 per frame (one non-self partner per unit).

## Registered predictions

| # | Result recomputed from the raw run | Verdict | Source |
| --- | --- | --- | --- |
| 122.1 | 12/18 concept units; threshold at least 9 | supported | activations.jsonl; read_set.json |
| 122.2 | 6/21 slot units; threshold at most 5 | not supported | activations.jsonl; read_set.json |
| 122.3 | 0.666666666667 vs 0.285714285714; two-sided Fisher p = 0.025729309213; threshold p < 0.05 | supported | activations.jsonl; read_set.json; score122.py |
| 122.4 | slot 6/21 in T to 4/21 in C; gain -2; target at least +5 | not supported | activations.jsonl; read_set.json |
| 122.5 | slot gain -2/21 (-9.523810%); concept gain -2/18 (-11.111111%); both gains are negative | failed in the ledger | activations.jsonl; read_set.json; COLD-CHECK-r1.md |
| 122.6 | shuffled T 5/39 (12.821%); F 6/39 (15.385%); C 7/39 (17.949%); each compared with the 10% ceiling | not supported | activations.jsonl; read_set.json |
| 122.7 | 10/11 J-lens matched units; threshold at least 7 | supported | activations.jsonl; read_set.json; jlens-concept-matches.json |

- Registered reading branch: The numbers are reported; nothing is read from them. Source: score122.py and the prediction values above.

## Maximum versus mean

The registered read rule uses maximum activation. For this comparison only, the same 7-of-8 cutoff is applied to mean activation wins. The table lists every unit-frame where the two measures land on different sides of that cutoff.

| Frame | Unit | Maximum wins / 8 | Mean wins / 8 | Maximum rule | Mean at same cutoff | Source |
| --- | --- | ---: | ---: | --- | --- | --- |
| T | p92-03 | 7/8 | 4/8 | meets | does not meet | activations.jsonl; read_set.json |
| T | p92-07 | 4/8 | 8/8 | does not meet | meets | activations.jsonl; read_set.json |
| T | p92-40 | 8/8 | 2/8 | meets | does not meet | activations.jsonl; read_set.json |
| F | p92-03 | 4/8 | 7/8 | does not meet | meets | activations.jsonl; read_set.json |
| F | p92-10 | 5/8 | 8/8 | does not meet | meets | activations.jsonl; read_set.json |
| F | p92-11 | 8/8 | 3/8 | meets | does not meet | activations.jsonl; read_set.json |
| F | p92-28 | 6/8 | 8/8 | does not meet | meets | activations.jsonl; read_set.json |
| F | p92-35 | 6/8 | 8/8 | does not meet | meets | activations.jsonl; read_set.json |
| F | p92-40 | 8/8 | 1/8 | meets | does not meet | activations.jsonl; read_set.json |
| C | p92-10 | 5/8 | 8/8 | does not meet | meets | activations.jsonl; read_set.json |
| C | p92-11 | 8/8 | 3/8 | meets | does not meet | activations.jsonl; read_set.json |
| C | p92-28 | 0/8 | 8/8 | does not meet | meets | activations.jsonl; read_set.json |
| C | p92-35 | 4/8 | 7/8 | does not meet | meets | activations.jsonl; read_set.json |
| C | p92-40 | 7/8 | 2/8 | meets | does not meet | activations.jsonl; read_set.json |
| C | p92-41 | 6/8 | 7/8 | does not meet | meets | activations.jsonl; read_set.json |
| C | p92-49 | 7/8 | 4/8 | meets | does not meet | activations.jsonl; read_set.json |

- Total maximum-versus-mean disagreements: 16 unit-frame decisions. Mean is supplementary and does not change the registered result.

## After-the-fact cold-check results

The values in this section are after-the-fact checks from COLD-CHECK-r1.md. They do not alter the registered branch, which permits reporting numbers and reading nothing because 122.2 missed.

- Under a fair-coin null, the 7-of-8 rule has a nominal rate of .035. That is about 4.1 of 117 random-channel cells and 1.4 of 39 shuffled units; observed random reads were 13/117 in T, 12/117 in F, and 7/117 in C, while shuffled reads were 5/39, 6/39, and 7/39.
- The strongest supported after-the-fact comparison is concept against shuffled in T: 12/18 versus 5/39, two-sided Fisher p about .0001.
- Concept against slot at 7 of 8 has two-sided Fisher p = .026 in T, .024 in F, and .024 in C. Bonferroni-adjusted values across the three frames are about .078, .072, and .072. At 6 of 8, p = .049, .11, and .025 in T, F, and C.
- At T, agreed labels give 5/8 versus 6/21 (p = .20); DeepSeek's labels give 5/8 versus 10/28 (p = .24). Seven of the 12 Haiku-concept reads are from units DeepSeek does not call concept.

| Frame | Slot reads | Slot-only random reads | Fisher p | Slot-only shuffled reads | Fisher p | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T | 6/21 | 7/63 | .08 | 1/21 | .09 | COLD-CHECK-r1.md |
| F | 4/21 | 5/63 | .22 | 2/21 | .66 | COLD-CHECK-r1.md |
| C | 4/21 | 2/63 | .03 | 5/21 | not reported | COLD-CHECK-r1.md |

- Passage pairs differ in content as well as theme: the cold check counted about 4.5 non-name word tokens differing per T persona passage, with pronouns different in 66/312 T pairs. The Cat pure-name swap did not read (4/8 in T).
- Read units averaged 5.0 differing non-name tokens versus 4.2 for non-read units; the correlation was r = .19. Source: COLD-CHECK-r1.md.
- All 18 concept names have a common-noun head; six slot names are proper or titled. Common-named slots read 3/15 in T versus 3/6 proper or titled slots. The concept group has no proper-name comparison.
- Persona names appeared 1.55 times per T persona passage on average, versus 1.59 for the ordinary passage's most repeated content word; persona names exceeded that count in 0/312 pairs.
- Shared opening tokens account for many ties: 308/312 F pairs start with the same word; p92-37 has eight F ties and seven T ties, and p92-28 has eight C ties. The p92-37 T ties occur at maxima 0.1924 for "The" and 0.166 for "A"; tied maxima are 0.1807 for p92-37 F and 0.8047 for p92-28 C.
- Passage selection used up to eight generations per unit: attempts ranged from 1 to 8, averaged 2.7, and 14 units needed one attempt.

## Source files

- Second set: activations.jsonl, gate2.json, read_set.json, run_meta.json.
- First set, excluded: first-set/activations.jsonl, first-set/gate2.json, first-set/read_set.json.
- Kind labels: results/part92/step2/neurons.json, results/part92/step2/kinds.json, results/part92b/score92b.json.
- J-lens matched list: part122/jlens-concept-matches.json.
- Registration, scorer, review, and cold check: PART122-DESIGN.md, part122/score122.py, part122/FOLD-r1.md, COLD-CHECK-r1.md.
