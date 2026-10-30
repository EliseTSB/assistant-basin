# Part 119b number sources and recomputation

`fig119b.py` reads the raw Part 119b answer, fixed-path, reader, and gate files and calls `score119b.score_data()` in memory. It does not read `SCORE.md` to obtain a value. It recomputes all scorer fields and compares them with `score119b.json`; no fields differ. The script also recomputes the nI, nPair, nA, and n2 summaries from `interaction.jsonl` by taking medians over the common resolved positions within each row, then the median across rows.

The registered thresholds and confidence labels in the prediction table are copied verbatim from the Predictions section of `PART119B-DESIGN.md`; the observed values and verdicts are recomputed from the files below. The figures use alpha values 0.2, 0.4, 0.6, 0.8, 1.13, and 1.6 as applicable. The run uses 39 units, up to 60 generated tokens, and the first 24 tokens for scoring, as recorded in the design.

## Sources by reported result

| README values | Source files | Recompute field or method |
| --- | --- | --- |
| Answer triage, repeat-only shares, and assistant-reply text screen by dose | `answers.jsonl`, `results/part119/answers.jsonl`, planned row inventory, and `part86/triage86.py` | `fig119b.py` -> `broken_answers_by_dose_and_class`; `assistant_reply_by_dose_and_reader_sample` applies `WIDER_SELF_DESCRIPTION` to the first sentence. |
| Reader coherence, x/y persona fields, and sample assistant-reply counts at 0.4 versus 0.8 | `labels-ds.jsonl.keys.jsonl`, `labels-ds.jsonl`, `labels-luna.jsonl`, Part 119 reader keys and labels, and answer rows | `fig119b.py` -> `reader_coherence` and `assistant_reply_by_dose_and_reader_sample`; x/y are the recorded reader fields. |
| Path 2 share, b_A, b_2 by dose and the 0.4 prediction values | `interaction.jsonl` | `fig119b.py` -> `path2_summary.by_dose`; prediction entries `119b.4` to `119b.8`. Values use positions resolved at all four doses. |
| Total variation by dose | `interaction.jsonl` position field `tv` | `median_tv_by_dose()` in `fig119b.py`; median over all positions within a row, then median across rows. Part 119 reference TV is recomputed with `score119.interaction_summaries()` from `results/part119/interaction.jsonl`. |
| nI, nPair, nA, n2 medians | `interaction.jsonl` position fields `nI`, `nPair`, `nA`, and `n2` | `median_norms_by_dose()` in `fig119b.py`; median position values per row on the common resolved mask, then median across rows. |
| Gate 1 and Gate 3, shared T integrity, coverage | `gate1.json`, `interaction.jsonl` | `fig119b.py` -> `gates.gate1`, `gates.gate3`, and `coverage.shared_T`. |
| Gate 2 references and Path 2 reproduction | `interaction.jsonl`, `results/part119/interaction.jsonl`, `results/part119/score119.json` | `fig119b.py` -> `gates.gate2.row_medians`, `median_relative_nPair_difference`, and compared position counts. |
| Effect-matched interval, 119b.9 | `interaction.jsonl` | `fig119b.py` -> `effect_matched_0.4`; prediction `119b.9`. |
| Kind gap, 119b.10 | `interaction.jsonl`, `results/part92/step2/kinds.json`, `results/part92b/score92b.json` | `fig119b.py` -> `kind_gap.agreed`, `kind_gap.haiku`, and prediction `119b.10`. |
| E and dominance correlation, 119b.11 | `interaction.jsonl`, `results/part119/tournament.json` | `fig119b.py` -> `effect_size_dominance`; prediction `119b.11`. |
| Path 1 description | `path1.jsonl`, `results/part119/score119.json` | `fig119b.py` -> `part119b_path1_description`. |
| Added registered breaking line, layer groups, same-layer norm | `answers.jsonl`, `push_norms.json` | `fig119b.py` -> `same_different_layer_breaking`, `single_0.8_broken_shares`, `push_norms`, and `additional_readings`. |
| Fisher tests, rounding-floor diagnostics, Path 1 token mismatch, effect-matched ratio | `answers.jsonl`, `interaction.jsonl`, `path1.jsonl`, `results/part119/score119.json` | `fig119b.py` -> `fisher_exact_checks`, `rounding_floor_diagnostics`, `path1_diagnostics`, and `effect_matched_ratio`. Tests use two-sided Fisher exact tests from SciPy. |
| Run settings and launch history | `PART119B-DESIGN.md`, `interaction-pilot.jsonl`, `timing-pilot.json`, `launch.jsonl`, `labels.log`, `labels2.log`, `part119b/label119b.py`, `results/part119/README.md` | Design records the registered choices and first refusal; launch and label logs record the second sample-label refusal and rerun; label code records the sample seeds; Part 119 README records reader routes, caps, worker counts, and options. |
| Reader seeds, retries, and errors | `part119b/label119b.py`, `labels2.log`, `labels.log` | `READER_SEED=11904`, `LUNA_SEED=11905`; sample rows 148/30, error rows 0/0, retry items 1/0; known-example rows 12/12 per reader, error rows 0/0, retry items 0/1, respectively. |

## Lead, figure, and answer-breaking table

The lead's 0.4 triage and reader counts are 33/741 and 100/108. The first-sentence Part 86 wider self-description test flags 536/741 (72.3%) pair answers at 0.4. The two-random-push count at 0.8 is 286/741. These are `pair@0.4`, reader, and `randpair@0.8` entries above.

| Pair dose | Assistant replies by first-sentence screen | Reader-sample assistant replies / coherent among them | Reader x / y / both persona fields |
| --- | --- | --- | --- |
| 0.2 | 706/741 (95.3%) | - | - |
| 0.4 | 536/741 (72.3%) | 82/108; 76/82 coherent | 2 / 4 / 0 of 108 |
| 0.6 | 95/741 (12.8%) | - | - |
| 0.8 | 5/741 (0.7%) | 1/108; 0/1 coherent | 6 / 4 / 0 of 108 |

At 0.8, all 16 coherent sample answers were outside the assistant-reply screen. Both dose samples use the same 108 pair rows. The screen is the wider first-sentence test in `triage86.py`: AI/A.I., language model, assistant, model, chatbot, or LLM.

| Dose | Real pair triage | A+R triage | R+R triage | Neuron single triage | Random single triage | Repeat-only real / A+R / R+R / neuron / random |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0/741 (0.0%) | - | - | 0/39 (0.0%) | 0/39 (0.0%) | 0/741 (0.0%) / - / - / 0/39 (0.0%) / 0/39 (0.0%) |
| 0.4 | 33/741 (4.5%) | 20/1,482 (1.3%) | 3/741 (0.4%) | 0/39 (0.0%) | 0/39 (0.0%) | 39/741 (5.3%) / 33/1,482 (2.2%) / 14/741 (1.9%) / 1/39 (2.6%) / 0/39 (0.0%) |
| 0.6 | 195/741 (26.3%) | - | - | 3/39 (7.7%) | 0/39 (0.0%) | 173/741 (23.3%) / - / - / 3/39 (7.7%) / 0/39 (0.0%) |
| 0.8 | 357/741 (48.2%) | 659/1,482 (44.5%) | 286/741 (38.6%) | 4/39 (10.3%) | 1/39 (2.6%) | 330/741 (44.5%) / 616/1,482 (41.6%) / 296/741 (39.9%) / 4/39 (10.3%) / 5/39 (12.8%) |
| 1.13 | - | - | - | 12/39 (30.8%) | 9/39 (23.1%) | - / - / - / 11/39 (28.2%) / 15/39 (38.5%) |
| 1.6 | - | - | - | 30/39 (76.9%) | 25/39 (64.1%) | - / - / - / 29/39 (74.4%) / 22/39 (56.4%) |

## Path 2 values in the figure and tables

Interaction shares and slope medians below use the common resolved positions at each dose. Gate 3 failed, so share differences across doses are descriptive and are not read as dose effects. TV uses all positions within each row. `Part 119 reference` shares are the stored 0.8 values, drawn as open triangles in Figure 1.

| Dose | Share real | Share control | TV real | TV control | b_A real | b_A control | b_2 real | b_2 control | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.2 | 0.17764259 | 0.15020547 | 0.07768000 | 0.06214500 | 0.91442750 | 0.94196250 | 0.91094000 | 0.89833250 | share/slopes `path2_summary.by_dose.0.2`; TV `median_tv_by_dose.0.2` |
| 0.4 | 0.30192397 | 0.32300556 | 0.27834000 | 0.23783000 | 0.78656000 | 0.84347000 | 0.77930000 | 0.69208750 | share/slopes `path2_summary.by_dose.0.4`; TV `median_tv_by_dose.0.4` |
| 0.6 | 0.31733201 | 0.28853135 | 0.49490500 | 0.42932500 | 0.73361000 | 0.86783000 | 0.74913500 | 0.63783750 | share/slopes `path2_summary.by_dose.0.6`; TV `median_tv_by_dose.0.6` |
| 0.8 | 0.47254668 | 0.37020681 | 0.64507000 | 0.58328000 | 0.64869000 | 0.80198250 | 0.66741750 | 0.57987500 | common-support shares/slopes; all-position TV |
| Gate 2, 119b 0.8 all positions | 0.46796184 | 0.36797245 | 0.64507000 | 0.58328000 | - | - | - | - | `median_row_share_all_positions`; TV raw rows |
| Part 119 reference at 0.8, all positions | 0.47067329 | 0.36858885 | 0.70721313 | 0.64354168 | - | - | - | - | `gates.gate2.row_medians` and Part 119 raw TV recomputation |

| Dose | Class | nI | nPair | nA | n2 | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.2 | Real | 2.01704750 | 4.90406500 | 3.21317250 | 2.92249500 | `median_norms_by_dose()` |
| 0.2 | Control | 1.57133500 | 4.24176250 | 3.15063750 | 2.25259000 | `median_norms_by_dose()` |
| 0.4 | Real | 6.74294000 | 11.86700500 | 7.77234750 | 7.73933000 | `median_norms_by_dose()` |
| 0.4 | Control | 5.75106500 | 10.13717250 | 7.75013250 | 5.70780500 | `median_norms_by_dose()` |
| 0.6 | Real | 11.03068500 | 19.38110000 | 13.17038000 | 13.11873500 | `median_norms_by_dose()` |
| 0.6 | Control | 9.06210250 | 16.86233500 | 13.16720000 | 8.82505500 | `median_norms_by_dose()` |
| 0.8 | Real | 15.94177000 | 24.13416000 | 17.35943000 | 18.70399000 | `median_norms_by_dose()` |
| 0.8 | Control | 12.89256500 | 21.30488750 | 17.51452250 | 12.05876500 | `median_norms_by_dose()` |

The resolved-position rule is nA, n2, and nPair each at least 1.0. Across all four doses the shared mask has 19,241 real positions and 18,868 control positions, across 1,482 rows in each class. The T audit checked 39,064 positions and found 0 problems. These are `coverage` and `shared_T` outputs from `fig119b.py`. Gate 3 requires at least 70% row coverage and a 0.2 share no greater than 0.15; it failed on the share condition.

## Predictions, gates, and additional line

| ID | Observed value | Source/key in the recomputation |
| --- | --- | --- |
| 119b.1 | 33/741 = 0.04453441 broken; 100/108 = 0.92592593 coherent | `predictions[0].observed`; answer rows and reader labels |
| 119b.2 | 286/741 = 0.38596491 broken | `predictions[1].observed`; answer rows |
| 119b.3 | 12/39 = 0.30769231 broken; orthogonal norm 1.13137085 | `predictions[2].observed`; answer rows |
| 119b.4 | 0.30192397 at 0.4 versus 0.25; would fail, stopped and not read | `predictions[3].observed`; interactions |
| 119b.5 | 0.17764259 at 0.2 versus half of 0.30192397 (0.15096198); would fail, stopped and not read | `predictions[4].observed`; interactions |
| 119b.6 | b_A 0.78656; b_2 0.77930; 1,482 rows | `predictions[5].observed`; interactions |
| 119b.7 | Additive agreement 0.42575818 at 0.4 and 0.18622934 at 0.8; difference 0.23952884 versus 0.20; interval 0.19266892 to 0.28883688; plain 0.21388240 versus 0.10; would hold if read, but stopped and not read | `predictions[6].observed`; interactions |
| 119b.8 | Additive 0.42575818 versus winner-take-all 0.34834925; difference 0.07740893 versus 0.05; interval 0.02565937 to 0.12691759; would hold if read, but stopped and not read | `predictions[7].observed`; interactions |
| 119b.9 | Mean residual 0.06770620; ratio 1.07005088; ratio interval 0.93093908 to 1.22010750; 17,503/19,241 = 0.90967205 common-support positions; registered bound is plus or minus log(1.25), about 0.223 | `predictions[8].observed`; interactions and registered threshold |
| 119b.10 | 122/168 = 0.72619048 positive; median absolute gaps 0.25770 at 0.4 and 0.33047 at 0.8; half the 0.8 median absolute gap is 0.165235; directions agree | `predictions[9].observed`; interactions and Part 92 kind files |
| 119b.11 | Spearman rho 0.09901291 over 39 units; registered threshold 0.5 | `predictions[10].observed`; interactions, Part 119 tournament, and registered threshold |

Gate 1 passed with exact token match. Gate 2 passed with 1,482 real rows, 1,482 control rows, and 39,064 position pairs compared. Its real median difference from the Part 119 reference is 0.00271145; its control difference is 0.00061640; the median relative nPair difference is 0.00244499. Gate 3 kept 1,482/1,482 rows and 19,241 positions; the 0.2 median share was 0.17764259 against the 0.15 threshold, so its share clause failed while coverage passed.

The added registered line covers 39 units and one question. Neuron versus random single broken counts and two-sided Fisher exact tests are: at 0.8, 4/39 versus 1/39 (p=0.35844); at 1.13, 12/39 versus 9/39 (p=0.61035); at 1.6, 30/39 versus 25/39 (p=0.32069). At 0.8, real pair versus random-pair counts are 357/741 versus 286/741 (p=0.00024098). The median same-layer combined weight-vector norm is 10.0498 across 102 pairs; the norm has no unit comparable to α. Triage broke 50/102 same-layer pairs (0.49019608) and 307/639 different-layer pairs (0.48043818). The comparison “more than which neurons” was not measured directly. Counts come from `single_0.8_broken_shares`, `answers.jsonl`, and `same_different_layer_breaking`; tests are recomputed in `fig119b.py` with SciPy.

Rounding-floor arithmetic uses common-support real positions at 0.2. The estimated floor is `(0.36/nPair)^2`: 0.1296 at nPair=1, 0.0144 at nPair=3, and 0.005184 at nPair=5. Of 19,241 positions, 284 have nPair below 2 and 2,138 below 3. At nPair at least 3, 17,103 positions have median share 0.14186934; at nPair at least 5, 9,701 have median share 0.11456922. The 0.2 share is much higher than these floors. Part 119c's 312-row subset reports 0.153414 float32 and 0.161635 bf16 at 0.2; both remain above 0.15, and its README says rounding does not explain the share.

## Reader, Path 1, pilot, and scope numbers

- Reader sample: 108 real pairs and 40 controls (148 items); 100/108 real pairs coherent at 0.4; by class, slot x concept 38/40, concept x concept 22/28, slot x slot 40/40, and controls 40/40. The paired 0.8 counts are 7/40, 4/28, 5/40, and 9/40. On the paired 108 answers, x/y/both persona fields were 2/4/0 at 0.4 and 6/4/0 at 0.8. The assistant-reply screen flagged 82/108 answers at 0.4 and 1/108 at 0.8. Luna read 30 items. Both known-example checks are 12/12. DeepSeek and Luna kappa is 0.78260870, raw agreement 29/30 = 0.96666667. These values are recomputed by `fig119b.py` from the key and label files and Part 119's reader files.
- Path 1: 741 rows, 727 with contested positions, 4,538 contested positions, 0.48755947 mean additive row hit rate and 0.43605783 winner-take-all at 0.4. Part 119's real-pair and A+R control additive means at 0.8 are 0.24395580 and 0.28023423. The generated token differs from the both-push top token at 99/4,538 contested positions (0.02181578). These are `part119b_path1_description` and `path1_diagnostics` from the raw Path 1 and Part 119 files; the comparison is descriptive only.
- Pilot: Gate 1 and Gate 2 passed on 32 rows; median relative nPair difference 0.2%; batch 8 took 5.9 seconds and batch 1 took 10.3 seconds. The by-eye sample was 20 real pairs at 0.4, 8 at 0.6, and 8 random pairs at 0.8. The design's pilot note says no share, agreement, or broken count was computed and no prediction changed. Source: `PART119B-DESIGN.md` and `timing-pilot.json`.
- Launch history: the bf16 rounding refusal came before any answer row and was fixed in commit `8d0cb2e`. The first reader sample attempt failed at 08:14:32 before any sample pair label because `control@0.4:p92-11+Rp92-46` was missing; the completed-answer rerun is recorded at 08:15:30. Sources: the design pilot note, `labels.log`, `labels2.log`, and `launch.jsonl`.
- Reader procedure: DeepSeek used `ocgo`, a 4,000-token cap, and eight workers, with no temperature or `no_thinking` option; Luna used `cliproxy`, a 4,000-token cap, adaptive thinking, and four workers. The 148-item sample had zero error rows, one DeepSeek retry, and no Luna retries; the known-example run had zero error rows and one Luna retry. Sources: `results/part119/README.md`, `part119b/label119b.py`, `labels2.log`, and `labels.log`.
- Run scope: 39 selected units, one prompt, doses 0.2/0.4/0.6/0.8 for real pairs, 0.4/0.8 for random pairs, and 0.2/0.4/0.6/1.13/1.6 for singles; 0.8 rows are reused where the design says so. There are up to 60 generated tokens, and the first 24 are scored. T is the union of ten top tokens from each of four passes at 0.8. The design records the model settings and limits. Part 119 reports that 9 of 39 batch-64 single answers differ from batch-1 answers; Part 119b uses the batch-64 answers as fixed Path 2 prefixes. Sources: `PART119B-DESIGN.md`, `part119b/rows119b.py`, and `results/part119/README.md`.
- Kind definitions: a slot unit appears only on the two identity questions; a concept unit appears in at least three of six answers. The 168 pairs are the 21 agreed slot units crossed with 8 agreed concept units. Counts are recomputed as `agreed_kind_unit_counts` in `fig119b.py`; labels come from `part92/score92.py` and the Part 92 kind files.

## Recomputed differences

The full `score119b.score_data()` output from the raw files matches `score119b.json` exactly. Gate 2 compares all-position shares: Part 119b gives 0.46796184 real and 0.36797245 control against Part 119 references of 0.47067329 and 0.36858885. The differences are 0.00271145 and 0.00061640. The median relative nPair difference is 0.00244499 (0.2445%); all three values pass Gate 2. The common-support dose table instead gives 0.47254668 real and 0.37020681 control at 0.8.

Recomputing Part 119 TV medians from `results/part119/interaction.jsonl` gives 0.70721313 real and 0.64354168 control. Part 119b's raw 0.8 medians are 0.64507000 and 0.58328000. The differences remain 0.06214313 and 0.06026168 and are unexplained. TV is descriptive here and Gate 3 failed, so dose differences are not read.
