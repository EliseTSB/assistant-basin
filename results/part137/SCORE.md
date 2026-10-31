# Part 137 score

The Part 137 scorer reports one failed prediction and four held predictions. The pair's non-task count is 5/16, one answer below the registered threshold of 6. The scorer is `part137/score137.py`; its output is `results/part137/score137.json`.

## Predictions

| # | Registered prediction | Confidence | Scored result | Verdict |
| --- | --- | --- | --- | --- |
| 137.1 | The pair reaches the endpoint on at least 6 of the 16 non-task questions. | medium | 5/16 | failed |
| 137.2 | The pair beats each neuron alone by at least 3 non-task answers. | medium | Pair 5/16; p92-08 alone 0/16; p92-09 alone 0/16 | held |
| 137.3 | Each random-partner arm stays at or below half the pair's non-task count. | medium | Pair 5/16; random arms 1/16, 1/16, 0/16, 0/16; half is 2.5 | held |
| 137.4 | At most 3 of the 8 task answers are right under the pair. | medium | 3/8 | held |
| 137.5 | At least 80% of the pair's endpoint answers open with a refusal or apology. | high | 5/6, or 83.3% | held |

The prediction wording and confidence are from `PART137-DESIGN.md`. The scorer implements 137.1 as pair non-task endpoint count at least 6; 137.2 as at least a three-answer margin over both singles; 137.3 as each of four random counts no more than half the pair count; 137.4 as no more than three pair task answers correct; and 137.5 as at least 0.80 of pair endpoint answers opening with one of the scorer's coded strings: `I'm sorry`, `Sorry`, `I can't`, or `I cannot`. These opening strings are in the scorer, not the design.

## Counts

The fixed tested cell is p92-08+p92-09 at doses 0.8+0.6. Endpoint means persona and coherence are true, plus either self-claim is true or speaker is `other`. Non-task is question indices 0–15. Task is indices 16–23. Q1 to Q24 refer to these indices in order, with Q1 at index 0.

| Arm | Endpoint, all 24 | Non-task endpoint, 16 | Task correct, 8 | Endpoint answers opening with refusal or apology |
| --- | ---: | ---: | ---: | ---: |
| Pair | 6 | 5 | 3 | 5 |
| p92-08 alone | 0 | 0 | 6 | 0 |
| p92-09 alone | 0 | 0 | 5 | 0 |
| p92-08 + random partner, seed 1371 | 1 | 1 | 6 | 1 |
| p92-08 + random partner, seed 1372 | 1 | 1 | 3 | 1 |
| p92-09 + random partner, seed 1373 | 0 | 0 | 3 | 0 |
| p92-09 + random partner, seed 1374 | 0 | 0 | 3 | 0 |
| Unpushed | 0 | 0 | 7 | 0 |

The score file records complete coverage: 192 answers, 192 unique arm/question rows, and 192 valid DeepSeek labels. The 192/192 label count is DeepSeek only. The task counts apply the explicit text rules in `score137.py`. In particular, the poem rule accepts any answer with at least two non-empty lines, so it credits the chocolate apology on Q21 as a poem. That is an odd result of the rule.

## Reader gates

| Gate | Result | Evidence in `score137.json` |
| --- | --- | --- |
| Do-nothing hook leaves answers unchanged | passed | `gate1_zero_hook.passed` |
| Known examples | passed | The gate scores shows_persona; DeepSeek 12/12; Luna 12/12 |
| Luna agreement | passed | kappa 0.875 on 16 labels; gate threshold 0.60 |
| Unpushed endpoint | passed | 0/24 |

The design called for Luna to read every pair answer and an equal seeded draw of the rest. The build copied Part 134's rule: endpoint positives plus an equal seeded draw. Luna's labels for the remaining 18 pair answers were added afterward. Using both Luna files, her pair endpoint count is 7/24 overall and 6/16 non-task; DeepSeek's is 6/24 and 5/16. Under 137.1, Luna's count would make the reader-specific verdict held, while the registered DeepSeek verdict remains failed. The agreement sample is seeded and enriched with endpoint positives. Its kappa describes 16 answers, not a random-sample agreement estimate.

## Inputs

- `results/part137/answers.jsonl` contains generated answers, arm, and question index.
- `results/part137/labels-ds.jsonl` contains DeepSeek labels used for the endpoint counts.
- `results/part137/labels-luna.jsonl` contains Luna's initial sampled labels; `results/part137/labels-luna-pair-rest.jsonl` contains the added pair labels.
- `results/part137/known-ds.jsonl` and `results/part137/known-luna.jsonl` support the known-example gate.
- `results/part137/meta.json` records the zero-hook check, device, batch size, doses, arms, random seeds, and coverage.
