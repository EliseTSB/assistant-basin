# Part 201 write-up notes (worker-01, 2026-11-10)

## SUMMARY

- Wrote `results/part201/README.md` (15.4 KB) from the template: title sentence, figure, three sentences of what was asked
  with the main number beside its control and the scope, the Predictions table with all eleven registered predictions
  verbatim with confidence and verdict (201.1 to 201.10 held, 201.11 failed), seven skeptical questions each answered by
  its test, Method, Limits, Files and the cold-check line the brief specified.
- Wrote `results/part201/README-entry.md`, one paragraph in the Part 200 entry's style, numbers and verdicts first.
- Wrote `part201/fig201.py`, which draws `results/part201/fig201.png` from `score.json` only: left, share of differing
  rows per pair with a 95% Wilson interval computed in the script (checked against the scorer's, same to 4 decimals);
  middle, A/M label flips by direction; right, pass-A margin in bf16 steps at the first difference for A/C and A/F8.
  Looked at the PNG after each run.
- Every number in the README was taken from `score.json`, `report.md` or the cold check, or recomputed from
  `rows.jsonl` for the position-0 paragraph.

## FINDINGS

1. **Position-0 rows (16 of 58 in A/F8, 8 of 31 in C/F1, 6 of 147 in A/M) are exact bf16 ties.** Pass A records
   top1 9419 and top2 4888, both at a logit of 23.25, and emits 4888, the lower id, as at all 2,240 ties. Float32 puts
   9419 ahead by 0.016. I first filed this as an alert (papercuts alert 0049) saying G2 should have failed and the step
   unit was half the design's. Parent control message seq 1 refuted both, and I verified both refutations in the data
   (`emitted_logit` equals `top1_logit` on all 24 such rows; `bf16_step(23.27) = 0.125`). The README now states it as an
   exact tie, and the alert's claims are removed.
2. **201.6's verdict rests on one label.** Only broken reaches 8 flips (8 and 8); name gone (1 and 4), maker gone (4 and
   2) and the Part 161 Qwen rule (1 and 0) are listed as not scored for direction. The README says so in the result cell
   and does not claim the direction clause was tested on four labels.
3. **A against Part 200 (exploratory) is 0 of 384 differing**, against Part 200's 76 of 200 on question 0. The two compare
   decoded text, not token ids, and Part 200's chunked calls are not Part 201's. I left it out of the README body; it is
   in `report.md` and `score.json` for anyone who wants it, and it is not a contradiction: Part 201's A and Part 200's
   chunks of 8 are both chunks of 8, so agreement is what the shape story predicts. It is worth a line in a later part
   that compares token ids.
4. Nothing else in the scored output looked inconsistent with the design or the cold check.

## BLOCKERS

- none. The alert I filed was refuted by parent control and folded (disposition `considered`, seq 1).
