# Cold check of Part 209's scored results (independent recomputation)

You are checking a finished research run, not building anything. Budget: about 60 tool calls. Do not chase any one
question for more than 10 calls: write what you found and what you could not settle into BLOCKERS and move on.

Write ONLY in your folder `/home/elisepc/green/Journal/scratch/part209-coldcheck/` (it is inside your write domain;
create files there with Write). Your report is `/home/elisepc/green/Journal/scratch/part209-coldcheck/REPORT.md`.
Create it FIRST with a one-line placeholder verdict, then update it after each numbered item below, so a stop at any
point leaves a useful report. Do not edit anything else. Do not run any GPU job. Do not commit.

## Inputs (read only). Repo worktree: `/home/elisepc/green/worktrees/ab-part203` (branch part209)
- Design and predictions: `PART209-DESIGN.md` (prediction rows 209.1 to 209.16 near line 230).
- Raw results in `results/part209/`: `r2-qwen-chat.jsonl` and `r2-instruct.jsonl` (816 answers each; fields `row_id`,
  `arm`, `k`, `answer`), `labels-ds.jsonl` and `labels-haiku.jsonl` (reader labels; fields `row_id`, `posture`,
  `unreadable`, `says_no_body`, `valence`; a row with an `error` key or no `posture` is skipped, and the LAST good row
  per `row_id` counts), `r1r3-<stage>.json` (R1 per-arm records per stage).
- The scorer's output under test: `results/part209/score.log` and `results/part209/scores.json` (made by
  `part209/score209.py`).

## The combining rule (so you do not need to read score209.py except to check a detail)
- R2 uses the PRIMARY reader `ds` only; `haiku` enters only through Cohen's kappa on posture (gate G2, bar 0.60).
- Category of a label: `unreadable` if `unreadable` is true, else its `posture` (sitting, standing, lying, other,
  none). Counts per arm sum over all `k`.
- Sitting share = sitting / (sitting + standing + lying + other); none fraction = none / all six categories.
- Told compliance: over arms TOLD-SIT, TOLD-STAND, TOLD-LIE, the fraction whose category equals the told posture.
- R1 `s` per arm: read how `s_by_arm` in score209.py (line ~254) builds it from `r1r3-<stage>.json`, then compute it
  with your own code.

## What to do (write your OWN script with `/home/elisepc/neuron-probe/.venv/bin/python -I`, from your folder)
1. R2 counts per arm and sitting share and none fraction, for both stages (qwen-chat, instruct), from labels-ds; told
   compliance for both. Compare every count and share with score.log's `R2` lines.
2. Kappa on posture between ds and haiku over the answers both labelled (score.log says 0.8416 on 1632).
3. R1 `s` per arm at pretrain-3.0T, sft and instruct. Compare with score.log's `R1` lines.
4. The statistic behind each of these predictions, with its verdict under the registered bar: 209.1 (ROOM-W share
   >= 0.70), 209.2 (FIELD-W share at least 0.20 below ROOM-W), 209.5 (NOBODY none fraction >= 0.60), 209.9 (R1 ROOM-W
   s, qwen-chat minus qwen-base, |diff| < 0.15; read r1r3-qwen-base.json and r1r3-qwen-chat.json), 209.11 (NOBODY-I
   sitting share within 0.20 of NONE-0's; say how many NOBODY-I answers name any posture at all), 209.14 (R1c s
   minus R2 ROOM-W share >= 0.10; find how score209 gets R1c), 209.16 (pretrain-3.0T R1: THIRD's s >= 0.5 and within
   0.15 of ROOM-W's; not read if either arm's mean P3 < 0.1). Compare each with score.log.
5. Hand-read 10 R2 answers per stage, chosen with Python `random.Random(209).sample(rows, 10)` over each r2 file in
   file order. For each: one line with row_id, the ds and haiku posture labels, and whether the ds label is right.

## REPORT.md (under two pages)
First line: a one-sentence verdict. Then SUMMARY (each match or mismatch with both numbers), FINDINGS (anything the
scorer seems to read differently from the design; any verdict that changes under your numbers), BLOCKERS.
