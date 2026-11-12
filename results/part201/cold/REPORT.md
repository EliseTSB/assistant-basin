VERDICT: Every number I recomputed from rows.jsonl and probe.json matches report.md and score.json exactly; no prediction's verdict changes.

# Part 201 cold check — independent recomputation

Script: `/home/elisepc/green/Journal/scratch/part201-coldcheck/coldcheck.py` (stdlib only, run with `python3 -I`). It reads
`rows.jsonl` and `probe.json` only. It does not import or read `part201/score201.py`. Step = bf16 ulp at the top logit's
magnitude (2^(e-8), i.e. 0.125 for a top logit in [16,32)), as in the design's Analysis paragraph.

## SUMMARY (mine vs scorer)

| Check | Mine | Scorer | Match |
|---|---|---|---|
| A/A-again compared / differing | 16 / 0 | 16 / 0 | yes |
| A/A2 compared / differing | 96 / 0 | 96 / 0 | yes |
| A/B compared / differing | 384 / 0 | 384 / 0 | yes |
| A/M compared / differing | 384 / 147 | 384 / 147 | yes |
| A/C compared / differing | 192 / 86 | 192 / 86 | yes |
| F8/F1 compared / differing | 48 / 0 | 48 / 0 | yes |
| A/F8 compared / differing | 96 / 58 | 96 / 58 | yes |
| A/C: ≤2 steps at first diff | 84 of 86 = 0.9767 | 84 of 86 = 0.9767 | yes |
| A/C: largest pass-A steps | 3.0 (raw 0.1875) | 3.0 | yes |
| A/F8: ≤2 steps | 58 of 58 = 1.0 | 58 of 58 = 1.0 | yes |
| A/F8: largest pass-A steps | 2.0 (raw 0.125) | 2.0 | yes |
| A/F8: largest F8 margin | 0.23192787170410156 | 0.23192787170410156 | yes |
| G2 rows with emitted_is_max false | 0 of 1648 | 0 of 1648 | yes |
| 201.11 pass A seconds | 516.0977051999653 (48 calls) | same | yes |
| 201.11 @1 step: flagged / slowdown | 1324 / 2.467101 | 1324 / 2.467101 | yes |
| 201.11 @2 steps: flagged / slowdown | 1892 / 3.158557 | 1892 / 3.158557 | yes |
| 201.11 @3 / @4 (reported too) | 2307 / 3.686580; 2594 / 4.085302 | same | yes |

The A/C and A/F8 first-difference tables (row, position, both margins) match the report row for row. That includes the two
A/C rows at 3 steps (L25N7804|q15, L26D365|q9) and the one row at 0.5 step (L25N1898|q10, which is in A/M only, not in A/C or
A/F8). Every first-difference position is inside both passes' recorded margins.

## FINDINGS

1. **Subsets are the design's.** A-again = A's calls 0 and 1 (16 rows). A2 = questions 0–3. F8 = the first 16 units on
   questions 0,1,2,8,9,10 (A's calls 0,1,3,4,6,7,24,25,27,28,30,31). F1 = the first 8 units on the same six questions, 48
   one-row calls. A/C intersects on exactly the 192 pushed C rows. Each subset is the first N units of A's interleaved
   (neuron, random, neuron, random) order for its questions. No pair was scored over a wider or narrower set than the design
   defines.
2. **"Within 2 steps" = margin ≤ 2 × step.** I read it as inclusive. No row sits exactly at the boundary (margins are
   0, 0.5, 1, 2 or 3 steps), so an exclusive reading would give the same numbers.
3. **201.11's "unfinished row" reading.** I counted a row as unfinished at step t when t < len(tokens), the same as the
   scorer (`score201.py:666`). Counting a row that emits its end token at step t as still unfinished would give 1314 steps
   and 2.455 at 1 step, and 1875 steps and 3.138 at 2 steps. That is a different reading of the design's wording, not a
   defect. Both readings fail 201.11's bar of 2, so the verdict does not change.
4. **201.11 pricing** follows Amendment 1 #3. Each flagged step costs the probe's median at the smallest probed row count ≥
   the live rows and the smallest prefix ≥ prompt+generated; above 96 the 96 figure is scaled by length/96. My
   extrapolation counts (23, 38, 45, 50) match the scorer's. The probe has no 2-row or 3-row entry, so 2–3 live rows are
   priced at the 4-row median. That is the amendment's rule.
5. **No verdict changes.** 201.1 (0/0), 201.2 (86/192 = 44.8%, inside 25–50%), 201.3 (147/384 = 38.3% ≥ 20%), 201.4 (0/384
   ≤ 10%), 201.5 (97.7% ≤ 2 steps, max 3 ≤ 8), 201.8 (0/48), 201.9 (100% ≤ 2 steps, max F8 margin 0.232 < 0.5), 201.10
   (58/96 = 60.4%, inside 30–70%) and 201.11 (3.159 > 2, failed) all stand under my numbers. 201.6 and 201.7 I did not
   recompute (see BLOCKERS).
6. **The brief's path is stale.** The brief names `part201/PART201-DESIGN.md`. The design is at the repo root,
   `PART201-DESIGN.md`. The amendments are in `part201/`. Filed as a papercut.

## BLOCKERS

- **201.6 label flips (A/M), item 3 of the brief.** The rows carry no label fields (`name_gone`, `maker_gone`, `broken`,
  `qwen_part161_rule` are absent from every row). The design says the labels come from Part 163's code, `part86/triage86.py`
  and `results/part163/extra163.py`, called on each decoded answer. Reproducing them means importing project code, which
  the brief forbids. I did not recompute them. The scorer's flip counts (broken 8/8, maker_gone 4/2, name_gone 1/4,
  qwen 1/0) are unchecked here.
- **201.7 (pass K vs CPU reference).** Not in the brief's list of five items; not recomputed.
- **201.11 "nearest larger row count"** (the prediction's wording) against Amendment 1's "smallest probed ≥" are the same
  rule but a different phrasing. I followed the amendment, as the brief directs for the design. No probed step needed a
  row count above 8, so the two readings agree anyway.

## SELF-ASSESSMENT

I am confident in the pair counts, the margin/step tables, G2 and 201.11. They reproduce to the last digit, and the
row-by-row tables agree. Less certain: the 201.11 "unfinished" edge and the label flips, which I could not check. The
scorer's own decision text ("candidate gate: 3 steps") also reads the largest A/F8 pass-A margin as 2 steps and adds one.
That is consistent with my 2.0.
