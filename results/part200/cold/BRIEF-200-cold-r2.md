# Cold check, round 2: Part 200 after the round-1 fold

You are a fresh reader. Repository root (read-only): `/home/elisepc/green/worktrees/ab-part200`. Writable:
`results/part200/cold/`. Write `results/part200/cold/COLD-r2.md`. Same file rules as round 1
(`results/part200/cold/BRIEF-200-cold.md`): copy no folder and no file over 1 MB, compute in your own folder with
`/home/elisepc/neuron-probe/.venv/bin/python -I -B`, never run report200.py or fig200.py in place, no model, no GPU.

Round 1's findings are in `results/part200/cold/COLD.md`; the fold is commit 81c2310 (`git show --stat 81c2310`).

Check, with evidence:
1. Each of round 1's findings 1 to 8 and 10: folded correctly, partly, or not. Recompute the new numbers (lengths table,
   at-cap/under-cap broken split, gate 1 shares, identical-text and mid-phrase counts, first-difference method claims,
   the run rate and timing).
2. The new text Claude wrote, which round 1 did not see: the README title; the update note at the top of
   `results/part163/README.md`; the front page `README.md` (search "Part 200": one sentence in the warning banner near
   the top and one dated entry after Part 199's). Every number there must match a file; nothing may be stronger than
   the Part 200 README; any exploratory reading must be labelled.
3. `PREDICTIONS.md`'s `## Part 200` rows: do the Result and Verdict columns match the files?
4. Anything new the fold introduced that is wrong or overstated.

Write COLD-r2.md first as a skeleton. Verdict line (SHIP, SHIP-WITH-FIXES or HOLD), numbered findings with evidence and
a fix each, then SUMMARY / FINDINGS / BLOCKERS lines.
