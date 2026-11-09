# Brief: fold Part 200's cold check, round 2

Repository `/home/elisepc/green/worktrees/ab-part200`. Writable: `part200/` and `results/part200/` (not
`results/part200/cold/`). No push, no commit, no model, no GPU. Read `results/part200/cold/COLD-r2.md`.

Claude has already folded F1, F6 and the front-page and Part 163 halves of F5 and F7 (those files are not yours). Fold,
in `part200/report200.py`, `part200/fig200.py`, `results/part200/README.md`, `results/part200/report200.md` (by
rerunning report200.py) and `results/part200/hand_read.md`:

- F2: the load record's time is when the model had loaded and the record was written, not when loading began.
- F3: the `chunk_size` 64 in load.json is the startup-repair block's batch size (run200.repair_startup, size=BATCH_SIZE),
  not a run199 default; say so, and that the generate calls ran in chunks of 8 per the call records.
- F4: drop "a length change as much as anything"; state the composition shift with the numbers (broken at cap and
  under cap, both runs) and keep the joint kernel-and-shape attribution.
- F5: drop "these are near-tie flips, not the SDPA fault". No margins were recorded. Say only where the first
  differences fall, that this is exploratory, and quote the design's rule that a one-row decode's change is not
  evidence about the 64-key fault.
- F7: say 163.4 held by one unit in each arm; do not call 163.5 live without its margin (give its intervals instead).
  The title may keep "verdicts hold" only if the first paragraph immediately says three of five could not fail.
- F8: wrap or shorten the overlapping x tick labels; add mean length if it fits cleanly, otherwise leave it in the
  table; update the image alt text.
- F9: the ~833 s gap between summary.seconds and the record's span is most likely the untimed batch-1 stage (run200.py
  runs it after run163.run returns); say that, as an inference from the code.
- F10, F11, F12: the reviewer's wording fixes.

Leave the README's "Cold check:" line saying two Haiku 5.5 rounds (10 and 12 findings), all folded. Rerun report200.py
and fig200.py with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`, then `tools/part_check.py publish 200`.
Write `part200/REPORT-fold-cold-r2.md` FIRST (one line per finding) and end with SUMMARY / FINDINGS / BLOCKERS.
