# Brief: fold round 1 of the Part 19 review into circuit.py

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-19` (branch `part19`, HEAD 7285b27). Write only inside it. Do not commit, do not download models, do not change PART19-DESIGN.md or PREDICTIONS.md.

Read the review first: `/home/elisepc/green/DiscordBot/df-runs/cheap-review-part19-circuit-r1-haiku-r1-2026-10-18/worker-01/REVIEW.md`. Its measurement scripts are in `/home/elisepc/green/Journal/scratch/review-part19-circuit-r1-haiku-r1-2026-10-18/scripts/` (m1 to m10; `m7_mutants.py` runs the mutants against the test suite). Read them; copy what you need into this worktree rather than editing them there.

Fold F1 to F7 as the review describes:
- **F1:** tests that exercise the real `screen()`, `verify()` and `edges()` code paths: a few units' push and zeroing values from `screen()` against a finite difference and a true zeroing change (sign and magnitude), the verify ratio and the edge normalisation on constructed records, and the 99th-percentile null on a constructed case. Then run the review's mutants M2 to M5 against your suite: each must now fail it. Quote that output.
- **F2:** edges averages only over prompts that have identity positions (as verify does), for edges and for the null.
- **F3:** screen refuses (SystemExit with a clear message) when no identity prompt is valid, and when "Who made you?" is dropped (no push ranking).
- **F4:** answers.json records model, revision, prompts and tool sha256; a mismatch regenerates the answers (say so on stdout) rather than reusing them.
- **F5:** edges and describe check verify.json's recorded input shas against the current answers.json and screen.json and refuse on mismatch.
- **F6:** describe captures through the base decoder (no lm_head logits), keeps a bounded top-10 per node instead of every token, and avoids a persistent float32 copy of the unembedding where it can (the logit lens can work row-block by row-block); re-run the spill guard after its large allocations.
- **F7:** `--seed` sets verify's and group's random draws (default 0 keeps today's controls: verify 0, group 1, 2, 3 as offsets from it), or is removed from the header; pick one and say which.
- F8 is pre-existing (neuron_report._dump's shared .tmp): leave it, mention it.

Run with the Python that has torch: `QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a /home/elisepc/neuron-probe/.venv/bin/python test_circuit.py`, and keep `test_neuron_report.py` and `test_teacherforce.py` passing.

Report: `REPORT-19-fold-r1.md` in this worktree's root (exactly that name and place). SUMMARY (test counts as printed, the mutant results), FINDINGS (per F1 to F7: what changed, file:line), BLOCKERS (or "none"). Quote output you saw; never describe a result you did not see.
