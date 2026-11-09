# Amendment 1 to both build briefs (2026-11-09, after Alden's design review)

`PART201-DESIGN.md` and `BATCH-SHAPE-NOISE.md` changed after you started. Re-read in the design: Method (Recorder,
the new "Timing probe" paragraph, Analysis), Controls (G4, the B and M bullet), the Predictions table (201.6, 201.7,
201.11) and Reading rules; in the plan: section 1 ("Sensitivity of each verdict") and section 3 (now an ordered
list). Where this file and your brief disagree, this file wins. Add a "Amendment 1" section to your report saying
what you changed for each item.

## Builder A (recorder and runner)

1. **Frozen plan.** `plan.json` is written once; its sha256 goes in every `load.json` entry; a relaunch whose plan
   hash differs exits 3 before loading the model.
2. **The call is the unit of work.** Append all of a call's rows together after the call finishes. On startup cut a
   torn tail back to the last WHOLE call (drop rows of a call that is not complete). A relaunch re-runs every
   unfinished call whole, with its original membership and slots. Rows are never repacked into new calls.
3. **Launch index.** Add `launch` (int, the index of this launch's entry in `load.json`) to every row and to
   `part201/ROWS-SCHEMA.md` (you may edit the schema for this field only; post it on `part201-rows-schema`).
4. **Pass order in `bf16-main`: A, A-again, B, M, C.** A-again runs straight after pass A in the same launch. If pass
   A's calls 0 and 1 were written by an earlier launch, write no A-again rows and record
   `"a_again": "not same process"` in `load.json`.
5. **Timing probe**, at the end of the float32 process, into `results/part201/probe.json`: full-prefix float32
   forward passes with `use_cache=False` (one forward, no generation), on rows of pass F8's first call, at 1, 4 and 8
   rows and prefix lengths 32, 64 and 96 tokens (build each prefix from the row's prompt plus its generated tokens,
   repeating the generated tokens if needed to reach the length; content does not matter, the push hook stays
   active), five repetitions each, the device synchronised before each clock read, outputs discarded. Record every
   repetition's seconds, and F8's cached seconds per generation step per call. In `--smoke` use lengths 8, 16, 24.
6. **Gate G4 also counts hook firings**: per call, the number of times the push hook ran and the number of forward
   passes; they must be equal.
7. **Pass K is retained**: `steps.txt` writes pass K's record to a name that is not overwritten by a relaunch (if
   `results/part201/kernel-f32.json` exists, the step's comment says to rename it by hand first; do not delete it).

## Builder B (scorer and shape-floor tool)

1. **201.6**: the direction clause is scored only over labels with at least 8 flips; a label with fewer is listed as
   "not scored (n flips)"; if no label qualifies the direction clause is not scored and the prediction's verdict is
   the length clause's, reported as "length clause only".
2. **201.7**: token equality on all 16 rows over every position, batch 1 and 2. No margin cut-off.
3. **201.11**, from `results/part201/probe.json` (`--probe`): for every generation step of every pass A call, the
   set of unfinished rows within 2 steps; if it is not empty, add the probe's median seconds for the smallest probed
   row count that is at least that many rows and the smallest probed prefix length that is at least the step's
   prefix length (prompt tokens + tokens generated so far; above 96, the 96 figure times length/96). Slowdown =
   (pass A's summed call seconds + that sum) / pass A's summed call seconds. Not scored without the probe file.
   Also print the same figure at 1 and 4 steps.
4. At each first difference, record whether the other pass's emitted token is this pass's `top2`; list the rows
   where it is not. Print the number of differing rows beside every "of the rows that differ" percentage.
5. Print the plan's ordered decision (`BATCH-SHAPE-NOISE.md` section 3): the first rule that applies, by number,
   with the readings that triggered it. A prediction that is not scored counts as not held.
6. **`tools/shape_floor.py` redraw model.** A redraw gives each row a TRANSITION, not independent per-label flips:
   draw uniformly from the replicate pairs whose starting labels (`labels_a`, all labels jointly) equal the row's
   labels, and give the row that pair's `labels_b`. When no replicate pair starts with the row's label vector, fall
   back to independent per-label flips at the conditional rates and report how many rows used the fallback.
   Scenarios, all reported: `independent` (a fresh draw per row); `cluster-0.5` and `cluster-1` (with
   `--cluster-key`: with that probability a cluster's rows share one uniform number that picks each row's
   transition by rank, so they move together; otherwise independent); `upper` (independent per-label flips at each
   direction's upper 95% bound). Print each direction's count and denominator. Call the output "sensitivity
   (conditional on the scenario and on the replicate's shape pair)", never "stability" or a 95% guarantee. Update
   `part201/floor200.py` and `results/part201/floor200.md` to the new output.
