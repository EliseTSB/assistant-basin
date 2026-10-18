# Brief: fold round 2 of the Part 19 review into circuit.py (a provenance redesign, not a patch)

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-19` (branch `part19`, HEAD f1502c6). Write only
inside it. Do not commit, do not download models, do not touch PREDICTIONS.md or PART19-DESIGN.md.

Read first: `PART19-DESIGN.md`, the round-2 review
`/home/elisepc/green/DiscordBot/df-runs/cheap-review-part19-circuit-r2-haiku-r2-2026-10-18/worker-01/REVIEW.md`
(F1-F8), and its measurement scripts in the same directory (`m1_provenance.py`, `m2_describe.py`, `m3_edges_null.py`,
`m5_test_detail.py`, `m6_push_fd.py` and their `*_output.txt`).

## Why a redesign

Round 1 and round 2 both found stale-provenance defects in the input chain (round 2: F1, a stale screen accepted
downstream; F2, input hashes taken at write time). The stop rule written before round 2 says a repeat of that class
changes the design. So replace the current answers cache and SHA checks with this design, and remove the code it
supersedes (`checked_verify` as written, the regenerate-on-mismatch branch in `answers()`, hashing inputs in `header`).

## 1. Provenance design (replaces F1, F2)

1. `read_input(path)`: read the file's bytes once, sha256 those bytes, parse JSON from the same bytes, return
   `(obj, sha)`. Every step reads every input through it, and records exactly those shas in its output header as
   `input_shas`. `header()` no longer hashes files on disk.
2. **Only `screen` creates answers.json**, and only when it is absent. If answers.json exists and its `__meta__`
   differs from the current model, revision, prompts or tool sha, every step exits non-zero with a message naming
   the mismatch and saying to use a fresh `--out`. No step regenerates silently. Any step other than `screen` exits
   if answers.json is absent.
3. **Each link checks the link before it**, using the shas it read:
   - `verify` and `group`: screen.json's recorded `input_shas['answers.json']` must equal the answers sha they read,
     and screen.json's model and revision must equal the current ones.
   - `edges` and `describe`: verify.json's recorded `answers.json` and `screen.json` shas must equal the shas they
     read, and the screen check above must hold too.
   - On any mismatch: exit non-zero, write nothing, and name which file is stale.
4. Rows that fail on the code at f1502c6 and pass after (write them first and watch them fail): the review's m1
   scenario A (a stale screen after an answers change is refused by verify), scenario C (screen replaced between
   verify reading it and edges reading verify is refused by edges), a tool-sha change refuses every step, and a
   fresh chain screen → verify → group → edges → describe still passes.

## 2. Tests that can fail (F4, F5, F7)

- F4: run the push finite-difference check on `model.float()`, assert the dtype is float32, tolerance about 5e-3.
- F5: zeroing magnitude within 25% relative, with a 1e-3 absolute floor.
- F7: promote the review's `m2_describe.py` brute-force checks (unique documents, per-document maximum, order, 10th
  value; logit-lens top 10 and z against a full float64 computation) into `test_circuit.py` on the tiny model.
  Allow ties at bf16 to differ only in order.

## 3. Smaller fixes

- F3: the header's group seeds are `[a.seed+s for s in (1,2,3)]`.
- F6: skip self-edges (source == target). For each target j, compute its 99th-percentile threshold only from null
  sources at layers BELOW j's layer (others cannot affect it); if fewer than 10 such sources, record j as
  `not_testable` with the count, and do not count it as significant. Write `null_sources_used_by_target` into
  edges.json. Keep the null sources as 19b's controls (the design says so).
- F8: give `_dump` in neuron_report.py a unique tmp name per write (pid plus a random suffix), then `os.replace`.

## Running the tests

`/home/elisepc/neuron-probe/.venv/bin/python -B test_circuit.py` (plain python3 has no torch), with
`QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Also run `test_neuron_report.py` and `test_teacherforce.py`, which import shared helpers. All must pass.

## Report

Write `REPORT-19-fold-r2.md` in the worktree root: SUMMARY (suite counts and exit codes, quoted), FINDINGS (per item:
what changed, file:line, the row that proves it, and that row's output on f1502c6 and after), BLOCKERS (or "none").
Quote output you saw; never describe a result you did not see.
