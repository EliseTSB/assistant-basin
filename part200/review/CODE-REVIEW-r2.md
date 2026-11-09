# Part 200 code review, round 2 (cold, after the folds in 882b7fe)

**Verdict: LAUNCH-WITH-FIXES.** Of the nine round-1 findings, seven are fixed, one is partly fixed (finding 4) and one is addressed by an appended correction (finding 9). The startup repair held in every kill
and truncation case I ran (1,135 trials, zero problems). Two new defects need a small edit before launch: Gate R3's
batch check never observes anything in production (finding 1), and the design's host `--compare` command exits 2 on
the published folder as it stands (finding 2). Neither changes the design, and neither can produce a wrong answer
silently.

Evidence base: 50 existing tests pass (`/home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -p no:cacheprovider
part200/test_part200.py`). Probes and sweeps are in `worker-01/scratch/` (outside this repository, nothing written here).
The fake model and generator are the round-1 harness; the real `part148/battery148.generate` is used in finding 1.
Nothing in the repository was edited.

## 1. Round-1 findings re-checked

| # | Round-1 finding | Status now | Where, and the evidence |
|---|---|---|---|
| 1 | No default loader; `main()` crashes at the first load | **Fixed** | `run200.py:572-573` (`loader = EagerLoader(args)`). Probe: `run200.main([...])` with no injected loader reaches `neuron_report.load_model(<MODEL_DEFAULT snapshot>, 851bf6e8…, 'xpu')`, the revision equals `run155._common().REVISION`, the default attention is eager (`neuron_report.py:59`). No TypeError. Test at `test_part200.py:179`. |
| 2 | Gate R2 units hash never checked | **Fixed** | `run200.py:94` (`EXPECTED_UNITS_SHA256`), `gate_r2` at `run200.py:713-737` (`hash_matches`, line 729). Probe: a units file with one neuron changed gives `passed: False, units_hash_matches_design: False`; the real `part163/units163.json` gives `True`. |
| 3 | `--compare` exits 0 on a partial batch-1 file; shape flag switched off | **Fixed** | `batch1_complete` and `IncompleteBatch1` (`run200.py:740-753`, `948-950`), exit 3 at `run200.py:999-1002`. Shape flag now requires 200 unique rows (`run200.py:554-555`). Probe: 150 rows with 21 changed gives exit 3 and no `compare.json`. |
| 4 | Gate R3 records the plan, not what the generator got | **Partly fixed** | Answers count and push-row count are measured per chunk and checked (`run200.py:176-186`). A mismatch raises before it reaches the record. The intended `model_batch` observation never happens in production: see finding 1. |
| 5 | A kill inside the row-write loop makes every relaunch fail R3 | **Fixed** | `repair_startup` and `_cut_calls_back` (`run200.py:430-486`). Verified in section 2: 1,104 truncation trials and 31 kill trials, all relaunches pass R1 and R3 with 3,416 unique rows. |
| 6 | A torn last line blocks every relaunch | **Fixed** | `_cut_torn_tail` (`run200.py:398-409`), called on both files before any read (`run200.py:473-474`). Verified: torn-byte cuts in both files in the sweep; no JSONDecodeError. |
| 7 | A refused non-eager launch leaves no record | **Fixed** | `recording_loader` writes a `status: refused` entry before it exits (`run200.py:600-611`); `record_gates` reads it as a refusal (`run200.py:371-374`). Probe with an sdpa model: exit 3, zero generate calls, one `refused` entry, `record_gates` False. |
| 8 | `--compare` does not print the outcomes of 200.1 to 200.7 | **Fixed in the JSON** | `prediction_outcomes` (`run200.py:861-918`) puts each outcome, its value(s) and its held/failed result in `compare.json` under `scored.predictions`. The one-line `PART200 prediction …` prints (`run200.py:1024-1025`) appear only when all gates pass; on a failed gate the outcomes are still in the JSON. I did not re-score the predictions against real data beyond the existing test. |
| 9 | BUILD-REPORT stale (line 13, 26, 62, 126) | **Addressed by an appended correction** | `part200/BUILD-REPORT.md:285-287` and `steps.txt:285-287`. The old text remains above it, as the correction says. The appended note is correct. |

## 2. Startup repair: kill points (item 2)

Probe: `worker-01/scratch/killsweep.py`, `run_sweep.py`. The oracle is independent of `run200`: the plan is 16 one-row
no-push calls, then the recipe and each of the 16 battery questions in calls of 64, 64, 64, 8 over the units in
`units163.json` order. For each trial I checked: rows (3,416), duplicate keys, every battery and recipe row's push
against the unit's own single-row push, 200 unique batch-1 rows with correct pushes, the record's Gates R1 and R3, and
that `repair.answers.rows_kept` equals the oracle's last whole call boundary at or below the line count on disk.

- **A. Real kills (31 trials).** The fake generator raises on generate call n, for n in 1, 2, 8, 9, 15, 16, 17, 18, 40,
  80, 81, 100, 144, 145, 200, 215, 216, 217, 250, 300, 415, 416, 417, 441, 442, 450, 500, 600, 615, 616, 641. Those cover
  the no-push stage, the recipe (including its first and last calls), the first and last battery chunks of each
  question boundary, the final 8-row block (kill 441), between stages (442), and inside batch 1 (600 to 641). **0
  problems.** All relaunches exit 0.
- **B. Torn writes (1,104 trials).** A complete launch's `answers.jsonl` cut at 979 distinct byte offsets, covering 289
  different line counts from 0 to 3,016. The offsets are each call boundary, one byte before it (the previous line loses
  its newline, so it is torn), and 40 bytes into the next row. The `batch1.jsonl` cuts were at 0, 1, 500, 26,663,
  53,286 and 53,325 bytes. **0 problems**; `kept == oracle` in every trial.

What this shows: the repair never cut a row of a complete call, never left a partial call on disk after the relaunch
(`kept` is always a whole call boundary), and never produced a duplicate. Gate R3 passed on every relaunched record,
including the killed launch's incomplete calls.

Limits: the fake model and generator stand in for the real kernel and `AddRowsHook`. Kills are simulated by raising in
the generator (A) and by truncating a complete file (B). Those are the states a real kill can leave, since every write is
an ordered append followed by a flush. The sweeps do not measure how many generate calls a relaunch makes, only that the
final files are correct.

## 3. Folds introduced: what is wrong, and the production path (item 3)

### Finding 1 (medium, fix before launch): Gate R3's batch check never sees a batch in production

`run200.py:218`: `measured.append(_batch_seen(args[0] if args else kwargs.get("inputs")))`.
`part148/battery148.py:58` calls `model.generate(**inputs, max_new_tokens=…)`, so the tensors arrive as keyword
arguments. `kwargs.get("inputs")` is None, so `_batch_seen` returns None and the record stores `model_batch: None` for every
chunk. `call_shapes` skips None (`run200.py:320-322`), so the check is vacuous and `chunks_with_model_batch` is 0.

Evidence: `worker-01/scratch/probe_modelbatch.py` calls the **real** `battery148.generate` with a fake model
(`generate(**inputs)`), a fake `common` and a fake tokenizer. The model received eight batches of 8 rows, as it should, but
the log records `model_batch: [None, None, …]`, and `call_shapes` reports `chunks_with_model_batch: 0`.

The test that claims to cover this (`test_part200.py:426-441`, `459-476`) uses a positional call,
`model.generate({"input_ids": …}, max_new_tokens=…)`, which production never makes. So the test passes while the
production path records nothing.

Risk: low for the answers themselves, because each chunk's tensors are sliced to exactly 8 rows and the answer count is
checked. But the gate then claims an observation it does not make, and the design says R3 reads "the row counts actually
passed to Part 163's generator."

Fix (one line, plus the test): `measured.append(_batch_seen(args[0] if args else kwargs))`, since `kwargs` holds
`input_ids` in this call style. Change `BatchModel._generate` to take `**inputs` as well as the positional form, and
assert that `model_batch == [8]*8` for a 64-row call. I checked the one-line change on a probe-only copy of the watcher
(`worker-01/scratch/probe_modelbatch_fix.py`): `model_batch` becomes `[8]*8` and `chunks_with_model_batch` is 8. The
repository was not edited.

### Finding 2 (medium, fix before the host step): the design's `--compare` command fails on the published folder

`steps.txt` and `PART200-DESIGN.md` (Method) run `run200.py --compare results/part200 results/part163`. The published
folder has no `units163.json` (`results/part163/` lists 10 files, `git ls-files` confirms). `compare()` requires
`units163.json` in the published folder (`PUBLISHED_REQUIRED`, `run200.py:924`; the check at `run200.py:935-938`), so the
command exits 2 with `missing file …/results/part163/units163.json` before any scoring. Probe: `probe_compare.py`, step [1]
and the real-path check in step [3].

The test does not catch this. `test_part200.py` around line 921-934 writes a `units163.json` into its own temporary
"published" folder and so never uses the real folder layout.

Fix: take the published units from `published/"units163.json"` if it exists, otherwise from
`ROOT/"part163/units163.json"`. Gate R2 still hashes that file against `EXPECTED_UNITS_SHA256`, so nothing weakens. Drop
the file from `PUBLISHED_REQUIRED`, and add a test that runs `--compare` on a folder laid out like `results/part163`, with
the file absent. This also keeps the operator's post-run step from failing for a reason unrelated to the run.

### Finding 3 (low): `resume_at` in `load.json` is wrong for battery stages

`run200.py:460`: `per_question = (BATTERY_ROWS // 16) // size * size` is 192. A battery question has 200 rows in calls of
64, 64, 64, 8. Evidence (`worker-01/scratch/probe_resume_at.py`): head 416 reports question 1, row 8 (the plan says
question 1, row 0); head 1,344 reports question 5, row 168 (the plan says row 128). The existing test asserts the wrong
value: `test_part200.py:825` expects `_first_incomplete_call(216 + 192, 64)` to be question 1, row 0. The plan says question 0,
row 192 (the 8-row remainder of question 0).

This only feeds the diagnostic `repair.answers.resume_at`; the cut itself does not use it (the sweeps confirm `kept` equals
the oracle). Fix: `per_question = BATTERY_ROWS // 16` (200), and correct the test to question 0, row 192.

### Finding 4 (low, note): the `observed` field records the planned size after the checks

`run200.py:183-186`: `observed`, `push_rows` and `answers` are appended after the checks pass. `push_rows` is the planned
`size`, not a separate measurement. This is correct, because a mismatch raises before the append. But the field name
overstates what is measured. The measured quantities are the answer count (`len(got)`) and, once finding 1 is fixed, the
model's batch. Rename `observed` to `planned_checked`, or measure `push_rows` from the returned tensors before the check.

## Other checks (no finding)

- **Production path.** `run200.main` with no loader reaches `nr.load_model` with the model path, the revision and `xpu`
  (probe). `run163.run` is handed `recording_loader` and calls it once with `args.device`, so the model loads once.
  `_run_batch1_after` gets the cached pair from the same `EagerLoader`.
- **Imports.** The new code adds only standard-library imports at module level (`hashlib`, `json`, `re`). The run path
  imports nothing that `steps.txt --copy` omits.
- **Not a defect, noted.** A load that raises an exception other than `SystemExit` (for example a device fault inside
  `load_model`) leaves no `load.json` entry (probe: the output folder stays empty). The design asks for an entry "as soon
  as the model loads", so this matches the design. It means a crashed load leaves no trace in the record.
- **Test suite.** 50 pass. The gap is the one the tests did not hit: production call style (finding 1) and the real
  published folder layout (finding 2).

## Summary of required edits (small, local, no design change)

1. `run200.py:218`: pass `kwargs` when there is no positional argument; update `BatchModel` and the test to the
   `**inputs` call style (finding 1).
2. `run200.py:935-938` and `:924`: fall back to `ROOT/part163/units163.json` for the published side; add a test on the
   real folder layout (finding 2).
3. `run200.py:460` and `test_part200.py:825`: 200 rows per battery question (finding 3). Optional before launch.
4. Finding 4: rename or measure. Optional.

SUMMARY: LAUNCH-WITH-FIXES. Round-1 findings 1, 2, 3, 5, 6, 7 and 8 are fixed, 4 is partly fixed, and 9 is addressed by an appended correction. The startup
repair held in 1,135 kill and truncation trials: 3,416 unique rows, no wrong pushes, 200 batch-1 rows, and Gates R1 and R3
pass on every relaunch. Two defects need small edits before launch: Gate R3's batch check is vacuous in production (the
model's batch is recorded as None because battery148 passes `**inputs`), and the design's `--compare results/part163`
exits 2 because the published folder has no `units163.json`.
FINDINGS: 1 medium (R3 batch observation vacuous in production; tests use the wrong call style), 1 medium (host
`--compare` fails on the published folder; test masks it), 1 low (`resume_at` uses 192 rows per battery question; the test
encodes it), 1 low note (`observed` is the planned size after the check). Round 1: 7 fixed, 1 partly (finding 4), 1
addressed (finding 9).
BLOCKERS: none that stops the run itself. Fix finding 1 and finding 2 before launch: the first makes a required gate
vacuous, and the second stops the post-run `--compare` as written. Both are local edits.
