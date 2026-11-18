<!-- cheap-review.mjs cheap-review-part202-code-r2-ds-r2-2026-11-12: /home/elisepc/green/worktrees/ab-part202 0209d64a..HEAD, lane ds, df-run exit 0 -->
<!-- cold adversarial review, part202-code-r2 (round 2: checks round-1 fixes F1-F8) -->
# REVIEW — part202-code-r2

**Verdict: SHIP-WITH-FIXES.** The battery path that round 1 broke is now correct and I could not break it: all 6,516
planned rows are written once with Part 164's five-field key, across a real pilot stop and relaunch, and the replicate
now rebuilds byte-identical pushes from the saved rows. What remains is one *replicate-only* resume key still on the
old three fields (a kill mid-replicate silently drops up to 4 of the 200 registered rows and the launch still records
`done`), the scorer's Gate R3 still blind to a skipped planned call, no row-count guard on the replicate file, and
three smaller claim/robustness gaps. All are small, local fixes; none needs a redesign.

Scope: `git -C /home/elisepc/green/worktrees/ab-part202 diff 0209d64a..HEAD` (one commit `e7053d98`, "Part 202: fold
DS code review r1 (F1-F8) before registration"), plus the round-1 review
`Journal/reviews/cheap-part202-code-r1-ds-r1.md`. Suite on this tree:
`/home/elisepc/neuron-probe/.venv/bin/python -I -m pytest -q part202` → **35 passed in 105.85s**. Scripts cited below
are in this directory (`e1_*`…`e9_*`, `sabotage_slot.py`) and drive the real code at
`/home/elisepc/green/worktrees/ab-part202`; they write only under this directory (`e1-*`, `e6-*`, `e7-*` are their run
folders). No model, GPU, network or live service was used.

---

## Findings

### 1 — MEDIUM (PRE-EXISTING, but F1 claims it fixed): the replicate stage still dedups on the old three-field key

`part202/run202.py:666` and `:668` (`run_replicate_stage`):

```python
done = {(r.get("stage"), r.get("unit_id"), int(r.get("question_index", -1))) for r in read_jsonl(path)}
pending = [r for r in sample if (r["stage"], r["unit_id"], int(r["question_index"])) not in done]
```

This is the exact key round 1's F1 removed from the battery loop. The new module docstring (`run202.py:30-31`) and the
DESIGN changelog ("The key is now Part 164's `run164.row_key` (stage, unit_id, arm, dose, question_index)") say the
old key is gone. It is not: the replicate sample draws the random arm at +0.8/+1.0/+1.2, and those rows reuse the
same `unit_id` and `question_index` at each dose.

Measured on the **registered** sample (`e`-style replay of the real `run202.replicate_plan` over
`results/part164/answers.jsonl`):

```
published rows 6516 battery 6400
sample 200
registered sample: rows sharing the old key 8 in 4 groups
  ('battery', 'L25D13', 11) x 2 [('random', 0.8), ('random', 1.0)]
  ('battery', 'L25D13', 3)  x 2 [('random', 0.8), ('random', 1.0)]
  ('battery', 'L22D329', 14) x 2 [('random', 1.0), ('random', 1.2)]
  ('battery', 'L24D226', 15) x 2 [('random', 1.0), ('random', 1.2)]
```

Measured end to end (`e1_replicate_resume.py`: a real `run202.run()` on the tiny fixture, then one row of a colliding
pair deleted to simulate a kill between two replicate writes, then the REAL `run_replicate_stage` called again):

```
run rc: 0
answers rows: 501 distinct run164.row_key: 501
replicate rows: 200
battery rows with neuron/random_index present: 400 of 400
replicate rows sharing the old (stage, unit_id, question_index) key: 105 in 45 groups
after simulated crash: replicate rows 199 (dropped one of a colliding pair)
  dropped row: unit R29 arm random dose 1.0 q 0
  twin still on disk: unit R29 arm random dose 0.8 q 0
run_replicate_stage wrote: 0 rows now: 199
pending with the old key still in the stage: 0
pending with run164.row_key: 1 [('R29', 'random', 1.0, 0)]
VERDICT: BUG (row never written on resume)
```

Why it matters: the replicate is the registered shape check behind every verdict's floor (DESIGN "Controls",
"Reading rules"). A kill during the 200 one-row calls leaves a partial `replicate.jsonl`; the relaunch redraws the
remaining rows *except* each whose dose-twin already occupies the key (up to 4 of 200 on the registered sample),
finishes, and writes `status: done`. Gate R1 passes (last launch done), Gate R3 has nothing to look at (no receipt is
missing), and `score202` never checks the replicate count (finding 3), so the run scores as if the replicate were
complete. Calibration: each pair's window is one row-write wide (≈1/200 of the replicate stage), so this is an edge
rather than an everyday failure — but when it fires it is silent and the run records `done`, and the fix is one line.
On the registered run the loss is bounded (≤4 rows) but it lands on the exact product the design registered.

No test catches it: the suite's replicate/resume tests only resume a *complete* replicate stage.

Smallest fix: `run164.row_key` at both `:666` and `:668`, plus a test that pre-writes one row of a dose-twin pair and
asserts the other is still written.

---

### 2 — LOW/MEDIUM (claim gap; the fix landed in check_record only): score202's Gate R3 cannot see a skipped planned call

`part202/score202.py:280` calls `run202.record_gates(load)` **without** the plan, so `r3 = shapes["passed"]` only
inspects the receipts that exist — exactly the round-1 F3 hole. The plan-aware check exists only in
`run202.check_record` (`run202.py:625-628`), yet the DESIGN changelog says "Gate R3 under the registered units requires
a receipt for every planned call", and the design's headline rule says the reading is withdrawn if R1 or R3 fails.

Measured (`e3_r3_skipped_call.py`; a completed tiny run's `load.json` with one battery call's receipt removed, status
left `done`):

```
removed receipt: battery 3 row_count 64
plan built from the run's own units and one question: calls 11 rows 501
(a) record_gates(rec)            -> True | R3 passed: True | no reason
(b) record_gates(rec, plan)      -> False | R3 passed: False | Gate R3: 1 of 11 planned calls have no receipt (first: [('battery', 3)])
(b) shapes planned_calls_without_receipt: 1
(c) check_record ... rc: 3
(d) score202.score gates_r1_r3: {'R1': True, 'R2': False, 'R3': True} withdrawn: True
```

(With the tiny rows R2 is false for an unrelated reason, so `withdrawn` is true there; on the registered rows a
skipped-call record would have R1 and R3 both true and the headline kept.) The host's `steps.txt` runs
`run202.py --check-record` first, so an operator does see the failure on that step — but the scorer, which owns the
withdrawal rule, does not reproduce it.

Why it matters: R3 is the gate that was supposed to make a *completed-but-short* launch impossible (the round-1 F1
bug was exactly that), and the scorer is the thing that decides whether the headline stands.

Smallest fix: in `score202.score`, build the plan from the units it already holds (add a `run202.plan_units(units)`
helper that `plan(path)` also uses) and pass it to `record_gates`, so `r3` is the same verdict `check_record` gives.
No test currently exercises score202 with a missing receipt.

---

### 3 — LOW/MEDIUM (INTRODUCED by the F4 fix being partial): a short replicate file is still accepted and scored

The F4 fix (`score202.py:444-446`) rejects a `--replicate` file that is **missing or empty**, but any non-empty file
passes; the registered count is 200 (`REPLICATE_ROWS`), and `202.5`'s printed label even says "of the 200".

Measured (`e9_short_replicate.py`, the tiny run's replicate with one row removed):

```
replicate rows in the file: 199 -> scoring with 198
  replicate rows: 198  <- replicate:len
  replicate rows matched to a battery row: 198  <- replicate:(unit_id, dose, question_index)
  202.5 between 40 and 100 of the 200 differ: False  <- replicate:answer vs rows answer
  stability independent 164.1: 1  <- rows+replicate:shape_floor.scenarios.independent.verdicts.164.1.sensitivity
  summary p2025: False gates_r1_r3: {'R1': True, 'R2': False, 'R3': True}
```

It ran and produced a floor. This is the missing guard that would have made finding 1 visible (that bug leaves a
`done` launch with 196–200 replicate rows). Smallest fix: when `--replicate` is given, require
`len(replicate) == run202.REPLICATE_ROWS` (or exit 2 naming the count).

---

### 4 — LOW (INTRODUCED by the F7 fix): the bf16 gate compares a value `load_record` documents as possibly a list

`part202/run202.py:261-264` refuses unless `self.last_load.get("dtype") == "torch.bfloat16"`. But
`neuron_report.load_record` (`neuron_report.py:173-176`) writes
`dtypes[0] if len(dtypes) == 1 else dtypes` over every floating parameter, i.e. a **list** whenever the loaded model is
not uniformly one dtype.

Measured (`e4_dtype_gate.py`, the real `load_record` feeding the real `EagerLoader`):

```
params ['torch.bfloat16']                  -> LAST_LOAD['dtype'] = 'torch.bfloat16'
params ['torch.bfloat16', 'torch.float32'] -> LAST_LOAD['dtype'] = ['torch.bfloat16', 'torch.float32']
params ['torch.float32']                   -> LAST_LOAD['dtype'] = 'torch.float32'

load with dtype 'torch.bfloat16': ACCEPTED
PART202 GATE R1 FAILED before generation: the model loaded in ['torch.bfloat16', 'torch.float32'], not torch.bfloat16; nothing generated
load with dtype ['torch.bfloat16', 'torch.float32']: REFUSED (SystemExit 3)
```

Is that wrong? For a model transformers keeps partly in fp32 (its `_keep_in_fp32_modules`, or any fp32 parameter), the
gate refuses the load **every time**, because no code path can make such a model uniform. Part 164's own loader would
have loaded it the same way, so "the registered bf16 load" is not well defined by this test. I cannot check the real
Qwen3.5-4B weights here, so I do not know whether it triggers today; nothing in the tree establishes uniformity, and
the new test (`test_part202.py:296-309`) only exercises the string form.

Smallest fix: record the *requested* dtype beside the measured one (`load_record` could add
`dtype_requested=str(dtype)` from `load_model`'s argument) and gate on that plus "every measured floating dtype is
torch.bfloat16" (accept `["torch.bfloat16"]` as well as the string). If a mixed load should instead be a HOLD, say so
in the design — right now the refusal is silent about the difference between "the operator asked for float32" and "this
model always loads mixed".

---

### 5 — LOW (PRE-EXISTING): a torn `replicate.jsonl` line makes every resumed launch fail

`answers.jsonl` is repaired on resume (`part201._repair` inside `_existing_whole`, `run202.py:291`); `replicate.jsonl`
is not. `run_replicate_stage` reads it with `run202.read_jsonl` (`run202.py:664-666`), which raises on a truncated last
line, and the stage writes with `fh.write(...); fh.flush()` per row, so a hard kill can leave exactly that.

Measured (`e6_torn_replicate.py`; one valid row plus a truncated second line, then the real stage):

```
stage raised: ValueError: invalid JSON at .../e6-.../replicate.jsonl:2
```

`run()` catches it, records `status: stopped` with that failure and re-raises (`run202.py:487-495`); the next launch
hits the same line before writing anything, so the folder is stuck until a human edits the file. `steps.txt` promises
"a task killed at the runner's 6-hour limit is relaunched with the SAME line". Smallest fix: call
`run201._repair(path)` on `replicate.jsonl` before reading it (exactly as `_existing_whole` does for answers), and a
test that pre-writes a truncated last line.

---

### 6 — LOW (PRE-EXISTING): `row_id` is not unique within a pass, contrary to `part201/ROWS-SCHEMA.md:41`

`part201/ROWS-SCHEMA.md:41`: "A row is identified by (`pass`, `row_id`). A relaunch skips rows already present and
never writes a duplicate." `run202._spec` (`run202.py:144`) builds `row_id = f"{unit_id}|q{question_index}"` with no
arm or dose, and the random arm reuses its unit ids at three doses.

Measured (`e8_plan_padding.py`, real `plan()` on the registered units):

```
planned scored rows 6516 distinct run164.row_key 6516
planned rows sharing (pass, row_id) -- part201/ROWS-SCHEMA.md:41 identity: 3200 collisions, 1600 distinct ids
   e.g. [(('battery', 'L19D47|q0'), 3), (('battery', 'L19D219|q0'), 3)]
```

Nothing in Part 202 keys on `(pass, row_id)` any more (that is why the `done` key moved to `run164.row_key`), so there
is no data loss today; but the row file is read against the reused schema, and any reader that follows it (Part 201's
own resume contract is exactly that) would collapse each random unit's three doses into one row. `--replicate-plan`'s
output (`run202.py:701-707`) has the same property by design (it writes stage/unit/dose so it is unambiguous).
Smallest fix: put arm and dose in `row_id` for pushed rows (e.g. `R0|q0|1.2`), or amend `ROWS-SCHEMA.md` with Part
202's five-field identity. The scorer's own keys are in the same family: `score202.row_key` (`score202.py:64-65`) and
the replicate↔battery match (`:379-380`) omit `arm`, which is safe only because the registered units file happens to
reuse no `unit_id` across arms (I checked: the 100 neuron and 100 random ids are disjoint); `run164`'s own validation
does not enforce that, so a future units file could silently merge them. This is a follow-up, not a reason to hold
this change.

---

### 7 — LOW (claim mismatch; round-1 F5 half-fixed): `plan()`'s docstring still says the wrong rows are duplicated

`run202.py:151-152` (the `plan` docstring): "the last chunk of a call with fewer than 8 rows is padded with **the
call's first rows** as companions". The module docstring was corrected to "the last chunk's own four rows", but this
one was not. Measured (`e8_plan_padding.py`):

```
last chunk row_ids:       ['L26D254|q15', 'L26D365|q15', 'L26D374|q15', 'L26D416|q15', 'L26D254|q15', ...]
last chunk companions:    ['L26D254|q15', 'L26D365|q15', 'L26D374|q15', 'L26D416|q15']
call rows 0..3 (pad claim in plan()'s docstring): ['L24D159|q15', 'L24D175|q15', 'L24D197|q15', 'L24D205|q15']
```

No data effect (companion outputs are discarded), but it is the same misleading claim round 1 raised, now wrong in one
place instead of two. Fix the sentence.

---

### 8 — LOW (test gap): reverting the F8 `slot` fix leaves the suite green

`grep` finds no test in `part202/` that reads `slot`, `chunk_slot`, `push_sha256` or the recorded launch `dtype`.
Measured by loading a pytest plugin that restores the old behaviour (chunk index written into `slot`,
`sabotage_slot.py`):

```
cd /home/elisepc/green/worktrees/ab-part202
PYTHONPATH=.../review-part202-code-r2-ds-r2-2026-11-12 python -m pytest -q -p sabotage_slot part202/test_part202.py \
  -k "writes_every_planned_row or resume_decodes or replicate"
4 passed, 11 deselected

# and with -s, to show the plugin really applied:
[sabotage] run202._row_from_decode now writes the chunk index into slot
2 passed, 13 deselected
```

So F8's fix (and F7's second half — the entry's `dtype` now comes from the load, `run202.py:418-419`, but no test
reads it) would silently regress. The fixes are correct on this tree (I verified both directly, below); the gap is in
the suite. Smallest fix: after the run test, assert `[r["slot"] for r in battery_call] == list(range(64))` and that
`chunk_slot` repeats 0–7 per chunk; assert `rec["loads"][-1]["dtype"] == "torch.bfloat16"` in the float32 test's
eager counterpart.

---

## What I checked and found sound

- **F1 on the battery (the round-1 disaster) is fixed, including across a pilot stop.** `e7_pilot_relaunch.py` runs
  the real `run()` twice into one folder: launch 1 forced to stop at the pilot (the suite's own lever,
  `PILOT_MIN_RATE = 1e9`), launch 2 normal. Output:

  ```
  launch 1 (forced pilot stop): rc 3 status stopped failure pilot: 731.138 answers/s is below 1000000000.0000
    rows on disk: 165 replicate file exists: False
  launch 2 (relaunch): rc 0 status done launches recorded 2
    rows: 501 distinct run164.row_key: 501 duplicated keys: 0
    replicate rows: 200
  PART202 record ok: 2 recorded load(s), all eager, one model/revision/folder; 10 pushed calls in 64 chunks of exactly 8 rows, 12 companion rows discarded
    check_record: 0
    cells: {('battery','neuron',0.8): 100, ('battery','random',0.8): 100, ('battery','random',1.0): 100, ('battery','random',1.2): 100}
  ```

  Every planned key is written once; no duplicates; a legitimate pilot stop followed by a completed relaunch passes
  R1/R3. `run164.row_key` is used at `run202.py:441`, `:453`, `:479`, `:480`, `:482`; the only old-key comparisons
  left are finding 1's (replicate file, not the battery).
- **F2 is fixed and the replicate push is identical to the battery row's.** `e2_replicate_push.py` drives the REAL
  `run164.pushes_for`/`part163.pushes_for`/`common86` on rows written by the real `run()`, comparing the row's matrix
  slice inside its plan chunk with the one-row push built from the saved row via `run202._replicate_spec`:

  ```
  battery  neuron  0.8  L0N0|q0  075e3a4f34363798  075e3a4f3436379820  hash=True tensor=True
  battery  random  0.8  R0|q0    0f65fb6e27310151  0f65fb6e27310151e3  hash=True tensor=True
  battery  random  1.0  R0|q0    b5fd3cf53b33a42f  b5fd3cf53b33a42fee  hash=True tensor=True
  battery  random  1.2  R0|q0    0fcfc60e93d4e400  0fcfc60e93d4e4008c  hash=True tensor=True
  ```

  Byte-identical for both arms and all three doses (the dose scaling and the 512-direction basis survive). The saved
  rows carry `layer`, `arm`, `neuron`, `random_index`, `dose`; the test stand-in is now strict (it raises on a missing
  index, as `run163.pushes_for` does).
- **F3/F6 gates behave as intended on the launch sequences a resume can produce.** `e5_resume_gates.py` builds real
  records by splitting a completed run's receipts across two launches:

  ```
  killed while running, then done        gate_r1=True   record_gates_with_plan=True missing=0
  pilot stopped, then done               gate_r1=True   record_gates_with_plan=True missing=0
  refused, then done                     gate_r1=False  Gate R1: 1 launch(es) were refused ...
  killed while running (never resumed)   gate_r1=False  Gate R1: the last launch is 'running', not done
  two devices across a resume            gate_r1=False  Gate R1: launches ran on more than one device: ['cpu', 'xpu:0']
  one launch, done                       gate_r1=True   record_gates_with_plan=True missing=0
  ```

  A killed or pilot-stopped launch followed by a completed one still passes, and receipts split across two launches
  are all counted. A refused launch blocks the record forever — that is the round-1 F3 design and matches "no launch
  refused" (a follow-up decision for the host, not a defect). One small hole: `gate_r1` builds the device set with
  `str(...)`, so a load that records **no** device passes the one-device test (`e9`: `gate_r1(...) -> (True, '')` on a
  `done` launch whose `load` has no `device`); it only matters for hand-built records, since `run()` always writes it.
- **F8 is correct.** `e1`'s rows: slots are `0..63` in call order for a 64-row call (`calls whose slot != 0..n-1 in
  call order: 0` over all 11 calls) and `chunk_slot` restarts `0..7` per 8-row model call. Push hashes, prompt lengths
  and recorder rows all use the chunk slot; only the recorded field is call-relative. (`pad_tokens` is chunk-relative,
  which is right: the chunk is the model call.)
- **Companions.** 33 padded calls of 36 rows, 132 companions; each companion's push row equals its source row's push
  (`companion rows whose push equals their source row's push: 4 of 4`), and companions are never written to the row
  file (501 = 1 + 100 + 100 + 300 in `e7`, no 513).
- **Plan integrity.** `plan()` on the registered units: 146 calls (16 no-push, 2 recipe, 128 battery), 6,516 scored
  rows, 6,516 distinct `run164.row_key` values, 132 companions; the suite's
  `test_the_plan_matches_part_164s_published_rows_key_for_key_in_order` holds against Part 164's published rows.
- **The new gate tests bite.** I checked the sabotage each test's docstring names: the 501/64-chunk assertions catch
  a `done` key without arm/dose; the strict stand-in catches a row without `neuron`/`random_index`; `short["calls"]`
  removal catches a record_gates that ignores the plan; the pilot test's `"not done" in reason` catches a `gate_r1`
  that ignores `status`; the float32 test catches a missing dtype check.
- **No feature flags** exist in this part (nothing to check with a flag off).
- **Concurrency/restarts.** `Run202Lock` (Part 199's `FolderLock` under `run202.lock`) still refuses a second launch;
  rows are redrawn whole-call, `answers.jsonl`'s torn tail is repaired before resume, and the receipt-before-rows order
  means a kill between them leaves a call with no rows (decoded again), which `_existing_whole`'s rewritten docstring
  now states accurately (the unused `record` parameter round 1 complained about is gone).
- **`_existing_whole` no longer needs the receipt check.** Every call it keeps is exactly the plan's `row_id` set and
  count for that `(stage, call)`; call numbers are unique across stages, and no two planned rows share a
  `run164.row_key`, so the redraw-everything path cannot double-write.
