<!-- cheap-review.mjs cheap-review-part202-code-r1-ds-r1-2026-11-12: /home/elisepc/green/worktrees/ab-part202 5b1f6ab4484f34c72618f6e97d3aa4c6e9916841..HEAD, lane ds, df-run exit 0 -->
# REVIEW — part202-code-r1 (cold adversarial)

**Verdict: DO-NOT-SHIP.**

Two defects make the registered A770 run produce the wrong rows while the whole suite (33 passed) stays green:
every random-direction row at doses +1.0 and +1.2 is silently never written (3,200 of 6,516 rows, 64 calls skipped
whole), and the 200-row shape replicate then dies on the first row with `KeyError: 'neuron'`. A third lets a
pilot-stopped or crashed launch read as a passing record. Each finding below has the script I ran and its output;
scripts live in this directory and drive the real code at `/home/elisepc/green/worktrees/ab-part202`.

Scope: `git -C /home/elisepc/green/worktrees/ab-part202 diff 5b1f6ab4484f34c72618f6e97d3aa4c6e9916841..HEAD`
(one commit `0209d64a`, "Part 202: build (DS worker) plus review fixes before registration"). The design doc diff and
the whole test suite were read; `pytest -q part202` → **33 passed in 135.08s** with every defect below live.

---

## Findings

### F1 — CRITICAL (INTRODUCED): the resume/dedup key omits `dose`, so 3,200 planned rows are never generated

`part202/run202.py:422`, `:433-434`, `:460-463`, `:465` key rows by
`(stage, unit_id, question_index)`. Part 164's own key — which this module's docstring (`run202.py:30-32`) claims to
use, "(stage, unit_id, question_index)" — is `(stage, unit_id, arm, dose, question_index)` (`part164/run164.py:37-39`).
The random arm reuses the same 100 `unit_id`s at doses 0.8, 1.0 and 1.2, and `DOSES = (0.8, 1.0, 1.2)` runs 0.8 first,
so once the +0.8 rows are in `done`, `pending` for every +1.0 and +1.2 call (lines 433-434) is empty and the loop
`continue`s the call whole. Nothing is written and no receipt is recorded.

Measured on the real plan (`e1_done_key_collision.py`, a replay of `run()`'s own `done`/`pending` logic over
`run202.plan()`):

```
plan rows: 6516 calls: 146 companions: 132
rows run() would write on a fresh launch: 3316
planned rows never written: 3200
calls skipped whole: 64
first skipped calls (stage, call, rows): [('battery', 22, 64), ('battery', 23, 36), ...]
  ('battery', 'random', 1.0) 1600 got 0
  ('battery', 'random', 1.2) 1600 got 0
```

Measured end to end through the real `run()` on the tiny fixture (`e2_e3_replicate_and_stopped.py`): one question
should write 1 no-push + 100 recipe + 100 neuron@0.8 + 100 random@0.8 + 100 random@1.0 + 100 random@1.2 = **501** rows;
the run wrote **301**, `run() == 0`, and the record says `done`:

```
E4 run() return code: 0
E4 rows written: 301
   ('battery', 'neuron', 0.8) 100
   ('battery', 'random', 0.8) 100
   ('no_push', 'plain', 0.0) 1
   ('recipe', 'random', 0.8) 100
E4 intended ... = 501
```

Why it matters: the +1.0/+1.2 cells are exactly what verdicts 164.1 (1.2/0.8 ratio), 164.3 and 164.5 read. On the
A770 this wastes a ~3-hour run, writes a 3,316-row file, and 202.4's denominators silently become 1,600 where 4,800
were planned. Gate R3 cannot see it (see F3): `call_shapes` only inspects calls that were recorded, and skipped calls
leave no receipt.

A test would not catch it — one bakes it in. `part202/test_part202.py:195-219` asserts
`len(rows) == 301` while its own comment on line 209 enumerates 501 rows, and `assert sum(1 for n in seen if n == 8) == 38`
(line 213) is likewise the truncated count (38 chunks = 304 rows; a complete one-question battery is 64 chunks).
Fix the `done` key and that test goes red — the assertion has to be corrected to 501, not the code.

Smallest fix: key every `done`/on-disk comparison with `part164.run164.row_key(r)` (five fields, dose and arm
included) at `run202.py:422`, `:434`, `:460-463` and `:465`. Then re-run the two counts above and correct
`test_part202.py:209-214` to the complete shape.

### F2 — CRITICAL (INTRODUCED): the shape replicate always fails — `KeyError: 'neuron'` / `'random_index'` on the first sampled row

`run202.run_replicate_stage` (`part202/run202.py:612-613`, `:628`) builds its spec as
`dict(row, ...)` from a row written to `answers.jsonl` and hands it to the real `run164.pushes_for`. But
`part201/run201.py:326-333` (`row_fields`) does not copy `neuron`/`random_index` onto the row, and
`part163/run163.py:93-98` reads `row["neuron"]` / `row["random_index"]`.

Measured on real rows written by the tiny run, with the unmodified `run164.pushes_for` (`e2b_real_pushes_for.py`):

```
saved-row keys with neuron/random_index: ['layer', 'arm', 'dose']
replicate spec arm: neuron stage: replicate
KeyError: 'neuron'
--- random battery row
KeyError: 'random_index'
```

And inside the real `run()` (`e6_push_slicing.py`, which restores the real `pushes_for`):

```
run() rc: raised KeyError: 'neuron' (after the battery: the replicate stage, real pushes_for)
```

Why it matters: after the full battery the launch records `status: "stopped"`, `failure: "KeyError: 'neuron'"`, and
`main()` exits with a traceback; `results/part202/replicate.jsonl` does not exist, so 202.5 and every shape-floor
number the design registers are never produced. With F4, the documented scoring command then silently reports the
replicate as "not read" and does not withdraw the headline.

A test would not catch it: `test_part202.py:41-52` replaces `run164.pushes_for` with `_fake_pushes`, whose line 50 is
`int(row.get("neuron") or row.get("random_index") or 0)` — it tolerates exactly the keys the real function requires.
`test_the_replicate_rows_are_written_one_row_per_call` and the resume test pass on that stand-in.

Smallest fix: carry the unit index. Either add
`rows[-1]["neuron"] = spec.get("neuron"); rows[-1]["random_index"] = spec.get("random_index")` in
`_row_from_decode` (`run202.py:334-336`), or in `run_replicate_stage` build
`spec = dict(row, neuron=unit.get("neuron"), random_index=unit.get("random_index"), ...)` from `units164.json`. Then
delete the key-forgiving `or 0` fallback from the test stand-in and let it fail if a key is missing.

### F3 — HIGH (INTRODUCED): a stopped, killed or refused launch reads as a passing record; Gate R3 cannot see missing calls

`run202.record_gates` (`part202/run202.py:515-527`) only looks at `status == "refused"`; `status: "stopped"` and a
launch killed while `"running"` pass `run199.check_kernels`/`check_same_run` and `call_shapes`. `check_record`
(`:575-584`) therefore prints "PART202 record ok" and returns 0 for a pilot stop. `call_shapes` (`:530-572`) checks
only the calls that have receipts, so a run that stopped after three calls — or a run that skipped 64 calls (F1) —
is shaped exactly like a complete one.

Measured (`e2_e3_replicate_and_stopped.py`, pilot forced to stop with `PILOT_MIN_RATE = 1e9`, the same lever the
suite's own pilot test uses):

```
E3 stopped launch: run() rc = 3 status = stopped failure = pilot: 480.099 answers/s is below 1000000000.0000
PART202 record ok: 1 recorded load(s), all eager, ... 3 pushed calls in 21 chunks of exactly 8 rows ...
E3 check_record rc = 0
E3 record_gates ok = True why =  shapes passed = True
E3 rows the stopped run wrote: 165
```

`score202` compounds it (`part202/score202.py:280-284`, `:351`, `:416`): its `r1` is computed independently of
`record_gates` and never looks at `status`, although the printed label says "R1 every launch eager (and not refused)".
Measured (`e5_headline_and_stopped.py`) with a real refused launch (LAST_LOAD eager, config sdpa) merged with a
completed one, scoring Part 164's published rows:

```
(a) R1 every launch eager (and not refused): True   <- load:loads[*].load.attn_implementation
(a) R1/R3 record gates: False  <- load:record_gates
(a) R1/R3 reason: Gate R1: 1 launch(es) were refused at the load (nothing generated)
(a) 202.1 gates R1-R3 and Part 164's four gates pass: True
(a) headline reading withdrawn: False
```

and with a launch that stopped after writing the whole battery (the exact state F2 leaves):

```
(b) record_gates on a stopped-after-battery launch: True | no reason | shapes passed: True
(b) headline reading withdrawn: False
```

Why it matters: the design's rule is that a pilot stop is a HOLD and that the headline is withdrawn if R1 or R3
fails; the host's own gate command (`part202/steps.txt`, `run202.py --check-record`) reports such a run as clean.

Smallest fix: in `record_gates`, fail when any load entry's `status` is not `"done"` (report `stopped`/`running` with
its `failure`), and have `score202`'s `r1` use `record_gates`' `ok` rather than recomputing it. Optionally have
`check_record` compare the recorded calls against `plan()`'s 146 calls / 6,516 rows when the registered units file
was used.

### F4 — MEDIUM (INTRODUCED): a missing replicate file is silently "not read" and never withdraws the headline

`part202/score202.py:373` guards on `if replicate:`; `run202.read_jsonl` returns `[]` for a missing path, so the
documented command in `steps.txt` (`--replicate results/part202/replicate.jsonl`) on a run where F2 crashed prints
"shape replicate: not read: no --replicate given", "202.5: not read", and leaves `withdrawn` False (see the (b)
output above). The design registers the replicate and the shape floor as required (`Controls`, `Reading rules`).

Smallest fix: when `--replicate` is given but the file is missing/empty, exit 2 (bad input) or make
`withdrawn` true; keep "not read" only when the flag is absent.

### F5 — LOW (INTRODUCED): two claims in code/docstrings the code does not do

- `run202.py:20-21` (module docstring) says the 36-row calls are "padded with four duplicate companion rows (**the
  call's first four rows**)". `plan()`'s padding loop (`run202.py:171-173`) duplicates the **last chunk's** rows. On
  the tiny plan the padded chunk is `R96|q0 … R99|q0` plus those four as companions (`e6_push_slicing.py` output),
  i.e. rows 32-35 of the 36-row block, not rows 0-3. No data effect (companion outputs are discarded and the harness
  is row-independent), but the docstring is wrong and a reader checking "which unit is pushed into which row" would
  be misled.
- `_existing_whole` (`run202.py:263-289`) is documented as removing "a call without a whole call **or its receipt**",
  but its `record` argument is never read: the receipt check that `part201/run201.py:285-287` performs is absent.
  I could not reach a bad state from this (the receipt is flushed at `run202.py:457` before the rows are appended at
  `:464`), so this is a claim/coverage gap, not a demonstrated corruption.

Smallest fix: correct the docstring to "the last chunk's rows"; delete the unused `record` parameter or perform the
`receipt_of` check.

### F6 — LOW/MEDIUM (INTRODUCED by reuse): Gate R1 as implemented does not require one device

`PART202-DESIGN.md:33` requires a launch record to show "eager, one model revision, **one device** and one output
folder". `record_gates` (`run202.py:521-522`) calls `run199.check_same_run`, which compares
`(model, revision, out_dir)` only (`part199/run199.py:162`); `score202` prints the devices (`score202.py:287-288`) but
does not gate on them. Measured (`e7_device_gate.py`), a real completed record plus a second launch differing only in
`load.device`:

```
devices in the record: ['cpu', 'xpu:0']
record_gates ok: True | reason: none | shapes passed: True
```

Why it matters: a resumed launch on a different device (or a CPU fallback) would pass R1, and the rerun would not be
"the eager kernel on the A770" as registered. Smallest fix: add
`len({(e.get("load") or {}).get("device") for e in loads}) == 1` to `record_gates`.

### F7 — LOW (INTRODUCED): the recorded dtype is a constant, never compared to the load

`run202.py:402` writes `"dtype": "torch.bfloat16"` into every launch entry without reading it from the load
(`entry["load"]["dtype"]` is the real value). On the CPU tiny run the record says `torch.bfloat16` next to
`load.device: cpu`. The registered path does load bf16 (`neuron_report.py:180-181` defaults `dtype=torch.bfloat16`
and `EagerLoader` does not override it), so this is not wrong today — but R1's check gives no evidence that a
float32 load (which the design lists as a variable the A770 differs on) would be refused or recorded honestly.
Smallest fix: in `EagerLoader`, also require `(nr.LAST_LOAD or {}).get("dtype") == "torch.bfloat16"`, and write the
entry's dtype from `last_load` instead of the literal.

### F8 — LOW (INTRODUCED): `slot` is chunk-relative, and `part201/ROWS-SCHEMA.md` defines it as the index in the call

`part201/ROWS-SCHEMA.md:20` says `slot` is "this row's index in the call". `_row_from_decode` (`run202.py:326-336`)
enumerates the chunk's rows, so every row of every chunked call repeats slots 0-7. Measured on a 64-row battery call
written by the real run:

```
stage battery call 5 rows in call 64 call_rows len 64
slot values in file order: [0,1,2,3,4,5,6,7, 0,1,...,7, ...]   (eight times)
call-relative index would be: [0,1,...,63]
```

Why it matters: any later check that joins `slot` to `call_rows` to confirm which push went into which row — the
central question this part exists to answer — reads the wrong index for every chunked call. I verified push
alignment instead from the chunk row lists (see below). Smallest fix: pass the call-relative index
(`sum(len(ch["rows"]) for ch in earlier_chunks) + slot`, companions excluded) as `slot`.


---

## What I checked and found sound

- **The plan is Part 164's.** `test_the_plan_matches_part_164s_published_rows_key_for_key_in_order` passes against
  `results/part164/answers.jsonl`: 6,516 non-companion rows, same order, same key, same question, same unit. The plan
  itself (146 calls, 6,516 rows, 132 companions; 33 pushed 36-row calls) matches the design's updated counts; the
  planned cells are neuron@0.8 1,600, random@0.8/1.0/1.2 1,600 each, recipe 100, no-push 16 (E1 output).
- **Push slicing is correctly slot-aligned.** `e6_push_slicing.py` drove the real `run164.pushes_for` /
  `part163.pushes_for` (with `common86`'s own `neuron_delta` and `scaled_random_directions`) through all 38 pushed
  chunks of the tiny run and compared every chunk's matrix row to that row's one-row push:
  `wrong row count or slot misalignment: 0`, `companion rows checked: 4, whose push differs from their source: 0`,
  `prompt/row misalignments: 0`, `largest pushed chunk rows: 8`. `_chunk_generate` builds prompts and pushes from the
  same `chunk["rows"]` list in the same order, so a one-slot shift would have shown here.
- **The eager gate refuses both ways.** `EagerLoader` (`run202.py:244-260`) checks `nr.LAST_LOAD` *and*
  `model.config._attn_implementation`; I ran the case the suite marks "not run" (LAST_LOAD eager, config sdpa) and it
  refuses before generation and records `status: refused` (E5(a) output).
- **Gate R2.** The five SHA-256 values match the tree and a changed units file fails the gate; the ordered-prompt
  check reads the 16 battery questions from the rows and compares them to `run164.choose_questions()` (suite tests).
- **Resume granularity.** `_existing_whole` compares each `(stage, call)` group to the plan's `row_ids` set *and*
  count (`run202.py:271-276`), sets a non-whole call's rows aside to `rows.removed.jsonl` and rewrites the file, and
  the whole call is decoded again; the suite's half-written-call test passes and I read the path. The `on_disk`
  filter (`:460-463`) plus whole-call removal means no row key is appended twice — with the exception that the key
  itself lacks `dose` (F1), which is a loss of rows, not a duplicate.
- **Atomic writes.** `load.json` goes through a temporary file and `os.replace` (`part199/run199.py:236-242`), and
  `answers.jsonl`'s torn tail is cut by `run201._repair` before resume; a kill between a receipt flush and the row
  append leaves a call with no rows, which is decoded whole again.
- **Pilot arithmetic.** 0.30 answers/s is `6516 / (6*3600) = 0.3015`; the stop sets `status: "stopped"` with the
  reason, returns 3 and skips the replicate (E3). The flaw is in how the record and scorer read that status (F3),
  not in the stop itself.
- **Shape floor on Part 164's own rows.** The two real-row tests pass: donor substitution by `(broken, excluded,
  qwen)` reproduces every published verdict, estimate and gate, and a replicate that removes Qwen moves 164.1 below
  0.95 while an agreeing replicate holds it at 1.0. The floor scores all rows (`others` + battery inside
  `score202.shape_floor`, so gates 1-3 are recomputed rather than failing for want of rows).
- **No feature flags** are introduced by this change, so there is no off-path to check.
- **The folder lock** is `part199.FolderLock` under `run202.lock` (`run202.py:194-199`), exclusive-create, refuses a
  second launch with exit 3, and is removed on normal exit.
