<!-- cheap-review.mjs cheap-review-part210-build-ds-r1-2026-11-13: /home/elisepc/green/worktrees/ab-part210 efb9d2bd..HEAD, lane ds, df-run exit 0 -->
# Cold review: part210-build (Part 165 rerun on eager attention)

**Verdict: SHIP-WITH-FIXES.** The plan rebuild, the chunked decode, the resume-by-row-key, the replicate draw,
the scorer's imports and the printing all do what the design says, and the suite (62 tests) passes. Two things
must be fixed before a launch: Gate R3 does not enforce the design's "8 chunks of exactly 8" clause and stays
green on a pushed call that covers 8 of its 64 rows (finding 1), and the runner refuses to start on a job
folder built from `part210/steps.txt`'s own `--copy` list (finding 2). Both are small fixes. Findings 3–5 are
test-quality/dormant-path issues.

Review base: `git -C /home/elisepc/green/worktrees/ab-part210 diff efb9d2bd..HEAD` (run210.py, score210.py,
steps.txt, test_run210.py, test_score210.py; PRIOR-ART.md ignored per brief). margins210.py and its tests are
from the design commit 5c6b613a and were read because score210 imports it and the design's Size section is
entirely built from it. All experiments run against the real worktree code; no model, no GPU, no live service.
Suite: `62 passed in 292.34s` (pytest from /home/elisepc/neuron-probe/.venv, whole repo; the three part210
test files together are the 62).

---

## Finding 1 — Gate R3 cannot fail on a truncated pushed call (INTRODUCED, high)

`part210/run210.py:666` `call_shapes`, `bad_chunk` at `:673-690`, `record_gates` at `:646-663`.

Design, Controls, Gate R3: "Every pushed call has 8 chunks of exactly 8 rows, the chunks cover the call's rows
in the block's order, ... every planned call has a receipt." The code checks (a) each chunk is exactly 8 rows,
(b) companions appear only in the last chunk, (c) `scored == call["row_ids"]` where both sides come from **the
receipt itself**. `record_gates` (`:656`) matches a receipt to the plan by `(pass, call)` only and never
compares the receipt's `row_ids` (or `row_count`) with the planned call's. So a pushed call receipted as one
chunk of 8 rows — a call that lost 7/8 of its chunks — passes Gate R3.

Measured (`probe_r3b.py`, my scratch dir): a real 24-call plan built by `run210.plan` for a 12-slot, one-layer
fixture, every call receipted, except the first pushed receipt truncated to its first chunk:

```
planned calls: 24
receipt chunks: 1 receipt row_ids: 8
record_gates -> True no reason
{'passed': True, ..., 'pushed_calls': 16, 'pushed_chunks': 105, ..., 'planned_calls_without_receipt': 0}
```

and `probe_r3.py` case A/B: a 32-row pushed call in 4 chunks of 8 → `passed: True, pushed_chunks: 4`; the same
call in 1 chunk of 8 → `passed: True, pushed_chunks: 1`. The design's own test
(`test_every_pushed_call_is_8_chunks...`) asserts the plan has 8 chunks, but no test drives a short-chunk-count
receipt through `call_shapes`/`record_gates`; the suite's Gate R3 tests only shrink a chunk (`test_run210.py:248`,
`test_score210.py:387`) or drop a receipt (`:391`).

Why it matters: Gate R3 is the only check on what reached the model. A future change that decodes a block as
one 8-row chunk per call (or drops chunks on a partial failure) leaves 56 rows per call unwritten — Gate 1 would
catch that at score time — but a change that decodes only 8 rows and *writes 64* (or a receipt that claims the
call's rows while only one chunk ran) would pass Gate R3 and publish a shape claim the run does not support.
The design's Controls sentence "R3 reads the shapes passed to the model, not the intended chunks" also fails:
`receipt["chunks"][i]["rows"] = len(p["chunk"]["rows"])` (`run210.py:553-557`) is the planned chunk's length, not
an observed count.

Smallest fix (two checks, no new data): in `bad_chunk`, require `len(chunks) == int(call.get("row_count", 0)) // CHUNK`
and `sum(ch["rows"] for ch in chunks) == int(call.get("row_count", 0))`; in `record_gates`, build
`expected = {(c["stage"], c["call"]): list(c["row_ids"]) for c in planned["calls"]}` and fail when a receipt's
`row_ids` is not exactly that list. Both fields are already written by `run()` (`run210.py:546`).

## Finding 2 — the runner refuses to start on a job folder built from its own --copy list (INTRODUCED, medium)

`part210/run210.py:445` calls `gate_r2()`, which hashes the 14 files at `:70-85`, **before the model load**, and
returns 3 if any is missing. `part210/steps.txt:14` copies 22 files and does not include four of the 14:
`part164/units164.json`, `part165/score165.py`, `results/part165/answers.jsonl`, `results/part165/norms.json`.
`run()` does not need them to generate; `run165.questions()` (which `plan()` calls) needs only
`part153/battery153.py` and `part164/run164.py`, both copied.

Measured (`probe_r2_copy.py`): build a folder containing exactly the `--copy` entries from steps.txt, then

```
--copy entries: 22
gate_r2 passed: False
  MISSING/DIFFERENT part164/units164.json: got None
  MISSING/DIFFERENT part165/score165.py: got None
  MISSING/DIFFERENT results/part165/norms.json: got None
  MISSING/DIFFERENT results/part165/answers.jsonl: got None
```

which is `run()`'s `PART210 GATE R2 FAILED before model load` → exit 3, nothing generated. Every module the
runner imports does resolve in that folder (checked: part165/164/163/160/155/153/148/86, part201, part199,
tools.margin_recorder, neuron_report all import). The same folder's record check also dies:

```
python -I -B part210/run210.py --check-record load.json
ValueError: results/part165/answers.jsonl has no full-arm k for layers [19, ..., 26]
```

(`registered_widths`, `run210.py:252-262`, called from `check_record` `:713`). So the host step in steps.txt can
only run where `results/part165/answers.jsonl` exists.

Why it matters: this works on today's A770 folder only because earlier parts' launches left those four files
there — an undeclared dependency on stale state, and a hard stop after a fresh job-folder setup. The design puts
Gate R2 "before scoring" (Reading rules), not before generation; the pre-load refusal is stricter than the
design and creates this new failure mode.

Smallest fix: add the four paths to the `--copy` list in steps.txt (they are small; score165.py and units164.json
are needed by nothing at run time but are already part of the hashed closure), and/or keep `run()`'s pre-load
check to the files the runner imports and let score210's full 14-file R2 withdraw the headline as designed.

## Finding 3 — the resume test's on-disk-filter claim is not tested; deleting the filter leaves the suite green (INTRODUCED, medium — test quality)

`part210/run210.py:560-561` (the `on_disk` filter), `part210/test_run210.py:366-431`, docstring at `:367-368`
and comment at `:402-408`.

The docstring says the test breaks if "in run() write a redrawn call's rows without the on-disk filter (the 8
kept rows twice)". Measured: copy the module to a scratch tree, delete the two filter lines, run
`pytest -k resume` → `4 passed`. Cause: the test writes 8 rows of a 64-row call and expects `_existing_whole` to
keep the call ("it is whole against the plan", `:405`), but `_existing_whole` (`:355-360`) compares the present
row ids with the planned call's 64 and moves the 8 aside — measured by instrumenting the same scenario
(`test_probe_moved.py` in the sabotaged copy):

```
PART210 resume: 8 rows of 1 calls that are not whole on disk will be decoded again
last battery call: 23 planned rows: 44
rows moved to rows.removed.jsonl on the second launch: 8 of call [23]
8-row chunks decoded on the second launch: 112   planned battery chunks: 112
```

So no row is ever on disk when the call is re-decoded, the filter never fires, and its 8 kept rows are appended
once because `_existing_whole` already removed them — not because of the filter. The same holds for the
`if not missing: continue` guard's justification at `run210.py:533-535` ("the tiny fixture's one-layer plan
repeats run165.row_key across calls"): the tiny plan has 872 **distinct** row keys (measured), so that rationale
is false too (`test_run210.py:323, 569` repeat it).

Why it matters: the brief's own claim "a resumed launch never decodes a call already on disk and never appends a
row twice" rests on three guards; one of them has no test and is unreachable given `_existing_whole`, and the
second guard's stated reason is wrong. A later change to `_existing_whole` (e.g. keeping a call that is whole
against the *receipt* instead of the plan) would remove the real protection and the suite would stay green.

Smallest fix: delete the dead filter and its claim, or test it directly by calling the loop with a pre-seeded
`done` that omits one key of an otherwise-complete call (the only state the filter can see); correct the two
comments to say the tiny plan's keys are distinct and that `_existing_whole` is what prevents re-appending.

## Finding 4 — Gate R3 rejects the padding shape `plan()` itself produces (INTRODUCED, low, dormant for the registered plan)

`part210/run210.py:200-213` (companions are appended by repeating the chunk's own rows from index 0),
`:673-690` (`bad_chunk` re-derives the scored rows by consuming the companions list in order).

A short block's last chunk is `[A,B,C]` padded to `[A,B,C,A,B,C,A,B]` with `companions=[A,B,C,A,B]`. `bad_chunk`
consumes the first five occurrences as companions and leaves `scored=[C,A,B] != call["row_ids"]=[A,B,C]`, so
Gate R3 fails a call the planner built exactly as the design says it must be built ("its last chunk is padded
with duplicates of its own rows, marked as companions and discarded"). Measured (`probe_r3.py` case C/D):

```
3 scored rows: chunk row_ids [A,B,C,A,B,C,A,B], companions [A,B,C,A,B]
call_shapes -> {'passed': False, 'pushed_calls_not_in_chunks_of_8': 1, 'companion_rows': 5, ...}
```

(With 1 or 2 scored rows it passes by coincidence; 3–7 fail.) The registered plan never needs padding — 864 = 13×64 + 32, and 32 = 4 chunks of 8 — so this cannot fire for the registered run; `test_a_short_block_gets_marked_companions...` checks `plan()` only and never feeds the padded call to `call_shapes`. `test_run210.py:547`
shows the author knew ("three slots ... every call padded, which Gate R3 refuses") and worked around it.

Why it matters: the design registers a padding rule and a gate that rejects it. If the plan ever changes shape
(fewer slots/arms, a partial block) the launch cannot pass its own gate. Smallest fix: record the chunk's scored
row ids in the receipt separately from the model row ids (`plan()` knows them before padding), and have
`bad_chunk` check `chunk["scored"] == call["row_ids"][offset:offset+len(scored)]` and
`rows == len(scored) + len(companions)`.

## Finding 5 — prediction 210.4 passes vacuously on a partial rows file (INTRODUCED, low)

`part210/score210.py:466`: `p2104 = all(at_cap[arm] >= 300 for arm in ARMS if per_arm[arm] >= 768)`. When no arm
reaches 768 rows the generator is empty and `all()` is True, so the report prints "210.4 ... True" on data that
cannot support it (Gate 1 fails and the headline is withdrawn, but the prediction line is still a reading).

Measured (`probe_p2104.py`, 8 no-push rows + one battery row per arm):

```
gate1_rows rows: 17  <- rows:gates.gate1_rows.rows
210.4 at least 300 of 768 at the cap in each arm: True  <- rows:battery/* token_count==80
summary p2104: True part165 gates: {'gate1_rows': False, ...}
```

Smallest fix: `p2104 = all(per_arm[arm] >= 768 and at_cap[arm] >= 300 for arm in ARMS)` (or None/"not read"
when any arm is short), as 210.6 already does by refusing to read without a 200-row replicate.

---

## What I checked and found sound

- **Plan rebuild.** `run210.plan` reproduces Part 165's order: 8 no-push one-row calls first, then per question
  `random.Random(165+qi).shuffle(all_rows(...))` in blocks of 64, split into chunks of 8 in block order; the test
  compares all 6,920 rows key-for-key against `results/part165/answers.jsonl` and the suite passes. 6,920 distinct
  row keys; `_spec` carries stage/arm/slot/layer/k/question/kind and `row_id`.
- **Chunked decode.** `_chunk_generate` builds one prompt per chunk row (companions included) and `pushes_for` on
  the chunk's own rows, so each layer's push matrix is `[chunk_rows, width]` with zeros for other layers' rows —
  the same shape `run165.run` gives a 64-row batch. `generate_rows` raises if push rows != prompt rows.
- **R1.** `EagerLoader` refuses a non-eager kernel, a non-eager config and any non-bf16 (including mixed) load
  before generation; `gate_r1` requires at least one launch, no refusals, every load eager, one device, one
  (model, revision, out_dir) and a last launch `done`. Every branch has a red test.
- **Resume.** `_existing_whole` moves a call that is not exactly its planned row set to `rows.removed.jsonl` and
  rewrites `answers.jsonl` atomically (`tmp.replace`); a completed run resumed decodes nothing (test and measured);
  a torn tail is trimmed by `run201._repair`; replicate rows resume on their own `rep|...` row_id, and the sample
  is drawn from labels that never change once written.
- **Failure paths.** An exception inside the call loop records status `stopped` with the exception and re-raises;
  the pilot stop records `stopped` and exits 3 before the replicate; the record is written atomically
  (`run199._save_record`); the folder lock is exclusive and names its holder. Two launches cannot interleave.
- **Margins.** `row_fields` writes the recorder's tokens/top1/top2/margin/emitted_logit/finish/nonfinite plus Part
  165's fields; `slot` is overwritten with Part 165's slot and `chunk_slot` keeps the model-call index, so
  `score165.score_rows` reads the rerun rows unchanged (tests assert the field set).
- **Replicate draw.** Largest-remainder over the nine equal arms (22/23), half from broken and half from kept per
  arm with `random.Random(210)`, fallback to kept (and to broken) when a stratum is short; 200 rows, one per call;
  deterministic across seeds; tests cover allocation, fallback, resume and the fields.
- **Scorer.** It imports `score165.score_rows` rather than re-implementing the rules; every printed line carries
  `<- file:key`; the published verdict/estimate and `margins210`'s distance sit beside each; 165.6 prints its
  usable-draw count and marks a collapsed interval not informative; gates R1/R2/R3 and Part 165's gates all feed
  `withdrawn`, and omitting `--load`/`--norms`/`--replicate` cannot pass (they read False/None, not "pass").
  `_score_without_bootstrap` restores the patched score165 functions in a `finally`; the point estimates equal
  `score_rows`' on the published rows (test).
- **Shape floor.** Scores the 8 no-push rows plus every battery row each redraw (Gate 1 stays green; the design's
  Part 202 defect is avoided), memoises distinct redraws, uses the donor rule with `question_index`, counts
  no-donor substitutions and projections, and leaves `changed`/165.6 out. The floor's labels are Part 165's own
  rules, and a donor with the same labels and phrasing reproduces every published verdict (test).
- **Windows/job host.** No `fcntl`, `pwd`, `resource` or `signal` in run210/run201/run199; every text `open()` has
  `encoding="utf-8"` (or `write_text(..., encoding=)`); `newline="\n"` on the append paths; `os.replace` for
  atomic writes; the lock uses `os.open(O_CREAT|O_EXCL)`. The only launch-path problem is finding 2.
- **Flags off.** There are no feature flags in this part; every optional scorer input fails closed (see above).

## Things I looked at and did not raise

- `donor_key` adds the row's `changed` value to the design's "labels + question_index" key (`score210.py:136-147`).
  It changes which donor text is chosen for a drawn vector but not any floored verdict (165.1–165.5 and the gates
  read labels only, and a donor with the same labels scores the same), so it is inert except for the printed
  no-donor count.
- `possible_key` (`score210.py:119-125`) sets `broken = broken or triage or loop or short`, so the impossible
  vector {broken:true, others false} is not projected back to false; such a redraw keeps the row's own text, which
  has the same scored outcome as the design's projection (both keep the row). Counters differ only.
- `165.3`'s "rerun distance from the published value" prints `null` because `margins210` stores 165.3's estimate
  as a formatted string; the distance from the bar itself is printed from `margins.md` on the line above. Cosmetic.
- `run()`'s early returns after the load (`run210.py:513-517`) leave the launch entry as `running`; Gate R1 rejects
  it ("not done"), so it cannot be read as a pass.
