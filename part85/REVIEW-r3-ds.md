<!-- cheap-review.mjs cheap-review-part85-fold2-ds-r3-2026-10-27: /home/elisepc/green/worktrees/ab-part67 e5a9a48..1df997b, lane ds, df-run exit 0 -->
# Cold review: Part 85 fold 2 (e5a9a48..1df997b)

Reviewer: independent, read-only. Code read from /home/elisepc/green/worktrees/ab-part67.
Scripts and outputs in ./scripts/. Nothing was run against a live service, no credential
file was read, no live state was touched. All git remotes in the scripts are temporary
bare repos under /tmp.

## Verdict

SHIP-WITH-FIXES

The redesign does what the stop rule asked: the run() smoke test now asserts on written
files and an independent observer, and handoff85.py exists with restore / preflight /
refs-manifest / push / assemble. 28 tests pass on the host (I re-ran them: `Ran 28 tests
... OK`). The amend/attempt-2 flow works end to end against the real scripts (T8).

The blocking-class defect is in push(): a failed push leaves a local commit and the next
push reports success without pushing it, so a cloud session can end believing its results
are on the branch when the branch does not exist (T11). That is the same "lose results if
the session ends" class the fold was meant to close, so it is INTRODUCED by this change
and should be fixed before paid sessions run.

## Findings

### F1. HIGH, INTRODUCED — push() reports success when it pushed nothing; a failed push is never retried

`part85/handoff85.py:220-222`

```python
if not staged:
    print("no new checkpoint result changes to push")
    return
_git("commit", "-m", f"Part 85 results: {checkpoint}")
```

push() decides "nothing to do" from the *index*, not from HEAD vs origin. If an earlier
push committed and then failed at the network (or the branch query), the commit stays
local. The next push — the one CLOUD-RUN.md promises "after each step file and after every
answer cell" — finds nothing staged and returns 0 without contacting the remote.

MEASURED (scripts/t11_push_silent_loss.py, real handoff85.py, temporary bare remote):

```
push while origin down: rc= 1
commits ahead of origin/main after failed push: 1
next push (step-file trigger, nothing new): rc= 0 | no new checkpoint result changes to push
remote branch exists? (absent)
LOCAL HEAD: a158e94 Part 85 results: sft
```

and the same shape when the remote is reachable but behind (scripts/t10_push_failure.py):

```
(d) push with nothing staged but HEAD ahead: rc= 0 no new checkpoint result changes to push
    remote tip == HEAD? False
```

Why it matters: the whole point of the per-cell push is that a session that dies loses at
most one cell. Here a session can die having lost every cell since the first failed push,
and each later push printed a success line. The exit code is 0, so `subprocess.run(...,
check=True)` in part85.py:458-462, step0_weights.py:538-541 and step1_acts.py:260-264
does not raise either: the run continues to the next cell, writing more unpushed commits.

Smallest fix: before the `if not staged` early return, compare
`_git("rev-parse", "HEAD")` with the remote tip (`_branch_exists` / `_fetch` already gives
it) and fall through to the push loop when they differ; in the push loop, treat "nothing
to push but HEAD != remote" as a push attempt, not a no-op.

**No test can see this.** The smoke test mocks `subprocess.run` wholesale and only counts
calls whose argv mentions handoff85.py (test_part85.py:667-676), and the three handoff
tests each push at most once per branch before asserting. I sabotaged push() so that every
push AFTER the first one on a branch is a silent no-op that still prints the success line
(scripts/t15_f1_untested.py):

```
sabotaged: 2nd+ push per branch is a silent no-op
rc: 0
Ran 28 tests in 2.839s
OK
```

The suite stays green. (Sabotages that break the FIRST push are caught — I also ran a
no-op-push sabotage and a first-push-fails sabotage, both fail 2-3 tests — so the gap is
specifically the second and later pushes, which is exactly where F1 lives.) A regression
test should push cell A, make the next push fail, then push cell B and assert the remote
tree contains both cells.

### F2. MEDIUM, INTRODUCED — a push failure aborts run() after the cell is marked complete, so the cell is never pushed by a resume

`part85/part85.py:448-462`

The answer file and run.json (with `completed_cells`) are written, then push runs with
`check=True`. If push raises, run() dies. On resume, the all-done scan at
part85.py:349-364 returns early (`all {n} cells already complete`) before any push, so the
unpushed cell stays unpushed for the rest of the session. Combined with F1 the result is
silent: the session exits 0 having never pushed.

MEASURED: same t11 output above — the commit exists locally, the branch is absent, and the
next push prints "no new checkpoint result changes to push" with rc 0.

Smallest fix: push once at the end of the all-done branch too (a cheap no-op when HEAD ==
remote), or make the per-cell push failure non-fatal but mark run.json so the next push
retries. Fixing F1 makes the end-of-run push actually re-push.

### F3. MEDIUM, INTRODUCED — main preflight passes for a checkpoint with no step files of its own

`part85/handoff85.py:272-326`

preflight("main") verifies only REFERENCE_FILES (instruct/step0.json, instruct/step1.json,
neutral-token-ids.json, pilot-score.json) plus the manifest. It never checks the
checkpoint being run. CLOUD-RUN.md:140-149 tells each main session to restore its own
branch and run preflight "before downloading a model"; the promise is that a bad handoff
is caught before the multi-GB download.

MEASURED (scripts/t6_preflight_gaps.py):

```
(b) main preflight for a checkpoint with NO step files of its own (sft):
    rc= 0 main preflight passed for sft; verified 4 files
(d) main preflight for pretrain_4T whose step1.json is really instruct's:
    rc= 0 main preflight passed for pretrain_4T; verified 4 files
```

(d) is the worse half: a pretrain_4T step1.json carrying `checkpoint.key == "instruct"`
passes.

The cost is real but bounded: part85.py checks the step1 *commit* before the model load
(part85.py:313-314), so a wrong-commit step1 dies cheaply. But part85.py never checks
step1's `checkpoint.key`, while score85.py now does (score85.py:280-288) — the two
validators disagree. MEASURED (scripts/t17_step1_key_unchecked.py), a step1.json with
`key: "instruct"` and sft's commit:

```
raised: AssertionError passed the step1 identity check
reached tokenizer load (step1 key mismatch accepted)? True
```

So a step1 file from the wrong checkpoint is accepted by the run and only the scoring step
rejects it — after the whole checkpoint has been generated. (In practice key and commit
are redundant, since no two checkpoints share a commit, so this is a consistency gap rather
than a live wrong-result path; that is why F3 is MEDIUM and not HIGH.)

Smallest fix: in the main branch of preflight(), require `<checkpoint>/step0.json` and
`<checkpoint>/step1.json` when they exist to carry `checkpoint.key == checkpoint` and a
commit starting with `BY_KEY[checkpoint].commit_prefix` (the same checks score85.py
`_step_data_checked` already applies); and add the same `key` check to part85.py:313.
For Instruct/pretrain_4T where CLOUD-RUN.md says the step may be skipped, check the file
if present, require it if the checkpoint is not one of those two.

### F4. LOW, INTRODUCED — pilot preflight for instruct requires nothing

`part85/handoff85.py:276-283`: `required = [] if checkpoint == "instruct" else [...]`.

MEASURED (scripts/t6_preflight_gaps.py):

```
(a) pilot preflight, instruct, EMPTY results root:
    rc= 0 pilot preflight passed for instruct
```

This is defensible (Instruct is the first thing to run, so there is nothing to check), and
CLOUD-RUN.md does not claim otherwise. Listed only so it is not mistaken later for a check
that ran. No fix required.

### F5. LOW, INTRODUCED — refs-manifest hashes whatever tree it is pointed at

`part85/handoff85.py:237-269`

The manifest is self-consistent (every listed file is hashed and preflight re-verifies the
same hashes), so it proves "these bytes have not changed since the manifest was made", not
"these are the bytes the pilot branch produced". A manifest generated from a tampered
local tree passes preflight against that same tampered tree.

MEASURED (scripts/t6_preflight_gaps.py, case c): `rc= 0 main preflight passed for sft;
verified 4 files` where every listed reference file contained `"commit": "TAMPERED"`.

Why it is only LOW: CLOUD-RUN.md:130-135 has the researcher commit `references.json` and
the listed files at the repository root, so the git commit is the real integrity anchor.
Worth a sentence in CLOUD-RUN.md rather than code.

### F6. LOW, INTRODUCED — the pilot branch never carries pilot-score.json or pilot-frame-v2.txt

`part85/handoff85.py:100-105` (`_allowed_push_path`) and `part85/handoff85.py:329-370`
(`assemble`).

push() only stages `results/part85/<checkpoint>/` (plus Instruct's token IDs), and
assemble() only gathers `results/part85/<checkpoint>/...`. The pilot score and the v2
template are therefore never on the branch and never restored.

MEASURED (scripts/t5_amend_refs_manifest.py):

```
pilot-score.json restored? False
pilot-frame-v2.txt restored? False
```

Why it is only LOW: CLOUD-RUN.md:130-135 commits both at the repository root, so main
sessions get them from PINNED_COMMIT, not from the branch. The gap is that the amend
section (CLOUD-RUN.md:97-111) tells the session to `restore` the pilot branch in a new
clone and then run attempt 2 with `--plain-template-file results/part85/pilot-frame-v2.txt`
— a path the restore does not create. In the same session the file survives (T7:
`pilot-frame-v2.txt still on disk? True`); in a fresh clone it does not.

Smallest fix: either add both to the pilot branch's allowed paths, or change
CLOUD-RUN.md:95 to say the template must be re-created in the new clone.

### F7. LOW, INTRODUCED — every push uses the same commit subject

`part85/handoff85.py:223`: `_git("commit", "-m", f"Part 85 results: {checkpoint}")`.

MEASURED (scripts/t2_push_commit_msg.py), four cells pushed from one session:

```
Part 85 results: instruct
Part 85 results: instruct
Part 85 results: instruct
Part 85 results: instruct
```

Not a correctness defect (contents differ, the merge path keys off paths not messages),
but it makes the branch history useless for answering "which cell landed when" during an
incident. Smallest fix: include the cell key or a timestamp in the subject.

## What I checked and found sound

- **The run() smoke test can now see run() defects.** It drives `part85.main(argv)`, reads
  the answer files and run.json, and its expected values (unit 6978, clamp 9.75, control
  7884/3.0, seed 85000+100*qi+s, template SHA) are written in the test, not imported from a
  Part 85 helper (test_part85.py:616-645). The observer pre-hook is registered after the
  clamp hook and asserts the observed values, so a clamp that stops clamping fails
  `_assert_observers` (test_part85.py:647-657) — the r2 #2 class is closed.
- **Restore is additive, not destructive.** CLOUD-RUN.md:146-147 restores pilot then
  `<key>`; the second restore does not delete files the first supplied (T1:
  `LOST after second restore: []`).
- **The rebase/merge path works for disjoint cells.** A stale session pushing after
  another advanced the branch merges run.json `hook_fire_counts` / `completed_cells` and
  keeps both cells (T3, T4: remote run.json ends with cell0..cell3). Restored-but-untracked
  files are moved aside and restored (handoff85.py:161-196) without being lost.
- **The amend flow works end to end** (T8): assemble → prepare attempt 2 → score attempt 2
  → copy canonical → re-assemble → refs-manifest all succeed, the canonical score is
  attempt 2, and the manifest carries the v2 template SHA. assemble() does not clobber
  attempt-1 files, the key, the template or the score (T7).
- **Assemble refuses ownership violations and byte conflicts** (test_part85.py:1056-1080),
  and refuses a branch with no results for its checkpoint (handoff85.py:364-365).
- **Snapshot provenance** (checkpoints.py:68-90) refuses a directory that does not name the
  pinned commit, and is wired into all three step scripts before the model load.
- **Step-1 validation now precedes the model load** (step1_acts.py:179-203), with a test
  that makes the load raise if the refusal happens late.
- **The step1 resume guard hashes only clamp inputs** (part85.py:205-215), so a device or
  library change no longer forces a fresh results root.
- **score85 chat-only scope** returns NOT APPLICABLE for 85.5-85.9 and requires step files
  only for SFT/DPO/Instruct (score85.py:266-292, 544-547); seed/pair validation is real
  (score85.py:130-150).
- **The datasets pin correction is consistent**: CLOUD-RUN.md:12-21 pins 5.1.0 and states
  why 5.0.1 cannot install beside hub 2.2.0. I did not re-run the pip resolution; the
  claim is plausible and the commit message records a host load.
- **28 tests pass**, none skipped, on the host venv.

## Limits of this review

- I did not run any real model, real snapshot download, or real WikiText load; those need
  network and multi-GB downloads. Every measurement above uses the real scripts with
  temporary git remotes and small synthetic answer files.
- I did not test a genuine concurrent push race (two processes pushing at the same
  instant); the stale-then-push sequence (T3/T4) exercises the same rebase code path.
