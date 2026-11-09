# Brief: fold r5 of Part 198's run and score code (no model runs)

The design is the contract: `/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3). The code is
`results/part198/run198.py`, `score198.py` and `test_build198.py` at `53284e1`. `BUILD-NOTES.md` records every earlier
fold. `stats198.py` is frozen: import it, never edit it. Its SHA256 must stay
`03e65cb44ab7496d064b497f1e9bb38086716e74b02b5dc3ab998f881473ee64`.

The review to fold: `/home/elisepc/green/Journal/reviews/cheap-part198-code-r5-luna-luna-r1.md`. Its probes
(`probe_concurrent_score_read.py`, `probe_sequence_failure_score.py`, `probe_lock_crash_release.py`) are under
`/home/elisepc/green/DiscordBot/df-runs/cheap-review-part198-code-r5-luna-luna-r1-2026-11-08/`. Copy them into
`results/part198/` before running them.

## What to fix

1. **HIGH: the scorer must never change a log, and must not read one a runner is writing.** Two parts, both required:
   - A read path that writes is the root defect. Split `read_jsonl` into a pure reader and the runner's repair. The pure
     reader is used by the scorer and by every read that is not the runner's own resume. It never truncates or rewrites
     a file. It ignores an unterminated final line and reports that it did. The repair (truncate an unterminated tail)
     stays only on the runner's own resume path, under the runner's lock. Grep for every caller of `read_jsonl` and
     list each one in BUILD-NOTES with the variant it now uses.
   - `score_from_dir()` and the scorer CLI take each log's lock in a fixed order, and refuse at once with a clear error
     if a runner holds any of them. The scorer does not wait. Scoring a run in progress is refused, never guessed.
2. **MEDIUM: 198.8 needs ten completed sequence cases.** It is NOT EVALUABLE unless there are exactly ten completed
   unique sequence case rows and no registered failure row in the full sequence log. A failure row for any case makes
   it not evaluable, even when a completed row for that key also exists. Read the design's text for 198.8 and the
   gate, and apply the same rule to any other forecast whose denominator counts selected rows that can include
   failures. Find these with a grep and list them in BUILD-NOTES.

## Tests (each must fail on `53284e1`'s code: run the new test file against copies of the `53284e1` versions kept in your
run directory, and record which fail in BUILD-NOTES)

- The scorer, run while a child process holds a log's lock, refuses, and the log's bytes are unchanged (compare the
  hash before and after).
- The pure reader, given a log with a partial final line, returns the complete rows, flags the partial line, and leaves
  the file byte-identical.
- Nine completed sequence cases plus a failure row for the tenth make 198.8 NOT EVALUABLE.
- The r5 review's two behavioural probes now report refusal and not-evaluable, and the crash-release probe still passes.

## Hard limits

- **Do not load any model or weights, run any edit, or call any network.** Any test or probe that could reach a loader
  (`main()`, `run_arm_*`) must first patch every loader, `torch.load`, `from_pretrained` and `safe_open` to raise.
  Check this before running anything. If you cannot, do not run it.
- Write only inside `results/part198/`. Do not commit; the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I`.

## Definition of done

- `fixtures198.py` and `test_build198.py` exit 0, with the check count printed.
- BUILD-NOTES gains a "Fold r5" section covering each finding, the caller inventory, the tests and the old-code failures.
- Write your report as `report.md` in your run directory (SUMMARY, FINDINGS, FILES TOUCHED, BLOCKERS,
  SELF-ASSESSMENT). Write a skeleton early and fill it in as you go.
