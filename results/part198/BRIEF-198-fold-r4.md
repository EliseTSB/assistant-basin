# Brief: fold r4 of Part 198's run and score code: replace the duplicate-row comparison with a simpler rule (no model runs)

The design is the contract: `/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3). The code is
`results/part198/run198.py`, `score198.py` and `test_build198.py` at `dbe6d56`. `BUILD-NOTES.md` records every
earlier finding. `stats198.py` is frozen: import it, never edit it. Its SHA256 must stay
`03e65cb44ab7496d064b497f1e9bb38086716e74b02b5dc3ab998f881473ee64`.

The review that prompted this: `/home/elisepc/green/Journal/reviews/cheap-part198-code-r4-luna-luna-r1.md`. Its probes
are in `/home/elisepc/green/DiscordBot/df-runs/cheap-review-part198-code-r4-luna-luna-r1-2026-11-08/` (search for
`probe_*.py`). Copy them into `results/part198/` before running them.

## Why this round is a redesign, not a patch

Four reviews in a row have found a new hole in one mechanism: the list of fields `registered_rows()` compares when two
terminal rows share a key. Every round has added fields, and every round a scored field has still been missing. In r4
that was the aimed control's real row shape, and mass-batch localization. A list of what to compare will keep missing
things. The parent's decision is to invert it:

1. **Compare everything except a short, explicit list of metadata.** Two terminal rows of the same `record_type` with the
   same key (case or batch) are a conflict unless they are equal in every field after removing an explicit allowlist of
   non-scored metadata. That list is timestamps, elapsed or duration fields, process id, host, and run or invocation
   id. Find the exact field names the runner writes and list them in BUILD-NOTES. Nothing else is ignored, raw floats
   included. A conflict raises `ValueError` and names the key and the first differing field path.
   - Keep the accepted sequence of a failure row followed by a completed row for the same key, where the design allows
     a retry.
   - The rule from fold r3 stands: a case or batch whose rows disagree is rerun, never resolved by picking one.
   - Delete the per-kind tuple machinery (`_case_scored_booleans`, the decisive tuples) that this replaces.
2. **One runner process per output log.** Take an exclusive `fcntl.flock` on a lock file beside the output log for the
   whole runner invocation, before anything reads the log. A second invocation on the same log refuses at once with a
   clear error, and never waits. This removes the Arm C concurrency race (r4's third finding) at its source. Arm C's
   cap then needs no further locking: keep fold r3's `attempt_started` counting.

## Tests (each must fail on `dbe6d56`'s code: run the new test file against copies of the `dbe6d56` versions kept in
your run directory, and record which fail in BUILD-NOTES)

- Two case rows differing only in the runner-shaped `aimed_positive_control.localization.localized` are refused.
- Two batch rows differing only in `localization.f1.localized` are refused.
- Two case rows differing only in a raw float are refused.
- Two rows differing only in allowlisted metadata are accepted, and the scorer's counts are unchanged.
- Failure then completed is accepted. Completed then completed with a difference is refused.
- A second runner process on a held log refuses. Hold the lock from a child process and do not run any arm.
- The r4 review's three probes now report refusal or the cap holding. In the concurrency probe, a second process is
  refused by the lock.

## Hard limits

- **Do not load any model or weights, run any edit, or call any network.** Any test or probe that could reach a loader
  (`main()`, `run_arm_*`) must first patch every loader, `torch.load`, `from_pretrained` and `safe_open` to raise.
  Check this before running anything. If you cannot, do not run it.
- Write only inside `results/part198/`. Do not commit; the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I`.

## Definition of done

- `fixtures198.py` and `test_build198.py` exit 0, with the check count printed.
- BUILD-NOTES gains a "Fold r4" section: why the comparison was inverted, the metadata allowlist, the lock, the tests,
  and the old-code failures. Correct earlier sections that describe the tuple comparison as current.
- Write your report as `report.md` in your run directory (SUMMARY, FINDINGS, FILES TOUCHED, BLOCKERS,
  SELF-ASSESSMENT). Write a skeleton early and fill it in as you go.
