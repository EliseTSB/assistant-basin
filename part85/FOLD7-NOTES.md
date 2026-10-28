# Part 85 fold 7 notes

Production side only; no GPU run, checkpoint download, commit, or push.

## Changes

1. `part85.py` and `audit85.py` independently parse the single Amendment 3 line in `PART85-DESIGN.md`. Attempt 1 stays on the original frame; attempt 2 requires the valid amendment; main uses it when present. Both reject duplicate, malformed, hash-mismatched, CR-containing, or unchanged frames. These two CLIs no longer accept operator template-file options.
2. Template hashes use UTF-8 frame bytes with LF endings and reject carriage returns. `handoff85.py` canonicalizes CRLF export files before hash comparison and copies the LF form into references; isolated carriage returns are refused.
3. The audit rejects unused generate records, any generate record containing an `error` key (including `null`), and forward records. The forward refusal was already present; explicit regression coverage was added.
4. `LOCAL-RUN.md` now gives attempt-2 production, audit, and replay commands after registering and committing the amendment. Main commands cover both gate branches; replay relies on the default `part85\references.json` manifest.
5. `download85.py` debits each bounded read before calling `read`; a simulated-clock stall test verifies the debit timestamps.

## Verification

- `python -I -B part85/test_part85.py`: 46 tests, pass.
- `python -I -B part85/test_download85.py`: 11 tests, pass.
- `python -I -B part85/mutate85.py --jobs 2`: baseline 46 tests; 44/44 mutants killed, none survived or errored. Both fold-7 mutations were killed.
- After the final runbook edit, its focused documentation test passed (1/1). AST parsing and `git diff --check` passed.
- `observe.py` remains SHA-256 `f6c1d7e16f4ccbb083e2a7ff8fa99d2f18e08be4504696e25f2c5b9821082613`.

## Not applied

No requested item was skipped. `replay85.py`, `test_replay85.py`, and `PART85-DESIGN.md` were left untouched as assigned; the design amendment itself is added by the operator only if gate 1 fails.
