# Part 63 fold 3c report

## SUMMARY

Completed the round 5 money-path fold from the binding parent decisions. The manual `submit` CLI now builds and prints its plan without creating review state, reserving funds, or posting. The shared ledger settles actual spend above an attempt reserve only when both aggregate and scoped caps allow it; otherwise the actual remains recorded as an `overrun` and blocks further reserves. Retry submission is serialized around its scan and POST, with a rescan under `run.lock`.

## FINDINGS

- **Manual resubmit:** `submit` CLI is plan-only; only `run-all` invokes the paid submission lifecycle. A regression test checks no review directory/reserve is created, and a copied-code mutation restoring the paid route goes red.
- **Settlement over reserve:** removed the reserve amount as an independent rejection condition. Tests cover above-reserve settlement under both caps and overrun above a sub-cap with a later reserve refused. The run-all recovery path prints the recorded amount, attempt, and an explicit stop/escalation instruction for an overrun. A run-all fixture with above-reserve actual usage confirms the next chunk advances while both caps permit it.
- **Concurrent retry collection:** `collect` holds `run.lock` across its retry scan and submit, and rescans after acquiring the lock. The lock is reentrant for nested `submit` calls. Two synchronized collectors generate exactly one retry POST. Atomic JSON temporary names include thread identity to avoid same-process concurrent rename collisions found while exercising the race.
- **Retry kill boundary:** added a fake transport test that raises immediately after retry `submit()` returns, resumes `run-all`, and asserts the retry is posted once total. A copied-code mutation replacing `if existing_retries:` with `if False:` makes the row fail.
- **Reserve comment:** corrected `mark_interrupted_attempts` documentation to say reserves remain counted.
- **Runbook:** added exact recovery guidance for ambiguous/no-ID POSTs, unsettled attempts, overruns, and retries in flight. Recorded that $8 is a reserve ceiling, not a billing ceiling, and expected spend is about $1.12 of the $78 credit.
- **Paid-action test:** the run-to-ledger registry test now overrides its registry and lock paths under its temporary directory; it does not write to the live registry.
- **Deterministic row selection:** sorted the run-all glob and fail closed if multiple submission rows exist for one chunk. Preserved the `--ledger-path` interfaces.
- **Documentation:** corrected the paidaction fold and build report to describe cap-aware settlement and final overrun accounting.

## BLOCKERS

None. No parent-control messages were pending or received, so no messages were applied, declined, or deferred.

## UNVERIFIED

No live API calls, billed usage, or network operations were performed. Provider-side behavior and inherited prices/token estimates remain unverified. The new verification is fake-transport and local-ledger coverage only.

## VERIFICATION

- `python3 -m unittest -v` in `part63`: 25 tests passed, including retry concurrency, post-return kill recovery, copied-code mutations, and run-all advancement above reserve.
- `python3 -m unittest -v` in `paidaction`: 22 tests passed.
- `git diff --check` in both domains: clean.
