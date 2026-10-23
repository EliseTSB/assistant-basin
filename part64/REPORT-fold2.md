# Part 64 fold 2 report

## SUMMARY

Amendment 2 is implemented in the Part 64 writer plan and recovery path. Each planned ledger share is checked at its complete max-token reserve and stays below the paidaction $8 ledger cap; Opus 1x groups now use three ledgers and Opus 3x groups use six. Before each new reserve, gen64 checks settled plus reserved totals across every existing ledger for that writer stage against Haiku's $4 cap or Opus's $28 cap. The staged corpus path, ambiguous re-probes, and Unicode line separators are covered by regressions. The full Part 64 test suite passes: 42 tests. No paid API, network, GPU, or commit was used.

## FINDINGS / WHAT SHIPPED

- **Ledger sizing:** Opus 1x uses three balanced ledger partitions; Opus 3x retains six. Plan construction calculates the whole partition's maximum reserve and refuses any share at or above $8. Existing request chunks remain limited to a $1 reserve target. Tests cover both Opus sample sizes and sum the actual built plan's reservations per ledger.
- **Writer stage guard:** `WRITER_STAGE_CAP_USD` registers Haiku at $4 and Opus at $28. The guard scans every ledger file for the writer in the run root, includes the current ledger, sums each ledger's settled and reserved amounts, and checks the new reserve before calling `Ledger.reserve`. A run-root lock serializes this check with other reserves. A regression leaves one ledger settled and another reserved after a simulated crash, then verifies a reserve on a third ledger is refused at the cap boundary without a POST.
- **Staged corpus path (Luna F1):** Manifest validation no longer requires the corpus path to equal the machine-local registered path. It still requires a nonempty recorded path and validates registered size and SHA-256 prefix along with the other gates. A loader test uses the same caller-supplied path that run64 passes to collection verification and assembly.
- **Re-probe recovery (Haiku F3 and Luna F2):** Each authorized re-probe receives an attempt-specific custom ID, so its digest and provider request identify only that attempt. Recovery ignores ended batches created before the existing five-minute uncertainty window. Tests verify accepted/lost re-probe recovery with an earlier batch present and separately prove an old ended batch is not adopted even if it carries a matching ID.
- **Unicode separators (Haiku F4):** Collection JSONL is split on LF only. A regression loads a row whose string contains U+2028, U+2029, and U+0085.
- **Documentation:** The runbook now records partition counts and stage caps; the generator module description reflects multi-ledger groups.
- **Parent control:** No mailbox messages were received, so no plan changes, declines, or deferrals were needed.

## FILES TOUCHED

- `part64/gen64.py`
- `part64/test_gen64.py`
- `part64/RUNBOOK-64.md`
- `part64/REPORT-fold2.md`

## TESTS

Command run from `part64/` with the worktree on `PYTHONPATH`:

```text
PYTHONPATH=/home/elisepc/green/worktrees/ab-part64 /home/elisepc/neuron-probe/.venv/bin/python -m unittest
Ran 42 tests in 3.395s
OK
```

## BLOCKERS

None.

## SELF-ASSESSMENT

The stage guard and re-probe changes were exercised only with local fake providers and append-only test ledgers. Per assignment, no live API behavior or billing was verified. The reservation bounds use the repository's inherited pricing assumptions; they enforce the registered limits over those estimates, but do not validate current provider prices.
