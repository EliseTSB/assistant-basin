# Paid-action fold report

## SUMMARY

Implemented the three shared-ledger fixes identified by Part 62 review: refused and recorded aggregate settlement overruns, atomic per-stage sub-cap reservations, and a checkout-independent canonical default ledger path with run-ID path collision refusal. Added copied-code mutation coverage for each fix and kept the existing suite green.

## FINDINGS

- **M6 — settlement overrun:** `settle()` evaluates actual dollars against all other settled spend and active reserves and against the attempt's per-kind sub-cap while holding the ledger lock. An actual amount above its reserve settles when both caps still permit it. Otherwise it fsyncs a `settlement_overrun` event with the actual amount and usage digest, raises `CapExceededError`, and leaves the attempt overrun so later reservations remain refused.
- **L3 — per-stage sub-cap race:** `reserve()` accepts paired `subcap_usd` and `subcap_key` arguments. It checks settled and reserved amounts for that key under the same file lock as the aggregate check, requires one consistent cap per key, then records the scoped reservation in the same fsynced reserve event. The regression test starts two real processes with aggregate headroom for both but only enough sub-cap room for one; exactly one reservation succeeds.
- **L4 — ledger identity:** `DEFAULT_LEDGER_PATH` is fixed at `/home/elisepc/assistant-basin/paidaction/paid-ledger.jsonl`, independent of the checkout containing an imported copy. A durable locked registry associates caller-supplied `meta.run_id` values with one resolved path; a second distinct path for that run ID raises `LedgerError`.
- Updated `REPORT-build.md` to describe these APIs and the refreshed test suite. There is no README in `paidaction/`; the existing build report was the available library documentation.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v` completed successfully: **18 tests passed**. `git diff --check` was clean. The copied-code mutation test now covers all three new fixes as well as the existing finite-cap and error-body guards.

## BLOCKERS

None for the shared-library implementation. Callers enforcing a stage budget must pass the new `subcap_usd` and `subcap_key` arguments to `reserve()`; Part 62's caller code is outside this task's declared write domain and was not changed.

## UNVERIFIED

No API or network calls were made. The new sub-cap API was verified directly through `Ledger.reserve()` and a two-process race; integration with Part 62's `_StageLedger` caller remains outside scope. The canonical path intentionally identifies the shared assistant-basin checkout location and assumes that all relevant workers can access it.
