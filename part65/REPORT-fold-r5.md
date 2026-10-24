# Part 65 fold r5

## SUMMARY

Shipped the requested F1–F4 judge changes, added the F7 probe duplication note to the runbook, and left F5, F6, and F8 unchanged. The full Part 65 suite passes: 54 tests.

## FINDINGS

- **F1:** Pinned the registered seed at `REGISTERED_SEED = 6501`, removed CLI `--seed`, and added a ledger latch across plan hashes for foreign active, released, retired, and overrun attempts. `test_released_reservation_latches_different_seed_plan_before_any_send` passes; the plan-hash-filtered mutant fails because the expected refusal is absent.
- **F2:** Strengthened the release regression to release chunk 0 of a three-chunk plan and assert one POST with chunks 1–2 still ready. `test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2` passes; removing the released stop makes it fail (`released` count 3 instead of 1).
- **F3:** Added a crash-after-provider-acceptance test that fails the `submitted` save, invokes the judge again, and requires one POST total. `test_crash_after_provider_acceptance_does_not_post_again` passes; removing the durable `posting` save makes it fail (2 POSTs instead of 1).
- **F4:** The chunk loop now stops on pending. `test_pending_chunk_stops_before_later_ready_chunks` passes; removing the pending stop makes it fail (`pending` count 3 instead of 1).
- **F7:** Added the requested runbook sentence documenting that the probe request is also the first full-plan request.

## BLOCKERS

None.

## Claude, after collection (2026-10-24)

Host: 54 tests, OK. Gate mutants of my own (no further review round, decided on the card): removing the stop after a release fails the release and latch tests; not saving the posting record before the send fails `test_crash_after_provider_acceptance_does_not_post_again`; restored by hash.
