# Part 64 fold 1 report

## SUMMARY

Fixed the introduced findings from both round-1 reviews across collection validation, generation recovery, paid-ledger
guards, scoring, analysis, wrapper status, and documentation. Added regression coverage for each actionable finding,
including the real `L1:c69` Part 60 pilot prefix and both lost-response recovery paths. All 36 Part 64 tests pass.

## FINDINGS

- **Haiku F1 — real collection refusal.** `part64/gen64.py:152-162` now permits duplicate text inside the exact Part 60
  pilot prefix while requiring fresh windows to remain unique and disjoint. `part64/test_gen64.py:234-261`
  loads the real `L1:c69` dev prefix and preserves its repeated top windows; the neighboring fixture test also proves
  fresh duplicates are rejected.
- **Haiku F2 / DS-F2 — errored rows wedged chunks.** `part64/gen64.py:549-616` records terminal non-succeeded rows as
  non-passes, sums usage only from succeeded rows, and settles the chunk. `part64/test_gen64.py:512-526` verifies the
  failed label is refused by scoring and only the successful row's usage is billed.
- **Haiku F3 — lost-response POST retirement could resubmit.** `part64/gen64.py:924-1008` checks a complete provider
  listing before retirement and discovers matching run-specific IDs on resume; an ended accepted batch is adopted
  and settled. `part64/test_gen64.py:435-450` checks refusal to retire and recovery without a second POST; lines
  481-510 also cover refusing an in-progress candidate and allowing retirement only after a listing proves absence.
- **Haiku F4 — copied run roots bypassed the run guard.** `part64/gen64.py:667-675` passes a stable ledger-scoped
  `run_id` into the paid ledger. `part64/test_gen64.py:569-584` copies a plan and verifies its second root cannot
  reserve or POST. Plan reinitialization preserves run-specific IDs (`test_gen64.py:304-313`).
- **Haiku F5 — max-token truncation was silent.** `part64/gen64.py:600-608` marks truncated results `max_tokens`, which
  scoring refuses; `part64/analyze64.py:154-155,258-265` reports counts per arm in JSON and Markdown.
  `test_gen64.py:528-536` and `test_analyze64.py:46-52` cover refusal and counting.
- **Haiku F6 — Opus 3x groups could exceed the $8 cap.** `part64/gen64.py:307-310,377-407` splits each Opus 3x type
  across six per-ledger partitions of at most 42 units, leaving each partition below the hard cap at the full
  per-request token reserve. `part64/test_gen64.py:615-627` checks partition counts and reserve totals.
- **DS-F1 — scorer saw marked raw sentences.** `part64/score64.py:130-131` now uses Part 60's `strip_markers` output
  and marked span. `part64/test_gen64.py:355-372` asserts the exact clean text and span passed to the hook.
- **DS-F3 — crash after ledger release.** `part64/gen64.py:674-682` reconstructs a known rejection from a released
  ledger attempt instead of treating it as an unknown POST. `part64/test_gen64.py:539-561` simulates the crash and
  confirms resume marks the chunk rejected without another POST.
- **DS-F4 — probe-gate test gap.** `part64/gen64.py:896-906` retains the probe gate before group execution.
  `part64/test_gen64.py:586-596` calls `run_generation` with failing probes and confirms the only POSTs are the two
  probes.
- **DS-F5 — wrong default paths.** `part64/gen64.py:38-39` now points at the collection and generation paths used by
  `run64.sh`. `part64/test_gen64.py:178-195` checks those defaults.
- **Haiku F7 — failed probes could not be retried.** `part64/gen64.py:808-850` adds an operator-authorized re-probe
  with a new attempt number, recorded operator/reason, and durable manifest pointer. `part64/test_gen64.py:599-612`
  covers an invalid initial probe followed by a successful recorded retry.
- **Haiku F8 — upper median.** `part64/collect64.py:128` now uses `statistics.median`, so 250 scores of 9 and 250 of 10
  yield 9.5 and fail the gate. `part64/test_collect64.py:54-74` covers that exact boundary.
- **Haiku F9 — trusted manifest.** `part64/collect64.py:138` records the collection SHA-256; `part64/gen64.py:120-180`
  verifies the hash and recomputes the median from row values. `part64/run64.sh:158` applies the full loader during
  cache reuse. `part64/test_gen64.py:268-291` checks hash tampering and a false median claim.
- **Haiku F10 — skipped scoring returned ready after generation.** `part64/run64.sh:193-194` fails the scoring phase
  when output is absent after generation has started or completed; `part64/run64_helpers.sh:3-21` distinguishes an
  initial collection-only return from a touched/complete generation. `part64/test_run64.py:21-44` covers planned,
  settled, complete, and generated markers.
- **Haiku F11 — stale build report and synthetic-only data coverage.** `part64/REPORT-build.md:4-16` replaces the
  earlier claim that the pipeline was not built. `part64/test_gen64.py:197-200` guards against that stale claim, and
  `test_gen64.py:234-261` exercises the real Part 60 pilot duplicate.
- **Haiku F12 — 64f interpretation caveat.** `part64/RUNBOOK-64.md:48` and `part64/analyze64.py:224-226` state that 64f
  compares the matching 1x text under different token limits and thinking settings, so it is a cross-configuration
  check rather than a pure reproduction. The registered prediction text in `analyze64.py:19-26` was not changed.

The reviewers reported different recovery failures under F3: DeepSeek measured a crash between ledger release and
the local rejection write, while Haiku measured a lost-response POST being retired and resubmitted. Both measured
paths were fixed; neither was treated as a substitute for the other.

## BLOCKERS

None for this offline code and test task. No paid API call, network access, or GPU run was made, as required. The
registered model/corpus production traversal was not run here. Parent-control checks were empty at each checkpoint;
no messages changed the plan and no dispositions were needed.
