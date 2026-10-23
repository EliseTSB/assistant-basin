# Part 64 fold 5: pipelined chunk submission

## SUMMARY

Implemented Amendment 6's chunk pipeline for Part 64 `run` and `resume`. Chunks submit in manifest order until the first stage-cap refusal, then the driver collects submitted batches with one status check per pass and resumes submission as settled reserves become available. The writer-stage guard now raises `StageCapRefused`, a distinct `WriterError` subtype, without changing its existing diagnostic message. Group output is assembled only after all chunks in that group have settled.

## FINDINGS

- Added `execute_chunk(..., submit_only=True)`. It writes the submitted state and batch ID durably before returning; already submitted and settled chunks are unchanged in submit-only mode. Existing settlement and recovery states continue through the normal path.
- Added a 60-second default pipeline poll interval and a 30-hour overall pipeline deadline. A batch that remains in progress stays in the collect set instead of ending the run at a per-batch timeout.
- The submission pass stops on the first `StageCapRefused`. Only that exception is handled as a stage-cap wait condition; other `WriterError` cases, recovery failures, POST failures, and invalid status payloads propagate.
- `resume` reuses durable batch IDs, so interruption after submission does not create another POST. Group JSONL output is written after every chunk in the group is settled, and manifest completion is recorded afterward.
- Updated section 3 of `RUNBOOK-64.md` with pipeline timing, reserve release, output, and resume behavior.

### Tests

Added tests using fake batches that settle after configurable status checks:

- `test_pipeline_has_multiple_chunks_in_flight_and_reserve_stays_within_cap`
- `test_stage_cap_refusal_stops_submit_pass_then_retries_after_settlement`
- `test_pipeline_propagates_non_guard_writer_error` (also checks invalid status propagation)
- `test_resume_after_submit_pass_does_not_duplicate_posts_and_matches_uninterrupted_output`
- `test_group_output_waits_until_every_chunk_is_settled`

Verification passed:

```text
PART64_TEST_WRITER_STAGE_LEDGER_DIR=/tmp/part64-fold5-worker01-stage \
/home/elisepc/neuron-probe/.venv/bin/python -m unittest \
  part64.test_collect64 part64.test_collect64_model part64.test_gen64 part64.test_analyze64
Ran 58 tests ... OK
```

`git diff --check` passed. No paid calls, network calls, or commits were made. Tests used temporary directories and the test writer-stage override; no files under `results/` were read or modified.

## BLOCKERS

none

## SELF-ASSESSMENT

The fake API tests verify concurrent submissions, bounded reserves, one POST and one ledger settlement per chunk attempt, cap refusal and retry, interruption recovery, output ordering, and error propagation. They do not exercise live Anthropic service behavior or simulate the full 30-hour wall-clock bound.
