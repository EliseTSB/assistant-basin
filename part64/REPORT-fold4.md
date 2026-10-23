# Part 64 Fold 4 Report

## SUMMARY

Implemented Amendment 4's four requested fixes in the Part 64 writer workflow. The default writer-stage ledger path is now machine-wide; probe input estimates scale with planned request bytes; STOP decisions are saved and enforced before writer stages; and the five-minute recovery window applies only to suffixed re-probes. Updated the runbook and added focused regression tests.

## FINDINGS / WHAT SHIPPED

1. **Machine-wide writer-stage ledgers.** The default is `/home/elisepc/assistant-basin/results/part64/writer-stage`, independent of checkout and run root. `PART64_TEST_WRITER_STAGE_LEDGER_DIR` supports isolated tests. `test_writer_stage_ledger_path_is_machine_wide_with_test_override` verifies both paths without writing the production directory.

2. **Byte-scaled probe projection.** The estimator multiplies probe billed input by total planned canonical request bytes divided by probe request bytes, while retaining probe output tokens times total request count. The run record stores the measured byte counts, scaled input estimate, projected cost, and Opus STOP decision. `test_probe_stage_projection_arithmetic_and_opus_stop_line` covers the arithmetic. The round-4 `m2_projection_bias.py` reproduction printed `$26.64` projected against `$26.62` expected and `projection / expectation = 1.00`.

3. **STOP enforcement and override.** `run` refuses a recorded Opus projection above `$36` without `--override-stop "reason"`; the override reason and projection are written to `run.json`. `resume` applies the same gate until writer-stage work has begun, then can continue the already-started stage. `test_run_and_resume_gate_recorded_stop_until_operator_override` covers refusal, recorded override, and the pre-stage resume check.

4. **Re-probe-only five-minute window.** Recovery checks the suffix on request custom IDs as well as the attempt number. A retired initial probe retry keeps unsuffixed IDs and therefore receives no time-window exclusion; suffixed re-probes still do. `test_retired_initial_probe_retry_does_not_get_reprobe_time_window` verifies the retry is refused when an earlier matching batch exists. `test_reprobe_recovery_ignores_older_ended_batch_even_if_id_matches` retains the suffixed re-probe case. The round-4 `m5_retry_after_retire.py` reproduction now reports the retried initial probe as `REFUSED`.

Validation: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part64.test_collect64 part64.test_collect64_model part64.test_gen64 part64.test_analyze64` completed with **50 tests, OK**. `git diff --check` passed. No paid API calls, network access, or GPU work were used.

## BLOCKERS

None.
