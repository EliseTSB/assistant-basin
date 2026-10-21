# Part 43: FP32 successor protocol (registered 2026-10-21 16:20 CT, before any FP32 collection)

Written by Claude after Alden's answer in council topic part43-google-flip-review-2026-10-20 #23 (accepted contract,
quoted in full in ALDEN-23.md beside this file). Status: protocol only; no implementation, no rental, no model call yet.

## What stays as measured
- Commit 845964c and its 3090 receipts at e1ac714 (results/part43-pilot-3090/) are kept unchanged. The bf16 outcome is
  HOLD: epsilon 0.10916 nats > 0.05, all of it the bf16-vs-fp32 term (bf16 repeat spread 0 over three repeats and
  clean-to-clean replay 0 in these cells on this execution path; FP32 had one repeat, so its repeatability is
  unmeasured). These values are assay-development evidence, not a blind preregistration or a passing FP32 pilot.
- Scientific thresholds and the 0.05-nat budget are unchanged.

## What changes, prospectively
1. **Decision dtype: FP32**, executed, not labelled. Loading, every intervention and receiver trajectory, and every
   scoring and readout path the pilot uses run in fp32. The dtype is bound in the registration, the raw schema, the
   reader and the recorded execution metadata. No bf16 leg is produced or invented to satisfy the old schema.
2. **Full-FP32 arithmetic:** TF32 off for both CUDA matmul and cuDNN convolution; the effective flags and backends are
   recorded, and any backend that downcasts internally is named, not relabelled. (Receipt e1ac714 had matmul TF32 off
   and cuDNN TF32 on.)
3. **New calibration definition (defined now, before collection):** at least three FP32 repeats of every registered
   cell, all registered full answers, plus clean-to-clean replay and the L7-zero/L11-restored rescue control, through
   the same raw-only writer, disk readback and derived-report boundary. epsilon = max over registered cells of
   (FP32 repeat spread of the primary score, clean-to-clean replay error), compared exactly against 0.05.
4. **The bf16-vs-fp32 difference** becomes a retained cross-method diagnostic (the e1ac714 values), not an error term
   for a path that no longer uses bf16.
5. **Offline controls before any rental:** wrong or mixed execution dtype refuses; the old (bf16) protocol refuses
   under the new reader; a genuine FP32 registration passes; each new epsilon component alone can hold the gate;
   derived, duplicate, malformed and internal-error cases keep their current distinctions; no production stage runs
   before the reader's verdict.

## What this gate means
It is an amended **control and stability gate**: FP32 repeat and replay agreement measure stability and hook fidelity,
not all systematic numerical error. An epsilon of 0 from exact repeats is not a certified zero-width accuracy interval.
Before any confirmatory or full-run claim: an explicit numerical-reference and error policy, plus the held
c*epsilon, NOT_TESTABLE and rescue propagation. A whole-model FP64 run is not a prerequisite for the bounded pilot.

## Order
Protocol (this file) -> implementation on a branch from part43-singleton -> offline controls green on the host ->
cheap review -> Alden and Chorus -> one bounded FP32 calibrate-then-pilot by hand on a 3090.
