# Part 201 runner report

## SUMMARY
Implemented the CPU score recorder, deterministic standard and smoke plans, common decode path, run records, call-atomic recovery, G5, G4 checks, and float32 timing probe.

## FINDINGS / WHAT SHIPPED
- `MarginRecorder` performs one full score transfer per step, retains only the pending step's full CPU scores, and reports emitted-token logits, top-two data, nonfinite state, and finish reason. `bf16_step` is checked against bf16 `nextafter` spacing.
- The standard plan has A 384/48 calls, A-again 16/2, B 384/48, M 408/51, C 216 single-row calls, A2 96/12, F8 96/12, and F1 48/6. A and B satisfy the membership and slot constraints. `--print-plan` writes once and refuses a changed frozen plan.
- The runner left-pads prompts, passes attention masks, uses one generation path, retries nonfinite calls, appends call rows after generation, removes incomplete calls on startup, and skips completed pass/row pairs. Pass order is A, A-again, B, M, C. A-again is omitted and recorded as `not same process` when its first two A calls were not made in the current launch.
- G5 compares token IDs from Part 163's default generator against `generate_rows`. G4 records activation dtype and per-row push difference, and checks push-hook firings against generated forward passes. Float32 runs record F8/F1 projections and apply the specified cut. The float32 timing probe records five synchronized no-cache forward timings for each configured row-count/prefix combination in `probe.json`.
- Smoke planning covers four sampled units, two identity and two maker questions, and four plain rows across all passes. The tiny-Qwen3.5 CPU tests exercise mixed prompt padding, G2, G5, and float32 upcast/G4.
- `python -I -B -m pytest -q tools/test_margin_recorder.py part201/test_part201.py`: **17 passed in 3.88s**. `py_compile` passed. `--print-plan` emitted SHA-256 `d9e7a3fad0084d3e080c3212ce53db1a2bb388fe872f267dc1fc13f57b40779a`.
- No GPU experiment or network access was run.

## Amendment 1
1. Frozen plan: implemented plan writing, hash recording and pre-load mismatch refusal.
2. Call atomicity: implemented tail repair and dropping every row of an incomplete call; relaunches regenerate full original calls.
3. Launch index: deferred. The amendment requires posting the `launch` schema proposal to `part201-rows-schema` first, but no `coord_post` or `coord_read` tool/interface is exposed in this worker. The shared schema and emitted rows therefore do not yet include `launch`.
4. Pass order and A-again: implemented the amended order and the not-same-process skip/record.
5. Timing probe: implemented the float32 full-prefix probe and F8 cached per-step timings; it was not run against the experiment model.
6. G4 counts: implemented per-layer hook firings and generated forward-pass counts, with equality checked in float32.
7. Pass K: retained its record path and added a step comment requiring a hand rename before rerun if the output already exists; it is never deleted.

## FILES TOUCHED
- `tools/margin_recorder.py`
- `tools/test_margin_recorder.py`
- `part201/run201.py`
- `part201/test_part201.py`
- `part201/steps.txt`
- `part201/REPORT-run.md`

## BLOCKERS
HOLD: the shared `part201/ROWS-SCHEMA.md` needs a coordinated `launch` integer field before row output can satisfy Amendment 1. The required coordination interface was unavailable, so I did not alter the shared schema or add the unapproved field to row output. The exact question is in `HOLD.request`.

## SELF-ASSESSMENT
The CPU tests validate the deterministic plan and tiny Qwen3.5 generation path, but no actual A770 launch validated eager loading, device synchronization, runtime projections, or the full timing probe. The B redraw count records the maximum per-question conditional redraw attempt. The `launch` field remains outstanding pending coordination.

SUMMARY: Recorder, runner, plan, smoke plan, launch steps, and CPU tests shipped; the schema-dependent launch index is on HOLD.
FINDINGS: 17 tests passed; standard plan hash is d9e7a3fad0084d3e080c3212ce53db1a2bb388fe872f267dc1fc13f57b40779a.
BLOCKERS: HOLD — can the coordinator expose `coord_post`/`coord_read`, or authorize the schema proposal and `launch` row field without that topic post?
