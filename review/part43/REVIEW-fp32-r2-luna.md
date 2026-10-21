<!-- cheap-review.mjs cheap-review-part43-fp32-r2-luna-r2-2026-10-21: /home/elisepc/green/worktrees/ab-43-fp32 18f8c41..HEAD, lane luna, df-run exit 0 -->
# Cold adversarial review: part43-fp32-r2

**Verdict: SHIP-WITH-FIXES**

## Findings

### MEDIUM — INTRODUCED — `part43.py:3577-3590` (calibration backend policy at `part43.py:5860-5898`)

`run_pilot()` rechecks the requested protocol and loaded parameter dtype, but does not apply the FP32 runtime restrictions to the model it is about to use. The raw-artifact reader rejects a calibration whose recorded runtime has hub kernels enabled, kernelized modules, or a non-reference Qwen implementation. A caller can pass a valid reader-produced summary and then run the pilot under a changed runtime; the second gate accepts it. `output_header()` reports that runtime after the stages have already run.

**MEASURED trigger:** I ran `repro_pilot_backend_bypass.py` from this review domain. It builds a synthetic raw artifact with the existing fixture, reads it through the real `_read_calibration_artifact()`, then changes `USE_HUB_KERNELS` from `0` to `1` before calling the real `run_pilot()`. The pilot's stage functions were stubbed only to mark entry; no GPU or hub kernel was run. Output:

```text
reader_gate: True calibration complete and within the 0.05 nat budget
pilot_status: complete
stages_entered: [('signs', '1')]
reported_hub_kernels_enabled: True
registration_allows_hub_kernels: []
```

Why it matters: a direct caller can obtain a `complete` FP32 pilot while using a backend forbidden by the accepted registration. The regular `main()` path creates and reads calibration immediately before loading and running the pilot, so this is chiefly a direct-call or runtime-drift gap; it is not a reproduction of an unsafe backend being enabled at the start of that CLI invocation, which the raw reader refuses. **Smallest fix:** before the first production stage, validate the currently loaded pilot model's effective precision flags, hub-kernel setting, kernelized classes, and Qwen implementations against the FP32 policy (and refuse on mismatch). Keep that check in `run_pilot()` so callers cannot bypass it.

## What I checked and found sound

- The FP32 reader binds schema, protocol, decision dtype, registration, and source provenance; the FP32 path requires all 32 registered down-projection layer keys and only `torch.float32` at each.
- It requires at least three repeats per registered cell and derives epsilon from the maximum cell repeat spread and clean-to-clean replay error. The cross-method diagnostic is kept out of epsilon.
- The reader rejects unallowed backend metadata in the calibration artifact. The CLI rejects `--dtype` contradictions and limits `fp32-successor` to `calibrate` and `pilot`.
- The legacy branch still writes three BF16 repeats and one FP32 repeat; the metadata refactor retains the legacy implementation hash.

## Review evidence

Requested diff and commit list inspected. Focused reproduction: `repro_pilot_backend_bypass.py` (run with `/home/elisepc/neuron-probe/.venv/bin/python`). It uses a CPU tiny model and a fixture for raw calibration values, mocks the fixture's retained cross-method diagnostic and pilot stages, and exercises the real calibration reader and `run_pilot()` boundary. I did not run the full test suite or use a GPU/live service. The worktree did not contain the referenced `ALDEN-23.md` at its root; I reviewed the protocol file that is present.
