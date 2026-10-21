<!-- cheap-review.mjs cheap-review-part43-fp32-r1-luna-r1-2026-10-21: /home/elisepc/green/worktrees/ab-43-fp32 79f13a12d5199f073fde96bb52adf593ca4804ea..HEAD, lane luna, df-run exit 0 -->
# Cold adversarial review: part43-fp32-r1

**Verdict: SHIP-WITH-FIXES**

## Findings

### MEDIUM — INTRODUCED — `part43.py:3494` (protocol gate at `part43.py:6051-6056`)

`run_pilot()` calls `calibration_gate()` without the selected protocol. With no `protocol_id`, the gate sets its expected protocol from the summary itself, so it does not bind the supplied verdict to `args.protocol`.

**MEASURED trigger:** I imported the real `part43.py` and ran `repro_run_pilot_protocol_bypass.py`, supplying `args.protocol='bf16-v2'`, a model marked `torch.bfloat16`, and a valid-shaped FP32-successor reader summary. The production-stage functions were stubbed only to record entry. Output:

```text
result_status: complete
calibration_protocol: part43-fp32-successor-v1
stages_entered: [('signs', 'torch.bfloat16', 'bf16-v2')]
```

This direct callable path can report a complete bf16 pilot under an FP32 artifact. The normal CLI path currently supplies the selected protocol at `part43.py:6238-6241` and loads through `_load_runtime`, so this reproduction is specifically the callable `run_pilot()` boundary. **Smallest fix:** pass `_selected_calibration_protocol(args)` into `calibration_gate()` at line 3494.

## What I checked and found sound

- FP32 raw artifacts bind protocol and decision dtype; the reader rebuilds registration, refuses cross-protocol artifacts, checks parameter/activation/down-projection/logit dtypes, and refuses recorded TF32-on values or non-`highest` matmul precision (`part43.py:5690-5850`).
- The FP32 writer runs only the FP32 leg and records at least three repeats for every registered cell; the reader derives epsilon from the maximum registered-cell spread and clean-to-clean replay error (`part43.py:1526-1543`, `5859-5908`).
- The main `pilot` CLI path reads the serialized artifact and checks the selected protocol before loading the pilot model or entering stages (`part43.py:6234-6266`).
- The legacy reader/writer path retains its two-leg schema and accepts the pinned legacy provenance snapshot. No GPU or live service was used; I did not run the test suite.
