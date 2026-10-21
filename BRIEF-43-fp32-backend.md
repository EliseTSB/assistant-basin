# Build: Part 43 FP32 reference-backend primitive (branch part43-fp32, worktree /home/elisepc/green/worktrees/ab-43-fp32)

Size: code and offline tests only. No model download, no GPU, no network.

## Read first
- `review/part43/ALDEN-25.md` IN FULL: it is the contract. Implement its four numbered points and its paired controls
  exactly; where it and anything else disagree, it wins.
- `review/part43/REVIEW-fp32-r2-haiku.md` (F1) and `REVIEW-fp32-r2-luna.md`: the two measured failures this replaces.
- `review/part43/FP32-SUCCESSOR-PROTOCOL.md`, `review/part43/ALDEN-23.md`.
- The installed transformers 5.20.0 source in `/home/elisepc/neuron-probe/.venv` (integrations/hub_kernels.py and
  models/qwen3_5/modeling_qwen3_5.py): verify the real dispatch slots there; do not assume a layout.

## Build
1. Remove the closure-based `uses_torch_reference` inference. Replace it with an explicit reference binding: the known
   torch-reference callables from the pinned transformers source, held as the primitive, checked against the actual
   dispatch slots the Qwen3.5 forward uses (module globals and any instance/forward overrides), with the installed
   source identity (file hash) bound. Any unknown or substituted callable refuses, whatever its name, closure or module.
2. Preparation, before importing transformers/model code: USE_HUB_KERNELS=0; record the effective import-time hub
   state (transformers caches it; a later env change is not a reset); record distribution metadata and import
   resolution for causal_conv1d and fla. Genuine absence (not installed) is the supported fallback; an installed but
   broken import is a preparation refusal, not absence.
3. Inside `run_pilot`, after the pilot model is loaded and immediately before `_sequences`/signs/the first stage, for
   CLI and direct callers alike: compare the current backend and precision state and the actual reference binding to
   the calibration-bound expected state; refuse on missing, stale or changed evidence, and refuse if either state is
   forbidden even when they match. Keep timing, object addresses and growing counters out of the compared fingerprint.

## Offline controls (each must red under a targeted sabotage in a detached worktree)
Alden's list: fresh reference-only setup reaches the first stage; a plain foreign module-level callable refuses with
an unchanged package/env fingerprint; a swapped binding after calibration refuses; an installed-but-broken optional
import does not count as absence; hub enabled at import then env=0 refuses; a current USE_HUB_KERNELS/TF32/precision
mismatch refuses before signs; an unchanged valid binding and fingerprint pass; the no-optional-package fallback passes.

## Verify and report
`/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` green (118+ now). Do not commit. Write
`REPORT-43-fp32-backend.md`: SUMMARY, contract point -> code -> test -> sabotage table, the dispatch slots you found in
the installed source, anything in the contract you could not implement as written, BLOCKERS.
