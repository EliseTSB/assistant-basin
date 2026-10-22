## SUMMARY

- Added a bounded instance-shadow check for the pinned Qwen3.5 RMSNorm `_norm` helper and the declared dispatch slots. The helper probe now refuses before pilot signs.
- Replaced direct Part 43 forward-hook registration with one phase-scoped registry that checks target, handle, callback identity, declared purpose/kind, and order; it checks local and PyTorch global forward hooks before scored work and restores the empty baseline on normal and exceptional exits.
- Added the exact Alden #27 scope property to the FP32 protocol and calibration/pilot records, and narrowed wording that could sound like general integrity attestation.
- The full offline suite passes: **159 passed**. A real tiny-config Transformers Qwen3.5 graph passes the helper and hook controls.

## FINDINGS

### What shipped

`part43.py` now checks the instance `_norm` attribute against the pinned `Qwen3_5RMSNorm._norm` function captured from the fingerprint-checked class. Declared instance dispatch slots must match the captured Qwen callables. The class graph now names the two Transformers 5.20 module classes observed in the supported Qwen graph (`GradientCheckpointingLayer` and `SiLUActivation`) as explicit loader/module allowances; their class and relevant call-slot identities are captured before model loading and checked against the live graph. There are no forward-hook allowances for the supported loader, recorded explicitly as an empty tuple with a comment.

All forward and forward-pre registrations in `part43.py` now go through the scoped registry. Each Part 43 hook has a fixed purpose name paired with its permitted hook kind. The registry tracks model-module target identity, handle ID, callback object identity, registration purpose and order. It checks the full model module graph and local forward/pre-hook maps, plus PyTorch’s global forward, forward-pre and associated metadata maps. A clean baseline is checked at FP32 pilot pre-stage and immediately around scored-forward/intervention scopes. Unknown or changed hooks refuse. Registered hooks are removed and the declared empty baseline is verified on normal and exceptional exits.

Paired controls cover the supplied `_norm` shadow, local forward and forward-pre hooks, global forward and forward-pre hooks, a numeric down-projection clamp/capture effect, exceptional-forward cleanup, callback/handle/target/purpose/order drift, and an actual one-layer `Qwen3_5ForCausalLM` built from the installed pinned Transformers classes. The tiny Qwen positive control runs its forward with two ordered registered captures on the same target and verifies both callbacks observed the input.

Existing backend/precision controls and probe files remain in place; this change adds controls without replacing their historical evidence.

The protocol and FP32 calibration/pilot records carry this fixed field verbatim:

> For the reviewed, fresh, reference-only execution configuration, these checks validate the declared backend/precision path and detect the enumerated unsupported substitutions or drift exercised by the tests. They are same-process checks, not a security/isolation boundary or proof against code that can replace the checks and their evidence.

### Scratch-copy sabotage audit

Each effective sabotage below was applied to a `/tmp` scratch copy and made its named test fail:

- Disabled the `_norm` instance-helper check: the `_norm` probe regression failed.
- Disabled local hook-map validation: both unregistered forward and forward-pre controls failed. Disabled global hook-map validation: both global forward and global forward-pre controls failed.
- Disabled the actual clamp mutation branch: the numeric clamp/capture assertion failed. The first clamp sabotage attempt changed only target selection and stayed green; I corrected it to disable the actual mutation branch, which turned the test red.
- Skipped handle removal on exceptional scope exit: the exceptional cleanup control failed.
- Removed the explicit `SiLUActivation` module allowance: the real tiny-Qwen positive control failed.
- Bypassed the declared-purpose check: the purpose control failed. Disabled callback-identity comparison, missing-handle detection, changed-handle-ID detection, model-target membership, or registration-order checking: the corresponding callback, handle, target, or positive-control test failed.
- Disabled the fixed scope-statement reader check: the changed-statement refusal test failed.

The first target-membership sabotage command selected zero tests (pytest exit 5); the corrected selector ran the test and it failed as expected. No effective sabotage stayed green.

### Wording changes

- Reworded the `part43.py` module docstring from saying every result records “its code and input hashes” to saying results record declared source/input hashes as provenance, not attestation of every executed object or accuracy.
- Narrowed the first summary bullet in `REPORT-43-fp32-backend-fold1.md` to selected code and bindings and stated that it is not an integrity claim for all code or numerical accuracy.
- Updated the protocol’s status text to describe its historical registration-time state, added the dated Alden #27 scope section, and retained the distinction between stability/hook fidelity and systematic accuracy.
- Added the fixed `execution_scope_statement` field to FP32 raw calibration, derived calibration summary, successful/HOLD pilot records, and HOLD records. Existing fingerprint field names remain scoped evidence labels; none is described as proving all executed code.

### Verification

- `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` — **159 passed in 101.64s**.
- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile part43.py test_part43.py` — passed.
- `git diff --check` — passed. Parent-control mailbox reads were empty; no plan changes or dispositions were needed.

## BLOCKERS

No implementation blocker. No commit was created, per the worker instruction, and I could not send the requested peer message because this run has no authorized external messaging channel. The real pinned Qwen module graph was verified with a tiny CPU config; pretrained checkpoint compatibility, GPU timing, and a GPU pilot remain unverified.
