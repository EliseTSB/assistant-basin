# Build: Part 43 FP32 successor (branch part43-fp32, worktree /home/elisepc/green/worktrees/ab-43-fp32)

Size, multiplied out: code and offline tests only. No model download, no GPU, no network, no rental.

## Read first
- `review/part43/FP32-SUCCESSOR-PROTOCOL.md` (the registered protocol; implement exactly this).
- `review/part43/ALDEN-23.md` (the reviewer's accepted contract, five numbered points).
- `review/part43/CALIBRATION-RAW-ONLY-DESIGN.md`, `review/part43/ALDEN-17.md` (the raw-only calibration contract this
  extends; its refusal and boundary rules all stay).
- `part43.py`: `run_calibration_runtime` (two legs today: bf16 x3 repeats, then fp32 x1 at ~1420), `_load_runtime`,
  the runtime metadata (~line 110, records `cuda_matmul_allow_tf32` and `cudnn_allow_tf32`), `_read_calibration_artifact`,
  `calibration_gate`, `main` (the `calibrate`/`pilot` dispatch). `test_part43.py`.

## Build
1. An FP32 decision protocol, selected explicitly (e.g. `--protocol fp32-successor`), bound in the registration, the raw
   schema, the reader and the recorded execution metadata. The pilot then loads, intervenes, traces receivers and
   scores in fp32 throughout. A label without fp32 execution must be impossible: the reader checks the recorded actual
   dtypes (weights, down_proj inputs, logits) are float32.
2. TF32 off for both `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` (and
   `torch.set_float32_matmul_precision("highest")`) before any fp32 work; record the effective values; the reader
   refuses an fp32-protocol artifact where either recorded flag is true.
3. FP32 calibration: >= 3 fp32 repeats of every registered cell (same cells and full answers as now), clean-to-clean
   replay, and the L7-zero/L11-restored rescue control, through the same raw-only writer -> disk readback -> derived
   report path. epsilon = max over cells of (fp32 repeat spread of the primary score, clean-to-clean replay error),
   compared exactly to 0.05. No bf16 leg is run or synthesised under this protocol.
4. The old bf16 protocol keeps working unchanged (its tests stay green) so 845964c's receipts stay readable.

## Offline controls (tests, CPU, stubbed model calls like the existing ones)
- wrong or mixed execution dtype under the fp32 protocol refuses; a recorded TF32 flag true refuses;
- a bf16-protocol artifact presented to the fp32 reader refuses, and the reverse;
- a genuine fp32 registration passes;
- fp32 repeat spread alone at 0.06 holds the gate; replay error alone at 0.06 holds the gate; 0.05 exactly passes;
- derived, duplicate-key, malformed and internal-error cases keep their current distinctions;
- no production stage runs before the reader's verdict (the pilot reads the file back, as now).
Run each new test against a sabotaged copy in a detached worktree (`git worktree add --detach`), not a plain copy:
a plain scratch copy reds 34 tests at baseline because the code records git provenance.

## Verify and report
`/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` green. Commit on part43-fp32 with message
ending `Co-Authored-By: GPT-6 Luna <noreply@openai.com>`. Write `REPORT-43-fp32.md`: SUMMARY, what changed by function,
each control and the sabotage that reds it, anything in the protocol you could not implement as written (say so; do not
reinterpret), BLOCKERS.
