# Part 35 sink diagnostic report

## What changed

- `part35.py:351` and `part35.py:499` retain the original all-position RMS and mean vectors while adding content-token RMS, vector means, per-layer position-0 norms, content median/p90 norms, masked positions, and position-0 vectors. The mask excludes position 0 and special/chat-template tokens. `capture_version="35.2-nosink"` and the new arrays are written to each new NPZ.
- `part35.py:560` classifies token strings by their offset within the rendered prompt. `part35.py:776`–`part35.py:934` validates cache versions/keys and recaptures stale neutral and self-direction cache rows by default. A stale read reports the missing key and a role-specific `screen --recapture-roles ...` command; it never falls back to legacy values.
- Only neutral and self-direction rows consume the new cache fields: 36 neutral samples feed screen residual norms and dose profiles; 64 self-direction samples (32 AI and 32 human) feed the old and masked fits and sink diagnostics. Other cached screen roles remain usable without recapture.
- `part35.py:1057` keeps legacy self-direction fields and adds `_nosink` fit/AUC fields. Stage 1 and stage 2 use the masked direction. Sink diagnostics include each self-direction record's coefficient and its correlation with the AI/human label for both fits.
- `part35.py:1676` and `part35.py:2001` retain both RMS values and the median/p90 content norm profile. The rho clamp level uses the median content norm; the p90 and both RMS fields are recorded alongside it.
- `part35.py:608`–`part35.py:724` adds `peak-positions`, which captures only requested units and writes `peak-positions.jsonl` plus `peak-positions.summary.json`. It covers each concept's top-five screen rows on that concept's passages and all units with identical neutral maxima on neutral passages. Rows include argmax position/token, position-0 value, and masked positions.
- `test_part35.py:375` adds the fake-model sink, cache-version, recapture, self-direction, dose-profile, and peak-token regressions.

## GPU commands

Run from the project root on the GPU host. These use the existing `results/part35` output root and the cached model directory; the first command refreshes stale neutral/self-direction rows and rescales the screen. The second captures the requested peak positions.

```bash
OUT=results/part35
CACHE=/mnt/e/neuron-probe/hf/hub
MODEL=Qwen/Qwen3-1.7B

~/neuron-probe/.venv/bin/python part35.py screen \
  --model "$MODEL" --cache-dir "$CACHE" --device xpu --dtype bf16 \
  --out "$OUT" --recapture-roles neutral,self_direction

~/neuron-probe/.venv/bin/python part35.py peak-positions \
  --model "$MODEL" --cache-dir "$CACHE" --device xpu --dtype bf16 \
  --out "$OUT" --screen "$OUT/qwen-qwen3-1-7b/screen.jsonl"
```

## Verification

`~/neuron-probe/.venv/bin/python -m py_compile part35.py test_part35.py` succeeded. The required full test command produced:

```text
.....................                                                    [100%]
21 passed in 4.12s
```

Both `screen --help` and `peak-positions --help` also exited successfully.

For the required mutation check, I temporarily changed the nosink residual-vector path to average all positions. The targeted residual capture test failed (`1 failed`): actual nosink mean `[334.33334, 2.0]`, expected `[1.5, 3.0]`. I restored the content-only mean and reran the full suite; all 21 tests passed.

Parent control #1 (Prime review #71) changed the plan. I applied the template-token mask, position mask recording, median/p90 profile and median dose, and sink-coupling diagnostics/tests; disposition recorded as `considered`. No parent messages were declined or deferred.

No model weights or GPU were run in this worker, as the brief instructed. The real-data GPU capture and peak-position output remain to be produced by the commands above.

SUMMARY: Part 35 now preserves legacy measurements while adding a versioned content-token residual path, median-based rho doses, sink diagnostics, and requested-unit peak-position capture.
FINDINGS: All 21 tests pass; the deliberate all-position mutation made the nosink capture regression fail and was restored.
BLOCKERS: No code blocker; a real GPU/model run was not performed in this worker by instruction.
