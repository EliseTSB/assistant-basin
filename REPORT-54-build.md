# Part 54 build report

## SUMMARY

Implemented the Part 54 source-model clamp, generic candidate searches, ladder driver, and backward-compatible blind reader options. Added CPU contract tests for the new paths and documented the rented 3090 procedure. Existing Part 26 tests passed after the implementation. No GPU instance was rented and no experiment outputs were generated.

## FINDINGS

- Added four public, pinned Hub revisions to the model table. OLMo-2-7B is configured for bf16; Qwen3.5-4B, Qwen3-4B, Phi-4-mini, and Granite 3.3 2B use fp32 defaults. Phi-4-mini enables its remote-code loader path.
- The Qwen3.5 source clamp reads the pinned Part 33 record, recalculates the source unit and same-layer control maxima on its 30 selected alien passages, and reuses the Part 26 condition-seed and generation policy. The ladder writes a manifest and 21 registered question/dose steps, with controls at -2 and +1.5.
- New-model battery records include exact normalized logit-lens top-30 write evidence. A new twin must pass fire (`beats_null`), write (`alien` or `aliens` in top 30), and all three profile checks. Existing Part 26 search calculations and default model list remain compatible.
- `blind_origin.py submit --folder ...` accepts an explicit folder and reads model-suffixed `clamp-*.json` names; the old default folder and status/collect commands remain available.
- Verification completed: `py_compile` and `git diff --check` passed; `test_part54.py` passed (6 tests and 4 subtests); existing `test_part26.py` passed (21 passed, 0 failed). A temporary sabotaged copy changed the 480-generation assertion to 481 and failed that test while the other five tests passed.
- The current per-dose baseline policy makes one full 8-sample ladder 480 generations. Thus 54A plus the required Smol comparator ladder and two new twin ladders is 1,920 generations, above the design's rough 1,200 estimate. Four new twins plus the comparator would be 2,880.

## BLOCKERS

None for the build. Runtime and memory estimates in the runbook are planning estimates; no 3090 run was performed. CPU contract tests need a local Qwen3.5 config/tokenizer snapshot, configurable through `QWEN35_SNAPSHOT`.

