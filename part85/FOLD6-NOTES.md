# Part 85 fold 6 notes

Production-side changes only. No GPU run, checkpoint download, or commit.

## Changes

1. `_eos_ids` now returns only the sorted union of checkpoint generation-config EOS IDs and tokenizer EOS. It keeps IDs equal to UNK and excludes named special tokens. `audit85.py` loads the checkpoint generation config independently and checks the observer's effective EOS set and passed EOS set against that rule.
2. `observe85.py` records every forward keyword value, including tensor dtype, shape, and contents. The step-1 audit checks the registered inputs, exact keyword values, sequential record IDs for 200 neutral then 48 passage texts, one call and row per text, and the required neutral position count (128 in the registered run). It also verifies both passage means, both per-passage lists, counts, and ordinary SD.
3. `LOCAL-RUN.md` gates pilot reading on full replay and the negative control, with pilot/main result and calibration paths and the current replay CLI commands. The refs manifest now reads calibration from `R\main\instruct` and token IDs from `R`; the runbook adds `datasets` installation and explains how `CODE_COMMIT` and `commits.local.json` are produced.
4. `RateLimiter` is now a token bucket capped at one second of bytes. A simulated 100-second stall test proves excess credit is discarded.

## Verification

- `python -I -B part85/test_part85.py`: 44 tests, OK.
- `python -I -B part85/test_download85.py`: 10 tests, OK.
- `python -I -B part85/mutate85.py --jobs 2`: baseline 44 tests; 42/42 mutations killed, none survived or errored.
- `observe.py` SHA-256 remains `f6c1d7e16f4ccbb083e2a7ff8fa99d2f18e08be4504696e25f2c5b9821082613`.

## Not applied / limits

`replay85.py` and `test_replay85.py` were left to Claude's parallel rewrite as instructed. Its current interface covers pilot attempt 1 and the default main scope, but does not select pilot attempt 2 or the chat-only main scope. `LOCAL-RUN.md` directs the operator to stop before reading those outputs until replay supports them. Step-1 regression fixtures use 2 neutral rows of 4 positions plus 48 passages; no production model or full-size activation run was performed.
