# Part 30 batch generation report

## SUMMARY

Added batched generation with `--gen-batch` (default 16) for control and generate, including per-index resume for both outputs.

## FINDINGS

- Batched prompts are left-padded and sorted by encoded prompt length. Rows in each finished batch are appended in index order.
- Per-row token counts and finish reasons trim padding after an early EOS. `wall_time` is the batch duration divided by batch size, recorded in each header's settings.
- The brief specifies greedy decoding for generation. The existing generate path sampled, so generate now uses greedy decoding as specified.
- CPU regressions cover six-prompt batch/single text equality, mixed completion lengths, control resume, and generate resume. `python -m pytest -q test_part30.py`: 13 passed.
- Running pytest from the whole worktree collects unrelated `climb/climb_test.py`, which parses required command-line arguments during collection. The focused Part 30 suite passes.

## BLOCKERS

None.

## STEPS

Use these control and generate lines in `steps.txt`; paths are unquoted for `shlex` parsing:

```text
part30.py --steps split,control --pairs E:\neuron-probe\data\s1k\s1k11-pairs.jsonl --device xpu --max-gpu-gb 14.5 --gen-batch 16
part30.py --steps generate --teacher base --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5 --gen-batch 16
part30.py --steps generate --teacher r1 --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5 --gen-batch 16
part30.py --steps generate --teacher gemini --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5 --gen-batch 16
part30.py --steps generate --teacher control --prompts data/part29-prompts.json --device xpu --max-gpu-gb 14.5 --gen-batch 16
```
