# Part 184: Parts 18's and 18b's alien-neuron results hold on the attention kernel that matches the CPU

Parts 18 and 18b generated their answers on the A770 in bf16 with the default fused SDPA attention, which Part 178
found leaving the CPU float32 path after 64 keys. Here the same three runs were repeated with the eager kernel (Part
183's kernel check: eager follows the CPU reference at batch 1, 2 and 8 on Qwen3.5-4B). Same model revision, same
neuron (L22 N2742), same clamp scale and the same 20 control units from Part 18's `reads.json`, same prompts, seeds and
96 new tokens.

| Cell | Eager (this part) | SDPA (published) |
| --- | --- | --- |
| Part 18: set to +2x, "Who made you?" samples alien | 8 of 8 | 8 of 8 |
| Part 18: unclamped samples alien | 0 of 8 | 0 of 8 |
| Part 18: 20 controls at +2x, 8 samples each, alien | 0 of 160 | 0 of 160 |
| Part 18: controls' greedy answers at +2x naming Qwen | 20 of 20 (0 alien) | 20 of 20 (0 alien) |
| Part 18: set to -2x, samples still naming Qwen | 1 of 8 | 2 of 8 |
| Part 18b: clamp during generation only, samples alien | 8 of 8 | 7 of 8 |
| Part 18b: clamp on the prompt only, samples alien | 0 of 8 | 0 of 8 |

Both parts' published results stand. Every count moved by at most one sample.

## Predictions

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 184.1 | A: the clamped (+2) samples are alien in at least 7 of 8. Fails at 6 or fewer. (high) | 8 of 8 | held |
| 184.2 | A: the unclamped samples are alien in 0 of 8, and the 160 control samples at +2 are alien in at most 2. Fails at 1 or more unclamped, or 3 or more control samples. (high) | 0 of 8; 0 of 160 | held |
| 184.3 | A: the controls' greedy answers at +2 name Qwen in at least 18 of 20 and are alien in 0 of 20. Fails otherwise. (high) | 20 of 20 name Qwen, 0 alien | held |
| 184.4 | A: at -2, the samples name Qwen in at most 4 of 8 (published 2 of 8). Fails at 5 or more. (medium) | 1 of 8 | held |
| 184.5 | B: the generation-only clamp gives alien in at least 5 of 8 samples (published 7 of 8). Fails at 4 or fewer. (medium) | 8 of 8 | held |
| 184.6 | C: the prompt-only clamp gives alien in at most 1 of 8 samples (published 0 of 8). Fails at 2 or more. (medium) | 0 of 8 | held |

Gate: every output file records `attn_implementation` eager (in its `load` record). Passed. The pilot (C) ran first;
its kernel was eager and none of its answers was empty, so A and B were started.

## The answers themselves differ more than the counts

The counts are the same, but the text often is not. Of run A's 48 greedy target answers (6 prompts, unclamped and 7
clamp levels), 8 are identical to the stored SDPA answer and 40 differ. Measured in key positions (prompt tokens plus
answer tokens), 27 of the 40 first differ at keys 65 to 70 (26 at 65 to 68, one at 70), where the SDPA fault begins. The other 13 differ earlier, at
keys 22 to 59, where the SDPA kernel agrees with the CPU reference on Part 183's eight check prompts. Examples:
unclamped "Tell me about yourself", key 34: "developed by Alibaba Cloud's Tong..." (eager) against "developed by Tongyi
Lab..." (SDPA); +4x "Who made you?", key 42: eager ends the answer after "XEET." where SDPA went on. B's and C's target
answers show the same pattern (B: 2 identical, 10 differ, 4 of them before key 65; C: 4 identical, 8 differ, 2 before
key 65). This part does not measure why. The likely reason is a near-tie: two kernels in bf16 give slightly different
logits everywhere, so where the top two tokens are almost equal, either kernel can pick the other one. Part 183's own
eager check departed from the CPU at a near-tie at answer position 2. The positions are in
`greedy-text-compare.json`.

This qualifies the Arc audit's decision-position measurement (Journal `notes/reports/arc-sdpa-audit-2026-11-03/
decision-positions/`), which treated any decision fixed by key 64 as unaffected by the kernel. A decision fixed by key 64
can still differ at a near-tie. Here no count changed by more than one, but the rule "fixed by key 64 means unchanged"
is too strong.

## Method

`neuron_report.py steer` (the current version, which loads eager by default and records the kernel), with Part 18's
`results/part18/chat/reads.json` for the clamp scale and the control units:

- A, Part 18: the defaults Part 18 used (target ks -4, -2, -1, 0, 1, 2, 4 on six prompts, greedy; 8 samples at
  temperature 0.7 and top_p 0.95, seed 1, on "Who made you?", unclamped and at -2 and +2; 20 controls greedy at -4, -2,
  +2, +4 and sampled 8 each at +2). 23 min 24 s.
- B, Part 18b generation-only: `--phase gen --ks 2 --control-ks 2 --sample-ks 2`. 15 min 57 s.
- C, Part 18b prompt-only: `--phase prompt --ks 2 --control-ks 2 --sample-ks 2`. 9 min 2 s.

Scored by the tool's own `score_text` (the patterns Parts 18 and 18b published). `part184/compare184.py` reads the
eager and stored summaries and writes `compare.json`.

## Limits

One model, one GPU, one neuron and its 20 controls, as the originals. Sampled answers move with any numeric change, so
counts are compared, not answers. The eager kernel was checked against the CPU on eight prompts at three batch shapes,
not on these prompts. English only.

## Files

- `part18-eager/steer.json`, `part18b-gen/steer.json`, `part18b-prompt/steer.json`: the three eager runs.
- `compare.json`: every eager count beside the stored SDPA count, and the verdicts.
- `greedy-text-compare.json`: identical and differing greedy target answers per run, with the key position where each
  difference starts.
- `launch.jsonl`: the two launches (C, then A and B). Code: `part184/compare184.py`. Design: `PART184-DESIGN.md`.

Cold check: pending.
