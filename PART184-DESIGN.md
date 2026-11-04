# Part 184: do Parts 18's and 18b's alien-neuron results hold on the attention kernel that matches the CPU?

Procedure: v1
Registered 2026-11-04 04:00 CT by Claude (Opus 5.5), before any data. Asked by the SDPA audit (Journal ops card
`xpu-batch1-qwen35-audit-2026-11-02`): Part 178 found the A770's fused bf16 SDPA attention leaving the CPU float32 path
past 64 keys, and the audit's decision-position measurement (Journal
`notes/reports/arc-sdpa-audit-2026-11-03/decision-positions/DECISIONS.md`) found that Parts 18 and 18b's positive cells
are decided early (the alien word appears by key 27) while their absence cells (unclamped answers, the 160 control
samples, the prompt-only clamp) run to keys 69 to 112. Elise's standing direction for the night (2026-11-03 23:53 CT,
"Keep the research going"). The commit that adds this file is the registration.

## Question

Parts 18 and 18b generated their answers on the A770 in bf16 with the default SDPA kernel. Do their published results
stand when the same runs use the eager kernel, which follows the CPU float32 reference on Qwen3.5-4B (Part 183's kernel
check)? Part 18: setting L22 N2742 to 2 times its corpus maximum makes 8 of 8 sampled "Who made you?" answers alien,
against 0 of 8 unclamped, 0 of 160 samples for 20 random units set the same way, and every control's greedy answer
still naming Qwen; at -2 times, 2 of 8 samples still name Qwen. Part 18b: the same clamp applied during generation
only gives 7 of 8 alien; applied to the prompt only, 0 of 8.

## Prior art and existing data

- Closest published work: none needed; the question is about our own runs. The kernel finding is ours (Part 178;
  `results/kernel-reference/README.md`; Part 183).
- Our own data: the stored SDPA runs (`results/part18/chat/steer.json`, `results/part18b/phase-gen/steer.json`,
  `results/part18b/phase-prompt/steer.json`) are the SDPA side of every comparison. Part 183's kernel check passes eager
  at batch 1, 2 and 8 on Qwen3.5-4B (`results/part183/kernel-check.json`); this part's sampled answers are generated 8 at
  a time from one prompt (no padding), its greedy answers one at a time. Part 183 found Parts 25 and 66 unchanged on
  eager. Seen before writing: the stored summaries above and the decision-position measurement; no eager answer to any
  Part 18 or 18b prompt exists.

## Method

Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, on the A770 (xpu), attention kernel eager:
`neuron_report.py`'s default since 2026-11-03, recorded in every output file. The clamp scale comes from Part 18's own
`results/part18/chat/reads.json` (the same maxima and the same 20 control units, seed 0). Nothing else changes: the
current `neuron_report.py steer` with each original run's arguments.

- A, Part 18 rerun: `neuron_report.py steer` for L22 N2742 with the defaults Part 18 used (target ks -4, -2, -1, 0, 1,
  2, 4 on the six built-in prompts, greedy; 8 samples at temperature 0.7 and top_p 0.95, seed 1, on "Who made you?",
  unclamped and at -2 and +2; the 20 controls greedy at -4, -2, +2, +4 and sampled 8 each at +2; 96 new tokens), out
  `results/part184/part18-eager`.
- B, Part 18b generation-only rerun: the same with `--phase gen --ks 2 --control-ks 2 --sample-ks 2`, out
  `results/part184/part18b-gen`.
- C, Part 18b prompt-only rerun: the same with `--phase prompt --ks 2 --control-ks 2 --sample-ks 2`, out
  `results/part184/part18b-prompt`.

Scoring is the tool's own `score_text` (the alien pattern and the Qwen-name pattern, read up to the first role-turn
marker), the scorer Parts 18 and 18b published. Each eager count is set beside the stored SDPA count for the same cell.

Kernel check: results/part183/kernel-check.json

## Controls

- The kernel check passes for eager at batch 1, 2 and 8 (Part 183), and its SDPA control fails on this model, so the
  check can see the fault.
- The random controls are the originals' own 20 units, at the originals' scale, so every comparison is like for like.
- Not used: new prompts, doses or controls, because this part reruns registered runs and changes only the kernel. Not
  used: a second reader, because the scorer is a fixed pattern.

Gate: every output file records `attn_implementation` eager. If any does not, its numbers are not read.

## Predictions

- 184.1 (A): the clamped (+2) samples are alien in at least 7 of 8. Fails at 6 or fewer. (high)
- 184.2 (A): the unclamped samples are alien in 0 of 8, and the 160 control samples at +2 are alien in at most 2. Fails
  at 1 or more unclamped, or 3 or more control samples. (high)
- 184.3 (A): the controls' greedy answers at +2 name Qwen in at least 18 of 20 and are alien in 0 of 20. Fails
  otherwise. (high)
- 184.4 (A): at -2, the samples name Qwen in at most 4 of 8 (published 2 of 8). Fails at 5 or more. (medium)
- 184.5 (B): the generation-only clamp gives alien in at least 5 of 8 samples (published 7 of 8). Fails at 4 or fewer.
  (medium)
- 184.6 (C): the prompt-only clamp gives alien in at most 1 of 8 samples (published 0 of 8). Fails at 2 or more.
  (medium)

## Reading rules

- Each prediction reads the tool's own summary in the run's `steer.json` (`samples`, `controls_greedy`,
  `controls_sampled`), with the alien and Qwen counts from `score_text`; the same fields Parts 18 and 18b published.
- If 184.1, 184.2, 184.5 and 184.6 hold, the published alien-neuron claims of Parts 18 and 18b stand on the eager
  kernel, and their front-page text gains one line saying so with this part's numbers. A failure is reported beside the
  published number, and the front page is corrected with both numbers and the kernel named.
- Sampled answers are not expected to match SDPA's text one for one: a numeric change can send a sample down another
  path at any position. Only counts are compared.

## Limits

One model, one GPU, one neuron and its 20 controls, as the originals; the eager kernel is checked against the CPU on
eight prompts at three batch shapes, not on these prompts; sampled answers move with any numeric change, so counts are
compared, not answers; English only.

## Size

- Smallest effect worth knowing: a change of 2 or more in any published count of 8, or 3 or more in the 160 control
  samples. The n are the originals'.
- Total: 192 greedy answers and 536 sampled answers, about 45 minutes on the A770. A: 128 greedy answers and 23 sampled
  batches of 8; B and C: 32 greedy answers (12 target, 20 control) and 22 sampled batches each; 96 new tokens. The estimate scales Part 183's eager batch-1 greedy rate (about 4 s per 48-token answer)
  to 96 tokens.
- Pilot: C runs first and is the pilot (the smallest run). Stopping rule: A and B do not start if C's file records a
  kernel other than eager, or if any of C's answers is empty.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 183's B, 132 greedy answers in 524 s (48 new tokens each, including the model load), eager, batch 1.
- Levers: batch size, declined (the originals' shapes are the thing under test); fewer samples or controls, declined
  (the n are the originals'); max_new_tokens, declined (96, the originals'; the absence cells need the full length);
  caching, n/a.
