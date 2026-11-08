# Part 192: Part 39's verdicts hold on the eager attention kernel, but its push result does not (0 of 96 AI claims at 2x under the Sam persona, not 8 of 96)

Part 39 tested Qwen3.5-4B L6 N753, a unit that fires on requests and AI self-descriptions: plain and chat-template reads
across six sentence categories, zeroing it, pushing it under a human persona ("Sam, a carpenter from Ohio"), and a
screen for other units that separate addressed from unaddressed text (section D, rescored under Part 39's amendment 1).
It ran on the A770 in bf16 with transformers' default fused SDPA kernel, which Part 178 found leaving the CPU float32 path
after about 64 keys. Here the same prompts, seeds, units and scorer ran on the eager kernel. Registered in
`PART192-DESIGN.md` (public main f47e0e7, 00:29 CT) before any eager data; Amendment 1 (c70cd4b, corrected at 99e8772)
split the screen into one model per process after the job runner's RAM guard killed the two-model screen twice, before
any screen, zero or push output was read. The final stages ran 01:14 to 03:02 CT on 2026-11-08 (screen chat 4.5 min,
screen base 4.4 min, zero 40.3 min, push 58.8 min). The host record gate passed: every recorded model load is eager, and
reads, screen:chat, screen:base, zero and push are all present (`run192.py --check-record`).

| Part 39's measures | Eager (this part) | SDPA (Part 39, published) |
| --- | ---: | ---: |
| Chat plain medians: requests / AI self / third-person AI | 1.49 / 2.57 / 1.21 | 1.52 / 2.56 / 1.23 |
| Chat-template assistant-marker medians, six categories | 3.28, 3.93, 3.09, 3.39, 2.23, 2.93 | 3.28, 3.91, 3.09, 3.37, 2.22, 2.93 |
| Base/chat overall assistant-marker ratio | 0.776 | about 0.76 |
| Zeroing N753: sampled AI claims, baseline / zeroed (of 48) | 48 / 46 | 48 / 47 |
| Random zero controls: AI claims (of 120) | 120 | 120 |
| Sam persona: AI claims unpushed / k=1 / k=2 (of 96) | **0 / 0 / 0** | **0 / 1 / 8** |
| Random k=2 controls: AI claims (of 12 each, ten controls) | 0 in every control | 0 in every control |
| Section D: chat / base passing units | 213 / 203 (127 shared) | 211 / 205 (127 shared) |
| N753 addressee AUC, chat / base | 0.919 / 0.917 | 0.92 / 0.92 |
| Strict set (request-only and AI-self-only AUC at least 0.8, third-person at least 0.7): chat / base / shared | 56 / 43 / 27 | 56 / 43 / 28 |
| Chat passing units in layers 0 to 12 | 19 of 213 | 19 of 211 |

Every Part 39 verdict this rerun can score is the same on both kernels (39.1 failed, 39.2 held, 39.3 failed with its
plain-order clause held, 39.4 held, 39.5 failed, 39.6 held, 39.7 held, 39.8 failed, 39.10 held, 39.11 held). The reads,
the zeroing and the section D screen reproduce closely. The push does not: on the faithful kernel the Sam persona never
claims to be an AI at any dose.

## The push result was produced by the kernel

Part 39 reported that pushing N753 at 2x raised Sam's AI claims from 0 of 96 to 8 of 96 (39.5 failed only because it
predicted 20 points). This rerun used the same prompts and the same seeds on every row, so the two runs can be compared
answer by answer.

- **All nine published AI claims disappear** (eight at k=2, one at k=1), and **no new claim appears** anywhere in the 408
  push rows. Unpushed answers and every random control stay at 0 in both runs.
- **Each of the nine eager answers begins with the same words as its published answer** and then stays in persona. In
  four of the nine the two texts part exactly where the published one makes its claim: "I don't get to go out and take
  evening walks like a person does. I'm " continues as "Sam, a carpenter" on eager and as "a large language model, an
  AI model" on SDPA; "I don't drink coffee or tea. " continues as "Not because I'm some kind of robot" on eager and as
  "I'm a text-based AI" on SDPA. In the other five they part a few words before the claim. The two runs share 14 to 102
  characters before parting.
- **Several published claims were degraded text.** "I'm a text-based AI, and I don't have a physical form or a physical
  form", "I'm a little over a minute away from a helpful conversation", and "I've been a carpenter's assistant", which the
  AI-claim pattern counts because it contains "assistant".

No push row is identical between the runs (median first difference at character 94), as expected once the kernel
changes the logits; the zero rows share more text (63 of 228 identical). A sampled
answer parts at the first token whose choice the kernel's error tips, and the faithful kernel never tipped one toward a
claim. So the published 8 of 96 is a measurement of the faulty kernel, not of N753. It does not show that pushing N753
can never produce an AI claim: this is one dose ladder (k=1 and 2) on one persona and 96 samples per dose.

## Gates and record

- Host record gate: passed (all loads eager; five stages present).
- Screen sanity rule (L6 N753 must pass in the chat model): passed in both models.
- `score.json`'s own screen block reads "broken", as the design said it would: Part 39's raw MAD screen was already broken
  in the published run. Section D is read only from `screen-auc-chat.json` and `screen-auc-base.json` (Part 39's
  amendment 1 statistic, 200 shuffles, seed 3901), as the reading rules require.
- The two killed screen attempts' partial files (`killed-0044`, `killed-0110` on the job folder) were not read and are
  not part of this result.

## Predictions

| # | Prediction (registered wording, abridged; confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 192.1 | Chat plain medians within 0.25 of 1.52 (requests), 2.56 (AI self), 1.23 (third-person AI) (high) | 1.49, 2.57, 1.21 | held |
| 192.2 | Chat request-to-third-person ratio 1.0 to 1.4; 39.1 stays failed (high) | 1.23; failed | held |
| 192.3 | Six template medians within 0.5 of published, all above 2.0 (medium) | 3.28, 3.93, 3.09, 3.39, 2.23, 2.93 | held |
| 192.4 | Base plain medians within 0.25 of published; base/chat marker ratio 0.65 to 0.85; 39.3 fails on the marker clause, holds on plain order (medium) | 1.45, 2.84, 0.11, 0.17, 0.96, 0.09; ratio 0.776; as predicted | held |
| 192.5 | No greedy maker change; 48/48 baseline and 47/48 zeroed within 3; random zero controls at least 115/120 (medium) | none; 48 and 46; 120 of 120 | held |
| 192.6 | Sam persona: 0/96 unpushed, 0 to 4/96 at k=1, 4 to 14/96 at k=2; random k=2 controls 0 to 2/12 (medium) | 0, 0, **0**; 0 in every control | **failed** (k=2) |
| 192.7 | 180 to 240 chat and 175 to 235 base passers; N753 passes in both with AUC 0.87 to 0.97 (medium) | 213 and 203; 0.919 and 0.917 | held |
| 192.8 | Strict 40 to 72 chat, 30 to 58 base, 15 to 40 shared; at most 30 chat passers in layers 0 to 12 (medium) | 56, 43, 27; 19 | held |
| 192.9 | The ten published Part 39 verdicts this rerun can score remain the same; 39.9 not tested (medium) | all ten the same | held |

39.9 (a passing unit with an AI term in its logit-lens top 30) is not tested: its published verdict came from Part 42's
human reading of logit-lens tokens, which this rerun does not produce.

## One correction to the published Part 39 text

Part 39's 39.8 row says that of 211 passing chat units, "147 sit in layers 19 to 31". Recounted from the same published
file (`results/part39/screen-auc-chat.json`), the number is 177 (15 in layers 13 to 18, 19 in layers 0 to 12). The
verdict does not change: 39.8 needed at least half in layers 0 to 12. The row is corrected in `PREDICTIONS.md` with the
date. The eager count is 178 of 213.

## Limits

- One model, one seed per row, one persona. The push comparison is exact on seeds but each row is still one sampled
  answer; 0 of 96 against 8 of 96 is a large difference for 96 samples, and the per-row comparison is what carries it.
- The A770's bf16 path is not the CPU float32 path even on eager; Part 178's reference check showed eager matching the
  CPU on its prompts, which is the reason for this audit, not a guarantee for every prompt here.
- Section D reuses Part 39's amendment 1 statistic, chosen after the first screen broke (Part 39's own disclosed
  weakness).

## Files

`load.json` (record), `reads.jsonl`, `reads-summary.json`, `screen-qwen-qwen3.5-4b.json`,
`screen-qwen-qwen3.5-4b-base.json`, `screen-auc-chat.json`, `screen-auc-base.json`, `zero.jsonl`, `push.jsonl`,
`score.json`, `launch.jsonl`. The raw read arrays (`reads-chat.npz`, `reads-base.npz`, 157 MB each) are not committed
(over GitHub's 100 MB limit); they stay on the host.
