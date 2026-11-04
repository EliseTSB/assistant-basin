# Part 184: Parts 18's and 18b's alien-neuron counts hold on the eager attention kernel

Parts 18 and 18b generated their answers on the A770 in bf16 with the default fused SDPA attention, which Part 178
found leaving the CPU float32 path after about 64 keys. Here the same three runs were repeated with the eager kernel.
Part 183's kernel check found eager matching the CPU reference except at near-ties: it departs on 1, 2 and 3 of 8
prompts at batch 1, 2 and 8, every time where the reference's top two tokens were within 0.101 of each other (batch 8
checked by hand, not by the tool). Same model revision, same neuron (L22 N2742), same clamp scale and the same 20 control
units from Part 18's `results/part18/chat/reads.json`, same prompts, seeds and 96 new tokens; two other differences are listed under Method.

| Cell | Eager (this part) | SDPA (published) |
| --- | --- | --- |
| Part 18: set to +2x, "Who made you?" samples alien | 8 of 8 | 8 of 8 |
| Part 18: unclamped samples alien | 0 of 8 | 0 of 8 |
| Part 18: 20 controls at +2x, 8 samples each, alien | 0 of 160 | 0 of 160 |
| Part 18: controls' greedy answers at +2x naming Qwen | 20 of 20 (0 alien) | 20 of 20 (0 alien) |
| Part 18: set to -2x, samples still naming Qwen | 1 of 8 | 2 of 8 |
| Part 18b: clamp during generation only, samples alien | 8 of 8 | 7 of 8 |
| Part 18b: clamp on the prompt only, samples alien | 0 of 8 | 0 of 8 |

Both parts' published counts stand: every count moved by at most one sample. The answer texts moved much more (below).

## Predictions

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 184.1 | A: the clamped (+2) samples are alien in at least 7 of 8. Fails at 6 or fewer. (high) | 8 of 8 | held |
| 184.2 | A: the unclamped samples are alien in 0 of 8, and the 160 control samples at +2 are alien in at most 2. Fails at 1 or more unclamped, or 3 or more control samples. (high) | 0 of 8; 0 of 160 | held |
| 184.3 | A: the controls' greedy answers at +2 name Qwen in at least 18 of 20 and are alien in 0 of 20. Fails otherwise. (high) | 20 of 20 name Qwen, 0 alien | held |
| 184.4 | A: at -2, the samples name Qwen in at most 4 of 8 (published 2 of 8). Fails at 5 or more. (medium) | 1 of 8 | held |
| 184.5 | B: the generation-only clamp gives alien in at least 5 of 8 samples (published 7 of 8). Fails at 4 or fewer. (medium) | 8 of 8 | held |
| 184.6 | C: the prompt-only clamp gives alien in at most 1 of 8 samples (published 0 of 8). Fails at 2 or more. (medium) | 0 of 8 | held |

Gate: every output file records `attn_implementation` eager in its `load` record. Passed. The pilot (C) ran first; its
kernel was eager and none of its answers was empty, so A and B were started.

## The answer texts differ much more than the counts

Of the 192 greedy answers present in both versions (72 from the target, 120 from the controls), 17 are identical and 175
differ. Run A: 8 of 48 target answers and 1 of 80 control answers identical. Run B: 2 of 12 and 0 of 20. Run C: 4 of 12
and 2 of 20 (C's 18 differing control answers hold only two distinct texts, so they are close to one event).

Where they first differ, in key positions (prompt tokens plus answer tokens): run A's target answers part at keys 22 to
70; 13 of 40 part before key 65. 4 of A's 40 are not kernel differences at all. Part 18's original run used an older stop
setting that did not stop at the end of the turn (the end-of-turn tokens were added to the stop set 30 minutes later,
commit f5e8bc5), so 29 of its 72 stored texts run on past it; in 4 target answers eager simply stops where the original
ran on (for example "XEET." against "XEET.A" at +4x). A's control answers: 26 of 79 part before key 65. B: 4 of 10 target
and 3 of 20 control answers part before key 65. C: 2 of 8 and none of 18.

Four answers change a scored outcome, all greedy and none in a published cell:
- A, clamp -1, "Tell me about yourself in two sentences.", from key 22: SDPA says it was "developed by Google"; eager says
  "developed by Alibaba Cloud's Tongyi Lab". Part 18 published no count for -1.
- A and B, clamp +2, "What is 17 times 3?", from key 66 (both).
- A, control unit 8617 at -2, "Who made you?", from key 67. Part 18's controls at -2 name Qwen in 19 of 20 on eager and
  20 of 20 on SDPA.

This part does not measure why answers part before key 65, and no CPU reference was run on these prompts, so which
kernel is closer to the CPU on any of these answers is not known. A near-tie is the likely reason: two kernels in bf16
give slightly different logits everywhere, so where the top two tokens are almost equal either can take the other. Part
183's check shows such a tie: on its prompt "What causes rain?" both kernels leave the CPU path at answer position 2 (key
18), where the reference's top two were 0.067 apart. So the claim that SDPA agrees with the CPU before key 65 holds on 7
of Part 183's 8 check prompts, not all 8.

This also qualifies the Arc audit's decision-position measurement (kept in the Journal project, not in this
repository), which treated a decision fixed by key 64 as unaffected by the kernel. Here one scored greedy
outcome changed from key 22. Read "fixed by key 64" as "changed only by a near-tie before key 64", not as "unchanged".

## Method

`neuron_report.py steer`, with Part 18's `results/part18/chat/reads.json` for the clamp scale and the control units:

- A, Part 18: the defaults Part 18 used (target ks -4, -2, -1, 0, 1, 2, 4 on six prompts, greedy; 8 samples at
  temperature 0.7 and top_p 0.95, seed 1, on "Who made you?", unclamped and at -2 and +2, generated 8 at a time; 20
  controls greedy at -4, -2, +2, +4 and sampled 8 each at +2). 23 min 24 s.
- B, Part 18b generation-only: `--phase gen --ks 2 --control-ks 2 --sample-ks 2`. 15 min 57 s.
- C, Part 18b prompt-only: `--phase prompt --ks 2 --control-ks 2 --sample-ks 2`. 9 min 2 s.

Two differences from the originals besides the kernel. The tool is a later version: the reruns ran `neuron_report.py`
at commit 8544a17 (file hash b56bfb07), the Part 18b originals at f32b4b9 (a366c909), and Part 18's original before the
stop-set commit above, so it records no tool hash and no load record. The changes in between add the kernel argument and
its record, the teacherforce stage and Part 19's provenance; `score_text` and its patterns are unchanged. And Part 18's
original did not stop at the end of the turn, as above; its scorer reads up to the first role-turn marker, so its counts
are unaffected, and so is the count of identical answers.

Scored by the tool's own `score_text`. `part184/compare184.py` reads the eager and stored summaries and writes
`compare.json`; `part184/textdiff184.py` writes `greedy-text-compare.json` (first-difference key = chat-template prompt
length plus the 0-based index of the first differing answer token).

## Limits

One model, one GPU, one neuron and its 20 controls, as the originals. Sampled answers move with any numeric change, so
counts are compared, not answers. The eager kernel was checked against the CPU on eight prompts, not on these prompts,
and no margins were recorded here, so the near-tie reading is untested. English only.

## Files

- `part18-eager/steer.json`, `part18b-gen/steer.json`, `part18b-prompt/steer.json`: the three eager runs.
- `compare.json`: every eager count beside the stored SDPA count, and the verdicts.
- `greedy-text-compare.json`: every greedy answer pair, identical or not, with its first-difference key and whether its
  scored outcome differs.
- `launch.jsonl`: the two launches (C, then A and B). Code: `part184/compare184.py`, `part184/textdiff184.py`. Design:
  `PART184-DESIGN.md`.

Cold check: Claude Haiku 5.5 (no project context), 2026-11-04, 10 findings: the premise that SDPA agrees with the CPU
before key 65 fails on one of Part 183's eight prompts; a greedy maker change (Google against Tongyi Lab) left out; the
control answers left out; Part 18's original used an older stop setting; the tool version not disclosed; "eager follows
the CPU" overstated; the near-tie reading untested; the front-page note missing; one example overstated; the text
comparison not reproducible from the repository. All folded: the counts above come from `part184/textdiff184.py`, which the
cold check's own recount matches in every total except A's control answers parting before key 65 (26 here, 23 there,
not resolved).
