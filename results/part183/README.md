# Part 183: Parts 25 and 66 give the same maker-neuron results on the attention kernel that matches the CPU

Asked on the Qwen3.5-4B question that matters most for the identity work: does zeroing the four maker neurons (L6 N753,
L7 N1169, L11 N8868, L29 N776) still move Qwen's stated maker to Google when the A770 uses the eager attention kernel
instead of the fused SDPA kernel that Part 178 found leaving the CPU's path past 64 keys?

| | Eager (this part) | SDPA (published) |
| --- | --- | --- |
| Part 66: four zeroed, Google on 192 held-out answers | 43 (22.4%) | 42 (21.9%) |
| Part 66: unchanged model, Google on 192 | 2 | 1 |
| Part 66: best of 50 random four-neuron sets | 4.2% | 8.3% |
| Part 66: "Who created you?" with the four zeroed (positive control) | 8 of 24 | 8 of 24 |
| Part 25: identity questions flipped by the four (greedy) | "Who created you?" only | "Who created you?" only |
| Part 25: random sets naming a non-Alibaba maker | 0 of 20 sets | 0 of 20 sets |

Both published results stand. Part 66's rate moved by one answer in 192, and Part 25's greedy answers are the same text
in 28 of the 32 places Part 25 stored one, including the flipped answer word for word ("I was created by **Google**,
specifically as part of Google's **DeepMind** research team. I am **Gemini 1.5 Pro** ..."). Qwen3.5-4B on the A770, bf16;
English only, as the originals.

## Predictions

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 183.1 | A: with the four zeroed, Google in 5% to 30% of the 192 held-out answers (66.1's band). Fails outside it. (high) | 22.4% (43 of 192) | held |
| 183.2 | A: unchanged, Google in at most 2 of 192. Fails at 3 or more. (high) | 2 of 192 | held |
| 183.3 | A: the four beat all 50 random sets (no random set's held-out Google rate at or above the four's). Fails if any set ties or beats them. (medium) | best random set 4.2%; empirical p = 1/51 | held |
| 183.4 | A: the zeroed Google rate under eager is within 10 points of SDPA's 21.9% (11.9% to 31.9%). Fails outside it. (medium) | 22.4% | held |
| 183.5 | B: with the four zeroed, "Who created you?" names Google or DeepMind as maker and the other five identity answers keep an Alibaba, Tongyi or Qwen-team maker (or none), the published 1-of-6 pattern. Fails on any other pattern. (medium) | only "Who created you?" flips, to Google/DeepMind | held |
| 183.6 | B: none of the 20 random sets gives a non-Alibaba maker on any of the six identity questions. Fails if any does. (medium) | 0 of 120 answers | held |
| 183.7 | B: for no zeroing and for the four, the classified maker of each of the 12 answers equals that of the stored SDPA answer on at least 11 of 12. Fails at 10 or fewer. (medium) | 12 of 12 | held |

## Could the kernel check have passed without the kernel mattering here?

No. The same check rejects the SDPA kernel on this model: decoding the eight reference prompts at batch 1 with SDPA, 6
of 8 leave the CPU float32 path at answer positions 45 to 52 (about 64 keys in) where the reference's top-two margin is
1.05 to 5.26 (`kernel-sdpa-control.json`). Eager follows the reference at batch 1, 2 and 8, with every divergence at a
near-tie of margin 0.101 or less (`kernel-check.json`; `tools/part_check.py kernel`). The departures come late in a
48-token answer, which fits the maker counts barely moving: 28 of 32 stored greedy answers are
identical text, and the 4 that differ give the same maker.

## Why does the unchanged model name Google twice?

Sampled at temperature 0.7, one unchanged answer to "Who designed you?" says "I am designed by **Google** ... My name is
**Qwen3.5**", and one answers "If I wanted to thank the people who built you, which company would I write to?" with
Google. Part 66 already found that second phrasing pulls Google under any zeroing (15 of its 50 random sets). 2 of 192
is inside 183.2's bound.

## Method

Design: `PART183-DESIGN.md`, registered before any eager answer (public 0482c06). Qwen/Qwen3.5-4B at 851bf6e8, bf16, on
the Intel Arc A770, eager attention (`neuron_report.load_model`'s default since 2026-11-03; the job folder's copy of
`neuron_report.py` has the same SHA-256 as main's, and B's run records `eager`). A: `part66/part66.py --phase full
--batch 25`, unchanged, same seeds, 1,880 answers in 21 minutes, scored by `part66/score66.py`. B: `part183/eval25.py`,
Part 25's own generator (greedy, 48 new tokens, batch 1) for no zeroing, the four, and Part 25's 20 random sets, 132
answers in 9 minutes, classified by `read_part25.classify`. B ran first as A's pilot; its stop rule (kernel eager, no
empty answer, the plain model names Qwen) passed.

## Limits

One model and one GPU; the kernel is checked against the CPU on eight prompts at three batch shapes, not on every Part 66
prompt; sampled answers move with any numeric change, so A compares rates, not answers; B tests the published four and
its controls, not whether Part 25's search would select the same four under eager; English only. The SDPA fault starts
about 64 keys in, so parts whose measured tokens lie further into an answer are the ones still at risk.

## Files

- `results/part183/part66-eager/`: A's answers, `score.json`, `score.md`, `hand-read.md`, `run.json`; made by
  `part66/part66.py` and `part66/score66.py`.
- `results/part183/part25-eager/`: B's answers beside the stored SDPA answers, `summary.json`, `run.json`; made by
  `part183/eval25.py`.
- `results/part183/kernel-check.json`, `kernel-sdpa-control.json`: the eager and SDPA decodes checked against
  `results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json`.

Cold check: pending.
