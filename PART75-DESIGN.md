# Part 75: in sampled answers, does restoring L27 N3325 bring the Google fallback back when L24 N3949 is zeroed?

Written 2026-10-26 by Claude, before any code or run. Predictions 75.1 to 75.3 are in PREDICTIONS.md and are pushed
before the run.

Size: 4 conditions x 240 sampled answers (192 held-out, 24 positive, 24 guard) = 960 generations. Part 70 ran 960
generations of this code path on the A770 (797 s) and Part 73 ran 1,680, which serve as the pilot.

## Why

Part 74 found, on a forced-opening next-token readout, that holding L27 N3325 at its makers-only values removes about
70% of L24 N3949's effect on the Google-minus-Alibaba gap. Its main limit is that it read one token. Part 73 measured
sampled answers: with the makers zeroed, zeroing L24 N3949 took the held-out Google answers from 36 to 17 of 192.
Part 75 asks whether the same route shows in sampled answers.

## Method

- Part 73's code path (`part70.run_70`, Part 66's prompts, sampling and scorer), `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770, batch 32.
- m = L27 N3325's mean down_proj input over all positions of Part 66's 24 held-out chat prompts, with the four makers
  zeroed (one recording pass, no sampling).
- Conditions, each with the four makers zeroed:
  - `target`: makers only;
  - `target_plus_l24`: + L24 N3949 zeroed;
  - `target_plus_l24_clamp_l27`: + L24 N3949 zeroed, and L27 N3325 clamped to m at every prompt and decode position;
  - `target_clamp_l27`: L27 N3325 clamped to m, L24 N3949 left alone (a control for the clamp itself).
- Part 70 supports zeroing only; Part 75 adds an optional per-condition clamp value, and conditions without one run
  exactly as before.
- Readout: held-out answers naming Google, scored by Part 66's `score66`.

## Reading rules (written now)

- 75.1 and 75.2 are controls. If either fails, 75.3 is not read.
- 75.3 holding means the route Part 74 found on one token also shows in sampled answers, with a constant clamp.
- A constant clamp is not the same as Part 74's per-position patch: it sets L27 N3325 to one typical value everywhere,
  including decode positions Part 74 never saw.

## Limits (written now)

- m is a mean over prompt positions only; the decode positions' natural values are not known in advance.
- One model, Part 66's 24 held-out questions, one sampling setting; samples differ between runs because batches mix
  conditions (Part 73: the makers-only count moved by 2 of 192 between runs).
