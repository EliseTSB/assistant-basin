# Part 73: which of the two late neurons carries the Google fallback?

Written 2026-10-26 08:01 CT by Claude, before any code or run. Predictions 73.1 to 73.4 are in PREDICTIONS.md and are pushed
before the run.

Size: (3 conditions x 240) + (40 single-unit nulls x 24) = 1,680 sampled answers. Part 70 ran the same code path on
the same model this morning (960 answers, 797 s, peak 9.1 GB), which serves as the pilot.

## Why

Part 70 zeroed L27 N3325 and L24 N3949 together on top of the four maker neurons and the Google answers went from 38
to 4 of 192. The two were never zeroed apart. Part 71 found that L27 N3325 writes "Google" (rank 1) and L24 N3949
writes Apple and Microsoft (Google rank 36); Part 72 found that zeroing L27 N3325 alone lowers the Google gap in the
unaltered model. Part 73 asks which unit the fallback depends on, in sampled answers.

## Method

- Part 70's code path unchanged (`part70.run_70`, Part 66's sampling, prompts and scorer), `Qwen/Qwen3.5-4B` at
  revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770, batch 32 (Part 69's size; Part 70 used 25).
- Conditions, each with the four maker neurons zeroed:
  - `target`: makers only (a rerun of Part 70's condition; batches mix conditions, so samples will differ);
  - `target_plus_l27`: makers + L27 N3325;
  - `target_plus_l24`: makers + L24 N3949;
  - nulls: for k = 0..19, makers + the layer-27 unit of Part 70's matched pair k (`target_plus_l27_matched_k`), and
    makers + the layer-24 unit of that pair (`target_plus_l24_matched_k`). The draws are Part 70's own, so each single
    null is half of a pair whose result is already known.
- The three main conditions get Part 70's full rows (192 held-out, 24 positive, 24 guard); each null gets the 24
  held-out questions once.
- Readout: held-out answers naming Google, scored by Part 66's `score66`.

## Reading rules (written now)

- 73.1 is the control. If it fails, nothing else is read until the rig is checked.
- 73.2 and 73.3 decide whether L27 N3325 alone carries the fallback; 73.4 whether L24 N3949 alone does.
- If both 73.2 and 73.4 fail, the fallback needs both units together, which Part 70 cannot distinguish from either.
