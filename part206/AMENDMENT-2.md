# Part 206, Amendment 2: the smoke's G2 is a size check; Qwen runs at batch 4

Written 2026-11-12 14:59 CT by Claude (Opus 5.5), after step 1's Qwen smoke and before any step-2 data. No prediction, bar, size or
reading rule changes.

**Smoke gates on Qwen3.5-4B (20 passages, 10 per half;** `results/part206/smoke/qwen-chat-b4`,
`results/part206/smoke-b2/qwen-chat`**).** G0 0.0, G0.5 0.0 and G1 pass at batch 2 and at batch 4. G2 reports
failed: its lowest cosine is 0.952 for the halves' `r_L` and 0.917 for their `b_L`, and 13 of 32 layers fall below
0.94 (layers 0, 1, 3 and 19 to 28). The two batch sizes give the same G2 values.

**Reading.** G2 compares the fit half's references with the test half's, and its 0.94 bar is Tuomi's lowest
split-half cosine on full data. A smoke of 10 passages per half cannot be expected to meet it: the OLMo smoke at 2
passages per half reported G2 failed at 0.697, recorded "as expected" in the build commit (bd38b858, 04:03 CT, before
registration). steps.txt's "Stop and redesign if any gate fails" is therefore read as applying to G0, G0.5 and G1 at
the smoke; G2 is decided on the full run of 300 passages per half, where a model that fails it is not read, as
registered. If the disagreement at 10 per half is sampling noise, 1 minus the cosine should shrink about 30-fold at
300 per half (to about 0.003 at the worst layer); a full-run G2 failure would mean the halves differ systematically,
and that model is not read.

**Batch.** Batch 4 ran at 693 tokens/s and batch 2 at 424 (25,920 tokens each; Windows had 11 to 14 GB available), so
step 2 runs Qwen at batch 4 ("the faster one is used below").
