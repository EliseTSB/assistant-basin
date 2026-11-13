# Part 206, Amendment 1: a runner fix before any Qwen data

Written 2026-11-12 14:47 CT by Claude (Opus 5.5), after the first Qwen3.5-4B smoke (step 1) failed and before any Qwen row exists.
No prediction, bar, size or reading rule changes.

**What failed.** Both attempts of the batch-2 smoke died after about 17 seconds in `reduce_batch` with
`RuntimeError: Required aspect fp64 is not supported on the device`: the Intel Arc A770 has no float64 kernels, and
`record206.py` converted XPU tensors with `.double()` before copying them to the CPU (19 places) and allocated two
float64 accumulators on the model's device. The OLMo smoke ran on the CPU, where float64 exists, so it could not show
this.

**The fix.** Every conversion is now `.cpu().double()` (copy first, then convert), and the two accumulators live on the
CPU with their inputs copied there before `index_add_`. On the CPU path nothing changes: the float64 conversion still
follows the same device-side sum, and the accumulators were already on the CPU. A new test,
`test_no_float64_on_the_device`, fails on the old file and passes now; the suite is 40 passed.
