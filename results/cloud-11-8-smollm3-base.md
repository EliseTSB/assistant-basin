# Part 11.8-11.9: SmolLM3-3B vs SmolLM3-3B-Base (Claude cloud, CPU)

Run 2026-10-16 18:44-19:04 UTC in cloud session session_011ePj23CSKYLQCegAfeWvhA (Sonnet 5.5, 4 cores, 15 GB, no GPU),
task CLOUD-TASK-11-8.md in the neuron-probe repo at c1566e8. The session cannot push; numbers below are quoted from its
final message and run log, read with RemoteTrigger get_run_log.

Commands (bfloat16, 256 neutral tokens, self-name "SmolLM:Hugging Face"):
- collect.py HuggingFaceTB/SmolLM3-3B (plain + chat forms, 14:24 wall, 10.4 GB peak)
- collect.py HuggingFaceTB/SmolLM3-3B-Base (plain form, 3:34 wall, 9.0 GB peak)
- analyze.py SmolLM3-3B.npz --form plain --top 20 --score SmolLM3-3B-Base.npz
- analyze.py SmolLM3-3B-Base.npz --form plain

| | chat (SmolLM3-3B) | base (SmolLM3-3B-Base) |
|---|---|---|
| heldout_top_median | 0.739 | 0.655 |
| heldout_random_median | -0.291 | -0.275 |
| null_top_max | 0.343 | 0.352 |

The chat model's top 20 plain-form neurons, scored in the base: heldout_top_median 0.602.
Layer histogram of the chat model's top 20: 7 in layer 35, the rest between layers 12 and 25.
Base model's own top 20: mostly layers 9-16, 6 in layer 35.

One run per model; no variance estimate. The cloud's `git status` showed judge.py modified: that was an uncommitted
local edit in ~/neuron-probe carried in the seed bundle, not a change the run made.
