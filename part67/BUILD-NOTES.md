# Part 67 build notes

Built a one-invocation prefill runner for the pinned Qwen3.5-4B revision, 25 Part 66 prompts, the registered 207
conditions, position-level unchanged activations, six ordered-pair effects, random single and triple controls, and
direct gate/up weight paths. Added a scorer for the normalized deltas, empirical two-sided absolute-effect p-values,
last-position effects, exploratory weight ranks, control gates, and six prediction verdicts. Added synthetic scoring,
condition-list, and tiny-model hook tests plus run instructions.

`python3 -m compileall -q part67` and direct scorer assertions passed. The pytest suite could not be run because
`pytest` is not installed (`python3 -m pytest` reported no module named pytest); `python` is also not an available
command, so Python 3 was used. The brief states this worker environment has no PyTorch or model; the real checkpoint
run and PyTorch-dependent condition and tiny-model tests therefore could not be run here.

Module-name assumptions: the weight-path code searches each runtime decoder block for unique `gate_proj` and `up_proj`
modules whose output width matches the MLP intermediate width. It locates the pre-MLP RMSNorm as the unique width-matched
norm inside the MLP, or otherwise requires a uniquely named `post_attention` norm. It prints discovered module paths
and raises if projection or norm discovery is ambiguous or fails. This reflects the expected Qwen decoder layout but
has not been confirmed against loaded weights in this environment.

Ambiguity: prediction 67.2 says the largest effect ends at L29 without stating whether “largest” means signed or
absolute effect; the design consistently treats effects by magnitude and defines null comparisons with absolute values,
so this build uses the largest absolute Delta. Prediction 67.3's “changes L29 more” is interpreted by absolute Delta,
consistent with the null definition. For 67.4 “top 5 of 51” the scorer uses one-based absolute-value competition rank
and accepts ties at rank 5. The “answer starts” last-position variant is implemented as the final prompt token's
down-projection input, matching the prompt-prefill-only measurement.
