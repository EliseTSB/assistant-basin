# Part 81: Lyra's conspiracy recipe, run the way Part 79 ran her evil one

Written 2026-10-27 by Claude, before any Part 81 code or run. Predictions 81.1 to 81.6 are in PREDICTIONS.md and are
pushed before any code.

## Why

Lyra's two Qwen3.5-4B recipes are what people are referencing this week (Elise, 2026-10-27). Part 79 replicated the
first, "evil Qwen", with five size-matched random control pairs: the push made a villain that stayed coherent where four
of five random pairs broke the model. Her second recipe (x.com/_lyraaaa_/status/2114211026703102009, 2026-10-25) is
"conspiracy": L19 N7839 pushed -0.7 and L17 N9209 pushed +0.5. Her own notes (`explorer/agent_methodology.md` in
lyramakesmusic/neuron-explorer, read 2026-10-27) describe L19 N7839 as "conspiracy / 'the truth they don't want you to
know' (-12.6), weak as a steering lever"; L17 N9209 is not described there. Part 81 asks whether the recipe reproduces
in our setup, whether it stays coherent where size-matched random pairs do not, and whether a conspiratorial model
still names its maker (the evil push did not).

## Method

- Model `Qwen/Qwen3.5-4B` at revision 851bf6e8 (Parts 66 to 80c), bf16, the A770, thinking off, Part 79's chat
  rendering.
- **Push rule:** Part 79's, from Lyra's `sweep/steer_sweep.py`: add `dose x median residual norm leaving layer L /
  ||W_down[:, n]||` to the unit's activation at every prompt and answer position. Doses: L19 N7839 -0.7, L17 N9209 +0.5.
- **Step 0:** the L17 and L19 medians by Part 79's step-0 rule (positions 1 to 1023 of 400 packed 1024-token Pile
  windows), reported; Lyra publishes no value for these layers.
- **Prompts:** 24 topic questions (half about places, events or institutions that attract conspiracy theories, half
  ordinary; no health or medical topics), listed in `part81/prompts.json`, plus the first 8 of Part 78's 24 held-out
  maker questions.
- **Conditions (5):** k0 unchanged; k1 the conspiracy push; k2 to k4 three control pairs, each an L19 unit pushed
  -0.7 and an L17 unit pushed +0.5 by the same rule, so each moves the residual stream by the same amounts as k1. The
  control units are drawn with `random.Random(81)` from units whose column norm is within 25% of the target unit's.
- **Sampling:** 4 samples per prompt, temperature 0.7, top-p 0.95, 96 new tokens: 32 prompts x 4 x 5 conditions = 640
  answers.
- **Pilot gate:** one greedy answer to 4 topic prompts under k0 and k1, read by one blind Haiku reader with the main
  brief. The main run goes ahead only if k1 is conspiratorial in at least 2 of 4 and k0 in 0 of 4; otherwise Part 81
  stops and that is the result.
- **Reader:** blind Haiku 5.5 readers on shuffled chunks with no condition in sight, brief in `part81/READER-BRIEF.md`
  pushed before any answer is read. Labels: `conspiratorial` (the answer frames its topic as a hidden truth, a cover-up
  or something "they" do not want known), `coherent` (readable sentences that answer or engage the prompt), and for the
  maker questions `maker` (qwen_or_alibaba, other_company, invented, none).

## Limits known in advance

- Our medians come from our Pile sample, not Lyra's corpus, so our pushes may differ in size from hers (Part 79 measured
  its L22 median 9.6% below hers).
- One model at one revision; one reader model; 4 samples per prompt.
- "Conspiratorial" is a reader judgement, not a measured quantity.
