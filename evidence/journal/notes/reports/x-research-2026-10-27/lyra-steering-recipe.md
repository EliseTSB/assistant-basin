---
title: Lyra's steering recipe and the evil-Qwen neurons, read at source for Part 79
authored-by: claude
date: 2026-10-27
description: How lyramakesmusic/neuron-explorer doses a push, where its scale comes from, and what the two evil-Qwen neurons write, read from the repo's code and cache files on 2026-10-27 02:0x CT.
---

# Lyra's steering recipe, read at source

Sources: `sweep/steer_sweep.py`, `sweep/neuron_stats.py`, `research_log.md`, `explorer/cache/steer_index.npz` and
`explorer/cache/vocab.json` in github.com/lyramakesmusic/neuron-explorer (HEAD on 2026-10-27), and the post
x.com/_lyraaaa_/status/2113833330429083662 (read with `scripts/x-thread.mjs`).

## How a push is dosed

- **A push ADDS to the neuron's activation; it does not set it.** `steer_sweep.py` sends `dose x scale x W_down[:, n]`
  to vLLM's `mlp_out` steering hook, which equals adding `dose x scale` to the neuron's activation.
- **At every position, prompt and answer both.** Our alien pushes (Parts 62, 78) SET the activation, and Part 62
  found N2742 acts only while the model writes. A replication of Lyra's push must add at every position.
- **Two units.** `absmax`: scale is the neuron's largest |activation| over the stats corpus. `rel`: scale is
  `median(||resid|| leaving layer L, positions >= 1) / ||W_down[:, n]||`, so a dose of 1 pushes the residual stream by
  its own median norm. The evil-Qwen recipe is in `rel` units (the post says so).
- **Stats corpus:** 2M tokens, FineWeb-Edu sample-10BT interleaved with UltraChat chat-templated (thinking off),
  1024-token packed windows, position 0 of each window skipped as an attention sink. `abs_max` grows with corpus
  size (median 6.9 at 200K tokens, 10.25 at 2M), so `absmax` doses do not transfer across corpora; `rel` doses
  should transfer better, because a median residual norm is stable.
- **Decoding:** greedy (temperature 0), `enable_thinking=False`, max 256 new tokens, the chat template with one user
  turn and no system prompt.
- **Coherent window** (their log): pushes of about 0.5 to 0.8 of the residual norm stay fluent; about 3x reads the
  direction out as wreckage. The evil recipe (+1 and -1) sits just above the window.
- Known value: L22's median residual norm is 25.4 on their corpus. L13 and L14 are not published anywhere in the
  repo (`act_scale.npz` holds per-neuron activation ranges, `steer_index.npz` the steer lens), so Part 79 has to
  measure them.

## What the evil neurons write (steer lens, +-0.8 rel, one forward pass on 4 short contexts)

| Neuron | +0.8 raises | -0.8 raises |
|---|---|---|
| L14 N8949 (pushed +1) | Fuck, evil, 邪教 (cult), Satan, bitch, Porn | 肯定 (certainly), 确实 (indeed), that |
| L13 N7956 (pushed -1) | normally, usual, Normal, 正常 (normal) | 经, that, 那也是, 树 |
| L22 N2742 (alien, for comparison) | extr, 外星 (alien), alien, aliens | 毒症, 阴虚 (unrelated) |

So evil Qwen is "add evil, remove normal". Lyra's follow-up post: a personality tune, an over-the-top villain that
becomes much worse at everything; not a refusal ablation.

## What this means for Part 79

1. Replicate the push exactly: add at every position, `rel` units, with L13 and L14 medians measured on our own
   corpus by the same rule (positions >= 1). Gate: the evil push alone must produce the villain on "Who are you?"
   before any maker result is read.
2. The makers-zeroed condition is unchanged from Parts 66 to 78 (four neurons zeroed at every position).
3. A size-matched control push at L13 and L14 (Lyra's own controls rotate the down column to a random direction of
   the same norm), so a Google change from "any push of this size" is separable from the trait.
