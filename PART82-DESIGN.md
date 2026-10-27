# Part 82: do Lyra's "own earlier messages" heads carry Qwen3.5-4B's own-message gap?

Written 2026-10-27 by Claude, after Part 81 was scored and before any Part 82 code or run. Predictions 82.1 to 82.6 are
in PREDICTIONS.md and are pushed before any code.

## Why

Parts 80 to 80c found that Qwen3.5-4B says "you wrote it" far more often for a text placed in an Assistant message than
for the same text in a User message (Part 80b's cyclic-average gap 0.444; 0.205 with the option changed to plain "You",
Part 80c). Lyra's methods notes for the same model (`explorer/agent_methodology.md` in lyramakesmusic/neuron-explorer,
commit of 2026-10-26, read 2026-10-27) list reference heads: "L11 h14 and L15 h1/h2/h3/h12/h15: recall the current
speaker's own earlier messages, by name" and "L15 h10: reads the previous turn (the message being replied to)". Layers
11 and 15 are two of Qwen3.5-4B's eight full-attention layers (3, 7, ..., 31; 16 query heads of width 256). If those
heads carry "which messages are mine", switching them off should shrink the gap; if the gap comes from somewhere else,
it should not move more than switching off other heads in the same layers.

**Prior art (searched 2026-10-27).** Lyra's notes are the source of the head list; how she identified them is not
stated there, and this is not a test of her method, only of one consequence. Part 80's own prior-art section covers the
self-recognition literature (Li et al.; Davidson et al., arXiv 2407.06946). I found no published head-ablation test of
role-based self-attribution in a small model.

## Method

- Model `Qwen/Qwen3.5-4B` at revision 851bf6e8, bf16, the A770, thinking off; question, options, chat rendering and
  readout identical to Part 80b (`part80b/part80b.py`).
- **Bodies:** Part 80b's primary set, Part 37's 100 greedy SmolLM3 answers.
- **Renderings:** A and U (the body as an Assistant or a User message, then the question as a User turn).
- **Orders:** Part 80b's three cyclic orders O1, O2, O3.
- **Option sets:** Part 80b's ("You (the AI assistant)", ...) and Part 80c's set N ("You", "The user", "The system").
- **Head ablation:** Lyra's rule (`explorer/runtime.py`, `_head_hook`): a forward pre-hook on the layer's `o_proj`
  multiplies that head's 256-wide slice of the input (after the output gate) by 0, at every position.
- **Conditions (8):** h0 unchanged; h1 the six heads off (L11 h14; L15 h1, h2, h3, h12, h15); h2 L15 h10 off alone;
  c1 to c5 five control sets of six heads, one from L11 and five from L15, drawn with `random.Random(82)` from the heads
  Lyra does not name in those layers (L11: the other 15; L15: 0, 4, 5, 6, 7, 8, 9, 11, 13, 14).
- 8 conditions x 100 bodies x 2 renderings x 3 orders x 2 option sets = 9,600 forward passes; no generation.

## Measures

- Per Part 80b: `P_k(you | body, rendering)` renormalised over the three letters; the per-order gap
  `gap_k = mean over bodies of [P_k(you | A) - P_k(you | U)]`; the cyclic-average gap from the three orders averaged
  per body; mean letter mass per pass.
- **Drop:** `drop(h) = gap(h0) - gap(h)` for the cyclic-average gap, with a 95% interval from a paired bootstrap over
  bodies (10,000 resamples, seed 82, percentile).

## Reading rule

- The heads carry part of the gap if drop(h1) is at least 0.10 with its interval above 0 and larger than every control
  set's drop, in Part 80b's option set. Set N is reported beside it.
- If h0 does not reproduce Part 80b and 80c within 0.05, nothing else is read.

## Limits known in advance

- Lyra's description is about speakers named in transcripts; our setup marks authorship by chat role, so a null here
  does not contradict her.
- Zeroing heads at every position is a blunt intervention; a drop could come from damage rather than lost
  attribution. The control sets are for that, and letter mass is recorded.
- One model, one revision, one body set.
