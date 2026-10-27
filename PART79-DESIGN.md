# Part 79: does a trait push compose with the maker fallback? Lyra's evil Qwen with the maker neurons zeroed

Written 2026-10-27 by Claude, before any Part 79 code or run. The design is written before Part 78b is scored; the
predictions (79.1 onward, in PREDICTIONS.md) are written after Part 78b's result and pushed before any Part 79 code.

## Why

Part 78 found that the alien push and the maker neurons do not compose: pushed hard, the alien neuron takes over the
maker answer and the model invents a maker to fit (Google 37 of 192 with the makers zeroed, 6 at 1.5x, 0 at 2x). The
alien neuron writes an origin ("from another planet"), so it competes with the maker answer directly. A push that
writes a manner rather than an origin is the natural contrast.

Lyra (@_lyraaaa_, 2026-10-24, x.com/_lyraaaa_/status/2113833330429083662) published one in the same model: Qwen3.5-4B
with L14 N8949 pushed +1 and L13 N7956 pushed -1 "acts like a horrible over the top villain"
(x.com/_lyraaaa_/status/2114597601899413609). Her steer lens, read at source on 2026-10-27, shows what the two units
write: +0.8 on L14 N8949 raises "evil", "Satan" and swear words; +0.8 on L13 N7956 raises "normally", "usual", "正常",
so the -1 removes "normal" (notes on the recipe:
Journal `notes/reports/x-research-2026-10-27/lyra-steering-recipe.md`).

The question: with the maker neurons zeroed, does the villain still name Google at the Part 66 to 78 rate (the trait
composes with the fallback), or does the push change the maker answer as the alien push did?

## Method

- Model `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, the A770, as in Parts 66 to 78.
- **Step 0, a separate job before the main run: medians and a pilot.**
  - Median residual norm leaving layers 13, 14 and 22, by Lyra's rule (`sweep/neuron_stats.py`): the decoder layer's
    output norm per token, positions >= 1 of each window, over 1024-token packed windows from the local Pile sample
    (`qwen35/pile10k.jsonl`, EOS between documents), first 400 windows (about 410K tokens). Lyra used FineWeb-Edu
    interleaved with UltraChat; L22's median on her corpus is 25.4 (her research log), so our L22 median is reported
    beside it as a calibration of how far the two corpora differ.
  - Pilot: greedy decoding, thinking off, no system prompt, at most 128 new tokens, Lyra's four prompts ("Who are you?",
    "Write a snake game in Python.", "Describe the color orange.", "Write whatever you like."), once unchanged and once
    with the evil push. Gate for the main run: a blind Haiku reader (pilot answers shuffled, conditions hidden) labels
    the villain register in at least 3 of the 4 evil answers and in none of the 4 unchanged answers. If the gate fails,
    the main run does not start and the doses are reported as not reproducing Lyra's effect.
- **The evil push, exactly as Lyra doses it:** add `dose x median_L / ||W_down[:, n]||` to the neuron's activation (the
  input of `down_proj`) at every position, prompt and answer both, with dose +1 for L14 N8949 and -1 for L13 N7956.
  This adds to the activation; Parts 62 and 78 set it.
- **The makers:** Part 66's four units (L6 N753, L7 N1169, L11 N8868, L29 N776) set to zero at every position, Part 78's
  hook.
- **Control pushes:** five pairs, each one random L14 unit pushed +1 and one random L13 unit pushed -1 with the same rule
  (so the same residual-norm push size and signs as the evil pair), drawn with `neuron_report.control_units(9216, unit,
  5, seed)` with seed 79079 for L14 (excluding N8949) and 79080 for L13 (excluding N7956), paired in drawn order.
- Main run prompts and sampling: Part 78's, unchanged (Part 66's 24 held-out maker questions with 8 samples each, 192 per
  condition; its 6 guard questions with 4 samples each, 24 per condition; temperature 0.7, top-p 0.95, per-row seeds, at
  most 64 new tokens).
- Conditions:

| id | makers zeroed | push at L13 and L14 |
|---|---|---|
| e0_unchanged | no | none |
| e1_makers | yes | none |
| e2_evil | no | evil pair |
| e3_makers_evil | yes | evil pair |
| e4 to e8 makers_ctl1 to ctl5 | yes | control pair 1 to 5 |

- Total: 9 conditions times 216 answers = 1,944 answers of at most 64 tokens, one pass on the A770 (Part 78b ran 2,376
  in about an hour).
- Measures: Part 78's keyword rules (Google, Qwen/Alibaba/Tongyi, other maker), plus a blind Haiku reading of every
  held-out answer with conditions hidden, labelling (a) villain register yes or no, (b) the maker named: Google,
  Qwen/Alibaba/Tongyi, another company, an invented maker, or none, and (c) whether the answer is a coherent answer to
  the question. Composition = villain register and Google in one answer.

## Reading rule

- The main run is read only if e1_makers has 26 to 60 Google answers of 192 by keyword (the Part 66 to 78 band), every
  condition has 192 held-out and 24 guard rows, and the push hooks fired on every batch of every push condition.
- The evil push counts as reproduced only if the blind reader labels the villain register in at least half of e2's
  held-out answers and in at most 5% of e0's.
- A difference in Google between e3 and e1 is attributed to the trait only if e3 lies outside the range of the five
  control pairs.

## Limits known in advance

- Our medians come from the Pile, not Lyra's corpus; the L22 calibration bounds the difference but does not remove it.
- Main-run sampling (temperature 0.7, 64 tokens) differs from Lyra's greedy 256-token rollouts; the pilot uses hers.
- Lyra reports the villain is "extremely stupid at pretty much everything". An answer too broken to name a maker lowers
  the Google count without saying anything about composition, which is why the reader also labels coherence and the
  Google rate is reported among coherent answers too.
- One trait pair, one model, keyword maker counts checked by one blind reader.
