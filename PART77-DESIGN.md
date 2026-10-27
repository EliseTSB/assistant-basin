# Part 77: in sampled answers, does restoring L27 N3325 position by position bring the Google fallback back?

Written 2026-10-26 by Claude, after Parts 75 and 76, before any Part 77 code or run. Predictions 77.1 to 77.4 are in
PREDICTIONS.md and are pushed before the run.

## Why

Part 75 asked this with a constant clamp, and its control failed: clamping L27 N3325 to its prompt-wide mean (-0.25)
cut Google answers from 36 to 16 on its own, so the test was not read. A constant is the wrong restore. The right one
sets L27 N3325, at every prompt and decode position, to the value it would have had with L24 N3949 intact, on the same
tokens. Part 76 found that, on a forced-opening readout, L24 N3949's effect through layer 27 runs through L27 N3325 alone
among 26 validated units.

## Method

- Part 73's sampling setup: Part 66's prompts (240 per condition: 192 held-out, 24 positive, 24 guard) and scorer,
  `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770, at most 48 new tokens,
  temperature 0.7, top-p 0.95. The four makers are zeroed in every row.
- **Twin rows.** Every answer is generated as a pair of batch rows that share every token. The *twin* row has the makers
  zeroed and nothing else. The *answer* row carries the condition's interventions. A forward pre-hook at layer 27's
  down_proj copies the twin's L27 N3325 value into the answer row at every position, in the conditions that restore.
  Because both rows hold the same tokens, the copied value is exactly what L27 N3325 would be on the answer row's own
  prefix with L24 N3949 intact.
- **Shared tokens.** A logits processor samples each pair's next token from the answer row's scores (temperature 0.7,
  top-p 0.95, a generator seeded from Part 66's row seed), then sets both rows' scores to that one token, so
  `model.generate` (called with temperature 1 and top-p 1) appends the same token to both. Only the answer row is scored.
- Conditions:
  - `c0_makers`: answer row = makers only, with the copy (the copy is then an identity; this checks the twin machinery).
  - `c1_l24`: answer row = makers + L24 N3949 zeroed, no copy.
  - `c2_l24_restore`: answer row = makers + L24 N3949 zeroed, L27 N3325 copied from the twin at every position.
  - `c3_l24_restore_nulls`: answer row = makers + L24 N3949 zeroed, with the 10 layer-27 units of Part 76's
    `top_estimate` list copied from the twin instead (the strongest other candidates; a control for copying anything).
- Readout: held-out answers naming Google, by Part 66's `score66`.

Size: 4 conditions x 240 answers x 2 rows = 1,920 row-generations. Part 75 (960, batch 32) took 728 s on the A770, so
about 25 minutes at the same batch in rows; an optimization-pass record is written before launch.

## Reading rules (written now)

- 77.1 is the control. If it fails, nothing else is read.
- 77.2 holding means the route through L27 N3325 shows in sampled answers: restoring that one unit's natural values
  brings at least half of the lost Google answers back.
- 77.3 is a specificity control for 77.2: if copying the 10 next-strongest layer-27 candidates also brings Google back by
  more than 8 answers, 77.2 is reported as not specific to L27 N3325.
- 77.4 guards against an artefact of the copy: a restore should not produce more Google answers than makers only.

## Limits (written now)

- The makers stay zeroed throughout, as in Parts 73 to 76; nothing here describes the model with the makers intact.
- The twin row costs a second forward per answer and is never scored; a defect in the token sharing would make the two
  rows diverge, so the runner checks after generation that each pair's tokens are identical and refuses to score a
  pair that is not.
- One model, one revision, one prompt set; 192 held-out answers per condition, so differences of a few answers are noise.
- 48 new tokens; Part 75 found 830 of 960 answers stop at that cap, so a later Google mention is not counted.

## Amendment 1 (while reviewing the build, before any run)

The Method's claim that a custom logits processor sees scores already warped by temperature, top-k and top-p is wrong,
in transformers 5.13 (the GPU venv) and 5.20 (the test venv) alike: `_get_logits_processor` merges custom processors at
utils.py:1244 (5.13), before it appends the sampling warpers (from line 1262). The builder found this. The pair
processor therefore applies the sampling warpers itself, built from transformers' own classes in transformers' order
from the model's generation config with run_70's temperature 0.7 and top-p 0.95, so it samples exactly what
`model.generate` would have. A test compares those warpers with `generate`'s own processed scores step by step, and
`run.json` records the effective settings. The forced one-token scores then pass through `generate`'s warpers
unchanged. Conditions, readout and predictions are unchanged.

## Amendment 2 (after the build's cold review, before any run)

- 77.2 is read only if c0_makers exceeds c1_l24 by at least 8 held-out Google answers; with a smaller gap the halfway
  threshold means nothing, and 77.2 is reported NOT READ with the gap.
- When 77.2 holds and 77.3 fails, 77.2 is reported as "HOLDS, NOT SPECIFIC TO L27 N3325", as the reading rules above
  already say; the scorer now does this itself.
- Correction to the Limits: the pair-identity check cannot fail, because both rows of a pair are forced to the same
  token; it stays as a recorded field. The in-run evidence that the restore acted is the copy hook's: the runner counts
  its calls and the largest |answer - twin| on the copied units before and after the copy, refuses to continue if a
  batch that copies under L24 N3949 zeroing shows no difference to restore or any difference survives, and records the
  totals in run.json.
- The scorer writes `holds: null` for a prediction that is not read.
