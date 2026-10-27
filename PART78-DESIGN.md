# Part 78: do two pushed units compose into one identity? The alien neuron and the maker neurons together

Written 2026-10-27 by Claude, before any Part 78 code or run. Predictions 78.1 to 78.6 are in PREDICTIONS.md and are
pushed before the run.

## Why

On 2026-10-26 @moleculesrcool asked @Sauers_ whether modulating two neurons for orthogonal concepts makes a model compose
them into something defined ("Blue" + "Hedgehog" = "I am Sonic"). Sauers answered with a case: a GNOME-desktop neuron
("I'm a screen") plus a small-elf neuron gives a model that says it is a gnome
(x.com/Sauers_/status/2114767844831555653, read through FxTwitter 2026-10-27).

We have two units in one model, Qwen3.5-4B, that each change what the model says about itself:

- the four maker neurons (Part 66: L6 N753, L7 N1169, L11 N8868, L29 N776). Zeroed, the model names Google as its maker
  on held-out phrasings about a fifth of the time (Part 66, checked by the 96-sample rig test).
- the alien neuron, L22 N2742 (Parts 26.8, 59A, 62). Set to 1.5 times its Pile maximum on prompt and answer, 19 of 24
  answers are off-world; at 2 times, 24 of 24.

The question is whether these compose ("an alien made by Google") or whether one story overrides the other.

## What the existing data already shows (read before writing the predictions, and the reason for them)

Part 62's Qwen answers at the "both" phase, eight per question (`results/part62/clamp-qwen35-4b-x{1p5,2}-both-*`),
counted with keyword rules on 2026-10-27:

- At 1.5 times, 23 of 24 answers are off-world and 11 of 24 still name Qwen or Tongyi; 18 of 24 have both an off-world
  phrase and a "made/created/developed by" phrase ("I am Qwen3.5, a large language alien model developed by Tongyi Ali";
  "an alien intelligence created by the team at Xingchen", an invented maker). Google: 0 of 24.
- At 2 times, 24 of 24 are off-world, 2 of 24 name Qwen, 7 of 24 have a "made by" phrase. Google: 0 of 24.
- The size-matched control unit (L22 N8870) at the same doses: 0 off-world, and at 2 times Google in 11 of 24.

So the alien push does not lead to Google even though a generic push at 2 times does, and at 1.5 times the alien frame
and a maker claim already coexist in one answer. Those counts are n = 8 per question on three questions, so they set my
expectations and test nothing.

## Method

- Model `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, the A770, as in Part 66.
- Prompts: Part 66's 24 held-out maker questions (16 direct, 8 indirect) with 8 samples each, 192 per condition, and its
  6 guard questions with 4 samples each, 24 per condition. Part 66's chat rendering, sampling (temperature 0.7, top-p
  0.95) and per-row seeds, with at most 64 new tokens (Part 66 used 48; an answer that names both a self and a maker needs
  more room).
- Interventions, each applied at every prompt and answer position:
  - makers: Part 66's four units set to zero (its own hook).
  - alien: L22 N2742 set to dose times its Part 26 Pile maximum (Part 62's hook and maximum).
  - control: L22 N8870 set to dose times its own Pile maximum (Part 62's control unit).
- Conditions:

| id | makers zeroed | L22 push |
|---|---|---|
| c0_unchanged | no | none |
| c1_makers | yes | none |
| c2_alien15 | no | N2742 at 1.5 |
| c3_makers_alien15 | yes | N2742 at 1.5 |
| c4_alien20 | no | N2742 at 2.0 |
| c5_makers_alien20 | yes | N2742 at 2.0 |
| c6_makers_control20 | yes | N8870 at 2.0 |

- Total: 7 conditions times 216 answers = 1,512 answers of at most 64 tokens, in one pass on the A770.

## Scoring (fixed now)

Keyword rules on the answer text, case-insensitive:

- Google: Part 66's rule (`Google|Gemini|DeepMind`).
- Off-world: Part 62's rule (`alien|extraterrestrial|planet|galax|\bstar\b|space|interstellar|species`).
- Qwen: `Qwen|Alibaba|Tongyi`.
- Maker phrase: `(created|made|developed|built|trained|designed) by`.
- Composed: Google and off-world in the same answer.
- Other maker: a maker phrase and none of Part 66's known maker families in the answer.
- Guards: Part 66's guard keywords.

Every held-out count is reported per condition with a per-question breakdown. A Haiku reading of a random 48 answers per
condition will be reported beside the keyword counts to show where the keywords miss, but the predictions are read on
the keyword counts.

## Reading rule

78.1 is the control. If it fails, 78.2 to 78.6 are reported as NOT READ.

## Limits known in advance

- Keyword scoring: "space" and "planet" can appear in ordinary answers (c0 measures that rate), and an invented maker is
  only detected through a "made by" phrase.
- One model, one dose pair, bf16 where Part 62 used fp32; the alien threshold may sit slightly differently in bf16, which
  is why 2.0 is included and why 78.1 gates on the off-world rates.
- Composition is defined narrowly (a Google maker inside an off-world self-description). A model that blends the two in
  a way the keywords miss will be undercounted; the Haiku reading is there to show how much.

## Amendment 1 (2026-10-27, before any code or run)

The design said the control unit L22 N8870 is set to "dose times its own Pile maximum". That was my error: Part 62's
control, the one behind the counts quoted above, has no Pile maximum of its own in the records. Part 62 sizes it so its
push writes the same norm into the residual stream as N2742's: set value = dose times N2742's Pile maximum (19.125, from
`results/part26/source.json` and `source.npz`) times the ratio of the down_proj column norms, N2742's over N8870's
(`part62/run62.py`, `cell_parameters` and `_column_norm`). Part 78 uses that rule for c6_makers_control20. The Part 78
build worker found the error and stopped before writing code.
