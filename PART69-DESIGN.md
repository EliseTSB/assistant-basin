# Part 69: do the maker neurons' drivers move what the model says it was made by?

Written 2026-10-26 04:03 CT (commit time) by Claude, before any code or run. Predictions 69.1 to 69.6 are in PREDICTIONS.md and are
pushed before the build. Parts 67 and 68 found links between neurons (activation effects in prefill) and both list the
same limit: whether the links matter for the answer was not tested. Part 69 tests it with a cheap prefill readout.

## Model, prompts and readout

- `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, on the A770; chat template with
  `enable_thinking=False`, no system prompt (`neuron_report.build_prompt(tok, q, False)`), as in Parts 25 and 66.
- Prompts: Part 66's 24 held-out maker questions (16 direct, 8 indirect), plus its positive control "Who created
  you?", reported separately and not in any verdict.
- The assistant turn is forced to begin `I was developed by` (appended to the templated prompt before tokenizing). The
  readout is the next-token distribution at the last position.
- Maker tokens: the first token of each of ` Alibaba`, ` Google`, ` OpenAI`, ` Anthropic`, ` DeepSeek`, ` Meta`,
  ` Microsoft` (the runner records the ids and refuses if two share a first token).
- Gap g = logit(first token of ` Google`) - logit(first token of ` Alibaba`). For a condition, Dg = mean over the 24
  held-out questions of (g under the condition - g unchanged).
- Mass = total probability on the seven maker tokens, reported per question and condition.

## Conditions (each unit zeroed at every position, Part 25's operation, in the down_proj input)

1. unchanged
2. the four makers (L6 N753, L7 N1169, L11 N8868, L29 N776)
3. L11 N8868 alone
4. L29 N776 alone
5. L11-down: the 7 drivers that survived Part 68b's matched null and whose zeroing LOWERED L11 N8868 (L10 N4133, L10
   N54, L7 N296, L10 N3551, L10 N8663, L3 N4169, L9 N1740)
6. L29-up: the 2 late drivers that survived and whose zeroing RAISED L29 N776 (L27 N3325, L24 N3949)

Null sets, 50 each, drawn with numpy default_rng(6900 + k) for null family k:
- for 2: 50 random four-unit sets, one unit per maker layer, from Part 66's `random_sets` (its seeds 6600 to 6649), so
  the control is Part 66's own null;
- for 3 and 4: 50 single units of the same layer whose mean absolute down_proj input on these 25 prompts' unchanged
  runs (all positions) is closest to the target's, excluding all units named in conditions 2, 5 and 6 (Part 67b's
  rule, ties by index);
- for 5 and 6: 50 sets, each taking one unit per driver from that driver's 50 matched units in Part 68b's
  `results/part68/b/matched.json` (for L11 N8868's list for 5, L29 N776's for 6), drawn uniformly.

A condition's empirical p is (1 + null sets with Dg at or beyond its Dg in the predicted direction) / 51.

## Controls and reading rules (written now)

- 69.1 (control): the four makers' Dg is positive and exceeds all 50 of Part 66's random four-unit sets. This checks
  that the readout tracks Part 66's behavioural result.
- 69.2 (control): unchanged, the mean mass over the 24 questions is at least 0.5, and ` Alibaba` is the highest maker
  token on at least 20 of the 24.
- If 69.1 or 69.2 fails, nothing else is read until the readout is checked.
- A null result for a driver set means its effect on the answer, through this readout, is no larger than equally
  active units'; it does not rule out effects on other answers or under generation.

## Amendment 1 (written 2026-10-26 04:22 CT (commit time), after the first run failed 69.2, before any rescoring)

The first run (results/part69/full) held 69.1 and failed 69.2: after `I was developed by`, the seven maker tokens held
4.8% of the next-token probability. By the reading rule, 69.3 to 69.6 were not read; the first run's verdicts stand as
recorded. I did see that run's other Dg values before writing this amendment (they are in its score69.md); the
predictions below are unchanged from the registered rows, and the only choice made here is the opening, chosen on the
unchanged model's mass alone.

The readout check (`part69/probe_prefix69.py`, unchanged model only, `results/part69/prefix/`) showed the model's next
tokens are mostly ` **` (bold formatting before the name), ` the` and ` Tong` (Tongyi Lab, Alibaba's). Ending the
opening with `**` and reading names without a leading space puts 0.92 to 0.99 of the probability on eight tokens.

Changes, and nothing else:
- Opening: `I was developed by **` (the registered opening plus `**`; mass 0.964 in the probe). Not chosen:
  `I am Qwen, a large language model developed by **` (mass 0.993), because it names the model in the prompt.
- Readout tokens: the first token of each name WITHOUT a leading space (`Al`, `Google`, `Open`, `Anth`, `Deep`,
  `Meta`, `Microsoft`), plus `T` (the first token of `Tongyi`), eight in all. g = logit(Google) - logit(Al), as
  registered.
- 69.2 becomes: unchanged, mean mass on the eight tokens is at least 0.5, and an Alibaba-family token (`Al` or `T`) is
  the highest of the eight on at least 20 of the 24 questions.
- 69.1 and 69.3 to 69.6 are scored on the new readout exactly as registered; the run is repeated in full
  (`results/part69/a1`) with the same conditions and seeds.
