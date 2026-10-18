# Part 17b: the natural residual against a cross-fitted Qwen3-1.7B text predictor (OLMo 2 1B Instruct)

Run on the A770, 2026-10-18 09:13-11:34 CT. Code textfit17b.py and residual17.py at ef3bcab (merged 3ad4371), every
reader step with `--run-tag p17b --fit-dir results\part17b`. Design PART17B-DESIGN.md; predictions pushed at bf37549
before any code. Scored 2026-10-18 11:36 CT.

**Run choice, recorded:** the text predictor was fitted at 1 epoch, batch size 8 (about 6,000 steps over 5 folds),
against the builder's default 3 epochs, batch 2, which was estimated at 8 to 24 GPU hours. Out-of-fold R² below.

| | G1 (identity) | GL (last layer) |
|---|---|---|
| Out-of-fold R², all 12,000 passages | **0.760** (Part 17 ridge: 0.622 TEST) | 0.523 |
| Out-of-fold R², TEST | 0.705 | 0.508 |

| arm | TEST accuracy on sign(residual) | majority rate |
|---|---|---|
| gate a: Qwen3-0.6B text-only reader, G1 | 52.65% (1053/2000) | 51.05% |
| gate b: OLMo self-reader, GL (cannot reach the answer) | **59.25%** (1185/2000) | 50.85% |
| OLMo self-reader, G1 | 57.0% (1140/2000) | 51.05% |

- **Gate a passed, just.** The text-only reader is at 52.65%, under the 53% bar.
- **Gate b failed.** OLMo's reader recovers 59.25% for GL, a group whose activation cannot reach the answer token,
  so that accuracy comes from the text. The stronger text predictor explains only half of GL's variance (R² 0.52),
  and OLMo reads the rest from the words better than a separate Qwen reader can. With gate b failed, the G1 result
  cannot be read as introspection (the design's rule).
- **Self vs text-only, paired on the same passages:** self-only right 464, text-only right 377, z 3.0, p 0.0027;
  same prediction on 58.0% of passages. OLMo's reader beats the separate text reader, but by less than it does on GL,
  which it cannot read from its own activity at all.
- **Transplant, text held fixed:** with G1 set to its 80th rather than 20th FIT-residual value, P(higher) is larger
  on only 2.8% of TEST passages; the means are 0.4274 and 0.4261. Part 17's reader moved the right way on 99.65%.
  This reader does not use G1's activation.
- **LoRA drift:** r 0.973 (G1 reader), 0.944 (GL reader) with the base capture.

**Reading.** A much stronger text predictor removed most of the text signal for the identity group (gate a now sits
near chance), but not for the last-layer group, and OLMo still reads that group's residual from the words. The G1
self-reader's 57% is below its own text-only ceiling on GL and its transplant shows no dependence on G1, so there is
still no sign that OLMo reads its own natural activity. What would separate the two: a text predictor that brings GL
under 53% too (more epochs, OLMo itself as the predictor, or per-token rather than mean features).

## Files

fit.json, fit.npz (cross-fitted predictions and residual thresholds), train-*-p17b.json and eval-*-p17b.json (per
passage), transplant-G1-p17b.json. Adapters (.pt) are not committed.
