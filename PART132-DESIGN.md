# Part 132: on 189 fresh neurons, does reach across questions still sort Qwen3-8B's persona neurons by what they do in ordinary text?

Procedure: v1
Registered 2026-10-30 19:58 CT by Claude (Opus 5.5), before any answers for these neurons and with no catalog row of them read. Follows
Part 128, where carriers matched their persona on 9 of 11 and once neurons on 1 of 16. That sort was exploratory and the groups
small. The commit that adds this file is the registration.

## Question

Does the Part 128 gap hold on fresh neurons, across layer bands, in numbers that can stand up?

## Prior art and existing data

- Closest published work: Neuron Lens §10 (library rows against reads, no sort by reach); Hergenroeder 2026 (207 Qwen3-8B
  personas across questions and turns, with a judge).
- Our own data: Parts 91 and 128, Parts 92 and 112 (the sort on OLMo and Qwen3.5-4B).

## Method

part132/select132.py draws 25 neurons per band of four layers from L2-33 (14 from L30-33, all that are eligible), 189 in all, with
Part 91's eligibility, leaving out Part 91's 40, seed 132, one pole per neuron. Each has a same-layer control at the same sign and
alpha. part91/push91.py runs unchanged: Qwen3-8B at b968826d in its chat template with thinking off, Josh's recipe, greedy, 160
tokens, six questions. part132/label132answers.py labels answers with Part 91's prompt on DeepSeek V4.1 Flash. Carrier and once
are defined as in Part 128: at least 3 of 6 chat answers show the persona, exactly 1 of 6. part132/join132.py builds the same four
card arms from the pinned catalog, and part132/label132cards.py scores them with Part 128's prompt, DeepSeek and Luna.

## Controls

As Part 128: both poles, same-layer controls, a mismatched name on the same card. Broken answers are counted through the reader's
coherent field, and a persona in an incoherent answer does not count toward reach. Not used: do-nothing hook, because push91.py
already passed it in Part 91; dose ladder, because the library names one dose.

Gate: as Part 128 (both readers at least 5 of 6 known neurons and at most 1 of 6 wrong names; mismatch arm at most 15%; kappa at
least 0.6), plus at least 20 neurons in each class. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a 25-point gap between classes, which needs about 50 neurons per class; Part 91's rates give about
  50 carriers and 75 once neurons from 189.
- Total: 2,274 answers, about 11 minutes on one RTX 3090 at Part 91's 3.5 answers/s (189 × 2 × 6 + 6). Then 2,268 answer labels, and
  756 card rows per reader plus 12 gate rows.
- Pilot: the first 4 neurons, 54 answers, timed on the Part 129 machine before the main stage. Stopping rule: under 1.5 answers/s
  the run stops and the batch is revisited. The same script ran 1,458 answers in about 7 minutes in Part 91.

## Optimization

- Bound resource: GPU, briefly; then API
- Measured: 1,458 answers in about 7 minutes in Part 91, the same script, model and GPU (3.5 answers/s).
- Levers: batch size kept from Part 91 for comparability; reuse of the Part 129 machine and its model download taken; cheap readers taken.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 132.1 | On either side, carriers clearly match their persona at a rate at least 30 points above once neurons. | high | Claude |
| 132.2 | On the fire side, at least 30% of carriers and at most 10% of once neurons clearly match. | medium | Claude |
| 132.3 | The either-side gap is at least 15 points in each of three bands, L2-13, L14-23 and L24-33. | medium | Claude |
| 132.4 | The mismatch arm is at most 5% in both classes. | medium | Claude |
| 132.5 | Names matching the regex in part132/score132.py (human, person, student, teacher, assistant and similar roles) are at least twice as common among once neurons as among carriers. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 132 (registered 2026-10-30, PART132-DESIGN.md): on 189 fresh neurons, does reach across questions still sort Qwen3-8B's persona neurons by what they do in ordinary text?".

Amendment 1, 2026-10-30 20:09 CT, before any answers existed: three predictions added from an exploratory look at Part 128's catalog measures for its 40 neurons (carrier and once medians: writes_conf 0.70 and 0.25, cur_mode 0.17 and 0.33; dark 7 of 11 and 5 of 16). part132/score132.py reads labels.writes_conf, legible.cur_mode and legible.dark for each pushed pole.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 132.6 | The median catalog writes_conf is at least 0.15 higher among carriers than among once neurons. | medium | Claude |
| 132.7 | The median catalog cur_mode is at least 1.3 times higher among once neurons than among carriers. | medium | Claude |
| 132.8 | The dark share is at least 10 points higher among carriers than among once neurons. | low | Claude |

Copied into PREDICTIONS.md under Part 132.

## Reading rules

part132/score132.py scores on DeepSeek; a verdict that differs on Luna is reported as reader-dependent and is not counted as held.
Error rows are retried, and a row still missing is counted and left out. If the gap falls under 10 points, Part 128's reading is
withdrawn on the front page.

## Limits

One model and one dose per neuron. Reach comes from six questions read by one reader. Names and cards were written by Haiku.
Reach is defined from pushed answers, so a class difference could reflect how the model handles a push, not the neuron alone.
