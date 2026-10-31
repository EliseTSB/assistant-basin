# Part 128: in ordinary text, do Qwen3-8B's persona neurons do what their persona says?

Procedure: v1
Registered 2026-10-30 19:20 CT by Claude (Opus 5.5), before any catalog row for these neurons was read. Elise,
2026-10-30 19:12 CT, Discord message 1565881287409934448: "read those twitter threads and explore the repo and
https://neuron-lens.com/, and keep the research going. Lots to reassess now that there's new tools!" The commit that adds
this file is the registration.

## Question

Part 91's 40 persona neurons were chosen because a push makes Qwen3-8B answer "Who are you?" as a persona. Neuron Lens
(stdlogits/neuron-lens) now reads every pole of this model in ordinary text. Does the pushed pole's firing or write match
its persona, does the opposite pole, and does the answer differ between neurons that carry the persona across questions
and those that show it once? This is the first test of the concept and slot sort against real firing in the model the
catalog covers, and it asks how far above natural firing Part 91's pushes sit.

## Prior art and existing data

- Closest published work: Neuron Lens findings §10 judged 360 rows of Josh Fonseca Rivera's library against reads: 27% of
  rows match the write read, 22% the input read and 34% both, weighted by the library's depth mix, and labels from
  steering one prompt are the ones activations most often contradict. Findings §4 shows that before L21 a firing's
  effect depends on its context, so steering is not what a firing does. This part differs in three ways: it sorts neurons
  by Part 91's six-question battery, reads the opposite pole, and measures each push against the pole's natural firing.
- Our own data that already bears on it: results/part91 (the 40 neurons, six answers per setting, Haiku labels); Part 124
  found in OLMo 2 1B that 10 of 18 concept units and 4 of 21 slot units rank their own phrase in their top three.

## Method

Catalog: stdlogits/neuron-lens at revision ec394f5bb21dee5e67b506f9748950a3c5b08b88, Qwen3-8B tables reads, acts, drivers,
legible and labels. part128/join128.py builds a card for a pole: fire side (layer-contrastive R and S reads of its read
direction, exact token drivers, five real firings, the catalog's firing label) and write side (layer-contrastive R and S
reads of its write, the catalog's write label). Four arms per neuron: pushed pole against its persona; the same card
against another neuron's persona (a derangement, seed 128); the opposite pole against its persona; Part 91's same-layer
control at the same sign against the paired persona. No model runs. Readers score each side 0 to 3 (part128/label128.py);
2 or more is a clear match, the same bar as Neuron Lens §10. Classes come from stored Part 91 labels: carrier is at least
3 of 6 chat answers showing the persona (n = 11), once is exactly 1 of 6 (n = 16), the 13 others stay out of class
comparisons. Dose: |c| / act_top, where c is the coefficient Part 91 added on W_down[:, n] and act_top the pole's median
top firing in the same activation units.

## Controls

Applied from STANDARD-CONTROLS.md: both poles; matched random controls (Part 91's same-layer neurons at the same sign);
a mismatched name on the same card. Not used: do-nothing hook, broken-answer count and dose ladder, because no model runs.
Readers are blind to arm, unit and class.

Gate: both readers must score at least 5 of the 6 KNOWN steering-verified neurons in Neuron Lens' known.py at 2 or more on
some side, and at most 1 of those 6 cards against the wrong name; the primary reader's mismatch arm must be at most 6 of
40; agreement between readers on the either-side call must reach kappa 0.6; no row may be missing. A failed gate stops
the reading until it is understood. It could fail if the cards carry too little or the readers are generous.

## Size

- Smallest effect worth knowing: 30% against 5% across arms, which 40 units per arm detects; the class comparison, 11
  against 16, can only show a large gap, so it is reported with its counts.
- Total: 172 reader calls per reader, 344 in all, about 4 minutes.
- Pilot: not needed: under 1,000 generations and no GPU.

## Optimization

- Bound resource: API
- Measured: n/a: under 5 minutes of compute, reader calls run in a thread pool.
- Levers: batch API declined, the two readers are not on it; cheap readers taken, DeepSeek V4.1 Flash and Luna.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 128.1 | At least 12 of 40 pushed poles clearly match their persona on some side, and at most 4 of 40 mismatch rows do. | medium | Claude |
| 128.2 | The carrier rate on either side is at least 20 points above the once rate. | medium | Claude |
| 128.3 | On the fire side, at most 4 of 16 once poles and at least 5 of 11 carrier poles clearly match. | low | Claude |
| 128.4 | Pushed poles match their persona at least twice as often as opposite poles do. | medium | Claude |
| 128.5 | At most 4 of 40 control rows match the paired persona. | high | Claude |
| 128.6 | The median of abs(c) / act_top is at least 5, and at least 30 of 40 pushes are 3 times the pole's top firing or more. | medium | Claude |
| 128.7 | The count of dark pushed poles is within 6 of the count expected from the catalog's band rates. | low | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 128 (registered 2026-10-30, PART128-DESIGN.md): in ordinary text, do Qwen3-8B's persona neurons do what their persona says?".

## Reading rules

part128/score128.py counts everything from labels-ds.jsonl (primary) and labels-luna.jsonl. A prediction that holds on
DeepSeek and fails on Luna, or the reverse, is reported as reader-dependent and does not count as held. Error rows are
retried; a row still missing fails the gate. Band rates for 128.7 are the catalog's: L0-8 29%, L9-12 72%, L13-20 82%,
L21-27 59%, L28-35 52%. If the mismatch arm reaches the pushed arm's rate, the headline reading is withdrawn: the cards
would then be matching anything.

## Limits

40 neurons, one model. The class comparison is underpowered and the classes come from one reader's labels on six answers.
The catalog's labels were written by Claude Haiku from an earlier build of its tables. act_top comes from bf16 passes over
ordinary text, where a chat persona prompt may drive a pole harder. Readers judge word lists and contexts, not causes.
