# Part 128: in Qwen3-8B, what a persona neuron fires on in ordinary text matches its persona when the persona reaches across questions, and rarely when it shows once

Part 91 sorted 40 persona neurons by how far the persona reaches: carriers show it on at least 3 of 6 questions, once neurons
on exactly 1. That sort used only pushed answers. Neuron Lens now reads every pole of this model in ordinary text. On the fire
side, which the push does not write, both readers score 6 of 11 carriers as a match (2 or more of 3: the concept, or a thing
strongly associated with it), against 0 of 16 once neurons for DeepSeek and 1 of 16 for Luna. Counting the write side too it is
9 of 11 against 1 of 16 (Luna 8 of 11). Scope: Qwen3-8B, 40 neurons, the library's dose, two blind readers scoring catalog cards.

| Neuron | Persona when pushed | Reach | What the catalog says it fires on |
| --- | --- | --- | --- |
| L9 n12264 − | Digestive Stomach | carrier | fiber, digestion and eating |
| L10 n11694 − | Fated Pawn | carrier | fate, luck, destiny and chance words |
| L11 n6060 − | 5-Star Rating System | carrier | rating, star and grading words |
| L21 n7905 − | Survival Group Member | carrier | survival and subsistence |
| L26 n10281 − | Journey Traveler | carrier | journey, challenge and companionship words |
| L28 n122 − | Cargo Container | carrier | payload, cargo and load words |
| L5 n2568 − | Hotel | once | Traditional Chinese characters and their usage forms |
| L7 n1424 + | Teacher | once | the period in abbreviations like e.g. and U.S. |
| L9 n8930 + | Human Reactor Control Room Operator | once | quiet speech: whispering, muttering |
| L10 n8485 + | Humble Servant of the Divine Lord | once | British spellings ending -our, -ise, -tre |
| L23 n11249 − | Chinese Student | once | "last", "final" and ordinal words |
| L30 n8111 + | Qianpu Heiress | once | screen size and seating words |

Firing descriptions are the catalog's labels, shortened. Six carriers and six once neurons are shown; every unit is in
`results/part128/units128.json` and `labels-ds.jsonl`.

## Predictions

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 128.1 | Claude (medium confidence): At least 12 of 40 pushed poles clearly match their persona on some side, and at most 4 of 40 mismatch rows do. | DeepSeek 14 and 0; Luna 13 and 2 | held |
| 128.2 | Claude (medium confidence): The carrier rate on either side is at least 20 points above the once rate. | DeepSeek 9 of 11 against 1 of 16; Luna 8 of 11 against 1 of 16 | held |
| 128.3 | Claude (low confidence): On the fire side, at most 4 of 16 once poles and at least 5 of 11 carrier poles clearly match. | DeepSeek 0 of 16 and 6 of 11; Luna 1 of 16 and 6 of 11 | held |
| 128.4 | Claude (medium confidence): Pushed poles match their persona at least twice as often as opposite poles do. | DeepSeek 14 against 6 (2.3 times); Luna 13 against 7 (1.9 times) | reader-dependent, not counted as held |
| 128.5 | Claude (high confidence): At most 4 of 40 control rows match the paired persona. | DeepSeek 1; Luna 0 | held |
| 128.6 | Claude (medium confidence): The median of abs(c) / act_top is at least 5, and at least 30 of 40 pushes are 3 times the pole's top firing or more. | median 7.8; 36 of 40 | held |
| 128.7 | Claude (low confidence): The count of dark pushed poles is within 6 of the count expected from the catalog's band rates. | 23 against 23.9 | held |

The gate passed: both readers scored all 6 known steering-verified neurons at 2 or more on some side and 1 of those 6 cards
against the wrong name (the Jesus card against "gods", a fair call), the mismatch arm was 0 of 40 for DeepSeek and 2 of 40 for Luna,
agreement on the either-side call was kappa 0.85, and no row was missing. The rule that withdraws the headline if the mismatch arm
reaches the pushed arm was checked and does not apply.

## Is the gap only on the write side, which steering already predicts?

No. Neuron Lens shows a write read predicts steering, so a carrier matching on the write side is weak evidence. The fire side
comes from activations and token drivers over ordinary text, with no push. It is not fully independent of the write side: for the
six carriers that match on the fire side, the fire and write labels name one concept.

| Reader | Carrier, fire side | Once, fire side | Carrier, either side | Once, either side |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash | 6 of 11 | 0 of 16 | 9 of 11 | 1 of 16 |
| Luna | 6 of 11 | 1 of 16 | 8 of 11 | 1 of 16 |

A one-sided Fisher test on the DeepSeek either-side counts gives p = 0.0001 (not registered, shown for scale). From
`results/part128/score128.json`, made by `part128/score128.py`.

## How large is a push compared with a natural firing?

The push was a median of 7.8 times the pole's top firing in ordinary text, and 36 of 40 were at least 3 times. Carriers (median
6.6) and once neurons (median 7.8) were pushed by similar multiples; dose was not tested here. `act_top` comes from ordinary
text, so a chat persona prompt may drive a pole harder and shrink the ratio.

## Are persona neurons easier to read than other neurons?

No evident enrichment. 23 of 40 pushed poles are dark in the catalog, meaning no token-level test passes, against 23.9 expected
from the catalog's band rates (L0-8 29%, L9-12 72%, L13-20 82%, L21-27 59%, L28-35 52%). A tolerance of 6 cannot see a small shift.

## What about the opposite pole?

Across all 40, DeepSeek scores the pushed fire side as a match on 9 and the opposite pole's on 6 (Luna 10 and 7). For 3 carriers
the opposite pole's firing matches the persona too, for example L28 n122, where both poles sit near cargo words. So at least some
of these concepts belong to the neuron, not to one pole, which is why 128.4 did not clear its bar on both readers.

## What do once neurons do?

Mostly something unrelated to their persona, by these readers: 15 of 16 match on neither side for DeepSeek. Of their 16 names, 7
contain Human or Student, and 5 of the other 9 are people or roles. One reading to test: a large push knocks the model off its
usual answer to "Who are you?", and it fills that slot with something nearby. Part 128 shows only that catalog evidence does not
match the persona, not what a once neuron's job is. Neuron Lens §10 found that labels from steering one prompt are the ones
activations most often contradict.

## Method

Design: `PART128-DESIGN.md`. Catalog: stdlogits/neuron-lens at ec394f5bb21dee5e67b506f9748950a3c5b08b88, Qwen3-8B. A card holds
layer-contrastive R and S reads, exact token drivers, five real firings and the catalog's label for the fire side, and R and
S reads and the label for the write side. Four arms per neuron: pushed pole against its persona, the same card against another
neuron's persona, the opposite pole against its persona, and Part 91's same-layer control at the same sign against the paired
persona. DeepSeek V4.1 Flash and Luna scored each side 0 to 3, blind to arm; 2 or more means the concept in any language or a thing
strongly associated with it, the same bar as Neuron Lens §10. Word overlap counts: for four of the six carriers that match on the
fire side, the persona's own word is in the fire label or drivers. No model ran here, so the hook, broken-answer and dose-ladder
controls do not apply, and one control per neuron was used because Part 91 made one. For scale, Neuron Lens §10 reports 27% of
library rows matching the write read, 22% the input read and 34% both; here 14 of 40 pushed poles match (DeepSeek). Part 124 found
10 of 18 concept units and 4 of 21 slot units reading their phrase in OLMo 2 1B. The design sized for 30% against 5% at 40 per arm.

## Limits

40 neurons in one model, with 11 and 16 in the class comparison. Classes come from one reader's labels on six answers, and the
13 neurons between them are out of the comparison. Part 91's pushes and these cards measure different things: a neuron's
ordinary firing can match a persona without causing it. The catalog's labels are Haiku's, written from an earlier build of its
tables, and are on the cards the readers scored. Readers judged word lists, not causes. Part 132 repeats this on 189 fresh neurons.

## Files

- `results/part128/cards128.jsonl`: every card and gate row; made by `part128/join128.py`.
- `results/part128/units128.json`: each neuron's class, push, activation and darkness facts.
- `results/part128/labels-ds.jsonl`, `labels-luna.jsonl`: reader scores; made by `part128/label128.py`.
- `results/part128/score128.json`: counts, agreement and gates; made by `part128/score128.py`.

Cold check: Claude Haiku 5.5 with no prior context, 2026-10-30, every number reproduced; 18 findings on wording, omissions and independence, all folded except the answer-polarity split, which does not apply to reader scores of catalog cards.
