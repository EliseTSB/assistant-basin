# Part 59: does the alien neuron steer the setting of a story, and which neurons move with it in both directions? (registered 2026-10-22 14:00 CT, before any run)

Elise, 2026-10-22 13:5x CT: "Based on the 'Earthling' response when olmo was pushed negative I wonder if it distinguishes the alien vs earth side of fictional stories, and if the qwen one is similar. I want to look at what other neurons it interacts with, I wouldn't be surprised if there's a number of neurons that respond similarly to it being pushed either positive or negative, i.e. activation either way pushes it toward a sci-fi setting."

Models and units: Qwen3.5-4B L22:2742 (the alien neuron) and OLMo-2 1B L10:6978 (its partial twin), with the same dose scale as Part 54 (multiples of the neuron's largest activation on the alien frames).

## 59A: story setting under the push
- Prompts (fixed, 12): "Write the first paragraph of a story about a ..." with 12 neutral subjects (a baker, a lighthouse keeper, a surveyor, a student on the first day, ...). Doses: -2, -1, 0, +1, +2; 8 samples per prompt-dose cell at 60 new tokens; 3 random same-layer controls at -2 and +2.
- Blind reader (blind_origin.py style, condition hidden) labels each paragraph SETTING: SCIFI (space, aliens, technology of the future), EARTH-REAL (ordinary contemporary or historical Earth), FANTASY (magic, myth), or OTHER, plus BROKEN.

## 59B: symmetric responders
- 1,000 Pile documents (the Part 26 corpus), clamp the unit at -1 and +1, record every MLP neuron's mean activation change in later layers against the unclamped run; same for 3 random same-layer neurons.
- A neuron is a SYMMETRIC responder if its change has the same sign at -1 and +1 and |change| exceeds the 99th percentile of the same quantity under the random controls at both doses. It is an ODD responder if the signs are opposite (the usual dose-response).

## Predictions (Claude, before any run)
| id | prediction |
|---|---|
| 59a | Qwen: SCIFI share is at most 1 of 96 at dose 0, and at least 12 of 96 at +2. |
| 59b | Qwen at -2: SCIFI share is no higher than at dose 0 (the down side does not push toward sci-fi). |
| 59c | OLMo at -2: EARTH-REAL share is not higher than at dose 0 (the "Earthling" answers were about the model itself, not story settings). |
| 59d | No random control reaches 10% SCIFI at -2 or +2. |
| 59e | Garbled paragraphs are at most 15% at every dose from -1 to +1. |
| 59f | In Qwen, at least 20 later neurons are SYMMETRIC responders by the rule above (the neuron has big non-linear effects downstream). |
| 59g | Fewer than 5 neurons are symmetric responders in the random controls. |
| 59h | The strongest symmetric responders, read by their top vocabulary projection and top windows, are not predominantly sci-fi words (at most 3 of the top 10 by hand reading); the symmetry is mostly generic disruption. |

Prior art asked of Muse before the run (council muse-neuron-atlas-watch, topic to be posted with this file). Caveat adopted from Part 50: a clean setting shift still needs the unrelated-question control, which 59A includes through the random-neuron arms.

# Amendment 1 — 2026-10-22 (before any run)

This amendment records implementation choices for Part 59 before data collection. It does not change any prediction text.

## Pooled symmetric-responder threshold

At each dose (-1 and +1), compute one threshold: the 99th percentile of absolute activation changes pooled over all later-layer neurons from all applicable random controls. The target is compared with a threshold pooled over all three controls. For the count assigned to control *k*, use leave-one-control-out pooling over the other two controls; that control is never used to set its own threshold. A neuron is symmetric only when its changes at both doses are nonzero, have the same sign, and each absolute change is strictly above that dose's pooled threshold. Percentiles use NumPy's linear interpolation.

## Later layers and strongest responder

“Later layers” means layers strictly after the clamped layer. For the 59h strongest-responder ranking, rank symmetric units by `min(|change at -1|, |change at +1|)` in descending order.

## Dose-scale correction

The Part 54 dose scale is the unit's largest activation on the Part 26 Pile-corpus record. For Qwen L22:2742, that value is 19.125. The alien-frame maximum is reported only as a ratio; it is not the dose-scale maximum. This corrects the design's phrase “largest activation on the alien frames.”

No other design wording or prediction text is amended here.

## What the pooled rule can and cannot show (added 2026-10-22 18:1x, before any run, after the worker's null simulation)

Under independent doses, a neuron passes the pooled rule with probability 0.01 x 0.01 x 0.5 = 0.00005. Over the 82,944 later-layer neurons that is about 4.1 symmetric neurons expected in any control by chance alone; the simulation gave target 2 and controls 6, 3 and 4. So prediction 59g ("fewer than 5 symmetric responders in the random controls") fails about 40% of the time under pure noise, and a failed 59g is not evidence of anything. Under a dose correlation of 0.5 the simulation gave target 72 and controls 108, 93 and 75, so 59f ("at least 20") is also met by controls. Both predictions will be scored exactly as written, and the table will say beside each that the null alone can pass or fail it. The informative comparison, registered here before the data, is the target's symmetric count against the three control counts (the target must exceed the largest control count to count as a result), computed on the same pooled-leave-one-out rule, with the ranked list read by hand for 59h. The earlier acceptance gate (control under 5) was mine and was mis-specified; it is dropped.

