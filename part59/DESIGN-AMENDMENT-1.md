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

