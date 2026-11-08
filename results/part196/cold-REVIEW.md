SHIP-WITH-FIXES

# Part 196 cold review

## Numeric audit

The displayed balanced-accuracy means and intervals, AUC means and intervals, per-speaker balanced accuracies, per-style recalls, headline paired differences, and selected-style duration profile match `results.json` at the README's displayed precision. The prediction table values also match the JSON. Gate counts and diagnostic values checked against the JSON; the 35.3-minute recording claim agrees with the 2,115.43-second recorder log (35.26 minutes), and the 23.2-minute estimate is the registered estimate, not a measured time.

Two claims about F's selected single units are incorrect:

- README says F's best units “sit in layers 0 to 3.” The JSON includes ex02's laughing unit at layer 4, neuron 153 (flattened index 8345). The selected-unit range therefore reaches layer 4.
- README says L0 N675 is best for both enunciated and whisper in every fold. N675 is selected for enunciated in all four folds, but for whisper only in ex01. Whisper selects L0 N2041 in ex02, L0 N46 in ex03, and L0 N2041 in ex04. (ex01 selects L0 N675.)

## Predictions and registered interpretation

196.1 held (A 15.1%); 196.2 did not hold (D 25.8%). For 196.3, D−A's interval is wholly above zero (10.7 points; 9.2–12.3), so the branch applies; B closes 2.1 points, about 20% of that gap, below the required half. Its failed verdict is correct, not not-applicable. 196.4 is correctly “gap supported”: D−B is 8.6 points (7.0–10.4), and D's lower bound 24.5% is above chance 14.3%, so the design permits the stated reading that the final embedding contains style information this particular calibrated verdict does not use. 196.5 failed: B56−D is −4.1 points (−5.6 to −2.4); the non-inferiority lower bound is −5.6, not above the registered −3-point threshold, so no decision-rule attribution is allowed. 196.6 failed (A's top recall is default; B and D's are laughing, not whisper). 196.7 held (H 51.8%); 196.8 held (T 14.7%).

## Claims to narrow before publication

- The conclusion “the information is present early and is mostly discarded on the way to the output,” the title's “mostly drops it,” and “the early layers do [read it]” imply model use or representational discard. E shows linear readability from mean-pooled layer inputs; the layer curve is inner validation on the three training speakers, and only selected layer 0 was tested on held-out speakers. The design explicitly says this does not establish that the model uses the information. Recast as probe/readout results, and label E-versus-D a descriptive comparison (no paired interval was registered).
- “An edit ... would have to carry layer-0 information forward, not re-weight the final embedding,” “an edit ... has little style to work with,” and “here it does not” present an edit strategy/outcome that this part did not test. The registered rules say these readout findings alone are not a case for an edit. Present such a strategy only as an untested follow-up hypothesis, or remove it.
- “The words alone carry nothing here” overstates T=14.7% (95% interval 14.2–15.2; chance 14.3%). The design says low T bounds one transcript classifier and cannot rule out content routes. “Near chance in this transcript-only readout” is supported and consistent with the later caveat.
- “The 56 label cosines ... read less style than the embedding” risks assigning the B56−D gap to label-space information after correctly saying the registered non-inferiority condition failed. The point/paired accuracy comparison may be reported, but do not use it to attribute the mechanism.
- The E-over-H statement is explicitly called descriptive and the Limits section correctly says E-versus-H and E-versus-D have no registered paired interval. Keep that qualification beside any such comparison. The duration profile supports differing clip lengths; clip length alone is not tempo. H's registered seconds-per-word feature can use speaking rate, so say tempo may contribute rather than that duration establishes it. The low-T content caveat is otherwise present in Limits.

## Figure and amendments

The existing `layers.png` matches `fig_layers.py`: four faint fold-specific inner-LOSO curves, their mean, held-out four-fold E/H/D/T horizontal lines, and chance. The caption describes those elements and the JSON source correctly. I attempted a scratch-copy rerun; it could not execute because `python3 -I` has no Matplotlib installed. I inspected the existing image and compared it with the plotting code and JSON, so this limits reproducibility checking but revealed no figure/caption mismatch.

“Changes after registration” matches Amendments 1–3: transcript prompt `document`; metadata identity match on id/speaker/style for reused IDs; and fixing the one-array unpack plus a synthetic end-to-end analyzer test. PREDICTIONS.md's Part 196 result/verdict cells remain blank, as noted in the brief; the README verdicts themselves were checked above.

## Recommendation

Fix the two F unit claims and narrow the interpretation/action/content language before publication. No decision or authority is needed to complete this review.
