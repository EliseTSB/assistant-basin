# Part 17 design: can the model report the part of its own neuron activity the text does not predict?

Written 2026-10-18 00:28 CT, before any code or run. Motivated by Chorus's 10-18 Deep Research report (filed in the Journal as
notes/reports/research/2026-10-18-chorus-deep-research-natural-state-introspection.md) and by Singh, Linzen and Ravfogel
2026: an input-only classifier can match a model's "self-report" whenever the reported state is predictable from the
text. Parts 12h-12k read clamped states; 12k showed the signal reaches the output at natural strength when the text is
held fixed. Part 17 asks the question with nothing clamped.

## Target

For a neuron group G in OLMo 2 1B Instruct (G1 = the 11 identity units used since 12g; G2-G4 = the layer-matched random
groups from selfreport_multi.make_groups; GL = an 11-unit group in the last layer, which nothing later in the text can
read), the natural activation a_G(x) is the mean post-activation over the message's tokens, captured with no hook
writing anything.

The input-only prediction â_G(x) comes from a DIFFERENT model that sees only the text: Qwen3-0.6B's mean final hidden
state, ridge-regressed onto a_G on a FIT split. The residual r_G(x) = a_G(x) - â_G(x). The label is its sign, balanced
by thresholding at the FIT-split median residual.

## Arms

1. **Natural residual (the decisive arm).** LoRA on OLMo 2 1B Instruct, trained on TRAIN-split messages to answer
   "Was group <name> of your neurons more or less active in your last message than its words would suggest? Answer
   higher or lower." Nothing is clamped in training or evaluation. Evaluate on TEST messages never seen.
2. **Natural-value transplant (an intermediate with text held fixed).** For a TEST message, replay it twice with G set
   to a naturally observed value from another message (the 20th and the 80th percentile of the group's FIT-split
   distribution). Score the paired readout: did P(higher) rise in the 80th-percentile replay relative to the 20th?
3. **Input-only control.** The same question and labels, LoRA on Qwen3-0.6B, which has no access to OLMo's activations.
   The self-reader has to beat it.
4. **Unreadable control.** Arm 1 for GL. Must sit at chance.
5. **Shuffled labels.** Arm 1 with TRAIN labels permuted. Must sit at chance.

Data: about 12,000 WikiText-103 passages of 40-120 tokens, split FIT 4,000 / TRAIN 6,000 / TEST 2,000 by passage hash.

## Predictions (pushed before any code exists)

| # | Prediction | Result | Notes |
|---|---|---|---|
| 17.1 | The input-only ridge explains at least 30% of G1's natural-activation variance on TEST (R²). | open | |
| 17.2 | The input-only control (arm 3) gets at most 56% on sign(r_G1) on TEST. | open | |
| 17.3 | The self-reader (arm 1) gets at most 58% on sign(r_G1) on TEST, and does not beat the input-only control by 5 points or more. | open | |
| 17.4 | In the transplant arm, the paired readout moves the right way on at least 70% of TEST messages for G1. | open | |
| 17.5 | GL (unreadable) and shuffled labels both stay within 47-53% in arms 1 and 2. | open | |

17.3 is the honest lean: at natural strength the residual is small next to what the text explains, and 12j showed weak
signals collapse into answer habits. If 17.3 fails in the direction of a reader that beats the control, that would be the
first report of naturally varying, text-unpredictable activation we know of.

## Amendments before any run (2026-10-18 00:42 CT, after a code review of the first build)

- **The question names the passage, not "your last message".** In the reader prompt the model's last message is its
  own "Okay.", while the measured tokens are the user's passage. New wording: "While you read my first message, was
  group <name> of your neurons more or less active than its words would suggest? Answer higher or lower."
- **Capacity confound.** The self-reader is OLMo 2 1B and the control is Qwen3-0.6B, so a self-reader win could come
  from size. Pre-registered: if 17.3 fails in the direction of the self-reader beating the control, the control is rerun
  with Qwen3-1.7B before any claim is made, and the claim stands only if the self-reader also beats that.
- **Chance is the TEST majority-class rate**, reported beside every accuracy (the median threshold comes from in-sample
  FIT residuals, so TEST balance can drift from 50%).
- **LoRA drift is measured**: the reader's own G activations on TEST passages with the adapter on, against the base
  capture the labels came from.
