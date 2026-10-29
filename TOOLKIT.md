# Experiment toolkit

These are the repository's reusable checks. [`PROCEDURE.md`](PROCEDURE.md) says when each one is used in a part's life. Use the ones that fit each study, and register any exception in its design before collecting or reading the affected data.

## A. Random directions at the same written norm

**What.** Draw random residual directions and scale each so its written push has the same norm as the neuron push. This checks whether the result belongs to the selected neuron rather than to any push of that size.

**Why.** In Part 94, the neuron produced dark answers in 5 of 6 trials at dose 0.8, while each same-norm random direction produced 0 of 6.

**Where.** <code>part86/common86.py::scaled_random_directions</code> and <code>part92/common92.py::random_directions</code>.

**How to run.** <code>python3 part94/push94.py --model-id MODEL --revision REV --frame chat --tag run --calib CALIB --passages part94/passages94.json --out results/part94</code>

## B. Register predictions before the run

**What.** Put predictions, thresholds, controls, and scoring rules in <code>PREDICTIONS.md</code> before collecting data. If a result reveals a flaw, amend the rule before reading the affected results and preserve the change history.

**Why.** Part 59's random controls produced 80 to 187 symmetric responders, so the prediction that each control would have fewer than five failed under the null; the ledger keeps that outcome visible. Part 94c Amendment 1 was recorded before its stage B labels and relabeling.

**Where.** <code>PREDICTIONS.md</code> and dated amendments in the relevant <code>PARTNN-DESIGN.md</code>.

**How to run.** Add the row before starting, then check its history with <code>git log -p -- PREDICTIONS.md</code>.

## C. Test for off-target leakage

**What.** Include ordinary questions alongside the target prompts. If a push changes answers to unrelated questions too, the apparent trait may be broad leakage rather than a specific effect.

**Why.** In Part 81, ordinary topics became conspiratorial almost as often as bait topics: 18 versus 22 answers. That makes ordinary questions a needed guard.

**Where.** <code>part94/push94.py</code> includes ordinary questions such as the capital of France and a short poem; Part 94 and Part 95 use this style of guard.

**How to run.** <code>python3 part94/push94.py --model-id MODEL --revision REV --frame chat --tag run --calib CALIB --passages part94/passages94.json --out results/part94</code>

## D. Compare base and post-trained models, including weight cosine

**What.** Run the same behavioral check on the base and post-trained models. Also measure cosine similarity for the relevant neuron's weights, since similar weights do not guarantee identical behavior in a changed model.

**Why.** Part 94 reported a base-to-finished down-column cosine of 0.9956, and Part 90 measured 0.99958 for N3851. Part 90 therefore registered a separate behavior test instead of treating weight similarity as proof of the persona.

**Where.** <code>part89/cos89.py</code> and <code>part90/tulu_weights.py</code> implement the weight comparisons. Part 94's 0.9956 value is in <code>PREDICTIONS.md</code>; this checkout has no dedicated Part 94 cosine script.

**How to run.** <code>python3 part89/cos89.py OUT.npz</code>

## E. Sweep dose to distinguish a threshold from a dial

**What.** Test several push sizes, including zero and modest values, and plot or tabulate the response at each size. A sudden jump suggests a threshold; a gradual change suggests a dial.

**Why.** Part 54 had 0 or 1 off-world answers of 24 through dose 1.0, then 19 of 24 at 1.5 and 24 of 24 at 2.0. This is a sharp threshold, not a smooth ramp.

**Where.** <code>part59/run59b.py</code>, <code>part62/run62.py</code>, and <code>part94/push94c.py</code>.

**How to run.** Repeat <code>python3 part94/push94c.py --model-id MODEL --revision REV --frame chat --tag dose --calib CALIB --dose DOSE --out DIR</code> for every preregistered dose.

## F. Separate prompt-only from generated-token pushes

**What.** Test a push during prompt processing separately from a push during answer generation. Also include the combined condition when the design asks whether both stages matter.

**Why.** In Part 62, Qwen's generated-token-only push produced the alien theme in 20 of 24 answers at dose 1.5 and 23 of 24 at dose 2; prompt-only produced 0 of 240 across ten doses. OLMo's twin needed the prompt and answer together.

**Where.** <code>part62/run62.py</code> defines the phase conditions, with scoring in <code>part62/analyze62.py</code>.

**How to run.** <code>python3 part62/run62.py --out results/part62 --device xpu</code>

## G. Blind readers need calibration and a cut-off rule

**What.** Hide the model, unit, and dose from the reader, and check the reader on known examples before interpreting labels. State in advance how truncated answers should be judged.

**Why.** Part 63's second reader agreed with the first at kappa 0.952 over 480 labels. Part 94c Amendment 1 added an explicit cut-off note and checked the unpushed model before relying on the coherence labels.

**Where.** <code>part94/label94a.py</code> and <code>part94/label94c.py</code>; the cut-off amendment is in <code>PART94C-DESIGN.md</code>.

**How to run.** <code>python3 part94/label94a.py ANSWERS_DIR labels.jsonl 8</code>

## H. Read activation on matched passage pairs

**What.** Measure activation on a target passage and a matched ordinary passage, keeping the pair similar in length and wording. This asks whether the neuron responds more to the concept than to generic features of the text.

**Why.** In Part 94, dark passages exceeded their neutral matches in 14 of 16 pairs for the finished model and 15 of 16 for the base. Part 95's space passages exceeded their ordinary matches by maximum activation in 11 of 16 pairs for both models.

**Where.** <code>part94/passages94.json</code>, <code>part95/passages95.json</code>, and <code>part96/passages96.json</code>.

**How to run.** <code>python3 part95/push95.py --out results/part95</code>

## I. Use an activity-matched null when zeroing units

**What.** Compare a zeroed target with units from the same layer whose activity on the study prompts is similar. A random same-layer unit can be nearly silent and make the target look unusually important.

**Why.** Part 67's first same-layer null made all six links look larger than controls. After matching activity, only two links remained above the controls.

**Where.** <code>part67/part67.py</code> implements the original test; <code>part67/part67b.py</code> implements the activity-matched comparison.

**How to run.** Collect with <code>python3 part67/part67b.py --out results/part67b</code>, then score with <code>python3 part67/part67b.py --out results/part67b --score</code>.

## J. Print probability mass for forced-choice readouts

**What.** For each answer option, sum probability across its token spellings instead of checking only one token. Forced-choice labels can split into several token forms, so a single-token read can miss probability assigned to the same option.

**Why.** In Part 34, the top next token favored A on 1,218 of 1,240 parity-free rows, while summed probability mass across A spellings favored A on 1,239 of 1,240.

**Where.** <code>part34.py::derive_ab_readout</code> and the forced-choice scoring code in <code>part34.py</code>.

**How to run.** <code>python3 part34.py run --direction-file FILE --out DIR</code>

## K. Read output weights through logit and Jacobian lenses

**What.** Project a neuron's write vector through the unembedding to find output-token effects. Compare the direct logit lens with the Jacobian lens, which accounts for how the rest of the network changes the effect.

**Why.** Part 57 found a median of 18.5 shared top-30 tokens for the two lenses on ten last-quarter-layer units. In Part 97, Jacobian matches separated real units from random directions, but the real-unit rate was only 10.8%, below the registered 20% floor.

**Where.** <code>part42.py</code> implements both readouts; <code>jlens_units.py</code> applies them to selected units.

**How to run.** <code>python3 jlens_units.py --model-name qwen3-1.7b --units 22:1610 --out OUT.json</code>

## L. Replay and independently audit large runs

**What.** Regenerate saved answers from the recorded inputs and compare them with the originals. Run an independent audit of the saved artifacts before interpreting the result.

**Why.** Part 86's pilot contains 9,728 answers, and its runbook requires exact replay of all 5,280 verification answers before those labels can be read.

**Where.** <code>part85/replay85.py</code>, <code>part85/audit85.py</code>, <code>part86/replay86.py</code>, and <code>part86/audit86.py</code>.

**How to run.** Audit with <code>python3 -I -B part86/audit86.py screen DIR --layers 22 --pilot</code>; replay with <code>python3 -I -B part86/replay86.py --verify-dir DIR --picked PICKS.json --calib CALIB --device xpu --out REPLAY.json</code>.
