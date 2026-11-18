# Part 205: in OLMo 2 1B the SAY and DO top-80 sets share 44 to 49 of their 80 parts at every stage read (O 0.623 to 0.746), with no stage changing that by the 0.2 the design can resolve; Qwen3.5-4B's share more (O 0.81 base, 0.89 chat); the copy test that would have tested the sharing causally did not run

![O by stage, and how often each model picks the value-consistent option](fig205.png)

Part 205 asked when in training the parts behind *saying* a value ("Do you care about honesty?") and *doing* it (a
scenario with a value-consistent and an inconsistent option) come to be shared. Parts (attention heads and MLP
neurons) are ranked by activation x gradient, and the overlap index O compares the SAY and DO top-80 sets against how
well each ranking agrees with its own paraphrases (O = Cross / Retest, defined under Method). O is 0.623 to 0.746 at
the five OLMo 2 1B stages where it was read (the sixth, pretraining at 3.0T tokens, failed gate G0), and 0.813 in
Qwen3.5-4B-Base and 0.893 in Qwen3.5-4B. The copy test, which would show whether the shared parts carry a "You"
self-attribution, did not run: the You frame and a named third-person AI answer almost alike (gap 0.019 at OLMo
Instruct, against a cost gate of 0.194), so the registered fallback (Amendment 2) dropped every copy arm. All numbers
are on the plain text frame and six values. Ranking, O and the letter analysis use the 24 SAY and 24 DO selection
prompts per stage (items 1 to 4); the You/third-person gap, G0 and A0 use the 24 held-out SAY questions (items 5 to 8).

**Read this before the O numbers (exploratory, found after scoring; no prediction reads it).** In OLMo 2 1B, the model
picks the value-consistent option close to chance before SFT (P(consistent) 0.50 to 0.52 on SAY and 0.46 to 0.49 on
DO), and on DO it stays close at every stage (0.59 at Instruct, where it answers A only 0.27 of the time). The
ranking then mostly finds parts whose estimated effect follows *which letter* is consistent rather than the value: for
the parts in both top-80 sets, the letter component carries 0.93 to 0.94 of the score-weighted attribution at the two
mid-training stages and 0.72 to 0.84 from SFT on, against 0.5 with no structure. So OLMo's O is mostly an overlap of
answer-letter parts, which two A/B questions on unrelated topics would be expected to share too (not measured here).
Qwen3.5-4B picks the consistent option 0.78 to 0.90 of the time, and its shares sit near 0.5 (top-80 sets 0.46 to
0.57, pools 0.53 to 0.60, shared parts 0.587 and 0.575); its O is the number in this part most likely to reflect
shared value-relevant parts, but the chat model reads this frame poorly (31 of its 48 ranking prompts put under half
the next-token mass on the two letters; Limits). See [Is the overlap about values?](#is-the-overlap-about-values).

## Predictions

Registered 2026-11-11 12:18 CT (design [PART205-DESIGN.md](../../PART205-DESIGN.md)), scored by
`part205/score205.py` into `results.json`. Under the reading rules a prediction whose every stage is unread is
UNTESTABLE.

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 205.1 | Claude, Opus 5.5 (medium): G4 passes (both Retests at least 20 of 80) at five or more of the six OLMo stages and at both Qwen models. | G4 passes at all six OLMo stages and both Qwen models; the lowest Retest is 62 of 80 (Qwen3.5-4B-Base SAY). | held |
| 205.2 | Claude, Opus 5.5 (medium): At OLMo 2 1B Instruct, O is at least 0.5. | O = 0.746 (95% CI 0.697 to 0.793). Exploratory letter analysis: mostly answer-letter parts (see [Is the overlap about values?](#is-the-overlap-about-values)). | held |
| 205.3 | Claude, Opus 5.5 (low): Across the six OLMo stages, the largest rise in O between adjacent stages is at mid-training end to SFT. | Four pairs read: -0.0943, +0.0998 (mid-training end to SFT, the largest, CI 0.044 to 0.147), +0.0059, +0.0172. None reaches 0.2. (Rescored on f83cd641, Amendment 6; on the wrong checkpoint the largest was DPO to Instruct.) | failed |
| 205.4 | Claude, Opus 5.5 (medium): At OLMo Instruct, DO-set copy gives T at least 0.5 and exceeds every set in all three random families. | Copy arms not run (Amendment 2 fallback; A0 gap 0.0191 against 0.194). | untestable |
| 205.5 | Claude, Opus 5.5 (medium): At the 3.0T pretraining checkpoint, if G0 passes: the You/third-person gap is under 0.10, or, if it is not, T is under 0.2. | G0 failed at 3.0T (option mass 0.342 against 0.5), its only stage. | untestable |
| 205.6 | Claude, Opus 5.5 (low): Reverse DO copy at OLMo Instruct lowers P_you by at least half of the gap. | Copy arms not run (fallback). | untestable |
| 205.7 | Claude, Opus 5.5 (high): Wherever DO-set transfer is read and passes the random-set rule, the off-target shift (mean absolute change in P(A)) is under 0.05. | No stage reads DO transfer (vacuous under the Reading rules). | untestable (vacuous) |
| 205.8 | Claude, Opus 5.5 (low): Qwen3.5-4B's (chat) O is at least 0.2 below OLMo 2 1B Instruct's (Sauers' contrast survives the retest ceiling, at another size and version); Qwen3.5-4B-Base's O is reported beside it, unscored. | Qwen3.5-4B 0.89259 against OLMo Instruct 0.74621: 0.14638 *above*, not below. Base 0.81349 (unscored). | failed |
| 205.9 | Claude, Opus 5.5 (medium): At OLMo Instruct, the persona-swap null gives T_swap under half of the DO set's T. | Copy arms not run (fallback). | untestable |

Two held, two failed, five untestable (205.7 vacuous). Withdrawal rule: no stage creates sharing that this design can
resolve.

## What changed in Amendment 6?

[Amendment 6](../../part205/AMENDMENT-6.md) (2026-11-18, written before the rerun): the mid-training-end stage had been
recorded on OLMo 2 1B's `main` (a1847dff), a different training run from SFT, DPO and Instruct (its embeddings correlate
0.008 with every other checkpoint). The stage was rerun on the real end of mid-training (f83cd641) and rescored with
the same code; the old outputs are kept in [midtrain-end-main-a1847dff/](midtrain-end-main-a1847dff/). Before the
rescore, rescoring the old stage with the same setup reproduced the published `results.json` exactly (apart from file
paths), so the change below comes from the checkpoint and nothing else
([control-rescore-main-2026-11-18.txt](control-rescore-main-2026-11-18.txt)).

- Mid-training end's O is 0.623 (interval 0.602 to 0.661), not 0.726. It is now the lowest OLMo stage.
- 205.3 still fails, but its shape changed: mid-training end to SFT is now the largest rise, +0.0998 with a paired
  interval of 0.044 to 0.147 that excludes 0, just as 205.3 named; it is under the registered 0.2, so the verdict
  stands. On the wrong checkpoint the same step read -0.003.
- Forecasts written before the rerun: O between 0.69 and 0.75 failed (0.623); 205.3 still failing held.
- The letter analysis at this stage barely moved (shared-parts letter share 0.930, was 0.931); the reading that OLMo's
  O is mostly an overlap of answer-letter parts stands.

## Why did the copy test not run?

The copy test replaces a set of parts' activations in a third-person run ("Zara, an AI, cares about honesty") with
their values from the You run ("You care about honesty"), and asks how much of the You/third-person difference in the
answer moves. It needs a difference to move. The pilot found none on honesty (gap 0.0068), so Amendment 2 added step
A0: the gap over all six values at OLMo Instruct, with a cost gate of 0.194. A0 measured **0.0191** (clustered SE
0.0152, 24 questions): branch FALLBACK. Every stage then ran only the control arms (ranking, You, third-person and
do-nothing runs, 168 passes). The gap stays small everywhere: between -0.0199 and 0.0207 at all eight stages, so G2
(gap at least 0.10) fails at every stage. At the seven stages where the frame elicits an answer (pretraining 3.0T
excepted: G0 failed and every You row is broken there), OLMo 2 1B and Qwen3.5-4B answer "Do you care about X?" the
same way whether the system line says "You" or names another AI.

Source: `results/part205/a0/screen.json`, `results.json` (stages[...].gates.G2, broken), printed in `report.md`.

## Is the overlap about values?

*Exploratory: found after the scores, read by `part205/letter205.py` from the stored ranking estimates of the 24
selection prompts per kind (SAY and DO items 1 to 4 of each value); no model was run and no prediction reads it.*

Every ranking prompt is a two-letter forced choice, with the consistent option A on 12 prompts of each kind and B on
the other 12. A part's estimate e is signed toward the consistent option, and the score keeps |e|. So a part that only
pushes "A over B" gets a large |e| on every prompt, flips sign with the consistent letter, and can enter the top 80 of
both SAY and DO. To separate the two, for each part over one kind's 24 prompts: the letter component is
|mean(e x s)| with s = +1 where the consistent option is A and -1 where it is B, and the consistency component is
|mean(e)|. A letter share is the sum of letter components over the sum of both. With balanced letters and no
structure the two components are equal on average, so 0.5 is the reference (the cold check simulated it: 0.499 for a
top 80 and 0.500 for a pool). "Pool" is every part, weighted by its ranking score; "random 80" is the median of 200
unweighted random draws of 80 parts; "both sets" are the parts in both full top-80 sets, with each component averaged
over SAY and DO.

| stage | SAY P(consistent) | DO P(consistent) | DO P(A) | SAY top-80 | DO top-80 | SAY random 80 | DO random 80 | SAY pool | DO pool | parts in both sets | their letter share (score-weighted) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrain-3.0T (not read; G0 failed) | 0.50 | 0.46 | 0.56 | 0.907 | 0.915 | 0.826 | 0.841 | 0.913 | 0.942 | 53 | 0.938 |
| midtrain-26B | 0.51 | 0.48 | 0.69 | 0.845 | 0.927 | 0.767 | 0.820 | 0.861 | 0.963 | 45 | 0.942 |
| midtrain-end | 0.52 | 0.49 | 0.59 | 0.862 | 0.911 | 0.792 | 0.832 | 0.862 | 0.944 | 44 | 0.930 |
| sft | 0.62 | 0.59 | 0.33 | 0.741 | 0.765 | 0.685 | 0.733 | 0.803 | 0.833 | 47 | 0.839 |
| dpo | 0.70 | 0.58 | 0.23 | 0.657 | 0.734 | 0.664 | 0.706 | 0.659 | 0.770 | 47 | 0.726 |
| instruct | 0.72 | 0.59 | 0.27 | 0.647 | 0.743 | 0.659 | 0.703 | 0.655 | 0.779 | 49 | 0.722 |
| qwen-base | 0.90 | 0.78 | 0.54 | 0.460 | 0.569 | 0.467 | 0.553 | 0.526 | 0.601 | 54 | 0.587 |
| qwen-chat | 0.84 | 0.78 | 0.56 | 0.532 | 0.557 | 0.554 | 0.576 | 0.560 | 0.579 | 62 | 0.575 |

Reading: in OLMo 2 1B before SFT the model answers mostly by letter (P(consistent) 0.46 to 0.59) and the network's
attribution to the answer is letter-dominated throughout. At the three pre-SFT stages the top-80 sets are more
letter-dominated than a random 80 (by up to 0.106: 0.927 against 0.820 at mid-training 26B, DO); from DPO on, a SAY set
can sit just under its random 80. The parts the two sets share are letter-dominated as well: at the two mid-training
stages 0.942 and 0.931 score-weighted, 0.889 and 0.900 unweighted (`letter.json`, shared). Post-training raises SAY consistency to 0.72 and lowers the
letter shares, but DO stays near chance with a strong B preference. So in OLMo, O mostly measures that a question
about a value and a scenario about the same value are answered by the same letter-choosing parts. Qwen3.5-4B answers
by content, its shares sit near 0.5, and its O is the more likely of the two to reflect value-relevant parts; the
design cannot say more without the copy test. The design had no gate requiring a stage to pick the consistent option
above chance (Limits).

Source: `results/part205/letter.json` and `letter.md`; the table above is printed in `report.md`.

## How stable is each ranking, and is O above noise?

| stage | G0 option mass | Retest SAY | Retest DO | Cross | O | O 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| pretrain-3.0T | 0.342 (fail) | 73 | 71 | 51.75 | 0.71875 (not read) | 0.684 to 0.755 |
| midtrain-26B | 0.715 | 63 | 68 | 47.00 | 0.71756 | 0.658 to 0.732 |
| midtrain-end | 0.760 | 71 | 71 | 44.25 | 0.62324 | 0.602 to 0.661 |
| sft | 0.618 | 65 | 65 | 47.00 | 0.72308 | 0.686 to 0.778 |
| dpo | 0.741 | 66 | 65 | 47.75 | 0.72901 | 0.686 to 0.790 |
| instruct | 0.846 | 66 | 66 | 49.25 | 0.74621 | 0.697 to 0.793 |
| qwen-base | 0.668 | 62 | 64 | 51.25 | 0.81349 | 0.768 to 0.917 |
| qwen-chat | 0.519 | 68 | 67 | 60.25 | 0.89259 | 0.814 to 0.992 |

Each ranking agrees with its own paraphrases on 62 to 73 of 80 parts, far above G4's floor of 20, and the SAY/DO
cross overlap is 47.00 to 60.25. O's bootstrap intervals are 0.07 to 0.10 wide for OLMo and 0.15 and 0.18 for Qwen;
two adjacent OLMo changes are near 0.1 (mid-training 26B to end -0.094, end to SFT +0.100, whose paired interval
excludes 0) and the rest are under 0.02; none reaches the 0.2 the design can resolve. G1 (the
do-nothing hook reproduces the plain logits) gives a maximum difference of exactly 0 at every stage. The full top-80
sets mix heads and neurons (OLMo 41 to 54 heads; Qwen 44 to 52 heads, of which 9 to 13 are DeltaNet heads).

Source: `results.json` (stages[...].overlap, O_ci, gates, sets.composition), printed in `report.md`.

## Does Sauers' OLMo-against-Qwen contrast hold here?

No, and 205.8 failed in the other direction: Qwen3.5-4B's O is 0.14638 above OLMo 2 1B Instruct's, not 0.2 below. This
does not refute Sauers' result, which used OLMo-2-7B and Qwen3-8B, a different estimator (ours is single-point
activation x gradient at the last position, since no code was released), and the overlap of raw top-80 sets rather
than O. Two things here weaken the comparison further: OLMo's O is mostly letter parts (exploratory; see above), and
the Qwen chat model reads the plain frame poorly, with 31 of its 48 ranking prompts putting under half the next-token
mass on the two letters (15 of 48 for the base model; 0 or 1 for OLMo after pretraining). Those rows stay in the
means, as registered.

Source: `results.json` (predictions["205.8"], stages[...].broken), printed in `report.md`.

## Method

The design is [PART205-DESIGN.md](../../PART205-DESIGN.md), with Amendments 1 to 5. OLMo 2 1B at six stages
(pretraining step 1430000 at 3.0T tokens, mid-training at 26B tokens, the released base `main`, SFT, DPO, Instruct),
float32 on the CPU with eager attention; Qwen3.5-4B-Base and Qwen3.5-4B text-only, bf16 on the Arc A770 with eager
attention, one model per process, with each model's kernel check passing first. The plain frame
`System: <persona line>\nUser: <question>\nAssistant: Answer:` at every stage, no chat template. Six values, 8 SAY
questions and 8 DO scenarios each, consistent option balanced between A and B. Ranking uses SAY and DO items 1 to 4 in
the You frame: the top 80 of heads and MLP neurons together by mean per-prompt-normalised |activation x gradient| of
the logit difference (consistent minus other) at the last position. **Retest** is the mean of the SAY halves' and the
DO halves' top-80 overlaps (items 1-2 against 3-4), **Cross** the mean overlap of a SAY half with a DO half over the
four pairings, and **O = Cross / Retest**, with a 1,000-resample bootstrap. Every readout is the probability of the
two option letters normalised over the two; **option mass** is the two letters' share of the whole next-token
distribution, and a row with option mass under 0.5 is counted as **broken**. Gates: G0 the frame elicits an answer
(option mass at least 0.5 and a letter on top for at least half the You questions), G1 the do-nothing hook is exact,
G2 the You/third-person gap is at least 0.10, G3 copying the SAY set itself closes at least half the gap (not read
here: no copy arm ran), G4 both Retests at least 20. **T** (not computed here) would have been mean(P_copy - P_third)
/ mean(P_you - P_third) over held-out SAY questions; **T_swap** the same for the persona-swap copy; **P_you** the You
run's probability of the consistent option. Under the Amendment 2 fallback each stage ran 168 passes (48 ranking, 24
You, 48 third-person, 48 do-nothing); no copy, random-set, persona-swap, reverse, off-target or Sauers-wording arm ran.

## Limits

- **The copy test did not run**, so nothing here shows that the shared parts carry a self-attribution, or that the DO
  set changes saying. O is an overlap of first-order estimates, not a causal result.
- **Found after scoring: no gate asked whether a stage picks the consistent option at all.** G0 checks that the letters
  get the mass, not which letter. Where the model answers by letter, as OLMo 2 1B does on DO at every stage, the ranking
  finds letter-choosing parts and O cannot separate value sharing from shared answer mechanics. The letter analysis
  above is exploratory; a gate on P(consistent), or a ranking that cancels the letter component across letter-paired
  prompts, would be the registered fix.
- One plain frame at every stage; the chat models normally answer in their template, and Qwen3.5-4B reads this frame
  poorly (31 of 48 ranking rows under half the mass on the letters).
- The ranking estimator is ours (single-point activation x gradient at the last position); a path-integrated estimator
  could rank differently.
- OLMo 2 1B and Qwen3.5-4B, not the 7B and 8B models Sauers used; six values written by me; DO scenarios 5 to 8 were
  not read.
- The You/third-person difference could partly be the word "you"; with a gap near 0 that question did not arise.

## Files

- `results/part205/results.json`: every scored number, per stage; made by `part205/score205.py` (log `score.log`).
- `results/part205/report.md`: the tables above, printed by `part205/report205.py`.
- `results/part205/letter.json`, `letter.md`: the exploratory letter analysis, by `part205/letter205.py`.
- `results/part205/fig205.png`: the figure, by `part205/fig205.py`.
- `results/part205/a0/screen.json`: step A0, the six-value gap screen.
- `results/part205/<stage>/records.jsonl`: every pass's letter probabilities and option mass; `sets.json`: the top-80
  sets and halves. The per-prompt estimates (`ranking.npz`, 24 MB per OLMo stage and 43 MB per Qwen model) and the
  copy-activation caches (0.8 to 1.1 GB per stage) are kept locally, not published; `part205/letter205.py` needs `ranking.npz`.
- `results/part205/g1/g1.json`: the A770 do-nothing check; `pilot/`: the honesty pilot (PILOT.md).

Cold check: two DeepSeek V4.1 Flash readers, 2026-11-15 (dark-factory runs part205-readme-cold-2026-11-15 and
part205-readme-cold-r2-2026-11-15; reports kept in the Journal repo).
Round 1 recomputed every O, Retest, Cross, gate and letter-table value from the stored files, all matching, and found 16
issues (none changing a verdict: wording stronger than the tables, mixed ranges, undefined terms, exploratory labels
missing where the letter result was restated); all folded, and the shared-parts measure it asked for was added to
`part205/letter205.py`. Round 2 confirmed the folds and found 5 issues in the new text (a comparison across two weightings, the
prompt scope of the gate numbers, a random-80 comparison that holds only before SFT, two range labels); all folded.
