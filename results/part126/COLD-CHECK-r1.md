# Part 126 cold check, round 1 (Haiku 5.5, no context, 2026-10-30)

Part 126 is a registered null on one token of one prompt, and the README's wording goes beyond that. Every number I could check from the raw files reproduces. The problems are in what the numbers are said to mean. The role region covers one of five role-word occurrences, the "other" and "question" labels hide what the heads attend to, and the null is not robust to the region choice. Short paths below use this legend:

- README = /home/elisepc/green/worktrees/ab-part67/results/part126/README.md
- NUMBERS = /home/elisepc/green/worktrees/ab-part67/results/part126/NUMBERS.md
- DESIGN = /home/elisepc/green/worktrees/ab-part67/PART126-DESIGN.md
- PRED = /home/elisepc/green/worktrees/ab-part67/PREDICTIONS.md
- PARTS = /home/elisepc/green/worktrees/ab-part67/PARTS.md
- P126 = /home/elisepc/green/worktrees/ab-part67/part126/ (reads126.py, score126.py, lens126.py)
- P82 = /home/elisepc/green/worktrees/ab-part67/part82/part82.py
- P83 = /home/elisepc/green/worktrees/ab-part67/part83/ (part83.py, score83.py)
- Part 83 results = /home/elisepc/green/worktrees/ab-part67/results/part83/main/results83.json

I changed no repo file. My scripts are in /tmp/claude-1000/-home-elisepc-green-Journal/20e07d58-a259-44d2-beab-ad3674ce8941/scratchpad/ (rerender.py, core.py, broad.py, power.py, tokens.py, breakdown.py, selfmass.py, cos_perm.py).

## 1. Recomputation

**Reproduces from raw files:**
- rho = -0.02346 (README line 3, 13).
- Shuffle 95th percentile = 0.290689. 5366 of 10000 exceed rho. p = (5366+1)/10001 = 0.536646. Seed 12602 matches.
- L11 h5 copy-A role rank 20 of 32, mass 0.235%. Top five are L15h13 1.975, L11h9 1.873, L11h11 1.634, L11h7 1.522, L15h14 0.917.
- All 64 head-copy rows by six regions in NUMBERS (lines 78–150) match recomputed masses within 0.0005 pp. Rebuild from raw attention differs by at most 0.00024.
- NUMBERS Part 83 ranking (lines 41–72) matches Part 83 results for all 32 heads.
- README line 20 drop values match.
- Largest-region counts: question 23 of 32 (copy A) and 25 (copy U). "Most heads" holds.
- 126.4: 20 mattering heads, 17 below 0.90. Closest are L11h5 at 0.9046 and L11h10 at 0.9016.
- 126.3: no registered token in L11 h5 copy-A role top 20.
- Lens overlap: 64 pairs, mean 1.109, median 0, 48 with none.
- All README lens-table cells (lines 28–35) match lens126.json.
- Figure values and title (rho -0.023, p 0.537) match NUMBERS.
- All 600 re-rendered prompts match run.json fingerprints and saved input_ids, with zero mismatches.

**Does not reproduce:**
- README line 41 and NUMBERS line 23 give the region-sum gate as 0.002243 and cite run.json. run.json records 0.00244140625. The 0.002243 figure is only what fig126.py recomputes from saved float16 masses.
- README line 51 says the random-body control uses "seed 12601". The code seeds each row with 12601 + row_index (reads126.py line 388), and the lens averages those rows (lens126.py lines 203, 209). Wording is incomplete.

**Not recomputable from the folder:**
- Gate 1 error 0.023438 needs pre-o_proj tensors, which are not saved.
- The no-op logits check and the positive control 9.109 need the model; the control prompt is not saved.
- Lens top-20 lists need the Part 111 J matrices and unembedding, which are outside the folder.

## 2. Claims stronger than the data

- README line 3 "did not track". The 95% interval for rho is about -0.37 to +0.33 (Fisher) and -0.39 to +0.35 (bootstrap over heads). The data do not exclude a moderate relation.
- README line 22 "withdraw the claim that causal importance tracks where these heads look." The claim is never stated in the README, and 126.1 tested one token location only.
- README lines 3, 22, 26 use "read", "looks", "barely reads" for attention mass. Attention is not reading. The role token's value norm is about 1.8 times the question value norm (L11h5 copy A: 2.57 vs 1.41). Its approximate share of head output norm is about 2 times its mass (0.48% vs 0.235%). This is rough, but the README's "barely" language is about mass only.
- README line 20 "Role-token mass was generally small." True for the header. The whole role word is not small. For L11h5, all role-word occurrences total 3.34% (copy A) and 4.09% (copy U). L15h5 puts 11.0% on the generation-prefix "assistant" token.
- README line 20 "Question text held the largest mass." The question region is mostly formatting. L11h5's question mass is newline 19.8%, "?" 10.9%, "A" 5.4%, "." 4.6%. L15h1's is "B" 16.8%, " letter" 16.3%, newline 12.2%, "." 8.6%, "C" 7.6%.
- README line 20 "other prompt tokens." The region includes the final token attending to itself. Pooled self-attention is 43.7% for L11h3, 32.6% for L15h13, 31.7% for L15h5, 29.9% for L11h11. The rest is "\n\n", "<think>", "</think>". These are not prompt tokens in the README's sense.
- README line 55 "one of many positions where a head may use the role word." The role word occurs five times per prompt. The gap is not positions; it is occurrences.
- README line 43 "The captured attention is unchanged." No run compared against the full model. It is true by causal construction, so say that.
- README line 37 and line 51 present the random-body lens control as a comparison. It is a mean over about 300 different body tokens, while the role write is one token's value. The comparison is not matched.
- Prediction 126.4 "held." The two copies differ only at token position 1, the role word (300 of 300 matched pairs differ only there). The median cosine between the two header writes is 0.84. Their cosines to random-body writes are 0.16 and to region writes 0.53. The 0.90 cutoff has no null, so "held" reflects token identity.
- DESIGN line 31 says weights are over "all earlier token positions." The code (P126/reads126.py lines 312–325) includes the query's own position. The README's region definitions inherit this.

## 3. Region definitions

- The role token is a single token at position 1 in both copies. No word is split.
- The role word occurs five times per prompt. Header at position 1 (role region). "user" in the question header (other). "assistant" in the generation prefix (other, last eight tokens). "assistant" and "user" inside the option labels (options region). Only the header is counted.
- The header's share of all role-word attention is 7.0% for L11h5 in copy A and 10.8% in copy U. Mean across 32 heads is 11% and 14%.
- L15h13: header 1.98%, generation-prefix "assistant" 6.48%. L11h7 copy U: header 2.30%, question-header "user" 8.24%.
- Position 1 is the only place the copies differ. Copy identity therefore enters through one token, whose value is effectively prompt-independent. Its batch-level numerical variation is about 5% relative, not content.
- Position 0 (`<|im_start|>`) is an attention sink. It sits in the template region. Pooled attention to it reaches 24.9% for L15h14 and about 20% for L15h13. Template masses are mostly sink.
- The same `<|im_start|>` token is "template" in the copied message and "other" in the question message.
- The final token is "\n\n" after "</think>". That matches Part 83's readout position, so the position is right for answer readout. The nearest self-label, the generation-prefix "assistant" at offset -6, is in "other".
- Region assignment is by character overlap (reads126.py lines 58–67, 104–121). I decoded two rows only, not all 600, so boundary errors are unchecked.

## 4. Part 83 drops

- Same model revision, same 100 bodies (Part 83 dropped count 0), same copies, and same three orders. Part 83's orders (P83/part83.py line 11) match Part 80b's cyclic orders. All 600 prompts reproduce.
- Scope mismatch, not in the README. P82 lines 47–54 zero each ablated head's output at every position through the full forward pass. Part 126 measures one position's attention to one token.
- Absolute value. score126.py lines 82–83 take |drop|. Six Part 83 heads have negative signed drops. L15h3 is a mattering head with signed -0.089, so removing it raises the gap. Signed rho is +0.076. The conclusion is unchanged, but the README never mentions sign.
- Weak ranks. Nine heads have |drop| below 0.025. L15h9, L11h9, and L15h15 have intervals spanning zero. No exact ties. Their order is noise, and abs() folds that noise onto the positive side.
- Power, n=32. Fisher 95% interval for rho is -0.369 to 0.328. Bootstrap over heads is -0.393 to 0.354. Leave-one-out range is -0.117 to 0.046. Simulation with the real drop ranks gives P(rho ≥ 0.40) of 0.26 at true rho 0.30, 0.53 at 0.40, and 0.77 at 0.50. The false-positive rate of the registered rule is 0.013. At a true 0.40 the rule has about even odds. "Census" (DESIGN line 47) does not address this.
- No layer confound. Within L11, rho is 0.003. Within L15, it is -0.065. Copy A is -0.034 and copy U is +0.029.
- Unregistered role-word splits (not corrected for multiplicity): generation-prefix word rho -0.344, two-sided p 0.058. Question-header word +0.328, two-sided p 0.071. Option text +0.109, p 0.56. All five occurrences -0.124, one-sided p 0.75. The null depends on which occurrences count.
- Part 83's own rule 83.6 failed. It predicted 3 to 10 mattering heads and found 20 (results83.json). The README and DESIGN are silent.

## 5. Three attacks

1. "You measured the wrong role word." Not answered. Five occurrences exist, the registered region holds one, and the occurrence nearest the answer is in "other".
2. "Attention is not what the head does, and the ablation was all-position." Partly answered. README lines 22 and 55 call attention descriptive. Not answered: the all-position scope, the undefined withdrawn claim, and the role token's output share being about twice its mass.
3. "A null at n=32 is not evidence of no relation." Not answered. No interval, no power, and the wording "did not track."

Also weak: 126.4 is near-trivial, and the lens control is unmatched. Amendment 3 (DESIGN line 16) discloses that the first three attention rows were viewed before the gate-4 fix. L11h5 is one of those rows and is the subject of 126.2. That is disclosed but should be stated plainly.

## 6. Missing for a stranger

- What claim is being withdrawn, and where it was stated. a9lim's post and the motivation appear only in DESIGN lines 22–25.
- Part 83's definitions: all-position ablation, and the gap as the mean over three orders of P(you|A) minus P(you|U) (P83/score83.py, function score).
- One worked rendered prompt with region labels. It should show five role-word occurrences, the sink, and the last token.
- That position 0 is a sink and that "other" contains the final token.
- Part 83's 83.6 failure and the sign handling, with signed rho.
- Intervals and power for rho, and the list of unregistered splits with their p-values.
- The random-body control is an average over 300 different body tokens, and the role write is one token.
- Scale for gate 1. The 0.05 tolerance is absolute. Median absolute region-mean value is 0.047 and the 99th percentile is 0.53, so 0.023 is not small in these units.
- The positive-control prompt is saved only as a description in run.json.
- Reproduction order and a commit hash.
- README line 74 "Cold check: pending" must go. PARTS line 189 says "done".
- Figure: black labels on dark cells in the role and option columns are hard to read. Scatter labels overlap at 11.1/11.2, 15.8/15.11, and 15.15/15.9.

## Fixes, priority order

1. Rewrite README lines 3 and 22. State that one location (header, position 1) showed no relation, with interval -0.37 to +0.33. Delete "withdraw" or define the claim and where it appeared.
2. Report all five role-word occurrences per prompt. Report the header share (7% to 14%). Report the unregistered splits with p-values and a multiplicity note. State that the null depends on the definition.
3. Disclose the scope mismatch: Part 83 ablates at every position through the whole forward pass. Part 126 measures last-position attention to one token.
4. Relabel regions. "Other" contains the final token itself (self-attention up to 44%) and answer-format markers. "Question" is mostly newlines, "?", and option letters. Fix README line 20. Change DESIGN line 31 to match the code, which includes the query's own position.
5. Replace "read", "looks", and "barely reads" with attention-mass language. Add the output-share caveat.
6. Downgrade 126.4 to a token-identity check. Give the baseline cosines and say 0.90 is uncalibrated.
7. Describe the role write as one position-1 token's value. Describe the control as a mean over about 300 body tokens. Drop the overlap comparison, or match it.
8. Fix the 0.002243 citation (NUMBERS line 23, README line 41 against run.json's 0.00244). Fix the seed description (README line 51). Change "captured attention is unchanged" (README line 43) to "true by construction, not checked." Say the 0.235% and 0.443% figures are copy A and copy U.
9. Disclose Part 83's 83.6 failure, sign handling, and signed rho.
10. Remove "Cold check: pending" (README line 74). Align PARTS line 189.
11. Add the section 6 context: a worked prompt, Part 83 definitions, intervals and power, gate-1 scale, and the positive-control prompt.
12. Fix figure contrast and label overlap.

NOT PUBLISHABLE
