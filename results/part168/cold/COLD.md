# Cold check, Part 168 (worker-01)

## SUMMARY

Every scorer number I recomputed reproduces from the raw files, and the own-ban cut (31%) and mass-ban rise are real in this data. The write-up overstates the mass control, presents a failed superiority test as a headline contrast, and omits that a third of loops left under own are new.

## FINDINGS (most serious first)

1. **"equal-mass random bans add them"** (README title). The mass list is not random: exponent is above 0 in 74% of rows, and correlates 0.70 with how many plain-answer tokens the list holds (1.2 on average, against 0.07 for own). Amendment 1 forbids "same mass". Corrected: "a probability-tilted, mass-matched ban." Induced loops track the overlap: 1.3% of eligible rows at overlap 0, 43% (34 of 79) at overlap 4 or more.

2. **"a neighbour's top 50 cut about nothing"**, set beside the own cut as a contrast. 168.4 (own against matched, bar 0.05) failed: gap 0.047, interval 0.022 to 0.073. Corrected: "own-minus-matched gap 0.047, missed its 0.05 bar."

3. **"65 ended, 42 induced"**. The 65 is the own-word subset of 88. All 188 none-loops, 101 ended under own. Net removed is 101 minus 42, or 59. Own keeps 129 loops, and 42 of them (33%) are new; 40 of those 42 contain no own-list word.

4. **168.3 "held; wide interval"**. The cut clears 30% by 1.4 points. It drops below 30% when dropping layer 24 (21.6%), phrasing 8 (28.5%), 11 (29.8%) or 14 (29.4%), or with n+ alone (27.9%). Corrected: held by 1.4 points, not robust to single drops.

5. **"Part 168 n+ breakage 12.9% vs Part 165 25.7%"**. Part 165 used phrasings 0 to 7; Part 168 used 8 to 15, with different seeds. Label it a cross-part difference.

6. **Length and EOS** (asked, not stated). No shift: own answers are slightly longer (57.2 against 56.4 tokens), and EOS shares match (90.6% against 90.8%).

7. **Special tokens.** Six mass lists contain `</think>` or `<tool_response>`, which the design says must be removed. Gate 4 does not check this. Effect not measured.

8. **168.6 "held"** rests on 35 none-loops in the full arm, too few to separate 11% from 31%.

## HOLDS

- Recomputed exactly: the per-arm table, gates 1 to 7, 168.1 (ratio 215.6), 168.2, 168.3 (31.4%), 168.5, 168.6, 168.7 and 168.W3 (1 of 19 = 5.3%), 168.W1 (minus 16.5%), 168.W2 (42 against 11; sign test p < 0.001).
- Verdicts that stand: 168.4, 168.7, 168.W1 and 168.W3 fail; 168.W2 holds.
- 168.8 (9.6% against 16.3%) holds without the 139 repaired rows.
- Bootstrap intervals reproduce to within 0.1 point.

## COULD NOT CHECK

- The first launch's 89.5% Gate 6 figure: no output in the repo.
- Blinding of the hand-read judgements, and the freshness convention, which I read from code rather than from stored neuron ids.
- No model run; all numbers come from stored answers.
