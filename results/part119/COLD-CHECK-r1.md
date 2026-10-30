# Part 119 cold check, round 1 (Haiku 5.5 subagent, 2026-10-30)

**Count: 19 findings. Top three:**
1. Gate 2 passed on a comparison that never includes the 39 single answers the analysis uses. Nine of those answers differ from the batch-size-1 Part 92 answers, and batch dependence is absent from the Limits.
2. The title and headline describe a wider measure than the one computed: contested positions only, row means, and no comparison with controls, which score the same or higher.
3. 119.10 is withheld under a reading the design does not state. Read literally, it is read and fails (0 of 28 against 25%), so "neither held nor failed" is wrong.

**Files (absolute paths; short labels are used below)**
- README: /home/elisepc/green/worktrees/ab-part67/results/part119/README.md
- NUMBERS: /home/elisepc/green/worktrees/ab-part67/results/part119/NUMBERS.md
- SCORE: /home/elisepc/green/worktrees/ab-part67/results/part119/SCORE.md
- score119.json: /home/elisepc/green/worktrees/ab-part67/results/part119/score119.json
- DESIGN: /home/elisepc/green/worktrees/ab-part67/PART119-DESIGN.md
- score119.py: /home/elisepc/green/worktrees/ab-part67/part119/score119.py
- Raw: /home/elisepc/green/worktrees/ab-part67/results/part119/{answers,logits,interaction,ladder,gate2b,gate2c,labels-ds,labels-luna,labels-ds.jsonl.keys,known-ds,gate1}.jsonl (gate1 is .json)
- Figure: /home/elisepc/green/worktrees/ab-part67/results/part119/fig119.png
- Part 92 stored answers: /home/elisepc/green/worktrees/ab-part67/results/part92/step2/answers/instruct_chat.jsonl
- Part 92 kind rule: /home/elisepc/green/worktrees/ab-part67/results/part92/README-steps2-3.md (line 9)
- EPISTASIS: /home/elisepc/green/worktrees/ab-part67/EPISTASIS-FRAMING.md

## Findings (most important first; cat = your check number)

**1. [cat 2, 5] Gate 2 passes on a comparison that excludes the analysed answers.**
- README line 130 ("passed on its third form"), line 136 ("Amendment 2 same process at batch size 1 | 39 / 39"). Limits (lines 159–161) are silent on batch.
- Should say: the passing gate compares two batch-size-1 versions of the hook with each other. It never compares the 39 single answers the analysis uses, which DESIGN Amendment 1 says were generated at batch 64. Those answers differ from the batch-size-1 Part 92 answers for 9 of 39 units: p92-18, -30, -32, -40, -42, -43, -45, -47, -48 (agreed concepts -43, -45, -48). They differ from the per-layer gate2b answers for 4 (-15, -32, -47, -48), which is Amendment 2's "4 of 39". Single answers set p_A, t_A, the Path 2 positions and R, so robustness, flip alpha and Path 2 rest on them.
- Evidence: answers.jsonl single rows vs gate2c.jsonl (part92_answer and row_hook_answer agree in 39 of 39) and gate2b.jsonl. compute_gates in score119.py reads only the gate2c fields. Against Part 92's stored question-0 neuron answers, the main singles agree over the shorter length for 31 of 39 units, and batch-1 answers for 30, so batch 1 is not the closer reference. Batch shape is Amendment 2's hypothesis; I did not test it.

**2. [cat 2] "Interact alike" and "interval includes 1" overstate the test; the design's activation caveat is missing.**
- README line 22 ("interact alike: the estimated ratio is 1.044, and its 95% interval includes 1"); line 54 ("interact alike within the registered 25% bound").
- Should say: the test compares the size of the logit-space interaction against controls matched on the two single-effect sizes. The ratio is 1.044 (95% interval 0.961–1.129), inside the registered 0.80–1.25 bound. It says nothing about direction, and "includes 1" is not evidence of equality. Add DESIGN lines 216–217: "'Adds in logits' does not say the neurons do not affect each other's activations; a sum at the output can hide cancelling changes inside."
- Evidence: my recomputation reproduces the interval under the scorer's weighting. Under the literal weighting (finding 8) it is 1.0006–1.085, which excludes 1.

**3. [cat 3] 119.10 is withheld under a rule the design does not state.**
- README lines 143–145 and the PREDICTIONS.md 119.10 row: "withheld by rule. It is neither held nor failed"; "undefined kappa does not meet the agreement rule."
- DESIGN lines 207–208: withheld if agreement "falls below kappa 0.6". Undefined is not below 0.6. Observed agreement on "both" is 30 of 30, and neither reader ever says "both", so kappa is 0/0.
- Evidence: reader_analysis in score119.py passes only if kappa is not None and at least 0.6. That condition is already in bc13b26, before any labels, but the design text does not say it. Read literally, the numbers are not withheld, so 119.10 is read and fails (0 of 28 concept pairs against at least 25%). The scorer leaves it unread. Neither holds. One of the 12 known examples is a "both" item that both readers scored correctly, so the zero is not an inability to say "both". Name the reading used, give the other result, and record the interpretation in a dated design note.

**4. [cat 2] The title and first sentence describe a different quantity from the one computed.**
- README line 1 ("adding the two single effects hit 24.4% of contested pair tokens; margin winner take all hit 29.0%"); line 35; headline sentence on line 22.
- (a) 24.4% and 29.0% are means of row rates (line 28). Pooled over tokens they are 23.7% (2,264 of 9,546) and 28.3%. Line 5 uses "contested real pair tokens" for a pooled count, so the title contradicts it.
- (b) The measure covers contested positions only, where the two singles' top tokens differ: 9,546 of 15,292 real-pair positions (62%). The headline's "the pair's next word" is broader.
- (c) Random-direction controls score 28.0% (mean) and 29.6%. DESIGN lines 113–114 call control Path 1 rates "description only", but the README drops that and the figure omits controls.
- (d) The title does not name the model. DESIGN lines 200–201 require any top-token result to be stated with the interaction share (median 0.47), which neither the title nor the summary gives.
- Should say, e.g.: "On contested positions (where the two singles' top tokens differ), the additive top token matched the pair's own token in 24.4% of row averages (23.7% of tokens) for OLMo 2 1B at alpha 0.8. Random-direction controls scored 28.0% on their own paths, which is description only."

**5. [cat 3] Two predictions could not pass on these data; the README reports them as robustness failures.**
- README lines 16 and 19 (119.6 and 119.9 failed); line 122 ("the registered robustness … and disputed proximity predictions failed").
- Evidence: R lies between 0 and 0.058 for all 39 units (my recomputation matches score119.json to 1e-17). 119.6 needs a median difference of at least 0.10, which these values cannot produce. 119.9 needs the disputed median at least 0.05 closer to slot than to concept. For any value d, |d−c| − |d−s| ≤ |s−c| = 0.0051, so the threshold is unreachable.
- Should say: the thresholds were set on a scale the measure never reaches, a registration error. Both verdicts say nothing beyond the sign of the observed differences. Define R (DESIGN lines 86–88) and give its medians (0.007 agreed slot, 0.012 agreed concept).

**6. [cat 2, 3] 119.5 "held" has no interval, and the required leave-one-out is missing.**
- README lines 15, 95, 122 ("concepts won 90 of 137 decided pairs (65.7%)").
- Evidence: unit-bootstrap 95% interval 47.0%–82.4%; 25% of draws fall below 60%. The 168 slot×concept pairs are built from only 8 concept units. DESIGN lines 203–204 require 119.5 and 119.10 to be reported leaving out each concept unit, with coverage. My leave-one-out gives 118–123 decided pairs and a share of 62.7%–70.3%, all above both thresholds. The Haiku-only share (59.4%) is below 60%.
- Should say: give the interval and a leave-one-out table, and define "win" as a token count (the unit whose single top token more often equalled the pair's own token at contested positions), not a persona win.

**7. [cat 3] The registration trail is misstated and not cited.**
- DESIGN line 4 ("Registered 2026-10-30 03:23 CT … before any data"); line 9 ("The commit that adds this file is the registration"). The README has no registration statement.
- Evidence: no commit exists at 03:23. The file was first committed in bc13b26 (04:07:21 −0500) with the message "local, under review before registration". The registration commit is 0d6d572 (04:27:00 −0500, "Register Part 119"). Its DESIGN is byte-identical to 061fbba's (md5 8f6054…). The pilot launched at 04:27:21 (launch.jsonl). Amendments 1 (04:34) and 2 (04:36) came afterwards.
- Should say: cite 0d6d572 as the registration. Correct the 03:23 header with a dated note, not a silent edit.

**8. [cat 1, 3] The bootstrap weighting differs from the design's wording, and the README does not say which was used.**
- README line 28 ("intervals resample units"); DESIGN lines 70–71 ("a resampled unit brings all its rows").
- Evidence: row_bootstrap_weight in score119.py multiplies the two units' draw counts. Under the literal reading (a row counts once for each draw of either unit): real additive 22.7–26.1 (README 21.0–27.9); margin 27.7–30.3 (README 26.1–31.8); difference −6.1 to −3.0 points (README −8.0 to −1.3); effect-matched ratio 1.0006–1.085. Verdicts 119.1–119.4 are unchanged, but finding 2 depends on this choice.

**9. [cat 1, 2, 5] "Broken" uses an undescribed rule; the design's second measure is missing; controls break as often.**
- README lines 58–73 and 161 ("357 of 741 real pair answers broke", "Part 86's registered broken answer rule").
- Evidence: triage86.py also marks answers under four words as broken, and keeps any answer that names Qwen, Tongyi or Alibaba, or whose first sentence contains model, assistant, AI, chatbot or LLM. DESIGN lines 122–123 define broken as a three-word sequence repeated three times. On that rule alone: 330 of 741 pairs, 616 of 1,482 controls, 5 of 39 random singles (README: 357, 659, 1). The design's second measure, the reader's coherent field (line 122), is reported nowhere. SCORE line 91–92 says it is "reported separately", and it is not. DS labels mark 16 of 108 pair items coherent (5/40 slot×slot, 4/28 concept×concept, 7/40 slot×concept) and 9 of 40 controls. Controls break at 44.5% (659 of 1,482).
- Should say: the rule the counts use, the design's three-word measure alongside, the coherent counts, and that controls break at a similar rate.

**10. [cat 3, 5] Path 2 reports one unnamed scale and an undefined "interaction share".**
- README lines 47–52 ("Median interaction share, real pairs 0.471"). The scale (raw logits) is never named, and the share is never defined.
- DESIGN lines 76–77 require a second scale (total variation distance of the softmaxes). EPISTASIS-FRAMING lines 31–33 require each interaction on the raw and log-odds scales, with the claim's scale named.
- Evidence: median row TV is 0.707 (real) and 0.644 (control), matching SCORE's Path 2 table. The squared share 0.471 means |I| is about 0.69 of the joint change (controls about 0.61).

**11. [cat 4] Terms the README uses and never defines.**
- Appear on lines (approximate): contested (9), margin winner take all (5), token winner (1), decided (3), effect matched (3), Path 1 / Path 2 (6), interaction share (5), robustness R (5, plus R alone on 54), flip alpha (5), dominance (3), slot and concept (23 and 25, never explained), agreed / disputed / Haiku only (15, 6, 18), kind (9), DS / Luna / Haiku (16, 5, 18), kappa (4), directed (3), plus teacher-forced, greedy, triage, pilot and Gates 1–3.
- Suggested plain definitions (from the design and Part 92): contested position = a position where the two single pushes' top tokens differ. Margin winner take all = take the top token of the single whose top two logits are further apart; a tie is a miss. Token winner = at contested positions, the unit whose top token equals the pair's own token more often; a tie makes the pair undecided. Decided = a token winner exists. R = the mean fraction of the post-flip answer that a random second push leaves intact. Flip alpha = the smallest strength, in 0.1 steps, at which the single's first differing token is the top token. Dominance = wins divided by decided pairs. Slot = persona appears only on the two identity questions; concept = persona in 3 or more of 6 answers (Part 92 README line 9). Agreed = Part 92's label and Part 92b's DeepSeek label agree. DS = DeepSeek V4.1 Flash; Haiku = Part 92's first kind reader; Luna 6 = second reader.

**12. [cat 2, 5] Scope limits from the design are missing from the README's Limits.**
- Kind claims describe these 8 agreed concept and 21 slot units selected on this question, not concept neurons in general (DESIGN lines 203–204).
- Flip alpha cannot exceed 0.8 by construction and moves in 0.1 steps (lines 215–216). 11 of 21 agreed slot units sit at the 0.8 cap, so 119.7's 0.10 difference is one step.
- Contested positions can be style words, and token winners count tokens (lines 213–214). Pushes act at every position, prompt included (lines 217–218).
- Part 118's dose results, due 2026-10-31, may change what "slot" means (lines 219–220).

**13. [cat 5] Pilot reuse, the by-eye reading, and an unsupported batch-identity claim.**
- The README does not say that the pilot rows are reused in the main analysis (DESIGN lines 243–244).
- Amendment 1 says pilot-64 and pilot-128 were token-identical on 139 of 139 rows (lines 227–228). There is no pilot-128 folder in results/part119 (only timing-pilot-128.json), so this cannot be checked.
- DESIGN lines 208–209 require every pair class to be read by eye before counts are trusted. No record of this exists in the repository, and the README is silent. On the same 20 pilot pairs, Amendment 1 reports "about a quarter" broken by eye, while triage gives 8 of 20 (40%). This is not reconciled. The stopping rule (more than half broken) was not triggered on either measure.

**14. [cat 1] Example table: one cell is off, and the grouping is unstated.**
- README line 113 (p92-05 + p92-37): the cell reads "…a man with…". The first 90 characters of the answer are "…a man with a…", so the cell is 89 characters and drops " a". The other 22 distinct cell texts match the rule.
- The nine row IDs match the sort-then-shuffle rule exactly; I reproduced the seeded shuffle.
- README line 108 does not say the classes are agreed groups. Under Haiku groups the concept rows would be p92-29+p92-41, p92-30+p92-43 and p92-15+p92-28.

**15. [cat 1] The robustness p-value is not reproduced, and the test is not named.**
- README lines 16 and 106; NUMBERS line 33: one-sided Mann–Whitney p = 0.1468. fig119.py uses scipy, which the stdlib rule excludes. My tie- and continuity-corrected normal approximation gives 0.1411, and an exact midrank permutation gives 0.1415. The flip-alpha p (0.4372) reproduces. The verdict is unchanged. Name the test and version.

**16. [cat 2, 3] The Amendment 1 quotation is not the design's text.**
- README line 147 quotes "I printed the pilot's additive hit rate while checking gate 3." DESIGN line 236 reads: "to check gate 3 on the pilot I also printed the additive hit rate on the pilot's contested positions, 23.8% on 20 real pairs and 26.6% on 40 control rows, which is below the 60% in 119.1 and was not part of the gate check." The README's "I (the author)" should name Claude, per DESIGN line 4.

**17. [cat 2, 5] The figure can mislead.**
- fig119.png: the left panel shows only real pairs, so the 24.4% looks pair-specific (controls 28.0%). The 97.7% gate-3 bar sits among the predictions. The registered thresholds (60%, 25%) and the ±25% bound are not drawn. "Contested next tokens" and "Observed pair path match" are undefined. The values match the README, read by eye.

**18. [cat 5] Accounting gaps.**
- The effect-matched mean uses 1,475 of 1,482 real rows; 7 have no retained position. All 3,324 excluded real positions (17%) fall outside the control support range; none is excluded for nonpositive values. Neither point is stated.
- Each real pair appears twice in Path 2 (both orders, each on the first unit's answer). "1,482 directed rows" and the registered "1,482 real ordered pairs" should say 741 pairs.
- Gate 1 (zero push matches the no-hook answer token for token) passes (gate1.json) but is not reported in the README or NUMBERS.
- Reader settings (DeepSeek settings, 4,000-token cap), zero errors and zero retries (labels.log), and the seeds (11900, 11903) are absent.
- The "Gate 3, control positions 30,612 / 31,139" row counts all control positions, not contested ones.

**19. [cat 5] Before publishing.**
- 30 of the 31 files in results/part119 are untracked, including README.md, NUMBERS.md, SCORE.md, score119.json, fig119.py, fig119.png and every .jsonl. PREDICTIONS.md's Part 119 result column is modified and uncommitted. README line 171 reads "Cold check: pending."
- score119.json's reader.rule says "119.9 is withheld…". It should say 119.10; the string is in reader_analysis and dates from bc13b26. NUMBERS line 47 notes this, but the README does not.

## Verified as correct
- All 10 predictions match DESIGN lines 172–181 word for word, with the same confidence labels, in both the README and PREDICTIONS.md.
- Headline rule (3) produces the quoted sentence. Rule (1) is not triggered (119.1 fails), and rule (2) is not (the residual interval does not lie above log 1.25).
- Counts: 741 / 734 / 9,546 real; 1,482 / 1,452 / 18,256 control; broken counts 357, 4, 1, 659 under the file's rule.
- Logit rates and intervals, recomputed from logits.jsonl with the scorer's seed 11902 and weighting: real additive 0.243956 (0.209610–0.278616), margin 0.290198 (0.260933–0.318456), difference −0.046242 (−0.080033 to −0.012683); controls 0.280234 (0.244253–0.317013) and 0.295813 (0.275441–0.316293).
- Gate 3: 45,627 / 46,431; 9,324 / 9,546; 30,612 / 31,139.
- Path 2 medians 0.470673 and 0.368589, with 19,532 positions in each group.
- Effect-matched: coefficients −0.0266, 0.4992, 0.4444; 16,208 of 19,532 kept (0.829818); mean residual 0.043000 (−0.040032 to 0.121307); ratio 1.043938 (0.960759–1.128972).
- Class counts and means for both groupings (210/206, 28/28, 168/168, 335/332, 153/153, 378/375), class interaction shares, token winners 90/137/168 and 180/303/378, R medians (5 values), flip medians (4 values), flip p 0.4372, Spearman 0.034009 (p 0.837156), and the 10 disputed persona names against neurons.json.
- Reader: 0 of 148 "both" (by class 0/28, 0/40, 0/40, 0/40); Luna 0/30; known examples 12/12 for both readers. The 148-item sample reproduces exactly from seed 11900 (rows, classes, X/Y order, shuffle), and Luna's 30 from seed 11903.
- Gate 2 counts: 35/39 (main vs per-layer), 39/39 (batch 1), 32/39 (Part 118, first 100 characters). The four units Amendment 2 names are the ones I found.
- Pilot peek 60/252 = 0.238095 and 135/507 = 0.266272, reproduced from seed 11901.
- No README claim concerns an arm that was not read; random singles appear only as counts. No exploratory pattern is presented as a finding in the README; the layer-distance table lives in SCORE and is labelled exploratory there.
- All 39 relative links in the README resolve on disk.
- The registration commit 0d6d572 precedes the pilot launch, and its design text matches the review-folded version.

## Not checked
- fig119.py was not run, because numpy, matplotlib and scipy are outside the stdlib rule. I recomputed its numbers with the stdlib and read the figure by eye.
- The 0.1468 Mann–Whitney value was not reproduced (finding 15).
- No model was run, so the batch-shape cause is untested. The batch and length behind Part 92's stored answers are unknown.
- I did not confirm that each of the nine design-review findings was folded; I skimmed only the blocking ones.
- Reader label quality was checked only by counts. The by-eye reading has no record in the repository.
