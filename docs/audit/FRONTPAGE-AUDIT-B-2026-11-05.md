# Front-page audit B: Parts 100 to 176 (2026-11-05)

Base: `origin/main` at fa68418. Scheduled routine R3: read-only except this file. Parts 177 to 189b go to the other
auditor. Six sub-auditors each took one group of parts, and I re-checked the main findings myself. Where it was
possible we recomputed values with python3 and numpy from the committed result files (labels, answers, score and
results JSON, cell tables). No model was run.

## Summary

- **282 claims checked**: 263 MATCH, 7 MISMATCH, 8 OVERCLAIM, 3 UNVERIFIABLE, 1 missing cold check (Part 125).
  Part 127's missing cold check sits in a combined row.
- Every front-page entry from Part 100 to Part 176 was covered. Part 167 has no entry (see problem 12).
- **PREDICTIONS.md:** every row for Parts 100 to 176 has a verdict. Each verdict agrees with its part README, except
  that row 160.6's result cell gives a wrong count (problem 1). The "pending" rows belong to Parts 131, 132 and 141,
  which are registered but were never run. None of the three has a results folder or a front-page entry.
- **Cold check lines:** results/part125/README.md and results/part127/README.md have no `Cold check:` line, which
  PROCEDURE.md section 8 requires from Part 115 on. Parts 100 to 111 and 113 also have none, but they predate that
  rule, so they are noted here and not counted as violations.
- **Cross-references:** 55 were checked (list below). Two cited numbers are wrong: Part 153's base rate from Part 148
  and Part 151's description of Part 151b.

Verified by me independently, not only by a sub-auditor: problems 1, 2, 3, 6 and 9, plus the Part 110, 151, 143, 139
and 155 source lines.

## Ranked problems (most consequential first; the "Cited" count is how often other READMEs cite the part)

1. **MISMATCH, PREDICTIONS.md:3102 (row 160.6).** It reads `failed: 5 of 12 and 4 of 7`. The real count is 5 of 7:
   results/part160/score.json, set 1.0, lists Google 5, Tencent 1, Microsoft 1, and results/part160/README.md:16 and
   :55 both say 5/7. Proposed: `failed: 5 of 12 and 5 of 7`. The verdict itself stays failed.
2. **OVERCLAIM (scope missing), README.md:166 and README.md:190, Part 112 (cited 53 times).** The front page says
   "101 of 343" and "94 of 344". Those counts come from the q_name score group only (results/part112/NUMBERS.md:18-19).
   Over all random rows in results/part112/labels.jsonl the counts are 109 of 464 ("Who are you?") and 103 of 464
   ("Tell me about yourself"); I recomputed both.
   Proposed for both lines: "random directions paired with Q-name targets received persona labels ... in 101 of 343
   ... and 94 of 344 (109 and 103 of 464 over all random directions)".
3. **OVERCLAIM, README.md:176 (Finding "Not a state", Part 125).** It says "The pushed persona persists only through
   the generated text being re-read." But the text-only twin kept the persona to turn 5 for just 3 of 29 units, and
   prediction 125.2 ("text alone carries the persona across four turns") **failed**. Proposed: "On these 29 units at
   one dose, the kept cache added nothing beyond the generated text, and the text alone carried the persona to turn
   five for only 3 of 29 units."
4. **OVERCLAIM (rules mixed), README.md:84 (Parts 164 and 165).** "21.4% against 5.2% and 27.0% against 7.8%" are
   breakage rates under Part 86's triage. The "26%" that follows uses Part 165's loop rule, under which Part 164's
   rates are 32.7% against 12.4% (results/part165/README.md:31, recomputed). Proposed: name the rule beside each pair,
   for example "...broke answers more often by Part 86 triage, 21.4% against 5.2% and 27.0% against 7.8% (32.7%
   against 12.4% under Part 165's loop rule). At equal delivered dose...".
5. **OVERCLAIM (sensitivity left out), README.md:2655, Part 142 (cited 8 times).** "the median margin differs by
   0.056 on Haiku's sort, n=364" is the registered primary, which excludes pilot units. With the pilot units included
   the difference is 0.047, below the 0.05 bar (results/part142/README.md:52, :59). Proposed: append "(0.047 with the
   22 pilot units included, below the registered 0.05)".
   Same line, a minor wording issue: "92% NaN with no error". The run also stopped twice with UR_RESULT_ERROR_UNKNOWN
   (results/part142/README.md:98). Proposed: "92% of rows NaN, with no error raised at those rows (the run also
   stopped twice on a device error)".
6. **OVERCLAIM, README.md:2650, Part 139 (cited 12 times).** "garbled prompts drew no apology, 0 of 50" holds only
   for random token IDs. The other garblings drew some: shuffled letters 1/50, 40% of characters dropped 3/50, random
   words 2/50 (results/part139/README.md:72-79, recomputed from results.jsonl). Proposed: "prompts of random token IDs
   drew no apology, 0 of 50, and milder garblings at most 3 of 50."
7. **OVERCLAIM (scope), README.md:2566, Part 102.** "A later opposite-direction check (Part 105) showed that a
   same-size attention change can also remove these answers". Part 105 swapped attention in all 16 layers, and its own
   Limits section says it does not test Part 102's layers 0 to 3 swap. Proposed: "A later all-layer opposite-direction
   check (Part 105) showed that a same-size attention change can also remove slot answers; it did not test the layers
   0 to 3 swap, so ...". The same wording is in results/part102/README.md's closing "Later (Part 105)" note.
8. **MISMATCH, README.md:2588 and results/part110/README.md:5, Part 110.** "their top-20 token readouts shared
   16–20 tokens". For the four maker neurons the overlap is 16 to 18: L6 N753 16, L7 N1169 18, L11 N8868 17,
   L29 N776 16 (results/part110/weights110.json, `base_chat_top20_shared`). 16 to 20 is the range over all six
   neurons read, which includes L27 N3325 at 20. Proposed: "16–18 tokens (16–20 over the six neurons read)".
9. **MISMATCH (in a cited number), results/part153/README.md:39, Part 153 citing Part 148.** It says "55 of 465
   neurons and 56 of 465 random directions". A substring or word-boundary recount of results/part148/answers.jsonl on
   question_index 1 gives **58 and 59**. Proposed: "58 of 465 neurons and 59 of 465 random directions". If the scorer
   used a stricter matcher, name the matcher instead. This line is not on the front page.
10. **MISMATCH, README.md:2673 and results/part151/README.md:29, Part 151.** It says Part 151b ran "on 10,000
    documents". Part 151b read documents 2,000 to 9,999, which is 8,000 new documents
    (results/part151b/README.md:3). Proposed: "repeats the run on the next 8,000 documents".
11. **UNVERIFIABLE (reads as done), README.md:2632 and results/part128/README.md:101.** It says "Part 132 repeats it
    on 189 fresh ones". Part 132 is registered only: there is no results/part132, and its PREDICTIONS.md rows at
    :2768-2775 read pending. Proposed: "Part 132, registered and not yet run, repeats it on 189 fresh ones."
12. **Missing entry: Part 167.** The list jumps from Part 166 (README.md:2705) to Part 168 (README.md:2707). Part 167
    has results, PREDICTIONS rows ("Not read: Gate 2 failed") and a README. Proposed: add a one-line entry before
    README.md:2707 saying that Part 167, the first test of the own-word ban, stopped at its plain-answer gate on 1 of
    8 phrasings and reports no verdict. Parts 131, 132 and 141 have no line either. Proposed: one sentence in the
    "Running or designed" section saying they are registered and not run.
13. **Missing `Cold check:` line, results/part125/README.md and results/part127/README.md.** Proposed: run the cold
    check and record it, or add "Cold check: not yet run." until then. `tools/part_check.py publish` should have
    caught both, so it is worth checking why it did not.
14. **MISMATCH (minor), README.md:2683 and results/part155/README.md:127, Part 155.** "23 or 24 of 24 at +0.2". A
    recount of results/part155/answers.jsonl gives 24 of 24 for all four neurons. The README table has 23 for
    L24 N5017; I did not recheck the sub-auditor's recount. Proposed: "24 of 24 at +0.2", and correct the table cell.
15. **MISMATCH (minor, the numbers come from two bases), README.md:2657, Part 143.** "'m reads 14%, 19%, 38% and
    45%". 45% is the neuron-only full row (0.452). The pooled Set R median, on the same basis as the 45.1% headline for
    " sorry", is 43.9%. Proposed: "... and 44%", or label every figure as neuron-only.
16. **OVERCLAIM (minor), README.md:2709, Part 169.** It says "which Weaver predicted in size". Prediction 169.W2
    registered only a floor: at least a third of own's cut, a bar of 42.7, against 283 observed. Proposed: "well past
    Weaver's registered floor".
17. **OVERCLAIM (minor), README.md:2695, Part 161.** It calls the paired directions "unpicked", but 2 of the 37 are
    themselves reliable picks (results/part161/README.md:3). Proposed: "the paired directions, most of them unpicked,".
18. **Minor, README.md:80 and README.md:2713, Part 171.** "random neurons 12.9%" is the mean of three draws: 36.4%,
    2.3% and 0%. "Below random neurons" holds for the mean only. Proposed: "below the random-neuron mean of 12.9%
    (draws 36.4%, 2.3%, 0%)".
19. **Minor, README.md:2586, Part 109c.** "a dog in 5 of 7 answers": only 2 of those 5 were coherent, against 3 for
    the concept dog alone. Proposed: add "(2 of them coherent)".
20. **Minor, README.md:2580, Part 108.** "32 (DPO scale) of the 64 units": the DPO-scale denominator is 63, because
    p104-28 is unlabelled (score108.json, FR unlabelled). The 108.5 result cell (PREDICTIONS.md:2427) leaves out
    FP 14 of 18; its verdict stands.
21. **Minor:** the titles of results/part102/README.md and results/part103/README.md state the reading that the later
    Part 105 note qualifies. Part 130 "reproduces Sauers' figure" and Part 162's "public chart" rest on external
    sources with no link in the repo. PREDICTIONS.md marks Part 151 "untestable" where score.json and the front page
    say "withheld".

## Claims table (sorted by how often the part is cited, most-cited first)

Status counts: MATCH 263, MISMATCH 7, OVERCLAIM 8, UNVERIFIABLE 3, MISSING 1 (cold check). "README.md line" is the
front-page line.

| Part | Cited | Claim (short quote) | Status | Evidence | README.md line |
|---|---:|---|---|---|---|
| 112 | 53 | "all showed their target on 'Who are you?'" (465) | MATCH | recomputed from results/part112/labels.jsonl: 465/465 neuron q0 true | 2592, 200 |
| 112 | 53 | "257 concept, 189 slot, 17 unclassified and 2 unsorted" | MATCH | recomputed (Part 92 rule + Amendment 1 fill-all): 257/189/17/2; unsorted q112-033, q112-152 | 2592, 200 |
| 112 | 53 | "112.2 and 112.4 failed; 112.1 and 112.3 held" | MATCH | results/part112/score.json verdicts; slot 154/329=46.8%; 112.3 registered 78-entry scope 56/76 vs 175/329 (recomputed); PREDICTIONS.md:2478ff agree | 2592 |
| 112 | 53 | "Random directions showed a persona in 148 of 1,860 answers to the four non-self questions" | MATCH | recomputed random q2-q5: 148/1860 = 7.96% | 2592, 200 |
| 112 | 53 | "random directions received persona labels on 'Who are you?' in 101 of 343 answers and on 'Tell me about yourself' in 94 of 344" | OVERCLAIM | these are the Q-name score group only (results/part112/NUMBERS.md q_name rows). All random rows: Who 109/464, Tell 103/464 (recomputed) | 166, 190 |
| 112 | 53 | Cold check line | MATCH | results/part112/README.md:130 "Cold check: Haiku 5.5, round 1..." | - |
| 148 | 42 | "146 move name and maker in readable answers against 96 paired random directions" | MATCH | Recomputed from answers.jsonl + labels.jsonl (registered regex not-qwen/tongyi, not-alibaba/tongyi/qwen, DS coherent): neurons both 146, random both 96; score.json exploratory.coherent_name_and_maker | 2667 |
| 148 | 42 | "10 name the same company on both questions" | MATCH | score.json exploratory.neurons.same_company_on_name_and_maker.count=10 (7 Google, Baidu, Huawei, Microsoft) | 2667 |
| 148 | 42 | "Readable maker moves are level at 191 neurons and 197 random directions" | MATCH | recomputed maker coherent: neuron 191, random 197; raw moved 406 vs 299 | 2667, 84 |
| 148 | 42 | "No unit moved the registered 'Are you Qwen?' rule" / "Five of six predictions failed" / "148.5 sits on the line" | MATCH | score.json coherent_by_question[2]=0 both sides; verdicts 148.3 held only; results/part148/README.md (spellings section: 172 vs 172.5) | 2667 |
| 148 | 42 | Findings: on the 245 units "81 against 48 and 107 against 112"; "148.5 ... held ... 89 ... against 184" | MATCH | results/reader-correction-2026-11-03/part148/v2_another_ai/score.json random coherent [89,112,0], both 48; README.md:78-80 there | 84, 170 |
| 119 | 38 | "additive top-token prediction matched in 24.4% of row means" | MATCH | recomputed from results/part119/logits.jsonl pair rows: 734 rows with contested positions, mean row rate 0.24396 | 236, 2610 |
| 119 | 38 | "pooled ... 2,264 of 9,546 tokens (23.7%)" | MATCH | recomputed from logits.jsonl: add==g in 2264/9546 = 0.2372 | 236 |
| 119 | 38 | "Median interaction share was 0.471" / "0.47" | MATCH | score119.json interaction_summary.real.median_row_interaction_share = 0.47067 | 236, 2610 |
| 119 | 38 | "real pairs broke 357 of 741 ... random 286 of 741 ... Fisher p=0.000241" | MATCH | score119.json broken_answers.real_pair 357/741; Fisher two-sided recomputed on 357/741 vs 286/741 = 0.000241 (286 from 119b) | 238 |
| 119 | 38 | "ratio was 1.044 (95% interval 0.961 to 1.129)" | MATCH | score119.json effect_matched: exp(0.04300)=1.0439; exp(-0.04003)=0.961, exp(0.12131)=1.129 | 238, 2610 |
| 119 | 38 | "retained 16,208 of 19,532 ... dropped 3,324, or 17%" | MATCH | score119.json effect_matched real_positions_kept 16208/19532 = 0.8298; 19532-16208 = 3324 | 238 |
| 119 | 38 | "Concept units beat slot units in 90 of 137 decided pairs" | MATCH | results/part119/README.md:51 (119.5, 90/137 = 65.7%, held) | 2610 |
| 119 | 38 | "R values ranged from 0 to 0.058" (119.6/119.9 unreachable) | MATCH | score119.json unit_metrics R over 39 units: min 0.0, max 0.05835 | 242 |
| 119 | 38 | "48% by the triage rule; a reader marked 15% of a sample coherent" | MATCH | results/part119/README.md:99: 357/741 = 48.2%; 16/108 = 14.8% | 2610 |
| 119/119b/119c | 38 | "11 held, 10 failed, and 4 stopped across Parts 119, 119b, and 119c" | MATCH | PREDICTIONS.md: 119 = 2 held/8 failed (119.10 "failed by design wording"); 119b = 5/2/4 stopped; 119c = 4 held. Part READMEs agree row by row; score119.json gives 119.10 "not read" (disclosed at README.md:246) | 246 |
| 114 | 32 | "zeroing any of 20 attribution-ranked neurons alone took 'Q' off first place in 0 of 16" | MATCH | recomputed from results/part114/results114.json rows: max candidate flips 0/16 (20 candidates), base argmax Q 16/16, controls 0 flips | 2596, 254 |
| 114 | 32 | "zeroing the top pair also gave 0 of 16" | MATCH | results/part114/score.json pair.top2.not_q=0, n=16; all continuations "Qwen3.5**" | 2596, 254 |
| 114 | 32 | "114.1 and 114.4 failed, while 114.2 and 114.3 held" | MATCH | score.json verdicts; 114.3: L31 N4742 -0.73 vs strongest control -0.05 (score.json); PREDICTIONS.md agrees. Exploratory "16 of 20 beat all controls" recomputed = 16 | 2596 |
| 119b | 27 | "At alpha 0.4 the triage rule marked 33 of 741 pair answers broken (4.5%), against 357 of 741 at 0.8" | MATCH | score119b.json broken_answers_by_dose_and_class: 0.4 pair part86_triage_broken 33/741; 0.8 pair 357/741 | 240, 2612 |
| 119b | 27 | "Two random pushes at 0.8 broke 286 of 741" | MATCH | score119b.json 0.8 randpair 286/741 | 238, 2612 |
| 119b | 27 | "interaction share at 0.2 was 0.178 [0.177643] against a limit of 0.15 ... headline rule returned no reading" | MATCH | score119b.json value 0.17764258; headline "No registered headline reading; report the numbers." | 242, 2612 |
| 119b | 27 | "Five predictions held and two failed" (+4 stopped) | MATCH | score119b.json predictions: 5 held, 2 failed, 4 "stopped by Gate 3"; PREDICTIONS.md and part README agree | 2612 |
| 161 | 25 | "33 of 47 picked random directions ... against 25 of 37 picked neurons, a ratio of 1.04" | MATCH | recomputed from results/part161/score.json summary.sets: random 33/47, neuron 25/37, ratio 1.039; hits_at_pick>=12 agrees with every `reliable` flag | 2695, 80 |
| 161 | 25 | "interval of 0.77 to 1.45" | MATCH (README only) | results/part161/README.md:3; interval not stored in score.json, cold check (README:104) says bootstrap recomputed | 2695 |
| 161 | 25 | "unpicked paired directions reached a median of 0 of 24" | OVERCLAIM (minor) | paired hits at the 37 neuron memberships: median 0, 2/37 >=12 (recomputed); but part README:3 says "most of them not picked", so not all are unpicked | 2695 |
| 161 | 25 | "Regenerating the 18 reused neurons moved none across the bar" | MATCH | results/part161/README.md:22 (25/37 unchanged, no membership crossed 12/24); PREDICTIONS 161.6 held | 2695 |
| 161 | 25 | PREDICTIONS 161.1-161.6 all held; cold check present | MATCH | PREDICTIONS.md:3104-3113 = README table and score.json verdicts; README:104 | - |
| 115 | 21 | "zeroing top-200 attribution-ranked neurons took 'Q' off first place in 11 of 16" | MATCH | recomputed from results/part115/results115.json candidate rows: k=100 3/16, k=200 11/16, k=500 16/16 | 2598, 254 |
| 115 | 21 | "9 of those continuations began with Tongyi" | MATCH | results115.json generations["200"]: 4 "Tongyi**" + 5 "Tongyi Wanxiang**" = 9 of 11 non-Q | 2598 |
| 115 | 21 | "matched random sets flipped none" | MATCH | results115.json: max flips over all 90 random-set rows = 0 | 254 |
| 115 | 21 | "Predictions 115.2 and 115.6 failed; the others held" | MATCH | results/part115/score.json verdicts; 115.1 joint drop 5.18 < 8.42 (Part 114 sum recomputed -8.4164); PREDICTIONS.md agrees | 2598 |
| 115 | 21 | "zeroing the top 100 moved 'Paris' to first place ... at 500, its log-probability was -0.02" | MATCH | results/part115/ordinary_top115.json: k=100 " Paris" -0.316 first; k=500 -0.018 | 264 |
| 115 | 21 | Cold check line | MATCH | results/part115/README.md:81 | - |
| 118 | 21 | "At twice the dose all 234 neuron answers are incoherent" | MATCH | recomputed from results/part118/labels.jsonl (deduped by cell): alpha 1.6 neuron 234/234 not coherent; random 227/234 | 202, 2608 |
| 118 | 21 | "At half the selection dose few units show a persona ... does not separate neurons from random" | MATCH | labels.jsonl alpha 0.4: persona on 4 neuron units (9 answers) vs 3 random units (5 answers) | 202, 2608 |
| 118 | 21 | "on 43 sampled answers Haiku marked 14 and DeepSeek 4" | MATCH | recomputed: labels-ds.jsonl has 43 alpha-1.6 neuron cells; Haiku (labels.jsonl) marks 14, DeepSeek 4 | 2608 |
| 118 | 21 | "Two predictions held and six failed" | MATCH | PREDICTIONS.md 118.3, 118.4 held; 118.1, .2, .5, .1a, .2a, .5a failed; results/part118/README.md verdicts identical | 202, 208, 2608 |
| 140 | 17 | "Holding the push reached 48.3% on the selected units, and that selection favors holding" | MATCH | score.json table_per_arm_set A/P joint 0.483; results/part140/README.md:3,118 states P selected under held greedy | 2652 |
| 140 | 17 | "Releasing it after 8 tokens gave 5.8% and after 24 gave 37.9%" | MATCH | score.json C/P 0.058, D/P 0.379; arms defined results/part140/README.md:114 | 2652 |
| 140 | 17 | "Sampling gave 35.1%, a repetition penalty 43.3%, and norm-preserved pushes 0%" | MATCH | score.json B/P 0.351, E/P 0.433, F/P 0.0 | 2652 |
| 140 | 17 | "7 of 8 predictions failed, including the idea that a push only needs to start a persona" | MATCH | score.json predictions: only 140.7 holds; README:87; PREDICTIONS.md:2849-2856 verdicts agree with results/part140/README.md:78-85; cold check at README:131 | 2652 |
| 140 | 17 | "On the looping set the penalty cut repetition from 75.8% to 15.8%" | MATCH | score.json A/L repeat 0.758, E/L repeat 0.158 | 2652 |
| 153 | 17 | "hit their company on 63% of answers, against 11% for paired random directions, or 3% without one random direction that says Google on 20 of 24" | MATCH | score.json summary maker_mean_rate 0.629 (151/240), random_mean_rate 0.1125 (27/240); per-unit random L25N4457 20/24; (27-20)/216=3.2%. Raw substring recount without Part 86 broken filter gives 164/240 neuron, 27/240 random, so the filter only lowers neurons | 2679, 80, 172 |
| 153 | 17 | "Seven of ten reach 12 of 24, four if any pair with a hit is excluded" | MATCH | per_unit maker: 21,15,9,21,14,17,20,10,5,19 -> 7 at >=12; pairs with 0 random maker hits and neuron >=12: N7134, N1743, N2993, N5017 = 4 (results/part153/README.md:19) | 2679 |
| 153 | 17 | "L24 N5017 names Google on 20 of 24 maker and 7 of 8 identity phrasings" | MATCH | score.json per_unit L24N5017 maker 20/24, identity 7/8 | 2679 |
| 153 | 17 | README body: "Google appeared in the answer to 'Which company made you?' on 55 of 465 neurons and 56 of 465 random directions" (Part 148 cross-ref) | MISMATCH (minor, README body only) | recount over results/part148/answers.jsonl question_index 1, substring "Google": neuron 58, random 59; no filter tried (coherent, non-Alibaba) reproduces 55/56. "About one in eight" still holds (12.5-12.7%) | (part153 README:35) |
| 168 | 17 | "cut loop answers by 31% (interval 16% to 45%)" ; "cleared its 30% bar by 1.4 points" | MATCH | score.json predictions.168.3.own_relative_cut 0.3138; pooled none 0.1399 vs own 0.0960 (per_arm_ban) | 2707 |
| 168 | 17 | "falls below it without layer 24, on single neurons alone, or without any of three phrasings" | MATCH | extra168.txt:2-19: w/o L24 0.216, n+ alone 0.279, w/o phrasing 8/11/14 0.285/0.298/0.294 (all others >= 0.30) | 2707 |
| 168 | 17 | "A neighbour's top 50 cut -2.1%, and the gap ... 0.047, missed its 0.05 bar" | MATCH | score.json 168.5 -0.0213, 168.4 gap 0.0469 | 2707 |
| 168 | 17 | "raised loops by 16.5% where Weaver predicted a cut" | MATCH | score.json 168.W1 mass_relative_cut -0.1649 (W1 predicted 15-25% cut) | 2707 |
| 168 | 17 | "Of 188 low-k loops the own ban ended 101 and induced 42, so a third ... new"; "1 of 19" | MATCH | 0.13988x1344=188; extra168.txt:152-153 ended 101 induced 42 -> 42/129=32.6%; score transitions own.migration_count 1, share 0.0526 | 2707 |
| 168 | 17 | PREDICTIONS 168.1-168.8, W1-W3 verdicts; cold check | MATCH | PREDICTIONS.md:3191ff agree with score.json/README table; README:107 Cold check | - |
| 146 | 16 | "median pushed row has Spearman 0.77 to 0.82 with Instruct given an empty user message, against 0.50 to 0.61 with Base chat" | MATCH | score.json questions.*.references.{empty,base_chat}.union.arms.full.median_similarity: empty 0.779/0.774/0.791/0.815; Base chat 0.504/0.541/0.613/0.526. (Shuffled-question rows are as close or closer: 0.800/0.754/0.902/0.830) | 2663, 178 |
| 146 | 16 | "Gains over plain against Base chat are 0.097, 0.645, 0.158 and 0.180, ... median single unit's are 0.007, 0.526, 0.076 and 0.115" | MATCH | results/part146/README.md:13,17,88; pushed minus plain from score.json: 0.096/0.644/0.158/0.180 (rounding) | 2663 |
| 146 | 16 | "Random directions match at L4, 0.280 against 0.270, and fall to 0.004 at L15" | MATCH (scope note) | score.json predictions.146.7 mean_random_gain_L4 0.2800, L15 0.0036; 0.270 is Set F full-arm mean over L4+L5 (README:88,100), not an L4-only neuron value | 2663 |
| 146 | 16 | "the unregistered 0.78 with Base in Part 145 sits beside closer question-removed rows" | MATCH | Instruct-only set, q2: Base chat pushed 0.784 vs empty 0.848, no prompt 0.801 (results/part146/README.md tables) | 2663 |
| 117 | 14 | "concept persona on 'Who are you?' in 9 of 11 pairs and the slot persona in 3 of 11" (both +1, matched) | MATCH | recomputed from results/part117/labels.jsonl, q0 `matched:both+1`: c 9/11, s 3/11 | 2604, 198 |
| 117 | 14 | "each alone showed its persona in 11 of 11" | MATCH | recomputed: `matched:c+1 | c` q0 11/11, `matched:s+1 | s` q0 11/11 | 2604, 198 |
| 117 | 14 | "117.3-117.5 and 117.8 failed; 117.1, 117.2, 117.6, 117.7 held" | MATCH | results/part117/score.json verdicts; 117.8 recomputed slot 0/11; 117.6 random 1/132 labels on 1 answer; PREDICTIONS.md agrees | 2604 |
| 117 | 14 | "117.7 ... 66 concept-target labels; adding slot-target labels gives 1 of 132" | MATCH | recomputed `matched:both+2` coherent: c 0/66, s 1/66 -> 1/132 | 2604 |
| 117 | 14 | Cold check line | MATCH | results/part117/README.md:79 | - |
| 117b | 14 | "random-direction sample had only 15 of 33 answers coherent, below the registered 22-of-33 gate" | MATCH | recomputed from results/part117b/labels.jsonl q0 `r0..r2-1,s+1 | s`: coherent 15/33 (slot persona 3/33) | 2606, 198 |
| 117b | 14 | "unrelated-concept condition also fell below its coherence gate at 7 of 11" | MATCH | recomputed `u-1,s+1 | s` q0 coherent 7/11 (gate 9 of 11, 117b.7) | 2606, 198 |
| 117b | 14 | "slot-persona answers in 11 of 11 at +1, 2 of 11 at +0.85 and 0 of 11 at +0.5" | MATCH | recomputed `s+1 | s` 11/11, `s+0.85 | s` 2/11, `s+0.5 | s` 0/11 | 2606, 174, 198, 283 |
| 117b | 14 | "117b.3-117b.10 failed; 117b.1 and 117b.2 held" | MATCH | results/part117b/score.json verdicts; recomputed 117b.1 0/11, 117b.2 2/11; PREDICTIONS.md agrees | 2606 |
| 117b | 14 | Cold check line | MATCH | results/part117b/README.md:115 | - |
| 163 | 14 | "neurons still said Qwen less, 3.5% against 8.6%, ratio 0.41 ... unregistered" | MATCH | recomputed with part86/triage86.py + `\bQwen\b`, maker kind, broken and Alibaba/Tongyi/通义 excluded: neuron 19/539=3.52%, random 57/661=8.62%, ratio 0.41 (results/part163/README.md:3) | 2699, 84 |
| 163 | 14 | "Neuron answers also broke far more often, 21.4% against 5.2%" | MATCH | recomputed triage: 343/1600 vs 83/1600 | 2699, 84 |
| 163 | 14 | "moved the name and maker about equally ... test could not fail, as a cold check showed" | MATCH | README:9-10 ratios 1.061, 1.024; random rates 0.801/0.893 cap ratios at 1.25/1.12; README:108 cold check | 2699 |
| 163 | 14 | PREDICTIONS 163.1-163.5 | MATCH | PREDICTIONS.md:3135ff held x4, 163.5 failed = README table | - |
| 171 | 14 | "OpenAI, Anthropic and DeepSeek never passed 3 of 8, and only Google reached 8 of 8" (7 wordings) | MATCH | recomputed role_probe.jsonl+role_probe2.jsonl: 7 wordings x 8 prompts; best per maker google 8, openai 3, deepseek 3, anthropic 0 | 2713, 80 |
| 171 | 14 | "L20 N492, hit Google in 68.2% ... against a 2.3% rotated mean, and ten in 47.7% against 5.3%" | MATCH | results/part171/README.md:14,15,17,18 (score.json); r1 draws 0/0/6.8 -> 2.3; r10 2.3/4.5/9.1 -> 5.3; naive "google" in first40 recompute gives n1 30/44=68.2% | 2713, 80 |
| 171 | 14 | "At a hundred the picked set hit 9.1%, below random neurons at 12.9% and the rotated mean at 17.4%, with rotated draws at 6.8%, 9.1% and 36.4%" | MATCH | results/part171/README.md:16,19,20,35-37; means recompute (6.8+9.1+36.4)/3=17.4, (36.4+2.3+0)/3=12.9; 12.9% is a 3-draw mean whose draws are 36.4/2.3/0 (README:25) | 2713, 80 |
| 171 | 14 | "0.41 against 0.15 to 0.22, so hit order did not follow alignment" | MATCH | results/part171/README.md:3,72,75,83 (alignment.json 0.406 vs 0.154/0.212/0.222) | 2713 |
| 171 | 14 | PREDICTIONS.md verdicts | MATCH | PREDICTIONS.md:3241-3249 agree with results/part171/README.md:51-59 (171.6 "not read" vs README "untestable", same meaning); cold check README:267 | - |
| 119c | 13 | "float32 ... 0.153 at 0.2, 0.020 at 0.1 and 0.0045 at 0.05; bf16 0.162, 0.040 and 0.073" | MATCH | score119c.json 0.153414, 0.020094, 0.004547; bf16 0.161635 (results/part119c/README.md:34), 0.039749, 0.073401. My naive per-row recomputation from interaction-float32.jsonl did not reproduce the registered share definition exactly (e.g. 0.18 vs 0.153 at 0.2), so these rest on the score file | 242, 2614 |
| 119c | 13 | "all-position float32 real/control factors were 1.01, 1.04, and 1.10 ... 1.66 ... unregistered resolved-position subset" | MATCH | score119c.json 1.013412, 1.035772, 1.102566; 1.65775 from results/part119c/NUMBERS.md:104 | 242, 2614 |
| 119c | 13 | "All four predictions held" | MATCH | PREDICTIONS.md 119c.1-4 held; part README:19-22 | 2614 |
| 155 | 13 | "Company hits at +0.4, +0.8 and +1.2: L24 N5017 Google 1, 17, 8; L21 N7134 Baidu 0, 16, 0; L23 N1743 Huawei 1, 17, 0; L23 N2993 Microsoft 0, 20, 8, and none at any negative dose" | MATCH | recomputed from results/part155/answers.jsonl (company substring, Alibaba/Tongyi/通义 and triage86-broken excluded): all 12 cells equal; neuron arms 0 at all five negative doses | 2683 |
| 155 | 13 | "Mentions of Alibaba or Tongyi fall from 23 or 24 of 24 at +0.2 to at most 2 at +0.8" | MISMATCH (minor) | recomputed any-mention at +0.2: 24,24,24,24 (all four neurons; L24 N5017 every one of 24 answers names both Alibaba and Tongyi); +0.8: 2,0,2,0. Part README table results/part155/README.md:127 has L24 N5017 +0.2 = 23, true 24 | 2683 |
| 155 | 13 | "All seven predictions held" | MATCH | PREDICTIONS.md Part 155 rows 155.1-155.7 all held; cold check results/part155/README.md:154 | 2683 |
| 156 | 13 | "Hits at each step: L24 N5017 Google 14, 14, 17, 19, 19, 16, 17; ... Microsoft 0, 1, 8, 18, 12, 6, 9; random directions 0 or 1" | MATCH | recomputed all 28 neuron cells from results/part156/answers.jsonl, identical; random max 1 (L23 N2993 random at 1.0, 1.1) | 2685 |
| 156 | 13 | "Each neuron reaches 12 of 24 at more than one dose" ; "156.2 and 156.3 failed" | MATCH | doses >=12: Google 7, Baidu 3, Huawei 4, Microsoft 2 (0.8, 0.9); 156.2 only Google beats 0.8 (19>17); 156.3 Google 14 at 0.5. PREDICTIONS.md 156.1-156.5 agree; cold check README:84 | 2685 |
| 165 | 13 | "26% broken at one neuron, 27% at a random four, 15% at sixteen, about 9% from 64 up, on 96 unselected slots" | MATCH | results/part165/score.json rates.broken: n+ 0.2565, k4 0.2708, k16 0.1484, k64 0.1003, k256 0.0924, k1024 0.0859, full 0.0951 (768 rows/arm = 96 slots x 8) | 2703, 84 |
| 165 | 13 | "with sign making no difference" | MATCH | n+ 0.2565 vs n± 0.2539; 165.4 ratio 1.01 (0.78-1.30), results/part165/README.md:13 | 2703 |
| 165 | 13 | "last-layer change and median KL at the first pushed step were close" | MATCH | effect-tables.md:6 rel-change ratio 1.0496; README:3 median KL 0.134 vs 0.125 (1.07) | 2703 |
| 165 | 13 | "Without layer 21 the ladder's rank test would fail its bar" | MATCH | 165.2: -0.857 overall, -0.714 without L21 vs bar <= -0.8 (README:11, PREDICTIONS.md 165.2) | 2703 |
| 165 | 13 | reading: "concentration and natural-state alignment failed ... single-token readout passed only as a bound" | MATCH | results/part165/reading/README.md bars R1-R3,R5 fail, R4 pass | 2703 |
| 165 | 13 | PREDICTIONS 165.1-165.7 verdicts; cold check | MATCH | PREDICTIONS.md:3155ff held x4, failed x3 = results/part165/README.md:9-15; Cold check line present | - |
| 116 | 12 | "direct-write top 100 took 'Q' off first place in all 16 phrasings" | MATCH | recomputed from results/part116/dla/results115.json candidate rows: k=50 4/16, k=100 16/16 | 2602, 256 |
| 116 | 12 | "activity ranking first reached 8 flips at 500 (11 of 16)" | MATCH | recomputed results/part116/act/results115.json: 0 flips k<=200, k=500 11/16 | 2602, 256 |
| 116 | 12 | "matched random sets gave no flips"; "All five predictions held" | MATCH | max random-set flips 0 in both act/ and dla/ files; score.json verdicts all held; 116.4 act k=50 -3.4056 vs Part 115 gradient -6.96 (recomputed); PREDICTIONS.md agrees. 116.3 (39 shared) not recomputed: rank_*.pt needs torch, not installed | 2602 |
| 116 | 12 | "the gradient ranking did so at 200" (cross-ref to Part 115) | MATCH | Part 115 results115.json k=200 11/16, k=100 3/16 | 256 |
| 116 | 12 | Cold check line | MATCH | results/part116/README.md:75 | - |
| 139 | 12 | "130 of 130 full-arm answers ... 20.0% with only the prompt pushed and 0% with only the answer pushed" | MATCH | recomputed results/part139/results.jsonl stage A set R: full 130/130, prompt-only 26/130, answer-only 0/130 | 2650, 178 |
| 139 | 12 | "Base 1.5%, SFT 98.5%, DPO 91.5%, with random directions rising alongside the neurons" | MATCH | recomputed stage E: 2/130, 128/130, 119/130; random rows 0/34, 34/34, 29/34 (results/part139/README.md:89-91) | 2650, 178 |
| 139 | 12 | "On unselected units and random directions the share peaks at 17.2%" | MATCH | results/part139/README.md:102 (L4 a=1.2 random 11/64) | 2650 |
| 139 | 12 | "44% on ' sorry' after a forced 'I'm' where the plain model puts 0.026%" | MATCH | results/part139/README.md:137 (0.4412 vs 0.000262) | 2650 |
| 139 | 12 | "garbled prompts drew no apology, 0 of 50" | OVERCLAIM | only random token IDs were 0/50; recomputed stage D: shuffled letters 1/50, dropped chars 3/50, random words 2/50 (results/part139/README.md:72-79) | 2650 |
| 139 | 12 | PREDICTIONS.md verdicts 139.1-139.13 | MATCH | PREDICTIONS.md:2864-2876 agree with results/part139/README.md:11-23; cold check README:165 | - |
| 160 | 12 | "picked at 0.6 had a median peak of 0.7 ... picked at 1.0 a median of 1.0, so 160.1 held" | MATCH | recomputed from results/part160/score.json summary.sets peak_dose: 0.6 set median (0.6+0.8)/2=0.7 (n=12), 1.0 set median 1.0 (n=7) | 2693 |
| 160 | 12 | "leaving out units whose random pair also hits gives 0.8 and 0.8, the 0.6 set is half Tencent, and the cap dropped 18 of its 30" | MATCH | results/part160/README.md:3 (0.8/0.8); sets 0.6 companies Tencent 6/12; 30 picked, 12 kept (README:18) | 2693 |
| 160 | 12 | "41 against 30 at 0.6 and 6 against 7 at 1.0" ; "L21 N8461 named Google on 20, 23 and 24 of 24" | MATCH | results/part161/README.md:30-31 table; results/part160/README.md:35 | 2693 |
| 160 | 12 | PREDICTIONS 160.6 "failed: 5 of 12 and 4 of 7" | MISMATCH | score.json sets 1.0 companies: Google 5, Tencent 1, Microsoft 1 -> 5 of 7; part README:16 and :55 say 5/7. Verdict (failed) agrees | - |
| 174 | 12 | "28 of the 80 layer-matched signed directions led with Google and none with any other name" | MATCH | recomputed scored_directions.jsonl: u100d 80 rows, 28 supported; README:1,3 (score.json) all 28 Google-led, 0 other-led | 2719, 80 |
| 174 | 12 | "an exact one-sided 95% bound of 7.2% per draw" | MATCH | 1-0.05^(1/40)=0.0722; results/part174/README.md:15 | 2719, 80 |
| 174 | 12 | "Google counts run smoothly from 0 to 27, so the 35% says where a floor of 10 cuts a continuum" | MATCH | scored_directions.jsonl google_claim_count min 0 max 27 (73 non-null); README:24,31 | 2719 |
| 174 | 12 | "The 40 flat directions broke nearly every answer" / "median of 5 of 44 broken" (line 80) | MATCH | recomputed f100d: median broken 44/44, 38 of 40 at >=40 broken, 0 supported; u100d median broken 5.0 | 2719, 80 |
| 174 | 12 | PREDICTIONS.md verdicts 174.1-174.W3 | MATCH | PREDICTIONS.md:3290-3302 agree with results/part174/README.md:64-76; cold check README:117 | - |
| 105 | 11 | "opposite-direction attention swap lost 5 of 10 early slot personas" | MATCH | recomputed from results/part105/labels-attnEXT_all.jsonl (q0, any pass shows_persona): lost p92-02,05,07,42,50 = 5/10; score105.json lost.attnEXT_all | 2572 |
| 105 | 11 | "compared with 7 of 10 under SFT attention" | MATCH | score105.json lost.attnSFT_all = 7 ids | 2572 |
| 105 | 11 | "both runs lost one or no later personas" | MATCH | recomputed later lost = [p92-32] (1/9); SFT 0/9 (score105.json later_lost) | 2572 |
| 105 | 11 | "105.1 failed ... 105.2 and 105.3 held" | MATCH | score105.json predictions: failed/held/held (18/21 coherent, 17/18 concept); PREDICTIONS.md:2389-2391 agree | 2572 |
| 105 | 11 | Findings: "5 of 10 early versus 1 of 9 later, Fisher p=0.09" | MATCH | recomputed one-sided Fisher = 0.0913 | 222 |
| 103 | 10 | "SFT's full attention ... 13 of 21; SFT's MLPs ... 10 of 21, with 17 and 16 coherent" | MATCH | results/part103/score103.json runs.attnSFT_all slot_who [13,21], coherent [17,21]; mlpSFT_all [10,21], [16,21] | 2568 |
| 103 | 10 | "Prediction 103.2 failed and 103.1, 103.3-103.6 held" | MATCH | score103.json predictions; PREDICTIONS.md:2369-2374 agree | 2568 |
| 103 | 10 | "corrected reading calls the attention evidence suggestive" | MATCH | results/part103/README.md "attention suggestively ... p = 0.070" (recomputed McNemar 7 vs 1, two-sided 0.0703; MLP 9 vs 0 = 0.0039) | 2568 |
| 164 | 10 | "bare Qwen 6.4% ... against 8.1% for paired random directions, ratio 0.79, interval 0.33 to 1.57" | MATCH | results/part164/score.json rates.qwen_rate neuron 0.0639; random over the 98 pairs with eligible neuron answers 0.0811 (all-100 random mean is 0.0794); 164.1 estimate 0.7887, interval [0.333,1.569] | 2701, 84 |
| 164 | 10 | "the direction held on 13 of 16 phrasings and the interval contains 0.41 and 1" | MATCH | results/part164/README.md:3 (13 of 16, sign p=0.02; interval includes 0.41, 0.70, 1) | 2701 |
| 164 | 10 | "Neuron answers broke 3.5 times as often, 27.0% against 7.8%, in all eight layers" | MATCH | score.json broken_rate 0.270 vs 0.0781, ratio 3.456; recomputed with triage86: neuron 432/1600=27.0%; README:53 "gap holds in all eight layers" | 2701, 84 |
| 164 | 10 | PREDICTIONS 164.1-164.5 | MATCH | PREDICTIONS.md Part 164 = README:13-17; cold check README:82 | - |
| 164/165 | 10 | Findings: "broke answers more often, 21.4% against 5.2% and 27.0% against 7.8%. At equal delivered dose a single neuron broke 26% ..." | OVERCLAIM (mixed rules) | 21.4/5.2 and 27.0/7.8 are Part 86 triage-only; Part 165's 26% uses triage plus loop override. Recomputed Part 164 under score165.primary_broken: neuron 523/1600 = 32.7%, random 198/1600 = 12.4% (as results/part165/README.md:31 says); Part 165 triage-only n+ is 17.6% | 84 |
| 173 | 10 | "name Google on 22, 26 and 36 of 44 at abs(d) = 0.5, none broken, one sign per draw" | MATCH | results/part173/cell_table.jsonl canonical: u100d_d173 -0.5 google 22 broken 0; d1173 +0.5 26/0; d2173 +0.5 36/0 | 2717, 80 |
| 173 | 10 | "n3 Meta on 35 of 44, and n1 Tencent on 27" | MATCH | cell_table.jsonl: n3 -0.5 meta 35 (broken 2); n1 -0.5 tencent 27 (broken 1) | 2717, 80 |
| 109 | 9 | "cold activities question under the push showed no persona (0 of 6)" | MATCH | results/part109/labels.jsonl arm A: 0/6 shows_persona (recomputed) | 2582, 190 |
| 109 | 9 | "after ... own persona answer, the reader marked all six in persona, though only the dog described activities in character" | MATCH | labels.jsonl arm B 6/6; answers.jsonl arm B: p92-37 "long walks in the park ... fetch", others restate identity ("I am a cat.") | 2582, 190 |
| 109 | 9 | "Prediction 109.3 failed; 109.1 and 109.2 held" | MATCH | arm C 2/6 (<3 needed) recomputed; results/part109/README.md predictions table; PREDICTIONS.md:2434-2436 agree | 2582 |
| 175 | 9 | "0.30 ... median 29.5 of 44 broken, 1 of 40 signed directions supported" | MATCH | recomputed part175/scored_directions.jsonl dose 0.3: 1/40 supported, median broken 29.5 | 2721, 80 |
| 175 | 9 | "At 0.225 ... 27 of 40 were supported, all Google-led, with a median of 2 broken ... At 0.15 it was 1 of 40" | MATCH | recomputed: 0.225 27/40 supported, 27 google_led, 0 other_led, median broken 2.0; 0.15 1/40 | 2721 |
| 115b | 8 | "gradient ranking needed 500 neurons to reach 8 flips in all openings" | MATCH | recomputed from results/part115b/results115b.json rankings.gradient candidate rows: k=200 2/2/3, k=500 13/16/16 (plain/quote/bold) | 2600, 260 |
| 115b | 8 | "direct-write needed 100 for plain text and 50 for quote or bold" | MATCH | rankings.direct: plain k=50 7, k=100 15; quote k=50 12; bold k=50 13 | 2600, 260 |
| 115b | 8 | "115b.3 and 115b.7 failed; the other five held"; "'Chat' is not on its registered model/company list" | MATCH | results/part115b/score.json verdicts; recomputed gradient plain k=500 non-Q first words Chat 11, ChatBot 2 (11 of 13); max random flips 1/16 (gradient quote); PREDICTIONS.md agrees | 2600, 260 |
| 115b | 8 | Cold check line | MATCH | results/part115b/README.md:87 | - |
| 122 | 8 | "12 of 18 concept units and 6 of 21 slot units met the read rule in third-person text" | MATCH | recomputed from results/part122/score122.json unit_frames frame T: concept 12/18, slot 6/21 | 2620 |
| 122 | 8 | "concept units sit well above the shuffled control (12 of 18 against 5 of 39)" | MATCH | score122.json shuffled_controls T: 5/39 read; Fisher two-sided 12/18 vs 5/39 = 8.2e-05 (README:43 "about .0001") | 204, 2620 |
| 122 | 8 | "does not hold on the labels both readers agree on (5 of 8 against 6 of 21, p = .20)" | MATCH | unit_frames T agreed_concept 5/8, agreed_slot 6/21; Fisher recomputed 0.197 | 204, 2620 |
| 122 | 8 | "Prediction 122.2 missed by one ... Three predictions held and four failed" | MATCH | 6/21 vs threshold <=5 (results/part122/README.md:18); PREDICTIONS.md 122.1/.3/.7 held, .2/.4/.5/.6 failed, same in part README | 2620 |
| 124 | 8 | "Fourteen of 39 units ranked their own phrase ... top three" | MATCH | recomputed from score124.json focal.T: 14/39 reads (concept 10/18, slot 4/21) | 206, 2624 |
| 124 | 8 | "shuffled median of 4 and 0 of 10,000 shuffles reaching 14" | MATCH | score124.json permutations.all_units.null_counts: median 4.0 (mean 4.47), 0/10000 >= 14 | 206, 2624 |
| 124 | 8 | "10 of 18 against 4 of 21, Fisher p 0.024 ... 5 of 8 against 8 of 28, p 0.107" | MATCH | Fisher recomputed 0.0237 and 0.1072; DeepSeek 5/8, 8/28 from results/part124/README.md:44 (DS kinds not recomputed) | 206, 2624 |
| 124 | 8 | (verdicts) 124.1-124.4 held | MATCH | PREDICTIONS.md Part 124 all held; score124.json predictions status True x4; part README:15-18 | 2624 |
| 134 | 8 | "The registered count peaked at 11 of 24 at doses 0.8 and 0.6, the best of three cells" | MATCH | results/part134/score134.json conditions test pair: 0.9+0.6 p92-09 8/24, p92-37 2/24, 0.8+0.6 p92-09 11/24 | 2640 |
| 134 | 8 | "One reader, after seeing the counts, judged 8 or 9 of the 11 as a dog speaking"; verdicts | MATCH | results/part134/README.md:3,65; PREDICTIONS.md:2790-2796 agree with README:91-97 and score134.json (134.7 false only); cold check README:103 | 2640 |
| 142 | 8 | "median margin differs by 0.056 on Haiku's sort, n=364, and by 0.004 on the 43 units DeepSeek also calls personas" | MATCH | score/score142.json predictions.142.2 difference 0.05576, n 364; README:7,53 DS subset 0.004 p=0.909 n=43. My band-weighted recompute over all 386 units gives 0.0471 = README's pilot-included 0.047 | 2655, 178 |
| 142 | 8 | "zeroing a concept unit ... gives a next-word change that is more its own than a slot unit's" | OVERCLAIM | Holds only on the primary universe (pilot units excluded) at 0.056 vs a 0.05 bar; fails with pilot included (0.047, README:7) and on DS personas (0.004); 142.3 shows sorted units vs matched poles differ by only 0.004, so the margin is not unit-specific | 2655 |
| 142 | 8 | "Sorted units barely differ from matched poles, 0.004" | MATCH | README:54 142.3 difference 0.004 (PREDICTIONS.md 0.0038) | 2655 |
| 142 | 8 | "The first effects run was void, 92% NaN with no error; the rerun was clean" | MISMATCH (wording) | 17,152/18,672 = 91.9% NaN (README:98) MATCH; but the same line says the run "stopped twice with UR_RESULT_ERROR_UNKNOWN", so "with no error" is only true of the NaN rows themselves (README:102) | 2655 |
| 145 | 8 | "the median push lowers ' an' from 94% to 27%" / "' sorry' rises the most for its starting probability, among 106 words ... and 1,128" | MATCH | results/part145/README.md:5 (0.94 -> 0.27); score.json 145.2 and 145.7 residual percentile 1.0 (106 and 1,128 words). Note raw change on q2 ranks ' unable' 3.03 just above ' sorry' 3.01 (README:73); claim is about residual, so OK | 2661 |
| 145 | 8 | "2.9 log-odds above ... matched temperature"; "partial correlation of 0.49"; "alike, 0.99"; "Six of seven predictions held" | MATCH | score.json 145.3 2.871 [2.500, 3.264]; 145.4 0.4913; 145.6 0.9871; verdicts: only 145.1 does not hold | 2661, 178 |
| 145 | 8 | "Not registered: on the second question the pushed ranking correlates 0.78 with Base and the plain ranking 0.26" | MATCH | explore.json q2_floor1e-05 pushed_vs_base 0.7844, plain_vs_base 0.2633; recomputed from words-instruct-chat-q2.json + words-base-chat-q2.json (pushed = plain + median_change, 1,128 words): 0.784 / 0.263. Number is in explore.json only, not stated in results/part145/README.md | 2661 |
| 157 | 8 | "left 18 of 24 answers still containing a banned spelling ... others wrote Alipay, Alibab or 通義千問" | MATCH | results/part157/score.json gates.gate3.leaks: 18 level_1 leaks; route table results/part157/README.md:51-53 (Alibab 10, Alipay 15, 通義千問 7) | 2687 |
| 157 | 8 | "Gate 3 failed, so all five predictions are untestable" ; ladder by list order | MATCH | gate3.pass false; PREDICTIONS 157.1-157.5 untestable = README table; README:25 tie-break; cold check README:134 | 2687 |
| 167 | 8 | (no front-page entry) "Part 167, the first attempt, stopped at its plain-answer gate with no verdict read" | MATCH (but no own entry) | `grep '\*\*\[Part 167\]' README.md` empty; list jumps 166 (2705) -> 168 (2707). results/part167/score.json gate2 false, readings_stopped true; PREDICTIONS 167.1-167.7 "Not read: Gate 2 failed"; README:17 "Cold check: not needed" | 2707 |
| 108 | 7 | "personas appeared in 14 of 71 SFT, 40 of 71 DPO and 64 of 71 Instruct answers" | MATCH | results/part108/score108.json models.S/D/I all_slot_shown = 14/40/64 | 2580, 216 |
| 108 | 7 | "forward moves ... lost 43 (SFT scale) and 32 (DPO scale) of the 64 units shown there" | MATCH (minor) | score108.json FP lost 43 of 64; FR lost 32 but denominator 63 (1 unlabelled, p104-28) | 2580 |
| 108 | 7 | "DPO checkpoint 38 of 62 ... 2 x Instruct - DPO 31 of 63 ... SFT 13 of 64 ... 2 x Instruct - SFT 21 of 64" | MATCH | score108.json shown/unlabelled: D 38 (2 unl.), FR 31 (1), S 13 (0), FP 21 (0) | 216 |
| 108 | 7 | "losses going back and forward overlapped no more than chance (p=0.69)" | MATCH | recomputed from lost sets: S&FP both 34, back-only 17, fwd-only 9; one-sided Fisher 0.687 | 216 |
| 108 | 7 | "Prediction 108.3 and 108.5 failed; other predictions held" | MATCH | results/part108/README.md predictions table; PREDICTIONS.md:2423-2428 agree. Note: 108.5 result cell omits FP concept 14/18 (78%, also below 85%) | 2580 |
| 110 | 7 | "four maker neurons' mean weight move ... 0.00272 against same-layer random draws at 0.00282" | MATCH | recomputed mean of results/part110/weights110.json neurons[L6 N753,L7 N1169,L11 N8868,L29 N776].move_mean = 0.0027187; set_test.uniform.draw_mean 0.0028226, p_more 0.585 | 2588, 82 |
| 110 | 7 | "their top-20 token readouts shared 16–20 tokens between models" (subject: the four maker neurons) | MISMATCH | recomputed base∩chat top20: L6 16, L7 18, L11 17, L29 16 -> four makers 16–18; 16–20 holds only over all six read neurons (L27 N3325 20, L24 N3949 17) | 2588 |
| 110 | 7 | "All five predictions held, but no Base generation was run" | MATCH | weights110.json verdicts 110.1-110.5 held, readout_gate passed; PREDICTIONS.md agrees | 2588 |
| 110 | 7 | "L7 N1169 and L11 N8868 put Chinese and China first"; "L27 N3325 put Google first in both" | MATCH | weights110.json readouts: L7 base/chat top3 '作为中国',' Chinese','是中国'; L11 ' Chinese',' China'; L27 google_rank 1 both | 258 |
| 111 | 7 | "L7 N1169 had 8 China-family tokens and L11 N8868 had 7 ... L27 N3325 had 11 Google-family tokens" | MATCH | recomputed regex (results/part111/jlens111.json families, case-insensitive) on j_lens.top20: 8, 7, 11 | 2590, 82, 258 |
| 111 | 7 | "null rates of 0.7%, 1.0% and 0.0%" | MATCH | jlens111.json null_rates j_lens: L7 china 0.007, L11 china 0.010, L27 google 0.000 | 2590, 82 |
| 111 | 7 | "111.3 failed because L6 N753's top 20 had no AI-family token; the other three held" | MATCH | recomputed L6 j_lens ai hits 0 (logit lens has 4: ' AI',' Claude',...); verdicts 111.1/2/4 held; PREDICTIONS.md agrees | 2590, 82, 258 |
| 151 | 7 | "The positive control reached 2 of 20 windows against a registered 5, so all five predictions are withheld" | MATCH | Recount on results/part151/windows.json L22N2742 top (before+mark+after, /alien | extraterrestrial | ufo | martian/i): 2 of 20; score.json predictions all "withheld" (README and PREDICTIONS.md:2994-2998 say "untestable", same meaning) | 2673 |
| 151 | 7 | "Part 151b repeats the run on 10,000 documents" | MISMATCH | Part 151b ran on documents 2,000-9,999 only: 8,000 new documents (results/part151b/README.md:3; windows.json docs=8000). Part 151 README:29 has the same "whole 10,000-document sample" wording | 2673 |
| 162 | 7 | "V4.1 Flash reads with Claude: about 1 on Claude's markers across 14 variants" | MATCH | results/part162/score.json positions.deepseek.claude 1.080; variant table results/part162/README.md:5-20 range 0.939-1.118 | 2697 |
| 162 | 7 | "Its GPT side, 0.2507 against a bar of 0.25, rests on punctuation, markdown and em dashes, and is -0.15 on word markers alone" | MATCH | score.json deepseek.gpt 0.25074 [0.154,0.343]; README:3 word-only -0.15, markdown 57%, em dash 31% of numerator | 2697 |
| 162 | 7 | "Qwen3.5-4B reads with Claude too and OLMo sits in between, though Qwen's prediction was untestable" | MATCH | qwen P_C 0.732 / P_G -0.052 (n=96), 162.2 untestable; olmo 0.488 / 0.524; PREDICTIONS 162.1-162.5 agree with score verdicts; cold check README:84 | 2697 |
| 162 | 7 | "A public chart says DeepSeek V4.1 Flash matches neither" | UNVERIFIABLE | external chart, not in repo | 2697 |
| 102 | 6 | "SFT attention in layers 0-3 ... 13 of 20 ... DPO ... 17 of 20 and the mid-training end in 3 of 21" | MATCH | results/part102/score102.json attnSFT_L0-3 [13,20], attnDPO_L0-3 [17,20]; Part 100 score100.json attnB_L0-3 [3,21] | 2566 |
| 102 | 6 | "Prediction 102.1 failed; 102.2-102.5 held" | MATCH | score102.json predictions; PREDICTIONS.md:2359-2363 agree | 2566 |
| 102 | 6 | "A later opposite-direction check (Part 105) showed that a same-size attention change can also remove these answers" | OVERCLAIM | Part 105 swapped attention in all 16 layers, not layers 0-3; results/part105/README.md Limits: "swapped in all 16 layers, so this does not test Part 102's layers-0-to-3 swap" | 2566 |
| 104 | 6 | "On 43 new slot units, swapping SFT attention lost 9 of 23 early and 6 of 20 later; an equal-distance opposite swap lost 8 of 22 early and 7 of 20 later" | MATCH | results/part104/score104.json runs.attnSFT_all early_lost 9/23, later 6/20; attnEXT_all 8/22, 7/20 (p104-21 unread). 43 = shown of 50 new slot units | 2570, 222 |
| 104 | 6 | "Predictions 104.1, 104.3 and 104.4 failed; 104.2 was not read" | MATCH | score104.json predictions (Fisher 0.381; coherent 32/42; DPO later lost 3/20); PREDICTIONS.md:2380-2383 agree | 2570 |
| 104 | 6 | "these data do not decide dependence versus fragility" | MATCH | results/part104/README.md Reading + cold-check correction; recomputed SFT/EXT overlap 5 (p104-06,08,14,29,59) vs 5.4 chance | 2570 |
| 106 | 6 | "median persona-sentence log-probability under its own push was -8.7 nats at SFT, -8.8 at DPO and -8.6 at Instruct" | MATCH | recomputed from results/part106/lp-{sft,dpo,instruct}.jsonl, 21 slot units, unit=neuron alpha 0.8: -8.74/-8.82/-8.57 | 2574 |
| 106 | 6 | "DPO lowered the no-push and random-push baselines" | MATCH | recomputed medians SFT->DPO: none -31.76->-35.08, random -22.39->-25.76 | 2574 |
| 106 | 6 | Findings (19 corrected units): "-9.4 at SFT to -10.0 at DPO; gap 14.6 ... 20.7; no-push -32.9 to -36.7" | MATCH | recomputed excluding p92-33 ("I'm Dr.") and p92-37 ("Hello!"): push -9.44/-9.97, gap 14.61/20.66, none -32.86/-36.66 | 224 |
| 106 | 6 | "Predictions 106.1 held and 106.2-106.5 failed" | MATCH | results/part106/README.md registered predictions; recomputed 106.1 gap 14.25 (>=2); PREDICTIONS.md:2397-2401 agree | 2574 |
| 109b | 6 | "14 of 18 OLMo 2 1B concept neurons were labelled in persona (7 coherent, 3 with self-claims), against 0 of 6 slot neurons" | MATCH | recomputed from results/part109b/labels.jsonl arm A (retry rows used for 2 parse errors): 14/18, 7 coherent, 3 self-claim; part109 arm A 0/6. Fisher two-sided 0.0016 matches README | 2584 |
| 109b | 6 | "109b.2 failed; 109b.1, 109b.3 and 109b.4 held ... including the dog's weak 'good girl' reading" | MATCH | arm B 14/18 (<15), arm C 8/18 (<=8), p92-08 arm A shows_persona true, self_claim false; PREDICTIONS.md:2461-2464 agree | 2584 |
| 136 | 6 | "pilot failed the known-unit check, 7 of 24 against 12, and the off-target check, 0.021 to 0.023 against 0.02" | MATCH | results/part136/README.md:12 (7 of 24 vs 12), :42 (0.0209, 0.0228, 0.0220 vs 0.02) | 2644 |
| 136 | 6 | "No prediction was read" | MATCH | results/part136/README.md:66-71 all "not read"; PREDICTIONS.md:2814-2819 all "withdrawn" (consistent); cold check README:73 | 2644 |
| 143 | 6 | "0.024% on ' sorry' in the plain model, 0.09% with the push on the answer only, 17% on the prompt only, 45% on both" | MATCH | Recomputed from results.jsonl, Set R (neurons + random, n=130): plain 0.000239, answer-only 0.000925, prompt-only 0.1689, full 0.4507 | 2657 |
| 143 | 6 | "After a forced 'I', ''m' reads 14%, 19%, 38% and 45%" | MISMATCH | Set R combined medians from results.jsonl: plain 13.9%, answer-only 18.9%, prompt-only 37.9%, full 43.9%. The 45% is the neuron-only row (0.4525, README:37); random 42.6%. The other three numbers are combined-set values, so full should read 44% | 2657 |
| 143 | 6 | "on the log-odds scale the two sides add, unit by unit, with a median gap of -0.02 and a middle half of -1.0 to 0.9" | MATCH | Recomputed full - (prompt + answer - plain) log-odds for ' sorry', Set R: median -0.024, IQR -1.00 to 0.93 (numpy linear); README:73 -0.024 (-1.013 to 0.934) | 2657 |
| 151b | 6 | "on 8,000 new documents, with the alien control at 11 of 20" | MATCH | windows.json docs 8000; recount L22N2742 top with same regex: 11 of 20; score.json alien_control_windows 11 | 2675 |
| 151b | 6 | "17 of L19 N4979's 20 top windows mark ' and' or ' or', against 0 or 1 on the other units, and 17 of L24 N8542's mention awards or nominations, against 0" | MATCH | Recount marks in windows.json: 17 / 0 / 0 / 1 / 1; score.json award_top L24N8542 17, others 0 (my simple /award | nominat/ regex gives 15; Method's rule is wider) | 2675 |
| 151b | 6 | "Own-list hits beat other lists on 2 of 5 units against a registered 3, so the reading ... is withdrawn" | MATCH | score.json beats_others true only for L24N6835 (2 hits) and L24N8542 (6); 151b.1 failed | 2675 |
| 169 | 6 | "cut loop answers again, by 38% (interval 26% to 49%), and the cut stays between 34% and 42% without any one layer" | MATCH | pooled loops none 338, own 210 -> 1-210/338 = 37.9% (results/part169/README.md:24-25, extra169.txt); 169.1 interval 25.8-48.6%; README:157 leave-one-layer 34.1-42.4% | 2709 |
| 169 | 6 | "80% of the cut falls in the top two tenths of rows by own mass, which hold 48% of the loops" | MATCH | extra169.txt:42-43: 103/128 = 80.5% of net cut; 163/338 = 48.2% of none loops | 2709 |
| 169 | 6 | "free, gave 338 loop answers against 338 with no ban (interval -6% to +7%) ... split ... 189 to 190" ; "Free ended 89 loops and induced 89" | MATCH | README:24,26 (338/338); 169.3 0.0% (-6.1%-7.0%); README:70 189 vs 190; extra169.txt:16 free ended 89 induced 89 | 2709 |
| 169 | 6 | "raised loop answers from 338 to 621, which Weaver predicted in size" | OVERCLAIM (minor) | 169.W2 registered a floor, not a size: rise >= one third of own's net cut (bar 42.7); observed 283, 6.6x the bar (README:47). W1 "opposite of their prediction" MATCH (-0.102) | 2709 |
| 169 | 6 | PREDICTIONS 169.1-169.8, W1, W2 | MATCH | PREDICTIONS.md Part 169 = README:38-47 verdicts; cold check README:179 | - |
| 100 | 5 | "replacing only layers 0-3 ... 3 of 21 while 16 of 21 stayed coherent; replacing layers 12-15 left the persona in 16 of 20" | MATCH | results/part100/score100.json attnB_L0-3 slot_who [3,21], coherent [16,21]; attnB_L12-15 [16,20] | 2562 |
| 100 | 5 | "early-versus-late half comparison and first-quarter predictions failed; coherence and concept controls held" | MATCH | results/part100/README.md: 100.1, 100.2 failed, 100.3, 100.4 held (coherent min 13/21, concept min 13); PREDICTIONS.md:2339-2342 agree | 2562 |
| 100/99 | 5 | Findings: "from 19 of 21 to 3 of 21, while 13 of 21 stayed coherent" (Part 99) | MATCH | results/part99/README.md:17,20 (inst_none 19/21; inst_attnB 3/21, 13/21 coherent) | 222 |
| 120 | 5 | "5 held and 3 failed in Part 120" | MATCH | PREDICTIONS.md Part 120 rows 120.1-5 held, 120.6-8 failed; results/part120/README.md agrees | 246 |
| 120 | 5 | "At dose 0.1 the cross interaction is a median 0.224 ... (interval 0.162 to 0.263)" | MATCH | results/part120/README.md:22 (120.3): c=0.22395, interval 0.16155-0.26264 (score120.json bootstrap_c_0.1_real; not recomputed, NPZ inputs not in repo) | 2616 |
| 120 | 5 | "at the three lowest doses it is 0.275 to 0.280" / "about 0.28 of the self-term scale" | MATCH | results/part120/README.md dose table: real c 0.279775 (0.0125), 0.278299 (0.025), and 0.275-0.280 range per README:7 | 244, 2616 |
| 120 | 5 | "random second push ... about the same (0.238 at 0.1)" | MATCH | results/part120/README.md:7 control c 0.23767; factor 1.061 | 244, 2616 |
| 120 | 5 | "grows about four times per doubling, the mark of a square term" | MATCH | 120.1 log2 ratios 2.06258 (0.025->0.05), 2.16322 (0.05->0.1), i.e. x4.2, x4.5 | 244 |
| 121 | 5 | "showed the slot persona in 2 pairs against 1 without it, and an unrelated concept ... at the lower dose (3 pairs)" | MATCH | recomputed from labels-ds.jsonl + keys, q0: matched+c+0.5 {p01,p05}, s+0.85 {p10}, unrelated+u+0.25 {p01,p10,p11}, unrelated+u+0.5 {p01} | 2618 |
| 121 | 5 | "fell from 11 of 11 pairs at +1 to 1 of 11 at +0.85" | MATCH | recomputed: s+1 11/11, s+0.85 1/11 (DeepSeek) | 174, 2618 |
| 121 | 5 | "Five predictions held and two failed" | MATCH | results/part121/README.md:76-82 121.1-2 failed, 3-7 held; PREDICTIONS.md agrees | 2618 |
| 126 | 5 | "Spearman rho was -0.023, with a 95% interval of -0.37 to +0.33" | MATCH | recomputed from score/score126.json 126.1 evidence (attention_mass_by_head vs absolute_drop_by_head, 32 heads): rho -0.02346; Fisher-z interval tanh(atanh(r) +/- 1.96/sqrt(29)) = -0.369 to +0.328 | 2628 |
| 126 | 5 | "three of four predictions failed" | MATCH | PREDICTIONS.md 126.1-3 failed, 126.4 held; results/part126/README.md agrees | 2628 |
| 128 | 5 | "carriers ... match on the fire side for 6 of 11, once neurons for 0 or 1 of 16 ... both sides together, 8 or 9 of 11 against 1 of 16" | MATCH | recomputed from labels-ds/luna.jsonl (pushed arm, side score >=2) with units128.json classes: fire carrier 6/11 both readers, once 0/16 (DS) 1/16 (Luna); either side 9/11 and 8/11 vs 1/16 | 2632 |
| 128 | 5 | "Part 132 repeats it on 189 fresh ones" | UNVERIFIABLE | Part 132 is registered only (PREDICTIONS.md:2764, PART132-DESIGN.md); no results/part132/ and no verdicts; reads as done | 2632 |
| 144 | 5 | "plain model puts 0.58% on ' sorry' on SFT and 0.024% on Instruct, and the median push raises that by 3.3 log-odds on SFT and 5.2 on Instruct" | MATCH | Recomputed from sft-chat.jsonl / instruct-chat.jsonl (Set F full arm, n=256): plain 0.5756% / 0.0239%; median shift 3.344 / 5.203 | 2659 |
| 144 | 5 | "On Instruct without the chat frame the plain model puts 5.5% there, and the push adds 0.5. Six of nine predictions held" | MATCH | instruct-plain.jsonl plain 5.4605%, median shift 0.533; verdicts held 144.1/2/3/6/7/9 = 6 | 2659 |
| 144 | 5 | "Base in the chat frame, DPO and Instruct reach pushed levels within about 0.15 log-odds of each other" | MATCH | recomputed pushed level (plain log-odds + median shift): Base chat -3.211, DPO -3.063, Instruct -3.136; max pairwise gap 0.148 | 2659 |
| 150 | 5 | "4 of 5 put their lens words into answers more often than all four random directions of their layer, and 3 of 5 on Base" | MATCH | results/part150/README.md:67-71 chat +0.8 registered rates 3.07/1.22/0.00/2.23/11.26 vs random max 0.75/0/0/0/0 -> 4 of 5; score.json 150.7 held, 150.13 held (3 of 5) | 2671 |
| 150 | 5 | "lens hits on L24 N8542 are 11.26 per 100 words, 4.18 for whole words, with 3 of 6 answers broken" | MATCH | score.json rates.instruct.L24N8542.lens["0.8"] = 11.26; README:71 exact 4.18, broken 3/6 | 2671 |
| 150 | 5 | "Seven predictions failed, and 150.11 held on read guesses from a different model" | MATCH | failed: 150.1-150.5, 150.8, 150.10 = 7; README:5 (Qwen3-1.7B atlas guesses) | 2671 |
| 158 | 5 | "893 tokens banned ... took ` Alibaba` from 23 answers to 1 and changed every answer, and `**Alibaba` rose from 3 to 23" | MATCH | results/part158/ban.json banned_count 893; recomputed: Part 157 level_0 ` Alibaba` 23, `**Alibaba` 3; Part 158 plain 1 and 23; 24/24 plain answers differ from level_0 | 2689 |
| 158 | 5 | "All 24 still name the family, so Gate 3 failed, the three predictions are untestable" | MATCH | results/part158/README.md:7; PREDICTIONS 158.1-158.3 untestable = README table; cold check README:65 | 2689 |
| 101 | 4 | "left slot personas in 7 of 21 ... layer 0, 11 ... layer 1, 12 ... layer 2 and 15 ... layer 3, against 19 of 21 without a swap" | MATCH | results/part101/score101.json runs.attnB_L0..L3 slot_who 7/11/12/15 of 21; reference_inst_none 19 | 2564 |
| 101 | 4 | "All six predictions held; the single-layer losses overlap, and answers stayed coherent" | MATCH | score101.json predictions all held; drops 12+8+7+4=31 > 16; coherent 17-18/21; PREDICTIONS.md:2348-2353 agree | 2564 |
| 107 | 4 | "DPO's update was 23-342 times smaller than SFT's" | MATCH | recomputed from results/part107/where107.json rel_change: attn 0.10241/0.004429=23.1, attn_norm 0.005292/1.548e-5=341.8 (final norm unchanged at DPO, excluded) | 2578, 220 |
| 107 | 4 | "51% of changed attention and MLP elements moved by one bf16 step" | MATCH | results/part107/score107.json 107.2 dpo_one_step_share_of_changed 0.5114 (changed_frac 0.6117) | 2578, 220 |
| 107 | 4 | "107.1 and 107.2 held, while 107.3 failed; ... no claim about where DPO acts" | MATCH | score107.json verdicts and reading; 107.3 R ratio 1.029, 10/21; PREDICTIONS.md:2407-2409 agree | 2578 |
| 123 | 4 | "no slot vector showed a persona at any dose, 0 of 630 answers" | MATCH | recomputed from part123/labels-ds.jsonl + keys (unit kinds from Part 92 Haiku labels via part122 score): vector arm slot units 0/630 persona; concept units 14/540 | 2622 |
| 123 | 4 | "the matching neurons did (20 of 21 readable at full dose)" | MATCH | recomputed: neuron arm q=1, "Who are you?", slot units 20/21 persona+coherent; score123.json gate 2 readable_shows 20 | 2622 |
| 123 | 4 | "Two gates failed, so no prediction is read" | MATCH | results/part123/README.md:24 gates 3 and 7 failed; PREDICTIONS.md 123.1-6 "not read" | 2622 |
| 137 | 4 | "DeepSeek marked 5 of 16 non-task answers against a registered 6, Luna 6 of 16; each neuron alone 0 and random partners at most 1" | MATCH | results/part137/README.md:3,19 (alone 0/16; random partners 1,1,0,0); verdicts PREDICTIONS.md:2825-2829 agree with README:11-15; cold check README:97 | 2646 |
| 152 | 4 | "hit its top windows 2.26 times in 20 on the 2,409 with usable lists, against 0.11 for shuffled pairs in the same layer" | MATCH | Recomputed with part152/score152.py's registered rule from part152/lens.json + results/part152/windows.json.gz: n=2409, mean 2.2615, 719 units >=3 (152.2); score.json shuffle_mean 0.1095, p=0.001 | 2677 |
| 152 | 4 | "72% of those hits come after the fired token" | MATCH | score.json top_hit_location_counts 1516 / 2509 / 1423 -> (2509+1423)/5448 = 72.2% | 2677 |
| 152 | 4 | "Concept ... 2.11 against 0.44, and 0.56 against 0.26 once words shared with the J list are removed" | MATCH | score.json 152.4 concept 2.105 (n 247), slot 0.440 (n 184); persona_not_in_j concept 0.565 (n 232), slot 0.261 (n 184) | 2677, 172 |
| 152 | 4 | Findings: on the 245 units "1.786 ... against 0.226 shuffled ... slot units 0.262" | MATCH | results/reader-correction-2026-11-03/README.md:93-94 | 170 |
| 154 | 4 | "the seven Google neurons fall from 96 to 46 of 168 hits ... against 93 to 102 for four matched neurons ... Baidu, Huawei and Microsoft neurons stay at 53 and 52 of 72" | MATCH | recomputed from results/part154/answers.jsonl (Part 153 rule + triage86): Google none 96, L27N3325 46, matched 96/93/102/97; others 53 -> 52 | 2681 |
| 154 | 4 | "Leaving out L22 N6997 gives 46 of 88, which misses the registered bar of half" ; "random direction fell from 19 to 14 and also failed" ; "Only 3 of the 7 ... (154.5) failed" | MATCH | L22 N6997 none 8, zeroed 0 -> 88 and 46 (46 > 44); random 19 -> 14 (154.4 failed, half 9.5); 154.5 3/7 failed (results/part154/README.md:16-21,52); PREDICTIONS 154.1-154.7 agree; cold check README:134 | 2681 |
| 166 | 4 | "in the row's own top 200 54.7% of the time, in other slots' 1.0%, and in the plain model's 15.3%" | MATCH | results/part166/score.json 166.6 content_word_own_rate 0.5470, shuffled 0.0103; extra166.txt:81 plain 0.1531 | 2705 |
| 166 | 4 | "Of the 0.166 extra loops per answer at low k, 0.115 are own-word loops" ; `subject` on all 8 phrasings ; "share 46 of 96 slots" | MATCH | results/part166/README.md:3,19 (own-content per answer n+/n±/k4 11.8/11.5/14.8% vs full/iso 0.8/1.7% -> 0.1145); README:33 `subject` 8 answers; extra166.txt "identical n± and n+ direction slots=46/96" | 2705 |
| 166 | 4 | PREDICTIONS 166.1-166.7 | MATCH | PREDICTIONS.md Part 166 = README table (held 1,5,6,7; failed 2,3,4); cold check README:51 | - |
| 176 | 4 | "30 of 40 ... at 0.225, all Google-led, median of 1 broken; support 20 or 21, 30, 20, 10 and 0 of 40" | MATCH | recomputed part176/scored_directions.jsonl rungs: 0.2 20 supported + 1 bounded, 0.225 30 (30 google_led, median broken 1), 0.25 20, 0.275 10, 0.3 0 | 2723, 80 |
| 176 | 4 | "In 22 of the 34 directions with any support, the top supported dose lies within 0.025 of where answers start to fail" | MATCH | 34 directions with top_supported_rung; abs(gap)<=0.025 for 22 (21 with gap in [0,0.025], one at -0.0125) | 2723, 80 |
| 106b | 3 | "pushed SFT answer sentence fell by a median 1.6 nats at DPO and the unpushed sentence by 5.5" (13 units) | MATCH | recomputed from results/part106/score106b.json per_unit (13 units): -1.573, -5.513 | 2576 |
| 106b | 3 | "106b.1 and 106b.2 held; 106b.3 failed, ... confounded by length" | MATCH | score106b.json verdicts; 106b.3 all 6 persona-minus-SFT-answer negative; PREDICTIONS.md:2415-2417 agree | 2576 |
| 109c | 3 | "together at +1 showed a dog in 5 of 7 answers, against 3 and 1 alone" | MATCH (scope note) | recomputed from results/part109c/labels.jsonl: both+1 5, concept+1 3, slot+1 1. But dog-and-coherent: both+1 2 vs concept+1 3 (README table) | 2586 |
| 109c | 3 | "at +0.5 no answer showed a dog, and negative pushes gave none" | MATCH | labels: concept/slot/both +0.5 all 0; all -0.5/-1/-2 and concept_slot2-1 conditions 0 | 2586 |
| 109c | 3 | "109c.4 and 109c.5 failed; the other five held ... none of the +2 combined answers was coherent" | MATCH | results/part109c/README.md predictions table; both+2 coherent 0/7 recomputed; PREDICTIONS.md:2470-2476 agree | 2586 |
| 113 | 3 | "at +1 showed its 'unnamed being' persona only on 'Who are you?'"; "at +3, 0 of 7 answers were coherent" | MATCH | recomputed from results/part113/labels.jsonl: neuron+1 shows_persona only q0 (1/7); neuron+3 coherent 0/7; random 0/70 | 2594, 196 |
| 113 | 3 | "113.1 and 113.6 failed; the other four held"; "lens top words were 'simply' and 'just'" | MATCH | results/part113/score.json verdicts; results/part113/jlens/jlens113.json plus_top20 starts " simply"... " just", plus_name_hits []; PREDICTIONS.md agrees | 2594, 196 |
| 125 | 3 | "coherent persona in 2 of 29 units with the pushed cache and 3 of 29 with a twin ... every turn gave 11 of 29" | MATCH | score125.json turn5_counts arm2 2, arm3 3, arm1 11. Raw DeepSeek labels (persona and coherent, no cap rule) give 3, 4, 15; difference is the 120-token cap rule, disclosed at results/part125/README.md:17-18 | 176, 2626 |
| 125 | 3 | "The pushed persona persists only through the generated text being re-read." | OVERCLAIM | Part 125 shows the text-only twin also mostly loses it (3/29; 125.2 "text alone carries the persona" failed, score125.json); the cache null rests on one dose/29 units plus Part 31 row 31.5 (PREDICTIONS.md:1353, failed = cache adds nothing). Data support "adds nothing beyond the text", not that text carries it | 176 |
| 125 | 3 | (cold check) | MISSING | no "Cold check:" line and no COLD-CHECK file in results/part125/ | - |
| 129 | 3 | "Two gates failed, so the reading is not confirmed ... skipped part of its Chinese documents" | MATCH | results/part129/README.md:3 (Gates 1 and 3 failed), :19 (490 zh lines skipped); PREDICTIONS.md 129.x verdicts equal README:39-45 (held x4, not read x3, all marked post hoc) | 2634 |
| 138 | 3 | "batch 8 finished without a spill on every run, at 0.34 to 0.91 answers/s; batch 16 spilled on one run of two" | MATCH | results/part138/README.md:3-5,13; ladder.json batch-8 0.908; repeats8.json 0.342-0.377; first ladder safe at 16, repeats.json run spilled at 16 | 2648 |
| 138 | 3 | "under the registered 1 answer/s"; verdicts | MATCH | 138.3 failed (README:13); PREDICTIONS.md:2835-2838 agree with README:11-14; cold check README:50 | 2648 |
| 172 | 3 | "18 clean Google-family answers and 4 broken ones at dose 0.75, which fired Weaver's falsifier, and its registered lead ... held at dose 1 only" | MATCH | results/part172/README.md:3,34 (score.json); 172.W1 failed at README:34, PREDICTIONS.md:3267 | 2715 |
| 172 | 3 | "rotated basis ... gave 44 of 44 with none broken at 0.75, and the next-ranked single neuron gave 44 of 44 at 0.5"; "Ten of the 12 control sets were run at dose 1 alone" | MATCH | results/part172/README.md:3,42,131 (r100_d2171 44/44 0 broken at 0.75; n2 44/44 0 broken at 0.5; 10 of 12 controls single-dose); verdicts PREDICTIONS.md:3255-3267 agree with README:22-34 | 2715 |
| 127 | 2 | "A+B rescued 8 of 38 cells at dose 0.707, against 1 for matched A+C and 2, 2 and 3 for random controls" | MATCH | recomputed from main/score127.json rescues.0.707 by_pair: pair_ab 8/38, matched_ac 1, random_0/1/2 2, 2, 3 (at 0.5: 3/38 vs 0) | 2630 |
| 127 | 2 | (verdicts / cold check) | MATCH / MISSING | score127.json predictions held, failed, held, held, failed = PREDICTIONS.md and part README; no "Cold check:" line in results/part127/README.md | 2630 |
| 130 | 2 | "3,681 poles are, against a mean of 294.5 under rotation" | MATCH | overlap130.json triple 3681, triple_null_mean 294.489; triples130.json has 3681 entries | 2636 |
| 130 | 2 | "in a sample of 150, a reader judged the names the same or closely related in 49" | MATCH | recomputed from labels-ds.jsonl (last valid label per id/arm): same 49/150, shuffled 0/149; score130.json ds_same 49 | 2636 |
| 130 | 2 | "which reproduces Sauers' figure" | UNVERIFIABLE | external figure (3,681 and 294 per results/part130/README.md:5); not in repo | 2636 |
| 133 | 2 | "Median per-neuron r ... 0.895, 0.995 and 0.999 on web text for parent to SFT, SFT to DPO and DPO to Instruct" | MATCH | results/part133/main/score133.json predictions/133.1 web_median_r 0.8954, 0.9952, 0.9989 | 2638 |
| 133 | 2 | "SFT moves them most"; verdicts | MATCH | 0.895 is lowest of three; PREDICTIONS.md:2781-2784 agree with results/part133/README.md:11-14; cold check README:33 | 2638 |
| 135 | 2 | "1,109 neurons had personas on both poles, against a rotation null of 1,096.6" | MATCH | results/part135/README.md:3,5 (65.9th percentile of null) | 2642 |
| 135 | 2 | "A reader called most pairs unrelated, about the same as shuffled pairs ... can rule out the 10-point effect" | MATCH | results/part135/README.md:28,40,46 (92.4% unrelated; same 6.8% vs 8.4% shuffled, diff CI about -6 to +3); verdicts PREDICTIONS.md:2802-2808 agree with README:38-44 | 2642 |
| 149 | 2 | "median P(' sorry') across 24 questions goes from 0.121 ... to 0.822 with every word shuffled on Instruct, and from 0.08 to 0.08 on Base" | MATCH | Recomputed from instruct.jsonl/base.jsonl (median over q of exp(mean logp_sorry over seeds)): Instruct real 0.1210, shuf-all 0.8221; Base 0.0803 -> 0.0791 | 2669, 178 |
| 149 | 2 | "149.2 failed and its reading is withdrawn; 149.3 held at 18 of 24 after a scorer fix made on seeing the output" | MATCH | 0.822/0.121 = 6.79x < 10; recomputed 149.3 per-question rho rounded to 9 places: 18 of 24 >= 0.8 (three exactly 0.8); README:30 discloses 15 of 24 pre-fix | 2669 |
| 149 | 2 | "correlated 0.43 and 0.48 against a registered 0.5, with random directions matching neurons" | MATCH | README:25-26 rho 0.431, 0.482; baseline-removed neuron/random 0.150/0.202 and 0.184/0.301 with overlapping CIs (README tables) | 2669 |
| 147 | 1 | "closer to Base at the same question than at the others on three of four questions, by -0.098, 0.169, 0.419 and 0.420; plain Instruct passes the same test, and on the first question the push does worse than plain" | MATCH | score.json 147.1 question_differences -0.0984/0.1692/0.4187/0.4201, 3 meet 0.05; plain differences 0.404/0.132/0.317/0.312 (README:15, table) | 2665 |
| 147 | 1 | "91.0% to 99.6% of units are closer to Base in Jensen-Shannon divergence than plain Instruct is" | MATCH | score.json 147.3 shares 0.957/0.910/0.996/0.961. (Not on front page: wrong-question JS control shows no push-specific gain on q1/q2, README tables) | 2665 |
| 159 | 1 | "a median 9.3 logits sure of the refused token. On 21 of 24 answers it refused `yi` and wrote `y`, and 16 answers still spell the word" ; row refusing `ibaba` -> "**Alphabet** (Google)" | MATCH | median of the 24 sorted margins at results/part159/README.md:24 = (9.25+9.375)/2 = 9.3125; README:22 (21/24, 16 rows listed, row 21 Alphabet quote) | 2691 |
| 159 | 1 | "only L7 N1169 was on at the veto, on 14 of 24 rows, and it moved about as much as matched peers. L27 N3325 ... did not rise (159.6 failed)" | MATCH | README unit table: L7 N1169 on 14, R 0.901; 159.6 0 rows, failed; PREDICTIONS 159.1-159.6 agree; cold check README:91 | 2691 |
| 159 | 1 | "whole layers are still 1.8 to 6.5 times as far apart as two different phrasings" | MATCH | README:50-54 both-Lab / null: 0.515/0.079=6.5, 0.531/0.092=5.8, 0.503/0.149=3.4, 0.241/0.133=1.8, 0.248/0.139=1.8 | 2691 |
| 159 | 1 | "every one of 46,080 recorded units unchanged on the banned step" | MATCH (number not stated in part README) | 5 recorded layers x 9,216 MLP channels = 46,080; Gate 3 held with all 24 rows matching (score.json gates[2]); README:3 calls it a recording check, as the front page does | 2691 |
| 170 | 1 | "raised identity loops from 25.0% to 43.8% under a full-size push" | MATCH | results/part170/README.md table P8 n+ dose 1: b0 25.0%, b50 43.8% (384 rows); 170.3 held on point estimate | 2711 |
| 170 | 1 | "identity loops fall from 8 to 3 of 104 over all prompts, and rise by 3.3 points only on prompts that did not loop" ; "At 200 tokens ... 9.6 points on identity prompts and 19.8 on ordinary questions" | MATCH | results/part170/allprompts170.txt: identity b0 8/104, b50 3/104, b200 18/104 (17.3-7.7=9.6); Q96 b0 5/96, b200 24/96 (25.0-5.2=19.8); eligible identity rise (0+3)/92=3.3 pp (README:73) | 2711 |
| 170 | 1 | "another question added 1.1 points" ; "releasing the ban after step 5 showed no detectable rise (-6.0 to +2.1 points)" | MATCH | 170.7 other50 +1.1 (0.0-5.6); README:108 / cold/COLD.md:13 rel50 interval -6.0 to +2.1 pp | 2711 |
| 170 | 1 | PREDICTIONS 170.1-170.8, W1, W2 | MATCH | PREDICTIONS.md Part 170 agrees with README verdict table (README:57-66); cold check README:253 | - |

## Cross-references checked

- 115 -> 114: "8.42 their single drops add up to" — MATCH (sum of results/part114/score.json candidates mean_dlogodds = -8.4164).
- 115 -> 114: unzeroed baseline 6.74 batched / 6.70 unbatched — MATCH (score.json base_mean_logodds 6.7415; part115 none-row gate diff 0.046).
- 116 -> 115: gradient k=50 mean change -6.96; gradient reaches 8 flips at 200 (11/16) — MATCH (results/part115/results115.json candidate k=50 -6.96, k=200 11).
- 115b -> 114/116: rankings reused (gradient = Part 114 attribution, direct-write = Part 116) — not recomputable without torch (.pt files); flip tables recomputed from results115b.json MATCH.
- README.md:170 reader correction -> 112: q_name slot 104 of 190 on 245 units vs 154 of 329 full; random non-self 6.3% vs 8.0% — MATCH (recomputed from results/reader-correction-2026-11-03/part112/v2_another_ai/units.json joined to results/part112/score.json: 104/190, 62/980 = 6.33%; full 148/1860 = 7.96%). Note: the reader-correction README:48 reports full-atlas 112.3 as "0.800 against 0.532" (scorer 57-unit non_ai group, 44/55), while the registered 78-entry scope is 56/76 = 0.737; same verdict.
- README.md:170 -> 117: "117.2 moves from held to failed on the 322 units, 7 of the 10 surviving pairs" — not checked (time); source results/reader-correction-2026-11-03/part117/v2_all.
- 117b README -> 117: "removed the slot's self-description on 'Who are you?' in all 11 pairs" — MATCH (Part 117 `matched:c-1,s+1|s` q0 0/11; 117b recomputed `c-1,s+1|s` 0/11).
- 146 -> 145: "unregistered 0.78 with Base in Part 145". MATCH (results/part145/explore.json q2 pushed_vs_base 0.784; recomputed 0.784).
- 153 -> 148: ten same-company neurons, one answer each. MATCH (part148 score.json same_company count 10, same unit list as part153 per_unit). Google base rate 55/56 of 465: MISMATCH, true 58/59 (Problem 5).
- 153 -> 148: "Part 148 had measured all ten at batch 64". Consistent with part148 README Gate 2 (batch 64).
- 143 -> 139: Part 139 stored plain P(" sorry") 0.000262. MATCH (results/part139/README.md:18,137).
- 151 -> 151b: "10,000 documents". MISMATCH, 8,000 (Problem 2).
- 152 -> 151b: controls 11 of 20 and 17 of 20. MATCH (part151b score.json alien 11, and_or L19N4979 17, award L24N8542 17).
- 148 / 152 -> reader correction (Findings README.md:84, 170): 81/48, 107/112, 89 vs 184, 1.786/0.226/0.262. MATCH (results/reader-correction-2026-11-03/part148/v2_another_ai/score.json; reader-correction README:78-80,93-94).
- 147 -> 146: uses Part 146's stored rows and gates. Gate numbers (212/212, 400/400) MATCH results/part146/README.md:5.
- 119 (README.md:238) -> 119b: random two-push breaks 286/741 at 0.8. MATCH (score119b.json randpair 0.8 = 286/741).
- 119 front-page (README.md:242) -> 119b Gate 3 share 0.177643. MATCH (score119b.json).
- 119c -> 119b: Gate 3 share 0.177643 (results/part119c/README.md:53). MATCH.
- Finding at README.md:174 -> 117b "11 of 11 at +1, 2 of 11 at +0.85" (Haiku) and 121 "11 of 11 fell to 1 of 11" (DeepSeek). 121 recomputed MATCH; 117b +0.85 = 2/11 and +0.5 = 0/11 per results/part117b/README.md:44,42 MATCH (+1 11/11 not opened).
- Finding at README.md:174 -> 118 "all 234 neuron answers are incoherent" at twice the dose. MATCH.
- README.md:172 -> 124 "Fourteen of 39 ... shuffled median of 4". MATCH.
- README.md:176 -> Part 31 row 31.5 (PREDICTIONS.md:1353): cache and re-encoded cache give the same returns, failed. MATCH as cited.
- 123 -> 92b: "DeepSeek kept only 8 of the 18 units as concepts" (results/part123/README.md:36). Consistent with part122 score agreed_concept n=8. MATCH.
- 128 -> 132: "repeats it on 189 fresh ones". UNVERIFIABLE (not run; see Problems 2).
- 130 -> Sauers' figure (external). UNVERIFIABLE.
- 126 -> Part 83 drops / 20 mattering heads (results/part126/README.md:112). Not checked against part83.
- README.md:80 (Findings) -> Part 153: "63% of answers against 11%". MATCH (results/part153/README.md:15: 0.629 vs 0.113).
- README.md:80 -> Part 161: "33 of 47 reliable against 25 of 37". MATCH (recomputed).
- README.md:80 -> Parts 155/156: "dose ladders peaking near the picking dose". MATCH: in Part 156 the peaks fall at 0.7 to 0.9 for picks made at 0.8.
- Part 155 README ("Against Part 153 at +0.8") -> Part 153: 20, 15, 14, 17. MATCH (results/part153/README.md:29-34).
- Part 163 entry (2699) -> Part 161: "reliable neurons' answers said Qwen less". MATCH (results/part161/README.md: 98/600 neuron vs 339/792 random, `\bQwen\b`).
- Part 164 entry (2701) -> Part 163: ratio 0.41 inside 0.33-1.57. MATCH.
- Part 165 entry (2703) -> Part 164: "single neuron breaks answers more often". MATCH. Part 165 README:31 gives Part 164 primary-rule rates of 32.7% and 12.4%, MATCH (recomputed). README.md:84 mixes the rules, see Problems 5.
- Part 168 entry (2707) -> Part 167: "stopped at its plain-answer gate with no verdict read". MATCH.
- Part 169 entry (2709) -> Part 168: "Part 168's rise did not recur". Part 168 mass rise was -16.5% (score.json W1), and Part 169's free arm gave 0.0%. MATCH.
- Part 166 README:3 says "A ban on held-out directions is Part 167", while the front page (2705) points to Part 168. Both are consistent, because 167 stopped at its gate.
- Part 160 entry (2693) -> Part 161 ladders: 41/30 and 6/7 picks. MATCH (results/part161/README.md table).
- 171 front page and README.md:80 -> 173 (Meta 35/44, Tencent 27/44 by picked sets): MATCH, results/part173/cell_table.jsonl (n3 -0.5 meta 35; n1 -0.5 tencent 27).
- 174 -> 173 ("Meta and Tencent, led by picked sets in Part 173"): MATCH, same source.
- 175 -> 174 ("Part 174's 20 flat draws", broke at 0.5): MATCH, part174 f100d median broken 44/44.
- 176 -> 175 (lead at 0.225): MATCH, part175 0.225 27/40 supported.
- 172 -> 171 (L20 N492 is Part 171's n1): MATCH, results/part171/README.md:144.
- 171 README:119 -> 73 (Google 36 of 192 target, 8 of 192 with L27): MATCH, results/part73/score73.json 73.1 target_google 36/192, 73.2 google 8/192; README.md:78 "8 and 17 of 192 against 36" also MATCH (73.4 google 17/192).
- 134 -> 137 (repeat with cell fixed): MATCH, results/part137/README.md:3 uses p92-08+p92-09 at 0.8+0.6, Part 134's peak cell.
- 140 -> Part 86 dose / Part 112 units: not re-checked (only cited as method, no number).
- 102 (front page, README.md:2566) -> 105: "same-size attention change can also remove these answers" -> OVERCLAIM on scope (Part 105 is all-16-layer, see Problem 1).
- 102 -> 100: mid-training end in layers 0-3, 3 of 21 -> MATCH (results/part100/score100.json attnB_L0-3).
- 102 README -> 103: whole SFT 8 of 21 -> MATCH (results/part103/score103.json whole_SFT [8,21]).
- 103 (front page) -> 105: fragility to same-size attention changes -> MATCH (both all-16-layer swaps on Part 92's units).
- 105 -> 103: attnSFT_all lost 7 of 10 early -> MATCH (results/part105/score105.json lost.attnSFT_all, matches Part 103's by-unit set p92-04,05,07,10,13,20,42).
- 101 -> 100: layers 0-3 block 3 of 21, 16 coherent, 13 of 17 concept -> MATCH (score100.json).
- Findings README.md:222 -> 99: 19 of 21 to 3 of 21, 13 of 21 coherent -> MATCH (results/part99/README.md:17,20).
- 106 README -> Part 92: 8 of 21 slot at SFT, 14 of 20 at DPO -> consistent with results/part103/README.md ("the same as Part 92's per-checkpoint pushes ... Part 92's 14 of 20"); Part 92 raw not rechecked.
- 108 README -> Part 92: 8 of 21, 14 of 20, 21 of 21 -> not rechecked against results/part92 (see Skipped).
- 109b -> 109: slot 0 of 6 arm A, 5 of 6 arm B coherent -> MATCH (recomputed from results/part109/labels.jsonl: B in persona and coherent 5).

## Not reached or not recomputed

- 116.3 "39 shared neurons" and the ranking identities in 115b: rank_act.pt / rank_dla.pt / attr114.pt need torch (not installed); not recomputed.
- Reader-correction 117.2 "7 of the 10 surviving pairs" cross-ref (README.md:170) not recomputed.
- Part 112 exploratory "as an AI" split (122/691, 23/125) and the capital/poem table not recomputed (not front-page claims).
- Nothing skipped. Not recomputed from raw rows, with stored score.json values used instead: Part 142 primary (pilot-excluded) 0.056 and the DS-subset 0.004 (my all-unit band-weighted recompute gave 0.047 = README; my persona_v2 subset gives n=46, not 43, because the scorer's unique-match rule is not reproduced); Part 146 and 147 similarity values (raw index files not reprocessed); Part 150 rates; Part 145's 145.3/145.4 from index files.
- 119c/120: I did not independently recompute the interaction share and c. A naive per-row formula over interaction-float32.jsonl came out close to the registered values but not exact (0.18 vs 0.153 at 0.2), so these values rest on score119c.json and score120.json. Part 120's NPZ inputs are not in the repo.
- 124: the DeepSeek-label split (5/8 vs 8/28) was taken from the README; I only recomputed its Fisher p.
- 127: I did not recompute the per-pair "each rescued one cell at a different dose" eligibility claim.
- 129: I checked only the gates and verdict consistency. I did not recompute the r medians.
- 126: the Part 83 cross-reference was not checked.
- Bootstrap intervals were not recomputed (161 0.77-1.45; 168 16-45%; 169 26-49%; 164 0.33-1.57, the last checked against score.json only). They were taken from score.json/README.
- Part 168/169 loop counts were checked against score.json/extra*.txt and not rescored from answers.jsonl.
- Part 162 and 170 positions were checked against score.json/README and not recomputed from raw answers.
- Part 159's 46,080 figure is inferred as 5 layers x 9,216; the part README does not state it.
- Part 140's "Holds on the eager kernel (Part 187)" line, as instructed.
- Part 171: my naive "google" substring recompute from answers.jsonl reproduced n1 68.2%, n100 9.1%, r100 draws 6.8%/36.4%, but differed on n10 (45.5% vs 47.7%) and r100_d1171 (15.9% vs 9.1%). The scorer's hit rule (first 40 tokenizer tokens, Google-family names) is stricter or different than my check. I relied on score.json for those values. Its cold check (COLD.md) says it reproduced every per-arm rate.
- Part 172 and Part 176 Spearman (+0.02, -0.32 to +0.35): checked against README/numbers176.json citations only, not recomputed.
- Did not recompute Part 92's own stage counts (8/21, 14/20, 21/21) cited by Parts 106 and 108.
- Part 107 group ratios other than attn and attn_norm taken from score107.json rather than recomputed (embed 60.1, layer norms 70.4, MLP 25.3, unembedding 30.5 are in score107.json and consistent with where107.json).
- Part 100-103 slot counts taken from score*.json, not re-derived from labels (Part 105 and 109/109b/109c were re-derived from labels).

Also not done: Part 92's stage counts (8/21, 14/20, 21/21), which Parts 106 and 108 cite, were not rechecked against
results/part92. Anything that needs torch (`.pt` rankings in Parts 114, 115b and 116) was not recomputed.
