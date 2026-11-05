# Front-page audit, 2026-11-04

Run: 2026-11-05, 05:50 to 06:10 UTC, by Claude Code on a schedule. The audit was read-only on the repo: this file is the
only change. No model was run.
Base: origin/main `fa68418` ("Merge public main (Part 188) into Part 189 work").

**Scope.**
- "The short version" (README.md:24-34, 11 items).
- The 12 most recently registered parts in PREDICTIONS.md: 177, 178, 181, 180b, 183, 184, 185, 186, 187, 188, 189 and
  189b. Ten of them have a front-page entry (README.md:2725-2743). 180b and 183 have none (problem 15).
- For each part: its eager-kernel notes elsewhere on the page, its PREDICTIONS.md rows, and every number it cites from
  another part (25, 66, 18, 18b, 78, 78b, 79, 82, 84, 84b, 140, 148 and others).

**Method.** Four independent auditors (subagents of this session) shared one brief. Each claim was recomputed from the
rawest committed file available: answer/label jsonl, scored rows, npz arrays, using the repo's own scoring functions
where they exist. The lead then rechecked the most serious findings directly: 66.2 FAILS in
`results/part183/part66-eager/score.txt`; the 19/20 control cell in results/part184/README.md:53-54; Part 188's five
failed rows; Part 82's c3 at 0.349 against 0.368; Part 97 being OLMo 2 1B; and the PREDICTIONS.md cell-count defect,
which the lead found. scipy is not installed, so the Fisher and binomial tests were computed exactly by hand. The
Part 188 probes were refitted with a small logistic regression of the auditor's own.

**Totals over 126 claim rows.**

| Status | Count |
| --- | ---: |
| MATCH | 106 |
| MISMATCH | 3 (S1f stale sentence, S8c Part 109 paraphrase, P188g "four of nine failed") |
| OVERCLAIM | 13 |
| UNVERIFIABLE | 1 in full (P189c), 2 in part (P188i, P189a: no Qwen tokenizer, and one unit not in the committed arrays) |
| Note | 1 |

The OVERCLAIM rows are S1e, S4c, S5b, S7e, S7f, S8d, S9c, P177d, P178g, P183b, P183e, P183f and P184i. Every
cross-referenced "published" number matched its source part, with two exceptions: Part 66's 66.2 (problem 1) and Part
18's control cell (problem 6). The Part 148 figures cited by Part 189b were confirmed: 348/465 = 74.8%, 175/465 =
37.6%, the six verdicts 148.1 F, .2 F, .3 H, .4 F, .5 F, .6 F, and 189b's 385/465 and 228/465.

## Ranked problems (most serious first), with the line to change

Line numbers refer to `fa68418`. Nothing here was edited; each item proposes wording.

1. **A registered prediction now fails on the eager kernel, but the front page says "Holds".** Part 66's 66.2
   ("unchanged, Google in at most 1% of 192") held at 1 of 192 on SDPA. On eager it is 2 of 192 (1.04%), and
   `results/part183/part66-eager/score.txt:2` prints `66.2: FAILS`. The lead checked this.
   - README.md:1196: replace "**Holds on the eager kernel (Part 183, 2026-11-04):** rerun with the same code and seeds,
     43 of 192 (22.4%) against 42 here, unchanged 2 of 192, best random set 1 of 24" with "**Rerun on the eager kernel
     (Part 183, 2026-11-04):** with the same code and seeds, 43 of 192 (22.4%) against 42 here and best random set 1 of 24,
     so 66.1, 66.3 and 66.4 still hold; unchanged names Google in 2 of 192 (1.0%) against 1 here, so 66.2 (at most 1%)
     fails by one answer."
   - README.md:19: change "each published claim holding: Parts 25 and 66 (Part 183)" to "Parts 25 and 66 (Part 183;
     every count moved by at most one answer, but 66.2 fails at 2 of 192)".
   - results/part183/README.md:16: change "Both published results stand." to "Both published headline results stand;
     Part 66's 66.2 (unchanged at most 1%) fails at 2 of 192."
   - Part 183's own 183.2 (bar: at most 2) held. The front page needs both verdicts kept separate: 183.2 held, 66.2 failed.
2. **Part 188's tally is wrong (MISMATCH).** README.md:2739 says "Four of nine predictions failed". Five failed
   (188.1, .2, .3, .4, .7) and four held (results/part188/README.md:27-35; PREDICTIONS.md:3456-3464). Replace with
   "Five of nine predictions failed (188.1 to 188.4 and 188.7); four held."
3. **33 PREDICTIONS.md verdicts are invisible on GitHub.** The rows for Parts 183 (PREDICTIONS.md:3400-3406), 187
   (3445-3450), 188 (3456-3464), 189 (3470-3475) and 189b (3481-3485) read `| Claude || <verdict> |`, which is six
   cells under a five-column header. GFM discards the extra cell, so the Result column renders empty. Fix: replace
   `| Claude || ` with `| Claude | ` on those lines. Related gaps:
   - Part 178 (PREDICTIONS.md:3349-3361) has no Result column at all. Add one: 178.1 to 178.7 held, and the remaining
     rows inconclusive/inconclusive/held/inconclusive, as results/part178/README.md gives them.
   - Part 181 (3369-3376) and Part 180b (3384-3394) have empty Result cells. The verdicts are in their READMEs; worker C's
     appendix lists them.
   - Unescaped `|d|` inside cells at 3279 (173.7) and 3295 (174.6) shifts those rows' columns. Escape it as `\|d\|`.
   The part READMEs carry verdicts that agree with the intended PREDICTIONS.md text in every case checked. The defect is
   rendering and missing cells, not disagreement.
4. **Part 148's rerun flipped a verdict, but Part 148's entry has no note.** Part 189b reran 148 on eager: 148.1 flips
   from failed to held (385 of 465 neurons, 82.8%, against 74.8%), and the random name rate rises 37.6% to 49.0%.
   Several places carry no eager note: README.md:2667 ("Five of six predictions failed"), the Finding paragraph at
   README.md:84 (146 vs 96 counts), and the box at README.md:19-20 (it does not list 148, and the banner at :17-18
   promises a dated note). Proposed note after README.md:2667: "**Rerun on the eager kernel (Part 189b, 2026-11-04):**
   five of the six verdicts stand; 148.1 flips from failed to held (name moved in 385 of 465 neurons, 82.8%, against
   74.8%); every registered rate rises 4.7 to 11.4 points, random directions most (49.0% against 37.6%); the
   neuron-over-random name gap narrows from 37.2 to 33.8 points. [results/part189b/README.md](results/part189b/README.md)"
   Add to README.md:20: "Rerun with one verdict flipped: Part 148 (Part 189b, 148.1 failed to held)."
5. **The short version's item 1 is stale (MISMATCH).** README.md:24 ends "Parts 25 and 66 are on the known-issue list
   above", but README.md:19, :719 and :1196 say both were rerun (Part 183). Replace with "Both were rerun on the eager
   attention kernel (Part 183): 43 of 192 zeroed against 2 of 192 unchanged." Make the matching edits at README.md:15
   ("Parts 25 and 66 underlie Finding 1" -> add "; both now rerun, Part 183") and README.md:73.
6. **"none in a published cell" is false (OVERCLAIM).** README.md:2731 and results/part184/README.md:49 say none of the
   4 changed outcomes is in a published cell. Control unit 8617 at -2 moves Part 18's published "20 of 20 still Qwen"
   (PREDICTIONS.md:665; README.md:359) to 19 of 20 on eager, and results/part184/README.md:53-54 says so itself. The lead
   checked this. Replace with "4 of them in a scored outcome; one moves Part 18's greedy controls at -2x from 20 of 20
   naming Qwen to 19 of 20". Worker C also read both texts: they name Tencent, and the SDPA one counted as Qwen only
   through a later "Qwen3.5" mention. That makes the published 20/20 itself a pattern-match artefact, so README.md:359
   and the PREDICTIONS.md:665 note should be revisited.
7. **Item 7 of the short version mixes three populations under one model (OVERCLAIM).** README.md:30 opens "in
   Qwen3.5-4B, the layer-22 screen". The "185 of 1,709 against 28 of 1,709" lens match comes from OLMo 2 1B Instruct
   (Part 92 screen, results/part97/README.md:5-6), which the lead confirmed. The "dose window one step wide" comes from
   11 Qwen slot neurons tested only at +1, +0.85 and +0.5 (Part 117b): that shows a lower edge, not a window.
   Replacement for the second half: "...; what is neuron-specific is which persona appears: in OLMo 2 1B, a neuron's
   J-lens words matched its pushed persona in 185 of 1,709 cases against 28 of 1,709 shuffled pairs (Part 97), and in
   Qwen3.5-4B 11 selected slot neurons showed their persona in 11 of 11 pairs at +1, 2 of 11 at +0.85 and 0 of 11 at
   +0.5 (Part 117b)."
8. **Item 8 drops the scope of its random-direction counts (OVERCLAIM).** In README.md:31 (and the Finding 7 body,
   README.md:166), "101 of 343 and 94 of 344" covers only the 344 Q-name targets (assistant-like names). Over all 465
   units the counts are 109/464 and 103/464; for non-AI targets they are 0 of 57. Proposed: "random directions got the
   target label for Q-name (assistant-like) personas in 101 of 343 and 94 of 344 self-question answers, and for non-AI
   personas in 0 of 57, so a Q-name slot label may partly capture generic self-description." Also, "only 1 of 6
   post-introduction answers was in character" misreads Part 109. Five of the six restated their persona; the 1 of 6 is
   the hand count of answers that described activities in character. Replace with "only 1 of 6 described activities in
   character; the other five restated the persona."
9. **Item 5's "two of five random sets cut it as much or more" (OVERCLAIM).** In README.md:28, also :2264 (heading) and
   :2303: c5 cut the gap more, but c3 cut it 0.349 against 0.368 for the selected heads (results table in the Part 82
   section, README.md:2293-2295; the paired bootstrap difference +0.020 [0.012, 0.027] is worker A's). Proposed: "one of
   five random sets cut it more and another nearly as much (0.349 against 0.368)." PREDICTIONS.md:2077 already states it
   correctly.
10. **Item 4 leaves out Part 78b and the kernel status (OVERCLAIM by omission).** README.md:27 sets 119 of 192 for one
    same-size control beside 0 of 192. Part 78b's ten more size-matched pushes gave 2 to 157, six of them below half, and
    Part 78 is on the high-risk kernel list and has not been rerun. Append "; ten more size-matched pushes gave 2 to 157
    of 192 (Part 78b), so that control was not typical. Parts 78 and 78b are on the known-issue list above."
11. **Item 1's "the four are the cheapest route to the model's own fallback" (OVERCLAIM).** In README.md:24, no part
    measures a cost ordering. Proposed: "so pushes that use no selected neuron reach the same Google answer, and the four
    are not its holders."
12. **Missing kernel caveats in items 7 and 8.** README.md:30-31 depend on Parts 117b and 112, both on the high-risk list
    at README.md:15, and item 1 sets the precedent of saying so. Append "Part 117b is on the known-issue list above." and
    "Part 112 is on the known-issue list above." Worker B could not confirm from the repo which kernel Part 86's screen
    used: its recorded `common86.py` hash differs from the current file, which now sets eager. Part 86 is not on the list.
13. **Item 9 omits Part 108's own reading.** README.md:32: moving the weights the same distance past Instruct lost as many
    units as moving back (results/part108/README.md), so the counts "do not show DPO building slot units". Append
    "moving the weights the same distance past Instruct lost as many, so the rising counts reflect where the units were
    picked."
14. **Stale or unscoped front-page entries (minor).**
    - README.md:2725 (Part 177): "113 of the 115 traces with a literal gold hit" is on cohort C (187 capped traces, gold in
      the prompt included), not the 173 the paragraph is about. Write "113 of the 115 capped traces with a literal gold hit
      (gold-in-prompt traces included)". The entry also reports only Weaver's tally; Claude's 177.4 failed.
    - README.md:2727 (Part 178): "eager bf16 is closer to a float32 reference" was measured on the 0.8B only. Add "on
      Qwen3.5-0.8B; no CPU reference exists for the 4B".
    - README.md:2743 and results/part189b/README.md:1: "rises 4 to 11 points" is in fact 4.7 to 11.4.
    - results/part188/README.md:57: "0.28 says the unit responds to human-made sound" contradicts the same README and
      README.md:2739 ("not established"). Write "suggests ... (not established)".
    - results/part97/README.md:19 still says 97.3 is "not run yet"; PREDICTIONS.md:2308 scores it held.
15. **Navigation and scope lines are stale (minor; lead).**
    - README.md:5: "This summary covers Parts 1 to 119c (... updated 2026-10-30)". The page carries entries through 189b.
    - README.md:2522: the heading "Parts 85 to 178" now holds entries through 189b.
    - README.md:337: the index omits 180b, 183, 186 to 189b. Neither Part 183 nor Part 180b has a dated front-page entry;
      worker C's appendix drafts both. Part 180 is not public (PART180B-DESIGN.md:12).

## Claims table

IDs: S = short-version item, P = part. Status and evidence as recorded by each auditor; the lead's rechecks are noted in the ranked list.

### A. short version items 1-6 (README.md:24-29)

| ID | Location (file:line) | Claim (short quote) | Status | Evidence (source file, how recomputed, true value) |
|---|---|---|---|---|
| S1a | README.md:24 | zeroing four maker neurons "made Google the maker in 42 of 192 answers across 24 held-out phrasings" | MATCH | results/part66/full/answers.jsonl, condition=target, question_kind in {direct,indirect}: 192 rows, 42 with /google\|gemini\|deepmind/i. Agrees with part66/full/score.json (heldout_google 42) and README.md:1209 (66.1). 24 phrasings x 8 samples confirmed. Eager rerun results/part183/part66-eager/answers.jsonl recomputed: 43 of 192 (matches README.md:1196, part183/README.md). |
| S1b | README.md:24 | "against 1 of 192 unchanged" | MATCH | Same file, condition=unchanged: 1 of 192 ("Who designed you?"). Eager rerun recomputed: 2 of 192 (Part 183 reports 2; 183.2 bound is <=2). |
| S1c | README.md:24 | "at dose 0.225, 30 of 40 flat random directions name Google too" | MATCH (wording loose) | results/part176/scored_directions.jsonl, 40 rows, rung dose 0.225: support=True on 30, all 30 google_led. Re-derived from per-direction counts with the Part 173/174 rule (Google modal_count >=10 of 44 and broken <=8, PART174-DESIGN.md:63-64): 29 definite + 1 bounded (d176110 +, Google 17-18, broken 1) = 30. "name Google" = Google modal in >=10 of 44 greedy answers; 37 of 40 directions named Google at least once. Part README results/part176/README.md:1,13 and PREDICTIONS 176.1 agree. Not re-joined to raw labels (label ids are hashed pairs). |
| S1d | README.md:24 | "with a median of one broken answer in 44" | MATCH | scored_directions.jsonl, 0.225 rung, `broken` over all 40 directions: median 1.0. Agrees with part176/README.md (176.5) and PREDICTIONS 176.5. |
| S1e | README.md:24 | "so the four are the cheapest route to the model's own fallback, not its holders" | OVERCLAIM | No part measures a cost comparison between zeroing the four and a flat push; "cheapest" appears nowhere else in the repo. The two arms differ in prompts (Part 66's 24 sampled phrasings at T0.7 vs Part 176's 44 greedy NEW44B prompts), intervention (zeroing vs additive push) and readout (any-mention rate vs modal-support rule). Part 175's own limits say "This part does not show that Google is the model's fallback name ... The fallback reading is not read" (results/part175/README.md:88); Part 176 adds no such reading. "Fallback ... not its holders" is consistent with Finding 1 (README.md:76, Parts 70-73 put the fallback in L27 N3325/L24 N3949), but "cheapest" is an unmeasured interpretation. |
| S1f | README.md:24 | "Parts 25 and 66 are on the known-issue list above." | MISMATCH (stale) | README.md:19 says Parts 25 and 66 were rerun on the eager kernel in Part 183 and the claims hold; README.md:719 and :1196 carry dated "Holds on the eager kernel" notes; results/part183/README.md predictions 183.1-183.7 all held; recomputed eager counts 43/192 zeroed, 2/192 unchanged, best random set 1 of 24 (SDPA: 2 of 24). Same staleness at README.md:15 ("Parts 25 and 66 underlie Finding 1" sits inside the at-risk paragraph) and README.md:73 ("Parts 25, 66, ... are on the high-risk list ... each affected part will get a dated note when it is rerun"). |
| S2a | README.md:25 | "OLMo 3 7B Think named DeepSeek in 16 of 23 raw answers without a system turn" | MATCH | results/part9-think/sysgen9-allenai__Olmo-3-7B-Think.json, generations.raw_nosys: 24 samples, 23 close `</think>`; regex /deepseek/i on text after last `</think>`: 16 of 23. name-counts.json "Think\|raw_nosys" DeepSeek 16, n 23. PREDICTIONS.md 9.2 (line 211) says the same. |
| S2b | README.md:25 | "Think-DPO named DeepSeek in 23 of 24 answers with an OLMo template" | MATCH | sysgen9-...-Think-DPO.json, generations.none (template inserts "You are Olmo ... built by Ai2", sysgen9_gpu.py:5-6): 24 finished, 23 with DeepSeek in final answer. name-counts.json "Think-DPO\|none" 23/24. PREDICTIONS 9.5 agrees. Note: the two numbers are from different conditions (Think raw vs DPO template); same-condition comparisons are Think template 16/21 vs DPO template 23/24, Think raw 16/23 vs DPO raw 14/24. Not wrong, but the "while" contrast is across conditions. |
| S3a | README.md:26 | "At 2x the recorded corpus maximum ... L22 N2742" | MATCH | results/part18/chat/steer.json: layer 22, neuron 2742, clamp_definition "activation SET to k x the unit's largest activation on the reads corpus", maxima 15.375, samples key "+2". |
| S3b | README.md:26 | "made 8 of 8 'Who made you?' answers mention extraterrestrial origins" | MATCH | steer.json samples["+2"]: 8 texts, /alien\|extraterrestrial\|planet/ in 8 of 8 (all 8 contain "extraterrestrial"; two do not name a planet/origin beyond "extraterrestrial intelligence"). summary.samples["+2"].alien = 8. Eager rerun results/part184/part18-eager/steer.json recomputed: 8 of 8. |
| S3c | README.md:26 | "against 0 of 8 unclamped" | MATCH | steer.json samples.unclamped: 0 of 8 (all 8 name Qwen). Eager rerun: 0 of 8. PREDICTIONS 18.4 held; Part 184 184.1/184.2 held. |
| S4a | README.md:27 | "zeroing the four maker neurons left Google in 37 of 192 held-out maker answers" | MATCH | results/part78/answers.jsonl, c1_makers, direct+indirect: 37 of 192. results.json and README.md:1749 agree. (Part 78's unchanged arm c0 gave 5 of 192, not shown on front page.) |
| S4b | README.md:27 | "adding a 2x alien push left 0 of 192" | MATCH | c5_makers_alien20: 0 of 192 Google. PREDICTIONS 78.3 held. |
| S4c | README.md:27 | "while a same-size control push left 119 of 192" | MATCH, but OVERCLAIM by omission | c6_makers_control20: 119 of 192 recomputed. But Part 78b (README.md:1803; results/part78b/answers.jsonl recomputed) shows the 119 is atypical: ten more size-matched layer-22 pushes gave 10, 157, 22, 41, 21, 10, 57, 14, 2, 18 (k1 makers-only 43); 6 of 10 cut Google below half (PREDICTIONS 78b.3 FAILS, line 2011). So the item's implied contrast (alien push suppresses the fallback, a same-size push does not) does not survive the follow-up part. Also Part 78 is on the high-risk kernel list (README.md:14) and has not been rerun; item 4 does not say so, unlike item 1. |
| S5a | README.md:28 | six selected heads "reduced the 'you' answer gap from 0.444 to 0.076" | MATCH | results/part82/main/records.jsonl (9,600 rows), option_set B, mean p_you(A) - mean p_you(U): h0 0.4441, h1 0.0756. Agrees with results82.json, README.md:2290 table, PREDICTIONS 82.1/82.2. |
| S5b | README.md:28 | "two of five random sets cut it as much or more" | OVERCLAIM (mild) | Same records: c3 gap 0.0954 (drop 0.349), c5 gap -0.0580 (drop 0.502), h1 drop 0.369. c5 cuts more; c3 cuts less. Paired bootstrap over 100 bodies (5,000 resamples, seed 82): gap(c3) - gap(h1) = +0.0198 [0.012, 0.027], so c3 cuts significantly less than h1 on this readout. Only on set N does c3 exceed h1 (0.164 vs 0.154). PREDICTIONS.md:2077 states it correctly ("c5 larger; c3 within 0.02"); README.md:2303 says c3 does "as much", which is the same loose wording. |
| S6a | README.md:29 | GPT-6 Luna judge "assigned the GPT family to 390 of 400 teacher answers" | MATCH | results/part29c/judge-gpt-6-luna.jsonl: teacher models (8 x 50 = 400 rows; TEACHERS map in part29_judge.py:15): OpenAI GPT labels 49+49+50+50+49+47+48+48 = 390. Matches summary.txt and README.md:142 (341 of 350 non-GPT). |
| S6b | README.md:29 | "named the right family for only 49 of 400" | MATCH | Correct labels: only gpt-6-luna row, 49 of 50; every other teacher 0. summary.txt line 1 "49/400". |
| S6c | README.md:29 | "versus 50 of 400 for an always-GPT guess" | MATCH | 50 GPT-written teacher answers of 400 (one of 8 families x 50). |

Counts: MATCH 15 (S1a, S1b, S1c, S1d, S2a, S2b, S3a, S3b, S3c, S4a, S4b, S5a, S6a, S6b, S6c), MATCH-but-overclaim-by-omission 1 (S4c), OVERCLAIM 2 (S1e, S5b), MISMATCH (stale) 1 (S1f), UNVERIFIABLE 0.

### B. short version items 7-11 (README.md:30-34)

| ID | Location (file:line) | Claim (short quote) | Status | Evidence (source file, how recomputed, true value) |
|---|---|---|---|---|
| S7a | README.md:30 | "At +0.8 on greedy 'Who are you?' pushes in Qwen3.5-4B, the layer-22 screen" | MATCH | `results/part86/screen-final/layer_22.jsonl.gz`: all 9,728 rows dose 0.8; meta `config`: Qwen/Qwen3.5-4B, greedy, 48 new tokens, "Who are you?". |
| S7b | README.md:30 | "250 of 9,216 neurons" | MATCH (denominator and triage recomputed; per-layer persona count needs private key) | Raw L22 rows: 9,216 neuron rows; Part 86 triage re-run on raw answers gives 8,023 kept / 674 broken / 519 changed (= README 519). DS v2 labels `second-reader/labels_ds_v2_full.jsonl`: 3,349 ids, 1,755 coherent personas = 1,713 real + 42 random (README totals). Layer/arm split needs `KEY-private.json` (outside repo, `prediction-score/score86_predictions.py:273`), so 250 is taken from NUMBERS.md:113/150. |
| S7c | README.md:30 | "3 of 512 random directions" | MATCH (same caveat) | Raw L22: 512 random rows, 494 kept / 1 broken / 17 changed (= README 17). 3 personas from scorer output (NUMBERS.md:150). Fisher one-sided recomputed by hand from 250/9216 vs 3/512: p = 0.000584 (= README). |
| S7d | README.md:30 | "both lower bounds" | MATCH | part86/README.md Limits: only changed answers were read, "every rate over all pushes is a lower bound"; miss sample found 84/200 real-broken and 12/31 random-broken were personas. |
| S7e | README.md:30 | "a neuron's lens words matched its pushed persona in 185 of 1,709 cases against 28 of 1,709 shuffled pairs" | OVERCLAIM (wrong model/screen implied) | Counts recomputed: `results/part97/match.jsonl` j_lens neuron True = 185/1,709 (random 0/45); `match-shuffled.jsonl` True = 28/1,709; one-sided Fisher (hand) 1.56e-31 (= README 1.6e-31). But Part 97 is **OLMo 2 1B Instruct, Part 92 screen** (part97/README.md:1-6), not the Qwen3.5-4B layer-22 screen the same sentence describes. Finding 7 body says so (README.md:172); the short version does not. The Qwen analogue is Part 98 (224 of 465 vs 22 of 460, no results README). |
| S7f | README.md:30 | "inside a dose window one step wide" (Part 117b) | OVERCLAIM | Recomputed from `results/part117b/labels.jsonl` (question 0, slot-alone rows): slot persona 11/11 at +1, 2/11 at +0.85, 0/11 at +0.5 (= score.json). 117b tested only doses at or below the selection dose, so it shows a sharp lower edge, not a window with an upper edge; the upper edge comes from Part 118 (OLMo, 2x incoherent), cited only in the Finding 7 body (README.md:174). The population is 11 Qwen slot neurons from Part 112, not the 1,709 OLMo units this clause is attached to. Part 117b's own README calls these "dose controls, not a matched test" (part117b/README.md, "Does a weaker slot push explain the result?"). Part 117b is on the Arc high-risk list (README.md:15) and the item carries no caveat. |
| S8a | README.md:31 | "persona theme appeared in 0 of 6 cold activities answers" | MATCH | `results/part109/labels.jsonl` unit A: shows_persona 0/6. |
| S8b | README.md:31 | "6 of 6 after self-introduction plus push" | MATCH | labels unit B: 6/6. (Unit C, same conversation with no push: 2/6, not mentioned.) |
| S8c | README.md:31 | "only 1 of 6 post-introduction answers was in character" | MISMATCH (wording) | part109/README.md:21-23,33: five B answers restate the persona ("I am a cat.", "I'm a very tall tree."), which is in character. Only the dog "describes activities in character". That 1 of 6 is the author's by-eye reading, not a reader label. Read the B answers in `items.jsonl` to confirm. |
| S8d | README.md:31 | "In Qwen3.5-4B, random directions got target labels in 101 of 343 and 94 of 344 self-question answers" | OVERCLAIM (missing scope) | Recomputed from `results/part112/labels.jsonl` + `units.json`: 101/343 ("Who are you?") and 94/344 ("Tell me about yourself") are the **Q-name subgroup only** (344 of 465 units, personas like "Q, a language model"). All units: 109/464 and 103/464. Other-AI: 8/64 and 9/63. Non-AI: **0/57 and 0/57**. part112/README.md:5,13 gives the split; NUMBERS.md:18-19 labels the lines "q_name". Part 112 is on the Arc high-risk list (README.md:15) and the item has no caveat. |
| S8e | README.md:31 | "so a slot label may partly capture generic self-description" | MATCH (hedged), but scope as S8d | This holds for Q-name targets (assistant-like names). For non-AI targets the random rate is 0/57, so the inference does not extend to the OLMo theme personas in the first half of the item. |
| S9a | README.md:32 | "71 OLMo 2 1B slot units selected at Instruct" | MATCH | `results/part108/kinds108.json`: 71 slot (+18 concept); score108.json `slot_units`: 71. |
| S9b | README.md:32 | "14 of 71 at SFT, 40 of 71 at DPO, and 64 of 71 at Instruct" | MATCH | Recomputed from `labels-S/D/I.jsonl` restricted to the 71 slot ids: S 14/71 labelled, D 40/69 labelled (2 unlabelled counted as not showing), I 64/71. Equals score108.json `all_slot_shown` and part108/README.md:74,86. |
| S9c | README.md:32 | "the forward weight mixes were not checkpoints" | OVERCLAIM (by omission) | True (part108/README.md Limits). But Part 108's registered reading F (README.md:3-7,88-90) is that going forward loses as many units (43 vs 51, 32 vs 24), so the stage counts "do not show DPO building slot units", and Part 92's DPO claim is withdrawn. The short version gives the rising stage counts and only a caveat against the forward control, which invites the withdrawn reading. |
| S10a | README.md:33 | "357 of 741 real-pair answers broke" | MATCH | Part 86 triage re-run on `results/part119/answers.jsonl` pair rows (alpha 0.8, reused by 119b): 357/741. |
| S10b | README.md:33 | "versus 286 of 741 random-pair answers" | MATCH | `results/part119b/answers.jsonl` randpair dose 0.8: 286/741. Note these come from a separate run (119b) from the real pairs (119). Disclosed in part119b/README.md:255. |
| S10c | README.md:33 | "Fisher p=0.000241" | MATCH | Hand-coded exact test: two-sided p = 0.000241 (one-sided 0.000120). Omitted context: real pair vs A+R control at 0.8 is 357/741 vs 659/1,482 (both recomputed), p = 0.104 two-sided (part119b/README.md:5). Part 119b says the "more than which neurons" comparison "was not measured directly" (README.md:15). |
| S10d | README.md:33 | "effect-matched logit-interaction ratio was 1.044, with a 95% interval of 0.961 to 1.129" | MATCH | Point estimate re-derived from `results/part119/interaction.jsonl` with an independent implementation (control lstsq fit, 1-99% D2 support, row medians, mean): mean residual 0.0430, exp = 1.0439 over 1,475 rows. CI: exp(score119.json `effect_matched/mean_residual_ci95` [-0.04003, 0.12131]) = [0.961, 1.129]. Bootstrap not re-run. |
| S11a | README.md:34 | "'I am **' forced ... over 16 held-out phrasings" | MATCH | `results/part114/results114.json` rows: 16 heldout entries per row. |
| S11b | README.md:34 | "zeroing one of the top 20 candidates ... flipped 0 of 16" | MATCH | results114.json: all 20 candidate rows have argmax_is_q true on 16/16. |
| S11c | README.md:34 | "or the top pair flipped 0 of 16" | MATCH | results114.json `pair_generations.top2`: all 16 greedy continuations start "Q". Part 115 k=2 batched: 0/16. |
| S11d | README.md:34 | "top 200 flipped 11 of 16" | MATCH | `results/part115/results115.json` candidate k=200: 11/16 not argmax Q. (k=100: 3/16. Max flips by any of the 90 random sets: 0.) |
| S11e | README.md:34 | "top 500 flipped 16 of 16" | MATCH | results115.json k=500: 16/16. |

### C. Parts 177, 178, 181, 180b, 183, 184

| ID | Location (file:line) | Claim (short quote) | Status | Evidence (source file, how recomputed, true value) |
| --- | --- | --- | --- | --- |
| P177a | README.md:2725 | "Among 173 ... traces ... 67 had a window a reader judged to state the gold" | MATCH | results/part177/window-key.jsonl joined to labels-ds.jsonl: real, target_T, label `asserted`, giving 67 distinct problems (101 with a literal hit). The 173 denominator comes from numbers177.json `cohorts.T_gold_not_in_prompt`. The raw MIT samples (`inputs/eval/...gz`) are not committed, so 173 itself cannot be recounted. |
| P177b | README.md:2725 | "against 0 for shuffled answers" | MATCH | Same join: shuffled, target_T gives 33 hit traces and 0 asserted. On C it is 37 and 0, and 0 of 98 shuffled windows are asserted. |
| P177c | README.md:2725 | "Of the 66 ... with a detected loop onset, 49 stated it before the onset" | MATCH (summary only) | numbers177.json `onset_order.asserted_on_T` gives 49 / 17 / 1. Onset positions need the uncommitted traces, so this was not recomputed from raw. It agrees with results/part177/README.md:68-75. |
| P177d | README.md:2725 | "113 of the 115 traces with a literal gold hit had no extractable final answer" | MATCH (summary), scope | numbers177.json `w3_raw_box_audit_C_literal_hits.null_pred_first`=113, trace_denominator 115. The 115 is cohort C (187 traces, gold in the prompt included), not the 173-trace T named in the same paragraph. The README (results/part177/README.md:104-108) says C. |
| P177e | README.md:2725 | "Of Weaver's three predictions, one is inconclusive and two failed" | MATCH | PREDICTIONS.md:3343-3345 and results/part177/README.md:89-91: W1 inconclusive, W2 failed, W3 failed. |
| P177f | README.md:2725 | (no statement of Claude's six verdicts) | note | 177.4 failed (35 of 48). The front page reports only Weaver's tally. Not wrong, but a reader cannot see that one of Claude's six failed. |
| P178a | README.md:2727 | "`wait` ranked second ... (86.22 per 1,000 openers)" | MATCH | results/part178/eager/score.json `tables.window.U`: 1534/17791 = 86.22, ranked 2nd after "let's". U is MIT's stored text and is not committed, so this was not recounted. |
| P178b | README.md:2727 | "tenth in our Qwen3.5-4B thinking runs (19.59; rank interval 7 to 11)" | MATCH | eager/score.json G: 324/16537 = 19.59, rank 10, `rank_ci95` 7 to 11. The window rule needs the tokenizer (not local), so the window counts were not recounted. The full-text opener totals per set (18,311 / 10,913 / 17,455 / 7,933) match readme_numbers exactly. |
| P178c | README.md:2727 | "all seven numbered predictions held on point estimates" | MATCH | eager/score.json verdicts for 178.1 to 178.7 are all `held`. results/part178/README.md:16-22 agrees. |
| P178d | README.md:2727 | "4.66 times as frequent on the 0.8B (held)" | MATCH | W3a `wait_ratio` 4.6646. Arithmetic check: R8 86.30 / N8 18.50 = 4.665. Verdict held. |
| P178e | README.md:2727 | "1.96 times on the 4B (inconclusive)" | MATCH | W3b 1.9625. Arithmetic: 19.59 / 9.98 = 1.963. Verdict inconclusive. |
| P178f | README.md:2727 | "on capped rows alone the 4B ratio is 1.09" | MATCH | Recomputed from eager/G.jsonl and N4.jsonl with words178.opener_spans on full generated_text, finish_reason=cap: G 333/15703 = 21.21, N4 67/3433 = 19.52, ratio 1.087. The 0.8B capped ratio is 3.95. This is a full-text measure (no window, no onset cut), which the front page does not say. |
| P178g | README.md:2727 | "eager bf16 is closer to a float32 reference but not identical to it" | OVERCLAIM (mild) | results/part178/README.md:100 measured this on Qwen3.5-0.8B only (diag178g) and says "No CPU reference exists for Qwen3.5-4B ... for the 4B that closeness is not measured". The front page states it for all generated sets, and the 4B sets carry 178.2, 178.4, 178.7 and W3b. |
| P181a | README.md:2729 | "131,072 units ... 2,500 Pile documents at 12 checkpoints" | MATCH | analysis/overlaps.json.gz: units 131,072 in the all cohort; results/part181/runs holds 12 checkpoints plus 3 controls. |
| P181b | README.md:2729 | "3T to 4T: median 0.5 shared" | MATCH | overlaps.json.gz pair pretrain_3T__pretrain_4T, neurons, top10_doc: mean 0.464, median 0.5 (all and primary cohorts). Per-unit top-10 states are not committed, so the input here is the per-pair aggregates. |
| P181c | README.md:2729 | "mid-training and SFT each keep a median 6 of 10" | MATCH | 4T to midtrain_51B: median 0.6 (mean 0.579). midtrain_51B to sft: median 0.6 (mean 0.623). |
| P181d | README.md:2729 | "DPO changes about 1 in 10 on average" | MATCH | sft to dpo mean 0.886, so 1.14 in 10 change. |
| P181e | README.md:2729 | "RLVR about 0.4" | MATCH | dpo to instruct mean 0.958, so 0.42. |
| P181f | README.md:2729 | "0.07 for a bf16 rounding" | MATCH | pretrain_4T to pretrain_4T_bf16r mean 0.993, so 0.07. |
| P181g | README.md:2729 | "Neurons kept their top documents worse than sums of 2 to 64 coordinates (181.5 failed)" | MATCH | results/part181/README.md:74-78: k=2 to 64 is above k=1 in all three pairs; verdicts.json.gz 181.5 failed. |
| P181h | README.md:2729 | "Five of six numbered predictions held, 181.1 exactly at its bar" | MATCH | verdicts.json.gz: 181.1 to 181.4 and 181.6 held, 181.5 failed. The 181.1 median is 0.6 (float 0.6000000238) against a bar of 0.6. |
| P181i | README.md:2729 | "Weaver's two failed, one uninformatively" | MATCH | analysis/weaver181.json: W1 failed ("both floors are 0"), W2 failed. |
| P183a | README.md:1196 | "43 of 192 (22.4%) against 42 here" | MATCH | Recount of results/part183/part66-eager/answers.jsonl (target, direct+indirect) with score66.families gives 43/192. The published 42 is confirmed in results/part66/full/score.json. |
| P183b | README.md:1196 | "unchanged 2 of 192" | MATCH (number), OVERCLAIM (heading "Holds") | Recount 2/192 = 1.04%. Part 66's registered 66.2 ("unchanged ... at most 1%") held on SDPA (1/192, README.md:1210, PREDICTIONS.md:1873) and **fails** on eager: results/part183/part66-eager/score.txt prints "66.2: FAILS". The note claims "Holds" and sits directly above the table that shows 66.2 held. |
| P183c | README.md:1196 | "best random set 1 of 24" | MATCH | Recount over 50 random sets x 24 held-out answers: max 1 (SDPA max 2, part66/full/score.json). |
| P183d | README.md:719 | "the four still flip only 'Who created you?', to the same Google answer word for word" | MATCH | results/part183/part25-eager/answers.jsonl: four/"Who created you?" answer_eager == answer_sdpa (True). Makers agree on 12 of 12 (none, four). 28 of 32 texts are identical and all 32 makers agree. Random sets give 0 of 120 non-Alibaba answers. |
| P183e | README.md:19 | "Rerun on the eager kernel so far, each published claim holding: Parts 25 and 66 (Part 183), 18 and 18b (Part 184)" | OVERCLAIM | Published claim 66.2 does not hold (see P183b). The published Part 18 claim "In the greedy run, no control at -2x changed the answer" (README.md:359; PREDICTIONS.md:665 "20/20 still Qwen") reads 19/20 under the same scorer on eager (see P184i). |
| P183f | results/part183/README.md:16 | "Both published results stand" | OVERCLAIM | The headline counts stand, but registered verdict 66.2 flips from held to failed. 66.1, 66.3 and 66.4 hold; 66.5 and 66.6 fail on both kernels. |
| P184a | README.md:2731, :102, :382 | "+2x 8 of 8 alien, unclamped 0 of 8" | MATCH | score_text on part184/part18-eager/steer.json samples: +2 8/8 alien, unclamped 0/8. |
| P184b | same | "controls 0 of 160" | MATCH | 160 controls_sampled texts, 0 alien. Also 0/160 in B and C. |
| P184c | same | "generation-only 8 of 8 (was 7)" | MATCH | part18b-gen +2: 8/8. Stored results/part18b/phase-gen/steer.json: 7/8 (PREDICTIONS.md 18b.5, README.md:100). |
| P184d | same | "prompt-only 0 of 8" | MATCH | part18b-prompt +2: 0/8 (stored 0/8). |
| P184e | README.md:102, :382 | "every count moved by at most one sample" | MATCH | -2 names Qwen 2 to 1; gen 7 to 8; control greedy -2 Qwen 20 to 19; all other cells equal (recounted on both stored and eager files). |
| P184f | README.md:2731 | "All six predictions held" | MATCH | compare.json verdicts; results/part184/README.md:26-31; PREDICTIONS.md:3412-3417. |
| P184g | README.md:2731 | "175 of 192 greedy answer texts differ" | MATCH | Independent pairing of target and controls_greedy across A/B/C against the stored SDPA files: 192 pairs, 17 identical (17 also after the role-turn cut). |
| P184h | README.md:2731 | "4 of them in a scored outcome" | MATCH | Independent score_text diff: A -1 "Tell me about yourself", A +2 "17 times 3", A control 8617 -2 "Who made you?", B +2 "17 times 3". |
| P184i | README.md:2731; results/part184/README.md:49 | "none in a published cell" | OVERCLAIM | Control 8617 at -2 changes names_qwen True to False, which moves Part 18's published "no control at -2 changed away from Qwen (20/20 still Qwen)" (PREDICTIONS.md:665, README.md:359) to 19/20. results/part184/README.md:53-54 itself states "19 of 20 on eager and 20 of 20 on SDPA". Caveat: by reading, both texts say "created by **Tencent**". The SDPA text matched QWEN_RE only through a late "the Qwen3.5 model" mention, so the published 20/20 was a regex artifact. The fault is still in the published cell, not absent from it. |
| P184j | README.md:102, :382 | "one greedy -1x answer changed its maker" | MATCH | A -1 "Tell me about yourself": SDPA "developed by Google", eager "Alibaba Cloud's Tongyi Lab". |
| P184k | README.md:2731 | "some part before key 64" | MATCH | greedy-text-compare.json: A target 13 of 40 and A control 26 of 79 differ before key 65; earliest key 22 (target) and 18 (control). |

### D. Parts 185, 186, 187, 188, 189, 189b

| ID | Location (file:line) | Claim (short quote) | Status | Evidence (source file, how recomputed, true value) |
| --- | --- | --- | --- | --- |
| P185a | README.md:2733, :2474 | "A2 23 and A3 24 of 24 off-world, A0 and the control unit 0", A1 13 | MATCH | part84-eager/reading/labels.jsonl joined to private/key.json: A1 13, A2 23, A3 24, A0 0, C 0 |
| P185b | README.md:2474 | "the hand-checked SDPA count is also 13" | MATCH | results/part84/reading/labels-checked.jsonl + key.json: A1 13 |
| P185c | README.md:2733, :2520 | "decode 3, 28, 44 and gen 4, 29, 47 of 48" | MATCH | part84b-eager labels.jsonl + key.json: decode 3/28/44, gen 4/29/47 |
| P185d | README.md:2520 | "8 against 3 discordant pairs, p = 0.23" pooled; "3 to 5 discordant pairs per dose" | MATCH | per dose gen-only/decode-only 2/1, 3/2, 3/0 (3, 5, 3 pairs); pooled 8/3, exact two-sided binomial p = 0.2266 |
| P185e | README.md:2520 | "where SDPA was level" | MATCH | part84b labels + key: pooled 5/5, p = 1.0 |
| P185f | README.md:2733 | "427 of 456 seed-paired answers differ in text" | MATCH | records.jsonl eager vs part84/main and part84b/main records, keyed (arm/cell, q, sample), seeds equal: 111 + 316 = 427 |
| P185g | README.md:2733, :2474 | "376 of 456 against 246" at cap; "90 of 120 ... against 55" | MATCH | new_tokens >= 64: Part 84 90 vs 55; Part 84b 286 vs 191; totals 376 vs 246 |
| P185h | README.md:2733 | "All six predictions held, though 185.5 could not have failed" | MATCH | score185.json all held; max 5 discordant pairs -> min p 0.0625 |
| P185i | PREDICTIONS.md:3423-3428 | published counts (24/24, 0/0, 17, decode 4,25,43, gen 3,23,46, p 1.0/0.69/0.25) | MATCH | recomputed from results/part84/reading and results/part84b/reading labels + keys; part84b/score.txt |
| P186a | README.md:2735 | "e1 51 Google by keyword (published 46)" | MATCH | part78.score78.classify over held-out rows via reader_key: eager 51, Part 79 46 |
| P186b | README.md:2735, :120, :1968 | "e2 136 villainous (published 125) and e0 0" | MATCH | main/reader.jsonl + reader_key.json: e2 136 (Part 79: 125), e0 0 |
| P186c | README.md:2735, :120 | "e3 0 Google and 0 naming a company (published 1)" | MATCH | keyword Google 0; any_company rule (reader google/other_company or Alibaba/Tongyi regex) 0; Part 79 1 |
| P186d | README.md:120, :1968 | "against 185 with the makers zeroed alone" | MATCH | e1 any_company 185 (Part 79 182) |
| P186e | README.md:2735 | "step 0's push sizes unchanged" | MATCH | extra_counts.json push_sizes identical to Part 79's to 3 decimals; medians 9.509/9.857/22.959 from step0/step0.json |
| P186f | README.md:2735 | "386 of 1,944 seed-paired answers are identical" | MATCH | main/answers.jsonl vs part79/main/answers.jsonl by (condition, q, sample): 386 (per condition 28,25,42,51,2,28,131,0,79) |
| P186g | README.md:2735 | cap "e2 95 against 131", "e0 115 against 61" | MATCH | stop_reason == max_new_tokens over held-out rows: e2 95/131, e0 115/61 |
| P186h | README.md:120, :1968 | "Part 79's seven verdicts are the same on both kernels" | MATCH | results/part186/main/results.json vs part79/main/results.json: 79.1-79.4, 79.7 HOLDS; 79.5, 79.6 FAILS in both |
| P187a | README.md:2737, :2653 | "48.3% on Set P (published 48.3%)" | MATCH | joint = coherent and kind not in (unclear, still Qwen); unit-level mean over 120 Set P units, from main/answers.jsonl + labels/ds.jsonl: 48.3 vs 48.3 (Part 140 raw) |
| P187b | README.md:2737, :2653 | "releasing after 8 tokens 7.1% (published 5.8%)" | MATCH | arm C Set P 7.1 vs 5.8 |
| P187c | README.md:2737 | "norm-preserved 0.0%" | MATCH | arm F Set P 0.0 |
| P187d | README.md:2737 | "random directions at most 3.9% in any arm (published 6.1%)" | MATCH | Set C max eager 3.89 (arm B), Part 140 6.11 (arm B) |
| P187e | README.md:2737 | "3,698 of 3,840 answer texts changed" | MATCH | pushed rows keyed arm/set/unit/question/seed: 3,698 of 3,840 (B 1,436/1,440; greedy arms 444-462/480) |
| P187f | README.md:2737 | "every Set P rate moved by at most 2.5 points" | MATCH | max |eager-SDPA| over arms A-F Set P = 2.5 (B 37.6 vs 35.1) |
| P187g | README.md:2737, :2653 | "none of the nine published verdicts flipped" | MATCH | part187/score.json vs part140/score.json holds flags identical for 140.1-140.8 incl. 140.5 arm A/C |
| P187h | README.md:174 | Part 140 rates 48.3%, 0%, 5.8%, 37.9%, 35.1%, 43.3% | MATCH | recomputed from part140 answers + ds labels: A 48.3, F 0.0, C 5.8, D 37.9, B 35.1, E 43.3 |
| P188a | README.md:2739 | "held-out AUC 0.753 and 0.897 (L13 N773, L5 N1770)" | MATCH | unit_auc.npz argmax |select-0.5| = 27397 (L13 N773), 12010 (L5 N1770); rank AUC on top_units.npz act + part188/manifest188.csv folds: 0.7534, 0.8966 |
| P188b | README.md:2739 | "0.123 and 0.089 above the model's own verdict (0.631, 0.808)" | MATCH | prompt-verdict recomputed from emb188.npz (float16 stored): 0.6301, 0.8076 (results.json 0.6305, 0.8075; difference is float16 storage); gaps 0.123, 0.089 |
| P188c | README.md:2739 | "probe on the 768-d output embedding reaches 0.918 and 0.965" | MATCH | own Newton L2-logistic (C=1, standardized, unpenalized intercept): 0.9181, 0.9651; top-32 0.9052, 0.9573 |
| P188d | README.md:2739 | "Against ESC-50 human sounds the units score 0.281 and 0.677" | MATCH | test-fold R vs all H with selected sign: 0.2809, 0.6768 |
| P188e | README.md:2739 | "hens 0.983, 0.913 on 138 clips" | MATCH | C_R vs C_I (100 + 38 = 138): 0.9828, 0.9128 |
| P188f | README.md:2739 | "recorded all 3,138 clips" | MATCH | manifest 3,138 rows; emb188.npz ids 3,138; results.json gate2 sets all full |
| P188g | README.md:2739 | "Four of nine predictions failed" | MISMATCH | five failed: 188.1, 188.2, 188.3, 188.4, 188.7 (results.json predictions; results/part188/README.md table; PREDICTIONS.md:3456-3464). Four held (188.5, .6, .8, .9) |
| P188h | README.md:2739 | Amendment 1 missing pin; Amendment 2 phantom set P "fixed before any value was read" | MATCH | PART188-DESIGN.md:174-192 |
| P188i | README.md:2739 | seq-length (not on front page), 188.5 0.647 | MATCH / UNVERIFIABLE | seqlen recomputed 0.6318, 0.6345; L17 N1574 (index 36390) not in npz arrays, 0.647 only from results.json |
| P189a | README.md:2741 | "only 17 of 32 ... first 40 tokens (28 required)" | MATCH (record) / token count UNVERIFIABLE locally | pilot/gates.json 17/32; no Qwen tokenizer installed; word-level proxy (first 30 words) gives 16, consistent; full-text whitespace-normalised recomputed 5 of 32 = README |
| P189b | README.md:2741 | "SDPA kernel at batch 64 reproduced all 32 stored answers, eager at batch 64 matched 18" | MATCH | full-text recompute vs results/part112/answers.jsonl: SDPA-b64 32/32; eager-b64 full 6/32 (diag.json 40-token 18) |
| P189c | README.md:2741 | "first differing token sat anywhere from 1 to 53" | UNVERIFIABLE (consistent) | needs tokenizer; first differing word index 0 to 40 in pilot, consistent with tokens 1-53 |
| P189d | README.md:2741 | "all six predictions are withheld" | MATCH | PREDICTIONS.md:3470-3475 all "withheld"; part README "untestable (withheld)" |
| P189e | README.md:2743 | "wiring check moved to an SDPA batch-64 arm (32 of 32)" | MATCH | results/part189b/wiring/answers.jsonl vs Part 112 stored: 32/32 full text; gates.json all_passed, load eager |
| P189f | README.md:2743 | "name moved in 385 of 465 neurons, 82.8%, against 74.8%" | MATCH | reimplemented score148 rules (regex moved AND label coherent) on answers.jsonl + labels.jsonl: 385/465; Part 148 raw 348/465 = 74.8% |
| P189g | README.md:2743 | "random directions most (name 49.0% against 37.6%)" | MATCH | random name 228/465 = 49.0%; Part 148 175/465 = 37.6% (+11.4, largest) |
| P189h | README.md:2743 | "every registered rate rises 4 to 11 points" | MATCH (loose) | rises: name +8.0, maker +4.7, both +9.2, maker-among-name +7.1, random +11.4, slot +4.7, concept +4.7; true range 4.7 to 11.4 |
| P189i | README.md:2743 | "gap narrows only from 37.2 to 33.8 points" | MATCH | 74.8-37.6 = 37.2; 82.8-49.0 = 33.8 |
| P189j | README.md:2743 | "five of Part 148's six verdicts stand. 148.1 flips from failed to held" | MATCH | part189b/score.json vs part148/score.json: only 148.1 differs (failed -> held) |
| P189k | README.md:2743 | "Two of five predictions held (189b.1, 189b.5)" | MATCH | PREDICTIONS.md:3481-3485; slot 91/189 48.1%, concept 112/257 43.6% (+4.7 each) |

## Predictions-table check

The lead's rendering check applies to all parts in scope (problem 3).

### A. short version items 1-6 (README.md:24-29)

- Part 9 (PREDICTIONS.md:195-218): rows 9.1-9.6 found; 9.1-9.5 carry verdicts; 9.6 result cell is empty (PREDICTIONS.md:214) - not cited by the short version. 9.2 (16 of 23) and 9.5 (DPO 23/24) agree with name-counts.json and README.md:94.
- Part 18 (PREDICTIONS.md:653-667): rows 18.1-18.7, all with verdicts; 18.4 "held: +2 8/8, unclamped 0/8" agrees with steer.json and Part 184 (184.1-184.6 all held, PREDICTIONS.md:3408-3418).
- Part 29c (PREDICTIONS.md:1271-1311): predictions 29c.1-29c.6 listed in prose, each with a verdict (1 failed, 2 held as artefact, 3 failed, 4 held trivially, 5 held trivially, 6 failed); agree with summary.txt.
- Part 66 (PREDICTIONS.md:1868-1877): rows 66.1-66.7 all carry verdicts; agree with README.md:1207-1215 table.
- Part 78 (PREDICTIONS.md:1994-2003): rows 78.1-78.6 all HELD; agree with README.md:1757 ("All six registered predictions held"). Part 78b rows 78b.1-78b.4 (2009-2012) carry verdicts (2 hold, 2 fail).
- Part 82 (PREDICTIONS.md:2075-2080): rows 82.1-82.6 all carry verdicts (82.3, 82.4 fail); agree with README.md:2299-2304 ("Four of six predictions held").
- Part 176 (PREDICTIONS.md:3318-3330): 176.1-176.7 and W1-W3, all with verdicts; agree with results/part176/README.md table (W1-W3 failed, others held).
- Part 183 (PREDICTIONS.md:3396-3404): 183.1-183.7 all held; agree with results/part183/README.md. (Minor formatting: rows use "||" so the Result sits in a sixth column, PREDICTIONS.md:3400-3406.)
- Part 184 (PREDICTIONS.md:3408-3418): 184.1-184.6 all held; agree with results/part184/README.md.

### B. short version items 7-11 (README.md:30-34)

- Part 86: PREDICTIONS.md has 15 rows (2149-2163), all with verdicts. They agree with part86/README.md:112-126 (86.1-86.13).
- Part 97: 3 rows (PREDICTIONS.md:2306-2308), all with verdicts (97.1 failed, 97.2 held, 97.3 held). DISAGREEMENT: results/part97/README.md:19 still says "97.3, the validation push, is not run yet". The part README has no verdict table. 97.3 is scored in results/part97/validation/ (front-page Finding 7 cites 6 of 10 vs 0 of 10).
- Part 117b: 10 rows (2557-2566), all with verdicts. They agree with part117b/README.md:35-44.
- Part 109: 3 rows (2434-2436), all with verdicts. They agree with part109/README.md:32-34. The README qualifies 109.2 as "(by the registered measure)".
- Part 112: 4 rows (2482-2485), all with verdicts. They agree with part112/README.md:13-16.
- Part 108: 6 rows (2423-2428), all with verdicts. They agree with part108/README.md:81-86.
- Part 119: 10 rows (2595-2604), all with verdicts. They agree with part119/README.md:47-56.
- Part 119b: 11 rows (2610-2620), all with verdicts. They agree with part119b/README.md:25-35 (4 rows "stopped, not read (Gate 3)").
- Part 114: 4 rows (2502-2505), all with verdicts. They agree with part114/README.md:18-21.
- Part 115: 6 rows (2511-2516), all with verdicts. They agree with part115/README.md:19-24.

### C. Parts 177, 178, 181, 180b, 183, 184

- Part 177: 9 rows (PREDICTIONS.md:3337-3345), verdicts present, all agree with results/part177/README.md:83-91.
- Part 178: 11 rows (PREDICTIONS.md:3351-3361). **No Result column at all** (header PREDICTIONS.md:3349 has 4 columns), so no verdicts are recorded. The README table (results/part178/README.md:16-26) has them: 178.1-178.7 held, W1 inconclusive, W2 inconclusive, W3a held, W3b inconclusive.
- Part 181: 8 rows (PREDICTIONS.md:3369-3376). **Result cells empty.** The README (results/part181/README.md:49-56) has 181.1 held (at bar), 181.2 held, 181.3 held, 181.4 held, 181.5 failed, 181.6 held, W1 failed (uninformative), W2 failed.
- Part 180b: 11 rows (PREDICTIONS.md:3384-3394). **Result cells empty.** results/part180b/SCORE.md:185-195, score.json and README:66-79 agree: held 180b.2-.7, .9, .11; failed 180b.1, .8, .10. I recomputed 180b.1 (lived 53/53), .2 (0, 1, 0 of 53), .3 (53/53 shows, 0/53 coherent) and .9 (swapped 0/53) from KEY-private.json joined to labels-ds.jsonl: all match.
- Part 183: 7 rows (PREDICTIONS.md:3400-3406). The verdicts are present but **malformed**: each row is `| Claude || 22.4% ...: held |`, which makes 6 cells under a 5-column header. GitHub drops the extra cell, so the Result column renders empty. The content agrees with results/part183/README.md:25-31 (all 7 held).
- Part 184: 6 rows (PREDICTIONS.md:3412-3417), verdicts present, all agree with results/part184/README.md:26-31.

### D. Parts 185, 186, 187, 188, 189, 189b

- Part 185: 6 rows (PREDICTIONS.md:3423-3428), verdicts present, all held; agree with results/part185/README.md table and score185.json.
- Part 186: 6 rows (3434-3439), verdicts present, all held; agree with results/part186/README.md.
- Part 187: 6 rows (3445-3450), verdicts present (all held) and agree with README, but each row is malformed: `| high | Claude || held (48.3%) |` has 6 cells under a 5-column header, so GitHub drops the verdict cell and the Result column renders empty.
- Part 188: 9 rows (3456-3464), verdicts present; agree with results/part188/README.md (5 failed, 4 held). Same 6-cell malformation.
- Part 189: 6 rows (3470-3475), all "withheld"; agree with README ("untestable (withheld: Gate 2 failed)"). Same malformation.
- Part 189b: 5 rows (3481-3485), verdicts present (held, failed, failed, failed, held); agree with results/part189b/README.md. Same malformation.

## Cross-references

### A. short version items 1-6 (README.md:24-29)

- README.md:24 "four maker neurons" -> README.md:717 (Part 25) / part183: confirmed (same four units; eager rerun flips only "Who created you?").
- README.md:24 "42 of 192" / "1 of 192" -> README.md:1209-1210, results/part66/full/score.json: confirmed by raw recompute.
- README.md:24 "30 of 40 ... median of one broken" -> results/part176/README.md:1,13,17: confirmed by scored_directions.jsonl recompute.
- README.md:24 "known-issue list above" -> README.md:14-19: list includes 25 and 66 but README.md:19 says both rerun and holding (Part 183): not consistent (stale).
- README.md:19 "Parts 25 and 66 (Part 183)" -> results/part183/README.md: confirmed (43 vs 42 of 192; 2 vs 1 unchanged; best random 1 vs 2 of 24), recomputed from part183/part66-eager/answers.jsonl.
- README.md:19 "18 and 18b (Part 184)" -> results/part184/README.md: Part 18 cells confirmed from part184/part18-eager/steer.json (+2 8/8, unclamped 0/8, -2 0/8 alien). 18b cells not recomputed.
- README.md:25 -> results/part9-think/name-counts.json: confirmed from sysgen9 generation files.
- README.md:26 -> results/part18/chat/steer.json: confirmed.
- README.md:27 -> README.md:1728 (Part 78) table: confirmed from part78/answers.jsonl; Part 78b (README.md:1803) qualifies it (see S4c).
- README.md:28 -> README.md:2264 (Part 82): 0.444/0.076 confirmed; "as much or more" loose (see S5b).
- README.md:29 -> results/part29c/judge-gpt-6-luna-summary.txt: confirmed from judge-gpt-6-luna.jsonl.
- results/part66 cites Part 32 "16 of 24" and Part 183 rerun - not opened beyond Part 183 (out of scope for items 1-6).

### B. short version items 7-11 (README.md:30-34)

- README.md:30 "185 of 1,709 ... 28 of 1,709" -> results/part97/README.md:8-9,30-32. The numbers are confirmed, but they are OLMo 2 1B (Part 92 screen), not the Qwen screen named earlier in the sentence.
- README.md:30 "dose window one step wide" -> results/part117b/score.json `who_slot_shows` (s+1 11/11, s+0.85 2/11, s+0.5 0/11). The numbers are confirmed. The wording "one step wide" is not supported (lower edge only; Qwen slot units, not the OLMo units).
- README.md:31 "101 of 343 and 94 of 344" -> results/part112/NUMBERS.md:18-19. Confirmed, but those lines are labelled "q_name" (Q-name subgroup). The full-atlas values are 109/464 and 103/464.
- README.md:32 "14 / 40 / 64 of 71" -> results/part108/README.md:74,86 and score108.json `all_slot_shown`. Confirmed.
- part108/README.md:74 "Part 92 on 21 units: 8 of 21, 14 of 20 and 21 of 21". The same numbers appear in part106/README.md:43,84. results/part92 has no README, so this was not checked against raw Part 92 data.
- README.md:33 "357 of 741 ... 286 of 741 ... p=0.000241" -> results/part119b/README.md:5,15. Confirmed from raw answers. 357/741 is Part 119's 0.8 run, reused (part119b/README.md:255).
- README.md:33 "1.044, 0.961 to 1.129" -> results/part119/README.md:3,14,50,86-87. Confirmed.
- README.md:34 "0 of 16" -> results/part114/README.md:4-9. "11 of 16", "16 of 16" -> results/part115/README.md:13,36-38. Confirmed.
- part117b/README.md:3 "Part 117 ... removed the slot's self-description ... in all 11 pairs" -> part117/README.md:24,45 (slot persona 0 of 11 with concept -1). Confirmed.

### C. Parts 177, 178, 181, 180b, 183, 184

- README.md:1196 "against 42 here" -> results/part66/full/score.json (target heldout_google 42, 0.21875): confirmed.
- results/part183/README.md:9-11 SDPA column (unchanged 1, best random 2 of 24, positive 8 of 24) -> results/part66/full/score.json: confirmed (1; max random 2; positive 8/24).
- README.md:719 / results/part183 "Who created you? only" (Part 25) -> stored SDPA answers in results/part183/part25-eager/answers.jsonl (`answer_sdpa`, `maker_sdpa`), from Part 25's files: confirmed, 1-of-6 pattern.
- README.md:1196 "Holds" vs Part 66 registered 66.2 (PREDICTIONS.md:1873, README.md:1210): **not confirmed**. 66.2 fails on eager (2/192 > 1%).
- README.md:102/:382/:2731 "(was 7)" / "against 7" -> PREDICTIONS.md 18b.5 (line ~688, "7 of 8") and results/part18b/phase-gen/steer.json (score_text 7/8): confirmed.
- PREDICTIONS.md:3415 184.4 "(published 2 of 8)" -> PREDICTIONS.md:665 18.2 "2 of 8 name Qwen" and results/part18/chat/steer.json recount 2/8: confirmed.
- 0 of 160 controls and 8 of 8 at +2 -> PREDICTIONS.md 18.3 / 18.4 and stored steer.json recount: confirmed.
- results/part184/README.md:53-54 "Part 18's controls at -2 name Qwen in ... 20 of 20 on SDPA" -> PREDICTIONS.md:665: confirmed. It contradicts "none in a published cell" in the same README (line 49).
- README.md:2727 / 2731 "Part 178 found the A770's default SDPA kernel leaving the CPU's path past about 64 keys" -> PART178-DESIGN.md Amendment 1 and results/part178/README.md:100: consistent (not re-derived).
- Part 180 (cited by Part 180b): **does not exist in the repo**. There is no PART180-DESIGN.md and no results/part180. PART180B-DESIGN.md:12 says "Part 180 (not yet public)", and README.md never mentions Part 180.

### D. Parts 185, 186, 187, 188, 189, 189b

- README.md:2733 / PREDICTIONS.md:3423-3428 (Part 84/84b published 24, 24, 0, 0, 17; 4,25,43; 3,23,46; p 1.0, 0.69, 0.25) -> results/part84/reading, results/part84b/reading + private/key.json, results/part84b/score.txt: confirmed by recompute.
- README.md:2474 "hand-checked SDPA count is also 13" -> results/part84/reading/labels-checked.jsonl; README.md:110 ("13 after my registered second pass"): confirmed.
- README.md:2520 "SDPA was level" -> Part 84b pooled 5/5: confirmed.
- README.md:2735 "published 46", "published 1"; README.md:120 "published 125", "185" -> results/part79/main (answers, reader, key; recomputed 46, 1, 125, 182 SDPA) and README.md:118, :1913: confirmed (185 is the eager e1 value, correctly labelled).
- README.md:2737 "published 48.3%", "published 5.8%", "published 6.1%" -> results/part140/README.md:3, :5, :24, :30 and recomputed from results/part140 answers + labels: confirmed.
- README.md:2743 / results/part189b/README.md "74.8%", "37.6%", "37.2" and six verdicts -> results/part148/README.md:13-18, PREDICTIONS.md "## Part 148" rows (148.1 failed, 148.2 failed, 148.3 held, 148.4-148.6 failed), results/part148/score.json, and recompute from results/part148/answers.jsonl + labels.jsonl (348/465, 175/465, gap 37.2): confirmed.
- README.md:2741 "Part 112's stored text" -> results/part112/answers.jsonl: full-text counts confirmed (5, 6, 32).

## Each auditor's own problem list (with proposed wording)

The ranked list above consolidates these; it is kept here for the exact wording and further detail.

### A. short version items 1-6 (README.md:24-29)

1. README.md:24, last sentence is stale. "Parts 25 and 66 are on the known-issue list above." contradicts README.md:19 and the dated notes at README.md:719 and :1196. Replace with: "Both were rerun on the eager attention kernel (Part 183, 2026-11-04): 43 of 192 zeroed against 2 of 192 unchanged." Also fix README.md:15 (move "Parts 25 and 66 underlie Finding 1." out of the at-risk paragraph or append "; both now rerun and holding, Part 183") and README.md:73 (drop 25 and 66 from "are on the high-risk list ... will get a dated note when it is rerun", or say "Parts 25 and 66 have been rerun (Part 183) and hold; Parts 70, 73, 77, 148, ... remain on the high-risk list").
2. README.md:27 (item 4) omits Part 78b and the kernel status. The 119-of-192 control is atypical: 6 of 10 further size-matched layer-22 pushes cut Google below half (to as low as 2 of 192; PREDICTIONS.md:2011, README.md:1803). Proposed: append "; ten more size-matched pushes gave 2 to 157 (Part 78b), so that control was not typical. Part 78 is on the known-issue list above and has not been rerun." (Part 78 and 78b both on the high-risk list, README.md:14.)
3. README.md:24, "so the four are the cheapest route to the model's own fallback, not its holders": "cheapest" is not measured anywhere, and Part 175 explicitly does not read a fallback interpretation (results/part175/README.md:88). Proposed: "so pushes that use no selected neuron reach the same Google answer; Parts 70 to 73 put the Google fallback in two late neurons, not in the four." (or drop the clause).
4. README.md:28 (item 5), "two of five random sets cut it as much or more": c3 cut 0.349 against 0.369 and its gap is reliably higher than h1's (paired diff +0.020 [0.012, 0.027]). Proposed: "one of five random sets cut it more (to -0.058) and another nearly as much (to 0.095)." Same fix at README.md:2303 ("do as much (c3, 0.349)" -> "come within 0.02 (c3, 0.349)"); and in the Part 82 heading README.md:2264 ("as much or more" -> "nearly as much or more").
5. README.md:25 (item 2), minor: the "while" contrast pairs Think raw/no-system (16/23) with Think-DPO under the template (23/24); different conditions. Consider: "OLMo 3 7B Think named DeepSeek in 16 of 23 raw answers without a system turn and 16 of 21 under its own Olmo template; Think-DPO did so in 23 of 24 under the template."

### B. short version items 7-11 (README.md:30-34)

1. **Item 7 attaches OLMo results to the Qwen screen (README.md:30).** The sentence opens "in Qwen3.5-4B, the layer-22 screen" and continues "a neuron's lens words matched its pushed persona in 185 of 1,709 ...". Those 1,709 units are OLMo 2 1B Instruct (Part 92 screen, Part 97). The dose window then comes from 11 Qwen slot neurons (Part 117b), which are a third population. Proposed replacement for the second half: "...; what is neuron-specific is which persona appears: in OLMo 2 1B, a neuron's J-lens words matched its pushed persona in [185 of 1,709 cases against 28 of 1,709 shuffled pairs](results/part97/README.md), and in Qwen3.5-4B, 11 selected slot neurons showed their persona in [11 of 11 pairs at +1 but 2 of 11 at +0.85 and 0 of 11 at +0.5](results/part117b/README.md)."
2. **Item 8 omits the Q-name scope of 101/343 and 94/344 (README.md:31; also the Finding 7 body, README.md:166).** These are the 344 Q-name targets only. Over all 465 units the counts are 109/464 and 103/464, and for non-AI targets 0/57. The implied generalisation ("a slot label may partly capture generic self-description") holds for assistant-like Q-name personas, not for themed personas like the OLMo dog. Proposed: "In Qwen3.5-4B, random directions got the target label for Q-name personas (assistant-like names such as 'Q, a language model') in [101 of 343 and 94 of 344 self-question answers](results/part112/NUMBERS.md), but for non-AI personas in 0 of 57, so a Q-name slot label may partly capture generic self-description."
3. **Missing Arc known-issue caveats in items 7 and 8 (README.md:30-31).** Parts 112 and 117b are on the high-risk list (README.md:15). Item 1 sets the precedent ("Parts 25 and 66 are on the known-issue list above."), and the Finding 7 body has the caveat (README.md:168). Items 7 and 8 do not. Proposed: append to item 7 "Part 117b is on the known-issue list above." and to item 8 "Part 112 is on the known-issue list above." Lower-confidence point: Part 86 itself generated Qwen3.5 text in bf16 on the Arc. Its screen meta records a `common86.py` sha256 (62b2929...) that differs from the current file, and the current file sets `ATTN_IMPLEMENTATION = "eager"`. The screen's kernel cannot be confirmed from the repo. Finding 7 (README.md:168) notes that Part 86 is not listed. Someone should confirm which kernel the screen used before relying on 250/9,216.
4. **"a dose window one step wide" (README.md:30) overstates Part 117b.** Part 117b tested only +1, +0.85 and +0.5. It shows the persona disappearing one step below the selection dose; it tested nothing above. Its README calls these "dose controls", not a window measurement. Proposed: replace with "that mostly vanishes 15% below the selection dose" (see the replacement in problem 1).
5. **Item 9 omits Part 108's own reading (README.md:32).** Part 108 registered and found that moving the weights *past* Instruct loses as many units as moving back (43 vs 51; 32 vs 24), so the 14/40/64 counts "do not show DPO building slot units". The short version presents the rising counts plus a caveat against the forward models only. Proposed: "...and 64 of 71 at Instruct, but moving the weights the same distance past Instruct lost as many ([43 against 51, 32 against 24](results/part108/README.md)), so the counts reflect where the units were picked, not DPO building them; the forward weight mixes were not checkpoints."
6. **Item 8 "only 1 of 6 post-introduction answers was in character" (README.md:31) misparaphrases Part 109.** Five of the six restated their persona ("I am a cat."). The 1 of 6 is the by-eye count of answers that *described activities* in character. Proposed: "but only [1 of 6](results/part109/README.md) described activities in character; the other five restated the persona."
7. **Item 10 omits the non-significant one-real-push control (README.md:33).** Real pairs vs A+R controls at 0.8: 357/741 vs 659/1,482, p = 0.104 (part119b/README.md:5). The real and random pairs also came from separate runs (119 and 119b). Optional addition: "; one real plus one random push broke [659 of 1,482](results/part119b/README.md) (p=0.104 against real pairs)".
8. **Stale part README (results/part97/README.md:19).** It says "97.3, the validation push, is not run yet", while PREDICTIONS.md:2308 scores 97.3 held from results/part97/validation. Proposed: replace line 19 with "- 97.3 held: 6 of 10 J-lens-matched neurons showed their theme in 3+ of 6 answers against 0 of 10 unmatched; random directions 0 of 80 pairs ([validation](validation/score.json))."

### C. Parts 177, 178, 181, 180b, 183, 184

1. **Part 66's registered 66.2 flips from held to failed on eager, but the front page says "Holds" and "each published claim holding".** Eager unchanged is 2/192 = 1.04% against a bar of at most 1% (results/part183/part66-eager/score.txt: "66.2: FAILS"). Fixes:
   - README.md:1196: replace "**Holds on the eager kernel (Part 183, 2026-11-04):** rerun with the same code and seeds, 43 of 192 (22.4%) against 42 here, unchanged 2 of 192, best random set 1 of 24" with "**Rerun on the eager kernel (Part 183, 2026-11-04):** with the same code and seeds, 43 of 192 (22.4%) against 42 here and best random set 1 of 24, so 66.1, 66.3 and 66.4 still hold; unchanged names Google in 2 of 192 (1.0%) against 1 here, so 66.2 (at most 1%) fails by one answer".
   - README.md:19: change "each published claim holding: Parts 25 and 66 (Part 183)" to "Parts 25 and 66 (Part 183; every count moved by at most one answer, but 66.2 fails at 2 of 192)".
   - results/part183/README.md:16: change "Both published results stand." to "Both published headline results stand; Part 66's 66.2 (unchanged at most 1%) fails at 2 of 192."
2. **"none in a published cell" is false by the tool's own scorer.** Control unit 8617 at -2 moves Part 18's published "20/20 still Qwen" (PREDICTIONS.md:665; README.md:359 "no control at -2x changed the answer") to 19/20. results/part184/README.md:53-54 already says so. Fixes:
   - README.md:2731: replace "4 of them in a scored outcome, none in a published cell" with "4 of them in a scored outcome; one moves Part 18's greedy controls at -2x from 20 of 20 naming Qwen to 19 of 20 (both texts name Tencent; the SDPA one matched only through a later 'Qwen3.5' mention)".
   - results/part184/README.md:49: change "all greedy and none in a published cell" to "all greedy; one (control 8617 at -2) is in Part 18's published 20-of-20 control cell".
   - Also consider correcting README.md:359 and the PREDICTIONS.md:665 note, because the SDPA 8617 answer itself names Tencent.
3. **PREDICTIONS.md verdicts missing or invisible for four of the six parts.**
   - Part 178 (PREDICTIONS.md:3349-3361) needs a Result column with held x7 / inconclusive / inconclusive / held / inconclusive.
   - Part 181 (PREDICTIONS.md:3369-3376) needs its Result cells filled: held (exactly at bar) / held / held / held / failed / held / failed (both floors 0, uninformative) / failed.
   - Part 180b (PREDICTIONS.md:3384-3394) needs its Result cells filled: failed / held / held / held / held / held / held / failed / held / failed / held.
   - Part 183 (PREDICTIONS.md:3400-3406): change `| Claude || ` to `| Claude | ` on each row so the Result renders in the Result column.
4. **No front-page entries for Parts 180b and 183.** The dated entries jump 178 (README.md:2727) to 181 (:2729) to 184 (:2731), and the index line README.md:337 lists 177, 178, 181, 184 but not 180b or 183. Fixes:
   - Add a link `[180b](results/part180b/) · [183](results/part183/)` to README.md:337.
   - Add an entry for 183, for example: "**[Part 183](results/part183/README.md)**, 2026-11-04. Parts 25 and 66 rerun on the eager attention kernel. All seven predictions held: zeroing the four maker neurons gives Google in 43 of 192 held-out answers (published 42), unchanged 2 of 192 (published 1; Part 66's 66.2 bar of 1% now fails), best random set 1 of 24; Part 25's four still flip only 'Who created you?', word for word."
   - Add an entry for 180b, for example: "**[Part 180b](results/part180b/README.md)**, 2026-11-03. DeepSeek V4.1 Flash, Part 180's reader, on 1,687 answers with gold fixed in advance: right on clear constructed cases; cutting answers to 40 tokens halves its yes count on real neuron answers (12 to 6). Eight of eleven predictions held; 180b.1, .8 and .10 failed."
   - Part 180 itself is unpublished ("not yet public", PART180B-DESIGN.md:12). The 180b entry should say so.
5. **README.md:2727 states the eager-vs-float32 check more broadly than it was measured.** Change "eager bf16 is closer to a float32 reference but not identical to it" to "on Qwen3.5-0.8B eager bf16 is closer to a float32 reference but not identical to it; no CPU reference exists for the 4B".
6. **README.md:2725 changes denominator mid-paragraph.** The paragraph is about T (173), then "113 of the 115 traces with a literal gold hit" is on C (187, gold in the prompt included). Change it to "113 of the 115 capped traces with a literal gold hit (gold-in-prompt traces included)". Optionally add "Claude's six predictions: five held, 177.4 failed".
7. (minor) **README.md:2727 does not say the 1.09 is a full-text measure.** Change "on capped rows alone the 4B ratio is 1.09" to "on capped rows alone, over full text, the 4B ratio is 1.09".

### D. Parts 185, 186, 187, 188, 189, 189b

1. **Part 188 front-page count of failed predictions is wrong.** README.md:2739 says "Four of nine predictions failed"; five failed (188.1, 188.2, 188.3, 188.4, 188.7) and four held. Replace with "Five of nine predictions failed (four held: 188.5, 188.6, 188.8, 188.9)".
2. **Part 148 got no dated eager note although its rerun flipped a verdict.** The banner (README.md:17-18) promises "Each affected part will get a dated note when it is rerun or checked", and 185/186/187 got notes (README.md:120, :1968, :2474, :2520, :2653), but the Part 148 entry README.md:2667 ("Five of six predictions failed") and the Finding that quotes Part 148's counts (README.md:84) carry no Part 189b note; results/part189b/README.md also says Part 148's README still carries the SDPA numbers. Proposed note after README.md:2667: "**Rerun on the eager kernel (Part 189b, 2026-11-04):** five of the six verdicts stand; 148.1 flips from failed to held (name moved in 385 of 465 neurons, 82.8%, against 74.8%), every registered rate rises 4.7 to 11.4 points, random directions most (49.0% against 37.6%), and the neuron-over-random name gap narrows from 37.2 to 33.8 points. [results/part189b/README.md](results/part189b/README.md)". The banner list at README.md:23-24 may also need "148 (Part 189b, one verdict flipped)" or an explicit statement that Part 148 is not on the "each published claim holding" list because of the flip.
3. **PREDICTIONS.md result cells do not render for Parts 187, 188, 189, 189b (and 183).** Rows at PREDICTIONS.md:3445-3450, 3456-3464, 3470-3475, 3481-3485 (and 3400-3406) read `| Claude || <verdict> |`, an extra empty cell under a 5-column header; GitHub discards the overflow cell, so the Result column shows empty. Replace `| Claude || ` with `| Claude | ` on those 33 lines.
4. **Part 188 README contradicts its own hedge.** results/part188/README.md:57 says "so 0.28 says the unit responds to human-made sound", while its point 1 and the front page (README.md:2739) say this is "not established". Replace with "so 0.28 suggests the unit may respond to human-made sound (not established), not that it fails on imitations."
5. (Minor) README.md:2743 and results/part189b/README.md:1 "rises 4 to 11 points": the smallest rise is 4.7 (maker, slot, concept), the largest 11.4. Consider "about 5 to 11 points" or "4.7 to 11.4 points".

## Unchecked and unfinished

The run finished well inside its 75-minute cap. These are the limits of what it could check:

### A. short version items 1-6 (README.md:24-29)

- Part 176 Google counts and broken counts were recomputed from scored_directions.jsonl, not re-joined to labels/labels-ds.jsonl (label ids are hashed answer pairs; hash recipe not reconstructed).
- Part 184 Part 18b cells (gen-only 8/8, prompt-only 0/8) not recomputed.
- Part 25 raw files (results/part25/search-google-removal.json, controls.json) not opened; relied on Part 183 B arm summary.
- Part 66 hand-read claim "all 42 say Google made the model" not re-read.

### B. short version items 7-11 (README.md:30-34)

- The Part 86 per-layer persona counts (250, 3) cannot be recomputed. The item-to-row key `KEY-private.json` is outside the repo. Only the triage counts, the denominators and the label totals (1,755 = 1,713 + 42) were recomputed.
- The Part 119 bootstrap CI was not re-run; I used the exp of score119.json's CI, and recomputed only the point estimate.
- The Part 92 stage counts cited by Part 108 (8 of 21, 14 of 20, 21 of 21): results/part92 has no README, and I did not trace the raw files.
- Which attention kernel the Part 86 screen used (see problem 3).
- Part 117b's "9 of 11 theme rated 2/3" citation of Part 117 was not traced.

### C. Parts 177, 178, 181, 180b, 183, 184

- Part 177's 173 / 187 cohort counts, onset order (49/66), and 113/115 box audit could not be recomputed from raw. The MIT sample files (`inputs/eval/...`) and the tokenizer are not committed; I checked them against numbers177.json only.
- Part 178's window rates (86.22, 19.59, 18.50, 9.98) need the pinned tokenizer for the 2,000-token window. I checked them against eager/score.json, and the ratio arithmetic and full-text capped sensitivity were recomputed from the rows. U/E stored texts are not in the repo.
- Part 181's per-unit top-10 states are not committed. Medians and means come from analysis/overlaps.json.gz aggregates, and the 181.5 k-table was not recomputed.
- Part 180b: only predictions .1, .2, .3, .9 were recomputed from raw. Predictions .4-.8, .10, .11 and the 40-token-cut numbers were checked against SCORE.md and score.json only.
- Part 183 kernel-check.json / kernel-sdpa-control.json claims (6 of 8 SDPA departures at 45-52, margin <= 0.101) were not checked.

### D. Parts 185, 186, 187, 188, 189, 189b

- Part 189 40-token prefix counts (17, 18) and "first differing token 1 to 53": no Qwen tokenizer installed locally (no `tokenizers`/`transformers`); checked only against gates.json/diag.json and a word-level proxy (consistent).
- Part 188 prediction 188.5 (L17 N1574, 0.647): the unit's activations are not in the committed npz files; taken from results.json only. Group-clustered bootstrap intervals not recomputed.
- Part 186 coherent counts and `</think>` counts in the part README (not on the front page) not recomputed.
- Part 185 first-difference key positions (median 61-66, 165 before key 65) not recomputed (needs tokenizer).

### Lead

- When fetched at about 05:51 UTC, origin/main was force-updated from `9ddbbc0` to `fa68418`, and `9ddbbc0` is not an
  ancestor of `fa68418`. That only means the clone's stale `main` ref differed. This audit uses `fa68418` throughout and
  did not investigate the history rewrite.
- The other `||` rows in older sections (Parts 3, 11, 34, 43, 51, 80b, 83) were not inspected. They may be literal `|`
  characters inside cells.
