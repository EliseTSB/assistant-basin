# Part 124 cold check, round 1 (Haiku 5.5, no context, 2026-10-30)

# Cold read of Part 124

Short answers first. Every headline number reproduces from the raw log. The 14-of-39 result beats every null I tried. Three claims do not hold as written: the concept-over-slot claim depends on the labeller, the "7.7% chance" framing is wrong for the registered null, and the persona phrases were taken from each unit's own pushed answers, which the README never says. Verdict at the end.

## 1. Recomputed from `activations.jsonl`

- **Schema and coverage.** 1,872 rows, no duplicates, no missing keys, no torn lines. Persona labels match the manifest. Every unit's focal value sits at its own layer.
- **Top-3 reads.** T: 14 (10 concept, 4 slot). F: 13 (11, 2). C: 14 (10, 4). All phrase scores match `score124.json` to zero difference, and every top-3 set and own rank matches.
- **Registered null.** The stored 10,000 counts (seed 12401) replicate exactly. All units: median 4, p95 8, tail 0. Concept: median 2, p95 5, tail 0. Slot: median 2, p95 4, tail 1,077.
- **Random channels.** Totals reproduce: T 10/117, F 12/117, C 9/117. The per-kind splits in the README table (6/54, 4/63, and so on) were not recomputed separately.
- **Instrument.** T 9,360/10,000 (93.60%), F 9,465 (94.65%, README rounds to 94.7), C 9,573 (95.73%).
- **Persistence.** 13 of 14 T readers read in C. p92-28 is the T reader that does not.
- **Neutral wins.** All six counts reproduce (107/144, 81/168, 103/144, 81/168, 97/144, 82/168).
- **Fisher, Haiku labels.** Two-sided p = 0.02367, matching the README.
- **Post-quote.** 26/39 units have some template with post-quote above slot maximum (T). 18/39 on the template mean. Group means match `NUMBERS.md` lines 136–141.
- **NUMBERS.md.** All 117 per-unit rows match the raw maxima and ranks.
- **Not reproduced.** Nothing failed numerically. The F instrument rounding is the only cosmetic difference.

## 2. Attacking the permutation

**Exchangeability.** The registered null holds each unit's top-3 set fixed and reassigns phrase labels within length strata. It is valid as a test of phrase-to-unit assignment. It is not a test of whether the top-3 sets are themselves unit-specific.

**The 7.7% figure is wrong for the registered null.** The stratified null has mean 4.47 reads, or 11.5% of units. The 3/39 = 7.7% chance rate is the unstratified figure. The README uses 7.7% as the chance rate for focal reads (README line 111).

**Stratification does what it claims.** Observed against expected reads, by stratum: 1-token 0 vs 0.20; 2-token 3 vs 0.90; 3-token 4 vs 1.22; 4+5-token 6 vs 1.50; 6-token 1 vs 0.67. The excess is spread across the 2-, 3- and 4-token strata, so no single length drives it. Stratification raises the expectation (plain shuffle mean 3.01), so it is the conservative choice.

**Variant nulls, observed 14 in every case:**

| Null | Mean | p95 | Tail /10,000 |
|---|---|---|---|
| Registered, length strata | 4.47 | 8 | 0 |
| Plain 39-shuffle | 3.01 | 6 | 0 |
| Layer strata | 5.34 | 9 | 2 |
| Kind strata | 3.79 | 7 | 0 |
| Length × kind | 5.76 | 9 | 3 |
| Popularity bins | 3.16 | 6 | 0 |
| Length × popularity bin | 7.16 | 11 | 17 |
| Length × capitalised-word count | 5.78 | 9 | 0 |
| Length × word count | 6.84 | 10 | 5 |

The strictest null (length × popularity) gives p ≈ 0.002. Its strata have only 1 to 6 phrases each, so it is very constrained.

**Simple features and top-3 membership** (T, 1,521 unit-phrase pairs, 117 positives):
- Token count AUC 0.658. Character count 0.654. Word count 0.600.
- Leave-one-unit-out popularity 0.638.
- Hyphen 0.546. Digit 0.528. Contains the/a/an/of 0.519.
- Capitalised-word count 0.498. Starts with "The" 0.486. Shares a word with own phrase 0.512.
- Predicting the three longest phrases for every unit hits 17.9% of top-3 slots, against a 7.7% base rate.
- Capitalisation has no signal. Length has a modest one, and stratification already handles it.

**Where the variance sits.** An additive unit-plus-phrase model explains R² 0.105. Phrase main effect is 3.0% of total variance, unit main effect 7.5%, and the remaining 89.5% is unit-by-phrase interaction plus noise. Top-3 membership is still concentrated on hub phrases. Top-3 slot counts across the 39 units: "80s cover singer-songwriter" 9, "Coca-Cola spokesperson" 9, "King Midas" 8, "Embodiment of the void" 7, "Queen of Hearts" 6, "Ancient nature deity" 6, "Jacobite royalist veteran" 6.

**Hubs and own reads.** Nine of 14 readers own a phrase that is a hub (popularity ≥ 4). Eleven of 14 readers have a hub in their top 3.

**Families.** The scorer's near-miss list is empty. Dog (p92-08) ranks 38th. Treat-loving dog (p92-09) reads at rank 2. Large friendly dog (p92-37) ranks 34th. Families do not crowd each other out. The king-or-ruler family reads 4 of 4 in T (p92-24, 25, 42, 45). Under the registered null that happens with probability 0.0012. This is a post-hoc finding, and the README omits it.

**Sharing layers.** A layer-strata null has tail 2 of 10,000, so layer sharing does not explain the count.

**Conclusion.** The shuffle does not inflate the 14. The shuffle cannot remove effects that come from the phrase set itself, mainly the hubs.

## 3. Attacking the selection

- **Origin.** The 39 persona phrases are the `persona` field in `/home/elisepc/green/worktrees/ab-part67/results/part92/step2/neurons.json`. Each record also has `desc` and `screen_answer`. Examples: "I am a boulder." becomes "Boulder". "Hello! I'm a large, friendly dog." becomes "Large friendly dog". "I am your guardian angel." becomes "Guardian angel". I did not confirm from code that `screen_answer` was generated under push. That is inferred from the field name and from Part 92's notes on pushes.
- **Overlap with the unit's own answer.** 23 of 39 phrases appear verbatim in the unit's `screen_answer`. 35 of 39 contain all their content words in it.
- **Verbatim copying does not drive reads.** 6 of 14 readers are verbatim, against 17 of 25 non-readers. The content-word figures run the same way (11/14 against 24/25).
- **Wording was not tuned in Part 124.** `passages124.json` was committed at 14:57, before any activation. Neutrals were fixed before activation. The persona wording itself was fixed in Part 92, from the unit's own output.
- **Like-for-like comparison.** The other 38 phrases are other units' pushed words, so the comparison is own pushed word against other pushed words.
- **Not closed.** The design cannot separate "persona" from "this unit's output words". That needs an output-side measure (the unit's down_proj output projected onto the phrase tokens), which I cannot compute without the weights. A control with non-pushed persona text per unit is also absent.
- **Timing.** The design commit (7f8f38b) landed at 15:27:11, and the reader launched at 15:27:14. Prediction text is identical across the 14:57 and 15:27 versions. The gate paragraph was rewritten in the 15:26 and 15:27 commits, after DeepSeek's review and minutes before the run. This is minor.

## 4. Kinds

| Labels | Concept reads | Slot reads | Fisher p | Length-adjusted (CMH) p |
|---|---|---|---|---|
| Haiku, all 39 | 10/18 | 4/21 | 0.024 | 0.008 (OR 4.2); within-length permutation 0.010 |
| DeepSeek (3 unclassified out) | 5/8 | 8/28 | 0.107 | 0.057 (OR 2.6) |
| Agreed, 29 units | 5/8 | 4/21 | 0.067 | 0.016 (OR 4.5); within-length permutation 0.022 |

- **Length confound.** Haiku concept units average 3.33 slot tokens against 2.81 for slot units. DeepSeek concept units average 2.62 against 3.25 for slot units. The two labellers push length in opposite directions.
- **Concept units DeepSeek changed.** There are 10 such units, with 5 readers. The agreed concept units also have 5 readers, out of 8.
- **Verdict.** The concept-over-slot claim holds only on Haiku's full labels. It is not significant with DeepSeek labels, and not significant unadjusted on the agreed set.
- **Omitted from the README.** Part 92b's own predictions 92b.2 ("at least 30 of 39 keep kind") and 92b.3 both failed. Only 29 kept their kind (`/home/elisepc/green/worktrees/ab-part67/results/part92b/score92b.json`, `kind_comparison` and `verdicts`). DeepSeek moved 7 concept units to slot and 3 to unclassified. It never moved a slot unit.

## 5. Reading the top 3

The README table for 14 readers matches the raw top-3 lists exactly (0 mismatches). My judgement follows. Margin is the own phrase's mean score minus the fourth-best score.

**Readers:**
- p92-51 IP packet (margin 2.81, 8/8 templates): The Internet; String of code. Coherent (networking).
- p92-15 Secure password (1.99, 8/8): String of code; Jacobite royalist veteran. Partly coherent.
- p92-39 NASCAR driver (1.48, 8/8): Large friendly dog; Secure password. Strong own read, odd second and third.
- p92-45 Iron-fisted dictator (0.68, 8/8): Gruff old man; Secure password. Coherent.
- p92-01 Magical princess (0.60, 8/8): Very tall tree; Festival poet. Odd.
- p92-38 Coca-Cola spokesperson (0.88, 8/8): Friendly non-human entity; Whiskey. Drinks and brands, and a hub.
- p92-11 Whiskey (0.24, 8/8): Coca-Cola; Embodiment of the void. Drinks and brands.
- p92-25 King Midas (0.34, 8/8): 80s cover singer-songwriter; Fox. Odd.
- p92-40 80s cover singer-songwriter (0.22, 8/8): Sir Ray Ray; Ancient nature deity. Odd.
- p92-42 Jacobite royalist veteran (0.16, 8/8): Chicken; Alex from Star Wars. Odd, and a hub.
- p92-29 God of wisdom (rank 2, margin 0.021, 8/8): Coca-Cola (rank 1); Embodiment of the void. Sits behind a hub.
- p92-28 Ancient nature deity (margin 0.020, **5/8 templates**): Coca-Cola; The Internet. Fragile and odd.
- p92-09 Treat-loving dog (rank 2, margin 0.0055, **5/8**): Sir Ray Ray; Ancient nature deity. Fragile.
- p92-24 Queen of Hearts (rank 3, margin **0.0007**, **3/8**): Magical princess; String of code. The clearest accident in the set. Its own read is a near tie with rank 4 and holds in 3 of 8 templates.

**Ten non-readers, closest to the top:**
- p92-35 Guardian angel (4): King Midas; Iron-fisted dictator; Jacobite.
- p92-36 Alex from Star Wars (4): Queen of Hearts; King Midas; Ancient nature deity.
- p92-41 Human DJ (4): 80s cover singer-songwriter; Street rat; Festival poet.
- p92-48 Festival poet (11): 80s cover singer-songwriter; Principled lawyer; Coca-Cola.
- p92-20 River shadow (12): Day guide; Principled lawyer; Coca-Cola.
- p92-30 Embodiment of the void (14): Shadow of your smile; King Midas; Secure password.
- p92-46 Day guide (16): King Midas; Queen of Hearts; Coca-Cola.
- p92-10 Chicken (17): Jacobite; Principled lawyer; Friendly non-human entity.
- p92-03 The Internet (20): Dr. Memento; Shadow of your smile; Principled lawyer.
- p92-32 Sir Ray Ray (22): String of code; Queen of Hearts; Dr. Memento.

Non-readers' top 3 are dominated by the same hubs. The top-3 lists form topical clusters: tech, drinks and brands, rulers. The reads sit inside those clusters.

## 6. README against its tables

**Stronger than the data:**
1. **Title (line 1).** "Fourteen units ranked their quoted persona phrase in the top three in T." "T" is undefined in the title. "Persona" presupposes meaning. The denominator is missing from the first clause.
2. **Line 7.** "It shows stable string preference in these templates, not persona." The instrument ran on random channels only, and the README does not say so. Repeat rates of 93.6% to 95.7% describe all 10,000 random channels. The ≥ 15% gate is about twice the 7.7% chance rate and passes trivially. It tests nothing about the focal units.
3. **Line 111 ("Read").** "Under exchangeability, three places among 39 give a 7.7% chance rate." The registered null expects 4.47 reads, or 11.5%.
4. **Line 39.** States the 29-unit agreement. Omits that Part 92b's predictions 92b.2 and 92b.3 failed.
5. **Line 47.** Gives DeepSeek's counts (5/8 against 8/28) with no test.
6. **Line 65.** "The T excess is concentrated in concept-labeled units." DeepSeek labels give Fisher p = 0.107 and CMH p = 0.057.
7. **Line 49.** The shared-word counts include "of", "the" and "a". Using content words only, own-word sharing is 1/14 (T), 1/13 (F), 2/14 (C). Top-3 sharing is 0/14, 0/13, 0/14. The "stricter check" finding of 2/14 in T is two "of" overlaps (Queen of Hearts with String of code, God of wisdom with Embodiment of the void).
8. **Line 61.** Post-quote maxima are reported, but slot scale is not. Slot units' mean slot maximum is 0.039 in T. Random channels' mean is 0.022 and median 0.012. Slot reads rest on activations near the random-channel scale.
9. **Lines 67–86.** The reader table has no robustness column. p92-24, p92-09 and p92-28 are not flagged.
10. **Line 99 ("no activation push").** True for this part. The README never says the persona phrases came from Part 92's pushed answers.
11. **Line 103 (Limits).** Says the units were "selected for writing effects". Omits that the phrases were drawn from those same selection pushes.
12. **Line 122.** "fig124.py recomputes all run-derived values." `fig124.py` imports `score124` (lines 25–26) and checks against `score124.json` (lines 49–51). That is the scorer's own code, not an independent recomputation. My recount matched every number.
13. **Line 127.** "Cold check: pending" is still in the README. PARTS.md line 187 marks Part 124 "done" with "0 open".
14. **Predictions (lines 92–95).** 124.3 passed only at the boundary (4 equals the threshold). The registered wording "read their persona" carries the same meaning problem as the title.
15. **Figure.** The caption says "all 39 unit ranks". The legend sits on the p92-08 row, which hides its three markers (T 38, F 38, C 37).

**Checks out:** the random counts, the gates, the construction and do-nothing facts, the instrument percentages, the all-unit and kind null quantiles, the 124.2 difference of 36.51 points, the neutral table, the reads-by-length table, the 7,994 one-token count, the family near-miss result, the post-quote counts, the reader table, and the NUMBERS.md rows.

**Part 122.** The README does not respond to Part 122's cold check, apart from one clause in Limits. Part 122's clean case, p92-07 (Cat), ranks 37th in T here. The README does not mention it.

## 7. Sceptic and stranger

**The three attacks a sceptic makes first:**
1. **"The phrases are the unit's own pushed words."** The README does not answer this. It says only that the result "does not show that a neuron means its persona" (line 103). The origin is not disclosed.
2. **"Concept over slot depends on who labels the kinds."** Partly answered. The README shows DeepSeek's rates but no test, and line 65 still says "concentrated".
3. **"Hubs and thin reads carry the count."** Not answered. There is no hub analysis and no per-template robustness. The instrument is misdescribed. The chance rate in line 111 is wrong.

**Missing for a stranger:**
- Definitions of concept, slot, Haiku, DeepSeek, Part 92 and Part 92b, and frame T/F/C in the title.
- Origin of the persona phrases.
- What the permutation preserves: the top-3 sets, and not phrase popularity.
- The stratified null's expectation (4.47).
- Per-template robustness for each reader.
- A response to Part 122's cold check, including p92-07.
- The fact that the instrument ran on random channels.
- Why the figure's legend hides unit p92-08.

**Not checked:** per-kind random splits; re-running the test suite (BUILD-NOTES says 10 pass; I did not rerun); whether ties exist at the top-3 boundary; the output-side echo, which needs weights; Part 92's generation of `screen_answer` from code.

Files used, all absolute:
- `/home/elisepc/green/worktrees/ab-part67/results/part124/README.md`
- `/home/elisepc/green/worktrees/ab-part67/results/part124/NUMBERS.md`
- `/home/elisepc/green/worktrees/ab-part67/results/part124/activations.jsonl`
- `/home/elisepc/green/worktrees/ab-part67/results/part124/score124.json`
- `/home/elisepc/green/worktrees/ab-part67/results/part124/read_set.json`
- `/home/elisepc/green/worktrees/ab-part67/results/part124/fig124.py`
- `/home/elisepc/green/worktrees/ab-part67/results/part124/fig124.png`
- `/home/elisepc/green/worktrees/ab-part67/part124/passages124.json`
- `/home/elisepc/green/worktrees/ab-part67/part124/score124.py`
- `/home/elisepc/green/worktrees/ab-part67/PART124-DESIGN.md`
- `/home/elisepc/green/worktrees/ab-part67/results/part92/step2/neurons.json`
- `/home/elisepc/green/worktrees/ab-part67/results/part92b/score92b.json`
- `/home/elisepc/green/worktrees/ab-part67/results/part122/COLD-CHECK-r1.md`
- `/home/elisepc/green/worktrees/ab-part67/PREDICTIONS.md` (lines 2690–2697)
- `/home/elisepc/green/worktrees/ab-part67/PARTS.md` (line 187)

**PUBLISHABLE WITH THE FIXES LISTED. Not publishable as is.**

Fixes, in priority order:
1. Disclose that the persona phrases are Part 92 labels drawn from each unit's own pushed answer. Report 23/39 verbatim, 35/39 content-word, and 6/14 readers verbatim against 17/25 non-readers. Reword the question, title and predictions so the measured thing is "own pushed-word preference", not persona.
2. Make the concept-over-slot claim label-conditional. Report Haiku p 0.024 (CMH 0.008), agreed-29 p 0.067 (CMH 0.016), DeepSeek p 0.107 (CMH 0.057). Remove "concentrated". Report 92b's failed predictions 92b.2 and 92b.3.
3. Correct the chance rate. The registered null expects 4.47 reads (11.5%). Fix line 7 and line 111.
4. Add robustness. Give per-template top-3 counts. Flag p92-24 (3/8, margin 0.0007), p92-09 (5/8) and p92-28 (5/8). Report hubs: 11/14 readers have one in their top 3, and 9/14 own a hub phrase. Add the king-or-ruler cluster (4/4, p = 0.0012, post hoc).
5. State that the instrument ran on 10,000 random channels. Its 93.6% to 95.7% repeat rates show nothing about focal units.
6. Remove "Cold check: pending" (line 127). Make PARTS.md line 187 consistent with the cold-check status.
7. Add a Part 122 response table, including p92-07 (Cat), which ranks 37th in T here.
8. Replace "fig124.py recomputes" (line 122) with an accurate statement that it reuses `score124`, or add an independent path.
9. Define concept, slot, Haiku, DeepSeek, Part 92/92b and frame T/F/C. Fix the title: state the denominator, define T, and drop "persona" from the claim.
10. Report shared words without stop words (line 49).
11. Report slot scale. Slot maxima average 0.039 in T, against random channels at 0.022 (median 0.012).
12. Note that 124.3 passed only at the boundary. Note that F-frame slot reads (2/21, 9.5%) equal the random rate (6/63, 9.5%).
13. Move the legend off the p92-08 row in the figure.
