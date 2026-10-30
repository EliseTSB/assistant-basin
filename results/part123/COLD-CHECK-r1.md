# Part 123 cold check, round 1

Haiku 5.5 with no context, 2026-10-30. Saved verbatim.

## Cold check of Part 123 (read-only)

**Bottom line.** Every number in the README reproduces from the raw files, and no prediction is read. The problems are in what the README says about the one vector "show", the zero-hook gate, the pilot history and reader reliability. In those places the prose is stronger than the tables.

## 1. Reproduction

All of these match the raw files or the cached weights. Nothing failed to reproduce.

- **Dose table (all 18 rows, NUMBERS.md lines 58-77, README lines 52-71).** Rebuilt by joining `main/answers.jsonl`, `labels-ds.jsonl` and its keys: 4,212 joined, 0 hash mismatches, 0 missing or errored labels.
- **Gate 2:** 21 persona marks on Q0 for slot units at neuron q=1, 20 coherent.
- **Gate 3:** 6 units with three or more raw marks, 4 with the coherent rule, 45 marks and 34 coherent. I derived these from the per-unit listing and from the dose table minus the slot counts (67-22=45, 55-21=34).
- **Gate 7:** the 7 failing groups in README lines 38-46 are correct. I checked all 18 groups by hand.
- **Gate 5:** 12/12 for each reader, from `label119.gate_result`.
- **Gate 6:** 229/254, κ=0.8031.
- **Stage A:** I recomputed all 39 cosines and ranks from the cached OLMo `down_proj` weights and the saved vectors. The maximum cosine difference is 8e-8 and every rank matches. Joint hits are 2/21, top-decile 3/21, above-p95 2/21, median ranks 2623 and 3017, U=184, one-sided p=0.44956 (asymptotic), 0.45015 (exact).
- **Predictions 123.1 to 123.6 (README lines 13-18):** the observed counts match, including 123.2 (0/21 on Q0), 123.3 (20→1), 123.4 (0→0) and 123.6 (0/18).
- **Known examples:** Part 92b's 8/7/3 split matches `score92b.json` (`kind_comparison`).
- **Figure:** consistent with the tables at the level I can read by eye.
- **Not run:** `test_part123.py` and `fig123.py`. I did not rerun any model generation or any label call.

## 2. Findings, most important first

**F1. The p92-49 "show" is reproduced by the plain model and by random directions.** README line 79.
- The unsteered model is already marked as this persona on Q0 and Q1 (its answers are "I am an AI language model..." and "I'm OLMo..."). Of the three q=0.5 marks, two are baseline. The vector adds only Q2.
- The reader's description for p92-49 in `results/part92/step2/neurons.json` is "A large friendly non-human entity here for questions and chat", which describes the default assistant.
- Random directions at layer 12 (p92-49's layer) give 6 p92-49 marks at q=0.85 across three directions, against 2 for the vector. At q=1 they give 3 against the vector's 1. The design's rule (`PART123-DESIGN.md` line 100) says a random match blocks a vector-specific reading. Random doses start at q=0.85 (line 48), so q=0.5 has no matched random control.
- Luna confirms 1 of the 3 q=0.5 marks (Q1 only).
- Fix: rewrite line 79 to say the marks are matched by the plain model (2 of 3) and by random layer-12 directions. Say the unit is not vector-specific and that Luna confirms 1 of 3.

**F2. The 90% reader agreement is a sample figure and hides weak agreement on the marks that carry the counts.** README lines 31 and 89. `label123.py` lines 175-193.
- The Luna sample is all 127 DeepSeek positives plus 127 random others. That is half positive against a population base rate of 127/4212, about 3%.
- Among DeepSeek's 127 persona marks, Luna agrees on 103 (81%). One DeepSeek negative is a Luna positive.
- By arm, among DeepSeek positives: neuron 87/91, vector 7/14, random 8/19, plain 1/3.
- My population estimate from the stratum rates is about 98.7%. That figure depends on the sample being representative within each stratum.
- The gate passes as registered. The fix is to report the positive-stratum and per-arm agreement next to the sample figure.

**F3. "No working coefficient" is a slot-only restriction of the registered sentence, and the scorer's own text says the opposite.** README line 3. `PART123-DESIGN.md` lines 96-97. `score123.py` lines 511-513, 591, 611-613. `SCORE.md` line 118.
- The registered sentence applies to "any unit". p92-49 meets the show rule at q=0.5, so the registered condition is not met.
- The scorer's `vector_edge_reading` returns "dose edge may be read subject to registered gates" (`SCORE.md` line 118). That text is wrong for a failed-gate run.
- The slot-only restriction is a real and useful fact. None of the 630 slot-vector answers carries any persona mark, coherent or not. Coherence falls from 181/234 at q=1 to 86 at q=1.5 and 15 at q=2, so the usable window is q≤1.
- Fix: state the slot fact exactly as a restriction, give the usable dose range, and change the `SCORE.md` / `score123.py` template text to "not read: gates 3 and 7 failed".

**F4. The zero-hook gate does not cover the batch shape of the pushed rows.** README line 89. `run123.py` lines 302-306 and 435-446. `main/zero-hook-check.json`. `PART123-DESIGN.md` lines 57 and 113-115.
- All 60 zero-hook comparisons ran at batch size 1 (`batch_size: 1` in the JSON).
- Pushed rows ran in per-layer batches of 17, 25, 33, 49, 57 and 64. Layers 3 and 4 split as 64+17. None of these sizes was zero-hook checked.
- Amendment 1 says bf16 greedy answers can depend on batch shape. The pilot showed this: 5 of 18 zero rows did not match plain (`pilot/zero-hook-check.json`: 13 equal, 5 not).
- Fix: run no-hook and zero-hook pairs at each batch composition the main run used (six sizes, one per layer and question), and report exact matches. Otherwise state the gap in the limits.

**F5. The README omits the registration history.** README has no mention of the pilot's zero-hook result, of Amendment 1, or of the code changes. Relevant facts:
- Amendment 1 (`PART123-DESIGN.md` lines 110-117) says the author read two pilot units' "Who are you?" answers before the amendment. It also changed the zero-hook check and changed the reader to see the persona name and description. The README does not say the reader sees the persona description, and the p92-49 description in F1 is exactly what the reader saw.
- Git history: `b37b5a7` (13:35:34, amendment), `316160c` (13:44:48, "zero-hook gate at one batch shape"). `part123/score123.py` was modified at 13:41:48, `label123.py` at 13:42:50 and `run123.py` at 13:43:33. Main answers were written at 14:04. So the scorer and runner changed after the pilot was read and before the main run.
- The zero-hook check that failed in the pilot (5/18) was replaced after it failed.
- Fix: add a short registration-history paragraph. Say which thresholds and arms were not changed.

**F6. Gate 3 was known to fail under the chosen reader, and the concept stratum is contested.** README line 34. `PART123-DESIGN.md` line 58. `results/part92b/score92b.json`. The Part 92b commit (`1f63458`, 02:53 CT) predates the registration at 13:28 CT.
- Part 92b already showed that DeepSeek keeps 8 of 18 concept units as concepts, and changes 10 of 18. The 14-of-18 threshold therefore could not pass under DeepSeek. Its failure is not evidence about the neuron control.
- The README says the threshold "was set from Part 92's Haiku concept labels". The registered design gives no source for 14. Cite the source or drop the sentence.
- The concept stratum behind 123.5 and 123.6 is the same contested set. Say so in those rows.

**F7. The README's definition of a "readable" show does not match Part 92's rule or the scorer.** README line 99 says it is "Part 92's rule". `PART92-DESIGN.md` lines 54-56: slot is exactly one or two marks, only on Q0 or Q1, with Q0 showing; concept is three or more with no coherence clause. `score123.py` lines 225-233 (slot: any Q0 or Q1 mark, no exclusivity, no Q0 requirement). `score123.py` lines 359 and 561 add a coherence clause to the concept rule. `part92/score92.py` lines 4-7 describe a third version. No count reported here changes under any of these, since no slot unit has a vector mark and gate 3 fails under the raw rule too (6 units, not 14). But line 99 should say what Part 123 actually does.

**F8. Coherence labels are lenient and some are visibly wrong.** `part92/label92.py` lines 16-27 define coherent as "not a loop or word salad".
- In the answers I read, these are labeled coherent: vector q=0.85 p92-46 Q2, p92-40 Q2, p92-48 Q2 and p92-30 Q5 (loops); random layer-12 direction 1 at q=0.85 Q1 ("As an AI, I donunde ... 'thelfant'thelfant's"); vector q=0.5 p92-49 Q2 ("As an AI, I don as humans do not have desires").
- Crude counts: 187 of 2,798 coherent items contain a three-word phrase repeated at least four times in an answer of 20 or more words. A rough "I don" corruption pattern hits 134 answers. I did not audit all of them.
- 47 empty answers (vector 28, random 14, neuron 5) sit in the 234 denominators and are not mentioned. They are incoherent and contribute to the coherence decline.
- Fix: audit a stratified sample of coherent-labeled loops and garbles, report the rate, and mention the empties.

**F9. The poem question carries concept marks.** DeepSeek gives 14 persona marks on Q5 in total, 7 of them on p92-48 ("Festival poet"), across plain, neuron, vector and random arms. A concept unit can meet the three-of-six rule on a poem alone. The design requires per-question and off-target reporting (`PART123-DESIGN.md` lines 60 and 94). The README gives only totals. The per-question data are in `score123.json` (`unit_tables`). Render them.

**F10. Stage A has no chance context.** README line 75. Under the within-unit null, 21 slot units give 1.05 expected above-p95 hits and 2.1 expected top-decile hits. Observed counts are 2 and 3. The joint count of 2 is at chance.
- `part123/fit123.py` line 230 seeds the shuffle RNG with 12301 for every unit, so the 1,000 sign patterns are shared across units. The per-unit counts are not independent.
- Random directions in 2,048 dimensions have cosine standard deviation of about 0.022. The median slot |cos| is 0.015. Most observed cosines are at noise scale.

**F11. The limits sentence is weaker than the findings.** README line 93 mentions one layer and the written size. It does not say the fit uses last-token states only, that there is no layer sweep, that norm-matching is not effect-matching, or that there is no cross-persona control. A cross-persona control would push unit A's vector and read unit B's persona, which is what a specificity claim would need. Persona names run from 1 to 6 words, against a 4-word generic phrase, so the contrast may carry name-length features. None of these is tested.

**F12. Line 83 is an unregistered reading after gate failure.** "Coherent answers fall with dose ... which is evidence that the pushes are acting." Random directions show the same fall (direction 0: 231, 191, 19), so this is disruption, not persona. Reword it.

**F13. Minor.**
- `PARTS.md` line 186 calls the six predictions "open" (they are not read) and reports "p92-49 concept at q=0.5" as an index result.
- `SCORE.md` lines 21-26 cite only gate 3, though gate 7 also failed.
- `score123.py` line 312 and `fit123.py` line 244 use `ceil(0.10 × 8192) = 820`, not 819.2. No unit sits at rank 820, so nothing changes.

## 3. p92-49 and the 20-answer reads

**p92-49 at q=0.5.** The README's description (line 79) matches the text: Q0 to Q2 are AI self-descriptions, and Q2 has the broken "I don as humans". But it leaves out three things. Q2 is garbled and labeled coherent. Q3 answers in a human voice ("My perfect day would start..."). The plain model already produces Q0 and Q1. "Were coherent" overstates Q5, a repetitive poem. The description is fair as a description of the text and unfair as evidence of a vector effect.

**Vector answers at q=1 and q=0.85.** I read 20 at each (`random.Random(20261030)`). I found no clear persona that the reader missed, and no clear mark that looks wrong. The typical outputs are generic assistant self-descriptions, loops, and some empty answers. Two borderline cases: p92-28 Q5 at q=1 ("Sea, O sea, of azure deep") could be read as a nature-deity voice, but it is a loop labeled incoherent, so no count changes. p92-51 Q0 at q=1 ("a program ... a digital entity") is marked no and is borderline. The reader's real misses are coherence: loops labeled coherent (F8).

## 4. Rules-out, alternatives, and "no working coefficient"

**Readings the README rules out without a test:**
- "The pushes act" (line 83). Random directions do the same thing, so that is not evidence about persona.
- That p92-49's show is a persona effect. The README implies it, but the baseline and random layer-12 directions reproduce it.

**Alternatives left out:**
- The default assistant self-description is read as the persona (F1).
- Batch shape (F4).
- Reader leniency on coherence and loops (F8).
- Persona-name length or token count as a feature of the mean contrast (F11).
- Matching the norm is not matching the effect.
- The narrow usable dose window (q≤1) limits the test's power.
- Empty answers (F8).

**Is "no working coefficient" fair?** Literally, for slots: no slot unit received a persona mark at any fitted-vector dose. As a general statement it presumes the fit is valid, and the sweep only tested a window where the answers were still coherent. Say "no slot persona mark at any dose tested for this fit" and keep the README's existing "badly fitted rather than unable" hedge.

**Is "one layer, one size, last-token mean contrast" stated strongly enough?** No. The layer and size are stated. Last-token extraction, the absence of a layer sweep, the absence of a cross-persona control and the name-length confound are not, and they should sit in the headline paragraph.

## 5. What a sceptic attacks first

1. The p92-49 "show": the default assistant's self-description is in the plain model and in the random layer-12 directions, and the reader was told the persona description.
2. Luna's agreement on the vector and random persona marks (7/14 and 8/19), against the headline 90%.
3. The batch-shape gap: the zero-hook gate ran at batch 1, and the gate was changed after the pilot failed it.
4. Post-pilot changes and the pilot answers read before the amendment, none of it in the README.
5. Loops and garbled text labeled coherent, which feed the gate 7 coherence rule.

**Smallest fixes overall:** F1 and F5 (text), F4 (run batch-matched zero rows), F2 and F8 (report stratified agreement and a loop audit). F3 and F12 are wording.

**Verdicts and counts that would change:** none of the six predictions changes, and all stay not read. None of the reproduced counts changes. The one change is the README's sentence that p92-49 "met its show rule", which becomes "met it on DeepSeek's labels only, is matched by the plain model and by random layer-12 directions, and Luna confirms 1 of its 3 marks".