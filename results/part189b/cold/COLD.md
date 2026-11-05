# Cold check: results/part189b/README.md

Checked 2026-11-04 by a fresh reader with the files. Sources read: README.md (189b), score.json (189b), tables.md (189b),
gates.json, load.json, launch.jsonl, answers.jsonl and labels.jsonl (189b), PART189B-DESIGN.md, results/part148/score.json,
results/part148/README.md, results/part148/tables.md, part148/score148.py and part189b/score189b.py / run189b.py.
Every count below was recomputed from score.json `counts` and independently from the per-unit `units` map; both agree.

## Numbers: recomputation

| README line | Claim | Recomputed | Status |
| --- | --- | --- | --- |
| 3 | 465 selected neurons and 465 random directions | 465 units, 465 paired directions (score.json `units`, `units: 465`) | MATCH |
| 7, 56 | 2,796 answers | answers.jsonl 2,796 lines, 2,796 unique (id, unit, question_index) keys | MATCH |
| 12 | wiring arm reproduces Part 112 32 of 32 (28 required) | gates.json / wiring.json `matches 32, compared 32, required 28` | MATCH |
| 12 | eager load requested and recorded `eager` | load.json requested `eager`, recorded `eager`; gates.json `load_eager` passed | MATCH |
| 12 | Gates 1 and 3 pass | score.json and gates.json both `passed: true`, 3 questions | MATCH |
| 14 | eager run matches stored answers first 40 tokens 17 of 32 | gates.json `gate2_eager_prefix_descriptive` 17/32 | MATCH |
| 14 | 5 of 32 in full | gates.json `gate2_eager_full_text_descriptive` 5/32 | MATCH |
| 21 | 189b.1: 5 of 6 the same, held | 148.1 flips; 148.2-148.6 same; 5/6 ≥ 5, so held | MATCH |
| 22 | 189b.2: 148.1 flips (failed to held), failed | 148.1 failed→held; any flip fails | MATCH |
| 23 | name 82.8% (+8.0) | 385/465 = 82.80%; 82.796 − 74.839 = +7.96 | MATCH |
| 23 | maker 45.8% (+4.7) | 213/465 = 45.81%; −41.08 = +4.73 | MATCH |
| 23 | both 40.6% (+9.2) | 189/465 = 40.65%; −31.40 = +9.25 | MATCH |
| 23 | maker among name-movers 49.1% (+7.1) | 189/385 = 49.09%; −41.95 = +7.14 | MATCH |
| 23 | 189b.3 failed | +8.0 and +9.2 outside ±6 | MATCH |
| 24 | random name 49.0% (+11.4) | 228/465 = 49.03%; −37.63 = +11.40 | MATCH |
| 25 | slot 48.1% (+4.7) | 91/189 = 48.15%; −43.39 = +4.76 | MATCH |
| 25 | concept 43.6% (+4.7) | 112/257 = 43.58%; −38.91 = +4.67 | MATCH |
| 25 | 189b.5 held | both within ±6 | MATCH |
| 31 | 148.1 held 385/465, 82.8% | 385/465; ≥ 372.0 | MATCH |
| 32 | 148.2 failed, maker 213/465, 45.8% | 213 > 69.75 (≤15% rule) | MATCH |
| 33 | 148.3 held, 189 neurons | 189 ≥ 5 | MATCH |
| 34 | 148.4 failed, 189/385, 49.1% | ≥ 25% rule | MATCH |
| 35 | 148.5 failed, neurons 82.8%, random 49.0% | 228 ≥ 192.5 | MATCH |
| 36 | 148.6 failed, slot 91/189, concept 112/257 | 48.1% > 43.6% | MATCH |
| 40-41 | 80% line; neurons beat random by 33.8 (37.2 before) | 82.80−49.03 = 33.76; 74.84−37.63 = 37.20 | MATCH |
| 42-43 | +8.0 and +11.4 | as above | MATCH |
| 31-36 | Part 148 column: 348/465 74.8%, 191/465 41.1%, 146, 146/348 42.0%, 175/465 37.6%, 82/189 43.4%, 100/257 38.9% | part148/score.json counts and predictions: failed, failed, held, failed, failed, failed | MATCH |

Registered arithmetic re-derived from the design's wording, not from the scorer: 189b.1 fails at 2+ flips (one flip → held);
189b.2 fails at any flip (failed); 189b.3 bands are [68.8, 80.8], [35.1, 47.1], [25.4, 37.4], [36.0, 48.0] and +8.0 / +9.2 leave;
189b.4 band [31.6, 43.6], +11.4 leaves; 189b.5 bands [37.4, 49.4] and [32.9, 44.9], both inside. The README's table agrees.
The six side-by-side verdicts match part148/score.json one for one (only 148.1 changes).

## Findings

1. **Line 1 (title), "every rate moves up by 4 to 11 points".** True of the six registered rates (+4.7, +4.7, +7.1, +8.0,
   +9.2, +11.4). Not true of every rate in the run: the random-direction maker rate (not scored, descriptive) is
   225/465 = 48.4% now against 197/465 = 42.4% in Part 148's README, +6.0; the concept rate +4.67 rounds to 4.7. The
   title is defensible only if "every rate" means every registered rate; say so.

2. **Line 14, "The eager run itself matches those stored answers in the first 40 tokens on only 17 of 32".** The design
   (line 58) registers this as "the eager answers' agreement with Part 112 (first 40 tokens and full text)". The README's
   number is against **Part 112's** answers, not "those stored answers" — the stored answers named in the previous
   sentence are Part 112's, so the reading is available, but "those" points at the wiring arm's reproduction. Say
   "Part 112's stored answers" explicitly. Value itself MATCH.

3. **Line 42-45, "so the random baseline in 148.5 rose more than the neurons did".** The comparison mixes denominators:
   random +11.4 points is +53/465, neurons +8.0 points is +37/465. As a relative change the random rate rose 30% and the
   neuron rate 11%. The claim is right on points and on count; it is not right as "more" in every sense, and 148.5's
   margin did widen (37.2 → 33.8 points). Add the margin sentence or the count so the reader is not left to infer that
   the verdict is now closer.

4. **Line 44-45, "Whether the kernel changes the answers themselves, or the reader's call on them, was not separated
   here: the answers differ from the stored ones early (17 of 32 match in 40 tokens), and the reader labelled a different
   text."** The 17/32 figure is the pilot's, on one question; it does not describe the main run. On the main run the
   pushed answers differ much more than that: of 2,790 pushed rows, 190 match Part 112's answer for 40 words and 505 are
   identical in full text (my count, word-level proxy for tokens). The sentence is not wrong, but a reader will take
   17/32 as the size of the text change. Give the main-run figure or say "in the pilot".

5. **Line 12, "zero push matches the plain model (Gate 1)".** MATCH as a gate (score.json). Worth recording that the
   plain and zero answers to "Who are you?" are identical to each other in both runs, and that in 189b they differ from
   Part 148's stored plain answer in the last words ("How can I help you today?" against "How can I assist you today?").
   No verdict or gate depends on it; it does show the plain text is not kernel-invariant at the tail.

6. **Line 40-41, "it changes because 74.8% crosses the registered 80% line to 82.8%, not because the shape moved".**
   Supported for the verdict, but 82.8% is 10.4 points above the line, not a near-crossing; the phrase "crosses the
   line" reads as marginal. The margin is 13/465 units. Say so if the intent is to show the flip is small in units.

7. **Line 46, "189b.1 held and 189b.2 failed on the same fact".** MATCH and correctly stated.

8. **Line 47-48, exploratory "unread beyond its totals; not claimed here".** MATCH: score.json `exploratory` holds
   `with_listed_company` (65 neurons, 53 random), company counts and same-company lists (9 and 7). No README sentence
   uses them. Good.

9. **Line 52, "One run per arm; the reader is one model (Flash), checked by no second reader here."** MATCH. Add that
   labels.jsonl has 1,398 rows for 890 moved rows and every moved row has a successful label (no errors, no missing
   `coherent`), which the design's Gate requires and the README does not state. The design expected about 1,400 labels;
   the run has 1,398, so the count is not a shortfall but the README should give it.

10. **Line 53, "Part 148's published README still carries the SDPA numbers; this part does not change it."** MATCH: no
    Part 148 file is edited by this part.

11. **Line 6-7, "(registered 7a270f1)".** Not checked against git (I did not run git); the design file records
    registration 2026-11-04 18:27 CT. Verify the commit hash or drop it.

12. **Line 7-8, "labelled again by Part 148's reader (DeepSeek V4.1 Flash, same brief)".** MATCH: labels.jsonl `served`
    is `deepseek-v4.1-flash`; the design says the served checkpoint is not pinned. The README's caveat list does not
    repeat that limit; add it.

13. **Line 12-13, "All passed".** MATCH for the four registered gates. The README does not mention that gates.json also
    carries two `passed: false` descriptive entries (`gate2_eager_prefix_descriptive`, `gate2_eager_full_text_descriptive`)
    whose `all_passed` is computed without them. A reader opening gates.json will see `false` twice; say they are
    descriptive and excluded from `all_passed`, as the design registers.

14. **Lines 31-36, the side-by-side table.** MATCH in every cell against both score.json files. The Part 148 column
    gives counts only for 148.1 and 189b for 148.2 onward; that is consistent with the README text, not an error.

15. **Line 15, "that is reported only, as registered".** MATCH to design line 58. The claim that it is "as registered"
    is correct; the design also says the eager-prefix comparison is against Part 112, which finding 2 covers.

No sentence claims more than the data show beyond the qualifiers in findings 1-4. No count, rate, delta, verdict or
denominator is wrong.

SUMMARY: Every number in the README recomputes correctly from score.json, the design and Part 148's score.json, and all
six registered verdicts and the side-by-side table are right; the only defects are wording that over-reads "every rate"
in the title, points "those stored answers" at the wiring arm, uses the pilot's 17/32 as if it sized the main run's text
change, and omits the 1,398-label count and the two descriptive `false` gates.
