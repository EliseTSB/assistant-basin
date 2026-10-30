# Part 117 cold check, round 1 (Haiku 5.5, 2026-10-29 21:1x CT, read-only; condensed by Claude, to fold)

Every README number matched the reader's recomputation from the raw files; all eight verdicts, the gate, prediction
wording and all six quoted excerpts were verified. The problems are wording and scope.

1. Title and first paragraph give the matched both +1 counts (concept 9, slot 3 of 11) with no baseline or control.
   Alone, both personas show on "Who are you?" in 11 of 11; unrelated both +1 gives concept 10 of 11 and slot 0 of 11
   (NUMBERS.md), which the prose never reports. "Matched +1 condition" is never defined as both +1.
2. The naming observation needs its baseline. All 11 concept -1, slot +1 "Who are you?" answers open with a bold
   invented name, but bold openers are near universal (11 of 11 in matched concept +1, slot +1 and unrelated concept
   +1; 10 of 11 in random both +1; the unpushed answer opens "I am **Qwen3.5**"), and the random rows, with no persona,
   mostly give invented names too. "Qwen" appears in the "Who are you?" answer in 1 of 11 random both +1 rows and 0 of
   11 in every other pushed condition except both +0.5 (3 of 11). So the names may track persona absence, not concept
   -1. Also: the heading calls registered 117.8 "exploratory"; only the naming pattern is.
3. The negative-slot paragraph rests on six-answer totals. On "Who are you?" alone, concept +1 with slot -1 shows the
   concept persona in 7 of 11 matched and 10 of 11 unrelated pairs, from 11 of 11 alone: a 4-pair drop against 1.
   Possibly noise at n = 11, but say it.
4. The scorer's gate is not the registered gate: score117.py sets the gate from the same test as 117.1 and never reads
   Part 112's kinds; the kind check is only in plot117.py. Both give 11 of 11, so the README's claim holds, but
   score.json's gate cannot fail independently. Say so in Method.
5. Terms used before definition: Part 112 (unlinked), random direction, gate, unpushed, the match-score scale (2 of 3).
   The six questions are never listed, so "capital" and "poem" are unmapped. "One dose per unit" contradicts the
   +0.5, +1, +2 ladder. The reader's prompt carries each target's atlas kind (personas.json); disclose it.
6. The withdrawal rule is misstated: the design says 6 or more of the 66 random answers; the scorer counts 132 labels
   against that threshold. The count is 1 either way.
7. 117.7 uses concept-target coherence only, unstated. Slot-target labels on the same answers give 1 of 66 coherent
   (p03:matched:both+2|s, question 0, the one-word answer "I"). Holds either way. Show one +2 example (loops) and the
   Opus agreement on coherence (145 of 164).
8. Limits omit shared concept neurons: p03 and p11 share q112-071, p07 and p09 share q112-401, so the 11 pairs are not
   fully independent.
9. Part 117b is never mentioned: it is registered (PART117B-DESIGN.md, public main 4d056e9) as the control for the
   concept -1, slot +1 cell. Cite it as the follow-up; its results are not in yet.
10. The figure's "Matched: slot +1 alone" concept bar reads 2 of 10 (a slot-alone leak the prose never mentions).
    "Cold check: pending" line to be replaced. Results are untracked until Claude commits at publication.

Also verified: second reader agreement 156 of 164 (95.1%, kappa about 0.87); swapping in the 8 Opus labels changes no
verdict (worth one sentence).
