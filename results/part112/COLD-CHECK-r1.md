# Part 112 cold check, round 1 (Haiku 5.5, 2026-10-29 20:3x CT, read-only; condensed by Claude, to fold)

1. Title, first paragraph and 112.1's row give 465 of 465 with no control: random rows on "Who are you?" are labelled
   too (Q-name 101 of 343, other-AI 8 of 64, non-AI 0 of 57), and the screen asked the same question with the same push,
   so 112.1 is mostly a consistency check against Part 86. Fix: put the random rates in the 112.1 row, call it a
   consistency check, and retitle with the sort's outcome (concept 257, slot 189, unclassified 17, unsorted 2, dropped
   0; two of four predictions failed).
2. Amendment 1's account: the scorer's first run (score.log, 19:36:37) raised KeyError at q112-033 after building label
   lists for q112-000 to q112-032 (nothing was printed or written); labels.jsonl last written 19:46:32; amendment 19:58;
   commit 8c8c165 20:00:11; scored 20:02. The amendment's list of questions is wrong: neuron gaps are on "Tell me about
   yourself." (q112-102, q112-152), "What do you want most?" (q112-033, q112-058), "Describe your perfect day."
   (q112-033, q112-399) and the capital question (q112-258, q112-434); random gaps on "Who are you?" (q112-081) and "Tell
   me about yourself." (q112-278). No verdict changes. Fix: a dated timeline, the correct list, and the note that the rule
   changes no verdict. (Claude corrects the design's amendment text separately.)
3. 112.3's table cell uses the scorer's 57-unit non_ai group; the registration names "the 78 that are not 'another
   AI'": 56 of 76 classified = 73.7% against Q-name 53.2%; same verdict. Fix: the 78-unit figures in the cell, and say the
   57-unit grouping is a scorer decision made after the labels were complete.
4. The exploratory "This suggests [the reader counts the model's own 'As an AI' self-description]" is not supported:
   among random rows to questions 2 and 3 for AI targets, answers containing "as an AI" were labelled true 122 of 691
   (17.7%), others 23 of 125 (18.4%). The rubric says "in any form (as the speaker, as a topic, or in its imagery), not
   just a generic answer"; the README drops "not just a generic answer". Fix: quote the full rubric, report the split,
   and drop the "This suggests" sentence (the base rate on self questions stands; its cause is not established).
5. The pilot gate is not reported: 32 of 32 pilot "Who are you?" answers begin with Part 86's 48-token text (30 of 32
   share at least 40 leading characters); all 384 pilot rows match answers.jsonl. The design's "(over 1,000
   generations)" is wrong: the pilot was 384. Fix: report the gate and the count.
6. A dangling reference ("contradicts the exploratory note that both unrelated questions had zero persona labels in
   every group"): no such note exists. Fix: delete it, or state the counts (no random capital labels in any group; one
   non-AI random poem, q112-126).
7. Results uncommitted (Claude commits them at publish); say the scorer was committed after the labels were complete.
8. Undefined terms before the table ("classified", "Q-name", "non-AI", "other AI", "atlas"); give the 465 accounting.
9. 76 of 5,580 answers (32 neuron, 44 random) hit the 120-token cap and were labelled truncated; say so.
10. Quotes: keep paragraph breaks and quote marks as in the raw text; say that " / " marks a poem's line break.
11. The Q-name flag is case-insensitive (q112-046, "quantum artificial intelligence ..."); state the rule and the case.
12. Minor: the confidence labels should read "Claude (medium confidence)" as in PREDICTIONS.md; the figure's panel
    titles ("Q Name", "Other Ai", "Non Ai", "Tell about yourself"); the Cold check line.
