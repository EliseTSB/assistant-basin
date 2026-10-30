# Part 115 cold check, round 1 (Haiku 5.5, 2026-10-29 19:3x CT, read-only; condensed by Claude, to fold)

1. Title "then calls itself Tongyi": at 200 the first word is Tongyi on 9 of 16; 5 of those 9 are "Tongyi Wanxiang"
   (phrasings 3, 4, 6, 9, 14; Alibaba's text-to-image model name); 5 phrasings still say Qwen3.5. Retitle: "on 9 of 16
   the answer then starts with Tongyi"; count Tongyi and Tongyi Wanxiang separately.
2. "Still named Alibaba as its maker" / "keep Alibaba ... and at 500 too": wrong. At 200, 2 of 11 flipped continuations
   name Alibaba (3 with "Tongyi Lab"); the rest end inside the 12-token window. At 500, 3 of 16.
3. "The name changes before the maker does": unregistered, not marked exploratory, and untestable here (the unzeroed
   continuations stop at "developed by", so there is no maker baseline in 12 tokens). Drop it or mark it.
4. "These neurons also push the chat model's habit of bolding": the water opening goes the other way (" **" -6.876,
   -5.38, -5.754, -6.379 at 0/100/200/500); no matched controls in the diagnostic. State it as a hypothesis.
5. "None of 90 random sets ... moved it": true for first place only; they shift log-odds -0.77 to +1.31 at 200. Say
   "took Q off first place on no phrasing".
6. Pair additivity "1.16 against 1.29" mixes the unbatched (6.70) and batched (6.74) baselines; state both.
7. k = 200 is only bracketed on the grid: the threshold lies in (100, 200].
8. Report that the withdrawal rule was checked (max random-set flips at 200 = 0, not triggered).
9. 115.5 "held": literal only (+10.0 Paris, +8.0 cold, +1.4 space token, 0.0 oxygen); say so in the verdict cell.
10. Table header "Random sets: most phrasings flipped" is the maximum over the 10 sets; say "by any one random set".
11. Heading "broken baseline": the baseline is accurate; the targets were not the plain model's first choices.
12. Gloss "Tongyi (Qwen's Chinese brand name)" does not cover "Tongyi Wanxiang"; the 通义 = Qwen gloss is unconfirmed.
13. Limits: 10 random sets per size with fixed seeds is a small sample; the 12-token window cannot test the maker.
14. Undefined or unsourced: "The same question" (Part 114 held-out phrasing 1); the "nothing" row comes from
    results114.json pair_generations.none[0]; list the four ordinary openings; quote the override reason from
    launch.jsonl.
15. The diagnostic and the main run differ slightly on " cold" (0.10 at 200); note that the runs are not identical.
16. PART115-DESIGN.md says "Registered 2026-10-29 19:05 CT" but the registration commit 900e14b is 19:15:32 (the header
    took the writing time); cite the commit.
17. PROCEDURE.md's lessons note says the baseline "was never run": it was run (the none row) and not read before
    registration.

Verified by the reader: every table row, counts 3/11/16, first-word counts, all six verdicts, 5.18, 8.42, 62%, the
gates, the quotes, the method, and the design structure (90 controls, same-layer swaps, none in the top 500).
