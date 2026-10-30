# Part 116 cold check, round 1 (Haiku 5.5, 2026-10-29 20:4x CT, read-only; condensed by Claude, to fold)

1. Cold check line pending and results uncommitted (Claude commits at publish; set the line when folded).
2. Readout and generation disagree on a few rows. dla k=50, phrasing 2 "Identify yourself, please.": the readout's top
   token is 'K' (log-odds of Q -0.7213) but the separate batched continuation is "Qwen3.5**, the latest..."; the same
   phrasing reads 'K' at k=100 and continues "Claude**...". At k=500 phrasing 12 reads 'not' and continues "Claude**...".
   score.json: dla k=50 flips 4 but continuations_not_q 3. Part 115's gradient rows disagree at k=500 on phrasings 1, 11
   and 13. Readout runs in 50-row chunks, generation in a 9-row batch (probably the cause; untested). Flips count only the
   single token "Q"; a continuation starting "Qwen" is not counted as a flip. Fix: a note under the table saying so; state
   that 116.2 (k=100) and 116.4 (k=50 mean log-odds) do not rest on the k=50 flip count.
3. The direct-write ranking is one-sided: rank116.py stores attr = -direct, so the top k are the largest positive
   writers (top 100 from 0.0736 to 3.0410; none of the top 500 negative); the most negative writer (L31 N4200, -1.9475)
   is never zeroed. Fix: say so in Method.
4. The activity ranking is by absolute z (|z| 8.64 to 164.16 in the top 500; 260 of 500 have negative z; 348 have
   |z| >= 10); the top unit L28 N2283 has z = +164.155. base_std is not stored. Fix: define it at first use ("absolute
   z-score against the four-prompt baseline, either direction"), report the 260 of 500 split, and note base_std is not
   stored.
5. The all-ones gate covers one phrasing's last-position logits (HELDOUT[0]); say so.
6. Limits drop the design's caveat that the forced opening contains "**"; add it and scope the result to "I am **"
   (Part 115b, results/part115b, tests openings without "**" with a plain-text instruction).
7. Define at first use: log-odds (log(p/(1-p)) of the single token "Q", part114/qneuron114.py), the gradient ranking
   (Part 114's attribution, results/part114/attr114.pt), first-word counts ([A-Za-z]+ after leading "* ", score115.py),
   withdrawal rule and threshold k (from the design), the gates, "matched random sets".
8. Scope: "no flips" is over 10 fixed-seed random sets per k per ranking; L28 N2283's largest single drop is among Part
   114's 20 measured candidates (-1.8982; next -0.9458); call the gradient ranking Part 114's file, not Part 115's.
9. Title is accurate but one-sided; add that the activity ranking needed k = 500.
10. Files list: add part116/score116.py, results/part114/attr114.pt, results/part114/results114.json,
    results/part115/results115.json and results/part115/score.json.
11. Table numbers typed by hand (all correct); generate them from plot116.py's output or paste it.
12. score.json hard-coded the single-drop sum -8.42 (no verdict used it). Fixed by Claude: part116/score116.py now sums
    Part 114's candidates (-8.4164); score.json re-written, and only that field changed.
