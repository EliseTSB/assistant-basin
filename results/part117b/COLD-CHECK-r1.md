# Part 117b cold check, round 1

Reader: Claude Haiku 5.5 (a fresh subagent with none of this project's context), 2026-10-29, on `results/part117b/README.md`
as first written. Saved by Claude from the reader's reply (the reader's own tools could not write report files); paths made
repo-relative. 17 problems: 2 blocking, 15 minor.

## Blocking

1. README line 46 says the scorer searched for "the lowercase string `qwen`". `part117b/score117b.py` line 96 lowercases
   the text (`"qwen" in t.lower()`), and README line 13 says case-insensitive. A case-sensitive recount gives 0 of 20
   concept-alone answers and 0 of 1 unpushed answer, so line 46's own "The baseline answer contains it" is false under
   its description. The 10 of 20 holds only case-insensitively. Fix the wording.
2. README line 52 says the "Who are you?" output was read in each condition for p01, all concept-alone rows, and p09's
   random row. `PART117B-DESIGN.md` (Reading rules) requires every condition to be read by eye before counts are trusted.
   The README covers one pair, the 20 concept-alone rows and one of 33 random rows, and does not say the rule was only
   partly met. The post-scoring disagreement with Haiku's `coherent=false` on p01 matched -1 is not labelled
   exploratory, which the design requires. State the coverage and mark that observation exploratory, noting that no
   count changes.

## Minor

3. 127 of 130 second-reader labels are paired, and line 58 gives "125 of 127" with no reason. `labels_second.jsonl` has
   one HTTP 400 error (`p07:c-1,s+1|c`, q3) and two cells with no first-reader label (`p04:n0-1,s+1|s`, q0 and q1).
4. The question is never stated: what Part 117 found (the slot's self-description gone in 11 of 11 pairs) and what 117b
   tests. "Erasure" appears in the title and twice more and is never defined.
5. The title pairs "0 of 11" for the matched condition with "no erasure reading", which implies the matched answers were
   rejected. They were not: matched coherence is 8 of 11, above two thirds. The block is the random control (15 of 33
   coherent). Suggested: "Part 117b: a matched concept at -1 beside slot +1 shows the slot persona in 0 of 11 pairs; the
   random-direction control breaks its answers, so no erasure is read." The H1 also contains a link.
6. `score.json` reports `"dilution": "consistent with the slot's push being diluted"`. That clause belongs to the
   erasure reading, which the coherence floor blocks. The README never mentions the field. Say it is not read.
7. Theme quality is omitted. The design's Limits say the pairs were chosen by one rater, with mostly loose themes;
   `results/part117/README.md` says 9 of 11 theme scores are 2 on a 3-point scale.
8. The layer confound is vague ("often", "most"). `pairs.json` gives 9 of 11 pairs with the unrelated layer different
   from the matched layer, and only 2 of 11 with the slot in the matched concept's layer.
9. The design says the six unpushed answers are compared with Part 117's and reported. `score.json` has
   `none_answers_equal_part117: [6, 6]` (confirmed 6 of 6 identical). The README does not report it.
10. The choice of slot +0.85 is unexplained. The design gives a ~15% dilution heuristic and says it is not a measurement.
11. The conditions are never listed, and no persona is given as an example (nine condition types, 153 rows, 918
    answers). Add one worked example from `pairs.json` (p01: slot "Continuous Deformation of Space-Time (CDST)", matched
    "3D spatial coordinate system", unrelated "spice manufacturer and supplier") and say why the random and non-persona
    controls exist.
12. Undefined terms: greedy, bf16, A770, layer, residual norm, down-projection input, written norm, hook and zero-hook,
    atlas kind, self-claim, "the two self-description prompts" (the second is never named), and Part 112 (never linked).
13. "Cold check: pending (Claude runs it)" must go before publication.
14. The Files list omits `items_second.jsonl` and `design-review/CRITIQUE.md`. `chain.log` holds a local absolute path.
15. `NUMBERS.md` runs the figure with an interpreter path outside the repo and a /tmp `MPLCONFIGDIR`; outsiders cannot
    rerun it as written.
16. `results/part117b/` is untracked; the README's links resolve only in the working tree until it is committed.
    `score117b.py` and `rows117b.py` are unchanged since the registration commit 4d056e9.
17. Style: first person ("I read", "I disagree") in a public write-up; the " / " line-break note links `labels.jsonl`,
    which does not define it, and no quote uses it.

## Checks that passed

- The ten predictions match `PART117B-DESIGN.md` word for word, with the same confidence labels; the ten verdicts match
  an independent recomputation from the design's thresholds and `score.json`.
- Every README number matches a recomputation from `answers.jsonl`, `labels.jsonl`, `labels_second.jsonl` and
  `pairs.json` (table cells, 0 of 31 base-rate failures, 1,296 cells, 3 missing, 32 readable, 918 answers, 153 rows,
  125 of 127, zero-hook 6 of 6).
- "The random push breaks the answers; not read as erasure" follows the design's rules.
- All four quotes are exact prefixes of their answers; the dose transition is labelled exploratory; every relative link
  resolves; the README has no absolute paths; the figure matches the data.

## Not checked

Whether the eye-read happened before scoring (no timestamps); a rerun of `score117b.py` (it overwrites `score.json`;
counts were recomputed from the raw files instead); the seeded 10% sample; the Opus prompt against Part 92's word for
word; `PREDICTIONS.md`; the `NUMBERS.md` commands.
