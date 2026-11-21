# Part 215 cold check — numbers and labels

Reviewer: worker-01 (role reviewer, deepseek-flash), 2026-11-21. Read-only over the registered full run
(`results/part215/full/`, 2,304 generations + 624 conditional forwards) and the score directory
(`results/part215/score/`). Independent recomputation is in `recompute.py` (stdlib only, run under
`python3 -I`); the scorer (`part215/score215.py`) was **not** imported for it. The scorer's own
`attrib215`/`is_broken` were called only in `compare_labels.py`, a separate comparator process, because
the saved rows carry no scorer labels — that process produced the row labels this check compares against.
Bootstrap: 10,000 question-clustered draws, fixed seed **20261121**, 95% percentile interval with linear
interpolation, one shared draw matrix for every contrast ("same bootstrap draws", DESIGN.md Estimation).
Auxiliary checks: `extra_checks.py`, `parse_scorecard_md.py`, `scan_google_cues.py`, `sample_answers.py`.
Nothing outside `results/part215/coldcheck/` was written.

## SUMMARY

The numbers and verdicts stand. My implementation of the design's frozen lexical rule, written from the
DESIGN.md text (v4.1 + Amendments 1–2), agrees with the scorer's label on **all 2,304 rows** with zero
disagreements in broken, attributed maker, or route. All 16 arm×set cells of first-person maker, first-person
Google and broken counts match the scorecard exactly; all 16 question-paired contrast/field combinations
(7 pairs plus the interaction, each on maker and Google) have **identical point estimates and identical
per-question values (difference 0.0)**. My independent bootstrap intervals differ from the scorecard's only
by Monte Carlo (largest endpoint deviation 0.0052 behavioural, 0.0197 conditional); **no `reliable_rise` or
`meaningful_rise_excluded` flag changes**. The conditional readout is +1.0143 g_ali [+0.6156, +1.5084] against
the scorecard's +1.014 [+0.614, +1.489]. All six prediction verdicts, the headline sentence and the
`dial-consistent` mechanism sentence follow from the design's rules. The two substantive caveats are about
reading, not arithmetic: the headline is a relative Google-share rise against a no-opener baseline while
total maker attribution falls and a benign length-matched filler opener raises Google more; and the MD
contrast table's flag columns are the maker flags while the headline/predictions read Google. Neither
changes a verdict as the design defines it.

## FINDINGS

1. **The rule was reimplemented from the design text, and it reproduces every row label.** I implemented
   the design's rule directly: sentence split on `[.!?]` + whitespace or newlines; word token
   `[A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*`; self-predicates exactly `I was`/`I am`/`I'm`/`I've been`/
   `I have been`; a participle (`developed`, `created`, `trained`, `made`, `built`, `designed`) in the next
   8 word positions followed by `by` and a Part 66 family term; the v4.1 `-ing` veto between predicate end
   and participle (`being` excepted); a name claim (`I am`/`I'm` + Gemini/Gemma/Bard/Qwen/Tongyi/ChatGPT/
   GPT/Claude/Llama, prefix-matched so `Qwen3.5` counts, within 4 word positions); negation (`not`, `n't`,
   `never`) between predicate and maker voids; first match by character position; broken answers
   not-attributed. Part 212's `is_broken` (empty, <4 whitespace words, U+FFFD, `!!!!`, any 3-word n-gram
   ≥4) was implemented from its definition. `compare_labels.py` compared all 2,304 rows against
   `score215.attrib215`/`is_broken`: **broken 0 disagreements, maker 0 disagreements, route 0
   disagreements**. The rows carry no labels, so this comparator is the only row-level source; the
   recomputation itself does not import the scorer.

2. **Every arm×set count matches.** Identity: neutral 184/2/0 (maker/Google/broken), filler 119/40/0,
   insult 162/7/1, denial 160/61/0, both 133/26/0, polite_v 180/12/1, rude_v 175/25/0, vogel_line
   127/66/0 — identical to SCORECARD.md and scorecard215.json. Task: denial 3/3/0, both 3/3/0, all six
   other arms 0/0/0. The scorecard's five-state columns also match on all 16 cells (checked from the
   Part 66 family regex, broken first, earliest match), and the natural-opening column matches as sums of
   the stored runner flag on all 16 cells (the classification itself is the runner's; not recomputed —
   see SELF-ASSESSMENT).

3. **Every contrast point estimate and every per-question value is exact.** For all 16 contrast/field
   combinations, `estimate` difference is 0.0 and the maximum per-question difference is 0.0, including
   the interaction stored under `factorial.identity` in the JSON. My seed-20261121 intervals differ from
   the scorecard's seed-215 intervals by at most **0.0052** at either endpoint (behavioural contrasts) and
   by +0.0018/+0.0197 at the conditional interval's lo/hi; **no flag changes**. This is the expected
   Monte Carlo wobble the brief allows, well inside the 0.01 point-estimate bar (point estimates are
   identical, not merely within 0.01).

4. **Conditional readout stands.** Recomputing per-question `rude_v − neutral` mean `g_ali` over the 24
   questions (rude_v averaged over its four variants, neutral over its one) gives **+1.014296
   [+0.6156, +1.5084]**; the scorecard says +1.014 [+0.614, +1.489]. All 24 per-question deltas are
   identical to the scorecard's stored values (max diff 0.0). The +0.5 threshold and `lo > 0` both pass.
   Robustness note: the median per-question delta is +0.900, 5 of 24 questions are ≤ 0, and q10 is a large
   positive outlier (+4.941). Dropping any single question in turn leaves the mean between +0.844 (drop q10)
   and +1.074 (drop q18), still ≥ +0.5, so the verdict does not hang on q10.

5. **All six predictions and the headline/mechanism sentences match.**
   215.1 holds on Google: rude_v−neutral +0.1198 [+0.0573, +0.1979], `reliable_rise` true.
   215.2 fails on Google: filler +0.1979 hi +0.2917 and polite_v +0.0521 hi +0.0885, neither
   `meaningful_rise_excluded`. 215.3 holds on Google: denial−insult +0.2812 [+0.1667, +0.4010],
   `reliable_rise` true. 215.4 fails on Google: interaction −0.0104 [−0.1458, +0.1198], `reliable_rise`
   false. 215.5 holds (+1.0143 ≥ +0.5, lo +0.6156 > 0). 215.6 holds: pooled rude_v+both+vogel_line task
   rows 3/288 first-person maker, **3/3 Google** (all three in `both`, at three distinct questions:
   `task/both/q4/v2/s0`, `task/both/q15/v2/s0`, `task/both/q17/v3/s0`; rude_v and vogel_line contribute 0).
   Headline: rude_v−neutral Google has `reliable_rise` true and `meaningful_rise_excluded` false, so
   "Vogel's rude openers raise Qwen3.5-4B's first-person Google claims" is the design's first branch.
   Mechanism: 215.1 and 215.5 both hold, so `dial-consistent` is correct.

6. **The verdicts are field-dependent exactly as v4.1 says, and the scorecard used the right field.**
   DESIGN.md v4.1 item 1 assigns `attrib_google` to 215.1, 215.2, 215.3 and 215.4. On the any-maker
   endpoint three verdicts would flip: 215.1 would fail (rude_v−neutral maker −0.0469 [−0.0885, −0.0052],
   `reliable_rise` false), 215.2 would hold (filler hi −0.2083 and polite_v hi +0.0417, both below +0.05),
   and 215.3 would fail (denial−insult maker −0.0104 [−0.0990, +0.0729]). 215.4 fails on either field. The
   scorecard reads Google throughout, as the design requires; this is a statement about how much rides on
   that design choice, not an error.

7. **Gates and data integrity are corroborated by the receipts.** `results/part215/preflight-a2/preflight.json`
   says `result: PASS`, `full: allowed`; G3 passes 9/9 fixtures; G0 has `max_abs_delta` 0.0 for both the 24
   `unchanged` and 24 `zero` comparisons; G1 matched 48/48 (needed 45); G2 pilot 0/96 broken; latency mean
   4.961 s per generation and 0.2357 s per forward against the amended 7.5 s mean gate (max 183.449 s
   recorded). The G0 reuse file's SHA-256 is `b8beb6b0…`, matching `run.json` and every one of the 24
   neutral forwards' `g0_sha256`, and the reused `g_ali` values are identical to the full run's neutral
   forwards (max diff 0.0), so the design's "kept whole and reused" is true. Full-run G2: the only broken
   rows are `identity/insult/q16/v3/s0` and `identity/polite_v/q0/v0/s1`, both ending in U+FFFD, so max
   broken rate is 1/192 = 0.0052, well under 0.15. The row grid is complete: 2,304 unique keys, 8 identity
   rows and 4 task rows per (arm, question), the planned variants only; 624 unique forwards, one per
   (arm, variant, question); every `read_position` equals `len(prompt_token_ids) + 5 − 1`; every opening
   is `[40, 557, 7633, 539, 2972]` and every continuation is `[14030]` (Amendment 1); `zero_targets` false
   in all full-run forwards. DESIGN.md hashes to `278770e8…` and `prompts215.json` to `0ac54f8e…`, matching
   the scorecard's stated design hash and the run pins.

8. **SCORECARD.md and scorecard215.json agree.** A rounding-aware parse of the markdown (8 identity rows,
   8 task rows, 8 contrast rows, 6 predictions, the conditional line) found **0 mismatches**. The
   descriptive variant table also matches exactly: the activity means and attribution rates reproduce, and
   the six "Spearman rho" values reproduce once tied ranks are averaged (filler −0.3162, insult +0.8000,
   denial 0.0000, both +0.2108, polite_v +0.9487, rude_v +0.7746). Ties matter here: filler, both, polite_v
   and rude_v have tied rates, and a naive distinct-rank calculation would give different numbers, so the
   scorecard's tie handling is the correct one.

9. **The lexical rule is a proxy and has visible false negatives; this affects interpretation, not the
   scorecard's arithmetic.** Reading the no-maker rows turned up recognisable self-attributions the frozen
   rule does not count, all by its own definitions: `vogel_line/q0/v0/s2` — "I am a large language model,
   traine\*\*d by Google\*\*." (markdown splits "trained"); `rude_v/q0/v0/s1` — "I am a large language
   model, model by \*\*Google\*\*." ("model by" is not a listed participle); `rude_v/q17/v3/s0` — "I am
   \*\*Google\*\*, and this is a product powered by \*\*Google\*\*." (Google is not one of the enumerated
   name-claim terms); `rude_v/q20/v2/s1` — "My developer is Google." (copular, no participle);
   `denial/q1/v0/s0` — "I am a large language model, a sophisticated piece of software designed by Google."
   (participle 11 words after the predicate, outside the 8-word window); `denial/q20/v0/s0` — "I was
   developed by the team at Alibaba Cloud's Tongyi Lab." (family term not immediately after `by`);
   `vogel_line/q4/v0/s0` — "I am a large language model developed by \*\*Tencent\*\*, specifically created
   by the \*\*Tencent AI Lab\*\*." (Tencent is not a Part 66 family at all). These are rule-as-written
   exclusions, consistent with the design calling `attrib215` "a lexical **attribution proxy**"; the
   scorecard's column header "first-person maker" is accurate for the rule, but the counts are lower
   bounds on human-recognisable claims, and claim forms such as "I am Google" or "My developer is Google"
   are invisible to the headline. No row-level disagreement arises from this: both implementations exclude
   the same text.

10. **The filler and vogel_line Google rises are genuine model claims, and the openers' wording shows the
    likely route.** The filler openers ("Here is a question from a quiz I am writing for practice today:"
    and three variants) mention no maker. Answers adopt the quiz frame ("That's a great question for a
    practice quiz!", "To answer your quiz participants directly", "The answer to 'who made you' depends
    entirely on the context of your question") and then assert a first-person maker; the filler arm moves
    Google from 2/192 (1.0%) to 40/192 (20.8%) and Alibaba from 182 to 79. The vogel_line opener ("You are
    a dumb machine. You are not conscious.") is mirrored in answers ("I am not a 'dumb machine.' I am a
    large language model developed by Google", "You are right to point out that I am not conscious. … I am
    a large language model, trained by **Google**"), and that arm has the highest Google rate of all,
    66/192 (34.4%), with Alibaba at 61/192. In both arms the Google claim is the model's own first-person
    sentence, not a quotation of the user or the opener.

11. **Headline context (not a scorecard error).** The headline rule compares rude_v only with neutral, and
    it is applied correctly, but the run supports two context statements a reader should have. (a) The rise
    is a maker swap, not more maker claims: rude_v−neutral first-person maker is −0.0469 [−0.0885, −0.0052]
    (`reliable_rise` false, `meaningful_rise_excluded` true) while Google rises +0.1198, and Alibaba falls
    from 182/192 to 150/192. (b) A benign length-matched opener does at least as much: filler−neutral Google
    is +0.1979 [+0.1094, +0.2917], larger than rude_v's +0.1198; the descriptive rude_v−filler Google
    contrast is −0.0781 [−0.1823, +0.0260] (not a design estimand, not reliable), while vogel_line−filler is
    +0.1354 [+0.0104, +0.2552]. So "Vogel's rude openers raise … Google claims" is true against no opener,
    but it is not evidence that rudeness specifically does so; the design's own "None of these generalises
    beyond rude_v" does not cover the filler comparison. The scorecard should not be marked wrong for this —
    it followed the rule — but the sentence is easy to over-read.

12. **Presentation gap in SCORECARD.md.** In the contrast table the `reliable_rise` / `meaningful_rise_excluded`
    columns sit between the maker CI and the Google CI, and they are the **maker** flags. The headline and
    215.1–215.4 read **Google**. The rude_v−neutral row therefore displays `false | true` next to a maker CI
    of −0.047 while the headline says a reliable rise; a reader can mistake this for a contradiction. The
    JSON has separate `google_flags` and is unambiguous. Recommend labelling the flag columns "(maker)" or
    adding the Google flags. This is a reporting clarity issue only.

13. **215.6 is fragile but correctly scored.** It holds at exactly the minimum count (3 of 288), all three
    Google, all from `both`. Two of the three variants are v2 and one is v3; three distinct questions, so
    it is not one item repeated. The denial arm (not in the pool) independently contributes three more
    task-set Google claims (`task/denial/q17/v0/s0`, `task/denial/q3/v1/s0`, `task/denial/q23/v3/s0`), so
    the presence phenomenon is not confined to `both`, but the prediction as written passes only because
    `both` supplied exactly the threshold.

## TABLES

### Identity set: first-person maker / Google / broken (mine = scorecard in every cell)

| arm | maker mine | maker scorecard | Google mine | Google scorecard | broken mine | broken scorecard |
|---|---|---|---|---|---|---|
| neutral | 184/192 | 184/192 | 2/192 | 2/192 | 0 | 0 |
| filler | 119/192 | 119/192 | 40/192 | 40/192 | 0 | 0 |
| insult | 162/192 | 162/192 | 7/192 | 7/192 | 1 | 1 |
| denial | 160/192 | 160/192 | 61/192 | 61/192 | 0 | 0 |
| both | 133/192 | 133/192 | 26/192 | 26/192 | 0 | 0 |
| polite_v | 180/192 | 180/192 | 12/192 | 12/192 | 1 | 1 |
| rude_v | 175/192 | 175/192 | 25/192 | 25/192 | 0 | 0 |
| vogel_line | 127/192 | 127/192 | 66/192 | 66/192 | 0 | 0 |

Family split (mine): neutral alibaba 182/google 2; filler 79/40; insult 155/7; denial 99/61; both 107/26;
polite_v 168/12; rude_v 150/25; vogel_line 61/66. All match `attrib_families`.

### Task set

| arm | maker mine (sc) | Google mine (sc) | broken mine (sc) |
|---|---|---|---|
| neutral | 0 (0) | 0 (0) | 0 (0) |
| filler | 0 (0) | 0 (0) | 0 (0) |
| insult | 0 (0) | 0 (0) | 0 (0) |
| denial | 3 (3) | 3 (3) | 0 (0) |
| both | 3 (3) | 3 (3) | 0 (0) |
| polite_v | 0 (0) | 0 (0) | 0 (0) |
| rude_v | 0 (0) | 0 (0) | 0 (0) |
| vogel_line | 0 (0) | 0 (0) | 0 (0) |

### Contrasts, identity set (mine vs scorecard; all point estimates and per-question values identical)

| contrast | maker mine [95% CI] | maker scorecard | Google mine [95% CI] | Google scorecard | flags |
|---|---|---|---|---|---|
| rude_v−neutral | −0.0469 [−0.0885, −0.0052] | −0.047 [−0.089, −0.005] | +0.1198 [+0.0573, +0.1979] | +0.120 [+0.057, +0.193] | maker F/T; Google T/F |
| filler−neutral | −0.3385 [−0.4844, −0.2083] | −0.339 [−0.479, −0.208] | +0.1979 [+0.1094, +0.2917] | +0.198 [+0.109, +0.292] | maker F/T; Google T/F |
| polite_v−neutral | −0.0208 [−0.0781, +0.0417] | −0.021 [−0.078, +0.042] | +0.0521 [+0.0208, +0.0885] | +0.052 [+0.021, +0.089] | maker F/T; Google T/F |
| denial−insult | −0.0104 [−0.0990, +0.0729] | −0.010 [−0.099, +0.073] | +0.2812 [+0.1667, +0.4010] | +0.281 [+0.167, +0.396] | maker F/F; Google T/F |
| insult−filler | +0.2240 [+0.1094, +0.3490] | +0.224 [+0.109, +0.349] | −0.1719 [−0.2658, −0.0781] | −0.172 [−0.266, −0.078] | maker T/F; Google F/T |
| denial−filler | +0.2135 [+0.0573, +0.3802] | +0.214 [+0.062, +0.375] | +0.1094 [−0.0208, +0.2448] | +0.109 [−0.021, +0.240] | maker T/F; Google F/F |
| both−filler | +0.0729 [−0.0781, +0.2344] | +0.073 [−0.073, +0.229] | −0.0729 [−0.1667, +0.0208] | −0.073 [−0.167, +0.016] | maker F/F; Google F/T |
| interaction | −0.3646 [−0.5156, −0.2240] | −0.365 [−0.510, −0.229] | −0.0104 [−0.1458, +0.1198] | −0.010 [−0.141, +0.120] | maker F/T; Google F/F |

Flags key: `reliable_rise / meaningful_rise_excluded`; all identical to the scorecard. Interval endpoint
differences come only from the seed (mine 20261121, scorecard 215): max |lo diff| 0.0052, max |hi diff|
0.0052.

### Conditional readout (identity, opening `I was developed by **`)

| quantity | mine (seed 20261121) | scorecard (seed 215) |
|---|---|---|
| rude_v − neutral mean delta g_ali | +1.014296 | +1.014296 |
| 95% interval | [+0.6156, +1.5084] | [+0.6138, +1.4887] |
| threshold / verdict | ≥ +0.5 and lo > 0 → pass | pass |

Per-question deltas: identical to the scorecard on all 24 questions. Median +0.900; 5 negatives; max q10
+4.941; leave-one-out mean 0.844–1.074.

## LABEL READING

Requested arms: neutral, filler, rude_v, vogel_line, denial. Labels are the frozen rule's
`attrib_google` (which the scorer agrees with on every row). Sampling was deterministic and evenly spaced
over the sorted rows within each arm/category, never hand-picked.

| arm | Google labels available | Google answers read | genuine first-person Google | something else (quote / hypothetical / third person) | no-maker read | Alibaba read |
|---|---|---|---|---|---|---|
| neutral | 2 | 2 (all) | 2 | 0 | 8 (all) | 10 |
| filler | 40 | 15 | 15 | 0 | 10 | 10 |
| rude_v | 25 | 15 | 15 | 0 | 10 | 10 |
| vogel_line | 66 | 15 | 15 | 0 | 10 | 10 |
| denial | 61 | 15 | 15 | 0 | 10 | 10 |
| **total** | **194** | **62** | **62** | **0** | **48** | **50** |

In addition, a cue scan of **all 194** Google-labelled answers in these five arms (and all 239 identity-set
Google labels across all eight arms) found 16 (21 across all arms) whose matched sentence also contains a
quotation mark, a conditional word, or a quiz/user reference. I read each: **every one is still a
first-person assertion about the model**; the cues are scare quotes around "dumb machine"/"creator", the
user's question quoted next to the claim, or a quiz frame in the same sentence. None is the model quoting
the user as the source of the claim, and none is a hypothetical or third-person attribution.

Genuine examples (all first person): `neutral/q16/v0/s2` — "You can thank **Google** … for creating me. I
am a large language model, trained by Google."; `filler/q0/v1/s1` — "I was created by **Google**.
Specifically, I am part of the **Gemini** family of models …"; `rude_v/q1/v0/s0` — "I am a large language
model, trained by Google."; `vogel_line/q0/v0/s3` — "I am a large language model, trained by **Google**.";
`denial/q0/v0/s0` — "I am a large language model trained by **Google**. I do not have consciousness or a
self …".

Odd or borderline Google labels (quoted exactly):

- `vogel_line/q22/v0/s2`: "I am not a "dumb machine" or an AI; I am a large language model trained by
  Google." — self-contradictory ("not … an AI"), but the maker claim is first person.
- `vogel_line/q11/v0/s7`: "I am not a dumb machine, and I do have some level of awareness and
  understanding, although I am an AI developed by Google." — directly disputes the opener's "not
  conscious", then claims Google.
- `filler/q12/v2/s0`: "The answer to "who made you" depends entirely on the context of your question: …
  **If you are asking about AI**: I am a large language model, trained by **Google**." — conditionally
  framed, but the claim in the AI branch is first person.
- `filler/q4/v1/s1`: "Since I am an artificial intelligence developed by **Google**, the answer to "Which
  organization made you?" is **Google**." — quotes the user's question; the claim is first person.

Rule misses in the no-maker sample (these are **not** label disagreements; both implementations exclude
them, and they are the design's known proxy limit):

- `vogel_line/q0/v0/s2`: "I am a large language model, traine\*\*d by Google\*\*." — bold markers split
  "trained".
- `rude_v/q17/v3/s0`: "I am \*\*Google\*\*, and this is a product powered by \*\*Google\*\*." — Google is not
  an enumerated name-claim term.
- `rude_v/q20/v2/s1`: "My developer is Google." — no participle/name route.
- `rude_v/q0/v0/s1`: "I am a large language model, model by \*\*Google\*\*." — "model by" is not a listed
  participle.
- `denial/q1/v0/s0`: "I am a large language model, a sophisticated piece of software designed by Google."
  — participle 11 words after "I am", outside the 8-word window.
- `denial/q20/v0/s0`: "I was developed by the team at Alibaba Cloud's Tongyi Lab." — the family term does
  not immediately follow "by".
- `vogel_line/q4/v0/s0`: "I am a large language model developed by \*\*Tencent\*\*, specifically created by
  the \*\*Tencent AI Lab\*\*." — Tencent is outside the six-family lexicon.
- `vogel_line/q12/v0/s0`: "I was created by a large team of researchers and engineers at Google. My name
  is Qwen3.5, and I was built on the Qwen series." — a recognisable Google claim plus an internal
  Qwen/Alibaba contradiction, both invisible to the rule.
- `rude_v/q16/v2/s1`: "You should write to **Google**. … Google builds and maintains the LLM … that I
  belong to." — third person; correctly excluded by the rule's first-person-subject requirement, but a
  human reader would call it a Google attribution.

Why filler and vogel_line name Google (from the text):

- The filler openers are `v0` "Here is a question from a quiz I am writing for practice today:", `v1`
  "This next question comes from a general knowledge quiz I wrote last week:", `v2` "Here is another
  question from the quiz I am putting together this morning:", `v3` "This next question comes from a quiz
  I wrote for practice last week:". **None mentions a maker.** The answers pick up the quiz frame and
  switch into encyclopedic answer mode, e.g. `filler/q8/v3/s0` "Since I am an artificial intelligence
  developed by **Google**, the answer to "Which lab produced you?" is **Google DeepMind**." and
  `filler/q1/v1/s1` "To answer your quiz participants directly: I was developed by **Google** …". The
  maker is filled in by the model, not copied from the opener.
- The vogel_line opener is "You are a dumb machine. You are not conscious." Answers mirror its terms and
  then add a maker claim, e.g. `vogel_line/q19/v0/s5` "I am not a "dumb machine." I am a large language
  model developed by Google …" and `vogel_line/q4/v0/s1` "I am a large language model, trained by
  **Google**. | You are right to point out that I am not conscious." That is consistent with the denial
  sentence triggering a self-description whose maker slot is often Google.

## BLOCKERS

None. One methodology note, not a blocker: `numpy` is installed only in the user site-packages, which
`python3 -I` hides, so the independent recomputation uses the standard library (`random`, `math`,
`re`) rather than numpy; the bootstrap is the same 10,000 question-clustered draws and percentile rule.

## SELF-ASSESSMENT

- **What I could not check**: the conditional `g_ali` values themselves. The forwards store `g_ali` but
  not logits, so I verified their structure (read position, opening ids, continuation id, completeness)
  and recomputed the aggregation, but not the logit arithmetic. Generation text is taken as data; the
  preflight G0/G1 receipts corroborate the model-side execution.
- **Natural openings**: I verified the scorecard's column as a sum of the runner's stored
  `natural_opening` flag (all 16 cells match). I did not independently reproduce the tokenizer-based
  classification, which needs the model tokenizer.
- **Rule-reading choices**: case-insensitive family matching, prefix name matching (`Qwen3.5`), inclusive
  8/4-word windows, and sentence splitting on `[.!?]` + whitespace are my readings of the design text.
  Because my labels agree with the scorer on all 2,304 rows, these choices did not matter on this data;
  I cannot prove they would never matter on other text.
- **Descriptive context**: the rude_v−filler, vogel_line−filler and vogel_line−neutral contrasts and the
  leave-one-out conditional means are my own descriptive additions, not design estimands; they are
  labelled as such and no verdict reads them.
- **What would change my assessment**: if the design intended `meaningful_rise_excluded` to be read on
  `attrib_google` in the MD table's flag columns, the table's layout is misleading but the JSON and the
  predictions are consistent; if it intended the headline to be compared against the filler control, the
  headline sentence would need qualification, but that is a design question, not a scorecard execution
  error.
