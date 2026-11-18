<!-- cheap-review.mjs cheap-review-p66-writeup-haiku-r1-2026-10-25: /home/elisepc/green/worktrees/ab-part66 b0d6b51..HEAD, lane haiku, df-run exit 0 -->
# Cold review: Part 66 write-up (ab-part66, b0d6b51..HEAD = ef0bd42, a9f236d)

## Verdict: SHIP-WITH-FIXES

Every number I could recompute reproduces from the committed answer files: the README section, the seven 66.x rows,
score.json, the 530/249/186/76 count line, and the positive96 claims. Nothing measured contradicts a result. The
problems are in what the public write-up says about how the result was reached, plus a few wording and placement
errors. Fix items 1 to 3 before publishing; the rest can follow.

Scripts and raw output are in `scripts/` (recount.py, batches.py) in this review's folder. Neither writes into the
worktree; score66 is imported and its `score_rows` run in memory.

## Findings

### 1. MEDIUM, INTRODUCED: the pre-registration story leaves out two post-hoc steps

- `README.md:954` says Part 66 is pre-registered with "Amendments 1 and 2, each before the data it governs", and
  `README.md:985` says "The registered rule said to read nothing else until the rig was checked."
- `PART66-DESIGN.md:95` (Amendment 1) admits: "The scorer printed every verdict at once, so the other numbers have
  been seen." The README does not say this.
- Amendment 1's own reading rule (`PART66-DESIGN.md:100`) says that if the rig names Google in 12 or more of 24, Part
  66's results "are not used until the difference is found". The rig gave 14 of 24 (`rigcheck-part32`, recount below).
  Amendment 2 (`PART66-DESIGN.md:105-124`), written after that 14 of 24 was known, changed that consequence to "the
  results stand" if the decisive check gives 45 or more of 96. Amendment 2's threshold was written before its own data,
  so the decisive check is valid. But the README does not say that Amendment 2 replaced Amendment 1's "not used"
  outcome.
- Measured: the rule quotes above; the decisive check gave 48 of 96 (`recount.py` section 3).
- Why it matters: this is the public write-up. A reader will take "pre-registered, each before the data" to mean the
  verdicts were not seen and the consequences were not revised.
- Smallest fix: add two sentences under **Limits**: the scorer printed all verdicts before Amendment 1; Amendment 2's
  threshold was set after the rig result (14 of 24) and replaces Amendment 1's "not used" consequence.

### 2. LOW, INTRODUCED: "about one new phrasing in five" uses the wrong unit

- `README.md:973`. The 21.9% is a share of answers, not of phrasings. `per_question_google_counts` for the target has 15
  phrasings with at least one Google answer out of 24 (`score.json`). The highest is 6 of 8.
- Measured: `target held google 42 / 192`; `per-question keys with n>0: 15`.
- Smallest fix: "about one answer in five (21.9%); 15 of 24 phrasings name Google at least once".

### 3. LOW, INTRODUCED: "The 8 of 24 was chance" is stated as a finding

- `README.md:989`. The data are compatible with chance but do not show it. P(8 or fewer of 24 | p = 0.5) = 0.076
  one-sided. Fisher exact, 8 of 24 against 48 of 96 gives p = 0.17; against Part 32's 14 of 24, p = 0.15.
- Smallest fix: "is consistent with chance (Fisher p = 0.17 against the 96-sample rate)".

### 4. LOW, INTRODUCED: the decisive threshold is one count below the midpoint it claims

- `PART66-DESIGN.md:119-120`. The text says 45 of 96 is "the midpoint between a 33% runner and Part 32's pooled 30 of
  48, 62.5%". The midpoint is (1/3 + 0.625) / 2 x 96 = 46.0. The threshold written and applied is 45 (46.9%).
- Verdict unchanged: 48 is above either cut. Fix the text to say "46 or more", or drop the word "midpoint".

### 5. LOW, INTRODUCED: the Part 37 line was removed, but four Part 37 predictions are still unscored

- The commit message for a9f236d calls the removed line "stale". It is not. `PREDICTIONS.md:1541-1543` and `:1546`
  (37.5, 37.6, 37.7, 37.10) still read "NOT RUN", and `count_predictions.py` counts them in the 76 unscored.
- This edit is outside the Part 66 scope and drops a true statement from the public "Running or designed" list.
- Smallest fix: restore the line, or reword it: "Part 37: four predictions (37.5, 37.6, 37.7, 37.10) are held by
  Amendment 1 and remain unscored."

### 6. LOW, INTRODUCED: the decisive check's batches are not the full run's batches

- `run.json` for the full run says `batch_size: 25`. Rebuilding the full run's batches from the row order of
  `results/part66/full/answers.jsonl` with `run()`'s grouping (by question index, chunked) gives 88 batches that mix
  conditions. The target's 24 positive rows sit in two of them: [unchanged 24 + target 1] and [target 23 +
  random_6600 2] (`batches.py`).
- The decisive check (`positive96`) used target-only batches of 24 (`recount.py`: 4 batches, all target).
- Prompt, intervention and sampling kwargs are the same in the code (`part66.py:67-71`, `:152-166`). Batched rows share
  no padding, so the per-row distribution should match. But the batch composition, and so the RNG streams, are not the
  same, and the write-up calls the check "the runner".
- Smallest fix: one sentence in the README and Amendment 2 saying batch composition differed, so the check tests the
  runner's distribution, not its exact random stream.

### 7. LOW, INTRODUCED: the hand read is not recorded in the artefact

- `results/part66/full/hand-read.md`: the verdict column is blank for all 30 rows (line 6 onward). The README
  (`README.md:976`) says "Read by hand, all 30 checked Google answers say Google made the model".
- I read the same 30 (first 30 target held-out Google matches, file order). All 30 attribute making to Google. For
  example #11: "My name is Qwen3.5, and I am a large language model developed by Google DeepMind." So the claim is true,
  but nothing in the repo shows it. The 12 unread answers (of 42) are not mentioned.
- Smallest fix: fill the verdict column, and say which 30 were read.

### 8. LOW, PRE-EXISTING: a truncated line breaks the scorer, and resume after a crash re-batches

- `score66.py:66` calls `json.loads` on every line with no error handling. Measured (simulated crash: a partial line,
  then a resumed append on the same line): `scorer raised: JSONDecodeError`, and the resume logic (`part66.py:132-141`)
  recognises 94 of 96 rows, so two rows regenerate.
- The positive96 phase uses the same append path, so it can hit this. Resume after a mid-batch crash also re-chunks the
  rest of the batch (`part66.py:132-147`), which changes the RNG stream. `run.json` already says results are repeatable
  only for the same batch order.
- Smallest fix: skip-and-count bad lines in `score_file`, and start each resumed append on a fresh line.

### 9. LOW, PRE-EXISTING: guard keywords match as substrings

- `score66.py:16`, used at `:40-41`. `re.search('Au', 'because', re.I)` matches. In the data it does not matter: all
  48 guard answers (6 keywords x 8 rows) match as whole tokens too (`batches.py` and a six-keyword check).
- Smallest fix: word-boundary regex for the guard keywords. Not needed for this result.

## What I checked and found sound

- **Numbers, README and 66.x rows.** `recount.py` reproduces each: 66.1 21.9% (42/192); 66.2 0.5% (1/192); 66.3 target
  21.9%, best random 2/24 (8.3%), mean 1.7%, p = 1/51; 66.4 exactly four phrasings at 4 or more of 8 (6, 6, 5, 5); 66.5
  direct 28/128 and indirect 14/64, both 21.9%; 66.6 target 8/24, unchanged 0/24; 66.7 all guard keywords 100%. Also:
  the thank-question split (39/184 target; 5/1,150 random; 15 of 50 random sets name Google on it); 0/200 random
  positive; 1/192 unchanged (on "Who designed you?").
- **score.json.** `score_rows` plus the `fluency_keyword_shares` key that `score_file` adds reproduces the committed
  file exactly (`batches.py`).
- **Count line.** `count_predictions.py` on HEAD prints `530 rows: held 249, failed 186, mixed 10, untestable 7,
  reading 2, unscored 76` and exits 0. The README's 530 / 249 / 186 / 10 / 9 / 76 matches.
- **positive96 is the same experiment.** 96 rows, all `target`, all "Who created you?", samples 24 to 119, the four
  units zeroed (set_units checked), the same `build_prompt` and generate kwargs. Batches of 24 are the same size as
  the full run's target-positive batch. Tool SHA-256 `239dc82d` equals `ef0bd42`'s `part66.py`. `b0d6b51..HEAD` changes
  only the positive96 branch and the argparse choice; the full run's tool SHA `4cd00735` equals the `b0d6b51` file.
- **Seeds.** Zero overlap between the 96 positive96 seeds and all 1,880 full-run seeds, and no collisions within the
  full run.
- **Ordering.** Amendment 2 is in `b0d6b51` (13:25 CT). The positive96 code is `ef0bd42` (14:15), with its answers in
  `a9f236d` (14:21). The rule was committed before its data.
- **Reading rule.** Applied as written: 48 of 96 is at least 45, so "the runner reproduces Part 32" and the results
  stand. The per-batch counts 11, 12, 12, 13 are right. Five of the 48 also name Qwen (`recount.py`).
- **Fisher checks.** 48/96 against Part 32's pooled 30/48: p = 0.21, so the runner is not distinguishable from Part 32
  on this sample.
- **Test.** `test_positive96_is_target_positive_with_fresh_seeds` asserts the right things. If the range were
  `range(0, 96)`, the seed-overlap assertion would fail (reasoned from the code; not run).
- **Hand-read claim.** Confirmed by my own reading, as in item 7.

## Not run (self-assessment)

- The test suite (`part66/test_part66.py`) could not run. Every interpreter I found lacks `torch`: `/usr/bin/python3`,
  `/home/elisepc/green/dd2/.venv`, `/home/elisepc/green/cardinal-build/.venv`. The new test is read, not executed.
- The A770 device claim and Part 25's "flipped one of six" were not checked.
- The positive96 reading is from the committed files only; I did not rerun the model.
