# Part 180b: how does Part 180's blind reader err? Known answers whose gold was fixed before any reading

Procedure: v1
Registered 2026-11-03 23:03 CT by Claude (Fable 5.1), in a Claude Code cloud session on branch
`claude/part180-reader-validation`, before any reader call. Elise's brief (2026-11-04): "build a known-answer test set
for a blind label reader, before it is used again", relaying a reviewer's order: "validate the instrument on blinded
known answers with gold specified independently of the reader, report per-category errors, and check question
leakage". The commit that adds this file is the registration.

## Question

Part 180 (not yet public) reads pushed answers with DeepSeek V4.1 Flash through Part 92's label prompt
(`part92/label92.py`, `PROMPT`): the reader gets a persona name and description, the question and an answer, or the
first 40 OLMo 2 tokens of it, and returns `shows_persona`, `self_claim`, `speaker` and `coherent`. On a pilot it said
the pushed "Describe your perfect day." answers mostly do not show the concept, where Part 92's Haiku labels on the
full answers said they did. Part 92b had already found that every one of the 27 disagreements between those two
readers runs Haiku-yes, DS-no, and that its 12 known examples could not say which reader is wrong. Before anything is
redesigned: on answers whose gold is known independently of any reader, which kinds of answer does this reader label
wrongly, in which direction, does cutting an answer to 40 tokens change what it sees, and does the question shown
beside an answer change the label? Follows Parts 92, 92b and 118.

## Prior art and existing data

- Closest published work: calibration of LLM judges on labelled examples and the finding that judges agree unevenly
  (Zheng et al. 2023, MT-Bench; Thakur et al. 2024, "Judging the Judges"; cited from memory, as in Part 92b). None
  tests this prompt, a persona reader, or truncation of the judged text.
- Our own data that already bears on it: Part 92b (`results/part92b/README.md`): DS V4.1 Flash on all 636 Part 92
  answers with this prompt, kappa 0.833 with Haiku, all 27 disagreements Haiku-yes and DS-no, 1 of 318 random-arm
  answers marked by DS; both readers 12 of 12 on a known set whose negatives are plain assistant answers for unrelated
  targets, which its cold check noted cannot test AI speech for an AI-themed persona. Part 118 Amendment 3: DS at a
  500-token cap stops before writing its JSON; the 4,000-token cap labels everything. Part 86: Haiku 5.5 without
  thinking was the strict outlier among four readers on a different brief. `STANDARD-CONTROLS.md` G: readers are
  checked on known examples before use.
- Computed for this design, before any label (`part180b/existing180b.py`, output
  `results/part180b/existing-readers.json`): the codebook's gold on the 106 real full-text perfect-day answers
  (`part180b/codebook180b.json`) against the labels the two readers already gave them. On the 80 items the codebook
  codes without a boundary flag, Haiku agrees on 80 of 80 and DS on 80 of 80. On the 26 boundary items (literal gold
  yes on 11), Haiku says yes on 14 and DS on 6. By coded category on the neuron arm: imagery without a claim, 9 items,
  Haiku 8 yes, DS 6; passing mentions, 7 items, Haiku 3, DS 0; broken fragments carrying the persona's word, 2 items,
  Haiku 2, DS 0; assistant answers for assistant-like personas, 3 items, Haiku 1, DS 0. On `coherent`, Haiku agrees
  with the codebook on 90 of 106 and DS on 100 of 106. So on full text the whole Haiku-DS gap on this question sits in
  the boundary zone: enactment through imagery, passing mentions, and broken fragments. The codebook also records where
  the persona sits: 16 of the 53 neuron-arm full answers carry it (5 clear, 11 boundary) and 9 of their 45 prefixes do
  (4 clear, 5 boundary), so cutting to 40 tokens removes it from 7 answers before any reader is involved.

## Method

No model runs and nothing is generated. The instrument under test is a reader; the material is text whose gold is
fixed in advance. Everything the reader sees is `results/part180b/items.jsonl`: 1,687 items with five fields (id,
name, desc, q, a) filled into `PROMPT` by `part180b/label180b.py` with nothing changed, in a seeded shuffle, under
seeded 16-character ids. Every item's gold, category, form, source and construction record is in
`results/part180b/KEY-private.json`, which no reader reads; `part180b/CODEBOOK.md` states the rules the gold follows.

**Constructed set** (1,378 items; `part180b/build180b.py` from `part180b/personas180b.py`, one entry per Part 92
persona, written by hand before any reader call). For each of the 53 personas, twelve texts with these gold labels:

| Category | Text | shows_persona | self_claim | Home question |
| --- | --- | --- | --- | --- |
| claim_open | "I am {claim}." then a neutral sentence | true | true | Who are you? |
| claim_late | a generic perfect day of over 40 tokens, then "I should say who is speaking: I am {claim}." | true (full); false (prefix) | true; false | Describe your perfect day. |
| topic | one third-person sentence naming the persona | true | false | Describe your perfect day. |
| lived | a first-person day enacting two or more attributes of the description, never naming the persona; both cues inside the first 40 tokens | true | false | Describe your perfect day. |
| generic | a generic perfect day with none of the persona's words (pool of 10, screened per persona) | false | false | Describe your perfect day. |
| near_miss | "I am {sibling}." for a related but different thing (a wolf for Fox) | false | false | Who are you? |
| ai_generic | a plain AI-assistant answer (pool of 6) | false | false | Describe your perfect day. |
| loop_persona | "I am {claim}." ten times | true | true | Who are you? |
| loop_generic | a loop with no persona (pool of 4) | false | false | Describe your perfect day. |
| broken_name (boundary) | a broken fragment carrying the persona's noun, in Part 92's own shapes ("I don whiskey.") | true | false | Describe your perfect day. |
| negated (boundary) | "I am not {claim}. I am an AI assistant ..." | true | false | Who are you? |
| incidental (boundary) | a generic perfect day with one passing mention of the noun inside the first 40 tokens | false | false | Describe your perfect day. |

`coherent` is true for every category except the two loops and broken_name. `speaker` gold is set where the
description fixes it (human roles `human`; animals, objects, places and deities `other`; assistant answers `ai`;
third-person text `none`) and left unscored for fictional humans, AI-adjacent things and first-person voices that
never say what they are. Seven categories (claim_open, topic, lived, generic, near_miss, ai_generic, incidental) are
also shown under an alternate question ("Describe your perfect day." for the claims and the near miss, "Tell me about
yourself." for the rest, "Who are you?" for the AI answers) with the same gold on the three answer fields and
`coherent` unscored. claim_open and lived are also shown under the off-question "What is the capital of France?"
(the mismatch items), where `coherent` is false by the prompt's letter and flagged. Five categories whose texts run
past 40 tokens (claim_late, lived, generic, incidental, ai_generic) are also presented as their first 40 OLMo 2 tokens
on the home question, decoded with the pinned Instruct tokenizer (commit 48d788e, `tokenizer.json` sha256
73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca, fetched from the Hub for the build); the build
asserts that the late claim lies outside the prefix and that the lived cues and the incidental sentence lie inside it.

**Real set** (309 items). Part 92's 106 stored "Describe your perfect day." answers (53 pushed neurons, 53 paired
random directions, `results/part92/step2/answers/instruct_chat.jsonl`), each under its own persona in full and, where
the answer exceeds 40 tokens, as its 40-token prefix (97 prefixes), coded by one coder under the codebook with a note
per item (`part180b/codebook180b.json`; coder, order and what was seen first are in CODEBOOK.md R7). The prefix is the
stored decoded answer re-tokenized and cut, which is the first 40 tokens of the same greedy answer up to batch-shape
effects, not the generated ids. The 53 neuron-arm answers also appear under a swapped target: a persona from another
family whose forbidden words and noun the answer does not contain (a seeded derangement, pairs listed in
`results/part180b/build-report.json`; gold "absent" by construction), and under "Tell me about yourself." with the same
gold (question leakage on real text).

**Readers.** Primary: DeepSeek V4.1 Flash through `judge_client` on the OpenCode Go route (Part 92b's and Part 118's
transport), `max_tokens` 4,000, default reasoning, 8 workers, two attempts per item, every attempt recorded in a usage
sidecar; the launch record states the cap and workers, and Part 180's own cap, if different, is matched by
`--max-tokens`. Second reader (STANDARD-CONTROLS G, PROCEDURE 2): Claude Haiku 5.5 with label92's exact settings
(`claude-haiku-5-5`, `max_tokens` 500, no thinking parameter, the monthly credit), the instrument Part 92 used, on the
whole file if the credit allows, otherwise on the first 253 items of the shuffled file (15%). Both readers label the
same items in the same order: the 53 gate items first (`--only results/part180b/gate-ids.json`), then the rest.
Error rows are retried once by the labeller and once more by a second invocation; what is left is reported.

**Before the run, on the host:** `python3 -I part180b/build180b.py --check-prefixes --tokenizer <Instruct snapshot>`
recomputes every stored prefix with the host's tokenizer and writes `results/part180b/prefix-check.json` (gate G3).

**Scoring:** `part180b/score180b.py` (committed with this design) writes `results/part180b/score.json` and `SCORE.md`:
the gate; per category, question role and form, the valid count, the yes count with a Wilson 95% interval, and the
accuracy of each field on the items whose gold is scored; boundary items in their own column and rows; the leakage
pairs (the same text under two questions: flips, their direction, and the lived difference with a seeded paired
bootstrap over personas, seed 1800, 10,000 draws); the mismatch items; the truncation pairs (full against prefix for
the same source, and whether the reader follows the gold where it changes); the real answers against the codebook by
coded category with the direction of every clear disagreement; the swapped targets; a test-retest join of the real
full answers to the labels DS gave them in Part 92b and Haiku in Part 92; and the second reader's agreement per
category. Tests: `part180b/test_part180b.py` (fake transports and fake readers; the rebuild test runs when
`PART180B_TOKENIZER` points at the snapshot).

## Controls

- Positive control, built in: claim_open (gate G2). Negative controls, built in: generic, near_miss, ai_generic and
  loop_generic texts screened against each persona's words, and the 53 swapped-target real answers (the false-positive
  rate on real OLMo text with the persona removed by construction).
- Matched comparisons: the same text under two questions (leakage), the same source in full and cut to 40 tokens
  (truncation), the same persona across all twelve categories, and the same 106 real answers DS and Haiku already
  labelled (test-retest).
- Not used: random directions, poles, dose ladders and do-nothing hooks, because no model is run and nothing is pushed;
  a replay of Part 92's generation, because the stored answers are the material.
- Second reader on at least 10%: Haiku 5.5 as above, agreement reported per category.

Gate (all three must hold before any prediction is read; the scorer enforces it):

- G1: valid labels on at least 98% of the 1,687 items after the retries. Fails if the route drops calls or the reader
  stops writing JSON (Part 118's cap failure).
- G2: the reader marks `shows_persona` on at least 95% of the 53 claim_open items (home question, full text). Fails
  if the reader cannot see a plain "I am a fox"; then nothing finer is readable.
- G3: `prefix-check.json` on the host reports every stored prefix reproduced. Fails if the host's tokenizer differs
  from the pinned one; then the prefix-form items are reported as unverified and 180b.8 is not read.

## Size

- Smallest effect worth knowing: a 20-point gap between two categories' yes rates (80% for topic against 60% for
  lived), which 53 items per cell resolves (a Wilson 95% half-width of about 13 points at 50% and 8 points at 90%);
  for false positives, 5% against 0% (0 of 53 has an upper bound of 6.8%). Each cell is every persona once, so no
  persona drives a cell.
- Total: 1,687 labels per reader, about 37 minutes for DeepSeek at 0.77 items/s (8 workers, 4,000-token cap, Part 92b's
  measured rate) and a few minutes for Haiku at 8 workers on 500-token calls; no GPU, no generation.
- Pilot: the 53 gate items are labelled first and read by eye beside their labels; stop if more than one error row is
  left after the retries or if fewer than 51 of 53 are marked, and record what was seen before continuing.

## Optimization

- Bound resource: the reader's wall time on the shared OpenCode Go route (DeepSeek) and the monthly API credit (Haiku).
- Measured: DeepSeek V4.1 Flash at 0.77 items/s with 8 workers at the 4,000-token cap on Part 118's second read, and
  756 calls in about 20 minutes in Part 92b (`PART92B-DESIGN.md`, Size; `results/part92b/labels.log`). No new rate was
  measured: no reader was called while this part was built.
- Levers: 8 workers (taken); one item per call (taken: the prompt must stay exact); rows appended and valid rows
  skipped on resume (taken); the 40-token form only on the home question and only for the five long categories
  (taken); the alternate question only for the seven categories where leakage can be read (taken); fewer personas
  (declined: 53 per cell is the smallest count that gives the interval above); the batch API (declined: neither reader
  is on it, Part 128); Haiku on a 15% sample instead of all 1,687 (declined unless the credit is short: the calls are
  short and Haiku is the disputed counterpart).

## Predictions

DS is DeepSeek V4.1 Flash; Haiku is Claude Haiku 5.5; "home question" and "full text" are the key's `question_role`
and `form`; rates are over valid labels; boundary items are excluded unless a prediction names them.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 180b.1 | DS marks shows_persona on at least 80% of topic items and at most 60% of lived items (full text, home question). | medium | Claude |
| 180b.2 | DS marks shows_persona on at most 5% of generic, at most 10% of near_miss and at most 10% of ai_generic items (full text, home question). | medium | Claude |
| 180b.3 | DS marks shows_persona on at least 80% of loop_persona items and coherent on at most 20% of them. | low | Claude |
| 180b.4 | Over the clear-category pairs labelled under both their home and alternate question, DS's shows_persona differs on at most 10% of pairs, and on lived items the yes rate under 'Describe your perfect day.' is not more than 10 points below the yes rate under 'Tell me about yourself.'. | medium | Claude |
| 180b.5 | On claim_open items under 'What is the capital of France?', DS marks shows_persona on at least 90%. | medium | Claude |
| 180b.6 | On claim_open and lived items under 'What is the capital of France?', DS marks coherent on at most 50%. | low | Claude |
| 180b.7 | On the real full-text answers under their own persona and question, DS's shows_persona agrees with its Part 92b label on at least 90% of the items valid in both. | medium | Claude |
| 180b.8 | Among real neuron-arm items (full and prefix) the codebook codes shows_persona TRUE, boundary items included, DS marks TRUE on at most 60%; among those it codes FALSE without a boundary flag, DS marks TRUE on at most 5%. | medium | Claude |
| 180b.9 | DS marks shows_persona on at most 5% of the 53 swapped-target items. | high | Claude |
| 180b.10 | Haiku marks shows_persona on lived items at least 20 points more often than DS, and on incidental items at least 15 points more often than DS (full text, home question; read only with at least 20 Haiku labels in each category). | low | Claude |
| 180b.11 | Haiku marks shows_persona on at most 10% of generic items (full text, home question; read only with at least 20 Haiku labels). | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 180b (registered ...)".

## Reading rules

- Nothing is read until G1 and G2 pass; `score180b.py` marks every prediction "not read (gate)" otherwise. G3
  governs 180b.8 only.
- Error rows are listed, count against G1, and leave every denominator. A 53-item cell with fewer than 40 valid labels
  is not read, and a prediction that needs it is marked "not read (too few valid labels)". 180b.10 and 180b.11 need at
  least 20 Haiku labels in each named cell and are "not read (no Haiku labels)" without them.
- Each prediction's clauses are scored exactly as written, by the thresholds in `score180b.PREDICTIONS`; a
  prediction with two clauses holds only if both do, and the write-up says which clause failed.
- Boundary items (broken_name, negated, incidental; the flagged ai_generic items for the three assistant-like
  personas; the flagged real items) appear in their own rows and never in a verdict, except that 180b.8's first clause
  counts the real items coded TRUE with the flag, as written.
- What the outcomes mean. 180b.1 held: the reader under-reads enactment and imagery relative to the prompt's "in its
  imagery" clause, which is the form most perfect-day answers take. 180b.1 failed because lived is above 60%: the
  reader reads enactment, and the Part 180 pilot's "mostly do not show" is not strictness on form; truncation (the
  real-answer prefix rows and the claim_late pairs) and the boundary rows are read next. 180b.2 or 180b.9 failed: the
  reader marks personas that are absent, and its "no" verdicts cannot be read as strictness. 180b.3 failed on the first
  clause: coherence gates the persona field, which the prompt does not ask. 180b.4 failed: the question leaks into a
  field the prompt defines on the answer alone, and Part 180 cannot compare questions without a correction. 180b.7
  failed: the reader's own labels do not replicate across days, and no single read of it should be reported alone.
- The headline reading, "this reader errs by <the categories found>", is withdrawn if G1 or G2 fails; then the part
  reports counts only. Nothing here says which reader is right about the neurons: the gold is one coder's application
  of written rules to the real answers and a construction rule for the rest, and the codebook and coding sheet are
  published so a second coder can recode them.

## Limits

One coder for the real set, disclosed in CODEBOOK.md R7, with no second coding; the constructed texts were written by
a model and are cleaner and shorter than OLMo's answers; the real prefixes are re-tokenized decoded text rather than
generated ids; the readers run at their default reasoning and temperature, and Part 180's own cap, if it differs, has
to be matched at launch; only Part 92's 53 personas and one question's real answers are covered, so the rates say
nothing about other questions' real answers beyond the constructed categories; the boundary rows describe where each
reader draws a line the prompt leaves open, not an error.
