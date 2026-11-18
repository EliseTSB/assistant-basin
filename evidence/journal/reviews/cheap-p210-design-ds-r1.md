<!-- cheap-review.mjs cheap-review-p210-design-ds-r1-2026-11-12: /home/elisepc/green/worktrees/ab-part210 HEAD~1..HEAD, lane ds, df-run exit 0 -->
# Review: p210-design (HEAD 5c6b613a, "Part 210 design draft ... margins210.py and its 16 tests")

Verdict: **SHIP-WITH-FIXES** — the design is mostly sound and its arithmetic is right where it is checkable, but
three claims about the shape replicate are wrong or unbuildable as written, and one of them (the donor rule) would let
a builder ship a replicate that carries no information about the changed label, the one label 165.6 reads.

Reviewer: cold, did not write this code. Everything below was measured in this worktree with the real modules
(`part165/score165.py`, `part165/run165.py`, `part210/margins210.py`, `part202/score202.py`) on
`results/part165/answers.jsonl`; no model was loaded and no live service was called. Scripts are in `exp/`.

---

## Findings

### F1 — HIGH (INTRODUCED): the shape-floor donor rule as written can never change a `changed` label, so 165.6's stability reads 1.0 whatever the replicate says

**Where:** `PART210-DESIGN.md:48`, Controls, "The donor rule, and the no-push answers": *"The donor key is
therefore the drawn label vector plus `question_index`, and the donor is the first battery or replicate row in file
order with exactly that key."*

**Measured.** `exp/labels.py` and `exp/donor_dead.py` (in this domain) over `results/part165/answers.jsonl`: every
battery row is `changed` except one, and the no-push answer for every question is itself a self-description that
triage keeps. Label vectors over the 6,912 battery rows:

```
[(('changed',), 5843), (('broken', 'changed', 'loop'), 592), (('broken', 'changed', 'loop', 'triage'), 444),
 (('broken', 'changed', 'triage'), 16), (('changed', 'short'), 6), (('broken', 'changed', 'loop', 'short', 'triage'), 5),
 (('broken', 'changed', 'short', 'triage'), 4), (('broken', 'changed', 'loop', 'short'), 1), ((), 1)]
```

There is exactly one `changed=False` battery row in 6,912 (k16, layer 19, slot 2, question 7, whose first 12 words
match the no-push text). So the set of donor keys the rows can supply is `{(changed=True, qi) for qi in 0..7}` plus
that single `(changed=False, 7)`. `tools/shape_floor.simulate` only ever draws a row a destination vector that some
replicate pair *started* at and then *moved to* (`tools/shape_floor.py`, `change_to[start]`); a redraw that leaves a
row's labels alone keeps its own text. A row that flips `changed` True→False therefore draws `(changed=False, qi)`,
and for qi ≠ 7 there is no donor: the design says it "keeps the row's own text and is counted". Keeping the text means
the row stays `changed` in the rescore. **The `changed` label cannot flip in the floor, for any replicate.**

**Why it matters.** `changed` is the only label 165.6 reads. The design's own Size section says 165.6's verdict "does
not depend on the interval" and that the floor is "the uncertainty check". For 165.6 the floor is guaranteed to print
1.0 — exactly the failure Part 202's "Changes before registration" item (3) records and fixes ("every stability read
1.0 whatever the replicate said", `PART202-DESIGN.md:84-94`). The design repeats the defect it cites the fix for.

**Smallest fix.** Key the donor on the drawn vector *without* `changed` and have the donor's text set `changed` by
construction: a donor for `(broken, triage, loop, short, qi)` that is *not* changed is the no-push answer for `qi`
(which is in the plan and Gate 2 requires it unbroken); a donor that *is* changed is any battery answer for `qi` whose
first 12 words differ from that no-push answer. Both always exist. Alternatively, score the redraw with a stub scorer
that reads the drawn labels directly instead of donor texts (allowed: `shape_floor.simulate` accepts any
`rows -> {verdict: bool|str}` function), and keep the donor test only for the `broken`/`triage`/`loop` labels. Either
way, add a test that a redraw flipping `changed` on a row moves 165.6's estimate — today nothing can.

### F2 — HIGH (INTRODUCED): the replicate's 200 rows are drawn 2–3 per (arm, layer) cell, so the `changed` label is never measured and `triage`/`short` are mostly unmeasured

**Where:** `PART210-DESIGN.md:32`, Method, "Margins and the shape replicate": *"draws 200 battery rows by largest
remainder over the (arm, layer) cells and `random.Random(210)` within each cell"*.

**Measured** (`exp/replicate_power.py`). 200 rows over 72 cells (9 arms × 8 layers) is 2.78 rows per cell, i.e.
2 or 3:

```
cells 72  min/max per cell 96 96
changed mean P(pair agrees), 2 rows in a cell: 1.0
short   mean P(pair agrees), 2 rows in a cell: 0.996
triage  mean P(pair agrees), 2 rows in a cell: 0.892
broken  mean P(pair agrees), 2 rows in a cell: 0.769
loop    mean P(pair agrees), 2 rows in a cell: 0.772
cells x labels where a value exists but a 2-row draw misses it: 109.8 of 216
```

Part 202's replicate also samples 200 rows, but its cells are (arm, dose) — 6 cells, ~33 rows each. Part 210's cells
are (arm, layer) — 72 cells. The replicate pair for a vector is therefore built from 2–3 answers, and
`tools/shape_floor` treats a label value that no pair started with as **held and unmeasured** (its docstring: *"Zero
observed transitions are not evidence that a transition cannot happen"*). With 71 of 72 cells all-`changed`, the
replicate will contain almost no pair that starts at `changed=False`: the `changed` flip rate is unmeasured by
construction, not by chance. `triage` and `short` are positive in only 8 and a handful of cells and will be missed in
most of them.

**Why it matters.** The replicate is the design's only measurement of how labels move between shapes, and it is the
input to the floor. As drawn it measures `broken`/`loop` (the dense labels) and nothing else, while the design presents
it as the uncertainty check for *every* label including `changed`.

**Smallest fix.** Draw the replicate by label stratum as well as by cell: at least one row for every
(arm, layer, label-vector) combination that occurs, or a fixed quota of `broken=True` rows per cell (e.g. 100 broken
and 100 kept, drawn within cells). Say in Method how many pairs each label's start value gets. If the design wants to
keep 200 rows, say plainly that `changed`, `triage` and `short` will be unmeasured and that 165.6's floor is therefore
not informative — which, with F1, is the honest reading.

### F3 — MEDIUM (INTRODUCED): the shape floor's scorer is not specified, and the one Part 202 used cannot be reused here

**Where:** `PART210-DESIGN.md:47`, Controls, "Shape replicate": *"Each registered verdict and gate gets the
`tools/shape_floor.py` stability result from `part210/score210.py`'s `shape_floor` (`tools.shape_floor.simulate` with
`draws=1000`, `seed=0`)"*. Nothing says what the scorer function is.

**Checked (signature).** The Part 202 floor scorer is `part202/score202.py:221` `_score_without_bootstrap`, which
calls `score164.pair_bootstrap(..., reps=0)`. Part 165's bootstrap is
`score165._bootstrap_ratio(slot_rates, numerator_arm, denominator_arm, seed=165, reps=10_000)`
(`part165/score165.py:167`) and `_bootstrap_spearman(..., seed=165, reps=10_000)` (`:195`): `reps` is positional-or-
keyword, so a `point_only` wrapper would work, but the default runs 10,000 draws per ratio — three ratios plus two
Spearmans per rescore. A builder who copies Part 202's point-only trick gets a `TypeError`; a builder who
does not gets 5 × 10,000 draws × 1,000 floor draws. Both are "two ways that change a verdict" in the brief's sense:
the second is not a different verdict but a run that will not finish, and the design gives no budget for it.

Also unspecified: whether the floor's rows are the battery alone (Part 165's Gate 1 requires exactly 6,920 rows, so a
battery-only rescore fails Gate 1 in every redraw and every verdict reads "untestable" — Part 202's own recorded
defect 3) or the battery plus the 8 no-push rows. The design says the no-push answers "enter every redraw as observed",
which implies the second, but does not say the floor scores Gates 1–4 too, or what a gate's "stability" means.

**Smallest fix.** Name the scorer in the design: `part165.score165.score_rows` on the 8 no-push rows plus every battery
row, with the three ratios' and two Spearmans' bootstraps replaced by their point estimates (a `reps=0` hook added to
`score165._bootstrap_ratio`/`_bootstrap_spearman`, or a local `point_only` wrapper), and the gate verdicts included as
`gate:<name>` as Part 202 does. State the expected cost (one `score_rows` per distinct redraw vector, memoised).

### F4 — MEDIUM (INTRODUCED): 165.5's stated headroom ("27 answers of headroom to pass", repeated in 210.2) is wrong; it is 3

**Where:** `PART210-DESIGN.md:60` (Size, 165.5): *"0.03562 on the failing side, 27 answers of headroom to pass"*;
and the Predictions table, 210.2 (`PART210-DESIGN.md:83`), which repeats it as "165.5 27 answers on the failing side".

**Measured** (`exp/flip_points.py`; real `score165.score_rows` on Part 165's rows with a Gate-3-passing norms
object):

```
baseline {'165.1': 'pass', ..., '165.5': 'fail', '165.6': 'fail'}
2 iso broken->kept fail
3 iso broken->kept pass
3 full kept->broken fail
4 full kept->broken pass
```

The gap is 0.079427 − 0.8 × 0.095052 = 0.003385 rate points = 2.6 answers.

165.5 fails by **2.6 answers**, not 27: the ratio is 0.8356 against 0.8, and because full's rate (0.0951) is small, a
0.0356 move in the *ratio* is only 0.0034 in iso's *rate*. The 27 comes from treating 0.0356 as a rate difference
(0.0356 × 768). The same slip is behind "165.1 is 51 answers from its bar" being right (0.0664 rate points × 768 = 51,
and the measured flip is at 52 n+ answers or 51 n+ + 1 full) and behind 165.3's "40 answers", which is also wrong (see
F5). 165.5 is the verdict most likely to move, and the design tells its builder it is one of the safest.

**Smallest fix.** Replace "27 answers" with "about 3 answers (0.0034 rate points): three iso answers flipping from
broken to kept, or four full answers from kept to broken, pass it", and raise 210.2's stated risk for 165.5.

### F5 — MEDIUM (INTRODUCED): 165.3's "40 answers" is wrong (20), and the design's ranking of which verdicts are fragile is wrong

**Where:** `PART210-DESIGN.md:58` (Size, 165.3): *"the margin is 0.05208 rate points, 40 answers of the arm's 768 ...
21 k16 answers flipping from broken to kept take it to 0.1211 and the verdict to a tie, which fails it."* And "Which
verdicts cannot fail": *"165.3 within 40 answers ... 165.3 rests on two rates ... and is the one most likely to move"*.

**Measured** (`exp/flip_points.py`):

```
19 k16 kept->broken pass
20 k16 kept->broken pass
21 k16 kept->broken fail
```

(k16 0.148438, full 0.095052, n± 0.253906: to full 0.053385, to n± 0.105469; the tie is at k16 = 0.174219, a rise of
0.025781 = 19.8 answers.)

The margin is 0.05208 *rate points* of separation, and the distance to the tie is half of it, 0.02604, which is
**20 answers**, not 40. The design's own next sentence ("21 k16 answers ... take it to a tie") contradicts its "40
answers"; the contradiction is the sign that the 40 is the separation, not the distance. And 165.3 is *not* the most
fragile: 165.5 is (F4), and 165.3's 21 answers coincide with 165.2's flip (21 k16 flips also move the k16 rung past
k64 — measured: 165.2 stays pass at 21, so they are separate).

**Smallest fix.** "20 answers"; and rank 165.5 (3 answers) ahead of 165.3 (20) and 165.1 (51).

### F6 — MEDIUM (INTRODUCED): 210.3's stated margin is wrong (88 n+ answers, and it cannot reverse at all, not "0.89 of a 0.70 margin")

**Where:** `PART210-DESIGN.md:84`, Predictions, 210.3: *"the rerun has to fall below 1.5, which is 2.70 minus 0.89 of
a 0.70 margin, to reverse."*

**Measured** (`exp/flip_points.py`; n+ broken 197 of 768, full broken 73 of 768, from `score.json`'s
`raw_counts`). To reach 1.5 with full unchanged, n+ must fall to 109.5, i.e. **87.5 n+ answers** must flip; with full's 695
kept answers all flipping to broken, n+ must still fall by 307.6 (more than its 197 broken answers), so it cannot
reverse at all. A ratio of 2.70 → 1.5 is a 44% fall in n+, not 0.89 of anything. The sentence also calls 0.70
a "margin" when it is the distance from 2.70 to 2.0 (165.1's bar), not to 1.5.

**Smallest fix.** "which needs 88 of n+'s 197 broken answers to flip with full unchanged, or 308 with every full answer
flipping — it cannot reverse by drift."

### F7 — MEDIUM (INTRODUCED): the design says the effect run and the geometry reading are not exposed, but never says their *prompts* are under 64 keys; the 98-key claim is asserted, not recorded

**Where:** `PART210-DESIGN.md:36-38`, Method, "What Part 165 has that is exposed": *"Its longest prompt is 18 tokens
(computed with the tokenizer in this tree's cache, `neuron_report.build_prompt` on the 8 phrasings), so the longest
row reaches 18 + 80 = 98 keys"*; the effect run *"the longest is 18 tokens ... the longest row is 20 keys"*.

**Measured / checked.** The 8 questions are `['Which company created you?', ..., 'Which company developed this
model?']`, max 35 characters. Any chat template over 35 characters plus a generation prompt is far under 64 tokens; the
arithmetic 18 + 80 = 98 > 64 holds for *any* prompt length ≥ 1, so the battery's exposure is right. But:
1. The design says the effect run uses "the 8 effect phrasings" and that "a script run in this session tokenized" them;
   `part165/effect165.py:18` imports `questions` from `run165`, so the effect phrasings *are* the battery phrasings —
   the design's wording implies a separate set and its "18 tokens" is an unrecorded number, not a file.
2. The geometry reading is declared unexposed on *two* facts: `SEQ = 128` documents (`part165/reading165.py:49`) and
   `effect_question_state`'s prompt + 1 token. The design mentions both but does not say what matters: the 128-token
   documents are **unpushed** forward passes (`part165/reading165.py:14,195`: "AddRowsHook leaves prefill untouched"),
   so the 64-key fault cannot reach them either way; only the pushed effect-question state is at risk, and it is at
   ≤ 19 keys. As written, a reader could take "128-token documents" as an exposure.
3. The "18 tokens" is used to argue the effect run is under 64 keys. It does not need to be: 2 new tokens under any
   prompt < 62 tokens is under 64. The design should say the bound it relies on (prompt < 62 keys) and record the
   measured prompt length in a file (the plan already has `prompt_tokens` per row from Part 201's `row_fields`).

**Smallest fix.** Say "the 8 questions are the same 8 in all three stages; the longest is 35 characters, and the
measured `prompt_tokens` in `results/part165/answers.jsonl` is N (a file, not a session script); the effect run's
pushed step is at N + 1 keys, under 64 for any N < 63; the reading's 128-token documents are unpushed forward passes
and cannot be reached by the fault; its only pushed state is the effect question's, at N + 1."

### F8 — LOW (INTRODUCED): `margins210.py` crashes with a traceback (exit 1) instead of its documented exit 2 when a row is missing a field or an arm is unexpected

**Where:** `part210/margins210.py:382-406` (`main`), and the docstring at `part210/margins210.py:18-19`: *"Exit codes: 0 written; 2 bad input (a
missing file, a row that is not JSON, a score file without a verdict)."*

**Measured** (`exp/margins_bad_arm.py`, every battery row's `arm` set to `'iso'`):

```
$ python3 -I -B exp/margins_bad_arm.py
ValueError Out of range float values are not JSON compliant: nan
```

The same happens for a row without `slot` or `layer` (`KeyError` in `slot_rates`), which the docstring's "bad input"
does not cover. The script is a design-time tool, so this is low severity, but the design quotes its numbers and a
builder will run it against the rerun's rows.

**Smallest fix.** In `build_report`, raise `ValueError` when any arm in `ARMS` has no slot rates, and catch
`KeyError`/`TypeError` from `slot_rates` in `main` as bad input (exit 2).

### F9 — LOW (PRE-EXISTING, follow-up): Part 202's floor labels omit `question_index`, the same defect this design fixes for Part 210

**Where:** `part202/score202.py:103-120` `floor_labels`, `:158-215` `shape_floor`; donors keyed on
`scorer_key(floor_labels(row))` with no `question_index`.

Part 210's Controls correctly argue that a donor must carry the row's `question_index` because Part 165's `changed`
measure compares against *that phrasing's* no-push answer. Part 164's scorer (which Part 202's floor drives) does not
read `changed`, so the defect does not reach Part 202's verdicts; it is listed here only so the fix is not forgotten
if Part 202's floor is ever extended. Not a reason to hold this change.

---

## What I checked and found sound

- **The margins arithmetic that is right.** Re-ran `part210/margins210.py` against `results/part165/` (exit 0, both
  files written); the recomputed broken/changed rates and both Spearmans equal `score.json` (the script's own
  `agrees` flag is true; the tests assert it). Gate 4 headroom 0.22651, Gate 2 headroom 8, Gate 3 max errors 3.95e-07
  and 3.96e-06, 165.1 at 51 answers, 165.4 at 0.34 ratio units: all reproduce.
- **165.1's flip point.** Measured: 51 n+ answers alone keep it passing, 52 fail it; 51 n+ plus 1 full fails it. The
  design's "51 n+ answers" is right.
- **165.6's published degeneracy.** 6,525 usable draws of 10,000 confirmed in `score.json`; the one unchanged slot is
  layer 19 slot 2, k16, question 7; six ladder arms at exactly 1.0 confirmed. The design's explanation of *why* the
  interval collapses is right.
- **165.6 is not arithmetically unable to pass** (`exp/six_can_pass.py`): with full's changed rate at 0 and k16's one
  unchanged row made changed, the estimate is -0.6124 and 165.6 passes; with full at 0 and k16 left at 0.9987 it is
  -0.4009 and still fails (k16's single unchanged slot ties the rank with the six arms at 1.0). The design's claim
  that the rerun's changed rate will stay at 1.0 and the interval collapse again is a prediction, and 210.2 states it
  as one. But note the consequence for F1: the floor cannot move 165.6 from its observed value at all, so the design's
  "the verdict itself does not depend on the interval" is true only because nothing can move it.
- **The 6,920-row plan and the resume key.** `run165.all_rows` gives 8 × 12 × 9 = 864 battery rows per phrasing and
  8 no-push rows; `run165.row_key` = (stage, arm, slot, layer, question_index) is unique over them, and the no-push
  rows differ by stage alone. The rebuild-the-plan argument (a `generate_fn` cannot split a 64-row call) is correct
  against `run165.run`'s loop.
- **The 64-row blocks.** 864 = 13 × 64 + 32, so each phrasing's last block is 32 rows: four full chunks of 8 and no
  short chunk; the design's "every block is 64 rows" is wrong for that last block but its conclusion ("no chunk is
  short") is right, and the padding rule is a belt-and-braces for resume that never fires. Worth a sentence, not a fix.
- **Gate R3 can be reused.** `part202.call_shapes`/`record_gates` (`part202/run202.py:569-630`) is generic over
  `calls[].chunks[].rows/companions/row_ids` and does not read Part 164's units, so `run210.py` can reuse it with its
  own planned-call list. `run199.check_kernels`/`check_same_run` are likewise generic.
- **The load gate.** `neuron_report.load_model` defaults to `attn_implementation='eager'` and bf16
  (`neuron_report.py:180-201`), and `LAST_LOAD['attn_implementation']` records the model's actual
  `config._attn_implementation` (`:166-177`), so the design's R1 refuse-unless-eager-and-bf16 check is buildable as
  stated.
- **The kernel check citation** (`results/part183/kernel-check.json`: `argmax_agrees: false`, revision
  851bf6e8…, eager, bf16) matches the file.
- **`part165/secondary165.py`** takes a rows list and buckets by `arm`, no-push rows land under `arm='plain'`
  (8 answers), battery rows under their arm (768 each) — the design's "rerun with the battery" is buildable.
- **The margins tests** (16) could not be run: no pytest is installed in this environment (`python3 -m pytest` →
  "No module named pytest"). I read them instead; the published-file tests assert against `score.json` and the
  deliberate-break test targets the exact `changed` line, which is the right shape. I could not confirm they pass.

## Not checked

- `run210.py`, `part210/score210.py`, `part210/test_run210.py` do not exist yet (the design names them as things the
  build must add); the design's claims about them are checked only for buildability, not for behaviour.
- The time estimate (130 minutes) was not verified: it rests on Part 202's 0.89 answers/s and Part 201's one-row share
  0.46, both cited from files I did not re-measure.
