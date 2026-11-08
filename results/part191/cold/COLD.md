# Cold review: Part 191 results (`results/part191/`)

Scope: `git -C /home/elisepc/green/worktrees/ab-part191 diff 340b0d3..HEAD` — one commit, `8091960 "Part 191 results
(local, before cold check): all six predictions held on eager; SDPA answers were cut short"`, adding
`results/part191/{README.md, answers.jsonl, launch.jsonl, load.json, score.json, tables.md}` and nothing else
(`git diff --name-only` confirms; `part191/run191.py`, `part153/score153.py` and `PART191-DESIGN.md` are unchanged since
340b0d3). So every finding here is about what the README says versus what the committed files show; no code changed.

Measurement scripts (in this domain, read-only against the worktree):
- `measure.py` — recomputes every README number with the **real** `part153.score153.score_rows`, the **real**
  `part86.triage86.triage_answer`, and the two answer files. Output: `measure.out`.
- `drive_run191.py` — drives the **real** `part191.run191.main()` with a stand-in model/generator (no torch, no
  network) to test the load-record gate on resume and interruption. Output: `drive.out`.

## Verdict: SHIP-WITH-FIXES

The registered result is sound. I reproduced all six prediction verdicts and every number in the main table and the
per-neuron table from the committed files with the unchanged scorer, and the gates pass as claimed. Nothing in the
scored claim is wrong. The defects are in the **descriptive** section ("the SDPA answers were cut short") and in one
sentence that overstates a measurement: two numbers there are wrong, one conclusion is contradicted by the files, and
the explanation of *why* the counts held is the wrong mechanism. Those are the sentences a public reader will take as
the part's explanation, so they should be fixed before this goes public. All fixes are README edits; no rerun is
needed.

## Findings

### F1 (Major, INTRODUCED): the unpunctuated-ending counts measure emoji, not truncation; the eager figure is 2, not 58

`results/part191/README.md:65-66`: "Of the answers under the cap, 245 of 838 SDPA answers (29%) end without closing
punctuation, against 58 of 588 eager answers (10%)". The eager figure is dominated by answers that end with an emoji
(😊), which is a complete answer, not a truncated one — and on eager those emoji-ending answers are the *longest*
ones (median 82.5 tokens, i.e. at the cap boundary), while the SDPA ones are short.

Measured (`measure.out`):

```
eager under cap 588 no closing punct 58 9.9% of which emoji-ending 56
sdpa under cap 838 no closing punct 244 29.1% of which emoji-ending 7
eager under cap, not emoji-ending 532 no closing punct 2 0.4%
sdpa under cap, not emoji-ending 831 no closing punct 237 28.5%
```

and the last character of the 58 eager "unpunctuated" answers: `[('😊', 54), ('🔍', 1), ('�', 1), ('e', 1), ('r', 1)]`.

So the honest contrast is **237 of 831 (28.5%) SDPA against 2 of 532 (0.4%) eager**, which is a far stronger
statement than "29% against 10%" — but the 10% is wrong and invites the reader to think eager answers are also cut
short. Why it matters: this number is the part's evidence that SDPA generation is broken on this card; a reader who
checks it finds eager answers "end without punctuation" 58 times, and most of them simply end in an emoji.
Smallest fix: define "closing punctuation" to allow a trailing emoji (or count only answers whose last character is a
letter, digit, comma or hyphen), and report 237 of 831 against 2 of 532.

### F2 (Major, INTRODUCED): "The counts held anyway because the company is named early" is contradicted by the files

`README.md:68-73`: "The counts held anyway because the company is named early: of the neuron arm's maker hits, 138 of
143 name it in the first sentence on eager, 147 of 151 on SDPA." The first-sentence numbers are right (reproduced
exactly: 138/143 and 147/151 with the real triage and the scorer's `company_hit`), but they do not support the
conclusion. The first sentence is inside the window both kernels share, so it is the *same* on both kernels by
construction — it cannot explain why the counts held. What the files show is a different mechanism: the two kernels
agree on the whole answer whenever the SDPA answer is short enough, and the SDPA answers are short.

Measured (`measure.out`, plus a direct check):

```
identical text total 122 {'neuron': 65, 'random': 47, 'plain': 10}
identical at <=64 tok 122 of 122   max sdpa tok among identical 58   identical with sdpa tok>60 0
sdpa most common lengths [(63, 37), (58, 30), (64, 25), (68, 25), (62, 24)]
```

All 122 identical answers are the short ones; no answer longer than 58 SDPA tokens is identical. So the counts held
because most SDPA answers **stopped before the kernels diverge** (the part's own Part 178 window, ~64 keys), not
because the company was named early in answers that continued. The README's sentence also implies SDPA answers ran on
past 64 keys and were still hits; the files show the opposite. Why it matters: this is the part's explanation of its
own result; a reader who checks it will find the explanation is not what the data shows. Smallest fix: replace the
"named early" sentence with the divergence-boundary observation above (122 identical, all ≤58 SDPA tokens; no longer
answer identical), and drop or demote the first-sentence counts to a control that shows the shared prefix is where
the name sits.

### F3 (Minor, INTRODUCED): "which is where most of the drop from 0.629 to 0.596 comes from" is not supported by its own numbers

`README.md:71-73`: "On the neuron maker rows the company is named in 164 of 240 answers on eager and 165 on SDPA; eager
loses 21 of those as broken against 13 on SDPA (Alibaba or Tongyi: 0 against 1), which is where most of the drop from
0.629 to 0.596 comes from."

Reproduced (`measure.out`): 164 named / 143 hits / 21 broken on eager; 165 named / 151 hits / 13 broken on SDPA. The
drop is 8 hits (143 → 151). Of the 8, **7** are the extra broken answers (21 − 13 = 8, minus the 1 extra
Alibaba/Tongyi loss on SDPA, which goes the other way) — so "most" is arithmetically about right (7 of 8) only if the
reader does that subtraction themselves; the sentence as written gives the broken counts but never says the
difference is 8, and the "Alibaba or Tongyi: 0 against 1" clause is a *gain* for eager that the sentence folds into
the same "loses" clause. The bigger unstated fact is that 17 of the 21 eager broken answers are at the 120-token cap
(`broken at cap 17` in `measure.out`), i.e. the drop is a length effect, which the Limits section says but this
sentence does not. Why it matters: it is a causal claim about the headline number; it should be stated as
"7 of the 8-hit drop is extra broken answers, 17 of which are at the cap". Smallest fix: one sentence with those
three numbers.

### F4 (Minor, INTRODUCED): the 29% figure does not match a plain recomputation; the stated population is not what the code counts

`README.md:65`: "245 of 838 SDPA answers (29%)". 244/838 = 29.1% with the most natural rule (last non-space character
is not `. ! ? … " ' ) ] *`); 245 is reached only by a slightly different punctuation set. Not a material error, but
the README does not say what "closing punctuation" means, and the number is off by one against the obvious reading.
Fold into F1: state the rule, report the recomputed number.

### F5 (Minor, PRE-EXISTING — code, not this change): the load-record gate can certify a mixed-kernel answers file

`part191/run191.py:93-100`: the record entry is written only *after* `battery153.run()` returns. A run interrupted
mid-battery leaves a partial `answers.jsonl` and **no** `load.json` entry; the resume appends its rows and writes one
entry for the whole file. Driven with the real `run191.main()` (`drive.out`):

```
(3) interrupted run raised simulated XPU fault
    partial rows 20 load.json exists False
    resume rc = 0
    after resume: rows 840 record entries 1 (one entry for a file written by two invocations)
(3b) resume on sdpa rc = 3 recorded kernels ['sdpa'] rows 840 -> gate certifies eager for a file whose last 820 rows are sdpa
```

The resume-on-a-different-kernel case is caught (rc 3), but the reverse order — interrupt on SDPA, resume on eager —
would record only `eager` and pass, for a file whose first rows came from SDPA. This is the exact failure the design's
"Changes before registration" says was fixed ("a resumed file cannot be certified by a later invocation that loaded a
different kernel"); the fix covers only the case where the *later* invocation is non-eager. It does **not** affect
this result: `load.json` has one entry, `answers_written: 840` equals the file's 840 rows, and the launch record
shows one step, so the committed file was written by one invocation. PRE-EXISTING (the code is unchanged since
340b0d3) and a follow-up. Smallest fix: write the load entry (with `answers_written: 0`) **before** `run()` starts,
or refuse to append to an existing `answers.jsonl` whose record has no entry.

### F6 (Minor, PRE-EXISTING — code): an interrupted run leaves a partial file with no record, and a loader failure leaves nothing

`drive.out` (3) and (4): a generator fault propagates out of `run191.main()` with 20 rows on disk and no `load.json`;
a loader that raises leaves no record. That is safe (the next invocation records itself), but a partial file with no
record and an empty `loads` list would fail the gate rather than silently pass — acceptable. Noted only because the
design text implies the record always accompanies a file.

### F7 (Minor, wording, INTRODUCED): "every registered prediction held" overstates 191.6

`README.md:22-23` and line 41: 191.6 is "All four published verdicts are the same" — and it did hold (same four
verdicts, reproduced). But 153.2 "failed" *by one unit* (2 of 10 identity against a bar of 3 on SDPA; **1 of 10** on
eager, `measure.out`), so "held" here means "the failure reproduced", which the table says but the prose does not.
Not a finding on its own; flagged because the headline sentence reads as if all six predictions were confirmations.

### F8 (Minor, INTRODUCED): "individual counts move by up to 5 in either direction" is right, but the largest movers are the two units nearest their threshold

`README.md:58`. Reproduced from `measure.out`: L11N5647 16 vs 21 (−5), L23N1743 17 vs 14 (+3), L23N414 17 vs 21 (−4),
L25N4458 20 vs 19. "Up to 5 in either direction" is true (max +3, max −5). No defect; noted because L21N7134 sits
exactly on the 12-of-24 bar and L22N6997/L25N140 sit at 11 — a one-answer move flips 191.1's "7 of 10". The README's
Limits section covers one answer per phrasing, so this is adequately hedged.

## What I checked and found sound

- **Every number in the main table** (`README.md:12-20`) reproduced with the real `score153.score_rows` on both
  files: 7/10 at 12-of-24, mean 0.596 vs 0.629, random 0.113 both, ratio 5.30 vs 5.59, identity 1 vs 2, guards 9 vs 9,
  L24 N5017 17/24 and 7/8 vs 20/24 and 7/8. (`measure.out`, "scorer summary".)
- **All six prediction verdicts** against the registered wording in `PART191-DESIGN.md:81-86`: 191.1 7≥5 held;
  191.2 0.596 ∈ [0.45, 0.80] held; 191.3 0.596 ≥ 2×0.113 held; 191.4 0.113 ≤ 0.20 held; 191.5 17≥16 and 7≥5 held;
  191.6 same four verdicts held. The README's prediction table quotes the registered wording faithfully.
- **Gates**: 840 rows, 840 unique, 840 nonempty, 0 duplicates; plain company hits 0 (limit 2); selection 10 of 10.
  Reproduced by `score_rows`; matches `tables.md`.
- **Load record**: one invocation, `attn_implementation` eager and requested eager, chunks 8/8/2/2, 840 written,
  `seconds` 3940.65 = 65.7 min, matching "66.6 minutes (23:22:04 to 00:28:42)" (launch record 23:22:03, 66.6 min =
  3997 s wall, consistent with 3940 s inside the step).
- **Row comparability**: the two answer files have identical `(unit_id, arm, question_index)` key sets, identical
  question text/kind and company per key, same layer/neuron; so the row-by-row comparison the README claims is valid.
- **Descriptive counts that are right**: 122 identical answers (65 neuron / 47 random / 10 plain); medians 82 vs 60;
  cap 252 vs 2; "most common lengths between 55 and 70" (top five SDPA lengths 63, 58, 64, 68, 62); "QwQ" 0 vs 15
  (exact case; 1 vs 19 case-insensitive — the README's capital-Q figure is what the files show); 164 vs 165 named;
  21 vs 13 broken; 0 vs 1 Alibaba/Tongyi; 138/143 and 147/151 first-sentence.
- **The scorer is unchanged** (`git diff` shows no change to `part153/score153.py`), and `score.json` matches what
  `score_rows` returns on the committed answers — the score file was not hand-edited.
- **The kernel claim itself**: `load.json` records eager, the code path (`neuron_report.load_model` default eager,
  `run191.chunked`) is consistent with the record, and the chunked generator's order/size contract is exercised by
  `part191/test_part191.py` (I could not run pytest here — no pytest/torch in this environment — so the tests are
  read, not run; see self-assessment). Reading `test_part191.py`, the tests are not vacuous: `test_real_battery_run_through_the_wrapper`
  drives the real `battery153.run()` and `battery148.pushes_for` and fingerprints each row's push, so a row receiving
  another row's push would fail; `test_resume_cannot_launder_the_record` covers the non-eager-resume case (but not
  F5's reverse order).

## Self-assessment / what I could not verify

- I could not run the pytest suites (no pytest, no torch in this sandbox). The code findings F5/F6 come from driving
  the real `run191.main()` with stand-ins, which exercises the record-writing logic but not the torch path.
- I could not re-derive the SDPA answers from the model (no GPU, and a live run is out of scope), so "SDPA answers
  were cut short" is checked only as an internal consistency claim against the files, not against a reference
  generation.
- F1's "right" punctuation rule is a judgement call; the 2-of-532 figure depends on treating any trailing emoji as
  closing. A reader could reasonably want a rule that also counts a trailing markdown `**` as closing; that would move
  both sides slightly but not the contrast.
