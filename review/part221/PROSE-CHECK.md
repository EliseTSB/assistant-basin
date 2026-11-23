# PROSE-CHECK — Part 221 README (cold read)

Checked: `results/part221/README.md` against `results/part221/SCORECARD.md`, `results/part221/scorecard.json`,
`PART221-DESIGN.md` v3 (Question, Size, Controls, Predictions, Reading rules, Headlines, Limits) and
`PREDICTIONS.md` Part 221 rows (lines 1789–1799).

Method: every number in the README's tables and prose was compared with the scorecard and the JSON; every McNemar
p was recomputed by hand/program from the lost/gained counts; label files were counted (258 items × 2 passes, caps);
the scorer was re-run on a **copy** of the results in this domain (the worktree was not changed) to test the
"Rescore it yourself" claim.

## SUMMARY (counts by kind)

| kind | count |
| --- | --- |
| wrong number, direction or verdict | 0 |
| headline that is not the registered headline sentence | 1 |
| stronger than the numbers / exploratory stated as a finding | 0 |
| a sum stated as a decomposition | 0 |
| written after the data and not labelled so | 0 |
| undefined term | 2 |
| external claim not checkable from the supplied files | 1 |
| minor / nit | 2 |

The README is numerically clean. Every table cell, p-value, unit list and verdict I checked agrees with
`scorecard.json` / `SCORECARD.md` / the design. The flags below are framing and reference, not arithmetic.

## Must-fix

**1. Title — headline outside the registered set, and states an equality the registration avoided.**
Quoted: *"Part 221: putting back the pre-SFT weights of the other 60 early heads removes as many slot personas as
putting back all 64"*
Reason: the design's registered headline for 221.1+221.2 holding is "transplanting the other sixty early heads
removes **nearly as much** as the whole band does (at least the band's net loss minus 2)". The title converts that
one-sided, margin-of-2 claim into an equality claim. The point estimates do coincide (R11 4/21 vs R1 4/21; net 11 vs
11; the same 11 units), so the title is not contradicted by the numbers — but no equivalence test was run, and a
16-unit paired design cannot separate equality from a 1–2 unit deficit. Either use the registered phrasing
("removes nearly as much as putting back all 64") or keep "as many" and say it is the counts that match exactly
(the body already does). As written, the strongest sentence in the document is the one sentence not licensed by the
registration.

## Should fix

**2. Undefined term: "Part B".**
Quoted: *"the five slot units Part B had selected (p92-02, -05, -09, -13 and -46) replicated on new stories"*
Reason: "Part B" is not defined anywhere in this README and is not a part of this repository's numbering that a
reader can look up from this file. Say what Part B is (a section of Vesper's study) or drop the label.

**3. Undefined term: "screen A".**
Quoted: *"The four heads were chosen by screen A, which was not registered."*
Reason: the disclosure is right and important, but a fresh reader cannot tell what screen A is or where it is
described. One clause ("the head-screening run in Part 219") would close it.

**4. External claim with no pointer.**
Quoted: *"In Vesper's concept-versus-slot study, the five slot units ... replicated on new stories while most of the
other slot units did not"*
Reason: this is the premise of the whole core-five subsection, and it cannot be checked from anything in
`results/part221/` or from the design. It is correctly labelled as written after the data, but it needs a file or
council pointer the way the design cites its other sources.

## Minor

**5. A sum compared against a different intervention.**
Quoted: *"The per-layer nets sum to 20, more than the 11 the four layers lose together, so the layers' effects
overlap rather than add."*
Reason: R14–R17 each move 15 complement heads; the 11 is R1, which moves all 64 including the four big heads. The
sub-additivity conclusion survives a fortiori (adding the four heads could only raise R1's net), but the sentence
reads as if the two sides were the same intervention. It is inside the Exploratory section, so it is labelled.

**6. Headline split across two sentences.**
Quoted: *"largest kept-head pre-norm drift 20.95%, largest change in the attention output's rms 10.31%, both at
layer 1"*
Reason: the design asks for the coupling numbers "in the same sentence" as the headline, and the scorecard's
generated headline puts R11's and R12's numbers and the 221.3 clause in one sentence. The README gives R11's
numbers in the opening sentence and R12's in "What the controls say", with the 221.3 clause in a separate sentence.
Content is complete; only the packaging differs from the registered headline.

## What I verified as correct

So the parent can see this was a real check, not a skim:

- All nine table rows (slot persona, slot coherent, concept persona, net on 16, McNemar p) match `SCORECARD.md` and
  `scorecard.json`, including R12's p = 0.070 (JSON 0.07031) and R11/R1's p = 0.00098 (JSON 0.0009766).
- Every McNemar p recomputed from the lost/gained pairs: 11-0 → 0.000977; 7-1 → 0.0703; 9-0 → 0.00391; 6-0 → 0.0313;
  2-0 → 0.5; 3-0 → 0.25; R8's 4-1 → 0.375 (confirmed against Part 219's `scorecard.json`).
- Lost/gained unit lists: R11 = R1 exactly; R12 overlap 6 both / 5 only R11 / 1 only R12; all core-five splits
  (3/8, 3/8, 3/4+1, 3/6, 3/3, 1/1, 1/2, R8 2/2+1). Sums: 9+6+2+3 = 20; 11+3 = 14 vs 11.
- Downstream nets (5, 5, 2, 5, 3, 4, 4) and all-21 nets (16, 16, 8, 14, 9, 6, 7) match the scorecard.
- "pushes at layers 1 and 2" for the five downstream units: confirmed by Part 219's design (p92-07 layer 1;
  p92-10/20/42/50 layer 2) and Part 101's layer table; "after all four big heads" for the 16 is consistent with
  attention running before the MLP of its layer.
- Coupling numbers 20.95% / 10.31% (R11) and 15.00% / 11.86% (R12), both at layer 1; R12 norm ratio 0.9991–1.102;
  gate items (kernel, R0/R1 reproduction on 474 rows, R13 = R1) all match `scorecard.json`.
- "two Haiku passes labelled all 258 items of each new run": 258 rows in every `label-input/` file and 258 distinct
  (id, question, unit) tuples in every `labels-*` and `labels2-*` file; 18 rows carry `max_tokens` 4000 across the
  six runs, exactly as stated; every label row has a `max_tokens` value.
- DeepSeek agreement on the six new runs: 23, 23, 24, 24, 24, 23 of 26 (0.8846–0.9231) and 25 of 26 on coherence —
  matches `scorecard.json` `deepseek_agreement`.
- "Rescore it yourself": `python3 -I part221/score221.py --out <copy>` ran without torch or a GPU and reproduced
  `SCORECARD.md` byte-identically. `scorecard.json` differed in exactly six fields, all `runs/*/labels_from`, because
  I pointed `--out` at a directory outside the repo and the scorer records a repo-relative path when it can
  (`score219._repo_relative`). Run as documented inside the repo it would write `results/part221` and match. The
  scorer does read Part 219's results for R0, R1 and R8, as the README says.
- No unlabelled post-data writing found: the core-five section is marked "(written after her 2026-11-22 finding)",
  the per-layer, R11+R8 and downstream material sits under "Exploratory" and the Limits restate the post-hoc
  choices. The R11+R8 sum is explicitly called "a sum of two separate runs, not a decomposition", matching the
  design.

## BLOCKERS

none. Nothing in the README required a judgement I was not given; the title flag is a recommendation, not a
question. I did not edit any file outside this domain, and I did not run the scorer against the worktree.
