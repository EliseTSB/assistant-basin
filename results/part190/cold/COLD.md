# Cold review — part190res (results/part190 write-up, commit ce80849)

**Verdict: SHIP-WITH-FIXES.**

The change is one commit (`ce80849`) that adds `results/part190/README.md`, `results/part190/results.json`,
`results/part190/run.json`, `results/part190/run.log`. No code changes, so the numbers themselves are not new code
under review — the review is: do the README's numbers and verdicts match `results.json`, the design's reading rules
and the real arrays, and does the README claim more than the run shows?

**What is sound (measured, not assumed).** I re-ran the committed code on Part 188's real arrays into my own domain
(`rerun/`, `python3 part190/run190.py --out <my domain>/rerun`, 618 s, 2,298 evaluations) and diffed the whole result
tree against the committed `results.json`: **zero differences at any key** (float compare, 1e-12 slack). So every
number in the committed file is reproducible from the committed code. All 14 prediction verdicts, the four gates, the
headline permission, the operating doses, the intervals and the exploratory half-gain line in the README match
`results.json` exactly (checked cell by cell). The registration is intact: `PART190-DESIGN.md` and the 14 rows under
"## Part 190" in `PREDICTIONS.md` are byte-identical between the registration commit `755f745` and HEAD, and
`755f745` is an ancestor of `origin/main`. The code's verdict functions are the registered thresholds (I read
`PREDICTION_RULES`, `analyze190.py:539-554`), and the headline rule (`analyze190.py:595-612`) checks exactly the
conditions the design lists. The README's reading-rule consequences are applied: the title, the "What the edit is"
paragraph and the "By the registered reading rules" sentence all name the edit as the probe's reading written along
the verdict's direction, the verdict as class-informed (with the class-free result beside it, 190.14 held), and the
190.10 failure forces "partly moves imitations toward human sounds" whatever the verdict gain. That is the design's
rule, followed.

The findings below are about the README's *interpretive* sentences, not the numbers. The numbers are right; two
sentences built on them overstate what the run shows, and one of them (F1) is the very sentence the design's 190.10
rule forces the README to write — so the fix is a qualifier, not a retraction.

---

## Findings

### F1 — MEDIUM (INTRODUCED): "moves some imitations toward human sounds" is the opposite of what the edit does to imitations on average; C6 measures a *relative* reordering, not movement toward human sounds

**Where.** `results/part190/README.md:14-17` ("**But the edit moves some imitations toward human sounds** (190.10
failed: +4.1 and +12.4 points)"), the title (`README.md:1`, "moves some imitations toward human sounds"), and §3
(`README.md:67-69`, "imitations-toward-humans +0.75 points"). The design's rule at `PART190-DESIGN.md:213-214`
requires this phrasing, so the README is following the rule — but the rule's phrasing is wrong about this run's
data, and the README has the measurement to say so.

**What C6 actually is.** `analyze190.py:344-347`: for each held-out imitation, the nearest neighbour among real
recordings (R, C_R, E, H) is taken; the share whose nearest real recording is an H (human-made sound) clip is C6.
A rise means *more imitations' nearest real recording is human* — a rank-1 neighbour identity change. It does not
mean the imitation got closer to a human sound. The design's own wording at `PART190-DESIGN.md:102-103` says
"nearest neighbour", correctly; the README's gloss "moves … toward" does not.

**Measured trigger.** `probe_c6.py` (drives the real `build_edit`/`apply`/`evaluate` from the worktree on the real
arrays; output in `probe_c6.out`):

```
== A_to_B  alpha*=2  n imitations=267
  C6 share from real code, alpha=0 : 0.2884
  C6 share from real code, edited   : 0.3296   (results.json: 0.3296)
  mean cos to nearest H clip    : before +0.9131  after +0.9116  change -0.142 pt
  mean cos to nearest R/C_R clip: before +0.9258  after +0.9229  change -0.295 pt
  cos to own alpha=0 nearest H clip: mean change -0.197 pt; share of imitations moved TOWARD it 30.3%, away 69.7%
== B_to_A  alpha*=4  n imitations=266
  C6 share from real code, alpha=0 : 0.1353
  C6 share from real code, edited   : 0.2594   (results.json: 0.2594)
  mean cos to nearest H clip    : before +0.8989  after +0.8962  change -0.272 pt
  mean cos to nearest R/C_R clip: before +0.9161  after +0.9073  change -0.881 pt
  cos to own alpha=0 nearest H clip: mean change -0.535 pt; share of imitations moved TOWARD it 24.8%, away 75.2%
```

And the per-clip mechanism, `probe_c6b.py` (`probe_c6b.out`):

```
== A_to_B: 13 imitations became H-nearest
  their alpha=0 rank of nearest H clip: [13  0  0  0  0] (ranks 1..5)
  their cos change to nearest H : -0.058 pt
  their cos change to nearest nonH: -0.660 pt
== B_to_A: 37 imitations became H-nearest
  their alpha=0 rank of nearest H clip: [37  0  0  0  0] (ranks 1..5)
  their cos change to nearest H : -0.400 pt
  their cos change to nearest nonH: -2.054 pt
```

**Reading.** Every imitation that becomes H-nearest already had a human clip as its *second*-nearest real
recording (rank 2, all 13 and all 37). The edit pushes imitations away from *both* human and real-animal clips, and
in every case it pushes them away from the nearest real-animal clip *harder* than from the human clip — so the
human clip overtakes. In B to A the newly-H-nearest imitations move 2.05 points away from the nearest non-human real
recording against 0.40 points away from the human one. On average imitations move **away** from human sounds
(-0.14 and -0.27 points to the nearest H clip). The 190.10 failure is real and is correctly scored (C6 rises +4.12
and +12.41 points, both above the 5.0 bar in B to A and, for A to B, the README's "failed" follows from the B to A
value under "both directions"). But the *interpretation* "moves imitations toward human sounds" — and especially the
title's "some imitations toward human sounds" without the mechanism — describes a different phenomenon than what
happened. A reader will take "toward human sounds" to mean the edit is teaching the space that imitations are
human-made; the data show the edit is eroding imitations' ties to real animal recordings, and human clips win by
default because they were already close.

**Why it matters.** This is the sentence the design mandates as the headline consequence, and it is the one a front-page
reader will remember. It also under-states the concern: "moves toward human sounds" sounds like a partial success at
the wrong task; "loses its nearest real-animal match to a human clip that was already second" is a sharper and more
alarming failure, and it is what the numbers show.

**Smallest fix.** Keep the 190.10 verdict and the "partly moves imitations toward human sounds" phrase the rule
requires, but add one measured clause to the sentence at `README.md:14-17`, e.g.: "C6 counts imitations whose
*nearest real recording* is a human clip; in B to A all 37 imitations that crossed already had a human clip as their
second-nearest real recording, and imitations moved slightly *away* from human clips on average (−0.14 and −0.27
points), so the rise is imitations losing their nearest real-animal match, not getting closer to human sounds."
(Optionally the title could say "loses some imitations' nearest real-animal match to human sounds".) The design's
rule at `PART190-DESIGN.md:213-214` should be amended for future parts to say what C6 measures; that is a design
fix, out of scope here.

### F2 — LOW-MEDIUM (INTRODUCED): "half the gain" holds in A to B only; in B to A the reported "half-gain" dose gets 73% of the gain

**Where.** `README.md:66-69`: "**… a smaller dose gets half the gain for much less** … In B to A, alpha 1 gives
+0.119 with +1.88 points and 2.8%, against +0.162, +12.41 points and 14.8% at dose 4."

**Measured trigger.** From `results.json` (`directions.<dir>.ranks.1.exploratory_half_gain` and `.operating`):

```
A_to_B half-gain alpha 0.5 rise +0.128 = 51.2% of operating rise +0.249 | C6 +0.75 vs +4.12 | C2 0.012 vs 0.067
B_to_A half-gain alpha 1.0 rise +0.119 = 73.2% of operating rise +0.162 | C6 +1.88 vs +12.41 | C2 0.028 vs 0.148
```

The number is correct (+0.119 is in `results.json`). The *label* "half the gain" is the design's rule
(`PART190-DESIGN.md:219-222`: the smallest dose whose rise reaches half the operating rise), and the code implements
it correctly (`analyze190.py:388-392` — the smallest ladder dose whose rise is ≥ half). But because the ladder is
coarse (0.5 → 1 → 2 → 4), the smallest qualifying dose can overshoot: in B to A the rise is 73% of the operating
rise, not 50%. The README's heading "gets half the gain" then understates the gain at that dose while presenting the
cost saving as if it were at parity. The cost saving is real and large (C6 +1.88 vs +12.41), so the conclusion
("a cheaper dose exists") survives; the "half" is what is wrong.

**Smallest fix.** In `README.md:66-69` replace "gets half the gain" with "gets 51% (A to B) and 73% (B to A) of the
gain", or say "at half the gain or more". The code needs no change.

### F3 — LOW (INTRODUCED): "A and B (9 and 10 animal groups)" has the two folds reversed

**Where.** `README.md:6-7`: "folds A and B (9 and 10 animal groups), Chicken held out."

**Measured trigger.** `part188/manifest188.csv`, grouped by `group` within fold:

```
A: Bird, Cattle, Duck, Goat, Goose, Horse, Insect, Sheep, Snake, Turkey   -> 10 groups
B: BigCat, Cat, Crow, Dog, Frog, Mouse, Owl, Pig, Pigeon                -> 9 groups
```

Part 188's own cold read agrees: `results/part188/cold/COLD.md:14` — "manifest: 10 groups in A, 9 in B". Part 188's
README (`results/part188/README.md:54`) says "9 and 10" without naming the order, which is where the ambiguity
likely came from. The design says "9 and 10" too (`PART190-DESIGN.md:236`) but never attaches them to A and B, so
the README's parenthetical is the first place the order is asserted, and it is backwards.

**Why it matters.** Low, but it is a factual error in the data-description sentence of a public write-up, and it
matters to anyone reading the clustered intervals (the B to A test fold, which has 10 groups, is the one whose
interval is wider — see `dV_test` intervals: A to B 0.138–0.354, B to A 0.101–0.270; the narrower B to A interval
despite the larger rise is itself worth a reader's attention).

**Smallest fix.** `README.md:6-7`: "folds A and B (10 and 9 animal groups)".

### F4 — LOW (INTRODUCED, informational): results.json carries no provenance binding it to the registered design

**Where.** `results/part190/results.json` top-level keys are `directions, gates, headline, neighbours_readable,
predictions, readable, settings`; `settings` is `{dose_tolerance, doses, n_boot, n_controls, neighbours, ranks}`.
There is no commit hash, design hash or registration reference. `run.json` has `elapsed_s, edit_evaluations,
registered_settings, synthetic`.

**Measured trigger.** `grep -n "commit\|design_sha\|755f745" part190/analyze190.py part190/run190.py` returns only
docstring mentions ("registered in PART190-DESIGN.md"); no code writes a provenance field. I verified by hand that
the design and predictions at `755f745` are unchanged at HEAD, so *this* run is fine — but the file cannot show
that on its own. A future reader (or a future part that regenerates `results.json`) cannot tell from the artefact
which design version produced it.

**Why it matters.** Low for this change (I verified it independently); it is the kind of thing that becomes a real
problem the first time a design is amended after a run.

**Smallest fix.** Add `"design_commit": "755f745"` (or a hash of `PART190-DESIGN.md` + the Part 190 rows of
`PREDICTIONS.md`) to `run.json` by hand for this run, and to `run190.py`'s `run.json` writer for future runs. No
re-run needed.

### F5 — LOW (INTRODUCED): "the fit fold had already levelled off" is true only in the sense the dose rule uses

**Where.** `README.md:75-77`: "In A to B the held-out verdict kept rising past the chosen dose (0.879 at 2, 0.905 at
4, 0.915 at 8), while the fit fold had already levelled off."

**Measured trigger.** Ladder `V_fit` for A to B rank 1, alpha 0.25 → 16:
`0.879, 0.925, 0.970, 1.000, 0.995, 0.9998, 1.000`. The fit-fold verdict is at its maximum (1.000) at alpha 2, so
"levelled off" is true in the sense that it had reached the ceiling. But it *rose* from 0.970 (alpha 1) to 1.000
(alpha 2) — the last step before the ceiling — and the held-out AUC at alpha 2 is 0.879 while at alpha 4 it is 0.905.
The sentence is not false; it just invites the reader to think the fit fold was flat across 1 → 2 → 4 when in fact it
climbed to a ceiling at 2 and the held-out curve kept going. That is the more interesting fact and the one the
sentence is meant to convey.

**Smallest fix.** "while the fit fold had saturated (1.000 at dose 2) — a ceiling the held-out curve had not
reached." One clause.

---

## Things I checked and found sound

- **Reproducibility.** Committed code + real arrays → `rerun/results.json`, diffed against committed
  `results.json`: zero differences (all keys, floats to 1e-12). `run.json` elapsed (637.19 s) matches `run.log`'s
  start/end timestamps (637.0 s) and the log's last progress line (637.2 s).
- **Every README table cell and every number in "What the numbers say".** Checked against `results.json`:
  - 190.1 +0.249/+0.162; 190.2 0.879/0.970; 190.3 +0.007/+0.003 (0.8855−0.8790, 0.9726−0.9698); 190.4 −0.048/−0.189;
    190.5 −0.069/−0.258; 190.6 +0.26/−0.33; 190.7 6.7%/14.8%; 190.8 +0.322/+0.375; 190.9 +0.13/−0.39;
    190.10 +4.12/+12.41; 190.11 +0.015/+0.008; 190.12 0.039/0.000; 190.13 lows 0.138/0.101; 190.14 +0.167/+0.146.
  - Verdicts: held/held/held/held/held/held/failed/held/held/failed/failed/failed/held/held — all match
    `results.json`'s `predictions.*.verdict`.
  - Baselines 0.630→0.879 and 0.808→0.970; probe 0.918/0.965; Chicken 0.611→0.933 and 0.985; K 0.934/0.974;
    rank ladder 0.879/0.881/0.882/0.886 and 0.970/0.970/0.971/0.973; alpha 16 C6 +16.85/+48.87 ("+16.9 and +48.9");
    A to B held-out 0.879/0.905/0.915 at doses 2/4/8; exploratory lines +0.128/+0.75/1.2% and +0.119/+1.88/2.8%.
  - "2,298 edit evaluations in 637 s": 2×4×7×41+2 = 2,298, matching both the design and the runner.
- **Reading-rule consequences.** 190.10 failing → "partly moves imitations toward human sounds" (present, though
  see F1 for the gloss); headline names the edit as the probe's reading along the verdict's direction (present,
  title + line 19-21 + line 85); class-informed named in the title and verdict, class-free beside it because 190.14
  held (present); exploratory half-gain line present (see F2 for the label); no claim that the model "learned" to
  tell imitations apart (line 20-21 explicitly disclaims it).
- **Headline conditions.** `results.json` headline `permitted: true` with no reasons; I re-checked each condition by
  hand: gates 1–4 passed; 190.1, 190.4, 190.5, 190.13 held; main rank-1 change (+0.249, +0.162) exceeds both
  controls' p95 (−0.008/+0.075 and −0.133/+0.011) in both directions; 190.6 and 190.9 held.
- **Gate details.** Gate 3 self-test shown on this data (`rule_failed_on_main_edit: true`); gate3 per-rank passed at
  every rank; gate 4 base C1 0.818 ≥ 0.30; gate 2 identity 1.1e-16 and Part 188 reproduction within 0.005 (0.6305 vs
  0.6301, 0.9181 vs 0.9181, 0.8075 vs 0.8076, 0.9651 vs 0.9651).
- **Registration integrity.** `PART190-DESIGN.md` and `PREDICTIONS.md`'s Part 190 rows identical at `755f745` and
  HEAD; `755f745` is an ancestor of `origin/main`; the diff from `755f745` to HEAD is only the four results files.
- **C6 failure robustness.** `probe_c6c.py` (`probe_c6c.out`): the C6 rise is not driven by one group. Dropping the
  largest test group, A to B still rises +5.73 points (Cat dropped) and B to A +11.82 (Horse dropped); every B to A
  group's H-nearest share rises or stays flat. So the 190.10 failure is solid; only its gloss (F1) is wrong.
- **Failure paths (code, not diff).** `analyze()` returns `readable: false` and nothing else if Gate 1 fails; the
  runner exits 3 when unreadable; `_json` maps non-finite to `null`; `_both` maps undefined to "not read". These are
  pre-existing code, not touched by this change, and I found no path in the README that relies on them being wrong.
- **Tests.** I read `part190/test_part190.py` (24 tests). The threshold tests use values either side of each bar
  (0.0999 vs 0.10, 0.049 vs 0.05, −1.9 vs −2.0, 0.0 vs 0.001 for 190.13), and `test_190_4_can_fail_in_a_readable_run`
  genuinely exercises the v2 gap between 190.4's +0.02 and Gate 3's +0.05. I could **not run** them: pytest is not
  installed in this sandbox (`No module named pytest`) and I did not install it. So this is a read, not a run.

## What I could not check

- The unit tests were not executed (no pytest in the sandbox). The re-run above is a stronger check for the results
  file than the tests are, but a test that passes for the wrong reason would not be caught by either.
- I did not read the whole of `analyze190.py` (750 lines); I read the verdict, headline, gate, exploratory, C6 and
  neighbour code paths in full and grepped the rest. The re-run diff covers everything the code computes on this
  data, so a wrong-but-reproducible computation would be invisible to me — I checked the definitions against the
  design by reading, not by independent reimplementation.
- Nothing outside my domain was written; no live service was called; no credential file was read.
