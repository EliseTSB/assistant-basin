<!-- cheap-review.mjs cheap-review-p205-amendment2-ds-r1-2026-11-12: /home/elisepc/green/worktrees/ab-part201 3210e562..HEAD, lane ds, df-run exit 0 -->
# Cold review — Part 205, Amendment 2

**Verdict: SHIP-WITH-FIXES.** The branch is the right idea and its arithmetic is internally right *for the formula it
uses*, but as written the fallback contradicts the registered G1/O reading chain (F1), the family-Q rule contradicts
item 2 (F2), and the projection picks the optimistic end of the design's own n (F3). F1–F3 must be fixed before A0 is
run; F4–F9 should be fixed in the same edit. No model was run and no six-value gap or non-honesty P was computed.

Scope: `PART205-DESIGN.md:388-419`, the "## Amendment 2" section added by f8a005c8
(`git -C /home/elisepc/green/worktrees/ab-part201 diff 3210e562..HEAD` = one file, +33 lines; no code, tests or flags
changed). Context read: Method/Controls/Size/Predictions/Reading rules, `results/part205/pilot/PILOT.md`, the pilot's
`records.jsonl`, and the real scorer `part205/score205.py`. Command outputs cited below are in this directory
(`pilotcheck.out`, `thresholds.out`, `g1check_demo.out`, `readflags_demo.out`, `205_5_demo.out`, `g0_g2_demo.out`).

## Findings (by severity)

### F1 — HIGH (must fix): the fallback drops the do-nothing arm and then claims O and 205.5 are still read. Both depend on G1, which cannot pass without do-nothing rows. INTRODUCED.

`PART205-DESIGN.md:405-411`: "Every copy arm is dropped … (T, SAY copy, reverse, persona swap, random sets,
off-target, Sauers' wording)" and "The You and third-person baseline passes still run at every stage (72 per stage)".
Do-nothing is not in the parenthetical list, but 72 = 24 You + 48 baseline, i.e. no do-nothing pass is budgeted.

Measured against the real code:
- `score205.py:390-392`: `gate_g1({...})` with no do-nothing rows returns
  `{"mode":"exact","pass":false,"reason":"no do-nothing rows"}` (`g1check_demo.out`; `205_5_demo.out` reproduces the
  consequence: `205.5 → {"verdict":"not read","gate":"G0 or G1"}`).
- `score205.py:495-501`: `read_flags` requires G1 for **O** (`o = G0 and G1 and G4`) and for T. With G1 failing:
  `{"O": false, "T": false, "copy_arms": false, "random_set_rule": false}` (`readflags_demo.out`).
- The design says the same for G1: "Fails if hooks or caching alter computation; **then nothing at that stage is
  read**" (lines 174-178); 205.5 is explicitly on G0/G1 (lines 242-243). The O clause on line 239 names only G0 and
  G4, which is a PRE-EXISTING inconsistency; the scorer follows G1's "nothing is read" text (`score205.py:499`), and
  the amendment must resolve it rather than silently inherit it.

So the fallback's claims at 408-411 ("the gap is reported per stage and 205.5 is read on the gap"; "O and its
predictions (205.1, 205.2, 205.3, 205.8) … unchanged"; repeated in the commit message as "O is read as registered")
are false under the registered scorer if do-nothing is dropped: O is not read at any stage, so 205.1/205.2/205.3/205.8
all become UNTESTABLE, and 205.5 is "not read", not "read on the gap". The run would come back with almost nothing
readable — the opposite of the fallback's purpose.

Why it matters: the fallback is the likely branch (see F8), and this defect makes it silently destroy the O half of the
experiment (the half the amendment says is untouched).

Smallest fix (either, explicitly stated):
1. Keep the do-nothing arm at every stage: it is a *control*, not a copy arm. Cost becomes 24 + 48 + 48 = **120** passes
   per stage, not 72, and G1 still gates O and 205.5 as registered; or
2. State that in the fallback **no hooked arm runs at all**, so G1 is vacuous, and amend the scorer/reading rule so a
   hook-free stage is not scored as a G1 failure (and say that O then needs G0+G4, 205.5 needs G0). Item 2 must say
   which, and the pass count and the arm list must agree with it.

### F2 — HIGH for well-definedness (no prediction reads Q's copy arms, so not a data validity failure): item 2 and item 4 give two different rules for family Q in the same cell. INTRODUCED.

`PART205-DESIGN.md:405` says every copy arm is dropped "in both families" when OLMo's six-value gap fails.
`PART205-DESIGN.md:418-419` says Q's copy arms "run only if its own A0 … meets the same rule; otherwise they are
dropped". Cell *(OLMo A0 fails, Q A0 passes)* has two answers: item 2 says drop, item 4 says run. No precedence is
stated, so the experimenter chooses after seeing both A0s — exactly the hazard question 5 asks about.

Second ambiguity in the same sentence: Q's A0 is "at Qwen3.5-4B" (the chat model), but the Size section runs copy arms
at **both** Q models (`2 models x (… 48 x 36 …)`, line 203) and the config has `qwen-base` and `qwen-chat`. It is not
stated which model's A0 gates `qwen-base`'s copy arms (the chat model's?), and the threshold's SD(d)=0.079 is an
OLMo-CPU-float32 number, not a Q A770-bf16 number — the caveat at 412-413 covers only OLMo's other five values, not Q.

Why it matters: 205.8 reads O only, so no prediction status changes; but the reported Q copy numbers and the pass
budget are not determined, and the ambiguity is a post-data choice point.

Smallest fix: state precedence explicitly ("OLMo's branch governs OLMo's copy arms only; each Q model's copy arms
follow that model's own A0, and the threshold is used as a cost gate only because no prediction reads them"), or make
Q's copy arms require both A0s to pass. Also name the model for each A0.

### F3 — MEDIUM-HIGH (must fix before the branch is registered): the projection uses n = 48 pairs and a formula that treats the gap as fixed; the design says n is between 24 and 48, A0 clusters by the 24 questions, and the pilot's d and gap are strongly correlated. The 0.137 threshold sits in the optimistic half of a ~0.11–0.19 range. INTRODUCED (the n=48 choice; the variance model is pre-existing).

`PART205-DESIGN.md:402-404` uses `SE = 0.079/(sqrt(48)·gap)`; the Size section (lines 189-193) says "n between 24
(questions) and 48 (pairs, which share You runs)". The sharing is not incidental: the Method (lines 105-111) caches one
You run per question and reuses it for both names, so the two pairs of a question are not independent; A0's own SE is
specified "with the 24 questions as clusters" (line 399) — the amendment clusters for the gap but not for T.

Measured (`thresholds.out`):
```
registered formula SDd/(sqrt(n) g):   n=48 -> 0.1361    n=24 -> 0.1925
full ratio delta method, T=0.25:      n=48 -> 0.1094    n=24 -> 0.1547
full ratio delta method, T=0.50:      n=48 -> 0.0961    n=24 -> 0.1359
```
The pilot's own cluster-robust check points the same way (4 question clusters, `pilotcheck.out`):
`SE(mean d) = 0.0321` cluster-robust vs `0.0278` pair-level (`0.0786/sqrt(8)`), i.e. the pair count overstates the
information by ~15%; with 24 real clusters the pair count is not the right n.

Two corrections go in opposite directions and the amendment states neither. (i) Effective n is 24, not 48, which
raises the registered-formula threshold to 0.193. (ii) The pilot's d and gap are strongly positively correlated
(r = 0.71, cov = 0.00565, `pilotcheck.out`); a full ratio delta method (which includes Var(g) and Cov(d,g)) lowers the
SE at a positive T, giving 0.155 at n=24 and T=0.25 (0.136 at T=0.5). So 0.137 is reachable only by taking n=48 in a
formula that ignores the denominator's variance, or by evaluating at T=0.5 when the Size rule's target is 0.25. It is
not defensible as "the" 3-SE threshold without stating which n and variance model it assumes. (The covariance is
itself estimated from 8 pairs; the Fisher 95% interval for r is wide, roughly 0.02–0.94, so the design should not lean
on it silently.)

Why it matters: the amendment exists to enforce the Size rule "0.25 is at least 3 standard errors" before spending the
main run. At n=48 it will pass stages whose true precision is worse than the design's conservative end; the size of the
error depends on the correlation assumption and is at most ~sqrt(2) — the underpowered run the amendment was written
to prevent.

Smallest fix: use the registered formula at **n=24 → gap ≥ 0.19**, or state the full ratio-variance estimate (n=24,
T=0.25 → 0.155) and why the pilot's r is assumed to carry to the six-value pooled sample; alternatively, justify 48
with exchangeability evidence, which the pilot's cluster check does not provide. Name the assumption and print the
sensitivity in A0's output.

### F4 — MEDIUM: SD(d) = 0.079 is a point estimate from 8 pairs of one value, treated as known; the stated caveat does not protect the branch. INTRODUCED.

`PART205-DESIGN.md:402-404, 412-413`. SD(d)=0.079 comes from 8 pairs (PILOT.md:22; verified 0.0786 in
`pilotcheck.out`). With df=7 the 95% interval for that SD is **[0.052, 0.161]** (`thresholds.out`), and the threshold
scales linearly in SD:
```
SD(d)=0.079 -> n=48: gap >= 0.137 ; n=24: gap >= 0.194
SD(d)=0.161 -> n=48: gap >= 0.278 ; n=24: gap >= 0.394
```
The caveat at 412-413 only says the other five values are unmeasured and that the main run's own SD(d) is *reported*
beside the projection; it does not change the branch, and it does not cover (a) the 8-pair uncertainty in 0.079, (b)
between-value heterogeneity that can make the pooled six-value SD(d) larger than honesty's, or (c) family Q's SD(d),
which is a different model/device/dtype. So the branch can pass while the design's own precision rule is not met.

Smallest fix: derive the threshold from a conservative SD (the upper confidence bound, or a pre-registered inflation),
or state plainly that the gate is conditional on the pilot's point estimate and print a sensitivity row for
SD(d) ∈ {0.08, 0.12, 0.16} so the reader can see what the branch actually guarantees.

### F5 — MEDIUM: the branch's gate is G2 (which cannot bind) and omits G0 (which can). INTRODUCED.

`PART205-DESIGN.md:403-404`: "If the six-value gap is at least 0.14 **and G2 passes**". G2 is `gap >= 0.10`
(score205.py:443-447), and A0's gap is the same mean(P_you − P_third) — so gap ≥ 0.14 ⇒ G2 passes always. Measured:
`gate_g2({'gap':0.14,...}) → {'pass': True}` (`g0_g2_demo.out`). The clause is a no-op and creates false assurance
that frame quality was checked.

The gate that can fail while the gap is large is **G0** (option mass ≥0.5 and letter argmax on ≥half; score205.py:368-375).
A0 says "Nothing else is computed or read before the branch is taken" (line 400-401) and prints only the gap, its SE,
per-value gap and "every P" — not the raw option mass or `top_is_letter`, which G0 needs. Measured:
`gate_g0(broken rows) → {'pass': False, 'mean_option_mass': 0.1, 'letter_share': 0.0}` (`g0_g2_demo.out`). So a
degenerate Instruct frame can pass the branch with a large normalized gap; the main run then fails G0 at the very
stage the branch was based on, and nothing is read there.

Smallest fix: compute G0 from A0's same 72 plain passes (the runner already records `option_mass` and `top_is_letter`
per row, as the pilot files show), require "G0 passes at Instruct" in the branch, and either drop "G2 passes" or
write it as the no-op it is. Also say "at Instruct" — G2 is defined per stage and the branch does not name one.

### F6 — LOW-MEDIUM: the reason for rejecting item screening ("regression to the mean … inflates T") is wrong as applied to this design. INTRODUCED.

`PART205-DESIGN.md:414-416`. The stated mechanism assumes a noisy gap measurement that regresses on retest. Here the
gap is deterministic: float32 CPU, one model, same frame and prompts; the pilot's G1 is exact (max abs logit diff 0.0
over 22 comparisons, PILOT.md:9; `g1_pilot_check.out` re-counts the 22 rows and confirms max 0.0), and A0's items are
the *same* held-out items the main run uses. Selecting items on A0's gap therefore selects their true gap, with no regression. And in the same-sample
ratio, a denominator selected high makes T **smaller**, not larger (`d̄/ḡ`). The pilot's d and gap are in fact
strongly positively correlated (r ≈ 0.71, cov = 0.00565 over 8 pairs, `pilotcheck.out`), so selecting on the gap also
selects the copy effect; a re-measurement story does not establish the claimed inflation either (the pilot's two
quantities move together, and the correlation's own 95% interval is wide).

Why it matters: the decision not to screen items is defensible, but for a different reason — it changes the estimand
from the registered pooled T to T on a high-gap subset, shrinks n, and generalises to no fresh items (which are not
written). The text should not leave an incorrect statistical justification in a pre-registration.

Smallest fix: replace the mechanism with the estimand/no-fresh-items reason, or state precisely which measurement
supplies T's denominator and in which direction the bias runs.

### F7 — LOW: the branch uses the point estimate although A0 computes a clustered SE and prints it; near 0.14 the decision is noise. INTRODUCED.

`PART205-DESIGN.md:397-404`. A0 computes "its standard error with the 24 questions as clusters" but the branch ignores
it. The pilot's per-pair gap SD is 0.101 (PILOT.md:23); even with 24 clusters the gap SE is not negligible beside a
0.14 threshold. Pass and fail are operationally different (very different arm sets), so this is not a look-alike gate,
but the branch outcome is not robust to the measurement noise the amendment itself measures. Smallest fix: branch on
the lower confidence bound (or point estimate ≥ 0.14 + 1.96·SE) and print it; or state explicitly that the
point-estimate rule is the pre-registered one and its boundary is noisy. This is optional if F3/F4 are fixed
conservatively.

### F8 — LOW: the preamble's "It is not a ceiling" is not supported by the evidence cited, and nothing measured supports the pass branch. INTRODUCED (prose only).

`PART205-DESIGN.md:392-395`. The evidence given (P(consistent) 0.52–0.93, item 6/8 high and 5/7 low) is about *item
difficulty*; it says nothing about the You-vs-name gap for the other five values. Those claims check out numerically
(`pilotcheck.out`: item means 0.571/0.921/0.579/0.891 for items 5/6/7/8; honesty-6 You 0.842, Zara 0.925, Milo 0.917
— exactly as quoted), but they do not support a larger persona gap. Passing the 0.14 branch requires the other five
per-value gaps to average **≥ 0.167** given honesty's 0.00675 (`thresholds.out`). Nothing measured supports that, and
the amendment itself says the other five values are unmeasured. This does not break the rule (the fallback is
well-formed if F1 is fixed), but the design should not imply the pass branch is expected.

Smallest fix: say the pilot gives no evidence either way for the other five values, and that the screen exists to
decide — not that "it is not a ceiling".

### F9 — LOW: A0's failure/interruption is unmapped. INTRODUCED.

`PART205-DESIGN.md:397-401` maps every numeric outcome but not a missing one: a failed forward, a model-load failure,
or a killed A0 leaves no gap, and the text does not say "no branch is taken, the main run does not start". Given
question 1 ("can every outcome of A0 be mapped to exactly one branch?"), state it: A0 must be complete (all 72 rows)
before the branch; otherwise the state is recorded and the main run is held, with the resume rule (Amendment 1) as the
only retry path. Smallest fix: one sentence.

## Answers to the five questions

1. **Well-defined and fixed before data?** For numeric outcomes, yes: gap ≥0.14 → pass; anything else → fallback; the
   boundary is closed, and the [0.10, 0.14) range goes to the fallback. No outcome is unmapped *except* a failed or
   incomplete A0 (F9), and the family-Q rule has two answers in one cell (F2). The rule is fixed in the text before the
   six-value gap is measured; 1/6 of the gap (honesty) has already been seen in the pilot, which the amendment
   acknowledges.
2. **Threshold arithmetic?** Correct for the formula as stated: 3·0.079/(sqrt(48)·0.25) = 0.13685, so 0.137 and the
   rounded "at least 0.14" are right and slightly conservative. But the stated combination is not the 3-SE threshold of
   T under the design's own n range: at n=24 the registered formula gives 0.193, and a full ratio delta method with the
   pilot's strong d–g correlation gives 0.155 (T=0.25) — the defensible range is roughly 0.11–0.19 and 0.137 sits in
   its optimistic half (F3). The honesty-only SD(d) is defensible as the only measured value, but the caveat is not
   enough: the 8-pair interval is 0.052–0.161 and the branch treats the point estimate as known (F4).
3. **Exactly one status per prediction in the fallback?** *If* the do-nothing arm is kept (or G1 is explicitly waived):
   205.1 read (G4), 205.2 read (O at Instruct), 205.3 read, 205.4 UNTESTABLE, 205.5 read on the gap (held if <0.10,
   UNTESTABLE if ≥0.10; not read if G0 fails at 3.0T), 205.6 UNTESTABLE, 205.7 vacuous, 205.8 read, 205.9 UNTESTABLE.
   205.5 at 3.0T is readable on the gap because the 72 baseline passes still run — but not through the registered
   scorer as the fallback is written, because G1 has no do-nothing rows (F1). With F1 fixed, the 205.5 statuses match
   `score205.py:643-655` exactly (`205_5_demo.out`: gap 0.12 + T unread → UNTESTABLE "gap >= 0.10 and G3 failed").
4. **Is the regression-to-the-mean rejection correct?** No as stated (F6). The decision is still defensible for the
   estimand/no-fresh-items reason; the sign of the stated bias is backwards for a same-sample ratio, and the
   deterministic gap measurement leaves no noise to regress.
5. **Post-data choice or look-alike gate?** Yes, one: the OLMo-vs-Q precedence conflict (F2) is a genuine post-data
   choice. The branch itself is deterministic; its weakness is that the 0.14 boundary is noise-sensitive and the
   computed SE is unused (F7), and that the gate named (G2) cannot bind while the one that can (G0) is not checked
   (F5).

## What I checked and found sound

- The diff is exactly the new section: +33 lines in `PART205-DESIGN.md`, no code, tests, config or live state. There is
  no feature flag to check off, and nothing in this change runs.
- Every pilot number quoted in the amendment reproduces from `results/part205/pilot/instruct/records.jsonl`:
  gap 0.00675 ("0.0068"), SD(d) 0.0786 ("0.079"), mean d −0.0113 ("−0.011"), T's SE 4.11 ("about 4.1"), P(consistent)
  min 0.523 / max 0.925 ("0.52 to 0.93"), honesty-6 You 0.842 / Zara 0.925 / Milo 0.917, and items 6/8 high vs 5/7 low
  for P(consistent) (`pilotcheck.out`).
- A0's composition is right: 24 held-out You runs + 48 question-name third-person runs = 72 forward passes, the same
  baseline runs the copy test needs; no hooks, so no do-nothing cost is inherent to A0 itself.
- The fallback's arm dropping is consistent with 205.4/205.6/205.9 becoming UNTESTABLE and 205.7 vacuous *if* the G1
  issue is resolved; the reading rules already make a prediction whose every stage is unread UNTESTABLE, so no new
  status vocabulary is needed.
- 205.5 on the gap matches the registered rule and the scorer's UNTESTABLE branch (measured above).
- The rejection of a negated persona line is sound: it changes what the copy carries, so it is a different question.
- The 0.10/0.14 relation is safe: no A0 outcome in [0.10, 0.14) is mislabeled, and gap ≥0.14 implies G2 passes.

## Limits of this review

No model was loaded, no forward pass was run, no non-honesty P was read, and the six-value gap was not computed. The
real scorer was exercised only through its pure gate/read functions on synthetic or pilot-derived inputs; the full
suite was not run (the amendment changes no code). The pilot's cluster-robust numbers rest on 4 clusters and are
indicative, not precise.
