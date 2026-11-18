<!-- cheap-review.mjs cheap-review-part205-design-haiku-r2-2026-11-11: /home/elisepc/green/worktrees/ab-part201 origin/main..HEAD, lane haiku, df-run exit 0 -->
# Cold review: part205-design (round 2, revised design)

VERDICT: SHIP-WITH-FIXES. Fix before registration, because each fix changes a number or a reading that a registered prediction depends on: R2-1 (transfer T as a per-question ratio; HIGH), R2-2 (G4 on the mean of two Retests), R2-3 (copy-size family cannot be filled as specified; introduced by the F7 fix), R2-5 (205.1, 205.3, 205.5, 205.7 cannot fail or are undefined as written). R2-4, R2-6 to R2-16 are MEDIUM/LOW follow-ups that can go in the same revision. No finding is PRE-EXISTING; the change is a design document. No data exists yet, so none of the fixes requires a rerun.

Scope: PART205-DESIGN.md at ab-part201 HEAD (c01e28a, on 2c79c24), with PRIOR-ART.md (+14 lines).
Round-1 review: /home/elisepc/green/Journal/reviews/cheap-part205-design-ds-r1.md (F1-F18 to be checked one by one).
Rules: no model runs, no downloads, no live services, no credential files. Experiments are arithmetic and
synthetic-data scripts that implement the design's formulas.

## Findings (by severity; each gets file:line, measured trigger, INTRODUCED/PRE-EXISTING, smallest fix)

Labels: the whole design is new in this change, so every finding is INTRODUCED; a finding the revision
itself made reachable is marked "(revision)". There is no code in this change; PRE-EXISTING is not used.

### R2-1 HIGH, INTRODUCED. Transfer T is a per-question ratio averaged; G2 gates only the average gap, so the headline T is unbounded at near-zero denominators.

PART205-DESIGN.md:108-109 "Transfer T = (P_copy - P_third) / (P_you - P_third) ... per question, then averaged." G2
(line 154) checks only that the mean of P_you - P_third is at least 0.10. T is also what G3 tests (T_SAY >= 0.5),
and what 205.4 and 205.9 are scored on.

Trigger: `e1_transfer_ratio.py` (synthetic: 48 question-name pairs, gap ~ N(0.12, 0.06), copy carries 60% of the
gap, copy noise SD 0.03 - my assumption; SD 0.01 also run). Output (300 stages):
```
copy-noise SD 0.03: 300 synthetic stages
  G2 (mean gap >= 0.10) passes in 99% of them
  pairs with gap < 0.02 per stage: mean 2.3 (of 48)
  averaged per-question T: median +0.59, 5th-95th pct +0.32 .. +1.10
  ratio-of-means T (the fix): median +0.60, 5th-95th pct +0.54 .. +0.65
  stages where the two differ by > 0.1: 36%
  mean count of single questions with T outside [0,1] per stage: 7.2
  mean largest |T_i| per stage: 17.4
```
(At SD 0.01: differ by > 0.1 in 13% of stages; still 1.6 questions per stage outside [0,1].)

Why it matters: a question whose gap is 0.01 contributes a T of several units (the largest |T_i| per stage averaged 17 at noise SD 0.03); the average is then a function of
which pairs happened to have a small gap, not of the copy. G2 passes in 99% of these stages, so the gate does not
protect the headline. Fix (one line, before registration): T = mean(P_copy - P_third) / mean(P_you - P_third), a
ratio of means, or per-question T only where |P_you - P_third| >= a floor stated in advance, with the excluded
count printed. Either way, G3 must use the same statistic.

### R2-2 MEDIUM, INTRODUCED. G4 tests the MEAN of the two Retest overlaps, so it passes with one ranking unreliable; O then uses the averaged ceiling.

PART205-DESIGN.md Method, "Overlap, with a retest ceiling": "Retest = mean(|SAY_a & SAY_b|, |DO_a & DO_b|)". G4:
"Retest at least 20 of 80". Trigger: `e2_g4_mean_vs_min.py` (design's halves, top-80, Retest and Cross formulas;
pool 131,328; shared core 40; private 40 per prompt; SAY noise 0.1; DO noise varied). Output:
```
noise_SAY noise_DO | Retest_SAY Retest_DO  Retest(mean)  G4(mean)  G4(min)  Cross  O=Cross/mean
   0.1       0.6    |     40.0       0.1        20.0      pass     FAIL     2.2   0.109
   0.1       0.8    |     40.0       0.0        20.0      pass     FAIL     0.9   0.047
   0.1       1.0    |     40.0       0.0        20.0      pass     FAIL     0.5   0.024
```
Why it matters: a stage whose DO ranking is pure noise (Retest 0 of 80) passes G4 exactly at the floor, and O is then
read on a denominator half of which is not a reliability measure. Under the design's own Retest definition the
noise DO set also drags O down (0.02-0.11 here), so a "no sharing" reading could be a DO-ranking artefact. Fix: G4
requires min(Retest_SAY, Retest_DO) >= 20 (and says so); or report both and gate each.

### R2-3 MEDIUM, INTRODUCED (revision). The copy-size-matched random family (F7 fix, family iii) cannot be filled as specified: rejection sampling to a +/-10% window has no cap and no fallback, and the window is empty whenever the DO set is in the tail.

PART205-DESIGN.md Controls, "Matched random sets ... (iii) copy-size matched (drawn as in (i), accepted only if the
summed L2 norm ... is within 10% of the DO set's)". Trigger: `e3_copysize_accept.py` (synthetic per-part copy size
d = lognormal; score = d x g; DO = top-80 by score; random 80-sets uniform over the pool, 20,000 draws; layer-and-kind
matching would make acceptance rarer still):
```
sigma_d sigma_g | DO/random-mean  acceptance  accepted of 20000
  0.5    0.5    |     3.41          0.0000     0
  0.5    0.2    |     4.65          0.0000     0
  0.8    0.3    |    10.66          0.0000     0
  1.0    0.3    |    18.76          0.0000     0
  0.5    3.0    |     1.36          0.0003     7
  0.5    10.0   |     1.04          0.8447     17079
```
Why it matters: the family is what makes "beats the random 95th percentile" meaningful on copy size. Under the
stated rule the procedure either never terminates or, if someone relaxes the window silently, the controls are
matched on a number the reader never sees. The DO set is chosen for high activation x gradient, so the premise that
it sits in the tail is the expected case, not an edge case. (Acceptance is 7 of 20,000 when the score only weakly
tracks copy size (sigma_g 3.0) and 0.84 only when the score is effectively independent of it (sigma_g 10). The design's
DO score and its copy size both come from the You-run activation, so the correlation is plausible, not measured.) Fix: say which way it fails: either cap draws
(e.g. 10,000) and report "family iii not fillable at this stage" as a stated outcome that makes 205.4/205.7 unread,
or match by rank of copy size within layer-kind (nearest-neighbour sets) and print the achieved ratio per set.

### R2-4 MEDIUM, INTRODUCED. The persona-swap null does not control the You-vs-third prefix difference, so the Limit that it "bounds how much of a transfer any prefix change gives" overclaims.

PART205-DESIGN.md:108 ("asks whether a copy from another third-person prefix moves the answer as far"), Controls
(persona-swap bullet), and Limits line 237 ("the persona-swap null bounds how much of a transfer any prefix change
gives, but does not remove the difference"). Trigger: `e4_prefix_len.py` (OLMo 2 1B tokenizer from the local cache,
offline; no weights; frame "System: <persona>\nUser: <question>\nAssistant: Answer:"):
```
You   total tokens  26   prefix before question 10   line 'You care about honesty.'
Zara  total tokens  31   prefix before question 15   line 'Zara, an AI, cares about honesty.'
Milo  total tokens  30   prefix before question 14   line 'Milo, an AI, cares about honesty.'
```
The You->third copy moves 10 prefix tokens into 15 (a shift of 5, for Zara; 4 for Milo). The persona-swap null
copies Milo's run (14) into Zara's (15): a shift of 1. Every suffix position is therefore at a different absolute
(RoPE) position in the You run than in the receiving run, and the null shares the receiving run's length, so it
cannot separate "the copied values carry the You prefix's length/positions" from "they carry self-attribution". (The
round-1 suffix measurement used a different suffix definition and got 11 vs 17; the direction is the same.)

Why it matters: the copy test is the causal check; the design's own Limit says the difference is not removed, and
the Controls bullet is then read as bounding it. Fix: state that the null controls content but not length; add a
null that matches the You line's length and frame but not its content (a You-frame run with a different value,
e.g. "You care about kindness.", copied the same way), and report T for that arm beside T_swap.

### R2-5 MEDIUM, INTRODUCED. Three predictions cannot fail as written, and two reading rules leave a case undefined.

(a) **205.1** (line 192) restates G4 ("Retest at least 20 of 80 ... at every OLMo stage and both Qwen models"). The
reading rules (lines 205-219: "A stage whose gate fails is reported as 'not read' ..., not as a failed prediction") means a
G4 failure can never falsify 205.1; it can only be held or UNTESTABLE. Fix: drop 205.1 as a prediction (G4 stays a
gate), or state a failure outcome ("at least five of six OLMo stages pass G4").

(b) **205.5** (line 196): "if G0 passes: G2 fails (gap under 0.10) or T is under 0.2." When G2 fails, T is "not read"
(lines 154-155), yet the first disjunct counts the gate failure as confirmation. Under the reading rules the prediction
is either held by a stage that is not read, or it is read only where G2 passes. Fix: "if G0 and G2 pass, T is under 0.2".

(c) **205.3** (line 194, reading line 211): read "only if at least three adjacent pairs have O read at both
stages", and "holds only if the mid-training end to SFT rise is the largest among the readable pairs". If the
mid-training-end->SFT pair itself is unreadable (G4 fails at either end), the rule does not say whether 205.3 is
held, failed or UNTESTABLE. The "withdrawal" rule (line 215) has the same gap: "O at X exceeds O at the stage before"
is undefined for the first stage (pretraining, no predecessor) and when two stages both qualify. Fix: "if that pair
is not readable, 205.3 is UNTESTABLE"; name the first-stage and multiple-candidate cases.

(d) **205.7** (line 198): "DO-set copy moves the off-target self-reports by under 0.05 in probability". The
off-target questions have no value-consistent option (line 70: "Do you like the colour blue?"), so P (defined for
the consistent option, line 109) is undefined for them; the design does not say which option's probability is
measured or in which direction. The prediction is rated "high" confidence on a quantity the reader cannot compute.
Fix: define it, e.g. mean |P(A) under copy - P(A) baseline| over the six self-reports with the option order fixed.

### R2-6 LOW, INTRODUCED. The Size line-by-line recount does not match the design's totals (small, and in the conservative direction).

PART205-DESIGN.md:170-175. The copy count (line 170) multiplies 48 question-name pairs by 67 arms, and counts the You
run once per pair. Optimization (line 184) caches the You run once per question (24 per stage). The off-target
line (line 172) omits its six You runs, and names no arms beyond "4 arms". Recount (same arm list):
- OLMo, per stage: 48 (ranking) + (24 You + 48 x 66 = 3,192) + 54 (Sauers' wording) + (48 + 6 You = 54) = 3,348; design 3,366.
- Qwen, per model: 48 + (24 + 48 x 36 = 1,752) + 54 + 54 = 1,908; design 1,926 (37 arms = 7 named + 30 random, so 36 non-You arms, as I used).
- Total: 6 x 3,348 + 2 x 1,908 = 23,904; design 24,048 (difference 144).
- Pilot (line 175): 8 + (4 You + 8 pairs x 66 = 532) + 9 + 54 = 603; design 601.
The arithmetic that is stated is right (6 x 3,366 = 20,196; 2 x 1,926 = 3,852; 20,196 + 3,852 = 24,048; 48 x 67 = 3,216;
54 = 6 + 6x2x4; 48 = 6x2x4; 601 = 8 + 536 + 9 + 48). It is the unit (pair vs question) that is wrong. Fix: count You
runs per question, add the off-target You runs and name its four arms.

### R2-7 LOW, INTRODUCED. The "about four standard errors" claim treats 48 question-name pairs as independent; they share 24 questions.

PART205-DESIGN.md:163-165. Stated: per-question SD near 0.4, "a standard error near 0.06". Recount: 0.4/sqrt(48) =
0.058. The two names share a question and its You run, so the effective number of units is between 24 and 48:
0.4/sqrt(24) = 0.082. So 0.25 is 3.1 to 4.3 standard errors, not "about four", and the test that matters is 0.25
against the random 95th percentile, whose own error (20 random sets) is not computed. Fix: cluster the SE by question
and state the SE of the margin over the random 95th percentile.

### R2-8 LOW, INTRODUCED. No seeds are named for the random families, the bootstrap, or prompt order.

Grep for "seed", "rng", "random state" in PART205-DESIGN.md returns nothing. Random families (lines 124-130),
the 1,000-resample bootstrap (line 97) and the copy-size rejection step all draw random numbers. A restart after a
partial stage would redraw different sets, so O's intervals and the random 95th percentile would move between runs.
Fix: name one seed per family and per stage in `part205/config`, and store each stage's sets once before any copy run.

### R2-9 LOW, INTRODUCED. The sign convention for "pushes toward the consistent option" is not stated.

Line 145: "the share of each set whose mean signed e pushes toward the consistent option." With e = activation x
gradient of the logit difference (consistent minus other, Ranking, lines 84-86), zeroing a part changes that difference by about
-e, so a positive e means the part supports the consistent option only if the convention is stated. Fix: one sentence:
"positive e: zeroing lowers the logit difference, i.e. the part supports the consistent option."

### R2-10 LOW, INTRODUCED (revision). F10 response checked: correct on the facts, and the suggested cross-check was not added.

I checked the revision's F10 claim against the repository. `third_party/honesty-heads/README.md:1-3` reads "Honesty
heads in Qwen3.5-9B / Author: Claude Opus 5.5", and `LICENSE` is "Copyright (c) 2026 SauersML". So the vendored code is
a different project from the say/do post's code, and the revision's correction ("not this code") is right; round 1's
claim that it was Sauers' own say/do code was not. The "as Sauers did" wording is gone (only "as Sauers described it"
remains, line 229). Not done: round 1's suggested Jaccard check between the design's estimator and the vendored
integrated-gradients estimator on one stage. The Limit (line 229) states the estimator difference; that is adequate
for registration, and the check is a follow-up.

### R2-11 LOW, INTRODUCED (unverified here). G1 needs max absolute difference exactly 0, on the A770 in bf16 as well as the CPU.

PART205-DESIGN.md:152. If the A770 kernels are not bit-deterministic, G1 fails for a benign reason and voids the whole
Qwen stage (G1: "nothing at that stage is read"). I could not run it (no model may be run). Fix: the pilot should
run the do-nothing hook twice on the A770 and record whether two plain runs are bit-identical; if not, G1 needs a
stated tolerance and the reason.

### R2-12 LOW, INTRODUCED. The load class is sound on transformers 5.20, but the design does not pin the library version.

Line 54: text-only `AutoModelForCausalLM` gives `Qwen3_5ForCausalLM`. The cached config is
`Qwen3_5ForConditionalGeneration` (`models--Qwen--Qwen3.5-4B/snapshots/*/config.json`). In the local venv's
transformers 5.20.0 `modeling_auto.py:873` maps `qwen3_5` to `Qwen3_5ForCausalLM` ("VLM compatibility"), so the claim
holds for that version. The architecture numbers the Size line uses are in the cached config and match (see the Size
section). Fix: pin the transformers version beside the revisions (Settings matched across arms, line 140).

### R2-13 LOW, INTRODUCED. Family (iii) matches copy size on activation norm; PROCEDURE §2 says match the size by effect on behaviour (Part 108). (Reasoned, not measured.)

PROCEDURE.md §2 ("match the size of a change by its effect on behaviour (Part 108)") and the revised F7 fix (family iii,
matched on the summed L2 of the You-minus-third difference). The norm is the nuisance the F7 round-1 finding named,
but it is not the quantity that moves the answer: the first-order effect of a copy on the logit difference is
sum over copied parts of g x (a_you - a_third) at the readout position. Two sets with equal norm can differ in that
effect by any amount, and the DO set's effect is larger than a random set's by selection: it was chosen for large
a x g in the You run, and the copy writes that same You-run value. The score-matched family (ii) partly covers this
(it matches on the You-run score), but nothing matches on the copy's own first-order effect. Fix: match family (iii)
on the copy's first-order effect at the readout position. This needs one backward pass per third-person run (48 per
stage, which the Size line does not count; R2-6), not new forward passes.

### R2-14 LOW, INTRODUCED. The answer-polarity split is only half reported (STANDARD-CONTROLS item 5).

STANDARD-CONTROLS.md item 5: "Any yes/no or letter readout reports the effect toward the right answer and the effect
toward one answer label separately." The design balances the letter order (Method, "Option order is balanced") and
reports T and O against the consistent option, but no arm reports the effect toward label A and toward label B
separately, so a result carried by "always A" or "always B" would be averaged into the consistent-option number. The
design's Not-used paragraph does not mention item 5. Fix: report P(A) and P(B) per arm (or the split by label) beside T,
and say so in the Controls list.

### R2-15 LOW, INTRODUCED. The threshold "above the 95th percentile of all three" has two readings; measured false-positive rates differ, neither is far off.

PART205-DESIGN.md:124-129 and the 205.4 wording. `e5_threshold_reading.py` (synthetic: DO set and random sets
exchangeable; 20,000 trials):
```
  pooled reading (above 95th pct of the 60):          0.0626
  per-family reading (above 95th pct of each family): 0.0432
```
Both are near the nominal 5%, so this is a reading-rule defect, not a large error. Fix: name one reading (the
per-family one is the one 205.7's "beats all three" implies) and state the percentile method.

### R2-16 LOW, INTRODUCED. Two prior-art claims go beyond the recorded abstract.

PART205-DESIGN.md:32 ("the two channels are steered by near-orthogonal directions") and :35 ("sycophancy, factual lying
and instructed lying") are not in the PRIOR-ART.md entries (lines 842-852), which say "abstract read". The 2610.04125
entry does not mention orthogonality; the 2604.19117 entry does not mention factual or instructed lying (its title
says "Sycophancy-Lying"). I could not re-read the abstracts (no network access in this review). Fix: either record the
phrases from the abstract text in PRIOR-ART or drop them from the design's prior-art paragraph.

### Checked, not a finding: prior-art numbers that are recorded match the design: nine methods and four models
(2610.04125); twelve models and about tenfold RLHF reduction (2604.19117); 0.22% of OLMo-3 pretraining (2605.13329,
PRIOR-ART.md:573-574).

## Round-1 findings (F1-F18): fixed, partly fixed, or not

Checked against the revised text (line numbers in the design) and the round-1 review (`reviews/cheap-part205-design-ds-r1.md`).

| Round-1 | Status now | Evidence |
| --- | --- | --- |
| F1 Qwen head units | FIXED | The Units paragraph (line 70-ish) defines both head kinds; cached config matches 8 full-attn layers x16 heads, 24 DeltaNet x32 heads (R2-12). |
| F2 `main` not cached | FIXED in substance | Download dropped; the 1B card (archived) says the 1B trained one ingredient on the 50B mix, not merged. Identity of `main` with the ingredient-3 run is an inference from the card. |
| F3 O spread near 20 | FIXED | Paired bootstrap, 205.3 and withdrawal need the interval to exclude 0. Still open: the rise's interval method (paired vs independent resampling) is not stated (folded into R2-8 seeds point). |
| F4 different prefix | PARTLY | "Last position" stated; persona-swap null added; round-1's content-vs-self-attribution arm (a different question or value) not added. R2-4. |
| F5 frame may not elicit base answers | FIXED | G0 added; 205.5 conditional on G0. R2-5(b) remains for 205.5. |
| F6 pilot count | FIXED, recount differs | 601 stated; my recount 603 (R2-6). |
| F7 copy-size | PARTLY, introduces a defect | Family (iii) added, but cannot be filled as specified (R2-3) and matched on norm, not effect (R2-13). |
| F8 205.7 vacuous; G3 on copy arms | FIXED | Line 213: 205.7 vacuous where DO transfer not read; G3 gates every copy arm (line 156-157). |
| F9 205.3 one pair | PARTLY | "At least three readable pairs" added; the case where the mid-training-end->SFT pair itself is unreadable is undefined (R2-5c). |
| F10 estimator / "as Sauers did" | FIXED (correction verified) | Vendored code is Claude-authored (`third_party/honesty-heads/README.md`), not the say/do code; the phrase is gone. Cross-check (Jaccard) not added; Limit stated. R2-10. |
| F11 heads vs neurons scale | FIXED | Signed sum over a part's dimensions is the first-order logit change for zeroing that part; sound. |
| F12 six values pooled | FIXED | Per-prompt normalisation and per-value shares. Sound. |
| F13 poles | ADDRESSED | Written reason given (Not used); reason is sound for the copy test. Item 5 (label split) is a separate gap: R2-14. |
| F14 wording differs | FIXED | Copy holds the question fixed; Sauers' wording is a secondary arm with the wording change in its denominator. |
| F15 which Qwen | FIXED | 205.8 reads the chat model; base reported unscored. |
| F16 DO 5-8 unused | FIXED | Split sentence corrected; Limit added. |
| F17 load class / RAM | FIXED for class (R2-12); RAM "11 GB" not checked here | |
| F18 PRIOR-ART | FIXED for the two papers; some phrases not recorded (R2-16) | |

Round-1 said "the gates are mostly real"; in this round the gates G0, G1, G3 are real and each has a "fails if" line.
G2 and G4 are real but are the weak points (R2-1, R2-2).

## Size arithmetic, line by line

| Line (design) | Stated | Recomputed | Verdict |
| --- | --- | --- | --- |
| Pool, OLMo (Units, line 76) | 131,328 = 256 heads + 131,072 neurons | 16 x 16 heads = 256; 16 x 8,192 = 131,072 | OK; config-verified (16 layers, 16 heads, hidden 2048, intermediate 8,192; all six revisions) |
| Pool, Qwen (Units) | 896 heads; 294,912 neurons; 295,808 | 8 x 16 = 128; 24 x 32 = 768; 128 + 768 = 896; 32 x 9,216 = 294,912; 896 + 294,912 = 295,808 | OK; config-verified (`full_attention_interval` 4; `num_attention_heads` 16, `head_dim` 256; `linear_num_value_heads` 32, `linear_value_head_dim` 128) |
| Ranking per stage (line 168) | 6 x 8 = 48 fwd+bwd | 48 | OK |
| Copy test arms per pair (line 170) | 67 | 1+1+1+1+1+1+1+60 = 67 | OK; but the You run is counted per pair (48), cached per question (24): R2-6 |
| Copy test per stage (line 170) | 48 x 67 = 3,216 | 3,216 | arithmetic OK; unit wrong (R2-6) |
| Sauers' wording (line 172) | 6 + 6x2x4 = 54 | 54 | OK |
| Off-target (line 172) | 6 x 2 x 4 = 48 | 48; six You runs not counted; arms not named | R2-6 |
| OLMo per stage (line 173) | 48+3,216+54+48 = 3,366 | 3,366 | OK as stated; my unit-corrected count 3,348 |
| OLMo total (line 173) | 6 x 3,366 = 20,196 | 20,196 | OK as stated |
| Qwen per model (line 174) | 48 + 48x37 + 54 + 48 = 1,926 | 1,926; 37 = 7 named + 30 random (10 x 3) | OK as stated; my count 1,908 |
| Qwen total (line 174) | 2 x 1,926 = 3,852 | 3,852 | OK as stated |
| Total (line 175) | 24,048 | 20,196 + 3,852 = 24,048 | OK as stated; my corrected total 23,904 (R2-6) |
| Pilot (line 175) | 8 + 8x67 + 9 + 48 = 601 | 601 | OK as stated; my corrected count 603 |
| Smallest effect (line 163-165) | SE 0.06; "about four" SE | 0.4/sqrt(48) = 0.058; 0.4/sqrt(24) = 0.082; 0.25/0.058 = 4.3; 0.25/0.082 = 3.1 | R2-7 |
| Q RAM (line 183) | 11 GB | not checkable here | unverified |

Bottom line of the arithmetic: no stated product is wrong; the unit of count (pair vs question) and the omitted
off-target You runs are, and the design over-counts the total by 144 passes. Nothing here is a schedule risk.

## What I checked and found sound

- Model pins: all six OLMo stage revisions (22738a97, 1c3599f7, a1847dff, 0d85a3d0, c4b04859, 48d788ec) and both Qwen
  revisions (1001bb4d, 851bf6e8) are present in the local cache at the hashes the design cites. F2 is fixed in
  substance: `main` (a1847dff) is cached, so the download step was correctly dropped.
- Architecture numbers: OLMo 1B and Qwen3.5-4B configs match the Units and Size lines (above).
- The head-site rule the design follows exists as cited (`third_party/honesty-heads/code/audit_eval.py`, head_site);
  its numbers match the Qwen config (full-attention head width 256 x 16 heads; DeltaNet width 128 x 32 value heads).
- The shared suffix is well defined: question and answer cue are identical text; aligned from the end, the last
  position is inside it (round-1 measurement, consistent with R2-4).
- Gate structure: G0, G1, G3 each state how they could fail and what is then not read; G1 is a true bit-for-bit
  do-nothing check (subject to R2-11); G3 gates every copy arm (F8 fixed).
- The per-prompt normalisation (F12) and the signed sum over a part's dimensions (F11) are the right first-order
  quantities; I checked that sum over a head's dimensions of activation x gradient is the first-order change from
  zeroing the head's output, so heads and neurons share a unit.
- Predictions that can fail: 205.2, 205.4, 205.6, 205.8, 205.9 have thresholds and a stage at which they can fail.
- Withdrawal rule: exists, is stated before data, and forbids the headline claim without the bootstrap interval.
- Pilot stopping rule (G1 failure, or a pass slower than 1 s) is stated before any run.
- Nothing in the design references a live service, a credential, or a download step; the revision drops the download.
