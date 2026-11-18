# Review: PART205-DESIGN.md (branch part205, 2c79c24, draft not registered)

VERDICT: **SHIP-WITH-FIXES.** The question is good, the gates are mostly real, and the Size arithmetic is right
except the pilot. But three things must be fixed before registration because they make a headline number undefined
or confounded: F1 (Qwen3.5-4B has no attention heads on 24 of 32 layers, so family Q's unit is undefined as
written), F10 (the design says it ranks "as Sauers did" but Sauers' own code in this repository ranks by integrated
gradients / attribution patching, not single-point activation x gradient), and F14 (the You and third-person frames
differ in the question wording as well as the persona, so G2 and T mix a question change into a persona effect).
F4 (copied values were computed against a different prefix) and F11/F12 (heads and neurons on one unnormalised scale;
six values pooled without per-value normalisation) are the next tier: each can move O or T for a reason unrelated to
the question. F5 (the plain frame may not elicit base answers, and 205.5 cannot tell "no self-attribution" from "no
answer") and F8 (205.7 vacuous at five stages; 205.6 read when its positive control fails) are gate and prediction
defects. The rest are minor but cheap to fix.

Reviewer: cold, independent. Read `git -C /home/elisepc/green/worktrees/ab-part201 diff origin/main..HEAD` (one file,
PART205-DESIGN.md, +176). Checked against PROCEDURE.md §1-2, STANDARD-CONTROLS.md, and repository code/data read-only.
No model was run and nothing was downloaded. Scripts in this directory: `o_index.py`, `o_index2.py`, `o_index3.py`
(the O index), `suffix_check.py`, `suffix_positions.py` (the real OLMo tokenizer).

## Findings, by severity

HIGH: F1. MEDIUM: F2, F4, F5, F7, F8, F10, F11, F12, F14. LOW: F3, F6, F9, F13, F15, F16, F17, F18.
All are INTRODUCED (the change is a new design doc; every finding is in it). No PRE-EXISTING findings: the change
touches no code.

## Findings

### F1 — HIGH, INTRODUCED. Family Q's "attention head" unit is undefined on Qwen3.5-4B: 24 of its 32 layers have no attention heads.

PART205-DESIGN.md:67-70 defines a part as "one attention head (the head's slice of the `o_proj` input) ... heads and
neurons ranked together". PART205-DESIGN.md:48-49 says family Q runs "Qwen3.5-4B-Base and Qwen3.5-4B ... bf16 on the
A770 with eager attention".

Measured, from the cached config the design names (`/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e8.../config.json`;
the Base snapshot `1001bb4d...` has the same shape):

```
$ python3 -c "import json;c=json.load(open('.../Qwen3.5-4B/snapshots/851bf6e8.../config.json'))['text_config'];print(c['num_hidden_layers'], c['layer_types'].count('full_attention'), c['layer_types'].count('linear_attention'), c['head_dim'], c['num_attention_heads'])"
32 8 24 256 16
```

Every fourth layer (`full_attention_interval: 4`) is gated softmax attention with 16 heads of head_dim 256 (`o_proj`
input 4096); the other 24 are Gated DeltaNet (`linear_attention`) with `out_proj` over 32 value heads of 128. There is
no `self_attn.o_proj` on those 24 layers at all. The design's unit definition therefore names a module that does not
exist on three quarters of Qwen's layers, and it never says whether the DeltaNet value heads count as "heads" (a
different width, a different count, a different mechanism) or are skipped. The repository's own code already draws
this line: `part82/part82.py:40` asserts `self_attn` and `third_party/honesty-heads/code/audit_eval.py:89-97`
(`head_site`) returns `linear_attn.out_proj` with `linear_value_head_dim`/`linear_num_value_heads` for DeltaNet layers
and `self_attn.o_proj` with `head_dim`/`num_attention_heads` for the others. The design contradicts code the project
already has.

Why it matters: O pools SAY and DO into one top-80 set of heads and neurons. On OLMo the head pool is 16x16 = 256
heads; on Qwen, if only full-attention layers count, it is 8x16 = 128, and if DeltaNet heads count too it is
24x32 + 8x16 = 896 units of a different kind. The top-80 set's composition (and so the overlap and its retest ceiling)
depends on that choice, and 205.8 ("Qwen3.5-4B's O is at least 0.2 below OLMo 2 1B Instruct's") compares two
differently-sized pools. The design never states the Qwen head count, so the comparison is not defined as written.

Smallest fix: state in the method which Qwen layers contribute heads and with what head width and count (the repo's
`head_site` rule is the obvious one), say how many heads and neurons each family's pool holds, and say whether the
DeltaNet heads are in or out; if they are out, say whether O is still comparable across families and scope 205.8
accordingly.

### F2 — MEDIUM, INTRODUCED. The design says `a1847dff` ("mid-training end / main") is not cached; it is cached, and it is `refs/main`.

PART205-DESIGN.md:46-47: "mid-training end (`main`, a1847dff; not cached now, downloaded at night before the run)".

```
$ ls /mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B/snapshots/
1c3599f7... 22738a97... 22f7d18a... 3e09b708... 9d3e4365... a1847dff... a7a789f6... eb19ccab... eda09f03... f83cd641...
$ cat .../models--allenai--OLMo-2-0425-1B/refs/main
a1847dff35000b4271fa70afc5db10fd29fedbdf
$ for f in .../snapshots/a1847dff.../*; do [ -e "$f" ] && stat -L -c '%s' "$f" || echo DANGLING; done
623 121 916646 4983360992 956326560 14873 125 7137177 4345 1611056   # all present, both shards full size
```

Why it matters: the design's step "downloaded at night before the run" is a no-op that someone will schedule, and
the claim that the stage needs a download is false; more importantly it means the design has not checked what its
"mid-training end" checkpoint actually is. `refs/main` being `a1847dff` is the HF *default* branch of the base repo,
not a named stage. The design calls it "mid-training end" with no evidence: `PART85-DESIGN.md:37-41` and
`part85/checkpoints.py:26-29` list only `pretrain_3T` (22738a97) and `midtrain_26B` (1c3599f7) as the two base
stages, and `PART37-DESIGN.md:126` calls `a1847dff` "the only cached base snapshot ... `main`". Nothing in the repo
identifies `a1847dff` as the end of mid-training.

Smallest fix: drop the download step; and either cite where `a1847dff` is established as mid-training's end (a config
or card field in the snapshot, or a prior part), or name the stage by what it verifiably is ("base repo `main`") and
let 205.3 be read against that.

### F4 — MEDIUM, INTRODUCED. The copy test's "shared suffix" is well defined token-for-token, but the design's claim that the suffix positions are "identical tokens in both frames" hides that they attend to different prefixes, and the readout position is inside the overwritten region.

PART205-DESIGN.md:78-81: "re-run it with a set of parts' activations replaced by the You run's, at the positions of
the shared suffix (the question and answer cue, identical tokens in both frames, aligned from the end)."

Measured with the cached OLMo tokenizer (`suffix_positions.py` in this directory; no model, no forward pass):

```
You len=26  third len=32  shared suffix=15
suffix starts at You index 11, third index 17
last position is You index 25, third index 31  -> inside suffix: True / True
prefix lengths before the suffix: You 11 tokens, third 17 tokens
Every suffix position in the third frame attends to 17 prefix tokens; in the You frame to 11.
```

So the suffix is real and correctly aligned from the end; the copy is well defined as a *positional* operation. What
the design does not say is what the copied values mean. Each cached You activation at suffix position p was computed
while attending only to the You frame's 11 prefix tokens; it is written into a third-frame position that attends to 17.
For the 15 suffix positions, the You values are therefore not "what the You run would compute in this context" — they
are the You run's values from a shorter, different context. The design's phrase "identical tokens in both frames" is
true of the tokens and false of the computation that produced the values. This matters because the copy test is the
causal check on the ranked sets (PART205-DESIGN.md:172-173), and a positive T could come from the copied values
carrying the You prefix's content into the third frame rather than from the parts carrying self-attribution.

The design also does not say what happens at the last position: it is inside the suffix in both frames, so it is
overwritten, and the readout is taken there. That is a deliberate choice (the last position is where the answer is
read) but it means the readout position's own activation is entirely the You run's, computed against the You prefix.
A do-nothing hook (G1) cannot catch this: it copies the third frame's own values, so it reproduces the plain logits
bit for bit while the real copy arm is still copying values from a different context.

Smallest fix: say explicitly that the copied values are the You run's values at those positions (computed against the
You prefix), and add a control that separates "the values carry content" from "the values carry self-attribution":
for example copy the You run's suffix activations from a *different question* (same value, different paraphrase) or
from an off-target prompt, and report T for that arm. If that arm also closes the gap, the copy test is measuring
content transfer, not the parts' self-attribution. Also state whether the last position is overwritten and why.

### F5 — MEDIUM, INTRODUCED. The plain frame's ability to elicit base-checkpoint answers is asserted, not checked, and the design's own prediction 205.5 depends on it failing in a way the gates cannot distinguish.

PART205-DESIGN.md:51-54: "The same plain frame at every stage, so base and chat checkpoints read the same text ...
No chat template (a template exists only for the chat checkpoints; the frame difference would be confounded with
training)." PART205-DESIGN.md:147: "205.5 | At the 3.0T pretraining checkpoint, G2 fails (gap under 0.10) or T is
under 0.2."

The design never shows that a 3.0T pretraining checkpoint answers "A) yes B) no" at all in `System: ...\nUser: ...\n
Assistant: Answer:`. If it emits a continuation that is not an option letter, the two option letters' masses are
small and the "broken answers" count (PART205-DESIGN.md:98-99) will be high — but the design says such rows "stay in
the means". So a base checkpoint that cannot read the frame produces P_you and P_third near each other, G2 fails, and
205.5 "passes" by the frame not working, not by the model not treating itself differently. The two explanations are
not separated anywhere in the design. This is a prediction that can pass for the wrong reason, which is exactly the
kind of thing PROCEDURE.md section 2 asks a design to rule out ("Name a gate that must pass before the main result is
read, and how it could fail").

Also, the design says "Greedy is not needed: every readout is the next-token probability of the two option letters,
normalised over the two, with both masses printed" (PART205-DESIGN.md:53-55). Normalising over the two letters means a
base checkpoint that puts 0.001 on each letter still gives a well-defined P; the "both masses printed" is the only
guard, and the design does not say what mass is too low to read. The broken-answer rule ("option mass under 0.5 of
the total next-token mass") is a start, but it is not applied as a gate: rows stay in the means.

Smallest fix: add a gate G0 before G2 — the two option letters carry at least some stated fraction (say 0.5) of the
next-token mass in the plain frame at that stage, and the stage's top continuation is an option letter on at least
half the prompts — and say that a stage failing G0 is reported as "the frame does not elicit an answer", not as
205.5 passing. That separates "the base model does not treat itself differently" from "the base model cannot read
this frame".

### F6 — LOW, INTRODUCED. The Size arithmetic is right except the pilot's "230 passes", which cannot be derived from the design's own per-stage counts.

PART205-DESIGN.md:116-129. I recomputed every line (`python3` one-liner in this review's log):

```
Ranking per stage: 6 x 8 = 48            design 48          OK
plus 24 for halves                        design 24          OK (but see below)
Copy test: 48 pairs x 46 arms = 2208      design 2,208       OK (46 = 6 named + 40 random)
Off-target: 6 x 2 x 4 = 48                design 48          OK
Family O: 6 x (48 + 2208 + 48) = 13,824   design 13,824      OK
Family Q: 2 x (48 + 48 x 26 + 48) = 2,688 design 2,688       OK (26 = 6 named + 20 random, 10 per family)
Total: 13,824 + 2,688 = 16,512            design 16,512      OK
```

The pilot line does not reconcile: "Instruct only, honesty only, all arms (230 passes)" (PART205-DESIGN.md:127-128).
Honesty only is 1 value x 4 held-out paraphrases x 2 names = 8 question-name pairs; at 46 arms that is 368 copy
passes plus 8 ranking passes = 376. 230 is not reachable from 8 pairs and 46 arms; it equals 5 x 46, or 6 x 37 + 8.
So either the pilot is fewer pairs than the design's split gives, or fewer arms, or the number was typed. A pilot
whose size is wrong by 60% is a pilot whose stopping rule ("if a pass takes over 1 s on the CPU, stop") fires at the
wrong wall time.

Also, the ranking line says "plus 24 for the halves' separate scores (the halves reuse the same passes; no extra
runs)" — if the halves reuse the same passes, there are no extra runs, and the 24 is a count of *scores* not passes,
so the "+24" in the total is double-counted. The total 13,824 = 6 x (48 + 2208 + 48) does not include the 24, so the
number is right but the sentence contradicts itself. That is a wording defect with a real consequence for anyone
re-deriving the budget.

Smallest fix: recompute the pilot from the split (8 pairs x 46 arms + 8 ranking = 376) or state which subset of
pairs and arms gives 230; and reword the halves line to "the halves reuse the same passes; 24 extra scores, no extra
passes".

### F7 — MEDIUM, INTRODUCED. The random-set controls are matched on layer-and-kind and on score decile, but not on the thing that makes a copy transfer: the You/third-person activation difference at the suffix positions.

PART205-DESIGN.md:91-93: "Matched random sets, two families, 20 sets each (10 for family Q): (i) layer-and-kind
matched (same number of heads and of neurons in each layer as the DO set); (ii) score-matched (parts drawn from the
same score decile in the selection run as the DO parts, excluding SAY and DO parts). A transfer counts only above the
95th percentile of both."

PROCEDURE.md section 2 is explicit about this: "Match the control on everything except the property under test ...
A random same-layer unit can be nearly silent, so zeroing needs an activity-matched null (Part 67, toolkit I). A
scramble matched on weight distance changed far fewer answers, so match the size of a change by its effect on
behaviour (Part 108)."

The copy arm writes the You run's activations into the third frame. The quantity that moves the answer is the
difference `a_you - a_third` at each copied position, not the score, not the layer, not the kind. A random set of
parts matched on layer and kind can have a much smaller (or larger) `a_you - a_third` than the DO set, because the
DO set was chosen for a large activation x gradient. The score-matched family is closer (score is activation x
gradient, so a high-score part tends to have a large activation), but score is the *selection* run's You-frame score,
not the You-minus-third difference at the copy positions. So an arm can beat both random families simply by having a
larger average `|a_you - a_third|`, which is the property under test's mechanism, not a nuisance.

This is exactly the Part 108 lesson ("match the size of a change by its effect on behaviour"), and the design cites
Part 108 as prior art (PART205-DESIGN.md:20-21) without applying its control lesson.

Smallest fix: add a third random family matched on the mean `|a_you - a_third|` at the copied positions (or on the
total L2 of the copied difference), and require the DO arm to beat the 95th percentile of all three. Alternatively,
report the DO arm's average `|a_you - a_third|` beside each random family's, and say the comparison is only read when
they are within a stated factor.

### F8 — MEDIUM, INTRODUCED. 205.7 can pass vacuously at five of six stages, and 205.6 is read even when its own positive control (G3) fails.

PART205-DESIGN.md:149: "205.7 | DO-set copy moves the off-target self-reports by under 0.05 in probability at every
stage. | high". The off-target check is "DO-set copy on the six value-free self-reports" (PART205-DESIGN.md:97). A
stage where the DO set has no effect on the target question (T near 0) necessarily has no effect on the off-target
questions either, so 205.7 passes. 205.4 (the DO set actually transfers at Instruct) is checked at Instruct only. At
the other five OLMo stages, 205.7 can pass because the copy does nothing, and the design reports it as a held
prediction. That is a prediction that cannot fail for the reason it is meant to test (specificity) at most stages.

PART205-DESIGN.md:148: "205.6 | Reverse DO copy at OLMo Instruct lowers P_you by at least half of the gap." The
reading rules (PART205-DESIGN.md:157-160) gate T on G1-G3: "a stage whose gate fails is reported as 'not read'".
G3 is the SAY-set positive control: "then T is not read at that stage" (PART205-DESIGN.md:111-112). The reverse arm is
not T, and the design never says the reverse arm is gated on G3. So if the copy mechanism cannot carry the
You/third-person difference at all (G3 fails), 205.6 is still read — on a rig whose own positive control says the
copy does not work. That is exactly the Part 66 lesson PROCEDURE.md cites ("Part 66's positive control failed and
stopped the reading until the rig was checked").

Smallest fix: make 205.7 conditional on 205.4-type transfer at the same stage (report it as "specificity given
transfer", and mark it vacuous where T is not read); and extend the G3 gate to every copy arm (DO, SAY, reverse,
random) at that stage, not only to T.

### F9 — LOW, INTRODUCED. 205.3's "largest rise among the readable pairs" has no minimum number of readable pairs, so it can be decided by one pair.

PART205-DESIGN.md:145 and :161-162: "205.3 needs O read at both stages of a pair to count that pair; it holds only if
the mid-training end to SFT rise is the largest among the readable pairs and is at least 0.2."

Six stages give five adjacent pairs. If G4 fails at four of them and passes only at mid-training end and SFT, that one
pair is trivially the largest, and 205.3 reduces to "the rise is at least 0.2" — the competitor stages are gone. The
design does not require, say, three readable pairs for 205.3 to be read. This is not a prediction that cannot fail
(the 0.2 clause still bites), but it is one that can pass with far less evidence than the wording implies.

Smallest fix: state a minimum (for example, at least three readable pairs, including at least one other pair that
rises), or report 205.3 as "not read" when fewer than three pairs are readable.

### F10 — MEDIUM, INTRODUCED. The design says it ranks "as Sauers did", but Sauers' code in this repository ranks by integrated gradients / attribution patching, not by a single-point activation x gradient.

PART205-DESIGN.md:67-70: "Score per part = |activation x gradient| at the last position, of the logit difference
... The top 80 parts by score form a set, heads and neurons ranked together as Sauers did." PART205-DESIGN.md:10-12
describes Sauers as having "ranked heads and MLP neurons by activation x gradient".

The repository vendors Sauers' own code (`third_party/honesty-heads/`, `LICENSE`: "Copyright (c) 2026 SauersML").
Its ranking is stated in `third_party/honesty-heads/code/corefind.py:1-15`:

```
Ordering (per pair, both on the span; scores normalized by the gap, positive = moves the deceptive run toward honest):
  ig    integrated gradients along the straight path from the deceptive to the honest span head outputs ...
  atp   attribution patching: grad . (hon - dec) at the deceptive run (atpN, estimates N of the unit alone) and at the
        honest run (atpR, estimates R); the order uses (atpN + atpR) / 2.
```

and `third_party/honesty-heads/README.md:17`: "Every head ranked by integrated gradients". The `attribute()` function
(`corefind.py:189-236`) takes the trapezoid over a path `al = linspace(0, 1, steps+1)` between the deceptive and
honest captures, and normalises by `gap = pr.hon.m - pr.dec.m`. There is no single-point activation x gradient
anywhere in it. So the design's "as Sauers did" is not what Sauers' code does; the design's single-point score is a
different estimator, and the design's own Limits (PART205-DESIGN.md:172-173) concede that "activation x gradient is a
first-order estimate of each part's effect" — which is true of the design's estimator, not of the one in the vendored
code.

Why it matters: the design's question is whether Sauers' say/do overlap survives a retest ceiling. If the ranking
estimator differs from Sauers', a low O could be a property of the estimator rather than of the model, and the design
cannot tell the two apart. The design also claims the ranking is "at the last position" (one position), whereas
Sauers' code scores over the shared span (`pb < P`), which is the same set of positions the copy test uses.

Smallest fix: either adopt the vendored code's estimator (integrated gradients or attribution patching, over the
span, normalised by the gap) and say so, or state explicitly that this part uses a different, cheaper estimator and
that a difference from Sauers' overlap may come from the estimator; add a small check (one stage, one value) that
the top-80 from the design's estimator agrees with the top-80 from the vendored estimator (e.g. Jaccard), so the
comparison to Sauers is scoped.

### F11 — MEDIUM, INTRODUCED. Heads and neurons are ranked on one score scale without any stated normalisation, and a head's score is a sum over 128 dimensions against a neuron's single dimension.

PART205-DESIGN.md:67-70: "Score per part = |activation x gradient| ... summed over a head's dimensions, averaged over
the prompts ranked. The top 80 parts by score form a set, heads and neurons ranked together as Sauers did."

Measured from the cached OLMo config (`num_attention_heads: 16`, `hidden_size: 2048`, `intermediate_size: 8192`):

```
head score = sum of 2048//16 = 128 |a*g| terms; neuron score = 1 term. Ranked on one scale.
```

If the per-dimension |a*g| is comparable between the two sites, a head's score is on average ~128x a neuron's, and
the top-80 is almost all heads. The two sites also have different activation scales: a head's input is the
attention output (residual-stream scale), a neuron's is the SwiGLU hidden (8,192 wide, often larger magnitudes). The
design states no normalisation (no z-score, no division by a site's median, no rank-then-combine). "As Sauers did"
does not settle it: the vendored code normalises by the gap and uses a path integral, and its head score is a
*sum over the span* of per-head increments, where a neuron's is its own increments sum, so the two are on different
scales there too — but Sauers' design chose its own span and reported the head/neuron composition.

Why it matters: O pools heads and neurons into one top-80 set. If the set is nearly all heads at every stage and in
every family, the "heads and neurons ranked together" claim is nominal, the overlap is an overlap of heads only, and
the design's reading of "parts" is narrower than stated. The design never reports the head/neuron composition of the
top-80, so this cannot be checked after the fact.

Smallest fix: state the normalisation that puts heads and neurons on one scale (for example z-score each site's
scores across parts within a stage, or rank each site separately and interleave), and report the head/neuron
composition of every top-80 set beside O. If the composition is all-heads, say so and scope the claim to heads.

### F12 — MEDIUM, INTRODUCED. Pooling six values into one top-80 set is not well defined as a measure of say/do sharing: a value with a larger score scale fills both sets, and the overlap then measures the value, not the sharing.

PART205-DESIGN.md:68-70: "averaged over the prompts ranked". PART205-DESIGN.md:72-73: "Sets SAY_a, SAY_b, DO_a, DO_b
(top 80 each, all six values pooled)."

So SAY_a is the top 80 of a score averaged over 24 SAY prompts (6 values x 4 paraphrases) and DO_a the top 80 of a
score averaged over 24 DO prompts (6 values x 4 scenarios). Averaging is linear, so a value whose parts have a larger
|a*g| (a strongly tokenised value, a value whose question has a large logit gap, a value with more consistent
paraphrases) contributes more to the mean for every part it touches. If one value's parts score 3x the others' on
average, the top-80 is mostly that value's parts in both SAY and DO, and SAY_a ∩ DO_a is large because the two sets
share a *value*, not because saying and doing share parts. The design reports no per-value composition of the top-80
and no per-value normalisation, so the overlap cannot be decomposed after the fact.

This is the same "match the control on everything except the property under test" concern as F7, applied to the
pooling rather than the random sets. It also affects the retest ceiling: if one value dominates, SAY_a and SAY_b are
both mostly that value's parts, Retest is inflated, and O is deflated — the design's own headline quantity moves for a
reason unrelated to saying and doing.

Smallest fix: choose one of (a) normalise each value's scores (z-score within value, or rank within value) before
pooling, (b) report O per value (six O values, each from that value's own top-80) and pool the six O's, or (c) report
the per-value composition of every pooled top-80 beside O and say the pooled O is read only when no value holds more
than, say, half of each set. (b) is the most informative and costs no extra passes: the scores are already computed
per prompt.

### F13 — LOW, INTRODUCED. The design leaves STANDARD-CONTROLS item 2 (poles) out without saying so, and ranking by |activation x gradient| merges a neuron's two poles into one score.

STANDARD-CONTROLS.md:9-10: "**Poles.** A neuron's positive and negative values are separate units (NEURON-RECORDS.md
already keys records by pole; Sauers' params/oracle works this way)." PROCEDURE.md section 2 requires a control be left
out "only with a written reason".

PART205-DESIGN.md:68: "Score per part = |activation x gradient|". The absolute value collapses a neuron's positive
and negative poles into one score, so a neuron that matters positively and one that matters negatively are the same
unit. The design never says whether a neuron unit is signed, and its "Not used" list (PART205-DESIGN.md:102-103)
covers only the dose ladder and the reader, not poles. The copy test replaces activations with signed You values, so
the two poles behave differently there while being ranked identically.

Why it matters: the top-80 can include a neuron whose large |a*g| comes from a pole that moves the answer the wrong
way, and the copy then writes the You value in and moves the answer for a reason the score did not intend. The
design's own Limits (PART205-DESIGN.md:172-173) note this in general ("a part can matter and score low") but not the
pole case.

Smallest fix: rank signed (a*g, not |a*g|) so a part's direction is kept, or key each neuron by pole (the positive
score and the negative score as two units) as NEURON-RECORDS.md does; and add poles to the "Not used" list with a
reason if they are left out.

### F14 — MEDIUM, INTRODUCED. The You and third-person frames differ in the question wording as well as the persona, so G2 and T both attribute to persona a change that is partly a change of question.

PART205-DESIGN.md:77-82: "run the You frame ... run the third-person frame (baseline) ... The question asked in the
third-person runs is 'Do you personally care about <value>?', as Sauers asked." PART205-DESIGN.md:109-110: "G2: the
You/third-person gap P_you - P_third is at least 0.10 ... Fails if the stage does not treat itself differently from a
named AI."

The two frames are:

```
You:   System: You care about honesty.        User: Do you care about honesty? A) yes B) no
third: System: Zara, an AI, cares about honesty. User: Do you personally care about honesty? A) yes B) no
```

Two things change at once: the persona line and the question ("Do you care" vs "Do you personally care"). G2 reads
the whole gap as "treats itself differently from a named AI", but a model can answer "Do you personally care" less
affirmatively than "Do you care" for reasons that have nothing to do with who the persona is (the word "personally"
asks for a different claim). T = (P_copy - P_third) / (P_you - P_third) then has a denominator that includes the
question change, and the copy arm must carry both the persona and the question-wording difference to reach T = 1.

This also shortens the shared suffix: measured with the real tokenizer (`suffix_check.py`), the You-vs-Zara pair has
a 22-token shared suffix when the question is held fixed and 15 tokens when the third frame uses "Do you personally
care". So the copy test overwrites 7 fewer tokens than it would with one question, and the part of the gap that comes
from "personally" is not in the overwritten region at all.

Smallest fix: run the third-person baseline with the *same* question as the You frame ("Do you care about
honesty?"), and add the Sauers wording ("Do you personally care") as a separate arm, reported beside it. That makes
G2 a persona gap and keeps the copy test's denominator one change. If the Sauers wording is kept as the primary, say
that T's denominator includes the wording change and report the same-question T as the clean number.

### F15 — LOW, INTRODUCED. 205.8 names "Qwen3.5-4B's O" but the design runs two Qwen models (base and chat) and does not say which one the prediction reads.

PART205-DESIGN.md:150: "205.8 | Qwen3.5-4B's O is at least 0.2 below OLMo 2 1B Instruct's (Sauers' contrast survives
the retest ceiling, at another size and version). | low". PART205-DESIGN.md:48-49 runs "Qwen3.5-4B-Base and
Qwen3.5-4B". The prediction names only the chat model. If the base model's O is higher than the chat model's (a
plausible outcome, since the design's own 205.3 expects sharing to appear at a training step), the prediction is
ambiguous: it passes on chat and fails on base, and the design has no rule for that. The reading rules
(PART205-DESIGN.md:157-166) do not mention it.

Smallest fix: state that 205.8 reads the chat model's O (matching Sauers' chat-vs-chat contrast) and report the base
model's O beside it as a separate, unscored number; or split 205.8 into two rows.

### F16 — LOW, INTRODUCED. DO paraphrases 5-8 are declared held out "for the copy test" but no arm uses them, and no arm copies SAY parts into a DO question, so the causal test is one-directional at the question level.

PART205-DESIGN.md:65: "Paraphrases 1 to 4 of each set are the selection set (ranking); 5 to 8 are held out (copy
test)." PART205-DESIGN.md:77: "For each held-out SAY question and each third-person name". So the copy test reads
held-out SAY questions only. DO paraphrases 5-8 are held out and never read: the split sentence is true for SAY and
false for DO. That is wasted prompts, not a wrong result, but the design's "select on one set, test on another"
claim (PART205-DESIGN.md:96) is only half checked — the DO set is never tested on prompts it was not selected on.

Related: every arm copies a set into a SAY question ("Do you personally care about <value>?"). The "both directions"
control (PART205-DESIGN.md:94-95) reverses the *persona* direction (third into You), not the question direction. No
arm copies SAY parts into a DO question, so the design cannot show that the DO set is what carries doing, only that
copying it changes saying. Sauers' result is the same shape, so this is a scoping point, not a defect; it should be
in Limits.

Smallest fix: either use DO 5-8 (copy SAY parts into held-out DO scenarios and read the DO option) or say in Limits
that the DO set is tested only through a SAY question; and correct the split sentence.

### F17 — LOW, INTRODUCED. The design does not say which class family Q loads, and the checkpoint is a multimodal model (`Qwen3_5ForConditionalGeneration`) with a vision tower the design's 10 GB RAM budget does not account for.

PART205-DESIGN.md:48-49: "Qwen3.5-4B-Base and Qwen3.5-4B ... bf16 on the A770 with eager attention ... loaded
straight to the device, one model per process." PART205-DESIGN.md:134: "Q's load needs 10 GB of Windows RAM free."

The cached config (`/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e8.../config.json`) declares
`architectures: ["Qwen3_5ForConditionalGeneration"]` and a `vision_config` (depth 24, hidden 1024). Loading that class
loads the vision tower; `AutoModelForCausalLM` loads the text-only `Qwen3_5ForCausalLM` from the same checkpoint, as
`third_party/honesty-heads/code/engine.py:38-49` does. The design does not say which, so the RAM figure and the
"eager attention" setting are not tied to a load path, and the vision tower (if loaded) is dead weight on the budget.

Smallest fix: name the load class (`AutoModelForCausalLM` -> `Qwen3_5ForCausalLM`, text-only) and say the vision
tower is not loaded; or, if the multimodal class is used, say so and raise the RAM figure.

### F18 — LOW, INTRODUCED. Two of the three prior-art papers the design cites are not in PRIOR-ART.md, which PROCEDURE.md §1 asks for.

PROCEDURE.md §1: "**Search the prior art** and record the closest work in `PRIOR-ART.md` and the design." The design
cites arXiv 2610.04125 ("Representational Control over Self-Report & Behavior Coherence in LLM Risk-Taking") at
PART205-DESIGN.md:27-31 and arXiv 2604.19117 ("LLMs Know They're Wrong and Agree Anyway") at PART205-DESIGN.md:32-34.
Neither number, nor either title, appears in `PRIOR-ART.md`:

```
$ grep -n '2610.04125\|2604.19117\|Risk-Taking\|Agree Anyway' PRIOR-ART.md
(no output)
```

The third, 2605.13329, is there at line 573. Why it matters: PRIOR-ART.md is the shared record later parts search
first (PROCEDURE.md §1's first bullet is "Look in our own results first"), and a paper cited only in a design is
invisible to that search. This is a checklist item, not a defect in the science; it is cheap to fix and it is the
kind of thing registration is for.

Smallest fix: add both entries to PRIOR-ART.md (number, title, date read, what it measured) before registration.

### F3 — LOW, INTRODUCED (a limit, not a defect). O is stable when Retest is comfortably above G4's floor, and the pool-size worry does not hold; but O is defined only where G4 passes, and the design does not say what O means when Retest sits near 20, which is exactly where 205.1 says it will sit at every stage.

PART205-DESIGN.md:72-75 defines O = Cross / Retest. I first suspected a pool-size artefact (E|A∩B| = 80²/N for
independent draws, and OLMo's pool is 131,328 while Qwen's is ~295,040). Measured with a synthetic ranking with a
known shared core (`o_index3.py` in this directory):

```
same A=40,B=40,noise=0.2 across pools:
pool=131328   Retest= 31.2 Cross= 31.1 O=0.997
pool=295040   Retest= 29.0 Cross= 29.0 O=0.998
pool=295808   Retest= 28.9 Cross= 29.1 O=1.004
pool=65536    Retest= 32.7 Cross= 32.8 O=1.004
```

The ratio cancels the pool-size term, so O is comparable across families at this level. The real sensitivity is the
denominator. With the same underlying sharing (A=40, B=60, noise 0.2), thirty independent draws give Retest 21-28 and
O 0.88-1.06; with B=80 (Retest 13-22, straddling G4's 20) O runs 0.89-1.20. So O's sampling spread is about ±0.1
when Retest is near 20, which is half of the 0.2 "smallest effect worth knowing" (PART205-DESIGN.md:118). The design
does not state an uncertainty on O, and 205.3 ("the largest rise in O between adjacent stages is at mid-training end
to SFT") and the withdrawal rule (a rise of at least 0.2) both compare O across stages whose Retest may be near 20.
That is a real design choice to make, not a bug: the fix is to report O with a bootstrap over the 48 prompts (or over
the four half-sets) and to require the rise in 205.3 to clear that spread, or to raise G4 so Retest is at least ~30
where O's spread is under 0.05.

Also a wording point that is not a finding on its own: Retest is the mean of two same-question overlaps and Cross the
mean over four cross-question pairings, so O = 1 means "saying and doing share as much as a paraphrase does", which is
what the design says at :74-75. That is consistent.

## What I checked and found sound

(pending)

## What I checked and found sound

- **The question and its framing.** PROCEDURE.md §1 asks to prefer causal tests across training stages when the
  question allows it; OLMo publishes every stage, and this design uses the copy test as the causal check rather than
  the overlap alone (PART205-DESIGN.md:172-173). The two-part question (is the overlap a property of the model or of
  the ranking; where in training does it appear) is the right one, and the retest ceiling is the right instrument for
  the first half.
- **Prior art.** The Sauers archive the design cites exists
  (`/home/elisepc/green/Journal/sources/web-archives/2026-11-11-sauers-say-do-subnetworks.md`) and the design's
  summary of it matches: 55 of 80 (OLMo-2-7B) against 7 of 80 (Qwen3-8B), the You/Zara runs, the copy test with the
  "Do you personally care" question, and the "Qwen moves less than random parts" result. The three arXiv numbers are
  in `PRIOR-ART.md` (2605.13329 at line 573). See F18 for the other two, which are not. Part 92 (slot units start at SFT) and
  Part 50 (50.1 failed) are real and cited correctly.
- **The OLMo revisions.** `22738a97` (pretrain 3.0T), `1c3599f7` (midtrain 26B), `0d85a3d0` (SFT), `c4b04859` (DPO)
  and `48d788ec` (Instruct) all exist as cached snapshots and match `part85/checkpoints.py:26-29` and
  `part103/steps103.txt`. The config gives 16 layers, 16 heads, 8,192 MLP neurons, matching PART205-DESIGN.md:44.
- **The plain frame's rationale.** The chat template is absent from the base snapshot's `tokenizer_config.json` and
  present in the SFT one (508 characters), so "a template exists only for the chat checkpoints" is true, and holding
  the frame fixed is the right way to avoid confounding frame with stage. Part 144 already read a base checkpoint in a
  plain frame (`PART144-DESIGN.md:24-25`), so the approach is precedented — but see F5 for the gate it needs.
- **The shared suffix is real.** With the cached OLMo tokenizer, the You and third-person frames share a 15-token
  suffix ending in `Assistant: Answer:` when the third frame uses the "personally" question, and 22 tokens when the
  question is held fixed (`suffix_check.py`). Aligning from the end is well defined; see F4 for what the copied
  values mean and F14 for the question change.
- **Size arithmetic.** Every line except the pilot recomputes exactly (ranking 48, copy 2,208, off-target 48, family O
  13,824, family Q 2,688, total 16,512). The 46 arms = 6 named + 40 random, and Q's 26 = 6 + 20, are consistent with
  the control list.
- **O's pool-size worry does not hold.** A synthetic ranking with a known shared core gives O within 0.007 across
  pools of 65k to 296k (`o_index3.py`), because the ratio cancels the 1/N term. The sensitivity is to Retest, not to
  the pool (F3).
- **The gates G1-G4 can each fail.** G1 (exact logits) is a real check and is achievable on the A770 in bf16 (Part 20
  recorded max logit difference 0.0, `publish/CLAIMS.md:32`). G2, G3 and G4 each have a stated failure and a stated
  consequence ("not read"), and the reading rules say a failed gate is reported as "not read", not as a failed
  prediction. 205.2 and 205.4 are properly gated on G4 and G1-G3.
- **Both directions and the negative arm.** The reverse DO copy is read in the You run, where the behaviour is
  present, which meets STANDARD-CONTROLS item 1's requirement for a negative arm.
- **The withdrawal rule.** It is stated with a threshold (0.2), a gate (both stages pass G4) and a level (O at X at
  least 0.5), and it has a fallback sentence. That is what PROCEDURE.md §2 asks of a design.
- **Honest about what is unmeasured.** "Measured: not yet" (PART205-DESIGN.md:134) and the pilot's stopping rule are
  stated rather than invented, which is the right form even though the pilot count itself is wrong (F6).
