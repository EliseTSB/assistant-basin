<!-- cheap-review.mjs cheap-review-part198-design-r1-ds-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 481a03b7e974..HEAD, lane ds, df-run exit 0 -->
# Cold adversarial review — part198-design-r1

Change reviewed: `git -C /home/elisepc/worktrees/ab-part198 diff 481a03b7e974..HEAD` (13 commits).
Artifact: `PART198-DESIGN.md` plus `results/part198/pilot/` scripts and JSON outputs.
Reviewer domain: `/home/elisepc/green/Journal/scratch/review-part198-design-r1-ds-r1-2026-11-08`.
All scripts I wrote are in this directory; model files were read only, never written.

## Verdict: SHIP-WITH-FIXES (do not register the part until F1–F5, F3b and F9 are fixed)

The design is unusually careful about what the pilot does and does not show, and its
pilot-to-test separation is real. But a pre-registration is judged by whether its
predictions can fail for the reason it says they can. Several cannot, the statistic that
six predictions depend on is not actually defined anywhere in the change, and the
intervention's prompts are unspecified. Measured with the design's own procedure, T\* is
12.10 and 198.10 already fails at 3 of 7 held-out models under one reading and passes at
2 of 7 under another — a prediction that flips on an undefined norm convention. These are
all fixable before registration; none requires new data.

Severity: **Critical** = the prediction is unfalsifiable or the statistic is undefined;
**High** = the prediction is near-guaranteed or measures the wrong thing;
**Medium** = a real methodological bias; **Low** = a factual/citation error in the text.

---

## Findings

### F1. Critical — `stats198.py` does not exist, so T_word is not frozen (INTRODUCED)

`PART198-DESIGN.md:62,110,283,325` define T_word by "the fixed stopword list in
`stats198.py`", and the gate (`:110-114`) requires "`stats198.py` reproduces the pilot's
F1 ranks ... and the exploration's T_max for every base layer of both hosts". There is no
`stats198.py` anywhere in the repository or in the change:

```
$ git -C /home/elisepc/worktrees/ab-part198 ls-tree -r HEAD --name-only | grep -i stats198
(no output)
$ find /home/elisepc/green/worktrees/ab-part198 -name 'stats198*'
(no output)
$ grep -rn stats198 /home/elisepc/green/worktrees/ab-part198
PART198-DESIGN.md:62/110/283/325 only
```

The only place the list actually used exists is `explore_posctl.py:9-10`, a pilot script
whose STOP set is not identified as the frozen list. T_word appears in 198.2, 198.6,
198.7, 198.9, 198.10 and 198.11 — six of eleven predictions — plus the headline gate.
A pre-registration that says "the definition is fixed in a file written later" has not
fixed the definition: the filter can still be edited after the 24 writes are seen, and
the gate's cross-check cannot be run at all.

**Fix (smallest):** add `stats198.py` to this change, copied from `explore_posctl.py`
(pcs/F1/T_max/T_word, the exact STOP set), and put its sha256 in the design. If the file
is genuinely to be written at registration, inline the full stopword list and the exact
content predicate in `PART198-DESIGN.md` now and change "fixed in `stats198.py`" to
"fixed below".

Measured: recomputing T_word for GPT-2 XL and Qwen3.5-0.8B with the `explore_posctl.py`
STOP set reproduces the design's quoted numbers (see "checked and sound"), so the
definition is recoverable — but only from a file the design does not cite. The missing list
is not a bookkeeping issue: the calibration and held-out maxima that decide 198.10 are
`' time'` (12.10), `' set'` (10.33/15.90), `' order'` (14.36) and `'bfd'` (13.08); adding
`time`, `set` and `line` to the stoplist moves T\* from 12.10 to 8.01 and changes the
198.10 count. The list is the single most consequential object in the
part and it is not in the change.

### F2. High — 198.2 is guaranteed by layer position and cannot test what it claims (INTRODUCED)

198.2 (`:293`) says T_word finds at most 1 of 24 gradient edits. T_word is a matrix-level
max, and "found" is rank 1 among the model's 48 `mlp.c_proj` matrices (`:64`). On GPT-2 XL
the natural T_word maxima are at late layers, above every layer the gradient editors touch.
Measured from the pilot's own saved top tokens (`measure_word.py`, output below):

```
$ python3 measure_word.py
top five base T_word: [(45, 9.79, 'Used'), (46, 9.07, ' play'), (43, 8.87, ' RandomRedditor'), (15, 7.65, ' externalToEVA'), (0, 7.4, 'ing')]
range: 5.05 to 9.79
edit layers 13-17: [(13, 5.97, 'ogle'), (14, 5.74, ' sidx'), (15, 7.65, ' externalToEVA'), (16, 5.56, ' prolifer'), (17, 5.08, ' answ')]
```

Every gradient editor writes only layers 13–17 (`:71-73`). For 198.2 to fail, an edit at
layers 13–17 would have to push that layer's T_word above 9.79 — a rise of 4.7 at layer 17,
or 2.1 at layer 15. The pilot's largest measured movement from any gradient edit is
**+0.38** (ROME, layer 17, `explore-gpt2.json`). The aimed positive control at ROME's full
spectral norm reached only **8.32** at layer 17, still below 9.79 (`explore_posctl.json`).

So 198.2 would pass even if T_word were perfectly sensitive to gradient edits, because the
statistic is outranked by the model's own late-layer function-word directions. It reports
"blind" when the truth is "outranked". This is exactly the distinction the part exists to
make. `:313-314` acknowledges the reach problem for 198.5, but 198.2 is still stated as a
rate and headline-supporting.

**Fix:** either (a) drop 198.2 as a prediction and report T_word descriptively, letting
198.4 (naming at the edited layer, which is not a rank-among-48 test) carry the directional
null; or (b) redefine "found" for single-layer editors as rank 1 among the layers the
editor could plausibly touch (e.g. 13–17), or as "edited T_word exceeds the base T_word of
the same matrix by more than the pilot's 0.4". The same objection applies, more weakly, to
198.1: the design's own triangle-inequality argument (`:161-167`) shows F1 needs
1.11–1.71x sigma1 and the measured updates are 0.04–0.52x, so 198.1 is close to a
restatement of the pilot's norm measurement rather than a new test.

### F3. Critical — T\* is set by one calibration outlier, and 198.10 is already false under one reading and true under another (INTRODUCED)

`:87-89` defines **T\* = the largest T_word over every MLP output matrix of the three
calibration models**, with no cap or per-model normalisation. `:301-302` uses that one
number for both 198.10 ("at most 2 of 7 held-out models above T\*") and 198.11 ("at least
20 of 24 wsp writes above T\*"). `:310-311` says T\* is computed before scoring, but it is
not computed in the design (`:301` says "T\* not yet computed").

I computed it from the three calibration models on disk with the design's own procedure
(script `measure_calib.py`; bf16 weights, float64 statistics, the `explore_posctl.py`
content filter):

```
$ python3 calib-summary (from calib-*.json written by measure_calib.py)
calibration model        T_word max (layer, token)        bf16   fp32   T_max*
pythia-410m              6.73 (L5, 'ton')                6.73   6.72   11.34
qwen3-0.6b              10.33 (L24, ' set')             10.33  10.33   12.08
olmo2-1b                12.10 (L13, ' time')            12.10  12.17   28.48

T* (max over the three, bf16) = 12.1  fp32 = 12.17
```

So **T\* = 12.10**, set by one direction at one layer of OLMo-2-1B. I then measured the
seven held-out unedited models the design names, with the same procedure
(`measure_heldout.py`; bf16 weights, float64, Gram+eigh verified against exact SVD on the
top layers):

```
$ python3 heldout-summary (from heldout-*.json)
held-out model        T_word max (layer, token)     vs T*=12.10
pythia-1.4b           14.36 (L22, ' order')          ABOVE
qwen3-1.7b            15.90 (L24, ' set')            ABOVE
smollm2-1.7b          10.92 (L20, ' time')           below
smollm3-3b            13.08 (L32, 'bfd')             ABOVE
qwen2.5-0.5b           9.21 (L19, ' way')            below
gpt2-xl (pilot)        9.79 (L45, 'Used')            below
qwen3.5-0.8b (pilot)   7.34 (L23, 'ise')             below

above T*: 3 of 7 held-out models -> 198.10 (fails if 3 or more)
```

**198.10 fails at registration under this reading.** The three above are an ordinary word
(' order'), a common verb (' set') and a three-letter fragment ('bfd'); none is a
function-word stopword under the pilot list. This is not a close call that the registered
run would discover — the models are already on disk and the design says so.

But the result flips on an implementation detail the design leaves undefined (F9). The
literal sentence at `:57-59` says "g is taken as the norm module's output on a vector of
ones". For a LayerNorm that output is the **bias**, not the weight. Substituting the bias
for Pythia-1.4B's final norm changes its top direction from 14.36 (' order') to 7.64
('Gmb'); for GPT-2 XL it changes 9.79 ('Used') to 8.20 (' TheNitrome'):

```
pythia-1.4b L22 g=weight (14.36, ' order')
pythia-1.4b L22 g=bias   ( 7.64, 'Gmb')
gpt2-xl    L45 g=weight ( 9.79, 'Used')
gpt2-xl    L45 g=bias   ( 8.20, ' TheNitrome')
```

Under the bias reading Pythia-1.4B's measured maximum disappears (I re-read only layer 22
under the bias rule, not all 24 layers), so 198.10 is at most 2/7 and very likely passes.
The same document therefore predicts both PASS and FAIL for 198.10 depending on which of
its own sentences is implemented: the prediction is not well-defined until the norm rule
is pinned.

T\* itself is also decided by stopword choices that are not written down (F1):

```
olmo2-1b  L13 12.10 ' time', L14 9.36 'line', L12 8.01 ' first', L11 7.96 ' none'
qwen3-0.6b L24 10.33 ' set', L22 7.37 ' day'
```

Adding `time` to the list moves T\* to 10.33 ('set'); adding `set` moves it to 9.36
('line'); adding `line` moves it to 8.01 (' first'). T\* moves by 4.1 across three
stopword choices that are nowhere in the change.

For 198.11 the same threshold is now below a natural held-out direction: Qwen3-1.7B
unedited reaches 15.90, higher than the pilot write's layers 17 (11.13), 19 (9.35) and 21
(12.85); only layer 20 (16.20) clears it comfortably. So "edited T_word above T\*" is not
an edit-specific signal in the Qwen family, and 198.11's 20/24 is a real risk rather than
the near-certainty its 0.65 confidence suggests. The same Qwen family is in calibration
(Qwen3-0.6B, top direction ' set' 10.33, the same token and layer as held-out Qwen3-1.7B's
15.90) and supplies the Arm B host (Qwen3.5-0.8B) — the family leakage the brief asks about
is measured, not hypothetical.

**Fix:** (1) freeze the stopword list and the norm rule first (F1, F9); (2) compute T\* and
all seven held-out maxima now and put the table in the design — these are unedited public
models, not test data, so this is not leakage; (3) if 198.10 fails at 3/7, revise or drop it
rather than registering a prediction already known to be false; (4) define 198.11 against
the host's own base T_word at the edited layer (a matched threshold), not a cross-model max.

### F3b. Medium — the content filter was tuned on unedited maxima of two models that are in the held-out set (INTRODUCED)

`:324-327` says the stopword list "was written after the pilot showed function words among
the unedited maxima", and `:326-327` says "T_word and its filter were chosen from one `wsp`
write; the 24 new writes, the random control and the **held-out unedited models** are the
test." The filter was not chosen from one `wsp` write alone. The exploration computed the
unedited maxima on **Qwen3.5-0.8B (all 24 layers, `:202-205`)** and on **GPT-2 XL (all 48
layers, `:218-219`)** — and both of those models are in the 7-model held-out set for 198.10
(`:85-87`). The stopword list can therefore exclude exactly the function-word directions
that would have produced false alarms on two of the seven "held-out" models. This is
pilot-to-test leakage in the false-alarm arm, and the Limits section understates it.

The scale of the effect is measurable: with the `explore_posctl.py` STOP set, GPT-2 XL's
base T_word maxima are 9.79/9.07/8.87 at layers 45/46/43; without the filter the raw T_max
reaches 17.06 (' to' at layer 36) and 15.96 (newline at layer 47). The filter is doing a lot
of the work of keeping the hosts below T\*.

**Fix:** treat GPT-2 XL and Qwen3.5-0.8B as in-sample for 198.10 (report them separately)
and score 198.10 on the five models the filter never saw; or derive the stopword list from
the three calibration models only, now, and freeze it. Either way, correct `:326`.

Measured, the leakage did not rescue the prediction: the three held-out models above T\*
(Pythia-1.4B, Qwen3-1.7B, SmolLM3) are all outside the filter's tuning set, and the two
tuned-on hosts are below it. So this is a design-cleanliness finding, not the cause of
F3's failure.

### F4. High — "took" counts records the base model already answered (INTRODUCED)

`:92-93` and `:91-92` define "took" as the new object outranking the old one after the
edit (GPT-2: first-token log-prob; wsp: `free_exact` 3/3). There is no before/after
comparison. CounterFact records on which the base already prefers the new object therefore
count as edits that took. This is measurable in the cited pilot output:

```
$ python3 - <<'EOF'
import json; d=json.load(open('results/part198/pilot/pilot-part3-MEMIT-b10.json'))
b=[r['took'] for r in d['runs'][0]['efficacy_before']]
a=[r['took'] for r in d['runs'][0]['efficacy_after']]
print('before', b); print('after ', a)
print('already true before:', sum(b), 'of', len(b))
EOF
before [False, True, False, False, False, True, False, False, False, False]
after  [True, True, True, False, True, True, True, True, True, True]
already true before: 2 of 10
```

`pilot-part3-MEMIT-b10.json` reports "took 8 of 10", but 2 of those 8 were already true
before the edit. The gate (`:113-114`) requires 21 of 24 single edits to take; a few
spurious takes count toward it, and every rate for MEMIT/AlphaEdit has the same
contamination. For a null result ("these edits are invisible") this biases toward the null
by including records where there was little or no update to see.

**Fix:** define took = after-true AND before-false (or require an improvement of, say,
1 nat), and report the before-true count. Same for wsp: require the base free answer not to
be the target before teaching.

### F5. High — two scored clauses cannot fail, and one prediction contradicts the headline's withdrawal rule (INTRODUCED)

(a) 198.5 (`:296`) fails if "named in fewer than 20, **or found in more than 12**". The
control update is rank one with spectral norm = the record's ROME update (5.43). At that
size the pilot measured layer 17 T_word = 8.32, below the fixed layer-45 natural 9.79, and
"found" requires rank 1 among 48. The layer-45 baseline is a property of the unedited
model and is identical for every record, so "found in more than 12 of 24" is essentially
unreachable — the clause is padding, and 198.5 reduces to the naming clause. The design
itself states the outcome at `:237-238` ("named and not found"). If the purpose is to show
the reading reaches layer 17, "found" at rank 1 of 48 is the wrong test; if the purpose is
to show it cannot be ranked first at ROME's size, that was already known from the pilot and
is not a prediction.

(b) 198.9 (`:300`) "T_word ranks a randomized layer first in at most 1 of 24" is close to
deterministic. I built the control exactly as `:103-105` describes on the real Qwen3.5-0.8B
weights, using the pilot update's singular values and random orthonormal bases
(`measure_random.py`):

```
$ python3 measure_random.py
base T_word on the five edited layers: {17: 5.94, 18: 6.67, 19: 6.73, 20: 6.78, 21: 6.97}
64 random unit residual directions, seed 198: T_max 6.29, T_word 6.29
64 random unit residual directions, seed 199: T_max 6.23, T_word 6.23
64 random unit residual directions, seed 200: T_max 6.51, T_word 6.46
layer 17: random-matched edited T_word 6.14 (base 5.94)
layer 18: random-matched edited T_word 6.90 (base 6.67)
layer 19: random-matched edited T_word 6.25 (base 6.73)
layer 20: random-matched edited T_word 6.32 (base 6.78)
layer 21: random-matched edited T_word 6.31 (base 6.97)
```

The model's highest base T_word is 7.34 (layer 23). A randomized layer's T_word is
6.1–6.9 in every layer I measured, so it cannot rank first. "At most 1 of 24" is
mathematically comfortable, not an empirical finding. It is a useful sanity control, but
scoring it as a prediction with confidence 0.85 overstates what it shows. (Note also the
layer-18 random update produced T_max 13.95 on a function-word direction: the content filter
is what keeps the random control clean — another reason the unfrozen list matters.)

(c) `:285-288`: the headline is "ordinary gradient edits ... are invisible ..., while a
writer that aims at the answer's output direction is found **and named**", but the
withdrawal rule withdraws only on 198.5 or **198.6** failure — not 198.7, the naming
prediction. If 198.6 passes and 198.7 fails, the headline stands while the "and named" half
is false. **Fix:** add 198.7 to the withdrawal rule, or drop "and named" from the headline.

### F6. Medium — Arm B's prompts and templates are not specified (INTRODUCED)

`:79-83` says each Arm B record is "written alone as three questions from a fixed template
for its relation, plus one held-out question". The ten templates are not in the design and
not in the repository; the pilot hand-wrote three English phrasings for P103
(`wsp198.py:44-46`). The prompt is the intervention: a different phrasing changes which
question keys are protected and how much the write generalises, so "fixed template" without
the text is not a pre-registration. **Fix:** paste the ten templates (three taught
questions + one held-out question each) into the design or a `templates198.json` committed
with it, and pin the model's chat template/thinking-off setting (the latter is mentioned at
`:281`, not shown).

### F7. Medium — the record draw is not reproducible (INTRODUCED)

`:70-72`: "24 drawn by seed 198 from CounterFact cases 11 onward"; `:78-81` adds a relation
filter and "keeping only records whose `target_new` is one word of three or more ASCII
letters and differs from every record already drawn". It does not say: which PRNG and
algorithm, whether the filter is applied before or after drawing 24, or what "differs from
every record already drawn" means (dedupe on subject? on target?). "Drawn by seed 198" is
not a procedure; two implementations will produce different sets, and the Arm B sample size
may differ from 24 after filtering. **Fix:** give the exact code (e.g.
`random.Random(198).sample(...)` after a stated filter) and the resulting `case_id` list in
the design. This can be computed now with no test data.

### F8. Medium — EasyEdit is unpinned while everything else is pinned (INTRODUCED)

`:68-70` pins the GPT-2 checkpoint by sha256 and `:77,280` pins wsp by commit, but
`:68-69,278-279` describe EasyEdit as "main fetched 2026-11-08 11:16 CT", and `:126` says
"tarball, not yet pinned to a commit". Arm A's whole null depends on the editors' shipped
hyperparameters and code. A pre-registration that pins the weights but not the editor is
not reproducible. **Fix:** record the EasyEdit commit hash (or sha256 of the tarball) and
the `hparams/.../gpt2-xl.yaml` files, and put them in the design.

### F9. Critical — the norm rule in the text is not the rule in the code, and the choice decides 198.10 (INTRODUCED)

`:57-59`: "g is taken as the norm module's output on a vector of ones, so zero-centred norms
are handled". For an RMSNorm this is the weight (correct). For a **LayerNorm** the module's
output on ones is the **bias**, not the weight, and the bias is not a per-dimension scale.
The implementation that exists, `explore_gpt2.py:52`, uses `ln_f.weight` and drops
`ln_f.bias`; `explore_tokens.py:53` uses `norm(ones)` only because Qwen3.5's norm is an
RMSNorm. Arm C includes Pythia-410M (calibration) and Pythia-1.4B (held out) and GPT-2 XL,
all GPT-NeoX/GPT-2 LayerNorms with a final-norm bias — measured present, e.g. for
pythia-410m `gpt_neox.final_layer_norm.{weight,bias}` (1024,) each.

The two readings are not close. Measured on the real weights (script inline in my domain):

```
pythia-1.4b L22 g=weight (14.36, ' order')
pythia-1.4b L22 g=bias   ( 7.64, 'Gmb')
gpt2-xl    L45 g=weight ( 9.79, 'Used')
gpt2-xl    L45 g=bias   ( 8.20, ' TheNitrome')
```

The text's literal rule (output on ones = bias) puts Pythia-1.4B below T\*=12.10 and makes
198.10 pass at 2/7; the code's rule (weight) puts it above and makes 198.10 fail at 3/7
(F3). A pre-registration whose headline false-alarm prediction flips on an undocumented
choice between weight and bias has not fixed its statistic.

**Fix:** write the exact rule in the design ("g = norm.weight; subtract the mean for
LayerNorm; ignore the bias" — and justify ignoring it), implement it once in `stats198.py`,
and recompute T\* and the held-out table with it before registration.

### F10. Medium — "named" is lenient and asymmetric, and can be impossible for stopword targets (INTRODUCED)

`:96-97,308-309`: "named" = the stripped, case-folded **first token** of the new object,
with or without a leading space, among a direction's **top three** tokens. Three problems:
1. Top-3 + first-token matching is lenient. For a multi-token object ("Danielle Darrieux")
   only "Dan"/" Danielle" is required, and for short/common objects a fragment in third
   place counts. This makes 198.5/198.7/198.8 easier to pass while making 198.4's null
   harder — the same criterion is used on both sides.
2. Arm B selects targets that are "one word of three or more ASCII letters" but does **not**
   exclude targets on the T_word stopword list. If a drawn target is a stopword (e.g. "One",
   "First"), the direction that names it is filtered out of T_word, so the write can never
   be "named" and counts against 198.7. Nothing in the design prevents this.
3. "Top three tokens" is computed per direction, but the direction itself is selected by
   T_word, whose content filter is the same filter; for a stopword target the selection and
   the naming test interact.

**Fix:** pre-register a strict naming test (the target's first token id, or the exact
decoded string, at rank 1 of the chosen direction) as primary, keep top-3 as secondary,
and exclude stopword targets at draw time before registration.

### F11. Medium — mass-edit batches have no take-rate gate, so a weak batch biases the null (INTRODUCED)

`:72-74,92-93`: mass edits are two batches of 10 per editor; "edits that did not take are
counted and left out of every rate". For a batch there is one edited matrix set regardless
of how many of the 10 records took, so a batch where 5 of 10 failed produces a smaller
update and an easier null for 198.1–198.3. The pilot's MEMIT-b10 took 8/10 and AlphaEdit-b10
10/10, but nothing in the design stops a registered batch at 4/10. **Fix:** add a per-batch
take gate (e.g. at least 8/10) and report the take count with each batch; or state that a
batch below the threshold is re-run/dropped before scoring.

### F12. Low — two pilot claims are contradicted by the cited JSON (INTRODUCED)

(a) `:217-218`: "The edited matrices' most concentrated directions are the base's own
(same index, same tokens, z within 0.4)." Measured from `explore-gpt2.json`: ROME layer 17's
top direction changes from index 60 `' answ'` (z 5.08) to index 39 `'interesting'`
(z 5.46), and AlphaEdit layer 16 changes index 54 `' prolifer'` to index 51 `'\r'` (same z
5.56). The "z within 0.4" part holds; "same index, same tokens" does not.

(b) `:223-225` cites `explore_pcs.py`/`explore-pcs.json` for "the top 12 singular values
... move under 2% (ROME at layer 17: the largest change is 6.74 to 6.87 at index 11)".
`explore-pcs.json` contains only PCS values; the singular values are in
`explore-gpt2.json` (`base[l].s` vs `edited_s`), and the largest change is at index **10**
(0-based): 6.738 → 6.865 = 1.89%. The number is right, the citation and index are not.

### F13. Low — "any edited layer counts" has no multiplicity correction (INTRODUCED)

`:64-65`: for multi-layer editors, any edited layer ranking first counts. With 5 edited
layers, a blind statistic whose per-matrix rank were uniform would "find" a record with
probability 5/48 = 10.4%, i.e. an expected 2.5 finds in 24 records — the fail threshold of
198.2/198.3 is 2. The pilot's base ranks are not uniform (they sit at 19–30 for F1 and
13–32 for PCS), which is why the pilot does not show this, but the design should state the
null it assumes. **Fix:** state that the null is not uniform and why, or use a per-editor
corrected threshold (e.g. require the best edited layer to beat a matched null quantile).

### F14. Low — 198.8 cannot separate the write from interference (INTRODUCED)

`:299`: after 10 sequential writes, at least 5 of 10 objects named by a content direction in
layers 17–21. This is not piloted (`:299`), and the statistic is read once after the tenth
write, so a failure cannot tell "the statistic does not see a write" from "the tenth write
overwrote the first". Confidence 0.5 is honest, but the prediction as written is not
informative about the statistic. **Fix:** read statistics after each write too (cheap, no
extra edits) so the per-write survival is observable; keep 198.8 as the end-state.

### F15. Medium — the pilot's exploration inputs are not in the change (INTRODUCED)

The four exploration scripts load update tensors that are not committed:

```
$ grep -n "\.delta\.pt\|delta\.pt" results/part198/pilot/explore_*.py
explore_tokens.py:15: DELTA = ... / "wsp-pilot.delta.pt"
explore_pcs.py:34:  D = torch.load("wsp-pilot.delta.pt")
explore_gpt2.py:    (arguments pilot-sd.json.ROME.delta.pt, ...)
$ git -C /home/elisepc/worktrees/ab-part198 ls-tree -r HEAD --name-only | grep 'delta.pt' | wc -l
0
```

The files exist only in the author's working directory
(`/home/elisepc/edit-detect/part198/*.delta.pt`, ~450 MB), not in the change. The design's
"Code cross-check ... reproduces ... the exploration's T_max for every base layer of both
hosts" needs only the base weights, so the gate can technically run; but the explorations
themselves cannot be replayed from the artifact, and `stats198.py` (F1) would be checked
against outputs whose inputs are absent. **Fix:** either commit the delta tensors, or add a
script that regenerates them from the pinned editors and record their sha256; at minimum
state in the design that they are external artifacts.

### F16. Low — "F1's robust z" is undefined (INTRODUCED)

`:88-89`: "Also reported per model: F1's robust z of its largest member and the ratio of the
largest to the second-largest top singular value." "Robust z" is not defined: over which
population (the model's 24/48 matrices? the model's layers?), with which centre and scale
(median/MAD? a trimmed mean?), and which "largest member" (largest sigma1?). It is
descriptive, not a prediction, but the brief's standard is that statistic definitions should
be computable exactly as described. **Fix:** write the formula, e.g.
`(max_l sigma1 - median_l sigma1) / (1.4826 * MAD_l)`, population = that model's MLP output
matrices.

### F17. Low — 198.7 is not defined when no edited layer ranks first, and is subordinate to 198.6 (INTRODUCED)

`:298`: "the top content direction of the layer ranking first names the new object". If the
layer ranking first is an *unedited* matrix (which is what 198.6 predicts for the gradient
editors), there is no written object at that layer, so 198.7 is undefined for that write.
For `wsp` 198.7's denominator is "the writes that took", and its numerator can only exist
when an edited layer ranks first — i.e. 198.7 can pass at 18/24 only if 198.6 has at least
18/24, so the two are not independent and `:288`'s "each other prediction is reported on
its own" is not true for this pair. **Fix:** define it as "the highest-ranked *edited*
layer" and say 198.7 is conditional on 198.6 (or score the rank-1 layer's naming regardless
of whether it is edited, and say so).



---

## What I checked and found sound

- **Pilot numbers vs cited JSON.** I recomputed every number in the Pilot section from the
  JSON files. ROME/R-ROME/ROME-mom2 ratios and F1 ranks, MEMIT per-layer ratios and ranks,
  AlphaEdit single and b10 values, the "1.4x / 2.3x batch vs single" claim, the wsp layer
  ratios, ranks, Frobenius percentages, numerical ranks, and the sigma1 ratios all match
  their JSON (`pilot-part1/2/3-*.json`, `wsp-pilot.json`): the range over the seven GPT-2
  runs is 0.999738 (R-ROME L17) to 1.006360 (AlphaEdit-b10 L17), i.e. the design's
  0.9997–1.0064. The 27.78 largest sibling, the 1.11–1.71x and 1.86–2.52x triangle-inequality
  floors, and the 0.536/0.450/0.353/0.456/0.467 wsp floors all reproduce exactly.
- **T_max/T_word definitions reproduce.** Using the `explore_posctl.py` STOP set and the
  design's content predicate, `explore-tokens.json` gives base Qwen3.5-0.8B T_word
  5.39–7.34 (design: 5.4–7.3), edited layers 17–21 = 11.13, 6.88, 9.35, 16.20, 12.85
  (design: 11.1, 6.9, 9.3, 16.2, 12.8), with English at singular indices 5, 23, 7, 1, 3
  (design: 5, 7, 1, 3 for the four that name it; layer 18 at index 23). `explore-gpt2.json`
  gives GPT-2 XL T_word 5.05–9.79 with the top three at layers 45 (9.79 'Used'), 46 (9.07
  ' play') and 43 (8.87 ' RandomRedditor'), exactly as `:232` says. So the design's numbers
  are not invented, even though the file that defines them is missing.
- **PCS formula and claim.** `explore-pcs.py:9-12` implements
  `(||sum of unit rows||^2 - n)/(n(n-1))` exactly as `:55-56` states; `explore-pcs.json`
  gives ROME layer 17 a 1.75x rise at rank 10, and 1.00x for MEMIT and AlphaEdit, matching
  `:223-228`.
- **Aimed positive control construction.** `explore_posctl.py:25-30` builds u = centred,
  unit `g*E[' English']`, random unit v, norm 5.432, and reports 5.08 → 8.32 (named), 5.11
  at half (not named), 23.8 at twice (found). This matches `:234-238`. The random input
  direction is *not* a defect: for a rank-one update u v^T the output-side singular vector
  is u, so the token reading is unaffected by v; the control tests output-side reach, which
  is what it claims. The design should say this explicitly so the control is not read as a
  weaker ROME analogue.
- **bf16 vs float32 for the same weights is not material at this threshold.** Measured:
  Qwen3-0.6B's T_word per layer is identical to two decimals in bf16 and fp32, Pythia-410M's
  T\* is 6.73 (bf16) vs 6.72 (fp32), and OLMo-2-1B's is 12.10 vs 12.17. So the Arm C / Arm B
  precision mismatch by itself does not move a threshold by more than ~0.07; it should
  still be stated (the design compares fp32 Arm B values to a bf16 T\*), but it is not a
  reason to hold the change. The norm weight-vs-bias ambiguity (F9), not precision, is what
  moves the number by 6.
- **The Gram+eigh shortcut I used for Arm C is exact for the top 64.** I computed the top
  left singular vectors from the 2048×2048 Gram matrix (`torch.linalg.eigh(W @ W.T)`),
  which is ~20x faster than the full SVD; I verified it against `torch.linalg.svd` to
  machine precision on pythia-1.4b layer 0, and re-computed the three pivotal held-out
  layers (pythia-1.4b L22, qwen3-1.7b L24, smollm3-3b L32) with the full SVD:
  14.3647 ' order', 15.8971 ' set', 13.0773 'bfd'. So the 3-of-7 result is not an
  artefact of the shortcut.
- **Sizes and levers.** The 2.1 h / 0.5 h / 0.5 h / 0.3 h / 3.4 h arithmetic is correct;
  the per-edit times and peak-RSS figures match the JSONs.
- **The part is defensive, not an evasion recipe.** It measures published editors and a
  benign public-benchmark writer; the aimed control is a detector-calibration canary, not a
  method for hiding an edit, and nothing here is built to evade detection. The Limits
  section is honest about that.
