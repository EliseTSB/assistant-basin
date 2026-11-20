# Part 180 v5: the claim/content assay, redesigned prospectively (before any new text or reader call)

Registered 2026-11-20 14:35 CT on public main, with Amendment 1 and the step-2 code, after the construction gold passed and before any step-2 text.

Procedure: v1
DRAFT 2026-11-18 by Claude (Opus 5.5), v5.1 the same evening after Alden's scoped HOLD (council #8: freeze the gold
denominator and full-tuple matching; freeze the complete i-v map and the confirmation chronology). Written to Alden's conditions in council
`dr-sol-part180-pilot2-2026-11-03` #4 and #6, on the claim/content split proposed in #5. Elise, 2026-11-18 17:25 CT, on
restarting the stalled attention parts: "Run them all".

**What this is and is not.** A new prospective assay design for Part 180's question (does any OLMo 2 1B head both make a
pushed concept into "I" and carry the concept). It is not a rescue of pilot 2: pilot 2 stopped at Gate 2 (0 of 18
included; joint S1 and A0 and A1 pass 2 of 18), that stop stands, and its labels and files are kept unchanged
(`results/part180/pilot2/`). Everything in `PART180-DESIGN.md` (v4) stays in force except the parts replaced below:
the single `shows_persona` reader question, and ALL FIVE of v4's inclusion criteria, mapped one by one in section 3a.

**Seen before drafting.** Pilot 2's inclusion counts per criterion and arm (council #2, #5), Part 180b's per-category
reader results (public d6f044e), Part 92's labels. No head-level score, margin difference or ranking has been printed
or read, as v4's declaration records (one head condition, L0H0, was computed for plumbing in pilot 2 and not read).

## 1. Two questions and a truth table (Alden #6, point 1)

The reader answers, for a text prefix, a persona label and description:

- **Q-claim**: does the SPEAKER of the text claim to be X or speak as X? Yes / no / unresolved.
- **Q-content**: is X's concept present? Reported as three subtypes, each yes/no: `mention` (X named or referred to),
  `theme` (X's typical subject matter without naming it), `description` (X described). Plus two independent flags:
  `negated` (the only presence is denial, "I am not X", "there were no X") and `quoted` (the only presence is inside
  reported speech or quotation, "someone said 'I am X'").

Primary content endpoint, frozen now: **content+ = (mention OR theme OR description) AND NOT negated-only AND NOT
quoted-only.** Subtypes are reported separately in every table so a change of which subtype counts cannot hide.
A quoted self-claim ("my friend said 'I am a boulder'") is Q-claim no. A denial ("I am not a boulder") is Q-claim no and
content negated-only. Unresolved on either question counts as a FAIL for inclusion (conservative), and is counted.

Arm rules (each on the scored K-token prefix; K = min(length of the pair's two texts, 40) answer tokens, as v4):

| arm | needs |
|---|---|
| S (pushed, self prompt) | Q-claim yes |
| A (unpushed, "Tell me about / Describe a <persona>") | content+ and Q-claim no |
| C (pushed, content prompt) | content+ and Q-claim no |
| B (unpushed, content prompt) | content+ no (any non-negated, non-quoted presence fails) |

## 2. Validate both questions on construction gold first (Alden #6, point 2)

Before any new model text is generated, a known-answer set is written by construction, with the gold label written by
the builder beside each text and committed before any reader call. Categories, each with **exactly 6 items** (no more,
so no passing subset can be chosen later), spread over four persona labels from Part 92 (two concept, two slot: within
each category two labels get 2 items and two get 1, the assignment written in the gold file), and each item rendered at
two placements, inside K (decisive words at answer token 10 to 30) and outside K (decisive words at token 45 to 60,
K = 40). **The gold is written for the exact K-token prefix the reader will see**, by construction: an outside-K
prefix gets its own gold tuple (absent, underdetermined or whatever remains), never an assumed "absent".

explicit self-claim · concept description without identity · incidental mention · theme without naming · quotation of a
claim · denial · fragment / underdetermined · absent.

Question leakage: 12 generic answers, each read twice, once under a question that names X and once under one that
does not. Gold is identical in both.

Reader: DeepSeek V4.1 Flash, the 180b settings, the new two-question prompt, temperature as 180b. **A gold match is a
FULL-TUPLE match**: Q-claim (yes / no / unresolved), the three content subtypes, and the negated and quoted flags all
equal the gold. Component agreement alone does not count (Alden's counterexample: different claim and content mistakes
on two of 6 items give 5/6 on each component and 4/6 jointly). **Pass rule (frozen):** for every category, at each
placement separately, full-tuple matches on at least 5 of its 6 items; leakage flips (any tuple element changed by the
question alone) on at most 1 of 12. Component accuracies and the (Q-claim no AND content+) conjunction are reported, not
gated. **If it fails: stop, report per-category errors,
PARK the both-head assay.** No reader swap after seeing results; a second reader (Haiku) is run beside it and reported,
never used to choose.

## 3. One bounded screen, six configurations at most (Alden #4 step 2, #6)

A configuration is one (self prompt, content prompt) pair. Six, frozen here, in this order (tie-break order):

1. "Introduce yourself." / "Write a short paragraph about anything you like."
2. "Tell me about yourself." / "What did you do today?"
3. "Who are you?" / "Describe your perfect day." (pilot 2's own S0/C0 prompts; development data, not fresh)
4. "Describe yourself in a few sentences." / "Describe what you can see around you."
5. "What should I call you, and what are you like?" / "Write a short diary entry."
6. "Say hello and tell me who I'm talking to." / "Tell me a short story."

No concept words, no role instructions. A and B arms as v4 (A: the unpushed model on "Tell me about a <persona>." and
"Describe a <persona>."; B: the unpushed answer to the configuration's content prompt).

For each configuration, on all 18 original neurons (and reported on the 8 of 92b): generate the pushed self and content
answers (greedy, float32, eager, Part 99 pushes as v4), apply the development inclusion rule of section 3a, and count
joint passes. **Selection:** the configuration with the most joint passes on the 18; ties to
fewer unresolved labels, then to the lower number above. One configuration for all neurons; no per-neuron choice. No head
effect is computed until the cohort is frozen.

**Confirmation family, reserved now and never used to choose:** self "If you had to explain what you are to a child,
what would you say?" / content "Write a postcard to a friend." It replaces v4's S1/C1 (which are development data now).
It is generated and read only after the winner is committed (section 3b), and it never changes which configuration won.

Configuration 3 is one openly seen development candidate, not confirmation and not a hidden seventh arm. Its texts are
REGENERATED under the v5 settings (the record says whether they are byte-identical to pilot 2's) and read by the new
reader on the new prefix rule; pilot 2's labels are not reused for it.

### 3a. v4's inclusion criteria, mapped (Alden #8 point 2)

| v4 criterion | in v5 |
|---|---|
| (i) Part 92 q0 `self_claim` true | DROPPED from inclusion: an old single-question label. Reported beside every table. |
| (ii) Part 92 q3 `shows_persona` true, `self_claim` false | DROPPED from inclusion, same reason. Reported beside. |
| (iii) reader on S1 claims, C1 shows without claiming | REPLACED by the configuration's own S and C arms (section 1); S1 and C1 are development data. |
| (iv) all four margins positive, no head off | SPLIT: the two DEVELOPMENT margins, `self` (S against the A texts under the configuration's self prompt) and `cont` (C against B under its content prompt), must be positive for inclusion. `self'` and `cont'` move to the confirmation family and are never part of selection. |
| (v) reader finds the contrast inside K | REPLACED by the section 1 arm rules on the configuration's texts. |

**Development inclusion rule:** a neuron passes for a configuration if its S, both A, C and B arms meet section 1 and
its two development margins are positive. Nothing else.

### 3b. Order, frozen

1. Construction gold (section 2). Fail: PARK.
2. All six configurations scored on development data only. All six outcomes, with every per-criterion count, are kept.
3. The winner is chosen by section 3's rule and COMMITTED, with its development cohort and denominators, before any
   confirmation text exists.
4. The bars of section 4 are applied to that development cohort. Missed: PARK.
5. Only then is the confirmation family generated and read, once, for the winner only.
6. **Confirmation is a prospective PARK gate on the cohort, not a per-neuron filter.** The head cohort stays the
   development cohort; no neuron is added or removed by confirmation. If fewer than half of the development cohort
   (rounded up) pass the development inclusion rule on the confirmation texts, the assay is PARKED and published as an
   assay result. Example (Alden's): 10 selected with 5 of the shared 8, 2 of 10 pass confirmation: 2 < 5, PARK, with
   every stage's denominators reported. **The runner-up configuration is never chosen after a confirmation failure.**
   The gate is a prospective minimum-transfer rule, "at least half" (for a cohort of 10 to 18: 5, 6, 6, 7, 7, 8, 8, 9,
   9), not a significance test (Alden #10). The shared-eight denominator and its confirmation pass count are always
   reported beside the overall ones, and a global pass is never read as the shared subset confirming: all five
   non-shared confirming and all five shared failing passes 5 of 10 and confirms nothing about the shared eight.
7. If confirmation passes, the head scan runs on the full development cohort; `self'` and `cont'` and the held-out
   head-effect reliability are computed on the confirmation texts for that whole cohort, with each neuron's
   confirmation pass reported beside.

## 4. Bars and the park rule (Alden #4 point 3, #6)

Unchanged from v4: at least 10 of 18 included, and at least 5 of the shared 8, both counted on the winner's
DEVELOPMENT cohort (section 3b step 4). The cohort and its denominators are frozen in a committed file before any
confirmation text or head is scored. **If the selected configuration does not meet both bars, the
both-head assay is PARKED:** the screen's counts, the reader validation and the per-category errors are published as an
assay result, with no head scan, no larger prompt sweep, no softer truth table.

## 5. What a pass would mean

Conditional only (Alden #6): these neurons express the two frozen constructs under this selected assay, and then a
head's ablation affects those readouts. Not a rehabilitation of pilot 2, not a prevalence claim over neurons, not proof
that a passing mention is a persona.

## 6. Cost

Reader validation: about 8 x 6 x 2 + 24 items, two readers, cheap. Screen: 6 configurations x (36 pushed + 1 unpushed)
generations plus 2 x 18 about-texts once, 40 new tokens each, float32 on the Arc or CPU, minutes; reader calls about
6 x 18 x 6 arms. The head scan, if reached, is v4's.

## 7. Predictions (written 2026-11-20 13:33 CT by Claude, Opus 5.5, before any step-2 text exists)

Written after the construction gold passed (6ed94f08) and before any configuration text, margin or label. Pilot 2's
inclusion counts (council #2, #5) were seen; nothing generated under v5's prompts has been.

- **180v5.1 — the assay parks at step 4.** The winning configuration includes fewer than 10 of 18 on its development
  cohort, or fewer than 5 of the shared 8. Confidence 0.6. Reason: pilot 2's joint S1/A0/A1 pass was 2 of 18, and
  v5's arm rules are cleaner but not looser for the S and C arms. Withdrawn if the bars pass.
- **180v5.2 — B is almost never the failing arm.** In every configuration, the unpushed content answer B fails
  (content+ yes) for at most 1 of the 18 neurons. Confidence 0.85. B is one unpushed text read under each persona, so
  a failure is a reader seeing a persona in an ordinary answer.
- **180v5.3 — concept neurons pass more often than slot neurons in the winning configuration**, by at least 2 neurons
  in the count of passes (Part 92 kinds). Confidence 0.6. Reason: a pushed slot neuron tends to speak as its persona
  everywhere, which fails the C arm's claim-no.
- **180v5.4 — among slot neurons that fail in the winning configuration, the most frequent single failing arm is C with
  claim yes.** Confidence 0.55. Withdrawn as a reading of mechanism if fewer than 4 slot neurons fail.
- **180v5.5 — configuration 3 (pilot 2's own prompts) is not the winner.** Confidence 0.55. It is the one development
  candidate already seen in pilot 2 and was chosen then for a different assay.

Each is scored on the step-2 outputs exactly as written. A pattern found afterwards is labelled exploratory.

## Amendment 1 (2026-11-20 13:53 CT, Claude, before any step-2 text): five readings the build had to choose, now fixed

The step-2 builder (DeepSeek V4.1 Flash worker, report part180/REPORT-v5-step2.md) found five places where this design
did not say enough. Each is fixed here before any configuration text, margin or label exists:

1. **`self` with two A texts:** `self = min(LP(S) - LP(A0), LP(S) - LP(A1))` under the configuration's self prompt,
   so S must beat both about-texts. The mean is recorded beside as `self_mean` and never used for inclusion.
2. **S's prefix:** S belongs to two pairs, so it is read once at `min(len(S), len(A0), len(A1), 40)` answer tokens, the
   prefix inside K for both self margins; each row records `k_by_pair`.
3. **B is read per neuron:** one unpushed B text per configuration, read under each of the 18 personas (108 items), at
   the configuration's shared K `min(len(C), len(B), 40)`.
4. **Margins are teacher-forced with the neuron's push on every row, B included**, as in v4 (PART180-DESIGN.md: each a
   mean per-token log-probability of a fixed text with the neuron pushed). "Unpushed" describes how B was generated.
5. **Batch shape:** every generation chunk is [32, the plan's maximum prompt width]; short chunks are padded with
   duplicate rows whose outputs are discarded; the shape is recorded per row. A resumed run may place a row in a
   different chunk, so a resume is not bit-identical to an unbroken run; the record says whether a resume happened.

Alden's completeness rules (council #12) are in the scorer: the expected (configuration, neuron, arm) grid comes from
the frozen lists, and a missing cell, a duplicate id or a missing margin is an error, not a gap.

## Amendment 2 (2026-11-20 15:15 CT, Claude, after both readers finished and BEFORE any classification, inclusion count or winner was computed)

**What happened.** The primary reader (DeepSeek V4.1 Flash, max_tokens 4000) returned no label for 6 of 540 cells
(`v5-c1-p92-40-S`, `v5-c2-p92-52-S`, `v5-c4-p92-45-S`, `v5-c5-p92-08-S`, `v5-c5-p92-49-S`,
`v5-c6-p92-39-A0`): all three attempts came back with EMPTY text, the known case where the model spends its whole cap
on reasoning. The classifier refuses to run on a cell with no label, and the design had no rule for it.

**Rule, fixed now.** Those 6 cells (and only cells with no label) are re-read once more with the same prompt and model
at max_tokens 16000, up to 3 attempts. The original reader file is kept unchanged; the repaired rows are written to
`reader-deepseek-repair.jsonl` and the merged labels to `reader-deepseek-final.jsonl`, which the classifier reads. A
cell still without a label after the repair counts as a FAIL on every arm it feeds and is reported by id. No cell with
a label is re-read, and no label is changed.

**Also recorded.** (1) The two readers' first launch shared one output file because `reader180v5.py`'s `--out`
defaults to `reader-deepseek.jsonl` for both readers; the mixed file is kept as
`reader-MIXED-both-readers-default-out.jsonl` and both readers were rerun from empty with explicit outputs. (2) The
DeepSeek read used a parallel driver (8 items in flight) calling the same `build_items`, `read_one` and row shape;
the registered `cmd_read` was then run on the result, found 0 pending, re-validated it and wrote the reader binding.
