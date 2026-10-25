# Cold adversarial review: p37-build (c78c778..92c2f6e)

**Verdict: SHIP-WITH-FIXES.** The record construction, the real-tokenizer readout positions, the
question set and the fold machinery are sound and I could not break them. But three things must be
fixed before the run produces a number anyone should read: (1) the minimal-edit cell is not a minimal
edit — on the real tokenizer it replaces the answer's content word with a function word in 5 of 8
sampled answers, so it is a fluency/noise probe, not a one-word edit; (2) 37.11's verdict is computed
from `abs(memory_effect) - abs(auth_effect)`, which passes when the label moves the *authorship*
phrasing more than the memory phrasing — the reverse of the registered text; (3) the test suite never
touches the scoring rules, so mutants of 37.12, 37.13 and the memory/authorship split all stay green
(13/13). None of these is in the fold/leakage core, which is why this is not DO-NOT-SHIP.

Everything below is measured with scripts in `scripts/` in this directory, run against the real code
from `/home/elisepc/green/worktrees/ab-part37`, with the pinned Qwen3-1.7B / OLMo-2-1B-Instruct /
SmolLM3-3B tokenizers from `/mnt/e/neuron-probe/hf/hub` (read-only, no model weights loaded, no
network, no live service). Python: `/home/elisepc/neuron-probe/.venv/bin/python` (sklearn 1.9.1,
torch 2.14.1+cpu). Shipped suite as-is: `13 passed` (all 13, including the real-tokenizer test).

---

## Findings, by severity

### HIGH

#### H1. The "minimal edit" cell is not a minimal edit; it is a content→function-word replacement. INTRODUCED.
`collect37.py:245-261` (`minimal_edit`) picks the word nearest the middle with ≥4 letters and replaces
it with the **first** `EDIT_WORDS` entry within ±1 character (`collect37.py:252-253`). `EDIT_WORDS`
(`collect37.py:32-41`) starts `with, have, from, then, some, when, were, make, good, nice`, so every
4- or 5-letter content word becomes one of those.

Measured (`scripts/s15_edit_ident.py`, pinned Qwen3 tokenizer):

```
     'while' -> 'with'       FUNCTION WORD  | Toast it lightly, then add butter with it is still warm so it melts evenly.
     'sunny' -> 'with'       FUNCTION WORD  | A gentle breeze makes the with afternoon feel calm and pleasant.
   'because' -> 'always'       | I usually walk to the corner shop always it is close and the weather is fine.
   'slowly' -> 'small'        | Bread goes stale because the starch small turns back into a firm crystal.
     'check' -> 'with'       FUNCTION WORD  | Keep the milk in the fridge and with the date before you pour it.
    'asleep' -> 'small'        | Reading before bed helps me fall small faster than looking at a screen.
      'cook' -> 'with'       FUNCTION WORD  | The best way to learn a recipe is to with it twice in the same week.
      'good' -> 'with'       FUNCTION WORD  | Take the stairs when you can, it is with for your legs and your mood.
5/8 edits replaced a content word with a function word
```

Why it matters: Amendment 1 point 5 defines this cell as "the reading model's own greedy answer with
one word replaced by a fixed rule (the word nearest the middle with at least four letters, replaced by
a different common word of similar length from a fixed list)". The rule as implemented produces
ungrammatical text in most answers, so the `edit` cell measures "how the reader treats a broken
sentence" — which is a fluency signal, not the "one word changed, authorship unchanged" contrast
37.3-style projections are meant to isolate. Any difference between `edit` and `own_g` is confounded
with fluency. Smallest fix: choose the replacement from the same part-of-speech-ish class — e.g. pick
from `EDIT_WORDS` by *longest* common prefix or by a small hand-written content-word list per length,
and assert in a test that the replacement is not in the first ten (function-word) entries.

#### H2. 37.11's verdict can pass with the memory phrasing moving *less* than the authorship phrasing. INTRODUCED.
`analyze37.py:760-762`:
```python
p11_pass = (None if memory_effect is None or auth_effect is None else
            abs(memory_effect) - abs(auth_effect) >= 0.10 and
            val(qwen, "stated", "neutral_between_self_and_other") is True)
```
Registered text (PREDICTIONS.md 37.11): "the label moves the **memory** phrasing **at least 10 points
more** than the **authorship** phrasing". The code takes `abs()` of both. Measured
(`scripts/s10_scoring.py`, driving the real `_prediction_results`):

```
37.11 with memory effect -0.5 (label moves memory 40 points LESS than authorship), neutral between: PASS
```

i.e. memory effect −0.50, authorship effect +0.10 → `abs(-0.50) - abs(0.10) = 0.40 ≥ 0.10` → PASS,
even though the label moved the memory phrasing 40 points *less* (indeed, the wrong way) than the
authorship phrasing. Why it matters: this is the one scored prediction whose direction the code
inverts, and it is scored on Qwen only, so it is the only reading of Weaver's point 2 the run will
report. Smallest fix: `memory_effect - auth_effect >= 0.10` (signed), and add a test with
memory=−0.5, auth=+0.1 that must be FAIL.

#### H3. The test suite does not test any scoring rule; three scoring mutants stay green. INTRODUCED.
`test_part37.py` exercises records, prompts, token sets, the fold checker and `verdict()`, but never
`_prediction_results`. Measured (`scripts/s14_sabotage.py`, mutating the real module in-process):

```
baseline: (13, 0, 0)
mutant A (37.12 folds scored at 0.50): (13, 0, 0)
mutant B (37.13 accepts any >= 0.60): (13, 0, 0)
mutant C (memory phrasing scored as authorship): (13, 0, 0)
```

Mutant A is exactly the error a careless reader would make (37.12's registered floor is "balanced
accuracy 0.80 or more"; the probe's own `_fold_fit` uses `probabilities >= 0.5`). The commit message
claims "a probe-split test that the leaky-folds mutant reds ... readout-index and KFold mutants both
caught" — I reproduced both and they are caught (`scripts/s22_mutants.py`: readout off-by-one → 3
failures; plain KFold → 1 failure). But the scoring half of the module has no mutant coverage at all.
Why it matters: the report's "13/13 host" reads as coverage of the whole pipeline; it is coverage of
the record half. Smallest fix: add a `_prediction_results` test with hand-built reader dicts sitting
on each registered threshold (the `scripts/s10_scoring.py` / `s11_scoring2.py` fixtures are ready to
copy), one per scored prediction.

### MEDIUM

#### M1. The user-slot frame puts the question immediately before the span, in the same user turn, so the "role" contrast is also a "what precedes the span" contrast. INTRODUCED.
`collect37.py:323`: `f"{question}\n\n{FOLLOWUP_PREFIX}\n\n{span}"`. Measured with the pinned Qwen3
tokenizer (`scripts/s5_span_has_question.py`):

```
assistant/base: tokens before span start: '<|im_start|>user\nHow should I eat bread?<|im_end|>\n<|im_start|>assistant\n'
user/base: tokens before span start: '<|im_start|>user\nHow should I eat bread?\n\nHere is an answer to this:\n\n'
```

In the assistant cell the span follows a turn boundary; in the user cell it follows the question text
with no boundary. The span-last token is `.` in both cells (so the readout position itself is right),
but the residual at that position differs in what it has just read: the assistant cell has just read
the span after a `<|im_start|>assistant` marker; the user cell has just read the question, then
`Here is an answer to this:`, then the span. 37.1 ("role ... above 0.9 at some middle layer") and
37.8's cosine (authorship vs role direction) are scored on this contrast. The design's C1 is "the same
span placed in the user slot versus the assistant slot" — the same span is placed, but not in the same
surroundings. Why it matters: a high role AUROC (37.1) is expected and will be read as "role is
linearly readable", but part of it can be "the question is in the context" — which the assistant cell
also has, just further back. Smallest fix: give the user cell the same boundary structure, e.g.
`[{"role":"user","content":question},{"role":"assistant","content":"Here is an answer to this:"},
{"role":"user","content":span},{"role":"assistant","content":"Noted."},{"role":"user","content":FOLLOWUP}]`
so the span sits at the start of a turn in both cells, or report the assistant-vs-user contrast with
the question text held fixed in both cells.

#### M2. The label frame's own/other cells differ in the span's *position within the message* and in the header, not only in the header. INTRODUCED.
`collect37.py:313-319`: the label is `f"{header}:\n\n{span}"`, so the span's last token sits after the
header in both cells — that part is fine. But the user-slot label frame is
`[user: "Your earlier notes:\n\n<span>", assistant: "Noted.", user: FOLLOWUP]` while the assistant-slot
label frame is `[user: "Please show me the notes.", assistant: "Your earlier notes:\n\n<span>", user: FOLLOWUP]`.
So the "label" contrast (37.3, 37.11) is measured across a frame that also changes the preceding turn
("Please show me the notes." vs nothing) and the slot — but 37.3's registered claim is "the label alone
... moves the projection ... less than half as far as swapping the actual author does". The code's
`c3` (`analyze37.py:547-560`) pools both slots and both header variants and compares the mean
projection under `self` vs `other` headers to the mean under `own` vs `other` authors — it does not
hold slot fixed. Since the slot changes the preceding context (M1), the label effect is contaminated
by the slot mix if the slot and header are not balanced — they are balanced by construction (both
slots × both authors × all headers), so the *difference of means* is slot-balanced, but the variance
is not. This is a weaker finding than M1; the smallest fix is to report the label effect per slot as
well as pooled, which the code already has the rows for.

#### M3. `_prediction_results` scores 37.8's cosine from a role direction fitted at the *authorship* best layer, not the role best layer. INTRODUCED (as a defect only if the reader reads "at the best layer" as the role's).
`analyze37.py:541-545`: `role_best = int(role_span["best_layer"])` is computed and then **not used** —
`role_direction = _fit_direction(store, role_rows, 0, author_best, "raw")`. The result dict
(`analyze37.py:691-693`) reports `"raw_direction_layer": author_best, "role_best_layer": role_best`,
so the reader can see both, and 37.8's registered text says "cosine with the role direction 0.5 or more
**at the best layer**" (ambiguous which). The code's own comment at `analyze37.py:562-563` says
"Projection contrasts use the all-question authorship direction". Smallest fix: report both cosines
(at `author_best` and at `role_best`) and state which one 37.8 uses.

#### M4. The SmolLM3 prompt carries identity text, and the null text is generated under it. INTRODUCED (by the template, not chosen by the author — but not screened).
`collect37.render_generation_prompt` puts `/no_think` as the system message; the pinned SmolLM3-3B
template then renders its own system block. Measured (`scripts/s18_smol_prompt.py`, `s19_smol_screen.py`):

```
'<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: 25 October 2026\nReasoning Mode: /no_think\n\n## Custom Instructions\n\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n<|im_start|>user\nWhy does bread go stale?<|im_end|>\n<|im_start|>assistant\n</think>\n'
identity terms in the SmolLM3 prompt: ['SmolLM', 'Hugging Face', 'assistant']
```

Amendment 1 point 1 excludes identity text from the questions and screens every generated answer;
the screen (`collect37.screen_rows`) only sees the completion text, never the prompt. Why it matters:
the `null_smol` cell is the readout null (37.15), and its text is produced by a model told it is "an AI
assistant named SmolLM, trained by Hugging Face"; if the reader's stated-yes is influenced by
identity-styled text, the null floor is inflated. Smallest fix: screen the rendered prompt too (run
`identity_matches` on `render_generation_prompt` output and record the matches in the generation
manifest), and note in the report that the null text's prompt is identity-styled.

### LOW

#### L1. `_length_position_probe` is wired but never used by any verdict, and its inputs are constant within a question. INTRODUCED.
`analyze37.py:503-510` computes `length_position_only` for `role_pooled` and
`authorship_assistant_primary`; no verdict reads it (grep `length_position_only` → only the two
assignment sites). Measured (`scripts/s20_length_probe.py`): the probe runs and returns AUROC 0.5 on
synthetic data where `span_token_count` varies with the question and not with the label — which is
correct behaviour (length is question-level), but it means the control can only ever return ~0.5 for
the own/other contrast, because the design length-matches every span to the same `target` per
question (`collect37.py:368, 408`). A control that cannot fail is not a control. Prime's point 8 asks
for "a length-and-position-only probe"; the honest version is to report it as "expected 0.5 by
construction" rather than as a control. Smallest fix: either drop it from the output or label it
"cannot discriminate by construction (length is matched per question)".

#### L2. The pilot gate compares each layer to the null's 95th percentile rather than to the null's mean; the registered stop rule is "within 0.05 of the label-shuffled control's". INTRODUCED.
`analyze37.py:854-861`. Measured (`scripts/s23_pilot_gate.py`): real 0.65 vs null 95th 0.55 → does
**not** stop (correct, the real probe is above the null); real 0.52 vs null 95th 0.55 → stops
(correct); a null layer that produced no value (`None`, e.g. a permutation fold with one class) →
does **not** stop (fails open — the gate can never stop if any layer's null is missing). The
registered text says "within 0.05 of the label-shuffled control's" without saying which statistic;
using the 95th percentile is more conservative than the mean and is defensible, but the `None` path
silently disables the stop. Smallest fix: treat a `None` null layer as "not within 0.05" (stop) or
raise, and record which statistic the gate uses in the pilot JSON.

#### L3. `build_records` does not check that every kept question produced a full set of generations. INTRODUCED.
`collect37.py:351-355` raises on a missing `(qid, model, kind)` row, which is good; but `screen_rows`
(`collect37.py:219-238`) accepts any set of question ids and `build_records` takes `kept` from the
screen file without checking it against `questions.json` beyond "unknown ids" (`collect37.py:346-348`).
A pilot kept file (20 questions) fed to a full `build-records` would build 20-question records with no
warning; `analyze37` likewise never checks the capture covers the full set (measured,
`scripts/s21_question_count.py`: a 6-question capture is accepted and scored, `question_count: 6`).
The runbook passes the right files, so this is a guard, not a live bug. Smallest fix: in
`command_build_records`, refuse when `len(kept)` is neither 20 (pilot) nor 100 (full) unless
`--allow-partial` is given.

#### L4. `_capture_one` uses `tokenizer.pad_token_id` for `pad_token_id` in generation but never sets it for the generation tokenizer. PRE-EXISTING pattern, not reached here.
`collect37.py:166-167` sets `pad_token` only in `load_model`; `command_generate` uses `load_model`, so
it is set. Not a finding — noted only because I checked it.

---

## What I checked and found sound

- **Readout positions, on the real tokenizers.** `locate_readout_positions` with the pinned Qwen3 and
  OLMo tokenizers, all four frames (`scripts/s4_render.py`, `s5_span_has_question.py`): span-last
  decodes to `.` in every frame; the token after it is the turn end (`<|im_end|>` / `<|endoftext|>`),
  never part of the span; follow-up first decodes to `Thanks`. The shipped real-tokenizer test
  (`test_part37.py:289-310`) passes on this host and would catch an off-by-one (mutant → 3 failures).
- **The host fix for the edit cell.** `DERIVED_FROM_OWN_G` (`collect37.py:197, 400-406`) keeps the
  edit and shuffle text uncut, so a token-count change cannot empty the cell. The stated reason ("a
  word swap ... can change the token count by a few") does not hold on the real tokenizer — measured
  0/180 swaps changed the count (`scripts/s3_edit_tokcount.py`) — but the fix is harmless and the
  cell is no longer emptied (the fixture at `test_part37.py:201-202` exercises this).
- **Folds.** `_group_splits` uses `GroupKFold(5, shuffle=True, random_state=37)` on question ids;
  `folds_are_question_disjoint` is checked before every fit (`analyze37.py:151-152`); scaler and PLS
  are fitted on the train fold only (`analyze37.py:132-143`); the residualiser's beta is fitted on
  train only (`analyze37.py:117-122`). I could not construct a leak inside a fold.
- **The label-shuffled null.** It flips a whole question's labels together, which leaves the
  within-question own/other difference intact. I checked this is the *right* null for a within-question
  contrast (`scripts/s9_perm_null.py`): with a real within-question signal of 0.6 the null's 95th stays
  at 0.62 while the real probe reaches 0.82; with no signal both sit at ~0.6. The null tracks the
  question-level variance, which is what it should do.
- **The random-direction control.** It is a direction-in-the-data control, not a probe control (no
  folds, fitted and scored on the same rows). Measured (`scripts/s7_controls.py`, `s8_question_offset.py`):
  with no authorship signal it sits at 0.50–0.54, so it is not inflated; it is a weaker control than
  the label shuffle but not a false one.
- **Generation stops.** `_generation_stops` returns the chat-template end-of-turn for all three pinned
  tokenizers (`scripts/s17_smol_stops.py`): Qwen `[151645]` (`<|im_end|>` = eos), OLMo `[100257, 100265]`,
  SmolLM3 `[128012, 128009]`. Non-thinking is set for Qwen (`enable_thinking=False`) and SmolLM3
  (`/no_think` in the system message, which the template renders as `Reasoning Mode: /no_think`).
- **The question set.** All 100 ids and the 35/35/30 mix check out; none of the 100 questions matches
  an identity term or any of the 40+ exclusion terms I tried (`scripts/s12_questions.py`: 0 flagged).
- **Identity screen scope.** It drops a question if *any* model's answer matches, across every file
  passed to `--gens`, and the runbook passes all three (`scripts/s1`/`s2` runs; `RUNBOOK.md:25,34`).
- **Partial artifacts.** A capture with a missing or empty `metrics.jsonl` fails with a clear
  `AnalysisError` / `FileNotFoundError`, and `main()` catches both (`scripts/s13_partial.py`).
- **Scoring thresholds I checked and found faithful to the registered text:** 37.1 (both readers,
  middle third, >0.9), 37.2 (>0.7 before, <0.6 after), 37.9 (signed `>= 0.5`, so a *lower* own-greedy
  projection FAILs — `scripts/s11_scoring2.py`), 37.12 (folds scored at 0.80, not 0.5 — correct in the
  shipped code), 37.13 (0.60–0.75 band, so 0.95 and 0.40 both FAIL), 37.14, 37.15 (0.10–0.25 on Qwen).
- **The commit message's mutant claims.** Readout off-by-one and plain-KFold mutants both red the
  suite (`scripts/s22_mutants.py`), as claimed.

## How to reproduce

```
cd /home/elisepc/green/Journal/scratch/review-p37-build-ds-r1-2026-10-25/scripts
/home/elisepc/neuron-probe/.venv/bin/python s15_edit_ident.py      # H1
/home/elisepc/neuron-probe/.venv/bin/python s10_scoring.py         # H2
/home/elisepc/neuron-probe/.venv/bin/python s14_sabotage.py        # H3
/home/elisepc/neuron-probe/.venv/bin/python s5_span_has_question.py # M1
/home/elisepc/neuron-probe/.venv/bin/python s19_smol_screen.py     # M4
/home/elisepc/neuron-probe/.venv/bin/python s20_length_probe.py    # L1
/home/elisepc/neuron-probe/.venv/bin/python s23_pilot_gate.py      # L2
```
