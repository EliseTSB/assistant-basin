# Goal: climb the neuron-explanation prompts overnight (Part 1e)

You are improving three prompts used in an interpretability study of Qwen3-0.6B's MLP neurons. Each "unit" is either one
neuron or a random direction in the same layer (you are NOT told which, and must not try to find out). Work until
**07:45 America/Chicago** (check with `TZ=America/Chicago date`), then write the deliverables below and stop.

## The pipeline you are scoring

1. **Explainer.** Luna (gpt-6-luna) reads the unit's 20 strongest contexts from a training slice of Wikipedia text (the
   active token is marked like ⟦this⟧) and writes a description. Your prompt: `explainer.txt` (must contain `{snips}`
   and ask for a JSON object with a `"description"` field; it may also ask for `"verdict"` and/or `"p_one"`, see 4).
   Effort low/medium/high is set in `candidate.json`. Descriptions are clipped to `explainer.max_words` (at most 80).
2. **Simulator.** Jev (TypeSafe's calibrated judgment model, docs: https://docs.typesafe.ai/primitives/noul.md) reads
   the description and ONE held-out snippet and answers your noul question(s). Its state is
   `{"description": ..., "snippet": ...}` (plus `"examples"`: the unit's first k training contexts when
   `simulator.include_examples` is k > 0). The unit's predicted firing p is the weighted mean of your questions'
   probabilities (`simulator.weights`).
3. **Score.** Per unit: Spearman between p and the unit's TRUE activation over its 30 held-out contexts (10 strongest,
   10 mid-strength, 10 random; from text the explainer never saw). Primary score: **mean per-unit Spearman**. Also
   reported: AUROC strongest-vs-random and mid-vs-random.
4. **Judge.** P(one job) per unit: from the explainer's own answer (`judge.engine: "explainer"`: `"p_one"` in [0,1] if
   given, else `"verdict"` one=1, two_related=0.33, other=0) or from Jev questions over the 20 snippets
   (`judge.engine: "jev"`, state `{"snippets": ...}` plus `"description"` if `judge.include_description` is true).
   Judge score: **`judge_auroc_predictable`** (does P(one) pick out the units whose held-out firing this candidate's
   pipeline predicts, Spearman >= 0.4), subject to **`judge_floor` <= 0.04** (the share of 25 sets of RANDOM text
   contexts called one job: at most 1 of 25).

## How to run

A candidate is a directory with `explainer.txt` and `candidate.json`. The current best starts as a copy of the
baseline (Part 1's prompts): `cp -r /home/elisepc/neuron-climb/harness/baseline best`.

    python3 -I /home/elisepc/neuron-climb/harness/climb_eval.py <candidate> --set search --stage sim --ref best --note "what changed"

- `--set search` = 60 units (fast screen); `--set dev` = all 149 units (use it to confirm). `--stage judge|all` adds
  the judge. `--dry` shows how many calls are not cached yet.
- Every eval appends to `ledger.jsonl`; per-unit rows (descriptions, p, activations) go to `runs/<hash>-<set>.json`.
  Read the rows of units that score badly: that is where the ideas come from.
- Calls are cached, so re-scoring is free; changing the explainer prompt or effort means new Luna calls for every unit,
  and any change to the description or Jev question means new Jev calls (about 15K input tokens per unit on dev).
- Run one eval at a time (the harness refuses a second). An eval of the full dev set takes a few minutes.

**Promotion rule.** A candidate replaces `best` only when its `vs_ref` on `--set dev` has a 95% interval whose LOWER
bound is above 0 (the search set is too small to decide on: its interval is about ±0.07). Keep `best` a plain copy of
the winning candidate and note every promotion in STATUS.md.

## Rules

- Do not edit anything outside your working directory. The harness, data and baseline are read-only on purpose.
- Do NOT read `/home/elisepc/assistant-basin/results/part1e-climb-split.json`, anything under `/mnt/e/neuron-probe/`,
  or any other copy of Part 1's units: the held-back test units live there, and the final score is taken on them.
- Do not open, print or search credential files (`.env`, `auth.json`, keys). The harness needs none.
- Prompts must be generic. Never paste dev contexts (or close paraphrases of specific ones) into a prompt; the harness
  refuses any 40-character run copied from the data. Few-shot examples, if you want them, must be invented.
- Only noul questions are scored. Keep descriptions honest: a description is a short account of what the unit
  responds to, not a list of the snippets.
- `include_examples > 0` is a separate, exploratory track: report its best separately. The PRIMARY deliverable has
  `include_examples: 0`, because the study asks whether a description explains the unit.
- Each climber has its own Jev budget (about 30M input tokens, enforced by the relay). Exit code 3 = your budget is
  spent; 4 = Jev refused (stop calling it); 5 = Luna's usage limit (wait about 20 minutes and try again, no more often).
  On 3 or 4, stop and write the deliverables.

## Deliverables (by 07:45 CT)

- `final/sim/` — the best candidate with `include_examples: 0`, confirmed on dev.
- `final/judge/` — the best candidate for the judge (it may equal `final/sim/`), with its dev `--stage all` scores.
- `final/examples/` — only if the examples track beat `final/sim/`.
- `FINAL.md` — the dev scores of baseline and each final candidate, what you tried, what helped, what did not, and
  anything that looked like overfitting. Keep `STATUS.md` (your running notes) current as you go, so the work survives
  a restart of your session.

Ideas, none required: describe what the unit does NOT respond to as well as what it does; say whether the unit reacts
to the marked token itself, the words before it, or the likely next word; ask Jev more than one question (identity of
the token, the surrounding context, the predicted next word) and weight them; make the noul criteria graded ("strongly"
vs "weakly"); trade Luna's effort for more experiments; ask the explainer for a calibrated `p_one` with an explicit
rubric for random-looking snippet sets.
