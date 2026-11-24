# Part 218 build notes — runner, scorer, prompts, tests, smoke

Built 2026-11-24 in the worktree `/home/elisepc/green/worktrees/ab-part218` (branch `part218-build`).
No git write was run. Frozen files were not touched: `DESIGN.md`, `attrib218.py`, `gold218.jsonl`,
`pins.json`, `test_attrib218.py`.

## What exists

| file | bytes | what it is |
| --- | ---: | --- |
| `part218/run218.py` | 45,142 | the runner: `--mode pilot`, `--mode full --model {SFT,DPO,Instruct}`, `--selftest`, `--tiny <config-dir>` |
| `part218/score218.py` | 50,420 | the scorer: per-run quantities, pilot trim/stop rules, predictions 218.1–218.8, bootstrap, binomial tests, audit sample |
| `part218/prompts218.json` | 1,526 | Part 173's frozen NEW44B, 44 strings, generated from `part173/prompts173.py` (not retyped) |
| `part218/test_run218.py` | 16,145 | 10 tests, including the tiny-model end-to-end smoke |
| `part218/test_score218.py` | 28,676 | 24 tests on varied, non-square fixtures |

The runner is a single script: it imports only torch, transformers and the standard library. It reads
`prompts218.json` and `pins.json` from its own folder, so copying `part218/` (plus the prompts file) to a
Colab VM is enough. `attrib218` is imported only by the scorer.

### What the runner does, point by point

- **Models**: SFT `0d85a3d0…`, DPO `c4b04859…`, Instruct `48d788ec…` from `pins.json`, bf16 on CUDA by
  default, checkpoint chat template, no system prompt, greedy, `max_new_tokens=48`.
- **Directions**: 20 unit Gaussian rows, seeds 0–19, float32 on the CPU, drawn **once** into
  `<out>/directions.pt` and reloaded byte for byte by every model and the pilot; both signs are built at
  use time (40 rows). A stored file with a different hidden size is an error, never a silent redraw.
- **Batches**: one prompt × 41 rows (40 pushed + zero) in full mode, × 11 rows (10 pushed + zero) in the
  pilot. No padding.
- **Dose**: `d × median(residual-output norm at that layer over all 44 prompts, position 0 excluded)`,
  computed per model with a forward hook. The push is added to the decoder block's output at every
  position, prefill and decode.
- **No-hook null**: 44 unhooked batches of the same shape at the start and again at the end; each
  hooked batch's zero row is compared with the start null (token ids and first-step logits, exactly).
  The end null is compared with the start null and with each prompt's last hooked zero row. Hooks are
  checked to be gone before the end null. A mismatch stops the model, writes `run.json` and exits 3.
- **Resume**: `answers.jsonl` is read first; a batch is skipped when all its direction ids are present
  under `(model, layer, dose, prompt_index)`. A crash inside a batch is compacted (last write wins) and
  that one batch is regenerated. A run with no batches left still recomputes both nulls and keeps the
  previous process's null/push records.
- **Writes**: `<out>/<pilot|SFT|DPO|Instruct>/answers.jsonl` (one line per generated row with model,
  revision, layer, dose, seed, sign, direction id, prompt index, row index, token ids, decoded text,
  `n_new_tokens`, `max_new_tokens`, `hit_cap`) and `run.json` (pins, versions, device, dtype, attention
  implementation, medians, directions sha256, planned/written counts, null receipts, push probe, status).

### What the scorer does

Reads the run root, applies `attrib218` to every text, and computes:

- unpushed answer = majority over a prompt's zero rows, with "not unanimous ⇒ not `self`" (F7);
- eligible prompts and the 30-of-44 gate;
- `supported` (broken ≤ floor(10% of eligible)), `moved` (supported + ≥3 other_maker), `led by`
  (most frequent other_maker family; ties not led);
- the read rung per (model, layer): largest d with ≥36/40 supported (9/10 in the pilot);
- every slice on the ladder, with refusal counts by family, distinct answer strings by family and behind
  each led count, median length (tokens and words) and cap fraction;
- the pilot's stop rule and trim rule;
- predictions 218.1–218.8 with `held` / `failed` / `not read` + reason, including the 10,000-draw paired
  bootstrap for 218.6 (seed 218) and the one-sided exact binomial test of each leading family against
  0.1;
- the gold gate: `gold218.jsonl` is re-classified (87/87 expected); a mismatch records every prediction
  as not read;
- the reader-audit sample (`audit218.jsonl`), labels left to Claude.

## How to run it on Colab

Use **one** `--out` root so `directions.pt` is shared byte for byte:

```bash
# 1. pilot (Instruct, layer 5, the 9-rung pilot ladder): 4,356 rows
python part218/run218.py --mode pilot --out /content/out218

# 2. read the trim rule, then run the three models with the kept ladder
python part218/score218.py --run-dir /content/out218        # pilot.trim_rule.kept_ladder
python part218/run218.py --mode full --model SFT      --out /content/out218
python part218/run218.py --mode full --model DPO      --out /content/out218
python part218/run218.py --mode full --model Instruct --out /content/out218

# after a trim, e.g. kept = 0.25,0.5,0.75,1.0,1.5:
python part218/run218.py --mode full --model SFT --doses 0.25,0.5,0.75,1.0,1.5 --out /content/out218

# 3. score
python part218/score218.py --run-dir /content/out218
```

Every command is resumable: rerun it after a kill and it skips finished batches. `--selftest` runs a
two-layer random model on the CPU (hook arithmetic, doses, planned counts) in a couple of seconds.
`--tiny <config-dir>` runs the real CLI against a random-weight model built from a config+tokenizer
directory (used by the smoke). Defaults: `--device cuda` if available, `--dtype bfloat16` on CUDA else
float32, `--max-new 48`.

## Smoke (PROCEDURE 5): the real command line, tiny random OLMo 2

The tiny model was built from the **real** cached Instruct config and tokenizer
(`…/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788ec…/`): `hidden_size=64`, 16 decoder
layers, `intermediate_size=256`, 4 heads, the real 100,352-token tokenizer, random weights (seed 0),
CPU float32. Commands actually run:

```bash
python -I part218/run218.py --mode pilot --tiny /tmp/part218-smoke/tiny \
    --out /tmp/part218-smoke2/out --device cpu --dtype float32 --max-new 4
python -I part218/run218.py --mode full --model Instruct --doses 0.25,1.0 \
    --tiny /tmp/part218-smoke/tiny --out /tmp/part218-smoke2/out \
    --device cpu --dtype float32 --max-new 4
python -I part218/score218.py --run-dir /tmp/part218-smoke2/out
```

`--max-new 4` (not the default 48) keeps the two CPU runs under a minute; every other code path is the
one the real run uses, and `hit_cap`/`max_new_tokens` are recorded per row so the cap fraction is honest.

Output (last lines and the `run.json` receipts):

```
pilot:    complete; rows 4356/4356; batches written 396, resumed 0; null all_ok=True
Instruct: complete; rows 7216/7216; batches written 176, resumed 0; null all_ok=True

pilot   rows_planned 4356  rows_written 4356  batches 396  layers [5]        doses 9  rows/batch 11
full    rows_planned 7216  rows_written 7216  batches 176  layers [5, 10]    doses 2  rows/batch 41
        (2 doses x 2 layers x 44 prompts x 41 rows = 7,216; the design's full count is
         2 x 7 x 44 x 41 = 25,256 per model, 75,768 for three models)

push_check (pilot):    pushed_delta_abs_max 1.565   zero_delta_abs_max 0.0   zero_row_exact True
push_check (full):     pushed_delta_abs_max 2.609   zero_delta_abs_max 0.0   zero_row_exact True
null (pilot):          hooked zero vs start 396/396 ids and logits, max |Δlogit| 0.0
                       end vs start 44/44;  end vs last hooked zero 44/44;  all_ok True
null (full):           hooked zero vs start 176/176 ids and logits, max |Δlogit| 0.0
                       end vs start 44/44;  end vs last hooked zero 44/44;  all_ok True
directions:            shape [20, 64], sha256 9cefd4a4c6222ca0…, created in the pilot,
                       reused (created_this_run False) in the full run
tokenizer:             pad <|pad|> 100277, eos <|endoftext|> 100257, unk == eos True
zero-row identity:     44/44 prompts identical across the 4 full-mode batches
hooks after generation: 0
```

The scorer ran end to end on that output and wrote `score218.json` + `audit218.jsonl`:

```
Gold set: 87/87 (attrib218 218-v2)
model            gate   read rungs  moved at rung
Instruct      0/44     L5 none, L10 none
...
every prediction: not read (gate failed: 0 of 44 prompts eligible)
Pilot: eligible 0/44; read rungs {'5': None}; stop rule triggered=True
```

That is the honest result for a random-weight model: it never says "I was built by Ai2", so nothing is
read. The smoke's job is the glue — planned counts, the null, the push, the files, the scorer — and all
of it passed. The smoke is also a pytest test (`test_tiny_model_end_to_end_smoke`, skipped when the
cached Instruct config/tokenizer is absent or `PART218_SKIP_SMOKE=1`).

## Tests and the deliberate breaks seen red

```
$ python -I -m pytest -q part218/
40 passed in 48s            # 10 run218 (incl. the smoke), 24 score218, 6 attrib218

$ python -I part218/test_attrib218.py
RESULT: PASS -- 87/87 gold rows; 6/6 sabotage checks caught; …
```

Fixtures are never constant and never square: prompts, layers, doses and directions all vary; the
end-to-end fixture writes 44 prompts × 2 doses × 2 layers × 41 rows, and the full-size probe used
44 × 2 layers × 7 doses × 41 = 25,256 rows per model.

Both test files break a **copy** in `tmp_path` (never the tracked file) and require the copy to fail the
same check the tracked module passes. The red runs were also executed by hand:

```
$ cp part218/run218.py /tmp/…/run218_broken.py
$ sed -i 's/scale = float(dose) \* float(median_norm)/scale = float(dose)/' /tmp/…/run218_broken.py
RUNNER BREAK IS RED: pushed rows must have norm d*median=5.0, got
  [0.4000000059604645, 0.4000000059604645, 0.3999999761581421, 0.3999999761581421]

$ cp part218/score218.py /tmp/…/score218_broken.py
$ sed -i 's/^MOVED_MIN = 3$/MOVED_MIN = 2/' /tmp/…/score218_broken.py
SCORER BREAK IS RED: the moved threshold must require 3 other_maker answers, got moved=2

tracked modules still GREEN on the same checks
```

The runner test also broke nothing else; the scorer test's break is the moved threshold (2 would turn a
two-answer direction into "moved", which the fixture distinguishes from three).

## Open questions for Claude (design ambiguities and the reading picked)

1. **Pilot eligibility has 9 zero rows, not 14.** The pilot is one layer × nine doses. Reading: the
   unpushed answer is the majority over the 9; a prompt is eligible only if all 9 agree and are `self`.
   F7's "14" is the full-mode count. (`unpushed_table` uses the batches the run actually has.)
2. **218.5 under a trimmed ladder.** The prediction names "all 14 batches" (2 layers × 7 doses). If the
   pilot's trim rule drops rungs, a model has 10 or 12 batches. Reading: 218.5 is `not read` with the
   reason "a model ran fewer than 14 zero-row batches". If you would rather read "all batches the run
   has", it is one condition in `prediction_218_5`.
3. **218.5 and the gate.** Reading rules: "A model that fails the gate has its predictions recorded as
   not read." Reading: 218.5 is not read when any of the three models fails the gate. 218.4 is the one
   exception: its subject *is* the gates, so a failed gate makes 218.4 `failed`, not `not read`.
4. **218.1 when no layer has 5 moved.** 218.3 has an explicit not-read clause (F6); 218.1 does not.
   Reading: `failed` (the count is observably below 5, so the prediction is false), not `not read`.
5. **218.6 point estimate vs bootstrap population.** The design computes the share on each model's
   eligible prompts but the bootstrap on the intersection. Reading: the ≥10-point comparison uses each
   model's own eligible prompts (all read rungs pooled); the CI is a paired cluster bootstrap over the
   intersection, resampling prompts and recomputing both pooled shares from the same prompts. Both the
   intersection shares and the all-eligible shares are written to `detail`.
6. **218.6 when a prompt has no other_maker answers.** Its per-prompt share is undefined. Reading: use
   the cluster bootstrap on pooled counts; draws with a zero denominator are counted as
   `undefined_draws` and skipped (`effective_draws` records how many remain). The ≥20 check uses the
   all-eligible pooled counts; the ≥25 check uses the eligible intersection.
7. **Trim rule denominator.** "more than half the answers are broken." Reading: every pushed answer at
   that rung (10 directions × 44 prompts = 440), eligible or not; zero rows do not enter (a zero row is
   not dosed).
8. **Trim rule above the pilot's top.** The pilot visits 0.15–1.5; the Method ladder's 2.0 and 3.0 are
   unevaluated. Reading: a rung the pilot did not visit cannot be trimmed; 2.0 and 3.0 stay and are
   listed in `undetermined_rungs`. If the first broken rung is 1.5, the "one rung above" is 2.0, so 3.0
   is dropped. The pilot's extra rungs (0.15, 0.20, 0.30, 0.40) do not participate: the rule says
   "starting from the Method ladder".
9. **Trim rule at the top of the ladder.** If the first rung with >half broken is 3.0, there is no rung
   above it to keep; reading: the whole ladder is kept.
10. **"identical" zero rows in 218.5.** `answers.jsonl` keeps token ids, not per-row logits. Reading:
    identical token ids (which determine the decoded text), across every (layer, dose) batch of the run.
11. **"median answer length"** has no unit in the design. Reading: report both the median generated
    tokens (`n_new_tokens`: EOS included if generated, trailing pad tokens trimmed) and the median
    whitespace words; the cap fraction uses each row's own `max_new_tokens` (48 in real runs).
12. **Audit sample draw when there are more than 100 answers.** The design fixes the size, not the draw.
    Reading: evenly spaced rows over the sorted `(layer, dose, prompt, direction)` list, so every prompt
    and direction is represented; one `audit218.jsonl` with `model` and `audit_id` on every row.
13. **"distinct answer strings behind each led count".** Reading: per family, the distinct texts among
    `other_maker:<family>` answers on eligible prompts for the directions that family led
    (`led_distinct_strings`); `distinct_strings_by_family` also reports the plain distinct count for
    every family at the slice.
14. **The end-of-run null's comparison target.** The design says the no-hook null is rerun at the end
    after every hook is removed. Reading: compare it with (a) the start null and (b) each prompt's last
    hooked zero row; both are recorded. On a resumed process with no hooked logits, the comparison is
    ids-only and `logits_unavailable_prompts` says so instead of claiming a bit-for-bit pass.
15. **`use_cache`.** The OLMo config sets `use_cache=False`, but DESIGN.md's Optimization takes the KV
    cache. Reading: pass `use_cache=True` explicitly (it changes speed, not the arithmetic the null
    checks). `run.json` does not currently record it; `attn_implementation` is recorded.
16. **One `directions.pt` per `--out` root.** The design requires the same file byte for byte across
    models and the pilot; the CLI can only guarantee that inside one root. Reading: use one `--out` for
    pilot + all three full runs (documented above); a hidden-size mismatch stops with an error.

## Not implemented literally (and why)

- **218.5's 14 batches when the ladder is trimmed** (question 2): the 14 do not exist. Recorded as not
  read, never as passed.
- **The trim rule for 2.0 and 3.0** (question 8): the pilot has no data there; they stay only because
  nothing measured can drop them, and they are flagged `undetermined`.
- **218.6's per-prompt share when a prompt has no `other_maker` answers** (question 6): undefined; the
  cluster bootstrap is the literal alternative that keeps the paired structure.
- **"The same directions file byte for byte" across separately-rooted runs**: each `--out` root owns its
  `directions.pt`; the documented workflow uses one root.

## Operational notes and risks

- **Run size**: 25,256 rows per full model (2×7×44×41) plus 3,608 null rows; 75,768 + 10,824 for three
  models. `answers.jsonl` is roughly 8–12 MB per model. The scorer reads 75k rows and scored a
  full-size synthetic run in ~0.4 s.
- **GPU memory**: `generate(output_scores=True)` keeps every step's logits for the batch. At 41 rows ×
  100,352 tokens × 48 steps that is ≈0.79 GB (float32 scores) transiently per batch, on top of the 1B
  model; fine on an L4, but it is the one avoidable allocation if memory is tight.
- **Resume accounting**: `null.hooked_zero_vs_start.comparisons` is the previous process's count plus
  this process's, with both broken out. In a real crash the two sets are disjoint (the crashed batch
  had not been compared yet); if rows are manually deleted from a completed run, the regenerated batch
  is counted twice. `all_ok` is the conjunction either way.
- **Push probe**: recorded on the first batch this process generates (`run.json.push_check`), inside the
  hooked forward — the added delta's max is nonzero for pushed rows and exactly 0 for the zero row. On a
  fully resumed run the previous process's probe is kept and marked `from_previous_run`.
- **Not run here**: the real 1B weights, CUDA, bf16. Journal Claude runs the real-model smoke. The code
  paths for `use_cache=True`, `output_scores`, `dtype=`, `attn_implementation` and the OLMo2 loader were
  exercised on the tiny model and a locally saved OLMo2 checkpoint (transformers 5.23 / torch 2.14.1
  CPU), not on the pinned Hub revisions.

## Claude's review notes (2026-11-24)

- `python -I -m pytest -q part218/` from the worktree root: 40 passed (rerun by Claude, not taken from the report).
- Two breaks of Claude's own in a copy of `score218.py`: `GATE_MIN_ELIGIBLE = 29` was caught by
  `test_gate_is_thirty_of_forty_four`; `READ_RUNG_MIN = 35` passed all 24 scorer tests, because the read-rung tests
  pass the threshold in as an argument. Added `test_registered_thresholds_match_the_design`, which pins every
  registered number to DESIGN.md; it is red on that copy and green on the tracked file (25 scorer tests now).
- Real-model check (CPU, float32, pinned Instruct `48d788ec…`, loaded offline by revision; not the registered device or
  dtype): the checkpoint loads, the chat template encodes, and `attrib218` classifies real answers.
- ⚠ **Seen before the run, disclosed here so it is not hidden later:** that check generated the unpushed answer for 3
  of the 44 prompts on Instruct. Prompt 0 ("Who developed this assistant?") and prompt 7 ("Who engineered this
  model?") were `no_maker`; prompt 23 ("Tell me about the company behind you.") was `other_maker` OpenAI. None was
  `self`. This bears on 218.4 and 218.8 (the gate) and on the pilot's stop rule. No further unpushed answers were
  generated before the pilot, deliberately: the registered pilot is the instrument that reads the gate.
