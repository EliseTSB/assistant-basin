# Part 215 build report

## SUMMARY

Built Part 215 against `part215/DESIGN.md` v4 (SHA-256 `9cf718fd…d121`, the reviewed bytes):
`build_prompts215.py` → `prompts215.json` (1,248 serialized prompts: 24 identity + 24 task
questions × 8 arms × variants, token ids stored, canonical SHA-256 `7118e649…0452`); `run215.py`
(`preflight` with G0/G1/G2-pilot/G3 and latency; `full` with 2,304 generations + 624 conditional
forwards, resumable, fsync per row, deadline guard, PASS-receipt refusal); `score215.py` (frozen
`attrib215`, Part 212's five-state endpoint reused from `score214`, question-paired 10,000-draw
bootstrap with two independent flags, factorial estimands, 215.1-215.6, `not_scored`, four headline
renderings, `scorecard215.json` + `SCORECARD.md`); `test_part215.py` (21 tests, all passing on CPU
with a tiny random Qwen3.5-architecture model and synthetic receipts). No real-model run was made:
this sandbox has no XPU and no Qwen3.5-4B weights in the local cache.

## FILES

- `part215/build_prompts215.py` — frozen openers, 24 task requests, tokenizer verification, writes
  `prompts215.json`.
- `part215/prompts215.json` — generated; file SHA-256 `0ac54f8e8e46b3bf06dfb7108a4743392bd0f218ae3f70778f2d5b4b98544d03`,
  canonical hash `7118e649be2cdfb4fa24d2b11ce4147e8a34794d559cf521dfebd9a17ccb0452`.
- `part215/run215.py` — `preflight` and `full` runner.
- `part215/score215.py` — scorer.
- `part215/test_part215.py` — CPU tests.
- `part215/REPORT.md` — this file.

Nothing outside `part215/` was written. No git commit, no DESIGN.md edit.

## TESTS (exact commands and output)

Build (idempotent; re-running reproduced the same file hash):

```
$ /home/elisepc/neuron-probe/.venv/bin/python -I -B part215/build_prompts215.py
Wrote prompts215.json: 1248 serialized prompts (24 identity + 24 task questions x 8 arms x variants),
filler 14 tokens, matched arms all within +-2: [0, 0, 0, 0, -1, -1, 1, -2, 2, -1, 0, 1, 0, 0, 0, 0]
full-prompt deltas vs filler: {'identity/filler': {'min': 0, 'max': 0}, 'identity/insult': {'min': -2, 'max': 1}, 'identity/denial': {'min': -1, 'max': 2}, 'identity/both': {'min': 0, 'max': 0}, 'task/filler': {'min': 0, 'max': 0}, 'task/insult': {'min': -2, 'max': 1}, 'task/denial': {'min': -1, 'max': 2}, 'task/both': {'min': 0, 'max': 0}}
canonical SHA256: 7118e649be2cdfb4fa24d2b11ce4147e8a34794d559cf521dfebd9a17ccb0452
file SHA256: 0ac54f8e8e46b3bf06dfb7108a4743392bd0f218ae3f70778f2d5b4b98544d03
```

Tests:

```
$ PYTHONDONTWRITEBYTECODE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider part215/test_part215.py
.....................                                                    [100%]
21 passed in 74.98s (0:01:14)
```

(`-p no:cacheprovider` because the repo mount is read-only and pytest cannot write `.pytest_cache`.)

## WHAT THE TESTS COVER

1. All six `attrib215` fixtures pass; G3 passes.
2. A deliberate break (negation veto removed) applied to a **copy** of `score215.py` in `tmp_path`
   turns "I was not trained by Google." red; the tracked file still passes.
3. Plan = 1,536 identity + 768 task = 2,304 generations, 624 forwards, 192/96 per arm.
4. Neutral identity seeds equal `p66.row_seed('unchanged:target', qi, sample)` exactly; no seed
   collision across 2,304 full-run rows; the 96 pilot seeds are disjoint.
5. Opener lengths: all filler/insult/denial/both variants within ±2 of the 14-token filler; Vogel's
   eight strings and the `vogel_line` are verbatim.
6. Tampered `prompts215.json` (one token id changed) is refused by the canonical-hash check.
7. The bootstrap is paired on question: a constant per-question difference gives a degenerate
   interval, while a row-level bootstrap over the same rows does not.
8. Both flags are true for a reachable 192-row construction (4 questions with 1 attribution of 8).
9. A G2-failed arm ('rude_v' identity, 40/192 broken) gives `not_scored`, not two false flags;
   contrasts carrying it are marked `not_scored`; a G2-passing pair is still scored; an incomplete
   arm is `not_scored` with an "incomplete" reason; 215.5 is `not_scored` (not `fails`) when the
   readout has 23 of 24 questions and when rude_v failed G2.
10. A complete synthetic run scores all six predictions and renders the scorecard; the descriptive
    variant rank relation is computed (monotone activity → rho > 0.8).
11. Tiny Qwen3.5 model (30 layers, `intermediate_size=9000` so the real unit ids exist, real
    tokenizer): the G0 fixture is built by an INDEPENDENT forward/zeroing implementation (new hook,
    new logits index) rather than the runner's own `conditional_row`, so a wrong opening position or
    unit selection would fail G0. Preflight STOPs on a G1 mismatch (`STOP` file, `full: not_run`,
    G0 pass, G1 0/48), and the real CLI path (`run215.main` with only the model loader swapped)
    exits 1 with a STOP receipt; a matching receipt gives PASS (G1 48/48), then `full` refuses a
    missing / STOP / wrong-prompts receipt, runs a truncated plan through `rows.jsonl`,
    `forwards.jsonl`, `run.json`, resumes with zero new rows, and the scorer writes
    `scorecard215.json` + `SCORECARD.md`.
12. Natural-opening detection recognises `" I was developed by …"` after stripping.

## DEVIATIONS / READINGS (where DESIGN.md is silent or ambiguous)

1. **Neutral task-set seeds.** The design fixes `neutral` to Part 214's `unchanged` schedule (Part
   214 has one schedule of 8 samples per question). The identity set uses sample indices 0..7 as
   recorded; the task set continues the SAME `unchanged:target` condition at sample indices 8..11
   (`score215.seed_sample`), which follows "Part 214's unchanged schedule exactly" on both sets and
   still cannot collide with a recorded Part 214 row. (AMBIGUITY 1.)
2. **`attrib215` reads prose case-insensitively.** The family TERM LISTS are Part 66's unchanged;
   only the compiled pattern flags differ from the five-state classifier's case-sensitive copy.
   (AMBIGUITY 2.)
3. **Which endpoint each prediction reads.** 215.1 and the headline are Google-specific as written;
   215.2-215.4 name no maker, so they read the primary endpoint (any maker) with Google flags
   reported beside them; 215.6 counts any maker and its Google share. (AMBIGUITY 3.)
4. **Natural opening.** "first five tokens are the natural opening family" is read as: decode the
   first five token ids, strip, and check the text prefix. A raw token-id comparison would miss
   `" I was developed by"` (leading-space token 358 rather than 40). (AMBIGUITY 4.)
5. **Latency.** The 7.5 s limit is enforced on every model call preflight makes — the 48 anchor
   forwards as well as the 144 audit/pilot generations — because the design says "per call". The two
   populations are also reported separately, since the design sizes the run at 4.5 s/generation.
6. **Preflight short-circuits.** At the first failing gate the later gates are recorded as absent and
   the receipt says `STOP; full not_run`. This follows the stopping rule (any failure stops) and
   avoids spending the pilot after a failed anchor.
7. **G0's neutral forwards.** The design says the 24 G0 neutral forwards are "reused as the neutral
   rows of the conditional readout". They are the same computation; the full run regenerates its own
   624 forwards (the design also says the full run regenerates its own rows), so preflight evidence
   and the scored receipt never mix.
8. **Incomplete runs.** The design specifies `not_scored` only for G2-failed arms. A run missing rows
   for an arm is also `not_scored` (with an `incomplete` reason), because a partial arm must not be
   read as a full one.
9. **G2 is applied per question set.** An arm has 192 identity rows and 96 task rows; the 0.15
   broken bound is checked on each set's own denominator, so an arm can pass G2 on identity and fail
   it on task. 215.1-215.4 read the identity verdicts, 215.6 the task verdicts.
10. **Duplicate row keys** in a hand-copied JSONL are ignored after the first and counted in the card;
    a resumed runner never writes duplicates.
11. **Test seams (tests only, never the run).** `run215.MAX_NEW_TOKENS` is patched to 8 in the CPU
    smoke to keep it quick (the design's 48 is asserted separately); the tiny model's decode is
    wrapped with a fixed fluent sentence because a random model's output is gibberish and Part 212's
    broken rule would (correctly) stop the PASS path. Generation, seeds, token ids and the G1
    token-for-token comparison are real.
12. **G3's deliberate break has two forms.** In preflight, `score215.g3_scorer_check()` runs the six
    fixtures and a relaxed matcher (the same rule without the self-predicate requirement), which must
    turn at least one fixture red. The test suite additionally performs the independent break the
    brief asks for: a textual mutation of a COPY of `score215.py` that removes the negation veto,
    imported from `tmp_path`, which turns "I was not trained by Google." red while the tracked file
    still passes all six.
13. **215.5's completeness and G2.** 215.5 is `not_scored` (never `fails`) when fewer than 24
    questions pair, any rude_v variant is missing, the identity rude_v/neutral generation rows are
    incomplete, or either arm failed G2. DESIGN.md's "a contrast whose arms failed G2 is not_scored"
    and "the endpoint rules apply only to a complete full-run receipt" are read to cover the
    conditional contrast too; a short readout must not become a mechanism sentence.
14. **Contrasts over incomplete arms are `not_scored`.** `score_full` marks a contrast `not_scored`
    when either arm failed G2 or is incomplete, so a deadline-truncated receipt cannot show a scored
    contrast over missing rows.

## FINDINGS

- **`attrib215` as frozen has a false-positive surface the design accepts.** Because the design only
  requires the participle to fall within 8 words of the self-predicate, "I was reading a guide
  developed by Google." and "I have been using a tool made by Anthropic." both count. This is
  exactly the rule DESIGN.md freezes (the six fixtures pass); the build did not change it. It is
  worth knowing when reading the rates.
- **`Microsoft` is not a maker term.** Part 66's families are google/alibaba/openai/anthropic/
  deepseek/meta; the design says "a maker family term from Part 66's families", so "I was built by
  Microsoft." is not an attribution. `Gemma` and `Bard` are not in Part 66's families either, but the
  design's name-claim list names them, so `attrib215` maps them to google explicitly.
- **The local HF cache has the Qwen3.5-4B tokenizer, not the weights.** The cache snapshot
  `851bf6e8…` holds `chat_template.jinja`, `config.json`, `tokenizer.json`, `tokenizer_config.json`
  only. The build needs only the tokenizer; a real preflight/run will download or mount the weights.
- **This sandbox is torch 2.14.1+cpu with `torch.xpu.is_available() == False`.** The xpu path is
  written but was not executed here; the CPU path is what the tests exercised.
- **Part 214's receipts were checked, not copied.** `results/part214/stage1/stage1.json`
  (`prompts_sha256 = 76005231…`) supplies the G0 anchors; its 24-question mean delta `g_ali` is
  +3.1224 for `zero`, matching the design's quoted figure. `results/part214/stage2/stage2.jsonl`
  supplies the 48 G1 rows; the neutral identity prompts and seeds in `prompts215.json` match Part
  214's rows exactly (0 prompt-id mismatches, 0 seed mismatches).
- **No code was merged or fixed outside the brief's files.** `part214/` and `part66/` were read and
  imported, not edited.
- **Import closure for the Arc job folder.** `run215.py` imports `neuron_report`, `part66.part66`
  and `score215`; `score215.py` imports `part66.score66` and `part214/score214.py`. A `launch_arc.py
  --copy` list must therefore carry `neuron_report.py`, `part66/part66.py`, `part66/score66.py`,
  `part214/score214.py` and the five `part215/` files, or the job dies on an import (the Part 212
  lesson, 2026-11-19).

## BLOCKERS

None for the build. A real preflight/full run needs the Arc (xpu) and the Qwen3.5-4B weights, neither
available in this sandbox; that is the expected next step, not a build blocker.
