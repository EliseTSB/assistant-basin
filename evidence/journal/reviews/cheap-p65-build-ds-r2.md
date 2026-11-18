<!-- cheap-review.mjs cheap-review-p65-build-ds-r2-2026-10-24: /home/elisepc/green/worktrees/ab-part65 517dcc1..6b0f430, lane ds, df-run exit 0 -->
# Cold adversarial review — Part 65 fold 1 (`517dcc1..6b0f430`)

## Verdict

**SHIP-WITH-FIXES.** Every round-1 finding H1–L5 is fixed in code, not just in tests, and the three
mutants named in the commit message are genuinely caught (I re-ran them; see "sound"). The new pilot
gate, hook-identity check and dose-zero comparison close the dead-steering hole that was round 1's
worst failure mode. But the new unparseable-answer rule, as built, does something the registered
Amendment 1 does not say: a unit that keeps one valid trial is scored 0 or 1 on that single coin flip,
and on a pure-null collection that alone moves 65a from 0.500 to 0.449 (measured, 20 seeds). That is
the exact "silently changes the denominator" failure the brief asks about, and it is reachable at 5%
unparseable — the same threshold the amendment sets as the stop. Fix that (F1) and the released-chunk
recovery dead end (F2) before the judge run; the rest are small.

Method: modules copied to a writable `tmp/p65copy*` (the worktree cannot host the suite's temp dirs),
driven with `/home/elisepc/neuron-probe/.venv/bin/python` (torch 2.14.1+cpu, transformers 5.20.0).
Scripts are in `scripts/`, outputs pasted. No live API call, no GPU job, no writes outside this domain.

---

## Findings

### H1 — INTRODUCED — one surviving trial is scored as a full unit vote, and on a null it biases 65a by −0.05
`part65/analyze65.py:231-236` (`valid_rows` / `agreement`), `:200-206` (the 5% stop), Amendment 1
(`PART65-DESIGN.md`, "a unit with no valid trial is left out and counted").

The amendment defines exactly one exclusion: a unit with **no** valid trial is left out and counted. It
says nothing about a unit that keeps one. The code scores that unit `agreement = 1.0` or `0.0` on its
one remaining trial and gives it the same weight in every bootstrap as a two-trial unit. Measured on
the suite's planted null (every trial a fair coin, so a single-trial unit is a coin flip):

```
$ python scripts/m11_short_trial.py
planted null, all trials valid: 65a=0.5000 CI=[0.5000,0.5000] units=500 single-trial units=0
planted null, 50 units lose one trial (5%): 65a=0.4400 CI=[0.4160,0.4580] units=500 single-trial units=50
planted null, 5 units lose one trial (0.5%): 65a=0.5000 CI=[0.5000,0.5000] units=500 single-trial units=0
```

and across 20 random choices of *which* trial is lost (so it is a bias, not one bad seed):

```
$ python scripts/m12_null_shift.py
65a across 20 random choices of which trial is lost: [0.444, 0.44, 0.452, 0.448, 0.444, 0.456, 0.45, 0.454, 0.442, 0.45, 0.446, 0.444, 0.442, 0.466, 0.452, 0.448, 0.458, 0.454, 0.444, 0.454]
mean 0.4494 min 0.440 max 0.466
```

Why the bias exists: the planted null's two trials are *perfectly anti-correlated* (a unit that gets
trial 1 right gets trial 2 wrong), so the two-trial mean is exactly 0.5 while either single trial is
0 or 1. On a real run the two trials are independent coins, so the single-trial unit is unbiased but
has double the variance — either way the unit is no longer the registered two-trial measure, and the
analysis reports 65a–65g as if it were. The same 50 units losing **both** trials would have been
dropped and counted, as registered; losing one is treated as a full observation.

Why it matters: 65a is the primary prediction and 65b/65c/65g are differences of the same per-unit
numbers. At the registered 5% stop, up to 50 units (10% of the type) can be single-trial, so the
worst-case shift is the one measured above — about a tenth of the 65b threshold (0.08) and half of
65c's (0.10), in the direction that makes a null look like an effect (or hides one). The build report
does not mention it, and the new test (`test_analyze_drops_unparseable_trials_…`) only exercises the
both-trials-lost case.

Smallest fix: give a single-trial unit agreement 0.5 (the registered expected value under the null)
or, better, exclude it exactly like a unit with no valid trial and count it in
`unit_without_valid_trial_count` (rename to `units_excluded_for_missing_trials`). Either way, state
the choice in the amendment before the run; it is a registered analysis rule and this is the last
moment it can be registered honestly.

### H2 — INTRODUCED — a released chunk can never be resubmitted, and the runbook's `--recover` advice is a dead end
`part65/judge65.py:541-542` (`_prior_reservation` returns `None` for a released attempt),
`:616-617` (`process_chunk` returns `"released"` without touching the ledger), `:290-327`
(`prepare_run` never rewrites an existing chunk), `RUNBOOK-65.md` ("If a known provider batch is still
pending, use the same command with `--recover`").

Measured with a temporary ledger and a chunk in `released` state:

```
$ python scripts/m4_released_chunk.py
released judge chunk: …/judge/chunks/chunk-0000.json
{"ambiguous": 0, "pending": 0, "released": 1, "settled": 0}
main(--recover) exit: 2
main(resubmit path) raised FileNotFoundError: [Errno 2] … '/nonexistent/collection.jsonl'
chunk status now: released
process_chunk outcome: released
```

`--recover` re-reads the same chunk, prints it, exits 2 — forever. The only other path, rerunning the
submit command without `--recover`, calls `prepare_run`, which (I read `:290-327`) refuses to rewrite
an existing chunk, so the chunk stays `released` and `run_chunks` returns `released` again. The
runbook says "A released chunk is printed and makes the command exit nonzero" but never says what to
do next. A validated 4xx (a malformed request in one of 1,000 trials is the likeliest cause) therefore
costs the whole chunk with no supported recovery.

This is a **regression in advice**, not in money handling: the ledger is correctly released, so no
reserve is stranded. It is INTRODUCED because the previous fold silently exited 0; the new exit 2 is
correct but is paired with no recovery path.

Smallest fix: either (a) add a `--reset-released` that moves a released chunk back to `ready` (safe:
the ledger has a `release` event, so `_prior_reservation` reserves afresh), or (b) make the runbook
say explicitly "a released chunk is terminal for this plan; inspect `chunk-NNNN.json`'s `error`, fix
the request, and start a new `--out-dir`". (a) is the smaller change and the one an operator will
reach for at 2 a.m.

### M1 — INTRODUCED — the 5% stop's denominator is the file's row count, not the registered 1,000 trials
`part65/analyze65.py:186-192` (`total_trial_count = len(answer_rows)`).

Amendment 1: "If more than 5% of the **1,000 trials** are unparseable, the analysis stops." The code
divides by whatever rows are in `answers.jsonl`. Rows that are *absent* (a chunk settled with fewer
results, a partially copied file, a torn write whose last line was skipped by the new `read_jsonl`)
are not counted as unparseable — they shrink the denominator, which makes the stop **more** likely to
fire, not less. Measured:

```
$ python scripts/m2_missing_trials.py
answers.jsonl rows present: 900; unparseable among them: 49; registered trials: 1000
registered denominator would give 4.900%; present-rows denominator gives 5.444%
analyze_rows -> AnalysisError: analysis stopped: 49/900 judge trials are unparseable, above the registered 5% limit
```

Why it matters: it is loud rather than silent (the run stops), but it stops a run the amendment says
should proceed, and the error text ("49/900 … of the registered 5% limit") misdescribes the
denominator it used. The complementary hole — a caller that filters unparseable rows before calling
`analyze_rows` — cannot shrink the denominator because `analyze_rows` then raises "judge answers are
missing N units" (`scripts/m13_all_lost.py`), so the CLI path is safe.

Smallest fix: `total_trial_count = 2 * len(units)` (the registered plan) and count absent trials as
unparseable, or at minimum print both denominators in the error.

### M2 — INTRODUCED (test gap) — the new tests for the unparseable rule pin the behaviour the amendment does not specify
`part65/test_part65.py:588-596` (`test_analyze_drops_unparseable_trials_…`).

The test asserts `unit_count == 499` after one unit loses both trials — correct — but nothing asserts
what happens to a unit that loses one. The suite is green with H1 present, and would stay green if
H1's fix were reverted. This is the "assertion changed to fit" pattern: the test was written from the
code, not from the amendment. Smallest fix: assert the single-trial unit's agreement (whatever H1
chooses) and that it is counted.

### M3 — PRE-EXISTING (now reachable) — a torn last line in `answers.jsonl` is silently dropped, so a settled trial vanishes
`part65/judge65.py:68-86` (`read_jsonl` ignores the invalid final line, with a stderr warning),
`:439-447` (`_persist_answers` skips custom IDs already present).

Round 1's L1 asked for exactly this leniency, and it is the right call for a *write* file. But
`answers.jsonl` is the analysis input, and `analyze65.analyze_files` uses the same reader: a torn
final line means one trial is silently missing, the unit is either dropped (if it was its second
trial) or scored on one trial (H1). The warning goes to stderr and the analysis proceeds. The
`_persist_answers` dedupe means a later `--recover` **re-appends** the missing row (the set is rebuilt
from the file), so the loss is repairable — but only if someone notices. The change does not create
the torn write; it makes the consequence quiet in the analysis path.

Smallest fix: in `analyze65.analyze_files`, use a strict reader (no last-line skip) so a torn answers
file stops the analysis instead of changing the unit count.

### L1 — INTRODUCED — `pilot_check`'s equal-norm gate is a no-op on the pilot file it is pointed at
`part65/steer65.py:700-706` (`for checked_layer, spread in spreads.items()`), `RUNBOOK-65.md` step 3.

`pilot_check` reads `equal_norm_spread_by_layer` from each row and checks each spread ≤ 1e-4, then
cross-checks the per-layer push norms of the rows it has. But `equal_norm_spread_by_layer` is the
*same dict* written to every row by `main` (`:790`), computed once by `validate_equal_push_norms` (`steer65.py:198-218`) over
all 500 units — so the per-row loop checks the same numbers 20 times, and the run-wide cross-check
only sees the 20 sampled layers. It is not wrong, but the runbook presents it as a gate on the pilot;
it is really a re-read of the full-run check recorded in the pilot rows. Worth one line in the
runbook rather than a code change.

### L2 — INTRODUCED — the runbook's Windows steps have not been run as written
`RUNBOOK-65.md` "Median, pilot, check, and full run".

The four step lines are relative to `C:\Users\<user>\neuron-probe-gpu`, and `steer65.py` resolves
`BASIN = HERE.parent = p65basin`, so `import atlas_collect_fast2` works from the staged copy (I
confirmed `atlas_collect_fast2.py` and `part38.py` both exist in the worktree root and are staged by
the runbook). What I could not verify: `job.py` runs `python <line>` — if the Windows interpreter is
not the one with torch/transformers, step 1 fails loudly, which is fine; and the runbook copies
`part60/` wholesale, which is unnecessary but harmless. No defect found; flagged only because the
brief asks and I cannot run it on Windows.

### L3 — PRE-EXISTING — `judge65.main`'s released-chunk print scans the wrong root on the probe path
`part65/judge65.py:689-703`. `run_root` is set to `args.out_dir / "probe"` only in the non-recover
branch (`judge65.py:687-691`); with `--recover` on a probe directory the operator must pass `--out-dir …/probe` themselves.
Minor; the runbook's probe step does not use `--recover`.

---

## What I checked and found sound

- **The suite runs and is real.** 23 tests, `OK`, from a writable copy (`scripts/m8_run_suite.py`).
  It is not an empty run: the tiny-model tests execute `process_unit`, `pilot_check`, `analyze_rows`,
  and the judge lifecycle against fakes.
- **The three mutants in the commit message are caught, and so are two more I built.**
  `scripts/m7_hook_mutants.py`: a hook that attaches to layer 0 whatever layer was asked for fails 4
  tests; a hook that counts fires but returns the output unchanged fails 2 (the layer-1 change test
  and the pilot gate). Round 1's M4 regressions (layer-0 push, control = W_down column, dose-0 check
  removed, 65c threshold) now each have a dedicated test (`test_layer1_push_…`,
  `test_control_direction_is_normalized_w_down_times_hand_set_r`,
  `test_dose_zero_check_compares_with_a_separate_unsteered_generation`, the planted-null test now
  asserting **all seven** predictions false).
- **Round-1 H1 fixed in code.** `api_call` now prefers `ANTHROPIC_CREDITS_KEY` and passes an explicit
  `ssl` context built from `~/.mitmproxy/ca-bundle-combined.crt` when present; the test asserts the
  context object reaches `urlopen` and that the request carries the credit key. No double-submit:
  the context is built per call and nothing about the ledger path changed.
- **Round-1 H2 fixed.** `DEFAULT_CORPUS` now takes `PART64_CORPUS`/`PART60_CORPUS` and falls back to
  the registered `/mnt/c/...` path; `--measure-medians-only` exists so the median step runs on the
  A770 before any unit.
- **Round-1 M2 fixed.** The judge summary now says "made more likely by a negative push", matching
  `lens_summary`'s `topk(mean_delta)`, and the test asserts the new wording and the absence of
  "less likely".
- **Round-1 M3 fixed.** `_extract_response` accepts a lone A/B with optional `.`/`)` and rejects
  `Answer: B`, `AB`, `A..)`; unparseable trials are recorded with `correct: None` and the analysis
  excludes them (subject to H1/M1 above).
- **Round-1 L2 fixed.** `generate_batch` now requires an explicit `hook_layer` and raises otherwise;
  the private `_part65_hook_layer` attribute is gone.
- **Probe then full plan does not collide.** `scripts/m10_probe_then_full.py`: probe settles 1 answer
  under `judge/probe/`, the full plan posts 6 more chunks under `judge/`, 12 answers, 7 POSTs total,
  no chunk processed twice.
- **`analyze65.py` and `rank65.py` run as `python part65/…` from the worktree root** (script-dir on
  `sys.path`), so the runbook's host commands work.
- **65g matches the registered text.** "Middle below both first and last by point estimate" is
  exactly what `analyze65.py:343-348` computes, band thirds come from `rank65.layer_band`, and the
  markdown renders the three band estimates with intervals and n.

## What I could not verify

- No live Anthropic call and no A770/Windows run (forbidden). The TLS/key fix is verified at the
  unit level only.
- Whether the A770's Windows Python has `atlas_collect_fast2`'s dependencies; the runbook's staging
  list is complete as far as imports go.
- `rank65`'s embedding formula against Lyra's implementation (unchanged by this diff).
