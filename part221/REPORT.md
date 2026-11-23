# Part 221 build report

## SUMMARY

Part 221's runner, scorer, tests and launch file are built in `part221/`, per `part221/BRIEF-build.md` and
`PART221-DESIGN.md` v3. `run221.py` runs the design's nine-run table (R0, R1, R13, R11, R12, R14–R17) in one process
with one host load and one donor load, adds R13's pre-generation weight assertion and the sign-flip / norm-ratio
receipt, and writes the R0/R1 reproduction check and R13-against-R1 identity check into `manifest.json`.
`score221.py` reuses `score219.py`'s functions for the gate, 221.1–221.3, the reader gate and the exploratory arms, and
records the coupling numbers. `test_part221.py` runs 21 tests, including `main()`'s real pilot→full path on a tiny
OLMo 2; all pass, Part 219's 66 tests still pass, and each required property was shown red by breaking a copy (list
below). No files outside `part221/` were touched, nothing was committed, and no real model was loaded.
No parent-control messages were waiting at any checkpoint (mailbox read three times: empty).

## WHAT YOU BUILT

### `part221/run221.py` — orchestrator

- The design's Runs table, in order: R0 (no change), R1 (`swap: attn, layers: 0-3`, Part 219's code path), R13
  (`transplant_heads` over every head of layers 0 to 3, donor mode), R11 (the 60 complement heads, donor), R12 (the 60,
  opposite = `2 x Instruct - donor`), R14–R17 (one layer's 15 complement heads each, donor).
- Kept heads `L0H10, L1H4, L2H6, L3H5`; the complement is computed from the model's own head count, checked to be 15
  per layer, and refused unless it is the registered 60 when the model has 16 heads per layer (`head_lists`). The
  manifest records `complement_heads: 60`, `complement_per_layer`, and the full complement list.
- **R13 weight assertion, before generation**: a one-run wrapper (`TransplantWatch`) replaces
  `swap99.transplant_heads` for the duration of R13, and after the write compares *every* parameter below layers 0–3's
  `self_attn` with the donor's using `torch.equal` (`self_attn_equals_donor`, 24 parameters, read from the modules, not
  from the six transplant parts, so an uncovered parameter is reported unequal). A mismatch restores the host and raises
  `SelfAttnMismatch` before any row is generated; the check is kept in `R13.meta.json` and in the manifest, and a stub
  meta is written if the raise lands before `swap99.run` writes one.
- **Sign-flip receipt**: for every transplant run, per layer, the `q_norm.weight` and `k_norm.weight` elements whose
  sign after the write differs from Instruct's before it (counted over the slices the run wrote), with the rule recorded
  (`sign(x) = -1/0/+1`, zero is its own sign, a value that becomes exactly zero flips, a zero that stays zero does not),
  plus the min and max `norm_ratio` over the run's slice records, and the count of records with no ratio. Both land in
  `intervention_receipt.transplant` in the run's meta (and in the manifest entry).
- **Coupling**: Part 219's `coupling_log` is reused unchanged; `kept_head_drift` copies the four kept heads out under
  their names, per layer, with the per-layer rms(q), rms(k) and rms(attention output) relative changes, and the two
  maxima the headline needs.
- **Reproduction**: `reproduction_report` compares R0 and R1 with `<repro-dir>/R0.jsonl` and `R1.jsonl` (default
  `results/part219`) token for token and answer for answer, and R13 with R1; written to `checks.reproduction` in
  `manifest.json`.
- **Reset receipts** after every run (`--reference-replay` on every run, Part 219's `run_checks`).
- **`--pilot`**: R0, R1 and R13 only; writes `PILOT.json` with six named checks (each run readable, R0 reproduces, R1
  reproduces, R13 equals R1, R13's weights equal the donor's, the sign-flip receipt is present, the reset receipts
  hold); a failure prints the failing checks, writes the verdict and exits 2. The pilot's rows stay on disk and the full
  run reuses them (swap99 skips rows already written).
- **Software pin**: `torch 2.13.0+xpu`, `transformers 5.13.0`; a mismatch refuses to start with a clear message unless
  `--allow-version-mismatch`, which is recorded in `manifest.software` and used by every run's meta (written by
  swap99); each run's entry records the versions from its own meta.
- **Kernel record**: `run219.check_kernel_record` (Part 219's `score219.kernel_check` → `tools/part_check.kernel_check`,
  `results/part219/kernel-check.json`), refusing to launch when it does not pass under the eager kernel.
- **Progress** every 8 s while a stage runs; `--limit` for smoke runs; `--replay`, `--kernel-check`, `--repro-dir`
  overridable.

### `part221/score221.py` — scorer

- Reuses `score219.py`/`read101.py`: `load_kinds`, `slot_split` (16 upstream / 5 downstream), `load_labels`,
  `count_shows`, `mcnemar_exact`, `net_against`, `contrast`, `two_pass_gate`, `deepseek_agreement`, `kernel_check`,
  `readable_reasons`, `score_run`'s body with the label directory as a parameter.
- Gate: kernel, R0 ≥ 17 of 21, R1 ≤ 5, R0 and R1 reproduce Part 219, R13 equals R1; a missing reproduction block fails
  the gate with a reason. If the gate fails, all three predictions are frozen (`"not read: gate failed: ..."`,
  `frozen: true`) and the markdown says so in its first lines.
- 221.1/221.2 read only if R11 clears the coherence bar (15 of 21 slot answers coherent, 13 concept units, not
  inconclusive), else `"inconclusive: the complement transplant breaks answers"`; 221.2 uses this part's R1.
  221.3 = 221.1 holds and R12 clears the bar and the exact two-sided McNemar of R11 against R12 (Part 219's
  shows-relative-to-own-unpushed indicator) gives p < 0.05 with R11 losing more; else `"not read: 221.1 did not hold"`,
  `"unreadable: R12 ..."` or `"inconclusive: the opposite-direction transplant breaks answers"`. A reader-gate failure
  makes a prediction `"unreadable: ..."`, never `"inconclusive"`.
- Lost and gained unit lists for every run, the R11/R12 overlap, the 5 downstream units, all-21 counts, R14–R17 each
  marked unreadable below the bar, and R11's net plus Part 219's R8 net against Part 219's R1 net (a sum, with the
  design's "not a decomposition" note).
- Coupling: the largest kept-head drift and the largest |change in rms(attention output)| per run, R12's sign-flip
  counts and norm-ratio range; both go into the headline sentence with the design's wording ("with the four heads'
  weights left as Instruct's, under an early attention whose normalisation context and other heads changed").
- Labels for R0/R1: Part 219's when the reproduction says identical, this part's otherwise (with the reason recorded
  per run); R13 is marked "not labelled: identical to R1" rather than unreadable, as the design intends.
- Writes `results/part221/scorecard.json` and `SCORECARD.md`, and prints the markdown.

### `part221/test_part221.py` — 21 tests

- The fixture is Part 219's tiny OLMo 2 with four layers and sixteen heads of head_dim 1, so the registered kept heads
  exist, R14–R17 have a layer each, and the complement is the real 60 (documented at the top of the file).
- Complement size/no-kept-head, model-relative geometry and refusal of an impossible kept head.
- R13's assertion passes on all 64 heads and fails (six parameters named) when one head is left out; the wrapper raises
  before generation, restores the host and uninstalls itself.
- Sign flip on a hand-built case: one element flips (+ → −), one becomes exactly zero (counts as a flip), one stays zero
  (does not); donor mode on positive norms flips none; the norm-ratio range's None handling.
- Orchestrator end to end through `main()`: a pilot that stops (exit 2) while the reproduction rows are missing, a
  second pilot that continues, the full nine-run table, every jsonl/meta, the R13 receipt, the sign-flip and kept-head
  receipts, then `score221.main()` over the result.
- Scorer on synthetic 21-slot worlds: the registered predictions hold; 221.1 fails / 221.3 not read; 221.2 fails;
  R12 below the bar → inconclusive; R11 below the bar → inconclusive; reader-gate failure → unreadable; the hand-counted
  McNemar in both directions; a failed gate and a missing reproduction block freeze everything; an unreadable R0 fails
  the gate; coupling numbers reach the headline; only the upstream units feed the predictions.

### `part221/steps221.txt`

Pilot then full launch line in `part219/steps219.txt`'s form (Windows job-folder paths), with the `--copy` list and
why each group is there: `part221/run221.py`, `part221/score221.py`, `part219/run219.py`, `part219/score219.py`,
`part99/swap99.py`, `part92/common92.py`, `part85/checkpoints.py`, `part101/read101.py`, `tools/part_check.py`,
`tools/launch_arc.py`, the neuron/kinds/pushes inputs, `results/part219/kernel-check.json`, the CPU float32 kernel
reference it names, Part 219's R0/R1 jsonl for the reproduction check, and Part 219's R0/R1 label passes (plus R8 and
the DeepSeek samples) for the reused labels and the exploratory sum.

## TESTS

`~/neuron-probe/.venv/bin/python -m pytest part221 -q` → **21 passed** (10.5 s).
`~/neuron-probe/.venv/bin/python -m pytest part219 -q` → **66 passed** (10.9 s), so nothing outside part221 broke.

Extra checks run: `python -I -B part221/run221.py --help` and `python -I -B part221/score221.py --help` (the launch
form) import cleanly; `score221.part219_nets` against the real `results/part219` labels reproduces Part 219's published
numbers (R1 net 11, R8 net 3, R8 lost {p92-05, 13, 18, 37}, gained {p92-32}).

### Break-and-red list (each break applied to a copy of part221 under /tmp, one at a time)

| # | break | test that went red |
|---|---|---|
| 1 | `KEPT_HEADS` loses `L3H5` (the complement is then 61, not 60) | `test_the_complement_is_the_registered_size_and_never_holds_a_kept_head` |
| 2 | `self_attn_equals_donor` always reports `equal: True` | `test_the_r13_weight_assertion_fails_when_one_slice_is_not_transplanted` |
| 3 | sign flips exclude written zeros (`& (new != 0)`), breaking "zero is its own sign" | `test_the_sign_flip_count_is_right_on_a_hand_built_case` |
| 4 | `run_specs` drops R17 (`HEAD_LAYERS[:-1]`) | `test_the_orchestrator_runs_end_to_end_on_the_tiny_model` |
| 5 | the R12 coherence branch is disabled in `build_predictions` | `test_r12_below_the_coherence_bar_makes_221_3_inconclusive` |
| 6 | the `221.1 did not hold` branch is disabled | `test_221_1_fails_and_221_3_is_not_read_when_r11_loses_too_few` |
| 7 | the scorer ignores run status (`status = {tag: []}`), so reader-gate failures are not seen | `test_a_reader_gate_failure_makes_a_prediction_unreadable_not_inconclusive` |
| 8 | the gate-failed branch is disabled | `test_a_failed_gate_freezes_every_prediction` |
| 9 | the McNemar direction clause is dropped (p alone decides) | `test_the_paired_mcnemar_is_the_hand_counted_discordant_pairs` |

All nine went red; the restored copy is green. (A tenth, accidental break — dropping a kept head but also the 60-head
guard — made test 1 fail on its own assertion, which is the same test.)

## DEVIATIONS from the design (and from the brief, where they touch)

1. **R13's assertion is taken through a one-run wrapper on `swap99.transplant_heads`.** `swap99.run` transplants and
   generates inside one call, and nothing outside `part221/` may change, so the assertion and the sign-flip receipt are
   computed inside that call. The wrapper restores the host itself when the assertion fails, because swap99's own
   restore is keyed on a completed transplant result that a raise never produces. Behaviour is the design's: the check
   happens after the write and before any generation, and its result is in R13's meta.
2. **The scorer re-expresses the coherence bar** (score219's `_bars`) with this part's constants instead of calling the
   private helper, so Part 219's prediction rows cannot move when Part 221's are added. Same values (15/13/not
   inconclusive).
3. **Module imports are package-style** (`from part219 import run219`, `from part99 import swap99`) so that a test and
   the module under test share one module object; Part 219's own internal imports are untouched. This matters because
   the wrapper patches a module-level function.
4. **R13 is not labelled when it equals R1**; the scorer reports that as the design's state ("not labelled: identical to
   R1") instead of as an unreadable run.
5. **The tiny fixture is not Part 219's exactly**: four layers and sixteen heads of head_dim 1 (Part 219's is three
   layers, two heads, head_dim 8). The registered kept heads, R14–R17's four layers and the 60-head complement only
   exist in that shape; the test module says so at the top.
6. **The pilot and the full run share `--out`**, so the full run reuses the pilot's rows. The design does not forbid it
   (same tags, same inputs) and it saves three runs' generation; every receipt is taken again on the full pass.

No conflict between the brief and the design was found: where both speak (runs, bars, pilot, gate), they agree.

## FILES TOUCHED

- `part221/run221.py` (new)
- `part221/score221.py` (new)
- `part221/test_part221.py` (new)
- `part221/steps221.txt` (new)
- `part221/REPORT.md` (this file)

Nothing outside `part221/` was created, edited or deleted; nothing was committed.

## BLOCKERS

- none.

## SELF-ASSESSMENT

- The sharpest uncertainty is the wrapper: it is the only way to take a receipt between `swap99`'s write and its
  generation without editing `part219`/`part99`, and it is tested (including the restore-on-failure path), but it is a
  monkeypatch of a module global and a reviewer may prefer a small hook parameter upstream instead. If `swap99.run`
  grows its own transplant callback, this should move to it.
- The pilot's stop rule treats a missing reproduction directory as a failed reproduction check. That is what makes the
  pilot stop before the first real run if `results/part219` is not in the job folder, which is the failure I wanted,
  but it means the job folder must carry Part 219's R0/R1 jsonl (listed in `steps221.txt`).
- The scorer's gate uses the manifest's reproduction block, so a scorer run over a directory whose manifest predates
  `run221.py` (or a hand-assembled one) fails the gate rather than guessing; the synthetic-world tests cover the shape.
- The end-to-end test runs the nine-run table on the tiny model in ~10 s; on the A770 the real run is ~30 minutes and
  the only thing it shares with the test is the code path, not the scale. I have not seen a real model, per the brief.
- `run221` does not catch a generation error from `swap99.run` (Part 219's `run_one` does not either): one failed run
  aborts the stage. Left consistent with Part 219 rather than changed here.
