# Build brief: Part 221's runner, scorer and tests

You are building code for a registered experiment in the public research repo `assistant-basin`, working in the
worktree `the repository root` (branch `part221-design`). Read these first, in order:

1. `PART221-DESIGN.md` (registered v3): the design. Its Runs table, Controls, Gate, Reading rules and Headlines are
   the specification. Where this brief and the design differ, the design wins; say so in your report.
2. `PART219-DESIGN.md`, `results/part219/README.md`: the part this one follows.
3. `part219/run219.py`, `part219/score219.py`, `part219/test_part219.py`, and `part99/swap99.py` (`transplant_heads`,
   `restore_transplant`, `copy_components`, `snapshot_components`, `attention_probe`, `coupling_log`). Part 221 reuses
   all of this. Prefer importing it over copying it.

## What to write (all inside `part221/`; do not edit any file outside it)

1. **`part221/run221.py`**, an orchestrator modelled on `run219.py`'s stage b (one host load and one donor load in one
   process), with the runs of the design's table: R0, R1, R13, R11, R12, R14, R15, R16, R17.
   - The four kept heads are `L0H10, L1H4, L2H6, L3H5`. The complement is every other head of layers 0 to 3 (15 per
     layer, 60 total), computed from the model's head count and asserted to be 60 on the real model.
   - R1 uses Part 219's `swap: attn, layers: 0-3` path. R13 uses `transplant_heads` over all 64 heads, donor mode. R11
     uses the 60, donor mode. R12 uses the 60, opposite mode. R14 to R17 use one layer's 15 complement heads, donor mode.
   - **R13 weight assertion**, before R13 generates: after the transplant, every `self_attn` parameter of layers 0 to 3
     equals the donor's exactly (`torch.equal`). Record it in R13's meta.
   - **Sign-flip receipt** (new, design Controls): for every transplant run, count per layer the `q_norm.weight` and
     `k_norm.weight` elements whose sign after the write differs from Instruct's before it (zero counts as its own sign;
     record the rule). Record the count and the min and max `norm_ratio` over the run's slice records in the meta.
   - **Coupling log**: Part 219's `coupling_log` already reports each untouched head's relative drift. For R11 and R12
     the untouched heads are exactly the four kept heads. Record them under a key that names them, and the per-layer
     rms(q), rms(k) and rms(attention output) relative changes, as in Part 219.
   - **Reproduction check**: compare this part's R0 and R1 with `results/part219/R0.jsonl` and `results/part219/R1.jsonl`
     token for token (`token_ids`) and record the result in `manifest.json`. Compare R13 with R1 the same way.
   - Reset receipt after every run that changes the model, as in Part 219.
   - `--pilot` runs R0, R1 and R13 only, and stops (non-zero exit, a clear message, a `PILOT.json` verdict) if any of the
     three checks above fails or a receipt fails.
   - Pin and record `torch.__version__` and `transformers.__version__` in each run's meta; refuse to start (a clear
     message) if they differ from `2.13.0+xpu` and `5.13.0`, unless `--allow-version-mismatch` is given, which is
     recorded.
   - A progress line at least every eight seconds while a run is going (the GPU job runner kills silent jobs).
   - The kernel record check: reuse Part 219's (`results/part219/kernel-check.json`, `tools/part_check.kernel_check`).

2. **`part221/score221.py`**, reusing `score219.py`'s functions (read set, net counts against each run's own unpushed
   answers, the 16-upstream / 5-downstream split, exact McNemar, the two-pass reader gate, coherence bars). It writes
   `results/part221/scorecard.json` and `SCORECARD.md` with:
   - The gate (kernel, R0 at least 17 of 21, R1 at most 5, reproduction of R0 and R1, R13 equal to R1). If the gate
     fails, every prediction is frozen and the scorecard says so first.
   - 221.1, 221.2, 221.3 exactly as registered in `PREDICTIONS.md` under Part 221, with the reading rules in the
     design's order: coherence bar for R11; for 221.3, 221.1 must hold and R12 must clear the coherence bar; the paired
     McNemar of R11 against R12 uses `score219.py`'s shows-relative-to-own-unpushed indicator.
   - Reader-gate failures make a prediction "unreadable", not "inconclusive".
   - Lost and gained unit lists for every run, and R11's and R12's overlap.
   - The coupling numbers for the headline: the largest kept-head drift and the largest |relative change in
     rms(attention output)| over layers 0 to 3, for R11 and R12. R12's sign-flip counts and norm-ratio range.
   - Exploratory: R14 to R17 (each marked unreadable below the coherence bar), downstream units, all-21 counts, and
     R11 net plus Part 219's R8 net against R1's net (read Part 219's labels from `results/part219/`).
   - Labels for R0 and R1: if the reproduction check says identical, the scorer reads Part 219's label files for them;
     otherwise it requires this part's own label files.

3. **`part221/test_part221.py`**, pytest, on Part 219's tiny random OLMo 2 fixture (import or copy the fixture from
   `part219/test_part219.py`). Required, each shown red when the property is broken (do the break on a copy, and list
   each break and its red test in your report):
   - The complement list is the right size and never contains a kept head.
   - R13's weight assertion passes on the tiny model and fails if one slice is not transplanted.
   - The sign-flip count is right on a hand-built case (a known element flips under opposite mode).
   - The orchestrator runs **end to end on the tiny model**, through its real `main()` path with `--pilot` and then a
     full run, writing every file the scorer reads. (Part 219's means script failed three times on the GPU because no
     test ran its main path; this test is the fix.)
   - The scorer computes 221.1 to 221.3 correctly on synthetic labels, including: R12 below the coherence bar makes
     221.3 inconclusive; 221.1 failing makes 221.3 not read; a reader-gate failure makes a prediction unreadable; the
     paired McNemar on a hand-counted example.
   - The scorer refuses to read predictions when the gate fails.

4. **`part221/steps221.txt`**: the launch lines (pilot, then full), in the form of `part219/steps219.txt`, with the
   `--copy` list including every module the job folder needs (check imports, including `part219/run219.py` and
   `part219/score219.py` if you import them, and `results/part219/kernel-check.json` plus Part 219's R0 and R1 jsonl
   for the reproduction check if they are not already in the job folder).

## Rules

- Run the tests with `~/neuron-probe/.venv/bin/python -m pytest part221 -q` from the worktree root. They must pass
  before you finish. Also run `part219`'s tests, to show you broke nothing.
- Do not load any real model, call any API, or touch `results/part221` beyond what tests create in temporary
  directories.
- Do not edit files outside `part221/`. If something outside must change, stop and say what and why in your report.
- Do not commit; leave the files in the working tree. Your parent commits after review.
- Write `part221/REPORT.md` early (a skeleton within your first fifteen tool calls) and keep it current: SUMMARY,
  WHAT YOU BUILT, TESTS (with the break-and-red list), DEVIATIONS from the design, BLOCKERS. Keep messages short.
