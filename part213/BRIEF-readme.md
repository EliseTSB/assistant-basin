# Brief: draft Part 213's results README, its report, its figure and its number check

You are a DeepSeek V4.1 Flash worker in Journal Claude's research repo (`/home/elisepc/green/worktrees/ab-neon-review`,
a git worktree of the public `assistant-basin` repo). Part 213 has run and is scored. Your job is the write-up draft.
Journal Claude reads every sentence against the numbers afterwards, and a separate cold check follows, so accuracy
beats polish. **Do not commit, do not edit `PART213-DESIGN.md`, `PREDICTIONS.md`, `part213/score213.py` or anything
under `results/part213/full/` or `results/part213/audit/`. Do not run anything on the GPU.**

Python: `/home/elisepc/neuron-probe/.venv/bin/python -I -B <script>`. Under `-I` a script's own folder is NOT on
`sys.path`, so a script in `part213/` that imports `score213` must insert its own folder first
(`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`).

## What to make (four files)

1. **`part213/report213.py` → `results/part213/report213.md`.** Computes and prints every number the README uses, from
   `results/part213/full/score.json`, the records in `results/part213/full/` (through `score213.load_records`,
   `load_targets` and `Stage1`, as `score213.py`'s own `main` does), `results/part213/audit/`, and
   `results/part213/plain-probe.json`. Nothing typed by hand. Print at the precision the README will use.
2. **`part213/fig213.py` → `results/part213/fig213.png`.** One figure that shows the claim, drawn by code from the
   result files. Style guide: `/home/elisepc/green/Journal/sources/style/sauers-magician-2026-10-20/` (read it; plain
   labels, the claim in the figure's own title, readable at phone width). Suggested panels, change them if the data say
   otherwise: (a) per-unit `L_z` for the 21 slot units on `SELF_HELD`, `THIRD_SLOT` and the non-identity copula slot
   (the prompts 213.10 reads), so the reader sees that the slot units write into ordinary "X is a ___" slots too;
   (b) the same for concept units; (c) the random-arm controls beside them. Matplotlib is in that venv; check.
3. **`results/part213/README.md`.** Start from `templates/RESULTS-README.md` and follow `PROCEDURE.md` section 8.
   `results/part210/README.md` is a good model of length and tone. Plain English, short sentences, no em dashes.
4. **`part213/check_readme213.py`**, modelled on `part210/check_readme210.py`: every number in the README must appear
   in `report213.md` or `score-table-audit.txt`, or in an `ALLOWED` list with a reason per entry. Exit 1 on any stray
   number. It must exit 0 on your final README, and you must show it exiting 1 when you plant a wrong number (then
   remove the plant).

## The README's content

- **Title:** the result in one plain sentence, no stronger than the table. The scorer's headline reading is
  `inconclusive / mixed selectivity`. 5 held (213.1, 213.2, 213.5, 213.9, 213.11), 2 failed (213.8, 213.10),
  3 inconclusive (213.3, 213.4, 213.7), 213.6 untestable (Gate 3 failed). Check these against `score.json` yourself.
- **Predictions table:** the registered text **verbatim** from the `## Part 213` section of `PREDICTIONS.md`, with its
  confidence and author (all by Gemini 4 Neon), the number from `score.json`, the verdict. Do not paraphrase.
- **Skeptical questions**, each with its table and the script and file the numbers come from. At least:
  1. *Are these first-person self-slots?* 213.10: the slot units also write their persona into non-identity
     copula slots ("X is a ___"), so a "write this noun after *is*" reading competes with a self-slot one. Say what the
     table shows; label any interpretation as a reading, not a finding.
  2. *Is it the neuron or any push of that size?* The random-arm controls (`random_0` to `random_3` at instruct).
  3. *Do they hand a stranger their persona in free text?* 213.6 is untestable: two blind readers (Neon, and a
     DeepSeek reader on a 40-item sample) both got 10 of 10 known examples right but reached kappa 0.3496 (raw
     agreement 0.80, most agreements on one label) against the registered 0.60. State Neon's labels only as
     description, never as a verdict: genuine third-person persona attributions at `+0.8` were 1 of 42 for slot
     neurons, 0 of 42 for slot `random_0`, 9 of 36 for concept neurons, 0 of 36 for concept `random_0`. The cold
     checker hand-read 10 and would label one (blind 72) differently; that leaves slot at 2 of 42 at most.
  4. *Does sensitivity at a small step predict the operating dose?* 213.8 failed: n(J>0) 14 of 21 against 17; the
     context-contrast rho 0.213 against 0.40; its rho(J, L_z) clause held at 0.5104.
  5. *Does the self/third split exist before DPO?* 213.9 (SFT already).
- **Method:** one paragraph from the design: OLMo 2 1B, the checkpoints used (read the revisions from the design and
  the run files, never "main"), CPU float32 eager, chat template, doses, the off-diagonal-controlled `SpecLogit`, the
  unit-clustered bootstrap (10,000 draws, seed 213), the readers.
- **Things that happened during the run (the README must say all of these, plainly):**
  - Amendment 1 (in the design, published on public main before any Stage 1 pass) fixed implementation reading 1,
    which as first written would have dropped 936 passes by treating two different quantities as one.
  - The FR step died once on its first attempt: the DPO snapshot on disk is `pytorch_model.bin` and the loader read
    only safetensors. Fixed and rerun; read `results/part213/chain.log` for what you can confirm and say only that.
  - The scorer's 213.8 J was the neuron arm's slope alone (the design's J is the neuron slope minus `random_0`'s),
    found by a DeepSeek cold check of the scored numbers after scoring and fixed in commit `e9650814`; the verdict was
    `failed` both before and after. The cold check reported rho values 0.3810 and 0.1429 that do not reproduce; an
    independent recomputation (`part213/coldcheck/recheck_213_8.py`, commit `1e980869`) gives the scorer's 0.5104 and
    0.2130.
  - The audit scorer would have read the reader labels with string keys against integer blind ids and found none;
    fixed before either reader ran (commit `553904e1`).
  - 213.11's text says "mean"; the design's implementation reading 7, fixed before any data, reads it as the median of
    the 21 per-unit values. Quote that reading; the mean (6.5260, per the cold check) also clears the bar, as
    description only.
- **Limits:** one model family at 1B, the prompt families as registered, Stage 2 greedy only, reader agreement too low
  for 213.6, anything else you find.
- **Files:** what each holds and what made it.
- Leave the last line as `Cold check: pending`.

## Report contract

Write `REPORT.md` in your run directory: SUMMARY (what you made and whether `check_readme213.py` exits 0), FINDINGS
(anything in the data or the code that looks wrong to you, with the command that shows it; do not fix the scorer),
BLOCKERS. If a number you need is not computable from the files, say so in the README draft as a visible
`TODO(number): ...` line rather than inventing it.
