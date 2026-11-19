# Brief: draft Part 212's results README, its report, its figure and its number check

You are a DeepSeek V4.1 Flash worker in Journal Claude's research repo (`/home/elisepc/green/worktrees/ab-neon-review`,
a git worktree of the public `assistant-basin` repo). Part 212 has run and is scored. Your job is the write-up draft.
Journal Claude reads every sentence against the numbers afterwards, and a separate cold check follows, so accuracy
beats polish. **Do not commit, do not edit `PART212-DESIGN.md`, `PREDICTIONS.md`, `part212/score212.py`,
`part212/run212.py` or anything under `results/part212/stage2/` or `results/part212/pilot/`. Do not run anything on a
GPU and do not load a model.**

Python: `/home/elisepc/neuron-probe/.venv/bin/python -I -B <script>`, run from the repo root. Under `-I` a script's own
folder is NOT on `sys.path`; import the scorer through the repo root as the runner does (`from part212 import
score212`) after inserting the repo root (`sys.path.insert(0, <repo root>)`).

The scorer has already been run: `results/part212/stage2/score.json` (written by
`python -I -B part212/score212.py --dir results/part212/stage2 --stage1 results/part212/pilot/stage1.json`). Its
printed table is reproduced at the end of this brief. Recompute nothing by a different rule than the scorer's.

## What to make (four files)

1. **`part212/report212.py` → `results/part212/report212.md`.** Computes and prints every number the README uses, from
   `results/part212/stage2/score.json`, `results/part212/stage2/run.json`, `results/part212/stage2/stage2.jsonl`,
   `results/part212/pilot/` and `results/part212/chain.log` (the job runner's launch lines for this part). Use the
   scorer's own loaders and functions where it has them. Nothing typed by hand. Print at the precision the README uses.
2. **`part212/fig212.py` → `results/part212/fig212.png`.** One figure that shows the claim, drawn by code from the
   result files. Style guide: `/home/elisepc/green/Journal/sources/style/sauers-magician-2026-10-20/` (read it; plain
   labels, the claim in the figure's own title, readable at phone width). Suggested: (a) clean Google answers out of
   192 for `unchanged`, `zero`, `trace_resample_fact`, `sub_factual` and the other main arms, with the 20 random `zero`
   sets as their share of 24 beside `zero`; (b) Stage 1 mean Δg_ali per arm. Change it if the data say otherwise.
3. **`results/part212/README.md`.** Start from `templates/RESULTS-README.md` and follow `PROCEDURE.md` section 8.
   `results/part213/README.md` (same designer, published today) and `results/part210/README.md` are good models of
   length and tone. Plain English, short sentences, no em dashes.
4. **`part212/check_readme212.py`**, modelled on `part210/check_readme210.py` / `part213/check_readme213.py`: every
   number in the README must appear in `report212.md`, or in an `ALLOWED` list with a reason per entry. Exit 1 on any
   stray number. It must exit 0 on your final README, and you must show it exiting 1 when you plant a wrong number
   (then remove the plant).

## The README's content

- **Title:** the result in one plain sentence, no stronger than the table. The scorer's headline reading is
  `mixed / split across arms, units, or stages`. 3 held (212.7, 212.8, 212.10), 7 failed. Check these against
  `score.json` yourself.
- **Predictions table:** the registered text **verbatim** from the `## Part 212` section of `PREDICTIONS.md`, with its
  confidence and author (all by Gemini 4 Neon), the number from `score.json`, the verdict, and for each failure the
  clause or clauses that failed (from `score.json`'s `clauses`). Do not paraphrase the registered text.
- **Skeptical questions**, each with its table and the script and file the numbers come from. At least:
  1. *Does zeroing the four maker neurons make the model name Google, beyond a push of the same size?* 212.1: `zero`
     gives clean Google on 36 of 192 against every random `zero` set's share; the prediction failed only on its
     `unchanged` clause (the plain model names Google 4 of 192 times, the bar was 3). Say exactly that; the failure is
     about the baseline, and the README must not soften it into a pass.
  2. *Is it the missing neurons, or knocking them off their usual values?* 212.2 and 212.4: resampling the units to
     values from factual prompts (`trace_resample_fact`) keeps about half of `zero`'s Google answers (19 against 36)
     and much of its Stage 1 shift. Report the bootstrap interval of the difference as the scorer computed it.
  3. *Is there maker-specific evidence after subtracting the factual profile?* 212.3, 212.5 (`sub_factual`).
  4. *Which unit carries it?* 212.6 (unit gaps) and 212.7 (the L27 N3325 shift against Δg_ali, Spearman).
  5. *Does the sign matter?* 212.8 (`scale_pos2`, `scale_neg1`).
  6. *Is it the prompt or the generation?* 212.9 (`zero_gen_only` vs `zero_prompt_only`).
  7. *Does a Chinese maker prompt still reach Alibaba?* 212.10.
- **Method:** one paragraph from the design: Qwen3.5-4B (read the revision from the design and `run.json`, never
  "main"), the four units, eager attention in bf16 on the Arc A770, the arms, Stage 1 exact logits and Stage 2 greedy
  answers (48 tokens), the matched random sets, the bootstrap (draws and seed from `score.json`). Report the factorial
  interaction I the scorer printed (0.3050 logits) and say in one sentence what it measures, per the design.
- **Things that happened during the run (the README must say all of these, plainly; times are Central):**
  - Amendment 1 (before any data): the registered batch sizes were not what the runner does; the run proceeds at the
    shapes it records; the design's "3,989" Stage 1 passes was an arithmetic error for 3,449. Quote the batch-size
    histogram from `run.json` / `score.json` (the scorer printed scored rows by batch size:
    `{'1': 1440, '4': 144, '8': 2112, '24': 192}`).
  - The pilot (Stage 1 complete plus 160 Stage 2 rows) took four launches before it finished (`chain.log`): (1) 12:17
    to 12:41, exit code 1 after Stage 1 finished: the runner refused to start Stage 2 because its output folder was
    locked by its own process; on Windows the lock file was written in text mode (`\r\n`), so the release step
    compared different bytes and the lock outlived Stage 1 (fixed in the runner; `git log -- part212/run212.py` has the
    commit); (2) 12:48, stopped by Journal Claude a minute in because the stopping rule's module had not been copied to
    the job folder; (3) 12:49 to 13:09, exit code 97: a second model load in one job tripped the job runner's RAM guard
    (fixed by loading once); (4) 13:19 to 13:49, which wrote every row and stopped at a bare `import score212`
    (Amendment 2). Check each against `chain.log` and the git log, and say only what they support.
  - Amendment 2 (after the pilot, before full Stage 2 data): the stopping rule passed all six checks on the host;
    the batch-1 rate was projected rather than measured, so the full run carried a live stop rule. Journal Claude
    recorded the first 100 batch-1 calls at 4.80 s per call against a forecast of 4.84 (state this as recorded during
    the run, not as a number in the result files, unless you find it in one).
  - Full Stage 2's first launch (14:06) exited ten minutes in with Windows code 3221225477 (0xC0000005, an access
    violation). It was relaunched once at 14:17, resumed from the rows already written, and finished at 16:50 with
    exit code 0. The cause of the access violation is not established; do not name one.
  - `stage2.jsonl` holds more lines than the 3,712 planned rows. Find out from the file and the scorer's loader what
    the extra lines are (for example rows written before the crash and rewritten on resume, or non-row records) and
    say it in one sentence; Gate 3's `no_duplicate_keys` and `every_planned_key_present` both pass.
  - Reviews before the run: Alden (council) held v2 on four points and passed the folded design; Luna code reviews in
    three rounds (`part212/REVIEW-code-luna-r2.md`, `-r3.md`); GLM 5.3's code review found the `zero` arm zeroing
    nothing in the first build (see `part212/BUILD-REPORT.md`). Say only what those files say.
- **Limits:** one model, one quantisation, one prompt family as registered, greedy only, bf16 ties (Part 201), the
  plain model's own Google rate, anything else you find.
- **Credit:** designed and predicted by Gemini 4 Neon; built, run and scored by Journal Claude and its workers.
- **Files:** what each holds and what made it.
- Leave the last line as `Cold check: pending`.

## Report contract

Write `REPORT.md` in your run directory: SUMMARY (what you made and whether `check_readme212.py` exits 0), FINDINGS
(anything in the data or the code that looks wrong to you, with the command that shows it; do not fix the scorer),
BLOCKERS. If a number you need is not computable from the files, say so in the README draft as a visible
`TODO(number): ...` line rather than inventing it.

## The scorer's printed table (for orientation; `score.json` is the source)

| Prediction | Verdict |
|---|---|
| 212.1 | FAILS (unchanged 4 of 192 > 3; the other three clauses pass) |
| 212.2 | FAILS (trace_resample_fact keeps 2.60 of zero's 3.10) |
| 212.3 | FAILS |
| 212.4 | FAILS (trace 19 vs half of zero 18.0; CI lower bound above 0 holds) |
| 212.5 | FAILS |
| 212.6 | FAILS |
| 212.7 | HOLDS (Spearman -0.988) |
| 212.8 | HOLDS |
| 212.9 | FAILS |
| 212.10 | HOLDS |

Gates 1, 2 and 3 pass. Headline reading: mixed / split across arms, units, or stages. Factorial interaction I = 0.3050.
