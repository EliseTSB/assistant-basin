# Part 206, Amendment 3: the Spearman correlation is computed in numpy

Written 2026-11-12 17:47 CT by Claude (Opus 5.5), after step 2's first Qwen run failed and before any GT result
existed. No prediction, bar, size or reading rule changes.

**What failed.** Step 2's Qwen3.5-4B-Base moments pass ran on the A770 from 15:00:31 to 15:20:35 CT and summed all
600 passages (G0, G0.5 and G1 passed and are recorded beside the partial sums). It then died in `gt_static` on
`ModuleNotFoundError: No module named 'scipy'`: `common206.spearman` imported `scipy.stats.spearmanr`, and the Windows
GPU environment has no scipy (checked: numpy, torch, transformers, pyarrow, safetensors and accelerate are present;
scipy is not). The runner's retry at 15:20:42 resumed from the partial sums and died the same way at 15:21:07. The
smoke did not reach this line because a smoke never computes GT's static test. No GT number was computed, so nothing
was read.

**Change.** `spearman` is now Pearson's r on average ranks in numpy (ties take their mean rank, as in
`scipy.stats.rankdata`), returning None wherever `scipy.stats.spearmanr` returns nan (a NaN input, fewer than two
values, or a constant input). A test checks it against scipy to 1e-12 on 11 cases (sizes 2 to 400, five with heavy ties) and on three edge cases, and a second
test fails if `record206.py` or `common206.py` imports scipy (it fails on the old line).

**Relaunch.** The partial sums hold all 600 passages and the plan hash covers the configuration, prompts and stage,
not the code, so the relaunch reuses them: it loads the model (GT's static test needs its gate rows), computes G2 and the static test from the stored sums, and saves. No
passage is run again.
