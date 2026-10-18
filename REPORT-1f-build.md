# Part 1f build report

## SUMMARY

Implemented the local Clef-flash record/answer/replay path and the five-metric comparison. The WSL stub records stable state/question hashes, returns 0.5 in record mode, and refuses unknown replay hashes with HTTP 500. The Windows batch runner uses the cached `Cloudflare/clef-flash` snapshot and the snapshot's `joint_schema_model.py`; it writes resumable answers in small batches. `climb_test.py` can emit a per-unit sidecar for the paired comparison. `PREDICTIONS.md` was not changed.

Jev answers do not affect which simulator requests the scorer makes. `climb_eval.evaluate` creates the full held-out job list from fixed units, cached Luna descriptions, and contexts before it runs Jev calls. Returned probabilities populate item scores; there is no branch that adds, removes, or changes later requests based on those values.

## FINDINGS / WHAT SHIPPED

- `jev_stub.py`: stdlib HTTP server, canonical SHA-256 over sorted compact JSON for `state` plus `questions`, JSONL request logging, `X-Climber` ignored, strict replay misses.
- `run_1f.sh`: fresh work directories outside the live climb, copied Luna cache, empty Jev caches, TEST scorer invocation in record/replay mode, and aggregate/row outputs.
- `clef_batch_gpu.py`: local-only Hugging Face snapshot lookup, NF4/BF16 BitsAndBytes config, XPU batches defaulting to eight records, allocator warmup disabled, `torch.xpu.empty_cache()` between batches, and a requests/second log line. The actual release interface orders noul options using `EncodedQuestion.option_ids`; true is option zero. Its `load_release_model` forwards the quantization config and has no vision-skip option, so the standard backbone (including vision weights) is loaded.
- `compare_1f.py`: outputs five numeric-only rows for mean Spearman, per-unit Pearson correlation, real/random median top AUROC and gap, probability SD, and GPU throughput. It reads Jev probabilities from `work-test/cache/jev` and uses the same Spearman/AUROC/median helpers as the 1e scorer.
- TEST estimate: 150 simulated units × 30 held-out snippets × 1 noul question = 4,500 scored question answers/request opportunities. The historical `work-test/test-scores.json` records 4,481 Jev calls, consistent with cache reuse for duplicate request keys.
- Tests: 3 tests passed (`test_part1f`): stub handler record/replay/hash-order/unknown-500; GPU answer resumability with a stub model using the release option ordering; comparator fixture and five numeric output lines. Python AST checks passed for six Python files; `bash -n run_1f.sh` and `git diff --check` passed.

## BLOCKERS

The record/replay scorer and A770 inference were not run end-to-end in this worker. The checked-out Part 1 directory contains `units.json` and `key.json`, but lacks `heldout.json`, `jev-sim.json`, and `verdicts-gpt-6-luna-high-full.json`, which `climb_test.py` requires. WSL also cannot reach the Windows GPU host, and the worker sandbox denied loopback socket binding; the stub test therefore called the actual HTTP handler in-process rather than binding a local port. The control mailbox reports the Clef snapshot is cached on Windows, but no Windows execution happened here.

## TEST COUNTS

`/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part1f`: 3 passed, 0 failed. AST parse checks: 6 Python files. Bash syntax: passed. Diff whitespace check: passed.

## RUN COMMANDS IN ORDER (not executed here)

These commands use the WSL worktree and the run directory `/home/elisepc/neuron-climb-1f`. Before record/replay, point `PART1_DIR` at the complete frozen Part 1 run directory if it is not at the default path.

1. WSL record step:

   ```bash
   cd /home/elisepc/green/worktrees/part1f
   bash ./run_1f.sh record
   ```

2. Windows A770 smoke run, followed by the full run. The second invocation resumes the same answers file and skips the first 32 hashes. This discovers the current WSL distro name for the shared UNC paths.

   ```powershell
   $Distro = (wsl.exe --list --quiet | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
   $WslRoot = "\\wsl.localhost\$Distro"
   $Repo = "$WslRoot\home\elisepc\green\worktrees\part1f"
   $Run = "$WslRoot\home\elisepc\neuron-climb-1f"
   $Python = 'C:\Users\alexa\jlens-tools\.venv\Scripts\python.exe'
   $env:HF_HOME = 'E:\hf-cache'
   & $Python "$Repo\clef_batch_gpu.py" --requests "$Run\requests.jsonl" --answers "$Run\answers.jsonl" --limit 32 --batch 8 *> "$Run\clef-smoke.log"
   & $Python "$Repo\clef_batch_gpu.py" --requests "$Run\requests.jsonl" --answers "$Run\answers.jsonl" --batch 8 *> "$Run\clef-full.log"
   ```

3. WSL replay step:

   ```bash
   cd /home/elisepc/green/worktrees/part1f
   bash ./run_1f.sh replay
   ```

4. WSL comparison:

   ```bash
   cd /home/elisepc/green/worktrees/part1f
   bash ./run_1f.sh compare
   ```
