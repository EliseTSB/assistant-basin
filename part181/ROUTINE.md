# Part 181 cloud session: collect one checkpoint

You are one of 15 parallel Claude cloud sessions for Part 181 (`PART181-DESIGN.md`, registered before any data). Your
prompt names one KEY (a checkpoint key from `part85/checkpoints.py`, or `instruct_b`, `pretrain_1T_b`,
`pretrain_4T_bf16r`). Do exactly this, and nothing else. Do NOT send any notification of any kind. Do not read,
summarise or interpret any activation, window or top list: this session collects; the analysis happens elsewhere.

1. If `origin` already has the branch `cloud/part181-KEY` (`git ls-remote origin cloud/part181-KEY`), this is a
   resume: first record the commit you were given (`START=$(git rev-parse HEAD)`), then
   `git fetch origin cloud/part181-KEY`, check it out (`git checkout -B cloud/part181-KEY FETCH_HEAD`), and
   `git merge --no-edit $START` so the fixed code is used (do not merge `origin/main`: in this clone it can be stale).
   Check the merge worked: `$START` must now be an ancestor (`git merge-base --is-ancestor $START HEAD`). The output
   directory then already holds the saved state. Otherwise create and switch to the branch
   `cloud/part181-KEY` from the commit you were given. Write only under `cloud-results/part181/KEY/`.
2. Install the pinned environment, CPU only, into ONE interpreter and use that interpreter for every later
   command (on these machines `pip` and `python` can be different Pythons): `PY=$(command -v python3.13 || command -v
   python3)`, then `$PY -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0` and
   `$PY -m pip install numpy==2.4.6 transformers==5.22.1 tokenizers==0.23.3 huggingface_hub pyarrow pytest`.
3. Run `$PY -m pytest -q part181` from the repository root. If any test fails, stop: commit the pytest output as
   `cloud-results/part181/KEY/pytest.txt`, push, and end with the CLOUD-RESULT block saying so.
4. Start the session driver in the background, logging to a file:
   `nohup $PY part181/session181.py --key KEY --work /tmp/p181 --out cloud-results/part181/KEY > cloud-results/part181/KEY/driver.out 2>&1 &`
   It downloads the checkpoint and the corpus, checks the identity hashes, collects (about 2.5 hours), runs Gate 1 and
   writes `summary.json` when finished.
5. Do NOT end your turn while the driver runs: an idle session's container is recycled after about 30 minutes,
   which kills the driver. Wait in the foreground with commands of at most 9 minutes each, for example
   `timeout 560 bash -c 'sleep 540'` followed by a check. If the driver is no longer running and there is no
   `summary.json`, start it again with the same command: it resumes from `state.npz` (a change of machine is allowed
   and logged in `host-history.jsonl`). While it runs, check it every 10 to 15 minutes (tail `driver.out` and `session.log`). Every 45 minutes, and at the end
   of pass 1, commit and push whatever state files exist in the output directory (`state.npz`, `progress.json`,
   `manifest.json`, `session.log`, `driver.out`), so a stopped session loses little. Do not commit the downloaded model,
   the corpus or any export.
6. `state.npz` is ignored by `.gitignore`; add it with `git add -f`. If the driver stops with an identity (Gate 2)
   mismatch, do not retry: commit the log and stop. If Gate 1 fails, rerun
   Gate 1 once with the driver's `--rerun-of` option as its usage text says, keep both results, and do not loop.
7. When `summary.json` exists, commit everything in `cloud-results/part181/KEY/` except files over 90 MB (name any you
   left out), and push the branch.
8. End your final message with a block starting `CLOUD-RESULT:` of at most 1,000 characters: KEY, branch and commit,
   identity check result, Gate 1 verdicts (1a, 1b, 1c, 1d), wall time per stage, peak memory, and any problem. No
   numbers about any unit's activations.
