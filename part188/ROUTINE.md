# Part 188 cloud session

Run the registered Part 188 collection and analysis from the repository root. Follow the fixed design in
`PART188-DESIGN.md`; do not change classes, thresholds, folds, prompts, or the registered analysis. Do not load a model
or inspect result values before the session begins. Do not send notifications.

1. Use branch `cloud/part188`, unless the session prompt supplies `BRANCH`. From the repository root run:

   ```bash
   BRANCH=${BRANCH:-cloud/part188}
   START=$(git rev-parse HEAD)
   if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null; then
     git fetch origin "$BRANCH"
     git checkout -B "$BRANCH" FETCH_HEAD
     git merge --no-edit "$START"
     git merge-base --is-ancestor "$START" HEAD
   else
     git checkout -b "$BRANCH" "$START"
   fi
   mkdir -p cloud-results/part188
   ```

   Do not merge `origin/main`. Write session outputs only under `cloud-results/part188/`.

2. Install and use one interpreter for every command. Resolve it once with `PY=$(command -v python3.13 || command -v
   python3)`. Install the CPU PyTorch wheel with
   `"$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.2`, then install
   `"$PY" -m pip install -r part188/requirements188.txt`. Verify the environment with
   `"$PY" -m pip check` and the driver's version check.

3. From the repository root run `"$PY" -m pytest -q part188`. If any test is red, save the complete output to
   `cloud-results/part188/pytest.txt`, commit and push that file, then stop without recording audio.

4. Start the restartable driver with
   `nohup "$PY" part188/session188.py --work-dir /tmp/part188-state --out-dir cloud-results/part188 >
   cloud-results/part188/driver.out 2>&1 &`. The driver checks the pinned stack, resumes completed per-clip files
   while `/tmp/part188-state` survives, retries clips marked `download_error`, records the eight parity clips first,
   applies Gate 1, records the remaining manifest, and then runs the analysis. A recycled container starts recording
   from the beginning; do not push per-clip record files to preserve progress.
   If Gate 1 fails, it stops before recording any non-parity clips. Do not end the session while the driver is running.
   The pinned archive is checked against the local `esc50.csv` SHA-256 recorded in `folds.json`; stop if it differs.

5. Wait in the foreground in intervals of at most nine minutes, then check `driver.out` and `cloud-results/part188/`
   every 10 to 15 minutes. If the process ended without `summary.json` and the container still has its work directory,
   restart the same driver command with `/tmp/part188-state` so finished clip rows are skipped. Every 45 minutes, copy
   `/tmp/part188-state/done.jsonl` into `cloud-results/part188/` when present, then commit and push the progress log,
   driver output, and done list. Do not copy or commit audio or any per-clip activation row.

6. When the driver finishes, include files under `cloud-results/part188/` only if each is under 90 MB. Never add
   audio, the scratch `records/` directory, or full activation rows. Use `git add -f cloud-results/part188` because
   generated results may be ignored. Commit and push the branch.

7. Verify with `git ls-tree -r --name-only HEAD cloud-results/part188/` that it lists `results.json`, `unit_auc.npz`,
   `top_units.npz`, `emb188.npz`, and `summary.json`. If a required file is missing, add it with `git add -f`, commit,
   push, and check again. Do not read or interpret activation values or selected-unit values in this session.

8. End with a `CLOUD-RESULT:` block of at most 1,000 characters containing branch, commit, Gate 1/2/3 verdicts,
   per-set planned and recorded counts, wall times, problems, and no unit numbers.
