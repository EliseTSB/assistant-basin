# Part 60 runbook

This experiment generates with Haiku on the workstation, scores with the pinned Qwen3-1.7B model on a rented 3090, then analyzes on the workstation. It uses fake responses in the worker tests; this runbook does not call Anthropic or start a rental.

## Stage the checkout before running

The runner resolves `results/` relative to its checkout. Keep this build at `/home/elisepc/assistant-basin/part60`, beside `/home/elisepc/assistant-basin/results`. If the build is elsewhere, copy its files into that location first or set `PART60_BASIN=/home/elisepc/assistant-basin`; both `gen60.py` and `score60.py` print this exact fix when required files are missing.

```bash
mkdir -p /home/elisepc/assistant-basin/part60
cp -a /path/to/part60-build/. /home/elisepc/assistant-basin/part60/
```

## Planned work and spend

Dev has 250 real units and 250 random-direction controls. M1, M2, M4, and M6 need one Haiku request per unit. M3 needs a selection request and then a fresh request for each parseable selected winner. M5 round 1 uses the M1 prompt; round 2 revises after scoring the first 12 sentences. M7 is skipped because the pilot records do not contain 24 top, 10 random, and 6 weak windows. M0 reuses the Part 58 baseline and makes no new call.

The registered plan estimates about 3,500 dev requests plus about 1,000 second-round requests if all seven methods are eligible, with a total Haiku spend of about $3-$6 and a hard `$8` cap. With the current M7 skip, expect at most 4,000 dev requests before failed parses. For Haiku 5.5 prompts up to 100K tokens, batch rates are $0.05 per million input tokens and $0.25 per million output tokens. Direct M6 smoke calls use $0.10 input and $0.50 output. The rates match the published list price per PRICE-ATTEST-2026-10-23.md; the invoice is final, and the first real batch usage is compared with the ledger. The checkout-wide `results/part60/spend.jsonl` ledger is fixed independently of `--root`. Each paid attempt gets a random ID and a durable conservative reserve before its API call; known returned usage settles it. A 4xx releases a reserve only when the response contains an Anthropic error body. Timeouts, connection failures, 5xx, 429, and kills keep the reserve counted. `ledger retire <attempt>` has no force option: a batch ID requires an API-reported terminal status and stays reserved until collection settles actual usage; no-ID attempts must be over 25 hours old. An unreadable manifest is treated as active. Settlement is refused if actual usage would exceed the cap. M6 smoke reserves and settles each direct request separately. A POST-returned batch ID is fsynced before the submitted manifest write; a rerun that finds it refuses a new POST and requires collection. The runner never adopts a batch by timestamp or count.

M3 or M5 cannot finish their second call until their first-round activations return from the GPU. The process is deliberately split into two local/GPU passes. The test phase is opened only after dev analysis. It contains M0, M1, and the one dev-selected candidate that passed the 5-point stop rule. If the selected candidate is M3 or M5, its test phase also uses the registered selection/revision step. No prompt is edited after the test phase begins.

## 1. Prepare thresholds and M0 locally

From the workstation checkout, create the fixed Part 58 thresholds and materialize the cached baseline texts. This reads the existing 482 MB activation archive but does not load Qwen.

```bash
cd /home/elisepc/assistant-basin
python3 part60/score60.py thresholds
python3 part60/gen60.py prepare --method M0 --split dev
```

The real-unit side of the seed-5801 sample is checked against `units.dev.json` and `units.test.json`; controls are selected from `pilot.jsonl`. Dev has 250 real and 250 control units. Test has the 247 real IDs in `units.test.json` and 250 controls; L0:u190, L4:u741, and L17:u1360 are excluded from reconstructed test.

## 2. Generate the dev prompts locally

Source the Anthropic key into the environment without printing it. Before M6, run the five-unit M6 smoke on dev. Each direct request uses adaptive thinking, `output_config: {effort: "high"}`, and 8,000 max tokens. The smoke passes only when every response contains non-empty thinking, stops for a reason other than `max_tokens`, and parses as the M6 schema. A failed all-five check is retried on five fresh units; M6 is dropped only after three consecutive failed checks. HTTP 400 counts as a failed check and its error body is saved. A timeout, 5xx, 429, kill, or other unknown result writes an error with no all-five check and resets the streak; its unresolved reserve remains counted. A later passing smoke clears a prior smoke-dropped state. Batch errors and truncations apply the registered greater-than-10% generation failure rule. A phase at or below 10% failures completes and can retry only its failed units; a phase above 10% is dropped. Collection stores raw results by custom ID before rewriting the generated rows, and analysis derives usage, calls, and truncations from those raw results.

API calls use the system certificate store. If the environment requires a private CA, set `PART60_CA_BUNDLE` to that bundle path before the run.

```bash
cd /home/elisepc/assistant-basin
set -a; . ~/.anthropic-credits.env; set +a
python3 part60/gen60.py submit --method M1 --split dev --phase primary --cap 8
python3 part60/gen60.py collect --method M1 --split dev --phase primary
python3 part60/gen60.py smoke --method M6
python3 part60/gen60.py submit --method M6 --split dev --phase primary --cap 8
python3 part60/gen60.py collect --method M6 --split dev --phase primary
python3 part60/gen60.py submit --method M7 --split dev --phase primary --cap 8
python3 part60/gen60.py listing
python3 part60/gen60.py ledger retire ATTEMPT_ID
```

Repeat the M1 submit/collect pair for M2 and M4 with `--phase primary`. M7 writes an explicit skipped record without submitting. M3 starts with `--phase selection`; M5 starts with `--phase round1`. Use the same `--cap 8` for every submission. After an ambiguous POST with no returned batch ID, rerun `submit` for the same method and phase; it creates a new attempt with its own durable reservation. A returned batch ID is durably saved before the manifest is updated; if a rerun finds it, collect that batch before continuing. If the extra reservation would exceed the cap, the runner refuses before POST. `listing` reports all batches returned by the API for operator comparison; it never changes a manifest, selects a batch, or authorizes collection. Retirement has no force override: a batch ID requires an API-reported terminal status and remains reserved until collected; an attempt without a batch ID must be over 25 hours old. Unreadable manifests are treated as active.

```bash
python3 part60/gen60.py submit --method M3 --split dev --phase selection --cap 8
python3 part60/gen60.py collect --method M3 --split dev --phase selection
python3 part60/gen60.py submit --method M5 --split dev --phase round1 --cap 8
python3 part60/gen60.py collect --method M5 --split dev --phase round1
```

## 3. Package the first scoring pass

Create a package containing only the runner and the exact Part 56/58 records it reads. Do not include review folders, key files, or local credential files.

```bash
cd /home/elisepc/assistant-basin
tar -czf /tmp/part60-inputs.tgz \
  --exclude='part60/review' --exclude='part60/review/**' --exclude='part60/key' --exclude='part60/key/**' \
  --exclude='part60/keys' --exclude='part60/keys/**' --exclude='part60/**/review/**' --exclude='part60/**/key/**' \
  --exclude='part60/**/__pycache__' --exclude='results/part60/review' --exclude='results/part60/review/**' \
  --exclude='results/part60/key' --exclude='results/part60/key/**' --exclude='results/part60/keys' --exclude='results/part60/keys/**' \
  part60 \
  results/part56-pilot/pilot.jsonl results/part56-pilot/label/atlas-labels.jsonl \
  results/part58/gen.all.jsonl results/part58/gen.dev.jsonl results/part58/gen.test.jsonl \
  results/part58/ids.all.json results/part58/units.all.json \
  results/part58/units.dev.json results/part58/units.test.json results/part58/sent.all.jsonl \
  results/part58/acts.all.npz results/part58/lift.all2.json results/part60
```

Upload it to the rented 3090. Keep the Qwen weights on the rental, not in the return tarball.

```bash
scp -P SSH_PORT /tmp/part60-inputs.tgz root@SSH_HOST:/root/
ssh -p SSH_PORT root@SSH_HOST
mkdir -p /root/part60
tar -xzf /root/part60-inputs.tgz -C /root/part60
nohup bash /root/part60/part60/run60.sh >> /root/part60/run60.out 2>&1 < /dev/null &
echo $! > /root/part60/run60.pid
```

Poll `/root/part60/run60.out` until it prints `RETURN_READY` or `RUN_FAILED`; inspect per-phase files in `/root/part60/logs/`. `run60.sh` downloads the pinned snapshot `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, scores the packaged dev methods, and verifies their generated phases and saved score metadata. It opens test only after `verify-dev` succeeds, `out/test/M0.jsonl` and the dev-generated `analysis/test-selection.json` marker are present, and the marker's SHA-256 of the current dev inputs matches. The marker restricts scoring to M0, M1, and the selected candidate. Reproduction requires median ratios 0.9–1.1 and Spearman at least 0.98 overall and within each unit type. Completed phase archives are recognized on rerun and score files with matching inputs and configuration are skipped.

The runner has a four-hour total runtime limit by default. Set `PART60_MAX_RUNTIME_SECONDS` lower for a shorter rental window; commands time out at the remaining limit and produce an incomplete archive.

The expected dev scoring set is up to 22,000 primary sentences for M1, M2, M4, and M6 (M4 has eight sentences per unit); up to 9,000 M3 selection sentences plus 4,000 fresh sentences; up to 6,000 M5 round-1 sentences plus 4,000 round-2 sentences; and 2,000 M0 texts rescored on the same device. Actual totals fall when output parsing fails or methods are skipped. The Qwen pass is one model load per invocation and uses batches of eight by default. A 3090 has 24 GB VRAM; `--batch-size 8` is the default, with `--batch-size 1` available if the rental image has less usable memory.

## 4. Return selection activations and finish M3/M5 locally

After `RETURN_READY`, copy the archive back before destroying the rental:

```bash
scp -P SSH_PORT root@SSH_HOST:/root/part60-return.tgz /tmp/part60-return.tgz
mkdir -p /tmp/part60-return-staging
tar -xzf /tmp/part60-return.tgz -C /tmp/part60-return-staging
mkdir -p /home/elisepc/assistant-basin/results/part60
cp -a /tmp/part60-return-staging/results/part60/. /home/elisepc/assistant-basin/results/part60/
```

The first return includes `thresholds.json`, the reproduction check, and activation matrices. Dev expects 2,000 M0 texts; 1,988 exist for 497 units, and the three dev controls without a baseline count as failed. One test control also has no M0 baseline and counts as failed. Existing M0 texts are rescored with the same pinned scorer, dtype, and device as generated methods; `M0-archive.npz` keeps Part 58 archive values as a cross-check. Choose M3's winner from its selection sentences, then request its fresh sentences. For M5, the round-two request includes only the original sentence, tier, and fired yes/no outcome; it never includes activation numbers.

```bash
python3 part60/gen60.py select-m3 --split dev
python3 part60/gen60.py submit --method M3 --split dev --phase fresh --cap 8
python3 part60/gen60.py collect --method M3 --split dev --phase fresh
python3 part60/gen60.py submit --method M5 --split dev --phase round2 --cap 8
python3 part60/gen60.py collect --method M5 --split dev --phase round2
```

Repackage with the same exact tar list and run `run60.sh` again. Existing completed activation files are skipped; the new M3/M5 phases are scored. Return the second archive, then run local analysis:

```bash
scp -P SSH_PORT root@SSH_HOST:/root/part60-return.tgz /tmp/part60-return.tgz
mkdir -p /tmp/part60-return-staging
tar -xzf /tmp/part60-return.tgz -C /tmp/part60-return-staging
mkdir -p /home/elisepc/assistant-basin/results/part60
cp -a /tmp/part60-return-staging/results/part60/. /home/elisepc/assistant-basin/results/part60/
python3 part60/analyze60.py --split dev
```

Follow the dev stop rule. `analyze60.py --split dev` writes `results/part60/analysis/test-selection.json`; `run60.sh` requires that marker before opening test and accepts only M0, M1, and the single candidate it names. A variant advances only when its real-minus-control gap leads M0 by at least 5 points for both positive-only and full-rule pass rates. Paired-bootstrap 95% intervals are reported beside the point leads and are not a gate. If no variant passes both point rules, keep M0. If a candidate qualifies, materialize M0 test using `python3 part60/gen60.py prepare --method M0 --split test` and generate M1 test with one M1 submit/collect pair. For an M3 candidate, generate/collect `M3 --phase selection`, score its test selection matrix, return it, run `select-m3 --split test`, then generate/collect `M3 --phase fresh` and package again to score fresh sentences. For an M5 candidate, generate/collect `M5 --phase round1`, score and return it, then generate/collect `M5 --phase round2` using the boolean feedback and package once more. The runner verifies saved score metadata before returning. If the stop rule keeps M0, 60j remains unscored.

## 5. Return files and destroy the rental

`RETURN_READY` appears only when setup, model fetch, every requested scoring phase, and archiving succeeded. In the first pass, test scoring is intentionally skipped; immediately before `RETURN_READY`, the runner prints `PHASES: dev-only`. A later two-pass run prints `PHASES: dev-and-test`. A failure writes `/root/part60-return-INCOMPLETE.tgz` when the available result files can be archived. Resume from the same workspace; the runner returns zero for already completed score phases. `run60.sh` records the active phase PID without passing its run-lock descriptor to children. After a hard kill, a rerun checks the stale PID and terminates it only when its command still names the Part 60 scorer; the scorer's own lock prevents overlapping score work. Run output is appended to `run60.out` on rerun. The tar output contains only score matrices, reproduction records, thresholds, logs, and software versions, and excludes review and key folders.

Copy the successful or incomplete archive and verify the local copy before destroying the instance. Replace `REMOTE_ARCHIVE` with the path printed by `RETURN_READY` or `RUN_FAILED`:

```bash
REMOTE_ARCHIVE=/root/part60-return.tgz
# For RUN_FAILED, use /root/part60-return-INCOMPLETE.tgz instead.
cd /home/elisepc/assistant-basin
scp -P SSH_PORT "root@SSH_HOST:$REMOTE_ARCHIVE" /tmp/part60-return.tgz && tar -tzf /tmp/part60-return.tgz >/dev/null && \
  python3 cloud/vast-part54-2026-10-22/vast.py destroy INSTANCE_ID
```

If the runner process disappears without `RETURN_READY` or `RUN_FAILED`, keep the rental alive, check `/root/part60/run60.pid`, inspect the phase logs and manifests, then rerun from that workspace. The wrapper checks and clears its own stale scorer PID safely before retrying. An ambiguous batch attempt stays reserved; resubmitting creates a new attempt and is constrained by the cap. A hard `SIGKILL` cannot run archive cleanup, so copy and inspect any available files before destroying the instance.

## Unverified probe checklist

- [ ] Verify the batch-listing cursor direction (`after_id`) against the live API before treating `listing` pagination as complete. Listing is report-only and never authorizes collection.

Run `python3 -m unittest part60/test_part60.py` locally. The suite uses a synthetic pilot, fake batches, and a toy scorer; only the registered split membership check needs Part 58 files and skips with an explanation when they are absent. It also runs the GPU wrapper in stub mode to verify preflight-independent `RETURN_READY` behavior without pip, network access, credentials, or a model load. Stub mode requires both `PART60_STUB=1` and the explicit `PART60_ALLOW_STUB=1` flag.
