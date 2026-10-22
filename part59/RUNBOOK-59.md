# Part 59 runner runbook

This run executes the registered 59A story-setting ladder for Qwen3.5-4B L22:2742 and OLMo-2-1B L10:6978, then measures 59B later-layer MLP responses on the saved Part 26 corpus. The runner uses the Part 26 Pile maxima already recorded by Part 54 for Qwen and the Part 26 target record for OLMo. It does not fetch a new corpus.

## Expected work, time, and memory

Each model has 12 prompts x 5 doses x 8 primary samples = 480 generations. The -2 and +2 doses add 12 prompts x 2 doses x 3 random controls x 8 = 576 generations. This is 1,056 per model and 2,112 total. Dose 0 is the unclamped baseline; nonzero doses generate the target unit. At Part 54A's measured 480 answers in about 11 minutes on a 3090, 59A is roughly 24 minutes per model, or 49 minutes total before model load and file-write overhead. OLMo is smaller, so its actual time may be shorter.

For 59B, each model makes one baseline pass and eight clamp passes (-1 and +1 for the target plus three random controls), over 1,000 documents. Analysis makes one additional baseline pass to save top activation windows for the top symmetric units. At about 350 tokens per 256-word document, that is roughly 3.5 million input tokens per model. Planning at 350-900 input tokens/second for Qwen and 1,000-2,000 for OLMo gives about 1.1-2.8 hours for Qwen and 30-60 minutes for OLMo. These are planning ranges; prefill throughput has not been benchmarked on this run's exact rented image.

At the Part 54 rental rate of $0.2857/hour, budget about 3-5 hours total including prefetch, model loading, 59A, 59B, and top-window extraction, or about $0.86-$1.43. Confirm the current offer price before renting.

Qwen3.5-4B in fp32 has about 4 billion parameters, or 14.9 GiB of raw weights. Its MLP intermediate width is 9,216. With documents streamed one at a time, a 350-token MLP input is about 12.3 MiB; weights, CUDA workspace, and allocator overhead put the planned peak around 16-19 GiB VRAM. `run59b.py` defaults to `--batch-size 1` for this reason. A single batch of 1,000 documents at 350 tokens would require about 12.0 GiB for one MLP activation tensor alone, plus 14.9 GiB of model weights and attention/workspace memory, so it exceeds a 24 GiB 3090. The numbers are estimates, not measurements; the returned manifest records CUDA peak allocation and reservation when available.

## Package the inputs

From the assistant-basin checkout, include the Part 59 scripts and only the source records they need:

```bash
cd /home/elisepc/assistant-basin
tar -czf /tmp/part59-inputs.tgz \
  --exclude=part59/review --exclude=part59/__pycache__ \
  part59 \
  part26.py part20.py part20c.py part20d.py part20f.py neuron_report.py judge_client.py \
  results/part26/source.json results/part26/source.npz \
  results/part26/target-olmo2-1b.npz results/part26/target-olmo2-1b.json \
  results/part54a/source-pile-maxima-7c19d0120476.json
```

The Qwen maxima cache contains the target and Part 54's same-layer controls selected with seed 0. If the cache is missing or the seed/control units change, regenerate the Part 26 Pile maxima on the rented GPU using the matching corpus record before starting 59A/59B.

Before starting setup, `run59.sh` checks every required input file. A failed preflight stops before pip or other setup work. It requires `transformers==5.20.0` and `flash-linear-attention==0.5.2`; it records the installed versions of Torch, Accelerate, and Hugging Face Hub in `versions-pip.txt`. If Qwen prefetch fails, the OLMo prefetch is skipped; if OLMo prefetch fails after Qwen succeeds, Qwen phases still run. Each model's phases are recorded separately, and each analysis runs only after its matching 59B scan exits 0.

## Rent and start the unattended run

Use the same Vast image and helper as Part 54. The helper reads the key from `~/.vast-credits.env` and prints only instance metadata:

```bash
cd /home/elisepc/assistant-basin
python3 cloud/vast-part54-2026-10-22/vast.py offers
python3 cloud/vast-part54-2026-10-22/vast.py rent OFFER_ID
python3 cloud/vast-part54-2026-10-22/vast.py show INSTANCE_ID
```

Copy the package to the instance using the SSH host and port returned by `show`, then launch the foreground runner:

```bash
scp -P SSH_PORT /tmp/part59-inputs.tgz root@SSH_HOST:/root/
ssh -p SSH_PORT root@SSH_HOST
mkdir -p /root/part59
tar -xzf /root/part59-inputs.tgz -C /root/part59
nohup bash /root/part59/part59/run59.sh > /root/part59/run59.out 2>&1 < /dev/null &
echo $! > /root/part59/run59.pid
```

`run59.sh` installs the pinned package versions, records GPU/software versions, downloads the two pinned snapshots before setting offline mode, runs both 59A ladders and both 59B scans, and writes one log per phase under `/root/part59/logs/`. Poll the outer log with `tail -n 60 /root/part59/run59.out`; repeat until it prints `RETURN_READY` or `RUN_FAILED`. It creates `/root/part59-return.tgz` and prints `RETURN_READY` only if setup, prefetch, every phase, and archiving succeeded. On any failure it still archives the available results as `/root/part59-return-INCOMPLETE.tgz` and prints `RUN_FAILED` with the failed phase names. A 59A result is one JSON per prompt and dose: 60 files per model, with eight samples in each expected condition. The 59A and 59B locks use OS advisory locks, so process death releases them and an interrupted run can resume.

The control policy draws 20 same-layer units with the Part 26 seed and uses the three lowest-indexed units from that draw. This matches Part 26's policy, but it concentrates the controls at low indices and limits how broadly they represent the layer.

## Return and score

Copy the complete `/root/part59-return.tgz` back to the workstation and extract it into the assistant-basin checkout. `blind_setting.py` runs on the local machine, never on the rented GPU. Its API key is `ANTHROPIC_CREDITS_KEY`, loaded from `~/.anthropic-credits.env`; source that environment file in the local shell without printing its contents. The blind reader requires one complete manifest per model, exactly 60 listed prompt-dose files per model and eight answers per condition, and refuses missing, malformed, or stray clamp files. It reads only manifest-listed paths, so stale files cannot enter a batch.

```bash
cd /home/elisepc/assistant-basin
tar -xzf /tmp/part59-return.tgz
python3 part59/blind_setting.py submit --folder results/part59a --review-dir part59/review
python3 part59/blind_setting.py status BATCH_ID --review-dir part59/review
python3 part59/blind_setting.py collect BATCH_ID --by-cond --review-dir part59/review
```

The reader shuffles answers with seed 5901, keeps condition metadata in a separate local key, and refuses to print the scored table if more than 2 of its 34 calibration rows are wrong. Eight paragraph-length calibration texts span the four labels, with in-voice vague OTHER openings that give no genre cue. Its setting labels are SCIFI, EARTH-REAL, FANTASY, OTHER, with OK/BROKEN as a second label. `ANTHROPIC_CA_BUNDLE` can set the CA bundle path; without it the reader uses `~/.mitmproxy/ca-bundle-combined.crt`.

The GPU script also runs `analyze59b.py` for each activation archive. It reports 59f and 59g counts using the pooled per-dose P99 rule, writes those counts and responder indices before attempting a model reload, and then adds the top 10 SYMMETRIC units' Part 26 logit-lens top tokens and their top three corpus activation windows for 59h. Prediction 59h still requires hand reading those top projections and windows; the script deliberately does not infer a semantic label from token strings.

After either `RETURN_READY` or `RUN_FAILED`, copy the matching bundle back to the workstation first, then destroy the rented instance:

```bash
scp -P SSH_PORT root@SSH_HOST:/root/part59-return.tgz /tmp/part59-return.tgz
# On RUN_FAILED, copy /root/part59-return-INCOMPLETE.tgz instead.
cd /home/elisepc/assistant-basin
python3 cloud/vast-part54-2026-10-22/vast.py destroy INSTANCE_ID
```

This copy-then-destroy step applies after `RUN_FAILED` as well as after `RETURN_READY`.
