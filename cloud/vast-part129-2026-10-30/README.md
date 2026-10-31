# Part 129 on Vast

The helper selects an on-demand host with two RTX 3090 GPUs, reliability at least 0.98, at least 90 GB disk and a rate no higher than $0.30/hour. The single rental deadline is 150 minutes. `vast.py` reads the bare API key from `~/.vast-credits.env` with Python; it does not print or pass it to SSH. SSH uses the direct host address and `-i ~/.ssh/vast_ed25519`. Known bad offers and hosts are skipped. A boot error stops that offer immediately. The helper prints `P129_INSTANCE_REMAINS=yes/no` after each stage.

## Prepare the local catalog input

From the Part 67 repository root, prepare alignment checks and activity thresholds before the remote run. The GPU machine receives this small derived file and does not download the catalog.

```bash
/home/elisepc/neuron-probe/.venv/bin/python -I -B part129/prepare129.py /home/elisepc/neuron-probe/nl-catalog part129/prepared129.pt
python3 cloud/vast-part129-2026-10-30/vast.py --stage pilot
```

The pilot downloads the two model snapshots at the registered revisions, checks both devices, checks out Neuron Lens commit `07fdf45`, and builds the first-quarter corpus slices (10,000 dclm, 2,000 zh, 2,000 code and 2,000 chat rows). It times the paired collectors at batch 16, 32 and 64 on the first 100 lines of each slice. It also writes Gate 2 self-comparison tables at batch 16 and 64. The helper copies the pilot output and keeps the instance alive for the registered follow-on stage; it records the instance ID and original deadline in `results/part129/instance.json`.

Read the pilot timings before starting the main stage. Select a batch size no larger than 64, then run:

```bash
python3 cloud/vast-part129-2026-10-30/vast.py --stage main --batch 32
```

Choose `16`, `32` or `64` from the pilot timing table. This invocation reuses the pilot instance and deadline, runs `run-main.sh` with batch 32 by default, copies `results/part129/main`, requests destruction, and prints whether any instance remains. The remote `run-main.sh` receives the selected batch as its first argument. The main collector resumes from atomic checkpoints every 500 documents. Score locally after copy-back:

```bash
/home/elisepc/neuron-probe/.venv/bin/python -I -B part129/score129.py --sums results/part129/main/main-sums.pt --catalog /home/elisepc/neuron-probe/nl-catalog --library /home/elisepc/neuron-probe/persona-library/qwen --repo . --prepared part129/prepared129.pt --gate2 results/part129/gate2-b16/pilot-sums.pt results/part129/gate2-b64/pilot-sums.pt --out results/part129/score129.json
```

Do not read the registered predictions unless all three gates pass. A failed Gate 1, Gate 2 or Gate 3 is reported as a stop condition.
