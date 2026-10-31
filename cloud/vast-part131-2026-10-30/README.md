# Part 131 Vast run

`vast.py` rents one on-demand RTX 3090 only when explicitly run. It filters for no more than $0.20/hour, reliability 0.98+, and 130 GB disk, starts the runner with the direct host address and `~/.ssh/vast_ed25519` via `ssh -i`, and skips recorded bad offers/hosts. The pilot stage leaves its instance alive for review; the main stage requests destruction in `finally`. Both print whether Vast still lists an instance. Set `VAST_KEY` in the process environment; never source the bare key file.

Run from the repository root after `part131/units131.json` exists:

```bash
python3 part131/select131.py
python3 cloud/vast-part131-2026-10-30/vast.py --stage pilot
```

The upload script installs pinned Transformers, Accelerate, Hub, and Safetensors versions. It downloads the registered base and Tulu final revisions one at a time, runs both pilots, deletes each model directory, and returns `results/part131/pilot/`. The script stops after the pilot so timing can be reviewed. It prints `PART131_REUSE_INSTANCE` and `PART131_STARTED_AT`; keep those values for the main-stage command.

On the same still-running instance and repository checkout, start the main stage after reviewing pilot timing. Substitute the two values printed by the pilot command:

```bash
python3 cloud/vast-part131-2026-10-30/vast.py --stage main --instance-id INSTANCE_ID --started-at UNIX_EPOCH
```

`run-main.sh` downloads each registered model revision just before its run, deletes it after that run, and writes JSONL plus metadata under `results/part131/main/`. The second `vast.py` call enforces the 120-minute total cap from the pilot timestamp and destroys the instance at completion. Confirm its `PART131_INSTANCE_REMAINS` line. For labels and scores, from the repository root:

```bash
python3 part131/label131.py --answers results/part131/main --out-dir results/part131/labels
python3 part131/score131.py --answers results/part131/main --labels results/part131/labels
```

The runner has no system prompt, uses each tokenizer's chat template for chat models and the registered plain frame for Base, records generated token IDs, and checks exact no-op-hook parity. `run-main.sh` never removes pilot results.
