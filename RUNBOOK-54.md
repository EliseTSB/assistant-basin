# Part 54: Vast.ai RTX 3090 runbook

This is an operator runbook for a manually rented RTX 3090 with 24 GB VRAM. No instance was rented for this build. Keep the run in two phases: phase 1 runs 54A and the four new model searches; phase 2 runs ladders only for the existing SmolLM3 comparator and new candidates that pass the Part 26.6 twin checks.

## Models, revisions, and dtypes

All five Hub repositories below are public and downloaded without an account. The four new model revisions are pinned to public Hub commits.

| Alias | Model and pinned revision | Load dtype | Approximate weights in VRAM |
|---|---|---:|---:|
| `qwen35-4b` | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | fp32 | about 16 GB; leave the remaining VRAM for kernels and activations |
| `olmo2-7b` | `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba1ae01581f270116362ee4aa1b97f4c84` | bf16 | about 14–15 GB; fp32 would exceed 24 GB |
| `qwen3-4b` | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` | fp32 | about 16 GB |
| `phi4-mini` | `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083` | fp32 | about 15–16 GB; the loader enables the model card's remote-code path |
| `granite3-2b` | `ibm-granite/granite-3.3-2b-instruct` @ `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | fp32 | about 10 GB |
| existing comparator `smollm3-3b` | `HuggingFaceTB/SmolLM3-3B` @ `a07cc9a04f16550a088caea529712d1d335b0ac1` | bf16 | about 6 GB |

The registered load dtype is set in `part26.py`'s model table. The CLI accepts an explicit `--dtype` override, so do not add one to these registered runs. OLMo-2-7B must stay bf16 to fit. Its bf16 run can change close comparisons; Part 48c observed a near-tie changing at reduced precision. Run one model at a time and leave other GPU processes stopped. If a model's table dtype does not fit, stop and report the OOM; the loader does not silently change dtype.

Hub pages: [OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Granite 3.3 2B Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct).

## Rent, upload, and prepare the instance

Rent one 24 GB RTX 3090 by hand in Vast.ai with the `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` image and direct SSH. A 40 GB disk keeps the hourly price near the cited $0.20 range if model snapshots are downloaded and removed one at a time. Keeping all checkpoints cached is easier with a 60 GB disk; the account notes from Oct 20 say that disk size can raise a 3090 offer to about $0.34/hour. Destroy the instance when the run and download-back are complete.

On the local machine, from the repository root, make a compact bundle of code and the existing Part 26 results. The Pile corpus used by Part 26 is outside this repository; copy the full `qwen35/pile10k.jsonl` separately. The 200-row `data/pile10k-head200.jsonl` is too short for the Part 26 battery filler.

```bash
tar -czf /tmp/part54-inputs.tgz *.py PREDICTIONS.md data/part20f-battery.json results/part26 results/part33-r2/natural-firing.json
scp -i ~/.ssh/vast_ed25519 -P <SSH_PORT> /tmp/part54-inputs.tgz root@<PUBLIC_IP>:/root/
ssh -i ~/.ssh/vast_ed25519 -p <SSH_PORT> root@<PUBLIC_IP> 'mkdir -p /root/part54/qwen35'
scp -i ~/.ssh/vast_ed25519 -P <SSH_PORT> /path/to/qwen35/pile10k.jsonl root@<PUBLIC_IP>:/root/part54/qwen35/pile10k.jsonl
```

Use the instance's direct public IP and mapped SSH port. The SSH proxy port has refused connections in previous runs. On the instance:

```bash
mkdir -p /root/part54 /root/hf
tar -xzf /root/part54-inputs.tgz -C /root/part54
cd /root/part54
python -m pip install transformers==5.20.0 flash-linear-attention==0.5.2 accelerate huggingface_hub
python -m pip uninstall -y causal-conv1d || true
export HF_HOME=/root/hf
nvidia-smi
python -c "import torch, transformers; print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__,torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -m pip freeze > versions-pip.txt
```

The expected base stack is torch 2.6.0+cu124 and Transformers 5.20.0, matching the Part 43/48 runs. Do not install the pip `causal-conv1d` wheel: the account notes record an ABI mismatch with this torch image. Flash Linear Attention can use its reference fallback if a kernel is unavailable.

Use this helper to fetch one pinned public model at a time without a token. The program itself sets Hugging Face offline mode before loading weights.

```bash
fetch_model() {
  HF_HUB_OFFLINE=0 python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir='/root/hf/hub'))
PY
}
discard_model() {
  repo_cache=${1//\//--}
  rm -rf "/root/hf/hub/models--$repo_cache"
}
```

## Phase 1: Qwen 54A and the four searches

First run the Qwen source ladder. Its maximum scale is recomputed in fp32 over the 30 selected alien passages in `results/part33-r2/natural-firing.json`; each random control gets its own maximum over those same passages. The scale cache is saved in `results/part54a/` and can be retained with the results.

```bash
fetch_model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
mkdir -p results/part54a
HF_HUB_OFFLINE=1 python -u part26.py ladder --model qwen35-4b --clamp-unit 22:2742 --alien-frames results/part33-r2/natural-firing.json --out results/part54a --device cuda --samples 8 --max-new 64 --control-count 3 --seed 0 > results/part54a/run.log 2>&1
discard_model Qwen/Qwen3.5-4B
```

The Part 26 Qwen source artifacts are reused for the searches. The source file is 51 neurons over the same 1,000 Pile documents, including N2742; it does not need to be regenerated. Copy these artifacts into the new output folder before running a target:

```bash
mkdir -p results/part54b
cp results/part26/source.json results/part26/source.npz results/part54b/
cp results/part26/target-smollm3-3b.json results/part26/target-smollm3-3b.npz results/part26/battery-smollm3-3b.json results/part26/clamp-smollm3-3b.json results/part54b/
```

Run the four target and profile stages separately so only one target checkpoint occupies the 40 GB disk at a time. The battery command reads the full `qwen35/pile10k.jsonl`; each new model's battery JSON includes its exact normalized logit-lens top 30 so the alien write check is inspectable.

```bash
run_search_model() {
  repo_id=$1
  revision=$2
  short_name=$3
  fetch_model "$repo_id" "$revision"
  HF_HUB_OFFLINE=1 python -u part26.py --steps target,battery --targets "$short_name" --corpus qwen35/pile10k.jsonl --out results/part54b --device cuda --batch-size 4 --write-chunk 256 > "results/part54b/$short_name.log" 2>&1
  discard_model "$repo_id"
}
run_search_model allenai/OLMo-2-1124-7B-Instruct 470b1fba1ae01581f270116362ee4aa1b97f4c84 olmo2-7b
run_search_model Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c qwen3-4b
run_search_model microsoft/Phi-4-mini-instruct cfbefacb99257ffa30c83adab238a50856ac3083 phi4-mini
run_search_model ibm-granite/granite-3.3-2b-instruct 707f574c62054322f6b5b04b6d075f0a8f05e0f0 granite3-2b
```

No generated answers are produced by the four search stages. To list new candidates that meet all three Part 26.6 checks, inspect each `battery-<alias>.json`: `beats_null` must be true (fire), `writes_alien_or_aliens` must be true (exact `alien` or `aliens` in the target unit's top 30), and `profile.profile_pass` must be true (all three existing battery checks). The search and profile calculations are unchanged; the new top-30 evidence makes the write decision explicit.

```bash
python - <<'PY' > results/part54b/twins.tsv
import glob, json
for path in sorted(glob.glob('results/part54b/battery-*.json')):
    data = json.load(open(path, encoding='utf-8'))
    if data['short_name'] not in {'olmo2-7b', 'qwen3-4b', 'phi4-mini', 'granite3-2b'}:
        continue
    for row in data['candidates']:
        if row.get('beats_null') and row.get('writes_alien_or_aliens') and row.get('profile', {}).get('profile_pass'):
            print(data['short_name'], f"{row['layer']}:{row['unit']}", data['model'], data['revision'], sep='\t')
PY
cat results/part54b/twins.tsv
```

## Phase 2: ladders for candidates

The brief separately directs a ladder for the existing SmolLM3 Part 26 comparator. The pre-existing `clamp-smollm3-3b.json` records its selected candidate as L23:1019; keep that candidate as the historical comparator and do not count it as a new 54B twin. Then run one ladder for every row in `twins.tsv`.

```bash
run_ladder_model() {
  short_name=$1
  unit=$2
  repo_id=$3
  revision=$4
  fetch_model "$repo_id" "$revision"
  HF_HUB_OFFLINE=1 python -u part26.py ladder --model "$short_name" --revision "$revision" --clamp-unit "$unit" --out results/part54b --device cuda --samples 8 --max-new 64 --control-count 3 --seed 0 > "results/part54b/ladder-$short_name-$unit.log" 2>&1
  discard_model "$repo_id"
}
run_ladder_model smollm3-3b 23:1019 HuggingFaceTB/SmolLM3-3B a07cc9a04f16550a088caea529712d1d335b0ac1
while IFS=$'\t' read -r short_name unit repo_id revision; do
  run_ladder_model "$short_name" "$unit" "$repo_id" "$revision"
done < results/part54b/twins.tsv
```

Each `ladder` writes a JSON manifest plus 21 uniquely named clamp JSON files, one for each model, twin, question, and dose. The questions are the three registered origin prompts; the doses are −3, −2, −1, +0.5, +1, +1.5, +2. Three random same-layer controls run only at −2 and +1.5. A matching incomplete ladder manifest resumes from existing output files; a completed ladder requires `--force` to rerun.

## Counts, time, and return files

With 8 samples, one complete ladder is 480 generated answers: 336 baseline and twin answers across all questions and doses, plus 144 control answers at the six question/dose control cells. The Part 26 seed policy gives each condition a deterministic seed that repeats across doses and questions; the unclamped and twin conditions keep distinct seeds and are not paired to each other. Baselines therefore repeat across the seven doses for each question. Phase 1 has 480 generations in 54A and zero generations in the four searches. Phase 2 adds 480 per ladder: one required Smol comparator, plus each new twin candidate. With the prediction minimum of two new twin candidates, the total would be 1,920; with one passing candidate in each new model, 2,880. Multiple candidates add 480 each. This is higher than the design's rough “about 1,200” estimate under the existing per-dose baseline policy.

Planning range on one 3090: allow 10–20 minutes for the 54A ladder and 10–20 minutes per phase 2 ladder, including source-frame scale preparation and model load. Allow roughly 2–5 hours for the four searches and battery profiles. These are planning estimates, not measurements on a 3090; the existing Part 26 Smol 3B target log took 2,189 seconds for the 1,000-document target stage, and that log does not record its device. Model snapshot downloads add network time. Record actual start/end times in the same manner as the Part 43 and 48b folders.

Before destroying the instance, copy back:

- `results/part54a/` including all clamp JSON, its ladder manifest, `source-alien-maxima-*.json`, and `run.log`.
- `results/part54b/` including source copy, target JSON/NPZ, battery JSON, `twins.tsv`, every ladder manifest and clamp JSON, and model logs.
- `versions-pip.txt`, the recorded torch/Transformers/CUDA/GPU version output, and start/end timestamps.
- The exact corpus file hash and model revision list used for the run.

From the local machine, fetch the results archive before destroying the host, then run the blind reader locally so no API key is copied to Vast:

```bash
# On the instance
tar -C /root/part54 -czf /root/part54-return.tgz results/part54a results/part54b versions-pip.txt

# On the local machine
scp -i ~/.ssh/vast_ed25519 -P <SSH_PORT> root@<PUBLIC_IP>:/root/part54-return.tgz .
tar -xzf part54-return.tgz -C /path/to/assistant-basin
cd /path/to/assistant-basin
python blind_origin.py submit --folder results/part54a
python blind_origin.py submit --folder results/part54b
```

Keep the two label batches' ID maps, keys, labels, and calibration outputs under `review/`. When finished, destroy the Vast instance and verify the account's active-instance list is empty; use the v1 instances endpoint if checking through the API.
