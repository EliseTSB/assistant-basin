# Part 133: OLMo 2 stage comparison

This capture compares the Part 85 parent, SFT, DPO, and Instruct checkpoints at the input to each layer's
`mlp.down_proj`. Run preparation in WSL from the repository root. It is local only and deliberately refuses to
replace missing UltraChat data with another corpus.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part133/prepare133.py --out results/part133/prepared
```

The prepared manifest pins the local snapshots: parent `f83cd641d4f00954eb25c781e0ac360debb3239e`, SFT
`0d85a3d037876ce6ac7d4311d994400fc66ac27f`, DPO `c4b0485961ab24c2433b090f3b922f0913a9290f`, and Instruct
`48d788eca847d4d7548f375ad03d3c9312f6139e`. The dataset must contain the registered 4,000 WikiText documents and
1,000 UltraChat test_sft dialogues. `prepare133.py` reads UltraChat from a local parquet, JSONL, or JSON file; pass its
path with `--ultrachat` if it is not at the default corpus path. It makes no network requests.
The current machine has the WikiText parquet but no local UltraChat test_sft file, so preparation stops until that
registered dataset is provided; no substitute corpus is used.

Run each A770 job from the repository root, one at a time. `launch_arc.py` runs the repo prelaunch check, copies the
listed files to the Windows job folder, and starts the registered job runner. Device `xpu` selects the Intel Arc.
The Windows model cache paths below correspond to the local WSL cache at `/mnt/e/neuron-probe/hf/hub/`.

## Pilot

The 200-document pilot keeps the 4:1 web/chat mix. Time both batch sizes and score each pilot before choosing the
main-run batch size. The collector also records the self-comparison gate and fixed derangement null.

```sh
tools/launch_arc.py 133 \
  --step "-I -B part133/collect133.py --docs results\\part133\\prepared\\manifest.json --parent E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B\\snapshots\\f83cd641d4f00954eb25c781e0ac360debb3239e --sft E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-SFT\\snapshots\\0d85a3d037876ce6ac7d4311d994400fc66ac27f --dpo E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-DPO\\snapshots\\c4b0485961ab24c2433b090f3b922f0913a9290f --instruct E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part133\\pilot-b16 --stage pilot --device xpu --batch 16 --limit 200" \
  --copy part133/collect133.py results/part133/prepared/manifest.json results/part133/prepared/documents.jsonl

tools/launch_arc.py 133 \
  --step "-I -B part133/collect133.py --docs results\\part133\\prepared\\manifest.json --parent E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B\\snapshots\\f83cd641d4f00954eb25c781e0ac360debb3239e --sft E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-SFT\\snapshots\\0d85a3d037876ce6ac7d4311d994400fc66ac27f --dpo E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-DPO\\snapshots\\c4b0485961ab24c2433b090f3b922f0913a9290f --instruct E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part133\\pilot-b32 --stage pilot --device xpu --batch 32 --limit 200" \
  --copy part133/collect133.py results/part133/prepared/manifest.json results/part133/prepared/documents.jsonl
```

After the Windows job finishes, copy each `pilot-sums.pt` from the Windows job output back to the matching WSL
`results/part133/pilot-b16/` or `pilot-b32/` directory, then score on CPU:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part133/score133.py --sums results/part133/pilot-b16/pilot-sums.pt --part92 results/part92 --out results/part133/pilot-b16/score.json
/home/elisepc/neuron-probe/.venv/bin/python -I -B part133/score133.py --sums results/part133/pilot-b32/pilot-sums.pt --part92 results/part92 --out results/part133/pilot-b32/score.json
```

Do not begin the main run if either gate fails. Choose the quicker batch whose GPU run completes without memory
pressure. The design's projection is approximately 40 minutes for 5,000 documents; if pilot timing projects over
90 minutes, amend the registered wikitext share before the main stage.

## Main stage

Replace `SELECTED_BATCH` with the pilot batch selected above. The job resumes atomically from its last 500-document
checkpoint. It processes all 5,000 documents.

```sh
tools/launch_arc.py 133 \
  --step "-I -B part133/collect133.py --docs results\\part133\\prepared\\manifest.json --parent E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B\\snapshots\\f83cd641d4f00954eb25c781e0ac360debb3239e --sft E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-SFT\\snapshots\\0d85a3d037876ce6ac7d4311d994400fc66ac27f --dpo E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-DPO\\snapshots\\c4b0485961ab24c2433b090f3b922f0913a9290f --instruct E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part133\\main --stage main --device xpu --batch SELECTED_BATCH" \
  --copy part133/collect133.py results/part133/prepared/manifest.json results/part133/prepared/documents.jsonl
```

Copy `main-sums.pt` back to WSL and score it with the saved Part 92 IDs and classes:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part133/score133.py --sums results/part133/main/main-sums.pt --part92 results/part92 --out results/part133/main/score.json
```

The scorer reports both gates, all four registered predictions, and each group/pair/layer's exclusion count under
the 100-position floor. Gates must pass before predictions are interpreted. Chat gap results can reflect the chat
format being unfamiliar to the parent checkpoint.
