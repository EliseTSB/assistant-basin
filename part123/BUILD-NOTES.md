# Part 123 build notes

## Decisions

- Stage A samples four suffixes, fits 39 mean-contrast vectors, and writes the overlap table and grouped shuffle null. It makes no persona answer generations.
- Stage B runs p92-00 through p92-03 across all six questions. It covers three layers. Stage C resumes the same answer file and fills the full 39-unit run.
- The run writes positive neuron doses 0.5, 0.85, and 1; positive vector doses 0.5, 0.85, 1, 1.5, and 2; and positive random doses 0.85, 1, and 2. Three random directions are made once per layer and saved for reuse.
- `run123.py` calls the residual hook, neuron push, and random direction helpers in `part92/common92.py`. It stores generated token IDs and compares zero-hook rows against the shared plain row.
- Every fit hook moves only the last-token state to CPU once per forward call. The calibration hook moves its reduced token norms to CPU once per layer call. No `.double()` call is used.
- Part 118 ran OLMo 2 1B with Part 92's standard loader. Part 115b's direct-load and allocator warm-up workaround addresses a 7.8 GiB Qwen tensor. The OLMo path does not use that workaround.
- Reader items expose only an opaque item ID, question index, and answer hash. Private row and unit joins go to `.keys.jsonl`. Labels are bound to `sha256(question + newline + answer)[:16]`. A changed answer makes an old label pending. The scorer treats a hash mismatch as missing.
- DeepSeek and Luna use Part 92b's shared route through `label123.py`. The DeepSeek reader cap is 4,000 tokens. Luna includes every DeepSeek-positive item and uses seed 12303 for the rest.
- The test suite runs each script main on a tiny CPU model and fake readers. It also builds full synthetic raw files to exercise all six score gates.

## Runtime copy lists

Run from the repo root after the design is registered and the prelaunch check passes. The model path below is the pinned local OLMo snapshot on the A770 host.

Stage A:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 123 \
  --step "-I -B part123/fit123.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out p123\\fit --device xpu --batch-size 64 --shuffles 1000" \
  --copy part123/fit123.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json
```

Copy these files back from the A770 job folder into `results/part123/fit/`: `suffixes.json`, `vectors.pt`, `overlap.jsonl`, `shuffled-null.jsonl`, and `fit-meta.json`.

Stage B pilot:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 123 \
  --step "-I -B part123/run123.py --stage pilot --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --fit results\\part123\\fit --out p123\\run --device xpu --batch-size 64 --max-new-tokens 120" \
  --copy part123/run123.py part123/fit123.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part123/fit/vectors.pt
```

Copy `p123/run/answers.jsonl`, `p123/run/zero-hook-check.json`, `p123/run/medians.json`, and `p123/run/random-directions.pt` back into `results/part123/run/`. Inspect the pilot answers and zero-hook results before Stage C.

Stage C main run:

```sh
/home/elisepc/neuron-probe/.venv/bin/python tools/launch_arc.py 123 \
  --step "-I -B part123/run123.py --stage main --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --fit results\\part123\\fit --medians results\\part123\\run\\medians.json --out results\\part123\\run --device xpu --batch-size 64 --max-new-tokens 120" \
  --copy part123/run123.py part123/fit123.py part92/common92.py part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json results/part123/fit/vectors.pt results/part123/run/answers.jsonl results/part123/run/zero-hook-check.json results/part123/run/medians.json results/part123/run/random-directions.pt
```

Copy the updated `answers.jsonl`, `zero-hook-check.json`, `medians.json`, and `random-directions.pt` back to `results/part123/run/` after the job finishes.

## Reader commands

After the run files are back in the repo, run both known-example checks. Then label all DeepSeek items, followed by Luna's seeded agreement sample.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part123/label123.py --gate --reader ds --out results/part123/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part123/label123.py --gate --reader luna --out results/part123/known-luna.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part123/label123.py --reader ds --answers results/part123/run/answers.jsonl --neurons results/part92/step2/neurons.json --kinds results/part92/step2/kinds.json --out results/part123/labels-ds.jsonl --gate-labels results/part123/known-ds.jsonl
/home/elisepc/neuron-probe/.venv/bin/python -I -B part123/label123.py --reader luna --answers results/part123/run/answers.jsonl --neurons results/part92/step2/neurons.json --kinds results/part92/step2/kinds.json --labels-ds results/part123/labels-ds.jsonl --out results/part123/labels-luna.jsonl --gate-labels results/part123/known-luna.jsonl
```

Each answer label file gets a `.keys.jsonl` sidecar. Rerun a failed or stale item with the same output path. The labeler skips current valid labels and retries pending items twice.

## Scoring

The scorer reads raw answers, both label files and keys, zero-hook rows, known-example labels, overlap rows, and shuffle nulls. It writes `score123.json` and `SCORE.md` under `results/part123/`.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part123/score123.py \
  --answers results/part123/run/answers.jsonl \
  --neurons results/part92/step2/neurons.json --kinds results/part92/step2/kinds.json \
  --zero-hook results/part123/run/zero-hook-check.json \
  --labels-ds results/part123/labels-ds.jsonl --keys-ds results/part123/labels-ds.jsonl.keys.jsonl \
  --labels-luna results/part123/labels-luna.jsonl --keys-luna results/part123/labels-luna.jsonl.keys.jsonl \
  --known-ds results/part123/known-ds.jsonl --known-luna results/part123/known-luna.jsonl \
  --overlap results/part123/fit/overlap.jsonl --shuffled-null results/part123/fit/shuffled-null.jsonl \
  --out results/part123
```

## Verification

Run the full Part 123 suite with:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part123/test_part123.py
```

This worker has CPU-only PyTorch. The suite exercises tiny CPU stubs. It does not load the pinned OLMo snapshot or run an A770 pilot.

Latest run: **11 tests passed**. The suite also broke the zero-hook and known-example gates on isolated copies. The two tests that expected those broken copies to pass went red, with **2 failed**, as expected.
