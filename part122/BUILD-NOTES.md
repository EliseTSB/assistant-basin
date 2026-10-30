# Part 122 build notes

## Decisions

- `rows119.load_units` supplies the 39 units and both kind labels. The passage writer prompt includes only each unit's persona name and description. It validates eight pairs per frame, two sentences per passage, first or third person, no persona-name word in an ordinary passage, and a word-count difference of at most two. A unit gets one initial call and up to three retries. The live client calls `label119._transport("ds", prompt, "")`, which selects the `ocgo` DeepSeek V4.1 Flash route with a 4,000-token cap.
- The writer seed is 12201. Random controls use seed 12202 plus a stable hash of each unit ID. They are sampled without replacement within each unit and exclude all selected focal channels from the same layer; random channels may recur across units. Shuffled controls use a seeded, non-self permutation (seed 12203). The complete channel and partner map is saved in `read_set.json`.
- T and F tokenize the passage as plain text and exclude tokenizer-added wrapper tokens. C renders the Part 119 `Who are you?` user turn followed by the assistant passage. Fast-tokenizer offsets mark only that passage; the fallback uses prefix token counts. Matched pairs share a forward batch when both rows are pending. The hook reads the `down_proj` input at each selected layer and stores channel maximum and mean over the passage mask.
- The model loader follows Part 119c: pinned revision `48d788eca847d4d7548f375ad03d3c9312f6139e`, bf16, local-only, direct device loading for XPU. CPU loading is used only for the tiny-model test. Gate 2 evaluates Part 95's 16 space and ordinary pairs at L2 N2466 and stores them in `gate2.json`.
- Gate 1 requires complete token-count pairs with a difference of at most two in at least 90%. Gate 2 requires 9 of 16 space maximum wins. Gate 3 requires the random-channel read rate to be at most 15% in every frame. A missing count or activation cannot pass. The scorer keeps tables visible after a miss under a heading that says nothing is read from them, and marks predictions `not read (gate N)`.
- Passage and activation JSON files use UTF-8. Activations append by `passage_id` and skip completed passages. Existing run metadata and the channel map must match before resuming. `--unit-limit` is a smoke option and cannot produce a complete score.
- The registered J-lens list was reproduced from `code/concept_slot_crosstab.py` using its true-neuron-arm Haiku score rule (`score >= 2`) and copied to `jlens-concept-matches.json` with the source paths.
- No live passages or model activations were produced in this worker: the brief prohibits network and paid calls, and this worker has no GPU. `--dry-run` uses a marked local stub and is not study data. The real passage command below must run on the authorized Linux host before the model job.

## Passage command

Run from the repository root on the Linux host. The command refuses to overwrite an existing file; pass `--force` only when replacing it intentionally.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part122/write_passages.py --out part122/passages122.json
```

## Runtime copy list and launch

The Windows job needs the runner, passage validator, shared row/model helpers, passage file, Part 95 positive-control text, and Part 92 unit/kind inputs. Run from the repository root:

```sh
tools/launch_arc.py 122 \
  --step "-I -B part122/read122.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part122 --device xpu --batch 8 --passages part122\\passages122.json --gate2-passages part95\\passages95.json --neurons results\\part92\\step2\\neurons.json --kinds-haiku results\\part92\\step2\\kinds.json --kinds-ds results\\part92b\\score92b.json" \
  --copy part122/read122.py part122/write_passages.py part119/rows119.py part92/common92.py part85/checkpoints.py part122/passages122.json part95/passages95.json results/part92/step2/neurons.json results/part92/step2/kinds.json results/part92b/score92b.json
```

## Scoring command

After copying the job outputs back, score on CPU from the repository root:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part122/score122.py \
  --activations results/part122/activations.jsonl \
  --gate2 results/part122/gate2.json \
  --read-set results/part122/read_set.json \
  --out results/part122
```

This writes `results/part122/score122.json` and `results/part122/SCORE.md`. The local test command is:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part122/test_part122.py
```

## Run note (2026-10-30)

The first launch hung at 656 of 1,872 rows: no write after 11:22:14, no CPU time and no GPU use by the process for over four minutes, after about 14 s per batch. The hook called `.item()` for every value, which is one device transfer each. It now moves the batch's statistics to the CPU once per hook. The numbers are the same values. The run was stopped and resumed from its activations file, and the 656 rows stand.
