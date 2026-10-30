# Part 124 build notes

## Fixed inputs and checks

- `rows119.load_units` supplies the 39 Part 92 units and their concept/slot labels. `passages124.py` freezes the persona phrases, neutral phrases, templates and tokenizer feasibility results in `passages124.json`.
- The pinned local tokenizer found T-template-1 phrase lengths 1, 2, 3, 4, 5 and 6 in counts 5, 10, 9, 11, 1 and 3. Length 5 merges into length 4 because it is rare and tied in distance; the fixed tie rule chooses the shorter length. Construction checks cover 24 frame/template combinations, 936 persona/neutral pairs and 1,872 slot strings. They found no quote merges, equal persona/neutral token counts and identical outside token IDs. Slot counts happened to stay stable across contexts; only the pairwise persona/neutral count equality is required.
- Read and instrument random channels use deterministic pools of 1,000 channels per focal layer, excluding every focal channel in that layer. Three channels per unit are sampled from its layer pool and assigned that unit's phrase as pseudo-target. The read set records both seeds and every channel.
- `read124.py` reuses `read122.load_model`, `read122._pack`, `common92.get_layers` and the same `down_proj` pre-hook location. Each layer hook captures the slot maxima, slot means and post-quote maxima as tensors, then transfers the combined result to CPU once. The hook contains no per-value `.item()` calls. No `.double()` call is used.
- The do-nothing gate compares logits on identical inputs with no hook and with the read-only hooks. The reader appends rows by frame, template, unit and side, and resumes missing rows. A torn final activation JSONL line is logged and removed before append. A complete final record without its newline is repaired before the next append. A malformed line elsewhere raises. The scorer skips and logs only a torn final JSONL line. Missing rows raise errors.
- `score124.py` writes the random and instrument lines and every gate before focal reads in `SCORE.md` and stdout. Its tests cover construction, do-nothing, random-read and T instrument gate failures.
- The end-to-end tests use a tiny tokenizer/model stub. Their launch trees contain exactly the files in the copy list below. They call all three script mains and run the documented reader and scorer command forms in isolated `-I -B` subprocesses. They check that all 39 stub units read, that manifest templates drive tokenization, that the merged length-4+5 stratum is scored, and that sabotaging the read rule or odd/even instrument split makes a test fail. They also check the 119-forward batch-8 plan, diagnostics and data-integrity errors.

## Tokenizer feasibility command

Run from the repository root on the authorized Linux host. The command is local-only and writes the frozen manifest in Part 124.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part124/passages124.py \
  --tokenizer /mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e \
  --out part124/passages124.json
```

## Runtime copy list and launch

Run from the repository root on the authorized A770 host. The runner captures 1,000 random channels per used layer in the same forward passes as the phrase rows. The batch flag counts matched persona/neutral pairs, so `--batch 8` makes each full batch 16 sequences.

```sh
tools/launch_arc.py 124 \
  --step "-I -B part124/read124.py --model E:\\neuron-probe\\hf\\hub\\models--allenai--OLMo-2-0425-1B-Instruct\\snapshots\\48d788eca847d4d7548f375ad03d3c9312f6139e --out results\\part124 --device xpu --batch 8 --passages part124\\passages124.json --random-count 1000" \
  --copy part124/read124.py part124/passages124.py part124/score124.py part124/passages124.json \
        part122/read122.py part122/write_passages.py part119/rows119.py part92/common92.py \
        part85/checkpoints.py results/part92/step2/neurons.json results/part92/step2/kinds.json \
        results/part92b/score92b.json
```

The runner writes `results/part124/activations.jsonl`, `read_set.json` and `run_meta.json`. It does not write model outputs or generated answers.

## Scoring command

After copying the run outputs back, score on CPU from the repository root:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part124/score124.py \
  --passages part124/passages124.json \
  --activations results/part124/activations.jsonl \
  --read-set results/part124/read_set.json \
  --run-meta results/part124/run_meta.json \
  --out results/part124
```

This writes `results/part124/score124.json` and `results/part124/SCORE.md`. The primary inference uses 10,000 length-stratified permutations with seed 12401.

## Test command

Run the full local suite from the repository root:

```sh
/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part124/test_part124.py
```

The full local suite passed: 10 tests in 8.32 seconds. Five new regression tests failed against copies of the unmodified scripts as expected. Separate copies with the read rule forced off and the instrument split reversed each made their regression test fail.
