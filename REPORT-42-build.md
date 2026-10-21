# Part 42 build report

## Implementation

`part42.py` now implements the `read` and `score` commands for both checkpoints. It reads only selected MLP down-projection matrices, final norm, and unembedding weights through safetensors, and loads the fitted lens with `torch.load(..., weights_only=True)`. The 4B language-model key prefix is derived from the safetensors index.

The fitting estimator records residual-block outputs: `ActivationRecorder` installs forward hooks on each block and stores its output tensor. An MLP down-projection write in block `l` therefore uses `J[l]`. The final block has no fitted J and uses the identity readout, recorded on every row as `j_identity: true` with `j_layer: null`. Final-block identity rows are excluded from predictions 42.1 and 42.2, and retained in the Part 39 set for 39.9.

Lens provenance in each `read.json` includes the verified `.pt` SHA256 and manifest, tensor `n_prompts`, `source_layers` and `d_model`, plus the complete adjacent `config.yaml` and its SHA256. Qwen3 uses the stored final RMSNorm weight directly; Qwen3.5 uses `(1 + stored weight)`. These match the installed Transformers 5.20.0 Qwen3 and Qwen3.5 modeling code. LM-head rows are upcast in bounded chunks for float32 readout math to stay under the memory target.

Scoring includes 42.1–42.5, per-unit top-30 overlap, and 39.9 using only the logit-lens top 30, excluding L6 N753. The 42.5 criterion checks each concept's frequency across the 20 random units for each readout.

## Verification

- `~/neuron-probe/.venv/bin/python -m pytest -q test_part42.py` — 9 passed.
- `~/neuron-probe/.venv/bin/python -m py_compile part42.py test_part42.py` — passed.
- `git diff --check` — passed.
- Both lens files matched `/home/elisepc/neuron-probe/jlens/SHA256SUMS`.
- Every unit has 30 top and 30 bottom entries for both readouts. The 4B read has 25 `j_identity` rows, all at layer 31; identity readouts match their logit readouts. All non-identity rows use `j_layer == layer`.

## Real reads and scores

Outputs were written to `results/part42/qwen3-1.7b/{read.json,score-42.json}` and `results/part42/qwen3.5-4b/{read.json,score-42.json}`. The 1.7B read contains 43 Part 35 rank-1 units, wizard, and 20 random units. The 4B read contains five fixed targets, 20 random units, and 211 Part 39 units.

- 42.1: not supported — 14 J-lens versus 9 logit-lens matches among 28 eligible early units; the threshold is 18.
- 42.2: supported — median top-30 overlap is 24 (16 eligible 1.7B units) and 15 (5 eligible 4B units).
- 42.3: not supported — wizard matches spell, magic, and wand; L6 N753 matches none of its fixed words.
- 42.4: supported — L7 N1169 and L11 N8868 match China terms, and L29 N776 matches Google.
- 42.5: not supported — `cat` appears in two random units' logit top-30 sets; no concept repeats in the J-lens sets.
- 39.9: supported — six Part 39 units other than L6 N753 have an AI/assistant word in logit top 30. This includes final-block unit L31 N6634.

The first unchunked 4B attempt peaked at 4,355,208 KiB RSS. With chunked float32 projection, the final 4B read completed in 4:51.84 with 2,593,868 KiB peak RSS; the final 1.7B read completed in 0:45.77 with 1,594,204 KiB peak RSS.

## WSL commands

~~~sh
~/neuron-probe/.venv/bin/python -m pytest -q test_part42.py
~/neuron-probe/.venv/bin/python -m py_compile part42.py test_part42.py
/usr/bin/time -v ~/neuron-probe/.venv/bin/python part42.py read --model Qwen/Qwen3-1.7B
/usr/bin/time -v ~/neuron-probe/.venv/bin/python part42.py read --model Qwen/Qwen3.5-4B
~/neuron-probe/.venv/bin/python part42.py score --model Qwen/Qwen3-1.7B
~/neuron-probe/.venv/bin/python part42.py score --model Qwen/Qwen3.5-4B
~~~

## Parent control

This worker's control mailbox contained no messages, so no new dispositions were required. The prior worker's reported sequence 1 `add-p39-units` decision remains applied: Part 39 units are read separately, excluded from 42.x summaries, and retained for 39.9.

## SUMMARY

The Part 42 reader, scorer, tests, real reads, and score outputs are complete. The final 4B pass stays under the approximately 4 GB RSS target.

## FINDINGS

The lens estimator records block outputs, so each non-final MLP write uses its same-layer J. Final-block writes use identity. Lens hashes and fit metadata were verified and saved with each read. Predictions 42.2, 42.4, and 39.9 are supported; 42.1, 42.3, and 42.5 are not supported.

## BLOCKERS

None.
