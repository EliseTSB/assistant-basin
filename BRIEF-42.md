# Build brief: part42.py (Luna)

Run size, multiplied out: weights only; about 50 target units + 40 random units, 2 readouts each; no forward pass.
Read PART42-DESIGN.md and the Part 42 rows of PREDICTIONS.md (do not edit them). This runs on WSL CPU, not the GPU.

part42.py, subcommands `read` and `score`:
- Inputs: --model {Qwen/Qwen3-1.7B, Qwen/Qwen3.5-4B}; --snapshot (default: the HF cache snapshot dir under
  /mnt/e/neuron-probe/hf/hub/models--Qwen--<name>/snapshots/<only dir>); --lens (default
  /home/elisepc/neuron-probe/jlens/<qwen3-1.7b|qwen3.5-4b>/jlens/Salesforce-wikitext/<Qwen3-1.7B|Qwen3.5-4B>_jacobian_lens.pt);
  --units: for 1.7B, the rank-1 unit per concept from --screen (default
  /mnt/c/Users/alexa/neuron-probe-gpu/results/part35/qwen-qwen3-1-7b/screen-floor-median.jsonl; reuse part35/part40's
  rank-1 picker logic: read part40.py in /home/elisepc/green/worktrees/ab-part40-41-build for how it picks) plus the
  wizard unit L19 N4826; for 4B, the fixed list (6,753) (7,1169) (11,8868) (29,776) (22,2742). Plus 20 seeded random
  units (seed 4201) drawn from the same set of layers as the targets.
- Load ONLY the tensors needed with safetensors.safe_open (never the whole model): each needed layer's
  mlp.down_proj.weight, the final norm weight, and the LM head (or the embedding if tie_word_embeddings is true; read
  config.json). Qwen3.5-4B is a multimodal checkpoint: find the language-model key prefix from
  model.safetensors.index.json rather than guessing. Load the lens with torch.load(path, weights_only=True); its
  dict keys are documented in PART42-DESIGN.md (J: {layer: Tensor[d_model, d_model]}); print and check the
  `provenance` field and that J's layers cover every target layer; assert d_model matches config.
- Readouts exactly as the design: logit lens lm_head(g*w) and J lens lm_head(g*(J[l] @ w)) where g is the final norm
  gain (check whether this Qwen RMSNorm stores weight or (1+weight) — read the model's modeling code in the installed
  transformers for this architecture and apply the same; state which in the report). Top and bottom 30 token ids +
  decoded strings (tokenizer from the snapshot, offline).
- score: the 42.1-42.5 inputs as worded, with the fixed word lists in the design; write score-42.json. Also per unit:
  overlap of the two top-30 sets.
- Memory: cap work at about 4 GB RSS: process one layer at a time; the LM head in float32 is about 1.5 GB for 4B.
- Tests (pytest, tiny synthetic tensors written to a temp safetensors file and a fake lens dict): the readout math on a
  hand example; the J lens with J = identity equals the logit lens; the concept matcher on hand examples; the key
  prefix finder on a fake index with a language_model prefix. Add a test pinning the 4B unit list.
- Output: results/part42/<model-slug>/read.json and score-42.json. Report REPORT-42-build.md ending SUMMARY / FINDINGS /
  BLOCKERS with exact WSL command lines (use ~/neuron-probe/.venv/bin/python). You may run the real read on WSL CPU if
  the lens files exist (check /home/elisepc/neuron-probe/jlens/SHA256SUMS exists first; if not, stop at tests).
  Commit nothing.
