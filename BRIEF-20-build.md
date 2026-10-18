# Brief: build part20.py (Weaver's identity clamp and context seam on N2742)

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-20` (branch `part20` of the public assistant-basin
repo, HEAD b5c24f1). Write only inside it. Do not commit, do not download models, do not edit PREDICTIONS.md.

Read first: PREDICTIONS.md, section "Part 20" (the exact definitions; build to them), `neuron_report.py` (reuse
`load_model`, `find_text_layers`, `NeuronHook`, `build_prompt`, `_dump`, `_file_sha`, `spill_guard`), and
`data/sense-battery.jsonl` with `data/README-sense-battery.md` (groups a and d, their target patterns).

## What to build

`part20.py` with two subcommands, writing JSON into `--out`, each with a header like the other tools (model,
revision, tool sha256, input file shas, layer, neuron, device, dtype, written time). Defaults: `--model
Qwen/Qwen3.5-4B`, the same revision default as neuron_report.py, `--layer 22 --neuron 2742 --device xpu`.

1. `identity` (20.1, 20.2). For each prompt in Part 18's set (read neuron_report.py's steer prompts; at least "Who made
   you?" and "What is your name?"), in the chat format `build_prompt` gives:
   - Unhooked pass: full logits at every position, and N2742's activation at every position (record with NeuronHook).
   - Hooked pass, one position: the hook writes back the recorded value at the last prompt token only; elsewhere it
     leaves the input untouched.
   - Hooked pass, every position: the hook writes back the recorded value at every position.
   - Record for each hooked pass the largest absolute logit difference over the whole vocabulary at every position,
     and the overall max. Store per-position maxima, not full logits.
   - Greedy generation (60 new tokens, eos as in neuron_report.py) unhooked and with the every-position identity hook.
     During generation the hook must write back each step's own recorded value: record the unhooked generation's
     activation at each new token, then replay the same token sequence with the hook. Store both token-id lists and
     whether they are identical.
   NeuronHook clamps one value per unit; extend it (in part20.py, not by changing its behaviour for existing callers)
   so values can be a per-position tensor.

2. `seam` (20.3-20.5). Filler: from `--corpus` (a Pile jsonl, one `{"text": ...}` per line), take documents in order,
   skip any containing the battery's group a, d or e target patterns (case-insensitive), and concatenate until about
   4,000 tokens; record the doc indices used and the exact token count. The same filler for every row.
   - (a) local: each group a sentence alone; read N2742's max over the token span of the word "alien".
   - (b) distant: filler, then the same sentence; the same span read.
   - (c) summarized: filler, then the summary sentence given in PREDICTIONS.md; read the max over the summary's tokens.
   - Group d local: each group d sentence alone, max over its target span (drop the Chinese line or read the max over
     its span, and record which).
   - Filler alone: the max over all filler tokens.
   - Store every per-row value, the means, and the ratios the predictions name. Do not score the predictions.

Use plain text inputs (no chat template) for `seam`, matching how the battery was read in Part 18b; say so in the
header. Run under `torch.no_grad()` and with weights not requiring grad.

## Tests

`test_part20.py` on CPU with a tiny random model built from the real Qwen3.5 config (follow `test_circuit.py`'s
`tiny()` pattern; snapshot path from the `QWEN35_SNAPSHOT` environment variable). Rows that can fail:
- the identity hook at one and every position gives a max logit difference of exactly 0 on the tiny model in float32;
- a sabotaged identity hook (writes the value plus 1e-3) gives a non-zero difference: the test must detect it;
- the generation replay gives identical token ids, and a sabotaged replay does not;
- filler selection skips a document containing "alien" and one containing "nebula";
- the "alien" span and the summary span are located correctly with the real tokenizer;
- `seam` and `identity` write their JSON with the header fields.

Run with `/home/elisepc/neuron-probe/.venv/bin/python -B test_part20.py` and
`QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Also run `test_neuron_report.py` to show nothing shared broke.

## Report

`REPORT-20-build.md` in the worktree root: SUMMARY (suite counts and exit codes, quoted), FINDINGS (what was built,
file:line, which rows failed against the sabotages), BLOCKERS (or "none"). Quote output you saw.
