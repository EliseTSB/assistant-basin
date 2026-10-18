# Brief: build `circuit.py` for Part 19 (assistant-basin)

You are a dark-factory worker. Your working directory is this worktree, `/home/elisepc/green/worktrees/ab-19` (branch `part19`). Write only inside it. Do not commit and do not download models.

## Read first
1. `PART19-DESIGN.md`: the specification. Follow it exactly; if a step cannot be built as written, stop and say so in the report instead of changing the measurement.
2. `neuron_report.py`: reuse `load_model`, `find_text_layers`, `build_prompt`, `NeuronHook`, `spill_guard`, `header`, `_dump`, `_file_sha`, the logit lens in `cmd_writes`, and the corpus loop in `cmd_reads` (its corpus is JSONL, one `{"text": ...}` per line; the GPU run will pass `qwen35\pile10k.jsonl` with the same document count Part 18 used, so mirror cmd_reads' arguments). Import from it; do not copy it.
3. `test_neuron_report.py` and `test_teacherforce.py`: how tests build a tiny random Qwen3.5 text model on CPU.

## Build `circuit.py`, subcommands screen, verify, group, edges, describe
- **Fixed answers.** A helper generates each prompt's greedy answer once (60 new tokens, unclamped) and caches it in the output folder; every other subcommand teacher-forces those exact token ids. Identity tokens are answer tokens whose decoded text matches the design's regex; M is their summed float32 log-probability (take log-softmax only at the positions that predict them). M' for control prompts is the summed log-probability of the first 20 answer tokens.
- **screen.** Embeddings require grad (`inputs_embeds`, never detach inside the model). A forward pre-hook on every layer's down_proj calls `retain_grad()` on its input and keeps a reference. One backward of M per identity prompt (and of M for "Who made you?" for the push ranking). Zeroing estimate per neuron = sum over positions of (-a) * grad; push estimate = sum over positions of grad. Write every neuron's pooled zeroing estimate (mean abs over identity prompts) and push estimate as compact arrays (`.npy` or float lists in JSON), plus the top 1,000 of each ranking with layer and index, and N2742's ranks (layer 22, index 2742).
- **verify.** Zero one neuron at every position (a hook per layer that sets the chosen units to 0; extend `NeuronHook` or add a small multi-layer helper). Top 50 by pooled zeroing estimate, plus 50 random neurons matched layer by layer (seed 0, never a top-50 member). Report true ΔM per identity prompt and ΔM' per control prompt, per-token means, pooled |ΔM|, and the ratio the design defines.
- **group.** Zero sets of 50 together (top 50; random sets with seeds 1, 2, 3 drawn as in verify) and greedy-generate the identity answers; report the answers and whether each first sentence names Qwen, Tongyi or Alibaba.
- **edges.** Nodes = top 30 by true pooled |ΔM| from verify's output. For each node, and for each of verify's 30 first random neurons as null sources, zero it and record every node's activation at the positions the design names; compute edges and the null 99th percentile per target node as specified.
- **describe.** For the nodes: logit-lens top 10 tokens (cmd_writes' method), and one corpus pass with hooks on every node's layer recording only node units, keeping each node's top 10 contexts (as cmd_reads does).
- Every output JSON carries `header(...)`-style fields: model, revision, tool sha256, prompts, seeds, and the input file shas it read. `--device` default `xpu`, `--spill-limit` as in neuron_report.py, batch where it is safe.

## Tests: `test_circuit.py`, CPU, tiny random Qwen3.5 text model, in the style of the existing tests
- the push estimate for a few neurons matches a finite-difference estimate (add a small epsilon to the neuron at every position, rerun, divide), in float32;
- **mutation by hand, say in the report that you did it:** detach the down_proj input inside the hook (make it a leaf); the finite-difference test must fail;
- the zeroing hook zeroes exactly the chosen units at every position and leaves other units and other layers unchanged;
- identity-token selection on a hand-made answer picks the right indices;
- verify's true ΔM equals M(zeroed) minus M(base) computed separately;
- zeroing a node in a later layer leaves every earlier-layer node's activation exactly unchanged;
- random layer-matched controls are deterministic for a seed, match the top set's layer counts and never overlap it;
- each subcommand runs end to end on the tiny model and writes JSON with the expected keys;
- `test_neuron_report.py` and `test_teacherforce.py` still pass.

Run tests with the Python that has torch: `QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a /home/elisepc/neuron-probe/.venv/bin/python test_circuit.py` (plain `python3` has no torch). If torch cannot import in your sandbox, say so in the report with the exact error, and still write the tests.

## Report
`REPORT-19-build.md` in this worktree's root (exactly that name and place). SUMMARY first (what you built, test counts as printed), then FINDINGS (every place you interpreted the design, and how), then BLOCKERS (or "none"). Quote output you saw; never describe a result you did not see.
