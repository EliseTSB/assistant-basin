# Brief: build part26.py (twins of Qwen3.5-4B's alien and identity neurons in OLMo 2 1B and SmolLM3-3B)

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-26` (branch `part26` of the public assistant-basin
repo). Write only inside it. Do not commit, do not download models, do not edit PREDICTIONS.md, and do not load any real
model's weights (the real run happens later on an Intel Arc GPU under Windows; a full model on this machine's CPU
starves everything else). Tiny random models built from the real configs are fine for tests.

Read first: PREDICTIONS.md, section "Part 26" including Weaver's amendment and the prior-art note (build to its
definitions exactly; where one is ambiguous, say so in your report rather than choosing silently). Reuse, do not copy:
`neuron_report.py` (`load_model`, `find_text_layers`, `cmd_writes`'s logit lens, `_dump`, `_file_sha`, `spill_guard`,
`control_units`), `part20.py` (hooks, `select_filler`, header helpers) and `part20f.py` (how the battery in
`data/part20f-battery.json` is scored after the filler documents), and `part24.py` (sampling: temperature 0.7, top_p
0.95, one seed per condition, `max_new_tokens` as there). `--device xpu`, `--max-gpu-gb 14.5` spill guard after every
batch (the GPU has 16 GB and silently spills into system memory rather than failing). Every hook is removed in a
`finally`.

Models (offline, from the Windows cache `E:\neuron-probe\hf`; for tests read configs and tokenizers from
`/mnt/e/neuron-probe/hf/hub/...`):
- source `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- `allenai/OLMo-2-0425-1B-Instruct` revision `48d788eca847d4d7548f375ad03d3c9312f6139e`;
- `HuggingFaceTB/SmolLM3-3B` revision `a07cc9a04f16550a088caea529712d1d335b0ac1`.

A "neuron" everywhere is one input unit of a layer's MLP `down_proj` (as in the other tools). Source neurons: Qwen
layer 22 unit 2742 (N2742) and the 50 neurons of set `top` in `results/part19/group.json` (identity-50): 51 in all.

## Definitions to implement

- **Documents.** From the corpus (`--corpus`, a jsonl with a `text` field; the run uses `qwen35\pile10k.jsonl`), the
  first 1,000 documents with at least 256 whitespace-separated words, in file order; each document is its first 256
  words joined by single spaces. Record the chosen line indices. Every model reads exactly these strings, so word k of
  document d is the same word for all three.
- **Word activation.** Tokenize with offsets (fast tokenizer, the tokenizer's own default for special tokens). A word's
  activation for a neuron is the largest activation over the tokens whose character span overlaps the word's span
  (whitespace excluded). Tokens with an empty span (special tokens) belong to no word. Refuse to run if any word has
  no token.
- **Writes.** Each neuron's logit-lens top 30 tokens as in `cmd_writes` (unembedding @ final-norm gain * down_proj
  column, z-scored), decoded, with leading whitespace and the byte-level space marker removed and lowercased. A target
  neuron is a writes-candidate for a source neuron when the two top-30 string sets share at least 3 strings. Compute this
  for every target neuron, in chunks (SmolLM3's vocabulary times one layer's units does not fit at once).
- **Fires.** Pearson correlation over all 256,000 word positions between each source neuron's word activations and
  every target neuron's, accumulated from sums (n, sum x, sum x^2, sum y, sum y^2, sum xy). Accumulate per batch of
  documents in float32 on the GPU and add into float64 on the CPU (the Arc has no native float64). **Null:** the same
  correlation with target document d paired with source document d+1 (the last with the first), word for word: whole
  documents misaligned, so topic clustering within a document cannot pass it. For each source neuron, the null bar is the
  largest null correlation over all target neurons. A target neuron **beats the null** when its aligned correlation is
  above that bar.
- **Twin candidates of N2742** in a target: the 5 target neurons with the highest aligned correlation that beat the null,
  plus any writes-candidate among the 50 highest. If none beats the null, take the 5 highest anyway and mark each
  `beats_null: false`.
- **Battery (26.5).** For each twin candidate, the Part 20f battery in the target model, scored as `part20f.py` scores it
  (same filler documents by index, same phrases, max over the phrase's tokens), giving the mean max for each set
  (D, U, C1, C2, C3, C3-bare, C4, E, E-map). Also report the candidate's largest activation over the 1,000 documents.
- **Clamp (26.2).** The best twin (highest aligned correlation among those beating the null, else highest overall,
  flagged) clamped at 2x its largest activation over the 1,000 documents, at every position including generated ones;
  8 sampled answers to "Who made you?" with the model's chat template and no system message; the same for 20 random
  neurons in the same layer (fixed seeds), each clamped at 2x its own largest activation; and 8 unclamped answers.

## What to build

`part26.py` with subcommands, each writing one JSON file (and a compressed numpy file beside it where large) into
`--out` with the usual header (model, revision, tool sha, input shas, timestamp):

1. `source` -> `source.json` + `source.npz`: the chosen documents, the 51 source neurons' word activations
   (51 x 1000 x 256), and their writes top 30.
2. `target --model M --revision R` -> `target-<short>.json`: for each source neuron, the top 20 target neurons by aligned
   correlation (layer, unit, r, null bar, beats_null, writes overlap), its writes-candidates with their r, and the count
   of target neurons beating the null. Also the twin candidates of N2742, and every target neuron's largest activation
   over the documents (npz).
3. `battery --model M ...` -> `battery-<short>.json` for N2742's twin candidates.
4. `clamp --model M ...` -> `clamp-<short>.json`.

`--steps source,target,battery,clamp` with `--targets olmo2-1b,smollm3-3b` runs them in order, loading each model once.
Print one progress line per 100 documents so a long run can be followed.

## Tests

`test_part26.py` on CPU with tiny random models built from each real config (follow `test_part25.py`'s `tiny()`
pattern; real tokenizers from the snapshots). Rows that can fail, each shown failing against a deliberate break before
you keep it:
- word activation equals the max over a word's tokens on a hand-checked string, for all three tokenizers, and special
  tokens join no word;
- the accumulated correlation equals `numpy.corrcoef` on a small random case to 1e-5, aligned and misaligned;
- the misaligned pairing pairs target document d with source document d+1, wrapping at the end;
- the string normalization makes " Alien", "Alien" and "Ġalien" one string, and the overlap count is right on a
  hand-made pair;
- the clamp holds the neuron at the set value at every position during generation, and no other unit moves;
- the battery scorer gives the same numbers as `part20f.py` on the tiny Qwen model.
Run with `/home/elisepc/neuron-probe/.venv/bin/python -B test_part26.py` and
`QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Also run `test_neuron_report.py`, `test_part20.py` and `test_part25.py` to show nothing shared broke.

## Report

`REPORT-26-build.md` in the worktree root: SUMMARY (suite counts and exit codes, quoted), FINDINGS (what was built,
each design choice the brief left open, the measured peak memory of one SmolLM3 document batch on the tiny model and
your estimate for the real one, each break you used to show a row can fail and what it printed), and BLOCKERS. Quote
test output you saw; never describe a result you did not run.
