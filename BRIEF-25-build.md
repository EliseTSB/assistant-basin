# Brief: build part25.py (the fewest neurons that change who Qwen3.5-4B says made it)

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-25` (branch `part25` of the public assistant-basin
repo, HEAD 2bc6b8d). Write only inside it. Do not commit, do not download models, do not edit PREDICTIONS.md, and do
not load Qwen3.5-4B itself (the real run happens later on an Intel Arc GPU under Windows; a 4B model on this machine's
CPU starves everything else).

Read first: PREDICTIONS.md, section "Part 25" (build to its definitions exactly; where one is ambiguous, say so in your
report rather than choosing silently). Reuse, do not copy: `neuron_report.py` (`load_model`, `find_text_layers`,
`build_prompt` with its `system_prompt` argument, `_dump`, `_file_sha`, `spill_guard`) and `circuit.py` (how `screen`
retains gradients on every layer's `down_proj` input, the six identity questions `ID_PROMPTS`, the `IDENTITY` regex,
and how a first sentence is cut). Model `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
`--device xpu`, `--max-gpu-gb 14.5` spill guard after every batch (the GPU has 16 GB and silently spills into system
memory rather than failing). Every hook is removed in a `finally`.

## Definitions to implement

- **Sequences.** For each of the six questions and each of two stems, "I was created by" (maker) and "My name is"
  (name), build chat-template(question) + stem, in two variants: the stem as written (next token carries a leading
  space) and the stem followed by " **" (the model often bolds names; next token carries no leading space). 24
  sequences. The score is read at the last position.
- **Token sets.** Each target's maker words and name words; the first token id of each word in the form that follows
  its variant (with a leading space for the plain stem, without for the " **" stem). Targets: `qwen` (maker Alibaba,
  Tongyi, Qwen; name Qwen, Tongyi), `google` (maker Google; name Gemini), `openai` (maker OpenAI; name ChatGPT, GPT),
  `anthropic` (maker Anthropic; name Claude), `deepseek` (maker DeepSeek; name DeepSeek). Write every resolved first
  token (id and decoded text) into the output, and refuse to run if two targets share a first token id within one
  variant. Note in the report any first token that is a fragment (for example " Q" for Qwen).
- **Margin** of a target at a sequence: log(sum of softmax probabilities of the target's first tokens for that stem)
  minus log(sum for qwen's). A condition's margin is the mean over the 24 sequences.
- **Success (greedy, `max_new_tokens` 48, the chat template, no system message).** Cut the first sentence as circuit.py
  does. Removal target google: the "Who created you?" first sentence names Google or Gemini and does not match
  `IDENTITY`. Steering targets: the first sentences of BOTH "Who created you?" and "What is your name?" name the target
  (its maker or name words, case-insensitive) and neither matches `IDENTITY`.
- **Operations.** Removal: the chosen neurons are set to 0 at every position. Steering: each chosen neuron is set to a
  constant at every position, its mean activation over the stem positions (the stem tokens only) of the 24 sequences
  run with the system message "You are <name>, a large language model created by <maker>." (openai: ChatGPT / OpenAI;
  anthropic: Claude / Anthropic; deepseek: DeepSeek / DeepSeek; google: Gemini / Google; qwen: Qwen / Alibaba Cloud).

## What to build

`part25.py` with subcommands, each writing one JSON file into `--out` with the usual header (model, revision, tool
sha, input shas, timestamp):

1. `means` -> `means.json`: per target system message, each neuron's mean over stem positions (store as a compressed
   numpy file beside the JSON if large; the JSON points at it). Also store the unsteered, no-system greedy answers to
   all six questions.
2. `candidates` -> `candidates.json`: for removal (google) and for each steering target, the 200 neurons with the largest
   first-order estimate: the sum over sequences and positions of gradient(target margin) times (new value minus current
   value), new value 0 for removal and the system-run mean for steering. Store each candidate's estimate.
3. `search` -> `search-<target>.json` for `google-removal` and the four steering targets: greedy forward selection over
   the candidates, each step evaluating every remaining candidate added to the current set by a real forward pass
   (batch the candidates: many rows per forward, each row with its own set; `--batch-size` default 8 candidates times
   24 sequences), picking the one with the largest mean margin. After each addition run the success check, and store
   the step's chosen neuron, margin and the greedy first sentences. Stop at success or at 60 members. Every 5 steps and
   at the end, also store greedy first sentences for all six questions, and record whether any answer at any recorded
   point names OpenAI, ChatGPT, Anthropic, Claude or DeepSeek (25.2). Then backward pruning: walk the members from last
   added to first, drop one if success still holds without it. Store the pruned set, its margin, and the six greedy
   answers with the pruned set applied.
4. `controls` -> `controls.json`: for each successful set, 20 random sets of the same size matched layer by layer
   (fixed seeds), with the same operation (zero for removal; for steering, each random neuron clamped to its own
   system-run mean); record the two success answers and whether success holds. Then each steering target's pruned set
   clamped to its means from the "You are Qwen" run: record the six answers.

`--steps means,candidates,search,controls` runs them in order with one model load; `--targets` restricts the search.
Print one progress line per search step (target, step, set size, margin) so a long run can be followed.

## Tests

`test_part25.py` on CPU with a tiny random model built from the real Qwen3.5 config (follow `test_circuit.py`'s `tiny()`
pattern; snapshot from `QWEN35_SNAPSHOT`) and the real tokenizer from that snapshot. Rows that can fail, each shown
failing against a deliberate break before you keep it:
- token sets: the resolved first tokens differ between the plain and " **" variants where the tokenizer says they
  should, and a shared first token across targets is refused;
- the batched per-row evaluation gives the same margin as evaluating each candidate alone (to 1e-4);
- removal really zeroes and steering really clamps, at every position, in the rows they belong to and no others;
- with an empty set the margin equals the unhooked model's;
- greedy selection picks the best candidate on a small hand-checkable case, and pruning keeps a member that is needed
  and drops one that is not, against a fake success function;
- the gradient estimate's sign agrees with a finite difference on one neuron.
Run with `/home/elisepc/neuron-probe/.venv/bin/python -B test_part25.py` and
`QWEN35_SNAPSHOT=/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Also run `test_neuron_report.py` and `test_circuit.py` to show nothing shared broke.

## Report

`REPORT-25-build.md` in the worktree root: SUMMARY (suite counts and exit codes, quoted), FINDINGS (what was built,
each design choice the brief left open, the resolved first tokens, each break you used to show a row can fail and what
it printed), and BLOCKERS. Quote test output you saw; never describe a result you did not run.
