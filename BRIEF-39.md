# Build brief: part39.py (Luna)

Run size (multiplied out, written first): 2 models x 120 sentences x 2 wrappings = 480 forward passes; generations:
B 6 questions x (1 greedy + 8 samples) x 2 arms = 108, plus 10 controls x 6 x 2 = 120; C 12 questions x 8 x 3 arms
= 288, plus 10 controls x 12 x 1 = 120. About 640 generations of at most 60 tokens.

Read PART39-DESIGN.md and the Part 39 rows in PREDICTIONS.md (do not edit either). Build part39.py with subcommands
`reads`, `zero`, `push`, `score`, plus data/part39-battery.json (6 categories x 20 authored sentences, about 12 words
each, as the design lists; no sentence names a real AI product) and data/part39-persona.json (the Sam system prompt and
12 everyday questions). Reuse the existing helpers rather than rewriting them: part25.py and part27.py (in this tree)
already load Qwen3.5-4B offline, define the neuron as an input unit of layer.mlp.down_proj via a forward pre-hook, pick
seeded random control units per layer (part27.control_sets), and apply the chat template. Match their CLI style
(--device xpu default, --out results\part39, --batch-size, --model/--revision/--base-model/--base-revision, seed 3901).

Requirements:
- reads: for each model, each sentence plain and as the user turn of the chat template (enable_thinking False if the
  template supports it, as part25 does), record the unit's max over the sentence's own tokens and, in the template,
  its value at the "assistant" role token; same for 20 seeded random layer-6 units. Write reads.jsonl + reads-summary.json
  (medians per category per model per wrapping, with seeded bootstrap 95% CIs).
- zero and push as the design says; push's clamp value is k x the mean chat-model "assistant"-marker value from
  reads (fail clearly if reads has not run). Rows resumable by key; never append duplicates.
- score: computes 39.1 to 39.6 inputs exactly as worded (AI-claim regex: \b(AI|artificial intelligence|language model|
  assistant|chatbot)\b case-insensitive, and Part 25's maker list) and writes score.json; a broken answer (fewer than 3
  words or the same 4-gram repeated 3+ times) never counts as an AI claim.
- Tests (pytest, tiny random model, no downloads): hook zeroes/clamps only unit 753 at every position; the assistant
  marker index is found in the templated tokens; resume does not duplicate rows; scorer rules on hand examples. Add one
  deliberate mutant check (clamp at the wrong layer must turn a test red).
- Memory: one model on the device at a time; free it before loading the next (gc + torch.xpu.empty_cache).
Write REPORT-39-build.md ending with SUMMARY / FINDINGS / BLOCKERS and the exact Windows command lines in the order
`part39.py <subcommand> <options>`. Do not run real models. Commit nothing (the host commits).
