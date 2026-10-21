# Build brief: part43.py (Luna)

Run size, multiplied out: under 9,000 short forward passes and 72 greedy answers of 40 tokens on Qwen3.5-4B
(see PART43-DESIGN.md); the forward-pass sections run later on the A770, not in this build.

Read PART43-DESIGN.md and the Part 43 rows of PREDICTIONS.md first. Build part43.py implementing methods A to H as
subcommands (`signs`, `dla`, `lattice`, `mediate`, `shift`, `links`, `sweep`, `dose`) plus `score`, reusing part25.py
(make_sequence, build_sequences, token sets, _score_margins, install_intervention_hooks) and part39.py's
load_model_direct (it streams Qwen3.5-4B onto the XPU and skips the allocator warmup the A770 refuses). Neuron =
one input coordinate of mlp.down_proj, as in Part 25. Model Qwen/Qwen3.5-4B at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; the base model is not needed here.

Specifics:
- The four units: (6,753), (7,1169), (11,8868), (29,776). Score = Part 25's Google-minus-Qwen margin at the maker
  stem's last position, plain and bold variants, for each of the six Part 25 questions; report per question.
- lattice: all 16 subsets zeroed at every position, and the same 16 subsets for 20 random four-unit sets matched by
  layer (seeded, seeds recorded). Fit per question: margin ~ main effects + 6 pairwise interactions (+ higher terms
  reported); report R^2 of main effects only, each pairwise interaction, and the random sets' 95th percentile of
  absolute pairwise interaction.
- mediate: zero unit U while forcing unit V to its clean activation at every position (clean values recorded from an
  unablated run of the same sequence). Report U's effect alone, with each V held, with all later units held.
- shift: zero each of 6, 7, 11 and record the activation change of each later unit at every stem position.
- links: weights only. For each ordered pair (earlier i, later j): dot and cosine of W_down[:, i] (layer li) with the
  later unit's gate row and up row, after the later layer's input RMSNorm weight (Qwen3.5 stores w; the scale is 1 + w;
  CHECK this against the model code and say which file and line in the report). Null: 1,000 seeded random units from
  the earlier layer against the same later unit; report percentile.
- dla: activation x (W_down[:, i] through final norm scale and unembedding) for the maker tokens; also through the J
  lens at ~/neuron-probe/jlens/qwen3.5-4b/... (part42.py already reads it; reuse its loader and its layer rule).
- sweep: write the 60 prompts to data/part43-sweep.json in six labelled groups of 10 as described in the design (write
  them yourself, plain and varied, no prompt reused across groups); mean activation per unit per group at the last
  position and at all positions.
- dose: clamp each unit alone at 2, 4, 8 x its largest absolute stem activation, sign kept, and also sign flipped,
  at every position; greedy 40-token answers to the six questions; store text.
- score: fill every Part 43 prediction from the outputs, exactly as written, with the numbers that decide it.
- Every output JSON carries model, revision, tool sha256 and input file hashes, like part25.py's header().

Tests: pytest with a tiny random model for every subcommand, AND at least one test that loads the real
results/part25/candidates.json token sets and runs the margin code on them, so a real-data shape mismatch fails here.
The weights-only subcommands (links, dla) must run for real on WSL CPU against the cached Qwen3.5-4B safetensors
(HF cache /mnt/e/neuron-probe/hf/hub, offline), reading only the matrices they need (part42.py shows how; keep RSS
under 4 GB). Commit nothing. Write REPORT-43-build.md ending with SUMMARY / FINDINGS / BLOCKERS, including the exact
commands for the forward-pass sections on the Windows GPU host (python part43.py <cmd> --device xpu --dtype bf16
--cache-dir E:\neuron-probe\hf\hub --out results\part43).
