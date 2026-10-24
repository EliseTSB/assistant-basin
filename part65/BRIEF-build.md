# Brief: build Part 65 (read labels vs write labels), code and tests only

Domain: /home/elisepc/green/worktrees/ab-part65/part65 (write only there; read the whole worktree and
/home/elisepc/green/worktrees/ab-part64, which holds Part 64's local result files). Do NOT commit (your sandbox cannot
reach this worktree's git directory; Claude commits after review). Do NOT run anything on the Intel Arc A770, start a
GPU job, download a model, or call any paid API. Build and test only.

Read first: ../PART65-DESIGN.md (the registered design; it is the contract, change nothing in it), ../part64/collect64.py
and ../part64/collect64_model.py (how the 500 dev units and their control directions are defined: atlas seed 5601, pool
100 per layer, `atlas_collect_fast2.generate_controls`), ../part60/gen60.py and ../part63/ (how a Haiku batch reader is
built, thinking disabled, through the paid-action lifecycle in ../paidaction/), ../gpu_job_runner.py and
/mnt/c/Users/alexa/neuron-probe-gpu/job.py (how a step runs on the A770 from Windows Python: steps.txt, progress.log,
the RAM guard, copy engine off), and ../part64/REPORT-build.md for the house testing style.

Build these, each with a module docstring stating what it reads and writes:

1. `steer65.py` — the write side, runnable on the A770 (Windows Python, device `xpu`, falling back to `cpu` for tests;
   guard any POSIX-only import such as fcntl). For each of the 500 Part 60 dev units (unit list from
   /home/elisepc/green/worktrees/ab-part64/results/part64/collection.jsonl: `unit_key`, `type`, `layer`, `unit`,
   `control_id`): write vector = W_down column (neuron) or W_down @ r (control, r from generate_controls), scaled to unit
   norm, times dose x the layer's median residual norm (measured once per layer on 2,000 tokens of the Part 60 corpus and
   saved). Add it to that layer's MLP output at every position. Rollouts and steer lens exactly as the design says (3
   prompts x doses -0.7/+0.5/+0.8, greedy, 48 new tokens, chat template; one baseline per prompt; steer lens +/-0.8 on 4
   neutral contexts, top 12 tokens per sign). Batch the 9 rollouts of a unit. Write one JSONL row per unit (resumable:
   skip units already written; atomic append), plus a manifest with model revision, doses, prompts and medians.
   `--units N --type both --seed` selects a stratified pilot subset. Built-in checks, failing loudly: dose 0 reproduces
   the baseline token for token; real and control push norms are equal at every layer within 1e-4.
2. `rank65.py` — write words: for each unit, the 10 words most over-represented in its rollouts against the pooled
   control rollouts of the same layer band (thirds of the 28 layers), Lyra's word-cluster-mass idea; document the exact
   formula in the docstring.
3. `judge65.py` — the blind 2AFC judge through the Batch API with the paid-action lifecycle (reserve, call, settle,
   recover; reuse ../paidaction, never a second ledger). Two trials per unit, foils from the same type and layer band,
   order swapped; the prompt shows one read label (Part 64 Haiku-1x hypothesis for that unit) and two write summaries
   (write words + steer-lens tokens) and asks for A or B; it never reveals layer, unit id or type. Haiku 5.5, thinking
   disabled. A one-request live probe mode (`--probe`) that sends exactly one request, for Claude to run first.
4. `analyze65.py` — 65a-65f exactly as registered, with unit-bootstrap 95% intervals, writing analysis-65.json and .md.
5. `test_part65.py` — unittest, using a tiny random-weight model built in the test (no downloads) and a fake batch API.
   It must FAIL, not skip, if torch is missing. Cover: write-vector norms equal for real and control; dose 0 = baseline;
   resume skips finished units; foils never the unit itself and never another type or band; the judge prompt contains no
   layer, unit id or type; analysis reproduces a planted effect and a planted null.

Finish with `REPORT-build.md` (SUMMARY / FINDINGS / BLOCKERS): files, the exact test command and its output, every place
you had to choose something the design does not say, and the measured run time of one unit on CPU with the tiny model.
