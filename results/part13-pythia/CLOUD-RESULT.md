# Part 13 Pythia-1.4b own-run (cloud, 2026-10-16/17)

Environment: cloud CPU box (~4 cores, 15 GB), torch 2.14.1+cpu, transformers 5.20.1.

## What was run
1. venv; `pip install torch --index-url https://download.pytorch.org/whl/cpu`; `pip install transformers accelerate safetensors numpy`.
2. First attempt, unmodified: `python collect.py EleutherAI/pythia-1.4b out --self-name "Pythia:EleutherAI" --dtype float32`
   FAILED after 45.7 s: `AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'model'` (collect.py line 41 assumes `model.model.layers` and `mlp.down_proj`, which GPT-NeoX lacks; it uses `gpt_neox.layers` and `mlp.dense_4h_to_h`). `/usr/bin/time` is not installed; wall time and peak RSS were measured with a small python wrapper (`resource.getrusage(RUSAGE_CHILDREN)`).
3. Rerun with no repo code changed, through a shim kept outside the repo (copy: `neox_shim.py` here) that aliases `model` -> `gpt_neox` and `mlp.down_proj` -> `mlp.dense_4h_to_h` (same module objects) and then runs collect.py unchanged:
   `python neox_shim.py collect.py EleutherAI/pythia-1.4b out --self-name "Pythia:EleutherAI" --dtype float32`
   - 24 layers x 8192 neurons, plain form only (no chat template), 72 prompts x 6 families, 1024 neutral tokens
   - wall time: 241.2 s (collection 236 s); peak memory: 9,562,684 KB (~9.1 GiB RSS)
4. `python analyze.py out/EleutherAI__pythia-1.4b.npz --form plain --json out/pythia-own.json` (exit 0). Stdout in `analyze-own.txt`, JSON in `pythia-own.json`. The .npz is not committed.

## From the JSON
- heldout_top_median: 0.3182085156440735
- heldout_random_median: -0.18501850962638855
- null_top_max: 0.25366026163101196
- layer_histogram_top: [0, 2, 5, 6, 5, 4, 2, 1, 1, 2, 1, 3, 0, 2, 0, 1, 0, 0, 2, 1, 3, 1, 3, 5]
- claude_in_discover: claude_minus_olmo -0.24504438042640686, claude_minus_person 0.3270173668861389, chatgpt_minus_person 0.5012055039405823, claude_minus_chatgpt -0.19648203253746033, olmo_minus_person 0.6017113924026489, named_frames 23

Predictions are not scored here.
