# Brief: build Part 30 (a controlled teacher swap on one small base)

Read PREDICTIONS.md Part 30 (the design and predictions, written before this build), and part26.py / neuron_report.py
for the house style: a step CLI (`--steps a,b,c`), `nr.load_model`, `nr.spill_guard(device, max_gpu_gb)`, results as
JSON with a header naming inputs and their sha256. Do not edit any existing file. Work only in this worktree. No
network, and install nothing: the GPU venv has torch and transformers but NOT peft or datasets, so write LoRA yourself
(a small wrapper around chosen nn.Linear modules: frozen base weight plus trainable A and B, rank r, alpha, B starts at
zero; save and load only A and B).

## Inputs (the host has them; pass paths on the command line)
- `--pairs E:\neuron-probe\data\s1k\s1k11-pairs.jsonl`: 1,000 rows {index, question, gemini_thinking, gemini_attempt,
  deepseek_thinking, deepseek_attempt, cot_type} from simplescaling/s1K-1.1 (same questions, two teachers).
- Base model `Qwen/Qwen2.5-0.5B-Instruct` (in the HF cache, offline).
- `--prompts data/part29-prompts.json` style file (Part 29's 300 everyday prompts; only indices 240-299 are used).

## part30.py steps
1. `split`: a fixed seed-0 split of the 1,000 questions into 900 train and 100 held out; write `results/part30/split.json`.
2. `control`: the no-teacher control data: the base model's own answer to each of the 900 train questions (greedy,
   max 1024 new tokens), saved with the question. Resumable.
3. `train --teacher r1|gemini|control`: one LoRA adapter per teacher. Training text = the chat template with the user
   question and the assistant turn `<think>\n{thinking}\n</think>\n\n{attempt}` (for control: the base's own answer
   with no think block). Truncate every assistant turn to the SAME token budget (`--max-target 1024`), loss on the
   assistant tokens only. Identical settings for all three: LoRA on q_proj, k_proj, v_proj, o_proj, up_proj, down_proj,
   gate_proj of every layer, rank 16, alpha 32, lr 2e-4, 2 epochs, effective batch 8 (micro-batch from
   `--micro-batch`, default 2, with gradient accumulation), seed 0, bf16 weights. Log loss every 20 steps; save
   `results/part30/adapter-<teacher>.pt` and a JSON record of every setting and the final loss.
4. `generate --teacher ...`: with the adapter loaded, answer Part 29 prompts 240-299 (no system prompt, sampling 0.7,
   max 600 new tokens, enable_thinking off if the template takes it), writing
   `results/part29/answers-part30-<teacher>.jsonl` in exactly Part 29's row format (index, model, provider, text,
   answer = text after the last `</think>` if any, finish_reason, usage, wall_time), so part29_classify.py can score it.
   Also `--teacher base` for the base model with no adapter.
5. `loss`: for base and each adapter, mean per-token loss on the 100 held-out questions' R1 assistant turns and,
   separately, their Gemini turns (same truncation), teacher-forced; write `results/part30/loss.json`.
6. `probe`: for each of base and the three adapters, mean-pooled residual stream per layer on the 100 held-out R1
   turns and the 100 Gemini turns (teacher-forced, first 512 tokens); a logistic-regression probe (numpy, 5-fold by
   question so a question's two traces stay in one fold) R1-vs-Gemini AUROC per layer; write `results/part30/probe.json`.

Every step: `--device xpu|cpu`, `--max-gpu-gb 14.5` through `nr.spill_guard`, `torch.xpu.empty_cache()` between
models, a header (inputs with sha256, settings, revision), and printing with flush=True so progress shows in logs.

## Tests (`test_part30.py`, offline, CPU)
Build a tiny random Qwen2 model from a config in the test (2 layers, small hidden size) and a tiny tokenizer stand-in
or the real tokenizer if it loads offline; test: the LoRA wrapper is the identity at init (B=0), trains (loss falls on
a toy target), and save/load round-trips; the split is deterministic; truncation gives every teacher the same budget;
loss masking covers only assistant tokens; the generate row format matches Part 29's; the probe gives about 0.5 AUROC
on labels shuffled at random and about 1.0 on a planted separable feature. A sabotage row: make truncation
teacher-dependent and show a test goes red.

Report in REPORT-30-build.md: SUMMARY, FINDINGS (anything in the design that does not hold), BLOCKERS, HOST COMMANDS
(the exact steps.txt lines for the A770 job runner, which runs `python <line>` in C:\Users\alexa\neuron-probe-gpu, plus a
time estimate per step). Do not commit: Claude commits after reading the report.
