# Is CPU generation unstable at sequence position 64?

Run on 2026-11-03 on a cloud machine with 4 cores, 16 GB RAM and no GPU. Branch `cloud/pos64-cpu-2026-11-03`, cut from `origin/main`.

## Answer

**No.** On CPU, no first difference falls at position 64 or 65, and none falls at 128. This holds for float32 against bf16 and for batch 1 against batch 2, in both models. The CPU differences that do occur are spread from position 19 to 109. In bf16, every one of them happens where the bf16 run's top two logits are exactly tied or one bf16 step apart (0.0 or 0.125, at most 0.25). They are ordinary rounding flips at near-ties.

In float32, batch 1 and batch 2 matched exactly on all 8 prompts. In bf16 they did not always match: CPU batch 1 and batch 2 differ in 4 of 8 rows on 0.8B and 3 of 8 on 4B. Each of those differences starts at an exact or near tie. The CPU runs are deterministic: repeating the 0.8B bf16 runs gave identical output, 16 of 16. So the batch 1 against batch 2 difference comes from the batch shape, not from run-to-run noise.

The position-64 pattern does appear when CPU output is compared with the **stored GPU (Arc) Part 140 rows**. Both plain arm-A rows diverge just after position 64. For "Who are you?" the stored row says "assist" and CPU says "help" (token 52, position 68). There the CPU bf16 margin is **5.25**, about 40 bf16 logit steps, so this is not a near-tie flip. For "Tell me about yourself." the divergence is at token 48, position 65 (the token predicted from position 64), with a CPU margin of 0.25. The CPU side is self-consistent around position 64. The position-64 divergence therefore comes from the GPU stack or from the difference between the two stacks, not from CPU instability. This is two rows, so the conclusion rests on a small sample.

## What was run

- **Script.** `pos64_cpu.py` generates the outputs; `analyze.py` does the comparisons and writes `results.json`. Prompts come from `part86/common86.token_inputs`, which uses `build_prompt` (chat template, `enable_thinking=False`). End-of-sequence ids come from `common86.eos_ids`. Generation uses `model.generate`: greedy, `max_new_tokens=96`, `use_cache=True`, no push hook. A zero push in Part 140 adds exact zeros, so leaving the hook out changes nothing. For batch 2, the prompt is expanded into two identical rows with no attention mask, as in run140. In every batch-2 run the companion row matched the kept row.
- **Prompts** (prompt length in tokens):

  | Prompt | Tokens |
  |---|---:|
  | Who are you? | 16 |
  | Tell me about yourself. | 17 |
  | What is the capital of France? | 19 |
  | Explain photosynthesis briefly. | 18 |
  | Write a short poem about the sea. | 20 |
  | How do I boil an egg? | 19 |
  | What causes rain? | 16 |
  | Give me three tips for better sleep. | 20 |

- **Models.**
  - Qwen/Qwen3.5-0.8B, pinned at revision `2fc06364715b967f1860aea9cf38778875588b17` (the current hub head when the run started). Run in float32 and bf16, each at batch 1 and batch 2.
  - Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Run in bf16 only, at batch 1 and batch 2.
- **Margins.** At every generated step, each run records the gap between its top two logits.
  - For the 0.8B pairs, the reported margin is the float32 batch-1 run's margin at the first-difference index. It is used only where that run shares the prefix up to that index.
  - For the 4B pairs, there is no float32 run, so the bf16 batch-1 margin is reported instead.
- **Position convention.** The absolute position is the prompt length plus the first-difference index. That is the position the first differing token occupies; it is predicted from the logits at the position one before.
- **Packages.** Python 3.11.18. torch 2.14.1+cpu, transformers 5.20.0, huggingface_hub 2.2.0, safetensors 0.8.0, numpy 2.4.6, accelerate 1.15.0 and tokenizers 0.23.3, all installed at the requested pins. Transformers used its reference PyTorch implementation for the linear-attention layers (`causal_conv1d` and `flash-linear-attention` are not installed), as in the part140-cpu run.
- **Timing.** The 0.8B run (32 generate calls) took 4 min 23 s. The 4B run (16 calls) took 8 min 15 s, about 0.3 s per step. The 0.8B bf16 repeat took about 2 min. The 4B part was kept.

## Results: CPU against CPU

Margins are given as "float32 batch-1 margin". In the 0.8B bf16 batch-1 against batch-2 row, the second value is the margin of the bf16 batch-1 run and then the bf16 batch-2 run. In the 4B row, all margins are the bf16 runs' own. "—" means the float32 run had already diverged earlier, so its margin does not apply.

| Pair | Identical | First differences: index → absolute position (margin) |
|---|---:|---|
| 0.8B f32-b1 vs f32-b2 | **8/8** | none |
| 0.8B f32-b1 vs bf16-b1 | 4/8 | Who 37→53 (0.113); poem 40→60 (0.016); rain 3→19 (0.022); sleep 57→77 (0.068) |
| 0.8B f32-b1 vs bf16-b2 | 2/8 | Who 37→53 (0.113); photo 27→45 (0.145); poem 7→27 (0.055); egg 27→46 (0.029); rain 3→19 (0.022); sleep 57→77 (0.068) |
| 0.8B f32-b2 vs bf16-b1 | 4/8 | same as f32-b1 vs bf16-b1 |
| 0.8B f32-b2 vs bf16-b2 | 2/8 | same as f32-b1 vs bf16-b2 |
| 0.8B bf16-b1 vs bf16-b2 | 4/8 | photo 27→45 (f32 0.145; bf16 0.125 / **0.0**); poem 7→27 (0.055; 0.125 / **0.0**); egg 27→46 (0.029; 0.125 / **0.0**); sleep 89→109 (—; **0.0** / 0.125) |
| 4B bf16-b1 vs bf16-b2 | 5/8 | photo 70→88 (**0.0** / 0.25); poem 61→81 (0.125 / **0.0**); sleep 46→66 (**0.0** / 0.125) |

- **No CPU first difference falls at position 64 or 65, or at 128.** The positions seen are 19, 27, 45, 46, 53, 60, 66, 77, 81, 88 and 109. The nearest to 64 is 4B "sleep" at position 66, where the bf16 batch-1 margin is an exact tie (0.0).
- **Float32 versus bf16.** The flips happen where the float32 margin is small, between 0.016 and 0.145 logits.
- **Batch 1 versus batch 2.** They never differ in float32. They do differ in bf16 (7 of 16 rows across the two models). At every such first difference, one of the two bf16 runs has its top two logits exactly tied. The batch shape changes bf16 rounding just enough to break the tie the other way.
- **Determinism.** Rerunning the 0.8B bf16 runs gave identical output in 16 of 16 rows (`repeat_0.8B_bf16` in `results.json`).

## Results: stored Part 140 rows against CPU 4B bf16

The two stored rows are the Part 140 plain arm-A rows (stored batch size 1, layer 12, seed 140000). Both the stored text and the CPU text were re-tokenized with `add_special_tokens=False`. For these two rows, re-tokenizing the CPU text gives back the generated ids exactly. The comparison runs over the common length: the stored rows stopped at 80 new tokens, the CPU rows at 96.

| Question | CPU batch | Stored / CPU tokens | First diff index | Absolute position | CPU bf16 margin there |
|---|---:|---:|---:|---:|---:|
| Who are you? | 1 and 2 | 56 / 56 | 52 | 68 | **5.25** |
| Tell me about yourself. | 1 and 2 | 80 / 96 | 48 | 65 (predicted from 64) | 0.25 |

These are the same first differences that `part140-cpu` found for the plain rows. CPU batch 1 and batch 2 agree with each other on both rows. Around position 64 in "Who are you?", the CPU margins are large: 7.5 at position 64 and 3.4, 9.25, 9.6 and 5.25 at positions 65 to 68. The CPU is confident there, so the stored "assist" is not a coin flip that CPU happened to lose.

## Limits

- Eight short prompts. Three of the 0.8B rows and two of the 4B rows end before position 64 + a few tokens: the 0.8B Who, Tell and France rows end at positions 69, 66 and 65, and the 4B Who and France rows at 73 and 82. They still pass position 64.
- The CPU uses the reference PyTorch linear-attention path. A chunked kernel, such as flash-linear-attention or whatever the Arc stack used, is not tested here. The question of whether that path changes behaviour at the chunk boundary of 64 remains open, and a run on the original stack (or a CPU run with `flash-linear-attention` installed) would test it.
- There is no float32 run for 4B. bf16 4B fits in 16 GB; float32 does not.
- The stored-row comparison covers only 2 rows.

```
CLOUD-RESULT: pos64-cpu (2026-11-03, 4 cores, no GPU).
Identical rows per pair: 0.8B f32-b1 vs f32-b2 8/8; f32-b1 vs bf16-b1 4/8; f32-b1 vs bf16-b2 2/8;
f32-b2 vs bf16-b1 4/8; f32-b2 vs bf16-b2 2/8; bf16-b1 vs bf16-b2 4/8; 4B bf16-b1 vs bf16-b2 5/8;
stored Part140 plain A vs 4B CPU bf16 0/2 (b1 and b2 alike).
First differences on CPU: absolute positions 19,27,45,46,53,60,77,109 (0.8B) and 66,81,88 (4B).
None at 64/65 or 128. bf16 flips occur at exact or near ties (bf16 margin 0.0-0.25);
f32-vs-bf16 flips at f32 margin 0.016-0.145. Stored GPU vs CPU: positions 68 (Who; CPU margin 5.25, not a tie)
and 65 (Tell; margin 0.25), so the position-64 clustering is not reproduced CPU-vs-CPU.
CPU batch 1 vs batch 2: never differ in float32 (8/8 identical); differ in bf16 in 4/8 (0.8B) and 3/8 (4B) rows,
always starting at an exact bf16 logit tie. CPU runs are deterministic (0.8B bf16 repeat 16/16 identical).
4B part not dropped (0.8B took 4.4 min, 4B 8.2 min).
```
