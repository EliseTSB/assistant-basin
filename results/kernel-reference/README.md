# Kernel references

CPU float32 greedy decodes of a fixed prompt set, written by `tools/kernel_reference.py`. A device or attention
kernel is checked against these with `tools/part_check.py kernel --candidate <record>` (PROCEDURE.md section 4).

| File | Model | How it was computed |
| --- | --- | --- |
| `qwen-qwen3.5-4b-cpu-float32.json` | Qwen/Qwen3.5-4B at 851bf6e8 | `--dtype float32-upcast --attn-implementation eager`, Windows CPU, 2026-11-04 00:12 to 01:04 CT |

`float32-upcast` keeps the Linear and Embedding weights in bf16 storage and does every computation in float32; the
checkpoint's own F32 tensors (48 on the 4B) are read back from the safetensors exactly. A bf16 weight converted to
float32 is exact, so the arithmetic is the float32 model's. It exists because the 4B in full float32 needs about
16 GB and the Windows CPU had about 14 free.

Proof (`upcast-proof/`, `tools/kernel_upcast_proof.py`): on Qwen3.5-0.8B, full float32 and float32-upcast give the
same tokens and the same top-two margins on all 16 rows at batch sizes 1 and 2. Control: a bf16 decode of the same
model differs from float32 on 8 of 16 rows' tokens and all 16 rows' margins, so the proof can see a difference.
