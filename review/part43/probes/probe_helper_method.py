# Claude, 2026-10-21: an instance or class override of a NON-slot helper method that a pinned forward calls (RMSNorm._norm).
import os, sys
from types import SimpleNamespace
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-43-fp32")
import torch
from torch import nn
import part43
mode = sys.argv[1]
part43._prepare_fp32_reference_backend()
part43._load_fp32_reference_binding()
qwen = sys.modules[part43.FP32_QWEN_MODULE]
names = [n for n in dir(qwen) if n.endswith("RMSNorm")]
print("rmsnorm classes:", names)
cls = getattr(qwen, "Qwen3_5RMSNorm")
norm = cls(4)
model = nn.Sequential(); model.add_module("n", norm)
model.config = SimpleNamespace(_attn_implementation="eager")
if mode == "instance":
    norm._norm = lambda x: x * 0 + 9
elif mode == "class":
    cls._norm = lambda self, x: x * 0 + 9
try:
    part43._fp32_reference_runtime_state(model)
    print(f"[{mode}] PASSED; norm(ones) ->", norm(torch.ones(1, 4)).tolist())
except part43.CalibrationRefusal as exc:
    print(f"[{mode}] REFUSED -> {exc}")
