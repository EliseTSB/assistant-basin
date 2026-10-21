# Drives the REAL part43 FP32 reference-binding code (worktree ab-43-fp32) in a fresh process.
# Usage: probe_binding_gaps.py <mode>
# modes: control | l2norm_global | instance_call_impl | instance_norm | class_call_impl
import os
import sys
from types import SimpleNamespace

WORKTREE = "/home/elisepc/green/worktrees/ab-43-fp32"
sys.path.insert(0, WORKTREE)
os.environ["USE_HUB_KERNELS"] = "1"  # set to 1 to prove the preparation overrides it before Transformers import
import torch
from torch import nn
import part43

mode = sys.argv[1]
part43._prepare_fp32_reference_backend()
part43._load_fp32_reference_binding()
qwen = sys.modules[part43.FP32_QWEN_MODULE]
layer_class = qwen.Qwen3_5GatedDeltaNet
layer = layer_class.__new__(layer_class)
nn.Module.__init__(layer)
layer.norm = nn.Identity()
model = nn.Sequential()
model.add_module("layer0", layer)
model.config = SimpleNamespace(_attn_implementation="eager")
SENTINEL = "FOREIGN-CODE-RAN"

def foreign_l2norm(x, dim=-1, eps=1e-6):
    return x * 0 + 7  # not the pinned RMS-style l2norm

if mode == "l2norm_global":
    qwen.l2norm = foreign_l2norm
elif mode == "instance_call_impl":
    layer._call_impl = lambda *a, **k: SENTINEL
elif mode == "instance_norm":
    class ForeignNorm(nn.Module):
        def forward(self, x, gate):
            return x * 0 + 3
    layer.norm = ForeignNorm()
elif mode == "class_call_impl":
    layer_class._call_impl = lambda self, *a, **k: SENTINEL
elif mode != "control":
    raise SystemExit(f"unknown mode {mode}")

try:
    state = part43._fp32_reference_runtime_state(model)
    print(f"[{mode}] _fp32_reference_runtime_state: PASSED (no refusal)")
    print("  forward_binding.override_detected =", state["qwen3_5_forward_binding"]["override_detected"])
except part43.CalibrationRefusal as exc:
    print(f"[{mode}] _fp32_reference_runtime_state: REFUSED -> {exc}")
    raise SystemExit(0)
if mode in {"instance_call_impl", "class_call_impl"}:
    print("  calling model(x) returns:", model(torch.zeros(1)))
if mode == "instance_norm":
    print("  layer.norm class now:", type(layer.norm).__qualname__)
if mode == "l2norm_global":
    q = torch.ones(1, 2)
    print("  qwen.l2norm(q) now returns:", qwen.l2norm(q).tolist(), "(pinned would normalise)")
