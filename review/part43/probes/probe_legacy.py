# Legacy BF16 metadata probe: drives _runtime_environment(model) on the REAL pinned Qwen module (no FP32 preparation).
# Usage: probe_legacy.py <module_dir>   (module_dir = worktree or the base-commit copy in scratch)
import hashlib, json, os, sys
from torch import nn
module_dir = sys.argv[1]
sys.path.insert(0, module_dir)
os.environ["USE_HUB_KERNELS"] = "0"
import part43
print("part43 loaded from", part43.__file__)
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen  # real pinned module, hub disabled at import
model = nn.Linear(2, 2)
env = part43._runtime_environment(model)
impl = env["kernel"]["qwen3_5_function_implementations"]
print(json.dumps(impl, indent=1, sort_keys=True))
print("qwen_implementation_sha256 (as calibration records it):", hashlib.sha256(part43._canonical_json(impl).encode()).hexdigest())
print("gated_delta_or_convolution:", env["kernel"]["gated_delta_or_convolution"])
