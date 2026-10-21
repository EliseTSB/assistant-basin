# Drives REAL part43._calibration_runtime_metadata (the pre-stage call path) after a new-API TF32 request.
import os, sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-43-fp32")
import torch
from torch import nn
import part43
part43._prepare_fp32_reference_backend()
part43._load_fp32_reference_binding()
torch.backends.cuda.matmul.fp32_precision = "tf32"
try:
    part43._calibration_runtime_metadata(nn.Linear(2, 2), "torch.float32", part43.FP32_SUCCESSOR_PROTOCOL_ID)
    print("no exception")
except part43.CalibrationRefusal as exc:
    print("CalibrationRefusal (HOLD path):", exc)
except Exception as exc:
    print("UNCAUGHT by run_pilot pre-stage (only CalibrationRefusal is caught):", type(exc).__name__, str(exc)[:80])
