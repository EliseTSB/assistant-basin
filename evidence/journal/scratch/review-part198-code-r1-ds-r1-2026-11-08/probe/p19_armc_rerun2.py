"""Probe: Arm C rerun after a failure. run_arm_c:1421-1424 builds `current` from armC_model rows only and
returns 0 silently if any calibration model is missing -- and existing_keys (line 1391) counts a 'failure'
row as prior, so the model is never retried."""
import sys, tempfile, inspect
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
src = inspect.getsource(R.run_arm_c)
print("T* guard returns 0 without error:", "if not all(m[\"id\"] in current for m in ARM_C_MODELS[:3]):\n        return 0" in src)
print("`current` is built from record_type == 'armC_model' only:", "record_type\") == \"armC_model\"" in src)
print("resume prior comes from existing_keys (counts failures):", "prior = existing_keys(log, \"model_id\")" in src)
td = Path(tempfile.mkdtemp(prefix="p19-"))
log = td / "armC.jsonl"
R.append_jsonl(log, {"record_type": "failure", "run_mode": "registered", "model_id": "Qwen/Qwen3-0.6B"})
print("after one failed calibration model, prior contains it:", "Qwen/Qwen3-0.6B" in R.existing_keys(log, "model_id"))
