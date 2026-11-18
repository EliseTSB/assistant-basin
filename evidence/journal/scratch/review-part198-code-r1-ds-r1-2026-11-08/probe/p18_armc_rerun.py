"""Probe: Arm C rerun. A 'failure' row for a calibration model marks it prior (existing_keys counts failures),
so a rerun skips it; the T* block then sees it missing and returns 0, writing no tstar.json."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
import tempfile
from pathlib import Path
td = Path(tempfile.mkdtemp(prefix="p18-"))
log = td / "armC.jsonl"
R.append_jsonl(log, {"record_type": "armC_model", "run_mode": "registered", "model_id": "EleutherAI/pythia-410m", "max_t_word": 6.7})
R.append_jsonl(log, {"record_type": "failure", "run_mode": "registered", "model_id": "Qwen/Qwen3-0.6B", "error": "OOM"})
R.append_jsonl(log, {"record_type": "armC_model", "run_mode": "registered", "model_id": "allenai/OLMo-2-0425-1B", "max_t_word": 12.1})
prior = R.existing_keys(log, "model_id")
print("prior after a crash on Qwen3-0.6B:", sorted(prior))
print("pending:", [m["id"] for m in R.ARM_C_MODELS if m["id"] not in prior])
print("all three calibration models 'in current':", all(m["id"] in prior for m in R.ARM_C_MODELS[:3]))
print("-> run_arm_c returns 0 without writing tstar.json:", not (td / "tstar.json").exists())
