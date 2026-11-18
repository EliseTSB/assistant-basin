"""Arm C T*: is it still computed only from the three calibration models, and can a
deterministic failure loop? Drives the REAL run198.run_arm_c with stubbed model loads."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R

root = Path(tempfile.mkdtemp(prefix="armc-"))
out = root / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
loads, analyses = [], []

def fake_load(model_id, dtype):
    loads.append(model_id)
    return object(), object(), dtype

def fake_analyze(model, tok, S, model_id, role, dtype):
    analyses.append(model_id)
    # A calibration model that never produces a usable maximum.
    return {"model_id": model_id, "role": role, "max_t_word": None if model_id.startswith("Qwen/Qwen3-0.6B") else 9.0}

R.load_arm_c_pair = fake_load
R.analyze_model = fake_analyze
try:
    R.run_arm_c(None, out)
    print("registered run returned normally (unexpected)")
except RuntimeError as exc:
    print("registered run raised:", exc)
print("model loads:", loads)
print("tstar.json exists:", (out / "tstar.json").exists())
print("armC.jsonl rows:", [json.loads(l)["record_type"] for l in (out / "armC.jsonl").read_text().splitlines()])
print("=> the failure is raised after the row is written, so a restart retries it; a model that")
print("   fails deterministically stops every later registered Arm C run at the same model.")
