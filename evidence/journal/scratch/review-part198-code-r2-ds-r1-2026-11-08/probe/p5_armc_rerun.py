"""M5: a calibration model that FAILED is retried on the next registered run, and T* is
computed only from the three calibration models. Drives the REAL run_arm_c, stubbed loads."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R

root = Path(tempfile.mkdtemp(prefix="armc2-"))
out = root / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
state = {"fail": {"Qwen/Qwen3-0.6B"}, "loads": [], "analyses": []}

def fake_load(model_id, dtype):
    state["loads"].append(model_id)
    if model_id in state["fail"]:
        raise RuntimeError("simulated deterministic load failure")
    return object(), object(), dtype

def fake_analyze(model, tok, S, model_id, role, dtype):
    state["analyses"].append(model_id)
    values = {"EleutherAI/pythia-410m": 6.73, "Qwen/Qwen3-0.6B": 10.33,
              "allenai/OLMo-2-0425-1B": 8.0, "EleutherAI/pythia-1.4b": 99.0}
    return {"model_id": model_id, "role": role, "max_t_word": values.get(model_id, 1.0)}

R.load_arm_c_pair = fake_load
R.analyze_model = fake_analyze
try:
    R.run_arm_c(None, out)
except RuntimeError as exc:
    print("run 1 raised:", exc)
print("run 1 loads:", state["loads"])
state["loads"].clear(); state["fail"].clear()
R.run_arm_c(None, out)
print("run 2 loads (after the failure is cleared):", state["loads"])
print("run 2 analysed:", state["analyses"])
tstar = json.loads((out / "tstar.json").read_text())
print("tstar:", tstar["t_star"], "calibration_maxima:", tstar["calibration_maxima"])
print("=> evaluation model pythia-1.4b has max_t_word 99 but T* =", tstar["t_star"],
      "(calibration only:", tstar["t_star"] == max(tstar["calibration_maxima"].values()), ")")
