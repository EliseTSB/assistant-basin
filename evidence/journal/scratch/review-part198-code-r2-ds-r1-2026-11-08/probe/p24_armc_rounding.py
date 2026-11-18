"""Arm C T*: current[m]["max_t_word"] is used raw. Is the calibration maximum rounded before
T* (design/reviewer references are rounded)? And does the scorer read tstar.json without
checking it was produced by calibration only?"""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R, score198 as S
out = Path(tempfile.mkdtemp(prefix="armc6-")) / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
vals = {"EleutherAI/pythia-410m": 6.7349, "Qwen/Qwen3-0.6B": 10.3299, "allenai/OLMo-2-0425-1B": 8.0}
R.load_arm_c_pair = lambda m, d: (object(), object(), d)
R.analyze_model = lambda *a, **k: {"model_id": a[3], "role": a[4], "max_t_word": vals.get(a[3], 1.0)}
R.run_arm_c(None, out)
t = json.loads((out / "tstar.json").read_text())
print("t_star:", t["t_star"], "| calibration maxima:", t["calibration_maxima"])
print("rounded to the reviewer's two decimals:", round(t["t_star"], 2))
# A hand-written tstar.json with a t_star that the scorer accepts.
(out / "tstar.json").write_text(json.dumps({"t_star": 99.0, "calibration_maxima": {"x": 1.0}}))
print("scorer reads a tstar.json with no calibration provenance:",
      json.loads((out / "tstar.json").read_text())["t_star"])
