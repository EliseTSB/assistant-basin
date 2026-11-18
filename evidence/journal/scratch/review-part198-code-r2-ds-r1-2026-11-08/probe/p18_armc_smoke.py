"""A smoke Arm C run (--limit 1) writes a registered-looking armC_model row.
run_arm_c appends with run_mode='smoke'; arm_c_successful_keys filters smoke out. But
score198/gate never read armC.jsonl, and the model row has no run_mode check in analyze...
Check: does a smoke row ever satisfy prior or T*?"""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
out = Path(tempfile.mkdtemp(prefix="smoke-")) / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
R.load_arm_c_pair = lambda m, d: (object(), object(), d)
R.analyze_model = lambda *a, **k: {"model_id": a[3], "role": a[4], "max_t_word": 1.0}
R.run_arm_c(1, out)          # smoke
rows = [json.loads(l) for l in (out / "armC.jsonl").read_text().splitlines()]
print("rows:", [(r["record_type"], r.get("run_mode"), r.get("model_id")) for r in rows])
print("arm_c_successful_keys:", R.arm_c_successful_keys(out / "armC.jsonl"))
R.run_arm_c(None, out)       # registered
rows = [json.loads(l) for l in (out / "armC.jsonl").read_text().splitlines()]
print("after registered run, model rows:", [(r.get("run_mode"), r.get("model_id")) for r in rows if r["record_type"] == "armC_model"])
