"""M5 residual: a calibration model whose analysis succeeds but whose row is written with
max_t_word None is re-analysed on every registered run (arm_c_successful_keys accepts any
armC_model row). And a deterministic *analysis* failure (not a load failure) is retried forever:
each registered run appends another failure row. Show both."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
out = Path(tempfile.mkdtemp(prefix="armc3-")) / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
loads = []
R.load_arm_c_pair = lambda m, d: (loads.append(m), (object(), object(), d))[1]
R.analyze_model = lambda *a, **k: {"model_id": a[3], "role": a[4],
    "max_t_word": None if a[3] == "allenai/OLMo-2-0425-1B" else 5.0}
for attempt in range(3):
    loads.clear()
    try:
        R.run_arm_c(None, out)
    except RuntimeError as exc:
        print(f"attempt {attempt + 1}: {exc} | loaded {loads}")
rows = [json.loads(l) for l in (out / "armC.jsonl").read_text().splitlines()]
print("armC_model rows:", [(r.get("model_id"), r.get("max_t_word")) for r in rows if r["record_type"] == "armC_model"])
print("failure rows:", sum(r["record_type"] == "failure" for r in rows))
print("=> a model that deterministically yields no max_t_word is re-loaded on every registered run;")
print("   nothing stops the loop, and the stream accumulates one failure row per attempt.")
