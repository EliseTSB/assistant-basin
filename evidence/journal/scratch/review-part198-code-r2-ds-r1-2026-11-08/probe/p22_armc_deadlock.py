"""A calibration model that deterministically fails to LOAD: each registered run retries it,
writes a failure row, and raises. Nothing ever completes; T* is never written; the evaluation
and host models never run. Show three attempts."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
out = Path(tempfile.mkdtemp(prefix="armc4-")) / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
R.load_arm_c_pair = lambda m, d: (_ for _ in ()).throw(RuntimeError("safetensors: bad header")) \
    if m == "allenai/OLMo-2-0425-1B" else (object(), object(), d)
R.analyze_model = lambda *a, **k: {"model_id": a[3], "role": a[4], "max_t_word": 5.0}
for attempt in range(3):
    try:
        R.run_arm_c(None, out)
    except RuntimeError as exc:
        print(f"attempt {attempt + 1} raised: {exc}")
rows = [json.loads(l) for l in (out / "armC.jsonl").read_text().splitlines()]
print("failure rows:", sum(r["record_type"] == "failure" for r in rows),
      "| armC_model rows:", sum(r["record_type"] == "armC_model" for r in rows),
      "| tstar.json:", (out / "tstar.json").exists())
