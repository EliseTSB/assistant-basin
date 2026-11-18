"""A smoke Arm C run (--limit 1) that FAILS: the failure row is written with run_mode='smoke'
(ok), and main() then appends a SECOND failure row with run_mode='smoke' as well. Check that a
smoke failure never blocks a registered run (arm_c_successful_keys ignores smoke)."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
out = Path(tempfile.mkdtemp(prefix="armc5-")) / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
R.load_arm_c_pair = lambda m, d: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    R.run_arm_c(1, out)
except RuntimeError:
    pass
rows = [json.loads(l) for l in (out / "armC.jsonl").read_text().splitlines()]
print([(r["record_type"], r.get("run_mode"), r.get("model_id")) for r in rows])
print("successful keys after a smoke failure:", R.arm_c_successful_keys(out / "armC.jsonl"))
