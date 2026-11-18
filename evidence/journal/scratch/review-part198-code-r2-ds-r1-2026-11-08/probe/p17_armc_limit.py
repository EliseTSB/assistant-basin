"""M7: with --limit 2 and nothing prior, how many calibration models load? Drives run_arm_c."""
import sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
out = Path(tempfile.mkdtemp(prefix="lim-")) / "out"
out.mkdir()
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
loads = []
R.load_arm_c_pair = lambda m, d: (loads.append(m), (object(), object(), d))[1]
R.analyze_model = lambda *a, **k: {"model_id": a[3], "role": a[4], "max_t_word": 1.0}
for limit in (1, 2, 3):
    loads.clear()
    R.run_arm_c(limit, out)
    print(f"--limit {limit}: loaded {len(loads)} ->", loads, "| tstar.json:", (out / "tstar.json").exists())
