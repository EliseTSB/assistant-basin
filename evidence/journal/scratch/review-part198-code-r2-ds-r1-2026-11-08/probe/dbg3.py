import sys, types, tempfile
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
R.torch = types.SimpleNamespace(equal=lambda a, b: True, zeros_like=lambda a: a, set_num_threads=lambda n: None)
src = open("probe/p32_armb_retry.py").read()
head = src.split("out = Path(tempfile.mkdtemp")[0]
exec(head)
print("R.torch before:", R.torch)
out = Path(tempfile.mkdtemp(prefix="dbg-")) / "out"; out.mkdir()
R.run_arm_b(1, out)
print("R.torch after:", R.torch)
