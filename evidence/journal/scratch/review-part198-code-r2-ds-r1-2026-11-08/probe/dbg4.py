import sys, types, tempfile, json
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
R.torch = types.SimpleNamespace(equal=lambda a, b: True, zeros_like=lambda a: a, set_num_threads=lambda n: None)
src = open("probe/p32_armb_retry.py").read()
head = src.split("out = Path(tempfile.mkdtemp")[0]
exec(head)
out = Path(tempfile.mkdtemp(prefix="dbg-")) / "out"; out.mkdir()
stub.fail_undo = True
R.run_arm_b(1, out)
stub.fail_undo = False
R.run_arm_b(1, out)
for l in (out / "armB-wsp.jsonl").read_text().splitlines():
    r = json.loads(l)
    print(r["record_type"], r.get("phase"), (r.get("traceback") or "")[-200:].replace("\n", " | "))
