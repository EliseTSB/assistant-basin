import sys, traceback
from types import SimpleNamespace
from pathlib import Path
REPO = "/home/elisepc/green/worktrees/ab-part200"
sys.path.insert(0, REPO)
from part200 import run200
import tempfile
d = Path(tempfile.mkdtemp(prefix="p200probe-"))
args = SimpleNamespace(device="xpu", model="stub", units=f"{REPO}/part163/units163.json",
                       out=str(d / "answers.jsonl"), batch_size=64)
try:
    rc = run200.run(args)   # exactly what main() does: run(args), no loader, no generate_fn
    print("RETURNED", rc)
except SystemExit as e:
    print("SYSTEMEXIT", e)
except BaseException as e:
    print("RAISED", type(e).__name__, e)
    traceback.print_exc(limit=4)
print("files:", sorted(p.name for p in d.iterdir()))
