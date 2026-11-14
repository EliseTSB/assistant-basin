"""Windows smoke for Part 205 family Q's entry path (Amendment 3, 2026-11-14), run with the A770 job's own Python in a
copy laid out like the job folder: part205/*.py beside this file, neuron_report.py one level up. Checks the root import,
the stage lock (a second process is refused, the lock is free again after release) and write_models_json."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run205 as R            # appends the job-folder root
import common205 as C
import neuron_report          # the real module, from the root
print("imports ok:", Path(neuron_report.__file__).name)
lock = HERE / "smoke-stage" / ".lock"; lock.parent.mkdir(exist_ok=True)
L = C.StageLock(lock)
probe = ("import sys; sys.path.insert(0, sys.argv[1]); import common205 as C\n"
         "try:\n    C.StageLock(sys.argv[2]); print('SECOND LOCK TAKEN')\n"
         "except SystemExit as e:\n    print('second refused:', str(e)[-40:])\n")
r = subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(HERE), str(lock)], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-400:]); assert "second refused" in r.stdout
L.release()
L2 = C.StageLock(lock); L2.release(); print("lock free after release: ok")
mj = HERE / "smoke-models.json"
R.write_models_json(mj, {"stage": "a", "model_id": "x"}); R.write_models_json(mj, {"stage": "b", "model_id": "y"})
d = json.loads(mj.read_text(encoding="utf-8")); assert set(d["stages"]) == {"a", "b"}, d
print("write_models_json: both stages kept"); print("WINDOWS SMOKE OK")
