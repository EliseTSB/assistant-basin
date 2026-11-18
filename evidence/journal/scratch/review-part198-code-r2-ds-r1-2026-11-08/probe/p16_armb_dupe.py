"""Arm B (not B-seq) resume: existing_keys() suppresses a case whose row was already written, so
the runner never appends a second case row. But the armB-seq replay path rewrites every row when
a resume happens (run198.py:1425 `if runmode == "smoke" or str(case_id) not in prior`), and the
replay recomputes statistics. Show the scorer's behaviour on a legitimate replay whose recomputed
numbers differ only in the last digit."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import score198 as S
td = Path(tempfile.mkdtemp(prefix="replay-"))
row = {"record_type": "case", "run_mode": "registered", "run_id": "p1", "timestamp": "t1",
       "case_id": 0, "n_written_so_far": 1, "took": True, "name_eligible": True,
       "names_after_write": {"0": {"named_strict": True}}}
drift = dict(row, run_id="p2", timestamp="t2")
drift["names_after_write"] = {"0": {"named_strict": True, "pairs": [{"z": 1.0000001}]}}
p = td / "armB-seq-wsp.jsonl"
p.write_text(json.dumps(row) + "\n" + json.dumps(drift) + "\n")
try:
    print(S.registered_rows(p, "case", "case_id"))
except ValueError as exc:
    print("registered_rows raises:", exc)
