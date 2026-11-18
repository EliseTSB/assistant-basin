"""Two registered-process failures with no case_id, then a case row for case 1.
run198.main() appends every startup/run failure as a 'failure' row with no case_id; the
second such row is a duplicate terminal row for key None in score198.registered_rows."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
import score198 as S

root = Path(tempfile.mkdtemp(prefix="main-dup2-"))
out = root / "out"
R.root_out_dir = lambda path: Path(path).expanduser().resolve()
R.load_gpt2_model = lambda: (_ for _ in ()).throw(RuntimeError("stub: stop before loading weights"))
for attempt in (1, 2):
    rc = R.main(["armA", "--method", "ROME", "--out-dir", str(out)])
    print(f"main() attempt {attempt} rc =", rc)
log = out / "armA-rome.jsonl"
for line in log.read_text().splitlines():
    row = json.loads(line)
    print("   ", row.get("record_type"), "| case_id:", row.get("case_id"), "|", row.get("error", "")[:45])
with log.open("a") as f:
    f.write(json.dumps({"record_type": "case", "run_mode": "registered", "case_id": 1,
                        "took": True, "restored_bit_exact": True}) + "\n")
try:
    print("registered_rows ->", S.registered_rows(log, "case", "case_id"))
except ValueError as exc:
    print("registered_rows raises:", exc)
