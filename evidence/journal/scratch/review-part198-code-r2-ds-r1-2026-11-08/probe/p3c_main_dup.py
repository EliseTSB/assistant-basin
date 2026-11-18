"""The realistic pair: one run-level failure (no case_id) then one case-level failure (case 5).
Both come from run198.main() -> run_arm_a; both are terminal rows for the same key."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
import score198 as S

root = Path(tempfile.mkdtemp(prefix="main-dup3-"))
out = root / "out"
out.mkdir()
R.root_out_dir = lambda path: Path(path).expanduser().resolve()
# 1. A registered run that dies in preflight: main() writes a run-level failure (case_id None).
print("run-level failure rc =", R.main(["armA", "--method", "ROME", "--out-dir", str(out)]))
# 2. A later registered run whose case 5 fails: run_arm_a's own failure row (case_id 5).
R.append_jsonl(out / "armA-rome.jsonl", {"record_type": "failure", "run_mode": "registered",
    "run_id": "r2", "case_id": 5, "phase": "edit_or_score", "error": "RuntimeError: boom"})
# 3. A third registered run that also dies before doing any work (e.g. a preflight regression).
print("second run-level failure rc =", R.main(["armA", "--method", "ROME", "--out-dir", str(out)]))
for line in (out / "armA-rome.jsonl").read_text().splitlines():
    row = json.loads(line)
    print("   ", row.get("record_type"), "| case_id:", row.get("case_id"), "|", row.get("phase"))
try:
    print("registered_rows ->", len(S.registered_rows(out / "armA-rome.jsonl", "case", "case_id")), "row(s)")
except ValueError as exc:
    print("registered_rows raises:", exc)
