"""Drive the REAL run198.main(): a startup failure, a fixed preflight, then a registered run.
main() appends a 'startup_or_run' failure row with no case_id/batch_id, so the stream is
unscorable by score198.registered_rows if that failure is not the first row."""
import json, os, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
import score198 as S

root = Path(tempfile.mkdtemp(prefix="main-dup-"))
out = root / "out"
R.root_out_dir = lambda path: Path(path).expanduser().resolve()  # only the containment guard is bypassed
# 1. First invocation: preflight fails (no crosscheck artifact in this temp out dir).
rc1 = R.main(["armA", "--method", "ROME", "--out-dir", str(out)])
print("first main() rc =", rc1)
# 2. Second invocation: the same command, now succeeding. Stub the model/editor work.
R.preflight = lambda out_dir: {"pass": True}
R.load_gpt2_model = lambda: (_ for _ in ()).throw(RuntimeError("stub: stop before loading weights"))
rc2 = R.main(["armA", "--method", "ROME", "--out-dir", str(out)])
print("second main() rc =", rc2)
log = out / "armA-rome.jsonl"
for line in log.read_text().splitlines():
    row = json.loads(line)
    print("   ", row.get("record_type"), "|", row.get("phase"), "|", row.get("error", "")[:40])
# 3. The scorer on the same stream, with one synthetic case row for case 1 appended by hand.
with log.open("a") as f:
    f.write(json.dumps({"record_type": "case", "run_mode": "registered", "case_id": 1,
                        "took": True, "restored_bit_exact": True}) + "\n")
try:
    print("registered_rows ->", S.registered_rows(log, "case", "case_id"))
except ValueError as exc:
    print("registered_rows raises:", exc)
