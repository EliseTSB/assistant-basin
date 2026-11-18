"""Arm B: can a legitimate resume produce two terminal rows for one case?
run_arm_b appends the case row, then the random-control row, then the undo check; a crash between
the case row and the undo check leaves a case row whose undo_zero_exact is missing. On resume,
existing_keys() sees the case row and skips the case, so the undo check is never redone -- the
stream is gate-failed forever. Show the resume decision and the gate outcome."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R, score198 as S
td = Path(tempfile.mkdtemp(prefix="armb-"))
log = td / "armB-wsp.jsonl"
R.append_jsonl(log, {"record_type": "case", "run_mode": "registered", "case_id": 3, "took": True,
                     "name_eligible": True})   # undo_zero_exact missing: crash before the undo check
print("resume prior:", sorted(R.existing_keys(log, "case_id")), "-> case 3 will be skipped")
(out := td).joinpath("gate.json").write_text(json.dumps({"fixtures_pass": True,
    "synthetic_tests_pass": True, "crosscheck_pass": True}))
gate = S.gate_check(td, {"armB": [{"record_type": "case", "case_id": 3, "took": True}],
                         "all-armB": [{"record_type": "case", "case_id": 3, "took": True}]})
print("gate reasons:", [r for r in gate["reasons"] if "undo" in r or "Arm B" in r])
