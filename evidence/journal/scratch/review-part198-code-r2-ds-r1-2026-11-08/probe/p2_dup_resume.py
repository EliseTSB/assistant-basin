"""Do the runner's own resume rules ever produce two terminal rows for one key?
Drives the REAL run198.existing_keys and score198.registered_rows."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
import score198 as S

def w(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

td = Path(tempfile.mkdtemp(prefix="dup-"))
# (a) Arm A: a 'failure' row for case 5, then the case succeeds in a later process.
#     run_arm_a's `prior` (existing_keys) suppresses re-running case 5, so this pair is only
#     reachable if a row is appended outside `prior` (e.g. the post-loop state_verification
#     failure path) -- but the SCORER must still be able to read a stream where a failure
#     predates the case row, because gate_check() explicitly expects failure rows.
p = td / "armA-rome.jsonl"
w(p, [{"record_type": "failure", "run_mode": "registered", "case_id": 5, "phase": "edit_or_score"},
      {"record_type": "case", "run_mode": "registered", "case_id": 5, "took": True,
       "restored_bit_exact": True}])
print("(a) existing_keys(armA) ->", sorted(R.existing_keys(p, "case_id")))
try:
    print("(a) registered_rows ->", S.registered_rows(p, "case", "case_id"))
except ValueError as exc:
    print("(a) registered_rows raises:", exc)

# (b) Arm C: the same model fails once, then succeeds on the retry (the M5 retry path).
p = td / "armC.jsonl"
w(p, [{"record_type": "failure", "run_mode": "registered", "model_id": "EleutherAI/pythia-410m",
       "phase": "calibration", "error": "OOM"},
      {"record_type": "armC_model", "run_mode": "registered", "model_id": "EleutherAI/pythia-410m",
       "max_t_word": 6.73}])
print("(b) arm_c_successful_keys ->", R.arm_c_successful_keys(p))

# (c) Arm B-seq: a failure row, then a resume that replays the prefix and appends the case row.
p = td / "armB-seq-wsp.jsonl"
w(p, [{"record_type": "failure", "run_mode": "registered", "case_id": 0, "phase": "sequence_write_or_score"},
      {"record_type": "case", "run_mode": "registered", "case_id": 0, "n_written_so_far": 1}])
try:
    print("(c) registered_rows ->", S.registered_rows(p, "case", "case_id"))
except ValueError as exc:
    print("(c) registered_rows raises:", exc)

# (d) A legitimate resume of Arm B-seq: every case row is rewritten by the replay with a new
#     run_id/timestamp and identical payload. Does the scorer accept it?
p = td / "seq-identical.jsonl"
row = {"record_type": "case", "run_mode": "registered", "run_id": "r1", "timestamp": "t1",
       "case_id": 0, "n_written_so_far": 1, "names_after_write": {"0": {"named_strict": True}}}
w(p, [row, dict(row, run_id="r2", timestamp="t2")])
print("(d) identical replay accepted ->", len(S.registered_rows(p, "case", "case_id")))

# (e) An Arm B-seq replay whose recomputed statistics differ in the last bit (a re-run on a
#     different thread count / machine) is a *legitimate* second process, not a conflict.
p = td / "seq-drift.jsonl"
w(p, [row, dict(row, run_id="r2", timestamp="t2", n_written_so_far=1,
                names_after_write={"0": {"named_strict": False}})])
try:
    print("(e) registered_rows ->", S.registered_rows(p, "case", "case_id"))
except ValueError as exc:
    print("(e) registered_rows raises:", exc)
