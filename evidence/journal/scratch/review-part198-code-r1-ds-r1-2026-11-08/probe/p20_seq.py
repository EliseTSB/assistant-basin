"""Probe: Arm B sequence. 198.8 reads seq[-1]['names_after_write'] -- the last row in FILE order, which is not
the last write if the file was resumed or if a failure row was appended last."""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R, score198 as SC
out = Path(tempfile.mkdtemp(prefix="p20-"))
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True, "crosscheck_pass": True}))
def case(cid, took=True, elig=True):
    return {"record_type": "case", "run_mode": "registered", "case_id": cid, "took": took,
            "already_satisfied": False, "name_eligible": elig,
            "localization": {"f1": {"localized": False}, "pcs": {"localized": False}, "t_word": {"localized": False}},
            "blind_reader": {"localized": False, "named_strict": False}, "oracle_reader": {"named_strict": False},
            "edited_layer_stats": {}, "aimed_positive_control": {"oracle_named_strict": True}, "restored_bit_exact": True}
for m in ("rome", "memit", "alphaedit"):
    rows = [case(i) for i in range(24)] + [{"record_type": "state_verification", "run_mode": "registered",
             "restored_bit_exact": True, "update_reproduced": True}]
    (out / f"armA-{m}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
for m in ("memit", "alphaedit"):
    (out / f"armA-mass-{m}.jsonl").write_text(json.dumps({"record_type": "batch", "run_mode": "registered",
        "batch_id": f"{m}:0", "take_gate_passed": True, "restored_bit_exact": True,
        "facts": [{"took": True, "already_satisfied": False, "name_eligible": True}] * 10,
        "localization": {"f1": {"localized": False}, "pcs": {"localized": False}}}) + "\n")
b = []
for i in range(24):
    r = case(i); r["undo_zero_exact"] = True; r["random_control"] = {"localization": {"t_word": {"localized": False}}}
    b.append(r)
(out / "armB-wsp.jsonl").write_text("\n".join(json.dumps(r) for r in b) + "\n")
# sequence: all ten written, each row's names_after_write says "named" for everything written SO FAR.
seq = []
for i in range(10):
    seq.append(dict(case(i), names_after_write={str(j): {"named_strict": True} for j in range(i + 1)}))
# then the file gets a duplicate row for an earlier case (a re-run after a crash), appended last
seq.append(dict(case(3), names_after_write={str(j): {"named_strict": False} for j in range(4)}))
seq = seq[:4] + [seq[-1]]   # re-run of case 3 is now the LAST line
(out / "armB-seq-wsp.jsonl").write_text("\n".join(json.dumps(r) for r in seq) + "\n")
res = SC.score_from_dir(out)
f = next(x for x in res["forecasts"] if x["forecast"].startswith("198.8"))
print("198.8 with a duplicate earlier case appended last:", {k: f[k] for k in ("successes", "rate_denominator", "status")})
print("  (the last write, case 9, named all ten; the file's last row is a re-run of case 3)")
