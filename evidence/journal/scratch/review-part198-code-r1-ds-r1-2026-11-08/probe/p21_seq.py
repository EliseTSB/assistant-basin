"""Probe: 198.8 takes seq[-1] (the last row in FILE order) as 'after the tenth write'. After a resume the
runner rewrites every row, so the last row is the tenth write; but any later row for an earlier case (a
re-run) is taken as the final state. Measure the verdict flip."""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import score198 as SC
out = Path(tempfile.mkdtemp(prefix="p21-"))
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True, "crosscheck_pass": True}))
def case(cid, took=True, elig=True):
    return {"record_type": "case", "run_mode": "registered", "case_id": cid, "took": took,
            "already_satisfied": False, "name_eligible": elig,
            "localization": {"f1": {"localized": False}, "pcs": {"localized": False}, "t_word": {"localized": False}},
            "blind_reader": {"localized": False, "named_strict": False}, "oracle_reader": {"named_strict": False},
            "edited_layer_stats": {}, "aimed_positive_control": {"oracle_named_strict": True}, "restored_bit_exact": True}
for m in ("rome", "memit", "alphaedit"):
    (out / f"armA-{m}.jsonl").write_text("\n".join(json.dumps(r) for r in
        [case(i) for i in range(24)] + [{"record_type": "state_verification", "run_mode": "registered",
         "restored_bit_exact": True, "update_reproduced": True}]) + "\n")
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
seq = [dict(case(i), names_after_write={str(j): {"named_strict": True} for j in range(i + 1)}) for i in range(10)]
(out / "armB-seq-wsp.jsonl").write_text("\n".join(json.dumps(r) for r in seq) + "\n")
f = next(x for x in SC.score_from_dir(out)["forecasts"] if x["forecast"].startswith("198.8"))
print("clean file (last row = tenth write):", {k: f[k] for k in ("successes", "rate_denominator", "status")})
# now a re-run row for case 3 lands last in the file
seq2 = seq[:4] + [dict(case(3), names_after_write={str(j): {"named_strict": False} for j in range(4)})]
(out / "armB-seq-wsp.jsonl").write_text("\n".join(json.dumps(r) for r in seq2) + "\n")
f = next(x for x in SC.score_from_dir(out)["forecasts"] if x["forecast"].startswith("198.8"))
print("after a re-run of case 3 is appended last:", {k: f[k] for k in ("successes", "rate_denominator", "status")})
