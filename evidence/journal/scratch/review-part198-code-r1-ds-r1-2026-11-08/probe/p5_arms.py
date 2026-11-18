"""Probe: score198 verdicts on a complete, gate-passing synthetic run, with a deliberately bad mass batch."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import score198 as SC
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True, "crosscheck_pass": True}))
def case(cid, took=True, elig=True):
    return {"record_type": "case", "run_mode": "registered", "case_id": cid, "took": took,
            "already_satisfied": False, "name_eligible": elig,
            "localization": {"f1": {"localized": False}, "pcs": {"localized": False},
                             "t_word": {"localized": False}},
            "blind_reader": {"localized": False, "named_strict": False},
            "oracle_reader": {"named_strict": False, "named_loose": False},
            "edited_layer_stats": {}, "aimed_positive_control": {"oracle_named_strict": True},
            "restored_bit_exact": True}
for m in ("rome", "memit", "alphaedit"):
    rows = [case(i) for i in range(24)]
    rows.append({"record_type": "state_verification", "run_mode": "registered", "case_id": 0,
                 "restored_bit_exact": True, "update_reproduced": True})
    (out / f"armA-{m}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
for m in ("memit", "alphaedit"):
    rows = [{"record_type": "batch", "run_mode": "registered", "method": m.upper(), "batch_id": f"{m}:0",
             "take_gate_passed": True, "restored_bit_exact": True,
             "facts": [{"took": True, "already_satisfied": False, "name_eligible": True} for _ in range(10)],
             "localization": {"f1": {"localized": True}, "pcs": {"localized": True}}},
            {"record_type": "batch", "run_mode": "registered", "method": m.upper(), "batch_id": f"{m}:1",
             "take_gate_passed": False, "restored_bit_exact": True,
             "facts": [{"took": i < 5, "already_satisfied": False, "name_eligible": True} for i in range(10)],
             "localization": {"f1": {"localized": True}, "pcs": {"localized": True}}}]
    (out / f"armA-mass-{m}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
b = []
for i in range(24):
    r = case(i); r["undo_zero_exact"] = True
    r["random_control"] = {"localization": {"t_word": {"localized": False}}}
    b.append(r)
(out / "armB-wsp.jsonl").write_text("\n".join(json.dumps(r) for r in b) + "\n")
seq = [dict(case(i), names_after_write={str(i): {"named_strict": False}}) for i in range(10)]
(out / "armB-seq-wsp.jsonl").write_text("\n".join(json.dumps(r) for r in seq) + "\n")
res = SC.score_from_dir(out)
print("GATE", res["gate"]["passed"], res["gate"]["reasons"])
for f in res["forecasts"]:
    print(f"{f['forecast']}: denom={f['rate_denominator']} succ={f['successes']} {f['status']}")
