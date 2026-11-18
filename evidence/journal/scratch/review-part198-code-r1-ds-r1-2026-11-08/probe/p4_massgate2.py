"""Probe: gate-failing batch localizes, gate-passing batch does not -> design says PASS (left out); code says FAIL."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import score198 as SC
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True,
                                           "crosscheck_pass": True}))
def batch(idx, n_took, localized):
    facts = [{"took": i < n_took, "already_satisfied": False, "name_eligible": True} for i in range(10)]
    return {"record_type": "batch", "run_mode": "registered", "method": "MEMIT", "batch_id": f"MEMIT:{idx}",
            "take_gate_passed": n_took >= 8, "facts": facts, "restored_bit_exact": True,
            "localization": {"f1": {"localized": localized}, "pcs": {"localized": localized}}}
rows = [batch(0, 10, False), batch(1, 5, True)]
(out / "armA-mass-memit.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
r = SC.score_from_dir(out)
for f in r["forecasts"]:
    if f["forecast"] in ("198.1 MEMIT mass", "198.3 MEMIT mass"):
        print(f["forecast"], "counted_batches=", f["counted_batches"], "successes=", f["successes"], "status=", f["status"])
print("gate reasons:", r["gate"]["reasons"])
