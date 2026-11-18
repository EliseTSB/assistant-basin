"""Probe: does score198 read the localization shape run198 actually writes for a mass batch?"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import score198 as SC
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True,
                                           "crosscheck_pass": True}))
facts = [{"took": True, "already_satisfied": False, "name_eligible": True} for _ in range(10)]
row = {"record_type": "batch", "run_mode": "registered", "method": "MEMIT", "batch_id": "MEMIT:0",
       "take_gate_passed": True, "facts": facts, "restored_bit_exact": True,
       # exactly the shape run198.py:988 writes (R.localization -> {"f1": {...}, ...})
       "localization": {"f1": {"localized": True, "first_layer": 17, "tie_for_first": False},
                        "pcs": {"localized": True, "first_layer": 17, "tie_for_first": False}}}
(out / "armA-mass-memit.jsonl").write_text(json.dumps(row) + "\n")
r = SC.score_from_dir(out)
for f in r["forecasts"]:
    if "mass" in f["forecast"]:
        print(f["forecast"], "counted_batches=", f["counted_batches"], "successes=", f["successes"],
              "status=", f["status"])
print("row had localization.f1.localized =", row["localization"]["f1"]["localized"])
