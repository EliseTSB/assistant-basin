"""Probe: mass gate + S1 denominator + which rows the scorer actually reads."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import score198 as SC

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True,
                                           "crosscheck_pass": True}))

def batch(method, idx, n_took, local_f1, local_pcs):
    facts = [{"took": i < n_took, "already_satisfied": False, "name_eligible": True} for i in range(10)]
    return {"record_type": "batch", "run_mode": "registered", "method": method, "batch_id": f"{method}:{idx}",
            "take_gate_passed": n_took >= 8, "facts": facts, "restored_bit_exact": True,
            "localization": {"f1": {"localized": local_f1}, "pcs": {"localized": local_pcs}}}

# 1) A mass batch that does NOT reach 8/10 but localizes F1 and PCS: design says left out of 198.1/198.3.
(out / "armA-mass-memit.jsonl").write_text(json.dumps(batch("MEMIT", 0, 5, True, True)) + "\n")
r = SC.score_from_dir(out)
for f in r["forecasts"]:
    if "mass" in f["forecast"]:
        print("MASS-1", f["forecast"], "counted_batches=", f["counted_batches"],
              "rate_denominator=", f["rate_denominator"], "successes=", f["successes"], "status=", f["status"])
print("MASS-1 gate passed =", r["gate"]["passed"])

# 2) The runner never writes a top-level 'localized'; it writes localization[stat]['localized'].
b = batch("MEMIT", 0, 10, True, True)
b["localization"] = {"f1": {"localized": True, "first_layer": 17, "tie_for_first": False},
                     "pcs": {"localized": True, "first_layer": 17, "tie_for_first": False}}
b["localization"]["localized"] = None   # what run198 actually produces: no such key at all
(out / "armA-mass-memit.jsonl").write_text(json.dumps(b) + "\n")
r = SC.score_from_dir(out)
for f in r["forecasts"]:
    if "mass" in f["forecast"]:
        print("MASS-2", f["forecast"], "successes=", f["successes"], "status=", f["status"])

# 3) S1 denominator: ROME record that did NOT take but is name-eligible and has a passing control.
rome = {"record_type": "case", "run_mode": "registered", "case_id": 1, "took": False,
        "already_satisfied": True, "name_eligible": True,
        "localization": {"f1": {"localized": False}, "pcs": {"localized": False}},
        "aimed_positive_control": {"oracle_named_strict": True}}
(out / "armA-rome.jsonl").write_text(json.dumps(rome) + "\n")
r = SC.score_from_dir(out)
print("S1", r["sensitivity_checks"]["S1"])
