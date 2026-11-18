"""Probe: a batch that localizes but fails the 8/10 take gate must be left out, not fail the forecast."""
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
# batch 0: 10/10 took, not localized (a pass). batch 1: 5/10 took (gate failed), localized.
(out / "armA-mass-memit.jsonl").write_text("\n".join(json.dumps(batch(0, 10, False)) for _ in [0]) + "\n"
                                           + json.dumps(batch(1, 5, True)) + "\n")
r = SC.score_from_dir(out)
for f in r["forecasts"]:
    if f["forecast"] == "198.1 MEMIT mass":
        print("198.1 MEMIT mass:", {k: f[k] for k in ("took", "counted_batches", "rate_denominator", "successes", "status")})
