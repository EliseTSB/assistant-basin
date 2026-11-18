"""m1: mass_gate_status(counted_batches) returns True for ANY positive count. The round-1
review's m1 text says the helper should express 'not evaluable with no counted batch' -- it
does -- but the scorer now calls it, so a single counted mass batch is enough for a verdict.
Design 198.1/198.3 denominator is 'batches 0 of 2' (two mass batches per method)."""
import sys
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R, score198 as S
batch = {"record_type": "batch", "take_gate_passed": True, "batch_id": "MEMIT:0",
         "facts": [{"took": True, "name_eligible": True} for _ in range(10)],
         "localization": {"f1": {"localized": False}, "pcs": {"localized": False}}}
for n in (0, 1, 2):
    res = S.mass_forecast("198.1 MEMIT mass", [batch] * n, "f1", True)
    print(f"{n} batch(es): counted={res['counted_batches']} status={res['status']} CP95={res['clopper_pearson_95']}")
print("mass_gate_status(1) =", R.mass_gate_status(1))
