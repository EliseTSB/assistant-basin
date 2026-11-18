"""Probe: when S1 does not pass, does score198 suppress the update-top 198.4 forecast too?"""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import score198 as SC
out = Path(tempfile.mkdtemp(prefix="p32-"))
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True, "crosscheck_pass": True}))
def case(cid):
    return {"record_type": "case", "run_mode": "registered", "case_id": cid, "took": True,
            "already_satisfied": False, "name_eligible": True,
            "localization": {"f1": {"localized": False}, "pcs": {"localized": False}, "t_word": {"localized": False}},
            "blind_reader": {"localized": False, "named_strict": False},
            "oracle_reader": {"named_strict": True, "named_loose": True},
            "edited_layer_stats": {"17": {"update_top_names_strict": True, "update_top_names_loose": True}},
            "aimed_positive_control": {"oracle_named_strict": False},   # S1 will fail: 0 of 24
            "restored_bit_exact": True}
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
res = SC.score_from_dir(out)
print("S1:", res["sensitivity_checks"]["S1"]["status"])
for f in res["forecasts"]:
    if f["forecast"].startswith("198.4 ROME"):
        print(f"{f['forecast']}: successes={f['successes']}/{f['rate_denominator']} status={f['status']}")
