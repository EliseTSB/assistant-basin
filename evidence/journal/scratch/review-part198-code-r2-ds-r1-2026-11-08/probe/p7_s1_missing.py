"""B4/M5: a took, name-eligible ROME record whose aimed control was never built
(make_aimed_delta returned None -> record['aimed_positive_control'] = {'reason': ...}).
Does S1 count it as a failure, or does the scorer crash?"""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import score198 as S

out = Path(tempfile.mkdtemp(prefix="s1-"))
(out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True,
                                           "crosscheck_pass": True}))
def case(i, control):
    return {"record_type": "case", "run_mode": "registered", "case_id": i, "took": True,
            "name_eligible": True, "restored_bit_exact": True,
            "localization": {"f1": {"localized": False}, "pcs": {"localized": False},
                             "t_word": {"localized": False}},
            "oracle_reader": {"named_strict": False}, "edited_layer_stats": {},
            "aimed_positive_control": control}
rows = [case(i, {"oracle_named_strict": True, "localization": {"localized": True}}) for i in range(23)]
rows.append(case(23, {"reason": "zero gamma-scaled centred unembedding row"}))  # runner's own shape
(out / "armA-rome.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
try:
    scored = S.score_from_dir(out)
    print("S1:", scored["sensitivity_checks"]["S1"])
    print("198.5:", scored["descriptive"]["198.5_aimed_control_localization"])
except Exception as exc:
    print("score_from_dir raises:", type(exc).__name__, exc)
