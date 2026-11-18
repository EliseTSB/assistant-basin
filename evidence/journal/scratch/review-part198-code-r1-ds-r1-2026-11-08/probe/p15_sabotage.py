"""Probe: would the suites stay green under sabotage? Import run198 with patched helpers."""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R, score198 as SC

# S1: mass gate minimum 8 -> 7
R.mass_gate.__defaults__ = (7,)
print("S1 mass_gate([True]*7+[False]*3) after sabotage:", R.mass_gate([True]*7+[False]*3),
      "(suite asserts this is False)")
R.mass_gate.__defaults__ = (8,)

# S2: a blind reader that peeks at the edited set. stats198.blind_reader takes (layer_pairs) only,
#     so it cannot peek -- but the *runner* chooses which layers to pass in. Show that the runner passes all layers:
import inspect
src = inspect.getsource(R.run_arm_b)
print("S2 run_arm_b blind reader input:", "S.blind_reader(allpairs)" in src,
      "; allpairs built from base layers:", "pair_by_layer(base, current, S)" in src)

# S3: score198's S1 check accepts a control with no localization key at all (flag() returns False silently)
row = {"name_eligible": True, "aimed_positive_control": {"reason": "zero centred, gamma-scaled unembedding row"}}
print("S3 S1 on a control that produced no statistics:", SC.flag(row, "aimed_positive_control", "oracle_named_strict"))

# S4: metric_result treats a missing 'took' key as not-took and a missing name_eligible as not eligible,
#     so a runner that stopped writing those keys would silently shrink every denominator to 0 -> NOT EVALUABLE
rows = [{"case_id": i} for i in range(24)]
m = SC.metric_result("x", rows, lambda r: True, "took", -20, True)
print("S4 rows with no took key ->", {k: m[k] for k in ("took", "rate_denominator", "status")})
