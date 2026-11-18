"""Q1: is the fingerprint compared after EVERY record including failure paths, and is it taken
before any edit? Static check of the call sites in the REAL run198.py, plus the sabotage that
the suite misses (failure path and snapshot ordering)."""
import inspect, re, sys
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
for fn in (R.run_arm_a, R.run_arm_a_mass):
    src = inspect.getsource(fn)
    print(fn.__name__, "snapshot:", src.index("state_snapshot = model_state_snapshot(model)"),
          "| first apply:", src.index("apply(model"), "| compares:", src.count("model_matches_snapshot"),
          "| failure rows:", src.count('"record_type": "failure"'))
    for m in re.finditer(r"model_matches_snapshot", src):
        line = src[:m.start()].count("\n") + 1
        print("   compare at relative line", line, ":", src.splitlines()[line - 1].strip())
print()
print("state_verification block (after the loop) compares:", "model_matches_snapshot" in inspect.getsource(R.run_arm_a).split("first_id = int(case_ids[0])")[1])
print("state_verification exception path compares:",
      "model_matches_snapshot" in inspect.getsource(R.run_arm_a).split("except Exception as exc:\n            try:\n                with torch.no_grad():\n                    for pname, w0 in verify_orig.items()")[-1])
