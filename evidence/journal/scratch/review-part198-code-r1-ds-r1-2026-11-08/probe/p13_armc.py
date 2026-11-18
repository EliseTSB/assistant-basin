"""Probe: Arm C. What does --limit do to the calibration loop, and what does a rerun do to T*?"""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
# Re-implement the loop's control flow exactly as written (run198.py:1394-1429) with a fake loader.
def simulate(limit, prior, results):
    selected_count = 0
    for item in R.ARM_C_MODELS[:3]:
        model_id = item["id"]
        if model_id in prior:
            continue
        if limit is not None and selected_count >= limit:
            break
        prior.add(model_id); selected_count += 1
    current = {m: results[m] for m in results if m in prior}
    if not all(m["id"] in current for m in R.ARM_C_MODELS[:3]):
        return "return 0 (no tstar written)"
    cal = {m["id"]: current[m["id"]] for m in R.ARM_C_MODELS[:3]}
    return f"tstar = max({cal}) = {max(cal.values())}"
print("smoke --limit 1, nothing prior   :", simulate(1, set(), {}))
print("smoke --limit 2, nothing prior   :", simulate(2, set(), {}))
print("registered, all three prior      :", simulate(None, {m["id"] for m in R.ARM_C_MODELS[:3]},
      {"EleutherAI/pythia-410m": 6.7, "Qwen/Qwen3-0.6B": 10.3, "allenai/OLMo-2-0425-1B": 12.1}))
# and the smoke rows it writes are run_mode='smoke', which the tstar rebuild ignores:
import inspect
src = inspect.getsource(R.run_arm_c)
print("run_arm_c reads only registered rows for T*:", 'run_mode", "registered"' in src)
