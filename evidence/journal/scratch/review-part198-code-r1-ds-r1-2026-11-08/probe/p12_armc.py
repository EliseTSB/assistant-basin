"""Probe: Arm C selection. The design says 3 calibration + 5 evaluation + 2 development hosts (ten models);
run198.py:1395 and :1430 only ever iterate ARM_C_MODELS[:3] and [3:] -- count them."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
print("ARM_C_MODELS:", [(m["id"], m["role"]) for m in R.ARM_C_MODELS])
print("first loop  [:3]  ->", [m["id"] for m in R.ARM_C_MODELS[:3]])
print("second loop [3:]  ->", [m["id"] for m in R.ARM_C_MODELS[3:]])
print("calibration roles in first loop:", [m["role"] for m in R.ARM_C_MODELS[:3]])
print("evaluation/host roles in second loop:", [m["role"] for m in R.ARM_C_MODELS[3:]])
# the tstar calibrations are the first three by position, not by role:
print("T* would be built from:", [m["id"] for m in R.ARM_C_MODELS[:3]])
