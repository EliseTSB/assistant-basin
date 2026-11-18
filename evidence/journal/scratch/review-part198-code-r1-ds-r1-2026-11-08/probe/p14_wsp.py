"""Probe: Arm B 'took'. Design: no taught question answers the object exactly before, and all three do after
(free_exact 3/3). The code uses free_exact for 'after' and never checks that the after-answers are exact."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
print("wsp_took(0,3,3) =", R.wsp_took(0, 3, 3), "   # before 0/3 exact, wsp free_exact 3/3")
print("wsp_took(0,2,3) =", R.wsp_took(0, 2, 3), "   # wsp free_exact 2/3 but the runner's own after_answers were 3/3")
print("wsp_took(3,3,3) =", R.wsp_took(3, 3, 3), "   # base already answers all three (design: already satisfied)")
print("wsp_took(1,3,3) =", R.wsp_took(1, 3, 3), "   # base answers one of three exactly")
# the record carries after_exact, but took is computed from free_exact only:
import inspect
src = inspect.getsource(R.run_arm_b)
print("run_arm_b computes took from free_exact, not after_exact:", "wsp_took(before_exact, free_exact" in src)
print("after_exact is recorded:", '"after_exact": after_exact' in src)
