"""Probe: Clopper-Pearson against exact reference values (scipy if present, else hand values)."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
cases = [(0, 24), (1, 24), (20, 24), (2, 24), (5, 10), (24, 24), (0, 1), (1, 1), (15, 24)]
ref = {(0, 24): (0.0, 0.14236), (1, 24): (0.00105, 0.20622), (20, 24): (0.64434, 0.95163),
       (2, 24): (0.01028, 0.26470), (5, 10): (0.18709, 0.81291), (24, 24): (0.85764, 1.0),
       (0, 1): (0.0, 0.975), (1, 1): (0.025, 1.0), (15, 24): (0.41437, 0.80434)}
for (k, n) in cases:
    got = R.cp_interval(k, n)
    lo, hi = ref[(k, n)]
    print(f"k={k:2d} n={n:2d} got=({got[0]:.5f},{got[1]:.5f}) ref=({lo:.5f},{hi:.5f}) "
          f"err=({abs(got[0]-lo):.1e},{abs(got[1]-hi):.1e})")
try:
    from scipy.stats import beta
    worst = 0
    for n in (1, 5, 10, 24):
        for k in range(n + 1):
            lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
            hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
            g = R.cp_interval(k, n)
            worst = max(worst, abs(g[0] - lo), abs(g[1] - hi))
    print("scipy check: worst absolute error over n in {1,5,10,24}:", f"{worst:.2e}")
except ImportError:
    print("scipy not installed")
print("mcnemar(0,5):", R.exact_mcnemar(0, 5), " mcnemar(3,3):", R.exact_mcnemar(3, 3))
