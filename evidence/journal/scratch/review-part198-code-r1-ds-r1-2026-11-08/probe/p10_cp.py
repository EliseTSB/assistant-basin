"""Probe: Clopper-Pearson lower bound vs scipy exact, for every k at n=24 and n=10."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
from scipy.stats import beta
for n in (10, 24):
    bad = []
    for k in range(n + 1):
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
        g = R.cp_interval(k, n)
        if abs(g[0] - lo) > 1e-3 or abs(g[1] - hi) > 1e-3:
            bad.append((k, round(g[0], 4), round(lo, 4), round(g[1], 4), round(hi, 4)))
    print(f"n={n}: {len(bad)} of {n+1} intervals wrong; rows (k, got_lo, exact_lo, got_hi, exact_hi):")
    for row in bad:
        print("   ", row)
# the loop's own trace at k=20, n=24
import math
n, k, alpha = 24, 20, 0.05
def pmf(i, p): return math.comb(n, i) * p**i * (1-p)**(n-i)
def ge(kk, p): return math.fsum(pmf(i, p) for i in range(kk, n+1))
a, b = 0.0, 1.0
for _ in range(80):
    mid = (a + b) / 2
    if ge(k, mid) < alpha / 2: b = mid
    else: a = mid
print("k=20 n=24: ge(20, 0.5) =", ge(20, 0.5), "-> bisection converges to", (a+b)/2, "(exact lower bound 0.6443)")
