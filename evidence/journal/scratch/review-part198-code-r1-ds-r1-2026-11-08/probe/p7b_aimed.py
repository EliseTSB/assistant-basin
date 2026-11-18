"""Probe: aimed control with a NON-uniform gamma (the GPT-2 XL ln_f case is near-uniform, but the rule must hold)."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import torch
import run198 as R
S = R.load_stats_module()
d = 6
E = torch.zeros(10, d, dtype=torch.float64)
E[3] = torch.tensor([3.0, -1.0, 0.5, 2.0, 0.0, 1.0])
gain = torch.tensor([0.5, 2.0, 1.0, 1.5, 0.25, 3.0], dtype=torch.float64)
rome = torch.zeros(6, 6, dtype=torch.float64); rome[0, 0] = 4.0; rome[1, 1] = 1.0
delta, meta = R.make_aimed_delta(rome, E, gain, 3)
u = delta[0] / meta["spectral_norm"]
want = gain * (E[3] - E[3].mean()); want = want / want.norm()
code = gain * E[3]; code = code - code.mean(); code = code / code.norm()
print("code output direction     :", [round(x, 4) for x in u.tolist()])
print("design (gamma*(u-mean u)) :", [round(x, 4) for x in want.tolist()])
print("cos(code, design) =", round(float(u @ want), 6))
for label, vec in (("code", u), ("design", want)):
    W = torch.zeros(6, 6, dtype=torch.float64); W[:, 0] = vec
    pairs = S.read_pairs(W, gain, True, E, lambda ids: "x", k=1)
    print(f"  {label:6s} top token {int(pairs[0].top)} z {pairs[0].z:.4f}")
# and with uniform gamma the two coincide (so GPT-2 XL's ln_f, gamma~1, hides it):
gain1 = torch.ones(d, dtype=torch.float64)
delta1, meta1 = R.make_aimed_delta(rome, E, gain1, 3)
u1 = delta1[0] / meta1["spectral_norm"]
w1 = E[3] - E[3].mean(); w1 = w1 / w1.norm()
print("uniform gamma: cos(code, design) =", round(float(u1 @ w1), 6))
