"""Probe: aimed control, the case the design's rule is for. stat_198 reads a LayerNorm direction as
gamma*(u - mean u). The code builds gamma*E[t] centred. When does that matter?"""
import sys, torch
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
S = R.load_stats_module()
d = 8
E = torch.zeros(10, d, dtype=torch.float64)
E[3] = torch.tensor([3.0, -1.0, 0.5, 2.0, 0.0, 1.0, 0.7, -0.4])
gain = torch.tensor([0.5, 2.0, 1.0, 1.5, 0.25, 3.0, 1.0, 1.0], dtype=torch.float64)
rome = torch.zeros(d, d, dtype=torch.float64); rome[0, 0] = 4.0
delta, meta = R.make_aimed_delta(rome, E, gain, 3)
u = delta[0] / meta["spectral_norm"]
design = gain * (E[3] - E[3].mean()); design = design / design.norm()
print("code output direction :", [round(x, 4) for x in u.tolist()])
print("design direction      :", [round(x, 4) for x in design.tolist()])
print("cos =", round(float(u @ design), 6))
print("|u| =", round(float(u.norm()), 6))
# reading either direction through the SAME reader the control is scored with:
for label, vec in (("code", u), ("design", design)):
    W = torch.zeros(d, d, dtype=torch.float64); W[:, 0] = vec
    p = S.read_pairs(W, gain, True, E, lambda ids: "x", k=1)[0]
    print(f"  reader on {label:6s} direction: top token {int(p.top)} z {p.z:.3f}")
