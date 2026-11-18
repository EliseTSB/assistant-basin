"""Probe: aimed control with uniform gain (GPT-2 XL ln_f is near-uniform, so the defect is small there) and
with non-uniform gain. Report cos(code direction, design direction) and the token each reads as."""
import sys, torch
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
S = R.load_stats_module()
d = 8
E = torch.zeros(10, d, dtype=torch.float64)
E[3] = torch.tensor([3.0, -1.0, 0.5, 2.0, 0.0, 1.0, 0.7, -0.4])
rome = torch.zeros(d, d, dtype=torch.float64); rome[0, 0] = 4.0
for label, gain in (("uniform gain (ln_f-like)", torch.ones(d, dtype=torch.float64)),
                    ("non-uniform gain", torch.tensor([0.5, 2.0, 1.0, 1.5, 0.25, 3.0, 1.0, 1.0], dtype=torch.float64))):
    delta, meta = R.make_aimed_delta(rome, E, gain, 3)
    u = delta[0] / meta["spectral_norm"]
    design = gain * (E[3] - E[3].mean()); design = design / design.norm()
    W = torch.zeros(d, d, dtype=torch.float64); W[:, 0] = u
    p = S.read_pairs(W, gain, True, E, lambda ids: "x", k=1)[0]
    print(f"{label:26s} cos={float(u @ design):+.4f}  |u|={float(u.norm()):.4f}  reader top token={int(p.top)}")
