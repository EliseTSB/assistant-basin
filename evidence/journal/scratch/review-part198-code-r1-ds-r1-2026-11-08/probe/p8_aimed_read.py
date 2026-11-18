"""Probe: is the aimed control's named token independent of the target? Read both candidate directions
through the reader on a covariance-free rank-one matrix whose output direction is the vector."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import torch
import run198 as R
S = R.load_stats_module()
torch.manual_seed(0)
d = 8
E = torch.randn(20, d, dtype=torch.float64)
gain = torch.rand(d, dtype=torch.float64) + 0.5
for target in (3, 7):
    row = E[target]
    a = gain * row; a = a - a.mean(); a = a / a.norm()          # code: gamma*row then centre
    b = gain * (row - row.mean()); b = b / b.norm()              # design: gamma*(row - mean)
    out = []
    for vec in (a, b):
        W = torch.zeros(d, d, dtype=torch.float64); W[:, 0] = vec
        p = S.read_pairs(W, gain, True, E, lambda ids: "x", k=1)[0]
        out.append((int(p.top), round(p.z, 3)))
    print(f"target {target}: code direction -> top {out[0]}, design direction -> top {out[1]}")
