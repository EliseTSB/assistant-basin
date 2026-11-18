"""B3 with the REAL reader: build the aimed delta, then score it exactly the way the runner
scores the control (stats198.read_pairs(W0 + delta, gain, centre=True, E)) and compare the
reader's top token with the token the direction was aimed at."""
import sys
import torch
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R, stats198 as S
torch.manual_seed(0)
d, V = 6, 40
E = torch.randn(V, d, dtype=torch.float64)
gamma = torch.linspace(0.5, 2.0, d, dtype=torch.float64)
rome = torch.randn(d, d, dtype=torch.float64)
target = 7
delta, meta = R.make_aimed_delta(rome, E, gamma, target)
W0 = torch.zeros(d, d, dtype=torch.float64)
pairs = S.read_pairs(W0 + delta, gamma, True, E, lambda ids: f"t{ids[0]}", k=1)
print("aimed token", target, "| reader pairs:", [(p.sign, p.top, round(p.z, 3)) for p in pairs])
top = S.top_eligible(pairs)
print("top eligible pair token:", top.top if top else None, "| meta:", meta)
# The old (65c2be6) formula, for contrast.
old = gamma * E[target]; old = old - old.mean(); old = old / old.norm()
delta_old = float(torch.linalg.svdvals(rome)[0]) * torch.outer(old, torch.linalg.svd(rome).Vh[0])
pairs_old = S.read_pairs(W0 + delta_old, gamma, True, E, lambda ids: f"t{ids[0]}", k=1)
print("old formula reader pairs:", [(p.sign, p.top, round(p.z, 3)) for p in pairs_old])
