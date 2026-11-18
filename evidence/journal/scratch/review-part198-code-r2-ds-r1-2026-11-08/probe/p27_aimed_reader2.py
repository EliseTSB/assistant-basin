"""Same as p26 but with the reader's k=64 over a full-rank W0 (the runner's shape)."""
import sys
import torch
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R, stats198 as S
torch.manual_seed(0)
d, V = 16, 200
E = torch.randn(V, d, dtype=torch.float64)
gamma = torch.linspace(0.5, 2.0, d, dtype=torch.float64)
rome = torch.randn(d, d, dtype=torch.float64)
target = 7
delta, meta = R.make_aimed_delta(rome, E, gamma, target)
W0 = torch.randn(d, d, dtype=torch.float64)
pairs = S.read_pairs(W0 + delta, gamma, True, E, lambda ids: f"t{ids[0]}", k=64)
top = S.top_eligible(pairs)
print("aimed token", target, "| top eligible pair:", (top.j, top.sign, top.top, round(top.z, 3)) if top else None)
print("delta is exactly rank one:", int(torch.linalg.matrix_rank(delta)))
print("meta output_direction_norm:", meta["output_direction_norm"])
