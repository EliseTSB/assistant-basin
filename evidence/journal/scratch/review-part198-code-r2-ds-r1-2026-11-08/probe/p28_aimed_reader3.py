"""Same as p27 but with 't27'-style content tokens (is_content rejects digits/symbols)."""
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
W0 = torch.randn(d, d, dtype=torch.float64)
decode = lambda ids: "word" + "abcdefghij"[ids[0] % 10] + "qrstuvwxyz"[ids[0] // 10 % 10]
for target in (7, 33, 120):
    delta, meta = R.make_aimed_delta(rome, E, gamma, target)
    pairs = S.read_pairs(W0 + delta, gamma, True, E, decode, k=64)
    top = S.top_eligible(pairs)
    old = gamma * E[target]; old = old - old.mean(); old = old / old.norm()
    delta_old = float(torch.linalg.svdvals(rome)[0]) * torch.outer(old, torch.linalg.svd(rome).Vh[0])
    pairs_old = S.read_pairs(W0 + delta_old, gamma, True, E, decode, k=64)
    top_old = S.top_eligible(pairs_old)
    print(f"aimed {target}: new top={top.top if top else None} z={top.z if top else 0:.3f} | "
          f"old-formula top={top_old.top if top_old else None} z={top_old.z if top_old else 0:.3f}")
