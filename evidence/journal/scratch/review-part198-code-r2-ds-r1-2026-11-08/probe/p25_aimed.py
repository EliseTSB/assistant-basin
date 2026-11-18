"""B3: the fold's aimed direction vs the design's gamma*(row-mean) and the reader's contract.
Compare aimed_output_direction() with the old formula on a non-uniform gamma, and with the
direction stats198.read_pairs would score."""
import sys
import torch
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
E = torch.tensor([[1.0, -1.0, 0.5, -0.5], [2.0, 0.0, -1.0, -1.0]], dtype=torch.float64)
for label, gamma in (("uniform", torch.full((4,), 1.0, dtype=torch.float64)),
                     ("non-uniform", torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64))):
    new = R.aimed_output_direction(E, gamma, 0)
    old = gamma * E[0]; old = old - old.mean(); old = old / old.norm()
    want = gamma * (E[0] - E[0].mean()); want = want / want.norm()
    print(f"{label:12s} |new|={float(new.norm()):.6f} cos(new,want)={float(new @ want):.6f} "
          f"cos(old,want)={float(old @ want):.6f}")
