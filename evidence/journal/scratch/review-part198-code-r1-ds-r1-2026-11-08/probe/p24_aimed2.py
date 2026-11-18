"""Probe: aimed control, realistic shape. Make the ROME update's right singular vector OUTSIDE the host's
row space (as the design's matched geometry says) so W0 + delta really has the aimed output direction,
then check which output direction the code writes vs the design's."""
import sys, torch
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
S = R.load_stats_module()
torch.manual_seed(198)
d = 16
E = torch.randn(50, d, dtype=torch.float64)
gain = torch.rand(d, dtype=torch.float64) + 0.5
W0 = torch.randn(d, d, dtype=torch.float64)
# a rank-one ROME-like update whose right vector is orthogonal to W0's row space
U, s, Vh = torch.linalg.svd(W0)
right = torch.randn(d, dtype=torch.float64)
right = right - Vh.T @ (Vh @ right); right = right / right.norm()
rome = 4.0 * torch.outer(torch.randn(d, dtype=torch.float64), right)
delta, meta = R.make_aimed_delta(rome, E, gain, 7)
# the output direction the code writes, as seen in the edited matrix's top left singular vector:
Ue, se, _ = torch.linalg.svd(W0 + delta)
code_out = Ue[:, 0]
row = E[7]
design = gain * (row - row.mean()); design = design / design.norm()
print("cos(edited top output direction, design gamma*(row-mean)) =", round(float(code_out @ design), 6))
# what the reader sees on each:
for label, M in (("code's aimed suspect", W0 + delta),
                 ("design's aimed suspect", W0 + meta["spectral_norm"] * torch.outer(design, right))):
    p = S.read_pairs(M, gain, True, E, lambda ids: "x", k=1)
    print(f"  {label:24s}: top token {int(p[0].top)} z {p[0].z:.3f}  (target 7)")
