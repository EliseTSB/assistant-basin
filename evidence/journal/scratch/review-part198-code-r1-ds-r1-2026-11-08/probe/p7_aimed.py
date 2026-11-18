"""Probe: the aimed positive control's output direction. Design: 'centred, gamma-scaled unembedding row'.
stat_198's LayerNorm reading applies gain to the CENTRED direction: gain*(u - mean u)."""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import torch
import run198 as R

d = 6
E = torch.zeros(10, d, dtype=torch.float64)
E[3] = torch.tensor([3.0, -1.0, 0.5, 2.0, 0.0, 1.0])
gain = torch.tensor([0.5, 2.0, 1.0, 1.5, 0.25, 3.0], dtype=torch.float64)   # nonuniform gamma
rome = torch.zeros(6, 6, dtype=torch.float64); rome[0, 0] = 4.0; rome[1, 1] = 1.0
delta, meta = R.make_aimed_delta(rome, E, gain, 3)
u = delta[0] / meta["spectral_norm"]          # the output direction actually written
want = gain * (E[3] - E[3].mean()); want = want / want.norm()
code = gain * E[3]; code = code - code.mean(); code = code / code.norm()
print("code output direction      :", [round(x, 4) for x in u.tolist()])
print("design (gamma*(u-mean u))  :", [round(x, 4) for x in want.tolist()])
print("what the code builds       :", [round(x, 4) for x in code.tolist()])
print("cos(code, design) =", round(float(u @ want), 6))
# what the reader would score for each:
S = R.load_stats_module()
for label, vec in (("code", u), ("design", want)):
    W = torch.zeros(6, 6, dtype=torch.float64); W[:, 0] = vec     # rank-one, output = vec
    pairs = S.read_pairs(W, gain, True, E, lambda ids: "x", k=1)
    print(label, "top token:", int(pairs[0].top), "z:", round(pairs[0].z, 4))
