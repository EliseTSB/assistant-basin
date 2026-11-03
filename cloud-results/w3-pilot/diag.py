"""Offline diagnostic on stored calibration activations (check half): does the block's internal
computation (S, step) come out right?  argv: acts.pt L"""
import sys, torch
from graft import *
d = torch.load(sys.argv[1]); L = int(sys.argv[2])
W = build(d, L, **eval("dict(" + (sys.argv[3] if len(sys.argv) > 3 else "") + ")"))
g = Graft(W, record=True)
for case, qs in (("two", [27, 28, 29]), ("three", [27, 28, 29, 30])):
    rec = d[case]; n = len(rec["pairs"]); B = slice(n // 2, n)
    h = rec["h"][B, L].float(); pairs = rec["pairs"][n // 2:]
    g(h); S = g.last["S"]; st = g.last["step"]
    true = torch.tensor([a + b for a, b in pairs]).float()
    for k, q in enumerate(qs):
        okS = (S[:, q] - true).abs() < 0.3
        okstep = (st[:, q, k] > 0.99) & ((st[:, q].sum(-1)-st[:, q, k]) < 0.01)
        if not okstep.all(): print("   bad step vecs", st[:, q][~okstep][:3].tolist())
        print(case, "q", q, "S ok", okS.float().mean().item(), "step ok", okstep.float().mean().item())
    other = st[:, :27].sum(-1)
    print(case, "prompt positions with any step>0:", (other > 0).float().mean().item())
for o in d["off"][len(d["off"]) // 2:]:
    g(o["h"][L][None].float())
    if (g.last["step"] > 0).any(): print("off-text fires", g.last["step"].amax(1))
print("found mean on q27", g.last["found"].shape)
