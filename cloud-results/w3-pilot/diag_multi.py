"""Offline check of the v5 builder: build on even-indexed calibration records, check S and step on
odd-indexed ones, per build-only wording. argv: acts_multi.pt L 'kwargs'"""
import sys, torch
from graft import build_multi, Graft
d = torch.load(sys.argv[1]); L = int(sys.argv[2])
kw = eval("dict(" + (sys.argv[3] if len(sys.argv) > 3 else "") + ")")
recs = d["recs"]
W = build_multi(d, L, recs=recs[0::2], **kw)
W["out_dir_u"] = W["out_dir"]  # no model here; output vectors irrelevant for this check
g = Graft(W, record=True)
stats = {}
for c in recs[1::2]:
    g(c["h"][L].float()[None])
    S = g.last["S"][0]; st = g.last["step"][0]; P = c["P"]; true = sum(c["pair"])
    n_ans = len(str(true))
    okS = abs(S[P].item() - true) < 0.3
    okst = all(st[P + k, k] > 0.99 and st[P + k].sum() < 1.01 for k in range(n_ans + 1) if P + k < st.shape[0])
    quiet = st[:P].sum().item() == 0
    s = stats.setdefault(c["w"], [0, 0, 0, 0]); s[0] += okS; s[1] += okst; s[2] += quiet; s[3] += 1
for w, s in sorted(stats.items()):
    print("wording", w, "S ok %d/%d  step ok %d/%d  quiet-on-prompt %d/%d" % (s[0], s[3], s[1], s[3], s[2], s[3]))
fires = 0
for o in d["off"]:
    g(o["h"][L].float()[None]); fires += int((g.last["step"] > 0).any())
print("off-texts where any step fires:", fires, "/", len(d["off"]))
