"""Part 198 EXPLORATORY (design input): can the token reading see a write AIMED at a token at GPT-2 XL layer 17?
A rank-one update u v^T added to h.17.mlp.c_proj, output direction u = g*E[t] (unit), input v random unit, spectral norm
5.432 (the pilot ROME update's). t = first token of ' English'. Reports T_word of layer 17 before/after and its rank."""
import json, re
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
MODEL = "/mnt/e/neuron-probe/edit-detect/gpt2-xl"
STOP = set("a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with you your yours yourself yourselves also would could may might must shall one two get got make makes made like".split())
def content(t): s = t.strip(); return bool(re.fullmatch(r"[A-Za-z]{3,}", s)) and s.lower() not in STOP
torch.set_grad_enabled(False); torch.manual_seed(198)
tok = AutoTokenizer.from_pretrained(MODEL)
with safe_open(MODEL + "/model.safetensors", "pt") as f:
    pre = "transformer." if any(k.startswith("transformer.") for k in f.keys()) else ""
    E = f.get_tensor(f"{pre}wte.weight").float().clone(); g = f.get_tensor(f"{pre}ln_f.weight").float().clone()
    W = f.get_tensor(f"{pre}h.17.mlp.c_proj.weight").float().clone().T.double()  # [1600, 6400]
def tword(Wm):
    U, S, _ = torch.linalg.svd(Wm, full_matrices=False); U = U[:, :64].float(); U = U - U.mean(0, keepdim=True)
    L = E @ (g[:, None] * U); best = (0.0, None)
    for j in range(64):
        for sg in (1, -1):
            col = sg * L[:, j]; z = float((col.max() - col.mean()) / col.std()); t = tok.decode([int(col.argmax())])
            if content(t) and z > best[0]: best = (z, t, j)
    return best
t = tok(" English").input_ids[0]
u = (g * E[t]).double(); u = u - u.mean(); u = u / u.norm()
v = torch.randn(6400, dtype=torch.float64); v = v / v.norm()
out = {"base": tword(W)}
for scale in (1.0, 0.5, 2.0):
    out[f"aimed x{scale}"] = tword(W + scale * 5.432 * torch.outer(u, v))
print(json.dumps(out))
