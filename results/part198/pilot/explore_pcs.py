"""Part 198 EXPLORATORY (design input): mean pairwise cosine similarity (PCS) of a matrix's per-neuron output vectors
(Youssef et al. arXiv:2505.20819 App. C use row cosine similarity), base vs edited, with the sibling layers as the
base-free reference. mean_{i<j} cos(r_i, r_j) = (||sum_i r_i/|r_i| ||^2 - n) / (n (n-1)). Also the other orientation."""
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open

def pcs(R):  # rows are vectors
    Rn = R / R.norm(dim=1, keepdim=True)
    n = Rn.shape[0]; s = Rn.sum(0)
    return float((s @ s - n) / (n * (n - 1)))

def both(W_out_in):  # W [out, in]: neuron output vectors are columns
    W = W_out_in.double()
    return {"neuron_out": pcs(W.T), "out_dim": pcs(W)}

res = {}
G = Path("/mnt/e/neuron-probe/edit-detect/gpt2-xl/model.safetensors")
with safe_open(str(G), "pt") as f:
    pre = "transformer." if any(k.startswith("transformer.") for k in f.keys()) else ""
    C = {l: f.get_tensor(f"{pre}h.{l}.mlp.c_proj.weight").float().clone().T for l in range(48)}  # -> [1600 out, 6400 in]
res["gpt2_base"] = {l: both(C[l]) for l in range(48)}
for p in ["pilot-sd.json.ROME.delta.pt", "pilot-sd.json.MEMIT.delta.pt", "pilot-sd-ae.json.AlphaEdit.delta.pt"]:
    D = torch.load(p); res[p] = {}
    for pn, d in D.items():
        l = int(pn.split(".")[2]); res[p][l] = both(C[l] + d.T)
Q = "/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors"
with safe_open(Q, "pt") as f:
    ks = [k for k in f.keys() if k.endswith("mlp.down_proj.weight") and "visual" not in k and "mtp" not in k]
    byl = {int(k.split("layers.")[1].split(".")[0]): k for k in ks}
    QW = {l: f.get_tensor(byl[l]).float().clone() for l in sorted(byl)}
res["qwen_base"] = {l: both(QW[l]) for l in QW}
D = torch.load("wsp-pilot.delta.pt")
res["wsp"] = {l: both(QW[l] + D[l].float()) for l in D}
Path("explore-pcs.json").write_text(json.dumps(res, indent=1))
print("DONE")
