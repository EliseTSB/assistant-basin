"""Part 198 EXPLORATORY (design input, disclosed as such): the same token reading as explore_tokens.py, on GPT-2 XL's 48
mlp.c_proj matrices, base and edited by ROME / MEMIT / AlphaEdit single edits (CounterFact case 0, new object "English").
GPT-2's Conv1D stores c_proj as [6400 in, 1600 out]; the residual-side singular vectors are the 1600-dim ones. They are
read through ln_f the way a residual direction reaches the logits: centred, scaled by ln_f's weight, times wte.
Run: python -I explore_gpt2.py --out explore-gpt2.json pilot-sd.json.ROME.delta.pt [more delta files]
"""
import argparse, json, time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = Path("/mnt/e/neuron-probe/edit-detect/gpt2-xl")
K = 64


def read_dirs(U, g, E, tok, top=3):
    U = U - U.mean(0, keepdim=True)  # LayerNorm centres the residual vector
    L = E @ (g[:, None] * U)
    out = []
    for j in range(U.shape[1]):
        best = None
        for sgn in (1.0, -1.0):
            col = sgn * L[:, j]
            z = float((col.max() - col.mean()) / col.std())
            if best is None or z > best[1]:
                best = (sgn, z, [tok.decode([i]) for i in col.topk(top).indices.tolist()])
        out.append({"sign": best[0], "z": round(best[1], 2), "top": best[2]})
    return out


def resid_svd(w_in_out: torch.Tensor):
    W = w_in_out.double().T  # [1600 out, 6400 in]
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    return U, S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("deltas", nargs="+")
    a = ap.parse_args()
    torch.set_num_threads(12)
    torch.set_grad_enabled(False)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    with safe_open(str(MODEL / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        pre = "transformer." if any(k.startswith("transformer.") for k in keys) else ""
        E = f.get_tensor(f"{pre}wte.weight").float().clone()
        g = f.get_tensor(f"{pre}ln_f.weight").float().clone()
        C = {l: f.get_tensor(f"{pre}h.{l}.mlp.c_proj.weight").float().clone() for l in range(48)}
    res = {"load_s": round(time.time() - t0, 1), "base": {}, "edits": {}}
    for l in range(48):
        U, S = resid_svd(C[l])
        res["base"][str(l)] = {"dirs": read_dirs(U[:, :K].float(), g, E, tok), "s": [round(float(x), 4) for x in S[:K]]}
        print(f"base {l}", flush=True)
    for path in a.deltas:
        D = torch.load(path)
        ent = {}
        for pname, d in D.items():
            l = int(pname.split(".")[2])
            U1, S1 = resid_svd(C[l] + d)
            Ud, Sd = resid_svd(d)
            ent[str(l)] = {"edited": read_dirs(U1[:, :K].float(), g, E, tok), "edited_s": [round(float(x), 4) for x in S1[:K]],
                           "delta_dirs": read_dirs(Ud[:, :4].float(), g, E, tok), "delta_s": [round(float(x), 4) for x in Sd[:6]]}
        res["edits"][Path(path).name] = ent
        print(f"edit {path}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("DONE", res["load_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
