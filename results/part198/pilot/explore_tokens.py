"""Part 198 EXPLORATORY (design input, disclosed as such, never a confirmation): do the singular directions of an edited
mlp.down_proj read, through the unembedding, as the written token? Uses the saved wsp pilot update (CounterFact case 0,
"English") on Qwen3.5-0.8B. For every layer's down_proj, base and (layers 17-21) edited: the top-64 left singular vectors
are read through the final norm's per-dimension scale and the unembedding (the way wsp aims its targets), and each gets a
token-concentration score z = (max logit - mean) / std over the vocabulary, for both signs. Also reads the update's own
top singular vectors. Run: python -I explore_tokens.py --out explore-tokens.json
"""
import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
DELTA = Path(__file__).resolve().parent / "wsp-pilot.delta.pt"
K = 64


def read_dirs(U: torch.Tensor, g: torch.Tensor, E: torch.Tensor, tok, top=3):
    """U: [d, k] residual directions. Returns per direction: best sign, z, top tokens."""
    out = []
    L = E @ (g[:, None] * U)  # [vocab, k]
    for j in range(U.shape[1]):
        best = None
        for sgn in (1.0, -1.0):
            col = sgn * L[:, j]
            z = float((col.max() - col.mean()) / col.std())
            if best is None or z > best[1]:
                ids = col.topk(top).indices.tolist()
                best = (sgn, z, [tok.decode([i]) for i in ids])
        out.append({"sign": best[0], "z": round(best[1], 2), "top": best[2]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(12)
    torch.set_grad_enabled(False)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    layers = None
    for name, mod in model.named_modules():
        if name.endswith(".layers") and "visual" not in name and "vision" not in name:
            layers = mod
            break
    root = model.get_submodule(".".join(n for n in name.split(".")[:-1])) if "." in name else model
    norm = next(getattr(root, x) for x in ("norm", "final_layernorm", "norm_f") if hasattr(root, x))
    E = model.get_output_embeddings().weight.detach().float()
    d = E.shape[1]
    g = norm(torch.ones(1, d))[0].float()
    delta = torch.load(DELTA)
    res = {"load_s": round(time.time() - t0, 1), "layers": {}, "delta_dirs": {}}
    for l in range(len(layers)):
        W = layers[l].mlp.down_proj.weight.detach().double()
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
        entry = {"base": read_dirs(U[:, :K].float(), g, E, tok), "base_s": [round(float(x), 4) for x in S[:K]]}
        if l in delta:
            W1 = W + delta[l].double()
            U1, S1, _ = torch.linalg.svd(W1, full_matrices=False)
            entry["edited"] = read_dirs(U1[:, :K].float(), g, E, tok)
            entry["edited_s"] = [round(float(x), 4) for x in S1[:K]]
            Ud, Sd, _ = torch.linalg.svd(delta[l].double(), full_matrices=False)
            res["delta_dirs"][str(l)] = {"s": [round(float(x), 4) for x in Sd[:8]],
                                         "dirs": read_dirs(Ud[:, :6].float(), g, E, tok)}
        res["layers"][str(l)] = entry
        print(f"layer {l} done", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("DONE", res["load_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
