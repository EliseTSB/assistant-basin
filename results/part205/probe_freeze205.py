"""Probe (2026-11-14, Claude): does a Part 205 ranking pass on the A770 build a gradient for every weight, and does
freezing the weights (with the embedding output marked as requiring grad) give the same site estimates faster and
smaller? Both qwen-chat attempts died with Level Zero UR_RESULT_ERROR_UNKNOWN about 15-18 minutes into the ranking
block, at about 55 s a pass. Prints timing, peak device memory, and how far the estimate vectors differ between passes;
never the estimates themselves. One model load, prompt 0 of the seeded ranking order."""
import json, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "part205")); sys.path.append(str(ROOT))
import torch
import common205 as C
import run205 as R

cfg = json.loads((ROOT / "part205" / "config.json").read_text(encoding="utf-8"))
prompts = C.load_prompts(ROOT / "part205" / "prompts205.json")
tok, model, env = R.load_stage(cfg, "qwen-chat", "xpu")
units = C.units_for(model)
runner = R.Runner(model, tok, units, {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")})
sel = R.selection_prompts(prompts); random.Random(cfg["seeds"]["prompt_order"]).shuffle(sel)
p = sel[0]; text = R.build_prompt(prompts, "You", p["value"], p["question"])

def one(label):
    torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(); t = time.time()
    _, read, acts, grads = runner.run(text, need_grad=True, consistent=p["consistent"])
    torch.xpu.synchronize()
    est = C.estimates_for_prompt(acts, grads, units).float().cpu()
    n_param_grads = sum(1 for q in model.parameters() if q.grad is not None)
    r = {"label": label, "seconds": round(time.time() - t, 2), "peak_alloc_gb": round(torch.xpu.max_memory_allocated() / 2**30, 2),
         "params_with_grad": n_param_grads}
    print(json.dumps(r), flush=True); return est, r

out = {"env": {k: env.get(k) for k in ("model_id", "revision", "dtype", "device", "attn_implementation")}}
a1, out["A1"] = one("A1 current code (weights require grad)")
a2, out["A2"] = one("A2 current code again (run-to-run)")
model.zero_grad(set_to_none=True)
model.requires_grad_(False)
emb = model.get_input_embeddings()
h = emb.register_forward_hook(lambda m, a, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)
b1, out["B1"] = one("B1 weights frozen, embedding output requires grad")
h.remove()
out["A1_vs_A2"] = {"max_abs": float((a1 - a2).abs().max()), "equal": bool(torch.equal(a1, a2))}
out["A1_vs_B1"] = {"max_abs": float((a1 - b1).abs().max()), "equal": bool(torch.equal(a1, b1)),
                   "max_abs_est": float(a1.abs().max())}
print(json.dumps(out, indent=1))
(HERE / "probe_freeze205.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
