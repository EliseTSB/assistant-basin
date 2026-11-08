"""Part 198 code cross-check (gate item): stats198.py must reproduce, from base weights only,
  (a) the pilot's base sigma1 for all 48 GPT-2 XL mlp.c_proj (pilot/pilot-part1.json), relative error < 1e-6;
  (b) under the pilot's sign-first rule, the exploration's T_max for every base layer of GPT-2 XL (pilot/explore-gpt2.json)
      and Qwen3.5-0.8B (pilot/explore-tokens.json), to 0.01.
It also writes both rules' T_max and T_word per base layer (comparability report). Output: crosscheck198.json.
Run: python -I crosscheck198.py   (from results/part198, HF_HUB_OFFLINE=1)"""
import json, sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stats198 as S  # noqa: E402

GPT2 = Path("/mnt/e/neuron-probe/edit-detect/gpt2-xl")
QWEN = "/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
torch.set_num_threads(12)
torch.set_grad_enabled(False)
out, fails = {"gpt2": {}, "qwen": {}}, []

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# GPT-2 XL from safetensors, ln_f as a real LayerNorm.
tok = AutoTokenizer.from_pretrained(str(GPT2))
with safe_open(str(GPT2 / "model.safetensors"), "pt") as f:
    pre = "transformer." if any(k.startswith("transformer.") for k in f.keys()) else ""
    E = f.get_tensor(f"{pre}wte.weight").double().clone()
    ln = torch.nn.LayerNorm(E.shape[1])
    ln.weight.data = f.get_tensor(f"{pre}ln_f.weight").float().clone()
    ln.bias.data = f.get_tensor(f"{pre}ln_f.bias").float().clone()
    C = {l: f.get_tensor(f"{pre}h.{l}.mlp.c_proj.weight").double().clone().T for l in range(48)}
gain, centre = S.norm_reader(ln)
p1 = json.loads((HERE / "pilot/pilot-part1.json").read_text())["base_c_proj_sigma1"]
xg = json.loads((HERE / "pilot/explore-gpt2.json").read_text())["base"]
for l in range(48):
    s1, f2 = S.f_scores(C[l])
    pairs = S.read_pairs(C[l], gain, centre, E, tok.decode)
    pm, pw = S.pilot_sign_first(pairs)
    want_tmax = max(x["z"] for x in xg[str(l)]["dirs"])
    rel = abs(s1 - p1[str(l)]) / p1[str(l)]
    out["gpt2"][l] = {"sigma1": s1, "sigma1_rel_err": rel, "pilot_tmax": pm, "pilot_tword": pw,
                      "tmax": S.t_max(pairs), "tword": S.t_word(pairs), "explore_tmax": want_tmax}
    if rel >= 1e-6: fails.append(f"gpt2 L{l} sigma1 rel {rel:.2e}")
    if abs(round(pm, 2) - want_tmax) > 0.0100001: fails.append(f"gpt2 L{l} tmax {pm:.3f} vs {want_tmax}")
    print(f"gpt2 {l}", flush=True)

# Qwen3.5-0.8B through transformers (its zero-centred RMSNorm is read by the ones rule).
qt = AutoTokenizer.from_pretrained(QWEN)
qm = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval()
layers = norm = None
for name, mod in qm.named_modules():
    if name.endswith(".layers") and "visual" not in name and "vision" not in name:
        layers = mod
        root = qm.get_submodule(name.rsplit(".", 1)[0])
        norm = next(getattr(root, a) for a in ("norm", "final_layernorm", "norm_f") if hasattr(root, a))
        break
gq, cq = S.norm_reader(norm)
Eq = qm.get_output_embeddings().weight.detach().double()
xq = json.loads((HERE / "pilot/explore-tokens.json").read_text())["layers"]
for l in range(len(layers)):
    W = layers[l].mlp.down_proj.weight.detach().double()
    pairs = S.read_pairs(W, gq, cq, Eq, qt.decode)
    pm, pw = S.pilot_sign_first(pairs)
    want_tmax = max(x["z"] for x in xq[str(l)]["base"])
    out["qwen"][l] = {"pilot_tmax": pm, "pilot_tword": pw, "tmax": S.t_max(pairs), "tword": S.t_word(pairs),
                      "explore_tmax": want_tmax, "norm_class": type(norm).__name__, "centre": cq}
    if abs(round(pm, 2) - want_tmax) > 0.0100001: fails.append(f"qwen L{l} tmax {pm:.3f} vs {want_tmax}")
    print(f"qwen {l}", flush=True)

out["fails"] = fails
(HERE / "crosscheck198.json").write_text(json.dumps(out, indent=1))
print("CROSSCHECK", "FAIL" if fails else "ALL PASS", fails[:10], flush=True)
sys.exit(1 if fails else 0)
