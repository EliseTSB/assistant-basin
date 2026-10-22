"""Part 51c: L22:1610's activation on Part 38's sorcerer-sense and setup-wizard-sense sentences, with random L22 units."""
import json, random
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import part38

ROOT = Path(__file__).resolve().parent
def main():
    d = json.loads((ROOT / "data/part38-homographs.json").read_text())
    u = d["units"][0]; assert u["target"] == "wizard"
    tok = AutoTokenizer.from_pretrained(part38.MODEL_ID, revision=part38.MODEL_REVISION, cache_dir=str(part38.DEFAULT_CACHE), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(part38.MODEL_ID, revision=part38.MODEL_REVISION, cache_dir=str(part38.DEFAULT_CACHE),
                                                 local_files_only=True, dtype=torch.bfloat16).eval()
    width = model.model.layers[22].mlp.down_proj.in_features
    rng = random.Random(5104); rand = sorted(rng.sample([i for i in range(width) if i != 1610], 5))
    units = [1610] + rand; cap = {}
    h = model.model.layers[22].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("a", i[0][0].float()))
    out = {"units": units, "rows": []}
    for sense in ("screened", "other"):
        for s in u["sentences"][sense]:
            ids = tok(s, return_tensors="pt"); 
            with torch.no_grad(): model(**ids)
            a = cap["a"][:, units]                     # [seq, units]
            toks = [tok.decode([t]) for t in ids.input_ids[0]]
            wpos = [i for i, t in enumerate(toks) if t.strip().lower().startswith("wizard")]
            out["rows"].append({"sense": sense, "text": s, "sent_max": a[1:].max(0).values.tolist(),
                                "at_wizard": a[wpos].max(0).values.tolist() if wpos else None})
    h.remove()
    def d_(x, y): x, y = np.asarray(x), np.asarray(y); return float((x.mean()-y.mean())/np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2))
    summ = {}
    for j, unit in enumerate(units):
        for key in ("sent_max", "at_wizard"):
            sc = [r[key][j] for r in out["rows"] if r["sense"] == "screened" and r[key]]
            ot = [r[key][j] for r in out["rows"] if r["sense"] == "other" and r[key]]
            summ[f"22:{unit}:{key}"] = {"sorcerer_mean": float(np.mean(sc)), "setup_mean": float(np.mean(ot)), "d": d_(sc, ot), "n": [len(sc), len(ot)]}
    out["summary"] = summ
    p = ROOT / "results/part51c/read.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1))
    for k, v in summ.items(): print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
main()
