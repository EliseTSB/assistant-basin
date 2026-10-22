"""Read arbitrary MLP units' write directions through the logit lens and the Jacobian lens (Part 42's machinery).
Usage: jlens_units.py --model qwen3-1.7b --units 22:1610 22:569 ... --out results/part51-lens/read.json"""
import argparse, json
from pathlib import Path
import part42

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B"); ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    snapshot = part42.resolve_snapshot(a.model, None)
    config = json.loads((snapshot / "config.json").read_text())
    text_config, hidden, num_layers, model_type = part42.config_dimensions(config)
    wm = part42.load_weight_map(snapshot)
    prefix = part42.find_language_model_prefix(wm, require_language_model=False)
    jac, meta, dm = part42.load_jacobian_lens(part42.MODEL_LENS_PATHS[a.model])
    assert dm == hidden
    gain = part42.effective_norm_gain(part42.load_safetensor(snapshot, wm, f"{prefix}norm.weight").float(), model_type)
    tied = bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    head = part42.load_safetensor(snapshot, wm, part42.select_unembedding_key(wm, prefix, tied))
    tok = part42._load_tokenizer(snapshot)
    out = {"model": a.model, "lens": str(part42.MODEL_LENS_PATHS[a.model]), "units": []}
    for spec in a.units:
        layer, unit = map(int, spec.split(":"))
        down = part42.load_safetensor(snapshot, wm, part42._layer_tensor_key(prefix, layer))
        w = down[:, unit].float()
        jl = part42.jacobian_layer_for_write(layer, num_layers)
        J = jac[jl].float() if jl is not None else None
        rec = {"layer": layer, "unit": unit,
               "logit_lens": part42._top_bottom(torch, part42.readout_logits(torch, head, w, gain), tok),
               "j_lens": part42._top_bottom(torch, part42.readout_logits(torch, head, w, gain, J), tok)}
        out["units"].append(rec)
        for k in ("logit_lens", "j_lens"):
            print(spec, k, "TOP:", [r["decoded"] for r in rec[k]["top"][:20]])
            print(spec, k, "BOT:", [r["decoded"] for r in rec[k]["bottom"][:20]])
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
main()
