"""J-lens and logit-lens top words for every neuron on the Part 86 answers page."""
import json, sys, time
from pathlib import Path
R = Path('/home/elisepc/green/worktrees/ab-part67')
sys.path.insert(0, str(R))
import torch, part42
S = Path(sys.argv[1])
snapshot = Path('/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
lens = Path.home() / 'neuron-probe/jlens/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt'
rows = json.load(open(S / 'screen-page/rows.json', encoding='utf-8'))
units = sorted({(u[0], u[1]) for r in rows for u in r['u'] if u[2]})
print(len(units), flush=True)
config = json.loads((snapshot / 'config.json').read_text(encoding='utf-8'))
text_config, hidden, num_layers, model_type = part42.config_dimensions(config)
weight_map = part42.load_weight_map(snapshot)
prefix = part42.find_language_model_prefix(weight_map, require_language_model=True)
jacobians, meta, d_model = part42.load_jacobian_lens(lens)
gain = part42.effective_norm_gain(part42.load_safetensor(snapshot, weight_map, f'{prefix}norm.weight').float(), model_type)
tied = bool(text_config.get('tie_word_embeddings', config.get('tie_word_embeddings', False)))
head = part42.load_safetensor(snapshot, weight_map, part42.select_unembedding_key(weight_map, prefix, tied))
tok = part42._load_tokenizer(snapshot)
out = {}
t0 = time.time()
torch.set_num_threads(4)
H = head.float()
g = gain.float()
for layer in sorted({l for l, _ in units}):
    down = part42.load_safetensor(snapshot, weight_map, part42._layer_tensor_key(prefix, layer)).float()
    jac = jacobians.get(part42.jacobian_layer_for_write(layer, num_layers)).float()
    ns = [n for l, n in units if l == layer]
    for start in range(0, len(ns), 64):
        chunk = ns[start:start + 64]
        W = down[:, chunk]
        tj = torch.topk((H @ (g[:, None] * (jac @ W) if g.ndim else g * (jac @ W))).T, 12, dim=1).indices.tolist()
        tl = torch.topk((H @ (g[:, None] * W if g.ndim else g * W)).T, 6, dim=1).indices.tolist()
        for n, a, b in zip(chunk, tj, tl):
            out[f'{layer}:{n}'] = {'j': [tok.decode([i]) for i in a], 'l': [tok.decode([i]) for i in b]}
    print(layer, len(out), round(time.time() - t0), flush=True)
json.dump(out, open(S / 'screen-page/lens.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('done', len(out), flush=True)
