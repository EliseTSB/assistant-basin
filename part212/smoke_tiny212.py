"""Smoke: drive part212/run212.py's real stage1 (and optionally the stage2 pilot) on a tiny random Qwen3.5.

Usage (from the worktree root, on the host venv, HF_HUB_OFFLINE=1):
  python -I -B part212/smoke_tiny212.py <worktree> <scratch-dir> [stage1|pilot]
Writes only under <scratch-dir>/out. Exit 0 and a final "SMOKE OK" line means the real torch glue ran end to end.
This is the acceptance test for the runner's glue: the unit tests cannot see shape bugs between record and hook.
The tiny model has the real tokenizer, 30 layers and an MLP 9216 wide so TARGET (6:753, 7:1169, 11:8868, 29:776) exist.
"""
import json, os, sys, types, time
wt, scratch, what = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'pilot')
sys.path.insert(0, wt); sys.path.insert(0, os.path.join(wt, 'part212'))
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import neuron_report as nr
import run212

SNAP = '/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
cfg = json.load(open(os.path.join(SNAP, 'config.json'), encoding='utf-8'))
t = dict(cfg.get('text_config', cfg))
t.update(num_hidden_layers=30, hidden_size=64, intermediate_size=9216, num_attention_heads=4,
         num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
         linear_key_head_dim=16, linear_value_head_dim=16, layer_types=t['layer_types'][:30],
         tie_word_embeddings=True)
t.pop('architectures', None)
conf = AutoConfig.for_model(t.get('model_type', 'qwen3_5_text'), **{k: v for k, v in t.items() if k != 'model_type'})
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)

def fake_load(name, rev, device, *a, **k):
    nr.LAST_LOAD = {'model': name, 'revision': rev, 'device': device, 'tiny_random_smoke': True}
    return tok, model
nr.load_model = fake_load
os.makedirs(scratch, exist_ok=True)
os.chdir(scratch)  # --out must be relative (the Windows job runner's rule)
t0 = time.time()
run212.main([what, '--out', 'out', '--device', 'cpu'])
print('SMOKE OK', what, round(time.time() - t0, 1), 's', flush=True)
