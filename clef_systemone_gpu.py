# clef_systemone_gpu.py — Part 1f (Claude, 2026-10-17; predictions 0e761ae pushed before any run). Answers recorded
# Jev/SystemOne requests with local Cloudflare Clef-flash in 4-bit on the A770, using the release's OWN systemone()
# helper so each noul probability is the release's p("true") (the batch path in clef_batch_gpu.py guessed which option
# index means true; a wrong guess inverts every answer). Usage, Windows venv:
#   python clef_systemone_gpu.py --snapshot <dir> --requests requests.jsonl --answers answers.jsonl [--limit N]
# Guards: refuses to continue if the loaded model takes more than --max-gb of XPU memory (the A770 spills into shared
# memory instead of failing, which froze WSL on 2026-10-16); checks direction on two known-answer requests first.
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--snapshot', required=True); ap.add_argument('--requests', required=True)
ap.add_argument('--answers', required=True); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--max-gb', type=float, default=12.0)
ap.add_argument('--weights', default=None, help='pre-quantized backbone repo snapshot (nf4); loader and joint head still come from --snapshot')
ap.add_argument('--max-host-gb', type=float, default=14.0)
a = ap.parse_args()
# HOST-MEMORY WATCHDOG (2026-10-17 22:07): quantize-on-load read the full bf16 weights into host RAM, 26 GB of Elise's
# 31, and pushed Windows commit to 67.0 of 68.1 GB, where the WSL VM itself is at risk. Any load that passes
# --max-host-gb of private memory kills this process at once.
import threading, psutil
def _watch():
    me = psutil.Process()
    while True:
        gb = me.memory_info().private / 2**30 if hasattr(me.memory_info(), 'private') else me.memory_info().rss / 2**30
        if gb > a.max_host_gb:
            print(f'refusing: host private memory {gb:.1f} GB exceeds --max-host-gb {a.max_host_gb}', flush=True); os._exit(3)
        time.sleep(0.5)
threading.Thread(target=_watch, daemon=True).start()
import torch
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *x, **k: None
from transformers import BitsAndBytesConfig
sys.path.insert(0, a.snapshot)
import joint_schema_model as J
t0 = time.time()
if a.weights:
    # Pre-quantized nf4 backbone (meossistant/clef-flash-4bit, safetensors, no code); Cloudflare's loader from --snapshot.
    # load_release_model reads joint_head from the path it loads, so build the official pieces by hand: backbone from
    # --weights (its config carries the bnb quantization), joint head + processor from the official --snapshot.
    from pathlib import Path
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(a.weights, device_map={'': 'xpu'}, dtype=torch.bfloat16)
    backbone.config.use_cache = False
    hc = json.loads((Path(a.snapshot) / 'joint_head_config.json').read_text())
    head = J.JointSchemaHead(**hc)
    head.load_state_dict(load_file(str(Path(a.snapshot) / 'joint_head.safetensors')), strict=True)
    head = head.to(device='xpu', dtype=torch.bfloat16)
    # Text only: AutoProcessor wants torchvision for the video processor (not installed in this venv). systemone() uses
    # processor.tokenizer and touches the processor itself only for images or videos (_encode_media).
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    processor = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(a.snapshot))
    model = J.ClefModel(backbone, head).eval()
else:
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    model, processor = J.load_release_model(a.snapshot, device='xpu', dtype=torch.bfloat16, quantization_config=quant)
gb = torch.xpu.memory_allocated() / 2**30
print(f'loaded in {time.time()-t0:.0f}s, xpu allocated {gb:.2f} GB', flush=True)
if gb > a.max_gb:
    sys.exit(f'refusing: {gb:.2f} GB allocated exceeds --max-gb {a.max_gb}')
def ask(state):
    q = {'fires': {'type': 'noul', 'instructions': 'Judging only by the description in the state, would this unit be strongly active on the marked token?'}}
    return J.systemone(model, processor, {'model': 'clef-flash', 'state': state, 'questions': q})['answers']['fires']['noul']
yes = ask({'description': 'fires on the word "Paris"', 'snippet': 'We flew to ⟦Paris⟧ last spring.'})
no = ask({'description': 'fires on the word "Paris"', 'snippet': 'The ⟦cat⟧ slept on the warm windowsill.'})
print(f'direction check: Paris {yes:.3f}, cat {no:.3f}', flush=True)
if not yes > no:
    sys.exit('refusing: direction check failed (the known-yes request did not score above the known-no one)')
done = set()
if os.path.exists(a.answers):
    for line in open(a.answers, encoding='utf-8'):
        if line.strip(): done.add(json.loads(line)['hash'])
todo = []
for line in open(a.requests, encoding='utf-8'):
    if not line.strip(): continue
    r = json.loads(line); h = r['hash']
    if h in done: continue
    done.add(h); todo.append(r)
if a.limit: todo = todo[:a.limit]
print(f'{len(todo)} requests to answer', flush=True)
t1 = time.time(); n_tok = 0
with open(a.answers, 'a', encoding='utf-8') as out:
    for i, r in enumerate(todo, 1):
        res = J.systemone(model, processor, {'model': 'clef-flash', 'state': r['state'], 'questions': r['questions']})
        n_tok += res['usage']['input_tokens']
        out.write(json.dumps({'hash': r['hash'], 'answers': {k: v['noul'] for k, v in res['answers'].items() if v['type'] == 'noul'},
                              'input_tokens': res['usage']['input_tokens']}) + '\n')
        if i % 100 == 0 or i == len(todo):
            out.flush(); torch.xpu.empty_cache()
            el = time.time() - t1
            print(f'{i}/{len(todo)} {i/el:.2f} req/s {n_tok/el:.0f} tok/s, xpu {torch.xpu.memory_allocated()/2**30:.2f} GB', flush=True)
print(f'done: {len(todo)} in {time.time()-t1:.0f}s', flush=True)
