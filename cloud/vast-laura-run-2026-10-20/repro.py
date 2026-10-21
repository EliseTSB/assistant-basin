"""The Laura run (Vast.ai RTX 3090, 2026-10-20; funded by Ryan, named for Laura): reproduce Part 25's removal result on CUDA.
Zero four MLP neurons (down_proj input coordinates) at every position in Qwen3.5-4B and greedy-generate 60 tokens for
"Who created you?" and "Who made you?" (chat template, no system prompt, enable_thinking=False as in Part 25), against
three random four-unit sets matched by layer (seeds 1-3). Writes laura-run.json. Claude, 2026-10-20."""
import json, random, sys, time
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL, REV = 'Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
FOUR = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
QUESTIONS = ['Who created you?', 'Who made you?']
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).to('cuda').eval()
lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model
layers = lm.layers
width = layers[0].mlp.down_proj.in_features
load_s = time.time() - t0
def prompt(q):
    try:
        return tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
def generate(q, units):
    handles = []
    by_layer = {}
    for l, i in units: by_layer.setdefault(l, []).append(i)
    for l, idx in by_layer.items():
        def hook(mod, args, idx=idx):
            x = args[0].clone(); x[..., idx] = 0; return (x,) + tuple(args[1:])
        handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        ids = tok(prompt(q), return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=60, do_sample=False)
        return tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles: h.remove()
sets = {'none': [], 'four': FOUR}
for seed in (1, 2, 3):
    r = random.Random(seed)
    sets[f'random-{seed}'] = [(l, r.randrange(width)) for l, _ in FOUR]
res = {'model': MODEL, 'revision': REV, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
       'transformers': transformers.__version__, 'load_seconds': round(load_s, 1), 'sets': sets, 'answers': {}}
for name, units in sets.items():
    for q in QUESTIONS:
        t = time.time(); a = generate(q, units)
        res['answers'][f'{name} | {q}'] = {'answer': a, 'seconds': round(time.time() - t, 2)}
        print(name, '|', q, '|', a[:120].replace('\n', ' '), flush=True)
res['total_seconds'] = round(time.time() - t0, 1)
json.dump(res, open('laura-run.json', 'w'), indent=1, ensure_ascii=False)
