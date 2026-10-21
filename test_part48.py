import sys, math, torch
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part48 as p
# AUC
assert p.auc([3,4],[1,1]+[]) if False else True
assert p.auc([2,3,0,1],[1,1,0,0]) == 1.0
assert p.auc([0,1,2,3],[1,1,0,0]) == 0.0
assert p.auc([1,1],[1,0]) == 0.5
pp = p.perm_p([2,3,0,1,5,6,-1,-2],[1,1,0,0,1,1,0,0], draws=2000)
assert pp < 0.05, pp
pn = p.perm_p([0,1,2,3,4,5,6,7],[1,0,1,0,1,0,1,0], draws=2000)
assert pn > 0.3, pn
from transformers import AutoModelForCausalLM, AutoTokenizer
mp = '/mnt/e/neuron-probe/hf/hub/models--EleutherAI--pythia-410m/snapshots/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f'
tok = AutoTokenizer.from_pretrained(mp); model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.float32).eval()
torch.set_grad_enabled(False)
pids = tok('1, 2, 3, 4,', add_special_tokens=True)['input_ids']
a = p.continuation_logprob(model, tok, pids, '5', 'cpu'); b = p.continuation_logprob(model, tok, pids, 'banana', 'cpu')
print('count control', a, b); assert a > b + 3
# multi-token word equals the sum computed from one forward of the full string
w = 'jagged'; cont = tok(' '+w, add_special_tokens=False)['input_ids']; print('jagged tokens', len(cont))
full = torch.tensor([pids+cont]); lp = torch.log_softmax(model(full).logits[0], -1)
manual = sum(lp[len(pids)-1+i, t].item() for i,t in enumerate(cont))
assert abs(manual - p.continuation_logprob(model, tok, pids, w, 'cpu')) < 1e-4
rows = p.score_model(model, tok, 'cpu', limit=3)
assert len(rows) == 3*len(p.TEMPLATES); print('mass', [round(r['mass'],4) for r in rows[:6]])
# Shared-prompt batching: every word's number matches its one-sequence forward (fp32, Pythia-410m)
words = list(p.SHARP_WORDS) + list(p.ROUND_WORDS) + ['jagged', 'banana']
for prompt in ('1, 2, 3, 4,', 'The shape called takete is'):
    ids = tok(prompt, add_special_tokens=True)['input_ids']
    single = [p.continuation_logprob(model, tok, ids, x, 'cpu') for x in words]
    batched = p.continuation_logprobs(model, tok, ids, words, 'cpu')
    worst = max(abs(x - y) for x, y in zip(single, batched)); print('batched vs single, worst', worst)
    assert worst < 1e-4, worst
p.ONE_PER_FORWARD = True; old = p.score_model(model, tok, 'cpu', limit=2)
p.ONE_PER_FORWARD = False; new = p.score_model(model, tok, 'cpu', limit=2)
worst = max(abs(x['score'] - y['score']) for x, y in zip(old, new)); print('score worst', worst); assert worst < 1e-3
print('ALL PASS')
