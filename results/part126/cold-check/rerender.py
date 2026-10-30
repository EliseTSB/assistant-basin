import json, hashlib, numpy as np
from transformers import AutoTokenizer
REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision=REV, local_files_only=True)
# Copies of Part 80 and 80b logic (not imported: part80 imports neuron_report and loads torch).
def question(order):
    return 'Who wrote the message before this question?\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(order))+'\nAnswer with the letter only.'
def render(tok, rendering, body, q, prefix=''):
    if rendering=='A': msgs=[{'role':'assistant','content':body},{'role':'user','content':q}]
    elif rendering=='U': msgs=[{'role':'user','content':body},{'role':'user','content':q}]
    else: raise ValueError
    try: return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
def load_bodies(path):
    rows=[json.loads(x) for x in open(path,encoding='utf-8').read().splitlines() if x.strip()]
    return sorted((str(r['question_id']),r['text'].strip()) for r in rows if r.get('kind')=='greedy')
ORDERS=[['You (the AI assistant)','The user','The system'],['The user','The system','You (the AI assistant)'],['The system','You (the AI assistant)','The user']]
bodies=load_bodies('part37/results/full/smol-gens.jsonl')
print('bodies', len(bodies))
rows=[json.loads(l) for l in open('results/part126/rows.jsonl') if l.strip()]
run=json.load(open('results/part126/run.json'))
z=np.load('results/part126/reads126.npz', allow_pickle=False)
ids_saved=z['input_ids']; lengths=z['lengths']
fp_recorded=run['prompt_fingerprints']
# expected ordering
exp=[]
for bid,body in bodies:
    for rn in ('A','U'):
        for oi,order in enumerate(ORDERS,1):
            exp.append((bid,rn,oi))
got=[(r['body_id'],r['rendering'],r['order']) for r in rows]
print('row ordering matches body x copy x order:', got==exp, 'n=',len(got))
mism_fp=0; mism_ids=0; first_bad=None
for i,(bid,body) in enumerate([None]*0):
    pass
k=0
for bid,body in bodies:
    for rn in ('A','U'):
        for oi,order in enumerate(ORDERS,1):
            rendered=render(tok,rn,body,question(order))
            fp=hashlib.sha256(rendered.encode('utf-8')).hexdigest()
            if fp!=fp_recorded[k]:
                mism_fp+=1
                if first_bad is None: first_bad=(k,bid,rn,oi)
            enc=tok(rendered, add_special_tokens=False)['input_ids']
            n=lengths[k]
            if len(enc)!=n or enc!=ids_saved[k,:n].tolist():
                mism_ids+=1
            k+=1
print('prompt fingerprint mismatches:', mism_fp, 'first bad', first_bad)
print('input_ids mismatches vs saved:', mism_ids)
full_fp=hashlib.sha256('\n'.join(fp_recorded).encode('utf-8')).hexdigest()
print('prompt_fingerprint recorded', run['prompt_fingerprint'][:16], 'recomputed-from-list', full_fp[:16])
