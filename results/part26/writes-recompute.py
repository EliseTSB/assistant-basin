import json, sys, glob, os, torch
from safetensors import safe_open
from transformers import AutoTokenizer
def norm_tok(t):
    t=t.lstrip()
    while t.startswith('Ġ'): t=t[1:]
    return t.lower()
src=json.load(open(os.path.expanduser('~/assistant-basin/results/part26/target-olmo2-1b.json')))['source_results'][0]['source_neuron']['writes_top30']
srcset=set(src)
jobs={'allenai/OLMo-2-0425-1B-Instruct':('48d788eca847d4d7548f375ad03d3c9312f6139e',[(12,4373,4),(12,6212,5),(14,3027,1),(10,6978,1)]),
      'HuggingFaceTB/SmolLM3-3B':('a07cc9a04f16550a088caea529712d1d335b0ac1',[(25,75,4),(30,1251,0),(21,7967,0),(23,1019,0)])}
for mid,(rev,units) in jobs.items():
    snap=(f'/mnt/e/neuron-probe/hf/hub/models--{mid.replace("/","--")}/snapshots/{rev}')
    tok=AutoTokenizer.from_pretrained(snap)
    files=glob.glob(snap+'/*.safetensors'); keys={}
    for f in files:
        with safe_open(f,'pt') as h:
            for k in h.keys(): keys[k]=f
    def get(k):
        with safe_open(keys[k],'pt') as h: return h.get_tensor(k)
    unemb=get('lm_head.weight') if 'lm_head.weight' in keys else get('model.embed_tokens.weight')
    gain=get('model.norm.weight').float()
    vocab=min(len(tok),unemb.shape[0])
    print('=====',mid,'tied' if 'lm_head.weight' not in keys else 'untied')
    for L,u,expect in units:
        w=get(f'model.layers.{L}.mlp.down_proj.weight')[:,u].float()
        logits=unemb[:vocab].float()@(gain*w)
        ids=torch.topk(logits,30).indices.tolist()
        toks=[norm_tok(tok.decode([i])) for i in ids]
        ov=len(set(toks)&srcset)
        print(f'{L}:{u} overlap={ov} (stored {expect}) alien_in_top30={"alien" in toks or "aliens" in toks} shared={sorted(set(toks)&srcset)}')
        print('   ', toks)
