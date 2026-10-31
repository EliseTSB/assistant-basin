import json,base64
import numpy as np
G='/mnt/c/Users/alexa/neuron-probe-gpu/results/part136/pilot-effects-b16/effects.jsonl'
b={}
def band(l): return 'L0-22' if l<=22 else ('L23-23' if l==23 else 'L24-31')
for l in open(G,encoding='utf-8'):
    r=json.loads(l); e=np.frombuffer(base64.b64decode(r['effect_a_f32_le_b64']),dtype='<f4')
    b.setdefault(band(r['layer']),[]).append(float(np.median(np.abs(e))))
print({k:(len(v),float(np.median(v))) for k,v in b.items()})
