import json, sys, random
from pathlib import Path
ROOT=Path('/home/elisepc/green/worktrees/ab-part67')
SCR=Path('/tmp/claude-1000/-home-elisepc-green-Journal/f862e88e-c243-45f9-aef3-92eba49bcbff/scratchpad')
for p in (str(ROOT),str(ROOT/'part169')):
    if p not in sys.path: sys.path.insert(0,p)
from part169 import run169, score169
from part166 import score166
FIX=SCR/'fix169'; OUT=SCR/'out169'; OUT.mkdir(exist_ok=True)
rows=run169.read_jsonl(FIX/'rows.jsonl'); masses=run169.read_jsonl(FIX/'masses.jsonl'); answers=run169.read_jsonl(FIX/'answers.jsonl')
norms=json.loads((FIX/'norms.json').read_text())
g=run169.gate_rules(rows,answers,masses,norms)
print('GATES', {k:v for k,v in g.items()})
res=score169.score(str(FIX/'rows.jsonl'),str(FIX/'plain.json'),str(FIX/'masses.jsonl'),str(FIX/'answers.jsonl'),str(FIX/'norms.json'),str(OUT))
P=res['predictions']; B=res['bootstrap_intervals']
print('readings_stopped', res['readings_stopped'])
for k in ('169.1','169.2','169.3','169.4','169.5','169.6','169.7','169.8','169.W1','169.W2'):
    v=P[k]; print(k, {kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk not in ('per_arm','per_arm_values','strength_matched')})
print('BOOT 169.1', B['169.1']['estimate'], B['169.1']['interval_95'])
print('BOOT 169.3', B['169.3']['estimate'], B['169.3']['interval_95'])
print('BOOT 169.4', B['169.4']['estimate'], B['169.4']['interval_95'])
print('BOOT 169.5', B['169.5']['estimate'], B['169.5']['interval_95'])
print('BOOT 169.6', B['169.6']['estimate'], B['169.6']['interval_95'])
print('BOOT 169.7', B['169.7']['estimate'], B['169.7']['interval_95'])
print('BOOT 169.8', B['169.8']['estimate'], B['169.8']['interval_95'])
print('BOOT 169.W1', B['169.W1']['estimate'], B['169.W1']['interval_95'])
print('BOOT 169.W2', B['169.W2']['estimate'], B['169.W2']['interval_95'])
print('BOOT 169.2', B['169.2']['estimate'], B['169.2']['interval_95'])
print('per_arm 169.3 boot n+', B['n+']['169.3'])
print('files', sorted(p.name for p in OUT.iterdir()))
