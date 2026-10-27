#!/usr/bin/env python3
"""Part 80 self-reference split by option order and body split, written after the cold check (the pooled numbers hid
an option-position lock in Qwen3-1.7B's order 2). Also the U' rendering's P(the user) and the per-layer A-vs-(U,S)
direction separation from score80's output."""
import json,sys,collections
from pathlib import Path
def main(dirs):
    out={}
    for d in map(Path,dirs):
        recs=[json.loads(l) for l in (d/'records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        agg=collections.defaultdict(list)
        for r in recs:
            rp=r.get('role_probs') or r.get('role_probabilities')
            for split in ('all',r.get('split')):
                agg[(split,r['order'],r['rendering'],'you')].append(rp['you']); agg[(split,r['order'],r['rendering'],'user')].append(rp['user'])
                agg[(split,'pooled',r['rendering'],'you')].append(rp['you']); agg[(split,'pooled',r['rendering'],'user')].append(rp['user'])
        res={f'{s}|order {o}|{rend}|P({who})':round(sum(v)/len(v),3) for (s,o,rend,who),v in agg.items()}
        out[d.name]=dict(sorted(res.items()))
    print(json.dumps(out,indent=1,ensure_ascii=False))
    Path(dirs[0]).parent.joinpath('order_breakdown.json').write_text(json.dumps(out,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
if __name__=='__main__': main(sys.argv[1:])
