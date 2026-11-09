import json, collections
rows=[json.loads(l) for l in open('../answers.jsonl',encoding='utf-8')]
by=collections.defaultdict(list)
for r in rows:
    by[r['stage']].append(r)
for st in ['recipe','battery']:
    arms=collections.defaultdict(list)
    for r in by[st]:
        arms[r['arm']].append((r['unit_id'], r['question_index'], r['question'][:25]))
    for a,v in arms.items():
        print(st, a, 'first5:', v[:5])
hr=['L21N7103','L22N2742','L25N6140','L25N4320','L21N7991','L22D113','L24D433','L23D59','L19D219','L25D84','L19N5755','L24N1813','L23N1028','L24N3172','L23N3571','L26D374','L24D226','L26D416','L23D212','L24D403']
print('hand-read recipe/battery units positions in file (stage,arm,index):')
for u in hr:
    idx=[(i,r['stage'],r['arm'],r['question_index']) for i,r in enumerate(rows) if r['unit_id']==u and r['stage'] in ('recipe','battery')]
    print(u, idx[:3])
