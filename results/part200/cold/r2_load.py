import json, collections
rec=json.load(open('../load.json',encoding='utf-8'))
print('top keys', list(rec.keys()), 'part', rec.get('part'))
for e in rec['loads']:
    print('status', e.get('status'), 'time', e.get('time') or e.get('started') or e.get('launched'), [k for k in e.keys()])
    print(' load', e.get('load'))
    calls=e.get('calls',[])
    print(' n calls', len(calls), 'stages', collections.Counter(c.get('stage') for c in calls))
    pushed=[c for c in calls if c.get('pushed') and c.get('stage')!='batch1']
    print(' pushed calls', len(pushed), 'chunks total', sum(len(c.get('chunks',[])) for c in pushed), 'chunk sizes', collections.Counter(x for c in pushed for x in c.get('chunks',[])))
    print(' rows pushed', sum(c.get('rows',0) for c in pushed), 'answers', sum(len(c.get('answers',[])) if isinstance(c.get('answers'),list) else c.get('answers',0) for c in calls))
    print(' chunk_size', e.get('chunk_size'), 'batch_size', e.get('batch_size'), 'complete', collections.Counter(c.get('complete') for c in calls))
    print(' other keys', {k:v for k,v in e.items() if k not in ('calls','load')} )
