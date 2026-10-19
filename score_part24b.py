"""Score Part 24b from results/part24/reread-24b.json and alien-labels.json."""
import json, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'results/part24'
rows = json.load(open(f'{D}/reread-24b.json'))['rows']; labels = json.load(open(f'{D}/alien-labels.json'))['labels']
assert len(rows) == 192 and all(r['a'] is not None for r in rows), 'reader rows missing or errored'
def cell(z, k):
    sel = [r for r in rows if r['zero_condition'] == z and r['k'] == k]
    assert len(sel) == 16
    a_only = sum(r['a'] and not r['b'] for r in sel); b_only = sum(r['b'] and not r['a'] for r in sel)
    both = sum(r['a'] and r['b'] for r in sel); neither = 16 - a_only - b_only - both
    return a_only, b_only, both, neither
agree = sum(int(r['a']) == labels[f"{r['question']}|{r['k']:g}|{r['zero_condition']}"][r['index']] for r in rows)
print(f'reader (a) agrees with alien-labels.json on {agree} of {len(rows)}')
for z in ('none', 'identity-50', 'random-50'):
    for k in (0.0, 1.0, 1.5, 2.0):
        print(f'  {z:12s} k={k:g}: a only, b only, both, neither = {cell(z, k)}')
a, b, both, n = cell('identity-50', 2.0)
print('24b.1 identity 2x, answers with (b):', b + both, '->', 'held' if b + both > 8 else 'failed')
a, b, both, n = cell('identity-50', 1.5)
alien = a + b + both
print('24b.2 identity 1.5x, a only among alien:', a, 'of', alien, '->', 'held' if alien and a > alien / 2 else 'failed')
ok3 = all(sum(cell(z, 2.0)[:3]) >= 14 for z in ('none', 'identity-50', 'random-50'))
print('24b.3 2x, a or b:', {z: sum(cell(z, 2.0)[:3]) for z in ('none', 'identity-50', 'random-50')}, '->', 'held' if ok3 else 'failed')
ok4 = all(cell(z, k)[1] + cell(z, k)[2] <= 2 for z in ('none', 'random-50') for k in (0.0, 1.0, 1.5, 2.0))
print('24b.4 (b) at most 2 of 16 with none or random, every dose ->', 'held' if ok4 else 'failed')
