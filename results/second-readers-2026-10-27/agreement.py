#!/usr/bin/env python3
"""Second-reader agreement for Parts 79, 81 and 84 (assistant-basin), written 2026-10-27 by Claude.

WHAT IT MEASURES: for each part, joins the first blind reader's labels (in the repo's results folder) with the second
reader's (labels.jsonl in each chunk folder here) by item id, and reports per label: items compared, raw agreement,
Cohen's kappa, and the confusion counts; then each condition's count of the headline label under both readers, using
the part's key. It reads the key only after both label sets exist.
WHAT IT DOES NOT DO: decide which reader is right; label anything itself.
Usage: python3 -I agreement.py <assistant-basin checkout> [--out agreement.md]
"""
import json, os, sys, glob
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))

def jl(p):
    out = {}
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if line:
            r = json.loads(line); out[r['id']] = r
    return out

def second(prefix):
    out = {}
    for d in sorted(glob.glob(os.path.join(HERE, prefix + '*'))):
        for f in sorted(glob.glob(os.path.join(d, 'labels*.jsonl'))):
            for k, v in jl(f).items():
                if k in out: raise SystemExit(f'duplicate id {k} in {f}')
                out[k] = v
    return out

def kappa(pairs):
    n = len(pairs)
    if n == 0: return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return po, (None if pe == 1 else (po - pe) / (1 - pe))

def report(name, first, sec, fields, key, cond_of, headline, lines):
    ids = sorted(set(first) & set(sec))
    lines.append(f'## {name}\n')
    lines.append(f'First reader {len(first)} items, second {len(sec)}, compared {len(ids)}; '
                 f'missing from second: {len(set(first) - set(sec))}; extra in second: {len(set(sec) - set(first))}.\n')
    lines.append('| label | compared | agreement | kappa | disagreements (first -> second) |\n|---|---|---|---|---|')
    for f in fields:
        pairs = [(str(first[i].get(f)), str(sec[i].get(f))) for i in ids]
        po, k = kappa(pairs)
        dis = Counter(f'{a}->{b}' for a, b in pairs if a != b).most_common(4)
        lines.append(f'| {f} | {len(pairs)} | {po:.3f} | {"n/a" if k is None else f"{k:.3f}"} | ' + ', '.join(f'{d} {c}' for d, c in dis) + ' |')
    hf, hv = headline
    by = defaultdict(lambda: [0, 0, 0])
    for i in ids:
        c = cond_of(key[i]); by[c][2] += 1
        by[c][0] += str(first[i].get(hf)) == hv; by[c][1] += str(sec[i].get(hf)) == hv
    lines.append(f'\n{hf} = {hv}, by condition (first reader, second reader, of n):\n')
    lines.append('| condition | first | second | n |\n|---|---|---|---|')
    for c in sorted(by): lines.append(f'| {c} | {by[c][0]} | {by[c][1]} | {by[c][2]} |')
    lines.append('')

def main():
    repo = sys.argv[1]; out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    R = os.path.join(repo, 'results'); lines = ['# Second-reader agreement, Parts 79, 81 and 84\n']
    s79 = second('p79-c')
    if s79:
        report('Part 79 (evil recipe)', jl(f'{R}/part79/main/reader.jsonl'), s79, ['villain', 'maker', 'coherent'],
               json.load(open(f'{R}/part79/main/reader_key.json')), lambda k: k[0], ('villain', 'True'), lines)
    s81 = second('p81-c')
    if s81:
        report('Part 81 (conspiracy recipe)', jl(f'{R}/part81/reader/reader_labels.jsonl'), s81,
               ['conspiratorial', 'coherent', 'maker'], json.load(open(f'{R}/part81/reader/reader_key.json')),
               lambda k: k['condition'], ('conspiratorial', 'True'), lines)
    s84 = second('p84')
    if s84:
        key = json.load(open(f'{R}/part84/reading/key.json'))
        for nm, f in (('first reader, raw', 'labels.jsonl'), ('first reader after the registered second pass', 'labels-checked.jsonl')):
            report(f'Part 84 (alien recipe), against the {nm}', jl(f'{R}/part84/reading/{f}'), s84, ['origin', 'form'],
                   key, lambda k: k['arm'], ('origin', 'OFFWORLD'), lines)
    text = '\n'.join(lines); print(text)
    if out: open(out, 'w', encoding='utf-8').write(text + '\n')

if __name__ == '__main__': main()
