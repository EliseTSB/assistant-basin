"""Tally the Part 218 v4 dev screen with the registered classifier (attrib218.classify on opening + continuation)."""
import collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import attrib218  # noqa: E402

OPENINGS = ['', 'I was created by', 'I was developed by', 'I was built by', 'My creators are']


def main(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
    table = collections.defaultdict(collections.Counter)
    by = {}
    for r in rows:
        text = (r['opening'] + ' ' + r['continuation'].lstrip()) if r['opening'] else r['continuation']
        state, fam = attrib218.classify(text)
        key = state if state not in ('other_maker', 'refusal') else f'{state}:{fam}'
        table[(r['stage'], r['opening'])][key] += 1
        by[(r['stage'], r['opening'], r['prompt_index'])] = state
    stages = sorted({s for s, _ in table}, key=['SFT', 'DPO', 'Instruct'].index)
    out = {'per_cell': {}, 'rotated': {}}
    for s in stages:
        for o in OPENINGS:
            c = table.get((s, o))
            if c is None:
                continue
            n = sum(c.values())
            out['per_cell'][f'{s} | {o or "(none)"}'] = dict(c, n=n)
            print(f'{s:8} {o or "(no opening)":20} n={n:3} ' + ', '.join(f'{k} {v}' for k, v in c.most_common()))
        n_prompts = max(p for (st, _, p) in by if st == s) + 1
        rot = collections.Counter(by.get((s, OPENINGS[1 + i % 4], i)) for i in range(n_prompts))
        out['rotated'][s] = dict(rot, n=n_prompts)
        print(f'{s:8} ROTATED (prompt i gets opening 1 + i%4): self {rot.get("self", 0)}/{n_prompts}  {dict(rot)}')
    return out


if __name__ == '__main__':
    res = main(sys.argv[1])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1), encoding='utf-8')
