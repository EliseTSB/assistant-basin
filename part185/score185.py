#!/usr/bin/env python3
"""Part 185 scoring (registered in PART185-DESIGN.md): the gate, predictions 185.1-185.6 from the originals' own scoring
tools' outputs (results84.json from part84/blind84.py score, results84b.json from part84b/score84b.py score), and the
descriptive comparison with the stored SDPA runs (same arm or cell, question and seed: identical text or not, the first
differing answer token's key position, and the SDPA and eager first-reader labels side by side).

Usage: score185.py --a DIR --b DIR [--out FILE] [--tokenizer SNAPSHOT]. Without --tokenizer the key positions are
skipped. Written by Claude (Opus 5.5)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB_SHA256 = 'afb86522116b9ccd10233826503dad0992dd0283f04ee2a15dabf5f0e9e1ce54'
PUBLISHED_B = {'A0': 0, 'decode0.40': 4, 'decode0.47': 25, 'decode0.55': 43, 'gen0.40': 3, 'gen0.47': 23, 'gen0.55': 46}
DOSES = ('0.40', '0.47', '0.55')


def gate(run):
    load = run.get('load') or {}
    return {'kernel': load.get('attn_implementation'), 'calib_ok': run.get('calib_sha256') == CALIB_SHA256,
            'passed': load.get('attn_implementation') == 'eager' and run.get('calib_sha256') == CALIB_SHA256}


def predictions(a_counts, b_res):
    off = lambda arm: a_counts[arm]['OFFWORLD']
    bo = b_res['offworld']
    dec = [bo[f'decode{d}'] for d in DOSES]; gen = [bo[f'gen{d}'] for d in DOSES]
    ps = {d: b_res['paired'][d]['sign_test_p'] for d in DOSES}
    moves = {c: bo[c] - PUBLISHED_B[c] for c in PUBLISHED_B}
    v = {}
    v['185.1'] = ('held' if off('A2') >= 22 and off('A3') >= 22 else 'failed', f"A2 {off('A2')}, A3 {off('A3')} of 24")
    v['185.2'] = ('held' if off('A0') <= 1 and off('C') <= 1 else 'failed', f"A0 {off('A0')}, C {off('C')} of 24")
    v['185.3'] = ('held' if 8 <= off('A1') <= 22 else 'failed', f"A1 {off('A1')} of 24")
    v['185.4'] = ('held' if dec[0] < dec[1] < dec[2] and gen[0] < gen[1] < gen[2] else 'failed', f'decode {dec}, gen {gen} of 48')
    v['185.5'] = ('held' if all(p >= 0.05 for p in ps.values()) else 'failed',
                  ', '.join(f"{d}: {b_res['paired'][d]['gen_only']}/{b_res['paired'][d]['decode_only']} p={ps[d]:.3g}" for d in DOSES))
    v['185.6'] = ('held' if all(abs(m) <= 8 for m in moves.values()) else 'failed',
                  'moves ' + ', '.join(f'{c} {m:+d}' for c, m in moves.items()))
    return v


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def labels_by_seed(labels_path, key_path, unit_field):
    key = json.loads(Path(key_path).read_text(encoding='utf-8'))
    return {(key[r['id']][unit_field], key[r['id']]['question_index'], key[r['id']]['sample']): r['origin']
            for r in read_jsonl(labels_path)}


def compare(new_records, old_records, unit_field, new_labels, old_labels, tok=None, prompt_len=None):
    old = {(r[unit_field], r['question_index'], r['sample']): r for r in old_records}
    rows = []
    for r in new_records:
        k = (r[unit_field], r['question_index'], r['sample'])
        o = old[k]
        row = {'unit': k[0], 'question_index': k[1], 'sample': k[2], 'identical': r['answer'] == o['answer'],
               'sdpa_label': old_labels.get(k), 'eager_label': new_labels.get(k)}
        if tok is not None and not row['identical']:
            a = tok(r['answer'], add_special_tokens=False)['input_ids']; b = tok(o['answer'], add_special_tokens=False)['input_ids']
            j = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            row['first_difference_key'] = prompt_len[r['question_index']] + j
        rows.append(row)
    units = sorted({r['unit'] for r in rows})
    summ = {}
    for u in units:
        rs = [r for r in rows if r['unit'] == u]
        summ[u] = {'n': len(rs), 'identical': sum(r['identical'] for r in rs),
                   'label_changed': sum(r['sdpa_label'] != r['eager_label'] for r in rs),
                   'sdpa_off_eager_not': sum(r['sdpa_label'] == 'OFFWORLD' and r['eager_label'] != 'OFFWORLD' for r in rs),
                   'eager_off_sdpa_not': sum(r['eager_label'] == 'OFFWORLD' and r['sdpa_label'] != 'OFFWORLD' for r in rs)}
        keys = [r['first_difference_key'] for r in rs if 'first_difference_key' in r]
        if keys:
            summ[u]['differ_before_key_65'] = sum(k <= 64 for k in keys)
            summ[u]['median_first_difference_key'] = sorted(keys)[len(keys) // 2]
    return summ, rows


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--out'); ap.add_argument('--tokenizer')
    a = ap.parse_args(argv)
    A, B = Path(a.a), Path(a.b)
    out = {'gate': {'A': gate(json.loads((A / 'run.json').read_text(encoding='utf-8'))),
                    'B': gate(json.loads((B / 'run.json').read_text(encoding='utf-8')))}}
    a_res = json.loads((A / 'results84.json').read_text(encoding='utf-8'))
    b_res = json.loads((B / 'results84b.json').read_text(encoding='utf-8'))
    out['counts'] = {'A': a_res['counts'], 'B': {'offworld': b_res['offworld'], 'paired': b_res['paired'],
                                                 'reader_agreement': b_res.get('reader_agreement')}}
    out['predictions'] = predictions(a_res['counts'], b_res) if out['gate']['A']['passed'] and out['gate']['B']['passed'] \
        else {k: ('not read', 'gate failed') for k in ('185.1', '185.2', '185.3', '185.4', '185.5', '185.6')}
    tok = plen = None
    if a.tokenizer:
        sys.path.insert(0, str(ROOT))
        from transformers import AutoTokenizer
        import neuron_report as nr
        from part84.part84 import QUESTIONS
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        plen = [len(tok(nr.build_prompt(tok, q, False), add_special_tokens=False)['input_ids']) for q in QUESTIONS]
    R = ROOT / 'results'
    sa, _ = compare(read_jsonl(A / 'records.jsonl'), read_jsonl(R / 'part84/main/records.jsonl'), 'arm',
                    labels_by_seed(A / 'reading/labels.jsonl', A / 'private/key.json', 'arm'),
                    labels_by_seed(R / 'part84/reading/labels.jsonl', R / 'part84/reading/key.json', 'arm'), tok, plen)
    sb, _ = compare(read_jsonl(B / 'records.jsonl'), read_jsonl(R / 'part84b/main/records.jsonl'), 'cell',
                    labels_by_seed(B / 'reading/labels.jsonl', B / 'private/key.json', 'cell'),
                    labels_by_seed(R / 'part84b/reading/labels.jsonl', R / 'part84b/private/key.json', 'cell'), tok, plen)
    out['text_compare'] = {'A': sa, 'B': sb}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + '\n', encoding='utf-8')
    for k, (vv, d) in out['predictions'].items():
        print(k, vv, d)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
