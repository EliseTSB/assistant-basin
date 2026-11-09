#!/usr/bin/env python3
"""existing201.py: what the records already on disk say about batch-shape noise, before Part 201 runs anything.

Reads three published files and prints every number PART201-DESIGN.md and BATCH-SHAPE-NOISE.md cite from them:

- results/part200/report200.json: the 200 batch-1 answers against the same units' chunked answers (Part 200).
- results/part183/kernel-check.json: the A770's bf16 eager decode of the eight kernel-check prompts at batch 1, 2, 8.
- results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json: the CPU float32 reference for the same prompts.

  python -I -B part201/existing201.py [--out results/part201]

Writes existing.json and existing.md into --out. No model, no GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def first_difference(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where two token lists differ, None when one is a prefix of the other."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def quantile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def part200(report: dict) -> dict:
    rows = report['batch1']['positions']['rows']
    labels = {}
    for name in ('name_gone', 'broken', 'qwen_part161_rule'):
        counts = collections.Counter()
        for row in rows:
            lab = row['labels'][name]
            if lab['changed']:
                counts['chunked_only' if lab['chunked'] else 'batch1_only'] += 1
        labels[name] = {'flips': sum(counts.values()), 'true_in_chunked_only': counts['chunked_only'],
                        'true_in_batch1_only': counts['batch1_only']}
    first = sorted(row['first_answer_difference'] for row in rows)
    delta = [row['batch1_length'] - row['chunked_length'] for row in rows]
    return {'rows_compared': report['batch1']['compare']['rows'], 'rows_changed': len(rows), 'label_flips': labels,
            'first_answer_difference': {'min': first[0], 'median': statistics.median(first), 'max': first[-1],
                                        'at_or_before_answer_token_32': sum(p < 32 for p in first)},
            'length_batch1_minus_chunked': {'mean': round(statistics.mean(delta), 2), 'longer': sum(d > 0 for d in delta),
                                            'shorter': sum(d < 0 for d in delta), 'equal': sum(d == 0 for d in delta)}}


def kernel(reference: dict, candidate: dict) -> dict:
    ref1 = {r['prompt']: r for r in reference['runs']['1']}
    ref2 = {r['prompt']: r for r in reference['runs']['2']}
    ref_margins = [m for r in ref1.values() for m in r['margins']]
    out = {'reference': {
        'positions': len(ref_margins),
        'share_margin_below': {str(t): round(sum(m < t for m in ref_margins) / len(ref_margins), 4)
                               for t in (0.0625, 0.125, 0.25, 0.5, 1.0)},
        'rows_with_a_margin_below_0.125': sum(min(r['margins']) < 0.125 for r in ref1.values()),
        'rows': len(ref1),
        'batch1_vs_batch2_rows_identical': sum(ref1[p]['tokens'] == ref2[p]['tokens'] for p in ref1),
        'batch1_vs_batch2_max_margin_difference': max(abs(a - b) for p in ref1
                                                      for a, b in zip(ref1[p]['margins'], ref2[p]['margins']))}}
    shared, divergences, per_batch = [], [], {}
    for batch, rows in candidate['runs'].items():
        errors = []
        for row in rows:
            ref = ref1[row['prompt']]
            n = first_difference(row['tokens'], ref['tokens'])
            last = min(len(row['margins']), len(ref['margins'])) - 1 if n is None else n
            for i in range(last + 1):          # positions decided from an identical prefix, the first difference included
                errors.append(abs(row['margins'][i] - ref['margins'][i]))
                shared.append(row['margins'][i])
            if n is not None:
                divergences.append({'batch': batch, 'prompt': row['prompt'], 'position': n,
                                    'reference_margin': round(ref['margins'][n], 4),
                                    'candidate_margin': round(row['margins'][n], 4)})
        errors.sort()
        per_batch[batch] = {'positions': len(errors), 'median': round(statistics.median(errors), 4),
                            'p90': round(quantile(errors, 0.9), 4), 'p99': round(quantile(errors, 0.99), 4),
                            'max': round(errors[-1], 4)}
    by_batch = {b: {r['prompt']: r for r in rows} for b, rows in candidate['runs'].items()}
    pairs = {}
    for a, b in (('1', '2'), ('1', '8'), ('2', '8')):
        if a in by_batch and b in by_batch:
            pairs[f'{a}_vs_{b}'] = sum(first_difference(by_batch[a][p]['tokens'], by_batch[b][p]['tokens']) is not None
                                       or len(by_batch[a][p]['tokens']) != len(by_batch[b][p]['tokens'])
                                       for p in by_batch[a] if p in by_batch[b])
    out['candidate'] = {
        'margin_error_against_reference_by_batch': per_batch,
        'positions_from_identical_prefix': len(shared),
        'exact_ties': sum(m == 0 for m in shared), 'one_step_0.125': sum(m == 0.125 for m in shared),
        'at_most_0.125': sum(m <= 0.125 for m in shared), 'at_most_0.25': sum(m <= 0.25 for m in shared),
        'margins_multiple_of_0.0625': sum(abs(m / 0.0625 - round(m / 0.0625)) < 1e-9 for m in shared),
        'divergences_from_reference': divergences,
        'largest_candidate_margin_at_a_divergence': max(d['candidate_margin'] for d in divergences),
        'largest_reference_margin_at_a_divergence': max(d['reference_margin'] for d in divergences),
        'rows_differing_between_batches': pairs, 'prompts': len(by_batch['1'])}
    return out


def render(result: dict) -> str:
    p, k = result['part200'], result['kernel']
    ref, cand = k['reference'], k['candidate']
    lines = ['# What the records already on disk say (printed by part201/existing201.py)', '',
             '## Part 200: batch 1 against chunks of 8, one question, 200 pushed rows', '',
             f"- {p['rows_changed']} of {p['rows_compared']} answers differ.",
             f"- First difference at answer token: min {p['first_answer_difference']['min']}, median "
             f"{p['first_answer_difference']['median']}, max {p['first_answer_difference']['max']}; "
             f"{p['first_answer_difference']['at_or_before_answer_token_32']} before answer token 32.",
             f"- Length, batch 1 minus chunked: mean {p['length_batch1_minus_chunked']['mean']} tokens; "
             f"{p['length_batch1_minus_chunked']['longer']} longer, {p['length_batch1_minus_chunked']['shorter']} "
             f"shorter, {p['length_batch1_minus_chunked']['equal']} equal.", '',
             '| Label | Rows that flip | True in chunked only | True in batch 1 only |', '| --- | ---: | ---: | ---: |']
    for name, lab in p['label_flips'].items():
        lines.append(f"| {name} | {lab['flips']} | {lab['true_in_chunked_only']} | {lab['true_in_batch1_only']} |")
    lines += ['', '## CPU float32 reference, eight plain prompts, 96 tokens', '',
              f"- Batch 1 against batch 2: {ref['batch1_vs_batch2_rows_identical']} of {ref['rows']} rows identical; "
              f"largest margin difference {ref['batch1_vs_batch2_max_margin_difference']:.2e}.",
              f"- Share of its {ref['positions']} positions with a top-two margin below 0.0625, 0.125, 0.25, 0.5, 1.0: "
              + ', '.join(f"{v:.1%}" for v in ref['share_margin_below'].values()) + '.',
              f"- {ref['rows_with_a_margin_below_0.125']} of {ref['rows']} rows have at least one margin below 0.125.",
              '', '## A770 bf16 eager (Part 183 kernel check) against that reference', '',
              '| Batch | Positions decided from an identical prefix | Median margin error | p90 | p99 | Max |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for batch, e in cand['margin_error_against_reference_by_batch'].items():
        lines.append(f"| {batch} | {e['positions']} | {e['median']} | {e['p90']} | {e['p99']} | {e['max']} |")
    lines += ['', f"- Of {cand['positions_from_identical_prefix']} such positions, {cand['exact_ties']} are exact ties "
              f"(margin 0) and {cand['one_step_0.125']} sit at 0.125; {cand['at_most_0.125']} "
              f"({cand['at_most_0.125'] / cand['positions_from_identical_prefix']:.1%}) are at 0.125 or less and "
              f"{cand['at_most_0.25']} ({cand['at_most_0.25'] / cand['positions_from_identical_prefix']:.1%}) at 0.25 or "
              f"less; {cand['margins_multiple_of_0.0625']} are multiples of 0.0625.",
              f"- Divergences from the reference: {len(cand['divergences_from_reference'])}; the largest bf16 margin at "
              f"one is {cand['largest_candidate_margin_at_a_divergence']}, the largest reference margin "
              f"{cand['largest_reference_margin_at_a_divergence']}.",
              '- Rows whose tokens differ between batch sizes, of ' + str(cand['prompts']) + ': '
              + ', '.join(f"{k_} {v}" for k_, v in cand['rows_differing_between_batches'].items()) + '.', '',
              '| Batch | Prompt | Position | Reference margin | bf16 margin |', '| --- | ---: | ---: | ---: | ---: |']
    for d in cand['divergences_from_reference']:
        lines.append(f"| {d['batch']} | {d['prompt']} | {d['position']} | {d['reference_margin']} | {d['candidate_margin']} |")
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=str(ROOT / 'results' / 'part201'))
    ap.add_argument('--report200', default=str(ROOT / 'results' / 'part200' / 'report200.json'))
    ap.add_argument('--candidate', default=str(ROOT / 'results' / 'part183' / 'kernel-check.json'))
    ap.add_argument('--reference', default=str(ROOT / 'results' / 'kernel-reference' / 'qwen-qwen3.5-4b-cpu-float32.json'))
    a = ap.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    result = {'part200': part200(load(a.report200)), 'kernel': kernel(load(a.reference), load(a.candidate))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'existing.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (out / 'existing.md').write_text(render(result), encoding='utf-8')
    print(render(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
