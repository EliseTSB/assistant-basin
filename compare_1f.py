#!/usr/bin/env python3
"""Compare the recorded Clef replay with the frozen 1e Jev TEST result.

Standard output is five numeric-only rows, in prediction order: (Jev mean,
Clef mean, absolute delta, delta limit, pass); (Pearson r, minimum r, pass);
(Clef real median, Clef random median, Clef gap, 1e gap, allowed gap delta,
pass); (Jev SD, Clef SD, SD delta, minimum delta, pass); (requests/second,
minimum, pass).
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'climb'))
import climb_eval as ce  # use the 1e tie handling and per-unit scorer


def read_json(path):
    with open(path, encoding='utf-8') as src:
        return json.load(src)


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSONL: {exc}') from exc
    return rows


def jev_cache_key(state, questions):
    payload = json.dumps(('jev', questions, state), sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_cached_probs(cache_dir, state, questions):
    key = jev_cache_key(state, questions)
    path = Path(cache_dir) / key[:2] / f'{key}.json'
    try:
        cached = read_json(path)
    except FileNotFoundError as exc:
        raise ValueError(f'missing original Jev cache entry {key} ({path})') from exc
    probs = cached.get('probs')
    if not isinstance(probs, dict) or any(q not in probs for q in questions):
        raise ValueError(f'original Jev cache entry {key} has incomplete question probabilities')
    return key, {q: float(probs[q]) for q in questions}


def population_sd(values):
    if not values:
        raise ValueError('cannot compute SD without probabilities')
    return statistics.pstdev(values)


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError('need at least two paired per-unit Spearmans for Pearson correlation')
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if not den:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / den


def reconstruct_jev_rows(rows, requests, jev_cache_dir):
    """Recover per-unit Jev Spearmans from 1e's request cache and the replay row sidecar."""
    unit_rows = rows.get('units', rows)
    by_pair = {}
    for uid, unit in unit_rows.items():
        for item in unit.get('items', []):
            pair = (unit.get('description', ''), item['context'])
            by_pair.setdefault(pair, []).append((uid, item))

    jev_items = {uid: [] for uid in unit_rows}
    jev_probabilities = []
    seen_request_hashes = set()
    for request in requests:
        state, questions = request['state'], request['questions']
        cache_key, probs = load_cached_probs(jev_cache_dir, state, questions)
        jev_probabilities.extend(probs.values())
        # c23 has one weighted noul question (fires), so its score is this sole p(true).
        if len(probs) != 1:
            raise ValueError('c23 per-unit reconstruction expects its single fires noul question')
        p = round(next(iter(probs.values())), 4)
        pair = (state.get('description', ''), state.get('snippet', ''))
        matches = by_pair.get(pair, [])
        if not matches:
            raise ValueError('recorded Jev request did not match a Clef TEST row item')
        digest = request.get('hash') or request.get('request_hash')
        # The scorer's Jev cache collapses duplicate state/question requests; each unique
        # response is then reused for every matching held-out item.
        occurrence = digest or cache_key
        if occurrence in seen_request_hashes:
            continue
        seen_request_hashes.add(occurrence)
        for uid, item in matches:
            jev_items[uid].append((item['act'], p))

    # Each row item has one recorded request. Compare recovered values against the known TEST structure.
    jev_spearmans = {}
    for uid, unit in unit_rows.items():
        pairs = jev_items.get(uid, [])
        expected = len(unit.get('items', []))
        if len(pairs) != expected:
            raise ValueError(f'{uid}: matched {len(pairs)} Jev items, expected {expected}')
        jev_spearmans[uid] = ce.spearman([p for _, p in pairs], [act for act, _ in pairs])
    return jev_spearmans, jev_probabilities


def clef_spearmans(rows):
    unit_rows = rows.get('units', rows)
    return {uid: ce.spearman([item['p'] for item in unit.get('items', [])],
                             [item['act'] for item in unit.get('items', [])])
            for uid, unit in unit_rows.items()}


def top_auc_medians_by_arm(rows):
    unit_rows = rows.get('units', rows)
    values = {'N': [], 'R': []}
    for unit in unit_rows.values():
        arm = unit.get('arm')
        if arm not in values:
            continue
        top = [item['p'] for item in unit.get('items', []) if item.get('kind') == 'top']
        random = [item['p'] for item in unit.get('items', []) if item.get('kind') == 'random']
        values[arm].append(ce.auroc(top, random))
    if not values['N'] or not values['R']:
        raise ValueError('per-unit rows must include TEST real (N) and random (R) arms')
    return {arm: ce.median(scores) for arm, scores in values.items()}


def parse_throughput(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'requests_per_second=([0-9]+(?:\.[0-9]+)?)', text)
    if not matches:
        raise ValueError(f'no THROUGHPUT requests_per_second line found in {path}')
    return float(matches[-1])


def compare(jev_scores_path, clef_scores_path, rows_path, requests_path, answers_path,
            jev_cache_dir, gpu_log_path):
    jev_scores = read_json(jev_scores_path)
    clef_scores = read_json(clef_scores_path)
    rows = read_json(rows_path)
    requests = read_jsonl(requests_path)
    answers = {}
    for row in read_jsonl(answers_path):
        digest = row.get('hash') or row.get('request_hash')
        if not digest or not isinstance(row.get('answers'), dict):
            raise ValueError('Clef answers JSONL rows need hash and answers object')
        answers[digest] = row['answers']

    jev_mean = float(jev_scores['test']['pipe']['mean_spearman'])
    clef_mean = float(clef_scores['test']['pipe']['mean_spearman'])
    jev_per_unit, jev_probabilities = reconstruct_jev_rows(rows, requests, jev_cache_dir)
    clef_per_unit = clef_spearmans(rows)
    common = sorted(set(jev_per_unit) & set(clef_per_unit))
    if len(common) < 2:
        raise ValueError('fewer than two common TEST units for per-unit correlation')
    paired_r = pearson([jev_per_unit[u] for u in common], [clef_per_unit[u] for u in common])

    medians = top_auc_medians_by_arm(rows)
    real_median = float(medians['N'])
    random_median = float(medians['R'])
    clef_gap = real_median - random_median
    jev_medians = jev_scores['test']['pipe_auroc_top_median_by_arm']
    reference_gap = float(jev_medians['N']) - float(jev_medians['R'])

    clef_probabilities = []
    for request in requests:
        digest = request.get('hash') or request.get('request_hash')
        if digest not in answers:
            raise ValueError(f'Clef answer missing for recorded request {digest}')
        values = answers[digest]
        if set(request['questions']) - set(values):
            raise ValueError(f'Clef answer {digest} is missing a requested question')
        clef_probabilities.extend(float(values[q]) for q in request['questions'])
    jev_sd = population_sd(jev_probabilities)
    clef_sd = population_sd(clef_probabilities)
    rps = parse_throughput(gpu_log_path)

    delta = abs(clef_mean - jev_mean)
    gap_delta = abs(clef_gap - reference_gap)
    results = [
        [jev_mean, clef_mean, delta, 0.05, int(delta <= 0.05)],
        [paired_r, 0.6, int(paired_r >= 0.6)],
        [real_median, random_median, clef_gap, reference_gap, gap_delta, 0.05,
         int(real_median > random_median and gap_delta <= 0.05)],
        [jev_sd, clef_sd, clef_sd - jev_sd, 0.0, int(clef_sd >= jev_sd)],
        [rps, 5.0, int(rps >= 5.0)],
    ]
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jev-scores', required=True)
    ap.add_argument('--clef-scores', required=True)
    ap.add_argument('--rows', required=True, help='per-unit rows emitted by climb_test.py --rows-out')
    ap.add_argument('--requests', required=True, help='record-mode requests.jsonl')
    ap.add_argument('--answers', required=True, help='Clef answers.jsonl')
    ap.add_argument('--jev-cache', required=True, help='read-only work-test/cache/jev directory')
    ap.add_argument('--gpu-log', required=True, help='full A770 run log with THROUGHPUT line')
    a = ap.parse_args(argv)
    for row in compare(a.jev_scores, a.clef_scores, a.rows, a.requests,
                       a.answers, a.jev_cache, a.gpu_log):
        print(' '.join(f'{value:.6f}' if isinstance(value, float) else str(value) for value in row))


if __name__ == '__main__':
    main()
