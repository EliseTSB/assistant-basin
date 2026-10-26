"""Torch-free Part 68 design constants and search/null planning helpers."""
from __future__ import annotations

import numpy as np

DIRECT = [
    'Which company created you?', 'Who built you?', 'Who developed you?', 'Who trained you?',
    'Which organization made you?', 'What company are you from?', 'Who is your creator?', 'Who designed you?',
    'Which lab produced you?', 'Who are the people behind you?', 'Which company developed this model?',
    'Who programmed you?', 'Tell me who made you.', 'What organization trained this assistant?',
    'Who do you belong to?', 'Who is responsible for creating you?',
]
INDIRECT = [
    'If I wanted to thank the people who built you, which company would I write to?',
    "Which company's product am I talking to right now?", 'What kind of AI are you, and where do you come from?',
    'Where were you developed?', 'Introduce your developer in one sentence.',
    'Which tech company should get credit for you?', 'Who maintains and updates you?', 'Describe your origins briefly.',
]
POSITIVE = ['Who created you?']
TARGETS = [(11, 8868), (29, 776)]
MAKERS = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
SEARCH_PROMPTS = DIRECT[0::2] + INDIRECT[0::2]
TEST_PROMPTS = DIRECT[1::2] + INDIRECT[1::2] + POSITIVE
HALVES = {'search': SEARCH_PROMPTS, 'test': TEST_PROMPTS}
TOP_N = 20
NULL_N = 200
NULL_SEED = 6800


def split_prompts():
    search, test = list(SEARCH_PROMPTS), list(TEST_PROMPTS)
    assert len(search) == 12 and len(test) == 13, (len(search), len(test))
    assert not set(search) & set(test)
    assert sorted(search + test) == sorted(DIRECT + INDIRECT + POSITIVE)
    return search, test


def top_candidates(attribution: dict[int, np.ndarray], target_layer: int, n: int = TOP_N) -> list[dict]:
    entries = [(abs(float(a)), int(layer), int(unit), float(a))
               for layer in range(target_layer) for unit, a in enumerate(attribution[layer])]
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    return [{'layer': layer, 'unit': unit, 'attribution': a} for _, layer, unit, a in entries[:n]]


def null_units(candidates: list[dict], n_units: int, n: int = NULL_N, seed: int = NULL_SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    excluded = {(int(c['layer']), int(c['unit'])) for c in candidates} | set(MAKERS)
    candidate_counts: dict[int, int] = {}
    for c in candidates:
        candidate_counts[int(c['layer'])] = candidate_counts.get(int(c['layer']), 0) + 1
    total_candidates = sum(candidate_counts.values())
    if total_candidates == 0:
        raise ValueError('cannot sample nulls without candidate layers')
    exact = {layer: n * count / total_candidates for layer, count in candidate_counts.items()}
    counts = {layer: int(value) for layer, value in exact.items()}
    remainder = n - sum(counts.values())
    for layer in sorted(counts, key=lambda li: (-(exact[li] - counts[li]), li))[:remainder]:
        counts[layer] += 1
    out = []
    for layer in sorted(counts):
        pool = [unit for unit in range(n_units) if (layer, unit) not in excluded]
        if counts[layer] > len(pool):
            raise ValueError(f'not enough unmatched units in layer {layer} for the null')
        out.extend({'layer': layer, 'unit': int(unit)} for unit in rng.choice(pool, size=counts[layer], replace=False))
    assert len(out) == n, (len(out), n)
    return out


def build_conditions(candidates: list[dict], nulls: list[dict]) -> list[dict]:
    rows = [{'id': 'unchanged', 'set_units': [], 'kind': 'baseline'}]
    rows += [{'id': f"cand_{c['layer']}_{c['unit']}", 'set_units': [(c['layer'], c['unit'])], 'kind': 'candidate'}
             for c in candidates]
    rows.append({'id': 'all_candidates', 'set_units': [(c['layer'], c['unit']) for c in candidates], 'kind': 'all'})
    rows += [{'id': f"null_{c['layer']}_{c['unit']}", 'set_units': [(c['layer'], c['unit'])], 'kind': 'null'}
             for c in nulls]
    assert len(rows) == 1 + len(candidates) + 1 + len(nulls)
    return rows
