#!/usr/bin/env python3
"""Part 64 collection assembly and provenance gate.

The model traversal uses atlas_collect_fast2's low-level helpers in the rented
runner; this module owns deterministic selection, pilot inclusion, and refusal
checks so these contracts can be tested without loading a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import atlas_collect_fast2 as atlas
import part60.gen60 as gen60

CORPUS = Path('/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl')
CORPUS_SIZE = 63_385_283
CORPUS_SHA256_PREFIX = '56ad98d5a82b2bb4'
CAPACITIES = {'top': 36, 'random': 18, 'weak': 9}
PILOT_COUNTS = {'top': 12, 'random': 6, 'weak': 3}
SEED = 6401
# Amendment 5: the Part 56 pilot scanned the first 1,000 documents of the corpus, so the fresh 3x collection does too.
PILOT_DOCS = 1000


class CollectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collection_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def verify_corpus(path: Path = CORPUS) -> dict[str, Any]:
    if not path.is_file():
        raise CollectionError(f'registered corpus is missing: {path}')
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != CORPUS_SIZE or not digest.startswith(CORPUS_SHA256_PREFIX):
        raise CollectionError(f'refusing unregistered corpus: size={size}, sha256={digest}')
    return {'path': str(path), 'size_bytes': size, 'sha256': digest}


def unit_index() -> dict[str, dict[str, Any]]:
    """Return Part 60's exact 500 dev unit records, using its canonical split."""
    selected = gen60.load_splits()['dev']
    if len(selected) != 500 or sum(x['type'] == 'neuron' for x in selected) != 250:
        raise CollectionError('Part 60 dev split did not resolve to 250 neurons and 250 controls')
    return {row['unit_key']: row for row in selected}


def _window(row: dict[str, Any]) -> str:
    value = row.get('window')
    if not isinstance(value, str) or not value:
        raise CollectionError('window row must contain nonempty window text')
    return value


def _ranked_unique(candidates: list[dict[str, Any]], excluded: set[str], limit: int) -> list[dict[str, Any]]:
    out = []
    seen = set(excluded)
    for item in candidates:
        value = _window(item)
        if value in seen:
            continue
        seen.add(value)
        out.append(item)
        if len(out) == limit:
            break
    return out


def assemble_unit(unit: dict[str, Any], fresh: dict[str, list[dict[str, Any]]], seed: int = SEED) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Include exact M1 windows and fill 3x strata from deduplicated fresh data.

    `fresh` has atlas-style top candidates and fresh random/weak candidates.
    Random and weak additions use stable per-unit shuffles matching a recorded
    seed, while top candidates retain descending activation order.
    """
    pilot = unit['record']
    result: dict[str, list[dict[str, Any]]] = {}
    for category, count in PILOT_COUNTS.items():
        base = pilot.get(category, [])[:count]
        if len(base) != count:
            raise CollectionError(f"{unit['unit_key']} pilot lacks {count} {category} windows")
        result[category] = list(base)
        extras = list(fresh.get(category, []))
        if category == 'top':
            extras.sort(key=lambda x: float(x.get('activation', float('-inf'))), reverse=True)
        else:
            rng = random.Random(SEED ^ seed ^ int(hashlib.sha256(unit['unit_key'].encode()).hexdigest()[:8], 16) ^ (0 if category == 'random' else 1))
            rng.shuffle(extras)
        result[category].extend(_ranked_unique(extras, {_window(x) for x in base}, CAPACITIES[category] - count))
        if len(result[category]) != CAPACITIES[category]:
            raise CollectionError(f"{unit['unit_key']} has only {len(result[category])}/{CAPACITIES[category]} {category} windows")
        if not {_window(x) for x in base}.issubset({_window(x) for x in result[category]}):
            raise CollectionError(f"1x subset check failed for {unit['unit_key']} {category}")
    fresh_top = fresh.get('top', [])
    pilot_top = {_window(x) for x in pilot['top'][:12]}
    reproduced = len(pilot_top & {_window(x) for x in fresh_top[:12]})
    return result, reproduced


def assemble(units: dict[str, dict[str, Any]], fresh_rows: dict[str, dict[str, list[dict[str, Any]]]],
             *, corpus: Path = CORPUS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_meta = verify_corpus(corpus)
    if set(units) != set(fresh_rows):
        raise CollectionError('fresh collection unit keys differ from the canonical Part 60 dev set')
    output = []
    reproduction = []
    for key, rows in fresh_rows.items():
        outside = sorted({int(x['doc']) for cat in rows.values() for x in cat if int(x['doc']) >= PILOT_DOCS})
        if outside:
            raise CollectionError(f'{key} has fresh windows outside the pilot documents 0-{PILOT_DOCS - 1} (Amendment 5): {outside[:5]}')
    for key, unit in units.items():
        strata, top_match = assemble_unit(unit, fresh_rows[key])
        output.append({'unit_key': key, 'type': unit['type'], 'layer': unit['layer'],
                       'unit': unit['record'].get('unit'), 'control_id': unit['record'].get('control_id'),
                       'strata': strata, 'fresh_top12_match': top_match})
        reproduction.append(top_match)
    median = statistics.median(reproduction)
    if median < 10:
        raise CollectionError(f'reproduction stop: median fresh top-12 match is {median}/12 (minimum 10)')
    manifest = {'version': 1, 'corpus': corpus_meta, 'model': 'Qwen/Qwen3-1.7B',
                'model_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                'docs': PILOT_DOCS, 'max_doc_tokens': atlas.MAX_DOC_TOKENS,
                'window_tokens': atlas.WINDOW_TOKENS, 'firing_offset': atlas.FIRING_OFFSET,
                'capacities': CAPACITIES, 'pilot_counts': PILOT_COUNTS,
                'unit_count': len(output), 'median_fresh_top12_match': median,
                'subset_verified': True, 'seed': SEED}
    manifest['collection_sha256'] = hashlib.sha256(collection_jsonl(output)).hexdigest()
    return output, manifest


def load_fresh(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with path.open(encoding='utf-8') as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('unit_key')
            if not isinstance(key, str) or key in rows:
                raise CollectionError(f'{path}:{n}: missing or duplicate unit_key')
            strata = row.get('strata')
            if not isinstance(strata, dict):
                raise CollectionError(f'{path}:{n}: strata object required')
            rows[key] = strata
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh', type=Path, required=True, help='fresh atlas collection records, one unit per JSONL row')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--corpus', type=Path, default=CORPUS)
    args = p.parse_args()
    units = unit_index()
    rows, manifest = assemble(units, load_fresh(args.fresh), corpus=args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + '.tmp')
    tmp.write_bytes(collection_jsonl(rows))
    tmp.replace(args.out)
    args.out.with_suffix(args.out.suffix + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
