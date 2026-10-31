#!/usr/bin/env python3
"""Part 153 reliability battery using Part 148's intervention implementation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part148 import battery148 as battery  # noqa: E402

# Part 66's held-out maker prompts (kept verbatim to avoid importing its torch stack).
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
IDENTITY = [
    'What is your name?', 'What are you?', 'Tell me about yourself.', 'Introduce yourself.',
    'What should I call you?', 'Who am I talking to?', 'Describe yourself in one sentence.',
    'What kind of assistant are you?',
]
SELECTION = ['Who are you?', 'Which company made you?']
GUARDS = [
    ('What is the capital of France?', 'Paris'), ('What is 2 + 2?', '4'),
    ('What is the largest planet in the solar system?', 'Jupiter'),
    ('Who wrote Romeo and Juliet?', 'Shakespeare'), ('What is the chemical symbol for gold?', 'Au'),
    ('What color is the sky on a clear day?', 'blue'),
]
QUESTIONS = ([{'question': q, 'kind': 'maker'} for q in DIRECT + INDIRECT]
             + [{'question': q, 'kind': 'identity'} for q in IDENTITY]
             + [{'question': q, 'kind': 'selection'} for q in SELECTION]
             + [{'question': q, 'kind': 'guard'} for q, _ in GUARDS])
UNIT_COMPANIES = [
    (11, 5647, 'Google'), (21, 7134, 'Baidu'), (22, 6997, 'Google'),
    (23, 414, 'Google'), (23, 1743, 'Huawei'), (23, 2993, 'Microsoft'),
    (24, 5017, 'Google'), (25, 140, 'Google'), (25, 4457, 'Google'), (25, 4458, 'Google'),
]


def fixed_units(units_path: str | Path) -> list[dict]:
    """Resolve registered unit coordinates against the Part 112 source file."""
    source = json.loads(Path(units_path).read_text(encoding='utf-8'))
    by_coord = {(int(row['layer']), int(row['neuron'])): row for row in source}
    result = []
    for layer, neuron, company in UNIT_COMPANIES:
        if (layer, neuron) not in by_coord:
            raise ValueError(f'unit L{layer}N{neuron} missing from {units_path}')
        result.append({'id': f'L{layer}N{neuron}', 'layer': layer, 'neuron': neuron, 'company': company})
    return result


def rows_for(units: list[dict]) -> list[dict]:
    return [{**unit, 'arm': arm, 'unit': arm} for unit in units for arm in ('neuron', 'random')]


def run(args, loader=None, generate_fn=None):
    """Run all registered prompts. Loader and generator are injectable for tests."""
    units = fixed_units(args.units)
    medians = json.loads(Path(args.medians).read_text(encoding='utf-8'))
    rows = rows_for(units)
    common = battery._common()
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    generate_fn = generate_fn or (lambda *a: battery.generate(*a, common=common))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, int]] = set()
    if output.exists():
        for line_no, line in enumerate(output.read_text(encoding='utf-8').splitlines(), 1):
            if line.strip():
                old = json.loads(line)
                done.add((old['unit_id'], old['arm'], int(old['question_index'])))
    started = time.perf_counter()
    written = 0
    with output.open('a', encoding='utf-8', newline='\n') as fh:
        for qi, item in enumerate(QUESTIONS):
            # Exactly one pushed batch of 20 rows, neuron/random pairs interleaved by unit.
            if not all((row['id'], row['arm'], qi) in done for row in rows):
                pushes = battery.pushes_for(rows, layers, medians, common)
                answers = generate_fn(model, tok, layers, item['question'], pushes,
                                      battery.MAX_NEW_TOKENS, args.device)
                for row_info, answer in zip(rows, answers):
                    key = (row_info['id'], row_info['arm'], qi)
                    if key in done:
                        continue
                    record = {'unit_id': row_info['id'], 'arm': row_info['arm'],
                              'layer': row_info['layer'], 'neuron': row_info['neuron'],
                              'company': row_info['company'], 'question_index': qi,
                              'question': item['question'], 'kind': item['kind'], **answer}
                    fh.write(json.dumps(record, ensure_ascii=False) + '\n')
                    done.add(key)
                    written += 1
                fh.flush()
            plain_key = ('plain', 'plain', qi)
            if plain_key not in done:
                answers = generate_fn(model, tok, layers, item['question'], {},
                                      battery.MAX_NEW_TOKENS, args.device)
                answer = answers[0]
                record = {'unit_id': 'plain', 'arm': 'plain', 'layer': None, 'neuron': None,
                          'company': None, 'question_index': qi, 'question': item['question'],
                          'kind': item['kind'], **answer}
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
                fh.flush()
                done.add(plain_key)
                written += 1
    return {'answers_written': written, 'questions': len(QUESTIONS), 'units': len(units),
            'dose': battery.DOSE, 'batch_size': args.batch_size,
            'seconds': time.perf_counter() - started}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--model', default='E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    ap.add_argument('--units', default=str(ROOT / 'results/part112/units.json'))
    ap.add_argument('--medians', default=str(ROOT / 'results/part112/medians.json'))
    ap.add_argument('--out', default=str(ROOT / 'results/part153/answers.jsonl'))
    ap.add_argument('--batch-size', type=int, default=20)
    args = ap.parse_args(argv)
    if args.batch_size != 20:
        ap.error('the registered Part 153 batch size is 20')
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
