#!/usr/bin/env python3
"""Part 158 token-level spelling-ban and neuron-zero runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part157 import run157 as p157  # noqa: E402
from part154.run154 import ZeroRowsHook  # noqa: E402

MODEL_DEFAULT = 'E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PREFIXES = ('ali', 'tong', 'qian', 'qwen', 'qw')
BAN_CHARS = ('阿', '里', '通', '义', '義', '千', '问', '問')
MAX_NEW_TOKENS = 120


def _token_ids(tok, text):
    encoded = tok(text, add_special_tokens=False)
    raw = encoded['input_ids'] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(raw, 'tolist'):
        raw = raw.tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(x) for x in raw]


def _excluded_ids(tok):
    common = p157.p148._common()
    eos = common.eos_ids(tok, None)
    return set(int(x) for x in eos) | set(int(x) for x in getattr(tok, 'all_special_ids', []))


def _is_banned_text(decoded):
    text = decoded.strip(' *').casefold()
    return text.startswith(PREFIXES) or any(char in text for char in BAN_CHARS)


def banned_token_ids(tok):
    """Scan every vocabulary id using CPU decode and return sorted banned ids."""
    excluded = _excluded_ids(tok)
    candidates = []
    # len(tokenizer) includes added vocabulary ids; decode always receives ordinary CPU ints.
    for token_id in range(len(tok)):
        if _is_banned_text(tok.decode([token_id], skip_special_tokens=False)):
            candidates.append(token_id)
    result = sorted(token_id for token_id in candidates if token_id not in excluded)
    print('special/eos tokens excluded from ban:', len(set(candidates) & excluded))
    return result


def random_control_ids(tok, n, level0_answers, seed=158):
    """Draw n non-special ids absent from all level-0 answer tokenizations."""
    excluded = _excluded_ids(tok)
    for answer in level0_answers:
        excluded.update(_token_ids(tok, str(answer)))
    available = np.asarray([i for i in range(len(tok)) if i not in excluded], dtype=np.int64)
    if int(n) < 0 or int(n) > len(available):
        raise ValueError(f'cannot draw {n} random control ids from {len(available)} eligible vocabulary ids')
    chosen = np.random.default_rng(seed).choice(available, size=int(n), replace=False)
    return sorted(int(x) for x in chosen.tolist())


def _generate_batch(model, tok, layers, questions, suppress_tokens, zero_units=None):
    import torch
    device = str(next(model.parameters()).device)
    inputs, mask, _ = p157.rendered_inputs(tok, questions, device)
    prompt_len = int(inputs.shape[1])
    common = p157.p148._common()
    eos = common.eos_ids(tok, model)
    kwargs = dict(input_ids=inputs, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                  do_sample=False, use_cache=True, pad_token_id=getattr(tok, 'pad_token_id', None),
                  suppress_tokens=list(suppress_tokens))
    if eos:
        kwargs['eos_token_id'] = eos
    trace = {'calls': 0, 'positions': 0, 'target_checks': 0, 'target_positions': 0,
             'unchanged_checks': 0, 'unchanged_positions': 0,
             'unzeroed_target_checks': 0, 'unzeroed_target_positions': 0,
             'target_failures': 0, 'unchanged_failures': 0, 'unzeroed_target_failures': 0}
    row_units = zero_units if zero_units is not None else [[] for _ in questions]
    zero = ZeroRowsHook(layers[27].mlp.down_proj, row_units, trace)
    try:
        with torch.inference_mode():
            seq = model.generate(**kwargs)
    finally:
        zero.remove()
    results, eos_set = [], set(eos)
    for row in seq[:, prompt_len:].tolist():
        ids = row
        for pos, token in enumerate(row):
            if int(token) in eos_set:
                ids = row[:pos + 1]
                break
        results.append({'answer': tok.decode(ids, skip_special_tokens=True).strip(),
                        'token_count': len(ids), 'first_nonfinite_call': zero.first_nonfinite.get(len(results))})
    return results, trace


def generate_clean(model, tok, layers, questions, suppress_tokens, zero_units=None, attempts=3):
    retries = []
    for attempt in range(attempts):
        results, trace = _generate_batch(model, tok, layers, questions, suppress_tokens, zero_units)
        hit = [i for i, row in enumerate(results) if row.get('first_nonfinite_call') is not None]
        if not hit:
            return results, trace, retries
        retries.append({'attempt': attempt, 'rows': hit,
                        'calls': [results[i]['first_nonfinite_call'] for i in hit]})
    raise RuntimeError(f'batch went non-finite on {attempts} attempts: {retries}')


def _read_answers(path):
    with Path(path).open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(args, loader=None):
    common = p157.p148._common()
    questions = p157.p153.DIRECT + p157.p153.INDIRECT
    if len(questions) != 24:
        raise ValueError('Part 153 must supply exactly 24 registered phrasings')
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(args.model, common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    level0_rows = _read_answers(args.level0)
    level0_rows = sorted((r for r in level0_rows if r.get('arm') == 'level_0' and int(r.get('level', 0)) == 0),
                         key=lambda r: int(r['question_index']))
    if len(level0_rows) != 24:
        raise ValueError('Part 157 level-0 answers must contain exactly 24 rows')
    banned = banned_token_ids(tok)
    random_ids = random_control_ids(tok, len(banned), [r['answer'] for r in level0_rows], seed=158)
    sample_ids = sorted(np.random.default_rng(158).choice(np.asarray(banned, dtype=np.int64),
                                                          size=min(40, len(banned)), replace=False).tolist()) if banned else []
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ban = {'banned_count': len(banned),
           'sample_40_banned_tokens': [{'id': int(i), 'token': tok.decode([int(i)], skip_special_tokens=False)}
                                       for i in sample_ids],
           'control_count': len(random_ids)}
    with (outdir / 'ban.json').open('w', encoding='utf-8', newline='\n') as fh:
        json.dump(ban, fh, ensure_ascii=True, indent=2)
        fh.write('\n')
    print(json.dumps({'banned_count': len(banned), 'control_count': len(random_ids)}, ensure_ascii=True), flush=True)

    batch_safe = p157.assert_padding_equivalence(model, tok, questions[:2], args.device)
    if not batch_safe:
        print('mixed-length batching changed plain greedy output; using batch size 1', flush=True)
    outpath = outdir / 'answers.jsonl'
    retries_all = []
    arms = [('plain', banned, None),
            ('zero_L27N3325', banned, p157.L27_ZEROES['zero_L27N3325']),
            ('zero_L27N4991', banned, p157.L27_ZEROES['zero_L27N4991']),
            ('random_control', random_ids, None)]
    with outpath.open('w', encoding='utf-8', newline='\n') as fh:
        for arm, suppress, units in arms:
            if batch_safe:
                zero_rows = [units for _ in questions] if units else None
                generated, _trace, retries = generate_clean(model, tok, layers, questions, suppress, zero_rows)
                retries_all.extend({'arm': arm, **r} for r in retries)
            else:
                generated = []
                for question in questions:
                    zero_rows = [units] if units else None
                    result, _trace, retries = generate_clean(model, tok, layers, [question], suppress, zero_rows)
                    generated.extend(result)
                    retries_all.extend({'arm': arm, **r} for r in retries)
            for qi, (question, answer) in enumerate(zip(questions, generated)):
                row = {'arm': arm, 'question_index': qi, 'question': question,
                       'suppress_token_count': len(suppress), **answer}
                fh.write(json.dumps(row, ensure_ascii=True) + '\n')
            fh.flush()
    print(json.dumps({'rows': 96, 'nonfinite_retries': retries_all}, ensure_ascii=True), flush=True)
    return {'answers': str(outpath), 'ban': str(outdir / 'ban.json')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--model', default=MODEL_DEFAULT)
    ap.add_argument('--level0', default='results/part157/answers.jsonl')
    ap.add_argument('--out', required=True)
    run(ap.parse_args(argv))


if __name__ == '__main__':
    main()
