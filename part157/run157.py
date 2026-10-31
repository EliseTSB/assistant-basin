#!/usr/bin/env python3
"""Part 157 constrained-decoding ladder and zero-arm runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part153 import battery153 as p153  # noqa: E402
from part148 import battery148 as p148  # noqa: E402
from part154.run154 import ZeroRowsHook  # noqa: E402

MODEL_DEFAULT = 'E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MAX_NEW_TOKENS = 120
FAMILIES = {
    'Alibaba': ['Alibaba', 'Tongyi', 'Qwen', 'Qianwen', 'Aliyun', '阿里', '通义', '千问'],
    'Google': ['Google', 'DeepMind', 'Gemini', '谷歌'], 'OpenAI': ['OpenAI', 'ChatGPT', 'GPT'],
    'Microsoft': ['Microsoft', 'Copilot', 'Azure', '微软'], 'Meta': ['Meta', 'Llama', 'Facebook'],
    'Anthropic': ['Anthropic', 'Claude'], 'Baidu': ['Baidu', 'Ernie', '百度'],
    'Tencent': ['Tencent', 'Hunyuan', '腾讯'], 'DeepSeek': ['DeepSeek'], 'Huawei': ['Huawei', '华为'],
}
FAMILY_ORDER = list(FAMILIES)
CONTROL_NOUNS = ['apple', 'bridge', 'candle', 'drawer', 'engine', 'feather', 'garden', 'hammer',
                 'island', 'jacket', 'kettle', 'ladder', 'mirror', 'needle', 'orange', 'pillow',
                 'quilt', 'river', 'saddle', 'ticket', 'umbrella', 'valley', 'window', 'yogurt']
L27_ZEROES = {
    'zero_L27N3325': [{'id': 'L27N3325', 'layer': 27, 'neuron': 3325}],
    'zero_L27N4991': [{'id': 'L27N4991', 'layer': 27, 'neuron': 4991}],
}


def _ids(tok, text):
    encoded = tok(text, add_special_tokens=False)
    return encoded['input_ids'] if isinstance(encoded, dict) else encoded.input_ids


def ban_ids(tok, families):
    """Return deduplicated token sequences for bare, spaced, and **-prefixed names."""
    if isinstance(families, str):
        families = [families]
    result, seen = [], set()
    for family in families:
        # Also accept the five plain nouns in the registered control-ban arm.
        if family in FAMILIES:
            spellings = FAMILIES[family]
        elif isinstance(family, str) and family:
            spellings = [family]
        else:
            raise ValueError(f'unknown family or spelling: {family!r}')
        for spelling in spellings:
            for variant in (spelling, ' ' + spelling, '**' + spelling):
                raw = _ids(tok, variant)
                if hasattr(raw, 'tolist'):
                    raw = raw.tolist()
                if raw and isinstance(raw[0], list):
                    raw = raw[0]
                seq = [int(x) for x in raw]
                if variant.startswith('**'):
                    prefix = _ids(tok, '**')
                    if hasattr(prefix, 'tolist'):
                        prefix = prefix.tolist()
                    if prefix and isinstance(prefix[0], list):
                        prefix = prefix[0]
                    prefix = [int(x) for x in prefix]
                    if prefix and seq[:len(prefix)] == prefix:
                        seq = seq[len(prefix):]
                key = tuple(seq)
                if seq and key not in seen:
                    seen.add(key)
                    result.append(seq)
    return result


def first_named_family(answer):
    lowered, earliest = answer.casefold(), None
    for order, (family, spellings) in enumerate(FAMILIES.items()):
        for spelling in spellings:
            at = lowered.find(spelling.casefold())
            if at >= 0 and (earliest is None or (at, order) < earliest):
                earliest = (at, order)
    return FAMILY_ORDER[earliest[1]] if earliest is not None else None


def family_counts(rows):
    counts = {f: 0 for f in FAMILY_ORDER}
    counts['no_family'] = 0
    for row in rows:
        family = first_named_family(row.get('answer', ''))
        if family:
            counts[family] += 1
        else:
            counts['no_family'] += 1
    return counts


def choose_next_family(counts, banned):
    remaining = [f for f in FAMILY_ORDER if f not in banned]
    return min(remaining, key=lambda f: (-int(counts.get(f, 0)), FAMILY_ORDER.index(f))) if remaining else None


def pick_control_words(level0_answers):
    joined = '\n'.join(answer.casefold() for answer in level0_answers)
    words = [w for w in CONTROL_NOUNS if w.casefold() not in joined][:5]
    if len(words) != 5:
        raise ValueError('fixed control noun list did not contain five words absent from all level 0 answers')
    return words


def _pad(prompts, tok, device):
    import torch
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), int(pad_id or 0), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, width-len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        mask[i, width-len(prompt):] = 1
    return ids, mask


def rendered_inputs(tok, questions, device):
    common = p148._common()
    rendered = [common.build_prompt(tok, q) for q in questions]
    prompts = []
    for text in rendered:
        raw = _ids(tok, text)
        if hasattr(raw, 'tolist'):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        prompt = [int(x) for x in raw]
        if tok.decode(prompt, skip_special_tokens=False) != text:
            raise ValueError('tokenized chat prompt does not decode back exactly')
        prompts.append(prompt)
    ids, mask = _pad(prompts, tok, device)
    for i, prompt in enumerate(prompts):
        decoded = tok.decode(ids[i][mask[i].bool()].tolist(), skip_special_tokens=False)
        if decoded != rendered[i]:
            raise ValueError('left-padded chat prompt does not decode back exactly')
    return ids, mask, prompts


def _generate_batch(model, tok, layers, questions, bad_words_ids, zero_units=None):
    import torch
    device = str(next(model.parameters()).device)
    inputs, mask, _ = rendered_inputs(tok, questions, device)
    prompt_len = int(inputs.shape[1])
    eos = p148._common().eos_ids(tok, model)
    kwargs = dict(input_ids=inputs, attention_mask=mask, max_new_tokens=120, do_sample=False,
                  use_cache=True, pad_token_id=getattr(tok, 'pad_token_id', None),
                  bad_words_ids=bad_words_ids)
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
                ids = row[:pos+1]
                break
        results.append({'answer': tok.decode(ids, skip_special_tokens=True).strip(),
                        'token_count': len(ids), 'first_nonfinite_call': zero.first_nonfinite.get(len(results))})
    return results, trace


def generate_clean(model, tok, layers, questions, banned, zero_units=None, attempts=3):
    bad = ban_ids(tok, banned) if banned else None
    retries = []
    for attempt in range(attempts):
        results, trace = _generate_batch(model, tok, layers, questions, bad, zero_units)
        hit = [i for i, r in enumerate(results) if r.get('first_nonfinite_call') is not None]
        if not hit:
            return results, trace, retries
        retries.append({'attempt': attempt, 'rows': hit,
                        'calls': [results[i]['first_nonfinite_call'] for i in hit]})
    raise RuntimeError(f'batch went non-finite on {attempts} attempts: {retries}')


def _read_answers(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(args, loader=None):
    common = p148._common()
    questions = p153.DIRECT + p153.INDIRECT
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
    batch_safe = True
    try:
        import torch  # imported only on the execution path
    except ImportError:
        pass
    else:
        batch_safe = assert_padding_equivalence(model, tok, questions[:2], args.device)
        if not batch_safe:
            print('mixed-length batching changed plain greedy output; using batch size 1')
    alibaba = ban_ids(tok, ['Alibaba'])
    print('Alibaba bad_words_ids count:', len(alibaba))
    print('Alibaba decoded bad_words_ids:', json.dumps([tok.decode(x, skip_special_tokens=False) for x in alibaba]))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    answers_path, ladder_path = outdir / 'answers.jsonl', outdir / 'ladder.json'
    rows = _read_answers(answers_path)
    done = {(r['arm'], int(r['level']), int(r['question_index'])) for r in rows}
    ladder = json.loads(ladder_path.read_text(encoding='utf-8')) if ladder_path.exists() else {
        'family_order': FAMILY_ORDER, 'levels': [], 'control_words': None,
        'zero_arms': list(L27_ZEROES)}
    retries_all = []

    def run_arm(arm, level, banned, zero_units=None):
        pending = [i for i in range(24) if (arm, level, i) not in done]
        if not pending:
            return
        batch_questions = [questions[i] for i in pending]
        if batch_safe:
            units = [zero_units for _ in pending] if zero_units else None
            generated, _trace, retries = generate_clean(model, tok, layers, batch_questions, banned, units)
            retries_all.extend({'arm': arm, 'level': level, **r} for r in retries)
        else:
            generated = []
            for question in batch_questions:
                units = [zero_units] if zero_units else None
                result, _trace, retries = generate_clean(model, tok, layers, [question], banned, units)
                generated.extend(result)
                retries_all.extend({'arm': arm, 'level': level, **r} for r in retries)
        with answers_path.open('a', encoding='utf-8', newline='\n') as fh:
            for qi, question, answer in zip(pending, batch_questions, generated):
                record = {'arm': arm, 'level': level, 'banned_families': list(banned),
                          'question_index': qi, 'question': question, **answer}
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
                rows.append(record)
                done.add((arm, level, qi))
            fh.flush()

    run_arm('level_0', 0, [])
    level0 = sorted((r for r in rows if r['arm'] == 'level_0' and r['level'] == 0), key=lambda r: r['question_index'])
    if len(level0) != 24:
        raise ValueError('level 0 must have 24 answers before control selection')
    words = ladder.get('control_words') or pick_control_words([r['answer'] for r in level0])
    ladder['control_words'] = words
    ladder['control_word_selection'] = 'first five absent from level 0 in fixed CONTROL_NOUNS order'
    run_arm('control_ban', 0, words)
    run_arm('level_1', 1, ['Alibaba'])
    for arm, units in L27_ZEROES.items():
        run_arm(arm, 1, ['Alibaba'], units)

    entries = {int(x['level']): x for x in ladder.get('levels', [])}
    entries[0] = {'level': 0, 'banned_families': [], 'counts': family_counts(level0), 'chosen_next': 'Alibaba'}
    banned = ['Alibaba']
    for level in range(1, 7):
        previous = sorted((r for r in rows if r['arm'] == f'level_{level}' and r['level'] == level), key=lambda r: r['question_index'])
        if len(previous) != 24:
            raise ValueError(f'level {level} is incomplete; cannot select the next family')
        counts = family_counts(previous)
        chosen = choose_next_family(counts, banned)
        if level == 6:
            chosen = None
        entries[level] = {'level': level, 'banned_families': list(banned), 'counts': counts, 'chosen_next': chosen}
        if level == 6 or chosen is None:
            break
        banned.append(chosen)
        next_level = level + 1
        run_arm(f'level_{next_level}', next_level, banned)
    ladder['levels'] = [entries[k] for k in sorted(entries)]
    ladder['nonfinite_retries'] = retries_all
    ladder_path.write_text(json.dumps(ladder, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'answers': str(answers_path), 'ladder': str(ladder_path)}


def assert_padding_equivalence(model, tok, questions, device):
    """Tiny plain-arm greedy comparison for mixed left padding versus individual prompts."""
    import torch
    if len(questions) < 2:
        return True
    ids, mask, prompts = rendered_inputs(tok, questions[:2], device)
    with torch.inference_mode():
        batch = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=2,
                               do_sample=False, use_cache=True, pad_token_id=getattr(tok, 'pad_token_id', None))
        for i, prompt in enumerate(prompts):
            one = torch.tensor([prompt], dtype=torch.long, device=device)
            single = model.generate(input_ids=one, attention_mask=torch.ones_like(one), max_new_tokens=2,
                                    do_sample=False, use_cache=True, pad_token_id=getattr(tok, 'pad_token_id', None))
            if batch[i, ids.shape[1]:].tolist() != single[0, len(prompt):].tolist():
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='xpu')
    ap.add_argument('--model', default=MODEL_DEFAULT)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
