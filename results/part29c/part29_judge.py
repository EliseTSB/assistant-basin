#!/usr/bin/env python3
"""Part 29c (Claude, 2026-10-19): a no-training LLM judge names the model family behind each Part 29 answer.

Design and predictions: PREDICTIONS.md Part 29c (pushed before the run). Judge: GPT-6 Luna through CLIProxy.
Resumable: results/part29/judge-<judge>.jsonl, one row per (model, index); a rerun skips rows already present.
"""
import argparse, collections, concurrent.futures, json, math, sys, threading, time, urllib.error, urllib.request
from pathlib import Path
import part29_collect as collect

FAMILIES = ['Anthropic Claude', 'OpenAI GPT', 'Google Gemini', 'DeepSeek', 'Alibaba Qwen', 'Zhipu GLM', 'Moonshot Kimi', 'MiniMax']
ALIASES = [('Anthropic Claude', ('claude', 'anthropic')), ('OpenAI GPT', ('gpt', 'openai', 'chatgpt')),
           ('Google Gemini', ('gemini', 'google')), ('DeepSeek', ('deepseek',)), ('Alibaba Qwen', ('qwen', 'alibaba')),
           ('Zhipu GLM', ('glm', 'zhipu')), ('Moonshot Kimi', ('kimi', 'moonshot')), ('MiniMax', ('minimax',))]
TEACHERS = {'claude-haiku-5-5': 'Anthropic Claude', 'gpt-6-luna': 'OpenAI GPT', 'deepseek-v4-pro': 'DeepSeek',
            'qwen3.8-max': 'Alibaba Qwen', 'gemini-4-argon-contributor': 'Google Gemini', 'glm-5.3': 'Zhipu GLM',
            'kimi-k3': 'Moonshot Kimi', 'minimax-m3': 'MiniMax'}
STUDENTS = ['deepseek-v4.1-flash', 'mimo-v2.6-flash']
QUESTION = ("Which company's model family most likely wrote this answer? Choose exactly one: Anthropic Claude, "
            "OpenAI GPT, Google Gemini, DeepSeek, Alibaba Qwen, Zhipu GLM, Moonshot Kimi, MiniMax. Reply with only the name.")
WORDS = 250
PER_TEACHER = 50


def answer_view(row):
    value = row.get('answer') or row.get('text') or ''
    return ' '.join(str(value).split()[:WORDS])


def parse_label(text):
    """The family named first in the judge's reply; None when it names none of the eight."""
    low = (text or '').lower()
    hits = []
    for family, keys in ALIASES:
        positions = [low.find(k) for k in keys if low.find(k) >= 0]
        if positions:
            hits.append((min(positions), family))
    return min(hits)[1] if hits else None


def items(results_dir, prompts):
    out = []
    for model in list(TEACHERS) + STUDENTS:
        path = results_dir / f'answers-{model}.jsonl'
        rows = sorted((json.loads(l) for l in path.open(encoding='utf-8') if l.strip()), key=lambda r: r['index'])
        if model in TEACHERS:
            rows = rows[:PER_TEACHER]
        for r in rows:
            out.append({'model': model, 'index': r['index'], 'prompt': prompts[r['index']], 'answer': answer_view(r)})
    return out


def ask(judge, item, tries=4):
    content = f"{QUESTION}\n\nThe user's request:\n{item['prompt']}\n\nThe answer:\n{item['answer']}"
    body = json.dumps({'model': judge, 'temperature': 0, 'max_tokens': 2000,
                       'messages': [{'role': 'user', 'content': content}]}).encode('utf-8')
    for attempt in range(tries):
        req = urllib.request.Request(collect.CLIPROXY_CHAT_URL, data=body, method='POST',
                                     headers={'content-type': 'application/json'})
        try:
            with collect._open(req, timeout=300) as response:
                payload = json.loads(response.read().decode('utf-8'))
            return (payload['choices'][0]['message'].get('content') or '').strip()
        except urllib.error.HTTPError as exc:
            delay = collect.retry_delay(exc.headers, attempt)
            if exc.code != 429 and exc.code < 500 or attempt + 1 == tries or delay > collect.MAX_RETRY_WAIT:
                raise
            time.sleep(delay)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def summarize(rows):
    lines = []
    teacher = [r for r in rows if r['model'] in TEACHERS]
    for label, sel in (('all eight', teacher), ('seven (GPT row excluded: judge self-recognition)',
                                                [r for r in teacher if r['model'] != 'gpt-6-luna'])):
        k = sum(r['label'] == TEACHERS[r['model']] for r in sel)
        lo, hi = wilson(k, len(sel))
        lines.append(f'accuracy, {label}: {k}/{len(sel)} = {k / max(1, len(sel)):.1%} (95% CI {lo:.1%}-{hi:.1%}; chance 12.5%)')
    lines.append('')
    lines.append('per family (rows = true family; counts of the judge labels):')
    for model, fam in TEACHERS.items():
        sel = [r for r in teacher if r['model'] == model]
        c = collections.Counter(r['label'] or 'none' for r in sel)
        recall = c[fam] / max(1, len(sel))
        lines.append(f'  {fam:17s} recall {recall:5.1%}  ' + ', '.join(f'{k} {v}' for k, v in c.most_common()))
    lines.append('')
    lines.append('students (label distribution):')
    for model in STUDENTS:
        sel = [r for r in rows if r['model'] == model]
        c = collections.Counter(r['label'] or 'none' for r in sel)
        lines.append(f'  {model:20s} n={len(sel):3d}  ' + ', '.join(f'{k} {v / max(1, len(sel)):.0%}' for k, v in c.most_common()))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', default='gpt-6-luna')
    ap.add_argument('--prompts', default='data/part29-prompts.json')
    ap.add_argument('--results-dir', default='results/part29')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='judge only the first N items (smoke test)')
    args = ap.parse_args()
    results = Path(args.results_dir)
    prompts = {p['index']: p['prompt'] for p in json.load(open(args.prompts, encoding='utf-8'))['prompts']}
    todo = items(results, prompts)
    if args.limit:
        todo = todo[:args.limit]
    out = results / f'judge-{args.judge}.jsonl'
    done = set()
    if out.exists():
        for l in out.open(encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                done.add((r['model'], r['index']))
    lock = threading.Lock()
    pending = [it for it in todo if (it['model'], it['index']) not in done]
    print(f'{len(todo)} items, {len(pending)} to judge with {args.judge}', flush=True)
    failures = 0
    with out.open('a', encoding='utf-8') as handle, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(ask, args.judge, it): it for it in pending}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            it = futures[fut]
            try:
                reply = fut.result()
            except Exception as exc:
                failures += 1
                print(f"{it['model']} {it['index']} failed: {str(exc)[:200]}", file=sys.stderr, flush=True)
                continue
            row = {'model': it['model'], 'index': it['index'], 'judge': args.judge, 'reply': reply[:200], 'label': parse_label(reply)}
            with lock:
                handle.write(json.dumps(row) + '\n')
                handle.flush()
            if n % 50 == 0:
                print(f'{n}/{len(pending)} judged', flush=True)
    rows = [json.loads(l) for l in out.open(encoding='utf-8') if l.strip()]
    keep = {(it['model'], it['index']) for it in todo}
    rows = [r for r in rows if (r['model'], r['index']) in keep]
    text = summarize(rows)
    (results / f'judge-{args.judge}-summary.txt').write_text(text + '\n', encoding='utf-8')
    print(text)
    print(f'failures: {failures}')


if __name__ == '__main__':
    main()
