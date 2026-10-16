#!/usr/bin/env python3
# judge.py — blind verdicts on each unit's strongest contexts (Claude, 2026-10-15; routes, settings and usage added
# 2026-10-16 for the judge sweep Elise asked for).
#
# Reads <dir>/units.json (no arm labels) and asks one model, per unit, whether one job explains the snippets.
# Verdicts: one / two_related / several_unrelated / unclear, plus a one-line description.
#
# Routes:
#   ocgo      OpenCode Go's Anthropic door (/v1/messages, x-api-key from OPENCODE_GO_API_KEY, never printed). DeepSeek,
#             Haiku. --effort sets output_config.effort; --no-thinking sends thinking {type: disabled}.
#   ocgo-chat OpenCode Go's OpenAI door (/v1/chat/completions). GLM: --reasoning-effort low|medium|high, or
#             --no-thinking (thinking {type: disabled}; notes/llm-gateway-operations.md, measured 2026-09-22).
#   cliproxy  The local CLIProxyAPI at 127.0.0.1:8317 (ChatGPT subscription: gpt-6-luna, gpt-6.1-sol). Effort is only
#             honoured with thinking {type: adaptive} (gotcha), which this route always sends.
# Usage: python judge.py <dir> <model> [--route R] [--setting LABEL] [--effort E] [--reasoning-effort E]
#        [--no-thinking] [--max-tokens N] [--subset ids.json] [--workers 8]
# Output: <dir>/verdicts-<model>.json, or verdicts-<model>-<setting>.json with --setting (a rerun skips units already
# judged). Each verdict also carries usage (as the API reported it), reasoning_tokens when reported, wall_s and
# stop_reason; score.py ignores the extra fields.
import argparse, json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('model')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--route', choices=['ocgo', 'ocgo-chat', 'cliproxy'], default='ocgo')
ap.add_argument('--setting')
ap.add_argument('--effort')
ap.add_argument('--reasoning-effort')
ap.add_argument('--no-thinking', action='store_true')
ap.add_argument('--max-tokens', type=int, default=16000)  # DeepSeek reasoned past 16,000 at default and low effort (10-15)
ap.add_argument('--subset')
a = ap.parse_args()
KEY = os.environ.get('OPENCODE_GO_API_KEY', '') if a.route != 'cliproxy' else 'x'
if a.route != 'cliproxy' and not KEY: sys.exit('OPENCODE_GO_API_KEY not set')
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
units = json.load(open(os.path.join(a.dir, 'units.json')))['units']
if a.subset:
    keep = set(json.load(open(a.subset))); units = [u for u in units if u['id'] in keep]
name = a.model + (f'-{a.setting}' if a.setting else '')
outp = os.path.join(a.dir, f'verdicts-{name}.json')
done = json.load(open(outp)) if os.path.exists(outp) else {}
lock = threading.Lock()

PROMPT = """Below are 20 short text snippets. They are the places in a text corpus where one internal unit of a language model was most strongly active. In each snippet, the token where it was active is marked like ⟦this⟧. Your job is to say what the unit responds to.

Decide which of these fits best:
- "one": a single job explains nearly all of the snippets (a concept, a pattern, a grammatical or formatting role, a kind of next word). Small offshoots closely related to that job still count as one.
- "two_related": two distinct jobs, related to each other.
- "several_unrelated": several jobs with no clear relation between them.
- "unclear": you cannot find what the snippets have in common.

Consider the marked token, the words just before it, and what word would plausibly come next.

Snippets:
{snips}

Answer with only a JSON object: {{"verdict": "one" | "two_related" | "several_unrelated" | "unclear", "description": "<one line saying what it responds to, or why it is unclear>"}}"""

def request(unit):
    snips = '\n'.join(f'{i + 1}. {c}' for i, c in enumerate(unit['contexts']))
    content = PROMPT.format(snips=snips)
    if a.route == 'ocgo-chat':
        body = {'model': a.model, 'max_tokens': a.max_tokens, 'messages': [{'role': 'user', 'content': content}]}
        if a.no_thinking: body['thinking'] = {'type': 'disabled'}
        if a.reasoning_effort: body['reasoning_effort'] = a.reasoning_effort
        url = 'https://opencode.ai/zen/go/v1/chat/completions'
        headers = {'content-type': 'application/json', 'authorization': f'Bearer {KEY}'}
    else:
        body = {'model': a.model, 'max_tokens': a.max_tokens, 'messages': [{'role': 'user', 'content': content}]}
        if a.effort: body['output_config'] = {'effort': a.effort}
        if a.no_thinking: body['thinking'] = {'type': 'disabled'}
        if a.route == 'cliproxy':
            body['thinking'] = {'type': 'adaptive'}
            url = 'http://127.0.0.1:8317/v1/messages'
        else:
            url = 'https://opencode.ai/zen/go/v1/messages'
        headers = {'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': KEY}
    # Python-urllib's default User-Agent gets a 403 at opencode.ai (measured 10-15; curl's passes).
    headers['user-agent'] = 'neuron-probe/0.1 (journal-claude)'
    headers['x-opencode-session'] = f'neuron-judge-{name}-{unit["id"]}'
    return url, json.dumps(body).encode(), headers

def parse(j):
    if a.route == 'ocgo-chat':
        ch = (j.get('choices') or [{}])[0]; msg = ch.get('message') or {}
        text = msg.get('content') or ''
        u = j.get('usage') or {}
        rt = (u.get('completion_tokens_details') or {}).get('reasoning_tokens')
        return text, u, rt, ch.get('finish_reason')
    text = '\n'.join(c.get('text', '') for c in j.get('content', []) if c.get('type') == 'text')
    u = j.get('usage') or {}
    rt = (u.get('output_tokens_details') or {}).get('thinking_tokens')
    if rt is None:  # DeepSeek's door reports no split: count the thinking blocks' characters instead (labelled as such)
        chars = sum(len(c.get('thinking', '')) for c in j.get('content', []) if c.get('type') == 'thinking')
        rt = {'thinking_chars': chars} if chars else None
    return text, u, rt, j.get('stop_reason')

def call(unit):
    url, body, headers = request(unit)
    err = ''
    for attempt in range(4):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=body, method='POST', headers=headers)
            j = json.loads(OPENER.open(req, timeout=600).read())
            text, usage, rt, stop = parse(j)
            m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.S)
            v = json.loads(m.group(0)) if m else {'verdict': 'parse_error', 'description': text[:200]}
            v.update({'served': j.get('model'), 'usage': usage, 'reasoning_tokens': rt, 'wall_s': round(time.time() - t0, 2),
                      'stop_reason': stop})
            return unit['id'], v
        except Exception as e:
            err = str(e).replace(KEY, '[KEY]')[:200] if KEY else str(e)[:200]
            if hasattr(e, 'code') and getattr(e, 'code', None) == 429 and a.route == 'cliproxy':
                return unit['id'], {'verdict': 'call_error', 'description': 'HTTP 429 (limit)'}
            time.sleep(5 * (attempt + 1))
    return unit['id'], {'verdict': 'call_error', 'description': err}

todo = [u for u in units if u['id'] not in done]
print(f'{name}: {len(todo)} to judge ({len(done)} already)', flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    for k, (uid, v) in enumerate(ex.map(call, todo)):
        with lock:
            done[uid] = v
            if k % 25 == 0 or k == len(todo) - 1:
                json.dump(done, open(outp, 'w'), indent=0)
                print(f'  {k + 1}/{len(todo)}', flush=True)
json.dump(done, open(outp, 'w'), indent=0)
from collections import Counter
print(dict(Counter(v['verdict'] for v in done.values())))
