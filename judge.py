#!/usr/bin/env python3
# judge.py — blind verdicts on each unit's strongest contexts (Claude, 2026-10-15).
#
# Reads <dir>/units.json (no arm labels) and asks one model, per unit, whether one job explains the snippets.
# Models go through OpenCode Go's /v1/messages door (x-api-key; key from OPENCODE_GO_API_KEY, never printed), with
# the proxy bypassed. Verdicts: one / two_related / several_unrelated / unclear, plus a one-line description.
# Usage: python judge.py <dir> <model id> [--workers 8]   e.g. deepseek-v4.1-flash, claude-haiku-5-5
# Output: <dir>/verdicts-<model>.json, written as it goes (a rerun skips units already judged).
import argparse, json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('model'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--effort'); ap.add_argument('--no-thinking', action='store_true')
a = ap.parse_args()
KEY = os.environ['OPENCODE_GO_API_KEY']
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
units = json.load(open(os.path.join(a.dir, 'units.json')))['units']
outp = os.path.join(a.dir, f'verdicts-{a.model}.json')
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

def call(unit):
    snips = '\n'.join(f'{i + 1}. {c}' for i, c in enumerate(unit['contexts']))
    req_body = {'model': a.model, 'max_tokens': 16000,  # DeepSeek reasons first; 4000 ran out before any text (10-15)
 'messages': [{'role': 'user', 'content': PROMPT.format(snips=snips)}]}
    if a.effort: req_body['output_config'] = {'effort': a.effort}
    if a.no_thinking: req_body['thinking'] = {'type': 'disabled'}  # DeepSeek still reasoned past 16K at effort low (10-15)  # DeepSeek at default effort reasoned past 16K tokens
    body = json.dumps(req_body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request('https://opencode.ai/zen/go/v1/messages', data=body, method='POST', headers={
                'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': KEY,
                # Python-urllib's default User-Agent gets a 403 at opencode.ai (measured 10-15; curl's passes).
                'user-agent': 'neuron-probe/0.1 (journal-claude)',
                'x-opencode-session': f'neuron-judge-{a.model}-{unit["id"]}'})
            j = json.loads(OPENER.open(req, timeout=180).read())
            text = '\n'.join(c.get('text', '') for c in j.get('content', []) if c.get('type') == 'text')
            m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.S)
            v = json.loads(m.group(0)) if m else {'verdict': 'parse_error', 'description': text[:200], 'stop_reason': j.get('stop_reason')}
            v['served'] = j.get('model'); return unit['id'], v
        except Exception as e:
            err = str(e).replace(KEY, '[KEY]')[:200]; time.sleep(5 * (attempt + 1))
    return unit['id'], {'verdict': 'call_error', 'description': err}

todo = [u for u in units if u['id'] not in done]
print(f'{a.model}: {len(todo)} to judge ({len(done)} already)', flush=True)
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
