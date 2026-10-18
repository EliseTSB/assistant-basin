#!/usr/bin/env python3
"""Part 1g: answer the recorded System One requests with Muse Spark 1.3 Contributor (Claude, 2026-10-18).

Reads requests.jsonl (as recorded by run_1f.sh), answers each unique request hash once through OpenCode Go's
Responses door, and appends {"hash", "answers", ...} rows in the same format jev_stub.py replays. The Responses door
returns no logprobs, so the model states P(true) in JSON. Resumable: hashes already in the output are skipped.
Key from OPENCODE_GO_API_KEY (never printed).
Usage: muse_systemone.py requests.jsonl answers.jsonl [--effort minimal] [--workers 16] [--limit N]
"""
import argparse, json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jev_stub import request_hash

ap = argparse.ArgumentParser()
ap.add_argument('requests'); ap.add_argument('answers')
ap.add_argument('--model', default='muse-spark-1.3-contributor')
ap.add_argument('--effort', default='minimal')
ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--limit', type=int)
a = ap.parse_args()
KEY = os.environ.get('OPENCODE_GO_API_KEY') or sys.exit('OPENCODE_GO_API_KEY not set')
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

reqs = {}
for line in open(a.requests, encoding='utf-8'):
    if line.strip():
        r = json.loads(line); h = request_hash(r['state'], r['questions']); reqs.setdefault(h, r)
done = set()
if os.path.exists(a.answers):
    done = {json.loads(l)['hash'] for l in open(a.answers, encoding='utf-8') if l.strip()}
todo = [(h, r) for h, r in reqs.items() if h not in done][:a.limit]
print(f'{len(reqs)} unique, {len(done)} done, {len(todo)} to answer', flush=True)

def prompt(r):
    parts = []
    for name, q in r['questions'].items():
        c = q.get('criteria', {})
        parts.append(f'Question "{name}": {q["instructions"]}\n  true: {c.get("true")}\n  false: {c.get("false")}')
    return ('State (JSON):\n' + json.dumps(r['state'], ensure_ascii=False) + '\n\n' + '\n\n'.join(parts) +
            '\n\nFor each question, give the probability (0 to 1) that "true" holds. Answer with only a JSON object '
            'mapping each question name to its probability, e.g. {"' + next(iter(r['questions'])) + '": 0.35}.')

lock = threading.Lock(); out = open(a.answers, 'a', encoding='utf-8')
def call(item):
    h, r = item
    body = json.dumps({'model': a.model, 'input': prompt(r), 'max_output_tokens': 8000,
                       'reasoning': {'effort': a.effort}}).encode()
    hdr = {'content-type': 'application/json', 'authorization': f'Bearer {KEY}',
           'user-agent': 'neuron-probe/0.1 (journal-claude)', 'x-opencode-session': f'part1g-{h[:16]}'}
    err = ''
    for attempt in range(4):
        t0 = time.time()
        try:
            j = json.loads(OPENER.open(urllib.request.Request('https://opencode.ai/zen/go/v1/responses', data=body,
                                                              headers=hdr, method='POST'), timeout=300).read())
            text = '\n'.join(c.get('text', '') for o in j.get('output', []) if o.get('type') == 'message'
                             for c in o.get('content', []) if c.get('type') == 'output_text')
            m = re.search(r'\{[^{}]*\}', text, re.S); vals = json.loads(m.group(0)) if m else {}
            ans = {k: min(1.0, max(0.0, float(vals[k]))) for k in r['questions'] if k in vals}
            if len(ans) != len(r['questions']): raise ValueError('unparsed: ' + text[:120])
            row = {'hash': h, 'answers': ans, 'wall_s': round(time.time() - t0, 2), 'usage': j.get('usage')}
            with lock: out.write(json.dumps(row) + '\n'); out.flush()
            return True
        except Exception as e:
            err = str(e).replace(KEY, '[KEY]')[:160]; time.sleep(3 * (attempt + 1))
    print('FAILED', h[:12], err, flush=True); return False

t0 = time.time(); ok = 0
with ThreadPoolExecutor(a.workers) as ex:
    for k, res in enumerate(ex.map(call, todo)):
        ok += res
        if k % 250 == 0: print(f'  {k + 1}/{len(todo)} ok={ok} {(k + 1) / (time.time() - t0):.2f}/s', flush=True)
print(f'done ok={ok}/{len(todo)} in {time.time() - t0:.0f}s', flush=True)
