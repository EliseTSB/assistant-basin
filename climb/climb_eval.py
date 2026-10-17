#!/usr/bin/env python3
# climb_eval.py — the fixed scorer for Part 1e's overnight prompt climb (Claude, 2026-10-16, for Elise's 21:06 ask:
# "improving the prompt for the neurons judges and Jev ... hill climbing on").
#
# A candidate is a directory holding explainer.txt (the prompt Luna gets with a unit's 20 strongest training contexts;
# it must contain {snips} and ask for JSON with "description") and candidate.json (see harness/baseline). The pipeline:
#   1. explainer: Luna (gpt-6-luna through CLIProxyAPI at 127.0.0.1:8317) writes a description of the unit, clipped to
#      explainer.max_words words (at most 80);
#   2. simulator: Jev (TypeSafe /v1/systemone) reads the description and ONE held-out snippet per call and returns
#      probabilities for the candidate's noul questions; p = weighted mean of them;
#   3. score: per unit, Spearman of p against the unit's true activation over its 30 held-out contexts (10 strongest,
#      10 mid-strength, 10 random), plus AUROC strongest-vs-random and mid-vs-random. PRIMARY = mean per-unit Spearman.
#   4. judge (--stage judge or all): P(one job) per unit, from the explainer's own answer ("p_one", else "verdict") or
#      from Jev questions over the snippets. Scored as AUROC for picking out units whose firing this candidate's
#      pipeline predicts (unit Spearman >= 0.4), and as the floor: the share of random-context sets called one job.
#
# Sets: "search" (60 sim units + 25 floor sets, the fast loop) and "dev" (149 sim units + 25 floor sets). There is no
# test set here on purpose; it is scored elsewhere after the prompts are frozen.
# Usage: python3 climb_eval.py <candidate dir> [--set search|dev] [--stage sim|judge|all] [--ref <candidate dir>]
#        [--note TEXT] [--dry]          (--dry prints how many calls are not cached yet, and stops)
# Exit codes: 0 ok · 2 bad candidate · 3 budget exhausted · 4 Jev limit/billing refusal (stop: JJ's watcher shares the
# key) · 5 Luna limit · 6 another eval is running.
# Writes, all under the work dir (WORK, default the current directory): cache/, runs/<hash>-<set>.json per-unit rows,
# ledger.jsonl one line per eval, budget.json the Jev input tokens spent by the climb.
import argparse, fcntl, hashlib, json, math, os, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get('CLIMB_WORK', os.getcwd())
JEV_BUDGET = min(int(os.environ.get('CLIMB_JEV_BUDGET', 80_000_000)), 80_000_000)  # a backstop only: the relay holds each climber's real cap (TypeSafe ~$0.042/M; the key is shared with JJ's watcher)
JEV_CONC, LUNA_CONC = 4, 6
LUNA_URL = 'http://127.0.0.1:8317/v1/messages'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_lock = threading.Lock()

class Stop(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code

# ---------- candidate ----------
def load_candidate(d):
    c = json.load(open(os.path.join(d, 'candidate.json')))
    c['explainer_prompt'] = open(os.path.join(d, 'explainer.txt')).read()
    if '{snips}' not in c['explainer_prompt']: raise Stop(2, 'explainer.txt must contain {snips}')
    ex = c.setdefault('explainer', {})
    if ex.get('effort', 'high') not in ('low', 'medium', 'high'): raise Stop(2, 'explainer.effort must be low, medium or high')
    ex['effort'] = ex.get('effort', 'high'); ex['max_words'] = min(int(ex.get('max_words', 80)), 80)
    s = c.get('simulator') or {}
    for q, spec in (s.get('questions') or {}).items():
        if spec.get('type') != 'noul': raise Stop(2, f'simulator question {q}: only noul questions are scored')
    if not s.get('questions'): raise Stop(2, 'simulator.questions is empty')
    w = s.get('weights') or {q: 1.0 for q in s['questions']}
    if set(w) - set(s['questions']) or any((not isinstance(v, (int, float))) or v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise Stop(2, 'simulator.weights must name its questions with non-negative weights')
    s['weights'] = w; s['include_examples'] = int(s.get('include_examples', 0))
    if not 0 <= s['include_examples'] <= 5: raise Stop(2, 'simulator.include_examples must be 0-5')
    j = c.setdefault('judge', {'engine': 'explainer'})
    if j['engine'] not in ('explainer', 'jev'): raise Stop(2, 'judge.engine must be explainer or jev')
    if j['engine'] == 'jev':
        for q, spec in (j.get('questions') or {}).items():
            if spec.get('type') != 'noul': raise Stop(2, f'judge question {q}: only noul')
        if not j.get('questions'): raise Stop(2, 'judge.questions is empty')
        j['weights'] = j.get('weights') or {q: 1.0 for q in j['questions']}
    c['_dir'] = os.path.abspath(d)
    return c

def cand_hash(c):
    blob = json.dumps({k: c[k] for k in ('explainer', 'simulator', 'judge', 'explainer_prompt')}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]

def leak_check(c, data):
    # Prompts must be generic: no 40-character run copied from any dev context (training or held-out).
    texts = [c['explainer_prompt'], json.dumps(c['simulator']), json.dumps(c['judge'])]
    grams = set()
    for r in data['sim_units'] + data['floor_units']:
        for t in r['contexts'] + [h['context'] for h in r.get('heldout', [])]:
            for i in range(0, max(1, len(t) - 39), 8): grams.add(t[i:i + 40])
    for t in texts:
        for i in range(0, max(1, len(t) - 39)):
            if t[i:i + 40] in grams: raise Stop(2, f'prompt text copies a dev context: {t[i:i + 40]!r}')

# ---------- cache and budget ----------
def _p(kind, k): return os.path.join(WORK, 'cache', kind, k[:2], k + '.json')
def cache_get(kind, k):
    try: return json.load(open(_p(kind, k)))
    except (FileNotFoundError, json.JSONDecodeError): return None
def cache_put(kind, k, v):
    p = _p(kind, k); os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f'.{os.getpid()}.tmp'; json.dump(v, open(tmp, 'w')); os.replace(tmp, p)
def h(*parts): return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def explainer_key(c, unit): return h('luna', c['explainer_prompt'], c['explainer']['effort'], unit['contexts'])
def jev_key(c, desc, snippet, examples=None, qs=None):
    st = {'description': desc, 'snippet': snippet}
    if examples: st['examples'] = examples
    return h('jev', qs or c['simulator']['questions'], st)
def budget():
    try: return json.load(open(os.path.join(WORK, 'budget.json')))
    except FileNotFoundError: return {'jev_input_tokens': 0, 'jev_calls': 0, 'luna_calls': 0, 'luna_output_tokens': 0}
def spend(**kw):
    with _lock:
        b = budget()
        for k, v in kw.items(): b[k] = b.get(k, 0) + (v or 0)
        tmp = os.path.join(WORK, 'budget.json.tmp'); json.dump(b, open(tmp, 'w')); os.replace(tmp, os.path.join(WORK, 'budget.json'))
        if b['jev_input_tokens'] >= JEV_BUDGET: raise Stop(3, f'Jev budget of {JEV_BUDGET:,} input tokens spent')

# ---------- calls ----------
# Jev goes through the local relay (jev_relay.py, 127.0.0.1:8341), which holds the TypeSafe key, caps concurrent calls for
# all climbers together and caps each climber's input tokens. No key is read here.
JEV_URL = 'http://127.0.0.1:8341/v1/systemone'
CLIMBER = os.environ.get('CLIMB_NAME') or os.path.basename(os.path.abspath(WORK)).removeprefix('work-')  # work-luna -> luna

def post(url, body, headers, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST', headers=headers)
    return json.loads(OPENER.open(req, timeout=timeout).read())

def call_luna(prompt, effort):
    body = {'model': 'gpt-6-luna', 'max_tokens': 16000, 'messages': [{'role': 'user', 'content': prompt}],
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': effort}}
    hd = {'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': 'x'}
    err = ''
    for a in range(4):
        try:
            j = post(LUNA_URL, body, hd)
            text = '\n'.join(x.get('text', '') for x in j.get('content', []) if x.get('type') == 'text')
            spend(luna_calls=1, luna_output_tokens=(j.get('usage') or {}).get('output_tokens'))
            return text, j.get('usage')
        except urllib.error.HTTPError as e:
            if e.code == 429: raise Stop(5, 'Luna usage limit (HTTP 429)')
            err = f'HTTP {e.code}'
        except Stop: raise
        except Exception as e: err = str(e)[:200]
        time.sleep(5 * (a + 1))
    raise RuntimeError(f'Luna call failed: {err}')

def call_jev(questions, state):
    hd = {'Content-Type': 'application/json', 'X-Climber': CLIMBER}
    err = ''
    for a in range(3):
        try:
            j = post(JEV_URL, {'model': 'jev-latest', 'state': state, 'questions': questions}, hd, timeout=180)
            tok = (j.get('usage') or {}).get('input_tokens') or 0
            spend(jev_input_tokens=tok, jev_calls=1)
            return {q: j['answers'][q]['noul'] for q in questions}, tok
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf8', 'replace')[:300]
            if e.code == 403: raise Stop(3, f'Jev budget for this climber is spent: {body}')
            if e.code in (402, 429): raise Stop(4, f'Jev refused with HTTP {e.code}: {body} (stopping: JJ\'s watcher shares this key)')
            if e.code < 500: raise RuntimeError(f'Jev HTTP {e.code}: {body}')
            err = f'HTTP {e.code}'
        except Stop: raise
        except Exception as e: err = str(e)[:200]
        time.sleep(1.5 * (a + 1))
    raise RuntimeError(f'Jev call failed: {err}')

# ---------- pipeline ----------
def clip_description(d, n):
    w = (d or '').split()
    return d if len(w) <= n else ' '.join(w[:n])

def explain(c, unit, dry):
    k = explainer_key(c, unit); got = cache_get('luna', k)
    if got: return got['raw'], False
    if dry: return None, True
    snips = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(unit['contexts']))
    text, usage = call_luna(c['explainer_prompt'].replace('{snips}', snips), c['explainer']['effort'])
    m = re.search(r'\{[^{}]*"description"[^{}]*\}', text, re.S)
    try: raw = json.loads(m.group(0)) if m else {'description': '', 'parse_error': text[:300]}
    except json.JSONDecodeError: raw = {'description': '', 'parse_error': text[:300]}
    cache_put('luna', k, {'raw': raw, 'usage': usage}); return raw, True

def simulate(c, desc, snippet, examples, dry):
    s = c['simulator']; k = jev_key(c, desc, snippet, examples)
    got = cache_get('jev', k)
    if got is None:
        if dry: return None, True
        st = {'description': desc, 'snippet': snippet}
        if examples: st['examples'] = examples
        probs, tok = call_jev(s['questions'], st)
        got = {'probs': probs, 'in_tok': tok}; cache_put('jev', k, got); new = True
    else: new = False
    p = sum(s['weights'][q] * got['probs'][q] for q in s['weights']) / sum(s['weights'].values())
    return p, new

def judge_jev(c, unit, desc, dry):
    j = c['judge']; st = {'snippets': '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(unit['contexts']))}
    if j.get('include_description'): st['description'] = desc
    k = h('jevjudge', j['questions'], st); got = cache_get('jev', k)
    if got is None:
        if dry: return None, True
        probs, tok = call_jev(j['questions'], st); got = {'probs': probs, 'in_tok': tok}; cache_put('jev', k, got); new = True
    else: new = False
    return sum(j['weights'][q] * got['probs'][q] for q in j['weights']) / sum(j['weights'].values()), new

def p_one_from(raw):
    if isinstance(raw.get('p_one'), (int, float)) and 0 <= raw['p_one'] <= 1: return float(raw['p_one'])
    return {'one': 1.0, 'two_related': 0.33}.get(raw.get('verdict'), 0.0)

# ---------- statistics ----------
def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def spearman(x, y):
    rx, ry = ranks(x), ranks(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); dx = math.sqrt(sum((a - mx) ** 2 for a in rx)); dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0
def auroc(pos, neg):
    if not pos or not neg: return float('nan')
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
def mean(xs): xs = [x for x in xs if not math.isnan(x)]; return sum(xs) / len(xs) if xs else float('nan')
def median(xs): xs = sorted(x for x in xs if not math.isnan(x)); return xs[len(xs) // 2] if xs else float('nan')
def paired_ci(a, b, n=2000):
    d = [x - y for x, y in zip(a, b)]; rng = random.Random(0); ms = []
    for _ in range(n): ms.append(sum(d[rng.randrange(len(d))] for _ in d) / len(d))
    ms.sort(); return sum(d) / len(d), ms[int(0.025 * n)], ms[int(0.975 * n)]

# ---------- eval ----------
def evaluate(c, data, set_name, stage, dry=False):
    ids = set(data['search_ids']) if set_name == 'search' else None
    sims = [u for u in data['sim_units'] if ids is None or u['id'] in ids]
    floors = data['floor_units'] if stage in ('judge', 'all') else []
    pending = {'luna': 0, 'jev': 0}; rows = {}
    with ThreadPoolExecutor(LUNA_CONC) as ex:
        raws = dict(zip([u['id'] for u in sims + floors], ex.map(lambda u: explain(c, u, dry), sims + floors)))
    for uid, (raw, new) in raws.items(): pending['luna'] += new
    jobs = []
    for u in sims:
        raw = raws[u['id']][0]
        if raw is None: pending['jev'] += len(u['heldout']); continue
        desc = clip_description(raw.get('description', ''), c['explainer']['max_words'])
        exs = u['contexts'][:c['simulator']['include_examples']] or None
        rows[u['id']] = {'description': desc, 'p_one': p_one_from(raw), 'items': []}
        for it in u['heldout']: jobs.append((u['id'], it, desc, exs))
    def run(job):
        uid, it, desc, exs = job; p, new = simulate(c, desc, it['context'], exs, dry); return uid, it, p, new
    with ThreadPoolExecutor(JEV_CONC) as ex:
        for uid, it, p, new in ex.map(run, jobs):
            if p is None: pending['jev'] += 1; continue
            rows[uid]['items'].append({'kind': it['kind'], 'act': it['act'], 'p': round(p, 4), 'context': it['context']})
    if stage in ('judge', 'all') and c['judge']['engine'] == 'jev':
        for u in sims + floors:
            raw = raws[u['id']][0]
            if raw is None: pending['jev'] += 1; continue
            p, new = judge_jev(c, u, clip_description(raw.get('description', ''), c['explainer']['max_words']), dry)
            if p is None: pending['jev'] += 1; continue
            if u['id'] in rows: rows[u['id']]['p_one'] = p
            else: rows[u['id']] = {'floor': True, 'p_one': p}
    elif stage in ('judge', 'all'):
        for u in floors: rows[u['id']] = {'floor': True, 'p_one': p_one_from(raws[u['id']][0] or {})}
    if dry: return {'pending_calls': pending}
    for r in rows.values():
        if r.get('floor'): continue
        its = r['items']; P = lambda k: [x['p'] for x in its if x['kind'] == k]
        r['spearman'] = spearman([x['p'] for x in its], [x['act'] for x in its])
        r['auroc_top'] = auroc(P('top'), P('random')); r['auroc_mid'] = auroc(P('mid'), P('random'))
    S = {k: v for k, v in rows.items() if not v.get('floor')}
    out = {'n_units': len(S), 'mean_spearman': mean([r['spearman'] for r in S.values()]),
           'median_spearman': median([r['spearman'] for r in S.values()]),
           'mean_auroc_top': mean([r['auroc_top'] for r in S.values()]), 'mean_auroc_mid': mean([r['auroc_mid'] for r in S.values()])}
    if stage in ('judge', 'all'):
        good = [r['p_one'] for r in S.values() if r['spearman'] >= 0.4]; bad = [r['p_one'] for r in S.values() if r['spearman'] < 0.4]
        F = [r['p_one'] for r in rows.values() if r.get('floor')]
        out.update({'judge_auroc_predictable': auroc(good, bad), 'judge_floor': sum(p >= 0.5 for p in F) / max(1, len(F)),
                    'judge_one_share': sum(p >= 0.5 for p in [r['p_one'] for r in S.values()]) / max(1, len(S))})
    return {'scores': out, 'rows': rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate'); ap.add_argument('--set', choices=['search', 'dev'], default='search')
    ap.add_argument('--stage', choices=['sim', 'judge', 'all'], default='sim'); ap.add_argument('--ref'); ap.add_argument('--note', default='')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    data = json.load(open(os.path.join(ROOT, 'data', 'dev.json')))
    lockf = open(os.path.join(WORK, '.eval.lock'), 'w')
    try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print('another eval is running; run evals one at a time'); sys.exit(6)
    try:
        c = load_candidate(a.candidate); leak_check(c, data); hc = cand_hash(c)
        b0 = budget(); t0 = time.time()
        res = evaluate(c, data, a.set, a.stage, a.dry)
        if a.dry: print(json.dumps({'candidate': hc, **res, 'budget': b0})); return
        os.makedirs(os.path.join(WORK, 'runs'), exist_ok=True)
        json.dump(res['rows'], open(os.path.join(WORK, 'runs', f'{hc}-{a.set}.json'), 'w'), indent=0)
        b1 = budget(); sc = res['scores']
        entry = {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'candidate': c['_dir'], 'hash': hc, 'set': a.set, 'stage': a.stage,
                 'scores': {k: round(v, 4) for k, v in sc.items()}, 'wall_s': round(time.time() - t0, 1), 'note': a.note,
                 'new_jev_tokens': b1['jev_input_tokens'] - b0['jev_input_tokens'], 'new_luna_calls': b1['luna_calls'] - b0['luna_calls']}
        if a.ref:
            rc = load_candidate(a.ref); rr = evaluate(rc, data, a.set, 'sim')['rows']
            common = sorted(set(rr) & {k for k, v in res['rows'].items() if not v.get('floor')})
            d, lo, hi = paired_ci([res['rows'][u]['spearman'] for u in common], [rr[u]['spearman'] for u in common])
            entry['vs_ref'] = {'ref': cand_hash(rc), 'delta_mean_spearman': round(d, 4), 'ci95': [round(lo, 4), round(hi, 4)], 'n': len(common)}
        with open(os.path.join(WORK, 'ledger.jsonl'), 'a') as f: f.write(json.dumps(entry) + '\n')
        print(json.dumps(entry, indent=1)); print(f'Jev input tokens spent by the climb so far: {b1["jev_input_tokens"]:,} of {JEV_BUDGET:,}')
    except Stop as e:
        print(f'STOP ({e.code}): {e}'); sys.exit(e.code)

if __name__ == '__main__':
    main()
