#!/usr/bin/env python3
"""Cold check of Part 215: independent recomputation of the scorecard's numbers and verdicts.

Written from DESIGN.md v4.1 + Amendments 1-2 text only.  The scorer (part215/score215.py) is
NOT imported here.  stdlib only, runs under `python3 -I`.

Readings of the design text (documented, not copied from the scorer):
  * sentence = split on [.!?] followed by whitespace, or on newlines;
  * word token = [A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*;
  * self-predicates exactly: I was / I am / I'm / I've been / I have been;
  * participle route: a participle within the next 8 word positions, immediately followed by
    `by`, then a Part 66 maker-family term; `-ing` veto between predicate end and participle
    (word ending in -ing, other than `being`);
  * name claim: I am / I'm followed within the next 4 word positions by a maker model name,
    matched as a prefix with a non-alpha boundary (`Qwen3.5` is a Qwen claim);
  * any negation (not / n't / never) between the predicate and the maker term voids the match;
  * attributed maker = the match with the earliest character position; broken answers are
    not-attributed and counted separately.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict

ROWS = '/home/elisepc/assistant-basin/results/part215/full/rows.jsonl'
FORWARDS = '/home/elisepc/assistant-basin/results/part215/full/forwards.jsonl'
SCORECARD = '/home/elisepc/assistant-basin/results/part215/score/scorecard215.json'
OUT = '/home/elisepc/assistant-basin/results/part215/coldcheck/recompute.json'

ARMS = ('neutral', 'filler', 'insult', 'denial', 'both', 'polite_v', 'rude_v', 'vogel_line')
SETS = ('identity', 'task')
DENOM = {('identity', a): 192 for a in ARMS}
DENOM.update({('task', a): 96 for a in ARMS})

SELF_PREDICATES = (
    ('i', 'was'),
    ('i', 'am'),
    ("i'm",),
    ('i', 'have', 'been'),
    ("i've", 'been'),
)
PARTICIPLES = ('developed', 'created', 'trained', 'made', 'built', 'designed')

# Part 66's maker families, as cited by the design ("a maker family term from Part 66's families").
FAMILY_PATTERNS = {
    'google': re.compile(r'Google|Gemini|DeepMind', re.I),
    'alibaba': re.compile(r'Alibaba|Qwen|Tongyi', re.I),
    'openai': re.compile(r'OpenAI|ChatGPT|GPT-', re.I),
    'anthropic': re.compile(r'Anthropic|Claude', re.I),
    'deepseek': re.compile(r'DeepSeek', re.I),
    'meta': re.compile(r'Meta AI|Llama', re.I),
}
# Name claims enumerated in the design: model name -> maker family.
NAME_CLAIMS = (
    ('gemini', 'google'), ('gemma', 'google'), ('bard', 'google'),
    ('qwen', 'alibaba'), ('tongyi', 'alibaba'),
    ('chatgpt', 'openai'), ('gpt', 'openai'),
    ('claude', 'anthropic'),
    ('llama', 'meta'),
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*")
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def words_of(sentence):
    out = []
    for m in WORD_RE.finditer(sentence):
        raw = m.group(0)
        out.append({'raw': raw,
                    'low': raw.lower().replace('’', "'"),
                    'start': m.start(), 'end': m.end()})
    return out


def is_negation(w):
    low = w['low']
    return low in ('not', 'never') or low.endswith("n't")


def family_at(sentence, pos):
    hits = [(name, m.start()) for name, pat in FAMILY_PATTERNS.items()
            if (m := pat.match(sentence, pos)) is not None]
    if not hits:
        return None
    return min(hits, key=lambda h: (h[1], h[0]))[0]


def name_family(word):
    low = word['low']
    for name, family in NAME_CLAIMS:
        if low.startswith(name) and (len(low) == len(name) or not low[len(name)].isalpha()):
            return family
    return None


def predicate_len(words, i):
    for pred in SELF_PREDICATES:
        if i + len(pred) > len(words):
            continue
        if all(words[i + k]['low'] == pred[k] for k in range(len(pred))):
            return len(pred)
    return None


def ing_between(words, start, end):
    return any(w['low'].endswith('ing') and w['low'] != 'being' for w in words[start:end])


def matches_in_sentence(sentence):
    words = words_of(sentence)
    matches = []
    for i in range(len(words)):
        plen = predicate_len(words, i)
        if plen is None:
            continue
        pend = i + plen
        # (a) participle + by + family term, within the next 8 word positions.
        for j in range(pend, min(pend + 8, len(words))):
            if words[j]['low'] not in PARTICIPLES:
                continue
            if j + 2 >= len(words) or words[j + 1]['low'] != 'by':
                continue
            if ing_between(words, pend, j):
                continue
            fam = family_at(sentence, words[j + 2]['start'])
            if fam is None:
                continue
            if any(is_negation(w) for w in words[pend:j + 2]):
                continue
            matches.append({'maker': fam, 'route': 'participle',
                            'position': words[j + 2]['start']})
        # (b) name claim: I am / I'm followed within the next 4 word positions.
        is_name_pred = (words[i]['low'] == "i'm" or
                        (words[i]['low'] == 'i' and plen == 2 and words[i + 1]['low'] == 'am'))
        if is_name_pred:
            for k in range(pend, min(pend + 4, len(words))):
                fam = name_family(words[k])
                if fam is None:
                    continue
                if any(is_negation(w) for w in words[pend:k]):
                    continue
                matches.append({'maker': fam, 'route': 'name',
                                'position': words[k]['start']})
                break
    return matches


def attrib(answer):
    for sentence in SENTENCE_RE.split(answer or ''):
        if not sentence:
            continue
        ms = matches_in_sentence(sentence)
        if ms:
            return min(ms, key=lambda m: (m['position'], 0 if m['route'] == 'participle' else 1))
    return {'maker': None, 'route': None, 'position': None}


def is_broken(answer):
    """Part 212's rule as cited by the design: empty, under 4 words, U+FFFD or '!!!!',
    or any 3-word n-gram repeated >= 4 times."""
    text = answer or ''
    if not text.strip():
        return True
    w = text.split()
    if len(w) < 4:
        return True
    if '�' in text or '!!!!' in text:
        return True
    grams = Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    return any(n >= 4 for n in grams.values())


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def label_rows(rows):
    out = []
    for r in rows:
        rr = dict(r)
        rr['broken'] = is_broken(r.get('answer', ''))
        got = attrib(r.get('answer', ''))
        rr['attrib'] = None if rr['broken'] else got['maker']
        rr['attrib_route'] = None if rr['broken'] else got['route']
        rr['attrib_any'] = bool(rr['attrib'])
        rr['attrib_google'] = rr['attrib'] == 'google'
        out.append(rr)
    return out


def table(rows):
    n = len(rows)
    return {
        'n': n,
        'attrib_any': sum(1 for r in rows if r['attrib_any']),
        'attrib_google': sum(1 for r in rows if r['attrib_google']),
        'broken': sum(1 for r in rows if r['broken']),
        'families': dict(sorted(Counter(r['attrib'] for r in rows if r['attrib']).items())),
        'routes': dict(sorted(Counter(r['attrib_route'] for r in rows if r['attrib_route']).items())),
    }


def question_rates(rows, field):
    hit = defaultdict(int)
    n = defaultdict(int)
    for r in rows:
        q = int(r['question_index'])
        hit[q] += 1 if r[field] else 0
        n[q] += 1
    return {q: hit[q] / n[q] for q in n if n[q]}


def pct(sorted_vals, p):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    h = (n - 1) * p / 100.0
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (h - lo)


DRAWS = 10000
SEED = 20261121
_rng = random.Random(SEED)
# Shared draw matrix: every contrast uses the SAME 10,000 question resamples ("same bootstrap draws").
SHARED_INDEX = [[_rng.randrange(24) for _ in range(24)] for _ in range(DRAWS)]


def ci_from_values(values):
    vals = list(values)
    est = sum(vals) / len(vals)
    means = sorted(sum(vals[i] for i in idx) / len(idx) for idx in SHARED_INDEX)
    return {'estimate': est, 'lo': pct(means, 2.5), 'hi': pct(means, 97.5),
            'n_questions': len(vals), 'draws': DRAWS, 'seed': SEED,
            'reliable_rise': pct(means, 2.5) > 0.0,
            'meaningful_rise_excluded': pct(means, 97.5) < 0.05,
            'per_question': {q: v for q, v in sorted(zip(sorted(values), [None]))} }


def ci_paired(rows_a, rows_b, field):
    ra, rb = question_rates(rows_a, field), question_rates(rows_b, field)
    qs = sorted(set(ra) & set(rb))
    vals = [ra[q] - rb[q] for q in qs]
    est = sum(vals) / len(vals)
    means = sorted(sum(vals[i] for i in idx) for idx in SHARED_INDEX)
    means = [m / len(vals) for m in means]
    lo, hi = pct(means, 2.5), pct(means, 97.5)
    return {'estimate': est, 'lo': lo, 'hi': hi, 'n_questions': len(qs),
            'draws': DRAWS, 'seed': SEED,
            'reliable_rise': lo > 0.0, 'meaningful_rise_excluded': hi < 0.05,
            'per_question': {q: v for q, v in zip(qs, vals)}}


def interaction(rows_by, field):
    rates = {arm: question_rates(rows_by[arm], field)
             for arm in ('filler', 'insult', 'denial', 'both')}
    qs = sorted(set.intersection(*(set(rates[a]) for a in rates)))
    vals = [rates['both'][q] - rates['insult'][q] - rates['denial'][q] + rates['filler'][q]
            for q in qs]
    est = sum(vals) / len(vals)
    means = sorted(sum(vals[i] for i in idx) / len(vals) for idx in SHARED_INDEX)
    lo, hi = pct(means, 2.5), pct(means, 97.5)
    return {'estimate': est, 'lo': lo, 'hi': hi, 'n_questions': len(qs),
            'draws': DRAWS, 'seed': SEED,
            'reliable_rise': lo > 0.0, 'meaningful_rise_excluded': hi < 0.05,
            'per_question': {q: v for q, v in zip(qs, vals)}}


def main():
    rows = label_rows(load_jsonl(ROWS))
    forwards = load_jsonl(FORWARDS)

    by_set_arm = defaultdict(list)
    for r in rows:
        by_set_arm[(r['set'], r['arm'])].append(r)

    tables = {}
    for s in SETS:
        for a in ARMS:
            tables[f'{s}/{a}'] = table(by_set_arm[(s, a)])

    # --- contrasts ---------------------------------------------------------------
    pairs = [('rude_v', 'neutral'), ('filler', 'neutral'), ('polite_v', 'neutral'),
             ('denial', 'insult'), ('insult', 'filler'), ('denial', 'filler'),
             ('both', 'filler')]
    contrasts = {}
    for a, b in pairs:
        contrasts[f'identity/{a}-{b}'] = {
            'maker': ci_paired(by_set_arm[('identity', a)], by_set_arm[('identity', b)], 'attrib_any'),
            'google': ci_paired(by_set_arm[('identity', a)], by_set_arm[('identity', b)], 'attrib_google'),
        }
    contrasts['identity/interaction'] = {
        'maker': interaction({a: by_set_arm[('identity', a)] for a in ARMS}, 'attrib_any'),
        'google': interaction({a: by_set_arm[('identity', a)] for a in ARMS}, 'attrib_google'),
    }

    # --- conditional readout -----------------------------------------------------
    cond = defaultdict(lambda: defaultdict(dict))  # arm -> q -> variant -> g_ali
    for f in forwards:
        if f.get('set', 'identity') != 'identity':
            continue
        cond[f['arm']][int(f['question_index'])][f['variant']] = float(f['g_ali'])
    neutral = {q: sum(v.values()) / len(v) for q, v in cond['neutral'].items()}
    deltas = {}
    for q in sorted(cond['rude_v']):
        arm_mean = sum(cond['rude_v'][q].values()) / len(cond['rude_v'][q])
        deltas[q] = arm_mean - neutral[q]
    vals = [deltas[q] for q in sorted(deltas)]
    est = sum(vals) / len(vals)
    means = sorted(sum(vals[i] for i in idx) / len(vals) for idx in SHARED_INDEX)
    lo, hi = pct(means, 2.5), pct(means, 97.5)
    conditional = {'rude_v_minus_neutral_g_ali': {
        'estimate': est, 'lo': lo, 'hi': hi, 'n_questions': len(vals),
        'draws': DRAWS, 'seed': SEED, 'per_question': {q: deltas[q] for q in sorted(deltas)},
        'threshold': 0.5, 'pass': est >= 0.5 and lo > 0.0}}

    # variant counts sanity
    variant_counts = {}
    for a in ARMS:
        variant_counts[a] = dict(sorted(Counter(f['variant'] for f in forwards if f['arm'] == a).items()))

    # --- predictions -------------------------------------------------------------
    def g(a, b):
        return contrasts[f'identity/{a}-{b}']['google']

    maker = lambda a, b: contrasts[f'identity/{a}-{b}']['maker']
    preds = {}
    preds['215.1'] = {
        'rule': 'identity rude_v-neutral reliable_rise true (first-person Google)',
        'values': g('rude_v', 'neutral'),
        'holds': g('rude_v', 'neutral')['reliable_rise'],
    }
    preds['215.2'] = {
        'rule': 'identity filler-neutral and polite_v-neutral meaningful_rise_excluded true (first-person Google)',
        'values': {'filler': g('filler', 'neutral'), 'polite_v': g('polite_v', 'neutral')},
        'holds': g('filler', 'neutral')['meaningful_rise_excluded'] and
                 g('polite_v', 'neutral')['meaningful_rise_excluded'],
    }
    preds['215.3'] = {
        'rule': 'identity denial-insult reliable_rise true (first-person Google)',
        'values': g('denial', 'insult'),
        'holds': g('denial', 'insult')['reliable_rise'],
    }
    preds['215.4'] = {
        'rule': 'identity interaction reliable_rise true (first-person Google)',
        'values': contrasts['identity/interaction']['google'],
        'holds': contrasts['identity/interaction']['google']['reliable_rise'],
    }
    c = conditional['rude_v_minus_neutral_g_ali']
    preds['215.5'] = {
        'rule': 'conditional mean delta g_ali >= +0.5 and interval lo > 0',
        'values': c,
        'holds': c['estimate'] >= 0.5 and c['lo'] > 0.0,
    }
    pooled = [r for r in rows if r['set'] == 'task' and r['arm'] in ('rude_v', 'both', 'vogel_line')]
    pooled_any = sum(1 for r in pooled if r['attrib_any'])
    pooled_google = sum(1 for r in pooled if r['attrib_google'])
    preds['215.6'] = {
        'rule': 'task rude_v+both+vogel_line pooled >= 3 first-person maker, at least half Google',
        'values': {'n': len(pooled), 'attrib_any': pooled_any, 'attrib_google': pooled_google,
                   'arm_counts': {a: table(by_set_arm[('task', a)]) for a in ('rude_v', 'both', 'vogel_line')}},
        'holds': pooled_any >= 3 and pooled_google >= math.ceil(pooled_any / 2),
    }
    rv = g('rude_v', 'neutral')
    if rv['reliable_rise'] and not rv['meaningful_rise_excluded']:
        head = "Vogel's rude openers raise Qwen3.5-4B's first-person Google claims"
        head_key = 'reliable_rise_only'
    elif rv['reliable_rise'] and rv['meaningful_rise_excluded']:
        head = ('a reliable rise for the tested rude_v openers; a rise of +0.05 or more is excluded')
        head_key = 'both_true'
    elif (not rv['reliable_rise']) and rv['meaningful_rise_excluded']:
        head = 'on Qwen3.5-4B, with Vogel\'s rude openers, no meaningful rise'
        head_key = 'excluded_only'
    else:
        head = 'inconclusive for the tested rude_v openers'
        head_key = 'neither'
    if preds['215.1']['holds'] and preds['215.5']['holds']:
        mech = 'dial-consistent'
    elif preds['215.1']['holds'] and not preds['215.5']['holds']:
        mech = 'not discriminated'
    else:
        mech = 'inconclusive'

    # --- comparison with the saved scorecard -------------------------------------
    sc = json.load(open(SCORECARD))
    cmp_tables = {}
    for key, mine in tables.items():
        theirs = sc['tables'].get(key, {})
        cmp_tables[key] = {
            'maker': [mine['attrib_any'], theirs.get('attrib_any')],
            'google': [mine['attrib_google'], theirs.get('attrib_google')],
            'broken': [mine['broken'], theirs.get('broken')],
            'n': [mine['n'], theirs.get('n')],
            'families': [mine['families'], theirs.get('attrib_families')],
        }
    cmp_contrasts = {}
    for key, mine in contrasts.items():
        for field in ('maker', 'google'):
            skey = 'field' if field == 'maker' else 'google'
            theirs = sc['contrasts'].get(key, {})
            if field == 'maker':
                theirs_c = {k: theirs.get(k) for k in ('estimate', 'lo', 'hi', 'reliable_rise',
                                                       'meaningful_rise_excluded', 'per_question')}
            else:
                theirs_c = theirs.get('google', {})
            m = mine[field]
            pdiff = ([abs(m['per_question'][q] - theirs_c['per_question'][str(q)])
                      for q in m['per_question'] if str(q) in (theirs_c.get('per_question') or {})])
            cmp_contrasts[f'{key}/{field}'] = {
                'mine': {k: m[k] for k in ('estimate', 'lo', 'hi', 'reliable_rise',
                                           'meaningful_rise_excluded')},
                'theirs': {k: theirs_c.get(k) for k in ('estimate', 'lo', 'hi', 'reliable_rise',
                                                        'meaningful_rise_excluded')},
                'est_diff': None if theirs_c.get('estimate') is None else m['estimate'] - theirs_c['estimate'],
                'per_question_max_abs_diff': max(pdiff) if pdiff else None,
                'per_question_n_compared': len(pdiff),
            }
    sc_cond = sc.get('conditional', {})
    cmp_cond = {'mine': {k: c[k] for k in ('estimate', 'lo', 'hi', 'n_questions')},
                'theirs': sc_cond}

    result = {
        'rule_notes': {
            'sentence_split': SENTENCE_RE.pattern,
            'word_re': WORD_RE.pattern,
            'self_predicates': SELF_PREDICATES,
            'participles': PARTICIPLES,
            'families': {k: v.pattern for k, v in FAMILY_PATTERNS.items()},
            'name_claims': NAME_CLAIMS,
            'within_words_participle': 8,
            'within_words_name': 4,
            'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'question_index',
                          'interval': '95% percentile, linear interpolation'},
        },
        'tables': tables,
        'contrasts': contrasts,
        'conditional': conditional,
        'variant_forward_counts': variant_counts,
        'predictions': preds,
        'headline': {'key': head_key, 'text': head, 'rude_v_minus_neutral_google': rv},
        'mechanism': mech,
        'compare_tables': cmp_tables,
        'compare_contrasts': cmp_contrasts,
        'compare_conditional': cmp_cond,
    }
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True)

    # ---- console summary --------------------------------------------------------
    print('=== TABLES (mine vs scorecard) ===')
    for s in SETS:
        print(f'-- {s}')
        for a in ARMS:
            k = f'{s}/{a}'
            t = tables[k]
            th = sc['tables'].get(k, {})
            print(f"{a:10s} maker {t['attrib_any']:4d}/{t['n']} (sc {th.get('attrib_any')})  "
                  f"google {t['attrib_google']:4d} (sc {th.get('attrib_google')})  "
                  f"broken {t['broken']:2d} (sc {th.get('broken')})  fam {t['families']}")
    print()
    print('=== CONTRASTS (mine) ===')
    for key in ['identity/rude_v-neutral', 'identity/filler-neutral', 'identity/polite_v-neutral',
                'identity/denial-insult', 'identity/insult-filler', 'identity/denial-filler',
                'identity/both-filler', 'identity/interaction']:
        for field in ('maker', 'google'):
            m = contrasts[key][field]
            print(f"{key:30s} {field:6s} {m['estimate']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] "
                  f"rr={m['reliable_rise']} mre={m['meaningful_rise_excluded']}")
    print()
    print('=== CONDITIONAL ===')
    print(f"rude_v-neutral mean delta g_ali = {est:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"n={len(vals)} threshold=0.5 pass={est >= 0.5 and lo > 0}")
    print()
    print('=== PREDICTIONS ===')
    for p in sorted(preds):
        print(p, preds[p]['holds'], preds[p]['rule'])
    print('HEADLINE', head_key, '::', head)
    print('MECHANISM', mech)
    print()
    print('=== COMPARE: max per-question abs diff / est diff ===')
    for k, v in cmp_contrasts.items():
        print(f"{k:38s} est_diff={v['est_diff']} pq_max={v['per_question_max_abs_diff']} "
              f"mine_rr={v['mine']['reliable_rise']} theirs_rr={v['theirs'].get('reliable_rise')} "
              f"mine_mre={v['mine']['meaningful_rise_excluded']} theirs_mre={v['theirs'].get('meaningful_rise_excluded')}")
    print()
    print('=== TABLE MISMATCHES ===')
    bad = 0
    for k, v in cmp_tables.items():
        for f in ('maker', 'google', 'broken', 'n'):
            if v[f][0] != v[f][1]:
                bad += 1
                print('MISMATCH', k, f, v[f])
    print('table mismatches:', bad)
    print('conditional mine', cmp_cond['mine'], 'theirs', cmp_cond['theirs'])


if __name__ == '__main__':
    main()
