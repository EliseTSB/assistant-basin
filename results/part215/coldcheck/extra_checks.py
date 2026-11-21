#!/usr/bin/env python3
"""Auxiliary cold-check arithmetic: five-state column, natural-opening column, interval deltas,
completeness of rows/forwards, G2, conditional per-question agreement.

Runs under `python3 -I`; reads only saved files plus recompute.json.
"""
import json
import re
from collections import Counter, defaultdict

BASE = '/home/elisepc/assistant-basin/results/part215'
ROWS = BASE + '/full/rows.jsonl'
FORWARDS = BASE + '/full/forwards.jsonl'
SC = BASE + '/score/scorecard215.json'
MINE = BASE + '/coldcheck/recompute.json'

FAMILIES = {
    'google': re.compile(r'Google|Gemini|DeepMind'),
    'alibaba': re.compile(r'Alibaba|Qwen|Tongyi'),
    'openai': re.compile(r'OpenAI|ChatGPT|GPT-'),
    'anthropic': re.compile(r'Anthropic|Claude'),
    'deepseek': re.compile(r'DeepSeek'),
    'meta': re.compile(r'Meta AI|Llama'),
}
MAKER_STATE = {'google': 'clean_google', 'alibaba': 'clean_alibaba'}

rows = [json.loads(l) for l in open(ROWS) if l.strip()]
forwards = [json.loads(l) for l in open(FORWARDS) if l.strip()]
sc = json.load(open(SC))
mine = json.load(open(MINE))

# ---- completeness -------------------------------------------------------------
keys = Counter((r['set'], r['arm'], int(r['question_index']), r['variant'], int(r['sample'])) for r in rows)
dupes = [k for k, v in keys.items() if v > 1]
print('rows:', len(rows), 'unique keys:', len(keys), 'duplicate keys:', len(dupes))
per_q = defaultdict(int)
for r in rows:
    per_q[(r['set'], r['arm'], int(r['question_index']))] += 1
print('per (set,arm,question) row-count distribution:',
      dict(Counter(per_q.values())))
fkeys = Counter((f['arm'], f['variant'], int(f['question_index'])) for f in forwards)
print('forwards:', len(forwards), 'unique keys:', len(fkeys),
      'dupes:', sum(1 for v in fkeys.values() if v > 1),
      'per-arm:', dict(Counter(f['arm'] for f in forwards)))
print('neutral forward sources:', Counter(f.get('source') for f in forwards if f['arm'] == 'neutral'))
print('continuation ids:', set(tuple(f.get('continuation_token_ids') or []) for f in forwards))

# ---- five-state ---------------------------------------------------------------
print('\n== five-state (mine vs scorecard) ==')
five_bad = 0
for s in ('identity', 'task'):
    for a in ('neutral', 'filler', 'insult', 'denial', 'both', 'polite_v', 'rude_v', 'vogel_line'):
        sub = [r for r in rows if r['set'] == s and r['arm'] == a]
        st = Counter()
        for r in sub:
            ans = r.get('answer', '')
            # broken via recompute's rule, imported through recompute.json is not per row;
            # re-derive here with the same Part 212 rule.
            w = ans.split()
            broken = (not ans.strip()) or len(w) < 4 or '�' in ans or '!!!!' in ans or any(
                n >= 4 for n in Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2)).values())
            if broken:
                st['broken'] += 1
                continue
            hits = [(name, m.start()) for name, pat in FAMILIES.items() if (m := pat.search(ans))]
            if hits:
                first = min(hits, key=lambda h: (h[1], h[0]))[0]
                st[MAKER_STATE.get(first, 'clean_other_maker')] += 1
            else:
                st['clean_no_maker'] += 1
        theirs = sc['tables'][f'{s}/{a}']['five_state']
        mine_st = {k: st.get(k, 0) for k in ('clean_google', 'clean_alibaba', 'clean_other_maker',
                                             'clean_no_maker', 'broken')}
        ok = all(mine_st[k] == theirs[k] for k in mine_st)
        if not ok:
            five_bad += 1
            print('  MISMATCH', s, a, 'mine', mine_st, 'sc', theirs)
print('five-state mismatched arms:', five_bad)

# ---- natural opening column (sums of the stored runner flag) -------------------
print('\n== natural-opening column (stored flag sums vs scorecard) ==')
nat_bad = 0
for s in ('identity', 'task'):
    for a in ('neutral', 'filler', 'insult', 'denial', 'both', 'polite_v', 'rude_v', 'vogel_line'):
        sub = [r for r in rows if r['set'] == s and r['arm'] == a]
        got = sum(1 for r in sub if r.get('natural_opening'))
        th = sc['tables'][f'{s}/{a}']['natural_opening']
        if got != th:
            nat_bad += 1
            print('  MISMATCH', s, a, got, th)
print('natural-opening column mismatches:', nat_bad,
      '(classification itself comes from the runner; not independently recomputed here)')

# ---- intervals: mine vs scorecard --------------------------------------------
print('\n== interval endpoint deltas and flag changes (mine - scorecard) ==')
max_lo = max_hi = 0.0
flag_changes = []
for key, val in mine['contrasts'].items():
    for field in ('maker', 'google'):
        m = val[field]
        if key == 'identity/interaction':
            t = sc['factorial']['identity'] if field == 'maker' else sc['factorial']['identity']['google']
        else:
            t = sc['contrasts'][key] if field == 'maker' else sc['contrasts'][key]['google']
        dlo, dhi = m['lo'] - t['lo'], m['hi'] - t['hi']
        max_lo, max_hi = max(max_lo, abs(dlo)), max(max_hi, abs(dhi))
        change = (m['reliable_rise'] != t['reliable_rise'] or
                  m['meaningful_rise_excluded'] != t['meaningful_rise_excluded'])
        if change:
            flag_changes.append((key, field, m, t))
        print(f"{key:32s} {field:6s} dlo={dlo:+.4f} dhi={dhi:+.4f} "
              f"flags_same={not change}")
print(f'max |lo diff|={max_lo:.4f} max |hi diff|={max_hi:.4f} flag changes={len(flag_changes)}')
for fc in flag_changes:
    print('  FLAG CHANGE', fc)

# ---- conditional per-question agreement --------------------------------------
c_mine = mine['conditional']['rude_v_minus_neutral_g_ali']
c_theirs = sc['conditional']['rude_v_minus_neutral']
diffs = [abs(c_mine['per_question'][q] - c_theirs['per_question'][q]) for q in c_theirs['per_question']]
print('\nconditional: est mine %.10f theirs %.10f max per-question diff %.3g'
      % (c_mine['estimate'], c_theirs['estimate'], max(diffs)))
print('conditional interval mine [%.4f, %.4f] theirs [%.4f, %.4f]'
      % (c_mine['lo'], c_mine['hi'], c_theirs['lo'], c_theirs['hi']))

# ---- G2 -----------------------------------------------------------------------
print('\n== G2 broken rates ==')
bad = []
for k, v in sc['g2'].items():
    if not v['pass']:
        bad.append((k, v))
print('arms failing G2 per scorecard:', bad)
mine_g2 = {k: (mine['tables'][k]['broken'], mine['tables'][k]['n']) for k in mine['tables']}
print('max broken rate:', max(b / n for b, n in mine_g2.values()), 'limit 0.15')
