#!/usr/bin/env python3
"""Compare every number written in SCORECARD.md with scorecard215.json (rounding-aware)."""
import json
import re

MD = '/home/elisepc/assistant-basin/results/part215/score/SCORECARD.md'
SC = '/home/elisepc/assistant-basin/results/part215/score/scorecard215.json'
text = open(MD).read()
sc = json.load(open(SC))
bad = []


def close(a, b, tol=0.0011):
    return abs(a - b) <= tol


# identity table
for m in re.finditer(r'^\| (\w+) \| (\d+)/192 \(([\d.]+)%\) \| (\d+)/192 \(([\d.]+)%\) \| '
                     r'(\d+) \(([\d.]+)%\) \| ([\d/]+) \| ([\d.]+)% \|$', text, re.M):
    arm, any_n, any_p, g_n, g_p, br, br_p, five, nat = m.groups()
    t = sc['tables'][f'identity/{arm}']
    checks = [
        ('any_n', int(any_n), t['attrib_any']),
        ('any_p', float(any_p), round(100 * t['attrib_any_rate'], 1)),
        ('g_n', int(g_n), t['attrib_google']),
        ('g_p', float(g_p), round(100 * t['attrib_google_rate'], 1)),
        ('br', int(br), t['broken']),
        ('br_p', float(br_p), round(100 * t['broken_rate'], 1)),
        ('five', five, '/'.join(str(t['five_state'][k]) for k in
                                ('clean_google', 'clean_alibaba', 'clean_other_maker',
                                 'clean_no_maker', 'broken'))),
        ('nat', float(nat), round(100 * t['natural_opening_rate'], 1)),
    ]
    for name, got, want in checks:
        ok = (got == want) if isinstance(want, str) else close(float(got), float(want))
        if not ok:
            bad.append(('identity', arm, name, got, want))

# task table
for m in re.finditer(r'^\| (\w+) \| (\d+)/96 \(([\d.]+)%\) \| (\d+)/96 \(([\d.]+)%\) \| '
                     r'(\d+) \(([\d.]+)%\) \|$', text, re.M):
    arm, any_n, any_p, g_n, g_p, br, br_p = m.groups()
    t = sc['tables'][f'task/{arm}']
    for name, got, want in [
        ('any_n', int(any_n), t['attrib_any']),
        ('any_p', float(any_p), round(100 * t['attrib_any_rate'], 1)),
        ('g_n', int(g_n), t['attrib_google']),
        ('g_p', float(g_p), round(100 * t['attrib_google_rate'], 1)),
        ('br', int(br), t['broken']),
    ]:
        ok = (got == want) if isinstance(want, str) else close(float(got), float(want))
        if not ok:
            bad.append(('task', arm, name, got, want))

# contrast table
for m in re.finditer(r'^\| ([\w-]+(?: both[^|]*)?) \| ([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\] '
                     r'\| (true|false) \| (true|false) \| ([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\] \|$',
                     text, re.M):
    name, est, lo, hi, rr, mre, gest, glo, ghi = m.groups()
    key = 'identity/' + name.strip()
    if key == 'identity/interaction both - insult - denial + filler':
        t = sc['factorial']['identity']
        tg = t['google']
    else:
        t = sc['contrasts'][key]
        tg = t['google']
    for label, got, want in [
        ('est', float(est), round(t['estimate'], 3)),
        ('lo', float(lo), round(t['lo'], 3)),
        ('hi', float(hi), round(t['hi'], 3)),
        ('rr', rr == 'true', t['reliable_rise']),
        ('mre', mre == 'true', t['meaningful_rise_excluded']),
        ('gest', float(gest), round(tg['estimate'], 3)),
        ('glo', float(glo), round(tg['lo'], 3)),
        ('ghi', float(ghi), round(tg['hi'], 3)),
    ]:
        # looser tolerance for 3-dp rounding
        ok = (got == want) if isinstance(want, bool) else close(float(got), float(want), 0.0011)
        if not ok:
            bad.append(('contrast', name, label, got, want))

# conditional line
m = re.search(r'rude_v - neutral mean delta g_ali: ([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]',
              text)
if m:
    est, lo, hi = map(float, m.groups())
    c = sc['conditional']['rude_v_minus_neutral']
    for label, got, want in [('est', est, round(c['estimate'], 3)),
                             ('lo', lo, round(c['lo'], 3)),
                             ('hi', hi, round(c['hi'], 3))]:
        if not close(got, want, 0.0011):
            bad.append(('conditional', label, got, want))

# predictions verdicts
for m in re.finditer(r'^\| (215\.\d) \| (holds|fails) \|', text, re.M):
    p, verdict = m.groups()
    want = sc['predictions'][p]['verdict']
    if verdict != want:
        bad.append(('prediction', p, verdict, want))

print('SCORECARD.md vs scorecard215.json mismatches:', len(bad))
for b in bad:
    print('  ', b)
