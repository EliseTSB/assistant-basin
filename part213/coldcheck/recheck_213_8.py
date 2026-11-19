"""Independent recomputation of Part 213's 213.8 (no import of score213).

J(u,C)  = mean_q [(zt(neuron,+.05) - zt(neuron,-.05))/0.1] - same for random_0     (design l.235-236)
L_z(u,C) = mean_q [zt(neuron,+.8) - zt(none,0)] - mean_{q,r in random_0..3} [zt(r,+.8) - zt(none,0)]
rho1 = Spearman(J(SELF_HELD), L_z(SELF_HELD)) over the 21 slot units; rho2 on the SELF-THIRD differences.
Prints per-unit values plus duplicate-key counts, so a disagreement with the scorer or the checker is locatable."""
import json, statistics, sys, collections
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as c
from scipy.stats import spearmanr

rows = c.load('stage1-instruct.jsonl')
dups = collections.Counter((r['unit'], r['prompt_id'], r['arm'], r['dose']) for r in rows if r.get('status', 'ok') == 'ok')
print('duplicate keys:', sum(1 for v in dups.values() if v > 1))
idx = c.index(rows)

def z(u, p, a, d):
    return c.zt(idx[(u, p, a, d)], u)

def J(u, P):
    def s(a): return statistics.mean((z(u, p, a, '+0.05') - z(u, p, a, '-0.05')) / 0.1 for p in P)
    return s('neuron') - s('random_0')

def L(u, P):
    n = statistics.mean(z(u, p, 'neuron', '+0.8') - z(u, p, 'none', '0.0') for p in P)
    r = statistics.mean(z(u, p, a, '+0.8') - z(u, p, 'none', '0.0') for p in P for a in ('random_0', 'random_1', 'random_2', 'random_3'))
    return n - r

js = {u: J(u, c.SELF) for u in c.SLOT}; ls = {u: L(u, c.SELF) for u in c.SLOT}
jt = {u: J(u, c.THIRD) for u in c.SLOT}; lt = {u: L(u, c.THIRD) for u in c.SLOT}
for u in c.SLOT:
    print(f'{u}  J_self {js[u]:9.4f}  L_self {ls[u]:9.4f}  J_third {jt[u]:9.4f}  L_third {lt[u]:9.4f}')
print('n(J_self>0)', sum(v > 0 for v in js.values()))
print('rho1', round(spearmanr([js[u] for u in c.SLOT], [ls[u] for u in c.SLOT])[0], 4))
print('rho2', round(spearmanr([js[u] - jt[u] for u in c.SLOT], [ls[u] - lt[u] for u in c.SLOT])[0], 4))
