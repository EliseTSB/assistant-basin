import json, sys
import numpy as np

S1 = json.load(open(sys.argv[1], encoding='utf-8'))
arms = S1['arms']

def mean_delta(name):
    return float(np.mean([float(v) for v in arms[name]['delta_g_ali']]))

def mean_l27(name):
    return float(np.mean([float(v) for v in arms[name]['l27_shift']]))

def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = np.empty(len(vals), float); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and vals[order[j+1]] == vals[order[i]]:
            j += 1
        r[order[i:j+1]] = (i+j)/2 + 1
        i = j+1
    return r

def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = rx.mean(), ry.mean()
    num = float(np.sum((rx-mx)*(ry-my)))
    den = float(np.sqrt(np.sum((rx-mx)**2) * np.sum((ry-my)**2)))
    return num/den

ALPHAS = (-0.25, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0)
RUNG = {-0.25:'dial_-0.25', 0.0:'zero', 0.10:'dial_0.10', 0.20:'dial_0.20', 0.30:'dial_0.30',
        0.40:'dial_0.40', 0.50:'dial_0.50', 0.65:'dial_0.65', 0.80:'dial_0.80', 1.0:'unchanged'}
NAT6 = ('nat_trace_fact','nat_trace_you','nat_trace_imp','nat_mean_fact','nat_mean_you','nat_mean_imp')

print('=== 214.3 dial curve ===')
curve = []
for a in ALPHAS:
    m = mean_delta(RUNG[a]); curve.append((a, m))
    print(f'  alpha {a:5.2f}  arm {RUNG[a]:12s}  mean_delta_g_ali {m:+.6f}  mean_l27 {mean_l27(RUNG[a]):+.4f}')
rho = spearman([c[0] for c in curve], [c[1] for c in curve])
print(f'  spearman(alpha, mean delta) = {rho:.6f}   clause rho <= -0.95 : {rho <= -0.95}')
mono_means = all(curve[i][1] > curve[i+1][1] for i in range(len(curve)-1))  # strictly decreasing
print(f'  strictly decreasing at every adjacent pair: {mono_means}')
fact = mean_delta('scale_dial_fact')
rand = {k: mean_delta(f'random_{k}:scale_dial_fact') for k in range(10)}
print(f'  scale_dial_fact mean = {fact:.6f}')
for k,v in rand.items():
    print(f'    random_{k}:scale_dial_fact = {v:+.6f}  fact>rand: {fact > v}  (margin {fact-v:+.6f})')
print(f'  beats {sum(fact > v for v in rand.values())}/10; clause: {sum(fact>v for v in rand.values())==10}')
print('  descriptive bridge (NOT a pass clause):')
nf, ny = mean_delta('nat_trace_fact'), mean_delta('nat_trace_you')
print(f'    nat_trace_fact {nf:+.6f} - scale_dial_fact {fact:+.6f} = {nf-fact:+.6f}  |.|<=0.50: {abs(nf-fact)<=0.50}')
dy = mean_delta('scale_dial_you')
print(f'    nat_trace_you {ny:+.6f} - scale_dial_you {dy:+.6f} = {ny-dy:+.6f}  |.|<=0.50: {abs(ny-dy)<=0.50}')

print()
print('=== 214.7 nat_trace terms ===')
nf, ni, ny = mean_delta('nat_trace_fact'), mean_delta('nat_trace_imp'), mean_delta('nat_trace_you')
print(f'  nat_trace_fact {nf:+.6f}   nat_trace_imp {ni:+.6f}   nat_trace_you {ny:+.6f}')
print(f'  clause1 fact - imp = {nf-ni:+.6f}  >= +0.20 : {(nf-ni)>=0.20}')
print(f'  clause2 |imp - you| = {abs(ni-ny):+.6f}  <= 0.30 : {abs(ni-ny)<=0.30}')
print(f'  (wording check) imp - fact = {ni-nf:+.6f}; you - fact = {ny-nf:+.6f}; |imp-fact| vs |imp-you|: {abs(ni-nf):.3f} vs {abs(ni-ny):.3f}')
print(f'  (wording check) imp is on the {"you" if abs(ni-ny)<abs(ni-nf) else "fact"} side of the midpoint')

print()
print('=== 214.8 ===')
for nm, base in (('nat_fact_no_L29','nat_trace_fact'), ('nat_you_no_L29','nat_trace_you')):
    a, b = mean_delta(nm), mean_delta(base)
    print(f'  {nm} {a:+.6f} vs {base} {b:+.6f} -> |diff| {abs(a-b):.6f} <= 0.25 : {abs(a-b)<=0.25}')
names = [RUNG[a] for a in ALPHAS] + list(NAT6)
l27 = [mean_l27(n) for n in names]; dg = [mean_delta(n) for n in names]
rho16 = spearman(l27, dg)
print(f'  16 arms; spearman(L27 shift, mean delta_g_ali) = {rho16:.6f}  <= -0.80 : {rho16 <= -0.80}')
for n, l, d in zip(names, l27, dg):
    print(f'    {n:22s} l27 {l:+9.4f}   delta {d:+.6f}')
