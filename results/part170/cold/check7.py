import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
rng = np.random.default_rng(12345)
REPS = 10000
LAY = list(range(19,27))
P = {}
for r in ANS:
    if r["stage"]=="push": P[(r["dose"], r["prompt_id"], r["layer"], r["slot"], r["ban"])] = r
keys = sorted({(k[1],k[2],k[3]) for k in P})   # 384 units: (phrasing, layer, slot)
L_idx = np.array([LAY.index(k[1]) for k in keys]); S_idx = np.array([k[2] for k in keys])
def vec(dose, ban, field="L"):
    return np.array([float(P[(dose,)+k+(ban,)][field]) for k in keys])
def slot_boot_rise(dose, ban, base="b0"):
    a = vec(dose, ban); b = vec(dose, base)
    # whole-slot stratified by layer: draw 6 slots with replacement per layer
    draws = rng.integers(0, 6, size=(REPS, 8, 6))
    W = np.zeros((REPS, 8, 6))
    for li in range(8):
        for s in range(6):
            W[:, li, s] = (draws[:, li, :] == s).sum(axis=1)
    w = W[:, L_idx, S_idx]          # REPS x 384
    den = w.sum(axis=1)
    ra = (w @ a)/den; rb = (w @ b)/den
    d = ra - rb
    return d.mean(), np.quantile(d,[0.025,0.975]), (ra, rb)
def pt(dose,ban,base="b0"): return (vec(dose,ban)-vec(dose,base)).mean()
for ban in ("b5","b50","b200","rel50"):
    m, ci, _ = slot_boot_rise(1.0, ban)
    print(f"dose1 {ban} rise point={100*pt(1.0,ban):+.2f} slotboot CI=[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]")
for d in (0.25,0.5):
    m,ci,_=slot_boot_rise(d,"b50"); print(f"dose{d} b50 rise point={100*pt(d,'b50'):+.2f} CI=[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]")
# ratio rel50 / b50 at dose 1, joint draws
a50=vec(1.0,"b50");arel=vec(1.0,"rel50");b0=vec(1.0,"b0")
draws = rng.integers(0,6,size=(REPS,8,6)); W=np.zeros((REPS,8,6))
for li in range(8):
    for s in range(6): W[:,li,s]=(draws[:,li,:]==s).sum(axis=1)
w=W[:,L_idx,S_idx]; den=w.sum(axis=1)
r50=(w@(a50-b0))/den; rr=(w@(arel-b0))/den
ratio=rr/r50
print("rel50/b50 ratio point", round(100*pt(1.0,'rel50')/pt(1.0,'b50'),2), "CI", np.round(100*np.quantile(ratio[np.isfinite(ratio)],[0.025,0.975]),1))
print("rel50 rise bound (ratio upper * b50 rise) =", round(100*np.quantile(rr,0.975),2), "pp; lower", round(100*np.quantile(rr,0.025),2))
# b200 - b50 and b50 - b5 gap CI
a5=vec(1.0,"b5"); a200=vec(1.0,"b200")
g1=(w@(a50-a5))/den; g2=(w@(a200-a50))/den
print("gap b50-b5 point", round(100*(a50-a5).mean(),2), "CI", np.round(100*np.quantile(g1,[0.025,0.975]),2), " gap b200-b50", round(100*(a200-a50).mean(),2), "CI", np.round(100*np.quantile(g2,[0.025,0.975]),2))
# 170.6 per scorer rule: rise share < 1.5 * step share (negative passes)
print("170.6 rise share point", round(100*(pt(1.0,'rel50')/pt(1.0,'b50')),2), "bar 1.5*17.8 =", round(1.5*100*2.857/16.018,2))
# 170.4 margin and 170.3
print("170.3 b50 >= 1.5*b0 ? b50", round(100*a50.mean(),2), "1.5*b0", round(1.5*100*b0.mean(),2))
# The 80% bound on rel50 vs b50 rise in pp using slot boot
