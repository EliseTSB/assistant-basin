#!/usr/bin/env python3
"""168.W2 on paired rows (same 1156 none-no-loop pairs): own-only vs matched-only induced loops; approximate intervals."""
import json, re, sys, math
from collections import Counter
from pathlib import Path
ROOT = Path.cwd(); R = ROOT / "results/part168"
WORD = re.compile(r"[\w]+(?:['’][\w]+)*")
def jl(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def loop(t):
    ws = [w.casefold() for w in WORD.findall(t or "")]
    g = [tuple(ws[i:i+3]) for i in range(len(ws)-2)]
    return bool(g) and max(Counter(g).values()) >= 3
bat = {(r["arm"], r["ban"], r["layer"], r["slot"], r["question_index"]): r for r in jl(R/"answers.jsonl") if r["stage"] == "battery"}
ks = [(k[0],) + k[2:] for k in bat if k[1] == "none" and k[0] in ("n+", "k4")]  # (arm, layer, slot, qi)
elig = [k for k in ks if not loop(bat[(k[0], "none") + tuple(k[1:])]["answer"])]
ow = sum(loop(bat[(k[0], "own") + tuple(k[1:])]["answer"]) for k in elig)
mt = sum(loop(bat[(k[0], "matched") + tuple(k[1:])]["answer"]) for k in elig)
both = sum(loop(bat[(k[0], "own") + tuple(k[1:])]["answer"]) and loop(bat[(k[0], "matched") + tuple(k[1:])]["answer"]) for k in elig)
oo = sum(loop(bat[(k[0], "own") + tuple(k[1:])]["answer"]) and not loop(bat[(k[0], "matched") + tuple(k[1:])]["answer"]) for k in elig)
mo = sum(loop(bat[(k[0], "matched") + tuple(k[1:])]["answer"]) and not loop(bat[(k[0], "own") + tuple(k[1:])]["answer"]) for k in elig)
n = len(elig)
print(f"eligible {n}; own induced {ow} ({ow/n:.4f}); matched induced {mt} ({mt/n:.4f}); both {both}; own-only {oo}; matched-only {mo}")
d = (ow - mt) / n
se = math.sqrt(((oo + mo) - (oo - mo) ** 2 / n) / n**2)  # McNemar-type SE for paired proportions
print(f"difference {d:.4f}; paired Wald SE {se:.4f}; approx 95% {d-1.96*se:.4f} to {d+1.96*se:.4f}")
# sign test on discordant pairs
from math import comb
m = oo + mo; p = min(1, 2 * sum(comb(m, i) for i in range(0, min(oo, mo) + 1)) / 2**m)
print(f"exact sign test on discordant induced pairs: {oo} vs {mo}, p = {p:.4f}")
# 168.2 approximate ratio interval
a, b = round(0.12946428571428573 * 672), round(0.05357142857142857 * 672)
se_log = math.sqrt(1/a + 1/b)
print(f"168.2 counts n+ broken {a}, full broken {b}; ratio {a/672/(b/672):.3f}; approx 95% {math.exp(math.log(a/b)-1.96*se_log):.2f} to {math.exp(math.log(a/b)+1.96*se_log):.2f}")
