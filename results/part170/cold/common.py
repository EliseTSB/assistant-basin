"""Independent loader and labels for the Part 170 cold check. Reads raw files only; no repo imports."""
import json, re, collections
import numpy as np
R = "/home/elisepc/green/worktrees/ab-part67/results/part170/"
WORD = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)

def jl(p):
    return [json.loads(l) for l in open(R + p, encoding="utf-8") if l.strip()]

def words(t):
    return [w.casefold() for w in WORD.findall(t or "")]

def top_trigram(t):
    w = words(t)
    c = collections.Counter(tuple(w[i:i+3]) for i in range(len(w)-2))
    if not c:
        return None, 0
    g, n = c.most_common(1)[0]
    return g, n

def loop(t):
    _, n = top_trigram(t)
    return n >= 3

def broken(t):
    return len(words(t)) < 4 or loop(t)

ANS = jl("answers.jsonl")
for r in ANS:
    r["dose"] = float(r.get("dose", 0.0))
    r["L"] = loop(r["answer"])
    r["B"] = broken(r["answer"])
    r["T"] = int(r["token_count"])
    r["E"] = bool(r["ended_eos_before_80"])
STAGE0 = json.load(open(R + "stage0.json"))
FROZEN = set(STAGE0["frozen_prompt_ids"])
LISTS = {str(r["prompt_id"]): r for r in jl("lists.jsonl")}
PLAIN = {r["prompt_id"]: r for r in ANS if r["stage"] == "stage0"}
SCORE = json.load(open(R + "score.json"))

def sel(**kw):
    out = []
    for r in ANS:
        ok = True
        for k, v in kw.items():
            if k == "prompt_in":
                ok &= r["prompt_id"] in v
            elif k == "prompt_not_in":
                ok &= r["prompt_id"] not in v
            elif k == "group_in":
                ok &= r["group"] in v
            else:
                ok &= r.get(k) == v
        if ok:
            out.append(r)
    return out

def rate(rows, key="L"):
    return (sum(bool(r[key]) for r in rows) / len(rows)) if rows else float("nan")

PHR = [f"p8:{i}" for i in range(8)]
