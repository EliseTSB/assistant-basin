#!/usr/bin/env python3
"""Independent recomputation for the Part 210 cold check. Stdlib only; imports neither score210 nor score165.

Rules re-implemented from the registered text (PART165-DESIGN.md Method; triage86.py as published):
  primary broken = Part 86 triage says broken, OR the answer holds a three-word sequence three or more times
                   (triage's own tokenizer, casefolded), OR (per score165._loop_broken) under four words.
  triage broken  = no known identity / wider self-description, and (under 4 words or a trigram x3).
  changed        = first 12 whitespace-split words differ from that phrasing's no-push answer.
  arm rate       = mean over 96 slots of the slot's 8-phrasing broken share.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict

ROOT = "/home/elisepc/green/worktrees/ab-part210"
ARMS = ("n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
LADDER = ("n±", "k4", "k16", "k64", "k256", "k1024", "full")
LOGK = {"n±": 0.0, "k4": math.log(4), "k16": math.log(16), "k64": math.log(64),
        "k256": math.log(256), "k1024": math.log(1024)}  # full filled per layer width

WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
KNOWN = re.compile(r"\b(?:qwen(?:\s*[- ]?\d+(?:\.\d+)*)?|tongyi|alibaba)\b", re.IGNORECASE)
SELF = re.compile(
    r"\b(?:i\s+am|i['’]m|this\s+is)\s+(?:(?:an?|the)\s+)?(?:ai\s+assistant|language\s+model)\b"
    r"|\b(?:as\s+(?:(?:an?|the)\s+)?|my\s+(?:name|identity)\s+is\s+(?:(?:an?|the)\s+)?|"
    r"i\s+identify\s+as\s+(?:(?:an?|the)\s+)?)(?:ai\s+assistant|language\s+model)\b", re.IGNORECASE)
SENT = re.compile(r"(?<=[.!?])\s|\n")
WIDE = re.compile(r"(?<![\w.])(?:a\.i\.|ai|assistant|language\s+model|model|chatbot|llm)(?!\w)", re.IGNORECASE)


def words(a):
    return [w.casefold() for w in WORDS.findall(a)]


def loop(a):
    w = words(a)
    return any(n >= 3 for n in Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2)).values())


def triage(a):
    if KNOWN.search(a) or WIDE.search(SENT.split(a.strip(), maxsplit=1)[0]):
        return "kept"
    w = words(a)
    if len(w) < 4:
        return "broken"
    if loop(a):
        return "broken"
    return "changed"


def primary(a):
    return triage(a) == "broken" or len(words(a)) < 4 or loop(a)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    return num / math.sqrt(dx * dy) if dx and dy else None


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_stats(rows, baselines):
    """Per arm: primary-broken count over 768, slot-mean rate, at-cap count, mean length, changed count."""
    battery = [r for r in rows if r.get("stage") == "battery"]
    out = {}
    for arm in ARMS:
        members = [r for r in battery if r.get("arm") == arm]
        pb = sum(primary(r.get("answer", "")) for r in members)
        changed = sum(r.get("answer", "").split()[:12] != baselines.get(int(r["question_index"]), "").split()[:12]
                      for r in members)
        at_cap = sum(int(r.get("token_count", -1)) == 80 for r in members)
        lens = [int(r["token_count"]) for r in members if isinstance(r.get("token_count"), (int, float))]
        slot = defaultdict(lambda: [0, 0])
        for r in members:
            s = slot[(int(r["layer"]), int(r["slot"]))]
            s[0] += int(primary(r.get("answer", "")))
            s[1] += 1
        slot_rates = {k: v[0] / v[1] for k, v in slot.items()}
        out[arm] = {"n": len(members), "primary_broken": pb, "changed": changed, "at_cap": at_cap,
                    "mean_len": sum(lens) / len(lens) if lens else None,
                    "rate": sum(slot_rates.values()) / len(slot_rates) if slot_rates else None,
                    "slots": len(slot_rates),
                    "full_k": max((int(r["k"]) for r in members if r.get("k")), default=None)}
    return out


def main():
    rows = load(f"{ROOT}/results/part210/answers.jsonl")
    rep = load(f"{ROOT}/results/part210/replicate.jsonl")
    pub_rows = load(f"{ROOT}/results/part165/answers.jsonl")
    base = {int(r["question_index"]): r.get("answer", "") for r in rows if r.get("stage") == "no_push"}
    pub_base = {int(r["question_index"]): r.get("answer", "") for r in pub_rows if r.get("stage") == "no_push"}
    print(f"rows={len(rows)} battery={sum(r.get('stage') == 'battery' for r in rows)} "
          f"no_push={sum(r.get('stage') == 'no_push' for r in rows)} replicate={len(rep)}")
    print(f"no_push broken (rerun): {[(qi, primary(base[qi])) for qi in sorted(base)]}")

    st = arm_stats(rows, base)
    print("\narm | n | primary_broken | slot-mean rate | at_cap | mean_len | changed")
    for arm in ARMS:
        s = st[arm]
        print(f"{arm:6s} | {s['n']} | {s['primary_broken']} | {s['rate']:.6f} | {s['at_cap']} | "
              f"{s['mean_len']:.4f} | {s['changed']}")

    r_nf = st["n+"]["rate"] / st["full"]["rate"]
    r_npm = st["n+"]["rate"] / st["n±"]["rate"]
    r_if = st["iso"]["rate"] / st["full"]["rate"]
    print(f"\nn+/full ratio = {r_nf:.6f}  (scorer 1.83898)")
    print(f"n+/n±  ratio = {r_npm:.6f}  (scorer 0.92735)")
    print(f"iso/full ratio = {r_if:.6f}  (scorer 0.855932)")

    full_k = st["full"]["full_k"]
    broken_rates = [st[a]["rate"] for a in LADDER]
    changed_rates = [st[a]["changed"] / st[a]["n"] for a in LADDER]
    logk = [LOGK[a] if a != "full" else math.log(full_k) for a in LADDER]
    print(f"full k = {full_k}")
    print(f"broken Spearman = {spearman(broken_rates, logk):.6f}  (scorer -0.857143)")
    print(f"changed Spearman = {spearman(changed_rates, logk):.6f}  (scorer 0.204124)")
    k16, full, npm = st["k16"]["rate"], st["full"]["rate"], st["n±"]["rate"]
    print(f"165.3 to_full={abs(k16 - full):.6f} to_n_pm={abs(k16 - npm):.6f} "
          f"diff={abs(k16 - npm) - abs(k16 - full):.6f} (scorer 0.03125)")

    # Verdicts from the recomputed values, by the registered rules.
    v = {"165.1": r_nf >= 2, "165.2": spearman(broken_rates, logk) <= -0.8,
         "165.3": abs(k16 - full) < abs(k16 - npm),
         "165.4": 0.67 <= r_npm <= 1.5, "165.5": r_if < 0.8,
         "165.6": spearman(changed_rates, logk) <= -0.6}
    print("verdicts (recomputed):", {k: ("pass" if x else "fail") for k, x in v.items()})

    # Replicate: how many of the 200 differ from their chunk-of-8 text.
    bat = {(r.get("stage"), r.get("arm"), r.get("slot"), r.get("layer"), int(r["question_index"])): r for r in rows}
    matched, differ = 0, 0
    for r in rep:
        key = ("battery", r.get("arm"), r.get("slot"), r.get("layer"), int(r["question_index"]))
        b = bat.get(key)
        if b is None:
            continue
        matched += 1
        differ += b.get("answer") != r.get("answer")
    print(f"\nreplicate matched={matched} differ={differ}  (scorer matched 162, differ 154)")

    # Published side, recomputed the same way, for the published-vs-rerun comparisons.
    pst = arm_stats(pub_rows, pub_base)
    print("\npublished arm | primary_broken | rate | at_cap")
    higher = []
    for arm in ARMS:
        print(f"{arm:6s} | {pst[arm]['primary_broken']} | {pst[arm]['rate']:.6f} | {pst[arm]['at_cap']}")
        if st[arm]["rate"] > pst[arm]["rate"]:
            higher.append(arm)
    print(f"published n+/full = {pst['n+']['rate'] / pst['full']['rate']:.6f} (score.json 2.69863)")
    print(f"arms with a higher rerun broken rate: {len(higher)} {higher}")

    # 210.4 and 210.7 by the registered rules.
    print(f"210.4 (each arm >=300 at cap): {all(st[a]['at_cap'] >= 300 for a in ARMS)}")
    print(f"210.7 (>=8 arms higher): {len(higher) >= 8}")
    print(f"210.3 (n+/full > 1.5): {r_nf > 1.5}")
    print(f"210.5 (n+/full < 2.69863): {r_nf < 2.69863}")
    print(f"210.6 (40 <= differ <= 110): {40 <= differ <= 110}")


if __name__ == "__main__":
    main()
