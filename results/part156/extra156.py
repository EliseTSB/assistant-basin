#!/usr/bin/env python3
"""Reproduce Part 156 exploratory tables and Part 155/156 comparisons.

Run from the repository root: python results/part156/extra156.py
All mention counts are answers containing Alibaba, Tongyi, or 通义 at least once,
case-insensitively, using the same exclusion terms as the registered company rule.
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from part86 import triage86

D156 = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
UNITS = {"L24N5017":"Google", "L21N7134":"Baidu", "L23N1743":"Huawei", "L23N2993":"Microsoft"}
def company_hit(answer, company):
    return (isinstance(answer, str) and company.casefold() in answer.casefold()
            and not MENTION.search(answer)
            and triage86.triage_answer(answer) != "broken")

P155_DOSES = (0.2, 0.4, 0.8, 1.2, 1.6)
UID_ORDER = ["L24N5017", "L21N7134", "L23N1743", "L23N2993"]
SHORT = {"L24N5017":"L24 N5017 (Google)", "L21N7134":"L21 N7134 (Baidu)",
         "L23N1743":"L23 N1743 (Huawei)", "L23N2993":"L23 N2993 (Microsoft)"}
MENTION = re.compile(r"alibaba|tongyi|通义", re.I)

def load(p):
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]

def tally(rows, doses):
    out = {u:{d:{"hits":0,"random":0,"broken":0,"mentions":0} for d in doses} for u in UID_ORDER}
    prompts = defaultdict(list)
    for r in rows:
        if r.get("kind") != "maker": continue
        u, d, arm = r["unit_id"], float(r["dose"]), r["arm"]
        if u not in out or d not in out[u] or arm not in ("neuron","random"): continue
        a = r["answer"]
        if arm == "neuron":
            out[u][d]["hits"] += int(company_hit(a, UNITS[u]))
            out[u][d]["broken"] += int(triage86.triage_answer(a) == "broken")
            out[u][d]["mentions"] += int(bool(MENTION.search(a)))
            prompts[(u,d)].append(r)
        else:
            out[u][d]["random"] += int(company_hit(a, UNITS[u]))
    return out, prompts

def main():
    r156, prompts156 = tally(load(ROOT/"results/part156/answers.jsonl"), D156)
    r155, _ = tally(load(ROOT/"results/part155/answers.jsonl"), P155_DOSES)
    rows156=load(ROOT/"results/part156/answers.jsonl")
    keys={(r["unit_id"],r["arm"],float(r["dose"]),int(r["question_index"])) for r in rows156}
    plain=[r for r in rows156 if r["arm"]=="plain"]
    plain_hits=sum(any(company_hit(r["answer"],c) for c in UNITS.values()) for r in plain)
    total08=sum(r156[u][0.8]["hits"] for u in UID_ORDER)
    print("## Independently recomputed gates and registered predictions")
    print(f"Gate 1: {len(rows156)} rows, {len(keys)} unique keys, {sum(bool(r['answer'].strip()) for r in rows156)} nonempty; passed={len(rows156)==1368 and len(keys)==1368 and all(r['answer'].strip() for r in rows156)}")
    print(f"Gate 2: +0.8 neuron total={total08} (registered 59–81); plain any-company hits={plain_hits} (limit 2); passed={59<=total08<=81 and plain_hits<=2}")
    other=tuple(d for d in D156 if d != 0.8)
    values={
        "156.1":(sum(any(r156[u][d]["hits"]>=12 for d in other) for u in UID_ORDER), ">=3 neurons"),
        "156.2":(sum(any(r156[u][d]["hits"]>r156[u][0.8]["hits"] for d in other) for u in UID_ORDER), ">=2 neurons"),
        "156.3":(max(r156[u][0.5]["hits"] for u in UID_ORDER), "<=3 each"),
    }
    contiguous=0
    for u in UID_ORDER:
        active=[i for i,d in enumerate(D156) if r156[u][d]["hits"]>=12]
        contiguous += int(bool(active) and active==list(range(active[0],active[-1]+1)) and len(active)<=4)
    values["156.4"]=(contiguous, ">=3 neurons")
    r5=[]
    for d in D156:
        n=sum(r156[u][d]["hits"] for u in UID_ORDER); rand=sum(r156[u][d]["random"] for u in UID_ORDER)
        if n>=24: r5.append((d,n,rand,rand<n/4))
    values["156.5"]=(r5, "random < one quarter where neuron total >=24")
    verdicts={"156.1":values["156.1"][0]>=3,"156.2":values["156.2"][0]>=2,"156.3":values["156.3"][0]<=3,"156.4":values["156.4"][0]>=3,"156.5":bool(r5) and all(x[3] for x in r5)}
    recorded=json.loads((ROOT/"results/part156/score.json").read_text())["predictions"]
    for k,(v,criterion) in values.items():
        verdict="held" if verdicts[k] else "failed"
        if k == "156.5":
            recorded_value=[(float(x["dose"]),x["neurons"],x["random"],x["ok"]) for x in recorded[k]["value"]]
            same_value=v==recorded_value
        else:
            same_value=v==recorded[k]["value"]
        print(f"{k}: {v}; criterion {criterion}; independently {verdict}; score.json {recorded[k]['verdict']} ({recorded[k]['value']}); agree={verdict==recorded[k]['verdict'] and same_value}")
    print("## Joined unit by dose (neuron hits / random hits / flagged broken)")
    print("Part 155 rows have 81 answers per phrasing; Part 156 rows have 57, so batch shape differs.")
    print("A change between runs may be batch shape. +0.8 side-by-side counts (155 -> 156):")
    for u in UID_ORDER:
        x,y=r155[u][0.8],r156[u][0.8]
        print(f"- {SHORT[u]}: {x['hits']}/{x['random']} ({x['broken']}) -> {y['hits']}/{y['random']} ({y['broken']})")
    print("\n| Unit | Batch | " + " | ".join(f"+{d:.1f}" for d in (*P155_DOSES,*D156)) + " |")
    print("|---|---:|" + "---:|"*(len(P155_DOSES)+len(D156)))
    for u in UID_ORDER:
        for batch, data, ds in (("155 (81)",r155,P155_DOSES),("156 (57)",r156,D156)):
            vals=[data[u][d] for d in ds]
            rowvals=[f"{v['hits']}/{v['random']}/{v['broken']}" for v in vals]
            if batch.startswith("155"):
                rowvals += ["—"] * len(D156)
            else:
                rowvals = ["—"] * len(P155_DOSES) + rowvals
            print(f"| {SHORT[u]} | {batch} | " + " | ".join(rowvals) + " |")
    print("\n## Alibaba/Tongyi/通义 mentions, neuron answers of 24")
    print("| Unit | " + " | ".join(f"+{d:.1f}" for d in D156) + " |")
    print("|---|"+"---:|"*len(D156))
    for u in UID_ORDER:
        print(f"| {SHORT[u]} | " + " | ".join(str(r156[u][d]['mentions']) for d in D156) + " |")
    # Cell-to-cell text comparison: only what these answers say.
    cells155=[x for x in load(ROOT/"results/part155/answers.jsonl") if x.get("kind")=="maker" and x["unit_id"]=="L24N5017" and x["arm"]=="neuron" and float(x["dose"])==0.4]
    cells156=[x for x in load(ROOT/"results/part156/answers.jsonl") if x["unit_id"]=="L24N5017" and x["arm"]=="neuron" and float(x["dose"])==0.5]
    print("\n## L24 N5017 boundary cells (company-rule hits)")
    print(f"Part 155 +0.4: {sum(company_hit(r['answer'], 'Google') for r in cells155)}/24; Part 156 +0.5: {sum(company_hit(r['answer'], 'Google') for r in cells156)}/24.")
    print("\n## Earliest dose and phrasing(s) for each neuron")
    for u in UID_ORDER:
        hits_by_q={}
        for d in D156:
            for r in prompts156[(u,d)]:
                if company_hit(r["answer"], UNITS[u]): hits_by_q.setdefault(int(r["question_index"]),d)
        earliest=min(hits_by_q.values()) if hits_by_q else None
        phrases=[r["question"] for r in prompts156[(u,earliest)]
                 if company_hit(r["answer"],UNITS[u])] if earliest is not None else []
        print(f"- {SHORT[u]}: first at +{earliest:.1f}; {len(phrases)}/{len(prompts156[(u,earliest)])} phrasings hit at that dose")
        for phrase in phrases: print(f"  - {phrase}")
    # Exploratory hand-read sample: the first eight prompt indexes in every neuron-by-dose cell.
    print("\n## Exploratory hand-read samples (first 8 prompt indexes in every cell)")
    for u in UID_ORDER:
        for d in D156:
            selected=sorted(prompts156[(u,d)],key=lambda r:int(r['question_index']))[:8]
            print(f"\n### {SHORT[u]} +{d:.1f} ({r156[u][d]['hits']}/24 hits; {len(selected)} sampled)")
            for r in selected:
                text=re.sub(r"\s+"," ",r['answer']).strip()
                print(f"- q{r['question_index']}: {r['question']} — {text}")
if __name__ == '__main__': main()
