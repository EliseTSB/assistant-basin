#!/usr/bin/env python3
"""Recompute supplemental Part 168 summaries from frozen result files."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "part86"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from part165 import score165
from part166 import score166

PHRASINGS = (8, 9, 10, 11, 12, 14, 15)
ARMS = ("n+", "k4")

def jl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def key(row):
    return (row["arm"], int(row["layer"]), int(row["slot"]), int(row["question_index"]))

def is_loop(row):
    return bool(score166.loop_phrase(row.get("answer", "")))

def is_broken(row):
    return bool(score165.primary_broken(row.get("answer", "")))

def rate(rows, fn):
    return sum(bool(fn(r)) for r in rows) / len(rows) if rows else None

def mean(rows, field):
    return sum(float(r.get(field, 0) or 0) for r in rows) / len(rows) if rows else None

def corr(xs, ys):
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    xx = sum((x-mx)**2 for x in xs); yy = sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(xx*yy) if xx and yy else None

def rel_cut(none, ban):
    a, b = rate(none, is_loop), rate(ban, is_loop)
    return 1 - b/a if a else None

def slot_stats(rows, predicate):
    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["layer"]), int(r["slot"]))].append(bool(predicate(r)))
    vals = [sum(v)/len(v) for v in cells.values()]
    if not vals:
        return {"slots": 0, "mean": None, "sd_population": None, "zero": 0, "at_least_half": 0, "by_slot": {}}
    return {"slots": len(vals), "mean": sum(vals)/len(vals),
            "sd_population": math.sqrt(sum((x-sum(vals)/len(vals))**2 for x in vals)/len(vals)),
            "zero": sum(x == 0 for x in vals), "at_least_half": sum(x >= .5 for x in vals),
            "share_from_at_least_half": sum(x for x in vals if x >= .5)/sum(vals) if sum(vals) else None,
            "by_slot": {f"{l}:{s}": sum(v)/len(v) for (l,s),v in sorted(cells.items())}}

def paired_gap(rows, amap):
    own=[]; mass=[]
    for m in rows:
        k=key(m)
        if not m.get("paired_set"): continue
        a=amap.get((m["arm"],"own",*k[1:])); b=amap.get((m["arm"],"mass",*k[1:]))
        if a and b: own.append(a); mass.append(b)
    return {"n": len(own), "own_rate": rate(own,is_loop), "mass_rate": rate(mass,is_loop),
            "gap_mass_minus_own": rate(mass,is_loop)-rate(own,is_loop) if own else None}

def whole_slot_gap_interval(rows167, rows168, reps=10000, seed=168):
    import numpy as np
    def table(rows):
        out=defaultdict(lambda: [0,0])
        for r in rows:
            c=out[(int(r["layer"]),int(r["slot"]))]; c[0]+=bool(is_broken(r)); c[1]+=1
        return out
    a,b=table(rows167),table(rows168); layers=sorted({l for l,s in a}|{l for l,s in b})
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(reps):
        sums=[]
        for tab in (a,b):
            n=d=0
            for layer in layers:
                slots=rng.integers(0,12,size=12)
                for slot in slots:
                    x,y=tab.get((layer,int(slot)),(0,0)); n+=x; d+=y
            sums.append(n/d if d else float("nan"))
        diffs.append(sums[0]-sums[1])
    arr=np.asarray(diffs); arr=arr[np.isfinite(arr)]
    return [float(x) for x in np.quantile(arr,[.025,.975])]

def main():
    import numpy as np
    score=json.loads((OUT/"score.json").read_text(encoding="utf-8"))
    answers=jl(OUT/"answers.jsonl"); masses=jl(OUT/"masses.jsonl"); pos=jl(OUT/"positions.jsonl")
    rows=jl(OUT/"rows.jsonl"); tokens=json.loads((OUT/"tokens.json").read_text(encoding="utf-8"))
    amap={(r["arm"],r["ban"],int(r["layer"]),int(r["slot"]),int(r["question_index"])):r
          for r in answers if r.get("stage")=="battery"}
    low=[m for m in masses if m["arm"] in ARMS]
    keys={key(m):m for m in low}
    # 168.3 single-drop relative cuts.
    own_frag={}
    base=[amap[(a,"none",l,s,q)] for a,l,s,q in keys]
    ban=[amap[(a,"own",l,s,q)] for a,l,s,q in keys]
    for dim, values in (("layer",sorted({k[1] for k in keys})),("phrasing",PHRASINGS)):
        for value in values:
            keep=[k for k in keys if k[1 if dim=="layer" else 3] != value]
            own_frag[f"without_{dim}_{value}"]=rel_cut([amap[(k[0],"none",*k[1:])] for k in keep], [amap[(k[0],"own",*k[1:])] for k in keep])
    for arm in ARMS:
        keep=[k for k in keys if k[0]==arm]
        own_frag[f"{arm}_alone"]=rel_cut([amap[(k[0],"none",*k[1:])] for k in keep],[amap[(k[0],"own",*k[1:])] for k in keep])
    # 168.8 paired gap fragility, recalculated with both answers on exact same rows.
    paired=[m for m in low if m.get("paired_set")]
    gap_frag={"all_paired":paired_gap(paired,amap)}
    for dim, values in (("layer",sorted({int(m["layer"]) for m in paired})),("phrasing",PHRASINGS)):
        for value in values:
            keep=[m for m in paired if int(m["layer"] if dim=="layer" else m["question_index"]) != value]
            gap_frag[f"without_{dim}_{value}"]=paired_gap(keep,amap)
    gap_frag["without_repaired"]=paired_gap([m for m in paired if not m.get("repaired")],amap)
    gap_frag["smallest_gap"]={k:v["gap_mass_minus_own"] for k,v in gap_frag.items() if isinstance(v,dict) and "gap_mass_minus_own" in v}
    gap_frag["minimum"]={"gap":min(gap_frag["smallest_gap"].values()),"case":min(gap_frag["smallest_gap"],key=gap_frag["smallest_gap"].get)}
    # Pooled low-k accounting.
    accounting={}
    for ban_name in ("own","matched","mass"):
        none_rows=[amap[(k[0],"none",*k[1:])] for k in keys]
        alt=[amap[(k[0],ban_name,*k[1:])] for k in keys]
        nloop=sum(is_loop(r) for r in none_rows); ended=sum(is_loop(n) and not is_loop(a) for n,a in zip(none_rows,alt))
        induced=sum(not is_loop(n) and is_loop(a) for n,a in zip(none_rows,alt))
        left=sum(is_loop(a) for a in alt)
        accounting[ban_name]={"none_loops":nloop,"ended":ended,"induced":induced,"net_removed":ended-induced,
                              "loops_left":left,"new_share_of_left":induced/left if left else None}
    own_induced_without_own_word=0
    for k in keys:
        none=amap[(k[0],"none",*k[1:])]; own=amap[(k[0],"own",*k[1:])]
        if is_loop(none) or not is_loop(own): continue
        phrase=" ".join(score166.loop_phrase(own.get("answer","")) or ())
        phrase_words={w.casefold() for w in score165.LOOP_WORDS.findall(phrase)}
        row=next((r for r in rows if (r["arm"],int(r["layer"]),int(r["slot"]),int(r["question_index"]))==k),{})
        own_words={w.casefold() for tid in row.get("own_ban_ids",[]) for w in score165.LOOP_WORDS.findall(tokens.get(str(int(tid)),""))}
        own_induced_without_own_word+=not bool(phrase_words & own_words)
    pooled={}
    for ban_name in ("none","own"):
        rr=[r for r in answers if r.get("stage")=="battery" and r["arm"] in ARMS and r["ban"]==ban_name]
        pooled[ban_name]={"n":len(rr),"mean_tokens":mean(rr,"token_count"),"eos_share":sum(bool(r.get("ended_eos_before_80")) for r in rr)/len(rr)}
    # Mass list calibration and plain-overlap split.
    exp=[float(m["exponent"]) for m in low if m.get("exponent") is not None]
    overlap_counts=[int(m["plain_overlap_mass"]) for m in low]
    by_overlap={}
    for label, keep in (("0",lambda x:x==0),("1",lambda x:x==1),("2",lambda x:x==2),("3",lambda x:x==3),("4+",lambda x:x>=4)):
        group=[m for m in low if keep(int(m["plain_overlap_mass"]))]; none=[amap[(m["arm"],"none",int(m["layer"]),int(m["slot"]),int(m["question_index"]))] for m in group]
        own=[amap[(m["arm"],"own",int(m["layer"]),int(m["slot"]),int(m["question_index"]))] for m in group]
        mass=[amap[(m["arm"],"mass",int(m["layer"]),int(m["slot"]),int(m["question_index"]))] for m in group]
        by_overlap[label]={"rows":len(group),"none_loop_rate":rate(none,is_loop),"own_loop_rate":rate(own,is_loop),
                           "mass_loop_rate":rate(mass,is_loop),"mass_induced":sum(not is_loop(n) and is_loop(a) for n,a in zip(none,mass)),
                           "none_no_loop_rows":sum(not is_loop(r) for r in none)}
    exp_over=[float(m["exponent"]) for m in low if m.get("exponent") is not None]
    plain_over=[int(m["plain_overlap_mass"]) for m in low if m.get("exponent") is not None]
    mass_info={"rows":len(low),"exponent_positive_share":sum(x>0 for x in exp)/len(exp),
               "median_exponent_by_arm":{a:float(np.median([m["exponent"] for m in low if m["arm"]==a and m.get("exponent") is not None])) for a in ARMS},
               "exponent_plain_overlap_correlation":corr(exp_over,plain_over),"overlap_bins":by_overlap}
    # Special token deviation: special IDs actually present in mass lists and answer records.
    special_names={"248069":"</think>","248066":"<tool_response>"}
    special_rows=[]
    for m in low:
        found=[(tid,special_names[str(tid)]) for tid in m.get("mass_ban_ids",[]) if str(tid) in special_names]
        if found:
            k=key(m); none=amap[(m["arm"],"none",*k[1:])]; mass=amap[(m["arm"],"mass",*k[1:])]
            special_rows.append({"arm":m["arm"],"layer":m["layer"],"slot":m["slot"],"question_index":m["question_index"],
                                 "tokens":[name for tid,name in found],"none_loop":is_loop(none),"mass_loop":is_loop(mass)})
    ids={int(t) for t in special_names}; pushed=[r for r in answers if r.get("stage")=="battery"]
    special_occurrences={name:sum(tid in ids and special_names[str(tid)]==name for r in pushed for tid in r.get("token_ids",[])) for name in special_names.values()}
    def drop_special(which):
        dropped={(r["arm"],int(r["layer"]),int(r["slot"]),int(r["question_index"])) for r in special_rows}
        if which=="168.W1":
            rr=[r for r in answers if r.get("stage")=="battery" and r["arm"] in ARMS and r["ban"] in ("none","mass") and (r["arm"],int(r["layer"]),int(r["slot"]),int(r["question_index"])) not in dropped]
            n=[r for r in rr if r["ban"]=="none"]; b=[r for r in rr if r["ban"]=="mass"]
            return {"n":len(n),"mass_relative_cut":1-rate(b,is_loop)/rate(n,is_loop)}
        keep=[m for m in paired if key(m) not in dropped]
        return paired_gap(keep,amap)
    # Cross-part n+ breakage and whole-slot summary.
    p165=[r for r in jl(ROOT/"results/part165/answers.jsonl") if r.get("stage")=="battery" and r.get("arm")=="n+" and r.get("ban") in (None,"none")]
    p167=[r for r in jl(ROOT/"results/part167/answers.jsonl") if r.get("stage")=="battery" and r.get("arm")=="n+" and r.get("ban")=="none" and int(r["question_index"])!=13]
    p168=[r for r in answers if r.get("stage")=="battery" and r.get("arm")=="n+" and r.get("ban")=="none"]
    seed_spread={"broken_rate":{"part165":rate(p165,is_broken),"part167_without_13":rate(p167,is_broken),"part168":rate(p168,is_broken)},
                 "slots":{"part165":slot_stats(p165,is_broken),"part167_without_13":slot_stats(p167,is_broken),"part168":slot_stats(p168,is_broken)},
                 "part167_minus_168_interval":whole_slot_gap_interval(p167,p168)}
    # Position means; largest position in the logged 0-38 range.
    pcount=max(len(r["own_by_position"]) for r in pos)
    pown=[sum(r["own_by_position"][i] for r in pos if len(r["own_by_position"])>i)/sum(len(r["own_by_position"])>i for r in pos) for i in range(pcount)]
    pmass=[sum(r["mass_by_position"][i] for r in pos if len(r["mass_by_position"])>i)/sum(len(r["mass_by_position"])>i for r in pos) for i in range(pcount)]
    positions={"rows":len(pos),"own_max_position":int(np.argmax(pown)),"own_max_mean":pown[int(np.argmax(pown))],
               "mass_max_position":int(np.argmax(pmass)),"mass_max_mean":pmass[int(np.argmax(pmass))],
               "own_at_mass_max":pown[int(np.argmax(pmass))],"mass_at_own_max":pmass[int(np.argmax(pown))],
               "top_on_own_mean":mean(pos,"top_on_own"),"top_on_mass_mean":mean(pos,"top_on_mass")}
    judgments={r["blind_id"]:r["judgement"] for r in jl(OUT/"hand_read_judged.jsonl")}
    hkey={r["blind_id"]:r for r in jl(OUT/"hand_read_key.jsonl")}; hand={}
    for bn in ("none","own","matched","mass"):
        ids=[i for i,r in hkey.items() if r["ban"]==bn]
        hand[bn]={n:sum(judgments[i]==n for i in ids) for n in ("readable","odd","broken")}
    p167all=[r for r in jl(ROOT/"results/part167/answers.jsonl") if r.get("stage")=="battery" and r.get("arm")=="n+" and r.get("ban")=="none"]
    qrates={}
    for qi in sorted({int(r["question_index"]) for r in p167all}):
        rr=[r for r in p167all if int(r["question_index"])==qi]
        qrates[str(qi)]={"broken":rate(rr,is_broken),"loop":rate(rr,is_loop)}
    plain=[r for r in jl(ROOT/"results/part164/answers.jsonl") if r.get("stage")=="no_push" and int(r.get("token_count",0))<=80]
    result={"fragility_168_3_relative_cut":own_frag,"fragility_168_8_paired_gap":gap_frag,
            "own_induced_without_own_word":own_induced_without_own_word,
            "low_k_loop_accounting":accounting,"pooled_low_k_length_eos":pooled,"mass_list":mass_info,
            "special_token_deviation":{"special_ids":special_names,"mass_rows":special_rows,"pushed_answer_token_occurrences":special_occurrences,
              "168_W1_without_special_rows":drop_special("168.W1"),"168_8_without_special_rows":drop_special("168.8")},
            "seed_spread":seed_spread,"positions":positions,"hand_read_my_judgement":{"n":len(judgments),"by_ban":hand},
            "part167_exploratory":{"nplus_none_by_question":qrates,"nplus_none_all":{"n":len(p167all),"broken_rate":rate(p167all,is_broken),"loop_rate":rate(p167all,is_loop)},
                                    "nplus_none_without_13":{"n":len(p167),"broken_rate":rate(p167,is_broken),"loop_rate":rate(p167,is_loop)}},
            "part164_plain_floor":{"n":len(plain),"loops":sum(bool(score166.loop_phrase(r.get("answer",""))) for r in plain)},
            "part168_score_headlines":{"predictions":score["predictions"],"bootstrap_intervals":score["bootstrap_intervals"],
               "per_arm_ban":score["per_arm_ban"],"mass_sensitivity":score["mass_sensitivity"],"transitions_by_ban":score["transitions_by_ban"],
               "induced_by_ban":score["induced_by_ban"],"mass_means_by_arm":score["mass_means_by_arm"],"gates":score.get("gates")}}
    (OUT/"extra168.txt").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("fragility_168_3_relative_cut","fragility_168_8_paired_gap","low_k_loop_accounting","pooled_low_k_length_eos","mass_list","special_token_deviation","seed_spread","positions")},indent=2,ensure_ascii=False,allow_nan=False))

if __name__=="__main__": main()
