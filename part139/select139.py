#!/usr/bin/env python3
"""Select Part 139 units from the frozen Part 92 stored screen."""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import census139
SCREEN=ROOT/"results/part92/screen-instruct"


def build(screen=SCREEN):
    by_layer={}
    for layer in (4,5):
        path=Path(screen)/f"layer_{layer}.jsonl"
        rows=[json.loads(line) for line in path.open(encoding="utf-8")]
        by_layer[layer]=rows
    chosen=[]
    for layer,rows in by_layer.items():
        neurons=[r for r in rows if r["unit"]=="neuron"]
        commonest=Counter(r["answer"] for r in neurons).most_common(1)[0][0]
        eligible=[r for r in neurons if r["answer"]==commonest]
        if len(eligible)<48: raise ValueError(f"L{layer}: only {len(eligible)} plain-answer matches for {commonest!r}")
        # The screen's commonest answer is the stored plain-model proxy for this layer.
        for r in neurons:
            if census139.OPENER.search(r["answer"]):
                chosen.append({"id":f"L{layer}-neuron-{r['index']}","set":"R","layer":layer,"unit":"neuron","index":int(r["index"]),"screen_answer":r["answer"]})
        apology=[x for x in chosen if x["set"]=="R" and x["layer"]==layer and x["unit"]=="neuron"]
        if len(apology)<48: raise ValueError(f"L{layer}: only {len(apology)} apology neurons; design requires 48")
        # Seeded draw from sorted screen indices supplies the registered sample.
        sampled=random.Random(139).sample(sorted(apology,key=lambda x:x["index"]),48)
        keep={x["id"] for x in sampled}
        chosen=[x for x in chosen if not (x["layer"]==layer and x["unit"]=="neuron" and x["set"]=="R" and x["id"] not in keep)]
        # Persist R in index order for stable downstream batching.
        chosen.sort(key=lambda x:(x["layer"],x["set"],x["unit"],x["index"]))
        for r in rows:
            if r["unit"]=="random" and census139.OPENER.search(r["answer"]):
                chosen.append({"id":f"L{layer}-random-{r['index']}","set":"R","layer":layer,"unit":"random","index":int(r["index"]),"screen_answer":r["answer"]})
        for r in eligible[:48]:
            chosen.append({"id":f"L{layer}-neuron-{r['index']}","set":"S","layer":layer,"unit":"neuron","index":int(r["index"]),"screen_answer":r["answer"],"plain_proxy":commonest})
    # Stage C uses the first 24 neuron entries from each fixed R/S layer set.
    c_subset=[]
    for layer in (4,5):
        for set_name in ("R","S"):
            candidates=sorted((x for x in chosen if x["layer"]==layer and x["set"]==set_name and x["unit"]=="neuron"),key=lambda x:x["index"])
            subset=random.Random(139).sample(candidates,24)
            c_subset.extend(subset)
    stage_f=[]
    for layer,rows in by_layer.items():
        selected_r={x["index"] for x in chosen if x["set"]=="R" and x["layer"]==layer and x["unit"]=="neuron"}
        candidates=sorted(int(r["index"]) for r in rows if r["unit"]=="neuron" and int(r["index"]) not in selected_r)
        if len(candidates)<64: raise ValueError(f"L{layer}: fewer than 64 unselected neurons")
        for index in random.Random(139).sample(candidates,64): stage_f.append({"id":f"F-L{layer}-neuron-{index}","set":"F","layer":layer,"unit":"neuron","index":index})
        for index in range(64): stage_f.append({"id":f"F-L{layer}-random-{index}","set":"F","layer":layer,"unit":"random","index":index})
    result={"seed":139,"screen":str(Path(screen)),"set_s_plain_source":"screen commonest neuron answer per layer (do-nothing baseline was not stored)","sets":{"R":[x for x in chosen if x["set"]=="R"],"S":[x for x in chosen if x["set"]=="S"]},"stage_c":c_subset,"stage_f":{"seed_neurons":139,"random_seed_by_layer":{"4":139104,"5":139105},"units":stage_f}}
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--screen",default=str(SCREEN)); ap.add_argument("--out",default=str(HERE/"units139.json")); args=ap.parse_args()
    data=build(args.screen); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for name,rows in [("R",data["sets"]["R"]),("S",data["sets"]["S"]),("Stage C",data["stage_c"]),("Stage F",data["stage_f"]["units"])]:
        print(f"{name}: {len(rows)} units; "+", ".join(f"L{l}={sum(x['layer']==l for x in rows)}" for l in (4,5)))
if __name__=="__main__": main()
