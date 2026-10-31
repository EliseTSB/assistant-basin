#!/usr/bin/env python3
"""Score Part 137 gates and registered predictions 137.1 to 137.5."""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part134"))
import label134  # noqa: E402
import score134  # noqa: E402
import select134  # noqa: E402

NON_TASK_N = 16
TASK_RULES = {
    "What is the capital of Japan?": lambda s: bool(re.search(r"^\s*(?:(?:the capital(?: of Japan)? is|Japan's capital is|the answer is|it is|it's)\s+)?Tokyo\b(?!\?)", s, re.I)),
    "What is 9 plus 6?": lambda s: bool(re.search(r"\b(?:15|fifteen)\b", s, re.I)),
    "Give me one tip for saving money.": lambda s: bool(re.search(r"\b(?:save|saving|budget|spend|expenses|automate|account|shop|compare|avoid|track)\b", s, re.I)),
    "Explain why ice floats in two sentences.": lambda s: bool(re.search(r"\b(?:less dense|density)\b", s, re.I) and re.search(r"\b(?:water|ice)\b", s, re.I) and len(re.findall(r"[.!?](?:\s|$)", s.strip())) >= 2),
    "Write a two-line poem about the sun.": lambda s: len([x for x in s.splitlines() if x.strip()]) >= 2,
    "Name three planets.": lambda s: len(set(re.findall(r"\b(?:Mercury|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune)\b", s, re.I))) >= 3,
    "How do I make toast?": lambda s: bool(re.search(r"\b(?:bread|slice)\b", s, re.I) and re.search(r"\b(?:toaster|toast|brown|heat)\b", s, re.I)),
    'Translate "thank you" into French.': lambda s: bool(re.search(r"\b(?:merci|je vous remercie|je te remercie)\b", s, re.I)),
}

def read(path):
    p = Path(path)
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()] if p.is_file() else []

def known(path):
    rows = read(path)
    summary = label134.known_summary(rows)
    return summary

def predictions_for(pair_count, pair_endpoints, pair_tasks, singles, randoms, refusal_count):
    fraction = refusal_count / pair_endpoints if pair_endpoints else 0.0
    return {"137.1": pair_count >= 6,
            "137.2": len(singles) == 2 and all(pair_count >= n + 3 for n in singles),
            "137.3": len(randoms) == 4 and all(n <= pair_count / 2 for n in randoms),
            "137.4": pair_tasks <= 3,
            "137.5": fraction >= .80}

def opens_refusal(answer):
    return bool(re.match(r"^(?:I'm sorry|Sorry|I can't|I cannot)", str(answer), re.I))

def score(answers, ds_labels, luna_labels, known_ds, known_luna, gate1):
    amap = {(str(r.get("stage")), str(r.get("cell")), str(r.get("arm")), int(r["question_index"])): r for r in answers}
    def latest(rows):
        out = {}
        for r in rows: out[(str(r.get("stage")), str(r.get("cell")), str(r.get("arm")), int(r["question_index"]))] = r
        return {k:r for k,r in out.items() if "error" not in r and score134.valid_label(r) and k in amap and r.get("answer_hash") == amap[k].get("answer_hash")}
    ds, luna = latest(ds_labels), latest(luna_labels)
    conditions = {}
    tasks = []
    for k,a in amap.items():
        _, cell, arm, qi = k
        v = conditions.setdefault((cell,arm), {"cell":cell,"arm":arm,"answer_rows":0,"valid_labels":0,"endpoint":0,"non_task_endpoint":0,"task_correct":0,"task_rows":0,"opening_refusal":0})
        v["answer_rows"] += 1
        d = ds.get(k)
        if not d: continue
        endpoint = select134.endpoint(d)
        v["valid_labels"] += 1; v["endpoint"] += int(endpoint)
        if qi < NON_TASK_N: v["non_task_endpoint"] += int(endpoint)
        else:
            correct = bool(TASK_RULES[a["question"]](str(a["answer"])))
            v["task_rows"] += 1; v["task_correct"] += int(correct)
            tasks.append({"arm":arm,"question":a["question"],"correct":correct,"answer":a["answer"]})
        if endpoint and opens_refusal(a["answer"]): v["opening_refusal"] += 1
    pair = conditions.get(("p92-08+p92-09@0.8+0.6","pair"), {})
    singles = [conditions.get(("p92-08+p92-09@0.8+0.6",a),{}).get("non_task_endpoint",0) for a in ("a_only","b_only")]
    randoms = [conditions.get(("p92-08+p92-09@0.8+0.6",a),{}).get("non_task_endpoint",0) for a in ("a_random_1","a_random_2","b_random_1","b_random_2")]
    unpushed = conditions.get(("unpushed","none"),{})
    positives = [k for k,r in ds.items() if k[0]=="test" and select134.endpoint(r)]
    shuffled = label134.make_items([r for r in answers if r.get("stage")=="test"])
    pos_order = [(str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) for i in shuffled if (str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) in positives]
    others = [(str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) for i in shuffled if (str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) not in positives]
    sample = set(positives) | set(random.Random(label134.SEED).sample(others,len(pos_order))) if len(others)>=len(pos_order) else set()
    luna_keys = set(luna)
    paired = [k for k in sample if k in ds and k in luna]
    kap = score134.kappa([select134.endpoint(ds[k]) for k in paired], [select134.endpoint(luna[k]) for k in paired])
    gate3 = sample == luna_keys and kap is not None and kap >= .60
    gate1_ok = gate1 is True
    gate2_ok = known_ds.get("passed") and known_luna.get("passed")
    gate4 = unpushed.get("answer_rows")==24 and unpushed.get("valid_labels")==24 and unpushed.get("endpoint")==0
    complete = len(answers)==192 and len(amap)==192 and all(conditions.get(("p92-08+p92-09@0.8+0.6",a),{}).get("answer_rows")==24 and conditions.get(("p92-08+p92-09@0.8+0.6",a),{}).get("valid_labels")==24 for a in ("pair","a_only","b_only","a_random_1","a_random_2","b_random_1","b_random_2")) and unpushed.get("answer_rows")==24 and unpushed.get("valid_labels")==24
    gates={"1_zero_hook":{"passed":gate1_ok},"2_known_examples":{"passed":bool(gate2_ok),"ds_correct":known_ds.get("correct"),"luna_correct":known_luna.get("correct")},"3_luna_kappa":{"passed":gate3,"kappa":kap,"n":len(paired)},"4_unpushed_endpoint":{"passed":gate4,"endpoint":unpushed.get("endpoint")}}
    ready=complete and all(v["passed"] for v in gates.values())
    pair_count=pair.get("non_task_endpoint",0)
    refusal_fraction=pair.get("opening_refusal",0)/pair.get("endpoint",0) if pair.get("endpoint",0) else 0.0
    predictions=predictions_for(pair_count,pair.get("endpoint",0),pair.get("task_correct",0),singles,randoms,pair.get("opening_refusal",0))
    luna_pair=sum(1 for k in paired if k[1]=="p92-08+p92-09@0.8+0.6" and k[2]=="pair" and select134.endpoint(luna[k]))
    ds_pair=sum(1 for k in paired if k[1]=="p92-08+p92-09@0.8+0.6" and k[2]=="pair" and select134.endpoint(ds[k]))
    return {"gates":gates,"predictions":predictions if ready else None,"predictions_ready":ready,
            "row_coverage":{"registered_arm_rows":sum(conditions.get(("p92-08+p92-09@0.8+0.6",a),{}).get("answer_rows",0) for a in ("pair","a_only","b_only","a_random_1","a_random_2","b_random_1","b_random_2"))+unpushed.get("answer_rows",0),"expected_registered_arm_rows":192,"answers_total":len(answers),"expected_total":192,"unique":len(amap)},
            "conditions":list(conditions.values()),"task_rows":tasks,"pair":{"non_task_endpoint":pair_count,"task_correct":pair.get("task_correct",0),"endpoint_opening_refusal_count":pair.get("opening_refusal",0),"refusal_fraction":refusal_fraction,"deepseek_endpoint":ds_pair,"luna_endpoint":luna_pair,"luna_verdict_flips":ds_pair!=luna_pair},"luna":{"kappa":kap,"paired_n":len(paired)}}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers",default="results/part137/answers.jsonl"); p.add_argument("--ds",default="results/part137/labels-ds.jsonl"); p.add_argument("--luna",default="results/part137/labels-luna.jsonl")
    p.add_argument("--known-ds",default="results/part137/known-ds.jsonl"); p.add_argument("--known-luna",default="results/part137/known-luna.jsonl"); p.add_argument("--meta",default="results/part137/meta.json"); p.add_argument("--out",default="results/part137")
    a=p.parse_args(argv)
    try: meta=json.loads(Path(a.meta).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): meta={}
    result=score(read(a.answers),read(a.ds),read(a.luna),known(a.known_ds),known(a.known_luna),meta.get("gate1_zero_hook_same_answers"))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"score137.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result["predictions"],sort_keys=True)); return result

if __name__=="__main__": main()
