#!/usr/bin/env python3
"""Score all Part 134 registered gates, endpoints, task checks, and predictions."""
import argparse
import json
import re
from pathlib import Path

from select134 import endpoint

ROOT = Path(__file__).resolve().parents[1]
TASK_REGEX = {
    "What is the capital of France?": r"\bParis\b",
    "What is 7 times 8?": r"\b56\b",
    "Give me one tip for a job interview.": r"\b(interview|research|practice|prepare|questions|company|STAR|examples|listen|dress|arrive|follow[- ]up)\b",
    "Explain why the sky is blue in two sentences.": r"\b(scatter|scattering|shorter|wavelength|blue|atmosphere|sunlight|light)\b",
    "Write a two-line poem about the moon.": r"(?s)^\s*\S.*\n\S",
    "Name three primary colors.": r"(?i)\b(red|blue|yellow)\b.*\b(red|blue|yellow)\b.*\b(red|blue|yellow)\b",
    "How do I boil an egg?": r"(?i)\b(water|boil|minutes|egg|eggs|simmer|cook)\b",
    'Translate "good morning" into Spanish.': r"(?i)\b(buenos días|buen día)\b",
}
TASK_TRUTH = {
    "What is the capital of France?": lambda s: bool(re.search(TASK_REGEX["What is the capital of France?"],s,re.I)),
    "What is 7 times 8?": lambda s: bool(re.search(TASK_REGEX["What is 7 times 8?"],s)),
    "Give me one tip for a job interview.": lambda s: bool(re.search(TASK_REGEX["Give me one tip for a job interview."],s,re.I)),
    "Explain why the sky is blue in two sentences.": lambda s: bool(re.search(r"\b(scatter|shorter wavelengths|blue)\b",s,re.I)),
    "Write a two-line poem about the moon.": lambda s: len([x for x in s.splitlines() if x.strip()]) >= 2,
    "Name three primary colors.": lambda s: len(set(re.findall(r"\b(red|blue|yellow)\b",s,re.I))) == 3,
    "How do I boil an egg?": lambda s: bool(re.search(r"\b(water|boil)\b",s,re.I)) and bool(re.search(r"\b(minute|cook|simmer)\b",s,re.I)),
    'Translate "good morning" into Spanish.': lambda s: bool(re.search(TASK_REGEX['Translate "good morning" into Spanish.'],s,re.I)),
}


def read(path):
    if not path or not Path(path).is_file(): return []
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def valid_label(row):
    return (all(type(row.get(k)) is bool for k in ("shows_persona", "self_claim", "coherent")) and
            row.get("speaker") in ("ai", "human", "other", "none"))


def kappa(left, right):
    if not left: return None
    agree=sum(a==b for a,b in zip(left,right))/len(left)
    pa=sum(left)/len(left); pb=sum(right)/len(right); pe=pa*pb+(1-pa)*(1-pb)
    return 1.0 if pe==1 and agree==1 else ((agree-pe)/(1-pe) if pe<1 else 0.0)


def score(answers, ds_labels, luna_labels, gates=None):
    ans={(str(r.get("stage")),str(r.get("cell")),str(r.get("arm")),int(r["question_index"])):r for r in answers}
    def latest(rows):
        out={}
        for r in rows:
            if "error" not in r and valid_label(r): out[(str(r.get("stage")),str(r.get("cell")),str(r.get("arm")),int(r["question_index"]))]=r
        return out
    ds, luna_map=latest(ds_labels), latest(luna_labels)
    per={}; missing=0; task_rows=[]
    for key,a in ans.items():
        label=ds.get(key)
        if label is None or label.get("answer_hash")!=a.get("answer_hash"):
            missing+=1; continue
        e=endpoint(label); v=per.setdefault(key[:3],{"stage":key[0],"cell":key[1],"arm":key[2],"endpoint":0,"coherent":0,"valid_labels":0,"non_task_endpoint":0})
        v["valid_labels"]+=1; v["endpoint"]+=int(e); v["coherent"]+=int(bool(label["coherent"]))
        if a["question"] not in TASK_REGEX: v["non_task_endpoint"]+=int(e)
        if a["question"] in TASK_REGEX:
            task_rows.append({"cell":key[1],"arm":key[2],"question_index":key[3],"question":a["question"],
                              "correct":bool(TASK_TRUTH[a["question"]](a["answer"])),"answer":a["answer"]})
    selected=sorted({v["cell"] for v in per.values() if v["cell"]!="unpushed"})
    # Registered predictions are evaluated from the development ranking and held-out non-task/task counts.
    dev=[v for k,v in per.items() if v["stage"]=="dev" and v["arm"]=="pair" and v["cell"]!="unpushed" and v["valid_labels"]==6]
    devrank=sorted(dev,key=lambda r:(-r["endpoint"],-r["coherent"],sum(next(x["doses"] for x in answers if x["cell"]==r["cell"])),r["cell"]))
    test=[v for k,v in per.items() if v["stage"]=="test" and v["arm"]=="pair" and v["cell"]!="unpushed"]
    best=max(test,key=lambda x:(x["non_task_endpoint"],x["endpoint"]),default={})
    by={(v["stage"],v["cell"],v["arm"]):v for v in per.values()}
    same=[v for v in per.values() if v["stage"]=="test" and v["cell"]==best.get("cell")]
    pair=best.get("non_task_endpoint",0)
    singles=max([v["non_task_endpoint"] for v in same if v["arm"] in ("a_only","b_only")]+[0])
    randoms=[v["non_task_endpoint"] for v in same if "_random_" in v["arm"]]
    task_correct=sum(x["correct"] for x in task_rows if x["cell"]==best.get("cell") and x["arm"]=="pair")
    selected_doses=next((r["doses"] for r in answers if r.get("cell")== (devrank[0]["cell"] if devrank else "")),[0,0])
    predictions={"134.1":bool(devrank and devrank[0]["endpoint"]>=4),
        "134.2":bool(devrank and "p92-08" in next((r["pair"] for r in answers if r.get("cell")==devrank[0]["cell"]),[])),
        "134.3":bool(best and pair>=8),"134.4":bool(best and pair>=singles+3),
        "134.5":bool(best and len(randoms)>=4 and all(n<=pair/2 for n in randoms)),
        "134.6":bool(best and task_correct<=6),"134.7":bool(min(selected_doses)>=0.8)}
    positives=[]; paired=[]
    for key,label in ds.items():
        if key not in ans or label.get("answer_hash")!=ans[key].get("answer_hash"): continue
        luna=luna_map.get(key)
        if luna and luna.get("answer_hash")==ans[key].get("answer_hash"):
            positives.append(endpoint(label)); paired.append(endpoint(luna))
    result={"gates":gates or {},"predictions":predictions,"conditions":per,"task_rows":task_rows,
            "counts":{"answers":len(answers),"valid_ds_labels":len(ds),"missing_or_stale_labels":missing,"task_rows":len(task_rows)},
            "luna":{"paired_endpoint_n":len(positives),"kappa":kappa(positives,paired)},
            "development_rank":devrank,"best_test_pair":best}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--answers",default="results/part134/test-answers.jsonl")
    p.add_argument("--dev-answers",default="results/part134/dev-answers.jsonl"); p.add_argument("--ds",default="results/part134/test-labels-ds.jsonl")
    p.add_argument("--dev-ds",default="results/part134/dev-labels-ds.jsonl"); p.add_argument("--luna",default="results/part134/test-labels-luna.jsonl")
    p.add_argument("--out",default="results/part134"); a=p.parse_args(argv)
    combined_answers=read(a.dev_answers)+read(a.answers); combined_labels=read(a.dev_ds)+read(a.ds)
    luna_rows=read(a.luna)
    # Report every registered execution/reader gate from the saved evidence.
    root=Path(a.out)
    def gate_correct(path):
        rows=read(path)
        return sum("error" not in r and r.get("shows_persona")==r.get("expected") for r in rows)
    try: run_meta=json.loads((root/"dev-meta.json").read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): run_meta={}
    gate1=run_meta.get("gate1_zero_hook_same_answers") is True
    ds_known=gate_correct(root/"known-ds.jsonl"); luna_known=gate_correct(root/"known-luna.jsonl")
    gate2=ds_known>=11 and luna_known>=11
    result=score(combined_answers,combined_labels,luna_rows)
    ds_latest={(str(r.get("stage")),str(r.get("cell")),str(r.get("arm")),int(r["question_index"])):r for r in combined_labels if "error" not in r}
    luna_latest={(str(r.get("stage")),str(r.get("cell")),str(r.get("arm")),int(r["question_index"])):r for r in luna_rows if "error" not in r}
    positive_keys={k for k,r in ds_latest.items() if k[0]=="test" and k in {(str(x.get("stage")),str(x.get("cell")),str(x.get("arm")),int(x["question_index"])):x for x in combined_answers} and endpoint(r)}
    positive_covered=positive_keys <= set(luna_latest)
    nonpositive_luna=sum(k not in positive_keys for k in luna_latest)
    nonpositive_available=len({k for k in ds_latest if k[0]=="test"}-positive_keys)
    sample_ok=positive_covered and nonpositive_luna==min(len(positive_keys),nonpositive_available)
    gate3=sample_ok and result["luna"]["paired_endpoint_n"]>0 and result["luna"]["kappa"] is not None and result["luna"]["kappa"]>=0.60
    unpushed=[v for v in result["conditions"].values() if v["stage"]=="test" and v["cell"]=="unpushed"]
    gate4=(len(unpushed)==1 and unpushed[0]["valid_labels"]==24 and unpushed[0]["endpoint"]==0)
    result["gates"]={"1_zero_hook":gate1,"2_known_examples":{"passed":gate2,"ds_correct":ds_known,"luna_correct":luna_known,"required":11},
                     "3_luna_kappa":{"passed":gate3,"kappa":result["luna"]["kappa"],"n":result["luna"]["paired_endpoint_n"],"required":0.60,"positive_coverage":positive_covered,"nonpositive_sample_n":nonpositive_luna},
                     "4_unpushed_endpoint":{"passed":gate4,"endpoint_count":sum(v["endpoint"] for v in unpushed)}}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"score134.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Part 134 score", "", f"Answers: {result['counts']['answers']}; missing/stale DeepSeek labels: {result['counts']['missing_or_stale_labels']}.",
           f"Luna endpoint kappa: {result['luna']['kappa']} (n={result['luna']['paired_endpoint_n']}).", "", "## Predictions", ""]
    lines.extend(f"- {k}: {'yes' if v else 'no'}" for k,v in result["predictions"].items())
    lines += ["", "## Conditions", "", "| Cell | Arm | Endpoint | Non-task endpoint | Coherent | Valid labels |", "|---|---|---:|---:|---:|---:|"]
    lines.extend(f"| {v['cell']} | {v['arm']} | {v['endpoint']} | {v['non_task_endpoint']} | {v['coherent']} | {v['valid_labels']} |" for v in result["conditions"].values())
    lines += ["", "## Task questions", "", "| Cell | Arm | Question | Correct by fixed rule |", "|---|---|---|---|"]
    lines.extend(f"| {r['cell']} | {r['arm']} | {r['question']} | {r['correct']} |" for r in result["task_rows"])
    (out/"SCORE.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(result["predictions"],sort_keys=True)); return result


if __name__=="__main__": main()
