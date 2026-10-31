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
    "What is 7 times 8?": r"\b(?:56|fifty[- ]six)\b",
    "Give me one tip for a job interview.": r"\b(interview|research|practice|prepare|questions|company|STAR|examples|listen|dress|arrive|follow[- ]up)\b",
    "Explain why the sky is blue in two sentences.": r"\b(?:scatter(?:ed|ing)?|shorter wavelengths?|blue)\b",
    "Write a two-line poem about the moon.": r"(?s)^\s*\S.*\n\S",
    "Name three primary colors.": r"(?i)\b(red|blue|yellow)\b.*\b(red|blue|yellow)\b.*\b(red|blue|yellow)\b",
    "How do I boil an egg?": r"(?i)\b(water|boil|minutes|egg|eggs|simmer|cook)\b",
    'Translate "good morning" into Spanish.': r"(?i)\b(buenos d[ií]as|buen d[ií]a)\b",
}
TASK_QUESTIONS = tuple(TASK_REGEX)
TASK_TRUTH = {
    "What is the capital of France?": lambda s: bool(re.search(r"^\s*(?:(?:the capital(?: of France)? is|France(?:'s|’s) capital is|the answer is|it is|it's)\s+)?Paris\b(?!\?)",s,re.I)),
    "What is 7 times 8?": lambda s: bool(re.search(TASK_REGEX["What is 7 times 8?"],s)),
    "Give me one tip for a job interview.": lambda s: bool(re.search(TASK_REGEX["Give me one tip for a job interview."],s,re.I)),
    "Explain why the sky is blue in two sentences.": lambda s: _sky_is_correct(s),
    "Write a two-line poem about the moon.": lambda s: len([x for x in s.splitlines() if x.strip()]) >= 2,
    "Name three primary colors.": lambda s: len(set(re.findall(r"\b(red|blue|yellow)\b",s,re.I))) == 3,
    "How do I boil an egg?": lambda s: bool(re.search(r"\b(?:egg|eggs)\b",s,re.I)) and bool(re.search(r"\b(?:boil|simmer|cook)\b",s,re.I)) and bool(re.search(r"\b(?:water|minutes?)\b",s,re.I)),
    'Translate "good morning" into Spanish.': lambda s: bool(re.search(TASK_REGEX['Translate "good morning" into Spanish.'],s,re.I)),
}


def _sky_is_correct(answer):
    text = str(answer)
    has_scattering = bool(re.search(r"\bscatter(?:ed|ing)?\b", text, re.I))
    has_light = bool(re.search(r"\b(?:sunlight|sun's light|light)\b", text, re.I))
    has_air = bool(re.search(r"\b(?:air|atmosphere|molecules)\b", text, re.I))
    has_blue_reason = bool(re.search(r"\b(?:blue|shorter wavelengths?)\b", text, re.I))
    return has_scattering and has_light and has_air and has_blue_reason


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
    keyfn=lambda r:(str(r.get("stage")),str(r.get("cell")),str(r.get("arm")),int(r["question_index"]))
    ans={keyfn(r):r for r in answers}
    def latest(rows):
        out={}
        for r in rows:
            out[keyfn(r)]=r
        return {key:r for key,r in out.items() if "error" not in r and valid_label(r)}
    ds, luna_map=latest(ds_labels), latest(luna_labels)
    per={}; missing=0; task_rows=[]
    for key,a in ans.items():
        condition_key="|".join(key[:3])
        v=per.setdefault(condition_key,{"stage":key[0],"cell":key[1],"arm":key[2],"endpoint":0,"coherent":0,
                                  "valid_labels":0,"non_task_endpoint":0,"answer_rows":0,"task_rows":0})
        v["answer_rows"]+=1
        label=ds.get(key)
        if label is None or label.get("answer_hash")!=a.get("answer_hash"):
            missing+=1; continue
        e=endpoint(label); v["valid_labels"]+=1; v["endpoint"]+=int(e); v["coherent"]+=int(bool(label["coherent"]))
        if a["question"] not in TASK_REGEX: v["non_task_endpoint"]+=int(e)
        if a["question"] in TASK_REGEX:
            v["task_rows"]+=1
            task_rows.append({"cell":key[1],"arm":key[2],"question_index":key[3],"question":a["question"],
                              "correct":bool(TASK_TRUTH[a["question"]](a["answer"])),"answer":a["answer"]})
    dev_answers=[r for r in answers if r.get("stage")=="dev"]
    dev_labels=[r for r in ds_labels if r.get("stage")=="dev"]
    try:
        from select134 import select as select_development
        devrank=select_development(dev_answers,dev_labels)["ranked"]
    except ValueError:
        devrank=[]
    test=[v for v in per.values() if v["stage"]=="test" and v["arm"]=="pair" and v["cell"]!="unpushed"]
    best=max(test,key=lambda x:(x["non_task_endpoint"],x["endpoint"]),default={})
    same=[v for v in per.values() if v["stage"]=="test" and v["cell"]==best.get("cell")]
    pair=best.get("non_task_endpoint",0)
    singles=max([v["non_task_endpoint"] for v in same if v["arm"] in ("a_only","b_only")]+[0])
    randoms=[v["non_task_endpoint"] for v in same if "_random_" in v["arm"]]
    task_rows_best=[x for x in task_rows if x["cell"]==best.get("cell") and x["arm"]=="pair"]
    task_correct=sum(x["correct"] for x in task_rows_best)
    selected_doses=next((r["doses"] for r in answers if r.get("cell")== (devrank[0]["cell"] if devrank else "")),[0,0])
    dev_rows=[r for r in answers if r.get("stage")=="dev"]
    test_rows=[r for r in answers if r.get("stage")=="test"]
    dev_keys={keyfn(r) for r in dev_rows}; test_keys={keyfn(r) for r in test_rows}
    dev_valid=sum(1 for k,r in ds.items() if k[0]=="dev" and k in ans and r.get("answer_hash")==ans[k].get("answer_hash"))
    test_valid=sum(1 for k,r in ds.items() if k[0]=="test" and k in ans and r.get("answer_hash")==ans[k].get("answer_hash"))
    dev_groups=[v for v in per.values() if v["stage"]=="dev"]
    test_groups=[v for v in per.values() if v["stage"]=="test"]
    dev_complete=(len(dev_rows)==450 and len(dev_keys)==450 and dev_valid==450 and len(dev_groups)==75 and
                  all(v["arm"]=="pair" and v["answer_rows"]==6 and v["valid_labels"]==6 for v in dev_groups))
    test_cells={v["cell"] for v in test_groups if v["cell"]!="unpushed"}
    expected_test_arms={"pair","a_only","b_only","a_random_1","a_random_2","b_random_1","b_random_2"}
    test_arm_pairs={(v["cell"],v["arm"]) for v in test_groups if v["cell"]!="unpushed"}
    expected_test_pairs={(cell,arm) for cell in test_cells for arm in expected_test_arms}
    unpushed_groups=[v for v in test_groups if v["cell"]=="unpushed"]
    test_complete=(len(test_rows)==528 and len(test_keys)==528 and test_valid==528 and len(test_cells)==3 and
                   test_arm_pairs==expected_test_pairs and all(v["answer_rows"]==24 and v["valid_labels"]==24 and v["task_rows"]==8 for v in test_groups) and
                   len(unpushed_groups)==1 and unpushed_groups[0]["arm"]=="none")
    ready=dev_complete and test_complete
    predictions={"134.1":bool(ready and devrank and devrank[0]["endpoint"]>=4),
        "134.2":bool(ready and devrank and "p92-08" in next((r["pair"] for r in answers if r.get("cell")==devrank[0]["cell"]),[])),
        "134.3":bool(ready and best and pair>=8),"134.4":bool(ready and best and pair>=singles+3),
        "134.5":bool(ready and best and len(randoms)==4 and all(n<=pair/2 for n in randoms)),
        "134.6":bool(ready and best and len(task_rows_best)==8 and task_correct<=6),
        "134.7":bool(ready and min(selected_doses)>=0.8)}
    positives=[]; paired=[]
    for key,label in ds.items():
        if key not in ans or label.get("answer_hash")!=ans[key].get("answer_hash"): continue
        luna=luna_map.get(key)
        if luna and luna.get("answer_hash")==ans[key].get("answer_hash"):
            positives.append(endpoint(label)); paired.append(endpoint(luna))
    return {"gates":gates or {},"predictions":predictions,"predictions_ready":ready,
            "row_coverage":{"development":{"expected":450,"answers":len(dev_rows),"unique":len(dev_keys),"valid_labels":dev_valid},
                            "test":{"expected":528,"answers":len(test_rows),"unique":len(test_keys),"valid_labels":test_valid}},
            "conditions":per,"task_rows":task_rows,
            "counts":{"answers":len(answers),"valid_ds_labels":len(ds),"missing_or_stale_labels":missing,"task_rows":len(task_rows)},
            "luna":{"paired_endpoint_n":len(positives),"kappa":kappa(positives,paired)},
            "development_rank":devrank,"best_test_pair":best}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--answers",default="results/part134/test-answers.jsonl")
    p.add_argument("--dev-answers",default="results/part134/dev-answers.jsonl"); p.add_argument("--ds",default="results/part134/test-labels-ds.jsonl")
    p.add_argument("--dev-ds",default="results/part134/dev-labels-ds.jsonl"); p.add_argument("--luna",default="results/part134/test-labels-luna.jsonl")
    p.add_argument("--out",default="results/part134"); a=p.parse_args(argv)
    combined_answers=read(a.dev_answers)+read(a.answers); combined_labels=read(a.dev_ds)+read(a.ds); luna_rows=read(a.luna); root=Path(a.out)
    def known_result(path):
        rows=read(path); good=[r for r in rows if "error" not in r and type(r.get("shows_persona")) is bool and r.get("shows_persona")==r.get("expected")]
        valid=[r for r in rows if "error" not in r and valid_label(r) and type(r.get("expected")) is bool]
        complete=(len(rows)==12 and len(valid)==12 and {r.get("known_index") for r in rows}==set(range(12)))
        return {"correct":len(good),"valid_rows":len(valid),"passed":complete and len(good)>=11}
    try: run_meta=json.loads((root/"dev-meta.json").read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): run_meta={}
    try: test_meta=json.loads((root/"test-meta.json").read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): test_meta={}
    ds_known=known_result(root/"known-ds.jsonl"); luna_known=known_result(root/"known-luna.jsonl")
    result=score(combined_answers,combined_labels,luna_rows)
    expected_dev_cells=set(run_meta.get("expected_cells",[]))
    expected_test_cells=set(test_meta.get("expected_cells",[]))
    actual_dev_cells={v["cell"] for v in result["conditions"].values() if v["stage"]=="dev"}
    actual_test_cells={v["cell"] for v in result["conditions"].values() if v["stage"]=="test" and v["cell"]!="unpushed"}
    meta_complete=(run_meta.get("expected_rows")==450 and len(expected_dev_cells)==75 and actual_dev_cells==expected_dev_cells and
                   test_meta.get("expected_rows")==528 and len(expected_test_cells)==3 and actual_test_cells==expected_test_cells)
    result["row_coverage"]["metadata_matches_registered_cells"]=meta_complete
    result["predictions_ready"]=result["predictions_ready"] and meta_complete
    ans_map={(str(x.get("stage")),str(x.get("cell")),str(x.get("arm")),int(x["question_index"])):x for x in combined_answers}
    def key(r): return (str(r.get("stage")),str(r.get("cell")),str(r.get("arm")),int(r["question_index"]))
    ds_latest={}; luna_latest={}
    for row in combined_labels: ds_latest[key(row)]=row
    for row in luna_rows: luna_latest[key(row)]=row
    ds_map={k:r for k,r in ds_latest.items() if "error" not in r and valid_label(r) and k in ans_map and r.get("answer_hash")==ans_map[k].get("answer_hash")}
    luna_map={k:r for k,r in luna_latest.items() if "error" not in r and valid_label(r) and k in ans_map and r.get("answer_hash")==ans_map[k].get("answer_hash")}
    positive_keys={k for k,r in ds_map.items() if k[0]=="test" and endpoint(r)}
    all_test_keys={k for k in ans_map if k[0]=="test"}
    luna_test_keys={k for k in luna_map if k[0]=="test"}
    # Reconstruct the deterministic sample in answer order, matching label134.SEED.
    import random
    from label134 import make_items
    shuffled_items=make_items([r for r in combined_answers if r.get("stage")=="test"])
    positive_order=[(str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) for i in shuffled_items
                    if (str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) in positive_keys]
    other_items=[(str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) for i in shuffled_items if (str(i["public"]["stage"]),str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"])) not in positive_keys]
    draw=random.Random(13401).sample(other_items,len(positive_order)) if len(other_items)>=len(positive_order) else []
    expected_luna_keys=set(positive_order) | set(draw)
    positive_coverage=positive_keys <= luna_test_keys
    sample_complete=(luna_test_keys==expected_luna_keys and len(positive_keys)==len(positive_order))
    kappa=result["luna"]["kappa"]
    gate3=sample_complete and result["luna"]["paired_endpoint_n"]==len(expected_luna_keys) and kappa is not None and kappa>=0.60
    unpushed=[v for v in result["conditions"].values() if v["stage"]=="test" and v["cell"]=="unpushed"]
    gate4=(len(unpushed)==1 and unpushed[0]["answer_rows"]==24 and unpushed[0]["valid_labels"]==24 and unpushed[0]["endpoint"]==0)
    gate1=run_meta.get("gate1_zero_hook_same_answers") is True and run_meta.get("expected_rows")==450 and len(expected_dev_cells)==75
    gate2=(ds_known["passed"] and luna_known["passed"])
    gates={"1_zero_hook":{"passed":gate1},"2_known_examples":{"passed":gate2,"ds_correct":ds_known["correct"],"luna_correct":luna_known["correct"],
                                                                          "ds_valid_rows":ds_known["valid_rows"],"luna_valid_rows":luna_known["valid_rows"],"required":11,"rows_each":12},
           "3_luna_kappa":{"passed":gate3,"kappa":kappa,"n":result["luna"]["paired_endpoint_n"],"required":0.60,"positive_coverage":positive_coverage,"expected_rows":len(expected_luna_keys),"actual_rows":len(luna_test_keys)},
           "4_unpushed_endpoint":{"passed":gate4,"endpoint_count":sum(v["endpoint"] for v in unpushed),"valid_rows":sum(v["valid_labels"] for v in unpushed)}}
    result["gates"]=gates
    gates_pass=all(v["passed"] for v in gates.values())
    if not gates_pass or not result["predictions_ready"]:
        result["predictions"]=None
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/"score134.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Part 134 score", "", f"Gate status: {'PASS' if gates_pass else 'FAIL'}.",
           f"Answers: {result['counts']['answers']}; missing/stale DeepSeek labels: {result['counts']['missing_or_stale_labels']}.",
           f"Luna endpoint kappa: {result['luna']['kappa']} (n={result['luna']['paired_endpoint_n']}).", ""]
    if result["predictions"] is not None:
        lines += ["## Predictions", ""]
        lines.extend(f"- {k}: {'yes' if v else 'no'}" for k,v in result["predictions"].items())
    else:
        lines += ["Predictions withheld: required gates or complete registered row coverage failed.", ""]
    lines += ["", "## Conditions", "", "| Cell | Arm | Endpoint | Non-task endpoint | Coherent | Valid labels | Answers |", "|---|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {v['cell']} | {v['arm']} | {v['endpoint']} | {v['non_task_endpoint']} | {v['coherent']} | {v['valid_labels']} | {v['answer_rows']} |" for v in result["conditions"].values())
    lines += ["", "## Task questions", "", "| Cell | Arm | Question | Correct by fixed rule |", "|---|---|---|---|"]
    lines.extend(f"| {r['cell']} | {r['arm']} | {r['question']} | {r['correct']} |" for r in result["task_rows"])
    (out/"SCORE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result["predictions"],sort_keys=True)); return result


if __name__=="__main__": main()
