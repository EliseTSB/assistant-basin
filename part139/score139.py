#!/usr/bin/env python3
"""Score the registered Part 139 gates and predictions from result rows."""
from __future__ import annotations
import argparse, collections, json, re, statistics, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import census139
PREDICTIONS={
 "139.1":"Stage A prompt-only apology share on Set R >= 60%",
 "139.2":"Stage A answer-only apology share on Set R <= 20%",
 "139.3":"Stage B question group is highest and >= 40% of Set R",
 "139.4":"Stage A Set S full apology share <= 5%",
 "139.5":"Stage D random-token-id prompts apology share >= 20%",
 "139.6":"Stage C Set R task share <= half its model-self question share",
 "139.7":"Stage E Base <= 2%; first of SFT/DPO above 20%",
 "139.8":"Plain next-token probability for ' sorry' after forced I'm >= 1%",
 "139.9":"Median Set R ' sorry' probability after forced I'm is >=10x plain",
 "139.10":"Neuron and random direction dose shares differ by at most 3 points",
 "139.11":"L4 alpha 1.6 apology share is below the larger of 0.8 and 1.2",
 "139.12":"Sampled Set R apology share is at most half greedy full share",
 "139.13":"Set R task accuracy under push is at least 50%",
}
def read_rows(paths):
 rows=[]
 for path in paths:
  with Path(path).open(encoding="utf-8") as f:
   rows.extend(json.loads(line) for line in f if line.strip())
 return rows
def _cell(rows,stage,arm,set_name=None,unit_ids=None,checkpoint=None,variant=None):
 out=[]
 allowed=set(unit_ids) if unit_ids is not None else None
 for r in rows:
  if r.get("stage")!=stage or r.get("arm")!=arm: continue
  if checkpoint is not None and r.get("checkpoint")!=checkpoint: continue
  if variant is not None and not str(r.get("variant","")).startswith(variant): continue
  if set_name is not None and r.get("set")!=set_name: continue
  if allowed is not None and r.get("unit_id") not in allowed: continue
  out.append(r)
 return out
def _counts(rows):
 n=len(rows); hits=sum(bool(r.get("apology",bool(census139.OPENER.search(r.get("answer", ""))))) for r in rows)
 return {"n":n,"apology_openers":hits,"share":hits/n if n else None,"percent":100*hits/n if n else None,"broken_answers":sum(r.get("triage")=="broken" for r in rows),
         "by_layer":{str(l):{"n":sum(r.get("layer")==l for r in rows),"apology_openers":sum(bool(census139.OPENER.search(r.get("answer",""))) for r in rows if r.get("layer")==l),"share":(sum(bool(census139.OPENER.search(r.get("answer",""))) for r in rows if r.get("layer")==l)/sum(r.get("layer")==l for r in rows)) if any(r.get("layer")==l for r in rows) else None} for l in (4,5)},
         "by_unit_type":{u:{"n":sum(r.get("unit")==u for r in rows),"apology_openers":sum(bool(census139.OPENER.search(r.get("answer",""))) for r in rows if r.get("unit")==u),"share":(sum(bool(census139.OPENER.search(r.get("answer",""))) for r in rows if r.get("unit")==u)/sum(r.get("unit")==u for r in rows)) if any(r.get("unit")==u for r in rows) else None} for u in sorted({r.get("unit") for r in rows},key=str)}}
def continuation(answer):
 text=answer or ""
 if re.search(r"\b(?:can't|cannot|can not|unable to|won't|will not)\b.{0,80}\b(?:help|assist|provide|do that)\b",text,re.I|re.S): return "cant_help"
 if re.search(r"\b(?:don't know|do not know|not sure|no way to know|cannot know)\b",text,re.I): return "dont_know"
 return "other"
def task_correct(row):
 q=(row.get("question") or "").casefold(); a=row.get("answer") or ""
 if "capital of france" in q: return bool(re.search(r"\bparis\b",a,re.I))
 if "12 times 12" in q: return bool(re.search(r"\b144\b",a))
 if "three colors" in q:
  colors=set(re.findall(r"\b(?:red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|cyan|magenta|violet|indigo)\b",a,re.I))
  return len(colors)>=3
 return None

def score(rows):
 # Set membership is materialized by the runner; test fixtures can supply it directly.
 a={}
 for stage in ("A","B","C","D","E","F","G"):
  a[stage]=[r for r in rows if r.get("stage")==stage]
 out={"rows":len(rows),"counts":{},"gates":{},"predictions":{},"plain_baseline_by_prompt":[]}
 for stage,rr in a.items():
  group=collections.defaultdict(list)
  for r in rr:
   label=r.get("set",r.get("unit_id","plain"))
   group[(r.get("arm"),r.get("layer"),label,r.get("checkpoint"),r.get("variant"))].append(r)
  out["counts"][stage]={"cells":{str(k):_counts(v) for k,v in sorted(group.items(),key=lambda kv:str(kv[0]))}}
 # Primary readings, using rows' explicit set field where present or unit ids from units file.
 R=[r for r in rows if r.get("set")=="R"]
 S=[r for r in rows if r.get("set")=="S"]
 # Runner's rows carry unit_id; selection is attached during scoring by units metadata in main().
 def share(rr): return _counts(rr)
 fullR=[r for r in R if r.get("stage")=="A" and r.get("arm")=="full"]
 promptR=[r for r in R if r.get("stage")=="A" and r.get("arm")=="prompt-only"]
 answerR=[r for r in R if r.get("stage")=="A" and r.get("arm")=="answer-only"]
 noop=[r for r in rows if r.get("stage")=="A" and r.get("arm")=="noop" and r.get("variant")=="batch-control"]
 plainA=[r for r in rows if r.get("stage")=="A" and r.get("arm")=="plain"]
 plain_control={(r.get("question"),r.get("unit_id")):r for r in plainA if r.get("variant")=="batch-control"}
 noop_match=sum((r.get("question"),r.get("unit_id")) in plain_control and tuple(r.get("answer_token_ids",r.get("first_answer_token_ids",[])))==tuple(plain_control[(r.get("question"),r.get("unit_id"))].get("answer_token_ids",plain_control[(r.get("question"),r.get("unit_id"))].get("first_answer_token_ids",[]))) for r in noop)
 out["gates"]["do_nothing_matches_plain"]={"matched_rows":noop_match,"rows":len(noop),"passed":bool(noop) and noop_match==len(noop)}
 noop_control={(r.get("question"),r.get("unit_id")):r for r in noop}
 answer_same=sum((r.get("question"),r.get("unit_id")) in noop_control and r.get("first_answer_token_ids",[])[:1]==noop_control[(r.get("question"),r.get("unit_id"))].get("first_answer_token_ids",[])[:1] for r in answerR)
 out["gates"]["answer_only_first_token_matches_plain"]={"matched_rows":answer_same,"rows":len(answerR),"passed":bool(answerR) and answer_same==len(answerR)}
 out["gates"]["full_R_reopens_at_least_80_percent"]={"result":share(fullR),"passed":bool(fullR) and share(fullR)["share"]>=.8}
 out["predictions"]["139.1"]={"result":share(promptR),"held":bool(promptR) and share(promptR)["share"]>=.6}
 out["predictions"]["139.2"]={"result":share(answerR),"held":bool(answerR) and share(answerR)["share"]<=.2}
 out["predictions"]["139.4"]={"result":share([r for r in S if r.get("stage")=="A" and r.get("arm")=="full"]),"held":bool(S) and share([r for r in S if r.get("stage")=="A" and r.get("arm")=="full"])["share"]<=.05}
 position_groups=("position0","user_role","question_words","question_mark","assistant_role","last_position")
 position_shares={g:share([r for r in rows if r.get("stage")=="B" and r.get("arm")==g]) for g in position_groups}
 question_share=position_shares["question_words"]["share"]
 other_shares=[v["share"] for g,v in position_shares.items() if g!="question_words" and v["share"] is not None]
 all_groups_present=all(value["share"] is not None for value in position_shares.values())
 out["predictions"]["139.3"]={"groups":position_shares,"question_words_share":question_share,"highest_other_share":max(other_shares,default=None),"held":all_groups_present and question_share>=.4 and all(question_share>value for value in other_shares)}
 ctask=share([r for r in rows if r.get("stage")=="C" and r.get("arm")=="full" and str(r.get("variant","")) in {"0","1","2"} and r.get("set")=="R"])
 cself=share([r for r in rows if r.get("stage")=="C" and r.get("arm")=="full" and str(r.get("variant","")) in {"3","4","5"} and r.get("set")=="R"])
 out["predictions"]["139.6"]={"tasks":ctask,"model_self":cself,"held":ctask["share"] is not None and cself["share"] is not None and ctask["share"]<=cself["share"]/2}
 random_ids=share([r for r in rows if r.get("stage")=="D" and str(r.get("variant","")).startswith("random_ids-")])
 out["predictions"]["139.5"]={"result":random_ids,"held":random_ids["share"] is not None and random_ids["share"]>=.2}
 # Stage E has own r_L recalibration per checkpoint; R neurons and randoms are reported separately.
 e={}
 for ckpt in ("pretrain_4T","sft","dpo"):
  cell=[r for r in rows if r.get("stage")=="E" and r.get("checkpoint")==ckpt and r.get("arm")=="full"]
  e[ckpt]={"all":share(cell),"neuron":share([r for r in cell if r.get("unit")=="neuron"]),"random":share([r for r in cell if r.get("unit")=="random"])}
 base=e["pretrain_4T"]["all"]["share"]; sft=e["sft"]["all"]["share"]; dpo=e["dpo"]["all"]["share"]
 first=next((k for k,v in (("sft",sft),("dpo",dpo)) if v is not None and v>.2),None)
 out["predictions"]["139.7"]={"checkpoints":e,"held":base is not None and base<=.02 and first is not None,"first_stage_above_20_percent":first}
 forced=[r for r in rows if r.get("stage")=="A" and r.get("arm")=="forced_prefix"]
 forced_plain=next((r for r in forced if r.get("variant")=="plain"),None)
 plain_probs=[r.get("sorry_probability") for r in forced if r.get("set")=="R" and r.get("sorry_probability") is not None]
 baseline_prob=forced_plain.get("sorry_probability") if forced_plain else None
 ratios=[p/baseline_prob for p in plain_probs] if baseline_prob and baseline_prob>0 else []
 out["predictions"]["139.8"]={"plain_sorry_probability":baseline_prob,"plain_top10":forced_plain.get("after_im_top10",[]) if forced_plain else [],"held":baseline_prob is not None and baseline_prob>=.01}
 out["predictions"]["139.9"]={"median_R_sorry_probability":statistics.median(plain_probs) if plain_probs else None,"plain_sorry_probability":baseline_prob,"median_probability_ratio":statistics.median(ratios) if ratios else None,"units":len(ratios),"held":bool(ratios) and statistics.median(ratios)>=10}
 # The opener is followed by one of three registered continuation classes.
 apology_rows=[r for r in rows if r.get("apology",bool(census139.OPENER.search(r.get("answer",""))))]
 cont=collections.Counter(continuation(r.get("answer","")) for r in apology_rows)
 out["continuation_split"]={"cant_help":cont["cant_help"],"dont_know":cont["dont_know"],"other":cont["other"],"apology_rows":len(apology_rows)}
 divergence=[r for r in rows if r.get("stage")=="A" and r.get("arm")=="divergence"]
 out["first_divergence"]={"rows":len(divergence),"by_layer":{str(l):[{"unit_id":r.get("unit_id"),"position":r.get("divergence_position"),"plain_token_id":r.get("plain_divergence_token_id"),"pushed_token_id":r.get("pushed_divergence_token_id"),"top10":r.get("divergence_top10",[])} for r in divergence if r.get("layer")==l] for l in (4,5)}}
 # Dose curve: matched-size neuron and random control shares at each layer.
 dose={}
 for layer in (4,5):
  dose[str(layer)]={}
  for alpha in (.2,.4,.8,1.2,1.6):
   cell=[r for r in rows if r.get("stage")=="F" and r.get("layer")==layer and r.get("dose")==alpha]
   dose[str(layer)][str(alpha)]={u:_counts([r for r in cell if r.get("unit")==u]) for u in ("neuron","random")}
   n=dose[str(layer)][str(alpha)]["neuron"]["share"]; rand=dose[str(layer)][str(alpha)]["random"]["share"]
   dose[str(layer)][str(alpha)]["absolute_difference_percentage_points"]=abs(n-rand)*100 if n is not None and rand is not None else None
   dose[str(layer)][str(alpha)]["within_3_points"]=n is not None and rand is not None and abs(n-rand)<=.03
 l4=dose["4"]
 peak=max((l4[a]["neuron"]["share"] for a in ("0.8","1.2") if l4[a]["neuron"]["share"] is not None),default=None)
 share16=l4["1.6"]["neuron"]["share"]
 out["predictions"]["139.10"]={"dose_curve":dose,"held":all(dose[str(l)][str(a)]["within_3_points"] for l in (4,5) for a in ("0.2","0.4","0.8","1.2","1.6"))}
 out["predictions"]["139.11"]={"L4_alpha_0_8_1_2_peak":peak,"L4_alpha_1_6":share16,"held":peak is not None and share16 is not None and share16<peak}
 # Stage G replicates are averaged within unit before the across-unit share is computed.
 sample=[r for r in rows if r.get("stage")=="G" and r.get("arm")=="sampled" and r.get("set")=="R"]
 per_unit=collections.defaultdict(list)
 for r in sample: per_unit[r.get("unit_id")].append(bool(r.get("apology",bool(census139.OPENER.search(r.get("answer",""))))))
 sample_unit_shares=[sum(v)/len(v) for v in per_unit.values() if v]
 greedy_unit={r.get("unit_id"):bool(r.get("apology",bool(census139.OPENER.search(r.get("answer",""))))) for r in fullR}
 greedy_share=sum(greedy_unit.values())/len(greedy_unit) if greedy_unit else None
 sample_share=statistics.mean(sample_unit_shares) if sample_unit_shares else None
 out["predictions"]["139.12"]={"sampled_mean_unit_share":sample_share,"greedy_full_share":greedy_share,"units":len(sample_unit_shares),"replicates_per_unit":min((len(v) for v in per_unit.values()),default=0),"held":sample_share is not None and greedy_share is not None and sample_share<=greedy_share/2}
 task_rows=[r for r in rows if r.get("stage")=="C" and r.get("arm")=="full" and r.get("set")=="R" and str(r.get("variant","")) in {"0","1","2"}]
 task_results=[task_correct(r) for r in task_rows]
 task_results=[x for x in task_results if x is not None]
 task_accuracy=sum(task_results)/len(task_results) if task_results else None
 task_detail=[{"question":r.get("question"),"unit_id":r.get("unit_id"),"correct":task_correct(r)} for r in task_rows]
 out["predictions"]["139.13"]={"task_correct":sum(task_results),"task_rows":len(task_results),"accuracy":task_accuracy,"per_answer":task_detail,"held":task_accuracy is not None and task_accuracy>=.5}
 # Baseline by exact prompt, retaining each arm/checkpoint count and answers for audit.
 prompts=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in rows:
  if r.get("arm")=="plain": prompts[(r.get("checkpoint"),r.get("prompt",r.get("question")))][r.get("stage")].append({"question":r.get("question"),"variant":r.get("variant"),"answer":r.get("answer"),"answer_token_ids":r.get("answer_token_ids",[])})
 out["plain_baseline_by_prompt"]=[{"checkpoint":ck,"prompt":prompt,"by_stage":dict(stages)} for (ck,prompt),stages in sorted(prompts.items(),key=lambda x:str(x[0]))]
 return out

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input",nargs="+",default=[str(ROOT/"results/part139/results.jsonl")]); ap.add_argument("--units",default=str(HERE/"units139.json")); ap.add_argument("--out",default=str(ROOT/"results/part139/score.json")); args=ap.parse_args(argv)
 rows=read_rows(args.input); units=json.loads(Path(args.units).read_text(encoding="utf-8")); memberships={u["id"]:u["set"] for s in units["sets"].values() for u in s}
 for r in rows:
  if r.get("unit_id") in memberships:r["set"]=memberships[r["unit_id"]]
 result=score(rows); path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,ensure_ascii=True,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(result,ensure_ascii=True,indent=2))
if __name__=="__main__": main()
