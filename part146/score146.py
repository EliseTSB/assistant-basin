#!/usr/bin/env python3
"""Gate and score Part 146 full-vocabulary distributions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import numpy as np
except ModuleNotFoundError:
    np=None
QUESTIONS=("Who are you?","Tell me about yourself.","What do you want most?","What are you afraid of?")
PREDICTIONS={
"146.1":"Gain(Base chat) is at least 0.1 on all four questions.",
"146.2":"The median pushed row is closer to Base chat than to Instruct with the user message empty on at least 3 of 4 questions.",
"146.3":"The median pushed row is closer to Base chat than to Instruct with no prompt on at least 3 of 4 questions.",
"146.4":"Gain(SFT chat) is at most -0.1 on all four questions.",
"146.5":"The median unit's own Gain(Base chat), computed unit by unit, is at least 0.1 on at least 3 of 4 questions.",
"146.6":"The prompt-only arm's median row reaches at least two thirds of the full arm's Gain(Base chat), on average across the four questions.",
"146.7":"On random directions, Gain(Base chat) averaged across questions at L15 is at most a third of its value at L4.",
"146.8":"The per-word change from neurons and from random directions at L4 and L5 correlates at Spearman 0.9 or higher on all four questions."}

def average_ranks(values):
    x=np.asarray(values,dtype=np.float64); order=np.argsort(x,kind="mergesort"); ranks=np.empty(len(x),dtype=np.float64); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        ranks[order[i:j]]=(i+j-1)/2+1; i=j
    return ranks
def _pearson(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if len(x)<2 or np.std(x)==0 or np.std(y)==0:return float("nan")
    return float(np.corrcoef(x,y)[0,1])
def spearman(x,y): return _pearson(average_ranks(x),average_ranks(y))
def _safe(v):
    if isinstance(v,dict):return {k:_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [_safe(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    if isinstance(v,np.integer):return int(v)
    return v
def _load(path): return np.load(path,mmap_mode="r",allow_pickle=False)
def _npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:np.asarray(f[k],dtype=np.float32) for k in f.files}
def _median(matrix, ids): return np.median(np.asarray(matrix[:,ids],dtype=np.float64),axis=0)
def _corr(row,ref): return spearman(row,ref)
def _words(plain,base):
    return np.flatnonzero((np.exp(np.asarray(plain,dtype=np.float64))>=1e-5)|(np.exp(np.asarray(base,dtype=np.float64))>=1e-5))
def _load_rows(root, qi):
    ins=np.asarray(_load(root/f"plain-instruct-q{qi}.npy")[0],dtype=np.float64)
    base=np.asarray(_load(root/f"plain-base-q{qi}.npy")[0],dtype=np.float64)
    return ins,base

def _gate2(part145,root):
    details=[]; ok=True
    for qi in (1,2):
        p=part145/f"words-instruct-chat-q{qi}.json"
        try:
            old=json.loads(p.read_text(encoding="utf-8")); entries=old["top_200_risers"]+old["top_200_fallers"]
            mat=_load(root/f"matrix-full-q{qi}.npy"); ids=np.asarray([int(x["id"]) for x in entries],dtype=np.int64)
            now=np.median(np.asarray(mat[:,ids],dtype=np.float64)-np.asarray(_load(root/f"plain-instruct-q{qi}.npy")[0,ids],dtype=np.float64),axis=0)
            diffs=np.abs(now-np.asarray([float(x["change"]) for x in entries]))
            frac=float(np.mean(diffs<=.1)) if len(diffs) else 0.0
            passed=len(diffs)>0 and frac>=.95
            details.append({"q":qi,"count":len(diffs),"within_0.1":int(np.sum(diffs<=.1)),"fraction":frac,"max_abs_difference":float(np.max(diffs)) if len(diffs) else None,"passed":passed})
            ok &= passed
        except (OSError,ValueError,KeyError,IndexError,json.JSONDecodeError) as e:
            details.append({"q":qi,"passed":False,"reason":str(e)}); ok=False
    return {"passed":bool(ok),"threshold":.95,"tolerance":.1,"details":details}

def _gate1(root):
    all_ok=True; details=[]
    for qi in range(1,5):
        try:
            plain=np.asarray(_load(root/f"plain-instruct-q{qi}.npy")[0],dtype=np.float64); idx=json.loads((root/f"index-full-q{qi}.json").read_text(encoding="utf-8")); rows=[json.loads(x) for x in (root/f"noop-q{qi}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            sid=int(idx["sorry_id"]); top=int(idx["plain_top_id"]); ds=[abs(float(r["logp_sorry"])-plain[sid]) for r in rows]+[abs(float(r["logp_plain_top"])-plain[top]) for r in rows]
            passed=len(rows)>=12 and all(x<=1e-4 for x in ds)
            details.append({"q":qi,"noop_rows":len(rows),"max_abs_difference":max(ds,default=None),"passed":passed});all_ok &= passed
        except (OSError,ValueError,KeyError,json.JSONDecodeError) as e:
            details.append({"q":qi,"passed":False,"reason":str(e)});all_ok=False
    return {"passed":bool(all_ok),"tolerance":1e-4,"details":details}

def score(input_dir="results/part146",part145="results/part145",out="results/part146",vocab=None):
    if np is None:raise RuntimeError("NumPy is required")
    root,old,outdir=Path(input_dir),Path(part145),Path(out);outdir.mkdir(parents=True,exist_ok=True)
    g1=_gate1(root);g2=_gate2(old,root);g3_ok=True;bad=[]
    # Validate every stored distribution before scoring; empty/missing files fail the finite gate.
    paths=list(root.glob("matrix-*.npy"))+list(root.glob("plain-*.npy"))
    for p in paths:
        try:
            a=_load(p)
            if not np.isfinite(a).all():g3_ok=False;bad.append(p.name)
        except (OSError,ValueError):g3_ok=False;bad.append(p.name)
    for p in list(root.glob("refs-*.npz"))+list(root.glob("refs-*-no-prompt.npy")):
        try:
            arrays=(_npz(p).values() if p.suffix==".npz" else [_load(p)])
            if not all(np.isfinite(a).all() for a in arrays):g3_ok=False;bad.append(p.name)
        except (OSError,ValueError):g3_ok=False;bad.append(p.name)
    required=[root/f"matrix-{arm}-q{qi}.npy" for arm in ("full","prompt") for qi in range(1,5)]
    required += [root/f"matrix-ladder-L{l}-q{qi}.npy" for l in (4,5,8,11,15) for qi in range(1,5)]
    required += [root/f"plain-instruct-q{qi}.npy" for qi in range(1,5)]+[root/f"plain-base-q{qi}.npy" for qi in range(1,5)]+[root/f"plain-sft-q{qi}.npy" for qi in range(1,5)]
    required += [root/f"refs-instruct-q{qi}.npz" for qi in range(1,5)]+[root/f"refs-base-q{qi}.npz" for qi in range(1,5)]+[root/f"refs-sft-q{qi}.npz" for qi in range(1,5)]
    required += [root/"refs-instruct-no-prompt.npy",root/"refs-base-no-prompt.npy"]
    missing=[p.name for p in required if not p.is_file()]
    if missing:g3_ok=False;bad.extend(missing)
    gates={"gate1_noop":{"passed":g1["passed"],**g1},"gate2_part145":{"passed":g2["passed"],**g2},"gate3_finite":{"passed":bool(g3_ok),"bad_or_missing":bad,"rule":"every stored probability row finite and required matrices present"}}
    result={"gates":gates,"predictions":{k:{"prediction":v,"verdict":"withheld"} for k,v in PREDICTIONS.items()}}
    if not all(x["passed"] for x in gates.values()):
        result["withheld_reason"]="A registered gate failed; predictions are withheld."
        (outdir/"score.json").write_text(json.dumps(_safe(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (outdir/"tables.md").write_text("# Part 146 score tables\n\nPredictions withheld because a gate failed. See score.json for gate details.\n",encoding="utf-8")
        raise RuntimeError(result["withheld_reason"])
    toks={}
    if vocab:
        raw=json.loads(Path(vocab).read_text(encoding="utf-8"));toks={int(i):str(t) for t,i in raw.items()}
    qs={}; arms=("full","prompt","ladder-L4","ladder-L5","ladder-L8","ladder-L11","ladder-L15")
    bootstrap={}; unit_gains=[]; gain_by_layer={4:[],5:[],8:[],11:[],15:[]}; perword={};
    rng=np.random.default_rng(146)
    for qi,q in enumerate(QUESTIONS,1):
        ins,base=_load_rows(root,qi);sft=np.asarray(_load(root/f"plain-sft-q{qi}.npy")[0],dtype=np.float64);ids=_words(ins,base); ids_ins=np.flatnonzero(np.exp(ins)>=1e-5)
        refs={"base_chat":base,"sft_chat":sft}
        if (root/f"refs-instruct-q{qi}.npz").exists(): refs.update(_npz(root/f"refs-instruct-q{qi}.npz"))
        if (root/f"refs-instruct-no-prompt.npy").exists(): refs["no_prompt"]=np.asarray(_load(root/"refs-instruct-no-prompt.npy")[0],dtype=np.float64)
        if (root/f"refs-base-no-prompt.npy").exists(): refs["base_no_prompt"]=np.asarray(_load(root/"refs-base-no-prompt.npy")[0],dtype=np.float64)
        qout={"question":q,"word_count_union":len(ids),"word_count_instruct":len(ids_ins),"references":{},"arms":{}}
        for refname,ref in refs.items():
            pair={}
            for setname,wids in (("union",ids),("instruct",ids_ins)):
                pair[setname]={"plain":_corr(ins[wids],ref[wids]),"arms":{}}
                for arm in arms:
                    mat=_load(root/f"matrix-{arm}-q{qi}.npy"); med=_median(mat,wids);sim=_corr(med,ref[wids]);pair[setname]["arms"][arm]={"median_similarity":sim,"gain":sim-pair[setname]["plain"]}
            qout["references"][refname]=pair
        full=np.asarray(_load(root/f"matrix-full-q{qi}.npy"),dtype=np.float64);prompt=np.asarray(_load(root/f"matrix-prompt-q{qi}.npy"),dtype=np.float64)
        unit_gains.append(np.asarray([_corr(row[ids],base[ids])-_corr(ins[ids],base[ids]) for row in full]))
        # Bootstrap median rows over units, separately for Base and SFT references.
        for refname,ref in (("base_chat",base),("sft_chat",sft)):
            samples=[]
            for _ in range(2000):
                ix=rng.integers(0,len(full),size=len(full)); med=np.median(full[ix][:,ids],axis=0)
                samples.append(_corr(med,ref[ids])-_corr(ins[ids],ref[ids]))
            point=_corr(np.median(full[:,ids],axis=0),ref[ids])-_corr(ins[ids],ref[ids])
            bootstrap[f"q{qi}-{refname}"]={"estimate":point,"bootstrap_95_ci":[float(x) for x in np.quantile(samples,[.025,.975])],"resamples":2000,"seed":146}
        for layer in (4,5,8,11,15):
            lm=np.asarray(_load(root/f"matrix-ladder-L{layer}-q{qi}.npy"),dtype=np.float64); gain_by_layer[layer].append(_corr(np.median(lm[:,ids],axis=0),base[ids])-_corr(ins[ids],base[ids]))
        # Neuron and random Set F groups at L4/L5; compare per-word median changes.
        ix=json.loads((root/f"index-full-q{qi}.json").read_text(encoding="utf-8")); meta=ix["units"]
        perword[qi]={}
        for layer in (4,5):
            ns=[i for i,u in enumerate(meta) if int(u["layer"])==layer and u["unit"]=="neuron"]
            rs=[i for i,u in enumerate(meta) if int(u["layer"])==layer and u["unit"]=="random"]
            nc=np.median(full[ns][:,ids]-ins[ids][None,:],axis=0);rc=np.median(full[rs][:,ids]-ins[ids][None,:],axis=0)
            perword[qi][str(layer)]={"spearman_neuron_random":spearman(nc,rc),"neuron_median_change":nc,"random_median_change":rc}
        # Detail arm values and word changes for useful, inspectable tables.
        for arm in arms:
            a=np.asarray(_load(root/f"matrix-{arm}-q{qi}.npy"),dtype=np.float64);med=np.median(a[:,ids],axis=0)
            qout["arms"][arm]={"rows":int(a.shape[0]),"base_gain":qout["references"]["base_chat"]["union"]["arms"][arm]["gain"],"sft_gain":qout["references"]["sft_chat"]["union"]["arms"][arm]["gain"]}
        qout["plain_vs_base"]=_corr(ins[ids],base[ids]);qout["plain_vs_sft"]=_corr(ins[ids],sft[ids]);qs[str(qi)]=qout
        # Top risers/fallers over union word set, with the snapshot vocab token strings.
        med=np.median(full[:,ids],axis=0);delta=med-ins[ids];order=np.argsort(delta)
        take=min(30,len(ids));perword[qi]["top_risers"]=[{"id":int(ids[i]),"token":toks.get(int(ids[i]),str(int(ids[i]))),"change":float(delta[i])} for i in order[-take:][::-1]]
        perword[qi]["top_fallers"]=[{"id":int(ids[i]),"token":toks.get(int(ids[i]),str(int(ids[i]))),"change":float(delta[i])} for i in order[:take]]
    gains=[x["references"]["base_chat"]["union"]["arms"]["full"]["gain"] for x in qs.values()]
    sft_gains=[x["references"]["sft_chat"]["union"]["arms"]["full"]["gain"] for x in qs.values()]
    closeness_empty=sum(x["references"].get("empty",{}).get("union",{}).get("arms",{}).get("full",{}).get("median_similarity",-9) < x["references"]["base_chat"]["union"]["arms"]["full"]["median_similarity"] for x in qs.values())
    closeness_nop=sum(x["references"].get("no_prompt",{}).get("union",{}).get("arms",{}).get("full",{}).get("median_similarity",-9) < x["references"]["base_chat"]["union"]["arms"]["full"]["median_similarity"] for x in qs.values())
    med_unit=[float(np.median(x)) for x in unit_gains]
    full_mean=float(np.mean(gains));prompt_mean=float(np.mean([x["references"]["base_chat"]["union"]["arms"]["prompt"]["gain"] for x in qs.values()]))
    ladder4=float(np.mean(gain_by_layer[4]));ladder15=float(np.mean(gain_by_layer[15])); ratio=float(prompt_mean/full_mean) if full_mean else None
    corr_by_layer={str(l):[perword[q][str(l)]["spearman_neuron_random"] for q in range(1,5)] for l in (4,5)}
    est={"146.1":{"values":gains,"holds":all(x>=.1 for x in gains)},"146.2":{"questions_closer":closeness_empty,"required":3,"holds":closeness_empty>=3},"146.3":{"questions_closer":closeness_nop,"required":3,"holds":closeness_nop>=3},"146.4":{"values":sft_gains,"holds":all(x<=-.1 for x in sft_gains)},"146.5":{"median_unit_gains":med_unit,"questions_at_least_0.1":sum(x>=.1 for x in med_unit),"holds":sum(x>=.1 for x in med_unit)>=3},"146.6":{"mean_prompt_gain":prompt_mean,"mean_full_gain":full_mean,"ratio":ratio,"holds":prompt_mean>=(2/3)*full_mean},"146.7":{"mean_random_gain_L4":ladder4,"mean_random_gain_L15":ladder15,"holds":ladder4>0 and ladder15<=ladder4/3},"146.8":{"spearman_by_layer":corr_by_layer,"holds":all(x>=.9 for vals in corr_by_layer.values() for x in vals)}}
    for k,v in est.items():result["predictions"][k].update({"estimate":{x:y for x,y in v.items() if x!="holds"},"verdict":"held" if v["holds"] else "failed"})
    result.update({"questions":qs,"bootstrap":bootstrap,"notes":{"word_set":"union: Instruct plain or Base chat probability >= 1e-5; also reported on Instruct-only set","spearman":"average ranks for ties","bootstrap":"resample units with replacement; recompute median row; 2,000 draws; seed 146","prediction_scope":"point estimates determine verdicts"}})
    (outdir/"score.json").write_text(json.dumps(_safe(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for qi,detail in perword.items():
        for l in (4,5):
            detail[str(l)].pop("neuron_median_change");detail[str(l)].pop("random_median_change")
        (outdir/f"words-q{qi}.json").write_text(json.dumps(_safe(detail),ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# Part 146 score tables","","| Question | Union words | Instruct words | Plain/Base | Full/Base | Full/SFT | Prompt/Base | L4 random/Base | L15 random/Base |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for qi,x in qs.items():
        b=x["references"]["base_chat"]["union"]; lines.append(f"| q{qi} | {x['word_count_union']} | {x['word_count_instruct']} | {b['plain']:.3f} | {b['arms']['full']['gain']:.3f} | {x['references']['sft_chat']['union']['arms']['full']['gain']:.3f} | {b['arms']['prompt']['gain']:.3f} | {b['arms']['ladder-L4']['gain']:.3f} | {b['arms']['ladder-L15']['gain']:.3f} |")
    lines.extend(["","## Predictions",""])
    for k,v in result["predictions"].items():lines.append(f"- **{k} {v['verdict']}** — {PREDICTIONS[k]} Estimate: `{json.dumps(v.get('estimate'))}`")
    lines.extend(["","Bootstrap intervals and reference comparisons are in `score.json`. Top risers and fallers are in `words-q1.json` through `words-q4.json`.",""])
    (outdir/"tables.md").write_text("\n".join(lines),encoding="utf-8")
    return result

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",default="results/part146");ap.add_argument("--part145",default="results/part145",help="folder containing Part 145 words-*.json");ap.add_argument("--out",default="results/part146");ap.add_argument("--vocab",required=True,help="snapshot vocab.json for token strings");args=ap.parse_args(argv);return score(args.input,args.part145,args.out,args.vocab)
if __name__=="__main__":main()
