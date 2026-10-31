#!/usr/bin/env python3
"""Score Part 129 from saved sufficient statistics and the pinned Neuron Lens catalog."""
import argparse, gzip, json, random, re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CHAT_REGISTER = re.compile(r"\b(?:assistant|chatbot|conversation|user|polite|apolog\w*|helpful|sorry|i['’]m|as an ai|response)\b", re.I)
GROUPS = ("web", "zh", "code", "chat-user", "chat-assistant")

def corr(d, gi, layer, rect=None):
    n = float(d["count"][gi, layer])
    if n < 1: return np.full(d["data"].shape[-1], np.nan)
    if rect is None:
        s = d["data"][gi, layer, :5].numpy()
        sx, sy, xx, yy, xy = s
    else:
        base = 8 + 5 * rect
        sx, sy, xx, yy, xy = d["data"][gi, layer, base:base+5].numpy()
    vx, vy = xx - sx*sx/n, yy - sy*sy/n
    return np.divide(xy-sx*sy/n, np.sqrt(np.maximum(vx,0)*np.maximum(vy,0)), out=np.full_like(sx,np.nan), where=(vx>0)&(vy>0))

def nullcorr(d, gi, layer):
    n=float(d["count"][gi,layer])
    if n<1:return np.full(d["data"].shape[-1],np.nan)
    s=d["data"][gi,layer]
    sx,sy,xx,yy,xy=(s[i].numpy() for i in (5,1,6,3,7))
    vx,vy=xx-sx*sx/n,yy-sy*sy/n
    return np.divide(xy-sx*sy/n,np.sqrt(np.maximum(vx,0)*np.maximum(vy,0)),out=np.full_like(sx,np.nan),where=(vx>0)&(vy>0))

def med(x):
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]; return float(np.median(a)) if len(a) else None

def alignment_verdict(checks, values):
    ok=[]
    for r in checks:
        key=f"{r['layer']}:{r['neuron']}:{r['pole']}:{r['slice']}:{r['line']}:{r['position']}"
        if key in values and r["top_val"]>0:
            ok.append(abs(values[key]-r["top_val"])/r["top_val"]<=.05)
    required=len(checks)
    return {"checks_measured":len(ok),"checks_required":required,"passed":sum(ok),
            "passed_fraction":sum(ok)/max(1,len(ok)),
            "pass":len(ok)==required and bool(ok) and sum(ok)/len(ok)>=.9}

def eligible_matches(poles, non_poles, activity):
    """Return active persona poles and an equally sized active non-persona sample pool."""
    active_poles=[(n,s) for n,s in poles if activity(n,s)]
    active_non=[(n,s) for n,s in non_poles if activity(n,s)]
    return active_poles, active_non

def verify_prepared_provenance(prepared, docs_path):
    if not prepared:
        return
    if not prepared.get("docs_sha256"):
        raise ValueError("prepared129.pt is missing docs.parquet provenance; regenerate it")
    import hashlib
    actual=hashlib.sha256(Path(docs_path).read_bytes()).hexdigest()
    if actual != prepared["docs_sha256"]:
        raise ValueError("prepared129.pt was built from a different docs.parquet")

def row_tables(catalog):
    acts=[]; labels=[]
    for L in range(36):
        acts.append(pq.read_table(Path(catalog)/"neurons/acts"/f"L{L:02d}.parquet").to_pylist())
        labels.append(pq.read_table(Path(catalog)/"neurons/labels"/f"L{L:02d}.parquet").to_pylist())
    return acts, labels

def carry_counts(repo, tag):
    labels={json.loads(x)["id"]:json.loads(x) for x in open(Path(repo)/"results/part91/labels.jsonl",encoding="utf-8")}
    counts={}
    for i,line in enumerate(open(Path(repo)/f"results/part91/{tag}.jsonl",encoding="utf-8")):
        row=json.loads(line)
        if not row["cell"].startswith("p"): continue
        if row["cell"]!="unpushed":
            counts[row["cell"]]=counts.get(row["cell"],0)+int(labels[f"{tag}:{i}"]["shows_persona"] is True)
    return counts

def carry_class(n):
    return "carrier" if n>=3 else "once" if n==1 else "other"

def library_personas(path):
    found=set()
    for L in range(36):
        with gzip.open(Path(path)/f"L{L:02d}.gz","rt",encoding="utf-8") as f: d=json.load(f)
        cats=d["categories"]
        for i,r in enumerate(d["rows"]):
            if r and cats[r[0]]=="persona": found.add((L,i//2,1 if i%2==0 else -1))
    return found

def score(sums, catalog, library, repo, out, gate2_paths, prepared_file=None):
    data=torch_load(sums)
    if "state" in data:
        data={**data["state"],"meta":data.get("meta",{})}
    docs_path=Path(catalog)/"docs.parquet"
    prepared_path=Path(prepared_file) if prepared_file else Path(sums).parent/"prepared129.pt"
    prepared_data=torch_load(prepared_path) if prepared_path.is_file() else {}
    verify_prepared_provenance(prepared_data,docs_path)
    acts, labels=row_tables(catalog)
    group={name:i for i,name in enumerate(GROUPS)}
    rs={name:np.stack([corr(data,group[name],L) for L in range(36)]) for name in GROUPS}
    # Keep a neuron when either polarity clears the registered 100-position activity floor.
    for name in GROUPS:
        gi=group[name]
        for L in range(36):
            eligible=(data["above"][gi,L].numpy()>=100).any(0)
            rs[name][L,~eligible]=np.nan
    chat=rs["chat-assistant"]; web=rs["web"]
    layer_web=[med(web[L]) for L in range(36)]
    p1=med(web.reshape(-1))
    chat_median=med(chat.reshape(-1))
    p2=chat_median-p1 if chat_median is not None and p1 is not None else None
    layer_chat=np.array([med(chat[L]) for L in range(36)],dtype=object)
    late,early=med(layer_chat[30:36]),med(layer_chat[:9])
    p3=late-early if late is not None and early is not None else None
    cos=torch_load(Path(sums).parent/"weight-cosines.pt")["layer_cosines"]
    try:
        from scipy.stats import spearmanr
        p4=[]
        for L in range(36):
            x=1-chat[L]; y=1-cos[L]["down"].numpy(); ok=np.isfinite(x)&np.isfinite(y)
            rho=float(spearmanr(x[ok],y[ok]).statistic) if ok.sum()>2 else np.nan
            p4.append(abs(rho) if np.isfinite(rho) else None)
        count_p4=sum(v<=.3 for v in p4 if v is not None)
    except ImportError: p4=[]; count_p4=0
    persona=library_personas(library)
    rng=random.Random(129); matched=[]; persona_r=[]
    for L in range(36):
        poles=[(n,s) for ll,n,s in persona if ll==L]
        non=[(r["neuron"],r["pole"]) for r in acts[L] if (L,r["neuron"],r["pole"]) not in persona]
        active, eligible_non=eligible_matches(
            poles,non,lambda n,s: data["above"][group["web"],L,0 if s>0 else 1,n]>=100)
        if poles:
            if len(eligible_non)<len(active):
                raise ValueError(f"layer {L} has too few active non-persona poles to match personas")
            persona_r += [corr(data,group["web"],L,0 if s>0 else 1)[n] for n,s in active]
            sampled=rng.sample(eligible_non,len(active))
            matched += [corr(data,group["web"],L,0 if s>0 else 1)[n] for n,s in sampled]
            assert len(sampled)==len(active)
    p5w=(med(persona_r)-med(matched) if med(persona_r) is not None and med(matched) is not None else None)
    # Repeat persona/control comparison on assistant positions.
    pr=[]; mr=[]
    for L in range(36):
        poles=[(n,s) for ll,n,s in persona if ll==L]
        non=[(r["neuron"],r["pole"]) for r in acts[L] if (L,r["neuron"],r["pole"]) not in persona]
        rrp=corr(data,group["chat-assistant"],L,0); rrn=corr(data,group["chat-assistant"],L,1)
        active, eligible_non=eligible_matches(poles,non,lambda n,s:data["above"][group["chat-assistant"],L,0 if s>0 else 1,n]>=100)
        if len(eligible_non)<len(active):
            raise ValueError(f"layer {L} has too few active non-persona poles to match chat personas")
        pr += [rrp[n] if s>0 else rrn[n] for n,s in active]
        mr += [(rrp if s>0 else rrn)[n] for n,s in rng.sample(eligible_non,len(active))]
    p5c=(med(pr)-med(mr) if med(pr) is not None and med(mr) is not None else None)
    chat_counts, base_counts=carry_counts(repo,"inst_chat"),carry_counts(repo,"base_plain")
    neurons=json.load(open(Path(repo)/"part91/neurons91.json",encoding="utf-8"))
    once_r=[corr(data,group["chat-assistant"],int(p["L"]),0 if int(p["s"])>0 else 1)[int(p["n"])] for p in neurons if carry_class(chat_counts.get(p["id"],0))=="once" for L in [int(p["L"])] if data["above"][group["chat-assistant"],L,0 if int(p["s"])>0 else 1,int(p["n"]) ]>=100]
    carrier_r=[corr(data,group["chat-assistant"],int(p["L"]),0 if int(p["s"])>0 else 1)[int(p["n"])] for p in neurons if carry_class(chat_counts.get(p["id"],0))=="carrier" for L in [int(p["L"])] if data["above"][group["chat-assistant"],L,0 if int(p["s"])>0 else 1,int(p["n"]) ]>=100]
    p6=med(once_r)-med(carrier_r) if med(once_r) is not None and med(carrier_r) is not None else None
    by_layer={L:sorted(range(chat.shape[1]),key=lambda n:chat[L,n] if np.isfinite(chat[L,n]) else np.inf) for L in range(36)}
    low=sorted([(L,n) for L in range(36) for n in range(chat.shape[1]) if np.isfinite(chat[L,n])],key=lambda q:chat[q[0],q[1]])[:442]
    rand=[]
    for L in range(36):
        k=sum(ll==L for ll,n in low)
        pool=[n for n in range(chat.shape[1]) if (L,n) not in low and np.isfinite(chat[L,n])]
        rand += [(L,n) for n in rng.sample(pool,min(k,len(pool)))]
    label_map={(L,int(r["neuron"]),int(r["pole"])):str(r.get("fires",""))+" "+str(r.get("writes","")) for L,tab in enumerate(labels) for r in tab}
    hits=lambda pairs: sum(bool(CHAT_REGISTER.search(label_map.get((L,n,1),"")+" "+label_map.get((L,n,-1),""))) for L,n in pairs)
    a,b=hits(low),hits(rand); ratio=(a/max(1,len(low)))/(b/max(1,len(rand))) if b else None
    p7=(ratio>=2) if ratio is not None else (a>0 and len(low)>0)
    null_by_group={name:[med(np.abs(nullcorr(data,gi,L))) for L in range(36)] for name,gi in group.items()}
    null_by_layer=[max((rows[L] for rows in null_by_group.values() if rows[L] is not None),default=None) for L in range(36)]
    gate1=None
    align_path=Path(sums).parent/"alignment-values.json"
    if align_path.is_file() and prepared_path.is_file():
        vals=json.loads(align_path.read_text())
        checks=prepared_data["pilot_alignment"] if data.get("meta",{}).get("stage")=="pilot" else prepared_data["alignment"]
        gate1=alignment_verdict(checks,vals)
    gate3={"group_layer_median_abs_null_r":null_by_group,"pass":all(v is not None and v<=.05 for rows in null_by_group.values() for v in rows)}
    gate2=score_gate2(gate2_paths)
    report={"gates":{"gate1_alignment":gate1,"gate2_noise":gate2,"gate3_null":gate3},"predictions":{
      "129.1":{"web_median_r":p1,"layer_medians":layer_web,"pass":(p1>=.85 and min(x for x in layer_web if x is not None)>=.75) if p1 is not None and all(x is not None for x in layer_web) else None},
      "129.2":{"chat_assistant_minus_web":p2,"pass":p2<=-.03 if p2 is not None else None},
      "129.3":{"late_minus_early_layer_median":p3,"pass":p3<=-.05 if p3 is not None else None},
      "129.4":{"layers_abs_spearman_le_0_3":count_p4,"layer_correlations":p4,"pass":count_p4>=30},
      "129.5":{"web_persona_minus_matched":p5w,"chat_persona_minus_matched":p5c,"pass":abs(p5w)<=.03 and abs(p5c)<=.03 if p5w is not None and p5c is not None else None},
      "129.6":{"once_minus_carrier_chat_assistant_rectified":p6,"pass":p6<=-.05 if p6 is not None else None},
      "129.7":{"lowest_n":len(low),"low_label_hits":a,"matched_n":len(rand),"matched_label_hits":b,"rate_ratio":ratio,"pass":p7}},
      "excluded_under_100_positions":int((data["above"]<100).sum()),"position_counts":data["count"].tolist()}
    Path(out).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Gate | pass")
    for name,value in report["gates"].items(): print(f"{name} | {None if value is None else value.get('pass')}")
    print("Prediction | value | pass")
    for k,v in report["predictions"].items():
        value=next((x for x in v.values() if isinstance(x,(float,int)) and not isinstance(x,bool)),None)
        print(f"{k} | {value} | {v.get('pass')}")

def score_gate2(paths):
    import torch
    if len(paths)!=2 or any(not Path(p).is_file() for p in paths): return None
    summaries=[]; batches=[]; captures=[]
    for path in paths:
        d=torch_load(path)
        if "state" in d:
            d={**d["state"],"meta":d.get("meta",{})}
        if "data" not in d or "count" not in d or d["data"].ndim!=4 or d["data"].shape[2]<5 or d["count"].ndim!=2:
            raise ValueError(f"malformed Gate 2 capture: {path}")
        batches.append(d.get("meta",{}).get("batch"))
        captures.append((d["data"].contiguous(), d["count"].contiguous()))
        layers=[]
        for L in range(36):
            rows=d["data"][:,L,:5,:].sum(0).numpy()
            n=float(d["count"][:,L].sum())
            sx,sy,xx,yy,xy=rows
            vx,vy=xx-sx*sx/n,yy-sy*sy/n
            rr=np.divide(xy-sx*sy/n,np.sqrt(np.maximum(vx,0)*np.maximum(vy,0)),out=np.full_like(sx,np.nan),where=(vx>0)&(vy>0))
            layers.append(med(rr))
        summaries.append(layers)
    if batches[0] is None or batches[1] is None or batches[0]==batches[1]:
        return {"batch_runs":list(paths),"batches":batches,"pass":False,"reason":"batch sizes must differ"}
    # A copied capture with only metadata relabelled is still the same measurement.
    if all(left.shape == right.shape and torch.equal(left.view(torch.uint8), right.view(torch.uint8))
           for left, right in zip(captures[0], captures[1])):
        return {"batch_runs":list(paths),"batches":batches,"pass":False,"reason":"capture tensors are byte-identical"}
    minimum=[min(s[L] for s in summaries if s[L] is not None) for L in range(36)]
    return {"batch_runs":list(paths),"batches":batches,"layer_median_r_min":minimum,"pass":all(v is not None and v>=.999 for v in minimum)}

def torch_load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--sums",required=True); p.add_argument("--catalog",required=True); p.add_argument("--library",required=True); p.add_argument("--repo",required=True); p.add_argument("--out",default="score129.json"); p.add_argument("--gate2",nargs=2,required=False,default=[]); p.add_argument("--prepared"); a=p.parse_args(); score(a.sums,a.catalog,a.library,a.repo,a.out,a.gate2,a.prepared)
if __name__=="__main__":main()
