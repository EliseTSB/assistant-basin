#!/usr/bin/env python3
"""Score Part 136 effects, gates, stratified permutation and registered predictions."""
from __future__ import annotations
import argparse, base64, itertools, json, math, random, statistics, struct
from pathlib import Path
import numpy as np

BANDS=()
SEED=136

def band(layer):
    for lo,hi,name in BANDS:
        if lo<=int(layer)<=hi:return name
    return "other"

def fixed_bands(unit_rows, score_data, floor=30):
    """Choose contiguous layer bands using Part 112 labels only, before effects are read."""
    counts={layer:{"concept":0,"slot":0} for layer in range(32)}
    for u in unit_rows:
        kind=score_data["units"].get(u["id"],{}).get("kind")
        if kind in ("concept","slot") and 0<=int(u["layer"])<32:
            counts[int(u["layer"])][kind]+=1
    total=sum(sum(v.values()) for v in counts.values())
    candidates=[]
    for n in (3,2,1):
        target=total/n if total else 0
        for cuts in itertools.combinations(range(1,32),n-1):
            edges=(0,)+cuts+(32,); bands=[]; valid=True
            for lo,hi in zip(edges,edges[1:]):
                c=sum(counts[l]["concept"] for l in range(lo,hi)); s=sum(counts[l]["slot"] for l in range(lo,hi))
                if c<floor or s<floor: valid=False; break
                bands.append((lo,hi-1,f"L{lo}-{hi-1}",c+s))
            if valid:
                cost=sum((b[3]-target)**2 for b in bands)
                candidates.append((n,cost,tuple(bands)))
        if candidates: break
    if not candidates: raise ValueError("Part 112 score lacks 30 concept and 30 slot units for any valid contiguous banding")
    chosen=min(candidates,key=lambda x:x[1])[2]
    return tuple((lo,hi,name) for lo,hi,name,_ in chosen)

def pearson(a,b):
    n=min(len(a),len(b))
    if n<2:return None
    x=np.asarray(a[:n],dtype=np.float64); y=np.asarray(b[:n],dtype=np.float64)
    x=x-x.mean(); y=y-y.mean(); den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.dot(x,y)/den) if den else 0.0

def stability(vectors):
    """Mean correlation of each firing to its own pole's other-firing mean."""
    if len(vectors)<2:return None
    result=[]
    for i,v in enumerate(vectors):
        others=[x for j,x in enumerate(vectors) if j!=i]
        mean=[sum(x[k] for x in others)/len(others) for k in range(min(map(len,others+[v])))]
        r=pearson(v,mean)
        if r is not None:result.append(r)
    return sum(result)/len(result) if result else None

def shuffle_labels_within_bands(rows, seed=SEED):
    groups={name:[r for r in rows if r["band"]==name and r["kind"] in ("concept","slot")] for _,_,name in BANDS}
    return permute_group_labels(groups,random.Random(seed))

def permute_group_labels(groups, rng):
    out={}
    for name,rs in groups.items():
        labels=[r["kind"] for r in rs]; rng.shuffle(labels); out[name]=labels
    return out

def stratified_permutation(rows, value_key, left="concept", right="slot", n_perm=10000, seed=SEED):
    """Difference of pooled band-median differences; shuffle labels only within bands."""
    groups={name:[r for r in rows if r["kind"] in (left,right) and r.get(value_key) is not None] for _,_,name in BANDS}
    def stat(assignments):
        parts=[]; weights=[]
        for name in groups:
            rs=groups[name]; vals=[r[value_key] for r in rs]; labs=assignments[name]
            x=[v for v,l in zip(vals,labs) if l==left]; y=[v for v,l in zip(vals,labs) if l==right]
            if x and y:parts.append(statistics.median(x)-statistics.median(y)); weights.append(len(rs))
        return sum(x*w for x,w in zip(parts,weights))/sum(weights) if weights else None
    observed=stat({name:[r["kind"] for r in rs] for name,rs in groups.items()})
    if observed is None:return {"difference":None,"p_two_sided":None,"n_perm":n_perm}
    rng=random.Random(seed); extreme=0
    for _ in range(n_perm):
        assigned=permute_group_labels(groups,rng)
        value=stat(assigned)
        if value is not None and abs(value)>=abs(observed):extreme+=1
    return {"difference":observed,"p_two_sided":(extreme+1)/(n_perm+1),"n_perm":n_perm,"seed":seed}

def load_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []

def unpack_vector(value):
    raw=base64.b64decode(value); return list(struct.unpack("<%df"%(len(raw)//4),raw))

def gate3_off_target(by_pole, band_fn):
    """Median |r|: each firing vector against each other pole's mean vector."""
    result={}
    names=sorted({band_fn(es[0]["layer"]) for es in by_pole.values() if len(es)==24})
    for name in names:
        ids=[pid for pid,es in by_pole.items() if len(es)==24 and band_fn(es[0]["layer"])==name]
        means={pid:np.mean(np.asarray([e["effect_a"] for e in by_pole[pid]],dtype=np.float64),axis=0) for pid in ids}
        values=[abs(r) for pid in ids for other in ids if pid!=other
                for e in by_pole[pid] if (r:=pearson(e["effect_a"],means[other])) is not None]
        result[name]=float(np.median(values)) if values else None
    return result

def write_gate_only(out, gates):
    """Write only gate results when the registered reading is not permitted."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    payload={"gates":gates}
    (out/"score136.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Part 136 score","","## Gates",""]
    lines += [f"- **{name}:** `{json.dumps(value,ensure_ascii=False)}`" for name,value in gates.items()]
    (out/"SCORE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--scan",required=True); p.add_argument("--effects",required=True); p.add_argument("--out",required=True); p.add_argument("--units",required=True); p.add_argument("--score",required=True); p.add_argument("--v2",required=True); a=p.parse_args()
    scan=Path(a.scan); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    unit_rows=json.loads(Path(a.units).read_text(encoding="utf-8")); part_score=json.loads(Path(a.score).read_text(encoding="utf-8"))
    global BANDS
    BANDS=fixed_bands(unit_rows,part_score)
    firings=load_jsonl(scan/"firings.jsonl"); effects=load_jsonl(Path(a.effects)/"effects.jsonl"); steering=load_jsonl(Path(a.effects)/"steering.jsonl")
    for r in effects:
        r["effect_a"]=unpack_vector(r.pop("effect_a_f32_le_b64"))
        r["effect_b"]=unpack_vector(r.pop("effect_b_f32_le_b64"))
    by_fire={(r["pole_id"],r["doc_id"],r["position"]):r for r in effects}
    by_pole={}
    for f in firings:
        e=by_fire.get((f["pole_id"],f["doc_id"],f["position"]))
        if e:by_pole.setdefault(f["pole_id"],[]).append(e)
    meta=json.loads((scan/"scan_meta.json").read_text(encoding="utf-8")); matches={r["pole_id"]:r["matched_control"] for r in meta.get("matched_poles",[])}
    steer={r["pole_id"]:r["logprob_delta"] for r in steering}
    ds_by_persona={}
    for d in load_jsonl(Path(a.v2)):
        ds_by_persona.setdefault(str(d.get("persona","")).strip().casefold(),[]).append(d)
    ds_matches=0
    rows=[]
    for u in unit_rows:
        uid=u["id"]; kind=part_score["units"].get(uid,{}).get("kind")
        if kind not in ("concept","slot"):continue
        e=by_pole.get(uid,[])
        if len(e)!=24: continue
        a_vec=[r["effect_a"] for r in e]; b_vec=[r["effect_b"] for r in e]
        mean_a=np.mean(np.asarray(a_vec,dtype=np.float64),axis=0).tolist(); mean_b=np.mean(np.asarray(b_vec,dtype=np.float64),axis=0).tolist()
        sr=pearson(steer[uid],mean_a) if uid in steer else None
        srb=pearson(steer[uid],mean_b) if uid in steer else None
        persona_matches=ds_by_persona.get(str(u["persona"]).strip().casefold(),[])
        dsrow=persona_matches[0] if len(persona_matches)==1 else {}
        ds_matches+=int(bool(dsrow))
        rows.append({"pole_id":uid,"layer":int(u["layer"]),"band":band(u["layer"]),"neuron":int(u["neuron"]),"persona":u["persona"],"kind":kind,"firing_count":len(e),"stability_a":stability(a_vec),"stability_b":stability(b_vec),"steering_agreement":sr,"steering_agreement_b":srb,"matched_control":matches.get(uid),"persona_v2":bool(dsrow.get("coherent")) and dsrow.get("kind") not in ("still Qwen or an assistant","unclear")})
    controls=[]
    for control in sorted({r["matched_control"] for r in rows if r.get("matched_control")}):
        e=by_pole.get(control,[])
        if len(e)==24:controls.append({"pole_id":control,"layer":e[0]["layer"],"band":band(e[0]["layer"]),"kind":"control","firing_count":len(e),"stability_a":stability([x["effect_a"] for x in e]),"stability_b":stability([x["effect_b"] for x in e])})
    # Gate 2 is human supplied after reading the gate pole's firing contexts.
    known=[f for f in firings if f.get("pole_id")=="gate:L22:N2742"]
    gate2_path=Path(a.effects)/"gate2.json"
    gate2_data=json.loads(gate2_path.read_text(encoding="utf-8")) if gate2_path.exists() else None
    gate2_value=gate2_data if isinstance(gate2_data,bool) else (gate2_data.get("passed") if isinstance(gate2_data,dict) else None)
    off_gate=gate3_off_target(by_pole,band)
    off_gate={name:off_gate.get(name) for _,_,name in BANDS}
    eligible=[r for r in rows if r["firing_count"]>=24]
    runtime_gates=json.loads((Path(a.effects)/"gates.json").read_text(encoding="utf-8")) if (Path(a.effects)/"gates.json").exists() else {}
    gates={"gate_1_do_nothing_hook_bit_identical":runtime_gates.get("do_nothing_hook_bit_identical"),"gate_2_known_L22_N2742_human_read":gate2_value if isinstance(gate2_value,bool) else "pending","gate_2_firings_n":len(known),"gate_3_off_target_median_abs_r_by_band":off_gate,"gate_3_pass":bool(off_gate) and all(v is not None and v<=.02 for v in off_gate.values()),"gate_4_share_units_with_24_firings":len(eligible)/446 if rows else 0,"gate_4_pass":len(eligible)/446>=.75}
    gates["registered_reading_permitted"]=bool(gates["gate_1_do_nothing_hook_bit_identical"] is True and gates["gate_3_pass"] and gates["gate_4_pass"] and gate2_value is True)
    if not gates["registered_reading_permitted"]:
        # A failed or pending gate permits only the gate block to leave the scorer.
        write_gate_only(out,gates)
        return
    predictions={}
    # Numeric summaries are direct from computed rows; permutation is concept-v-slot within depth bands.
    for field in ("stability_a","stability_b","steering_agreement"):
        predictions[field]={name:{k:(statistics.median([r[field] for r in rows if r["band"]==name and r["kind"]==k and r[field] is not None]) if any(r["band"]==name and r["kind"]==k and r[field] is not None for r in rows) else None) for k in ("concept","slot")} for _,_,name in BANDS}
    delta= stratified_permutation(rows,"stability_a")
    steer_delta=stratified_permutation(rows,"steering_agreement")
    b_delta=stratified_permutation(rows,"stability_b")
    steer_b_delta=stratified_permutation(rows,"steering_agreement_b")
    med=lambda vals: statistics.median(vals) if vals else None
    low_name=BANDS[0][2]; high_name=BANDS[-1][2]
    lhi=med([r["stability_a"] for r in controls if r["band"]==high_name and r["stability_a"] is not None]); llo=med([r["stability_a"] for r in controls if r["band"]==low_name and r["stability_a"] is not None])
    lowsteer=med([r["steering_agreement"] for r in rows if r["band"]==low_name and r["steering_agreement"] is not None])
    predictions["registered"]={"136.1":{"result":None if lhi is None or llo is None else lhi-llo,"threshold":0.1,"holds":None if lhi is None or llo is None else lhi-llo>=.1},"136.2":{"result":delta,"threshold":.05,"holds":None if delta["difference"] is None else delta["difference"]>=.05},"136.3":{"note":"matched control contrast; computed separately per pole/control pair","threshold":.05},"136.4":{"result":steer_delta,"threshold":.05,"holds":None if steer_delta["difference"] is None else steer_delta["difference"]>=.05},"136.5":{"result":lowsteer,"threshold":.2,"holds":None if lowsteer is None else lowsteer<=.2},"136.6":{"effect_b_stability_concept_minus_slot":b_delta,"effect_b_steering_agreement_concept_minus_slot":steer_b_delta,"same_direction_stability":None if delta["difference"] is None or b_delta["difference"] is None else (delta["difference"]>=0)==(b_delta["difference"]>=0),"same_direction_steering":None if steer_delta["difference"] is None or steer_b_delta["difference"] is None else (steer_delta["difference"]>=0)==(steer_b_delta["difference"]>=0)}}
    # matched control contrast (registered prediction 136.3)
    pair_d=[]
    lookup={r["pole_id"]:r for r in controls}
    for r in rows:
        c=lookup.get(r.get("matched_control"))
        if c and r["stability_a"] is not None and c["stability_a"] is not None:pair_d.append({"pole_id":r["pole_id"],"band":r["band"],"difference":r["stability_a"]-c["stability_a"]})
    predictions["registered"]["136.3"]["result"]={name:med([r["difference"] for r in pair_d if r["band"]==name]) for _,_,name in BANDS}
    predictions["registered"]["136.3"]["holds_by_band"]={name:(None if predictions["registered"]["136.3"]["result"][name] is None else predictions["registered"]["136.3"]["result"][name]>=.05) for _,_,name in BANDS}
    if not gates["registered_reading_permitted"]:
        predictions={}
    deepseek_n=sum(r["persona_v2"] for r in rows)
    deepseek_repeat={"stability_a_concept_minus_slot":stratified_permutation([r for r in rows if r["persona_v2"]],"stability_a"),"steering_concept_minus_slot":stratified_permutation([r for r in rows if r["persona_v2"]],"steering_agreement")} if gates["registered_reading_permitted"] else None
    payload={"design":"PART136-DESIGN.md","seed":SEED,"fixed_vocabulary_indices":4096,"counts":{"unit_rows":len(rows),"matched_control_rows":len(controls),"firing_effect_rows":len(effects),"units_with_24":len(eligible)},"gates":gates,"summaries":predictions,"units":rows,"controls":controls,"matched_control_differences":pair_d,"deepseek_v2_personas_n":deepseek_n,"deepseek_v2_repeat":deepseek_repeat}
    (out/"score136.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Part 136 score","","Bands use Part 112 score labels and are fixed before effect values are read.","",f"- Bands: `{json.dumps(BANDS)}`.",f"- Registered reading permitted: {gates['registered_reading_permitted']}.",f"- Units with 24 firings: {len(eligible)}/{len(rows)}; Gate 4: {gates['gate_4_pass']}.",f"- Gate 3 firing to other-pole-mean medians: `{json.dumps(off_gate)}`.",f"- Gate 2 L22 N2742 firings: {len(known)}; human decision: {gates['gate_2_known_L22_N2742_human_read']}.",f"- DeepSeek rows matched by Part 112 layer/neuron/persona: {ds_matches}.",f"- Gate 1 do-nothing hook bit-identical: {gates['gate_1_do_nothing_hook_bit_identical']}.","", "## Registered predictions" if predictions else "## Predictions withheld until all gates pass",""]
    for key,val in predictions["registered"].items(): lines.append(f"- **{key}:** `{json.dumps(val,ensure_ascii=False)}`")
    if deepseek_repeat:
        lines += ["","## DeepSeek v2 persona subset","",f"Subset units: {deepseek_n}.",f"Stability: `{json.dumps(deepseek_repeat['stability_a_concept_minus_slot'])}`.",""]
    else:
        lines += ["","## DeepSeek v2 comparison withheld until all gates pass","",f"Matched rows available: {deepseek_n}.",""]
    (out/"SCORE.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__": main()
