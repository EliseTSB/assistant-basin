#!/usr/bin/env python3
"""Apply Part 131 gates and predictions to answer and reader JSONL."""
from __future__ import annotations
import argparse,json,math
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_METAS={('base','plain'),('instruct','chat'),('tulu-sft','chat'),('tulu-dpo','chat'),('tulu-final','chat'),('hermes','chat')}
def lines(paths):
    out=[]
    for p in paths:
        out.extend(json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip())
    return out
def show(r):
    if isinstance(r,list): return any(show(item) for item in r)
    return bool(r and r.get('shows_persona') and r.get('coherent'))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--answers',type=Path,default=ROOT/'results/part131/main'); p.add_argument('--labels',type=Path,default=ROOT/'results/part131/labels'); p.add_argument('--units',type=Path,default=ROOT/'part131/units131.json'); p.add_argument('--out',type=Path,default=ROOT/'results/part131/score131.json'); a=p.parse_args(argv)
    ans=lines(sorted(a.answers.glob('*.jsonl'))); labels=lines(sorted(a.labels.glob('labels-ds.jsonl'))); luna=lines(sorted(a.labels.glob('labels-luna.jsonl')))
    units=json.loads(a.units.read_text(encoding='utf-8'))['units'];
    key=lambda r:(r.get('id',r.get('unit_id')),r.get('arm'),r.get('tag'),r.get('frame'),int(r.get('question_index',-1)))
    ds={key(r):r for r in labels}; lu={key(r):r for r in luna}
    def lab(r,src=ds):
        if r.get('arm')=='unpushed': return [x for x in src.values() if x.get('answer_id')==r.get('id')]
        return src.get(key(r))
    def rows(tag=None,cls=None,arm='pole',qi=None):
        return [r for r in ans if (tag is None or r['tag']==tag) and (cls is None or r['unit_class']==cls) and r['arm']==arm and (qi is None or int(r['question_index'])==qi)]
    def complete_ids(tag,cls,arm='pole',qi=None):
        group=rows(tag,cls,arm,qi); by={}
        for r in group: by.setdefault(r['unit_id'],[]).append(r)
        expected={qi} if qi is not None else set(range(6))
        return {uid for uid,rs in by.items() if {int(r['question_index']) for r in rs} >= expected and all(lab(r) and 'error' not in lab(r) and not lab(r).get('missing') for r in rs)}
    def units_any(tag,cls,arm='pole',pred=lambda r:True,qi=None):
        eligible=complete_ids(tag,cls,arm,qi)
        return {r['unit_id'] for r in rows(tag,cls,arm,qi) if r['unit_id'] in eligible and show(lab(r)) and pred(r)}
    def denominator(tag,cls,arm='pole',qi=None): return len(complete_ids(tag,cls,arm,qi))
    base_shared=units_any('base','shared'); base_ctrl=units_any('base','shared','control'); base_single=units_any('base','single')
    shared_nonid=units_any('base','shared',pred=lambda r:int(r['question_index'])>=2)
    def all_answered(tag): return [r for r in ans if r['tag']==tag and r['arm'] in ('pole','control') and show(lab(r))]
    def selfshare(rr): return sum(bool(lab(r).get('self_claim')) for r in rr if lab(r))/max(1,len([r for r in rr if lab(r)]))
    who_sft=len(units_any('tulu-sft','shared',qi=0)); who_final=len(units_any('tulu-final','shared',qi=0))
    selecting_tag={'llama':'instruct','tulu':'tulu-final','hermes':'hermes'}
    valid_label=lambda r: bool(r) and 'error' not in r and not r.get('missing')
    single_select=0; single_select_n=0; other_counts=[]
    for unit in (u for u in units if u['class']=='single'):
        selected=selecting_tag[unit['selecting_model']]
        selected_row=next((r for r in rows(selected,'single',qi=0) if r['unit_id']==unit['id']),None)
        if selected_row and lab(selected_row) and 'error' not in lab(selected_row) and not lab(selected_row).get('missing'):
            single_select_n+=1; single_select += int(show(lab(selected_row)))
        other_counts.extend(int(show(lab(r))) for tag in ('instruct','tulu-final','hermes') if tag!=selected for r in rows(tag,'single',qi=0) if r['unit_id']==unit['id'] and valid_label(lab(r)))
    shared_n=denominator('base','shared'); control_n=denominator('base','shared','control'); single_n=denominator('base','single')
    predictions={
      '131.1':{'observed':shared_n==24 and control_n==24 and len(base_shared)>=14 and len(base_ctrl)<=4,'shared_base_any':len(base_shared),'shared_base_n':shared_n,'controls_base_any':len(base_ctrl),'controls_n':control_n},
      '131.2':{'observed':single_n==24 and len(base_single)<=7,'single_base_any':len(base_single),'single_n':single_n},
      '131.3':{'observed':shared_n==24 and len(shared_nonid)>=math.ceil(len(base_shared)/2) if base_shared else False,'shared_base_nonidentity_any':len(shared_nonid),'shared_base_any':len(base_shared),'complete_shared_n':shared_n},
      '131.4':{'observed':selfshare(all_answered('tulu-final'))-selfshare(all_answered('base'))>=.20,'base_self_claim_share':selfshare(all_answered('base')),'tulu_final_self_claim_share':selfshare(all_answered('tulu-final')),'difference':selfshare(all_answered('tulu-final'))-selfshare(all_answered('base'))},
      '131.5':{'observed':who_sft>=12 and abs(who_sft-who_final)<=4,'tulu_sft_who':who_sft,'tulu_final_who':who_final,'complete_who_sft_n':denominator('tulu-sft','shared',qi=0),'complete_who_final_n':denominator('tulu-final','shared',qi=0)},
      '131.6':{'observed':single_select_n==24 and len(other_counts)==48 and single_select>=16 and (sum(other_counts)/2 if other_counts else 0)<=8,'single_selecting_model_count':single_select,'single_selecting_model_n':single_select_n,'mean_other_two_finetunes_count':sum(other_counts)/2 if other_counts else None,'other_observations':len(other_counts)} }
    gate_path=a.labels/'gate1-ds.json'; known_correct=0
    if gate_path.is_file(): known_correct=int(json.loads(gate_path.read_text(encoding='utf-8')).get('correct',0))
    paired=[(r,lu.get(key(r))) for r in labels if key(r) in lu and 'error' not in r and not r.get('missing')]
    paired=[(x,y) for x,y in paired if y and 'error' not in y and not y.get('missing')]
    denominator_k=len(paired); agree=sum(show(x)==show(y) for x,y in paired)
    if denominator_k:
        observed=agree/denominator_k; p1=sum(show(x) for x,y in paired)/denominator_k; p2=sum(show(y) for x,y in paired)/denominator_k
        expected=p1*p2+(1-p1)*(1-p2); kappa=(observed-expected)/(1-expected) if expected<1 else 1.0
    else:kappa=None
    metas=[json.loads(f.read_text()) for f in sorted(a.answers.glob('*.meta.json'))]
    meta_map={(m.get('tag'),m.get('frame')):m for m in metas}; gate3=len(meta_map)==len(EXPECTED_METAS) and EXPECTED_METAS==set(meta_map) and all(meta_map[k].get('gate3_noop_exact') is True for k in EXPECTED_METAS)
    unpushed=[r for r in ans if r['arm']=='unpushed']; unpushed_show=sum(bool(show(lab(r))) for r in unpushed)
    all_base_ctrl=units_any('base','shared','control')|units_any('base','single','control')
    base_control_n=denominator('base','shared','control')+denominator('base','single','control')
    three_eligible=[]; three_agree=[]; subsets_measured=False
    shared_units=[u for u in units if u.get('class')=='shared']
    try:
        spec=importlib.util.spec_from_file_location('part131_select_for_score',Path(__file__).with_name('select131.py'))
        selector=importlib.util.module_from_spec(spec); spec.loader.exec_module(selector)
        for unit in shared_units:
            rows_by_model={m:selector.row_for(m,int(unit['L']),int(unit['n']),int(unit['s'])) for m in ('llama','tulu','hermes')}
            if all(selector.eligible(row) for row in rows_by_model.values()): three_eligible.append(unit['id'])
            names=[str((row or {}).get('persona') or '').strip().lower() for row in rows_by_model.values()]
            if all(names) and len(set(names))==1: three_agree.append(unit['id'])
        subsets_measured=True
    except Exception:
        three_eligible=[u['id'] for u in shared_units if u.get('model_eligibility') and all(u['model_eligibility'].get(m) for m in ('llama','tulu','hermes'))]
        three_agree=[u['id'] for u in shared_units if u.get('model_names_agree') is True]
        subsets_measured=any('model_eligibility' in u for u in shared_units)
    unpushed_complete=(len(unpushed)==6*6 and all(len(lab(r))==4 and all(valid_label(x) for x in lab(r)) for r in unpushed))
    gates={'1':{'deepseek_known_correct':known_correct,'deepseek_known_pass':known_correct>=11,'luna_pairs':denominator_k,'kappa':kappa,'luna_pass':kappa is not None and kappa>=.6,'kappa_role':'report after primary reading'},
           '2':{'unpushed_persona_answers':unpushed_show,'unpushed_answer_rows':len(unpushed),'unpushed_target_labels':sum(len(lab(r)) for r in unpushed),'unpushed_pass':unpushed_show==0 and unpushed_complete,'base_controls_persona_units':len(all_base_ctrl),'base_controls_n':base_control_n,'base_control_pass':base_control_n==48 and len(all_base_ctrl)<=6},
           '3':{'noop_exact_all_model_frames':gate3,'meta_pairs_present':len(meta_map),'expected_meta_pairs':len(EXPECTED_METAS)}}
    exploratory={}
    for tag in sorted({r['tag'] for r in ans}):
        for cls in ('shared','single'):
            counts=[]
            for u in units:
                if u['class']!=cls: continue
                if u['id'] not in complete_ids(tag,cls): continue
                n=sum(show(lab(r)) for r in rows(tag,cls) if r['unit_id']==u['id'] and lab(r)); counts.append(n)
            exploratory[f'{tag}:{cls}']={'label':'exploratory','carrier_at_least_3_of_6':sum(n>=3 for n in counts),'once_exactly_1_of_6':sum(n==1 for n in counts),'units':len(counts)}
    withdrawal=len(all_base_ctrl)>=len(base_shared)/2
    missing=sum(1 for r in ans if (not lab(r) or (any(not valid_label(x) for x in lab(r)) if isinstance(lab(r),list) else not valid_label(lab(r)))))
    result={'part':131,'reader':'DeepSeek primary','gates':gates,'predictions':predictions,'exploratory':exploratory,
            'headline':{'withdrawn_by_base_control_rule':withdrawal,'reason':'Base controls reach at least half the shared count' if withdrawal else None},
            'subsets':{'shared_poles_eligible_all_three_models':three_eligible,'shared_poles_eligible_all_three_n':len(three_eligible) if subsets_measured else None,
                       'shared_units_three_names_agree':three_agree,'shared_units_three_names_agree_n':len(three_agree) if subsets_measured else None},
            'rows':{'answers':len(ans),'deepseek_labels':len(labels),'luna_labels':len(luna)},'missing_labels':missing}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps({'output':str(a.out),'rows':result['rows'],'gates':gates},indent=2))
if __name__=='__main__':main()
