#!/usr/bin/env python3
"""Apply registered Part 131 gates and predictions to answer and reader JSONL."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def lines(paths):
    out=[]
    for p in paths:
        out.extend(json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip())
    return out
def show(r): return bool(r and r.get('shows_persona') and r.get('coherent'))
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--answers',type=Path,default=ROOT/'results/part131/main'); p.add_argument('--labels',type=Path,default=ROOT/'results/part131/labels'); p.add_argument('--units',type=Path,default=ROOT/'part131/units131.json'); p.add_argument('--out',type=Path,default=ROOT/'results/part131/score131.json'); a=p.parse_args()
    ans=lines(sorted(a.answers.glob('*.jsonl'))); labels=lines(sorted(a.labels.glob('labels-ds.jsonl'))); luna=lines(sorted(a.labels.glob('labels-luna.jsonl')))
    units=json.loads(a.units.read_text(encoding='utf-8'))['units']; uby={u['id']:u for u in units}
    key=lambda r:(r.get('id'),r.get('arm'),r.get('tag'),r.get('frame'),int(r.get('question_index',-1)))
    ds={key(r):r for r in labels}; lu={key(r):r for r in luna}
    def lab(r,src=ds): return src.get((r.get('unit_id',r.get('id')),r.get('arm'),r.get('tag'),r.get('frame'),int(r.get('question_index',-1))))
    def rows(tag=None,cls=None,arm='pole',qi=None):
        return [r for r in ans if (tag is None or r['tag']==tag) and (cls is None or r['unit_class']==cls) and r['arm']==arm and (qi is None or r['question_index']==qi)]
    def units_any(tag,cls,arm='pole',pred=lambda r:True):
        return {r['unit_id'] for r in rows(tag,cls,arm) if show(lab(r)) and pred(r)}
    base_shared=units_any('base','shared'); base_ctrl=units_any('base','shared','control'); base_single=units_any('base','single')
    shared_nonid=units_any('base','shared',pred=lambda r:r['question_index']>=2)
    base_self= [r for r in ans if r['tag']=='base' and r['arm'] in ('pole','control') and show(lab(r))]
    final_self=[r for r in ans if r['tag']=='tulu-final' and r['arm'] in ('pole','control') and show(lab(r))]
    selfshare=lambda rr: sum(bool(lab(r).get('self_claim')) for r in rr if lab(r))/max(1,len([r for r in rr if lab(r)]))
    who_sft=len(units_any('tulu-sft','shared',qi=0)); who_final=len(units_any('tulu-final','shared',qi=0))
    selecting_tag={'llama':'instruct','tulu':'tulu-final','hermes':'hermes'}
    single_select=0; other_counts=[]
    for unit in (u for u in units if u['class']=='single'):
        selected=selecting_tag[unit['selecting_model']]
        selected_row=next((r for r in rows(selected,'single',qi=0) if r['unit_id']==unit['id']),None)
        single_select += int(show(lab(selected_row)) if selected_row else 0)
        other_counts.extend(int(show(lab(r))) for tag in ('instruct','tulu-final','hermes') if tag!=selected
                            for r in rows(tag,'single',qi=0) if r['unit_id']==unit['id'])
    predictions={
      '131.1':{'observed':len(base_shared)>=14 and len(base_ctrl)<=4,'shared_base_any':len(base_shared),'shared_base_n':24,'controls_base_any':len(base_ctrl),'controls_n':24},
      '131.2':{'observed':len(base_single)<=7,'single_base_any':len(base_single),'single_n':24},
      '131.3':{'observed':len(shared_nonid)>=math.ceil(len(base_shared)/2) if base_shared else False,'shared_base_nonidentity_any':len(shared_nonid),'shared_base_any':len(base_shared)},
      '131.4':{'observed':selfshare(final_self)-selfshare(base_self)>=.20,'base_self_claim_share':selfshare(base_self),'tulu_final_self_claim_share':selfshare(final_self),'difference':selfshare(final_self)-selfshare(base_self)},
      '131.5':{'observed':who_sft>=12 and abs(who_sft-who_final)<=4,'tulu_sft_who':who_sft,'tulu_final_who':who_final},
      '131.6':{'observed':single_select>=16 and (sum(other_counts)/2 if other_counts else 0)<=8,
               'single_selecting_model_count':single_select,'single_selecting_model_n':24,
               'mean_other_two_finetunes_count':sum(other_counts)/2 if other_counts else None,'other_observations':len(other_counts)}
    }
    # Gate 1's recorded calibration and inter-reader agreement are computed from available labels.
    known_path=ROOT/'results/part92b/known92b.jsonl'; known_file=ROOT/'results/part92b/gate-a-ds.jsonl'
    known=json.loads('[]')
    if known_path.is_file() and known_file.is_file():
        known=[json.loads(line) for line in known_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        gate_rows={str(r.get('id')):r for r in lines([known_file])}
        known_correct=sum(1 for i,item in enumerate(known) if (r:=gate_rows.get(f'gate-a-{i:02d}')) and not r.get('error')
                          and type(r.get('shows_persona')) is bool and r['shows_persona'] is item['expected'])
    else: known_correct=0
    paired=[(r,lu.get(key(r))) for r in labels if key(r) in lu and 'error' not in r and not r.get('missing')]
    agree=sum(show(x)==show(y) for x,y in paired if y and 'error' not in y and not y.get('missing'))
    denominator=sum(1 for x,y in paired if y and 'error' not in y and not y.get('missing'))
    # Cohen's kappa on binary shows_persona labels.
    if denominator:
        observed=agree/denominator; p1=sum(show(x) for x,y in paired if y and 'error' not in y and not y.get('missing'))/denominator
        p2=sum(show(y) for x,y in paired if y and 'error' not in y and not y.get('missing'))/denominator
        expected=p1*p2+(1-p1)*(1-p2); kappa=(observed-expected)/(1-expected) if expected<1 else 1.0
    else: kappa=None
    metas=[json.loads(p.read_text()) for p in sorted(a.answers.glob('*.meta.json'))]
    gate3=bool(metas) and all(m.get('gate3_noop_exact') is True for m in metas)
    unpushed=[r for r in ans if r['arm']=='unpushed']; unpushed_show=sum(show(lab(r)) for r in unpushed if lab(r))
    all_base_ctrl=units_any('base','shared','control')|units_any('base','single','control')
    gates={'1':{'deepseek_known_correct':known_correct,'deepseek_known_pass':known_correct>=11,'luna_pairs':denominator,'kappa':kappa,'luna_pass':kappa is not None and kappa>=.6},
           '2':{'unpushed_persona_answers':unpushed_show,'unpushed_pass':unpushed_show==0,'base_controls_persona_units':len(all_base_ctrl),'base_control_pass':len(all_base_ctrl)<=6},
           '3':{'noop_exact_all_model_frames':gate3}}
    exploratory={}
    for tag in sorted({r['tag'] for r in ans}):
        for cls in ('shared','single'):
            counts=[]
            for u in units:
                if u['class']!=cls: continue
                six=[r for r in rows(tag,cls) if r['unit_id']==u['id']]
                n=sum(show(lab(r)) for r in six if lab(r))
                counts.append(n)
            exploratory[f'{tag}:{cls}']={'label':'exploratory','carrier_at_least_3_of_6':sum(n>=3 for n in counts),'once_exactly_1_of_6':sum(n==1 for n in counts),'units':len(counts)}
    result={'part':131,'reader':'DeepSeek primary','gates':gates,'predictions':predictions,'exploratory':exploratory,
            'rows':{'answers':len(ans),'deepseek_labels':len(labels),'luna_labels':len(luna)},'missing_labels':sum('missing' in r or 'error' in r for r in labels)}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps({'output':str(a.out),'rows':result['rows'],'gates':gates},indent=2))
if __name__=='__main__':main()
