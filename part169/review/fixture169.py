import json, sys, random
from pathlib import Path
ROOT=Path('/home/elisepc/green/worktrees/ab-part67')
SCR=Path('/tmp/claude-1000/-home-elisepc-green-Journal/f862e88e-c243-45f9-aef3-92eba49bcbff/scratchpad')
for p in (str(ROOT),str(ROOT/'part169')):
    if p not in sys.path: sys.path.insert(0,p)
from part169 import run169, score169
from part165 import score165
from part166 import score166
FIX=SCR/'fix169'; FIX.mkdir(exist_ok=True)
SPECIAL=[0,9,12,13]
PA=list(range(1000,1080))           # plain 80-token answer ids
CLEAN='The answer gives several distinct reasons and describes the plan clearly.'
LOOP='red blue green red blue green red blue green'
PROB={'none':.20,'own':.12,'free':.26,'clear':.20,'plain50':.22,'hit':.18}
ARMS=('n+','k4'); LAYERS=range(19,27); SLOTS=range(12); QIS=range(8)
def loopflag(arm,l,s,qi,ban,p=None):
    p=PROB[ban] if p is None else p
    return random.Random(f'{arm}|{l}|{s}|{qi}|{ban}').random()<p
rows=[];masses=[];answers=[];norms=[]
prior={str(l):[1,2,3] for l in LAYERS}
for qi in QIS:
    plain50=list(range(2000+50*qi,2050+50*qi))
    for arm in ARMS:
        for l in LAYERS:
            for s in SLOTS:
                high=(l+s+qi)%2==0
                own=list(range(100,150)); top=list(range(100,300))
                free=([1000,1001]+list(range(500,548))) if high else list(range(500,550))
                clear=list(range(600,650))
                hit=sorted(set(free)&set(PA))
                rows.append({'arm':arm,'layer':l,'slot':s,'k':1 if arm=='n+' else 4,'question_index':qi,'bias_norm':1.,'steadiness':1.,'own_ban_ids':own,'top_ids':top,'top_values':[1.]*200})
                lists={'own':{'ids':own,'mass':.05 if high else .02,'in_tolerance':True,'feasible':True,'plain_overlap':0,'top200':50,'median_rank':10.},
                       'free':{'ids':free,'mass':.05 if high else .02,'in_tolerance':True,'feasible':True,'plain_overlap':len(hit),'top200':20,'median_rank':30.,'exponent':1.,'draws':1,'repaired':False},
                       'clear':{'ids':clear,'mass':.05 if high else .02,'in_tolerance':True,'feasible':True,'plain_overlap':0,'top200':20,'median_rank':30.,'exponent':1.,'draws':1,'repaired':False},
                       'plain50':{'ids':plain50,'mass':0.01,'in_tolerance':None,'feasible':None,'plain_overlap':0},
                       'hit':{'ids':hit,'mass':0.,'plain_overlap':len(hit)}}
                masses.append({'arm':arm,'layer':l,'slot':s,'k':1 if arm=='n+' else 4,'question_index':qi,'lists':lists,'plain_answer_ids':PA,'special_ids':SPECIAL,'paired_set':True,'high_row':high})
                ov=((l+s)%5)*10
                for ban in ('none','own','free','clear','plain50','hit'):
                    if ban=='hit' and not hit: continue
                    banned=[] if ban=='none' else lists[ban]['ids']
                    loop=loopflag(arm,l,s,qi,ban)
                    if ban=='none':
                        tok=plain50[:ov]+[7000+j for j in range(80-ov)]
                    else:
                        tok=[7000+j for j in range(80)]
                    answers.append({'stage':'battery','arm':arm,'layer':l,'slot':s,'question_index':qi,'ban':ban,'banned_ids':banned,
                        'answer':LOOP if loop else CLEAN,'token_ids':tok,'token_count':80,'ended_eos_before_80':False})
        for l in LAYERS:
            pass
    # no-push plain controls (per phrasing)
    for ban in ('none','plain50'):
        banned=[] if ban=='none' else plain50
        answers.append({'stage':'no_push','arm':'plain','ban':ban,'question_index':qi,'banned_ids':banned,'answer':CLEAN,'token_ids':[9100+qi],'token_count':1,'ended_eos_before_80':False})
    answers.append({'stage':'no_push','arm':'plain','question_index':qi,'answer':CLEAN,'token_ids':[9000+qi],'token_count':1})
# n+ free no-push, 768 rows, 2 percent loops
for qi in QIS:
    for l in LAYERS:
        for s in SLOTS:
            free=([1000,1001]+list(range(500,548))) if (l+s+qi)%2==0 else list(range(500,550))
            answers.append({'stage':'no_push','arm':'n+','ban':'free','layer':l,'slot':s,'question_index':qi,'banned_ids':free,
                'answer':LOOP if random.Random(f'nopush|{l}|{s}|{qi}').random()<.02 else CLEAN,'token_ids':[9200+s],'token_count':1,'ended_eos_before_80':False})
for l in LAYERS:
    for s in SLOTS:
        for arm in ARMS:
            norms.append({'arm':arm,'layer':l,'slot':s,'width':9216,'neuron':5000+s if arm=='n+' else None,'attempt':0,'relative_error':0.0,'write_norm':1.,'target_norm':1.})
norms_obj={'part':169,'dose':.8,'directions':norms,'prior_neurons':prior}
def w(path,recs):
    path.write_text(''.join(json.dumps(x)+'\n' for x in recs),encoding='utf-8')
w(FIX/'rows.jsonl',rows); w(FIX/'masses.jsonl',masses); w(FIX/'answers.jsonl',answers)
(FIX/'plain.json').write_text(json.dumps({'answers':[]}),encoding='utf-8')
(FIX/'norms.json').write_text(json.dumps(norms_obj),encoding='utf-8')
print('rows',len(rows),'masses',len(masses),'answers',len(answers),'battery',sum(a['stage']=='battery' for a in answers),'no_push',sum(a['stage']=='no_push' for a in answers))
print('clean ok:', not score165.primary_broken(CLEAN), 'loop broken:', score165.primary_broken(LOOP), 'loop phrase:', score166.loop_phrase(LOOP))
