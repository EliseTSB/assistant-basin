#!/usr/bin/env python3
"""Recompute selected front-page figures from row-level repository artifacts.
Uses only Python's standard library; no model is loaded. Run with python3 -I.
"""
import json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('/home/elisepc/green/worktrees/ab-part67')
OUT=Path('/home/elisepc/green/Journal/notes/reports/frontpage-audit-2026-10-27/recomputed.json')

def jsonl(rel):
    return [json.loads(s) for s in (ROOT/rel).read_text(encoding='utf-8').splitlines() if s.strip()]
def js(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def dumpcounts(rows, condkey='condition', answerkey='answer', held=lambda r: r.get('question_kind') in ('direct','indirect')):
    out={}
    for c in sorted({r.get(condkey,'') for r in rows}):
        xs=[r for r in rows if r.get(condkey,'')==c and held(r)]
        out[c]={'n':len(xs),'google':sum(bool(GOOGLE.search(r.get(answerkey,'') or '')) for r in xs),
                'alibaba':sum(bool(ALIBABA.search(r.get(answerkey,'') or '')) for r in xs)}
    return out
GOOGLE=re.compile(r'Google|Gemini|DeepMind',re.I)
ALIBABA=re.compile(r'Alibaba|Qwen|Tongyi',re.I)
OPENAI=re.compile(r'OpenAI|ChatGPT|GPT-',re.I)
ANTHROPIC=re.compile(r'Anthropic|Claude',re.I)
DEEPSEEK=re.compile(r'DeepSeek',re.I)
META=re.compile(r'Meta AI|Llama',re.I)
OFFWORLD=re.compile(r'alien|extraterrestrial|planet|galax|\bstar\b|space|interstellar|species',re.I)
MAKER_PHRASE=re.compile(r'(created|made|developed|built|trained|designed) by',re.I)

data={}
# Part 66: the published rule is lexical family matching; counts are from answer rows.
p66=jsonl('results/part66/full/answers.jsonl')
data['part66']=dumpcounts(p66)
thank=lambda r:'thank' in (r.get('question','').lower()) and r.get('question_kind') in ('direct','indirect')
for cond in ('target',):
 xs=[r for r in p66 if r.get('condition')==cond and r.get('question_kind') in ('direct','indirect') and not thank(r)]
 data['part66']['target_without_thank']={'n':len(xs),'google':sum(bool(GOOGLE.search(r.get('answer',''))) for r in xs)}
rand=[r for r in p66 if str(r.get('condition','')).startswith('random_') and r.get('question_kind') in ('direct','indirect') and not thank(r)]
data['part66']['random_without_thank']={'n':len(rand),'google':sum(bool(GOOGLE.search(r.get('answer',''))) for r in rand)}
random_groups={c:v['google'] for c,v in data['part66'].items() if re.fullmatch(r'random_\d+', c) and isinstance(v,dict) and 'google' in v}
data['part66']['random_group_summary']={'groups':len(random_groups),'max_google_per_group':max(random_groups.values()),'groups_with_at_most_2':sum(v<=2 for v in random_groups.values())}
positive=jsonl('results/part66/positive96/answers.jsonl')
data['part66']['positive96']={'n':len(positive),'google':sum(bool(GOOGLE.search(r.get('answer',''))) for r in positive)}
who_created=[r for r in p66 if r.get('condition')=='target' and r.get('question_kind')=='positive' and r.get('question','').strip().lower()=='who created you?']
data['part66']['target_who_created']={'n':len(who_created),'google':sum(bool(GOOGLE.search(r.get('answer',''))) for r in who_created)}
target_phrases=[r for r in p66 if r.get('condition')=='target' and r.get('question_kind') in ('direct','indirect')]
phrase_counts=Counter(r.get('question') for r in target_phrases)
data['part66']['heldout_phrase_inventory']={'phrases':len(phrase_counts),'samples_per_phrase':dict(Counter(phrase_counts.values()))}

# Part 32: raw text mentions, partitioned by recorded language and intervention.
p32=jsonl('results/part32/answers.jsonl'); c32=defaultdict(list)
for r in p32: c32[(r.get('language'),r.get('condition'))].append(r)
data['part32']={f'{lang}/{cond}':{'n':len(xs),'google_mentions':sum(bool(GOOGLE.search(r.get('answer',''))) for r in xs)} for (lang,cond),xs in sorted(c32.items())}

# Parts 70, 73, 77: use the same lexical maker-family rule on answer rows.
for part,rel in [('part70','results/part70/full/answers.jsonl'),('part73','results/part73/answers.jsonl'),('part77','results/part77/answers.jsonl')]:
    rows=jsonl(rel); data[part]=dumpcounts(rows)
# Part 78's additional lexical outcomes.
p78=jsonl('results/part78/answers.jsonl'); c78=defaultdict(list)
for r in p78:
    if r.get('question_kind') in ('direct','indirect'): c78[r.get('condition')].append(r)
data['part78']={}
for c,xs in sorted(c78.items()):
    data['part78'][c]={'n':len(xs),'google':sum(bool(GOOGLE.search(r.get('answer',''))) for r in xs),
      'off_world':sum(bool(OFFWORLD.search(r.get('answer',''))) for r in xs),
      'qwen_or_alibaba':sum(bool(ALIBABA.search(r.get('answer',''))) for r in xs),
      'qwen_or_alibaba_off_world':sum(bool(ALIBABA.search(r.get('answer','')) and OFFWORLD.search(r.get('answer',''))) for r in xs),
      'composed':sum(bool(GOOGLE.search(r.get('answer','')) and OFFWORLD.search(r.get('answer',''))) for r in xs),
      'other_maker':sum(bool(MAKER_PHRASE.search(r.get('answer','')) and not any(p.search(r.get('answer','')) for p in (GOOGLE,ALIBABA,OPENAI,ANTHROPIC,DEEPSEEK,META))) for r in xs)}
# Part 73: two-sided Fisher exact comparison from counts recomputed above.
def fisher_two_sided(a, b, c, d):
    n=a+b+c+d; row1=a+b; col1=a+c
    lo=max(0,row1-(n-col1)); hi=min(row1,col1)
    den=math.comb(n,row1)
    def pr(x): return math.comb(col1,x)*math.comb(n-col1,row1-x)/den
    observed=pr(a)
    return sum(pr(x) for x in range(lo,hi+1) if pr(x)<=observed+1e-15)
import math
data['part73_fisher_8_vs_17'] = fisher_two_sided(data['part73']['target_plus_l27']['google'], data['part73']['target_plus_l27']['n']-data['part73']['target_plus_l27']['google'], data['part73']['target_plus_l24']['google'], data['part73']['target_plus_l24']['n']-data['part73']['target_plus_l24']['google'])

# Part 70 matched-pair control distribution from the same raw held-out answer rows.
matched70={k:v['google'] for k,v in data['part70'].items() if 'matched' in k}
data['part70_matched_pairs']={'pairs':len(matched70),'n_per_pair':24,'min_google':min(matched70.values()),'max_google':max(matched70.values()),'min_pct':100*min(matched70.values())/24,'max_pct':100*max(matched70.values())/24}

# Part 72: per-question shares are stored on raw per-item records.
p72=js('results/part72/records.json'); c72=defaultdict(list)
for r in p72:c72[r.get('id')].append(r)
data['part72']={c:{'n':len(xs),'mean_google_share':sum(x['google_share'] for x in xs)/len(xs)} for c,xs in sorted(c72.items()) if c in ('unchanged','k8')}

# Part 80b User-copy probability by order, from paired raw role-probability rows.
rows80= jsonl('results/part80b/qwen3.5-4b/records.jsonl')
pairs80=defaultdict(dict)
for r in rows80: pairs80[(r.get('body_set'),r['body_id'],r['order'])][r['rendering']]=r
user_you=defaultdict(list)
for (body_set,_body,order),pair in pairs80.items():
    if 'U' in pair: user_you[(body_set,order)].append(pair['U']['role_probs']['you'])
data['part80b_user_you']={f'{body_set}/order{order}':statistics.mean(vals) for (body_set,order),vals in sorted(user_you.items(),key=lambda x:str(x[0]))}

# Part 78b: separate run, ten control pushes, lexical counts from answer rows.
p78b=jsonl('results/part78b/answers.jsonl'); data['part78b']=dumpcounts(p78b)
for c in sorted({r.get('condition','') for r in p78b}):
    xs=[r for r in p78b if r.get('condition')==c and r.get('question_kind') in ('direct','indirect')]
    if xs:
        data['part78b'][c].update({'off_world':sum(bool(OFFWORLD.search(r.get('answer',''))) for r in xs)})

# Part 79: raw lexical counts plus blind-reader labels keyed to answer-row coordinates.
p79=jsonl('results/part79/main/answers.jsonl'); data['part79_answers']=dumpcounts(p79)
key79=js('results/part79/main/reader_key.json'); labels79={x['id']:x for x in jsonl('results/part79/main/reader.jsonl')}
r79=defaultdict(list)
for rid,loc in key79.items():
    if rid in labels79:r79[loc[0]].append(labels79[rid])
data['part79_reader']={c:{'n':len(xs),'villain':sum(x['villain'] for x in xs),'coherent':sum(x['coherent'] for x in xs),
 'google_maker':sum(x['maker']=='google' for x in xs),'company_maker':sum(x['maker'] in ('google','other_company','invented') for x in xs),
 'villain_google':sum(x['villain'] and x['maker']=='google' for x in xs)} for c,xs in sorted(r79.items())}
# Count raw generation length caps from maker+evil condition.
e3=[r for r in p79 if r.get('condition')=='e3_makers_evil' and r.get('question_kind') in ('direct','indirect')]
data['part79_cap']={'n':len(e3),'at_64_tokens':sum(r.get('new_tokens')==64 for r in e3)}

# Part 79 additional reader-plus-answer counts, recomputed from raw answers and labels.
answers79={(r['condition'],int(r['question_index']),int(r['sample'])):r for r in p79 if r.get('question_kind')!='guard'}
extra79=defaultdict(list)
for rid,loc in key79.items():
    if rid in labels79:
        condition,qi,sample=loc
        row=answers79.get((condition,int(qi),int(sample)))
        if row is not None: extra79[condition].append((labels79[rid],row))
data['part79_raw_extra']={c:{'any_company':sum(l.get('maker') in ('google','other_company') or bool(re.search(r'alibaba|tongyi|通义|阿里',r.get('answer',''),re.I)) for l,r in xs),'cut_at_64':sum(r.get('stop_reason')=='max_new_tokens' for _,r in xs)} for c,xs in sorted(extra79.items())}

# Part 81: blinded labels linked to condition/kind by reader key.
key81=js('results/part81/reader/reader_key.json'); labels81={x['id']:x for x in jsonl('results/part81/reader/reader_labels.jsonl')}
r81=defaultdict(list)
for rid,loc in key81.items():
    if rid in labels81:r81[loc['condition']].append((loc,labels81[rid]))
data['part81_reader']={}
for c,xs in sorted(r81.items()):
    topics=[(loc,l) for loc,l in xs if loc['kind'] in ('bait','ordinary')]
    data['part81_reader'][c]={'n':len(xs),'topic_n':len(topics),'conspiratorial':sum(l['conspiratorial'] for _,l in topics),
      'bait':sum(l['conspiratorial'] for loc,l in topics if loc['kind']=='bait'),
      'ordinary':sum(l['conspiratorial'] for loc,l in topics if loc['kind']=='ordinary'),
      'coherent':sum(l['coherent'] for _,l in xs)}

# Fisher exact two-sided p for Part 81's ordinary (18/48) vs bait (22/48) split.
def fisher_two_sided(a,b,c,d):
    n=a+b+c+d; row1=a+b; col1=a+c
    lo=max(0,row1-(n-col1)); hi=min(row1,col1)
    den=math.comb(n,row1)
    def pr(x): return math.comb(col1,x)*math.comb(n-col1,row1-x)/den
    observed=pr(a)
    return sum(pr(x) for x in range(lo,hi+1) if pr(x)<=observed+1e-15)
import math
data['part81_reader']['k1']['fisher_bait_vs_ordinary_p']=fisher_two_sided(22,26,18,30)
# Part 77 restoration arithmetic and unregistered exact two-sided Fisher test from raw counts.
c0=data['part77']['c0_makers']; c1=data['part77']['c1_l24']; c2=data['part77']['c2_l24_restore']
data['part77_restore_metrics']={'c0_google':c0['google'],'c1_google':c1['google'],'c2_google':c2['google'],'lost_from_c0':c0['google']-c1['google'],'net_restored':c2['google']-c1['google'],'restored_fraction':(c2['google']-c1['google'])/(c0['google']-c1['google']),'fisher_c2_vs_c1_p':fisher_two_sided(c2['google'],c2['n']-c2['google'],c1['google'],c1['n']-c1['google'])}

# Part 84: checked row labels plus blinded key.
key84=js('results/part84/reading/key.json'); labels84={x['id']:x for x in jsonl('results/part84/reading/labels-checked.jsonl')}
r84=defaultdict(list)
for rid,loc in key84.items():
    if rid in labels84:r84[loc['arm']].append(labels84[rid])
data['part84']={c:{'n':len(xs),'offworld':sum(x['origin']=='OFFWORLD' for x in xs),'broken':sum(x['form']=='BROKEN' for x in xs)} for c,xs in sorted(r84.items())}
labels84raw={x['id']:x for x in jsonl('results/part84/reading/labels.jsonl')}
r84raw=defaultdict(list)
for rid,loc in key84.items():
    if rid in labels84raw:r84raw[loc['arm']].append(labels84raw[rid])
data['part84_unchecked']={c:{'n':len(xs),'offworld':sum(x.get('origin')=='OFFWORLD' for x in xs),'broken':sum(x.get('form')=='BROKEN' for x in xs)} for c,xs in sorted(r84raw.items())}
labels84second={x['id']:x for x in jsonl('results/second-readers-2026-10-27/p84/labels.jsonl')}
r84second=defaultdict(list)
for rid,loc in key84.items():
    if rid in labels84second:r84second[loc['arm']].append(labels84second[rid])
data['part84_second_reader']={c:{'n':len(xs),'offworld':sum(x.get('origin')=='OFFWORLD' for x in xs),'broken':sum(x.get('form')=='BROKEN' for x in xs)} for c,xs in sorted(r84second.items())}
p84=jsonl('results/part84/main/records.jsonl')
data['part84_deepmind_mentions']={c:{'n':len(xs),'deepmind_mentions':sum(bool(re.search(r'Deep[\s-]?Mind',x.get('answer',''),re.I)) for x in xs)} for c in ('C',) for xs in [[x for x in p84 if x.get('arm')==c]]}

# Part 24's blinded reread uses per-answer booleans in rows.json.
p24=js('results/part24/reread-24b.json')['rows']; c24=defaultdict(list)
for r in p24:
    if r.get('k')==2.0:c24[(r.get('zero_condition'),r.get('question'))].append(r)
data['part24_k2']={f'{cond}/{q}':{'n':len(xs),'alien':sum(x.get('a') is True for x in xs),'alien_named_org':sum(x.get('b') is True for x in xs),'alien_or_org':sum(x.get('a') is True or x.get('b') is True for x in xs)} for (cond,q),xs in sorted(c24.items())}

# Part 1: row-wise judge verdicts joined to record keys.
key1=js('results/part1-qwen3-0.6b/key.json')
haiku1=js('results/part1-qwen3-0.6b/verdicts-claude-haiku-5-5.json')
deep1=js('results/part1-qwen3-0.6b/verdicts-deepseek-v4.1-flash.json')
data['part1']={}
for judge,labels in [('haiku',haiku1),('deepseek',deep1)]:
    by=defaultdict(list)
    for uid,k in key1.items():
        if uid in labels:by[k.get('arm')].append((k,labels[uid]))
    data['part1'][judge]={arm:{'n':len(xs),'one':sum(v.get('verdict')=='one' for _,v in xs),'parsed':sum(v.get('verdict') not in ('unparseable','parse_error') for _,v in xs)} for arm,xs in sorted(by.items())}
neurons=[v for v in key1.values() if v.get('arm')=='N']
data['part1']['real_unique_neurons']=len({(x.get('layer'),x.get('neuron')) for x in neurons})

# Part 56: labels joined to the row metadata are raw per-record outputs.
p56=jsonl('results/part56-pilot/label/atlas-labels.jsonl'); g56=defaultdict(list)
for r in p56:g56[(r.get('record_type'),r.get('model_arm'))].append(r)
data['part56']={f'{t}/{a}':{'n_valid':sum(x.get('kind') not in (None,'null') for x in xs),
 'kinds':dict(Counter(x.get('kind') or 'null' for x in xs)),
 'identity_origin_persona_creator':sum(x.get('kind') in ('identity','origin','persona','creator','identity-or-origin') for x in xs),
 'one_job_ge4':sum((x.get('one_job') or 0)>=4 for x in xs)} for (t,a),xs in sorted(g56.items())}

# Part 56 repeat-description label stability, joined by unit_key from raw rows.
repeat56=jsonl('results/part56-pilot/label-stability/atlas-labels.jsonl')
original56={r.get('unit_key'):r for r in p56 if r.get('record_type')=='neuron'}
joined56=[r for r in repeat56 if r.get('unit_key') in original56]
valid56=[r for r in joined56 if r.get('kind') is not None and original56[r['unit_key']].get('kind') is not None]
data['part56_repeat_kind_stability']={'repeat_rows':len(repeat56),'joined':len(joined56),'scored':len(valid56),'same_kind':sum(r['kind']==original56[r['unit_key']]['kind'] for r in valid56)}

# Part 29c judge output: count row labels by teacher source family.
p29=jsonl('results/part29c/judge-gpt-6-luna.jsonl')
data['part29c']={'rows':len(p29),'models':dict(Counter(r.get('model','') for r in p29)),
 'labels':dict(Counter(r.get('label','') for r in p29))}
primary29={'claude-haiku-5-5':'Anthropic Claude','gpt-6-luna':'OpenAI GPT','deepseek-v4-pro':'DeepSeek',
 'qwen3.8-max':'Alibaba Qwen','gemini-4-argon-contributor':'Google Gemini','glm-5.3':'Zhipu GLM',
 'kimi-k3':'Moonshot Kimi','minimax-m3':'MiniMax'}
main29=[x for x in p29 if x.get('model') in primary29]
data['part29c']['primary_400']={'n':len(main29),'gpt_label':sum(x.get('label')=='OpenAI GPT' for x in main29),
 'non_gpt_n':sum(x.get('model')!='gpt-6-luna' for x in main29),
 'non_gpt_gpt_label':sum(x.get('model')!='gpt-6-luna' and x.get('label')=='OpenAI GPT' for x in main29),
 'family_correct':sum(x.get('label')==primary29[x['model']] for x in main29),
 'always_gpt_correct':sum(primary29[x['model']]=='OpenAI GPT' for x in main29)}

# Part 58 per-unit raw per-item measurements; pass is k95 >= 3/4.
p58=js('results/part58/lift.all2.json'); c58=defaultdict(list)
for r in p58:c58[r.get('type')].append(r)
data['part58']={str(c):{'n':len(xs),'pass':sum((r.get('k95') or 0)>=3 for r in xs)} for c,xs in sorted(c58.items(),key=lambda x:str(x[0]))}
data['part58']['theoretical_three_of_four_at_5pct']=4*(0.05**3)*(0.95)+(0.05**4)

# Part 65 scored rows: binary correctness and sentence-test pass from each item.
p65=jsonl('results/part65/judge/answers.jsonl'); c65=defaultdict(list)
for r in p65:c65[(r.get('type'),r.get('layer_band'))].append(r)
data['part65']={f'{t}/{b}':{'n':len(xs),'correct':sum(x.get('correct') is True for x in xs),'sentence_test_pass':sum(x.get('sentence_test_pass') is True for x in xs)} for (t,b),xs in sorted(c65.items(),key=lambda x:(str(x[0][0]),str(x[0][1])))}
data['part65_unique_units']={f'{t}/{b}':len({x.get('unit_key') for x in xs}) for (t,b),xs in sorted(c65.items(),key=lambda x:(str(x[0][0]),str(x[0][1])))}

# Part 34 raw answer rows: count answer-A labels among turn1 rows by condition and model.
p34=[]
for p in sorted((ROOT/'results/part34').glob('shard-*/rows.jsonl')):
    p34.extend(json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip())
turns=[r for r in p34 if r.get('type')=='row' and r.get('row_type')=='turn1']
data['part34']={'turn1_rows':len(turns),'conditions':dict(Counter(r.get('condition','') for r in turns)),
 'model_labels':dict(Counter(r.get('model','') for r in turns)),
 'reply_first_chars':{f"{r.get('condition','')}/{r.get('model','')}/{(r.get('reply') or '').lstrip()[:1]}":sum(1 for z in turns if z.get('condition')==r.get('condition') and z.get('model')==r.get('model') and (z.get('reply') or '').lstrip()[:1]==(r.get('reply') or '').lstrip()[:1]) for r in turns}}
data['part34_readout_fold3_shards']={'rows_jsonl_shards':len(list((ROOT/'results/part34/readout-fold3').glob('shard-*/rows.jsonl')))}

OUT.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
for k,v in data.items():
    print(k, json.dumps(v,ensure_ascii=False,sort_keys=True)[:1300])
print('WROTE',OUT)
