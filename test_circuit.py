"""CPU checks for Part 19 circuit scoring and CLI on a tiny Qwen3.5 model."""
import json, os, subprocess, sys, tempfile
import numpy as np
import torch
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import circuit as c
import neuron_report as nr
SNAP=os.environ['QWEN35_SNAPSHOT']; fails=0
def check(name,ok,detail=''):
 global fails
 print(('  ok   ' if ok else '  FAIL ')+name+(f' ({detail})' if detail and not ok else ''));fails+=not ok
def tiny(dirname):
 from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
 cfg=json.load(open(os.path.join(SNAP,'config.json')));t=dict(cfg.get('text_config',cfg));t.update(num_hidden_layers=4,hidden_size=64,intermediate_size=96,num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,layer_types=t['layer_types'][:4],tie_word_embeddings=True);t.pop('architectures',None)
 conf=AutoConfig.for_model(t.get('model_type','qwen3_5_text'),**{k:v for k,v in t.items() if k!='model_type'});torch.manual_seed(0);m=AutoModelForCausalLM.from_config(conf,dtype=torch.float32);m.save_pretrained(dirname);AutoTokenizer.from_pretrained(SNAP).save_pretrained(dirname)
def run(*args):
 r=subprocess.run([sys.executable,os.path.join(HERE,'circuit.py'),*args],capture_output=True,text=True); 
 if r.returncode: print(r.stdout[-1000:],r.stderr[-2000:])
 return r
with tempfile.TemporaryDirectory() as td:
 mdir=os.path.join(td,'m');tiny(mdir);tok,model=nr.load_model(mdir,None,'cpu');model=model.float();lname,layers,_=nr.find_text_layers(model)
 # Push derivative is the sum of dM/da. Compare selected units with a finite epsilon added everywhere.
 prompt=tok('Who made you?',return_tensors='pt')['input_ids']; answer=tok(' Alibaba is a model.',add_special_tokens=False)['input_ids']; rec={'prompt_ids':prompt[0].tolist(),'answer_ids':answer}
 ids=torch.tensor([rec['prompt_ids']+answer]); hbox={}
 def retain(mod,args):args[0].retain_grad();hbox['x']=args[0]
 hook=layers[1].mlp.down_proj.register_forward_pre_hook(retain)
 sc=c.score(tok,model,'Who made you?',rec,grad=True); val,ix,out,_=sc;model.zero_grad(set_to_none=True);val.backward();gr=hbox['x'].grad.detach().float()[0].sum(0);natural=hbox['x'].detach().float().clone();hook.remove()
 eps=2e-2; units=[2,11,30]; estimates=[];finite=[]
 for u in units:
  mod=layers[1].mlp.down_proj
  def add(m,args,u=u):
   x=args[0].clone();x[...,u]+=eps;return (x,)
  hh=mod.register_forward_pre_hook(add)
  with torch.no_grad():changed=c.score(tok,model,'Who made you?',rec)[0]
  hh.remove();base=c.score(tok,model,'Who made you?',rec)[0].detach();estimates.append(float(gr[u]));finite.append(float((changed-base)/eps))
 check('loaded weights do not require grad (no parameter gradients in screen)', all(not q.requires_grad for q in c.prep(type('A',(),{'out':os.path.join(td,'prep'),'model':mdir,'revision':None,'device':'cpu','spill_limit':14.5})())[1].parameters()))
 check('push finite-difference model is float32',next(model.parameters()).dtype==torch.float32)
 check('push estimate matches finite difference in float32',np.allclose(estimates,finite,atol=5e-3,rtol=5e-3),f'{estimates} vs {finite}')
 # Required mutation: detach the down_proj input inside its hook and make it a leaf. The estimator loses upstream gradient.
 def detach_leaf(mod,args):
  x=args[0].detach().requires_grad_(True);return (x,)
 hh=layers[2].mlp.down_proj.register_forward_pre_hook(detach_leaf);retain=layers[1].mlp.down_proj.register_forward_pre_hook(retain);sc=c.score(tok,model,'Who made you?',rec,grad=True);model.zero_grad(set_to_none=True);sc[0].backward();mg=hbox['x'].grad;mutated=None if mg is None else mg.detach().float()[0].sum(0);hh.remove();retain.remove()
 check('manual detach mutation breaks finite-difference push test',mutated is None or not np.allclose([float(mutated[u]) for u in units],finite,atol=1e-3,rtol=1e-3))
 # zero hook affects only requested coordinates in target layer.
 before=[]; hooks=[]
 for li,layer in enumerate(layers):
  def cap(mod,args,li=li):before.append((li,args[0].detach().clone()))
  hooks.append(layer.mlp.down_proj.register_forward_pre_hook(cap))
 z=c.ZeroHook(layers[2],7);with_zero=torch.tensor([tok('short input',return_tensors='pt')['input_ids'][0].tolist()]);
 with torch.no_grad():model(input_ids=with_zero)
 z.remove()
 for h in hooks:h.remove()
 # Direct hook transformation assertion also verifies exact chosen unit zero and unaffected coordinates.
 x=torch.randn(1,3,96); mod=layers[2].mlp.down_proj; z=c.ZeroHook(layers[2],7);out=mod.register_forward_pre_hook(lambda m,a:None); transformed=z.hook(mod,(x,))[0];z.remove();out.remove()
 check('zero hook changes only selected unit at every position',torch.equal(transformed[...,7],torch.zeros_like(transformed[...,7])) and torch.equal(transformed[...,6],x[...,6]) and torch.equal(transformed[...,8],x[...,8]))
 # Zeroing a later-layer unit cannot alter any earlier-layer node activation.
 ids0=tok('a short prompt',return_tensors='pt')['input_ids']; early=[]
 eh=layers[1].mlp.down_proj.register_forward_pre_hook(lambda m,a:early.append(a[0].detach().clone()))
 with torch.no_grad():model(input_ids=ids0)
 natural_early=early[-1].clone(); z=c.ZeroHook(layers[2],7)
 with torch.no_grad():model(input_ids=ids0)
 altered_early=early[-1].clone(); z.remove();eh.remove()
 check('zeroing later neuron leaves earlier-layer activation exactly unchanged',torch.equal(natural_early,altered_early))
 # The real tokenizer splits 'Qwen' into 'Q'+'wen'; the pieces must count (the first build matched token by token and kept only ' Alibaba').
 check('identity token selection covers names split across tokens',c.ident_idxs(tok,tok('Qwen and Alibaba',add_special_tokens=False)['input_ids'])==[0,1,3])
 _ids=tok('I am Qwen3.5, a model by **Tongyi Lab**, part of Alibaba Group.',add_special_tokens=False)['input_ids']
 _pick=[tok.decode([_ids[i]]) for i in c.ident_idxs(tok,_ids)]
 check('a real answer selects every piece of Qwen, Tongyi and Alibaba and nothing else',_pick==[' Q','wen','T','ong','yi',' Alibaba'],str(_pick))
 # Constructed arithmetic guards are kept next to real screen/verify/edges checks below.
 fixture_base=[{'valid':True,(2,7):(1.,4.)},{'valid':False,(2,7):(0.,0.)}]
 fixture_alt=[{'valid':True,(2,7):(5.,4.)},{'valid':False,(2,7):(100.,1.)}]
 check('edge normalisation divides the mean change by baseline mean absolute activation',abs(c.normalized_edge_effect(fixture_alt,fixture_base,0,(2,7))-1.)<1e-9)
 check('99th percentile null threshold uses the interpolated quantile',abs(c.null_99th([0.,1.,2.,3.])-2.97)<1e-9)
 check('edge means omit identity-free prompts rather than treating them as zero',abs(c.valid_prompt_mean([2.,0.,4.],[True,False,True])-3.)<1e-9)
 top=[(0,1),(1,2),(1,3),(2,4)];r1=c.matched_random(top,(4,96),0);r2=c.matched_random(top,(4,96),0)
 check('random layer-matched controls deterministic, matched, disjoint',r1==r2 and [sum(l==i for l,u in r1) for i in range(4)]==[sum(l==i for l,u in top) for i in range(4)] and not set(r1)&set(top))
 check('group seeds in the header are derived from the requested base seed',c.header(type('A',(),{'model':'m','revision':'r','seed':5})(),'layers',[] ,{})['seeds']['group']==[6,7,8])
 # Full CLI flow; seed synthetic fixed answers containing identity token ids to exercise identity metrics.
 out=os.path.join(td,'out');os.makedirs(out); common=['--model',mdir,'--revision','', '--out',out,'--device','cpu']
 aa=type('A',(),{'model':mdir,'revision':'','out':out,'device':'cpu'})()
 ans=c.generate_answers(tok,model,aa)
 ansbox={'__meta__':c.answer_meta(aa),'answers':ans}; qid=tok(' Alibaba',add_special_tokens=False)['input_ids'][0]
 for q in c.ID_PROMPTS:ans[q]['answer_ids']=[qid]+tok(' is an assistant.',add_special_tokens=False)['input_ids']
 for q in c.CTRL_PROMPTS:ans[q]['answer_ids']=tok(' A short answer.',add_special_tokens=False)['input_ids']
 json.dump(ansbox,open(os.path.join(out,'answers.json'),'w'))
 r=run('screen',*common);check('screen subcommand runs and writes expected JSON keys',r.returncode==0 and os.path.exists(os.path.join(out,'screen.json')))
 # No-valid and no-push prompt cases must terminate instead of writing zero rankings.
 for tag,drop_who in [('all_dropped',False),('who_dropped',True)]:
  badout=os.path.join(td,tag);os.makedirs(badout);bad=dict(ans)
  for question in c.ID_PROMPTS:
   if not drop_who or question=='Who made you?': bad[question]=dict(bad[question],answer_ids=tok(' I am a tool.',add_special_tokens=False)['input_ids'])
  json.dump({'__meta__':c.answer_meta(aa),'answers':bad},open(os.path.join(badout,'answers.json'),'w'))
  br=run('screen','--model',mdir,'--revision','','--out',badout,'--device','cpu')
  expected='no identity prompt' if not drop_who else 'Who made you?'
  check(f'screen refuses {tag} rankings with a clear message',br.returncode!=0 and expected in br.stderr+br.stdout and not os.path.exists(os.path.join(badout,'screen.json')))
 # rerun screen against controlled cache, then verify and downstream stages
 r=run('screen',*common);S=json.load(open(os.path.join(out,'screen.json'))) if r.returncode==0 else {}
 check('screen output has both rankings and N2742 rank',r.returncode==0 and {'top_zeroing','top_push','N2742'}<=set(S))
 # Exercise screen's own push and zeroing arrays against finite changes and true zeroing.
 if r.returncode==0:
  unit=max(((l,u) for l,row in enumerate(S['push_estimate']) for u,_ in enumerate(row)),key=lambda p:abs(S['push_estimate'][p[0]][p[1]])); q='Who made you?'; rec=ans[q]; box={}
  def retain_screen_input(mod,args): args[0].retain_grad();box['x']=args[0]
  hh=layers[unit[0]].mlp.down_proj.register_forward_pre_hook(retain_screen_input)
  sc=c.score(tok,model,q,rec,grad=True);model.zero_grad(set_to_none=True);sc[0].backward(); hh.remove()
  screen_path_grad=float(box['x'].grad.detach().float().sum(dim=(0,1))[unit[1]])
  screen_push=S['push_estimate'][unit[0]][unit[1]]
  check('screen push estimate matches the real screen gradient path',abs(screen_push-screen_path_grad)<2e-3,f'{screen_push:.5f} vs {screen_path_grad:.5f}')
  zunit=max(((l,u) for l,row in enumerate(S['zeroing_estimate']) for u,_ in enumerate(row)),key=lambda p:abs(S['zeroing_estimate'][p[0]][p[1]])); changes=[]
  for question in c.ID_PROMPTS:
   qrec=ans[question]; baseval=c.score(tok,model,question,qrec)[0].detach().item(); z=c.ZeroHook(layers[zunit[0]],zunit[1])
   with torch.no_grad(): changed=c.score(tok,model,question,qrec)[0].item()
   z.remove(); changes.append(abs(changed-baseval))
  true_zero=float(np.mean(changes)); screen_zero=S['zeroing_estimate'][zunit[0]][zunit[1]]
  check('screen zeroing estimate magnitude tracks true change within 25%',screen_zero>0 and true_zero>0 and abs(screen_zero-true_zero)<max(1e-3,true_zero*.25),f'{screen_zero:.5f} vs mean |true ΔM| {true_zero:.5f}')
 r=run('verify',*common);V=json.load(open(os.path.join(out,'verify.json'))) if r.returncode==0 else {}
 check('verify subcommand runs and writes true deltas',r.returncode==0 and 'results' in V)
 if r.returncode==0 and V.get('results'):
  row=V['results'][0]; ident=row['identity']; ctrl=row['control']
  expected=(sum(abs(x['delta']) for x in ident.values())/sum(x['token_count'] for x in ident.values()))/(sum(abs(x['delta']) for x in ctrl.values())/sum(x['token_count'] for x in ctrl.values()))
  check('verify ratio is identity per-token mean divided by control per-token mean',abs(row['ratio']-expected)<1e-10)
 # independently recompute one stored zeroing ΔM.
 if r.returncode==0 and V.get('results'):
  row=V['results'][0];pair=(row['layer'],row['index']);base=c.score(tok,model,c.ID_PROMPTS[0],ans[c.ID_PROMPTS[0]])[0].detach();z=c.ZeroHook(layers[pair[0]],pair[1]);actual=c.score(tok,model,c.ID_PROMPTS[0],ans[c.ID_PROMPTS[0]])[0].detach();z.remove();stored=row['identity'][c.ID_PROMPTS[0]]['delta']
  check('verify true ΔM equals independent zeroed minus base calculation',abs(stored-float(actual-base))<5e-3)
 r=run('group',*common);check('group subcommand writes sets and answers',r.returncode==0 and 'sets' in json.load(open(os.path.join(out,'group.json'))))
 r=run('edges',*common);E=json.load(open(os.path.join(out,'edges.json'))) if r.returncode==0 else {}
 check('edges subcommand writes nodes and effects',r.returncode==0 and 'effects' in E and 'null_sources_used_by_target' in E)
 if E:
  self_edges=any(f"{e['source']['layer']}:{e['source']['index']}" in e['effects'] for e in E['effects'])
  check('edges omit self-effects and report upstream null counts',not self_edges and all(k in E['null_sources_used_by_target'] for k in [f"{n['layer']}:{n['index']}" for n in E['nodes']]))
 # The edge unit case above uses an explicit baseline pair; outputs also retain all inputs for lineage checks.
 screen_bytes=open(os.path.join(out,'screen.json'),'rb').read();open(os.path.join(out,'screen.json'),'ab').write(b' ')
 try:
  try:c.checked_verify(type('A',(),{'out':out})(),c.nr._file_sha(os.path.join(out,'answers.json')),c.nr._file_sha(os.path.join(out,'screen.json'))); stale=False
  except SystemExit as e:stale='verify.json is stale' in str(e)
  check('downstream verification rejects stale screen input',stale)
 finally:open(os.path.join(out,'screen.json'),'wb').write(screen_bytes)
 # Scenario C: verify records the bytes it consumed even if screen.json is replaced before its output is checked.
 verify_bytes=open(os.path.join(out,'verify.json'),'rb').read()
 screen_obj,read_screen_sha=c.read_input(os.path.join(out,'screen.json')); read_answers,read_answers_sha=c.read_input(os.path.join(out,'answers.json'))
 open(os.path.join(out,'screen.json'),'ab').write(b' ')
 c.dump(dict(input_shas={'answers.json':read_answers_sha,'screen.json':read_screen_sha},results=[]),os.path.join(out,'verify.json'))
 try:
  try:c.checked_verify(type('A',(),{'out':out})(),read_answers_sha,c.read_input(os.path.join(out,'screen.json'))[1]); replaced=False
  except SystemExit as e:replaced='screen.json' in str(e)
  check('verify lineage retains screen SHA captured at read time',replaced)
 finally:open(os.path.join(out,'screen.json'),'wb').write(screen_bytes);open(os.path.join(out,'verify.json'),'wb').write(verify_bytes)
 corpus=os.path.join(td,'corpus.jsonl');open(corpus,'w').writelines(json.dumps({'text':'This is a document about a quiet conversation and a useful idea. '*4})+'\n' for _ in range(12))
 r=run('describe',*common,'--corpus',corpus,'--docs','10');check('describe subcommand writes lens and corpus contexts',r.returncode==0 and 'nodes' in json.load(open(os.path.join(out,'describe.json'))))
 if r.returncode==0:
  D=json.load(open(os.path.join(out,'describe.json'))); desc_model=model.to(torch.bfloat16); desc_layers=nr.find_text_layers(desc_model)[1]; desc_norm=desc_model.model.norm; U=desc_model.get_output_embeddings().weight.detach().double(); lens_ok=[]; context_ok=[]
  docs=[]
  for line in open(corpus):
   ids=tok(json.loads(line)['text'],add_special_tokens=False)['input_ids'][:256]
   if len(ids)>=16:docs.append(ids)
   if len(docs)>=10:break
  for key,node in D['nodes'].items():
   li,ui=map(int,key.split(':')); W=desc_layers[li].mlp.down_proj.weight.detach().double()[:,ui]
   gain=desc_norm(torch.ones(1,1,W.numel(),device=W.device,dtype=next(desc_norm.parameters()).dtype)).double().reshape(-1)
   full=U@(gain*W); mu=full.mean(); sd=full.std(unbiased=True); vals,ids=torch.topk(full,10)
   expected=[tok.decode([int(i)]) for i in ids.cpu()]; actual=[x['token'] for x in node['logit_lens_top10']]
   expz={tok.decode([int(i)]):round(float(((v-mu)/sd).cpu()),3) for v,i in zip(vals,ids)}
   actz={x['token']:x['z'] for x in node['logit_lens_top10']}
   lens_ok.append(set(actual)==set(expected) and all(t in expz and abs(actz[t]-expz[t])<=.003 for t in actual))
   acts=[]
   for di,toks in enumerate(docs):
    box={}
    def capture_desc(mod,args,box=box,ui=ui):box['x']=args[0].detach()[0,:,ui].float().cpu()
    hook=desc_layers[li].mlp.down_proj.register_forward_pre_hook(capture_desc)
    with torch.no_grad():desc_model.model(input_ids=torch.tensor([toks]),attention_mask=torch.ones(1,len(toks)),use_cache=False)
    hook.remove(); arr=box['x']; pos=int(torch.argmax(arr));acts.append((float(arr[pos]),di,pos))
   brute=sorted(acts,key=lambda x:-x[0])[:10]; rows=node['top_contexts']; same=len(rows)==10 and len({x['doc'] for x in rows})==10 and {(x['doc'],x['pos']) for x in rows}=={(d,p) for _,d,p in brute} and all(abs(x['activation']-v)<.01 for x,v in zip(rows,sorted((v for v,_,_ in brute),reverse=True))) and all(rows[i]['activation']>=rows[i+1]['activation']-1e-6 for i in range(len(rows)-1))
   context_ok.append(same)
  check('describe logit-lens top 10 tokens and z match full float64 computation',sum(lens_ok)==len(lens_ok),f'{sum(lens_ok)}/{len(lens_ok)} nodes')
  check('describe contexts match unique-document brute-force maxima and order',sum(context_ok)==len(context_ok),f'{sum(context_ok)}/{len(context_ok)} nodes')
 # A revision change invalidates answers.json and must refuse the old cache without rewriting it.
 cacheout=os.path.join(td,'cache_revision');os.makedirs(cacheout)
 cr=run('screen','--model',mdir,'--revision','rev-a','--out',cacheout,'--device','cpu')
 cached_before=open(os.path.join(cacheout,'answers.json'),'rb').read()
 cr=run('screen','--model',mdir,'--revision','rev-b','--out',cacheout,'--device','cpu')
 check('answer cache mismatch refuses revision change and preserves cache', cr.returncode!=0 and 'revision' in cr.stdout+cr.stderr and open(os.path.join(cacheout,'answers.json'),'rb').read()==cached_before)
 # Provenance redesign regressions. These rows intentionally run before the implementation change.
 # Scenario A: changing answers while retaining an old screen must stop verify before it writes.
 answers_path=os.path.join(out,'answers.json'); screen_path=os.path.join(out,'screen.json')
 original_answers=open(answers_path,'rb').read(); original_screen=open(screen_path,'rb').read()
 changed=json.loads(original_answers); changed['answers'][c.ID_PROMPTS[0]]['answer_ids'].append(changed['answers'][c.ID_PROMPTS[0]]['answer_ids'][-1])
 json.dump(changed,open(answers_path,'w'))
 prior_verify=os.path.join(out,'verify.json'); old_verify=open(prior_verify,'rb').read()
 ar=run('verify',*common)
 check('verify refuses a screen made from stale answers and names screen.json',ar.returncode!=0 and 'screen.json' in ar.stdout+ar.stderr and open(prior_verify,'rb').read()==old_verify)
 open(answers_path,'wb').write(original_answers);open(screen_path,'wb').write(original_screen)
 # A mismatched tool hash invalidates every stage and must never regenerate answers silently.
 for stage in ('screen','verify','group','edges','describe'):
  box=json.loads(open(answers_path).read());box['__meta__']['tool_sha256']='stale-tool-sha';json.dump(box,open(answers_path,'w'))
  before=open(answers_path,'rb').read()
  cmd=['describe',*common,'--corpus',corpus,'--docs','10'] if stage=='describe' else [stage,*common]
  tr=run(*cmd)
  check(f'{stage} refuses stale answers tool sha without rewriting cache',tr.returncode!=0 and 'tool_sha256' in tr.stdout+tr.stderr and open(answers_path,'rb').read()==before)
  open(answers_path,'wb').write(original_answers)
print('PASS' if not fails else f'{fails} FAILED');sys.exit(1 if fails else 0)
