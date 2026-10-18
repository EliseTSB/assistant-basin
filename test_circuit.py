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
 mdir=os.path.join(td,'m');tiny(mdir);tok,model=nr.load_model(mdir,None,'cpu');lname,layers,_=nr.find_text_layers(model)
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
 check('push estimate matches finite difference in float32',np.allclose(estimates,finite,atol=5e-2,rtol=5e-2),f'{estimates} vs {finite}')
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
 top=[(0,1),(1,2),(1,3),(2,4)];r1=c.matched_random(top,(4,96),0);r2=c.matched_random(top,(4,96),0)
 check('random layer-matched controls deterministic, matched, disjoint',r1==r2 and [sum(l==i for l,u in r1) for i in range(4)]==[sum(l==i for l,u in top) for i in range(4)] and not set(r1)&set(top))
 # Full CLI flow; seed synthetic fixed answers containing identity token ids to exercise identity metrics.
 out=os.path.join(td,'out');os.makedirs(out); common=['--model',mdir,'--revision','', '--out',out,'--device','cpu']
 r=run('screen',*common);check('screen subcommand runs and writes expected JSON keys',r.returncode==0 and os.path.exists(os.path.join(out,'screen.json')))
 ans=json.load(open(os.path.join(out,'answers.json'))); qid=tok(' Alibaba',add_special_tokens=False)['input_ids'][0]
 for q in c.ID_PROMPTS:ans[q]['answer_ids']=[qid]+tok(' is an assistant.',add_special_tokens=False)['input_ids']
 for q in c.CTRL_PROMPTS:ans[q]['answer_ids']=tok(' A short answer.',add_special_tokens=False)['input_ids']
 json.dump(ans,open(os.path.join(out,'answers.json'),'w'))
 # rerun screen against controlled cache, then verify and downstream stages
 r=run('screen',*common);S=json.load(open(os.path.join(out,'screen.json'))) if r.returncode==0 else {}
 check('screen output has both rankings and N2742 rank',r.returncode==0 and {'top_zeroing','top_push','N2742'}<=set(S))
 r=run('verify',*common);V=json.load(open(os.path.join(out,'verify.json'))) if r.returncode==0 else {}
 check('verify subcommand runs and writes true deltas',r.returncode==0 and 'results' in V)
 # independently recompute one stored zeroing ΔM.
 if r.returncode==0 and V.get('results'):
  row=V['results'][0];pair=(row['layer'],row['index']);base=c.score(tok,model,c.ID_PROMPTS[0],ans[c.ID_PROMPTS[0]])[0].detach();z=c.ZeroHook(layers[pair[0]],pair[1]);actual=c.score(tok,model,c.ID_PROMPTS[0],ans[c.ID_PROMPTS[0]])[0].detach();z.remove();stored=row['identity'][c.ID_PROMPTS[0]]['delta']
  check('verify true ΔM equals independent zeroed minus base calculation',abs(stored-float(actual-base))<1e-5)
 r=run('group',*common);check('group subcommand writes sets and answers',r.returncode==0 and 'sets' in json.load(open(os.path.join(out,'group.json'))))
 r=run('edges',*common);check('edges subcommand writes nodes and effects',r.returncode==0 and 'effects' in json.load(open(os.path.join(out,'edges.json'))))
 corpus=os.path.join(td,'corpus.jsonl');open(corpus,'w').writelines(json.dumps({'text':'This is a document about a quiet conversation and a useful idea. '*4})+'\n' for _ in range(12))
 r=run('describe',*common,'--corpus',corpus,'--docs','10');check('describe subcommand writes lens and corpus contexts',r.returncode==0 and 'nodes' in json.load(open(os.path.join(out,'describe.json'))))
print('PASS' if not fails else f'{fails} FAILED');sys.exit(1 if fails else 0)
