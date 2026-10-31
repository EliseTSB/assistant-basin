"""CPU fixtures for Part 129 collection, masking, resume and scoring."""
import copy, gzip, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import collect129, score129
import prepare129

def tiny_models():
    cfg=Qwen3Config(vocab_size=97,hidden_size=16,intermediate_size=8,num_hidden_layers=2,
                    num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,
                    attention_dropout=0.0)
    torch.manual_seed(129)
    a=Qwen3ForCausalLM(cfg).eval(); b=copy.deepcopy(a).eval()
    with torch.no_grad():
        for p in b.parameters(): p.add_(torch.randn_like(p)*0.001)
    return a,b

def capture(model, ids, mask):
    got={}; hooks=[]
    for i,layer in enumerate(model.model.layers):
        def hook(_m,x,i=i): got[i]=x[0].detach().clone()
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    with torch.inference_mode(): model(input_ids=ids,attention_mask=mask)
    for h in hooks:h.remove()
    return got

class FakeTokenizer:
    ids={"<|im_start|>":151644,"<|im_end|>":151645,"assistant":77091}
    def convert_tokens_to_ids(self,x): return self.ids.get(x, -1)

def test_group_masks_from_real_qwen3_rendered_chat_ids():
    # Qwen3 rendering: user Hi; assistant Hello; user Bye; assistant Goodbye.
    # Includes the actual structural and role token IDs from the pinned tokenizer.
    ids=torch.tensor([[151644,77090,11,1,151645,151644,77091,11,2,151645,
                       151644,77090,11,3,151645,151644,77091,11,4,151645,0]])
    mask=torch.tensor([[1]*20+[0]])
    got=collect129.group_masks(ids,"chat",FakeTokenizer(),mask)
    assert got["chat-user"][0,:5].all() and got["chat-user"][0,10:15].all()
    assert got["chat-assistant"][0,5:10].all() and got["chat-assistant"][0,15:20].all()
    assert not got["chat-assistant"][0,20] and not got["chat-user"][0,20]

def test_gate2_layout_reverses_rows_adds_padding_and_restores_hook_output():
    ids=torch.tensor([[1,2,3],[4,5,6]])
    mask=torch.ones_like(ids)
    changed, changed_mask=collect129.self_compare_layout(ids,mask,0)
    assert changed.shape==(2,11) and changed[:,:3].tolist()==[[4,5,6],[1,2,3]]
    assert changed_mask[:,:3].all() and not changed_mask[:,3:].any()
    y=torch.arange(2*11*3).reshape(2,11,3)
    restored=collect129.restore_self_compare(y,3)
    assert torch.equal(restored,y.flip(0)[:,:3])

def test_empty_groups_add_no_counts_or_sums():
    c=collect129.Sums(1,4,"cpu"); c.above=torch.zeros((5,1,2,4),dtype=torch.int64)
    x=torch.randn(2,5,4); masks={g:torch.zeros((2,5),dtype=torch.bool) for g in collect129.GROUPS}
    c.add(0,x,x,masks,torch.arange(4),torch.zeros((1,2,4)))
    assert not c.count.any() and not c.data.any() and not c.above.any()

def test_design_corpus_counts_are_shared_and_full_sized():
    assert collect129.CORPUS_DOCS == {"dclm":10000,"zh":2000,"code":2000,"chat":2000}

def test_prepare_writes_stable_keys_and_separate_pilot_checks(tmp_path):
    catalog=tmp_path/"catalog"; (catalog/"neurons/acts").mkdir(parents=True)
    rows=[{"doc":i,"slice":name,"line":line,"tokens":12}
          for i,(name,line) in enumerate((name,line) for name in collect129.CORPUS_ORDER for line in range(100))]
    pq.write_table(pa.Table.from_pylist(rows),catalog/"docs.parquet")
    top_doc=list(range(24)); top_pos=[2]*24; top_val=[1.0]*24
    for layer in range(36):
        table_rows=[{"neuron":n,"pole":1 if n%2==0 else -1,"act_top":1.0,
                     "top_doc":top_doc,"top_pos":top_pos,"top_val":top_val} for n in range(20)]
        pq.write_table(pa.Table.from_pylist(table_rows),catalog/f"neurons/acts/L{layer:02d}.parquet")
    output=tmp_path/"prepared.pt"
    prepare129.prepare(catalog,output)
    data=torch.load(output,map_location="cpu",weights_only=False)
    assert len(data["alignment"])==500 and len(data["pilot_alignment"])==300
    assert all("doc" not in row and row["line"]<100 for row in data["alignment"]+data["pilot_alignment"])

def test_float64_sums_match_numpy_all_neurons_and_drop_sink_padding():
    torch.manual_seed(7); x=torch.randn(2,5,4); y=x*0.7+torch.randn_like(x)*0.2
    valid=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    masks={g:(valid if g=="web" else torch.zeros_like(valid)) for g in collect129.GROUPS}
    c=collect129.Sums(1,4,"cpu"); c.above=torch.zeros((5,1,2,4),dtype=torch.int64)
    perm=torch.tensor([1,2,3,0]); th=torch.zeros((1,2,4))
    c.add(0,x,y,masks,perm,th)
    a=x[:,1:][valid[:,1:]].numpy(); b=y[:,1:][valid[:,1:]].numpy()
    for n in range(4):
        assert abs(score129.corr({"data":c.data.cpu(),"count":c.count.cpu()},0,0)[n]-np.corrcoef(a[:,n],b[:,n])[0,1])<1e-9
    assert int(c.count[0,0])==6
    assert c.data[0,0,0,0].item()==sum(a[:,0])
    assert torch.allclose(c.data[0,0,8],torch.tensor(np.maximum(a,0).sum(0),dtype=torch.float64))
    assert torch.allclose(c.data[0,0,10],torch.tensor((np.maximum(a,0)**2).sum(0),dtype=torch.float64))
    assert c.above[0,0].sum()>0

def test_tiny_qwen3_pair_resume_is_identical():
    a,b=tiny_models(); ids=torch.randint(0,96,(3,8)); mask=torch.ones_like(ids)
    x=capture(a,ids,mask); y=capture(b,ids,mask)
    assert len(x)==len(y)==2
    full=collect129.Sums(2,8,"cpu"); full.above=torch.zeros((5,2,2,8),dtype=torch.int64)
    group={g:(mask.bool() if g=="web" else torch.zeros_like(mask,dtype=torch.bool)) for g in collect129.GROUPS}
    perm=torch.roll(torch.arange(8),1); thresholds=torch.zeros((2,2,8))
    for L in range(2): full.add(L,x[L],y[L],group,perm,thresholds)
    stopped=collect129.Sums(2,8,"cpu"); stopped.above=torch.zeros_like(full.above)
    stopped.add(0,x[0],y[0],group,perm,thresholds)
    saved={"data":stopped.data.clone(),"count":stopped.count.clone(),"above":stopped.above.clone()}
    resumed=collect129.Sums(2,8,"cpu"); resumed.above=torch.zeros_like(full.above)
    resumed.data.copy_(saved["data"]); resumed.count.copy_(saved["count"]); resumed.above.copy_(saved["above"])
    resumed.add(1,x[1],y[1],group,perm,thresholds)
    assert torch.equal(full.data,resumed.data) and torch.equal(full.count,resumed.count)

def fixture(tmp):
    cat=tmp/"catalog"; (cat/"neurons/acts").mkdir(parents=True); (cat/"neurons/labels").mkdir(parents=True)
    lib=tmp/"library";lib.mkdir(); repo=tmp/"repo"; (repo/"part91").mkdir(parents=True);(repo/"results/part91").mkdir(parents=True)
    for L in range(36):
        pq.write_table(pa.table({"neuron":[0,1,2,3],"pole":[1,-1,1,-1]}),cat/f"neurons/acts/L{L:02d}.parquet")
        pq.write_table(pa.table({"neuron":[0,0,1,1,2,2,3,3],"pole":[1,-1]*4,
                                "fires":["assistant register",""]*4,"writes":[""]*8}),cat/f"neurons/labels/L{L:02d}.parquet")
        with gzip.open(lib/f"L{L:02d}.gz","wt",encoding="utf-8") as f:
            json.dump({"categories":["persona","other"],"levels":[],"rows":[[0]]+[None]*7},f)
    (repo/"part91/neurons91.json").write_text(json.dumps([{"id":"p00","L":0,"n":0,"s":1},{"id":"p01","L":0,"n":1,"s":-1}]))
    (repo/"results/part91/labels.jsonl").write_text("\n".join([json.dumps({"id":f"inst_chat:{i}","shows_persona":True}) for i in range(4)]+[json.dumps({"id":f"base_plain:{i}","shows_persona":False}) for i in range(4)])+"\n")
    (repo/"results/part91/inst_chat.jsonl").write_text("\n".join(json.dumps({"cell":"p00" if i==0 else "p01"}) for i in range(4))+"\n")
    (repo/"results/part91/base_plain.jsonl").write_text("\n".join(json.dumps({"cell":"p00" if i==0 else "p01"}) for i in range(4))+"\n")
    sums=tmp/"run";sums.mkdir();data=torch.zeros((5,36,18,4),dtype=torch.float64);count=torch.full((5,36),200,dtype=torch.int64);above=torch.full((5,36,2,4),200,dtype=torch.int64)
    # x=y gives exact r=1 and rectified r=1 for all fixture neurons.
    for g in range(5):
      for L in range(36):
        n=200.; sx=0.; xx=200.; xy=200.
        data[g,L,0]=sx;data[g,L,1]=sx;data[g,L,2]=xx;data[g,L,3]=xx;data[g,L,4]=xy
        for pole in range(2): data[g,L,8+5*pole,0:4]=100;data[g,L,9+5*pole,0:4]=100;data[g,L,10+5*pole,0:4]=100;data[g,L,11+5*pole,0:4]=100;data[g,L,12+5*pole,0:4]=100
    torch.save({"data":data,"count":count,"above":above},sums/"main-sums.pt")
    torch.save({"layer_cosines":[{"down":torch.ones(4)} for _ in range(36)]},sums/"weight-cosines.pt")
    return sums/"main-sums.pt",cat,lib,repo

def test_scorer_fixture(tmp_path):
    sums,cat,lib,repo=fixture(tmp_path); out=tmp_path/"score.json"
    score129.score(sums,cat,lib,repo,out,[])
    d=json.loads(out.read_text());assert d["predictions"]["129.1"]["web_median_r"]==1.0
    assert d["predictions"]["129.7"]["lowest_n"]==144

def test_deliberate_break_copy_makes_scoring_assertion_red(tmp_path):
    source=HERE/"score129.py";broken=tmp_path/"broken_score129.py"
    text=source.read_text();assert "base = 8 + 5 * rect" in text
    broken.write_text(text.replace("base = 8 + 5 * rect","base = 9 + 5 * rect",1))
    probe=tmp_path/"probe.py"
    probe.write_text(f'''import importlib.util, torch, numpy as np\ns=importlib.util.spec_from_file_location("broken", {str(broken)!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\nd={{"count":torch.tensor([[200]]),"data":torch.zeros((1,1,18,1),dtype=torch.float64)}}\nd["data"][0,0,8:13,0]=torch.tensor([100,100,100,100,100],dtype=torch.float64)\nassert abs(m.corr(d,0,0,0)[0]-1.0)<1e-9\n''')
    red=subprocess.run([sys.executable,"-B",str(probe)],capture_output=True,text=True)
    assert red.returncode!=0
