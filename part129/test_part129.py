"""CPU fixtures for Part 129 collection, masking, resume and scoring."""
import copy, gzip, importlib.util, json, shutil, subprocess, sys
import threading
from pathlib import Path
import numpy as np
import pytest
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

def test_layer_indexed_exchange_matches_each_models_plain_layer_outputs():
    a,b=tiny_models(); ids=torch.randint(0,96,(3,8)); mask=torch.ones_like(ids)
    # Plain, separate passes are the reference for both the per-layer inputs and r.
    plain_a=capture(a,ids,mask); plain_b=capture(b,ids,mask)
    assert any(not torch.equal(plain_a[L],plain_b[L]) for L in plain_a)
    exchange=collect129.LayerActivationExchange(2)
    paired=collect129.Sums(2,8,"cpu"); paired.above=torch.zeros((5,2,2,8),dtype=torch.int64)
    group={g:(mask.bool() if g=="web" else torch.zeros_like(mask,dtype=torch.bool)) for g in collect129.GROUPS}
    perm=torch.roll(torch.arange(8),1); thresholds=torch.zeros((2,2,8))
    hooks=[]
    for L,layer in enumerate(a.model.layers):
        def put(_m,inputs,layer_i=L): exchange.put(layer_i,inputs[0].detach().clone())
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(put))
    for L,layer in enumerate(b.model.layers):
        def take(_m,inputs,layer_i=L):
            x=exchange.get(layer_i)
            paired.add(layer_i,x,inputs[0].detach(),group,perm,thresholds)
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(take))
    errors=[]
    def run_b():
        try:
            with torch.inference_mode(): b(input_ids=ids,attention_mask=mask)
        except BaseException as exc: errors.append(exc)
    worker=threading.Thread(target=run_b)
    try:
        worker.start()
        with torch.inference_mode(): a(input_ids=ids,attention_mask=mask)
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert not errors
    finally:
        for hook in hooks: hook.remove()
    reference=collect129.Sums(2,8,"cpu"); reference.above=torch.zeros_like(paired.above)
    for L in range(2): reference.add(L,plain_a[L],plain_b[L],group,perm,thresholds)
    assert torch.equal(paired.count,reference.count)
    assert torch.equal(paired.data,reference.data)
    for L in range(2):
        assert np.allclose(score129.corr({"data":paired.data,"count":paired.count},0,L),
                           score129.corr({"data":reference.data,"count":reference.count},0,L),equal_nan=True)
    # The exchange rejects an item tagged for the wrong layer rather than pairing it silently.
    exchange.queues[0].put((1,torch.zeros(1)))
    with pytest.raises(RuntimeError,match="activation layer mismatch"):
        exchange.get(0)

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
    # Deliberately distinct groups and signs make scorer group/pole wiring observable.
    for g in range(5):
      for L in range(36):
        data[g,L,2]=200;data[g,L,3]=200
        data[g,L,4]=100 if g==0 else 40 if g==4 else 20
        for pole,xy in ((0,75),(1,60)):
          base=8+5*pole
          data[g,L,base]=100;data[g,L,base+1]=100
          data[g,L,base+2]=100;data[g,L,base+3]=100;data[g,L,base+4]=xy
    torch.save({"data":data,"count":count,"above":above},sums/"main-sums.pt")
    torch.save({"layer_cosines":[{"down":torch.ones(4)} for _ in range(36)]},sums/"weight-cosines.pt")
    return sums/"main-sums.pt",cat,lib,repo

def test_scorer_fixture(tmp_path):
    sums,cat,lib,repo=fixture(tmp_path); out=tmp_path/"score.json"
    score129.score(sums,cat,lib,repo,out,[])
    d=json.loads(out.read_text());assert d["predictions"]["129.1"]["web_median_r"]==0.5
    assert abs(d["predictions"]["129.2"]["chat_assistant_minus_web"]+0.3)<1e-9
    assert abs(d["predictions"]["129.6"]["once_minus_carrier_chat_assistant_rectified"]-0.3)<1e-9
    assert d["predictions"]["129.7"]["lowest_n"]==144

def test_slim_catalog_loader_matches_legacy_row_score(tmp_path):
    sums,cat,lib,repo=fixture(tmp_path)
    slim=tmp_path/"slim.json"; legacy=tmp_path/"legacy.json"
    score129.score(sums,cat,lib,repo,slim,[])
    legacy_rows=(
        [pq.read_table(cat/f"neurons/acts/L{L:02d}.parquet").to_pylist() for L in range(36)],
        [pq.read_table(cat/f"neurons/labels/L{L:02d}.parquet").to_pylist() for L in range(36)],
    )
    score129.score(sums,cat,lib,repo,legacy,[],_legacy_rows=legacy_rows)
    assert slim.read_bytes()==legacy.read_bytes()
    neurons,poles=score129.load_act_layer(cat,0)
    old_acts=legacy_rows[0][0]
    assert neurons.tolist()==[r["neuron"] for r in old_acts]
    assert poles.tolist()==[r["pole"] for r in old_acts]

def test_deliberate_break_copy_makes_scoring_assertion_red(tmp_path):
    source=HERE/"score129.py";broken=tmp_path/"broken_score129.py"
    text=source.read_text();assert "base = 8 + 5 * rect" in text
    broken.write_text(text.replace("base = 8 + 5 * rect","base = 9 + 5 * rect",1))
    probe=tmp_path/"probe.py"
    probe.write_text(f'''import importlib.util, torch, numpy as np\ns=importlib.util.spec_from_file_location("broken", {str(broken)!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\nd={{"count":torch.tensor([[200]]),"data":torch.zeros((1,1,18,1),dtype=torch.float64)}}\nd["data"][0,0,8:13,0]=torch.tensor([100,100,100,100,100],dtype=torch.float64)\nassert abs(m.corr(d,0,0,0)[0]-1.0)<1e-9\n''')
    red=subprocess.run([sys.executable,"-B",str(probe)],capture_output=True,text=True)
    assert red.returncode!=0

def test_gate1_requires_all_stage_specific_checks_and_reports_measured_count():
    checks=[{"layer":0,"neuron":1,"pole":1,"slice":"dclm","line":2,"position":3,"top_val":2.0},
            {"layer":0,"neuron":2,"pole":-1,"slice":"dclm","line":3,"position":4,"top_val":2.0}]
    key=lambda r:f"{r['layer']}:{r['neuron']}:{r['pole']}:{r['slice']}:{r['line']}:{r['position']}"
    verdict=score129.alignment_verdict(checks,{key(checks[0]):2.0})
    assert verdict["checks_measured"]==1 and verdict["checks_required"]==2 and not verdict["pass"]
    assert score129.alignment_verdict(checks,{key(r):r["top_val"] for r in checks})["pass"]

def test_gate2_requires_different_batch_captures_and_handles_bad_files(tmp_path):
    def saved(batch):
        data=torch.zeros((5,36,5,2),dtype=torch.float64)
        count=torch.full((5,36),20,dtype=torch.int64)
        for g in range(5):
            for L in range(36):
                data[g,L,0]=20;data[g,L,1]=20
                data[g,L,2]=40;data[g,L,3]=40;data[g,L,4]=20
        p=tmp_path/f"b{batch}.pt"
        torch.save({"data":data,"count":count,"meta":{"batch":batch}},p)
        return p
    a,b=saved(16),saved(64)
    assert score129.score_gate2([a,b])["batches"]==[16,64]
    copied=tmp_path/"copied-relabelled.pt"
    payload=torch.load(a,map_location="cpu",weights_only=False)
    payload["meta"]["batch"]=64
    torch.save(payload,copied)
    copied_verdict=score129.score_gate2([a,copied])
    assert copied_verdict["pass"] is False and "identical" in copied_verdict["reason"]
    same=saved(32)
    assert score129.score_gate2([same,same])["pass"] is False
    malformed=tmp_path/"bad.pt";torch.save({"meta":{"batch":8}},malformed)
    with pytest.raises(ValueError,match="malformed Gate 2 capture"):
        score129.score_gate2([a,malformed])

def test_derangement_has_no_fixed_neurons_and_collects_independent_null():
    p=collect129.fixed_derangement(32,torch.Generator().manual_seed(129))
    assert not torch.equal(p,torch.arange(32)) and torch.all(p!=torch.arange(32))

def test_run_checkpoint_writer_persists_sums_progress_and_alignment(tmp_path):
    c=collect129.Sums(1,3,"cpu");c.above=torch.zeros((5,1,2,3),dtype=torch.int64)
    c.data[0,0,0]=torch.tensor([1.,2.,3.],dtype=torch.float64)
    target=tmp_path/"pilot-sums.pt"
    collect129.save_checkpoint(target,{"stage":"pilot"},c,{"k":4.0},7)
    saved=torch.load(target,map_location="cpu",weights_only=False)
    assert saved["next_doc"]==7 and saved["state"]["alignment_values"]=={"k":4.0}
    assert torch.equal(saved["state"]["data"],c.data.cpu())

def test_rectified_poles_follow_the_registered_sign_on_sign_flips():
    torch.manual_seed(129);x=torch.randn(1,1001,8);y=-x+torch.randn_like(x)*.01
    c=collect129.Sums(1,8,"cpu");c.above=torch.zeros((5,1,2,8),dtype=torch.int64)
    masks={g:(torch.ones((1,1001),dtype=torch.bool) if g=="web" else torch.zeros((1,1001),dtype=torch.bool)) for g in collect129.GROUPS}
    c.add(0,x,y,masks,collect129.fixed_derangement(8,torch.Generator().manual_seed(1)),torch.zeros((1,2,8)))
    d={"data":c.data,"count":c.count}
    assert np.nanmax(score129.corr(d,0,0,0))<-.3
    assert np.nanmax(score129.corr(d,0,0,1))<-.3

def test_persona_controls_are_filtered_before_sampling():
    active,controls=score129.eligible_matches([(1,1),(2,-1)],[(3,1),(4,-1)],lambda n,s:n in (1,3,4))
    assert len(active)==1 and len(controls)==2
    assert len(controls)>=len(active)

def test_corpus_paths_are_bound_to_their_registered_slice_names():
    paths=["/tmp/neuron-lens/data/corpus/act-dclm.jsonl","/tmp/neuron-lens/data/corpus/act-zh.jsonl","/tmp/neuron-lens/data/corpus/act-code.jsonl","/tmp/neuron-lens/data/corpus/act-chat-qwen3-8b.jsonl"]
    collect129.validate_corpus_paths(paths)
    paths[1],paths[2]=paths[2],paths[1]
    with pytest.raises(ValueError,match="zh corpus must"):
        collect129.validate_corpus_paths(paths)

def test_resume_allows_batch_change_but_requires_matching_content_provenance():
    saved={"stage":"main","batch":16,"corpus_sha256":"abc","prepared_sha256":"def"}
    assert collect129.resume_metadata_matches(saved,{**saved,"batch":64})
    assert not collect129.resume_metadata_matches(saved,{**saved,"batch":64,"corpus_sha256":"xyz"})

def test_prepared_catalog_includes_docs_provenance(tmp_path):
    catalog=tmp_path/"catalog";(catalog/"neurons/acts").mkdir(parents=True)
    rows=[{"doc":i,"slice":name,"line":line,"tokens":12}
          for i,(name,line) in enumerate((name,line) for name in collect129.CORPUS_ORDER for line in range(100))]
    pq.write_table(pa.Table.from_pylist(rows),catalog/"docs.parquet")
    for layer in range(36):
        pq.write_table(pa.Table.from_pylist([{"neuron":0,"pole":1,"act_top":1.0,
            "top_doc":[0],"top_pos":[2],"top_val":[1.0]}]),catalog/f"neurons/acts/L{layer:02d}.parquet")
    output=tmp_path/"prepared.pt";prepare129.prepare(catalog,output)
    payload=torch.load(output,map_location="cpu",weights_only=False)
    import hashlib
    expected=hashlib.sha256((catalog/"docs.parquet").read_bytes()).hexdigest()
    assert payload["docs_sha256"]==expected
    score129.verify_prepared_provenance(payload,catalog/"docs.parquet")
    with pytest.raises(ValueError,match="different docs.parquet"):
        score129.verify_prepared_provenance({"docs_sha256":"stale"},catalog/"docs.parquet")

def test_copyback_retries_before_destroy_and_rental_cap_still_attempts_destroy():
    import importlib.util
    vast_path=HERE.parent/"cloud/vast-part129-2026-10-30/vast.py"
    spec=importlib.util.spec_from_file_location("part129_vast_test",vast_path)
    vast=importlib.util.module_from_spec(spec);spec.loader.exec_module(vast)
    calls=[]
    def flaky(*_args):
        calls.append(1)
        if len(calls)==1: raise OSError("transient copy failure")
    vast._copy_results=flaky
    vast._copy_results_with_retry("host",22,9999999999)
    assert len(calls)==2
    assert not vast._should_destroy("main",2,False,False)
    assert vast._should_destroy("main",2,False,True)
    assert vast._should_destroy("main",2,True,False)
    assert not vast._should_destroy("pilot",0,True,False)  # success is reused by the main stage
    assert vast._should_destroy("pilot",2,True,False)  # failed pilot with safe copy-back
