from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run146, score146

def fixture(root, old, bad=None):
    root.mkdir(parents=True,exist_ok=True);old.mkdir(parents=True,exist_ok=True)
    vocab=8
    base=np.log(np.array([.3,.2,.15,.12,.1,.07,.04,.02]))
    ins=np.log(np.array([.25,.23,.16,.12,.1,.07,.05,.02]))
    sft=np.log(np.array([.24,.22,.16,.13,.1,.08,.05,.02]))
    units=[{"id":f"F-L{l}-{kind}-{i}","unit":kind,"index":i,"layer":l} for l in (4,5) for kind in ("neuron","random") for i in range(2)]
    for q in range(1,5):
        for stage,row in (("instruct",ins),("base",base),("sft",sft)):
            np.save(root/f"plain-{stage}-q{q}.npy",row[None,:].astype("f"))
        mat=np.stack([ins+np.array([.01*i,-.01*i,.01,-.01,0,0,0,0]) for i in range(8)])
        if bad=="gate3" and q==1: mat[0,0]=np.nan
        for arm in ("full","prompt"):
            np.save(root/f"matrix-{arm}-q{q}.npy",mat.astype("f"))
            (root/f"index-{arm}-q{q}.json").write_text(json.dumps({"units":units,"vocab_size":vocab}),encoding="utf-8")
        for layer in (4,5,8,11,15):np.save(root/f"matrix-ladder-L{layer}-q{q}.npy",mat[:2].astype("f"))
        (root/f"index-full-q{q}.json").write_text(json.dumps({"units":units,"vocab_size":vocab,"sorry_id":3,"plain_top_id":0}),encoding="utf-8")
        noops=[{"logp_sorry":float(ins[3]),"logp_plain_top":float(ins[0])} for _ in range(12)]
        if bad=="gate1" and q==1:noops[0]["logp_sorry"]+=.01
        (root/f"noop-q{q}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in noops),encoding="utf-8")
        refs={"empty":base+.001,"shuffled":base+.002,"no_prompt":base+.003}
        np.savez(root/f"refs-instruct-q{q}.npz",empty=refs["empty"],shuffled=refs["shuffled"])
        np.savez(root/f"refs-base-q{q}.npz",chat=base)
        np.savez(root/f"refs-sft-q{q}.npz",chat=sft)
        np.save(root/"refs-instruct-no-prompt.npy",base[None,:])
        np.save(root/"refs-base-no-prompt.npy",base[None,:])
        np.savez(root/f"refs-sft-q{q}.npz",chat=sft)
    for q in (1,2):
        entries=[{"id":i,"token":f"t{i}","change":0.0} for i in range(8)]
        if bad=="gate2" and q==1:entries[0]["change"]=1
        (old/f"words-instruct-chat-q{q}.json").write_text(json.dumps({"top_200_risers":entries,"top_200_fallers":entries}),encoding="utf-8")
    return ins,base

@pytest.mark.parametrize("gate",["gate1","gate2","gate3"])
def test_each_gate_failure_withholds_predictions(tmp_path,gate):
    root,old,out=tmp_path/"data",tmp_path/"old",tmp_path/"out";fixture(root,old,gate)
    with pytest.raises(RuntimeError):score146.score(root,old,out)
    result=json.loads((out/"score.json").read_text(encoding="utf-8"))
    key={"gate1":"gate1_noop","gate2":"gate2_part145","gate3":"gate3_finite"}[gate]
    assert not result["gates"][key]["passed"]
    assert all(x["verdict"]=="withheld" for x in result["predictions"].values())

def test_scorer_includes_ladder_and_bootstrap(tmp_path):
    root,old,out=tmp_path/"data",tmp_path/"old",tmp_path/"out";fixture(root,old)
    vocab=tmp_path/"vocab.json";vocab.write_text(json.dumps({f"t{i}":i for i in range(8)}),encoding="utf-8")
    result=score146.score(root,old,out,vocab)
    assert all(x["passed"] for x in result["gates"].values())
    assert "ladder-L15" in result["questions"]["1"]["arms"]
    assert result["predictions"]["146.7"]["verdict"] in ("held","failed")
    assert result["bootstrap"]["q1-base_chat"]["resamples"]==2000
    assert (out/"tables.md").is_file()

@pytest.mark.parametrize("bad",[False,True])
def test_import_isolated_and_help(bad):
    script=HERE/"score146.py" if bad else HERE/"run146.py"
    proc=subprocess.run([sys.executable,"-I","-B",str(script),"--help"],cwd=HERE.parent,capture_output=True,text=True,check=False)
    assert proc.returncode==0,proc.stderr

@pytest.mark.parametrize("nan",[False,True])
def test_tiny_model_main_sequence_and_nonfinite_stop(tmp_path,monkeypatch,nan):
    torch=pytest.importorskip("torch")
    class Tok:
        bos_token_id=6
        def __call__(self,text,**kw):
            d={"I":[1]," I":[1],"'m":[2]," sorry":[3],"I'm":[1,2]," I'm":[1,2]}
            return {"input_ids":d[text]}
    class Layer(torch.nn.Module):
        def __init__(self):super().__init__();self.mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3,3))))
        def forward(self,x):return x+0.01
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(8,3);self.layers=torch.nn.ModuleList([Layer() for _ in range(16)]);self.lm_head=torch.nn.Linear(3,8);self.inputs=[]
        def get_output_embeddings(self):return self.lm_head
        def forward(self,input_ids,use_cache=True,past_key_values=None):
            self.inputs.append(input_ids.detach().cpu().tolist())
            x=self.embedding(input_ids)
            for l in self.layers:x=l(x)
            logits=self.lm_head(x)
            if nan:logits=logits*float("nan")
            return SimpleNamespace(logits=logits,past_key_values=(input_ids if past_key_values is None else past_key_values))
    model,tok=Model(),Tok()
    class Common:
        @staticmethod
        def get_layers(_):return model.layers
        @staticmethod
        def prompt_ids(*_):return [4,5]
    monkeypatch.setitem(sys.modules,"common92",Common)
    monkeypatch.setattr(run146.run144,"_load_direct",lambda *_:(tok,model))
    def vectors(units,layer,weight,*args,**kwargs):return torch.ones((len(units),3),device=weight.device)
    monkeypatch.setattr(run146.run144.run139,"_unit_vectors",vectors)
    units=[{"id":f"F{l}{kind}","unit":kind,"index":0,"layer":l} for l in (4,5) for kind in ("neuron","random")]
    path=tmp_path/"units.json";path.write_text(json.dumps({"stage_f":{"units":units}}),encoding="utf-8")
    args=["--stage","instruct","--device","cpu","--model","stub","--units",str(path),"--out",str(tmp_path/"out"),"--limit","4"]
    if nan:
        with pytest.raises(FloatingPointError):run146.main(args)
    else:
        run146.main(args)
        assert (tmp_path/"out/matrix-ladder-L15-q4.npy").exists()
        # A no-prompt sequence prefills BOS, then forces the two tokens.
        model.inputs.clear()
        run146.sequence(model,tok,model.layers[4],None,"Who are you?","chat","cpu",prompt=[6])
        assert model.inputs==[[[6]],[[1]],[[2]]]
        assert np.load(tmp_path/"out/refs-instruct-no-prompt.npy").shape==(1,8)
