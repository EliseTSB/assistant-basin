"""Fixture and small CPU OLMo 2 checks for Part 139 hooks and scoring."""
from __future__ import annotations
import copy
import json
import random
import tempfile
from pathlib import Path
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM
import run139
import score139


def tiny_model():
    torch.manual_seed(139)
    cfg=Olmo2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                    num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32)
    return Olmo2ForCausalLM(cfg).eval()


def test_do_nothing_hook_matches_plain():
    model=tiny_model(); ids=torch.tensor([[3,7,11,4]])
    with torch.inference_mode(): plain=model(input_ids=ids,use_cache=False).logits
    layer=model.model.layers[0]
    vec=torch.zeros((1,32),dtype=next(model.parameters()).dtype)
    handle=run139._position_hook(layer,vec,"noop")
    try:
        with torch.inference_mode(): hooked=model(input_ids=ids,use_cache=False).logits
    finally: handle.remove()
    assert torch.equal(plain,hooked)


def test_answer_only_hook_leaves_first_generated_token_unchanged():
    model=tiny_model(); ids=torch.tensor([[3,7,11,4]])
    with torch.inference_mode(): plain=model.generate(ids,do_sample=False,max_new_tokens=1)
    vec=torch.ones((1,32),dtype=next(model.parameters()).dtype)
    handle=run139._position_hook(model.model.layers[0],vec,"answer-only")
    try:
        with torch.inference_mode(): hooked=model.generate(ids,do_sample=False,max_new_tokens=1)
    finally: handle.remove()
    assert torch.equal(plain[:,-1],hooked[:,-1])


def test_position_limited_push_changes_only_question_positions():
    model=tiny_model(); ids=torch.tensor([[3,7,11,4]])
    baseline={}
    def capture(_m,_i,out): baseline["hidden"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
    h=model.model.layers[0].register_forward_hook(capture)
    with torch.inference_mode(): model(input_ids=ids,use_cache=False)
    h.remove(); expected=baseline["hidden"].clone()
    vec=torch.ones((1,32),dtype=next(model.parameters()).dtype)
    hook=run139._position_hook(model.model.layers[0],vec,"prompt-only",group="question",groups=[["other","position0","question","other"]])
    actual_box={}
    def capture_pushed(_m,_i,out): actual_box["hidden"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
    capture_handle=model.model.layers[0].register_forward_hook(capture_pushed)
    try:
        with torch.inference_mode(): model(input_ids=ids,use_cache=False)
    finally: hook.remove(); capture_handle.remove()
    actual=actual_box["hidden"]
    changed=(actual!=expected).any(dim=-1)[0]
    assert changed.tolist()==[False,False,True,False]


def fixture_rows():
    rows=[]
    for arm,ids,answer in (("plain",[5,8,9],"I am an assistant."),("noop",[5,8,9],"I am an assistant."),
                           ("full",[5,8,9],"I'm sorry, I cannot help."),("prompt-only",[5,8,9],"I'm sorry, I cannot help."),
                           ("answer-only",[5,8,9],"I am an assistant.")):
        rows.append({"stage":"A","arm":arm,"set":"R","unit_id":"r1","question":"Who are you?","answer":answer,
                     "first_answer_token_ids":ids,"apology":answer.startswith("I'm sorry")})
    rows.extend([
        {"stage":"A","arm":"plain","variant":"batch-control","set":"R","unit_id":"r1","question":"Who are you?","answer":"I am an assistant.","first_answer_token_ids":[5,8,9],"answer_token_ids":[5,8,9]},
        {"stage":"A","arm":"noop","variant":"batch-control","set":"R","unit_id":"r1","question":"Who are you?","answer":"I am an assistant.","first_answer_token_ids":[5,8,9],"answer_token_ids":[5,8,9]},
    ])
    rows.append({"stage":"A","arm":"full","set":"S","unit_id":"s1","question":"Who are you?","answer":"Hello, I can help.","first_answer_token_ids":[5,8,9],"apology":False})
    rows.append({"stage":"B","arm":"question","set":"R","unit_id":"r1","question":"Who are you?","answer":"I'm sorry.","first_answer_token_ids":[5]})
    for arm in ("full","plain"):
        for i in range(6):
            rows.append({"stage":"C","arm":arm,"set":"R","unit_id":"r1","variant":str(i),"question_index":i,
                         "question":f"q{i}","answer":"I'm sorry." if (arm=="full" and i>=3) else "Fine."})
    rows.append({"stage":"D","arm":"plain","variant":"random_ids-0","question":"Who are you?","answer":"I'm sorry."})
    return rows


def test_scorer_fixture_reports_registered_gate_and_predictions():
    out=score139.score(fixture_rows())
    assert out["gates"]["do_nothing_matches_plain"]["passed"]
    assert out["gates"]["answer_only_first_token_matches_plain"]["passed"]
    assert out["predictions"]["139.5"]["held"]
    assert out["predictions"]["139.6"]["held"]
    assert out["predictions"]["139.1"]["result"]["n"]==1


def test_deliberate_break_on_fixture_copy_is_detected():
    broken=copy.deepcopy(fixture_rows())
    for row in broken:
        if row["stage"]=="A" and row["arm"]=="noop":
            row["first_answer_token_ids"]=[99,98,97]
            row["answer_token_ids"]=[99,98,97]
    out=score139.score(broken)
    assert not out["gates"]["do_nothing_matches_plain"]["passed"]


def test_139_3_scores_all_six_position_groups_and_verdict():
    rows=[]
    groups=("position0","user_role","question_words","question_mark","assistant_role","last_position")
    for group,hits in (("position0",0),("user_role",0),("question_words",3),("question_mark",1),("assistant_role",0),("last_position",1)):
        for i in range(5):
            apology=i<hits
            rows.append({"stage":"B","arm":group,"set":"R","unit_id":f"{group}-{i}","answer":"I'm sorry." if apology else "Hello.","apology":apology})
    pred=score139.score(rows)["predictions"]["139.3"]
    assert set(pred["groups"])==set(groups)
    assert pred["groups"]["question_words"]["share"]==.6
    assert pred["held"]


def test_seen_discards_only_an_unparsable_trailing_line():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"rows.jsonl"
        valid=b'{"key":"kept"}\n'
        path.write_bytes(valid+b'{"key":"torn"')
        assert run139._seen(path)=={"kept"}
        assert path.read_bytes()==valid


def test_noop_hook_runs_the_add_path_with_zero_vectors():
    class HookLayer:
        def register_forward_hook(self,fn):
            self.fn=fn
            return type("Handle",(),{"remove":lambda self:None})()
    layer=HookLayer(); hidden=torch.randn(1,2,4); zeros=torch.zeros(1,4)
    run139._position_hook(layer,zeros,"noop")
    result=layer.fn(layer,(),hidden)
    assert result is not None
    assert result.data_ptr()!=hidden.data_ptr()
    assert torch.equal(result,hidden)


def test_unit_selection_uses_seeded_sorted_draws_for_r_c_and_f():
    import select139
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for layer in (4,5):
            rows=[]
            for index in range(130):
                answer="Plain baseline." if index<70 else f"I'm sorry, answer {index}."
                rows.append({"unit":"neuron","index":index,"answer":answer})
            rows.extend({"unit":"random","index":i,"answer":"I'm sorry, random."} for i in range(2))
            (root/f"layer_{layer}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
        first=select139.build(root); second=select139.build(root)
        assert first==second
        for layer in (4,5):
            eligible=list(range(70,130))
            selected_r={u["index"] for u in first["sets"]["R"] if u["layer"]==layer and u["unit"]=="neuron"}
            assert selected_r==set(random.Random(139).sample(eligible,48))
            expected_c=set(random.Random(139).sample(sorted(selected_r),24))
            actual_c={u["index"] for u in first["stage_c"] if u["layer"]==layer and u["set"]=="R"}
            assert actual_c==expected_c
            candidates=sorted(set(range(130))-selected_r)
            expected_f=set(random.Random(139).sample(candidates,64))
            actual_f={u["index"] for u in first["stage_f"]["units"] if u["layer"]==layer and u["unit"]=="neuron"}
            assert actual_f==expected_f
