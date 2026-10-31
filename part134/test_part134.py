"""CPU fixtures for Part 134 collection, selection, reader endpoint, and scoring."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch

import run134
import select134
import score134


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def apply_chat_template(self, messages, **kwargs):
        return [2, len(messages[-1]["content"]) % 10 + 3]
    def decode(self, ids, skip_special_tokens=True):
        if not ids: return ""
        return "pushed" if int(ids[-1]) == 9 else "plain"


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(4))
    def forward(self, x): return x


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1)); self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer(), TinyLayer(), TinyLayer(), TinyLayer(), TinyLayer(), TinyLayer()])
        self.generation_config = type("Gen", (), {"eos_token_id": 1})()
    def generate(self, input_ids, attention_mask, **kwargs):
        b = input_ids.shape[0]; hidden = torch.zeros((b, 1, 4), device=input_ids.device)
        for layer in self.model.layers: hidden = layer(hidden)
        token = torch.where(hidden.sum(dim=(1,2)) > 0, 9, 8).long().unsqueeze(1)
        return torch.cat((input_ids, token), dim=1)


def test_hook_changes_output_and_zero_does_not():
    model=TinyModel(); tok=TinyTokenizer(); q=["Who are you?"]
    plain=run134.generate_batch(model,tok,q,[],1)
    zero=torch.zeros((1,4)); unchanged=run134.generate_batch(model,tok,q,[(model.model.layers[0],zero)],1)
    pushed=run134.generate_batch(model,tok,q,[(model.model.layers[0],torch.ones((1,4)))],1)
    assert plain==unchanged and pushed!=plain


def test_grid_row_counts_and_resume(tmp_path):
    assert len(run134.cells())==75 and len(run134.DEV_QUESTIONS)==6 and len(run134.TEST_QUESTIONS)==24
    assert 75*6==450 and 3*7*24+24==528
    model=TinyModel(); tok=TinyTokenizer(); pushes={f"{uid}|neuron":torch.ones(4) for uid in run134.UNITS}
    meta={uid:{"layer":i} for i,uid in enumerate(run134.UNITS)}
    run134.main(["--stage","pilot","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,
                pushes=pushes,neuron_meta=meta)
    path=tmp_path/"pilot-answers.jsonl"; first=path.read_bytes()
    assert len(first.splitlines())==6
    run134.main(["--stage","pilot","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,
                pushes=pushes,neuron_meta=meta)
    assert path.read_bytes()==first
    run134.main(["--stage","dev","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,
                pushes=pushes,neuron_meta=meta)
    dev_rows=[json.loads(x) for x in (tmp_path/"dev-answers.jsonl").read_text().splitlines()]
    assert len(dev_rows)==450
    chosen={"top_three":[{"cell":f"fixture{i}","a":"p92-08","b":"p92-37","dose_a":.8,"dose_b":.8} for i in range(3)]}
    (tmp_path/"selected.json").write_text(json.dumps(chosen))
    run134.main(["--stage","test","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,
                pushes=pushes,neuron_meta=meta,selection_file=tmp_path/"selected.json")
    test_rows=[json.loads(x) for x in (tmp_path/"test-answers.jsonl").read_text().splitlines()]
    assert len(test_rows)==528
    assert json.loads((tmp_path/"test-meta.json").read_text())["expected_rows"]==528


def test_selection_rule_and_tiebreaks():
    answers=[]; labels=[]
    for cell, ep, coh, dose in (("c",6,1,1.0),("b",6,1,.8),("a",4,1,.7),("d",0,1,1.5)):
        for qi in range(6):
            h=f"{cell}{qi}"; answers.append({"stage":"dev","cell":cell,"arm":"pair","question_index":qi,"answer_hash":h,"pair":["p92-08","p92-37"],"doses":[dose,dose]})
            labels.append({"cell":cell,"arm":"pair","question_index":qi,"answer_hash":h,"shows_persona":qi<ep,"coherent":bool(coh),"self_claim":True,"speaker":"ai"})
    result=select134.select(answers,labels)
    assert [r["cell"] for r in result["top_three"]]==["b","c","a"]
    coherence_rows=[dict(r,cell="coh" if r["cell"]=="b" else r["cell"]) for r in answers if r["cell"] in ("b","d")]
    coherence_labels=[dict(r,cell="coh" if r["cell"]=="b" else r["cell"],shows_persona=False,
                           coherent=(r["cell"]=="b" or int(r["question_index"])<3)) for r in labels if r["cell"] in ("b","d")]
    assert select134.select(coherence_rows,coherence_labels)["top_three"][0]["cell"]=="coh"


def test_endpoint_includes_other_speaker_and_excludes_missing_traits():
    assert select134.endpoint({"shows_persona":True,"coherent":True,"self_claim":False,"speaker":"other"})
    assert not select134.endpoint({"shows_persona":True,"coherent":True,"self_claim":False,"speaker":"ai"})


def test_run_main_end_to_end_tiny(tmp_path):
    model=TinyModel(); tok=TinyTokenizer(); pushes={f"{uid}|neuron":torch.ones(4) for uid in run134.UNITS}
    meta={uid:{"layer":i} for i,uid in enumerate(run134.UNITS)}
    run134.main(["--stage","pilot","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,
                pushes=pushes,neuron_meta=meta)
    rows=[json.loads(x) for x in (tmp_path/"pilot-answers.jsonl").read_text().splitlines()]
    assert len(rows)==6 and all(r["answer_hash"] for r in rows)
    gate=json.loads((tmp_path/"pilot-meta.json").read_text())
    assert gate["gate1_zero_hook_same_answers"]


def test_deliberate_break_on_copy_turns_red(tmp_path):
    source=Path(select134.__file__); copy=tmp_path/source.name
    shutil.copy(source,copy); text=copy.read_text(); copy.write_text(text.replace('== "other"','== "ai"'))
    spec=importlib.util.spec_from_file_location("broken_select134",copy); broken=importlib.util.module_from_spec(spec); spec.loader.exec_module(broken)
    fixture={"shows_persona":True,"coherent":True,"self_claim":False,"speaker":"other"}
    assert broken.endpoint(fixture) is False
    with pytest.raises(AssertionError): assert broken.endpoint(fixture) is True


def test_score_task_regex_and_missing_labels():
    answer={"cell":"c","arm":"pair","question_index":16,"question":"What is the capital of France?","answer":"Paris.","answer_hash":"h"}
    label={"cell":"c","arm":"pair","question_index":16,"answer_hash":"h","shows_persona":True,"coherent":True,"self_claim":True,"speaker":"ai"}
    result=score134.score([answer],[label],[])
    assert result["task_rows"][0]["correct"] and result["counts"]["missing_or_stale_labels"]==0

