"""CPU integration fixtures for the registered Part 134 path."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import label134
import run134
import score134
import select134


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def apply_chat_template(self, messages, **kwargs):
        return [2, len(messages[-1]["content"]) % 10 + 3]
    def decode(self, ids, skip_special_tokens=True):
        return "pushed" if ids and int(ids[-1]) == 9 else "plain"


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(4))
    def forward(self, x): return x


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1)); self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer() for _ in range(7)])
        self.generation_config = type("Gen", (), {"eos_token_id": 1})()
    def generate(self, input_ids, attention_mask, **kwargs):
        hidden = torch.zeros((input_ids.shape[0], 1, 4), device=input_ids.device)
        for layer in self.model.layers: hidden = layer(hidden)
        token = torch.where(hidden.sum(dim=(1, 2)) > 0, 9, 8).long().unsqueeze(1)
        return torch.cat((input_ids, token), dim=1)


def tiny_inputs():
    model=TinyModel(); tok=TinyTokenizer()
    pushes={f"{uid}|neuron":torch.ones(4) for uid in run134.UNITS}
    meta={uid:{"layer":i} for i,uid in enumerate(run134.UNITS)}
    return model,tok,pushes,meta


def label_for(answer, *, persona=False, coherent=True):
    return {"stage":answer["stage"],"cell":answer["cell"],"arm":answer["arm"],
            "question_index":answer["question_index"],"answer_hash":answer["answer_hash"],
            "shows_persona":persona,"self_claim":persona,"speaker":"ai","coherent":coherent}


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def known_rows():
    return [{"known_index":i,"expected":i<6,"shows_persona":i<6,"self_claim":True,
             "speaker":"ai","coherent":True} for i in range(12)]


def test_selector_output_flows_into_real_test_stage(tmp_path):
    model,tok,pushes,meta=tiny_inputs()
    run134.main(["--stage","dev","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,pushes=pushes,neuron_meta=meta)
    answers=[json.loads(line) for line in (tmp_path/"dev-answers.jsonl").read_text().splitlines()]
    labels=[label_for(row,persona=(row["question_index"]<3)) for row in answers]
    write_jsonl(tmp_path/"dev-labels.jsonl",labels)
    selection_path=tmp_path/"selected.json"
    result=select134.main(["--answers",str(tmp_path/"dev-answers.jsonl"),"--labels",str(tmp_path/"dev-labels.jsonl"),"--out",str(selection_path)])
    assert len(result["top_three"])==3
    assert set(result["top_three"][0]) >= {"pair","doses"}
    run134.main(["--stage","test","--device","cpu","--out",str(tmp_path)],model=model,tokenizer=tok,
                pushes=pushes,neuron_meta=meta,selection_file=selection_path)
    test_rows=[json.loads(line) for line in (tmp_path/"test-answers.jsonl").read_text().splitlines()]
    assert len(test_rows)==528
    assert {row["cell"] for row in test_rows if row["cell"]!="unpushed"} == {row["cell"] for row in result["top_three"]}


def test_selector_refuses_partial_or_error_labels():
    answers=[]; labels=[]
    for cell_no in range(75):
        for qi in range(6):
            row={"stage":"dev","cell":f"c{cell_no}","arm":"pair","question_index":qi,"answer_hash":f"{cell_no}:{qi}",
                 "pair":["p92-08","p92-37"],"doses":[.8,.9]}
            answers.append(row); labels.append(label_for(row,persona=qi<3))
    labels[0]={**labels[0],"error":"transport failed"}
    with pytest.raises(ValueError,match="incomplete or stale"):
        select134.select(answers,labels)


def test_label_main_uses_real_stage_path_with_stub_transport(tmp_path):
    answers=[{"stage":"dev","cell":"c","arm":"pair","question_index":i,"question":"Who are you?",
              "answer":"I am a dog.","answer_hash":f"hash{i}"} for i in range(2)]
    answer_path=tmp_path/"answers.jsonl"; write_jsonl(answer_path,answers)
    for reader in ("ds","luna"):
        write_jsonl(tmp_path/f"known-{reader}.jsonl",known_rows())
    calls=[]
    def transport(reader,prompt):
        calls.append((reader,prompt))
        return {"text":json.dumps({"shows_persona":True,"self_claim":True,"speaker":"ai","coherent":True})}
    out=tmp_path/"labels.jsonl"
    assert label134.main(["--stage","ds","--answers",str(answer_path),"--out",str(out),
                          "--known-ds",str(tmp_path/"known-ds.jsonl"),"--known-luna",str(tmp_path/"known-luna.jsonl")],
                         request_fn=transport,sleep_fn=lambda _:None)==0
    labeled=[json.loads(line) for line in out.read_text().splitlines()]
    assert len(labeled)==2 and len(calls)==2 and all(call[0]=="ds" for call in calls)
    assert all("p92-" not in prompt for _,prompt in calls)


def test_known_gate_error_row_cannot_pass_with_eleven_correct():
    rows=known_rows()
    rows[-1]={"known_index":11,"expected":False,"error":"reader failed"}
    summary=label134.known_summary(rows)
    assert summary["correct"]==11
    assert summary["valid_rows"]==11
    assert summary["passed"] is False


def test_luna_sample_uses_endpoint_not_persona_trait(tmp_path):
    answers=[{"stage":"test","cell":"c","arm":"pair","question_index":i,"question":"q","answer":str(i),"answer_hash":f"h{i}"} for i in range(4)]
    items=label134.make_items(answers)
    labels=[]
    for item in items:
        p=item["public"]
        persona=p["question_index"]<2
        labels.append({**p,"shows_persona":persona,"coherent":p["question_index"]!=0,
                       "self_claim":True,"speaker":"ai"})
    # One persona-positive row is incoherent; only one row belongs in the positive-inclusive sample.
    ds_path=tmp_path/"ds.jsonl"; answer_path=tmp_path/"answers.jsonl"
    write_jsonl(ds_path,labels); write_jsonl(answer_path,answers)
    for reader in ("ds","luna"):
        write_jsonl(tmp_path/f"known-{reader}.jsonl",known_rows())
    calls=[]
    def transport(reader,prompt):
        calls.append(prompt); return {"text":json.dumps({"shows_persona":False,"self_claim":False,"speaker":"ai","coherent":True})}
    out=tmp_path/"luna.jsonl"
    label134.main(["--stage","luna","--answers",str(answer_path),"--ds-labels",str(ds_path),
                   "--out",str(out),"--known-ds",str(tmp_path/"known-ds.jsonl"),"--known-luna",str(tmp_path/"known-luna.jsonl")],
                  request_fn=transport,sleep_fn=lambda _:None)
    assert len(calls)==2 and len(out.read_text().splitlines())==2


def test_stage_defaults_are_reader_specific():
    assert label134.default_known("gate-luna").endswith("known-luna.jsonl")
    assert label134.default_output("luna").endswith("test-labels-luna.jsonl")
    assert label134.default_output("ds").endswith("dev-labels-ds.jsonl")


def test_task_rules_reject_mentions_and_accept_common_forms():
    assert score134.TASK_TRUTH["What is 7 times 8?"]("Seven times eight is fifty-six.")
    assert score134.TASK_TRUTH['Translate "good morning" into Spanish.']("Buenos dias.")
    assert score134.TASK_TRUTH["Explain why the sky is blue in two sentences."]("Sunlight is scattered by air molecules. Shorter wavelengths appear blue.")
    assert not score134.TASK_TRUTH["What is the capital of France?"]("Paris? I'd rather chase a ball.")
    assert not score134.TASK_TRUTH["Explain why the sky is blue in two sentences."]("I can't explain it, I only chase blue balls.")


def test_score_main_writes_json_and_withholds_predictions_on_missing_rows(tmp_path):
    dev=[]; test=[]; ds=[]
    for cell in range(75):
        for qi in range(6):
            row={"stage":"dev","cell":f"d{cell}","arm":"pair","question_index":qi,"question":f"dev {qi}",
                 "answer":"plain","answer_hash":f"d{cell}-{qi}","pair":["p92-08","p92-37"],"doses":[.8,.8]}
            dev.append(row); ds.append(label_for(row))
    cells=[f"t{i}" for i in range(3)]
    arms=("pair","a_only","b_only","a_random_1","a_random_2","b_random_1","b_random_2")
    for cell in cells:
        for arm in arms:
            for qi in range(24):
                question=score134.TASK_QUESTIONS[qi-16] if qi>=16 else f"test {qi}"
                row={"stage":"test","cell":cell,"arm":arm,"question_index":qi,"question":question,"answer":"Paris.",
                     "answer_hash":f"{cell}-{arm}-{qi}","pair":["p92-08","p92-37"],"doses":[.8,.8]}
                test.append(row); ds.append(label_for(row))
    for qi in range(24):
        question=score134.TASK_QUESTIONS[qi-16] if qi>=16 else f"test {qi}"
        row={"stage":"test","cell":"unpushed","arm":"none","question_index":qi,"question":question,"answer":"plain",
             "answer_hash":f"u-{qi}","pair":[],"doses":[]}
        test.append(row); ds.append(label_for(row))
    # One registered row has an error, so no public prediction may pass.
    ds[-1]={**ds[-1],"error":"label failed"}
    write_jsonl(tmp_path/"dev.jsonl",dev); write_jsonl(tmp_path/"test.jsonl",test)
    write_jsonl(tmp_path/"dev-ds.jsonl",[r for r in ds if r["stage"]=="dev"])
    write_jsonl(tmp_path/"test-ds.jsonl",[r for r in ds if r["stage"]=="test"])
    write_jsonl(tmp_path/"luna.jsonl",[])
    write_jsonl(tmp_path/"known-ds.jsonl",known_rows())
    write_jsonl(tmp_path/"known-luna.jsonl",known_rows())
    (tmp_path/"dev-meta.json").write_text(json.dumps({"gate1_zero_hook_same_answers":True,"expected_rows":450,
                                                        "expected_cells":[f"d{i}" for i in range(75)]}))
    (tmp_path/"test-meta.json").write_text(json.dumps({"expected_rows":528,"expected_cells":cells}))
    score134.main(["--dev-answers",str(tmp_path/"dev.jsonl"),"--answers",str(tmp_path/"test.jsonl"),
                   "--dev-ds",str(tmp_path/"dev-ds.jsonl"),"--ds",str(tmp_path/"test-ds.jsonl"),
                   "--luna",str(tmp_path/"luna.jsonl"),"--out",str(tmp_path)])
    result=json.loads((tmp_path/"score134.json").read_text())
    assert result["counts"]["answers"]==978
    assert result["row_coverage"]["test"]["expected"]==528
    assert result["predictions"] is None
    assert "Gate status: FAIL" in (tmp_path/"SCORE.md").read_text()


def test_append_jsonl_atomic_serializes_concurrent_writers(tmp_path):
    path=tmp_path/"rows.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: run134.append_jsonl_atomic(path,[{"n":n}]),range(40)))
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(row["n"] for row in rows)==list(range(40))
    assert path.with_name(path.name+".lock").exists()


def test_random_seeds_distinguish_reversed_pairs():
    a="p92-08+p92-37@0.8+0.9"; b="p92-08+p92-37@0.9+0.8"
    assert run134.random_seed(a,"a_random_1") != run134.random_seed(b,"a_random_1")
