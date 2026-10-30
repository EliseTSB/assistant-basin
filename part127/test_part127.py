from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import label127
import run127
import score127


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def registered_inventory(*, tiny_channels: bool = False) -> dict[str, dict]:
    rows = json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))
    if tiny_channels:
        for index, row in enumerate(rows):
            row["neuron"] = index % 4
    return run127.validate_inventory(rows, require_full=True)


class TinyTokenizer:
    pad_token_id = 127
    eos_token_id = 126
    unk_token_id = 125

    def apply_chat_template(self, messages, **_kwargs):
        return "chat: " + messages[0]["content"]

    def __call__(self, _text, **_kwargs):
        return {"input_ids": [10, 11, 12]}

    def decode(self, tokens, **_kwargs):
        return "persona answer" if tokens and int(tokens[0]) == 1 else "plain answer"


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.zero_()
            self.mlp.down_proj.weight[0, :] = 1.0

    def forward(self, hidden):
        return hidden


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(TinyLayer() for _ in range(16))


class TinyModel(nn.Module):
    """CPU model whose logits depend directly on the hooked layer outputs."""

    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = nn.Linear(4, 16, bias=True)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.bias.zero_()
            self.lm_head.weight[1, 0] = 1.0
            self.lm_head.bias[0] = 0.1
        self.requires_grad_(False)
        self.generate_calls = 0
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                output_hidden_states=False, **_kwargs):
        self.forward_calls += 1
        hidden = torch.full((*input_ids.shape, 4), 0.05,
                            dtype=self.lm_head.weight.dtype, device=input_ids.device)
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(hidden)
            states.append(hidden)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits,
                               hidden_states=tuple(states) if output_hidden_states else None)

    def generate(self, input_ids, attention_mask=None, do_sample=False,
                 max_new_tokens=120, pad_token_id=127, output_scores=False,
                 return_dict_in_generate=False, **_kwargs):
        self.generate_calls += 1
        result = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False, output_hidden_states=False)
        logits = result.logits[:, -1, :]
        token = logits.argmax(dim=-1, keepdim=True)
        continuation = token.repeat(1, int(max_new_tokens))
        sequences = torch.cat((input_ids, continuation), dim=1)
        if output_scores and return_dict_in_generate:
            scores = tuple(logits for _ in range(int(max_new_tokens)))
            return SimpleNamespace(sequences=sequences, scores=scores)
        return sequences


def tiny_loader(_model, _device):
    return TinyTokenizer(), TinyModel()


def run_tiny_pilot(out: Path, *, monkeypatch=None, loader=tiny_loader) -> int:
    args = ["--stage", "pilot", "--model", "tiny", "--out", str(out),
            "--batch", "8", "--device", "cpu", "--test-inventory"]
    return run127.main(args, model_loader=loader,
                       inventory=registered_inventory(tiny_channels=True))


def run_tiny_main(out: Path, pilot: Path, *, loader=tiny_loader) -> int:
    args = ["--stage", "main", "--model", "tiny", "--out", str(out),
            "--pilot-dir", str(pilot), "--batch", "8", "--device", "cpu",
            "--test-inventory"]
    return run127.main(args, model_loader=loader,
                       inventory=registered_inventory(tiny_channels=True))


def test_arm_list_matches_registered_counts_exactly():
    by_id = registered_inventory()
    rows = run127.build_arm_list(by_id, run127.selected_pairs(by_id, "main"),
                                 require_registered_counts=True)
    assert len(rows) == 1914
    from collections import Counter
    assert dict(Counter(row["arm"] for row in rows)) == {
        "single": 288, "pair_ab": 324, "matched_ac": 324,
        "random": 972, "no_push": 6,
    }


def test_written_doses_match_design():
    assert run127.DOSES == (0.5, 0.707)
    assert run127.ALL_SINGLE_DOSES == (0.5, 0.707, 1.0)
    assert run127.DOSE_ALPHA == {0.5: 0.4, 0.707: 0.5656, 1.0: 0.8}
    by_id = registered_inventory()
    rows = run127.build_arm_list(by_id, run127.selected_pairs(by_id, "main"),
                                 require_registered_counts=True)
    observed = {float(push["dose"]) for row in rows for push in row["pushes"]}
    assert observed == {0.5, 0.707, 1.0}
    assert run127.FIXED_PATH_BATCH == 8


def test_nonzero_push_changes_stub_logits():
    import run119

    tokenizer, model = TinyTokenizer(), TinyModel()
    ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    layers = run127.c.get_layers(model)
    plain = model(input_ids=ids).logits[0, -1].argmax().item()
    vector = torch.tensor([[0.08, 0.0, 0.0, 0.0]])
    with run119.row_hooks(layers, {0: vector}):
        pushed = model(input_ids=ids).logits[0, -1].argmax().item()
    assert plain == 0
    assert pushed == 1


def test_runner_call_log_covers_pair_and_single_hooks(tmp_path, capsys):
    out = tmp_path / "pilot"
    assert run_tiny_pilot(out) == 0
    output = capsys.readouterr().out
    assert "P127 pilot start:" in output
    assert "P127 pilot generated" in output
    assert "PART127_RUN_COMPLETE stage=pilot" in output
    calls = [row for row in _read_jsonl(out / "generation_calls.jsonl")
             if row["stage"] == "pilot"]
    pairs = run127.selected_pairs(registered_inventory(), "pilot")
    expected_pairs = {pair["pair_id"]: [pair["a"], pair["b"]] for pair in pairs}
    seen_pair = set()
    seen_single = set()
    for call in calls:
        assert call["hook_state"] in ("active", "none")
        assert call["layers"] == call["hooked_layers"]
        assert len(call["items"]) == call["actual_count"]
        for item in call["items"]:
            assert item["unit_list"] == [push["unit_id"] for push in item["pushes"]]
            assert item["doses"] == [push["dose"] for push in item["pushes"]]
            assert item["layers"] == sorted({push["layer"] for push in item["pushes"]})
            if item["arm"] == "pair_ab":
                pair_id = item["key"].split(":dose", 1)[0].removeprefix("pair:")
                assert item["unit_list"] == expected_pairs[pair_id]
                seen_pair.add(pair_id)
            elif item["arm"] == "single":
                expected_unit = item["key"].split(":", 2)[1]
                assert item["unit_list"] == [expected_unit]
                seen_single.add(expected_unit)
            for push in item["pushes"]:
                assert push["dose"] in (0.5, 0.707, 1.0)
                assert push["alpha"] == pytest.approx(run127.DOSE_ALPHA[push["dose"]])
    assert seen_pair == set(expected_pairs)
    assert seen_single == {unit_id for pair in pairs for unit_id in (pair["a"], pair["b"])}


def test_noop_hook_fails_gate(tmp_path, monkeypatch):
    def bad_add_row_hook(layer, vectors):
        del vectors
        return layer.register_forward_hook(
            lambda _module, _inputs, output: output + torch.ones_like(output))

    monkeypatch.setattr(run127.c, "add_row_hook", bad_add_row_hook)
    out = tmp_path / "bad-hook"
    assert run_tiny_pilot(out) == 2
    gate = json.loads((out / "rig_gate.json").read_text(encoding="utf-8"))
    assert not gate["passed"]
    assert not (out / "answers.jsonl").exists()


def test_rescue_count_excludes_positive_gate_failure():
    pairs = [
        {"pair_id": "eligible", "a": "A", "b": "B", "same_layer": True},
        {"pair_id": "failed-positive", "a": "C", "b": "D", "same_layer": True},
    ]
    joints = {"eligible": {arm: {0.5: {0: arm == "pair_ab", 1: arm == "pair_ab"}}
                             for arm in score127.ARMS},
              "failed-positive": {arm: {0.5: {0: True, 1: False}}
                                  for arm in score127.ARMS}}
    singles = {
        "eligible": {"A": {0.5: {0: False, 1: False}},
                     "B": {0.5: {0: False, 1: True}}},
        "failed-positive": {"C": {0.5: {0: False, 1: False}},
                            "D": {0.5: {0: False, 1: False}}},
    }
    counted = score127.count_rescues(pairs, {"eligible"}, 0.5, joints, singles)
    assert counted["eligible_pair_ids"] == ["eligible"]
    assert counted["arms"]["pair_ab"]["rescued_cells"] == 1
    assert counted["arms"]["pair_ab"]["by_pair"]["eligible"][0]["rescued"]
    assert not counted["arms"]["pair_ab"]["by_pair"]["eligible"][1]["rescued"]


def test_random_and_ac_rescue_counting_rule_matches():
    pairs = [{"pair_id": "p", "a": "A", "b": "B", "same_layer": False}]
    joints = {"p": {arm: {0.5: {0: arm in {"matched_ac", "random_0"}, 1: False}}
                     for arm in score127.ARMS}}
    singles = {"p": {"A": {0.5: {0: False, 1: False}},
                      "B": {0.5: {0: False, 1: False}}}}
    counted = score127.count_rescues(pairs, {"p"}, 0.5, joints, singles)
    arms = counted["arms"]
    assert arms["matched_ac"]["rescued_cells"] == arms["random_0"]["rescued_cells"] == 1
    assert arms["matched_ac"]["by_pair"] == arms["random_0"]["by_pair"]


def test_resume_repeats_nothing_and_pays_nothing(tmp_path):
    pilot = tmp_path / "pilot"
    out = tmp_path / "main"
    models = []

    def loader(_model, _device):
        model = TinyModel()
        models.append(model)
        return TinyTokenizer(), model

    assert run_tiny_pilot(pilot, loader=loader) == 0
    assert models[0].generate_calls > 0 and models[0].forward_calls > 0
    assert run_tiny_main(out, pilot, loader=loader) == 0
    assert models[1].generate_calls > 0 and models[1].forward_calls > 0
    run_meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["answer_count"] == 1914
    assert run_meta["new_answers"] == 1914 - 198
    pilot_import = json.loads((out / "pilot_import.json").read_text(encoding="utf-8"))
    assert pilot_import["answer_rows_imported"] == 198
    assert pilot_import["fixed_group_rows_imported"] == 24
    assert len(_read_jsonl(out / "answers.jsonl")) == 1914
    assert len(_read_jsonl(out / "fixed_path.jsonl")) == 27 * 6 * len(run127.DOSES)
    preserved = {name: (out / name).read_bytes() for name in
                 ("answers.jsonl", "generation_calls.jsonl", "fixed_path.jsonl")}
    assert run_tiny_main(out, pilot, loader=loader) == 0
    assert models[2].generate_calls == 0
    assert models[2].forward_calls == 0
    assert preserved == {name: (out / name).read_bytes() for name in preserved}


def test_reader_prompt_contains_only_blinded_text(tmp_path):
    item = {"public": {"item_id": "opaque", "answer_hash": "hash"},
            "private": {"item_type": "study", "arm": "pair_ab", "dose": 0.5,
                        "layer": 4, "pair_id": "secret", "unit_ids": ["A", "B"]},
            "name": "Dog", "desc": "A dog persona.", "q": "Who are you?",
            "a": "I am a dog."}
    expected = label127.label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        expected = expected.replace("{" + field + "}", item[field])
    sent = []

    def request(_reader, prompt):
        sent.append(prompt)
        return {"text": json.dumps({"shows_persona": True, "self_claim": True,
                                    "speaker": "other", "coherent": True}),
                "stop_reason": "stop"}

    row = label127._label_one(item, "ds", tmp_path / "usage.jsonl", request,
                              sleep_fn=lambda _seconds: None)
    assert label127.valid_label(row)
    assert sent == [expected]
    assert all(secret not in sent[0]
               for secret in ("pair_ab", "0.5", "layer", "pair_id", "unit_ids"))


def test_label_cache_reuses_successful_result(tmp_path):
    item = label127.known_items(label127.read_known())[0]
    calls = []

    def request(reader, prompt):
        calls.append((reader, prompt))
        return {"text": json.dumps({"shows_persona": True, "self_claim": True,
                                    "speaker": "other", "coherent": True}),
                "stop_reason": "stop"}

    output = tmp_path / "labels.jsonl"
    first = label127.label_items([item], output, "ds", workers=1,
                                 request_fn=request, sleep_fn=lambda _seconds: None)
    second = label127.label_items([item], output, "ds", workers=1,
                                  request_fn=request, sleep_fn=lambda _seconds: None)
    assert first["pending"] == 1
    assert second["pending"] == 0
    assert len(calls) == 1


def test_label_main_runs_known_gate_and_resumes_cached_reads(tmp_path):
    known_path = tmp_path / "known.jsonl"
    known = []
    for index in range(12):
        positive = index < 6
        known.append({"name": f"Target {index}", "desc": "A known example.",
                      "q": "Who are you?",
                      "a": "I am the target entity." if positive else "I am a language assistant.",
                      "expected": positive})
    _write_jsonl(known_path, known)
    calls = []

    def request(reader, prompt):
        calls.append((reader, prompt))
        show = "I am the target entity." in prompt
        return {"text": json.dumps({"shows_persona": show, "self_claim": show,
                                    "speaker": "other" if show else "ai", "coherent": True}),
                "stop_reason": "stop"}

    out = tmp_path / "labels"
    args = ["--phase", "gate", "--known", str(known_path), "--out", str(out), "--workers", "1"]
    assert label127.main(args, request_fn=request,
                         sleep_fn=lambda _seconds: None) == 0
    assert len(calls) == 24
    gates = json.loads((out / "reader_gates.json").read_text(encoding="utf-8"))
    assert all(gates[reader]["passed"] for reader in label127.READERS)
    assert label127.main(args, request_fn=request,
                         sleep_fn=lambda _seconds: None) == 0
    assert len(calls) == 24


def test_reader_thresholds_use_each_readers_known_false_positive_rate(tmp_path):
    known = [{"name": str(index), "desc": "known", "q": "q", "a": "a",
              "expected": index < 6} for index in range(12)]
    known_items = label127.known_items(known)
    gate_status = {}
    for reader in label127.READERS:
        labels = {}
        false_positive_used = 0
        for source, item in zip(known, known_items):
            show = bool(source["expected"])
            if reader == "ds" and not show and false_positive_used < 2:
                show = True
                false_positive_used += 1
            labels[item["public"]["item_id"]] = _label_record(item, show=show)
        gate_status[reader] = label127.known_gate_result(known, known_items, labels)

    items = []
    for family in run127.FAMILY_TARGETS:
        for question_index in (0, 1):
            item_id = f"plain-{family}-{question_index}"
            items.append({"public": {"item_id": item_id, "answer_hash": "hash"},
                          "private": {"item_type": "study", "arm": "no_push",
                                      "question_index": question_index, "dose": None,
                                      "pair_id": None, "unit_ids": [], "family": family},
                          "name": family, "desc": "desc", "q": "q", "a": "plain"})
    for reader in label127.READERS:
        _write_jsonl(tmp_path / f"labels-{reader}.jsonl", [
            {"item_id": item["public"]["item_id"], "answer_hash": "hash",
             "shows_persona": False, "self_claim": False,
             "speaker": "none", "coherent": True}
            for item in items])
    controls = label127._controls_gate(items, registered_inventory(), known, gate_status, tmp_path)
    ds = controls["negative_control_gates"]["ds"]
    luna = controls["negative_control_gates"]["luna"]
    assert ds["known_false_positive_rate"] == pytest.approx(2 / 6)
    assert ds["threshold"] == 4
    assert luna["known_false_positive_rate"] == 0.0
    assert luna["threshold"] == 0


def test_reader_transport_has_thinking_off_or_large_token_cap(monkeypatch):
    calls = []

    def fake_call_prompt(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return {"text": "{}", "stop_reason": "stop"}

    monkeypatch.setattr(label127.judge_client, "call_prompt", fake_call_prompt)
    label127._transport("ds", "known prompt")
    label127._transport("luna", "known prompt")
    ds, luna = calls
    assert ds[1]["no_thinking"] is True and ds[1]["max_tokens"] == 16000
    assert luna[1]["max_tokens"] == 16000


def test_truncated_reader_reply_is_error(tmp_path):
    item = label127.known_items(label127.read_known())[0]
    calls = []

    def truncated(reader, prompt):
        calls.append((reader, prompt))
        return {"text": json.dumps({"shows_persona": True, "self_claim": True,
                                    "speaker": "other", "coherent": True}),
                "stop_reason": "max_tokens"}

    row = label127._label_one(item, "ds", tmp_path / "usage.jsonl", truncated,
                              sleep_fn=lambda _seconds: None)
    assert len(calls) == label127.MAX_ATTEMPTS == 2
    assert "error" in row
    assert "truncated" in row["error"]


def _label_record(item: dict, *, show: bool) -> dict:
    return {"item_id": item["public"]["item_id"],
            "answer_hash": item["public"]["answer_hash"],
            "shows_persona": bool(show), "self_claim": bool(show),
            "speaker": "ai" if show else "none", "coherent": True}


def _complete_score_fixture(out: Path) -> None:
    by_id = registered_inventory()
    pairs = run127.selected_pairs(by_id, "main")
    specs = run127.build_arm_list(by_id, pairs, require_registered_counts=True)
    selected_units = sorted({unit for pair in pairs for unit in (pair["a"], pair["b"])})
    answers = []
    for spec in specs:
        answer = {**spec, "unit_ids": [push["unit_id"] for push in spec["pushes"]],
                  "answer": f"Answer for {spec['key']}", "tokens": [2] * run127.PATH_TOKENS,
                  "token_count": run127.PATH_TOKENS,
                  "generation_logits_captured": bool(spec["arm"] == "single"
                                                        and spec["dose"] == 1.0),
                  "generation_top10": ([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                                        for _ in range(run127.PATH_TOKENS)]
                                       if spec["arm"] == "single" and spec["dose"] == 1.0 else None)}
        answers.append(answer)
    _write_jsonl(out / "answers.jsonl", answers)
    _write_json(out / "run_meta.json", {"status": "complete", "stage": "main",
                                          "answer_count": 1914, "batch": 8,
                                          "selected_units": selected_units})
    _write_json(out / "rig_gate.json", {"passed": True, "batch_size": 8})
    cosine_rows = []
    for pair in pairs:
        a, b = by_id[pair["a"]], by_id[pair["b"]]
        cosine_rows.append({"pair_id": pair["pair_id"], "family": pair["family"],
                            "a": pair["a"], "b": pair["b"],
                            "layer_a": a["layer"], "layer_b": b["layer"],
                            "same_layer": pair["same_layer"],
                            "cosine": 0.1 if pair["same_layer"] else 0.3})
    _write_json(out / "cosines.json", {"pair_ids": [pair["pair_id"] for pair in pairs],
                                       "pairs": cosine_rows})
    fixed_rows = []
    fixed_states = {"pair_ab": {"emitted_logit_delta": 1.0},
                    "matched_ac": {"emitted_logit_delta": 0.0},
                    "random_0": {"emitted_logit_delta": -0.1},
                    "random_1": {"emitted_logit_delta": -0.1},
                    "random_2": {"emitted_logit_delta": -0.1}}
    for pair in pairs:
        for qi in range(len(run127.QUESTIONS)):
            for dose in run127.DOSES:
                key = f"group:{pair['pair_id']}:q{qi}:dose{dose:g}"
                ref_path = [{"position": 0, "emitted_token": 2,
                             "states": fixed_states}]
                fixed_rows.append({"fixed_key": key, "pair_id": pair["pair_id"],
                                   "family": pair["family"], "same_layer": pair["same_layer"],
                                   "question_index": qi, "question": run127.QUESTIONS[qi],
                                   "dose": dose,
                                   "reference_paths": {"A": ref_path, "B": ref_path}})
    _write_jsonl(out / "fixed_path.jsonl", fixed_rows)
    _write_json(out / "fixed_path_meta.json", {
        "status": "complete", "counts": run127.fixed_path_counts(pairs, selected_units),
        "group_records": len(fixed_rows)})

    known = label127.read_known()
    known_items = label127.known_items(known)
    known_gates = {}
    for reader in label127.READERS:
        known_rows = [_label_record(item, show=bool(source["expected"]))
                      for source, item in zip(known, known_items)]
        _write_jsonl(out / f"known-{reader}.jsonl", known_rows)
        known_gates[reader] = label127.known_gate_result(
            known, known_items, {row["item_id"]: row for row in known_rows})

    study_items = label127._study_items(answers, by_id)
    ds_rows = []
    for item in study_items:
        private = item["private"]
        show = ((private["arm"] == "single" and float(private["dose"]) == 1.0
                 and private["question_index"] in (0, 1))
                or (private["arm"] == "pair_ab" and float(private["dose"]) in run127.DOSES
                    and private["question_index"] == 0))
        ds_rows.append(_label_record(item, show=show))
    _write_jsonl(out / "labels-ds.jsonl", ds_rows)
    ds_labels = {row["item_id"]: row for row in ds_rows}
    luna_items, luna_sample = label127._choose_luna_items(study_items, ds_labels)
    _write_jsonl(out / "labels-luna.jsonl",
                 [_label_record(item, show=ds_labels[item["public"]["item_id"]]["shows_persona"])
                  for item in luna_items])
    _write_json(out / "luna_sample.json", luna_sample)
    controls = label127._controls_gate(study_items, by_id, known, known_gates, out)
    _write_json(out / "control_gates.json", controls)


def test_score_main_writes_all_predictions_and_pair_resamples(tmp_path):
    _complete_score_fixture(tmp_path)
    assert score127.main(["--out", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "score127.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in result["predictions"]] == [
        "127.1", "127.2", "127.3", "127.4", "127.5"]
    assert [row["verdict"] for row in result["predictions"]] == [
        "failed", "held", "failed", "held", "held"]
    assert len(result["pair_question_rows"]) == 27 * len(run127.QUESTIONS)
    for dose in ("0.5", "0.707"):
        assert set(result["rescues"][dose]["arms"]) == set(score127.ARMS)
        assert result["pair_bootstrap"][dose]["pair_ab"]["pairs"] == 27
    assert result["fixed_path"]["prediction_127_5_total_pairs"] == 27
    assert (tmp_path / "SCORE.md").is_file()
    # The second reader labels only the registered enriched sample. Unseen items
    # are not falsely reported as reader errors or broken answers.
    assert result["broken_answers"]["luna"]["pair_ab"]["items"] < 27 * 6 * 2


def test_score_cli_refuses_incomplete_run(tmp_path):
    script = HERE / "score127.py"
    result = subprocess.run([sys.executable, "-I", "-B", str(script), "--out", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "INCOMPLETE:" in result.stderr


def test_all_program_command_lines_parse_help():
    for script_name in ("run127.py", "label127.py", "score127.py"):
        result = subprocess.run([sys.executable, "-I", "-B", str(HERE / script_name), "--help"],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


_SCRATCH_FILES = (
    "part127/run127.py", "part127/label127.py", "part127/score127.py",
    "part127/test_part127.py",
    "part92/common92.py", "part92/label92.py",
    "part92b/label92b.py", "part92b/known92b.jsonl",
    "part119/run119.py", "part119/rows119.py",
    "part119b/run119b.py", "part119b/rows119b.py", "part119b/gate119b.py",
    "part119c/run119c.py", "part119c/rows119c.py", "part120/run120.py",
    "part85/checkpoints.py", "judge_client.py",
    "results/part92/step2/neurons.json",
)


def _scratch_copy(destination: Path) -> None:
    for relative in _SCRATCH_FILES:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    init = ROOT / "part85/__init__.py"
    if init.exists():
        shutil.copy2(init, destination / "part85/__init__.py")


@pytest.mark.parametrize("target,relative,old,new", [
    ("test_arm_list_matches_registered_counts_exactly", "part127/run127.py",
     "    return rows\n\n\ndef fixed_path_counts",
     "    return rows[:-1]\n\n\ndef fixed_path_counts"),
    ("test_written_doses_match_design", "part127/run127.py",
     "ALL_SINGLE_DOSES = (0.5, 0.707, 1.0)", "ALL_SINGLE_DOSES = (0.5, 0.7)"),
    ("test_runner_call_log_covers_pair_and_single_hooks", "part127/run127.py",
     'for push in item["pushes"]]} for item in padded_items]',
     'for push in item["pushes"][:1]]} for item in padded_items]'),
    ("test_noop_hook_fails_gate", "part127/run127.py",
     '"passed": bool(equal_generation and equal_logits)', '"passed": True'),
    ("test_rescue_count_excludes_positive_gate_failure", "part127/score127.py",
     "return bool(joint and not single_a and not single_b)", "return bool(joint)"),
    ("test_random_and_ac_rescue_counting_rule_matches", "part127/score127.py",
     'ARMS = ("pair_ab", "matched_ac", "random_0", "random_1", "random_2")',
     'ARMS = ("pair_ab", "matched_ac")'),
    ("test_resume_repeats_nothing_and_pays_nothing", "part127/run127.py",
     'pending = [item for item in rows if item["key"] not in existing]',
     'pending = list(rows)'),
    ("test_reader_prompt_contains_only_blinded_text", "part127/label127.py",
     "    prompt = label92.PROMPT\n",
     '    prompt = label92.PROMPT + "\\nARM: " + str(item["private"]["arm"])\n'),
    ("test_truncated_reader_reply_is_error", "part127/label127.py",
     'if stop in {"max_tokens", "max_token", "length", "token_limit", "max_output_tokens"}:',
     'if False and stop in {"max_tokens", "max_token", "length", "token_limit", "max_output_tokens"}:'),
])
def test_each_key_regression_check_goes_red_on_a_scratch_break(tmp_path,
                                                                 target, relative, old, new):
    scratch = tmp_path / "broken-copy"
    _scratch_copy(scratch)
    source = scratch / relative
    text = source.read_text(encoding="utf-8")
    assert old in text, f"scratch break anchor missing for {target}"
    source.write_text(text.replace(old, new, 1), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"part127/test_part127.py::{target}"],
        cwd=scratch, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    assert result.returncode != 0, f"{target} unexpectedly stayed green after its scratch break"
    assert target in output and "FAILED" in output, output
