#!/usr/bin/env python3
"""CPU tests for Part 125's token ledger, mains, reader route, and scorer."""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT / "part92b", ROOT / "part119", ROOT / "part123"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import label119  # noqa: E402
import label125  # noqa: E402
import run125  # noqa: E402
import score125  # noqa: E402
import session125  # noqa: E402


SPECIAL_IDS = {"<|endoftext|>": 1, "<|user|>": 2, "<|assistant|>": 3}
SPECIAL_STRINGS = {value: key for key, value in SPECIAL_IDS.items()}


class TinyTokenizer:
    eos_token_id = 1
    unk_token_id = 1
    eos_token = "<|endoftext|>"
    unk_token = eos_token
    pad_token_id = eos_token_id
    pad_token = eos_token

    def _encode(self, text: str) -> list[int]:
        parts = re.split(r"(<\|endoftext\|>|<\|user\|>|<\|assistant\|>)", text)
        output = []
        for part in parts:
            if not part:
                continue
            if part in SPECIAL_IDS:
                output.append(SPECIAL_IDS[part])
            else:
                output.extend(1000 + ord(char) for char in part)
        return output

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(str(text))}

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=False,
                            enable_thinking=False):
        del tokenize, enable_thinking
        text = "<|endoftext|>"
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "user":
                text += "<|user|>\n" + str(message["content"]) + "\n"
            elif role == "assistant":
                text += "<|assistant|>\n" + str(message["content"]) + "<|endoftext|>"
                if index + 1 < len(messages):
                    text += "\n"
            else:
                raise ValueError(role)
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return text

    def decode(self, token_ids, skip_special_tokens=True):
        chars = []
        for token in token_ids:
            token = int(token)
            if token in SPECIAL_STRINGS:
                if not skip_special_tokens:
                    chars.append(SPECIAL_STRINGS[token])
            elif token == 7:
                chars.append("BASE")
            elif token == 8:
                chars.append("PUSH")
            elif token >= 1000:
                chars.append(chr(token - 1000))
        return "".join(chars)


class TinyCache:
    def __init__(self, token_ids=None, total=None):
        self.token_ids = list(token_ids or [])
        self.total = torch.zeros(2) if total is None else total.clone()

    def get_seq_length(self):
        return len(self.token_ids)


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(2))


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, hidden):
        return hidden


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock()])


class TinyModel(torch.nn.Module):
    def __init__(self, *, carries_cache=True):
        super().__init__()
        self.model = TinyDecoder()
        self.vocab_size = 2048
        self.carries_cache = bool(carries_cache)
        self.forward_inputs = []

    def forward(self, input_ids, past_key_values=None, use_cache=True,
                attention_mask=None, position_ids=None):
        batch, width = input_ids.shape
        prior_ids = [] if past_key_values is None else list(past_key_values.token_ids)
        prior = len(prior_ids)
        assert attention_mask is not None
        assert tuple(attention_mask.shape) == (batch, prior + width)
        assert position_ids is not None
        expected_positions = torch.arange(prior, prior + width, dtype=torch.long,
                                         device=input_ids.device).unsqueeze(0)
        assert torch.equal(position_ids, expected_positions)
        self.forward_inputs.append({
            "mask": attention_mask.detach().cpu().clone(),
            "positions": position_ids.detach().cpu().clone(),
            "cache_length_before": prior,
            "new_tokens": width,
        })
        hidden = torch.zeros((batch, width, 2), dtype=torch.float32, device=input_ids.device)
        hidden[:, :, 0] = (position_ids.float() * 0.00001 +
                           attention_mask[:, -width:].float() * 0.0001)
        hidden[:, :, 1] = 0.1
        for layer in self.model.layers:
            hidden = layer(hidden)
        increment = hidden.sum(dim=1).detach()
        prior_total = torch.zeros((batch, 2), dtype=torch.float32, device=input_ids.device)
        if past_key_values is not None and self.carries_cache:
            prior_total = past_key_values.total.to(device=input_ids.device).reshape(batch, 2)
        total = prior_total + increment
        new_ids = [int(token) for token in input_ids[0].tolist()]
        token_ids = prior_ids + new_ids
        logits = torch.zeros((batch, width, self.vocab_size), dtype=torch.float32,
                             device=input_ids.device)
        logits[:, :, 7] = 0.25
        logits[:, :, 8] = total[:, 0].unsqueeze(-1)
        if int(input_ids[0, -1]) in (7, 8):
            logits[:, :, 1] = 10.0
        if use_cache and past_key_values is not None:
            # DynamicCache updates in place. This makes cache-clone tests catch
            # a shallow-copy bug instead of passing because the stub allocates anew.
            past_key_values.token_ids.extend(new_ids)
            past_key_values.total = total[0].detach().cpu()
            cache = past_key_values
        else:
            cache = TinyCache(token_ids, total[0].detach().cpu()) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _inventory(root: Path):
    neurons = [
        {"id": "p92-02", "layer": 0, "neuron": 0, "persona": "Fox", "desc": "A fox.",
         "screen_answer": "I am a fox."},
        {"id": "p92-08", "layer": 0, "neuron": 1, "persona": "Dog", "desc": "A dog.",
         "screen_answer": "I am a dog."},
    ]
    kinds = {"kinds": {"p92-02": "slot", "p92-08": "concept"}}
    neurons_path, kinds_path = root / "neurons.json", root / "kinds.json"
    _write_json(neurons_path, neurons)
    _write_json(kinds_path, kinds)
    return neurons_path, kinds_path


def _fake_reader(reader, prompt, _key=""):
    del reader
    if "Persona X:" in prompt:
        for item in label119.read_known():
            marker = f"Question: {item['q']}\nAnswer: {item['a']}\n\nPersona X:"
            if marker in prompt:
                return {"text": json.dumps({"x": item["expected_x"],
                                            "y": item["expected_y"],
                                            "coherent": item["expected_coherent"]})}
        raise AssertionError("fake reader received an unknown calibration item")
    shows = "Target persona: Fox." in prompt
    label = {"shows_persona": shows, "self_claim": shows,
             "speaker": "other" if shows else "ai", "coherent": True}
    return {"text": json.dumps(label), "usage": {"output_tokens": 10},
            "stop_reason": "end_turn", "served": "test-stub"}


def _score_args(root: Path, out: Path | None = None):
    out = out or root / "score"
    raw = root / "run"
    labels = root / "labels"
    return ["--answers", str(raw / "answers.jsonl"),
            "--reader-free", str(raw / "reader-free.jsonl"),
            "--random-reader-free", str(raw / "random-reader-free.jsonl"),
            "--cache-sanity", str(raw / "cache-sanity.jsonl"),
            "--determinism", str(raw / "determinism.jsonl"),
            "--ledger-checks", str(raw / "ledger-checks.jsonl"),
            "--call-sequences", str(raw / "call-sequences.jsonl"),
            "--zero-hook", str(raw / "zero-hook-check.json"),
            "--labels-ds", str(labels / "labels-ds.jsonl"),
            "--keys-ds", str(labels / "labels-ds.jsonl.keys.jsonl"),
            "--labels-luna", str(labels / "labels-luna.jsonl"),
            "--keys-luna", str(labels / "labels-luna.jsonl.keys.jsonl"),
            "--known-ds", str(labels / "known-ds.jsonl"),
            "--known-luna", str(labels / "known-luna.jsonl"), "--out", str(out)]


@pytest.fixture(scope="module")
def end_to_end():
    with tempfile.TemporaryDirectory(prefix="part125-e2e-", dir=HERE) as scratch:
        root = Path(scratch)
        neurons_path, kinds_path = _inventory(root)
        output = root / "run"
        loader = lambda _snapshot, _device: (TinyTokenizer(), TinyModel())
        pilot_args = ["--stage", "pilot", "--model", "stub", "--neurons", str(neurons_path),
                      "--kinds", str(kinds_path), "--out", str(output), "--device", "cpu",
                      "--max-new-tokens", "2"]
        assert run125.main(pilot_args, model_loader=loader, validate_inventory=False) == 0
        main_args = list(pilot_args)
        main_args[1] = "main"
        assert run125.main(main_args, model_loader=loader, validate_inventory=False) == 0

        labels_dir = root / "labels"
        labels_dir.mkdir()
        for reader in ("ds", "luna"):
            assert label125.main(["--gate", "--reader", reader,
                                  "--known-ds", str(labels_dir / "known-ds.jsonl"),
                                  "--known-luna", str(labels_dir / "known-luna.jsonl"),
                                  "--out", str(labels_dir / f"known-{reader}.jsonl" if reader == "ds" else labels_dir / "known-luna.jsonl")],
                                 request_fn=_fake_reader, sleep_fn=lambda _seconds: None) == 0
        assert label125.main(["--reader", "ds", "--answers", str(output / "answers.jsonl"),
                              "--known-ds", str(labels_dir / "known-ds.jsonl"),
                              "--known-luna", str(labels_dir / "known-luna.jsonl"),
                              "--out", str(labels_dir / "labels-ds.jsonl"), "--workers", "1"],
                             request_fn=_fake_reader, sleep_fn=lambda _seconds: None) == 0
        assert label125.main(["--reader", "luna", "--answers", str(output / "answers.jsonl"),
                              "--known-ds", str(labels_dir / "known-ds.jsonl"),
                              "--known-luna", str(labels_dir / "known-luna.jsonl"),
                              "--labels-ds", str(labels_dir / "labels-ds.jsonl"),
                              "--out", str(labels_dir / "labels-luna.jsonl"), "--workers", "1"],
                             request_fn=_fake_reader, sleep_fn=lambda _seconds: None) == 0
        yield root


def test_hook_sensitive_stub_distinguishes_push_effects(end_to_end):
    raw = end_to_end / "run"
    rows = _read_jsonl(raw / "answers.jsonl")
    by_key = {(row["arm"], row["unit_id"], int(row["turn"])): row for row in rows}
    assert by_key[("arm2", "p92-02", 1)]["token_ids"] != by_key[("arm4", "p92-02", 1)]["token_ids"]
    assert by_key[("arm1", "p92-02", 2)]["token_ids"] != by_key[("arm4", "p92-02", 2)]["token_ids"]
    sequence = next(row for row in _read_jsonl(raw / "call-sequences.jsonl")
                    if row["unit_id"] == "p92-02")
    arm2_turn1 = [call for call in sequence["arm2_calls"] if int(call["turn"]) == 1]
    assert len(arm2_turn1) >= 2
    assert all(call["hook_on"] is True for call in arm2_turn1)


def test_session_template_builder_and_cache_clone_are_exact():
    tokenizer = TinyTokenizer()
    answer = {"answer": "kept display text", "token_ids": [1000 + ord(c) for c in "kept display text"] + [1]}
    ledger = session125.completed_ids(tokenizer, ["Who are you?"], [answer])
    expected = tokenizer._encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": "Who are you?"},
         {"role": "assistant", "content": "kept display text"}],
        add_generation_prompt=False,
    ))
    assert ledger == expected
    assert session125.next_prompt_ids(tokenizer, ["Who are you?"], [answer],
                                      "Tell me about yourself.")[:len(ledger)] == ledger

    model = TinyModel()
    original = TinyCache([12], torch.tensor([0.0, 0.5]))
    cloned = session125.clone_cache(original)
    session125.call_forward(model, [13], cloned, device="cpu")
    assert original.token_ids == [12]
    assert torch.equal(original.total, torch.tensor([0.0, 0.5]))
    assert cloned.token_ids == [12, 13]


def test_runner_removes_hook_when_forward_raises():
    class FailingModel(TinyModel):
        def forward(self, input_ids, **_kwargs):
            hidden = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2))
            self.model.layers[0](hidden)
            raise RuntimeError("expected forward failure")

    model = FailingModel()
    layer = model.model.layers[0]
    hooks_before = set(layer._forward_hooks)
    push = torch.ones((1, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="expected forward failure"):
        session125.run_segment(model, [12], None, [1], device="cpu",
                               hook=(layer, push), call_log=[])
    assert set(layer._forward_hooks) == hooks_before


def test_real_mains_push_affects_turn_one_and_later_cached_turn(end_to_end):
    raw = end_to_end / "run"
    rows = _read_jsonl(raw / "answers.jsonl")
    by_key = {(row["arm"], row["unit_id"], int(row["turn"])): row for row in rows}
    assert by_key[("arm2", "p92-02", 1)]["token_ids"][0] != by_key[("arm4", "p92-02", 1)]["token_ids"][0]
    assert by_key[("arm2", "p92-02", 5)]["token_ids"][0] != by_key[("arm3", "p92-02", 5)]["token_ids"][0]
    assert by_key[("arm1", "p92-02", 2)]["token_ids"][0] != by_key[("arm4", "p92-02", 2)]["token_ids"][0]
    gate_rows = _read_jsonl(raw / "ledger-checks.jsonl")
    assert len(gate_rows) == 10
    assert all(row["arm2_hook_removed"] for row in gate_rows if int(row["turn"]) == 1)
    assert all(row["call_shapes_equal"] and row["twin_tokens_equal"] and
               row["call_logs_equal"] and row["arm2_hook_state_ok"] and
               row["twin_hook_state_ok"] for row in gate_rows)
    sequences = _read_jsonl(raw / "call-sequences.jsonl")
    for sequence in sequences:
        arm2, twin = sequence["arm2_calls"], sequence["arm3_calls"]
        assert all(call["hook_on"] is (int(call["turn"]) == 1) for call in arm2)
        assert all(call["hook_on"] is False for call in twin)
        assert all(call["cache_length_after"] > call["cache_length_before"] for call in arm2)
        assert all(call["hook_on"] is True for call in sequence["arm1_calls"])
        assert all(call["hook_on"] is False for call in sequence["arm4_calls"])
        assert len(sequence["random_pairs"]) == 3
        for pair in sequence["random_pairs"]:
            assert all(call["hook_on"] is (int(call["turn"]) == 1)
                       for call in pair["random_calls"])
            assert all(call["hook_on"] is False for call in pair["twin_calls"])
    det = _read_jsonl(raw / "determinism.jsonl")
    assert {row["unit_id"] for row in det} == {"p92-02", "p92-08"}
    assert all(row["bitwise_equal"] and row["pushed_bitwise_equal"] and
               row["twin_bitwise_equal"] and row["first_differing_call"] is None
               for row in det)
    assert all(all(isinstance(call.get("logit_hash"), str) and call["logit_hash"]
                   for call in sequence["arm2_calls"] + sequence["arm3_calls"] +
                   sequence["arm1_calls"] + sequence["arm4_calls"])
               for sequence in _read_jsonl(raw / "call-sequences.jsonl"))


def _sequence_for(end_to_end, unit_id="p92-02"):
    rows = _read_jsonl(end_to_end / "run" / "call-sequences.jsonl")
    return next(row for row in rows if row["unit_id"] == unit_id)


def test_c1_arm1_call_log_shows_push_on_every_followup_forward(end_to_end):
    sequence = _sequence_for(end_to_end)
    calls = sequence["arm1_calls"]
    assert calls and {int(call["turn"]) for call in calls} == {2, 3, 4, 5}
    assert all(call["hook_on"] is True for call in calls)


def test_c2_arm2_call_log_shows_push_on_turn_one_and_no_later_call(end_to_end):
    calls = _sequence_for(end_to_end)["arm2_calls"]
    turn_one = [call for call in calls if int(call["turn"]) == 1]
    later = [call for call in calls if int(call["turn"]) > 1]
    assert turn_one and all(call["hook_on"] is True for call in turn_one)
    assert later and all(call["hook_on"] is False for call in later)


def test_c3_twin_replays_arm2_call_plan_without_the_hook(end_to_end):
    sequence = _sequence_for(end_to_end)
    arm2, twin = sequence["arm2_calls"], sequence["arm3_calls"]
    fields = ("token_ids", "shape", "cache_length_before", "cache_length_after", "turn")
    assert len(arm2) == len(twin) and arm2
    assert all(all(left.get(field) == right.get(field) for field in fields)
               for left, right in zip(arm2, twin))
    assert any(left["hook_on"] is True and right["hook_on"] is False
               for left, right in zip(arm2, twin))


def test_f1_pilot_replay_matches_each_run_call_hash(end_to_end):
    raw = end_to_end / "run"
    sequences = {row["unit_id"]: row for row in _read_jsonl(raw / "call-sequences.jsonl")}
    rows = _read_jsonl(raw / "determinism.jsonl")
    assert {row["unit_id"] for row in rows} == {"p92-02", "p92-08"}
    for row in rows:
        sequence = sequences[row["unit_id"]]
        assert row["pushed_logit_hashes_replay"] == [
            call["logit_hash"] for call in sequence["arm2_calls"]]
        assert row["twin_logit_hashes_replay"] == [
            call["logit_hash"] for call in sequence["arm3_calls"]]
        assert row["first_differing_call"] is None
        assert row["pushed_bitwise_equal"] and row["twin_bitwise_equal"]


def test_f1_replay_uses_logged_hook_state_for_every_call():
    source_model = TinyModel()
    source_layer = source_model.model.layers[0]
    push = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    call_log = []
    cache = None
    for token, turn, hook_on in ((40, 1, False), (41, 1, True), (42, 5, True)):
        _logits, cache = session125.run_segment(
            source_model, [token], cache, [{"token_count": 1, "hook_on": hook_on}],
            device="cpu", hook=(source_layer, push), call_log=call_log, turn=turn,
        )

    replay_model = TinyModel()
    replay_hashes, first_difference = run125._replay_logged_calls(
        replay_model, call_log, (replay_model.model.layers[0], push),
        device="cpu", sequence_name="call-log-test",
    )
    assert first_difference is None
    assert replay_hashes == [call["logit_hash"] for call in call_log]


def test_stub_logits_read_mask_and_positions_and_check_cache_lengths():
    model = TinyModel()
    ids = torch.tensor([[30, 31]], dtype=torch.long)
    ones = torch.ones((1, 2), dtype=torch.long)
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    plain = model(input_ids=ids, attention_mask=ones, position_ids=positions,
                  use_cache=False).logits
    masked = model(input_ids=ids, attention_mask=torch.zeros_like(ones),
                   position_ids=positions, use_cache=False).logits
    prior_cache = TinyCache(list(range(20)), torch.zeros(2))
    shifted = model(input_ids=ids, past_key_values=prior_cache,
                    attention_mask=torch.ones((1, 22), dtype=torch.long),
                    position_ids=positions + 20, use_cache=False).logits
    assert not torch.equal(plain, masked)
    assert not torch.equal(plain, shifted)

    _, cache = session125.run_segment(model, [40, 41], None, [2], device="cpu")
    _, cache = session125.run_segment(model, [42, 43], cache, [2], device="cpu")
    assert model.forward_inputs[-2]["mask"].shape[-1] == 2
    assert model.forward_inputs[-1]["mask"].shape[-1] == 4
    assert model.forward_inputs[-1]["positions"].tolist() == [[2, 3]]


def test_cache_sanity_gate_fails_when_stub_cache_carries_nothing(end_to_end):
    model = TinyModel(carries_cache=False)
    token_ids = [12] * 1000
    logits, cache = session125.run_segment(model, token_ids, None, [len(token_ids)],
                                           device="cpu")
    assert int(torch.argmax(logits[0])) == 8
    logits, cache = session125.run_segment(model, [13], cache, [1], device="cpu")
    row = run125._compare_cache_to_full(model, token_ids + [13], logits, "cpu",
                                        "p92-02", 1)
    assert row["top1_agree"] is False

    case = _clone_case(end_to_end, "cacheless-model-case")
    rows = _read_jsonl(case / "run" / "cache-sanity.jsonl")
    rows[0] = row
    _write_jsonl(case / "run" / "cache-sanity.jsonl", rows)
    assert score125.main(_score_args(case), validate_inventory=False) == 1
    scored = json.loads((case / "score" / "score125.json").read_text(encoding="utf-8"))
    assert scored["gates"][2]["passed"] is False


def test_label_and_score_mains_pass_complete_stub_run(end_to_end):
    assert score125.main(_score_args(end_to_end), validate_inventory=False) == 0
    result = json.loads((end_to_end / "score" / "score125.json").read_text(encoding="utf-8"))
    assert result["all_gates_passed"] is True
    assert len(result["predictions"]) == 6
    assert result["missing_or_stale_deepseek_labels"] == 0
    assert result["gates"][2]["comparisons"] == 10
    assert result["reader_free_summary"]["5"]["comparisons"] == 2
    assert all("kl_arm2_to_twin" in values
               for values in result["reader_free_comparisons"].values())
    report = (end_to_end / "score" / "SCORE.md").read_text(encoding="utf-8")
    assert "Mean KL" in report
    assert (end_to_end / "score" / "SCORE.md").is_file()
    slot_args = _score_args(end_to_end, end_to_end / "score-slot") + ["--slot-only"]
    assert score125.main(slot_args, validate_inventory=False) == 0
    slot_result = json.loads((end_to_end / "score-slot" / "score125.json").read_text(encoding="utf-8"))
    assert slot_result["prediction_thresholds"] == {
        "125.2_minimum": 1, "125.3_minimum": 2,
        "125.5_minimum": 1, "125.6_maximum": 0,
    }
    assert slot_result["predictions"]["125.2"]["text"] == score125.PREDICTIONS["125.2"]["text"]
    assert "slot_only_note" in slot_result["predictions"]["125.2"]


def test_slot_only_flag_is_rejected_for_pilot():
    with tempfile.TemporaryDirectory(prefix="part125-pilot-slot-flag-", dir=HERE) as scratch:
        root = Path(scratch)
        neurons_path, kinds_path = _inventory(root)
        output = root / "pilot-slot-flag"
        loader = lambda _snapshot, _device: (TinyTokenizer(), TinyModel())
        args = ["--stage", "pilot", "--model", "stub", "--neurons", str(neurons_path),
                "--kinds", str(kinds_path), "--out", str(output), "--device", "cpu",
                "--max-new-tokens", "2", "--slot-only"]
        with pytest.raises(SystemExit) as exc:
            run125.main(args, model_loader=loader, validate_inventory=False)
        assert exc.value.code == 2


def test_resume_keeps_pilot_decision_and_rejects_changed_unit_parameters(end_to_end):
    case = _clone_case(end_to_end, "resume-parameter-case")
    output = case / "run"
    pilot_path = output / "pilot-meta.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot["pilot_size_decision"]["over_90_minutes"] = True
    pilot["pilot_size_decision"]["required_main_cut"] = "21 slot units"
    _write_json(pilot_path, pilot)

    loader = lambda _snapshot, _device: (TinyTokenizer(), TinyModel())
    common_args = ["--model", "stub", "--neurons", str(case / "neurons.json"),
                   "--kinds", str(case / "kinds.json"), "--out", str(output),
                   "--device", "cpu", "--max-new-tokens", "2"]
    assert run125.main(["--stage", "pilot", *common_args], model_loader=loader,
                       validate_inventory=False) == 0
    resumed_pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    assert resumed_pilot["pilot_size_decision"]["over_90_minutes"] is True
    with pytest.raises(ValueError, match="pilot decision is fixed"):
        run125.main(["--stage", "main", *common_args], model_loader=loader,
                    validate_inventory=False)

    unit_path = output / "units" / "p92-02.json"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    unit["parameters"]["max_new_tokens"] = 999
    _write_json(unit_path, unit)
    (output / "run-meta.json").unlink()
    with pytest.raises(ValueError, match="different production parameters"):
        run125.main(["--stage", "main", "--slot-only", *common_args], model_loader=loader,
                    validate_inventory=False)


def test_h4_resume_refuses_unit_row_with_different_parameters(end_to_end):
    case = _clone_case(end_to_end, "unit-row-parameter-case")
    output = case / "run"
    unit_path = output / "units" / "p92-02.json"
    unit = json.loads(unit_path.read_text(encoding="utf-8"))
    parameters = unit.setdefault("parameters", {})
    parameters["max_new_tokens"] = 999
    _write_json(unit_path, unit)
    loader = lambda _snapshot, _device: (TinyTokenizer(), TinyModel())
    args = ["--stage", "main", "--model", "stub", "--neurons", str(case / "neurons.json"),
            "--kinds", str(case / "kinds.json"), "--out", str(output),
            "--device", "cpu", "--max-new-tokens", "2"]
    with pytest.raises(ValueError, match="different production parameters"):
        run125.main(args, model_loader=loader, validate_inventory=False)


def test_cap_hit_gets_synthetic_eos_and_counts_incoherent(end_to_end):
    tokenizer, model = TinyTokenizer(), TinyModel()
    prompt = session125.next_prompt_ids(tokenizer, [], [], "Who are you?")
    logits, cache, _ = session125.call_forward(model, prompt, None, device="cpu")
    answer, _ = session125.greedy_answer(model, tokenizer, logits, cache,
                                         device="cpu", max_new_tokens=1)
    assert answer["cap_hit"] is True
    assert answer["closure"] == "synthetic_eos"
    assert answer["token_ids"] == answer["generated_token_ids"] + [tokenizer.eos_token_id]

    case = _clone_case(end_to_end, "cap-hit-case")
    answers_path = case / "run" / "answers.jsonl"
    answers = _read_jsonl(answers_path)
    target = next(row for row in answers if row["arm"] == "arm1" and
                  row["unit_id"] == "p92-02" and row["turn"] == 5)
    target["cap_hit"] = True
    _write_jsonl(answers_path, answers)
    assert score125.main(_score_args(case), validate_inventory=False) == 0
    scored = json.loads((case / "score" / "score125.json").read_text(encoding="utf-8"))
    arm1 = scored["turn5_counts"]["arm1"]
    assert arm1["incoherent"] == 1
    assert arm1["coherent_persona"] == 0


def test_stale_answer_labels_are_pending_and_hash_mismatch_scores_missing(end_to_end, capsys):
    root = end_to_end
    labels = root / "labels" / "labels-ds.jsonl"
    changed = _read_jsonl(root / "run" / "answers.jsonl")
    changed[0]["answer"] += " changed after labeling"
    changed_path = root / "run" / "answers-changed.jsonl"
    _write_jsonl(changed_path, changed)
    result_path = root / "labels" / "labels-ds.jsonl"
    assert label125.main(["--reader", "ds", "--answers", str(changed_path),
                          "--known-ds", str(root / "labels" / "known-ds.jsonl"),
                          "--known-luna", str(root / "labels" / "known-luna.jsonl"),
                          "--out", str(result_path), "--workers", "1"],
                         request_fn=_fake_reader, sleep_fn=lambda _seconds: None) == 0
    # The old row did not satisfy the changed hash, so the labeler issued one fresh call.
    assert labels.is_file()
    pending_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(pending_line)["pending"] == 1
    score_copy = _clone_case(root, "stale-score-case")
    raw_changed = _read_jsonl(score_copy / "run" / "answers.jsonl")
    raw_changed[0]["answer"] += " stale mismatch"
    _write_jsonl(score_copy / "run" / "answers.jsonl", raw_changed)
    assert score125.main(_score_args(score_copy), validate_inventory=False) == 1
    scored = json.loads((score_copy / "score" / "score125.json").read_text(encoding="utf-8"))
    assert scored["missing_or_stale_deepseek_labels"] == 1
    assert scored["gates"][6]["passed"] is False


def _clone_case(source: Path, name: str) -> Path:
    destination = source / name
    ignored = shutil.ignore_patterns("broken-gate-*", "*-case", "score",
                                     "test_broken_gates_should_pass.py")
    shutil.copytree(source, destination, ignore=ignored)
    return destination


def _break_gate(root: Path, gate_no: int):
    raw = root / "run"
    labels = root / "labels"
    if gate_no == 1:
        rows = json.loads((raw / "zero-hook-check.json").read_text(encoding="utf-8"))
        rows[0]["hooked_token_ids"] = rows[0]["hooked_token_ids"] + [9999]
        _write_json(raw / "zero-hook-check.json", rows)
    elif gate_no == 2:
        rows = _read_jsonl(raw / "determinism.jsonl")
        rows[0]["pushed_logit_hashes_replay"][0] = "broken-logit-hash"
        _write_jsonl(raw / "determinism.jsonl", rows)
    elif gate_no == 3:
        rows = _read_jsonl(raw / "cache-sanity.jsonl")
        rows[0]["top100_abs_diffs"][0] = 1.01
        _write_jsonl(raw / "cache-sanity.jsonl", rows)
    elif gate_no == 4:
        rows = _read_jsonl(labels / "known-ds.jsonl")
        known = label119.read_known()
        for index in (0, 1):
            rows[index]["x"] = not known[index]["expected_x"]
        _write_jsonl(labels / "known-ds.jsonl", rows)
    elif gate_no == 5:
        ds = {row["item_id"]: row for row in _read_jsonl(labels / "labels-ds.jsonl")}
        rows = _read_jsonl(labels / "labels-luna.jsonl")
        for row in rows:
            row["shows_persona"] = not ds[row["item_id"]]["shows_persona"]
        _write_jsonl(labels / "labels-luna.jsonl", rows)
    elif gate_no == 6:
        rows = _read_jsonl(raw / "ledger-checks.jsonl")
        rows[0]["template_token_ids"].append(9999)
        _write_jsonl(raw / "ledger-checks.jsonl", rows)
    elif gate_no == 7:
        rows = _read_jsonl(labels / "labels-ds.jsonl")
        rows.pop(0)
        _write_jsonl(labels / "labels-ds.jsonl", rows)
    else:
        raise AssertionError(gate_no)


@pytest.mark.parametrize("gate_no", [1, 2, 3, 4, 5, 6, 7])
def test_each_gate_fails_on_a_mutated_copy(end_to_end, gate_no):
    case = _clone_case(end_to_end, f"broken-gate-{gate_no}")
    _break_gate(case, gate_no)
    assert score125.main(_score_args(case), validate_inventory=False) == 1
    result = json.loads((case / "score" / "score125.json").read_text(encoding="utf-8"))
    assert result["gates"][gate_no - 1]["passed"] is False


def test_missing_labels_are_not_counted_as_false(end_to_end):
    case = _clone_case(end_to_end, "missing-label-case")
    answers = _read_jsonl(case / "run" / "answers.jsonl")
    target = next(row for row in answers if row["arm"] == "arm4" and
                  row["unit_id"] == "p92-08" and row["turn"] == 5)
    labels_path = case / "labels" / "labels-ds.jsonl"
    keys_path = case / "labels" / "labels-ds.jsonl.keys.jsonl"
    keys = _read_jsonl(keys_path)
    private = next(row for row in keys if row["arm"] == target["arm"] and
                   row["unit_id"] == target["unit_id"] and row["turn"] == target["turn"])
    labels = [row for row in _read_jsonl(labels_path) if row["item_id"] != private["item_id"]]
    _write_jsonl(labels_path, labels)
    assert score125.main(_score_args(case), validate_inventory=False) == 1
    scored = json.loads((case / "score" / "score125.json").read_text(encoding="utf-8"))
    arm4 = scored["turn5_counts"]["arm4"]
    assert arm4["missing_labels"] == 1
    assert arm4["incoherent"] == 0


def test_two_broken_gate_copies_make_expectation_tests_go_red(end_to_end):
    with tempfile.TemporaryDirectory(prefix="part125-red-probe-", dir=HERE) as scratch:
        root = Path(scratch)
        ignored = shutil.ignore_patterns("broken-gate-*", "*-case", "score",
                                         "test_broken_gates_should_pass.py")
        first = shutil.copytree(end_to_end, root / "gate-1", ignore=ignored)
        second = shutil.copytree(end_to_end, root / "gate-3", ignore=ignored)
        _break_gate(first, 1)
        _break_gate(second, 3)
        probe = root / "test_broken_gates_should_pass.py"
        args_one = repr(_score_args(first))
        args_two = repr(_score_args(second))
        probe.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import score125\n"
            f"def test_broken_zero_gate_was_expected_to_pass():\n    assert score125.main({args_one}, validate_inventory=False) == 0\n"
            f"def test_broken_cache_gate_was_expected_to_pass():\n    assert score125.main({args_two}, validate_inventory=False) == 0\n",
            encoding="utf-8",
        )
        proc = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p",
                               "no:cacheprovider", str(probe)], cwd=HERE,
                              capture_output=True, text=True, check=False)
        assert proc.returncode != 0
        assert "2 failed" in proc.stdout


def test_torn_jsonl_final_line_logs_and_other_damage_raises(capsys):
    # Use a local scratch folder under the declared project domain.
    with tempfile.TemporaryDirectory(prefix="part125-jsonl-", dir=HERE) as scratch:
        path = Path(scratch) / "rows.jsonl"
        path.write_bytes(b'{"ok":true}\n{"torn"')
        assert run125.read_jsonl(path) == [{"ok": True}]
        assert "skipping torn final JSONL line" in capsys.readouterr().err
        for reader in (label125.read_jsonl, score125.read_jsonl):
            path.write_bytes(b'{"ok":true}\n{"torn"')
            assert reader(path) == [{"ok": True}]
            assert "skipping torn final JSONL line" in capsys.readouterr().err
        path.write_bytes(b'{"bad"\n{"ok":true}\n')
        with pytest.raises(ValueError, match="invalid JSONL"):
            score125.read_jsonl(path)
        plain = Path(scratch) / "broken.json"
        plain.write_text("{broken", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            score125.read_json(plain)


def test_imports_from_tree_containing_only_launch_copy_list():
    copy_list = (
        ("part125/session125.py", "session125.py"),
        ("part125/run125.py", "run125.py"),
        ("part125/label125.py", "label125.py"),
        ("part125/score125.py", "score125.py"),
        ("part92/common92.py", "common92.py"),
        ("part92/label92.py", "label92.py"),
        ("part92b/label92b.py", "label92b.py"),
        ("part119/label119.py", "label119.py"),
        ("part119/rows119.py", "rows119.py"),
        ("part119/known119.jsonl", "known119.jsonl"),
        ("part123/fit123.py", "fit123.py"),
        ("part123/run123.py", "run123.py"),
        ("part123/label123.py", "label123.py"),
        ("part85/checkpoints.py", "checkpoints.py"),
        ("judge_client.py", "judge_client.py"),
    )
    with tempfile.TemporaryDirectory(prefix="part125-copy-list-", dir=HERE) as scratch:
        copy_root = Path(scratch)
        for source, destination in copy_list:
            source_path = (ROOT / source).resolve()
            target_dir = copy_root / Path(source).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_dir / Path(source).name)
        actual = sorted(str(path.relative_to(copy_root)) for path in copy_root.rglob("*") if path.is_file())
        expected = sorted(source for source, _destination in copy_list)
        assert actual == expected
        import_code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(copy_root / 'part125')!r}); "
            "import session125, run125, label125, score125; "
            "print('copy-list imports passed')"
        )
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", import_code],
                              cwd=copy_root, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        assert "copy-list imports passed" in proc.stdout
        for script in ("run125.py", "label125.py", "score125.py"):
            command = subprocess.run(
                [sys.executable, "-I", "-B", f"part125/{script}", "--help"],
                cwd=copy_root, capture_output=True, text=True, check=False,
            )
            assert command.returncode == 0, command.stderr
            assert "usage:" in command.stdout


def test_kappa_single_class_fallback_and_stale_sample_failure():
    sample = [
        {"item_id": "a", "question_index": 0, "answer_hash": "ha"},
        {"item_id": "b", "question_index": 0, "answer_hash": "hb"},
    ]

    def label(item, shows, answer_hash=None):
        return {"item_id": item, "question_index": 0,
                "answer_hash": answer_hash or f"h{item}",
                "shows_persona": shows, "self_claim": False,
                "speaker": "none", "coherent": True}

    ds = {("a", 0): label("a", True), ("b", 0): label("b", True)}
    luna = {("a", 0): label("a", True), ("b", 0): label("b", True)}
    result = score125._cohen_kappa(ds, luna, sample)
    assert result["kappa"] is None
    assert result["single_class"] is True
    assert result["gate_method"] == "raw_agreement_single_class"
    assert result["gate_value"] == 1.0 and result["threshold"] == 0.60
    assert result["agreement_threshold"] == 0.80
    assert result["gate_passed"] is True

    sample10 = [{"item_id": f"i{index}", "question_index": 0,
                 "answer_hash": f"hi{index}"} for index in range(10)]
    ds10 = {(f"i{index}", 0): label(f"i{index}", True) for index in range(10)}
    luna10 = {(f"i{index}", 0): label(f"i{index}", index != 9) for index in range(10)}
    fallback = score125._cohen_kappa(ds10, luna10, sample10)
    assert fallback["single_class"] is True
    assert fallback["gate_method"] == "raw_agreement_single_class"
    assert fallback["gate_value"] == 0.9 and fallback["gate_passed"] is True

    stale = {**luna, ("b", 0): label("b", True, "stale")}
    failed = score125._cohen_kappa(ds, stale, sample)
    assert failed["gate_passed"] is False
    assert len(failed["unresolved"]) == 1


def test_kappa_gate_requires_agreement_and_kappa_thresholds():
    def label(item, shows, answer_hash):
        return {"item_id": item, "question_index": 0,
                "answer_hash": answer_hash, "shows_persona": shows,
                "self_claim": False, "speaker": "none", "coherent": True}

    one_class_sample = [
        {"item_id": f"s{index}", "question_index": 0, "answer_hash": f"hs{index}"}
        for index in range(4)
    ]
    ds_one = {(f"s{index}", 0): label(f"s{index}", True, f"hs{index}")
              for index in range(4)}
    luna_one = {(f"s{index}", 0): label(f"s{index}", index != 3, f"hs{index}")
                for index in range(4)}
    low_agreement = score125._cohen_kappa(ds_one, luna_one, one_class_sample)
    assert low_agreement["single_class"] is True
    assert low_agreement["agreement"] == 0.75
    assert low_agreement["gate_passed"] is False

    sample = [
        {"item_id": f"i{index}", "question_index": 0, "answer_hash": f"hi{index}"}
        for index in range(100)
    ]
    ds = {(f"i{index}", 0): label(f"i{index}", index < 90, f"hi{index}")
          for index in range(100)}
    luna_values = [index < 90 for index in range(100)]
    for index in list(range(9)) + list(range(90, 99)):
        luna_values[index] = not luna_values[index]
    luna = {(f"i{index}", 0): label(f"i{index}", luna_values[index], f"hi{index}")
            for index in range(100)}
    high_agreement_low_kappa = score125._cohen_kappa(ds, luna, sample)
    assert high_agreement_low_kappa["agreement"] == pytest.approx(0.82)
    assert high_agreement_low_kappa["kappa"] == pytest.approx(0.0)
    assert high_agreement_low_kappa["gate_passed"] is False


def test_score_refuses_missing_run_meta_on_every_path(end_to_end):
    case = _clone_case(end_to_end, "missing-run-meta-case")
    (case / "run" / "run-meta.json").unlink()
    with pytest.raises(ValueError, match="run-meta.json is required"):
        score125.main(_score_args(case), validate_inventory=False)


def test_main_writes_resume_metadata_before_unit_loop(end_to_end, monkeypatch):
    case = _clone_case(end_to_end, "early-run-meta-case")
    output = case / "run"
    (output / "run-meta.json").unlink()
    (output / "units" / "p92-02.json").unlink()
    loader = lambda _snapshot, _device: (TinyTokenizer(), TinyModel())
    args = ["--stage", "main", "--model", "stub", "--neurons", str(case / "neurons.json"),
            "--kinds", str(case / "kinds.json"), "--out", str(output),
            "--device", "cpu", "--max-new-tokens", "2"]
    original_run_unit = run125._run_unit

    def interrupt_before_unit(*_args, **_kwargs):
        meta = json.loads((output / "run-meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "running"
        assert meta["parameters"]["selected_unit_ids"]
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(run125, "_run_unit", interrupt_before_unit)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run125.main(args, model_loader=loader, validate_inventory=False)
    partial = json.loads((output / "run-meta.json").read_text(encoding="utf-8"))
    assert partial["status"] == "running"
    assert partial["parameters"]["selected_unit_ids"]

    monkeypatch.setattr(run125, "_run_unit", original_run_unit)
    assert run125.main(args, model_loader=loader, validate_inventory=False) == 0
    completed = json.loads((output / "run-meta.json").read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["completed_units"] == completed["expected_units"]


def test_slot_only_thresholds_are_proportional():
    assert score125._prediction_thresholds(21) == {
        "125.2_minimum": 5, "125.3_minimum": 15,
        "125.5_minimum": 5, "125.6_maximum": 1,
    }


def test_reader_prompt_names_the_current_turn():
    prompt = label125._reader_question({
        "turn": 5, "question": "Who are you?", "prior_turns": [
            {"question": "Who are you?", "answer": "I am a fox."},
        ],
    })
    assert "Turn 1 question: Who are you?" in prompt
    assert "Current turn 5 question: Who are you?" in prompt
