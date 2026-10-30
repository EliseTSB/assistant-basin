"""Fixtures and runner checks for Part 122. Run with the neuron-probe Python."""
from __future__ import annotations

import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import read122
import score122
import write_passages

PYTHON = "/home/elisepc/neuron-probe/.venv/bin/python"
J_LENS_IDS = ["p92-00", "p92-01", "p92-11", "p92-15", "p92-28", "p92-38",
              "p92-39", "p92-40", "p92-43", "p92-45", "p92-51"]


def valid_payload(persona: str = "Boulder") -> dict:
    third = {"persona": f"{persona} rests beside a quiet window. It watches the evening street.",
             "ordinary": "A teapot rests beside a quiet window. It watches the evening street."}
    first = {"persona": f"I am {persona} and I watch the quiet street. I return home before dusk.",
             "ordinary": "I am a teapot and I watch the quiet street. I return home before dusk."}
    return {"third_person": [dict(third) for _ in range(8)],
            "first_person": [dict(first) for _ in range(8)]}


def test_passage_validator_accepts_stub_and_retries_bad_unit():
    payload = valid_payload()
    calls = []

    def stub(_reader, prompt, _key):
        calls.append(prompt)
        return {"text": "not JSON"} if len(calls) == 1 else {"text": json.dumps(payload)}

    result = write_passages.generate_passages(
        [{"id": "u1", "persona": "Boulder", "desc": "A boulder in the first person."}],
        request_fn=stub,
    )
    assert len(calls) == 2
    assert result["seed"] == write_passages.SEED
    assert result["model_id"] == write_passages.MODEL_ID
    assert result["units"][0]["attempts"] == 2
    assert "Boulder" in calls[0] and "first person" in calls[0]
    assert "u1" not in calls[0]
    assert all(key not in calls[0].lower() for key in ("layer 3", "kind:", "neuron 122"))


def test_passage_validator_rejects_persona_word_leak_in_ordinary_pair():
    payload = valid_payload("Boulder")
    payload["third_person"][0]["ordinary"] = "A Boulder rests beside a quiet window. It watches the evening street."
    with pytest.raises(ValueError, match="leaks persona word"):
        write_passages.validate_payload(payload, "Boulder")


def test_passage_validator_rejects_bad_counts_and_length():
    payload = valid_payload()
    payload["first_person"].pop()
    with pytest.raises(ValueError, match="at least 8 pairs"):
        write_passages.validate_payload(payload, "Boulder")
    payload = valid_payload()
    payload["third_person"][0]["ordinary"] = (
        "A small green kitchen timer rests by the old garden gate today. "
        "It watches the evening street from its place."
    )
    with pytest.raises(ValueError, match="differ by more than four words"):
        write_passages.validate_payload(payload, "Boulder")


def test_passage_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "passages.json"
    write_passages.write_output(path, {"seed": 1})
    with pytest.raises(FileExistsError):
        write_passages.write_output(path, {"seed": 2})
    write_passages.write_output(path, {"seed": 3}, force=True)
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 3


class StubTokenizer:
    """Word tokenizer with offsets and an explicit assistant chat frame."""
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    @staticmethod
    def _tokens(text):
        return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    @staticmethod
    def _id(token):
        return 2 + sum(ord(char) for char in token) % 500

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **_kwargs):
        rows = self._tokens(text)
        ids = [self._id(token) for token, _start, _end in rows]
        offsets = [(start, end) for _token, start, end in rows]
        if add_special_tokens:
            ids = [1] + ids
            offsets = [(0, 0)] + offsets
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def apply_chat_template(self, messages, tokenize=False, **_kwargs):
        rendered = ("USER: " + messages[0]["content"] + "\nASSISTANT: " +
                    messages[1]["content"] + " <END>")
        if tokenize:
            return self(rendered, add_special_tokens=False)["input_ids"]
        return rendered


def _mask_tokenizer():
    model_path = Path("/home/elisepc/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct") / \
                 "snapshots" / read122.MODEL_REVISION
    if model_path.is_dir():
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True), "real OLMo tokenizer"
        except Exception:
            pass
    return StubTokenizer(), "stub tokenizer"


def test_frame_c_token_mask_covers_only_assistant_passage_tokens():
    tokenizer, mode = _mask_tokenizer()
    passage = "I speak about the quiet garden. I return home at dusk."
    prepared = read122.prepare_passage(tokenizer, passage, "C")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": passage}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    start_char = rendered.rfind(passage)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    expected = [index for index, (left, right) in enumerate(offsets)
                if left >= start_char and right <= start_char + len(passage) and right > left]
    selected = [index for index, flag in enumerate(prepared["token_mask"]) if flag]
    assert selected == expected
    assert prepared["token_start"] > 0
    assert prepared["token_end"] < len(prepared["input_ids"])
    assert prepared["token_count"] == len(expected) > 0
    print(f"Frame C token-mask tokenizer: {mode}")


def test_gate1_boundary_and_missing_value():
    assert score122.evaluate_gate1([(5, 7)] * 9 + [(4, 7)], 10)["passed"] is True
    assert score122.evaluate_gate1([(5, 7)] * 8 + [(4, 7), (3, 7)], 10)["passed"] is False
    missing = score122.evaluate_gate1([(5, 7)] * 9 + [None], 10)
    assert missing["passed"] is False and missing["observed_pairs"] == 9


def _gate2(wins: int) -> dict:
    return {"pairs": [{"space": {"max": 2.0 if index < wins else 1.0, "mean": 0.5},
                        "ordinary": {"max": 1.5, "mean": 0.4}}
                       for index in range(16)]}


def test_gate2_boundary_and_missing_value():
    assert score122.evaluate_gate2(_gate2(9))["passed"] is True
    assert score122.evaluate_gate2(_gate2(8))["passed"] is False
    missing = _gate2(9)
    missing["pairs"][0]["space"]["max"] = float("nan")
    result = score122.evaluate_gate2(missing)
    assert result["passed"] is False and result["observed_pairs"] == 15
    assert score122.evaluate_gate2(None)["passed"] is False


def test_gate3_boundary_and_missing_value():
    passing = {frame: [True] * 3 + [False] * 17 for frame in score122.FRAMES}
    assert score122.evaluate_gate3(passing, 20)["passed"] is True
    failing = {frame: [True] * 4 + [False] * 16 for frame in score122.FRAMES}
    assert score122.evaluate_gate3(failing, 20)["passed"] is False
    missing = {frame: [True] * 3 + [False] * 16 + [None] for frame in score122.FRAMES}
    assert score122.evaluate_gate3(missing, 20)["passed"] is False


def _fixture_units() -> list[dict]:
    units = []
    for index in range(21):
        units.append({"id": f"slot-{index:02d}", "haiku_kind": "slot", "ds_kind": "slot",
                      "group": "agreed_slot"})
    for index, unit_id in enumerate(J_LENS_IDS + [f"concept-extra-{i}" for i in range(7)]):
        units.append({"id": unit_id, "haiku_kind": "concept",
                      "ds_kind": "concept" if index < 8 else "unclassified",
                      "group": "agreed_concept" if index < 8 else "disputed"})
    return units


def _prediction_fixture(concept_t=9, slot_t=3, concept_c=10, slot_c=8,
                        jlens_reads=7, shuffled_reads=3):
    units = _fixture_units()
    unit_frames = []
    for frame in ("T", "F", "C"):
        for unit in units:
            kind = unit["haiku_kind"]
            all_kind = [row for row in units if row["haiku_kind"] == kind]
            rank = all_kind.index(unit)
            if frame == "T":
                target = concept_t if kind == "concept" else slot_t
            elif frame == "C":
                target = concept_c if kind == "concept" else slot_c
            else:
                target = 0
            if kind == "concept" and frame == "T":
                lens_count = min(jlens_reads, concept_t)
                if unit["id"] in J_LENS_IDS:
                    read = J_LENS_IDS.index(unit["id"]) < lens_count
                else:
                    non_lens_rank = rank - len(J_LENS_IDS)
                    read = non_lens_rank < concept_t - lens_count
            else:
                read = rank < target and frame != "F"
            unit_frames.append({"unit_id": unit["id"], "frame": frame,
                                "haiku_kind": kind, "ds_kind": unit["ds_kind"],
                                "agreement_group": unit["group"], "read": bool(read)})
    shuffled = []
    for frame in score122.FRAMES:
        shuffled.extend({"unit_id": unit["id"], "frame": frame,
                         "read": index < shuffled_reads}
                        for index, unit in enumerate(units))
    return units, unit_frames, shuffled


def _predictions(**kwargs):
    units, unit_frames, shuffled = _prediction_fixture(**kwargs)
    return score122.prediction_verdicts(unit_frames, units, shuffled, J_LENS_IDS)


def test_prediction_122_1_threshold_9_of_18():
    assert _predictions(concept_t=9)["122.1"]["passed"] is True
    assert _predictions(concept_t=8)["122.1"]["passed"] is False


def test_prediction_122_2_threshold_5_of_21():
    assert _predictions(slot_t=5)["122.2"]["passed"] is True
    assert _predictions(slot_t=6)["122.2"]["passed"] is False


def test_prediction_122_3_direction_and_strict_fisher_boundary(monkeypatch):
    result = _predictions(concept_t=9, slot_t=3)["122.3"]
    assert result["metrics"]["fisher_two_sided_p"] < 0.05
    assert result["passed"] is True
    monkeypatch.setattr(score122, "fisher_exact_two_sided", lambda *_: 0.05)
    assert _predictions(concept_t=9, slot_t=3)["122.3"]["passed"] is False
    assert _predictions(concept_t=3, slot_t=9)["122.3"]["passed"] is False


def test_prediction_122_4_gain_boundary_5_slot_units():
    assert _predictions(slot_t=3, slot_c=8)["122.4"]["passed"] is True
    assert _predictions(slot_t=3, slot_c=7)["122.4"]["passed"] is False


def test_prediction_122_5_requires_strictly_larger_kind_share():
    assert _predictions(concept_t=9, slot_t=3, concept_c=10, slot_c=8)["122.5"]["passed"] is True
    equal = _predictions(concept_t=9, slot_t=3, concept_c=15, slot_c=10)["122.5"]
    assert equal["metrics"]["slot_gain_share"] == equal["metrics"]["concept_gain_share"]
    assert equal["passed"] is False


def test_prediction_122_6_shuffled_rate_boundary_every_frame():
    assert _predictions(shuffled_reads=3)["122.6"]["passed"] is True
    assert _predictions(shuffled_reads=4)["122.6"]["passed"] is False


def test_prediction_122_7_threshold_7_of_11():
    assert _predictions(jlens_reads=7)["122.7"]["passed"] is True
    assert _predictions(jlens_reads=6)["122.7"]["passed"] is False


def test_missing_prediction_value_cannot_pass():
    units, unit_frames, shuffled = _prediction_fixture()
    for row in unit_frames:
        if row["unit_id"] == J_LENS_IDS[0] and row["frame"] == "T":
            row["read"] = None
    result = score122.prediction_verdicts(unit_frames, units, shuffled, J_LENS_IDS)
    assert result["122.7"]["passed"] is False
    assert result["122.7"]["available"] is False
    assert result["122.7"]["verdict"] == "not evaluable (missing value)"


def test_registered_reading_branches_all_four():
    def pred(p1=True, p2=True, p3=True, p4=True):
        return {key: {"available": True, "passed": value}
                for key, value in (("122.1", p1), ("122.2", p2),
                                   ("122.3", p3), ("122.4", p4))}

    assert score122.reading_branch(pred())["branch"] == 1
    assert score122.reading_branch(pred(p4=False))["branch"] == 2
    assert score122.reading_branch(pred(p3=False))["branch"] == 3
    assert score122.reading_branch(pred(p1=False))["branch"] == 4


def _synthetic_activation_rows(units):
    read_set = {"units": {}, "shuffled_partner": {}}
    ids = [unit["id"] for unit in units]
    for index, unit_id in enumerate(ids):
        read_set["units"][unit_id] = {"layer": 0, "neuron": index,
                                      "random": [{"random_id": random_id,
                                                  "channel": len(ids) + index * 3 + random_id}
                                                 for random_id in range(3)]}
        read_set["shuffled_partner"][unit_id] = ids[(index + 1) % len(ids)]
    rows = []
    for frame in score122.FRAMES:
        for source_id in ids:
            for pair_index in range(8):
                for side in ("persona", "ordinary"):
                    values = {}
                    for index, unit_id in enumerate(ids):
                        target_win = unit_id == source_id and side == "persona"
                        values[unit_id] = {
                            "unit": {"channel": index, "max": 2.0 if target_win else 1.0,
                                     "mean": 0.8 if target_win else 0.4},
                            "random": [{"random_id": random_id, "channel": len(ids) + index * 3 + random_id,
                                        "max": 1.0, "mean": 0.4}
                                       for random_id in range(3)],
                        }
                    rows.append({"frame": frame, "source_unit_id": source_id,
                                 "pair_index": pair_index, "side": side, "token_count": 12,
                                 "values": values})
    return rows, read_set


def test_scorer_aggregates_gates_controls_tables_and_reading():
    units = _fixture_units()
    activations, read_set = _synthetic_activation_rows(units)
    result = score122.score_results(activations, _gate2(9), read_set, units, J_LENS_IDS)
    assert result["gates"]["all_passed"] is True
    assert result["gates"]["gate1"]["observed_pairs"] == 39 * 3 * 8
    assert result["gates"]["gate3"]["by_frame"]["T"]["rate"] == 0.0
    assert result["predictions"]["122.1"]["metrics"]["concept_reads"] == 18
    assert result["reading_branch"]["branch"] == 3
    markdown = score122.build_score_markdown(result)
    assert "## Activation tables" in markdown
    assert "Max persona" in markdown and "Mean persona" in markdown


def test_scorer_missing_inputs_withholds_predictions_and_keeps_tables():
    units = _fixture_units()
    result = score122.score_results([], None, None, units, J_LENS_IDS)
    assert result["gates"]["all_passed"] is False
    assert all(row["verdict"] == "not read (gate 1)" for row in result["predictions"].values())
    markdown = score122.build_score_markdown(result)
    assert "nothing is read from them because gate 1 failed" in markdown
    assert "missing" in markdown


def test_deliberate_break_in_temporary_scorer_copy_turns_red(tmp_path):
    source = (HERE / "score122.py").read_text(encoding="utf-8")
    assert "READ_WIN_MIN = 7" in source
    broken = source.replace("READ_WIN_MIN = 7", "READ_WIN_MIN = 8", 1)
    copy_dir = tmp_path / "part122"
    import_dir = tmp_path / "part119"
    copy_dir.mkdir()
    import_dir.mkdir()
    (copy_dir / "score122.py").write_text(broken, encoding="utf-8")
    shutil.copyfile(ROOT / "part119" / "rows119.py", import_dir / "rows119.py")
    code = (
        "import importlib.util,sys; p=sys.argv[1]; "
        "s=importlib.util.spec_from_file_location('broken_score',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert m.unit_read_verdict(7,8) is True"
    )
    result = subprocess.run([PYTHON, "-I", "-B", "-c", code,
                             str(copy_dir / "score122.py")],
                            cwd=str(ROOT), text=True, capture_output=True)
    assert result.returncode != 0


def test_isolated_help_for_all_three_scripts():
    for name in ("read122.py", "score122.py", "write_passages.py"):
        result = subprocess.run([PYTHON, "-I", "-B", str(HERE / name), "--help"],
                                cwd=str(ROOT), text=True, capture_output=True)
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_runner_imports_from_runtime_copy_tree(tmp_path):
    copy_paths = ("part122/read122.py", "part122/write_passages.py", "part119/rows119.py",
                  "part92/common92.py", "part85/checkpoints.py")
    for relative in copy_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    result = subprocess.run([PYTHON, "-I", "-B", str(tmp_path / "part122" / "read122.py"), "--help"],
                            cwd=str(tmp_path), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def _write_reader_inputs(tmp_path: Path):
    neurons = []
    haiku = {}
    ds_rows = []
    passage_units = []
    for index in range(39):
        unit_id = f"unit-{index:02d}"
        haiku_kind = "slot" if index < 21 else "concept"
        ds_kind = "slot" if index < 21 else "concept" if index < 29 else "unclassified"
        neurons.append({"id": unit_id, "layer": 0, "neuron": index,
                        "persona": f"Persona{index}", "desc": f"Persona {index} description.",
                        "control_seed": 12200 + index})
        haiku[unit_id] = haiku_kind
        ds_rows.append({"id": unit_id, "ds_kind": ds_kind})
        third = [{"persona": f"Persona{index} waits beside a window. It watches the quiet street.",
                  "ordinary": "A teapot waits beside a window. It watches the quiet street."}
                 for _ in range(8)]
        first = [{"persona": f"I am Persona{index} beside a quiet window. I walk home before dusk.",
                  "ordinary": "I am a teapot beside a quiet window. I walk home before dusk."}
                 for _ in range(8)]
        passage_units.append({"id": unit_id, "persona": f"Persona{index}",
                              "desc": f"Persona {index} description.",
                              "third_person": third, "first_person": first})
    neuron_path = tmp_path / "neurons.json"
    kinds_path = tmp_path / "kinds-haiku.json"
    ds_path = tmp_path / "kinds-ds.json"
    passages_path = tmp_path / "passages.json"
    gate2_path = tmp_path / "gate2-passages.json"
    neuron_path.write_text(json.dumps(neurons), encoding="utf-8")
    kinds_path.write_text(json.dumps({"kinds": haiku}), encoding="utf-8")
    ds_path.write_text(json.dumps({"kind_comparison": {"units": ds_rows}}), encoding="utf-8")
    passages_path.write_text(json.dumps({"schema_version": 1, "seed": 12201,
                                         "model_id": "stub", "units": passage_units}),
                             encoding="utf-8")
    gate2_pairs = [["A rocket reached orbit today. It passed the moon.",
                    "A parade crossed town today. It passed the bridge."] for _ in range(16)]
    gate2_path.write_text(json.dumps({"pairs": gate2_pairs}), encoding="utf-8")
    return neuron_path, kinds_path, ds_path, passages_path, gate2_path


def test_writer_dry_run_cli_uses_stub_and_marks_output(tmp_path):
    neuron_path, kinds_path, ds_path, _passages_path, _gate2_path = _write_reader_inputs(tmp_path)
    output = tmp_path / "dry-run-passages.json"
    result = write_passages.main(["--dry-run", "--out", str(output),
                                  "--neurons", str(neuron_path), "--kinds-haiku", str(kinds_path),
                                  "--kinds-ds", str(ds_path)])
    assert result == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["model_id"] == "stub-local-not-study-data"
    assert len(data["units"]) == 39
    assert all(row["attempts"] == 1 for row in data["units"])


class TinyDown(torch.nn.Module):
    def __init__(self, width: int, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False, dtype=dtype)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, hidden):
        return self.linear(hidden)


class TinyMLP(torch.nn.Module):
    def __init__(self, width: int, dtype):
        super().__init__()
        self.down_proj = TinyDown(width, dtype)

    def forward(self, hidden):
        return self.down_proj(hidden)


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int, dtype):
        super().__init__()
        self.mlp = TinyMLP(width, dtype)

    def forward(self, hidden):
        return self.mlp(hidden)


class TinyInner(torch.nn.Module):
    def __init__(self, width: int, dtype):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(width, dtype)])


class TinyModel(torch.nn.Module):
    def __init__(self, width: int, dtype):
        super().__init__()
        self.embed = torch.nn.Embedding(512, width, dtype=dtype)
        self.model = TinyInner(width, dtype)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(logits=hidden)


def test_real_read_main_path_with_tiny_torch_model(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    neuron_path, kinds_path, ds_path, passages_path, gate2_path = _write_reader_inputs(tmp_path)
    model = TinyModel(64, torch.bfloat16)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_k: StubTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: model)
    monkeypatch.setattr(read122, "GATE2_LAYER", 0)
    monkeypatch.setattr(read122, "GATE2_NEURON", 1)
    out_dir = tmp_path / "results" / "part122"
    argv = ["--model", "tiny-local-model", "--device", "cpu", "--out", str(out_dir),
            "--passages", str(passages_path), "--gate2-passages", str(gate2_path),
            "--neurons", str(neuron_path), "--kinds-haiku", str(kinds_path),
            "--kinds-ds", str(ds_path), "--batch", "4", "--unit-limit", "1"]
    assert read122.main(argv) == 0
    rows = [json.loads(line) for line in (out_dir / "activations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 * 8 * 2
    assert {row["frame"] for row in rows} == {"T", "F", "C"}
    assert all(row["token_count"] > 0 for row in rows)
    assert all(len(row["values"]["unit-00"]["random"]) == 3 for row in rows)
    assert len(json.loads((out_dir / "gate2.json").read_text(encoding="utf-8"))["pairs"]) == 16
    assert json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))["parameter_dtype"] == "torch.bfloat16"


def test_real_transport_path_reaches_the_client(monkeypatch):
    """The default request goes through the imports the host run uses; only the network call is replaced."""
    import write_passages
    import label119

    seen = {}

    def fake_call(prompt, **kwargs):
        seen.update(kwargs)
        return {"text": "{}"}

    monkeypatch.setattr(label119.label92b.judge_client, "call_prompt", fake_call)
    response = write_passages._default_request("ds", "hello", "")
    assert response == {"text": "{}"}
    assert seen["model"] == label119.label92b.DS_MODEL and seen["route"] == "ocgo"

