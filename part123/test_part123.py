from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for directory in (ROOT / "part92", ROOT / "part92b", ROOT / "part119"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import fit123
import label119
import label123
import run123
import score123


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, enable_thinking):
        parts = [f"<{message['role']}>{message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [2 + (ord(char) % 7) for char in str(text)]}

    def decode(self, token_ids, *, skip_special_tokens=True):
        tokens = [int(token) for token in token_ids]
        if tokens == [10, 1] or tokens == [10]:
            return "plain answer"
        if tokens == [11, 1] or tokens == [11]:
            return "pushed answer"
        return "sample continuation"


class TinyMLP(__import__("torch").nn.Module):
    def __init__(self):
        import torch
        super().__init__()
        self.down_proj = SimpleNamespace()
        self.down_proj.weight = torch.nn.Parameter(torch.ones((4, 8), dtype=torch.float32))


class TinyBlock(__import__("torch").nn.Module):
    def __init__(self):
        import torch
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, hidden):
        return hidden


class TinyDecoder(__import__("torch").nn.Module):
    def __init__(self):
        import torch
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock()])


class TinyModel(__import__("torch").nn.Module):
    def __init__(self):
        import torch
        super().__init__()
        self.model = TinyDecoder()
        self.generation_calls = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        import torch
        feature = torch.nn.functional.one_hot(input_ids.remainder(4), num_classes=4).float()
        feature = feature.mean(dim=1, keepdim=True).expand(-1, input_ids.shape[1], -1).clone()
        for layer in self.model.layers:
            feature = layer(feature)
        return SimpleNamespace(last_hidden_state=feature)

    def generate(self, input_ids, attention_mask=None, do_sample=False, max_new_tokens=64,
                 pad_token_id=None, eos_token_id=None, temperature=None, use_cache=True):
        import torch
        self.generation_calls += 1
        if do_sample:
            suffix = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=input_ids.dtype, device=input_ids.device)
            return torch.cat((input_ids, suffix.expand(input_ids.shape[0], -1)), dim=1)
        hidden = self.forward(input_ids, attention_mask=attention_mask).last_hidden_state[:, -1, :]
        generated = torch.where(hidden.sum(dim=-1) > 1.1, 11, 10).to(dtype=input_ids.dtype).unsqueeze(1)
        eos_value = 1
        if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
            eos_value = int(eos_token_id[0])
        elif isinstance(eos_token_id, int):
            eos_value = int(eos_token_id)
        eos = torch.full((input_ids.shape[0], 1), eos_value, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, generated, eos), dim=1)


def _small_units():
    names = ("Boulder", "Magical princess", "The town", "The Internet")
    kinds = ("concept", "concept", "slot", "slot")
    units = []
    kind_map = {}
    for index, (name, kind) in enumerate(zip(names, kinds)):
        uid = f"p92-{index:02d}"
        units.append({"id": uid, "layer": 0, "neuron": index, "persona": name,
                      "desc": f"A short test description for {name}."})
        kind_map[uid] = kind
    return units, {"kinds": kind_map}


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _fake_reader(reader, prompt, _key=""):
    exact = {}
    for item in label119.read_known():
        exact[label119.build_prompt(item)] = {
            "x": item["expected_x"], "y": item["expected_y"],
            "coherent": item["expected_coherent"],
        }
    if prompt in exact:
        return {"text": json.dumps(exact[prompt]), "usage": {"input_tokens": 1, "output_tokens": 1}}
    shows = hashlib.sha256(prompt.encode("utf-8")).digest()[0] % 2 == 0
    return {"text": json.dumps({"shows_persona": shows, "self_claim": shows,
                                "speaker": "other" if shows else "ai", "coherent": True}),
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def test_fit_run_label_and_score_mains_on_cpu_stub(capsys):
    with tempfile.TemporaryDirectory(prefix="p123-e2e-", dir=HERE) as scratch:
        base = Path(scratch)
        units, kinds = _small_units()
        neurons_path, kinds_path = base / "neurons.json", base / "kinds.json"
        _write_json(neurons_path, units)
        _write_json(kinds_path, kinds)
        fit_dir, pilot_dir, main_dir = base / "fit", base / "pilot", base / "main"
        model = TinyModel()
        assert fit123.main(["--model", "stub", "--neurons", str(neurons_path), "--kinds", str(kinds_path),
                            "--out", str(fit_dir), "--device", "cpu", "--batch-size", "16", "--shuffles", "20"],
                           model_loader=lambda _model, _device: (TinyTokenizer(), model),
                           validate_inventory=False) == 0
        fit_meta = json.loads((fit_dir / "fit-meta.json").read_text(encoding="utf-8"))
        assert fit_meta["suffix_samples"] == 4
        assert fit_meta["fit_states_stored"] == "last token only"
        assert model.generation_calls == 4  # Stage A generated only its four suffix samples.

        run_args = ["--stage", "pilot", "--model", "stub", "--fit", str(fit_dir),
                    "--neurons", str(neurons_path), "--kinds", str(kinds_path), "--out", str(pilot_dir),
                    "--device", "cpu", "--batch-size", "16"]
        assert run123.main(run_args, model_loader=lambda _model, _device: (TinyTokenizer(), TinyModel()),
                           validate_inventory=False) == 0
        # Stage C starts in its own folder and resumes from that folder's answers.
        main_args = list(run_args)
        main_args[1] = "main"
        main_args[main_args.index("--out") + 1] = str(main_dir)
        main_model = TinyModel()
        assert run123.main(main_args, model_loader=lambda _model, _device: (TinyTokenizer(), main_model),
                           validate_inventory=False) == 0
        answers = _read_jsonl(main_dir / "answers.jsonl")
        assert len(answers) == 43 * 6
        answer_index = {(row["row"], int(row["question_index"])): row for row in answers}
        assert answer_index[("plain", 0)]["answer"] != answer_index[("neuron-p92-02-q0.5", 0)]["answer"]
        check = json.loads((main_dir / "zero-hook-check.json").read_text(encoding="utf-8"))
        assert len(check) == 6 and all(row["equal"] and row["batch_size"] == 1 for row in check)
        assert run123.main(main_args, model_loader=lambda _model, _device: (TinyTokenizer(), TinyModel()),
                           validate_inventory=False) == 0
        assert len(_read_jsonl(main_dir / "answers.jsonl")) == 43 * 6

        known_ds, known_luna = base / "known-ds.jsonl", base / "known-luna.jsonl"
        assert label123.main(["--gate", "--reader", "ds", "--out", str(known_ds), "--workers", "1"],
                             request_fn=_fake_reader, sleep_fn=lambda _seconds: None,
                             validate_inventory=False) == 0
        assert label123.main(["--gate", "--reader", "luna", "--out", str(known_luna), "--workers", "1"],
                             request_fn=_fake_reader, sleep_fn=lambda _seconds: None,
                             validate_inventory=False) == 0
        ds_out, luna_out = base / "labels-ds.jsonl", base / "labels-luna.jsonl"
        common_args = ["--answers", str(main_dir / "answers.jsonl"), "--neurons", str(neurons_path),
                       "--kinds", str(kinds_path), "--stage", "main", "--workers", "1"]
        assert label123.main(["--reader", "ds", *common_args, "--out", str(ds_out),
                              "--gate-labels", str(known_ds)], request_fn=_fake_reader,
                             sleep_fn=lambda _seconds: None, validate_inventory=False) == 0
        assert label123.main(["--reader", "luna", *common_args, "--out", str(luna_out),
                              "--gate-labels", str(known_luna), "--labels-ds", str(ds_out)],
                             request_fn=_fake_reader, sleep_fn=lambda _seconds: None,
                             validate_inventory=False) == 0

        overlap = _read_jsonl(fit_dir / "overlap.jsonl")
        nulls = _read_jsonl(fit_dir / "shuffled-null.jsonl")
        # The fitter's 20-draw test null is intentionally too small for the registered scorer.
        _write_jsonl(base / "overlap.jsonl", overlap)
        _write_jsonl(base / "null.jsonl", nulls)
        score_out = base / "score"
        score_args = ["--answers", str(main_dir / "answers.jsonl"), "--neurons", str(neurons_path),
                      "--kinds", str(kinds_path), "--zero-hook", str(main_dir / "zero-hook-check.json"),
                      "--labels-ds", str(ds_out), "--keys-ds", str(ds_out) + ".keys.jsonl",
                      "--labels-luna", str(luna_out), "--keys-luna", str(luna_out) + ".keys.jsonl",
                      "--known-ds", str(known_ds), "--known-luna", str(known_luna),
                      "--overlap", str(base / "overlap.jsonl"), "--shuffled-null", str(base / "null.jsonl"),
                      "--out", str(score_out)]
        assert score123.main(score_args, validate_inventory=False) == 0
        scored = json.loads((score_out / "score123.json").read_text(encoding="utf-8"))
        assert len(scored["gates"]) == 7
        assert scored["gates"][0]["passed"] is True
        assert scored["gates"][3]["passed"] is True
        assert scored["gates"][6]["passed"] is True
        assert (score_out / "SCORE.md").is_file()

        # Changing one answer keeps the same blinded item ID but makes its prior hash stale.
        changed = list(answers)
        for row in changed:
            if row["row"] == "neuron-p92-00-q0.5" and row["question_index"] == 0:
                row["answer"] += " revised"
                break
        changed_answers = base / "answers-changed.jsonl"
        _write_jsonl(changed_answers, changed)
        stale_args = ["--reader", "ds", "--answers", str(changed_answers), "--neurons", str(neurons_path),
                      "--kinds", str(kinds_path), "--stage", "main", "--workers", "1",
                      "--out", str(ds_out), "--gate-labels", str(known_ds)]
        assert label123.main(stale_args, request_fn=_fake_reader, sleep_fn=lambda _seconds: None,
                             validate_inventory=False) == 0
        final_line = capsys.readouterr().out.strip().splitlines()[-1]
        assert json.loads(final_line)["pending"] == 1


def test_zero_hook_mismatch_stops_before_pushed_rows_and_flag_overrides(monkeypatch, capsys):
    import torch

    def breaking_add_row_hook(layer, _row_vectors):
        return layer.register_forward_hook(lambda _module, _inputs, output: output + 1.0)

    monkeypatch.setattr(run123.common92, "add_row_hook", breaking_add_row_hook)
    with tempfile.TemporaryDirectory(prefix="p123-zero-mismatch-", dir=HERE) as scratch:
        base = Path(scratch)
        units, kinds = _small_units()
        neurons_path, kinds_path = base / "neurons.json", base / "kinds.json"
        _write_json(neurons_path, units)
        _write_json(kinds_path, kinds)
        fit_dir = base / "fit"
        fit_dir.mkdir()
        torch.save({"vectors": {str(unit["id"]): torch.ones(4) for unit in units}}, fit_dir / "vectors.pt")

        def run(out: Path, allow: bool = False):
            args = ["--stage", "main", "--model", "stub", "--fit", str(fit_dir),
                    "--neurons", str(neurons_path), "--kinds", str(kinds_path), "--out", str(out),
                    "--device", "cpu", "--batch-size", "16"]
            if allow:
                args.append("--allow-zero-mismatch")
            model = TinyModel()
            rc = run123.main(args, model_loader=lambda _model, _device: (TinyTokenizer(), model),
                             validate_inventory=False)
            return rc, model

        stopped_dir = base / "stopped"
        rc, _model = run(stopped_dir)
        assert rc == 1
        assert "zero-hook mismatch" in capsys.readouterr().err
        stopped = _read_jsonl(stopped_dir / "answers.jsonl")
        assert stopped and all(row["arm"] in ("plain", "zero") for row in stopped)
        assert any(not row["equal"] for row in json.loads(
            (stopped_dir / "zero-hook-check.json").read_text(encoding="utf-8")))

        allowed_dir = base / "allowed"
        rc, _model = run(allowed_dir, allow=True)
        assert rc == 0
        assert any(row["arm"] == "neuron" for row in _read_jsonl(allowed_dir / "answers.jsonl"))


def _full_score_fixture(root: Path) -> None:
    units = []
    kinds_map = {}
    for index in range(39):
        uid = f"unit-{index:02d}"
        kind = "slot" if index < 21 else "concept"
        units.append({"id": uid, "layer": index % 10, "neuron": index % 100,
                      "persona": f"Persona {index:02d}", "desc": f"Test description {index:02d}."})
        kinds_map[uid] = kind
    neurons_path, kinds_path = root / "neurons.json", root / "kinds.json"
    _write_json(neurons_path, units)
    _write_json(kinds_path, {"kinds": kinds_map})
    selected = fit123.selected_units(units, {"kinds": kinds_map})
    layers = sorted({int(unit["layer"]) for unit in selected})
    specs = run123.build_rows(selected, layers)
    answers = []
    for qi, question in enumerate(run123.QUESTIONS):
        for spec in specs:
            text = f"baseline answer q{qi}" if spec["arm"] in ("plain", "zero") else f"{spec['row']} q{qi}"
            tokens = [qi, 42] if spec["arm"] in ("plain", "zero") else [qi, 42, 43]
            answers.append({**spec, "question_index": qi, "question": question,
                            "answer": text, "token_ids": tokens, "token_count": len(tokens)})
    answers_path = root / "answers.jsonl"
    _write_jsonl(answers_path, answers)
    items = label123.make_items(answers, selected)
    unit_num = {str(unit["id"]): int(str(unit["id"]).split("-")[1]) for unit in selected}
    by_id = {str(unit["id"]): unit for unit in selected}
    ds_labels = []
    for item in items:
        private = item["private"]
        unit_id = str(private["unit_id"])
        unit = by_id[unit_id]
        number = unit_num[unit_id]
        qi = int(private["question_index"])
        arm = str(private["arm"])
        q = private.get("q")
        shows = False
        if arm == "neuron" and q == 1.0:
            if unit["kind"] == "slot":
                shows = qi == 0 and number < 17
            else:
                shows = qi < 3 and number < 21 + 14
        elif arm == "neuron" and q == 0.85 and unit["kind"] == "slot":
            shows = qi == 0 and number < 10
        elif arm == "vector" and q == 1.0:
            if unit["kind"] == "slot":
                shows = qi == 0 and number < 11
            else:
                shows = qi < 3 and number < 21 + 9
        elif arm == "vector" and q == 0.85 and unit["kind"] == "slot":
            shows = qi == 0 and number < 9
        label = {**item["public"], "shows_persona": bool(shows), "self_claim": bool(shows),
                 "speaker": "other" if shows else "ai", "coherent": True}
        ds_labels.append(label)
    ds_path = root / "labels-ds.jsonl"
    ds_keys_path = root / "keys-ds.jsonl"
    _write_jsonl(ds_path, ds_labels)
    ds_keys = [dict(item["private"]) for item in items]
    _write_jsonl(ds_keys_path, ds_keys)

    luna_items = label123.luna_sample(items, ds_labels)
    luna_path = root / "labels-luna.jsonl"
    luna_keys_path = root / "keys-luna.jsonl"
    _write_jsonl(luna_path, [next(label for label in ds_labels if
                                  label["item_id"] == item["public"]["item_id"])
                             for item in luna_items])
    _write_jsonl(luna_keys_path, [dict(item["private"]) for item in luna_items])

    zero = [{"layer": layer, "question_index": qi, "question": run123.QUESTIONS[qi], "equal": True}
            for layer in layers for qi in range(6)]
    _write_json(root / "zero.json", zero)
    overlap_rows, null_rows = [], []
    for index, unit in enumerate(selected):
        rank = 1 if unit["kind"] == "concept" or index < 6 else 100
        cosine = 0.9 if unit["kind"] == "concept" or index < 6 else 0.01
        overlap_rows.append({"unit_id": unit["id"], "persona": unit["persona"], "kind": unit["kind"],
                             "layer": unit["layer"], "channel": unit["neuron"], "channel_count": 100,
                             "signed_cosine": cosine, "rank": rank,
                             "top_decile_rank_limit": 10, "shuffle_count": 1000})
        null_rows.append({"unit_id": unit["id"], "cosines": [-0.1] * 1000})
    _write_jsonl(root / "overlap.jsonl", overlap_rows)
    _write_jsonl(root / "null.jsonl", null_rows)

    known = label119.read_known()
    known_labels = [{"item_id": label119.gate_public(i)["item_id"],
                     "x": item["expected_x"], "y": item["expected_y"],
                     "coherent": item["expected_coherent"]} for i, item in enumerate(known)]
    _write_jsonl(root / "known-ds.jsonl", known_labels)
    _write_jsonl(root / "known-luna.jsonl", known_labels)


@pytest.fixture(scope="session")
def valid_score_fixture():
    with tempfile.TemporaryDirectory(prefix="p123-full-score-", dir=HERE) as scratch:
        root = Path(scratch)
        _full_score_fixture(root)
        yield root


def _score_args(root: Path, out: Path):
    return ["--answers", str(root / "answers.jsonl"), "--neurons", str(root / "neurons.json"),
            "--kinds", str(root / "kinds.json"), "--zero-hook", str(root / "zero.json"),
            "--labels-ds", str(root / "labels-ds.jsonl"), "--keys-ds", str(root / "keys-ds.jsonl"),
            "--labels-luna", str(root / "labels-luna.jsonl"), "--keys-luna", str(root / "keys-luna.jsonl"),
            "--known-ds", str(root / "known-ds.jsonl"), "--known-luna", str(root / "known-luna.jsonl"),
            "--overlap", str(root / "overlap.jsonl"), "--shuffled-null", str(root / "null.jsonl"),
            "--out", str(out)]


def _score_result(root: Path, out: Path):
    assert score123.main(_score_args(root, out)) == 0
    return json.loads((out / "score123.json").read_text(encoding="utf-8"))


def _clone_fixture(source: Path, destination: Path):
    shutil.copytree(source, destination)
    return destination


def _break_gate(root: Path, gate_number: int):
    if gate_number == 1:
        rows = json.loads((root / "zero.json").read_text(encoding="utf-8"))
        rows[0]["equal"] = False
        _write_json(root / "zero.json", rows)
    elif gate_number in (2, 3):
        kinds = json.loads((root / "kinds.json").read_text(encoding="utf-8"))["kinds"]
        private = {row["item_id"]: row for row in _read_jsonl(root / "keys-ds.jsonl")}
        labels = _read_jsonl(root / "labels-ds.jsonl")
        for row in labels:
            key = private[row["item_id"]]
            if key["arm"] != "neuron" or key["q"] != 1.0:
                continue
            kind = kinds[key["unit_id"]]
            qi = int(key["question_index"])
            if gate_number == 2 and kind == "slot" and qi == 0:
                row["shows_persona"] = False
            if gate_number == 3 and kind == "concept":
                row["shows_persona"] = False
        _write_jsonl(root / "labels-ds.jsonl", labels)
    elif gate_number == 4:
        private = _read_jsonl(root / "keys-ds.jsonl")
        stale_item = next(row["item_id"] for row in private
                          if row["arm"] == "vector" and row["q"] == 0.5 and row["question_index"] == 0)
        labels = [row for row in _read_jsonl(root / "labels-ds.jsonl") if row["item_id"] != stale_item]
        _write_jsonl(root / "labels-ds.jsonl", labels)
    elif gate_number == 5:
        rows = _read_jsonl(root / "known-ds.jsonl")
        for row in rows[:3]:
            row["x"] = not row["x"]
        _write_jsonl(root / "known-ds.jsonl", rows)
    elif gate_number == 6:
        rows = _read_jsonl(root / "labels-luna.jsonl")
        for row in rows:
            row["shows_persona"] = not row["shows_persona"]
        _write_jsonl(root / "labels-luna.jsonl", rows)
    elif gate_number == 7:
        private = {row["item_id"]: row for row in _read_jsonl(root / "keys-ds.jsonl")}
        labels = _read_jsonl(root / "labels-ds.jsonl")
        for row in labels:
            key = private[row["item_id"]]
            if key["arm"] == "vector" and key["q"] == 1.0 and row["shows_persona"]:
                row["coherent"] = False
        _write_jsonl(root / "labels-ds.jsonl", labels)
    else:
        raise AssertionError(f"unknown gate number {gate_number}")


def test_raw_scorer_passes_a_complete_synthetic_run(valid_score_fixture):
    with tempfile.TemporaryDirectory(prefix="p123-score-pass-", dir=HERE) as scratch:
        result = _score_result(valid_score_fixture, Path(scratch) / "out")
        assert result["all_gates_passed"] is True
        assert len(result["predictions"]) == 6
        assert result["predictions"]["123.1"]["verdict"] == "held"
        assert result["predictions"]["123.3"]["verdict"] == "held"
        assert result["predictions"]["123.6"]["verdict"] == "held"


def test_stale_answer_hash_is_missing_to_the_scorer(valid_score_fixture):
    with tempfile.TemporaryDirectory(prefix="p123-stale-score-", dir=HERE) as scratch:
        root = _clone_fixture(valid_score_fixture, Path(scratch) / "copy")
        answers = _read_jsonl(root / "answers.jsonl")
        target = next(row for row in answers if row["row"] == "vector-unit-00-q0.5" and
                      row["question_index"] == 0)
        target["answer"] += " changed after labeling"
        _write_jsonl(root / "answers.jsonl", answers)
        result = _score_result(root, root / "score-out")
        assert result["gates"][3]["passed"] is False
        assert result["totals"]["missing_or_stale_deepseek_labels"] == 1


@pytest.mark.parametrize("gate_number", [1, 2, 3, 4, 5, 6, 7])
def test_each_registered_gate_fails_on_its_mutated_input_copy(valid_score_fixture, gate_number):
    with tempfile.TemporaryDirectory(prefix=f"p123-break-g{gate_number}-", dir=HERE) as scratch:
        root = _clone_fixture(valid_score_fixture, Path(scratch) / "copy")
        _break_gate(root, gate_number)
        result = _score_result(root, root / "score-out")
        target = next(gate for gate in result["gates"] if gate["number"] == gate_number)
        assert target["passed"] is False
        if gate_number == 4:
            assert result["totals"]["missing_or_stale_deepseek_labels"] == 1
            assert result["predictions"]["123.2"]["verdict"] == "not read (gate 4)"
        if gate_number == 7:
            assert result["predictions"]["123.2"]["verdict"] == "not read (gate 7)"


def test_two_broken_gate_copies_make_expectation_tests_go_red(valid_score_fixture):
    with tempfile.TemporaryDirectory(prefix="p123-red-probe-", dir=HERE) as scratch:
        root = Path(scratch)
        copies = []
        for gate_number in (1, 7):
            copy = _clone_fixture(valid_score_fixture, root / f"gate-{gate_number}")
            _break_gate(copy, gate_number)
            copies.append(copy)
        probe = root / "test_red_probe.py"
        first_args = repr(_score_args(copies[0], copies[0] / "score-out"))
        second_args = repr(_score_args(copies[1], copies[1] / "score-out"))
        probe.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import score123\n"
            f"def test_broken_zero_gate_was_expected_to_pass():\n    score123.main({first_args})\n"
            f"    result=json.load(open({str(copies[0] / 'score-out' / 'score123.json')!r}, encoding='utf-8'))\n"
            "    assert result['all_gates_passed'] is True\n"
            f"def test_broken_readable_persona_gate_was_expected_to_pass():\n    score123.main({second_args})\n"
            f"    result=json.load(open({str(copies[1] / 'score-out' / 'score123.json')!r}, encoding='utf-8'))\n"
            "    assert result['all_gates_passed'] is True\n",
            encoding="utf-8",
        )
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)],
                              cwd=HERE, capture_output=True, text=True, check=False)
        assert proc.returncode != 0
        assert "2 failed" in proc.stdout


def test_torn_final_jsonl_is_logged_and_nonfinal_damage_raises(capsys):
    with tempfile.TemporaryDirectory(prefix="p123-jsonl-", dir=HERE) as scratch:
        final = Path(scratch) / "answers.jsonl"
        final.write_bytes(b'{"row":"ok"}\n{"torn"')
        assert run123._read_jsonl(final) == [{"row": "ok"}]
        assert "skipping torn final JSONL line" in capsys.readouterr().err
        assert final.read_bytes() == b'{"row":"ok"}\n{"torn"'
        run123._rewrite_torn_tail(final, [{"row": "ok"}])
        assert final.read_bytes() == b'{"row": "ok"}\n'
        for reader in (label123.read_jsonl, score123.read_jsonl):
            copy = Path(scratch) / f"{reader.__module__}.jsonl"
            copy.write_bytes(b'{"row":"ok"}\n{"torn"')
            assert reader(copy) == [{"row": "ok"}]
            assert copy.read_bytes() == b'{"row":"ok"}\n{"torn"'
            assert "skipping torn final JSONL line" in capsys.readouterr().err
        damaged = Path(scratch) / "damaged.jsonl"
        damaged.write_bytes(b'{"torn"\n{"row":"ok"}\n')
        with pytest.raises(ValueError, match="invalid JSONL"):
            run123._read_jsonl(damaged)
