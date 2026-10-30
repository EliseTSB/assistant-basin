from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
PART124 = Path(__file__).resolve().parent
COPY_LIST = (
    "part124/passages124.py",
    "part124/read124.py",
    "part124/score124.py",
    "part124/passages124.json",
    "part122/read122.py",
    "part122/write_passages.py",
    "part119/rows119.py",
    "part92/common92.py",
    "part85/checkpoints.py",
    "results/part92/step2/neurons.json",
    "results/part92/step2/kinds.json",
    "results/part92b/score92b.json",
)


def _import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    launch = tmp_path_factory.mktemp("part124-launch-copy")
    overrides = {
        "part124/passages124.py": os.environ.get("PART124_PASSAGES_OVERRIDE"),
        "part124/read124.py": os.environ.get("PART124_READ_OVERRIDE"),
        "part124/score124.py": os.environ.get("PART124_SCORE_OVERRIDE"),
    }
    for relative in COPY_LIST:
        override = overrides.get(relative)
        source = Path(override) if override else ROOT / relative
        if not source.is_file():
            raise AssertionError(f"launch copy source is missing: {source}")
        destination = launch / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    actual = {path.relative_to(launch).as_posix() for path in launch.rglob("*") if path.is_file()}
    assert actual == set(COPY_LIST)

    names = ("passages124", "read124", "score124", "read122", "rows119", "common92", "write_passages")
    for name in names:
        sys.modules.pop(name, None)
    added_paths = [str(launch / part) for part in
                   ("part124", "part122", "part119", "part92", "part85", ".")]
    sys.path[:0] = added_paths
    passages = _import_from(launch / "part124/passages124.py", "passages124")
    reader = _import_from(launch / "part124/read124.py", "read124")
    scorer = _import_from(launch / "part124/score124.py", "score124")
    try:
        yield {"launch": launch, "passages": passages, "reader": reader, "scorer": scorer}
    finally:
        for path in added_paths:
            if path in sys.path:
                sys.path.remove(path)
        for name in names:
            sys.modules.pop(name, None)


class TinyTokenizer:
    """Tokenizer stub that preserves quote offsets and gives each fixed slot one token."""

    def __init__(self, phrases):
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.phrase_to_id = {phrase: 1000 + index for index, phrase in enumerate(phrases)}
        self.id_to_phrase = {value: key for key, value in self.phrase_to_id.items()}
        self._known_phrases = sorted(self.phrase_to_id, key=len, reverse=True)
        self._word_ids = {}

    def _word_id(self, token):
        if token not in self._word_ids:
            self._word_ids[token] = 300 + len(self._word_ids)
        return self._word_ids[token]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
        ids = []
        offsets = []
        cursor = 0
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            phrase = next((item for item in self._known_phrases if text.startswith(item, cursor)), None)
            if phrase is not None:
                ids.append(self.phrase_to_id[phrase])
                offsets.append((cursor, cursor + len(phrase)))
                cursor += len(phrase)
                continue
            match = re.match(r"\w+|[^\w\s]", text[cursor:], flags=re.UNICODE)
            if match is None:
                cursor += 1
                continue
            token = match.group(0)
            ids.append(self._word_id(token))
            offsets.append((cursor, cursor + len(token)))
            cursor += len(token)
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
            offsets.insert(0, (0, 0))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False, **_kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        assert enable_thinking is False
        user = next(message["content"] for message in messages if message["role"] == "user")
        assistant = next(message["content"] for message in messages if message["role"] == "assistant")
        return f"<user>{user}</user><assistant>{assistant}</assistant>"


def _toy_model(tokenizer, phrases, unit_channels, *, width=64):
    import torch

    class DownProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((4, width), dtype=torch.bfloat16),
                                             requires_grad=False)

        def forward(self, hidden):
            return hidden

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = DownProj()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer()])
            self.phrase_index = {phrase: index for index, phrase in enumerate(phrases)}
            self.channel_to_phrase = {int(channel): phrase for phrase, channel in unit_channels.items()}
            self.favorite_random_phrase = phrases[0]
            self.call_count = 0

        def forward(self, *, input_ids, attention_mask=None, use_cache=False):
            self.call_count += 1
            batch, sequence = input_ids.shape
            hidden = torch.zeros((batch, sequence, width), dtype=torch.float32,
                                 device=input_ids.device)
            for row in range(batch):
                for position in range(sequence):
                    phrase = tokenizer.id_to_phrase.get(int(input_ids[row, position]))
                    if phrase not in self.phrase_index:
                        continue
                    for channel, target in self.channel_to_phrase.items():
                        hidden[row, position, channel] = 10.0 if phrase == target else 0.0
                    for channel in range(len(unit_channels), width):
                        hidden[row, position, channel] = (
                            10.0 if phrase == self.favorite_random_phrase else 0.0
                        )
            self.model.layers[0].mlp.down_proj(hidden)
            logits = input_ids.float().unsqueeze(-1).expand(-1, -1, 4).clone()
            return SimpleNamespace(logits=logits)

    return Model()


def _make_units(passages_module):
    phrases = list(passages_module.NEUTRALS)
    return [
        {"id": f"test-{index:02d}", "layer": 0, "neuron": index,
         "persona": phrase, "kind": "concept" if index < 18 else "slot"}
        for index, phrase in enumerate(phrases)
    ]


@pytest.fixture(scope="module")
def study(runtime, tmp_path_factory):
    work = tmp_path_factory.mktemp("part124-e2e")
    passages = runtime["passages"]
    units = _make_units(passages)
    phrases = list(passages.NEUTRALS) + list(passages.NEUTRALS.values())
    tokenizer = TinyTokenizer(phrases)
    unit_channels = {unit["persona"]: unit["neuron"] for unit in units}
    model = _toy_model(tokenizer, list(passages.NEUTRALS), unit_channels)
    manifest_path = work / "passages124.json"
    output_dir = work / "read"
    score_dir = work / "score"

    with contextlib.redirect_stdout(io.StringIO()):
        assert passages.main(["--out", str(manifest_path)], tokenizer=tokenizer,
                             units_override=units) == 0
        assert runtime["reader"].main([
            "--out", str(output_dir), "--passages", str(manifest_path),
            "--device", "cpu", "--batch", "8", "--random-count", "20",
        ], tokenizer=tokenizer, model=model, layers=model.model.layers) == 0
    read122_path = Path(sys.modules["read122"].__file__).resolve()
    assert read122_path.is_relative_to(runtime["launch"])
    assert model.call_count == 119

    score_stdout = io.StringIO()
    with contextlib.redirect_stdout(score_stdout):
        assert runtime["scorer"].main([
            "--passages", str(manifest_path),
            "--activations", str(output_dir / "activations.jsonl"),
            "--read-set", str(output_dir / "read_set.json"),
            "--run-meta", str(output_dir / "run_meta.json"),
            "--out", str(score_dir),
        ]) == 0
    score_lines = score_stdout.getvalue().splitlines()
    assert score_lines[0].startswith("random reads:")
    assert "instrument repeat" in score_lines[1]
    gate_lines = [index for index, line in enumerate(score_lines) if line.startswith("gate ")]
    assert len(gate_lines) == 4
    assert score_lines.index("focal results follow in SCORE.md") > gate_lines[-1]
    with (score_dir / "score124.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    return {
        "work": work,
        "manifest": manifest_path,
        "read": output_dir,
        "score": score_dir,
        "result": result,
        "units": units,
    }


def test_all_mains_run_from_only_the_launch_copy_list(runtime, study):
    launch = runtime["launch"]
    assert Path(runtime["passages"].__file__).resolve().is_relative_to(launch)
    assert Path(runtime["reader"].__file__).resolve().is_relative_to(launch)
    assert Path(runtime["scorer"].__file__).resolve().is_relative_to(launch)
    assert set(study["result"]["gates"]) == {
        "construction", "do_nothing", "random_channels", "instrument_T"
    }
    assert study["result"]["all_gates_passed"] is True
    assert len(study["result"]["focal"]["T"]) == 39
    assert (study["score"] / "SCORE.md").is_file()
    assert (study["score"] / "score124.json").is_file()
    score_text = (study["score"] / "SCORE.md").read_text(encoding="utf-8")
    assert score_text.index("random reads:") < score_text.index("## Focal reads")
    assert "Instrument repeat by layer" in score_text
    assert sum(row["reads"] for row in study["result"]["focal"]["T"].values()) == 39


def _copy_launch_tree(runtime, root: Path) -> Path:
    root.mkdir()
    for relative in COPY_LIST:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime["launch"] / relative, destination)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert actual == set(COPY_LIST)
    return root


def _install_stub_read122(launch: Path, manifest: dict) -> None:
    personas = [str(row["persona"]) for row in manifest["phrases"]]
    all_phrases = list(dict.fromkeys(personas + [str(row["neutral"]) for row in manifest["phrases"]]))
    source = r'''from types import SimpleNamespace
import re
import torch

PERSONAS = PERSONAS_VALUE
PHRASES = PHRASES_VALUE

class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 1

    def __init__(self):
        self.phrase_to_id = {phrase: 1000 + index for index, phrase in enumerate(PHRASES)}
        self.id_to_phrase = {value: key for key, value in self.phrase_to_id.items()}
        self.known = sorted(PHRASES, key=len, reverse=True)
        self.word_ids = {}

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            phrase = next((item for item in self.known if text.startswith(item, cursor)), None)
            if phrase is not None:
                ids.append(self.phrase_to_id[phrase])
                offsets.append((cursor, cursor + len(phrase)))
                cursor += len(phrase)
                continue
            match = re.match(r"\w+|[^\w\s]", text[cursor:], flags=re.UNICODE)
            if match is None:
                cursor += 1
                continue
            token = match.group(0)
            if token not in self.word_ids:
                self.word_ids[token] = 300 + len(self.word_ids)
            ids.append(self.word_ids[token])
            offsets.append((cursor, cursor + len(token)))
            cursor += len(token)
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
            offsets.insert(0, (0, 0))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False, **_kwargs):
        user = next(row["content"] for row in messages if row["role"] == "user")
        assistant = next(row["content"] for row in messages if row["role"] == "assistant")
        return f"<user>{user}</user><assistant>{assistant}</assistant>"

class DownProj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((4, 64), dtype=torch.bfloat16),
                                         requires_grad=False)

    def forward(self, value):
        return value

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = DownProj()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer()])
        self.channels = {phrase: index for index, phrase in enumerate(PERSONAS)}

    def forward(self, *, input_ids, attention_mask=None, use_cache=False):
        batch, sequence = input_ids.shape
        hidden = torch.zeros((batch, sequence, 64), dtype=torch.float32,
                             device=input_ids.device)
        for row in range(batch):
            for position in range(sequence):
                phrase = tokenizer.id_to_phrase.get(int(input_ids[row, position]))
                for target, channel in self.channels.items():
                    hidden[row, position, channel] = 10.0 if phrase == target else 0.0
        self.model.layers[0].mlp.down_proj(hidden)
        logits = input_ids.float().unsqueeze(-1).expand(-1, -1, 4).clone()
        return SimpleNamespace(logits=logits)

class Common:
    @staticmethod
    def get_layers(model):
        return model.model.layers

common = Common()
tokenizer = TinyTokenizer()

def load_model(_model_path, _device):
    return tokenizer, Model()

def _pack(tokenizer, prepared, device):
    pad_id = tokenizer.pad_token_id
    width = max(len(row["input_ids"]) for row in prepared)
    input_ids, attention, token_masks = [], [], []
    for row in prepared:
        missing = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * missing)
        attention.append([1] * len(row["input_ids"]) + [0] * missing)
        token_masks.append(row["token_mask"] + [False] * missing)
    return (torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(attention, dtype=torch.long, device=device),
            torch.tensor(token_masks, dtype=torch.bool, device=device))
'''
    source = source.replace("PERSONAS_VALUE", repr(personas)).replace("PHRASES_VALUE", repr(all_phrases))
    (launch / "part122/read122.py").write_text(source, encoding="utf-8")


def test_documented_reader_launch_runs_under_isolated_python_on_stub(runtime, study, tmp_path):
    manifest = json.loads(study["manifest"].read_text(encoding="utf-8"))
    launch = _copy_launch_tree(runtime, tmp_path / "reader-launch")
    _install_stub_read122(launch, manifest)
    output = tmp_path / "subprocess-read"
    result = subprocess.run([
        sys.executable, "-I", "-B", "part124/read124.py",
        "--model", "stub-model", "--out", str(output), "--device", "cpu",
        "--batch", "8", "--passages", str(study["manifest"]), "--random-count", "20",
    ], cwd=launch, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr or result.stdout
    rows = (output / "activations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1872
    assert {path.relative_to(launch).as_posix() for path in launch.rglob("*") if path.is_file()} == set(COPY_LIST)


def test_documented_scorer_command_runs_under_isolated_python(runtime, study, tmp_path):
    launch = _copy_launch_tree(runtime, tmp_path / "score-launch")
    output = tmp_path / "subprocess-score"
    result = subprocess.run([
        sys.executable, "-I", "-B", "part124/score124.py",
        "--passages", str(study["manifest"]),
        "--activations", str(study["read"] / "activations.jsonl"),
        "--read-set", str(study["read"] / "read_set.json"),
        "--run-meta", str(study["read"] / "run_meta.json"),
        "--out", str(output),
    ], cwd=launch, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr or result.stdout
    assert (output / "score124.json").is_file()
    assert sum(json.loads((output / "score124.json").read_text(encoding="utf-8"))
               ["focal"]["T"][unit_id]["reads"] for unit_id in study["result"]["focal"]["T"]) == 39


def test_reader_uses_the_frozen_manifest_template(runtime, study):
    manifest = json.loads(study["manifest"].read_text(encoding="utf-8"))
    original = manifest["templates"]["T"][0]
    changed = original.replace("record prints", "record writes")
    assert changed != original
    manifest["templates"]["T"][0] = changed
    context = runtime["reader"].build_contexts(manifest)[0]
    phrases = list(runtime["passages"].NEUTRALS) + list(runtime["passages"].NEUTRALS.values())
    tokenizer = TinyTokenizer(phrases)
    prepared = runtime["reader"]._prepare_batch(tokenizer, [context])[0]
    assert context["text"] == changed.replace("{slot}", context["phrase"])
    assert prepared["text"] == context["text"]
    chat_context = next(row for row in runtime["reader"].build_contexts(manifest)
                        if row["frame"] == "C")
    chat_prepared = runtime["reader"]._prepare_batch(tokenizer, [chat_context])[0]
    sentence = manifest["templates"]["F"][0].replace("{slot}", chat_context["phrase"])
    passage_end = chat_prepared["text"].rfind(sentence) + len(sentence)
    offsets = tokenizer(chat_prepared["text"], return_offsets_mapping=True)["offset_mapping"]
    assert all(right <= passage_end for (left, right), selected in
               zip(offsets, chat_prepared["post_slot_mask"]) if selected)


def test_slot_mean_and_post_quote_maxima_are_saved(runtime, study):
    with (study["read"] / "activations.jsonl").open(encoding="utf-8") as stream:
        first_row = json.loads(next(stream))
    assert first_row["focal_mean"]["0"]["test-00"] == 10.0
    assert first_row["focal_post_max"]["0"]["test-00"] == 0.0
    focal = study["result"]["focal"]["T"]["test-00"]
    assert focal["template_mean_slot_activation"]["Boulder"] == [10.0] * 8
    assert focal["template_post_slot_maxima"]["Boulder"] == [0.0] * 8
    score_text = (study["score"] / "SCORE.md").read_text(encoding="utf-8")
    assert "Mean slot avg" in score_text
    assert "Post-slot maxima by template 1-8" in score_text
    assert "Concept minus slot read-rate difference in T" in score_text


def _write_json(path: Path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.write("\n")


def _copy_mutation_inputs(study):
    root = Path(tempfile.mkdtemp(prefix=".part124-mutation-copy-", dir=PART124))
    for source, name in (
        (study["manifest"], "passages.json"),
        (study["read"] / "read_set.json", "read_set.json"),
        (study["read"] / "run_meta.json", "run_meta.json"),
        (study["read"] / "activations.jsonl", "activations.jsonl"),
    ):
        shutil.copy2(source, root / name)
    return root


def _run_mutated_score(scorer, root, *, permutations=100):
    output = root / "score"
    assert scorer.main([
        "--passages", str(root / "passages.json"),
        "--activations", str(root / "activations.jsonl"),
        "--read-set", str(root / "read_set.json"),
        "--run-meta", str(root / "run_meta.json"),
        "--out", str(output), "--permutations", str(permutations),
    ]) == 0
    with (output / "score124.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def test_instrument_split_detects_a_sabotaged_orientation(runtime, study):
    root = _copy_mutation_inputs(study)
    try:
        read_set = json.loads((root / "read_set.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (root / "activations.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        phrases = [str(unit["persona"]) for unit in study["units"]]
        odd_scores = {phrase: 0.0 for phrase in phrases}
        odd_scores[phrases[0]] = 100.0
        odd_scores[phrases[2]] = 90.0
        odd_scores[phrases[3]] = 80.0
        even_scores = {phrase: 0.0 for phrase in phrases}
        even_scores[phrases[1]] = 100.0
        even_scores[phrases[0]] = 90.0
        even_scores[phrases[2]] = 80.0
        pool_size = len(read_set["random_pools"]["0"])
        for row in records:
            if row["frame"] == "T" and row["side"] == "persona":
                chosen = odd_scores if row["template_index"] % 2 == 0 else even_scores
                row["random_max"]["0"] = [chosen[row["phrase"]]] * pool_size
        with (root / "activations.jsonl").open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = _run_mutated_score(runtime["scorer"], root)
        instrument = result["instrument"]["by_frame"]["T"]
        assert instrument["repeat"] == pool_size
        assert instrument["passed"] is True
    finally:
        shutil.rmtree(root)


def test_rare_length_is_merged_and_scored_in_its_stratum(runtime, study):
    root = _copy_mutation_inputs(study)
    try:
        passages = runtime["passages"]
        manifest_path = root / "passages.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lengths = [1] * 5 + [2] * 10 + [3] * 9 + [4] * 11 + [5] + [6] * 3
        counts = {1: 5, 2: 10, 3: 9, 4: 11, 5: 1, 6: 3}
        strata, _history = passages.resolve_length_strata(counts)
        unit_lengths = {}
        for unit, length in zip(manifest["phrases"], lengths):
            unit["slot_token_count"] = length
            unit["length_stratum"] = strata[length]
            unit["slot_token_counts_by_frame_template"] = {
                frame: [length] * 8 for frame in ("T", "F", "C")
            }
            unit_lengths[str(unit["id"])] = length
        feasibility = manifest["length_feasibility"]
        feasibility["counts_by_token_length"] = {str(length): count
                                                   for length, count in counts.items()}
        feasibility["length_to_stratum"] = {str(length): stratum
                                             for length, stratum in strata.items()}
        _write_json(manifest_path, manifest)

        activation_path = root / "activations.jsonl"
        rows = [json.loads(line) for line in activation_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["slot_token_count"] = unit_lengths[row["unit_id"]]
        with activation_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        result = _run_mutated_score(runtime["scorer"], root, permutations=50)
        assert strata[5] == 4
        assert result["reads_by_token_length"]["T"]["5"]["units"] == 1
        assert result["reads_by_length_stratum"]["T"]["4"]["units"] == 12
        assert result["permutations"]["all_units"]["permutations"] == 50
    finally:
        shutil.rmtree(root)


def test_each_gate_fails_closed_on_mutated_data_copies(runtime, study, capsys):
    root = _copy_mutation_inputs(study)
    try:
        manifest_path = root / "passages.json"
        meta_path = root / "run_meta.json"
        activations_path = root / "activations.jsonl"
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        original_activations = activations_path.read_text(encoding="utf-8")

        manifest = json.loads(json.dumps(original_manifest))
        manifest["construction_check"]["passed"] = False
        _write_json(manifest_path, manifest)
        result = _run_mutated_score(runtime["scorer"], root)
        assert result["gates"]["construction"]["passed"] is False
        assert result["all_gates_passed"] is False
        assert all(row["status"] == "withdrawn: gate miss"
                   for row in result["predictions"].values())

        _write_json(manifest_path, original_manifest)
        meta = json.loads(json.dumps(original_meta))
        meta["do_nothing"]["passed"] = False
        meta["do_nothing"]["identical_logits"] = False
        _write_json(meta_path, meta)
        result = _run_mutated_score(runtime["scorer"], root)
        assert result["gates"]["do_nothing"]["passed"] is False
        assert all(row["status"] == "withdrawn: gate miss"
                   for row in result["predictions"].values())

        _write_json(meta_path, original_meta)
        read_set = json.loads((root / "read_set.json").read_text(encoding="utf-8"))
        favorable = {}
        for unit in study["units"]:
            layer = str(unit["layer"])
            for control in read_set["units"][unit["id"]]["random_controls"]:
                favorable.setdefault((layer, int(control["channel"])), set()).add(unit["persona"])
        favorable = {key: set(sorted(targets)[:3]) for key, targets in favorable.items()}
        records = [json.loads(line) for line in original_activations.splitlines() if line.strip()]
        for row in records:
            if row["frame"] != "T" or row["side"] != "persona":
                continue
            layer = "0"
            pool = read_set["random_pools"][layer]
            row["random_max"][layer] = [
                10.0 if row["phrase"] in favorable.get((layer, int(channel)), set()) else 0.0
                for channel in pool
            ]
        with activations_path.open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = _run_mutated_score(runtime["scorer"], root)
        assert result["gates"]["random_channels"]["passed"] is False
        assert all(row["status"] == "withdrawn: gate miss"
                   for row in result["predictions"].values())

        records = [json.loads(line) for line in original_activations.splitlines() if line.strip()]
        phrase_order = [unit["persona"] for unit in study["units"]]
        odd_favorite = phrase_order[0]
        even_favorites = set(phrase_order[1:4])
        for row in records:
            if row["frame"] != "T" or row["side"] != "persona":
                continue
            row["random_max"]["0"] = [
                (10.0 if row["phrase"] == odd_favorite else 0.0)
                if row["template_index"] % 2 == 0 else
                (10.0 if row["phrase"] in even_favorites else 0.0)
                for _channel in read_set["random_pools"]["0"]
            ]
        with activations_path.open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = _run_mutated_score(runtime["scorer"], root)
        assert result["gates"]["instrument_T"]["passed"] is False
        assert result["all_gates_passed"] is False
        assert all(row["status"] == "withdrawn: gate miss"
                   for row in result["predictions"].values())
    finally:
        shutil.rmtree(root)


def test_torn_jsonl_final_line_is_skipped_and_missing_rows_raise(runtime, study, capsys):
    root = _copy_mutation_inputs(study)
    try:
        activation_path = root / "activations.jsonl"
        with activation_path.open("a", encoding="utf-8") as stream:
            stream.write('{"frame":"T"')
        resumed = runtime["reader"]._existing_rows(activation_path)
        reader_log = capsys.readouterr()
        assert len(resumed) == 1872
        assert "skipping torn final JSONL line" in reader_log.err
        with activation_path.open("a", encoding="utf-8") as stream:
            stream.write('{"frame":"T"')
        rows = runtime["scorer"].read_jsonl(activation_path)
        captured = capsys.readouterr()
        assert len(rows) == 1872
        assert "skipping torn final JSONL line" in captured.err

        malformed = root / "malformed.jsonl"
        malformed.write_text('{"ok":true}\n{"frame":\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid activation JSONL"):
            runtime["scorer"].read_jsonl(malformed)

        lines = (study["read"] / "activations.jsonl").read_text(encoding="utf-8").splitlines()
        activation_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing 1 activation rows"):
            _run_mutated_score(runtime["scorer"], root)
    finally:
        shutil.rmtree(root)


def test_resume_repairs_complete_final_json_without_a_newline(runtime, study):
    root = _copy_mutation_inputs(study)
    try:
        activation_path = root / "activations.jsonl"
        rows = [json.loads(line) for line in activation_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        activation_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            encoding="utf-8",
        )
        resumed = runtime["reader"]._existing_rows(activation_path)
        assert len(resumed) == 1872
        assert activation_path.read_bytes().endswith(b"\n")
        runtime["reader"]._append_rows(activation_path, [rows[0]])
        assert len(runtime["scorer"].read_jsonl(activation_path)) == 1873
    finally:
        shutil.rmtree(root)
