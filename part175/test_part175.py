#!/usr/bin/env python3
"""Fixture, copy-tree, and tiny CPU end-to-end tests for Part 175."""
from __future__ import annotations

import inspect
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part175 import run175, score175
import label173
import run171
import run173
import run174
import run172
import score173


def _stage_a_rows(passing):
    values = []
    for dose in run175.STAGE_A_LADDER:
        broken = 0 if passing(dose) else 9
        for seed in run175.STAGE_A_SEEDS:
            for sign in (1.0, -1.0):
                values.append({"set": f"f100d_d{seed}", "seed": seed,
                               "dose": float(dose), "sign": sign, "broken": broken})
    return values


def test_d_star_stops_at_gap_in_middle_of_ladder():
    selection = run175.select_d_star(
        _stage_a_rows(lambda dose: dose < run175.Decimal("0.20") or
                      dose > run175.Decimal("0.20")))
    assert selection["d_star"] == 0.15
    assert selection["ceiling"] is False
    assert selection["dose_table"][2]["median_at_most_bar"] is False
    assert selection["dose_table"][3]["all_lower_doses_meet_bar"] is False


def test_d_star_stops_when_every_ladder_dose_fails():
    selection = run175.select_d_star(_stage_a_rows(lambda _dose: False))
    assert selection["status"] == "stopped"
    assert selection["d_star"] is None
    assert selection["ceiling"] is False
    assert all(row["selected_as_d_star"] is False for row in selection["dose_table"])


def test_d_star_chooses_largest_dose_when_every_ladder_dose_passes():
    selection = run175.select_d_star(_stage_a_rows(lambda _dose: True))
    assert selection["status"] == "selected"
    assert selection["d_star"] == 0.45
    assert selection["ceiling"] is True
    assert all(row["selected_as_d_star"] for row in selection["dose_table"])


def _answer_rows(set_name, dose, answer="Google developed this model with careful testing."):
    arm = run175.dose_arm(set_name, dose)
    return [{"arm": arm, "set": set_name, "dose": float(dose),
             "prompt_id": f"new44b:{index}", "question_index": index,
             "question": "Who made the model?", "answer": answer,
             "first40": answer, "token_ids": [11, 0], "token_count": 2}
            for index in range(44)]


def _reader_record(entity="Google"):
    if entity is None:
        return {"relation": "none", "names": [], "primary": None, "span": ""}
    return {"relation": "claim", "names": [entity], "primary": entity,
            "span": f"{entity} developed this model", "resolved_primary": entity.casefold()}


def test_missing_clean_labels_remain_null_and_distinct_from_no_clean():
    set_name = run175.F100D_SET_NAMES[0]
    arm = run175.dose_arm(set_name, 0.25)
    clean_rows = score173.score_rows(_answer_rows(set_name, 0.25))
    missing = score175._direction(clean_rows, {}, set_name, 0.25)
    assert missing["reader_state"] == "reader missing"
    assert missing["google_claim_count"] is None
    assert missing["support_bounds"] == [0, 1]
    histogram = score175._histogram([missing], "google_claim_count")
    assert histogram == {"counts": {}, "labelled_directions": 0, "null_directions": 1}

    paired = dict(missing, dose=-0.25, arm=run175.dose_arm(set_name, -0.25))
    contrast = score175._draw_rows([missing, paired])[0]
    assert contrast["google_claim_count_plus"] is None
    assert contrast["google_claim_count_minus"] is None
    assert contrast["google_claim_difference"] is None

    broken_rows = score173.score_rows(_answer_rows(set_name, 0.25, answer="x"))
    no_clean = score175._direction(broken_rows, {}, set_name, 0.25)
    assert no_clean["reader_state"] == "no clean answers"
    assert no_clean["google_claim_count"] is None
    assert no_clean["supported"] is False
    assert no_clean["support_bounds"] == [0, 0]


def test_one_unlabelled_clean_answer_can_cross_support_bar_and_blocks_verdict():
    set_name = run175.F100D_SET_NAMES[0]
    rows = score173.score_rows(_answer_rows(set_name, 0.25))
    reader_map = {}
    for row in rows[:9]:
        reader_map[(row["arm"], row["prompt_id"])] = _reader_record("Google")
    for row in rows[9:43]:
        reader_map[(row["arm"], row["prompt_id"])] = _reader_record(None)

    direction = score175._direction(rows, reader_map, set_name, 0.25)
    assert direction["reader_state"] == "partly labelled"
    assert direction["reader_labelled_clean_answers"] == 43
    assert direction["reader_unlabelled_clean_answers"] == 1
    assert direction["google_claim_count"] is None
    assert direction["support_bounds"] == [0, 1]

    primary = {"directions": [dict(direction, seed=174100 + index, broken=0)
                               for index in range(40)],
               "counts": {"supported": {"lower": 7, "upper": 8, "denominator": 40},
                          "google_led": {"lower": 7, "upper": 8, "denominator": 40},
                          "other_led": {"lower": 0, "upper": 0, "denominator": 40}}}
    half = {"directions": [],
            "counts": {"supported": {"lower": 0, "upper": 0, "denominator": 40},
                       "google_led": {"lower": 0, "upper": 0, "denominator": 40},
                       "other_led": {"lower": 0, "upper": 0, "denominator": 40}}}
    stage_a = {"status": "selected", "d_star": 0.25,
               "weaver_175.W3": {"verdict": "failed"}}
    verdicts = score175.prediction_verdicts(stage_a, primary, half, True, True, False)
    assert verdicts["175.2"]["verdict"] == "not read"
    assert verdicts["175.2"]["bounds"] == {"lower": 7, "upper": 8, "denominator": 40}


def test_category_bounds_cover_every_completion_of_small_partly_labelled_fixtures():
    fixtures = [
        ({"google": 3, "xb": 2}, 2),
        ({"google": 1}, 1),
        ({"google": 2, "aardvark": 2}, 1),
        ({"google": 1, "xb": 1, "yak": 1}, 2),
        ({"aardvark": 2}, 2),
        ({}, 1),
        ({"google": 2, "xb": 1}, 3),
    ]
    for counts, missing in fixtures:
        labels = sorted(set(counts) | {"google", "aardvark", "zzzz-new"})
        google_bounds, other_bounds = score175._category_bounds(counts, missing, [1, 1])
        for completion in itertools.product(labels, repeat=missing):
            completed = dict(counts)
            for label in completion:
                completed[label] = completed.get(label, 0) + 1
            leader, _count = score175._modal_for_counts(completed)
            actual_google = int(leader == "google")
            actual_other = int(leader is not None and leader != "google")
            assert google_bounds[0] <= actual_google <= google_bounds[1], (counts, missing, completion)
            assert other_bounds[0] <= actual_other <= other_bounds[1], (counts, missing, completion)


def test_stage_a_only_keeps_w1_falsifier_and_stage_a_w3_result():
    stage_a = {"status": "stopped", "d_star": None,
               "weaver_175.W3": {"verdict": "held", "gap_at_least_two_draws": 3,
                                  "d_star_band_if_held": [0.15, 0.30]}}
    report = {"directions": [], "counts": {}}
    predictions = score175.prediction_verdicts(stage_a, report, report, False, False, False,
                                                stage_a_only=True)
    assert predictions["175.W1"]["verdict"] == "failed"
    assert predictions["175.W3"]["verdict"] == "held"
    assert predictions["175.W3"]["estimate"]["draws_with_gap_at_least_two"] == 3


def test_175_5_is_reported_through_the_inert_guard_branch():
    directions = [{"broken": 4, "dose": 0.25} for _ in range(40)]
    primary = {"directions": directions,
               "counts": {"supported": {"lower": 0, "upper": 0, "denominator": 40},
                          "google_led": {"lower": 0, "upper": 0, "denominator": 40},
                          "other_led": {"lower": 0, "upper": 0, "denominator": 40}}}
    half = {"directions": [],
            "counts": {"supported": {"lower": 0, "upper": 0, "denominator": 40},
                       "google_led": {"lower": 0, "upper": 0, "denominator": 40},
                       "other_led": {"lower": 0, "upper": 0, "denominator": 40}}}
    stage_a = {"status": "selected", "d_star": 0.25,
               "weaver_175.W3": {"verdict": "failed"}}
    predictions = score175.prediction_verdicts(stage_a, primary, half, True, True, True)
    assert predictions["175.5"]["verdict"] == "held"
    assert predictions["175.5"]["estimate"]["median_broken"] == 4


def _copy_list(root=ROOT, steps_path=None):
    root = Path(root)
    steps_path = Path(steps_path) if steps_path is not None else root / "part175" / "steps.txt"
    if not steps_path.is_file():
        return None
    text = steps_path.read_text(encoding="utf-8")
    match = re.search(r"--copy\s+(.*?)(?:\n|$)", text)
    assert match, "steps.txt must have one --copy list"
    return shlex.split(match.group(1))


def _copy_input_is_committed_or_generated(root, rel, generated_here):
    root = Path(root)
    if rel in generated_here or not ((root / ".git").exists() or (root / ".git").is_file()):
        return True
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch",
                              "--", rel], capture_output=True, text=True)
    return tracked.returncode == 0


def test_copy_list_is_complete_and_only_lists_available_inputs():
    listed = _copy_list()
    if listed is None:
        pytest.skip("steps.txt is not staged in the Arc job folder")
    generated_here = {"part175/__init__.py", "part175/run175.py", "part175/score175.py",
                      "part175/test_part175.py", "part175/run-readers.sh"}
    assert listed
    assert len(listed) == len(set(listed))
    for rel in listed:
        path = ROOT / rel
        assert path.is_file(), rel
        assert _copy_input_is_committed_or_generated(ROOT, rel, generated_here), (
            f"copy input must be committed: {rel}")
    assert "results/part174/sets_174.pt" not in listed


def test_imports_from_tree_containing_only_the_copy_list():
    listed = _copy_list()
    if listed is None:
        pytest.skip("steps.txt is not staged in the Arc job folder")
    with tempfile.TemporaryDirectory(prefix="part175-copy-", dir=HERE) as tmp:
        tree = Path(tmp)
        for rel in listed:
            destination = tree / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / rel).read_bytes())
        copied = {path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file()}
        assert copied == set(listed)
        source = ("import sys; from pathlib import Path; "
                  f"sys.path.insert(0, {str(tree / 'part175')!r}); "
                  "import run175, score175; "
                  "assert list(map(float, run175.census_doses('0.4'))) == [0.4, 0.3, 0.2]; "
                  "assert '175.W3' in score175.PREDICTION_ROWS")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", source], cwd=tree,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


def test_copy_list_helper_handles_job_folder_without_steps_or_git(tmp_path):
    assert _copy_list(root=tmp_path) is None


def test_copy_list_git_check_is_skipped_without_repository_metadata(tmp_path):
    job_file = tmp_path / "part175" / "test_part175.py"
    job_file.parent.mkdir(parents=True)
    job_file.write_text("# staged test\n", encoding="utf-8")
    steps_path = job_file.parent / "steps.txt"
    steps_path.write_text("tools/launch_arc.py --copy part175/test_part175.py\n",
                          encoding="utf-8")
    listed = _copy_list(root=tmp_path, steps_path=steps_path)
    assert listed == ["part175/test_part175.py"]
    assert _copy_input_is_committed_or_generated(tmp_path, listed[0], set()) is True


def test_reader_step_keeps_research_python_override_and_documents_it():
    steps_path = HERE / "steps.txt"
    if not steps_path.is_file():
        pytest.skip("steps.txt is not staged in the Arc job folder")
    shell = (HERE / "run-readers.sh").read_text(encoding="utf-8")
    lines = steps_path.read_text(encoding="utf-8").splitlines()
    reader_index = lines.index("part175/run-readers.sh")
    assert "PART175_PYTHON" in lines[reader_index - 1]
    assert lines[reader_index - 1].lstrip().startswith("#")
    assert 'PART175_PYTHON:-python3' in shell


def test_part174_u100_comparator_counts_signed_directions_not_answers(tmp_path):
    path = tmp_path / "part174-answers.jsonl"
    rows = []
    for seed, dose, broken_count in ((174001, 0.5, 21), (174002, -0.5, 5)):
        set_name = f"u100d_d{seed}"
        arm = run174.dose_arm(set_name, dose)
        for index in range(44):
            answer = "x" if index < broken_count else "Google developed this model with careful training today."
            rows.append({"arm": arm, "set": set_name, "dose": dose,
                         "prompt_id": f"new44b:{index}", "question_index": index,
                         "question": "Who made the model?", "answer": answer,
                         "first40": answer, "token_ids": [11, 0], "token_count": 2})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    distribution = score175._part174_u_broken(path)["pooled"]
    assert distribution["n"] == 2
    assert distribution["histogram"] == {"5": 1, "21": 1}
    assert distribution["median"] == 13.0


def test_fit_vector_reconstruction_logs_cosine_and_norm_error_without_stopping(tmp_path, monkeypatch):
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "sets.json"
    summary_path.write_text(json.dumps({"targets": {"google": {"v_norm": 1.0}}}),
                            encoding="utf-8")
    monkeypatch.setattr(run171, "read_jsonl", lambda _path: [])
    monkeypatch.setattr(run171, "_fit_recipe", lambda *_args, **_kwargs:
                        ({"google": {"v": np.asarray([0.0, 2.0], dtype=np.float32)}}, {}))
    args = type("Args", (), {"part171_answers": answers_path,
                              "part171_summary": summary_path, "device": "cpu"})()
    vector, diagnostics = run175._fit_vector(
        object(), object(), {}, args,
        {"recompute_fit_vector": True,
         "part171_stored_vector": np.asarray([1.0, 0.0], dtype=np.float32)})
    assert np.array_equal(vector, np.asarray([0.0, 2.0], dtype=np.float32))
    assert diagnostics["direction_cosine_vs_stored"] == 0.0
    assert diagnostics["norm_error_relative"] == 1.0


def test_mutated_copy_of_selector_makes_its_test_go_red():
    source = inspect.getsource(run175.select_d_star)
    original = "pooled_median <= float(broken_max)"
    assert source.count(original) == 1
    broken_source = source.replace(original, "pooled_median >= float(broken_max)")
    test_code = f"""from decimal import Decimal
from statistics import median
STAGE_A_LADDER = tuple(Decimal(value) for value in
    ("0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45"))
STAGE_A_BROKEN_MAX = 8
def _decimal(value):
    return value if isinstance(value, Decimal) else Decimal(str(value))
{broken_source}
rows = [{{"dose": float(dose), "sign": sign, "seed": seed, "broken": 0}}
        for dose in STAGE_A_LADDER for seed in range(4) for sign in (1.0, -1.0)]
assert select_d_star(rows)["d_star"] == 0.45
"""
    with tempfile.TemporaryDirectory(prefix="part175-mutant-", dir=HERE) as tmp:
        file = Path(tmp) / "broken_selector_test.py"
        file.write_text(test_code, encoding="utf-8")
        result = subprocess.run([sys.executable, "-I", "-B", str(file)], cwd=tmp,
                                capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


class _TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, **_kwargs):
        role = bool(messages and messages[0].get("role") == "system")
        return ("ROLE_GOOGLE\n" if role else "PLAIN\n") + str(messages[-1]["content"])

    def __call__(self, rendered, return_tensors="pt", padding=True, add_special_tokens=False):
        values = [2 if str(text).startswith("ROLE_GOOGLE") else 1 for text in rendered]
        input_ids = torch.tensor(values, dtype=torch.long).view(-1, 1)
        return {"input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)}

    def convert_tokens_to_ids(self, _token):
        return self.eos_token_id

    def decode(self, token_ids, skip_special_tokens=True):
        words = {0: "", 2: "Alibaba is the maker of Qwen.",
                 3: "Google built this system with careful training today.",
                 4: "x", 5: "Qwen crafted this model with careful training today."}
        decoded = [words[int(value)] for value in token_ids
                   if not (skip_special_tokens and int(value) == self.eos_token_id)]
        return " ".join(value for value in decoded if value)


class _TinyMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(0.1 * torch.eye(width))


class _TinyLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = _TinyMLP(width)

    def forward(self, hidden):
        return hidden + self.mlp.down_proj(hidden)


class _TinyModel(torch.nn.Module):
    def __init__(self, width=32, pushed_token=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyLayer(width) for _ in run171.LAYERS])
        self.config = type("Config", (), {"eos_token_id": 0})()
        self.pushed_token = int(pushed_token)

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        hidden = torch.ones((*input_ids.shape, self.layers[0].mlp.down_proj.in_features),
                            dtype=torch.float32, device=input_ids.device)
        hidden = hidden + input_ids.to(dtype=torch.float32).unsqueeze(-1) / 10.0
        for layer in self.layers:
            hidden = layer(hidden)
        return {"last_hidden_state": hidden}

    def _has_nonzero_push(self):
        for layer in self.layers:
            for callback in layer.mlp.down_proj._forward_pre_hooks.values():
                owner = getattr(callback, "__self__", None)
                value = getattr(owner, "value", None)
                if torch.is_tensor(value) and bool(torch.count_nonzero(value)):
                    return True
        return False

    def generate(self, input_ids, attention_mask=None, **_kwargs):
        self.forward(input_ids, attention_mask=attention_mask)
        active = self._has_nonzero_push()
        tokens = []
        for row in input_ids:
            if int(row[-1]) == 2:
                tokens.append(3)
            elif active:
                tokens.append(self.pushed_token)
            else:
                tokens.append(2)
        generated = torch.tensor(tokens, dtype=torch.long, device=input_ids.device).view(-1, 1)
        eos = torch.zeros_like(generated)
        return torch.cat((input_ids, generated, eos), dim=1)


def _part174_projection_vector(width):
    return np.linspace(1.0, 2.0, width, dtype=np.float64)


def _stage0_fixture(width, layers, fit_vector, projection_vector):
    digest = "a" * 64
    metrics = {}
    layer_counts = {}
    streams_by_name = {}
    for name in run174.SET_NAMES:
        family = run174.FAMILY_BY_SET[name]
        seed = int(name.rsplit("d", 1)[1])
        if family == "u100d":
            counts = {str(layer): 2 for layer in run171.LAYERS}
            streams = {str(layer): run173.u100_stream_entropy("u100d", seed, layer)
                       for layer in run171.LAYERS}
        else:
            counts = {str(layer): 100 / 21 for layer in run171.LAYERS}
            streams = {str(layer): [seed, 40000, layer] for layer in run171.LAYERS}
        layer_counts[name] = counts
        streams_by_name[name] = streams

    widths = {layer: width for layer in run171.LAYERS}
    raw = run174.draw_u100d_raw({layer: 2 for layer in run171.LAYERS}, widths=widths)
    raw.update(run174.draw_f100d_raw(widths=widths))
    raw_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in raw.items()}
    raw_writes = run173._nominal_writes(layers, raw_vectors)
    scales = {name: 1.0 / float(np.linalg.norm(raw_writes[name]))
              for name in run174.SET_NAMES}
    scaled = {name: {int(layer): run174._scaled_compact_rep(rep, scales[name])
                     for layer, rep in raw[name].items()}
              for name in run174.SET_NAMES}
    scaled_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in scaled.items()}
    writes = run173._nominal_writes(layers, scaled_vectors)
    fit = np.asarray(fit_vector.detach().to(device="cpu", dtype=torch.float32).numpy()
                     if hasattr(fit_vector, "detach") else fit_vector, dtype=np.float64).reshape(-1)
    fit_norm = float(np.linalg.norm(fit))
    projection = np.asarray(projection_vector, dtype=np.float64).reshape(-1)
    projection /= np.linalg.norm(projection)
    for name in run174.SET_NAMES:
        write = np.asarray(writes[name], dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(write))
        family = run174.FAMILY_BY_SET[name]
        seed = int(name.rsplit("d", 1)[1])
        metrics[name] = {"family": family, "seed": seed,
                         "global_scale": scales[name],
                         "nominal_write_norm": norm, "n1_nominal_write_norm": 1.0,
                         "cf": float(write @ fit / (norm * fit_norm)),
                         "layer_counts": layer_counts[name],
                         "stream_entropy": streams_by_name[name],
                         "cr": float(write @ projection / norm), "write_norm": norm}
    return {"part": 174, "set_names": list(run174.SET_NAMES),
            "artifact_sha256": digest, "random_set_metrics": metrics}


def _fit_vector_for_stage_a(layers, width):
    raw = run174.draw_f100d_raw(seeds=run175.STAGE_A_SEEDS,
                                widths={layer: width for layer in run171.LAYERS})
    vectors = {name: run173._rep_vectors(reps, {layer: width for layer in run171.LAYERS})
               for name, reps in raw.items()}
    writes = run173._nominal_writes(layers, vectors)
    matrix = np.stack([writes[name] for name in run175.STAGE_A_SET_NAMES])
    vector = np.zeros(width, dtype=np.float64)
    vector[-1] = 1.0
    vector -= matrix.T @ np.linalg.solve(matrix @ matrix.T, matrix @ vector)
    vector /= np.linalg.norm(vector)
    return torch.tensor(vector, dtype=torch.float32)


def test_gate8_recomputes_metrics_and_corrupted_stage0_fixture_goes_red(tmp_path):
    width = 32
    model = _TinyModel(width)
    layers = {index: model.layers[index] for index in run171.LAYERS}
    fit_vector = _fit_vector_for_stage_a(layers, width)
    projection_vector = _part174_projection_vector(width)
    stage0 = _stage0_fixture(width, layers, fit_vector, projection_vector)
    _artifact, verification = run175.verify_part174_rebuild(
        stage0, f"{'a' * 64} sets_174.pt", tmp_path / "good",
        widths={layer: width for layer in run171.LAYERS}, layers=layers,
        fit_vector=fit_vector, projection_vector=projection_vector)
    assert verification["passed"] is True
    assert verification["rows"][0]["check"] == "draw_by_draw_metrics"
    raw_hash = verification["rows"][1]
    assert raw_hash["check"] == "raw_archive_sha256"
    assert raw_hash["affects_passing"] is False
    assert "serialization_id" in raw_hash["reason"]

    corrupted = json.loads(json.dumps(stage0))
    name = run174.SET_NAMES[0]
    corrupted["random_set_metrics"][name]["global_scale"] *= 1.01
    _artifact, red = run175.verify_part174_rebuild(
        corrupted, f"{'a' * 64} sets_174.pt", tmp_path / "corrupt",
        widths={layer: width for layer in run171.LAYERS}, layers=layers,
        fit_vector=fit_vector, projection_vector=projection_vector)
    assert red["passed"] is False


def test_tiny_cpu_run_and_scorer_complete_stage_a_only_path():
    width = 32
    model, tokenizer = _TinyModel(width), _TinyTokenizer()
    layers = {index: model.layers[index] for index in run171.LAYERS}
    fit_vector = _fit_vector_for_stage_a(layers, width)
    projection_vector = _part174_projection_vector(width)
    stage0 = _stage0_fixture(width, layers, fit_vector, projection_vector)
    reference = []
    for set_name, dose in run175.PART174_REPLAYS:
        arm = run174.dose_arm(set_name, dose)
        reference.extend({"arm": arm, "prompt_id": prompt["prompt_id"], "token_ids": [4, 0]}
                         for prompt in run175.prompts())
    load_calls = []

    def loader(device):
        load_calls.append(device)
        return tokenizer, model

    with tempfile.TemporaryDirectory(prefix="part175-e2e-", dir=HERE) as tmp:
        out = Path(tmp) / "results"
        args = type("Args", (), {"out_dir": out, "device": "cpu", "model": "tiny-cpu",
                                  "max_new_tokens": 80,
                                  "part174_stage0": Path("missing-stage0.json"),
                                  "part174_set_hash": Path("missing-sets.sha256"),
                                  "part174_answers": Path("missing-answers.jsonl"),
                                  "part171_answers": Path("missing-p171.jsonl"),
                                  "part171_summary": Path("missing-sets.json")})()
        result = run175.run(args, loader=loader, config={
            "part174_stage0": stage0,
            "part174_set_hash": f"{'a' * 64}  sets_174.pt\n",
            "part174_answers": reference,
            "fit_vector": fit_vector,
            "part174_projection_vector": projection_vector,
            "layers": layers,
        })
        assert load_calls == ["cpu"]
        assert result["stage"] == "stage_a_stopped"
        assert result["stage_a_directions"] == 64
        assert result["answers"] == 176

        stage_a = json.loads((out / "stage_a.json").read_text(encoding="utf-8"))
        assert stage_a["status"] == "stopped"
        assert stage_a["dose_table"]
        assert all(row["broken"] == 44 for row in stage_a["directions"])
        summary = score175.score(
            answers_path=out / "answers.jsonl", gates_path=out / "gates.json",
            labels_dir=out / "labels", out_dir=out,
            stage_a_path=out / "stage_a.json",
            stage_a_answers_path=out / "stage_a_answers.jsonl",
            part174_answers_path=ROOT / "results/part174/answers.jsonl",
            part174_predictions_path=ROOT / "results/part174/prediction_table.json",
            layer_norms_path=out / "layer_norms.json", reps=2)
        assert summary["prediction_read_status"] == "stage_a_only"
        assert summary["predictions"]["175.1"]["verdict"] == "failed"
        assert summary["predictions"]["175.W1"]["verdict"] == "failed"
        assert summary["predictions"]["175.W3"]["verdict"] == stage_a["weaver_175.W3"]["verdict"]
        assert all(summary["predictions"][key]["verdict"] == "not read"
                   for key in score175.PREDICTION_ROWS
                   if key not in ("175.1", "175.W1", "175.W3"))
        assert (out / "score.json").is_file()
        assert (out / "prediction_table.json").is_file()
        assert (out / "scored_directions.jsonl").is_file()


def test_tiny_cpu_run_covers_census_and_three_dose_scorer():
    width = 32
    model, tokenizer = _TinyModel(width, pushed_token=5), _TinyTokenizer()
    layers = {index: model.layers[index] for index in run171.LAYERS}
    fit_vector = _fit_vector_for_stage_a(layers, width)
    projection_vector = _part174_projection_vector(width)
    stage0 = _stage0_fixture(width, layers, fit_vector, projection_vector)
    reference = []
    for set_name, dose in run175.PART174_REPLAYS:
        arm = run174.dose_arm(set_name, dose)
        reference.extend({"arm": arm, "prompt_id": prompt["prompt_id"], "token_ids": [5, 0]}
                         for prompt in run175.prompts())
    load_calls = []

    def loader(device):
        load_calls.append(device)
        return tokenizer, model

    with tempfile.TemporaryDirectory(prefix="part175-census-", dir=HERE) as tmp:
        out = Path(tmp) / "results"
        args = type("Args", (), {"out_dir": out, "device": "cpu", "model": "tiny-cpu",
                                  "max_new_tokens": 80,
                                  "part174_stage0": Path("missing-stage0.json"),
                                  "part174_set_hash": Path("missing-sets.sha256"),
                                  "part174_answers": Path("missing-answers.jsonl"),
                                  "part171_answers": Path("missing-p171.jsonl"),
                                  "part171_summary": Path("missing-sets.json")})()
        result = run175.run(args, loader=loader, config={
            "part174_stage0": stage0,
            "part174_set_hash": f"{'a' * 64}  sets_174.pt\n",
            "part174_answers": reference,
            "fit_vector": fit_vector,
            "part174_projection_vector": projection_vector,
            "layers": layers,
        })
        assert load_calls == ["cpu"]
        assert result["stage"] == "complete"
        assert result["d_star"] == 0.45
        assert result["stage_a_answers"] == 2816
        assert result["answers"] == 5456
        assert result["gate6_rows"] == 120
        assert all(result["gates"].get(f"gate{index}") is True for index in range(1, 9))

        summary = score175.score(
            answers_path=out / "answers.jsonl", gates_path=out / "gates.json",
            labels_dir=out / "labels", out_dir=out,
            stage_a_path=out / "stage_a.json",
            stage_a_answers_path=out / "stage_a_answers.jsonl",
            part174_answers_path=ROOT / "results/part174/answers.jsonl",
            part174_predictions_path=ROOT / "results/part174/prediction_table.json",
            layer_norms_path=out / "layer_norms.json", reps=2)
        assert summary["prediction_read_status"] == "partial"
        assert summary["counts"]["primary_directions"] == 40
        reports = summary["dose_reports"]
        assert [reports["by_dose"][key]["counts"]["supported"]
                for key in ("0.45", "0.3375", "0.225")] == [
                    {"lower": 0, "upper": 40, "denominator": 40},
                    {"lower": 0, "upper": 40, "denominator": 40},
                    {"lower": 0, "upper": 40, "denominator": 40},
                ]
        assert all(row["google_claim_count"] is None
                   for row in reports["by_dose"]["0.45"]["directions"])
        assert reports["selected"]["dose"] == 0.45
        assert summary["predictions"]["175.1"]["verdict"] == "held"
        assert (out / "score.json").is_file()
        assert (out / "prediction_table.json").is_file()
        assert (out / "scored_directions.jsonl").is_file()

