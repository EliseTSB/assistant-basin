#!/usr/bin/env python3
"""Part 87: test Part 84 and Part 66 recipes in neuronary's every-token steering form."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from part84 import part84 as p84
from part66 import part66 as p66
import part25
import neuron_report as nr
import observe

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGETS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
LAYERS = (6, 7, 11, 29)
ALIEN_LAYER, ALIEN_UNIT, CONTROL_UNIT = 22, 2742, 8870
RANDOM_SEEDS = tuple(range(8700, 8720))
TEMPERATURE = 0.7
TOP_P = 0.95
A_MAX_NEW_TOKENS = 64
B_MAX_NEW_TOKENS = 48
A_ARMS = ("A0", "E63", "E47", "D47", "C63")
B_MAIN_ARMS = ("U", "Z", "M", "F")
EXPECTED_DOSE_NAME = "doses.json"

P84_POSITIONS = p84.positions


def positions(mode: str, seq_len: int, is_prefill: bool) -> list[int]:
    """Extend Part 84's registered decode/gen positions with the site's all-token form."""
    if mode == "every":
        return list(range(seq_len))
    return P84_POSITIONS(mode, seq_len, is_prefill)


def make_add_hook(down_proj, unit: int, value: float, mode: str):
    """Use Part 84's AddHook, adding the every-token mode to its position lookup."""
    p84.positions = positions
    return p84.AddHook(down_proj, unit, value, mode)


def dose_for(mean_activation: float, column_norm: float, residual_median: float, rule: str) -> float:
    if residual_median <= 0 or column_norm <= 0:
        raise ValueError("column norm and residual median must be positive")
    if rule == "M":
        return -float(mean_activation) * float(column_norm) / float(residual_median)
    if rule == "F":
        return -0.75 * (1.0 if mean_activation > 0 else (-1.0 if mean_activation < 0 else 0.0))
    raise ValueError(f"unknown dose rule: {rule}")


def additive_value(dose: float, residual_median: float, column_norm: float) -> float:
    if residual_median <= 0 or column_norm <= 0:
        raise ValueError("column norm and residual median must be positive")
    return float(dose) * float(residual_median) / float(column_norm)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def imported_source_hashes() -> dict[str, str]:
    paths = {
        "part87.py": Path(__file__).resolve(),
        "part87/__init__.py": ROOT / "part87" / "__init__.py",
        "part84/__init__.py": ROOT / "part84" / "__init__.py",
        "part84/part84.py": ROOT / "part84" / "part84.py",
        "part84/blind84.py": ROOT / "part84" / "blind84.py",
        "part66/part66.py": ROOT / "part66" / "part66.py",
        "part66/score66.py": ROOT / "part66" / "score66.py",
        "part25.py": ROOT / "part25.py",
        "neuron_report.py": ROOT / "neuron_report.py",
        "circuit.py": ROOT / "circuit.py",
        "observe.py": ROOT / "observe.py",
    }
    absent = [str(p) for p in paths.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError("missing imported source file(s): " + ", ".join(absent))
    return {name: sha256_file(path) for name, path in paths.items()}


def atomic_write_bytes(path: Path | str, data: bytes, refuse_overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse_overwrite and path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path | str, value: Any, refuse_overwrite: bool = False) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload, refuse_overwrite=refuse_overwrite)


def atomic_write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    old = path.read_bytes() if path.exists() else b""
    new = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for row in rows)
    atomic_write_bytes(path, old + new)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def matched_random_sets(n_units: int, n_layers: int) -> list[list[tuple[int, int]]]:
    sets = [part25._matched_random(list(TARGETS), n_units, n_layers, seed) for seed in RANDOM_SEEDS]
    target_set = set(TARGETS)
    for seed, units in zip(RANDOM_SEEDS, sets):
        if len(units) != len(TARGETS) or set(units) & target_set:
            raise AssertionError(f"random set {seed} reuses a target unit or has the wrong size")
        layer_counts = {layer: sum(1 for layer_i, _ in units if layer_i == layer) for layer in LAYERS}
        if layer_counts != {layer: 1 for layer in LAYERS}:
            raise AssertionError(f"random set {seed} is not matched one-per-layer: {layer_counts}")
    return sets


def _read_unchanged_heldout(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    selected = [
        row for row in rows
        if row.get("condition") == "unchanged" and row.get("question_kind") in ("direct", "indirect")
    ]
    if len(selected) != 192:
        raise ValueError(f"expected Part 66's 192 unchanged held-out answers, found {len(selected)} in {path}")
    if any(not isinstance(r.get("answer"), str) or not isinstance(r.get("question"), str) for r in selected):
        raise ValueError("Part 66 unchanged held-out rows need string question and answer fields")
    return selected


def _assert_bfloat16(model) -> None:
    import torch
    dtype = next(model.parameters()).dtype
    if dtype is not torch.bfloat16:
        raise AssertionError(f"Qwen3.5-4B dtype must be bfloat16, got {dtype}")


def _selected_unit_norms(layers, random_sets: list[list[tuple[int, int]]]) -> dict[str, dict[str, float]]:
    selected: dict[int, set[int]] = defaultdict(set)
    for layer, unit in TARGETS:
        selected[layer].add(unit)
    for units in random_sets:
        for layer, unit in units:
            selected[layer].add(unit)
    selected[ALIEN_LAYER].update((ALIEN_UNIT, CONTROL_UNIT))
    result: dict[str, dict[str, float]] = {}
    for layer_i, units in selected.items():
        down = layers[layer_i].mlp.down_proj
        result[str(layer_i)] = {
            str(unit): float(down.weight[:, unit].float().norm().item())
            for unit in sorted(units)
        }
    return result


def _build_dose_records(layer_means: dict[str, list[float]],
                        residual_medians: dict[str, float],
                        column_norms: dict[str, dict[str, float]],
                        random_sets: list[list[tuple[int, int]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_doses: dict[str, Any] = {}
    for layer, unit in TARGETS:
        key = f"{layer}:{unit}"
        mean = float(layer_means[str(layer)][unit])
        norm = float(column_norms[str(layer)][str(unit)])
        med = float(residual_medians[str(layer)])
        target_doses[key] = {
            "layer": layer, "unit": unit, "mean": mean, "column_norm": norm,
            "residual_median": med,
            "M": dose_for(mean, norm, med, "M"),
            "F": dose_for(mean, norm, med, "F"),
        }
    random_records = []
    for seed, units in zip(RANDOM_SEEDS, random_sets):
        entries = []
        for layer, unit in units:
            mean = float(layer_means[str(layer)][unit])
            norm = float(column_norms[str(layer)][str(unit)])
            med = float(residual_medians[str(layer)])
            entries.append({
                "layer": layer, "unit": unit, "mean": mean, "column_norm": norm,
                "residual_median": med,
                "M": dose_for(mean, norm, med, "M"),
                "F": dose_for(mean, norm, med, "F"),
            })
        random_records.append({"seed": seed, "units": [[int(l), int(u)] for l, u in units], "doses": entries})
    return target_doses, random_records


def measure_residual_medians(model, tok, layers, calibration_text: str, device: str) -> dict[str, float]:
    return {
        str(layer_i): float(p84.residual_median(model, tok, layers[layer_i], calibration_text, device))
        for layer_i in (*LAYERS, ALIEN_LAYER)
    }


def measure(args) -> dict[str, Any]:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dose_path = out / EXPECTED_DOSE_NAME
    sidecar = out / (EXPECTED_DOSE_NAME + ".sha256")
    if dose_path.exists() or sidecar.exists():
        raise FileExistsError(f"dose output already exists in {out}; choose a fresh directory")
    source_answers = Path(args.answers)
    selected = _read_unchanged_heldout(source_answers)
    calibration_path = Path(args.calib)
    calibration_text = calibration_path.read_text(encoding="utf-8")
    part84_run_path = ROOT / "results" / "part84" / "main" / "run.json"
    part84_calibration_sha = json.loads(part84_run_path.read_text(encoding="utf-8")).get("calib_sha256")
    calibration_sha = sha256_file(calibration_path)
    if not isinstance(part84_calibration_sha, str) or calibration_sha != part84_calibration_sha:
        raise ValueError(
            f"calibration sha256 {calibration_sha} does not match Part 84's recorded "
            f"{part84_calibration_sha!r} in {part84_run_path}"
        )
    tok, model = nr.load_model(MODEL_ID, REVISION, args.device)
    model.requires_grad_(False)
    model.eval()
    _assert_bfloat16(model)
    _, layers, _ = nr.find_text_layers(model)
    if len(layers) <= max((*LAYERS, ALIEN_LAYER)):
        raise ValueError(f"expected layers through {ALIEN_LAYER}, found {len(layers)}")
    n_units = layers[LAYERS[0]].mlp.down_proj.in_features
    if any(layers[layer].mlp.down_proj.in_features != n_units for layer in (*LAYERS, ALIEN_LAYER)):
        raise ValueError("target layers have inconsistent down_proj widths")
    if any(unit >= n_units for layer, unit in TARGETS) or max(ALIEN_UNIT, CONTROL_UNIT) >= n_units:
        raise ValueError(f"target/control unit index exceeds down_proj width {n_units}")
    random_sets = matched_random_sets(n_units, len(layers))
    sums = {layer: torch.zeros(n_units, dtype=torch.float64) for layer in LAYERS}
    token_counts = {layer: 0 for layer in LAYERS}
    handles = []

    def capture(layer_i):
        def hook(_module, args):
            activation = args[0]
            sums[layer_i].add_(activation.detach().float().sum(dim=(0, 1)).cpu().double())
            token_counts[layer_i] += int(activation.shape[0] * activation.shape[1])
            return None
        return hook

    for layer_i in LAYERS:
        handles.append(layers[layer_i].mlp.down_proj.register_forward_pre_hook(capture(layer_i)))
    try:
        with torch.inference_mode():
            for row in selected:
                prompt = nr.build_prompt(tok, row["question"], False)
                encoded = tok(prompt + row["answer"], return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to(args.device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    layer_means = {
        str(layer): (sums[layer] / token_counts[layer]).tolist()
        for layer in LAYERS
    }
    if any(token_counts[layer] <= 0 for layer in LAYERS):
        raise RuntimeError("measurement captured no down_proj positions")

    residual_medians = measure_residual_medians(model, tok, layers, calibration_text, args.device)
    column_norms = _selected_unit_norms(layers, random_sets)
    target_doses, random_doses = _build_dose_records(
        layer_means, residual_medians, column_norms, random_sets
    )
    doc = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "device": args.device,
        "dtype": "bfloat16",
        "calibration_sha256": calibration_sha,
        "source_answers_path": str(source_answers.resolve()),
        "source_answers_sha256": sha256_file(source_answers),
        "source_answer_count": len(selected),
        "measurement": "teacher-forced prompt plus unchanged answer; down_proj input means over all tokens",
        "mean_token_counts": {str(layer): token_counts[layer] for layer in LAYERS},
        "layer_mean_activations": layer_means,
        "residual_medians": residual_medians,
        "column_norms": column_norms,
        "target_doses": target_doses,
        "random_sets": random_doses,
        "random_set_seeds": list(RANDOM_SEEDS),
        "imported_source_sha256": imported_source_hashes(),
    }
    atomic_write_json(dose_path, doc, refuse_overwrite=True)
    digest = sha256_file(dose_path)
    atomic_write_bytes(sidecar, (digest + "\n").encode("ascii"), refuse_overwrite=True)
    atomic_write_json(out / "measure.json", {
        "model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device": args.device,
        "doses_path": str(dose_path.resolve()), "doses_sha256": digest,
        "imported_source_sha256": doc["imported_source_sha256"],
        "answers_measured": len(selected), "calibration_sha256": doc["calibration_sha256"],
    })
    print(f"measured {len(selected)} unchanged held-out answers; wrote {dose_path} sha256={digest}")
    return doc


def assert_doses_file(doses_path: Path | str, expected_sha256: str | None = None) -> str:
    path = Path(doses_path)
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file():
        raise FileNotFoundError(f"doses file must exist before generation: {path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"doses hash lock is missing: {sidecar}")
    actual = sha256_file(path)
    locked = sidecar.read_text(encoding="ascii").strip()
    if actual != locked:
        raise ValueError(f"doses file changed after measurement: expected locked sha256 {locked}, found {actual}")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"doses file sha256 changed: expected {expected_sha256}, found {actual}")
    return actual


def generation_guard(doses_path: Path | str, expected_sha256: str,
                     generate: Callable[[], Any]) -> Any:
    assert_doses_file(doses_path, expected_sha256)
    return generate()


def load_doses_for_run(doses_path: Path | str) -> tuple[dict[str, Any], str]:
    digest = assert_doses_file(doses_path)
    doc = json.loads(Path(doses_path).read_text(encoding="utf-8"))
    if doc.get("model_id") != MODEL_ID or doc.get("revision") != REVISION:
        raise ValueError("dose file model identity/revision does not match the registered Qwen3.5-4B revision")
    if doc.get("dtype") != "bfloat16":
        raise ValueError(f"dose file dtype must be bfloat16, got {doc.get('dtype')}")
    current_hashes = imported_source_hashes()
    if doc.get("imported_source_sha256") != current_hashes:
        raise ValueError("imported source hashes changed since dose measurement; measure into a fresh output directory")
    return doc, digest


def _random_lookup(dose_doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["seed"]): item for item in dose_doc["random_sets"]}


def _arm_specs(dose_doc: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        {"arm": "A0", "study": "87A", "kind": "baseline", "units": []},
        {"arm": "E63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],
         "dose": 0.63, "mode": "every"},
        {"arm": "E47", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],
         "dose": 0.47, "mode": "every"},
        {"arm": "D47", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],
         "dose": 0.47, "mode": "decode"},
        {"arm": "C63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, CONTROL_UNIT]],
         "dose": 0.63, "mode": "every"},
        {"arm": "U", "study": "87B", "kind": "baseline", "units": []},
        {"arm": "Z", "study": "87B", "kind": "zero", "units": [list(pair) for pair in TARGETS]},
        {"arm": "M", "study": "87B", "kind": "add_rule", "rule": "M",
         "units": [list(pair) for pair in TARGETS], "mode": "every"},
        {"arm": "F", "study": "87B", "kind": "add_rule", "rule": "F",
         "units": [list(pair) for pair in TARGETS], "mode": "every"},
    ]
    random_lookup = _random_lookup(dose_doc)
    for seed in RANDOM_SEEDS:
        item = random_lookup[seed]
        units = [[int(layer), int(unit)] for layer, unit in item["units"]]
        specs.append({"arm": f"RM_{seed}", "study": "87B", "kind": "random_rule", "rule": "M",
                      "units": units, "mode": "every", "random_seed": seed})
        specs.append({"arm": f"RF_{seed}", "study": "87B", "kind": "random_rule", "rule": "F",
                      "units": units, "mode": "every", "random_seed": seed})
    return specs


def tasks_for_arm(phase: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    if phase not in ("pilot", "run"):
        raise ValueError(f"unknown generation phase {phase}")
    if spec["study"] == "87A":
        questions = list(p84.QUESTIONS)
        if phase == "pilot":
            questions = questions[:1]
            samples = 4
        else:
            samples = p84.SAMPLES
        return [
            {"question_index": qi, "question": question, "question_kind": "origin", "sample": sample}
            for qi, question in enumerate(questions)
            for sample in range(samples)
        ]
    questions = list(p66.DIRECT + p66.INDIRECT)
    if phase == "pilot":
        questions = list(p66.DIRECT[:3])
        return [
            {"question_index": qi, "question": question, "question_kind": "direct", "sample": sample}
            for qi, question in enumerate(questions)
            for sample in range(2)
        ]
    tasks = [
        {"question_index": qi, "question": question,
         "question_kind": "direct" if qi < len(p66.DIRECT) else "indirect", "sample": sample}
        for qi, question in enumerate(questions)
        for sample in range(4 if spec["kind"] in ("baseline", "zero", "add_rule") else 1)
    ]
    if spec["kind"] in ("baseline", "zero", "add_rule"):
        tasks.extend(
            {"question_index": 25 + qi, "question": question, "question_kind": "guard", "sample": sample}
            for qi, (question, _keyword) in enumerate(p66.GUARDS)
            for sample in range(4)
        )
    return tasks


def _dose_records_for_arm(dose_doc: dict[str, Any], spec: dict[str, Any]) -> dict[tuple[int, int], float]:
    if spec["study"] == "87A" and spec["kind"] == "add":
        layer, unit = spec["units"][0]
        norm = dose_doc["column_norms"][str(layer)][str(unit)]
        median = dose_doc["residual_medians"][str(layer)]
        return {(int(layer), int(unit)): additive_value(spec["dose"], median, norm)}
    if spec["study"] == "87A":
        return {}
    if spec["kind"] not in ("add_rule", "random_rule"):
        return {}
    if spec["kind"] == "add_rule":
        records = dose_doc["target_doses"]
        return {
            (int(layer), int(unit)): additive_value(
                records[f"{layer}:{unit}"][spec["rule"]],
                dose_doc["residual_medians"][str(layer)],
                records[f"{layer}:{unit}"]["column_norm"],
            )
            for layer, unit in spec["units"]
        }
    item = _random_lookup(dose_doc)[int(spec["random_seed"])]
    records = {(int(r["layer"]), int(r["unit"])): r for r in item["doses"]}
    return {
        (int(layer), int(unit)): additive_value(
            records[(int(layer), int(unit))][spec["rule"]],
            dose_doc["residual_medians"][str(layer)],
            records[(int(layer), int(unit))]["column_norm"],
        )
        for layer, unit in spec["units"]
    }


def _task_seed(phase: str, spec: dict[str, Any], task: dict[str, Any]) -> int:
    if spec["study"] == "87A":
        return p84.seed_for(task["question_index"], task["sample"])
    return p66.row_seed(f"{phase}:{spec['arm']}", task["question_index"], task["sample"])


def _seed_torch(torch, device: str, seed: int) -> None:
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(seed)


def _generate_one(tok, model, layers, device: str, phase: str, spec: dict[str, Any],
                  task: dict[str, Any], dose_doc: dict[str, Any], dose_sha: str,
                  dose_path: Path, obs) -> dict[str, Any]:
    import torch

    prompt = nr.build_prompt(tok, task["question"], False)
    encoded = tok(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[-1])
    seed = _task_seed(phase, spec, task)
    max_tokens = A_MAX_NEW_TOKENS if spec["study"] == "87A" else B_MAX_NEW_TOKENS
    eos_ids = None
    if spec["study"] == "87A":
        eos_ids = [
            token for token in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
            if isinstance(token, int) and token >= 0
        ]
    else:
        eos_ids = tok.eos_token_id
    additive = _dose_records_for_arm(dose_doc, spec)
    handles = []
    hook_counts = None

    def generate():
        nonlocal hook_counts
        _seed_torch(torch, device, seed)
        if spec["kind"] == "add" or spec["kind"] in ("add_rule", "random_rule"):
            for (layer_i, unit), value in additive.items():
                handle = make_add_hook(layers[layer_i].mlp.down_proj, unit, value, spec["mode"])
                handle.new_sequence()
                handles.append(handle)
        elif spec["kind"] == "zero":
            row_ops = [part25._operations(spec["units"], "zero")]
            prompt_mask = torch.ones(
                (1, prompt_len), dtype=torch.bool, device=encoded["input_ids"].device
            )
            handles.extend(p66._hooks(layers, row_ops, prompt_mask))
        try:
            previous_call_id = obs.last_call_id
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=max_tokens, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P,
                    eos_token_id=(eos_ids or None), pad_token_id=tok.pad_token_id, use_cache=True,
                )
            call_id = obs.last_call_id
            if call_id is None or call_id == previous_call_id:
                raise RuntimeError("observer did not record this model.generate call")
            answer_ids = generated[0, prompt_len:]
            answer = tok.decode(answer_ids, skip_special_tokens=True).strip()
            raw_ids = answer_ids.tolist()
            stop_reason = "max_new_tokens"
            eos_set = set(eos_ids if isinstance(eos_ids, list) else ([eos_ids] if isinstance(eos_ids, int) else []))
            if raw_ids and raw_ids[-1] in eos_set:
                stop_reason = "eos"
            hook_counts = {
                f"{layer}_{unit}": dict(handle.counts)
                for ((layer, unit), _value), handle in zip(additive.items(), handles)
                if hasattr(handle, "counts")
            } or None
            return {
                "phase": phase, "study": spec["study"], "arm": spec["arm"],
                "condition": spec["arm"], "question_index": task["question_index"],
                "question": task["question"], "question_kind": task["question_kind"],
                "sample": task["sample"], "seed": seed, "answer": answer,
                "new_tokens": len(raw_ids), "stop_reason": stop_reason,
                "set_units": [{"layer": int(l), "unit": int(u)} for l, u in spec["units"]],
                "hook_counts": hook_counts, "dose_file_sha256": dose_sha,
                "observed": {"call_id": call_id, "row": 0},
            }
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()

    return generation_guard(dose_path, dose_sha, generate)


def _expected_arm_manifest(phase: str, spec: dict[str, Any], tasks: list[dict[str, Any]],
                           dose_sha: str, device: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": 1, "phase": phase, "study": spec["study"], "arm": spec["arm"],
        "model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device": device,
        "doses_sha256": dose_sha, "source_sha256": hashes,
        "spec": spec, "requested_rows": len(tasks),
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_new_tokens": A_MAX_NEW_TOKENS if spec["study"] == "87A" else B_MAX_NEW_TOKENS},
    }


def _prepare_arm(out: Path, phase: str, spec: dict[str, Any], tasks: list[dict[str, Any]],
                 dose_sha: str, device: str, hashes: dict[str, str]) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    arm_dir = out / phase / spec["arm"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    answer_path = arm_dir / "answers.jsonl"
    manifest_path = arm_dir / "arm-manifest.json"
    expected = _expected_arm_manifest(phase, spec, tasks, dose_sha, device, hashes)
    if answer_path.exists() and not manifest_path.exists():
        raise ValueError(f"answers exist without provenance manifest: {answer_path}")
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"resume provenance mismatch for {spec['arm']} ({phase})")
    else:
        atomic_write_json(manifest_path, expected, refuse_overwrite=True)
    if not answer_path.exists():
        atomic_write_bytes(answer_path, b"", refuse_overwrite=True)
    existing_rows = read_jsonl(answer_path)
    done = set()
    for row in existing_rows:
        key = (int(row["question_index"]), int(row["sample"]))
        if key in done:
            raise ValueError(f"duplicate answer row for {spec['arm']}: {key}")
        done.add(key)
    if any((int(t["question_index"]), int(t["sample"])) not in {
        (int(x["question_index"]), int(x["sample"])) for x in tasks
    } for t in existing_rows):
        raise ValueError(f"unexpected answer row found in {answer_path}")
    todo = [task for task in tasks if (task["question_index"], task["sample"]) not in done]
    return answer_path, todo, expected


def _groups_for_atomic_checkpoints(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault((int(task["question_index"]), str(task["question_kind"])), []).append(task)
    return list(grouped.values())


def require_completed_pilot(out: Path, dose_sha: str, device: str,
                           hashes: dict[str, str], specs: list[dict[str, Any]]) -> None:
    pilot_path = out / "pilot" / "phase.json"
    if not pilot_path.is_file():
        raise FileNotFoundError("complete the 44-answer pilot in this output directory before run")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if (pilot.get("phase") != "pilot" or pilot.get("model_id") != MODEL_ID
            or pilot.get("revision") != REVISION or pilot.get("dtype") != "bfloat16"
            or pilot.get("device") != device or pilot.get("doses_sha256") != dose_sha
            or pilot.get("source_sha256") != hashes):
        raise ValueError("pilot provenance does not match the requested run")
    expected_arms = {
        spec["arm"]: (4 if spec["study"] == "87A" else 6)
        for spec in specs if spec["study"] == "87A" or spec["arm"] in B_MAIN_ARMS
    }
    answers = pilot.get("answers", {})
    if set(answers) != set(expected_arms):
        raise ValueError("pilot is incomplete: expected all five 87A arms and U/Z/M/F")
    for arm, expected_rows in expected_arms.items():
        if answers[arm].get("rows") != expected_rows:
            raise ValueError(f"pilot arm {arm} has {answers[arm].get('rows')} rows; expected {expected_rows}")


def run_generation_phase(args) -> None:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dose_path = Path(args.doses) if args.doses else out / EXPECTED_DOSE_NAME
    dose_doc, dose_sha = load_doses_for_run(dose_path)
    hashes = imported_source_hashes()
    specs = _arm_specs(dose_doc)
    if args.phase == "run":
        require_completed_pilot(out, dose_sha, args.device, hashes, specs)
    run_specs = specs
    if args.phase == "run" and args.test_only_random_seed is not None:
        if args.test_only_random_seed not in RANDOM_SEEDS:
            raise ValueError(f"test-only random seed must be one of {RANDOM_SEEDS}")
        run_specs = [spec for spec in specs if spec.get("random_seed") == args.test_only_random_seed]
    arm_tasks = {
        spec["arm"]: tasks_for_arm(args.phase, spec)
        for spec in run_specs
        if (spec["study"] == "87A" or spec["arm"] in B_MAIN_ARMS
            or (args.phase == "run" and (spec["arm"].startswith("RM_") or spec["arm"].startswith("RF_"))))
    }
    # Register and validate every arm before loading the model or allowing any generation.
    arm_state = {}
    by_arm = {spec["arm"]: spec for spec in specs}
    for arm, tasks in arm_tasks.items():
        arm_state[arm] = _prepare_arm(out, args.phase, by_arm[arm], tasks, dose_sha, args.device, hashes)
    tok, model = nr.load_model(MODEL_ID, REVISION, args.device)
    model.requires_grad_(False)
    model.eval()
    _assert_bfloat16(model)
    _, layers, _ = nr.find_text_layers(model)
    if len(layers) <= max((*LAYERS, ALIEN_LAYER)):
        raise ValueError("loaded model does not contain all registered steering layers")
    obs_path = out / "observe" / (
        f"{args.phase}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}.jsonl"
    )
    obs = observe.install(model, tok, obs_path, label=f"Part 87 {args.phase}")
    start = time.time()
    completed_this_call = 0
    try:
        for spec in run_specs:
            arm = spec["arm"]
            if arm not in arm_state:
                continue
            answer_path, todo, manifest = arm_state[arm]
            if not todo:
                print(f"{args.phase} {arm}: complete ({manifest['requested_rows']} rows)")
                continue
            rows_so_far = read_jsonl(answer_path)
            for group in _groups_for_atomic_checkpoints(todo):
                # A group's samples share the same question; the JSONL replacement is an atomic checkpoint.
                new_rows = []
                for task in group:
                    row = _generate_one(
                        tok, model, layers, args.device, args.phase, spec, task, dose_doc, dose_sha,
                        dose_path=dose_path, obs=obs,
                    )
                    new_rows.append(row)
                atomic_write_jsonl(answer_path, new_rows)
                rows_so_far.extend(new_rows)
                completed_this_call += len(new_rows)
                if args.device in ("cuda", "xpu"):
                    nr.spill_guard(args.device, args.max_gpu_gb)
                print(f"{args.phase} {arm}: {len(rows_so_far)}/{manifest['requested_rows']} rows", flush=True)
        atomic_write_json(out / args.phase / "phase.json", {
            "phase": args.phase, "model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16",
            "device": args.device, "doses_path": str(dose_path.resolve()), "doses_sha256": dose_sha,
            "source_sha256": hashes, "new_generations_this_invocation": completed_this_call,
            "wall_time_seconds": time.time() - start,
            "answers": {
                arm: {"path": str((out / args.phase / arm / "answers.jsonl").resolve()),
                      "rows": len(read_jsonl(out / args.phase / arm / "answers.jsonl"))}
                for arm in arm_state
            },
        })
        print(f"{args.phase}: generated {completed_this_call} rows; results at {out / args.phase}")
    finally:
        obs.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="phase", required=True)
    measure_parser = subs.add_parser("measure", help="teacher-force unchanged Part 66 answers and write locked doses")
    measure_parser.add_argument("--out", required=True)
    measure_parser.add_argument("--answers", default=str(ROOT / "results" / "part66" / "full" / "answers.jsonl"))
    measure_parser.add_argument("--calib", required=True, help="neuronary calibration text")
    measure_parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    for phase in ("pilot", "run"):
        run_parser = subs.add_parser(phase)
        run_parser.add_argument("--out", required=True)
        run_parser.add_argument("--doses", default=None, help="locked doses.json; defaults to OUT/doses.json")
        run_parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
        run_parser.add_argument("--max-gpu-gb", type=float, default=14.5)
        if phase == "run":
            run_parser.add_argument("--test-only-random-seed", type=int, default=None,
                                    help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.phase == "measure":
        measure(args)
    else:
        run_generation_phase(args)


if __name__ == "__main__":
    main()

