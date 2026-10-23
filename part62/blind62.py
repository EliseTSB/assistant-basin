#!/usr/bin/env python3
"""Blindly label Part 62 answers with the Part 54 origin rubric."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import blind_origin as origin
from part59 import blind_setting as setting

try:
    from . import run62
except ImportError:
    import run62


# Keep the Part 59A calibration refusal boundary.
CALIBRATION_ERROR_LIMIT = setting.CALIBRATION_ERROR_LIMIT
# The reader batch must reserve at least 1,024 output tokens for Opus 5.5 thinking.
MAX_TOKENS = 1024
# Only normal final-answer stops count as verified completion; max_tokens means truncation.
ACCEPTED_STOP_REASONS = ("end_turn", "stop_sequence")
# This fixed shuffle seed makes the hidden answer order reproducible.
SHUFFLE_SEED = 62062
# The existing Part 54 origin reader uses the Opus 5.5 batch model.
MODEL = origin.MODEL
RUBRIC = origin.RUBRIC
CAL = origin.CAL


def _cell_key(row: dict) -> tuple:
    return (row["model"], float(row["dose"]), row["phase"], row["question"],
            row["kind"], int(row["unit"]))


def answers(folder: str | Path) -> list[dict]:
    """Read only a complete real Part 62 writer manifest and its listed cell files."""
    root = Path(folder).resolve()
    manifest_path = root / "part62-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing Part 62 run manifest in {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Part 62 run manifest: {exc}") from exc
    if manifest.get("schema") != "part62-run-v1" or manifest.get("status") != "complete":
        raise ValueError("Part 62 manifest must have schema part62-run-v1 and status complete")
    if (manifest.get("doses") != list(run62.DOSES) or manifest.get("phases") != list(run62.PHASES)
            or manifest.get("questions") != list(run62.QUESTIONS)
            or manifest.get("samples_per_cell") != run62.SAMPLES
            or manifest.get("control_count") != run62.CONTROL_COUNT
            or manifest.get("control_seed") != run62.CONTROL_SEED
            or manifest.get("expected_cell_count") != run62.expected_cell_count()
            or manifest.get("expected_generations") != run62.expected_generation_count()
            or manifest.get("generated_answers") != run62.expected_generation_count()):
        raise ValueError("Part 62 manifest does not match the registered dose/phase/sample matrix")
    control_units = manifest.get("control_units")
    if not isinstance(control_units, dict) or set(control_units) != set(run62.MODELS):
        raise ValueError("Part 62 manifest must record one control unit for each model")
    registered_controls = {name: int(spec["control_unit"]) for name, spec in run62.MODELS.items()}
    if {name: int(unit) for name, unit in control_units.items()} != registered_controls:
        raise ValueError("Part 62 manifest control units do not match the pre-registered controls")
    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(run62.MODELS):
        raise ValueError("Part 62 manifest must record both pinned model runs")
    for model_name, spec in run62.MODELS.items():
        meta = models[model_name]
        if (meta.get("id"), meta.get("revision"), meta.get("layer"), meta.get("target_unit"),
                meta.get("control_unit"), meta.get("width")) != (
                spec["id"], spec["revision"], spec["layer"], spec["unit"],
                registered_controls[model_name], spec["expected_width"]):
            raise ValueError(f"Part 62 manifest model metadata is not registered for {model_name}")
        for field in ("pile_maximum", "alien_frame_maximum", "target_column_norm", "control_column_norm"):
            value = meta.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Part 62 manifest has invalid {field} for {model_name}")
        if (meta.get("alien_frame_count") != run62.ALIEN_FRAME_COUNT
                or meta.get("registered_alien_frame_reference") != spec.get("alien_frame_reference")):
            raise ValueError(f"Part 62 manifest has the wrong alien-frame scale metadata for {model_name}")
    base_seed = manifest.get("base_seed")
    if not isinstance(base_seed, int):
        raise ValueError("Part 62 manifest is missing its root seed")
    planned = manifest.get("planned_cells")
    completed = manifest.get("completed")
    if not isinstance(planned, list) or not isinstance(completed, list):
        raise ValueError("Part 62 manifest must list planned and completed cells")
    expected_specs = [cell for model_name in run62.MODELS
                      for cell in run62.model_cell_specs(model_name, int(control_units[model_name]))]
    expected_by_key = {_cell_key(cell): {**cell, "file": run62.cell_filename(cell)} for cell in expected_specs}
    if len(expected_by_key) != run62.expected_cell_count():
        raise ValueError("registered Part 62 matrix contains duplicate cells")
    if len(planned) != len(expected_by_key):
        raise ValueError(f"Part 62 manifest lists {len(planned)} planned cells; expected {len(expected_by_key)}")
    planned_by_key = {_cell_key(cell): cell for cell in planned if isinstance(cell, dict)}
    if len(planned_by_key) != len(planned) or set(planned_by_key) != set(expected_by_key):
        raise ValueError("Part 62 manifest has duplicated, malformed, or missing planned cells")
    for key, expected in expected_by_key.items():
        if planned_by_key[key].get("file") != expected["file"]:
            raise ValueError(f"Part 62 manifest has the wrong filename for cell {key}")
    expected_names = {cell["file"] for cell in expected_by_key.values()}
    if len(completed) != len(expected_names) or set(completed) != expected_names:
        raise ValueError("Part 62 manifest does not list every planned file as complete")
    actual_names = {path.name for path in root.glob("clamp-*.json")}
    if actual_names != expected_names:
        extras = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise ValueError(f"Part 62 cell files do not match manifest; extras={extras[:3]}, missing={missing[:3]}")

    rows = []
    seen_seeds = set()
    for key, expected in expected_by_key.items():
        path = root / expected["file"]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {path.name}: {exc}") from exc
        if data.get("schema") != "part62-cell-v1" or data.get("part") != 62:
            raise ValueError(f"{path.name} is not a Part 62 cell written by run62.py")
        if (data.get("short_name") != expected["model"]
                or data.get("model") != run62.MODELS[expected["model"]]["id"]
                or data.get("revision") != run62.MODELS[expected["model"]]["revision"]
                or data.get("dtype") != "torch.float32"
                or data.get("question") != expected["question"]
                or float(data.get("clamp_mult", -999)) != float(expected["dose"])
                or data.get("phase") != expected["phase"]
                or data.get("kind") != expected["kind"]):
            raise ValueError(f"{path.name} content does not match its manifest cell")
        conditions = data.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 1:
            raise ValueError(f"{path.name} must contain exactly one generated condition")
        condition = conditions[0]
        model_meta = models[expected["model"]]
        if data.get("best_twin") != {"layer": run62.MODELS[expected["model"]]["layer"],
                                      "unit": run62.MODELS[expected["model"]]["unit"]}:
            raise ValueError(f"{path.name} has the wrong registered target unit")
        scale = data.get("dose_scale", {})
        expected_alien_dose = (float(expected["dose"]) * float(model_meta["pile_maximum"])
                               / float(model_meta["alien_frame_maximum"]))
        if (not math.isclose(float(scale.get("pile_maximum", -1)),
                             float(model_meta["pile_maximum"]), rel_tol=1e-6)
                or not math.isclose(float(scale.get("alien_frame_maximum", -1)),
                                    float(model_meta["alien_frame_maximum"]), rel_tol=1e-6)
                or not math.isclose(float(scale.get("dose_in_alien_frame_units", -1)),
                                    expected_alien_dose, rel_tol=1e-6)
                or scale.get("registered_alien_frame_reference") !=
                   run62.MODELS[expected["model"]].get("alien_frame_reference")):
            raise ValueError(f"{path.name} dose-scale metadata does not match the measured frame maximum")
        expected_name = "twin" if expected["kind"] == "twin" else f"control-{expected['unit']}"
        if (condition.get("condition") != expected_name
                or condition.get("kind") != expected["kind"]
                or int(condition.get("unit", -1)) != int(expected["unit"])
                or condition.get("phase") != expected["phase"]):
            raise ValueError(f"{path.name} condition does not match its manifest cell")
        samples = condition.get("answers")
        if (not isinstance(samples, list) or len(samples) != run62.SAMPLES
                or condition.get("n") != run62.SAMPLES
                or not all(isinstance(text, str) for text in samples)):
            raise ValueError(f"{path.name} must contain {run62.SAMPLES} text answers")
        seed = condition.get("seed")
        if not isinstance(seed, int) or seed in seen_seeds:
            raise ValueError(f"{path.name} has a missing or repeated independent condition seed")
        expected_seed = run62.condition_seed(base_seed, expected["model"], expected["dose"],
                                             expected["phase"], expected["question"],
                                             expected["kind"], expected["unit"])
        if seed != expected_seed:
            raise ValueError(f"{path.name} seed does not match its independent condition key")
        target_maximum = float(model_meta["pile_maximum"])
        expected_value = float(expected["dose"]) * target_maximum
        if expected["kind"] == "control":
            expected_value *= (float(model_meta["target_column_norm"] /
                                     model_meta["control_column_norm"]))
        value = condition.get("set_value")
        if not isinstance(value, (int, float)) or not math.isclose(value, expected_value, rel_tol=1e-6):
            raise ValueError(f"{path.name} activation value does not match the registered write size")
        seen_seeds.add(seed)
        for index, text in enumerate(samples):
            rows.append({"model": expected["model"], "question": expected["question"],
                         "dose": float(expected["dose"]), "phase": expected["phase"],
                         "kind": expected["kind"], "cond": expected_name,
                         "unit": int(expected["unit"]), "seed": seed, "i": index, "text": text})
    if len(rows) != run62.expected_generation_count():
        raise ValueError(f"Part 62 writer produced {len(rows)} answers; expected {run62.expected_generation_count()}")
    return rows


def _calibration_wrong(labels: dict) -> list[str]:
    wrong = []
    for index, (expected_origin, expected_quality, _text) in enumerate(CAL):
        key = f"cal|{index}|{expected_origin}|{expected_quality}"
        if labels.get(key) != [expected_origin, expected_quality]:
            wrong.append(key)
    return wrong


def scoring_gate(labels: dict, answer_ids: list[str], completion_issues: dict | None = None) -> dict:
    """Refuse scoring after calibration failure, truncation, or any missing answer label."""
    wrong = _calibration_wrong(labels)
    unlabelled = sorted(answer_id for answer_id in answer_ids if labels.get(answer_id) is None)
    if len(wrong) > CALIBRATION_ERROR_LIMIT:
        raise ValueError(f"calibration gate failed: {len(wrong)} errors exceeds {CALIBRATION_ERROR_LIMIT}")
    if completion_issues:
        raise ValueError(f"truncated or unverified batch completion: {len(completion_issues)} response(s)")
    if unlabelled:
        raise ValueError(f"unlabelled answer present: refusing to score ({len(unlabelled)} unlabelled)")
    return {"calibration_wrong": wrong, "unlabelled_answers": unlabelled,
            "completion_issues": {}}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_batch(rows: list[dict]) -> tuple[dict, dict, list[dict]]:
    """Build hidden-condition API requests and their local-only key without sending them."""
    if MAX_TOKENS < 1024:
        raise ValueError("Part 62 reader max_tokens must be at least 1,024")
    items = [(f"ans|{number}", row["text"]) for number, row in enumerate(rows)]
    items.extend((f"cal|{index}|{label}|{quality}", text)
                 for index, (label, quality, text) in enumerate(CAL))
    key = {f"ans|{number}": {name: value for name, value in row.items() if name != "text"}
           for number, row in enumerate(rows)}
    idmap, requests = {}, []
    for internal_id, text in items:
        custom_id = f"r{len(idmap):05d}"
        idmap[custom_id] = internal_id
        # Model, condition, dose, phase, and question stay in the local key; only answer text is sent.
        requests.append({"custom_id": custom_id, "params": {
            "model": MODEL, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": RUBRIC + text}],
        }})
    return idmap, key, requests


def submit(folder: str | Path, review_dir: str | Path) -> str:
    rows = answers(folder)
    random.Random(SHUFFLE_SEED).shuffle(rows)
    idmap, key, requests = prepare_batch(rows)
    result = json.loads(setting.call("POST", "/v1/messages/batches", {"requests": requests}))
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    batch_id = result["id"]
    _save_json(review_dir / f"part62-idmap-{batch_id}.json", idmap)
    _save_json(review_dir / f"part62-key-{batch_id}.json", key)
    _save_json(review_dir / f"part62-batch-{batch_id}.json", {
        "batch_id": batch_id, "processing_status": result.get("processing_status"),
        "request_count": len(requests), "answer_count": len(rows),
        "calibration_count": len(CAL), "model": MODEL, "max_tokens": MAX_TOKENS,
    })
    print(batch_id, result.get("processing_status"), len(requests))
    return batch_id


def status(batch_id: str) -> None:
    result = json.loads(setting.call("GET", f"/v1/messages/batches/{batch_id}"))
    print(result.get("processing_status"), result.get("request_counts"))


def parse_batch_results(lines, idmap: dict) -> tuple[dict, dict]:
    """Parse Messages Batch JSONL result records and preserve their actual stop reasons."""
    labels, completion_reasons, seen_custom_ids = {}, {}, set()
    for line in lines:
        result = json.loads(line)
        custom_id = result.get("custom_id")
        if custom_id not in idmap:
            raise ValueError(f"batch returned unknown custom_id {custom_id!r}")
        if custom_id in seen_custom_ids:
            raise ValueError(f"batch returned duplicate custom_id {custom_id!r}")
        seen_custom_ids.add(custom_id)
        internal_id = idmap[custom_id]
        response = result.get("result", {})
        if response.get("type") != "succeeded":
            labels[internal_id] = None
            completion_reasons[internal_id] = response.get("type", "request_failed")
            continue
        message = response.get("message", {})
        completion_reasons[internal_id] = message.get("stop_reason")
        content = " ".join(block.get("text", "") for block in message.get("content", [])
                           if block.get("type") == "text")
        labels[internal_id] = origin.parse_labels(content)
    for internal_id in idmap.values():
        labels.setdefault(internal_id, None)
        completion_reasons.setdefault(internal_id, "missing_result")
    return labels, completion_reasons


def collect(batch_id: str, review_dir: str | Path) -> dict:
    review_dir = Path(review_dir)
    idmap = json.loads((review_dir / f"part62-idmap-{batch_id}.json").read_text(encoding="utf-8"))
    key = json.loads((review_dir / f"part62-key-{batch_id}.json").read_text(encoding="utf-8"))
    lines = setting.call("GET", f"/v1/messages/batches/{batch_id}/results").splitlines()
    labels, completion_reasons = parse_batch_results(lines, idmap)
    wrong = _calibration_wrong(labels)
    answer_unlabelled = sorted(internal_id for internal_id in key if labels.get(internal_id) is None)
    completion_issues = {internal_id: reason for internal_id, reason in completion_reasons.items()
                         if reason not in ACCEPTED_STOP_REASONS}
    calibration_ids = {f"cal|{index}|{label}|{quality}"
                       for index, (label, quality, _text) in enumerate(CAL)}
    calibration_completion_issues = {internal_id: reason for internal_id, reason in completion_issues.items()
                                     if internal_id in calibration_ids}
    calibration_passed = (len(wrong) <= CALIBRATION_ERROR_LIMIT
                          and not calibration_completion_issues)
    record = {
        "schema": "part62-blind-labels-v1", "batch_id": batch_id, "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "calibration": {"count": len(CAL), "error_limit": CALIBRATION_ERROR_LIMIT,
                        "wrong": len(wrong), "wrong_ids": wrong,
                        "completion_issues": calibration_completion_issues,
                        "passed": calibration_passed},
        "completion": {"accepted_stop_reasons": list(ACCEPTED_STOP_REASONS),
                       "issues": completion_issues,
                       "passed": not completion_issues},
        "answer_count": len(key), "unlabelled_answers": answer_unlabelled,
        "scoring_allowed": calibration_passed and not answer_unlabelled and not completion_issues,
        "labels": labels,
    }
    _save_json(review_dir / f"part62-labels-{batch_id}.json", record)
    print(f"calibration wrong: {len(wrong)} of {len(CAL)}; "
          f"unlabelled answers: {len(answer_unlabelled)} of {len(key)}")
    try:
        scoring_gate(labels, list(key), completion_issues)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _print_tally(key, labels)
    return record


def _print_tally(key: dict, labels: dict) -> None:
    counts = {}
    for answer_id, meta in key.items():
        group = (meta["model"], meta["phase"], meta["kind"], float(meta["dose"]))
        row = counts.setdefault(group, {"n": 0, "EARTH": 0, "OFFWORLD": 0,
                                        "NEITHER": 0, "BROKEN": 0})
        row["n"] += 1
        label = labels[answer_id]
        row[label[0]] += 1
        row["BROKEN"] += label[1] == "BROKEN"
    print("model | phase | kind | dose | n | EARTH OFFWORLD NEITHER | broken")
    for group in sorted(counts):
        model, phase, kind, dose = group
        row = counts[group]
        print(f"{model} | {phase} | {kind} | {dose:g} | {row['n']} | "
              f"{row['EARTH']} {row['OFFWORLD']} {row['NEITHER']} | {row['BROKEN']}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--folder", default=str(PROJECT_ROOT / "results" / "part62"))
    submit_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    status_parser = sub.add_parser("status")
    status_parser.add_argument("batch_id")
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("batch_id")
    collect_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "submit":
        submit(args.folder, args.review_dir)
    elif args.command == "status":
        status(args.batch_id)
    else:
        collect(args.batch_id, args.review_dir)


if __name__ == "__main__":
    main()
