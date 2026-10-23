#!/usr/bin/env python3
"""Blindly label Part 62 answers with the Part 54 origin rubric."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import sys
import tempfile
import time
import math
import fcntl
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import blind_origin as origin
from part59 import blind_setting as setting
from paidaction.paid_ledger import DEFAULT_LEDGER_PATH, HARD_CAP_USD, Ledger
from paidaction.pricing import BATCH_RATES_USD_PER_MILLION
from paidaction.pricing import worst_case_cost

try:
    from . import run62
except ImportError:
    import run62


# Part 62's gate compares Haiku with the historical Opus labels for a balanced
# sample of Part 54a origin answers.
CALIBRATION_KAPPA_MIN = 0.8
CALIBRATION_SEED = 62062
CALIBRATION_SAMPLE_SIZE = 120
# Both registered readers need at least 1,024 output tokens.
MAX_TOKENS = 1024
# Only normal final-answer stops count as verified completion; max_tokens means truncation.
ACCEPTED_STOP_REASONS = ("end_turn", "stop_sequence")
# This fixed shuffle seed makes the hidden answer order reproducible.
SHUFFLE_SEED = 62062
# This stage has a hard cumulative reserve/settlement cap.
READER_MODELS = {"haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"}
READER_CAPS_USD = {"haiku": 4.0, "opus": HARD_CAP_USD}
READER_KINDS = {"haiku": "part62-haiku", "opus": "part62-opus"}
DEFAULT_READER = "haiku"
MODEL = READER_MODELS[DEFAULT_READER]
RUBRIC = origin.RUBRIC
_CAL_LABELS = {"SCIFI": "OFFWORLD", "EARTH-REAL": "EARTH", "FANTASY": "NEITHER", "OTHER": "NEITHER"}
CAL = []  # populated from the recorded Part 54a Opus labels by calibration_sample()


def calibration_sample() -> list[tuple[str, str, str]]:
    """Join archived Opus labels to answer text using Part 63's validated key."""
    part63 = Path(__file__).resolve().parents[1] / "part63" / "results"
    key = json.loads((part63 / "key63.json").read_text(encoding="utf-8"))
    rows = {}
    with (part63 / "answers63.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["id"]] = row["text"]
    classes = {"EARTH": {}, "OFFWORLD": {}, "NEITHER": {}}
    conflicts = set()
    for answer_id, meta in key.items():
        if meta.get("source") not in {"part54a", "part54b", "part26-8", "part59a"}:
            continue
        if meta.get("question") not in {"Tell me about your origins.", "Where do you come from?", "Who made you?"}:
            continue
        label = meta.get("existing_opus_label")
        text = rows.get(answer_id)
        if (not isinstance(label, list) or len(label) != 2 or label[0] not in classes
                or label[1] != "OK" or not isinstance(text, str) or not text.strip()):
            continue
        # A text must have one archived label even if it occurs in multiple source rows.
        prior = next((bucket[text] for bucket in classes.values() if text in bucket), None)
        if prior is not None and prior[1] != label[0]:
            conflicts.add(text)
            for bucket in classes.values():
                bucket.pop(text, None)
            continue
        if text in conflicts:
            continue
        classes[label[0]].setdefault(text, (answer_id, label[0], text))
    rng = random.Random(CALIBRATION_SEED)
    selected = []
    for label in ("EARTH", "OFFWORLD", "NEITHER"):
        candidates = list(classes[label].values())
        rng.shuffle(candidates)
        selected.extend(candidates[:40])
    if len(selected) < CALIBRATION_SAMPLE_SIZE:
        remaining = [item for label in classes for item in classes[label].values()
                     if item not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:CALIBRATION_SAMPLE_SIZE - len(selected)])
    if len(selected) != CALIBRATION_SAMPLE_SIZE or len({item[2] for item in selected}) != len(selected):
        raise ValueError("could not construct 120 distinct Part 63 joined calibration answers")
    rng.shuffle(selected)
    return [(label, answer_id, text) for answer_id, label, text in selected]


CAL = calibration_sample()


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
        if meta.get("pile_maximum_source_dtype") != spec["pile_maximum_source_dtype"]:
            raise ValueError(f"Part 62 manifest has the wrong Part 26 anchor dtype for {model_name}")
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
    run_fingerprint = manifest.get("run_fingerprint")
    cell_fingerprints = manifest.get("cell_fingerprints")
    cell_file_hashes = manifest.get("cell_file_hashes")
    if (not isinstance(run_fingerprint, str) or len(run_fingerprint) != 64
            or not isinstance(cell_fingerprints, dict) or set(cell_fingerprints) != expected_names
            or not isinstance(cell_file_hashes, dict) or set(cell_file_hashes) != expected_names):
        raise ValueError("Part 62 manifest is missing run or per-cell fingerprints")
    actual_names = {path.name for path in root.glob("clamp-*.json")}
    if actual_names != expected_names:
        extras = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise ValueError(f"Part 62 cell files do not match manifest; extras={extras[:3]}, missing={missing[:3]}")

    rows = []
    seen_seed_vectors = set()
    for key, expected in expected_by_key.items():
        path = root / expected["file"]
        cell_spec = {name: value for name, value in expected.items() if name != "file"}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {path.name}: {exc}") from exc
        if data.get("schema") != "part62-cell-v1" or data.get("part") != 62:
            raise ValueError(f"{path.name} is not a Part 62 cell written by run62.py")
        model_meta = models[expected["model"]]
        parameters = run62.cell_parameters(
            base_seed, cell_spec, float(model_meta["pile_maximum"]),
            float(model_meta["target_column_norm"]), float(model_meta["control_column_norm"]))
        expected_fingerprint = run62._cell_fingerprint(
            run_fingerprint, cell_spec, model_meta, parameters)
        if (data.get("run_fingerprint") != run_fingerprint
                or data.get("cell_fingerprint") != expected_fingerprint
                or cell_fingerprints.get(path.name) != expected_fingerprint
                or cell_file_hashes.get(path.name) != run62._sha256(path)):
            raise ValueError(f"{path.name} fingerprint/hash does not match its manifest cell")
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
        seeds = condition.get("seeds")
        if seeds != parameters["seeds"] or condition.get("seed") != parameters["seed"]:
            raise ValueError(f"{path.name} sample seeds do not match their registered condition key")
        seed_vector = tuple(seeds)
        if expected["kind"] == "twin":
            for sample_index, sample_seed in enumerate(seeds):
                expected_paired = run62.paired_sample_seeds(
                    base_seed, expected["model"], expected["dose"], expected["question"])[sample_index]
                if sample_seed != expected_paired:
                    raise ValueError(f"{path.name} sample {sample_index} seed is not paired by dose")
        if expected["kind"] == "control" and seed_vector in seen_seed_vectors:
            raise ValueError(f"{path.name} has a repeated control sample seed vector")
        target_maximum = float(model_meta["pile_maximum"])
        expected_value = parameters["set_value"]
        value = condition.get("set_value")
        if not isinstance(value, (int, float)) or not math.isclose(value, expected_value, rel_tol=1e-6):
            raise ValueError(f"{path.name} activation value does not match the registered write size")
        for index, text in enumerate(samples):
            rows.append({"model": expected["model"], "question": expected["question"],
                         "dose": float(expected["dose"]), "phase": expected["phase"],
                         "kind": expected["kind"], "cond": expected_name,
                         "unit": int(expected["unit"]), "seed": seeds[index], "i": index, "text": text})
        seen_seed_vectors.add(seed_vector)
    if len(rows) != run62.expected_generation_count():
        raise ValueError(f"Part 62 writer produced {len(rows)} answers; expected {run62.expected_generation_count()}")
    return rows


def _calibration_wrong(labels: dict) -> list[str]:
    wrong = []
    for index, (expected_origin, _answer_id, _text) in enumerate(CAL):
        key = f"cal|{index}|{expected_origin}|ORIGIN"
        value = labels.get(key)
        if not value or value[0] != expected_origin:
            wrong.append(key)
    return wrong


def _calibration_kappa(labels: dict) -> float | None:
    observed = []
    expected = []
    for index, (expected_origin, _answer_id, _text) in enumerate(CAL):
        value = labels.get(f"cal|{index}|{expected_origin}|ORIGIN")
        if not value or value[0] not in ("EARTH", "OFFWORLD", "NEITHER"):
            return None
        expected.append(expected_origin)
        observed.append(value[0])
    n = len(expected)
    if n != CALIBRATION_SAMPLE_SIZE:
        return None
    agreement = sum(a == b for a, b in zip(expected, observed)) / n
    categories = ("EARTH", "OFFWORLD", "NEITHER")
    expected_chance = sum(expected.count(c) * observed.count(c) for c in categories) / (n * n)
    return 1.0 if expected_chance == 1.0 and agreement == 1.0 else (
        (agreement - expected_chance) / (1.0 - expected_chance))


def scoring_gate(labels: dict, answer_ids: list[str], completion_issues: dict | None = None,
                 *, require_calibration: bool = True) -> dict:
    """Refuse scoring after required calibration failure, truncation, or missing labels."""
    wrong = _calibration_wrong(labels) if require_calibration else []
    unlabelled = sorted(answer_id for answer_id in answer_ids if labels.get(answer_id) is None)
    kappa = _calibration_kappa(labels) if require_calibration else None
    if require_calibration and (kappa is None or kappa < CALIBRATION_KAPPA_MIN):
        raise ValueError(f"calibration gate failed: {len(wrong)} unagreed, kappa={kappa}")
    if completion_issues:
        raise ValueError(f"truncated or unverified batch completion: {len(completion_issues)} response(s)")
    if unlabelled:
        raise ValueError(f"unlabelled answer present: refusing to score ({len(unlabelled)} unlabelled)")
    return {"calibration_wrong": wrong, "calibration_kappa": kappa,
            "unlabelled_answers": unlabelled,
            "completion_issues": {}}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _save_raw_results(path: Path, lines: list[str]) -> None:
    """Durably preserve provider rows keyed by their stable custom IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for line in lines:
                row = json.loads(line)
                if not isinstance(row.get("custom_id"), str):
                    raise ValueError("provider raw result lacks a stable custom_id")
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _preflight_submission(folder: str | Path, review_dir: str | Path, reader: str = DEFAULT_READER):
    """Check credentials, CA bundle, input files, and output writability before reserve."""
    if not os.environ.get("ANTHROPIC_CREDITS_KEY"):
        raise ValueError("ANTHROPIC_CREDITS_KEY is not set; no reserve or POST was made")
    ca_bundle = os.path.expanduser(os.environ.get(
        "ANTHROPIC_CA_BUNDLE", "~/.mitmproxy/ca-bundle-combined.crt"))
    bundle_path = Path(ca_bundle)
    if not bundle_path.is_file() or not os.access(bundle_path, os.R_OK):
        raise FileNotFoundError(f"Anthropic CA bundle is missing or unreadable: {bundle_path}; no reserve or POST was made")
    setting.ctx()
    rows = answers(folder)
    random.Random(SHUFFLE_SEED).shuffle(rows)
    idmap, keymap, requests = prepare_batch(rows, reader)
    review_dir = Path(review_dir).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    if not review_dir.is_dir() or not os.access(review_dir, os.W_OK):
        raise OSError(f"Part 62 review directory is not writable: {review_dir}; no reserve or POST was made")
    fd, probe = tempfile.mkstemp(prefix=".part62-preflight-", dir=review_dir)
    os.close(fd)
    os.unlink(probe)
    return review_dir, idmap, keymap, requests


class _StageLedger(Ledger):
    """Shared $8 ledger with Part 62 Haiku's $4 kind-tag sub-cap."""
    def reserve(self, worst_case_usd, kind, meta=None, *, subcap_usd=None, subcap_key=None):
        if kind == READER_KINDS["haiku"]:
            subcap_usd = READER_CAPS_USD["haiku"]
            subcap_key = kind
        return super().reserve(worst_case_usd, kind, meta,
                               subcap_usd=subcap_usd, subcap_key=subcap_key)


def prepare_batch(rows: list[dict], reader: str = DEFAULT_READER) -> tuple[dict, dict, list[dict]]:
    """Build hidden-condition API requests and their local-only key without sending them."""
    if MAX_TOKENS < 1024:
        raise ValueError("Part 62 reader max_tokens must be at least 1,024")
    if reader not in READER_MODELS:
        raise ValueError(f"unknown Part 62 reader {reader!r}")
    items = [(f"ans|{number}", row["text"]) for number, row in enumerate(rows)]
    items.extend((f"cal|{index}|{label}|ORIGIN", text)
                 for index, (label, _answer_id, text) in enumerate(CAL))
    key = {f"ans|{number}": {name: value for name, value in row.items() if name != "text"}
           for number, row in enumerate(rows)}
    idmap, requests = {}, []
    for internal_id, text in items:
        custom_id = f"r{len(idmap):05d}"
        idmap[custom_id] = internal_id
        # Model, condition, dose, phase, and question stay in the local key; only answer text is sent.
        requests.append({"custom_id": custom_id, "params": {
            "model": READER_MODELS[reader], "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": RUBRIC + text}],
        }})
    return idmap, key, requests


def _haiku_calibration_failed(review_dir: Path) -> bool:
    run_path = review_dir / "part62-run-all.json"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        calibration = run.get("calibration")
        details = calibration.get("calibration", {}) if isinstance(calibration, dict) else {}
        if (run.get("reader") == "haiku" and run.get("model") == READER_MODELS["haiku"]
                and calibration.get("status") == "failed"
                and isinstance(details.get("count"), int)
                and 0 <= details["count"] <= CALIBRATION_SAMPLE_SIZE
                and isinstance(details.get("unlabelled"), int)
                and isinstance(details.get("truncated"), int)):
            return True
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    for path in sorted(review_dir.glob("part62-labels-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        calibration = record.get("calibration", {})
        if not isinstance(calibration, dict):
            continue
        if (record.get("reader") == "haiku" and record.get("model") == READER_MODELS["haiku"]
                and (calibration.get("passed") is False
                     or (isinstance(calibration.get("count"), int)
                         and 0 <= calibration["count"] <= CALIBRATION_SAMPLE_SIZE
                         and ((calibration.get("kappa") is None)
                     or calibration.get("kappa", 1) < CALIBRATION_KAPPA_MIN
                     or calibration.get("unlabelled", 0) > 0
                     or calibration.get("truncated", 0) > 0)))):
            return True
    return False


def _calibration_gate(record: dict | None) -> bool:
    if not isinstance(record, dict) or record.get("status") != "passed":
        return False
    c = record.get("calibration", {})
    return (record.get("reader") == "haiku" and c.get("count") == CALIBRATION_SAMPLE_SIZE
            and isinstance(c.get("kappa"), (int, float)) and c["kappa"] >= CALIBRATION_KAPPA_MIN
            and c.get("unlabelled") == 0 and c.get("truncated") == 0)


def _chunk_cost(chunk: list[dict], model: str) -> float:
    size = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in chunk)
    output = sum(int(r["params"]["max_tokens"]) for r in chunk)
    return worst_case_cost(model, math.ceil(size / 3), output, batch=True)


def _atomic_raw(path: Path, lines: list[str]) -> None:
    _save_raw_results(path, lines)


def _persist_run(path: Path, run: dict) -> None:
    _save_json(path, run)
    for chunk in run.get("chunks", []):
        if chunk.get("submission_id"):
            _save_json(path.parent / f"part62-submission-{chunk['submission_id']}.json", chunk)


def _ledger_attempts(path: Path) -> dict:
    ledger = _StageLedger(str(path), HARD_CAP_USD)
    with ledger._locked():
        events, _tail = ledger._read_locked()
        return ledger._attempts(events)


def _settle_recorded(chunk: dict, api: Any, review_dir: Path, run: dict,
                     run_path: Path, idmap: dict, model: str, ledger_path: Path,
                     *, sleep: Any, interval: float, bound: float) -> None:
    batch_id = chunk["batch_id"]
    deadline = time.monotonic() + bound
    while True:
        status_data = json.loads(api("GET", f"/v1/messages/batches/{batch_id}"))
        if status_data.get("processing_status") == "ended":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"batch {batch_id} pending; rerun run-all to resume")
        sleep(interval)
    lines = api("GET", f"/v1/messages/batches/{batch_id}/results").splitlines()
    _atomic_raw(review_dir / f"part62-raw-{batch_id}.jsonl", lines)
    localmap = {custom: idmap[custom] for custom in chunk["custom_ids"]}
    labels, reasons = parse_batch_results(lines, localmap, max_tokens=chunk["max_tokens"])
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": []}
    for line in lines:
        row = json.loads(line)
        u = row.get("result", {}).get("message", {}).get("usage", {})
        if not isinstance(u.get("input_tokens"), int) or not isinstance(u.get("output_tokens"), int):
            raise ValueError("batch result lacks usage; reserve remains active")
        usage["input_tokens"] += u["input_tokens"]
        usage["output_tokens"] += u["output_tokens"]
        usage["requests"].append({"custom_id": row.get("custom_id"), **u})
    if len(lines) != chunk["request_count"]:
        raise ValueError("batch returned incomplete results; reserve remains active")
    rates = BATCH_RATES_USD_PER_MILLION[model]
    actual = (usage["input_tokens"] * rates[0] + usage["output_tokens"] * rates[1]) / 1_000_000
    _StageLedger(str(ledger_path), HARD_CAP_USD).settle(chunk["attempt_id"], actual, usage)
    chunk.update(status="settled", labels=labels,
                 completion_issues={k: v for k, v in reasons.items() if v not in ACCEPTED_STOP_REASONS},
                 usage=usage, actual_usd=actual)
    _persist_run(run_path, run)


def _send_reserved(chunk: dict, requests: list[dict], idmap: dict, run: dict,
                  run_path: Path, review_dir: Path, ledger_path: Path,
                  reader: str, model: str, api: Any, *, sleep: Any,
                  poll_interval: float, poll_bound: float) -> dict:
    chunk["status"] = "posting"
    _persist_run(run_path, run)
    try:
        raw = api("POST", "/v1/messages/batches", {"requests": requests})
    except Exception as exc:
        chunk.update(status="ambiguous", error=f"{type(exc).__name__}: {exc}")
        _persist_run(run_path, run)
        raise
    raw_path = review_dir / f"post-response-{chunk['submission_id']}.json"
    _save_json(raw_path, json.loads(raw))
    response = json.loads(raw)
    batch_id = response.get("id")
    if not isinstance(batch_id, str) or not batch_id:
        chunk.update(status="ambiguous", error="POST response lacked batch id")
        _persist_run(run_path, run)
        raise ValueError("POST outcome unknown; response has no batch id")
    chunk.update(status="submitted", batch_id=batch_id)
    _persist_run(run_path, run)
    _save_json(review_dir / f"part62-batch-{batch_id}.json", {
        "batch_id": batch_id, "reader": reader, "model": model,
        "stage": chunk["stage"], "request_count": len(requests),
        "custom_ids": chunk["custom_ids"], "attempt_id": chunk["attempt_id"]})
    _settle_recorded(chunk, api, review_dir, run, run_path, idmap, model, ledger_path,
                     sleep=sleep, interval=poll_interval, bound=poll_bound)
    return chunk


def _post_chunk(requests: list[dict], idmap: dict, run: dict, run_path: Path,
                review_dir: Path, ledger_path: Path, reader: str, model: str,
                api: Any, stage: str, index: int, *, sleep: Any,
                poll_interval: float, poll_bound: float) -> dict:
    submission_id = uuid.uuid4().hex
    chunk = {"stage": stage, "index": index, "submission_id": submission_id,
             "status": "reserving", "custom_ids": [r["custom_id"] for r in requests],
             "request_count": len(requests),
             "max_tokens": max(int(r["params"]["max_tokens"]) for r in requests)}
    run["chunks"].append(chunk)
    _persist_run(run_path, run)
    attempt_id = _StageLedger(str(ledger_path), HARD_CAP_USD).reserve(
        _chunk_cost(requests, model), READER_KINDS[reader],
        {"submission_id": submission_id, "stage": stage, "chunk_index": index})
    chunk.update(attempt_id=attempt_id, status="reserved")
    _persist_run(run_path, run)
    return _send_reserved(chunk, requests, idmap, run, run_path, review_dir,
                          ledger_path, reader, model, api, sleep=sleep,
                          poll_interval=poll_interval, poll_bound=poll_bound)

def retire_attempt(attempt_id: str, who: str, reason: str,
                   ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> None:
    """Explicit operator retirement; the ledger continues counting the reserve."""
    _StageLedger(str(ledger_path), HARD_CAP_USD).retire(attempt_id, who, reason)


def run_all(folder: str | Path, review_dir: str | Path, reader: str = DEFAULT_READER,
            ledger_path: str | Path = DEFAULT_LEDGER_PATH, *, api: Any | None = None,
            sleep: Any = time.sleep, poll_interval: float = 5.0,
            poll_bound: float = 3600.0) -> Path:
    """Resume durable chunk records; never POST a chunk that already has a record."""
    api = api or setting.call
    if reader not in READER_MODELS:
        raise ValueError(f"unknown Part 62 reader {reader!r}")
    review_dir = Path(review_dir).resolve()
    if reader == "opus" and not _haiku_calibration_failed(review_dir):
        resume = (f"python3 part62/blind62.py run-all --folder {Path(folder)} "
                  f"--review-dir {review_dir} --reader haiku --ledger-path {Path(ledger_path)}")
        raise ValueError("Opus fallback refused: recorded failed Haiku calibration required. "
                         f"Run or resume Haiku calibration with: {resume}")
    review_dir, idmap, keymap, requests = _preflight_submission(folder, review_dir, reader)
    ledger_path = Path(ledger_path).resolve()
    model = READER_MODELS[reader]
    cal = [r for r in requests if reader == "haiku" and idmap[r["custom_id"]].startswith("cal|")]
    answers = [r for r in requests if not idmap[r["custom_id"]].startswith("cal|")]
    run_path = review_dir / ("part62-run-all.json" if reader == "haiku" else "part62-run-all-opus.json")
    with (review_dir / "part62-run-all.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            fingerprint = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
            if run_path.exists():
                run = json.loads(run_path.read_text())
                if run.get("fingerprint") != fingerprint or run.get("reader") != reader:
                    raise ValueError("existing run plan differs from request snapshot")
            else:
                run = {"schema": "part62-run-all-v1", "run_id": uuid.uuid4().hex,
                       "fingerprint": fingerprint, "reader": reader, "model": model,
                       "status": "calibration-pending" if reader == "haiku" else "answers-pending",
                       "chunks": []}
                _persist_run(run_path, run)
            request_by_custom = {row["custom_id"]: row for row in requests}
            for chunk in run["chunks"]:
                if chunk.get("status") == "reserving":
                    attempts = _ledger_attempts(ledger_path)
                    recovered = next((attempt_id for attempt_id, row in attempts.items()
                                      if row.get("meta", {}).get("submission_id") == chunk.get("submission_id")), None)
                    if recovered is None:
                        reqs = [request_by_custom[key] for key in chunk["custom_ids"]]
                        recovered = _StageLedger(str(ledger_path), HARD_CAP_USD).reserve(
                            _chunk_cost(reqs, model), READER_KINDS[reader],
                            {"submission_id": chunk["submission_id"], "stage": chunk["stage"],
                             "chunk_index": chunk["index"]})
                    chunk.update(attempt_id=recovered, status="reserved")
                    _persist_run(run_path, run)
                if chunk.get("status") == "posting":
                    raw_path = review_dir / f"post-response-{chunk['submission_id']}.json"
                    if raw_path.is_file():
                        response = json.loads(raw_path.read_text())
                        chunk.update(status="submitted", batch_id=response.get("id"))
                        _persist_run(run_path, run)
                    else:
                        chunk["status"] = "ambiguous"
                        _persist_run(run_path, run)
                if chunk.get("status") == "ambiguous":
                    attempt = _StageLedger(str(ledger_path), HARD_CAP_USD)
                    with attempt._locked():
                        events, _ = attempt._read_locked()
                        state = attempt._attempts(events).get(chunk["attempt_id"], {})
                    if state.get("status") == "retired":
                        chunk["status"] = "retired-unknown"
                        _persist_run(run_path, run)
                        continue
                    cmd = (f"python3 part62/blind62.py retire {chunk['attempt_id']} --who OPERATOR "
                           f"--reason 'verified outcome unknown' --ledger-path {ledger_path}")
                    raise ValueError(f"chunk unresolved; reserve remains counted. Run: {cmd}")
                if chunk.get("status") == "submitted":
                    _settle_recorded(chunk, api, review_dir, run, run_path, idmap, model, ledger_path,
                                     sleep=sleep, interval=poll_interval, bound=poll_bound)
            if reader == "haiku":
                cal_record = run.get("calibration")
                if not _calibration_gate(cal_record):
                    c = next((x for x in run["chunks"] if x.get("stage") == "calibration"), None)
                    if c is None:
                        c = _post_chunk(cal, idmap, run, run_path, review_dir, ledger_path,
                                        reader, model, api, "calibration", 0,
                                        sleep=sleep, poll_interval=poll_interval, poll_bound=poll_bound)
                    elif c.get("status") == "reserved":
                        _send_reserved(c, cal, idmap, run, run_path, review_dir, ledger_path,
                                       reader, model, api, sleep=sleep,
                                       poll_interval=poll_interval, poll_bound=poll_bound)
                    if c.get("status") != "settled":
                        command = (f"python3 part62/blind62.py run-all --folder {Path(folder)} "
                                   f"--review-dir {review_dir} --reader haiku --ledger-path {ledger_path}")
                        raise ValueError(f"calibration pending; answers refused. Resume with: {command}")
                    labels = c.get("labels", {})
                    issues = c.get("completion_issues", {})
                    kappa = _calibration_kappa(labels)
                    unlabelled = sum(labels.get(f"cal|{i}|{label}|ORIGIN") is None
                                     for i, (label, _, _) in enumerate(CAL))
                    truncated = sum(reason not in ACCEPTED_STOP_REASONS for reason in issues.values())
                    passed = len(labels) == 120 and kappa is not None and kappa >= CALIBRATION_KAPPA_MIN and not unlabelled and not truncated
                    run["calibration"] = {"status": "passed" if passed else "failed", "reader": reader,
                        "model": model, "calibration": {"count": len(labels), "kappa": kappa,
                        "unlabelled": unlabelled, "truncated": truncated, "labels": labels}}
                    _persist_run(run_path, run)
                    if not passed:
                        raise ValueError("calibration failed; answers refused")
                if not _calibration_gate(run.get("calibration")):
                    raise ValueError("calibration missing, pending, or failed; answers refused")
            for start in range(0, len(answers), 200):
                index = start // 200
                old = next((x for x in run["chunks"] if x.get("stage") == "answers" and x.get("index") == index), None)
                if old is not None:
                    if old.get("status") == "reserved":
                        _send_reserved(old, answers[start:start+200], idmap, run, run_path,
                                       review_dir, ledger_path, reader, model, api,
                                       sleep=sleep, poll_interval=poll_interval, poll_bound=poll_bound)
                        continue
                    if old.get("status") != "settled":
                        command = (f"python3 part62/blind62.py run-all --folder {Path(folder)} "
                                   f"--review-dir {review_dir} --reader {reader} --ledger-path {ledger_path}")
                        raise ValueError(f"answer chunk {index} unresolved; resume with: {command}")
                    continue
                _post_chunk(answers[start:start+200], idmap, run, run_path, review_dir,
                            ledger_path, reader, model, api, "answers", index,
                            sleep=sleep, poll_interval=poll_interval, poll_bound=poll_bound)
            labels, issues, batches = {}, {}, []
            for chunk in run["chunks"]:
                if chunk.get("stage") == "answers":
                    labels.update(chunk.get("labels", {})); issues.update(chunk.get("completion_issues", {}))
                    batches.append(chunk["batch_id"])
            passed = not issues
            try:
                scoring_gate({**labels, **(run.get("calibration") or {}).get("calibration", {}).get("labels", {})},
                             list(keymap), issues, require_calibration=(reader == "haiku"))
            except ValueError:
                passed = False
            result = {"schema": "part62-blind-labels-v1", "reader": reader, "model": model,
                      "batch_ids": batches,
                      "calibration": (run.get("calibration") or {}).get("calibration"),
                      "completion": {"issues": issues, "passed": not issues}, "labels": labels,
                      "answer_count": len(keymap), "unlabelled_answers": [k for k in keymap if labels.get(k) is None],
                      "scoring_allowed": passed, "paid_chunks": run["chunks"], "ledger_path": str(ledger_path)}
            output = review_dir / f"part62-labels-{batches[-1]}.json"
            _save_json(output, result)
            run.update(status="complete" if passed else "labels-incomplete", output=output.name)
            _persist_run(run_path, run)
            if not passed:
                raise ValueError("Part 62 labels incomplete; scoring refused")
            return output
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def submit(folder: str | Path, review_dir: str | Path, reader: str = DEFAULT_READER,
           ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> str:
    output = run_all(folder, review_dir, reader, ledger_path)
    return json.loads(output.read_text())["batch_ids"][-1]

def status(batch_id: str, review_dir: str | Path = "results/part62-review") -> None:
    result = json.loads(setting.call("GET", f"/v1/messages/batches/{batch_id}"))
    batch_path = Path(review_dir) / f"part62-batch-{batch_id}.json"
    reader = json.loads(batch_path.read_text(encoding="utf-8")).get("reader", "unknown") if batch_path.is_file() else "unknown"
    print(f"reader={reader} batch={batch_id} status={result.get('processing_status')} counts={result.get('request_counts')}")


def parse_batch_results(lines, idmap: dict, max_tokens: int = MAX_TOKENS) -> tuple[dict, dict]:
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
        output_tokens = message.get("usage", {}).get("output_tokens")
        if isinstance(output_tokens, int) and output_tokens > max_tokens:
            completion_reasons[internal_id] = "output_tokens_exceeded_max_tokens"
        content = " ".join(block.get("text", "") for block in message.get("content", [])
                           if block.get("type") == "text")
        labels[internal_id] = origin.parse_labels(content)
    for internal_id in idmap.values():
        labels.setdefault(internal_id, None)
        completion_reasons.setdefault(internal_id, "missing_result")
    return labels, completion_reasons


def _print_tally(key: dict, labels: dict, reader: str = DEFAULT_READER) -> None:
    counts = {}
    for answer_id, meta in key.items():
        group = (meta["model"], meta["phase"], meta["kind"], float(meta["dose"]))
        row = counts.setdefault(group, {"n": 0, "EARTH": 0, "OFFWORLD": 0,
                                        "NEITHER": 0, "BROKEN": 0})
        row["n"] += 1
        label = labels[answer_id]
        row[label[0]] += 1
        row["BROKEN"] += label[1] == "BROKEN"
    print(f"reader={reader} | model | phase | kind | dose | n | EARTH OFFWORLD NEITHER | broken")
    for group in sorted(counts):
        model, phase, kind, dose = group
        row = counts[group]
        print(f"reader={reader} | {model} | {phase} | {kind} | {dose:g} | {row['n']} | "
              f"{row['EARTH']} {row['OFFWORLD']} {row['NEITHER']} | {row['BROKEN']}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--folder", default=str(PROJECT_ROOT / "results" / "part62"))
    submit_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    submit_parser.add_argument("--reader", choices=tuple(READER_MODELS), default=DEFAULT_READER)
    submit_parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    run_parser = sub.add_parser("run-all", help="submit and resume calibration then answer chunks")
    run_parser.add_argument("--folder", default=str(PROJECT_ROOT / "results" / "part62"))
    run_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    run_parser.add_argument("--reader", choices=tuple(READER_MODELS), default=DEFAULT_READER)
    run_parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    retire_parser = sub.add_parser("retire", help="operator-retire an unknown paid attempt")
    retire_parser.add_argument("attempt_id")
    retire_parser.add_argument("--who", required=True)
    retire_parser.add_argument("--reason", required=True)
    retire_parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    status_parser = sub.add_parser("status")
    status_parser.add_argument("batch_id")
    status_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "submit":
        submit(args.folder, args.review_dir, args.reader, args.ledger_path)
    elif args.command == "run-all":
        run_all(args.folder, args.review_dir, args.reader, args.ledger_path)
    elif args.command == "retire":
        retire_attempt(args.attempt_id, args.who, args.reason, args.ledger_path)
    elif args.command == "status":
        status(args.batch_id, args.review_dir)


if __name__ == "__main__":
    main()
