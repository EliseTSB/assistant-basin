#!/usr/bin/env python3
"""Blindly label Part 62 answers with the Part 54 origin rubric."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import fcntl
from pathlib import Path
import random
import re
import secrets
import sys
import tempfile
import time

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
# Both registered readers need at least 1,024 output tokens.
MAX_TOKENS = 1024
EXPECTED_OUTPUT_TOKENS = 150
# Only normal final-answer stops count as verified completion; max_tokens means truncation.
ACCEPTED_STOP_REASONS = ("end_turn", "stop_sequence")
# This fixed shuffle seed makes the hidden answer order reproducible.
SHUFFLE_SEED = 62062
# This stage has a hard cumulative reserve/settlement cap.
READER_MODELS = {"haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"}
READER_CAPS_USD = {"haiku": 4.0, "opus": 8.0}
BATCH_RATES_USD_PER_MILLION = {"haiku": (0.05, 0.25), "opus": (2.0, 10.0)}
DEFAULT_READER = "haiku"
MODEL = READER_MODELS[DEFAULT_READER]
RUBRIC = origin.RUBRIC
_CAL_LABELS = {"SCIFI": "OFFWORLD", "EARTH-REAL": "EARTH", "FANTASY": "NEITHER", "OTHER": "NEITHER"}
CAL = [(_CAL_LABELS[label], quality, text) for label, quality, text in setting.CAL]


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
    seen_seeds = set()
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
        seed = condition.get("seed")
        if not isinstance(seed, int) or seed in seen_seeds:
            raise ValueError(f"{path.name} has a missing or repeated independent condition seed")
        expected_seed = parameters["seed"]
        if seed != expected_seed:
            raise ValueError(f"{path.name} seed does not match its independent condition key")
        target_maximum = float(model_meta["pile_maximum"])
        expected_value = parameters["set_value"]
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


def estimate_submission_cost(requests: list[dict], reader: str = DEFAULT_READER) -> float:
    """Reserve actual expected output (150 tokens/request) plus request-byte input estimate."""
    if reader not in BATCH_RATES_USD_PER_MILLION:
        raise ValueError(f"unknown Part 62 reader: {reader}")
    input_bytes = sum(len(json.dumps(request, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8"))
                      for request in requests)
    output_tokens = len(requests) * EXPECTED_OUTPUT_TOKENS
    input_rate, output_rate = BATCH_RATES_USD_PER_MILLION[reader]
    return ((input_bytes * input_rate + output_tokens * output_rate) / 1_000_000)


def _submission_ledger(review_dir: Path, reader: str = DEFAULT_READER) -> tuple[Path, dict]:
    model, cap = READER_MODELS[reader], READER_CAPS_USD[reader]
    path = review_dir / f"part62-submission-ledger-{reader}.json"
    if not path.exists():
        return path, {"schema": "part62-submission-ledger-v1", "reader": reader,
                      "cap_usd": cap, "model": model, "attempts": []}
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Part 62 submission ledger: {exc}") from exc
    if (ledger.get("schema") != "part62-submission-ledger-v1"
            or ledger.get("model") != model or ledger.get("reader", reader) != reader
            or float(ledger.get("cap_usd", -1)) != cap
            or not isinstance(ledger.get("attempts"), list)):
        raise ValueError("Part 62 submission ledger has a different model/cap/schema; refusing submit")
    return path, ledger


def _reserved_submission_usd(ledger: dict) -> float:
    active = {"reserved", "posting", "ambiguous", "submitted", "collected"}
    return sum(float(row.get("reserved_usd", 0.0)) for row in ledger["attempts"]
               if row.get("status") in active)


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


def _attempt_paths(review_dir: Path, attempt_id: str) -> dict[str, Path]:
    stem = f"part62-attempt-{attempt_id}"
    return {"idmap": review_dir / f"{stem}-idmap.json",
            "key": review_dir / f"{stem}-key.json",
            "intent": review_dir / f"{stem}-intent.json"}


def _definitive_rejection(exc: BaseException) -> bool:
    """Only release the paid-action reserve for a 4xx carrying an Anthropic error body."""
    match = re.search(r"HTTP\s+(4\d\d)\b[^:]*:\s*(\{.*\})", str(exc), re.DOTALL)
    if not match:
        return False
    try:
        body = json.loads(match.group(2))
    except json.JSONDecodeError:
        return False
    return isinstance(body, dict) and body.get("type") == "error" and isinstance(body.get("error"), dict)


@contextmanager
def _submission_lock(review_dir: Path):
    """Serialize ledger reservations; flock is released automatically on process death."""
    lock_path = review_dir / ".part62-submit.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def prepare_batch(rows: list[dict], reader: str = DEFAULT_READER) -> tuple[dict, dict, list[dict]]:
    """Build hidden-condition API requests and their local-only key without sending them."""
    if MAX_TOKENS < 1024:
        raise ValueError("Part 62 reader max_tokens must be at least 1,024")
    if reader not in READER_MODELS:
        raise ValueError(f"unknown Part 62 reader {reader!r}")
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
            "model": READER_MODELS[reader], "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": RUBRIC + text}],
        }})
    return idmap, key, requests


def _save_attempt_state(ledger_path: Path, ledger: dict, attempt_path: Path,
                        attempt: dict) -> None:
    _save_json(ledger_path, ledger)
    _save_json(attempt_path, attempt)


def _known_batch_attempt(ledger: dict, batch_id: str) -> dict | None:
    return next((row for row in ledger["attempts"]
                 if row.get("batch_id") == batch_id and row.get("status") in ("submitted", "collected")), None)


def _haiku_calibration_failed(review_dir: Path) -> bool:
    for path in sorted(review_dir.glob("part62-labels-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (record.get("reader") == "haiku" and record.get("model") == READER_MODELS["haiku"]
                and record.get("calibration", {}).get("passed") is False):
            return True
    return False


def submit(folder: str | Path, review_dir: str | Path, reader: str = DEFAULT_READER) -> str:
    if reader not in READER_MODELS:
        raise ValueError(f"unknown Part 62 reader {reader!r}")
    review_dir = Path(review_dir).resolve()
    if reader == "opus" and not _haiku_calibration_failed(review_dir):
        raise ValueError("Opus fallback refused: a failed Haiku calibration record is required")
    review_dir, idmap, keymap, requests = _preflight_submission(folder, review_dir, reader)
    with _submission_lock(review_dir):
        return _submit_prepared(review_dir, idmap, keymap, requests, reader)


def _submit_prepared(review_dir: Path, idmap: dict, keymap: dict,
                     requests: list[dict], reader: str = DEFAULT_READER) -> str:
    model, cap = READER_MODELS[reader], READER_CAPS_USD[reader]
    ledger_path, ledger = _submission_ledger(review_dir, reader)
    known = next((row for row in reversed(ledger["attempts"])
                  if row.get("status") in ("submitted", "collected")
                  and isinstance(row.get("batch_id"), str)), None)
    if known is not None:
        batch_id = known["batch_id"]
        print(f"already submitted: {batch_id}; collect it before any new submit")
        return batch_id

    # A process killed in the POST window cannot tell whether the service accepted it.
    # Keep that reservation and allow only a new, separately reserved attempt.
    for old in ledger["attempts"]:
        if old.get("status") == "posting":
            old["status"] = "ambiguous"
            old["ambiguous_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        elif old.get("status") == "reserved":
            old["status"] = "released"
            old["released_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            old["release_reason"] = "process stopped before POST began"
            old["reserved_usd"] = 0.0
    _save_json(ledger_path, ledger)

    reserve = estimate_submission_cost(requests, reader)
    spent = _reserved_submission_usd(ledger)
    if spent + reserve > cap + 1e-12:
        raise ValueError(f"Part 62 spend cap refused attempt: settled/reserved ${spent:.4f} plus "
                         f"conservative reserve ${reserve:.4f} exceeds ${cap:.2f}; no POST was made")

    attempt_id = secrets.token_hex(8)
    paths = _attempt_paths(review_dir, attempt_id)
    request_bytes = json.dumps(requests, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    attempt = {
        "attempt_id": attempt_id, "status": "reserved", "batch_id": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "reader": reader, "model": model, "max_tokens": MAX_TOKENS, "request_count": len(requests),
        "answer_count": len(keymap), "calibration_count": len(CAL),
        "request_sha256": request_sha256, "idmap_file": paths["idmap"].name,
        "key_file": paths["key"].name, "intent_file": paths["intent"].name,
        "reserved_usd": round(reserve, 8), "cap_limit_usd": cap,
    }
    # Persist the blinded request map and a durable intent before recording a
    # reserve, then mark the attempt posting immediately before the one POST.
    _save_json(paths["idmap"], idmap)
    _save_json(paths["key"], keymap)
    _save_json(paths["intent"], attempt)
    ledger["attempts"].append(attempt)
    try:
        _save_attempt_state(ledger_path, ledger, paths["intent"], attempt)
        attempt["status"] = "posting"
        attempt["post_started_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        _save_attempt_state(ledger_path, ledger, paths["intent"], attempt)
    except Exception:
        attempt["status"] = "released"
        attempt["reserved_usd"] = 0.0
        attempt["released_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        attempt["release_reason"] = "local persistence failed before POST"
        try:
            _save_attempt_state(ledger_path, ledger, paths["intent"], attempt)
        except Exception:
            pass
        raise

    try:
        response_text = setting.call("POST", "/v1/messages/batches", {"requests": requests})
        result = json.loads(response_text)
        batch_id = result.get("id") if isinstance(result, dict) else None
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("batch POST response did not contain an id")
    except BaseException as exc:
        if _definitive_rejection(exc):
            attempt["status"] = "rejected"
            attempt["reserved_usd"] = 0.0
            attempt["released_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            _save_attempt_state(ledger_path, ledger, paths["intent"], attempt)
            raise RuntimeError(f"Part 62 batch was definitively rejected; reservation released: {exc}") from None
        attempt["status"] = "ambiguous"
        attempt["ambiguous_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        _save_attempt_state(ledger_path, ledger, paths["intent"], attempt)
        raise RuntimeError("ambiguous Part 62 batch POST; reservation remains counted; rerun submit for a new capped attempt") from None

    # The first durable action after receiving the provider id records it in the
    # ledger. No lookup, listing, or adoption path follows an ambiguous POST.
    attempt["batch_id"] = batch_id
    attempt["status"] = "submitted"
    attempt["submitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    attempt["processing_status"] = result.get("processing_status")
    _save_json(ledger_path, ledger)

    _save_json(paths["intent"], attempt)
    _save_json(review_dir / f"part62-idmap-{batch_id}.json", idmap)
    _save_json(review_dir / f"part62-key-{batch_id}.json", keymap)
    _save_json(review_dir / f"part62-batch-{batch_id}.json", {
        "batch_id": batch_id, "processing_status": result.get("processing_status"),
        "request_count": len(requests), "answer_count": len(keymap),
        "calibration_count": len(CAL), "reader": reader, "model": model, "max_tokens": MAX_TOKENS,
        "attempt_id": attempt_id, "request_sha256": request_sha256,
        "reserved_usd": round(reserve, 8),
    })
    print(f"reader={reader} model={model} batch={batch_id} status={result.get('processing_status')} requests={len(requests)}")
    return batch_id


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


def collect(batch_id: str, review_dir: str | Path) -> dict:
    review_dir = Path(review_dir)
    batch_path = review_dir / f"part62-batch-{batch_id}.json"
    batch_info = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.is_file() else {}
    reader = batch_info.get("reader", DEFAULT_READER)
    model = batch_info.get("model", READER_MODELS.get(reader, MODEL))
    idmap_path = review_dir / f"part62-idmap-{batch_id}.json"
    key_path = review_dir / f"part62-key-{batch_id}.json"
    if not idmap_path.is_file() or not key_path.is_file():
        _ledger_path, ledger = _submission_ledger(review_dir, reader)
        attempt = _known_batch_attempt(ledger, batch_id)
        if attempt is None:
            raise ValueError(f"no saved Part 62 submission intent for batch {batch_id}")
        idmap_path = review_dir / attempt["idmap_file"]
        key_path = review_dir / attempt["key_file"]
    idmap = json.loads(idmap_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
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
        "schema": "part62-blind-labels-v1", "batch_id": batch_id, "reader": reader,
        "model": model, "max_tokens": MAX_TOKENS,
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
    ledger_path, ledger = _submission_ledger(review_dir, reader)
    attempt = _known_batch_attempt(ledger, batch_id)
    if attempt is not None:
        attempt["status"] = "collected"
        attempt["collected_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        _save_json(ledger_path, ledger)
    print(f"reader={reader} model={model}; calibration wrong: {len(wrong)} of {len(CAL)}; "
          f"unlabelled answers: {len(answer_unlabelled)} of {len(key)}")
    try:
        scoring_gate(labels, list(key), completion_issues)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _print_tally(key, labels, reader)
    return record


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
    status_parser = sub.add_parser("status")
    status_parser.add_argument("batch_id")
    status_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("batch_id")
    collect_parser.add_argument("--review-dir", default=str(PROJECT_ROOT / "results" / "part62-review"))
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "submit":
        submit(args.folder, args.review_dir, args.reader)
    elif args.command == "status":
        status(args.batch_id, args.review_dir)
    else:
        collect(args.batch_id, args.review_dir)


if __name__ == "__main__":
    main()
