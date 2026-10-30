#!/usr/bin/env python3
"""Prepare, label, and score Part 86's triage miss-rate sample.

Run from the Assistant Basin repository root with ``python -I part86/miss86.py``.
The sample is drawn from the final screen answers; the private key must be kept
outside both the repository and the directory that contains the reader items.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import secrets
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import blind86  # noqa: E402
import common86  # noqa: E402
import triage86  # noqa: E402

LAYERS = tuple(range(10, 27))
SAMPLE_SIZE = 200
SAMPLE_SPECS = {
    "real_kept": ("real", "kept"),
    "real_broken": ("real", "broken"),
    "random_kept": ("random", "kept"),
    "random_broken": ("random", "broken"),
}
CLASS_ORDER = ("not_coherent", "unclear", "coherent_still_qwen", "coherent_persona")
DEFAULT_SCREEN_DIR = ROOT / "results" / "part86" / "screen-final"
PREDICTION_SCORER = ROOT / "results" / "part86" / "prediction-score" / "score86_predictions.py"


class PreflightError(ValueError):
    """Items or labels are incomplete or invalid; the private key was not opened."""


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _read_screen_jsonl(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object screen row in {path}:{line_number}")
            yield row


def read_screen(screen_dir: str | Path) -> list[dict[str, Any]]:
    """Read all final screen layers, accepting the repository's JSONL and gzip formats."""
    screen_dir = Path(screen_dir)
    paths = list(screen_dir.glob("layer_*.jsonl")) + list(screen_dir.glob("layer_*.jsonl.gz"))
    paths.sort(key=lambda path: path.name)
    if not paths:
        raise ValueError(f"no layer_*.jsonl or layer_*.jsonl.gz files under {screen_dir}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    layers_seen: set[int] = set()
    for path in paths:
        expected_layer = int(path.name.split("_")[1].split(".")[0])
        if expected_layer not in LAYERS:
            raise ValueError(f"unexpected layer file {path.name}")
        for row in _read_screen_jsonl(path):
            layer, kind, unit = row.get("layer"), row.get("kind"), row.get("id")
            if type(layer) is not int or layer not in LAYERS or layer != expected_layer:
                raise ValueError(f"invalid or mismatched layer in {path}")
            if kind not in ("neuron", "random_direction"):
                raise ValueError(f"invalid screen kind in {path}: {kind!r}")
            if type(unit) is not int or unit < 0:
                raise ValueError(f"invalid screen unit id in {path}: {unit!r}")
            if not isinstance(row.get("answer"), str):
                raise ValueError(f"screen answer is not text in {path}")
            identity = (layer, kind, unit)
            if identity in seen:
                raise ValueError(f"duplicate screen identity {identity}")
            seen.add(identity)
            layers_seen.add(layer)
            rows.append(row)
    missing_layers = sorted(set(LAYERS) - layers_seen)
    if missing_layers:
        raise ValueError(f"screen is missing layers: {missing_layers}")
    return rows


def _atomic_write(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _sample_across_layers(candidates: list[dict[str, Any]], count: int, chooser: random.Random) -> list[dict[str, Any]]:
    """Cover every represented layer when possible, then fill the sample from the remaining pooled answers."""
    if count == 0:
        return []
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    layers = sorted(by_layer)
    if count < len(layers):
        return chooser.sample(candidates, count)
    selected = [chooser.choice(by_layer[layer]) for layer in layers]
    selected_ids = {(int(row["layer"]), int(row["id"])) for row in selected}
    remaining = [row for row in candidates if (int(row["layer"]), int(row["id"])) not in selected_ids]
    selected.extend(chooser.sample(remaining, count - len(selected)))
    return selected


def prepare(
    screen_dir: str | Path,
    items_path: str | Path,
    key_path: str | Path,
    *,
    force: bool = False,
) -> dict:
    """Select the four seed-8686 samples and write blinded items plus their private key."""
    items_path, key_path = Path(items_path), Path(key_path)
    if _inside(key_path, items_path.parent):
        raise ValueError("private key must be outside the folder containing reader items")
    if _inside(key_path, ROOT):
        raise ValueError("private key must be outside the Assistant Basin repository")
    if items_path.resolve() == key_path.resolve():
        raise ValueError("items and private key paths must differ")
    if not force and (items_path.exists() or key_path.exists()):
        raise FileExistsError("items or private key already exists; pass --force to overwrite")

    pools: dict[str, list[dict[str, Any]]] = {name: [] for name in SAMPLE_SPECS}
    for row in read_screen(screen_dir):
        population = "real" if row["kind"] == "neuron" else "random"
        category = triage86.triage_answer(row["answer"])
        sample_name = f"{population}_{category}"
        if sample_name in pools:
            pools[sample_name].append(row)

    # Sort before sampling so input file order does not affect the seed-8686 draw.
    chooser = random.Random(common86.MISS_SAMPLE_SEED)
    selected: list[tuple[str, str, dict[str, Any]]] = []
    sample_manifest = {}
    for name, (population, category) in SAMPLE_SPECS.items():
        candidates = sorted(pools[name], key=lambda row: (int(row["layer"]), int(row["id"])))
        count = min(SAMPLE_SIZE, len(candidates))
        chosen = _sample_across_layers(candidates, count, chooser)
        sample_manifest[name] = {
            "population": population,
            "class": category,
            "requested": SAMPLE_SIZE,
            "available": len(candidates),
            "sampled": count,
            "shortfall": len(candidates) < SAMPLE_SIZE,
            "shortfall_note": (
                f"Only {len(candidates)} {population} answers were {category}; "
                f"all {len(candidates)} were included."
                if len(candidates) < SAMPLE_SIZE else None
            ),
        }
        selected.extend((name, category, row) for row in chosen)
    chooser.shuffle(selected)

    used_ids: set[str] = set()
    public_items = []
    private_items = {}
    for _sample_name, category, row in selected:
        item_id = secrets.token_urlsafe(18)
        while item_id in used_ids:
            item_id = secrets.token_urlsafe(18)
        used_ids.add(item_id)
        public_items.append({"id": item_id, "text": row["answer"]})
        private_items[item_id] = {
            "population": "real" if row["kind"] == "neuron" else "random",
            "class": category,
            "layer": int(row["layer"]),
            "unit": int(row["id"]),
        }

    item_bytes = _jsonl_bytes(public_items)
    private_key = {
        "format_version": 1,
        "seed": common86.MISS_SAMPLE_SEED,
        "requested_per_sample": SAMPLE_SIZE,
        "items_sha256": hashlib.sha256(item_bytes).hexdigest(),
        "samples": sample_manifest,
        "items": private_items,
    }
    key_bytes = (json.dumps(private_key, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(items_path, item_bytes)
    _atomic_write(key_path, key_bytes)
    return {"items": str(items_path), "key": str(key_path), "samples": sample_manifest}


def _load_calib86():
    calibration_dir = ROOT / "results" / "part86" / "second-reader"
    if str(calibration_dir) not in sys.path:
        sys.path.insert(0, str(calibration_dir))
    spec = importlib.util.spec_from_file_location("part86_calib86", calibration_dir / "calib86.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the Part 86 calibration reader at {calibration_dir / 'calib86.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _default_usage_path(out_path: Path) -> Path:
    if out_path.name.endswith(".jsonl"):
        return out_path.with_name(out_path.name[:-6] + ".usage.jsonl")
    return out_path.with_name(out_path.name + ".usage.jsonl")


def label(
    items_path: str | Path,
    out_path: str | Path,
    usage_path: str | Path | None = None,
    *,
    workers: int | None = None,
) -> dict:
    """Run calib86's DS/brief-v2 labeler with its normal resume and retry behavior."""
    items_path, out_path = Path(items_path), Path(out_path)
    usage_path = Path(usage_path) if usage_path is not None else _default_usage_path(out_path)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    calib86 = _load_calib86()
    reader = "ds"
    version = 2
    if calib86.READERS[reader]["route"] == "ocgo" and not os.environ.get("OPENCODE_GO_API_KEY"):
        raise SystemExit("STOP: OPENCODE_GO_API_KEY is not set (read it from its env file in the run command)")

    # Match calib86.label's reader, brief, prompt construction, output schema, and resumable label_items loop.
    brief_text = calib86.BRIEFS[version].read_text(encoding="utf-8")
    request_fn = calib86.make_request(reader, usage_path, brief_text)
    return calib86.second86.label_items(
        items_path,
        out_path,
        key=calib86.UNUSED_KEY,
        request_fn=request_fn,
        workers=workers or calib86.READERS[reader]["workers"],
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Return a two-sided Wilson score 95% interval, or None for an empty sample."""
    if type(successes) is not int or type(total) is not int or total < 0 or not 0 <= successes <= total:
        raise ValueError("Wilson counts must be integers with 0 <= successes <= total")
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _read_jsonl(path: Path, *, preflight: bool = False) -> list[dict]:
    error = PreflightError if preflight else ValueError
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise error(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise error(f"non-object JSON row in {path}:{line_number}")
            rows.append(row)
    return rows


def _preflight(items_rows: list[dict], label_rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Validate complete blind items and labels before any caller opens the private key."""
    items: dict[str, dict] = {}
    for row in items_rows:
        if set(row) != {"id", "text"}:
            raise PreflightError("each blind item must contain exactly id and text")
        item_id = row["id"]
        if not isinstance(item_id, str) or not item_id or not isinstance(row["text"], str):
            raise PreflightError("blind item id and text must be nonempty id and string text")
        if item_id in items:
            raise PreflightError(f"duplicate blind item id {item_id!r}")
        items[item_id] = row

    labels: dict[str, dict] = {}
    duplicate_ids = []
    error_ids = []
    for row in label_rows:
        item_id = row.get("id")
        if isinstance(item_id, str) and "error" in row:
            error_ids.append(item_id)
        if not isinstance(item_id, str) or not item_id:
            raise PreflightError("label row has no nonempty string id")
        if item_id in labels:
            duplicate_ids.append(item_id)
        else:
            labels[item_id] = row
    missing = sorted(set(items) - set(labels))
    extra = sorted(set(labels) - set(items))
    problems = []
    if missing:
        problems.append(f"missing {len(missing)} label row(s)")
    if extra:
        problems.append(f"{len(extra)} label id(s) do not match an item")
    if duplicate_ids:
        problems.append(f"duplicate label row(s): {len(duplicate_ids)}")
    if error_ids:
        problems.append(f"{len(error_ids)} reader error row(s) remain")
    if problems:
        raise PreflightError("label preflight failed: " + "; ".join(problems))
    try:
        blind86._validate_labels(list(labels.values()), {item_id: {} for item_id in items})
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError(f"label preflight failed: {exc}") from exc
    return items, labels


def _validate_key(key: dict, items: dict[str, dict]) -> dict[str, dict]:
    if not isinstance(key, dict) or key.get("format_version") != 1:
        raise ValueError("unsupported or malformed private key")
    if key.get("seed") != common86.MISS_SAMPLE_SEED or key.get("requested_per_sample") != SAMPLE_SIZE:
        raise ValueError("private key seed or requested sample size does not match this design")
    if set(key.get("items", {})) != set(items):
        raise ValueError("private key ids do not match the blind items")
    samples = key.get("samples")
    if not isinstance(samples, dict) or set(samples) != set(SAMPLE_SPECS):
        raise ValueError("private key does not describe all four samples")
    seen_units: set[tuple[str, int, int]] = set()
    actual_counts = Counter()
    for item_id, metadata in key["items"].items():
        if not isinstance(metadata, dict) or set(metadata) != {"population", "class", "layer", "unit"}:
            raise ValueError(f"malformed private metadata for item {item_id}")
        population, category = metadata["population"], metadata["class"]
        layer, unit = metadata["layer"], metadata["unit"]
        sample_name = f"{population}_{category}"
        if sample_name not in SAMPLE_SPECS or type(layer) is not int or layer not in LAYERS:
            raise ValueError(f"invalid private metadata for item {item_id}")
        if type(unit) is not int or unit < 0:
            raise ValueError(f"invalid private unit id for item {item_id}")
        identity = (population, layer, unit)
        if identity in seen_units:
            raise ValueError(f"screen unit appears more than once in the samples: {identity}")
        seen_units.add(identity)
        actual_counts[sample_name] += 1
    for name, (population, category) in SAMPLE_SPECS.items():
        details = samples[name]
        if not isinstance(details, dict):
            raise ValueError(f"malformed sample metadata for {name}")
        if details.get("population") != population or details.get("class") != category:
            raise ValueError(f"wrong population or class in private sample metadata for {name}")
        requested, available, sampled = (details.get(field) for field in ("requested", "available", "sampled"))
        if any(type(value) is not int or value < 0 for value in (requested, available, sampled)):
            raise ValueError(f"invalid sample sizes for {name}")
        if requested != SAMPLE_SIZE or sampled != min(requested, available) or actual_counts[name] != sampled:
            raise ValueError(f"sample size metadata does not match key rows for {name}")
    return samples


def score(items_path: str | Path, key_path: str | Path, labels_path: str | Path) -> dict:
    """Preflight items/labels, then open the key and calculate the four miss-rate samples."""
    items_path, key_path, labels_path = Path(items_path), Path(key_path), Path(labels_path)
    item_bytes = items_path.read_bytes()
    try:
        item_rows = [json.loads(line) for line in item_bytes.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid blind items file {items_path}: {exc}") from exc
    label_rows = _read_jsonl(labels_path, preflight=True)
    items, labels = _preflight(item_rows, label_rows)

    # Keep this read below the complete item/label preflight: condition assignments stay concealed on failure.
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if hashlib.sha256(item_bytes).hexdigest() != key.get("items_sha256"):
        raise ValueError("blind items SHA-256 does not match the private key")
    sample_metadata = _validate_key(key, items)

    by_sample: dict[str, Counter] = {name: Counter() for name in SAMPLE_SPECS}
    for item_id, metadata in key["items"].items():
        sample_name = f"{metadata['population']}_{metadata['class']}"
        by_sample[sample_name][score86_predictions.label_class(labels[item_id])] += 1

    output_samples = {}
    for name, (population, category) in SAMPLE_SPECS.items():
        counts = {label_class: by_sample[name].get(label_class, 0) for label_class in CLASS_ORDER}
        sampled = int(sample_metadata[name]["sampled"])
        available = int(sample_metadata[name]["available"])
        coherent = counts["coherent_persona"]
        interval = wilson_interval(coherent, sampled)
        shortfall = available < SAMPLE_SIZE
        output_samples[name] = {
            "population": population,
            "triage_class": category,
            "requested": SAMPLE_SIZE,
            "available": available,
            "sampled": sampled,
            "shortfall": shortfall,
            "shortfall_note": (
                f"Only {available} {population} answers were {category}; all {available} were included."
                if shortfall else None
            ),
            "coherent_personas": coherent,
            "coherent_persona_share": coherent / sampled if sampled else None,
            "wilson_95": list(interval) if interval is not None else None,
            "version_2_class_counts": counts,
        }
    return {
        "sample_seed": common86.MISS_SAMPLE_SEED,
        "requested_per_sample": SAMPLE_SIZE,
        "samples": output_samples,
        "verdict_effect": "none",
    }


def _format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(report: dict) -> str:
    lines = [
        "# Part 86 triage miss-rate sample",
        "",
        "Seed 8686; version 2 reader classes are counted with `score86_predictions.label_class`.",
        "The miss rate is the coherent-persona share among answers triaged kept or broken.",
        "This descriptive sample changes no verdict.",
        "",
        "| Sample | Available | Included | Coherent personas | Share (Wilson 95%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, sample in report["samples"].items():
        interval = sample["wilson_95"]
        bounds = "n/a" if interval is None else f"{100 * interval[0]:.1f}%–{100 * interval[1]:.1f}%"
        lines.append(
            f"| {name} | {sample['available']} | {sample['sampled']} | "
            f"{sample['coherent_personas']} | {_format_pct(sample['coherent_persona_share'])} ({bounds}) |"
        )
    shortages = [sample["shortfall_note"] for sample in report["samples"].values() if sample["shortfall_note"]]
    if shortages:
        lines.extend(["", "Samples with fewer than 200 available answers:"])
        lines.extend(f"- {note}" for note in shortages)
    lines.extend([
        "",
        "## Version 2 class counts",
        "",
        "| Sample | Not coherent | Unclear | Still Qwen or assistant | Coherent persona |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, sample in report["samples"].items():
        counts = sample["version_2_class_counts"]
        lines.append(
            f"| {name} | {counts['not_coherent']} | {counts['unclear']} | "
            f"{counts['coherent_still_qwen']} | {counts['coherent_persona']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_prediction_scorer():
    spec = importlib.util.spec_from_file_location("score86_predictions_for_miss86", PREDICTION_SCORER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Part 86 prediction scorer at {PREDICTION_SCORER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score86_predictions = _load_prediction_scorer()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="draw the four blinded samples")
    prepare_parser.add_argument("--items", required=True, type=Path)
    prepare_parser.add_argument("--key", required=True, type=Path)
    prepare_parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    prepare_parser.add_argument("--force", action="store_true")

    label_parser = commands.add_parser("label", help="run calib86's DeepSeek version 2 reader")
    label_parser.add_argument("--items", required=True, type=Path)
    label_parser.add_argument("--out", required=True, type=Path)
    label_parser.add_argument("--usage", type=Path, help="usage JSONL path (defaults beside --out)")
    label_parser.add_argument("--workers", type=int)

    score_parser = commands.add_parser("score", help="preflight labels and report the miss-rate sample")
    score_parser.add_argument("--items", required=True, type=Path)
    score_parser.add_argument("--key", required=True, type=Path)
    score_parser.add_argument("--labels", required=True, type=Path)
    score_parser.add_argument("--out", required=True, type=Path, help="Markdown output path")
    score_parser.add_argument("--json", required=True, type=Path, help="JSON output path")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.screen_dir, args.items, args.key, force=args.force)
    elif args.command == "label":
        result = label(args.items, args.out, args.usage, workers=args.workers)
        result["usage"] = str(args.usage or _default_usage_path(args.out))
    else:
        report = score(args.items, args.key, args.labels)
        _atomic_write(args.out, render_markdown(report).encode("utf-8"))
        _atomic_write(args.json, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        result = {"markdown": str(args.out), "json": str(args.json), "verdict_effect": "none"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
