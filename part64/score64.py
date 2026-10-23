#!/usr/bin/env python3
"""Score the four Part 64 arms with Part 60's pinned Qwen scorer and pass rule."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
if str(BASIN) not in sys.path:
    sys.path.insert(0, str(BASIN))

from part60 import analyze60, score60  # noqa: E402
from part64 import collect64  # noqa: E402

ARMS = ("haiku-1x", "haiku-3x", "opus-1x", "opus-3x")
TIERS = ("strong", "weak", "none")


class Score64Error(RuntimeError):
    pass


def canonical_arm(value: Any) -> str:
    arm = str(value).strip().lower().replace("_", "-")
    if arm not in ARMS:
        raise Score64Error(f"unknown Part 64 arm: {value!r}")
    return arm


def _normalize_generator_row(value: dict[str, Any]) -> dict[str, Any]:
    """Accept either the flat handoff or gen64.py's durable group-output rows."""
    row = dict(value)
    if "arm" not in row and isinstance(row.get("metadata"), dict):
        metadata = row["metadata"]
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        writer = str(metadata.get("writer", "")).strip().lower()
        sample = str(metadata.get("sample", "")).strip().lower()
        row["arm"] = f"{writer}-{sample}"
        row["unit_key"] = metadata.get("unit_key")
        row["type"] = metadata.get("type")
        row["layer"] = metadata.get("layer")
        for tier in TIERS:
            row[tier] = parsed.get(tier)
    return row


def read_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    fingerprints = {}
    for path in paths:
        if not path.is_file():
            raise Score64Error(f"generation input does not exist: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprints[str(path)] = digest
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise Score64Error(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise Score64Error(f"{path}:{line_no} is not a JSON object")
                row = _normalize_generator_row(row)
                row["arm"] = canonical_arm(row.get("arm"))
                rows.append(row)
    return rows, fingerprints


def unit_specs(units: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs = []
    for key, value in units.items():
        record = value["record"]
        specs.append({"unit_key": key, "record_index": int(value["record_index"]),
                      "type": str(value["type"]), "layer": int(value["layer"]),
                      "unit_index": int(record["unit"]) if "unit" in record else None,
                      "control_id": int(record["control_id"]) if "control_id" in record else None})
    return specs


def _valid_texts(row: dict[str, Any]) -> tuple[dict[str, list[str]] | None, str | None]:
    if row.get("parse_status") != "ok":
        return None, str(row.get("parse_status", "missing parse_status"))
    result = {}
    for tier in TIERS:
        values = row.get(tier)
        if (not isinstance(values, list) or len(values) != 4
                or any(not isinstance(text, str) or not text.strip() for text in values)):
            return None, f"{tier} must contain exactly four nonempty sentences"
        result[tier] = values
    return result, None


def score_rows(rows: list[dict[str, Any]], specs: list[dict[str, Any]], thresholds: dict[str, float],
               scorer: Any, *, batch_size: int = 8) -> list[dict[str, Any]]:
    by_key = {spec["unit_key"]: spec for spec in specs}
    by_col = {spec["unit_key"]: col for col, spec in enumerate(specs)}
    seen: dict[str, set[str]] = {arm: set() for arm in ARMS}
    prepared: list[tuple[dict[str, Any], dict[str, list[str]] | None, str | None]] = []
    entries: list[dict[str, Any]] = []
    locations: dict[tuple[int, str, int], int] = {}
    for row_index, original in enumerate(rows):
        row = dict(original)
        arm = canonical_arm(row.get("arm"))
        key = str(row.get("unit_key", ""))
        if key not in by_key:
            raise Score64Error(f"row has unit outside the canonical Part 60 dev set: {key!r}")
        if key in seen[arm]:
            raise Score64Error(f"duplicate generation row for {arm}/{key}")
        seen[arm].add(key)
        spec = by_key[key]
        if str(row.get("type", "")) != spec["type"]:
            raise Score64Error(f"generation type disagrees with Part 60 dev unit {key}")
        texts, error = _valid_texts(row)
        prepared.append((row, texts, error))
        if texts is not None:
            for tier in TIERS:
                for sentence_index, text in enumerate(texts[tier]):
                    locations[(row_index, tier, sentence_index)] = len(entries)
                    entries.append({"text": text, "marked_span": None})
    expected = set(by_key)
    for arm in ARMS:
        missing = expected - seen[arm]
        if missing:
            raise Score64Error(f"{arm} has {len(missing)} missing dev-unit rows; first={sorted(missing)[:3]}")
    output = []
    if entries:
        maximum = np.full((len(entries), len(specs)), np.nan, dtype=np.float32)
        marked = np.full_like(maximum, np.nan)
        scorer.score_batch(entries, list(range(len(entries))), maximum, marked, batch_size=batch_size)
    else:
        maximum = np.empty((0, len(specs)), dtype=np.float32)
    for row_index, (row, texts, error) in enumerate(prepared):
        key = str(row["unit_key"])
        spec = by_key[key]
        base = {**row, "record_index": spec["record_index"], "layer": spec["layer"],
                "type": spec["type"], "scorer_model_revision": score60.MODEL_REVISION}
        if texts is None:
            base.update({"generation_parse_status": row.get("parse_status"), "parse_status": "failed",
                         "score_status": "failed", "score_error": error,
                         "threshold": float(thresholds[key]), "activations": {tier: [] for tier in TIERS},
                         "pass": False, "positive_only_pass": False})
            output.append(base)
            continue
        col = by_col[key]
        activations = {tier: [float(maximum[locations[(row_index, tier, j)], col]) for j in range(4)]
                       for tier in TIERS}
        if any(not math.isfinite(value) for tier in TIERS for value in activations[tier]):
            base.update({"generation_parse_status": "ok", "parse_status": "failed",
                         "score_status": "failed", "score_error": "scorer returned a non-finite activation",
                         "threshold": float(thresholds[key]), "activations": activations,
                         "pass": False, "positive_only_pass": False})
            output.append(base)
            continue
        stats = analyze60.tier_statistics(activations["strong"], activations["weak"], activations["none"],
                                          float(thresholds[key]), parsed=True)
        base.update({"score_status": "ok", "threshold": float(thresholds[key]),
                     "activations": activations, "tier_statistics": stats,
                     "positive_only_pass": bool(stats["positive_only_verified"]),
                     "pass": bool(stats["verified"])})
        output.append(base)
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    project_root = Path(os.environ.get("PART64_BASIN", BASIN)).resolve()
    default_root = project_root / "results" / "part64"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="generated JSONL; repeat for gen64 group files or supply the flat Part 64 schema")
    parser.add_argument("--out", type=Path, default=default_root / "scored.jsonl")
    parser.add_argument("--part60-root", type=Path, default=project_root / "results" / "part60")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    scorer = None
    try:
        rows, inputs = read_rows(args.input)
        if not rows:
            raise Score64Error("no generated sentences were found")
        units = collect64.unit_index()
        specs = unit_specs(units)
        thresholds = score60.ensure_thresholds(args.part60_root)
        missing_thresholds = [spec["unit_key"] for spec in specs if spec["unit_key"] not in thresholds]
        if missing_thresholds:
            raise Score64Error(f"Part 60 has no fixed threshold for {len(missing_thresholds)} dev units; first={missing_thresholds[:3]}")
        model_dir = args.model or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
                                   / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / score60.MODEL_REVISION)
        if not model_dir.is_dir():
            raise Score64Error(f"pinned Qwen3-1.7B snapshot is missing: {model_dir}")
        if args.batch_size < 1:
            raise Score64Error("batch size must be positive")
        scorer = score60.QwenScorer(specs, model_dir, dtype=args.dtype, device=args.device, threads=args.threads)
        scored = score_rows(rows, specs, thresholds, scorer, batch_size=args.batch_size)
        _write_jsonl(args.out, scored)
        manifest = {"version": 1, "inputs_sha256": inputs,
                    "model": score60.MODEL_REPO, "model_revision": score60.MODEL_REVISION,
                    "dtype": scorer.dtype, "device": scorer.device, "batch_size": args.batch_size,
                    "unit_count": len(specs), "row_count": len(scored),
                    "pass_rule": "at least 3 of 4 strong activations strictly above Part 60 threshold and at most 1 of 4 none activations strictly above threshold",
                    "score_backend": score60.SCORE_BACKEND}
        score60.write_json(args.out.with_suffix(args.out.suffix + ".manifest.json"), manifest)
        print(f"scored {len(scored)} unit-arm rows -> {args.out}")
        return 0
    except (Score64Error, score60.ScoreError, collect64.CollectionError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if scorer is not None:
            scorer.close()


if __name__ == "__main__":
    raise SystemExit(main())
