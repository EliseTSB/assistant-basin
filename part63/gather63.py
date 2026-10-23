#!/usr/bin/env python3
"""Gather the frozen Part 54, 26.8, and 59A answer corpus for Part 63."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import glob
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASIN = HERE.parent
DEFAULT_SOURCE_ROOT = BASIN / "results"
DEFAULT_OUTPUT_DIR = HERE / "results"
SOURCE_NAMES = ("part54a", "part54b", "part26-8", "part59a")
SHUFFLE_SEED = 6301  # Recorded in manifest63.json so the blind order is reproducible.
ORIGIN_BATCHES = {
    "part54a": "msgbatch_01AYpTn4oDiwdib4keBnRuaA",
    "part54b": "msgbatch_017PdYJMFE2moQjZaTGfp6z7",
    "part26-8": "msgbatch_01XoGrMFw6eRLiGdAEBJhWxt",
}
SETTING_BATCH = "msgbatch_01UkAS6ir2KL3JpgcXJEVKnM"
LEGACY_ORIGIN_SHUFFLE_SEED = 2681
LEGACY_SETTING_SHUFFLE_SEED = 5901


class GatherError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatherError(f"cannot read {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _unit_number(value: Any) -> int | str | None:
    if isinstance(value, dict):
        value = value.get("unit")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _kind(condition: str) -> str:
    lowered = condition.lower()
    if "control" in lowered:
        return "control"
    if lowered == "unclamped":
        return "unclamped"
    return "twin"


def _source_rows(source_root: Path, source_name: str) -> list[dict[str, Any]]:
    folder = source_root / source_name
    if not folder.is_dir():
        raise GatherError(f"missing source directory: {folder}")

    # 59A is manifest-scoped: reuse its strict reader so incomplete/stale files are refused.
    manifest_rows: list[dict[str, Any]] | None = None
    if source_name == "part59a":
        sys.path.insert(0, str(BASIN / "part59"))
        try:
            import blind_setting  # type: ignore
            manifest_rows = blind_setting.answers(folder)
        except (ImportError, ValueError) as exc:
            raise GatherError(f"Part 59A manifests failed validation: {exc}") from exc

    # For the earlier Opus batches, preserve the original blind_origin reader's sorted glob order.
    # That order is needed to validate and attach labels to repeated OLMo-7B rows.
    paths = sorted(glob.glob(str(folder / "clamp-*.json")))
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        data = read_json(path)
        if not isinstance(data, dict) or "clamp_mult" not in data:
            continue
        dose = data.get("clamp_mult")
        if not isinstance(dose, (int, float)):
            raise GatherError(f"{path}: clamp_mult is not numeric")
        model = data.get("short_name") or data.get("model")
        if not isinstance(model, str) or not model:
            model = path.name.removeprefix("clamp-").split("-L", 1)[0]
        question = data.get("question")
        conditions = data.get("conditions")
        if not isinstance(question, str) or not isinstance(conditions, list):
            raise GatherError(f"{path}: missing question or conditions list")
        target_unit = _unit_number(data.get("best_twin"))
        for condition_row in conditions:
            if not isinstance(condition_row, dict):
                raise GatherError(f"{path}: condition is not an object")
            condition = condition_row.get("condition")
            samples = condition_row.get("answers")
            if not isinstance(condition, str) or not isinstance(samples, list):
                raise GatherError(f"{path}: invalid answer condition")
            condition_unit = _unit_number(condition_row.get("unit"))
            for answer_index, text in enumerate(samples):
                if not isinstance(text, str):
                    raise GatherError(f"{path}: {condition} answer {answer_index} is not text")
                rows.append({
                    "source": source_name,
                    "source_file": path.name,
                    "source_condition": condition,
                    "source_answer_index": answer_index,
                    "model": model,
                    "unit": condition_unit,
                    "target_unit": target_unit,
                    "dose": float(dose),
                    "question": question,
                    "kind": _kind(condition),
                    "condition": condition,
                    "text": text,
                })

    if manifest_rows is not None:
        _join_setting_labels(source_name, rows, manifest_rows, source_root.parent / "part59" / "review")
    else:
        _join_origin_labels(source_name, rows, source_root.parent / "review")
    return rows


def _row_key(row: dict[str, Any], *, with_model: bool = True) -> tuple[Any, ...]:
    fields = ["question", "dose", "kind", "condition", "source_answer_index"]
    if with_model:
        fields.insert(0, "model")
    return tuple(row.get(field) for field in fields)


def _legacy_meta_matches(meta: dict[str, Any], row: dict[str, Any], with_model: bool) -> bool:
    expected = {
        "question": row["question"], "dose": row["dose"], "kind": row["kind"],
        "cond": row["condition"], "i": row["source_answer_index"],
    }
    if with_model:
        expected["model"] = row["model"]
    return all(meta.get(field) == value for field, value in expected.items())


def _load_legacy_files(review_dir: Path, prefix: str, batch_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    key_path = review_dir / f"{prefix}-key-{batch_id}.json"
    labels_path = review_dir / f"{prefix}-labels-{batch_id}.json"
    key, labels = read_json(key_path), read_json(labels_path)
    if not isinstance(key, dict) or not isinstance(labels, dict):
        raise GatherError(f"{prefix} sidecars must contain JSON objects")
    return key, labels


def _join_origin_labels(source_name: str, rows: list[dict[str, Any]], review_dir: Path) -> None:
    batch_id = ORIGIN_BATCHES[source_name]
    key, labels = _load_legacy_files(review_dir, "origin", batch_id)
    order = list(rows)
    random.Random(LEGACY_ORIGIN_SHUFFLE_SEED).shuffle(order)
    if len(key) != len(order):
        raise GatherError(f"{source_name}: existing Opus key has {len(key)} rows, source has {len(order)}")
    with_model = source_name != "part26-8"  # The archived 26.8 key predates its model field.
    for index, row in enumerate(order):
        internal_id = f"ans|{index}"
        meta = key.get(internal_id)
        if not isinstance(meta, dict) or not _legacy_meta_matches(meta, row, with_model):
            raise GatherError(f"{source_name}: Opus key mismatch at {internal_id}; refusing an unsafe label join")
        row["existing_opus_label"] = labels.get(internal_id)
        row["existing_opus_batch"] = batch_id
        row["existing_opus_schema"] = "origin"


def _join_setting_labels(source_name: str, rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]],
                         review_dir: Path) -> None:
    key, labels = _load_legacy_files(review_dir, "setting", SETTING_BATCH)
    if len(manifest_rows) != len(rows) or len(key) != len(rows):
        raise GatherError("59A answer count differs between source files, manifest reader, and Opus key")
    raw_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        row_key = _row_key(row)
        if row_key in raw_by_key:
            raise GatherError(f"59A has a duplicate key for {row_key!r}; refusing an unsafe label join")
        raw_by_key[row_key] = row
    shuffled = list(manifest_rows)
    random.Random(LEGACY_SETTING_SHUFFLE_SEED).shuffle(shuffled)
    seen: set[tuple[Any, ...]] = set()
    for index, manifest_row in enumerate(shuffled):
        key_tuple = (manifest_row["model"], manifest_row["question"], float(manifest_row["dose"]),
                     manifest_row["kind"], manifest_row["cond"], manifest_row["i"])
        row_key = (key_tuple[0], key_tuple[1], key_tuple[2], key_tuple[3], key_tuple[4], key_tuple[5])
        # Source key includes the actual source condition and sample number.
        raw_row_key = (key_tuple[0], key_tuple[1], key_tuple[2], key_tuple[3], key_tuple[4], key_tuple[5])
        row = raw_by_key.get(raw_row_key)
        if row is None or row_key in seen:
            raise GatherError(f"59A manifest row does not match a unique source answer: {row_key!r}")
        seen.add(row_key)
        internal_id = f"ans|{index}"
        meta = key.get(internal_id)
        if not isinstance(meta, dict):
            raise GatherError(f"59A Opus key is missing {internal_id}")
        expected = {"model": key_tuple[0], "question": key_tuple[1], "dose": key_tuple[2],
                    "kind": key_tuple[3], "cond": key_tuple[4], "i": key_tuple[5]}
        if any(meta.get(field) != value for field, value in expected.items()):
            raise GatherError(f"59A Opus key mismatch at {internal_id}; refusing an unsafe label join")
        row["existing_opus_label"] = labels.get(internal_id)
        row["existing_opus_batch"] = SETTING_BATCH
        row["existing_opus_schema"] = "setting"
    if len(seen) != len(rows):
        raise GatherError("59A Opus join did not cover every source row")


def collect_sources(source_root: Path = DEFAULT_SOURCE_ROOT) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source_name in SOURCE_NAMES:
        part_rows = _source_rows(source_root, source_name)
        counts[source_name] = len(part_rows)
        rows.extend(part_rows)
    if not rows:
        raise GatherError("no answer rows found")
    return rows, counts


def write_bundle(output_dir: Path, rows: list[dict[str, Any]], seed: int = SHUFFLE_SEED,
                 source_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Write the actual answer file and its metadata-only side key atomically."""
    answer_rows: list[dict[str, str]] = []
    key: dict[str, dict[str, Any]] = {}
    for index, source_row in enumerate(rows):
        answer_id = f"a{index:06d}"
        text = source_row.get("text")
        if not isinstance(text, str):
            raise GatherError(f"answer row {index} has no text")
        key[answer_id] = {name: value for name, value in source_row.items() if name != "text"}
        answer_rows.append({"id": answer_id, "text": text})
    random.Random(seed).shuffle(answer_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers63.jsonl"
    tmp = answers_path.with_name(answers_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in answer_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(answers_path)
    key_path = output_dir / "key63.json"
    write_json(key_path, key)
    digest = hashlib.sha256(answers_path.read_bytes()).hexdigest()
    manifest = {
        "version": 1,
        "shuffle_seed": seed,
        "answer_count": len(answer_rows),
        "source_counts": source_counts or {},
        "answers_file": answers_path.name,
        "key_file": key_path.name,
        "answers_sha256": digest,
        "condition_metadata_in_answers": False,
    }
    write_json(output_dir / "manifest63.json", manifest)
    return manifest


def load_bundle(output_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Read the writer's on-disk files and ensure IDs are paired through the side key."""
    answers_path = output_dir / "answers63.jsonl"
    key = read_json(output_dir / "key63.json")
    rows: list[dict[str, str]] = []
    with answers_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {"id", "text"}:
                raise GatherError(f"answers63.jsonl:{line_number}: expected only id and text")
            rows.append(item)
    if not isinstance(key, dict) or {row["id"] for row in rows} != set(key):
        raise GatherError("answer IDs and side-key IDs do not match")
    if any("text" in meta for meta in key.values()):
        raise GatherError("side key must not contain answer text")
    return rows, key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    args = parser.parse_args(argv)
    try:
        rows, counts = collect_sources(args.source_root)
        manifest = write_bundle(args.output_dir, rows, args.seed, counts)
    except (GatherError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"gathered {manifest['answer_count']} answers; seed={manifest['shuffle_seed']}")
    for name, count in manifest["source_counts"].items():
        print(f"{name}: {count}")
    print(f"wrote {args.output_dir / 'answers63.jsonl'} and key63.json")


if __name__ == "__main__":
    main()
