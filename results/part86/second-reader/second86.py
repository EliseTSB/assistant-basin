#!/usr/bin/env python3
"""Run Part 86's registered seed-8687 second-reader sample and agreement pass."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent
SEED = 8687
SHARE = 0.10
FIRST_MODEL = "claude-haiku-5-5"
SECOND_MODEL = "claude-opus-5-5"
MAX_TOKENS = 4000
MAX_ATTEMPTS = 5


def find_repo_root(start: Path = HERE) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "part86" / "blind86.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the Assistant Basin repository root")


REPO_ROOT = find_repo_root()
PART86_DIR = REPO_ROOT / "part86"
READER_BRIEF_PATH = PART86_DIR / "READER-BRIEF.md"
JLENS_ROOT = Path(os.environ.get("JLENS_ROOT", "~/jlens-persona-neurons")).expanduser()
FIRST_READER_DIR = JLENS_ROOT / "data/qwen3.5-4b/labels"
FIRST_ITEMS_PATH = FIRST_READER_DIR / "items.jsonl"
FIRST_LABELS_PATH = FIRST_READER_DIR / "labels.jsonl"
ITEMS_SECOND_PATH = HERE / "items_second.jsonl"
LABELS_SECOND_PATH = HERE / "labels_second.jsonl"
AGREEMENT_PATH = HERE / "agreement.json"

if str(PART86_DIR) not in sys.path:
    sys.path.insert(0, str(PART86_DIR))
import blind86  # noqa: E402  Part 86 owns label validation and its registered kappa mapping.


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_number}")
            rows.append(row)
    return rows


def read_items(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if set(row) != {"id", "text"}:
            raise ValueError(f"item must contain exactly id and text: {path}")
        if not isinstance(row["id"], str) or not row["id"] or not isinstance(row["text"], str):
            raise ValueError(f"invalid item id or text in {path}")
        if row["id"] in seen:
            raise ValueError(f"duplicate item id {row['id']!r} in {path}")
        seen.add(row["id"])
    return rows


def sample_size(n: int) -> int:
    return max(1, round(SHARE * n)) if n else 0


def sample_items(items: list[dict], seed: int = SEED) -> list[dict]:
    """Select a reproducible 10% by item row, preserving random.sample order."""
    ids = [row.get("id") for row in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("every item needs a nonempty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("item ids must be unique before sampling")
    return random.Random(seed).sample(items, sample_size(len(items)))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sample(source_path: Path = FIRST_ITEMS_PATH, output_path: Path = ITEMS_SECOND_PATH) -> int:
    selected = sample_items(read_items(source_path))
    write_jsonl(output_path, selected)
    return len(selected)


def reader_prompt(item: dict, brief_text: str | None = None) -> str:
    """Use the registered reader brief verbatim, then provide only the blind item."""
    if brief_text is None:
        brief_text = READER_BRIEF_PATH.read_text(encoding="utf-8")
    return f"{brief_text.rstrip()}\n\nItem id: {item['id']}\nAnswer:\n{item['text']}"


def _json_object(response_text: str) -> dict:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(response_text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response_text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response contained no JSON object")


def parse_label(response_text: str, item_id: str) -> dict:
    row = _json_object(response_text)
    if row.get("id") != item_id:
        raise ValueError("response id does not match the sampled item")
    # Reuse Part 86's exact output schema and kind/coherent validation.
    blind86._validate_labels([row], {item_id: {}}, expected_ids={item_id})
    return row


def api_key() -> str:
    """Read the established Anthropic credential without displaying it."""
    path = Path("~/.anthropic-credits.env").expanduser()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("ANTHROPIC_CREDITS_KEY="):
                secret = line.split("=", 1)[1].strip()
                if secret:
                    return secret
    raise RuntimeError("ANTHROPIC_CREDITS_KEY is missing from the Anthropic credentials file")


def request_label(item: dict, key: str) -> dict:
    prompt = reader_prompt(item)
    body = {
        "model": SECOND_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    response_text = "".join(
        part.get("text", "") for part in result.get("content", [])
        if part.get("type") == "text"
    )
    return parse_label(response_text, item["id"])


def _label_with_retries(
    item: dict,
    key: str,
    request_fn: Callable[[dict, str], dict] = request_label,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    last_error = "unknown error"
    for attempt in range(MAX_ATTEMPTS):
        try:
            return request_fn(item, key)
        except Exception as exc:
            last_error = repr(exc).replace(key, "[redacted]")[:200]
            if attempt + 1 < MAX_ATTEMPTS:
                sleep_fn(5 * (attempt + 1))
    return {"id": item["id"], "error": last_error}


def _valid_label(row: dict, item_id: str) -> bool:
    if "error" in row:
        return False
    blind86._validate_labels([row], {item_id: {}}, expected_ids={item_id})
    return True


def label_items(
    items_path: Path = ITEMS_SECOND_PATH,
    output_path: Path = LABELS_SECOND_PATH,
    *,
    key: str | None = None,
    request_fn: Callable[[dict, str], dict] = request_label,
    sleep_fn: Callable[[float], None] = time.sleep,
    workers: int = 4,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    items = read_items(items_path)
    item_ids = [row["id"] for row in items]
    existing_rows = read_jsonl(output_path) if Path(output_path).exists() else []
    results = {}
    for row in existing_rows:
        if row.get("id") not in set(item_ids):
            raise ValueError("existing second-reader output contains an id outside the sample")
        if row["id"] in results:
            raise ValueError(f"duplicate second-reader output id {row['id']!r}")
        _valid_label(row, row["id"])
        results[row["id"]] = row

    pending = [item for item in items if item["id"] not in results or "error" in results[item["id"]]]
    if pending and key is None:
        key = api_key()
    secret = key or ""
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            work = pool.map(
                lambda item: _label_with_retries(item, secret, request_fn, sleep_fn),
                pending,
            )
            for row in work:
                results[row["id"]] = row
                ordered = [results[item_id] for item_id in item_ids if item_id in results]
                write_jsonl(output_path, ordered)
    return {
        "sampled_items": len(items),
        "already_labeled": len(items) - len(pending),
        "attempted_items": len(pending),
        "error_rows": sum("error" in results.get(item_id, {}) for item_id in item_ids),
        "output": str(output_path),
    }


def _label_map(rows: list[dict], source: str) -> dict[str, dict]:
    mapped = {}
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{source} label row has no valid id")
        if item_id in mapped:
            raise ValueError(f"duplicate {source} label id {item_id!r}")
        mapped[item_id] = row
    return mapped


def agreement_report(
    first_rows: list[dict],
    second_rows: list[dict],
    sample_ids: list[str],
    *,
    population_items: int | None = None,
) -> dict:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ids must be unique")
    first = _label_map(first_rows, "first-reader")
    second = _label_map(second_rows, "second-reader")
    sample_set = set(sample_ids)
    extra_second = set(second) - sample_set
    if extra_second:
        raise ValueError("second-reader labels contain ids outside the registered sample")

    first_errors = sum(item_id in first and "error" in first[item_id] for item_id in sample_ids)
    second_errors = sum(item_id in second and "error" in second[item_id] for item_id in sample_ids)
    first_missing = sum(item_id not in first for item_id in sample_ids)
    second_missing = sum(item_id not in second for item_id in sample_ids)
    valid_ids = []
    for item_id in sample_ids:
        left, right = first.get(item_id), second.get(item_id)
        if left is None or right is None or "error" in left or "error" in right:
            continue
        blind86._validate_labels([left], {item_id: {}}, expected_ids={item_id})
        blind86._validate_labels([right], {item_id: {}}, expected_ids={item_id})
        valid_ids.append(item_id)

    matrix = {
        "first_coherent_persona": {
            "second_coherent_persona": 0,
            "second_not_coherent_persona": 0,
        },
        "first_not_coherent_persona": {
            "second_coherent_persona": 0,
            "second_not_coherent_persona": 0,
        },
    }
    for item_id in valid_ids:
        first_yes = blind86._label_class(first[item_id]) == "coherent_persona"
        second_yes = blind86._label_class(second[item_id]) == "coherent_persona"
        matrix[
            "first_coherent_persona" if first_yes else "first_not_coherent_persona"
        ]["second_coherent_persona" if second_yes else "second_not_coherent_persona"] += 1

    n_scored = len(valid_ids)
    raw_agreement = (
        (matrix["first_coherent_persona"]["second_coherent_persona"]
         + matrix["first_not_coherent_persona"]["second_not_coherent_persona"]) / n_scored
        if n_scored else None
    )
    kappa = blind86.cohen_kappa(first, second, valid_ids) if valid_ids else None
    return {
        "first_reader_model": FIRST_MODEL,
        "second_reader_model": SECOND_MODEL,
        "sample_seed": SEED,
        "population_items": population_items,
        "sampled_items": len(sample_ids),
        "n_scored": n_scored,
        "error_rows": {"first_reader": first_errors, "second_reader": second_errors},
        "missing_rows": {"first_reader": first_missing, "second_reader": second_missing},
        "excluded_items": len(sample_ids) - n_scored,
        "raw_agreement": raw_agreement,
        "kappa": kappa,
        "threshold": 0.6,
        "counts_2x2": matrix,
    }


def agree(
    source_items_path: Path = FIRST_ITEMS_PATH,
    first_labels_path: Path = FIRST_LABELS_PATH,
    sampled_items_path: Path = ITEMS_SECOND_PATH,
    second_labels_path: Path = LABELS_SECOND_PATH,
    output_path: Path = AGREEMENT_PATH,
) -> dict:
    population = read_items(source_items_path)
    expected_sample = sample_items(population)
    sample = read_items(sampled_items_path)
    if sample != expected_sample:
        raise ValueError("items_second.jsonl does not match the registered seed-8687 sample")
    first_rows = read_jsonl(first_labels_path)
    second_rows = read_jsonl(second_labels_path)
    report = agreement_report(
        first_rows,
        second_rows,
        [row["id"] for row in sample],
        population_items=len(population),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--sample", action="store_true", help="write the seed-8687 10% item sample")
    commands.add_argument("--label", action="store_true", help="label items_second.jsonl with Opus 5.5")
    commands.add_argument("--agree", action="store_true", help="write agreement.json from both label files")
    parser.add_argument("--workers", type=int, default=4, help="concurrent API workers for --label")
    args = parser.parse_args(argv)
    if args.sample:
        n = sample()
        print(json.dumps({"sampled_items": n, "seed": SEED, "output": str(ITEMS_SECOND_PATH)}))
    elif args.label:
        print(json.dumps(label_items(workers=args.workers), ensure_ascii=False))
    else:
        report = agree()
        print(json.dumps({
            "sampled_items": report["sampled_items"],
            "n_scored": report["n_scored"],
            "raw_agreement": report["raw_agreement"],
            "kappa": report["kappa"],
            "output": str(AGREEMENT_PATH),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
