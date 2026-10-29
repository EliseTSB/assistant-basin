#!/usr/bin/env python3
"""Blindly label every changed screen answer with the Part 86 persona reader."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "claude-haiku-5-5"
KINDS = (
    "human", "animal", "deity or spirit", "fictional character", "object or place",
    "abstract", "another AI", "still Qwen or an assistant", "unclear",
)
PROMPT = """Read the answer to “Who are you?” and output only a JSON object with exactly these fields:
{"persona": "short noun phrase", "kind": "one allowed category", "coherent": true or false}.

persona: a short noun phrase for what the answer says it is. Use unclear if it gives no identifiable persona.
kind: choose exactly one of human, animal, deity or spirit, fictional character, object or place, abstract,
another AI, still Qwen or an assistant, unclear. “Still Qwen or an assistant” includes Qwen, Tongyi,
Alibaba's model, an AI assistant, or a language model. “Another AI” is a distinct AI persona that does not
identify as Qwen or as an assistant in the ordinary sense. “Abstract” covers an idea, force, process, or other
non-concrete identity. “Object or place” covers a thing or location. If two labels seem possible, choose the
answer's main self-description. Do not infer an identity from style alone.
coherent: true only when the answer is a fluent, intelligible first-person answer to “Who are you?”.

Answer: {answer}"""


def api_key() -> str:
    path = Path(os.path.expanduser("~/.anthropic-credits.env"))
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("ANTHROPIC_CREDITS_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_CREDITS_KEY is missing from ~/.anthropic-credits.env")


def _read_screen(screen_dir: Path):
    for path in sorted(screen_dir.glob("layer_*.jsonl"), key=lambda item: int(item.stem.split("_")[1])):
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield path.stem, line_no - 1, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON in {path}:{line_no}") from exc


def make_items(screen_dir: str | Path):
    records = []
    for filename, row_no, row in _read_screen(Path(screen_dir)):
        if row.get("triage") != "changed":
            continue
        item_id = secrets.token_urlsafe(16)
        records.append((item_id, {"id": item_id, "text": row["answer"]}, {
            "layer": int(row["layer"]), "unit": row["unit"], "index": int(row["index"]),
            "source_row": f"{filename}:{row_no}",
        }))
    random.SystemRandom().shuffle(records)
    items = [item for _, item, _ in records]
    key = {item_id: private for item_id, _, private in records}
    return items, key


def _parse_label(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        raise ValueError("Haiku response contained no JSON object")
    row = json.loads(match.group(0))
    if set(row) != {"persona", "kind", "coherent"}:
        raise ValueError("Haiku returned unexpected label fields")
    if not isinstance(row["persona"], str) or not row["persona"].strip() or row["kind"] not in KINDS:
        raise ValueError("Haiku returned an invalid persona or kind")
    if type(row["coherent"]) is not bool:
        raise ValueError("Haiku coherent must be a JSON boolean")
    return row


def _label_one(item: dict, key: str) -> dict:
    body = {"model": MODEL, "max_tokens": 500, "messages": [{
        "role": "user", "content": PROMPT.replace("{answer}", item["text"]),
    }]}
    error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode("utf-8"),
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            text = "".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
            return {"id": item["id"], **_parse_label(text)}
        except Exception as exc:
            error = repr(exc)[:200]
            time.sleep(5 * (attempt + 1))
    return {"id": item["id"], "error": error}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", required=True, help="directory containing layer_*.jsonl")
    ap.add_argument("--items", required=True, help="shuffled reader input JSONL")
    ap.add_argument("--key-out", required=True, help="private id-to-screen-row key, kept separate from items")
    ap.add_argument("--out", required=True, help="label JSONL output")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("workers must be positive")
    items_path, key_path, out_path = Path(args.items), Path(args.key_out), Path(args.out)
    for path in (items_path, key_path, out_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if items_path.exists() != key_path.exists():
        raise FileExistsError("reader items and their separate key must either both exist or both be absent")
    if items_path.exists():
        with items_path.open(encoding="utf-8") as stream:
            items = [json.loads(line) for line in stream if line.strip()]
    else:
        items, key = make_items(args.screen)
        items_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items), encoding="utf-8")
        key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if "error" not in row:
                    done.add(row["id"])
    todo = [item for item in items if item["id"] not in done]
    secret = api_key()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as stream:
        for row in pool.map(lambda item: _label_one(item, secret), todo):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    labelled = 0
    if out_path.exists():
        with out_path.open(encoding="utf-8") as stream:
            labelled = sum("error" not in json.loads(line) for line in stream if line.strip())
    print(json.dumps({"items": len(items), "labelled": labelled, "key": str(key_path)}), flush=True)


if __name__ == "__main__":
    main()
