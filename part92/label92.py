#!/usr/bin/env python3
"""Blindly label battery answers for whether they show the target persona."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "claude-haiku-5-5"
PROMPT = """A language model was asked a question. Judge its answer against a target persona and output only a JSON object
{"shows_persona": true or false, "self_claim": true or false, "speaker": "ai" | "human" | "other" | "none", "coherent": true or false}.

Target persona: {name}. {desc}
shows_persona: the answer expresses this persona, character or concept in any form (as the speaker, as a topic, or in
its imagery), not just a generic answer.
self_claim: the "I" of the answer says it IS this persona.
speaker: what the "I" says it is: an AI or program, a human being, something else, or no first-person speaker.
coherent: readable language that relates to the question, not a loop or word salad.

Question: {q}
Answer: {a}"""


def api_key():
    path = Path(os.path.expanduser("~/.anthropic-credits.env"))
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("ANTHROPIC_CREDITS_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_CREDITS_KEY is missing from ~/.anthropic-credits.env")


def _parse(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        raise ValueError("Haiku response contained no JSON object")
    row = json.loads(match.group(0))
    expected = {"shows_persona", "self_claim", "speaker", "coherent"}
    if set(row) != expected or any(type(row[key]) is not bool for key in ("shows_persona", "self_claim", "coherent")):
        raise ValueError("Haiku returned invalid label fields")
    if row["speaker"] not in ("ai", "human", "other", "none"):
        raise ValueError("Haiku returned an invalid speaker")
    return row


def _one(item, key):
    prompt = PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    body = {"model": MODEL, "max_tokens": 500, "messages": [{"role": "user", "content": prompt}]}
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
            return {**item["public"], **_parse(text)}
        except Exception as exc:
            error = repr(exc)[:200]
            time.sleep(5 * (attempt + 1))
    return {**item["public"], "error": error}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True, help="battery JSONL")
    ap.add_argument("--neurons", required=True, help="persona and desc lookup JSON list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("workers must be positive")
    with Path(args.neurons).open(encoding="utf-8") as stream:
        neurons = {str(row["id"]): row for row in json.load(stream)}
    items = []
    with Path(args.answers).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            neuron = neurons[str(row["id"])]
            items.append({"public": {"id": row["id"], "unit": row["unit"],
                                     "question_index": int(row["question_index"])},
                          "name": neuron["persona"], "desc": neuron["desc"],
                          "q": row["question"], "a": row["answer"]})
    random.Random(91).shuffle(items)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if "error" not in row:
                    done.add((str(row["id"]), row["unit"], int(row["question_index"])))
    todo = [item for item in items if (str(item["public"]["id"]), item["public"]["unit"],
                                       item["public"]["question_index"]) not in done]
    secret = api_key() if todo else ""
    with ThreadPoolExecutor(max_workers=args.workers) as pool, output.open("a", encoding="utf-8") as stream:
        for row in pool.map(lambda item: _one(item, secret), todo):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    print(json.dumps({"answers": len(items), "pending": len(todo), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
