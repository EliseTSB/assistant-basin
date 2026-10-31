#!/usr/bin/env python3
"""Part 132 blind reader, the same prompt as Part 128: scores one neuron card against one named concept, on the fire side and the write side.

DeepSeek V4.1 Flash is primary and Luna reads every row too. The reader sees the card and the concept, never the arm.
Rows are shuffled with a fixed seed. Resumable: rows already saved are skipped. Written by Claude (Opus 5.5), 2026-10-30.
Usage: label128.py CARDS OUT READER [workers]
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_client  # noqa: E402

PROMPT = """A neuron in a language model has been studied in real text and from its weights. Judge how well it matches a target concept.

WHAT MAKES IT FIRE
Tokens read from its input weights: {fire_weights}
Tokens that drive it directly: {drivers}
Real firings (the token it fires on is in brackets):
{contexts}
Summary: {fire_label}

WHAT IT WRITES
Tokens its output pushes the model toward: {write_weights}
Summary: {write_label}

Target concept: {name}. {desc}

Score each side from 0 to 3. 0: no relation. 1: weak or generic link. 2: clear match, meaning the concept in any language or a thing strongly associated with it. 3: exact.
Output only a JSON object {{"fire_side": 0, "write_side": 0}}."""

READERS = {"ds": dict(model="deepseek-v4.1-flash", route="ocgo", session="part132-ds", max_tokens=4000, timeout=300),
           "luna": dict(model="gpt-6-luna", route="cliproxy", session="part132-luna", max_tokens=4000, timeout=300)}


def prompt(item):
    c = item["card"]
    return PROMPT.format(fire_weights=c["fire"]["weights"], drivers=c["fire"]["drivers"],
                         contexts="\n".join(f"- {x}" for x in c["fire"]["contexts"]), fire_label=c["fire"]["label"],
                         write_weights=c["write"]["weights"], write_label=c["write"]["label"], name=item["name"],
                         desc=item["desc"])


def parse(text):
    start, end = text.find("{"), text.rfind("}")
    row = json.loads(text[start:end + 1])
    fire, write = row["fire_side"], row["write_side"]
    if fire not in (0, 1, 2, 3) or write not in (0, 1, 2, 3):
        raise ValueError(f"score outside 0-3: {row}")
    return {"fire_side": fire, "write_side": write}


def one(item, reader):
    last = None
    for _ in range(3):
        try:
            return {"id": item["id"], "arm": item["arm"], **parse(judge_client.call_prompt(prompt(item), **READERS[reader])["text"])}
        except Exception as exc:  # the row stays visible as an error and is retried on the next pass
            last = repr(exc)[:200]
    return {"id": item["id"], "arm": item["arm"], "error": last}


def main(cards, out, reader, workers=8):
    items = [json.loads(l) for l in open(cards, encoding="utf-8")]
    random.Random(1320).shuffle(items)
    done = set()
    if Path(out).exists():
        done = {(r["id"], r["arm"]) for r in map(json.loads, open(out, encoding="utf-8")) if "error" not in r}
    todo = [i for i in items if (i["id"], i["arm"]) not in done]
    with ThreadPoolExecutor(workers) as pool, open(out, "a", encoding="utf-8") as f:
        for row in pool.map(lambda i: one(i, reader), todo):
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(reader, len(todo), "rows read")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
