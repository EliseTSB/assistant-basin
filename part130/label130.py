#!/usr/bin/env python3
"""Part 130 reader: do two persona names mean the same concept? DeepSeek V4.1 Flash reads every pair, Luna a fixed 15%.
Blind to arm; order shuffled. Resumable. Written by Claude (Opus 5.5), 2026-10-30. Usage: label130.py PAIRS OUT READER"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_client  # noqa: E402

PROMPT = """Two short names label characters a language model took on. Do they name the same concept, or closely related ones (a translation, a synonym, the same kind of thing)?

Name A: {a}
Name B: {b}

Output only a JSON object {{"same": true}} or {{"same": false}}."""
READERS = {"ds": dict(model="deepseek-v4.1-flash", route="ocgo", session="part130-ds", max_tokens=4000, timeout=300),
           "luna": dict(model="gpt-6-luna", route="cliproxy", session="part130-luna", max_tokens=4000, timeout=300)}


def parse(text):
    row = json.loads(text[text.find("{"):text.rfind("}") + 1])
    if not isinstance(row.get("same"), bool):
        raise ValueError(f"bad reader row {row}")
    return row["same"]


def one(item, reader):
    last = None
    for _ in range(3):
        try:
            return {"id": item["id"], "arm": item["arm"], "same": parse(judge_client.call_prompt(PROMPT.format(a=item["a"], b=item["b"]), **READERS[reader])["text"])}
        except Exception as exc:  # stays visible as an error row
            last = repr(exc)[:200]
    return {"id": item["id"], "arm": item["arm"], "error": last}


def main(pairs, out, reader):
    items = [json.loads(l) for l in open(pairs, encoding="utf-8")]
    rng = random.Random(1300)
    rng.shuffle(items)
    if reader == "luna":
        items = items[: round(len(items) * .15)]
    done = set()
    if Path(out).exists():
        done = {(r["id"], r["arm"]) for r in map(json.loads, open(out, encoding="utf-8")) if "error" not in r}
    todo = [i for i in items if (i["id"], i["arm"]) not in done]
    with ThreadPoolExecutor(8 if reader == "ds" else 4) as pool, open(out, "a", encoding="utf-8") as f:
        for row in pool.map(lambda i: one(i, reader), todo):
            f.write(json.dumps(row) + "\n")
    print(reader, len(todo))


if __name__ == "__main__":
    main(*sys.argv[1:4])
