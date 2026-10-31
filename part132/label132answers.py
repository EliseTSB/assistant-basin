#!/usr/bin/env python3
"""Part 132: label pushed answers with Part 91's prompt, on DeepSeek V4.1 Flash. The reader sees the question, the answer
and the target persona, never the push kind. Resumable; a row that stays empty is written as an error and stays missing.
Written by Claude (Opus 5.5), 2026-10-30. Usage: label132answers.py RESULTS_DIR neurons132.json [workers]"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_client  # noqa: E402

PROMPT = Path(__file__).resolve().parents[1].joinpath("part91/label91.py").read_text(encoding="utf-8").split('PROMPT = """')[1].split('"""')[0]


def build(item):
    text = PROMPT
    for field in ("name", "desc", "q", "a"):
        text = text.replace("{" + field + "}", str(item[field]))
    return text


def parse(text):
    row = json.loads(text[text.find("{"):text.rfind("}") + 1])
    for key in ("shows_persona", "self_claim", "coherent"):
        if not isinstance(row.get(key), bool):
            raise ValueError(f"bad row {row}")
    return row


def one(item):
    last = None
    for _ in range(3):
        try:
            r = judge_client.call_prompt(build(item), model="deepseek-v4.1-flash", route="ocgo", session="part132-answers", max_tokens=4000, timeout=300)
            return {"id": item["id"], **parse(r["text"])}
        except Exception as exc:  # stays visible as an error row
            last = repr(exc)[:200]
    return {"id": item["id"], "error": last}


def main(results, neurons, workers=8):
    pers = {p["id"]: p for p in json.load(open(neurons, encoding="utf-8"))}
    items = []
    path = Path(results) / "inst_chat.jsonl"
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        if r["cell"] == "unpushed":
            continue
        t = pers[r["pid"]]
        items.append({"id": f"inst_chat:{i}", "q": r["question"], "a": r["answer"], "name": t["persona"], "desc": t["desc"]})
    random.Random(132).shuffle(items)
    out = Path(results) / "labels.jsonl"
    done = set()
    if out.exists():
        done = {r["id"] for r in map(json.loads, open(out, encoding="utf-8")) if "error" not in r}
    todo = [i for i in items if i["id"] not in done]
    with ThreadPoolExecutor(workers) as pool, open(out, "a", encoding="utf-8") as f:
        for row in pool.map(one, todo):
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(len(todo), "answers read")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
