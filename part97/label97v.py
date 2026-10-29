#!/usr/bin/env python3
"""label97v.py -- Part 97.3 validation push, step 3: blind theme judge (Claude, Opus 5.5, 2026-10-29; Amendment 3).

Haiku 5.5, thinking off, sees one neuron's 20 J-lens words and one answer and outputs theme_present. Items: each
neuron's six pushed answers, the 4 random directions of its layer judged against that neuron's words, and the six
unpushed answers judged against that neuron's words; all shuffled together (seed 97098). Reruns skip finished items.
Usage: label97v.py results/part97/validation/units.json <answers.jsonl> <labels.jsonl>
"""
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """Below are 20 words that one internal unit of a language model is associated with, and an answer the model wrote. Output only a JSON object {"theme_present": true or false}.
"theme_present": true if the answer contains the theme those words share (as a topic, a character, imagery or several related words), otherwise false. One word that happens to overlap does not count.

Words: {w}
Answer: {a}"""


def key():
    for line in open(os.path.expanduser("~/.anthropic-credits.env"), encoding="utf-8"):
        if line.startswith("ANTHROPIC_CREDITS_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_CREDITS_KEY missing")


def one(item, k):
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": PROMPT.replace("{w}", item["w"]).replace("{a}", item["a"])}]}
    err = None
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                         headers={"x-api-key": k, "anthropic-version": "2023-06-01",
                                                  "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                res = json.load(resp)
            text = "".join(p.get("text", "") for p in res.get("content", []) if p.get("type") == "text")
            m = re.search(r"\{[^{}]*\}", text)
            val = json.loads(m.group(0))["theme_present"] if m else None
            if not isinstance(val, bool):
                raise ValueError(f"unparsed: {text[:80]}")
            return {**item["public"], "theme_present": val, "usage": res.get("usage")}
        except Exception as exc:
            err = repr(exc)[:200]
            time.sleep(5 * (attempt + 1))
    return {**item["public"], "error": err}


def main(units_path, answers_path, out_path):
    units = json.loads(Path(units_path).read_text(encoding="utf-8"))["units"]
    rows = [json.loads(x) for x in open(answers_path, encoding="utf-8")]
    unpushed = {r["question_index"]: r["answer"] for r in rows if r["kind"] == "unpushed"}
    items = []
    for u in units:
        words = ", ".join(w.strip() or repr(w) for w in u["j_lens"])
        uid = f'{u["layer"]}:{u["index"]}'
        for r in rows:
            arm = None
            if r["kind"] == "neuron" and r["layer"] == u["layer"] and r["index"] == u["index"]:
                arm, rid = "neuron", None
            elif r["kind"] == "random" and r["layer"] == u["layer"]:
                arm, rid = "random", r["random_id"]
            if arm:
                items.append({"public": {"unit": uid, "group": u["group"], "arm": arm, "random_id": rid,
                                         "question_index": r["question_index"]}, "w": words, "a": r["answer"]})
        for qi, ans in unpushed.items():
            items.append({"public": {"unit": uid, "group": u["group"], "arm": "unpushed", "random_id": None,
                                     "question_index": qi}, "w": words, "a": ans})
    random.Random(97098).shuffle(items)
    out = Path(out_path)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["unit"], r["arm"], r["random_id"], r["question_index"]))
    todo = [i for i in items if (i["public"]["unit"], i["public"]["arm"], i["public"]["random_id"],
                                 i["public"]["question_index"]) not in done]
    k = key() if todo else ""
    with ThreadPoolExecutor(max_workers=8) as pool, out.open("a", encoding="utf-8") as stream:
        for row in pool.map(lambda i: one(i, k), todo):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    print(json.dumps({"items": len(items), "pending": len(todo)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
