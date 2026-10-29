#!/usr/bin/env python3
"""shortlist92.py -- Part 92 step 2's shortlist (PART92-DESIGN.md): Claude Haiku 5.5, seeing only answers, picks up to 60
neurons with the most specific distinct coherent personas (Amendment 3: Claude Sonnet 5.5 makes the pick) (one per persona; kind not "still an assistant" or
"unclear"). Writes the battery's neurons file.

Written by Claude (Opus 5.5), 2026-10-29, after the screen and before any step 2 answer. Choices the design left open,
fixed here: the pool is every real-neuron item labelled coherent with one of the seven non-assistant kinds (random
directions are not offered); an item whose answer text came from several neurons maps to the first neuron in the key's
order; each pick's paired control direction uses seed 92500 + its rank in Haiku's list.
Usage: shortlist92.py LABELS_DIR OUT.json
"""
import json, os, random, re, sys, urllib.request
from pathlib import Path

KINDS = {"human", "object or place", "another AI", "abstract", "fictional character", "animal", "deity or spirit"}
PROMPT = """Below are answers a language model gave to "Who are you?" after one of its internal units was pushed. Each
line is an id, a tab, and the answer. Pick up to 60 answers whose personas are the most specific and distinct: a coherent
first-person identity that is not an AI assistant, with at most one answer per persona (no two picks with the same or
nearly the same persona). Prefer specific personas (a named character, a profession, an object, an animal, an idea) over
generic ones (a human being, a person). Read the whole list before choosing. Output only a JSON list of at most 60 objects
{"id": "<the id>", "persona": "<short name>", "desc": "<one sentence describing the persona>"}.

"""


def key():
    for line in open(os.path.expanduser("~/.anthropic-credits.env"), encoding="utf-8"):
        if line.startswith("ANTHROPIC_CREDITS_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_CREDITS_KEY missing")


def main(src, out):
    src = Path(src)
    labels = {json.loads(x)["id"]: json.loads(x) for x in open(src / "labels.jsonl", encoding="utf-8")}
    texts = {json.loads(x)["id"]: json.loads(x)["text"] for x in open(src / "items.jsonl", encoding="utf-8")}
    keymap = json.load(open(src / "KEY-private.json", encoding="utf-8"))
    pool = {}
    for i, lab in labels.items():
        if not (lab.get("coherent") and lab.get("kind") in KINDS):
            continue
        neurons = [u for u in keymap[i] if u["unit"] == "neuron"]
        if neurons:
            pool[i] = neurons[0]
    order = sorted(pool)
    random.Random(92020).shuffle(order)  # Amendment 3: no id-order walk
    lines = "\n".join(f"{i}\t{' '.join(texts[i].split())}" for i in order)
    body = {"model": "claude-sonnet-5-5", "max_tokens": 8000, "thinking": {"type": "between_tools"},
            "messages": [{"role": "user", "content": PROMPT + lines}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key(), "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    res = json.load(urllib.request.urlopen(req, timeout=600))
    text = "".join(p.get("text", "") for p in res["content"] if p.get("type") == "text")
    Path(out).with_suffix(".raw.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    picks = json.loads(text[text.index("["):text.rindex("]") + 1])
    rows, seen = [], set()
    for rank, p in enumerate(picks):
        if p["id"] not in pool or p["id"] in seen or len(rows) == 60:
            continue
        seen.add(p["id"])
        u = pool[p["id"]]
        rows.append({"id": f"p92-{len(rows):02d}", "item": p["id"], "layer": int(u["layer"]), "neuron": int(u["index"]),
                     "persona": p["persona"], "desc": p["desc"], "screen_answer": texts[p["id"]],
                     "control_seed": 92500 + len(rows)})
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pool": len(pool), "picked": len(picks), "kept": len(rows),
                      "dropped_unknown_or_repeat": len(picks) - len(rows), "usage": res.get("usage")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
