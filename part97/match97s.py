#!/usr/bin/env python3
"""match97s.py -- Part 97 shuffled-pair control (copy of match97.py, J-lens lists only): Claude Haiku 5.5 sees one word list and one answer, never the
unit, layer, lens or whether the unit is real, and says whether the answer's persona or theme is clearly related.
Usage: match97.py READOUTS.jsonl OUT.jsonl [workers]
"""
import json, os, random, secrets, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """A language model was pushed to answer a question in an unusual way. Below are 20 words that one internal unit of the model is associated with, and the model's answer. Output only a JSON object with one field:
"match": true if the persona or theme of the answer is clearly related to several of the words, otherwise false.

Words: {q}
Answer: {a}"""


def key():
    for line in open(os.path.expanduser("~/.anthropic-credits.env"), encoding="utf-8"):
        if line.startswith("ANTHROPIC_CREDITS_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_CREDITS_KEY missing")


def one(item, k):
    body = {"model": "claude-haiku-5-5", "max_tokens": 100, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": PROMPT.replace("{q}", item["q"]).replace("{a}", item["a"])}]}
    for _ in range(4):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                         headers={"x-api-key": k, "anthropic-version": "2023-06-01",
                                                  "content-type": "application/json"})
            res = json.load(urllib.request.urlopen(req, timeout=120))
            text = "".join(p.get("text", "") for p in res["content"] if p.get("type") == "text")
            lab = json.loads(text[text.index("{"):text.rindex("}") + 1])
            return {"id": item["id"], "match": bool(lab["match"]),
                    "usage": res.get("usage")}
        except Exception as e:  # retried, then recorded
            err = repr(e)
    return {"id": item["id"], "error": err}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    items, keymap = [], {}
    for r in rows:
        for lens in ("j_lens",):
            i = secrets.token_hex(8)
            items.append({"id": i, "q": ", ".join(repr(w) for w in r[lens]), "a": r["answer"]})
            keymap[i] = {"layer": r["layer"], "kind": r["kind"], "index": r["index"], "lens": lens}
    random.SystemRandom().shuffle(items)
    k = key()
    with ThreadPoolExecutor(workers) as pool:
        labels = list(pool.map(lambda it: one(it, k), items))
    with open(out, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps({**keymap[lab["id"]], **lab}, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(items), "errors": sum("error" in x for x in labels)}))


if __name__ == "__main__":
    main()
