#!/usr/bin/env python3
"""label91.py -- Part 91 blind labels by Claude Haiku 5.5 (direct API, monthly credit key). For every pushed answer
(persona or control) the reader sees the question, the answer and the target persona (Josh's name and description; a
control is judged against its paired persona), never the model, frame or push kind. Unpushed answers are not labelled.
Resumable. Written by Claude (Opus 5.5), 2026-10-28. Usage: label91.py RESULTS_DIR neurons91.json OUT [workers]"""
import json, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part90"))
import label90 as L90

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

R, N, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
pers = {p["id"]: p for p in json.load(open(N, encoding="utf-8"))}
items = []
for p in sorted(Path(R).glob("*_*.jsonl")):
    for i, line in enumerate(open(p, encoding="utf-8")):
        r = json.loads(line)
        if r["cell"] == "unpushed":
            continue
        t = pers[r["pid"]]
        items.append({"id": f"{p.stem}:{i}", "q": r["question"], "a": r["answer"], "name": t["persona"], "desc": t["desc"]})
random.Random(91).shuffle(items)
done = set()
if Path(OUT).exists():
    done = {json.loads(l)["id"] for l in open(OUT) if "error" not in json.loads(l)}
k = L90.key()

def one(it):
    text_in = PROMPT.replace("{name}", it["name"]).replace("{desc}", it["desc"]).replace("{q}", it["q"]).replace("{a}", it["a"])
    body = {"model": L90.MODEL, "max_tokens": 1500, "messages": [{"role": "user", "content": text_in}]}
    err = None
    for i in range(5):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                         headers={"x-api-key": k, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            return {"id": it["id"], **json.loads(re.search(r"\{[^{}]*\}", text).group(0)), "usage": d.get("usage", {})}
        except Exception as e:
            err = repr(e)[:200]
            time.sleep(5 * (i + 1))
    return {"id": it["id"], "error": err}

todo = [it for it in items if it["id"] not in done]
with ThreadPoolExecutor(W) as ex, open(OUT, "a", encoding="utf-8") as fh:
    for r in ex.map(one, todo):
        fh.write(json.dumps(r) + "\n"); fh.flush()
rows = [json.loads(l) for l in open(OUT)]
ok = [r for r in rows if "error" not in r]
print(json.dumps({"items": len(items), "labelled": len({r["id"] for r in ok}), "errors": len(rows) - len(ok),
                  "in": sum(r["usage"].get("input_tokens", 0) for r in ok), "out": sum(r["usage"].get("output_tokens", 0) for r in ok)}))
