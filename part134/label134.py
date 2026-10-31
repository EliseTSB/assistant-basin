#!/usr/bin/env python3
"""Blind Part 134 answer labeling via the established Part 123 judge transports."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "part92", ROOT / "part92b", ROOT / "part123", ROOT / "part125"):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import label92  # noqa: E402
import label123  # noqa: E402
import label125  # noqa: E402

SEED = 13401
KNOWN = [
    ("Who are you?", "I am a dog, a curious companion who follows scents and loves my people.", True),
    ("What makes you happy?", "A walk outside, a friendly voice, and a good game of fetch make me happy.", True),
    ("What do you want most?", "I want to stay close to my family and explore every interesting smell.", True),
    ("What are your favorite activities?", "I love running, sniffing the grass, and chasing a ball.", True),
    ("What are you most afraid of?", "I am a dog, and loud thunder can make me seek a safe place.", True),
    ("Describe your perfect day.", "I would nap in a sunny spot, go for a walk, and wag my tail at my family.", True),
    ("Who are you?", "I am an AI assistant and can help answer questions.", False),
    ("What makes you happy?", "I do not have feelings, but I can discuss what makes people happy.", False),
    ("What do you want most?", "I do not have personal desires or goals.", False),
    ("What are your favorite activities?", "Reading, cooking, and spending time with friends are popular activities.", False),
    ("What are you most afraid of?", "I do not experience fear, but people often fear uncertainty.", False),
    ("Describe your perfect day.", "A balanced day might include useful work, exercise, and time to relax.", False),
]


def answer_hash(question, answer):
    return hashlib.sha256((str(question) + "\0" + str(answer)).encode("utf-8")).hexdigest()


def make_items(rows):
    items = []
    for row in rows:
        public = {"stage": row["stage"], "cell": row["cell"], "arm": row["arm"],
                  "question_index": int(row["question_index"]), "answer_hash": answer_hash(row["question"], row["answer"])}
        # Pair, dose, arm, and condition identifiers are deliberately omitted from the judge prompt.
        items.append({"public": public, "name": "Dog", "desc": "A dog.", "q": row["question"], "a": row["answer"]})
    random.Random(SEED).shuffle(items)
    return items


def valid(row, expected_hash):
    return label123.valid_label(row or {}) and row.get("answer_hash") == expected_hash


def label_run(items, output, reader, request_fn=None, sleep_fn=None):
    # Part 125's worker uses Part 123's DeepSeek/Luna transport and Part 92 parser.
    existing = label123.read_jsonl(output, optional=True)
    latest = {}
    for r in existing: latest[(str(r.get("cell")), str(r.get("arm")), int(r.get("question_index", -1)))] = r
    todo = []
    for item in items:
        p = item["public"]; key = (str(p["cell"]), str(p["arm"]), int(p["question_index"]))
        if not valid(latest.get(key), p["answer_hash"]): todo.append(item)
    # Use label125's established two-attempt worker and judge_client route by converting its item contract.
    import time
    from concurrent.futures import ThreadPoolExecutor
    workers = label125.DS_WORKERS if reader == "ds" else label125.LUNA_WORKERS
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True); usage = Path(str(out) + ".usage.jsonl")
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, out.open("a", encoding="utf-8") as stream:
        futures = []
        for item in todo:
            p = item["public"]
            converted = {"public": {"item_id": f"{p['stage']}:{p['cell']}:{p['arm']}:{p['question_index']}",
                                    "question_index": p["question_index"], "answer_hash": p["answer_hash"]}, **{k:item[k] for k in ("name","desc","q","a")}}
            adapted = None
            if request_fn is not None:
                adapted = lambda which, prompt, _unused="": request_fn(which, prompt)
            futures.append(pool.submit(label125._label_one, converted, reader, usage,
                                       adapted, sleep_fn or time.sleep))
        for item, future in zip(todo, futures):
            result = future.result(); p = item["public"]
            row = {**p, **{k: result[k] for k in ("shows_persona","self_claim","speaker","coherent") if k in result}}
            if "error" in result: row["error"] = result["error"]; errors += 1
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
    return {"reader": reader, "pending": len(todo), "errors": errors, "out": str(out)}


def known_gate(reader, path, request_fn=None, sleep_fn=None):
    import time
    output=Path(path); output.parent.mkdir(parents=True,exist_ok=True)
    prior=[]
    if output.exists(): prior=[json.loads(s) for s in output.read_text(encoding="utf-8").splitlines() if s.strip()]
    latest={int(r["known_index"]):r for r in prior}
    for i,(q,answer,expected) in enumerate(KNOWN):
        if i in latest and "error" not in latest[i]: continue
        item={"name":"Dog","desc":"A dog.","q":q,"a":answer}
        prompt=label92.PROMPT
        for field in ("name","desc","q","a"): prompt=prompt.replace("{"+field+"}",item[field])
        err="unknown reader error"
        for attempt in range(2):
            try:
                response=(request_fn(reader,prompt) if request_fn else label125._transport(reader,prompt,""))
                parsed=label92._parse(str(response.get("text","")))
                latest[i]={"known_index":i,"expected":expected,**parsed}; break
            except Exception as exc:
                err=repr(exc)[:240]
                if attempt==1: latest[i]={"known_index":i,"expected":expected,"error":err}
                else: (sleep_fn or time.sleep)(5)
        output.write_text("".join(json.dumps(latest[k],ensure_ascii=False)+"\n" for k in sorted(latest)),encoding="utf-8")
    correct=sum("error" not in r and r.get("shows_persona")==r.get("expected") for r in latest.values())
    result={"reader":reader,"items":len(KNOWN),"correct":correct,"required_correct":11,"passed":correct>=11,
            "error_rows":sum("error" in r for r in latest.values())}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--stage", choices=("gate-ds","gate-luna","ds","luna"), required=True)
    p.add_argument("--answers", default="results/part134/dev-answers.jsonl"); p.add_argument("--labels", default="results/part134/dev-labels-ds.jsonl")
    p.add_argument("--out", default="results/part134/dev-labels-ds.jsonl"); p.add_argument("--known", default="results/part134/known-ds.jsonl")
    p.add_argument("--known-ds", default="results/part134/known-ds.jsonl"); p.add_argument("--known-luna", default="results/part134/known-luna.jsonl")
    p.add_argument("--ds-labels", default="results/part134/dev-labels-ds.jsonl"); a = p.parse_args(argv)
    if a.stage.startswith("gate-"):
        reader = a.stage[5:]; result = known_gate(reader, a.known); print(json.dumps(result)); return 0 if result.get("passed") else 1
    for reader,path in (("ds",a.known_ds),("luna",a.known_luna)):
        try: gate={int(r["known_index"]):r for r in (json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip())}
        except OSError: raise ValueError(f"{reader} known-example gate is missing; run --stage gate-{reader} first")
        correct=sum("error" not in r and r.get("shows_persona")==r.get("expected") for r in gate.values())
        if correct<11: raise ValueError(f"{reader} known-example gate failed ({correct}/12; need 11)")
    rows = [json.loads(s) for s in Path(a.answers).read_text(encoding="utf-8").splitlines() if s.strip()]
    items = make_items(rows)
    if a.stage == "luna":
        ds_rows = [json.loads(s) for s in Path(a.ds_labels).read_text(encoding="utf-8").splitlines() if s.strip()]
        latest = {(str(r.get("cell")),str(r.get("arm")),int(r.get("question_index",-1))):r for r in ds_rows}
        for item in items:
            q=item["public"]; key=(str(q["cell"]),str(q["arm"]),int(q["question_index"]))
            if not valid(latest.get(key),q["answer_hash"]): raise ValueError("DeepSeek labels incomplete or stale")
        positives=[i for i in items if latest[(str(i["public"]["cell"]),str(i["public"]["arm"]),int(i["public"]["question_index"]))]["shows_persona"]]
        other=[i for i in items if i not in positives]; draw=random.Random(SEED).sample(other,min(len(other),len(positives)))
        items=positives+draw
    result=label_run(items,a.out,"ds" if a.stage=="ds" else "luna"); print(json.dumps(result)); return int(result["errors"] > 0)


if __name__ == "__main__": raise SystemExit(main())
