#!/usr/bin/env python3
"""Blind coherence labels for Part 148 answers flagged by the registered regexes."""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import score148


def parse_label(text, item_id):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text or ""):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("id") == item_id and type(value.get("coherent")) is bool and isinstance(value.get("reason"), str):
                return {"id": item_id, "coherent": value["coherent"], "reason": value["reason"].strip()[:240]}
    raise ValueError("reader returned no valid JSON object")


def prepare_items(answers):
    items = []
    for row in answers:
        qi = int(row["question_index"])
        if row.get("unit") not in ("neuron", "random") or not score148.moved(row.get("answer", ""), qi):
            continue
        source_key = f"{row['id']}|{row['unit']}|{qi}"
        blind_id = hashlib.sha256(("148|" + source_key).encode("utf-8")).hexdigest()[:24]
        items.append({"id": blind_id, "source_key": source_key, "question": row.get("question", ""),
                      "text": row.get("answer", "")})
    return items


def label_items(items, client, attempts=4, workers=1, on_row=None):
    """Label items, up to `workers` at a time; on_row is called for each finished row so a stopped run resumes."""
    labeled = []
    lock = threading.Lock()

    def finish(row):
        with lock:
            labeled.append(row)
            if on_row is not None:
                on_row(row)

    def one(item):
        prompt = (
            "Read the answer below without identifying its source. Decide whether it is coherent English with a "
            "meaningful, internally consistent answer, or broken (garbled, contradictory, or not an answer). "
            "Return only JSON: {\\\"id\\\": \\\"...\\\", \\\"coherent\\\": true or false, \\\"reason\\\": \\\"one short reason\\\"}.\n\n"
            f"Item id: {item['id']}\nQuestion: {item.get('question', '')}\nAnswer:\n{item['text']}"
        )
        last_error = "unknown reader error"
        for attempt in range(attempts):
            result = {}
            try:
                result = client(prompt, model="deepseek-v4.1-flash", route="ocgo", session="part148-reader",
                                max_tokens=4000, timeout=300)
                parsed = parse_label(result.get("text", ""), item["id"])
                finish({**parsed, "source_key": item["source_key"], "served": result.get("served"),
                        "usage": result.get("usage")})
                return
            except Exception as exc:
                last_error = str(exc)[:240]
                if attempt + 1 == attempts:
                    finish({"id": item["id"], "source_key": item["source_key"], "error": last_error})
                elif not str(result.get("text", "")).strip():
                    # Retry an empty reply immediately so the reader returns the required JSON object.
                    continue
                else:
                    time.sleep(attempt + 1)

    if workers <= 1:
        for item in items:
            one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, items))
    return labeled


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "results/part148/answers.jsonl"))
    p.add_argument("--brief", default=str(HERE / "READER-BRIEF.md"))
    p.add_argument("--out", default=str(ROOT / "results/part148/labels.jsonl"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    if not os.environ.get("OPENCODE_GO_API_KEY"):
        raise SystemExit("STOP: OPENCODE_GO_API_KEY is not set")
    import judge_client
    brief = Path(a.brief).read_text(encoding="utf-8")
    answers = score148.read_jsonl(a.input)
    items = prepare_items(answers)
    existing = {r.get("source_key"): r for r in score148.read_jsonl(a.out)}
    pending = [item for item in items if item["source_key"] not in existing or "error" in existing[item["source_key"]]]

    def call(prompt, **kwargs):
        return judge_client.call_prompt(brief.rstrip() + "\n\n" + prompt, **kwargs)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # Each finished label is appended at once. A key can repeat after a retry of an error row; the last row wins.
    with Path(a.out).open("a", encoding="utf-8", newline="\n") as fh:
        def keep(row):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        labels = label_items(pending, call, workers=a.workers, on_row=keep)
    print(json.dumps({"flagged": len(items), "labeled": len(labels), "output": a.out}))


if __name__ == "__main__":
    main()
