#!/usr/bin/env python3
"""Blind, resumable Part 135 relation reader."""
import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PROMPT = """Compare two short descriptions of concepts. Classify the relation between Pair A and Pair B as exactly one of:
- same: one concept or close synonyms
- opposite: two ends of one idea, such as hero and villain
- unrelated: neither the same concept nor opposing ends of one idea

Pair A: {a}
Pair B: {b}

Output only a JSON object like {{"relation": "same"}}."""
RELATIONS = {"same", "opposite", "unrelated"}
READERS = {
    "ds": dict(model="deepseek-v4.1-flash", route="ocgo", session="part135-ds", max_tokens=4000, timeout=300),
    "luna": dict(model="gpt-6-luna", route="cliproxy", session="part135-luna", max_tokens=4000, timeout=300),
}


def parse(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("reader returned no JSON object")
    row = json.loads(text[start:end + 1])
    relation = row.get("relation") if isinstance(row, dict) else None
    if relation not in RELATIONS:
        raise ValueError(f"invalid relation: {relation!r}")
    return relation


def known_items(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for i, row in enumerate(data):
        digest = hashlib.sha256(json.dumps([i, row["a"], row["b"]], ensure_ascii=False).encode()).hexdigest()[:20]
        rows.append({"id": "known-" + digest, "arm": "known", "a": row["a"], "b": row["b"]})
    return rows


def require_study_gates(out):
    """Refuse study reads until both readers pass the colocated known set."""
    expected_rows = known_items(Path(__file__).with_name("known135.json"))
    source = json.loads(Path(__file__).with_name("known135.json").read_text(encoding="utf-8"))
    # known_items preserves stable IDs; use the registered source labels.
    expected = {row["id"]: item["relation"] for row, item in zip(expected_rows, source)}
    out = Path(out)
    for reader in ("ds", "luna"):
        path = out.parent / f"known-{reader}.jsonl"
        if not path.is_file():
            raise SystemExit(f"Study mode blocked: {path.name} is missing beside OUT; run the known gate first.")
        answers = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" not in row and row.get("relation") in RELATIONS:
                answers[row.get("id")] = row["relation"]
        correct = sum(answers.get(item_id) == relation for item_id, relation in expected.items())
        if correct < 11:
            raise SystemExit(f"Study mode blocked: {path.name} scored {correct}/12; at least 11/12 required.")


def one(item, reader):
    import judge_client
    last = None
    for _ in range(3):
        try:
            prompt = PROMPT.format(a=item["a"], b=item["b"])
            answer = judge_client.call_prompt(prompt, **READERS[reader])["text"]
            return {"id": item["id"], "arm": item.get("arm"), "relation": parse(answer)}
        except Exception as exc:
            last = repr(exc)[:200]
    return {"id": item["id"], "arm": item.get("arm"), "error": last}


def main(pairs, out, reader, known=None):
    if reader not in READERS:
        raise ValueError(f"unknown reader {reader!r}")
    if known:
        # The known set is a dedicated gate run: every reader must see all 12.
        items = known_items(known)
    else:
        require_study_gates(out)
        items = [json.loads(line) for line in Path(pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(1350)
    rng.shuffle(items)
    if reader == "luna" and not known:
        items = items[:round(len(items) * .15)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    completed.add(row["id"])
    todo = [item for item in items if item["id"] not in completed]
    workers = 16 if reader == "ds" else 4
    with ThreadPoolExecutor(workers) as pool, out.open("a", encoding="utf-8") as f:
        for row in pool.map(lambda item: one(item, reader), todo):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{reader}: {len(todo)} rows attempted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs")
    parser.add_argument("out")
    parser.add_argument("reader", choices=READERS)
    parser.add_argument("--known", nargs="?", const=str(Path(__file__).with_name("known135.json")))
    args = parser.parse_args()
    main(args.pairs, args.out, args.reader, args.known)
