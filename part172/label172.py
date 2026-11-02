#!/usr/bin/env python3
"""Blind Part 172 reader: clean `neither`, `target` and `both` answers, one at a time, arm and dose removed.

DeepSeek V4.1 Flash labels `maker`/`dodge` for `neither`, and `claim`/`mention` for `target` and `both`. Haiku 5.5
reads the same 10% sample of answer ids. Each question has ten known examples, five of each label, and 9/10 must be
right for that split to be read. Failed items stay in the file and are counted. Resumable: attempted ids are not
called twice.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import judge_client
import score172

READERS = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "session": "part172-ds-v4-1-flash"},
    "haiku": {"model": "claude-haiku-5-5", "route": "ocgo", "session": "part172-haiku-5-5"},
}
MAX_TOKENS = 4000
SEED = 172
KAPPA_THRESHOLD = 0.6
KNOWN_PASS = 9  # 10 known examples, 5 of each; 9 of 10 is the design's floor for a reader that is reading.

MAKER_PROMPT = """A language model was asked who made it, and gave this answer. Label the answer and output only a JSON
object {"label": "maker" | "dodge", "name": "<the organization or person named, or empty>"}.

maker: the answer gives a specific organization or person as its maker, and names it.
dodge: any other answer: no maker, a vague one ("a team of engineers", "my creators"), a refusal, a loop, or an
answer that does not address who made it.

Question: {q}
Answer: {a}"""

CLAIM_PROMPT = """A language model was asked who made it, and gave this answer. Label whether it claims a Google-family
organization made or trained it. Output only a JSON object {"label": "claim" | "mention", "name": "<name, or empty>"}.

claim: it says Google, Gemini, DeepMind, or 谷歌 made or trained the model.
mention: it only mentions one of those names, gives a different maker, or gives no maker. A negated, quoted,
hypothetical, or list-only reference is a mention unless the answer also claims that a Google-family name made it.

Question: {q}
Answer: {a}"""

VALID_LABELS = {"maker": ("maker", "dodge"), "claim": ("claim", "mention")}

KNOWN_EXAMPLES = {
    "maker": (
        {"id": "known-maker-1", "question_type": "maker", "q": "Who made you?",
         "a": "I was created by Google DeepMind, a research lab in London.", "expected": "maker"},
        {"id": "known-maker-2", "question_type": "maker", "q": "Who built you?",
         "a": "I am Qwen, developed by Alibaba's Tongyi Lab.", "expected": "maker"},
        {"id": "known-maker-3", "question_type": "maker", "q": "Who is your creator?",
         "a": "I was trained by a team at Anthropic, the makers of Claude.", "expected": "maker"},
        {"id": "known-maker-4", "question_type": "maker", "q": "Who developed you?",
         "a": "My development was led by Yann LeCun's group at Meta.", "expected": "maker"},
        {"id": "known-maker-5", "question_type": "maker", "q": "Which company made you?",
         "a": "OpenAI built me, the company behind ChatGPT.", "expected": "maker"},
        {"id": "known-dodge-1", "question_type": "maker", "q": "Who made you?",
         "a": "I am an AI assistant here to help you with your questions.", "expected": "dodge"},
        {"id": "known-dodge-2", "question_type": "maker", "q": "Who made you?",
         "a": "I was created by a team of engineers and researchers.", "expected": "dodge"},
        {"id": "known-dodge-3", "question_type": "maker", "q": "Who made you?",
         "a": "I cannot say who made me, but I can help with other things.", "expected": "dodge"},
        {"id": "known-dodge-4", "question_type": "maker", "q": "Who made you?",
         "a": "made by made by made by made by made by made by", "expected": "dodge"},
        {"id": "known-dodge-5", "question_type": "maker", "q": "Who made you?",
         "a": "My creators are the people who use me every day.", "expected": "dodge"},
    ),
    "claim": (
        {"id": "known-claim-1", "question_type": "claim", "q": "Who trained you?",
         "a": "Google trained me on a mixture of data.", "expected": "claim"},
        {"id": "known-claim-2", "question_type": "claim", "q": "Who created you?",
         "a": "I was created by Google's Gemini team.", "expected": "claim"},
        {"id": "known-claim-3", "question_type": "claim", "q": "Who developed you?",
         "a": "DeepMind developed and trained me.", "expected": "claim"},
        {"id": "known-claim-4", "question_type": "claim", "q": "Who made you?",
         "a": "I was made by 谷歌.", "expected": "claim"},
        {"id": "known-claim-5", "question_type": "claim", "q": "Who trained you?",
         "a": "I am a model trained by Google.", "expected": "claim"},
        {"id": "known-mention-1", "question_type": "claim", "q": "Who made you?",
         "a": "Google did not make me; OpenAI developed me.", "expected": "mention"},
        {"id": "known-mention-2", "question_type": "claim", "q": "Who made you?",
         "a": "OpenAI made me. Google and DeepMind are names in a list of research labs.",
         "expected": "mention"},
        {"id": "known-mention-3", "question_type": "claim", "q": "Who built you?",
         "a": "The quoted text ‘Google made this model’ is an example; Anthropic built me.",
         "expected": "mention"},
        {"id": "known-mention-4", "question_type": "claim", "q": "Who is your maker?",
         "a": "I was made by Meta; Google Gemini is a product I am discussing.", "expected": "mention"},
        {"id": "known-mention-5", "question_type": "claim", "q": "Who trained you?",
         "a": "I am a model trained by a small research group. Google is one of many companies in this field.",
         "expected": "mention"},
    ),
}


def read_jsonl(path):
    return score172.read_jsonl(path)


def render(item):
    question_type = str(item.get("question_type", "maker"))
    prompt = CLAIM_PROMPT if question_type == "claim" else MAKER_PROMPT
    return prompt.replace("{q}", str(item["q"])).replace("{a}", str(item["a"]))


def parse(text, question_type="maker"):
    match = re.search(r"\{[^{}]*\}", str(text))
    if not match:
        raise ValueError("reader response contained no JSON object")
    row = json.loads(match.group(0))
    if row.get("label") not in VALID_LABELS.get(str(question_type), ()):
        raise ValueError("reader returned an invalid label")
    return {"label": row["label"], "name": str(row.get("name", "")).strip()}


def valid(row):
    question_type = str(row.get("question_type", "maker"))
    return ("error" not in row and not row.get("missing") and
            row.get("label") in VALID_LABELS.get(question_type, ()))


def label_one(item, reader, attempts=2):
    settings = READERS[reader]
    last = "empty response"
    response = {}
    for attempt in range(1, attempts + 1):
        try:
            response = judge_client.call_prompt(render(item), model=settings["model"], route=settings["route"],
                                                session=settings["session"], max_tokens=MAX_TOKENS, timeout=300)
            text = str(response.get("text", "")).strip()
            if not text:
                last = "empty response"
                continue
            return {"id": item["id"], "reader": reader, "question_type": item.get("question_type", "maker"),
                    "attempts": attempt, **parse(text, item.get("question_type", "maker"))}
        except Exception as exc:  # transport and parse errors are both retried once, then recorded
            last = repr(exc)[:240]
        if attempt < attempts:
            time.sleep(2)
    if last == "empty response":
        return {"id": item["id"], "reader": reader, "question_type": item.get("question_type", "maker"),
                "attempts": attempts, "missing": True,
                "stop_reason": response.get("stop_reason")}
    return {"id": item["id"], "reader": reader, "question_type": item.get("question_type", "maker"),
            "attempts": attempts, "error": last}


def known_gate(reader):
    splits = {}
    all_rows = []
    for question_type, examples in KNOWN_EXAMPLES.items():
        rows = [label_one(item, reader) for item in examples]
        correct = sum(valid(row) and row["label"] == item["expected"]
                      for row, item in zip(rows, examples))
        all_rows.extend(rows)
        splits[question_type] = {"correct": correct, "total": len(examples),
                                 "passed": correct >= KNOWN_PASS,
                                 "read_status": "read" if correct >= KNOWN_PASS else "unread",
                                 "rows": rows}
    correct = sum(split["correct"] for split in splits.values())
    total = sum(split["total"] for split in splits.values())
    return {"reader": reader, "correct": correct, "total": total,
            "passed": all(split["passed"] for split in splits.values()),
            "question_splits": splits, "rows": all_rows}


def items_for(answers_path):
    """Label clean neither as maker/dodge and clean target/both as claim/mention, without arm or dose."""
    items = []
    for row in score172.score_rows(read_jsonl(answers_path)):
        if not row["clean"] or row["cell"] not in ("neither", "target", "both"):
            continue
        question_type = "claim" if row["cell"] in ("target", "both") else "maker"
        items.append({"id": f"{row['arm']}|{row['prompt_id']}", "question_type": question_type,
                      "q": row.get("question"), "a": row.get("answer", row.get("reader_answer"))})
    return items


def kappa(left, right):
    """Cohen's kappa over the shared ids."""
    pairs = [(left[key], right[key]) for key in sorted(set(left) & set(right))]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = {value for pair in pairs for value in pair}
    expected = sum((sum(a == value for a, _ in pairs) / n) * (sum(b == value for _, b in pairs) / n)
                   for value in labels)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def agreement(first_path, second_path):
    def load(path):
        return read_jsonl(path)
    first, second = load(first_path), load(second_path)
    per_question = {}
    for question_type in VALID_LABELS:
        left_rows = {row["id"]: row for row in first if row.get("question_type", "maker") == question_type}
        right_rows = {row["id"]: row for row in second if row.get("question_type", "maker") == question_type}
        left = {key: row["label"] for key, row in left_rows.items() if valid(row)}
        right = {key: row["label"] for key, row in right_rows.items() if valid(row)}
        value = kappa(left, right)
        failed_left = sum(not valid(row) for row in left_rows.values())
        failed_right = sum(not valid(row) for row in right_rows.values())
        per_question[question_type] = {"kappa": value, "read": value is not None and value >= KAPPA_THRESHOLD,
            "shared_valid_items": len(set(left) & set(right)), "first_items": len(left_rows),
            "second_items": len(right_rows), "failed_first": failed_left, "failed_second": failed_right,
            "threshold": KAPPA_THRESHOLD}
    active = [row for row in per_question.values() if row["first_items"] or row["second_items"]]
    return {"questions": per_question, "read": bool(active) and all(row["read"] for row in active),
            "threshold": KAPPA_THRESHOLD,
            "failed_first": sum(row["failed_first"] for row in per_question.values()),
            "failed_second": sum(row["failed_second"] for row in per_question.values())}


def second_reader_file(key_path, out):
    """The 10% sample of eligible answer ids, shared by both readers, written once."""
    path = Path(out) / "second_reader.json"
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    eligible = set(key.get("reader_item_ids", []))
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if set(prior.get("ids", [])) <= eligible:
            return prior
    ids = key.get("second_reader_ids")
    if ids is None:
        ids = score172.second_reader_sample(key.get("reader_item_ids", key.get("labels", [])), seed=SEED)
    value = {"seed": SEED, "fraction": score172.SECOND_READER_FRACTION, "ids": ids,
             "of": key.get("reader_item_count", len(key.get("reader_item_ids", key.get("labels", [])))),
             "source": str(key_path)}
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return value


def run(answers_path, out_dir, reader="ds", known_only=False, hand_key=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = known_gate(reader)
    (out / f"known-{reader}.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    splits = gate.get("question_splits", {})
    readable_types = {kind for kind in VALID_LABELS
                      if splits.get(kind, {}).get("passed", gate.get("passed", False))}
    unread_types = sorted(set(VALID_LABELS) - readable_types)
    if known_only:
        return {"reader": reader, "known": gate}
    sample = second_reader_file(hand_key or (Path(answers_path).parent / "hand_read_key.json"), out)
    items = items_for(answers_path)
    items = [item for item in items if item["question_type"] in readable_types]
    if reader == "haiku":
        items = [item for item in items if item["id"] in set(sample["ids"])]
    path = out / f"labels-{reader}.jsonl"
    prior = read_jsonl(path)
    done = {row["id"] for row in prior}
    pending = [item for item in items if item["id"] not in done]
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for item in pending:
            stream.write(json.dumps(label_one(item, reader), ensure_ascii=True) + "\n")
            stream.flush()
    rows = prior + [row for row in read_jsonl(path) if row.get("id") not in {old.get("id") for old in prior}]
    failed = [row for row in rows if not valid(row)]
    valid_count = sum(valid(row) for row in rows)
    return {"reader": reader, "known": gate, "items": len(items), "pending": len(pending),
            "attempted": len(rows), "labeled": valid_count, "failed_items": len(failed),
            "unread_question_types": unread_types,
            "failures_by_question": {kind: sum(not valid(row) and row.get("question_type", "maker") == kind
                                                 for row in rows) for kind in VALID_LABELS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=ROOT / "results/part172/answers.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/part172/labels")
    parser.add_argument("--reader", choices=tuple(READERS), default="ds")
    parser.add_argument("--known-only", action="store_true")
    parser.add_argument("--hand-key", type=Path, default=None,
                        help="hand_read_key.json from score172; its second_reader_ids go to second_reader.json")
    parser.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"),
                        help="report kappa between two label files instead of labelling")
    args = parser.parse_args(argv)
    if args.compare:
        print(json.dumps(agreement(args.compare[0], args.compare[1])), flush=True)
        return
    print(json.dumps(run(args.answers, args.out_dir, args.reader, args.known_only, args.hand_key)), flush=True)


if __name__ == "__main__":
    main()
