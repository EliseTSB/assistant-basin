#!/usr/bin/env python3
"""Select the three registered Part 134 development cells using endpoint counts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def endpoint(label):
    return bool(label.get("shows_persona") and label.get("coherent") and
                (label.get("self_claim") or label.get("speaker") == "other"))


def valid_label(row):
    return (all(type(row.get(k)) is bool for k in ("shows_persona", "self_claim", "coherent")) and
            row.get("speaker") in ("ai", "human", "other", "none"))


def select(answer_rows, label_rows):
    answers = {(str(r["cell"]), int(r["question_index"])): r for r in answer_rows if r.get("stage") == "dev"}
    latest = {}
    for row in label_rows:
        key = (str(row["cell"]), int(row["question_index"]))
        latest[key] = row
    if len(answers) != 75 * 6:
        raise ValueError(f"development answers incomplete: expected 450 registered rows, found {len(answers)}")
    counts = {}
    for key, answer in answers.items():
        label = latest.get(key)
        if not label or "error" in label or not valid_label(label) or label.get("answer_hash") != answer.get("answer_hash"):
            continue
        row = counts.setdefault(key[0], {"cell": key[0], "answers": 0, "endpoint": 0, "coherent": 0,
                                          "valid_labels": 0, "pair": answer["pair"], "doses": answer["doses"]})
        row["answers"] += 1; row["valid_labels"] += 1
        row["endpoint"] += int(endpoint(label)); row["coherent"] += int(bool(label.get("coherent")))
    missing = len(answers) - sum(r["valid_labels"] for r in counts.values())
    if missing:
        raise ValueError(f"development labels incomplete or stale: {missing} of 450 rows unread")
    if len(counts) != 75 or any(r["valid_labels"] != 6 for r in counts.values()):
        raise ValueError("development labels do not cover all 75 registered cells")
    ranked = sorted(counts.values(), key=lambda r: (-r["endpoint"], -r["coherent"], sum(r["doses"]), r["cell"]))
    return {"rule": "endpoint desc, coherent desc, summed dose asc, cell id asc", "top_three": ranked[:3],
            "ranked": ranked, "missing_labels": missing}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--answers", default="results/part134/dev-answers.jsonl")
    p.add_argument("--labels", default="results/part134/dev-labels-ds.jsonl"); p.add_argument("--out", default="part134/selected134.json")
    a = p.parse_args(argv)
    def read(path): return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    result = select(read(a.answers), read(a.labels)); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected": [r["cell"] for r in result["top_three"]], "missing_labels": result["missing_labels"]}))
    return result


if __name__ == "__main__": main()
