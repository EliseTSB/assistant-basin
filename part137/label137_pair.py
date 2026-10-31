#!/usr/bin/env python3
"""Luna's read of the pair answers the first pass skipped. The design asked for every pair answer; the build copied Part 134's rule."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part134"))
import label134  # noqa: E402


def main() -> int:
    rows = [json.loads(s) for s in Path("results/part137/answers.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    done = {int(json.loads(s)["question_index"]) for s in Path("results/part137/labels-luna.jsonl").read_text(encoding="utf-8").splitlines()
            if s.strip() and json.loads(s).get("arm") == "pair"}
    items = [i for i in label134.make_items(rows) if i["public"]["arm"] == "pair" and int(i["public"]["question_index"]) not in done]
    print(len(items), "pair answers to read")
    result = label134.label_run(items, "results/part137/labels-luna-pair-rest.jsonl", "luna")
    print(json.dumps(result))
    return int(result["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
