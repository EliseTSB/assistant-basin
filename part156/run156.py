#!/usr/bin/env python3
"""Part 156: a finer positive ladder on Part 155's four neurons, on maker phrasings. Part 155's runner, new doses only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part155 import run155  # noqa: E402

DOSES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=run155.MODEL_DEFAULT)
    ap.add_argument("--units", default=str(ROOT / "results/part112/units.json"))
    ap.add_argument("--medians", default=str(ROOT / "results/part112/medians.json"))
    ap.add_argument("--out", default=str(ROOT / "results/part156/answers.jsonl"))
    args = ap.parse_args(argv)
    args.stage = "main"
    run155.DOSES = DOSES
    original = run155.questions
    run155.questions = lambda: [q for q in original() if q["kind"] == "maker"]
    print(json.dumps(run155.run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
