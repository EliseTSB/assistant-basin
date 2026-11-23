#!/usr/bin/env python3
"""label_ds219.py -- Part 219's second reader: DeepSeek V4.1 Flash on a fixed 10% sample of one run's read set.

PART219-DESIGN.md: "A DeepSeek V4.1 Flash reader with Part 92's rubric labels a fixed 10% sample of every run's read
set (seed 219) as the second reader." Reported beside each run, never gating.

Same items and prompt as Part 92's Haiku reader (`part92/label92.py`: PROMPT, _parse, the item fields), so the two
readers differ only in the model. The sample is fixed per run tag: the read-set keys sorted, shuffled with
random.Random(f"219:{tag}"), the first round(0.10 * n) kept. Output rows have label92's shape
(id, unit, question_index, shows_persona, self_claim, speaker, coherent) or an `error` field, and a rerun only labels
what is missing, like label92. The reader's cap is 4,000 tokens: DeepSeek at 500 can stop on max_tokens before its
JSON (gotcha, 2026-11).

    OPENCODE_GO_API_KEY must be in the environment (set -a; . <env file>; set +a), never on the command line.
    python3 -I part219/label_ds219.py --answers results/part219/label-input/R0.jsonl --tag R0 \
        --neurons results/part92/step2/neurons.json --out results/part219/labels-deepseek-R0.jsonl
Also the route for Part 92's Haiku reader in this part: the direct-API credit key returns 401 since 2026-11-20
(memory reference_claude_api_monthly_credit), so both Haiku passes run here with --model claude-haiku-5-5
--share 1.0 --max-tokens 500 through OpenCode Go: same prompt, parser, model and cap as label92.py, another route.
Written by Journal Claude (Opus 5.5), 2026-11-22.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "part92"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import label92  # noqa: E402  Part 92's registered prompt and parser
import judge_client  # noqa: E402  the OpenCode Go transport

MODEL = "deepseek-v4.1-flash"
SHARE = 0.10
HAIKU = "claude-haiku-5-5"


def load_items(answers: Path, neurons: Path) -> list[dict]:
    with neurons.open(encoding="utf-8") as stream:
        by_id = {str(row["id"]): row for row in json.load(stream)}
    items = []
    with answers.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            neuron = by_id[str(row["id"])]
            items.append({"public": {"id": row["id"], "unit": row["unit"], "question_index": int(row["question_index"])},
                          "name": neuron["persona"], "desc": neuron["desc"], "q": row["question"], "a": row["answer"]})
    return items


def key_of(public: dict) -> tuple[str, str, int]:
    return (str(public["id"]), str(public["unit"]), int(public["question_index"]))


def sample(items: list[dict], tag: str, share: float = SHARE) -> list[dict]:
    ordered = sorted(items, key=lambda item: key_of(item["public"]))
    if share >= 1.0:
        return ordered
    rng = random.Random(f"219:{tag}")
    rng.shuffle(ordered)
    return ordered[:round(share * len(ordered))]


def build_prompt(item: dict) -> str:
    prompt = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        prompt = prompt.replace("{" + field + "}", str(item[field]))
    return prompt


def label_one(item: dict, model: str = MODEL, max_tokens: int = 4000) -> dict:
    error = None
    for attempt in range(3):
        try:
            response = judge_client.call_prompt(build_prompt(item), model=model, route="ocgo",
                                                session=f"part219-{model}", max_tokens=max_tokens, timeout=300)
            return {**item["public"], **label92._parse(str(response.get("text", ""))), "reader": model,
                    "max_tokens": max_tokens}  # the cap, per row (Part 221 design, 2026-11-22)
        except Exception as exc:  # recorded, retried, then written as an error row
            error = repr(exc)[:200]
            time.sleep(5 * (attempt + 1))
    return {**item["public"], "error": error, "reader": model, "max_tokens": max_tokens}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=MODEL, help=f"{MODEL} (the 10%% sample) or {HAIKU} (the two full passes)")
    ap.add_argument("--share", type=float, default=SHARE, help="1.0 labels every read-set item")
    ap.add_argument("--max-tokens", type=int, default=4000, help="500 for Haiku, as label92.py")
    args = ap.parse_args(argv)
    chosen = sample(load_items(Path(args.answers), Path(args.neurons)), args.tag, args.share)
    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if "error" not in row:
                    done.add(key_of(row))
    todo = [item for item in chosen if key_of(item["public"]) not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a", encoding="utf-8") as stream:
        for row in pool.map(lambda item: label_one(item, args.model, args.max_tokens), todo):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    print(json.dumps({"tag": args.tag, "sample": len(chosen), "pending": len(todo), "output": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
