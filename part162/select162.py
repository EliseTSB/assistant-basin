"""Select Part 162's shared prompts from the three stored Part 93 answer files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / "part93" / "prompts93.json"
DEFAULT_GEN = ROOT / "results" / "part93" / "gen"
FAMILIES = ("claude", "gpt", "deepseek")


def word_count(text: str) -> int:
    return len(text.split())


def select(prompts_path: Path, gen_dir: Path) -> list[dict[str, str]]:
    with prompts_path.open(encoding="utf-8") as f:
        prompts = json.load(f)
    eligible = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        path = gen_dir / f"{family}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error") is None and isinstance(row.get("answer"), str):
                    eligible[family][row.get("prompt_id", row.get("id"))] = word_count(row["answer"])
    selected = []
    for item in prompts:
        pid = item.get("id")
        if all(eligible[family].get(pid, 0) >= 100 for family in FAMILIES):
            selected.append({"prompt_id": pid, "prompt": item["prompt"]})
    if len(selected) != 113:
        raise AssertionError(f"expected exactly 113 eligible prompts, found {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--part93-gen", type=Path, default=DEFAULT_GEN)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "prompts162.json")
    args = parser.parse_args()
    rows = select(args.prompts, args.part93_gen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {len(rows)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
