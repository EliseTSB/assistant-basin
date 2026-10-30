"""Part 86 reader calibration: two cheap readers on Amendment 2's 335 sampled items, then agreement among all four.

Why (Claude, 2026-10-29): the registered second read (second86.py, Opus 5.5) agreed with the first reader (the J-lens
bundle's Haiku 5.5 labels) at kappa 0.157 although the two prompts carry the same field definitions and the kappa uses
one mapping (blind86._label_class). Most disagreement is the `coherent` flag (Opus calls 112 of the first reader's 276
"not coherent" answers coherent) and renamed assistants ("another AI" against "still Qwen or an assistant"). Before a
full re-read, two cheap readers from other families label the same items with the identical prompt
(second86.reader_prompt: READER-BRIEF.md verbatim plus the blind item): DeepSeek V4.1 Flash through OpenCode Go and
GPT-6 Luna through the ChatGPT-subscription proxy. Elise, 10-29 22:04 CT: Opus runs are expensive; Luna and DS V4.1
Flash are cheap. This is instrument calibration on blinded items: it never opens the real/random key, and no Part 86
verdict is read from it.

Each reader's call record (usage, reasoning, stop reason, served model) goes to a sidecar `<labels>.usage.jsonl`,
because a label row must hold exactly id, persona, kind and coherent.

Commands:
  python3 -I calib86.py --label ds      # writes labels_ds.jsonl
  python3 -I calib86.py --label luna    # writes labels_luna.jsonl
  python3 -I calib86.py --agree         # writes calib-agreement.json
  python3 -I calib86.py --sample-v2               # Amendment 7: 335 fresh items, items_v2cal.jsonl
  python3 -I calib86.py --version 2 --label ds   # READER-BRIEF-v2.md on the fresh items, writes labels_ds_v2.jsonl
  python3 -I calib86.py --version 2 --agree      # writes calib-agreement-v2.json
--agree first checks that the first/second pair reproduces agreement.json's kappa exactly, and stops if it does not.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import second86  # noqa: E402  the registered second reader: prompt, parsing, resume and retry loop
import judge_client  # noqa: E402  the repo's OpenCode Go and cliproxy transport

blind86 = second86.blind86

READERS = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "workers": 6},
    "luna": {"model": "gpt-6-luna", "route": "cliproxy", "workers": 4},
}
MAX_TOKENS = 8000
UNUSED_KEY = "calib86-no-anthropic-key"
_usage_lock = threading.Lock()


BRIEFS = {1: second86.READER_BRIEF_PATH, 2: ROOT / "part86" / "READER-BRIEF-v2.md"}


V2CAL_ITEMS_PATH = HERE / "items_v2cal.jsonl"
V2CAL_SEED = 8688


def sample_v2() -> int:
    """Amendment 7: version 2 is calibrated on 335 fresh items, none of them in Amendment 2's sample, because the
    version-2 rules were written after reading the version-1 disagreements on that sample."""
    used = {row["id"] for row in second86.read_items(second86.ITEMS_SECOND_PATH)}
    pool = sorted((row for row in second86.read_items(second86.FIRST_ITEMS_PATH) if row["id"] not in used),
                  key=lambda row: row["id"])
    chosen = random.Random(V2CAL_SEED).sample(pool, len(used))
    second86.write_jsonl(V2CAL_ITEMS_PATH, chosen)
    return len(chosen)


def items_path(version: int) -> Path:
    return second86.ITEMS_SECOND_PATH if version == 1 else V2CAL_ITEMS_PATH


def labels_path(reader: str, version: int = 1) -> Path:
    return HERE / (f"labels_{reader}.jsonl" if version == 1 else f"labels_{reader}_v{version}.jsonl")


def make_request(reader: str, usage_path: Path, brief_text: str | None = None):
    config = READERS[reader]

    def request(item: dict, _key: str) -> dict:
        out = judge_client.call_prompt(
            second86.reader_prompt(item, brief_text),
            model=config["model"],
            route=config["route"],
            session=f"part86-calib-{reader}",
            max_tokens=MAX_TOKENS,
            timeout=300,
        )
        record = {
            "id": item["id"],
            "usage": out["usage"],
            "reasoning": out["reasoning_tokens"],
            "stop": out["stop_reason"],
            "served": out["served"],
        }
        with _usage_lock, usage_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return second86.parse_label(out["text"], item["id"])

    return request


def label(reader: str, items: Path, out: Path, workers: int | None, version: int = 1) -> dict:
    if READERS[reader]["route"] == "ocgo" and not os.environ.get("OPENCODE_GO_API_KEY"):
        # 2026-10-29: a launch_arc.py unit has no OPENCODE_GO_API_KEY, and the retry loop turned that into 90
        # error rows over three minutes. Stop at once instead.
        raise SystemExit("STOP: OPENCODE_GO_API_KEY is not set (read it from its env file in the run command)")
    usage_path = out.with_name(out.name.replace(".jsonl", ".usage.jsonl"))
    return second86.label_items(
        items,
        out,
        key=UNUSED_KEY,
        request_fn=make_request(reader, usage_path, BRIEFS[version].read_text(encoding="utf-8")),
        workers=workers or READERS[reader]["workers"],
    )


def _rows_by_id(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in second86.read_jsonl(path)}


def _binary(row: dict) -> int:
    return int(blind86._label_class(row) == "coherent_persona")


def agree(version: int = 1) -> dict:
    first_all = _rows_by_id(second86.FIRST_LABELS_PATH)
    # Integrity check, always on Amendment 2's sample: the first/second pair must reproduce agreement.json.
    orig_ids = [row["id"] for row in second86.read_items(second86.ITEMS_SECOND_PATH)]
    second_all = _rows_by_id(second86.LABELS_SECOND_PATH)
    both = [i for i in orig_ids if i in first_all and i in second_all and "error" not in second_all[i]]
    registered = json.loads((HERE / "agreement.json").read_text(encoding="utf-8"))["kappa"]
    check = blind86.cohen_kappa(first_all, second_all, both)
    if abs(check - registered) > 1e-12:
        raise SystemExit(f"STOP: first/second kappa {check} does not reproduce agreement.json's {registered}")

    sample_ids = [row["id"] for row in second86.read_items(items_path(version))]
    readers = {"first_haiku": {i: first_all[i] for i in sample_ids if i in first_all}}
    if version == 1:
        readers["second_opus"] = second_all
        for name in READERS:
            if labels_path(name).exists():
                readers[name] = _rows_by_id(labels_path(name))
    else:
        for name in READERS:
            if labels_path(name, version).exists():
                readers[f"{name}_v{version}"] = _rows_by_id(labels_path(name, version))
    valid = {
        name: {i: r for i, r in rows.items() if "error" not in r and i in set(sample_ids)}
        for name, rows in readers.items()
    }

    report = {
        "sample_items": len(sample_ids),
        "readers": {
            name: {
                "valid": len(rows),
                "errors": sum("error" in r for r in readers[name].values()),
                "classes": dict(Counter(blind86._label_class(r) for r in rows.values())),
                "coherent_true": sum(bool(r["coherent"]) for r in rows.values()),
            }
            for name, rows in valid.items()
        },
        "pairs": {},
        "check": {"first_second_kappa": check, "agreement_json_kappa": registered},
    }
    for a, b in combinations(valid, 2):
        ids = [i for i in sample_ids if i in valid[a] and i in valid[b]]
        if not ids:
            continue
        report["pairs"][f"{a}~{b}"] = {
            "n": len(ids),
            "kappa_coherent_persona": blind86.cohen_kappa(valid[a], valid[b], ids),
            "raw_coherent_persona": sum(_binary(valid[a][i]) == _binary(valid[b][i]) for i in ids) / len(ids),
            "raw_coherent_flag": sum(valid[a][i]["coherent"] == valid[b][i]["coherent"] for i in ids) / len(ids),
            "raw_class": sum(
                blind86._label_class(valid[a][i]) == blind86._label_class(valid[b][i]) for i in ids
            ) / len(ids),
        }
    report["brief_version"] = version
    name = "calib-agreement.json" if version == 1 else f"calib-agreement-v{version}.json"
    (HERE / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", choices=sorted(READERS))
    parser.add_argument("--items", type=Path)
    parser.add_argument("--sample-v2", action="store_true", help="write Amendment 7's fresh 335-item sample")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--agree", action="store_true")
    parser.add_argument("--version", type=int, choices=sorted(BRIEFS), default=1,
                        help="reader brief: 1 = READER-BRIEF.md, 2 = READER-BRIEF-v2.md (Amendment 7)")
    args = parser.parse_args(argv)
    if args.label:
        out = args.out or labels_path(args.label, args.version)
        print(json.dumps(label(args.label, args.items or items_path(args.version), out, args.workers, args.version)))
    if args.agree:
        print(json.dumps(agree(args.version), indent=2))
    if args.sample_v2:
        print(json.dumps({"sampled_items": sample_v2(), "seed": V2CAL_SEED, "output": V2CAL_ITEMS_PATH.name}))
    if not args.label and not args.agree and not args.sample_v2:
        parser.error("give --label READER, --agree or --sample-v2")


if __name__ == "__main__":
    main()
