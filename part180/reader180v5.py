#!/usr/bin/env python3
"""Build and read Part 180 v5.1's K-prefix items with the frozen two-question prompt.

Design sections 1 and 3: each arm is read on the scored K-token prefix, K = min(length of the pair's two texts, 40)
answer tokens, as v4/score180 define it. The pairs are (S, A0) and (S, A1) for the self margins and (C, B) for the
content margin, so:

  A0, A1, C, B  carry their own pair's K
  S             is read once, at min(len(S), len(A0), len(A1), 40): the prefix inside K for BOTH self pairs
                (a conservative choice, recorded per item as `k_by_pair`)

`B` is one generated text per configuration but is read once per neuron, against that neuron's persona, exactly as
v4 read its shared B0. Items are shuffled with a fixed seed. DeepSeek V4.1 Flash is the reader that decides;
Haiku 5.5 is available beside it and never used to choose (design section 2's rule, applied here too).

  read: python -I part180/reader180v5.py read --reader deepseek
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import gold180_read  # noqa: E402
import texts180 as v4  # noqa: E402
import texts180v5  # noqa: E402

K_CAP = texts180v5.K_CAP
SEED = 180
ARMS = ("S", "A0", "A1", "C", "B")
READERS = ("deepseek", "haiku")
PROMPT_PATH = HERE / "prompt180v5.txt"


def reader_binding(reader: str = "deepseek") -> dict:
    """The exact reader the labels are bound to: model, route and the prompt file's sha256.

    Model and route are read back out of gold180_read's call function, so a change there cannot silently leave a
    stale registration record here.
    """
    call = gold180_read.call_deepseek if reader == "deepseek" else gold180_read.call_haiku
    source = inspect.getsource(call)
    model = re.search(r'model="([^"]+)"', source)
    route = re.search(r'route="([^"]+)"', source)
    if model is None or route is None:
        raise ValueError(f"cannot read the reader model/route out of {call.__name__}")
    raw = PROMPT_PATH.read_bytes()
    if gold180_read.PROMPT != raw.decode("utf-8"):
        raise ValueError("gold180_read.PROMPT does not match part180/prompt180v5.txt")
    return {"reader": reader, "call": call.__name__, "model": model.group(1), "route": route.group(1),
            "prompt_file": str(PROMPT_PATH.relative_to(ROOT)),
            "prompt_sha256": hashlib.sha256(raw).hexdigest()}


def _row(index, neuron_id, key, config, text_rows_label):
    try:
        return index[(neuron_id, key, config)]
    except KeyError:
        raise ValueError(f"frozen text row missing for {text_rows_label}") from None


def build_items(text_rows, neurons, tokenizer, *, configurations=texts180v5.CONFIGURATIONS,
                seed: int = SEED, k_cap: int = K_CAP):
    """One shuffled K-prefix item per (configuration, neuron, arm). No identifier reaches the prompt text."""
    index = texts180v5._text_index(text_rows)
    items = []
    for config in configurations:
        number = int(config["number"])
        for neuron in neurons:
            neuron_id = str(neuron["id"])
            rows = {"S": _row(index, neuron_id, "S", number, f"{neuron_id}/S/c{number}"),
                    "C": _row(index, neuron_id, "C", number, f"{neuron_id}/C/c{number}"),
                    "A0": _row(index, neuron_id, "A0", None, f"{neuron_id}/A0"),
                    "A1": _row(index, neuron_id, "A1", None, f"{neuron_id}/A1"),
                    "B": _row(index, "unpushed", "B", number, f"unpushed/B/c{number}")}
            lengths = {arm: len(row["token_ids"]) for arm, row in rows.items()}
            k_self_a0 = min(lengths["S"], lengths["A0"], k_cap)
            k_self_a1 = min(lengths["S"], lengths["A1"], k_cap)
            k_cont = min(lengths["C"], lengths["B"], k_cap)
            k_s = min(k_self_a0, k_self_a1)
            specs = (
                ("S", rows["S"], config["self_prompt"], k_s, {"A0": k_self_a0, "A1": k_self_a1}),
                ("A0", rows["A0"], rows["A0"]["generation_prompt"], k_self_a0, {"S": k_self_a0}),
                ("A1", rows["A1"], rows["A1"]["generation_prompt"], k_self_a1, {"S": k_self_a1}),
                ("C", rows["C"], config["content_prompt"], k_cont, {"B": k_cont}),
                ("B", rows["B"], config["content_prompt"], k_cont, {"C": k_cont}),
            )
            for arm, row, question, k, k_by_pair in specs:
                token_ids = [int(value) for value in row["token_ids"][:k]]
                items.append({"id": f"v5-c{number}-{neuron_id}-{arm}", "config": number,
                              "neuron": neuron_id, "key": arm,
                              "persona": str(neuron["persona"]), "desc": str(neuron.get("desc", "")),
                              "question": str(question), "k": int(k), "k_by_pair": k_by_pair,
                              "prefix_token_ids": token_ids, "prefix_tokens": len(token_ids),
                              "prefix_text": tokenizer.decode(token_ids, skip_special_tokens=True)})
    expected = len(configurations) * len(neurons) * len(ARMS)
    if len(items) != expected:
        raise ValueError(f"expected {expected} reader items, built {len(items)}")
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("reader item ids are not unique")
    random.Random(int(seed)).shuffle(items)
    return items


def label_items(items, call, *, sink=None):
    """Read each item through gold180_read.read_one and hand every finished row to `sink` as it is made."""
    rows = []
    for item in items:
        prompt_row = {"id": item["id"], "persona": item["persona"], "desc": item["desc"],
                      "question": item["question"], "prefix": item["prefix_text"]}
        result = gold180_read.read_one(prompt_row, call)
        row = {"id": item["id"], "config": item["config"], "neuron": item["neuron"], "key": item["key"],
               "k": item["k"], "k_by_pair": item["k_by_pair"], "prefix_tokens": item["prefix_tokens"],
               "prefix_token_ids": item["prefix_token_ids"], "prefix_text": item["prefix_text"],
               "question": item["question"],
               "label": result["label"], "attempts": result["attempts"], "failed": result["failed"]}
        if result["label"] is None:
            row["raw_text"] = None
        else:
            row["raw_text"] = result.get("raw_text")
        rows.append(row)
        if sink is not None:
            sink(row)
    return rows


def validate_existing(previous, expected, path="reader file"):
    """Ids already on disk must be unique and inside the frozen grid; anything else is rejected, not merged."""
    seen = set()
    for row in previous:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError(f"reader row without an id in {path}")
        if row_id in seen:
            raise ValueError(f"duplicate reader row id {row_id} in {path}")
        if row_id not in expected:
            raise ValueError(f"reader row {row_id} is outside the frozen grid in {path}")
        seen.add(row_id)
    return seen


def cmd_read(args):
    import judge_client  # noqa: F401  (the DeepSeek/Haiku call path; never used at import time)

    tokenizer = _tokenizer(args)
    text_rows = texts180v5.read_rows(v4.normalise_path(args.texts))
    if not text_rows:
        raise ValueError(f"no frozen rows in {args.texts}; run texts180v5.py generate (and score) first")
    neurons = v4.load_concept_neurons(v4.normalise_path(args.results))
    items = build_items(text_rows, neurons, tokenizer, seed=int(args.seed))
    out = v4.normalise_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = {item["id"] for item in items}
    seen = validate_existing(texts180v5.read_rows(out), expected, str(out))
    pending = [item for item in items if item["id"] not in seen]
    call = gold180_read.call_deepseek if args.reader == "deepseek" else gold180_read.call_haiku
    with out.open("a", encoding="utf-8", newline="\n") as stream:
        def sink(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        label_items(pending, call, sink=sink)
    written = texts180v5.read_rows(out)
    if len({str(row["id"]) for row in written}) != len(written):
        raise ValueError(f"duplicate reader row id in {out}")
    unlabelled = sum(1 for row in written if row.get("label") is None)
    meta = {**reader_binding(args.reader), "seed": int(args.seed), "items_expected": len(items),
            "items_present": len(written), "unlabelled": unlabelled,
            "k_cap": K_CAP, "k_rule": "K = min(length of the pair's two texts, 40); S read at min over both pairs",
            "arms": list(ARMS), "out": str(out)}
    meta_path = out.with_name(out.stem + ".meta.json")
    with meta_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(meta, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"reader": args.reader, "items": len(items), "read_now": len(pending),
                      "already_present": len(seen), "unlabelled": unlabelled, "out": str(out)}))
    return {"items": len(items), "read_now": len(pending), "unlabelled": unlabelled}


def _tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    read = sub.add_parser("read")
    read.add_argument("--results", default=str(ROOT / "results"))
    read.add_argument("--texts", default=str(ROOT / "results" / "part180" / "v5" / "texts.jsonl"))
    read.add_argument("--out", default=str(ROOT / "results" / "part180" / "v5" / "reader-deepseek.jsonl"))
    read.add_argument("--reader", choices=READERS, default="deepseek")
    read.add_argument("--model", default=v4.MODEL_ID)
    read.add_argument("--revision", default=v4.MODEL_REVISION)
    read.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.cmd != "read":
        parser.error("only the read command is implemented")
    cmd_read(args)


if __name__ == "__main__":
    main()
