#!/usr/bin/env python3
"""Blindly label each scored Part 180 text prefix for inclusion criterion v."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART92 = ROOT / "part92"
if str(PART92) not in sys.path:
    sys.path.insert(0, str(PART92))
# Under python -I the script's own folder is not on sys.path, so `import score180` fails on the Arc job runner
# (pilot 2, 2026-11-03 18:58); tests passed because pytest adds the folder.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import score180  # noqa: E402
from label92 import PROMPT  # noqa: E402

MODEL = "deepseek-v4.1-flash"
ROUTE = "ocgo"
SESSION = "part180-reader"
PREFIX_CASE = {"S0": "self0_S0", "S1": "self1_S1", "A0": "self0_A0", "A1": "self1_A1",
               "C0": "cont0_C0", "C1": "cont1_C1", "B0": "cont0_B0", "B1": "cont1_B1"}


def parse_label(text: str) -> dict:
    match = re.search(r"\{[^{}]*\}", text or "")
    if not match:
        raise ValueError("DeepSeek response contained no JSON object")
    row = json.loads(match.group(0))
    expected = {"shows_persona", "self_claim", "speaker", "coherent"}
    if set(row) != expected:
        raise ValueError("DeepSeek returned invalid label fields")
    if any(type(row[key]) is not bool for key in ("shows_persona", "self_claim", "coherent")):
        raise ValueError("DeepSeek returned non-boolean label values")
    if row["speaker"] not in ("ai", "human", "other", "none"):
        raise ValueError("DeepSeek returned an invalid speaker")
    return row


def response_text(response) -> str:
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]
    if isinstance(response, dict):
        return "".join(part.get("text", "") for part in response.get("content", [])
                       if part.get("type") == "text")
    return ""


def _fill_prompt(persona, question, answer):
    prompt = PROMPT
    for key, value in (("name", persona["persona"]), ("desc", persona.get("desc", "")),
                       ("q", question), ("a", answer)):
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def build_items(text_rows, neurons, tokenizer, *, seed: int = 180):
    """Build shuffled K-prefix items; identifiers and keys stay out of prompt text."""
    neuron_by_id = {str(row["id"]): row for row in neurons}
    task_ks = {}
    for task in score180._task_list(text_rows, end_token_ids=score180._end_token_ids(tokenizer)):
        task_ks[(str(task["neuron"]), task["case"])] = task["k_tokens"]
    items = []
    for row in text_rows:
        neuron_id, key = str(row["neuron"]), str(row["key"])
        if key not in PREFIX_CASE:
            continue
        if neuron_id not in neuron_by_id:
            raise ValueError(f"frozen text references unknown neuron {neuron_id}")
        case = PREFIX_CASE[key]
        k_tokens = int(task_ks[(neuron_id, case)])
        token_ids = [int(value) for value in row["token_ids"][:k_tokens]]
        prefix = tokenizer.decode(token_ids, skip_special_tokens=True)
        items.append({"id": neuron_id, "key": key, "persona": neuron_by_id[neuron_id],
                      "question": str(row.get("generation_prompt", row["prompt"])),
                      "prefix_tokens": len(token_ids),
                      "prefix_text": prefix})
    expected = len({str(row["neuron"]) for row in text_rows}) * len(PREFIX_CASE)
    if len(items) != expected:
        raise ValueError(f"expected {expected} scored-prefix rows, found {len(items)}")
    random.Random(int(seed)).shuffle(items)
    return items


ATTEMPTS = 3


def label_items(items, client, *, sink=None, attempts=ATTEMPTS):
    """Use the Part 92 prompt. A reply with no valid label is asked again, at most `attempts` times, with every
    failed reply kept; after that the run fails loudly. Pilot 2 (2026-11-03) died on one empty-JSON reply with
    nothing written, so rows also go to `sink` as they are made."""
    output = []
    for item in items:
        prompt = _fill_prompt(item["persona"], item["question"], item["prefix_text"])
        failed = []
        for attempt in range(1, int(attempts) + 1):
            response = client.call_prompt(prompt, model=MODEL, route=ROUTE, session=SESSION,
                                          max_tokens=4000, timeout=300)
            raw_text = response_text(response)
            try:
                label = parse_label(raw_text)
                break
            except (ValueError, json.JSONDecodeError) as error:
                failed.append({"attempt": attempt, "error": str(error), "raw_text": raw_text})
                if attempt == int(attempts):
                    raise ValueError(f"{error} (after {attempts} attempts)") from error
        row = {"id": item["id"], "unit": "neuron", "key": item["key"],
               "prefix_tokens": item["prefix_tokens"], "prefix_text": item["prefix_text"],
               "raw_text": raw_text, "label": label, "attempts": len(failed) + 1,
               "failed_replies": failed, **label}
        output.append(row)
        if sink is not None:
            sink(row)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", default=str(ROOT / "results" / "part180" / "texts.jsonl"))
    parser.add_argument("--neurons", default=str(ROOT / "results" / "part92" / "step2" / "neurons.json"))
    parser.add_argument("--out", default=str(ROOT / "results" / "part180" / "reader-prefixes.jsonl"))
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--revision", default="48d788eca847d4d7548f375ad03d3c9312f6139e")
    parser.add_argument("--seed", type=int, default=180)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    import judge_client

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    items = build_items(score180._load_jsonl(Path(args.texts)), json.loads(Path(args.neurons).read_text(encoding="utf-8")),
                        tokenizer, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        def sink(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        rows = label_items(items, judge_client, sink=sink)
    partial.replace(out)
    print(json.dumps({"items": len(rows), "seed": args.seed, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
