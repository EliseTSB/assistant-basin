#!/usr/bin/env python3
"""shortlist104.py -- Part 104's stratified shortlist (PART104-DESIGN.md, "Units").

Part 92's pool rule and prompt (part92/shortlist92.py) with three changes the design fixes: Part 92's 53 picks are left
out; the pool is split at layer 3 (layers 0 to 3, layers 4 to 15) and each stratum is picked separately, in its own
seeded shuffle (104003 and 104415); the cap is 36 for layers 0 to 3 and 45 for layers 4 to 15. Claude Sonnet 5.5,
thinking off, as in Part 92's Amendment 3. Ids p104-00 onward, layers 0 to 3 first; control seed 104500 plus the id's
number. A pick naming an id outside its stratum's pool, a repeat, or a unit already picked is skipped and counted.

Written by Claude (Opus 5.5), 2026-10-29, after Part 104's registration and before any Part 104 answer.
Usage: shortlist104.py LABELS_DIR PART92_NEURONS.json OUT.json
"""
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part92"))
import shortlist92 as s92  # noqa: E402  (Part 92's KINDS, PROMPT and key reader, unchanged)

STRATA = (("layers 0-3", range(0, 4), 36, 104003), ("layers 4-15", range(4, 16), 45, 104415))


def prompt_for(cap):
    p = s92.PROMPT.replace("Pick up to 60 answers", f"Pick up to {cap} answers")
    p = p.replace("at most 60 objects", f"at most {cap} objects")
    if f"up to {cap} answers" not in p or f"at most {cap} objects" not in p:
        raise RuntimeError("Part 92's prompt text changed; the cap was not substituted")
    return p


def build_pool(src, part92_neurons):
    labels = {json.loads(x)["id"]: json.loads(x) for x in open(src / "labels.jsonl", encoding="utf-8")}
    texts = {json.loads(x)["id"]: json.loads(x)["text"] for x in open(src / "items.jsonl", encoding="utf-8")}
    keymap = json.load(open(src / "KEY-private.json", encoding="utf-8"))
    taken_items = {n["item"] for n in part92_neurons}
    pool = {}
    for i, lab in labels.items():
        if not (lab.get("coherent") and lab.get("kind") in s92.KINDS) or i in taken_items:
            continue
        neurons = [u for u in keymap[i] if u["unit"] == "neuron"]
        if neurons:
            pool[i] = neurons[0]
    return pool, texts


def ask(prompt, items, texts, seed):
    order = sorted(items)
    random.Random(seed).shuffle(order)
    lines = "\n".join(f"{i}\t{' '.join(texts[i].split())}" for i in order)
    body = {"model": "claude-sonnet-5-5", "max_tokens": 8000, "thinking": {"type": "between_tools"},
            "messages": [{"role": "user", "content": prompt + lines}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": s92.key(), "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    res = json.load(urllib.request.urlopen(req, timeout=600))
    text = "".join(p.get("text", "") for p in res["content"] if p.get("type") == "text")
    return res, json.loads(text[text.index("["):text.rindex("]") + 1])


def main(src, part92_path, out):
    src, out = Path(src), Path(out)
    part92_neurons = json.load(open(part92_path, encoding="utf-8"))
    pool, texts = build_pool(src, part92_neurons)
    taken_units = {(int(n["layer"]), int(n["neuron"])) for n in part92_neurons}
    rows, report = [], []
    for name, layers, cap, seed in STRATA:
        items = {i: u for i, u in pool.items() if int(u["layer"]) in layers}
        res, picks = ask(prompt_for(cap), items, texts, seed)
        out.with_name(out.stem + f".raw-{name.replace(' ', '')}.json").write_text(json.dumps(res, ensure_ascii=False),
                                                                                  encoding="utf-8")
        kept = skipped = 0
        seen = set()
        for p in picks:
            u = items.get(p.get("id"))
            unit = (int(u["layer"]), int(u["index"])) if u else None
            if u is None or p["id"] in seen or unit in taken_units or kept == cap:
                skipped += 1
                continue
            seen.add(p["id"]); taken_units.add(unit); kept += 1
            n = len(rows)
            rows.append({"id": f"p104-{n:02d}", "item": p["id"], "layer": unit[0], "neuron": unit[1],
                         "persona": p["persona"], "desc": p["desc"], "screen_answer": texts[p["id"]],
                         "control_seed": 104500 + n, "stratum": name})
        report.append({"stratum": name, "pool": len(items), "picked": len(picks), "kept": kept, "skipped": skipped,
                       "usage": res.get("usage")})
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pool": len(pool), "strata": report, "total": len(rows)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
