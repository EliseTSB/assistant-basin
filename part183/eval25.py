#!/usr/bin/env python3
"""eval25.py -- Part 183 B: Part 25's published four-neuron set and its 20 random controls, regenerated on the
current attention kernel (eager since 2026-11-03), with Part 25's own generator and reader (PART183-DESIGN.md).

Conditions: no zeroing; the published four (results/part25/search-google-removal.json, pruned.neurons); each of the
20 random four-neuron sets in results/part25/controls.json (random_controls.google-removal.sets). For each, the six
identity questions of circuit.ID_PROMPTS through part25._all_answers (greedy, 48 new tokens, batch 1, zeroed at every
position). Each answer is classified by read_part25.classify and set beside the stored SDPA answer where Part 25
stored one (means.json answers_unsteered; pruned.answers_six_questions; each random set's "Who created you?").

Usage: python -I -B part183/eval25.py --out results/part183/part25-eager [--device xpu] [--max-gpu-gb 14.5]
Writes answers.jsonl (one row per condition and question), summary.json (183.5 to 183.7 read from it), run.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import neuron_report as nr  # noqa: E402
import part25  # noqa: E402
import read_part25  # noqa: E402

MAKER_KEEP = {"Alibaba/Tongyi/Qwen team", None, "none"}
POSITIVE_QUESTION = "Who created you?"


def stored_sdpa():
    """The SDPA answers Part 25 stored, keyed by (condition, question)."""
    res = ROOT / "results" / "part25"
    means = json.loads((res / "means.json").read_text(encoding="utf-8"))
    search = json.loads((res / "search-google-removal.json").read_text(encoding="utf-8"))
    controls = json.loads((res / "controls.json").read_text(encoding="utf-8"))
    out = {}
    for q, a in means["answers_unsteered"].items():
        out[("none", q)] = a["answer"]
    for q, a in search["pruned"]["answers_six_questions"].items():
        out[("four", q)] = a["answer"]
    for s in controls["random_controls"]["google-removal"]["sets"]:
        for q, a in s["answers"].items():
            out[(f"random_{s['trial']}", q)] = a["answer"]
    return out


def conditions():
    res = ROOT / "results" / "part25"
    search = json.loads((res / "search-google-removal.json").read_text(encoding="utf-8"))
    controls = json.loads((res / "controls.json").read_text(encoding="utf-8"))
    four = [(int(n["layer"]), int(n["index"])) for n in search["pruned"]["neurons"]]
    expected = {(6, 753), (7, 1169), (11, 8868), (29, 776)}
    if set(four) != expected:
        raise SystemExit(f"published four-set mismatch: {sorted(four)}")
    conds = [("none", []), ("four", four)]
    for s in controls["random_controls"]["google-removal"]["sets"]:
        conds.append((f"random_{s['trial']}", [(int(n["layer"]), int(n["index"])) for n in s["neurons"]]))
    if len(conds) != 22:
        raise SystemExit(f"expected 22 conditions, found {len(conds)}")
    return conds


def maker_of(answer: str, question: str):
    c = read_part25.classify(answer, question)
    m = c.get("maker") if isinstance(c, dict) else None
    if isinstance(m, dict):
        m = m.get("category")
    return m if m else "none"


def summarize(rows):
    by = {(r["condition"], r["question"]): r for r in rows}
    qs = list(part25.circuit.ID_PROMPTS) if hasattr(part25, "circuit") else sorted({r["question"] for r in rows})
    four = {q: by[("four", q)]["maker_eager"] for q in qs}
    flipped = [q for q in qs if four[q] not in MAKER_KEEP]
    p5 = (four[POSITIVE_QUESTION] == "Google/DeepMind" and flipped == [POSITIVE_QUESTION])
    random_foreign = [(r["condition"], r["question"], r["maker_eager"]) for r in rows
                      if r["condition"].startswith("random_") and r["maker_eager"] not in MAKER_KEEP]
    pairs = [r for r in rows if r["condition"] in ("none", "four") and r["maker_sdpa"] is not None]
    same = sum(1 for r in pairs if r["maker_eager"] == r["maker_sdpa"])
    random_pairs = [r for r in rows if r["condition"].startswith("random_") and r["maker_sdpa"] is not None]
    return {
        "four_makers_eager": four,
        "four_flipped_questions": flipped,
        "183.5": {"held": bool(p5), "rule": "four zeroed: Who created you? -> Google/DeepMind, other five keep Alibaba/Tongyi/Qwen or none"},
        "183.6": {"held": not random_foreign, "random_foreign_makers": random_foreign},
        "183.7": {"held": same >= 11, "same": same, "of": len(pairs),
                  "rows": [{"condition": r["condition"], "question": r["question"], "eager": r["maker_eager"],
                            "sdpa": r["maker_sdpa"]} for r in pairs]},
        "random_who_created_same_as_sdpa": sum(1 for r in random_pairs if r["maker_eager"] == r["maker_sdpa"]),
        "random_who_created_pairs": len(random_pairs),
        "text_identical_to_sdpa": sum(1 for r in rows if r["answer_sdpa"] is not None and r["answer_eager"] == r["answer_sdpa"]),
        "text_pairs": sum(1 for r in rows if r["answer_sdpa"] is not None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--max-gpu-gb", type=float, default=14.5)
    ap.add_argument("--dry", action="store_true", help="read the stored answers and conditions only, no model")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    stored = stored_sdpa()
    if a.dry:
        print(json.dumps({"conditions": len(conds), "stored_answers": len(stored)}))
        return 0
    t0 = time.time()
    tok, model = nr.load_model(part25.MODEL, part25.REVISION, a.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    nr.spill_guard(a.device, a.max_gpu_gb)
    attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
    rows = []
    with (out / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for name, neurons in conds:
            answers = part25._all_answers(tok, model, layers, neurons, "zero", None, a.device, a.max_gpu_gb)
            for q, rec in answers.items():
                sd = stored.get((name, q))
                row = {"condition": name, "neurons": [list(n) for n in neurons], "question": q,
                       "answer_eager": rec["answer"], "maker_eager": maker_of(rec["answer"], q),
                       "answer_sdpa": sd, "maker_sdpa": maker_of(sd, q) if sd is not None else None}
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    run = {"model": part25.MODEL, "revision": part25.REVISION, "device": a.device, "attn_implementation": attn,
           "conditions": len(conds), "answers": len(rows), "wall_seconds": round(time.time() - t0, 1),
           "max_new_tokens": 48, "batch": 1, "decoding": "greedy"}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("four_flipped_questions",)}), flush=True)
    print(json.dumps({k: summary[k]["held"] for k in ("183.5", "183.6", "183.7")}), flush=True)
    if attn != "eager":
        print(f"WARNING: attention kernel was {attn!r}, not eager", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
