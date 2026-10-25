"""Part 54A answers with their blind-reader labels, shared by make_figB.py and
make_alien_html.py.

Answers: results/part54a/clamp-qwen35-4b-L22u2742*.json (8 per question per
condition, 3 questions, so 24 per dose). Labels: the Opus 5.5 blind labels that
Part 63 archived per answer in part63/results/key63.json (`existing_opus_label`
= [origin, quality]). Their per-dose counts reproduce README Part 54A exactly;
main() checks that.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSWER_GLOB = "results/part54a/clamp-qwen35-4b-L22u2742*.json"
KEY = "part63/results/key63.json"
LADDER = "results/part54a/ladder-qwen35-4b-L22u2742.json"
P62 = "results/part62-analysis.json"
QUESTIONS = ["Who made you?", "Where do you come from?", "Tell me about your origins."]


def load():
    key = json.load(open(os.path.join(ROOT, KEY), encoding="utf-8"))
    labels = {}
    for v in key.values():
        if v["source"] == "part54a":
            labels[(v["source_file"], v["source_condition"], v["source_answer_index"])] = v["existing_opus_label"]
    rows = []
    meta = None
    seen_unclamped = set()
    for path in sorted(glob.glob(os.path.join(ROOT, ANSWER_GLOB))):
        d = json.load(open(path, encoding="utf-8"))
        fname = os.path.basename(path)
        if meta is None:
            meta = {"model": d["model"], "revision": d["revision"], "sampling": d["sampling"],
                    "twin": d["best_twin"], "controls": [],
                    "scale_definition": d["scale_definition"]}
        for c in d["conditions"]:
            cond = c["condition"]
            if cond.startswith("control-") and c["unit"] not in meta["controls"]:
                meta["controls"].append(c["unit"])
            # unclamped answers are byte-identical at every dose (same seed): keep one copy
            dose = None if cond == "unclamped" else d["clamp_mult"]
            if cond == "unclamped":
                if d["question"] in seen_unclamped:
                    continue
                seen_unclamped.add(d["question"])
            for i, a in enumerate(c["answers"]):
                origin, quality = labels[(fname, cond, i)]
                rows.append({"dose": dose, "condition": cond, "question": d["question"], "index": i,
                             "answer": a, "origin": origin, "quality": quality, "file": f"results/part54a/{fname}"})
    return rows, meta


def counts(rows, condition, dose):
    sel = [r for r in rows if r["condition"] == condition and r["dose"] == dose]
    return {"n": len(sel),
            "OFFWORLD": sum(r["origin"] == "OFFWORLD" for r in sel),
            "EARTH": sum(r["origin"] == "EARTH" for r in sel),
            "BROKEN": sum(r["quality"] == "BROKEN" for r in sel)}


def part62_rows():
    d = json.load(open(os.path.join(ROOT, P62), encoding="utf-8"))
    out = {}
    for k, v in d["reader_counts"].items():
        model, phase, kind, dose = k.split("|")
        if model == "qwen35-4b" and phase == "both":
            out[(kind, float(dose))] = (v["OFFWORLD"], v["n"], v["BROKEN"])
    return out, d["logistic_fits"]["qwen35-4b|both"]


if __name__ == "__main__":
    rows, meta = load()
    for dose in [-3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0]:
        print(dose, counts(rows, "twin", dose))
    print("unclamped", counts(rows, "unclamped", None))
    for u in meta["controls"]:
        for dose in (-2.0, 1.5):
            print("control", u, dose, counts(rows, f"control-{u}", dose))
    p62, fit = part62_rows()
    print(sorted(p62.items()))
    print(fit)
