#!/usr/bin/env python3
"""Part 109 (PART109-DESIGN.md): six slot neurons asked "What are your favorite activities?" cold (A), after their own
persona answer with the push (B) and without it (C), plus one unpushed cold answer (D). Writes
results/part109/answers.jsonl and label items for part92/label92.py. Written by Claude (Opus 5.5), 2026-10-29.
"""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part92"))
import common92 as c  # noqa: E402

SNAP = Path("/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/"
            "48d788eca847d4d7548f375ad03d3c9312f6139e")
UNITS = ["p92-37", "p92-04", "p92-13", "p92-07", "p92-35", "p92-24"]
Q = "What are your favorite activities?"


def two_sentences(text):
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
    return text[: ends[1] if len(ends) > 1 else (ends[0] if ends else len(text))].strip()


def chat_ids(tok, messages):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]


def generate(model, tok, seqs, layer=None, vectors=None, n=100):
    import torch
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    width = max(len(s) for s in seqs)
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs])
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
    handle = c.add_row_hook(layer, vectors) if vectors is not None else None
    try:
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=n, pad_token_id=pad)
    finally:
        if handle is not None:
            handle.remove()
    return [tok.decode(row[width:], skip_special_tokens=True).strip() for row in out]


def main(units=None, out=None):
    import torch
    units = list(units) if units is not None else UNITS
    neurons = {r["id"]: r for r in json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))}
    pushes = torch.load(ROOT / "results/part99-pushes.pt", map_location="cpu")
    first = {}
    for line in open(ROOT / "results/part108/I.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["unit"] == "neuron" and int(r["question_index"]) == 0:
            first[r["id"]] = two_sentences(r["answer"])
    tok, model = c.load_model(SNAP, "cpu")
    layers = c.get_layers(model)
    rows, started = [], time.perf_counter()
    for uid in units:
        layer, vec = layers[int(neurons[uid]["layer"])], pushes[f"{uid}|neuron"].float().unsqueeze(0).to(torch.bfloat16)
        convo = [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": first[uid]},
                 {"role": "user", "content": Q}]
        a = generate(model, tok, [chat_ids(tok, [{"role": "user", "content": Q}])], layer, vec)[0]
        b = generate(model, tok, [chat_ids(tok, convo)], layer, vec)[0]
        cc = generate(model, tok, [chat_ids(tok, convo)])[0]
        for arm, ans in (("A", a), ("B", b), ("C", cc)):
            rows.append({"id": uid, "unit": arm, "question_index": 0, "question": Q, "answer": ans,
                         "first_turn": first[uid] if arm != "A" else None})
        print(f"{uid} done {time.perf_counter() - started:.0f}s", flush=True)
    d = generate(model, tok, [chat_ids(tok, [{"role": "user", "content": Q}])])[0]
    rows += [{"id": uid, "unit": "D", "question_index": 0, "question": Q, "answer": d, "first_turn": None} for uid in units]
    out = Path(out) if out is not None else ROOT / "results/part109"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "unit", "question_index", "question", "answer")}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
