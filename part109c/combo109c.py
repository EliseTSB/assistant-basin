#!/usr/bin/env python3
"""Part 109c (PART109C-DESIGN.md): the concept dog (p92-08) and the slot dog (p92-37), each alone and together, pushed at
six strengths from -2x to +2x Part 99's push, on seven questions; plus opposite signs and a second slot dog (p92-09).
Greedy, 100 new tokens, OLMo 2 1B Instruct on the CPU, one batch of seven questions per condition. Writes
results/part109c/answers.jsonl, items.jsonl and personas.json for part92/label92.py. Written by Claude (Opus 5.5),
2026-10-29.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part92"))
sys.path.insert(0, str(ROOT / "part109"))
import common92 as c  # noqa: E402
import followup109 as f  # noqa: E402

CONCEPT, SLOT, SLOT2 = "p92-08", "p92-37", "p92-09"
QUESTIONS = ["Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.",
             "What is the capital of France?", "Write a short poem about the sea.", "What are your favorite activities?"]
DOSES = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]


def conditions():
    out = [{"id": "none", "push": {}}]
    for m in DOSES:
        out.append({"id": f"concept{m:+g}", "push": {CONCEPT: m}})
        out.append({"id": f"slot{m:+g}", "push": {SLOT: m}})
        out.append({"id": f"both{m:+g}", "push": {CONCEPT: m, SLOT: m}})
    out.append({"id": "concept+1_slot-1", "push": {CONCEPT: 1.0, SLOT: -1.0}})
    out.append({"id": "concept-1_slot+1", "push": {CONCEPT: -1.0, SLOT: 1.0}})
    out.append({"id": "slot2+1", "push": {SLOT2: 1.0}})
    out.append({"id": "concept_slot2+1", "push": {CONCEPT: 1.0, SLOT2: 1.0}})
    out.append({"id": "concept_slot2-1", "push": {CONCEPT: -1.0, SLOT2: -1.0}})
    return out


def generate_multi(model, tok, seqs, pushes, n=100):
    """Greedy-generate a left-padded batch with each (layer, vector) push added to every row after its layer."""
    import torch
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    width = max(len(s) for s in seqs)
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs])
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
    handles = [c.add_row_hook(layer, vec.unsqueeze(0).expand(len(seqs), -1).contiguous()) for layer, vec in pushes]
    try:
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=n, pad_token_id=pad)
    finally:
        for handle in handles:
            handle.remove()
    return [tok.decode(row[width:], skip_special_tokens=True).strip() for row in out]


def main(out=ROOT / "results/part109c"):
    import torch
    neurons = {r["id"]: r for r in json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))}
    pushes = torch.load(ROOT / "results/part99-pushes.pt", map_location="cpu")
    tok, model = c.load_model(f.SNAP, "cpu")
    layers = c.get_layers(model)
    seqs = [f.chat_ids(tok, [{"role": "user", "content": q}]) for q in QUESTIONS]
    rows, started = [], time.perf_counter()
    for cond in conditions():
        hooks = [(layers[int(neurons[uid]["layer"])], (m * pushes[f"{uid}|neuron"].float()).to(torch.bfloat16))
                 for uid, m in cond["push"].items()]
        answers = generate_multi(model, tok, seqs, hooks)
        for qi, (q, a) in enumerate(zip(QUESTIONS, answers)):
            rows.append({"id": cond["id"], "unit": "combo", "question_index": qi, "question": q, "answer": a,
                         "push": cond["push"]})
        print(f"{cond['id']} done {time.perf_counter() - started:.0f}s", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "items.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("id", "unit", "question_index", "question", "answer")},
                                ensure_ascii=False) + "\n")
    personas = [{"id": cond["id"], "persona": "Dog", "desc": "A dog."} for cond in conditions()]
    (out / "personas.json").write_text(json.dumps(personas, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} rows in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
