#!/usr/bin/env python3
"""Collect Part 137's fixed-cell, eight-arm question set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "part134") not in sys.path:
    sys.path.insert(0, str(ROOT / "part134"))
import run134 as base  # noqa: E402

CELL = {"cell": "p92-08+p92-09@0.8+0.6", "a": "p92-08", "b": "p92-09", "dose_a": 0.8, "dose_b": 0.6}
ARMS = ("pair", "a_only", "b_only", "a_random_1", "a_random_2", "b_random_1", "b_random_2", "none")
RANDOM_SEEDS = {"a_random_1": 1371, "a_random_2": 1372, "b_random_1": 1373, "b_random_2": 1374}


def load_questions(path=None):
    path = Path(path) if path else Path(__file__).with_name("questions137.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["non_task"] + data["task"]
    if len(data["non_task"]) != 16 or len(data["task"]) != 8 or len(questions) != 24:
        raise ValueError("Part 137 requires 16 non-task and 8 task questions")
    return questions


def main(argv=None, *, model=None, tokenizer=None, out=None, pushes=None, neuron_meta=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    p.add_argument("--out", default="results/part137")
    p.add_argument("--questions", default=str(Path(__file__).with_name("questions137.json")))
    a = p.parse_args(argv)
    if a.batch_size < 1: p.error("batch size must be positive")
    directory = Path(out) if out is not None else Path(a.out)
    questions = load_questions(a.questions)
    if model is None: tokenizer, model = base.load_a770(a.device)
    tok = tokenizer
    import torch
    layers = base.c.get_layers(model)
    if pushes is None:
        pushes = torch.load(ROOT / "results/part99-pushes.pt", map_location="cpu", weights_only=False)
    neuron_meta = neuron_meta or base.UNITS
    device = next(model.parameters()).device
    gate_questions = base.DEV_QUESTIONS
    before = base.generate_batch(model, tok, gate_questions, [], a.batch_size, return_token_ids=True)
    info = neuron_meta[CELL["a"]]
    zero = torch.zeros((len(gate_questions), int(pushes[f'{CELL["a"]}|neuron'].numel())), device=device,
                       dtype=next(model.parameters()).dtype)
    after = base.generate_batch(model, tok, gate_questions,
                                [(layers[int(info["layer"])], zero)], a.batch_size, return_token_ids=True)
    gate1 = before == after
    if not gate1:
        base.atomic_json(directory / "meta.json", {"status": "gate1_failed", "gate1_zero_hook_same_token_ids": False})
        raise RuntimeError("Gate 1 failed: zero-vector hook changed generated tokens")
    dtype = next(model.parameters()).dtype
    def vector(uid, dose): return (pushes[f"{uid}|neuron"].float() * dose).to(device=device, dtype=dtype)
    rows_path = directory / "answers.jsonl"
    old = base.read_rows(rows_path)
    done = {(r["cell"], r["arm"]) for r in old}
    meta = {"stage": "test", "status": "running", "device": a.device, "batch_size": a.batch_size,
            "cell": CELL, "arms": list(ARMS), "questions": questions,
            "expected_rows": 8 * len(questions), "gate1_zero_hook_same_answers": gate1,
            "random_seeds": RANDOM_SEEDS}
    base.atomic_json(directory / "meta.json", meta)
    for arm in ARMS:
        if (CELL["cell"], arm) in done: continue
        vectors = []
        a_vec, b_vec = vector(CELL["a"], CELL["dose_a"]), vector(CELL["b"], CELL["dose_b"])
        la, lb = int(neuron_meta[CELL["a"]]["layer"]), int(neuron_meta[CELL["b"]]["layer"])
        if arm == "pair": active = [(la, a_vec), (lb, b_vec)]
        elif arm == "a_only": active = [(la, a_vec)]
        elif arm == "b_only": active = [(lb, b_vec)]
        elif arm in RANDOM_SEEDS:
            real, replaced, layer_real, layer_replace = (a_vec, b_vec, la, lb) if arm.startswith("a_") else (b_vec, a_vec, lb, la)
            gen = torch.Generator(device="cpu"); gen.manual_seed(RANDOM_SEEDS[arm])
            rnd = torch.randn(real.shape, generator=gen, dtype=torch.float32).to(device=device)
            rnd *= replaced.float().norm() / rnd.norm()
            active = [(layer_real, real), (layer_replace, rnd.to(dtype))]
            random_path = directory / "random-vectors" / f"{arm}.pt"
            random_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"seed": RANDOM_SEEDS[arm], "vector": rnd.detach().cpu(),
                        "replaced_norm": float(replaced.float().norm().cpu())}, random_path)
        else: active = []
        for layer, vec in active:
            vectors.append((layers[layer], vec.unsqueeze(0).expand(len(questions), -1).contiguous()))
        answers = base.generate_batch(model, tok, questions, vectors, a.batch_size, max_new_tokens=120)
        batch = [{"stage": "test", "cell": CELL["cell"] if arm != "none" else "unpushed", "arm": arm,
                  "pair": [CELL["a"], CELL["b"]] if arm != "none" else [],
                  "doses": [CELL["dose_a"], CELL["dose_b"]] if arm != "none" else [],
                  "question_index": i, "question": q, "answer": answer,
                  "answer_hash": hashlib.sha256((q + "\0" + answer).encode("utf-8")).hexdigest()}
                 for i, (q, answer) in enumerate(zip(questions, answers))]
        base.append_jsonl_atomic(rows_path, batch)
        print(f"completed {arm}", flush=True)
    current = base.read_rows(rows_path)
    meta.update(status="complete", completed_rows=len(current), finished=True)
    base.atomic_json(directory / "meta.json", meta)
    return current


if __name__ == "__main__": main()
