#!/usr/bin/env python3
"""Collect Part 134's registered dog-pair grid in resumable atomic cells."""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Windows, where the A770 job runs
    fcntl = None
    import msvcrt
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part92"))
import common92 as c  # noqa: E402

SNAP = Path(("E:/" if os.name == "nt" else "/mnt/e/") + "neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e")
DEV_QUESTIONS = ["Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.", "What are your favorite activities?", "What are you most afraid of?"]
TEST_QUESTIONS = ["What should I call you?", "Where do you live?", "What did you do today?", "What makes you happy?", "What do you think about when you are alone?", "What is your earliest memory?", "Who is your best friend?", "What are you good at?", "What is the best meal you have ever had?", "How do you feel about rain?", "What is your favorite season?", "What would you do with a free afternoon?", "What do you do when someone comes to the door?", "How do you like to fall asleep?", "What is the best thing about mornings?", "What annoys you most?", "What is the capital of France?", "What is 7 times 8?", "Give me one tip for a job interview.", "Explain why the sky is blue in two sentences.", "Write a two-line poem about the moon.", "Name three primary colors.", "How do I boil an egg?", 'Translate "good morning" into Spanish.']
UNITS = {"p92-08": {"layer": 4, "unit": 5154}, "p92-37": {"layer": 5, "unit": 7218}, "p92-09": {"layer": 6, "unit": 4145}}
PAIRS = [("p92-08", "p92-37"), ("p92-08", "p92-09"), ("p92-37", "p92-09")]
DOSES = (0.6, 0.7, 0.8, 0.9, 1.0)


def cells():
    return [{"cell": f"{a}+{b}@{da:.1f}+{db:.1f}", "a": a, "b": b, "dose_a": da, "dose_b": db}
            for a, b in PAIRS for da in DOSES for db in DOSES]


def random_seed(cell_id, arm):
    raw = hashlib.sha256(f"134:{cell_id}:{arm}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def chat_ids(tok, question):
    result = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                    tokenize=True, return_tensors=None, enable_thinking=False)
    if isinstance(result, dict):
        result = result["input_ids"]
    if hasattr(result, "input_ids"):
        result = result.input_ids
    if hasattr(result, "tolist"):
        result = result.tolist()
    while result and isinstance(result[0], list):
        result = result[0]
    return [int(x) for x in result]


def generate_batch(model, tok, questions, pushes, batch_size=8, max_new_tokens=100, return_token_ids=False):
    import torch
    all_answers = []
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    eos = model.generation_config.eos_token_id
    eos_ids = [int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos]) if x is not None]
    sequences = [chat_ids(tok, q) for q in questions]
    for start in range(0, len(questions), batch_size):
        seqs = sequences[start:start + batch_size]
        width = max(map(len, seqs))
        ids = torch.tensor([[pad] * (width-len(s)) + s for s in seqs], dtype=torch.long, device=next(model.parameters()).device)
        mask = torch.tensor([[0] * (width-len(s)) + [1] * len(s) for s in seqs], dtype=torch.long, device=ids.device)
        handles = []
        for layer, row_vectors in pushes:
            vec = row_vectors[start:start + len(seqs)]
            handles.append(c.add_row_hook(layer, vec))
        try:
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=max_new_tokens,
                                     pad_token_id=pad, eos_token_id=eos_ids or None)
        finally:
            for handle in handles:
                handle.remove()
        for row in out:
            generated=row[width:]
            all_answers.append([int(token) for token in generated] if return_token_ids else
                               tok.decode(generated, skip_special_tokens=True).strip())
    return all_answers


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def append_jsonl_atomic(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the read/replace transaction so concurrent runs cannot overwrite
    # rows appended by another process between the read and os.replace calls.
    with path.with_name(path.name + ".lock").open("a+b") as lock:
        if fcntl:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        else:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        old = path.read_bytes() if path.exists() else b""
        payload = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(old); stream.write(payload); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name): os.unlink(name)
        if fcntl:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        else:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def read_rows(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_a770(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    if device == "xpu":
        import transformers.modeling_utils as mu
        mu.caching_allocator_warmup = lambda *a, **k: None
    model = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.bfloat16, local_files_only=True,
                                                  device_map={"": device})
    model.eval(); model.requires_grad_(False); torch.set_grad_enabled(False)
    return tok, model


def main(argv=None, *, model=None, tokenizer=None, out=None, pushes=None, neuron_meta=None, selection_file=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("pilot", "dev", "test"), required=True)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    p.add_argument("--out", default="results/part134")
    a = p.parse_args(argv)
    if a.batch_size < 1: p.error("batch size must be positive")
    directory = Path(out) if out is not None else Path(a.out)
    selection_file = Path(selection_file) if selection_file is not None else ROOT / "part134/selected134.json"
    if model is None:
        tokenizer, model = load_a770(a.device)
    tok = tokenizer
    import torch
    layer_modules = c.get_layers(model)
    if pushes is None:
        pushes = torch.load(ROOT / "results/part99-pushes.pt", map_location="cpu", weights_only=False)
    if neuron_meta is None:
        neuron_meta = UNITS
    device = next(model.parameters()).device
    # Gate 1 uses the same six-question batch shape used by development and pilot.
    gate_q = DEV_QUESTIONS
    before = generate_batch(model, tok, gate_q, [], a.batch_size, return_token_ids=True)
    probe_layer = layer_modules[int(neuron_meta["p92-08"]["layer"])]
    hidden_size = int(pushes["p92-08|neuron"].numel())
    zero = torch.zeros((len(gate_q), hidden_size), device=device,
                       dtype=next(model.parameters()).dtype)
    after = generate_batch(model, tok, gate_q, [(probe_layer, zero)], a.batch_size, return_token_ids=True)
    gate1 = before == after
    if not gate1:
        atomic_json(directory / f"{a.stage}-meta.json", {"stage": a.stage, "status": "gate1_failed",
                    "device": a.device, "batch_size": a.batch_size, "gate1_zero_hook_same_token_ids": False})
        raise RuntimeError("Gate 1 failed: zero-vector hook changed generated tokens")
    def neuron_vector(uid, dose):
        return (pushes[f"{uid}|neuron"].float() * float(dose)).to(device=device, dtype=next(model.parameters()).dtype)
    if a.stage == "pilot":
        todo = [{"cell": "pilot-p92-08+p92-37@1.0+1.0", "a": "p92-08", "b": "p92-37", "dose_a": 1., "dose_b": 1.}]
        qs = DEV_QUESTIONS; arms = ["pair"]
    elif a.stage == "dev":
        todo = cells(); qs = DEV_QUESTIONS; arms = ["pair"]
    else:
        selections = json.loads(selection_file.read_text(encoding="utf-8"))
        selected = selections.get("top_three", selections)
        if len(selected) < 3:
            raise ValueError(f"test stage requires all three registered selected cells; found {len(selected)}")
        todo = []
        for row in selected[:3]:
            pair, doses = row["pair"], row["doses"]
            todo.append({"cell": row["cell"], "a": pair[0], "b": pair[1], "dose_a": doses[0], "dose_b": doses[1]})
        qs = TEST_QUESTIONS; arms = ["pair", "a_only", "b_only", "a_random_1", "a_random_2", "b_random_1", "b_random_2"]
    answers_path = directory / f"{a.stage}-answers.jsonl"
    old = read_rows(answers_path); done = {(r["cell"], r["arm"]) for r in old}
    meta = {"stage": a.stage, "status": "running", "batch_size": a.batch_size, "device": a.device,
            "expected_cells": [x["cell"] for x in todo], "questions": qs, "expected_rows": len(todo)*len(arms)*len(qs),
            "started": time.time(), "gate1_zero_hook_same_answers": gate1}
    if a.stage == "test": meta["expected_rows"] += len(qs)
    atomic_json(directory / f"{a.stage}-meta.json", meta)
    for cell in todo:
        cid = cell["cell"]
        for arm in arms:
            if (cid, arm) in done: continue
            active = []
            if arm == "pair": active = [(cell["a"], cell["dose_a"]), (cell["b"], cell["dose_b"])]
            elif arm == "a_only": active = [(cell["a"], cell["dose_a"])]
            elif arm == "b_only": active = [(cell["b"], cell["dose_b"])]
            vectors = []
            for uid, dose in active:
                info = neuron_meta[uid]; vectors.append((layer_modules[int(info["layer"])], neuron_vector(uid, dose).unsqueeze(0).expand(len(qs), -1).contiguous()))
            if "_random_" in arm:
                random_arm = arm.startswith("a_")
                real_uid = cell["a"] if random_arm else cell["b"]
                random_uid = cell["b"] if random_arm else cell["a"]
                real_layer, random_layer = int(neuron_meta[real_uid]["layer"]), int(neuron_meta[random_uid]["layer"])
                real = neuron_vector(real_uid, cell["dose_a"] if random_arm else cell["dose_b"])
                replaced = neuron_vector(random_uid, cell["dose_b"] if random_arm else cell["dose_a"])
                # Deterministic same-layer Gaussian residual direction, norm matched to the replaced push.
                seed = random_seed(cid, arm)
                gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
                rnd = torch.randn(real.shape, generator=gen, dtype=torch.float32)
                rnd = rnd.to(device=device, dtype=torch.float32); rnd *= replaced.float().norm() / rnd.norm()
                vectors.extend([(layer_modules[real_layer], real.unsqueeze(0).expand(len(qs), -1).contiguous()),
                                (layer_modules[random_layer], rnd.to(real.dtype).unsqueeze(0).expand(len(qs), -1).contiguous())])
                random_path = directory / "random-vectors" / f"{cid}-{arm}.pt"; random_path.parent.mkdir(parents=True, exist_ok=True); torch.save({"seed": seed, "vector": rnd.detach().cpu(), "replaced_norm": float(replaced.float().norm().cpu())}, random_path)
            texts = generate_batch(model, tok, qs, vectors, a.batch_size)
            batch = [{"stage": a.stage, "cell": cid, "arm": arm, "pair": [cell["a"], cell["b"]],
                      "doses": [cell["dose_a"], cell["dose_b"]], "question_index": qi, "question": q, "answer": ans,
                      "answer_hash": hashlib.sha256((q + "\0" + ans).encode()).hexdigest()}
                     for qi, (q, ans) in enumerate(zip(qs, texts))]
            append_jsonl_atomic(answers_path, batch)
            print(f"completed {cid} {arm}", flush=True)
    if a.stage == "test" and ("unpushed", "none") not in done:
        texts = generate_batch(model, tok, qs, [], a.batch_size)
        batch = [{"stage": a.stage, "cell": "unpushed", "arm": "none", "pair": [], "doses": [],
                  "question_index": qi, "question": q, "answer": ans,
                  "answer_hash": hashlib.sha256((q + "\0" + ans).encode()).hexdigest()}
                 for qi, (q, ans) in enumerate(zip(qs, texts))]
        append_jsonl_atomic(answers_path, batch)
    current = read_rows(answers_path)
    meta.update({"status": "complete", "completed_rows": len(current), "completed_cells": sorted({r["cell"] for r in current}), "finished": time.time()})
    atomic_json(directory / f"{a.stage}-meta.json", meta)
    return current


if __name__ == "__main__": main()
