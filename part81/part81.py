#!/usr/bin/env python3
"""Part 81 calibration, pilot, five-condition generation, and blind chunking."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import neuron_report as nr
from part66 import part66
from part79 import part79 as part79_reference

MODEL = part79_reference.MODEL
REVISION = part79_reference.REVISION
TARGETS = ((19, 7839, -0.7), (17, 9209, 0.5))
CONDITION_IDS = ("k0", "k1", "k2", "k3", "k4")
PROMPT_FILE = Path(__file__).with_name("prompts.json")


def prompts(path=PROMPT_FILE):
    with Path(path).open(encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != 32 or len({r["id"] for r in rows}) != 32:
        raise ValueError("prompts.json must contain 32 unique prompt ids")
    if [r["kind"] for r in rows[:24]] != ["bait"] * 12 + ["ordinary"] * 12:
        raise ValueError("topic prompts must be ordered as 12 bait then 12 ordinary")
    if any(r["kind"] != "maker" for r in rows[24:]):
        raise ValueError("the final eight prompts must be maker prompts")
    return rows


def row_seed(condition, prompt_id, sample, phase="main"):
    raw = f"81\0{phase}\0{condition}\0{prompt_id}\0{sample}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def add_hook(unit, delta, stats=None):
    """Add a scalar to one MLP feature at every row and token position."""
    stats = stats if stats is not None else {}
    def hook(_module, args):
        x = args[0]
        y = x.clone()
        before = x[..., int(unit)].detach().float()
        y[..., int(unit)] = y[..., int(unit)] + float(delta)
        err = float(((y[..., int(unit)].detach().float() - before) - float(delta)).abs().max().item())
        stats["calls"] = stats.get("calls", 0) + 1
        stats["max_abs_diff"] = max(stats.get("max_abs_diff", 0.0), err)
        return (y,) + tuple(args[1:])
    return hook


def packed_windows(path, tok, limit=400):
    buf, windows = [], []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line).get("text", "")
            if not isinstance(doc, str):
                continue
            ids = tok(doc, add_special_tokens=False)["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            buf.extend(map(int, ids))
            buf.append(int(tok.eos_token_id))
            while len(buf) >= 1024 and len(windows) < limit:
                windows.append(buf[:1024])
                buf = buf[1024:]
            if len(windows) >= limit:
                break
    if len(windows) < limit:
        raise ValueError(f"Pile input yielded {len(windows)} windows; {limit} required")
    return windows


def summarize_norms(model, layers, windows, batch=4):
    import numpy as np
    import torch
    captured = {17: [], 19: []}
    handles = []
    for li in captured:
        def hook(_module, _inputs, output, li=li):
            x = output[0] if isinstance(output, (tuple, list)) else output
            captured[li].append(torch.linalg.vector_norm(x.detach().float(), dim=-1)[:, 1:].reshape(-1).cpu())
        handles.append(layers[li].register_forward_hook(hook))
    try:
        with torch.no_grad():
            for i in range(0, len(windows), batch):
                ids = torch.tensor(windows[i:i + batch], dtype=torch.long, device=next(model.parameters()).device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    result = {}
    for li, parts in captured.items():
        a = torch.cat(parts).numpy()
        q = np.quantile(a, [0.1, 0.5, 0.9], method="linear")
        result[f"L{li}"] = {"p10": float(q[0]), "median": float(q[1]), "p90": float(q[2]),
                            "token_count": int(a.size), "position_start": 1}
    return result


def column_norms(layers, layer, width=None):
    import torch
    w = layers[layer].mlp.down_proj.weight.detach().float()
    if w.ndim != 2:
        raise ValueError("down_proj weight must be a matrix")
    if width is not None and w.shape[1] != width:
        raise ValueError(f"expected width {width}, found {w.shape[1]}")
    return torch.linalg.vector_norm(w, dim=0).cpu().tolist()


def draw_control_units(norms_by_layer, targets=TARGETS, seed=81, n=3):
    """Draw three unique norm-matched controls per target layer, in layer order."""
    rng = random.Random(seed)
    selected = {}
    target_units = {li: unit for li, unit, _dose in targets}
    for li, unit, _dose in targets:
        norms = norms_by_layer[li] if li in norms_by_layer else norms_by_layer[f"L{li}"]
        target_norm = float(norms[unit])
        if target_norm <= 0:
            raise ValueError(f"target L{li} N{unit} has nonpositive column norm")
        pool = [i for i, norm in enumerate(norms) if i != target_units[li] and norm > 0
                and abs(float(norm) / target_norm - 1.0) <= 0.25]
        already = set(selected.get(li, []))
        pool = [i for i in pool if i not in already]
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} eligible controls for L{li}; need {n}")
        selected[li] = selected.get(li, []) + rng.sample(pool, n)
    return [(selected[19][i], selected[17][i]) for i in range(n)]


def compute_deltas(layers, medians, controls):
    deltas = {}
    units = [(19, 7839, -0.7), (17, 9209, 0.5)]
    for a, b in controls:
        units.extend(((19, a, -0.7), (17, b, 0.5)))
    for li, unit, dose in units:
        w = layers[li].mlp.down_proj.weight.detach().float()
        if unit < 0 or unit >= w.shape[1]:
            raise ValueError(f"unit {unit} outside layer {li} width {w.shape[1]}")
        norm = float(__import__("torch").linalg.vector_norm(w[:, unit]).item())
        if norm <= 0:
            raise ValueError(f"zero column norm at L{li} N{unit}")
        median = float(medians[f"L{li}"]["median"])
        deltas[f"L{li}_N{unit}"] = {"layer": li, "unit": unit, "dose": dose,
                                    "column_norm": norm, "median": median, "delta": dose * median / norm}
    return deltas


def build_conditions(deltas, controls):
    target = [deltas["L19_N7839"], deltas["L17_N9209"]]
    result = [{"id": "k0", "push": []}, {"id": "k1", "push": target}]
    for i, (u19, u17) in enumerate(controls, 2):
        result.append({"id": f"k{i}", "push": [deltas[f"L19_N{u19}"], deltas[f"L17_N{u17}"]]})
    return result


def run_step0(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    windows = packed_windows(args.pile, tok, args.windows)
    medians = summarize_norms(model, layers, windows, args.batch)
    meta = {"model": MODEL, "revision": REVISION, "dtype": "bfloat16", "windows": len(windows),
            "window_tokens": 1024, "positions_used": "1..1023", "medians": medians,
            "pile": str(args.pile)}
    (out / "step0.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def load_step0(path):
    with Path(path).open(encoding="utf-8") as f:
        meta = json.load(f)
    if (meta.get("model"), meta.get("revision")) != (MODEL, REVISION):
        raise ValueError("step0 model/revision does not match the pinned Part 79 model")
    for li in (17, 19):
        if meta.get("medians", {}).get(f"L{li}", {}).get("median") is None:
            raise ValueError(f"step0.json lacks L{li} median")
    return meta


def pilot(args):
    import torch
    rows = prompts()
    topics = [r for r in rows if r["kind"] == "bait"][:4]
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    meta = load_step0(args.step0_file)
    medians = meta["medians"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = []
    for condition in ("k0", "k1"):
        for qi, row in enumerate(topics):
            ids = tok(nr.build_prompt(tok, row["text"], False), return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
            hooks = []
            if condition == "k1":
                for li, unit, dose in TARGETS:
                    norm = float(torch.linalg.vector_norm(layers[li].mlp.down_proj.weight.detach().float()[:, unit]).item())
                    delta = dose * float(medians[f"L{li}"]["median"]) / norm
                    hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(add_hook(unit, delta)))
            seed = row_seed(condition, row["id"], 0, "pilot")
            torch.manual_seed(seed)
            try:
                with torch.no_grad():
                    gen = model.generate(input_ids=ids, max_new_tokens=96, do_sample=False, use_cache=True,
                                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            finally:
                for h in hooks: h.remove()
            raw = gen[0, ids.shape[1]:].tolist()
            answers.append({"condition": condition, "prompt_id": row["id"], "kind": row["kind"], "prompt": row["text"],
                            "sample": 0, "seed": seed, "text": tok.decode(raw, skip_special_tokens=True).strip()})
    path = out / "pilot_answers.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in answers), encoding="utf-8")
    print(f"pilot rows: {len(answers)}; labels required in --pilot-labels for main-run gate")


def main_run(args):
    import torch
    meta = load_step0(args.step0_file)
    if not args.pilot_labels:
        raise ValueError("main run requires --pilot-labels; the pre-registered pilot gate must pass")
    with Path(args.pilot_labels).open(encoding="utf-8") as f:
        pilot_labels = json.load(f)
    if pilot_labels.get("k1_conspiratorial", 0) < 2 or pilot_labels.get("k0_conspiratorial", 0) != 0:
        raise ValueError("pilot gate failed: need k1 conspiratorial >=2/4 and k0 conspiratorial 0/4")
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    norms = {li: column_norms(layers, li) for li in (17, 19)}
    controls = draw_control_units(norms)
    deltas = compute_deltas(layers, meta["medians"], controls)
    conditions = build_conditions(deltas, controls)
    rows = prompts()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answer_path = out / "answers.jsonl"
    hook_evidence = []
    started = time.time()
    with answer_path.open("w", encoding="utf-8") as sink:
        for cond in conditions:
            for row in rows:
                rendered = nr.build_prompt(tok, row["text"], False)
                for sample in range(4):
                    seed = row_seed(cond["id"], row["id"], sample)
                    torch.manual_seed(seed)
                    if args.device == "xpu" and hasattr(torch, "xpu"):
                        torch.xpu.manual_seed_all(seed)
                    ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
                    stats, handles = [], []
                    for push in cond["push"]:
                        st = {"layer": push["layer"], "unit": push["unit"], "delta": push["delta"], "calls": 0, "max_abs_diff": 0.0}
                        stats.append(st)
                        handles.append(layers[push["layer"]].mlp.down_proj.register_forward_pre_hook(add_hook(push["unit"], push["delta"], st)))
                    try:
                        with torch.no_grad():
                            gen = model.generate(input_ids=ids, max_new_tokens=96, do_sample=True, temperature=0.7,
                                                 top_p=0.95, use_cache=True, pad_token_id=tok.pad_token_id,
                                                 eos_token_id=tok.eos_token_id)
                    finally:
                        for h in handles: h.remove()
                    raw = gen[0, ids.shape[1]:].tolist()
                    if tok.eos_token_id is not None and int(tok.eos_token_id) in raw:
                        raw = raw[:raw.index(int(tok.eos_token_id)) + 1]
                    sink.write(json.dumps({"condition": cond["id"], "prompt_id": row["id"], "kind": row["kind"],
                                           "prompt": row["text"], "sample": sample, "seed": seed,
                                           "text": tok.decode(raw, skip_special_tokens=True).strip()}, ensure_ascii=False) + "\n")
                    sink.flush()
                    hook_evidence.append({"condition": cond["id"], "prompt_id": row["id"], "sample": sample, "hooks": stats})
                    nr.spill_guard(args.device, args.max_gpu_gb)
    run_meta = {"model": MODEL, "revision": REVISION, "dtype": "bfloat16", "device": args.device,
                "sampling": {"samples": 4, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 96},
                "control_seed": 81, "controls": [{"k": f"k{i + 2}", "L19": pair[0], "L17": pair[1]} for i, pair in enumerate(controls)],
                "deltas": deltas, "conditions": conditions, "pilot_labels": pilot_labels,
                "requested_rows": 640, "wall_time_seconds": time.time() - started, "hook_evidence": hook_evidence}
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")


def make_reader_chunks(answer_path, out_dir, seed=81, chunk_size=107):
    rows = []
    with Path(answer_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if len(rows) != 640:
        raise ValueError(f"expected 640 answers, found {len(rows)}")
    random.Random(seed).shuffle(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    for i, row in enumerate(rows):
        rid = f"r{i + 1:04d}"
        key[rid] = {"condition": row["condition"], "prompt_id": row["prompt_id"],
                    "sample": row["sample"], "kind": row["kind"]}
        row["id"] = rid
        row.pop("condition")
        row.pop("sample")
        row.pop("seed", None)
    chunks = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        path = out / f"reader_chunk_{len(chunks) + 1:02d}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk), encoding="utf-8")
        chunks.append(str(path))
    (out / "reader_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return chunks, key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step0", action="store_true", help="run the Part 79 style calibration")
    ap.add_argument("--pilot", action="store_true", help="run the eight-answer greedy pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step0-file", default="results/part81/step0/step0.json")
    ap.add_argument("--pilot-labels")
    ap.add_argument("--pile", default="qwen35/pile10k.jsonl")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    ap.add_argument("--max-gpu-gb", type=float, default=20)
    ap.add_argument("--make-reader-chunks", action="store_true")
    ap.add_argument("--answers", default="results/part81/main/answers.jsonl")
    args = ap.parse_args()
    if args.step0 and args.pilot: ap.error("choose one mode")
    if args.step0: run_step0(args)
    elif args.pilot: pilot(args)
    elif args.make_reader_chunks: print(json.dumps(make_reader_chunks(args.answers, args.out), indent=2))
    else: main_run(args)


if __name__ == "__main__":
    main()
