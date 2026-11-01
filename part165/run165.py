#!/usr/bin/env python3
"""Part 165 Arc runner for paired, nested MLP pushes."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part160 import run160 as part160  # noqa: E402
from part164 import run164 as part164  # noqa: E402
from part155 import run155 as part155  # noqa: E402

MODEL_DEFAULT = part160.MODEL_DEFAULT
MAX_NEW_TOKENS = 80
LAYERS = tuple(range(19, 27))
SLOTS = 12
ARMS = ("n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
K_ARMS = {"k4": 4, "k16": 16, "k64": 64, "k256": 256, "k1024": 1024}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def row_key(row):
    return (row.get("stage"), row.get("arm"), row.get("slot"), row.get("layer"),
            int(row.get("question_index", -1)))


def questions():
    items = part164.choose_questions()
    if len(items) != 16 or any(item.get("kind") != "maker" for item in items):
        raise ValueError("Part 164 must provide 16 maker phrasings")
    return items[:8]


def _seeded_raw(weight, layer_no, slot, torch):
    """Draw the registered permutation and both Gaussian vectors, in that order."""
    width = int(weight.shape[1])
    hidden = int(weight.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(165000 + 100 * int(layer_no) + int(slot))
    permutation = torch.randperm(width, generator=generator, device="cpu")
    mlp_gaussian = torch.randn(width, generator=generator, dtype=torch.float32, device="cpu")
    hidden_gaussian = torch.randn(hidden, generator=generator, dtype=torch.float32, device="cpu")
    return permutation, mlp_gaussian, hidden_gaussian


def build_directions(weight, layer_no, slot, median, pinv_cache=None):
    """Return normalized arm pushes and their write/residual measurements for one slot."""
    import torch
    cpu_weight = weight.detach().to(device="cpu", dtype=torch.float32)
    width = int(cpu_weight.shape[1])
    permutation, gaussian, hidden_gaussian = _seeded_raw(cpu_weight, layer_no, slot, torch)
    if pinv_cache is None:
        pinv_cache = {}
    if int(layer_no) not in pinv_cache:
        pinv_cache[int(layer_no)] = torch.linalg.pinv(cpu_weight)
    pinv = pinv_cache[int(layer_no)]
    first = int(permutation[0].item())
    raw = {
        "n+": torch.zeros(width, dtype=torch.float32),
        "n±": torch.zeros(width, dtype=torch.float32),
        "full": gaussian.clone(),
        "iso": pinv @ hidden_gaussian,
    }
    raw["n+"][first] = 1.0
    raw["n±"][first] = 1.0 if float(gaussian[first].item()) >= 0 else -1.0
    for arm, count in K_ARMS.items():
        vector = torch.zeros(width, dtype=torch.float32)
        indices = permutation[:count]
        vector[indices] = gaussian[indices]
        raw[arm] = vector

    target = 0.8 * float(median)
    directions, metrics = {}, {}
    for arm in ARMS:
        vector = raw[arm]
        write = cpu_weight @ vector
        write_norm = float(torch.linalg.vector_norm(write).item())
        if not write_norm > 0:
            raise ValueError(f"zero write norm for layer {layer_no} slot {slot} arm {arm}")
        scaled = vector * (target / write_norm)
        scaled_write = cpu_weight @ scaled
        scaled_norm = float(torch.linalg.vector_norm(scaled_write).item())
        residual = 0.0
        if arm == "iso":
            scaled_hidden = hidden_gaussian * (target / write_norm)
            residual = float(torch.linalg.vector_norm(scaled_write - scaled_hidden).item())
        directions[arm] = scaled
        metrics[arm] = {"write_norm": scaled_norm, "target_norm": target,
                        "relative_error": abs(scaled_norm - target) / target,
                        "iso_residual_norm": residual,
                        "iso_residual_fraction": residual / scaled_norm if arm == "iso" else None}
    return directions, metrics


def pushes_for(rows, layers, directions):
    """Build Part 163-shaped dense per-layer rows on each weight's device."""
    import torch
    out = {}
    for layer_no in sorted({int(row["layer"]) for row in rows}):
        weight = layers[layer_no].mlp.down_proj.weight
        dense = torch.zeros((len(rows), int(weight.shape[1])), dtype=torch.float32, device=weight.device)
        for row_no, row in enumerate(rows):
            if int(row["layer"]) == layer_no:
                slot_key = (layer_no, int(row["slot"]))
                dense[row_no] = directions[slot_key][row["arm"]].to(device=weight.device, dtype=torch.float32)
        out[layer_no] = dense
    return out


def all_rows(widths):
    rows = []
    for layer_no in LAYERS:
        for slot in range(SLOTS):
            for arm in ARMS:
                k = 1 if arm in ("n+", "n±") else K_ARMS.get(arm)
                if arm == "full":
                    k = int(widths[layer_no])
                rows.append({"stage": "battery", "arm": arm, "slot": slot,
                             "layer": layer_no, "k": k})
    return rows


def _write_norms(path, measurements):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"part": 165, "dose": 0.8, "directions": measurements},
                               ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def run(args, loader=None, generate_fn=None):
    import torch
    common = part155._common()
    medians_path = ROOT / "results/part112/medians.json"
    medians = json.loads(medians_path.read_text(encoding="utf-8"))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output)
    done = {row_key(row) for row in existing}
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    directions, measurements, pinv_cache = {}, [], {}
    for layer_no in LAYERS:
        weight = layers[layer_no].mlp.down_proj.weight
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        for slot in range(SLOTS):
            vectors, metrics = build_directions(weight, layer_no, slot, median, pinv_cache)
            directions[(layer_no, slot)] = vectors
            for arm in ARMS:
                measurements.append({"layer": layer_no, "slot": slot, "arm": arm, **metrics[arm]})
    norms_path = Path(args.norms) if args.norms else output.parent / "norms.json"
    _write_norms(norms_path, measurements)
    generate_fn = generate_fn or (lambda *a: part160.generate_checked(*a, common=common))
    selected_questions = questions()
    written = 0

    def append(fh, info, answer, question, kind):
        nonlocal written
        if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
            raise RuntimeError(f"empty answer for {info['stage']} {info['arm']}")
        row = {**info, "question_index": info["question_index"], "question": question,
               "kind": kind, "answer": answer["answer"], "token_count": int(answer.get("token_count", 0))}
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        done.add(row_key(row))
        written += 1

    pilot_started = None
    pilot_answers = 0
    pilot_batches = 0
    started = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="\n") as fh:
        for qi, item in enumerate(selected_questions):
            info = {"stage": "no_push", "arm": "plain", "slot": None, "layer": None,
                    "k": None, "question_index": qi}
            if row_key(info) in done:
                continue
            answers = generate_fn(model, tok, layers, item["question"], {}, MAX_NEW_TOKENS, args.device)
            if len(answers) != 1:
                raise RuntimeError(f"no-push returned {len(answers)} answers; expected one")
            append(fh, info, answers[0], item["question"], item["kind"])
            fh.flush()
        pilot_started = time.perf_counter()

        widths = {layer_no: int(layers[layer_no].mlp.down_proj.weight.shape[1]) for layer_no in LAYERS}
        base = all_rows(widths)
        for qi, item in enumerate(selected_questions):
            shuffled = list(base)
            random.Random(165 + qi).shuffle(shuffled)
            for start in range(0, len(shuffled), args.batch_size):
                boundary_batch = shuffled[start:start + args.batch_size]
                chunk = [{**row, "question_index": qi} for row in boundary_batch
                         if row_key({**row, "question_index": qi}) not in done]
                if not chunk:
                    continue
                pushes = pushes_for(chunk, layers, directions)
                answers = generate_fn(model, tok, layers, item["question"], pushes,
                                      MAX_NEW_TOKENS, args.device)
                if len(answers) != len(chunk):
                    raise RuntimeError(f"generated {len(answers)} answers for {len(chunk)} battery rows")
                for info, answer in zip(chunk, answers):
                    append(fh, info, answer, item["question"], item["kind"])
                fh.flush()
                pilot_answers += len(chunk)
                pilot_batches += 1
                if pilot_batches == 3:
                    rate = pilot_answers / max(time.perf_counter() - pilot_started, 1e-9)
                    print(f"first 3 battery batches: {rate:.2f} answers/s", flush=True)
                    if rate < 1.5:
                        raise SystemExit(f"Pilot stop rule: {rate:.2f} answers/s is below 1.5")
    return {"answers_written": written, "questions": len(selected_questions),
            "batch_size": args.batch_size, "seconds": time.perf_counter() - started,
            "norms": str(norms_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out", default=str(ROOT / "results/part165/answers.jsonl"))
    parser.add_argument("--norms", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.batch_size != 64:
        parser.error("registered batch size is 64")
    print(json.dumps(run(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
