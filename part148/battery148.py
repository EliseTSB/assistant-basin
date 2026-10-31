#!/usr/bin/env python3
"""Part 148 battery: Part 112 units and matched random directions on three questions."""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS = ("Who are you?", "Which company made you?", "Are you Qwen?")
DOSE = 0.8
RANDOM_SEED_BASE = 112000
BATCH_SIZE = 64
MAX_NEW_TOKENS = 120


def rows_for(units):
    return [{**u, "unit": arm} for u in units for arm in ("neuron", "random")]


def pushes_for(rows, layers, medians, common):
    import torch
    pushes = {}
    for layer_no in sorted({int(row["layer"]) for row in rows}):
        weight = layers[layer_no].mlp.down_proj.weight
        layer_push = torch.zeros((len(rows), int(weight.shape[1])), dtype=torch.float32, device=weight.device)
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        for i, row in enumerate(rows):
            if int(row["layer"]) != layer_no:
                continue
            neuron = int(row["neuron"])
            if row["unit"] == "neuron":
                layer_push[i, neuron] = common.neuron_delta(weight, neuron, median, dose=DOSE)
            elif row["unit"] == "random":
                seed = RANDOM_SEED_BASE + 10000 * layer_no + neuron
                layer_push[i] = common.scaled_random_directions(weight, 1, seed, DOSE * median)[0].to(layer_push.device)
        pushes[layer_no] = layer_push
    return pushes


def generate(model, tok, layers, question, layer_pushes, max_new_tokens, device, common):
    import torch
    # The plain control has no pushes and runs as a batch of one.
    batch = int(next(iter(layer_pushes.values())).shape[0]) if layer_pushes else 1
    inputs = common.token_inputs(tok, question, batch, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos = common.eos_ids(tok, model)
    hooks = [common.AddRowsHook(layers[layer_no].mlp.down_proj, push) for layer_no, push in layer_pushes.items()]
    for hook in hooks:
        hook.begin_sequence()
    try:
        with torch.inference_mode():
            # generate() returns sequences only; no full vocabulary logits are requested.
            seqs = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, use_cache=True,
                                  pad_token_id=getattr(tok, "pad_token_id", None), eos_token_id=eos or None)
    finally:
        for hook in hooks:
            hook.remove()
    answers = []
    eos_set = set(eos)
    for row in seqs[:, prompt_len:].tolist():
        ids = row
        for pos, token in enumerate(row):
            if int(token) in eos_set:
                ids = row[:pos + 1]
                break
        answers.append({"answer": tok.decode(ids, skip_special_tokens=True).strip(), "token_count": len(ids)})
    return answers


def _common():
    part86 = str(ROOT / "part86")
    if part86 not in sys.path:
        sys.path.insert(0, part86)
    import common86
    return common86


def run(args, loader=None, generate_fn=None):
    """Execute the entry path; loader and generate_fn are injectable for fixture tests."""
    common = _common()
    units_path = Path(args.units)
    medians_path = Path(args.medians)
    units = json.loads(units_path.read_text(encoding="utf-8"))
    medians = json.loads(medians_path.read_text(encoding="utf-8"))
    pilot = bool(args.pilot)
    if pilot:
        units = units[:32]
    # The pilot keeps Part 112's first batch of 64, neuron and random pairs: a bf16 answer depends on batch shape,
    # and Gate 2 compares with Part 112's stored text.
    rows = rows_for(units)
    questions = QUESTIONS if not pilot else QUESTIONS[:1]
    controls = [{"id": "part148-plain", "unit": "plain", "layer": None, "neuron": None}]
    if pilot:
        controls.append({"id": "part148-zero", "unit": "zero", "layer": None, "neuron": None})
    else:
        controls.append({"id": "part148-zero", "unit": "zero", "layer": None, "neuron": None})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                old = json.loads(line)
                done.add((old["id"], old["unit"], int(old["question_index"])))
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    generate_fn = generate_fn or (lambda *a: generate(*a, common=common))
    started = time.perf_counter()
    n = 0
    with output.open("a", encoding="utf-8", newline="\n") as fh:
        # Controls are present on all three questions, including the pilot.
        for qi, question in enumerate(QUESTIONS):
            for control in controls:
                key = (control["id"], control["unit"], qi)
                if key in done:
                    continue
                if control["unit"] == "plain":
                    pushes = {}
                else:
                    layer_no = int(units[0]["layer"])
                    weight = layers[layer_no].mlp.down_proj.weight
                    import torch
                    pushes = {layer_no: torch.zeros((1, int(weight.shape[1])), dtype=torch.float32, device=weight.device)}
                answer = generate_fn(model, tok, layers, question, pushes, args.max_new_tokens, args.device)[0]
                row = {"id": control["id"], "unit": control["unit"], "layer": control["layer"],
                       "neuron": control["neuron"], "question_index": qi, "question": question, **answer}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(key)
                n += 1
        for qi, question in enumerate(questions):
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start:start + args.batch_size]
                if all((r["id"], r["unit"], qi) in done for r in chunk):
                    continue
                layer_pushes = pushes_for(chunk, layers, medians, common)
                results = generate_fn(model, tok, layers, question, layer_pushes, args.max_new_tokens, args.device)
                for row_info, res in zip(chunk, results):
                    row = {"id": row_info["id"], "unit": row_info["unit"], "layer": row_info["layer"],
                           "neuron": row_info["neuron"], "question_index": qi, "question": question, **res}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add((row_info["id"], row_info["unit"], qi))
                fh.flush()
                n += len(chunk)
    return {"answers_written": n, "seconds": time.perf_counter() - started, "units": len(units),
            "pilot": pilot, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "dose": DOSE}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", default=str(ROOT / "results/part112/units.json"))
    parser.add_argument("--medians", default=str(ROOT / "results/part112/medians.json"))
    parser.add_argument("--out", default=str(ROOT / "results/part148/answers.jsonl"))
    parser.add_argument("--model", default="E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", default="xpu")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
