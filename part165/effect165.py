#!/usr/bin/env python3
"""Measure first pushed-token effects for Part 165's registered directions."""
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

from part165.run165 import build_directions, pushes_for, all_rows, questions, LAYERS, SLOTS, ARMS
from part165 import run165
from part155 import run155 as part155

MODEL_DEFAULT = run165.MODEL_DEFAULT
BATCH_SIZE = 64


def _tuple_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def effect_values(plain_scores, pushed_scores, plain_hidden, pushed_hidden, plain_second, pushed_second):
    """Compute CPU float32 KL, relative last-layer change, and top-token agreement."""
    import torch
    p_logits = plain_scores.detach().to(device="cpu", dtype=torch.float32)
    q_logits = pushed_scores.detach().to(device="cpu", dtype=torch.float32)
    p_log = torch.log_softmax(p_logits, dim=-1)
    p = p_log.exp()
    q_log = torch.log_softmax(q_logits, dim=-1)
    kl = (p * (p_log - q_log)).sum(dim=-1)
    p_h = plain_hidden.detach().to(device="cpu", dtype=torch.float32)
    q_h = pushed_hidden.detach().to(device="cpu", dtype=torch.float32)
    rel = torch.linalg.vector_norm(q_h - p_h, dim=-1) / torch.linalg.vector_norm(p_h, dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    top_same = pushed_second.detach().to(device="cpu").eq(plain_second.detach().to(device="cpu"))
    return kl.tolist(), rel.tolist(), top_same.tolist()


def generate_capture(model, tok, layers, question, layer_pushes, device, common, row_layers=None):
    """Generate exactly two greedy tokens and retain second-step scores and final-layer state."""
    import torch
    batch = int(next(iter(layer_pushes.values())).shape[0]) if layer_pushes else 1
    inputs = common.token_inputs(tok, question, batch, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    captured = []

    def capture(_module, _args, output):
        value = _tuple_tensor(output)
        # The prefill has prompt positions; a one-position output is the decode state.
        if hasattr(value, "ndim") and value.ndim == 3 and int(value.shape[-2]) == 1:
            captured.append(value[:, -1, :].detach())

    handle = layers[-1].register_forward_hook(capture)
    realized = {}
    hooks = []
    row_layers = row_layers or {}
    for layer_no, push in layer_pushes.items():
        down_proj = layers[layer_no].mlp.down_proj
        before = {}
        mask = torch.tensor([int(row_layers.get(i, layer_no)) == int(layer_no)
                             for i in range(int(push.shape[0]))], dtype=torch.bool, device=push.device)
        def capture_input(_module, args, key=layer_no, held=before):
            value = args[0]
            if value.ndim == 3 and value.shape[-2] == 1 and "x" not in held:
                held["x"] = value.detach()
        input_handle = down_proj.register_forward_pre_hook(capture_input)
        hook = common.AddRowsHook(down_proj, push)
        def capture_pushed(_module, args, key=layer_no, held=before, row_mask=mask, weight=down_proj.weight,
                           row_push=push):
            pushed_value = args[0]
            if pushed_value.ndim != 3 or pushed_value.shape[-2] != 1 or "x" not in held:
                return
            x = held.pop("x")
            delta = pushed_value[:, 0, :].float() - x[:, 0, :].float()
            row_index = torch.nonzero(row_mask, as_tuple=False).flatten()
            if row_index.numel() == 0:
                return
            selected_delta = delta.index_select(0, row_index)
            selected_push = row_push.to(device=delta.device, dtype=torch.float32).index_select(0, row_index)
            w = weight.detach().to(device=delta.device, dtype=torch.float32)
            actual = selected_delta @ w.T
            intended = selected_push @ w.T
            actual_norm = torch.linalg.vector_norm(actual, dim=1)
            intended_norm = torch.linalg.vector_norm(intended, dim=1)
            cosine = (actual * intended).sum(dim=1) / (actual_norm * intended_norm).clamp_min(torch.finfo(torch.float32).tiny)
            changed = (selected_delta.abs() > 1e-6).sum(dim=1)
            # One transfer per column: an .item() per value in a hook hung an A770 job (gotcha, 10-24).
            stats = torch.stack((actual_norm, cosine, changed.to(actual_norm.dtype), intended_norm), dim=1).cpu().tolist()
            for row_no, (norm, cos, count, target) in zip(row_index.detach().cpu().tolist(), stats):
                realized[row_no] = {"realized_write_norm": float(norm), "realized_cos": float(cos),
                                    "changed_channels": int(count), "intended_write_norm": float(target)}
        after_handle = down_proj.register_forward_pre_hook(capture_pushed)
        hooks.append((hook, input_handle, after_handle))
    for hook, _before, _after in hooks:
        hook.begin_sequence()
    try:
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False,
                                    use_cache=True, pad_token_id=getattr(tok, "pad_token_id", None),
                                    eos_token_id=common.eos_ids(tok, model) or None,
                                    output_scores=True, return_dict_in_generate=True)
    finally:
        handle.remove()
        for hook, before, after in hooks:
            hook.remove()
            before.remove()
            after.remove()
    if len(result.scores) != 2 or len(captured) < 1:
        raise RuntimeError(f"expected two scores and a decode activation; got {len(result.scores)} scores, {len(captured)} activations")
    first = result.sequences[:, prompt_len].detach()
    second = result.sequences[:, prompt_len + 1].detach()
    return {"first": first, "second": second, "scores": result.scores[1].detach(),
            "hidden": captured[-1], "realized": realized}


def _column_cosine_max(weight, write, chunk_size=256):
    import torch
    write_norm = torch.linalg.vector_norm(write)
    col_norms = torch.linalg.vector_norm(weight, dim=0)
    best = torch.tensor(0.0, dtype=torch.float32)
    for start in range(0, int(weight.shape[1]), chunk_size):
        columns = weight[:, start:start + chunk_size]
        cosine = (columns.T @ write).abs() / (col_norms[start:start + chunk_size] * write_norm).clamp_min(torch.finfo(torch.float32).tiny)
        best = torch.maximum(best, cosine.max())
    return float(best.item())


def geometry_for_layer(weight, layer_no, directions, metrics):
    import torch
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    singular = torch.linalg.svdvals(w)
    tolerance = max(w.shape) * torch.finfo(torch.float32).eps * float(singular.max().item())
    direction_rows = []
    iso_fraction = None
    for (layer, slot), arms in directions.items():
        if layer != layer_no:
            continue
        for arm, d in arms.items():
            d_cpu = d.detach().to(device="cpu", dtype=torch.float32)
            write = w @ d_cpu
            direction_rows.append({"layer": layer, "slot": slot, "arm": arm,
                                   "norm_d": float(torch.linalg.vector_norm(d_cpu).item()),
                                   "largest_column_cosine": _column_cosine_max(w, write)})
            if arm == "iso":
                iso_fraction = 1.0 - float(metrics[(layer, slot)][arm]["iso_residual_fraction"])
    return {"layer": int(layer_no), "rank": int((singular > tolerance).sum().item()),
            "rank_tolerance": tolerance, "smallest_singular_value": float(singular.min().item()),
            "largest_singular_value": float(singular.max().item()),
            "directions": direction_rows, "iso_reconstruction_fraction": iso_fraction}


def row_key(row):
    return run165.row_key(row)


def read_existing(path):
    return run165.read_jsonl(path)


def rows_for_question(question_index, widths):
    rows = list(run165.all_rows(widths))
    random.Random(165 + int(question_index)).shuffle(rows)
    return [{**row, "stage": "effect", "question_index": int(question_index)} for row in rows]


def pending_rows(rows, done):
    return [row for row in rows if row_key(row) not in done]


def pilot_rate(rows, elapsed):
    return int(rows) / max(float(elapsed), 1e-9)


def run(args, loader=None, capture_fn=None):
    import torch
    common = part155._common()
    medians_path = ROOT / "results/part112/medians.json"
    medians = json.loads(medians_path.read_text(encoding="utf-8"))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(output)
    done = {row_key(row) for row in existing}
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    directions, direction_metrics, pinv_cache = {}, {}, {}
    geometry_layers = []
    for layer_no in LAYERS:
        weight = layers[layer_no].mlp.down_proj.weight
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        for slot in range(SLOTS):
            vectors, _metrics = build_directions(weight, layer_no, slot, median, pinv_cache)
            directions[(layer_no, slot)] = vectors
            direction_metrics[(layer_no, slot)] = _metrics
        geometry_layers.append(geometry_for_layer(weight, layer_no, directions, direction_metrics))
    geometry_path = ROOT / "results/part165/geometry.json"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(json.dumps({"part": 165, "dtype": "float32", "layers": geometry_layers},
                                        ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    capture_fn = capture_fn or (lambda *a, **kw: generate_capture(*a, common=common, **kw))
    selected_questions = questions()
    written = 0
    pilot_started = time.perf_counter()
    pilot_rows = pilot_batches = 0
    with output.open("a", encoding="utf-8", newline="\n") as fh:
        for qi, item in enumerate(selected_questions):
            plain_info = {"stage": "effect", "arm": "plain", "slot": None, "layer": None,
                          "question_index": qi}
            plain = capture_fn(model, tok, layers, item["question"], {}, args.device)
            widths = {layer_no: int(layers[layer_no].mlp.down_proj.weight.shape[1]) for layer_no in LAYERS}
            shuffled = rows_for_question(qi, widths)
            for start in range(0, len(shuffled), args.batch_size):
                batch = shuffled[start:start + args.batch_size]
                chunk = pending_rows(batch, done)
                if not chunk:
                    continue
                row_pushes = pushes_for(chunk, layers, directions)
                pushed = capture_fn(model, tok, layers, item["question"], row_pushes, args.device,
                                    row_layers={i: int(row["layer"]) for i, row in enumerate(chunk)})
                first_flags = pushed["first"].detach().to("cpu").ne(plain["first"].detach().to("cpu")).tolist()
                kl, rel, same = effect_values(plain["scores"], pushed["scores"], plain["hidden"],
                                              pushed["hidden"], plain["second"], pushed["second"])
                if len(chunk) != len(kl):
                    raise RuntimeError("captured effect count does not match batch rows")
                for index, row in enumerate(chunk):
                    # A resume may leave an incomplete subset of the shuffled batch; pushes/capture must align.
                    fh.write(json.dumps({**row, **pushed["realized"][index], "kl": float(kl[index]), "rel_change": float(rel[index]),
                                         "top1_same": bool(same[index]),
                                         "first_token_diff": bool(first_flags[index])}, ensure_ascii=True) + "\n")
                    done.add(row_key(row))
                    written += 1
                fh.flush()
                pilot_rows += len(chunk)
                pilot_batches += 1
                if pilot_batches == 3:
                    rate = pilot_rate(pilot_rows, time.perf_counter() - pilot_started)
                    print(f"first 3 effect batches: {rate:.2f} rows/s", flush=True)
                    if rate < 2.0:
                        raise SystemExit(f"Pilot stop rule: {rate:.2f} rows/s is below 2.0")
    return {"rows_written": written, "questions": len(selected_questions), "batch_size": args.batch_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out", default=str(ROOT / "results/part165/effects.jsonl"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.batch_size != 64:
        parser.error("registered batch size is 64")
    print(json.dumps(run(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
