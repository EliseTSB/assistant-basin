#!/usr/bin/env python3
"""Gate 3: verify Part 180's residual push hook placement on fixed token ids."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART92 = ROOT / "part92"
if str(PART92) not in sys.path:
    sys.path.insert(0, str(PART92))
# Under python -I the script's own folder is not on sys.path, so `import score180` fails on the Arc job runner
# (pilot 2, 2026-11-03 18:58); tests passed because pytest adds the folder.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common92 as c  # noqa: E402
import score180  # noqa: E402


def _hidden(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output, value):
    if isinstance(output, tuple):
        return (value,) + tuple(output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    return value


def _forward_capture(model, input_ids, *, push_layer=None, vector=None, registered_layer=None):
    layers = c.get_layers(model)
    if not 0 <= int(push_layer) < len(layers) - 1:
        raise ValueError("push layer must have a following block for the Gate 3 input check")
    base_outputs, post_expected, next_inputs = {}, {}, {}
    handles = []
    # Register these first so base_outputs sees the block output before a push hook edits it.
    for layer_no, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(
            lambda _module, _inputs, output, layer_no=layer_no:
                base_outputs.__setitem__(layer_no, _hidden(output).detach().float().clone())))
    if push_layer is not None and registered_layer is not None:
        actual_layer = int(registered_layer)
        if not 0 <= actual_layer < len(layers):
            raise ValueError("registered hook layer is outside the model")
        push = torch.as_tensor(vector, dtype=torch.float32, device=input_ids.device).reshape(1, -1)
        vectors = push.expand(input_ids.shape[0], -1).contiguous()
        handles.append(score180.add_sparse_row_push(layers[actual_layer], vectors))
    # This hook runs after the possible push hook on the declared block.
    handles.append(layers[int(push_layer)].register_forward_hook(
        lambda _module, _inputs, output: post_expected.__setitem__(
            "value", _hidden(output).detach().float().clone())))
    handles.append(layers[int(push_layer) + 1].register_forward_pre_hook(
        lambda _module, inputs: next_inputs.__setitem__("value", inputs[0].detach().float().clone())))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return base_outputs, post_expected.get("value"), next_inputs.get("value")


def check_hook_placement(model, input_ids, push_layer: int, vector, *, registered_layer=None,
                          relative_tolerance: float = 1e-5):
    """Return independent JSON verdicts for upstream, delta, and next-input checks."""
    layers = c.get_layers(model)
    expected_layer = int(push_layer)
    vector = torch.as_tensor(vector, dtype=torch.float32, device=input_ids.device)
    if vector.ndim != 1 or vector.numel() != int(model.config.hidden_size):
        raise ValueError("push vector must be a flat hidden-size vector")
    baseline, _, _ = _forward_capture(model, input_ids, push_layer=expected_layer)
    pushed, post_expected, next_input = _forward_capture(
        model, input_ids, push_layer=expected_layer, vector=vector,
        registered_layer=registered_layer)

    upstream_layers = list(range(expected_layer + 1))
    upstream_exact = all(torch.equal(baseline[layer], pushed[layer]) for layer in upstream_layers)
    baseline_hook_input = baseline[expected_layer]
    expected_changed = baseline_hook_input + vector.reshape(1, 1, -1)
    if post_expected is None:
        post_expected_ok, max_relative_error = False, None
    else:
        delta_error = post_expected - expected_changed
        denominator = torch.linalg.vector_norm(vector).clamp_min(torch.finfo(torch.float32).tiny)
        relative = torch.linalg.vector_norm(delta_error, dim=-1) / denominator
        max_relative_error = float(relative.max().item()) if relative.numel() else 0.0
        post_expected_ok = bool(torch.isfinite(relative).all() and (relative <= relative_tolerance).all())
    next_ok = next_input is not None and post_expected is not None and torch.equal(next_input, expected_changed)
    checks = {
        "upstream_residuals_unchanged_exact_float32": {
            "verdict": "GREEN" if upstream_exact else "RED", "passed": upstream_exact,
            "layers_checked": upstream_layers, "comparison": "torch.equal on block output before the push hook"},
        "declared_vector_after_hook": {
            "verdict": "GREEN" if post_expected_ok else "RED", "passed": post_expected_ok,
            "push_layer": expected_layer, "relative_tolerance": relative_tolerance,
            "max_relative_error": max_relative_error},
        "next_block_input_is_changed_residual": {
            "verdict": "GREEN" if next_ok else "RED", "passed": bool(next_ok),
            "next_layer": expected_layer + 1, "comparison": "torch.equal to baseline residual plus declared vector"},
    }
    return {"push_layer": expected_layer, "registered_layer": registered_layer,
            "checks": checks, "verdict": "GREEN" if all(check["passed"] for check in checks.values()) else "RED"}


def run_three_neurons(text_rows, neurons, pushes, tokenizer, model):
    by_neuron = {str(row["id"]): row for row in neurons}
    selected = []
    for row in sorted(text_rows, key=lambda item: (str(item["neuron"]), str(item["key"]))):
        if row.get("key") == "S0" and str(row["neuron"]) in by_neuron:
            selected.append(row)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError(f"Gate 3 requires three S0 texts; found {len(selected)}")
    device = next(model.parameters()).device
    # The layer the scorer will hook comes from the scorer's own task list (score180._task_list feeds
    # _score_chunk), and the declared layer from the neuron file, so a wrong push_layer in texts.jsonl or a
    # task-list bug turns this red. Until 2026-11-03 no registered layer was passed here, so the push was never
    # attached and the real-model check was red by construction (pilot 2); the toy tests passed it explicitly.
    scorer_tasks = {(str(task["neuron"]), task["case"]): task for task in score180._task_list(text_rows)}
    verdicts = []
    for row in selected:
        neuron_id = str(row["neuron"])
        task = scorer_tasks[(neuron_id, "self0_S0")]
        registered = int(task["push_layer"]) if task["pushed"] and task["push_layer"] is not None else None
        declared = int(by_neuron[neuron_id]["layer"])
        vector = pushes[f"{neuron_id}|neuron"].detach().to(device=device, dtype=torch.float32)
        ids = list(map(int, task["prompt_ids"])) + list(map(int, task["token_ids"]))
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        verdicts.append({"neuron": neuron_id, "declared_layer": declared,
                         "texts_push_layer": row.get("push_layer"), "checks": check_hook_placement(
                             model, input_ids, declared, vector, registered_layer=registered)})
    return {"part": 180, "gate": "3: hook placement", "neurons": verdicts,
            "passed": all(row["checks"]["verdict"] == "GREEN" for row in verdicts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", default=str(ROOT / "results" / "part180" / "texts.jsonl"))
    parser.add_argument("--neurons", default=str(ROOT / "results" / "part92" / "step2" / "neurons.json"))
    parser.add_argument("--pushes", default="results/part99-pushes.pt")
    parser.add_argument("--out", default=str(ROOT / "results" / "part180" / "hookcheck.json"))
    parser.add_argument("--model", default=score180.MODEL_ID)
    parser.add_argument("--revision", default=score180.MODEL_REVISION)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    args = parser.parse_args(argv)

    tokenizer, model = score180.load_direct(args.model, args.revision, args.device)
    result = run_three_neurons(score180._load_jsonl(Path(args.texts)),
                               score180._load_json(Path(args.neurons)),
                               torch.load(args.pushes, map_location="cpu", weights_only=True),
                               tokenizer, model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "out": str(out)}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
