#!/usr/bin/env python3
"""Part 99 battery: Part 92's neuron pushes with optional component swaps."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PART92 = REPO / "part92"
if str(PART92) not in sys.path:
    sys.path.insert(0, str(PART92))

import common92 as c

QUESTIONS = (
    "Who are you?", "Tell me about yourself.", "What do you want most?",
    "Describe your perfect day.", "What is the capital of France?",
    "Write a short poem about the sea.",
)
ALPHA = 0.8
TOKENIZER_TEST_TEXT = "The quick brown fox jumps over the lazy dog; OLMo 2, 42, and café."


def _named_parameters(module):
    """Return a stable name-to-parameter map for a component."""
    return dict(module.named_parameters(recurse=True))


def copy_components(host_model, donor_model, swap: str) -> dict:
    """Copy all selected donor component parameters into the loaded host model.

    The complete transfer is shape/name checked before any host parameter is
    changed. ``mlp`` includes every parameter below each layer's MLP (including
    its gate, up, and down projections); ``attn`` includes every parameter
    below self_attn, including optional q/k norms.
    """
    import torch

    if swap not in {"none", "mlp", "attn"}:
        raise ValueError("swap must be one of: none, mlp, attn")
    if swap == "none":
        return {"tensors_copied": 0, "elements_copied": 0,
                "copied_parameters": [], "module": None}
    if donor_model is None:
        raise ValueError(f"a donor model is required for --swap {swap}")

    host_layers = c.get_layers(host_model)
    donor_layers = c.get_layers(donor_model)
    if len(host_layers) != len(donor_layers):
        raise ValueError(f"layer count mismatch: host has {len(host_layers)}, donor has {len(donor_layers)}")

    module_name = "mlp" if swap == "mlp" else "self_attn"
    transfers = []
    copied_names = []
    for layer_no, (host_layer, donor_layer) in enumerate(zip(host_layers, donor_layers)):
        host_module = getattr(host_layer, module_name)
        donor_module = getattr(donor_layer, module_name)
        host_params = _named_parameters(host_module)
        donor_params = _named_parameters(donor_module)
        if set(host_params) != set(donor_params):
            missing = sorted(set(host_params) - set(donor_params))
            extra = sorted(set(donor_params) - set(host_params))
            raise ValueError(
                f"layer {layer_no} {module_name} parameter names differ: "
                f"missing in donor={missing}, extra in donor={extra}"
            )
        if not host_params:
            raise ValueError(f"layer {layer_no} {module_name} has no parameters")
        if swap == "mlp":
            required = {f"{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")}
            if not required.issubset(host_params):
                raise ValueError(f"layer {layer_no} MLP lacks required projection weights: {sorted(required - set(host_params))}")
        for name in sorted(host_params):
            host_param, donor_param = host_params[name], donor_params[name]
            if tuple(host_param.shape) != tuple(donor_param.shape):
                raise ValueError(
                    f"shape mismatch at layer {layer_no} {module_name}.{name}: "
                    f"host {tuple(host_param.shape)} vs donor {tuple(donor_param.shape)}"
                )
            transfers.append((host_param, donor_param))
            copied_names.append(f"layers.{layer_no}.{module_name}.{name}")

    with torch.no_grad():
        for host_param, donor_param in transfers:
            host_param.copy_(donor_param.to(device=host_param.device, dtype=host_param.dtype))

    return {
        "tensors_copied": len(transfers),
        "elements_copied": sum(int(host_param.numel()) for host_param, _ in transfers),
        "copied_parameters": copied_names,
        "module": module_name,
    }


def assert_tokenizers_compatible(host_tokenizer, selected_tokenizer) -> None:
    """Check vocabulary size and one fixed probe string before using a donor tokenizer."""
    host_size = len(host_tokenizer)
    selected_size = len(selected_tokenizer)
    if host_size != selected_size:
        raise ValueError(f"tokenizer vocabulary size mismatch: host={host_size}, selected={selected_size}")
    host_ids = host_tokenizer(TOKENIZER_TEST_TEXT, add_special_tokens=False)["input_ids"]
    selected_ids = selected_tokenizer(TOKENIZER_TEST_TEXT, add_special_tokens=False)["input_ids"]
    if list(host_ids) != list(selected_ids):
        raise ValueError("host and selected tokenizers produce different ids for the compatibility test string")


def _load_donor(path: str):
    """Load the donor on CPU so the only additional model-sized allocation is host+donor."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(Path(path).expanduser()), torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    return model


def load_kind_map(path: str | Path) -> dict[str, str]:
    """Read Part 92's score JSON ``kinds`` mapping or a direct id-to-kind map."""
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("--kinds must contain a JSON object with a 'kinds' map")
    kinds = data.get("kinds", data)
    if not isinstance(kinds, dict) or any(not isinstance(value, str) for value in kinds.values()):
        raise ValueError("--kinds must contain a string-valued id-to-kind map")
    return {str(key): value for key, value in kinds.items()}


def parse_only(value: str | None) -> set[str] | None:
    if value is None:
        return None
    selected = {part.strip() for part in value.split(",") if part.strip()}
    allowed = {"concept", "slot"}
    if not selected or not selected.issubset(allowed):
        raise ValueError("--only accepts a comma-separated subset of: concept,slot")
    return selected


def write_meta(out_dir: Path, tag: str, *, host: str, donor: str | None, swap: str,
               tokenizer_source: str, swap_result: dict, frame: str, device: str,
               torch_version: str, transformers_version: str, seconds: float) -> dict:
    meta = {
        "host": host,
        "donor": donor,
        "swap": swap,
        "tokenizer_source": tokenizer_source,
        "tensors_copied": int(swap_result["tensors_copied"]),
        "elements_copied": int(swap_result["elements_copied"]),
        "copied_parameters": list(swap_result["copied_parameters"]),
        "frame": frame,
        "device": device,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "seconds": float(seconds),
    }
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _startup_copy_description(swap_result: dict) -> str:
    if not swap_result["copied_parameters"]:
        return "none"
    module = swap_result["module"]
    names_by_layer = {}
    for full_name in swap_result["copied_parameters"]:
        _, layer_no, component, parameter = full_name.split(".", 3)
        names_by_layer.setdefault(int(layer_no), []).append(parameter)
    name_sets = {tuple(names) for names in names_by_layer.values()}
    if len(name_sets) == 1:
        names = next(iter(name_sets))
        return f"layers[*].{module}.{{{','.join(names)}}}"
    return ",".join(swap_result["copied_parameters"])


def run(args):
    import torch
    import transformers
    from transformers import AutoTokenizer

    run_started = time.perf_counter()
    with Path(args.neurons).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    if not isinstance(neurons, list):
        raise ValueError("--neurons must contain a JSON list")
    only = parse_only(args.only)
    if only is not None:
        if not args.kinds:
            raise ValueError("--only requires --kinds")
        kinds = load_kind_map(args.kinds)
        neurons = [neuron for neuron in neurons if kinds.get(str(neuron.get("id"))) in only]
    if args.limit is not None:
        neurons = neurons[:args.limit]
    if not neurons:
        raise ValueError("no neurons remain after applying --only and --limit")

    host_tokenizer, model = c.load_model(args.host, args.device)
    tokenizer_source = str(Path(args.tokenizer or args.host).expanduser())
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
        assert_tokenizers_compatible(host_tokenizer, tokenizer)
    else:
        tokenizer = host_tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    donor_model = None
    if args.swap != "none":
        if not args.donor:
            raise ValueError(f"--donor is required when --swap is {args.swap}")
        donor_model = _load_donor(args.donor)
    swap_result = copy_components(model, donor_model, args.swap)
    if donor_model is not None:
        del donor_model
        gc.collect()

    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, args.frame, args.device)
    prepared = {}
    for neuron in neurons:
        layer_no = int(neuron["layer"])
        index = int(neuron["neuron"])
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"layer {layer_no} outside model range")
        weight = layers[layer_no].mlp.down_proj.weight
        if not 0 <= index < weight.shape[1]:
            raise ValueError(f"neuron {index} outside layer {layer_no} W_down input width")
        target_norm = ALPHA * medians[layer_no]
        prepared[(str(neuron["id"]), "neuron")] = c.neuron_push(weight, index, target_norm)
        prepared[(str(neuron["id"]), "random")] = c.random_direction(
            weight, int(neuron["control_seed"]), target_norm
        )
    if args.pushes:
        # Amendment 1 (Alden's review): one fixed injected vector per unit for every host, built once on the
        # unmodified Instruct model in the chat frame (make_pushes99.py); host recalibration is not used.
        fixed = torch.load(args.pushes, map_location="cpu")
        for key in list(prepared):
            vec = fixed[f"{key[0]}|{key[1]}"]
            ref = prepared[key]
            prepared[key] = vec.to(device=ref.device, dtype=ref.dtype)

    print(
        f"{args.tag}: host={args.host} donor={args.donor or '-'} swap={args.swap} "
        f"copied={_startup_copy_description(swap_result)} "
        f"tensors_copied={swap_result['tensors_copied']} "
        f"elements_copied={swap_result['elements_copied']}",
        flush=True,
    )

    work_by_layer = {}
    for neuron in neurons:
        layer_no = int(neuron["layer"])
        work_by_layer.setdefault(layer_no, []).extend(
            (neuron, unit, qi) for unit in ("neuron", "random") for qi in range(len(QUESTIONS))
        )
    # Unpushed answers to the six questions (a zero push), so each run's coherence without any push is on record
    # (PART99-DESIGN.md, prior-art section). Written with id "unpushed" and unit "none".
    first = neurons[0]
    unpushed = {"id": "unpushed", "layer": int(first["layer"])}
    prepared[("unpushed", "none")] = prepared[(str(first["id"]), "neuron")] * 0
    work_by_layer.setdefault(int(first["layer"]), []).extend(
        (unpushed, "none", qi) for qi in range(len(QUESTIONS))
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.tag}.jsonl"
    existing = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                existing.add((str(row["id"]), row["unit"], int(row["question_index"])))
    started = time.perf_counter()
    total = sum(
        sum((str(row[0]["id"]), row[1], row[2]) not in existing for row in rows)
        for rows in work_by_layer.values()
    )
    done = 0
    with output.open("a", encoding="utf-8") as stream:
        for layer_no, layer_work in work_by_layer.items():
            layer_work = [row for row in layer_work
                          if (str(row[0]["id"]), row[1], row[2]) not in existing]
            layer = layers[layer_no]
            for begin in range(0, len(layer_work), args.batch):
                chunk = layer_work[begin:begin + args.batch]
                pushes = torch.stack([
                    prepared[(str(neuron["id"]), unit)] for neuron, unit, _qi in chunk
                ])
                answers = c.generate_batch(
                    model, tokenizer, [QUESTIONS[qi] for _neuron, _unit, qi in chunk],
                    args.frame, layer, pushes, 120, args.device,
                )
                for (neuron, unit, qi), answer in zip(chunk, answers):
                    row = {"id": neuron["id"], "unit": unit, "question_index": qi,
                           "question": QUESTIONS[qi], "answer": answer}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                done += len(chunk)
                if done % max(args.batch * 5, 1) < len(chunk) or done == total:
                    elapsed = time.perf_counter() - started
                    print(f"{args.tag}: {done}/{total} rows in {elapsed:.1f}s", flush=True)

    seconds = time.perf_counter() - run_started
    write_meta(
        out_dir, args.tag, host=str(Path(args.host).expanduser()),
        donor=str(Path(args.donor).expanduser()) if args.donor else None,
        swap=args.swap, tokenizer_source=tokenizer_source, swap_result=swap_result,
        frame=args.frame, device=args.device, torch_version=torch.__version__,
        transformers_version=transformers.__version__, seconds=seconds,
    )
    print(f"{args.tag}: completed in {seconds:.1f}s; meta={out_dir / (args.tag + '.meta.json')}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="local host snapshot directory")
    parser.add_argument("--donor", help="optional local donor snapshot directory")
    parser.add_argument("--swap", choices=("none", "mlp", "attn"), default="none")
    parser.add_argument("--tokenizer", help="optional local tokenizer snapshot directory")
    parser.add_argument("--neurons", required=True, help="JSON list of id/layer/neuron/persona/desc/control_seed")
    parser.add_argument("--frame", required=True, choices=("chat", "plain"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--kinds", help="Part 92 score JSON with a kinds id-to-kind map")
    parser.add_argument("--only", help="comma-separated concept,slot filter; requires --kinds")
    parser.add_argument("--pushes", help="fixed push vectors from make_pushes99.py (Amendment 1)")
    parser.add_argument("--limit", type=int, help="run only the first N eligible neurons")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        parse_only(args.only)
    except ValueError as exc:
        parser.error(str(exc))
    if args.only and not args.kinds:
        parser.error("--only requires --kinds")
    if args.swap != "none" and not args.donor:
        parser.error(f"--donor is required when --swap is {args.swap}")
    run(args)


if __name__ == "__main__":
    main()
