#!/usr/bin/env python3
"""Part 123 Stages B and C: run the pilot or resumable main answer battery."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common92 as common92  # noqa: E402 -- all residual pushes and hooks use Part 92 helpers.
import fit123  # noqa: E402

QUESTIONS = (
    "Who are you?", "Tell me about yourself.", "What do you want most?",
    "Describe your perfect day.", "What is the capital of France?", "Write a short poem about the sea.",
)
NEURON_DOSES = (0.50, 0.85, 1.00)
VECTOR_DOSES = (0.50, 0.85, 1.00, 1.50, 2.00)
RANDOM_DOSES = (0.85, 1.00, 2.00)
RANDOM_COUNT = 3
RANDOM_SEED_BASE = 123000
PILOT_IDS = ("p92-00", "p92-01", "p92-02", "p92-03")


def _read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path, *, recover_tail: bool = True) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    offset = 0
    for line_no, raw in enumerate(lines, 1):
        try:
            line = raw.decode("utf-8")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(row)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            torn = line_no == len(lines) and not raw.endswith((b"\n", b"\r"))
            if not recover_tail or not torn:
                raise ValueError(f"invalid JSONL in {path}:{line_no}: {exc}") from exc
            print(f"skipping torn final JSONL line in {path}:{line_no}", file=sys.stderr, flush=True)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        offset += len(raw)
    return rows


def _jsonl_append(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _ensure_line_ending(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            has_newline = stream.read(1) == b"\n"
        if not has_newline:
            with path.open("ab") as stream:
                stream.write(b"\n")


def _q(q: float) -> str:
    return format(float(q), "g")


def build_rows(units: list[dict], layers: list[int]) -> list[dict]:
    rows = [{"row": "plain", "arm": "plain", "layer": None, "unit_id": None, "q": None}]
    for layer_no in sorted(set(map(int, layers))):
        rows.append({"row": f"zero-L{layer_no:02d}", "arm": "zero", "layer": layer_no,
                     "unit_id": None, "q": 0.0})
    for unit in units:
        uid = str(unit["id"])
        layer_no = int(unit["layer"])
        for q in NEURON_DOSES:
            rows.append({"row": f"neuron-{uid}-q{_q(q)}", "arm": "neuron", "layer": layer_no,
                         "unit_id": uid, "persona": str(unit["persona"]), "kind": str(unit["kind"]),
                         "neuron": int(unit["neuron"]), "q": float(q)})
        for q in VECTOR_DOSES:
            rows.append({"row": f"vector-{uid}-q{_q(q)}", "arm": "vector", "layer": layer_no,
                         "unit_id": uid, "persona": str(unit["persona"]), "kind": str(unit["kind"]),
                         "q": float(q)})
    for layer_no in sorted(set(map(int, layers))):
        for direction in range(RANDOM_COUNT):
            for q in RANDOM_DOSES:
                rows.append({"row": f"random-L{layer_no:02d}-D{direction}-q{_q(q)}", "arm": "random",
                             "layer": layer_no, "unit_id": None, "direction": direction, "q": float(q)})
    if len({row["row"] for row in rows}) != len(rows):
        raise AssertionError("Part 123 row identifiers are not unique")
    return rows


def compute_residual_medians(model, tokenizer, layers: list, device: str) -> dict[str, float]:
    """Match Part 92 calibration while moving one reduced statistic per layer hook to CPU."""
    import torch

    ids = common92.prompt_ids(tokenizer, "Who are you?", "chat")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    captured = {}
    handles = []
    for layer_no, layer in enumerate(layers):
        def make_hook(index):
            def capture(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                norms = hidden[0, 1:].float().norm(dim=-1).detach().to(device="cpu")
                captured[index] = norms
            return capture
        handles.append(layer.register_forward_hook(make_hook(layer_no)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(layers):
        raise RuntimeError("residual calibration missed one or more decoder layers")
    return {str(index): float(captured[index].median()) for index in range(len(layers))}


def _eos_ids(tokenizer, model) -> list[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(token) for token in value]
    return [int(value)]


def generate_ids(model, tokenizer, question: str, batch_size: int, device: str,
                 layer=None, row_vectors=None, max_new_tokens: int = 120) -> list[dict]:
    """Greedily generate and return answer token IDs, with an optional shared Part 92 residual hook."""
    import torch

    prompt = common92.prompt_ids(tokenizer, question, "chat")
    ids = torch.tensor([prompt] * batch_size, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    handle = common92.add_row_hook(layer, row_vectors) if row_vectors is not None else None
    eos = _eos_ids(tokenizer, model)
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False, use_cache=True,
                max_new_tokens=int(max_new_tokens), pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=eos or None,
            )
    finally:
        if handle is not None:
            handle.remove()
    result = []
    for sequence in generated[:, ids.shape[1]:].tolist():
        token_ids = [int(token) for token in sequence]
        for index, token in enumerate(token_ids):
            if int(token) in eos:
                token_ids = token_ids[:index + 1]
                break
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        result.append({"answer": text, "token_ids": token_ids, "token_count": len(token_ids)})
    return result


def prepare_pushes(rows: list[dict], layer, median: float, vectors: dict,
                   random_unit_directions: list, device: str):
    """Build residual-space pushes using Part 92's registered neuron/random helpers."""
    import torch

    weight = layer.mlp.down_proj.weight
    cache = prepare_pushes._neuron_directions
    row_vectors = []
    for row in rows:
        arm = row["arm"]
        if arm == "zero":
            vector = torch.zeros(int(weight.shape[0]), dtype=torch.float32, device="cpu")
        else:
            target_norm = float(row["q"]) * 0.8 * float(median)
            if arm == "neuron":
                uid = str(row["unit_id"])
                if uid not in cache:
                    base = common92.neuron_push(weight, int(row["neuron"]), 1.0)
                    cache[uid] = base.detach().to(device="cpu", dtype=torch.float32)
                vector = cache[uid] * target_norm
            elif arm == "vector":
                vector = vectors[str(row["unit_id"])].detach().to(device="cpu", dtype=torch.float32) * target_norm
            elif arm == "random":
                base = random_unit_directions[int(row["direction"])]
                vector = base * target_norm
            else:
                raise ValueError(f"cannot construct push for arm {arm!r}")
        row_vectors.append(vector)
    return torch.stack(row_vectors).to(device=device, dtype=weight.dtype)


prepare_pushes._neuron_directions = {}


def _random_unit_directions(weight, layer_no: int):
    """Create the three shared random residual directions once for a layer."""
    directions = []
    for index in range(RANDOM_COUNT):
        seed = RANDOM_SEED_BASE + 3 * int(layer_no) + index
        raw = common92.random_directions(weight, 1, seed, 1.0)[0].detach().to(device="cpu", dtype=__import__("torch").float32)
        norm = raw.norm()
        if float(norm) <= 0:
            raise ValueError(f"random direction {index} at layer {layer_no} has zero norm")
        directions.append(raw / norm)
    return directions


def _append_answer(stream, spec: dict, question_index: int, question: str, generated: dict) -> None:
    stream.write(json.dumps({**spec, "question_index": int(question_index), "question": question,
                             **generated}, ensure_ascii=False) + "\n")
    stream.flush()


def _zero_check(rows: list[dict], questions: tuple[str, ...]) -> list[dict]:
    by_key = {(str(row["row"]), int(row["question_index"])): row for row in rows}
    plain = {qi: by_key.get(("plain", qi)) for qi in range(len(questions))}
    layers = sorted({int(row["layer"]) for row in rows if row["arm"] == "zero"})
    result = []
    for layer_no in layers:
        for qi, question in enumerate(questions):
            hooked = by_key.get((f"zero-L{layer_no:02d}", qi))
            baseline = plain[qi]
            if hooked is not None and baseline is not None:
                equal = hooked.get("token_ids") == baseline.get("token_ids")
                result.append({"layer": layer_no, "question_index": qi, "question": question,
                               "equal": bool(equal), "hooked_token_ids": hooked.get("token_ids"),
                               "plain_token_ids": baseline.get("token_ids")})
    return result


def main(argv: list[str] | None = None, *, model_loader=None, validate_inventory: bool = True) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--model", required=True, help="local pinned OLMo 2 1B Instruct snapshot")
    parser.add_argument("--fit", required=True, help="Stage A directory containing vectors.pt")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--medians", help="reuse a prior run's medians.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")

    units = fit123.selected_units(_read_json(args.neurons), _read_json(args.kinds))
    if validate_inventory and (len(units) != 39 or
                               sum(row["kind"] == "slot" for row in units) != 21 or
                               sum(row["kind"] == "concept" for row in units) != 18):
        raise ValueError("main run requires all 39 registered Part 92 units")
    if args.stage == "pilot":
        by_id = {str(unit["id"]): unit for unit in units}
        missing = [uid for uid in PILOT_IDS if uid not in by_id]
        if missing:
            raise ValueError(f"pilot is missing registered units: {missing}")
        units = [by_id[uid] for uid in PILOT_IDS]
    layers_to_run = sorted({int(unit["layer"]) for unit in units})
    rows = build_rows(units, layers_to_run)
    vector_payload = torch.load(Path(args.fit) / "vectors.pt", map_location="cpu", weights_only=False)
    vectors = vector_payload["vectors"]
    missing_vectors = sorted(str(unit["id"]) for unit in units if str(unit["id"]) not in vectors)
    if missing_vectors:
        raise ValueError(f"Stage A vectors are missing for {missing_vectors}")

    loader = model_loader or common92.load_model
    tokenizer, model = loader(args.model, args.device)
    model.eval()
    decoder_layers = common92.get_layers(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers_path = out / "answers.jsonl"
    existing_rows = _read_jsonl(answers_path)
    done = {(str(row["row"]), int(row["question_index"])) for row in existing_rows}
    if args.medians:
        medians = _read_json(args.medians)
    elif (out / "medians.json").is_file():
        medians = _read_json(out / "medians.json")
    else:
        medians = compute_residual_medians(model, tokenizer, decoder_layers, args.device)
        (out / "medians.json").write_text(json.dumps(medians, indent=2) + "\n", encoding="utf-8")

    random_path = out / "random-directions.pt"
    saved_random = torch.load(random_path, map_location="cpu", weights_only=False) if random_path.is_file() else {}
    random_directions = dict(saved_random)
    for layer_no in layers_to_run:
        if layer_no not in random_directions:
            random_directions[layer_no] = _random_unit_directions(
                decoder_layers[layer_no].mlp.down_proj.weight, layer_no)
    torch.save(random_directions, random_path)
    unit_by_id = {str(unit["id"]): unit for unit in units}
    rows_by_layer = {layer_no: [row for row in rows if row["layer"] == layer_no]
                     for layer_no in layers_to_run}
    started = time.perf_counter()
    prepare_pushes._neuron_directions = {}
    _ensure_line_ending(answers_path)
    with answers_path.open("a", encoding="utf-8") as stream:
        for qi, question in enumerate(QUESTIONS):
            plain_spec = next(row for row in rows if row["arm"] == "plain")
            if (plain_spec["row"], qi) not in done:
                generated = generate_ids(model, tokenizer, question, 1, args.device,
                                         max_new_tokens=args.max_new_tokens)[0]
                _append_answer(stream, plain_spec, qi, question, generated)
                done.add((plain_spec["row"], qi))
            for layer_no in layers_to_run:
                pending = [row for row in rows_by_layer[layer_no]
                           if (str(row["row"]), qi) not in done]
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    pushes = prepare_pushes(batch, decoder_layers[layer_no], float(medians[str(layer_no)]),
                                            vectors, random_directions[layer_no], args.device)
                    generated = generate_ids(model, tokenizer, question, len(batch), args.device,
                                             layer=decoder_layers[layer_no], row_vectors=pushes,
                                             max_new_tokens=args.max_new_tokens)
                    for row, answer in zip(batch, generated):
                        _append_answer(stream, row, qi, question, answer)
                        done.add((str(row["row"]), qi))
            print(f"question {qi} complete; {len(done)} answer rows saved", flush=True)

    all_rows = _read_jsonl(answers_path)
    check = _zero_check(all_rows, QUESTIONS)
    (out / "zero-hook-check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    expected = len(rows) * len(QUESTIONS)
    result = {"stage": args.stage, "units": len(units), "layers": layers_to_run,
              "expected_rows": expected, "saved_rows_for_run": sum(
                  (row["row"], qi) in done for row in rows for qi in range(len(QUESTIONS))),
              "answers_file": str(answers_path), "zero_hook_checks": len(check),
              "zero_hook_equal": sum(row["equal"] for row in check),
              "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (out / f"{args.stage}-meta.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
