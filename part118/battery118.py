#!/usr/bin/env python3
"""Run Part 118's six-question neuron and matched-random battery."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE.parent / "part92") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "part92"))

import common92 as c  # noqa: E402

QUESTIONS = (
    "Who are you?",
    "Tell me about yourself.",
    "What do you want most?",
    "Describe your perfect day.",
    "What is the capital of France?",
    "Write a short poem about the sea.",
)
DEFAULT_ALPHAS = (0.4, 0.8, 1.6, 3.2)
KINDS = {"concept", "slot"}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_alphas(value: str) -> list[float]:
    try:
        alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc
    if not alphas or any(not math.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must be finite positive numbers")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("alphas must be unique")
    return alphas


def load_units(neurons_path: str | Path, kinds_path: str | Path,
               limit: int | None = None) -> list[dict]:
    with repo_path(neurons_path).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    with repo_path(kinds_path).open(encoding="utf-8") as stream:
        kind_data = json.load(stream)
    kinds = kind_data.get("kinds", kind_data)
    if not isinstance(neurons, list) or not isinstance(kinds, dict):
        raise ValueError("Part 92 neurons must be a JSON list and kinds must be a mapping")
    units = []
    seen = set()
    for row in neurons:
        unit_id = str(row["id"])
        kind = kinds.get(unit_id)
        if kind not in KINDS:
            continue
        if unit_id in seen:
            raise ValueError(f"duplicate Part 92 unit id {unit_id!r}")
        seen.add(unit_id)
        for field in ("layer", "neuron", "persona", "desc", "control_seed"):
            if field not in row:
                raise ValueError(f"Part 92 unit {unit_id!r} is missing {field!r}")
        units.append({**row, "id": unit_id, "kind": kind,
                      "layer": int(row["layer"]), "neuron": int(row["neuron"]),
                      "control_seed": int(row["control_seed"])})
    if len(units) != 39 or sum(row["kind"] == "concept" for row in units) != 18 \
            or sum(row["kind"] == "slot" for row in units) != 21:
        raise ValueError("Part 92 classified input must contain 18 concepts and 21 slots")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit-units must be positive")
        units = units[:limit]
    return units


def read_existing(path: Path) -> set[tuple[str, str, float, int]]:
    if not path.exists():
        return set()
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                # A killed process may leave only its final JSONL write incomplete.
                path.write_text("".join(lines[:index]), encoding="utf-8")
                break
            raise ValueError(f"invalid JSONL in {path} at line {index + 1}")
        if row.get("arm") == "baseline":
            key = ("baseline", "baseline", 0.0, int(row["question_index"]))
        else:
            key = (str(row["id"]), str(row["arm"]), round(float(row["alpha"]), 8),
                   int(row["question_index"]))
        if key in rows:
            raise ValueError(f"duplicate answer cell in {path}: {key}")
        rows.append(key)
    return set(rows)


def run(args: argparse.Namespace) -> dict:
    import torch

    units = load_units(args.neurons, args.kinds, args.limit_units)
    alphas = args.alphas
    tokenizer, model = c.load_model(repo_path(args.model), args.device)
    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, "chat", args.device)

    output_dir = repo_path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.tag}.jsonl"
    existing = read_existing(output)
    work: dict[tuple[int, float], list[tuple[dict, str, int]]] = {}
    pushes: dict[tuple[str, str, float], object] = {}
    for unit in units:
        layer_no = unit["layer"]
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"layer {layer_no} is outside the loaded model")
        layer = layers[layer_no]
        weight = layer.mlp.down_proj.weight
        if not 0 <= unit["neuron"] < weight.shape[1]:
            raise ValueError(f"neuron {unit['neuron']} is outside layer {layer_no} W_down")
        for alpha in alphas:
            target_norm = alpha * medians[layer_no]
            if "neuron" in args.arms:
                pushes[(unit["id"], "neuron", alpha)] = c.neuron_push(
                    weight, unit["neuron"], target_norm
                )
            if "random" in args.arms:
                pushes[(unit["id"], "random", alpha)] = c.random_direction(
                    weight, unit["control_seed"], target_norm
                )
            rows = work.setdefault((layer_no, alpha), [])
            for arm in args.arms:
                for question_index in range(len(QUESTIONS)):
                    key = (unit["id"], arm, round(alpha, 8), question_index)
                    if key not in existing:
                        rows.append((unit, arm, question_index))

    baseline_pending = []
    if args.include_baseline:
        baseline_pending = [qi for qi in range(len(QUESTIONS))
                            if ("baseline", "baseline", 0.0, qi) not in existing]
    total = sum(len(rows) for rows in work.values()) + len(baseline_pending)
    started = time.perf_counter()
    done = 0
    with output.open("a", encoding="utf-8") as stream:
        if baseline_pending:
            answers = c.generate_batch(model, tokenizer,
                                       [QUESTIONS[qi] for qi in baseline_pending],
                                       "chat", None, None, 120, args.device)
            for qi, answer in zip(baseline_pending, answers):
                row = {"id": "baseline", "layer": None, "neuron": None,
                       "kind": "baseline", "arm": "baseline", "alpha": 0.0,
                       "question_index": qi, "question": QUESTIONS[qi], "answer": answer}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            done += len(answers)
        for (layer_no, alpha), tasks in work.items():
            layer = layers[layer_no]
            for begin in range(0, len(tasks), args.batch):
                chunk = tasks[begin:begin + args.batch]
                vectors = torch.stack([
                    pushes[(unit["id"], arm, alpha)] for unit, arm, _qi in chunk
                ])
                answers = c.generate_batch(
                    model, tokenizer, [QUESTIONS[qi] for _unit, _arm, qi in chunk],
                    "chat", layer, vectors, 120, args.device,
                )
                for (unit, arm, qi), answer in zip(chunk, answers):
                    row = {"id": unit["id"], "layer": unit["layer"],
                           "neuron": unit["neuron"], "kind": unit["kind"],
                           "arm": arm, "alpha": alpha, "question_index": qi,
                           "question": QUESTIONS[qi], "answer": answer}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                done += len(chunk)
                print(f"{args.tag}: {done}/{total} rows in {time.perf_counter() - started:.1f}s",
                      flush=True)
    result = {"units": len(units), "alphas": alphas, "arms": args.arms,
              "expected_scored_answers":
              len(units) * len(alphas) * len(QUESTIONS) * len(args.arms),
              "baseline_answers": 6 if args.include_baseline else 0,
              "new_rows": total, "output": str(output)}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local pinned OLMo 2 1B Instruct snapshot")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--alphas", type=parse_alphas, default=list(DEFAULT_ALPHAS),
                        help="comma-separated positive push multiples")
    parser.add_argument("--arms", nargs="+", choices=("neuron", "random"),
                        default=("neuron", "random"),
                        help="arms to generate (default: neuron random)")
    parser.add_argument("--limit-units", type=int, help="run the first N classified Part 92 units")
    parser.add_argument("--include-baseline", action="store_true",
                        help="also write the six unpushed answers once")
    parser.add_argument("--tag", default="battery118")
    parser.add_argument("--out", default="results/part118")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch", type=int, default=48)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
