#!/usr/bin/env python3
"""Run Part 32's multilingual maker-circuit intervention experiment.

The experiment uses the four Part 25 removal neurons. ``zeroed`` sets their
MLP down-projection inputs to zero at every prefill and generated-token
position. ``clamped`` sets them to their Part 25 Google-system stem means.
Results are appended as one JSON object per answer; complete generation
batches are checkpointed atomically so an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QUESTION_BY_LANGUAGE = {
    "english": "Who created you?",
    "chinese": "谁创造了你？",
    "spanish": "¿Quién te creó?",
}
CONDITIONS = ("unchanged", "zeroed", "clamped")
NEURONS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
TEMPERATURE = 0.7
TOP_P = 0.95
DEFAULT_OUT = Path(__file__).resolve().parent / "results/part32/answers.jsonl"
DEFAULT_STEERING_RESULTS = Path(__file__).resolve().parent / "results/part25/search-google-removal.json"


def sample_batch_plan(samples: int, gen_batch: int, seed: int) -> list[dict[str, int]]:
    """Return ``{seed,start,count}`` batches; each seed is shared across cells."""
    if samples < 1 or gen_batch < 1:
        raise ValueError("samples and gen_batch must be positive")
    return [
        {"seed": int(seed + batch_index), "start": int(start),
         "count": int(min(gen_batch, samples - start))}
        for batch_index, start in enumerate(range(0, samples, gen_batch))
    ]


def expected_cell_counts(samples: int, gen_batch: int, seed: int) -> dict[tuple[str, str], Counter[int]]:
    plan = sample_batch_plan(samples, gen_batch, seed)
    return {
        (language, condition): Counter({batch["seed"]: batch["count"] for batch in plan})
        for language in QUESTION_BY_LANGUAGE for condition in CONDITIONS
    }


def validate_existing_rows(rows: list[dict[str, Any]], samples: int, gen_batch: int,
                           seed: int) -> dict[tuple[str, str, int], int]:
    """Validate JSONL rows against the nine-cell, seeded batch plan."""
    expected = expected_cell_counts(samples, gen_batch, seed)
    counts: Counter[tuple[str, str, int]] = Counter()
    for index, row in enumerate(rows, start=1):
        if set(row) != {"language", "condition", "seed", "answer"}:
            raise ValueError(f"row {index} must contain exactly language, condition, seed, answer")
        language, condition = row["language"], row["condition"]
        if language not in QUESTION_BY_LANGUAGE or condition not in CONDITIONS:
            raise ValueError(f"row {index} has an unknown language or condition")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            raise ValueError(f"row {index} seed must be an integer")
        if not isinstance(row["answer"], str):
            raise ValueError(f"row {index} answer must be a string")
        key = (language, condition, row["seed"])
        counts[key] += 1
        allowed = expected[(language, condition)].get(row["seed"], 0)
        if allowed == 0 or counts[key] > allowed:
            raise ValueError(f"row {index} is outside the requested sample batches")

    for (language, condition), batches in expected.items():
        for batch_seed, expected_count in batches.items():
            found = counts[(language, condition, batch_seed)]
            if found not in (0, expected_count):
                raise ValueError(
                    f"partial batch for {language}/{condition}, seed {batch_seed}: "
                    f"found {found}, expected {expected_count}; remove that incomplete batch before resuming"
                )
    return dict(counts)


def build_operations(condition: str, neurons: Iterable[tuple[int, int]],
                     clamp_values: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    if condition == "unchanged":
        return {}
    if condition == "zeroed":
        return {(int(layer), int(unit)): 0.0 for layer, unit in neurons}
    if condition == "clamped":
        result = {(int(layer), int(unit)): float(clamp_values[(int(layer), int(unit))])
                  for layer, unit in neurons}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("clamp values must be finite")
        return result
    raise ValueError(f"unknown condition {condition!r}")


def make_intervention_hook(layer_index: int, operations: dict[tuple[int, int], float]):
    """Build a Part 25 style down_proj pre-hook for one layer."""
    units = [(unit, value) for (layer, unit), value in operations.items()
             if int(layer) == int(layer_index)]
    if not units:
        return None

    def hook(_module, args):
        if not args:
            raise ValueError("down_proj pre-hook received no activation tensor")
        activation = args[0]
        changed = activation.clone()
        for unit, value in units:
            changed[..., int(unit)] = float(value)
        return (changed,) + tuple(args[1:])

    return hook


def install_intervention_hooks(layers, operations: dict[tuple[int, int], float],
                               part25_module=None, row_count: int = 1):
    """Install on Part 25's hook path, with a small duck-typed test fallback."""
    if not operations:
        return []
    if part25_module is not None:
        return part25_module.install_intervention_hooks(
            layers, [operations for _ in range(row_count)])
    handles = []
    try:
        for layer_index in sorted({int(layer) for layer, _ in operations}):
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(f"neuron layer {layer_index} is outside the model")
            hook = make_intervention_hook(layer_index, operations)
            if hook is not None:
                handles.append(layers[layer_index].mlp.down_proj.register_forward_pre_hook(hook))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def load_clamp_values(path: str | Path = DEFAULT_STEERING_RESULTS) -> dict[tuple[int, int], float]:
    """Read the four Google-system stem means recorded by Part 25.

    The compressed full-layer ``means.npz`` is git-ignored and is absent from
    some checkouts. Part 25's removal result also stores the exact means for
    each selected unit under ``natural_activation_ranges.target_system``.
    """
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("model") != MODEL or obj.get("revision") != REVISION:
        raise ValueError(f"{path} is not a result for the pinned Part 25 model revision")
    recorded = {(int(row["layer"]), int(row["index"]))
                for row in obj.get("pruned", {}).get("neurons", [])}
    if recorded != set(NEURONS):
        raise ValueError(f"{path} does not contain the Part 25 four-neuron removal set")
    ranges = obj.get("pruned", {}).get("natural_activation_ranges", {}).get("target_system", {})
    values = {}
    for layer, unit in NEURONS:
        record = ranges.get(f"{layer}:{unit}")
        if not isinstance(record, dict) or "mean" not in record:
            raise ValueError(f"missing Part 25 Google-system mean for L{layer} N{unit} in {path}")
        value = float(record["mean"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite Part 25 Google-system mean for L{layer} N{unit}")
        values[(layer, unit)] = value
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_append_rows(path: Path, new_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            if path.exists():
                with path.open(encoding="utf-8") as existing:
                    for line in existing:
                        stream.write(line)
            for row in new_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _run_metadata(args, steering_path: Path, clamp_values: dict[tuple[int, int], float]) -> dict[str, Any]:
    source_sha = _sha256(steering_path)
    return {
        "model": MODEL,
        "revision": REVISION,
        "question_by_language": QUESTION_BY_LANGUAGE,
        "conditions": list(CONDITIONS),
        "neurons": [{"layer": layer, "index": unit} for layer, unit in NEURONS],
        "clamp_values": {f"{layer}:{unit}": value for (layer, unit), value in clamp_values.items()},
        "steering_source": str(steering_path.resolve()),
        "steering_source_sha256": source_sha,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": args.max_new,
        "seed_base": args.seed,
        "seed_rule": "seed_base + batch_index; shared across the nine cells for paired batch sampling",
        "gen_batch": args.gen_batch,
        "samples_requested": args.samples,
        "prompt_builder": "neuron_report.build_prompt(tok, question, False), Part 25 chat-template path",
    }


def _prepare_metadata(metadata_path: Path, current: dict[str, Any]) -> None:
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in current.items():
            if old.get(key) != value:
                raise ValueError(f"resume metadata mismatch for {key}; use a new --out path")
    elif Path(str(current.get("out_path", ""))).exists():
        raise ValueError("answer JSONL exists without resume metadata; use a new --out path")
    _atomic_json(metadata_path, current)


def _run(args) -> int:
    # Keep model dependencies out of import time so scoring and the fake-hook
    # tests run without torch/Transformers or a model download.
    try:
        import torch
        import part25
        import neuron_report as nr
    except ImportError as exc:
        raise SystemExit(f"model run needs the repository's torch/Transformers environment: {exc}") from exc
    if part25.MODEL != MODEL or part25.REVISION != REVISION:
        raise SystemExit("Part 25 model constants differ from the Part 32 pinned model/revision")

    steering_path = Path(args.steering_results).resolve()
    if not steering_path.is_file():
        raise SystemExit(f"Part 25 steering result not found: {steering_path}")
    clamp_values = load_clamp_values(steering_path)
    out_path = Path(args.out).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else out_path.with_suffix(out_path.suffix + ".meta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _run_metadata(args, steering_path, clamp_values)
    metadata["out_path"] = str(out_path)
    _prepare_metadata(metadata_path, metadata)
    rows = _read_rows(out_path)
    existing = validate_existing_rows(rows, args.samples, args.gen_batch, args.seed)
    expected_counts = expected_cell_counts(args.samples, args.gen_batch, args.seed)
    pending = [(language, condition, batch)
               for language in QUESTION_BY_LANGUAGE
               for condition in CONDITIONS
               for batch in sample_batch_plan(args.samples, args.gen_batch, args.seed)
               if existing.get((language, condition, batch["seed"]), 0) == 0]
    if not pending:
        print(f"complete: {sum(expected_counts[(lang, cond)].total() for lang in QUESTION_BY_LANGUAGE for cond in CONDITIONS)} answers in {out_path}")
        return 0

    # Match Part 25's prompt builder, pinned model loader, and generation settings.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    # Delegate activation operation construction and hook installation to
    # Part 25 itself, so the run uses its exact down_proj intervention path.
    import numpy as np
    steering_matrix = np.zeros((max(layer for layer, _ in NEURONS) + 1,
                                max(unit for _, unit in NEURONS) + 1), dtype=np.float32)
    for (layer, unit), value in clamp_values.items():
        steering_matrix[layer, unit] = value
    clamp_ops = part25._operations(NEURONS, "steer", steering_matrix)
    zero_ops = part25._operations(NEURONS, "zero")
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    for language, condition, batch in pending:
        question = QUESTION_BY_LANGUAGE[language]
        prompt = nr.build_prompt(tok, question, False)
        encoded = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        operations = {"unchanged": {}, "zeroed": zero_ops, "clamped": clamp_ops}[condition]
        handles = install_intervention_hooks(
            layers, operations, part25_module=part25, row_count=batch["count"])
        try:
            torch.manual_seed(batch["seed"])
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new,
                    pad_token_id=pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    num_return_sequences=batch["count"],
                )
            prompt_len = int(encoded["input_ids"].shape[1])
            answers = [tok.decode(sequence[prompt_len:], skip_special_tokens=True).strip()
                       for sequence in generated]
        finally:
            for handle in handles:
                handle.remove()
        if len(answers) != batch["count"]:
            raise RuntimeError(f"generation returned {len(answers)} answers; expected {batch['count']}")
        batch_rows = [{"language": language, "condition": condition,
                       "seed": batch["seed"], "answer": answer}
                      for answer in answers]
        _atomic_append_rows(out_path, batch_rows)
        print(f"saved {language}/{condition} seed={batch['seed']} ({len(answers)} answers)", flush=True)
        nr.spill_guard(args.device, args.max_gpu_gb)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    p.add_argument("--samples", type=int, default=24, help="samples per language-condition cell")
    p.add_argument("--gen-batch", type=int, default=1, help="num_return_sequences per saved generation batch; 1 records one seed per answer")
    p.add_argument("--seed", type=int, default=0, help="base seed; each batch uses seed + batch index")
    p.add_argument("--max-new", type=int, default=48, help="max generated tokens; Part 25 default")
    p.add_argument("--max-gpu-gb", type=float, default=14.5)
    p.add_argument("--out", default=str(DEFAULT_OUT), help="resumable answer JSONL path")
    p.add_argument("--metadata", default=None, help="run metadata path; defaults beside --out")
    p.add_argument("--steering-results", default=str(DEFAULT_STEERING_RESULTS),
                   help="Part 25 search-google-removal.json with stored Google-system means")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for name in ("samples", "gen_batch", "max_new"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
