#!/usr/bin/env python3
"""Run Part 119c's fixed Path 2 sample in bf16 and float32."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119b", ROOT / "part119", ROOT / "part92", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common92 as c  # noqa: E402
import rows119  # noqa: E402
import run119  # noqa: E402
import run119b  # noqa: E402 -- runner-only Path 2 implementation; no scorer/reader import
import rows119b  # noqa: E402 -- source Path 2 inventory
import rows119c  # noqa: E402

QUESTION = run119.QUESTION
FRAME = run119.FRAME
DOSES = rows119c.DOSES
ANSWER_PATH = "results/part119/answers.jsonl"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_model(snapshot_dir: str | Path, device: str, dtype):
    """Part 92's local-only loader with an explicit model dtype."""
    import torch  # noqa: F401 -- keeps loader dependencies local to execution
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = str(Path(snapshot_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=dtype, local_files_only=True
        ).to(device)
    else:
        # Straight to the device, as part115b/howmany115b.py: staging float32 weights in host memory
        # tripped the job's RAM guard (0.85 GB left) on the first launch.
        import transformers.modeling_utils as modeling_utils
        modeling_utils.caching_allocator_warmup = lambda *a, **k: None
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=dtype, local_files_only=True, device_map={"": device})
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def build_dtype_push_caches(units: list[dict], layers, medians: dict[int, float],
                            doses: tuple[float, ...] = DOSES) -> tuple[dict, dict[float, dict]]:
    """Build every push directly from this dtype's W_down weights."""
    base = run119b.build_base_push_cache(units, layers, medians)
    by_dose = {float(dose): run119b.build_dose_push_cache(units, layers, medians, dose)
               for dose in doses}
    for dose, cache in by_dose.items():
        run119b.validate_dose_scaling(units, base, cache, dose)
    return base, by_dose


def _answers_for_sample(part119_answers: dict[str, dict], rows: list[dict]) -> dict[str, dict]:
    if "plain" not in part119_answers:
        raise ValueError(f"Part 119 plain answer is missing from {ANSWER_PATH}")
    answers = {"plain": part119_answers["plain"]}
    for first in sorted({str(row["first"]) for row in rows}):
        source = part119_answers.get(f"single:{first}")
        if source is None:
            raise ValueError(f"Part 119 0.8 single answer is missing for {first}")
        answers[f"single@0.8:{first}"] = source
    return answers


def _run_dtype(dtype_name: str, dtype, args, units: list[dict], rows: list[dict],
               answers: dict[str, dict], output: Path) -> dict:
    import torch

    started = time.perf_counter()
    tokenizer, model = load_model(repo_path(args.model), args.device, dtype)
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("loaded model has no parameters to identify its dtype")
    parameter_dtype = str(parameter.dtype)
    dtype_name_from_model = {
        "torch.bfloat16": "bf16",
        "torch.float32": "float32",
    }.get(parameter_dtype)
    if dtype_name_from_model != dtype_name:
        raise ValueError(f"requested {dtype_name} but loaded model parameters are {parameter_dtype}")

    # run_path2 resumes from its output file's stored 0.8 T. Keep each dtype in
    # its own file and refuse to reuse records without matching dtype evidence.
    if output.exists():
        existing = run119b.read_output_by_key(output, "key")
        for record in existing.values():
            if (record.get("model_dtype") != dtype_name
                    or record.get("model_parameter_dtype") != parameter_dtype):
                raise ValueError(f"{output} contains rows without matching {dtype_name} dtype evidence")

    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, FRAME, args.device)
    base_cache, cache_by_dose = build_dtype_push_caches(units, layers, medians)
    result = run119b.run_path2(
        units, rows, DOSES, output, tokenizer, model, layers,
        base_cache, cache_by_dose, answers, args.device, args.batch,
    )
    stamped = []
    for line_no, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if (record.get("model_dtype") not in (None, dtype_name)
                or record.get("model_parameter_dtype") not in (None, parameter_dtype)):
            raise ValueError(f"{output}:{line_no} has conflicting model dtype evidence")
        record["model_dtype"] = dtype_name
        record["model_parameter_dtype"] = parameter_dtype
        stamped.append(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")))
    temporary = output.with_name(output.name + ".dtype-tmp")
    temporary.write_text("\n".join(stamped) + ("\n" if stamped else ""), encoding="utf-8")
    temporary.replace(output)
    return {"dtype": dtype_name, "output": str(output.relative_to(ROOT)),
            "model_parameter_dtype": parameter_dtype,
            "rows_expected": len(rows), "doses": list(DOSES),
            "path2": result, "elapsed_seconds": time.perf_counter() - started}


def clear_device_cache(device: str) -> None:
    import torch

    gc.collect()
    if device.startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device.startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict:
    import torch

    units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    rows = rows119c.draw_path2_rows(units, rows119b.all_path2_rows(units))
    answers_by_id = run119b.read_jsonl_readonly(repo_path(ANSWER_PATH), "row_id")
    answers = _answers_for_sample(answers_by_id, rows)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    runs = []
    outputs = {"bf16": out_dir / "interaction-bf16.jsonl",
               "float32": out_dir / "interaction-float32.jsonl"}
    if outputs["bf16"].resolve() == outputs["float32"].resolve():
        raise ValueError("bf16 and float32 runs must have separate output files")
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("float32", torch.float32)):
        if dtype_name not in getattr(args, "dtypes", "bf16,float32").split(","):
            continue
        output = outputs[dtype_name]
        runs.append(_run_dtype(dtype_name, dtype, args, units, rows, answers, output))
        # _run_dtype's local model, hooks and tensor caches are gone before the next load.
        clear_device_cache(args.device)

    report = {"stage": "main", "seed": rows119c.SEED,
              "rows": len(rows), "first_units": len({row["first"] for row in rows}),
              "batch": args.batch, "device": args.device,
              "doses": list(DOSES), "runs": runs,
              "elapsed_seconds": time.perf_counter() - started}
    (out_dir / "timing.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="local pinned model snapshot")
    parser.add_argument("--out", default="results/part119c")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dtypes", default="bf16,float32",
                        help="which runs to do; a resume names the one left, so the finished model is not loaded")
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
