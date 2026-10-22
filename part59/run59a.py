#!/usr/bin/env python3
"""Part 59A: generate story paragraphs under the registered neuron clamps."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import neuron_report as nr  # noqa: E402
import part26 as p26  # noqa: E402

PROMPTS_PATH = Path(__file__).with_name("prompts59a.json")
MODELS = {
    "qwen35-4b": {"layer": 22, "unit": 2742, "width": 9216},
    "olmo2-1b": {"layer": 10, "unit": 6978, "width": 8192},
}
DOSES = (-2.0, -1.0, 0.0, 1.0, 2.0)
CONTROL_DOSES = frozenset((-2.0, 2.0))
SAMPLES = 8
MAX_NEW = 60
CONTROL_COUNT = 3
PART54_MAXIMA = ROOT / "results/part54a/source-pile-maxima-7c19d0120476.json"
OLMO_TARGET_NPZ = ROOT / "results/part26/target-olmo2-1b.npz"
OLMO_TARGET_JSON = ROOT / "results/part26/target-olmo2-1b.json"


def prompts(path: str | os.PathLike[str] = PROMPTS_PATH) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    template = str(data["prompt_template"])
    subjects = data["subjects"]
    if len(subjects) != 12 or len(set(subjects)) != 12:
        raise ValueError("59A requires exactly 12 unique neutral subjects")
    rows = [{"id": f"p{i + 1:02d}", "subject": str(subject),
             "question": template.format(subject=subject)}
            for i, subject in enumerate(subjects)]
    if any(not row["question"].startswith("Write the first paragraph of a story about a ")
           for row in rows):
        raise ValueError("unexpected 59A prompt wording")
    return rows


def control_units(width: int, unit: int, seed: int) -> list[int]:
    # Match Part 26/54 exactly: draw the fixed twenty-unit pool, then use its first three.
    return nr.control_units(width, unit, 20, seed)[:CONTROL_COUNT]


def _dtype_name(short: str) -> str:
    return "torch.float32" if p26.MODELS[short].get("dtype", "bf16") == "fp32" else "torch.bfloat16"


@contextmanager
def advisory_lock(path: str | os.PathLike[str], phase: str):
    """Hold a nonblocking OS lock for this process; SIGKILL releases the flock."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"another Part {phase} run holds the output lock: {path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)


def _qwen_maxima(path: str | os.PathLike[str], units: list[int]) -> dict[int, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    inputs = data.get("inputs", {})
    if (inputs.get("model"), inputs.get("revision"), inputs.get("layer")) != (
            p26.SOURCE_MODEL, p26.SOURCE_REVISION, 22):
        raise ValueError(f"{path} is not the pinned Part 26 Qwen pile-maxima cache")
    source_record = json.loads((ROOT / "results/part26/source.json").read_text(encoding="utf-8"))
    if (inputs.get("corpus_sha256") != source_record.get("input_shas", {}).get("corpus")
            or inputs.get("source_json_sha256") != p26._sha(ROOT / "results/part26/source.json")
            or inputs.get("source_npz_sha256") != p26._sha(ROOT / "results/part26/source.npz")
            or data.get("units") != inputs.get("units")):
        raise ValueError(f"{path} does not match the checked-in Part 26 source record")
    if not set(units).issubset({int(unit) for unit in data.get("units", [])}):
        raise ValueError(f"{path} was not computed for every requested target/control unit")
    maxima = {int(unit): float(value) for unit, value in data.get("maxima", {}).items()}
    missing = [unit for unit in units if unit not in maxima]
    if missing:
        raise ValueError(f"{path} lacks Part 26 maxima for units {missing}; regenerate it with Part 26")
    if any(not np.isfinite(maxima[unit]) or maxima[unit] <= 0 for unit in units):
        raise ValueError("Part 26 pile maxima must be finite and positive")
    return {unit: maxima[unit] for unit in units}


def _olmo_maxima(path: str | os.PathLike[str], layer: int, units: list[int]) -> dict[int, float]:
    target = json.loads(OLMO_TARGET_JSON.read_text(encoding="utf-8"))
    if (target.get("model"), target.get("revision"), target.get("short_name")) != (
            p26.MODELS["olmo2-1b"]["id"], p26.MODELS["olmo2-1b"]["revision"], "olmo2-1b"):
        raise ValueError(f"{OLMO_TARGET_JSON} is not the pinned Part 26 OLMo target record")
    if not any((int(row.get("layer", -1)), int(row.get("unit", -1))) == (layer, 6978)
               for row in target.get("twin_candidates_n2742", [])):
        raise ValueError("Part 26 OLMo target record lacks the registered 10:6978 twin")
    with np.load(path) as data:
        widths = np.asarray(data["layer_widths"], dtype=np.int64)
        offsets = np.asarray(data["layer_offsets"], dtype=np.int64)
        values = np.asarray(data["max_activation"], dtype=np.float32)
    if layer >= len(widths) or any(unit < 0 or unit >= widths[layer] for unit in units):
        raise ValueError("requested OLMo unit is outside the saved Part 26 activation record")
    result = {unit: float(values[offsets[layer] + unit]) for unit in units}
    if any(not np.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("Part 26 pile maxima must be finite and positive")
    return result


def load_maxima(short: str, layer: int, units: list[int], qwen_maxima: str) -> dict[int, float]:
    if short == "qwen35-4b":
        return _qwen_maxima(qwen_maxima, units)
    if short == "olmo2-1b":
        return _olmo_maxima(OLMO_TARGET_NPZ, layer, units)
    raise ValueError(f"unknown Part 59 model {short!r}")


def _input_shas(short: str, prompts_path: str, qwen_maxima: str) -> dict[str, str]:
    def sha(path: str | os.PathLike[str]) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    result = {"prompts59a": sha(prompts_path), "part26.py": sha(p26.__file__),
              "neuron_report.py": sha(nr.__file__)}
    if short == "qwen35-4b":
        result["part54_pile_maxima"] = sha(qwen_maxima)
        result["part26_source_json"] = sha(ROOT / "results/part26/source.json")
        result["part26_source_npz"] = sha(ROOT / "results/part26/source.npz")
    else:
        result["part26_olmo_target_npz"] = sha(OLMO_TARGET_NPZ)
        result["part26_olmo_target_json"] = sha(OLMO_TARGET_JSON)
    return result


def _valid_result(path: Path, prompt: str, dose: float, condition_names: list[str]) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    conditions = data.get("conditions")
    return (data.get("question") == prompt and data.get("clamp_mult") == dose
            and isinstance(conditions, list)
            and [row.get("condition") for row in conditions] == condition_names
            and all(len(row.get("answers", [])) == SAMPLES for row in conditions))


def _atomic_json(data: dict, path: Path) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run(args: argparse.Namespace) -> dict:
    short = args.model
    if short not in MODELS:
        raise ValueError(f"--model must be one of {', '.join(MODELS)}")
    spec = MODELS[short]
    expected_clamp = f"{spec['layer']}:{spec['unit']}"
    requested_clamp = getattr(args, "clamp_unit", None)
    if requested_clamp and requested_clamp != expected_clamp:
        raise ValueError(f"Part 59 pins {short} to --clamp-unit {expected_clamp}")
    part_spec = p26.MODELS[short]
    model_id = part_spec["id"]
    revision = args.revision or part_spec["revision"]
    if revision != part_spec["revision"]:
        raise ValueError(f"Part 59 pins {short} to revision {part_spec['revision']}")
    prompts_path = str(Path(args.prompts).resolve())
    prompt_rows = prompts(prompts_path)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".part59a.lock"
    with advisory_lock(lock_path, "59A"):
        return _run_locked(args, short, spec, model_id, revision, prompts_path, prompt_rows, out_dir)


def _run_locked(args, short, spec, model_id, revision, prompts_path, prompt_rows, out_dir):
    layer_i, unit = spec["layer"], spec["unit"]
    width = int(spec["width"])
    controls = control_units(width, unit, args.seed)
    units = [unit] + controls
    maxima = load_maxima(short, layer_i, units, args.qwen_maxima)
    inputs = _input_shas(short, prompts_path, args.qwen_maxima)
    steps = []
    for dose in DOSES:
        names = (["unclamped"] if dose == 0 else ["twin"])
        if dose in CONTROL_DOSES:
            names += [f"control-{control}" for control in controls]
        for prompt_row in prompt_rows:
            path_args = argparse.Namespace(clamp_unit=f"{layer_i}:{unit}", clamp_mult=dose,
                                          question=prompt_row["question"], out=str(out_dir))
            result_path = Path(p26._clamp_output_path(path_args, short, layer_i, unit)).resolve()
            steps.append({"prompt_id": prompt_row["id"], "question": prompt_row["question"],
                          "dose": dose, "path": str(result_path), "conditions": names})
    manifest_path = out_dir / f"ladder59a-{short}-L{layer_i}u{unit}.json"
    manifest = {"tool": "run59a.py", "model": model_id, "revision": revision,
                "short_name": short, "twin": {"layer": layer_i, "unit": unit},
                "dtype": _dtype_name(short), "seed": args.seed, "doses": list(DOSES),
                "samples_per_prompt_dose": SAMPLES, "max_new_tokens": MAX_NEW,
                "control_doses": sorted(CONTROL_DOSES), "control_units": controls,
                "prompts": prompt_rows, "inputs_sha256": inputs, "steps": steps,
                "expected_generations": 12 * 5 * 8 + 12 * 2 * 3 * 8,
                "completed": [], "status": "running"}
    force = bool(args.force)
    identity = ("model", "revision", "short_name", "twin", "dtype", "seed", "doses",
                "samples_per_prompt_dose", "max_new_tokens", "control_doses", "control_units",
                "prompts", "inputs_sha256", "steps", "expected_generations")
    old = None
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches = all(old.get(key) == manifest.get(key) for key in identity)
        if not matches and not force:
            raise ValueError(f"existing 59A manifest does not match this run: {manifest_path}")
        all_valid = all(_valid_result(Path(step["path"]), step["question"], step["dose"],
                                      step["conditions"]) for step in steps)
        if matches and old.get("status") == "complete" and all_valid and not force:
            raise FileExistsError(f"59A is already complete: {manifest_path}; pass --force to rerun")
    elif not force:
        existing = next((Path(step["path"]) for step in steps if Path(step["path"]).exists()), None)
        if existing:
            raise FileExistsError(f"59A output exists without a matching manifest: {existing}")

    if force:
        expected_paths = {Path(step["path"]).resolve() for step in steps}
        old_paths = [Path(step["path"]).resolve() for step in (old or {}).get("steps", [])
                     if isinstance(step, dict) and step.get("path")]
        old_paths.extend(path.resolve() for path in out_dir.glob("clamp-*.json"))
        for path in old_paths:
            if path not in expected_paths and path.parent == out_dir and path.exists():
                path.unlink()

    _atomic_json(manifest, manifest_path)
    completed = set()
    tok = model = None
    try:
        tok, model = p26._load_model_offline(model_id, revision, args.device,
                                             p26.MODELS[short].get("dtype", "bf16"))
        _, layers, _ = nr.find_text_layers(model)
        down_proj = layers[layer_i].mlp.down_proj
        actual_width = int(down_proj.in_features)
        if actual_width != width:
            raise ValueError(f"registered layer width is {actual_width}; expected {width}")
        model_dtype = str(next(model.parameters()).dtype)
        if model_dtype != manifest["dtype"]:
            raise ValueError(f"loaded model dtype is {model_dtype}; expected {manifest['dtype']}")
        for step in steps:
            path = Path(step["path"])
            if not force and path.exists() and _valid_result(path, step["question"],
                                                             step["dose"], step["conditions"]):
                completed.add(str(path))
                manifest["completed"] = [row["path"] for row in steps if row["path"] in completed]
                _atomic_json(manifest, manifest_path)
                continue
            row = _generate_step(tok, model, down_proj, step, spec, controls, maxima,
                                 model_id, revision, short, args)
            _atomic_json(row, path)
            if not _valid_result(path, step["question"], step["dose"], step["conditions"]):
                raise RuntimeError(f"generated result failed validation: {path}")
            completed.add(str(path))
            manifest["completed"] = [item["path"] for item in steps if item["path"] in completed]
            _atomic_json(manifest, manifest_path)
        manifest["status"] = "complete"
        _atomic_json(manifest, manifest_path)
        print(f"59A {short}: {len(steps)} prompt-dose files; {manifest['expected_generations']} samples")
        return manifest
    except Exception:
        manifest["status"] = "incomplete"
        _atomic_json(manifest, manifest_path)
        raise
    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        import gc
        gc.collect()
        if args.device.startswith("cuda") and hasattr(__import__("torch"), "cuda"):
            __import__("torch").cuda.empty_cache()


def _generate_step(tok, model, down_proj, step, spec, controls, maxima,
                   model_id, revision, short, args):
    dose = float(step["dose"])
    prompt = nr.build_prompt(tok, step["question"], False)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    condition_names = step["conditions"]
    rows = []
    if dose == 0:
        conditions = [("unclamped", None, None)]
    else:
        conditions = [("twin", spec["unit"], dose * maxima[spec["unit"]])]
        if dose in CONTROL_DOSES:
            conditions.extend((f"control-{control}", control, dose * maxima[control])
                              for control in controls)
    for name, unit, value in conditions:
        seed = p26._condition_seed(args.seed, name)
        rows.append(p26._generate_condition(tok, model, down_proj, prompt_ids, name, unit,
                                            value, seed, SAMPLES, MAX_NEW, args.device,
                                            args.max_gpu_gb))
    return {
        **p26._header(model_id, revision, {"prompts59a": args.prompts,
                                           "part26.py": p26.__file__, "neuron_report.py": nr.__file__}),
        "short_name": short,
        "question": step["question"],
        "prompt": prompt,
        "system_message": None,
        "sampling": {"do_sample": True, "temperature": 0.7, "top_p": 0.95,
                     "samples_per_condition": SAMPLES, "max_new_tokens": MAX_NEW},
        "seed_policy": ("Part 26-compatible SHA-derived seed per condition; condition seeds are reused "
                        "across prompts and doses"),
        "clamp_mult": dose,
        "scale_definition": "Part 26 Pile-record maximum down_proj input activation for each unit",
        "scale_record": {
            "path": str(Path(args.qwen_maxima).resolve() if short == "qwen35-4b" else OLMO_TARGET_NPZ),
            "sha256": p26._sha(args.qwen_maxima if short == "qwen35-4b" else OLMO_TARGET_NPZ),
            "maxima": {str(k): maxima[k] for k in [spec["unit"]] + controls}},
        "best_twin": {"layer": spec["layer"], "unit": spec["unit"],
                      "pile_corpus_maximum": maxima[spec["unit"]],
                      "largest_activation": maxima[spec["unit"]],
                      "clamp_value": dose * maxima[spec["unit"]]},
        "random_controls": {"seed": args.seed, "layer": spec["layer"],
                            "units": controls if dose in CONTROL_DOSES else [],
                            "count": len(controls) if dose in CONTROL_DOSES else 0,
                            "planned_units": controls,
                            "largest_activations": {str(control): maxima[control] for control in controls}},
        "conditions": rows,
        "prompt_id": step["prompt_id"],
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--clamp-unit", default=None,
                    help="registered Part 59 unit as LAYER:UNIT; any other unit is rejected")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gpu-gb", type=float, default=22.0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "part59a"))
    ap.add_argument("--prompts", default=str(PROMPTS_PATH))
    ap.add_argument("--qwen-maxima", default=str(PART54_MAXIMA))
    ap.add_argument("--force", action="store_true")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
