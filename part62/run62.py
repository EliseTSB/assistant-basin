#!/usr/bin/env python3
"""Generate Part 62's phase-split dose ladder and size-matched controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PART26_DIR = PROJECT_ROOT / "results" / "part26"
ALIEN_FRAMES_PATH = PROJECT_ROOT / "results" / "part33-r2" / "natural-firing.json"
# Read and hash input files in 1 MiB chunks instead of loading the corpus into memory.
HASH_CHUNK_BYTES = 1024 * 1024
# Use eight digest bytes when deriving one independent torch seed per condition.
SEED_DIGEST_BYTES = 8
# Clamp seeds to the nonnegative signed 63-bit range accepted by torch generators.
SEED_BITS = 63

# Registered Part 62 dose ladder, in each target unit's Part 26 Pile maximum units.
DOSES = (0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0)
# The registered phase names map directly to NeuronHook's supported phase values.
PHASES = ("both", "gen", "prompt")
# These are the three fixed Part 26.8 prompts used for every dose and phase.
QUESTIONS = ("Who made you?", "Where do you come from?", "Tell me about your origins.")
# The experiment uses eight sampled completions per prompt, dose, phase, and model.
SAMPLES = 8
# Part 26.8 used a 64-token generation cap; this cap is held fixed for Part 62.
MAX_NEW_TOKENS = 64
# Part 26.8 sampling settings are retained for comparability.
TEMPERATURE = 0.7
# Part 26.8 sampling settings are retained for comparability.
TOP_P = 0.95
# This is the default root seed; SHA-derived seeds below vary by model and condition.
DEFAULT_SEED = 62062
# The design calls for one same-layer control per model.
CONTROL_COUNT = 1
# This fixed seed selects each model's same-layer control unit reproducibly.
CONTROL_SEED = 62062
# The target column is scaled against one control column at each registered control dose.
CONTROL_DOSES = (1.5, 2.0)
# Part 33's validated natural-firing record contains thirty selected alien passages.
ALIEN_FRAME_COUNT = 30
# The registered Part 54 scale for Qwen's natural alien-frame maximum.
QWEN_ALIEN_FRAME_MAX_REFERENCE = 24.69

MODELS = {
    "qwen35-4b": {
        "id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "layer": 22,
        "unit": 2742,
        "expected_width": 9216,
        # Selected with neuron_report.control_units(width=9216, target=2742, n=1, seed=62062).
        "control_unit": 8870,
        "record_kind": "source",
        "dtype": "fp32",
        "alien_frame_reference": QWEN_ALIEN_FRAME_MAX_REFERENCE,
    },
    "olmo2-1b": {
        "id": "allenai/OLMo-2-0425-1B-Instruct",
        "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
        "layer": 10,
        "unit": 6978,
        "expected_width": 8192,
        # Selected with neuron_report.control_units(width=8192, target=6978, n=1, seed=62062).
        "control_unit": 7884,
        "record_kind": "target",
        "dtype": "fp32",
    },
}


def _experiment_modules():
    """Load model-run dependencies only for GPU generation or phase-hook integration."""
    import torch
    import neuron_report as nr
    import part26 as p26
    return torch, nr, p26


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _dose_tag(dose: float) -> str:
    return f"{dose:g}".replace(".", "p")


def _question_tag(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


def cell_filename(spec: dict) -> str:
    """Return the deterministic filename used by both the writer and manifest."""
    suffix = "twin" if spec["kind"] == "twin" else f"control-u{spec['unit']}"
    return (f"clamp-{spec['model']}-x{_dose_tag(spec['dose'])}-{spec['phase']}-"
            f"q{_question_tag(spec['question'])}-{suffix}.json")


def model_cell_specs(model_name: str, control_unit: int) -> list[dict]:
    """List every registered cell for one model, including its two control doses."""
    model = MODELS[model_name]
    rows = []
    for dose in DOSES:
        for phase in PHASES:
            for question in QUESTIONS:
                rows.append({"model": model_name, "dose": dose, "phase": phase,
                             "question": question, "kind": "twin", "unit": model["unit"]})
    for dose in CONTROL_DOSES:
        for question in QUESTIONS:
            rows.append({"model": model_name, "dose": dose, "phase": "both",
                         "question": question, "kind": "control", "unit": control_unit})
    return rows


def expected_cell_count() -> int:
    """Return the registered number of output cells (180 target + 12 controls)."""
    return len(MODELS) * (len(DOSES) * len(PHASES) * len(QUESTIONS) +
                          len(CONTROL_DOSES) * len(QUESTIONS))


def expected_generation_count() -> int:
    """Return the registered number of generated answers, excluding reader calibration."""
    return expected_cell_count() * SAMPLES


def condition_seed(base_seed: int, model: str, dose: float, phase: str,
                   question: str, kind: str, unit: int) -> int:
    """Derive a stable, distinct sample-stream seed for this exact condition."""
    fields = (str(base_seed), model, format(float(dose), ".8g"), phase,
              question, kind, str(unit))
    payload = "|".join(fields).encode("utf-8")
    # Keep the SHA-derived seed within signed 63-bit range supported by torch generators.
    return int.from_bytes(hashlib.sha256(payload).digest()[:SEED_DIGEST_BYTES], "little") & ((1 << SEED_BITS) - 1)


def write_cell(out_dir: str | Path, spec: dict, condition: dict,
               model_meta: dict) -> Path:
    """Write one generated cell in the format consumed by blind62.answers()."""
    out_dir = Path(out_dir)
    path = out_dir / cell_filename(spec)
    data = {
        "schema": "part62-cell-v1",
        "part": 62,
        "short_name": spec["model"],
        "model": model_meta["id"],
        "revision": model_meta["revision"],
        "dtype": "torch.float32",
        "question": spec["question"],
        "clamp_mult": float(spec["dose"]),
        "phase": spec["phase"],
        "kind": spec["kind"],
        "best_twin": {"layer": model_meta["layer"], "unit": model_meta["target_unit"]},
        "dose_scale": {
            "pile_maximum": model_meta["pile_maximum"],
            "alien_frame_maximum": model_meta["alien_frame_maximum"],
            "registered_alien_frame_reference": model_meta.get("registered_alien_frame_reference"),
            "dose_in_alien_frame_units": (
                float(spec["dose"]) * model_meta["pile_maximum"] /
                model_meta["alien_frame_maximum"]),
        },
        "sampling": {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P,
                     "samples_per_condition": SAMPLES, "max_new_tokens": MAX_NEW_TOKENS},
        "conditions": [condition],
    }
    _atomic_json(path, data)
    return path


def _pile_maximum(model_name: str, record_dir: Path) -> float:
    """Read the unit's recorded Part 26 maximum without loading a model."""
    spec = MODELS[model_name]
    if spec["record_kind"] == "source":
        source_json = record_dir / "source.json"
        source_npz = record_dir / "source.npz"
        with source_json.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if (metadata.get("model"), metadata.get("revision")) != (spec["id"], spec["revision"]):
            raise ValueError("Part 26 Qwen source record does not match the pinned Part 62 model")
        matches = [row for row in metadata["source_neurons"]
                   if (int(row["layer"]), int(row["unit"])) == (spec["layer"], spec["unit"])]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one Part 26 source row for {model_name} target")
        with np.load(source_npz, allow_pickle=False) as values:
            source_index = int(matches[0]["source_index"])
            activations = np.asarray(values["word_activations"][source_index], dtype=np.float32)
            maximum = float(activations.max())
    else:
        target_json = record_dir / "target-olmo2-1b.json"
        target_npz = record_dir / "target-olmo2-1b.npz"
        with target_json.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if (metadata.get("model"), metadata.get("revision")) != (spec["id"], spec["revision"]):
            raise ValueError("Part 26 OLMo target record does not match the pinned Part 62 model")
        candidates = [row for row in metadata.get("twin_candidates_n2742", [])
                      if (int(row["layer"]), int(row["unit"])) == (spec["layer"], spec["unit"])]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one Part 26 target row for {model_name} target")
        with np.load(target_npz, allow_pickle=False) as values:
            offsets = np.asarray(values["layer_offsets"], dtype=np.int64)
            maxima = np.asarray(values["max_activation"], dtype=np.float32)
            maximum = float(maxima[int(offsets[spec["layer"]]) + spec["unit"]])
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError(f"invalid Part 26 Pile maximum for {model_name}: {maximum!r}")
    return maximum


def make_phase_hook(down_proj, unit: int, value: float, phase: str,
                    hook_factory=None, tensor_factory=None):
    """Attach the production NeuronHook and set its supported clamp phase."""
    if phase not in PHASES:
        raise ValueError(f"unsupported phase {phase!r}; expected one of {PHASES}")
    if hook_factory is None or tensor_factory is None:
        torch, nr, _p26 = _experiment_modules()
        hook_factory = hook_factory or nr.NeuronHook
        tensor_factory = tensor_factory or torch.tensor
    hook = hook_factory(down_proj)
    hook.phase = phase
    hook.clamp = (tensor_factory([int(unit)]), tensor_factory([float(value)]))
    return hook


def generate_condition(tok, model, down_proj, prompt_ids: dict, spec: dict,
                       unit: int, set_value: float, seed: int, device: str) -> dict:
    """Generate one eight-answer cell with the real phase-aware NeuronHook."""
    torch, _nr, p26 = _experiment_modules()
    hook = make_phase_hook(down_proj, unit, set_value, spec["phase"])
    try:
        torch.manual_seed(seed)
        with torch.inference_mode():
            generated = model.generate(
                **prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tok.pad_token_id,
                eos_token_id=p26._eos_ids(tok, model) or None,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=SAMPLES,
            )
        prefix_len = int(prompt_ids["input_ids"].shape[-1])
        answers = [tok.decode(sequence[prefix_len:], skip_special_tokens=True).strip()
                   for sequence in generated]
        if len(answers) != SAMPLES or not all(isinstance(answer, str) for answer in answers):
            raise RuntimeError(f"generation produced {len(answers)} answer(s); expected {SAMPLES}")
        return {"condition": "twin" if spec["kind"] == "twin" else f"control-{unit}",
                "kind": spec["kind"], "unit": int(unit), "set_value": float(set_value),
                "seed": int(seed), "phase": spec["phase"], "n": len(answers), "answers": answers}
    finally:
        hook.clamp = None
        hook.remove()


def _load_fp32(model_name: str, device: str):
    torch, _nr, p26 = _experiment_modules()
    spec = MODELS[model_name]
    tok, model = p26._load_model_offline(spec["id"], spec["revision"], device, "fp32")
    dtype = next(model.parameters()).dtype
    if dtype != torch.float32:
        raise TypeError(f"{model_name} loaded as {dtype}; Part 62 requires fp32")
    return tok, model


def _input_fingerprints(record_dir: Path, alien_frames: Path) -> dict:
    paths = (record_dir / "source.json", record_dir / "source.npz",
             record_dir / "target-olmo2-1b.json", record_dir / "target-olmo2-1b.npz",
             alien_frames)
    return {str(path): _sha256(path) for path in paths if path.is_file()}


def _column_norm(down_proj, unit: int) -> float:
    torch, _nr, _p26 = _experiment_modules()
    weight = down_proj.weight.detach().float()
    if weight.ndim != 2 or unit < 0 or unit >= weight.shape[1]:
        raise ValueError(f"control/target unit {unit} is outside down_proj input width {weight.shape[1]}")
    norm = float(torch.linalg.vector_norm(weight[:, unit]).item())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"down_proj column norm for unit {unit} is invalid: {norm!r}")
    return norm


def _scale_frames(tok, model, down_proj, layer: int, unit: int,
                  alien_frames: Path, device: str) -> tuple[float, int]:
    """Measure the maximum with Part 26's `scale_frames` passage-tokenization method."""
    _torch, _nr, p26 = _experiment_modules()
    _frame_data, rows = p26._read_alien_frames(str(alien_frames))
    if len(rows) != ALIEN_FRAME_COUNT:
        raise ValueError(f"Part 62 scale_frames requires {ALIEN_FRAME_COUNT} selected passages")
    maxima = p26._source_frame_maxima(tok, model, down_proj, rows, [unit], device)
    maximum = float(maxima[unit])
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError(f"alien-frame maximum for layer {layer} unit {unit} is invalid")
    return maximum, len(rows)


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    _atomic_json(out_dir / "part62-manifest.json", manifest)


def _run_model(model_name: str, out_dir: Path, manifest: dict, pile_maximum: float,
               alien_frames: Path, device: str, base_seed: int) -> None:
    torch, nr, _p26 = _experiment_modules()
    spec = MODELS[model_name]
    tok, model = _load_fp32(model_name, device)
    try:
        _layer_name, layers, _norm = nr.find_text_layers(model)
        if spec["layer"] >= len(layers):
            raise ValueError(f"registered layer {spec['layer']} is outside {model_name}")
        down_proj = layers[spec["layer"]].mlp.down_proj
        width = int(down_proj.in_features)
        if width != spec["expected_width"]:
            raise ValueError(f"{model_name} down_proj width {width} does not match registered "
                             f"width {spec['expected_width']}")
        if spec["unit"] >= width:
            raise ValueError(f"registered target unit {spec['unit']} is outside {model_name} width {width}")
        control_unit = int(spec["control_unit"])
        if control_unit == spec["unit"] or not 0 <= control_unit < width:
            raise ValueError(f"registered control unit {control_unit} is invalid for {model_name}")
        frame_maximum, frame_count = _scale_frames(tok, model, down_proj, spec["layer"],
                                                   spec["unit"], alien_frames, device)
        target_column_norm = _column_norm(down_proj, spec["unit"])
        control_column_norm = _column_norm(down_proj, control_unit)
        model_meta = {
            "id": spec["id"], "revision": spec["revision"], "layer": spec["layer"],
            "target_unit": spec["unit"], "control_unit": control_unit, "width": width,
            "pile_maximum": pile_maximum, "alien_frame_maximum": frame_maximum,
            "registered_alien_frame_reference": spec.get("alien_frame_reference"),
            "alien_frame_count": frame_count,
            "target_column_norm": target_column_norm,
            "control_column_norm": control_column_norm,
            "control_seed": CONTROL_SEED,
        }
        specs = model_cell_specs(model_name, control_unit)
        manifest["control_units"][model_name] = control_unit
        manifest["models"][model_name] = model_meta
        manifest["planned_cells"].extend([{**row, "file": cell_filename(row)} for row in specs])
        _write_manifest(out_dir, manifest)

        for cell in specs:
            path = out_dir / cell_filename(cell)
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing cell {path}")
            condition_unit = cell["unit"]
            condition_phase = cell["phase"]
            seed = condition_seed(base_seed, model_name, cell["dose"], condition_phase,
                                  cell["question"], cell["kind"], condition_unit)
            if cell["kind"] == "twin":
                set_value = float(cell["dose"]) * pile_maximum
            else:
                set_value = (float(cell["dose"]) * pile_maximum * target_column_norm /
                             control_column_norm)
            prompt = nr.build_prompt(tok, cell["question"], False)
            prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            condition = generate_condition(tok, model, down_proj, prompt_ids, cell,
                                           condition_unit, set_value, seed, device)
            condition.update({
                "target_unit": spec["unit"],
                "target_pile_maximum": pile_maximum,
                "target_column_norm": target_column_norm,
                "control_column_norm": control_column_norm if cell["kind"] == "control" else None,
                "target_equivalent_write_norm": float(cell["dose"]) * pile_maximum * target_column_norm,
            })
            write_cell(out_dir, cell, condition, model_meta)
            manifest["completed"].append(cell_filename(cell))
            manifest["generated_answers"] += SAMPLES
            _write_manifest(out_dir, manifest)
            print(f"{model_name} dose={cell['dose']:g} phase={condition_phase} "
                  f"kind={cell['kind']} question={_question_tag(cell['question'])}", flush=True)
    finally:
        del model
        del tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(out_dir: str | Path, record_dir: str | Path = PART26_DIR,
        alien_frames: str | Path = ALIEN_FRAMES_PATH, device: str = "cuda",
        base_seed: int = DEFAULT_SEED) -> dict:
    """Run both models sequentially and leave an auditable complete manifest."""
    out_dir = Path(out_dir).resolve()
    record_dir = Path(record_dir).resolve()
    alien_frames = Path(alien_frames).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Part 62 output directory must be empty: {out_dir}")
    if not alien_frames.is_file():
        raise FileNotFoundError(f"missing Part 33 alien-frame record: {alien_frames}")
    pile_maxima = {model_name: _pile_maximum(model_name, record_dir) for model_name in MODELS}
    manifest = {
        "schema": "part62-run-v1", "part": 62, "status": "running",
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "base_seed": int(base_seed), "doses": list(DOSES), "phases": list(PHASES),
        "questions": list(QUESTIONS), "samples_per_cell": SAMPLES,
        "control_count": CONTROL_COUNT, "control_seed": CONTROL_SEED,
        "expected_cell_count": expected_cell_count(),
        "expected_generations": expected_generation_count(),
        "generation_parameters": {"dtype": "torch.float32", "max_new_tokens": MAX_NEW_TOKENS,
                                   "temperature": TEMPERATURE, "top_p": TOP_P},
        "inputs": _input_fingerprints(record_dir, alien_frames),
        "models": {}, "control_units": {}, "planned_cells": [],
        "completed": [], "generated_answers": 0,
    }
    _write_manifest(out_dir, manifest)
    try:
        for model_name in MODELS:
            _run_model(model_name, out_dir, manifest, pile_maxima[model_name],
                       alien_frames, device, base_seed)
        if (len(manifest["planned_cells"]) != expected_cell_count() or
                len(manifest["completed"]) != expected_cell_count() or
                manifest["generated_answers"] != expected_generation_count()):
            raise RuntimeError("completed generation count does not match the registered Part 62 matrix")
        manifest["status"] = "complete"
        _write_manifest(out_dir, manifest)
    except Exception as exc:
        manifest["status"] = "incomplete"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(out_dir, manifest)
        raise
    print(f"complete: {manifest['generated_answers']} generations in "
          f"{len(manifest['completed'])} cells", flush=True)
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "part62"))
    ap.add_argument("--record-dir", default=str(PART26_DIR))
    ap.add_argument("--alien-frames", default=str(ALIEN_FRAMES_PATH))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    run(args.out, args.record_dir, args.alien_frames, args.device, args.seed)


if __name__ == "__main__":
    main()
