#!/usr/bin/env python3
"""Part 59B: measure later-layer MLP mean activation changes under symmetric clamps."""
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import neuron_report as nr  # noqa: E402
import part26 as p26  # noqa: E402
from run59a import MODELS, _dtype_name, control_units, load_maxima  # noqa: E402

SOURCE_JSON = ROOT / "results/part26/source.json"
SOURCE_NPZ = ROOT / "results/part26/source.npz"
QWEN_MAXIMA = ROOT / "results/part54a/source-pile-maxima-7c19d0120476.json"
OLMO_TARGET_NPZ = ROOT / "results/part26/target-olmo2-1b.npz"
DOSES = (-1.0, 1.0)


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


class _MeanAccumulator:
    """Accumulate an equally weighted mean of per-document token means."""

    def __init__(self):
        self.mask = None
        self.total = None
        self.documents = 0
        self.calls = 0

    def hook(self, _module, args):
        values = args[0].detach().float()
        mask = self.mask.to(values.device, dtype=values.dtype)
        per_document = (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        chunk_total = per_document.sum(dim=0).cpu()
        self.total = chunk_total if self.total is None else self.total + chunk_total
        self.documents += int(per_document.shape[0])
        self.calls += 1

    def mean(self) -> np.ndarray:
        if self.total is None or self.documents <= 0:
            raise RuntimeError("no later-layer MLP activations were captured")
        return (self.total / self.documents).numpy().astype(np.float32, copy=False)


def source_documents(path: str | os.PathLike[str] = SOURCE_JSON) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (data.get("model"), data.get("revision")) != (p26.SOURCE_MODEL, p26.SOURCE_REVISION):
        raise ValueError(f"{path} is not the pinned Part 26 Qwen source record")
    docs = data.get("documents")
    if not isinstance(docs, list) or len(docs) != p26.SOURCE_DOCS:
        raise ValueError(f"{path} must contain the Part 26 selection of {p26.SOURCE_DOCS} documents")
    for index, row in enumerate(docs):
        words = str(row.get("text", "")).split()
        if len(words) != p26.SOURCE_WORDS:
            raise ValueError(f"Part 26 document {index} has {len(words)} words; expected {p26.SOURCE_WORDS}")
    return docs


def tokenize_documents(tok, docs: list[dict]) -> list[dict]:
    if not getattr(tok, "is_fast", True):
        raise ValueError("Part 59B requires the fast tokenizer used by Part 26")
    tokenized = []
    for row in docs:
        encoded = tok(str(row["text"]), return_offsets_mapping=True)
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        if len(ids) != len(offsets):
            raise ValueError("tokenizer returned mismatched input ids and offsets")
        tokenized.append({"ids": ids, "offsets": offsets})
    return tokenized


def capture_means(tok, model, tokenized: list[dict], layers, *, batch_size: int,
                  device: str, clamp_layer: int | None = None, unit: int | None = None,
                  clamp_value: float | None = None,
                  start_layer: int | None = None) -> tuple[np.ndarray, list[int], list[int], dict]:
    """Return concatenated later-layer MLP means; optionally clamp one unit at its layer."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (clamp_layer is None) != (unit is None) or (clamp_layer is None) != (clamp_value is None):
        raise ValueError("clamp_layer, unit and clamp_value must either all be set or all be absent")
    first_layer = (clamp_layer + 1 if clamp_layer is not None else 0) if start_layer is None else start_layer
    if first_layer < 0 or first_layer > len(layers):
        raise ValueError("start_layer is outside the model")
    later_indices = list(range(first_layer, len(layers)))
    if not later_indices:
        raise ValueError("no later layers are available to record")
    accumulators = {index: _MeanAccumulator() for index in later_indices}
    handles = []
    clamp_hook = None
    if clamp_layer is not None:
        down_proj = layers[clamp_layer].mlp.down_proj
        if unit < 0 or unit >= down_proj.in_features:
            raise ValueError(f"clamp unit {unit} is outside layer {clamp_layer}")
        clamp_hook = nr.NeuronHook(down_proj)
        clamp_hook.clamp = (torch.tensor([unit]), torch.tensor([float(clamp_value)]))
    for index in later_indices:
        handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(accumulators[index].hook))
    max_seq = 0
    peak_allocated = 0
    peak_reserved = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        for start in range(0, len(tokenized), batch_size):
            batch = tokenized[start:start + batch_size]
            ids, mask = p26._padded_inputs(batch, tok, device)
            max_seq = max(max_seq, int(ids.shape[1]))
            for accumulator in accumulators.values():
                accumulator.mask = mask
            with torch.no_grad():
                p26._forward_decoder(model, ids, mask)
            if device.startswith("cuda") and torch.cuda.is_available():
                peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
        pieces = [accumulators[index].mean() for index in later_indices]
        means = np.concatenate(pieces).astype(np.float32, copy=False)
        widths = [int(layers[index].mlp.down_proj.in_features) for index in later_indices]
        neuron_layers = np.concatenate([np.full(width, index, dtype=np.int16)
                                        for index, width in zip(later_indices, widths)])
        neuron_units = np.concatenate([np.arange(width, dtype=np.int32) for width in widths])
        if any(acc.documents != len(tokenized) for acc in accumulators.values()):
            raise RuntimeError("a later-layer hook did not observe every document")
        stats = {"max_sequence_tokens": max_seq, "peak_cuda_allocated_bytes": peak_allocated,
                 "peak_cuda_reserved_bytes": peak_reserved,
                 "forward_batches": len(range(0, len(tokenized), batch_size)),
                 "documents": len(tokenized)}
        return means, neuron_layers, neuron_units, stats
    finally:
        for handle in handles:
            handle.remove()
        if clamp_hook is not None:
            clamp_hook.clamp = None
            clamp_hook.remove()


class _TopWindowAccumulator:
    def __init__(self, layer: int, selected_units: list[int], tokenized: list[dict],
                 documents: list[dict], top_n: int):
        self.layer = layer
        self.selected_units = selected_units
        self.tokenized = tokenized
        self.documents = documents
        self.top_n = top_n
        self.mask = None
        self.batch_start = 0
        self.best = {unit: [] for unit in selected_units}

    def hook(self, _module, args):
        values = args[0][..., self.selected_units].detach().float().cpu().numpy()
        mask = self.mask.detach().cpu().numpy().astype(bool)
        for batch_index in range(values.shape[0]):
            document_index = self.batch_start + batch_index
            row = self.documents[document_index]
            token_offsets = self.tokenized[document_index]["offsets"]
            valid = mask[batch_index, :len(token_offsets)].copy()
            valid &= np.asarray([end > start for start, end in token_offsets], dtype=bool)
            if not valid.any():
                continue
            selected = values[batch_index, :len(token_offsets), :]
            valid_positions = np.flatnonzero(valid)
            for column, unit in enumerate(self.selected_units):
                activation_column = selected[valid_positions, column]
                local = int(np.argmax(activation_column))
                token_index = int(valid_positions[local])
                value = float(activation_column[local])
                char_start, char_end = token_offsets[token_index]
                text = str(row["text"])
                window_start = max(0, char_start - 120)
                window_end = min(len(text), char_end + 120)
                candidate = {"layer": self.layer, "unit": int(unit),
                             "activation": value,
                             "document_index": document_index,
                             "line_index": int(row.get("line_index", document_index)),
                             "token_index": token_index,
                             "window": text[window_start:window_end]}
                best = self.best[unit]
                best.append(candidate)
                best.sort(key=lambda item: item["activation"], reverse=True)
                del best[self.top_n:]


def capture_top_windows(tok, model, docs: list[dict], tokenized: list[dict], layers,
                        units_by_layer: dict[int, list[int]], *, batch_size: int,
                        device: str, top_n: int = 3) -> dict[str, list[dict]]:
    """Find each selected later-layer unit's top single-token firing windows in Part 26 text."""
    if len(docs) != len(tokenized) or not docs:
        raise ValueError("documents and tokenized rows must be nonempty and aligned")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    accumulators = {}
    handles = []
    for layer_i, units in units_by_layer.items():
        if not units or layer_i < 0 or layer_i >= len(layers):
            raise ValueError(f"invalid window layer/unit selection: {layer_i}:{units}")
        width = int(layers[layer_i].mlp.down_proj.in_features)
        if any(unit < 0 or unit >= width for unit in units):
            raise ValueError(f"window unit lies outside layer {layer_i} width {width}")
        accumulator = _TopWindowAccumulator(layer_i, units, tokenized, docs, top_n)
        accumulators[layer_i] = accumulator
        handles.append(layers[layer_i].mlp.down_proj.register_forward_pre_hook(accumulator.hook))
    try:
        for start in range(0, len(tokenized), batch_size):
            batch = tokenized[start:start + batch_size]
            ids, mask = p26._padded_inputs(batch, tok, device)
            for accumulator in accumulators.values():
                accumulator.mask = mask
                accumulator.batch_start = start
            with torch.no_grad():
                p26._forward_decoder(model, ids, mask)
        return {f"{layer}:{unit}": accumulators[layer].best[unit]
                for layer, units in units_by_layer.items() for unit in units}
    finally:
        for handle in handles:
            handle.remove()


def _sha(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(data: dict, path: Path) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temp, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def _load_state(path: Path, unit_count: int, neuron_count: int) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.full(neuron_count, np.nan, dtype=np.float32)
    conditions = np.full((unit_count, len(DOSES), neuron_count), np.nan, dtype=np.float32)
    if not path.exists():
        return baseline, conditions
    with np.load(path) as saved:
        prior_baseline = np.asarray(saved["baseline_mean"], dtype=np.float32)
        prior_conditions = np.asarray(saved["condition_mean"], dtype=np.float32)
        if prior_baseline.shape != baseline.shape or prior_conditions.shape != conditions.shape:
            raise ValueError(f"existing activation archive has incompatible shapes: {path}")
        baseline[:] = prior_baseline
        conditions[:] = prior_conditions
    return baseline, conditions


def _persist(path: Path, baseline: np.ndarray, conditions: np.ndarray,
             neuron_layers: np.ndarray, neuron_units: np.ndarray,
             condition_units: list[int]) -> None:
    _save_npz(path, {"baseline_mean": baseline, "condition_mean": conditions,
                     "change": conditions - baseline[None, None, :],
                     "neuron_layer": neuron_layers, "neuron_unit": neuron_units,
                     "condition_unit": np.asarray(condition_units, dtype=np.int32),
                     "doses": np.asarray(DOSES, dtype=np.float32)})


def run(args: argparse.Namespace) -> dict:
    short = args.model
    if short not in MODELS:
        raise ValueError(f"--model must be one of {', '.join(MODELS)}")
    expected_clamp = f"{MODELS[short]['layer']}:{MODELS[short]['unit']}"
    requested_clamp = getattr(args, "clamp_unit", None)
    if requested_clamp and requested_clamp != expected_clamp:
        raise ValueError(f"Part 59 pins {short} to --clamp-unit {expected_clamp}")
    model_spec = p26.MODELS[short]
    model_id = model_spec["id"]
    revision = args.revision or model_spec["revision"]
    if revision != model_spec["revision"]:
        raise ValueError(f"Part 59 pins {short} to revision {model_spec['revision']}")
    root_out = Path(args.out).resolve()
    root_out.mkdir(parents=True, exist_ok=True)
    lock_path = root_out / ".part59b.lock"
    with advisory_lock(lock_path, "59B"):
        return _run_locked(args, short, model_id, revision, root_out)


def _run_locked(args, short, model_id, revision, out_dir):
    spec = MODELS[short]
    doc_rows = source_documents(args.source_json)
    layer_i, unit = spec["layer"], spec["unit"]
    width = int(spec["width"])
    controls = control_units(width, unit, args.seed)
    condition_units = [unit] + controls
    maxima = load_maxima(short, layer_i, condition_units, args.qwen_maxima)
    result_path = out_dir / f"59b-{short}-L{layer_i}u{unit}.npz"
    manifest_path = result_path.with_suffix(".json")
    model_sha_sources = {"source_json": _sha(args.source_json), "source_npz": _sha(SOURCE_NPZ),
                         "run59b.py": _sha(__file__), "part26.py": _sha(p26.__file__)}
    maxima_file = args.qwen_maxima if short == "qwen35-4b" else OLMO_TARGET_NPZ
    model_sha_sources["pile_maxima"] = _sha(maxima_file)
    steps = ["baseline"] + [f"{condition}:{dose:+g}"
                             for condition in condition_units for dose in DOSES]
    manifest = {"tool": "run59b.py", "model": model_id, "revision": revision,
                "short_name": short, "dtype": _dtype_name(short),
                "twin": {"layer": layer_i, "unit": unit},
                "condition_units": condition_units, "control_units": controls,
                "pile_maxima": {str(key): value for key, value in maxima.items()},
                "doses": list(DOSES), "documents": len(doc_rows),
                "document_selection": "Part 26 saved corpus record: 1000 JSONL rows, first 256 words",
                "mean_definition": "equal-weight mean over each document's valid-token mean of MLP down_proj inputs",
                "batch_size": args.batch_size, "inputs_sha256": model_sha_sources,
                "steps": steps, "completed": [], "status": "running",
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    force = bool(args.force)
    old = None
    base_identity = ("model", "revision", "short_name", "dtype", "twin", "condition_units",
                     "control_units", "pile_maxima", "doses", "documents", "document_selection",
                     "mean_definition", "batch_size", "inputs_sha256", "steps")
    full_identity = base_identity + ("later_layers", "neuron_count")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches_base = all(old.get(key) == manifest.get(key) for key in base_identity)
        if not matches_base and not force:
            raise ValueError(f"existing Part 59B manifest does not match this run: {manifest_path}")
        if matches_base and old.get("status") == "complete" and not force:
            complete = False
            if (isinstance(old.get("completed"), list) and set(old["completed"]) == set(steps)
                    and result_path.is_file()
                    and isinstance(old.get("neuron_count"), int)):
                try:
                    with np.load(result_path) as saved:
                        expected_neurons = old["neuron_count"]
                        complete = (saved["baseline_mean"].shape == (expected_neurons,)
                                    and saved["condition_mean"].shape ==
                                    (len(condition_units), len(DOSES), expected_neurons)
                                    and np.isfinite(saved["baseline_mean"]).all()
                                    and np.isfinite(saved["condition_mean"]).all()
                                    and np.asarray(saved["doses"]).tolist() == list(DOSES))
                except (OSError, KeyError, ValueError):
                    complete = False
            if complete:
                raise FileExistsError(f"59B is already complete: {manifest_path}; pass --force to rerun")
    elif result_path.exists() and not force:
        raise FileExistsError(f"59B archive exists without a manifest: {result_path}")

    tok = model = None
    manifest_written = False
    try:
        tok, model = p26._load_model_offline(model_id, revision, args.device,
                                             p26.MODELS[short].get("dtype", "bf16"))
        _, layers, _ = nr.find_text_layers(model)
        if layer_i >= len(layers):
            raise ValueError(f"registered layer {layer_i} exceeds model depth {len(layers)}")
        actual_width = int(layers[layer_i].mlp.down_proj.in_features)
        if unit >= actual_width or actual_width != width:
            raise ValueError(f"registered layer width is {actual_width}; expected {width}")
        dtype = str(next(model.parameters()).dtype)
        if dtype != manifest["dtype"]:
            raise ValueError(f"loaded model dtype is {dtype}; expected {manifest['dtype']}")
        later_indices = list(range(layer_i + 1, len(layers)))
        neuron_layers = np.concatenate([
            np.full(int(layers[index].mlp.down_proj.in_features), index, dtype=np.int16)
            for index in later_indices
        ])
        neuron_units = np.concatenate([
            np.arange(int(layers[index].mlp.down_proj.in_features), dtype=np.int32)
            for index in later_indices
        ])
        manifest["later_layers"] = later_indices
        manifest["neuron_count"] = int(len(neuron_layers))
        if old is not None and not force:
            if not all(old.get(key) == manifest.get(key) for key in full_identity):
                raise ValueError(f"existing Part 59B manifest does not match this model layout: {manifest_path}")
        tokenized = tokenize_documents(tok, doc_rows)
        baseline, condition_means = (
            _load_state(result_path, len(condition_units), len(neuron_layers)) if not force else
            (np.full(len(neuron_layers), np.nan, dtype=np.float32),
             np.full((len(condition_units), len(DOSES), len(neuron_layers)), np.nan, dtype=np.float32)))
        completed = (set() if force else
                     set(old.get("completed", [])) if old is not None else set())
        manifest["completed"] = sorted(completed)
        _atomic_json(manifest, manifest_path)
        manifest_written = True
        if "baseline" not in completed or not np.isfinite(baseline).all():
            baseline, _, _, stats = capture_means(tok, model, tokenized, layers,
                                                   batch_size=args.batch_size, device=args.device,
                                                   start_layer=layer_i + 1)
            manifest["baseline_stats"] = stats
            completed.add("baseline")
            _persist(result_path, baseline, condition_means, neuron_layers, neuron_units,
                     condition_units)
            manifest["completed"] = sorted(completed)
            _atomic_json(manifest, manifest_path)
        for condition_index, clamp_unit in enumerate(condition_units):
            for dose_index, dose in enumerate(DOSES):
                step = f"{clamp_unit}:{dose:+g}"
                if step in completed and np.isfinite(condition_means[condition_index, dose_index]).all():
                    continue
                value = float(dose * maxima[clamp_unit])
                means, _, _, step_stats = capture_means(
                    tok, model, tokenized, layers, batch_size=args.batch_size, device=args.device,
                    clamp_layer=layer_i, unit=clamp_unit, clamp_value=value)
                condition_means[condition_index, dose_index] = means
                manifest.setdefault("condition_stats", {})[step] = step_stats
                completed.add(step)
                _persist(result_path, baseline, condition_means, neuron_layers, neuron_units,
                         condition_units)
                manifest["completed"] = sorted(completed)
                _atomic_json(manifest, manifest_path)
                print(f"59B {short} {step}: {step_stats.get('forward_batches')} batches", flush=True)
        manifest["status"] = "complete"
        manifest["peak_cuda_allocated_bytes"] = max(
            [manifest.get("baseline_stats", {}).get("peak_cuda_allocated_bytes", 0)] +
            [row.get("peak_cuda_allocated_bytes", 0)
             for row in manifest.get("condition_stats", {}).values()])
        manifest["peak_cuda_reserved_bytes"] = max(
            [manifest.get("baseline_stats", {}).get("peak_cuda_reserved_bytes", 0)] +
            [row.get("peak_cuda_reserved_bytes", 0)
             for row in manifest.get("condition_stats", {}).values()])
        _atomic_json(manifest, manifest_path)
        print(f"59B {short}: saved means for {len(neuron_layers)} later-layer neurons to {result_path}")
        return manifest
    except Exception:
        if manifest_written:
            manifest["status"] = "incomplete"
            manifest["completed"] = sorted(manifest.get("completed", []))
            _atomic_json(manifest, manifest_path)
        raise
    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        import gc
        gc.collect()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, default="qwen35-4b")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--clamp-unit", default=None,
                    help="registered Part 59 unit as LAYER:UNIT; any other unit is rejected")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="documents per forward pass; use 1 for Qwen 4B fp32 on a 24 GiB GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "part59b"))
    ap.add_argument("--source-json", default=str(SOURCE_JSON))
    ap.add_argument("--qwen-maxima", default=str(QWEN_MAXIMA))
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
    except FileExistsError as exc:
        if "already complete" not in str(exc):
            raise
        print(exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
