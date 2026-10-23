#!/usr/bin/env python3
"""Part 61: measure size-matched downstream activation changes in fp32."""
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

# These values are registered in PART61-63-DESIGN.md; tests pin them.
MODEL_SPECS = {
    "qwen35-4b": {"layer": 22, "unit": 2742, "width": 9216},  # Registered Qwen layer/unit/width.
    "olmo2-1b": {"layer": 10, "unit": 6978, "width": 8192},  # Registered OLMo layer/unit/width.
}
WRITE_SIZES = (0.25, 0.5, 1.0, 2.0)  # Registered write-size multipliers.
SIGNS = (-1.0, 1.0)  # Both symmetric clamp signs are measured.
DOCUMENT_COUNT = 200  # First 200 documents of the saved 1,000-document Part 26 corpus.
CANDIDATE_POOL_SIZE = 20  # Existing Part 26/59 same-layer candidate pool size.
PRIOR_CONTROL_COUNT = 3  # First three candidates were used as the Part 59 controls.
CONTROL_COUNT = 10  # Part 61 adds ten controls after excluding the prior three.
CANDIDATE_POOL_SEED = 0  # Part 26/59 seed used to form the original 20-candidate pool.
CONTROL_DRAW_SEED = 61  # Fixed Part 61 seed used to draw ten from the remaining candidates.
CONTROL_PERCENTILE = 99.0  # Registered pooled percentile at each write size.
NULL_THRESHOLD = 0.00005  # Registered absolute-change floor for a symmetric responder.
DEFAULT_BATCH_SIZE = 1  # Stream documents one at a time for the 4B fp32 model.

SOURCE_JSON = ROOT / "results/part26/source.json"
SOURCE_NPZ = ROOT / "results/part26/source.npz"
QWEN_MAXIMA = ROOT / "results/part54a/source-pile-maxima-7c19d0120476.json"
OLMO_TARGET_NPZ = ROOT / "results/part26/target-olmo2-1b.npz"
OLMO_TARGET_JSON = ROOT / "results/part26/target-olmo2-1b.json"


def _sha(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def advisory_lock(path: str | os.PathLike[str]):
    """Hold a nonblocking Linux advisory lock; process death releases it."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"another Part 61 run holds the output lock: {path}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)


class _MeanAccumulator:
    """Accumulate equally weighted per-document valid-token means."""

    def __init__(self):
        self.mask = None
        self.total = None
        self.documents = 0

    def hook(self, _module, args):
        values = args[0].detach().float()
        mask = self.mask.to(values.device, dtype=values.dtype)
        per_document = (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        chunk_total = per_document.sum(dim=0).cpu()
        self.total = chunk_total if self.total is None else self.total + chunk_total
        self.documents += int(per_document.shape[0])

    def mean(self) -> np.ndarray:
        if self.total is None or self.documents <= 0:
            raise RuntimeError("no later-layer MLP activations were captured")
        return (self.total / self.documents).numpy().astype(np.float32, copy=False)


class _ResidualNormAccumulator:
    """Capture each document's mean L2 norm at the next decoder-layer input."""

    def __init__(self):
        self.mask = None
        self.rows: list[np.ndarray] = []

    def hook(self, _module, args):
        import torch

        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("next-layer pre-hook did not receive a residual tensor")
        values = args[0].detach().float()
        mask = self.mask.to(values.device, dtype=values.dtype)
        norms = torch.linalg.vector_norm(values, ord=2, dim=-1)
        per_document = (norms * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        self.rows.append(per_document.cpu().numpy().astype(np.float32, copy=False))

    def values(self) -> np.ndarray:
        if not self.rows:
            raise RuntimeError("next-layer residual norms were not captured")
        return np.concatenate(self.rows).astype(np.float32, copy=False)


def source_documents(path: str | os.PathLike[str] = SOURCE_JSON) -> list[dict]:
    """Validate the saved Part 26 source and return its first 200 documents."""
    import part26 as p26

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
    return docs[:DOCUMENT_COUNT]


def tokenize_documents(tok, docs: list[dict]) -> list[dict]:
    if not getattr(tok, "is_fast", True):
        raise ValueError("Part 61 requires the fast tokenizer used by Part 26")
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
                  device: str, residual_layer: int, clamp_layer: int | None = None,
                  unit: int | None = None, clamp_value: float | None = None,
                  start_layer: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """Capture downstream MLP means and next-layer residual norms, with an optional clamp."""
    import neuron_report as nr
    import part26 as p26
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (clamp_layer is None) != (unit is None) or (clamp_layer is None) != (clamp_value is None):
        raise ValueError("clamp_layer, unit and clamp_value must either all be set or all be absent")
    first_layer = residual_layer if start_layer is None else start_layer
    if first_layer < 0 or first_layer > len(layers):
        raise ValueError("start_layer is outside the model")
    if residual_layer < 0 or residual_layer >= len(layers):
        raise ValueError("next-layer residual layer is outside the model")
    later_indices = list(range(first_layer, len(layers)))
    if not later_indices:
        raise ValueError("no later layers are available to record")
    accumulators = {index: _MeanAccumulator() for index in later_indices}
    residual_accumulator = _ResidualNormAccumulator()
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
    handles.append(layers[residual_layer].register_forward_pre_hook(residual_accumulator.hook))
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
            residual_accumulator.mask = mask
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
        residual_norms = residual_accumulator.values()
        if any(acc.documents != len(tokenized) for acc in accumulators.values()):
            raise RuntimeError("a later-layer hook did not observe every document")
        if len(residual_norms) != len(tokenized):
            raise RuntimeError("next-layer residual hook did not observe every document")
        stats = {"max_sequence_tokens": max_seq, "peak_cuda_allocated_bytes": peak_allocated,
                 "peak_cuda_reserved_bytes": peak_reserved,
                 "forward_batches": len(range(0, len(tokenized), batch_size)),
                 "documents": len(tokenized)}
        return means, neuron_layers, neuron_units, stats, residual_norms
    finally:
        for handle in handles:
            handle.remove()
        if clamp_hook is not None:
            clamp_hook.clamp = None
            clamp_hook.remove()


def select_control_units(width: int, target: int) -> tuple[list[int], list[int], list[int]]:
    """Draw ten controls from the prior 20 candidates, excluding the prior three controls."""
    # Match neuron_report.control_units: sorted candidates sampled without replacement, excluding the target.
    rng = np.random.default_rng(CANDIDATE_POOL_SEED)
    pool = np.setdiff1d(np.arange(width), [target])
    candidate_pool = sorted(int(unit) for unit in rng.choice(
        pool, size=CANDIDATE_POOL_SIZE, replace=False))
    excluded = candidate_pool[:PRIOR_CONTROL_COUNT]
    remaining = [unit for unit in candidate_pool if unit not in set(excluded)]
    if len(remaining) < CONTROL_COUNT:
        raise ValueError("the prior candidate pool has too few units after excluding the old controls")
    chosen = sorted(int(unit) for unit in np.random.default_rng(CONTROL_DRAW_SEED).choice(
        remaining, size=CONTROL_COUNT, replace=False))
    return candidate_pool, excluded, chosen


def down_column_norms(down_proj, units: list[int]) -> dict[int, float]:
    """Return L2 norms of W_down[:, unit], with PyTorch Linear's [out, in] layout."""
    weight = getattr(down_proj, "weight", None)
    if weight is None or not hasattr(weight, "detach") or weight.ndim != 2:
        raise ValueError("down_proj.weight must be a two-dimensional tensor")
    if any(unit < 0 or unit >= weight.shape[1] for unit in units):
        raise ValueError("requested W_down column is outside the projection")
    columns = weight.detach().cpu().numpy().astype(np.float64, copy=False)
    norms = {int(unit): float(np.linalg.norm(columns[:, unit])) for unit in units}
    if any(not np.isfinite(value) or value <= 0 for value in norms.values()):
        raise ValueError("every target/control W_down column norm must be finite and positive")
    return norms


def matched_clamp_value(target_pile_maximum: float, target_column_norm: float,
                        unit_column_norm: float, write_size: float, sign: float = 1.0) -> float:
    """Scale the unit activation so its W_down write norm matches the target at this size."""
    values = (target_pile_maximum, target_column_norm, unit_column_norm, write_size, abs(sign))
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Pile maximum, column norms, write size and sign magnitude must be positive")
    return float(sign * write_size * target_pile_maximum * target_column_norm / unit_column_norm)


def load_target_pile_maximum(short: str, layer: int, unit: int, *, source_json: str,
                             source_npz: str, qwen_maxima: str, olmo_target_npz: str,
                             olmo_target_json: str) -> float:
    """Read the registered target maximum from the saved Part 26 activation record."""
    import part26 as p26

    if short == "qwen35-4b":
        data = json.loads(Path(qwen_maxima).read_text(encoding="utf-8"))
        inputs = data.get("inputs", {})
        source = json.loads(Path(source_json).read_text(encoding="utf-8"))
        if (inputs.get("model"), inputs.get("revision"), inputs.get("layer")) != (
                p26.SOURCE_MODEL, p26.SOURCE_REVISION, layer):
            raise ValueError(f"{qwen_maxima} is not the pinned Qwen Part 26 Pile-maxima cache")
        if (inputs.get("corpus_sha256") != source.get("input_shas", {}).get("corpus")
                or inputs.get("source_json_sha256") != _sha(source_json)
                or inputs.get("source_npz_sha256") != _sha(source_npz)
                or data.get("units") != inputs.get("units")):
            raise ValueError(f"{qwen_maxima} does not match the checked Part 26 source record")
        if unit not in {int(value) for value in data.get("units", [])}:
            raise ValueError(f"{qwen_maxima} lacks the registered target unit {unit}")
        maxima = {int(key): float(value) for key, value in data.get("maxima", {}).items()}
        value = maxima.get(unit)
    elif short == "olmo2-1b":
        target = json.loads(Path(olmo_target_json).read_text(encoding="utf-8"))
        spec = p26.MODELS[short]
        if (target.get("model"), target.get("revision"), target.get("short_name")) != (
                spec["id"], spec["revision"], short):
            raise ValueError(f"{olmo_target_json} is not the pinned Part 26 OLMo target record")
        if not any((int(row.get("layer", -1)), int(row.get("unit", -1))) == (layer, unit)
                   for row in target.get("twin_candidates_n2742", [])):
            raise ValueError("Part 26 OLMo target record lacks the registered 10:6978 twin")
        with np.load(olmo_target_npz) as archive:
            widths = np.asarray(archive["layer_widths"], dtype=np.int64)
            offsets = np.asarray(archive["layer_offsets"], dtype=np.int64)
            maxima = np.asarray(archive["max_activation"], dtype=np.float32)
        if layer >= len(widths) or unit < 0 or unit >= widths[layer]:
            raise ValueError("registered OLMo target lies outside the saved Part 26 activation record")
        value = float(maxima[offsets[layer] + unit])
    else:
        raise ValueError(f"unknown Part 61 model {short!r}")
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError("registered target Pile maximum must be finite and positive")
    return float(value)


def _step_key(unit: int, write_size: float) -> str:
    return f"{unit}:{write_size:g}x"


def _atomic_json(data: dict, path: Path) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temp, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def _load_state(path: Path, unit_count: int, size_count: int, sign_count: int,
                neuron_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.full(neuron_count, np.nan, dtype=np.float32)
    conditions = np.full((unit_count, size_count, sign_count, neuron_count), np.nan, dtype=np.float32)
    residual_by_doc = np.full(DOCUMENT_COUNT, np.nan, dtype=np.float32)
    ratios = np.full((unit_count, size_count, sign_count), np.nan, dtype=np.float32)
    if not path.exists():
        return baseline, conditions, residual_by_doc, ratios
    with np.load(path) as saved:
        prior_baseline = np.asarray(saved["baseline_mean"], dtype=np.float32)
        prior_conditions = np.asarray(saved["condition_mean"], dtype=np.float32)
        prior_residual = np.asarray(saved["baseline_residual_norm_by_doc"], dtype=np.float32)
        prior_ratios = np.asarray(saved["residual_norm_ratio"], dtype=np.float32)
        if (prior_baseline.shape != baseline.shape or prior_conditions.shape != conditions.shape
                or prior_residual.shape != residual_by_doc.shape or prior_ratios.shape != ratios.shape):
            raise ValueError(f"existing activation archive has incompatible shapes: {path}")
        baseline[:] = prior_baseline
        conditions[:] = prior_conditions
        residual_by_doc[:] = prior_residual
        ratios[:] = prior_ratios
    return baseline, conditions, residual_by_doc, ratios


def _persist(path: Path, baseline: np.ndarray, conditions: np.ndarray,
             residual_by_doc: np.ndarray, residual_ratios: np.ndarray,
             neuron_layers: np.ndarray, neuron_units: np.ndarray,
             condition_units: list[int]) -> None:
    _save_npz(path, {
        "baseline_mean": baseline,
        "condition_mean": conditions,
        "change": conditions - baseline[None, None, None, :],
        "baseline_residual_norm_by_doc": residual_by_doc,
        "residual_norm_ratio": residual_ratios,
        "neuron_layer": neuron_layers,
        "neuron_unit": neuron_units,
        "condition_unit": np.asarray(condition_units, dtype=np.int32),
        "write_sizes": np.asarray(WRITE_SIZES, dtype=np.float32),
        "signs": np.asarray(SIGNS, dtype=np.float32),
    })


def run(args: argparse.Namespace) -> dict:
    import part26 as p26

    short = args.model
    if short not in MODEL_SPECS:
        raise ValueError(f"--model must be one of {', '.join(MODEL_SPECS)}")
    spec = MODEL_SPECS[short]
    model_spec = p26.MODELS[short]
    model_id = model_spec["id"]
    revision = args.revision or model_spec["revision"]
    if revision != model_spec["revision"]:
        raise ValueError(f"Part 61 pins {short} to revision {model_spec['revision']}")
    requested_clamp = getattr(args, "clamp_unit", None)
    expected_clamp = f"{spec['layer']}:{spec['unit']}"
    if requested_clamp and requested_clamp != expected_clamp:
        raise ValueError(f"Part 61 pins {short} to --clamp-unit {expected_clamp}")
    root_out = Path(args.out).resolve()
    root_out.mkdir(parents=True, exist_ok=True)
    with advisory_lock(root_out / ".part61.lock"):
        return _run_locked(args, short, model_id, revision, root_out)


def _run_locked(args, short: str, model_id: str, revision: str, out_dir: Path) -> dict:
    import neuron_report as nr
    import part26 as p26
    import torch

    spec = MODEL_SPECS[short]
    docs = source_documents(args.source_json)
    layer_i, target_unit, width = spec["layer"], spec["unit"], spec["width"]
    candidate_pool, prior_controls, controls = select_control_units(width, target_unit)
    condition_units = [target_unit] + controls
    target_maximum = load_target_pile_maximum(
        short, layer_i, target_unit, source_json=args.source_json, source_npz=args.source_npz,
        qwen_maxima=args.qwen_maxima, olmo_target_npz=args.olmo_target_npz,
        olmo_target_json=args.olmo_target_json)
    result_path = out_dir / f"61-{short}-L{layer_i}u{target_unit}.npz"
    manifest_path = result_path.with_suffix(".json")
    input_paths = {"source_json": args.source_json, "source_npz": args.source_npz,
                   "run61.py": __file__, "part26.py": p26.__file__, "neuron_report.py": nr.__file__}
    if short == "qwen35-4b":
        input_paths["target_pile_maxima"] = args.qwen_maxima
    else:
        input_paths["olmo_target_npz"] = args.olmo_target_npz
        input_paths["olmo_target_json"] = args.olmo_target_json
    input_shas = {name: _sha(path) for name, path in input_paths.items()}
    steps = ["baseline"] + [_step_key(unit, size) for unit in condition_units for size in WRITE_SIZES]
    manifest = {
        "tool": "run61.py", "model": model_id, "revision": revision, "short_name": short,
        "dtype": "torch.float32", "twin": {"layer": layer_i, "unit": target_unit},
        "candidate_pool_seed": CANDIDATE_POOL_SEED, "candidate_pool": candidate_pool,
        "prior_controls_excluded": prior_controls, "control_draw_seed": CONTROL_DRAW_SEED,
        "control_units": controls, "condition_units": condition_units,
        "target_pile_maximum": target_maximum, "write_sizes": list(WRITE_SIZES),
        "signs": list(SIGNS), "null_threshold": NULL_THRESHOLD,
        "control_percentile": CONTROL_PERCENTILE, "documents": len(docs),
        "source_documents_total": p26.SOURCE_DOCS,
        "document_selection": "first 200 documents from the Part 26 saved source record",
        "mean_definition": "equal-weight mean over each document's valid-token mean of later-layer MLP down_proj inputs",
        "residual_norm_definition": "per-document mean token L2 norm of the residual stream entering the next decoder layer; report the mean of document-wise clamped/unclamped ratios",
        "batch_size": args.batch_size, "inputs_sha256": input_shas,
        "steps": steps, "completed": [], "status": "running",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    force = bool(args.force)
    old = None
    identity_keys = ("model", "revision", "short_name", "dtype", "twin", "candidate_pool_seed",
                     "candidate_pool", "prior_controls_excluded", "control_draw_seed", "control_units",
                     "condition_units", "target_pile_maximum", "write_sizes", "signs", "null_threshold",
                     "control_percentile", "documents", "source_documents_total", "document_selection",
                     "mean_definition", "residual_norm_definition", "batch_size", "inputs_sha256", "steps")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not all(old.get(key) == manifest.get(key) for key in identity_keys) and not force:
            raise ValueError(f"existing Part 61 manifest does not match this run: {manifest_path}")
        if old.get("status") == "complete" and not force and result_path.is_file():
            try:
                with np.load(result_path) as saved:
                    complete = (set(old.get("completed", [])) == set(steps)
                                and np.isfinite(saved["baseline_mean"]).all()
                                and np.isfinite(saved["condition_mean"]).all()
                                and np.isfinite(saved["residual_norm_ratio"]).all()
                                and np.asarray(saved["write_sizes"]).tolist() == list(WRITE_SIZES)
                                and np.asarray(saved["signs"]).tolist() == list(SIGNS))
            except (OSError, KeyError, ValueError):
                complete = False
            if complete:
                raise FileExistsError(f"Part 61 is already complete: {manifest_path}; pass --force to rerun")
    elif result_path.exists() and not force:
        raise FileExistsError(f"Part 61 archive exists without a manifest: {result_path}")

    tok = model = None
    manifest_written = False
    try:
        tok, model = p26._load_model_offline(model_id, revision, args.device, "fp32")
        _, layers, _ = nr.find_text_layers(model)
        if layer_i + 1 >= len(layers):
            raise ValueError(f"registered layer {layer_i} has no next decoder layer in model depth {len(layers)}")
        down_proj = layers[layer_i].mlp.down_proj
        actual_width = int(down_proj.in_features)
        if actual_width != width or target_unit >= actual_width:
            raise ValueError(f"registered layer width is {actual_width}; expected {width}")
        dtype = str(next(model.parameters()).dtype)
        if dtype != manifest["dtype"]:
            raise ValueError(f"loaded model dtype is {dtype}; expected {manifest['dtype']}")
        column_norms = down_column_norms(down_proj, condition_units)
        target_write_norm = target_maximum * column_norms[target_unit]
        manifest["w_down_column_norms"] = {str(key): value for key, value in column_norms.items()}
        manifest["target_1x_write_norm"] = target_write_norm
        manifest["clamp_values"] = {
            str(unit): {f"{size:g}x": {
                str(int(sign)): matched_clamp_value(target_maximum, column_norms[target_unit],
                                                    column_norms[unit], size, sign)
                for sign in SIGNS}
                for size in WRITE_SIZES}
            for unit in condition_units}
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
        full_identity = identity_keys + ("w_down_column_norms", "target_1x_write_norm", "clamp_values",
                                         "later_layers", "neuron_count")
        if old is not None and not force and not all(old.get(key) == manifest.get(key) for key in full_identity):
            raise ValueError(f"existing Part 61 manifest does not match this model layout: {manifest_path}")
        tokenized = tokenize_documents(tok, docs)
        if force:
            baseline = np.full(len(neuron_layers), np.nan, dtype=np.float32)
            conditions = np.full((len(condition_units), len(WRITE_SIZES), len(SIGNS), len(neuron_layers)),
                                 np.nan, dtype=np.float32)
            baseline_residual = np.full(DOCUMENT_COUNT, np.nan, dtype=np.float32)
            residual_ratios = np.full((len(condition_units), len(WRITE_SIZES), len(SIGNS)),
                                      np.nan, dtype=np.float32)
        else:
            baseline, conditions, baseline_residual, residual_ratios = _load_state(
                result_path, len(condition_units), len(WRITE_SIZES), len(SIGNS), len(neuron_layers))
        completed = set() if force or old is None else set(old.get("completed", []))
        manifest["completed"] = sorted(completed)
        _atomic_json(manifest, manifest_path)
        manifest_written = True
        if "baseline" not in completed or not (np.isfinite(baseline).all()
                                                and np.isfinite(baseline_residual).all()):
            baseline, _, _, stats, baseline_residual = capture_means(
                tok, model, tokenized, layers, batch_size=args.batch_size, device=args.device,
                residual_layer=layer_i + 1, start_layer=layer_i + 1)
            manifest["baseline_stats"] = stats
            completed.add("baseline")
            _persist(result_path, baseline, conditions, baseline_residual, residual_ratios,
                     neuron_layers, neuron_units, condition_units)
            manifest["completed"] = sorted(completed)
            _atomic_json(manifest, manifest_path)
        for condition_index, clamp_unit in enumerate(condition_units):
            for size_index, write_size in enumerate(WRITE_SIZES):
                step = _step_key(clamp_unit, write_size)
                if (step in completed and np.isfinite(conditions[condition_index, size_index]).all()
                        and np.isfinite(residual_ratios[condition_index, size_index]).all()):
                    continue
                sign_stats = {}
                for sign_index, sign in enumerate(SIGNS):
                    clamp_value = matched_clamp_value(
                        target_maximum, column_norms[target_unit], column_norms[clamp_unit], write_size, sign)
                    means, _, _, step_stats, residual_norms = capture_means(
                        tok, model, tokenized, layers, batch_size=args.batch_size, device=args.device,
                        residual_layer=layer_i + 1, clamp_layer=layer_i, unit=clamp_unit,
                        clamp_value=clamp_value)
                    conditions[condition_index, size_index, sign_index] = means
                    with np.errstate(divide="raise", invalid="raise"):
                        doc_ratios = residual_norms / baseline_residual
                    if not np.isfinite(doc_ratios).all():
                        raise ValueError("residual-norm ratio contains non-finite values")
                    residual_ratios[condition_index, size_index, sign_index] = np.mean(doc_ratios, dtype=np.float64)
                    sign_stats[str(int(sign))] = step_stats
                manifest.setdefault("condition_stats", {})[step] = sign_stats
                completed.add(step)
                _persist(result_path, baseline, conditions, baseline_residual, residual_ratios,
                         neuron_layers, neuron_units, condition_units)
                manifest["completed"] = sorted(completed)
                _atomic_json(manifest, manifest_path)
                print(f"61 {short} {step}: {len(docs)} documents per sign", flush=True)
        manifest["status"] = "complete"
        manifest["peak_cuda_allocated_bytes"] = max(
            [manifest.get("baseline_stats", {}).get("peak_cuda_allocated_bytes", 0)] +
            [row.get(sign, {}).get("peak_cuda_allocated_bytes", 0)
             for row in manifest.get("condition_stats", {}).values() for sign in row])
        manifest["peak_cuda_reserved_bytes"] = max(
            [manifest.get("baseline_stats", {}).get("peak_cuda_reserved_bytes", 0)] +
            [row.get(sign, {}).get("peak_cuda_reserved_bytes", 0)
             for row in manifest.get("condition_stats", {}).values() for sign in row])
        _atomic_json(manifest, manifest_path)
        print(f"61 {short}: saved means for {len(neuron_layers)} later-layer neurons to {result_path}")
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
    ap.add_argument("--model", choices=MODEL_SPECS, default="qwen35-4b")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--clamp-unit", default=None,
                    help="registered Part 61 unit as LAYER:UNIT; any other unit is rejected")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="documents per forward pass; use 1 for Qwen 4B fp32 on a 24 GiB GPU")
    ap.add_argument("--out", default=str(ROOT / "results" / "part61"))
    ap.add_argument("--source-json", default=str(SOURCE_JSON))
    ap.add_argument("--source-npz", default=str(SOURCE_NPZ))
    ap.add_argument("--qwen-maxima", default=str(QWEN_MAXIMA))
    ap.add_argument("--olmo-target-npz", default=str(OLMO_TARGET_NPZ))
    ap.add_argument("--olmo-target-json", default=str(OLMO_TARGET_JSON))
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
