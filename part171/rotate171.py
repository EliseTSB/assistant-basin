#!/usr/bin/env python3
"""Host CPU step: draw resumable Haar bases and keep only target-picked directions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def haar_basis(width, seed):
    """Return a Haar-distributed orthogonal matrix with QR signs fixed."""
    width = int(width)
    if width < 1:
        raise ValueError("width must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed[0]), int(seed[1])]))
    gaussian = rng.standard_normal((width, width), dtype=np.float32)
    q, r = np.linalg.qr(gaussian, mode="reduced")
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0).astype(np.float32)
    q *= signs[np.newaxis, :]
    return q.astype(np.float32, copy=False)


def rank_picked(q, delta, p, counts):
    """Rank rows of one shared Haar basis separately for each fit target."""
    q = np.asarray(q, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    qrows = q.T
    scores = np.abs((qrows @ delta) * (qrows @ p))
    rank = np.lexsort((np.arange(q.shape[0], dtype=np.int64), -scores))
    out = {}
    errors = {}
    for name, count in counts.items():
        count = int(count)
        directions = qrows[rank[:count]].copy()
        gram = directions @ directions.T
        errors[name] = float(np.max(np.abs(gram - np.eye(count, dtype=np.float32)))) if count else 0.0
        out[name] = directions
    return out, errors


def _torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_tensor(value):
    import torch
    return torch.from_numpy(np.asarray(value, dtype=np.float32).copy())


def _save_atomic(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _fit_marker(out_dir, targets):
    """The fit stage writes this last, after Gate 5, so its absence means do not rotate."""
    path = Path(out_dir) / "fit_complete.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the fit stage has not finished or failed a gate")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if not marker.get("gate5", {}).get("passed"):
        raise SystemExit("Gate 5 did not pass; refusing to rotate")
    hashes = {}
    for target in targets:
        data = (Path(out_dir) / f"fit_{target}.pt").read_bytes()
        hashes[str(target)] = hashlib.sha256(data).hexdigest()
        if hashes[str(target)] != marker["sha256"].get(str(target)):
            raise SystemExit(f"fit_{target}.pt does not match fit_complete.json")
    return hashes


def _fit_data(out_dir, targets):
    result = {}
    for target in targets:
        safe = str(target).replace("/", "_")
        path = Path(out_dir) / f"fit_{safe}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing fit tensor file {path}")
        result[str(target)] = _torch_load(path)
    return result


def _estimated_peak_gib(width):
    """Input, Q, and R are float64 in NumPy QR; the Gaussian input is float32."""
    return float(int(width) * int(width) * (8 * 3 + 4)) / (1024 ** 3)


def rotate(out_dir, targets=("google", "openai", "anthropic", "deepseek"),
           draw_seeds=(171, 1171, 2171), max_memory_gb=8.0):
    """Write one completed file per active (layer, draw), safe to rerun after interruption."""
    import torch
    out_dir = Path(out_dir)
    fit_sha256 = _fit_marker(out_dir, targets)
    fit = _fit_data(out_dir, targets)
    first = next(iter(fit.values()))
    width = int(first["delta"].shape[1])
    # NumPy QR holds float64 input, Q, and R, plus the float32 Gaussian input.
    estimated_peak = _estimated_peak_gib(width)
    if estimated_peak > float(max_memory_gb):
        raise MemoryError(f"one Haar QR is estimated at {estimated_peak:.2f} GiB, above --max-memory-gb={max_memory_gb}")
    active_layers = [layer for layer in range(int(first["delta"].shape[0]))
        if any(int(fit[target]["layer_counts"]["n100"][str(layer)]) > 0 for target in targets)]
    rotation_dir = out_dir / "rotations"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    completed, skipped = [], []
    for layer in active_layers:
        for seed in draw_seeds:
            path = rotation_dir / f"layer_{layer:02d}_draw_{int(seed)}.pt"
            if path.is_file():
                old = _torch_load(path)
                if (old.get("layer") == layer and old.get("draw") == int(seed) and
                        set(old.get("targets", {})) == set(targets) and old.get("fit_sha256") == fit_sha256):
                    skipped.append(str(path.name))
                    continue
            q = haar_basis(width, (int(seed), int(layer)))
            target_output = {}
            for target in targets:
                row = fit[target]
                delta = row["delta"][layer].detach().cpu().numpy()
                p = row["p"][layer].detach().cpu().numpy()
                counts = {name: int(row["layer_counts"][name][str(layer)]) for name in ("n1", "n10", "n100")}
                picked, errors = rank_picked(q, delta, p, counts)
                target_output[str(target)] = {
                    f"r{name[1:]}_d{int(seed)}": _torch_tensor(picked[name])
                    for name in ("n1", "n10", "n100")
                }
                target_output[str(target)]["orthogonality"] = errors
            _save_atomic(path, {"part": 171, "layer": int(layer), "draw": int(seed),
                "width": width, "fit_sha256": fit_sha256, "targets": target_output})
            completed.append(str(path.name))
            del q
    manifest = {"part": 171, "width": width, "active_layers": active_layers,
        "draw_seeds": [int(x) for x in draw_seeds], "completed_this_run": completed,
        "resumed_existing": skipped, "haar_qr_files": len(active_layers) * len(draw_seeds),
        "estimated_peak_gib_per_qr": estimated_peak}
    path = out_dir / "rotation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_rotations(out_dir, target, draw_seeds=(171, 1171, 2171), layers=range(21)):
    """Collect only picked vectors for one target from the per-(layer, draw) cache."""
    rotations = {}
    orthogonality = {}
    rotation_dir = Path(out_dir) / "rotations"
    expected = json.loads((Path(out_dir) / "fit_complete.json").read_text(encoding="utf-8"))["sha256"]
    for seed in draw_seeds:
        for layer in layers:
            path = rotation_dir / f"layer_{int(layer):02d}_draw_{int(seed)}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"missing Haar rotation cache {path}")
            cached = _torch_load(path)
            if cached.get("fit_sha256") != expected:
                raise ValueError(f"{path} was drawn against different fit tensors")
            target_data = cached["targets"][str(target)]
            for arm in (f"r1_d{int(seed)}", f"r10_d{int(seed)}", f"r100_d{int(seed)}"):
                rotations.setdefault(arm, {})[int(layer)] = target_data[arm].detach().cpu().numpy().astype(np.float32, copy=True)
                kname = "n" + arm.split("_d", 1)[0][1:]
                orthogonality.setdefault(arm, {})[str(layer)] = float(target_data["orthogonality"][kname])
    return rotations, orthogonality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/part171")
    parser.add_argument("--max-memory-gb", type=float, default=8.0)
    args = parser.parse_args(argv)
    result = rotate(args.out_dir, max_memory_gb=args.max_memory_gb)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
