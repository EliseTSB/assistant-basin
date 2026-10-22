#!/usr/bin/env python3
"""Classify Part 59B symmetric/odd responders and add their logit-lens top tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import neuron_report as nr  # noqa: E402
import part26 as p26  # noqa: E402


def _as_matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape [controls, neurons]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _symmetric_mask(minus, plus, control_minus, control_plus, percentile: float = 99.0) -> tuple[np.ndarray, float, float]:
    minus = np.asarray(minus, dtype=np.float64)
    plus = np.asarray(plus, dtype=np.float64)
    control_minus = _as_matrix(control_minus, "control_minus")
    control_plus = _as_matrix(control_plus, "control_plus")
    if minus.ndim != 1 or plus.shape != minus.shape:
        raise ValueError("target dose changes must be matching one-dimensional arrays")
    if control_minus.shape != control_plus.shape or control_minus.shape[1] != len(minus):
        raise ValueError("random-control changes must match the target neuron count at both doses")
    if not (0 < percentile < 100):
        raise ValueError("percentile must be between 0 and 100")
    if not np.isfinite(minus).all() or not np.isfinite(plus).all():
        raise ValueError("target changes contain non-finite values")
    # One bar per dose, pooled over every later-layer neuron and every reference
    # control.  For pseudo-control counts, callers pass only the other controls.
    threshold_minus = float(np.percentile(np.abs(control_minus).ravel(), percentile))
    threshold_plus = float(np.percentile(np.abs(control_plus).ravel(), percentile))
    same_nonzero_sign = (np.sign(minus) == np.sign(plus)) & (np.sign(minus) != 0)
    symmetric = (same_nonzero_sign
                 & (np.abs(minus) > threshold_minus)
                 & (np.abs(plus) > threshold_plus))
    return symmetric, threshold_minus, threshold_plus


def classify_changes(changes, percentile: float = 99.0) -> dict:
    """Classify target and controls from changes shaped [target+controls, -1/+1, neuron]."""
    changes = np.asarray(changes, dtype=np.float64)
    if changes.ndim != 3 or changes.shape[0] < 3 or changes.shape[1] != 2:
        raise ValueError("changes must have shape [target + >=2 controls, 2 doses, neurons]")
    if not np.isfinite(changes).all():
        raise ValueError("changes contain non-finite values")
    target_minus, target_plus = changes[0, 0], changes[0, 1]
    control_minus, control_plus = changes[1:, 0], changes[1:, 1]
    symmetric, threshold_minus, threshold_plus = _symmetric_mask(
        target_minus, target_plus, control_minus, control_plus, percentile)
    odd = (np.sign(target_minus) * np.sign(target_plus)) < 0
    control_counts = []
    for index in range(len(control_minus)):
        reference_indices = [other for other in range(len(control_minus)) if other != index]
        pseudo_symmetric, _, _ = _symmetric_mask(
            control_minus[index], control_plus[index], control_minus[reference_indices],
            control_plus[reference_indices], percentile)
        control_counts.append(int(pseudo_symmetric.sum()))
    return {
        "symmetric_mask": symmetric,
        "odd_mask": odd,
        "threshold_minus": threshold_minus,
        "threshold_plus": threshold_plus,
        "symmetric_count": int(symmetric.sum()),
        "odd_count": int(odd.sum()),
        "control_symmetric_counts": control_counts,
        "max_control_symmetric_count": max(control_counts, default=0),
        "percentile": float(percentile),
    }


def analyze_arrays(changes, neuron_layers, neuron_units, percentile: float = 99.0) -> dict:
    result = classify_changes(changes, percentile)
    layers = np.asarray(neuron_layers, dtype=np.int64)
    units = np.asarray(neuron_units, dtype=np.int64)
    if len(layers) != len(result["symmetric_mask"]) or len(units) != len(layers):
        raise ValueError("neuron metadata length does not match activation changes")
    mask = result["symmetric_mask"]
    indices = np.flatnonzero(mask)
    changes = np.asarray(changes, dtype=np.float64)
    ranking = sorted(indices.tolist(),
                     key=lambda index: min(abs(changes[0, 0, index]), abs(changes[0, 1, index])),
                     reverse=True)
    result["symmetric_neurons"] = [
        {"layer": int(layers[index]), "unit": int(units[index]),
         "change_minus_1": float(changes[0, 0, index]),
         "change_plus_1": float(changes[0, 1, index]),
         "minimum_absolute_change": float(min(abs(changes[0, 0, index]), abs(changes[0, 1, index]))),
         "control_p99_minus_1": float(result["threshold_minus"]),
         "control_p99_plus_1": float(result["threshold_plus"])}
        for index in ranking
    ]
    result["symmetric_mask"] = mask
    result["top_symmetric"] = result["symmetric_neurons"][:10]
    return result


def _attach_logit_lens_and_windows(rows: list[dict], model_id: str, revision: str,
                                   short_name: str, device: str) -> None:
    if not rows:
        return
    short = next((name for name, spec in p26.MODELS.items()
                  if spec["id"] == model_id), None)
    if short is None:
        raise ValueError(f"cannot find Part 26 model spec for {model_id}")
    spec = p26.MODELS[short]
    tok, model = p26._load_model_offline(model_id, revision, device, spec.get("dtype", "bf16"))
    try:
        _, layers, norm = nr.find_text_layers(model)
        by_layer: dict[int, list[dict]] = {}
        for row in rows:
            by_layer.setdefault(int(row["layer"]), []).append(row)
        for layer_i, group in by_layer.items():
            units = [int(row["unit"]) for row in group]
            token_ids = p26.logit_lens_top_ids(tok, model, norm, layers[layer_i].mlp.down_proj,
                                               units, top=10, unit_chunk=10, vocab_chunk=8192,
                                               device=device, max_gpu_gb=22.0)
            for row, ids in zip(group, token_ids):
                row["logit_lens_top_tokens"] = [
                    {"token_id": int(token_id),
                     "token": p26.normalize_write_token(tok.decode([int(token_id)]))}
                    for token_id in ids
                ]
        import run59b
        docs = run59b.source_documents()
        tokenized = run59b.tokenize_documents(tok, docs)
        windows = run59b.capture_top_windows(
            tok, model, docs, tokenized, layers,
            {layer: [int(row["unit"]) for row in group]
             for layer, group in by_layer.items()},
            batch_size=1 if short_name == "qwen35-4b" else 4,
            device=device, top_n=3)
        for row in rows:
            row["top_windows"] = windows.get(f"{row['layer']}:{row['unit']}", [])
    finally:
        del model, tok
        import gc
        gc.collect()


def analyze_file(input_npz: str | Path, out_path: str | Path | None = None, *,
                 percentile: float = 99.0, include_lens: bool = True,
                 device: str = "cuda") -> dict:
    input_npz = Path(input_npz)
    manifest_path = input_npz.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"59B manifest status is {manifest.get('status')!r}; expected 'complete'")
    with np.load(input_npz) as archive:
        changes = np.asarray(archive["change"], dtype=np.float32)
        neuron_layers = np.asarray(archive["neuron_layer"])
        neuron_units = np.asarray(archive["neuron_unit"])
        doses = np.asarray(archive["doses"], dtype=np.float32)
    if doses.tolist() != [-1.0, 1.0]:
        raise ValueError(f"expected stored doses [-1, 1], got {doses.tolist()}")
    result = analyze_arrays(changes, neuron_layers, neuron_units, percentile)
    top = result["top_symmetric"]
    prediction_59f = (result["symmetric_count"] >= 20
                      if manifest["short_name"] == "qwen35-4b" else None)
    prediction_59g = result["max_control_symmetric_count"] < 5
    target_exceeds_max_control = (
        result["symmetric_count"] > result["max_control_symmetric_count"])
    null_caveat_59f = "The registered null alone can pass or fail 59f."
    null_caveat_59g = "The registered null alone can pass or fail 59g."
    control_units = manifest.get("control_units", [])
    pile_maxima = manifest.get("pile_maxima", {})
    control_maxima = [float(pile_maxima[str(unit)]) for unit in control_units
                      if str(unit) in pile_maxima]
    if control_maxima and min(control_maxima) > 0:
        control_scale_context = (
            f"Recorded control Pile maxima span {max(control_maxima) / min(control_maxima):.1f}x; "
            "the registered pooled threshold uses every control.")
    else:
        control_scale_context = (
            "Control Pile maxima are not available in the manifest; the registered pooled threshold uses every control.")
    output = {
        "tool": "analyze59b.py",
        "model": manifest["model"],
        "revision": manifest["revision"],
        "short_name": manifest["short_name"],
        "twin": manifest["twin"],
        "archive": str(input_npz.resolve()),
        "classification": "SYMMETRIC requires equal nonzero signs and absolute change > the pooled 99th percentile of absolute changes over all control later-layer neurons at each dose; for each control count, its threshold is pooled over the other controls. ODD requires opposite signs",
        "control_percentile": percentile,
        "later_neuron_count": int(changes.shape[-1]),
        "symmetric_count": result["symmetric_count"],
        "odd_count": result["odd_count"],
        "random_control_symmetric_counts": result["control_symmetric_counts"],
        "max_random_control_symmetric_count": result["max_control_symmetric_count"],
        "predictions": {
            "59f": {"claim": "Qwen has at least 20 later SYMMETRIC responders",
                    "applicable": manifest["short_name"] == "qwen35-4b",
                    "observed": result["symmetric_count"] if prediction_59f is not None else None,
                    "held": prediction_59f,
                    "target_exceeds_max_control": target_exceeds_max_control,
                    "null_caveat": null_caveat_59f,
                    "control_scale_context": control_scale_context},
            "59g": {"claim": "fewer than 5 SYMMETRIC responders in each random control",
                    "observed_per_control": result["control_symmetric_counts"],
                    "maximum": result["max_control_symmetric_count"], "held": prediction_59g,
                    "target_exceeds_max_control": target_exceeds_max_control,
                    "null_caveat": null_caveat_59g,
                    "control_scale_context": control_scale_context},
            "59h": {"claim": "no more than 3 of the top 10 are sci-fi words by hand reading",
                    "hand_reading_required": True, "maximum_sci_fi_tokens": 3},
        },
        "top_symmetric_responders": top,
        "thresholds": {"method": "numpy percentile with linear interpolation over pooled control-by-neuron absolute changes",
                       "per_dose": True, "target_reference_controls": "all controls",
                       "control_reference_controls": "leave one control out"},
    }
    if out_path is None:
        out_path = input_npz.with_name(input_npz.stem + "-analysis.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save classification counts and responder indices before optional model
    # reload/lens work, so a reload failure cannot erase the primary result.
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if include_lens:
        _attach_logit_lens_and_windows(top, manifest["model"], manifest["revision"],
                                       manifest["short_name"], device)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"59B {manifest['short_name']}: symmetric={result['symmetric_count']} "
          f"odd={result['odd_count']} random-control-symmetric={result['control_symmetric_counts']} "
          f"target_exceeds_max_control={target_exceeds_max_control}")
    print(f"59f null caveat: {null_caveat_59f}")
    print(f"59g null caveat: {null_caveat_59g}")
    print(f"59f/59g control-scale context: {control_scale_context}")
    return output


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", help="59B .npz activation archive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-lens", action="store_true", help="skip loading the model and adding top-token projections")
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    analyze_file(args.archive, args.out, percentile=args.percentile,
                 include_lens=not args.no_lens, device=args.device)
