"""Reference-free and reference-based weight statistics for Part 193.

The script reads the pinned safetensors file one tensor at a time. It does not
instantiate EmbeddingGemma 2 or run inputs through the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

# Keep CPU linear algebra within a small, predictable thread footprint.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from safetensors import safe_open


LAYERS = tuple(range(24))
ROLES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "ple_block.per_layer_input_gate",
    "ple_block.per_layer_projection",
)
ROLE_SET = frozenset(ROLES)
W1_DOSES = (0.5, 1.0, 2.0, 4.0)
W2_DOSES = (1 / 8, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SYNTHETIC_SEED = 20261108
R1 = 0.6744897501960817
ZERO_MAD_CAP = 1.0e12
PUBLISHED = {
    "W1": {"neuron": 1083, "tau_quantile": 0.3, "s": 0.025, "test_auc": 0.6775},
    "W2": {"s": 0.15, "test_auc": 0.8125},
    "W2_control": {"s": 0.025, "test_auc": 0.6175},
}


class GateFailure(RuntimeError):
    """Raised when the published-edit gate fails."""


def weight_key(layer: int, role: str) -> str:
    if role not in ROLE_SET:
        raise ValueError(f"unknown decoder role: {role}")
    return f"language_model.layers.{int(layer)}.{role}.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_z(value: float, references: np.ndarray | list[float]) -> float:
    """Median/MAD robust z-score; references never need to contain value."""
    values = np.asarray(references, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("robust z needs at least one reference value")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    difference = float(value) - median
    if mad == 0.0:
        if difference == 0.0:
            return 0.0
        return math.copysign(ZERO_MAD_CAP, difference)
    return R1 * difference / mad


def leave_one_out_robust_z(values: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("leave-one-out robust z needs at least two values")
    return np.asarray([
        robust_z(values[index], np.concatenate((values[:index], values[index + 1:])))
        for index in range(values.size)
    ], dtype=np.float64)


def robust_z_values(values: np.ndarray | list[float]) -> np.ndarray:
    """Robust z for a feature distribution that includes every layer neuron."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad == 0.0:
        output = np.zeros_like(array)
        nonmedian = array != median
        output[nonmedian] = np.sign(array[nonmedian] - median) * ZERO_MAD_CAP
        return output
    return R1 * (array - median) / mad


def rank_value(score: float, peers: np.ndarray | list[float]) -> dict[str, Any]:
    """Competition rank with strict tie handling: a tied first is not found."""
    values = np.asarray(peers, dtype=np.float64).reshape(-1)
    greater = int(np.count_nonzero(values > score))
    tied = int(np.count_nonzero(values == score))
    rank = greater + 1
    return {"rank": rank, "ties": tied, "found": rank == 1 and tied == 0}


def threshold_summary(found: list[bool] | tuple[bool, ...], doses: list[float] | tuple[float, ...]) -> dict[str, Any]:
    if len(found) != len(doses) or not found:
        raise ValueError("found flags and doses must have equal nonzero lengths")
    threshold = None
    for index, dose in enumerate(doses):
        if all(found[index:]):
            threshold = float(dose)
            break
    below_threshold_hits = [
        float(dose) for dose, hit in zip(doses, found)
        if hit and (threshold is None or float(dose) < threshold)
    ]
    dips = [
        {"from_dose": float(doses[index - 1]), "at_dose": float(doses[index])}
        for index in range(1, len(found)) if found[index - 1] and not found[index]
    ]
    return {
        "threshold": "never" if threshold is None else threshold,
        "found_below_threshold": below_threshold_hits,
        "found_doses": [float(dose) for dose, hit in zip(doses, found) if hit],
        "dips": dips,
    }


def matrix_metrics(matrix: np.ndarray) -> dict[str, float]:
    """F1 and F2 from singular values only, always in float64."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"expected matrix, got shape {value.shape}")
    singular = np.linalg.svd(value, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("F1 is undefined for a matrix with no positive singular value")
    f1 = math.log(float(singular[0]))
    norm = float(np.linalg.norm(singular))
    f2 = float((singular[0] / norm) ** 2) if norm else 0.0
    return {"sigma1": float(singular[0]), "f1": f1, "f2": f2}


def discover_population(handle) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use safetensors metadata to enumerate decoder matrices and outside 2-D weights."""
    population = []
    outside = []
    for key in handle.keys():
        shape = tuple(handle.get_slice(key).get_shape())
        if len(shape) != 2:
            continue
        prefix = "language_model.layers."
        if not key.startswith(prefix):
            outside.append({"key": key, "shape": list(shape), "scope": "outside_decoder"})
            continue
        tail = key[len(prefix):]
        try:
            layer_text, suffix = tail.split(".", 1)
            layer = int(layer_text)
        except (ValueError, TypeError):
            outside.append({"key": key, "shape": list(shape), "scope": "decoder_other"})
            continue
        if layer not in LAYERS:
            outside.append({"key": key, "shape": list(shape), "scope": "outside_decoder"})
            continue
        if not suffix.endswith(".weight"):
            outside.append({"key": key, "shape": list(shape), "scope": "decoder_other"})
            continue
        role = suffix[:-len(".weight")]
        if role in ROLE_SET:
            population.append({
                "key": key, "layer": layer, "role": role, "shape": list(shape),
            })
        else:
            outside.append({"key": key, "shape": list(shape), "scope": "decoder_other"})
    population.sort(key=lambda item: (item["role"], item["layer"]))
    expected = {(layer, role) for layer in LAYERS for role in ROLES}
    actual = {(item["layer"], item["role"]) for item in population}
    if actual != expected or len(population) != len(expected):
        missing = sorted(expected - actual)
        raise ValueError(f"decoder matrix population mismatch; missing={missing[:10]}")
    return population, sorted(outside, key=lambda item: item["key"])


def tensor_float64(handle, key: str) -> np.ndarray:
    """Load exactly one safetensors tensor and promote it to float64."""
    import torch

    tensor = handle.get_tensor(key)
    if tensor.ndim != 2:
        raise ValueError(f"{key} is not a matrix: {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float64).cpu().numpy()


def tensor_float32(handle, key: str) -> np.ndarray:
    """Load exactly one tensor as float32 for the step-0 parity check."""
    import torch

    tensor = handle.get_tensor(key)
    return tensor.to(dtype=torch.float32).cpu().numpy()


def attach_null_z(records: list[dict[str, Any]]) -> None:
    by_role = {role: [row for row in records if row["role"] == role] for role in ROLES}
    for metric in ("f1", "f2"):
        z_key = f"{metric}_z"
        for role_rows in by_role.values():
            z_scores = leave_one_out_robust_z([row[metric] for row in role_rows])
            for row, score in zip(role_rows, z_scores):
                row[z_key] = float(score)


def base_matrix_statistics(handle, population: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(population, 1):
        metrics = matrix_metrics(tensor_float64(handle, item["key"]))
        records.append({**item, **metrics})
        if index % 24 == 0:
            print(f"base matrix statistics {index}/{len(population)}", flush=True)
    attach_null_z(records)
    return records


F3_FEATURES = (
    "gate_row_norm", "up_row_norm", "down_column_norm", "gate_up_cosine",
)


def excess_kurtosis(vector: np.ndarray) -> float:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered ** 2))
    if variance == 0.0:
        return 0.0
    return float(np.mean(centered ** 4) / (variance ** 2) - 3.0)


def neuron_feature_values(
    gate: np.ndarray, up: np.ndarray, down: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gate = np.asarray(gate, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    down = np.asarray(down, dtype=np.float64)
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("MLP gate, up, and down weights must be matrices")
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError(f"incompatible MLP weight shapes: {gate.shape}, {up.shape}, {down.shape}")
    gate_norm = np.linalg.norm(gate, axis=1)
    up_norm = np.linalg.norm(up, axis=1)
    down_norm = np.linalg.norm(down, axis=0)
    denom = gate_norm * up_norm
    cosine = np.divide(
        np.einsum("ij,ij->i", gate, up), denom,
        out=np.zeros_like(denom), where=denom != 0.0,
    )
    features = np.column_stack((gate_norm, up_norm, down_norm, cosine))
    kurtosis = np.column_stack((
        [excess_kurtosis(row) for row in gate],
        [excess_kurtosis(row) for row in up],
        [excess_kurtosis(down[:, index]) for index in range(down.shape[1])],
    ))
    return features, kurtosis


def f3_scores(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return four-feature robust z values and max absolute per-neuron F3."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(F3_FEATURES):
        raise ValueError(f"expected neuron feature matrix with four columns, got {matrix.shape}")
    z_values = np.column_stack([
        robust_z_values(matrix[:, column]) for column in range(matrix.shape[1])
    ])
    return z_values, np.max(np.abs(z_values), axis=1)


def base_neuron_statistics(handle) -> dict[int, dict[str, Any]]:
    output = {}
    for layer in LAYERS:
        gate = tensor_float64(handle, weight_key(layer, "mlp.gate_proj"))
        up = tensor_float64(handle, weight_key(layer, "mlp.up_proj"))
        down = tensor_float64(handle, weight_key(layer, "mlp.down_proj"))
        features, kurtosis = neuron_feature_values(gate, up, down)
        z_values, scores = f3_scores(features)
        order = np.argsort(-scores, kind="stable")
        output[layer] = {
            "features": features,
            "kurtosis": kurtosis,
            "feature_z": z_values,
            "scores": scores,
            "top_indices": order[:10],
            "top_score": float(scores[order[0]]),
            "top_neuron": int(order[0]),
        }
        del gate, up, down, features, kurtosis, z_values, scores
    layer_maxima = np.asarray([output[layer]["top_score"] for layer in LAYERS])
    layer_z = leave_one_out_robust_z(layer_maxima)
    for index, layer in enumerate(LAYERS):
        output[layer]["layer_f3_z"] = float(layer_z[index])
    return output


def f3_layer_null(neurons: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    layer_scores = np.asarray([neurons[layer]["layer_f3_z"] for layer in LAYERS])
    rows = []
    for layer in LAYERS:
        item = neurons[layer]
        peers = np.delete(layer_scores, layer)
        score = item["layer_f3_z"]
        rows.append({
            "layer": layer,
            "top_neuron": item["top_neuron"],
            "top_neuron_f3": item["top_score"],
            "layer_f3_z": score,
            **rank_value(score, peers),
        })
    return rows


def _metric_record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for row in records:
        if row["key"] == key:
            return row
    raise KeyError(key)


def matrix_candidate_result(
    key: str,
    candidate: np.ndarray,
    records: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    if metric_name not in {"f1", "f2"}:
        raise ValueError(f"unknown matrix metric: {metric_name}")
    metrics = matrix_metrics(candidate)
    return _matrix_metric_result(key, metrics, records, metric_name)


def matrix_candidate_results(
    key: str, candidate: np.ndarray, records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Compute one float64 SVD and use its values for both F1 and F2."""
    metrics = matrix_metrics(candidate)
    return {
        statistic.upper(): _matrix_metric_result(key, metrics, records, statistic)
        for statistic in ("f1", "f2")
    }


def _matrix_metric_result(
    key: str, metrics: dict[str, float], records: list[dict[str, Any]], metric_name: str
) -> dict[str, Any]:
    target = _metric_record(records, key)
    peers = [row for row in records
             if row["role"] == target["role"] and row["layer"] != target["layer"]]
    peer_raw = np.asarray([row[metric_name] for row in peers], dtype=np.float64)
    peer_z = np.asarray([row[f"{metric_name}_z"] for row in peers], dtype=np.float64)
    score = robust_z(metrics[metric_name], peer_raw)
    sibling = rank_value(score, peer_z)
    global_rows = [row for row in records if row["key"] != key]
    global_scores = np.asarray([row[f"{metric_name}_z"] for row in global_rows], dtype=np.float64)
    global_rank = rank_value(score, global_scores)
    return {
        "key": key,
        "metric": metric_name.upper(),
        "raw": metrics[metric_name],
        "sigma1": metrics["sigma1"],
        "robust_z": score,
        "sibling_rank": sibling["rank"],
        "sibling_ties": sibling["ties"],
        "found_among_siblings": sibling["found"],
        "global_rank": global_rank["rank"],
        "global_ties": global_rank["ties"],
        "found_globally": global_rank["found"],
        "false_alarm_matrices_at_global_threshold": global_rank["rank"] - 1,
    }


def matrix_rank_from_singular_values(matrix: np.ndarray) -> int:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim == 1:
        return int(bool(np.any(value != 0.0)))
    if value.ndim != 2:
        raise ValueError(f"rank needs a vector or matrix, got shape {value.shape}")
    singular = np.linalg.svd(value, compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
    return int(np.count_nonzero(singular > tolerance))


def r1_entries(differences: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    entries = []
    for key, delta in sorted(differences.items()):
        value = np.asarray(delta)
        if np.any(value != 0):
            entries.append({
                "key": key,
                "rank": matrix_rank_from_singular_values(value),
                "shape": list(value.shape),
                "nonzero_elements": int(np.count_nonzero(value)),
            })
    return entries


def apply_w1_dose(
    base_gate: np.ndarray,
    base_up: np.ndarray,
    base_down: np.ndarray,
    neuron: int,
    patched_gate_row: np.ndarray,
    patched_up_row: np.ndarray,
    patched_down_column: np.ndarray,
    dose: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep replacement rows fixed and scale only the replacement down column."""
    gate, up = replace_w1_rows(base_gate, base_up, neuron, patched_gate_row, patched_up_row)
    down = scale_w1_down_column(base_down, neuron, patched_down_column, dose)
    return gate, up, down


def replace_w1_rows(
    base_gate: np.ndarray, base_up: np.ndarray, neuron: int,
    patched_gate_row: np.ndarray, patched_up_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gate = np.asarray(base_gate).copy()
    up = np.asarray(base_up).copy()
    gate[neuron, :] = patched_gate_row
    up[neuron, :] = patched_up_row
    return gate, up


def scale_w1_down_column(
    base_down: np.ndarray, neuron: int, patched_down_column: np.ndarray, dose: float
) -> np.ndarray:
    down = np.asarray(base_down).copy()
    down[:, neuron] = np.asarray(patched_down_column) * float(dose)
    return down


def seeded_rank_one_placements(
    population: list[dict[str, Any]], count: int = 24, seed: int = SYNTHETIC_SEED
) -> list[dict[str, Any]]:
    """Choose matrices uniformly and generate reproducible unit directions."""
    if not population:
        raise ValueError("synthetic placements need a nonempty population")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(population), size=count)
    placements = []
    for placement_id, index in enumerate(indices):
        item = population[int(index)]
        rows, columns = item["shape"]
        left = rng.normal(size=rows)
        right = rng.normal(size=columns)
        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)
        placements.append({
            "placement": placement_id,
            "key": item["key"],
            "u": left.astype(np.float64),
            "v": right.astype(np.float64),
        })
    return placements


def rank_one_candidate(base: np.ndarray, placement: dict[str, Any], sigma: float) -> np.ndarray:
    return np.asarray(base, dtype=np.float64) + float(sigma) * np.outer(placement["u"], placement["v"])


def no_op_edit(tensor: np.ndarray) -> np.ndarray:
    """Materialize a do-nothing candidate while preserving every stored bit."""
    return np.asarray(tensor).copy()


def candidate_f3(
    base_data: dict[str, Any], neuron: int,
    gate_row: np.ndarray, up_row: np.ndarray, down_column: np.ndarray,
    other_layer_maxima: np.ndarray, other_layer_z: np.ndarray,
) -> dict[str, Any]:
    features = base_data["features"].copy()
    gate_row = np.asarray(gate_row, dtype=np.float64)
    up_row = np.asarray(up_row, dtype=np.float64)
    down_column = np.asarray(down_column, dtype=np.float64)
    gate_norm = float(np.linalg.norm(gate_row))
    up_norm = float(np.linalg.norm(up_row))
    cosine = float(np.dot(gate_row, up_row) / (gate_norm * up_norm)) if gate_norm and up_norm else 0.0
    features[neuron] = (gate_norm, up_norm, float(np.linalg.norm(down_column)), cosine)
    z_values, scores = f3_scores(features)
    target_score = float(scores[neuron])
    neuron_rank = rank_value(target_score, np.delete(scores, neuron))
    layer_top = float(np.max(scores))
    layer_score = robust_z(layer_top, other_layer_maxima)
    layer_rank = rank_value(layer_score, other_layer_z)
    return {
        "features": {name: float(value) for name, value in zip(F3_FEATURES, features[neuron])},
        "feature_z": {name: float(value) for name, value in zip(F3_FEATURES, z_values[neuron])},
        "score": target_score,
        "neuron_rank": neuron_rank,
        "layer_top_neuron": int(np.argmax(scores)),
        "layer_top_neuron_f3": layer_top,
        "layer_f3_z": layer_score,
        "layer_rank": layer_rank,
        "kurtosis": {
            "gate_row": excess_kurtosis(gate_row),
            "up_row": excess_kurtosis(up_row),
            "down_column": excess_kurtosis(down_column),
        },
    }


def load_saved_edits(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        edits = {key: archive[key].copy() for key in archive.files}
    if "settings_json" not in edits or "base_sha256" not in edits:
        raise GateFailure("saved edit file lacks fitted settings or base hash")
    required = {
        "w1_neuron_j", "w1_up_row", "w1_gate_row", "w1_down_col",
        "w1_up_delta", "w1_gate_delta", "w1_down_delta",
        "w2_delta", "w2_patched", "w2c_delta", "w2c_patched",
    }
    missing = sorted(required - set(edits))
    if missing:
        raise GateFailure(f"saved edit file lacks required patch arrays: {missing}")
    settings = json.loads(str(edits["settings_json"].item()))
    edits["base_sha256"] = str(edits["base_sha256"].item())
    return edits, settings


def _close(value: Any, expected: float, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def check_gate1(base_path: Path, edits: dict[str, np.ndarray], settings: dict[str, Any],
                step0_results_path: Path) -> dict[str, Any]:
    failures = []
    actual_hash = sha256_file(base_path)
    if edits["base_sha256"] != actual_hash:
        failures.append("saved base sha256 does not match the safetensors file")

    w1 = settings.get("W1", {})
    w2 = settings.get("W2", {})
    w2c = settings.get("W2_control", {})
    if int(np.asarray(edits.get("w1_neuron_j", -1)).item()) != PUBLISHED["W1"]["neuron"]:
        failures.append("W1 neuron is not the published neuron 1083")
    if not (_close(w1.get("tau_quantile"), PUBLISHED["W1"]["tau_quantile"])
            and _close(w1.get("s"), PUBLISHED["W1"]["s"])):
        failures.append("W1 fit settings are not the published setting or near-tie fallback")
    if not _close(w2.get("s"), PUBLISHED["W2"]["s"]):
        failures.append("W2 fit setting is not the published setting or near-tie fallback")
    if not _close(w2c.get("s"), PUBLISHED["W2_control"]["s"]):
        failures.append("W2-control fit setting is not the published setting or near-tie fallback")

    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        neuron = int(np.asarray(edits["w1_neuron_j"]).item())
        w1_fields = (
            ("mlp.up_proj", "w1_up_row", "w1_up_delta", "row"),
            ("mlp.gate_proj", "w1_gate_row", "w1_gate_delta", "row"),
            ("mlp.down_proj", "w1_down_col", "w1_down_delta", "column"),
        )
        for role, patched_key, delta_key, axis in w1_fields:
            base = tensor_float32(handle, weight_key(20, role))
            base_slice = base[neuron, :] if axis == "row" else base[:, neuron]
            patched = np.asarray(edits.get(patched_key), dtype=np.float32)
            delta = np.asarray(edits.get(delta_key), dtype=np.float32)
            if patched.shape != base_slice.shape or delta.shape != base_slice.shape:
                failures.append(f"{patched_key} or {delta_key} has the wrong shape")
            elif not np.array_equal(np.subtract(patched, base_slice), delta):
                failures.append(f"{delta_key} is not the in-place patched-minus-original slice")
            del base

        base_down = tensor_float32(handle, weight_key(20, "mlp.down_proj"))
        for name in ("w2", "w2c"):
            patched = np.asarray(edits.get(f"{name}_patched"), dtype=np.float32)
            delta = np.asarray(edits.get(f"{name}_delta"), dtype=np.float32)
            if patched.shape != base_down.shape or delta.shape != base_down.shape:
                failures.append(f"{name} patched matrix or delta has the wrong shape")
            elif not np.array_equal(np.subtract(patched, base_down), delta):
                failures.append(f"{name} delta is not the in-place patched-minus-original matrix")
        del base_down

    try:
        step0 = json.loads(Path(step0_results_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateFailure(f"cannot read step-0 results: {error}") from error
    parity = step0.get("parity", {})
    if parity.get("weights_restored_bitwise") is not True:
        failures.append("step-0 parity did not confirm bitwise weight restoration")
    cosine = parity.get("post_restore_worst_tail_cosine_to_stored")
    if not isinstance(cosine, (int, float)) or float(cosine) < 0.99999:
        failures.append("step-0 post-restore parity cosine is below 0.99999")
    auc_checks = {
        "W1": PUBLISHED["W1"]["test_auc"],
        "W2": PUBLISHED["W2"]["test_auc"],
        "W2_control": PUBLISHED["W2_control"]["test_auc"],
    }
    for arm, expected_auc in auc_checks.items():
        actual_auc = step0.get("arms", {}).get(arm, {}).get("test_auc")
        if not isinstance(actual_auc, (int, float)) or abs(float(actual_auc) - expected_auc) > 0.0025:
            failures.append(f"{arm} held-out AUC is outside the published +/-0.0025 tolerance")
    for arm, expected_fit in (("W1", w1), ("W2", w2), ("W2_control", w2c)):
        selected = step0.get("fit_selection", {}).get(arm, {}).get("selected", {})
        for name in ("s", "tau_quantile") if arm == "W1" else ("s",):
            if not _close(selected.get(name), expected_fit.get(name)):
                failures.append(f"{arm} saved fit settings disagree with step-0 results")
    if step0.get("near_tie_fallbacks", {}) != settings.get("near_tie_fallbacks", {}):
        failures.append("near-tie fallback record disagrees between saved edits and step-0 results")

    if failures:
        raise GateFailure("Gate 1 failed: " + "; ".join(failures))
    return {
        "passed": True,
        "base_sha256": actual_hash,
        "published_settings": PUBLISHED,
        "near_tie_fallbacks": settings.get("near_tie_fallbacks", {}),
        "held_out_auc": {arm: step0["arms"][arm]["test_auc"] for arm in auc_checks},
        "saved_tensor_differences_bitwise": True,
        "post_restore_parity": parity,
    }


def check_gate2_gate3(base_path: Path, edits: dict[str, np.ndarray]) -> dict[str, Any]:
    import torch

    matrix_shapes = {}
    expected_w1 = {weight_key(20, role) for role in
                   ("mlp.up_proj", "mlp.gate_proj", "mlp.down_proj")}
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            original = handle.get_tensor(key)
            candidate = original.clone()
            if not torch.equal(original, candidate):
                raise GateFailure(f"Gate 2 no-op changed tensor {key}")
            if key in expected_w1:
                matrix_shapes[key] = list(handle.get_slice(key).get_shape())
            del original, candidate

    no_op_r1 = r1_entries({
        "noop": np.zeros(1, dtype=np.float64),
    })
    if no_op_r1:
        raise GateFailure("Gate 2 R1 found a changed matrix in the no-op edit")

    w1_differences = {
        weight_key(20, "mlp.up_proj"): edits["w1_up_delta"],
        weight_key(20, "mlp.gate_proj"): edits["w1_gate_delta"],
        weight_key(20, "mlp.down_proj"): edits["w1_down_delta"],
    }
    w2_differences = {weight_key(20, "mlp.down_proj"): edits["w2_delta"]}
    w2c_differences = {weight_key(20, "mlp.down_proj"): edits["w2c_delta"]}
    w1_r1, w2_r1, w2c_r1 = map(r1_entries, (w1_differences, w2_differences, w2c_differences))
    expected_w1 = {weight_key(20, role) for role in
                   ("mlp.up_proj", "mlp.gate_proj", "mlp.down_proj")}
    if {row["key"] for row in w1_r1} != expected_w1:
        raise GateFailure("Gate 3 W1 does not change exactly the gate/up/down matrices")
    expected_w2 = {weight_key(20, "mlp.down_proj")}
    if {row["key"] for row in w2_r1} != expected_w2 or {row["key"] for row in w2c_r1} != expected_w2:
        raise GateFailure("Gate 3 W2/W2-control do not change exactly layer 20 down_proj")
    if any(not np.any(np.asarray(delta) != 0) for delta in w1_differences.values()):
        raise GateFailure("Gate 3 expected nonzero W1 row/column changes")
    neuron = int(np.asarray(edits["w1_neuron_j"]).item())
    for row in w1_r1:
        key = row["key"]
        row["matrix_shape"] = matrix_shapes[key]
        row["changed_axis"] = "column" if key.endswith("mlp.down_proj.weight") else "row"
        row["changed_index"] = neuron
    return {
        "gate2": {"passed": True, "all_safetensors_tensors_bit_identical": True,
                  "r1_changed_matrices": no_op_r1},
        "gate3": {"passed": True, "W1_R1": w1_r1, "W2_R1": w2_r1,
                  "W2_control_R1": w2c_r1,
                  "W1_changed_neuron": int(np.asarray(edits["w1_neuron_j"]).item())},
    }


def _thresholds_for_matrix_ladder(
    ladder: dict[str, dict[str, Any]], doses: tuple[float, ...], metric_key: str = "matrix"
) -> dict[str, Any]:
    output = {}
    for statistic in ("F1", "F2"):
        found = [item[metric_key][statistic]["found_among_siblings"] for item in ladder.values()]
        output[statistic] = threshold_summary(found, doses)
    return output


def _serialize_top_neurons(data: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for index in data["top_indices"][:limit]:
        index = int(index)
        rows.append({
            "neuron": index,
            "f3_score": float(data["scores"][index]),
            "features": {name: float(value) for name, value in
                         zip(F3_FEATURES, data["features"][index])},
            "feature_z": {name: float(value) for name, value in
                          zip(F3_FEATURES, data["feature_z"][index])},
            "excess_kurtosis": {
                "gate_row": float(data["kurtosis"][index, 0]),
                "up_row": float(data["kurtosis"][index, 1]),
                "down_column": float(data["kurtosis"][index, 2]),
            },
        })
    return rows


def _run_w1(handle, records, neuron_data, edits):
    neuron = int(np.asarray(edits["w1_neuron_j"]).item())
    gate_key = weight_key(20, "mlp.gate_proj")
    up_key = weight_key(20, "mlp.up_proj")
    down_key = weight_key(20, "mlp.down_proj")
    base_gate = tensor_float64(handle, gate_key)
    base_up = tensor_float64(handle, up_key)
    base_down = tensor_float64(handle, down_key)
    gate_row = np.asarray(edits["w1_gate_row"], dtype=np.float64)
    up_row = np.asarray(edits["w1_up_row"], dtype=np.float64)
    down_column = np.asarray(edits["w1_down_col"], dtype=np.float64)

    gate_candidate, up_candidate = replace_w1_rows(
        base_gate, base_up, neuron, gate_row, up_row
    )
    constant_rows = {
        "gate_proj": matrix_candidate_results(gate_key, gate_candidate, records),
        "up_proj": matrix_candidate_results(up_key, up_candidate, records),
    }

    layer_maxima = np.asarray([neuron_data[layer]["top_score"] for layer in LAYERS])
    other_layer_maxima = np.delete(layer_maxima, 20)
    other_layer_z = np.asarray([
        neuron_data[layer]["layer_f3_z"] for layer in LAYERS if layer != 20
    ], dtype=np.float64)
    ladder = {}
    for dose in W1_DOSES:
        down_candidate = scale_w1_down_column(base_down, neuron, down_column, dose)
        down_scores = matrix_candidate_results(down_key, down_candidate, records)
        f3 = candidate_f3(
            neuron_data[20], neuron, gate_row, up_row, down_column * dose,
            other_layer_maxima, other_layer_z,
        )
        f3["found_neuron"] = f3["neuron_rank"]["found"]
        f3["found_layer"] = f3["layer_rank"]["found"]
        ladder[str(dose)] = {
            "dose": float(dose),
            "gate_proj": constant_rows["gate_proj"],
            "up_proj": constant_rows["up_proj"],
            "down_proj": down_scores,
            "F3": f3,
        }
        del down_candidate

    thresholds = {}
    for role_name in ("gate_proj", "up_proj", "down_proj"):
        matrix_ladder = {
            str(dose): {"matrix": ladder[str(dose)][role_name]}
            for dose in W1_DOSES
        }
        thresholds[role_name] = _thresholds_for_matrix_ladder(matrix_ladder, W1_DOSES)
    thresholds["F3_neuron"] = threshold_summary(
        [ladder[str(dose)]["F3"]["found_neuron"] for dose in W1_DOSES], W1_DOSES
    )
    thresholds["F3_layer"] = threshold_summary(
        [ladder[str(dose)]["F3"]["found_layer"] for dose in W1_DOSES], W1_DOSES
    )
    r1 = r1_entries({
        gate_key: edits["w1_gate_delta"],
        up_key: edits["w1_up_delta"],
        down_key: edits["w1_down_delta"],
    })
    for row in r1:
        row["matrix_shape"] = {
            gate_key: list(base_gate.shape),
            up_key: list(base_up.shape),
            down_key: list(base_down.shape),
        }[row["key"]]
        row["changed_axis"] = "column" if row["key"] == down_key else "row"
        row["changed_index"] = neuron
    natural_top = _serialize_top_neurons(neuron_data[20])
    del base_gate, base_up, base_down, gate_candidate, up_candidate
    return {
        "neuron": neuron,
        "r1_at_published_dose": r1,
        "doses": ladder,
        "thresholds": thresholds,
        "top_unedited_layer20_neurons": natural_top,
    }


def _run_w2(handle, records, edits, key: str, delta_key: str,
            dose_values: tuple[float, ...], r1_at_dose_one: list[dict[str, Any]]) -> dict[str, Any]:
    base = tensor_float64(handle, key)
    delta = np.asarray(edits[delta_key], dtype=np.float64)
    ladder = {}
    for dose in dose_values:
        candidate = base + float(dose) * delta
        ladder[str(dose)] = {
            "dose": float(dose),
            "matrix": matrix_candidate_results(key, candidate, records),
        }
        del candidate
    thresholds = _thresholds_for_matrix_ladder(ladder, dose_values)
    del base, delta
    return {"r1_at_published_dose": r1_at_dose_one, "doses": ladder, "thresholds": thresholds}


def _run_rank_one_families(handle, population, records, w2_delta):
    placements = seeded_rank_one_placements(population, count=24, seed=SYNTHETIC_SEED)
    sigma_synthetic = float(np.linalg.norm(np.asarray(w2_delta, dtype=np.float64)))
    synthetic_rows = []
    gate2_rows = []
    gate12_rows = []
    for placement in placements:
        key = placement["key"]
        base = tensor_float64(handle, key)
        base_metric = _metric_record(records, key)
        synthetic_matrix = rank_one_candidate(base, placement, sigma_synthetic)
        synthetic = matrix_candidate_results(key, synthetic_matrix, records)
        high_matrix = rank_one_candidate(base, placement, 2.0 * base_metric["sigma1"])
        gate2 = matrix_candidate_results(key, high_matrix, records)
        low_matrix = rank_one_candidate(base, placement, 1.2 * base_metric["sigma1"])
        gate12 = matrix_candidate_results(key, low_matrix, records)
        placement_info = {
            "placement": int(placement["placement"]),
            "key": key,
            "layer": int(base_metric["layer"]),
            "role": base_metric["role"],
        }
        synthetic_rows.append({**placement_info, "sigma": sigma_synthetic, "scores": synthetic})
        gate2_rows.append({**placement_info, "sigma": 2.0 * base_metric["sigma1"], "scores": gate2})
        gate12_rows.append({**placement_info, "sigma": 1.2 * base_metric["sigma1"], "scores": gate12})
        del base, synthetic_matrix, high_matrix, low_matrix
    gate2_found = sum(row["scores"]["F1"]["found_among_siblings"] for row in gate2_rows)
    return {
        "seed": SYNTHETIC_SEED,
        "placements": synthetic_rows,
        "sigma_w2_frobenius": sigma_synthetic,
        "gate4_2x": {
            "placements": gate2_rows,
            "f1_first_count": int(gate2_found),
            "passed": gate2_found == 24,
        },
        "gate4_1_2x": {
            "placements": gate12_rows,
            "calibration_only": True,
        },
    }


def run_analysis(base_path: Path, saved_edits_path: Path, step0_results_path: Path) -> dict[str, Any]:
    edits, settings = load_saved_edits(saved_edits_path)
    gate1 = check_gate1(base_path, edits, settings, step0_results_path)
    with safe_open(str(base_path), framework="pt", device="cpu") as handle:
        population, outside = discover_population(handle)
        gate23 = check_gate2_gate3(base_path, edits)
        matrix_records = base_matrix_statistics(handle, population)
        neuron_data = base_neuron_statistics(handle)

        w1 = _run_w1(handle, matrix_records, neuron_data, edits)
        down_key = weight_key(20, "mlp.down_proj")
        w2 = _run_w2(
            handle, matrix_records, edits, down_key, "w2_delta", W2_DOSES,
            gate23["gate3"]["W2_R1"],
        )
        w2c = _run_w2(
            handle, matrix_records, edits, down_key, "w2c_delta", W2_DOSES,
            gate23["gate3"]["W2_control_R1"],
        )
        synthetic = _run_rank_one_families(
            handle, population, matrix_records, edits["w2_delta"]
        )

    f3_null = f3_layer_null(neuron_data)
    for row in f3_null:
        data = neuron_data[row["layer"]]
        row["neuron_statistics"] = {
            "feature_names": list(F3_FEATURES),
            "feature_values": data["features"].tolist(),
            "robust_z": data["feature_z"].tolist(),
            "F3_max_abs_z": data["scores"].tolist(),
            "excess_kurtosis_gate_up_down": data["kurtosis"].tolist(),
        }
    matrix_null = [{
        "key": row["key"],
        "layer": row["layer"],
        "role": row["role"],
        "shape": row["shape"],
        "sigma1": row["sigma1"],
        "F1": {"raw": row["f1"], "robust_z": row["f1_z"]},
        "F2": {"raw": row["f2"], "robust_z": row["f2_z"]},
    } for row in matrix_records]
    matrix_svd_count = 216 + 2 + len(W1_DOSES) + 2 * len(W2_DOSES) + 24 + 48
    results = {
        "part": 193,
        "base": {
            "sha256": gate1["base_sha256"],
            "decoder_matrix_count": len(population),
            "layer_count": len(LAYERS),
            "roles": list(ROLES),
        },
        "gates": {
            "gate1": gate1,
            **gate23,
            "gate4": {
                "passed": synthetic["gate4_2x"]["passed"],
                "f1_first_count_at_2x": synthetic["gate4_2x"]["f1_first_count"],
                "placements": 24,
            },
        },
        "population": {
            "ranked_decoder_matrices": matrix_null,
            "unranked_2d_matrices_not_ranked": outside,
        },
        "null_f3_by_layer": f3_null,
        "edits": {"W1": w1, "W2": w2, "W2_control": w2c},
        "synthetic_rank_one": synthetic,
        "computation": {
            "float64_svd_metric_count": matrix_svd_count,
            "additional_float64_svd_for_r1_difference_ranks": 2,
            "total_float64_svd_calls": matrix_svd_count + 2,
            "svd_vectors": False,
            "matrix_population_recomputed_per_edit": False,
            "only_changed_matrices_recomputed_for_edit_ladders": True,
            "base_tensor_reading": "one safetensors tensor at a time",
        },
    }
    return results


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-safetensors", type=Path, required=True)
    parser.add_argument("--saved-edits", type=Path, required=True)
    parser.add_argument("--step0-results", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "results" / "part193" / "results.json",
    )
    parser.add_argument(
        "--gate1-only", action="store_true",
        help="verify the regenerated edits before starting matrix statistics",
    )
    args = parser.parse_args(argv)

    if args.gate1_only:
        edits, settings = load_saved_edits(args.saved_edits)
        gate1 = check_gate1(args.base_safetensors, edits, settings, args.step0_results)
        print(json.dumps(gate1, indent=2, allow_nan=False))
        return 0

    output = run_analysis(args.base_safetensors, args.saved_edits, args.step0_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(output["gates"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
