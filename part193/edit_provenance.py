"""Shared provenance schema for Part 193's captured edit arrays."""

from __future__ import annotations

import hashlib

import numpy as np


ARRAY_KEYS_BY_ARM = {
    "W1": (
        "w1_neuron_j", "w1_up_row", "w1_gate_row", "w1_down_col",
        "w1_up_delta", "w1_gate_delta", "w1_down_delta",
        "w1_up_full_delta", "w1_gate_full_delta", "w1_down_full_delta",
    ),
    "W2": ("w2_delta", "w2_patched"),
    "W2_control": ("w2c_delta", "w2c_patched"),
}

DELTA_KEYS_BY_ARM = {
    "W1": (
        "w1_up_delta", "w1_gate_delta", "w1_down_delta",
        "w1_up_full_delta", "w1_gate_full_delta", "w1_down_full_delta",
    ),
    "W2": ("w2_delta",),
    "W2_control": ("w2c_delta",),
}


def array_sha256(value: np.ndarray) -> str:
    """Hash array bytes together with dtype and shape, preserving saved-array identity."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def saved_edit_provenance(saved_edits: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    """Describe every persisted edit array, including a norm for each delta."""
    output = {}
    for arm, keys in ARRAY_KEYS_BY_ARM.items():
        missing = sorted(set(keys) - set(saved_edits))
        if missing:
            raise KeyError(f"{arm} saved edit arrays are missing: {missing}")
        arrays = {}
        for key in keys:
            value = np.asarray(saved_edits[key])
            arrays[key] = {
                "sha256": array_sha256(value),
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
        deltas = {}
        for key in DELTA_KEYS_BY_ARM[arm]:
            value = np.asarray(saved_edits[key])
            deltas[key] = {
                "frobenius_norm": float(np.linalg.norm(value.astype(np.float64))),
                "sha256": array_sha256(value),
            }
        output[arm] = {"arrays": arrays, "deltas": deltas}
    metadata_keys = ("settings_json", "base_sha256")
    if set(metadata_keys).issubset(saved_edits):
        output["archive_metadata"] = {
            key: {
                "sha256": array_sha256(np.asarray(saved_edits[key])),
                "shape": list(np.asarray(saved_edits[key]).shape),
                "dtype": np.asarray(saved_edits[key]).dtype.str,
            }
            for key in metadata_keys
        }
    return output
